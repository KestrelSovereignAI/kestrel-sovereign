"""Canonical host orchestration of the SDK's visual computer contracts.

Providers acquire; drivers interact. The runtime never substitutes host shell
execution. Accepted intents precede effects; unknown outcomes retain original
IDs and require explicit reconciliation. Safety input fencing is independent
of normal action capacity and precedes the driver's priority control call.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
import math
from uuid import uuid4

from kestrel_sdk.computer_use import (
    ActionReceipt, ActionRequest, ComputerSession, ControlRequest,
    EnvironmentRequest, ScreenObservation, SessionRequest,
)
from kestrel_sdk.resources import ResourceConflict, ResourceError, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope
from kestrel_sdk.operator import OperatorAuthorizationError

from .repository import CustodyRecord, VisualCustodyRepository


class VisualAccessDenied(ResourceError, OperatorAuthorizationError):
    """Visual operation has no current original-owner authority."""


class VisualAction(StrEnum):
    ADMIT = "visual.admit"
    STATUS = "visual.status"
    OBSERVE = "visual.observe"
    ACT = "visual.act"
    STOP = "visual.stop"
    TAKEOVER = "visual.takeover"
    HANDBACK = "visual.handback"
    RELEASE = "visual.release"
    TRANSFER = "visual.transfer"


@dataclass(frozen=True)
class VisualPolicy:
    """Explicit operator configuration; shell consent never populates this."""

    permitted_locations: frozenset[str] = frozenset({"local"})
    max_lifetime_seconds: int = 1800
    max_hourly_cost_usd: Decimal = Decimal("0")
    max_total_cost_usd: Decimal = Decimal("0")
    operation_seconds: float = 30
    max_inflight_actions: int = 8

    def __post_init__(self):
        if (not self.permitted_locations or not self.permitted_locations <= {"local", "remote"}
                or type(self.max_lifetime_seconds) is not int or self.max_lifetime_seconds <= 0
                or isinstance(self.operation_seconds, bool) or not math.isfinite(self.operation_seconds) or self.operation_seconds <= 0
                or type(self.max_inflight_actions) is not int or self.max_inflight_actions <= 0
                or not self.max_hourly_cost_usd.is_finite() or self.max_hourly_cost_usd < 0
                or not self.max_total_cost_usd.is_finite() or self.max_total_cost_usd < 0):
            raise ValueError("invalid visual computer policy bounds")


class VisualComputerRuntime:
    def __init__(self, repository: VisualCustodyRepository, authority, *, providers, drivers,
                 policy: VisualPolicy | None = None, check_destination=None):
        self.repository, self.authority = repository, authority
        self.providers, self.drivers = dict(providers), dict(drivers)
        self.policy = policy or VisualPolicy()
        self.check_destination = check_destination
        self._locks = {}
        self._fences = set()
        self._verified_sessions = set()
        self._inflight = 0

    @staticmethod
    def scope(context):
        return ResourceScope(tenant_id=context.tenant_id, principal_id=context.principal_id,
                             agent_id=context.acting_agent_id)

    @staticmethod
    def _key(scope, session_id):
        return (scope.tenant_id, scope.principal_id, scope.agent_id, session_id)

    def _lock(self, session_id):
        return self._locks.setdefault(session_id, asyncio.Lock())

    def _now(self):
        return self.authority.clock()

    @asynccontextmanager
    async def operation(self, context, action):
        async with self.authority.operation(context, action, self.repository.storage_id) as check:
            await self.repository.ensure_initialized()
            check()
            yield check
            try:
                check()
            except OperatorAuthorizationError:
                if action in {VisualAction.ADMIT, VisualAction.ACT, VisualAction.STOP, VisualAction.TAKEOVER, VisualAction.HANDBACK, VisualAction.RELEASE}:
                    raise ResourceOutcomeUnknown("visual operation completed without current delivery authority; reconcile original IDs") from None
                raise

    async def _call(self, context, check, function, *args):
        check()
        remaining = min(self.policy.operation_seconds, (context.expires_at - self._now()).total_seconds())
        if remaining <= 0:
            raise OperatorAuthorizationError("visual authority expired")
        async with asyncio.timeout(remaining):
            result = await function(*args)
        check()
        return result

    def _provider(self, record):
        try:
            return self.providers[record.request.provider_id]
        except KeyError:
            raise ResourceError("visual environment provider is unavailable") from None

    def _driver(self, record):
        driver_id = record.connection.driver_id if record.connection else record.session.driver_id
        matches = [driver for driver in self.drivers.values() if driver.driver_id == driver_id]
        if len(matches) != 1:
            raise ResourceError("visual computer driver is unavailable or ambiguous")
        return matches[0]

    def _destination(self, location):
        if location not in self.policy.permitted_locations:
            raise OperatorAuthorizationError("visual destination is not granted")
        if self.check_destination is not None:
            self.check_destination(location)
        elif location != "local":
            raise OperatorAuthorizationError("remote visual destination policy is unavailable")

    def _request(self, context, request):
        request = EnvironmentRequest.model_validate(request)
        now = self._now()
        self._destination(request.location)
        if request.scope != self.scope(context):
            raise OperatorAuthorizationError("visual request owner does not match")
        if (request.provider_id not in self.providers
                or request.location not in self.policy.permitted_locations
                or request.requested_at > now or request.expires_at <= now
                or (request.expires_at - request.requested_at).total_seconds() > self.policy.max_lifetime_seconds
                or request.max_hourly_cost_usd > self.policy.max_hourly_cost_usd
                or request.max_total_cost_usd > self.policy.max_total_cost_usd):
            raise OperatorAuthorizationError("visual request exceeds explicit host grant")
        return request

    def _quote(self, record, quote):
        request = record.request
        if (quote.request_id != request.request_id or quote.request_sha256 != request.digest()
                or quote.scope != request.scope or quote.provider_id != request.provider_id
                or quote.location != request.location or quote.region != request.region
                or quote.expires_at <= self._now()
                or quote.hourly_cost_usd > request.max_hourly_cost_usd
                or quote.estimated_total_cost_usd > request.max_total_cost_usd
                or request.surface not in quote.capabilities.surfaces
                or not request.required_capabilities <= quote.capabilities.operations):
            raise ResourceConflict("environment quote does not match admitted constraints")

    def _lease(self, record, lease, *, releasing=False, acquiring=False):
        request, quote = record.request, record.quote
        if (lease.request_id != request.request_id or lease.quote_id != quote.quote_id
                or lease.scope != request.scope or lease.provider_id != request.provider_id
                or lease.location != request.location or lease.expires_at > request.expires_at
                or (record.lease is not None and lease.lease_id != record.lease.lease_id)
                or (not releasing and (lease.state not in ({"pending", "ready"} if acquiring else {"ready"}) or lease.expires_at <= self._now()))):
            raise ResourceConflict("environment lease does not match admitted custody")

    def _session(self, record, session, *, opening=False):
        if (session.session_id != record.session_id or session.scope != record.request.scope
                or session.lease_id != record.lease.lease_id
                or session.driver_id != record.connection.driver_id
                or session.surface != record.request.surface
                or not record.request.required_capabilities <= session.capabilities
                or session.expires_at > min(record.lease.expires_at, record.connection.expires_at)
                or (opening and (session.state != "ready" or session.control_epoch != 0))):
            raise ResourceConflict("computer session does not match admitted custody")

    async def admit(self, context, request):
        async with self.operation(context, VisualAction.ADMIT) as check:
            request = self._request(context, request)
            # Serialize admission by request ID, not a caller-selected resource name.
            async with self._lock((self.scope(context).model_dump_json(), request.request_id)):
                record = await self.repository.admit(self.scope(context), CustodyRecord(request=request, session_id=uuid4().hex, last_activity_at=self._now()), check)
                if record.phase == "active":
                    check()
                    return record.session
                if record.phase not in {"admitted", "quoted"}:
                    raise ResourceOutcomeUnknown("existing request requires original-ID reconciliation")
                provider = self._provider(record)
                if record.quote is None:
                    capabilities = await self._call(context, check, provider.capabilities)
                    if (capabilities.provider_id != request.provider_id or capabilities.location != request.location
                            or request.surface not in capabilities.surfaces
                            or not request.required_capabilities <= capabilities.operations):
                        raise ResourceConflict("provider capabilities do not satisfy visual request")
                    quote = await self._call(context, check, provider.quote, request)
                    self._quote(record, quote)
                    record.quote, record.phase = quote, "quoted"
                    await self.repository.save(self.scope(context), record, check)
                self._quote(record, record.quote)
                record.phase = "acquiring"
                await self.repository.save(self.scope(context), record, check)
                return await self._acquire_open(context, record, check)

    async def _acquire_open(self, context, record, check):
        scope, provider = self.scope(context), self._provider(record)
        # Re-enter only through explicit request reconciliation. acquire is
        # idempotent for this exact admitted request/accepted quote.
        if record.lease is None:
            lease = await self._call(context, check, provider.acquire, record.request, record.quote)
            self._lease(record, lease, acquiring=True)
            record.lease, record.phase = lease, "leased"
            await self.repository.save(scope, record, check)
        lease = record.lease
        connection = record.connection or await self._call(context, check, provider.connect, scope, lease)
        if (connection.lease_id != lease.lease_id or connection.expires_at > lease.expires_at
                or connection.expires_at <= self._now()):
            raise ResourceConflict("driver connection does not match admitted lease")
        record.connection, record.phase = connection, "opening"
        await self.repository.save(scope, record, check)
        driver = self._driver(record)
        session_request = SessionRequest(session_id=record.session_id, scope=scope,
            lease_id=lease.lease_id, surface=record.request.surface,
            required_capabilities=record.request.required_capabilities,
            expires_at=min(lease.expires_at, connection.expires_at))
        session = await self._call(context, check, driver.open, session_request, connection)
        self._session(record, session, opening=True)
        if record.session is not None and session != record.session:
            raise ResourceOutcomeUnknown("restored worker differs from accepted session custody")
        record.session = session
        await self.repository.save(scope, record, check)
        ready = await self._call(context, check, provider.inspect, scope, record.lease)
        self._lease(record, ready)
        record.lease, record.phase = ready, "active"
        await self.repository.save(scope, record, check)
        self._verified_sessions.add(self._key(scope, record.session_id))
        return session

    async def reconcile_request(self, context, request_id):
        async with self.operation(context, VisualAction.ADMIT) as check:
            scope = self.scope(context)
            async with self._lock((scope.model_dump_json(), request_id)):
                record = await self.repository.get(scope, request_id=request_id, guard=check)
                self._request(context, record.request)
                if record.phase not in {"acquiring", "leased", "opening"}:
                    raise ResourceConflict("request has no pending acquisition/open reconciliation")
                return await self._acquire_open(context, record, check)

    def _ready(self, record):
        self._destination(record.request.location)
        session = record.session
        if (session is None or record.phase != "active" or session.state != "ready"
                or session.controller != "agent" or self._key(record.request.scope, record.session_id) in self._fences
                or self._key(record.request.scope, record.session_id) not in self._verified_sessions
                or record.pending_control is not None
                or session.expires_at <= self._now() or record.lease.expires_at <= self._now()):
            raise ResourceConflict("visual session is fenced or requires reconciliation")

    @asynccontextmanager
    async def session_operation(self, context, session_id, action=VisualAction.TRANSFER):
        """Broker/inference boundary; retains original privacy/lease custody."""
        action = VisualAction(action)
        async with self.operation(context, action) as check:
            async with self._lock(session_id):
                record = await self.repository.get(self.scope(context), session_id=session_id, guard=check)
                self._ready(record)
                def current():
                    check()
                    self._ready(record)
                yield record.session, self._driver(record), current
                current()

    async def status(self, context, session_id):
        async with self.operation(context, VisualAction.STATUS) as check:
            record = await self.repository.get(self.scope(context), session_id=session_id, guard=check)
            # Custody and physical release are separate facts. No private
            # endpoint, credential or action text enters an operator view.
            return {"request_id": record.request.request_id, "session_id": record.session_id,
                    "phase": record.phase, "session": record.session.model_dump(mode="json") if record.session else None,
                    "pending_control": record.pending_control.model_dump(mode="json") if record.pending_control else None,
                    "lease": record.lease.model_dump(mode="json") if record.lease else None,
                    "requires_reconciliation": self._key(record.request.scope, record.session_id) not in self._verified_sessions or record.pending_control is not None}

    async def observe(self, context, session_id):
        async with self.session_operation(context, session_id, VisualAction.OBSERVE) as (session, driver, check):
            observation = await self._call(context, check, driver.observe, session)
            if (observation.frame.session_id != session_id or observation.frame.control_epoch != session.control_epoch
                    or observation.frame.observed_at > self._now() or observation.frame.observed_at >= session.expires_at):
                raise ResourceConflict("observation does not match current session")
            record = await self.repository.get(self.scope(context), session_id=session_id, guard=check)
            record.frame, record.last_activity_at = observation.frame, self._now()
            # Pixels remain response-only; durable raw recording is disabled.
            await self.repository.save(self.scope(context), record, check)
            check()
            return observation

    async def act(self, context, request):
        request = ActionRequest.model_validate(request)
        async with self.operation(context, VisualAction.ACT) as check:
            scope = self.scope(context)
            async with self._lock(request.session_id):
                record = await self.repository.get(scope, session_id=request.session_id, guard=check)
                # Deduplication precedes stale-frame checks: an acknowledged
                # duplicate returns the same receipt without another effect.
                try:
                    previous = await self.repository.action(scope, request.session_id, request.action_id, guard=check)
                except ResourceNotFound:
                    previous = None
                if previous is not None:
                    if previous.request != request:
                        raise ResourceConflict("action ID already identifies different input")
                    if previous.receipt is None or previous.receipt.outcome == "unknown":
                        raise ResourceOutcomeUnknown("action requires original-ID status reconciliation")
                    check()
                    return previous.receipt
                self._ready(record)
                if await self.repository.has_pending_actions(scope, request.session_id, guard=check):
                    raise ResourceOutcomeUnknown("reconcile accepted session input before new actions")
                frame = record.frame
                if (frame is None or request.observation_id != frame.observation_id
                        or request.control_epoch != record.session.control_epoch
                        or request.not_after <= self._now()
                        or request.not_after > min(record.session.expires_at, context.expires_at)
                        or (request.not_after-self._now()).total_seconds() > self.policy.operation_seconds
                        or request.action.kind not in record.session.capabilities):
                    raise ResourceConflict("action does not match current observation and authority")
                if request.action.kind == "click" and (request.action.x >= frame.width or request.action.y >= frame.height):
                    raise ResourceConflict("click is outside current screen")
                if self._inflight >= self.policy.max_inflight_actions:
                    raise ResourceConflict("normal visual input capacity exhausted")
                await self.repository.action(scope, request.session_id, request.action_id, accept=request, guard=check)
                record.frame, record.last_activity_at = None, self._now()
                await self.repository.save(scope, record, check)
                self._inflight += 1
            try:
                def effect_check():
                    check()
                    self._ready(record)
                    if request.not_after <= self._now():
                        raise ResourceConflict("action deadline expired before effect")
                effect_check()
                receipt = await self._call(context, check, self._driver(record).act, record.session, request)
                self._receipt(request, receipt)
                # Preserve accepted outcome even if Stop raced its response;
                # no session save can overwrite a newer control epoch.
                await self.repository.action(scope, request.session_id, request.action_id, receipt=receipt, guard=check)
                check()
                return receipt
            finally:
                self._inflight -= 1

    @staticmethod
    def _receipt(request, receipt):
        if (receipt.action_id != request.action_id or receipt.session_id != request.session_id
                or receipt.observation_id != request.observation_id or receipt.control_epoch != request.control_epoch):
            raise ResourceConflict("action receipt does not identify the accepted input")

    async def action_status(self, context, session_id, action_id):
        async with self.operation(context, VisualAction.STATUS) as check:
            scope = self.scope(context)
            record = await self.repository.get(scope, session_id=session_id, guard=check)
            accepted = await self.repository.action(scope, session_id, action_id, guard=check)
            if accepted.receipt is not None and accepted.receipt.outcome != "unknown":
                return accepted.receipt
            try:
                receipt = await self._call(context, check, self._driver(record).action_status, record.session, action_id)
            except ResourceNotFound:
                raise ResourceOutcomeUnknown("original accepted action is not durably known to this worker") from None
            self._receipt(accepted.request, receipt)
            await self.repository.action(scope, session_id, action_id, receipt=receipt, guard=check)
            return receipt

    async def reconcile_session(self, context, session_id):
        async with self.operation(context, VisualAction.STATUS) as check:
            scope = self.scope(context)
            async with self._lock(session_id):
                record = await self.repository.get(scope, session_id=session_id, guard=check)
                if record.session is None or record.pending_control is not None:
                    raise ResourceOutcomeUnknown("session/control acceptance requires explicit safety reconciliation")
                session = await self._call(context, check, self._driver(record).status, scope, session_id)
                self._session(record, session)
                if session != record.session or session.expires_at <= self._now():
                    self._fences.add(self._key(scope, session_id))
                    raise ResourceOutcomeUnknown("worker incarnation or accepted session state changed")
                if await self.repository.has_pending_actions(scope, session_id, guard=check):
                    raise ResourceOutcomeUnknown("reconcile original accepted actions before resuming the session")
                record.frame = None
                await self.repository.save(scope, record, check)
                self._verified_sessions.add(self._key(scope, session_id))
                return session

    async def control(self, context, session_id, request, *, action):
        request = ControlRequest.model_validate(request)
        expected = {VisualAction.STOP: "pause", VisualAction.TAKEOVER: "takeover", VisualAction.HANDBACK: "resume"}
        if action not in expected or request.operation != expected[action] or request.session_id != session_id:
            raise ResourceConflict("invalid visual safety operation")
        async with self.operation(context, action) as check:
            scope = self.scope(context)
            # Fence synchronously, before any queued database work. Safety
            # never reserves a normal action/history slot.
            fence = self._key(scope, session_id)
            self._fences.add(fence)
            try:
                record = await self.repository.get(scope, session_id=session_id, guard=check)
            except ResourceNotFound:
                self._fences.discard(fence)
                raise
            if record.last_control is not None and record.last_control.request_id == request.request_id:
                if record.last_control != request:
                    raise ResourceConflict("control ID identifies different input")
                if record.phase == "active" and record.session == record.control_receipt:
                    self._fences.discard(self._key(scope, session_id))
                return record.control_receipt
            if record.session is None:
                raise ResourceConflict("visual session has not opened")
            if record.pending_control is not None and record.pending_control != request:
                raise ResourceOutcomeUnknown("reconcile the original control request")
            if record.pending_control is None and request.expected_control_epoch != record.session.control_epoch:
                raise ResourceConflict("stale visual control epoch")
            if action == VisualAction.HANDBACK:
                self._destination(record.request.location)
            if action == VisualAction.HANDBACK and record.session.controller != "human":
                raise ResourceConflict("handback requires explicit current human control")
            record.pending_control, record.frame, record.phase = request, None, "fenced"
            await self.repository.save(scope, record, check)
            session = await self._call(context, check, self._driver(record).control, record.session, request)
            self._session(record, session)
            target = {VisualAction.STOP: ("paused", "none"), VisualAction.TAKEOVER: ("paused", "human"), VisualAction.HANDBACK: ("ready", "agent")}[action]
            if ((session.state, session.controller) != target or session.control_epoch <= request.expected_control_epoch):
                raise ResourceOutcomeUnknown("driver did not verify the requested control fence")
            record = await self.repository.get(scope, session_id=session_id, guard=check)
            record.session, record.pending_control = session, None
            record.last_control, record.control_receipt = request, session
            record.phase = "active" if action == VisualAction.HANDBACK else "fenced"
            await self.repository.save(scope, record, check)
            self._verified_sessions.add(self._key(scope, session_id))
            if action == VisualAction.HANDBACK:
                # A fresh screen must be observed before any new action.
                self._fences.discard(self._key(scope, session_id))
            return session

    async def release(self, context, session_id):
        async with self.operation(context, VisualAction.RELEASE) as check:
            scope = self.scope(context)
            fence = self._key(scope, session_id)
            self._fences.add(fence)
            try:
                record = await self.repository.get(scope, session_id=session_id, guard=check)
            except ResourceNotFound:
                self._fences.discard(fence)
                raise
            if record.lease is None:
                raise ResourceOutcomeUnknown("unknown acquisition retains request cleanup custody")
            if record.phase == "released":
                return {"session_id": session_id, "released": True}
            record.frame, record.phase = None, "fenced"
            await self.repository.save(scope, record, check)
            # Physical destruction has its own verb and outcome. A timeout
            # does not erase the original lease/connection or claim release.
            lease = await self._call(context, check, self._provider(record).release, scope, record.lease)
            self._lease(record, lease, releasing=True)
            record.lease = lease
            if lease.state == "released":
                record.phase, record.connection = "released", None
            await self.repository.save(scope, record, check)
            return {"session_id": session_id, "released": lease.state == "released"}

    async def fence_owned_sessions(self, scope):
        """Host lifecycle safety, without minting a user/delegated grant.

        This may only fence existing custody. It cannot observe, hand back,
        acquire, extend a lease, or destroy a process. Unknown control remains
        durable and is reported to lifecycle diagnostics.
        """
        if self.repository._key is None:
            if not await self.repository.db.table_exists("visual_computer_requests"):
                return []  # An unused capability never acquired a worker.
            await self.repository.ensure_initialized()
        uncertain = []
        for record in await self.repository.records(scope):
            self._fences.add(self._key(scope, record.session_id))
            if record.session is None or record.phase == "released":
                continue
            request = record.pending_control
            if request is not None and request.operation != "pause":
                # Never replay an uncertain handback/takeover as lifecycle stop.
                uncertain.append(record.session_id)
                continue
            request = request or ControlRequest(request_id=uuid4().hex, session_id=record.session_id,
                expected_control_epoch=record.session.control_epoch, operation="pause")
            record.pending_control, record.phase, record.frame = request, "fenced", None
            try:
                await self.repository.save(scope, record)
                async with asyncio.timeout(self.policy.operation_seconds):
                    session = await self._driver(record).control(record.session, request)
                self._session(record, session)
                if session.controller != "none" or session.control_epoch <= request.expected_control_epoch:
                    raise ResourceOutcomeUnknown("lifecycle input fence requires reconciliation")
                record.session, record.pending_control = session, None
                await self.repository.save(scope, record)
            except (ResourceError, TimeoutError, OSError):
                uncertain.append(record.session_id)
        return uncertain

    async def maintain_owned_sessions(self, scope):
        """Release expired/idle original leases; failed is never released.

        Cleanup is a consequence of the admitted fixed lifetime, not an
        inference action or an implicit lease renewal. It consumes no input
        capacity and retains uncertain physical custody for the next sweep.
        """
        if self.repository._key is None:
            if not await self.repository.db.table_exists("visual_computer_requests"):
                return []
            await self.repository.ensure_initialized()
        uncertain = []
        for record in await self.repository.records(scope):
            if record.phase == "released" or record.lease is None:
                continue
            last = record.last_activity_at or record.request.requested_at
            if (self._now() < record.lease.expires_at
                    and (self._now()-last).total_seconds() < record.request.idle_ttl_seconds):
                continue
            self._fences.add(self._key(scope, record.session_id))
            record.frame, record.phase = None, "fenced"
            try:
                await self.repository.save(scope, record)
                async with asyncio.timeout(self.policy.operation_seconds):
                    lease = await self._provider(record).release(scope, record.lease)
                self._lease(record, lease, releasing=True)
                record.lease = lease
                if lease.state == "released":
                    record.phase, record.connection = "released", None
                else:
                    uncertain.append(record.session_id)
                await self.repository.save(scope, record)
            except (ResourceError, TimeoutError, OSError):
                uncertain.append(record.session_id)
        return uncertain

    async def list_sessions(self, context):
        async with self.operation(context, VisualAction.STATUS):
            records = await self.repository.records(self.scope(context))
            return [{"session_id":record.session_id, "request_id":record.request.request_id,
                     "phase":record.phase, "controller":record.session.controller if record.session else "none"}
                    for record in records]
