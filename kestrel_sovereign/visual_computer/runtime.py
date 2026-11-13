"""Canonical host orchestration of the SDK's visual computer contracts.

Providers acquire; drivers interact. The runtime never substitutes host shell
execution. Accepted intents precede effects; unknown outcomes retain original
IDs and require explicit reconciliation. Safety input fencing is independent
of normal action capacity and precedes the driver's priority control call.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from enum import StrEnum
import math
from uuid import uuid4

from kestrel_sdk.computer_use import (
    ActionRequest, ControlRequest,
    EnvironmentRequest, SessionRequest,
)
from kestrel_sdk.resources import ResourceConflict, ResourceError, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope, ResourceUnsupported
from kestrel_sdk.computer_use.acquisition import CustodiedEnvironmentProvider
from kestrel_sdk.operator import OperatorAuthorizationError

from .dispatch import original_dispatch, NativeActionNotDispatched, raise_driver_borrow_outcome
from .repository import CustodyRecord, VisualCustodyRepository
from .egress import ActiveCallEgress


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
    HUMAN_VIEW = "visual.human.view"
    HUMAN_INPUT = "visual.human.input"
    HUMAN_REVOKE = "visual.human.revoke"


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


_LOCAL_ADMIT_ISSUER = object()


class VisualComputerRuntime:
    def __init__(self, repository: VisualCustodyRepository, authority, *, providers, drivers,
                 policy: VisualPolicy | None = None, check_destination=None, egress=None):
        self.repository, self.authority = repository, authority
        self.providers, self.drivers = dict(providers), dict(drivers)
        self.policy = policy or VisualPolicy()
        self.check_destination = check_destination
        self.egress = egress or ActiveCallEgress(clock=authority.clock)
        self._locks = {}
        self._fences = set()
        self._control_serials = {}
        self._verified_sessions = set()
        self._inflight = 0
        self.human = None
        from .local_worker import bind_local_worker_hosts
        bind_local_worker_hosts(self)

    @staticmethod
    def scope(context):
        return ResourceScope(tenant_id=context.tenant_id, principal_id=context.principal_id,
                             agent_id=context.acting_agent_id)

    @staticmethod
    def _key(scope, session_id):
        return (scope.tenant_id, scope.principal_id, scope.agent_id, session_id)

    def _fence(self, scope, session_id):
        key = self._key(scope, session_id)
        self._fences.add(key)
        self._control_serials[key] = self._control_serials.get(key, 0)+1

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

    async def _session_call(self, context, record, check, operation, *args):
        """Join an exact original SDK read/control before physical release.

        The existing active-call owner lends its original synchronous checker
        to one native child. No child resolves authority or replaces custody.
        The finite method selection excludes acquisition and provider cleanup.
        """
        from kestrel_sovereign._async_ownership import await_owned_operation
        driver, session, scope = self._driver(record), record.session, record.request.scope
        if operation == "observe":
            method, arguments = driver.observe, (session,)
        elif operation == "status":
            method, arguments = driver.status, (scope, record.session_id)
        elif operation == "action_status":
            method, arguments = driver.action_status, (session, *args)
        elif operation == "control":
            method, arguments = driver.control, (session, *args)
        else:
            raise ValueError("unsupported original session call")
        serial = self._control_serials.get(self._key(scope, record.session_id), 0)
        def current():
            check()
            self._require_native_session(record)
            if self._control_serials.get(self._key(scope, record.session_id), 0) != serial:
                raise ResourceConflict("original visual session call was fenced")
        identifier = operation + ":" + uuid4().hex
        deadline = min(context.expires_at, self._now()+timedelta(seconds=self.policy.operation_seconds))
        async with self.egress._activate(session, identifier, current, deadline=deadline, purpose="session"):
            async def dispatch():
                async with self.egress._operation(session, identifier, purpose="session") as borrowed:
                    latest = await self.repository.get(scope, session_id=record.session_id, guard=borrowed)
                    self._require_native_session(latest)
                    if latest.session != session or latest.lease != record.lease:
                        raise ResourceConflict("original visual session changed before SDK entry")
                    return await self._call(context, borrowed, method, *arguments)
            # Priority release, expiry and original caller withdrawal forward
            # cancellation through the SAME call so cleanup receives it once.
            call = self.egress._calls[self.egress._key(session, identifier, "session")]
            outcome = await await_owned_operation(dispatch(), retain_originals=True,
                cancel_on_caller_cancellation=call.cancel)
            return raise_driver_borrow_outcome(outcome, operation="original visual session call", retain_originals=True)

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

    @staticmethod
    def _custodied_provider(provider):
        if isinstance(provider, CustodiedEnvironmentProvider):
            if not callable(getattr(provider, 'acquire_with_custody', None)):
                raise ResourceUnsupported('provider acquisition custody negotiation is invalid')
            return True
        if hasattr(provider, 'acquire_with_custody'):
            raise ResourceUnsupported('provider acquisition custody contract is incomplete')
        return False

    def _native_quote_host(self, provider):
        from .runpod_environment import RunpodEnvironmentHost
        version = getattr(provider, 'native_environment_host_api_version', None)
        host = getattr(provider, 'environment_host', None)
        if version is None and host is None:
            return None
        if (type(version) is not int or version != 1 or type(host) is not RunpodEnvironmentHost
                or host.api_version != 1 or host._runtime is not self or host._provider is not provider
                or not self._custodied_provider(provider)):
            raise ResourceUnsupported('Native environment quote custody negotiation is incomplete')
        return host

    def _native_lifecycle_host(self, provider):
        from .runpod_environment import RunpodEnvironmentHost
        version = getattr(provider, 'native_environment_lifecycle_api_version', None)
        host = getattr(provider, 'environment_host', None)
        if (type(version) is not int or version != 1 or type(host) is not RunpodEnvironmentHost
                or host.lifecycle_api_version != 1 or host._runtime is not self
                or host._provider is not provider or self._native_quote_host(provider) is not host):
            raise ResourceUnsupported('Native current environment lifecycle negotiation is incomplete')
        return host

    @staticmethod
    def _requires_remote_worker_admission(record):
        # Native preparation binding is authenticated on the SAME encrypted
        # acquisition row even when the optional native quote was not used.
        # A plain, genuinely supported deterministic provider remains on its
        # own contract; no provider name/DTO can turn these native facts into
        # worker/bootstrap conformance or a resident erase permission.
        return (record.remote_environment is not None or
            (record.acquisition is not None and record.acquisition.binding is not None))

    async def _acquire_custodied(self, context, record, check, provider):
        from .storage_acquisition import _CoreAcquisitionCustody
        scope = self.scope(context)
        custody = _CoreAcquisitionCustody(self.repository, scope, record, guard=check)
        async with custody._activate_original():
            lease = await self._call(context, check, provider.acquire_with_custody,
                record.request, record.quote, custody)
        self._lease(record, lease, acquiring=True)
        # The same original operation's accepted native ACK is mandatory.
        # Reload its exact ciphertext/revision instead of saving a stale DTO.
        return await self.repository._accepted_acquisition(scope, custody, lease, check)

    async def admit(self, context, request):
        async with self.operation(context, VisualAction.ADMIT) as check:
            request = self._request(context, request)
            # Serialize admission by request ID, not a caller-selected resource name.
            async with self._lock((self.scope(context).model_dump_json(), request.request_id)):
                record = await self.repository.admit(self.scope(context), CustodyRecord(request=request, session_id=uuid4().hex, last_activity_at=self._now()), check)
                if record.phase == "active":
                    check()
                    return record.session
                if record.phase == "retired":
                    if record.retired_from == "acquiring":
                        raise ResourceConflict("original request retired after verified setup cleanup; use a new request ID")
                    raise ResourceConflict("original request retired without acquisition; use a new request ID")
                if record.phase not in {"admitted", "quoted"}:
                    raise ResourceOutcomeUnknown("existing request requires original-ID reconciliation")
                native_quote = record.remote_environment is not None
                original_setup = None
                try:
                    provider = self._provider(record)
                    native_quote = native_quote or (getattr(provider, "native_environment_host_api_version", None) is not None
                        or getattr(provider, "environment_host", None) is not None)
                    custodied = self._custodied_provider(provider)
                    quote_host = self._native_quote_host(provider)
                    if quote_host is not None and record.quote is not None and record.remote_environment is None:
                        raise ResourceUnsupported('Original private native environment quote is absent; no retrofit')
                    if record.remote_environment is not None:
                        if quote_host is None:
                            raise ResourceUnsupported('Original native environment quote custody is required; fallback forbidden')
                        async with quote_host._activate_quote(context, record) as original_quote:
                            original_quote.restore_original()
                    if record.quote is None:
                        capabilities = await self._call(context, check, provider.capabilities)
                        if (capabilities.provider_id != request.provider_id or capabilities.location != request.location
                                or request.surface not in capabilities.surfaces
                                or not request.required_capabilities <= capabilities.operations):
                            raise ResourceConflict("provider capabilities do not satisfy visual request")
                        if quote_host is not None:
                            async with quote_host._activate_quote(context, record) as original_quote:
                                quote = await self._call(context, check, provider.quote, request)
                                self._quote(record, quote)
                                if self._native_quote_host(provider) is not quote_host:
                                    raise ResourceUnsupported('Native environment quote custody changed')
                                record = await self.repository._accepted_runpod_quote(self.scope(context), original_quote, quote)
                        else:
                            quote = await self._call(context, check, provider.quote, request)
                            self._quote(record, quote)
                            record.quote, record.phase = quote, "quoted"
                            await self.repository.save(self.scope(context), record, check)
                    self._quote(record, record.quote)
                    if self._custodied_provider(provider) != custodied:
                        raise ResourceUnsupported('provider acquisition custody negotiation changed')
                    from .local_worker import LocalTwoDomainProvider, _OriginalLocalWorkerAdmission
                    if type(provider) is LocalTwoDomainProvider:
                        # SAME original ADMIT check: no nested authority
                        # operation and no fabricated pre-session SDK object.
                        admission = _OriginalLocalWorkerAdmission(self, context, record, check,
                                                                 _issuer=_LOCAL_ADMIT_ISSUER)
                        try:
                            original_setup = provider._setup_host._issue_local_setup(admission)
                            async with self.egress._activate_local_worker_setup(original_setup):
                                return await original_setup.run_original()
                        finally:
                            admission._active = False
                    if custodied:
                        record = await self._acquire_custodied(context, record, check, provider)
                    else:
                        record.phase = "acquiring"
                        await self.repository.save(self.scope(context), record, check)
                    return await self._acquire_open(context, record, check)
                except BaseException as original:
                    # Issuance alone is not a failed physical setup. An unused
                    # setup follows the existing no-acquisition CAS below,
                    # which independently validates its exact durable facts.
                    if original_setup is not None and original_setup._used and not original_setup._handed_off:
                        try:
                            async with self._lock(record.session_id):
                                await self._settle_failed_local_setup(self.scope(context), record, check)
                        except BaseException as settlement:
                            from kestrel_sovereign.execution_custody import execution_terminal_error
                            raise execution_terminal_error(original, settlement, retain_originals=True)
                        raise
                    try:
                        await self.repository.retire_unacquired(self.scope(context), record, at=self._now(),
                            guard=check if native_quote else None)
                    except ResourceConflict:
                        pass  # Acquiring/unknown or a newer owner remains charged.
                    except BaseException as retirement_error:
                        original.add_note("original request retirement remains unconfirmed: "+type(retirement_error).__name__)
                    raise

    async def _acquire_open(self, context, record, check):
        scope, provider = self.scope(context), self._provider(record)
        # Re-enter only through explicit request reconciliation. The existing
        # deterministic provider contract is idempotent for this exact request
        # and accepted quote. A custody provider never falls through to it.
        if record.lease is None:
            if self._custodied_provider(provider) and record.acquisition is None:
                raise ResourceOutcomeUnknown('custodied provider requires original quoted admission; uncustodied create is forbidden')
            if record.acquisition is not None:
                raise ResourceOutcomeUnknown('original custodied acquisition has no native resource ID; create replay is forbidden')
            lease = await self._call(context, check, provider.acquire, record.request, record.quote)
            self._lease(record, lease, acquiring=True)
            record.lease, record.phase = lease, "leased"
            await self.repository.save(scope, record, check)
        lease = record.lease
        if self._requires_remote_worker_admission(record):
            if record.remote_environment is not None:
                host = self._native_lifecycle_host(provider)
                async with host._activate_inspect(context, record) as original:
                    inspected = await self._call(context, check, provider.inspect, scope, lease)
                    record = await self.repository._accepted_runpod_lifecycle(scope, original, inspected)
            # API placement facts cannot supply a supported isolated remote worker.
            raise ResourceUnsupported('Remote browser bootstrap and authenticated worker transport are unverified')
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

    async def _settle_failed_local_setup(self, scope, record, check):
        from .local_worker import LocalTwoDomainProvider
        provider = self.providers.get(record.request.provider_id)
        if type(provider) is not LocalTwoDomainProvider:
            return None
        # Callers hold the existing session lock. Foreground operations retain
        # current authority; maintenance passes no new grant after rechecking
        # the already admitted fixed lease's cleanup eligibility under that lock.
        # A stale in-memory lease assignment is never a durable lease receipt.
        record = await self.repository.get(scope, session_id=record.session_id, guard=check)
        host = provider._setup_host
        original = host._retained_setup(record)
        if original is None:
            return None
        await original._join_retirement(record, failed=True)
        if record.phase in {"admitted", "quoted"}:
            record = await self.repository.retire_unacquired(scope, record, at=self._now(), guard=check)
        elif record.phase not in {"retired", "released"}:
            if record.lease is None:
                record = await self.repository._retire_failed_local_setup(scope, record, original, at=self._now(), guard=check)
            else:
                lease = record.lease.model_copy(update={"state": "released", "released_at": self._now(), "actual_cost_usd": Decimal(0)})
                self._lease(record, lease, releasing=True)
                record.lease, record.phase = lease, "released"
                await self.repository.save(scope, record, check)
        # Only a durable ACK permits dropping exact completion/connection
        # custody. Every failed/ambiguous write leaves both retained.
        host._discard_retired_setup(original, record)
        return record

    async def _discard_released_local_connection(self, record, check=None):
        from .local_worker import LocalTwoDomainProvider
        provider = self.providers.get(record.request.provider_id)
        if type(provider) is LocalTwoDomainProvider:
            if check is not None:
                check()
            await provider._setup_host._discard_released_connection(record)

    async def reconcile_request(self, context, request_id):
        async with self.operation(context, VisualAction.ADMIT) as check:
            scope = self.scope(context)
            async with self._lock((scope.model_dump_json(), request_id)):
                record = await self.repository.get(scope, request_id=request_id, guard=check)
                async with self._lock(record.session_id):
                    settled = await self._settle_failed_local_setup(scope, record, check)
                    if settled is not None:
                        return self._retirement(settled) if settled.phase == "retired" else {"session_id": settled.session_id, "released": True}
                    record = await self.repository.get(scope, session_id=record.session_id, guard=check)
                    if record.phase == "released":
                        # The original join and same-entry removal serialize
                        # with foreground release on this existing session lock.
                        await self._discard_released_local_connection(record, check)
                        return {"session_id": record.session_id, "released": True}
                if record.phase == "retired":
                    return self._retirement(record)
                if record.phase in {"admitted", "quoted"}:
                    record = await self.repository.retire_unacquired(scope, record, at=self._now(), guard=check)
                    return self._retirement(record)
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

    def _transfer_ready(self, record):
        self._destination(record.request.location)
        key = self._key(record.request.scope, record.session_id)
        session = record.session
        if (record.human_private or session is None or session.state != "paused" or session.controller != "none"
                or record.pending_control is not None or record.phase != "fenced"
                or key not in self._verified_sessions
                or session.expires_at <= self._now() or record.lease.expires_at <= self._now()):
            raise ResourceConflict("workspace transfer requires exact verified paused control epoch")

    @asynccontextmanager
    async def session_operation(self, context, session_id, action=VisualAction.TRANSFER):
        """Broker/inference boundary; retains original privacy/lease custody."""
        action = VisualAction(action)
        async with self.operation(context, action) as check:
            async with self._lock(session_id):
                record = await self.repository.get(self.scope(context), session_id=session_id, guard=check)
                serial = self._control_serials.get(self._key(record.request.scope, record.session_id), 0)
            def current():
                check()
                if self._control_serials.get(self._key(record.request.scope, record.session_id), 0) != serial:
                    raise ResourceConflict("retained visual session control epoch changed")
                if action == VisualAction.TRANSFER:
                    self._transfer_ready(record)
                else:
                    self._ready(record)
            current()
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
                    "requires_reconciliation": record.phase == "releasing" or (record.phase not in {"retired", "released"} and (self._key(record.request.scope, record.session_id) not in self._verified_sessions or record.pending_control is not None))}

    async def observe(self, context, session_id):
        async with self.session_operation(context, session_id, VisualAction.OBSERVE) as (session, driver, check):
            record = await self.repository.get(self.scope(context), session_id=session_id, guard=check)
            observation = await self._session_call(context, record, check, "observe")
            if (observation.frame.session_id != session_id or observation.frame.control_epoch != session.control_epoch
                    or observation.frame.observed_at > self._now() or observation.frame.observed_at >= session.expires_at):
                raise ResourceConflict("observation does not match current session")
            record.frame, record.last_activity_at = observation.frame, self._now()
            # Pixels remain response-only; durable raw recording is disabled.
            await self.repository.save(self.scope(context), record, check)
            check()
            return observation

    async def act(self, context, request, *, _step_owner=None):
        request = ActionRequest.model_validate(request)
        async with self.operation(context, VisualAction.ACT) as check:
            scope = self.scope(context)
            prepared = dispatched = inflight = driver_entered = False
            result_stack = AsyncExitStack()
            result_owner = original = None
            try:
                async with self._lock(request.session_id):
                    record = await self.repository.get(scope, session_id=request.session_id, guard=check)
                    # Deduplication precedes stale-frame checks: an acknowledged
                    # duplicate returns the same receipt without another effect.
                    try:
                        previous = await self.repository.action(scope, request.session_id, request.action_id, guard=check)
                    except ResourceNotFound:
                        previous = None
                    from .step import PlanningRecord, _OriginalVisualStep
                    step_planning = type(previous) is PlanningRecord
                    if step_planning:
                        if type(_step_owner) is not _OriginalVisualStep or not _step_owner.own_pending(scope, request.session_id, request.action_id,
                                _step_owner._claim):
                            raise ResourceOutcomeUnknown("Original step planning cannot be dispatched")
                        _step_owner._require_owner()
                        previous = None
                    elif _step_owner is not None:
                        raise ResourceConflict("Step promotion requires its original planning row")
                    if previous is not None:
                        if previous.channel != "agent" or previous.request != request:
                            raise ResourceConflict("action ID already identifies different input")
                        if not previous.dispatch_started and previous.receipt is None:
                            previous = await self.repository.action(scope, request.session_id, request.action_id,
                                abandon_prepared=True, guard=check)
                        if previous.receipt is None or previous.receipt.outcome == "unknown":
                            raise ResourceOutcomeUnknown("action requires original-ID status reconciliation")
                        check()
                        return previous.receipt
                    self._ready(record)
                    if await self.repository.has_pending_actions(scope, request.session_id, guard=check, _step_owner=_step_owner):
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
                    self._inflight += 1
                    inflight = True
                    serial = self._control_serials.get(self._key(scope, request.session_id), 0)
                    if step_planning:
                        await _step_owner.promote(record, request)
                    await self.repository.action(scope, request.session_id, request.action_id, accept=request, prepare=True, guard=check)
                    prepared = True
                    record.frame, record.last_activity_at = None, self._now()
                    await self.repository.save(scope, record, check)
                    original = result_stack.enter_context(original_dispatch(record.session, request))
                    result_owner = await result_stack.enter_async_context(self.repository.original_action_result(
                        scope, record.session, request, guard=check))
                    await self.repository.action(scope, request.session_id, request.action_id,
                        dispatch_revision=record.revision, guard=check, _result_owner=result_owner)
                    dispatched = True
                def effect_check():
                    from kestrel_sovereign.execution_custody import require_execution_work
                    require_execution_work(self.repository.db.backend)
                    check()
                    if _step_owner is not None:
                        _step_owner._check_original()
                    self._ready(record)
                    if self._control_serials.get(self._key(scope, request.session_id), 0) != serial:
                        raise ResourceConflict("original action control epoch changed")
                    if request.not_after <= self._now():
                        raise ResourceConflict("action deadline expired before effect")
                effect_check()
                async with self.egress.activate(record.session, request.action_id, effect_check,
                        deadline=min(request.not_after, context.expires_at)):
                    async def dispatch():
                        nonlocal driver_entered
                        current = await self.repository.get(scope, session_id=request.session_id, guard=effect_check)
                        if current.revision != record.revision or current.session != record.session or current.lease != record.lease:
                            raise ResourceConflict("accepted visual frame/session changed before driver dispatch")
                        effect_check()
                        driver_entered = True
                        original.enter_driver()
                        try:
                            return await self._driver(record).act(record.session, request)
                        except NativeActionNotDispatched as fact:
                            original.require(fact)
                            try:
                                await result_owner.not_dispatched(original, fact)
                            except BaseException as settlement:
                                from kestrel_sovereign.execution_custody import execution_observation_error
                                raise execution_observation_error(fact.original, settlement) from None
                            raise fact.original
                    receipt = await self._call(context, check, dispatch)
                self._receipt(request, receipt)
                # Preserve accepted outcome even if Stop raced its response;
                # no session save can overwrite a newer control epoch.
                await self.repository.action(scope, request.session_id, request.action_id, receipt=receipt, guard=check)
                check()
                return receipt
            except BaseException as failure:
                if result_owner is not None and not driver_entered:
                    try:
                        await result_owner.not_dispatched(original, original.refuse_unentered(failure))
                    except BaseException as settlement:
                        from kestrel_sovereign.execution_custody import execution_observation_error
                        raise execution_observation_error(failure, settlement) from None
                elif prepared and not dispatched:
                    await self._abandon_prepared_action(scope, request)
                raise
            finally:
                if inflight:
                    self._inflight -= 1
                await result_stack.aclose()

    async def _abandon_prepared_action(self, scope, request):
        """Retire only a preparation whose dispatch CAS never acknowledged.

        The repository retains an ambiguous dispatch. Actual negative results
        require the separately retained original result owner, never booleans
        supplied to this preparation cleanup.
        """
        await self.repository.action(scope, request.session_id, request.action_id,
            abandon_prepared=True)

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
            from .step import PlanningRecord
            if type(accepted) is PlanningRecord:
                raise ResourceOutcomeUnknown("Host visual step has no dispatched SDK action")
            if accepted.channel != "agent":
                raise ResourceNotFound("visual action not found")
            if not accepted.dispatch_started and accepted.receipt is None:
                accepted = await self.repository.action(scope, session_id, action_id,
                    abandon_prepared=True, guard=check)
            if accepted.receipt is not None and accepted.receipt.outcome != "unknown":
                return accepted.receipt
            try:
                receipt = await self._session_call(context, record, check, "action_status", action_id)
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
                self._require_native_session(record)
                if record.session is None or record.pending_control is not None:
                    raise ResourceOutcomeUnknown("session/control acceptance requires explicit safety reconciliation")
                session = await self._session_call(context, record, check, "status")
                self._session(record, session)
                if session != record.session or session.expires_at <= self._now():
                    self._fence(scope, session_id)
                    await self.egress.revoke_session(scope, session_id)
                    raise ResourceOutcomeUnknown("worker incarnation or accepted session state changed")
                if await self.repository.has_pending_actions(scope, session_id, guard=check):
                    raise ResourceOutcomeUnknown("reconcile original accepted actions before resuming the session")
                record.frame = None
                await self.repository.save(scope, record, check)
                self._verified_sessions.add(self._key(scope, session_id))
                return session

    async def control(self, context, session_id, request, *, action, share_screen=False):
        request = ControlRequest.model_validate(request)
        expected = {VisualAction.STOP: "pause", VisualAction.TAKEOVER: "takeover", VisualAction.HANDBACK: "resume"}
        if action not in expected or request.operation != expected[action] or request.session_id != session_id:
            raise ResourceConflict("invalid visual safety operation")
        async with self.operation(context, action) as check, AsyncExitStack() as handback_stack:
            scope = self.scope(context)
            # Fence synchronously, before any queued database work. Safety
            # never reserves a normal action/history slot.
            fence = self._key(scope, session_id)
            self._fence(scope, session_id)
            serial = self._control_serials[fence]
            def current():
                check()
                if self._control_serials.get(fence) != serial:
                    raise ResourceConflict("original visual control was superseded")
            await self.egress.revoke_session(scope, session_id)
            try:
                record = await self.repository.get(scope, session_id=session_id, guard=current)
            except ResourceNotFound:
                self._fences.discard(fence)
                self._control_serials.pop(fence, None)
                raise
            if record.last_control is not None and record.last_control.request_id == request.request_id:
                if record.last_control != request:
                    raise ResourceConflict("control ID identifies different input")
                if (action == VisualAction.HANDBACK and record.phase == "active"
                        and record.pending_control is None and record.session == record.control_receipt
                        and record.session.state == "ready" and record.session.controller == "agent"):
                    if self.human is not None and record.frame is None:
                        if share_screen is not True:
                            raise ResourceConflict("confirm the screen is ready to share before handback")
                        await handback_stack.enter_async_context(self.operation(context, VisualAction.OBSERVE))
                        current()
                        self._fences.discard(fence)
                        await self.observe(context, session_id)
                    current()
                    self._fences.discard(self._key(scope, session_id))
                return record.control_receipt
            self._require_native_session(record)
            if record.session is None:
                raise ResourceConflict("visual session has not opened")
            if record.pending_control is not None and record.pending_control != request:
                raise ResourceOutcomeUnknown("reconcile the original control request")
            if record.pending_control is None and request.expected_control_epoch != record.session.control_epoch:
                raise ResourceConflict("stale visual control epoch")
            if action == VisualAction.HANDBACK and self.human is not None and share_screen is not True:
                raise ResourceConflict("confirm the screen is ready to share before handback")
            if action == VisualAction.HANDBACK:
                self._destination(record.request.location)
                if self.human is not None:
                    await handback_stack.enter_async_context(self.operation(context, VisualAction.OBSERVE))
                    current()
            if action == VisualAction.HANDBACK and record.session.controller != "human":
                raise ResourceConflict("handback requires explicit current human control")
            if self.human is not None:
                await self.human.retire_owned(context, record, check)
                if action == VisualAction.HANDBACK and await self.repository.has_pending_actions(scope, session_id, guard=check):
                    raise ResourceOutcomeUnknown("reconcile original accepted input before handback")
                # Original Human retirement has its own durable publication
                # fence. The following same-row CAS still rejects any release
                # which raced it, including a newly read releasing revision.
                serial = self._control_serials.get(fence, 0)
            record.pending_control, record.frame, record.phase = request, None, "fenced"
            await self.repository.save(scope, record, current)
            session = await self._session_call(context, record, current, "control", request)
            self._session(record, session)
            target = {VisualAction.STOP: ("paused", "none"), VisualAction.TAKEOVER: ("paused", "human"), VisualAction.HANDBACK: ("ready", "agent")}[action]
            if ((session.state, session.controller) != target or session.control_epoch <= request.expected_control_epoch):
                raise ResourceOutcomeUnknown("driver did not verify the requested control fence")
            record = await self.repository.get(scope, session_id=session_id, guard=current)
            record.session, record.pending_control = session, None
            if self.human is not None and action == VisualAction.TAKEOVER:
                record.human_private = True
            if self.human is not None and action == VisualAction.HANDBACK:
                record.human_private = False
            record.last_control, record.control_receipt = request, session
            record.phase = "active" if action == VisualAction.HANDBACK else "fenced"
            await self.repository.save(scope, record, current)
            self._verified_sessions.add(self._key(scope, session_id))
            if action == VisualAction.HANDBACK:
                # A fresh screen must be observed before any new action.
                self._fences.discard(self._key(scope, session_id))
                if self.human is not None:
                    # Explicitly confirmed sharing gets a fresh agent frame,
                    # never any retained human image or observation ID.
                    await self.observe(context, session_id)
            return session

    @staticmethod
    def _retirement(record):
        if record.retired_from == "acquiring":
            return {"session_id": record.session_id, "request_id": record.request.request_id,
                    "retired": True, "acquired": None, "released": True,
                    "setup_cleanup": "verified_original_physical_retirement"}
        return {"session_id":record.session_id, "request_id":record.request.request_id,
                "retired":True, "acquired":False, "released":False}

    @staticmethod
    def _require_native_session(record):
        if record.phase in {"retired", "releasing", "released"}:
            raise ResourceConflict(f"{record.phase} visual session cannot accept native calls")

    async def _begin_release(self, scope, record, check=None):
        record.frame, record.phase = None, "releasing"
        await self.repository.save(scope, record, check)
        # Close stale readers at the durable publication boundary. A reader
        # cannot adopt the first fence's serial and enter after its drain.
        self._fence(scope, record.session_id)
        self._verified_sessions.discard(self._key(scope, record.session_id))
        await self.egress.revoke_session(scope, record.session_id)

    async def _release_and_settle(self, scope, record, check, *, context=None):
        """Keep caller withdrawal separate from an original local release fact.

        Both callers hold the existing session lock. Only the eligible sweep
        passes no user grant; foreground recovery retains its same checker.
        A failed observation may settle only an already issued exact local
        closer, through the provider's canonical original receipt boundary.
        """
        from kestrel_sovereign.execution_custody import execution_terminal_error
        from .local_worker import LocalTwoDomainProvider
        provider = self._provider(record)
        failure = None
        try:
            if context is not None:
                lease = await self._call(context, check, provider.release, scope, record.lease)
            else:
                async with asyncio.timeout(self.policy.operation_seconds):
                    lease = await provider.release(scope, record.lease)
        except BaseException as error:
            if type(provider) is not LocalTwoDomainProvider:
                raise
            failure = error
            try:
                current = await self.repository.get(scope, session_id=record.session_id, guard=check)
                if (current.phase != "releasing" or current.revision != record.revision
                        or current.request != record.request or current.quote != record.quote
                        or current.lease != record.lease or current.connection != record.connection
                        or current.session != record.session):
                    raise ResourceConflict("original local releasing custody changed")
                lease = await provider._original_release_receipt(scope, current, current.lease)
                if check is not None:
                    check()
                record = current
            except BaseException as settlement:
                raise execution_terminal_error(error, settlement, retain_originals=True)
        try:
            self._lease(record, lease, releasing=True)
            record.lease = lease
            if lease.state == "released":
                record.phase, record.connection = "released", None
            await self.repository.save(scope, record, check)
            if record.phase == "released":
                await self._discard_released_local_connection(record, check)
        except BaseException as settlement:
            raise execution_terminal_error(failure, settlement, retain_originals=True)
        return record, failure

    async def release(self, context, session_id):
        async with self.operation(context, VisualAction.RELEASE) as check:
            scope = self.scope(context)
            fence = self._key(scope, session_id)
            self._fence(scope, session_id)
            await self.egress.revoke_session(scope, session_id)
            async with self._lock(session_id):
                try:
                    record = await self.repository.get(scope, session_id=session_id, guard=check)
                except ResourceNotFound:
                    self._fences.discard(fence)
                    self._control_serials.pop(fence, None)
                    raise
                settled = await self._settle_failed_local_setup(scope, record, check)
                if settled is not None:
                    return self._retirement(settled) if settled.phase == "retired" else {"session_id": session_id, "released": True}
                if record.phase == "retired":
                    self._fences.discard(fence)
                    self._control_serials.pop(fence, None)
                    return self._retirement(record)
                if record.phase in {"admitted", "quoted"}:
                    record = await self.repository.retire_unacquired(scope, record, at=self._now(), guard=check)
                    self._fences.discard(fence)
                    self._control_serials.pop(fence, None)
                    return self._retirement(record)
                if record.lease is None:
                    raise ResourceOutcomeUnknown("unknown acquisition retains request cleanup custody")
                if record.phase == "released":
                    await self._discard_released_local_connection(record, check)
                    return {"session_id": session_id, "released": True}
                if record.remote_environment is not None:
                    provider = self._provider(record)
                    host = self._native_lifecycle_host(provider)
                    async with host._activate_release(context, record) as original:
                        lease = await self._call(context, check, provider.release, scope, record.lease)
                        self._lease(record, lease, releasing=True)
                        record = await self.repository._accepted_runpod_lifecycle(scope, original, lease)
                    return {"session_id": session_id, "released": False}
                if self._requires_remote_worker_admission(record):
                    raise ResourceUnsupported("Original private native quote required for foreground release; retained Pod custody is unresolved")
                await self._begin_release(scope, record, check)
                # Physical destruction has its own verb and outcome. A timeout
                # does not erase the original lease/connection or claim release.
                record, failure = await self._release_and_settle(scope, record, check, context=context)
                if failure is not None:
                    raise failure
                return {"session_id": session_id, "released": record.lease.state == "released"}

    async def fence_owned_sessions(self, scope):
        """Host lifecycle safety, without minting a user/delegated grant.

        This may only fence existing custody. It cannot observe, hand back,
        acquire, extend a lease, or destroy a process. Unknown control remains
        durable and is reported to lifecycle diagnostics.
        """
        from kestrel_sovereign._async_ownership import await_owned_operation
        if self.repository._key is None:
            if not await self.repository.db.table_exists("visual_computer_requests"):
                return []  # An unused capability never acquired a worker.
            await self.repository.ensure_initialized()
        uncertain = []
        for record in await self.repository.records(scope):
            self._fence(scope, record.session_id)
            fence = self._key(scope, record.session_id)
            serial = self._control_serials[fence]  # Capture before the first drain await.
            def current():
                if self._control_serials.get(fence) != serial:
                    raise ResourceConflict("original lifecycle pause was superseded")
            try:
                await self.egress.revoke_session(scope, record.session_id)
                # The sweep snapshot may predate a priority Stop/release.
                record = await self.repository.get(scope, session_id=record.session_id, guard=current)
                if record.session is None or record.phase == "released":
                    continue
                if record.phase == "releasing":
                    uncertain.append(record.session_id)
                    continue
                request = record.pending_control
                if request is not None and request.operation != "pause":
                    # Never replay uncertain handback/takeover as lifecycle stop.
                    uncertain.append(record.session_id)
                    continue
                request = request or ControlRequest(request_id=uuid4().hex, session_id=record.session_id,
                    expected_control_epoch=record.session.control_epoch, operation="pause")
                record.pending_control, record.phase, record.frame = request, "fenced", None
                await self.repository.save(scope, record, current)
                session, lease, connection, revision = record.session, record.lease, record.connection, record.revision
                def pause_current():
                    current()
                    self._require_native_session(record)
                    if record.phase != "fenced" or record.pending_control != request or request.operation != "pause":
                        raise ResourceConflict("original lifecycle pause intent changed")
                identifier = "lifecycle-pause:" + request.request_id
                # Only this existing Host-owned pause borrows a session call;
                # no OperatorContext, input grant or replacement lease is issued.
                async with asyncio.timeout(self.policy.operation_seconds):
                    async with self.egress._activate(session, identifier, pause_current,
                            deadline=self._now()+timedelta(seconds=self.policy.operation_seconds), purpose="session"):
                        async def dispatch():
                            async with self.egress._operation(session, identifier, purpose="session") as borrowed:
                                latest = await self.repository.get(scope, session_id=record.session_id, guard=borrowed)
                                if (latest.phase != "fenced" or latest.pending_control != request
                                        or latest.session != session or latest.lease != lease
                                        or latest.connection != connection or latest.revision != revision):
                                    raise ResourceConflict("original lifecycle custody changed before SDK entry")
                                return await self._driver(record).control(session, request)
                        call = self.egress._calls[self.egress._key(session, identifier, "session")]
                        outcome = await await_owned_operation(dispatch(), retain_originals=True,
                            cancel_on_caller_cancellation=call.cancel)
                        result = raise_driver_borrow_outcome(outcome, operation="original lifecycle pause", retain_originals=True)
                self._session(record, result)
                if result.controller != "none" or result.control_epoch <= request.expected_control_epoch:
                    raise ResourceOutcomeUnknown("lifecycle input fence requires reconciliation")
                record.session, record.pending_control = result, None
                await self.repository.save(scope, record, current)
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
            if record.phase in {"released", "retired"}:
                continue
            if record.lease is None:
                if record.phase in {"admitted", "quoted"} and record.request.expires_at <= self._now():
                    try:
                        await self.repository.retire_unacquired(scope, record, at=self._now())
                    except ResourceError:
                        uncertain.append(record.session_id)
                elif record.phase not in {"admitted", "quoted"}:
                    uncertain.append(record.session_id)
                continue
            last = record.last_activity_at or record.request.requested_at
            if self._requires_remote_worker_admission(record):
                # Housekeeping owns expired input-use retirement, never a new
                # current control-plane grant. In particular it must not revoke
                # an independently admitted foreground Inspect or RELEASE.
                if (record.phase == "releasing" or self._now() >= record.lease.expires_at
                        or (self._now()-last).total_seconds() >= record.request.idle_ttl_seconds):
                    if self._key(scope, record.session_id) not in self._fences:
                        self._fence(scope, record.session_id)
                    try:
                        await self.egress.revoke_session_use(scope, record.session_id)
                    except (ResourceError, TimeoutError, OSError):
                        pass  # This original remains uncertain below; continue other sessions.
                uncertain.append(record.session_id)
                continue
            if (record.phase != "releasing" and self._now() < record.lease.expires_at
                    and (self._now()-last).total_seconds() < record.request.idle_ttl_seconds):
                continue
            self._fence(scope, record.session_id)
            try:
                await self.egress.revoke_session(scope, record.session_id)
                async with self._lock(record.session_id):
                    # The earlier sweep snapshot may predate another original
                    # release. Decide only from this freshly read current row.
                    record = await self.repository.get(scope, session_id=record.session_id)
                    if record.phase in {"released", "retired"}:
                        continue
                    if record.lease is None or self._requires_remote_worker_admission(record):
                        uncertain.append(record.session_id)
                        continue
                    last = record.last_activity_at or record.request.requested_at
                    if (record.phase != "releasing" and self._now() < record.lease.expires_at
                            and (self._now()-last).total_seconds() < record.request.idle_ttl_seconds):
                        continue
                    # A failed setup may already have joined its physical
                    # closer while its durable settlement was unavailable.
                    # Reuse that same original join/ACK/discard boundary before
                    # the generic lease release path drops connection custody.
                    if await self._settle_failed_local_setup(scope, record, None) is not None:
                        continue
                    await self._begin_release(scope, record)
                    record, failure = await self._release_and_settle(scope, record, None)
                    if failure is not None and not isinstance(failure, (ResourceError, TimeoutError, OSError)):
                        raise failure
                    if record.phase != "released":
                        uncertain.append(record.session_id)
            except (ResourceError, TimeoutError, OSError):
                uncertain.append(record.session_id)
        return uncertain

    async def list_sessions(self, context):
        async with self.operation(context, VisualAction.STATUS):
            records = await self.repository.records(self.scope(context))
            return [{"session_id":record.session_id, "request_id":record.request.request_id,
                     "phase":record.phase, "controller":record.session.controller if record.session else "none"}
                    for record in records]
