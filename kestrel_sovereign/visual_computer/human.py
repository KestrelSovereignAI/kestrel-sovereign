"""Original-owner manual view/input; never registered as agent/model tools.

This lane requires the published SDK 0.45 contract and explicit host enablement.
Every HTTP call supplies its own authenticated grant; stored descriptors and prior
leases confer no authority.
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sdk.computer_use.human_control import (
    HumanActionReceipt, HumanActionRequest, HumanControlLease, HumanControlRequest,
    HumanControlRevocation, HumanControlRevocationTombstone, HumanScreenFrame,
)
from kestrel_sovereign.security.private_delivery import PrivateTransportBorrow

from .docker_human import human_operation_id
from .dispatch import original_human_dispatch, NativeActionNotDispatched
from .runtime import VisualAction
from .repository import ActionRecord


class HumanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    request: HumanControlRequest | None = None
    lease: HumanControlLease | None = None
    frame: HumanScreenFrame | None = None
    revocation: HumanControlRevocation | None = None
    tombstone: HumanControlRevocationTombstone | None = None

    def _require_successor(self, original):
        """Same-row coherence only; the enclosing operation supplies authority."""
        if type(self) is not HumanRecord or type(original) is not HumanRecord:
            raise ResourceConflict("original human custody schema required")
        if original.request is not None and self.request != original.request:
            raise ResourceConflict("accepted human request changed")
        if original.revocation is not None:
            if (self.revocation is None or
                    original.revocation.model_copy(update={"request_id": self.revocation.request_id}) != self.revocation):
                raise ResourceConflict("accepted human revocation changed")
            # A pre-open tombstone can acknowledge another request ID for the
            # same target. Accepted leases retain their original revocation.
            if (self.revocation != original.revocation
                    and (original.lease is not None or original.tombstone is None)):
                raise ResourceConflict("accepted human revocation changed")
        if original.tombstone is not None:
            if (self.tombstone is None or self.request != original.request or self.lease is not None
                    or self.tombstone.model_copy(update={"request": original.tombstone.request}) != original.tombstone
                    or self.tombstone.request.model_copy(update={"request_id": original.tombstone.request.request_id}) != original.tombstone.request):
                raise ResourceConflict("accepted human tombstone changed")
        if original.lease is not None and original.lease.state == "revoked":
            if (self.lease is None or self.lease.state != "revoked"
                    or self.lease.request != original.lease.request):
                raise ResourceConflict("retired human lease cannot become active")
        if (original.revocation is not None or original.tombstone is not None
                or (original.lease is not None and original.lease.state == "revoked")) and self.frame is not None:
            raise ResourceConflict("retired human frame cannot become available")


class HumanComputerRuntime:
    def __init__(self, runtime, *, lease_seconds=120):
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 120:
            raise ValueError("human lease lifetime must be an integer from 1 to 120 seconds")
        self.lease_seconds = lease_seconds
        self.runtime = runtime
        self.repository = runtime.repository

    async def _record(self, scope, sid, cid, *, check, accept=None, update=None):
        return await self.repository.private_record(scope, sid, "human", cid, HumanRecord,
            accept=accept, update=update, guard=check)

    def _ready(self, record):
        r, s = self.runtime, record.session
        if (s is None or s.state != "paused" or s.controller != "human" or record.phase != "fenced"
                or record.pending_control is not None or r._key(record.request.scope, record.session_id) not in r._verified_sessions
                or r._now() >= min(s.expires_at, record.lease.expires_at)):
            raise ResourceConflict("manual control requires a verified paused human session")

    def _extension(self, record):
        driver = self.runtime._driver(record)
        method = getattr(driver, "human_control", None)
        if not callable(method):
            raise ResourceUnsupported("driver has no human channel")
        extension = method()
        from kestrel_sdk.computer_use.human_control import HumanControl
        if not isinstance(extension, HumanControl):
            raise ResourceUnsupported("driver human contract is unavailable")
        return extension

    @asynccontextmanager
    async def delivery(self, context, sid, cid, *, metadata=False, revoke=False, input=False):
        r, scope = self.runtime, self.runtime.scope(context)
        action = VisualAction.HUMAN_REVOKE if revoke else VisualAction.HUMAN_VIEW
        async with AsyncExitStack() as stack:
            view_check = await stack.enter_async_context(r.operation(context, action))
            input_check = await stack.enter_async_context(r.operation(context, VisualAction.HUMAN_INPUT)) if input else lambda: None
            def check():
                view_check()
                input_check()
            serial = r._control_serials.get(r._key(scope, sid), 0)
            record = await self.repository.get(scope, session_id=sid, guard=check)
            try:
                original = await self._record(scope, sid, cid, check=check)
            except ResourceNotFound:
                original = None
            def current():
                check()
                if not revoke and serial != r._control_serials.get(r._key(scope, sid), 0):
                    raise ResourceConflict("manual control changed before private delivery")
                if not metadata and not revoke:
                    self._ready(record)
                    if original is not None:
                        if original.revocation is not None or original.tombstone is not None:
                            raise ResourceConflict("original human control is fenced")
                        if original.lease is None:
                            raise ResourceOutcomeUnknown("reconcile original human open")
                        original.lease.require_active(record.session, at=r._now())
            current()
            yield current
            current()

    @asynccontextmanager
    async def transport_delivery(self, context, sid, cid, check, *, metadata=False):
        """Retain the original HTTP authority through its actual outer send.

        Only trusted response construction supplies this callback. The child
        borrows the original checker; IDs or stored leases grant no authority.
        """
        r, scope = self.runtime, self.runtime.scope(context)
        serial = r._control_serials.get(r._key(scope, sid), 0)
        record = await self.repository.get(scope, session_id=sid, guard=check)
        r._require_native_session(record)
        stored = await self._record(scope, sid, cid, check=check)
        if stored.request is None:
            raise ResourceConflict("human delivery has no original channel")
        request = stored.request
        def current():
            check()
            if serial != r._control_serials.get(r._key(scope, sid), 0):
                raise ResourceConflict("human channel changed before transport delivery")
            if not metadata:
                self._ready(record)
                if stored.lease is None or stored.revocation is not None or stored.tombstone is not None:
                    raise ResourceConflict("human delivery channel is fenced")
                stored.lease.require_active(record.session, at=r._now())
        identifier = json.dumps([cid, request.consumer_id, request.control_epoch, uuid4().hex], separators=(",", ":"))
        deadline = min(context.expires_at, r._now()+timedelta(seconds=r.policy.operation_seconds))
        if not metadata:
            deadline = min(deadline, request.expires_at, record.session.expires_at)
        async with r.egress.activate_human_delivery(record.session, identifier, current,
                deadline=deadline, metadata=metadata):
            yield PrivateTransportBorrow(
                operation=lambda: r.egress.human_delivery_operation(record.session, identifier, metadata=metadata),
                cancel=lambda task: r.egress.cancel_human_delivery(record.session, identifier, task, metadata=metadata))

    @asynccontextmanager
    async def _operation(self, context, sid, cid, *, metadata=False, action=VisualAction.HUMAN_VIEW):
        r, scope = self.runtime, self.runtime.scope(context)
        async with r.operation(context, action) as check:
            serial = r._control_serials.get(r._key(scope, sid), 0)
            record = await self.repository.get(scope, session_id=sid, guard=check)
            stored = await self._record(scope, sid, cid, check=check)
            if stored.request is None or stored.tombstone is not None:
                raise ResourceConflict("original human control ID is fenced")
            def current():
                check()
                if serial != r._control_serials.get(r._key(scope, sid), 0):
                    raise ResourceConflict("human control epoch changed")
                if not metadata:
                    if stored.revocation is not None:
                        raise ResourceConflict("original human control is fenced")
                    self._ready(record)
                    if stored.lease is None:
                        raise ResourceOutcomeUnknown("reconcile original human open")
                    stored.lease.require_active(record.session, at=r._now())
            current()
            yield record, stored, current
            current()

    async def _call(self, context, record, cid, operation, check, method, *args, action_id=None, deadline=None, _origin=None):
        r = self.runtime
        identifier = human_operation_id(cid, operation, action_id)
        metadata = operation in {"status", "action_status", "revoke"}
        r._require_native_session(record)
        serial = r._control_serials.get(r._key(record.request.scope, record.session_id), 0)
        def current():
            check()
            if serial != r._control_serials.get(r._key(record.request.scope, record.session_id), 0):
                raise ResourceConflict("original human native call was fenced")
        latest = await self.repository.get(record.request.scope, session_id=record.session_id, guard=current)
        r._require_native_session(latest)
        if latest.session != record.session or latest.lease != record.lease:
            raise ResourceConflict("original human session changed before SDK entry")
        limit = min(context.expires_at, r._now()+timedelta(seconds=r.policy.operation_seconds))
        if not metadata:
            limit = min(limit, record.session.expires_at, deadline or record.session.expires_at)
        async with r.egress.activate_human(record.session, identifier, current, deadline=limit, metadata=metadata) as call:
            async def invoke():
                if _origin is not None:
                    _origin.enter_driver()
                return await method(record.session, *args)
            return await call.invoke_sdk(invoke)

    async def open(self, context, sid, cid):
        from kestrel_sdk.resources import Identifier
        from pydantic import TypeAdapter
        cid = TypeAdapter(Identifier, config=ConfigDict(hide_input_in_errors=True)).validate_python(cid)
        r, scope = self.runtime, self.runtime.scope(context)
        async with r.operation(context, VisualAction.HUMAN_VIEW) as check:
            async with r._lock(sid):
                record = await self.repository.get(scope, session_id=sid, guard=check)
                self._ready(record)
                if record.human_control_epoch == record.session.control_epoch and record.human_control_id != cid:
                    raise ResourceConflict("a fresh takeover epoch is required for a new human consumer")
                now = r._now()
                expires = min(record.session.expires_at, now+timedelta(seconds=self.lease_seconds))
                frame_age = min(15, int((expires-now).total_seconds()))
                if frame_age < 1:
                    raise ResourceConflict("insufficient original session lifetime for human control")
                request = HumanControlRequest(control_id=cid, scope=scope, session_id=sid,
                    lease_id=record.session.lease_id, driver_id=record.session.driver_id,
                    consumer_id=self.repository._address("human-consumer", scope.model_dump_json(), sid, cid),
                    control_epoch=record.session.control_epoch, issued_at=now,
                    expires_at=expires,
                    capabilities=(frozenset({"screenshot", "click", "type", "key", "scroll"}) if VisualAction.HUMAN_INPUT.value in context.granted_actions else frozenset({"screenshot"})) & record.session.capabilities,
                    max_frame_age_seconds=frame_age)
                stored = await self._record(scope, sid, cid, check=check, accept=HumanRecord(request=request))
                if stored.request is None or stored.tombstone is not None or stored.revocation is not None:
                    raise ResourceConflict("original human control ID is fenced")
                request = stored.request
                request.require_session(record.session, at=r._now())
                if stored.lease is not None:
                    stored.lease.require_active(record.session, at=r._now())
                    return stored.lease
                record.human_control_id, record.human_control_epoch = cid, record.session.control_epoch
                record.frame, record.human_private = None, True
                await self.repository.save(scope, record, check)
            serial = r._control_serials.get(r._key(scope, sid), 0)
            def current():
                check()
                self._ready(record)
                request.require_session(record.session, at=r._now())
                if serial != r._control_serials.get(r._key(scope, sid), 0):
                    raise ResourceConflict("human open fenced before dispatch")
            lease = await self._call(context, record, cid, "open", current, self._extension(record).open, request, deadline=request.expires_at)
            current()
            now = r._now()
            request.require_session(record.session, at=now)
            try:
                lease.require_request(request)
                lease.require_active(record.session, at=now)
            except ValueError:
                raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
            stored.lease = lease
            await self._record(scope, sid, cid, check=check, update=stored)
            return lease

    async def status(self, context, sid, cid):
        async with self._operation(context, sid, cid, metadata=True) as (record, stored, check):
            lease = await self._call(context, record, cid, "status", check, self._extension(record).status, stored.request)
            try:
                lease.require_request(stored.request)
            except ValueError:
                raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
            stored = await self._record(self.runtime.scope(context), sid, cid, check=check)
            if stored.revocation is not None and lease.state == "active":
                raise ResourceOutcomeUnknown("human retirement requires original-ID reconciliation")
            if stored.lease is not None and stored.lease.state == "revoked" and lease.state != "revoked":
                raise ResourceOutcomeUnknown("retired human lease cannot become active")
            stored.lease = lease
            await self._record(self.runtime.scope(context), sid, cid, check=check, update=stored)
            return lease

    async def observe(self, context, sid, cid):
        async with self._operation(context, sid, cid) as (record, stored, check):
            value = await self._call(context, record, cid, "observe", check, self._extension(record).observe,
                stored.lease, deadline=stored.request.expires_at)
            now = self.runtime._now()
            # Original currentness is not a malformed peer response.
            check()
            stored.lease.require_active(record.session, at=now)
            try:
                value.require_lease(stored.lease, record.session, at=now)
            except ValueError:
                raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
            stored.frame = value.frame_record
            await self._record(self.runtime.scope(context), sid, cid, check=check, update=stored)
            return value

    async def act(self, context, sid, cid, request):
        request = HumanActionRequest.model_validate(request)
        r, scope = self.runtime, self.runtime.scope(context)
        # An input grant alone never exposes a frame; retain BOTH independent
        # original current grants through native dispatch and private delivery.
        async with self._operation(context, sid, cid, action=VisualAction.HUMAN_INPUT) as (record, stored, input_check):
            async with r.operation(context, VisualAction.HUMAN_VIEW) as view_check:
                def check():
                    input_check()
                    view_check()
                prepared = dispatched = inflight = False
                result_stack = AsyncExitStack()
                result_owner = original = None
                try:
                    async with r._lock(sid):
                        try:
                            previous = await self.repository.action(scope, sid, request.action.action_id, guard=check)
                        except ResourceNotFound:
                            previous = None
                        if previous is not None:
                            if (type(previous) is not ActionRecord or previous.channel != "human"
                                    or previous.control_id != cid or previous.request != request.action):
                                raise ResourceConflict("original input ID identifies another channel or input")
                            if not previous.dispatch_started and previous.receipt is None:
                                previous = await self.repository.action(scope, sid, request.action.action_id,
                                    abandon_prepared=True, guard=check)
                            if previous.receipt is None or previous.receipt.outcome == "unknown":
                                raise ResourceOutcomeUnknown("reconcile original manual input without replay")
                            result = HumanActionReceipt(control_id=cid, consumer_id=stored.request.consumer_id, receipt=previous.receipt)
                            result.require_request(request)
                            return result
                        latest = await self._record(scope, sid, cid, check=check)
                        stored.frame = latest.frame
                        if stored.frame is None:
                            raise ResourceConflict("view the current human screen before input")
                        request.require_lease(stored.lease, record.session, stored.frame, at=r._now())
                        if request.action.not_after > context.expires_at:
                            raise ResourceConflict("manual input exceeds original grant deadline")
                        if await self.repository.has_pending_actions(scope, sid, guard=check):
                            raise ResourceOutcomeUnknown("reconcile accepted input before another action")
                        if r._inflight >= r.policy.max_inflight_actions:
                            raise ResourceConflict("normal visual input capacity exhausted")
                        r._inflight += 1
                        inflight = True
                        await self.repository.action(scope, sid, request.action.action_id, accept=request.action,
                            channel="human", control_id=cid, prepare=True, guard=check)
                        prepared = True
                        original = result_stack.enter_context(original_human_dispatch(
                            record.session, stored.lease, request, stored.frame))
                        result_owner = await result_stack.enter_async_context(
                            self.repository.original_human_action_result(scope, record.session, request, guard=check))
                        stored.frame = None
                        await self._record(scope, sid, cid, check=check, update=stored)
                        await self.repository.action(scope, sid, request.action.action_id,
                            dispatch_revision=record.revision, guard=check, _result_owner=result_owner)
                        dispatched = True
                    def effect_check():
                        from kestrel_sovereign.execution_custody import require_execution_work
                        require_execution_work(self.repository.db.backend)
                        check()
                        if r._now() >= request.action.not_after:
                            raise ResourceConflict("original human action deadline expired")
                    async def dispatch(session, lease, human_request):
                        current = await self.repository.get(scope, session_id=sid, guard=effect_check)
                        if current.revision != record.revision or current.session != record.session or current.lease != record.lease:
                            raise ResourceConflict("manual session changed before native dispatch")
                        effect_check()
                        try:
                            return await self._call(context, record, cid, "act", effect_check,
                                self._extension(record).act, lease, human_request,
                                action_id=request.action.action_id, deadline=request.action.not_after, _origin=original)
                        except BaseException as failure:
                            # Caller cancellation can enclose the SDK's exact
                            # negative result after its owned Task has joined.
                            # Only the original native trace can supply this
                            # fact; an arbitrary refusal remains unknown.
                            fact = (failure if type(failure) is NativeActionNotDispatched
                                else original.trace.proof if original.trace is not None else None)
                            if fact is None:
                                raise
                            original.require(fact)
                            primary = fact.original if failure is fact else failure
                            try:
                                await result_owner.not_dispatched(original, fact)
                            except BaseException as settlement:
                                from kestrel_sovereign.execution_custody import execution_observation_error
                                raise execution_observation_error(primary, settlement) from None
                            if failure is fact:
                                raise fact.original
                            raise
                    receipt = await dispatch(record.session, stored.lease, request)
                    try:
                        receipt.require_request(request)
                    except ValueError:
                        raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
                    await self.repository.action(scope, sid, request.action.action_id, receipt=receipt.receipt, guard=check)
                    return receipt
                except BaseException as failure:
                    if result_owner is not None and not original.driver_entered:
                        try:
                            await result_owner.not_dispatched(original, original.refuse_unentered(failure))
                        except BaseException as settlement:
                            from kestrel_sovereign.execution_custody import execution_observation_error
                            raise execution_observation_error(failure, settlement) from None
                    elif prepared and not dispatched:
                        await r._abandon_prepared_action(scope, request.action)
                    raise
                finally:
                    if inflight:
                        r._inflight -= 1
                    await result_stack.aclose()

    async def action_status(self, context, sid, cid, action_id):
        r, scope = self.runtime, self.runtime.scope(context)
        async with self._operation(context, sid, cid, metadata=True) as (record, stored, check):
            accepted = await self.repository.action(scope, sid, action_id, guard=check)
            if type(accepted) is not ActionRecord or accepted.channel != "human" or accepted.control_id != cid:
                raise ResourceNotFound("manual input not found")
            request = HumanActionRequest(control_id=cid, consumer_id=stored.request.consumer_id, action=accepted.request)
            if not accepted.dispatch_started and accepted.receipt is None:
                accepted = await self.repository.action(scope, sid, action_id, abandon_prepared=True, guard=check)
            if accepted.receipt is not None and accepted.receipt.outcome != "unknown":
                return HumanActionReceipt(control_id=cid, consumer_id=stored.request.consumer_id, receipt=accepted.receipt)
            if stored.lease is None:
                raise ResourceOutcomeUnknown("original human open requires reconciliation")
            try:
                receipt = await self._call(context, record, cid, "action_status", check, self._extension(record).action_status,
                    stored.lease, action_id, action_id=action_id)
            except ResourceNotFound:
                raise ResourceOutcomeUnknown("original manual input is unknown to this worker") from None
            try:
                receipt.require_request(request)
            except ValueError:
                raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
            await self.repository.action(scope, sid, action_id, receipt=receipt.receipt, guard=check)
            return receipt

    async def revoke(self, context, sid, cid, request_id, epoch):
        r, scope = self.runtime, self.runtime.scope(context)
        async with r.operation(context, VisualAction.HUMAN_REVOKE) as check:
            key = r._key(scope, sid)
            serial = r._control_serials.get(key, 0)
            def current():
                check()
                if serial != r._control_serials.get(key, 0):
                    raise ResourceConflict("original human revocation was superseded")
            consumer = self.repository._address("human-consumer", scope.model_dump_json(), sid, cid)
            request = HumanControlRevocation(request_id=request_id, control_id=cid, session_id=sid,
                consumer_id=consumer, control_epoch=epoch)
            record = await self.repository.get(scope, session_id=sid, guard=current)
            r._require_native_session(record)
            try:
                stored = await self._record(scope, sid, cid, check=current)
            except ResourceNotFound:
                self._current_revoke_target(record, None, request)
                stored = await self._record(scope, sid, cid, check=current, accept=HumanRecord(revocation=request))
            if stored.request is not None and (stored.request.control_epoch != epoch
                    or stored.request.consumer_id != consumer
                    or (stored.request.scope, stored.request.session_id, stored.request.lease_id, stored.request.driver_id)
                    != (scope, sid, record.session.lease_id, record.session.driver_id)):
                raise ResourceConflict("human revocation differs from the original accepted target")
            if stored.revocation is not None and stored.revocation != request:
                if (stored.lease is not None or stored.tombstone is None
                        or stored.revocation.model_copy(update={"request_id":request.request_id}) != request):
                    raise ResourceConflict("original human revocation target differs")
            current()
            if stored.tombstone is not None and stored.tombstone.request == request:
                stored.tombstone.require_revocation(request, record.session)
                return stored.tombstone
            if stored.lease is not None and stored.lease.state == "revoked" and stored.revocation == request:
                request.require_target(stored.lease, record.session)
                if stored.request is not None:
                    stored.lease.require_request(stored.request)
                return stored.lease
            # Confirming a recorded pre-open fence needs a genuine new-ID RPC,
            # but that original target cannot revoke a later agent's session.
            drain = stored.tombstone is None
            if drain:
                self._current_revoke_target(record, stored, request)
                current()
                r._fence(scope, sid)
                serial = r._control_serials[key]
                await r.egress.revoke_session(scope, sid)
            stored.revocation, stored.frame = request, None
            await self._record(scope, sid, cid, check=current, update=stored)
            return await self._retire(context, record, stored, check, serial=serial, drain=drain)

    @staticmethod
    def _current_revoke_target(record, stored, request):
        session = record.session
        if (session is None or session.controller != "human" or session.state != "paused"
                or record.phase != "fenced" or record.pending_control is not None
                or request.control_epoch != session.control_epoch):
            raise ResourceConflict("human revocation requires its original current target or recorded fence")
        # Never-opened IDs remain fenceable within the current Human epoch. An
        # accepted channel must match the published consumer, except a prepared
        # request whose channel publication has not yet completed.
        if stored is not None and stored.request is not None:
            if record.human_control_id != request.control_id or record.human_control_epoch != request.control_epoch:
                if stored.lease is not None or record.human_control_id is not None:
                    raise ResourceConflict("human revocation differs from the current consumer")

    async def _retire(self, context, record, stored, check, *, serial, drain=True):
        r = self.runtime
        request = stored.revocation
        key = r._key(record.request.scope, record.session_id)
        def current():
            check()
            if serial != r._control_serials.get(key, 0):
                raise ResourceConflict("original human retirement was superseded")
        # Both callers have acknowledged the original pending revocation row.
        # Fence/drain again at that publication boundary: earlier snapshots
        # cannot adopt its new serial or register delivery after the first drain.
        current()
        if drain:
            r._fence(record.request.scope, record.session_id)
            serial = r._control_serials[key]
            await r.egress.revoke_session(record.request.scope, record.session_id)
        value = await self._call(context, record, request.control_id, "revoke", current, self._extension(record).revoke, request)
        current()
        if isinstance(value, HumanControlRevocationTombstone):
            if stored.lease is not None:
                raise ResourceOutcomeUnknown("accepted human lease cannot become a pre-open tombstone")
            try:
                value.require_revocation(request, record.session)
            except ValueError:
                raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
            stored.tombstone = value
        else:
            try:
                request.require_target(value, record.session)
                if stored.request is not None:
                    value.require_request(stored.request)
            except ValueError:
                raise ResourceOutcomeUnknown("invalid_private_worker_response") from None
            if value.state != "revoked":
                raise ResourceOutcomeUnknown("manual input retirement remains unverified")
            stored.lease = value
        stored.frame = None
        await self._record(record.request.scope, record.session_id, request.control_id, check=current, update=stored)
        return value

    async def retire_owned(self, context, record, check):
        """Existing runtime safety custody only; never opens or reads a screen."""
        if record.human_control_id is None:
            return
        r = self.runtime
        key = r._key(record.request.scope, record.session_id)
        serial = r._control_serials.get(key, 0)
        def current():
            check()
            if serial != r._control_serials.get(key, 0):
                raise ResourceConflict("original owned human retirement was superseded")
        stored = await self._record(record.request.scope, record.session_id, record.human_control_id, check=current)
        if stored.tombstone is not None or (stored.lease is not None and stored.lease.state == "revoked"):
            return
        if stored.revocation is None:
            stored.revocation = HumanControlRevocation(request_id=uuid4().hex,
                control_id=stored.request.control_id, session_id=record.session_id,
                consumer_id=stored.request.consumer_id, control_epoch=stored.request.control_epoch)
        stored.frame = None
        await self._record(record.request.scope, record.session_id, record.human_control_id, check=current, update=stored)
        await self._retire(context, record, stored, check, serial=serial)
