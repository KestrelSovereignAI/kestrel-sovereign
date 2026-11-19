"""Native Task host custody inside the original governed MCP handler.

The feature validates native protocol results. Core proves the selected tool,
captures original arguments, and owns durable Task/Input ordering. Neither a
Task handle nor any of these private descriptions authorizes an execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
import hashlib
import json
from uuid import UUID
from weakref import WeakKeyDictionary

from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kestrel_sovereign.a2a.media import encode_inline_json
from kestrel_sovereign.agent.tool_registry import _GovernedToolBinding, _ToolExecution
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.features.security.hooks import reserved_session_id
from kestrel_sovereign.mcp_input import MCPInputHost
from kestrel_sovereign.mcp_input_store import MCPInputConflict
from kestrel_sovereign.mcp_task_input import MCPTaskInputCustody
from kestrel_sovereign.mcp_task_store import MCPTaskBinding, MCPTaskConflict, MCPTaskStore, task_routing_payload
from kestrel_sovereign.mcp_task_submission import (
    MCPTaskDelivery, MCPTaskInteractiveOrigin, MCPTaskServiceOrigin, MCPTaskSubmission,
)
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied, require_raw_capability_disclosure
from kestrel_sovereign.tools.effect_invocation import current_service_operation_attribution
from kestrel_sovereign.tools.effect_journal import current_tool_effect_execution, require_persistent_tool_effects
from kestrel_sovereign.storage.db.interface import DatabaseError
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate


class MCPTaskHostPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True,
        allow_inf_nan=False, hide_input_in_errors=True)
    max_active: int = Field(default=64, gt=0, le=4096)
    max_records: int = Field(default=4096, gt=0, le=65536)
    max_submission_bytes: int = Field(default=1_048_576, gt=0, le=16_777_216)
    max_native_claim_bytes: int = Field(default=1_048_576, gt=0, le=16_777_216)
    max_continuations: int = Field(default=16, ge=0, le=256)
    observation_reserve_bytes: int = Field(default=1_048_576, gt=0, le=16_777_216)
    max_held_tools: int = Field(default=16, gt=0)
    max_payload_bytes: int = Field(default=4_194_304, gt=0, le=67_108_864)

    @model_validator(mode='after')
    def budgets(self):
        if (self.max_active > self.max_records or self.max_payload_bytes <=
                2 * self.max_native_claim_bytes + self.observation_reserve_bytes):
            raise ValueError('Task custody requires retained-record and native-result headroom')
        return self


@dataclass(eq=False, repr=False)
class _RetainedOwner:
    hold: object
    binding: MCPTaskBinding
    writes: int = 0

    @contextmanager
    def writing(self):
        """Separate an actual in-flight writer from an uncertain completed one."""
        self.writes += 1
        try:
            yield
        finally:
            self.writes -= 1


class MCPTaskHost:
    def __init__(self, agent, feature, input_host, *, configuration, is_current):
        if (type(input_host) is not MCPInputHost or input_host.agent is not agent
                or input_host.feature is not feature or not callable(is_current)):
            raise TypeError('Task custody requires its actual agent, feature and Input host')
        self.agent, self.feature, self.input_host = agent, feature, input_host
        self._agent_id, self._is_current = agent.did, is_current
        self.policy = MCPTaskHostPolicy.model_validate(configuration)
        if self.policy.max_held_tools >= agent.MAX_DIRECT_TOOLS:
            raise ValueError('Task owner retention requires direct-tool registry headroom')
        database = require_persistent_tool_effects(agent).database
        if input_host.store.database is not database:
            raise ResourceConflict('Task and Input custody must borrow the same original database')
        self.store = MCPTaskStore(database, agent_did=agent.did, max_active=self.policy.max_active,
            max_records=self.policy.max_records, max_payload_bytes=self.policy.max_payload_bytes,
            max_submission_bytes=self.policy.max_submission_bytes)
        self.joint = None
        self._rounds = WeakKeyDictionary()
        self._owners = set()
        self._unreconciled = set()
        self._accepting, self.retired = True, False

    def require_prepared(self):
        self.input_host.require_prepared()
        if (not self._accepting or self.retired or self._agent_id != self.agent.did
                or self._is_current() is not True
                or require_persistent_tool_effects(self.agent).database is not self.store.database
                or self.input_host.store.database is not self.store.database):
            raise HostCapabilityAccessDenied('Original MCP Task host changed')

    def require_current(self):
        self.require_prepared()
        self.input_host.require_current()
        require_raw_capability_disclosure(self.agent, 'mcp-tasks')

    async def initialize(self):
        self.require_prepared()
        await self.store.initialize()
        self.require_prepared()
        self.joint = MCPTaskInputCustody(self.store, self.input_host.store)

    def _execution(self, binding, *, require_registration=True):
        self.require_current()
        frame = current_tool_effect_execution(self.agent)
        witness = frame.witness
        if (type(binding) is not _GovernedToolBinding or binding.agent is not self.agent
                or type(witness) is not _ToolExecution or not witness.started or witness.returned
                or witness.native_action is not None or witness.registration is not binding.registration
                or witness.tool is not binding.tool or witness.tool_name != binding.name
                or witness.owner != binding.owner):
            raise ResourceConflict('Task admission requires the original selected tool handler')
        if require_registration:
            binding.require_current()
        if self.joint is None:
            raise ResourceConflict('Original MCP Task custody is not initialized')
        return frame

    def current_round(self):
        frame = current_tool_effect_execution(self.agent)
        result = self._rounds.get(frame)
        if result is not None:
            result._require()
        return result

    def _attribution(self, origin):
        if origin.allow_blocking:
            return MCPTaskInteractiveOrigin(origin.principal_id, origin.session_id)
        kind, digest = current_service_operation_attribution()
        session = origin.session_id
        if session is not None and (not session.strip() or reserved_session_id(session) is not None):
            session = None
        return MCPTaskServiceOrigin(kind, digest, session)

    def _delivery(self, origin, view_label):
        if isinstance(origin, MCPTaskInteractiveOrigin):
            for host in getattr(self.agent, '_mcp_apps_hosts', ()):
                if host.current_call_session_id() == origin.session_id:
                    reader = host._reader.get()
                    for identifier, physical in host._views.items():
                        if physical is reader.physical and view_label in (None, identifier):
                            return MCPTaskDelivery('app', origin.session_id, identifier)
        if view_label is not None:
            raise ResourceConflict('Task App delivery requires its original physical View')
        return (MCPTaskDelivery('chat', origin.session_id) if origin.session_id is not None
            else MCPTaskDelivery('internal'))

    def _headroom(self, binding):
        # Core effect claims and Input request/reservation IDs are UUIDs. These
        # are size samples only, never a stored identity or execution credential.
        width = len(str(UUID(int=0)))
        identifier = '0' * width
        continuity = {'operations': [identifier] * self.policy.max_continuations,
            'input': dict(request_id=identifier, reservation_id=identifier, operation_id=identifier)}
        overhead = len(task_routing_payload(binding, None, {'version': 1, 'native': {}}, continuity))
        required = (overhead + 2 * self.policy.max_native_claim_bytes
            + self.policy.observation_reserve_bytes)
        if required > self.policy.max_payload_bytes:
            raise ResourceConflict('Task custody cannot retain this binding and its future native history')

    def _hold(self, binding, original):
        existing = next((owner for owner in self._owners if owner.hold.handle is not None
            and owner.binding == original and owner.hold.registration is binding.registration
            and not owner.hold.released), None)
        if existing is not None:
            return existing
        hold = self.agent._acquire_task_owner_hold(binding, max_holds=self.policy.max_active,
            max_held_tools=self.policy.max_held_tools)
        owner = _RetainedOwner(hold, original)
        self._owners.add(owner)
        return owner

    def _release(self, owner):
        self.agent._release_task_owner_hold(owner.hold)
        self._owners.discard(owner)
        self._unreconciled.discard(owner)

    async def _reconcile_reservations(self):
        for owner in tuple(self._unreconciled):
            if owner.writes:
                continue
            actual = await self.store.lookup(owner.binding)
            self.require_prepared()
            if owner.writes:
                continue
            if actual is None:
                self._release(owner)
            else:
                owner.hold.handle = actual.handle
                self._unreconciled.discard(owner)
                if actual.phase in {'SETTLED', 'NOT_SENT', 'INPUT_CLOSED'}:
                    self._release(owner)

    async def admit(self, binding, *, connection_name, authorization_id, contract, view_label=None):
        frame = self._execution(binding)
        if frame in self._rounds:
            raise ResourceConflict('This original handler already owns its Task admission')
        if (not isinstance(connection_name, str) or not connection_name or len(connection_name) > 1024
                or type(contract) is not bytes):
            raise ResourceConflict('Task admission requires its original connection and contract bytes')
        origin = self.input_host.origin()
        attribution = self._attribution(origin)
        connection_id = hashlib.sha256(encode_inline_json(
            ['mcp-task-connection-v1', self._agent_id, binding.owner, connection_name])).hexdigest()
        contract_digest = hashlib.sha256(contract).hexdigest()
        expected = MCPTaskBinding(self._agent_id, attribution, frame.claim.claim_id,
            connection_id, authorization_id, binding.name, contract_digest)
        self._headroom(expected)
        submission = MCPTaskSubmission.seal(arguments=json.loads(frame.arguments_json),
            contract=contract, delivery=self._delivery(attribution, view_label))
        if len(submission.canonical_json) > self.policy.max_submission_bytes:
            raise ResourceConflict('Original Task submission exceeded its admission budget')
        await self._reconcile_reservations()
        self._execution(binding)
        origin.require_current()
        reader = self.input_host._reader.get()
        record = reader.record if reader is not None else None
        issuance, original = None, expected
        if record is not None:
            self.input_host._require_reader()
            if record.binding.custody is None:
                raise ResourceConflict('Tasks cannot adopt an unlinked original Input call')
            records = await self.store.list_retained()
            actual = next((row for row in records if row.handle == record.binding.custody), None)
            if actual is None:
                raise ResourceConflict('Original Input Task custody is unavailable')
            sealed = await self.store.open_submission(actual)
            original = actual.binding
            self.joint._waiting_link(actual, record)
            if (replace(expected, operation_id=original.operation_id) != original
                    or record.binding.connection_id != connection_name
                    or encode_inline_json(sealed.arguments, sort_keys=True) != frame.arguments_json
                    or sealed.contract != contract
                    or len(actual.continuity['operations']) >= self.policy.max_continuations):
                raise ResourceConflict('Original Task submission or continuation lifetime changed')
            self._execution(binding)
            origin.require_current()
            owner = self._hold(binding, original)
            owner.hold.handle = actual.handle
        else:
            owner = self._hold(binding, original)
            # Park before the await: a cancellation or lost commit ACK must keep
            # counted retention until authenticated read-only reconciliation.
            self._unreconciled.add(owner)
            with owner.writing():
                try:
                    reservation = await self.store.reserve(expected, submission)
                except (MCPTaskConflict, ValueError) as error:
                    self._release(owner)
                    raise ResourceConflict('Original Task reservation was refused') from error
            owner.hold.handle = reservation.record.handle
            self._unreconciled.discard(owner)
            if not reservation.fresh:
                raise ResourceConflict('Original Task issuance cannot be recovered or replayed')
            issuance = reservation.issuance
        result = MCPTaskRound(self, binding, origin, connection_name, original,
            owner, issuance, record)
        self._rounds[frame] = result
        try:
            self._execution(binding)
            origin.require_current()
        except ResourceConflict:
            await result.refuse()
            raise
        return result

    async def maintain(self):
        async with require_durable_persistence_gate(self.agent).shared():
            await self._maintain()

    async def _maintain(self):
        self.require_current()
        await self._reconcile_reservations()
        records = await self.store.list_retained()
        self.require_current()
        for actual in records:
            if actual.phase in {'SETTLED', 'NOT_SENT', 'INPUT_CLOSED'}:
                # Lost ACKs may leave a retained owner after its real durable
                # settlement. The normal publication lease is separate and is
                # no longer in _owners, so maintenance cannot retire it early.
                for owner in tuple(self._owners):
                    if owner.hold.handle == actual.handle and not owner.writes:
                        self._release(owner)
            if actual.phase != 'INPUT_WAITING':
                continue
            origin, link = actual.binding.origin, actual.continuity['input']
            row = await self.input_host.store.read_for(origin.principal_digest, origin.session_id, link['request_id'])
            now = self.input_host.authority.clock().timestamp()
            if row.expires_at <= now:
                try:
                    await self.joint.close_expired(actual, now=now)
                except (MCPTaskConflict, MCPInputConflict):
                    current = await self.store.read(actual.binding, actual.handle)
                    self.require_current()
                    if current.revision == actual.revision:
                        raise
                    # Authenticate the changed revision before treating this
                    # as another responder's successful CAS, not corruption.
                    continue
                self.require_current()
                for owner in tuple(self._owners):
                    if owner.hold.handle == actual.handle:
                        self._release(owner)

    def withdraw(self):
        self._accepting = False

    async def close(self):
        self.withdraw()
        for owner in tuple(self._owners):
            self._release(owner)
        self._rounds.clear()
        await self.store.close()
        self.retired = True


class MCPTaskRound:
    def __init__(self, host, binding, origin, connection_name, original, owner, issuance, predecessor):
        self.host, self.binding, self.origin = host, binding, origin
        self.connection_name, self.original, self.owner = connection_name, original, owner
        self.issuance, self.predecessor = issuance, predecessor
        self._wire, self._closed = False, False
        self._suspended = None
        self.response_claim = None
        self._resume_uncertain = False

    def _require(self, *, local_refusal=False):
        try:
            frame = self.host._execution(self.binding, require_registration=not local_refusal)
            if self.host._rounds.get(frame) is not self or self._closed:
                raise ResourceConflict('Original Task handler round is unavailable')
            self.origin.require_current()
        except ResourceConflict as error:
            if self._wire:
                raise ResourceOutcomeUnknown('Original Task authority changed after possible dispatch') from error
            raise
        return frame

    @property
    def arguments(self):
        """Detached decided arguments, never a Feature checkpoint replacement."""
        return json.loads(self._require().arguments_json)

    @property
    def authorization_identity(self):
        self._require()
        return self.original.authorization_id

    def require_wire(self):
        self._require()
        if self.issuance is None or self.issuance.record.phase != 'ISSUING':
            raise ResourceConflict('Original Task issuance must be durable before wire dispatch')

    def start_wire(self):
        self.require_wire()
        if self._wire:
            raise ResourceConflict('An original Task round cannot dispatch twice')
        self._wire = True

    @property
    def dispatch_started(self):
        """Lifecycle observation only; it cannot authorize wire or settlement."""
        return self._wire

    def unknown(self):
        self._require()
        self._closed = True

    def _settled(self, actual, *, close=True):
        frame = self._require(local_refusal=actual.phase == 'NOT_SENT')
        if actual.phase in {'SETTLED', 'NOT_SENT'}:
            if frame.witness.owner_lease is not None:
                raise ResourceOutcomeUnknown('Original dispatch already retains another owner lease')
            frame.witness.owner_lease = self.owner.hold
            self.host._owners.discard(self.owner)
        reader = self.host.input_host._reader.get()
        if self.predecessor is not None:
            if reader is None or reader.record is not self.predecessor:
                raise ResourceOutcomeUnknown('Original Input Task reader changed after custody commit')
            reader.task_settled = True
        self._closed = close

    def finish_suspension(self, reference):
        self._require()
        if self._suspended is None or self._suspended.request_id != reference:
            raise ResourceOutcomeUnknown('Original native Input suspension was not acknowledged')
        self._closed = True

    def _native(self, native):
        self._require()
        if not self._wire or type(native) is not dict:
            raise ResourceOutcomeUnknown('Task outcome requires its actual original wire result')
        try:
            raw = encode_inline_json(native, sort_keys=True)
        except (ValueError, TypeError, RecursionError) as error:
            raise ResourceOutcomeUnknown('Original native Task result is not retainable') from error
        if len(raw) > self.host.policy.max_native_claim_bytes:
            raise ResourceOutcomeUnknown('Native Task result exceeded its reserved claim budget')
        return json.loads(raw)

    async def settle_result(self, native):
        native = self._native(native)
        if native.get('resultType') != 'complete':
            raise ResourceOutcomeUnknown('Original Task call requires an actual native complete result')
        await self._finish(lambda: self.host.store.settle_initial(self.issuance, disposition='SETTLED',
            envelope={'version': 1, 'native': native}), phase='SETTLED', native=native)

    async def attach(self, native):
        native = self._native(native)
        actual = await self._finish(lambda: self.host.store.attach_task(self.issuance,
            remote_task_id=native.get('taskId'), envelope={'version': 1, 'native': native}),
            phase='ATTACHED', native=native)
        return actual.handle

    async def _finish(self, write, *, phase, native):
        with self.owner.writing():
            return await self._finish_write(write, phase=phase, native=native)

    async def _finish_write(self, write, *, phase, native):
        try:
            actual = await write()
        except (DatabaseError, OSError, ValueError, ResourceOutcomeUnknown) as error:
            if is_execution_control_error(error):
                # Native commit/cancellation authority survives unchanged.
                # Maintenance may later reconcile custody without this turn.
                raise
            try:
                actual = await self.host.store.read(self.original, self.owner.hold.handle)
                if (actual.phase == 'ISSUING' and actual.revision == self.issuance.record.revision):
                    self._require()
                    actual = await write()  # one local retry, never another wire call
                if actual.phase == phase and actual.envelope == {'version': 1, 'native': native}:
                    self._settled(actual)
            except (DatabaseError, OSError, ValueError, ResourceOutcomeUnknown) as reconciliation:
                if is_execution_control_error(reconciliation):
                    raise
            raise ResourceOutcomeUnknown('Original native Task custody could not be confirmed') from error
        self._settled(actual)
        return actual

    async def refuse(self, *, native_errors=None):
        with self.owner.writing():
            return await self._refuse(native_errors=native_errors)

    async def _refuse(self, *, native_errors):
        # Closing a genuine never-sent issuance is metadata cleanup, not a
        # new execution. The exact running frame, original selected witness,
        # host/privacy and caller must still be valid; a retired registration
        # cannot authorize wire, but need not strand this private disposition.
        self._require(local_refusal=True)
        if self._resume_uncertain:
            raise ResourceOutcomeUnknown('Original joint Input claim has no confirmed acknowledgement')
        if self._wire and native_errors is None:
            raise ResourceOutcomeUnknown('Task refusal cannot erase a possible wire dispatch')
        if self.issuance is None:
            # A continuation rejected before joint claim leaves its existing
            # WAITING reservation untouched; it creates no fresh issuance.
            if self._wire or native_errors is not None:
                raise ResourceOutcomeUnknown('Original Task refusal has no issuance')
            self._closed = True
            return
        refusal = {'reason': 'dispatch_not_started'}
        if native_errors is not None:
            # As with native success/claims, Feature owns protocol validation
            # and the one canonical pre-execution error classifier. Preserve
            # its actual native errors, never an abbreviated success receipt.
            if type(native_errors) is not list or not native_errors or any(type(e) is not dict for e in native_errors):
                raise ResourceOutcomeUnknown('Task refusal requires its actual native error evidence')
            raw = encode_inline_json(native_errors, sort_keys=True)
            if len(raw) > self.host.policy.max_native_claim_bytes:
                raise ResourceOutcomeUnknown('Native refusal exceeded its admission budget')
            refusal = {'reason': 'native_rejected_before_execution', 'native_errors': json.loads(raw)}
        envelope = {'version': 1, 'refusal': refusal}
        if self.predecessor is not None:
            if self._wire or self.response_claim is None:
                raise ResourceOutcomeUnknown('Original linked refusal requires both unsent claims')
            try:
                actual, _ = await self.host.joint.retire_unsent(self.issuance,
                    self.response_claim, envelope=envelope)
            except (MCPTaskConflict, MCPInputConflict) as error:
                raise ResourceOutcomeUnknown('Original linked refusal could not be confirmed') from error
        else:
            actual = await self.host.store.settle_initial(self.issuance, disposition='NOT_SENT',
                envelope=envelope)
        self._settled(actual)

    async def suspend(self, binding, payload, *, now, expires_at, predecessor):
        with self.owner.writing():
            return await self._suspend(binding, payload, now=now, expires_at=expires_at,
                predecessor=predecessor)

    async def _suspend(self, binding, payload, *, now, expires_at, predecessor):
        self.require_wire()
        if (not self._wire or binding.connection_id != self.connection_name
                or binding.contract_digest != self.original.contract_digest
                or predecessor is not (self.host.input_host._reader.get().response_claim
                    if self.predecessor is not None else None)):
            raise ResourceOutcomeUnknown('Original Task input suspension binding changed')
        native = self._native(payload['native'])
        try:
            if len(self.issuance.record.continuity['operations']) >= self.host.policy.max_continuations:
                raise MCPTaskConflict('Task continuation budget cannot admit another operator wait')
            record = await self.host.joint.suspend(self.issuance, binding, payload,
                now=now, expires_at=expires_at, predecessor=predecessor)
        except ValueError as error:
            # The joint transaction rolled back. Retain the actual native
            # InputRequiredResult as a local disposition, never a completion.
            await self.host.store.settle_initial(self.issuance, disposition='SETTLED',
                envelope={'version': 1, 'native': native})
            actual = await self.host.store.read(self.original, self.owner.hold.handle)
            self._settled(actual)
            raise ResourceOutcomeUnknown('Original native input could not be retained') from error
        actual = await self.host.store.read(self.original, self.owner.hold.handle)
        self._suspended = record
        # The native connection rechecks current authority after the Input
        # handler returns. Keep that check live until Feature observes the
        # actual MCPInputSuspension; start_wire still forbids another call.
        self._settled(actual, close=False)
        return record

    async def claim_response(self, record, responses, *, operation_id, now):
        with self.owner.writing():
            return await self._claim_response(record, responses, operation_id=operation_id, now=now)

    async def _claim_response(self, record, responses, *, operation_id, now):
        self._require()
        if self.predecessor is not record or self.issuance is not None:
            raise ResourceConflict('Original linked Input response round is required')
        self._resume_uncertain = True
        try:
            claim, issuance = await self.host.joint.resume(record, responses,
                operation_id=operation_id, now=now)
        except (MCPTaskConflict, MCPInputConflict) as error:
            self._resume_uncertain = False
            raise ResourceConflict('Original linked Input claim was refused') from error
        self.issuance, self.response_claim = issuance, claim
        self._resume_uncertain = False
        self._require()
        return claim


class MCPTaskHostMixin:
    # Public capability advertisement is a later acceptance-gated change.
    def create_mcp_task_host(self, feature, *, input_host, configuration, is_current):
        if any(not host.retired for host in getattr(self, '_mcp_task_hosts', ())):
            raise ResourceConflict('Close the original MCP Task host before replacement')
        host = MCPTaskHost(self, feature, input_host, configuration=configuration, is_current=is_current)
        self._mcp_task_hosts = [host]
        return host
