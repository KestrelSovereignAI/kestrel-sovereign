"""Original-authority dispatch for native Task mutations."""

from contextvars import ContextVar
from dataclasses import dataclass
import json
from secrets import compare_digest
from collections.abc import Callable

from kestrel_sdk.resources import ResourceConflict
from kestrel_sdk.tools import NativeActionContext

from kestrel_sovereign.a2a.media import encode_inline_json
from kestrel_sovereign.agent.tool_registry import _GovernedToolBinding, _ToolExecution
from kestrel_sovereign.agent.turn_lifecycle import current_turn_is_signal_driven
from kestrel_sovereign.auth import caller_identity_digest, current_caller_context
from kestrel_sovereign.features.security.hooks import MCP_TASK_SERVICE_SESSION_ID
from kestrel_sovereign.mcp_task_store import MCPTaskConflict, MCPTaskRecord
from kestrel_sovereign.mcp_task_submission import MCPTaskInteractiveOrigin
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.host_authority import require_sovereign_caller
from kestrel_sovereign.security.request_auth import InteractiveRequestAuthentication, current_request_authentication
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate
from kestrel_sovereign.tools.effect_journal import current_tool_effect_execution


@dataclass(eq=False, repr=False)
class _MutationAdmission:
    binding: _GovernedToolBinding
    record: MCPTaskRecord
    context: NativeActionContext
    authentication: InteractiveRequestAuthentication
    view_check: Callable[[], None] | None = None
    active: bool = True
    entered: bool = False

    def require_current(self):
        authentication = self.authentication
        if (not self.active or current_request_authentication() is not authentication
                or current_caller_context() is not authentication.caller
                or current_turn_is_signal_driven()):
            raise HostCapabilityAccessDenied('Original MCP Task operator authority changed')
        authentication.require_current()
        if self.context.action == 'cancel':
            require_sovereign_caller('MCP Task cancellation')
        if self.view_check is not None:
            self.view_check()
        self.binding.require_current()


class MCPTaskActions:
    """Select the original Task before dispatching its captured tool action."""

    def __init__(self, host):
        self.host = host
        self._admitted = ContextVar('mcp_task_mutation_admission', default=None)

    async def invoke(self, binding, *, handle, session_id, operation_id, contract_digest,
                     kind, arguments):
        host = self.host
        host.require_current()
        authentication = current_request_authentication()
        if (authentication is None or authentication.caller is not current_caller_context()
                or current_turn_is_signal_driven()):
            raise HostCapabilityAccessDenied('MCP Task mutations require a current operator')
        authentication.require_current()
        if kind == 'cancel':
            require_sovereign_caller('MCP Task cancellation')
        elif kind != 'update':
            raise ValueError('Unknown native MCP Task mutation')
        if (type(binding) is not _GovernedToolBinding or binding.agent is not host.agent
                or type(contract_digest) is not str):
            raise ResourceConflict('Original MCP Task tool registration is unavailable')
        binding.require_current()
        async with require_durable_persistence_gate(host.agent).shared():
            host.require_current()
            record = await host.store.read_handle(handle)
            if (record is None or record.phase != 'ATTACHED' or record.remote_task_id is None
                    or record.binding.tool_name != binding.name
                    or not compare_digest(record.binding.contract_digest, contract_digest)):
                raise ResourceConflict('Original MCP Task mutation target is unavailable')
            if kind == 'cancel' and record.cancel_requested:
                raise ResourceConflict('Original MCP Task already has a cancellation request')
            origin = record.binding.origin
            if kind == 'update' and (type(origin) is not MCPTaskInteractiveOrigin
                    or origin.session_id != session_id
                    or origin.principal_digest != caller_identity_digest(authentication.caller)):
                raise HostCapabilityAccessDenied('Original MCP Task input owner is required')
            submission = await host.store.open_submission(record)
            host.require_current()
            delivery = submission.delivery
            if submission.authorization.owner != binding.owner:
                raise HostCapabilityAccessDenied('Original MCP Task delivery authority is required')
            if kind == 'update' and delivery.kind not in {'chat', 'app'}:
                raise HostCapabilityAccessDenied('Original MCP Task conversation is required')
            view_check = host.results.require_app_delivery(delivery) if delivery.kind == 'app' else None
            origin_digest = (origin.principal_digest if type(origin) is MCPTaskInteractiveOrigin
                else origin.namespace_digest)
            context = NativeActionContext(action=kind, method='tasks/' + kind,
                task_handle=record.handle, remote_task_digest=host.store._remote_digest(
                    record.binding, record.remote_task_id),
                owner=binding.owner, tool_name=binding.name, registration_id=binding.registration_id,
                schema_digest=record.binding.contract_digest, attribution_kind=origin.kind,
                attribution_digest=origin_digest, input_revision=record.revision,
                conversation_id=delivery.session_id, view_label=delivery.view_token)
            admission = _MutationAdmission(binding, record, context, authentication, view_check)
            admission.require_current()

        # Ordinary governance and approval cover the captured action. They do
        # not hold the persistence gate while a person decides.
        reset = self._admitted.set(admission)
        try:
            return await binding.invoke_action(kind, arguments, native_action=context,
                session_id=delivery.session_id or MCP_TASK_SERVICE_SESSION_ID,
                operation_id=operation_id)
        finally:
            admission.active = False
            self._admitted.reset(reset)

    def round(self, binding, context, arguments):
        host = self.host
        host.require_current()
        admission = self._admitted.get()
        frame = current_tool_effect_execution(host.agent)
        witness = frame.witness
        if (admission is None or admission.binding is not binding or admission.entered
                or admission.context != context or type(witness) is not _ToolExecution
                or not witness.started or witness.returned or witness.native_action != context
                or witness.registration is not binding.registration or witness.tool is not binding.tool
                or witness.owner != binding.owner or witness.tool_name != binding.name
                or encode_inline_json(arguments, sort_keys=True) != frame.arguments_json):
            raise HostCapabilityAccessDenied('Original governed MCP Task mutation is required')
        admission.require_current()
        admission.entered = True
        return MCPTaskMutationRound(host, admission, frame)


class MCPTaskMutationRound:
    """One SQL issuance under the original governed leaf."""

    def __init__(self, host, admission, frame):
        self.host, self._admission, self._frame = host, admission, frame
        self._issuance = None
        self._wire_started = False

    def require_current(self):
        self.require_transport_current()
        if current_tool_effect_execution(self.host.agent) is not self._frame:
            raise HostCapabilityAccessDenied('Original MCP Task mutation handler changed')

    def require_transport_current(self):
        """Recheck the original running handler from its managed SDK transport."""
        self.host.require_current()
        self._admission.require_current()
        if not self._frame.live or self._frame.witness.returned:
            raise HostCapabilityAccessDenied('Original MCP Task mutation handler ended')

    @property
    def operator_id(self):
        self.require_current()
        return caller_identity_digest(self._admission.authentication.caller)

    @property
    def record(self):
        self.require_current()
        return self._admission.record

    async def reserve(self):
        self.require_current()
        if self._issuance is not None:
            raise ResourceConflict('Original MCP Task mutation cannot be replayed')
        arguments = json.loads(self._frame.arguments_json)
        kind = self._admission.context.action
        if set(arguments) != ({'inputResponses'} if kind == 'update' else set()):
            raise ResourceConflict('Original MCP Task mutation arguments changed')
        responses = arguments.get('inputResponses')
        async with require_durable_persistence_gate(self.host.agent).shared():
            self.require_current()
            original = self.record
            try:
                latest = await self.host.store.read(original.binding, original.handle)
                self.require_current()
                if latest.phase != 'ATTACHED' or latest.remote_task_id != original.remote_task_id:
                    raise ResourceConflict('Original MCP Task is no longer available for mutation')
                if kind == 'update':
                    self._require_input_requests(original, latest, responses)
                elif latest.cancel_requested:
                    raise ResourceConflict('Original MCP Task already has a cancellation request')
                reservation = await self.host.store.reserve_mutation(latest,
                    operation_id=self._frame.claim.claim_id, kind=kind,
                    input_responses=responses)
            except MCPTaskConflict as error:
                raise ResourceConflict('Original MCP Task mutation changed') from error
            if not reservation.fresh:
                raise ResourceConflict('Original MCP Task mutation cannot be replayed')
            self._issuance = reservation.issuance
            self.require_current()
        return reservation.record

    @staticmethod
    def _require_input_requests(original, latest, responses):
        earlier, pending = original.pending_inputs, latest.pending_inputs
        if (type(responses) is not dict or not responses
                or not responses.keys() <= earlier.keys() & pending.keys()
                or not all(encode_inline_json(earlier[key], sort_keys=True)
                    == encode_inline_json(pending[key], sort_keys=True) for key in responses)):
            raise ResourceConflict('Original MCP Task input requests changed or were consumed')

    def start_wire(self):
        self.require_current()
        if self._issuance is None or self._wire_started:
            raise ResourceConflict('Original MCP Task mutation has no fresh wire admission')
        self._wire_started = True

    async def acknowledge(self, acknowledgement):
        self.require_current()
        if not self._wire_started:
            raise ResourceConflict('Original MCP Task mutation did not start on the wire')
        async with require_durable_persistence_gate(self.host.agent).shared():
            self.require_current()
            record = await self.host.store.acknowledge_mutation(self._issuance, acknowledgement)
            self.require_current()
            return record
