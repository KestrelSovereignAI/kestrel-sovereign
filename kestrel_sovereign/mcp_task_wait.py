"""Local wait status over authenticated native Task custody."""

from typing import ClassVar
from uuid import UUID

from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.auth import caller_identity_digest, current_caller_context
from kestrel_sovereign.mcp_task_store import RECOVERABLE_OBSERVATION_BLOCKS
from kestrel_sovereign.mcp_task_submission import MCPTaskInteractiveOrigin
from kestrel_sovereign.mcp_tasks import MCPTaskHost
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate
from kestrel_sovereign.tools.effect_invocation import current_service_operation_attribution
from kestrel_sovereign.tools.effect_journal import ToolEffectJournalUnavailable
from kestrel_sovereign.waits.engine import TERMINAL_EVENT_FINAL_KEY, TERMINAL_EVENT_KEY


class MCPTaskWaitable:
    kind: ClassVar[str] = 'mcp'
    signal: ClassVar[None] = None

    def __init__(self, agent, feature):
        self._agent, self._feature = agent, feature

    @staticmethod
    def _belongs_to_caller(origin):
        caller = current_caller_context()
        if caller is None:
            return True
        if type(origin) is MCPTaskInteractiveOrigin:
            return caller_identity_digest(caller) == origin.principal_digest
        try:
            return current_service_operation_attribution() == (origin.service_kind, origin.namespace_digest)
        except ToolEffectJournalUnavailable:
            return False

    async def _record(self, handle):
        try:
            if type(handle) is not str or len(handle) != 36 or str(UUID(handle)) != handle:
                return None
        except ValueError:
            return None
        host = next(iter(getattr(self._agent, '_mcp_task_hosts', ())), None)
        if type(host) is not MCPTaskHost or host.agent is not self._agent or host.feature is not self._feature:
            return None
        try:
            host.require_current()
            async with require_durable_persistence_gate(self._agent).shared():
                host.require_current()
                record = await host.store.read_handle(handle)
                host.require_current()
                if (record is None or record.remote_task_id is None
                        or not self._belongs_to_caller(record.binding.origin)):
                    return None
                return record
        except (HostCapabilityAccessDenied, ToolEffectJournalUnavailable):
            return None

    async def owns_handle(self, handle: str) -> bool:
        return await self._record(handle) is not None

    async def poll(self, handle: str) -> WaitStatus:
        record = await self._record(handle)
        if record is None:
            return WaitStatus(Outcome.PENDING, 'Original MCP Task is unavailable')
        local = record.local
        block = local['observation_block']
        observation = record.observation
        state = (observation if observation is not None else record.envelope['native'])['status']
        data = {'mcp_task': record.handle, 'state': state}
        final = record.phase == 'SETTLED' or (
            block is not None and block['reason'] not in RECOVERABLE_OBSERVATION_BLOCKS)
        if not final:
            description = 'MCP Task observation is suspended' if block else 'MCP Task is pending'
            return WaitStatus(Outcome.PENDING, description, data=data)
        data.update({TERMINAL_EVENT_KEY: f'mcp-task:{record.handle}', TERMINAL_EVENT_FINAL_KEY: True})
        if block is not None:
            return WaitStatus(Outcome.PARTIAL, 'MCP Task outcome is unknown', data=data)
        if local['withheld'] is not None:
            return WaitStatus(Outcome.PARTIAL, 'MCP Task completed with its result withheld', data=data)
        if state == 'failed' or (state == 'completed' and observation['result'].get('isError') is True):
            return WaitStatus(Outcome.FAILED, 'MCP Task failed', data=data)
        if state == 'cancelled':
            return WaitStatus(Outcome.PARTIAL, 'MCP Task was cancelled', data=data)
        return WaitStatus(Outcome.DONE, 'MCP Task completed', data=data)
