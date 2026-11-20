"""One durable commit for original-method MCP input and Task issuance.

These private storage credentials grant no permission to execute. The host must
independently govern every continuation and retain its actual connection borrow.
"""

from dataclasses import asdict, replace
import hashlib
import secrets

from kestrel_sovereign.mcp_input_store import MCPInputBinding, MCPInputClaim, MCPInputRecord, MCPInputStore
from kestrel_sovereign.mcp_task_store import (
    MCPTaskConflict, MCPTaskIssuance, MCPTaskRecord, MCPTaskStore,
)
from kestrel_sovereign.mcp_task_submission import (
    MCPTaskInteractiveOrigin, _binding_text, decode_origin,
)


class MCPTaskInputCustody:
    def __init__(self, tasks: MCPTaskStore, inputs: MCPInputStore):
        if type(tasks) is not MCPTaskStore or type(inputs) is not MCPInputStore:
            raise TypeError('Joint MCP custody requires its actual canonical stores')
        self.tasks, self.inputs = tasks, inputs
        self._database, self._agent_did, self._scope = tasks.database, tasks.agent_did, tasks.scope
        self._require()

    def _require(self):
        tasks, inputs = self.tasks, self.inputs
        if (tasks.database is not self._database or inputs.database is not self._database
                or tasks.backend is not self._database.backend or inputs.backend is not self._database.backend
                or tasks.agent_did != self._agent_did or inputs.agent_did != self._agent_did
                or tasks.scope != self._scope or inputs.scope != self._scope):
            raise MCPTaskConflict('Joint MCP custody requires its original database, agent and scope')
        tasks._ready()
        inputs._ready()

    @staticmethod
    def _binding(task: MCPTaskRecord, record: MCPInputRecord):
        origin, binding = task.binding.origin, record.binding
        if (type(origin) is not MCPTaskInteractiveOrigin or binding.custody != task.handle
                or binding.agent_did != task.binding.agent_did or binding.method != 'tools/call'
                or binding.principal_id != origin.principal_digest or binding.session_id != origin.session_id
                or binding.contract_digest != task.binding.contract_digest):
            raise MCPTaskConflict('MCP input does not belong to its original Task attribution and contract')

    def _input_link(self, task, record):
        self._binding(task, record)
        expected = {'request_id': record.request_id, 'reservation_id': record.reservation_id,
            'operation_id': record.binding.operation_id}
        if task.continuity['input'] != expected:
            raise MCPTaskConflict('MCP input has another original Task lineage')

    def _waiting_link(self, task, record):
        self._input_link(task, record)
        if (task.phase != 'INPUT_WAITING' or task.current_operation_id != record.binding.operation_id):
            raise MCPTaskConflict('MCP input has another original waiting Task revision')

    async def suspend(self, issuance: MCPTaskIssuance, binding: MCPInputBinding,
                      payload: dict, *, now: float, expires_at: float, predecessor=None) -> MCPInputRecord:
        self._require()
        identity = self.tasks._issuance_identity(issuance)
        if type(binding) is not MCPInputBinding or binding.custody not in (None, identity.handle):
            raise MCPTaskConflict('MCP input requires its original Task binding')
        linked = replace(binding, custody=identity.handle)
        prepared = await self.inputs._prepare_create(linked, payload, now=now,
            expires_at=expires_at, predecessor=predecessor)
        self._require()
        async with self.inputs.write_transaction():
            record = await self.inputs._apply_create(prepared)
            await self.tasks._lock()
            records = await self.tasks._all()
            actual, row = self.tasks._select_issuance(self.tasks._in_inventory(records, identity.handle), identity)
            self._binding(actual, record)
            if record.binding.operation_id != actual.current_operation_id:
                raise MCPTaskConflict('MCP input requires its actual current Task operation')
            continuity = actual.continuity
            lineage = self.tasks._digest('mcp-task-input-lineage-v1', record.reservation_id)
            if actual.input_lineage_digest is None:
                if (prepared.predecessor is not None or record.predecessor_id is not None
                        or record.reservation_id != record.request_id):
                    raise MCPTaskConflict('Initial MCP Task input cannot adopt another reservation')
            else:
                parent, previous = prepared.predecessor, continuity['input']
                if (parent is None or record.predecessor_id != previous['request_id']
                        or record.reservation_id != previous['reservation_id']
                        or parent.identity.request_id != previous['request_id']
                        or parent.identity.binding.operation_id != previous['operation_id']
                        or actual.input_lineage_digest != lineage):
                    raise MCPTaskConflict('MCP Task successor requires its actual original response lineage')
            if any(other.handle != actual.handle and other.input_lineage_digest == lineage
                    for other, _ in records):
                raise MCPTaskConflict('MCP input reservation already belongs to another Task')
            continuity['input'] = {'request_id': record.request_id,
                'reservation_id': record.reservation_id, 'operation_id': record.binding.operation_id}
            await self.tasks._persist(actual, row, 'INPUT_WAITING', None,
                continuity=continuity, input_lineage_digest=lineage)
        return record

    async def resume(self, expected_input: MCPInputRecord, responses: dict, *, operation_id: str, now: float):
        self._require()
        _binding_text(operation_id)
        prepared = await self.inputs._prepare_response(expected_input, responses, now=now)
        handle = prepared.claim.record.binding.custody
        if handle is None:
            raise MCPTaskConflict('MCP Task continuation requires its actual linked Input')
        secret = secrets.token_urlsafe(32)
        claim_hash = hashlib.sha256(secret.encode()).hexdigest()
        self._require()
        async with self.inputs.write_transaction():
            claimed = await self.inputs._apply_response(prepared)
            await self.tasks._lock()
            records = await self.tasks._all()
            selected = self.tasks._in_inventory(records, handle)
            if selected is None:
                raise MCPTaskConflict('Original waiting MCP Task is absent')
            actual, row = selected
            self._waiting_link(actual, claimed.record)
            if self.tasks._operation_owner(records, operation_id) is not None:
                raise MCPTaskConflict('MCP Task continuation operation already has original custody')
            continuity = actual.continuity
            continuity['operations'].append(operation_id)
            resumed = await self.tasks._persist(actual, row, 'ISSUING', None,
                continuity=continuity, claim_hash=claim_hash)
        return claimed, MCPTaskIssuance(resumed, secret)

    async def retire_unsent(self, issuance: MCPTaskIssuance, claim: MCPInputClaim, *,
                            envelope: dict) -> tuple[MCPTaskRecord, MCPInputRecord]:
        """Retire both original claims; the host independently proves no send."""
        self._require()
        expected = self.tasks._issuance_identity(issuance)
        envelope = self.tasks._snapshot(envelope)
        prepared = await self.inputs._prepare_settlement(claim, state='NON_RESUMABLE')
        self._require()
        async with self.inputs.write_transaction():
            closed_input = await self.inputs._apply_settlement(prepared)
            await self.tasks._lock()
            actual, row = self.tasks._select_issuance(await self.tasks._read_row(expected.handle), expected)
            self._input_link(actual, closed_input)
            closed_task = await self.tasks._persist(actual, row, 'NOT_SENT', envelope)
        return closed_task, closed_input

    async def close_expired(self, expected: MCPTaskRecord, *, now: float) -> MCPTaskRecord:
        self._require()
        self.tasks._expected(expected)
        expected = replace(expected, binding=replace(expected.binding,
            origin=decode_origin(asdict(expected.binding.origin))))
        actual = await self.tasks.read(expected.binding, expected.handle)
        if actual.phase != 'INPUT_WAITING' or actual.revision != expected.revision:
            raise MCPTaskConflict('MCP Task expiry requires its original waiting revision')
        origin, link = actual.binding.origin, actual.continuity['input']
        record = await self.inputs.read_for(origin.principal_digest, origin.session_id, link['request_id'])
        self._waiting_link(actual, record)
        prepared = await self.inputs._prepare_expiry(record, now=now)
        self._require()
        async with self.inputs.write_transaction():
            await self.inputs._apply_expiry(prepared)
            await self.tasks._lock()
            actual, row = self.tasks._require_binding(await self.tasks._read_row(expected.handle), expected.binding)
            if actual.revision != expected.revision:
                raise MCPTaskConflict('MCP Task waiting expiry revision was already consumed')
            self._waiting_link(actual, record)
            closed = await self.tasks._persist(actual, row, 'INPUT_CLOSED', None)
        return closed
