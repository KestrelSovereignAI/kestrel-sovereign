"""Private MCP Task custody; neither a handle nor an issuer grants permission.

The feature validates native wire types and the host authorizes each operation.
This repository reserves before dispatch, retains uncertain submissions forever,
and never recovers an issuance secret or enables replay after restart.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import secrets
from typing import Literal
from uuid import uuid4

from kestrel_sdk.resources import ResourceError
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.encryption import get_fernet
from kestrel_sdk.security.exceptions import DecryptionError
from kestrel_sovereign.a2a.media import encode_inline_json
from kestrel_sovereign.a2a.stores.unified.base import NativeStoreConflict, UnifiedStoreBase
from kestrel_sovereign.security.envelope_keys import resolve_envelope_key
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.initialization_errors import repository_initialization_errors


MCP_TASK_SCHEMA_REVISION = 2


class MCPTaskConflict(NativeStoreConflict):
    """The original reservation cannot admit this change."""


class MCPTaskIntegrityError(ResourceError):
    """Original custody cannot safely be opened or replaced."""


def _binding_text(value: str) -> None:
    if (not isinstance(value, str) or not value or len(value.encode('utf-8')) > 1024
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('MCP Task bindings require bounded control-free strings')


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True, repr=False)
class MCPTaskBinding:
    agent_did: str
    principal_id: str
    session_id: str
    operation_id: str
    connection_id: str
    authorization_id: str
    tool_name: str
    contract_digest: str

    def __post_init__(self):
        for value in asdict(self).values():
            _binding_text(value)
        if not self.agent_did.startswith('did:'):
            raise ValueError('MCP Task binding requires an agent DID')
        if not all(_is_digest(value) for value in (
                self.connection_id, self.authorization_id, self.contract_digest)):
            raise ValueError('MCP Task binding requires original identity and contract digests')


@dataclass(frozen=True)
class MCPTaskRecord:
    handle: str
    revision: int
    phase: str
    payload_limit: int
    binding: MCPTaskBinding = field(repr=False)
    _payload: bytes = field(repr=False)

    @property
    def remote_task_id(self) -> str | None:
        return json.loads(self._payload)['remote_task_id']

    @property
    def envelope(self) -> dict | None:
        """Return a detached private claim/result, never a public wait payload."""
        return json.loads(self._payload)['envelope']

    @property
    def control(self) -> dict:
        """Detached private lifetime history; never an execution credential."""
        envelope = self.envelope
        return envelope.get('control', {'inputs': {}, 'mutations': {}}) if envelope else {
            'inputs': {}, 'mutations': {}}

    @property
    def pending_inputs(self) -> dict:
        observation = self.observation
        if observation is None or observation['status'] != 'input_required':
            return {}
        consumed = {key for receipt in self.control['mutations'].values()
            for key in receipt['intent']['params'].get('inputResponses', {})}
        return {key: value for key, value in observation['inputRequests'].items() if key not in consumed}

    @property
    def observation(self) -> dict | None:
        envelope = self.envelope
        return envelope.get('observation') if envelope is not None else None


@dataclass(frozen=True)
class MCPTaskIssuance:
    record: MCPTaskRecord
    secret: str = field(repr=False)


@dataclass(frozen=True)
class MCPTaskReservation:
    record: MCPTaskRecord
    issuance: MCPTaskIssuance | None = field(default=None, repr=False)

    @property
    def fresh(self) -> bool:
        return self.issuance is not None


@dataclass(frozen=True)
class MCPTaskMutationIssuance:
    record: MCPTaskRecord
    operation_id: str = field(repr=False)
    secret: str = field(repr=False)


@dataclass(frozen=True)
class MCPTaskMutationReservation:
    record: MCPTaskRecord
    issuance: MCPTaskMutationIssuance | None = field(default=None, repr=False)

    @property
    def fresh(self) -> bool:
        return self.issuance is not None


class MCPTaskStore(UnifiedStoreBase):
    """Borrow primary storage; conserve custody independently of remote state.

    No time-based reclamation or deletion is provided. Active and historical
    record budgets refuse new work; lowered budgets cannot strand existing work.
    """

    def __init__(self, database: AsyncDatabase, *, agent_did: str,
                 max_active: int, max_records: int, max_payload_bytes: int):
        super().__init__(database.backend, owns_backend=False)
        _binding_text(agent_did)
        if not agent_did.startswith('did:'):
            raise ValueError('MCP Task store requires an agent DID')
        for limit in (max_active, max_records, max_payload_bytes):
            if type(limit) is not int or limit <= 0:
                raise ValueError('MCP Task storage budgets must be positive integers')
        self.database, self.agent_did = database, agent_did
        self.scope = hashlib.sha256(agent_did.encode()).hexdigest()
        self.max_active, self.max_records = max_active, max_records
        self.max_payload_bytes = max_payload_bytes
        self._root: bytes | None = None
        self._initialization_lock = asyncio.Lock()

    def _outermost(self):
        if self.database.owns_open_transaction is not False:
            raise MCPTaskConflict('MCP Task custody requires its own durable transaction')

    def _ready(self):
        self._outermost()
        if self._root is None:
            raise RuntimeError('MCP Task store is not initialized')

    async def close(self):
        await super().close()
        self._root = None

    async def initialize(self):
        self._outermost()
        async with self._initialization_lock:
            if self._root is not None:
                return
            if not (self.is_postgres or self.is_sqlite):
                raise ValueError('MCP Task custody requires SQLite or PostgreSQL')
            cipher = get_fernet()
            if cipher is None:
                raise MCPTaskIntegrityError('MCP Task custody requires encryption')
            binary = 'BYTEA' if self.is_postgres else 'BLOB'
            with repository_initialization_errors(integrity_error=MCPTaskIntegrityError,
                    unknown_message='MCP Task initialization requires reconciliation'):
                # Keep the original shared lock across schema revisions.
                # Revision1 retained only snapshots and cannot prove lifetime history.
                async with self.database.migration_lock('kestrel-mcp-task-schema-v1'):
                    await self.database.execute('CREATE TABLE IF NOT EXISTS mcp_task_schema (singleton BIGINT PRIMARY KEY, revision BIGINT NOT NULL)')
                    revision = await self.database.fetchone('SELECT revision FROM mcp_task_schema WHERE singleton=1')
                    if revision is None:
                        await self.database.execute(f'CREATE TABLE mcp_task_keys (scope TEXT PRIMARY KEY, key_id TEXT NOT NULL UNIQUE, wrapped_key {binary} NOT NULL, revision BIGINT NOT NULL)')
                        await self.database.execute(f"CREATE TABLE mcp_task_records (scope TEXT NOT NULL, handle TEXT NOT NULL, operation_digest TEXT NOT NULL, binding_digest TEXT NOT NULL, remote_digest TEXT, revision BIGINT NOT NULL, phase TEXT NOT NULL CHECK(phase IN ('ISSUING','ATTACHED','SETTLED','NOT_SENT')), claim_hash TEXT, payload_limit BIGINT NOT NULL, payload {binary} NOT NULL, PRIMARY KEY(scope,handle), UNIQUE(scope,operation_digest), UNIQUE(scope,remote_digest), FOREIGN KEY(scope) REFERENCES mcp_task_keys(scope))")
                        await self.database.execute('INSERT INTO mcp_task_schema(singleton,revision) VALUES (1,?)', (MCP_TASK_SCHEMA_REVISION,))
                    elif revision != (MCP_TASK_SCHEMA_REVISION,):
                        raise MCPTaskIntegrityError('Unsupported MCP Task schema revision')
                    await self.database.fetchone('SELECT scope,key_id,wrapped_key,revision FROM mcp_task_keys WHERE 1=0')
                    await self.database.fetchone('SELECT scope,handle,operation_digest,binding_digest,remote_digest,revision,phase,claim_hash,payload_limit,payload FROM mcp_task_records WHERE 1=0')
                    row = await self.database.fetchone('SELECT wrapped_key FROM mcp_task_keys WHERE scope=?', (self.scope,))
                    dependent = await self.database.fetchone('SELECT 1 FROM mcp_task_records WHERE scope=? LIMIT 1', (self.scope,))
                    root, envelope = resolve_envelope_key(cipher, bytes(row[0]) if row else None,
                        has_dependents=dependent is not None, integrity_error=MCPTaskIntegrityError,
                        missing_message='Original MCP Task envelope is missing',
                        unavailable_message='Original MCP Task encryption key is unavailable',
                        invalid_message='Invalid MCP Task envelope')
                    if envelope is not None:
                        await self.database.execute('INSERT INTO mcp_task_keys(scope,key_id,wrapped_key,revision) VALUES (?,?,?,0)',
                            (self.scope, self.scope + ':v1', envelope))
            self._root = root

    def _digest(self, domain: str, value: object) -> str:
        if self._root is None:
            raise RuntimeError('MCP Task store is not initialized')
        return hmac.new(self._root, encode_inline_json([domain, value], sort_keys=True), hashlib.sha256).hexdigest()

    def _binding_digest(self, binding: MCPTaskBinding) -> str:
        if type(binding) is not MCPTaskBinding or binding.agent_did != self.agent_did:
            raise MCPTaskConflict('MCP Task binding belongs to another agent')
        return self._digest('mcp-task-binding-v1', asdict(binding))

    def _remote_digest(self, binding, task_id):
        if task_id is None:
            return None
        return self._digest('mcp-task-remote-v1', [binding.connection_id, binding.authorization_id, task_id])

    def _aad(self, columns):
        return encode_inline_json(['mcp-task-v1', *columns])

    @staticmethod
    def _envelope(phase, task_id, envelope):
        if phase == 'ISSUING':
            if task_id is not None or envelope is not None:
                raise ValueError('Unsubmitted MCP Task custody cannot have a result')
            return
        member = 'refusal' if phase == 'NOT_SENT' else 'native'
        remote_bound = task_id is not None
        allowed = ({'version', 'native'}, {'version', 'native', 'observation'},
            {'version', 'native', 'control'}, {'version', 'native', 'observation', 'control'}) if remote_bound else ({'version', member},)
        if (type(envelope) is not dict or set(envelope) not in allowed
                or type(envelope['version']) is not int or envelope['version'] != 1
                or type(envelope[member]) is not dict):
            raise ValueError('MCP Task custody requires a versioned private JSON envelope')
        if remote_bound:
            if (not isinstance(task_id, str) or envelope['native'].get('taskId') != task_id
                    or envelope['native'].get('resultType') != 'task'
                    or phase not in {'ATTACHED','SETTLED'}):
                raise ValueError('MCP Task claim must bind its exact native task ID')
            observation = envelope.get('observation')
            if 'observation' in envelope:
                terminal = MCPTaskStore._observation(task_id, observation)
                if terminal != (phase == 'SETTLED'):
                    raise ValueError('MCP Task phase must match its confirmed observation')
            elif phase != 'ATTACHED':
                raise ValueError('Settled remote Tasks require a terminal observation')
            MCPTaskStore._control(task_id, envelope)
        elif phase == 'ATTACHED':
            raise ValueError('Attached MCP Tasks require their remote identity')

    @staticmethod
    def _observation(task_id, observation):
        if (type(observation) is not dict or observation.get('taskId') != task_id
                or observation.get('resultType') != 'complete'
                or not isinstance(observation.get('status'), str)
                or observation.get('status') not in {'working','input_required','completed','failed','cancelled'}):
            raise ValueError('MCP Task observation must bind its exact native identity and status')
        member = {'input_required': 'inputRequests', 'completed': 'result', 'failed': 'error'}.get(observation['status'])
        if member is not None and type(observation.get(member)) is not dict:
            raise ValueError('MCP Task observation requires its native status payload')
        return observation['status'] in {'completed','failed','cancelled'}

    @staticmethod
    def _control(task_id, envelope):
        control = envelope.get('control', {'inputs': {}, 'mutations': {}})
        if (type(control) is not dict or set(control) != {'inputs', 'mutations'}
                or type(control['inputs']) is not dict or type(control['mutations']) is not dict
                or any(not isinstance(key, str) or type(request) is not dict
                    for key, request in control['inputs'].items())):
            raise ValueError('Invalid MCP Task input history')
        consumed = set()
        for operation, receipt in control['mutations'].items():
            _binding_text(operation)
            if type(receipt) is not dict or set(receipt) != {'intent', 'claim_hash', 'acknowledgement'}:
                raise ValueError('Invalid MCP Task mutation receipt')
            intent, ack, claim_hash = receipt['intent'], receipt['acknowledgement'], receipt['claim_hash']
            if (type(intent) is not dict or set(intent) != {'method', 'params'}
                    or intent['method'] not in ('tasks/update', 'tasks/cancel')
                    or type(intent['params']) is not dict or intent['params'].get('taskId') != task_id):
                raise ValueError('Invalid MCP Task mutation intent')
            params = intent['params']
            if intent['method'] == 'tasks/update':
                responses = params.get('inputResponses')
                if (set(params) != {'taskId', 'inputResponses'} or type(responses) is not dict
                        or not responses or any(not isinstance(key, str) or type(value) is not dict
                            for key, value in responses.items())
                        or not set(responses) <= set(control['inputs']) or consumed.intersection(responses)):
                    raise ValueError('Invalid MCP Task consumed input history')
                consumed.update(responses)
            elif set(params) != {'taskId'}:
                raise ValueError('Invalid MCP Task cancellation intent')
            if ack is None:
                if not _is_digest(claim_hash):
                    raise ValueError('Invalid MCP Task pending mutation claim')
            elif claim_hash is not None or type(ack) is not dict or ack.get('resultType') != 'complete':
                raise ValueError('Invalid MCP Task mutation acknowledgement')
        observation = envelope.get('observation')
        if observation is not None and observation['status'] == 'input_required':
            for key, request in observation['inputRequests'].items():
                if (not isinstance(key, str) or type(request) is not dict or key not in control['inputs']
                        or not MCPTaskStore._same_json(request, control['inputs'][key])):
                    raise ValueError('MCP Task snapshot does not match its original input history')
        return control

    @staticmethod
    def _same_json(first, second):
        return encode_inline_json(first, sort_keys=True) == encode_inline_json(second, sort_keys=True)

    @staticmethod
    def _snapshot(value):
        return json.loads(encode_inline_json(value))

    @staticmethod
    def _retain_inputs(envelope, observation):
        if observation['status'] != 'input_required':
            return
        control = envelope.setdefault('control', {'inputs': {}, 'mutations': {}})
        for key, request in observation['inputRequests'].items():
            if not isinstance(key, str) or type(request) is not dict:
                raise ValueError('MCP Task input requests require opaque string keys and objects')
            if key in control['inputs'] and not MCPTaskStore._same_json(control['inputs'][key], request):
                raise MCPTaskConflict('MCP Task input key was reused for another request')
            control['inputs'][key] = request

    def _encode(self, binding, columns, task_id, envelope):
        self._envelope(columns[6], task_id, envelope)
        raw = encode_inline_json({'binding': asdict(binding), 'remote_task_id': task_id, 'envelope': envelope})
        if len(raw) > columns[8]:
            raise ValueError('MCP Task custody exceeded its storage budget')
        return raw, AEADCipher(self._root).encrypt(raw, aad=self._aad(columns))

    def _open(self, row) -> MCPTaskRecord:
        scope, handle, operation, digest, remote, revision, phase, claim_hash, limit, ciphertext = row
        if (scope != self.scope or not isinstance(handle, str) or not handle
                or not _is_digest(operation) or not _is_digest(digest)
                or (remote is not None and not _is_digest(remote))
                or type(revision) is not int or revision < 0
                or phase not in {'ISSUING','ATTACHED','SETTLED','NOT_SENT'}
                or ((phase == 'ISSUING') != (claim_hash is not None))
                or (claim_hash is not None and not _is_digest(claim_hash))
                or (phase == 'ATTACHED' and remote is None)
                or (phase in {'ISSUING','NOT_SENT'} and remote is not None)
                or type(limit) is not int or limit <= 0 or len(ciphertext) > 4 * limit + 128):
            raise MCPTaskIntegrityError('Invalid MCP Task custody record')
        try:
            raw = AEADCipher(self._root).decrypt(bytes(ciphertext), aad=self._aad(row[:-1]))
        except DecryptionError:
            raise MCPTaskIntegrityError('Original MCP Task custody cannot be opened') from None
        try:
            value = json.loads(raw)
            if (len(raw) > limit or type(value) is not dict
                    or set(value) != {'binding','remote_task_id','envelope'}):
                raise ValueError('Invalid private payload')
            binding = MCPTaskBinding(**value['binding'])
            self._envelope(phase, value['remote_task_id'], value['envelope'])
        except (ValueError, TypeError):
            raise MCPTaskIntegrityError('Invalid private MCP Task payload') from None
        if (binding.agent_did != self.agent_did or digest != self._binding_digest(binding)
                or operation != self._digest('mcp-task-operation-v1', binding.operation_id)
                or remote != self._remote_digest(binding, value['remote_task_id'])):
            raise MCPTaskIntegrityError('Original MCP Task custody indexes are invalid')
        return MCPTaskRecord(handle, revision, phase, limit, binding, raw)

    async def _all(self):
        # Open every retained row before trusting phase/uniqueness projections.
        rows = await self.database.fetchall('SELECT scope,handle,operation_digest,binding_digest,remote_digest,revision,phase,claim_hash,payload_limit,payload FROM mcp_task_records WHERE scope=? ORDER BY handle', (self.scope,))
        records = [(self._open(row), row) for row in rows]
        operations = set()
        for record, _ in records:
            for operation in (record.binding.operation_id, *record.control['mutations']):
                if operation in operations:
                    raise MCPTaskIntegrityError('MCP Task operation has conflicting original custody')
                operations.add(operation)
        return records

    async def _lock(self):
        changed = await self.database.execute('UPDATE mcp_task_keys SET revision=revision+1 WHERE scope=?', (self.scope,))
        if changed != 1:
            raise MCPTaskIntegrityError('Original MCP Task admission scope is missing')

    async def lookup(self, binding: MCPTaskBinding) -> MCPTaskRecord | None:
        self._ready()
        self._binding_digest(binding)
        for record, _ in await self._all():
            if record.binding.operation_id == binding.operation_id:
                if record.binding != binding:
                    raise MCPTaskConflict('MCP Task original operation has another binding')
                return record
        return None

    async def read(self, binding: MCPTaskBinding, handle: str) -> MCPTaskRecord:
        record = await self.lookup(binding)
        if record is None or record.handle != handle:
            raise MCPTaskConflict('MCP Task custody is absent or belongs to another operation')
        return record

    async def list_attached(self) -> tuple[MCPTaskRecord, ...]:
        """Private restart inventory; the owning host must authorize every poll."""
        self._ready()
        return tuple(record for record, _ in await self._all() if record.phase == 'ATTACHED')

    async def reserve(self, binding: MCPTaskBinding) -> MCPTaskReservation:
        self._ready()
        digest = self._binding_digest(binding)
        async with self.write_transaction():
            await self._lock()
            records = await self._all()
            for record, _ in records:
                if record.binding.operation_id == binding.operation_id:
                    if record.binding != binding:
                        raise MCPTaskConflict('MCP Task original operation has another binding')
                    existing = record
                    break
            else:
                existing = None
            if existing is None:
                if any(binding.operation_id in record.control['mutations'] for record, _ in records):
                    raise MCPTaskConflict('MCP Task original operation already owns a mutation')
                if len(records) >= self.max_records:
                    raise MCPTaskConflict('MCP Task retained record capacity is exhausted')
                if sum(record.phase in {'ISSUING','ATTACHED'} for record, _ in records) >= self.max_active:
                    raise MCPTaskConflict('MCP Task active capacity is exhausted')
                handle, secret = str(uuid4()), secrets.token_urlsafe(32)
                columns = (self.scope, handle, self._digest('mcp-task-operation-v1', binding.operation_id),
                    digest, None, 0, 'ISSUING', hashlib.sha256(secret.encode()).hexdigest(), self.max_payload_bytes)
                raw, ciphertext = self._encode(binding, columns, None, None)
                changed = await self.database.execute('INSERT INTO mcp_task_records(scope,handle,operation_digest,binding_digest,remote_digest,revision,phase,claim_hash,payload_limit,payload) VALUES (?,?,?,?,?,?,?,?,?,?)', (*columns, ciphertext))
                if changed != 1:
                    raise MCPTaskIntegrityError('Original MCP Task reservation was not stored')
                record = MCPTaskRecord(handle, 0, 'ISSUING', self.max_payload_bytes, binding, raw)
        if existing is not None:
            return MCPTaskReservation(existing)
        return MCPTaskReservation(record, MCPTaskIssuance(record, secret))

    async def _finish(self, issuance, phase, task_id, envelope):
        self._ready()
        if (type(issuance) is not MCPTaskIssuance or type(issuance.record) is not MCPTaskRecord
                or issuance.record.phase != 'ISSUING' or not isinstance(issuance.secret, str)
                or len(issuance.secret) != 43):
            raise MCPTaskConflict('MCP Task requires its original issuance claim')
        expected = issuance.record
        self._binding_digest(expected.binding)
        envelope = self._snapshot(envelope)
        claim_hash = hashlib.sha256(issuance.secret.encode()).hexdigest()
        async with self.write_transaction():
            await self._lock()
            records = await self._all()
            selected = [(record, row) for record, row in records if record.handle == expected.handle]
            if len(selected) != 1:
                raise MCPTaskConflict('MCP Task original issuance is absent')
            actual, row = selected[0]
            if (actual.binding != expected.binding or actual.revision != expected.revision
                    or actual.phase != 'ISSUING' or not hmac.compare_digest(row[7], claim_hash)):
                raise MCPTaskConflict('MCP Task issuance is no longer current')
            remote = self._remote_digest(actual.binding, task_id)
            if remote is not None and any(other[4] == remote for _, other in records):
                raise MCPTaskConflict('MCP Task remote identity already has original custody')
            columns = (*row[:4], remote, actual.revision + 1, phase, None, actual.payload_limit)
            raw, ciphertext = self._encode(actual.binding, columns, task_id, envelope)
            changed = await self.database.execute('UPDATE mcp_task_records SET remote_digest=?,revision=?,phase=?,claim_hash=NULL,payload=? WHERE scope=? AND handle=? AND revision=? AND phase=\'ISSUING\' AND claim_hash=?',
                (remote, actual.revision + 1, phase, ciphertext, self.scope, actual.handle, actual.revision, claim_hash))
            if changed != 1:
                raise MCPTaskConflict('MCP Task issuance was already consumed')
            result = MCPTaskRecord(actual.handle, actual.revision + 1, phase, actual.payload_limit, actual.binding, raw)
        return result

    async def attach_task(self, issuance: MCPTaskIssuance, *, remote_task_id: str, envelope: dict) -> MCPTaskRecord:
        """Bind a feature-validated native claim without exposing its raw payload."""
        if type(envelope) is not dict or set(envelope) != {'version','native'}:
            raise ValueError('Initial MCP Task attachment requires only its native creation claim')
        self._envelope('ATTACHED', remote_task_id, envelope)
        return await self._finish(issuance, 'ATTACHED', remote_task_id, envelope)

    async def _persist(self, actual, row, phase, envelope):
        columns = (*row[:5], actual.revision + 1, phase, None, actual.payload_limit)
        raw, ciphertext = self._encode(actual.binding, columns, actual.remote_task_id, envelope)
        changed = await self.database.execute('UPDATE mcp_task_records SET revision=?,phase=?,payload=? WHERE scope=? AND handle=? AND revision=? AND phase=?',
            (actual.revision + 1, phase, ciphertext, self.scope, actual.handle, actual.revision, actual.phase))
        if changed != 1:
            raise MCPTaskConflict('MCP Task original custody write was not stored')
        return MCPTaskRecord(actual.handle, actual.revision + 1, phase, actual.payload_limit, actual.binding, raw)

    def _expected(self, expected):
        if (type(expected) is not MCPTaskRecord or type(expected.revision) is not int
                or expected.revision < 0 or not isinstance(expected.handle, str)):
            raise MCPTaskConflict('MCP Task requires its original custody record')
        self._binding_digest(expected.binding)

    @staticmethod
    def _select(records, expected):
        selected = [(record, row) for record, row in records if record.handle == expected.handle]
        if len(selected) != 1 or selected[0][0].binding != expected.binding:
            raise MCPTaskConflict('MCP Task original custody is absent or has another binding')
        return selected[0]

    async def reserve_mutation(self, expected: MCPTaskRecord, *, operation_id: str,
                               kind: Literal['update', 'cancel'], input_responses: dict | None = None
                               ) -> MCPTaskMutationReservation:
        """Consume an authorized intent before wire, never recover its secret.

        This private repository establishes no permission. The host must bind
        operation_id to a fresh actual governed leaf and validate native input.
        Even an unknown/no-send outcome keeps all admitted identities consumed.
        """
        self._ready()
        self._expected(expected)
        _binding_text(operation_id)
        if kind not in ('update', 'cancel') or (kind == 'cancel' and input_responses is not None):
            raise ValueError('Invalid MCP Task mutation kind or responses')
        responses = self._snapshot(input_responses)
        if kind == 'update' and (type(responses) is not dict or not responses
                or any(not isinstance(key, str) or type(value) is not dict for key, value in responses.items())):
            raise ValueError('MCP Task update requires nonempty native input responses')
        async with self.write_transaction():
            await self._lock()
            records = await self._all()
            actual, row = self._select(records, expected)
            if actual.remote_task_id is None:
                raise MCPTaskConflict('MCP Task mutation requires original remote custody')
            params = {'taskId': actual.remote_task_id}
            if kind == 'update':
                params['inputResponses'] = responses
            intent = {'method': 'tasks/' + kind, 'params': params}
            control = actual.control
            for record, _ in records:
                if record.binding.operation_id == operation_id or (
                        record.handle != actual.handle and operation_id in record.control['mutations']):
                    raise MCPTaskConflict('MCP Task mutation operation already has original custody')
            previous = control['mutations'].get(operation_id)
            if previous is not None:
                if not self._same_json(previous['intent'], intent):
                    raise MCPTaskConflict('MCP Task original mutation has another intent')
                existing = actual
            else:
                existing = None
                if actual.phase != 'ATTACHED' or actual.revision != expected.revision:
                    raise MCPTaskConflict('MCP Task mutation revision is no longer current')
                if kind == 'update' and not set(responses) <= set(actual.pending_inputs):
                    raise MCPTaskConflict('MCP Task original input is absent or already consumed')
                secret = secrets.token_urlsafe(32)
                control['mutations'][operation_id] = {'intent': intent,
                    'claim_hash': hashlib.sha256(secret.encode()).hexdigest(), 'acknowledgement': None}
                envelope = actual.envelope
                envelope['control'] = control
                result = await self._persist(actual, row, actual.phase, envelope)
        if existing is not None:
            return MCPTaskMutationReservation(existing)
        return MCPTaskMutationReservation(result, MCPTaskMutationIssuance(result, operation_id, secret))

    async def acknowledge_mutation(self, issuance: MCPTaskMutationIssuance, acknowledgement: dict) -> MCPTaskRecord:
        """Retain an exact native acknowledgement, never infer task completion."""
        self._ready()
        if (type(issuance) is not MCPTaskMutationIssuance or not isinstance(issuance.secret, str)
                or len(issuance.secret) != 43):
            raise MCPTaskConflict('MCP Task requires its original mutation claim')
        self._expected(issuance.record)
        _binding_text(issuance.operation_id)
        ack = self._snapshot(acknowledgement)
        if type(ack) is not dict or ack.get('resultType') != 'complete':
            raise ValueError('MCP Task requires a native mutation acknowledgement')
        claim_hash = hashlib.sha256(issuance.secret.encode()).hexdigest()
        async with self.write_transaction():
            await self._lock()
            actual, row = self._select(await self._all(), issuance.record)
            envelope = actual.envelope
            if actual.remote_task_id is None:
                raise MCPTaskConflict('MCP Task original mutation custody is absent')
            control = actual.control
            receipt = control['mutations'].get(issuance.operation_id)
            if (receipt is None or receipt['claim_hash'] is None
                    or not hmac.compare_digest(receipt['claim_hash'], claim_hash)):
                raise MCPTaskConflict('MCP Task mutation claim is absent or already finished')
            receipt['acknowledgement'], receipt['claim_hash'] = ack, None
            envelope['control'] = control
            result = await self._persist(actual, row, actual.phase, envelope)
        return result

    async def observe(self, expected: MCPTaskRecord, observation: dict) -> MCPTaskRecord:
        """Retain a feature-validated native get result, never an inferred finish.

        The host authorizes the observer independently. This is strict CAS,
        including for identical repeats; uncertainty reconciles through read.
        """
        self._ready()
        if (type(expected) is not MCPTaskRecord or expected.phase != 'ATTACHED'
                or type(expected.revision) is not int or expected.revision < 0):
            raise MCPTaskConflict('MCP Task observation requires original attached custody')
        self._binding_digest(expected.binding)
        observation = self._snapshot(observation)
        async with self.write_transaction():
            await self._lock()
            selected = [(record, row) for record, row in await self._all() if record.handle == expected.handle]
            if len(selected) != 1:
                raise MCPTaskConflict('MCP Task original observation custody is absent')
            actual, row = selected[0]
            if (actual.binding != expected.binding or actual.revision != expected.revision
                    or actual.phase != 'ATTACHED'):
                raise MCPTaskConflict('MCP Task observation revision is no longer current')
            terminal = self._observation(actual.remote_task_id, observation)
            phase = 'SETTLED' if terminal else 'ATTACHED'
            envelope = actual.envelope
            envelope['observation'] = observation
            self._retain_inputs(envelope, observation)
            result = await self._persist(actual, row, phase, envelope)
        return result

    async def settle_initial(self, issuance: MCPTaskIssuance, *,
                             disposition: Literal['SETTLED','NOT_SENT'], envelope: dict) -> MCPTaskRecord:
        """Retain consumed identity after an ordinary result or proven refusal.

        A timeout, cancellation or unconfirmed submission must stay ISSUING.
        The host, not this private JSON repository, establishes outcome evidence.
        """
        if disposition not in {'SETTLED','NOT_SENT'}:
            raise MCPTaskConflict('MCP Task initial settlement requires confirmed evidence')
        self._envelope(disposition, None, envelope)
        return await self._finish(issuance, disposition, None, envelope)
