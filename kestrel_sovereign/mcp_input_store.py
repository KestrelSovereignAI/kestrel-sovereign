"""Encrypted MCP continuation custody, separate from permission to execute.

The host authenticates and authorizes every reader and response. These records
bind an original request; neither their identifiers nor a response claim grant
tool authority. A response claimed before a lost acknowledgement is never
automatically reclaimed or replayed.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import math
import secrets
from uuid import uuid4

from kestrel_sdk.resources import ResourceError
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.encryption import get_agent_key, get_fernet
from kestrel_sdk.security.exceptions import DecryptionError
from kestrel_sovereign.a2a.media import encode_inline_json
from kestrel_sovereign.a2a.stores.unified.base import NativeStoreConflict, UnifiedStoreBase
from kestrel_sovereign.security.envelope_keys import resolve_envelope_key
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.initialization_errors import repository_initialization_errors

MCP_INPUT_SCHEMA_REVISION = 2
MCP_INPUT_WRAP_VERSION = 2


class MCPInputConflict(NativeStoreConflict):
    """The original waiting revision no longer accepts this response."""


class MCPInputIntegrityError(ResourceError):
    """The original continuation cannot safely be opened."""


def mcp_input_envelope_aad(scope: str) -> bytes:
    """One scope/version context for initialization, migration and rotation."""
    return encode_inline_json(['mcp-input-root', MCP_INPUT_WRAP_VERSION, scope])


def _text(value: str) -> None:
    if (not isinstance(value, str) or not value or len(value.encode('utf-8')) > 1024
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('MCP input bindings require bounded control-free strings')


def _time(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('MCP input timestamps must be finite numbers')


@dataclass(frozen=True, repr=False)
class MCPInputBinding:
    agent_did: str
    principal_id: str
    session_id: str
    operation_id: str
    connection_id: str
    contract_digest: str
    method: str
    custody: str | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == 'custody' and value is None:
                continue
            _text(value)
        if not self.agent_did.startswith('did:'):
            raise ValueError('MCP input binding requires an agent DID')
        if self.method not in {'tools/call', 'resources/read', 'prompts/get'}:
            raise ValueError('MCP input binding has an unsupported operation')
        if len(self.contract_digest) != 64 or any(c not in '0123456789abcdef' for c in self.contract_digest):
            raise ValueError('MCP input binding requires the original contract digest')


def _binding_value(binding: MCPInputBinding) -> dict:
    """Keep ordinary schema-2 binding bytes unchanged; bind every Task link."""
    value = asdict(binding)
    if value['custody'] is None:
        del value['custody']
    return value


@dataclass(frozen=True)
class MCPInputRecord:
    request_id: str
    revision: int
    state: str
    expires_at: float
    binding: MCPInputBinding = field(repr=False)
    _payload: bytes = field(repr=False)
    reservation_id: str = field(repr=False)
    predecessor_id: str | None = field(repr=False)

    @property
    def payload(self) -> dict:
        """Return a detached copy, never mutable storage or a conversation part."""
        return json.loads(self._payload)['checkpoint']

    @property
    def responses(self) -> dict | None:
        return json.loads(self._payload)['responses']

    @property
    def continuation(self) -> dict | None:
        return json.loads(self._payload)['continuation']


@dataclass(frozen=True, slots=True, repr=False)
class _InputExpectation:
    request_id: str
    revision: int
    state: str
    expires_at: float
    binding: MCPInputBinding


def _expectation(record: MCPInputRecord) -> _InputExpectation:
    """Capture comparison metadata once; caller payloads never grant authority."""
    if type(record) is not MCPInputRecord or type(record.binding) is not MCPInputBinding:
        raise MCPInputConflict('MCP input requires its original checkpoint identity')
    _text(record.request_id)
    _time(record.expires_at)
    if type(record.revision) is not int or record.revision < 0:
        raise MCPInputConflict('MCP input requires its original checkpoint revision')
    return _InputExpectation(record.request_id, record.revision, record.state, record.expires_at,
        MCPInputBinding(**_binding_value(record.binding)))


@dataclass(frozen=True)
class MCPInputClaim:
    record: MCPInputRecord
    secret: str = field(repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedPredecessor:
    identity: _InputExpectation
    binding_digest: str
    claim_hash: str
    before: tuple


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedCreate:
    owner: object
    record: MCPInputRecord
    operation_digest: str
    reader_digest: str
    binding_digest: str
    ciphertext: bytes
    now: float
    predecessor: _PreparedPredecessor | None


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedResponse:
    owner: object
    claim: MCPInputClaim
    reader_digest: str
    binding_digest: str
    claim_hash: str
    ciphertext: bytes
    before: tuple
    now: float


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedExpiry:
    owner: object
    record: MCPInputRecord
    reader_digest: str
    binding_digest: str
    ciphertext: bytes
    before: tuple
    now: float


class MCPInputStore(UnifiedStoreBase):
    """Borrow primary storage and atomically consume each waiting revision."""

    def __init__(self, database: AsyncDatabase, *, agent_did: str,
                 max_pending: int, max_payload_bytes: int):
        super().__init__(database.backend, owns_backend=False)
        _text(agent_did)
        if not agent_did.startswith('did:'):
            raise ValueError('MCP input store requires an agent DID')
        for limit in (max_pending, max_payload_bytes):
            if type(limit) is not int or limit <= 0:
                raise ValueError('MCP input storage budgets must be positive integers')
        self.database = database
        self.agent_did = agent_did
        self.scope = hashlib.sha256(agent_did.encode()).hexdigest()
        self.max_pending = max_pending
        self.max_payload_bytes = max_payload_bytes
        self._root: bytes | None = None
        self._initialization_lock = asyncio.Lock()

    def _ready(self):
        if self.database.owns_open_transaction is not False:
            raise MCPInputConflict('MCP input custody requires its own durable transaction')
        if self._root is None:
            raise RuntimeError('MCP input store is not initialized')

    async def initialize(self):
        async with self._initialization_lock:
            if self._root is not None:
                return
            if self.database.owns_open_transaction is not False:
                raise MCPInputConflict('MCP input initialization requires its own transaction')
            cipher = get_fernet()
            if cipher is None:
                raise MCPInputIntegrityError('MCP input custody requires encryption')
            binary = 'BYTEA' if self.is_postgres else 'BLOB'
            if not (self.is_postgres or self.is_sqlite):
                raise ValueError('MCP input custody requires SQLite or PostgreSQL')
            with repository_initialization_errors(integrity_error=MCPInputIntegrityError,
                    unknown_message='MCP input initialization requires reconciliation'):
                async with self.database.migration_lock('kestrel-mcp-input-schema-v1'):
                    await self.database.execute('CREATE TABLE IF NOT EXISTS mcp_input_schema (singleton BIGINT PRIMARY KEY, revision BIGINT NOT NULL)')
                    revision = await self.database.fetchone('SELECT revision FROM mcp_input_schema WHERE singleton=1')
                    if revision is None:
                        await self.database.execute(f'CREATE TABLE mcp_input_scopes (scope TEXT PRIMARY KEY, wrapped_key {binary} NOT NULL, revision BIGINT NOT NULL, observed_at DOUBLE PRECISION, wrap_version BIGINT NOT NULL)')
                        await self.database.execute(f"CREATE TABLE mcp_inputs (scope TEXT NOT NULL, request_id TEXT NOT NULL, operation_digest TEXT NOT NULL, reader_digest TEXT NOT NULL, binding_digest TEXT NOT NULL, revision BIGINT NOT NULL, state TEXT NOT NULL CHECK(state IN ('WAITING','RESPONDING','RESOLVED','NON_RESUMABLE','UNKNOWN')), expires_at DOUBLE PRECISION NOT NULL, claim_hash TEXT, reservation_id TEXT NOT NULL, predecessor_id TEXT, payload {binary} NOT NULL, PRIMARY KEY(scope,request_id), UNIQUE(scope,operation_digest), UNIQUE(scope,predecessor_id), FOREIGN KEY(scope,predecessor_id) REFERENCES mcp_inputs(scope,request_id), FOREIGN KEY(scope) REFERENCES mcp_input_scopes(scope))")
                        await self.database.execute('CREATE INDEX mcp_inputs_reader ON mcp_inputs(scope,reader_digest,state)')
                        await self.database.execute('CREATE TABLE mcp_input_decisions (scope TEXT NOT NULL, decision_id TEXT NOT NULL, operation_digest TEXT NOT NULL, input_digest TEXT NOT NULL, reason TEXT NOT NULL, decided_at DOUBLE PRECISION NOT NULL, PRIMARY KEY(scope,decision_id), FOREIGN KEY(scope) REFERENCES mcp_input_scopes(scope))')
                        await self.database.execute('INSERT INTO mcp_input_schema(singleton,revision) VALUES (1,?)', (MCP_INPUT_SCHEMA_REVISION,))
                    elif revision == (1,):
                        await self.database.execute('ALTER TABLE mcp_input_scopes ADD COLUMN wrap_version BIGINT NOT NULL DEFAULT 1')
                        changed = await self.database.execute('UPDATE mcp_input_schema SET revision=? WHERE singleton=1 AND revision=1', (MCP_INPUT_SCHEMA_REVISION,))
                        if changed != 1:
                            raise MCPInputIntegrityError('MCP input schema migration was not stored')
                    elif revision != (MCP_INPUT_SCHEMA_REVISION,):
                        raise MCPInputIntegrityError('Unsupported MCP input schema revision')
                    await self.database.fetchone('SELECT scope,wrapped_key,revision,observed_at,wrap_version FROM mcp_input_scopes WHERE 1=0')
                    await self.database.fetchone('SELECT scope,request_id,operation_digest,reader_digest,binding_digest,revision,state,expires_at,claim_hash,reservation_id,predecessor_id,payload FROM mcp_inputs WHERE 1=0')
                    await self.database.fetchone('SELECT scope,decision_id,operation_digest,input_digest,reason,decided_at FROM mcp_input_decisions WHERE 1=0')
                    row = await self.database.fetchone('SELECT wrapped_key,wrap_version FROM mcp_input_scopes WHERE scope=?', (self.scope,))
                    dependent = await self.database.fetchone('SELECT 1 FROM mcp_inputs WHERE scope=? LIMIT 1', (self.scope,))
                    if dependent is None:
                        dependent = await self.database.fetchone('SELECT 1 FROM mcp_input_decisions WHERE scope=? LIMIT 1', (self.scope,))
                    if row is not None and (type(row[1]) is not int or row[1] not in {1, MCP_INPUT_WRAP_VERSION}):
                        raise MCPInputIntegrityError('Unsupported MCP input envelope version')
                    legacy = row is not None and row[1] == 1
                    opening_cipher = AEADCipher(get_agent_key(self.agent_did, 'service-keys')) if legacy else cipher
                    root, envelope = resolve_envelope_key(opening_cipher, bytes(row[0]) if row else None,
                        has_dependents=dependent is not None, integrity_error=MCPInputIntegrityError,
                        missing_message='Original MCP input envelope is missing',
                        unavailable_message='Original MCP input encryption key is unavailable',
                        invalid_message='Invalid MCP input envelope',
                        aad=None if legacy else mcp_input_envelope_aad(self.scope))
                    if legacy:
                        # A finite migration of the known original format, never
                        # a decryption fallback for an already migrated scope.
                        envelope = cipher.encrypt(root, aad=mcp_input_envelope_aad(self.scope))
                        changed = await self.database.execute('UPDATE mcp_input_scopes SET wrapped_key=?,wrap_version=? WHERE scope=? AND wrap_version=1', (envelope, MCP_INPUT_WRAP_VERSION, self.scope))
                    elif envelope is not None:
                        changed = await self.database.execute('INSERT INTO mcp_input_scopes(scope,wrapped_key,revision,wrap_version) VALUES (?,?,0,?)', (self.scope, envelope, MCP_INPUT_WRAP_VERSION))
                    if envelope is not None:
                        stored = await self.database.fetchone('SELECT wrapped_key,wrap_version FROM mcp_input_scopes WHERE scope=?', (self.scope,))
                        if changed != 1 or stored is None or bytes(stored[0]) != envelope or stored[1] != MCP_INPUT_WRAP_VERSION:
                            raise MCPInputIntegrityError('Original MCP input envelope migration was not stored')
            self._root = root

    def _binding_digest(self, binding):
        self._ready()
        if type(binding) is not MCPInputBinding or binding.agent_did != self.agent_did:
            raise MCPInputConflict('MCP input binding belongs to another agent')
        return hmac.new(self._root, encode_inline_json(_binding_value(binding), sort_keys=True), hashlib.sha256).hexdigest()

    def _aad(self, request_id, digest, revision, state, expires_at, claim_hash, reservation_id, predecessor_id):
        return encode_inline_json(['mcp-input-v1', self.scope, request_id, digest,
            revision, state, expires_at, claim_hash, reservation_id, predecessor_id])

    def _reader_digest(self, principal_id, session_id):
        self._ready()
        _text(principal_id)
        _text(session_id)
        return hmac.new(self._root, encode_inline_json(
            ['mcp-input-reader-v1', principal_id, session_id]), hashlib.sha256).hexdigest()

    def _encode(self, binding, request_id, revision, state, expires_at, claim_hash, payload, responses=None,
                continuation=None, *, reservation_id, predecessor_id):
        if (type(payload) is not dict or (responses is not None and type(responses) is not dict)
                or (continuation is not None and (type(continuation) is not dict or state != 'RESOLVED'))):
            raise TypeError('MCP input checkpoint must be a JSON object')
        raw = encode_inline_json({'binding': _binding_value(binding), 'checkpoint': payload, 'responses': responses,
            'continuation': continuation})
        if len(raw) > self.max_payload_bytes:
            raise ValueError('MCP input checkpoint exceeded its storage budget')
        digest = self._binding_digest(binding)
        return AEADCipher(self._root).encrypt(raw,
            aad=self._aad(request_id, digest, revision, state, expires_at, claim_hash,
                reservation_id, predecessor_id))

    async def read(self, binding: MCPInputBinding, request_id: str) -> MCPInputRecord:
        digest = self._binding_digest(binding)
        record = await self.read_for(binding.principal_id, binding.session_id, request_id)
        if not hmac.compare_digest(self._binding_digest(record.binding), digest):
            raise MCPInputConflict('MCP input request belongs to another operation')
        return record

    async def read_for(self, principal_id: str, session_id: str, request_id: str) -> MCPInputRecord:
        """Read only after the host independently authenticates this reader."""
        record, _ = await self._read_for(principal_id, session_id, request_id)
        return record

    async def _read_for(self, principal_id, session_id, request_id):
        reader_digest = self._reader_digest(principal_id, session_id)
        _text(request_id)
        row = await self.database.fetchone('SELECT binding_digest,revision,state,expires_at,claim_hash,reservation_id,predecessor_id,payload FROM mcp_inputs WHERE scope=? AND request_id=? AND reader_digest=?', (self.scope, request_id, reader_digest))
        if row is None:
            raise MCPInputConflict('MCP input request is absent or belongs to another operation')
        digest, revision, state, expires_at, claim_hash, reservation_id, predecessor_id, ciphertext = row
        if (type(revision) is not int or revision < 0 or state not in
                {'WAITING','RESPONDING','RESOLVED','NON_RESUMABLE','UNKNOWN'}
                or ((state == 'RESPONDING') != (claim_hash is not None))):
            raise MCPInputIntegrityError('Invalid MCP input custody record')
        _time(expires_at)
        _text(reservation_id)
        if predecessor_id is not None:
            _text(predecessor_id)
        if ((predecessor_id is None) != (reservation_id == request_id)
                or predecessor_id == request_id):
            raise MCPInputIntegrityError('Invalid MCP input reservation lineage')
        if len(ciphertext) > 4 * self.max_payload_bytes + 128:
            raise MCPInputIntegrityError('Stored MCP input exceeded its storage budget')
        try:
            raw = AEADCipher(self._root).decrypt(bytes(ciphertext),
                aad=self._aad(request_id, digest, revision, state, expires_at, claim_hash,
                    reservation_id, predecessor_id))
        except DecryptionError:
            raise MCPInputIntegrityError('Original MCP input checkpoint cannot be opened') from None
        value = json.loads(raw)
        if (len(raw) > self.max_payload_bytes or type(value) is not dict
                or set(value) != {'binding', 'checkpoint', 'responses', 'continuation'} or type(value['checkpoint']) is not dict
                or (value['responses'] is not None and type(value['responses']) is not dict)
                or (value['continuation'] is not None and (type(value['continuation']) is not dict or state != 'RESOLVED'))):
            raise MCPInputIntegrityError('Invalid MCP input checkpoint')
        binding = MCPInputBinding(**value['binding'])
        if (binding.principal_id != principal_id or binding.session_id != session_id
                or not hmac.compare_digest(self._binding_digest(binding), digest)):
            raise MCPInputIntegrityError('Original MCP input reader binding is invalid')
        return (MCPInputRecord(request_id, revision, state, expires_at, binding, raw,
            reservation_id, predecessor_id), tuple(row))

    async def list_for(self, principal_id: str, session_id: str, *, now: float) -> tuple[MCPInputRecord, ...]:
        """Expose waiting/uncertain metadata to the independently authorized UI."""
        digest = self._reader_digest(principal_id, session_id)
        _time(now)
        # Capacity bounds chains. Project one current card per chain too, so
        # an unsettled predecessor cannot hide its confirmed waiting successor.
        rows = await self.database.fetchall("WITH candidates AS (SELECT request_id,CASE WHEN state='WAITING' THEN 0 WHEN state='RESPONDING' THEN 1 ELSE 2 END AS priority,ROW_NUMBER() OVER (PARTITION BY reservation_id ORDER BY CASE WHEN state='WAITING' THEN 0 WHEN state='RESPONDING' THEN 1 ELSE 2 END,request_id) AS chain_position FROM mcp_inputs WHERE scope=? AND reader_digest=? AND (state IN ('RESPONDING','UNKNOWN') OR (state='WAITING' AND expires_at>?))) SELECT request_id FROM candidates WHERE chain_position=1 ORDER BY priority,request_id LIMIT ?", (self.scope, digest, now, self.max_pending))
        return tuple([await self.read_for(principal_id, session_id, row[0]) for row in rows])

    async def audit_decisions(self, operation_id: str, checkpoint_digest: str,
                             decisions: tuple[tuple[str, str], ...], *, now: float) -> None:
        """Acknowledge content-free policy decisions before standard declines."""
        self._ready()
        _text(operation_id)
        _text(checkpoint_digest)
        _time(now)
        reasons = {'url_mode_declined', 'no_approver', 'unsupported_input', 'no_restartable_state'}
        if (type(decisions) is not tuple or any(type(item) is not tuple or len(item) != 2
                or not isinstance(item[0], str) or item[1] not in reasons for item in decisions)):
            raise ValueError('Invalid MCP input audit decisions')
        operation = hmac.new(self._root, encode_inline_json(['mcp-input-audit-operation-v1', operation_id]), hashlib.sha256).hexdigest()
        entries = [(str(uuid4()), hmac.new(self._root,
            encode_inline_json(['mcp-input-audit-key-v1', checkpoint_digest, key]), hashlib.sha256).hexdigest(), reason)
            for key, reason in decisions]
        async with self.write_transaction():
            for decision_id, input_digest, reason in entries:
                await self.database.execute('INSERT INTO mcp_input_decisions(scope,decision_id,operation_digest,input_digest,reason,decided_at) VALUES (?,?,?,?,?,?)', (self.scope, decision_id, operation, input_digest, reason, now))

    async def _lock_admission(self, now: float) -> float:
        """Serialize writers and make observed expiry irreversible in this scope.

        A request timestamp may predate its blocked write transaction. Keep the
        latest admitted time in the same transaction as the reservation change,
        so a delayed response cannot revive a reservation a creator has reused.
        Existing reservations remain admissible when an operator lowers quota.
        """
        changed = await self.database.execute('UPDATE mcp_input_scopes SET revision=revision+1,observed_at=CASE WHEN observed_at IS NULL OR observed_at<? THEN ? ELSE observed_at END WHERE scope=?', (now, now, self.scope))
        if changed != 1:
            raise MCPInputIntegrityError('Original MCP input admission scope is missing')
        row = await self.database.fetchone('SELECT observed_at FROM mcp_input_scopes WHERE scope=?', (self.scope,))
        if row is None or isinstance(row[0], bool) or not isinstance(row[0], (int, float)) or not math.isfinite(row[0]):
            raise MCPInputIntegrityError('Original MCP input admission time is invalid')
        return float(row[0])

    def _require_apply(self, prepared, expected_type):
        if (type(prepared) is not expected_type or prepared.owner is not self or self._root is None
                or self.database.owns_open_transaction is not True):
            raise MCPInputConflict('MCP input prepared write requires its owning transaction')

    def _prepare_record(self, binding, request_id, revision, state, expires_at, claim_hash,
                        payload, responses=None, continuation=None, *, reservation_id, predecessor_id):
        ciphertext = self._encode(binding, request_id, revision, state, expires_at,
            claim_hash, payload, responses, continuation, reservation_id=reservation_id,
            predecessor_id=predecessor_id)
        raw = AEADCipher(self._root).decrypt(ciphertext, aad=self._aad(request_id,
            self._binding_digest(binding), revision, state, expires_at, claim_hash,
            reservation_id, predecessor_id))
        return (MCPInputRecord(request_id, revision, state, expires_at, binding, raw,
            reservation_id, predecessor_id), ciphertext)

    async def _prepare_create(self, binding: MCPInputBinding, payload: dict, *, expires_at: float,
                              now: float, predecessor: MCPInputClaim | None = None) -> _PreparedCreate:
        self._ready()
        _time(now)
        _time(expires_at)
        if expires_at <= now:
            raise MCPInputConflict('MCP input checkpoint has already expired')
        expires_at = float(expires_at)
        request_id = str(uuid4())
        if type(binding) is MCPInputBinding:
            binding = MCPInputBinding(**_binding_value(binding))
        digest = self._binding_digest(binding)
        if type(payload) is not dict:
            raise TypeError('MCP input checkpoint must be a JSON object')
        payload = json.loads(encode_inline_json(payload))
        operation_digest = hmac.new(self._root, encode_inline_json(
            ['mcp-input-operation-v1', binding.operation_id]), hashlib.sha256).hexdigest()
        reader_digest = self._reader_digest(binding.principal_id, binding.session_id)
        reservation_id, predecessor_id, predecessor_guard = request_id, None, None
        if predecessor is not None:
            parent, parent_digest, parent_hash = self._claim_identity(predecessor)
            original, before = await self._read_for(parent.binding.principal_id,
                parent.binding.session_id, parent.request_id)
            old_binding, new_binding = _binding_value(original.binding), _binding_value(binding)
            old_binding.pop('operation_id')
            new_binding.pop('operation_id')
            if (original.binding != parent.binding or original.state != 'RESPONDING'
                    or original.revision != parent.revision or not hmac.compare_digest(before[4], parent_hash)
                    or old_binding != new_binding or parent.binding.operation_id == binding.operation_id):
                raise MCPInputConflict('MCP input successor changed its original response binding')
            reservation_id, predecessor_id = original.reservation_id, original.request_id
            predecessor_guard = _PreparedPredecessor(parent, parent_digest, parent_hash, before)
        record, ciphertext = self._prepare_record(binding, request_id, 0, 'WAITING', expires_at, None, payload,
            reservation_id=reservation_id, predecessor_id=predecessor_id)
        return _PreparedCreate(self, record, operation_digest, reader_digest, digest, ciphertext,
            float(now), predecessor_guard)

    async def _apply_create(self, prepared: _PreparedCreate) -> MCPInputRecord:
        """SQL-only half, shared by ordinary creation and the joint Task commit."""
        self._require_apply(prepared, _PreparedCreate)
        record = prepared.record
        admitted_at = await self._lock_admission(prepared.now)
        if record.expires_at <= admitted_at:
            raise MCPInputConflict('MCP input checkpoint has already expired')
        count = await self.database.fetchone("SELECT COUNT(DISTINCT reservation_id) FROM mcp_inputs WHERE scope=? AND (state='RESPONDING' OR (state='WAITING' AND expires_at>?))", (self.scope, admitted_at))
        if prepared.predecessor is not None:
            parent = prepared.predecessor
            request_id, digest, claim_hash, before = (parent.identity.request_id,
                parent.binding_digest, parent.claim_hash, parent.before)
            locked = await self.database.execute("UPDATE mcp_inputs SET revision=revision WHERE scope=? AND request_id=? AND binding_digest=? AND revision=? AND state='RESPONDING' AND claim_hash=? AND expires_at=? AND reservation_id=? AND COALESCE(predecessor_id,'')=? AND payload=?", (self.scope, request_id, digest, before[1], claim_hash, before[3], before[5], before[6] or '', before[7]))
            if locked != 1:
                raise MCPInputConflict('MCP input successor requires its original live response claim')
        elif count[0] >= self.max_pending:
            raise MCPInputConflict('MCP input pending capacity is exhausted')
        changed = await self.database.execute("INSERT INTO mcp_inputs(scope,request_id,operation_digest,reader_digest,binding_digest,revision,state,expires_at,claim_hash,reservation_id,predecessor_id,payload) VALUES (?,?,?,?,?,0,'WAITING',?,NULL,?,?,?) ON CONFLICT DO NOTHING", (self.scope, record.request_id, prepared.operation_digest, prepared.reader_digest, prepared.binding_digest, record.expires_at, record.reservation_id, record.predecessor_id, prepared.ciphertext))
        if changed != 1:
            raise MCPInputConflict('MCP input original operation or predecessor already has a checkpoint')
        return record

    async def create(self, binding: MCPInputBinding, payload: dict, *, expires_at: float,
                     now: float, predecessor: MCPInputClaim | None = None) -> MCPInputRecord:
        self._ready()
        if type(binding) is not MCPInputBinding:
            raise MCPInputConflict('MCP input requires its original binding')
        if binding.custody is not None:
            raise MCPInputConflict('Task-linked MCP input requires its joint custody owner')
        prepared = await self._prepare_create(binding, payload, expires_at=expires_at,
            now=now, predecessor=predecessor)
        async with self.write_transaction():
            result = await self._apply_create(prepared)
        return result

    async def _prepare_response(self, expected: MCPInputRecord, responses: dict, *, now: float) -> _PreparedResponse:
        self._ready()
        _time(now)
        expected = _expectation(expected)
        if expected.state != 'WAITING' or expected.expires_at <= now:
            raise MCPInputConflict('MCP input is no longer waiting or has expired')
        if type(responses) is not dict:
            raise TypeError('MCP input responses must be a JSON object')
        responses = json.loads(encode_inline_json(responses))
        self._binding_digest(expected.binding)
        actual, before = await self._read_for(expected.binding.principal_id,
            expected.binding.session_id, expected.request_id)
        if (actual.binding != expected.binding or actual.state != 'WAITING'
                or actual.revision != expected.revision or actual.expires_at != expected.expires_at):
            raise MCPInputConflict('MCP input waiting revision was already consumed')
        digest = before[0]
        secret = secrets.token_urlsafe(32)
        claim_hash = hashlib.sha256(secret.encode()).hexdigest()
        revision = actual.revision + 1
        record, ciphertext = self._prepare_record(actual.binding, actual.request_id, revision,
            'RESPONDING', actual.expires_at, claim_hash, actual.payload, responses,
            reservation_id=actual.reservation_id, predecessor_id=actual.predecessor_id)
        return _PreparedResponse(self, MCPInputClaim(record, secret),
            self._reader_digest(actual.binding.principal_id, actual.binding.session_id),
            digest, claim_hash, ciphertext, before, float(now))

    async def _apply_response(self, prepared: _PreparedResponse) -> MCPInputClaim:
        """Consume the exact authenticated before-image inside the owning commit."""
        self._require_apply(prepared, _PreparedResponse)
        admitted_at = await self._lock_admission(prepared.now)
        record, before = prepared.claim.record, prepared.before
        changed = await self.database.execute("UPDATE mcp_inputs SET revision=?,state='RESPONDING',claim_hash=?,payload=? WHERE scope=? AND request_id=? AND reader_digest=? AND binding_digest=? AND revision=? AND state='WAITING' AND claim_hash IS NULL AND expires_at=? AND expires_at>? AND reservation_id=? AND COALESCE(predecessor_id,'')=? AND payload=?", (record.revision, prepared.claim_hash, prepared.ciphertext, self.scope, record.request_id, prepared.reader_digest, prepared.binding_digest, before[1], record.expires_at, admitted_at, record.reservation_id, record.predecessor_id or '', before[7]))
        if changed != 1:
            raise MCPInputConflict('MCP input waiting revision was already consumed')
        return prepared.claim

    async def claim_response(self, expected: MCPInputRecord, responses: dict, *, now: float) -> MCPInputClaim:
        """Persist validated responses before any continuation wire request."""
        prepared = await self._prepare_response(expected, responses, now=now)
        if prepared.claim.record.binding.custody is not None:
            raise MCPInputConflict('Task-linked MCP input requires its joint custody owner')
        async with self.write_transaction():
            result = await self._apply_response(prepared)
        return result

    async def _prepare_expiry(self, expected: MCPInputRecord, *, now: float) -> _PreparedExpiry:
        self._ready()
        _time(now)
        expected = _expectation(expected)
        self._binding_digest(expected.binding)
        actual, before = await self._read_for(expected.binding.principal_id,
            expected.binding.session_id, expected.request_id)
        if (actual.binding != expected.binding or actual.state != 'WAITING'
                or actual.revision != expected.revision or actual.expires_at != expected.expires_at
                or actual.binding.custody is None):
            raise MCPInputConflict('MCP input expiry requires its original linked waiting revision')
        record, ciphertext = self._prepare_record(actual.binding, actual.request_id,
            actual.revision + 1, 'NON_RESUMABLE', actual.expires_at, None, actual.payload,
            reservation_id=actual.reservation_id, predecessor_id=actual.predecessor_id)
        return _PreparedExpiry(self, record,
            self._reader_digest(actual.binding.principal_id, actual.binding.session_id),
            before[0], ciphertext, before, float(now))

    async def _apply_expiry(self, prepared: _PreparedExpiry) -> MCPInputRecord:
        self._require_apply(prepared, _PreparedExpiry)
        admitted_at = await self._lock_admission(prepared.now)
        record, before = prepared.record, prepared.before
        changed = await self.database.execute("UPDATE mcp_inputs SET revision=?,state='NON_RESUMABLE',payload=? WHERE scope=? AND request_id=? AND reader_digest=? AND binding_digest=? AND revision=? AND state='WAITING' AND claim_hash IS NULL AND expires_at=? AND expires_at<=? AND reservation_id=? AND COALESCE(predecessor_id,'')=? AND payload=?", (record.revision, prepared.ciphertext, self.scope, record.request_id, prepared.reader_digest, prepared.binding_digest, before[1], record.expires_at, admitted_at, record.reservation_id, record.predecessor_id or '', before[7]))
        if changed != 1:
            raise MCPInputConflict('MCP input original waiting revision is not expired or was consumed')
        return record

    def _claim_identity(self, claim: MCPInputClaim):
        self._ready()
        if (type(claim) is not MCPInputClaim
                or not isinstance(claim.secret, str) or len(claim.secret) != 43):
            raise MCPInputConflict('MCP input requires its original response claim')
        record = _expectation(claim.record)
        if record.state != 'RESPONDING':
            raise MCPInputConflict('MCP input requires its original response claim')
        return record, self._binding_digest(record.binding), hashlib.sha256(claim.secret.encode()).hexdigest()

    async def settle(self, claim: MCPInputClaim, *, state: str, continuation: dict | None = None) -> MCPInputRecord:
        """Record the original claim's outcome; uncertain work never reopens."""
        record, digest, claim_hash = self._claim_identity(claim)
        if state not in {'RESOLVED','NON_RESUMABLE','UNKNOWN'}:
            raise MCPInputConflict('MCP input settlement requires a terminal state')
        if continuation is not None:
            if type(continuation) is not dict or state != 'RESOLVED':
                raise TypeError('MCP input checkpoint must be a JSON object')
            continuation = json.loads(encode_inline_json(continuation))
        actual = await self.read(record.binding, record.request_id)
        if (actual.state != 'RESPONDING' or actual.revision != record.revision
                or actual.expires_at != record.expires_at):
            raise MCPInputConflict('MCP input response claim is no longer current')
        revision = actual.revision + 1
        digest = self._binding_digest(actual.binding)
        ciphertext = self._encode(actual.binding, actual.request_id, revision, state,
            actual.expires_at, None, actual.payload, actual.responses, continuation,
            reservation_id=actual.reservation_id, predecessor_id=actual.predecessor_id)
        async with self.write_transaction():
            changed = await self.database.execute("UPDATE mcp_inputs SET revision=?,state=?,claim_hash=NULL,payload=? WHERE scope=? AND request_id=? AND binding_digest=? AND revision=? AND state='RESPONDING' AND claim_hash=?", (revision, state, ciphertext, self.scope, actual.request_id, digest, actual.revision, claim_hash))
            if changed != 1:
                raise MCPInputConflict('MCP input response claim is no longer current')
        return await self.read(actual.binding, actual.request_id)
