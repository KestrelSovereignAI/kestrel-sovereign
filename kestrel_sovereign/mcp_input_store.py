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
from kestrel_sdk.security.encryption import get_agent_key
from kestrel_sdk.security.exceptions import DecryptionError
from kestrel_sovereign.a2a.media import encode_inline_json
from kestrel_sovereign.a2a.stores.unified.base import NativeStoreConflict, UnifiedStoreBase
from kestrel_sovereign.security.envelope_keys import resolve_envelope_key
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.initialization_errors import repository_initialization_errors


class MCPInputConflict(NativeStoreConflict):
    """The original waiting revision no longer accepts this response."""


class MCPInputIntegrityError(ResourceError):
    """The original continuation cannot safely be opened."""


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

    def __post_init__(self):
        for value in asdict(self).values():
            _text(value)
        if not self.agent_did.startswith('did:'):
            raise ValueError('MCP input binding requires an agent DID')
        if self.method not in {'tools/call', 'resources/read', 'prompts/get'}:
            raise ValueError('MCP input binding has an unsupported operation')
        if len(self.contract_digest) != 64 or any(c not in '0123456789abcdef' for c in self.contract_digest):
            raise ValueError('MCP input binding requires the original contract digest')


@dataclass(frozen=True)
class MCPInputRecord:
    request_id: str
    revision: int
    state: str
    expires_at: float
    binding: MCPInputBinding = field(repr=False)
    _payload: bytes = field(repr=False)

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


@dataclass(frozen=True)
class MCPInputClaim:
    record: MCPInputRecord
    secret: str = field(repr=False)


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
            cipher = AEADCipher(get_agent_key(self.agent_did, 'service-keys'))
            binary = 'BYTEA' if self.is_postgres else 'BLOB'
            if not (self.is_postgres or self.is_sqlite):
                raise ValueError('MCP input custody requires SQLite or PostgreSQL')
            with repository_initialization_errors(integrity_error=MCPInputIntegrityError,
                    unknown_message='MCP input initialization requires reconciliation'):
                async with self.database.migration_lock('kestrel-mcp-input-schema-v1'):
                    await self.database.execute('CREATE TABLE IF NOT EXISTS mcp_input_schema (singleton BIGINT PRIMARY KEY, revision BIGINT NOT NULL)')
                    revision = await self.database.fetchone('SELECT revision FROM mcp_input_schema WHERE singleton=1')
                    if revision is None:
                        await self.database.execute(f'CREATE TABLE mcp_input_scopes (scope TEXT PRIMARY KEY, wrapped_key {binary} NOT NULL, revision BIGINT NOT NULL)')
                        await self.database.execute(f"CREATE TABLE mcp_inputs (scope TEXT NOT NULL, request_id TEXT NOT NULL, operation_digest TEXT NOT NULL, reader_digest TEXT NOT NULL, binding_digest TEXT NOT NULL, revision BIGINT NOT NULL, state TEXT NOT NULL CHECK(state IN ('WAITING','RESPONDING','RESOLVED','NON_RESUMABLE','UNKNOWN')), expires_at DOUBLE PRECISION NOT NULL, claim_hash TEXT, payload {binary} NOT NULL, PRIMARY KEY(scope,request_id), UNIQUE(scope,operation_digest), FOREIGN KEY(scope) REFERENCES mcp_input_scopes(scope))")
                        await self.database.execute('CREATE INDEX mcp_inputs_reader ON mcp_inputs(scope,reader_digest,state)')
                        await self.database.execute('CREATE TABLE mcp_input_decisions (scope TEXT NOT NULL, decision_id TEXT NOT NULL, operation_digest TEXT NOT NULL, input_digest TEXT NOT NULL, reason TEXT NOT NULL, decided_at DOUBLE PRECISION NOT NULL, PRIMARY KEY(scope,decision_id), FOREIGN KEY(scope) REFERENCES mcp_input_scopes(scope))')
                        await self.database.execute('INSERT INTO mcp_input_schema(singleton,revision) VALUES (1,1)')
                    elif revision != (1,):
                        raise MCPInputIntegrityError('Unsupported MCP input schema revision')
                    await self.database.fetchone('SELECT scope,request_id,operation_digest,reader_digest,binding_digest,revision,state,expires_at,claim_hash,payload FROM mcp_inputs WHERE 1=0')
                    await self.database.fetchone('SELECT scope,decision_id,operation_digest,input_digest,reason,decided_at FROM mcp_input_decisions WHERE 1=0')
                    row = await self.database.fetchone('SELECT wrapped_key FROM mcp_input_scopes WHERE scope=?', (self.scope,))
                    dependent = await self.database.fetchone('SELECT 1 FROM mcp_inputs WHERE scope=? LIMIT 1', (self.scope,))
                    if dependent is None:
                        dependent = await self.database.fetchone('SELECT 1 FROM mcp_input_decisions WHERE scope=? LIMIT 1', (self.scope,))
                    root, envelope = resolve_envelope_key(cipher, bytes(row[0]) if row else None,
                        has_dependents=dependent is not None, integrity_error=MCPInputIntegrityError,
                        missing_message='Original MCP input envelope is missing',
                        unavailable_message='Original MCP input encryption key is unavailable',
                        invalid_message='Invalid MCP input envelope')
                    if envelope is not None:
                        await self.database.execute('INSERT INTO mcp_input_scopes(scope,wrapped_key,revision) VALUES (?,?,0)', (self.scope, envelope))
            self._root = root

    def _binding_digest(self, binding):
        self._ready()
        if type(binding) is not MCPInputBinding or binding.agent_did != self.agent_did:
            raise MCPInputConflict('MCP input binding belongs to another agent')
        return hmac.new(self._root, encode_inline_json(asdict(binding), sort_keys=True), hashlib.sha256).hexdigest()

    def _aad(self, request_id, digest, revision, state, expires_at, claim_hash):
        return encode_inline_json(['mcp-input-v1', self.scope, request_id, digest,
            revision, state, expires_at, claim_hash])

    def _reader_digest(self, principal_id, session_id):
        self._ready()
        _text(principal_id)
        _text(session_id)
        return hmac.new(self._root, encode_inline_json(
            ['mcp-input-reader-v1', principal_id, session_id]), hashlib.sha256).hexdigest()

    def _encode(self, binding, request_id, revision, state, expires_at, claim_hash, payload, responses=None,
                continuation=None):
        if (type(payload) is not dict or (responses is not None and type(responses) is not dict)
                or (continuation is not None and (type(continuation) is not dict or state != 'RESOLVED'))):
            raise TypeError('MCP input checkpoint must be a JSON object')
        raw = encode_inline_json({'binding': asdict(binding), 'checkpoint': payload, 'responses': responses,
            'continuation': continuation})
        if len(raw) > self.max_payload_bytes:
            raise ValueError('MCP input checkpoint exceeded its storage budget')
        digest = self._binding_digest(binding)
        return AEADCipher(self._root).encrypt(raw,
            aad=self._aad(request_id, digest, revision, state, expires_at, claim_hash))

    async def read(self, binding: MCPInputBinding, request_id: str) -> MCPInputRecord:
        digest = self._binding_digest(binding)
        record = await self.read_for(binding.principal_id, binding.session_id, request_id)
        if not hmac.compare_digest(self._binding_digest(record.binding), digest):
            raise MCPInputConflict('MCP input request belongs to another operation')
        return record

    async def read_for(self, principal_id: str, session_id: str, request_id: str) -> MCPInputRecord:
        """Read only after the host independently authenticates this reader."""
        reader_digest = self._reader_digest(principal_id, session_id)
        _text(request_id)
        row = await self.database.fetchone('SELECT binding_digest,revision,state,expires_at,claim_hash,payload FROM mcp_inputs WHERE scope=? AND request_id=? AND reader_digest=?', (self.scope, request_id, reader_digest))
        if row is None:
            raise MCPInputConflict('MCP input request is absent or belongs to another operation')
        digest, revision, state, expires_at, claim_hash, ciphertext = row
        if (type(revision) is not int or revision < 0 or state not in
                {'WAITING','RESPONDING','RESOLVED','NON_RESUMABLE','UNKNOWN'}
                or ((state == 'RESPONDING') != (claim_hash is not None))):
            raise MCPInputIntegrityError('Invalid MCP input custody record')
        _time(expires_at)
        if len(ciphertext) > 4 * self.max_payload_bytes + 128:
            raise MCPInputIntegrityError('Stored MCP input exceeded its storage budget')
        try:
            raw = AEADCipher(self._root).decrypt(bytes(ciphertext),
                aad=self._aad(request_id, digest, revision, state, expires_at, claim_hash))
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
        return MCPInputRecord(request_id, revision, state, expires_at, binding, raw)

    async def list_for(self, principal_id: str, session_id: str, *, now: float) -> tuple[MCPInputRecord, ...]:
        """Expose waiting/uncertain metadata to the independently authorized UI."""
        digest = self._reader_digest(principal_id, session_id)
        _time(now)
        rows = await self.database.fetchall("SELECT request_id FROM mcp_inputs WHERE scope=? AND reader_digest=? AND (state IN ('RESPONDING','UNKNOWN') OR (state='WAITING' AND expires_at>?)) ORDER BY CASE WHEN state='UNKNOWN' THEN 1 ELSE 0 END,request_id LIMIT ?", (self.scope, digest, now, self.max_pending))
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

    async def create(self, binding: MCPInputBinding, payload: dict, *, expires_at: float,
                     now: float) -> MCPInputRecord:
        self._ready()
        _time(now)
        _time(expires_at)
        if expires_at <= now:
            raise MCPInputConflict('MCP input checkpoint has already expired')
        expires_at = float(expires_at)
        request_id = str(uuid4())
        digest = self._binding_digest(binding)
        ciphertext = self._encode(binding, request_id, 0, 'WAITING', expires_at, None, payload)
        operation_digest = hmac.new(self._root, encode_inline_json(
            ['mcp-input-operation-v1', binding.operation_id]), hashlib.sha256).hexdigest()
        reader_digest = self._reader_digest(binding.principal_id, binding.session_id)
        async with self.write_transaction():
            await self.database.execute('UPDATE mcp_input_scopes SET revision=revision+1 WHERE scope=?', (self.scope,))
            count = await self.database.fetchone("SELECT COUNT(*) FROM mcp_inputs WHERE scope=? AND (state='RESPONDING' OR (state='WAITING' AND expires_at>?))", (self.scope, now))
            if count[0] >= self.max_pending:
                raise MCPInputConflict('MCP input pending capacity is exhausted')
            changed = await self.database.execute("INSERT INTO mcp_inputs(scope,request_id,operation_digest,reader_digest,binding_digest,revision,state,expires_at,claim_hash,payload) VALUES (?,?,?,?,?,0,'WAITING',?,NULL,?) ON CONFLICT(scope,operation_digest) DO NOTHING", (self.scope, request_id, operation_digest, reader_digest, digest, expires_at, ciphertext))
            if changed != 1:
                raise MCPInputConflict('MCP input original operation already has a checkpoint')
        return await self.read(binding, request_id)

    async def claim_response(self, expected: MCPInputRecord, responses: dict, *, now: float) -> MCPInputClaim:
        """Persist validated responses before any continuation wire request."""
        self._ready()
        _time(now)
        if type(expected) is not MCPInputRecord or expected.state != 'WAITING' or expected.expires_at <= now:
            raise MCPInputConflict('MCP input is no longer waiting or has expired')
        if type(responses) is not dict:
            raise TypeError('MCP input responses must be a JSON object')
        actual = await self.read(expected.binding, expected.request_id)
        if (actual.state != 'WAITING' or actual.revision != expected.revision
                or actual.expires_at != expected.expires_at):
            raise MCPInputConflict('MCP input waiting revision was already consumed')
        secret = secrets.token_urlsafe(32)
        claim_hash = hashlib.sha256(secret.encode()).hexdigest()
        revision = expected.revision + 1
        digest = self._binding_digest(expected.binding)
        ciphertext = self._encode(expected.binding, expected.request_id, revision,
            'RESPONDING', expected.expires_at, claim_hash, actual.payload, responses)
        async with self.write_transaction():
            changed = await self.database.execute("UPDATE mcp_inputs SET revision=?,state='RESPONDING',claim_hash=?,payload=? WHERE scope=? AND request_id=? AND binding_digest=? AND revision=? AND state='WAITING' AND expires_at=? AND expires_at>?", (revision, claim_hash, ciphertext, self.scope, expected.request_id, digest, expected.revision, expected.expires_at, now))
            if changed != 1:
                raise MCPInputConflict('MCP input waiting revision was already consumed')
        return MCPInputClaim(await self.read(expected.binding, expected.request_id), secret)

    async def settle(self, claim: MCPInputClaim, *, state: str, continuation: dict | None = None) -> MCPInputRecord:
        """Record the original claim's outcome; uncertain work never reopens."""
        self._ready()
        if (type(claim) is not MCPInputClaim or claim.record.state != 'RESPONDING'
                or not isinstance(claim.secret, str) or len(claim.secret) != 43
                or state not in {'RESOLVED','NON_RESUMABLE','UNKNOWN'}):
            raise MCPInputConflict('MCP input settlement requires its original response claim')
        record = claim.record
        actual = await self.read(record.binding, record.request_id)
        if (actual.state != 'RESPONDING' or actual.revision != record.revision
                or actual.expires_at != record.expires_at):
            raise MCPInputConflict('MCP input response claim is no longer current')
        digest = self._binding_digest(record.binding)
        claim_hash = hashlib.sha256(claim.secret.encode()).hexdigest()
        revision = record.revision + 1
        ciphertext = self._encode(record.binding, record.request_id, revision, state,
            record.expires_at, None, actual.payload, actual.responses, continuation)
        async with self.write_transaction():
            changed = await self.database.execute("UPDATE mcp_inputs SET revision=?,state=?,claim_hash=NULL,payload=? WHERE scope=? AND request_id=? AND binding_digest=? AND revision=? AND state='RESPONDING' AND claim_hash=?", (revision, state, ciphertext, self.scope, record.request_id, digest, record.revision, claim_hash))
            if changed != 1:
                raise MCPInputConflict('MCP input response claim is no longer current')
        return await self.read(record.binding, record.request_id)
