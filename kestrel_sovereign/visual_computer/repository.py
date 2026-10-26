"""Encrypted visual custody in the original agent database.

This repository is not an authority resolver. Every caller holds the host's
current capability/privacy operation. Original IDs survive cancellation and
unknown effects; records are never deleted merely because a lease expired.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Literal

from pydantic import BaseModel, ConfigDict
from kestrel_sdk.computer_use import (
    ActionReceipt, ActionRequest, ComputerSession, ControlRequest, DriverConnection,
    EnvironmentLease, EnvironmentQuote, EnvironmentRequest, ScreenFrame,
)
from kestrel_sdk.resources import ResourceConflict, ResourceError, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.encryption import get_fernet
from kestrel_sdk.security.exceptions import DecryptionError
from kestrel_sdk.storage.database.interface import TransactionError

from kestrel_sovereign.execution_custody import execution_commit_outcome


class CustodyIntegrityError(ResourceError):
    """Original encrypted custody cannot be verified; never replace its key."""


class CustodyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = 0
    request: EnvironmentRequest
    session_id: str
    last_activity_at: datetime | None = None
    quote: EnvironmentQuote | None = None
    lease: EnvironmentLease | None = None
    connection: DriverConnection | None = None
    session: ComputerSession | None = None
    frame: ScreenFrame | None = None
    pending_control: ControlRequest | None = None
    last_control: ControlRequest | None = None
    control_receipt: ComputerSession | None = None
    phase: Literal["admitted", "quoted", "acquiring", "leased", "opening", "active", "fenced", "unknown", "released"] = "admitted"


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: ActionRequest
    receipt: ActionReceipt | None = None


class VisualCustodyRepository:
    def __init__(self, db, *, storage_id="visual-computer", max_requests=256, max_actions=4096):
        self.db, self.storage_id = db, storage_id
        self.max_requests, self.max_actions = max_requests, max_actions
        self._key = None
        self._initialization_lock = asyncio.Lock()

    async def ensure_initialized(self):
        if self._key is not None:
            return
        async with self._initialization_lock:
            if self._key is not None:
                return
            cipher = get_fernet()
            if cipher is None:
                raise CustodyIntegrityError("visual custody requires encryption")
            if self.db.owns_open_transaction is not False:
                raise ResourceConflict("visual custody requires an outermost transaction")
            try:
                async with self.db.migration_lock("visual-computer-custody-v1"):
                    binary = "BYTEA" if self.db.backend_type == "postgres" else "BLOB"
                    for sql in (
                        f"CREATE TABLE IF NOT EXISTS visual_computer_keys (key_id TEXT PRIMARY KEY, wrapped_key {binary} NOT NULL)",
                        "CREATE TABLE IF NOT EXISTS visual_computer_owners (owner TEXT PRIMARY KEY, generation BIGINT NOT NULL)",
                        f"CREATE TABLE IF NOT EXISTS visual_computer_requests (owner TEXT NOT NULL, request_id TEXT NOT NULL, session_key TEXT NOT NULL, payload {binary} NOT NULL, revision BIGINT NOT NULL, PRIMARY KEY (owner, request_id), UNIQUE(owner, session_key))",
                        f"CREATE TABLE IF NOT EXISTS visual_computer_actions (owner TEXT NOT NULL, session_key TEXT NOT NULL, action_id TEXT NOT NULL, payload {binary} NOT NULL, PRIMARY KEY (owner, session_key, action_id))",
                    ):
                        await self.db.execute(sql)
                    row = await self.db.fetchone("SELECT wrapped_key FROM visual_computer_keys WHERE key_id = ?", ("v1",))
                    if row is None:
                        if await self.db.fetchone("SELECT 1 FROM visual_computer_requests LIMIT 1"):
                            raise CustodyIntegrityError("original visual custody key is missing")
                        key = secrets.token_bytes(32)
                        await self.db.execute("INSERT INTO visual_computer_keys (key_id, wrapped_key) VALUES (?, ?)", ("v1", cipher.encrypt(key)))
                    else:
                        try:
                            key = cipher.decrypt(bytes(row[0]))
                        except DecryptionError:
                            raise CustodyIntegrityError("original visual custody key is unavailable") from None
            except TransactionError as error:
                if execution_commit_outcome(error) is not None:
                    raise ResourceOutcomeUnknown("visual custody initialization requires reconciliation") from error
                if isinstance(error.__cause__, CustodyIntegrityError):
                    raise error.__cause__ from None
                raise
            self._key = key

    def _address(self, *parts):
        if self._key is None:
            raise RuntimeError("visual custody is not initialized")
        return hmac.new(self._key, json.dumps(parts, ensure_ascii=True).encode(), hashlib.sha256).hexdigest()

    def _owner(self, scope):
        scope = ResourceScope.model_validate(scope)
        return self._address("owner", scope.tenant_id, scope.principal_id, scope.agent_id)

    def _seal(self, owner, identity, value):
        # DriverConnection's generic dumps intentionally omit these secrets.
        # Their only serialization is this authenticated encrypted envelope.
        payload = value.model_dump(mode="json")
        if isinstance(value, CustodyRecord) and value.connection is not None:
            payload["connection"].update(endpoint=value.connection.endpoint.get_secret_value(),
                                         credential=value.connection.credential.get_secret_value())
        return AEADCipher(self._key).encrypt(json.dumps(payload).encode(), aad=f"{owner}:{identity}".encode())

    def _open(self, owner, identity, payload, model):
        try:
            plain = AEADCipher(self._key).decrypt(bytes(payload), aad=f"{owner}:{identity}".encode())
            return model.model_validate_json(plain)
        except (DecryptionError, ValueError):
            raise CustodyIntegrityError("visual custody verification failed") from None

    @asynccontextmanager
    async def _transaction(self, guard=None, *, write=True):
        await self.ensure_initialized()
        if self.db.owns_open_transaction is not False:
            raise ResourceConflict("visual custody requires an outermost transaction")
        try:
            async with self.db.transaction(immediate=write):
                if guard:
                    guard()
                yield
                if guard:
                    guard()
        except Exception as error:
            if execution_commit_outcome(error) is not None:
                raise ResourceOutcomeUnknown("visual custody commit requires reconciliation") from error
            if isinstance(error, TransactionError) and isinstance(error.__cause__, ResourceError):
                raise error.__cause__ from None
            raise

    async def admit(self, scope, record, guard):
        async with self._transaction(guard):
            owner = self._owner(scope)
            await self._lock_owner(owner)
            previous = await self.db.fetchone("SELECT payload FROM visual_computer_requests WHERE owner = ? AND request_id = ?", (owner, record.request.request_id))
            if previous:
                stored = self._open(owner, record.request.request_id, previous[0], CustodyRecord)
                if stored.request.digest() != record.request.digest():
                    raise ResourceConflict("request ID already identifies different constraints")
                return stored
            count = await self.db.fetchone("SELECT COUNT(*) FROM visual_computer_requests WHERE owner = ?", (owner,))
            if count[0] >= self.max_requests:
                raise ResourceConflict("visual custody request capacity exhausted")
            await self.db.execute("INSERT INTO visual_computer_requests (owner, request_id, session_key, payload, revision) VALUES (?, ?, ?, ?, 0)",
                (owner, record.request.request_id, self._address("session", record.session_id), self._seal(owner, record.request.request_id, record)))
        return record

    async def get(self, scope, *, request_id=None, session_id=None, guard=None):
        async with self._transaction(guard, write=False):
            owner = self._owner(scope)
            if request_id is not None:
                row = await self.db.fetchone("SELECT request_id, payload FROM visual_computer_requests WHERE owner = ? AND request_id = ?", (owner, request_id))
            else:
                row = await self.db.fetchone("SELECT request_id, payload FROM visual_computer_requests WHERE owner = ? AND session_key = ?", (owner, self._address("session", session_id)))
            if row is None:
                raise ResourceNotFound("visual custody not found")
            return self._open(owner, row[0], row[1], CustodyRecord)

    async def save(self, scope, record, guard=None):
        async with self._transaction(guard):
            owner = self._owner(scope)
            updated = record.model_copy(update={"revision": record.revision + 1})
            count = await self.db.execute("UPDATE visual_computer_requests SET payload = ?, revision = revision + 1 WHERE owner = ? AND request_id = ? AND revision = ?", (self._seal(owner, record.request.request_id, updated), owner, record.request.request_id, record.revision))
            if count != 1:
                raise ResourceConflict("visual custody changed concurrently; reconcile")
        record.revision += 1

    async def action(self, scope, session_id, action_id, *, accept=None, receipt=None, guard=None):
        async with self._transaction(guard, write=accept is not None or receipt is not None):
            owner, session_key = self._owner(scope), self._address("session", session_id)
            if accept is not None or receipt is not None:
                await self._lock_owner(owner)
            identity = f"{session_key}:{action_id}"
            row = await self.db.fetchone("SELECT payload FROM visual_computer_actions WHERE owner = ? AND session_key = ? AND action_id = ?", (owner, session_key, action_id))
            if row is None:
                if accept is None:
                    raise ResourceNotFound("visual action not found")
                count = await self.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions WHERE owner = ?", (owner,))
                if count[0] >= self.max_actions:
                    raise ResourceConflict("visual custody action capacity exhausted")
                stored = ActionRecord(request=accept)
                await self.db.execute("INSERT INTO visual_computer_actions (owner, session_key, action_id, payload) VALUES (?, ?, ?, ?)", (owner, session_key, action_id, self._seal(owner, identity, stored)))
            else:
                stored = self._open(owner, identity, row[0], ActionRecord)
                if accept is not None and stored.request != accept:
                    raise ResourceConflict("action ID already identifies different input")
            if receipt is not None:
                if stored.receipt is not None and stored.receipt.outcome != "unknown" and stored.receipt != receipt:
                    raise ResourceConflict("accepted action receipt changed")
                stored.receipt = receipt
                await self.db.execute("UPDATE visual_computer_actions SET payload = ? WHERE owner = ? AND session_key = ? AND action_id = ?", (self._seal(owner, identity, stored), owner, session_key, action_id))
            return stored

    async def _lock_owner(self, owner):
        await self.db.execute("INSERT INTO visual_computer_owners (owner, generation) VALUES (?, 0) ON CONFLICT (owner) DO NOTHING", (owner,))
        await self.db.execute("UPDATE visual_computer_owners SET generation = generation + 1 WHERE owner = ?", (owner,))

    async def has_pending_actions(self, scope, session_id, guard=None):
        async with self._transaction(guard, write=False):
            owner, session_key = self._owner(scope), self._address("session", session_id)
            rows = await self.db.fetchall("SELECT action_id, payload FROM visual_computer_actions WHERE owner = ? AND session_key = ?", (owner, session_key))
            for row in rows:
                record = self._open(owner, f"{session_key}:{row[0]}", row[1], ActionRecord)
                if record.receipt is None or record.receipt.outcome == "unknown":
                    return True
            return False

    async def records(self, scope):
        """Internal host lifecycle only; exact original owner, no public list API."""
        async with self._transaction(write=False):
            owner = self._owner(scope)
            rows = await self.db.fetchall("SELECT request_id, payload FROM visual_computer_requests WHERE owner = ?", (owner,))
            return [self._open(owner, row[0], row[1], CustodyRecord) for row in rows]
