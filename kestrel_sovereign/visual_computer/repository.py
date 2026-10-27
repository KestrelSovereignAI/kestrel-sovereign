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
from .step import PlanningRecord, StepIntent


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
    retired_at: datetime | None = None
    retired_from: Literal["admitted", "quoted"] | None = None
    phase: Literal["retired", "admitted", "quoted", "acquiring", "leased", "opening", "active", "fenced", "unknown", "released"] = "admitted"


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: ActionRequest
    receipt: ActionReceipt | None = None
    # Legacy receipt-less rows may already have dispatched; never replay them.
    dispatch_started: bool = True
    step_intent: StepIntent | None = None


class VisualCustodyRepository:
    def __init__(self, db, *, storage_id="visual-computer", max_requests=256, max_actions=4096):
        self.db, self.storage_id = db, storage_id
        self.max_requests, self.max_actions = max_requests, max_actions
        self._key = None
        self._original_results = {}
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
                    # These indexed flags are projections of authenticated
                    # payloads, never an independent authority/ownership store.
                    for table, column in (("visual_computer_requests", "released"),
                                          ("visual_computer_actions", "pending")):
                        if not await self.db.column_exists(table, column):
                            default = 0 if column == "released" else 1
                            await self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} BIGINT NOT NULL DEFAULT {default}")
                            rows = await self.db.fetchall(f"SELECT * FROM {table}")
                            for item in rows:
                                if column == "released":
                                    owner, identity, _, payload = item[:4]
                                    record = self._open(owner, identity, payload, CustodyRecord, key=key)
                                    value = int(record.phase == "released" and record.lease is not None
                                                and record.lease.state == "released")
                                    await self.db.execute(f"UPDATE {table} SET released = ? WHERE owner = ? AND request_id = ?", (value, owner, identity))
                                else:
                                    owner, session_key, action_id, payload = item[:4]
                                    record = self._open_action(owner, f"{session_key}:{action_id}", payload, key=key)
                                    value = int(record.pending() if type(record) is PlanningRecord else record.receipt is None or record.receipt.outcome == "unknown")
                                    await self.db.execute(f"UPDATE {table} SET pending = ? WHERE owner = ? AND session_key = ? AND action_id = ?", (value, owner, session_key, action_id))
                    if not await self.db.column_exists("visual_computer_requests", "retired"):
                        await self.db.execute("ALTER TABLE visual_computer_requests ADD COLUMN retired BIGINT NOT NULL DEFAULT 0")
                        for owner, identity, payload in await self.db.fetchall("SELECT owner, request_id, payload FROM visual_computer_requests"):
                            original = self._open(owner, identity, payload, CustodyRecord, key=key)
                            if original.phase == "retired":
                                if not self._retired(original):
                                    raise CustodyIntegrityError("invalid original acquisition retirement")
                                await self.db.execute("UPDATE visual_computer_requests SET retired = 1 WHERE owner = ? AND request_id = ?", (owner, identity))
                    await self.db.execute("CREATE INDEX IF NOT EXISTS visual_computer_live_requests ON visual_computer_requests (owner, released)")
                    await self.db.execute("CREATE INDEX IF NOT EXISTS visual_computer_retired_requests ON visual_computer_requests (owner, retired, released)")
                    await self.db.execute("CREATE INDEX IF NOT EXISTS visual_computer_pending_actions ON visual_computer_actions (owner, session_key, pending)")
                    await self.db.execute("DELETE FROM visual_computer_actions WHERE pending = 0 AND EXISTS (SELECT 1 FROM visual_computer_requests r WHERE r.owner = visual_computer_actions.owner AND r.session_key = visual_computer_actions.session_key AND r.released = 1)")
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

    def _open(self, owner, identity, payload, model, *, key=None):
        try:
            plain = AEADCipher(key or self._key).decrypt(bytes(payload), aad=f"{owner}:{identity}".encode())
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
            count = await self.db.fetchone("SELECT COUNT(*) FROM visual_computer_requests r WHERE owner = ? AND ((released = 0 AND retired = 0) OR EXISTS (SELECT 1 FROM visual_computer_actions a WHERE a.owner = r.owner AND a.session_key = r.session_key AND a.pending = 1))", (owner,))
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
            await self._lock_owner(owner)
            original = await self.db.fetchone("SELECT payload FROM visual_computer_requests WHERE owner = ? AND request_id = ?", (owner, record.request.request_id))
            if original is None:
                raise ResourceNotFound("visual custody not found")
            previous = self._open(owner, record.request.request_id, original[0], CustodyRecord)
            if previous.request != record.request or previous.session_id != record.session_id:
                raise ResourceConflict("original visual custody identity changed")
            if previous.phase == "retired":
                if record != previous:
                    raise ResourceConflict("retired visual custody is immutable")
                return
            if record.phase == "retired":
                raise ResourceConflict("retirement requires original never-acquired CAS")
            if previous.phase not in {"admitted", "quoted"} and record.phase in {"admitted", "quoted"}:
                raise ResourceConflict("possible acquisition cannot become unacquired")
            released = record.phase == "released" and record.lease is not None and record.lease.state == "released"
            if record.phase == "released" and not released:
                raise ResourceConflict("release requires verified physical lease retirement")
            if previous.phase == "released" and record != previous:
                raise ResourceConflict("released visual custody is immutable")
            if released:
                # Retain exact constraints, original lease and session identity;
                # remove obsolete private connection/frame/control derivatives.
                record.connection = record.frame = record.pending_control = None
                record.last_control = record.control_receipt = None
            updated = record.model_copy(update={"revision": record.revision + 1})
            count = await self.db.execute("UPDATE visual_computer_requests SET payload = ?, revision = revision + 1, released = ? WHERE owner = ? AND request_id = ? AND revision = ?", (self._seal(owner, record.request.request_id, updated), int(released), owner, record.request.request_id, record.revision))
            if count != 1:
                raise ResourceConflict("visual custody changed concurrently; reconcile")
            if released:
                await self.db.execute("DELETE FROM visual_computer_actions WHERE owner = ? AND session_key = ? AND pending = 0", (owner, self._address("session", record.session_id)))
        record.revision += 1

    @staticmethod
    def _unacquired(record):
        return (record.phase in {"admitted", "quoted"} and all(value is None for value in (
            record.lease, record.connection, record.session, record.frame, record.pending_control,
            record.last_control, record.control_receipt, record.retired_at, record.retired_from)))

    @staticmethod
    def _retired(record):
        return (record.phase == "retired" and record.retired_at is not None
            and record.retired_at.tzinfo is not None and record.retired_from in {"admitted", "quoted"}
            and all(value is None for value in (record.lease, record.connection, record.session,
                record.frame, record.pending_control, record.last_control, record.control_receipt)))

    async def retire_unacquired(self, scope, record, *, at, guard=None):
        """CAS exact original pre-acquisition facts; never claim physical release.

        Authenticated admitted/quoted phases precede the canonical acquiring CAS.
        Legacy rows with any possible-effect facts/actions remain charged.
        """
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("retirement time must be aware")
        async with self._transaction(guard):
            owner = self._owner(scope)
            await self._lock_owner(owner)
            row = await self.db.fetchone("SELECT payload FROM visual_computer_requests WHERE owner = ? AND request_id = ?", (owner, record.request.request_id))
            if row is None:
                raise ResourceNotFound("original visual custody not found")
            original = self._open(owner, record.request.request_id, row[0], CustodyRecord)
            if original.request != record.request or original.session_id != record.session_id:
                raise ResourceConflict("retirement differs from original custody")
            if self._retired(original):
                return original
            actions = await self.db.fetchone("SELECT 1 FROM visual_computer_actions WHERE owner = ? AND session_key = ? LIMIT 1", (owner, self._address("session", record.session_id)))
            if original.revision != record.revision or not self._unacquired(original) or actions:
                raise ResourceConflict("possible acquisition retains original custody")
            retired = original.model_copy(update={"phase":"retired", "retired_at":at,
                "retired_from":original.phase, "revision":original.revision+1})
            count = await self.db.execute("UPDATE visual_computer_requests SET payload = ?, revision = revision + 1, retired = 1 WHERE owner = ? AND request_id = ? AND revision = ? AND released = 0 AND retired = 0", (self._seal(owner, record.request.request_id, retired), owner, record.request.request_id, original.revision))
            if count != 1:
                raise ResourceConflict("original retirement changed concurrently")
            return retired

    @asynccontextmanager
    async def original_action_result(self, scope, session, request, *, guard):
        """Capture one existing prepared row before dispatch, never across IO."""
        from kestrel_sovereign.storage.original_action_result import _OriginalActionResult, _AcceptedActionResult
        from .dispatch import _ORIGIN
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
        result = acceptance = None
        try:
            async with self._transaction(guard, write=False):
                owner, session_key = self._owner(scope), self._address("session", session.session_id)
                if session.scope != scope or request.session_id != session.session_id:
                    raise ResourceConflict("original action result session differs")
                row = await self.db.fetchone("SELECT payload FROM visual_computer_actions WHERE owner = ? AND session_key = ? AND action_id = ?", (owner, session_key, request.action_id))
                if row is None:
                    raise ResourceNotFound("original prepared action absent")
                stored = self._open(owner, f"{session_key}:{request.action_id}", row[0], ActionRecord)
                parent = await self.db.fetchone("SELECT request_id, payload FROM visual_computer_requests WHERE owner = ? AND session_key = ?", (owner, session_key))
                original_session = self._open(owner, parent[0], parent[1], CustodyRecord) if parent else None
                if original_session is None or original_session.session != session or request.control_epoch != session.control_epoch:
                    raise ResourceConflict("original action result custody session differs")
                origin = _ORIGIN.get()
                if (stored.request != request or stored.receipt is not None or stored.dispatch_started
                        or origin is None or origin.session != session):
                    raise ResourceConflict("original prepared action result differs")
                acceptance = _AcceptedActionResult(asyncio.current_task(), origin,
                    (owner,session_key,request.action_id), bytes(row[0]), stored.model_dump_json())
                self._original_results[acceptance] = None
                result = _OriginalActionResult(self, origin, owner, session_key, row[0], stored, admission=acceptance)
            backend = self.db.backend
            if type(backend) is SQLiteBackend:
                result._sqlite_relation = await backend._read_original_signal_outcome_relation(result, capture=True)
                if result._sqlite_relation is None:
                    raise CustodyIntegrityError("original action result requires its plain native relation")
                table = 'main.visual_computer_actions'
            else:
                result._postgres_relation = await backend._capture_original_signal_outcome_relation(result)
                table = result._postgres_relation.qualified_name
            result._query = f"UPDATE {table} SET payload = ?, pending = 0 WHERE owner = ? AND session_key = ? AND action_id = ? AND payload = ? AND pending = 1"
            cast = "?" if type(backend) is SQLiteBackend else "CAST(? AS BYTEA)"
            result._match_query = f"SELECT 1 FROM {table} WHERE payload = ? AND owner = ? AND session_key = ? AND action_id = ? AND {cast} IS NOT NULL AND pending = 0"
            guard()
            yield result
        finally:
            if result is not None:
                result.close()
            elif acceptance is not None:
                self._original_results.pop(acceptance, None)

    def _open_action(self, owner, identity, payload, *, key=None):
        try:
            plain = AEADCipher(key or self._key).decrypt(bytes(payload), aad=f"{owner}:{identity}".encode())
            raw = json.loads(plain)
            if type(raw) is not dict: raise ValueError("invalid action payload")
        except (DecryptionError, ValueError, TypeError):
            raise CustodyIntegrityError("visual action custody verification failed") from None
        if raw.get("kind") == "visual-step-v1":
            try:
                record = PlanningRecord.model_validate(raw)
                if record.kind != "visual-step-v1" or record.state not in {"planning", "finished"} or (record.state == "planning" and record.answer is not None):
                    raise ValueError("invalid step phase")
                return record
            except (ValueError, TypeError):
                raise CustodyIntegrityError("invalid original visual step") from None
        return self._open(owner, identity, payload, ActionRecord, key=key)

    async def action(self, scope, session_id, action_id, *, accept=None, receipt=None, guard=None,
                     prepare=False, dispatch_revision=None, abandon_prepared=False, _result_owner=None):
        async with self._transaction(guard, write=accept is not None or receipt is not None or dispatch_revision is not None or abandon_prepared):
            owner, session_key = self._owner(scope), self._address("session", session_id)
            if accept is not None or receipt is not None or dispatch_revision is not None or abandon_prepared:
                await self._lock_owner(owner)
            identity = f"{session_key}:{action_id}"
            row = await self.db.fetchone("SELECT payload FROM visual_computer_actions WHERE owner = ? AND session_key = ? AND action_id = ?", (owner, session_key, action_id))
            parent = await self.db.fetchone("SELECT released, retired FROM visual_computer_requests WHERE owner = ? AND session_key = ?", (owner, session_key))
            if (accept is not None or dispatch_revision is not None) and (parent is None or parent[0] or parent[1]):
                raise ResourceConflict("original action session is absent or physically released")
            if row is None:
                if accept is None:
                    raise ResourceNotFound("visual action not found")
                count = await self.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions WHERE owner = ?", (owner,))
                if count[0] >= self.max_actions:
                    raise ResourceConflict("visual custody action capacity exhausted")
                stored = ActionRecord(request=accept, dispatch_started=not prepare)
                await self.db.execute("INSERT INTO visual_computer_actions (owner, session_key, action_id, payload) VALUES (?, ?, ?, ?)", (owner, session_key, action_id, self._seal(owner, identity, stored)))
            else:
                stored = self._open_action(owner, identity, row[0])
                if type(stored) is PlanningRecord:
                    if accept is not None or receipt is not None or dispatch_revision is not None or abandon_prepared or _result_owner is not None:
                        raise ResourceOutcomeUnknown("Original planning row is not an SDK action")
                    return stored
                if accept is not None and stored.request != accept:
                    raise ResourceConflict("action ID already identifies different input")
            changed = False
            if dispatch_revision is not None:
                if stored.dispatch_started or stored.receipt is not None:
                    raise ResourceConflict("original action dispatch is no longer available")
                current = await self.db.fetchone("SELECT revision FROM visual_computer_requests WHERE owner = ? AND session_key = ?", (owner, session_key))
                if current is None or current[0] != dispatch_revision:
                    raise ResourceConflict("visual session changed before original dispatch")
                stored.dispatch_started, changed = True, True
            if abandon_prepared and not stored.dispatch_started and stored.receipt is None:
                request = stored.request
                stored.receipt = ActionReceipt(action_id=request.action_id, session_id=request.session_id,
                    observation_id=request.observation_id, control_epoch=request.control_epoch,
                    outcome="denied", recorded_at=datetime.now(request.not_after.tzinfo), reason_code="host_not_dispatched")
                changed = True
            if receipt is not None:
                if stored.receipt is not None and stored.receipt.outcome != "unknown" and stored.receipt != receipt:
                    raise ResourceConflict("accepted action receipt changed")
                stored.receipt, changed = receipt, True
            if changed:
                payload = self._seal(owner, identity, stored)
                if _result_owner is not None:
                    if dispatch_revision is None or _result_owner._identity != (owner, session_key, action_id):
                        raise ResourceConflict("original result owner differs from dispatch row")
                    _result_owner.dispatch_payload(bytes(row[0]), payload, stored)
                await self.db.execute("UPDATE visual_computer_actions SET payload = ?, pending = ? WHERE owner = ? AND session_key = ? AND action_id = ?", (payload, int(stored.receipt is None or stored.receipt.outcome == "unknown"), owner, session_key, action_id))
            if changed and parent is not None and parent[0] and stored.receipt is not None and stored.receipt.outcome != "unknown":
                await self.db.execute("DELETE FROM visual_computer_actions WHERE owner = ? AND session_key = ? AND action_id = ? AND pending = 0", (owner, session_key, action_id))
            return stored

    async def _lock_owner(self, owner):
        await self.db.execute("INSERT INTO visual_computer_owners (owner, generation) VALUES (?, 0) ON CONFLICT (owner) DO NOTHING", (owner,))
        await self.db.execute("UPDATE visual_computer_owners SET generation = generation + 1 WHERE owner = ?", (owner,))

    async def has_pending_actions(self, scope, session_id, guard=None, *, _step_owner=None):
        async with self._transaction(guard, write=False):
            owner, session_key = self._owner(scope), self._address("session", session_id)
            rows = await self.db.fetchall("SELECT action_id, payload FROM visual_computer_actions WHERE owner = ? AND session_key = ?", (owner, session_key))
            for row in rows:
                record = self._open_action(owner, f"{session_key}:{row[0]}", row[1])
                if _step_owner is not None:
                    from .step import _OriginalVisualStep
                    if type(_step_owner) is not _OriginalVisualStep:
                        raise ResourceConflict("Original step owner required")
                    if _step_owner.own_pending(scope, session_id, row[0], row[1]):
                        continue
                if (record.pending() if type(record) is PlanningRecord else record.receipt is None or record.receipt.outcome == "unknown"):
                    return True
            return False

    async def records(self, scope):
        """Internal host lifecycle only; exact original owner, no public list API."""
        async with self._transaction(write=False):
            owner = self._owner(scope)
            rows = await self.db.fetchall("SELECT request_id, payload FROM visual_computer_requests r WHERE owner = ? AND ((released = 0 AND retired = 0) OR EXISTS (SELECT 1 FROM visual_computer_actions a WHERE a.owner = r.owner AND a.session_key = r.session_key AND a.pending = 1))", (owner,))
            return [self._open(owner, row[0], row[1], CustodyRecord) for row in rows]
