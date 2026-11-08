"""Content-free, durable admission records for governed tool effects.

This is an internal store. The host selects the identity and caller namespace;
neither an operation key nor a stored claim grants permission to execute.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import hmac
import math
import os
from typing import Literal, NoReturn
from uuid import uuid4

from kestrel_sdk._validation import stable_token
from kestrel_sdk.resources import ResourceConflict, ResourceError
from kestrel_sdk.security.encryption import get_fernet
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.media import encode_inline_json
from kestrel_sovereign.a2a.stores.unified.base import NativeStoreConflict, UnifiedStoreBase
from kestrel_sovereign.security.envelope_keys import resolve_envelope_key
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.initialization_errors import repository_initialization_errors


class ToolEffectIntegrityError(ResourceError):
    """The original journal cannot be opened; replacement would permit replay."""


class ToolEffectOperationConflict(NativeStoreConflict):
    """An operation key was reused with different input."""


class ToolEffectAttemptSuperseded(NativeStoreConflict):
    """This execution attempt no longer owns admission."""


class ToolEffectAttemptClosed(NativeStoreConflict):
    """The original attempt still exists but no longer admits this work."""


class ToolEffectClaimConflict(NativeStoreConflict):
    """A settlement does not match an original unsettled claim."""


class ToolEffectReconciliationRequired(NativeStoreConflict):
    """Existing effects make replay unsafe, regardless of their reported result."""

    def __init__(self, counts: dict[str, int]):
        super().__init__("tool effect operation requires reconciliation")
        self.counts = counts


@dataclass(frozen=True)
class ToolEffectAttempt:
    scope_digest: str
    operation_digest: str
    attempt_id: str
    _terminal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ToolEffectClaim:
    attempt: ToolEffectAttempt
    claim_id: str


TOOL_EFFECT_OPERATIONS_SQL = "CREATE TABLE tool_effect_operations (scope_digest TEXT NOT NULL, operation_digest TEXT NOT NULL, request_digest TEXT NOT NULL, attempt_id TEXT NOT NULL, revision BIGINT NOT NULL, state TEXT NOT NULL CHECK(state IN ('OPEN', 'CLOSED')), latched BIGINT NOT NULL CHECK(latched IN (0, 1)), PRIMARY KEY(scope_digest, operation_digest))"

TOOL_EFFECT_OPERATIONS_COLUMNS = {
    'scope_digest': 'TEXT', 'operation_digest': 'TEXT', 'request_digest': 'TEXT',
    'attempt_id': 'TEXT', 'revision': 'BIGINT', 'state': 'TEXT', 'latched': 'BIGINT',
}

TOOL_EFFECT_CLAIMS_SQL = "CREATE TABLE tool_effect_claims (scope_digest TEXT NOT NULL, operation_digest TEXT NOT NULL, claim_id TEXT NOT NULL, attempt_id TEXT NOT NULL, parent_claim_id TEXT, registration_digest TEXT NOT NULL, args_digest TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('STARTED', 'RETURNED_OK', 'RETURNED_ERROR', 'UNCONFIRMED', 'NOT_EXECUTED')), PRIMARY KEY(scope_digest, operation_digest, claim_id), FOREIGN KEY(scope_digest, operation_digest) REFERENCES tool_effect_operations(scope_digest, operation_digest))"

TOOL_EFFECT_CLAIMS_COLUMNS = dict.fromkeys((
    'scope_digest', 'operation_digest', 'claim_id', 'attempt_id', 'parent_claim_id',
    'registration_digest', 'args_digest', 'state',
), 'TEXT')


TOOL_EFFECT_SCHEMA_REVISION = 2

ClaimTerminal = Literal["RETURNED_OK", "RETURNED_ERROR", "UNCONFIRMED", "NOT_EXECUTED"]


class ToolEffectStore(UnifiedStoreBase):
    """Borrow the primary database and fence replays with acknowledged claims."""

    def __init__(self, database: AsyncDatabase, *, scope: NativeRecipientScope,
                 terminal_timeout_seconds: float | None = None):
        super().__init__(database.backend, owns_backend=False)
        timeout = (float(os.environ.get('KESTREL_TOOL_EFFECT_TERMINAL_TIMEOUT_S', '5'))
            if terminal_timeout_seconds is None else terminal_timeout_seconds)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Tool effect terminal timeout must be finite and positive')
        self.terminal_timeout_seconds = float(timeout)
        self.database = database
        self.scope_digest = hashlib.sha256(encode_inline_json([
            "kestrel-tool-effect-scope-v2", scope.recipient_id, scope.tenant,
        ], sort_keys=True)).hexdigest()
        self._root: bytes | None = None
        self._initialization_lock = asyncio.Lock()

    @property
    def initialized(self) -> bool:
        """Whether the original envelope has been durably acknowledged."""
        return self._root is not None

    async def close(self) -> None:
        """Retire the borrowed handle without closing the primary database."""
        await super().close()
        self._root = None

    def _require_outermost(self) -> None:
        if self.database.owns_open_transaction is not False:
            raise ResourceConflict("tool effect journal requires its own durable transaction")

    async def initialize(self) -> None:
        if self._root is not None:
            return
        async with self._initialization_lock:
            if self._root is not None:
                return
            self._require_outermost()
            if not (self.is_sqlite or self.is_postgres):
                raise ValueError("tool effect journal requires SQLite or PostgreSQL")
            cipher = get_fernet()
            if cipher is None:
                raise ToolEffectIntegrityError("tool effect journal requires encryption")
            with repository_initialization_errors(
                integrity_error=ToolEffectIntegrityError,
                unknown_message="tool effect initialization requires reconciliation",
            ):
                async with self.database.migration_lock("kestrel-tool-effect-schema-v1"):
                    await self.database.execute(
                        "CREATE TABLE IF NOT EXISTS tool_effect_schema (singleton BIGINT PRIMARY KEY, revision BIGINT NOT NULL)"
                    )
                    revision = await self.database.fetchone("SELECT revision FROM tool_effect_schema WHERE singleton = 1")
                    if revision is not None and revision[0] != TOOL_EFFECT_SCHEMA_REVISION:
                        raise ToolEffectIntegrityError("unsupported tool effect schema revision")
                    if revision is None:
                        binary = "BYTEA" if self.is_postgres else "BLOB"
                        for sql in (
                            f"CREATE TABLE tool_effect_keys (key_id TEXT PRIMARY KEY, scope_digest TEXT NOT NULL UNIQUE, wrapped_key {binary} NOT NULL)",
                            TOOL_EFFECT_OPERATIONS_SQL,
                            TOOL_EFFECT_CLAIMS_SQL,
                        ):
                            await self.database.execute(sql)
                        await self.database.execute(
                            "INSERT INTO tool_effect_schema(singleton, revision) VALUES (1, ?)",
                            (TOOL_EFFECT_SCHEMA_REVISION,),
                        )
                    # A completed revision never silently recreates lost tables.
                    await self.database.fetchone("SELECT request_digest, attempt_id, revision, state, latched FROM tool_effect_operations WHERE 1 = 0")
                    await self.database.fetchone("SELECT claim_id, attempt_id, parent_claim_id, registration_digest, args_digest, state FROM tool_effect_claims WHERE 1 = 0")
                    row = await self.database.fetchone("SELECT wrapped_key FROM tool_effect_keys WHERE scope_digest = ?", (self.scope_digest,))
                    has_dependents = False
                    if row is None:
                        for table in ("tool_effect_operations", "tool_effect_claims"):
                            if await self.database.fetchone(f"SELECT 1 FROM {table} WHERE scope_digest = ? LIMIT 1", (self.scope_digest,)):
                                has_dependents = True
                    root, new_envelope = resolve_envelope_key(
                        cipher, bytes(row[0]) if row is not None else None,
                        has_dependents=has_dependents, integrity_error=ToolEffectIntegrityError,
                        missing_message="original tool effect envelope key is missing",
                        unavailable_message="tool effect envelope cannot be opened; restore its original KESTREL_DATA_KEY or complete offline key rotation",
                        invalid_message="invalid tool effect envelope key",
                    )
                    if new_envelope is not None:
                        await self.database.execute(
                            "INSERT INTO tool_effect_keys(key_id, scope_digest, wrapped_key) VALUES (?, ?, ?)",
                            (self.scope_digest + ":v1", self.scope_digest, new_envelope),
                        )
            self._root = root

    def _digest(self, domain: str, value: object) -> str:
        if self._root is None:
            raise RuntimeError("tool effect journal must be initialized")
        return hmac.new(self._root, encode_inline_json([domain, value], sort_keys=True), hashlib.sha256).hexdigest()

    def _require_scope(self, attempt: ToolEffectAttempt) -> None:
        if attempt.scope_digest != self.scope_digest:
            raise ResourceConflict("tool effect attempt belongs to another scope")
        if self._root is None:
            raise RuntimeError("tool effect journal must be initialized")
        self._require_outermost()
        from kestrel_sovereign.storage.original_tool_effect_finish import _OriginalToolEffectFinish

        if type(attempt._terminal) is not _OriginalToolEffectFinish:
            raise ResourceConflict('tool effect attempt lacks its original native backing')
        attempt._terminal.require_attempt(self, attempt)

    async def _counts(self, operation: str) -> dict[str, int]:
        counts = dict(returned_ok=0, returned_error=0, unconfirmed=0, unsettled=0, not_executed=0)
        for state, count in await self.database.fetchall(
            "SELECT state, COUNT(*) FROM tool_effect_claims WHERE scope_digest = ? AND operation_digest = ? GROUP BY state",
            (self.scope_digest, operation),
        ):
            counts["unsettled" if state == "STARTED" else state.lower()] = count
        return counts

    async def _refuse_inactive_attempt(self, attempt: ToolEffectAttempt) -> NoReturn:
        row = await self.database.fetchone(
            "SELECT attempt_id FROM tool_effect_operations WHERE scope_digest = ? AND operation_digest = ?",
            (self.scope_digest, attempt.operation_digest),
        )
        if row is not None and row[0] == attempt.attempt_id:
            raise ToolEffectAttemptClosed("tool effect attempt is closed or stopped")
        raise ToolEffectAttemptSuperseded("tool effect attempt was superseded or is absent")

    async def admit(self, *, namespace: object, client_key: str, request: object) -> ToolEffectAttempt:
        self._require_outermost()
        stable_token(client_key, "operation_id")
        operation = self._digest("op-v2", [namespace, client_key])
        request_digest = self._digest("request-v2", request)
        from kestrel_sovereign.storage.original_tool_effect_finish import _OriginalToolEffectFinish

        attempt_id = str(uuid4())
        terminal = await _OriginalToolEffectFinish.prepare(self,
            (self.scope_digest, operation, attempt_id))
        attempt = ToolEffectAttempt(self.scope_digest, operation, attempt_id, terminal)
        async with self.write_transaction():
            await self.database.execute(
                "INSERT INTO tool_effect_operations(scope_digest, operation_digest, request_digest, attempt_id, revision, state, latched) VALUES (?, ?, ?, ?, 0, 'OPEN', 0) ON CONFLICT(scope_digest, operation_digest) DO NOTHING",
                (self.scope_digest, operation, request_digest, attempt.attempt_id),
            )
            # The same row lock serializes supersession and effect admission.
            await self.database.execute(
                "UPDATE tool_effect_operations SET revision = revision + 1 WHERE scope_digest = ? AND operation_digest = ?",
                (self.scope_digest, operation),
            )
            row = await self.database.fetchone("SELECT request_digest, attempt_id, state, latched FROM tool_effect_operations WHERE scope_digest = ? AND operation_digest = ?", (self.scope_digest, operation))
            if row[0] != request_digest:
                raise ToolEffectOperationConflict("tool effect operation key has different input")
            effect = await self.database.fetchone(
                "SELECT 1 FROM tool_effect_claims WHERE scope_digest = ? AND operation_digest = ? AND state <> 'NOT_EXECUTED' LIMIT 1",
                (self.scope_digest, operation),
            )
            # Only acknowledged clean closure certifies a claim-free attempt.
            # An interrupted OPEN operation can include a required audit whose
            # ACK was lost before any business-effect claim was created.
            interrupted = row[1] != attempt.attempt_id and row[2] != 'CLOSED'
            if effect is not None or interrupted or row[3]:
                raise ToolEffectReconciliationRequired(await self._counts(operation))
            await self.database.execute(
                "UPDATE tool_effect_operations SET attempt_id = ?, state = 'OPEN', latched = 0 WHERE scope_digest = ? AND operation_digest = ?",
                (attempt.attempt_id, self.scope_digest, operation),
            )
        terminal.admitted(attempt)
        return attempt

    async def claim(self, attempt: ToolEffectAttempt, *, registration: object,
                    arguments: object, parent: ToolEffectClaim | None = None) -> ToolEffectClaim:
        self._require_scope(attempt)
        if parent is not None and parent.attempt != attempt:
            raise ToolEffectAttemptSuperseded("parent claim belongs to another attempt")
        registration_digest = self._digest("registration-v2", registration)
        args_digest = self._digest("args-v2", arguments)
        claim = ToolEffectClaim(attempt, str(uuid4()))
        async with self.write_transaction():
            changed = await self.database.execute(
                "UPDATE tool_effect_operations SET revision = revision + 1 WHERE scope_digest = ? AND operation_digest = ? AND attempt_id = ? AND state = 'OPEN' AND latched = 0",
                (self.scope_digest, attempt.operation_digest, attempt.attempt_id),
            )
            if changed != 1:
                await self._refuse_inactive_attempt(attempt)
            await self.database.execute(
                "INSERT INTO tool_effect_claims(scope_digest, operation_digest, claim_id, attempt_id, parent_claim_id, registration_digest, args_digest, state) VALUES (?, ?, ?, ?, ?, ?, ?, 'STARTED')",
                (self.scope_digest, attempt.operation_digest, claim.claim_id, attempt.attempt_id,
                 parent.claim_id if parent is not None else None, registration_digest, args_digest),
            )
        return claim

    async def settle(self, claim: ToolEffectClaim, *, state: ClaimTerminal) -> None:
        self._require_scope(claim.attempt)
        if state not in {"RETURNED_OK", "RETURNED_ERROR", "UNCONFIRMED", "NOT_EXECUTED"}:
            raise ValueError("invalid tool effect claim terminal state")
        attempt = claim.attempt
        async with self.write_transaction():
            # Only a live attempt can attest that it never dispatched. A closed
            # attempt may still settle a real outcome while its owners drain.
            changed = await self.database.execute(
                "UPDATE tool_effect_operations SET revision = revision + 1 WHERE scope_digest = ? AND operation_digest = ? AND attempt_id = ? AND (? <> 'NOT_EXECUTED' OR state = 'OPEN')",
                (self.scope_digest, attempt.operation_digest, attempt.attempt_id, state),
            )
            if changed != 1:
                await self._refuse_inactive_attempt(attempt)
            changed = await self.database.execute(
                "UPDATE tool_effect_claims SET state = ? WHERE scope_digest = ? AND operation_digest = ? AND claim_id = ? AND attempt_id = ? AND state = 'STARTED'",
                (state, self.scope_digest, attempt.operation_digest, claim.claim_id, attempt.attempt_id),
            )
            if changed != 1:
                raise ToolEffectClaimConflict("tool effect claim is already terminal or absent")
            if state == "UNCONFIRMED":
                await self.database.execute(
                    "UPDATE tool_effect_operations SET latched = 1 WHERE scope_digest = ? AND operation_digest = ?",
                    (self.scope_digest, attempt.operation_digest),
                )

    async def finish(self, attempt: ToolEffectAttempt, *, unconfirmed: bool = False) -> None:
        self._require_scope(attempt)
        await attempt._terminal.finish(unconfirmed=unconfirmed)
