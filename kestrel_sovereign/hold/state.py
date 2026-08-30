"""Typed, durable host and agent Hold latches.

A Hold is state, not a cancellation event.  The current latch and its immutable
mutation receipts live in the host control database so every worker observes
the same answer after a restart.  This module owns only storage and composition;
turn-start refusal is the separate enforcement seam tracked by #3162.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional
from uuid import uuid4

from kestrel_sovereign.storage.database_clock import database_now_sql


HOST_HOLD_TARGET = "host"
_SCHEMA_LOCK = "hold_state_v1"
_WITNESS_BACKFILL = "hold_state_witness_ledgers_v1"
_LATCH_COLUMNS = (
    "scope, target_id, active, hold_receipt_id, reason, actor_id, set_at, revision"
)
_RECEIPT_COLUMNS = (
    "receipt_id, operation_id, action, disposition, scope, target_id, reason, "
    "actor_id, occurred_at, expected_hold_receipt_id, prior_hold_receipt_id, "
    "resulting_hold_receipt_id"
)


class HoldScope(str, Enum):
    """The two scopes on which a durable Hold may latch."""

    HOST = "host"
    AGENT = "agent"


class HoldAction(str, Enum):
    HOLD = "hold"
    RELEASE = "release"


class HoldDisposition(str, Enum):
    APPLIED = "applied"
    ALREADY_IN_STATE = "already_in_state"
    REFUSED_STALE = "refused_stale"


class HoldStateError(RuntimeError):
    """Base class for a durable Hold-state failure."""


class HoldIdempotencyConflict(HoldStateError):
    """One operation id was reused for a different mutation."""


class HoldCorruptStateError(HoldStateError):
    """Persisted Hold state cannot be interpreted safely."""


def _domain_error_from_chain(error: BaseException) -> Optional[HoldStateError]:
    """Recover a typed Hold failure wrapped by a transaction backend."""

    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HoldStateError):
            return current
        current = current.__cause__ or current.__context__
    return None


@dataclass(frozen=True)
class HoldState:
    scope: HoldScope
    target_id: str
    reason: str
    actor_id: str
    set_at: str
    hold_receipt_id: str
    revision: int


@dataclass(frozen=True)
class EffectiveHoldState:
    """Independent latches applying to one agent."""

    host: Optional[HoldState]
    agent: Optional[HoldState]

    @property
    def held(self) -> bool:
        return self.host is not None or self.agent is not None

    @property
    def sources(self) -> tuple[HoldScope, ...]:
        sources: list[HoldScope] = []
        if self.host is not None:
            sources.append(HoldScope.HOST)
        if self.agent is not None:
            sources.append(HoldScope.AGENT)
        return tuple(sources)


@dataclass(frozen=True)
class HoldReceipt:
    receipt_id: str
    operation_id: str
    action: HoldAction
    disposition: HoldDisposition
    scope: HoldScope
    target_id: str
    reason: str
    actor_id: str
    occurred_at: str
    expected_hold_receipt_id: str
    prior_hold_receipt_id: str
    resulting_hold_receipt_id: str


@dataclass(frozen=True)
class HoldMutation:
    receipt: HoldReceipt
    current: Optional[HoldState]


def _terminal_authority_ids(
    authorities: Mapping[str, HoldReceipt],
    consumers: Mapping[str, HoldReceipt],
) -> set[str]:
    """Return open Hold authorities with one linear visit per authority.

    Applied receipts form a functional graph: an authority has at most one
    successor, and a Hold successor is itself an authority.  Remembering every
    completed suffix avoids re-walking the full history from each ancestor.
    Any path that revisits its own unfinished suffix is a closed cycle.
    """

    terminal_authorities: set[str] = set()
    completed: set[str] = set()
    for authority_id in authorities:
        if authority_id in completed:
            continue
        cursor = authority_id
        path: list[str] = []
        path_positions: dict[str, int] = {}
        while cursor not in completed:
            if cursor in path_positions:
                raise HoldCorruptStateError("Hold receipt graph contains a cycle")
            path_positions[cursor] = len(path)
            path.append(cursor)
            successor = consumers.get(cursor)
            if successor is None:
                terminal_authorities.add(cursor)
                break
            if successor.action is HoldAction.RELEASE:
                break
            cursor = successor.receipt_id
        completed.update(path)
    return terminal_authorities


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _coerce_scope(value: HoldScope | str) -> HoldScope:
    if isinstance(value, HoldScope):
        return value
    try:
        return HoldScope(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("scope must be 'host' or 'agent'") from exc


def _target(scope: HoldScope, target_id: Optional[str]) -> str:
    if scope is HoldScope.HOST:
        if target_id not in (None, "", HOST_HOLD_TARGET):
            raise ValueError("host Hold target is fixed by the host control store")
        return HOST_HOLD_TARGET
    return _required_text(target_id, "target_id")


def _exact_nonnegative_revision(value: object) -> int:
    """Accept only the exact integer representation the latch schema promises."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HoldCorruptStateError("hold latch revision is invalid")
    return value


def _exact_active_flag(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in (0, 1)
    ):
        raise HoldCorruptStateError("hold latch active flag is invalid")
    return value


def _aware_timestamp(value: object, field: str) -> str:
    """Validate persisted Hold time evidence without rewriting its digest."""

    if not isinstance(value, str) or not value:
        raise HoldCorruptStateError(f"hold {field} timestamp is invalid")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise HoldCorruptStateError(
            f"hold {field} timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HoldCorruptStateError(
            f"hold {field} timestamp must be timezone-aware"
        )
    return value


def _latch_from_row(row: Any) -> Optional[HoldState]:
    if row is None:
        return None
    if len(row) != 8:
        raise HoldCorruptStateError("hold latch row has an unexpected shape")
    try:
        scope = HoldScope(str(row[0]))
        target_id = str(row[1])
        active = _exact_active_flag(row[2])
        revision = _exact_nonnegative_revision(row[7])
    except (TypeError, ValueError) as exc:
        raise HoldCorruptStateError("hold latch row has invalid typed fields") from exc
    hold_receipt_id = str(row[3] or "")
    reason = str(row[4] or "")
    actor_id = str(row[5] or "")
    set_at = str(row[6] or "")
    if not target_id:
        raise HoldCorruptStateError("hold latch is missing its target identity")
    if scope is HoldScope.HOST and target_id != HOST_HOLD_TARGET:
        raise HoldCorruptStateError("host hold latch has a foreign target")
    evidence = (hold_receipt_id, reason, actor_id, set_at)
    if active == 0:
        if any(evidence):
            raise HoldCorruptStateError(
                "inactive latch retains active hold evidence"
            )
        return None
    if not all(evidence):
        raise HoldCorruptStateError("active hold latch is missing required evidence")
    _aware_timestamp(set_at, "latch")
    return HoldState(
        scope=scope,
        target_id=target_id,
        reason=reason,
        actor_id=actor_id,
        set_at=set_at,
        hold_receipt_id=hold_receipt_id,
        revision=revision,
    )


def _receipt_from_row(row: Any) -> HoldReceipt:
    if row is None or len(row) != 12:
        raise HoldCorruptStateError("hold receipt row has an unexpected shape")
    if any(value is None for value in row):
        # SQLite does not implicitly make a non-INTEGER PRIMARY KEY non-null,
        # and an imported/older schema may lack any of the v1 constraints.
        # Never manufacture the string ``"None"`` as durable audit evidence.
        raise HoldCorruptStateError("hold receipt has missing required evidence")
    if any(not isinstance(value, str) for value in row):
        raise HoldCorruptStateError("hold receipt has invalid evidence types")
    try:
        receipt = HoldReceipt(
            receipt_id=str(row[0]),
            operation_id=str(row[1]),
            action=HoldAction(str(row[2])),
            disposition=HoldDisposition(str(row[3])),
            scope=HoldScope(str(row[4])),
            target_id=str(row[5]),
            reason=str(row[6] or ""),
            actor_id=str(row[7]),
            occurred_at=str(row[8]),
            expected_hold_receipt_id=str(row[9] or ""),
            prior_hold_receipt_id=str(row[10] or ""),
            resulting_hold_receipt_id=str(row[11] or ""),
        )
    except (TypeError, ValueError) as exc:
        raise HoldCorruptStateError("hold receipt has invalid typed fields") from exc
    common_evidence = (
        receipt.receipt_id,
        receipt.operation_id,
        receipt.target_id,
        receipt.reason,
        receipt.actor_id,
        receipt.occurred_at,
    )
    if not all(common_evidence):
        raise HoldCorruptStateError("hold receipt invariant is invalid")
    _aware_timestamp(receipt.occurred_at, "receipt")
    if receipt.scope is HoldScope.HOST and receipt.target_id != HOST_HOLD_TARGET:
        raise HoldCorruptStateError("hold receipt invariant is invalid")

    prior = receipt.prior_hold_receipt_id
    resulting = receipt.resulting_hold_receipt_id
    expected = receipt.expected_hold_receipt_id
    if receipt.action is HoldAction.HOLD:
        valid = expected == "" and (
            (
                receipt.disposition is HoldDisposition.APPLIED
                and resulting == receipt.receipt_id
            )
            or (
                receipt.disposition is HoldDisposition.ALREADY_IN_STATE
                and bool(prior)
                and resulting == prior
            )
        )
    else:
        valid = bool(expected) and (
            (
                receipt.disposition is HoldDisposition.APPLIED
                and prior == expected
                and resulting == ""
            )
            or (
                receipt.disposition is HoldDisposition.ALREADY_IN_STATE
                and prior == ""
                and resulting == ""
            )
            or (
                receipt.disposition is HoldDisposition.REFUSED_STALE
                and bool(prior)
                and prior != expected
                and resulting == prior
            )
        )
    if not valid:
        raise HoldCorruptStateError("hold receipt invariant is invalid")
    return receipt


def _receipt_content_digest(row: Any) -> str:
    """Hash every typed receipt field with unambiguous length framing."""

    _receipt_from_row(row)
    digest = hashlib.sha256()
    for value in row:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class HoldStore:
    """Durable latch + append-only receipt store on an ``AsyncDatabase``."""

    def __init__(self, db: Any):
        self._db = db

    async def ensure_schema(self) -> None:
        """Create the Hold schema while preserving typed integrity failures."""

        try:
            await self._ensure_schema_transaction()
        except BaseException as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _ensure_schema_transaction(self) -> None:
        """Create both Hold tables as one serialized schema unit."""

        async with self._db.migration_lock(_SCHEMA_LOCK):
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_latches ("
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "active INTEGER NOT NULL DEFAULT 0, "
                "hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "reason TEXT NOT NULL DEFAULT '', "
                "actor_id TEXT NOT NULL DEFAULT '', "
                "set_at TEXT NOT NULL DEFAULT '', "
                "revision INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (scope, target_id), "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'), "
                "CHECK (active IN (0, 1)), "
                "CHECK (revision >= 0))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_receipts ("
                "receipt_id TEXT NOT NULL PRIMARY KEY, "
                "operation_id TEXT NOT NULL UNIQUE, "
                "action TEXT NOT NULL, "
                "disposition TEXT NOT NULL, "
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "reason TEXT NOT NULL DEFAULT '', "
                "actor_id TEXT NOT NULL, "
                "occurred_at TEXT NOT NULL, "
                "expected_hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "prior_hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "resulting_hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "CHECK (action IN ('hold', 'release')), "
                "CHECK (disposition IN "
                "('applied', 'already_in_state', 'refused_stale')), "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'))"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_hold_receipts_target "
                "ON hold_receipts(scope, target_id, occurred_at, receipt_id)"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_receipt_witnesses ("
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "receipt_count INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (scope, target_id), "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'), "
                "CHECK (receipt_count >= 0))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_receipt_content_witnesses ("
                "receipt_id TEXT NOT NULL PRIMARY KEY, "
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "receipt_digest TEXT NOT NULL, "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'))"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_hold_receipt_content_witnesses_target "
                "ON hold_receipt_content_witnesses(scope, target_id)"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_operation_witnesses ("
                "operation_id TEXT NOT NULL PRIMARY KEY, "
                "receipt_id TEXT NOT NULL UNIQUE)"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_schema_migrations ("
                "name TEXT NOT NULL PRIMARY KEY)"
            )
            migration_complete = await self._db.fetchone(
                "SELECT 1 FROM hold_schema_migrations WHERE name = ?",
                (_WITNESS_BACKFILL,),
            )
            if migration_complete is not None:
                await self._assert_completed_witness_migration_intact()
                return

            # Seed the witness exactly once for upgraded databases. Future
            # schema checks never derive missing witnesses from mutable receipt
            # rows after the durable migration marker exists. Without that
            # gate, deleting a witness on a later boot re-blessed whatever
            # receipt content happened to remain.
            await self._db.execute(
                "INSERT INTO hold_receipt_witnesses "
                "(scope, target_id, receipt_count) "
                "SELECT scope, target_id, COUNT(*) FROM hold_receipts "
                "WHERE scope <> 'host' OR target_id = 'host' "
                "GROUP BY scope, target_id "
                "ON CONFLICT (scope, target_id) DO NOTHING"
            )
            # One-time backfill for upgraded databases. A witness is never
            # overwritten from receipt rows after it exists, so later in-place
            # mutation remains detectable across schema checks and restarts.
            missing_content_witnesses = await self._db.fetchall(
                "SELECT r.receipt_id, r.operation_id, r.action, r.disposition, "
                "r.scope, r.target_id, r.reason, r.actor_id, r.occurred_at, "
                "r.expected_hold_receipt_id, r.prior_hold_receipt_id, "
                "r.resulting_hold_receipt_id FROM hold_receipts AS r "
                "LEFT JOIN hold_receipt_content_witnesses AS w "
                "ON w.receipt_id = r.receipt_id WHERE w.receipt_id IS NULL"
            )
            for row in missing_content_witnesses:
                receipt = _receipt_from_row(row)
                await self._db.execute(
                    "INSERT INTO hold_receipt_content_witnesses "
                    "(receipt_id, scope, target_id, receipt_digest) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (receipt_id) DO NOTHING",
                    (
                        receipt.receipt_id,
                        receipt.scope.value,
                        receipt.target_id,
                        _receipt_content_digest(row),
                    ),
                )
            # Operation identities are global, not target-local. Keep a
            # separate append-only tombstone so deleting a receipt cannot make
            # its operation id available to a different target. The anti-join
            # keeps repeat startup proportional to genuinely missing legacy
            # witnesses rather than total receipt history.
            await self._db.execute(
                "INSERT INTO hold_operation_witnesses (operation_id, receipt_id) "
                "SELECT r.operation_id, r.receipt_id FROM hold_receipts AS r "
                "LEFT JOIN hold_operation_witnesses AS w "
                "ON w.operation_id = r.operation_id "
                "WHERE w.operation_id IS NULL "
                "ON CONFLICT (operation_id) DO NOTHING"
            )
            await self._db.execute(
                "INSERT INTO hold_receipt_witnesses "
                "(scope, target_id, receipt_count) "
                "SELECT scope, target_id, 0 FROM hold_latches "
                "WHERE scope <> 'host' OR target_id = 'host' "
                "ON CONFLICT (scope, target_id) DO NOTHING"
            )
            await self._db.execute(
                "INSERT INTO hold_schema_migrations (name) VALUES (?) "
                "ON CONFLICT (name) DO NOTHING",
                (_WITNESS_BACKFILL,),
            )

    async def _assert_completed_witness_migration_intact(self) -> None:
        """Fail closed if a completed migration later loses any witness."""

        missing_content = await self._db.fetchone(
            "SELECT r.receipt_id FROM hold_receipts AS r "
            "LEFT JOIN hold_receipt_content_witnesses AS w "
            "ON w.receipt_id = r.receipt_id "
            "WHERE w.receipt_id IS NULL LIMIT 1"
        )
        if missing_content is not None:
            raise HoldCorruptStateError(
                "completed Hold witness migration is missing a content witness"
            )

        missing_operation = await self._db.fetchone(
            "SELECT r.operation_id FROM hold_receipts AS r "
            "LEFT JOIN hold_operation_witnesses AS w "
            "ON w.operation_id = r.operation_id "
            "WHERE w.operation_id IS NULL LIMIT 1"
        )
        if missing_operation is not None:
            raise HoldCorruptStateError(
                "completed Hold witness migration is missing an operation witness"
            )

        missing_count = await self._db.fetchone(
            "SELECT source.scope, source.target_id FROM ("
            "SELECT scope, target_id FROM hold_receipts "
            "UNION SELECT scope, target_id FROM hold_latches"
            ") AS source LEFT JOIN hold_receipt_witnesses AS w "
            "ON w.scope = source.scope AND w.target_id = source.target_id "
            "WHERE w.scope IS NULL LIMIT 1"
        )
        if missing_count is not None:
            raise HoldCorruptStateError(
                "completed Hold witness migration is missing a receipt-count witness"
            )

    async def _lock_operation_and_target(
        self, operation_id: str, scope: HoldScope, target_id: str
    ) -> None:
        if getattr(self._db, "backend_type", "") != "postgres":
            return
        # One global acquisition order for every writer: operation first,
        # target second.  The operation lock closes cross-target reuse of one
        # id; the target lock closes the absent-row authorization/mutation gap.
        for key in (
            f"kestrel:hold:operation:{operation_id}",
            f"kestrel:hold:target:{scope.value}:{target_id}",
        ):
            await self._db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (key,),
            )

    async def _lock_read_targets(
        self, targets: tuple[tuple[HoldScope, str], ...]
    ) -> None:
        """Serialize a PostgreSQL read snapshot with legitimate target writers."""

        if getattr(self._db, "backend_type", "") != "postgres":
            return
        keys = sorted(
            {
                f"kestrel:hold:target:{scope.value}:{target_id}"
                for scope, target_id in targets
            }
        )
        for key in keys:
            await self._db.execute(
                "SELECT pg_advisory_xact_lock_shared(hashtextextended(?, 0))",
                (key,),
            )

    async def _ensure_latch_row(self, scope: HoldScope, target_id: str) -> None:
        await self._db.execute(
            "INSERT INTO hold_latches (scope, target_id) VALUES (?, ?) "
            "ON CONFLICT (scope, target_id) DO NOTHING",
            (scope.value, target_id),
        )
        await self._db.execute(
            "INSERT INTO hold_receipt_witnesses "
            "(scope, target_id, receipt_count) VALUES (?, ?, 0) "
            "ON CONFLICT (scope, target_id) DO NOTHING",
            (scope.value, target_id),
        )

    async def _read_latch_row(
        self, scope: HoldScope, target_id: str, *, for_update: bool = False
    ) -> Any:
        suffix = (
            " FOR UPDATE"
            if for_update and getattr(self._db, "backend_type", "") == "postgres"
            else ""
        )
        rows = await self._db.fetchall(
            f"SELECT {_LATCH_COLUMNS} FROM hold_latches "
            f"WHERE scope = ? AND target_id = ?{suffix}",
            (scope.value, target_id),
        )
        if len(rows) > 1:
            raise HoldCorruptStateError("duplicate hold latch key")
        return rows[0] if rows else None

    async def _assert_host_latch_shape(self, *, for_update: bool = False) -> None:
        """Fail closed if an upgraded/shared database has a foreign host row."""

        suffix = (
            " FOR UPDATE"
            if for_update and getattr(self._db, "backend_type", "") == "postgres"
            else ""
        )
        latch_rows = await self._db.fetchall(
            "SELECT target_id FROM hold_latches WHERE scope = ?" + suffix,
            (HoldScope.HOST.value,),
        )
        receipt_rows = await self._db.fetchall(
            "SELECT target_id FROM hold_receipts WHERE scope = ?" + suffix,
            (HoldScope.HOST.value,),
        )
        if any(
            str(row[0]) != HOST_HOLD_TARGET
            for row in (*latch_rows, *receipt_rows)
        ):
            raise HoldCorruptStateError(
                "host hold state has a foreign target identity"
            )

    async def _read_receipt_by_operation(self, operation_id: str) -> Any:
        rows = await self._db.fetchall(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts WHERE operation_id = ?",
            (operation_id,),
        )
        if len(rows) > 1:
            raise HoldCorruptStateError(
                "Hold receipt history has a duplicate operation identity"
            )
        return rows[0] if rows else None

    async def _validate_operation_witness(self, operation_id: str) -> Any:
        """Return the receipt row only when its global identity is intact."""

        witnesses = await self._db.fetchall(
            "SELECT receipt_id FROM hold_operation_witnesses "
            "WHERE operation_id = ?",
            (operation_id,),
        )
        if len(witnesses) > 1:
            raise HoldCorruptStateError(
                "Hold operation witness has a duplicate operation identity"
            )
        receipt_row = await self._read_receipt_by_operation(operation_id)
        if not witnesses:
            if receipt_row is not None:
                raise HoldCorruptStateError(
                    "Hold receipt is missing its global operation witness"
                )
            return None
        if receipt_row is None:
            raise HoldCorruptStateError(
                "Hold operation witness refers to a missing receipt"
            )
        if str(witnesses[0][0]) != str(receipt_row[0]):
            raise HoldCorruptStateError(
                "Hold operation witness does not match receipt identity"
            )
        return receipt_row

    async def _read_receipt_by_id(self, receipt_id: str) -> Any:
        return await self._db.fetchone(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts WHERE receipt_id = ?",
            (receipt_id,),
        )

    async def _validate_receipt_authority_graph(
        self,
        latch: Optional[HoldState],
        scope: HoldScope,
        target_id: str,
    ) -> None:
        """Prove the append-only authority graph agrees with its latch.

        Each applied Hold creates one authority. A later applied Hold or
        Release may consume it exactly once. Every chain must be acyclic and
        terminate either in an applied Release or in the one active authority
        named by the latch. This catches projection deletion, latch rewind,
        forked successors, and closed cycles rather than trusting a locally
        well-formed latch row.
        """

        rows = await self._db.fetchall(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts "
            "WHERE scope = ? AND target_id = ?",
            (scope.value, target_id),
        )
        receipts = [_receipt_from_row(row) for row in rows]
        receipt_ids = {receipt.receipt_id for receipt in receipts}
        if len(receipt_ids) != len(receipts):
            raise HoldCorruptStateError("Hold receipt graph has duplicate identities")
        witness_rows = await self._db.fetchall(
            "SELECT receipt_id, receipt_digest "
            "FROM hold_receipt_content_witnesses "
            "WHERE scope = ? AND target_id = ?",
            (scope.value, target_id),
        )
        content_witnesses: dict[str, str] = {}
        content_witness_valid = True
        for witness in witness_rows:
            if (
                len(witness) != 2
                or not isinstance(witness[0], str)
                or not witness[0]
                or not isinstance(witness[1], str)
                or len(witness[1]) != 64
                or witness[0] in content_witnesses
            ):
                content_witness_valid = False
                continue
            content_witnesses[witness[0]] = witness[1]
        if set(content_witnesses) != receipt_ids:
            content_witness_valid = False
        for row, receipt in zip(rows, receipts):
            if content_witnesses.get(receipt.receipt_id) != _receipt_content_digest(row):
                content_witness_valid = False
        projection_row = await self._read_latch_row(scope, target_id)
        projection_revision = 0
        if projection_row is not None:
            if len(projection_row) != 8:
                raise HoldCorruptStateError(
                    "hold latch row has an unexpected shape"
                )
            try:
                projection_revision = _exact_nonnegative_revision(
                    projection_row[7]
                )
            except (TypeError, ValueError) as exc:
                raise HoldCorruptStateError(
                    "hold latch row has invalid typed fields"
                ) from exc
        witness_row = await self._db.fetchone(
            "SELECT receipt_count FROM hold_receipt_witnesses "
            "WHERE scope = ? AND target_id = ?",
            (scope.value, target_id),
        )
        if witness_row is None and not receipts and projection_row is None:
            witnessed_receipts = 0
        elif witness_row is None or len(witness_row) != 1:
            raise HoldCorruptStateError("Hold receipt-count witness is missing")
        else:
            try:
                witnessed_receipts = _exact_nonnegative_revision(witness_row[0])
            except (TypeError, ValueError) as exc:
                raise HoldCorruptStateError(
                    "Hold receipt-count witness has invalid typed fields"
                ) from exc
        applied = [
            receipt
            for receipt in receipts
            if receipt.disposition is HoldDisposition.APPLIED
        ]
        authorities = {
            receipt.receipt_id: receipt
            for receipt in applied
            if receipt.action is HoldAction.HOLD
        }
        # Every prior/resulting reference names an applied Hold authority,
        # including non-applied audit outcomes.  ALREADY_IN_STATE and
        # REFUSED_STALE do not consume authority, but accepting a dangling
        # reference would let an idempotent replay return a receipt whose
        # requested latch never existed (or was deleted).
        for receipt in receipts:
            for authority_id in {
                receipt.prior_hold_receipt_id,
                receipt.resulting_hold_receipt_id,
            }:
                if authority_id and authority_id not in authorities:
                    raise HoldCorruptStateError(
                        "Hold history references missing authority receipt"
                    )
        consumers: dict[str, HoldReceipt] = {}
        for receipt in applied:
            prior = receipt.prior_hold_receipt_id
            if not prior:
                continue
            if prior not in authorities:
                raise HoldCorruptStateError(
                    "applied Hold history consumes missing authority"
                )
            if prior in consumers:
                raise HoldCorruptStateError(
                    "applied Hold authority has multiple successors"
                )
            consumers[prior] = receipt

        terminal_authorities = _terminal_authority_ids(authorities, consumers)

        if latch is None:
            if witnessed_receipts != len(receipts):
                raise HoldCorruptStateError(
                    "hold receipt-count revision does not match receipt history"
                )
            if not terminal_authorities:
                if projection_revision != len(applied):
                    raise HoldCorruptStateError(
                        "hold latch revision does not match applied receipt history"
                    )
                if not content_witness_valid:
                    raise HoldCorruptStateError(
                        "Hold receipt content witness does not match receipt history"
                    )
                return
            raise HoldCorruptStateError(
                "unheld projection retains active Hold authority"
            )

        receipt = authorities.get(latch.hold_receipt_id)
        if receipt is None:
            referenced_row = await self._read_receipt_by_id(latch.hold_receipt_id)
            if referenced_row is not None:
                _receipt_from_row(referenced_row)
                raise HoldCorruptStateError(
                    "active hold latch does not match its authority receipt"
                )
            raise HoldCorruptStateError(
                "active hold latch references a missing authority receipt"
            )
        if (
            receipt.scope is not latch.scope
            or receipt.target_id != latch.target_id
            or receipt.reason != latch.reason
            or receipt.actor_id != latch.actor_id
            or receipt.occurred_at != latch.set_at
        ):
            raise HoldCorruptStateError(
                "active hold latch does not match its authority receipt"
            )
        if terminal_authorities != {latch.hold_receipt_id}:
            raise HoldCorruptStateError(
                "active hold latch is not the receipt graph's terminal authority"
            )
        if witnessed_receipts != len(receipts):
            raise HoldCorruptStateError(
                "hold receipt-count revision does not match receipt history"
            )
        if projection_revision != len(applied):
            raise HoldCorruptStateError(
                "hold latch revision does not match applied receipt history"
            )
        if not content_witness_valid:
            raise HoldCorruptStateError(
                "Hold receipt content witness does not match receipt history"
            )

    async def _validate_latch_projection(
        self,
        latch: Optional[HoldState],
        scope: HoldScope,
        target_id: str,
    ) -> None:
        await self._validate_receipt_authority_graph(latch, scope, target_id)

    @staticmethod
    def _assert_replay(
        receipt: HoldReceipt,
        *,
        action: HoldAction,
        scope: HoldScope,
        target_id: str,
        reason: str,
        actor_id: str,
        expected_hold_receipt_id: str,
    ) -> None:
        supplied = (
            action,
            scope,
            target_id,
            reason,
            actor_id,
            expected_hold_receipt_id,
        )
        recorded = (
            receipt.action,
            receipt.scope,
            receipt.target_id,
            receipt.reason,
            receipt.actor_id,
            receipt.expected_hold_receipt_id,
        )
        if supplied != recorded:
            raise HoldIdempotencyConflict(
                "hold operation id was already used for a different mutation"
            )

    async def _insert_receipt(
        self,
        *,
        operation_id: str,
        action: HoldAction,
        disposition: HoldDisposition,
        scope: HoldScope,
        target_id: str,
        reason: str,
        actor_id: str,
        expected_hold_receipt_id: str,
        prior_hold_receipt_id: str,
        resulting_hold_receipt_id: str,
        receipt_id: Optional[str] = None,
    ) -> HoldReceipt:
        receipt_id = receipt_id or str(uuid4())
        now_sql = database_now_sql(self._db)
        await self._db.execute(
            "INSERT INTO hold_receipts ("
            "receipt_id, operation_id, action, disposition, scope, target_id, "
            "reason, actor_id, occurred_at, expected_hold_receipt_id, "
            "prior_hold_receipt_id, resulting_hold_receipt_id"
            f") VALUES (?, ?, ?, ?, ?, ?, ?, ?, {now_sql}, ?, ?, ?)",
            (
                receipt_id,
                operation_id,
                action.value,
                disposition.value,
                scope.value,
                target_id,
                reason,
                actor_id,
                expected_hold_receipt_id,
                prior_hold_receipt_id,
                resulting_hold_receipt_id,
            ),
        )
        receipt_row = await self._read_receipt_by_operation(operation_id)
        receipt = _receipt_from_row(receipt_row)
        await self._db.execute(
            "INSERT INTO hold_operation_witnesses (operation_id, receipt_id) "
            "VALUES (?, ?)",
            (receipt.operation_id, receipt.receipt_id),
        )
        await self._db.execute(
            "INSERT INTO hold_receipt_content_witnesses "
            "(receipt_id, scope, target_id, receipt_digest) "
            "VALUES (?, ?, ?, ?)",
            (
                receipt.receipt_id,
                receipt.scope.value,
                receipt.target_id,
                _receipt_content_digest(receipt_row),
            ),
        )
        await self._db.execute(
            "UPDATE hold_receipt_witnesses "
            "SET receipt_count = receipt_count + 1 "
            "WHERE scope = ? AND target_id = ?",
            (scope.value, target_id),
        )
        return receipt

    async def set_hold(
        self,
        *,
        scope: HoldScope | str,
        actor_id: str,
        reason: str,
        operation_id: str,
        target_id: Optional[str] = None,
    ) -> HoldMutation:
        """Set or replace one latch and append an immutable receipt."""

        try:
            return await self._set_hold(
                scope=scope,
                actor_id=actor_id,
                reason=reason,
                operation_id=operation_id,
                target_id=target_id,
            )
        except Exception as exc:
            # AsyncDatabase deliberately wraps transaction-body exceptions.
            # Every Hold domain failure is part of this store's public
            # contract, so preserve its type across that backend boundary.
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _set_hold(
        self,
        *,
        scope: HoldScope | str,
        actor_id: str,
        reason: str,
        operation_id: str,
        target_id: Optional[str] = None,
    ) -> HoldMutation:
        resolved_scope = _coerce_scope(scope)
        resolved_target = _target(resolved_scope, target_id)
        actor = _required_text(actor_id, "actor_id")
        why = _required_text(reason, "reason")
        operation = _required_text(operation_id, "operation_id")

        async with self._db.transaction(immediate=True):
            await self._assert_host_latch_shape(
                for_update=resolved_scope is HoldScope.HOST
            )
            await self._lock_operation_and_target(
                operation, resolved_scope, resolved_target
            )
            await self._ensure_latch_row(resolved_scope, resolved_target)
            prior_row = await self._read_latch_row(
                resolved_scope, resolved_target, for_update=True
            )
            prior = _latch_from_row(prior_row)
            await self._validate_latch_projection(
                prior, resolved_scope, resolved_target
            )
            replay_row = await self._validate_operation_witness(operation)
            if replay_row is not None:
                replay = _receipt_from_row(replay_row)
                self._assert_replay(
                    replay,
                    action=HoldAction.HOLD,
                    scope=resolved_scope,
                    target_id=resolved_target,
                    reason=why,
                    actor_id=actor,
                    expected_hold_receipt_id="",
                )
                return HoldMutation(receipt=replay, current=prior)

            if prior is not None and prior.actor_id == actor and prior.reason == why:
                receipt = await self._insert_receipt(
                    operation_id=operation,
                    action=HoldAction.HOLD,
                    disposition=HoldDisposition.ALREADY_IN_STATE,
                    scope=resolved_scope,
                    target_id=resolved_target,
                    reason=why,
                    actor_id=actor,
                    expected_hold_receipt_id="",
                    prior_hold_receipt_id=prior.hold_receipt_id,
                    resulting_hold_receipt_id=prior.hold_receipt_id,
                )
                return HoldMutation(receipt=receipt, current=prior)

            receipt_id = str(uuid4())
            receipt = await self._insert_receipt(
                operation_id=operation,
                action=HoldAction.HOLD,
                disposition=HoldDisposition.APPLIED,
                scope=resolved_scope,
                target_id=resolved_target,
                reason=why,
                actor_id=actor,
                expected_hold_receipt_id="",
                prior_hold_receipt_id=(prior.hold_receipt_id if prior else ""),
                resulting_hold_receipt_id=receipt_id,
                receipt_id=receipt_id,
            )
            await self._db.execute(
                "UPDATE hold_latches SET active = 1, hold_receipt_id = ?, "
                "reason = ?, actor_id = ?, set_at = ?, revision = revision + 1 "
                "WHERE scope = ? AND target_id = ?",
                (
                    receipt.receipt_id,
                    why,
                    actor,
                    receipt.occurred_at,
                    resolved_scope.value,
                    resolved_target,
                ),
            )
            current = _latch_from_row(
                await self._read_latch_row(resolved_scope, resolved_target)
            )
            return HoldMutation(receipt=receipt, current=current)

    async def release_hold(
        self,
        *,
        scope: HoldScope | str,
        actor_id: str,
        reason: str,
        operation_id: str,
        expected_hold_receipt_id: str,
        target_id: Optional[str] = None,
    ) -> HoldMutation:
        """Release exactly the observed latch, refusing a stale release."""

        try:
            return await self._release_hold(
                scope=scope,
                actor_id=actor_id,
                reason=reason,
                operation_id=operation_id,
                expected_hold_receipt_id=expected_hold_receipt_id,
                target_id=target_id,
            )
        except Exception as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _release_hold(
        self,
        *,
        scope: HoldScope | str,
        actor_id: str,
        reason: str,
        operation_id: str,
        expected_hold_receipt_id: str,
        target_id: Optional[str] = None,
    ) -> HoldMutation:
        resolved_scope = _coerce_scope(scope)
        resolved_target = _target(resolved_scope, target_id)
        actor = _required_text(actor_id, "actor_id")
        why = _required_text(reason, "reason")
        operation = _required_text(operation_id, "operation_id")
        expected = _required_text(
            expected_hold_receipt_id, "expected_hold_receipt_id"
        )

        async with self._db.transaction(immediate=True):
            await self._assert_host_latch_shape(
                for_update=resolved_scope is HoldScope.HOST
            )
            await self._lock_operation_and_target(
                operation, resolved_scope, resolved_target
            )
            await self._ensure_latch_row(resolved_scope, resolved_target)
            prior_row = await self._read_latch_row(
                resolved_scope, resolved_target, for_update=True
            )
            prior = _latch_from_row(prior_row)
            await self._validate_latch_projection(
                prior, resolved_scope, resolved_target
            )
            replay_row = await self._validate_operation_witness(operation)
            if replay_row is not None:
                replay = _receipt_from_row(replay_row)
                self._assert_replay(
                    replay,
                    action=HoldAction.RELEASE,
                    scope=resolved_scope,
                    target_id=resolved_target,
                    reason=why,
                    actor_id=actor,
                    expected_hold_receipt_id=expected,
                )
                return HoldMutation(receipt=replay, current=prior)

            if prior is None:
                disposition = HoldDisposition.ALREADY_IN_STATE
                prior_receipt_id = ""
                resulting_receipt_id = ""
            elif prior.hold_receipt_id != expected:
                disposition = HoldDisposition.REFUSED_STALE
                prior_receipt_id = prior.hold_receipt_id
                resulting_receipt_id = prior.hold_receipt_id
            else:
                disposition = HoldDisposition.APPLIED
                prior_receipt_id = prior.hold_receipt_id
                resulting_receipt_id = ""

            receipt = await self._insert_receipt(
                operation_id=operation,
                action=HoldAction.RELEASE,
                disposition=disposition,
                scope=resolved_scope,
                target_id=resolved_target,
                reason=why,
                actor_id=actor,
                expected_hold_receipt_id=expected,
                prior_hold_receipt_id=prior_receipt_id,
                resulting_hold_receipt_id=resulting_receipt_id,
            )
            if disposition is HoldDisposition.APPLIED:
                await self._db.execute(
                    "UPDATE hold_latches SET active = 0, hold_receipt_id = '', "
                    "reason = '', actor_id = '', set_at = '', revision = revision + 1 "
                    "WHERE scope = ? AND target_id = ? AND active = 1 "
                    "AND hold_receipt_id = ?",
                    (resolved_scope.value, resolved_target, expected),
                )
            current = _latch_from_row(
                await self._read_latch_row(resolved_scope, resolved_target)
            )
            return HoldMutation(receipt=receipt, current=current)

    async def get_hold(
        self, scope: HoldScope | str, target_id: Optional[str] = None
    ) -> Optional[HoldState]:
        try:
            return await self._get_hold(scope, target_id)
        except Exception as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _get_hold(
        self, scope: HoldScope | str, target_id: Optional[str] = None
    ) -> Optional[HoldState]:
        resolved_scope = _coerce_scope(scope)
        resolved_target = _target(resolved_scope, target_id)
        targets = ((resolved_scope, resolved_target),)
        async with self._db.transaction():
            await self._lock_read_targets(targets)
            await self._assert_host_latch_shape()
            latch = _latch_from_row(
                await self._read_latch_row(resolved_scope, resolved_target)
            )
            await self._validate_latch_projection(
                latch, resolved_scope, resolved_target
            )
            return latch

    async def get_effective(self, agent_id: str) -> EffectiveHoldState:
        """Read host + agent latches in one database snapshot."""

        try:
            return await self._get_effective(agent_id)
        except Exception as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _get_effective(self, agent_id: str) -> EffectiveHoldState:
        """Read host + agent latches in one locked database snapshot."""

        agent = _required_text(agent_id, "agent_id")
        targets = (
            (HoldScope.HOST, HOST_HOLD_TARGET),
            (HoldScope.AGENT, agent),
        )
        async with self._db.transaction():
            await self._lock_read_targets(targets)
            await self._assert_host_latch_shape()
            rows = await self._db.fetchall(
                f"SELECT {_LATCH_COLUMNS} FROM hold_latches "
                "WHERE (scope = ? AND target_id = ?) "
                "OR (scope = ? AND target_id = ?)",
                (
                    HoldScope.HOST.value,
                    HOST_HOLD_TARGET,
                    HoldScope.AGENT.value,
                    agent,
                ),
            )
            host: Optional[HoldState] = None
            agent_state: Optional[HoldState] = None
            seen: set[tuple[str, str]] = set()
            for row in rows:
                key = (str(row[0]), str(row[1]))
                if key in seen:
                    raise HoldCorruptStateError("duplicate hold latch key")
                seen.add(key)
                state = _latch_from_row(row)
                if state is None:
                    continue
                if state.scope is HoldScope.HOST:
                    host = state
                elif state.target_id == agent:
                    agent_state = state
                else:
                    raise HoldCorruptStateError(
                        "effective hold query returned a foreign agent"
                    )
            await self._validate_latch_projection(
                host, HoldScope.HOST, HOST_HOLD_TARGET
            )
            await self._validate_latch_projection(
                agent_state, HoldScope.AGENT, agent
            )
            return EffectiveHoldState(host=host, agent=agent_state)

    async def get_receipt(self, operation_id: str) -> Optional[HoldReceipt]:
        try:
            return await self._get_receipt(operation_id)
        except Exception as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _get_receipt(self, operation_id: str) -> Optional[HoldReceipt]:
        """Read one receipt only after proving its target authority graph."""

        operation = _required_text(operation_id, "operation_id")
        async with self._db.transaction():
            row = await self._validate_operation_witness(operation)
            if row is None:
                return None
            receipt = _receipt_from_row(row)
            targets = ((receipt.scope, receipt.target_id),)
            await self._lock_read_targets(targets)
            await self._assert_host_latch_shape()
            latch = _latch_from_row(
                await self._read_latch_row(receipt.scope, receipt.target_id)
            )
            await self._validate_latch_projection(
                latch, receipt.scope, receipt.target_id
            )
            return receipt


__all__ = [
    "HOST_HOLD_TARGET",
    "EffectiveHoldState",
    "HoldAction",
    "HoldCorruptStateError",
    "HoldDisposition",
    "HoldIdempotencyConflict",
    "HoldMutation",
    "HoldReceipt",
    "HoldScope",
    "HoldState",
    "HoldStateError",
    "HoldStore",
]
