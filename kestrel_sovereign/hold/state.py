"""Typed, durable host and agent Hold latches.

A Hold is state, not a cancellation event.  The current latch and its immutable
mutation receipts live in the host control database so every worker observes
the same answer after a restart.  This module owns only storage and composition;
turn-start refusal is the separate enforcement seam tracked by #3162.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional
from uuid import uuid4

from kestrel_sovereign.storage.database_clock import database_now_sql


HOST_HOLD_TARGET = "host"
_SCHEMA_LOCK = "hold_state_v1"
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


@dataclass(frozen=True)
class _HoldAuthorityCheckpoint:
    scope: HoldScope
    target_id: str
    active: bool
    hold_receipt_id: str
    latch_revision: int
    dirty: bool


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


def _latch_from_row(row: Any) -> Optional[HoldState]:
    if row is None:
        return None
    if len(row) != 8:
        raise HoldCorruptStateError("hold latch row has an unexpected shape")
    try:
        scope = HoldScope(str(row[0]))
        target_id = str(row[1])
        active = int(row[2])
        revision = int(row[7])
    except (TypeError, ValueError) as exc:
        raise HoldCorruptStateError("hold latch row has invalid typed fields") from exc
    if active not in (0, 1):
        raise HoldCorruptStateError("hold latch active flag is invalid")
    if revision < 0:
        raise HoldCorruptStateError("hold latch revision is invalid")
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
    return HoldState(
        scope=scope,
        target_id=target_id,
        reason=reason,
        actor_id=actor_id,
        set_at=set_at,
        hold_receipt_id=hold_receipt_id,
        revision=revision,
    )


def _checkpoint_from_row(row: Any) -> Optional[_HoldAuthorityCheckpoint]:
    if row is None:
        return None
    if len(row) != 6:
        raise HoldCorruptStateError(
            "hold authority checkpoint has an unexpected shape"
        )
    try:
        scope = HoldScope(str(row[0]))
        target_id = str(row[1])
        active = int(row[2])
        latch_revision = int(row[4])
        dirty = int(row[5])
    except (TypeError, ValueError) as exc:
        raise HoldCorruptStateError(
            "hold authority checkpoint has invalid typed fields"
        ) from exc
    hold_receipt_id = str(row[3] or "")
    if active not in (0, 1) or dirty not in (0, 1):
        raise HoldCorruptStateError(
            "hold authority checkpoint has invalid flags"
        )
    if latch_revision < 0:
        raise HoldCorruptStateError(
            "hold authority checkpoint has invalid counters"
        )
    if not target_id:
        raise HoldCorruptStateError(
            "hold authority checkpoint is missing its target identity"
        )
    if scope is HoldScope.HOST and target_id != HOST_HOLD_TARGET:
        raise HoldCorruptStateError(
            "host hold authority checkpoint has a foreign target"
        )
    if bool(active) != bool(hold_receipt_id):
        raise HoldCorruptStateError(
            "hold authority checkpoint active evidence is inconsistent"
        )
    return _HoldAuthorityCheckpoint(
        scope=scope,
        target_id=target_id,
        active=bool(active),
        hold_receipt_id=hold_receipt_id,
        latch_revision=latch_revision,
        dirty=bool(dirty),
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


class HoldStore:
    """Durable latch + append-only receipt store on an ``AsyncDatabase``."""

    def __init__(self, db: Any):
        self._db = db

    async def ensure_schema(self) -> None:
        """Create Hold storage and its bounded admission checkpoint."""

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
                "CREATE TABLE IF NOT EXISTS hold_authority_checkpoints ("
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "active INTEGER NOT NULL DEFAULT 0, "
                "hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "latch_revision INTEGER NOT NULL DEFAULT 0, "
                "dirty INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (scope, target_id), "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'), "
                "CHECK (active IN (0, 1)), "
                "CHECK (dirty IN (0, 1)), "
                "CHECK (latch_revision >= 0))"
            )
            await self._backfill_authority_checkpoints()
            await self._ensure_checkpoint_dirty_triggers()

    async def _backfill_authority_checkpoints(self) -> None:
        """Build a verified checkpoint once for each pre-checkpoint target."""

        rows = await self._db.fetchall(
            "SELECT existing.scope, existing.target_id FROM ("
            "SELECT scope, target_id FROM hold_latches "
            "UNION SELECT scope, target_id FROM hold_receipts"
            ") existing LEFT JOIN hold_authority_checkpoints checkpoint "
            "ON checkpoint.scope = existing.scope "
            "AND checkpoint.target_id = existing.target_id "
            "WHERE checkpoint.scope IS NULL"
        )
        for raw_scope, raw_target in rows:
            try:
                scope = HoldScope(str(raw_scope))
            except (TypeError, ValueError) as exc:
                raise HoldCorruptStateError(
                    "hold checkpoint backfill found an invalid scope"
                ) from exc
            target_id = str(raw_target)
            if scope is HoldScope.HOST and target_id != HOST_HOLD_TARGET:
                # Do not bless an imported foreign host row with a valid
                # checkpoint. Every state path detects it explicitly.
                continue
            latch_row = await self._read_latch_row(scope, target_id)
            latch = _latch_from_row(latch_row)
            receipt_rows = await self._db.fetchall(
                f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts "
                "WHERE scope = ? AND target_id = ?",
                (scope.value, target_id),
            )
            await self._validate_latch_projection(
                latch,
                scope,
                target_id,
                receipt_rows=receipt_rows,
            )
            revision = int(latch_row[7]) if latch_row is not None else 0
            await self._db.execute(
                "INSERT INTO hold_authority_checkpoints ("
                "scope, target_id, active, hold_receipt_id, latch_revision, dirty"
                ") VALUES (?, ?, ?, ?, ?, 0) "
                "ON CONFLICT (scope, target_id) DO NOTHING",
                (
                    scope.value,
                    target_id,
                    int(latch is not None),
                    latch.hold_receipt_id if latch is not None else "",
                    revision,
                ),
            )

    async def _ensure_checkpoint_dirty_triggers(self) -> None:
        """Make any out-of-band receipt mutation invalidate admission."""

        if getattr(self._db, "backend_type", "") == "postgres":
            await self._db.execute(
                "CREATE OR REPLACE FUNCTION kestrel_hold_checkpoint_dirty() "
                "RETURNS trigger AS $$ BEGIN IF TG_OP = 'INSERT' THEN "
                "UPDATE hold_authority_checkpoints SET dirty = 1 "
                "WHERE scope = NEW.scope AND target_id = NEW.target_id; "
                "RETURN NEW; ELSIF TG_OP = 'DELETE' THEN "
                "UPDATE hold_authority_checkpoints SET dirty = 1 "
                "WHERE scope = OLD.scope AND target_id = OLD.target_id; "
                "RETURN OLD; ELSE "
                "UPDATE hold_authority_checkpoints SET dirty = 1 "
                "WHERE (scope = OLD.scope AND target_id = OLD.target_id) "
                "OR (scope = NEW.scope AND target_id = NEW.target_id); "
                "RETURN NEW; END IF; END; $$ LANGUAGE plpgsql"
            )
            await self._db.execute(
                "DROP TRIGGER IF EXISTS hold_receipts_checkpoint_dirty "
                "ON hold_receipts"
            )
            for operation in ("INSERT", "UPDATE", "DELETE"):
                trigger_name = f"hold_receipts_checkpoint_dirty_{operation.lower()}"
                # Trigger names are relation-local in PostgreSQL. A catalog
                # lookup by name alone can find another schema's trigger and
                # leave this authority table without invalidation. Rebuild the
                # inexpensive trigger on the exact target relation instead.
                await self._db.execute(
                    f"DROP TRIGGER IF EXISTS {trigger_name} ON hold_receipts"
                )
                await self._db.execute(
                    f"CREATE TRIGGER {trigger_name} "
                    f"AFTER {operation} ON hold_receipts FOR EACH ROW "
                    "EXECUTE FUNCTION kestrel_hold_checkpoint_dirty()"
                )
            return
        await self._db.execute(
            "DROP TRIGGER IF EXISTS hold_receipts_checkpoint_dirty"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS hold_receipts_checkpoint_dirty_insert "
            "AFTER INSERT ON hold_receipts BEGIN "
            "UPDATE hold_authority_checkpoints SET dirty = 1 "
            "WHERE scope = NEW.scope AND target_id = NEW.target_id; END"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS hold_receipts_checkpoint_dirty_update "
            "AFTER UPDATE ON hold_receipts BEGIN "
            "UPDATE hold_authority_checkpoints SET dirty = 1 "
            "WHERE (scope = OLD.scope AND target_id = OLD.target_id) "
            "OR (scope = NEW.scope AND target_id = NEW.target_id); END"
        )
        await self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS hold_receipts_checkpoint_dirty_delete "
            "AFTER DELETE ON hold_receipts BEGIN "
            "UPDATE hold_authority_checkpoints SET dirty = 1 "
            "WHERE scope = OLD.scope AND target_id = OLD.target_id; END"
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
            "INSERT INTO hold_authority_checkpoints (scope, target_id) "
            "VALUES (?, ?) ON CONFLICT (scope, target_id) DO NOTHING",
            (scope.value, target_id),
        )

    async def _mark_checkpoint_clean(
        self,
        scope: HoldScope,
        target_id: str,
        *,
        latch: Optional[HoldState],
        latch_revision: int,
    ) -> None:
        updated = await self._db.execute(
            "UPDATE hold_authority_checkpoints SET active = ?, "
            "hold_receipt_id = ?, latch_revision = ?, dirty = 0 "
            "WHERE scope = ? AND target_id = ? AND dirty = 1",
            (
                int(latch is not None),
                latch.hold_receipt_id if latch is not None else "",
                latch_revision,
                scope.value,
                target_id,
            ),
        )
        if updated != 1:
            raise HoldCorruptStateError(
                "hold authority checkpoint did not observe its receipt append"
            )

    async def _read_latch_row(
        self, scope: HoldScope, target_id: str, *, for_update: bool = False
    ) -> Any:
        suffix = (
            " FOR UPDATE"
            if for_update and getattr(self._db, "backend_type", "") == "postgres"
            else ""
        )
        return await self._db.fetchone(
            f"SELECT {_LATCH_COLUMNS} FROM hold_latches "
            f"WHERE scope = ? AND target_id = ?{suffix}",
            (scope.value, target_id),
        )

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
        return await self._db.fetchone(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts WHERE operation_id = ?",
            (operation_id,),
        )

    async def _read_receipt_by_id(self, receipt_id: str) -> Any:
        return await self._db.fetchone(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts WHERE receipt_id = ?",
            (receipt_id,),
        )

    async def _validate_admission_projection(
        self,
        *,
        latch_row: Any,
        checkpoint_row: Any,
        authority_receipt_row: Any,
        scope: HoldScope,
        target_id: str,
        receipt_exists: bool,
    ) -> Optional[HoldState]:
        """Validate one constant-size turn-admission integrity projection."""

        latch = _latch_from_row(latch_row)
        checkpoint = _checkpoint_from_row(checkpoint_row)
        if checkpoint is None:
            if latch_row is not None or receipt_exists:
                raise HoldCorruptStateError(
                    "hold authority checkpoint is missing for persisted state"
                )
            if authority_receipt_row is not None:
                raise HoldCorruptStateError(
                    "hold admission snapshot has orphan authority evidence"
                )
            return None
        if checkpoint.scope is not scope or checkpoint.target_id != target_id:
            raise HoldCorruptStateError(
                "hold authority checkpoint returned a foreign target"
            )
        if checkpoint.dirty:
            raise HoldCorruptStateError(
                "hold authority checkpoint has an unprojected receipt append"
            )
        if latch_row is None:
            message = (
                "active Hold authority checkpoint has lost its latch projection"
                if checkpoint.active
                else "Hold authority checkpoint has lost its latch projection"
            )
            raise HoldCorruptStateError(message)
        try:
            latch_active = int(latch_row[2])
            latch_revision = int(latch_row[7])
        except (TypeError, ValueError, IndexError) as exc:
            raise HoldCorruptStateError(
                "hold latch checkpoint fields are invalid"
            ) from exc
        if (
            checkpoint.active != bool(latch_active)
            or checkpoint.latch_revision != latch_revision
            or checkpoint.hold_receipt_id != str(latch_row[3] or "")
        ):
            message = (
                "active Hold authority checkpoint disagrees with its latch projection"
                if checkpoint.active
                else "Hold authority checkpoint disagrees with its latch projection"
            )
            raise HoldCorruptStateError(message)
        if latch is None:
            if authority_receipt_row is not None:
                raise HoldCorruptStateError(
                    "inactive Hold checkpoint retained authority evidence"
                )
            return None
        if authority_receipt_row is None:
            raise HoldCorruptStateError(
                "active hold latch references a missing authority receipt"
            )
        authority = _receipt_from_row(authority_receipt_row)
        if (
            authority.action is not HoldAction.HOLD
            or authority.disposition is not HoldDisposition.APPLIED
            or authority.receipt_id != latch.hold_receipt_id
            or authority.scope is not scope
            or authority.target_id != target_id
            or authority.reason != latch.reason
            or authority.actor_id != latch.actor_id
            or authority.occurred_at != latch.set_at
        ):
            raise HoldCorruptStateError(
                "active hold latch does not match its authority receipt"
            )
        return latch

    async def _validate_receipt_authority_graph(
        self,
        latch: Optional[HoldState],
        scope: HoldScope,
        target_id: str,
        *,
        receipt_rows: Optional[list[Any]] = None,
    ) -> None:
        """Prove the append-only authority graph agrees with its latch.

        Each applied Hold creates one authority. A later applied Hold or
        Release may consume it exactly once. Every chain must be acyclic and
        terminate either in an applied Release or in the one active authority
        named by the latch. This catches projection deletion, latch rewind,
        forked successors, and closed cycles rather than trusting a locally
        well-formed latch row.
        """

        rows = receipt_rows
        if rows is None:
            rows = await self._db.fetchall(
                f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts "
                "WHERE scope = ? AND target_id = ?",
                (scope.value, target_id),
            )
        receipts = [_receipt_from_row(row) for row in rows]
        receipt_ids = {receipt.receipt_id for receipt in receipts}
        if len(receipt_ids) != len(receipts):
            raise HoldCorruptStateError("Hold receipt graph has duplicate identities")
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
        for receipt in receipts:
            if (
                receipt.disposition is not HoldDisposition.APPLIED
                and receipt.prior_hold_receipt_id
                and receipt.prior_hold_receipt_id not in authorities
            ):
                raise HoldCorruptStateError(
                    "non-applied Hold receipt references missing authority"
                )
            if (
                receipt.action is HoldAction.HOLD
                and receipt.disposition is HoldDisposition.ALREADY_IN_STATE
                and receipt.prior_hold_receipt_id
            ):
                authority = authorities[receipt.prior_hold_receipt_id]
                if (
                    receipt.actor_id != authority.actor_id
                    or receipt.reason != authority.reason
                ):
                    raise HoldCorruptStateError(
                        "same-state Hold receipt does not match its authority"
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
            if not terminal_authorities:
                return
            raise HoldCorruptStateError(
                "unheld projection retains active Hold authority"
            )

        receipt = authorities.get(latch.hold_receipt_id)
        if receipt is None:
            referenced_row = next(
                (
                    row
                    for row in rows
                    if str(row[0]) == latch.hold_receipt_id
                ),
                None,
            )
            if referenced_row is None and receipt_rows is None:
                referenced_row = await self._read_receipt_by_id(
                    latch.hold_receipt_id
                )
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

    async def _validate_latch_projection(
        self,
        latch: Optional[HoldState],
        scope: HoldScope,
        target_id: str,
        *,
        receipt_rows: Optional[list[Any]] = None,
    ) -> None:
        await self._validate_receipt_authority_graph(
            latch,
            scope,
            target_id,
            receipt_rows=receipt_rows,
        )

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
        return _receipt_from_row(await self._read_receipt_by_operation(operation_id))

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
            replay_row = await self._read_receipt_by_operation(operation)
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
                await self._mark_checkpoint_clean(
                    resolved_scope,
                    resolved_target,
                    latch=prior,
                    latch_revision=prior.revision,
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
            updated = await self._db.execute(
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
            if updated != 1:
                raise HoldCorruptStateError(
                    "applied Hold did not update exactly one latch"
                )
            current = _latch_from_row(
                await self._read_latch_row(resolved_scope, resolved_target)
            )
            if current is None:
                raise HoldCorruptStateError(
                    "applied Hold did not produce an active latch"
                )
            await self._mark_checkpoint_clean(
                resolved_scope,
                resolved_target,
                latch=current,
                latch_revision=current.revision,
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
            replay_row = await self._read_receipt_by_operation(operation)
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
                updated = await self._db.execute(
                    "UPDATE hold_latches SET active = 0, hold_receipt_id = '', "
                    "reason = '', actor_id = '', set_at = '', revision = revision + 1 "
                    "WHERE scope = ? AND target_id = ? AND active = 1 "
                    "AND hold_receipt_id = ?",
                    (resolved_scope.value, resolved_target, expected),
                )
                if updated != 1:
                    raise HoldCorruptStateError(
                        "applied Hold release did not clear exactly one latch"
                    )
            current_row = await self._read_latch_row(
                resolved_scope, resolved_target
            )
            current = _latch_from_row(current_row)
            if current_row is None:
                raise HoldCorruptStateError(
                    "Hold release lost its durable latch projection"
                )
            await self._mark_checkpoint_clean(
                resolved_scope,
                resolved_target,
                latch=current,
                latch_revision=int(current_row[7]),
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
        async with self._db.transaction(immediate=True):
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
        """Read the bounded host + agent admission projection."""

        agent = _required_text(agent_id, "agent_id")
        if getattr(self._db, "backend_type", "") == "sqlite":
            return await self._get_effective_sqlite_snapshot(agent)
        targets = (
            (HoldScope.HOST, HOST_HOLD_TARGET),
            (HoldScope.AGENT, agent),
        )
        async with self._db.transaction(immediate=True):
            await self._lock_read_targets(targets)
            rows = await self._db.fetchall(
                self._effective_admission_query(),
                self._effective_admission_params(agent),
            )
            return await self._effective_from_admission_rows(rows, agent)

    @staticmethod
    def _effective_admission_query() -> str:
        """Return the two-row, constant-size admission snapshot query."""

        return (
            "WITH requested(scope, target_id) AS ("
            "SELECT ? AS scope, ? AS target_id UNION ALL SELECT ?, ?"
            ") SELECT requested.scope, requested.target_id, "
            "latch.active, latch.hold_receipt_id, latch.reason, "
            "latch.actor_id, latch.set_at, latch.revision, "
            "checkpoint.active, checkpoint.hold_receipt_id, "
            "checkpoint.latch_revision, checkpoint.dirty, "
            "authority.receipt_id, authority.operation_id, authority.action, "
            "authority.disposition, authority.scope, authority.target_id, "
            "authority.reason, authority.actor_id, authority.occurred_at, "
            "authority.expected_hold_receipt_id, "
            "authority.prior_hold_receipt_id, "
            "authority.resulting_hold_receipt_id, "
            "EXISTS(SELECT 1 FROM hold_receipts candidate "
            "WHERE candidate.scope = requested.scope "
            "AND candidate.target_id = requested.target_id), "
            "(SELECT target_id FROM hold_latches "
            "WHERE scope = 'host' AND target_id <> 'host' LIMIT 1), "
            "(SELECT target_id FROM hold_receipts "
            "WHERE scope = 'host' AND target_id <> 'host' LIMIT 1) "
            "FROM requested "
            "LEFT JOIN hold_latches latch ON latch.scope = requested.scope "
            "AND latch.target_id = requested.target_id "
            "LEFT JOIN hold_authority_checkpoints checkpoint "
            "ON checkpoint.scope = requested.scope "
            "AND checkpoint.target_id = requested.target_id "
            "LEFT JOIN hold_receipts authority "
            "ON authority.receipt_id = checkpoint.hold_receipt_id"
        )

    @staticmethod
    def _effective_admission_params(agent: str) -> tuple[str, str, str, str]:
        return (
            HoldScope.HOST.value,
            HOST_HOLD_TARGET,
            HoldScope.AGENT.value,
            agent,
        )

    async def _effective_from_admission_rows(
        self,
        rows: list[Any],
        agent: str,
    ) -> EffectiveHoldState:
        if len(rows) != 2:
            raise HoldCorruptStateError(
                "effective Hold admission snapshot has an unexpected shape"
            )
        states: dict[tuple[str, str], Optional[HoldState]] = {}
        for row in rows:
            if row is None or len(row) != 27:
                raise HoldCorruptStateError(
                    "effective Hold admission snapshot has an unexpected shape"
                )
            if row[25] is not None or row[26] is not None:
                raise HoldCorruptStateError(
                    "host hold state has a foreign target identity"
                )
            key = (str(row[0]), str(row[1]))
            if key in states:
                raise HoldCorruptStateError("duplicate hold admission target")
            try:
                scope = HoldScope(key[0])
            except ValueError as exc:
                raise HoldCorruptStateError(
                    "effective Hold admission snapshot has an invalid scope"
                ) from exc
            expected_target = HOST_HOLD_TARGET if scope is HoldScope.HOST else agent
            if key[1] != expected_target:
                raise HoldCorruptStateError(
                    "effective hold query returned a foreign target"
                )
            latch_row = (
                (row[0], row[1], *row[2:8]) if row[2] is not None else None
            )
            checkpoint_row = (
                (row[0], row[1], *row[8:12]) if row[8] is not None else None
            )
            authority_row = tuple(row[12:24]) if row[12] is not None else None
            states[key] = await self._validate_admission_projection(
                latch_row=latch_row,
                checkpoint_row=checkpoint_row,
                authority_receipt_row=authority_row,
                scope=scope,
                target_id=key[1],
                receipt_exists=bool(row[24]),
            )
        host_key = (HoldScope.HOST.value, HOST_HOLD_TARGET)
        agent_key = (HoldScope.AGENT.value, agent)
        if set(states) != {host_key, agent_key}:
            raise HoldCorruptStateError(
                "effective Hold admission snapshot omitted a requested target"
            )
        return EffectiveHoldState(
            host=states[host_key],
            agent=states[agent_key],
        )

    async def _get_effective_sqlite_snapshot(
        self, agent: str
    ) -> EffectiveHoldState:
        """Validate two fixed-size projections from one committed statement."""

        rows = await self._db.fetchall_snapshot(
            self._effective_admission_query(),
            self._effective_admission_params(agent),
        )
        return await self._effective_from_admission_rows(rows, agent)

    async def get_receipt(self, operation_id: str) -> Optional[HoldReceipt]:
        operation = _required_text(operation_id, "operation_id")
        row = await self._read_receipt_by_operation(operation)
        return _receipt_from_row(row) if row is not None else None


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
