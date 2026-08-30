"""Durable store helpers for ``restart_requests`` (#1512).

The table lives in the same SQLite database the rest of the feature
suite uses (resolved via :func:`resolve_feature_database`). Each row
captures one agent-initiated restart request through its full life:
``pending`` → ``approved`` → ``executing`` → ``completed`` (terminal)
or ``rejected`` / ``canceled`` (terminal).

Schema is additive: when a feature loads against a pre-existing DB
without the table, ``ensure_restart_requests_table`` creates it. No
existing tables are modified.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from kestrel_sovereign.storage.database_clock import database_clock

from .authority import (
    RestartAuthorityError,
    issue_restart_authority,
    reseal_restart_safety_state,
    verify_restart_authority,
)


# Terminal states — a request in any of these is locked. The
# coordinator must not re-execute, the agent must not re-modify.
TERMINAL_STATES = frozenset({"completed", "rejected", "canceled"})

# In-flight states the coordinator considers when picking the next
# request to execute. Two further in-flight states exist but are NOT
# candidates and NOT cancelable: ``updating`` (an update_then_restart
# row whose allowlisted update profile is mid-run — reset to pending on
# boot if interrupted) and ``executing`` (already past the safety gate;
# the coordinator only moves it to ``completed`` or leaves it for the
# post-restart sweep to mark).
PENDING_STATES = frozenset({"pending", "approved"})

# Every status a row can carry through its life. ``pending`` → ``approved``
# → (``updating`` for update_then_restart rows) → ``executing`` →
# ``completed``; or the terminal ``rejected`` / ``canceled``. Used to
# validate the ``list_restart_requests`` status filter so an unknown value
# is rejected with a clear message rather than silently returning 0 rows.
KNOWN_STATUSES = frozenset(
    {
        "pending",
        "approved",
        "updating",
        "executing",
        "completed",
        "rejected",
        "canceled",
    }
)

# All policies the coordinator understands. Anything else is rejected
# at request time so the table never carries an unknown value.
KNOWN_POLICIES = frozenset(
    {"idle_agents_only", "allow_busy_after_timeout", "manual_only"}
)

# All urgencies the coordinator understands.
KNOWN_URGENCIES = frozenset({"low", "normal", "high", "critical"})

# Operation modes. ``restart_only`` (the historical behaviour, default)
# spawns ``kestrel restart``. ``update_then_restart`` first runs an
# allowlisted update/install profile against a local checkout, then
# restarts — an explicit, audited step, never an implicit side effect of
# a plain restart.
KNOWN_OPERATIONS = frozenset({"restart_only", "update_then_restart"})

# Stamped onto ``wake_dispatch_boot_id`` for rows already delivered when the
# dispatch-observability columns were added (#2774). Reads as "delivered under
# the old flow; whether a wake was actually dispatched is unrecoverable" — it
# does NOT assert a dispatch happened, because some of those rows provably had
# none: a host with no usable dispatcher marks a row delivered without sending
# anything (see ``_deliver_restart_completed``). Its job is to keep '' meaning
# "no wake was ever dispatched for this row", which is the negative evidence
# #2774 needs.
PRE_MIGRATION_BOOT_ID = "pre-migration"

# Columns added after the original #1512 schema. Applied additively via
# ALTER TABLE so a feature loading against a pre-existing table picks
# them up without losing data.
#
# ORDER IS LOAD-BEARING: ``_COLUMN_BACKFILLS`` runs in this order, and the
# wake_dispatch_boot_id backfill reads what the wake_delivered backfill
# writes. Asserted at import time below.
_ADDED_COLUMNS = (
    ("operation", "TEXT DEFAULT 'restart_only'"),
    ("update_repo_path", "TEXT DEFAULT ''"),
    ("update_target_ref", "TEXT DEFAULT ''"),
    ("update_profile", "TEXT DEFAULT ''"),
    ("update_allow_migrations", "INTEGER DEFAULT 0"),
    ("update_log", "TEXT DEFAULT ''"),
    ("requester_request_id", "TEXT DEFAULT ''"),
    ("executing_boot_id", "TEXT DEFAULT ''"),
    # Session the request was filed from, so the post-restart wake can be
    # dispatched back into the SAME chat window the agent asked from (#1809).
    ("origin_session_id", "TEXT DEFAULT ''"),
    # Whether the post-restart ``restart.completed`` wake has actually been
    # DELIVERED (#1819). Decouples "restart finished" (status=completed, set as
    # soon as a prior-boot row is swept) from "agent has been notified" (this
    # flag, set once the COGNITION wake lands Status.OK). The sweep retries the
    # wake while this is 0; it never re-terminalizes an already-completed row.
    ("wake_delivered", "INTEGER DEFAULT 0"),
    # When the post-restart wake was DISPATCHED, as distinct from when its
    # turn completed (#2774). ``wake_delivered`` only flips once the woken
    # cognition turn returns Status.OK, which is necessarily AFTER that turn
    # ends — so the turn the wake itself woke can never observe the flag as
    # true, and the row appears to contradict its own consumer. These columns
    # are stamped just before the signal is handed to the dispatcher, so "did
    # this boot try to wake this row, and when" is answerable during that turn
    # and usable as negative evidence when a wake genuinely never fired.
    ("wake_dispatched_at", "TEXT DEFAULT ''"),
    ("wake_dispatch_boot_id", "TEXT DEFAULT ''"),
    # How many completion wakes have been dispatched for this row. The #2738
    # failure was ~18 re-emissions inside one boot; a count makes that storm
    # visible in the row itself rather than only in the logs.
    ("wake_dispatch_count", "INTEGER DEFAULT 0"),
    # Start of the current uninterrupted busy deferral. This must not be
    # inferred from requested_at: old pending rows may predate this policy by
    # weeks and cannot become immediately escalation-eligible on upgrade.
    ("first_blocked_at", "TEXT DEFAULT ''"),
    # Legacy rows receive 0 from the additive migration. Fresh requests are
    # inserted with 1 explicitly, so only a pre-upgrade backlog needs the
    # one-time acknowledgement before bounded escalation may override idle.
    ("escalation_acknowledged", "INTEGER DEFAULT 0"),
    # Exact sovereign-key authorization for the host mutation. Legacy rows
    # stay blank and are rejected by the executor; authority is never inferred
    # from their requester, age, status, or prior approval.
    ("authority_evidence", "TEXT DEFAULT ''"),
    ("authority_signature", "TEXT DEFAULT ''"),
)

# One-time data backfills for legacy rows, keyed by the column whose addition
# makes them necessary. Each runs in the same transaction as its ``ALTER``
# (see ``AsyncDatabase.migrate_columns_once``), so the schema itself is the
# marker: if the column is present, this backfill has already run or was never
# needed.
_COLUMN_BACKFILLS = {
    "wake_delivered": (
        # Pre-#1819 a row only reached 'completed' AFTER its wake was
        # delivered — terminalization was delivery-gated.
        # ``list_requests_needing_wake`` selects ``completed AND
        # wake_delivered = 0``, so without this every historical completed
        # restart is re-woken on the first post-upgrade sweep.
        "UPDATE restart_requests SET wake_delivered = 1 "
        "WHERE status = 'completed'",
        (),
    ),
    "wake_dispatch_boot_id": (
        # A row delivered before these columns existed carries no dispatch
        # record. Stamp the sentinel rather than a fabricated timestamp.
        #
        # Keyed on wake_delivered, NOT on status: a row that is completed but
        # undelivered has not had a wake land, so it must keep '' and stay
        # eligible for the sweep's retry.
        "UPDATE restart_requests SET wake_dispatch_boot_id = ? "
        "WHERE wake_delivered = 1 AND wake_dispatch_boot_id = ''",
        (PRE_MIGRATION_BOOT_ID,),
    ),
}

# The sentinel backfill reads what the delivered backfill writes, and both are
# driven off ``_ADDED_COLUMNS`` order. Not an ``assert``: this is load-bearing
# and asserts are stripped under ``python -O``, which is exactly when a silent
# miscompile would hurt.
_backfill_order = [c for c, _ in _ADDED_COLUMNS if c in _COLUMN_BACKFILLS]
if _backfill_order.index("wake_delivered") > _backfill_order.index(
    "wake_dispatch_boot_id"
):
    raise RuntimeError(
        "wake_delivered must be backfilled before the wake_dispatch_boot_id "
        "sentinel that reads it"
    )
del _backfill_order

# Canonical column order shared by every SELECT below and ``from_row``.
_COLUMNS = (
    "id, requested_by_agent, reason, requested_at, desired_window, "
    "urgency, policy, status, status_reason, completed_at, operation, "
    "update_repo_path, update_target_ref, update_profile, "
    "update_allow_migrations, update_log, requester_request_id, "
    "executing_boot_id, origin_session_id, wake_delivered, "
    "wake_dispatched_at, wake_dispatch_boot_id, wake_dispatch_count, "
    "first_blocked_at, escalation_acknowledged, authority_evidence, "
    "authority_signature"
)


@dataclass
class RestartRequest:
    id: str
    requested_by_agent: str
    reason: str
    requested_at: str
    desired_window: str
    urgency: str
    policy: str
    status: str
    status_reason: str
    completed_at: Optional[str]
    operation: str = "restart_only"
    update_repo_path: str = ""
    update_target_ref: str = ""
    update_profile: str = ""
    update_allow_migrations: bool = False
    update_log: str = ""
    requester_request_id: str = ""
    # Identifier of the host process that crossed this row into
    # ``executing`` (#1796). The post-restart sweep only wakes a row whose
    # stamp differs from the CURRENT process's id — i.e. a row left
    # ``executing`` by a PRIOR process, which provably means the restart
    # already happened. A row stamped by the live process (restart still
    # in flight, or a detached restart that failed to kill the parent)
    # must NOT be falsely terminalized as ``completed``.
    executing_boot_id: str = ""
    # Session id the request was filed from (#1809). Empty for
    # system/CLI-filed requests with no chat session; when set, the
    # restart.completed wake is dispatched into this session so it surfaces
    # in the same chat window the request came from.
    origin_session_id: str = ""
    # Whether the post-restart wake has been delivered (#1819). A row can be
    # ``completed`` (restart finished) with ``wake_delivered=False`` while the
    # wake is still being retried.
    wake_delivered: bool = False
    # When the wake was dispatched and from which boot (#2774). Set on
    # dispatch acceptance, so it is already visible to the turn the wake
    # woke — unlike ``wake_delivered``, which cannot be by construction.
    wake_dispatched_at: str = ""
    wake_dispatch_boot_id: str = ""
    # Number of completion wakes dispatched for this row (#2738).
    wake_dispatch_count: int = 0
    # Start of the current continuous idle-policy deferral (#2900).
    first_blocked_at: str = ""
    # Fresh rows opt into bounded escalation. Migrated rows remain false until
    # an explicit acknowledgement records acceptance of the new behavior.
    escalation_acknowledged: bool = False
    # Host-sealed exact request bounds. The signature is intentionally omitted
    # from ``to_public_dict`` so an agent cannot harvest/replay authority.
    authority_evidence: str = ""
    authority_signature: str = ""

    @classmethod
    def from_row(cls, row: Iterable[Any]) -> "RestartRequest":
        cols = list(row)

        def g(i: int, default: Any = None) -> Any:
            return cols[i] if i < len(cols) else default

        return cls(
            id=str(g(0)),
            requested_by_agent=str(g(1) or ""),
            reason=str(g(2) or ""),
            requested_at=str(g(3) or ""),
            desired_window=str(g(4) or ""),
            urgency=str(g(5) or "normal"),
            policy=str(g(6) or "idle_agents_only"),
            status=str(g(7) or "pending"),
            status_reason=str(g(8) or ""),
            completed_at=(str(g(9)) if g(9) is not None else None),
            operation=str(g(10) or "restart_only"),
            update_repo_path=str(g(11) or ""),
            update_target_ref=str(g(12) or ""),
            update_profile=str(g(13) or ""),
            update_allow_migrations=bool(int(g(14) or 0)),
            update_log=str(g(15) or ""),
            requester_request_id=str(g(16) or ""),
            executing_boot_id=str(g(17) or ""),
            origin_session_id=str(g(18) or ""),
            wake_delivered=bool(int(g(19) or 0)),
            wake_dispatched_at=str(g(20) or ""),
            wake_dispatch_boot_id=str(g(21) or ""),
            wake_dispatch_count=int(g(22) or 0),
            first_blocked_at=str(g(23) or ""),
            escalation_acknowledged=bool(int(g(24) or 0)),
            authority_evidence=str(g(25) or ""),
            authority_signature=str(g(26) or ""),
        )

    def update_log_dict(self) -> Dict[str, Any]:
        """Parse ``update_log`` JSON into a dict (``{}`` if empty/invalid)."""
        if not self.update_log:
            return {}
        try:
            data = json.loads(self.update_log)
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "requested_by_agent": self.requested_by_agent,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "desired_window": self.desired_window,
            "urgency": self.urgency,
            "policy": self.policy,
            "status": self.status,
            "status_reason": self.status_reason,
            "completed_at": self.completed_at,
            "operation": self.operation,
            "update_repo_path": self.update_repo_path,
            "update_target_ref": self.update_target_ref,
            "update_profile": self.update_profile,
            "update_allow_migrations": self.update_allow_migrations,
            "update": self.update_log_dict(),
            "requester_request_id": self.requester_request_id,
            "executing_boot_id": self.executing_boot_id,
            "origin_session_id": self.origin_session_id,
            "wake_delivered": self.wake_delivered,
            # Exposed so introspection can distinguish "no wake was ever sent"
            # from "a wake was sent and its turn has not finished yet" (#2774).
            # Reading only ``wake_delivered`` conflates the two.
            "wake_dispatched_at": self.wake_dispatched_at,
            "wake_dispatch_boot_id": self.wake_dispatch_boot_id,
            "wake_dispatch_count": self.wake_dispatch_count,
            "first_blocked_at": self.first_blocked_at,
            "escalation_acknowledged": self.escalation_acknowledged,
        }


async def ensure_restart_requests_table(db) -> None:
    """Create the table + indices if they don't already exist."""
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS restart_requests (
            id TEXT PRIMARY KEY,
            requested_by_agent TEXT NOT NULL,
            reason TEXT NOT NULL,
            requested_at TEXT NOT NULL,
            desired_window TEXT DEFAULT '',
            urgency TEXT DEFAULT 'normal',
            policy TEXT DEFAULT 'idle_agents_only',
            status TEXT DEFAULT 'pending',
            status_reason TEXT DEFAULT '',
            completed_at TEXT,
            operation TEXT DEFAULT 'restart_only',
            update_repo_path TEXT DEFAULT '',
            update_target_ref TEXT DEFAULT '',
            update_profile TEXT DEFAULT '',
            update_allow_migrations INTEGER DEFAULT 0,
            update_log TEXT DEFAULT '',
            requester_request_id TEXT DEFAULT '',
            executing_boot_id TEXT DEFAULT '',
            origin_session_id TEXT DEFAULT '',
            wake_delivered INTEGER DEFAULT 0,
            wake_dispatched_at TEXT DEFAULT '',
            wake_dispatch_boot_id TEXT DEFAULT '',
            wake_dispatch_count INTEGER DEFAULT 0,
            first_blocked_at TEXT DEFAULT '',
            escalation_acknowledged INTEGER DEFAULT 0,
            authority_evidence TEXT DEFAULT '',
            authority_signature TEXT DEFAULT ''
        )
        """
    )
    # The platform owns the mechanism (one transaction per ALTER + backfill,
    # schema-as-marker, post-migration verification); this declares only what.
    await db.migrate_columns_once(
        "restart_requests", _ADDED_COLUMNS, _COLUMN_BACKFILLS,
    )
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_restart_requests_status
        ON restart_requests(status)
        """
    )
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_restart_requests_agent
        ON restart_requests(requested_by_agent, status)
        """
    )


async def insert_request(
    db,
    *,
    requested_by_agent: str,
    reason: str,
    urgency: str = "normal",
    policy: str = "idle_agents_only",
    desired_window: str = "",
    operation: str = "restart_only",
    update_repo_path: str = "",
    update_target_ref: str = "",
    update_profile: str = "",
    update_allow_migrations: bool = False,
    requester_request_id: str = "",
    origin_session_id: str = "",
) -> RestartRequest:
    """Insert a fresh pending request. Returns the dataclass row."""
    req_id = uuid.uuid4().hex
    requested_at = (await database_clock(db)).isoformat()
    authority_evidence, authority_signature = issue_restart_authority(
        request_id=req_id,
        requested_by_agent=requested_by_agent,
        reason=reason,
        urgency=urgency,
        policy=policy,
        desired_window=desired_window,
        operation=operation,
        update_repo_path=update_repo_path,
        update_target_ref=update_target_ref,
        update_profile=update_profile,
        update_allow_migrations=update_allow_migrations,
        requester_request_id=requester_request_id,
        origin_session_id=origin_session_id,
        requested_at=requested_at,
    )
    await db.execute(
        """
        INSERT INTO restart_requests (
            id, requested_by_agent, reason, requested_at,
            desired_window, urgency, policy, status, status_reason,
            completed_at, operation, update_repo_path, update_target_ref,
            update_profile, update_allow_migrations, update_log,
            requester_request_id, origin_session_id, escalation_acknowledged,
            authority_evidence, authority_signature
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', '', NULL,
                ?, ?, ?, ?, ?, '', ?, ?, 1, ?, ?)
        """,
        (req_id, requested_by_agent, reason, requested_at, desired_window,
         urgency, policy, operation, update_repo_path, update_target_ref,
         update_profile, 1 if update_allow_migrations else 0,
         requester_request_id, origin_session_id, authority_evidence,
         authority_signature),
    )
    inserted = await get_request(db, req_id)
    if inserted is None:
        raise RuntimeError("restart request insert did not become visible")
    return inserted


async def list_requests(
    db, *, status: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> List[RestartRequest]:
    """Return all rows, optionally filtered."""
    where: List[str] = []
    params: List[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if agent_id:
        where.append("requested_by_agent = ?")
        params.append(agent_id)
    sql = f"SELECT {_COLUMNS} FROM restart_requests"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY requested_at DESC"
    rows = await db.fetchall(sql, tuple(params))
    return [RestartRequest.from_row(r) for r in rows]


async def list_requests_needing_wake(
    db, *, agent_id: Optional[str] = None,
) -> List[RestartRequest]:
    """Rows whose post-restart wake still has to be delivered (#1819).

    Two shapes qualify: a still-``executing`` row from a prior boot (the
    restart landed but the row hasn't been terminalized yet) and a
    ``completed`` row whose ``wake_delivered`` is still 0 (terminalized, but
    the wake hasn't landed Status.OK — a retry). The sweep terminalizes the
    former before dispatching, and retries the wake for both until delivered.
    """
    where = ["(status = 'executing' OR (status = 'completed' AND wake_delivered = 0))"]
    params: List[Any] = []
    if agent_id:
        where.append("requested_by_agent = ?")
        params.append(agent_id)
    sql = (
        f"SELECT {_COLUMNS} FROM restart_requests WHERE "
        + " AND ".join(where)
        + " ORDER BY requested_at DESC"
    )
    rows = await db.fetchall(sql, tuple(params))
    return [RestartRequest.from_row(r) for r in rows]


async def _write_landed(
    db,
    result: Any,
    request_id: str,
    verify,
    *,
    requested_by_agent: Optional[str] = None,
) -> bool:
    """Did an ``UPDATE`` actually change a row?

    The single contract for this module. Extracted from ``update_status``,
    which has always checked this correctly, so the two cannot drift: the
    project's SQLite/Postgres backends return the integer rowcount directly
    from ``execute``, and only a legacy cursor-style backend needs the
    re-read. ``verify`` confirms the intended change on that last path.
    """
    if isinstance(result, int):
        return result > 0
    rowcount = getattr(result, "rowcount", None)
    if isinstance(rowcount, int):
        return rowcount > 0
    row = (
        await get_request_for_agent(db, request_id, requested_by_agent)
        if requested_by_agent is not None
        else await get_request(db, request_id)
    )
    return row is not None and verify(row)


async def mark_wake_delivered(db, request_id: str) -> bool:
    """Flag a request's post-restart wake as delivered (#1819).

    Returns whether the write actually landed. It previously returned ``None``
    and discarded the rowcount ``execute`` hands back — the same value its
    sibling ``update_status`` treats as authoritative twenty lines away. So a
    write that matched no row was indistinguishable from one that succeeded,
    ``wake_delivered`` stayed 0 with nothing recorded anywhere, and the next
    sweep rediscovered the row and re-emitted a completion wake the agent had
    already consumed — once a minute, for eighteen minutes (#2738).
    """
    result = await db.execute(
        "UPDATE restart_requests SET wake_delivered = 1 WHERE id = ?",
        (request_id,),
    )
    return await _write_landed(
        db, result, request_id, lambda row: row.wake_delivered,
    )


async def mark_wake_dispatched(
    db, request_id: str, *, dispatched_at: str, boot_id: str,
) -> bool:
    """Record that a post-restart wake was DISPATCHED (#2774).

    Stamped immediately before the signal is handed to the dispatcher, which
    is the last moment guaranteed to precede the woken turn — the dispatcher
    starts the turn as soon as it has the signal, so any later write races the
    reader it exists for. ``wake_delivered`` structurally cannot answer this:
    it is only set once the woken turn returns ``Status.OK``, i.e. after the
    turn that would read it has ended.

    Every dispatch is recorded, not only the first: ``wake_dispatched_at``
    means what its name says (when we last dispatched) and
    ``wake_dispatch_count`` makes a re-emission storm countable from the row.
    The original failure was ~18 re-emissions within a single boot (#2738), and
    observability that kept only the first would have shown one timestamp for
    the whole event.
    """
    result = await db.execute(
        "UPDATE restart_requests SET wake_dispatched_at = ?, "
        "wake_dispatch_boot_id = ?, "
        "wake_dispatch_count = COALESCE(wake_dispatch_count, 0) + 1 "
        "WHERE id = ?",
        (dispatched_at, boot_id, request_id),
    )
    return await _write_landed(
        db, result, request_id,
        lambda row: (
            row.wake_dispatched_at == dispatched_at
            and row.wake_dispatch_boot_id == boot_id
        ),
    )


async def get_request(db, request_id: str) -> Optional[RestartRequest]:
    rows = await db.fetchall(
        f"SELECT {_COLUMNS} FROM restart_requests WHERE id = ?",
        (request_id,),
    )
    if not rows:
        return None
    return RestartRequest.from_row(rows[0])


async def get_request_for_agent(
    db,
    request_id: str,
    requested_by_agent: str,
) -> Optional[RestartRequest]:
    """Read one row only within its durable requesting-agent principal."""

    rows = await db.fetchall(
        f"SELECT {_COLUMNS} FROM restart_requests "
        "WHERE id = ? AND requested_by_agent = ?",
        (request_id, requested_by_agent),
    )
    if not rows:
        return None
    return RestartRequest.from_row(rows[0])


async def mark_deferral_started(
    db,
    request_id: str,
    *,
    expected_current_status: str,
    blocked_at: Optional[str] = None,
) -> Optional[RestartRequest]:
    """Persist the beginning of one uninterrupted busy deferral.

    The conditional write preserves the first timestamp when multiple host
    coordinators observe the same row. The status compare-and-set prevents a
    stale coordinator from annotating a canceled or executing row. Returning
    the authoritative row lets the caller distinguish an existing timestamp
    from a lost lifecycle race.
    """

    current = await get_request(db, request_id)
    if current is None:
        return None
    verified, _ = verify_restart_authority(current)
    if not verified:
        return None
    if current.first_blocked_at:
        return current
    stamped_at = blocked_at or (await database_clock(db)).isoformat()
    try:
        authority_evidence, authority_signature = reseal_restart_safety_state(
            current,
            first_blocked_at=stamped_at,
        )
    except RestartAuthorityError:
        return None
    await db.execute(
        "UPDATE restart_requests SET first_blocked_at = ?, "
        "authority_evidence = ?, authority_signature = ? "
        "WHERE id = ? AND status = ? "
        "AND (first_blocked_at IS NULL OR first_blocked_at = '') "
        "AND authority_signature = ?",
        (
            stamped_at,
            authority_evidence,
            authority_signature,
            request_id,
            expected_current_status,
            current.authority_signature,
        ),
    )
    return await get_request(db, request_id)


async def clear_deferral_started(
    db, request_id: str, *, expected_current_status: str,
) -> Optional[RestartRequest]:
    """Clear a busy interval and return the exact newly resealed row."""

    current = await get_request(db, request_id)
    if current is None:
        return None
    verified, _ = verify_restart_authority(current)
    if not verified:
        return None
    try:
        authority_evidence, authority_signature = reseal_restart_safety_state(
            current,
            first_blocked_at="",
        )
    except RestartAuthorityError:
        return None
    result = await db.execute(
        "UPDATE restart_requests SET first_blocked_at = '', "
        "authority_evidence = ?, authority_signature = ? "
        "WHERE id = ? AND status = ? AND first_blocked_at = ? "
        "AND authority_signature = ?",
        (
            authority_evidence,
            authority_signature,
            request_id,
            expected_current_status,
            current.first_blocked_at,
            current.authority_signature,
        ),
    )
    landed = await _write_landed(
        db, result, request_id, lambda row: row.first_blocked_at == "",
    )
    if not landed:
        return None
    refreshed = await get_request(db, request_id)
    if (
        refreshed is None
        or refreshed.status != expected_current_status
        or refreshed.first_blocked_at
        or refreshed.authority_signature != authority_signature
    ):
        return None
    return refreshed


async def acknowledge_escalation(
    db,
    request_id: str,
    *,
    requested_by_agent: str,
) -> bool:
    """Acknowledge a migrated row only for its durable requester."""

    result = await db.execute(
        "UPDATE restart_requests SET escalation_acknowledged = 1 "
        "WHERE id = ? AND requested_by_agent = ? "
        "AND status IN ('pending', 'approved')",
        (request_id, requested_by_agent),
    )
    return await _write_landed(
        db,
        result,
        request_id,
        lambda row: row.escalation_acknowledged,
        requested_by_agent=requested_by_agent,
    )


async def cancel_request_if_owned(
    db,
    request_id: str,
    *,
    requested_by_agent: str,
    status_reason: str,
    completed_at: str,
) -> bool:
    """Atomically cancel a pending row owned by ``requested_by_agent``."""

    result = await db.execute(
        "UPDATE restart_requests SET status = 'canceled', "
        "status_reason = ?, completed_at = ? "
        "WHERE id = ? AND requested_by_agent = ? "
        "AND status IN ('pending', 'approved')",
        (status_reason, completed_at, request_id, requested_by_agent),
    )
    return await _write_landed(
        db,
        result,
        request_id,
        lambda row: row.status == "canceled",
        requested_by_agent=requested_by_agent,
    )


async def record_update_log(db, request_id: str, update_log: str) -> None:
    """Persist the observed update steps/outcomes JSON onto the row.

    Kept separate from :func:`update_status` so the durable audit trail
    of what the update profile actually did survives independently of
    the request's lifecycle state.
    """
    await db.execute(
        "UPDATE restart_requests SET update_log = ? WHERE id = ?",
        (update_log, request_id),
    )


async def update_status(
    db,
    request_id: str,
    *,
    status: str,
    status_reason: str = "",
    completed_at: Optional[str] = None,
    expected_current_status: Optional[str] = None,
    expected_authority_signature: Optional[str] = None,
    executing_boot_id: Optional[str] = None,
) -> bool:
    """Atomic status transition. Returns True if a row was updated.

    ``expected_current_status`` protects against lifecycle races.
    ``expected_authority_signature`` additionally binds a transition to the
    exact signed safety-state version the caller evaluated, so a concurrent
    deferral-clock reseal cannot turn a stale safe observation into execution.

    When ``executing_boot_id`` is provided, it is stamped onto the row
    alongside the status (#1796) — the caller passes the current host
    process's id when crossing a row into ``executing`` so the
    post-restart sweep can tell a prior-process restart (real) from a
    live-process one (still in flight / failed).
    """
    sql = (
        "UPDATE restart_requests SET status = ?, status_reason = ?"
        + (", completed_at = ?" if completed_at is not None else "")
        + (", executing_boot_id = ?" if executing_boot_id is not None else "")
        + " WHERE id = ?"
    )
    params_final: List[Any] = [status, status_reason]
    if completed_at is not None:
        params_final.append(completed_at)
    if executing_boot_id is not None:
        params_final.append(executing_boot_id)
    params_final.append(request_id)
    if expected_current_status is not None:
        sql += " AND status = ?"
        params_final.append(expected_current_status)
    if expected_authority_signature is not None:
        sql += " AND authority_signature = ?"
        params_final.append(expected_authority_signature)
    result = await db.execute(sql, tuple(params_final))
    # >0 = updated; 0 = expected-status mismatch, i.e. lost the race.
    return await _write_landed(
        db, result, request_id, lambda row: row.status == status,
    )
