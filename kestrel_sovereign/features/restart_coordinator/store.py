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
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


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

# Columns added after the original #1512 schema. Applied additively via
# ALTER TABLE so a feature loading against a pre-existing table picks
# them up without losing data.
_ADDED_COLUMNS = (
    ("operation", "TEXT DEFAULT 'restart_only'"),
    ("update_repo_path", "TEXT DEFAULT ''"),
    ("update_target_ref", "TEXT DEFAULT ''"),
    ("update_profile", "TEXT DEFAULT ''"),
    ("update_allow_migrations", "INTEGER DEFAULT 0"),
    ("update_log", "TEXT DEFAULT ''"),
    ("requester_request_id", "TEXT DEFAULT ''"),
)

# Canonical column order shared by every SELECT below and ``from_row``.
_COLUMNS = (
    "id, requested_by_agent, reason, requested_at, desired_window, "
    "urgency, policy, status, status_reason, completed_at, operation, "
    "update_repo_path, update_target_ref, update_profile, "
    "update_allow_migrations, update_log, requester_request_id"
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
            requester_request_id TEXT DEFAULT ''
        )
        """
    )
    # Additively backfill the #1539 columns on a pre-existing table.
    for col, col_def in _ADDED_COLUMNS:
        try:
            await db.execute(
                f"ALTER TABLE restart_requests ADD COLUMN {col} {col_def}"
            )
        except Exception:
            # Column already exists — expected on every non-first run.
            pass
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
) -> RestartRequest:
    """Insert a fresh pending request. Returns the dataclass row."""
    req_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        """
        INSERT INTO restart_requests (
            id, requested_by_agent, reason, requested_at,
            desired_window, urgency, policy, status, status_reason,
            completed_at, operation, update_repo_path, update_target_ref,
            update_profile, update_allow_migrations, update_log,
            requester_request_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', '', NULL, ?, ?, ?, ?, ?, '', ?)
        """,
        (req_id, requested_by_agent, reason, now, desired_window,
         urgency, policy, operation, update_repo_path, update_target_ref,
         update_profile, 1 if update_allow_migrations else 0,
         requester_request_id),
    )
    return RestartRequest(
        id=req_id,
        requested_by_agent=requested_by_agent,
        reason=reason,
        requested_at=now,
        desired_window=desired_window,
        urgency=urgency,
        policy=policy,
        status="pending",
        status_reason="",
        completed_at=None,
        operation=operation,
        update_repo_path=update_repo_path,
        update_target_ref=update_target_ref,
        update_profile=update_profile,
        update_allow_migrations=update_allow_migrations,
        update_log="",
        requester_request_id=requester_request_id,
    )


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


async def get_request(db, request_id: str) -> Optional[RestartRequest]:
    rows = await db.fetchall(
        f"SELECT {_COLUMNS} FROM restart_requests WHERE id = ?",
        (request_id,),
    )
    if not rows:
        return None
    return RestartRequest.from_row(rows[0])


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
) -> bool:
    """Atomic status transition. Returns True if a row was updated.

    When ``expected_current_status`` is provided, the update is gated
    on the row currently having that status — protects against
    racing coordinators (or a concurrent cancel) overwriting an
    in-flight ``executing`` row.
    """
    sql_parts = ["UPDATE restart_requests SET status = ?", "status_reason = ?"]
    params: List[Any] = [status, status_reason]
    if completed_at is not None:
        sql_parts.append("completed_at = ?")
        params.append(completed_at)
    sql = (
        "UPDATE restart_requests SET status = ?, status_reason = ?"
        + (", completed_at = ?" if completed_at is not None else "")
        + " WHERE id = ?"
    )
    params_final: List[Any] = [status, status_reason]
    if completed_at is not None:
        params_final.append(completed_at)
    params_final.append(request_id)
    if expected_current_status is not None:
        sql += " AND status = ?"
        params_final.append(expected_current_status)
    result = await db.execute(sql, tuple(params_final))
    # The project SQLite/Postgres backends return the integer
    # rowcount directly from ``execute``. Treat the int form as
    # authoritative (>0 = updated, 0 = expected-status mismatch
    # i.e. lost the race). Only fall back to SELECT for legacy
    # cursor-style backends.
    if isinstance(result, int):
        return result > 0
    rowcount = getattr(result, "rowcount", None)
    if isinstance(rowcount, int):
        return rowcount > 0
    row = await get_request(db, request_id)
    return row is not None and row.status == status
