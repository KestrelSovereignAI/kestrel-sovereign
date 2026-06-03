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

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)


# Terminal states — a request in any of these is locked. The
# coordinator must not re-execute, the agent must not re-modify.
TERMINAL_STATES = frozenset({"completed", "rejected", "canceled"})

# In-flight states the coordinator considers when picking the next
# request to execute. ``executing`` is in-flight but already past the
# safety gate — the coordinator only moves it to ``completed`` (or
# leaves it for the post-restart sweep to mark).
PENDING_STATES = frozenset({"pending", "approved"})

# All policies the coordinator understands. Anything else is rejected
# at request time so the table never carries an unknown value.
KNOWN_POLICIES = frozenset(
    {"idle_agents_only", "allow_busy_after_timeout", "manual_only"}
)

# All urgencies the coordinator understands.
KNOWN_URGENCIES = frozenset({"low", "normal", "high", "critical"})


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

    @classmethod
    def from_row(cls, row: Iterable[Any]) -> "RestartRequest":
        cols = list(row)
        return cls(
            id=str(cols[0]),
            requested_by_agent=str(cols[1] or ""),
            reason=str(cols[2] or ""),
            requested_at=str(cols[3] or ""),
            desired_window=str(cols[4] or ""),
            urgency=str(cols[5] or "normal"),
            policy=str(cols[6] or "idle_agents_only"),
            status=str(cols[7] or "pending"),
            status_reason=str(cols[8] or ""),
            completed_at=(str(cols[9]) if cols[9] is not None else None),
        )

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
            completed_at TEXT
        )
        """
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
) -> RestartRequest:
    """Insert a fresh pending request. Returns the dataclass row."""
    req_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        """
        INSERT INTO restart_requests (
            id, requested_by_agent, reason, requested_at,
            desired_window, urgency, policy, status, status_reason,
            completed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', '', NULL)
        """,
        (req_id, requested_by_agent, reason, now, desired_window,
         urgency, policy),
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
    sql = (
        "SELECT id, requested_by_agent, reason, requested_at, "
        "desired_window, urgency, policy, status, status_reason, "
        "completed_at FROM restart_requests"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY requested_at DESC"
    rows = await db.fetchall(sql, tuple(params))
    return [RestartRequest.from_row(r) for r in rows]


async def get_request(db, request_id: str) -> Optional[RestartRequest]:
    rows = await db.fetchall(
        "SELECT id, requested_by_agent, reason, requested_at, "
        "desired_window, urgency, policy, status, status_reason, "
        "completed_at FROM restart_requests WHERE id = ?",
        (request_id,),
    )
    if not rows:
        return None
    return RestartRequest.from_row(rows[0])


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
