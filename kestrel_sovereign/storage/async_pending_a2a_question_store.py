"""Sender-side store for in-flight ``send_a2a_question`` correlation rows.

Owns CRUD on the ``pending_a2a_questions`` table defined in
``async_database.py``. Used by:

  - ``PeersFeature._post_a2a_task`` (insert on POST + spawn subscription;
    update to RESOLVED when terminal event arrives)
  - Startup-replay sweep (read WAITING rows on boot; for each, GET the
    recipient's task state; if terminal, fire signal locally)
  - Hourly expiry sweep (mark rows past deadline as EXPIRED; fire signal
    with ``state='expired'`` so the resumed prompt has a clean branch)

The table is sender-side state — not visible to the recipient. Auth is
implicit (it lives in the sender's per-agent DB, behind the same access
gate as the rest of agent storage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingA2AQuestion:
    """One in-flight ``send_a2a_question`` row.

    Fields mirror the schema. Times are ISO-8601 with UTC offset for
    backend-portability (SQLite stores TEXT, Postgres stores TIMESTAMP;
    callers always see ISO strings)."""

    task_id: str
    recipient: str
    original_question: str
    origin_turn_id: Optional[str]
    origin_session_id: Optional[str]
    deadline: str  # ISO-8601 UTC
    status: str  # WAITING | RESOLVED | EXPIRED
    created_at: str
    resolved_at: Optional[str]


class PendingA2AQuestionStore:
    """Async CRUD over ``pending_a2a_questions`` (#1444).

    Construct with the agent's ``AsyncDatabase`` AND the owning agent's
    id. Every query is filtered by ``agent_id`` so a shared backend
    (e.g. Postgres in a multi-agent deployment) cannot leak rows
    between agents — codex round 1 P1 on PR #1453. ``agent_id`` is
    the agent's DID; tests can pass any non-empty string.
    """

    def __init__(self, db: AsyncDatabase, agent_id: str):
        if not isinstance(agent_id, str):
            raise TypeError(
                f"PendingA2AQuestionStore agent_id must be a string, "
                f"got {type(agent_id).__name__}"
            )
        self._db = db
        # Empty string is a valid back-compat default for solo-agent
        # SQLite deployments where the rows are scoped to the file
        # rather than a column; the schema default is also ''.
        self._agent_id = agent_id

    async def insert(
        self,
        *,
        task_id: str,
        recipient: str,
        original_question: str,
        origin_turn_id: Optional[str],
        origin_session_id: Optional[str],
        deadline: datetime,
    ) -> None:
        """Record a freshly-POSTed question. ``deadline`` is the wall-clock
        UTC moment past which the hourly expiry sweep will mark this row
        EXPIRED and fire a synthetic ``a2a.question_answered`` signal.

        Idempotency: if a row for the same ``(agent_id, task_id)`` PK
        already exists, ``INSERT OR IGNORE`` skips silently. Callers
        should not need this in practice (task_id is a fresh UUID per
        POST) but the guard prevents the startup-replay sweep from
        double-inserting on a crash-restart-during-write boundary."""
        # Normalize to UTC then strip tzinfo: Postgres TIMESTAMP columns
        # require naive datetimes (asyncpg rejects strings; see the
        # ``_track_model_usage`` pattern in llm/usage_tracking.py). SQLite
        # also accepts datetime values via aiosqlite's adapter. Codex
        # round 6 P2 on PR #1453 — the prior ``.isoformat()`` worked on
        # SQLite by coincidence (TEXT column) and silently broke the
        # Postgres path after the task had already been POSTed.
        if deadline.tzinfo is not None:
            deadline = deadline.astimezone(timezone.utc).replace(tzinfo=None)
        await self._db.execute(
            """
            INSERT OR IGNORE INTO pending_a2a_questions
                (agent_id, task_id, recipient, original_question,
                 origin_turn_id, origin_session_id, deadline, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'WAITING')
            """,
            (
                self._agent_id,
                task_id,
                recipient,
                original_question,
                origin_turn_id,
                origin_session_id,
                deadline,
            ),
        )

    async def get(self, task_id: str) -> Optional[PendingA2AQuestion]:
        rows = await self._db.fetchall(
            """
            SELECT task_id, recipient, original_question,
                   origin_turn_id, origin_session_id, deadline,
                   status, created_at, resolved_at
            FROM pending_a2a_questions
            WHERE agent_id = ? AND task_id = ?
            """,
            (self._agent_id, task_id),
        )
        if not rows:
            return None
        r = rows[0]
        return PendingA2AQuestion(
            task_id=r[0],
            recipient=r[1],
            original_question=r[2],
            origin_turn_id=r[3],
            origin_session_id=r[4],
            deadline=str(r[5]),
            status=r[6],
            created_at=str(r[7]),
            resolved_at=str(r[8]) if r[8] else None,
        )

    async def mark_resolved(self, task_id: str) -> bool:
        """Transition WAITING → RESOLVED for THIS agent's row. Returns
        True if a WAITING row was found and updated, False if the row
        was already terminal (RESOLVED or EXPIRED) or belonged to a
        different agent — both cases are benign (subscription racing a
        startup-replay; another agent's row on a shared backend) and
        the caller should drop their resolve-side signal silently.

        Codex round 4 P2 on PR #1453: use ``execute()`` (which commits
        on SQLite) rather than ``fetchall(UPDATE ... RETURNING ...)``
        — the prior implementation surfaced the row to the current
        connection but never durably wrote it, so restarting the agent
        resurrected RESOLVED rows as WAITING and the startup-replay
        sweep re-fired the resumption signal as a duplicate. Rowcount
        > 0 carries the same conditional-update semantics as a
        ``RETURNING task_id`` row count."""
        rowcount = await self._db.execute(
            """
            UPDATE pending_a2a_questions
            SET status = 'RESOLVED', resolved_at = CURRENT_TIMESTAMP
            WHERE agent_id = ? AND task_id = ? AND status = 'WAITING'
            """,
            (self._agent_id, task_id),
        )
        return rowcount > 0

    async def list_waiting(self) -> List[PendingA2AQuestion]:
        """All rows still in WAITING for THIS agent — startup-replay's
        input set."""
        return await self._list_by_status("WAITING")

    async def list_waiting_past_deadline(
        self, now: Optional[datetime] = None,
    ) -> List[PendingA2AQuestion]:
        """WAITING rows whose deadline has passed for THIS agent —
        hourly-expiry input set. ``now`` defaults to current UTC; pass
        an explicit value in tests so the sweep is deterministic
        without monkey-patching ``datetime.utcnow``."""
        # See ``insert`` for why we strip tzinfo — Postgres TIMESTAMP
        # columns reject strings AND tz-aware datetimes. Codex round 6
        # P2.
        ts_dt = (now or datetime.now(timezone.utc))
        if ts_dt.tzinfo is not None:
            ts_dt = ts_dt.astimezone(timezone.utc).replace(tzinfo=None)
        rows = await self._db.fetchall(
            """
            SELECT task_id, recipient, original_question,
                   origin_turn_id, origin_session_id, deadline,
                   status, created_at, resolved_at
            FROM pending_a2a_questions
            WHERE agent_id = ? AND status = 'WAITING' AND deadline < ?
            """,
            (self._agent_id, ts_dt),
        )
        return [self._row_to_dc(r) for r in rows]

    async def mark_expired(self, task_id: str) -> bool:
        """Transition WAITING → EXPIRED (terminal) for THIS agent's
        row. Same idempotency + cross-agent semantics as
        ``mark_resolved``. ``execute()`` is used for the same
        durability reason — see ``mark_resolved`` docstring + codex
        round 4 P2."""
        rowcount = await self._db.execute(
            """
            UPDATE pending_a2a_questions
            SET status = 'EXPIRED', resolved_at = CURRENT_TIMESTAMP
            WHERE agent_id = ? AND task_id = ? AND status = 'WAITING'
            """,
            (self._agent_id, task_id),
        )
        return rowcount > 0

    async def mark_waiting_for_retry(self, task_id: str) -> bool:
        """Restore a terminal row to WAITING after a dispatch failure.

        Callers that mark a question RESOLVED/EXPIRED before enqueueing the
        resumption signal must use this when enqueue fails. Otherwise the
        terminal row disappears from startup replay/hourly sweep and the
        asker is never woken.
        """
        rowcount = await self._db.execute(
            """
            UPDATE pending_a2a_questions
            SET status = 'WAITING', resolved_at = NULL
            WHERE agent_id = ? AND task_id = ? AND status IN ('RESOLVED', 'EXPIRED')
            """,
            (self._agent_id, task_id),
        )
        return rowcount > 0

    async def _list_by_status(self, status: str) -> List[PendingA2AQuestion]:
        rows = await self._db.fetchall(
            """
            SELECT task_id, recipient, original_question,
                   origin_turn_id, origin_session_id, deadline,
                   status, created_at, resolved_at
            FROM pending_a2a_questions
            WHERE agent_id = ? AND status = ?
            """,
            (self._agent_id, status),
        )
        return [self._row_to_dc(r) for r in rows]

    @staticmethod
    def _row_to_dc(r) -> PendingA2AQuestion:
        return PendingA2AQuestion(
            task_id=r[0],
            recipient=r[1],
            original_question=r[2],
            origin_turn_id=r[3],
            origin_session_id=r[4],
            deadline=str(r[5]),
            status=r[6],
            created_at=str(r[7]),
            resolved_at=str(r[8]) if r[8] else None,
        )
