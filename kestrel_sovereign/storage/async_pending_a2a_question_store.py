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

    Construct with the agent's ``AsyncDatabase`` so the writes go to the
    same backend as the rest of agent state (and benefit from the same
    transaction + retry posture).
    """

    def __init__(self, db: AsyncDatabase):
        self._db = db

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

        Idempotency: if a row for the same ``task_id`` already exists,
        ``INSERT OR IGNORE`` skips silently. Callers should not need this
        in practice (task_id is a fresh UUID per POST) but the guard
        prevents the startup-replay sweep from double-inserting on a
        crash-restart-during-write boundary."""
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        await self._db.execute(
            """
            INSERT OR IGNORE INTO pending_a2a_questions
                (task_id, recipient, original_question,
                 origin_turn_id, origin_session_id, deadline, status)
            VALUES (?, ?, ?, ?, ?, ?, 'WAITING')
            """,
            (
                task_id,
                recipient,
                original_question,
                origin_turn_id,
                origin_session_id,
                deadline.isoformat(),
            ),
        )

    async def get(self, task_id: str) -> Optional[PendingA2AQuestion]:
        rows = await self._db.fetchall(
            """
            SELECT task_id, recipient, original_question,
                   origin_turn_id, origin_session_id, deadline,
                   status, created_at, resolved_at
            FROM pending_a2a_questions
            WHERE task_id = ?
            """,
            (task_id,),
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
        """Transition WAITING → RESOLVED. Returns True if a WAITING row
        was found and updated, False if the row was already terminal
        (RESOLVED or EXPIRED) — that case is benign (subscription racing
        a startup-replay both seeing the same terminal event) and the
        caller should drop their resolve-side signal silently."""
        rows = await self._db.fetchall(
            """
            UPDATE pending_a2a_questions
            SET status = 'RESOLVED', resolved_at = CURRENT_TIMESTAMP
            WHERE task_id = ? AND status = 'WAITING'
            RETURNING task_id
            """,
            (task_id,),
        )
        return bool(rows)

    async def list_waiting(self) -> List[PendingA2AQuestion]:
        """All rows still in WAITING — startup-replay's input set."""
        return await self._list_by_status("WAITING")

    async def list_waiting_past_deadline(
        self, now: Optional[datetime] = None,
    ) -> List[PendingA2AQuestion]:
        """WAITING rows whose deadline has passed — hourly-expiry input
        set. ``now`` defaults to current UTC; pass an explicit value in
        tests so the sweep is deterministic without monkey-patching
        ``datetime.utcnow``."""
        ts = (now or datetime.now(timezone.utc)).isoformat()
        rows = await self._db.fetchall(
            """
            SELECT task_id, recipient, original_question,
                   origin_turn_id, origin_session_id, deadline,
                   status, created_at, resolved_at
            FROM pending_a2a_questions
            WHERE status = 'WAITING' AND deadline < ?
            """,
            (ts,),
        )
        return [self._row_to_dc(r) for r in rows]

    async def mark_expired(self, task_id: str) -> bool:
        """Transition WAITING → EXPIRED (terminal). Same idempotency
        semantics as ``mark_resolved``."""
        rows = await self._db.fetchall(
            """
            UPDATE pending_a2a_questions
            SET status = 'EXPIRED', resolved_at = CURRENT_TIMESTAMP
            WHERE task_id = ? AND status = 'WAITING'
            RETURNING task_id
            """,
            (task_id,),
        )
        return bool(rows)

    async def _list_by_status(self, status: str) -> List[PendingA2AQuestion]:
        rows = await self._db.fetchall(
            """
            SELECT task_id, recipient, original_question,
                   origin_turn_id, origin_session_id, deadline,
                   status, created_at, resolved_at
            FROM pending_a2a_questions
            WHERE status = ?
            """,
            (status,),
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
