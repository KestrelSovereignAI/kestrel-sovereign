"""Sender-side store for in-flight ``send_a2a_question`` correlation rows.

Owns CRUD on the ``pending_a2a_questions`` table defined in
``async_database.py``. Used by:

  - ``PeersFeature._post_a2a_task`` (insert on POST + spawn subscription;
    update to RESOLVED when terminal event arrives)
  - Startup-replay sweep (read WAITING rows on boot; for each, GET the
    recipient's task state; if terminal, fire signal locally)
  - Hourly expiry sweep (mark rows past deadline as EXPIRED; fire signal
    with ``state='expired'`` so the resumed prompt has a clean branch)
  - Receiver settlement (#3440): an EXPIRED row records whether the
    recipient's task is confirmed terminal. ``PENDING`` rows are retried
    by the sweep until the recipient's task is withdrawn or answered; an
    answer is held on the row as ``LATE_ANSWER`` until its wake is
    delivered. Each pass works on a row only under its settlement lease.

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

# ``receiver_settlement`` values (#3440). NULL means nothing is owed: the row
# is WAITING, was RESOLVED with the recipient's own answer, or predates the
# column. An EXPIRED row is PENDING until the recipient's task is confirmed
# terminal. A recipient that answered on its own moves it to LATE_ANSWER,
# which holds that answer on the row until its wake to the asker is
# delivered. Only then, or once the recipient's task is withdrawn, is it
# CONFIRMED.
RECEIVER_SETTLEMENT_PENDING = "PENDING"
RECEIVER_SETTLEMENT_LATE_ANSWER = "LATE_ANSWER"
RECEIVER_SETTLEMENT_CONFIRMED = "CONFIRMED"

# How long one pass owns an unsettled row. The expiry that leaves a row
# PENDING takes the lease too, so its own ``expired`` wake reaches the asker
# before any late answer can. Whoever holds it renews it while a wake for the
# row is in flight (a cognition turn has no fixed length). A pass that dies
# holding it (a crash between recording a late answer and delivering it) stops
# renewing, and the first pass after it lapses resumes the row; the sweep runs
# hourly, so this must stay well under an hour.
RECEIVER_SETTLEMENT_LEASE = timedelta(minutes=15)

_COLUMNS = """
    task_id, recipient, original_question,
    origin_turn_id, origin_session_id, deadline,
    status, created_at, resolved_at,
    retry_state, retry_reply_text, recipient_agent_id,
    receiver_settlement, receiver_late_state, receiver_late_reply_text
"""


def _naive_utc(value: Optional[datetime]) -> datetime:
    """``value`` (default now) as the naive UTC that TIMESTAMP columns bind.

    See ``PendingA2AQuestionStore.insert`` for why tzinfo is stripped."""
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


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
    retry_state: Optional[str] = None
    retry_reply_text: Optional[str] = None
    recipient_agent_id: Optional[str] = None
    receiver_settlement: Optional[str] = None
    # The recipient's own terminal state and reply, owed to the asker while
    # ``receiver_settlement`` is LATE_ANSWER.
    receiver_late_state: Optional[str] = None
    receiver_late_reply_text: Optional[str] = None


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
        recipient_agent_id: Optional[str] = None,
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
        if recipient_agent_id is not None and (
            not isinstance(recipient_agent_id, str)
            or not recipient_agent_id.strip()
        ):
            raise ValueError("recipient_agent_id must be a non-empty string or None")
        await self._db.execute(
            """
            INSERT OR IGNORE INTO pending_a2a_questions
                (agent_id, task_id, recipient, recipient_agent_id, original_question,
                 origin_turn_id, origin_session_id, deadline, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'WAITING', CURRENT_TIMESTAMP)
            """,
            (
                self._agent_id,
                task_id,
                recipient,
                recipient_agent_id,
                original_question,
                origin_turn_id,
                origin_session_id,
                deadline,
            ),
        )

    async def get(self, task_id: str) -> Optional[PendingA2AQuestion]:
        rows = await self._db.fetchall(
            f"""
            SELECT {_COLUMNS}
            FROM pending_a2a_questions
            WHERE agent_id = ? AND task_id = ?
            """,
            (self._agent_id, task_id),
        )
        if not rows:
            return None
        return self._row_to_dc(rows[0])

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
            SET status = 'RESOLVED',
                resolved_at = CURRENT_TIMESTAMP,
                retry_state = NULL,
                retry_reply_text = NULL
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
            f"""
            SELECT {_COLUMNS}
            FROM pending_a2a_questions
            WHERE agent_id = ? AND status = 'WAITING' AND deadline < ?
            """,
            (self._agent_id, ts_dt),
        )
        return [self._row_to_dc(r) for r in rows]

    async def mark_expired(
        self,
        task_id: str,
        *,
        receiver_settled: bool = False,
        now: Optional[datetime] = None,
    ) -> bool:
        """Transition WAITING → EXPIRED (terminal) for THIS agent's
        row. Same idempotency + cross-agent semantics as
        ``mark_resolved``. ``execute()`` is used for the same
        durability reason — see ``mark_resolved`` docstring + codex
        round 4 P2.

        ``receiver_settled`` records whether the recipient's task is
        confirmed terminal (#3440). An unconfirmed expiry is left
        ``PENDING`` so the sweep keeps settling it with the recipient. A
        row re-expired after a failed wake keeps the settlement it already
        recorded: only a confirmation can move it. An unsettled expiry
        leases the row to itself, so no settlement pass relays a late answer
        while its ``expired`` wake is still on its way. (A lease means
        nothing on a confirmed row, which no pass claims.)"""
        if receiver_settled:
            settlement_sql = f"'{RECEIVER_SETTLEMENT_CONFIRMED}'"
            lease_until = None
        else:
            settlement_sql = (
                f"COALESCE(receiver_settlement, '{RECEIVER_SETTLEMENT_PENDING}')"
            )
            lease_until = _naive_utc(now) + RECEIVER_SETTLEMENT_LEASE
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET status = 'EXPIRED',
                resolved_at = CURRENT_TIMESTAMP,
                retry_state = NULL,
                retry_reply_text = NULL,
                receiver_settlement = {settlement_sql},
                receiver_settlement_lease_until = ?
            WHERE agent_id = ? AND task_id = ? AND status = 'WAITING'
            """,
            (lease_until, self._agent_id, task_id),
        )
        return rowcount > 0

    async def list_expired_awaiting_receiver(self) -> List[PendingA2AQuestion]:
        """EXPIRED rows whose recipient task is not confirmed terminal, or
        whose recipient's answer has not yet reached the asker.

        There is no age cutoff: the recipient retains an undelivered
        submission for as long as it stays undelivered, and its execution
        gate checks only the task's state, so a recipient returning after
        any absence can still run a live task. A row leaves this set only
        once both stores agree.

        Least recently leased first: a row a pass just attempted goes to the
        back, so one row that keeps failing cannot hold its siblings behind
        it pass after pass."""
        rows = await self._db.fetchall(
            f"""
            SELECT {_COLUMNS}
            FROM pending_a2a_questions
            WHERE agent_id = ? AND status = 'EXPIRED'
              AND receiver_settlement IN (
                  '{RECEIVER_SETTLEMENT_PENDING}',
                  '{RECEIVER_SETTLEMENT_LATE_ANSWER}'
              )
            ORDER BY (receiver_settlement_lease_until IS NOT NULL),
                     receiver_settlement_lease_until, task_id
            """,
            (self._agent_id,),
        )
        return [self._row_to_dc(r) for r in rows]

    async def claim_receiver_settlement(
        self, task_id: str, *, now: Optional[datetime] = None,
    ) -> Optional[PendingA2AQuestion]:
        """Lease an unsettled row to the calling pass; return it as claimed.

        ``None`` when another pass (or the expiry that left it unsettled)
        holds a live lease, or there is nothing left to settle. A pass never
        hands its lease back: it lapses (or is cleared by confirmation), so a
        pass that dies holding it is resumed by the next pass after
        ``RECEIVER_SETTLEMENT_LEASE``."""
        now = _naive_utc(now)
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET receiver_settlement_lease_until = ?
            WHERE agent_id = ? AND task_id = ? AND status = 'EXPIRED'
              AND receiver_settlement IN (
                  '{RECEIVER_SETTLEMENT_PENDING}',
                  '{RECEIVER_SETTLEMENT_LATE_ANSWER}'
              )
              AND (
                  receiver_settlement_lease_until IS NULL
                  OR receiver_settlement_lease_until <= ?
              )
            """,
            (now + RECEIVER_SETTLEMENT_LEASE, self._agent_id, task_id, now),
        )
        if rowcount <= 0:
            return None
        return await self.get(task_id)

    async def renew_receiver_settlement_lease(
        self, task_id: str, *, now: Optional[datetime] = None,
    ) -> bool:
        """Extend the lease on an unsettled row while a wake for it is in
        flight. Never shortens one.

        Returns whether the row is still unsettled, so still worth holding:
        False once it is confirmed, or no longer EXPIRED."""
        lease_until = _naive_utc(now) + RECEIVER_SETTLEMENT_LEASE
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET receiver_settlement_lease_until = ?
            WHERE agent_id = ? AND task_id = ? AND status = 'EXPIRED'
              AND receiver_settlement IN (
                  '{RECEIVER_SETTLEMENT_PENDING}',
                  '{RECEIVER_SETTLEMENT_LATE_ANSWER}'
              )
              AND (
                  receiver_settlement_lease_until IS NULL
                  OR receiver_settlement_lease_until < ?
              )
            """,
            (lease_until, self._agent_id, task_id, lease_until),
        )
        if rowcount > 0:
            return True
        # Not extended: settled, or already leased for longer.
        rows = await self._db.fetchall(
            f"""
            SELECT 1 FROM pending_a2a_questions
            WHERE agent_id = ? AND task_id = ? AND status = 'EXPIRED'
              AND receiver_settlement IN (
                  '{RECEIVER_SETTLEMENT_PENDING}',
                  '{RECEIVER_SETTLEMENT_LATE_ANSWER}'
              )
            """,
            (self._agent_id, task_id),
        )
        return bool(rows)

    async def record_late_answer(
        self, task_id: str, *, state: str, reply_text: str,
    ) -> bool:
        """PENDING → LATE_ANSWER, holding the recipient's own terminal state
        and reply on the row.

        This is the durable obligation to wake the asker, written before
        that wake is enqueued: a pass that dies before delivering it leaves
        the answer here for the next pass. Returns False when the row was not
        PENDING."""
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET receiver_settlement = '{RECEIVER_SETTLEMENT_LATE_ANSWER}',
                receiver_late_state = ?,
                receiver_late_reply_text = ?
            WHERE agent_id = ? AND task_id = ? AND status = 'EXPIRED'
              AND receiver_settlement = '{RECEIVER_SETTLEMENT_PENDING}'
            """,
            (state, reply_text, self._agent_id, task_id),
        )
        return rowcount > 0

    async def confirm_receiver_withdrawn(self, task_id: str) -> bool:
        """PENDING → CONFIRMED: the recipient's task is withdrawn and nothing
        is owed to the asker."""
        return await self._confirm_receiver_settlement(
            task_id, from_value=RECEIVER_SETTLEMENT_PENDING,
        )

    async def confirm_late_answer_delivered(self, task_id: str) -> bool:
        """LATE_ANSWER → CONFIRMED, once the late answer's wake was delivered.
        Returns False when it was already confirmed."""
        return await self._confirm_receiver_settlement(
            task_id, from_value=RECEIVER_SETTLEMENT_LATE_ANSWER,
        )

    async def _confirm_receiver_settlement(
        self, task_id: str, *, from_value: str,
    ) -> bool:
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET receiver_settlement = '{RECEIVER_SETTLEMENT_CONFIRMED}',
                receiver_settlement_lease_until = NULL,
                receiver_late_state = NULL,
                receiver_late_reply_text = NULL
            WHERE agent_id = ? AND task_id = ? AND status = 'EXPIRED'
              AND receiver_settlement = ?
            """,
            (self._agent_id, task_id, from_value),
        )
        return rowcount > 0

    async def mark_waiting_for_retry(
        self,
        task_id: str,
        *,
        state: Optional[str] = None,
        reply_text: Optional[str] = None,
    ) -> bool:
        """Restore a terminal row to WAITING after a dispatch failure.

        Callers that mark a question RESOLVED/EXPIRED before enqueueing the
        resumption signal must use this when enqueue fails. Otherwise the
        terminal row disappears from startup replay/hourly sweep and the asker
        is never woken. When the terminal payload is known, preserve it so a
        later replay can wake the asker with the real answer instead of an
        empty expiry.
        """
        rowcount = await self._db.execute(
            """
            UPDATE pending_a2a_questions
            SET status = 'WAITING',
                resolved_at = NULL,
                retry_state = ?,
                retry_reply_text = ?
            WHERE agent_id = ? AND task_id = ? AND status IN ('RESOLVED', 'EXPIRED')
            """,
            (state, reply_text, self._agent_id, task_id),
        )
        return rowcount > 0

    async def _list_by_status(self, status: str) -> List[PendingA2AQuestion]:
        rows = await self._db.fetchall(
            f"""
            SELECT {_COLUMNS}
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
            retry_state=str(r[9]) if len(r) > 9 and r[9] else None,
            retry_reply_text=str(r[10]) if len(r) > 10 and r[10] else None,
            recipient_agent_id=str(r[11]) if len(r) > 11 and r[11] else None,
            receiver_settlement=str(r[12]) if len(r) > 12 and r[12] else None,
            receiver_late_state=str(r[13]) if len(r) > 13 and r[13] else None,
            receiver_late_reply_text=(
                str(r[14]) if len(r) > 14 and r[14] is not None else None
            ),
        )
