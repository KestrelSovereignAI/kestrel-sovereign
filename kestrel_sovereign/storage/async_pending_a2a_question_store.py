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
    delivered.

A row leaves WAITING when a pass claims it to wake the asker, and the wake it
owes stays on the row (``retry_state`` / ``retry_reply_text``) until its
delivery is confirmed (#3440). A pass that dies in between leaves the wake
owed, and once its claim lapses the row returns to WAITING carrying that wake,
where startup replay and the hourly sweep deliver it.

Each pass acts on a row only under its claim: an opaque token that every
later mutation names, and a lease that decides when another pass may take the
row over.

The table is sender-side state — not visible to the recipient. Auth is
implicit (it lives in the sender's per-agent DB, behind the same access
gate as the rest of agent storage).
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)

# ``receiver_settlement`` values (#3440). NULL means nothing is owed: the row
# is WAITING or was RESOLVED with the recipient's own answer. (An EXPIRED row
# that predates the column is adopted as PENDING when the schema is
# initialized.) An EXPIRED row is PENDING until the recipient's task is
# confirmed terminal. A recipient that answered on its own moves it to
# LATE_ANSWER, which holds that answer on the row until its wake to the asker
# is delivered. Only then, or once the recipient's task is withdrawn, is it
# CONFIRMED.
RECEIVER_SETTLEMENT_PENDING = "PENDING"
RECEIVER_SETTLEMENT_LATE_ANSWER = "LATE_ANSWER"
RECEIVER_SETTLEMENT_CONFIRMED = "CONFIRMED"
_UNSETTLED = (
    f"('{RECEIVER_SETTLEMENT_PENDING}', '{RECEIVER_SETTLEMENT_LATE_ANSWER}')"
)

# A wake the asker is owed that was claimed for delivery (the row left
# WAITING) and is not yet confirmed delivered.
_WAKE_OWED = "(status IN ('RESOLVED', 'EXPIRED') AND retry_state IS NOT NULL)"
# An expired question its recipient does not yet agree on.
_SETTLEMENT_OPEN = f"(status = 'EXPIRED' AND receiver_settlement IN {_UNSETTLED})"
# Everything a claim can hold a row for.
_CLAIMED_WORK = f"({_WAKE_OWED} OR {_SETTLEMENT_OPEN})"

# How long one claim keeps a row from being claimed by anyone else. Whoever
# holds the claim renews its lease while a wake for the row is in flight (a
# cognition turn has no fixed length). A pass that dies holding it (a crash
# between claiming a row and delivering its wake, or between recording a late
# answer and relaying it) stops renewing, and the first pass after it lapses
# takes the row over and resumes it; the sweep runs hourly, so this must stay
# well under an hour. The claim on a row whose first wake is owed also keeps
# settlement off it, so the asker hears that wake before any late answer.
#
# A lease decides only when a row may be taken from its holder. Who may act
# on the row is decided by its claim token: each claim replaces it, and every
# renewal, confirmation, restore, recorded answer and late relay names the
# claim it acts under and is refused once that claim was replaced. A pass
# paused past its lease and overtaken by another can therefore do nothing
# more to the row (#3440).
QUESTION_CLAIM_LEASE = timedelta(minutes=15)


def new_question_claim() -> str:
    """A fresh, opaque claim token."""
    return secrets.token_hex(16)


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
    # The wake the asker is owed and is not yet confirmed to have received:
    # on a WAITING row, waiting to be retried; on a RESOLVED or EXPIRED row,
    # in flight under the row's claim.
    retry_state: Optional[str] = None
    retry_reply_text: Optional[str] = None
    recipient_agent_id: Optional[str] = None
    receiver_settlement: Optional[str] = None
    # The recipient's own terminal state and reply, owed to the asker while
    # ``receiver_settlement`` is LATE_ANSWER.
    receiver_late_state: Optional[str] = None
    receiver_late_reply_text: Optional[str] = None


@dataclass(frozen=True)
class ReceiverSettlementClaim:
    """One pass's claim on an unsettled expired question (#3440).

    ``token`` is opaque. Every later mutation of the row passes it back and
    is refused once a later claim has replaced it."""

    question: PendingA2AQuestion
    token: str


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

    async def mark_resolved(
        self,
        task_id: str,
        *,
        state: str,
        reply_text: str,
        claim: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """Transition WAITING → RESOLVED for THIS agent's row, claiming it
        under ``claim`` to wake the asker with ``state`` and ``reply_text``.
        Returns True if a WAITING row was found and updated, False if the
        row was already terminal (RESOLVED or EXPIRED) or belonged to a
        different agent — both cases are benign (subscription racing a
        startup-replay; another agent's row on a shared backend) and the
        caller should drop their resolve-side signal silently.

        The wake stays owed on the row until ``confirm_wake_delivered``
        (#3440): the claim is the dedupe, not the retirement. A pass that
        dies before confirming leaves the wake for ``reclaim_abandoned_wakes``.

        Codex round 4 P2 on PR #1453: use ``execute()`` (which commits
        on SQLite) rather than ``fetchall(UPDATE ... RETURNING ...)``
        — the prior implementation surfaced the row to the current
        connection but never durably wrote it, so restarting the agent
        resurrected RESOLVED rows as WAITING and the startup-replay
        sweep re-fired the resumption signal as a duplicate. Rowcount
        > 0 carries the same conditional-update semantics as a
        ``RETURNING task_id`` row count."""
        if not isinstance(state, str) or not state:
            raise ValueError("a resolved question's wake needs its terminal state")
        rowcount = await self._db.execute(
            """
            UPDATE pending_a2a_questions
            SET status = 'RESOLVED',
                resolved_at = CURRENT_TIMESTAMP,
                retry_state = ?,
                retry_reply_text = ?,
                claim_token = ?,
                claim_lease_until = ?
            WHERE agent_id = ? AND task_id = ? AND status = 'WAITING'
            """,
            (
                state,
                reply_text,
                claim,
                _naive_utc(now) + QUESTION_CLAIM_LEASE,
                self._agent_id,
                task_id,
            ),
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
        claim: str,
        receiver_settled: bool = False,
        now: Optional[datetime] = None,
    ) -> bool:
        """Transition WAITING → EXPIRED (terminal) for THIS agent's row,
        claiming it under ``claim`` to wake the asker with ``expired``. Same
        idempotency + cross-agent semantics as ``mark_resolved``, and the
        ``expired`` wake stays owed on the row the same way until
        ``confirm_wake_delivered``. ``execute()`` is used for the same
        durability reason — see ``mark_resolved`` docstring + codex round 4
        P2.

        ``receiver_settled`` records whether the recipient's task is
        confirmed terminal (#3440). An unconfirmed expiry is left
        ``PENDING`` so the sweep keeps settling it with the recipient. A
        row re-expired after a failed wake keeps the settlement it already
        recorded: only a confirmation can move it, and a late answer held
        on the row is retired only by its delivery. Settlement cannot claim
        the row while its ``expired`` wake is owed, so no late answer is
        relayed ahead of it.

        A row that owes the asker a real answer is not expired: ``expired``
        would replace that answer. Its wake is delivered as the retry it is.
        """
        if receiver_settled:
            settlement_sql = (
                f"CASE WHEN receiver_settlement = "
                f"'{RECEIVER_SETTLEMENT_LATE_ANSWER}' "
                f"THEN receiver_settlement "
                f"ELSE '{RECEIVER_SETTLEMENT_CONFIRMED}' END"
            )
        else:
            settlement_sql = (
                f"COALESCE(receiver_settlement, '{RECEIVER_SETTLEMENT_PENDING}')"
            )
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET status = 'EXPIRED',
                resolved_at = CURRENT_TIMESTAMP,
                retry_state = 'expired',
                retry_reply_text = NULL,
                receiver_settlement = {settlement_sql},
                claim_token = ?,
                claim_lease_until = ?
            WHERE agent_id = ? AND task_id = ? AND status = 'WAITING'
              AND (retry_state IS NULL OR retry_state = 'expired')
            """,
            (
                claim,
                _naive_utc(now) + QUESTION_CLAIM_LEASE,
                self._agent_id,
                task_id,
            ),
        )
        return rowcount > 0

    async def confirm_wake_delivered(self, task_id: str, *, claim: str) -> bool:
        """Retire the wake owed on a RESOLVED or EXPIRED row once it was
        delivered under ``claim`` (#3440).

        Returns False when no wake is owed, or ``claim`` no longer holds the
        row. A row with nothing else to do drops its claim; an expired row
        its recipient does not yet agree on keeps it until the lease lapses,
        so settlement starts no sooner than it would have."""
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET retry_state = NULL,
                retry_reply_text = NULL,
                claim_token = CASE WHEN {_SETTLEMENT_OPEN}
                    THEN claim_token ELSE NULL END,
                claim_lease_until = CASE WHEN {_SETTLEMENT_OPEN}
                    THEN claim_lease_until ELSE NULL END
            WHERE agent_id = ? AND task_id = ? AND {_WAKE_OWED}
              AND claim_token = ?
            """,
            (self._agent_id, task_id, claim),
        )
        return rowcount > 0

    async def reclaim_abandoned_wakes(
        self, *, now: Optional[datetime] = None,
    ) -> List[PendingA2AQuestion]:
        """Return to WAITING every row whose owed wake's claim lapsed.

        Such a wake was claimed for delivery and never confirmed: its pass
        died, or lost its row. The row goes back to WAITING carrying that
        wake, as a failed delivery restores it, so replay and the sweep
        deliver it under a fresh claim; an expired row keeps the receiver
        settlement it recorded. A lease that has not lapsed may still have
        its wake in flight, and is left alone. Returns the rows this call
        returned to WAITING."""
        now = _naive_utc(now)
        candidates = await self._db.fetchall(
            f"""
            SELECT task_id FROM pending_a2a_questions
            WHERE agent_id = ? AND {_WAKE_OWED}
              AND (claim_lease_until IS NULL OR claim_lease_until <= ?)
            """,
            (self._agent_id, now),
        )
        reclaimed: List[PendingA2AQuestion] = []
        for candidate in candidates:
            task_id = candidate[0]
            rowcount = await self._db.execute(
                f"""
                UPDATE pending_a2a_questions
                SET status = 'WAITING',
                    resolved_at = NULL,
                    claim_token = NULL,
                    claim_lease_until = NULL
                WHERE agent_id = ? AND task_id = ? AND {_WAKE_OWED}
                  AND (claim_lease_until IS NULL OR claim_lease_until <= ?)
                """,
                (self._agent_id, task_id, now),
            )
            if rowcount <= 0:
                continue  # another pass reclaimed it, or its holder renewed
            question = await self.get(task_id)
            if question is not None:
                reclaimed.append(question)
        return reclaimed

    async def list_expired_awaiting_receiver(self) -> List[PendingA2AQuestion]:
        """EXPIRED rows whose recipient task is not confirmed terminal, or
        whose recipient's answer has not yet reached the asker.

        There is no age cutoff: the recipient retains an undelivered
        submission for as long as it stays undelivered, and its execution
        gate checks only the task's state, so a recipient returning after
        any absence can still run a live task. A row leaves this set only
        once both stores agree. Rows expired before this column existed are
        adopted into it as PENDING at schema initialization. A row whose
        ``expired`` wake is still owed is not settled until that wake is
        delivered.

        Least recently leased first: a row a pass just attempted goes to the
        back, so one row that keeps failing cannot hold its siblings behind
        it pass after pass."""
        rows = await self._db.fetchall(
            f"""
            SELECT {_COLUMNS}
            FROM pending_a2a_questions
            WHERE agent_id = ? AND {_SETTLEMENT_OPEN} AND retry_state IS NULL
            ORDER BY (claim_lease_until IS NOT NULL),
                     claim_lease_until, task_id
            """,
            (self._agent_id,),
        )
        return [self._row_to_dc(r) for r in rows]

    async def claim_receiver_settlement(
        self, task_id: str, *, now: Optional[datetime] = None,
    ) -> Optional[ReceiverSettlementClaim]:
        """Claim an unsettled row for the calling pass.

        ``None`` when another pass (or the expiry that left it unsettled)
        holds a lease that has not lapsed, the row's ``expired`` wake is
        still owed, or there is nothing left to settle. A pass never hands
        its claim back: it lapses (or is cleared by confirmation), so a pass
        that dies holding it is resumed by the next pass after
        ``QUESTION_CLAIM_LEASE``. Claiming replaces the row's claim token,
        which fences the previous holder out of every later mutation."""
        now = _naive_utc(now)
        token = new_question_claim()
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET claim_lease_until = ?,
                claim_token = ?
            WHERE agent_id = ? AND task_id = ? AND {_SETTLEMENT_OPEN}
              AND retry_state IS NULL
              AND (claim_lease_until IS NULL OR claim_lease_until <= ?)
            """,
            (now + QUESTION_CLAIM_LEASE, token, self._agent_id, task_id, now),
        )
        if rowcount <= 0:
            return None
        question = await self.get(task_id)
        if question is None:
            return None
        return ReceiverSettlementClaim(question=question, token=token)

    async def renew_claim_lease(
        self, task_id: str, *, claim: str, now: Optional[datetime] = None,
    ) -> bool:
        """Extend ``claim``'s lease while a wake for its row is in flight.
        Never shortens one.

        Returns whether ``claim`` still holds the row with work left on it
        (a wake owed, or a recipient not yet settled): False once that work
        is done, or another claim has replaced it. A late relay asks this
        immediately before it enqueues, so it is also the check that a pass
        still owns the row it is about to act on."""
        lease_until = _naive_utc(now) + QUESTION_CLAIM_LEASE
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET claim_lease_until = ?
            WHERE agent_id = ? AND task_id = ? AND {_CLAIMED_WORK}
              AND claim_token = ?
              AND (claim_lease_until IS NULL OR claim_lease_until < ?)
            """,
            (lease_until, self._agent_id, task_id, claim, lease_until),
        )
        if rowcount > 0:
            return True
        # Not extended: no longer ours to hold, or already leased for longer.
        return await self._holds_claim(task_id, claim)

    async def _holds_claim(self, task_id: str, claim: str) -> bool:
        rows = await self._db.fetchall(
            f"""
            SELECT 1 FROM pending_a2a_questions
            WHERE agent_id = ? AND task_id = ? AND {_CLAIMED_WORK}
              AND claim_token = ?
            """,
            (self._agent_id, task_id, claim),
        )
        return bool(rows)

    async def record_late_answer(
        self, task_id: str, *, claim: str, state: str, reply_text: str,
    ) -> bool:
        """PENDING → LATE_ANSWER under ``claim``, holding the recipient's own
        terminal state and reply on the row.

        This is the durable obligation to wake the asker, written before
        that wake is enqueued: a pass that dies before delivering it leaves
        the answer here for the next pass. Returns False when the row was not
        PENDING, or ``claim`` no longer holds it."""
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET receiver_settlement = '{RECEIVER_SETTLEMENT_LATE_ANSWER}',
                receiver_late_state = ?,
                receiver_late_reply_text = ?
            WHERE agent_id = ? AND task_id = ? AND status = 'EXPIRED'
              AND receiver_settlement = '{RECEIVER_SETTLEMENT_PENDING}'
              AND claim_token = ?
            """,
            (state, reply_text, self._agent_id, task_id, claim),
        )
        return rowcount > 0

    async def confirm_receiver_withdrawn(self, task_id: str, *, claim: str) -> bool:
        """PENDING → CONFIRMED under ``claim``: the recipient's task is
        withdrawn and nothing is owed to the asker."""
        return await self._confirm_receiver_settlement(
            task_id, claim=claim, from_value=RECEIVER_SETTLEMENT_PENDING,
        )

    async def confirm_late_answer_delivered(
        self, task_id: str, *, claim: str,
    ) -> bool:
        """LATE_ANSWER → CONFIRMED under ``claim``, once the late answer's
        wake was delivered. Returns False when it was already confirmed, or
        ``claim`` no longer holds the row."""
        return await self._confirm_receiver_settlement(
            task_id, claim=claim, from_value=RECEIVER_SETTLEMENT_LATE_ANSWER,
        )

    async def _confirm_receiver_settlement(
        self, task_id: str, *, claim: str, from_value: str,
    ) -> bool:
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET receiver_settlement = '{RECEIVER_SETTLEMENT_CONFIRMED}',
                claim_lease_until = NULL,
                claim_token = NULL,
                receiver_late_state = NULL,
                receiver_late_reply_text = NULL
            WHERE agent_id = ? AND task_id = ? AND status = 'EXPIRED'
              AND receiver_settlement = ?
              AND claim_token = ?
            """,
            (self._agent_id, task_id, from_value, claim),
        )
        return rowcount > 0

    async def mark_waiting_for_retry(
        self,
        task_id: str,
        *,
        claim: str,
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

        Only the pass whose ``claim`` took the row out of WAITING to deliver
        the wake still owed on it may restore it (#3440). One overtaken after
        its lease lapsed would otherwise pull the row out from under the pass
        now delivering it; one whose wake was already confirmed delivered, or
        a settlement pass, has no wake to restore.
        """
        rowcount = await self._db.execute(
            f"""
            UPDATE pending_a2a_questions
            SET status = 'WAITING',
                resolved_at = NULL,
                retry_state = ?,
                retry_reply_text = ?,
                claim_token = NULL,
                claim_lease_until = NULL
            WHERE agent_id = ? AND task_id = ? AND {_WAKE_OWED}
              AND claim_token = ?
            """,
            (state, reply_text, self._agent_id, task_id, claim),
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
