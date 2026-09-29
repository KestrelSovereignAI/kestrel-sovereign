"""Durable dedup/delivery ledger for the generic wait reconciler (Wave 2 of #1860).

Owns CRUD on the ``wait_signal_state`` table defined in ``async_database.py``.
This is the generic successor to the per-job ``last_signaled_status`` +
``pending_signal_*`` fields talon_monitor used to stash inside ``jobs.json``:
one row per ``(agent_id, kind, handle)`` the reconciler has observed, tracking

  - ``last_signaled_outcome`` — application-level dedup: the terminal
    :class:`~kestrel_sdk.tools.Outcome` value we already delivered a signal
    for, so the next tick does not re-fire the same transition.
  - ``last_delivery_*`` — diagnostics + retry accounting; ``attempts`` caps
    the soft-fail retry loop. ``last_delivery_status`` composes the dispatch
    result with a VISIBILITY verdict (``ok_queued`` / ``ok_unsurfaced`` /
    ``ok_unbound`` / ``ok_visibility_unknown``) rather than recording a bare
    ``ok``, which used to mean only "persisted somewhere" while reading as
    "the user saw it" (#2922).
  - ``last_surface_status`` — the dispatcher's raw account of what the
    ``signal_completed`` SSE emit did, backing the verdict above.
  - ``pending_signal_*`` — the two-phase harvest set: a signal we enqueued
    but have not yet confirmed delivered. ``record_pending`` sets them;
    ``record_delivery``/``clear_pending`` clear them.
  - ``delivery_deferred_until`` / ``delivery_deferrals`` — a wake whose
    cognition could not run because the model provider named its own retry
    time is PARKED until then, and that attempt is refunded (#3302). The
    retry cap guards against a signal the dispatcher will always reject; it
    must not be spent waiting out a rate limit the provider has dated.

Like :class:`PendingA2AQuestionStore`, every query is filtered by
``agent_id`` so a shared backend (e.g. Postgres) cannot leak rows between
agents. ``agent_id`` is the owning agent's DID; tests can pass any string.

Durability rule (codex round 4 P2 on PR #1453 carried forward): writes go
through :meth:`AsyncDatabase.execute`, which commits — never a
``fetchall(UPDATE ... RETURNING ...)`` which surfaces the row to the current
connection but may not durably commit, resurrecting state across a restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Union

from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)

# Accept either a datetime or an ISO string for any timestamp argument.
TimeArg = Union[datetime, str, None]

# Synthetic delivery status the reconciler writes when a transition's retry
# cap is reached and the transition is locked without ever being delivered.
MAX_ATTEMPTS_EXCEEDED = "max_attempts_exceeded"
# Delivery status for a wake parked until a provider-advised retry time
# (#3302). Not a failure of the wake: its attempt was refunded.
DEFERRED_RATE_LIMITED = "deferred_rate_limited"

# The one column list every read selects, in ``_row_to_dc`` order.
_STATE_COLUMNS = """
    kind, handle, last_signaled_outcome, last_delivery_status,
    last_delivery_error, last_delivery_attempts,
    last_delivery_attempt_at, pending_signal_id,
    pending_signaled_target, pending_signal_enqueued_at,
    watching, last_surface_status,
    attempts_signaled_target, last_attempt_started_at,
    delivery_deferred_until, delivery_deferrals
"""


def _coerce_ts(value: TimeArg) -> Optional[datetime]:
    """Normalize a timestamp argument to a naive-UTC datetime.

    Mirrors :meth:`PendingA2AQuestionStore.insert`'s tzinfo-strip: Postgres
    TIMESTAMP columns reject tz-aware datetimes and ISO strings, while SQLite
    accepts datetime values via aiosqlite's adapter. So we parse ISO strings
    to datetimes and strip tzinfo for backend portability.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            # Unparseable string — let SQLite store it as TEXT; Postgres
            # would reject it, but a malformed timestamp is a caller bug we
            # surface rather than silently drop.
            return value  # type: ignore[return-value]
    if isinstance(value, datetime) and value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


@dataclass(frozen=True)
class WaitSignalState:
    """One ``wait_signal_state`` row. Times are returned as ISO-ish strings
    (the backend's stringified TEXT/TIMESTAMP value), matching the
    :class:`PendingA2AQuestion` convention."""

    kind: str
    handle: str
    last_signaled_outcome: Optional[str]
    last_delivery_status: Optional[str]
    last_delivery_error: Optional[str]
    last_delivery_attempts: int
    last_delivery_attempt_at: Optional[str]
    pending_signal_id: Optional[str]
    pending_signaled_target: Optional[str]
    pending_signal_enqueued_at: Optional[str]
    watching: int
    # What the dispatcher observed of the wake's ``signal_completed`` SSE emit
    # (#2922). ``None`` on rows written before the column existed and on
    # outcomes that never reached an emit — genuinely unknown, not "fine".
    last_surface_status: Optional[str] = None
    # Which signaled token ``last_delivery_attempts`` is counting (#3105).
    # Empty on legacy rows, which reads as "a different transition" and
    # restarts the counter — the safe direction for a wake.
    attempts_signaled_target: str = ""
    # When the current attempt was DISPATCHED, as opposed to
    # ``last_delivery_attempt_at``, which the harvest rewrites.
    last_attempt_started_at: Optional[str] = None
    # The provider-advised time this wake is parked until (#3302), or None.
    delivery_deferred_until: Optional[str] = None
    # How many times this transition's wake has been parked that way.
    delivery_deferrals: int = 0

    def deferred_until_utc(self) -> Optional[datetime]:
        """``delivery_deferred_until`` as an aware UTC datetime, or ``None``.

        The column holds naive UTC (see :func:`_coerce_ts`). A value that does
        not parse is treated as no deferral: failing open delivers the wake,
        which is the safe direction.
        """
        raw = self.delivery_deferred_until
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            logger.warning(
                "wait_signal_state %s:%s has an unparseable "
                "delivery_deferred_until %r; ignoring it",
                self.kind, self.handle, raw,
            )
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)


class WaitSignalStore:
    """Async CRUD over ``wait_signal_state`` (Wave 2 of #1860).

    Construct with the agent's :class:`AsyncDatabase` AND the owning agent's
    id (its DID). Every query is filtered by ``agent_id``.
    """

    def __init__(self, db: AsyncDatabase, agent_id: str):
        if not isinstance(agent_id, str):
            raise TypeError(
                f"WaitSignalStore agent_id must be a string, "
                f"got {type(agent_id).__name__}"
            )
        self._db = db
        # Empty string is the back-compat default for solo-agent SQLite
        # (the schema default is also ''), matching PendingA2AQuestionStore.
        self._agent_id = agent_id

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get(self, kind: str, handle: str) -> Optional[WaitSignalState]:
        rows = await self._db.fetchall(
            f"""
            SELECT {_STATE_COLUMNS}
            FROM wait_signal_state
            WHERE agent_id = ? AND kind = ? AND handle = ?
            """,
            (self._agent_id, kind, handle),
        )
        if not rows:
            return None
        return self._row_to_dc(rows[0])

    async def list_pending(self) -> List[WaitSignalState]:
        """All rows with an un-harvested enqueued signal for THIS agent —
        the reconciler's Phase-0 harvest input set."""
        rows = await self._db.fetchall(
            f"""
            SELECT {_STATE_COLUMNS}
            FROM wait_signal_state
            WHERE agent_id = ? AND pending_signal_id IS NOT NULL
            """,
            (self._agent_id,),
        )
        return [self._row_to_dc(r) for r in rows]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def seed_signaled(
        self, kind: str, handle: str, outcome: str,
    ) -> bool:
        """Seed a confirmed-signaled row ONLY IF ABSENT (idempotent).

        Used for one-time migration of legacy per-feature dedup state into
        this generic ledger — e.g. talon_monitor stashed
        ``last_signaled_status`` inside ``jobs.json``; on upgrade we seed a
        row here so the first ``wait_reconcile`` tick does not re-fire a
        signal for an already-delivered terminal handle (codex Wave 2 P2).

        ``INSERT OR IGNORE`` so it never clobbers a row the reconciler is
        already managing — re-running on every startup is a safe no-op.
        Returns True if a row was inserted, False if one already existed.

        Portable across backends: the storage layer's ``sqlite_to_postgres``
        translation (storage/db/placeholder.py) rewrites ``INSERT OR IGNORE
        INTO`` to ``... ON CONFLICT DO NOTHING`` for Postgres, with matching
        rowcount semantics (0 on conflict). This is the same pattern
        ``PendingA2AQuestionStore.insert`` relies on.
        """
        rowcount = await self._db.execute(
            """
            INSERT OR IGNORE INTO wait_signal_state
                (agent_id, kind, handle, last_signaled_outcome, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (self._agent_id, kind, handle, outcome),
        )
        return rowcount > 0

    async def record_pending(
        self,
        kind: str,
        handle: str,
        *,
        signal_id: str,
        target: str,
        attempts: int,
        attempt_at: TimeArg = None,
    ) -> None:
        """Stash an enqueued (but not yet confirmed) signal.

        Sets the three ``pending_signal_*`` fields plus the attempt
        accounting, and spends any provider-advised deferral (the wake is
        being dispatched now). ``delivery_deferrals`` carries over within one
        transition and restarts at 0 for a new one, like the attempt counter.
        Preserves ``last_signaled_outcome`` (this is an in-flight
        record, not a confirmed delivery). Upsert via try-UPDATE / fallback
        INSERT so an existing row keeps its prior ``last_signaled_outcome``
        instead of an ``INSERT OR REPLACE`` blowing it away.
        """
        attempt_dt = _coerce_ts(attempt_at) or datetime.now(timezone.utc).replace(
            tzinfo=None
        )
        rowcount = await self._db.execute(
            """
            UPDATE wait_signal_state
            SET pending_signal_id = ?,
                pending_signaled_target = ?,
                pending_signal_enqueued_at = ?,
                last_delivery_attempts = ?,
                last_delivery_attempt_at = ?,
                delivery_deferrals = CASE
                    WHEN attempts_signaled_target = ? THEN delivery_deferrals
                    ELSE 0
                END,
                attempts_signaled_target = ?,
                last_attempt_started_at = ?,
                delivery_deferred_until = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ? AND kind = ? AND handle = ?
            """,
            (
                signal_id,
                target,
                attempt_dt,
                int(attempts),
                attempt_dt,
                target,
                target,
                attempt_dt,
                self._agent_id,
                kind,
                handle,
            ),
        )
        if rowcount == 0:
            await self._db.execute(
                """
                INSERT INTO wait_signal_state
                    (agent_id, kind, handle, last_delivery_attempts,
                     last_delivery_attempt_at, pending_signal_id,
                     pending_signaled_target, pending_signal_enqueued_at,
                     attempts_signaled_target, last_attempt_started_at,
                     updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    self._agent_id,
                    kind,
                    handle,
                    int(attempts),
                    attempt_dt,
                    signal_id,
                    target,
                    attempt_dt,
                    target,
                    attempt_dt,
                ),
            )

    async def record_delivery(
        self,
        kind: str,
        handle: str,
        *,
        delivery_status: str,
        delivery_error: Optional[str] = None,
        signaled_outcome: Optional[str] = None,
        attempt_at: TimeArg = None,
        surface_status: Optional[str] = None,
    ) -> None:
        """Record the outcome of a harvested delivery.

        Always sets ``last_delivery_status``/``last_delivery_error``/
        ``last_delivery_attempt_at``/``last_surface_status`` and ALWAYS clears
        the three ``pending_signal_*`` fields (the harvest is done). When
        ``signaled_outcome`` is not None it also locks
        ``last_signaled_outcome`` — callers pass it for delivered + hard-fail
        states (stop the loop) and OMIT it for soft-fails (so the next tick
        re-detects and retries).

        ``surface_status`` is the dispatcher's raw account of what the
        ``signal_completed`` emit did (#2922) — provenance for the visibility
        verdict already composed into ``delivery_status``. It is written even
        when ``None`` so a later, less-observable attempt cannot inherit an
        earlier attempt's verdict and read as better than it was.
        """
        attempt_dt = _coerce_ts(attempt_at) or datetime.now(timezone.utc).replace(
            tzinfo=None
        )
        # Build the UPDATE so we only touch last_signaled_outcome when asked.
        # Try UPDATE first; if the row is missing (a soft-fail/lost harvest
        # against a row that was never persisted) INSERT a fresh diagnostic
        # row so list_pending stays clean and the state is queryable.
        if signaled_outcome is not None:
            rowcount = await self._db.execute(
                """
                UPDATE wait_signal_state
                SET last_delivery_status = ?,
                    last_surface_status = ?,
                    last_delivery_error = ?,
                    last_delivery_attempt_at = ?,
                    last_signaled_outcome = ?,
                    pending_signal_id = NULL,
                    pending_signaled_target = NULL,
                    pending_signal_enqueued_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE agent_id = ? AND kind = ? AND handle = ?
                """,
                (
                    delivery_status,
                    surface_status,
                    delivery_error,
                    attempt_dt,
                    signaled_outcome,
                    self._agent_id,
                    kind,
                    handle,
                ),
            )
        else:
            rowcount = await self._db.execute(
                """
                UPDATE wait_signal_state
                SET last_delivery_status = ?,
                    last_surface_status = ?,
                    last_delivery_error = ?,
                    last_delivery_attempt_at = ?,
                    pending_signal_id = NULL,
                    pending_signaled_target = NULL,
                    pending_signal_enqueued_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE agent_id = ? AND kind = ? AND handle = ?
                """,
                (
                    delivery_status,
                    surface_status,
                    delivery_error,
                    attempt_dt,
                    self._agent_id,
                    kind,
                    handle,
                ),
            )
        if rowcount == 0:
            await self._db.execute(
                """
                INSERT INTO wait_signal_state
                    (agent_id, kind, handle, last_signaled_outcome,
                     last_delivery_status, last_surface_status,
                     last_delivery_error, last_delivery_attempt_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    self._agent_id,
                    kind,
                    handle,
                    signaled_outcome,
                    delivery_status,
                    surface_status,
                    delivery_error,
                    attempt_dt,
                ),
            )

    async def record_deferral(
        self,
        kind: str,
        handle: str,
        *,
        target: str,
        retry_at: TimeArg,
        delivery_error: Optional[str] = None,
        attempt_at: TimeArg = None,
    ) -> None:
        """Park a harvested wake until a provider-advised ``retry_at`` (#3302).

        The wake's cognition could not run because every model route declined
        a wait the provider itself named. That is not the dispatcher rejecting
        the signal, so the attempt :meth:`record_pending` charged for this
        dispatch is refunded — only while the row is still counting
        ``target``'s transition — and the wake is re-emitted once
        ``retry_at`` passes. Clears the ``pending_signal_*`` harvest fields and
        leaves ``last_signaled_outcome`` unset, exactly like a soft fail.
        """
        attempt_dt = _coerce_ts(attempt_at) or datetime.now(timezone.utc).replace(
            tzinfo=None
        )
        rowcount = await self._db.execute(
            """
            UPDATE wait_signal_state
            SET last_delivery_status = ?,
                last_surface_status = NULL,
                last_delivery_error = ?,
                last_delivery_attempt_at = ?,
                last_delivery_attempts = CASE
                    WHEN attempts_signaled_target = ?
                         AND last_delivery_attempts > 0
                    THEN last_delivery_attempts - 1
                    ELSE last_delivery_attempts
                END,
                delivery_deferred_until = ?,
                delivery_deferrals = delivery_deferrals + 1,
                pending_signal_id = NULL,
                pending_signaled_target = NULL,
                pending_signal_enqueued_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ? AND kind = ? AND handle = ?
            """,
            (
                DEFERRED_RATE_LIMITED,
                delivery_error,
                attempt_dt,
                target,
                _coerce_ts(retry_at),
                self._agent_id,
                kind,
                handle,
            ),
        )
        if rowcount == 0:
            await self._db.execute(
                """
                INSERT INTO wait_signal_state
                    (agent_id, kind, handle, last_delivery_status,
                     last_delivery_error, last_delivery_attempt_at,
                     attempts_signaled_target, delivery_deferred_until,
                     delivery_deferrals, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                """,
                (
                    self._agent_id,
                    kind,
                    handle,
                    DEFERRED_RATE_LIMITED,
                    delivery_error,
                    attempt_dt,
                    target,
                    _coerce_ts(retry_at),
                ),
            )

    async def list_undelivered(self, limit: int = 50) -> List[WaitSignalState]:
        """Wakes for THIS agent that have not reached it and will not soon.

        Two kinds, most recently updated first:

          * locked — the retry cap was reached and the transition was locked
            as ``max_attempts_exceeded`` without ever being delivered;
          * parked — the wake is deferred until a provider-advised retry time.

        The read surface behind ``wait_status`` (#3302): before it, the only
        way to learn that a job's completion wake had been locked away was to
        query the database by hand.
        """
        rows = await self._db.fetchall(
            f"""
            SELECT {_STATE_COLUMNS}
            FROM wait_signal_state
            WHERE agent_id = ?
                  AND (last_delivery_status = ?
                       OR delivery_deferred_until IS NOT NULL)
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (self._agent_id, MAX_ATTEMPTS_EXCEEDED, int(limit)),
        )
        return [self._row_to_dc(r) for r in rows]

    async def clear_pending(self, kind: str, handle: str) -> None:
        """Null the three ``pending_signal_*`` fields without touching the
        signaled/delivery state. Used on a restart-lost harvest so the next
        tick re-detects + retries the transition."""
        await self._db.execute(
            """
            UPDATE wait_signal_state
            SET pending_signal_id = NULL,
                pending_signaled_target = NULL,
                pending_signal_enqueued_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ? AND kind = ? AND handle = ?
            """,
            (self._agent_id, kind, handle),
        )

    # ------------------------------------------------------------------
    # Explicit watched handles (wait mode="signal")
    # ------------------------------------------------------------------

    async def start_watch(self, kind: str, handle: str) -> None:
        """Register an explicit watch on ``(kind, handle)`` (set watching=1).

        Upsert that PRESERVES any existing delivery/signaled/pending fields:
        a row may already exist from a prior delivery cycle. Try-UPDATE first;
        if no row, INSERT a fresh one with watching=1 and zeroed counters.
        This is the durable half of ``wait(target, mode="signal")`` — the
        reconciler polls watched rows so even a poll-only provider is wakeable.
        """
        rowcount = await self._db.execute(
            """
            UPDATE wait_signal_state
            SET watching = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ? AND kind = ? AND handle = ?
            """,
            (self._agent_id, kind, handle),
        )
        if rowcount == 0:
            await self._db.execute(
                """
                INSERT INTO wait_signal_state
                    (agent_id, kind, handle, watching, updated_at)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                """,
                (self._agent_id, kind, handle),
            )

    async def stop_watch(self, kind: str, handle: str) -> None:
        """Clear the explicit watch on ``(kind, handle)`` (set watching=0)."""
        await self._db.execute(
            """
            UPDATE wait_signal_state
            SET watching = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = ? AND kind = ? AND handle = ?
            """,
            (self._agent_id, kind, handle),
        )

    async def list_watched(self) -> List[WaitSignalState]:
        """Active explicit watches — watching=1 AND not yet signaled.

        Once a watch's transition is delivered (``last_signaled_outcome`` is
        set) it drops out of the reconciler's watched-poll set, mirroring the
        application-level dedup the active_handles loop applies.
        """
        rows = await self._db.fetchall(
            f"""
            SELECT {_STATE_COLUMNS}
            FROM wait_signal_state
            WHERE agent_id = ? AND watching = 1
                  AND last_signaled_outcome IS NULL
            """,
            (self._agent_id,),
        )
        return [self._row_to_dc(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dc(r) -> WaitSignalState:
        return WaitSignalState(
            kind=r[0],
            handle=r[1],
            last_signaled_outcome=str(r[2]) if r[2] is not None else None,
            last_delivery_status=str(r[3]) if r[3] is not None else None,
            last_delivery_error=str(r[4]) if r[4] is not None else None,
            last_delivery_attempts=int(r[5]) if r[5] is not None else 0,
            last_delivery_attempt_at=str(r[6]) if r[6] is not None else None,
            pending_signal_id=str(r[7]) if r[7] is not None else None,
            pending_signaled_target=str(r[8]) if r[8] is not None else None,
            pending_signal_enqueued_at=str(r[9]) if r[9] is not None else None,
            watching=int(r[10]) if r[10] is not None else 0,
            last_surface_status=str(r[11]) if r[11] is not None else None,
            attempts_signaled_target=str(r[12]) if r[12] is not None else "",
            last_attempt_started_at=str(r[13]) if r[13] is not None else None,
            delivery_deferred_until=str(r[14]) if r[14] is not None else None,
            delivery_deferrals=int(r[15]) if r[15] is not None else 0,
        )
