"""Durable operator-notice audit record (#2530).

An operator notice is trusted runtime state annotated onto a turn that is
*already in flight* — an auto-mode change, a low-token-budget warning, a
governance delta. It does not wake the agent and it does not originate work,
so by the definition in ``docs/architecture/SIGNAL_DISPATCHER.md`` ("mechanisms
that legitimately wake it up or trigger work") it is not a signal.

Before #2530 its audit was written into ``signal_log`` anyway, which required
fabricating an unregistered ``SourceRegistration`` and a hardcoded
``SignalResult(Status.OK, {"delivered": True})``. Two lies to make one row
fit: a source contract nothing had registered, and a delivery claim asserted
at COLLECT time — before the notice had been injected, before any provider
saw it, before a turn completed. A store you have to lie to write to is
telling you the data does not belong in it. The parallel mechanism already
existed; it was just hidden inside ``signal_log``. This module names it.

Lifecycle (see :class:`OperatorNoticeState`)::

    collected ──> injected ──> delivered
                          └──> failed | cancelled

``collected`` is written the moment the producer consumes its pending events
and advances its dedupe state, and it deliberately does NOT claim delivery.
That is the negative evidence this ticket exists to preserve: a notice that
was collected and then vanished shows up as a row that never settled, rather
than as nothing at all. It is the same lesson as ``restart_requests``'
``wake_dispatched_at`` (#2774) — a record written only after the fact is
invisible to the reader that needs it.

The rows carry operator facts (scope, remaining tokens, governance state) and
routing metadata only. The rendered notice prose is not stored here, and no
user content ever reaches this table.

Schema note: the table is declared in ``async_database.CORE_SCHEMA`` alongside
``wait_signal_state`` and ``pending_a2a_questions``, so it is created by the
one idempotent ``CREATE TABLE IF NOT EXISTS`` path every backend already runs.
There is no hand-rolled ``ALTER``/backfill here; any future additive column
must go through :meth:`AsyncDatabase.migrate_columns_once` (#2791).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TABLE = "operator_notice_audit"

# Matches the retention the fabricated signal_log registration used to carry,
# so the move to a dedicated table does not silently make this history
# unbounded. Swept by the `trash_retention` maintenance rail.
DEFAULT_RETENTION_DAYS = 14


class OperatorNoticeState(str, Enum):
    """Every state one operator notice can be in.

    ``COLLECTED`` and ``INJECTED`` are explicitly NOT delivery claims:

    - ``COLLECTED`` — the producer built the notice, drained the pending
      auto-mode queue, and advanced budget/governance dedupe state. Nothing
      has been sent.
    - ``INJECTED`` — the notice message was appended to the outbound message
      array for this turn. The provider has still not accepted it.
    - ``DELIVERED`` — the notice is beyond loss at its own transport's
      boundary (provider-accept for the ephemeral inline form, conversation
      persistence for the durable fallback form).
    - ``FAILED`` / ``CANCELLED`` — the turn died before that boundary. What
      left no durable trace is requeued by the caller's lifecycle.
    """

    COLLECTED = "collected"
    INJECTED = "injected"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: States a row can still be moved out of. A settled row is locked — the
#: first terminal write wins so a late "the turn failed" cannot overwrite an
#: honest "the provider already accepted this".
OPEN_STATES = frozenset(
    {OperatorNoticeState.COLLECTED, OperatorNoticeState.INJECTED}
)

TERMINAL_STATES = frozenset(
    {
        OperatorNoticeState.DELIVERED,
        OperatorNoticeState.FAILED,
        OperatorNoticeState.CANCELLED,
    }
)

_COLUMNS = (
    "id, notice_id, session_id, source, event_index, delivery_role, "
    "fallback, route, payload, state, state_reason, durable_trace, "
    "requeued, collected_at, injected_at, settled_at"
)


def _naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Normalize to naive UTC for backend portability.

    Mirrors :mod:`async_wait_signal_store`: Postgres ``TIMESTAMP`` columns
    reject tz-aware datetimes, while SQLite accepts datetime values via
    aiosqlite's adapter.
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _dump_payload(payload: Any) -> Optional[str]:
    if payload is None:
        return None
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return json.dumps({"unserializable": str(type(payload).__name__)})


@dataclass(frozen=True)
class OperatorNoticeRecord:
    """One ``operator_notice_audit`` row."""

    id: str
    notice_id: str
    session_id: Optional[str]
    source: str
    event_index: int
    delivery_role: str
    fallback: bool
    route: str
    payload: Dict[str, Any]
    state: str
    state_reason: str
    durable_trace: bool
    requeued: bool
    collected_at: Optional[str]
    injected_at: Optional[str]
    settled_at: Optional[str]

    @property
    def claims_delivery(self) -> bool:
        """Whether this row asserts the notice actually reached the model."""
        return self.state == OperatorNoticeState.DELIVERED.value

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "OperatorNoticeRecord":
        raw_payload = row[8]
        try:
            payload = json.loads(raw_payload) if raw_payload else {}
        except (TypeError, ValueError):
            payload = {}
        return cls(
            id=str(row[0]),
            notice_id=str(row[1]),
            session_id=str(row[2]) if row[2] is not None else None,
            source=str(row[3]),
            event_index=int(row[4] or 0),
            delivery_role=str(row[5] or ""),
            fallback=bool(row[6]),
            route=str(row[7] or ""),
            payload=payload if isinstance(payload, dict) else {},
            state=str(row[9]),
            state_reason=str(row[10] or ""),
            durable_trace=bool(row[11]),
            requeued=bool(row[12]),
            collected_at=str(row[13]) if row[13] is not None else None,
            injected_at=str(row[14]) if row[14] is not None else None,
            settled_at=str(row[15]) if row[15] is not None else None,
        )


class OperatorNoticeAuditStore:
    """Async CRUD over ``operator_notice_audit``.

    Construct with the agent's :class:`AsyncDatabase` and the owning agent's
    id (its DID). Every query is filtered by ``agent_id`` so a shared Postgres
    backend cannot leak rows between agents — the same isolation rule
    ``WaitSignalStore`` and ``PendingA2AQuestionStore`` follow.
    """

    def __init__(self, db: Any, agent_id: str):
        if not isinstance(agent_id, str):
            raise TypeError(
                "OperatorNoticeAuditStore agent_id must be a string, "
                f"got {type(agent_id).__name__}"
            )
        self._db = db
        self._agent_id = agent_id

    # ------------------------------------------------------------------
    # Write path — two phase, collect then settle
    # ------------------------------------------------------------------

    async def record_collected(
        self,
        *,
        notice_id: str,
        session_id: Optional[str],
        delivery_role: str,
        fallback: bool,
        route: str,
        events: Iterable[Tuple[str, Any]],
        retention_days: int = DEFAULT_RETENTION_DAYS,
        collected_at: Optional[datetime] = None,
    ) -> int:
        """Write the collect-phase rows. One row per source event.

        These rows exist precisely so that "collected, then vanished" is
        observable. They claim nothing beyond what has been observed at this
        point: the producer built these events and consumed the state that
        produced them.
        """
        now = _naive_utc(collected_at) or _now()
        retention_until = now + timedelta(days=int(retention_days))
        written = 0
        for index, (source, payload) in enumerate(events):
            await self._db.execute(
                f"""
                INSERT INTO {TABLE} (
                    id, notice_id, agent_id, session_id, source, event_index,
                    delivery_role, fallback, route, payload, state,
                    state_reason, durable_trace, requeued,
                    collected_at, retention_until
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 0, 0, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    notice_id,
                    self._agent_id,
                    session_id,
                    str(source),
                    index,
                    str(delivery_role),
                    1 if fallback else 0,
                    str(route),
                    _dump_payload(payload),
                    OperatorNoticeState.COLLECTED.value,
                    now,
                    retention_until,
                ),
            )
            written += 1
        return written

    async def mark_injected(
        self, notice_id: str, *, at: Optional[datetime] = None
    ) -> int:
        """``collected`` → ``injected``. Returns the number of rows moved.

        Gated on the current state so a row that already settled is never
        walked backwards, and so a duplicate call is a no-op rather than a
        second timestamp.
        """
        return await self._db.execute(
            f"""
            UPDATE {TABLE}
            SET state = ?, injected_at = ?
            WHERE agent_id = ? AND notice_id = ? AND state = ?
            """,
            (
                OperatorNoticeState.INJECTED.value,
                _naive_utc(at) or _now(),
                self._agent_id,
                notice_id,
                OperatorNoticeState.COLLECTED.value,
            ),
        )

    async def settle(
        self,
        notice_id: str,
        *,
        state: OperatorNoticeState,
        reason: str = "",
        durable_trace: bool = False,
        requeued: bool = False,
        at: Optional[datetime] = None,
    ) -> int:
        """Move an open notice to a terminal state. Returns rows updated.

        The ``state IN (open)`` predicate is the whole point: the first
        terminal write wins. A fallback notice that settled ``delivered`` when
        it was persisted must not be re-settled ``failed`` when the
        surrounding turn later dies — the notice is in the user's history
        either way.
        """
        if state not in TERMINAL_STATES:
            raise ValueError(
                f"settle() requires a terminal state, got {state!r}"
            )
        return await self._db.execute(
            f"""
            UPDATE {TABLE}
            SET state = ?, state_reason = ?, durable_trace = ?,
                requeued = ?, settled_at = ?
            WHERE agent_id = ? AND notice_id = ? AND state IN (?, ?)
            """,
            (
                state.value,
                reason or "",
                1 if durable_trace else 0,
                1 if requeued else 0,
                _naive_utc(at) or _now(),
                self._agent_id,
                notice_id,
                OperatorNoticeState.COLLECTED.value,
                OperatorNoticeState.INJECTED.value,
            ),
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def list_for_notice(self, notice_id: str) -> List[OperatorNoticeRecord]:
        rows = await self._db.fetchall(
            f"SELECT {_COLUMNS} FROM {TABLE} "
            f"WHERE agent_id = ? AND notice_id = ? ORDER BY event_index",
            (self._agent_id, notice_id),
        )
        return [OperatorNoticeRecord.from_row(row) for row in rows]

    async def list_unsettled(self) -> List[OperatorNoticeRecord]:
        """Notices collected or injected but never settled.

        The forensic query this table exists for: every row here is a notice
        the system consumed state for and then lost track of.
        """
        rows = await self._db.fetchall(
            f"SELECT {_COLUMNS} FROM {TABLE} "
            f"WHERE agent_id = ? AND state IN (?, ?) ORDER BY collected_at",
            (
                self._agent_id,
                OperatorNoticeState.COLLECTED.value,
                OperatorNoticeState.INJECTED.value,
            ),
        )
        return [OperatorNoticeRecord.from_row(row) for row in rows]

    async def list_recent(self, limit: int = 50) -> List[OperatorNoticeRecord]:
        rows = await self._db.fetchall(
            f"SELECT {_COLUMNS} FROM {TABLE} WHERE agent_id = ? "
            f"ORDER BY collected_at DESC, event_index ASC LIMIT ?",
            (self._agent_id, int(limit)),
        )
        return [OperatorNoticeRecord.from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    async def purge_expired(self, *, now: Optional[datetime] = None) -> int:
        """Delete this agent's rows past ``retention_until``.

        Unsettled rows are purged too: an unsettled notice from three weeks
        ago is a historical fact, not outstanding work, and nothing retries
        from this table.
        """
        cutoff = _naive_utc(now) or _now()
        return await self._db.execute(
            f"DELETE FROM {TABLE} WHERE agent_id = ? AND retention_until < ?",
            (self._agent_id, cutoff),
        )
