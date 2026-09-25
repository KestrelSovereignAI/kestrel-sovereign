"""Fleet circuit breaker for repeated peer Stop (#3170).

Any authenticated peer may pull the andon cord (#3169), and one peer is
bounded by the ``a2a.peer_stop`` rate limit.  Many peers -- or one peer over
time -- can still stop an agent so often that it never completes a turn, which
is Hold granted through the back door.  This breaker closes that door: once
the honored peer Stops against one target reach ``threshold`` inside a sliding
``window``, further peer Stops against that target are refused and a human is
told.

What counts, and why it cannot be reset by a restart or spoofed:

* The unit is an **admission**: the breaker's durable decision to honor one
  peer Stop *operation*, written in the same transaction that counted the
  window, under a per-target lock (a PostgreSQL advisory lock; SQLite's
  ``BEGIN IMMEDIATE`` writer slot).  Two peers that both see ``threshold - 1``
  concurrently cannot both be honored.
* The Stop receipt decides finality.  An admission whose operation's receipt
  records no ``stopped`` outcome (the agent was idle, or the Stop was refused)
  stops counting; an admission with no receipt yet -- a Stop still in flight --
  counts, so a burst cannot outrun its own receipts.
* The actor is the verified signal principal and the target is the recipient's
  own DID; neither comes from payload.  Rows live in the host's Stop evidence
  database, so a worker restart changes nothing.

Only the peer rail admits.  Operator and local sovereign Stops (the
``/api/agent/stop`` and ``/api/host/stop`` doors) are never counted and never
refused here.

Transitions are receipted in ``stop_circuit_events``: ``opened`` (the count
reached the threshold), ``closed`` (automatic recovery: the count fell below
the threshold inside the window), and ``reset`` (the sovereign door; its actor
is recorded).  Nothing latches: an open circuit closes by itself as the window
slides past its admissions, and the breaker never writes a Hold.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from kestrel_sovereign.storage.database_clock import (
    database_backend_type,
    database_clock,
    database_timestamp_bound_text,
)

from .receipt import opaque_stop_identifier
from .types import StopRequest, StopScope

logger = logging.getLogger(__name__)

_SCHEMA_LOCK = "stop_circuit_v1"
_LOCK_PREFIX = "kestrel:stop:circuit:"
MAX_CIRCUIT_TARGET_LENGTH = 512
MAX_CIRCUIT_REASON_LENGTH = 1024
MAX_CIRCUIT_EVENT_PAGE = 200
_PEER_SCOPES = frozenset({StopScope.AGENT, StopScope.TURN})
_EVENT_COLUMNS = (
    "event_id, target_agent_id, kind, epoch, admitted_count, threshold, "
    "window_seconds, actor_id, reason, occurred_at"
)

Clock = Callable[[Any], Awaitable[datetime]]


class PeerStopCircuitError(RuntimeError):
    """The circuit's durable state could not be read or written."""


class PeerStopCircuitEventKind(str, Enum):
    OPENED = "opened"
    CLOSED = "closed"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class PeerStopCircuitPolicy:
    """How many honored peer Stops against one agent a window tolerates."""

    threshold: int
    window_seconds: int

    def __post_init__(self) -> None:
        for name, value in (
            ("threshold", self.threshold),
            ("window_seconds", self.window_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"peer Stop circuit {name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PeerStopCircuitEvent:
    """One receipted circuit transition."""

    event_id: str
    target_agent_id: str
    kind: PeerStopCircuitEventKind
    epoch: int
    admitted_count: int
    threshold: int
    window_seconds: int
    actor_id: str | None
    reason: str | None
    occurred_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "target_agent_id": self.target_agent_id,
            "kind": self.kind.value,
            "admitted_count": self.admitted_count,
            "threshold": self.threshold,
            "window_seconds": self.window_seconds,
            "actor_id": self.actor_id,
            "reason": self.reason,
            "occurred_at": self.occurred_at,
        }


@dataclass(frozen=True, slots=True)
class PeerStopCircuitDecision:
    """The breaker's verdict on one peer Stop operation."""

    honored: bool
    admitted_count: int
    threshold: int
    window_seconds: int
    opened: PeerStopCircuitEvent | None = None
    closed: PeerStopCircuitEvent | None = None


@dataclass(frozen=True, slots=True)
class PeerStopCircuit:
    """One open circuit as the sovereign status surface renders it."""

    target_agent_id: str
    opened_at: str
    opened_event_id: str
    admitted_count: int
    threshold: int
    window_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_agent_id": self.target_agent_id,
            "opened_at": self.opened_at,
            "opened_event_id": self.opened_event_id,
            "admitted_count": self.admitted_count,
            "threshold": self.threshold,
            "window_seconds": self.window_seconds,
        }


@dataclass(slots=True)
class _State:
    epoch: int
    is_open: bool
    opened_at: str | None
    opened_event_id: str | None
    exists: bool


def _required_text(value: object, field: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"peer Stop circuit {field} must be a concrete string")
    if len(value) > max_length:
        raise ValueError(f"peer Stop circuit {field} is too long")
    return value


class PeerStopCircuitStore:
    """Durable, per-target admission accounting for peer Stop."""

    def __init__(
        self,
        db: Any,
        *,
        policy: PeerStopCircuitPolicy,
        clock: Clock = database_clock,
    ) -> None:
        if not isinstance(policy, PeerStopCircuitPolicy):
            raise TypeError("policy must be a PeerStopCircuitPolicy")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._db = db
        self._policy = policy
        self._clock = clock

    @property
    def policy(self) -> PeerStopCircuitPolicy:
        return self._policy

    async def ensure_schema(self) -> None:
        """Create the circuit tables beside the Stop receipts they join."""

        if not await self._db.table_exists("stop_receipts"):
            raise PeerStopCircuitError(
                "peer Stop circuit requires the Stop receipt schema first"
            )
        async with self._db.migration_lock(_SCHEMA_LOCK):
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_circuit_state ("
                "target_agent_id TEXT NOT NULL PRIMARY KEY, "
                "epoch INTEGER NOT NULL, "
                "is_open INTEGER NOT NULL, "
                "opened_at TEXT, "
                "opened_event_id TEXT, "
                "CHECK (epoch >= 0), "
                "CHECK (is_open IN (0, 1)))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_circuit_admissions ("
                "target_agent_id TEXT NOT NULL, "
                "operation_id TEXT NOT NULL, "
                "epoch INTEGER NOT NULL, "
                "actor_id TEXT NOT NULL, "
                "admitted_at TEXT NOT NULL, "
                "PRIMARY KEY (target_agent_id, operation_id))"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_circuit_admissions_window "
                "ON stop_circuit_admissions(target_agent_id, epoch, admitted_at)"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_circuit_events ("
                "event_id TEXT NOT NULL PRIMARY KEY, "
                "target_agent_id TEXT NOT NULL, "
                "seq INTEGER NOT NULL, "
                "kind TEXT NOT NULL, "
                "epoch INTEGER NOT NULL, "
                "admitted_count INTEGER NOT NULL, "
                "threshold INTEGER NOT NULL, "
                "window_seconds INTEGER NOT NULL, "
                "actor_id TEXT, "
                "reason TEXT, "
                "occurred_at TEXT NOT NULL, "
                "UNIQUE (target_agent_id, seq), "
                "CHECK (seq >= 1), "
                "CHECK (kind IN ('opened', 'closed', 'reset')), "
                "CHECK (admitted_count >= 0))"
            )

    # ------------------------------------------------------------------
    # Admission (the peer rail's one question)
    # ------------------------------------------------------------------

    async def admit(self, request: StopRequest) -> PeerStopCircuitDecision:
        """Honor or refuse one peer Stop operation, durably and atomically.

        An operation already admitted in the current epoch's window is honored
        again without being counted twice: a retry of an interrupted Stop is
        the same Stop.  An older admission (before a reset, or slid out of the
        window) is no longer counted, so its retry faces the current threshold
        like any new Stop and, if honored, is re-admitted into this window.
        """

        if not isinstance(request, StopRequest):
            raise TypeError("peer Stop circuit admits a StopRequest")
        if request.scope not in _PEER_SCOPES:
            raise ValueError("peer Stop circuit admits only agent or turn scope")
        target = _required_text(
            request.target_agent_id, "target", MAX_CIRCUIT_TARGET_LENGTH
        )
        actor_id = _required_text(request.actor_id, "actor", 4096)
        operation_id = opaque_stop_identifier("operation", request.correlation_id)
        policy = self._policy
        try:
            async with self._db.transaction(immediate=True):
                await self._lock_target(target)
                now = await self._clock(self._db)
                state = await self._state(target)
                count = await self._count(target, state.epoch, now)
                closed = None
                if state.is_open and count < policy.threshold:
                    closed = await self._close(target, state, count, now)

                existing = await self._db.fetchone(
                    "SELECT 1 FROM stop_circuit_admissions "
                    "WHERE target_agent_id = ? AND operation_id = ?",
                    (target, operation_id),
                )
                if existing is not None and await self._admission_is_current(
                    target, operation_id, state.epoch, now
                ):
                    # Already counted in this epoch's window: the retry is the
                    # same Stop, honored without being counted twice.
                    await self._save_state(target, state)
                    return PeerStopCircuitDecision(
                        honored=True,
                        admitted_count=count,
                        threshold=policy.threshold,
                        window_seconds=policy.window_seconds,
                        closed=closed,
                    )

                opened = None
                if count >= policy.threshold:
                    if not state.is_open:
                        opened = await self._open(target, state, count, now)
                    await self._save_state(target, state)
                    decision = PeerStopCircuitDecision(
                        honored=False,
                        admitted_count=count,
                        threshold=policy.threshold,
                        window_seconds=policy.window_seconds,
                        opened=opened,
                        closed=closed,
                    )
                else:
                    if existing is None:
                        await self._db.execute(
                            "INSERT INTO stop_circuit_admissions ("
                            "target_agent_id, operation_id, epoch, actor_id, "
                            "admitted_at) VALUES (?, ?, ?, ?, ?)",
                            (
                                target,
                                operation_id,
                                state.epoch,
                                actor_id,
                                self._timestamp(now),
                            ),
                        )
                    else:
                        # An admission from an earlier epoch or outside the
                        # window no longer counts, so it confers nothing: the
                        # retry faced the current threshold above and, being
                        # honored, is re-admitted into the current window.
                        await self._db.execute(
                            "UPDATE stop_circuit_admissions SET epoch = ?, "
                            "actor_id = ?, admitted_at = ? "
                            "WHERE target_agent_id = ? AND operation_id = ?",
                            (
                                state.epoch,
                                actor_id,
                                self._timestamp(now),
                                target,
                                operation_id,
                            ),
                        )
                    count += 1
                    if count >= policy.threshold and not state.is_open:
                        opened = await self._open(target, state, count, now)
                    await self._save_state(target, state)
                    decision = PeerStopCircuitDecision(
                        honored=True,
                        admitted_count=count,
                        threshold=policy.threshold,
                        window_seconds=policy.window_seconds,
                        opened=opened,
                        closed=closed,
                    )
        except Exception as error:
            raise PeerStopCircuitError(
                "peer Stop circuit state could not be recorded"
            ) from error
        if decision.opened is not None:
            logger.warning(
                "Peer Stop circuit OPEN for %s: %d honored peer Stops within "
                "%ds reached the threshold of %d; further peer Stops against "
                "this agent are refused until the window clears or the "
                "sovereign resets it (POST /api/host/stop/circuit/reset)",
                target,
                decision.opened.admitted_count,
                policy.window_seconds,
                policy.threshold,
            )
        if decision.closed is not None:
            logger.info("Peer Stop circuit closed for %s (window cleared)", target)
        return decision

    # ------------------------------------------------------------------
    # Sovereign surfaces
    # ------------------------------------------------------------------

    async def open_circuits(self) -> tuple[PeerStopCircuit, ...]:
        """Every circuit still open, receipting any that recovered by time.

        An open circuit closes when its window slides past enough admissions,
        which nobody observes as it happens.  The first read afterwards
        records the ``closed`` event, so the history converges on what the
        status surface reports rather than showing a circuit that is open in
        the table but closed in fact.
        """

        try:
            rows = await self._db.fetchall(
                "SELECT target_agent_id FROM stop_circuit_state "
                "WHERE is_open = 1 ORDER BY target_agent_id",
                (),
            )
            targets = [row[0] for row in rows]
            if not targets:
                return ()
            circuits: list[PeerStopCircuit] = []
            closed_targets: list[str] = []
            async with self._db.transaction(immediate=True):
                # One ordered pass, so two readers cannot deadlock.
                for target in targets:
                    await self._lock_target(target)
                now = await self._clock(self._db)
                for target in targets:
                    state = await self._state(target)
                    if not state.is_open:
                        continue
                    count = await self._count(target, state.epoch, now)
                    if count < self._policy.threshold:
                        await self._close(target, state, count, now)
                        await self._save_state(target, state)
                        closed_targets.append(target)
                        continue
                    circuits.append(
                        PeerStopCircuit(
                            target_agent_id=target,
                            opened_at=state.opened_at or "",
                            opened_event_id=state.opened_event_id or "",
                            admitted_count=count,
                            threshold=self._policy.threshold,
                            window_seconds=self._policy.window_seconds,
                        )
                    )
        except Exception as error:
            raise PeerStopCircuitError(
                "peer Stop circuit state could not be read"
            ) from error
        for target in closed_targets:
            logger.info("Peer Stop circuit closed for %s (window cleared)", target)
        return tuple(circuits)

    async def reset(
        self,
        target_agent_id: str,
        *,
        actor_id: str,
        reason: str,
    ) -> PeerStopCircuitEvent:
        """Close a target's circuit and discard its count; sovereign-only.

        The reset begins a new counting epoch: admissions before it no longer
        count, and a receipted ``reset`` event names who did it and why.
        """

        target = _required_text(target_agent_id, "target", MAX_CIRCUIT_TARGET_LENGTH)
        actor_id = _required_text(actor_id, "actor", 4096)
        reason = _required_text(reason, "reason", MAX_CIRCUIT_REASON_LENGTH)
        try:
            async with self._db.transaction(immediate=True):
                await self._lock_target(target)
                now = await self._clock(self._db)
                state = await self._state(target)
                count = await self._count(target, state.epoch, now)
                state.epoch += 1
                state.is_open = False
                state.opened_at = None
                state.opened_event_id = None
                event = await self._event(
                    target,
                    PeerStopCircuitEventKind.RESET,
                    state.epoch,
                    count,
                    now,
                    actor_id=actor_id,
                    reason=reason,
                )
                await self._save_state(target, state)
        except Exception as error:
            raise PeerStopCircuitError(
                "peer Stop circuit reset could not be recorded"
            ) from error
        logger.warning(
            "Peer Stop circuit reset for %s by %s (%d honored peer Stops discarded)",
            target,
            actor_id,
            count,
        )
        return event

    async def list_events(
        self,
        *,
        target_agent_id: str | None = None,
        limit: int = 50,
    ) -> tuple[PeerStopCircuitEvent, ...]:
        """Receipted circuit transitions, newest first."""

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_CIRCUIT_EVENT_PAGE
        ):
            raise ValueError("peer Stop circuit event page size is out of range")
        filters = ""
        params: list[Any] = []
        if target_agent_id is not None:
            filters = "WHERE target_agent_id = ? "
            params.append(
                _required_text(
                    target_agent_id, "target", MAX_CIRCUIT_TARGET_LENGTH
                )
            )
        params.append(limit)
        try:
            rows = await self._db.fetchall(
                f"SELECT {_EVENT_COLUMNS} FROM stop_circuit_events "
                f"{filters}ORDER BY occurred_at DESC, seq DESC LIMIT ?",
                tuple(params),
            )
            return tuple(self._event_from_row(row) for row in rows)
        except Exception as error:
            raise PeerStopCircuitError(
                "peer Stop circuit history could not be read"
            ) from error

    # ------------------------------------------------------------------
    # Internals -- every caller holds the target's lock in one transaction
    # ------------------------------------------------------------------

    async def _lock_target(self, target: str) -> None:
        if database_backend_type(self._db) != "postgres":
            # SQLite: ``BEGIN IMMEDIATE`` already holds the single writer slot.
            return
        await self._db.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (f"{_LOCK_PREFIX}{opaque_stop_identifier('circuit_target', target)}",),
        )

    def _timestamp(self, value: datetime) -> str:
        return database_timestamp_bound_text(self._db, value)

    async def _state(self, target: str) -> _State:
        row = await self._db.fetchone(
            "SELECT epoch, is_open, opened_at, opened_event_id "
            "FROM stop_circuit_state WHERE target_agent_id = ?",
            (target,),
        )
        if row is None:
            return _State(0, False, None, None, exists=False)
        return _State(
            epoch=int(row[0]),
            is_open=int(row[1]) == 1,
            opened_at=row[2],
            opened_event_id=row[3],
            exists=True,
        )

    async def _save_state(self, target: str, state: _State) -> None:
        params = (
            state.epoch,
            int(state.is_open),
            state.opened_at,
            state.opened_event_id,
            target,
        )
        if state.exists:
            await self._db.execute(
                "UPDATE stop_circuit_state SET epoch = ?, is_open = ?, "
                "opened_at = ?, opened_event_id = ? WHERE target_agent_id = ?",
                params,
            )
            return
        await self._db.execute(
            "INSERT INTO stop_circuit_state ("
            "epoch, is_open, opened_at, opened_event_id, target_agent_id"
            ") VALUES (?, ?, ?, ?, ?)",
            params,
        )
        state.exists = True

    def _window_start(self, now: datetime) -> str:
        return self._timestamp(now - timedelta(seconds=self._policy.window_seconds))

    async def _admission_is_current(
        self, target: str, operation_id: str, epoch: int, now: datetime
    ) -> bool:
        """Whether an operation's admission sits in this epoch's window.

        Only such an admission is part of the count ``admit`` just compared
        with the threshold.  An older one -- before a reset, or slid out of
        the window -- must not let its retry bypass a circuit opened since.
        """

        row = await self._db.fetchone(
            "SELECT 1 FROM stop_circuit_admissions "
            "WHERE target_agent_id = ? AND operation_id = ? "
            "AND epoch = ? AND admitted_at > ?",
            (target, operation_id, epoch, self._window_start(now)),
        )
        return row is not None

    async def _count(self, target: str, epoch: int, now: datetime) -> int:
        """Honored peer Stops against ``target`` inside the sliding window.

        An admission counts until its operation's Stop receipt proves it
        stopped nothing; a Stop still in flight counts.
        """

        window_start = self._window_start(now)
        value = await self._db.fetchval(
            "SELECT COUNT(*) FROM stop_circuit_admissions AS admission "
            "WHERE admission.target_agent_id = ? "
            "AND admission.epoch = ? "
            "AND admission.admitted_at > ? "
            "AND NOT EXISTS ("
            "SELECT 1 FROM stop_receipts AS receipt "
            "WHERE receipt.operation_id = admission.operation_id "
            "AND NOT EXISTS ("
            "SELECT 1 FROM stop_receipt_outcomes AS outcome "
            "WHERE outcome.receipt_id = receipt.receipt_id "
            "AND outcome.disposition = 'stopped'))",
            (target, epoch, window_start),
        )
        return int(value or 0)

    async def _open(
        self, target: str, state: _State, count: int, now: datetime
    ) -> PeerStopCircuitEvent:
        event = await self._event(
            target, PeerStopCircuitEventKind.OPENED, state.epoch, count, now
        )
        state.is_open = True
        state.opened_at = event.occurred_at
        state.opened_event_id = event.event_id
        return event

    async def _close(
        self, target: str, state: _State, count: int, now: datetime
    ) -> PeerStopCircuitEvent:
        event = await self._event(
            target, PeerStopCircuitEventKind.CLOSED, state.epoch, count, now
        )
        state.is_open = False
        state.opened_at = None
        state.opened_event_id = None
        return event

    async def _event(
        self,
        target: str,
        kind: PeerStopCircuitEventKind,
        epoch: int,
        count: int,
        now: datetime,
        *,
        actor_id: str | None = None,
        reason: str | None = None,
    ) -> PeerStopCircuitEvent:
        event = PeerStopCircuitEvent(
            event_id=str(uuid4()),
            target_agent_id=target,
            kind=kind,
            epoch=epoch,
            admitted_count=count,
            threshold=self._policy.threshold,
            window_seconds=self._policy.window_seconds,
            actor_id=actor_id,
            reason=reason,
            occurred_at=self._timestamp(now),
        )
        # Per-target order, allocated under the target's lock: two transitions
        # recorded at one clock reading still read back in the order they
        # happened.
        seq = int(
            await self._db.fetchval(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM stop_circuit_events "
                "WHERE target_agent_id = ?",
                (target,),
            )
        )
        await self._db.execute(
            f"INSERT INTO stop_circuit_events (seq, {_EVENT_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                seq,
                event.event_id,
                event.target_agent_id,
                event.kind.value,
                event.epoch,
                event.admitted_count,
                event.threshold,
                event.window_seconds,
                event.actor_id,
                event.reason,
                event.occurred_at,
            ),
        )
        return event

    @staticmethod
    def _event_from_row(row: Any) -> PeerStopCircuitEvent:
        if row is None or len(row) != 10:
            raise PeerStopCircuitError("peer Stop circuit event has an unexpected shape")
        return PeerStopCircuitEvent(
            event_id=str(row[0]),
            target_agent_id=str(row[1]),
            kind=PeerStopCircuitEventKind(row[2]),
            epoch=int(row[3]),
            admitted_count=int(row[4]),
            threshold=int(row[5]),
            window_seconds=int(row[6]),
            actor_id=row[7],
            reason=row[8],
            occurred_at=str(row[9]),
        )


__all__ = [
    "MAX_CIRCUIT_REASON_LENGTH",
    "MAX_CIRCUIT_TARGET_LENGTH",
    "PeerStopCircuit",
    "PeerStopCircuitDecision",
    "PeerStopCircuitError",
    "PeerStopCircuitEvent",
    "PeerStopCircuitEventKind",
    "PeerStopCircuitPolicy",
    "PeerStopCircuitStore",
]
