"""One durable obligation per A2A task input, the only decider of its wake (#4041).

Table ``a2a_wake_obligations`` holds one row per task input, keyed
``(agent_id, consumer_id, task_id, input_revision)``; the primary key alone
enforces that (I1). Its ``state`` alone says what the input still needs:

``owed``
    A wake must be admitted for ``generation``.
``bound``
    The wake ``live_delivery_id`` was admitted for it. Only that delivery may
    be claimed (I2).
``ran``
    The bound wake was acknowledged.
``conflict``
    More than one live legacy wake shares the input. None may run; operator
    work.
``ended`` / ``released``
    Settled: the task was ended because its bound wake ended undelivered, or
    the obligation was let go because the task no longer holds this input.

Every task transition into ``submitted`` arms the obligation in the task
write's own transaction (I3): ``owed``, ``generation + 1``, no live delivery.
Admission binds it by compare-and-set from ``owed`` at that generation, in
the transaction that inserts the delivery (I4). Acknowledging the bound
delivery marks it ``ran``; settlement ends or releases it in the task
mutation's transaction (I5). Delivery rows decide nothing, and no clock
orders anything: ``generation`` and these compare-and-set transitions do.

Every path that writes an obligation and a task or delivery row takes the
agent's ``a2a.task_submitted`` handoff lock first, then the obligation, then
the delivery, then the task.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Optional

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.source_handoff import lock_source_handoff

logger = logging.getLogger(__name__)

TABLE = "a2a_wake_obligations"
SOURCE = "a2a.task_submitted"
CONSUMER_ID = "core.a2a-task-submitted-cognition-v1"

OWED = "owed"
BOUND = "bound"
RAN = "ran"
CONFLICT = "conflict"
ENDED = "ended"
RELEASED = "released"
STATES = (OWED, BOUND, RAN, CONFLICT, ENDED, RELEASED)
SETTLED = frozenset({ENDED, RELEASED})

_SCHEMA_LOCK = "kestrel.a2a_wake_obligations.schema"
# Task IDs bound into one IN list of a lookup by task.
_TASK_ID_BATCH = 200
_COLUMNS = (
    "agent_id, consumer_id, task_id, input_revision, generation, state, "
    "live_delivery_id"
)


def wake_identity(task_id: str, input_revision: int, generation: int) -> str:
    """The durable source identity of one generation's wake for one input.

    It names the generation, never only the input, so a re-armed obligation's
    wake can never deduplicate against an earlier generation's event.
    """
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Wake identity requires a concrete task ID")
    if type(input_revision) is not int or input_revision < 0:
        raise ValueError("Wake identity requires a nonnegative input revision")
    if type(generation) is not int or generation < 1:
        raise ValueError("Wake identity requires a positive obligation generation")
    return json.dumps(
        {"generation": generation, "input_revision": input_revision, "task_id": task_id},
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_wake_identity(value: Any) -> Optional[tuple[str, int, int]]:
    """``(task_id, input_revision, generation)``, or None for another form.

    A JSON object that is not a canonical wake identity is refused rather
    than read as some other input.
    """
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if set(parsed) != {"generation", "input_revision", "task_id"}:
        raise ValueError("Invalid task input wake identity")
    canonical = wake_identity(
        parsed["task_id"], parsed["input_revision"], parsed["generation"]
    )
    if canonical != value:
        raise ValueError("Noncanonical task input wake identity")
    return parsed["task_id"], parsed["input_revision"], parsed["generation"]


def claimable_predicate(qualifier: str) -> str:
    """SQL: a delivery row of ``qualifier`` that its obligation lets run (I2).

    Deliveries of other consumers are unaffected. A wake of the obligation
    consumer is claimable only while an obligation is ``bound`` to it, so a
    conflicting, unbound, superseded or settled wake never runs.
    """
    return (
        f"({qualifier}.consumer_id <> '{CONSUMER_ID}' OR EXISTS ("
        f"SELECT 1 FROM {TABLE} obligation "
        f"WHERE obligation.agent_id = {qualifier}.agent_id "
        f"AND obligation.consumer_id = {qualifier}.consumer_id "
        f"AND obligation.live_delivery_id = {qualifier}.delivery_id "
        f"AND obligation.state = '{BOUND}'))"
    )


@dataclass(frozen=True)
class WakeObligation:
    """One stored obligation row."""

    agent_id: str
    consumer_id: str
    task_id: str
    input_revision: int
    generation: int
    state: str
    live_delivery_id: Optional[str]

    @property
    def identity(self) -> str:
        """The source identity of this generation's wake."""
        return wake_identity(self.task_id, self.input_revision, self.generation)


class WakeObligationRefused(RuntimeError):
    """Admission found the input's obligation not owed at this generation (I4).

    Raised inside the admission transaction, so nothing it wrote commits. The
    input has a wake already, has run, is settled, or was re-armed at a newer
    generation; the caller re-reads the obligation rather than retrying.
    """


NOT_OWED = "not_owed"


@dataclass(frozen=True)
class WakeTaskSettlement:
    """What a task settlement decided, in the settlement's own transaction."""

    disposition: str
    task: Any = None
    previous_state: Optional[str] = None


@dataclass(frozen=True)
class WakeSettlement:
    """The outcome of one settlement attempt.

    ``disposition`` is ``ended``, ``released`` or ``not_owed``. ``recorded``
    is True only when this attempt committed the settlement; a retry that
    finds it already committed returns its disposition with ``recorded``
    False, and projection follows the disposition either way.
    """

    disposition: str
    recorded: bool
    obligation: Optional["WakeObligation"] = None
    task: Any = None
    previous_state: Optional[str] = None


@dataclass(frozen=True)
class BackfillReport:
    """What one backfill pass created."""

    owed: int = 0
    bound: int = 0
    conflict: int = 0


class WakeObligationStore(UnifiedStoreBase):
    """Reads and compare-and-set writes of ``a2a_wake_obligations``.

    Borrows its caller's backend and never opens a transaction for a write:
    each write runs inside the caller's task, delivery or settlement
    transaction, which is the point of the table.
    """

    def __init__(self, backend: Any):
        super().__init__(backend, owns_backend=False)

    async def ensure_schema(self) -> None:
        """Create the table and its indexes; additive and idempotent."""
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        states = ", ".join(f"'{state}'" for state in STATES)
        async with self._backend.transaction():
            if self.is_postgres:
                await self._backend.fetch_val(
                    "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                    (_SCHEMA_LOCK,),
                )
            await self._backend.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    agent_id TEXT NOT NULL,
                    consumer_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    input_revision BIGINT NOT NULL CHECK (input_revision >= 0),
                    generation BIGINT NOT NULL CHECK (generation >= 1),
                    state TEXT NOT NULL CHECK (state IN ({states})),
                    live_delivery_id TEXT,
                    updated_at {ts_type} {ts_default},
                    PRIMARY KEY (agent_id, consumer_id, task_id, input_revision),
                    CHECK (state <> '{BOUND}' OR live_delivery_id IS NOT NULL),
                    CHECK (state NOT IN ('{OWED}', '{CONFLICT}') OR live_delivery_id IS NULL)
                )
                """
            )
            await self._backend.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE}_live "
                f"ON {TABLE} (agent_id, consumer_id, live_delivery_id) "
                "WHERE live_delivery_id IS NOT NULL"
            )
            await self._backend.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_state "
                f"ON {TABLE} (agent_id, consumer_id, state)"
            )

    async def lock_scope(self, *, agent_id: str) -> None:
        """Take the agent's wake handoff lock, first in every writer's order."""
        await lock_source_handoff(
            self._backend, agent_id=agent_id, source=SOURCE,
            sqlite_reserve_table=TABLE,
        )

    def _require_owner_transaction(self, operation: str) -> None:
        if not self._backend.owns_open_transaction:
            raise RuntimeError(
                f"Wake obligation {operation} runs in its caller's transaction"
            )

    @staticmethod
    def _row(row: Any) -> WakeObligation:
        return WakeObligation(
            agent_id=str(row[0]),
            consumer_id=str(row[1]),
            task_id=str(row[2]),
            input_revision=int(row[3]),
            generation=int(row[4]),
            state=str(row[5]),
            live_delivery_id=None if row[6] is None else str(row[6]),
        )

    async def get(
        self,
        *,
        agent_id: str,
        task_id: str,
        input_revision: int,
        for_update: bool = False,
    ) -> Optional[WakeObligation]:
        """One input's obligation, locked when ``for_update``."""
        suffix = " FOR UPDATE" if for_update and self.is_postgres else ""
        row = await self._backend.fetch_one(
            f"SELECT {_COLUMNS} FROM {TABLE} "
            "WHERE agent_id = ? AND consumer_id = ? AND task_id = ? "
            f"AND input_revision = ?{suffix}",
            (agent_id, CONSUMER_ID, task_id, input_revision),
        )
        return None if row is None else self._row(row)

    async def get_bound_to(
        self, *, agent_id: str, consumer_id: str, delivery_id: str,
        for_update: bool = False,
    ) -> Optional[WakeObligation]:
        """The obligation whose live delivery is ``delivery_id``, if any."""
        suffix = " FOR UPDATE" if for_update and self.is_postgres else ""
        row = await self._backend.fetch_one(
            f"SELECT {_COLUMNS} FROM {TABLE} "
            "WHERE agent_id = ? AND consumer_id = ? "
            f"AND live_delivery_id = ?{suffix}",
            (agent_id, consumer_id, delivery_id),
        )
        return None if row is None else self._row(row)

    async def list_by_state(
        self, *, agent_id: str, state: str, limit: int = 500,
        after: Optional[tuple[str, int]] = None,
    ) -> list[WakeObligation]:
        """One page of an agent's obligations in one state, by key.

        ``after`` is the last key of the previous page.
        """
        if state not in STATES:
            raise ValueError(f"Unknown wake obligation state {state!r}")
        keyset, values = "", ()
        if after is not None:
            keyset = " AND (task_id > ? OR (task_id = ? AND input_revision > ?))"
            values = (after[0], after[0], after[1])
        rows = await self._backend.fetch_all(
            f"SELECT {_COLUMNS} FROM {TABLE} "
            f"WHERE agent_id = ? AND consumer_id = ? AND state = ?{keyset} "
            "ORDER BY task_id, input_revision LIMIT ?",
            (agent_id, CONSUMER_ID, state, *values, limit),
        )
        return [self._row(row) for row in rows]

    async def task_ids_in_state(
        self, *, agent_id: str, state: str, task_ids: Iterable[str]
    ) -> set[str]:
        """Which of ``task_ids`` hold one of this agent's obligations in ``state``."""
        if state not in STATES:
            raise ValueError(f"Unknown wake obligation state {state!r}")
        ids = sorted(set(task_ids))
        found: set[str] = set()
        for start in range(0, len(ids), _TASK_ID_BATCH):
            batch = ids[start:start + _TASK_ID_BATCH]
            placeholders = ", ".join("?" for _ in batch)
            rows = await self._backend.fetch_all(
                f"SELECT DISTINCT task_id FROM {TABLE} "
                "WHERE agent_id = ? AND consumer_id = ? AND state = ? "
                f"AND task_id IN ({placeholders})",
                (agent_id, CONSUMER_ID, state, *batch),
            )
            found.update(str(row[0]) for row in rows)
        return found

    async def count_by_state(self, *, agent_id: str, state: str) -> int:
        """How many of one agent's obligations are in ``state``."""
        if state not in STATES:
            raise ValueError(f"Unknown wake obligation state {state!r}")
        return int(
            await self._backend.fetch_val(
                f"SELECT COUNT(*) FROM {TABLE} "
                "WHERE agent_id = ? AND consumer_id = ? AND state = ?",
                (agent_id, CONSUMER_ID, state),
            )
            or 0
        )

    async def arm(self, *, agent_id: str, task_id: str, input_revision: int) -> int:
        """I3: owe a fresh-generation wake for one input; return the generation.

        Runs in the transaction of the task write into ``submitted``, after
        :meth:`lock_scope`. Whatever the obligation held, it is owed again at
        the next generation with no live delivery, so any earlier wake of the
        input can no longer be claimed or recorded as having run.
        """
        self._require_owner_transaction("arming")
        now = self.now_sql()
        generation = await self._backend.fetch_val(
            f"""
            INSERT INTO {TABLE} (
                agent_id, consumer_id, task_id, input_revision, generation,
                state, live_delivery_id, updated_at
            ) VALUES (?, ?, ?, ?, 1, '{OWED}', NULL, {now})
            ON CONFLICT (agent_id, consumer_id, task_id, input_revision)
            DO UPDATE SET generation = {TABLE}.generation + 1,
                state = '{OWED}', live_delivery_id = NULL, updated_at = {now}
            RETURNING generation
            """,
            (agent_id, CONSUMER_ID, task_id, input_revision),
        )
        if generation is None:
            raise RuntimeError("Arming a wake obligation returned no generation")
        return int(generation)

    async def bind(
        self, *, agent_id: str, task_id: str, input_revision: int,
        generation: int, delivery_id: str,
    ) -> bool:
        """I4: bind an owed obligation to the delivery admitted for it."""
        self._require_owner_transaction("binding")
        updated = await self._backend.execute(
            f"""
            UPDATE {TABLE}
            SET state = '{BOUND}', live_delivery_id = ?, updated_at = {self.now_sql()}
            WHERE agent_id = ? AND consumer_id = ? AND task_id = ?
              AND input_revision = ? AND state = '{OWED}' AND generation = ?
            """,
            (delivery_id, agent_id, CONSUMER_ID, task_id, input_revision, generation),
        )
        return updated == 1

    async def mark_ran(
        self, *, agent_id: str, consumer_id: str, delivery_id: str
    ) -> bool:
        """I5: the bound delivery was acknowledged; its input has run."""
        self._require_owner_transaction("acknowledgement")
        updated = await self._backend.execute(
            f"""
            UPDATE {TABLE}
            SET state = '{RAN}', updated_at = {self.now_sql()}
            WHERE agent_id = ? AND consumer_id = ? AND live_delivery_id = ?
              AND state = '{BOUND}'
            """,
            (agent_id, consumer_id, delivery_id),
        )
        return updated == 1

    async def settle(self, obligation: WakeObligation, *, disposition: str) -> None:
        """I5: record a settlement decided in the caller's transaction.

        ``obligation`` is the row as locked there, so exactly one row moves;
        anything else means the lock order was broken, and the caller's
        transaction is rolled back with its task write.
        """
        self._require_owner_transaction("settlement")
        if disposition not in SETTLED:
            raise ValueError(f"Settlement disposition must be one of {sorted(SETTLED)}")
        updated = await self._backend.execute(
            f"""
            UPDATE {TABLE}
            SET state = ?, updated_at = {self.now_sql()}
            WHERE agent_id = ? AND consumer_id = ? AND task_id = ?
              AND input_revision = ? AND state = ? AND generation = ?
            """,
            (
                disposition, obligation.agent_id, obligation.consumer_id,
                obligation.task_id, obligation.input_revision, obligation.state,
                obligation.generation,
            ),
        )
        if updated != 1:
            raise RuntimeError(
                "A locked wake obligation changed under its settlement"
            )

    async def insert_backfilled(
        self, *, agent_id: str, task_id: str, input_revision: int,
        state: str, live_delivery_id: Optional[str],
    ) -> bool:
        """Insert one backfilled obligation unless the input already has one."""
        if state not in (OWED, BOUND, CONFLICT):
            raise ValueError("Backfill derives only owed, bound or conflict")
        inserted = await self._backend.execute(
            f"""
            INSERT INTO {TABLE} (
                agent_id, consumer_id, task_id, input_revision, generation,
                state, live_delivery_id, updated_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?, {self.now_sql()})
            ON CONFLICT (agent_id, consumer_id, task_id, input_revision) DO NOTHING
            """,
            (agent_id, CONSUMER_ID, task_id, input_revision, state, live_delivery_id),
        )
        return inserted == 1


__all__ = [
    "BOUND",
    "BackfillReport",
    "CONFLICT",
    "CONSUMER_ID",
    "ENDED",
    "NOT_OWED",
    "OWED",
    "RAN",
    "RELEASED",
    "SETTLED",
    "SOURCE",
    "STATES",
    "TABLE",
    "WakeObligation",
    "WakeObligationRefused",
    "WakeObligationStore",
    "WakeSettlement",
    "WakeTaskSettlement",
    "claimable_predicate",
    "parse_wake_identity",
    "wake_identity",
]
