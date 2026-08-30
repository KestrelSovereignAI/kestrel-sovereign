"""Distributed ownership and cooperative cancellation for live invocations.

Stop receipts prove what an operation reported.  This module supplies the
missing live-work authority in horizontally scaled deployments: every process
registers its active generations in the shared Stop database, and the process
that receives Stop marks those rows for their owning process to cancel.

Only domain-separated digests of turn IDs are durable.  The raw ID required to
address an in-process task remains in the owning process's memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    raise_owned_outcome,
)
from kestrel_sovereign.storage.database_clock import (
    database_backend_type,
    database_now_sql,
)

from .receipt import StopReceiptStore, opaque_stop_identifier
from .types import StopDisposition

_SCHEMA_LOCK = "stop_invocations_v1"
_TURN_ID_DOMAIN = b"kestrel:distributed-stop-turn:v1\0"
_DEFAULT_POLL_SECONDS = 0.1
_DEFAULT_WAIT_SECONDS = 4.0
_DEFAULT_OWNER_LEASE_SECONDS = 2.0
logger = logging.getLogger(__name__)


def _turn_digest(turn_id: str) -> str:
    encoded = turn_id.encode("utf-8")
    return (
        "sha256:"
        + hashlib.sha256(
            _TURN_ID_DOMAIN + len(encoded).to_bytes(4, "big") + encoded
        ).hexdigest()
    )


def _required_identity(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"distributed Stop {field} must be concrete")
    return value


@dataclass(frozen=True, slots=True)
class DistributedStopTicket:
    """The exact durable generations selected at Stop linearization."""

    generation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _OwnerPoll:
    """One non-revivable heartbeat result for an invocation owner."""

    live_generation_ids: tuple[str, ...]
    stop_generation_ids: tuple[str, ...]


def _lease_cutoff_sql(db: Any, lease_seconds: float) -> tuple[str, tuple[object, ...]]:
    backend_type = database_backend_type(db)
    if backend_type == "postgres":
        return (
            "(to_char((clock_timestamp() - (? * INTERVAL '1 second')) "
            "AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US') || '+00:00')",
            (lease_seconds,),
        )
    if backend_type == "sqlite":
        return (
            "strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now', ?)",
            (f"-{lease_seconds} seconds",),
        )
    raise RuntimeError("distributed Stop lease clock is unavailable")


class DistributedInvocationStore:
    """Portable SQL authority for active invocation ownership."""

    def __init__(self, db: Any):
        self._db = db

    async def ensure_schema(self) -> None:
        # Admission consults acknowledged receipt evidence as well as the
        # in-progress fence, so the paired schema is part of this authority's
        # readiness contract even when constructed outside the server.
        await StopReceiptStore(self._db).ensure_schema()
        async with self._db.migration_lock(_SCHEMA_LOCK):
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_active_invocations ("
                "generation_id TEXT NOT NULL PRIMARY KEY, "
                "agent_id TEXT NOT NULL, "
                "turn_digest TEXT NOT NULL, "
                "owner_id TEXT NOT NULL, "
                "stop_requested INTEGER NOT NULL DEFAULT 0, "
                "registered_at TEXT NOT NULL, "
                "heartbeat_at TEXT NOT NULL, "
                "CHECK (stop_requested IN (0, 1)))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_invocation_fences ("
                "agent_id TEXT NOT NULL, "
                "turn_digest TEXT NOT NULL, "
                "created_at TEXT NOT NULL, "
                "PRIMARY KEY (agent_id, turn_digest))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_unresolved_invocations ("
                "generation_id TEXT NOT NULL PRIMARY KEY, "
                "agent_id TEXT NOT NULL, "
                "turn_digest TEXT NOT NULL, "
                "owner_id TEXT NOT NULL, "
                "expired_at TEXT NOT NULL)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_active_agent_turn "
                "ON stop_active_invocations(agent_id, turn_digest)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_active_owner "
                "ON stop_active_invocations(owner_id, stop_requested)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_unresolved_agent_turn "
                "ON stop_unresolved_invocations(agent_id, turn_digest)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_unresolved_owner "
                "ON stop_unresolved_invocations(owner_id)"
            )

    async def _lock_agent(self, agent_id: str) -> None:
        if getattr(self._db, "backend_type", "") != "postgres":
            return
        await self._db.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (f"kestrel:stop:active-agent:{agent_id}",),
        )

    async def register(
        self,
        *,
        generation_id: str,
        agent_id: str,
        turn_id: str,
        owner_id: str,
    ) -> bool:
        """Register before cognition, or refuse an exact fenced turn."""

        generation_id = _required_identity(generation_id, "generation identity")
        agent_id = _required_identity(agent_id, "agent identity")
        turn_id = _required_identity(turn_id, "turn identity")
        owner_id = _required_identity(owner_id, "owner identity")
        digest = _turn_digest(turn_id)
        async with self._db.transaction(immediate=True):
            await self._lock_agent(agent_id)
            fenced = await self._db.fetchone(
                "SELECT 1 FROM stop_invocation_fences "
                "WHERE agent_id = ? AND turn_digest = ?",
                (agent_id, digest),
            )
            acknowledged = await self._db.fetchone(
                "SELECT 1 FROM stop_receipts AS receipt "
                "JOIN stop_receipt_outcomes AS outcome "
                "ON outcome.receipt_id = receipt.receipt_id "
                "WHERE receipt.scope = 'turn' "
                "AND receipt.target_agent_id = ? "
                "AND receipt.requested_target = ? "
                "AND outcome.disposition IN ('stopped', 'already_complete') "
                "LIMIT 1",
                (agent_id, opaque_stop_identifier("target", turn_id)),
            )
            if fenced is not None or acknowledged is not None:
                return False
            now_sql = database_now_sql(self._db)
            inserted = await self._db.execute(
                "INSERT INTO stop_active_invocations ("
                "generation_id, agent_id, turn_digest, owner_id, "
                "stop_requested, registered_at, heartbeat_at"
                f") VALUES (?, ?, ?, ?, 0, {now_sql}, {now_sql})",
                (generation_id, agent_id, digest, owner_id),
            )
            if inserted != 1:
                raise RuntimeError("distributed Stop registration was not durable")
        return True

    async def complete(self, generation_id: str, owner_id: str) -> None:
        generation_id = _required_identity(generation_id, "generation identity")
        owner_id = _required_identity(owner_id, "owner identity")
        async with self._db.transaction(immediate=True):
            row = await self._db.fetchone(
                "SELECT agent_id FROM stop_active_invocations "
                "WHERE generation_id = ? AND owner_id = ? "
                "UNION ALL "
                "SELECT agent_id FROM stop_unresolved_invocations "
                "WHERE generation_id = ? AND owner_id = ?",
                (generation_id, owner_id, generation_id, owner_id),
            )
            if row is None:
                return
            agent_id = _required_identity(row[0], "stored agent identity")
            await self._lock_agent(agent_id)
            deleted_active = await self._db.execute(
                "DELETE FROM stop_active_invocations "
                "WHERE generation_id = ? AND owner_id = ?",
                (generation_id, owner_id),
            )
            deleted_unresolved = await self._db.execute(
                "DELETE FROM stop_unresolved_invocations "
                "WHERE generation_id = ? AND owner_id = ?",
                (generation_id, owner_id),
            )
            if deleted_active + deleted_unresolved != 1:
                raise RuntimeError(
                    "distributed Stop completion changed inside its agent lock"
                )

    async def mark_turn(self, agent_id: str, turn_id: str) -> DistributedStopTicket:
        """Fence one exact turn and mark every live generation atomically."""

        agent_id = _required_identity(agent_id, "agent identity")
        turn_id = _required_identity(turn_id, "turn identity")
        digest = _turn_digest(turn_id)
        async with self._db.transaction(immediate=True):
            await self._lock_agent(agent_id)
            now_sql = database_now_sql(self._db)
            await self._db.execute(
                "INSERT INTO stop_invocation_fences "
                "(agent_id, turn_digest, created_at) "
                f"SELECT ?, ?, {now_sql} WHERE NOT EXISTS ("
                "SELECT 1 FROM stop_invocation_fences "
                "WHERE agent_id = ? AND turn_digest = ?)",
                (agent_id, digest, agent_id, digest),
            )
            rows = await self._db.fetchall(
                "SELECT generation_id FROM stop_active_invocations "
                "WHERE agent_id = ? AND turn_digest = ? "
                "UNION ALL "
                "SELECT generation_id FROM stop_unresolved_invocations "
                "WHERE agent_id = ? AND turn_digest = ? "
                "ORDER BY generation_id",
                (agent_id, digest, agent_id, digest),
            )
            generation_ids = tuple(str(row[0]) for row in rows)
            if generation_ids:
                changed = await self._db.execute(
                    "UPDATE stop_active_invocations SET stop_requested = 1 "
                    "WHERE agent_id = ? AND turn_digest = ?",
                    (agent_id, digest),
                )
                active_count = await self._db.fetchone(
                    "SELECT COUNT(*) FROM stop_active_invocations "
                    "WHERE agent_id = ? AND turn_digest = ?",
                    (agent_id, digest),
                )
                expected_changed = int(active_count[0]) if active_count else 0
                if changed != expected_changed:
                    raise RuntimeError(
                        "distributed Stop turn inventory changed inside its lock"
                    )
        return DistributedStopTicket(generation_ids)

    async def mark_agent(self, agent_id: str) -> DistributedStopTicket:
        """Mark the agent's current work; later units are outside this Stop."""

        agent_id = _required_identity(agent_id, "agent identity")
        async with self._db.transaction(immediate=True):
            await self._lock_agent(agent_id)
            rows = await self._db.fetchall(
                "SELECT generation_id FROM stop_active_invocations "
                "WHERE agent_id = ? "
                "UNION ALL "
                "SELECT generation_id FROM stop_unresolved_invocations "
                "WHERE agent_id = ? ORDER BY generation_id",
                (agent_id, agent_id),
            )
            generation_ids = tuple(str(row[0]) for row in rows)
            if generation_ids:
                changed = await self._db.execute(
                    "UPDATE stop_active_invocations SET stop_requested = 1 "
                    "WHERE agent_id = ?",
                    (agent_id,),
                )
                active_count = await self._db.fetchone(
                    "SELECT COUNT(*) FROM stop_active_invocations "
                    "WHERE agent_id = ?",
                    (agent_id,),
                )
                expected_changed = int(active_count[0]) if active_count else 0
                if changed != expected_changed:
                    raise RuntimeError(
                        "distributed Stop agent inventory changed inside its lock"
                    )
        return DistributedStopTicket(generation_ids)

    async def mark_generation(
        self,
        generation_id: str,
        agent_id: str,
    ) -> DistributedStopTicket:
        """Mark one exact durable generation without fencing its request ID."""

        generation_id = _required_identity(generation_id, "generation identity")
        agent_id = _required_identity(agent_id, "agent identity")
        async with self._db.transaction(immediate=True):
            await self._lock_agent(agent_id)
            row = await self._db.fetchone(
                "SELECT generation_id, 1 AS is_active "
                "FROM stop_active_invocations "
                "WHERE generation_id = ? AND agent_id = ? "
                "UNION ALL "
                "SELECT generation_id, 0 AS is_active "
                "FROM stop_unresolved_invocations "
                "WHERE generation_id = ? AND agent_id = ?",
                (generation_id, agent_id, generation_id, agent_id),
            )
            if row is None:
                return DistributedStopTicket(())
            if int(row[1]) == 1:
                changed = await self._db.execute(
                    "UPDATE stop_active_invocations SET stop_requested = 1 "
                    "WHERE generation_id = ? AND agent_id = ?",
                    (generation_id, agent_id),
                )
                if changed != 1:
                    raise RuntimeError(
                        "distributed Stop generation changed inside its lock"
                    )
        return DistributedStopTicket((generation_id,))

    async def poll_owner(
        self,
        owner_id: str,
        *,
        lease_seconds: float,
    ) -> _OwnerPoll:
        """Renew a still-live owner and return its exact durable inventory.

        Expiry is non-revivable.  A process that resumes after its heartbeat
        crossed the lease boundary must self-fence and start a new registry
        owner rather than silently reclaiming work another replica may already
        have reaped.
        """

        owner_id = _required_identity(owner_id, "owner identity")
        if lease_seconds <= 0:
            raise ValueError("distributed Stop owner lease must be positive")
        async with self._db.transaction(immediate=True):
            now_sql = database_now_sql(self._db)
            cutoff_sql, cutoff_args = _lease_cutoff_sql(
                self._db, float(lease_seconds)
            )
            await self._db.execute(
                "UPDATE stop_active_invocations "
                f"SET heartbeat_at = {now_sql} WHERE owner_id = ? "
                f"AND heartbeat_at > {cutoff_sql}",
                (owner_id, *cutoff_args),
            )
            rows = await self._db.fetchall(
                "SELECT generation_id, stop_requested "
                "FROM stop_active_invocations WHERE owner_id = ? "
                f"AND heartbeat_at > {cutoff_sql} "
                "ORDER BY generation_id",
                (owner_id, *cutoff_args),
            )
        return _OwnerPoll(
            live_generation_ids=tuple(str(row[0]) for row in rows),
            stop_generation_ids=tuple(
                str(row[0]) for row in rows if int(row[1]) == 1
            ),
        )

    async def reap_expired(
        self,
        generation_ids: tuple[str, ...],
        *,
        lease_seconds: float,
    ) -> tuple[str, ...]:
        """CAS-retire expired owners while preserving indeterminate work."""

        if not generation_ids:
            return ()
        if lease_seconds <= 0:
            raise ValueError("distributed Stop owner lease must be positive")
        placeholders = ", ".join("?" for _ in generation_ids)
        cutoff_sql, cutoff_args = _lease_cutoff_sql(
            self._db, float(lease_seconds)
        )
        reaped: list[str] = []
        async with self._db.transaction(immediate=True):
            rows = await self._db.fetchall(
                "SELECT generation_id, agent_id, heartbeat_at "
                "FROM stop_active_invocations "
                f"WHERE generation_id IN ({placeholders}) "
                f"AND heartbeat_at <= {cutoff_sql} ORDER BY generation_id",
                (*generation_ids, *cutoff_args),
            )
            for agent_id in sorted({str(row[1]) for row in rows}):
                await self._lock_agent(
                    _required_identity(agent_id, "stored agent identity")
                )
            for generation_id, _agent_id, observed_heartbeat in rows:
                # heartbeat_at is part of the retirement predicate: a renewal
                # that won the race makes this a no-op instead of retiring a
                # live owner from a stale read.
                cutoff_sql, cutoff_args = _lease_cutoff_sql(
                    self._db, float(lease_seconds)
                )
                now_sql = database_now_sql(self._db)
                changed = await self._db.execute(
                    "INSERT INTO stop_unresolved_invocations ("
                    "generation_id, agent_id, turn_digest, owner_id, expired_at) "
                    "SELECT generation_id, agent_id, turn_digest, owner_id, "
                    f"{now_sql} FROM stop_active_invocations "
                    "WHERE generation_id = ? AND heartbeat_at = ? "
                    f"AND heartbeat_at <= {cutoff_sql} "
                    "AND NOT EXISTS (SELECT 1 FROM stop_unresolved_invocations "
                    "WHERE generation_id = ?)",
                    (
                        str(generation_id),
                        observed_heartbeat,
                        *cutoff_args,
                        str(generation_id),
                    ),
                )
                if changed == 1:
                    deleted = await self._db.execute(
                        "DELETE FROM stop_active_invocations "
                        "WHERE generation_id = ? AND heartbeat_at = ?",
                        (str(generation_id), observed_heartbeat),
                    )
                    if deleted != 1:
                        raise RuntimeError(
                            "distributed Stop expired-owner retirement lost its row"
                        )
                    reaped.append(str(generation_id))
                elif changed != 0:
                    raise RuntimeError(
                        "distributed Stop stale-owner reap changed multiple rows"
                    )
        return tuple(reaped)

    async def remaining(
        self,
        generation_ids: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        """Return selected live or unresolved generations and their timestamps."""

        if not generation_ids:
            return ()
        placeholders = ", ".join("?" for _ in generation_ids)
        rows = await self._db.fetchall(
            "SELECT generation_id, heartbeat_at FROM stop_active_invocations "
            f"WHERE generation_id IN ({placeholders}) "
            "UNION ALL "
            "SELECT generation_id, expired_at FROM stop_unresolved_invocations "
            f"WHERE generation_id IN ({placeholders}) ORDER BY generation_id",
            (*generation_ids, *generation_ids),
        )
        return tuple((str(row[0]), str(row[1])) for row in rows)

    async def delete_owner(self, owner_id: str) -> None:
        owner_id = _required_identity(owner_id, "owner identity")
        async with self._db.transaction(immediate=True):
            rows = await self._db.fetchall(
                "SELECT agent_id FROM stop_active_invocations "
                "WHERE owner_id = ? "
                "UNION "
                "SELECT agent_id FROM stop_unresolved_invocations "
                "WHERE owner_id = ? ORDER BY agent_id",
                (owner_id, owner_id),
            )
            for row in rows:
                await self._lock_agent(
                    _required_identity(row[0], "stored agent identity")
                )
            await self._db.execute(
                "DELETE FROM stop_active_invocations WHERE owner_id = ?",
                (owner_id,),
            )
            await self._db.execute(
                "DELETE FROM stop_unresolved_invocations WHERE owner_id = ?",
                (owner_id,),
            )


class DistributedInvocationRegistry:
    """One process's live map and relay for shared Stop requests."""

    def __init__(
        self,
        store: DistributedInvocationStore,
        *,
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
        owner_lease_seconds: float = _DEFAULT_OWNER_LEASE_SECONDS,
    ) -> None:
        if not isinstance(store, DistributedInvocationStore):
            raise TypeError("distributed Stop registry requires its typed store")
        if poll_seconds <= 0:
            raise ValueError("distributed Stop poll interval must be positive")
        if owner_lease_seconds <= poll_seconds:
            raise ValueError(
                "distributed Stop owner lease must exceed its poll interval"
            )
        self._store = store
        self._owner_id = uuid4().hex
        self._poll_seconds = float(poll_seconds)
        self._owner_lease_seconds = float(owner_lease_seconds)
        self._active: dict[str, tuple[object, str, int]] = {}
        self._by_local_generation: dict[tuple[int, str, int], str] = {}
        self._registration_lock = asyncio.Lock()
        self._registration_tasks: set[asyncio.Task[bool]] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_keys: set[tuple[int, str, int]] = set()
        self._relay_task: asyncio.Task[None] | None = None
        self._closing = False
        self._lease_lost = False
        self._last_heartbeat_monotonic: float | None = None

    def start(self) -> None:
        if self._relay_task is None:
            self._relay_task = asyncio.create_task(
                self._relay(), name="distributed-stop-relay"
            )

    def attach(self, agent: object) -> None:
        if self._closing:
            raise RuntimeError("distributed Stop registry is closing")
        if self._lease_lost:
            raise RuntimeError("distributed Stop owner lease was lost")
        agent.__dict__["_distributed_invocation_registry"] = self

    @staticmethod
    def _agent_id(agent: object) -> str:
        agent_id = getattr(agent, "agent_id", None)
        return (
            agent_id
            if isinstance(agent_id, str) and agent_id.strip()
            else "local-agent"
        )

    async def register(
        self,
        agent: object,
        turn_id: str,
        generation: int,
    ) -> bool:
        if self._closing:
            raise RuntimeError("distributed Stop registry is closing")
        if self._lease_lost:
            return False
        key = (id(agent), turn_id, generation)

        async def publish() -> bool:
            async with self._registration_lock:
                if self._lease_lost:
                    return False
                last_heartbeat = self._last_heartbeat_monotonic
                if (
                    self._active
                    and last_heartbeat is not None
                    and asyncio.get_running_loop().time() - last_heartbeat
                    >= self._owner_lease_seconds
                ):
                    self._fail_closed_owner()
                    return False
                if key in self._by_local_generation:
                    return True
                generation_id = uuid4().hex
                admitted = await self._store.register(
                    generation_id=generation_id,
                    agent_id=self._agent_id(agent),
                    turn_id=turn_id,
                    owner_id=self._owner_id,
                )
                if not admitted:
                    return False
                if self._lease_lost:
                    await self._store.complete(generation_id, self._owner_id)
                    return False
                self._by_local_generation[key] = generation_id
                self._active[generation_id] = (agent, turn_id, generation)
                self._last_heartbeat_monotonic = (
                    asyncio.get_running_loop().time()
                )
                return True

        owner = asyncio.create_task(publish(), name="distributed-stop-register")
        self._registration_tasks.add(owner)
        owner.add_done_callback(self._registration_tasks.discard)
        outcome = await await_owned_task(owner)
        return raise_owned_outcome(
            outcome, operation="distributed Stop invocation registration"
        )

    def complete_soon(self, agent: object, turn_id: str, generation: int) -> None:
        """Own durable cleanup from the mixin's synchronous finally path."""

        key = (id(agent), turn_id, generation)
        generation_id = self._by_local_generation.get(key)
        if generation_id is None or key in self._cleanup_keys:
            return
        self._cleanup_keys.add(key)

        async def complete() -> None:
            try:
                while not self._closing:
                    try:
                        await self._store.complete(generation_id, self._owner_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        logger.error(
                            "Distributed Stop completion failed; retrying (%s)",
                            type(error).__name__,
                            exc_info=(type(error), error, error.__traceback__),
                        )
                        await asyncio.sleep(self._poll_seconds)
                        continue
                    self._by_local_generation.pop(key, None)
                    self._active.pop(generation_id, None)
                    if not self._active:
                        # A lease protects durable owner rows, not an idle
                        # process identity. The next admission starts a fresh
                        # lease generation instead of inheriting elapsed idle
                        # time from work that already completed.
                        self._last_heartbeat_monotonic = None
                    return
            finally:
                self._cleanup_keys.discard(key)

        task = asyncio.create_task(
            complete(), name=f"distributed-stop-complete:{generation_id}"
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._consume_cleanup)

    def _consume_cleanup(self, task: asyncio.Task[None]) -> None:
        self._cleanup_tasks.discard(task)
        try:
            task.result()
        except BaseException as error:
            logger.error(
                "Distributed Stop completion failed (%s)",
                type(error).__name__,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def request_turn(self, agent_id: str, turn_id: str) -> DistributedStopTicket:
        return await self._store.mark_turn(agent_id, turn_id)

    async def request_generation(
        self,
        agent: object,
        turn_id: str,
        generation: int,
    ) -> DistributedStopTicket:
        """Stop one locally resolved generation without fencing later reuse."""

        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation <= 0
        ):
            raise ValueError("distributed Stop local generation must be positive")
        generation_id = self._by_local_generation.get(
            (id(agent), turn_id, generation)
        )
        if generation_id is None:
            return DistributedStopTicket(())
        return await self._store.mark_generation(
            generation_id,
            self._agent_id(agent),
        )

    async def request_agent(self, agent_id: str) -> DistributedStopTicket:
        return await self._store.mark_agent(agent_id)

    async def wait_for_stop(
        self,
        ticket: DistributedStopTicket,
        *,
        timeout_seconds: float = _DEFAULT_WAIT_SECONDS,
    ) -> StopDisposition:
        if not ticket.generation_ids:
            return StopDisposition.ALREADY_COMPLETE
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = await self._store.remaining(ticket.generation_ids)
            if not remaining:
                return StopDisposition.STOPPED
            reaped = await self._store.reap_expired(
                tuple(generation_id for generation_id, _heartbeat in remaining),
                lease_seconds=self._owner_lease_seconds,
            )
            if reaped:
                continue
            if asyncio.get_running_loop().time() >= deadline:
                # A dead or partitioned owner remains UNREACHABLE; liveness
                # uncertainty is never rewritten as already complete.
                return StopDisposition.UNREACHABLE
            await asyncio.sleep(self._poll_seconds)

    def _fail_closed_owner(self) -> None:
        if self._lease_lost:
            return
        self._lease_lost = True
        for agent, turn_id, generation in tuple(self._active.values()):
            cancel = getattr(agent, "cancel_current_request", None)
            if callable(cancel):
                try:
                    cancel(request_id=turn_id, generation=generation)
                except Exception:
                    logger.exception(
                        "Distributed Stop owner self-fence cancellation failed"
                    )

    async def _relay(self) -> None:
        while not self._closing:
            if not self._active:
                await asyncio.sleep(self._poll_seconds)
                continue
            try:
                polled = await self._store.poll_owner(
                    self._owner_id,
                    lease_seconds=self._owner_lease_seconds,
                )
                now = asyncio.get_running_loop().time()
                self._last_heartbeat_monotonic = now
                live = set(polled.live_generation_ids)
                if any(generation_id not in live for generation_id in self._active):
                    self._fail_closed_owner()
                else:
                    for generation_id in polled.stop_generation_ids:
                        target = self._active.get(generation_id)
                        if target is None:
                            continue
                        agent, turn_id, generation = target
                        cancel = getattr(agent, "cancel_current_request", None)
                        if callable(cancel):
                            cancel(request_id=turn_id, generation=generation)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "Distributed Stop relay failed (%s)",
                    type(error).__name__,
                    exc_info=(type(error), error, error.__traceback__),
                )
                last_heartbeat = self._last_heartbeat_monotonic
                if (
                    last_heartbeat is not None
                    and asyncio.get_running_loop().time() - last_heartbeat
                    >= self._owner_lease_seconds
                ):
                    self._fail_closed_owner()
            await asyncio.sleep(self._poll_seconds)

    async def close(self) -> None:
        self._closing = True
        relay = self._relay_task
        if relay is not None:
            relay.cancel()
            try:
                await relay
            except asyncio.CancelledError:
                pass
            self._relay_task = None
        while self._registration_tasks:
            await asyncio.gather(
                *tuple(self._registration_tasks), return_exceptions=True
            )
        while self._cleanup_tasks:
            await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
        await self._store.delete_owner(self._owner_id)
        self._active.clear()
        self._by_local_generation.clear()
        self._cleanup_keys.clear()


__all__ = [
    "DistributedInvocationRegistry",
    "DistributedInvocationStore",
    "DistributedStopTicket",
]
