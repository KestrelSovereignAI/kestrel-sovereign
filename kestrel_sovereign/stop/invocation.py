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
from kestrel_sovereign.storage.database_clock import database_now_sql

from .receipt import StopReceiptStore, opaque_stop_identifier
from .types import StopDisposition

_SCHEMA_LOCK = "stop_invocations_v2"
_TURN_ID_DOMAIN = b"kestrel:distributed-stop-turn:v1\0"
_DEFAULT_POLL_SECONDS = 0.1
_DEFAULT_WAIT_SECONDS = 4.0
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
                "turn_address_digest TEXT, "
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
            if getattr(self._db, "backend_type", "") == "postgres":
                await self._db.execute(
                    "ALTER TABLE stop_active_invocations "
                    "ADD COLUMN IF NOT EXISTS turn_address_digest TEXT"
                )
            else:
                columns = await self._db.fetchall(
                    "PRAGMA table_info(stop_active_invocations)"
                )
                if "turn_address_digest" not in {str(row[1]) for row in columns}:
                    await self._db.execute(
                        "ALTER TABLE stop_active_invocations "
                        "ADD COLUMN turn_address_digest TEXT"
                    )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_active_agent_turn "
                "ON stop_active_invocations(agent_id, turn_digest)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_active_agent_address "
                "ON stop_active_invocations(agent_id, turn_address_digest)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_active_owner "
                "ON stop_active_invocations(owner_id, stop_requested)"
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

    async def bind_turn_address(
        self,
        *,
        generation_id: str,
        agent_id: str,
        turn_id: str,
        owner_id: str,
    ) -> bool:
        """Bind a public turn address to one admitted transport generation.

        The address is minted after request admission.  Binding therefore has
        its own fence race: a remote Stop may durably fence the public address
        before the owning replica publishes it.  That race fails closed and
        the caller aborts before cognition begins.
        """

        generation_id = _required_identity(generation_id, "generation identity")
        agent_id = _required_identity(agent_id, "agent identity")
        turn_id = _required_identity(turn_id, "turn identity")
        owner_id = _required_identity(owner_id, "owner identity")
        digest = _turn_digest(turn_id)
        async with self._db.transaction(immediate=True):
            await self._lock_agent(agent_id)
            row = await self._db.fetchone(
                "SELECT turn_address_digest FROM stop_active_invocations "
                "WHERE generation_id = ? AND agent_id = ? AND owner_id = ?",
                (generation_id, agent_id, owner_id),
            )
            if row is None:
                raise RuntimeError(
                    "distributed Stop turn binding lost its admitted generation"
                )
            if row[0] is not None:
                if row[0] != digest:
                    raise RuntimeError(
                        "distributed Stop generation already has another turn address"
                    )
                return True
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
            changed = await self._db.execute(
                "UPDATE stop_active_invocations SET turn_address_digest = ? "
                "WHERE generation_id = ? AND agent_id = ? AND owner_id = ? "
                "AND turn_address_digest IS NULL",
                (digest, generation_id, agent_id, owner_id),
            )
            if changed != 1:
                raise RuntimeError(
                    "distributed Stop turn binding changed inside its agent lock"
                )
        return True

    async def complete(self, generation_id: str, owner_id: str) -> None:
        generation_id = _required_identity(generation_id, "generation identity")
        owner_id = _required_identity(owner_id, "owner identity")
        async with self._db.transaction(immediate=True):
            row = await self._db.fetchone(
                "SELECT agent_id FROM stop_active_invocations "
                "WHERE generation_id = ? AND owner_id = ?",
                (generation_id, owner_id),
            )
            if row is None:
                return
            agent_id = _required_identity(row[0], "stored agent identity")
            await self._lock_agent(agent_id)
            deleted = await self._db.execute(
                "DELETE FROM stop_active_invocations "
                "WHERE generation_id = ? AND owner_id = ?",
                (generation_id, owner_id),
            )
            if deleted != 1:
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
                "WHERE agent_id = ? "
                "AND (turn_digest = ? OR turn_address_digest = ?) "
                "ORDER BY generation_id",
                (agent_id, digest, digest),
            )
            generation_ids = tuple(str(row[0]) for row in rows)
            if generation_ids:
                changed = await self._db.execute(
                    "UPDATE stop_active_invocations SET stop_requested = 1 "
                    "WHERE agent_id = ? "
                    "AND (turn_digest = ? OR turn_address_digest = ?)",
                    (agent_id, digest, digest),
                )
                if changed != len(generation_ids):
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
                "WHERE agent_id = ? ORDER BY generation_id",
                (agent_id,),
            )
            generation_ids = tuple(str(row[0]) for row in rows)
            if generation_ids:
                changed = await self._db.execute(
                    "UPDATE stop_active_invocations SET stop_requested = 1 "
                    "WHERE agent_id = ?",
                    (agent_id,),
                )
                if changed != len(generation_ids):
                    raise RuntimeError(
                        "distributed Stop agent inventory changed inside its lock"
                    )
        return DistributedStopTicket(generation_ids)

    async def poll_owner(self, owner_id: str) -> tuple[str, ...]:
        """Heartbeat this process and return generations it must cancel."""

        owner_id = _required_identity(owner_id, "owner identity")
        async with self._db.transaction(immediate=True):
            now_sql = database_now_sql(self._db)
            await self._db.execute(
                "UPDATE stop_active_invocations "
                f"SET heartbeat_at = {now_sql} WHERE owner_id = ?",
                (owner_id,),
            )
            rows = await self._db.fetchall(
                "SELECT generation_id FROM stop_active_invocations "
                "WHERE owner_id = ? AND stop_requested = 1 "
                "ORDER BY generation_id",
                (owner_id,),
            )
        return tuple(str(row[0]) for row in rows)

    async def remaining(
        self,
        generation_ids: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        """Return selected generations still live and their heartbeat stamps."""

        if not generation_ids:
            return ()
        placeholders = ", ".join("?" for _ in generation_ids)
        rows = await self._db.fetchall(
            "SELECT generation_id, heartbeat_at FROM stop_active_invocations "
            f"WHERE generation_id IN ({placeholders}) ORDER BY generation_id",
            generation_ids,
        )
        return tuple((str(row[0]), str(row[1])) for row in rows)

    async def delete_owner(self, owner_id: str) -> None:
        owner_id = _required_identity(owner_id, "owner identity")
        async with self._db.transaction(immediate=True):
            rows = await self._db.fetchall(
                "SELECT DISTINCT agent_id FROM stop_active_invocations "
                "WHERE owner_id = ? ORDER BY agent_id",
                (owner_id,),
            )
            for row in rows:
                await self._lock_agent(
                    _required_identity(row[0], "stored agent identity")
                )
            await self._db.execute(
                "DELETE FROM stop_active_invocations WHERE owner_id = ?",
                (owner_id,),
            )


class DistributedInvocationRegistry:
    """One process's live map and relay for shared Stop requests."""

    def __init__(
        self,
        store: DistributedInvocationStore,
        *,
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
    ) -> None:
        if not isinstance(store, DistributedInvocationStore):
            raise TypeError("distributed Stop registry requires its typed store")
        if poll_seconds <= 0:
            raise ValueError("distributed Stop poll interval must be positive")
        self._store = store
        self._owner_id = uuid4().hex
        self._poll_seconds = float(poll_seconds)
        self._active: dict[str, tuple[object, str, int]] = {}
        self._by_local_generation: dict[tuple[int, str, int], str] = {}
        self._registration_lock = asyncio.Lock()
        self._registration_tasks: set[asyncio.Task[bool]] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_keys: set[tuple[int, str, int]] = set()
        self._relay_task: asyncio.Task[None] | None = None
        self._closing = False

    def start(self) -> None:
        if self._relay_task is None:
            self._relay_task = asyncio.create_task(
                self._relay(), name="distributed-stop-relay"
            )

    def attach(self, agent: object) -> None:
        if self._closing:
            raise RuntimeError("distributed Stop registry is closing")
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
        key = (id(agent), turn_id, generation)

        async def publish() -> bool:
            async with self._registration_lock:
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
                self._by_local_generation[key] = generation_id
                self._active[generation_id] = (agent, turn_id, generation)
                return True

        owner = asyncio.create_task(publish(), name="distributed-stop-register")
        self._registration_tasks.add(owner)
        owner.add_done_callback(self._registration_tasks.discard)
        outcome = await await_owned_task(owner)
        return raise_owned_outcome(
            outcome, operation="distributed Stop invocation registration"
        )

    async def bind_turn_address(
        self,
        agent: object,
        turn_id: str,
        request_id: str,
        generation: int,
    ) -> bool:
        """Publish the lifecycle-owned public alias for an admitted generation."""

        key = (id(agent), request_id, generation)
        async with self._registration_lock:
            generation_id = self._by_local_generation.get(key)
            if generation_id is None:
                raise RuntimeError(
                    "distributed Stop turn binding requires durable admission"
                )
            return await self._store.bind_turn_address(
                generation_id=generation_id,
                agent_id=self._agent_id(agent),
                turn_id=turn_id,
                owner_id=self._owner_id,
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
            if asyncio.get_running_loop().time() >= deadline:
                # A dead or partitioned owner remains UNREACHABLE; liveness
                # uncertainty is never rewritten as already complete.
                return StopDisposition.UNREACHABLE
            await asyncio.sleep(self._poll_seconds)

    async def _relay(self) -> None:
        while not self._closing:
            if not self._active:
                await asyncio.sleep(self._poll_seconds)
                continue
            try:
                requested = await self._store.poll_owner(self._owner_id)
                for generation_id in requested:
                    target = self._active.get(generation_id)
                    if target is None:
                        continue
                    agent, request_id, generation = target
                    cancel = getattr(agent, "cancel_current_request", None)
                    if callable(cancel):
                        cancel(request_id=request_id, generation=generation)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "Distributed Stop relay failed (%s)",
                    type(error).__name__,
                    exc_info=(type(error), error, error.__traceback__),
                )
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
