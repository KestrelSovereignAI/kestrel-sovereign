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
from kestrel_sovereign.agent.invocation import (
    InvocationSelfFencedError,
    validate_invocation_id,
)
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from kestrel_sdk.storage.database.interface import TransactionError
from kestrel_sovereign.storage.database_clock import (
    database_lease_cutoff_sql,
    database_now_sql,
)

from .receipt import (
    STOP_OWNER_LEASE_SECONDS,
    StopReceiptStore,
    opaque_stop_identifier,
)
from .types import StopDisposition

_SCHEMA_LOCK = "stop_invocations_v1"

# Canonical DDL for the two generation ledgers. ONE spelling each: the same
# template creates a table fresh and adopts a legacy-shaped one, so the two
# cannot drift (#2804). ``{table}`` is the only placeholder.
STOP_ACTIVE_INVOCATIONS_DDL = (
    "CREATE TABLE IF NOT EXISTS {table} ("
    "generation_id TEXT NOT NULL PRIMARY KEY, "
    "agent_id TEXT NOT NULL, "
    "turn_digest TEXT NOT NULL, "
    "public_turn_digest TEXT, "
    "request_generation INTEGER NOT NULL, "
    "owner_id TEXT NOT NULL, "
    "stop_requested INTEGER NOT NULL DEFAULT 0, "
    "registered_at TEXT NOT NULL, "
    "heartbeat_at TEXT NOT NULL, "
    "CHECK (request_generation > 0), "
    "CHECK (stop_requested IN (0, 1)))"
)
STOP_UNRESOLVED_INVOCATIONS_DDL = (
    "CREATE TABLE IF NOT EXISTS {table} ("
    "generation_id TEXT NOT NULL PRIMARY KEY, "
    "agent_id TEXT NOT NULL, "
    "turn_digest TEXT NOT NULL, "
    "public_turn_digest TEXT, "
    "request_generation INTEGER NOT NULL, "
    "owner_id TEXT NOT NULL, "
    "expired_at TEXT NOT NULL, "
    "CHECK (request_generation > 0))"
)
_EXACT_GENERATION_COLUMN = "request_generation"
_EXACT_GENERATION_CHECK = "request_generation > 0"
_TURN_ID_DOMAIN = b"kestrel:distributed-stop-turn:v1\0"
_PUBLIC_TURN_ID_DOMAIN = b"kestrel:distributed-stop-public-turn:v1\0"
_DEFAULT_POLL_SECONDS = 0.1
_DEFAULT_WAIT_SECONDS = 4.0
_DEFAULT_OWNER_LEASE_SECONDS = STOP_OWNER_LEASE_SECONDS
# Teardown must be bounded. close() runs in the server's shutdown phases, and
# the completion retry loop only consults ``_closing`` on its EXCEPTION path --
# a settle that hangs rather than raises never reaches that check, so an
# unbounded join here is a wedge no signal can break.
_DEFAULT_CLOSE_DRAIN_SECONDS = 10.0
logger = logging.getLogger(__name__)


def _turn_digest(turn_id: str) -> str:
    encoded = turn_id.encode("utf-8")
    return (
        "sha256:"
        + hashlib.sha256(
            _TURN_ID_DOMAIN + len(encoded).to_bytes(4, "big") + encoded
        ).hexdigest()
    )


def _public_turn_digest(turn_id: str) -> str:
    encoded = turn_id.encode("utf-8")
    return (
        "sha256:"
        + hashlib.sha256(
            _PUBLIC_TURN_ID_DOMAIN + len(encoded).to_bytes(4, "big") + encoded
        ).hexdigest()
    )


def _required_identity(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"distributed Stop {field} must be concrete")
    return value


def _required_opaque_identity(value: object, field: str) -> str:
    try:
        return validate_invocation_id(value)
    except ValueError as error:
        raise ValueError(
            f"distributed Stop {field} must be a valid opaque identity"
        ) from error


class StopLegacyRegistrationsError(RuntimeError):
    """A generation ledger predates exact generations and still holds rows.

    Raised by :meth:`DistributedInvocationStore.ensure_schema` instead of
    adopting the table. A registration written before ``request_generation``
    existed has no recorded generation, and no value in the column's domain
    is honest for it: inventing one is exactly the imprecision the column was
    added to remove (``cb5154e2b``). The boot that raises this never registers
    a turn, so nothing runs against the half-shaped ledger; it keeps refusing
    until the rows are gone (#3292).
    """

    def __init__(self, table: str, count: int):
        self.table = table
        self.count = count
        message = (
            f"{table} predates exact Stop generations and still holds {count} "
            "registration(s) with no recorded generation; refusing to adopt "
            "the schema. Those rows were written by pre-release Stop code "
            "(#3292): confirm no owner is live, delete the rows WHERE "
            f"{_EXACT_GENERATION_COLUMN} IS NULL, and start again. No value "
            "can be invented for them."
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class DistributedStopTicket:
    """The exact durable generations selected at Stop linearization."""

    generation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _OwnerPoll:
    """One non-revivable heartbeat result for an invocation owner."""

    live_generation_ids: tuple[str, ...]
    stop_generation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LocalGeneration:
    """One in-process generation and the owner identity that admitted it.

    The owner is recorded per generation because a fenced owner is replaced
    rather than revived (#3337): settlement and turn binding must keep using
    the identity the durable row was written under, while the lease relay
    renews only the current owner's rows.
    """

    agent: object
    turn_id: str
    generation: int
    owner_id: str


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
                STOP_ACTIVE_INVOCATIONS_DDL.format(
                    table="stop_active_invocations"
                )
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_invocation_fences ("
                "agent_id TEXT NOT NULL, "
                "turn_digest TEXT NOT NULL, "
                "created_at TEXT NOT NULL, "
                "PRIMARY KEY (agent_id, turn_digest))"
            )
            await self._db.execute(
                STOP_UNRESOLVED_INVOCATIONS_DDL.format(
                    table="stop_unresolved_invocations"
                )
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
        # A ledger created before ``request_generation`` (pre-release Stop
        # code, #3292) must reach the canonical shape before anything reads or
        # writes it: ``register()`` names the column on every cognition turn.
        await self._adopt_exact_generation(
            "stop_active_invocations",
            STOP_ACTIVE_INVOCATIONS_DDL,
            lock_name=f"{_SCHEMA_LOCK}:active",
        )
        await self._adopt_exact_generation(
            "stop_unresolved_invocations",
            STOP_UNRESOLVED_INVOCATIONS_DDL,
            lock_name=f"{_SCHEMA_LOCK}:unresolved",
        )
        await self._db.migrate_columns_once(
            "stop_active_invocations",
            (("public_turn_digest", "TEXT"),),
            lock_name=f"{_SCHEMA_LOCK}:active-public-turn",
        )
        await self._db.migrate_columns_once(
            "stop_unresolved_invocations",
            (("public_turn_digest", "TEXT"),),
            lock_name=f"{_SCHEMA_LOCK}:unresolved-public-turn",
        )
        await self._db.ensure_index(
            "idx_stop_active_agent_public_turn",
            "stop_active_invocations",
            "agent_id, public_turn_digest",
            where="public_turn_digest IS NOT NULL",
        )
        await self._db.ensure_index(
            "idx_stop_unresolved_agent_public_turn",
            "stop_unresolved_invocations",
            "agent_id, public_turn_digest",
            where="public_turn_digest IS NOT NULL",
        )

    async def _adopt_exact_generation(
        self, table: str, canonical_ddl: str, *, lock_name: str
    ) -> None:
        """Carry a ledger created before ``request_generation`` to canonical.

        ``cb5154e2b`` put the column in the ``CREATE TABLE`` alone, so a table
        that already existed never gained it, and on such a host every
        ``register()`` fails at its INSERT: no agent can begin a cognition
        turn, while every action-mode scheduled tool still reports success
        (#3292). Adoption takes the #3289 posture: an **empty** legacy ledger
        is carried to the canonical shape; one that still holds rows is
        refused with the reason named, because a registration that never
        recorded its exact generation has no honest value for the column.

        The column is added nullable first (``migrate_columns_once`` is the
        only portable ALTER: SQLite cannot add a NOT NULL column without a
        default, and a default *is* the invented value), which makes a legacy
        row exactly a NULL row. ``ensure_check_constraint`` then converges the
        shape inside one migration transaction: its remediation refuses while
        any NULL row exists and, on PostgreSQL, sets NOT NULL in place (a
        rebuild there discards concurrent writes; on SQLite the rebuild into
        ``canonical_ddl`` supplies NOT NULL itself). A refused host refuses on
        every boot until the rows are gone and registers nothing in between:
        the server does not come up. Both probes are no-ops on a canonical
        ledger, so a fresh host pays two reads per boot and is never rebuilt.
        """
        await self._db.migrate_columns_once(
            table,
            ((_EXACT_GENERATION_COLUMN, "INTEGER"),),
            lock_name=f"{lock_name}-request-generation",
        )

        async def _refuse_unrecorded_generations() -> None:
            unrecorded = await self._db.fetchval(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE {_EXACT_GENERATION_COLUMN} IS NULL"
            )
            if unrecorded:
                raise StopLegacyRegistrationsError(table, int(unrecorded))
            if self._db.backend_type == "postgres":
                await self._db.execute(
                    f"ALTER TABLE {table} "
                    f"ALTER COLUMN {_EXACT_GENERATION_COLUMN} SET NOT NULL"
                )

        try:
            await self._db.ensure_check_constraint(
                table,
                f"{table}_{_EXACT_GENERATION_COLUMN}_check",
                _EXACT_GENERATION_CHECK,
                canonical_ddl=canonical_ddl,
                remediation=_refuse_unrecorded_generations,
                lock_name=f"{lock_name}-request-generation-check",
            )
        except TransactionError as exc:
            # The migration transaction wraps whatever its block raised. The
            # refusal is a named boot outcome, not a storage fault: surface it
            # as itself so the boot log and any caller can tell the two apart.
            if isinstance(exc.__cause__, StopLegacyRegistrationsError):
                raise exc.__cause__ from exc
            raise

    async def _lock_agent(self, agent_id: str) -> None:
        if getattr(self._db, "backend_type", "") != "postgres":
            return
        await self._db.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (f"kestrel:stop:active-agent:{agent_id}",),
        )

    async def agent_has_unsettled_work(self, agent_id: str) -> bool:
        """Report whether any replica still owns or preserves this agent's work."""

        agent_id = _required_identity(agent_id, "agent identity")
        row = await self._db.fetchone(
            "SELECT generation_id FROM stop_active_invocations "
            "WHERE agent_id = ? UNION ALL "
            "SELECT generation_id FROM stop_unresolved_invocations "
            "WHERE agent_id = ? LIMIT 1",
            (agent_id, agent_id),
        )
        return row is not None

    async def register(
        self,
        *,
        generation_id: str,
        agent_id: str,
        turn_id: str,
        owner_id: str,
        request_generation: int,
    ) -> bool:
        """Register before cognition, or refuse an exact fenced turn."""

        generation_id = _required_identity(generation_id, "generation identity")
        agent_id = _required_identity(agent_id, "agent identity")
        turn_id = _required_opaque_identity(turn_id, "turn identity")
        owner_id = _required_identity(owner_id, "owner identity")
        if (
            not isinstance(request_generation, int)
            or isinstance(request_generation, bool)
            or request_generation <= 0
        ):
            raise ValueError(
                "distributed Stop request generation must be a positive integer"
            )
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
                "generation_id, agent_id, turn_digest, request_generation, owner_id, "
                "stop_requested, registered_at, heartbeat_at"
                f") VALUES (?, ?, ?, ?, ?, 0, {now_sql}, {now_sql})",
                (
                    generation_id,
                    agent_id,
                    digest,
                    request_generation,
                    owner_id,
                ),
            )
            if inserted != 1:
                raise RuntimeError("distributed Stop registration was not durable")
        return True

    async def bind_public_turn(
        self,
        *,
        generation_id: str,
        owner_id: str,
        agent_id: str,
        turn_id: str,
    ) -> bool:
        """Bind a public turn to one durable generation before cognition."""

        generation_id = _required_identity(generation_id, "generation identity")
        owner_id = _required_identity(owner_id, "owner identity")
        agent_id = _required_identity(agent_id, "agent identity")
        turn_id = _required_opaque_identity(turn_id, "public turn identity")
        digest = _public_turn_digest(turn_id)
        async with self._db.transaction(immediate=True):
            await self._lock_agent(agent_id)
            row = await self._db.fetchone(
                "SELECT public_turn_digest, stop_requested "
                "FROM stop_active_invocations "
                "WHERE generation_id = ? AND owner_id = ? AND agent_id = ?",
                (generation_id, owner_id, agent_id),
            )
            if row is None:
                return False
            if int(row[1]) == 1:
                return False
            if row[0] is not None:
                if str(row[0]) != digest:
                    raise RuntimeError(
                        "distributed Stop generation already has a public turn"
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
                (
                    agent_id,
                    opaque_stop_identifier("public_turn_target", turn_id),
                ),
            )
            if fenced is not None or acknowledged is not None:
                return False
            conflict = await self._db.fetchone(
                "SELECT generation_id FROM stop_active_invocations "
                "WHERE agent_id = ? AND public_turn_digest = ? "
                "UNION ALL "
                "SELECT generation_id FROM stop_unresolved_invocations "
                "WHERE agent_id = ? AND public_turn_digest = ? LIMIT 1",
                (agent_id, digest, agent_id, digest),
            )
            if conflict is not None and str(conflict[0]) != generation_id:
                raise RuntimeError(
                    "distributed Stop public turn has multiple generations"
                )
            changed = await self._db.execute(
                "UPDATE stop_active_invocations SET public_turn_digest = ? "
                "WHERE generation_id = ? AND owner_id = ? AND agent_id = ? "
                "AND public_turn_digest IS NULL",
                (digest, generation_id, owner_id, agent_id),
            )
            if changed != 1:
                raise RuntimeError(
                    "distributed Stop public-turn binding changed inside its lock"
                )
        return True

    async def settle(
        self,
        generation_id: str,
        owner_id: str,
        disposition: RequestCompletionDisposition,
    ) -> None:
        """Apply the only valid terminal transition for one generation.

        A generation may disappear from shared authority only when its owner
        observed completion.  ``ABANDONED`` means the terminal state is
        unknown, so it is moved to the unresolved ledger instead.  This is the
        lifecycle invariant used by local cleanup and remote Stop results.
        """

        generation_id = _required_identity(generation_id, "generation identity")
        owner_id = _required_identity(owner_id, "owner identity")
        if not isinstance(disposition, RequestCompletionDisposition):
            raise TypeError("distributed Stop settlement disposition must be typed")
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
            active = await self._db.fetchone(
                "SELECT agent_id FROM stop_active_invocations "
                "WHERE generation_id = ? AND owner_id = ?",
                (generation_id, owner_id),
            )
            unresolved = await self._db.fetchone(
                "SELECT agent_id FROM stop_unresolved_invocations "
                "WHERE generation_id = ? AND owner_id = ?",
                (generation_id, owner_id),
            )
            if active is not None and unresolved is not None:
                raise RuntimeError(
                    "distributed Stop generation exists in multiple lifecycle states"
                )
            if disposition is RequestCompletionDisposition.ABANDONED:
                if unresolved is not None:
                    return
                if active is None:
                    return
                now_sql = database_now_sql(self._db)
                inserted = await self._db.execute(
                    "INSERT INTO stop_unresolved_invocations ("
                    "generation_id, agent_id, turn_digest, public_turn_digest, "
                    "request_generation, owner_id, expired_at) "
                    "SELECT generation_id, agent_id, turn_digest, "
                    "public_turn_digest, request_generation, owner_id, "
                    f"{now_sql} FROM stop_active_invocations "
                    "WHERE generation_id = ? AND owner_id = ?",
                    (generation_id, owner_id),
                )
                deleted = await self._db.execute(
                    "DELETE FROM stop_active_invocations "
                    "WHERE generation_id = ? AND owner_id = ?",
                    (generation_id, owner_id),
                )
                if inserted != 1 or deleted != 1:
                    raise RuntimeError(
                        "distributed Stop abandonment changed inside its agent lock"
                    )
                return
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

    async def complete(self, generation_id: str, owner_id: str) -> None:
        """Record an owner-observed terminal completion."""

        await self.settle(
            generation_id,
            owner_id,
            RequestCompletionDisposition.COMPLETED,
        )

    async def abandon(self, generation_id: str, owner_id: str) -> None:
        """Preserve a generation whose terminal outcome is indeterminate."""

        await self.settle(
            generation_id,
            owner_id,
            RequestCompletionDisposition.ABANDONED,
        )

    async def mark_turn(
        self,
        agent_id: str,
        turn_id: str,
    ) -> DistributedStopTicket:
        """Fence one request address and mark all of its live deliveries."""

        agent_id = _required_identity(agent_id, "agent identity")
        turn_id = _required_opaque_identity(turn_id, "turn identity")
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

    async def mark_public_turn(
        self,
        agent_id: str,
        turn_id: str,
    ) -> DistributedStopTicket:
        """Fence a public turn and mark its one durable UUID generation."""

        agent_id = _required_identity(agent_id, "agent identity")
        turn_id = _required_opaque_identity(turn_id, "public turn identity")
        digest = _public_turn_digest(turn_id)
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
                "SELECT generation_id, 1 AS active "
                "FROM stop_active_invocations "
                "WHERE agent_id = ? AND public_turn_digest = ? "
                "UNION ALL "
                "SELECT generation_id, 0 AS active "
                "FROM stop_unresolved_invocations "
                "WHERE agent_id = ? AND public_turn_digest = ? "
                "ORDER BY generation_id",
                (agent_id, digest, agent_id, digest),
            )
            if len(rows) > 1:
                raise RuntimeError(
                    "distributed Stop public turn resolved to multiple generations"
                )
            generation_ids = tuple(str(row[0]) for row in rows)
            if rows and int(rows[0][1]) == 1:
                changed = await self._db.execute(
                    "UPDATE stop_active_invocations SET stop_requested = 1 "
                    "WHERE generation_id = ? AND agent_id = ? "
                    "AND public_turn_digest = ?",
                    (generation_ids[0], agent_id, digest),
                )
                if changed != 1:
                    raise RuntimeError(
                        "distributed Stop public turn changed inside its lock"
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
            cutoff_sql, cutoff_args = database_lease_cutoff_sql(
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
        cutoff_sql, cutoff_args = database_lease_cutoff_sql(
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
                cutoff_sql, cutoff_args = database_lease_cutoff_sql(
                    self._db, float(lease_seconds)
                )
                now_sql = database_now_sql(self._db)
                changed = await self._db.execute(
                    "INSERT INTO stop_unresolved_invocations ("
                    "generation_id, agent_id, turn_digest, public_turn_digest, "
                    "request_generation, owner_id, expired_at) "
                    "SELECT generation_id, agent_id, turn_digest, "
                    "public_turn_digest, request_generation, owner_id, "
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

    async def abandon_owner(self, owner_id: str) -> None:
        """Preserve every unsettled generation when an owner exits."""

        owner_id = _required_identity(owner_id, "owner identity")
        rows = await self._db.fetchall(
            "SELECT generation_id FROM stop_active_invocations "
            "WHERE owner_id = ? ORDER BY generation_id",
            (owner_id,),
        )
        for row in rows:
            await self.abandon(str(row[0]), owner_id)


class DistributedInvocationRegistry:
    """One process's live map and relay for shared Stop requests.

    Owner lease loss fences the owner's in-flight generations immediately; it
    is never a permanent state of the process (#3337). The fenced owner
    identity is retired, and ownership for new work is re-established under a
    fresh owner identity and epoch, so a stale lease can never be revived. If
    re-establishment fails, the registry stays fenced and reports why through
    :attr:`owner_fence_reason` until a later attempt succeeds.
    """

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
        self._owner_epoch = 1
        self._poll_seconds = float(poll_seconds)
        self._owner_lease_seconds = float(owner_lease_seconds)
        self._active: dict[str, _LocalGeneration] = {}
        self._by_local_generation: dict[tuple[int, str, int], str] = {}
        self._registration_lock = asyncio.Lock()
        self._registration_tasks: set[asyncio.Task[bool]] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_keys: set[tuple[int, str, int]] = set()
        self._completing_generation_ids: set[str] = set()
        self._relay_task: asyncio.Task[None] | None = None
        self._closing = False
        self._lease_lost = False
        self._fence_reason: str | None = None
        self._reacquisition_failure: str | None = None
        self._next_relay_reacquisition_at: float | None = None
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

    @property
    def owner_lifecycle_status(self) -> str:
        """Health-facing status of this process's current owner lease."""

        return "self_fenced" if self._lease_lost else "healthy"

    @property
    def owner_epoch(self) -> int:
        """Monotonic count of owner identities this registry has held."""

        return self._owner_epoch

    @property
    def owner_fence_reason(self) -> str | None:
        """Why the registry is fenced now, or ``None`` while it can admit.

        Names both the lease loss and, once one was attempted, why ownership
        could not yet be re-established.
        """

        if not self._lease_lost:
            return None
        reason = self._fence_reason or "owner lease lost"
        if self._reacquisition_failure is None:
            return f"epoch {self._owner_epoch}: {reason}"
        return (
            f"epoch {self._owner_epoch}: {reason}; re-establishment failed: "
            f"{self._reacquisition_failure}"
        )

    def _lease_owned_generation_ids(self) -> tuple[str, ...]:
        """Generations the current owner's lease must keep renewed."""

        owner_id = self._owner_id
        return tuple(
            generation_id
            for generation_id, target in self._active.items()
            if target.owner_id == owner_id
            and generation_id not in self._completing_generation_ids
        )

    @staticmethod
    def _agent_id(agent: object) -> str:
        agent_id = getattr(agent, "agent_id", None)
        return (
            agent_id
            if isinstance(agent_id, str) and agent_id.strip()
            else "local-agent"
        )

    async def _renew_owner_lease(
        self,
        generation_id: str,
        *,
        operation: str,
    ) -> _OwnerPoll:
        """Renew the whole durable inventory without reviving an old lease."""

        loop = asyncio.get_running_loop()
        poll_started = loop.time()
        try:
            polled = await self._store.poll_owner(
                self._owner_id,
                lease_seconds=self._owner_lease_seconds,
            )
        except Exception as error:
            self._fail_closed_owner(
                f"owner lease could not be renewed {operation} "
                f"({type(error).__name__})"
            )
            raise InvocationSelfFencedError(
                f"distributed Stop owner lease could not be renewed {operation}"
            ) from error
        # The database heartbeat occurs no earlier than ``poll_started``.
        # Retaining that lower bound (rather than response time) prevents a
        # delayed database response from making an already-expired lease look
        # fresh to this process.
        if loop.time() - poll_started >= self._owner_lease_seconds:
            self._fail_closed_owner(f"owner lease renewal was late {operation}")
            raise InvocationSelfFencedError(
                f"distributed Stop owner lease expired {operation}"
            )
        if generation_id not in polled.live_generation_ids:
            self._fail_closed_owner(f"durable generation was lost {operation}")
            raise InvocationSelfFencedError(
                f"distributed Stop durable generation was lost {operation}"
            )
        self._last_heartbeat_monotonic = poll_started
        return polled

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
                if self._lease_lost:
                    # The fenced owner's generations were already cancelled
                    # when the lease was lost. New work never inherits that
                    # identity: it waits for a fresh owner or is refused.
                    await self._reacquire_owner_lease()
                if self._lease_lost:
                    raise InvocationSelfFencedError(
                        "distributed Stop owner lease was lost and could not "
                        f"be re-established ({self.owner_fence_reason})"
                    )
                last_heartbeat = self._last_heartbeat_monotonic
                lease_owned_generation_ids = self._lease_owned_generation_ids()
                if (
                    lease_owned_generation_ids
                    and last_heartbeat is not None
                    and asyncio.get_running_loop().time() - last_heartbeat
                    >= self._owner_lease_seconds
                ):
                    self._fail_closed_owner(
                        "owner lease expired before admission"
                    )
                    raise InvocationSelfFencedError(
                        "distributed Stop owner lease expired before admission"
                    )
                existing_generation_id = self._by_local_generation.get(key)
                if existing_generation_id is not None:
                    existing = self._active.get(existing_generation_id)
                    if existing is None or existing.owner_id != self._owner_id:
                        # Admitted under an owner that was fenced and replaced.
                        # Its durable row is neither renewed nor polled by the
                        # successor, so nested work under that generation
                        # would run where a Stop cannot reach it.
                        raise InvocationSelfFencedError(
                            "distributed Stop owner lease was lost for this "
                            "request generation"
                        )
                    return True
                generation_id = uuid4().hex
                admission_started = asyncio.get_running_loop().time()
                owner_id = self._owner_id
                admitted = await self._store.register(
                    generation_id=generation_id,
                    agent_id=self._agent_id(agent),
                    turn_id=turn_id,
                    owner_id=owner_id,
                    request_generation=generation,
                )
                if not admitted:
                    return False
                still_lease_owned = self._lease_owned_generation_ids()
                had_other_lease_owned_work = bool(still_lease_owned)
                # The durable insert establishes cleanup ownership. Publish
                # that ownership locally before any lease-loss branch can
                # fail, so complete_soon can retry transient deletion errors.
                self._by_local_generation[key] = generation_id
                self._active[generation_id] = _LocalGeneration(
                    agent, turn_id, generation, owner_id
                )
                lease_expired_during_admission = bool(
                    lease_owned_generation_ids
                    and last_heartbeat is not None
                    and any(
                        owned_generation_id in still_lease_owned
                        for owned_generation_id in lease_owned_generation_ids
                    )
                    and asyncio.get_running_loop().time() - last_heartbeat
                    >= self._owner_lease_seconds
                )
                if lease_expired_during_admission:
                    # The durable insert awaited outside the process clock's
                    # lease window. A relay stalled on the same database may
                    # not have observed this yet, while another replica is
                    # already entitled to reap the older rows. The new row is
                    # provisional, never authority to revive that owner.
                    self._fail_closed_owner(
                        "owner lease expired during admission"
                    )
                if self._lease_lost:
                    self.complete_soon(agent, turn_id, generation)
                    raise InvocationSelfFencedError(
                        "distributed Stop owner lease was lost during admission"
                    )
                if not had_other_lease_owned_work:
                    # The insert starts an idle owner's lease, but its reply
                    # can be delayed past that lease while a peer marks and
                    # reaps the row. Never use response time as proof of fresh
                    # ownership, and re-read the durable row before cognition.
                    if (
                        asyncio.get_running_loop().time() - admission_started
                        >= self._owner_lease_seconds
                    ):
                        self._fail_closed_owner(
                            "first admission reply outlasted the owner lease"
                        )
                        self.complete_soon(agent, turn_id, generation)
                        raise InvocationSelfFencedError(
                            "distributed Stop owner lease expired during admission"
                        )
                    try:
                        polled = await self._renew_owner_lease(
                            generation_id,
                            operation="during admission",
                        )
                    except InvocationSelfFencedError:
                        self.complete_soon(agent, turn_id, generation)
                        raise
                    if generation_id in polled.stop_generation_ids:
                        self.complete_soon(agent, turn_id, generation)
                        return False
                return True

        owner = asyncio.create_task(publish(), name="distributed-stop-register")
        self._registration_tasks.add(owner)
        owner.add_done_callback(self._registration_tasks.discard)
        outcome = await await_owned_task(owner)
        return raise_owned_outcome(
            outcome, operation="distributed Stop invocation registration"
        )

    async def bind_public_turn(
        self,
        agent: object,
        turn_id: str,
        request_id: str,
        generation: int,
    ) -> bool:
        """Publish a public turn against its server-owned durable UUID."""

        if self._closing:
            raise RuntimeError("distributed Stop registry is closing")
        async with self._registration_lock:
            if self._lease_lost:
                raise InvocationSelfFencedError(
                    "distributed Stop owner lease was lost before turn binding"
                )
            key = (id(agent), request_id, generation)
            generation_id = self._by_local_generation.get(key)
            if (
                generation_id is None
                or generation_id in self._completing_generation_ids
            ):
                return False
            target = self._active.get(generation_id)
            if target is None or target.owner_id != self._owner_id:
                # Admitted under an owner that was fenced and replaced. Its
                # lease cannot be revived by the owner that succeeded it.
                raise InvocationSelfFencedError(
                    "distributed Stop owner lease was lost before turn binding"
                )
            last_heartbeat = self._last_heartbeat_monotonic
            if (
                last_heartbeat is None
                or asyncio.get_running_loop().time() - last_heartbeat
                >= self._owner_lease_seconds
            ):
                self._fail_closed_owner(
                    "owner lease expired before turn binding"
                )
                raise InvocationSelfFencedError(
                    "distributed Stop owner lease expired before turn binding"
                )
            polled = await self._renew_owner_lease(
                generation_id,
                operation="before turn binding",
            )
            if generation_id in polled.stop_generation_ids:
                return False
            bound = await self._store.bind_public_turn(
                generation_id=generation_id,
                owner_id=self._owner_id,
                agent_id=self._agent_id(agent),
                turn_id=turn_id,
            )
            last_heartbeat = self._last_heartbeat_monotonic
            if (
                last_heartbeat is None
                or asyncio.get_running_loop().time() - last_heartbeat
                >= self._owner_lease_seconds
            ):
                self._fail_closed_owner(
                    "owner lease expired during turn binding"
                )
                raise InvocationSelfFencedError(
                    "distributed Stop owner lease expired during turn binding"
                )
            return bound

    def complete_soon(
        self,
        agent: object,
        turn_id: str,
        generation: int,
        *,
        disposition: RequestCompletionDisposition = (
            RequestCompletionDisposition.COMPLETED
        ),
    ) -> None:
        """Own durable lifecycle settlement from synchronous cleanup."""

        if not isinstance(disposition, RequestCompletionDisposition):
            raise TypeError("distributed Stop completion disposition must be typed")

        key = (id(agent), turn_id, generation)
        generation_id = self._by_local_generation.get(key)
        if generation_id is None or key in self._cleanup_keys:
            return
        # Settle under the owner that wrote the durable row, even when that
        # owner has since been fenced and replaced by a newer epoch.
        owner_id = self._active[generation_id].owner_id
        self._cleanup_keys.add(key)
        # Durable deletion may become visible to a concurrent relay before
        # this task resumes to retire the local map. Mark the row synchronously
        # so that ordinary completion cannot look like owner lease loss.
        self._completing_generation_ids.add(generation_id)

        async def complete() -> None:
            try:
                while True:
                    try:
                        await self._store.settle(
                            generation_id,
                            owner_id,
                            disposition,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        logger.error(
                            "Distributed Stop completion failed; retrying (%s)",
                            type(error).__name__,
                            exc_info=(type(error), error, error.__traceback__),
                        )
                        if self._closing:
                            # Shutdown will conservatively move the still-active
                            # row to unresolved. A queued settlement always gets
                            # this first attempt, so a healthy store cannot turn
                            # known completion into indeterminacy merely because
                            # close won the event-loop race.
                            return
                        await asyncio.sleep(self._poll_seconds)
                        continue
                    self._by_local_generation.pop(key, None)
                    self._active.pop(generation_id, None)
                    self._completing_generation_ids.discard(generation_id)
                    if not self._lease_owned_generation_ids():
                        # A lease protects durable owner rows, not an idle
                        # process identity. The next admission starts a fresh
                        # lease generation instead of inheriting elapsed idle
                        # time from work that already completed.
                        self._last_heartbeat_monotonic = None
                    return
            finally:
                self._cleanup_keys.discard(key)
                if self._closing:
                    self._completing_generation_ids.discard(generation_id)

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

    async def request_turn(
        self,
        agent_id: str,
        turn_id: str,
    ) -> DistributedStopTicket:
        return await self._store.mark_turn(agent_id, turn_id)

    async def request_public_turn(
        self,
        agent_id: str,
        turn_id: str,
    ) -> DistributedStopTicket:
        return await self._store.mark_public_turn(agent_id, turn_id)

    async def request_agent(self, agent_id: str) -> DistributedStopTicket:
        return await self._store.mark_agent(agent_id)

    async def agent_has_unsettled_work(self, agent_id: str) -> bool:
        """Expose the durable fleet inventory without marking it for Stop."""

        return await self._store.agent_has_unsettled_work(agent_id)

    def cancel_local_ticket(
        self,
        ticket: DistributedStopTicket,
    ) -> tuple[tuple[str, int], ...]:
        """Cancel only local generations captured by a durable Stop ticket.

        Agent-wide Stop deliberately has no fence against work admitted after
        its database snapshot. Mapping the ticket's server-owned generation
        UUIDs back to this process's exact request generations preserves that
        same linearization point locally; re-reading an agent-wide live set
        here would widen Stop to later work on this replica only.
        """

        if not isinstance(ticket, DistributedStopTicket):
            raise TypeError("distributed Stop cancellation requires a typed ticket")
        cancelled: list[tuple[str, int]] = []
        for generation_id in ticket.generation_ids:
            target = self._active.get(generation_id)
            if target is None:
                continue
            cancel = getattr(target.agent, "cancel_current_request", None)
            if callable(cancel) and cancel(
                request_id=target.turn_id,
                generation=target.generation,
            ):
                cancelled.append((target.turn_id, target.generation))
        return tuple(cancelled)

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

    def _fail_closed_owner(
        self,
        reason: str = "owner lease lost",
        *,
        owner_id: str | None = None,
    ) -> None:
        """Fence every generation the current owner's lease protects.

        ``owner_id`` names the owner an asynchronous observation was made
        about. An observation about an owner that has already been replaced
        must not fence its successor.
        """

        if self._lease_lost:
            return
        if owner_id is not None and owner_id != self._owner_id:
            return
        self._lease_lost = True
        self._fence_reason = reason
        self._reacquisition_failure = None
        self._next_relay_reacquisition_at = None
        live_work = tuple(
            self._active[generation_id]
            for generation_id in self._lease_owned_generation_ids()
        )
        logger.error(
            "Distributed Stop owner FENCED (epoch %d): %s. %d in-flight "
            "generation(s) self-fenced; new admissions wait for ownership to "
            "be re-established under a fresh owner identity.",
            self._owner_epoch,
            reason,
            len(live_work),
        )
        for target in live_work:
            agent = target.agent
            self_fence = getattr(
                type(agent),
                "self_fence_current_request",
                None,
            )
            cancel = getattr(agent, "cancel_current_request", None)
            if callable(self_fence) or callable(cancel):
                try:
                    if callable(self_fence):
                        self_fence(
                            agent,
                            request_id=target.turn_id,
                            generation=target.generation,
                        )
                    else:
                        cancel(
                            request_id=target.turn_id,
                            generation=target.generation,
                        )
                except Exception:
                    logger.exception(
                        "Distributed Stop owner self-fence cancellation failed"
                    )

    def _record_reacquisition_failure(self, failure: str) -> None:
        if failure != self._reacquisition_failure:
            logger.error(
                "Distributed Stop owner lease could NOT be re-established "
                "(fenced epoch %d: %s): %s. Every new invocation on this "
                "process is refused until it is; /health/detailed reports "
                "distributed_invocation_owner. Operator action: restore the "
                "Stop database's availability and latency; the registry "
                "retries on its own.",
                self._owner_epoch,
                self._fence_reason,
                failure,
            )
        self._reacquisition_failure = failure

    async def _reacquire_owner_lease(self) -> None:
        """Re-establish ownership for new work under a fresh owner identity.

        Callers hold ``_registration_lock``. The fenced owner identity is
        retired, never renewed: its rows keep that identity, stop being
        heartbeated, and are either settled by their own cleanup or reaped by
        a peer. The replacement identity must prove the shared lease clock
        answers inside one lease before it may admit anything. An idle owner
        holds no durable rows, so that round trip is the whole acquisition;
        the first admission then starts the new owner's lease as usual.
        """

        if not self._lease_lost or self._closing:
            return
        loop = asyncio.get_running_loop()
        candidate = uuid4().hex
        started = loop.time()
        try:
            polled = await self._store.poll_owner(
                candidate,
                lease_seconds=self._owner_lease_seconds,
            )
        except Exception as error:
            self._record_reacquisition_failure(
                f"owner lease store unavailable ({type(error).__name__})"
            )
            return
        if loop.time() - started >= self._owner_lease_seconds:
            self._record_reacquisition_failure(
                "owner lease store answered after the lease window"
            )
            return
        if polled.live_generation_ids:
            self._record_reacquisition_failure(
                "fresh owner identity already holds durable generations"
            )
            return
        if not self._lease_lost or self._closing:
            return
        fenced_epoch = self._owner_epoch
        fenced_reason = self._fence_reason
        self._owner_id = candidate
        self._owner_epoch = fenced_epoch + 1
        self._last_heartbeat_monotonic = None
        self._lease_lost = False
        self._fence_reason = None
        self._reacquisition_failure = None
        self._next_relay_reacquisition_at = None
        logger.warning(
            "Distributed Stop owner lease re-established as epoch %d after "
            "epoch %d was fenced (%s); admitting new invocations.",
            self._owner_epoch,
            fenced_epoch,
            fenced_reason,
        )

    async def _relay_reacquire_owner_lease(self) -> None:
        """Converge a fenced registry without waiting for new traffic.

        Readiness reports a fenced owner as unhealthy, so a load balancer may
        stop sending the very admissions that would otherwise heal it.
        """

        loop = asyncio.get_running_loop()
        next_attempt = self._next_relay_reacquisition_at
        if next_attempt is not None and loop.time() < next_attempt:
            return
        async with self._registration_lock:
            await self._reacquire_owner_lease()
        if self._lease_lost:
            self._next_relay_reacquisition_at = (
                loop.time() + self._owner_lease_seconds
            )

    async def _relay(self) -> None:
        while not self._closing:
            if self._lease_lost:
                try:
                    await self._relay_reacquire_owner_lease()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.error(
                        "Distributed Stop owner re-establishment failed (%s)",
                        type(error).__name__,
                        exc_info=(type(error), error, error.__traceback__),
                    )
                await asyncio.sleep(self._poll_seconds)
                continue
            owner_id = self._owner_id
            lease_owned_generation_ids = self._lease_owned_generation_ids()
            if not lease_owned_generation_ids:
                await asyncio.sleep(self._poll_seconds)
                continue
            try:
                poll_started = asyncio.get_running_loop().time()
                polled = await self._store.poll_owner(
                    owner_id,
                    lease_seconds=self._owner_lease_seconds,
                )
                if owner_id != self._owner_id:
                    # The owner this poll renewed was fenced and replaced
                    # while it was in flight; its result describes no
                    # generation the current lease protects.
                    await asyncio.sleep(self._poll_seconds)
                    continue
                if (
                    asyncio.get_running_loop().time() - poll_started
                    >= self._owner_lease_seconds
                ):
                    self._fail_closed_owner(
                        "owner lease renewal was late in the relay",
                        owner_id=owner_id,
                    )
                else:
                    self._last_heartbeat_monotonic = poll_started
                live = set(polled.live_generation_ids)
                # ``polled`` can describe only the durable snapshot taken for
                # the inventory captured above. A concurrent admission may be
                # published locally after that SQL snapshot; comparing it to
                # this older reply would falsely declare the healthy owner
                # lease lost. Conversely, an originally captured row can enter
                # ordinary completion before the database snapshot and be
                # legitimately absent. Compare the captured inventory only
                # after subtracting rows that have since completed or begun
                # completion; fresh admissions are covered by the next poll.
                still_lease_owned = set(self._lease_owned_generation_ids())
                lease_owned_generation_ids = tuple(
                    generation_id
                    for generation_id in lease_owned_generation_ids
                    if generation_id in still_lease_owned
                )
                if any(
                    generation_id not in live
                    for generation_id in lease_owned_generation_ids
                ):
                    self._fail_closed_owner(
                        "durable generation was lost in the relay",
                        owner_id=owner_id,
                    )
                else:
                    for generation_id in polled.stop_generation_ids:
                        target = self._active.get(generation_id)
                        if target is None:
                            continue
                        cancel = getattr(
                            target.agent, "cancel_current_request", None
                        )
                        if callable(cancel):
                            cancel(
                                request_id=target.turn_id,
                                generation=target.generation,
                            )
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
                    self._fail_closed_owner(
                        "owner lease could not be renewed in the relay "
                        f"({type(error).__name__})",
                        owner_id=owner_id,
                    )
            await asyncio.sleep(self._poll_seconds)

    async def _abandon_owner_before(self, owner_id: str, deadline: float) -> None:
        remaining = deadline - asyncio.get_running_loop().time()
        try:
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(
                self._store.abandon_owner(owner_id),
                timeout=remaining,
            )
        except (asyncio.TimeoutError, Exception) as error:  # noqa: BLE001
            # Owner abandonment is recoverable: the lease expires and
            # reap_expired() retires the rows into the unresolved ledger.
            # Never hold teardown open for it.
            logger.warning(
                "Distributed Stop owner abandonment did not complete for "
                "the %s owner (%s)",
                "current" if owner_id == self._owner_id else "fenced",
                type(error).__name__,
            )

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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _DEFAULT_CLOSE_DRAIN_SECONDS
        for label, tasks in (
            ("registration", self._registration_tasks),
            ("completion", self._cleanup_tasks),
        ):
            # `while tasks:` alone never terminates if a task schedules more
            # work; bound it. Outcomes stay owned either way -- registration
            # tasks are gathered with return_exceptions and completion tasks
            # carry `_consume_cleanup` as a done-callback -- so an abandoned
            # tail cannot surface as a never-retrieved exception.
            while tasks:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    logger.warning(
                        "Distributed Stop close timed out after %.1fs with %d "
                        "%s task(s) still running; releasing teardown",
                        _DEFAULT_CLOSE_DRAIN_SECONDS,
                        len(tasks),
                        label,
                    )
                    break
                pending = tuple(tasks)
                done, _ = await asyncio.wait(pending, timeout=remaining)
                if not done:
                    continue
        # Fenced epochs keep their own identity on rows they still hold. Each
        # owner is attempted independently and concurrently under the one
        # shutdown deadline: one epoch's failure must not skip the others
        # (#3342), and one epoch that hangs must not starve the others of any
        # attempt (#3345). The current owner is started first.
        owner_ids = [self._owner_id] + sorted(
            {target.owner_id for target in self._active.values()}
            - {self._owner_id}
        )
        abandon_deadline = max(deadline, loop.time() + 1.0)
        await asyncio.gather(
            *(
                self._abandon_owner_before(owner_id, abandon_deadline)
                for owner_id in owner_ids
            )
        )
        self._active.clear()
        self._by_local_generation.clear()
        self._cleanup_keys.clear()
        self._completing_generation_ids.clear()


__all__ = [
    "DistributedInvocationRegistry",
    "DistributedInvocationStore",
    "DistributedStopTicket",
]
