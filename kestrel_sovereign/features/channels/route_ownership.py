"""Durable exclusive ownership for externally provisioned channel routes.

An external host provisioner owns provider-specific webhook/polling setup, but
Core must still provide the durable conflict boundary: two agents/processes
must never attest that they own the same canonical bot route.  This store is
deliberately generic (``channel_type`` + ``canonical_route_identity``) so a
host can use it for Telegram today without coupling Core to an absent HTTP
provisioning endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import secrets
from typing import Any, AsyncIterator, Optional


@dataclass(frozen=True, slots=True)
class ChannelRouteClaim:
    """Opaque generation capability for one route-ownership claim.

    A same-agent reassertion replaces the live generation.  A stale teardown
    therefore cannot delete a newer claim that happens to have the same
    ``(channel_type, route, agent_id)`` tuple.
    """

    generation: str


class ChannelRouteOwnershipStore:
    """Atomically claim one canonical external route for one agent.

    The API intentionally never returns a competing agent identifier.  A host
    can learn whether *its requested agent* owns a route, release only that
    agent's claim, or reconcile/reassert the same claim; it cannot use this
    primitive to enumerate another user's channel assignments.

    ``database`` must be a durable Core database selected by the host.  In a
    multi-process deployment this is normally the shared PostgreSQL Core
    database; a host using SQLite must pass the one shared database file rather
    than separate per-agent files.  The primitive does not accept feature
    configuration, tokens, or user-provided ownership booleans as evidence.
    """

    TABLE = "channel_route_ownership"

    def __init__(self, database: Any) -> None:
        if database is None:
            raise ValueError("channel route ownership requires a durable database")
        self._database = database
        self._initialized = False

    @staticmethod
    def _require_identifier(value: object, *, label: str, maximum: int) -> str:
        if type(value) is not str or not value or value != value.strip():
            raise ValueError(f"{label} must be a non-empty trimmed string")
        if len(value) > maximum or "\x00" in value or any(
            ord(character) < 32 for character in value
        ):
            raise ValueError(f"{label} contains an invalid identifier")
        return value

    @classmethod
    def _validate_scope(
        cls,
        *,
        channel_type: object,
        canonical_route_identity: object,
        agent_id: object,
    ) -> tuple[str, str, str]:
        channel = cls._require_identifier(
            channel_type, label="channel_type", maximum=64
        )
        if (
            not channel.isascii()
            or channel != channel.lower()
            or not channel.replace("_", "").replace("-", "").isalnum()
        ):
            raise ValueError("channel_type must be a canonical lowercase identifier")
        route = cls._require_identifier(
            canonical_route_identity,
            label="canonical_route_identity",
            maximum=512,
        )
        agent = cls._require_identifier(agent_id, label="agent_id", maximum=512)
        return channel, route, agent

    @property
    def backend_type(self) -> str | None:
        value = getattr(self._database, "backend_type", None)
        if isinstance(value, str):
            return value
        backend = getattr(self._database, "backend", None)
        value = getattr(backend, "backend_type", None)
        return value if isinstance(value, str) else None

    @property
    def _transaction_backend(self) -> Any:
        """Use the raw backend when the AsyncDatabase facade exposes it."""

        return getattr(self._database, "backend", self._database)

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        """Serialize ownership mutations across tasks and database processes."""

        backend = self._transaction_backend
        transaction = getattr(backend, "transaction", None)
        if not callable(transaction):
            raise RuntimeError("channel route ownership database has no transaction API")
        if self.backend_type == "sqlite":
            # Establish the SQLite writer before either a schema check or an
            # UPSERT; ordinary BEGIN allows two initializers to inspect the
            # same absent table and race a later mutation.
            async with transaction(immediate=True):
                yield
            return
        async with transaction():
            yield

    async def _execute(self, query: str, params: tuple = ()) -> int:
        execute = getattr(self._database, "execute", None)
        if not callable(execute):
            raise RuntimeError("channel route ownership database has no execute API")
        return await execute(query, params)

    async def _fetchone(self, query: str, params: tuple = ()) -> Any:
        fetch = getattr(self._database, "fetchone", None)
        if not callable(fetch):
            fetch = getattr(self._database, "fetch_one", None)
        if not callable(fetch):
            raise RuntimeError("channel route ownership database has no fetch-one API")
        return await fetch(query, params)

    async def _fetchall(self, query: str, params: tuple = ()) -> list[Any]:
        fetch = getattr(self._database, "fetchall", None)
        if not callable(fetch):
            fetch = getattr(self._database, "fetch_all", None)
        if not callable(fetch):
            raise RuntimeError("channel route ownership database has no fetch-all API")
        return await fetch(query, params)

    async def initialize(self) -> None:
        """Create the shared ownership ledger once, safely under contention."""

        if self._initialized:
            return
        async with self._transaction():
            # PostgreSQL's relation locks cannot protect the absent-table
            # branch.  Hold one transaction-scoped lock across the *entire*
            # bootstrap/check/migrate sequence, then re-check catalog state
            # while it is held.  SQLite reaches the same point through the
            # BEGIN IMMEDIATE transaction selected by ``_transaction``.
            if self.backend_type == "postgres":
                await self._execute(
                    "SELECT pg_advisory_xact_lock("
                    "hashtext('kestrel.channel_route_ownership.bootstrap'))"
                )
            await self._execute(
                f"""CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    channel_type TEXT NOT NULL,
                    canonical_route_identity TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    generation TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (channel_type, canonical_route_identity)
                )"""
            )
            await self._ensure_generation_column()
        self._initialized = True

    async def _ensure_generation_column(self) -> None:
        """Add an ABA fence for databases created before generations existed.

        A pre-generation row remains non-releasable until its owner reasserts
        it.  That is intentional: manufacturing a token for a live legacy
        process would make a stale release unsafe during a rolling upgrade.
        """

        if self.backend_type == "sqlite":
            columns = await self._fetchall(f"PRAGMA table_info({self.TABLE})")
            if any(row[1] == "generation" for row in columns):
                return
        elif self.backend_type == "postgres":
            column = await self._fetchone(
                """SELECT 1 FROM pg_attribute
                   WHERE attrelid = to_regclass('channel_route_ownership')
                     AND attname = 'generation'
                     AND attnum > 0 AND NOT attisdropped"""
            )
            if column is not None:
                return
            # The advisory lock above is the primary serialization boundary.
            # Keep PostgreSQL's native idempotent form as secondary protection
            # for a manually repaired/rolling deployment that races an older
            # initializer outside this class.
            await self._execute(
                f"ALTER TABLE {self.TABLE} ADD COLUMN IF NOT EXISTS generation TEXT"
            )
            return
        else:
            raise RuntimeError(
                "channel route ownership supports only sqlite or postgres databases"
            )
        await self._execute(f"ALTER TABLE {self.TABLE} ADD COLUMN generation TEXT")

    async def claim(
        self,
        *,
        channel_type: str,
        canonical_route_identity: str,
        agent_id: str,
    ) -> Optional[ChannelRouteClaim]:
        """Create or reconcile an exclusive claim without exposing its owner.

        A same-agent reassertion replaces its opaque generation capability;
        a stale release cannot delete that replacement. A different-agent call
        affects zero rows and receives only ``None``. The single UPSERT makes
        that decision atomic across separate Core processes.
        """

        channel, route, agent = self._validate_scope(
            channel_type=channel_type,
            canonical_route_identity=canonical_route_identity,
            agent_id=agent_id,
        )
        await self.initialize()
        claim = ChannelRouteClaim(generation=secrets.token_urlsafe(24))
        async with self._transaction():
            affected = await self._execute(
                f"""INSERT INTO {self.TABLE}
                    (channel_type, canonical_route_identity, agent_id, generation)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(channel_type, canonical_route_identity) DO UPDATE
                    SET generation = excluded.generation,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE {self.TABLE}.agent_id = excluded.agent_id""",
                (channel, route, agent, claim.generation),
            )
        return claim if int(affected or 0) == 1 else None

    async def is_claimed_by(
        self,
        *,
        channel_type: str,
        canonical_route_identity: str,
        agent_id: str,
    ) -> bool:
        """Return only whether this exact agent owns the supplied route."""

        channel, route, agent = self._validate_scope(
            channel_type=channel_type,
            canonical_route_identity=canonical_route_identity,
            agent_id=agent_id,
        )
        await self.initialize()
        row = await self._fetchone(
            f"""SELECT 1 FROM {self.TABLE}
                WHERE channel_type = ? AND canonical_route_identity = ?
                  AND agent_id = ?""",
            (channel, route, agent),
        )
        return row is not None

    async def release(
        self,
        *,
        channel_type: str,
        canonical_route_identity: str,
        agent_id: str,
        claim: ChannelRouteClaim,
    ) -> bool:
        """Release one exact claim generation without revealing competing state."""

        channel, route, agent = self._validate_scope(
            channel_type=channel_type,
            canonical_route_identity=canonical_route_identity,
            agent_id=agent_id,
        )
        if not isinstance(claim, ChannelRouteClaim):
            raise TypeError("channel route release requires a ChannelRouteClaim")
        generation = self._require_identifier(
            claim.generation, label="claim generation", maximum=512
        )
        await self.initialize()
        async with self._transaction():
            affected = await self._execute(
                f"""DELETE FROM {self.TABLE}
                    WHERE channel_type = ? AND canonical_route_identity = ?
                      AND agent_id = ? AND generation = ?""",
                (channel, route, agent, generation),
            )
        return int(affected or 0) == 1
