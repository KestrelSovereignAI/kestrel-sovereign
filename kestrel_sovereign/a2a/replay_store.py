"""Shared replay-nonce reservation for A2A signed envelopes.

The in-process ``ReplayGuard`` in :mod:`kestrel_sovereign.a2a.envelope_signing`
is still useful as a fast path and as a degraded-mode fallback, but it cannot
see replays that land on another worker. This store provides the shared,
atomic reservation keyed by ``(sender, nonce)`` for deployments whose primary
database is shared across workers or instances.
"""
from __future__ import annotations

import asyncio
import secrets


class SharedReplayNonceStore:
    """DB-backed nonce reservation table for inbound signed A2A envelopes."""

    def __init__(self, db) -> None:
        self._db = db
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def ensure_table(self) -> None:
        """Create the replay table and expiry index if missing."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS a2a_replay_nonces (
                    sender TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    seen_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION NOT NULL,
                    envelope_digest TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (sender, nonce)
                )
                """
            )
            if getattr(self._db, "backend_type", "sqlite") == "postgres":
                await self._db.execute(
                    "ALTER TABLE a2a_replay_nonces ADD COLUMN IF NOT EXISTS "
                    "envelope_digest TEXT NOT NULL DEFAULT ''"
                )
            else:
                columns = await self._db.fetchall(
                    "PRAGMA table_info(a2a_replay_nonces)"
                )
                if "envelope_digest" not in {str(row[1]) for row in columns}:
                    try:
                        await self._db.execute(
                            "ALTER TABLE a2a_replay_nonces ADD COLUMN "
                            "envelope_digest TEXT NOT NULL DEFAULT ''"
                        )
                    except Exception:
                        # Two first-use workers can both observe the legacy
                        # shape before either ALTER commits. SQLite has no ADD
                        # COLUMN IF NOT EXISTS, so accept the losing worker's
                        # duplicate-column error only after re-reading the
                        # schema and proving that the migration landed.
                        migrated = await self._db.fetchall(
                            "PRAGMA table_info(a2a_replay_nonces)"
                        )
                        if "envelope_digest" not in {
                            str(row[1]) for row in migrated
                        }:
                            raise
            await self._db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_a2a_replay_nonces_expires
                ON a2a_replay_nonces(expires_at)
                """
            )
            self._initialized = True

    async def reserve(
        self,
        sender: str,
        nonce: str,
        *,
        now_ts: float,
        ttl_seconds: int,
    ) -> bool:
        """Reserve a nonce through the stable, pre-digest store protocol."""
        return await self._reserve(
            sender,
            nonce,
            now_ts=now_ts,
            ttl_seconds=ttl_seconds,
            envelope_digest="",
        )

    async def reserve_envelope(
        self,
        sender: str,
        nonce: str,
        *,
        now_ts: float,
        ttl_seconds: int,
        envelope_digest: str,
    ) -> bool:
        """Reserve a nonce and bind it to the canonical signed envelope."""
        return await self._reserve(
            sender,
            nonce,
            now_ts=now_ts,
            ttl_seconds=ttl_seconds,
            envelope_digest=envelope_digest,
        )

    async def _reserve(
        self,
        sender: str,
        nonce: str,
        *,
        now_ts: float,
        ttl_seconds: int,
        envelope_digest: str,
    ) -> bool:
        """Atomically reserve ``(sender, nonce)`` across workers.

        Returns ``True`` when the nonce was newly reserved and ``False`` when
        it is already present inside its TTL window. Expired rows are pruned
        opportunistically before the insert so a nonce can be reused after the
        full envelope validity window has elapsed.
        """
        if not nonce:
            return True

        await self.ensure_table()
        await self._db.execute(
            "DELETE FROM a2a_replay_nonces WHERE expires_at < ?",
            (now_ts,),
        )
        expires_at = now_ts + ttl_seconds
        backend_type = getattr(self._db, "backend_type", "sqlite")
        if backend_type == "postgres":
            sql = """
                INSERT INTO a2a_replay_nonces (
                    sender, nonce, seen_at, expires_at, envelope_digest
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (sender, nonce) DO NOTHING
            """
        else:
            sql = """
                INSERT OR IGNORE INTO a2a_replay_nonces
                    (sender, nonce, seen_at, expires_at, envelope_digest)
                VALUES (?, ?, ?, ?, ?)
            """
        affected = await self._db.execute(
            sql, (sender, nonce, now_ts, expires_at, envelope_digest)
        )
        return int(affected or 0) > 0

    async def matches_envelope(
        self,
        sender: str,
        nonce: str,
        *,
        envelope_digest: str,
    ) -> bool:
        """Return whether the durable reservation binds this exact envelope."""
        await self.ensure_table()
        row = await self._db.fetchone(
            "SELECT envelope_digest FROM a2a_replay_nonces "
            "WHERE sender = ? AND nonce = ?",
            (sender, nonce),
        )
        return bool(
            row
            and row[0]
            and secrets.compare_digest(str(row[0]), envelope_digest)
        )
