"""
Service API Key Storage for Kestrel.

Encrypted storage for external service API keys (OpenRouter, OpenAI, etc.).
All keys are encrypted with agent-derived keys using the unified encryption module.

Every stored key belongs to an agent. No agent = no storage.
"""
import base64
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, List, Optional, TYPE_CHECKING

from kestrel_sovereign.security.agent_encryption import encrypt
from kestrel_sovereign.security.legacy_decrypt import (
    decrypt_with_legacy_fallback as decrypt,
)
from kestrel_sovereign.security.exceptions import (
    KeyStorageError,
    KeyNotConfiguredError,
)
from kestrel_sovereign.storage.db.interface import QueryError

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


def _is_unique_violation(err: Exception) -> bool:
    """Return True if a wrapped DB error is a UNIQUE-constraint violation.

    Backend-portable: the SQLite backend surfaces ``UNIQUE constraint failed``
    and PostgreSQL (asyncpg) surfaces ``duplicate key value violates unique
    constraint`` — both wrapped in ``QueryError``. We match on the substring
    common to each rather than importing backend-specific driver exceptions.
    """
    message = str(err).lower()
    return "unique constraint" in message or "duplicate key value" in message


def _as_datetime(value: Any) -> datetime:
    """Coerce a DB timestamp value to a ``datetime``.

    Backend-portable: PostgreSQL (asyncpg) returns native ``datetime`` objects
    for TIMESTAMP columns, while SQLite returns ISO-format strings. Accept
    either, falling back to ``utcnow()`` for NULLs. Previously this code called
    ``datetime.fromisoformat()`` unconditionally, which raises
    ``TypeError: fromisoformat: argument must be str`` on Postgres.
    """
    if value is None:
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


# Exception classes imported from kestrel_sovereign.security.exceptions
# KeyStorageError, KeyNotFoundError, KeyNotConfiguredError, DecryptionError
# are now available from the import above


@dataclass
class ServiceKeyInfo:
    """Information about a stored service key (no secret exposed)."""
    id: str
    provider_id: str
    is_active: bool
    created_at: datetime
    quota_limit: Optional[int] = None
    quota_used: int = 0


@dataclass
class UsageRecord:
    """Record of key usage for tracking/billing."""
    id: str
    key_id: str
    provider_id: str
    operation: str
    units_consumed: int
    recorded_at: datetime
    cost_estimate_usd: Optional[float] = None


# Known service providers
KNOWN_PROVIDERS = {
    "openrouter": {"name": "OpenRouter", "supports_sub_accounts": True},
    "openai": {"name": "OpenAI", "supports_sub_accounts": False},
    "anthropic": {"name": "Anthropic", "supports_sub_accounts": False},
    "lighthouse": {"name": "Lighthouse (IPFS/Filecoin)", "supports_sub_accounts": False},
    "github": {"name": "GitHub", "supports_sub_accounts": False},
    "runpod": {"name": "RunPod", "supports_sub_accounts": True},
    "vastai": {"name": "Vast.ai", "supports_sub_accounts": False},
}


class ServiceKeyStorage:
    """
    Agent-scoped encrypted storage for service API keys.

    All operations require an agent_did. Keys are encrypted using the agent's
    derived encryption key from the unified encryption module.
    """

    def __init__(self, db: "AsyncDatabase", agent_did: str):
        """
        Initialize service key storage for an agent.

        Args:
            db: AsyncDatabase instance for persistence
            agent_did: Agent's DID (required)

        Raises:
            ValueError: If agent_did is empty
        """
        if not agent_did:
            raise ValueError("agent_did is required for ServiceKeyStorage")

        self._db = db
        self._agent_did = agent_did

    def _hash_key(self, api_key: str) -> str:
        """Create a hash of the API key for lookup without decryption."""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]

    async def _ensure_provider(self, provider_id: str) -> None:
        """Ensure provider exists in database."""
        provider_info = KNOWN_PROVIDERS.get(provider_id, {"name": provider_id})

        await self._db.execute(
            """
            INSERT OR IGNORE INTO service_providers
            (id, name, supports_sub_accounts, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                provider_id,
                provider_info["name"],
                1 if provider_info.get("supports_sub_accounts") else 0,
            )
        )

    async def _key_exists(self, provider_id: str) -> bool:
        """Return True if any key row exists for this agent+provider.

        Deliberately ignores ``is_active`` so an inactive/deactivated key still
        counts as "exists" — a fresh ``store_key`` over a deactivated row would
        otherwise silently replace it, bypassing rotation approval (F196).
        """
        rows = await self._db.fetchall(
            """
            SELECT 1 FROM agent_service_keys
            WHERE agent_did = ? AND provider_id = ?
            """,
            (self._agent_did, provider_id)
        )
        return len(rows) > 0

    async def store_key(
        self,
        provider_id: str,
        api_key: str,
        quota_limit: Optional[int] = None,
        replace: bool = False,
        reactivate_inactive: bool = False,
    ) -> str:
        """
        Store an encrypted API key for this agent.

        Insert-only by default: if a key already exists for this agent+provider
        (active OR inactive), this raises ``KeyStorageError`` unless ``replace``
        or ``reactivate_inactive`` permits the write. ``replace=True`` is the
        approval-gated rotation path (``rotate_service_key``) and may overwrite
        an ACTIVE key. ``reactivate_inactive=True`` is the host-internal
        PayerResolver re-provision path and may overwrite ONLY an inactive
        tombstone (an active row still raises). This makes storage the single
        enforcement point that prevents a model-controlled ``add_service_key``
        from silently rotating a live credential without constitutional approval
        (F196), while still letting the host re-provision a removed key.

        Args:
            provider_id: Service provider (openrouter, openai, etc.)
            api_key: The API key to store
            quota_limit: Optional usage limit
            replace: Allow overwriting an existing key (rotation only)
            reactivate_inactive: For the host-internal PayerResolver re-provision
                path. Overwrite ONLY an inactive (removed) tombstone row and
                reactivate it; an ACTIVE row still raises ``KeyStorageError``
                (never silently clobbered). Race-safe: the reactivation UPDATE is
                scoped ``WHERE is_active = 0`` and the fall-through INSERT hits
                ``UNIQUE(agent_did, provider_id)`` if an active row appears, so a
                concurrent ``add_service_key``/rotation can never be overwritten.

        Returns:
            Key ID

        Raises:
            KeyStorageError: If a key already exists and neither ``replace`` nor
                (for an *inactive* row) ``reactivate_inactive`` permits it.
        """
        await self._ensure_provider(provider_id)

        # Encrypt using agent's service-keys derived key
        encrypted_bytes = encrypt(self._agent_did, "service-keys", api_key.encode("utf-8"))

        # Store as base64 for database compatibility
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode("ascii")

        key_id = str(uuid.uuid4())
        key_hash = self._hash_key(api_key)

        params = (
            key_id,
            self._agent_did,
            provider_id,
            encrypted_b64,
            key_hash,
            quota_limit,
        )
        columns = (
            "(id, agent_did, provider_id, encrypted_key, key_hash, "
            "quota_limit, quota_used, is_active, created_at)"
        )
        values = "VALUES (?, ?, ?, ?, ?, ?, 0, 1, CURRENT_TIMESTAMP)"

        if replace:
            # Approval-gated rotation: INSERT OR REPLACE upserts over the
            # existing UNIQUE(agent_did, provider_id) row (only reachable
            # via rotate_service_key).
            await self._db.execute(
                f"INSERT OR REPLACE INTO agent_service_keys {columns} {values}",
                params,
            )
        elif reactivate_inactive:
            # Host-internal re-provision: reactivate ONLY an inactive tombstone,
            # never an active row. Scoped WHERE is_active = 0 so a concurrently
            # added/rotated ACTIVE key is untouched. If no inactive row matched
            # (rows_affected == 0), fall through to a plain INSERT — which
            # succeeds when there's no row at all, or raises KeyStorageError on
            # the UNIQUE constraint if an active row exists (F196 preserved).
            from kestrel_sovereign.storage.async_conversation_store import (
                _rows_affected,
            )
            reactivated = await self._db.execute_commit(
                "UPDATE agent_service_keys "
                "SET id = ?, encrypted_key = ?, key_hash = ?, quota_limit = ?, "
                "    quota_used = 0, is_active = 1, created_at = CURRENT_TIMESTAMP "
                "WHERE agent_did = ? AND provider_id = ? AND is_active = 0",
                (key_id, encrypted_b64, key_hash, quota_limit,
                 self._agent_did, provider_id),
            )
            if _rows_affected(reactivated) == 0:
                try:
                    await self._db.execute(
                        f"INSERT INTO agent_service_keys {columns} {values}",
                        params,
                    )
                except QueryError as e:
                    if _is_unique_violation(e):
                        raise KeyStorageError(
                            f"active key for {provider_id} exists — "
                            f"use rotate_service_key"
                        ) from e
                    raise
        else:
            # Insert-only add. The preflight check gives a friendly error for
            # the common serial case, but is NOT the enforcement point — two
            # concurrent adds could both observe no row. Enforcement is the
            # plain INSERT hitting UNIQUE(agent_did, provider_id): the loser of
            # the race raises, which we translate to KeyStorageError. This
            # closes the F196 approval bypass under real concurrency.
            if await self._key_exists(provider_id):
                raise KeyStorageError(
                    f"key for {provider_id} exists — use rotate_service_key"
                )
            try:
                await self._db.execute(
                    f"INSERT INTO agent_service_keys {columns} {values}",
                    params,
                )
            except QueryError as e:
                if _is_unique_violation(e):
                    raise KeyStorageError(
                        f"key for {provider_id} exists — use rotate_service_key"
                    ) from e
                raise

        logger.info(f"Stored key for agent={self._agent_did[:30]}..., provider={provider_id}")
        return key_id

    async def get_key(self, provider_id: str) -> str:
        """
        Get decrypted API key for a provider.

        Args:
            provider_id: Service provider

        Returns:
            Decrypted API key

        Raises:
            KeyNotConfiguredError: If no key found for provider
            DecryptionError: If decryption fails
        """
        rows = await self._db.fetchall(
            """
            SELECT encrypted_key FROM agent_service_keys
            WHERE agent_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._agent_did, provider_id)
        )

        if not rows or not rows[0][0]:
            raise KeyNotConfiguredError(
                f"No API key configured for provider '{provider_id}' and agent '{self._agent_did[:30]}...'"
            )

        encrypted_b64 = rows[0][0]
        encrypted_bytes = base64.b64decode(encrypted_b64)

        # Decrypt using agent's service-keys derived key
        plaintext_bytes = decrypt(self._agent_did, "service-keys", encrypted_bytes)
        return plaintext_bytes.decode("utf-8")

    async def has_key(self, provider_id: str) -> bool:
        """
        Check if agent has a key configured for a provider.

        Args:
            provider_id: Service provider

        Returns:
            True if key exists
        """
        rows = await self._db.fetchall(
            """
            SELECT 1 FROM agent_service_keys
            WHERE agent_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._agent_did, provider_id)
        )
        return len(rows) > 0

    async def list_keys(self) -> List[ServiceKeyInfo]:
        """
        List all configured keys for this agent (no secrets exposed).

        Returns:
            List of key info objects
        """
        rows = await self._db.fetchall(
            """
            SELECT id, provider_id, is_active, created_at, quota_limit, quota_used
            FROM agent_service_keys
            WHERE agent_did = ?
            ORDER BY created_at DESC
            """,
            (self._agent_did,)
        )

        return [
            ServiceKeyInfo(
                id=row[0],
                provider_id=row[1],
                is_active=bool(row[2]),
                created_at=_as_datetime(row[3]),
                quota_limit=row[4],
                quota_used=row[5] or 0,
            )
            for row in rows
        ]

    async def deactivate_key(self, provider_id: str) -> bool:
        """
        Deactivate key for a provider.

        Args:
            provider_id: Service provider

        Returns:
            True if an ACTIVE row was actually deactivated, False if no active
            key existed for this agent+provider (nothing was transitioned). The
            ``is_active = 1`` predicate ensures a repeated remove of an
            already-inactive key reports no-op rather than false success.
        """
        affected = await self._db.execute(
            """
            UPDATE agent_service_keys
            SET is_active = 0
            WHERE agent_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._agent_did, provider_id)
        )

        logger.info(
            f"Deactivated key for agent={self._agent_did[:30]}..., "
            f"provider={provider_id} (rows affected={affected})"
        )
        return bool(affected)

    async def delete_key(self, provider_id: str) -> bool:
        """
        Delete key for a provider (hard delete, not just deactivate).

        Args:
            provider_id: Service provider

        Returns:
            True if a row was actually deleted, False if no key existed for
            this agent+provider (nothing was affected).
        """
        affected = await self._db.execute(
            """
            DELETE FROM agent_service_keys
            WHERE agent_did = ? AND provider_id = ?
            """,
            (self._agent_did, provider_id)
        )

        logger.info(
            f"Deleted key for agent={self._agent_did[:30]}..., "
            f"provider={provider_id} (rows affected={affected})"
        )
        return bool(affected)

    async def record_usage(
        self,
        provider_id: str,
        operation: str,
        units: int = 1,
        cost_estimate: Optional[float] = None,
    ) -> None:
        """
        Record usage and update quota counters.

        Args:
            provider_id: Service provider
            operation: Operation type (inference, upload, etc.)
            units: Units consumed
            cost_estimate: Estimated cost in USD
        """
        # Get key ID
        rows = await self._db.fetchall(
            """
            SELECT id FROM agent_service_keys
            WHERE agent_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._agent_did, provider_id)
        )

        if not rows:
            logger.warning(f"No key found for usage tracking: agent={self._agent_did[:30]}..., provider={provider_id}")
            return

        key_id = rows[0][0]

        # Update quota
        await self._db.execute(
            """
            UPDATE agent_service_keys
            SET quota_used = quota_used + ?
            WHERE id = ?
            """,
            (units, key_id)
        )

        # Record usage
        usage_id = str(uuid.uuid4())
        await self._db.execute(
            """
            INSERT INTO service_key_usage
            (id, key_id, key_type, provider_id, operation, units_consumed, cost_estimate_usd, recorded_at)
            VALUES (?, ?, 'agent', ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (usage_id, key_id, provider_id, operation, units, cost_estimate)
        )

    async def get_usage(self, provider_id: str, days: int = 30) -> List[UsageRecord]:
        """
        Get usage records for a provider.

        Args:
            provider_id: Service provider
            days: Number of days to look back

        Returns:
            List of usage records
        """
        # Get key ID
        key_rows = await self._db.fetchall(
            """
            SELECT id FROM agent_service_keys
            WHERE agent_did = ? AND provider_id = ?
            """,
            (self._agent_did, provider_id)
        )

        if not key_rows:
            return []

        key_id = key_rows[0][0]

        # Compute the lookback cutoff in Python rather than with SQLite's
        # ``datetime('now', ?)`` modifier, which does not exist in PostgreSQL
        # (``function datetime(unknown, unknown) does not exist``). A datetime
        # parameter compares correctly against TIMESTAMP (PG) and the ISO-string
        # ``recorded_at`` written via CURRENT_TIMESTAMP (SQLite).
        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = await self._db.fetchall(
            """
            SELECT id, key_id, provider_id, operation, units_consumed,
                   cost_estimate_usd, recorded_at
            FROM service_key_usage
            WHERE key_id = ?
            AND recorded_at >= ?
            ORDER BY recorded_at DESC
            """,
            (key_id, cutoff)
        )

        return [
            UsageRecord(
                id=row[0],
                key_id=row[1],
                provider_id=row[2],
                operation=row[3],
                units_consumed=row[4],
                cost_estimate_usd=row[5],
                recorded_at=_as_datetime(row[6]),
            )
            for row in rows
        ]

    async def check_quota(self, provider_id: str, units: int = 1) -> bool:
        """
        Check if quota allows operation.

        Args:
            provider_id: Service provider
            units: Units to consume

        Returns:
            True if quota allows operation (or no quota set)
        """
        rows = await self._db.fetchall(
            """
            SELECT quota_limit, quota_used FROM agent_service_keys
            WHERE agent_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._agent_did, provider_id)
        )

        if not rows:
            return False  # No key = no access

        limit, used = rows[0]
        if limit is None:
            return True  # No quota limit

        return (used or 0) + units <= limit
