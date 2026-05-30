"""
Service API Key Storage for Kestrel.

Encrypted storage for external service API keys (OpenRouter, OpenAI, etc.).
All keys are encrypted with agent-derived keys using the unified encryption module.

Every stored key belongs to an agent. No agent = no storage.
"""
import base64
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kestrel_sovereign.security.agent_encryption import encrypt
from kestrel_sovereign.security.legacy_decrypt import (
    decrypt_with_legacy_fallback as decrypt,
)
from kestrel_sovereign.security.exceptions import (
    KeyStorageError,
    KeyNotFoundError,
    KeyNotConfiguredError,
    DecryptionError,
)

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)

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
            (id, name, supports_sub_accounts)
            VALUES (?, ?, ?)
            """,
            (
                provider_id,
                provider_info["name"],
                1 if provider_info.get("supports_sub_accounts") else 0,
            )
        )

    async def store_key(
        self,
        provider_id: str,
        api_key: str,
        quota_limit: Optional[int] = None,
    ) -> str:
        """
        Store an encrypted API key for this agent.

        Args:
            provider_id: Service provider (openrouter, openai, etc.)
            api_key: The API key to store
            quota_limit: Optional usage limit

        Returns:
            Key ID
        """
        await self._ensure_provider(provider_id)

        # Encrypt using agent's service-keys derived key
        encrypted_bytes = encrypt(self._agent_did, "service-keys", api_key.encode("utf-8"))

        # Store as base64 for database compatibility
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode("ascii")

        key_id = str(uuid.uuid4())
        key_hash = self._hash_key(api_key)

        # Upsert (replace if exists for same agent+provider)
        await self._db.execute(
            """
            INSERT OR REPLACE INTO agent_service_keys
            (id, agent_did, provider_id, encrypted_key, key_hash,
             quota_limit, quota_used, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, 1, CURRENT_TIMESTAMP)
            """,
            (
                key_id,
                self._agent_did,
                provider_id,
                encrypted_b64,
                key_hash,
                quota_limit,
            )
        )

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
                created_at=datetime.fromisoformat(row[3]) if row[3] else datetime.utcnow(),
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
            True if key was deactivated
        """
        await self._db.execute(
            """
            UPDATE agent_service_keys
            SET is_active = 0
            WHERE agent_did = ? AND provider_id = ?
            """,
            (self._agent_did, provider_id)
        )

        logger.info(f"Deactivated key for agent={self._agent_did[:30]}..., provider={provider_id}")
        return True

    async def delete_key(self, provider_id: str) -> bool:
        """
        Delete key for a provider (hard delete, not just deactivate).

        Args:
            provider_id: Service provider

        Returns:
            True if key was deleted
        """
        await self._db.execute(
            """
            DELETE FROM agent_service_keys
            WHERE agent_did = ? AND provider_id = ?
            """,
            (self._agent_did, provider_id)
        )

        logger.info(f"Deleted key for agent={self._agent_did[:30]}..., provider={provider_id}")
        return True

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

        rows = await self._db.fetchall(
            """
            SELECT id, key_id, provider_id, operation, units_consumed,
                   cost_estimate_usd, recorded_at
            FROM service_key_usage
            WHERE key_id = ?
            AND recorded_at >= datetime('now', ?)
            ORDER BY recorded_at DESC
            """,
            (key_id, f'-{days} days')
        )

        return [
            UsageRecord(
                id=row[0],
                key_id=row[1],
                provider_id=row[2],
                operation=row[3],
                units_consumed=row[4],
                cost_estimate_usd=row[5],
                recorded_at=datetime.fromisoformat(row[6]) if row[6] else datetime.utcnow(),
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
