"""
Platform Service Key Storage for Kestrel.

Vending machine model: Platform-managed shared API key pool.
Keys encrypted with platform master key (PLATFORM_KEY_MASTER env var).
Companions pay via wallet + margin for usage.

Security Model:
- Keys encrypted with AES-256-GCM using platform master key
- Only platform admins can manage keys
- Rate limits per companion to prevent abuse
- Margin configuration for monetization
"""
import base64
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, TYPE_CHECKING

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

if TYPE_CHECKING:
    from asyncpg import Connection, Pool

from kestrel_sovereign.security.exceptions import (
    KeyStorageError as PlatformKeyStorageError,  # Alias for backward compat
    KeyNotFoundError,
    MasterKeyNotConfiguredError,
    DecryptionError,
)

logger = logging.getLogger(__name__)

# Constants
NONCE_LENGTH = 12  # 96 bits for AES-GCM
KEY_LENGTH = 32  # 256 bits for AES-256
SALT_LENGTH = 16  # Salt for key derivation from passphrase-style master key

# Exception classes imported from kestrel_sovereign.security.exceptions
# PlatformKeyStorageError, KeyNotFoundError, MasterKeyNotConfiguredError,
# DecryptionError are now available from the import above


@dataclass
class PlatformKeyInfo:
    """Information about a platform-managed key (no secret exposed)."""
    id: str
    provider_id: str
    display_name: Optional[str]
    is_active: bool
    rate_limit_rpm: Optional[int]
    rate_limit_per_companion: Optional[int]
    margin_pct: Decimal
    created_at: datetime


def _get_master_key() -> bytes:
    """
    Get the platform master encryption key.

    Reads from PLATFORM_KEY_MASTER environment variable.
    Can be a 32-byte hex string or a passphrase.

    Returns:
        32-byte key for AES-256-GCM

    Raises:
        MasterKeyNotConfiguredError: If env var not set
    """
    master = os.environ.get("PLATFORM_KEY_MASTER")
    if not master:
        raise MasterKeyNotConfiguredError(
            "PLATFORM_KEY_MASTER environment variable not set. "
            "Platform keys cannot be encrypted/decrypted."
        )

    # Try to decode as hex first (64 hex chars = 32 bytes)
    if len(master) == 64:
        try:
            return bytes.fromhex(master)
        except ValueError:
            pass

    # Otherwise derive from passphrase using fixed salt
    # Note: Fixed salt is acceptable here because the key is not user-derived
    # and we need deterministic encryption across server restarts
    fixed_salt = b"kestrel_platform_key_v1"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=fixed_salt,
        iterations=100_000,  # Lower than user keys since this is server-side
        backend=default_backend(),
    )
    return kdf.derive(master.encode("utf-8"))


def _encrypt_key(api_key: str) -> tuple[bytes, bytes]:
    """
    Encrypt an API key with the platform master key.

    Args:
        api_key: The plaintext API key

    Returns:
        Tuple of (ciphertext, nonce)

    Raises:
        MasterKeyNotConfiguredError: If master key not set
    """
    master_key = _get_master_key()
    nonce = os.urandom(NONCE_LENGTH)

    aesgcm = AESGCM(master_key)
    ciphertext = aesgcm.encrypt(nonce, api_key.encode("utf-8"), None)

    return ciphertext, nonce


def _decrypt_key(ciphertext: bytes, nonce: bytes) -> str:
    """
    Decrypt an API key with the platform master key.

    Args:
        ciphertext: Encrypted API key
        nonce: Nonce used for encryption

    Returns:
        Decrypted API key

    Raises:
        MasterKeyNotConfiguredError: If master key not set
        DecryptionError: If decryption fails
    """
    try:
        master_key = _get_master_key()
        aesgcm = AESGCM(master_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    except MasterKeyNotConfiguredError:
        raise
    except Exception as e:
        raise DecryptionError(
            "Failed to decrypt platform key. Check PLATFORM_KEY_MASTER."
        ) from e


def _hash_key(api_key: str) -> str:
    """Create a hash of the API key for verification without decryption."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]


class PlatformKeyStorage:
    """
    Platform-managed vending machine key pool.

    Provides shared API keys to companions for a fee (margin).
    Keys encrypted with platform master key.
    """

    def __init__(self, pool: "Pool"):
        """
        Initialize platform key storage.

        Args:
            pool: asyncpg connection pool
        """
        self._pool = pool

    async def store_key(
        self,
        provider_id: str,
        api_key: str,
        display_name: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        rate_limit_per_companion: Optional[int] = None,
        margin_pct: Decimal = Decimal("0.15"),
    ) -> str:
        """
        Store an encrypted platform API key.

        Args:
            provider_id: Service provider (openrouter, openai, etc.)
            api_key: The API key to store
            display_name: Friendly name (e.g., "OpenRouter Pool")
            rate_limit_rpm: Total requests per minute for this key
            rate_limit_per_companion: Per-companion rate limit
            margin_pct: Margin percentage (default 15%)

        Returns:
            Key ID

        Raises:
            MasterKeyNotConfiguredError: If platform master key not set
            PlatformKeyStorageError: If storage fails
        """
        # Encrypt the key
        ciphertext, nonce = _encrypt_key(api_key)

        key_id = str(uuid.uuid4())
        key_hash = _hash_key(api_key)

        # Base64 encode for storage
        encrypted_b64 = base64.b64encode(ciphertext).decode("ascii")

        async with self._pool.acquire() as conn:
            # Upsert (replace if exists for same provider)
            await conn.execute(
                """
                INSERT INTO platform_service_keys
                (id, provider_id, encrypted_key, key_nonce, key_hash,
                 display_name, rate_limit_rpm, rate_limit_per_companion,
                 margin_pct, is_active, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, TRUE, NOW())
                ON CONFLICT (provider_id) DO UPDATE SET
                    encrypted_key = EXCLUDED.encrypted_key,
                    key_nonce = EXCLUDED.key_nonce,
                    key_hash = EXCLUDED.key_hash,
                    display_name = COALESCE(EXCLUDED.display_name, platform_service_keys.display_name),
                    rate_limit_rpm = EXCLUDED.rate_limit_rpm,
                    rate_limit_per_companion = EXCLUDED.rate_limit_per_companion,
                    margin_pct = EXCLUDED.margin_pct,
                    is_active = TRUE,
                    updated_at = NOW()
                """,
                key_id,
                provider_id,
                encrypted_b64,
                nonce,
                key_hash,
                display_name,
                rate_limit_rpm,
                rate_limit_per_companion,
                margin_pct,
            )

        logger.info(f"Stored platform key for provider={provider_id}")
        return key_id

    async def get_key(self, provider_id: str) -> str:
        """
        Get decrypted API key for a provider.

        Args:
            provider_id: Service provider

        Returns:
            Decrypted API key

        Raises:
            KeyNotFoundError: If no key found for provider
            DecryptionError: If decryption fails
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT encrypted_key, key_nonce
                FROM platform_service_keys
                WHERE provider_id = $1 AND is_active = TRUE
                """,
                provider_id,
            )

        if not row:
            raise KeyNotFoundError(
                f"No platform key configured for provider '{provider_id}'"
            )

        encrypted_b64 = row["encrypted_key"]
        nonce = row["key_nonce"]

        # Decode from base64
        ciphertext = base64.b64decode(encrypted_b64)

        # Decrypt
        return _decrypt_key(ciphertext, bytes(nonce))

    async def has_key(self, provider_id: str) -> bool:
        """
        Check if platform has a key configured for a provider.

        Args:
            provider_id: Service provider

        Returns:
            True if key exists
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM platform_service_keys
                WHERE provider_id = $1 AND is_active = TRUE
                """,
                provider_id,
            )
        return row is not None

    async def list_keys(self) -> List[PlatformKeyInfo]:
        """
        List all platform keys (no secrets exposed).

        Returns:
            List of key info objects
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, provider_id, display_name, is_active,
                       rate_limit_rpm, rate_limit_per_companion,
                       margin_pct, created_at
                FROM platform_service_keys
                ORDER BY created_at DESC
                """
            )

        return [
            PlatformKeyInfo(
                id=str(row["id"]),
                provider_id=row["provider_id"],
                display_name=row["display_name"],
                is_active=row["is_active"],
                rate_limit_rpm=row["rate_limit_rpm"],
                rate_limit_per_companion=row["rate_limit_per_companion"],
                margin_pct=row["margin_pct"] or Decimal("0.15"),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def get_key_info(self, provider_id: str) -> Optional[PlatformKeyInfo]:
        """
        Get metadata for a specific provider's key.

        Args:
            provider_id: Service provider

        Returns:
            Key info or None if not found
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, provider_id, display_name, is_active,
                       rate_limit_rpm, rate_limit_per_companion,
                       margin_pct, created_at
                FROM platform_service_keys
                WHERE provider_id = $1
                """,
                provider_id,
            )

        if not row:
            return None

        return PlatformKeyInfo(
            id=str(row["id"]),
            provider_id=row["provider_id"],
            display_name=row["display_name"],
            is_active=row["is_active"],
            rate_limit_rpm=row["rate_limit_rpm"],
            rate_limit_per_companion=row["rate_limit_per_companion"],
            margin_pct=row["margin_pct"] or Decimal("0.15"),
            created_at=row["created_at"],
        )

    async def deactivate_key(self, provider_id: str) -> bool:
        """
        Deactivate key for a provider (soft delete).

        Args:
            provider_id: Service provider

        Returns:
            True if key was deactivated
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE platform_service_keys
                SET is_active = FALSE, updated_at = NOW()
                WHERE provider_id = $1
                """,
                provider_id,
            )

        logger.info(f"Deactivated platform key for provider={provider_id}")
        return "UPDATE" in result

    async def delete_key(self, provider_id: str) -> bool:
        """
        Delete key for a provider (hard delete).

        Args:
            provider_id: Service provider

        Returns:
            True if key was deleted
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM platform_service_keys
                WHERE provider_id = $1
                """,
                provider_id,
            )

        logger.info(f"Deleted platform key for provider={provider_id}")
        return "DELETE" in result

    async def update_margin(
        self,
        provider_id: str,
        margin_pct: Decimal,
    ) -> bool:
        """
        Update margin percentage for a provider.

        Args:
            provider_id: Service provider
            margin_pct: New margin percentage (e.g., 0.15 for 15%)

        Returns:
            True if updated
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE platform_service_keys
                SET margin_pct = $2, updated_at = NOW()
                WHERE provider_id = $1
                """,
                provider_id,
                margin_pct,
            )

        logger.info(f"Updated margin for provider={provider_id} to {margin_pct}")
        return "UPDATE" in result

    async def update_rate_limits(
        self,
        provider_id: str,
        rate_limit_rpm: Optional[int] = None,
        rate_limit_per_companion: Optional[int] = None,
    ) -> bool:
        """
        Update rate limits for a provider.

        Args:
            provider_id: Service provider
            rate_limit_rpm: Total requests per minute (None to keep current)
            rate_limit_per_companion: Per-companion limit (None to keep current)

        Returns:
            True if updated
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE platform_service_keys
                SET
                    rate_limit_rpm = COALESCE($2, rate_limit_rpm),
                    rate_limit_per_companion = COALESCE($3, rate_limit_per_companion),
                    updated_at = NOW()
                WHERE provider_id = $1
                """,
                provider_id,
                rate_limit_rpm,
                rate_limit_per_companion,
            )

        logger.info(f"Updated rate limits for provider={provider_id}")
        return "UPDATE" in result

    async def get_margin(self, provider_id: str) -> Decimal:
        """
        Get margin percentage for billing calculation.

        Args:
            provider_id: Service provider

        Returns:
            Margin percentage (e.g., 0.15 for 15%)
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT margin_pct FROM platform_service_keys
                WHERE provider_id = $1 AND is_active = TRUE
                """,
                provider_id,
            )

        if not row or row["margin_pct"] is None:
            return Decimal("0.15")  # Default 15% margin

        return row["margin_pct"]
