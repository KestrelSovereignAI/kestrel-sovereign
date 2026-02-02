"""
User BYOK (Bring Your Own Key) Storage for Kestrel.

Stores user's own API keys encrypted with their passphrase.
Keys are shared across all user's companions.
User pays provider directly - no wallet debit.

Security Model:
- Keys encrypted with PBKDF2-derived AES-256-GCM key
- Salt stored per-key for derivation
- Platform cannot decrypt without user's passphrase
- If user forgets passphrase, keys must be re-added
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
    KeyStorageError as UserKeyStorageError,  # Alias for backward compat
    KeyNotFoundError,
    DecryptionError,
    PassphraseRequiredError,
)

logger = logging.getLogger(__name__)

# PBKDF2 parameters (OWASP 2024 recommendations)
PBKDF2_ITERATIONS = 600_000  # High iteration count for passphrase-based keys
SALT_LENGTH = 32  # 256 bits
NONCE_LENGTH = 12  # 96 bits for AES-GCM
KEY_LENGTH = 32  # 256 bits for AES-256

# Exception classes imported from kestrel_sovereign.security.exceptions
# UserKeyStorageError, KeyNotFoundError, DecryptionError, PassphraseRequiredError
# are now available from the import above


@dataclass
class UserKeyInfo:
    """Information about a user's stored key (no secret exposed)."""
    id: str
    user_id: str
    provider_id: str
    display_name: Optional[str]
    is_active: bool
    quota_limit: Optional[int]
    quota_used: int
    created_at: datetime


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """
    Derive encryption key from passphrase using PBKDF2.

    Args:
        passphrase: User's passphrase
        salt: Random salt (stored with the key)

    Returns:
        32-byte key for AES-256-GCM
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend(),
    )
    return kdf.derive(passphrase.encode("utf-8"))


def _encrypt_key(api_key: str, passphrase: str) -> tuple[bytes, bytes, bytes]:
    """
    Encrypt an API key with the user's passphrase.

    Args:
        api_key: The plaintext API key
        passphrase: User's passphrase

    Returns:
        Tuple of (ciphertext, salt, nonce)
    """
    salt = os.urandom(SALT_LENGTH)
    nonce = os.urandom(NONCE_LENGTH)
    key = _derive_key(passphrase, salt)

    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, api_key.encode("utf-8"), None)

    return ciphertext, salt, nonce


def _decrypt_key(ciphertext: bytes, salt: bytes, nonce: bytes, passphrase: str) -> str:
    """
    Decrypt an API key with the user's passphrase.

    Args:
        ciphertext: Encrypted API key
        salt: Salt used for key derivation
        nonce: Nonce used for encryption
        passphrase: User's passphrase

    Returns:
        Decrypted API key

    Raises:
        DecryptionError: If passphrase is wrong
    """
    try:
        key = _derive_key(passphrase, salt)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    except Exception as e:
        raise DecryptionError(
            "Failed to decrypt key. Check your passphrase."
        ) from e


def _hash_key(api_key: str) -> str:
    """Create a hash of the API key for lookup without decryption."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]


class UserKeyStorage:
    """
    Passphrase-encrypted storage for user's own API keys.

    Keys are encrypted client-side with PBKDF2-derived AES-256-GCM.
    Platform cannot decrypt without user's passphrase.
    """

    def __init__(self, pool: "Pool", user_id: str):
        """
        Initialize user key storage.

        Args:
            pool: asyncpg connection pool
            user_id: User's UUID

        Raises:
            ValueError: If user_id is empty
        """
        if not user_id:
            raise ValueError("user_id is required for UserKeyStorage")

        self._pool = pool
        self._user_id = user_id

    async def store_key(
        self,
        provider_id: str,
        api_key: str,
        passphrase: str,
        display_name: Optional[str] = None,
        quota_limit: Optional[int] = None,
    ) -> str:
        """
        Store an encrypted API key for this user.

        Args:
            provider_id: Service provider (openrouter, openai, etc.)
            api_key: The API key to store
            passphrase: User's passphrase for encryption
            display_name: Optional friendly name
            quota_limit: Optional usage limit

        Returns:
            Key ID

        Raises:
            UserKeyStorageError: If storage fails
        """
        # Encrypt the key
        ciphertext, salt, nonce = _encrypt_key(api_key, passphrase)

        key_id = str(uuid.uuid4())
        key_hash = _hash_key(api_key)

        # Store as bytes directly (database expects BYTEA)
        async with self._pool.acquire() as conn:
            # Upsert (replace if exists for same user+provider)
            await conn.execute(
                """
                INSERT INTO user_service_keys
                (id, user_id, provider_id, key_mode, encrypted_key, key_salt, key_nonce,
                 key_hash, display_name, quota_limit, quota_used, is_active, created_at)
                VALUES ($1, $2, $3, 'byok', $4, $5, $6, $7, $8, $9, 0, TRUE, NOW())
                ON CONFLICT (user_id, provider_id) DO UPDATE SET
                    encrypted_key = EXCLUDED.encrypted_key,
                    key_salt = EXCLUDED.key_salt,
                    key_nonce = EXCLUDED.key_nonce,
                    key_hash = EXCLUDED.key_hash,
                    display_name = COALESCE(EXCLUDED.display_name, user_service_keys.display_name),
                    quota_limit = EXCLUDED.quota_limit,
                    is_active = TRUE,
                    updated_at = NOW()
                """,
                key_id,
                self._user_id,
                provider_id,
                ciphertext,  # Store as bytes directly
                salt,
                nonce,
                key_hash,
                display_name,
                quota_limit,
            )

        logger.info(f"Stored BYOK for user={self._user_id[:8]}..., provider={provider_id}")
        return key_id

    async def get_key(self, provider_id: str, passphrase: str) -> str:
        """
        Get decrypted API key for a provider.

        Args:
            provider_id: Service provider
            passphrase: User's passphrase for decryption

        Returns:
            Decrypted API key

        Raises:
            KeyNotFoundError: If no key found for provider
            DecryptionError: If passphrase is wrong
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT encrypted_key, key_salt, key_nonce
                FROM user_service_keys
                WHERE user_id = $1 AND provider_id = $2 AND is_active = TRUE
                """,
                self._user_id,
                provider_id,
            )

        if not row:
            raise KeyNotFoundError(
                f"No BYOK configured for provider '{provider_id}' and user '{self._user_id[:8]}...'"
            )

        # Data stored as bytes directly (BYTEA column)
        ciphertext = bytes(row["encrypted_key"])
        salt = bytes(row["key_salt"])
        nonce = bytes(row["key_nonce"])

        # Decrypt
        return _decrypt_key(ciphertext, salt, nonce, passphrase)

    async def has_key(self, provider_id: str) -> bool:
        """
        Check if user has a key configured for a provider.

        Note: Does not require passphrase - just checks existence.

        Args:
            provider_id: Service provider

        Returns:
            True if key exists
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM user_service_keys
                WHERE user_id = $1 AND provider_id = $2 AND is_active = TRUE
                """,
                self._user_id,
                provider_id,
            )
        return row is not None

    async def list_keys(self) -> List[UserKeyInfo]:
        """
        List all configured keys for this user (no secrets exposed).

        Returns:
            List of key info objects
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, user_id, provider_id, display_name, is_active,
                       quota_limit, quota_used, created_at
                FROM user_service_keys
                WHERE user_id = $1
                ORDER BY created_at DESC
                """,
                self._user_id,
            )

        return [
            UserKeyInfo(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                provider_id=row["provider_id"],
                display_name=row["display_name"],
                is_active=row["is_active"],
                quota_limit=row["quota_limit"],
                quota_used=row["quota_used"] or 0,
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
                UPDATE user_service_keys
                SET is_active = FALSE, updated_at = NOW()
                WHERE user_id = $1 AND provider_id = $2
                """,
                self._user_id,
                provider_id,
            )

        logger.info(f"Deactivated BYOK for user={self._user_id[:8]}..., provider={provider_id}")
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
                DELETE FROM user_service_keys
                WHERE user_id = $1 AND provider_id = $2
                """,
                self._user_id,
                provider_id,
            )

        logger.info(f"Deleted BYOK for user={self._user_id[:8]}..., provider={provider_id}")
        return "DELETE" in result

    async def record_usage(
        self,
        provider_id: str,
        units: int = 1,
    ) -> None:
        """
        Record usage for quota tracking.

        Note: BYOK users pay provider directly, so this is just for
        user's visibility and optional quota limits.

        Args:
            provider_id: Service provider
            units: Units consumed
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE user_service_keys
                SET quota_used = quota_used + $3, updated_at = NOW()
                WHERE user_id = $1 AND provider_id = $2
                """,
                self._user_id,
                provider_id,
                units,
            )

    async def check_quota(self, provider_id: str, units: int = 1) -> bool:
        """
        Check if quota allows operation.

        Args:
            provider_id: Service provider
            units: Units to consume

        Returns:
            True if quota allows operation (or no quota set)
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT quota_limit, quota_used
                FROM user_service_keys
                WHERE user_id = $1 AND provider_id = $2 AND is_active = TRUE
                """,
                self._user_id,
                provider_id,
            )

        if not row:
            return False  # No key = no access

        limit = row["quota_limit"]
        used = row["quota_used"] or 0

        if limit is None:
            return True  # No quota limit

        return used + units <= limit

    async def verify_passphrase(self, provider_id: str, passphrase: str) -> bool:
        """
        Verify a passphrase can decrypt a key without returning the key.

        Useful for session validation.

        Args:
            provider_id: Service provider
            passphrase: Passphrase to verify

        Returns:
            True if passphrase is correct
        """
        try:
            await self.get_key(provider_id, passphrase)
            return True
        except (KeyNotFoundError, DecryptionError):
            return False
