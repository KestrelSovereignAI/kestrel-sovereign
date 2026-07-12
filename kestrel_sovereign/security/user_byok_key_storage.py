"""Zero-knowledge user BYOK service-key storage.

Unlike ``ServiceKeyStorage``, ``HostKeyStorage``, and
``UserMasterKeyStorage``, this class never uses ``KESTREL_DATA_KEY`` or any
platform-readable master. A user passphrase is supplied at write/read time,
stretched with a per-row salt, and used only to encrypt/decrypt that row's
provider credential.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, TYPE_CHECKING

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from kestrel_sovereign.security.exceptions import (
    DecryptionError,
    KeyNotConfiguredError,
    PassphraseRequiredError,
)
from kestrel_sovereign.security.key_storage import (
    KEY_SIZE,
    NONCE_SIZE,
    PBKDF2_ITERATIONS,
    SALT_SIZE,
)
from kestrel_sovereign.security.service_key_storage import _as_datetime

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)

_KNOWN_PROVIDERS = {
    "openrouter": {"name": "OpenRouter", "supports_sub_accounts": True},
    "openai": {"name": "OpenAI", "supports_sub_accounts": False},
    "anthropic": {"name": "Anthropic", "supports_sub_accounts": False},
    "lighthouse": {"name": "Lighthouse (IPFS/Filecoin)", "supports_sub_accounts": False},
    "github": {"name": "GitHub", "supports_sub_accounts": False},
    "runpod": {"name": "RunPod", "supports_sub_accounts": True},
    "vastai": {"name": "Vast.ai", "supports_sub_accounts": False},
}


@dataclass
class UserBYOKKeyInfo:
    """Information about a stored zero-knowledge BYOK key."""

    id: str
    agent_did: str
    provider_id: str
    is_active: bool
    created_at: datetime


class UserBYOKKeyStorage:
    """Agent-scoped zero-knowledge BYOK credential store."""

    def __init__(self, db: "AsyncDatabase", agent_did: str) -> None:
        if not agent_did:
            raise ValueError("agent_did is required for UserBYOKKeyStorage")
        self._db = db
        self._agent_did = agent_did

    @staticmethod
    def _hash_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _require_passphrase(passphrase: str) -> bytes:
        if not passphrase:
            raise PassphraseRequiredError("USER_BYOK requires a per-request passphrase")
        return passphrase.encode("utf-8")

    @staticmethod
    def _derive_key(passphrase: bytes, salt: bytes, iterations: int) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=iterations,
            backend=default_backend(),
        )
        return kdf.derive(passphrase)

    def _aad(self, provider_id: str) -> bytes:
        return f"user_byok:{self._agent_did}:{provider_id}".encode("utf-8")

    async def _ensure_provider(self, provider_id: str) -> None:
        provider_info = _KNOWN_PROVIDERS.get(provider_id, {"name": provider_id})
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
            ),
        )

    async def store_key(
        self,
        provider_id: str,
        api_key: str,
        passphrase: str,
    ) -> str:
        """Store a provider key encrypted only by the user's passphrase."""
        provider_id = provider_id.lower()
        await self._ensure_provider(provider_id)

        salt = os.urandom(SALT_SIZE)
        nonce = os.urandom(NONCE_SIZE)
        derived_key = self._derive_key(
            self._require_passphrase(passphrase),
            salt,
            PBKDF2_ITERATIONS,
        )
        ciphertext = AESGCM(derived_key).encrypt(
            nonce,
            api_key.encode("utf-8"),
            self._aad(provider_id),
        )

        key_id = str(uuid.uuid4())
        await self._db.execute(
            """
            INSERT OR REPLACE INTO user_byok_service_keys
            (id, agent_did, provider_id, encrypted_key, key_salt, key_nonce,
             key_hash, kdf, kdf_iterations, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PBKDF2-SHA256', ?, 1, CURRENT_TIMESTAMP)
            """,
            (
                key_id,
                self._agent_did,
                provider_id,
                base64.b64encode(ciphertext).decode("ascii"),
                base64.b64encode(salt).decode("ascii"),
                base64.b64encode(nonce).decode("ascii"),
                self._hash_key(api_key),
                PBKDF2_ITERATIONS,
            ),
        )
        logger.info(
            "Stored USER_BYOK key for agent=%s..., provider=%s",
            self._agent_did[:30],
            provider_id,
        )
        return key_id

    async def get_key(self, provider_id: str, passphrase: str) -> str:
        """Decrypt a provider key with the per-request user passphrase."""
        provider_id = provider_id.lower()
        rows = await self._db.fetchall(
            """
            SELECT encrypted_key, key_salt, key_nonce, kdf, kdf_iterations
            FROM user_byok_service_keys
            WHERE agent_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._agent_did, provider_id),
        )
        if not rows:
            raise KeyNotConfiguredError(
                f"No USER_BYOK key configured for provider '{provider_id}' "
                f"and agent '{self._agent_did[:30]}...'"
            )

        encrypted_b64, salt_b64, nonce_b64, kdf, iterations = rows[0]
        if kdf != "PBKDF2-SHA256":
            raise DecryptionError(f"Unsupported USER_BYOK KDF: {kdf}")

        passphrase_bytes = self._require_passphrase(passphrase)
        salt = base64.b64decode(salt_b64)
        nonce = base64.b64decode(nonce_b64)
        ciphertext = base64.b64decode(encrypted_b64)
        derived_key = self._derive_key(passphrase_bytes, salt, int(iterations))
        try:
            plaintext = AESGCM(derived_key).decrypt(
                nonce,
                ciphertext,
                self._aad(provider_id),
            )
        except InvalidTag as exc:
            raise DecryptionError("USER_BYOK passphrase could not decrypt key") from exc
        return plaintext.decode("utf-8")

    async def has_key(self, provider_id: str) -> bool:
        rows = await self._db.fetchall(
            """
            SELECT 1 FROM user_byok_service_keys
            WHERE agent_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._agent_did, provider_id.lower()),
        )
        return bool(rows)

    async def list_keys(self) -> List[UserBYOKKeyInfo]:
        rows = await self._db.fetchall(
            """
            SELECT id, agent_did, provider_id, is_active, created_at
            FROM user_byok_service_keys
            WHERE agent_did = ?
            ORDER BY created_at DESC
            """,
            (self._agent_did,),
        )
        return [
            UserBYOKKeyInfo(
                id=row[0],
                agent_did=row[1],
                provider_id=row[2],
                is_active=bool(row[3]),
                created_at=_as_datetime(row[4]),
            )
            for row in rows
        ]

    async def delete_key(self, provider_id: str) -> bool:
        existed = await self.has_key(provider_id)
        if not existed:
            return False
        await self._db.execute(
            """
            DELETE FROM user_byok_service_keys
            WHERE agent_did = ? AND provider_id = ?
            """,
            (self._agent_did, provider_id.lower()),
        )
        return True

    async def verify_passphrase(self, provider_id: str, passphrase: str) -> bool:
        """Verify that a passphrase can decrypt the stored key.

        Args:
            provider_id: The provider whose key to verify against.
            passphrase: The passphrase to test.

        Returns:
            True if the passphrase successfully decrypts the key, False otherwise.
        """
        try:
            # Attempt to decrypt the key with the passphrase
            await self.get_key(provider_id, passphrase)
            return True
        except (KeyNotConfiguredError, DecryptionError):
            return False


class UserBYOKKeyResolutionService:
    """No-fallback resolver for a single request's USER_BYOK passphrase."""

    def __init__(
        self,
        storage: UserBYOKKeyStorage,
        passphrase: str,
    ) -> None:
        self._storage = storage
        self._passphrase = passphrase

    async def resolve_key(self, provider: str, require: bool = True) -> str | None:
        try:
            return await self._storage.get_key(provider, self._passphrase)
        except KeyNotConfiguredError:
            if require:
                raise
            return None

    async def has_key(self, provider: str) -> bool:
        return await self._storage.has_key(provider)


__all__ = [
    "UserBYOKKeyInfo",
    "UserBYOKKeyResolutionService",
    "UserBYOKKeyStorage",
]
