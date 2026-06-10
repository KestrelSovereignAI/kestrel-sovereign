"""
User Master Key Storage for Kestrel.

Encrypted storage for per-user master credentials backing the
``USER_MASTER_PROVISIONED`` payer-policy path: a named user holds a master
account (e.g. their own OpenRouter key), and the foundation ``PayerResolver``
mints a per-agent child credential against it on first use, then stores the
child in the agent's ``ServiceKeyStorage`` — exactly the
``HOST_MASTER_PROVISIONED`` mechanism, but the master belongs to a user
instead of the operator.

Distinct from:
- ``ServiceKeyStorage`` (per-agent *child* credentials, keyed by agent DID)
- ``HostKeyStorage`` (the single operator master, identity ``"host"``)

Key derivation
--------------
Encryption uses the same SDK ``encrypt(identity, purpose, plaintext)`` contract
as ``ServiceKeyStorage`` / ``HostKeyStorage``, with the user's DID
(``master_did``) as the identity salt. Each user's master credentials get their
own HKDF-derived key under the same ``KESTREL_DATA_KEY`` master, with no overlap
with any agent's, the host's, or another user's key material.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, TYPE_CHECKING

from kestrel_sovereign.security.agent_encryption import encrypt
from kestrel_sovereign.security.legacy_decrypt import (
    decrypt_with_legacy_fallback as decrypt,
)
from kestrel_sovereign.security.exceptions import (
    KeyNotConfiguredError,
)
from kestrel_sovereign.security.service_key_storage import KNOWN_PROVIDERS

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


@dataclass
class UserMasterKeyInfo:
    """Information about a stored user master key (no secret exposed)."""

    id: str
    master_did: str
    provider_id: str
    is_active: bool
    created_at: datetime


class UserMasterKeyStorage:
    """User-scoped encrypted storage for master credentials.

    All operations are scoped to a single ``master_did`` (the funding user's
    DID, carried by ``PayerSpec.master_did``). Keys are encrypted with the
    user's own HKDF-derived key (identity = ``master_did``), so they cannot
    collide with any agent's, the host's, or another user's key material.
    """

    def __init__(self, db: "AsyncDatabase", master_did: str) -> None:
        if not master_did:
            raise ValueError("master_did is required for UserMasterKeyStorage")
        self._db = db
        self._master_did = master_did

    @staticmethod
    def _hash_key(api_key: str) -> str:
        """Create a hash of the API key for lookup without decryption."""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]

    async def _ensure_provider(self, provider_id: str) -> None:
        """Ensure provider exists in the shared service_providers table."""
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
            ),
        )

    async def store_key(self, provider_id: str, api_key: str) -> str:
        """Store an encrypted master key for this user.

        Idempotent: if a key already exists for ``(master_did, provider_id)``
        it is replaced. Returns the row id.
        """
        await self._ensure_provider(provider_id)

        encrypted_bytes = encrypt(
            self._master_did,
            "service-keys",
            api_key.encode("utf-8"),
        )
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode("ascii")

        key_id = str(uuid.uuid4())
        key_hash = self._hash_key(api_key)

        await self._db.execute(
            """
            INSERT OR REPLACE INTO user_master_service_keys
            (id, master_did, provider_id, encrypted_key, key_hash, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            """,
            (key_id, self._master_did, provider_id, encrypted_b64, key_hash),
        )

        logger.info(
            f"Stored user master key for user={self._master_did[:30]}..., "
            f"provider={provider_id}"
        )
        return key_id

    async def get_key(self, provider_id: str) -> str:
        """Get the decrypted user master key for ``provider_id``.

        Raises:
            KeyNotConfiguredError: If no active key exists for that provider.
        """
        rows = await self._db.fetchall(
            """
            SELECT encrypted_key FROM user_master_service_keys
            WHERE master_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._master_did, provider_id),
        )

        if not rows or not rows[0][0]:
            raise KeyNotConfiguredError(
                f"No user master key configured for provider '{provider_id}' "
                f"and user '{self._master_did[:30]}...'"
            )

        encrypted_b64 = rows[0][0]
        encrypted_bytes = base64.b64decode(encrypted_b64)
        plaintext_bytes = decrypt(
            self._master_did,
            "service-keys",
            encrypted_bytes,
        )
        return plaintext_bytes.decode("utf-8")

    async def has_key(self, provider_id: str) -> bool:
        """True iff this user has an active master key for ``provider_id``."""
        rows = await self._db.fetchall(
            """
            SELECT 1 FROM user_master_service_keys
            WHERE master_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._master_did, provider_id),
        )
        return len(rows) > 0

    async def list_keys(self) -> List[UserMasterKeyInfo]:
        """List this user's master keys (no secrets exposed)."""
        rows = await self._db.fetchall(
            """
            SELECT id, master_did, provider_id, is_active, created_at
            FROM user_master_service_keys
            WHERE master_did = ?
            ORDER BY created_at DESC
            """,
            (self._master_did,),
        )
        return [
            UserMasterKeyInfo(
                id=row[0],
                master_did=row[1],
                provider_id=row[2],
                is_active=bool(row[3]),
                created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.utcnow(),
            )
            for row in rows
        ]

    async def delete_key(self, provider_id: str) -> bool:
        """Hard-delete this user's master key for ``provider_id``.

        Returns True iff a row was actually removed.
        """
        existed = await self.has_key(provider_id)
        if not existed:
            return False

        await self._db.execute(
            """
            DELETE FROM user_master_service_keys
            WHERE master_did = ? AND provider_id = ?
            """,
            (self._master_did, provider_id),
        )
        logger.info(
            f"Deleted user master key for user={self._master_did[:30]}..., "
            f"provider={provider_id}"
        )
        return True


__all__ = [
    "UserMasterKeyStorage",
    "UserMasterKeyInfo",
]
