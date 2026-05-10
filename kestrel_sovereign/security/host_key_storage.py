"""
Host Master Key Storage for Kestrel.

Encrypted storage for host-level master credentials (the operator's
OpenRouter / Lighthouse / etc. master keys). Distinct from
``ServiceKeyStorage`` (per-agent child credentials) and from the
relocated Frinz ``PlatformKeyStorage`` (vending-machine pool with
margin-based billing).

Used by the foundation ``PayerResolver`` to back the
``HOST_MASTER_PROVISIONED`` policy: the operator stores their master
credential here once at setup time, and the resolver mints a
per-agent child credential against it on first use, then stores the
child in the agent's ``ServiceKeyStorage``.

Single host per deployment in this initial implementation. Sponsor
and user-master variants of the master-credential storage are
modeled as out-of-scope follow-ups (see ``SUPPORT_MATRIX`` in
``kestrel_sdk.payer_policy``).

Key derivation
--------------
Encryption uses the same SDK ``encrypt(identity, purpose, plaintext)``
contract as ``ServiceKeyStorage`` does, but with the literal identity
string ``"host"`` instead of an agent DID. That gives the host's master
credentials their own HKDF-derived encryption key under the same
``KESTREL_DATA_KEY`` master, with no overlap with any agent's
key material.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from kestrel_sovereign.security.agent_encryption import encrypt, decrypt
from kestrel_sovereign.security.exceptions import (
    KeyNotConfiguredError,
)
from kestrel_sovereign.security.service_key_storage import KNOWN_PROVIDERS

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


# The single non-agent identity used to derive the host's encryption
# key. Distinct from any valid DID, so an agent and the host can never
# collide on storage rows or derived keys.
_HOST_IDENTITY = "host"


@dataclass
class HostKeyInfo:
    """Information about a stored host master key (no secret exposed)."""

    id: str
    provider_id: str
    is_active: bool
    created_at: datetime


class HostKeyStorage:
    """Operator-scoped encrypted storage for host master credentials.

    All operations are scoped to a single host (the operator running
    Kestrel). There is no ``host_id`` parameter — sponsor / user-master
    variants are explicitly out of scope for this primitive and modeled
    separately if needed later.

    Keys are encrypted using the host-derived encryption key from the
    unified encryption module. The same ``encrypt`` / ``decrypt`` API
    that ``ServiceKeyStorage`` uses, with ``"host"`` as the identity
    salt, gives the host its own derivation that cannot collide with
    any agent's key material.
    """

    def __init__(self, db: "AsyncDatabase") -> None:
        self._db = db

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
        """Store an encrypted master key for the host.

        Idempotent: if a key already exists for ``provider_id`` it is
        replaced. Returns the row id.
        """
        await self._ensure_provider(provider_id)

        encrypted_bytes = encrypt(
            _HOST_IDENTITY,
            "service-keys",
            api_key.encode("utf-8"),
        )
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode("ascii")

        key_id = str(uuid.uuid4())
        key_hash = self._hash_key(api_key)

        await self._db.execute(
            """
            INSERT OR REPLACE INTO host_service_keys
            (id, provider_id, encrypted_key, key_hash, is_active, created_at)
            VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            """,
            (key_id, provider_id, encrypted_b64, key_hash),
        )

        logger.info(f"Stored host master key for provider={provider_id}")
        return key_id

    async def get_key(self, provider_id: str) -> str:
        """Get the decrypted host master key for ``provider_id``.

        Raises:
            KeyNotConfiguredError: If no active key exists for that provider.
        """
        rows = await self._db.fetchall(
            """
            SELECT encrypted_key FROM host_service_keys
            WHERE provider_id = ? AND is_active = 1
            """,
            (provider_id,),
        )

        if not rows or not rows[0][0]:
            raise KeyNotConfiguredError(
                f"No host master key configured for provider '{provider_id}'"
            )

        encrypted_b64 = rows[0][0]
        encrypted_bytes = base64.b64decode(encrypted_b64)
        plaintext_bytes = decrypt(
            _HOST_IDENTITY,
            "service-keys",
            encrypted_bytes,
        )
        return plaintext_bytes.decode("utf-8")

    async def has_key(self, provider_id: str) -> bool:
        """True iff the host has an active master key for ``provider_id``."""
        rows = await self._db.fetchall(
            """
            SELECT 1 FROM host_service_keys
            WHERE provider_id = ? AND is_active = 1
            """,
            (provider_id,),
        )
        return len(rows) > 0

    async def list_keys(self) -> List[HostKeyInfo]:
        """List all host master keys (no secrets exposed)."""
        rows = await self._db.fetchall(
            """
            SELECT id, provider_id, is_active, created_at
            FROM host_service_keys
            ORDER BY created_at DESC
            """,
            (),
        )
        return [
            HostKeyInfo(
                id=row[0],
                provider_id=row[1],
                is_active=bool(row[2]),
                created_at=datetime.fromisoformat(row[3]) if row[3] else datetime.utcnow(),
            )
            for row in rows
        ]

    async def delete_key(self, provider_id: str) -> bool:
        """Hard-delete the host master key for ``provider_id``.

        Returns True iff a row was actually removed.
        """
        # SQLite/asyncpg both support COUNT-after-DELETE via a separate
        # query; the AsyncDatabase.execute() return value is not
        # uniformly a row-count, so we read affected count via a
        # SELECT-then-DELETE pattern to keep portability honest.
        existed = await self.has_key(provider_id)
        if not existed:
            return False

        await self._db.execute(
            """
            DELETE FROM host_service_keys
            WHERE provider_id = ?
            """,
            (provider_id,),
        )
        logger.info(f"Deleted host master key for provider={provider_id}")
        return True


__all__ = [
    "HostKeyStorage",
    "HostKeyInfo",
]
