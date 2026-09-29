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

Single host per deployment. The per-user and per-sponsor master
credentials behind ``USER_MASTER_PROVISIONED`` and ``SPONSOR`` live in
``UserMasterKeyStorage`` and ``SponsorKeyStorage``; all three are thin
facades over the shared ``PrincipalMasterKeyStore`` persistence, each with
its own table and encryption identity.

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

from dataclasses import dataclass
from datetime import datetime
from typing import List, TYPE_CHECKING

# Re-exported: callers import KeyNotConfiguredError from this module.
from kestrel_sovereign.security.exceptions import (  # noqa: F401
    KeyNotConfiguredError,
)
from kestrel_sovereign.security.principal_master_key_store import (
    MasterKeyRecord,
    MasterKeyTable,
    PrincipalMasterKeyScope,
    PrincipalMasterKeyStore,
)

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase


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


def _host_key_info(record: MasterKeyRecord) -> HostKeyInfo:
    return HostKeyInfo(
        id=record.id,
        provider_id=record.provider_id,
        is_active=record.is_active,
        created_at=record.created_at,
    )


class HostKeyStorage:
    """Operator-scoped encrypted storage for host master credentials.

    All operations are scoped to a single host (the operator running
    Kestrel), so there is no ``host_id`` parameter. Keys are encrypted
    with ``"host"`` as the identity salt, which gives the host its own
    derivation that cannot collide with any agent's, user's or
    sponsor's key material.
    """

    def __init__(self, db: "AsyncDatabase") -> None:
        self._store = PrincipalMasterKeyStore(
            db,
            PrincipalMasterKeyScope(
                table=MasterKeyTable.HOST,
                principal=None,
                encryption_identity=_HOST_IDENTITY,
                kind="host",
                project_info=_host_key_info,
            ),
        )

    async def store_key(self, provider_id: str, api_key: str) -> str:
        """Store an encrypted master key for the host.

        Idempotent: if a key already exists for ``provider_id`` it is
        replaced. Returns the row id.
        """
        return await self._store.store_key(provider_id, api_key)

    async def get_key(self, provider_id: str) -> str:
        """Get the decrypted host master key for ``provider_id``.

        Raises:
            KeyNotConfiguredError: If no active key exists for that provider.
            DecryptionError: If the stored ciphertext cannot be decrypted.
        """
        return await self._store.get_key(provider_id)

    async def has_key(self, provider_id: str) -> bool:
        """True iff the host has an active master key for ``provider_id``."""
        return await self._store.has_key(provider_id)

    async def list_keys(self) -> List[HostKeyInfo]:
        """List all host master keys (no secrets exposed), newest first."""
        return await self._store.list_keys()

    async def delete_key(self, provider_id: str) -> bool:
        """Hard-delete the host master key for ``provider_id``.

        Returns True iff an active row was actually removed.
        """
        return await self._store.delete_key(provider_id)


__all__ = [
    "HostKeyStorage",
    "HostKeyInfo",
]
