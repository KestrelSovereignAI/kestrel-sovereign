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

Persistence is the shared ``PrincipalMasterKeyStore``; this module is the
typed user-scoped facade over the ``user_master_service_keys`` table.

Key derivation
--------------
Encryption uses the same SDK ``encrypt(identity, purpose, plaintext)`` contract
as ``ServiceKeyStorage`` / ``HostKeyStorage``, with the user's DID
(``master_did``) as the identity salt. Each user's master credentials get their
own HKDF-derived key under the same ``KESTREL_DATA_KEY`` master, with no overlap
with any agent's, the host's, or another user's key material.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, TYPE_CHECKING

from kestrel_sovereign.security.principal_master_key_store import (
    MasterKeyRecord,
    MasterKeyTable,
    PrincipalMasterKeyScope,
    PrincipalMasterKeyStore,
)

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase


@dataclass
class UserMasterKeyInfo:
    """Information about a stored user master key (no secret exposed)."""

    id: str
    master_did: str
    provider_id: str
    is_active: bool
    created_at: datetime


def _user_master_key_info(record: MasterKeyRecord) -> UserMasterKeyInfo:
    return UserMasterKeyInfo(
        id=record.id,
        master_did=record.principal,
        provider_id=record.provider_id,
        is_active=record.is_active,
        created_at=record.created_at,
    )


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
        self._store = PrincipalMasterKeyStore(
            db,
            PrincipalMasterKeyScope(
                table=MasterKeyTable.USER_MASTER,
                principal=master_did,
                encryption_identity=master_did,
                kind="user",
                project_info=_user_master_key_info,
            ),
        )

    async def store_key(self, provider_id: str, api_key: str) -> str:
        """Store an encrypted master key for this user.

        Idempotent: if a key already exists for ``(master_did, provider_id)``
        it is replaced. Returns the row id.
        """
        return await self._store.store_key(provider_id, api_key)

    async def get_key(self, provider_id: str) -> str:
        """Get the decrypted user master key for ``provider_id``.

        Raises:
            KeyNotConfiguredError: If no active key exists for that provider.
            DecryptionError: If the stored ciphertext cannot be decrypted.
        """
        return await self._store.get_key(provider_id)

    async def has_key(self, provider_id: str) -> bool:
        """True iff this user has an active master key for ``provider_id``."""
        return await self._store.has_key(provider_id)

    async def list_keys(self) -> List[UserMasterKeyInfo]:
        """List this user's master keys (no secrets exposed), newest first."""
        return await self._store.list_keys()

    async def delete_key(self, provider_id: str) -> bool:
        """Hard-delete this user's master key for ``provider_id``.

        Returns True iff an active row was actually removed.
        """
        return await self._store.delete_key(provider_id)


__all__ = [
    "UserMasterKeyStorage",
    "UserMasterKeyInfo",
]
