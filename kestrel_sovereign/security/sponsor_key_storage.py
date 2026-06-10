"""
Sponsor Master Key Storage + beneficiary roster for Kestrel.

Backs the ``SPONSOR`` payer-policy path: a named third party (the *sponsor* —
e.g. an organization) holds a master credential and funds a group of agents
(*beneficiaries*). The foundation ``PayerResolver`` mints a per-agent child
credential against the sponsor's master on first use, then stores the child in
the agent's ``ServiceKeyStorage`` — the same delegated-master mechanism as
``HOST_MASTER_PROVISIONED`` / ``USER_MASTER_PROVISIONED``, but the master
belongs to a sponsor funding many agents.

Two primitives:

- ``SponsorKeyStorage`` — per-sponsor master credentials, keyed by the
  sponsor's DID (``encrypt(sponsor_did, "service-keys", ...)``), mirroring
  ``UserMasterKeyStorage``. One master per (sponsor, provider).
- ``SponsorBeneficiaryStore`` — the sponsor→agent roster ("which sponsor funds
  this agent", "list a sponsor's agents"). A policy builder consults it to set
  ``PayerSpec(kind=SPONSOR, master_did=<sponsor>)`` for an enrolled agent.

Scope notes
-----------
The sponsor is a *generic* funding principal; no patient / healthcare semantics
live here (those are a product concern for consuming layers). Enrollment
authority / consent is likewise **not** enforced here — this is the mechanism;
who may enroll a beneficiary is a product policy. Group-level (aggregate) spend
caps are out of scope for this primitive: each agent gets a per-agent capped
child, and the sponsor's own provider-account limit bounds the group.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

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
class SponsorKeyInfo:
    """Information about a stored sponsor master key (no secret exposed)."""

    id: str
    sponsor_did: str
    provider_id: str
    is_active: bool
    created_at: datetime


class SponsorKeyStorage:
    """Sponsor-scoped encrypted storage for master credentials.

    All operations are scoped to a single ``sponsor_did`` (the funding
    sponsor's DID, carried by ``PayerSpec.master_did`` for the SPONSOR kind).
    Keys are encrypted with the sponsor's own HKDF-derived key
    (identity = ``sponsor_did``), so they cannot collide with any agent's, the
    host's, a user's, or another sponsor's key material.
    """

    def __init__(self, db: "AsyncDatabase", sponsor_did: str) -> None:
        if not sponsor_did:
            raise ValueError("sponsor_did is required for SponsorKeyStorage")
        self._db = db
        self._sponsor_did = sponsor_did

    @staticmethod
    def _hash_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]

    async def _ensure_provider(self, provider_id: str) -> None:
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
        """Store an encrypted master key for this sponsor (idempotent per
        ``(sponsor_did, provider_id)``). Returns the row id."""
        await self._ensure_provider(provider_id)

        encrypted_bytes = encrypt(
            self._sponsor_did,
            "service-keys",
            api_key.encode("utf-8"),
        )
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode("ascii")

        key_id = str(uuid.uuid4())
        key_hash = self._hash_key(api_key)

        await self._db.execute(
            """
            INSERT OR REPLACE INTO sponsor_master_service_keys
            (id, master_did, provider_id, encrypted_key, key_hash, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            """,
            (key_id, self._sponsor_did, provider_id, encrypted_b64, key_hash),
        )

        logger.info(
            f"Stored sponsor master key for sponsor={self._sponsor_did[:30]}..., "
            f"provider={provider_id}"
        )
        return key_id

    async def get_key(self, provider_id: str) -> str:
        """Get the decrypted sponsor master key for ``provider_id``.

        Raises KeyNotConfiguredError if no active key exists.
        """
        rows = await self._db.fetchall(
            """
            SELECT encrypted_key FROM sponsor_master_service_keys
            WHERE master_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._sponsor_did, provider_id),
        )

        if not rows or not rows[0][0]:
            raise KeyNotConfiguredError(
                f"No sponsor master key configured for provider '{provider_id}' "
                f"and sponsor '{self._sponsor_did[:30]}...'"
            )

        encrypted_bytes = base64.b64decode(rows[0][0])
        plaintext_bytes = decrypt(
            self._sponsor_did,
            "service-keys",
            encrypted_bytes,
        )
        return plaintext_bytes.decode("utf-8")

    async def has_key(self, provider_id: str) -> bool:
        """True iff this sponsor has an active master key for ``provider_id``."""
        rows = await self._db.fetchall(
            """
            SELECT 1 FROM sponsor_master_service_keys
            WHERE master_did = ? AND provider_id = ? AND is_active = 1
            """,
            (self._sponsor_did, provider_id),
        )
        return len(rows) > 0

    async def list_keys(self) -> List[SponsorKeyInfo]:
        """List this sponsor's master keys (no secrets exposed)."""
        rows = await self._db.fetchall(
            """
            SELECT id, master_did, provider_id, is_active, created_at
            FROM sponsor_master_service_keys
            WHERE master_did = ?
            ORDER BY created_at DESC
            """,
            (self._sponsor_did,),
        )
        return [
            SponsorKeyInfo(
                id=row[0],
                sponsor_did=row[1],
                provider_id=row[2],
                is_active=bool(row[3]),
                created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.utcnow(),
            )
            for row in rows
        ]

    async def delete_key(self, provider_id: str) -> bool:
        """Hard-delete this sponsor's master key for ``provider_id``."""
        if not await self.has_key(provider_id):
            return False
        await self._db.execute(
            """
            DELETE FROM sponsor_master_service_keys
            WHERE master_did = ? AND provider_id = ?
            """,
            (self._sponsor_did, provider_id),
        )
        logger.info(
            f"Deleted sponsor master key for sponsor={self._sponsor_did[:30]}..., "
            f"provider={provider_id}"
        )
        return True


class SponsorBeneficiaryStore:
    """The sponsor→agent funding roster.

    One funding sponsor per agent (per-agent model). Enrolling an agent that is
    already enrolled re-points it to the new sponsor. This is the mechanism
    only — who is *authorized* to enroll/disenroll a beneficiary is a product
    policy enforced by the consuming layer.
    """

    def __init__(self, db: "AsyncDatabase") -> None:
        self._db = db

    async def enroll(self, sponsor_did: str, agent_did: str) -> None:
        """Enroll ``agent_did`` as a beneficiary funded by ``sponsor_did``."""
        if not sponsor_did or not agent_did:
            raise ValueError("sponsor_did and agent_did are required")
        await self._db.execute(
            """
            INSERT OR REPLACE INTO sponsor_beneficiaries
            (sponsor_did, agent_did, is_active, enrolled_at)
            VALUES (?, ?, 1, CURRENT_TIMESTAMP)
            """,
            (sponsor_did, agent_did),
        )
        logger.info(
            f"Enrolled agent={agent_did[:30]}... under sponsor="
            f"{sponsor_did[:30]}..."
        )

    async def disenroll(self, agent_did: str) -> bool:
        """Remove ``agent_did`` from its sponsor's roster. Returns True iff a
        row was removed. (Revoking the agent's minted child credential is a
        caller concern — see retirement / key teardown.)"""
        sponsor = await self.get_sponsor_for_agent(agent_did)
        if sponsor is None:
            return False
        await self._db.execute(
            "DELETE FROM sponsor_beneficiaries WHERE agent_did = ?",
            (agent_did,),
        )
        logger.info(f"Disenrolled agent={agent_did[:30]}... from sponsor roster")
        return True

    async def get_sponsor_for_agent(self, agent_did: str) -> Optional[str]:
        """The DID of the sponsor funding ``agent_did``, or None if unenrolled."""
        rows = await self._db.fetchall(
            """
            SELECT sponsor_did FROM sponsor_beneficiaries
            WHERE agent_did = ? AND is_active = 1
            """,
            (agent_did,),
        )
        return rows[0][0] if rows else None

    async def list_beneficiaries(self, sponsor_did: str) -> List[str]:
        """All agent DIDs currently funded by ``sponsor_did``."""
        rows = await self._db.fetchall(
            """
            SELECT agent_did FROM sponsor_beneficiaries
            WHERE sponsor_did = ? AND is_active = 1
            ORDER BY enrolled_at
            """,
            (sponsor_did,),
        )
        return [row[0] for row in rows]

    async def is_enrolled(self, sponsor_did: str, agent_did: str) -> bool:
        """True iff ``agent_did`` is currently funded by ``sponsor_did``."""
        return (await self.get_sponsor_for_agent(agent_did)) == sponsor_did


__all__ = [
    "SponsorKeyStorage",
    "SponsorKeyInfo",
    "SponsorBeneficiaryStore",
]
