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
  sponsor's DID (``encrypt(sponsor_did, "service-keys", ...)``). A typed
  facade over the shared ``PrincipalMasterKeyStore``, like
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

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from kestrel_sovereign.security.principal_master_key_store import (
    MasterKeyRecord,
    MasterKeyTable,
    PrincipalMasterKeyScope,
    PrincipalMasterKeyStore,
)

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


def _sponsor_key_info(record: MasterKeyRecord) -> SponsorKeyInfo:
    return SponsorKeyInfo(
        id=record.id,
        sponsor_did=record.principal,
        provider_id=record.provider_id,
        is_active=record.is_active,
        created_at=record.created_at,
    )


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
        self._store = PrincipalMasterKeyStore(
            db,
            PrincipalMasterKeyScope(
                table=MasterKeyTable.SPONSOR,
                principal=sponsor_did,
                encryption_identity=sponsor_did,
                kind="sponsor",
                project_info=_sponsor_key_info,
            ),
        )

    async def store_key(self, provider_id: str, api_key: str) -> str:
        """Store an encrypted master key for this sponsor (idempotent per
        ``(sponsor_did, provider_id)``). Returns the row id."""
        return await self._store.store_key(provider_id, api_key)

    async def get_key(self, provider_id: str) -> str:
        """Get the decrypted sponsor master key for ``provider_id``.

        Raises KeyNotConfiguredError if no active key exists, and
        DecryptionError if the stored ciphertext cannot be decrypted.
        """
        return await self._store.get_key(provider_id)

    async def has_key(self, provider_id: str) -> bool:
        """True iff this sponsor has an active master key for ``provider_id``."""
        return await self._store.has_key(provider_id)

    async def list_keys(self) -> List[SponsorKeyInfo]:
        """List this sponsor's master keys (no secrets exposed), newest first."""
        return await self._store.list_keys()

    async def delete_key(self, provider_id: str) -> bool:
        """Hard-delete this sponsor's master key for ``provider_id``.

        Returns True iff an active row was actually removed.
        """
        return await self._store.delete_key(provider_id)


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
