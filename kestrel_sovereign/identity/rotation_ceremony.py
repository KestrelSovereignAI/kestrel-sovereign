"""
Hybrid-identity rotation ceremony — Wave 3 sub-PR 4 of Quantum Hardening (#921, #918).

Operational primitive for migrating a legacy Kestrel agent (did:pkh
ECDSA-only) to a hybrid did:web identity (Ed25519 + ML-DSA-65) via a
signed succession statement.

Deployment-timing critical
--------------------------

This ceremony MUST run while classical crypto is still trusted. If a
predecessor's ECDSA private key has already been recovered via Shor's
algorithm, an adversary can produce a competing back-dated succession
statement and fork the chain. The window to migrate Kestrel #1, Emma,
Meridian, and current Frinz tenants is "before HNDL becomes practical."
NIST projects 2030+ for cryptographically-relevant quantum computers;
the conservative posture is to migrate now.

Pure orchestration
------------------

This module composes existing Wave 2-3 primitives:

- :func:`identity.inception_did_web.create_did_web_identity`
  generates the new hybrid keypair + DID document
- :func:`identity.succession.SuccessionStatement` /
  :func:`sign_predecessor` / :func:`sign_successor` /
  :func:`archival_countersign` / :func:`finalize` build the statement
- :func:`identity.succession_chain.build_chain` validates the result
  before it leaves the ceremony

It does NOT touch persistence (the caller stores private keys via
``SecureKeyStorage`` and publishes the new DID document at the
agent's web origin) and does NOT touch ``inception_service`` (legacy
agents keep their existing inception path). That separation is
intentional: the ceremony is purely cryptographic; deployment side-
effects are the caller's responsibility.

Key destruction
---------------

When the ceremony succeeds, the caller SHOULD secure-delete the legacy
ECDSA private key (the predecessor's). The succession statement is
signed and archived; the legacy key is no longer needed for any
forward-looking operation. Holding onto it after ``effective_from``
expands the attack surface (a future Shor break could recover that key
and forge a competing succession). However, **do not auto-delete here**:
key destruction is a destructive operation that the operator should
run with eyes open, after they've verified the new identity is live.
A separate ``destroy_legacy_key`` helper in operations runbooks handles
that step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Mapping, Optional, Sequence

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.inception_did_web import (
    HybridDidWebIdentity,
    create_did_web_identity,
)
from kestrel_sovereign.identity.succession import (
    SuccessionStatement,
    archival_countersign,
    finalize,
    sign_predecessor,
    sign_successor,
    verify_succession,
)
from kestrel_sovereign.identity.succession_chain import (
    SuccessionChain,
    build_chain,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_SLH_DSA_SHA2_128S,
    Keypair,
    SLHDSASHA2128sSuite,
    get_suite,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RotationCeremonyResult:
    """Output of :func:`run_rotation_ceremony`.

    - ``predecessor_did``: input legacy DID (unchanged copy for callers)
    - ``new_identity``: full hybrid did:web bundle (DID + DID document
      ready to publish + HybridKeypair to safeguard)
    - ``succession_statement``: fully signed and finalized statement
    - ``archival_keypair``: optional SLH-DSA keypair the caller passed
      in (returned for symmetry; the caller already has the private
      key, so this is just the public + suite_id for VM publication)
    - ``chain``: ``SuccessionChain`` containing only this statement,
      pre-validated. Hand this to :func:`verify_artifact_against_chain`
      directly.
    """

    predecessor_did: str
    new_identity: HybridDidWebIdentity
    succession_statement: SuccessionStatement
    archival_keypair: Optional[Keypair]
    chain: SuccessionChain


# ---------------------------------------------------------------------------
# Ceremony
# ---------------------------------------------------------------------------

def run_rotation_ceremony(
    *,
    predecessor_did: str,
    predecessor_keypair: Keypair,
    predecessor_kid: str,
    predecessor_verification_methods: Sequence[Mapping],
    new_did_domain: str,
    new_did_slug: str,
    reason: str,
    effective_from: Optional[str] = None,
    archival_keypair: Optional[Keypair] = None,
    archival_verification_method: Optional[Mapping] = None,
    extra_path_segments: Sequence[str] = (),
    also_known_as: Optional[Sequence[str]] = None,
    services: Optional[Sequence[dict]] = None,
) -> RotationCeremonyResult:
    """Run a complete legacy → hybrid rotation ceremony.

    Steps performed:

    1. Mint a new hybrid did:web identity at
       ``did:web:<new_did_domain>:<new_did_slug>``. The successor's DID
       document includes ``alsoKnownAs`` linking back to
       ``predecessor_did`` if not overridden via ``also_known_as``,
       so verifiers walking the new identity can find the chain.
    2. Build a :class:`SuccessionStatement` containing the predecessor
       DID + verification methods, the new successor DID + methods,
       the cutoff timestamp, and the reason.
    3. Sign with the predecessor's keypair (proof of authorization).
    4. Sign with the successor's hybrid keypair (proof of acceptance,
       both classical and post-quantum halves).
    5. (Optional) Apply the SLH-DSA-SHA2-128s archival countersignature.
    6. Finalize (stamp statement_id + created_at).
    7. Build a :class:`SuccessionChain` of this single statement and
       run structural validation.
    8. Run :func:`verify_succession` against the result to confirm
       every signature crypto-verifies before returning. If this fails
       the ceremony raises rather than returning a malformed result.

    Args:
        predecessor_did: legacy DID (e.g. ``did:pkh:eip155:1:0x…``)
        predecessor_keypair: legacy ECDSA keypair (or, for an already-
            hybrid agent rotating again, the current hybrid keypair —
            not yet supported by this helper; this PR ships the
            single-keypair predecessor case which covers Kestrel #1,
            Emma, Meridian, and current Frinz tenants).
        predecessor_kid: kid fragment matching one entry in
            ``predecessor_verification_methods``
        predecessor_verification_methods: VMs authoritative for the
            legacy DID at ceremony time
        new_did_domain: hostname for the new did:web identity
        new_did_slug: agent slug; resolves to
            ``https://<domain>/<slug>/did.json``
        reason: free-text human-facing description (committed in
            signable payload — cannot be retconned)
        effective_from: ISO 8601 UTC cutoff. Defaults to
            ``datetime.utcnow()`` when not provided. Pick a near-now
            timestamp; the ceremony itself runs in the
            "classical-still-trusted" window.
        archival_keypair: optional SLH-DSA-SHA2-128s keypair for the
            archival countersignature. Strongly recommended for
            irrevocable events (Wave 3 ceremonies).
        archival_verification_method: optional VM for the archival
            keypair's public key. Required if ``archival_keypair`` is
            provided.
        extra_path_segments: additional path segments after slug
            (rare; used for versioned identities).
        also_known_as: optional alsoKnownAs entries for the new DID
            document. Defaults to ``[predecessor_did]`` so consumers
            walking the new identity can find the predecessor without
            an out-of-band lookup.
        services: optional service endpoints for the new DID document.

    Returns:
        :class:`RotationCeremonyResult` with everything the caller
        needs to publish the new DID document, archive the succession
        statement, and (eventually) destroy the legacy private key.
    """
    if not effective_from:
        effective_from = datetime.now(timezone.utc).isoformat()

    if archival_keypair is not None:
        if archival_keypair.suite_id != ALG_SLH_DSA_SHA2_128S:
            from kestrel_sovereign.security.crypto_suite import CryptoSuiteError
            raise CryptoSuiteError(
                f"archival_keypair must be {ALG_SLH_DSA_SHA2_128S}; "
                f"got {archival_keypair.suite_id!r}"
            )
        if archival_verification_method is None:
            # Mint one ourselves — caller still owns the private key,
            # but at least the public-key Multikey is consistent.
            slh_suite = SLHDSASHA2128sSuite()
            archival_verification_method = build_verification_methods(
                # The archival VM controller is typically the new DID
                # (the successor underwrites the long-horizon signature
                # they're delegating to a hash-based key).
                f"did:web:{new_did_domain}:{new_did_slug}",
                [(slh_suite, archival_keypair.public_key)],
                kid_prefix="archival",
            )[0]

    # 1) New hybrid identity
    aka = list(also_known_as) if also_known_as is not None else [predecessor_did]
    new_identity = create_did_web_identity(
        new_did_domain,
        new_did_slug,
        extra_path_segments=extra_path_segments,
        also_known_as=aka,
        services=services,
    )

    # 2-4) Build + sign the succession statement
    statement = SuccessionStatement(
        predecessor_did=predecessor_did,
        successor_did=new_identity.did,
        effective_from=effective_from,
        reason=reason,
        predecessor_verification_methods=list(predecessor_verification_methods),
        successor_verification_methods=[
            m for m in new_identity.did_document["verificationMethod"]
        ],
    )
    statement = sign_predecessor(statement, [(predecessor_keypair, predecessor_kid)])
    statement = sign_successor(statement, [
        (
            new_identity.keypair.classical,
            new_identity.did_document["verificationMethod"][0]
                ["id"].rsplit("#", 1)[-1],
        ),
        (
            new_identity.keypair.pq,
            new_identity.did_document["verificationMethod"][1]
                ["id"].rsplit("#", 1)[-1],
        ),
    ])

    # 5) Optional archival countersignature
    if archival_keypair is not None:
        statement = archival_countersign(
            statement,
            archival_keypair,
            verification_method=dict(archival_verification_method),
        )

    # 6) Stamp id + timestamp
    statement = finalize(statement)

    # 7) Validate as a chain
    chain = build_chain([statement])

    # 8) Belt-and-suspenders: re-verify the statement before returning.
    # If we've made any plumbing mistake (kid mismatch, lost VMs, etc.)
    # we want to fail HERE, not when the consumer tries to use it.
    #
    # The successor's did:web document hasn't been published yet at this
    # point (the caller publishes it after the ceremony returns), so we
    # pass a self-attesting resolver: it returns the VMs we just minted
    # as the "published" doc. That's structurally honest because we ARE
    # the authority for those VMs at mint time. The third-party verifier
    # downstream will run ``did_web.resolve`` against the actual HTTPS
    # URL and catch any divergence between what was minted and what
    # got published.
    def _self_attesting_resolver(did: str) -> dict:
        if did == statement.successor_did:
            return {
                "id": did,
                "verificationMethod": list(statement.successor_verification_methods),
            }
        if did == statement.predecessor_did:
            return {
                "id": did,
                "verificationMethod": list(statement.predecessor_verification_methods),
            }
        raise ValueError(f"unknown DID at ceremony self-verify: {did!r}")

    verify_result = verify_succession(
        statement, did_web_resolver=_self_attesting_resolver,
    )
    if not verify_result.ok:
        raise RuntimeError(
            f"rotation ceremony produced an unverifiable succession "
            f"statement (this is a bug, not a key issue): "
            f"{verify_result.reason}"
        )

    return RotationCeremonyResult(
        predecessor_did=predecessor_did,
        new_identity=new_identity,
        succession_statement=statement,
        archival_keypair=archival_keypair,
        chain=chain,
    )


__all__ = [
    "RotationCeremonyResult",
    "run_rotation_ceremony",
]
