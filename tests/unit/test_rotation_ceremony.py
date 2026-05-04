"""
Rotation ceremony tests — Wave 3 sub-PR 4 (#918).

The ceremony composes existing primitives (#960, #963, #964) into one
operational entry point. Tests focus on:

- End-to-end ceremony for a legacy did:pkh agent → hybrid did:web
- Default ``alsoKnownAs`` links the predecessor DID into the new
  DID document (so verifiers can walk back to the chain)
- Default ``effective_from`` is "now" (UTC, tz-aware)
- Optional archival countersignature is plumbed through
- Output ``chain`` is pre-validated and passes
  :func:`verify_artifact_against_chain` end-to-end
- An artifact signed under the new hybrid identity verifies against
  the ceremony's chain at a post-cutoff timestamp under HYBRID_REQUIRED
- Ceremony refuses a non-SLH-DSA archival keypair
- Belt-and-suspenders: ceremony re-verifies internally and raises if
  signature plumbing is broken
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
from kestrel_sovereign.identity.rotation_ceremony import (
    run_rotation_ceremony,
)
from kestrel_sovereign.identity.succession import verify_succession
from kestrel_sovereign.identity.succession_chain import (
    verify_artifact_against_chain,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_SLH_DSA_SHA2_128S,
    CryptoSuiteError,
    Secp256k1Suite,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy


def _self_attesting_resolver(statement):
    """Test-only resolver — see test_succession.py for rationale.

    Returns the statement's own VMs as the "published" doc. Production
    callers use ``identity.did_web.resolve``.
    """
    def _resolve(did):
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
        raise ValueError(f"unknown did: {did!r}")
    return _resolve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def legacy_kestrel():
    """Legacy ECDSA-only Kestrel #1-style identity.

    DID derived from the keypair via ``public_key_to_ethereum_address``
    so the ``verify_did_binding`` check from #963 holds.
    """
    from kestrel_sovereign.inception_service import (
        public_key_to_ethereum_address,
    )
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(kp.public_key)
    did = f"did:pkh:eip155:1:{address}"
    vms = build_verification_methods(did, [(secp, kp.public_key)])
    return {
        "did": did,
        "kp": kp,
        "kid": vms[0]["id"].rsplit("#", 1)[-1],
        "vms": vms,
    }


# ---------------------------------------------------------------------------
# End-to-end ceremony
# ---------------------------------------------------------------------------

def test_ceremony_produces_verifiable_succession_statement(legacy_kestrel):
    """Most important test: run the ceremony, get back a result, and
    confirm the succession statement verifies cryptographically."""
    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel["did"],
        predecessor_keypair=legacy_kestrel["kp"],
        predecessor_kid=legacy_kestrel["kid"],
        predecessor_verification_methods=legacy_kestrel["vms"],
        new_did_domain="example.com",
        new_did_slug="kestrel-v2",
        reason="Wave 3 PQ-hardening migration",
    )

    # New identity has the expected DID
    assert result.new_identity.did == "did:web:example.com:kestrel-v2"

    # Statement crypto-verifies. Self-attesting resolver because the
    # successor did:web URL hasn't been "published" in this test —
    # but the embedded VMs are exactly what the ceremony just minted,
    # which is what a real production resolver against the deployed
    # URL would also return.
    verify = verify_succession(
        result.succession_statement,
        did_web_resolver=_self_attesting_resolver(result.succession_statement),
    )
    assert verify.ok, verify.reason


def test_ceremony_default_aka_links_predecessor(legacy_kestrel):
    """The default alsoKnownAs entry for the new DID document is the
    predecessor DID — a discoverability hint for verifiers."""
    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel["did"],
        predecessor_keypair=legacy_kestrel["kp"],
        predecessor_kid=legacy_kestrel["kid"],
        predecessor_verification_methods=legacy_kestrel["vms"],
        new_did_domain="example.com",
        new_did_slug="kestrel-v2",
        reason="aka test",
    )
    assert result.new_identity.did_document["alsoKnownAs"] == [legacy_kestrel["did"]]


def test_ceremony_explicit_aka_overrides_default(legacy_kestrel):
    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel["did"],
        predecessor_keypair=legacy_kestrel["kp"],
        predecessor_kid=legacy_kestrel["kid"],
        predecessor_verification_methods=legacy_kestrel["vms"],
        new_did_domain="example.com",
        new_did_slug="kestrel-v2",
        reason="explicit aka",
        also_known_as=["did:web:other.example:custom"],
    )
    assert result.new_identity.did_document["alsoKnownAs"] == [
        "did:web:other.example:custom",
    ]


def test_ceremony_default_effective_from_is_now_utc(legacy_kestrel):
    """No ``effective_from`` means "now"; we just verify the result is
    a tz-aware UTC ISO 8601 string within a sensible window of now."""
    before = datetime.now(timezone.utc)
    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel["did"],
        predecessor_keypair=legacy_kestrel["kp"],
        predecessor_kid=legacy_kestrel["kid"],
        predecessor_verification_methods=legacy_kestrel["vms"],
        new_did_domain="example.com",
        new_did_slug="kestrel-v2",
        reason="default effective_from",
    )
    after = datetime.now(timezone.utc)

    eff = result.succession_statement.effective_from
    assert eff.endswith("+00:00") or eff.endswith("Z")
    parsed = datetime.fromisoformat(eff.replace("Z", "+00:00"))
    assert before <= parsed <= after


def test_ceremony_explicit_effective_from_is_used(legacy_kestrel):
    explicit = "2026-12-31T23:59:59+00:00"
    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel["did"],
        predecessor_keypair=legacy_kestrel["kp"],
        predecessor_kid=legacy_kestrel["kid"],
        predecessor_verification_methods=legacy_kestrel["vms"],
        new_did_domain="example.com",
        new_did_slug="kestrel-v2",
        reason="explicit time",
        effective_from=explicit,
    )
    assert result.succession_statement.effective_from == explicit


# ---------------------------------------------------------------------------
# Archival countersignature path
# ---------------------------------------------------------------------------

def test_ceremony_with_archival_countersignature(legacy_kestrel):
    """Archival keypair plumbed all the way through; the resulting
    statement carries an archival_signature that verifies."""
    slh = SLHDSASHA2128sSuite()
    archival_kp = slh.generate_keypair()

    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel["did"],
        predecessor_keypair=legacy_kestrel["kp"],
        predecessor_kid=legacy_kestrel["kid"],
        predecessor_verification_methods=legacy_kestrel["vms"],
        new_did_domain="example.com",
        new_did_slug="kestrel-v2",
        reason="archival included",
        archival_keypair=archival_kp,
    )

    assert result.succession_statement.archival_signature is not None
    assert result.archival_keypair is archival_kp

    # Verify with require_archival=True passes
    verify = verify_succession(
        result.succession_statement,
        require_archival=True,
        did_web_resolver=_self_attesting_resolver(result.succession_statement),
    )
    assert verify.ok, verify.reason


def test_ceremony_refuses_non_slhdsa_archival_keypair(legacy_kestrel):
    """Hand the ceremony an ML-DSA-65 keypair where SLH-DSA is required."""
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
    bad_kp = generate_hybrid_keypair().pq  # ML-DSA-65, not SLH-DSA
    with pytest.raises(CryptoSuiteError, match="archival_keypair must be"):
        run_rotation_ceremony(
            predecessor_did=legacy_kestrel["did"],
            predecessor_keypair=legacy_kestrel["kp"],
            predecessor_kid=legacy_kestrel["kid"],
            predecessor_verification_methods=legacy_kestrel["vms"],
            new_did_domain="example.com",
            new_did_slug="kestrel-v2",
            reason="bad archival",
            archival_keypair=bad_kp,
        )


# ---------------------------------------------------------------------------
# End-to-end chain walker
# ---------------------------------------------------------------------------

def test_ceremony_chain_passes_artifact_verification_post_cutoff(legacy_kestrel):
    """Strongest end-to-end test: ceremony output is plugged into the
    chain walker and used to verify a hybrid-signed artifact dated
    AFTER the succession's effective_from. This is the exact flow
    Wave 3 deployment will exercise.
    """
    explicit_cutoff = "2026-05-04T18:00:00+00:00"
    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel["did"],
        predecessor_keypair=legacy_kestrel["kp"],
        predecessor_kid=legacy_kestrel["kid"],
        predecessor_verification_methods=legacy_kestrel["vms"],
        new_did_domain="example.com",
        new_did_slug="kestrel-v2",
        reason="end-to-end",
        effective_from=explicit_cutoff,
    )

    payload = b"a post-cutoff artifact signed by the new hybrid identity"
    classical_kid = result.new_identity.did_document["verificationMethod"][0]["id"]\
        .rsplit("#", 1)[-1]
    pq_kid = result.new_identity.did_document["verificationMethod"][1]["id"]\
        .rsplit("#", 1)[-1]
    artifact_signatures = sign_hybrid(
        payload, result.new_identity.keypair,
        classical_kid=classical_kid, pq_kid=pq_kid,
    )

    chain_verdict = verify_artifact_against_chain(
        root_did=legacy_kestrel["did"],
        root_verification_methods=legacy_kestrel["vms"],
        chain=result.chain,
        artifact_timestamp="2026-06-01T00:00:00+00:00",
        artifact_payload=payload,
        artifact_signatures=artifact_signatures,
        policy=VerifyPolicy.HYBRID_REQUIRED,
    )

    assert chain_verdict.ok, chain_verdict.reason
    assert chain_verdict.active_identity.did == result.new_identity.did
    assert chain_verdict.active_identity.post_cutoff


def test_ceremony_chain_rejects_classical_only_post_cutoff(legacy_kestrel):
    """Same ceremony, but artifact signed with ONLY the new identity's
    classical half. Post-cutoff classical-only must fail. This is what
    the temporal cutoff was built for."""
    explicit_cutoff = "2026-05-04T18:00:00+00:00"
    result = run_rotation_ceremony(
        predecessor_did=legacy_kestrel["did"],
        predecessor_keypair=legacy_kestrel["kp"],
        predecessor_kid=legacy_kestrel["kid"],
        predecessor_verification_methods=legacy_kestrel["vms"],
        new_did_domain="example.com",
        new_did_slug="kestrel-v2",
        reason="cutoff-test",
        effective_from=explicit_cutoff,
    )

    payload = b"classical-only post-cutoff"
    classical_kp = result.new_identity.keypair.classical
    classical_kid = result.new_identity.did_document["verificationMethod"][0]["id"]\
        .rsplit("#", 1)[-1]
    from kestrel_sovereign.security.crypto_suite import get_suite
    classical_suite = get_suite(classical_kp.suite_id)
    sig = classical_suite.sign(payload, classical_kp.private_key)
    artifact_signatures = [{
        "alg": classical_suite.alg_id,
        "kid": classical_kid,
        "sig": sig.hex(),
    }]

    chain_verdict = verify_artifact_against_chain(
        root_did=legacy_kestrel["did"],
        root_verification_methods=legacy_kestrel["vms"],
        chain=result.chain,
        artifact_timestamp="2026-06-01T00:00:00+00:00",  # post-cutoff
        artifact_payload=payload,
        artifact_signatures=artifact_signatures,
        policy=VerifyPolicy.LEGACY_ALLOWED,  # most permissive available
    )
    assert not chain_verdict.ok
    assert "post-cutoff" in chain_verdict.policy_result.reason
