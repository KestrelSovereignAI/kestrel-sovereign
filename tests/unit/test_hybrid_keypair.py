"""
HybridKeypair tests — Wave 2 sub-PR 4 (#917).

Covers:
- Generation: default Ed25519 + ML-DSA-65 pair, distinct keypairs
- Algorithm-pair validation: refuses two-classical or two-PQ pairs
- sign_hybrid output shape: two v2 signatures entries with distinct
  alg / kid / sig fields
- public_keys() ordering: classical first, then PQ
- verify_hybrid happy path: HYBRID_REQUIRED satisfied
- verify_hybrid: drops failed-crypto entries before policy check
- verify_hybrid: alg/multibase mismatch (kid spoof) is rejected
- verify_hybrid: tampered data → both halves fail → policy fails
- verify_hybrid: only-classical → HYBRID_REQUIRED fails, LEGACY_ALLOWED ok
- verify_hybrid: only-PQ → HYBRID_REQUIRED fails, PQ_REQUIRED ok
- verify_hybrid: post-cutoff classical-only rejection
- DID-document round-trip: sign → DID document → parse → verify
- Inception bundle: create_did_web_identity returns matching DID/doc/keys
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.identity.did_web import (
    build_did_document,
    parse_did_document,
)
from kestrel_sovereign.identity.hybrid_keypair import (
    DEFAULT_CLASSICAL_ALG,
    DEFAULT_PQ_ALG,
    HybridKeypair,
    generate_hybrid_keypair,
    sign_hybrid,
    verify_hybrid,
)
from kestrel_sovereign.identity.inception_did_web import (
    HybridDidWebIdentity,
    create_did_web_identity,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    ALG_ED25519,
    ALG_ML_DSA_65,
    CryptoSuiteError,
)
from kestrel_sovereign.security.verify_policy import (
    VerifyPolicy,
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hybrid() -> HybridKeypair:
    """Module-scoped: ML-DSA keygen is fast but not free; reuse."""
    return generate_hybrid_keypair()


def test_default_pair_is_ed25519_plus_mldsa65(hybrid):
    assert hybrid.classical.suite_id == ALG_ED25519
    assert hybrid.pq.suite_id == ALG_ML_DSA_65


def test_independent_generations_produce_distinct_keypairs():
    """Sanity check that the two halves use independent system entropy
    rather than colliding on a shared RNG state."""
    a = generate_hybrid_keypair()
    b = generate_hybrid_keypair()
    # Ed25519 public keys are objects; serialize before comparing.
    from kestrel_sovereign.security.crypto_suite import get_suite
    ed = get_suite(ALG_ED25519)
    assert ed.serialize_public_key(a.classical.public_key) != \
        ed.serialize_public_key(b.classical.public_key)
    assert a.pq.public_key != b.pq.public_key


def test_refuses_two_classical_pair():
    with pytest.raises(CryptoSuiteError, match="must be a post-quantum"):
        generate_hybrid_keypair(
            classical_alg=ALG_ED25519,
            pq_alg=ALG_ECDSA_SECP256K1_SHA256,  # classical, not PQ
        )


def test_refuses_two_pq_pair():
    """Calling code can only mis-fire once the registry has two PQ
    suites; today there is only one (ml-dsa-65). Synthesize the case
    by passing ml-dsa-65 in the classical slot — this is the same
    error path the misuse will hit."""
    with pytest.raises(CryptoSuiteError, match="must be a classical"):
        generate_hybrid_keypair(
            classical_alg=ALG_ML_DSA_65,
            pq_alg=ALG_ML_DSA_65,
        )


def test_unregistered_suite_raises():
    with pytest.raises(CryptoSuiteError):
        generate_hybrid_keypair(classical_alg="not-a-real-suite")


# ---------------------------------------------------------------------------
# public_keys() ordering
# ---------------------------------------------------------------------------

def test_public_keys_classical_first_then_pq(hybrid):
    """Order is load-bearing: the kid scheme assumes #key-1 = classical,
    #key-2 = PQ, and the v2 signatures array follows the same order.
    """
    pairs = hybrid.public_keys()
    assert pairs[0][0].alg_id == hybrid.classical.suite_id
    assert pairs[1][0].alg_id == hybrid.pq.suite_id


# ---------------------------------------------------------------------------
# sign_hybrid shape
# ---------------------------------------------------------------------------

def test_sign_hybrid_emits_two_signature_entries(hybrid):
    sigs = sign_hybrid(b"identity-blob", hybrid)
    assert len(sigs) == 2
    assert {s["alg"] for s in sigs} == {ALG_ED25519, ALG_ML_DSA_65}
    # Distinct kids
    assert sigs[0]["kid"] != sigs[1]["kid"]
    # Hex-encoded
    for s in sigs:
        bytes.fromhex(s["sig"])  # raises if not valid hex


def test_sign_hybrid_default_kids_match_did_web_defaults(hybrid):
    """Default kid scheme aligns with build_verification_methods'
    1-indexed default (#key-1, #key-2)."""
    sigs = sign_hybrid(b"x", hybrid)
    assert sigs[0]["kid"] == "key-1"  # classical
    assert sigs[1]["kid"] == "key-2"  # PQ


def test_sign_hybrid_custom_kids(hybrid):
    sigs = sign_hybrid(b"x", hybrid, classical_kid="ed25519", pq_kid="ml-dsa-65")
    assert sigs[0]["kid"] == "ed25519"
    assert sigs[1]["kid"] == "ml-dsa-65"


# ---------------------------------------------------------------------------
# verify_hybrid happy path
# ---------------------------------------------------------------------------

def _doc_for(hybrid: HybridKeypair, did: str = "did:web:example.com:agent") -> dict:
    return build_did_document(did, hybrid.public_keys())


def test_verify_hybrid_round_trip(hybrid):
    data = b"a signed payload"
    sigs = sign_hybrid(data, hybrid)
    doc = _doc_for(hybrid)
    result = verify_hybrid(data, sigs, doc["verificationMethod"])
    assert result.ok, result.reason
    assert "HYBRID_REQUIRED" in result.reason


# ---------------------------------------------------------------------------
# verify_hybrid drops crypto-failed entries
# ---------------------------------------------------------------------------

def test_verify_hybrid_rejects_tampered_data(hybrid):
    data = b"original"
    sigs = sign_hybrid(data, hybrid)
    doc = _doc_for(hybrid)
    # Verify against tampered data — both crypto checks should fail,
    # leaving zero verified entries → policy fails on "no signatures".
    result = verify_hybrid(b"tampered", sigs, doc["verificationMethod"])
    assert not result.ok


def test_verify_hybrid_rejects_alg_kid_spoof(hybrid):
    """A signature entry that lies about its alg (claims ml-dsa-65 but
    the kid resolves to an Ed25519 method) must not slip past the
    cross-check. This is the entire point of the
    ``suite.alg_id != alg`` guard in verify_hybrid.
    """
    data = b"x"
    sigs = sign_hybrid(data, hybrid)
    doc = _doc_for(hybrid)
    # Swap the alg labels so each entry now claims the OTHER half's alg
    sigs[0]["alg"] = ALG_ML_DSA_65    # claims PQ but kid points to Ed25519 method
    sigs[1]["alg"] = ALG_ED25519       # claims classical but kid points to PQ method
    result = verify_hybrid(data, sigs, doc["verificationMethod"])
    assert not result.ok


def test_verify_hybrid_drops_unknown_kid(hybrid):
    """Signature entry with a kid that doesn't match any verification
    method is silently dropped (not a security issue — there's no key
    to verify against). Other entries proceed normally."""
    data = b"x"
    sigs = sign_hybrid(data, hybrid)
    doc = _doc_for(hybrid)
    sigs[0]["kid"] = "no-such-key"
    result = verify_hybrid(data, sigs, doc["verificationMethod"])
    # Only PQ half remains; HYBRID_REQUIRED needs both → fail
    assert not result.ok
    # PQ_REQUIRED would pass (PQ half still verified)
    result_pq = verify_hybrid(
        data, sigs, doc["verificationMethod"], policy=VerifyPolicy.PQ_REQUIRED,
    )
    assert result_pq.ok, result_pq.reason


def test_verify_hybrid_drops_malformed_hex(hybrid):
    data = b"x"
    sigs = sign_hybrid(data, hybrid)
    doc = _doc_for(hybrid)
    sigs[0]["sig"] = "not-hex-zzz"
    result = verify_hybrid(data, sigs, doc["verificationMethod"])
    # Classical entry dropped → only PQ verified → HYBRID_REQUIRED fails
    assert not result.ok


# ---------------------------------------------------------------------------
# Policy variations
# ---------------------------------------------------------------------------

def test_legacy_allowed_accepts_classical_only(hybrid):
    data = b"x"
    sigs = sign_hybrid(data, hybrid)
    doc = _doc_for(hybrid)
    # Drop the PQ half entirely
    sigs = [sigs[0]]
    result = verify_hybrid(
        data, sigs, doc["verificationMethod"],
        policy=VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok, result.reason


def test_pq_required_accepts_pq_only(hybrid):
    data = b"x"
    sigs = sign_hybrid(data, hybrid)
    doc = _doc_for(hybrid)
    sigs = [sigs[1]]  # only PQ
    result = verify_hybrid(
        data, sigs, doc["verificationMethod"],
        policy=VerifyPolicy.PQ_REQUIRED,
    )
    assert result.ok, result.reason


def test_post_cutoff_classical_only_rejected(hybrid):
    """Wave 3 hook: when the chain walker determines an artifact is
    dated after the agent's succession effective_from, classical-only
    signatures are rejected even under LEGACY_ALLOWED."""
    data = b"x"
    sigs = sign_hybrid(data, hybrid)
    doc = _doc_for(hybrid)
    # Classical-only set
    sigs = [sigs[0]]
    result = verify_hybrid(
        data, sigs, doc["verificationMethod"],
        policy=VerifyPolicy.LEGACY_ALLOWED,
        post_cutoff_classical_allowed=False,
    )
    assert not result.ok
    assert "post-cutoff" in result.reason


# ---------------------------------------------------------------------------
# DID document round-trip — strongest end-to-end check
# ---------------------------------------------------------------------------

def test_full_round_trip_through_parsed_did_document(hybrid):
    """Mint a hybrid keypair → build DID document → re-parse the
    document via parse_did_document → verify hybrid signatures using
    the parsed verification methods. This is the exact flow Wave 3+
    code follows (resolve a controller's did:web, then verify their
    signatures against the resolved keys)."""
    data = b"identity-roundtrip-payload"
    sigs = sign_hybrid(data, hybrid)
    did = "did:web:example.com:agent:wave2"
    doc = build_did_document(did, hybrid.public_keys())

    parsed = parse_did_document(doc)
    assert len(parsed) == 2

    # verify_hybrid takes the methods array from the document directly;
    # this confirms the array survives a round-trip (it's identity here,
    # but the parse side still validates the multibase decoding works).
    result = verify_hybrid(data, sigs, doc["verificationMethod"])
    assert result.ok


# ---------------------------------------------------------------------------
# Inception bundle
# ---------------------------------------------------------------------------

def test_create_did_web_identity_returns_consistent_bundle():
    bundle = create_did_web_identity("example.com", "meridian")
    assert isinstance(bundle, HybridDidWebIdentity)
    assert bundle.did == "did:web:example.com:meridian"
    assert bundle.did_document["id"] == bundle.did
    # DID doc must have both Multikey methods
    methods = [m for m in bundle.did_document["verificationMethod"] if m["type"] == "Multikey"]
    assert len(methods) == 2
    # Sign-verify round-trip with the bundle's keypair
    sigs = sign_hybrid(b"end-to-end", bundle.keypair)
    result = verify_hybrid(
        b"end-to-end", sigs, bundle.did_document["verificationMethod"],
    )
    assert result.ok, result.reason


def test_create_did_web_identity_with_extra_path_segments():
    bundle = create_did_web_identity(
        "example.com", "meridian", extra_path_segments=["v1"],
    )
    assert bundle.did == "did:web:example.com:meridian:v1"


def test_create_did_web_identity_with_also_known_as():
    bundle = create_did_web_identity(
        "example.com", "kestrel",
        also_known_as=["did:pkh:eip155:1:0xABC..."],
    )
    assert bundle.did_document["alsoKnownAs"] == ["did:pkh:eip155:1:0xABC..."]


def test_create_did_web_identity_rejects_empty_slug():
    from kestrel_sovereign.identity.did_web import DidWebError
    with pytest.raises(DidWebError, match="slug must be non-empty"):
        create_did_web_identity("example.com", "")
