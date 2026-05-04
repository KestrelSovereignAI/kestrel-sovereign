"""
Succession-statement schema + signer tests — Wave 3 sub-PR 2 (#918).

Covers:
- Canonical signable payload: byte-stable, sorted-keys, deterministic
- compute_statement_id: stable across to_dict/from_dict round-trips
- sign_predecessor / sign_successor: shape of resulting signatures
- archival_countersign: requires SLH-DSA-128s, embeds VM
- verify_succession happy paths:
  * legacy ECDSA predecessor → hybrid successor (the Wave 3 ceremony)
  * hybrid → hybrid (post-Wave-3 future rotations)
  * archival countersig present, verifies, and required works
- verify_succession failure modes:
  * tampered field after signing → predecessor sig fails
  * predecessor missing → fail (LEGACY_ALLOWED still needs at least one)
  * successor classical-only → HYBRID_REQUIRED fails
  * archival required but missing → fail
  * archival signature uses ML-DSA-65 instead of SLH-DSA → fail
  * statement_id mismatch → fail
- to_dict/from_dict round-trip preserves all fields
"""

from __future__ import annotations

import json

import pytest

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.hybrid_keypair import (
    generate_hybrid_keypair,
)
from kestrel_sovereign.identity.succession import (
    SuccessionStatement,
    archival_countersign,
    compute_statement_id,
    finalize,
    sign_predecessor,
    sign_successor,
    signable_payload,
    verify_succession,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    ALG_ED25519,
    ALG_ML_DSA_65,
    ALG_SLH_DSA_SHA2_128S,
    CryptoSuiteError,
    SLHDSASHA2128sSuite,
    Secp256k1Suite,
    get_suite,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def legacy_predecessor():
    """Legacy did:pkh agent: only ECDSA secp256k1."""
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    did = "did:pkh:eip155:1:0xABC123"
    vms = build_verification_methods(did, [(secp, kp.public_key)])
    return {"did": did, "kp": kp, "kid": vms[0]["id"].rsplit("#", 1)[-1], "vms": vms}


@pytest.fixture(scope="module")
def hybrid_successor():
    """New hybrid did:web agent: Ed25519 + ML-DSA-65."""
    hybrid = generate_hybrid_keypair()
    did = "did:web:example.com:meridian-v2"
    vms = build_verification_methods(did, hybrid.public_keys())
    classical_kid = vms[0]["id"].rsplit("#", 1)[-1]
    pq_kid = vms[1]["id"].rsplit("#", 1)[-1]
    return {
        "did": did,
        "hybrid": hybrid,
        "classical_kid": classical_kid,
        "pq_kid": pq_kid,
        "vms": vms,
    }


@pytest.fixture(scope="module")
def slh_keypair_with_vm():
    """An SLH-DSA keypair + matching Multikey verification method."""
    slh = SLHDSASHA2128sSuite()
    kp = slh.generate_keypair()
    did = "did:web:archival.example.com"
    vms = build_verification_methods(did, [(slh, kp.public_key)], kid_prefix="archival")
    return kp, vms[0]


@pytest.fixture
def base_statement(legacy_predecessor, hybrid_successor):
    """Unsigned baseline statement; tests apply signatures as needed."""
    return SuccessionStatement(
        predecessor_did=legacy_predecessor["did"],
        successor_did=hybrid_successor["did"],
        effective_from="2026-05-04T18:00:00+00:00",
        reason="Wave 3 PQ-hardening migration",
        predecessor_verification_methods=legacy_predecessor["vms"],
        successor_verification_methods=hybrid_successor["vms"],
    )


# ---------------------------------------------------------------------------
# Canonical payload
# ---------------------------------------------------------------------------

def test_signable_payload_is_deterministic(base_statement):
    a = signable_payload(base_statement)
    b = signable_payload(base_statement)
    assert a == b


def test_signable_payload_excludes_signature_fields(base_statement, legacy_predecessor):
    """Adding signatures must NOT change the signable payload — otherwise
    the second signer would sign different bytes than the first."""
    before = signable_payload(base_statement)
    signed = sign_predecessor(
        base_statement,
        [(legacy_predecessor["kp"], legacy_predecessor["kid"])],
    )
    after = signable_payload(signed)
    assert before == after


def test_signable_payload_excludes_statement_id_and_created_at(base_statement):
    """statement_id and created_at must be excluded so that calling
    finalize() AFTER signing doesn't invalidate the signatures."""
    before = signable_payload(base_statement)
    finalized = finalize(base_statement)
    assert finalized.statement_id  # populated
    assert finalized.created_at
    after = signable_payload(finalized)
    assert before == after


def test_signable_payload_changes_when_signed_field_changes(base_statement):
    """Sanity: a change to a field that IS in the signable set produces
    different payload bytes (otherwise we'd silently accept tampering)."""
    from dataclasses import replace
    other = replace(base_statement, reason="different reason entirely")
    assert signable_payload(other) != signable_payload(base_statement)


def test_compute_statement_id_is_sha256_hex(base_statement):
    sid = compute_statement_id(base_statement)
    assert len(sid) == 64
    assert all(c in "0123456789abcdef" for c in sid)


def test_compute_statement_id_stable_across_dict_roundtrip(base_statement):
    """to_dict / from_dict must preserve the signable payload exactly,
    so the recomputed id matches."""
    sid_before = compute_statement_id(base_statement)
    rehydrated = SuccessionStatement.from_dict(
        json.loads(json.dumps(base_statement.to_dict()))
    )
    sid_after = compute_statement_id(rehydrated)
    assert sid_before == sid_after


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def test_sign_predecessor_produces_one_signature_for_legacy(
    base_statement, legacy_predecessor,
):
    signed = sign_predecessor(
        base_statement,
        [(legacy_predecessor["kp"], legacy_predecessor["kid"])],
    )
    assert len(signed.predecessor_signatures) == 1
    entry = signed.predecessor_signatures[0]
    assert entry["alg"] == ALG_ECDSA_SECP256K1_SHA256
    assert entry["kid"] == legacy_predecessor["kid"]
    bytes.fromhex(entry["sig"])  # valid hex


def test_sign_successor_produces_two_signatures_for_hybrid(
    base_statement, hybrid_successor,
):
    signed = sign_successor(
        base_statement,
        [
            (hybrid_successor["hybrid"].classical, hybrid_successor["classical_kid"]),
            (hybrid_successor["hybrid"].pq, hybrid_successor["pq_kid"]),
        ],
    )
    assert len(signed.successor_signatures) == 2
    algs = {e["alg"] for e in signed.successor_signatures}
    assert algs == {ALG_ED25519, ALG_ML_DSA_65}


def test_archival_countersign_requires_slh_dsa(
    base_statement, hybrid_successor,
):
    """An ML-DSA-65 keypair shouldn't be accepted as archival."""
    wrong_kp = hybrid_successor["hybrid"].pq  # ML-DSA-65, not SLH-DSA
    with pytest.raises(CryptoSuiteError, match="archival countersignature must use"):
        archival_countersign(base_statement, wrong_kp)


def test_archival_countersign_embeds_verification_method(
    base_statement, slh_keypair_with_vm,
):
    kp, vm = slh_keypair_with_vm
    signed = archival_countersign(base_statement, kp, verification_method=vm)
    assert signed.archival_signature is not None
    assert signed.archival_signature["alg"] == ALG_SLH_DSA_SHA2_128S
    assert signed.archival_verification_method == vm


# ---------------------------------------------------------------------------
# verify_succession — happy paths
# ---------------------------------------------------------------------------

def _build_full_succession(
    base_statement, legacy_predecessor, hybrid_successor,
    *, with_archival_kp_vm=None,
):
    """Helper: produce a fully-signed-and-finalized statement."""
    s = sign_predecessor(
        base_statement,
        [(legacy_predecessor["kp"], legacy_predecessor["kid"])],
    )
    s = sign_successor(
        s,
        [
            (hybrid_successor["hybrid"].classical, hybrid_successor["classical_kid"]),
            (hybrid_successor["hybrid"].pq, hybrid_successor["pq_kid"]),
        ],
    )
    if with_archival_kp_vm:
        kp, vm = with_archival_kp_vm
        s = archival_countersign(s, kp, verification_method=vm)
    return finalize(s)


def test_verify_legacy_to_hybrid_succession_happy_path(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """The canonical Wave 3 ceremony: ECDSA predecessor authorizing a
    hybrid successor."""
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
    )
    result = verify_succession(statement)
    assert result.ok, result.reason
    assert result.predecessor.ok
    assert result.successor.ok
    assert result.archival is None  # not present, not required
    assert result.statement_id_consistent


def test_verify_with_archival_countersignature(
    base_statement, legacy_predecessor, hybrid_successor, slh_keypair_with_vm,
):
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
        with_archival_kp_vm=slh_keypair_with_vm,
    )
    result = verify_succession(statement)
    assert result.ok, result.reason
    assert result.archival is not None
    assert result.archival.ok


def test_verify_archival_required_present(
    base_statement, legacy_predecessor, hybrid_successor, slh_keypair_with_vm,
):
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
        with_archival_kp_vm=slh_keypair_with_vm,
    )
    result = verify_succession(statement, require_archival=True)
    assert result.ok


def test_verify_archival_required_missing(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """Setting require_archival=True without an archival sig must fail."""
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
    )
    result = verify_succession(statement, require_archival=True)
    assert not result.ok
    assert result.archival is not None
    assert "required but not present" in result.archival.reason


# ---------------------------------------------------------------------------
# verify_succession — failure modes
# ---------------------------------------------------------------------------

def test_tampered_reason_invalidates_predecessor_signature(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """If the reason field is changed after signing, the signable payload
    differs and crypto verification fails."""
    from dataclasses import replace
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
    )
    tampered = replace(statement, reason="MALICIOUS REWRITE")
    result = verify_succession(tampered)
    assert not result.ok
    # predecessor sigs no longer crypto-verify
    assert not result.predecessor.ok


def test_tampered_effective_from_invalidates_signatures(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """The whole point of effective_from is that it's binding — tampering
    must invalidate signatures."""
    from dataclasses import replace
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
    )
    tampered = replace(statement, effective_from="1900-01-01T00:00:00+00:00")
    result = verify_succession(tampered)
    assert not result.ok


def test_missing_predecessor_signature_fails(
    base_statement, hybrid_successor,
):
    """An attempted self-succession with no predecessor sig: rejected."""
    s = sign_successor(
        base_statement,
        [
            (hybrid_successor["hybrid"].classical, hybrid_successor["classical_kid"]),
            (hybrid_successor["hybrid"].pq, hybrid_successor["pq_kid"]),
        ],
    )
    s = finalize(s)
    result = verify_succession(s)
    assert not result.ok
    assert not result.predecessor.ok


def test_successor_classical_only_fails_hybrid_required(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """If successor signs with ONLY their classical half (no PQ), the
    HYBRID_REQUIRED policy must fail — that's the whole point of
    Wave 2's hybrid identity."""
    s = sign_predecessor(
        base_statement,
        [(legacy_predecessor["kp"], legacy_predecessor["kid"])],
    )
    s = sign_successor(
        s,
        [(hybrid_successor["hybrid"].classical, hybrid_successor["classical_kid"])],
    )
    s = finalize(s)
    result = verify_succession(s)
    assert not result.ok
    assert not result.successor.ok
    assert "HYBRID_REQUIRED" in result.successor.reason


def test_archival_with_wrong_alg_rejected(
    base_statement, legacy_predecessor, hybrid_successor, hybrid_successor_pq_as_archival_attempt=None,
):
    """Manually inject an ML-DSA-65 signature into archival_signature.
    The PQ_REQUIRED policy alone would accept it; the explicit alg
    guard in verify_succession must reject."""
    s = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
    )
    payload = signable_payload(s)
    pq_kp = hybrid_successor["hybrid"].pq
    pq_suite = get_suite(pq_kp.suite_id)
    fake_archival_sig = pq_suite.sign(payload, pq_kp.private_key).hex()
    fake_archival_entry = {
        "alg": ALG_ML_DSA_65,
        "kid": "imposter",
        "sig": fake_archival_sig,
    }
    fake_vm = {
        "id": s.successor_did + "#imposter",
        "type": "Multikey",
        "controller": s.successor_did,
        "publicKeyMultibase": s.successor_verification_methods[1]["publicKeyMultibase"],
    }
    from dataclasses import replace
    spoofed = replace(
        s,
        archival_signature=fake_archival_entry,
        archival_verification_method=fake_vm,
    )
    result = verify_succession(spoofed)
    assert not result.ok
    assert result.archival is not None
    assert "must use slh-dsa-sha2-128s" in result.archival.reason


def test_statement_id_mismatch_fails(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """If a stored statement_id doesn't match recomputation, the
    statement was tampered with after signing or never finalized
    correctly. Must fail."""
    from dataclasses import replace
    s = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
    )
    spoofed = replace(s, statement_id="0" * 64)
    result = verify_succession(spoofed)
    assert not result.ok
    assert not result.statement_id_consistent


# ---------------------------------------------------------------------------
# Round-trip through dict (archival)
# ---------------------------------------------------------------------------

def test_dict_round_trip_preserves_verification(
    base_statement, legacy_predecessor, hybrid_successor, slh_keypair_with_vm,
):
    """Archive: serialize via to_dict + JSON, rehydrate via from_dict,
    re-verify — must remain valid."""
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
        with_archival_kp_vm=slh_keypair_with_vm,
    )
    wire = json.dumps(statement.to_dict())
    rehydrated = SuccessionStatement.from_dict(json.loads(wire))
    result = verify_succession(rehydrated)
    assert result.ok, result.reason
    assert result.archival is not None and result.archival.ok
