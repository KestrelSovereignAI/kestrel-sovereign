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

P1 review-finding regression coverage:
- Attacker takeover scenario: forged statement claiming a victim's
  did:pkh with attacker-controlled keys is rejected by the new
  predecessor-DID-binding check (verify_did_binding)
- did:pkh address-binding rule rejects mismatched VMs
- did:key multibase-binding rule rejects mismatched VMs
- did:web binding fails-closed without an explicit resolver
- did:web binding accepts when resolver returns matching VMs
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
    """Legacy did:pkh agent: only ECDSA secp256k1.

    The DID is derived from the keypair via the same Ethereum-address
    rule the inception_service uses, so the verify_did_binding check
    holds.
    """
    from kestrel_sovereign.inception_service import (
        public_key_to_ethereum_address,
    )
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(kp.public_key)
    did = f"did:pkh:eip155:1:{address}"
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


def _self_attesting_resolver(statement):
    """Test-only resolver: returns the statement's own successor VMs as
    the "published" DID document for the successor's did:web URI.

    Real production callers MUST use ``identity.did_web.resolve`` so the
    resolver fetches the actual DID document from HTTPS — that's the
    binding's whole point. Tests don't have a real network and would
    otherwise have to spin up an HTTP server, so this synthesizes the
    document directly. The result is structurally identical to what a
    legitimate resolver would return for a correctly-published DID;
    only the publication step is faked.
    """
    successor_doc = {
        "id": statement.successor_did,
        "verificationMethod": [
            dict(vm) for vm in statement.successor_verification_methods
        ],
    }
    predecessor_doc = {
        "id": statement.predecessor_did,
        "verificationMethod": [
            dict(vm) for vm in statement.predecessor_verification_methods
        ],
    }
    def _resolve(did):
        if did == statement.successor_did:
            return successor_doc
        if did == statement.predecessor_did:
            return predecessor_doc
        raise ValueError(f"unknown did in test resolver: {did!r}")
    return _resolve


def test_verify_legacy_to_hybrid_succession_happy_path(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """The canonical Wave 3 ceremony: ECDSA predecessor authorizing a
    hybrid successor."""
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
    )
    result = verify_succession(statement, did_web_resolver=_self_attesting_resolver(statement))
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
    result = verify_succession(statement, did_web_resolver=_self_attesting_resolver(statement))
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
    result = verify_succession(statement, require_archival=True, did_web_resolver=_self_attesting_resolver(statement))
    assert result.ok


def test_verify_archival_required_missing(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """Setting require_archival=True without an archival sig must fail."""
    statement = _build_full_succession(
        base_statement, legacy_predecessor, hybrid_successor,
    )
    result = verify_succession(statement, require_archival=True, did_web_resolver=_self_attesting_resolver(statement))
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
    result = verify_succession(tampered, did_web_resolver=_self_attesting_resolver(tampered))
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
    result = verify_succession(tampered, did_web_resolver=_self_attesting_resolver(tampered))
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
    result = verify_succession(s, did_web_resolver=_self_attesting_resolver(s))
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
    result = verify_succession(s, did_web_resolver=_self_attesting_resolver(s))
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
    result = verify_succession(spoofed, did_web_resolver=_self_attesting_resolver(spoofed))
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
    result = verify_succession(spoofed, did_web_resolver=_self_attesting_resolver(spoofed))
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
    result = verify_succession(rehydrated, did_web_resolver=_self_attesting_resolver(rehydrated))
    assert result.ok, result.reason
    assert result.archival is not None and result.archival.ok


# ---------------------------------------------------------------------------
# Predecessor DID binding (P1 regression: codex review of #963)
# ---------------------------------------------------------------------------

def test_attacker_takeover_with_forged_did_pkh_rejected(hybrid_successor):
    """The codex-flagged attack: build a statement claiming a victim's
    did:pkh, embed the attacker's own keys as predecessor_verification_
    methods, sign with the attacker's key, and ship.

    Pre-fix: verify_succession returned ok=True because the signatures
    DID crypto-verify against the embedded VMs. The verifier never
    cross-checked that the embedded VMs actually correspond to the
    claimed DID.

    Post-fix: verify_did_binding rejects this — the keccak hash of the
    attacker's pubkey does NOT equal the victim's address.
    """
    from dataclasses import replace as _replace

    secp = Secp256k1Suite()
    attacker_kp = secp.generate_keypair()
    # Victim DID is unrelated to the attacker's keypair
    victim_did = "did:pkh:eip155:1:0x1234567890123456789012345678901234567890"
    # But VMs are the attacker's keys — controller field can be anything
    # the attacker chooses; the embedded VMs are not authenticated by
    # the binding check until this fix.
    attacker_vms = build_verification_methods(victim_did, [(secp, attacker_kp.public_key)])
    attacker_kid = attacker_vms[0]["id"].rsplit("#", 1)[-1]

    forged = SuccessionStatement(
        predecessor_did=victim_did,
        successor_did=hybrid_successor["did"],
        effective_from="2026-05-04T18:00:00+00:00",
        reason="ATTACKER FORGED THIS",
        predecessor_verification_methods=attacker_vms,
        successor_verification_methods=hybrid_successor["vms"],
    )
    forged = sign_predecessor(forged, [(attacker_kp, attacker_kid)])
    forged = sign_successor(forged, [
        (hybrid_successor["hybrid"].classical, hybrid_successor["classical_kid"]),
        (hybrid_successor["hybrid"].pq, hybrid_successor["pq_kid"]),
    ])
    forged = finalize(forged)

    result = verify_succession(forged, did_web_resolver=_self_attesting_resolver(forged))
    assert not result.ok, "attacker takeover with forged DID must be rejected"
    assert not result.predecessor_did_bound
    assert "binding" in result.reason


def test_did_pkh_binding_check_with_correct_address(legacy_predecessor):
    from kestrel_sovereign.identity.succession import verify_did_binding
    ok, reason = verify_did_binding(legacy_predecessor["did"], legacy_predecessor["vms"])
    assert ok, reason


def test_did_key_binding_check_with_matching_multibase():
    """did:key:zX — a VM with matching publicKeyMultibase satisfies binding."""
    from kestrel_sovereign.identity.succession import verify_did_binding
    from kestrel_sovereign.security.multikey import public_key_to_multibase

    ed = get_suite("ed25519")
    kp = ed.generate_keypair()
    multibase = public_key_to_multibase(ed, kp.public_key)
    did = f"did:key:{multibase}"
    vms = [{
        "id": f"{did}#0",
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": multibase,
    }]
    ok, reason = verify_did_binding(did, vms)
    assert ok, reason


def test_did_key_binding_rejects_mismatched_multibase():
    from kestrel_sovereign.identity.succession import verify_did_binding
    from kestrel_sovereign.security.multikey import public_key_to_multibase

    ed = get_suite("ed25519")
    real_kp = ed.generate_keypair()
    other_kp = ed.generate_keypair()
    real_mb = public_key_to_multibase(ed, real_kp.public_key)
    other_mb = public_key_to_multibase(ed, other_kp.public_key)
    did = f"did:key:{real_mb}"
    # VM holds a DIFFERENT key than the DID claims
    vms = [{
        "id": f"{did}#0",
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": other_mb,
    }]
    ok, reason = verify_did_binding(did, vms)
    assert not ok
    assert "did:key binding FAILED" in reason


def test_did_web_binding_fails_closed_without_resolver():
    """did:web binding cannot be checked without a resolver — fail loud
    rather than silently accept embedded VMs the attacker chose."""
    from kestrel_sovereign.identity.succession import verify_did_binding

    did = "did:web:attacker.example:agent"
    vms = [{
        "id": f"{did}#key-1",
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": "z6MkjGenericMultibaseValueHere",
    }]
    ok, reason = verify_did_binding(did, vms)
    assert not ok
    assert "did:web binding requires a resolver" in reason


def test_did_web_binding_with_matching_resolver_passes():
    from kestrel_sovereign.identity.succession import verify_did_binding

    did = "did:web:legit.example:agent"
    vms = [{
        "id": f"{did}#key-1",
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": "z6MkABCpublishedAndMatching",
    }]
    # Resolver returns a doc with matching VMs
    def resolver(d):
        assert d == did
        return {"id": did, "verificationMethod": vms}
    ok, reason = verify_did_binding(did, vms, did_web_resolver=resolver)
    assert ok, reason


def test_did_web_binding_rejects_mismatched_resolver_response():
    from kestrel_sovereign.identity.succession import verify_did_binding

    did = "did:web:legit.example:agent"
    embedded_vms = [{
        "id": f"{did}#key-1",
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": "z6MkATTACKERkey",
    }]
    published_vms = [{
        "id": f"{did}#key-1",
        "type": "Multikey",
        "controller": did,
        "publicKeyMultibase": "z6MkPUBLISHEDdifferent",
    }]
    def resolver(d):
        return {"id": did, "verificationMethod": published_vms}
    ok, reason = verify_did_binding(did, embedded_vms, did_web_resolver=resolver)
    assert not ok
    assert "publicKeyMultibase does not match" in reason


def test_unknown_did_method_rejected():
    from kestrel_sovereign.identity.succession import verify_did_binding
    ok, reason = verify_did_binding("did:unknown:foo", [])
    assert not ok
    assert "unsupported DID method" in reason


# ---------------------------------------------------------------------------
# Successor DID binding (P1 codex follow-up review of #963)
# ---------------------------------------------------------------------------

def test_successor_did_mismatch_rejected(legacy_predecessor):
    """Codex P1 follow-up: a fully signed statement whose ``successor_did``
    is one DID but whose embedded successor VMs are controlled by an
    attacker (different DID) must be rejected.

    Pre-fix: only the predecessor side was bound. ok=True returned even
    when the successor_did was 'did:web:victim.example' but the VMs
    were 'did:web:attacker.example'. A consumer indexing on
    successor_did would be misled.

    Post-fix: successor binding runs symmetrically with predecessor
    binding; mismatch fails-closed.
    """
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair

    # Attacker's successor identity, published as did:web:attacker.example
    attacker_hybrid = generate_hybrid_keypair()
    attacker_real_did = "did:web:attacker.example"
    attacker_vms = build_verification_methods(attacker_real_did, attacker_hybrid.public_keys())
    attacker_classical_kid = attacker_vms[0]["id"].rsplit("#", 1)[-1]
    attacker_pq_kid = attacker_vms[1]["id"].rsplit("#", 1)[-1]

    # Statement claims a DIFFERENT successor DID
    claimed_victim_did = "did:web:victim.example"

    s = SuccessionStatement(
        predecessor_did=legacy_predecessor["did"],
        successor_did=claimed_victim_did,    # claimed
        effective_from="2026-05-04T18:00:00+00:00",
        reason="successor takeover attempt",
        predecessor_verification_methods=legacy_predecessor["vms"],
        successor_verification_methods=attacker_vms,  # actually attacker's
    )
    s = sign_predecessor(s, [(legacy_predecessor["kp"], legacy_predecessor["kid"])])
    s = sign_successor(s, [
        (attacker_hybrid.classical, attacker_classical_kid),
        (attacker_hybrid.pq, attacker_pq_kid),
    ])
    s = finalize(s)

    # Self-attesting resolver returns the statement's claimed successor
    # DID with the embedded VMs — but the binding check sees the DID
    # mismatch via the published-doc resolution and fails. Actually
    # in our self-attesting setup the resolver returns the claimed DID
    # with the attacker's VMs (because that's what's embedded), so the
    # binding check has no published-doc to compare against. To
    # actually catch the cross-DID issue, we use a resolver that
    # returns a published doc for the claimed DID with DIFFERENT VMs
    # (the legitimate ones the victim would have published).
    def _victim_resolver(did):
        if did == claimed_victim_did:
            # Victim's REAL published VMs (different from attacker's)
            real_hybrid = generate_hybrid_keypair()
            real_vms = build_verification_methods(claimed_victim_did, real_hybrid.public_keys())
            return {"id": claimed_victim_did, "verificationMethod": real_vms}
        if did == legacy_predecessor["did"]:
            return {"id": did, "verificationMethod": legacy_predecessor["vms"]}
        raise ValueError(did)

    result = verify_succession(s, did_web_resolver=_victim_resolver)
    assert not result.ok, "successor takeover with mismatched DID must be rejected"
    assert not result.successor_did_bound
    assert "successor DID binding" in result.reason


# ---------------------------------------------------------------------------
# statement_id strictness (P2 codex follow-up review of #963)
# ---------------------------------------------------------------------------

def test_duplicate_kid_takeover_rejected(legacy_predecessor, hybrid_successor):
    """Codex P1 (third round): an attacker embeds the victim's REAL VM
    (to satisfy did:pkh binding via any-match) AND an attacker-controlled
    VM with the SAME ``#key-1`` fragment. Pre-fix: the attacker's VM
    silently overwrote the victim's in ``methods_by_kid``, so the
    attacker's signature verified for the victim's DID.

    Post-fix: ``_check_unique_vm_kids`` runs first in
    ``verify_did_binding`` and refuses any VM list with duplicate kid
    fragments.
    """
    # Attacker: their own secp256k1 keypair
    att_secp = Secp256k1Suite()
    att_kp = att_secp.generate_keypair()
    att_vm = build_verification_methods(
        legacy_predecessor["did"], [(att_secp, att_kp.public_key)],
    )[0]
    # Force the attacker's VM to share the victim's kid fragment
    att_vm["id"] = legacy_predecessor["vms"][0]["id"]  # same id → same kid

    # VMs list: legitimate FIRST (passes binding any-match), attacker SECOND
    pred_vms = list(legacy_predecessor["vms"]) + [att_vm]
    att_kid = att_vm["id"].rsplit("#", 1)[-1]

    s = SuccessionStatement(
        predecessor_did=legacy_predecessor["did"],
        successor_did=hybrid_successor["did"],
        effective_from="2026-05-04T18:00:00+00:00",
        reason="duplicate-kid takeover attempt",
        predecessor_verification_methods=pred_vms,
        successor_verification_methods=hybrid_successor["vms"],
    )
    s = sign_predecessor(s, [(att_kp, att_kid)])  # attacker signs
    s = sign_successor(s, [
        (hybrid_successor["hybrid"].classical, hybrid_successor["classical_kid"]),
        (hybrid_successor["hybrid"].pq, hybrid_successor["pq_kid"]),
    ])
    s = finalize(s)

    result = verify_succession(s, did_web_resolver=_self_attesting_resolver(s))
    assert not result.ok, "duplicate-kid takeover must be rejected"
    assert not result.predecessor_did_bound
    assert "duplicate kid" in result.reason


def test_extra_unbound_vm_decoy_rejected(legacy_predecessor, hybrid_successor):
    """Codex P1 (round 4): the duplicate-kid fix only stops attackers
    who reuse the same kid. An attacker can still:

    1. Include the victim's REAL VM under one kid (passes any-match
       binding because the real key derives the address)
    2. Include their own attacker secp256k1 VM under a DIFFERENT kid
       (passes unique-kid check because no collision)
    3. Sign with the attacker's key under the attacker's kid

    Pre-fix (any-match binding): predecessor side returned ok because
    the attacker's signature crypto-verifies against THEIR VM, and
    binding passed via any-match on the victim's decoy VM.

    Post-fix: ``_verify_did_pkh_eip155_binding`` requires EVERY VM in
    the list to derive the claimed address. The attacker's VM is
    rejected because its derived address doesn't match the victim's.
    """
    # Attacker's own keypair
    att_secp = Secp256k1Suite()
    att_kp = att_secp.generate_keypair()
    # Attacker mounts an UNBOUND VM (different address) under a unique kid
    att_vm = build_verification_methods(
        legacy_predecessor["did"],  # claims victim's DID...
        [(att_secp, att_kp.public_key)],  # ...but key is attacker's
        kid_prefix="attacker",
    )[0]

    # VMs list: legitimate victim VM + unbound attacker VM with unique kid
    pred_vms = list(legacy_predecessor["vms"]) + [att_vm]
    att_kid = att_vm["id"].rsplit("#", 1)[-1]

    s = SuccessionStatement(
        predecessor_did=legacy_predecessor["did"],
        successor_did=hybrid_successor["did"],
        effective_from="2026-05-04T18:00:00+00:00",
        reason="extra unbound VM decoy",
        predecessor_verification_methods=pred_vms,
        successor_verification_methods=hybrid_successor["vms"],
    )
    s = sign_predecessor(s, [(att_kp, att_kid)])
    s = sign_successor(s, [
        (hybrid_successor["hybrid"].classical, hybrid_successor["classical_kid"]),
        (hybrid_successor["hybrid"].pq, hybrid_successor["pq_kid"]),
    ])
    s = finalize(s)

    result = verify_succession(s, did_web_resolver=_self_attesting_resolver(s))
    assert not result.ok
    assert not result.predecessor_did_bound
    # Reason should mention the address mismatch, not just generic failure
    assert (
        "decoy" in result.reason
        or "claims" in result.reason
        or "derives address" in result.reason
    )


def test_did_binding_handles_non_mapping_vm(legacy_predecessor):
    """Codex P2 round 4: a malformed VM list (e.g. archived data with a
    non-dict entry) used to crash with AttributeError on vm.get(). Now
    it should fail-closed cleanly."""
    from kestrel_sovereign.identity.succession import verify_did_binding

    bogus_vms = list(legacy_predecessor["vms"]) + ["not-a-dict"]
    ok, reason = verify_did_binding(legacy_predecessor["did"], bogus_vms)
    # Doesn't crash; returns failure.
    assert not ok


def test_unfinalized_statement_rejected(
    base_statement, legacy_predecessor, hybrid_successor,
):
    """Codex P2: an empty statement_id must NOT be silently treated as
    consistent. Audit logs and chain walkers index by id; an
    unaddressable statement is not safe to accept.
    """
    s = sign_predecessor(
        base_statement, [(legacy_predecessor["kp"], legacy_predecessor["kid"])],
    )
    s = sign_successor(s, [
        (hybrid_successor["hybrid"].classical, hybrid_successor["classical_kid"]),
        (hybrid_successor["hybrid"].pq, hybrid_successor["pq_kid"]),
    ])
    # NOTE: NOT calling finalize() — statement_id stays empty
    assert not s.statement_id

    result = verify_succession(s, did_web_resolver=_self_attesting_resolver(s))
    assert not result.ok
    assert not result.statement_id_consistent
    assert "statement_id is empty" in result.reason
