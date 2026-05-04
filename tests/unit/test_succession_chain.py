"""
Succession-chain walker tests — Wave 3 sub-PR 3 (#918).

Covers:
- build_chain structural validation:
  * empty chain accepted
  * single statement accepted
  * linkage check (predecessor[i+1] = successor[i])
  * temporal monotonicity (effective_from strictly increasing)
  * self-succession rejected
  * timezone-naive timestamps rejected
- resolve_active_identity:
  * empty chain → root active, post_cutoff=False
  * timestamp before first effective_from → root active
  * timestamp at/after effective_from → successor active, post_cutoff=True
  * multi-link chain: latest applicable successor wins
  * future succession does NOT push earlier artifact post-cutoff
- verify_chain_signatures:
  * all valid → ok=True
  * one tampered → ok=False, per-statement diagnostics
- verify_artifact_against_chain:
  * pre-cutoff classical-only artifact under LEGACY_ALLOWED → ok
  * post-cutoff classical-only artifact → fail (post_cutoff_classical
    _allowed=False kicks in)
  * post-cutoff hybrid artifact → ok
  * tampered chain statement → fail (chain signatures fail before
    artifact even gets evaluated)
  * artifact signed by wrong identity (key not in active VMs) → fail
"""

from __future__ import annotations

import json

import pytest

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.hybrid_keypair import (
    generate_hybrid_keypair,
    sign_hybrid,
)
from kestrel_sovereign.identity.succession import (
    SuccessionStatement,
    finalize,
    sign_predecessor,
    sign_successor,
)
from kestrel_sovereign.identity.succession_chain import (
    SuccessionChainError,
    build_chain,
    resolve_active_identity,
    verify_artifact_against_chain,
    verify_chain_signatures,
)
from kestrel_sovereign.security.crypto_suite import (
    Secp256k1Suite,
    get_suite,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def legacy_root():
    """Legacy ECDSA-only root identity (Kestrel #1 style)."""
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    did = "did:pkh:eip155:1:0xKESTRELONE"
    vms = build_verification_methods(did, [(secp, kp.public_key)])
    return {
        "did": did,
        "kp": kp,
        "kid": vms[0]["id"].rsplit("#", 1)[-1],
        "vms": vms,
    }


def _hybrid_identity(slug: str):
    hybrid = generate_hybrid_keypair()
    did = f"did:web:example.com:{slug}"
    vms = build_verification_methods(did, hybrid.public_keys())
    return {
        "did": did,
        "hybrid": hybrid,
        "classical_kid": vms[0]["id"].rsplit("#", 1)[-1],
        "pq_kid": vms[1]["id"].rsplit("#", 1)[-1],
        "vms": vms,
    }


@pytest.fixture(scope="module")
def successor_v1():
    return _hybrid_identity("kestrel-v2")


@pytest.fixture(scope="module")
def successor_v2():
    return _hybrid_identity("kestrel-v3")


@pytest.fixture(scope="module")
def first_succession(legacy_root, successor_v1):
    """Legacy → hybrid (canonical Wave 3 ceremony)."""
    s = SuccessionStatement(
        predecessor_did=legacy_root["did"],
        successor_did=successor_v1["did"],
        effective_from="2026-05-04T18:00:00+00:00",
        reason="Wave 3 PQ-hardening migration #1",
        predecessor_verification_methods=legacy_root["vms"],
        successor_verification_methods=successor_v1["vms"],
    )
    s = sign_predecessor(s, [(legacy_root["kp"], legacy_root["kid"])])
    s = sign_successor(s, [
        (successor_v1["hybrid"].classical, successor_v1["classical_kid"]),
        (successor_v1["hybrid"].pq, successor_v1["pq_kid"]),
    ])
    return finalize(s)


@pytest.fixture(scope="module")
def second_succession(successor_v1, successor_v2):
    """Hybrid → hybrid (post-Wave-3 future rotation)."""
    s = SuccessionStatement(
        predecessor_did=successor_v1["did"],
        successor_did=successor_v2["did"],
        effective_from="2027-01-01T00:00:00+00:00",
        reason="rotation #2",
        predecessor_verification_methods=successor_v1["vms"],
        successor_verification_methods=successor_v2["vms"],
    )
    s = sign_predecessor(s, [
        (successor_v1["hybrid"].classical, successor_v1["classical_kid"]),
        (successor_v1["hybrid"].pq, successor_v1["pq_kid"]),
    ])
    s = sign_successor(s, [
        (successor_v2["hybrid"].classical, successor_v2["classical_kid"]),
        (successor_v2["hybrid"].pq, successor_v2["pq_kid"]),
    ])
    return finalize(s)


# ---------------------------------------------------------------------------
# build_chain — structural validation
# ---------------------------------------------------------------------------

def test_build_chain_empty_is_valid():
    chain = build_chain([])
    assert chain.is_empty()


def test_build_chain_single_statement_valid(first_succession):
    chain = build_chain([first_succession])
    assert len(chain) == 1


def test_build_chain_two_links_valid(first_succession, second_succession):
    chain = build_chain([first_succession, second_succession])
    assert len(chain) == 2


def test_build_chain_rejects_broken_linkage(first_succession, successor_v2, legacy_root):
    """A statement whose predecessor_did doesn't match the previous
    successor_did represents either a forked chain or a misordered
    list. Either way: refuse."""
    rogue = SuccessionStatement(
        predecessor_did=legacy_root["did"],  # wrong: should be successor_v1's DID
        successor_did=successor_v2["did"],
        effective_from="2027-01-01T00:00:00+00:00",
        reason="forked",
        predecessor_verification_methods=legacy_root["vms"],
        successor_verification_methods=successor_v2["vms"],
    )
    with pytest.raises(SuccessionChainError, match="chain link broken"):
        build_chain([first_succession, rogue])


def test_build_chain_rejects_temporal_regression(first_succession, successor_v2, successor_v1):
    """Two successions that go backward in time would let an attacker
    unwind a rotation. Refuse."""
    backward = SuccessionStatement(
        predecessor_did=successor_v1["did"],
        successor_did=successor_v2["did"],
        # Earlier than first_succession's effective_from
        effective_from="2025-01-01T00:00:00+00:00",
        reason="time travel",
        predecessor_verification_methods=successor_v1["vms"],
        successor_verification_methods=successor_v2["vms"],
    )
    with pytest.raises(SuccessionChainError, match="temporal monotonicity"):
        build_chain([first_succession, backward])


def test_build_chain_rejects_self_succession(legacy_root):
    self_loop = SuccessionStatement(
        predecessor_did=legacy_root["did"],
        successor_did=legacy_root["did"],
        effective_from="2026-05-04T18:00:00+00:00",
        reason="loop",
        predecessor_verification_methods=legacy_root["vms"],
        successor_verification_methods=legacy_root["vms"],
    )
    with pytest.raises(SuccessionChainError, match="self-succession"):
        build_chain([self_loop])


def test_build_chain_rejects_naive_timestamp(legacy_root, successor_v1):
    """Cross-succession comparisons require unambiguous tz info."""
    s1 = SuccessionStatement(
        predecessor_did=legacy_root["did"],
        successor_did=successor_v1["did"],
        effective_from="2026-05-04T18:00:00",  # NO TZ
        reason="naive",
        predecessor_verification_methods=legacy_root["vms"],
        successor_verification_methods=successor_v1["vms"],
    )
    s2 = SuccessionStatement(
        predecessor_did=successor_v1["did"],
        successor_did="did:web:example.com:other",
        effective_from="2027-01-01T00:00:00+00:00",
        reason="ok",
        predecessor_verification_methods=successor_v1["vms"],
        successor_verification_methods=successor_v1["vms"],  # placeholder
    )
    with pytest.raises(SuccessionChainError, match="timezone-naive"):
        build_chain([s1, s2])


# ---------------------------------------------------------------------------
# resolve_active_identity
# ---------------------------------------------------------------------------

def test_resolve_empty_chain_root_is_active(legacy_root):
    chain = build_chain([])
    active = resolve_active_identity(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2026-01-01T00:00:00+00:00",
    )
    assert active.did == legacy_root["did"]
    assert active.is_root
    assert not active.post_cutoff
    assert active.succession_index is None


def test_resolve_pre_first_effective_from_root_is_active(legacy_root, first_succession):
    """An artifact dated BEFORE the first succession is in the
    root's era."""
    chain = build_chain([first_succession])
    active = resolve_active_identity(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2025-12-31T23:59:59+00:00",  # well before
    )
    assert active.did == legacy_root["did"]
    assert active.is_root
    assert not active.post_cutoff


def test_resolve_at_or_after_first_effective_from(legacy_root, successor_v1, first_succession):
    chain = build_chain([first_succession])
    active = resolve_active_identity(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2026-06-01T00:00:00+00:00",  # after
    )
    assert active.did == successor_v1["did"]
    assert not active.is_root
    assert active.post_cutoff
    assert active.succession_index == 0


def test_resolve_multi_link_picks_latest_applicable(
    legacy_root, successor_v1, successor_v2, first_succession, second_succession,
):
    """Between effective_from of first and second succession: v1 active."""
    chain = build_chain([first_succession, second_succession])
    active = resolve_active_identity(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2026-12-15T00:00:00+00:00",  # between v1 and v2
    )
    assert active.did == successor_v1["did"]
    assert active.succession_index == 0


def test_resolve_multi_link_after_second_uses_v2(
    legacy_root, successor_v2, first_succession, second_succession,
):
    chain = build_chain([first_succession, second_succession])
    active = resolve_active_identity(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2027-06-01T00:00:00+00:00",
    )
    assert active.did == successor_v2["did"]
    assert active.succession_index == 1


def test_resolve_future_succession_does_not_retroactively_trigger_cutoff(
    legacy_root, first_succession,
):
    """An artifact dated BEFORE the only succession's effective_from
    must NOT have post_cutoff=True. Otherwise a back-dated artifact
    at the moment of rotation would be retroactively rejected for
    being classical-only."""
    chain = build_chain([first_succession])
    active = resolve_active_identity(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2026-04-01T00:00:00+00:00",  # before succession
    )
    assert not active.post_cutoff
    assert active.is_root


# ---------------------------------------------------------------------------
# verify_chain_signatures
# ---------------------------------------------------------------------------

def test_verify_chain_signatures_all_valid(first_succession, second_succession):
    chain = build_chain([first_succession, second_succession])
    result = verify_chain_signatures(chain)
    assert result.ok, result.reason
    assert len(result.per_statement) == 2


def test_verify_chain_signatures_tampered_one_fails(first_succession, second_succession):
    """Tamper with the second statement after signing — chain check must
    fail and identify the bad statement."""
    from dataclasses import replace
    bad = replace(second_succession, reason="MUTATED")
    chain = build_chain([first_succession, bad])
    result = verify_chain_signatures(chain)
    assert not result.ok
    # Failure reason should mention statement[1]
    assert "statement[1]" in result.reason


# ---------------------------------------------------------------------------
# verify_artifact_against_chain
# ---------------------------------------------------------------------------

def test_artifact_pre_cutoff_classical_ok(legacy_root, first_succession):
    """An artifact dated BEFORE the succession, signed by the legacy
    ECDSA key, is acceptable under LEGACY_ALLOWED — no cutoff has fired."""
    chain = build_chain([first_succession])
    payload = b"some pre-rotation document"

    secp = get_suite(legacy_root["kp"].suite_id)
    sig = secp.sign(payload, legacy_root["kp"].private_key)
    artifact_signatures = [{
        "alg": secp.alg_id,
        "kid": legacy_root["kid"],
        "sig": sig.hex(),
    }]

    result = verify_artifact_against_chain(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2026-04-01T00:00:00+00:00",  # before cutoff
        artifact_payload=payload,
        artifact_signatures=artifact_signatures,
        policy=VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok, result.reason
    assert result.active_identity.is_root
    assert not result.active_identity.post_cutoff


def test_artifact_post_cutoff_classical_only_fails(
    successor_v1, legacy_root, first_succession,
):
    """An artifact dated AFTER the succession, signed by ONLY the
    successor's classical half (no PQ), must fail the policy because
    post_cutoff_classical_allowed=False kicks in."""
    chain = build_chain([first_succession])
    payload = b"a post-rotation document"

    classical_kp = successor_v1["hybrid"].classical
    classical_suite = get_suite(classical_kp.suite_id)
    sig = classical_suite.sign(payload, classical_kp.private_key)
    artifact_signatures = [{
        "alg": classical_suite.alg_id,
        "kid": successor_v1["classical_kid"],
        "sig": sig.hex(),
    }]

    result = verify_artifact_against_chain(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2026-06-01T00:00:00+00:00",  # after cutoff
        artifact_payload=payload,
        artifact_signatures=artifact_signatures,
        policy=VerifyPolicy.LEGACY_ALLOWED,  # most permissive policy
    )
    assert not result.ok
    assert result.active_identity.post_cutoff
    # The cutoff itself is what caused the failure
    assert "post-cutoff" in result.policy_result.reason


def test_artifact_post_cutoff_hybrid_passes(
    successor_v1, legacy_root, first_succession,
):
    """Same post-cutoff timestamp, but with BOTH classical and PQ
    signatures — passes. This is the design intent: hybrid signing
    is the safe path forward."""
    chain = build_chain([first_succession])
    payload = b"a post-rotation document"

    artifact_signatures = sign_hybrid(
        payload, successor_v1["hybrid"],
        classical_kid=successor_v1["classical_kid"],
        pq_kid=successor_v1["pq_kid"],
    )

    result = verify_artifact_against_chain(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2026-06-01T00:00:00+00:00",
        artifact_payload=payload,
        artifact_signatures=artifact_signatures,
        policy=VerifyPolicy.HYBRID_REQUIRED,
    )
    assert result.ok, result.reason


def test_artifact_signed_by_wrong_identity_fails(
    successor_v1, successor_v2, legacy_root, first_succession,
):
    """If the artifact's signatures use successor_v2's keys but the
    chain only contains successor_v1, the kids won't resolve to a key
    in v1's verification methods → no signatures verify → policy fails."""
    chain = build_chain([first_succession])  # only knows about v1
    payload = b"signed by wrong identity"

    # Sign with v2's keys
    artifact_signatures = sign_hybrid(
        payload, successor_v2["hybrid"],
        classical_kid=successor_v2["classical_kid"],
        pq_kid=successor_v2["pq_kid"],
    )

    result = verify_artifact_against_chain(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2026-06-01T00:00:00+00:00",
        artifact_payload=payload,
        artifact_signatures=artifact_signatures,
    )
    assert not result.ok


def test_artifact_chain_signatures_failed_propagates(
    legacy_root, first_succession, second_succession,
):
    """If the chain itself has a tampered statement, the artifact
    verdict must fail even if the artifact's own signatures would
    have passed in isolation."""
    from dataclasses import replace
    bad = replace(second_succession, reason="TAMPERED CHAIN")
    chain = build_chain([first_succession, bad])
    # Don't bother actually crafting an artifact — chain failure
    # alone should kill the verdict.
    result = verify_artifact_against_chain(
        root_did=legacy_root["did"],
        root_verification_methods=legacy_root["vms"],
        chain=chain,
        artifact_timestamp="2027-06-01T00:00:00+00:00",
        artifact_payload=b"x",
        artifact_signatures=[],
    )
    assert not result.ok
    assert "chain" in result.reason
