"""Phase 0 chunk C — workflow signing/verification tests.

The signing module is the third leg of Phase 0: dataclasses define the
shape, storage persists the bytes, and these helpers prove an author
DID committed to a specific canonical payload. Tests cover both the
workflow-spec path (signed at workflow_define time) and the
stage-transition path (signed each time the runner advances a stage)
plus the negative cases the verifier must catch.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.signals import SignalMode

from kestrel_sovereign.features.workflows import (
    GateOutcome,
    Stage,
    StageLink,
    WorkflowSpec,
)
from kestrel_sovereign.features.workflows import WorkflowDefinitionError
from kestrel_sovereign.features.workflows.signing import (
    canonical_transition_payload,
    sign_stage_transition,
    sign_workflow_spec,
    verify_stage_transition,
    verify_workflow_spec,
)
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    Keypair,
    get_suite,
)


def _agent_identity(did: str = "did:web:k.example") -> AgentIdentity:
    """Construct a minimally-functional AgentIdentity for signing tests
    by generating a fresh secp256k1 keypair and wrapping it in the
    legacy slot. The ``legacy_did_document`` is empty because this
    helper short-circuits the ceremony machinery — callers that need a
    real DID document use the higher-level ``load_agent_identity``."""
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    kp = suite.generate_keypair()
    return AgentIdentity(
        legacy_did=did,
        legacy_keypair=kp,
        legacy_did_document={},
    )


def _hybrid_identity() -> AgentIdentity:
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    legacy_kp = suite.generate_keypair()
    hybrid = generate_hybrid_keypair()
    new_did = "did:web:k.example:hybrid"
    vms = build_verification_methods(new_did, hybrid.public_keys())
    return AgentIdentity(
        legacy_did="did:pkh:eip155:1:0xabc",
        legacy_keypair=legacy_kp,
        legacy_did_document={},
        hybrid_keypair=hybrid,
        new_did=new_did,
        new_verification_methods=vms,
    )


def _resolver_for(*identities: AgentIdentity):
    """Return a public-key resolver that knows the supplied identities.

    The resolver maps DID → uncompressed X9.62 bytes (the form
    ``CryptoSuite.deserialize_public_key`` expects for secp256k1).
    Phase 0 helpers sign with the legacy ECDSA key and identify as
    the LEGACY DID (not signing_did, which can be the new did:web on
    a hybrid agent), so the resolver maps legacy_did → legacy pubkey.
    """
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    table = {
        ai.legacy_did: suite.serialize_public_key(ai.legacy_keypair.public_key)
        for ai in identities
    }

    def resolve(did: str) -> bytes:
        return table[did]

    return resolve


def _action_stage(**overrides):
    base = dict(
        name="lint",
        signal_source="ci.lint",
        signal_mode=SignalMode.ACTION,
        read_only=True,
    )
    base.update(overrides)
    return Stage(**base)


def _spec(**overrides):
    base = dict(
        name="release",
        version=1,
        stages=[_action_stage(name="lint")],
    )
    base.update(overrides)
    return WorkflowSpec(**base)


# ---------------------------------------------------------------------------
# WorkflowSpec sign/verify
# ---------------------------------------------------------------------------


def test_sign_then_verify_round_trip():
    ai = _agent_identity()
    signed = sign_workflow_spec(_spec(), ai)
    # Phase 0: sign with legacy ECDSA key; author_did is the legacy DID
    # (see signing.py rationale — pre-ceremony agents have legacy_did
    # == signing_did, hybrid agents see the legacy DID here too).
    assert signed.author_did == ai.legacy_did
    assert signed.spec_hash and len(signed.spec_hash) == 64
    assert signed.author_sig
    assert verify_workflow_spec(signed, _resolver_for(ai)) is True


def test_verify_fails_when_payload_tampered():
    """Recompute_spec_hash detects any change to the canonical payload —
    we exercise that at the dataclass level by re-signing a spec
    whose stages list is different and re-using the original
    signature."""
    ai = _agent_identity()
    original = sign_workflow_spec(_spec(), ai)
    # Forge a new spec with the original signature; recomputed hash
    # will not match.
    forged = WorkflowSpec(
        name="release",
        version=1,
        stages=[_action_stage(name="lint"), _action_stage(name="extra")],
        author_did=original.author_did,
        spec_hash=original.spec_hash,
        author_sig=original.author_sig,
    )
    assert verify_workflow_spec(forged, _resolver_for(ai)) is False


def test_verify_fails_under_wrong_public_key():
    ai = _agent_identity()
    other = _agent_identity(did="did:web:other.example")
    signed = sign_workflow_spec(_spec(), ai)
    # Resolver maps the *signing* DID to the *other* agent's pubkey.
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    other_pub_bytes = suite.serialize_public_key(other.legacy_keypair.public_key)

    def resolver(did: str) -> bytes:
        return other_pub_bytes

    assert verify_workflow_spec(signed, resolver) is False


def test_verify_fails_when_unsigned():
    spec = _spec()  # no signature at all
    assert verify_workflow_spec(spec, lambda did: b"") is False


def test_verify_fails_under_unresolvable_did():
    ai = _agent_identity()
    signed = sign_workflow_spec(_spec(), ai)

    def resolver(did: str) -> bytes:
        raise KeyError(did)

    assert verify_workflow_spec(signed, resolver) is False


def test_verify_fails_with_malformed_signature_hex():
    ai = _agent_identity()
    signed = sign_workflow_spec(_spec(), ai)
    bad = WorkflowSpec(
        **{
            **signed.__dict__,
            "author_sig": "not-hex-zz",
        }
    )
    assert verify_workflow_spec(bad, _resolver_for(ai)) is False


def test_sign_rejects_mismatched_author_did():
    """Round-2 P2: a pre-set author_did that doesn't match the agent's
    signing_did would produce an unverifiable spec (signature is over
    the agent's key, but verify resolves the foreign author_did).
    Phase 0 helper rejects the mismatch. Delegated-signing flows are
    a Phase 1+ multisig surface."""
    ai = _agent_identity(did="did:web:k.example")
    spec = _spec(author_did="did:web:author.example")
    with pytest.raises(WorkflowDefinitionError):
        sign_workflow_spec(spec, ai)


def test_sign_uses_legacy_did_for_hybrid_agent():
    """Round-2 P2: post-ceremony hybrid agents have ``signing_did``
    pointing at the new did:web (whose VMs publish ed25519 + ml-dsa-65).
    Phase 0 signs with the LEGACY ECDSA key, so the spec must list
    ``legacy_did`` as author so verifiers resolve the right key."""
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    legacy_kp = suite.generate_keypair()
    # Construct a synthetic hybrid AgentIdentity by setting new_did.
    # We don't need real hybrid keys — sign_workflow_spec should
    # pick legacy_did regardless of the hybrid state.
    ai = AgentIdentity(
        legacy_did="did:pkh:eip155:1:0xabc",
        legacy_keypair=legacy_kp,
        legacy_did_document={},
        # hybrid_keypair left None to keep this fixture simple; the
        # invariant we care about (author_did == legacy_did) doesn't
        # depend on the hybrid keypair shape.
        new_did="did:web:k.example",
    )
    # Pre-ceremony assertion: signing_did != legacy_did when new_did
    # is set without hybrid_keypair (is_hybrid=False because the
    # property checks hybrid_keypair).
    assert ai.legacy_did == "did:pkh:eip155:1:0xabc"
    signed = sign_workflow_spec(_spec(), ai)
    # MUST be the legacy DID, not new_did, because that's the DID
    # whose VMs cover the ECDSA secp256k1 alg we just used to sign.
    assert signed.author_did == ai.legacy_did
    assert verify_workflow_spec(signed, _resolver_for(ai)) is True


def test_sign_accepts_matching_pre_set_author_did():
    ai = _agent_identity(did="did:web:k.example")
    spec = _spec(author_did="did:web:k.example")
    signed = sign_workflow_spec(spec, ai)
    assert signed.author_did == "did:web:k.example"
    assert verify_workflow_spec(signed, _resolver_for(ai)) is True


def test_hybrid_sign_workflow_spec_uses_new_did_and_verifies():
    ai = _hybrid_identity()
    signed = sign_workflow_spec(_spec(), ai, use_hybrid=True)

    assert signed.author_did == ai.signing_did
    assert signed.author_sig.startswith("hybrid:")
    assert verify_workflow_spec(
        signed,
        _resolver_for(ai),
        verification_methods_resolver=lambda did: ai.new_verification_methods,
    ) is True


def test_hybrid_workflow_spec_requires_verification_methods_resolver():
    ai = _hybrid_identity()
    signed = sign_workflow_spec(_spec(), ai, use_hybrid=True)

    assert verify_workflow_spec(signed, _resolver_for(ai)) is False


# ---------------------------------------------------------------------------
# Stage transition signing
# ---------------------------------------------------------------------------


def test_canonical_transition_payload_stable():
    a = canonical_transition_payload(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id="sig-1",
        gate_outcome="pass",
    )
    b = canonical_transition_payload(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id="sig-1",
        gate_outcome="pass",
    )
    assert a == b
    assert b"workflow.stage_transition.v1" in a


def test_canonical_transition_distinguishes_none_from_string():
    """A signal_id of None (pre-dispatch) must NOT collide with a
    signal_id of any string value."""
    none_form = canonical_transition_payload(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id=None,
        gate_outcome=None,
    )
    str_form = canonical_transition_payload(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id="some_id",
        gate_outcome="pass",
    )
    assert none_form != str_form


def test_canonical_transition_no_sentinel_collision():
    """Codex chunk-C round-1 P2: the prior 'none' literal collided with
    a persisted ``signal_id='none'``. JSON serialization makes
    ``null`` (None) and ``"none"`` (string) byte-distinct."""
    none_payload = canonical_transition_payload(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id=None,
        gate_outcome=None,
    )
    string_none_payload = canonical_transition_payload(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id="none",
        gate_outcome="none",
    )
    assert none_payload != string_none_payload


def test_verify_stage_transition_does_not_accept_sentinel_replay():
    """End-to-end: a signature produced for ``signal_id=None`` must NOT
    verify against a StageLink whose ``signal_id='none'``."""
    ai = _agent_identity()
    actor_did, sig_none = sign_stage_transition(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id=None,
        gate_outcome=None,
        agent_identity=ai,
    )
    link = StageLink(
        link_id="l-1",
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        idempotency_key="0" * 64,
        actor_did=actor_did,
        actor_sig=sig_none,
        signal_id="none",  # literal string — MUST NOT match the None sig
    )
    assert verify_stage_transition(link, _resolver_for(ai)) is False


def test_sign_then_verify_stage_transition():
    ai = _agent_identity()
    actor_did, sig_hex = sign_stage_transition(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id="signal-1",
        gate_outcome="pass",
        agent_identity=ai,
    )
    assert actor_did == ai.legacy_did
    link = StageLink(
        link_id="l-1",
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        idempotency_key="0" * 64,
        actor_did=actor_did,
        actor_sig=sig_hex,
        signal_id="signal-1",
        gate_outcome=GateOutcome.PASS,
    )
    assert verify_stage_transition(link, _resolver_for(ai)) is True


def test_hybrid_sign_then_verify_stage_transition():
    ai = _hybrid_identity()
    actor_did, sig = sign_stage_transition(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id="signal-1",
        gate_outcome="pass",
        agent_identity=ai,
        use_hybrid=True,
    )
    assert actor_did == ai.signing_did
    assert sig.startswith("hybrid:")
    link = StageLink(
        link_id="l-1",
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        idempotency_key="0" * 64,
        actor_did=actor_did,
        actor_sig=sig,
        signal_id="signal-1",
        gate_outcome=GateOutcome.PASS,
    )
    assert verify_stage_transition(
        link,
        _resolver_for(ai),
        verification_methods_resolver=lambda did: ai.new_verification_methods,
    ) is True


def test_verify_stage_transition_fails_under_tamper():
    ai = _agent_identity()
    actor_did, sig_hex = sign_stage_transition(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id="signal-1",
        gate_outcome="pass",
        agent_identity=ai,
    )
    # Sign for attempt_number=1 but persist as attempt_number=2 →
    # canonical_transition_payload differs → verify must fail.
    tampered = StageLink(
        link_id="l-1",
        run_id="r-1",
        stage_name="lint",
        attempt_number=2,
        idempotency_key="0" * 64,
        actor_did=actor_did,
        actor_sig=sig_hex,
        signal_id="signal-1",
        gate_outcome=GateOutcome.PASS,
    )
    assert verify_stage_transition(tampered, _resolver_for(ai)) is False


def test_verify_stage_transition_fails_with_bad_hex():
    ai = _agent_identity()
    link = StageLink(
        link_id="l-1",
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        idempotency_key="0" * 64,
        actor_did=ai.signing_did,
        actor_sig="garbage-not-hex",
    )
    assert verify_stage_transition(link, _resolver_for(ai)) is False


def test_verify_stage_transition_fails_when_resolver_unknown():
    ai = _agent_identity()
    actor_did, sig_hex = sign_stage_transition(
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        signal_id=None,
        gate_outcome=None,
        agent_identity=ai,
    )
    link = StageLink(
        link_id="l-1",
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        idempotency_key="0" * 64,
        actor_did=actor_did,
        actor_sig=sig_hex,
    )

    def resolver(did: str) -> bytes:
        raise KeyError(did)

    assert verify_stage_transition(link, resolver) is False
