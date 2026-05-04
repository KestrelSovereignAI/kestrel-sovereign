"""
verify_policy tests — Wave 1 sub-PR 4 (#916).

Covers the three policy modes, per-context defaults, the
post-succession-cutoff hook (the second feedback note's protection
against post-quantum forgery of historical statements), and the
unknown-alg fail-loud behavior.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    ALG_ED25519,
    ALG_ML_DSA_65,
    CryptoSuite,
    register_suite,
    _REGISTRY,
)
from kestrel_sovereign.security.verify_policy import (
    Context,
    PolicyResult,
    VerifyPolicy,
    default_policy_for,
    evaluate_signatures,
)


# ---------------------------------------------------------------------------
# Stub suites for testing — Wave 2 ships real Ed25519/MLDSA suites; this
# file only needs ``is_post_quantum`` classification, not actual signing.
# ---------------------------------------------------------------------------

class _StubEd25519(CryptoSuite):
    alg_id: ClassVar[str] = ALG_ED25519
    is_post_quantum: ClassVar[bool] = False
    public_key_multicodec: ClassVar[bytes] = b"\xed\x01"

    def generate_keypair(self): raise NotImplementedError
    def sign(self, data, private_key): raise NotImplementedError
    def verify(self, data, signature, public_key): return False
    def serialize_public_key(self, public_key): return b""
    def deserialize_public_key(self, raw): raise NotImplementedError


class _StubMLDSA65(CryptoSuite):
    alg_id: ClassVar[str] = ALG_ML_DSA_65
    is_post_quantum: ClassVar[bool] = True
    public_key_multicodec: ClassVar[bytes] = b"\x12\x07"  # placeholder

    def generate_keypair(self): raise NotImplementedError
    def sign(self, data, private_key): raise NotImplementedError
    def verify(self, data, signature, public_key): return False
    def serialize_public_key(self, public_key): return b""
    def deserialize_public_key(self, raw): raise NotImplementedError


@pytest.fixture(autouse=True)
def _register_stub_suites():
    """Register stub Ed25519 and ML-DSA-65 suites for the duration of the
    test, then remove them so we don't pollute other tests."""
    ed = _StubEd25519()
    mldsa = _StubMLDSA65()
    # Register only if not already present (idempotent across test reruns)
    if ALG_ED25519 not in _REGISTRY:
        register_suite(ed)
    if ALG_ML_DSA_65 not in _REGISTRY:
        register_suite(mldsa)
    yield
    # Don't unregister — other tests may also reach for these. Stubs are
    # safe to leave; if Wave 2 ships real suites they'll fail
    # register_suite's "competing-instance" guard, which is the desired
    # signal to update these stubs.


# ---------------------------------------------------------------------------
# Per-context defaults
# ---------------------------------------------------------------------------

def test_archival_import_defaults_to_legacy_allowed():
    assert default_policy_for(Context.ARCHIVAL_IMPORT) == VerifyPolicy.LEGACY_ALLOWED


def test_live_identity_defaults_to_legacy_allowed_pre_wave_2():
    """Pre-Wave-2 the live-identity default is permissive; flips to
    HYBRID_REQUIRED with the v0.Z release that bumps new-agent default
    to did:web. Test pins the current state — update the assertion (and
    the schedule comment) when the release flip happens."""
    assert default_policy_for(Context.LIVE_IDENTITY_ASSERTION) == VerifyPolicy.LEGACY_ALLOWED


def test_new_identity_issuance_defaults_to_hybrid_required():
    assert default_policy_for(Context.NEW_IDENTITY_ISSUANCE) == VerifyPolicy.HYBRID_REQUIRED


def test_constitution_checkpoint_defaults_to_pq_required():
    assert default_policy_for(Context.CONSTITUTION_CHECKPOINT) == VerifyPolicy.PQ_REQUIRED


def test_every_context_has_a_default():
    for ctx in Context:
        # Should not raise
        result = default_policy_for(ctx)
        assert isinstance(result, VerifyPolicy)


# ---------------------------------------------------------------------------
# LEGACY_ALLOWED
# ---------------------------------------------------------------------------

def _sig(alg: str, kid: str = "kid", sig: str = "s") -> dict:
    return {"alg": alg, "kid": kid, "sig": sig}


def test_legacy_allowed_accepts_classical_only():
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256)],
        VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok is True


def test_legacy_allowed_accepts_pq_only():
    result = evaluate_signatures(
        [_sig(ALG_ML_DSA_65)],
        VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok is True


def test_legacy_allowed_accepts_hybrid():
    result = evaluate_signatures(
        [_sig(ALG_ED25519), _sig(ALG_ML_DSA_65)],
        VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok is True


def test_legacy_allowed_rejects_empty():
    result = evaluate_signatures([], VerifyPolicy.LEGACY_ALLOWED)
    assert result.ok is False
    assert "no signatures" in result.reason


# ---------------------------------------------------------------------------
# HYBRID_REQUIRED
# ---------------------------------------------------------------------------

def test_hybrid_required_accepts_classical_plus_pq():
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256), _sig(ALG_ML_DSA_65)],
        VerifyPolicy.HYBRID_REQUIRED,
    )
    assert result.ok is True


def test_hybrid_required_rejects_classical_only():
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256), _sig(ALG_ED25519)],
        VerifyPolicy.HYBRID_REQUIRED,
    )
    assert result.ok is False
    assert "post-quantum" in result.reason


def test_hybrid_required_rejects_pq_only():
    result = evaluate_signatures(
        [_sig(ALG_ML_DSA_65)],
        VerifyPolicy.HYBRID_REQUIRED,
    )
    assert result.ok is False
    assert "classical" in result.reason


# ---------------------------------------------------------------------------
# PQ_REQUIRED
# ---------------------------------------------------------------------------

def test_pq_required_accepts_pq_only():
    result = evaluate_signatures(
        [_sig(ALG_ML_DSA_65)],
        VerifyPolicy.PQ_REQUIRED,
    )
    assert result.ok is True


def test_pq_required_accepts_hybrid():
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256), _sig(ALG_ML_DSA_65)],
        VerifyPolicy.PQ_REQUIRED,
    )
    assert result.ok is True


def test_pq_required_rejects_classical_only():
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256)],
        VerifyPolicy.PQ_REQUIRED,
    )
    assert result.ok is False
    assert "post-quantum" in result.reason.lower()


# ---------------------------------------------------------------------------
# Post-succession-cutoff hook (the second feedback-note protection)
# ---------------------------------------------------------------------------

def test_cutoff_rejects_classical_only_under_legacy_allowed():
    """The feedback-note attack: a post-quantum forgery of a historical
    statement signed only by a Shor-broken classical key. Under
    LEGACY_ALLOWED the policy alone would accept it — but if the chain
    walker has determined the artifact is post-cutoff, it passes
    ``post_cutoff_classical_allowed=False`` and the policy must reject
    even though LEGACY_ALLOWED is otherwise permissive."""
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256)],
        VerifyPolicy.LEGACY_ALLOWED,
        post_cutoff_classical_allowed=False,
    )
    assert result.ok is False
    assert "post-cutoff" in result.reason


def test_cutoff_accepts_pq_post_cutoff():
    """A post-cutoff artifact with a PQ signature is fine — the cutoff
    only forbids classical-only chain segments."""
    result = evaluate_signatures(
        [_sig(ALG_ML_DSA_65)],
        VerifyPolicy.LEGACY_ALLOWED,
        post_cutoff_classical_allowed=False,
    )
    assert result.ok is True


def test_cutoff_accepts_hybrid_post_cutoff():
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256), _sig(ALG_ML_DSA_65)],
        VerifyPolicy.HYBRID_REQUIRED,
        post_cutoff_classical_allowed=False,
    )
    assert result.ok is True


def test_cutoff_default_is_permissive():
    """Backwards compat: callers that don't know about the cutoff still
    see Wave-1-and-earlier behavior (no cutoff applied)."""
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256)],
        VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok is True


# ---------------------------------------------------------------------------
# Unknown alg / malformed input
# ---------------------------------------------------------------------------

def test_unknown_alg_rejected_loudly():
    """Unknown alg_ids must NOT be silently classified as classical or
    PQ — fail loud rather than risk a future suite shipping under an
    unrecognized id and being silently mis-classified."""
    result = evaluate_signatures(
        [_sig("ml-dsa-87-future-build")],
        VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok is False
    assert "unknown alg_id" in result.reason


def test_unknown_alg_rejected_even_alongside_known():
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256), _sig("future-suite")],
        VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok is False


def test_missing_alg_field_rejected():
    result = evaluate_signatures(
        [{"kid": "kid", "sig": "s"}],
        VerifyPolicy.LEGACY_ALLOWED,
    )
    assert result.ok is False
    assert "missing 'alg'" in result.reason


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

def test_result_records_all_alg_ids_seen():
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256), _sig(ALG_ML_DSA_65)],
        VerifyPolicy.HYBRID_REQUIRED,
    )
    assert result.alg_ids_seen == frozenset({
        ALG_ECDSA_SECP256K1_SHA256, ALG_ML_DSA_65,
    })


def test_result_is_immutable():
    """PolicyResult is frozen — callers can't mutate the verdict mid-flight."""
    result = evaluate_signatures(
        [_sig(ALG_ECDSA_SECP256K1_SHA256)],
        VerifyPolicy.LEGACY_ALLOWED,
    )
    with pytest.raises(Exception):
        result.ok = False  # type: ignore[misc]
