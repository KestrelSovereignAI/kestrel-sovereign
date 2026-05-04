"""
SLHDSASHA2128sSuite tests — Wave 3 sub-PR 1 (#918).

Covers:
- Round-trip sign/verify
- NIST FIPS 205 SLH-DSA-SHA2-128s sizes (32 / 64 / 7856)
- Tampering rejection
- Wrong-key rejection
- Type validation (bytes-only API)
- Multikey integration (multicodec 0x1208 ``slh-dsa-sha2-128s-pub``)
- ``is_post_quantum = True`` (verify-policy classification)
- Cross-suite isolation (SLH-DSA sig ≠ ML-DSA-65 sig under wrong verifier)
- Library-version smoke check (pqcrypto installed and usable)
"""

from __future__ import annotations

import os

import pytest

from kestrel_sovereign.security.crypto_suite import (
    ALG_ML_DSA_65,
    ALG_SLH_DSA_SHA2_128S,
    CryptoSuiteError,
    Keypair,
    SLHDSASHA2128sSuite,
    get_suite,
    list_registered,
)
from kestrel_sovereign.security.multikey import (
    base58btc_decode,
    multibase_to_public_key,
    public_key_to_multibase,
)


# ---------------------------------------------------------------------------
# Library smoke check
# ---------------------------------------------------------------------------

def test_pqcrypto_sphincs_installed_and_importable():
    """If pqcrypto's SPHINCS+ binding is missing the entire suite is
    unusable. Smoke check so a missing-dep failure produces a clear test
    failure rather than a confusing AttributeError elsewhere."""
    from pqcrypto.sign import sphincs_sha2_128s_simple as slh
    assert slh.PUBLIC_KEY_SIZE == 32
    assert slh.SECRET_KEY_SIZE == 64
    assert slh.SIGNATURE_SIZE == 7856
    assert slh.ALGORITHM == "sphincs_sha2_128s_simple"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_slhdsa_suite_self_registers():
    suite = get_suite(ALG_SLH_DSA_SHA2_128S)
    assert isinstance(suite, SLHDSASHA2128sSuite)


def test_slhdsa_classified_as_post_quantum():
    """Verify-policy uses this to enforce HYBRID_REQUIRED / PQ_REQUIRED."""
    assert SLHDSASHA2128sSuite().is_post_quantum is True


def test_slhdsa_listed_in_registry():
    assert ALG_SLH_DSA_SHA2_128S in list_registered()


# ---------------------------------------------------------------------------
# Sizes — pinned to NIST FIPS 205 SLH-DSA-SHA2-128s
# ---------------------------------------------------------------------------

def test_class_size_constants_match_nist_fips_205_128s():
    suite = SLHDSASHA2128sSuite()
    assert suite.PUBLIC_KEY_SIZE == 32
    assert suite.SECRET_KEY_SIZE == 64
    assert suite.SIGNATURE_SIZE == 7856


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------

# SLH-DSA-128s signatures are slow to generate (~1s typical on commodity
# hardware) — module-scoped keypair shared across round-trip tests so we
# pay keygen + at most a couple of signs total in this module.
@pytest.fixture(scope="module")
def suite() -> SLHDSASHA2128sSuite:
    return SLHDSASHA2128sSuite()


@pytest.fixture(scope="module")
def keypair(suite) -> Keypair:
    return suite.generate_keypair()


@pytest.fixture(scope="module")
def signed_data(suite, keypair) -> tuple[bytes, bytes]:
    """A single signature reused across verify-side tests. Avoids
    re-signing 7856-byte signatures repeatedly when the test only
    cares about the verify path."""
    data = b"the canonical succession-statement bytes"
    sig = suite.sign(data, keypair.private_key)
    return data, sig


def test_keypair_carries_correct_suite_id(keypair):
    assert keypair.suite_id == ALG_SLH_DSA_SHA2_128S


def test_keypair_byte_sizes_match_constants(keypair, suite):
    assert len(keypair.public_key) == suite.PUBLIC_KEY_SIZE
    assert len(keypair.private_key) == suite.SECRET_KEY_SIZE


def test_sign_verify_round_trip(suite, keypair, signed_data):
    data, sig = signed_data
    assert suite.verify(data, sig, keypair.public_key) is True


def test_sign_verify_round_trip_empty(suite, keypair):
    sig = suite.sign(b"", keypair.private_key)
    assert suite.verify(b"", sig, keypair.public_key) is True


def test_signature_byte_length_within_bounds(suite, signed_data):
    """SLH-DSA-128s signatures are at most SIGNATURE_SIZE bytes."""
    _, sig = signed_data
    assert len(sig) <= suite.SIGNATURE_SIZE


def test_generate_yields_distinct_keypairs(suite):
    """Independent keygens produce distinct keys — sanity check that
    the underlying RNG is not seeded deterministically across calls."""
    a = suite.generate_keypair()
    b = suite.generate_keypair()
    assert a.public_key != b.public_key


# ---------------------------------------------------------------------------
# Tampering / wrong-key rejection
# ---------------------------------------------------------------------------

def test_verify_rejects_tampered_data(suite, keypair, signed_data):
    _, sig = signed_data
    assert suite.verify(b"tampered", sig, keypair.public_key) is False


def test_verify_rejects_tampered_signature(suite, keypair, signed_data):
    data, sig = signed_data
    bad = bytearray(sig)
    bad[100] ^= 0x01
    assert suite.verify(data, bytes(bad), keypair.public_key) is False


def test_verify_rejects_wrong_key(suite, keypair, signed_data):
    data, sig = signed_data
    other = suite.generate_keypair()
    assert suite.verify(data, sig, other.public_key) is False


def test_verify_rejects_garbage_signature(suite, keypair):
    assert suite.verify(b"x", b"not-a-signature", keypair.public_key) is False


def test_verify_rejects_empty_signature(suite, keypair):
    assert suite.verify(b"x", b"", keypair.public_key) is False


def test_verify_rejects_short_signature(suite, keypair):
    assert suite.verify(b"x", b"\x00" * 100, keypair.public_key) is False


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------

def test_sign_rejects_non_bytes_private_key(suite):
    with pytest.raises(CryptoSuiteError, match="must be bytes"):
        suite.sign(b"x", "not-bytes")  # type: ignore[arg-type]


def test_verify_rejects_non_bytes_public_key(suite, signed_data):
    data, sig = signed_data
    # Non-bytes public key returns False (not raises) per the suite contract
    assert suite.verify(data, sig, "not-bytes") is False  # type: ignore[arg-type]


def test_serialize_rejects_non_bytes_public_key(suite):
    with pytest.raises(CryptoSuiteError, match="must be bytes"):
        suite.serialize_public_key("not-bytes")  # type: ignore[arg-type]


def test_deserialize_rejects_wrong_length(suite):
    with pytest.raises(CryptoSuiteError, match="must be"):
        suite.deserialize_public_key(b"\x00" * 16)  # not 32 bytes


def test_deserialize_rejects_non_bytes(suite):
    with pytest.raises(CryptoSuiteError, match="must be bytes"):
        suite.deserialize_public_key("not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public-key serialization
# ---------------------------------------------------------------------------

def test_public_key_serialize_round_trip(suite, keypair, signed_data):
    raw = suite.serialize_public_key(keypair.public_key)
    rebuilt = suite.deserialize_public_key(raw)
    data, sig = signed_data
    assert suite.verify(data, sig, rebuilt) is True


def test_public_key_serialize_emits_32_bytes(suite, keypair):
    raw = suite.serialize_public_key(keypair.public_key)
    assert len(raw) == 32


def test_multikey_form_equals_legacy_form(suite, keypair):
    """SLH-DSA-128s has only one canonical wire form (32-byte raw).
    Multikey and legacy must produce identical bytes."""
    legacy = suite.serialize_public_key(keypair.public_key)
    mk = suite.serialize_public_key_for_multikey(keypair.public_key)
    assert legacy == mk


# ---------------------------------------------------------------------------
# Multikey integration
# ---------------------------------------------------------------------------

def test_multikey_carries_slhdsa_codec(suite, keypair):
    """Cross-implementation interop: multicodec 0x88 0x24 (varint of
    0x1208 slh-dsa-sha2-128s-pub, proposed) must precede the public-key
    bytes."""
    mb = public_key_to_multibase(suite, keypair.public_key)
    assert mb.startswith("z")
    raw = base58btc_decode(mb[1:])
    assert raw[:2] == b"\x88\x24", (
        f"slh-dsa-sha2-128s Multikey body must start with codec varint "
        f"b'\\x88\\x24'; got {raw[:2]!r}"
    )
    assert len(raw[2:]) == 32, "slh-dsa-sha2-128s public key body must be 32 bytes"


def test_multikey_round_trip(suite, keypair, signed_data):
    mb = public_key_to_multibase(suite, keypair.public_key)
    rebuilt_suite, rebuilt_pub = multibase_to_public_key(mb)
    assert rebuilt_suite.alg_id == ALG_SLH_DSA_SHA2_128S

    data, sig = signed_data
    assert rebuilt_suite.verify(data, sig, rebuilt_pub) is True


# ---------------------------------------------------------------------------
# Cross-suite isolation
# ---------------------------------------------------------------------------

def test_slhdsa_signature_does_not_verify_under_mldsa65(suite, keypair, signed_data):
    """An SLH-DSA-128s signature must NOT verify under an ML-DSA-65 cipher.
    Different algorithm, different sizes — verifier must reject cleanly
    without crashing."""
    mldsa = get_suite(ALG_ML_DSA_65)
    mldsa_kp = mldsa.generate_keypair()

    data, sig = signed_data
    assert mldsa.verify(data, sig, mldsa_kp.public_key) is False


def test_distinct_codec_from_mldsa65(suite):
    """Wire-format invariant: SLH-DSA's multicodec must NOT collide with
    ML-DSA-65's. A registry that mistakenly assigned them the same codec
    would mis-route public keys at multibase decode time."""
    from kestrel_sovereign.security.crypto_suite import MLDSA65Suite
    assert suite.public_key_multicodec != MLDSA65Suite().public_key_multicodec


# ---------------------------------------------------------------------------
# Determinism / non-determinism observation
# ---------------------------------------------------------------------------

def test_signatures_may_be_non_deterministic(suite, keypair):
    """FIPS 205 allows both deterministic (det:) and randomized variants
    of SLH-DSA. This pqcrypto build defaults to the randomized form, so
    two successive signs over the same input may produce different
    signatures — but BOTH must verify. Locks the observation; if the
    library ever flips to deterministic mode the test still passes.

    Workflow note: succession ceremonies that hash a chain of signatures
    therefore must canonicalize the ``signatures`` array carefully if they
    rely on byte-stable countersigning across re-signs — which Wave 3
    does NOT (each succession statement is signed once, archived, and
    referenced by content hash thereafter).
    """
    sig_a = suite.sign(b"same-input", keypair.private_key)
    sig_b = suite.sign(b"same-input", keypair.private_key)
    assert suite.verify(b"same-input", sig_a, keypair.public_key) is True
    assert suite.verify(b"same-input", sig_b, keypair.public_key) is True
