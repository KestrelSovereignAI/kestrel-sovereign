"""
MLDSA65Suite tests — Wave 2 sub-PR 2 (#917).

Covers:
- Round-trip sign/verify
- NIST FIPS 204 Cat-3 sizes (1952 / 4032 / 3309)
- Tampering rejection
- Wrong-key rejection
- Type validation (bytes-only API)
- Multikey integration (multicodec 0x1207 ``ml-dsa-65-pub``)
- ``is_post_quantum = True`` (verify-policy classification)
- Cross-suite isolation (ML-DSA sig ≠ secp256k1 sig)
- Library-version smoke check (pqcrypto installed and usable)
"""

from __future__ import annotations

import os

import pytest

from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    ALG_ML_DSA_65,
    CryptoSuiteError,
    Keypair,
    MLDSA65Suite,
    get_suite,
    list_registered,
)
from kestrel_sovereign.security.multikey import (
    base58btc_decode,
    public_key_to_multibase,
    multibase_to_public_key,
)


# ---------------------------------------------------------------------------
# Library smoke check
# ---------------------------------------------------------------------------

def test_pqcrypto_installed_and_importable():
    """If pqcrypto is missing the entire suite is unusable. Smoke check
    so a missing-dep failure produces a clear test failure rather than
    a confusing AttributeError elsewhere."""
    from pqcrypto.sign import ml_dsa_65
    assert ml_dsa_65.PUBLIC_KEY_SIZE == 1952
    assert ml_dsa_65.SECRET_KEY_SIZE == 4032
    assert ml_dsa_65.SIGNATURE_SIZE == 3309
    assert ml_dsa_65.ALGORITHM == "ml_dsa_65"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_mldsa65_suite_self_registers():
    suite = get_suite(ALG_ML_DSA_65)
    assert isinstance(suite, MLDSA65Suite)


def test_mldsa65_classified_as_post_quantum():
    """Verify-policy uses this to enforce HYBRID_REQUIRED / PQ_REQUIRED."""
    assert MLDSA65Suite().is_post_quantum is True


def test_mldsa65_listed_in_registry():
    assert ALG_ML_DSA_65 in list_registered()


# ---------------------------------------------------------------------------
# Sizes — pinned to NIST FIPS 204 Cat-3
# ---------------------------------------------------------------------------

def test_class_size_constants_match_nist_fips_204_cat_3():
    suite = MLDSA65Suite()
    assert suite.PUBLIC_KEY_SIZE == 1952
    assert suite.SECRET_KEY_SIZE == 4032
    assert suite.SIGNATURE_SIZE == 3309


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def suite() -> MLDSA65Suite:
    return MLDSA65Suite()


@pytest.fixture(scope="module")
def keypair(suite) -> Keypair:
    """Module-scoped: ML-DSA keygen is ~ms-class; sharing a keypair across
    tests in this module avoids repeated keygen overhead."""
    return suite.generate_keypair()


def test_keypair_carries_correct_suite_id(keypair):
    assert keypair.suite_id == ALG_ML_DSA_65


def test_keypair_byte_sizes_match_constants(keypair, suite):
    assert len(keypair.public_key) == suite.PUBLIC_KEY_SIZE
    assert len(keypair.private_key) == suite.SECRET_KEY_SIZE


def test_sign_verify_round_trip(suite, keypair):
    data = b"the canonical hash bytes that get signed"
    sig = suite.sign(data, keypair.private_key)
    assert suite.verify(data, sig, keypair.public_key) is True


def test_sign_verify_round_trip_empty(suite, keypair):
    sig = suite.sign(b"", keypair.private_key)
    assert suite.verify(b"", sig, keypair.public_key) is True


def test_sign_verify_round_trip_large(suite, keypair):
    data = os.urandom(100_000)  # 100 KB — big enough to exercise the path
    sig = suite.sign(data, keypair.private_key)
    assert suite.verify(data, sig, keypair.public_key) is True


def test_signature_byte_length_within_bounds(suite, keypair):
    """ML-DSA-65 signatures are at most SIGNATURE_SIZE bytes."""
    sig = suite.sign(b"x", keypair.private_key)
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

def test_verify_rejects_tampered_data(suite, keypair):
    sig = suite.sign(b"original", keypair.private_key)
    assert suite.verify(b"tampered", sig, keypair.public_key) is False


def test_verify_rejects_tampered_signature(suite, keypair):
    sig = bytearray(suite.sign(b"x", keypair.private_key))
    sig[100] ^= 0x01
    assert suite.verify(b"x", bytes(sig), keypair.public_key) is False


def test_verify_rejects_wrong_key(suite, keypair):
    other = suite.generate_keypair()
    sig = suite.sign(b"x", keypair.private_key)
    assert suite.verify(b"x", sig, other.public_key) is False


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


def test_verify_rejects_non_bytes_public_key(suite, keypair):
    sig = suite.sign(b"x", keypair.private_key)
    # Non-bytes public key returns False (not raises) per the suite contract
    assert suite.verify(b"x", sig, "not-bytes") is False  # type: ignore[arg-type]


def test_serialize_rejects_non_bytes_public_key(suite):
    with pytest.raises(CryptoSuiteError, match="must be bytes"):
        suite.serialize_public_key("not-bytes")  # type: ignore[arg-type]


def test_deserialize_rejects_wrong_length(suite):
    with pytest.raises(CryptoSuiteError, match="must be"):
        suite.deserialize_public_key(b"\x00" * 100)  # not 1952 bytes


def test_deserialize_rejects_non_bytes(suite):
    with pytest.raises(CryptoSuiteError, match="must be bytes"):
        suite.deserialize_public_key("not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public-key serialization
# ---------------------------------------------------------------------------

def test_public_key_serialize_round_trip(suite, keypair):
    raw = suite.serialize_public_key(keypair.public_key)
    rebuilt = suite.deserialize_public_key(raw)
    sig = suite.sign(b"verify-with-rebuilt", keypair.private_key)
    assert suite.verify(b"verify-with-rebuilt", sig, rebuilt) is True


def test_public_key_serialize_emits_1952_bytes(suite, keypair):
    raw = suite.serialize_public_key(keypair.public_key)
    assert len(raw) == 1952


def test_multikey_form_equals_legacy_form(suite, keypair):
    """ML-DSA-65 has only one canonical wire form (unlike secp256k1's
    compressed/uncompressed split). Multikey and legacy must produce
    identical bytes."""
    legacy = suite.serialize_public_key(keypair.public_key)
    mk = suite.serialize_public_key_for_multikey(keypair.public_key)
    assert legacy == mk


# ---------------------------------------------------------------------------
# Multikey integration
# ---------------------------------------------------------------------------

def test_multikey_carries_mldsa65_codec(suite, keypair):
    """Cross-implementation interop: multicodec 0x87 0x24 (varint of
    0x1207 ml-dsa-65-pub, proposed) must precede the public-key bytes."""
    mb = public_key_to_multibase(suite, keypair.public_key)
    assert mb.startswith("z")
    raw = base58btc_decode(mb[1:])
    assert raw[:2] == b"\x87\x24", (
        f"ml-dsa-65 Multikey body must start with codec varint b'\\x87\\x24'; "
        f"got {raw[:2]!r}"
    )
    assert len(raw[2:]) == 1952, "ml-dsa-65 public key body must be 1952 bytes"


def test_multikey_round_trip(suite, keypair):
    mb = public_key_to_multibase(suite, keypair.public_key)
    rebuilt_suite, rebuilt_pub = multibase_to_public_key(mb)
    assert rebuilt_suite.alg_id == ALG_ML_DSA_65

    sig = suite.sign(b"identity-roundtrip", keypair.private_key)
    assert rebuilt_suite.verify(b"identity-roundtrip", sig, rebuilt_pub) is True


# ---------------------------------------------------------------------------
# Cross-suite isolation
# ---------------------------------------------------------------------------

def test_mldsa65_signature_does_not_verify_under_secp256k1(suite, keypair):
    """An ML-DSA-65 signature must NOT verify under a secp256k1 cipher.
    Different algorithm, completely different size — verifier must reject
    cleanly without crashing."""
    secp = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    secp_kp = secp.generate_keypair()

    sig = suite.sign(b"x", keypair.private_key)
    # secp_suite.verify takes a secp256k1 public key; passing an ML-DSA
    # public key (1952 bytes) is the wrong type entirely — verify must
    # not crash. The Secp256k1Suite.verify catches all exceptions and
    # returns False.
    assert secp.verify(b"x", sig, secp_kp.public_key) is False


# ---------------------------------------------------------------------------
# Determinism / non-determinism observation
# ---------------------------------------------------------------------------

def test_signatures_may_be_non_deterministic(suite, keypair):
    """ML-DSA per FIPS 204 is randomized by default in pqcrypto. Two
    successive sign() calls on the same input may produce different
    signatures, but BOTH must verify. Locks the assumption — if the
    library ever switches to deterministic mode the test would still
    pass (== same sig, both verify).
    """
    sig_a = suite.sign(b"same-input", keypair.private_key)
    sig_b = suite.sign(b"same-input", keypair.private_key)
    # Both must verify (whether they're equal or not)
    assert suite.verify(b"same-input", sig_a, keypair.public_key) is True
    assert suite.verify(b"same-input", sig_b, keypair.public_key) is True
