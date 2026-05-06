"""
Ed25519Suite tests — Wave 2 sub-PR 1 (#917).

Covers:
- Round-trip sign/verify
- Tampering rejection
- Wrong-key rejection
- 32-byte raw public-key serialization (matches W3C did:key multicodec 0xed)
- RFC 8032 Test 1 known-answer vector (deterministic ed25519 sig)
- Multikey integration (multicodec 0xed -> 32-byte raw payload)
- Registry self-registration at import time
- Cross-suite registry coexistence with secp256k1
"""

from __future__ import annotations

import os

import pytest

from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    ALG_ED25519,
    CryptoSuiteError,
    Ed25519Suite,
    Keypair,
    get_suite,
    list_registered,
)
from kestrel_sovereign.security.multikey import (
    base58btc_decode,
    public_key_to_multibase,
    multibase_to_public_key,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_ed25519_suite_self_registers_at_import():
    suite = get_suite(ALG_ED25519)
    assert isinstance(suite, Ed25519Suite)


def test_ed25519_suite_listed_alongside_secp256k1():
    ids = list_registered()
    assert ALG_ED25519 in ids
    assert ALG_ECDSA_SECP256K1_SHA256 in ids


def test_ed25519_classified_as_classical():
    assert Ed25519Suite().is_post_quantum is False


# ---------------------------------------------------------------------------
# Sign / verify round-trips
# ---------------------------------------------------------------------------

@pytest.fixture
def suite() -> Ed25519Suite:
    return Ed25519Suite()


@pytest.fixture
def keypair(suite) -> Keypair:
    return suite.generate_keypair()


def test_keypair_carries_correct_suite_id(keypair):
    assert keypair.suite_id == ALG_ED25519


def test_sign_verify_round_trip(suite, keypair):
    data = b"the canonical hash bytes that get signed"
    sig = suite.sign(data, keypair.private_key)
    assert suite.verify(data, sig, keypair.public_key) is True


def test_sign_verify_round_trip_empty(suite, keypair):
    sig = suite.sign(b"", keypair.private_key)
    assert suite.verify(b"", sig, keypair.public_key) is True


def test_sign_verify_round_trip_large(suite, keypair):
    data = os.urandom(1_000_000)
    sig = suite.sign(data, keypair.private_key)
    assert suite.verify(data, sig, keypair.public_key) is True


def test_sign_returns_64_byte_signature(suite, keypair):
    """Ed25519 signatures are exactly 64 bytes per RFC 8032."""
    sig = suite.sign(b"x", keypair.private_key)
    assert len(sig) == 64, f"expected 64-byte ed25519 signature, got {len(sig)}"


def test_generate_yields_distinct_keys(suite):
    a = suite.generate_keypair()
    b = suite.generate_keypair()
    a_pub = suite.serialize_public_key(a.public_key)
    b_pub = suite.serialize_public_key(b.public_key)
    assert a_pub != b_pub


# ---------------------------------------------------------------------------
# Tampering / wrong-key rejection
# ---------------------------------------------------------------------------

def test_verify_rejects_tampered_data(suite, keypair):
    sig = suite.sign(b"original", keypair.private_key)
    assert suite.verify(b"tampered", sig, keypair.public_key) is False


def test_verify_rejects_tampered_signature(suite, keypair):
    sig = bytearray(suite.sign(b"x", keypair.private_key))
    sig[10] ^= 0x01
    assert suite.verify(b"x", bytes(sig), keypair.public_key) is False


def test_verify_rejects_wrong_key(suite, keypair):
    other = suite.generate_keypair()
    sig = suite.sign(b"x", keypair.private_key)
    assert suite.verify(b"x", sig, other.public_key) is False


def test_verify_rejects_garbage(suite, keypair):
    assert suite.verify(b"x", b"not-a-signature", keypair.public_key) is False


def test_verify_rejects_empty_signature(suite, keypair):
    assert suite.verify(b"x", b"", keypair.public_key) is False


# ---------------------------------------------------------------------------
# Public-key serialization
# ---------------------------------------------------------------------------

def test_serialize_emits_32_raw_bytes(suite, keypair):
    """Ed25519 public keys are 32 raw bytes — same shape as the W3C
    did:key multicodec 0xed body."""
    raw = suite.serialize_public_key(keypair.public_key)
    assert len(raw) == 32, f"expected 32-byte ed25519 pubkey, got {len(raw)}"


def test_public_key_round_trip(suite, keypair):
    raw = suite.serialize_public_key(keypair.public_key)
    rebuilt = suite.deserialize_public_key(raw)
    sig = suite.sign(b"verify-with-rebuilt", keypair.private_key)
    assert suite.verify(b"verify-with-rebuilt", sig, rebuilt) is True


def test_multikey_form_equals_legacy_form(suite, keypair):
    """Ed25519 has only one canonical form — multikey and legacy must
    produce identical bytes (unlike secp256k1 where multikey is
    compressed and legacy is uncompressed)."""
    legacy = suite.serialize_public_key(keypair.public_key)
    mk = suite.serialize_public_key_for_multikey(keypair.public_key)
    assert legacy == mk


def test_deserialize_rejects_garbage(suite):
    with pytest.raises(CryptoSuiteError):
        suite.deserialize_public_key(b"\x00" * 10)


# ---------------------------------------------------------------------------
# Multikey integration
# ---------------------------------------------------------------------------

def test_multikey_string_carries_ed25519_codec(suite, keypair):
    """Cross-implementation interop: multicodec 0xed 0x01 must precede
    the 32-byte public key in the base58btc body."""
    mb = public_key_to_multibase(suite, keypair.public_key)
    assert mb.startswith("z")
    raw = base58btc_decode(mb[1:])
    assert raw[:2] == b"\xed\x01", (
        f"ed25519 Multikey body must be 0xed 0x01 prefixed; got {raw[:2]!r}"
    )
    assert len(raw[2:]) == 32, "ed25519 public key body must be 32 bytes"


def test_multikey_round_trip(suite, keypair):
    mb = public_key_to_multibase(suite, keypair.public_key)
    rebuilt_suite, rebuilt_pub = multibase_to_public_key(mb)
    assert rebuilt_suite.alg_id == ALG_ED25519
    sig = suite.sign(b"identity-roundtrip", keypair.private_key)
    assert rebuilt_suite.verify(b"identity-roundtrip", sig, rebuilt_pub) is True


# ---------------------------------------------------------------------------
# RFC 8032 Test 1 known-answer vector
# ---------------------------------------------------------------------------
#
# Ed25519 is deterministic — the same (key, message) always produces the
# same signature. Anchor with the RFC 8032 Section 7.1 Test 1 vector to
# detect any future implementation drift.

# RFC 8032 §7.1 Test 1:
#   secret key seed: 9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60
#   public key:      d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a
#   message:         (empty)
#   signature:       e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b
RFC_8032_T1_SEED = bytes.fromhex(
    "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
)
RFC_8032_T1_PUBKEY = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
)
RFC_8032_T1_SIG = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f"
    "b8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)


def test_rfc_8032_test_1_known_answer():
    """Deterministic ed25519: sign-then-byte-compare against RFC 8032 §7.1.

    Anchors that ``Ed25519Suite.sign`` produces the spec-mandated bytes,
    not just *some* valid signature. Catches implementation drift in
    a future ``cryptography`` library upgrade.
    """
    from cryptography.hazmat.primitives.asymmetric import ed25519
    suite = Ed25519Suite()
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(RFC_8032_T1_SEED)
    sig = suite.sign(b"", priv)
    assert sig == RFC_8032_T1_SIG, (
        "ed25519 signature must match RFC 8032 §7.1 Test 1 known-answer"
    )

    # And cross-check verification with the spec's public key
    pub = ed25519.Ed25519PublicKey.from_public_bytes(RFC_8032_T1_PUBKEY)
    assert suite.verify(b"", RFC_8032_T1_SIG, pub) is True

    # Also verify the suite's own keypair derivation produces the right pubkey
    assert suite.serialize_public_key(priv.public_key()) == RFC_8032_T1_PUBKEY


# ---------------------------------------------------------------------------
# Cross-suite isolation
# ---------------------------------------------------------------------------

def test_ed25519_signature_does_not_verify_under_secp256k1(suite, keypair):
    """An Ed25519 signature must NOT verify under a secp256k1 cipher;
    different curve, different verification math."""
    secp_suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    secp_kp = secp_suite.generate_keypair()

    sig = suite.sign(b"x", keypair.private_key)
    # secp_suite.verify takes a secp256k1 public key — passing an Ed25519
    # public key would TypeError or fail; the cross-suite path should be
    # explicitly rejected by callers, not relied on to "just work".
    assert secp_suite.verify(b"x", sig, secp_kp.public_key) is False
