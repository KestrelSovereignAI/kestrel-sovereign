"""
KEMSuite tests — Wave 4 sub-PR 1 (#919).

Covers both X25519Suite (classical KEM half) and MLKEM768Suite
(post-quantum KEM half):

- Library smoke checks (sizes match RFC 7748 / FIPS 203)
- Registry self-registration
- Round-trip encapsulate / decapsulate
- Wrong-key behavior:
  * X25519: produces a different shared secret (no error)
  * ML-KEM-768: FIPS-203 implicit rejection — different SS, no error
- Tamper / malformed ciphertext rejection
- Type validation (bytes-only API where applicable)
- Public-key serialization round-trip
- Multikey codec correctness
- ``is_post_quantum`` classification
"""

from __future__ import annotations

import os

import pytest

from kestrel_sovereign.security.kem_suite import (
    ALG_ML_KEM_768,
    ALG_X25519,
    KEMKeypair,
    KEMSuiteError,
    MLKEM768Suite,
    X25519Suite,
    get_kem_suite,
    list_registered_kems,
)


# ---------------------------------------------------------------------------
# Library smoke checks
# ---------------------------------------------------------------------------

def test_x25519_library_available():
    from cryptography.hazmat.primitives.asymmetric import x25519
    priv = x25519.X25519PrivateKey.generate()
    assert priv.public_key() is not None


def test_mlkem768_library_available():
    """If pqcrypto.kem.ml_kem_768 is missing the entire suite is unusable."""
    from pqcrypto.kem import ml_kem_768
    assert ml_kem_768.PUBLIC_KEY_SIZE == 1184
    assert ml_kem_768.SECRET_KEY_SIZE == 2400
    assert ml_kem_768.CIPHERTEXT_SIZE == 1088
    assert ml_kem_768.PLAINTEXT_SIZE == 32
    assert ml_kem_768.ALGORITHM == "ml_kem_768"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_x25519_self_registers():
    suite = get_kem_suite(ALG_X25519)
    assert isinstance(suite, X25519Suite)


def test_mlkem768_self_registers():
    suite = get_kem_suite(ALG_ML_KEM_768)
    assert isinstance(suite, MLKEM768Suite)


def test_registry_lists_both():
    registered = list_registered_kems()
    assert ALG_X25519 in registered
    assert ALG_ML_KEM_768 in registered


def test_unknown_alg_raises():
    with pytest.raises(KEMSuiteError, match="no registered KEM suite"):
        get_kem_suite("not-a-real-kem")


# ---------------------------------------------------------------------------
# is_post_quantum classification
# ---------------------------------------------------------------------------

def test_x25519_classified_as_classical():
    assert X25519Suite().is_post_quantum is False


def test_mlkem768_classified_as_post_quantum():
    assert MLKEM768Suite().is_post_quantum is True


# ---------------------------------------------------------------------------
# Sizes pinned per spec
# ---------------------------------------------------------------------------

def test_x25519_size_constants():
    s = X25519Suite()
    assert s.PUBLIC_KEY_SIZE == 32
    assert s.PRIVATE_KEY_SIZE == 32
    assert s.CIPHERTEXT_SIZE == 32
    assert s.SHARED_SECRET_SIZE == 32


def test_mlkem768_size_constants():
    s = MLKEM768Suite()
    assert s.PUBLIC_KEY_SIZE == 1184
    assert s.SECRET_KEY_SIZE == 2400
    assert s.CIPHERTEXT_SIZE == 1088
    assert s.SHARED_SECRET_SIZE == 32


# ---------------------------------------------------------------------------
# X25519 round-trips
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def x25519_kp():
    return X25519Suite().generate_keypair()


def test_x25519_encap_decap_round_trip(x25519_kp):
    suite = X25519Suite()
    ct, ss_alice = suite.encapsulate(x25519_kp.public_key)
    ss_bob = suite.decapsulate(ct, x25519_kp.private_key)
    assert ss_alice == ss_bob
    assert len(ss_alice) == suite.SHARED_SECRET_SIZE


def test_x25519_keypair_carries_correct_suite_id(x25519_kp):
    assert x25519_kp.suite_id == ALG_X25519


def test_x25519_distinct_keypairs():
    suite = X25519Suite()
    a = suite.generate_keypair()
    b = suite.generate_keypair()
    a_pub = suite.serialize_public_key(a.public_key)
    b_pub = suite.serialize_public_key(b.public_key)
    assert a_pub != b_pub


def test_x25519_wrong_key_produces_different_secret(x25519_kp):
    """X25519 returns the ECDH product of any two keys; using a wrong
    key produces a different shared secret (no error)."""
    suite = X25519Suite()
    ct, ss_alice = suite.encapsulate(x25519_kp.public_key)
    other = suite.generate_keypair()
    ss_other = suite.decapsulate(ct, other.private_key)
    assert ss_alice != ss_other


def test_x25519_decapsulate_rejects_wrong_length(x25519_kp):
    suite = X25519Suite()
    with pytest.raises(KEMSuiteError, match="must be 32 bytes"):
        suite.decapsulate(b"\x00" * 16, x25519_kp.private_key)


def test_x25519_decapsulate_rejects_non_bytes(x25519_kp):
    suite = X25519Suite()
    with pytest.raises(KEMSuiteError, match="must be bytes"):
        suite.decapsulate("not-bytes", x25519_kp.private_key)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# X25519 serialization
# ---------------------------------------------------------------------------

def test_x25519_serialize_round_trip(x25519_kp):
    suite = X25519Suite()
    raw = suite.serialize_public_key(x25519_kp.public_key)
    assert len(raw) == 32
    rebuilt = suite.deserialize_public_key(raw)
    # Verify by encapsulating to the rebuilt key and decapsulating
    ct, ss_alice = suite.encapsulate(rebuilt)
    ss_bob = suite.decapsulate(ct, x25519_kp.private_key)
    assert ss_alice == ss_bob


def test_x25519_deserialize_rejects_wrong_length():
    suite = X25519Suite()
    with pytest.raises(KEMSuiteError, match="must be 32 bytes"):
        suite.deserialize_public_key(b"\x00" * 16)


def test_x25519_deserialize_rejects_non_bytes():
    suite = X25519Suite()
    with pytest.raises(KEMSuiteError, match="must be bytes"):
        suite.deserialize_public_key("not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# X25519 multikey
# ---------------------------------------------------------------------------

def test_x25519_multikey_codec(x25519_kp):
    """Cross-implementation interop: multicodec 0xec (x25519-pub),
    varint-encoded as b'\\xec\\x01'."""
    from kestrel_sovereign.security.multikey import (
        base58btc_decode,
        public_key_to_multibase,
    )
    suite = X25519Suite()
    mb = public_key_to_multibase(suite, x25519_kp.public_key)
    raw = base58btc_decode(mb[1:])
    assert raw[:2] == b"\xec\x01"
    assert len(raw[2:]) == 32


# ---------------------------------------------------------------------------
# ML-KEM-768 round-trips
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mlkem_kp():
    return MLKEM768Suite().generate_keypair()


def test_mlkem768_keypair_byte_sizes(mlkem_kp):
    s = MLKEM768Suite()
    assert len(mlkem_kp.public_key) == s.PUBLIC_KEY_SIZE
    assert len(mlkem_kp.private_key) == s.SECRET_KEY_SIZE


def test_mlkem768_encap_decap_round_trip(mlkem_kp):
    suite = MLKEM768Suite()
    ct, ss_alice = suite.encapsulate(mlkem_kp.public_key)
    ss_bob = suite.decapsulate(ct, mlkem_kp.private_key)
    assert ss_alice == ss_bob
    assert len(ss_alice) == suite.SHARED_SECRET_SIZE


def test_mlkem768_ciphertext_length(mlkem_kp):
    suite = MLKEM768Suite()
    ct, _ = suite.encapsulate(mlkem_kp.public_key)
    assert len(ct) == suite.CIPHERTEXT_SIZE


def test_mlkem768_distinct_encapsulations_produce_distinct_ciphertexts(mlkem_kp):
    """ML-KEM uses fresh randomness per encapsulation; two calls to the
    same public key produce different ciphertexts AND different
    shared secrets."""
    suite = MLKEM768Suite()
    ct_a, ss_a = suite.encapsulate(mlkem_kp.public_key)
    ct_b, ss_b = suite.encapsulate(mlkem_kp.public_key)
    assert ct_a != ct_b
    assert ss_a != ss_b


def test_mlkem768_wrong_key_implicit_rejection(mlkem_kp):
    """FIPS 203 implicit rejection: decapsulating with a wrong secret
    key produces a 32-byte secret that DIFFERS from the encapsulator's
    secret, but no error is raised. The AEAD downstream catches the
    mismatch via authentication failure."""
    suite = MLKEM768Suite()
    ct, ss_alice = suite.encapsulate(mlkem_kp.public_key)
    other = suite.generate_keypair()
    ss_wrong = suite.decapsulate(ct, other.private_key)
    assert len(ss_wrong) == suite.SHARED_SECRET_SIZE
    assert ss_wrong != ss_alice


def test_mlkem768_tampered_ciphertext_yields_different_secret(mlkem_kp):
    """Same FIPS-203 implicit-rejection rule for tampered ciphertext:
    no exception, just a different (and unusable) shared secret."""
    suite = MLKEM768Suite()
    ct, ss_alice = suite.encapsulate(mlkem_kp.public_key)
    tampered = bytearray(ct)
    tampered[42] ^= 0x01
    ss_after_tamper = suite.decapsulate(bytes(tampered), mlkem_kp.private_key)
    assert ss_after_tamper != ss_alice


def test_mlkem768_encapsulate_rejects_wrong_pubkey_length():
    suite = MLKEM768Suite()
    with pytest.raises(KEMSuiteError, match=f"must be {suite.PUBLIC_KEY_SIZE}"):
        suite.encapsulate(b"\x00" * 100)


def test_mlkem768_decapsulate_rejects_wrong_ciphertext_length(mlkem_kp):
    suite = MLKEM768Suite()
    with pytest.raises(KEMSuiteError, match=f"must be {suite.CIPHERTEXT_SIZE}"):
        suite.decapsulate(b"\x00" * 100, mlkem_kp.private_key)


def test_mlkem768_decapsulate_rejects_non_bytes_secret(mlkem_kp):
    suite = MLKEM768Suite()
    ct, _ = suite.encapsulate(mlkem_kp.public_key)
    with pytest.raises(KEMSuiteError, match="must be bytes"):
        suite.decapsulate(ct, "not-bytes")  # type: ignore[arg-type]


def test_mlkem768_encapsulate_rejects_non_bytes_pubkey():
    suite = MLKEM768Suite()
    with pytest.raises(KEMSuiteError, match="must be bytes"):
        suite.encapsulate("not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ML-KEM-768 serialization
# ---------------------------------------------------------------------------

def test_mlkem768_serialize_round_trip(mlkem_kp):
    suite = MLKEM768Suite()
    raw = suite.serialize_public_key(mlkem_kp.public_key)
    assert len(raw) == 1184
    rebuilt = suite.deserialize_public_key(raw)
    ct, ss_alice = suite.encapsulate(rebuilt)
    ss_bob = suite.decapsulate(ct, mlkem_kp.private_key)
    assert ss_alice == ss_bob


def test_mlkem768_deserialize_rejects_wrong_length():
    suite = MLKEM768Suite()
    with pytest.raises(KEMSuiteError, match="must be 1184"):
        suite.deserialize_public_key(b"\x00" * 100)


# ---------------------------------------------------------------------------
# ML-KEM-768 multikey
# ---------------------------------------------------------------------------

def test_mlkem768_multikey_codec(mlkem_kp):
    """Multicodec 0x1209 (ml-kem-768-pub, proposed), varint-encoded
    as b'\\x89\\x24'."""
    from kestrel_sovereign.security.multikey import base58btc_decode
    suite = MLKEM768Suite()
    raw_with_codec = suite.public_key_multicodec + suite.serialize_public_key_for_multikey(mlkem_kp.public_key)
    assert raw_with_codec[:2] == b"\x89\x24"
    assert len(raw_with_codec[2:]) == 1184


# ---------------------------------------------------------------------------
# Cross-suite isolation
# ---------------------------------------------------------------------------

def test_distinct_codecs_for_x25519_and_mlkem768():
    """Wire-format invariant: codecs must NOT collide across KEM suites."""
    assert X25519Suite().public_key_multicodec != MLKEM768Suite().public_key_multicodec


# ---------------------------------------------------------------------------
# Multikey round-trip via the KEM-aware decoder (codex P2 round 1)
# ---------------------------------------------------------------------------

def test_x25519_multikey_decoder_round_trip(x25519_kp):
    """Codex P2: KEM codecs were registered only in _KEM_REGISTRY, so
    the existing multibase_to_public_key (signing-only) couldn't decode
    them. Now multibase_to_kem_public_key handles the KEM registry.
    """
    from kestrel_sovereign.security.multikey import (
        multibase_to_kem_public_key,
        public_key_to_multibase,
    )
    suite = X25519Suite()
    mb = public_key_to_multibase(suite, x25519_kp.public_key)
    rebuilt_suite, rebuilt_pub = multibase_to_kem_public_key(mb)
    assert rebuilt_suite.alg_id == ALG_X25519

    # Strongest end-to-end check: encapsulate to the rebuilt key,
    # decapsulate with the original private — must agree.
    ct, ss_alice = rebuilt_suite.encapsulate(rebuilt_pub)
    ss_bob = suite.decapsulate(ct, x25519_kp.private_key)
    assert ss_alice == ss_bob


def test_mlkem768_multikey_decoder_round_trip(mlkem_kp):
    from kestrel_sovereign.security.multikey import (
        multibase_to_kem_public_key,
        public_key_to_multibase,
    )
    suite = MLKEM768Suite()
    mb = public_key_to_multibase(suite, mlkem_kp.public_key)
    rebuilt_suite, rebuilt_pub = multibase_to_kem_public_key(mb)
    assert rebuilt_suite.alg_id == ALG_ML_KEM_768

    ct, ss_alice = rebuilt_suite.encapsulate(rebuilt_pub)
    ss_bob = suite.decapsulate(ct, mlkem_kp.private_key)
    assert ss_alice == ss_bob


def test_signing_decoder_refuses_kem_codec(x25519_kp):
    """The SIGNING decoder must NOT silently accept a KEM codec —
    callers that accidentally hand a key-agreement key into a signing
    code path should fail loud."""
    from kestrel_sovereign.security.crypto_suite import CryptoSuiteError
    from kestrel_sovereign.security.multikey import (
        multibase_to_public_key,
        public_key_to_multibase,
    )
    suite = X25519Suite()
    mb = public_key_to_multibase(suite, x25519_kp.public_key)
    with pytest.raises(CryptoSuiteError, match="No registered signing suite"):
        multibase_to_public_key(mb)


def test_kem_decoder_refuses_signing_codec():
    """And vice-versa: a signing-suite-encoded multibase shouldn't
    decode as a KEM key."""
    from kestrel_sovereign.security.crypto_suite import (
        Ed25519Suite,
    )
    from kestrel_sovereign.security.multikey import (
        multibase_to_kem_public_key,
        public_key_to_multibase,
    )
    ed = Ed25519Suite()
    kp = ed.generate_keypair()
    mb = public_key_to_multibase(ed, kp.public_key)
    with pytest.raises(KEMSuiteError, match="No registered KEM suite"):
        multibase_to_kem_public_key(mb)
