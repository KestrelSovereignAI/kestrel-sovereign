"""
CryptoSuite + Secp256k1Suite KAT tests — Wave 1 (#916).

Covers the abstract-base contract, the registry behavior, and the
concrete Secp256k1Suite (round-trips, tampering rejection, wrong-key
rejection, serialization, behavior-preservation against today's
``cryptography.hazmat`` calls).
"""

from __future__ import annotations

import os

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat,
)

from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    CryptoSuite,
    CryptoSuiteError,
    Keypair,
    Secp256k1Suite,
    _REGISTRY,
    get_suite,
    list_registered,
    register_suite,
)


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

def test_secp256k1_suite_registered_at_import():
    """Importing ``crypto_suite`` self-registers the default suite so callers
    can ``get_suite("ecdsa-secp256k1-sha256")`` without setup."""
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    assert isinstance(suite, Secp256k1Suite)


def test_unknown_alg_id_raises():
    with pytest.raises(CryptoSuiteError, match="No suite registered"):
        get_suite("never-registered-alg")


def test_register_same_instance_is_idempotent():
    """Re-registering the SAME instance must not raise (so reload-friendly)."""
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    register_suite(suite)  # no-op; same instance
    assert get_suite(ALG_ECDSA_SECP256K1_SHA256) is suite


def test_register_competing_instance_raises():
    """A different instance for the same alg_id must raise — guards against
    silent overwrites during library bake-off."""
    new_instance = Secp256k1Suite()
    with pytest.raises(CryptoSuiteError, match="already registered"):
        register_suite(new_instance)


def test_list_registered_includes_default():
    assert ALG_ECDSA_SECP256K1_SHA256 in list_registered()


# ---------------------------------------------------------------------------
# Abstract base contract
# ---------------------------------------------------------------------------

def test_cryptosuite_is_abstract():
    """A subclass that doesn't implement all abstract methods cannot be instantiated."""
    class Incomplete(CryptoSuite):
        alg_id = "incomplete"
        def generate_keypair(self):  # pragma: no cover
            raise NotImplementedError
    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Secp256k1Suite round-trips
# ---------------------------------------------------------------------------

@pytest.fixture
def suite() -> Secp256k1Suite:
    return Secp256k1Suite()


@pytest.fixture
def keypair(suite) -> Keypair:
    return suite.generate_keypair()


def test_keypair_carries_correct_suite_id(keypair):
    assert keypair.suite_id == ALG_ECDSA_SECP256K1_SHA256


def test_generate_yields_distinct_keypairs(suite):
    a = suite.generate_keypair()
    b = suite.generate_keypair()
    a_pub = suite.serialize_public_key(a.public_key)
    b_pub = suite.serialize_public_key(b.public_key)
    assert a_pub != b_pub


def test_sign_verify_round_trip_bytes(suite, keypair):
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


# ---------------------------------------------------------------------------
# Tampering / wrong-key rejection
# ---------------------------------------------------------------------------

def test_verify_rejects_tampered_data(suite, keypair):
    sig = suite.sign(b"original", keypair.private_key)
    assert suite.verify(b"tampered", sig, keypair.public_key) is False


def test_verify_rejects_tampered_signature(suite, keypair):
    data = b"payload"
    sig = bytearray(suite.sign(data, keypair.private_key))
    sig[10] ^= 0x01
    assert suite.verify(data, bytes(sig), keypair.public_key) is False


def test_verify_rejects_wrong_key(suite, keypair):
    """A signature produced under one key must not verify under another."""
    other = suite.generate_keypair()
    sig = suite.sign(b"x", keypair.private_key)
    assert suite.verify(b"x", sig, other.public_key) is False


def test_verify_rejects_garbage_signature(suite, keypair):
    assert suite.verify(b"x", b"not-a-signature", keypair.public_key) is False


def test_verify_rejects_empty_signature(suite, keypair):
    assert suite.verify(b"x", b"", keypair.public_key) is False


# ---------------------------------------------------------------------------
# Public-key serialization round-trip
# ---------------------------------------------------------------------------

def test_public_key_serialize_round_trip(suite, keypair):
    raw = suite.serialize_public_key(keypair.public_key)
    rebuilt = suite.deserialize_public_key(raw)
    # Same canonical bytes after reserialization
    assert suite.serialize_public_key(rebuilt) == raw

    # And the rebuilt key actually verifies signatures from the original
    sig = suite.sign(b"verify-with-rebuilt", keypair.private_key)
    assert suite.verify(b"verify-with-rebuilt", sig, rebuilt) is True


def test_serialize_emits_uncompressed_x962(suite, keypair):
    """Wire-compat with existing ``inception_service.public_key_to_hex``:
    65-byte 04-prefixed uncompressed point, same shape that ships in v1
    DID documents.
    """
    raw = suite.serialize_public_key(keypair.public_key)
    assert len(raw) == 65, f"expected 65-byte uncompressed point, got {len(raw)}"
    assert raw[0] == 0x04, "uncompressed-point prefix byte"


def test_deserialize_rejects_garbage(suite):
    with pytest.raises(CryptoSuiteError, match="deserialization"):
        suite.deserialize_public_key(b"\x00" * 10)


# ---------------------------------------------------------------------------
# Behavior preservation vs current ad-hoc calls
# ---------------------------------------------------------------------------

def test_signature_verifies_against_raw_cryptography_call(suite, keypair):
    """A signature produced by the suite must verify under the same call
    pattern the existing ``identity/signing.py`` already uses
    (``public_key.verify(sig, data, ec.ECDSA(hashes.SHA256()))``).

    Pins behavior preservation: Wave 1 migration of call sites to the
    suite must not break readers that haven't migrated yet.
    """
    data = b"hello"
    sig = suite.sign(data, keypair.private_key)
    # Verify directly via the legacy call shape
    keypair.public_key.verify(sig, data, ec.ECDSA(hashes.SHA256()))  # raises on failure


def test_raw_cryptography_signature_verifies_through_suite(suite, keypair):
    """And the inverse: a signature produced by the legacy
    ``private_key.sign(data, ec.ECDSA(hashes.SHA256()))`` shape must
    verify through the suite. This is the migration safety net — call
    sites can be migrated incrementally without orphaning artifacts.
    """
    data = b"hello"
    legacy_sig = keypair.private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    assert suite.verify(data, legacy_sig, keypair.public_key) is True


def test_serialize_matches_inception_service_format(suite, keypair):
    """``inception_service.public_key_to_hex`` emits uncompressed X9.62 hex.
    The suite's serialize method must produce the same bytes (so v1 DID
    documents written before Wave 1 still round-trip after Wave 1)."""
    suite_bytes = suite.serialize_public_key(keypair.public_key)
    legacy_bytes = keypair.public_key.public_bytes(
        encoding=Encoding.X962,
        format=PublicFormat.UncompressedPoint,
    )
    assert suite_bytes == legacy_bytes
