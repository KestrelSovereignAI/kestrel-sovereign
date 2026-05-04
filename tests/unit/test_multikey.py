"""
multikey + KeypairFactory tests — Wave 1 sub-PR 2 (#916).

Covers base58btc encode/decode (RFC test vectors), unsigned LEB128
varint roundtrips, multikey/multibase end-to-end for Secp256k1Suite,
and the thin KeypairFactory orchestration layer.
"""

from __future__ import annotations

import os

import pytest

from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    CryptoSuiteError,
    Keypair,
    Secp256k1Suite,
    get_suite,
)
from kestrel_sovereign.security.keypair_factory import (
    DEFAULT_SUITE_ID,
    KeypairFactory,
)
from kestrel_sovereign.security.multikey import (
    MULTIBASE_BASE58BTC_PREFIX,
    base58btc_decode,
    base58btc_encode,
    decode_varint,
    encode_varint,
    multibase_to_public_key,
    public_key_to_multibase,
)


# ---------------------------------------------------------------------------
# base58btc round-trips
# ---------------------------------------------------------------------------

def test_base58btc_empty():
    assert base58btc_encode(b"") == ""
    assert base58btc_decode("") == b""


@pytest.mark.parametrize("payload", [
    b"\x00",
    b"\x00\x00\x00",
    b"\x01",
    b"\xff",
    b"\xff" * 32,
    b"hello world",
    b"\x00\x01\x02\x03",  # leading zero bytes
    b"\x00\x00hello",     # multiple leading zeros
])
def test_base58btc_round_trip(payload):
    encoded = base58btc_encode(payload)
    assert base58btc_decode(encoded) == payload


def test_base58btc_random_round_trip():
    for length in (1, 16, 32, 64, 128, 256, 1000):
        data = os.urandom(length)
        assert base58btc_decode(base58btc_encode(data)) == data


def test_base58btc_leading_zeros_preserved():
    """Leading zero bytes must be preserved as leading '1' chars."""
    assert base58btc_encode(b"\x00\x00abc").startswith("11")


def test_base58btc_rejects_non_alphabet():
    with pytest.raises(ValueError, match="non-base58btc"):
        base58btc_decode("abc0")  # '0' is not in the alphabet


def test_base58btc_rejects_capital_o_etc():
    """The alphabet excludes 0, O, I, l. Any of those must reject."""
    for invalid in ("0", "O", "I", "l"):
        with pytest.raises(ValueError, match="non-base58btc"):
            base58btc_decode(f"abc{invalid}def")


# ---------------------------------------------------------------------------
# multicodec varint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected_bytes", [
    (0, b"\x00"),
    (1, b"\x01"),
    (127, b"\x7f"),
    (128, b"\x80\x01"),
    (231, b"\xe7\x01"),  # secp256k1-pub
    (237, b"\xed\x01"),  # ed25519-pub
    (16384, b"\x80\x80\x01"),
])
def test_varint_known_values(value, expected_bytes):
    assert encode_varint(value) == expected_bytes
    decoded, consumed = decode_varint(expected_bytes)
    assert decoded == value
    assert consumed == len(expected_bytes)


def test_varint_negative_rejects():
    with pytest.raises(ValueError):
        encode_varint(-1)


def test_varint_decode_truncated():
    with pytest.raises(ValueError, match="truncated"):
        decode_varint(b"\x80")  # continuation bit set with no follow-up


def test_varint_consumes_only_its_bytes():
    """decode_varint must report num_bytes_consumed so a multicodec prefix
    can be split from the raw payload that follows."""
    # 0xe7 0x01 followed by 5 raw bytes
    data = b"\xe7\x01\xaa\xbb\xcc\xdd\xee"
    value, consumed = decode_varint(data)
    assert value == 231
    assert consumed == 2
    assert data[consumed:] == b"\xaa\xbb\xcc\xdd\xee"


# ---------------------------------------------------------------------------
# Multikey end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def keypair() -> Keypair:
    return Secp256k1Suite().generate_keypair()


def test_public_key_to_multibase_starts_with_z(keypair):
    suite = get_suite(keypair.suite_id)
    s = public_key_to_multibase(suite, keypair.public_key)
    assert s.startswith(MULTIBASE_BASE58BTC_PREFIX), (
        f"Multikey strings must use base58btc prefix 'z'; got {s[:1]!r}"
    )


def test_multibase_round_trip(keypair):
    suite = get_suite(keypair.suite_id)
    encoded = public_key_to_multibase(suite, keypair.public_key)
    rebuilt_suite, rebuilt_pub = multibase_to_public_key(encoded)
    assert rebuilt_suite.alg_id == keypair.suite_id
    # And the rebuilt key serializes back to the same bytes
    assert (
        rebuilt_suite.serialize_public_key(rebuilt_pub)
        == suite.serialize_public_key(keypair.public_key)
    )


def test_multibase_carries_secp256k1_codec(keypair):
    """The multicodec prefix in the encoded payload must be 0xe7 0x01
    (secp256k1-pub). Locks the wire identity for cross-implementation
    interop."""
    suite = get_suite(keypair.suite_id)
    encoded = public_key_to_multibase(suite, keypair.public_key)
    raw = base58btc_decode(encoded[1:])  # strip 'z' prefix
    assert raw[:2] == b"\xe7\x01"


def test_multibase_secp256k1_body_is_33_byte_compressed_point(keypair):
    """W3C did:key / Multikey specifies multicodec 0xe7 = secp256k1-pub
    as a 33-byte compressed X9.62 point (leading byte 0x02 or 0x03 for
    Y parity).

    Pre-fix this test would have asserted ``len(body) == 65`` and prefix
    0x04 — the legacy uncompressed form ``inception_service.public_key_to_hex``
    emits today. That format is incompatible with cross-implementation
    DID/Multikey readers under the same multicodec; they would reject
    the string or rederive a different key.

    The legacy uncompressed form stays in ``serialize_public_key`` (used
    by ``publicKeyHex``); only the Multikey path switches to compressed.
    """
    suite = get_suite(keypair.suite_id)
    encoded = public_key_to_multibase(suite, keypair.public_key)
    raw = base58btc_decode(encoded[1:])  # strip 'z' prefix
    body = raw[2:]  # strip 2-byte multicodec varint
    assert len(body) == 33, (
        f"secp256k1 Multikey body must be 33 bytes (compressed); got {len(body)}. "
        f"This breaks cross-implementation interop under multicodec 0xe7."
    )
    assert body[0] in (0x02, 0x03), (
        f"secp256k1 compressed point must start with 0x02 or 0x03 (Y parity); "
        f"got 0x{body[0]:02x}"
    )


def test_legacy_serialize_public_key_stays_uncompressed(keypair):
    """``serialize_public_key`` (NOT the multikey one) must stay
    uncompressed 65-byte X9.62 — pinned for backwards compatibility with
    v1 DID documents that ship ``publicKeyHex`` from
    ``inception_service.public_key_to_hex``.
    """
    suite = get_suite(keypair.suite_id)
    raw = suite.serialize_public_key(keypair.public_key)
    assert len(raw) == 65
    assert raw[0] == 0x04


def test_multikey_round_trip_via_compressed_form(keypair):
    """End-to-end: encode via the multikey path, decode through the
    multikey path, sign with original private key, verify with rebuilt
    public key. Locks the compressed-form contract."""
    suite = get_suite(keypair.suite_id)
    encoded = public_key_to_multibase(suite, keypair.public_key)
    rebuilt_suite, rebuilt_pub = multibase_to_public_key(encoded)
    assert rebuilt_suite.alg_id == suite.alg_id

    sig = suite.sign(b"compressed-roundtrip", keypair.private_key)
    assert rebuilt_suite.verify(b"compressed-roundtrip", sig, rebuilt_pub) is True


def test_multibase_unknown_codec_rejected():
    """A multibase string with an unregistered multicodec must fail loud
    rather than silently mis-decode under a wrong suite."""
    # Construct a payload with codec 0x99 0x99 (unregistered)
    fake = MULTIBASE_BASE58BTC_PREFIX + base58btc_encode(b"\x99\x99" + b"\x00" * 30)
    with pytest.raises(CryptoSuiteError, match="No registered suite for multicodec"):
        multibase_to_public_key(fake)


def test_multibase_wrong_prefix_rejected():
    with pytest.raises(CryptoSuiteError, match="multibase base58btc prefix"):
        multibase_to_public_key("babc")  # 'b' is base32lower, not base58btc


def test_multibase_empty_payload_rejected():
    with pytest.raises(CryptoSuiteError, match="empty multibase payload"):
        multibase_to_public_key("z")


def test_suite_without_multicodec_raises():
    """A suite whose ``public_key_multicodec`` is unset (e.g. a future
    placeholder before its multicodec is registered) cannot produce a
    Multikey string. Fail loud so identity-package writers don't ship
    a placeholder string."""

    class FakeSuite(Secp256k1Suite):
        alg_id = "fake-no-codec"
        public_key_multicodec = b""  # no codec assigned

    suite = FakeSuite()
    kp = suite.generate_keypair()
    with pytest.raises(CryptoSuiteError, match="no public_key_multicodec"):
        public_key_to_multibase(suite, kp.public_key)


# ---------------------------------------------------------------------------
# KeypairFactory
# ---------------------------------------------------------------------------

def test_factory_default_suite_id():
    assert DEFAULT_SUITE_ID == ALG_ECDSA_SECP256K1_SHA256


def test_factory_generate_explicit_suite():
    kp = KeypairFactory.generate(ALG_ECDSA_SECP256K1_SHA256)
    assert kp.suite_id == ALG_ECDSA_SECP256K1_SHA256
    # And the keypair is usable through the suite
    suite = get_suite(kp.suite_id)
    sig = suite.sign(b"x", kp.private_key)
    assert suite.verify(b"x", sig, kp.public_key) is True


def test_factory_generate_default():
    kp = KeypairFactory.generate_default()
    assert kp.suite_id == DEFAULT_SUITE_ID


def test_factory_generate_unknown_suite_raises():
    with pytest.raises(CryptoSuiteError, match="No suite registered"):
        KeypairFactory.generate("never-registered")


def test_factory_multibase_helpers_round_trip():
    kp = KeypairFactory.generate_default()
    encoded = KeypairFactory.public_key_to_multibase(kp)
    suite_id, pub = KeypairFactory.multibase_to_public_key(encoded)
    assert suite_id == kp.suite_id

    # Verify a signature produced by the original key against the
    # rebuilt public key — full identity round-trip.
    suite = get_suite(suite_id)
    sig = suite.sign(b"identity", kp.private_key)
    assert suite.verify(b"identity", sig, pub) is True


def test_factory_lists_registered_suites():
    ids = KeypairFactory.registered_suite_ids()
    assert ALG_ECDSA_SECP256K1_SHA256 in ids
