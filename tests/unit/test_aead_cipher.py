"""
AEADCipher unit tests — Wave 0C (#915).

Covers:
- v2 round-trip
- Legacy Fernet round-trip (read-compatibility for migration)
- Cross-version: encrypt v2, decrypt v2; encrypt Fernet, decrypt via AEADCipher
- AAD binding (correct, mismatched, missing)
- Key coercion (raw vs URL-safe base64)
- Tampering detection (nonce flip, ct flip, prefix flip)
- Format invariants (always emits v2, never emits Fernet)
"""

import base64
import os

import pytest
from cryptography.fernet import Fernet

from kestrel_sdk.security.aead import AEADCipher, KSA_V2_PREFIX, KEY_SIZE
from kestrel_sdk.security.exceptions import DecryptionError


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def raw_key() -> bytes:
    return b"\x00" * KEY_SIZE  # KAT-style fixed key for stability across runs


@pytest.fixture
def random_key() -> bytes:
    return os.urandom(KEY_SIZE)


@pytest.fixture
def cipher(raw_key) -> AEADCipher:
    return AEADCipher(raw_key)


# -----------------------------------------------------------------------------
# v2 round-trip
# -----------------------------------------------------------------------------

def test_v2_round_trip_bytes(cipher):
    pt = b"hello world"
    ct = cipher.encrypt(pt)
    assert AEADCipher.is_v2(ct), "encrypt() must always emit v2 tokens"
    assert ct.startswith(KSA_V2_PREFIX)
    assert cipher.decrypt(ct) == pt


def test_v2_round_trip_string_plaintext(cipher):
    ct = cipher.encrypt("hello world")
    assert cipher.decrypt(ct) == b"hello world"


def test_v2_round_trip_empty(cipher):
    ct = cipher.encrypt(b"")
    assert cipher.decrypt(ct) == b""


def test_v2_round_trip_large(cipher):
    pt = os.urandom(1_000_000)  # 1 MB
    ct = cipher.encrypt(pt)
    assert cipher.decrypt(ct) == pt


def test_v2_token_decoded_form(cipher):
    """Sanity-check the token shape: prefix + base64(nonce + ct+tag)."""
    ct = cipher.encrypt(b"x")
    body = ct[len(KSA_V2_PREFIX):]
    raw = base64.urlsafe_b64decode(body)
    # nonce=12, plaintext=1 byte, tag=16 → 29 bytes
    assert len(raw) == 12 + 1 + 16


def test_each_encrypt_produces_unique_token(cipher):
    """Random nonces guarantee distinct ciphertexts for the same plaintext."""
    pt = b"deterministic input"
    tokens = {cipher.encrypt(pt) for _ in range(20)}
    assert len(tokens) == 20, "expected 20 distinct tokens (random nonces)"


# -----------------------------------------------------------------------------
# AAD
# -----------------------------------------------------------------------------

def test_aad_round_trip(cipher):
    aad = b"agent_id=did:web:alice.example|row=42|context=conversation"
    ct = cipher.encrypt(b"secret", aad=aad)
    assert cipher.decrypt(ct, aad=aad) == b"secret"


def test_aad_mismatch_fails(cipher):
    ct = cipher.encrypt(b"secret", aad=b"context-A")
    with pytest.raises(DecryptionError):
        cipher.decrypt(ct, aad=b"context-B")


def test_aad_missing_on_decrypt_fails(cipher):
    """Encrypted with AAD; decrypted without AAD must fail."""
    ct = cipher.encrypt(b"secret", aad=b"context-A")
    with pytest.raises(DecryptionError):
        cipher.decrypt(ct)


def test_aad_extra_on_decrypt_fails(cipher):
    """Encrypted without AAD; decrypted with AAD must fail."""
    ct = cipher.encrypt(b"secret")
    with pytest.raises(DecryptionError):
        cipher.decrypt(ct, aad=b"context-A")


# -----------------------------------------------------------------------------
# Tampering detection
# -----------------------------------------------------------------------------

def test_ciphertext_byte_flip_detected(cipher):
    ct = cipher.encrypt(b"important")
    # Flip a byte inside the base64-encoded body
    body = bytearray(ct[len(KSA_V2_PREFIX):])
    body[15] ^= 0x01
    tampered = KSA_V2_PREFIX + bytes(body)
    with pytest.raises(DecryptionError):
        cipher.decrypt(tampered)


def test_truncated_token_rejected(cipher):
    ct = cipher.encrypt(b"important")
    truncated = ct[:-10]
    with pytest.raises(DecryptionError):
        cipher.decrypt(truncated)


def test_too_short_v2_token_rejected(cipher):
    """A v2-prefixed payload too short to contain nonce+tag must fail loudly."""
    fake = KSA_V2_PREFIX + base64.urlsafe_b64encode(b"\x00" * 10)
    with pytest.raises(DecryptionError):
        cipher.decrypt(fake)


def test_garbled_v2_base64_rejected(cipher):
    fake = KSA_V2_PREFIX + b"!!!not base64!!!"
    with pytest.raises(DecryptionError):
        cipher.decrypt(fake)


# -----------------------------------------------------------------------------
# Legacy Fernet read-compat
# -----------------------------------------------------------------------------

def test_decrypts_legacy_fernet_token(raw_key):
    """A token written by stock Fernet must still decrypt under AEADCipher."""
    legacy = Fernet(base64.urlsafe_b64encode(raw_key))
    fernet_token = legacy.encrypt(b"old data from before wave 0c")
    assert not AEADCipher.is_v2(fernet_token)

    cipher = AEADCipher(raw_key)
    assert cipher.decrypt(fernet_token) == b"old data from before wave 0c"


def test_legacy_fernet_with_wrong_key_fails(raw_key):
    legacy = Fernet(base64.urlsafe_b64encode(raw_key))
    fernet_token = legacy.encrypt(b"x")

    other_cipher = AEADCipher(os.urandom(KEY_SIZE))
    with pytest.raises(DecryptionError):
        other_cipher.decrypt(fernet_token)


def test_legacy_fernet_token_with_aad_fails_explicitly(raw_key):
    """Passing AAD on a legacy Fernet decode must fail loudly, not silently
    ignore — Fernet has no AAD support and a silent skip would hide a
    context-binding mismatch."""
    legacy = Fernet(base64.urlsafe_b64encode(raw_key))
    fernet_token = legacy.encrypt(b"x")
    cipher = AEADCipher(raw_key)
    with pytest.raises(DecryptionError, match="AAD"):
        cipher.decrypt(fernet_token, aad=b"context-A")


def test_writes_only_emit_v2(cipher):
    """No write path emits Fernet — this is the migration guarantee."""
    for _ in range(10):
        ct = cipher.encrypt(os.urandom(64))
        assert ct.startswith(KSA_V2_PREFIX)
        # Specifically: must NOT start with Fernet's URL-safe-base64 'g' prefix
        assert not ct.startswith(b"g")


# -----------------------------------------------------------------------------
# Key coercion
# -----------------------------------------------------------------------------

def test_constructor_accepts_raw_32_bytes():
    AEADCipher(os.urandom(KEY_SIZE))  # should not raise


def test_constructor_accepts_urlsafe_base64_fernet_key():
    raw = os.urandom(KEY_SIZE)
    fernet_key = base64.urlsafe_b64encode(raw)  # 44 bytes
    a = AEADCipher(fernet_key)
    b = AEADCipher(raw)
    # Same underlying key → b can decrypt what a writes
    ct = a.encrypt(b"shared")
    assert b.decrypt(ct) == b"shared"


def test_constructor_accepts_str_key():
    raw = os.urandom(KEY_SIZE)
    fernet_key = base64.urlsafe_b64encode(raw).decode("ascii")
    AEADCipher(fernet_key)  # str form, should not raise


def test_constructor_rejects_wrong_length_raw():
    with pytest.raises(ValueError):
        AEADCipher(b"\x00" * 16)  # AES-128-sized, not allowed


def test_constructor_rejects_garbage():
    with pytest.raises(ValueError):
        AEADCipher(b"!!! definitely not a key !!!")


# -----------------------------------------------------------------------------
# is_v2 helper
# -----------------------------------------------------------------------------

def test_is_v2_helper(cipher):
    assert AEADCipher.is_v2(cipher.encrypt(b"x")) is True

    legacy = Fernet(base64.urlsafe_b64encode(b"\x00" * 32))
    assert AEADCipher.is_v2(legacy.encrypt(b"x")) is False

    assert AEADCipher.is_v2(b"") is False
    assert AEADCipher.is_v2("") is False
    assert AEADCipher.is_v2(b"random") is False
