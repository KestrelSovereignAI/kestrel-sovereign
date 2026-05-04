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

from kestrel_sdk.security.aead import (
    AEADCipher,
    KSA_V2_PREFIX,
    KEY_SIZE,
    NONCE_SIZE,
    ALG_ID_SIZE,
    GCM_TAG_SIZE,
    ALG_AES_256_GCM,
    ALG_NONE,
)
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
    """Sanity-check the token shape: prefix + base64(alg_id + nonce + ct+tag)."""
    ct = cipher.encrypt(b"x")
    body = ct[len(KSA_V2_PREFIX):]
    raw = base64.urlsafe_b64decode(body)
    # alg_id=1, nonce=12, plaintext=1 byte, tag=16 → 30 bytes
    assert len(raw) == ALG_ID_SIZE + NONCE_SIZE + 1 + GCM_TAG_SIZE
    # First byte is the algorithm identifier
    assert raw[0] == ALG_AES_256_GCM


def test_v2_token_carries_explicit_alg_byte(cipher):
    """The alg_id byte must be present in the framing so future suites
    can be added without bumping the version prefix. Bound into AAD so
    flipping it fails authentication."""
    ct = cipher.encrypt(b"payload")
    raw = base64.urlsafe_b64decode(ct[len(KSA_V2_PREFIX):])
    assert raw[0] == ALG_AES_256_GCM, (
        "v2 framing must start with an explicit alg_id byte "
        "(see SERIALIZATION_COMPATIBILITY.md cross-cutting rule #1)"
    )


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
# Strict base64 (regression for review feedback on PR #926)
# -----------------------------------------------------------------------------

def test_strict_b64_rejects_appended_junk(cipher):
    """Regression: lenient urlsafe_b64decode used to silently ignore non-base64
    characters appended to a valid token, so cipher.decrypt(ct + b'!!!') would
    return the original plaintext. Strict decoding rejects this."""
    ct = cipher.encrypt(b"important")
    with pytest.raises(DecryptionError):
        cipher.decrypt(ct + b"!!!")


def test_strict_b64_rejects_appended_newline(cipher):
    """Regression: same issue with whitespace/newlines."""
    ct = cipher.encrypt(b"important")
    with pytest.raises(DecryptionError):
        cipher.decrypt(ct + b"\n")


def test_strict_b64_rejects_internal_whitespace(cipher):
    """Whitespace inside the base64 body must also be rejected."""
    ct = cipher.encrypt(b"important")
    body = ct[len(KSA_V2_PREFIX):]
    spliced = KSA_V2_PREFIX + body[:8] + b"\n" + body[8:]
    with pytest.raises(DecryptionError):
        cipher.decrypt(spliced)


def test_strict_b64_rejects_outside_alphabet(cipher):
    """Characters outside the URL-safe-base64 alphabet must be rejected.

    Pre-fix: ``base64.b64decode(altchars=b"-_", validate=True)`` silently
    accepted standard-base64 ``+`` / ``/`` as aliases for ``-`` / ``_``,
    decoding them to the same byte values. Same plaintext could be carried
    by two different token strings — canonical encoding broken even though
    AEAD authentication still worked. The explicit alphabet pre-check in
    ``_strict_urlsafe_b64decode`` closes that hole.
    """
    ct = cipher.encrypt(b"important")
    body = bytearray(ct[len(KSA_V2_PREFIX):])
    body[3] = ord("+")
    with pytest.raises(DecryptionError) as excinfo:
        cipher.decrypt(KSA_V2_PREFIX + bytes(body))
    # The rejection must come from the alphabet check (canonical-encoding
    # diagnostic), not from a downstream AEAD tag failure
    assert "non-canonical encoding" in str(excinfo.value).lower()


def test_strict_b64_rejects_standard_alphabet_slash(cipher):
    """Standard-base64 ``/`` must be rejected (alias attack on canonical encoding)."""
    ct = cipher.encrypt(b"important")
    body = bytearray(ct[len(KSA_V2_PREFIX):])
    body[5] = ord("/")
    with pytest.raises(DecryptionError, match="non-canonical encoding"):
        cipher.decrypt(KSA_V2_PREFIX + bytes(body))


def test_token_carries_no_standard_alphabet_chars(cipher):
    """A freshly-encrypted token must not contain ``+`` or ``/``.

    AEADCipher uses ``urlsafe_b64encode`` on write, so this is a static
    invariant — but pinning it in a test guards against a future refactor
    accidentally switching to standard-base64 output, which would then be
    rejected by the strict decoder above and corrupt every read.
    """
    for _ in range(20):
        ct = cipher.encrypt(os.urandom(64))
        body = ct[len(KSA_V2_PREFIX):]
        assert b"+" not in body and b"/" not in body


# -----------------------------------------------------------------------------
# Algorithm identifier (alg_id byte)
# -----------------------------------------------------------------------------

def test_alg_id_tampering_fails_authentication(cipher):
    """Flipping the alg_id byte to coerce a different suite must fail
    authentication, because alg_id is bound into the AEAD AAD."""
    ct = cipher.encrypt(b"payload")
    raw = bytearray(base64.urlsafe_b64decode(ct[len(KSA_V2_PREFIX):]))
    # Flip alg_id from AES-256-GCM to "none" — this is the suite-swap attack
    raw[0] = ALG_NONE
    tampered = KSA_V2_PREFIX + base64.urlsafe_b64encode(bytes(raw))
    with pytest.raises(DecryptionError):
        cipher.decrypt(tampered)


def test_unknown_alg_id_rejected_with_diagnostic(cipher):
    """A future alg_id this build doesn't know about must fail with a clear
    message rather than mis-decrypt."""
    ct = cipher.encrypt(b"payload")
    raw = bytearray(base64.urlsafe_b64decode(ct[len(KSA_V2_PREFIX):]))
    raw[0] = 0x99  # unknown future suite
    tampered = KSA_V2_PREFIX + base64.urlsafe_b64encode(bytes(raw))
    with pytest.raises(DecryptionError, match="unknown alg_id"):
        cipher.decrypt(tampered)


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
