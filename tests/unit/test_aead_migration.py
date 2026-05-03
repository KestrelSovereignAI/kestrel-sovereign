"""
Migration tests — Wave 0C sub-PR 2.

Proves the cross-version contract for the SDK encryption module:

1. Data written with the previous Fernet implementation still decrypts
   under the new AEADCipher-backed callers.
2. Data written by the new callers uses the v2 (``KSAv2:``) format.
3. Per-agent and purpose-based key derivation paths preserve their key
   values across the migration (so legacy ciphertext under derived keys
   keeps working).
"""

import base64
import os

import pytest
from cryptography.fernet import Fernet

from kestrel_sdk.security.aead import AEADCipher, KSA_V2_PREFIX
from kestrel_sdk.security import encryption as enc_sdk


@pytest.fixture
def master_key_env(monkeypatch):
    """Set a deterministic KESTREL_DATA_KEY for the duration of the test."""
    raw = b"\x00" * 32
    fernet_key = base64.urlsafe_b64encode(raw).decode("ascii")
    monkeypatch.setenv("KESTREL_DATA_KEY", fernet_key)
    yield raw


# -----------------------------------------------------------------------------
# get_fernet() now returns AEADCipher, but legacy Fernet data still decrypts
# -----------------------------------------------------------------------------

def test_get_fernet_returns_aeadcipher(master_key_env):
    """Legacy name preserved; new return type is AEADCipher."""
    cipher = enc_sdk.get_fernet()
    assert isinstance(cipher, AEADCipher)


def test_legacy_fernet_data_decrypts_via_get_fernet(master_key_env):
    """Data written with stock Fernet under the same KESTREL_DATA_KEY must
    decrypt under the new get_fernet() return."""
    raw = master_key_env
    legacy = Fernet(base64.urlsafe_b64encode(raw))
    fernet_token = legacy.encrypt(b"old data from before wave 0c")

    cipher = enc_sdk.get_fernet()
    assert cipher.decrypt(fernet_token) == b"old data from before wave 0c"


def test_new_writes_emit_v2(master_key_env):
    cipher = enc_sdk.get_fernet()
    token = cipher.encrypt(b"new data after wave 0c")
    assert token.startswith(KSA_V2_PREFIX)


# -----------------------------------------------------------------------------
# Per-agent derivation
# -----------------------------------------------------------------------------

def test_get_agent_fernet_returns_aeadcipher(master_key_env):
    cipher = enc_sdk.get_agent_fernet("did:web:alice.example")
    assert isinstance(cipher, AEADCipher)


def test_agent_keyed_data_round_trips_across_versions(master_key_env):
    """A row encrypted with the previous (Fernet-based) per-agent key
    must decrypt under the new AEADCipher-based get_agent_fernet."""
    agent_id = "did:web:alice.example"

    # Reproduce what the old code did: HKDF → urlsafe-b64 → Fernet
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    master = enc_sdk.get_master_key_bytes()
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=agent_id.encode(),
        info=b"kestrel-agent-v1",
    ).derive(master)
    legacy = Fernet(base64.urlsafe_b64encode(derived))
    legacy_token = legacy.encrypt(b"agent-keyed-legacy")

    # Now read it through the migrated path
    cipher = enc_sdk.get_agent_fernet(agent_id)
    assert cipher.decrypt(legacy_token) == b"agent-keyed-legacy"

    # And new writes are v2
    new_token = cipher.encrypt(b"agent-keyed-new")
    assert new_token.startswith(KSA_V2_PREFIX)


# -----------------------------------------------------------------------------
# Purpose-based encrypt/decrypt
# -----------------------------------------------------------------------------

def test_purpose_based_legacy_data_decrypts(master_key_env):
    """Purpose-based ``encrypt`` previously wrapped via Fernet; the same
    derived key must still decrypt that legacy ciphertext."""
    agent_did = "did:web:alice.example"
    purpose = "conversations"

    # Reproduce the old encrypt path
    key = enc_sdk.get_agent_key(agent_did, purpose)
    legacy = Fernet(base64.urlsafe_b64encode(key))
    legacy_token = legacy.encrypt(b"purpose-legacy")

    # Read through the migrated decrypt
    assert enc_sdk.decrypt(agent_did, purpose, legacy_token) == b"purpose-legacy"


def test_purpose_based_new_writes_emit_v2(master_key_env):
    agent_did = "did:web:alice.example"
    purpose = "wallet"
    token = enc_sdk.encrypt(agent_did, purpose, b"purpose-new")
    assert token.startswith(KSA_V2_PREFIX)
    # Round-trip
    assert enc_sdk.decrypt(agent_did, purpose, token) == b"purpose-new"


# -----------------------------------------------------------------------------
# Cipher-instance helpers (encrypt_string_fernet / decrypt_string_fernet)
# -----------------------------------------------------------------------------

def test_string_helpers_round_trip_v2(master_key_env):
    cipher = enc_sdk.get_fernet()
    encrypted, did_encrypt = enc_sdk.encrypt_string_fernet("hello", cipher)
    assert did_encrypt is True
    assert encrypted.startswith("KSAv2:")
    decrypted = enc_sdk.decrypt_string_fernet(encrypted, {"enc": True}, cipher)
    assert decrypted == "hello"


def test_string_helpers_decrypt_legacy_fernet(master_key_env):
    raw = master_key_env
    legacy = Fernet(base64.urlsafe_b64encode(raw))
    legacy_token = legacy.encrypt(b"legacy-string-data").decode("utf-8")

    cipher = enc_sdk.get_fernet()
    decrypted = enc_sdk.decrypt_string_fernet(legacy_token, {"enc": True}, cipher)
    assert decrypted == "legacy-string-data"


def test_string_helpers_skip_when_no_cipher_and_not_marked():
    """No cipher, no enc flag → pass-through."""
    out = enc_sdk.decrypt_string_fernet("plaintext", None, None)
    assert out == "plaintext"


def test_string_helpers_raise_when_no_cipher_but_marked():
    from kestrel_sdk.security.exceptions import DecryptionError
    with pytest.raises(DecryptionError, match="No decryption key"):
        enc_sdk.decrypt_string_fernet("ciphertext", {"enc": True}, None)
