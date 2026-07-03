#!/usr/bin/env pytest
"""F187 portability: an encrypted export must be restorable on a *different*
host from only the CID + ``key_hash``, with no local key sidecar present.

Before the fix, ``_encrypt_content`` generated a random per-content key and
wrapped it into ``storage_cache/key_<hash>.key`` — a file that never travels
with the CID, so a fresh host raised ``FileNotFoundError`` on import. The key
is now derived deterministically from ``KESTREL_DATA_KEY`` + the content hash,
so the same key reproduces anywhere the data key is configured.
"""
import pytest
from cryptography.fernet import Fernet

# Import the storage providers package before filecoin_adapter so this module
# is runnable in isolation: filecoin_adapter <-> storage.sovereign_adapter form
# a pre-existing import cycle that only bites when filecoin_adapter is the very
# first module imported.
import kestrel_sovereign.storage.providers.base  # noqa: F401
from kestrel_sovereign.filecoin_adapter import FilecoinAdapter


@pytest.fixture
def data_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("KESTREL_DATA_KEY", key)
    return key


def test_encrypted_content_restores_without_sidecar(tmp_path, data_key):
    """Encrypt on 'host A', decrypt on 'host B' (fresh cache) with no sidecar."""
    plaintext = b'{"did": "did:pkh:eip155:1:0xabc", "secret": "state"}'

    # Host A encrypts.
    host_a = FilecoinAdapter(cache_dir=str(tmp_path / "host_a"))
    encrypted, key_hash = host_a._encrypt_content(plaintext)

    # No key sidecar is written any more — the whole point of the fix.
    assert not list((tmp_path / "host_a").glob("key_*.key"))
    # The key_hash is the plaintext content hash (portable identifier).
    import hashlib
    assert key_hash == hashlib.sha256(plaintext).hexdigest()

    # Host B has a completely separate cache dir (no sidecar) but the same
    # KESTREL_DATA_KEY. It must decrypt from ciphertext + key_hash alone.
    host_b = FilecoinAdapter(cache_dir=str(tmp_path / "host_b"))
    assert not list((tmp_path / "host_b").glob("key_*.key"))
    decrypted = host_b._decrypt_content(encrypted, key_hash)
    assert decrypted == plaintext


def test_store_and_retrieve_roundtrip_encrypted(tmp_path, data_key):
    """Full local-tier store/retrieve round trip with encryption on."""
    from kestrel_sovereign.storage.providers.base import StorageTier

    plaintext = b"portable identity package bytes"
    host_a = FilecoinAdapter(cache_dir=str(tmp_path / "a"))
    result = host_a.store_content(
        content=plaintext,
        storage_tier=StorageTier.LOCAL_ONLY,
        encrypt=True,
    )
    assert result.encrypted is True
    assert result.encryption_key_hash

    got = host_a.retrieve_content(
        result.content_hash, key_hash=result.encryption_key_hash
    )
    assert got == plaintext


def test_legacy_sidecar_still_decrypts(tmp_path, data_key):
    """Backward compatibility: content encrypted the OLD way (random key in a
    sidecar) must keep decrypting via the sidecar branch."""
    from kestrel_sdk.security.aead import AEADCipher
    import hashlib

    plaintext = b"legacy-encrypted-bytes"
    adapter = FilecoinAdapter(cache_dir=str(tmp_path / "legacy"))

    # Reproduce the pre-fix encryption: random content key wrapped in a sidecar.
    content_key = AEADCipher.generate_key()
    encrypted_content = AEADCipher(content_key).encrypt(plaintext)
    master_key = adapter._get_master_key()
    encrypted_key = AEADCipher(master_key).encrypt(content_key)
    legacy_hash = hashlib.sha256(encrypted_key).hexdigest()
    (adapter.cache_dir / f"key_{legacy_hash}.key").write_bytes(encrypted_key)

    assert adapter._decrypt_content(encrypted_content, legacy_hash) == plaintext
