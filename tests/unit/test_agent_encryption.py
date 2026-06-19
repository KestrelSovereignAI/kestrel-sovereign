"""
Tests for the unified agent encryption module.

No mocks. No fallbacks. Just verify the encryption works correctly.
"""
import os
import pytest

from kestrel_sovereign.security.agent_encryption import (
    get_agent_key,
    encrypt,
    decrypt,
    encrypt_string,
    decrypt_string,
    MasterKeyNotConfiguredError,
    InvalidPurposeError,
    DecryptionError,
    VALID_PURPOSES,
)


# Test DID constants
TEST_DID = "did:pkh:eip155:1:0xfeedfacefeedfacefeedfacefeedfacefeedface"
OTHER_DID = "did:pkh:eip155:1:0x1234567890abcdef1234567890abcdef12345678"


@pytest.fixture(autouse=True)
def set_test_key(monkeypatch):
    """Set a test encryption key for all tests."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-encryption-key-for-unit-tests")


class TestKeyDerivation:
    """Test key derivation properties."""

    def test_deterministic_key_derivation(self):
        """Same DID + purpose = same key."""
        key1 = get_agent_key(TEST_DID, "service-keys")
        key2 = get_agent_key(TEST_DID, "service-keys")
        assert key1 == key2

    def test_different_agents_different_keys(self):
        """Different DIDs = different keys."""
        key1 = get_agent_key(TEST_DID, "service-keys")
        key2 = get_agent_key(OTHER_DID, "service-keys")
        assert key1 != key2

    def test_different_purposes_different_keys(self):
        """Different purposes = different keys."""
        key1 = get_agent_key(TEST_DID, "service-keys")
        key2 = get_agent_key(TEST_DID, "conversations")
        assert key1 != key2

    def test_all_purposes_produce_unique_keys(self):
        """All four purposes produce unique keys for the same agent."""
        keys = [get_agent_key(TEST_DID, purpose) for purpose in VALID_PURPOSES]
        assert len(set(keys)) == len(VALID_PURPOSES)

    def test_key_length_is_32_bytes(self):
        """Keys are always 32 bytes (256 bits)."""
        for purpose in VALID_PURPOSES:
            key = get_agent_key(TEST_DID, purpose)
            assert len(key) == 32


class TestEncryptDecrypt:
    """Test encryption and decryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypt then decrypt returns original."""
        plaintext = b"my secret api key"
        ciphertext = encrypt(TEST_DID, "service-keys", plaintext)
        decrypted = decrypt(TEST_DID, "service-keys", ciphertext)
        assert decrypted == plaintext

    def test_encrypt_decrypt_string_roundtrip(self):
        """String convenience functions work correctly."""
        plaintext = "my secret api key with unicode: 🔐"
        ciphertext = encrypt_string(TEST_DID, "service-keys", plaintext)
        decrypted = decrypt_string(TEST_DID, "service-keys", ciphertext)
        assert decrypted == plaintext

    def test_encryption_produces_different_ciphertext_each_time(self):
        """Same plaintext produces different ciphertext (random nonce)."""
        plaintext = b"same data"
        c1 = encrypt(TEST_DID, "service-keys", plaintext)
        c2 = encrypt(TEST_DID, "service-keys", plaintext)
        assert c1 != c2  # Different random nonces

        # But both decrypt to same plaintext
        assert decrypt(TEST_DID, "service-keys", c1) == plaintext
        assert decrypt(TEST_DID, "service-keys", c2) == plaintext

    def test_ciphertext_longer_than_plaintext(self):
        """Ciphertext includes nonce (12 bytes) + auth tag (16 bytes)."""
        plaintext = b"test"
        ciphertext = encrypt(TEST_DID, "service-keys", plaintext)
        # At minimum: 12 (nonce) + 4 (plaintext) + 16 (tag) = 32 bytes
        assert len(ciphertext) >= len(plaintext) + 28

    def test_wrong_agent_cannot_decrypt(self):
        """Data encrypted for one agent cannot be decrypted by another."""
        plaintext = b"emma's secret"
        ciphertext = encrypt(TEST_DID, "service-keys", plaintext)

        with pytest.raises(DecryptionError):
            decrypt(OTHER_DID, "service-keys", ciphertext)

    def test_wrong_purpose_cannot_decrypt(self):
        """Data encrypted for one purpose cannot be decrypted with another."""
        plaintext = b"api key"
        ciphertext = encrypt(TEST_DID, "service-keys", plaintext)

        with pytest.raises(DecryptionError):
            decrypt(TEST_DID, "conversations", ciphertext)


class TestErrorHandling:
    """Test error conditions."""

    def test_no_master_key_raises_error(self, monkeypatch):
        """Missing KESTREL_DATA_KEY raises MasterKeyNotConfiguredError."""
        monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)

        with pytest.raises(MasterKeyNotConfiguredError):
            get_agent_key(TEST_DID, "service-keys")

    def test_empty_agent_did_raises_error(self):
        """Empty agent_did raises ValueError."""
        with pytest.raises(ValueError, match="agent_did is required"):
            get_agent_key("", "service-keys")

    def test_invalid_purpose_raises_error(self):
        """Invalid purpose raises InvalidPurposeError."""
        with pytest.raises(InvalidPurposeError):
            get_agent_key(TEST_DID, "invalid-purpose")

    def test_decrypt_short_data_raises_error(self):
        """Invalid/short data raises DecryptionError (Fernet rejects invalid tokens)."""
        with pytest.raises(DecryptionError):
            decrypt(TEST_DID, "service-keys", b"short")

    def test_decrypt_corrupted_data_raises_error(self):
        """Corrupted ciphertext raises DecryptionError."""
        plaintext = b"test"
        ciphertext = encrypt(TEST_DID, "service-keys", plaintext)

        # Corrupt the ciphertext
        corrupted = ciphertext[:-1] + bytes([(ciphertext[-1] + 1) % 256])

        with pytest.raises(DecryptionError):
            decrypt(TEST_DID, "service-keys", corrupted)


class TestAllPurposes:
    """Test all valid purposes work correctly."""

    @pytest.mark.parametrize("purpose", list(VALID_PURPOSES))
    def test_purpose_encrypt_decrypt(self, purpose):
        """Each purpose can encrypt and decrypt."""
        plaintext = f"data for {purpose}".encode()
        ciphertext = encrypt(TEST_DID, purpose, plaintext)
        decrypted = decrypt(TEST_DID, purpose, ciphertext)
        assert decrypted == plaintext
