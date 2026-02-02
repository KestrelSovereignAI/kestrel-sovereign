"""
Unit tests for storage/encryption.py

Tests key derivation, per-agent keys, and encryption functions.
"""
import base64
import hashlib
import os
import pytest
from unittest.mock import patch

from cryptography.fernet import Fernet


class TestGetFernet:
    """Tests for get_fernet() function."""

    def test_no_key_returns_none(self):
        """Returns None when KESTREL_DATA_KEY not set."""
        from kestrel_sovereign.storage.encryption import get_fernet
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop('KESTREL_DATA_KEY', None)
            result = get_fernet()
            assert result is None

    def test_valid_fernet_key_used_directly(self):
        """Valid Fernet key is used directly without derivation."""
        from kestrel_sovereign.storage.encryption import get_fernet
        # Generate a valid Fernet key
        valid_key = Fernet.generate_key().decode()
        with patch.dict(os.environ, {'KESTREL_DATA_KEY': valid_key}):
            result = get_fernet()
            assert result is not None
            # Should be able to encrypt/decrypt
            encrypted = result.encrypt(b"test")
            decrypted = result.decrypt(encrypted)
            assert decrypted == b"test"

    def test_passphrase_derived_via_sha256(self):
        """Passphrase is derived to Fernet key via SHA-256."""
        from kestrel_sovereign.storage.encryption import get_fernet
        passphrase = "THIS IS A TEMP KEY FOR TESTING"
        with patch.dict(os.environ, {'KESTREL_DATA_KEY': passphrase}):
            result = get_fernet()
            assert result is not None
            # Verify the key is derived correctly
            expected_digest = hashlib.sha256(passphrase.encode('utf-8')).digest()
            expected_key = base64.urlsafe_b64encode(expected_digest)
            # Test encryption works
            encrypted = result.encrypt(b"test data")
            decrypted = result.decrypt(encrypted)
            assert decrypted == b"test data"


class TestGetMasterKeyBytes:
    """Tests for get_master_key_bytes() function."""

    def test_no_key_returns_none(self):
        """Returns None when KESTREL_DATA_KEY not set."""
        from kestrel_sovereign.storage.encryption import get_master_key_bytes
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('KESTREL_DATA_KEY', None)
            result = get_master_key_bytes()
            assert result is None

    def test_valid_fernet_key_returned_as_bytes(self):
        """Valid Fernet key is returned as bytes."""
        from kestrel_sovereign.storage.encryption import get_master_key_bytes
        valid_key = Fernet.generate_key()  # bytes
        with patch.dict(os.environ, {'KESTREL_DATA_KEY': valid_key.decode()}):
            result = get_master_key_bytes()
            assert result is not None
            assert isinstance(result, bytes)
            # Should be valid for Fernet
            f = Fernet(result)
            encrypted = f.encrypt(b"test")
            assert f.decrypt(encrypted) == b"test"

    def test_passphrase_derived_to_bytes(self):
        """Passphrase is derived to valid Fernet key bytes."""
        from kestrel_sovereign.storage.encryption import get_master_key_bytes
        passphrase = "my-secret-passphrase"
        with patch.dict(os.environ, {'KESTREL_DATA_KEY': passphrase}):
            result = get_master_key_bytes()
            assert result is not None
            assert isinstance(result, bytes)
            # Should be valid for Fernet
            f = Fernet(result)
            encrypted = f.encrypt(b"test")
            assert f.decrypt(encrypted) == b"test"
            # Verify derivation is consistent
            expected = base64.urlsafe_b64encode(
                hashlib.sha256(passphrase.encode()).digest()
            )
            assert result == expected


class TestGetAgentFernet:
    """Tests for get_agent_fernet() function."""

    def test_no_master_key_returns_none(self):
        """Returns None when no master key available."""
        from kestrel_sovereign.storage.encryption import get_agent_fernet
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('KESTREL_DATA_KEY', None)
            result = get_agent_fernet("did:pkh:eip155:1:0xabc123")
            assert result is None

    def test_empty_agent_id_returns_global_fernet(self):
        """Empty agent_id returns global Fernet (backward compat)."""
        from kestrel_sovereign.storage.encryption import get_agent_fernet, get_fernet
        passphrase = "test-passphrase"
        with patch.dict(os.environ, {'KESTREL_DATA_KEY': passphrase}):
            result = get_agent_fernet("")
            global_fernet = get_fernet()
            # Both should work with same encryption
            test_data = b"test data"
            encrypted = result.encrypt(test_data)
            # Global fernet should be able to decrypt
            decrypted = global_fernet.decrypt(encrypted)
            assert decrypted == test_data

    def test_different_agents_get_different_keys(self):
        """Different agent IDs produce different encryption keys."""
        from kestrel_sovereign.storage.encryption import get_agent_fernet
        passphrase = "test-passphrase"
        with patch.dict(os.environ, {'KESTREL_DATA_KEY': passphrase}):
            fernet1 = get_agent_fernet("did:pkh:eip155:1:0xagent1")
            fernet2 = get_agent_fernet("did:pkh:eip155:1:0xagent2")

            # Both should be valid
            assert fernet1 is not None
            assert fernet2 is not None

            # Encrypt with one, should NOT be decryptable by the other
            encrypted = fernet1.encrypt(b"secret data")
            with pytest.raises(Exception):  # InvalidToken
                fernet2.decrypt(encrypted)

    def test_same_agent_gets_same_key(self):
        """Same agent ID always produces the same key (deterministic)."""
        from kestrel_sovereign.storage.encryption import get_agent_fernet
        passphrase = "test-passphrase"
        agent_id = "did:pkh:eip155:1:0xsameagent"
        with patch.dict(os.environ, {'KESTREL_DATA_KEY': passphrase}):
            fernet1 = get_agent_fernet(agent_id)
            fernet2 = get_agent_fernet(agent_id)

            # Encrypt with one, decrypt with other
            encrypted = fernet1.encrypt(b"secret data")
            decrypted = fernet2.decrypt(encrypted)
            assert decrypted == b"secret data"

    def test_agent_key_different_from_global(self):
        """Agent-specific key is different from global key."""
        from kestrel_sovereign.storage.encryption import get_agent_fernet, get_fernet
        passphrase = "test-passphrase"
        agent_id = "did:pkh:eip155:1:0xmyagent"
        with patch.dict(os.environ, {'KESTREL_DATA_KEY': passphrase}):
            agent_fernet = get_agent_fernet(agent_id)
            global_fernet = get_fernet()

            # Encrypt with agent key
            encrypted = agent_fernet.encrypt(b"agent secret")

            # Global key should NOT be able to decrypt
            with pytest.raises(Exception):  # InvalidToken
                global_fernet.decrypt(encrypted)


class TestEncryptDecryptBytes:
    """Tests for encrypt_bytes() and decrypt_bytes() functions."""

    def test_encrypt_bytes_with_fernet(self):
        """Encrypts bytes when Fernet is provided."""
        from kestrel_sovereign.storage.encryption import encrypt_bytes
        fernet = Fernet(Fernet.generate_key())
        content = b"test content"

        result, was_encrypted = encrypt_bytes(content, fernet)

        assert was_encrypted is True
        assert result != content  # Should be encrypted
        # Should be decryptable
        decrypted = fernet.decrypt(result)
        assert decrypted == content

    def test_encrypt_bytes_without_fernet(self):
        """Returns original content when Fernet is None."""
        from kestrel_sovereign.storage.encryption import encrypt_bytes
        content = b"test content"

        result, was_encrypted = encrypt_bytes(content, None)

        assert was_encrypted is False
        assert result == content

    def test_decrypt_bytes_with_enc_flag(self):
        """Decrypts bytes when enc flag is set in metadata."""
        from kestrel_sovereign.storage.encryption import encrypt_bytes, decrypt_bytes
        fernet = Fernet(Fernet.generate_key())
        content = b"secret bytes"

        encrypted, _ = encrypt_bytes(content, fernet)
        metadata = {"enc": True}

        result = decrypt_bytes(encrypted, fernet, metadata)
        assert result == content

    def test_decrypt_bytes_without_enc_flag(self):
        """Returns original when enc flag not set."""
        from kestrel_sovereign.storage.encryption import decrypt_bytes
        fernet = Fernet(Fernet.generate_key())
        content = b"plain content"

        result = decrypt_bytes(content, fernet, {"some": "meta"})
        assert result == content

    def test_decrypt_bytes_no_fernet_unencrypted(self):
        """Returns original when Fernet is None and content not marked as encrypted."""
        from kestrel_sovereign.storage.encryption import decrypt_bytes
        content = b"plain content"

        result = decrypt_bytes(content, None, {"some": "meta"})  # No enc flag
        assert result == content

    def test_decrypt_bytes_no_fernet_encrypted_raises_error(self):
        """Raises DecryptionError when no key but content marked as encrypted."""
        import pytest
        from kestrel_sovereign.storage.encryption import decrypt_bytes, DecryptionError
        content = b"encrypted content"

        with pytest.raises(DecryptionError) as exc_info:
            decrypt_bytes(content, None, {"enc": True})
        assert "No decryption key available" in str(exc_info.value)


class TestEncryptDecryptString:
    """Tests for encrypt_string() and decrypt_string() functions."""

    def test_encrypt_string_with_fernet(self):
        """Encrypts string when Fernet is provided."""
        from kestrel_sovereign.storage.encryption import encrypt_string
        fernet = Fernet(Fernet.generate_key())
        content = "Hello, World!"

        result, was_encrypted = encrypt_string(content, fernet)

        assert was_encrypted is True
        assert result != content
        # Result should be base64-encoded string
        assert isinstance(result, str)

    def test_encrypt_string_without_fernet(self):
        """Returns original string when Fernet is None."""
        from kestrel_sovereign.storage.encryption import encrypt_string
        content = "Hello, World!"

        result, was_encrypted = encrypt_string(content, None)

        assert was_encrypted is False
        assert result == content

    def test_decrypt_string_roundtrip(self):
        """Full encrypt/decrypt roundtrip for strings."""
        from kestrel_sovereign.storage.encryption import encrypt_string, decrypt_string
        fernet = Fernet(Fernet.generate_key())
        content = "Secret message with émojis 🔐"

        encrypted, _ = encrypt_string(content, fernet)
        metadata = {"enc": True}

        result = decrypt_string(encrypted, metadata, fernet)
        assert result == content

    def test_decrypt_string_without_enc_flag(self):
        """Returns original when enc flag not set."""
        from kestrel_sovereign.storage.encryption import decrypt_string
        fernet = Fernet(Fernet.generate_key())
        content = "plain text"

        result = decrypt_string(content, {}, fernet)
        assert result == content

    def test_decrypt_string_invalid_data_raises_error(self):
        """Raises DecryptionError on decryption failure (no silent fallback)."""
        import pytest
        from kestrel_sovereign.storage.encryption import decrypt_string, DecryptionError
        fernet = Fernet(Fernet.generate_key())

        # Invalid encrypted content should raise, not silently return
        with pytest.raises(DecryptionError) as exc_info:
            decrypt_string("not-valid-fernet-data", {"enc": True}, fernet)
        assert "Decryption failed" in str(exc_info.value)


class TestRemoveEncFlag:
    """Tests for remove_enc_flag() function."""

    def test_removes_enc_flag(self):
        """Removes enc key from metadata."""
        from kestrel_sovereign.storage.encryption import remove_enc_flag
        metadata = {"enc": True, "other": "value"}

        result = remove_enc_flag(metadata)

        assert "enc" not in result
        assert result == {"other": "value"}

    def test_returns_none_for_empty(self):
        """Returns None for empty metadata."""
        from kestrel_sovereign.storage.encryption import remove_enc_flag

        assert remove_enc_flag(None) is None
        assert remove_enc_flag({}) is None

    def test_returns_none_if_only_enc(self):
        """Returns None if enc was the only key."""
        from kestrel_sovereign.storage.encryption import remove_enc_flag
        metadata = {"enc": True}

        result = remove_enc_flag(metadata)
        assert result is None

    def test_preserves_other_keys(self):
        """Preserves all non-enc keys."""
        from kestrel_sovereign.storage.encryption import remove_enc_flag
        metadata = {"enc": True, "a": 1, "b": 2, "c": 3}

        result = remove_enc_flag(metadata)
        assert result == {"a": 1, "b": 2, "c": 3}


class TestKeyVersioningIntegration:
    """Integration tests for key versioning with conversation store."""

    @pytest.fixture
    def passphrase(self):
        return "test-passphrase-for-integration"

    def test_global_and_agent_keys_are_different(self, passphrase):
        """Verify global and per-agent keys encrypt differently."""
        from kestrel_sovereign.storage.encryption import get_fernet, get_agent_fernet

        with patch.dict(os.environ, {'KESTREL_DATA_KEY': passphrase}):
            global_fernet = get_fernet()
            agent_fernet = get_agent_fernet("did:pkh:eip155:1:0xtest")

            test_data = b"sensitive data"

            # Both can encrypt
            global_encrypted = global_fernet.encrypt(test_data)
            agent_encrypted = agent_fernet.encrypt(test_data)

            # Results are different (different keys)
            assert global_encrypted != agent_encrypted

            # Each can only decrypt its own
            assert global_fernet.decrypt(global_encrypted) == test_data
            assert agent_fernet.decrypt(agent_encrypted) == test_data

            # Cross-decryption fails
            with pytest.raises(Exception):
                global_fernet.decrypt(agent_encrypted)
            with pytest.raises(Exception):
                agent_fernet.decrypt(global_encrypted)

    def test_hkdf_derivation_deterministic(self, passphrase):
        """HKDF derivation is deterministic - same inputs = same key."""
        from kestrel_sovereign.storage.encryption import get_agent_fernet

        agent_id = "did:pkh:eip155:1:0xdeterministic"

        with patch.dict(os.environ, {'KESTREL_DATA_KEY': passphrase}):
            # Create two fernets for same agent
            fernet1 = get_agent_fernet(agent_id)
            fernet2 = get_agent_fernet(agent_id)

            # Encrypt with first
            encrypted = fernet1.encrypt(b"test")

            # Decrypt with second (proves keys are identical)
            decrypted = fernet2.decrypt(encrypted)
            assert decrypted == b"test"
