"""
Tests for decryption failure behavior.

Ensures that decryption failures raise explicit errors rather than
silently returning encrypted content (which is a security anti-pattern).
"""
import os
import pytest
from unittest.mock import patch

from kestrel_sovereign.storage.encryption import (
    decrypt_string,
    decrypt_bytes,
    encrypt_string,
    encrypt_bytes,
    get_fernet,
    DecryptionError,
)


class TestDecryptionFailure:
    """Tests that decryption properly fails with wrong key."""

    def test_decrypt_string_wrong_key_raises_error(self):
        """Decrypting string with wrong key should raise DecryptionError."""
        # Set up correct key and encrypt
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "correct-key-for-test"}):
            fernet_correct = get_fernet()
            content = "sensitive data that must be protected"
            encrypted, was_encrypted = encrypt_string(content, fernet_correct)
            assert was_encrypted
            metadata = {"enc": True}

        # Try to decrypt with wrong key
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "wrong-key-for-test"}):
            fernet_wrong = get_fernet()
            with pytest.raises(DecryptionError) as exc_info:
                decrypt_string(encrypted, metadata, fernet_wrong)

            assert "decryption failed" in str(exc_info.value).lower()

    def test_decrypt_bytes_wrong_key_raises_error(self):
        """Decrypting bytes with wrong key should raise DecryptionError."""
        # Set up correct key and encrypt
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "correct-key-for-test"}):
            fernet_correct = get_fernet()
            content = b"sensitive binary data"
            encrypted, was_encrypted = encrypt_bytes(content, fernet_correct)
            assert was_encrypted
            metadata = {"enc": True}

        # Try to decrypt with wrong key
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "wrong-key-for-test"}):
            fernet_wrong = get_fernet()
            with pytest.raises(DecryptionError) as exc_info:
                decrypt_bytes(encrypted, fernet_wrong, metadata)

            assert "decryption failed" in str(exc_info.value).lower()

    def test_decrypt_string_no_key_but_encrypted_raises_error(self):
        """Decrypting encrypted content without any key should raise DecryptionError."""
        encrypted_content = "gAAAAABfake_encrypted_content"
        metadata = {"enc": True}

        with pytest.raises(DecryptionError) as exc_info:
            decrypt_string(encrypted_content, metadata, None)

        assert "No decryption key available" in str(exc_info.value)

    def test_decrypt_bytes_no_key_but_encrypted_raises_error(self):
        """Decrypting encrypted bytes without any key should raise DecryptionError."""
        encrypted_content = b"gAAAAABfake_encrypted_content"
        metadata = {"enc": True}

        with pytest.raises(DecryptionError) as exc_info:
            decrypt_bytes(encrypted_content, None, metadata)

        assert "No decryption key available" in str(exc_info.value)

    def test_decrypt_string_unencrypted_returns_content(self):
        """Content not marked as encrypted should be returned as-is."""
        content = "plain text content"
        metadata = {}  # No enc flag

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "any-key"}):
            fernet = get_fernet()
            result = decrypt_string(content, metadata, fernet)
            assert result == content

    def test_decrypt_bytes_unencrypted_returns_content(self):
        """Bytes not marked as encrypted should be returned as-is."""
        content = b"plain bytes content"
        metadata = {}  # No enc flag

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "any-key"}):
            fernet = get_fernet()
            result = decrypt_bytes(content, fernet, metadata)
            assert result == content

    def test_decrypt_string_correct_key_succeeds(self):
        """Decrypting with correct key should succeed."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "correct-key-for-test"}):
            fernet = get_fernet()
            original = "secret message"
            encrypted, _ = encrypt_string(original, fernet)
            metadata = {"enc": True}

            decrypted = decrypt_string(encrypted, metadata, fernet)
            assert decrypted == original

    def test_decrypt_bytes_correct_key_succeeds(self):
        """Decrypting bytes with correct key should succeed."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "correct-key-for-test"}):
            fernet = get_fernet()
            original = b"secret binary data"
            encrypted, _ = encrypt_bytes(original, fernet)
            metadata = {"enc": True}

            decrypted = decrypt_bytes(encrypted, fernet, metadata)
            assert decrypted == original


class TestDecryptionErrorMessage:
    """Tests that error messages are helpful for debugging."""

    def test_error_includes_context(self):
        """DecryptionError should include helpful context."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "key1"}):
            fernet1 = get_fernet()
            encrypted, _ = encrypt_string("data", fernet1)

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "key2"}):
            fernet2 = get_fernet()
            try:
                decrypt_string(encrypted, {"enc": True}, fernet2)
                pytest.fail("Should have raised DecryptionError")
            except DecryptionError as e:
                # Error should mention what went wrong
                error_msg = str(e).lower()
                assert "decrypt" in error_msg or "wrong key" in error_msg or "failed" in error_msg


class TestNoSilentFailures:
    """Tests ensuring no silent failures remain."""

    def test_encrypted_content_never_returned_silently(self):
        """Encrypted content should never be silently returned on failure."""
        # This is the key test - we want to ensure the old behavior
        # (returning encrypted content as-is) is gone

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "key1"}):
            fernet1 = get_fernet()
            original = "this is sensitive data"
            encrypted, _ = encrypt_string(original, fernet1)

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "key2"}):
            fernet2 = get_fernet()

            # OLD BEHAVIOR (BAD): Would return encrypted as-is
            # NEW BEHAVIOR (GOOD): Should raise DecryptionError

            with pytest.raises(DecryptionError):
                result = decrypt_string(encrypted, {"enc": True}, fernet2)
                # If we get here, we've failed - check it's not encrypted content
                assert result != encrypted, "Silent failure: encrypted content returned!"
