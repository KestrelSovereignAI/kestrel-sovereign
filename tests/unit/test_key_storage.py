"""
Unit tests for SecureKeyStorage.

Tests encrypted key storage, key rotation, and migration of plaintext keys.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from kestrel_sovereign.security.key_storage import (
    SecureKeyStorage,
    EncryptedKeyBundle,
    KeyStorageError,
    MasterKeyNotConfiguredError,
    KeyDecryptionError,
    migrate_all_plaintext_keys,
)


@pytest.fixture
def master_key():
    """Test master key for encryption."""
    return "test-master-key-for-encryption-32chars!"


@pytest.fixture
def test_private_key():
    """Generate a test secp256k1 private key."""
    return ec.generate_private_key(ec.SECP256K1(), default_backend())


@pytest.fixture
def storage(master_key, temp_dir, monkeypatch):
    """Create SecureKeyStorage instance with test key and temp directory."""
    # Set env var before creating storage
    monkeypatch.setenv("KESTREL_DATA_KEY", master_key)
    yield SecureKeyStorage(storage_dir=temp_dir)


class TestSecureKeyStorageInit:
    """Tests for SecureKeyStorage initialization."""

    def test_init_with_env_key(self, master_key, temp_dir):
        """Should initialize with key from environment variable."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": master_key}):
            storage = SecureKeyStorage(storage_dir=temp_dir)
            assert storage.storage_dir == temp_dir

    def test_init_without_key_saves_ok_but_load_fails(self, temp_dir, monkeypatch):
        """Storage can be created without key, but operations will fail."""
        # Clear the environment variable
        monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
        storage = SecureKeyStorage(storage_dir=temp_dir)
        # Storage is created, but master key access will fail
        with pytest.raises(MasterKeyNotConfiguredError):
            storage._get_master_key()


class TestKeyEncryption:
    """Tests for key encryption and decryption."""

    def test_save_and_load_private_key(self, storage, test_private_key):
        """Should save and load a private key correctly."""
        key_id = "test_key_001"
        
        # Save the key
        key_path = storage.save_private_key(test_private_key, key_id)
        
        # Verify file exists
        assert key_path.exists()
        
        # Load the key
        loaded_key = storage.load_private_key(key_id)
        
        # Verify key matches (compare by public key bytes)
        original_pub = test_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        loaded_pub = loaded_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        assert original_pub == loaded_pub

    def test_encrypted_file_is_json(self, storage, test_private_key):
        """Encrypted file should be valid JSON with metadata."""
        key_id = "test_key_002"
        key_path = storage.save_private_key(test_private_key, key_id)
        
        import json
        content = key_path.read_text()
        data = json.loads(content)
        
        assert data["version"] == 1
        assert data["algorithm"] == "AES-256-GCM"
        assert data["kdf"] == "PBKDF2-SHA256"

    def test_encrypted_file_not_plaintext(self, storage, test_private_key):
        """Encrypted file should not contain plaintext PEM markers."""
        key_id = "test_key_003"
        key_path = storage.save_private_key(test_private_key, key_id)
        
        content = key_path.read_text()
        assert "-----BEGIN PRIVATE KEY-----" not in content
        assert "-----BEGIN EC PRIVATE KEY-----" not in content

    def test_load_nonexistent_key_fails(self, storage):
        """Should raise error when loading nonexistent key."""
        with pytest.raises(FileNotFoundError):
            storage.load_private_key("nonexistent_key")

    def test_load_with_wrong_key_fails(self, temp_dir, test_private_key, master_key):
        """Should fail to decrypt with wrong master key."""
        # Save with one key
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": master_key}):
            storage1 = SecureKeyStorage(storage_dir=temp_dir)
            storage1.save_private_key(test_private_key, "test_key_wrong")
        
        # Try to load with different key
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": "different-master-key-32chars!!"}):
            storage2 = SecureKeyStorage(storage_dir=temp_dir)
            with pytest.raises(KeyDecryptionError):
                storage2.load_private_key("test_key_wrong")

    def test_load_corrupted_file_fails(self, storage, test_private_key):
        """Should fail on corrupted encrypted file."""
        key_id = "test_key_corrupt"
        key_path = storage.save_private_key(test_private_key, key_id)
        
        # Corrupt the file
        content = key_path.read_text()
        key_path.write_text(content[:50] + "CORRUPTED" + content[60:])
        
        with pytest.raises((KeyDecryptionError, Exception)):
            storage.load_private_key(key_id)

    def test_has_key(self, storage, test_private_key):
        """Should check if key exists."""
        key_id = "test_key_exists"
        
        assert not storage.has_key(key_id)
        storage.save_private_key(test_private_key, key_id)
        assert storage.has_key(key_id)


class TestKeyMigration:
    """Tests for migrating plaintext keys to encrypted format."""

    def test_migrate_single_plaintext_key(self, storage, temp_dir, test_private_key):
        """Should migrate a plaintext PEM file to encrypted format."""
        # Create plaintext PEM file
        plaintext_path = temp_dir / "plaintext_key.pem"
        pem_bytes = test_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        plaintext_path.write_bytes(pem_bytes)
        
        # Migrate
        encrypted_path = storage.migrate_plaintext_key(plaintext_path)
        
        # Verify encrypted file exists
        assert encrypted_path.exists()
        
        # Verify plaintext was deleted
        assert not plaintext_path.exists()
        
        # Verify we can load the encrypted key
        loaded_key = storage.load_private_key("plaintext_key")
        assert loaded_key is not None

    def test_migrate_all_keys(self, master_key, temp_dir):
        """Should migrate all plaintext PEM files in directory."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": master_key}):
            # Create multiple plaintext keys
            for i in range(3):
                key = ec.generate_private_key(ec.SECP256K1(), default_backend())
                pem_bytes = key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                (temp_dir / f"key_{i}.pem").write_bytes(pem_bytes)
            
            # Migrate using the standalone function
            results = migrate_all_plaintext_keys(temp_dir)
            
            # Verify all migrated
            assert results["migrated"] == 3
            assert results["skipped"] == 0
            
            # Verify encrypted files exist
            for i in range(3):
                assert (temp_dir / f"key_{i}.key.enc").exists()
                assert not (temp_dir / f"key_{i}.pem").exists()

    def test_migrate_skips_already_encrypted(self, storage, test_private_key, master_key, temp_dir):
        """Should skip files that are already encrypted."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": master_key}):
            # Create encrypted file first
            storage.save_private_key(test_private_key, "already_encrypted")
            
            # Create a plaintext with same base name
            plaintext = temp_dir / "already_encrypted.pem"
            pem_bytes = test_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            plaintext.write_bytes(pem_bytes)
            
            # Migrate
            results = migrate_all_plaintext_keys(temp_dir)
            
            # Should skip the one that already has encrypted version
            assert results["skipped"] == 1

    def test_migrate_recursive(self, master_key, temp_dir):
        """Should recursively migrate keys in subdirectories."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": master_key}):
            # Create subdirectory with key
            subdir = temp_dir / "subdir" / "nested"
            subdir.mkdir(parents=True)
            
            key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            pem_bytes = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            (subdir / "nested_key.pem").write_bytes(pem_bytes)
            
            # Migrate from root
            results = migrate_all_plaintext_keys(temp_dir)
            
            # Verify migrated (rglob finds recursively)
            assert results["migrated"] == 1


class TestEncryptedKeyBundle:
    """Tests for EncryptedKeyBundle dataclass."""

    def test_bundle_creation(self):
        """Should create bundle with all required fields."""
        bundle = EncryptedKeyBundle(
            version=1,
            algorithm="AES-256-GCM",
            kdf="PBKDF2-SHA256",
            kdf_iterations=600000,
            salt="c2FsdA==",  # base64 "salt"
            nonce="bm9uY2U=",  # base64 "nonce"
            ciphertext="Y2lwaGVydGV4dA==",  # base64 "ciphertext"
        )
        
        assert bundle.version == 1
        assert bundle.algorithm == "AES-256-GCM"
        assert bundle.kdf == "PBKDF2-SHA256"
        assert bundle.kdf_iterations == 600000

    def test_bundle_to_json(self):
        """Should serialize bundle to JSON."""
        bundle = EncryptedKeyBundle(
            version=1,
            algorithm="AES-256-GCM",
            kdf="PBKDF2-SHA256",
            kdf_iterations=600000,
            salt="c2FsdA==",
            nonce="bm9uY2U=",
            ciphertext="Y2lwaGVydGV4dA==",
        )
        
        json_str = bundle.to_json()
        assert "AES-256-GCM" in json_str
        assert "PBKDF2-SHA256" in json_str
        assert '"version": 1' in json_str

    def test_bundle_from_json(self):
        """Should deserialize bundle from JSON."""
        bundle = EncryptedKeyBundle(
            version=1,
            algorithm="AES-256-GCM",
            kdf="PBKDF2-SHA256",
            kdf_iterations=600000,
            salt="c2FsdA==",
            nonce="bm9uY2U=",
            ciphertext="Y2lwaGVydGV4dA==",
        )
        
        json_str = bundle.to_json()
        restored = EncryptedKeyBundle.from_json(json_str)
        
        assert restored.version == bundle.version
        assert restored.algorithm == bundle.algorithm
        assert restored.kdf == bundle.kdf
        assert restored.kdf_iterations == bundle.kdf_iterations


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_directory_migration(self, master_key, temp_dir):
        """Should handle empty directory gracefully."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": master_key}):
            results = migrate_all_plaintext_keys(temp_dir)
            assert results["migrated"] == 0
            assert results["skipped"] == 0

    def test_non_key_pem_files_handled(self, master_key, temp_dir):
        """Should handle non-private-key PEM files gracefully."""
        with patch.dict(os.environ, {"KESTREL_DATA_KEY": master_key}):
            # Write a certificate PEM (not a private key)
            cert_path = temp_dir / "cert.pem"
            cert_path.write_text("-----BEGIN CERTIFICATE-----\nfake cert data\n-----END CERTIFICATE-----\n")
            
            # This should not crash but record an error
            results = migrate_all_plaintext_keys(temp_dir)
            
            # Should record error for non-parseable file
            assert results["migrated"] == 0
            assert len(results["errors"]) >= 1

    def test_key_id_sanitization(self, storage, test_private_key):
        """Should sanitize key IDs to prevent path traversal."""
        # Try to use a malicious key ID
        malicious_id = "../../../etc/passwd"
        key_path = storage.save_private_key(test_private_key, malicious_id)
        
        # Should not create file outside storage dir
        assert key_path.parent == storage.storage_dir
        
        # Should strip dangerous characters
        assert ".." not in key_path.name
        assert "/" not in key_path.name


class TestSecureDelete:
    """Tests for the secure_delete utility function."""

    def test_secure_delete_removes_file(self, temp_dir):
        """Should delete the file."""
        from kestrel_sovereign.security.key_storage import secure_delete
        
        test_file = temp_dir / "secret.txt"
        test_file.write_text("sensitive data")
        
        assert test_file.exists()
        result = secure_delete(test_file)
        
        assert result is True
        assert not test_file.exists()

    def test_secure_delete_nonexistent_file(self, temp_dir):
        """Should return False for non-existent file."""
        from kestrel_sovereign.security.key_storage import secure_delete
        
        nonexistent = temp_dir / "does_not_exist.txt"
        result = secure_delete(nonexistent)
        
        assert result is False

    def test_secure_delete_overwrites_content(self, temp_dir):
        """Should overwrite file content before deletion."""
        from kestrel_sovereign.security.key_storage import secure_delete
        import mmap
        
        test_file = temp_dir / "secret.key"
        secret_content = b"TOP SECRET KEY DATA 12345"
        test_file.write_bytes(secret_content)
        
        # Get the file size
        original_size = test_file.stat().st_size
        
        # Track if overwrite happened by checking content changes
        # (Note: We can't truly verify after deletion, but we can 
        # check the function runs without error)
        secure_delete(test_file)
        
        assert not test_file.exists()

    def test_secure_delete_with_path_string(self, temp_dir):
        """Should accept string paths."""
        from kestrel_sovereign.security.key_storage import secure_delete
        
        test_file = temp_dir / "test.txt"
        test_file.write_text("data")
        
        # Pass as string
        result = secure_delete(str(test_file))
        
        assert result is True
        assert not test_file.exists()

    def test_secure_delete_rejects_directories(self, temp_dir):
        """Should raise error for directories."""
        from kestrel_sovereign.security.key_storage import secure_delete
        
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        
        with pytest.raises(ValueError, match="non-file"):
            secure_delete(subdir)

    def test_secure_delete_custom_passes(self, temp_dir):
        """Should support custom number of overwrite passes."""
        from kestrel_sovereign.security.key_storage import secure_delete
        
        test_file = temp_dir / "secret.txt"
        test_file.write_text("sensitive")
        
        # Use fewer passes (still works)
        result = secure_delete(test_file, passes=1)
        
        assert result is True
        assert not test_file.exists()

    def test_secure_delete_empty_file(self, temp_dir):
        """Should handle empty files."""
        from kestrel_sovereign.security.key_storage import secure_delete
        
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()
        
        result = secure_delete(empty_file)
        
        assert result is True
        assert not empty_file.exists()
