"""
Secure Key Storage for Kestrel Agent.

This module provides encrypted storage for private keys using
the KESTREL_DATA_KEY environment variable as the master key.

Security Model:
- Private keys are encrypted with AES-256-GCM (AEAD)
- Master key is derived from KESTREL_DATA_KEY using PBKDF2
- Each key has a unique salt for key derivation
- Nonce is randomly generated per encryption

Usage:
    storage = SecureKeyStorage()
    
    # Save a private key (encrypts automatically)
    storage.save_private_key(private_key_obj, "agent_0x123")
    
    # Load a private key (decrypts automatically)
    private_key = storage.load_private_key("agent_0x123")
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, NoEncryption,
    load_pem_private_key
)
from cryptography.hazmat.backends import default_backend

from kestrel_sovereign.security.exceptions import (
    KeyStorageError,
    MasterKeyNotConfiguredError,
    DecryptionError as KeyDecryptionError,  # Alias for backward compat
)

logger = logging.getLogger(__name__)

# Constants
SALT_SIZE = 16  # 128 bits
NONCE_SIZE = 12  # 96 bits for GCM
KEY_SIZE = 32   # 256 bits
PBKDF2_ITERATIONS = 600_000  # OWASP recommendation for 2023+
SECURE_DELETE_PASSES = 3  # Number of overwrite passes for secure deletion


def secure_delete(path: Union[str, Path], passes: int = SECURE_DELETE_PASSES) -> bool:
    """
    Securely delete a file by overwriting with random data before unlinking.
    
    This makes recovery of the original data more difficult, though it cannot
    guarantee complete erasure on all storage types (e.g., SSDs with wear leveling,
    copy-on-write filesystems). For maximum security, use encrypted storage.
    
    Args:
        path: Path to the file to delete
        passes: Number of overwrite passes (default: 3)
        
    Returns:
        True if file was successfully deleted, False if file didn't exist
        
    Raises:
        OSError: If file operations fail
    """
    path = Path(path)
    
    if not path.exists():
        return False
    
    if not path.is_file():
        raise ValueError(f"Cannot securely delete non-file: {path}")
    
    try:
        file_size = path.stat().st_size
        
        # Overwrite with random data multiple times
        for pass_num in range(passes):
            with open(path, 'wb') as f:
                f.write(os.urandom(max(file_size, 1)))
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
        
        # Final overwrite with zeros
        with open(path, 'wb') as f:
            f.write(b'\x00' * max(file_size, 1))
            f.flush()
            os.fsync(f.fileno())
        
        # Now unlink
        path.unlink()
        logger.debug(f"Securely deleted: {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to securely delete {path}: {e}")
        # Try regular delete as fallback
        try:
            path.unlink()
            logger.warning(f"Fell back to regular delete for {path}")
            return True
        except Exception:
            raise


# Exception classes imported from kestrel_sovereign.security.exceptions
# KeyStorageError, MasterKeyNotConfiguredError, KeyDecryptionError are now
# available from the import above


@dataclass
class EncryptedKeyBundle:
    """
    Container for an encrypted private key with all metadata needed for decryption.
    Stored as JSON for easy serialization.
    """
    version: int  # Schema version for future compatibility
    algorithm: str  # "AES-256-GCM"
    kdf: str  # "PBKDF2-SHA256"
    kdf_iterations: int
    salt: str  # Base64 encoded
    nonce: str  # Base64 encoded
    ciphertext: str  # Base64 encoded
    
    def to_json(self) -> str:
        return json.dumps({
            "version": self.version,
            "algorithm": self.algorithm,
            "kdf": self.kdf,
            "kdf_iterations": self.kdf_iterations,
            "salt": self.salt,
            "nonce": self.nonce,
            "ciphertext": self.ciphertext
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "EncryptedKeyBundle":
        data = json.loads(json_str)
        return cls(
            version=data["version"],
            algorithm=data["algorithm"],
            kdf=data["kdf"],
            kdf_iterations=data["kdf_iterations"],
            salt=data["salt"],
            nonce=data["nonce"],
            ciphertext=data["ciphertext"]
        )


class SecureKeyStorage:
    """
    Provides encrypted storage for private keys.
    
    Keys are encrypted with AES-256-GCM using a key derived from
    the KESTREL_DATA_KEY environment variable.
    """
    
    ENV_VAR_NAME = "KESTREL_DATA_KEY"
    ENCRYPTED_EXTENSION = ".key.enc"
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize the secure key storage.
        
        Args:
            storage_dir: Directory for key storage. Defaults to agent_data/
        """
        if storage_dir is None:
            try:
                from kestrel_sovereign.storage import get_default_agent_data_dir
                storage_dir = Path(get_default_agent_data_dir())
            except ImportError:
                storage_dir = Path("agent_data")
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_master_key(self) -> bytes:
        """
        Get the master key from environment.
        
        Returns:
            The master key bytes
            
        Raises:
            MasterKeyNotConfiguredError: If KESTREL_DATA_KEY is not set
        """
        master_key = os.environ.get(self.ENV_VAR_NAME)
        if not master_key:
            raise MasterKeyNotConfiguredError(
                f"{self.ENV_VAR_NAME} environment variable is not set. "
                "This is required for secure key storage. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        return master_key.encode('utf-8')
    
    def _derive_key(self, master_key: bytes, salt: bytes) -> bytes:
        """
        Derive an encryption key from the master key using PBKDF2.
        
        Args:
            master_key: The master key from environment
            salt: Random salt for this derivation
            
        Returns:
            32-byte derived key for AES-256
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
            backend=default_backend()
        )
        return kdf.derive(master_key)
    
    def _encrypt_key(self, private_key_pem: bytes) -> EncryptedKeyBundle:
        """
        Encrypt a private key.
        
        Args:
            private_key_pem: PEM-encoded private key
            
        Returns:
            EncryptedKeyBundle with all encryption metadata
        """
        master_key = self._get_master_key()
        
        # Generate random salt and nonce
        salt = os.urandom(SALT_SIZE)
        nonce = os.urandom(NONCE_SIZE)
        
        # Derive encryption key
        derived_key = self._derive_key(master_key, salt)
        
        # Encrypt with AES-GCM
        aesgcm = AESGCM(derived_key)
        ciphertext = aesgcm.encrypt(nonce, private_key_pem, None)
        
        return EncryptedKeyBundle(
            version=1,
            algorithm="AES-256-GCM",
            kdf="PBKDF2-SHA256",
            kdf_iterations=PBKDF2_ITERATIONS,
            salt=base64.b64encode(salt).decode('ascii'),
            nonce=base64.b64encode(nonce).decode('ascii'),
            ciphertext=base64.b64encode(ciphertext).decode('ascii')
        )
    
    def _decrypt_key(self, bundle: EncryptedKeyBundle) -> bytes:
        """
        Decrypt a private key bundle.
        
        Args:
            bundle: The encrypted key bundle
            
        Returns:
            PEM-encoded private key bytes
            
        Raises:
            KeyDecryptionError: If decryption fails
        """
        master_key = self._get_master_key()
        
        # Decode from base64
        salt = base64.b64decode(bundle.salt)
        nonce = base64.b64decode(bundle.nonce)
        ciphertext = base64.b64decode(bundle.ciphertext)
        
        # Derive the same key using stored salt and iterations
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=bundle.kdf_iterations,
            backend=default_backend()
        )
        derived_key = kdf.derive(master_key)
        
        # Decrypt
        try:
            aesgcm = AESGCM(derived_key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext
        except Exception as e:
            raise KeyDecryptionError(
                f"Failed to decrypt key: {e}. "
                "This usually means the master key (KESTREL_DATA_KEY) is incorrect "
                "or the encrypted key file is corrupted."
            ) from e
    
    def _get_key_path(self, key_id: str) -> Path:
        """Get the file path for a key ID."""
        # Sanitize key_id to prevent path traversal
        safe_id = "".join(c for c in key_id if c.isalnum() or c in "-_")
        return self.storage_dir / f"{safe_id}{self.ENCRYPTED_EXTENSION}"
    
    def save_private_key(self, private_key, key_id: str) -> Path:
        """
        Save a private key securely.
        
        Args:
            private_key: The cryptography private key object
            key_id: Identifier for the key (e.g., "kestrel_0x123...")
            
        Returns:
            Path to the saved encrypted key file
        """
        # Serialize to PEM (unencrypted in memory)
        private_pem = private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption()
        )
        
        # Encrypt
        bundle = self._encrypt_key(private_pem)
        
        # Save
        key_path = self._get_key_path(key_id)
        with open(key_path, 'w', encoding='utf-8') as f:
            f.write(bundle.to_json())
        
        logger.info(f"Saved encrypted private key: {key_path}")
        return key_path
    
    def load_private_key(self, key_id: str):
        """
        Load and decrypt a private key.
        
        Args:
            key_id: Identifier for the key
            
        Returns:
            The cryptography private key object
            
        Raises:
            FileNotFoundError: If the key file doesn't exist
            KeyDecryptionError: If decryption fails
        """
        key_path = self._get_key_path(key_id)
        
        if not key_path.exists():
            raise FileNotFoundError(f"Encrypted key not found: {key_path}")
        
        with open(key_path, 'r', encoding='utf-8') as f:
            bundle = EncryptedKeyBundle.from_json(f.read())
        
        private_pem = self._decrypt_key(bundle)
        
        private_key = load_pem_private_key(private_pem, password=None, backend=default_backend())
        
        logger.info(f"Loaded encrypted private key: {key_path}")
        return private_key
    
    def has_key(self, key_id: str) -> bool:
        """Check if an encrypted key exists."""
        return self._get_key_path(key_id).exists()
    
    def migrate_plaintext_key(self, pem_path: Path, key_id: Optional[str] = None) -> Path:
        """
        Migrate a plaintext PEM file to encrypted storage.
        
        Args:
            pem_path: Path to the plaintext PEM file
            key_id: Optional key ID (defaults to filename without extension)
            
        Returns:
            Path to the new encrypted key file
        """
        pem_path = Path(pem_path)
        
        if not pem_path.exists():
            raise FileNotFoundError(f"PEM file not found: {pem_path}")
        
        if key_id is None:
            key_id = pem_path.stem
        
        # Load the plaintext key
        with open(pem_path, 'rb') as f:
            private_pem = f.read()
        
        private_key = load_pem_private_key(private_pem, password=None, backend=default_backend())
        
        # Save encrypted
        encrypted_path = self.save_private_key(private_key, key_id)
        
        # Securely delete the plaintext file using the secure_delete utility
        secure_delete(pem_path)
        
        logger.info(f"Migrated and securely deleted: {pem_path} -> {encrypted_path}")
        return encrypted_path

def migrate_all_plaintext_keys(storage_dir: Optional[Path] = None) -> dict:
    """
    Migrate all plaintext PEM files in the storage directory to encrypted format.
    
    Args:
        storage_dir: Directory to scan. Defaults to agent_data/
        
    Returns:
        Dict with migration statistics
    """
    storage = SecureKeyStorage(storage_dir)
    storage_path = storage.storage_dir
    
    stats = {"migrated": 0, "skipped": 0, "errors": []}
    
    # Find all .pem files
    for pem_file in storage_path.rglob("*.pem"):
        key_id = pem_file.stem
        
        # Skip if already migrated
        if storage.has_key(key_id):
            logger.info(f"Skipping already migrated: {key_id}")
            stats["skipped"] += 1
            continue
        
        try:
            storage.migrate_plaintext_key(pem_file, key_id)
            stats["migrated"] += 1
        except Exception as e:
            logger.error(f"Failed to migrate {pem_file}: {e}")
            stats["errors"].append({"file": str(pem_file), "error": str(e)})
    
    return stats
