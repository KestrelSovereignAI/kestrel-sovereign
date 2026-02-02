"""
Security module for Kestrel Agent.

Provides secure key storage, encryption utilities, and authentication.
"""

from .key_storage import (
    SecureKeyStorage,
    EncryptedKeyBundle,
    KeyStorageError,
    MasterKeyNotConfiguredError,
    KeyDecryptionError,
    migrate_all_plaintext_keys,
)

__all__ = [
    "SecureKeyStorage",
    "EncryptedKeyBundle",
    "KeyStorageError",
    "MasterKeyNotConfiguredError",
    "KeyDecryptionError",
    "migrate_all_plaintext_keys",
]
