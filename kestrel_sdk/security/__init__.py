"""Kestrel SDK — Security interfaces and encryption helpers.

Requires the 'crypto' extra: pip install kestrel-sovereign-sdk[crypto]
"""

from .exceptions import (
    SecurityError,
    EncryptionError,
    DecryptionError,
    MasterKeyNotConfiguredError,
    InvalidPurposeError,
    KeyStorageError,
    KeyNotFoundError,
    KeyNotConfiguredError,
    PassphraseRequiredError,
)

__all__ = [
    "SecurityError",
    "EncryptionError",
    "DecryptionError",
    "MasterKeyNotConfiguredError",
    "InvalidPurposeError",
    "KeyStorageError",
    "KeyNotFoundError",
    "KeyNotConfiguredError",
    "PassphraseRequiredError",
]

# encryption submodule requires cryptography — import lazily
