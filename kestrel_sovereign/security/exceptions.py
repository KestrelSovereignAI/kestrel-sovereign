"""
Unified exception hierarchy for Kestrel security module.

Re-exports from kestrel_sdk.security.exceptions for backward compatibility.
Feature packages should import from kestrel_sdk.security.exceptions directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.security.exceptions import (  # noqa: F401
    SecurityError,
    KeyStorageError,
    KeyNotFoundError,
    KeyNotConfiguredError,
    EncryptionError,
    DecryptionError,
    MasterKeyNotConfiguredError,
    InvalidPurposeError,
    PassphraseRequiredError,
    KeyDecryptionError,
    PlatformKeyStorageError,
    UserKeyStorageError,
)
