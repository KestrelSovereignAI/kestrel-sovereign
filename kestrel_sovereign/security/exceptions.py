"""
Unified exception hierarchy for Kestrel security module.

All key storage and encryption exceptions inherit from these base classes.
This eliminates duplicate exception definitions across modules.

Usage:
    from kestrel_sovereign.security.exceptions import (
        KeyStorageError,
        KeyNotFoundError,
        DecryptionError,
        MasterKeyNotConfiguredError,
    )

    try:
        key = storage.get_key("openai")
    except KeyNotFoundError:
        logger.warning("No key configured")
    except DecryptionError:
        logger.error("Wrong encryption key")
"""


class SecurityError(Exception):
    """Base exception for all security module errors."""
    pass


# =============================================================================
# Key Storage Exceptions
# =============================================================================

class KeyStorageError(SecurityError):
    """Base exception for key storage operations."""
    pass


class KeyNotFoundError(KeyStorageError):
    """Raised when a requested key doesn't exist."""
    pass


class KeyNotConfiguredError(KeyStorageError):
    """Raised when no key is configured for a provider."""
    pass


# =============================================================================
# Encryption Exceptions
# =============================================================================

class EncryptionError(SecurityError):
    """Base exception for encryption operations."""
    pass


class DecryptionError(EncryptionError):
    """Raised when decryption fails (wrong key, corrupted data, or wrong passphrase)."""
    pass


class MasterKeyNotConfiguredError(EncryptionError):
    """Raised when master encryption key is not set (KESTREL_DATA_KEY or PLATFORM_KEY_MASTER)."""
    pass


class InvalidPurposeError(EncryptionError):
    """Raised when an invalid encryption purpose is specified."""
    pass


# =============================================================================
# User Key Exceptions
# =============================================================================

class PassphraseRequiredError(KeyStorageError):
    """Raised when passphrase is required but not provided."""
    pass


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# These aliases maintain backward compatibility with existing code
# that catches specific exception names from different modules.

# From key_storage.py
KeyDecryptionError = DecryptionError

# From platform_key_storage.py
PlatformKeyStorageError = KeyStorageError

# From user_key_storage.py
UserKeyStorageError = KeyStorageError
