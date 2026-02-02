"""
Backward-compatible agent encryption re-exports.

This module re-exports from kestrel_sovereign.security.encryption for backward compatibility.
New code should import directly from kestrel_sovereign.security.encryption.

Deprecated:
    Import from kestrel_sovereign.security.encryption instead:

    # Old (deprecated)
    from kestrel_sovereign.security.agent_encryption import encrypt, decrypt

    # New (recommended)
    from kestrel_sovereign.security.encryption import encrypt, decrypt
"""
# Re-export everything from the unified module
from kestrel_sovereign.security.encryption import (
    # Exception classes
    EncryptionError,
    DecryptionError,
    MasterKeyNotConfiguredError,
    InvalidPurposeError,
    # Constants
    VALID_PURPOSES,
    KEY_SIZE,
    # Purpose-specific encryption (the main API for this module)
    get_agent_key,
    encrypt,
    decrypt,
    encrypt_string,
    decrypt_string,
)

# Legacy constants
NONCE_SIZE = 12  # Was used for AES-GCM, kept for compatibility
ENV_VAR_NAME = "KESTREL_DATA_KEY"

__all__ = [
    # Exceptions
    "EncryptionError",
    "DecryptionError",
    "MasterKeyNotConfiguredError",
    "InvalidPurposeError",
    # Constants
    "VALID_PURPOSES",
    "KEY_SIZE",
    "NONCE_SIZE",
    "ENV_VAR_NAME",
    # Functions
    "get_agent_key",
    "encrypt",
    "decrypt",
    "encrypt_string",
    "decrypt_string",
]
