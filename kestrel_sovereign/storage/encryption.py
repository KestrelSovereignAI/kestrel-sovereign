"""
Backward-compatible encryption re-exports.

This module re-exports from kestrel_sovereign.security.encryption for backward compatibility.
New code should import directly from kestrel_sovereign.security.encryption.

Deprecated:
    Import from kestrel_sovereign.security.encryption instead:

    # Old (deprecated)
    from kestrel_sovereign.storage.encryption import get_fernet, DecryptionError

    # New (recommended)
    from kestrel_sovereign.security.encryption import get_fernet, DecryptionError
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
    # Key loading (internal, but exposed for key_rotation)
    _get_data_key,
    _read_key_from_file,
    # Fernet-based encryption
    get_fernet,
    get_master_key_bytes,
    get_agent_fernet,
    # Purpose-specific encryption
    get_agent_key,
    encrypt,
    decrypt,
    encrypt_string,
    decrypt_string,
    # Legacy Fernet helpers
    encrypt_bytes,
    decrypt_bytes,
    remove_enc_flag,
)

# Backward-compatible aliases for storage layer
# The storage layer uses these names with different signatures
encrypt_string_storage = encrypt_string  # Alias for clarity

# Re-export the Fernet-based string functions with original names
# These have different signatures than purpose-specific versions
from kestrel_sovereign.security.encryption import (
    encrypt_string_fernet as _encrypt_string_fernet,
    decrypt_string_fernet as _decrypt_string_fernet,
)


def encrypt_string(content, fernet=None, agent_did=None, purpose=None):
    """
    Encrypt string - supports both old and new APIs.

    Old API (storage layer):
        encrypt_string(content, fernet) -> (encrypted, was_encrypted)

    New API (purpose-specific):
        encrypt_string(content, agent_did=did, purpose=purpose) -> bytes
    """
    if agent_did is not None and purpose is not None:
        # New API: purpose-specific encryption
        from kestrel_sovereign.security.encryption import encrypt_string as _encrypt_str
        return _encrypt_str(agent_did, purpose, content)
    else:
        # Old API: Fernet-based with tuple return
        return _encrypt_string_fernet(content, fernet)


def decrypt_string(content, metadata=None, fernet=None, agent_did=None, purpose=None):
    """
    Decrypt string - supports both old and new APIs.

    Old API (storage layer):
        decrypt_string(content, metadata, fernet) -> str

    New API (purpose-specific):
        decrypt_string(content, agent_did=did, purpose=purpose) -> str
    """
    if agent_did is not None and purpose is not None:
        # New API: purpose-specific decryption
        from kestrel_sovereign.security.encryption import decrypt_string as _decrypt_str
        return _decrypt_str(agent_did, purpose, content)
    else:
        # Old API: Fernet-based with metadata
        return _decrypt_string_fernet(content, metadata, fernet)


__all__ = [
    # Exceptions
    "EncryptionError",
    "DecryptionError",
    "MasterKeyNotConfiguredError",
    "InvalidPurposeError",
    # Constants
    "VALID_PURPOSES",
    # Fernet functions
    "get_fernet",
    "get_master_key_bytes",
    "get_agent_fernet",
    # Purpose-specific
    "get_agent_key",
    "encrypt",
    "decrypt",
    "encrypt_string",
    "decrypt_string",
    # Legacy helpers
    "encrypt_bytes",
    "decrypt_bytes",
    "remove_enc_flag",
    # Internal (for key_rotation)
    "_get_data_key",
    "_read_key_from_file",
]
