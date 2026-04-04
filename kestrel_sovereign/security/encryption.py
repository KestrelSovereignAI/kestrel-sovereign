"""
Unified encryption utilities for Kestrel.

Re-exports from kestrel_sdk.security.encryption for backward compatibility.
Feature packages should import from kestrel_sdk.security.encryption directly.

Usage:
    from kestrel_sovereign.security.encryption import (
        get_fernet,           # Global Fernet (backward compatible)
        get_agent_fernet,     # Per-agent Fernet
        encrypt, decrypt,     # Purpose-specific encryption
        DecryptionError,
    )
"""

# Re-export exceptions from SDK
from kestrel_sdk.security.exceptions import (  # noqa: F401
    EncryptionError,
    DecryptionError,
    MasterKeyNotConfiguredError,
    InvalidPurposeError,
)

# Re-export everything from SDK encryption module
from kestrel_sdk.security.encryption import (  # noqa: F401
    ENV_VAR_NAME,
    KEY_SIZE,
    NONCE_SIZE,
    VALID_PURPOSES,
    # Key loading (internal, but exposed for storage/encryption.py compatibility)
    _read_key_from_file,
    _strip_quotes,
    _get_data_key,
    _get_master_key,
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
    encrypt_string_fernet,
    decrypt_string_fernet,
    remove_enc_flag,
)
