"""
Security module for Kestrel Agent.

Provides secure key storage, encryption utilities, authentication,
and prompt injection guardrails.
"""

from .key_storage import (
    SecureKeyStorage,
    EncryptedKeyBundle,
    KeyStorageError,
    MasterKeyNotConfiguredError,
    KeyDecryptionError,
    migrate_all_plaintext_keys,
)
from .input_guardrails import (
    wrap_user_input,
    check_prompt_injection,
    validate_tool_arguments,
    ANTI_INJECTION_SYSTEM_PROMPT,
    TOOL_HONESTY_SYSTEM_PROMPT,
    append_security_addendum,
)

__all__ = [
    "SecureKeyStorage",
    "EncryptedKeyBundle",
    "KeyStorageError",
    "MasterKeyNotConfiguredError",
    "KeyDecryptionError",
    "migrate_all_plaintext_keys",
    "wrap_user_input",
    "check_prompt_injection",
    "validate_tool_arguments",
    "ANTI_INJECTION_SYSTEM_PROMPT",
    "TOOL_HONESTY_SYSTEM_PROMPT",
    "append_security_addendum",
]
