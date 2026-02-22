"""
Unified encryption utilities for Kestrel.

Provides Fernet-based encryption at rest with:
- Per-agent key derivation (each agent gets unique keys)
- Purpose-specific subkeys (conversations, service-keys, wallet, backup)
- Multiple key sources (env var, Docker Secrets, file paths)
- Explicit error handling (no silent failures)

Key Hierarchy:
    KESTREL_DATA_KEY (env var or secrets file)
        ↓ (SHA-256 if passphrase)
    Master Key (32-byte Fernet key)
        ↓ (HKDF with agent DID as salt)
    Agent Key
        ↓ (HKDF with purpose as info)
        ├── kestrel-conversations-v1
        ├── kestrel-service-keys-v1
        ├── kestrel-wallet-v1
        └── kestrel-backup-v1

Usage:
    from kestrel_sovereign.security.encryption import (
        get_fernet,           # Global Fernet (backward compatible)
        get_agent_fernet,     # Per-agent Fernet
        encrypt, decrypt,     # Purpose-specific encryption
        DecryptionError,
    )

    # Simple encryption (global key)
    fernet = get_fernet()
    ciphertext = fernet.encrypt(b"data")

    # Per-agent encryption (recommended)
    agent_fernet = get_agent_fernet("did:pkh:eip155:1:0x...")
    ciphertext = agent_fernet.encrypt(b"data")

    # Purpose-specific encryption
    ciphertext = encrypt("did:...", "service-keys", api_key.encode())
    plaintext = decrypt("did:...", "service-keys", ciphertext)

Error Handling:
    DecryptionError is raised when decryption fails due to wrong key.
    Callers should handle this explicitly - silent failures are not allowed.
"""
import hashlib
import base64
import logging
import os
from typing import Optional, Dict, Tuple, Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

logger = logging.getLogger(__name__)


# =============================================================================
# Exception Classes (imported from shared module)
# =============================================================================

from kestrel_sovereign.security.exceptions import (
    EncryptionError,
    DecryptionError,
    MasterKeyNotConfiguredError,
    InvalidPurposeError,
)


# =============================================================================
# Constants
# =============================================================================

ENV_VAR_NAME = "KESTREL_DATA_KEY"
KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 12  # 96 bits (legacy AES-GCM, kept for compatibility)

# Valid purposes for purpose-specific encryption
VALID_PURPOSES = frozenset([
    "conversations",
    "service-keys",
    "wallet",
    "backup",
])


# =============================================================================
# Key Loading
# =============================================================================

def _read_key_from_file(path: str) -> Optional[str]:
    """Read a key from a secrets file (Docker Secrets, Kubernetes Secrets, etc.)."""
    try:
        if os.path.isfile(path):
            with open(path, 'r') as f:
                return f.read().strip()
    except Exception as e:
        logger.warning(f"Could not read key from {path}: {e}")
    return None


def _strip_quotes(value: str) -> str:
    """Strip surrounding quotes from a value.

    Docker's --env-file includes quotes literally, while python-dotenv strips them.
    This ensures consistent key values regardless of how env vars are loaded.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _get_data_key() -> Optional[str]:
    """
    Get the KESTREL_DATA_KEY from multiple sources in order of preference:

    1. KESTREL_DATA_KEY_FILE - Path to a secrets file (Docker/K8s secrets)
    2. /run/secrets/kestrel_data_key - Default Docker Secrets location
    3. KESTREL_DATA_KEY - Environment variable (legacy, less secure)

    Using file-based secrets prevents key exposure via `docker inspect`.
    """
    # Priority 1: Explicit file path
    key_file = os.environ.get("KESTREL_DATA_KEY_FILE")
    if key_file:
        key = _read_key_from_file(key_file)
        if key:
            logger.debug("Loaded key from KESTREL_DATA_KEY_FILE")
            return _strip_quotes(key)

    # Priority 2: Default Docker Secrets path
    docker_secret_path = "/run/secrets/kestrel_data_key"
    if os.path.exists(docker_secret_path):
        key = _read_key_from_file(docker_secret_path)
        if key:
            logger.debug("Loaded key from Docker Secret")
            return _strip_quotes(key)

    # Priority 3: Environment variable (legacy)
    env_key = os.environ.get(ENV_VAR_NAME)
    if env_key:
        # Warn about ENV usage if running in Docker
        if os.path.exists("/.dockerenv"):
            logger.warning(
                "KESTREL_DATA_KEY in ENV is insecure in Docker. "
                "Use Docker Secrets: --secret kestrel_data_key or mount to /run/secrets/"
            )
        return _strip_quotes(env_key)

    return None


def _get_master_key() -> bytes:
    """
    Get the master key, raising if not configured.

    Returns:
        32-byte master key

    Raises:
        MasterKeyNotConfiguredError: If KESTREL_DATA_KEY not set
    """
    key = _get_data_key()
    if not key:
        raise MasterKeyNotConfiguredError(
            f"{ENV_VAR_NAME} environment variable is not set. "
            "Encryption requires this to be configured."
        )

    # Normalize to 32 bytes via SHA-256
    return hashlib.sha256(key.encode("utf-8")).digest()


# =============================================================================
# Fernet-based Encryption (Global and Per-Agent)
# =============================================================================

def get_fernet() -> Optional[Fernet]:
    """
    Initialize Fernet encryption from KESTREL_DATA_KEY.

    Returns:
        Fernet instance if key is available, None otherwise

    Key sources (in priority order):
        1. KESTREL_DATA_KEY_FILE - Path to secrets file
        2. /run/secrets/kestrel_data_key - Docker Secrets default
        3. KESTREL_DATA_KEY environment variable (legacy)

    Key formats:
        1. Valid 32-byte base64 Fernet key (used directly)
        2. Arbitrary passphrase (SHA-256 derived to Fernet key)
    """
    key = _get_data_key()
    if not key:
        return None

    try:
        # Try using key directly as Fernet key
        Fernet(key)  # Validate
        return Fernet(key)
    except Exception as e:
        # Derive Fernet key from passphrase using SHA-256
        logger.debug(f"Key is not a raw Fernet key, deriving from passphrase: {e}")
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        fernet_key = base64.urlsafe_b64encode(digest)
        return Fernet(fernet_key)


def get_master_key_bytes() -> Optional[bytes]:
    """
    Get master encryption key as bytes for Fernet.

    Handles both raw Fernet keys and passphrases (via SHA-256 derivation).
    Use this when you need the raw key bytes (e.g., for Lighthouse, Filecoin).

    Returns:
        32-byte URL-safe base64 encoded key, or None if key not available
    """
    key = _get_data_key()
    if not key:
        return None

    # Try as raw Fernet key first (32 bytes, URL-safe base64)
    try:
        key_bytes = key.encode() if isinstance(key, str) else key
        Fernet(key_bytes)  # Validate it's a valid Fernet key
        return key_bytes
    except Exception as e:
        # Derive from passphrase via SHA-256
        logger.debug(f"Key is not a raw Fernet key, deriving bytes from passphrase: {e}")
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest)


def get_agent_fernet(agent_id: str) -> Optional[Fernet]:
    """
    Get Fernet instance with per-agent derived key using HKDF.

    Each agent gets a unique encryption key derived from:
    - Master key (from KESTREL_DATA_KEY)
    - Agent DID as salt
    - Version info for future key rotation

    Args:
        agent_id: Agent's DID (e.g., "did:pkh:eip155:1:0x...")

    Returns:
        Fernet instance with agent-specific key, or None if no master key
    """
    master = get_master_key_bytes()
    if not master:
        return None

    if not agent_id:
        logger.warning("get_agent_fernet called with empty agent_id, falling back to global key")
        return get_fernet()

    try:
        # HKDF derivation with agent DID as salt
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=agent_id.encode('utf-8'),
            info=b"kestrel-agent-v1"
        )
        derived = hkdf.derive(master)
        return Fernet(base64.urlsafe_b64encode(derived))
    except Exception as e:
        logger.error(f"Failed to derive agent key: {e}")
        return None


# =============================================================================
# Purpose-Specific Encryption (for service keys, wallets, etc.)
# =============================================================================

def get_agent_key(agent_did: str, purpose: str) -> bytes:
    """
    Derive a 32-byte key for an agent and purpose.

    The key is deterministically derived:
    KESTREL_DATA_KEY -> SHA256 -> HKDF(agent_did) -> HKDF(purpose)

    Args:
        agent_did: Agent's DID (required, cannot be empty)
        purpose: One of "conversations", "service-keys", "wallet", "backup"

    Returns:
        32-byte derived key

    Raises:
        MasterKeyNotConfiguredError: If KESTREL_DATA_KEY not set
        InvalidPurposeError: If purpose not in VALID_PURPOSES
        ValueError: If agent_did is empty
    """
    if not agent_did:
        raise ValueError("agent_did is required for encryption")

    if purpose not in VALID_PURPOSES:
        raise InvalidPurposeError(
            f"Invalid purpose '{purpose}'. Must be one of: {', '.join(sorted(VALID_PURPOSES))}"
        )

    master_key = _get_master_key()

    # First HKDF: derive agent master key
    hkdf_agent = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=agent_did.encode("utf-8"),
        info=b"kestrel-agent-master-v1",
    )
    agent_master_key = hkdf_agent.derive(master_key)

    # Second HKDF: derive purpose-specific key
    info = f"kestrel-{purpose}-v1".encode("utf-8")
    hkdf_purpose = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=None,  # Salt already used in agent master derivation
        info=info,
    )
    return hkdf_purpose.derive(agent_master_key)


def encrypt(agent_did: str, purpose: str, plaintext: bytes) -> bytes:
    """
    Encrypt bytes for an agent with purpose-specific key.

    Uses Fernet (symmetric encryption with authentication).
    The key is derived from master key + agent DID + purpose.

    Args:
        agent_did: Agent's DID
        purpose: Encryption purpose (conversations, service-keys, wallet, backup)
        plaintext: Data to encrypt

    Returns:
        bytes: Fernet-encrypted ciphertext

    Raises:
        MasterKeyNotConfiguredError: If KESTREL_DATA_KEY not set
        InvalidPurposeError: If purpose not valid
        ValueError: If agent_did is empty
    """
    key = get_agent_key(agent_did, purpose)
    fernet_key = base64.urlsafe_b64encode(key)
    fernet = Fernet(fernet_key)
    return fernet.encrypt(plaintext)


def decrypt(agent_did: str, purpose: str, ciphertext: bytes) -> bytes:
    """
    Decrypt bytes for an agent with purpose-specific key.

    Args:
        agent_did: Agent's DID
        purpose: Encryption purpose
        ciphertext: Fernet-encrypted data

    Returns:
        bytes: Decrypted plaintext

    Raises:
        MasterKeyNotConfiguredError: If KESTREL_DATA_KEY not set
        InvalidPurposeError: If purpose not valid
        DecryptionError: If decryption fails (wrong key or corrupted data)
        ValueError: If agent_did is empty
    """
    key = get_agent_key(agent_did, purpose)
    fernet_key = base64.urlsafe_b64encode(key)
    fernet = Fernet(fernet_key)

    try:
        return fernet.decrypt(ciphertext)
    except InvalidToken as e:
        raise DecryptionError(f"Decryption failed - wrong key or corrupted data: {e}") from e
    except Exception as e:
        raise DecryptionError(f"Decryption failed: {e}") from e


def encrypt_string(agent_did: str, purpose: str, plaintext: str) -> bytes:
    """
    Encrypt a string for an agent.

    Convenience wrapper that encodes string to UTF-8 before encryption.

    Args:
        agent_did: Agent's DID
        purpose: Encryption purpose
        plaintext: String to encrypt

    Returns:
        bytes: Encrypted data
    """
    return encrypt(agent_did, purpose, plaintext.encode("utf-8"))


def decrypt_string(agent_did: str, purpose: str, ciphertext: bytes) -> str:
    """
    Decrypt bytes to a string for an agent.

    Convenience wrapper that decodes UTF-8 after decryption.

    Args:
        agent_did: Agent's DID
        purpose: Encryption purpose
        ciphertext: Encrypted data

    Returns:
        str: Decrypted string
    """
    return decrypt(agent_did, purpose, ciphertext).decode("utf-8")


# =============================================================================
# Legacy Fernet Helpers (for storage layer compatibility)
# =============================================================================

def encrypt_bytes(content: bytes, fernet: Optional[Fernet]) -> Tuple[bytes, bool]:
    """
    Encrypt bytes content if Fernet is available.

    Args:
        content: Raw bytes to encrypt
        fernet: Fernet instance or None

    Returns:
        Tuple of (content, was_encrypted)
        - If fernet is None, returns original content with False
        - If fernet provided, returns encrypted content with True
    """
    if fernet is None:
        return content, False
    return fernet.encrypt(content), True


def decrypt_bytes(content: bytes, fernet: Optional[Fernet], metadata: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Decrypt bytes content if it was encrypted.

    Args:
        content: Bytes to decrypt (may be encrypted or plain)
        fernet: Fernet instance or None
        metadata: Optional metadata dict with 'enc' flag

    Returns:
        Decrypted content, or original if not marked as encrypted

    Raises:
        DecryptionError: If metadata indicates content is encrypted but
                        decryption fails (wrong key or corrupted data)
    """
    if fernet is None:
        # No key available - if data is marked encrypted, we can't decrypt
        if metadata and metadata.get("enc"):
            raise DecryptionError("No decryption key available but content is marked as encrypted")
        return content

    # Check if metadata indicates encryption
    if metadata and metadata.get("enc"):
        try:
            return fernet.decrypt(content)
        except InvalidToken as e:
            raise DecryptionError(f"Decryption failed - wrong key or corrupted data: {e}") from e
        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}") from e

    # Not marked as encrypted - return as-is
    return content


def encrypt_string_fernet(content: str, fernet: Optional[Fernet]) -> Tuple[str, bool]:
    """
    Encrypt string content if Fernet is available.

    Args:
        content: String to encrypt
        fernet: Fernet instance or None

    Returns:
        Tuple of (content, was_encrypted)
        - If fernet is None, returns original content with False
        - If fernet provided, returns base64-encoded encrypted content with True
    """
    if fernet is None:
        return content, False
    encrypted = fernet.encrypt(content.encode('utf-8')).decode('utf-8')
    return encrypted, True


def decrypt_string_fernet(content: str, metadata: Optional[Dict[str, Any]], fernet: Optional[Fernet]) -> str:
    """
    Decrypt string content if it was encrypted.

    Args:
        content: String to decrypt (may be encrypted or plain)
        metadata: Optional metadata dict with 'enc' flag
        fernet: Fernet instance or None

    Returns:
        Decrypted content, or original if not marked as encrypted

    Raises:
        DecryptionError: If metadata indicates content is encrypted but
                        decryption fails (wrong key or corrupted data)
    """
    if fernet is None:
        # No key available - if data is marked encrypted, we can't decrypt
        if metadata and metadata.get("enc"):
            raise DecryptionError("No decryption key available but content is marked as encrypted")
        return content

    # Check if metadata indicates encryption
    if metadata and metadata.get("enc"):
        try:
            return fernet.decrypt(content.encode('utf-8')).decode('utf-8')
        except InvalidToken as e:
            raise DecryptionError(f"Decryption failed - wrong key or corrupted data: {e}") from e
        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}") from e

    # Not marked as encrypted - return as-is
    return content


def remove_enc_flag(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Remove the internal 'enc' flag from metadata for external use.

    Args:
        metadata: Metadata dict that may contain 'enc' flag

    Returns:
        Metadata with 'enc' key removed, or None if input was None/empty
    """
    if not metadata:
        return None
    cleaned = {k: v for k, v in metadata.items() if k != 'enc'}
    return cleaned if cleaned else None
