"""
Fernet encryption helpers for Kestrel SDK.

Requires the 'crypto' extra: pip install kestrel-sovereign-sdk[crypto]

Provides Fernet-based encryption at rest with:
- Per-agent key derivation (each agent gets unique keys)
- Purpose-specific subkeys (conversations, service-keys, wallet, backup)
- Multiple key sources (env var, Docker Secrets, file paths)
- Explicit error handling (no silent failures)

Key Hierarchy:
    KESTREL_DATA_KEY (env var or secrets file)
        | (SHA-256 if passphrase)
    Master Key (32-byte Fernet key)
        | (HKDF with agent DID as salt)
    Agent Key
        | (HKDF with purpose as info)
        |-- kestrel-conversations-v1
        |-- kestrel-service-keys-v1
        |-- kestrel-wallet-v1
        +-- kestrel-backup-v1
"""
import hashlib
import base64
import logging
import os
from typing import Optional, Dict, Tuple, Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from .exceptions import (
    DecryptionError,
    MasterKeyNotConfiguredError,
    InvalidPurposeError,
)

logger = logging.getLogger(__name__)


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
            with open(path, 'r', encoding='utf-8') as f:
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

    Returns:
        32-byte URL-safe base64 encoded key, or None if key not available
    """
    key = _get_data_key()
    if not key:
        return None

    try:
        key_bytes = key.encode() if isinstance(key, str) else key
        Fernet(key_bytes)  # Validate it's a valid Fernet key
        return key_bytes
    except Exception as e:
        logger.debug(f"Key is not a raw Fernet key, deriving bytes from passphrase: {e}")
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest)


def get_agent_fernet(agent_id: str) -> Optional[Fernet]:
    """
    Get Fernet instance with per-agent derived key using HKDF.

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
# Purpose-Specific Encryption
# =============================================================================

def get_agent_key(agent_did: str, purpose: str) -> bytes:
    """
    Derive a 32-byte key for an agent and purpose.

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
        salt=None,
        info=info,
    )
    return hkdf_purpose.derive(agent_master_key)


def encrypt(agent_did: str, purpose: str, plaintext: bytes) -> bytes:
    """Encrypt bytes for an agent with purpose-specific key."""
    key = get_agent_key(agent_did, purpose)
    fernet_key = base64.urlsafe_b64encode(key)
    fernet = Fernet(fernet_key)
    return fernet.encrypt(plaintext)


def decrypt(agent_did: str, purpose: str, ciphertext: bytes) -> bytes:
    """Decrypt bytes for an agent with purpose-specific key."""
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
    """Encrypt a string for an agent."""
    return encrypt(agent_did, purpose, plaintext.encode("utf-8"))


def decrypt_string(agent_did: str, purpose: str, ciphertext: bytes) -> str:
    """Decrypt bytes to a string for an agent."""
    return decrypt(agent_did, purpose, ciphertext).decode("utf-8")


# =============================================================================
# Legacy Fernet Helpers
# =============================================================================

def encrypt_bytes(content: bytes, fernet: Optional[Fernet]) -> Tuple[bytes, bool]:
    """Encrypt bytes content if Fernet is available."""
    if fernet is None:
        return content, False
    return fernet.encrypt(content), True


def decrypt_bytes(content: bytes, fernet: Optional[Fernet], metadata: Optional[Dict[str, Any]] = None) -> bytes:
    """Decrypt bytes content if it was encrypted."""
    if fernet is None:
        if metadata and metadata.get("enc"):
            raise DecryptionError("No decryption key available but content is marked as encrypted")
        return content

    if metadata and metadata.get("enc"):
        try:
            return fernet.decrypt(content)
        except InvalidToken as e:
            raise DecryptionError(f"Decryption failed - wrong key or corrupted data: {e}") from e
        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}") from e

    return content


def encrypt_string_fernet(content: str, fernet: Optional[Fernet]) -> Tuple[str, bool]:
    """Encrypt string content if Fernet is available."""
    if fernet is None:
        return content, False
    encrypted = fernet.encrypt(content.encode('utf-8')).decode('utf-8')
    return encrypted, True


def decrypt_string_fernet(content: str, metadata: Optional[Dict[str, Any]], fernet: Optional[Fernet]) -> str:
    """Decrypt string content if it was encrypted."""
    if fernet is None:
        if metadata and metadata.get("enc"):
            raise DecryptionError("No decryption key available but content is marked as encrypted")
        return content

    if metadata and metadata.get("enc"):
        try:
            return fernet.decrypt(content.encode('utf-8')).decode('utf-8')
        except InvalidToken as e:
            raise DecryptionError(f"Decryption failed - wrong key or corrupted data: {e}") from e
        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}") from e

    return content


def remove_enc_flag(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Remove the internal 'enc' flag from metadata for external use."""
    if not metadata:
        return None
    cleaned = {k: v for k, v in metadata.items() if k != 'enc'}
    return cleaned if cleaned else None
