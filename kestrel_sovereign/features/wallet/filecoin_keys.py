"""
Filecoin Key Management for Kestrel Wallet.

Generates and securely stores Filecoin wallet keys using the existing
SecureKeyStorage infrastructure.

Key Types:
- secp256k1: Standard Filecoin address (f1.../t1...)
- BLS: Aggregated signatures (f3...) - not yet implemented

Security:
- Private keys encrypted at rest using KESTREL_DATA_KEY
- Follows existing key management patterns from security/key_storage.py

Address Format:
- Mainnet: f1<base32(blake2b-160(pubkey))>
- Calibration: t1<base32(blake2b-160(pubkey))>

Usage:
    manager = FilecoinKeyManager()

    # Generate new address for agent
    address, pub_key = await manager.generate_address("agent_123")

    # Get existing address
    address = manager.get_address("agent_123")
"""

import os
import logging
import hashlib
from typing import Optional, Tuple
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat
)
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

# Filecoin address protocol byte for secp256k1
SECP256K1_PROTOCOL = 1

# Base32 alphabet used by Filecoin (RFC 4648, lowercase, no padding)
BASE32_ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"


def _base32_encode(data: bytes) -> str:
    """
    Encode bytes to base32 string (Filecoin-compatible, no padding).

    Args:
        data: Raw bytes to encode

    Returns:
        Base32 encoded string without padding
    """
    import base64
    # Use standard base64 base32 encoding and convert to lowercase, strip padding
    encoded = base64.b32encode(data).decode('ascii').lower().rstrip('=')
    return encoded


def _blake2b_160(data: bytes) -> bytes:
    """
    Compute BLAKE2b-160 hash (20 bytes).

    This is the hash function used for Filecoin secp256k1 addresses.

    Args:
        data: Data to hash

    Returns:
        20-byte hash
    """
    return hashlib.blake2b(data, digest_size=20).digest()


def _derive_f1_address(public_key: ec.EllipticCurvePublicKey, network: str = "calibration") -> str:
    """
    Derive a Filecoin secp256k1 address from a public key.

    Address format: <network_prefix><protocol><base32(blake2b-160(pubkey))><checksum>

    For secp256k1 (protocol 1):
    - The public key is the uncompressed 65-byte representation
    - Hash with BLAKE2b-160 (20 bytes)
    - Encode with base32
    - Add 4-byte checksum (blake2b of protocol + payload)

    Args:
        public_key: secp256k1 public key object
        network: "calibration" (testnet) or "mainnet"

    Returns:
        Filecoin address string (t1... or f1...)
    """
    # Get uncompressed public key bytes (65 bytes: 04 prefix + 32 x + 32 y)
    pub_bytes = public_key.public_bytes(
        Encoding.X962,
        PublicFormat.UncompressedPoint
    )

    # Compute BLAKE2b-160 hash of public key
    address_payload = _blake2b_160(pub_bytes)

    # Compute checksum: blake2b-32 of (protocol byte + payload)
    checksum_input = bytes([SECP256K1_PROTOCOL]) + address_payload
    checksum = hashlib.blake2b(checksum_input, digest_size=4).digest()

    # Encode payload + checksum
    encoded = _base32_encode(address_payload + checksum)

    # Network prefix: 't' for testnet/calibration, 'f' for mainnet
    prefix = "t" if network.lower() in ("calibration", "testnet") else "f"

    return f"{prefix}{SECP256K1_PROTOCOL}{encoded}"


class FilecoinKeyManager:
    """
    Manages Filecoin wallet keys for Kestrel agents.

    Keys are encrypted at rest using the existing SecureKeyStorage
    infrastructure (requires KESTREL_DATA_KEY environment variable).
    """

    KEY_PREFIX = "filecoin"

    def __init__(self, storage_dir: Optional[Path] = None, network: str = "calibration"):
        """
        Initialize key manager.

        Args:
            storage_dir: Directory for key storage. Defaults to KESTREL_DB_PATH or ./agent_dbs
            network: Filecoin network ("calibration" or "mainnet")
        """
        if storage_dir is None:
            db_path = os.environ.get("KESTREL_DB_PATH", "./agent_dbs")
            storage_dir = Path(db_path)

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.network = network
        self._secure_storage = None
        self._init_secure_storage()

    def _init_secure_storage(self):
        """Initialize secure key storage if master key available."""
        # Check if master key is configured before trying to use storage
        master_key = os.environ.get("KESTREL_DATA_KEY")
        if not master_key:
            logger.warning(
                "KESTREL_DATA_KEY not set. "
                "Filecoin keys will not be persisted securely. "
                "Generated addresses can only be used for receiving."
            )
            self._secure_storage = None
            return

        try:
            from kestrel_sovereign.security.key_storage import SecureKeyStorage
            self._secure_storage = SecureKeyStorage(storage_dir=self.storage_dir)
            logger.info("FilecoinKeyManager using encrypted key storage")
        except ImportError as e:
            logger.warning(
                f"SecureKeyStorage not available (import error): {e}. "
                "Keys will not be persisted securely."
            )
            self._secure_storage = None
        except (OSError, ValueError) as e:
            logger.warning(
                f"SecureKeyStorage not available (initialization error): {e}. "
                "Keys will not be persisted securely.",
                exc_info=True
            )
            self._secure_storage = None
        except Exception as e:
            logger.warning(
                f"SecureKeyStorage not available: {e}. "
                "Keys will not be persisted securely.",
                exc_info=True
            )
            self._secure_storage = None

    def _get_key_id(self, agent_id: str) -> str:
        """Generate key ID for an agent."""
        return f"{self.KEY_PREFIX}_{agent_id}"

    async def generate_address(self, agent_id: str) -> Tuple[str, bytes]:
        """
        Generate a new Filecoin address for an agent.

        Creates a secp256k1 keypair and derives a Filecoin address.
        The private key is stored encrypted if KESTREL_DATA_KEY is set.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Tuple of (address_string, public_key_bytes)

        Raises:
            ValueError: If agent already has an address
        """
        key_id = self._get_key_id(agent_id)

        # Check if already exists
        if self.has_address(agent_id):
            existing = self.get_address(agent_id)
            raise ValueError(
                f"Agent {agent_id} already has a Filecoin address: {existing}. "
                "Use get_address() to retrieve it."
            )

        # Generate secp256k1 keypair
        private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
        public_key = private_key.public_key()

        # Derive Filecoin address
        address = _derive_f1_address(public_key, self.network)

        # Get public key bytes for reference
        public_bytes = public_key.public_bytes(
            encoding=Encoding.X962,
            format=PublicFormat.UncompressedPoint
        )

        # Store private key securely
        if self._secure_storage:
            self._secure_storage.save_private_key(private_key, key_id)
            logger.info(f"Generated and stored Filecoin address: {address}")
        else:
            logger.warning(
                f"Generated Filecoin address {address} but KESTREL_DATA_KEY not set. "
                "Private key will NOT be persisted. "
                "This address can only be used for receiving, not sending."
            )

        return address, public_bytes

    def get_address(self, agent_id: str) -> Optional[str]:
        """
        Get stored Filecoin address for an agent.

        Loads the private key and derives the address from it.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Address string or None if not found
        """
        key_id = self._get_key_id(agent_id)

        if not self._secure_storage or not self._secure_storage.has_key(key_id):
            return None

        try:
            private_key = self._secure_storage.load_private_key(key_id)
            public_key = private_key.public_key()
            return _derive_f1_address(public_key, self.network)
        except (OSError, ValueError) as e:
            logger.error(f"Failed to load Filecoin key for {agent_id} (storage/decryption error): {e}", exc_info=True)
            return None
        except (TypeError, AttributeError) as e:
            logger.error(f"Failed to load Filecoin key for {agent_id} (key format error): {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to load Filecoin key for {agent_id}: {e}", exc_info=True)
            return None

    def has_address(self, agent_id: str) -> bool:
        """
        Check if agent has a stored Filecoin address.

        Args:
            agent_id: Unique agent identifier

        Returns:
            True if address exists
        """
        key_id = self._get_key_id(agent_id)
        return self._secure_storage is not None and self._secure_storage.has_key(key_id)

    def get_public_key(self, agent_id: str) -> Optional[bytes]:
        """
        Get the public key bytes for an agent.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Uncompressed public key bytes (65 bytes) or None
        """
        key_id = self._get_key_id(agent_id)

        if not self._secure_storage or not self._secure_storage.has_key(key_id):
            return None

        try:
            private_key = self._secure_storage.load_private_key(key_id)
            public_key = private_key.public_key()
            return public_key.public_bytes(
                encoding=Encoding.X962,
                format=PublicFormat.UncompressedPoint
            )
        except (OSError, ValueError) as e:
            logger.error(f"Failed to get public key for {agent_id} (storage/decryption error): {e}", exc_info=True)
            return None
        except (TypeError, AttributeError) as e:
            logger.error(f"Failed to get public key for {agent_id} (key format error): {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to get public key for {agent_id}: {e}", exc_info=True)
            return None

    def get_explorer_url(self, address: str) -> str:
        """
        Get block explorer URL for an address.

        Args:
            address: Filecoin address

        Returns:
            Explorer URL string
        """
        if self.network.lower() in ("calibration", "testnet"):
            return f"https://calibration.filfox.info/en/address/{address}"
        else:
            return f"https://filfox.info/en/address/{address}"

    def get_faucet_url(self) -> Optional[str]:
        """
        Get faucet URL for requesting test FIL.

        Returns:
            Faucet URL or None for mainnet
        """
        if self.network.lower() in ("calibration", "testnet"):
            return "https://faucet.calibnet.chainsafe-fil.io/"
        return None
