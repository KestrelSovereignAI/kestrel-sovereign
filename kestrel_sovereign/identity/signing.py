#!/usr/bin/env python3
"""
Identity Signing: DID-based cryptographic signing for identity packages.

This module provides functions for signing and verifying AgentIdentityPackage
instances using the agent's DID key pair.

The signing process:
1. Compute content hash of package (excluding signature fields)
2. Sign the hash using the agent's private key (secp256k1/ECDSA)
3. Store signature in the package

Verification:
1. Recompute content hash
2. Verify signature using the public key from DID document
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.exceptions import InvalidSignature

from .identity_package import AgentIdentityPackage

logger = logging.getLogger(__name__)


class SigningError(Exception):
    """Error during signing operation."""
    pass


class VerificationError(Exception):
    """Error during verification operation."""
    pass


def extract_address_from_did(did: str) -> str:
    """
    Extract Ethereum address from DID.

    Args:
        did: DID string like "did:pkh:eip155:1:{address}"

    Returns:
        The Ethereum address portion

    Raises:
        ValueError: If DID format is invalid
    """
    parts = did.split(":")
    if len(parts) < 5 or parts[0] != "did":
        raise ValueError(f"Invalid DID format: {did}")
    return parts[4]


def get_key_id(did: str) -> str:
    """
    Get the key ID used for storage from a DID.

    Args:
        did: The agent's DID

    Returns:
        Key ID like "kestrel_{address}"
    """
    address = extract_address_from_did(did)
    return f"kestrel_{address}"


def sign_package(
    package: AgentIdentityPackage,
    storage_dir: Optional[Path] = None,
) -> AgentIdentityPackage:
    """
    Sign an identity package using the agent's private key.

    Args:
        package: The identity package to sign
        storage_dir: Directory containing the agent's keys

    Returns:
        The package with content_hash and signature fields populated

    Raises:
        SigningError: If signing fails
    """
    try:
        from kestrel_sovereign.inception_service import load_kestrel_identity

        # Load private key
        key_id = get_key_id(package.did)
        private_key, _ = load_kestrel_identity(key_id, storage_dir)

        # Compute content hash
        content_hash = package.compute_content_hash()
        package.content_hash = content_hash

        # Sign the hash
        signature = private_key.sign(
            content_hash.encode('utf-8'),
            ec.ECDSA(hashes.SHA256())
        )

        # Store signature as hex
        package.signature = signature.hex()

        logger.info(f"Signed package for {package.did[:20]}...")
        return package

    except FileNotFoundError as e:
        raise SigningError(f"Private key not found: {e}")
    except Exception as e:
        raise SigningError(f"Signing failed: {e}")


def verify_package_signature(
    package: AgentIdentityPackage,
    storage_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Verify the signature on an identity package.

    This function:
    1. Recomputes the content hash
    2. Verifies the signature using the agent's public key

    Args:
        package: The identity package to verify
        storage_dir: Directory containing the agent's DID document

    Returns:
        Tuple of (is_valid, message)
    """
    if not package.signature:
        return False, "Package is not signed"

    if not package.content_hash:
        return False, "Package has no content hash"

    # Verify content hash matches
    computed_hash = package.compute_content_hash()
    if computed_hash != package.content_hash:
        return False, "Content hash mismatch - package may have been modified"

    try:
        from kestrel_sovereign.inception_service import load_kestrel_identity

        # Load private key to get public key
        key_id = get_key_id(package.did)
        private_key, did_document = load_kestrel_identity(key_id, storage_dir)
        public_key = private_key.public_key()

        # Verify signature
        signature_bytes = bytes.fromhex(package.signature)
        public_key.verify(
            signature_bytes,
            package.content_hash.encode('utf-8'),
            ec.ECDSA(hashes.SHA256())
        )

        return True, "Signature valid"

    except InvalidSignature:
        return False, "Invalid signature"
    except FileNotFoundError:
        # Try to verify using public key from DID document
        return _verify_with_did_document(package, storage_dir)
    except Exception as e:
        return False, f"Verification failed: {e}"


def _verify_with_did_document(
    package: AgentIdentityPackage,
    storage_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Verify signature using only the public key from DID document.

    This is useful when importing a package from another source where
    we don't have the private key, but we can verify the signature
    using the public key embedded in the DID document.

    Tries two sources for the public key:
    1. Local DID document file ({key_id}.json) if previously imported
    2. Could be extended for DID resolution in the future
    """
    import json

    try:
        key_id = get_key_id(package.did)

        # Try loading the DID document from local storage
        if storage_dir is None:
            from kestrel_sovereign.storage import get_default_agent_data_dir
            storage_dir = Path(get_default_agent_data_dir())
        else:
            storage_dir = Path(storage_dir)

        did_path = storage_dir / f"{key_id}.json"
        if not did_path.exists():
            return False, f"DID document not found at {did_path} - cannot verify without public key"

        with open(did_path, 'r') as f:
            did_document = json.load(f)

        # Extract publicKeyHex from the DID document
        public_keys = did_document.get("publicKey", [])
        if not public_keys:
            return False, "DID document has no publicKey entries"

        public_key_hex = public_keys[0].get("publicKeyHex")
        if not public_key_hex:
            return False, "DID document publicKey entry has no publicKeyHex"

        # Reconstruct EC public key from the uncompressed hex
        public_key_bytes = bytes.fromhex(public_key_hex)
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), public_key_bytes
        )

        # Verify the signature
        signature_bytes = bytes.fromhex(package.signature)
        public_key.verify(
            signature_bytes,
            package.content_hash.encode('utf-8'),
            ec.ECDSA(hashes.SHA256())
        )

        return True, "Signature valid (verified via DID document)"

    except InvalidSignature:
        return False, "Invalid signature (DID document verification)"
    except (ValueError, KeyError, IndexError) as e:
        return False, f"DID document parsing failed: {e}"
    except Exception as e:
        return False, f"DID document verification failed: {e}"


async def sign_and_export(
    package: AgentIdentityPackage,
    storage_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Sign a package and export to JSON.

    Args:
        package: The identity package to sign and export
        storage_dir: Directory containing the agent's keys
        output_path: Optional path to write JSON file

    Returns:
        JSON string of signed package
    """
    # Sign the package
    signed_package = sign_package(package, storage_dir)

    # Export to JSON
    json_str = signed_package.to_json()

    # Write to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(json_str)
        logger.info(f"Exported signed package to {output_path}")

    return json_str


def verify_and_load(
    json_str: str,
    storage_dir: Optional[Path] = None,
    require_valid_signature: bool = False,
) -> Tuple[AgentIdentityPackage, bool, str]:
    """
    Load and verify an identity package from JSON.

    Args:
        json_str: JSON string of the package
        storage_dir: Directory containing the agent's keys (for verification)
        require_valid_signature: If True, raise error on invalid signature

    Returns:
        Tuple of (package, is_valid, message)

    Raises:
        VerificationError: If require_valid_signature and signature is invalid
    """
    # Load package
    package = AgentIdentityPackage.from_json(json_str)

    # Verify constitution if present
    if package.constitution_text:
        if not package.verify_constitution():
            if require_valid_signature:
                raise VerificationError("Constitution hash verification failed")
            return package, False, "Constitution hash verification failed"

    # Verify signature if present
    if package.signature:
        is_valid, message = verify_package_signature(package, storage_dir)
        if not is_valid and require_valid_signature:
            raise VerificationError(message)
        return package, is_valid, message

    return package, True, "Package has no signature (unsigned)"


class PackageSigner:
    """
    Context manager for signing packages with a specific key.

    Usage:
        with PackageSigner(agent_did, storage_dir) as signer:
            signed_package = signer.sign(package)
    """

    def __init__(self, agent_did: str, storage_dir: Optional[Path] = None):
        self.agent_did = agent_did
        self.storage_dir = storage_dir
        self._private_key = None

    def __enter__(self):
        from kestrel_sovereign.inception_service import load_kestrel_identity
        key_id = get_key_id(self.agent_did)
        self._private_key, _ = load_kestrel_identity(key_id, self.storage_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._private_key = None
        return False

    def sign(self, package: AgentIdentityPackage) -> AgentIdentityPackage:
        """Sign a package with the loaded key."""
        if self._private_key is None:
            raise SigningError("Signer not initialized - use within 'with' block")

        content_hash = package.compute_content_hash()
        package.content_hash = content_hash

        signature = self._private_key.sign(
            content_hash.encode('utf-8'),
            ec.ECDSA(hashes.SHA256())
        )
        package.signature = signature.hex()

        return package
