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

    Behavior:

    - **Hybrid agent** (post-rotation ceremony): populates the v2
      ``signatures`` array with both an Ed25519 and an ML-DSA-65
      signature via ``sign_hybrid``, AND keeps a legacy ECDSA hex
      in ``signature`` so v1 readers (importers that predate the
      v2 array) still verify.
    - **Legacy agent** (pre-ceremony): populates ``signature`` only,
      same shape as before.

    The v2 array is the authoritative form; v1 readers fall back to
    ``signature``. ``content_hash`` is unchanged in either path.

    Args:
        package: The identity package to sign
        storage_dir: Directory containing the agent's keys

    Returns:
        The package with content_hash + signature(s) populated.

    Raises:
        SigningError: If signing fails
    """
    try:
        # Try hybrid load first: if a succession statement exists in
        # storage_dir, this returns the AgentIdentity with both halves
        # of the hybrid keypair. Pre-ceremony agents fall back to the
        # legacy single-key path below.
        agent_identity = None
        try:
            from kestrel_sovereign.identity.runtime_identity import (
                load_agent_identity,
            )
            key_id = get_key_id(package.did)
            agent_identity = load_agent_identity(key_id, storage_dir=storage_dir)
        except Exception as e:
            logger.debug(f"Hybrid identity load fell through to legacy: {e}")
            agent_identity = None

        from kestrel_sovereign.security.crypto_suite import (
            ALG_ECDSA_SECP256K1_SHA256, get_suite,
        )
        secp = get_suite(ALG_ECDSA_SECP256K1_SHA256)

        if agent_identity is not None and agent_identity.is_hybrid:
            # Hybrid path: populate v2 signatures array with sign_hybrid
            # output, plus a legacy ECDSA in signature for v1 fallback.
            #
            # Order matters: ``compute_content_hash`` for a v2 package
            # INCLUDES ``verification_methods`` in the hashed payload
            # (so an attacker can't swap pubkeys post-sign). We must
            # set verification_methods BEFORE computing the hash. The
            # `signature` / `signatures` / `content_hash` fields are
            # explicitly excluded by ``compute_content_hash`` so we
            # populate them after.
            from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
            vms = agent_identity.new_verification_methods or []
            package.verification_methods = list(vms)

            content_hash = package.compute_content_hash()
            package.content_hash = content_hash

            classical_kid = vms[0]["id"].rsplit("#", 1)[-1] if vms else "key-1"
            pq_kid = (
                vms[1]["id"].rsplit("#", 1)[-1] if len(vms) > 1 else "key-2"
            )
            hybrid_sigs = sign_hybrid(
                content_hash.encode("utf-8"),
                agent_identity.hybrid_keypair,
                classical_kid=classical_kid,
                pq_kid=pq_kid,
            )
            package.signatures = list(hybrid_sigs)
            # Legacy v1 fallback: also sign with the legacy ECDSA key
            # so importers that don't yet read the v2 array continue
            # to verify.
            legacy_sig = secp.sign(
                content_hash.encode("utf-8"),
                agent_identity.legacy_keypair.private_key,
            )
            package.signature = legacy_sig.hex()
            logger.info(
                f"Signed package for {package.did[:20]}... HYBRID "
                f"(ed25519 + ml-dsa-65) + legacy ecdsa fallback"
            )
            return package

        # Legacy path: single ECDSA signature, hex-encoded.
        content_hash = package.compute_content_hash()
        package.content_hash = content_hash

        from kestrel_sovereign.inception_service import load_kestrel_identity
        key_id = get_key_id(package.did)
        private_key, _ = load_kestrel_identity(key_id, storage_dir)
        signature = secp.sign(content_hash.encode("utf-8"), private_key)
        package.signature = signature.hex()
        logger.info(f"Signed package for {package.did[:20]}... (legacy ecdsa)")
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

    Verification order:

    1. **Content hash check** — recompute from the package and compare
       to ``package.content_hash``. Mismatch means the package was
       modified after signing.
    2. **v2 hybrid signatures** — if ``package.signatures`` is non-empty,
       use it as the authoritative form. Both ed25519 AND ml-dsa-65
       must be present and verify (HYBRID_REQUIRED on identity-package
       signing). Verification methods come from ``package.verification_methods``
       embedded in the package itself; no external fetch needed.
    3. **Legacy v1 fallback** — if ``signatures`` is empty, fall back to
       ``package.signature`` (single ECDSA hex over content_hash) for
       v1 packages that predate the v2 array.

    Args:
        package: The identity package to verify
        storage_dir: Directory containing the agent's DID document

    Returns:
        Tuple of (is_valid, message)
    """
    if not package.content_hash:
        return False, "Package has no content hash"

    # Step 1: content hash unchanged
    computed_hash = package.compute_content_hash()
    if computed_hash != package.content_hash:
        return False, "Content hash mismatch - package may have been modified"

    # Step 2: prefer v2 signatures array if present (hybrid agent)
    if package.signatures:
        return _verify_v2_signatures(package)

    # Step 3: legacy v1 fallback
    if not package.signature:
        return False, "Package is not signed"
    try:
        from kestrel_sovereign.inception_service import load_kestrel_identity
        key_id = get_key_id(package.did)
        private_key, _ = load_kestrel_identity(key_id, storage_dir)
        public_key = private_key.public_key()

        from kestrel_sovereign.security.crypto_suite import (
            ALG_ECDSA_SECP256K1_SHA256, get_suite,
        )
        signature_bytes = bytes.fromhex(package.signature)
        suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
        if suite.verify(
            package.content_hash.encode("utf-8"),
            signature_bytes,
            public_key,
        ):
            return True, "Signature valid (legacy ecdsa)"
        return False, "Invalid signature"

    except FileNotFoundError:
        # Try to verify using public key from DID document
        return _verify_with_did_document(package, storage_dir)
    except Exception as e:
        return False, f"Verification failed: {e}"


def _verify_v2_signatures(
    package: AgentIdentityPackage,
) -> Tuple[bool, str]:
    """Verify the v2 ``signatures`` array on a hybrid identity package.

    Both ed25519 AND ml-dsa-65 signatures must be present and crypto-
    verify. Stripping the PQ half (leaving only the classical) is
    rejected — that's the canonical attack hybrid identity defends
    against.

    Public keys come from ``package.verification_methods`` embedded in
    the package itself: each entry's ``id`` ends in ``#<kid>`` matching
    a ``signatures.kid``, and ``publicKeyMultibase`` carries the raw
    pubkey in W3C Multikey form. No network fetch needed.
    """
    from kestrel_sovereign.security.crypto_suite import (
        ALG_ED25519, ALG_ML_DSA_65, get_suite,
    )
    from kestrel_sovereign.security.multikey import multibase_to_public_key

    if not package.verification_methods:
        return False, (
            "v2 signatures present but verification_methods missing — "
            "cannot resolve kids to public keys"
        )

    # Build kid -> (alg, public_key) from the package's VMs
    kid_to_pub: dict = {}
    for vm in package.verification_methods:
        vm_id = vm.get("id") or ""
        kid = vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id
        mb = vm.get("publicKeyMultibase")
        if not kid or not mb:
            continue
        try:
            suite, pub = multibase_to_public_key(mb)
        except Exception as e:
            logger.warning(f"VM {kid!r} multibase decode failed: {e}")
            continue
        kid_to_pub[kid] = (suite.alg_id, pub)

    payload = package.content_hash.encode("utf-8")
    algs_seen: set = set()
    for entry in package.signatures:
        try:
            alg = entry["alg"]
            kid = entry["kid"]
            sig_hex = entry["sig"]
        except (TypeError, KeyError):
            return False, "Malformed signature entry"
        info = kid_to_pub.get(kid)
        if info is None:
            return False, f"No verification method for kid {kid!r}"
        expected_alg, public_key = info
        if expected_alg != alg:
            return False, (
                f"alg/kid mismatch: signature claims {alg!r} but "
                f"verification method {kid!r} uses {expected_alg!r}"
            )
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            return False, f"Signature {kid!r} has malformed hex"
        suite = get_suite(alg)
        if not suite.verify(payload, sig_bytes, public_key):
            return False, f"Signature {kid!r} ({alg}) failed crypto verify"
        algs_seen.add(alg)

    required = {ALG_ED25519, ALG_ML_DSA_65}
    if not required.issubset(algs_seen):
        missing = required - algs_seen
        return False, (
            f"Missing required hybrid algs: {sorted(missing)} "
            f"(HYBRID_REQUIRED on identity-package signing)"
        )
    return True, "Signature valid (hybrid)"


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

        with open(did_path, 'r', encoding='utf-8') as f:
            did_document = json.load(f)

        # Extract publicKeyHex from the DID document
        public_keys = did_document.get("publicKey", [])
        if not public_keys:
            return False, "DID document has no publicKey entries"

        public_key_hex = public_keys[0].get("publicKeyHex")
        if not public_key_hex:
            return False, "DID document publicKey entry has no publicKeyHex"

        # Reconstruct EC public key + verify via Secp256k1Suite
        # (Wave 1 sub-PR 5).
        from kestrel_sovereign.security.crypto_suite import (
            ALG_ECDSA_SECP256K1_SHA256, get_suite,
        )
        suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
        public_key_bytes = bytes.fromhex(public_key_hex)
        public_key = suite.deserialize_public_key(public_key_bytes)

        signature_bytes = bytes.fromhex(package.signature)
        if suite.verify(
            package.content_hash.encode("utf-8"),
            signature_bytes,
            public_key,
        ):
            return True, "Signature valid (verified via DID document)"
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
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        logger.info(f"Exported signed package to {output_path}")

    return json_str


def verify_and_load(
    json_str: str,
    storage_dir: Optional[Path] = None,
    require_valid_signature: bool = False,
    allow_unsigned: bool = False,
) -> Tuple[AgentIdentityPackage, bool, str]:
    """
    Load and verify an identity package from JSON.

    Args:
        json_str: JSON string of the package
        storage_dir: Directory containing the agent's keys (for verification)
        require_valid_signature: If True, raise error on invalid signature
        allow_unsigned: If True, treat unsigned packages as valid (for
            development/testing). Defaults to False for security.

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

    # Package is unsigned
    if allow_unsigned:
        return package, True, "Package has no signature (unsigned, allowed)"

    if require_valid_signature:
        raise VerificationError("Package has no signature (unsigned)")

    return package, False, "Package has no signature (unsigned)"


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
        """Sign a package with the loaded key.

        Wave 1 sub-PR 5: routed through Secp256k1Suite (behavior-identical).
        """
        if self._private_key is None:
            raise SigningError("Signer not initialized - use within 'with' block")

        from kestrel_sovereign.security.crypto_suite import (
            ALG_ECDSA_SECP256K1_SHA256, get_suite,
        )

        content_hash = package.compute_content_hash()
        package.content_hash = content_hash

        suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
        signature = suite.sign(content_hash.encode("utf-8"), self._private_key)
        package.signature = signature.hex()

        return package
