"""
SpawnMandate: Data structure and cryptographic operations for DID delegation chains.

A SpawnMandate authorizes a parent agent to spawn a child agent with specific
constraints, budget, and purpose. The mandate is signed by the parent's private
key and can be verified by anyone with the parent's public key.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

from kestrel_sovereign.inception_service import (
    create_did_document,
    public_key_to_hex,
    public_key_to_ethereum_address,
)

logger = logging.getLogger(__name__)


@dataclass
class SpawnMandate:
    """Authorization for a parent agent to spawn a child agent.

    The mandate captures the delegation relationship, constraints, and is
    cryptographically signed by the parent to prove authorization.
    """

    parent_did: str
    child_did: Optional[str] = None  # Set after child identity is generated
    constitution_hash: str = ""
    additional_constraints: dict = field(default_factory=dict)
    budget_allocation: float = 0.0
    ttl_seconds: int = 3600
    features_allowed: list[str] = field(default_factory=list)
    purpose: str = ""
    max_child_depth: int = 0
    parent_signature: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def _signable_payload(self) -> bytes:
        """Return the canonical bytes representation for signing.

        Excludes parent_signature (obviously) and created_at (to allow
        signing before or after timestamp assignment without invalidating).
        """
        payload = {
            "parent_did": self.parent_did,
            "child_did": self.child_did,
            "constitution_hash": self.constitution_hash,
            "additional_constraints": self.additional_constraints,
            "budget_allocation": self.budget_allocation,
            "ttl_seconds": self.ttl_seconds,
            "features_allowed": self.features_allowed,
            "purpose": self.purpose,
            "max_child_depth": self.max_child_depth,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    def to_dict(self) -> dict:
        """Serialize the mandate to a plain dict."""
        return asdict(self)


def sign_mandate(
    mandate: SpawnMandate,
    parent_private_key: ec.EllipticCurvePrivateKey,
) -> SpawnMandate:
    """Sign the mandate with the parent's secp256k1 private key.

    Returns the mandate with parent_signature set to the hex-encoded
    DER signature.
    """
    payload = mandate._signable_payload()
    signature = parent_private_key.sign(payload, ec.ECDSA(hashes.SHA256()))
    mandate.parent_signature = signature.hex()
    return mandate


def verify_mandate(
    mandate: SpawnMandate,
    parent_public_key: ec.EllipticCurvePublicKey,
) -> bool:
    """Verify the mandate's signature against the parent's public key.

    Returns True if the signature is valid, False otherwise.
    """
    if not mandate.parent_signature:
        logger.warning("Mandate has no signature to verify")
        return False

    payload = mandate._signable_payload()
    signature_bytes = bytes.fromhex(mandate.parent_signature)

    try:
        parent_public_key.verify(signature_bytes, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        logger.warning("Mandate signature verification failed")
        return False
    except Exception as e:
        logger.error("Unexpected error verifying mandate signature: %s", e)
        return False


def create_child_did_document(
    parent_did: str,
    child_public_key: ec.EllipticCurvePublicKey,
) -> dict:
    """Create a DID document for a child agent with controller pointing to parent.

    The child DID document is a standard W3C DID document with an additional
    ``controller`` field set to the parent's DID, establishing the delegation
    chain.
    """
    child_public_key_hex = public_key_to_hex(child_public_key)
    child_address = public_key_to_ethereum_address(child_public_key)

    did_doc = create_did_document(child_public_key_hex, child_address)
    did_doc["controller"] = parent_did

    return did_doc
