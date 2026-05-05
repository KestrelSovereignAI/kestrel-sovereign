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


_HYBRID_PREFIX = "hybrid:"


def sign_mandate(
    mandate: SpawnMandate,
    parent_private_key: ec.EllipticCurvePrivateKey,
    *,
    parent_identity=None,
) -> SpawnMandate:
    """Sign the mandate with the parent's keys.

    Two paths:

    - **Hybrid parent**: pass ``parent_identity`` (an
      :class:`AgentIdentity` from ``runtime_identity.load_agent_identity``).
      When ``parent_identity.is_hybrid`` is True, the mandate is signed
      with both the Ed25519 and ML-DSA-65 halves via ``sign_hybrid``;
      ``parent_signature`` is set to ``"hybrid:" + base64(json.dumps(sigs))``.
      ``parent_private_key`` is still required (and must be the legacy
      ECDSA key) so the function has a uniform fall-through if hybrid
      signing fails.
    - **Legacy parent**: omit ``parent_identity``. The mandate is signed
      with secp256k1 ECDSA (Wave 1 sub-PR 5 path), and
      ``parent_signature`` is the hex-encoded DER signature. Same
      shape and bytes as before this PR.

    The verify side accepts both formats by prefix-sniffing
    ``parent_signature``.
    """
    from kestrel_sovereign.security.crypto_suite import (
        ALG_ECDSA_SECP256K1_SHA256, get_suite,
    )
    payload = mandate._signable_payload()

    if parent_identity is not None and parent_identity.is_hybrid:
        import base64
        from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
        vms = parent_identity.new_verification_methods or []
        classical_kid = vms[0]["id"].rsplit("#", 1)[-1] if vms else "key-1"
        pq_kid = (
            vms[1]["id"].rsplit("#", 1)[-1] if len(vms) > 1 else "key-2"
        )
        hybrid_sigs = sign_hybrid(
            payload,
            parent_identity.hybrid_keypair,
            classical_kid=classical_kid,
            pq_kid=pq_kid,
        )
        mandate.parent_signature = (
            _HYBRID_PREFIX
            + base64.b64encode(json.dumps(hybrid_sigs).encode()).decode()
        )
        return mandate

    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    signature = suite.sign(payload, parent_private_key)
    mandate.parent_signature = signature.hex()
    return mandate


def verify_mandate(
    mandate: SpawnMandate,
    parent_public_key: ec.EllipticCurvePublicKey,
    *,
    parent_verification_methods=None,
) -> bool:
    """Verify the mandate's signature.

    The verify path is wire-format-aware:

    - ``parent_signature`` starts with ``"hybrid:"`` → parse the
      base64-wrapped JSON list of ``{alg, kid, sig}`` entries and
      require BOTH ed25519 AND ml-dsa-65 to verify against the
      caller-supplied ``parent_verification_methods`` (which list
      the parent's hybrid public keys, multibase-encoded). Matches
      the HYBRID_REQUIRED policy: stripping the PQ half is rejected.
    - Otherwise (bare hex) → classic secp256k1 ECDSA verify against
      ``parent_public_key``. Same path mandates produced before this
      PR follow.
    """
    if not mandate.parent_signature:
        logger.warning("Mandate has no signature to verify")
        return False

    payload = mandate._signable_payload()

    if mandate.parent_signature.startswith(_HYBRID_PREFIX):
        return _verify_mandate_hybrid(
            mandate, payload, parent_verification_methods,
        )

    from kestrel_sovereign.security.crypto_suite import (
        ALG_ECDSA_SECP256K1_SHA256, get_suite,
    )
    try:
        signature_bytes = bytes.fromhex(mandate.parent_signature)
    except ValueError:
        logger.warning("Mandate signature is neither hex nor a known prefix")
        return False
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    if suite.verify(payload, signature_bytes, parent_public_key):
        return True
    logger.warning("Mandate signature verification failed")
    return False


def _verify_mandate_hybrid(
    mandate: SpawnMandate,
    payload: bytes,
    parent_verification_methods,
) -> bool:
    """Verify a ``hybrid:``-prefixed mandate signature.

    Both ed25519 AND ml-dsa-65 must be present in the array AND
    crypto-verify against pubkeys resolved from
    ``parent_verification_methods``. Stripping the PQ half is the
    canonical attack we defend against here.
    """
    import base64
    from kestrel_sovereign.security.crypto_suite import (
        ALG_ED25519, ALG_ML_DSA_65, get_suite,
    )
    from kestrel_sovereign.security.multikey import (
        multibase_to_public_key,
    )

    if not parent_verification_methods:
        logger.warning(
            "Mandate has hybrid: signature but no parent_verification_methods "
            "supplied; cannot resolve pubkeys"
        )
        return False

    try:
        b64 = mandate.parent_signature[len(_HYBRID_PREFIX):]
        sigs = json.loads(base64.b64decode(b64).decode("utf-8"))
        if not isinstance(sigs, list) or not sigs:
            logger.warning("Hybrid mandate signature payload empty or malformed")
            return False
    except Exception as e:
        logger.warning(f"Hybrid mandate signature parse failed: {e}")
        return False

    # Build kid -> (alg, pubkey) from the supplied VMs
    kid_to_pub: dict = {}
    for vm in parent_verification_methods:
        vm_id = vm.get("id") or ""
        kid = vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id
        mb = vm.get("publicKeyMultibase")
        if not kid or not mb:
            continue
        try:
            suite, pub = multibase_to_public_key(mb)
        except Exception:
            continue
        kid_to_pub[kid] = (suite.alg_id, pub)

    algs_seen: set = set()
    for entry in sigs:
        try:
            alg = entry["alg"]
            kid = entry["kid"]
            sig_hex = entry["sig"]
        except (TypeError, KeyError):
            logger.warning("Hybrid mandate signature entry malformed")
            return False
        info = kid_to_pub.get(kid)
        if info is None:
            logger.warning(f"No verification method for mandate kid {kid!r}")
            return False
        expected_alg, public_key = info
        if expected_alg != alg:
            logger.warning(
                f"Mandate kid={kid!r} alg={alg!r} doesn't match VM alg={expected_alg!r}"
            )
            return False
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            logger.warning(f"Mandate hybrid signature {kid!r} has malformed hex")
            return False
        suite = get_suite(alg)
        if not suite.verify(payload, sig_bytes, public_key):
            logger.warning(f"Mandate hybrid signature {kid!r} ({alg}) failed verify")
            return False
        algs_seen.add(alg)

    required = {ALG_ED25519, ALG_ML_DSA_65}
    if not required.issubset(algs_seen):
        missing = required - algs_seen
        logger.warning(
            f"Mandate missing required hybrid algs: {sorted(missing)} "
            f"(HYBRID_REQUIRED on mandate signing)"
        )
        return False
    return True


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
