"""Signed constitution amendment/reanchor artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from kestrel_sovereign.identity.hybrid_keypair import (
    HybridKeypair,
    sign_hybrid,
    verify_hybrid,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy


ARTIFACT_TYPE = "kestrel.constitution.reanchor.v1"
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class AmendmentArtifactVerification:
    ok: bool
    reason: str
    signer: str = ""
    constitution_sha256: str = ""


def canonical_amendment_bytes(artifact: Mapping[str, Any]) -> bytes:
    """Return the stable byte payload signed by an amendment authority."""
    signed_fields = {
        "artifact_type": artifact.get("artifact_type"),
        "version": artifact.get("version"),
        "signer": artifact.get("signer"),
        "subject": artifact.get("subject"),
        "constitution_sha256": artifact.get("constitution_sha256"),
        "created_at": artifact.get("created_at"),
        "reason": artifact.get("reason", ""),
    }
    return json.dumps(
        signed_fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def build_legacy_signed_reanchor_artifact(
    *,
    signer_did: str,
    constitution_sha256: str,
    private_key: ec.EllipticCurvePrivateKey,
    created_at: Optional[str] = None,
    reason: str = "",
    kid: str = "keys-1",
) -> dict[str, Any]:
    """Build a detached ECDSA-signed reanchor artifact for operator tooling."""
    artifact: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "version": ARTIFACT_VERSION,
        "signer": signer_did,
        "subject": "constitution_reanchor",
        "constitution_sha256": constitution_sha256,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    sig = suite.sign(canonical_amendment_bytes(artifact), private_key)
    artifact["signature"] = {
        "alg": ALG_ECDSA_SECP256K1_SHA256,
        "kid": kid,
        "sig": sig.hex(),
    }
    return artifact


def build_hybrid_signed_reanchor_artifact(
    *,
    signer_did: str,
    constitution_sha256: str,
    keypair: HybridKeypair,
    created_at: Optional[str] = None,
    reason: str = "",
    classical_kid: str = "key-1",
    pq_kid: str = "key-2",
) -> dict[str, Any]:
    """Build a detached hybrid-signed reanchor artifact for operator tooling."""
    artifact: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "version": ARTIFACT_VERSION,
        "signer": signer_did,
        "subject": "constitution_reanchor",
        "constitution_sha256": constitution_sha256,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    artifact["signatures"] = sign_hybrid(
        canonical_amendment_bytes(artifact),
        keypair,
        classical_kid=classical_kid,
        pq_kid=pq_kid,
    )
    return artifact


def verify_reanchor_artifact(
    artifact: Mapping[str, Any],
    *,
    trusted_did_document: Mapping[str, Any],
    expected_constitution_sha256: str,
) -> AmendmentArtifactVerification:
    """Verify a reanchor artifact against the trusted Sovereign root DID doc."""
    signer = str(artifact.get("signer") or "")
    constitution_sha256 = str(artifact.get("constitution_sha256") or "")
    trusted_did = str(trusted_did_document.get("id") or "")

    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        return AmendmentArtifactVerification(False, "unsupported artifact_type", signer)
    if artifact.get("version") != ARTIFACT_VERSION:
        return AmendmentArtifactVerification(False, "unsupported artifact version", signer)
    if artifact.get("subject") != "constitution_reanchor":
        return AmendmentArtifactVerification(
            False,
            "artifact subject is not constitution_reanchor",
            signer,
        )
    if not signer:
        return AmendmentArtifactVerification(False, "artifact has no signer DID")
    if signer != trusted_did:
        return AmendmentArtifactVerification(
            False,
            f"artifact signer {signer!r} is not trusted Sovereign DID {trusted_did!r}",
            signer,
            constitution_sha256,
        )
    if constitution_sha256 != expected_constitution_sha256:
        return AmendmentArtifactVerification(
            False,
            "artifact constitution_sha256 does not match the constitution bytes",
            signer,
            constitution_sha256,
        )

    data = canonical_amendment_bytes(artifact)
    if artifact.get("signatures"):
        result = verify_hybrid(
            data,
            artifact.get("signatures") or [],
            trusted_did_document.get("verificationMethod") or [],
            policy=VerifyPolicy.HYBRID_REQUIRED,
        )
        return AmendmentArtifactVerification(
            result.ok,
            (
                result.reason
                if result.ok
                else f"hybrid signature check failed: {result.reason}"
            ),
            signer,
            constitution_sha256,
        )

    signature = artifact.get("signature")
    if not isinstance(signature, Mapping):
        return AmendmentArtifactVerification(
            False,
            "artifact has no signature",
            signer,
            constitution_sha256,
        )
    if signature.get("alg") != ALG_ECDSA_SECP256K1_SHA256:
        return AmendmentArtifactVerification(
            False,
            "unsupported signature algorithm",
            signer,
            constitution_sha256,
        )

    public_keys = trusted_did_document.get("publicKey") or []
    kid = str(signature.get("kid") or "")
    trusted_key = None
    for key in public_keys:
        key_id = str(key.get("id") or "")
        if key_id.rsplit("#", 1)[-1] == kid:
            trusted_key = key
            break
    if trusted_key is None:
        return AmendmentArtifactVerification(
            False,
            f"trusted DID doc has no public key for kid {kid!r}",
            signer,
            constitution_sha256,
        )
    public_key_hex = trusted_key.get("publicKeyHex")
    if not public_key_hex:
        return AmendmentArtifactVerification(
            False,
            "trusted DID public key has no publicKeyHex",
            signer,
            constitution_sha256,
        )
    try:
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(),
            bytes.fromhex(public_key_hex),
        )
        sig_bytes = bytes.fromhex(str(signature.get("sig") or ""))
    except ValueError as exc:
        return AmendmentArtifactVerification(
            False,
            f"malformed signature material: {exc}",
            signer,
            constitution_sha256,
        )

    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    if not suite.verify(data, sig_bytes, public_key):
        return AmendmentArtifactVerification(
            False,
            "legacy ECDSA signature check failed",
            signer,
            constitution_sha256,
        )
    return AmendmentArtifactVerification(
        True,
        "signature valid (legacy ecdsa)",
        signer,
        constitution_sha256,
    )


def did_document_from_legacy_public_key(
    did: str,
    public_key: ec.EllipticCurvePublicKey,
) -> dict[str, Any]:
    """Build the legacy DID-doc shape used by inception for tests/tooling."""
    public_key_hex = public_key.public_bytes(
        encoding=Encoding.X962,
        format=PublicFormat.UncompressedPoint,
    ).hex()
    return {
        "id": did,
        "publicKey": [
            {
                "id": f"{did}#keys-1",
                "type": "EcdsaSecp256k1VerificationKey2019",
                "controller": did,
                "publicKeyHex": public_key_hex,
            }
        ],
    }
