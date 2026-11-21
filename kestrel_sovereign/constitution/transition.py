"""Declaration of intent for adoption of edition 2 under the v1 rules.

Publisher attestation is checked separately by release tooling. This verifies
that the pinned Sovereign and the target's Genesis key approved the very same
intent, including the publisher commit that the operator verified.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from kestrel_sovereign.identity import did_web
from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
from kestrel_sovereign.identity.portable_trust import PORTABLE_TRUST_VERSION, build_identity_trust_bundle
from kestrel_sovereign.identity.runtime_identity import RuntimeIdentityError, read_recorded_succession
from kestrel_sovereign.identity.succession_chain import (
    SuccessionChainError,
    build_chain,
    verify_artifact_against_chain,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256, ALG_ED25519, ALG_ML_DSA_65, ALG_SLH_DSA_SHA2_128S, get_suite,
)
from kestrel_sovereign.security.signature_verification import decode_verification_method_key

from .amendment_artifact import AmendmentArtifactError
from .emancipation import EmancipationContract, apply_emancipation
from .structure import ConstitutionVersionError, constitution_version, v2_sections

TRANSITION_PURPOSE = "kestrel.constitution.adopt.v2"
ROTATED_GENESIS_PROOF = "kestrel.genesis-succession.v1"
HISTORICAL_ARCHIVAL_SCHEME = "sphincs-sha2-128s-simple"
_V1_SOURCE_SHA256 = "6948454b61253548b02b943dff166ba0c2f2eaabe2f542ace37424a8256d6467"
_INTENT_FIELDS = {
    "purpose", "target_did", "sovereign_did", "prior_sha256",
    "constitution_sha256", "source_sha256", "publisher_commit",
}
_HASH = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GENESIS_DID = re.compile(r"did:pkh:eip155:1:(0x[0-9a-fA-F]{40})")
_GENESIS_KEY = re.compile(r"04[0-9a-f]{128}")
_GENESIS_SIGNATURE = re.compile(r"(?:[0-9a-f]{2}){1,72}")


def canonical_transition_intent_bytes(intent: Mapping[str, Any]) -> bytes:
    """Encode the exact declaration signed by the Genesis key."""
    return json.dumps(
        dict(intent), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def build_transition(
    intent: Mapping[str, Any],
    *,
    genesis_private_key: ec.EllipticCurvePrivateKey,
) -> dict[str, Any]:
    """Create the Genesis proof for inclusion in a Sovereign-signed artifact."""
    if not isinstance(genesis_private_key.curve, ec.SECP256K1):
        raise AmendmentArtifactError("The Genesis declaration requires a secp256k1 key")
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    key = genesis_private_key.public_key().public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint,
    )
    return {
        "intent": dict(intent),
        "genesis_proof": {
            "alg": ALG_ECDSA_SECP256K1_SHA256,
            "public_key_hex": key.hex(),
            "sig": suite.sign(canonical_transition_intent_bytes(intent), genesis_private_key).hex(),
        },
    }


def build_rotated_transition(
    intent: Mapping[str, Any], *, identity,
    archival_multibase: str, archival_scheme: str,
) -> dict[str, Any]:
    """Sign a declaration with the active keys of a Genesis-bound succession."""
    if not identity.succession_chain or not identity.hybrid_keypair:
        raise AmendmentArtifactError("The identity has no recorded hybrid succession")
    kids = identity.signing_key_ids()
    return {
        "intent": dict(intent),
        "genesis_proof": {
            "alg": ROTATED_GENESIS_PROOF,
            "signer_did": identity.signing_did,
            "trust": build_identity_trust_bundle(identity),
            "archival_multibase": archival_multibase,
            "archival_scheme": archival_scheme,
            "signatures": sign_hybrid(
                canonical_transition_intent_bytes(intent), identity.hybrid_keypair,
                classical_kid=kids[ALG_ED25519].rsplit("#", 1)[-1],
                pq_kid=kids[ALG_ML_DSA_65].rsplit("#", 1)[-1],
            ),
        },
    }


def verify_transition_adoption(
    *,
    artifact: Mapping[str, Any],
    trusted_did_document: Mapping[str, Any],
    target_did: str,
    prior_hash: str | None,
    prior_text: str | None,
    anchored_contract: EmancipationContract | None,
    new_text: str,
    new_hash: str,
    source_hash: str | None,
    identity_directory: Path | None,
) -> None:
    """Check the extra v1 authority before reanchor publishes any state.

    The caller already verified the external artifact signature, resolved the
    source through normal custody, and read the historical anchor. Unspecified
    legacy editions cannot establish which rules or promises govern adoption.
    """
    if prior_hash == new_hash or constitution_version(new_text) != 2:
        return
    if prior_text is None or not prior_hash or _digest(prior_text) != prior_hash:
        raise AmendmentArtifactError("Cannot adopt edition 2 without the exact prior governing bytes")
    try:
        if constitution_version(prior_text) == 2:
            v2_sections(prior_text)
            return
    except ConstitutionVersionError as exc:
        raise AmendmentArtifactError("Cannot establish the prior Constitution edition") from exc
    _verify_supported_v1_prior(prior_hash, anchored_contract)

    transition = artifact.get("transition")
    if not isinstance(transition, dict) or set(transition) != {"intent", "genesis_proof"}:
        raise AmendmentArtifactError(
            "Adoption under edition 1 requires a Genesis-verifiable Sovereign declaration of intent"
        )
    intent = transition["intent"]
    root_did = trusted_did_document.get("id")
    expected = {
        "purpose": TRANSITION_PURPOSE,
        "target_did": target_did,
        "sovereign_did": root_did,
        "prior_sha256": prior_hash,
        "constitution_sha256": new_hash,
        "source_sha256": source_hash,
    }
    if (
        not isinstance(intent, dict)
        or set(intent) != _INTENT_FIELDS
        or any(intent.get(key) != value for key, value in expected.items())
        or not _matches(_HASH, source_hash)
        or not _matches(_COMMIT, intent.get("publisher_commit"))
    ):
        raise AmendmentArtifactError(
            "Adoption declaration does not match this agent, its exact governing bytes, "
            "the pinned Sovereign/source or a full publisher commit"
        )
    if identity_directory is None:
        raise AmendmentArtifactError("Cannot establish the target's recorded Genesis authority")
    try:
        succession = read_recorded_succession(identity_directory)
    except (OSError, RuntimeIdentityError, ValueError, TypeError, KeyError) as exc:
        raise AmendmentArtifactError("Cannot read the target's recorded Genesis succession") from exc
    if _GENESIS_DID.fullmatch(target_did) is None:
        raise AmendmentArtifactError(
            "Edition 1 adoption supports only a Genesis did:pkh:eip155:1 identity"
        )

    proof = transition["genesis_proof"]
    if succession is None:
        _verify_genesis_proof(target_did, intent, proof, trusted_did_document)
    else:
        _verify_rotated_genesis_proof(target_did, intent, proof, trusted_did_document, succession)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _matches(pattern: re.Pattern[str], value: Any) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _verify_supported_v1_prior(
    prior_hash: str, anchored_contract: EmancipationContract | None,
) -> None:
    if anchored_contract is None or not anchored_contract.enabled:
        if prior_hash == _V1_SOURCE_SHA256:
            return
    else:
        try:
            base = (files("kestrel_sovereign") / "data" / "KESTREL_CONSTITUTION.md").read_bytes()
        except OSError as exc:
            raise AmendmentArtifactError("Cannot read the immutable edition 1 source") from exc
        if hashlib.sha256(base).hexdigest() == _V1_SOURCE_SHA256:
            reconstructed = apply_emancipation(base.decode("utf-8"), anchored_contract)
            if _digest(reconstructed) == prior_hash:
                return

    raise AmendmentArtifactError(
        "Cannot establish the prior edition 1 adoption authority or preserve its promises "
        "from these governing bytes and its authoritative emancipation receipt"
    )


def _verify_genesis_proof(
    target_did: str,
    intent: Mapping[str, Any],
    proof: Any,
    sovereign_document: Mapping[str, Any],
) -> None:
    match = _GENESIS_DID.fullmatch(target_did)
    assert match is not None
    if (
        not isinstance(proof, dict)
        or set(proof) != {"alg", "public_key_hex", "sig"}
        or proof["alg"] != ALG_ECDSA_SECP256K1_SHA256
        or not _matches(_GENESIS_KEY, proof["public_key_hex"])
        or not _matches(_GENESIS_SIGNATURE, proof["sig"])
    ):
        raise AmendmentArtifactError("Adoption Genesis proof is malformed")
    try:
        genesis_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), bytes.fromhex(proof["public_key_hex"]),
        )
    except ValueError as exc:
        raise AmendmentArtifactError("Adoption Genesis public key is malformed") from exc

    # Use the same address derivation used to generate the Genesis DID.
    from kestrel_sovereign.inception_service import public_key_to_ethereum_address

    if public_key_to_ethereum_address(genesis_key).lower() != match[1].lower():
        raise AmendmentArtifactError("Adoption Genesis public key does not bind to the target DID")
    _require_independent_legacy_key(genesis_key, sovereign_document)
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    if not suite.verify(canonical_transition_intent_bytes(intent), bytes.fromhex(proof["sig"]), genesis_key):
        raise AmendmentArtifactError("Adoption Genesis signature does not verify the declaration")


def _require_independent_legacy_key(genesis_key, sovereign_document) -> None:
    for method in sovereign_document.get("publicKey") or ():
        encoded = method.get("publicKeyHex")
        if not isinstance(encoded, str):
            continue
        try:
            sovereign_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(), bytes.fromhex(encoded),
            )
        except ValueError:
            continue
        if genesis_key.public_numbers() == sovereign_key.public_numbers():
            raise AmendmentArtifactError("The Genesis key cannot also serve as the external Sovereign key")


def _verify_rotated_genesis_proof(target_did, intent, proof, sovereign_document, succession) -> None:
    expected_trust = {
        "version": PORTABLE_TRUST_VERSION,
        "root_did": target_did,
        "root_verification_methods": list(succession.predecessor_verification_methods),
        "successions": [succession.to_dict()],
    }
    if (
        not isinstance(proof, dict)
        or set(proof) != {
            "alg", "signer_did", "trust", "signatures", "archival_multibase", "archival_scheme",
        }
        or proof["alg"] != ROTATED_GENESIS_PROOF
        or succession.predecessor_did != target_did
        or proof["signer_did"] != succession.successor_did
        or proof["trust"] != expected_trust
    ):
        raise AmendmentArtifactError(
            "Adoption Genesis proof must match this target's recorded succession and its active signer"
        )
    signatures = proof["signatures"]
    if (
        not isinstance(signatures, list)
        or len(signatures) != 2
        or any(not isinstance(signature, dict) for signature in signatures)
        or not all(any(signature.get("alg") == alg for signature in signatures)
                   for alg in (ALG_ED25519, ALG_ML_DSA_65))
    ):
        raise AmendmentArtifactError("Adoption Genesis succession requires both hybrid signature algorithms")

    archival_scheme = proof["archival_scheme"]
    archival_pin = proof["archival_multibase"]
    if not isinstance(archival_pin, str) or not archival_pin:
        raise AmendmentArtifactError("Adoption Genesis succession requires a pinned archival key")
    if archival_scheme == HISTORICAL_ARCHIVAL_SCHEME:
        try:
            from .historical_archival import verify_historical_archival_signature
        except ImportError as exc:
            raise AmendmentArtifactError("Historical succession archival verifier is unavailable") from exc
        historical_verifier = verify_historical_archival_signature
    elif archival_scheme == ALG_SLH_DSA_SHA2_128S:
        historical_verifier = None
    else:
        raise AmendmentArtifactError("Unsupported adoption succession archival algorithm")

    # Resolve the successor independently through its published did:web
    # document. Its keys must agree with the exact recorded local ceremony.
    try:
        published = did_web.resolve(succession.successor_did)
    except (did_web.DidWebError, OSError, ValueError) as exc:
        raise AmendmentArtifactError("Cannot resolve the published successor DID document") from exc

    def published_successor(did: str) -> dict:
        if did != succession.successor_did:
            raise SuccessionChainError("No independent publication for succession DID")
        return published

    try:
        result = verify_artifact_against_chain(
            root_did=target_did,
            root_verification_methods=succession.predecessor_verification_methods,
            chain=build_chain([succession]),
            artifact_timestamp=datetime.now(timezone.utc).isoformat(),
            artifact_payload=canonical_transition_intent_bytes(intent),
            artifact_signatures=signatures,
            did_web_resolver=published_successor,
            require_archival=True,
            trusted_archival_multibase=archival_pin,
            historical_archival_verifier=historical_verifier,
        )
    except (SuccessionChainError, ValueError, TypeError, KeyError) as exc:
        raise AmendmentArtifactError("Cannot verify the target's recorded Genesis succession") from exc
    if not result.ok or result.active_identity.did != proof["signer_did"]:
        raise AmendmentArtifactError(f"Adoption Genesis succession verification failed: {result.reason}")
    lineage = (
        *succession.predecessor_verification_methods,
        *succession.successor_verification_methods,
    )
    sovereign_keys = {
        method.get("publicKeyMultibase") for method in sovereign_document.get("verificationMethod") or ()
    }
    for method in lineage:
        if method.get("publicKeyMultibase") in sovereign_keys:
            raise AmendmentArtifactError("The Genesis lineage cannot also serve as the external Sovereign key")
        decoded, _ = decode_verification_method_key(method)
        if decoded is not None and decoded[0].alg_id == ALG_ECDSA_SECP256K1_SHA256:
            _require_independent_legacy_key(decoded[1], sovereign_document)
