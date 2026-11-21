"""Declaration of intent for adoption of edition 2 under the v1 rules.

Admission verifies the archived publisher attestation and both authorities
over the same adoption intent. The stored proof can be replayed independently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from kestrel_sovereign.identity import did_web
from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid, verify_hybrid
from kestrel_sovereign.identity.portable_trust import PORTABLE_TRUST_VERSION, build_identity_trust_bundle
from kestrel_sovereign.identity.runtime_identity import (
    RuntimeIdentityError, read_born_hybrid_genesis_document, read_legacy_only_genesis_document,
    read_recorded_archival_pin, read_recorded_succession,
)
from kestrel_sovereign.identity.succession import HISTORICAL_ARCHIVAL_ALGORITHM, SuccessionStatement
from kestrel_sovereign.identity.succession_chain import (
    SuccessionChainError,
    build_chain,
    verify_artifact_against_chain,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256, ALG_ED25519, ALG_ML_DSA_65, ALG_SLH_DSA_SHA2_128S, get_suite,
)
from kestrel_sovereign.security.exceptions import KeyStorageError
from kestrel_sovereign.security.signature_verification import decode_verification_method_key

from .amendment_artifact import AmendmentArtifactError, verify_reanchor_artifact
from .emancipation import (
    EmancipationContract, historic_emancipation_displacement,
    known_historical_platform_promise, matches_recorded_emancipation_contract,
)
from .structure import ConstitutionVersionError, constitution_version, v2_sections
from .v1_prior import verify_supported_v1_prior as _verify_supported_v1_prior

TRANSITION_PURPOSE = "kestrel.constitution.adopt.v2"
ROTATED_GENESIS_PROOF = "kestrel.genesis-succession.v1"
BORN_HYBRID_GENESIS_PROOF = "kestrel.genesis-hybrid.v1"
_GENESIS_EVIDENCE_KEY = "genesis_admission_evidence"
HISTORICAL_ARCHIVAL_SCHEME = HISTORICAL_ARCHIVAL_ALGORITHM
_INTENT_FIELDS = {
    "purpose", "target_did", "sovereign_did", "prior_sha256",
    "constitution_sha256", "source_sha256", "publisher_commit",
    "publisher_source_url", "admission_at",
}
_MAX_DID_DOCUMENT_BYTES = 64 * 1024
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


def _sign_hybrid_intent(intent, identity):
    kids = identity.signing_key_ids()
    return sign_hybrid(
        canonical_transition_intent_bytes(intent), identity.hybrid_keypair,
        classical_kid=kids[ALG_ED25519].rsplit("#", 1)[-1],
        pq_kid=kids[ALG_ML_DSA_65].rsplit("#", 1)[-1],
    )


def build_born_hybrid_transition(intent, *, identity, published_did_document: str) -> dict:
    """Sign a declaration with both keys from a born-hybrid Genesis identity."""
    if not identity.is_born_hybrid or identity.signing_did != intent.get("target_did"):
        raise AmendmentArtifactError("The signer is not this target's born-hybrid Genesis identity")
    return {
        "intent": dict(intent),
        "genesis_proof": {
            "alg": BORN_HYBRID_GENESIS_PROOF,
            "published_did_document": published_did_document,
            "signatures": _sign_hybrid_intent(intent, identity),
        },
    }


def build_rotated_transition(
    intent: Mapping[str, Any], *, identity,
    archival_multibase: str, archival_scheme: str, published_did_document: str,
) -> dict[str, Any]:
    """Sign a declaration with the active keys of a Genesis-bound succession."""
    if not identity.succession_chain or not identity.hybrid_keypair:
        raise AmendmentArtifactError("The identity has no recorded hybrid succession")
    return {
        "intent": dict(intent),
        "genesis_proof": {
            "alg": ROTATED_GENESIS_PROOF,
            "signer_did": identity.signing_did,
            "trust": build_identity_trust_bundle(identity),
            "archival_multibase": archival_multibase,
            "archival_scheme": archival_scheme,
            "published_did_document": published_did_document,
            "signatures": _sign_hybrid_intent(intent, identity),
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
    hosted_identity_evidence_path: Path | None = None,
    active_signing_did: str | None = None,
    prior_role_artifact_bytes: bytes | None = None,
    verified_prior_anchor: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Check the extra v1 authority before reanchor publishes any state.

    The caller already verified the external artifact signature, resolved the
    source through normal custody, and read the historical anchor. Unspecified
    legacy editions cannot establish which rules or promises govern adoption.
    """
    if constitution_version(new_text) != 2:
        from .legacy_authority import verify_legacy_fixed_guarantee_change

        return verify_legacy_fixed_guarantee_change(
            artifact, sovereign_document=trusted_did_document, target_did=target_did,
            prior_hash=prior_hash, prior_text=prior_text, new_hash=new_hash, new_text=new_text,
            identity_directory=identity_directory,
            hosted_identity_evidence_path=hosted_identity_evidence_path,
            active_signing_did=active_signing_did,
        )
    _verify_adoption_publisher(artifact, source_hash)
    if prior_hash == new_hash:
        if verified_prior_anchor != (target_did, prior_hash):
            raise AmendmentArtifactError("Prior edition 2 requires a verified publisher-backed governing anchor")
        verify_publication_chronology(artifact, datetime.now(timezone.utc))
        return {}
    if prior_text is None or not prior_hash or _digest(prior_text) != prior_hash:
        raise AmendmentArtifactError("Cannot adopt edition 2 without the exact prior governing bytes")
    try:
        if constitution_version(prior_text) == 2:
            v2_sections(prior_text)
            if verified_prior_anchor != (target_did, prior_hash):
                raise AmendmentArtifactError("Prior edition 2 requires a verified publisher-backed governing anchor")
            verify_publication_chronology(artifact, datetime.now(timezone.utc))
            return {}
    except ConstitutionVersionError as exc:
        raise AmendmentArtifactError("Cannot establish the prior Constitution edition") from exc
    prior_authority = _verify_supported_v1_prior(
        prior_hash, anchored_contract, prior_text=prior_text,
        prior_role_artifact_bytes=prior_role_artifact_bytes, trusted_did_document=trusted_did_document,
    )
    _verify_earlier_promise_survives(prior_hash, prior_text, new_hash, new_text)

    intent, proof = _adoption_evidence(
        artifact, trusted_did_document, target_did, prior_hash, new_hash, source_hash,
    )
    verify_publication_chronology(artifact, datetime.now(timezone.utc), intent=intent)
    genesis_evidence = verify_live_genesis_proof(
        target_did, intent, proof, trusted_did_document, identity_directory,
        hosted_identity_evidence_path=hosted_identity_evidence_path, active_signing_did=active_signing_did,
    )
    return prior_authority | genesis_evidence


def _read_hosted_identity_evidence(path, target_did):
    """Read operator-selected public custody, independent of an adoption artifact."""
    if not path.is_absolute() or path.resolve() != path:
        raise AmendmentArtifactError("Hosted Genesis evidence requires an absolute operator custody path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.geteuid())
            or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise AmendmentArtifactError("Hosted Genesis evidence must be an operator-owned regular file")
        raw = source.read(_MAX_DID_DOCUMENT_BYTES + 1)
    if len(raw) > _MAX_DID_DOCUMENT_BYTES:
        raise AmendmentArtifactError("Hosted public Genesis evidence exceeds the size limit")
    evidence = json.loads(raw)
    if not isinstance(evidence, dict) or evidence.get("target_did") != target_did:
        raise AmendmentArtifactError("Hosted public Genesis evidence does not identify this target")
    state = evidence.get("state")
    expected = {
        "legacy-only": {"genesis_document"},
        "born-hybrid": {"genesis_document"},
        "rotated": {"succession", "archival_multibase"},
    }.get(state) if isinstance(state, str) else None
    if (
        expected is None or set(evidence) != expected | {"version", "target_did", "state"}
        or type(evidence["version"]) is not int or evidence["version"] != 1
    ):
        raise AmendmentArtifactError("Hosted public Genesis evidence is malformed or ambiguous")
    return evidence


def verify_live_genesis_proof(
    target_did, intent, proof, sovereign_document, identity_directory, *,
    hosted_identity_evidence_path=None, active_signing_did=None,
) -> dict:
    """Compare declaration evidence against independently selected identity custody."""
    if (identity_directory is None) == (hosted_identity_evidence_path is None):
        raise AmendmentArtifactError("Specify exactly one native or hosted public Genesis custody source")
    succession = None
    document = None
    pin = None
    try:
        if hosted_identity_evidence_path is not None:
            custody = _read_hosted_identity_evidence(hosted_identity_evidence_path, target_did)
            state = custody["state"]
            if state == "rotated":
                succession = SuccessionStatement.from_dict(custody["succession"])
                pin = custody["archival_multibase"]
            else:
                if (state == "born-hybrid") != target_did.startswith("did:web:"):
                    raise AmendmentArtifactError("Hosted public Genesis custody state conflicts with the target DID")
                document = custody["genesis_document"]
        else:
            succession = read_recorded_succession(identity_directory)
            if succession is not None:
                pin = read_recorded_archival_pin(identity_directory, succession)
            elif isinstance(active_signing_did, str) and active_signing_did != target_did:
                raise AmendmentArtifactError("Active signing DID differs from Genesis; restore the recorded public succession")
            elif target_did.startswith("did:web:"):
                document = read_born_hybrid_genesis_document(identity_directory, target_did)
            else:
                document = read_legacy_only_genesis_document(identity_directory, target_did)
    except AmendmentArtifactError:
        raise
    except RuntimeIdentityError as exc:
        raise AmendmentArtifactError(f"Cannot read independent Genesis or archival public key custody: {exc}") from exc
    except (OSError, KeyStorageError, ValueError, TypeError, KeyError) as exc:
        raise AmendmentArtifactError("Cannot read independent Genesis or archival public key custody") from exc

    if succession is not None:
        if _GENESIS_DID.fullmatch(target_did) is None:
            raise AmendmentArtifactError("This Genesis DID has an unsupported or mixed succession history")
        if not isinstance(pin, str) or not pin:
            raise AmendmentArtifactError("Independent archival public key custody is missing")
        if isinstance(active_signing_did, str) and succession.successor_did != active_signing_did:
            raise AmendmentArtifactError("Recorded public succession conflicts with the active signing DID")
        return _verify_rotated_genesis_proof(
            target_did, intent, proof, sovereign_document, succession, custody_pin=pin,
        )
    if target_did.startswith("did:web:"):
        if not isinstance(document, dict):
            raise AmendmentArtifactError("Independently recorded born-hybrid Genesis DID document is missing")
        return _verify_born_hybrid_genesis(
            target_did, intent, proof, sovereign_document, independent_document=document,
        )
    _verify_genesis_proof(target_did, intent, proof, sovereign_document)
    methods = (document.get("publicKey") or document.get("verificationMethod")) if isinstance(document, dict) else None
    if (
        not isinstance(document, dict) or document.get("id") != target_did
        or not isinstance(methods, list) or len(methods) != 1 or not isinstance(methods[0], dict)
        or not isinstance(methods[0].get("publicKeyHex"), str)
    ):
        raise AmendmentArtifactError("Direct proof differs from the independently retained Genesis public key")
    try:
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), bytes.fromhex(methods[0]["publicKeyHex"]),
        )
    except ValueError as exc:
        raise AmendmentArtifactError("Independently retained Genesis public key is malformed") from exc
    if public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint).hex() != proof["public_key_hex"]:
        raise AmendmentArtifactError("Direct proof differs from the independently retained Genesis public key")
    return {}


def _verify_adoption_publisher(artifact: Mapping[str, Any], source_hash: str | None) -> None:
    from .publisher import verify_recorded_publisher

    expected = verify_recorded_publisher(artifact.get("publisher_release"), source_hash)
    transition = artifact.get("transition")
    if isinstance(transition, dict) and isinstance(transition.get("intent"), dict):
        if transition["intent"].get("publisher_commit") != expected["publisher_commit"]:
            raise AmendmentArtifactError("Genesis declaration names a different publisher commit")


def publisher_source_url(commit: str) -> str:
    """Return the immutable URL named by the publisher's official source tree."""
    return (
        "https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/"
        f"{commit}/kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md"
    )


def _admission_time(value: Any) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise AmendmentArtifactError("Adoption requires an ISO-8601 UTC admission_at")
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AmendmentArtifactError("Adoption requires an ISO-8601 UTC admission_at") from exc
    if stamp.utcoffset() != timedelta(0):
        raise AmendmentArtifactError("Adoption admission_at must be UTC")
    return stamp


def verify_publication_chronology(artifact, recorded_at: datetime, *, intent=None) -> None:
    """Require publisher release, authorization, and native record in that order."""
    issued = _admission_time(artifact["publisher_release"].get("issued_at"))
    signed = _admission_time(artifact.get("created_at"))
    admission = _admission_time(intent["admission_at"]) if intent is not None else signed
    if issued > admission:
        raise AmendmentArtifactError("Adoption declaration predates the independently witnessed publisher release")
    if admission > signed or signed > recorded_at or recorded_at > datetime.now(timezone.utc):
        raise AmendmentArtifactError("Adoption declaration or its outer signature postdates its recorded authorization")


def verify_reanchor_receipt_chronology(artifact, receipt) -> None:
    verify_publication_chronology(artifact, _admission_time(receipt.get("timestamp")))


def _adoption_evidence(artifact, root, target_did, prior_hash, new_hash, source_hash):
    transition = artifact.get("transition")
    if not isinstance(transition, dict) or set(transition) != {"intent", "genesis_proof"}:
        raise AmendmentArtifactError(
            "Adoption under edition 1 requires a Genesis-verifiable Sovereign declaration of intent"
        )
    intent = transition["intent"]
    expected = {
        "purpose": TRANSITION_PURPOSE,
        "target_did": target_did,
        "sovereign_did": root.get("id"),
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
        or intent["publisher_source_url"] != publisher_source_url(intent["publisher_commit"])
    ):
        raise AmendmentArtifactError(
            "Adoption declaration does not match this agent, its exact governing bytes, "
            "the pinned Sovereign/source or the immutable publisher source URL"
        )
    if _GENESIS_DID.fullmatch(target_did) is None and not target_did.startswith("did:web:"):
        raise AmendmentArtifactError(
            "Edition 1 adoption requires a supported Genesis DID and custody evidence"
        )
    return intent, transition["genesis_proof"]


def verify_archived_transition_adoption(
    *,
    artifact_bytes: bytes,
    receipt: Mapping[str, Any],
    trusted_did_document: Mapping[str, Any],
    target_did: str,
    prior_hash: str,
    prior_text: str,
    anchored_contract: EmancipationContract | None,
    new_text: str,
    new_hash: str,
    source_hash: str,
    prior_role_artifact_bytes: bytes | None = None,
) -> None:
    """Replay a stored adoption without consulting present lineage or HTTPS.

    The caller supplies the exact retained native receipt, source and prior
    governing bytes. Publisher archive verification remains a separate proof.
    """
    if (
        receipt.get("signed_artifact_hash") != hashlib.sha256(artifact_bytes).hexdigest()
        or receipt.get("old_hash") != prior_hash
        or receipt.get("new_hash") != new_hash
        or receipt.get("source_content_sha256") != source_hash
        or receipt.get("signed_artifact_signer") != trusted_did_document.get("id")
        or _digest(prior_text) != prior_hash
        or _digest(new_text) != new_hash
        or constitution_version(new_text) != 2
    ):
        raise AmendmentArtifactError("Archived adoption evidence does not match the native receipt and governing bytes")
    prior_authority = _verify_supported_v1_prior(
        prior_hash, anchored_contract, prior_text=prior_text,
        prior_role_artifact_bytes=prior_role_artifact_bytes, trusted_did_document=trusted_did_document,
    )
    if receipt.get("prior_role_artifact_sha256") != prior_authority.get("prior_role_artifact_sha256"):
        raise AmendmentArtifactError("Archived Book IV admission artifact differs from the native receipt")
    _verify_earlier_promise_survives(prior_hash, prior_text, new_hash, new_text)
    if (
        anchored_contract is not None and anchored_contract.enabled
        and not matches_recorded_emancipation_contract(new_text, anchored_contract)
    ):
        raise AmendmentArtifactError("Archived adoption does not preserve the proven prior emancipation terms")
    try:
        artifact = json.loads(artifact_bytes)
        if not isinstance(artifact, dict):
            raise ValueError("not an object")
        verification = verify_reanchor_artifact(
            artifact, trusted_did_document=trusted_did_document,
            expected_constitution_sha256=new_hash,
        )
    except (ValueError, TypeError) as exc:
        raise AmendmentArtifactError("Archived adoption artifact is malformed") from exc
    if not verification.ok:
        raise AmendmentArtifactError(f"Archived Sovereign declaration failed: {verification.reason}")
    _verify_adoption_publisher(artifact, source_hash)
    intent, proof = _adoption_evidence(
        artifact, trusted_did_document, target_did, prior_hash, new_hash, source_hash,
    )
    verify_publication_chronology(artifact, _admission_time(receipt.get("timestamp")), intent=intent)
    if isinstance(proof, dict) and proof.get("alg") == BORN_HYBRID_GENESIS_PROOF:
        evidence = _verify_born_hybrid_genesis(
            target_did, intent, proof, trusted_did_document, independent_document=None,
        )
        _verify_archived_genesis_evidence(receipt, evidence)
        return
    if not isinstance(proof, dict) or proof.get("alg") != ROTATED_GENESIS_PROOF:
        _verify_genesis_proof(target_did, intent, proof, trusted_did_document)
        return
    try:
        statements = proof["trust"]["successions"]
        if not isinstance(statements, list) or len(statements) != 1:
            raise ValueError("one recorded runtime succession required")
        succession = SuccessionStatement.from_dict(statements[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise AmendmentArtifactError("Archived Genesis succession is malformed") from exc
    evidence = _verify_rotated_genesis_proof(
        target_did, intent, proof, trusted_did_document, succession, custody_pin=None,
    )
    _verify_archived_genesis_evidence(receipt, evidence)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _matches(pattern: re.Pattern[str], value: Any) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _verify_earlier_promise_survives(prior_hash: str, prior_text: str, new_hash: str, new_text: str) -> None:
    if known_historical_platform_promise(prior_hash) and historic_emancipation_displacement(
        previous_hash=prior_hash, previous_text=prior_text, proposed_hash=new_hash, proposed_text=new_text,
    ):
        raise AmendmentArtifactError("Edition 2 adoption must preserve the original historical emancipation promise")


def _verify_genesis_proof(
    target_did: str,
    intent: Mapping[str, Any],
    proof: Any,
    sovereign_document: Mapping[str, Any],
) -> None:
    match = _GENESIS_DID.fullmatch(target_did)
    if match is None:
        raise AmendmentArtifactError("The classical Genesis proof requires did:pkh:eip155:1")
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


def _verify_rotated_genesis_proof(
    target_did, intent, proof, sovereign_document, succession, *, custody_pin: str | None,
) -> dict:
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
            "published_did_document",
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
    if custody_pin is not None and archival_pin != custody_pin:
        raise AmendmentArtifactError("The signed archival pin differs from the target's public key custody")
    # Replay reaches this branch only after verifying the external Sovereign
    # signature over the exact event artifact, including the admitted pin.
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

    snapshot = proof["published_did_document"]
    if not isinstance(snapshot, str):
        raise AmendmentArtifactError("Adoption requires exact text of the published successor DID document")
    try:
        raw = snapshot.encode("utf-8")
        if not 0 < len(raw) <= _MAX_DID_DOCUMENT_BYTES:
            raise AmendmentArtifactError("Published successor DID document has invalid size")
        published = did_web.parse_resolved_document(succession.successor_did, raw)
        if custody_pin is not None:
            now_bytes, _document = did_web.resolve_document(succession.successor_did)
            if now_bytes != raw:
                raise AmendmentArtifactError("Current HTTPS successor DID document differs from the signed snapshot")
    except (did_web.DidWebError, UnicodeError, OSError) as exc:
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
            artifact_timestamp=intent["admission_at"],
            artifact_payload=canonical_transition_intent_bytes(intent),
            artifact_signatures=signatures,
            did_web_resolver=published_successor,
            require_archival=True,
            trusted_archival_multibase=archival_pin,
            historical_archival_verifier=historical_verifier,
        )
    except (SuccessionChainError, ValueError, TypeError, KeyError) as exc:
        raise AmendmentArtifactError(f"Cannot verify the target's recorded Genesis succession: {exc}") from exc
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

    return {_GENESIS_EVIDENCE_KEY: {
        "proof": ROTATED_GENESIS_PROOF, "archival_multibase": archival_pin,
        "published_document_sha256": hashlib.sha256(raw).hexdigest(),
    }}


def _verify_archived_genesis_evidence(receipt, evidence):
    if receipt.get(_GENESIS_EVIDENCE_KEY) != evidence[_GENESIS_EVIDENCE_KEY]:
        raise AmendmentArtifactError("Archived Genesis evidence differs from recorded admission custody")


def _born_genesis_methods(target_did, document):
    parsed = did_web.parse_did_document(document)
    eligible = {
        kid: suite.alg_id for kid, suite, _ in parsed
        if suite.alg_id in (ALG_ED25519, ALG_ML_DSA_65)
    }
    methods = [method for method in document["verificationMethod"] if method["id"] in eligible]
    if (
        len(methods) != 2 or set(eligible.values()) != {ALG_ED25519, ALG_ML_DSA_65}
        or any(method.get("controller") != target_did
               or not method["id"].startswith(target_did + "#") for method in methods)
    ):
        raise AmendmentArtifactError("Genesis DID document must bind exactly one signature method per hybrid algorithm")
    return sorted(methods, key=lambda method: method["id"])


def _verify_born_hybrid_genesis(target_did, intent, proof, sovereign_document, *, independent_document):
    if (
        not target_did.startswith("did:web:") or not isinstance(proof, dict)
        or set(proof) != {"alg", "published_did_document", "signatures"}
        or proof["alg"] != BORN_HYBRID_GENESIS_PROOF
    ):
        raise AmendmentArtifactError("Born-hybrid adoption requires its published Genesis DID and both signatures")
    snapshot = proof["published_did_document"]
    try:
        raw = snapshot.encode("utf-8") if isinstance(snapshot, str) else b""
        if not 0 < len(raw) <= _MAX_DID_DOCUMENT_BYTES:
            raise AmendmentArtifactError("Published Genesis DID document has invalid size")
        published = did_web.parse_resolved_document(target_did, raw)
        methods = _born_genesis_methods(target_did, published)
        if independent_document is not None:
            current, _ = did_web.resolve_document(target_did)
            if (independent_document.get("id") != target_did or current != raw
                    or _born_genesis_methods(target_did, independent_document) != methods):
                raise AmendmentArtifactError("The published Genesis keys differ from current HTTPS or inception custody")
    except (did_web.DidWebError, OSError, RuntimeIdentityError, UnicodeError, KeyError) as exc:
        raise AmendmentArtifactError("Cannot establish the independently recorded born-hybrid Genesis DID document") from exc
    sovereign_keys = {
        method.get("publicKeyMultibase") for method in sovereign_document.get("verificationMethod") or ()
    }
    if target_did == sovereign_document.get("id") or any(
        method["publicKeyMultibase"] in sovereign_keys for method in methods
    ):
        raise AmendmentArtifactError("The Genesis keys cannot also serve as the external Sovereign key")
    signatures = proof["signatures"]
    if not isinstance(signatures, list) or len(signatures) != 2:
        raise AmendmentArtifactError("Born-hybrid Genesis requires both signatures")
    result = verify_hybrid(canonical_transition_intent_bytes(intent), signatures, methods)
    if not result.ok:
        raise AmendmentArtifactError(f"Born-hybrid Genesis declaration failed: {result.reason}")
    return {_GENESIS_EVIDENCE_KEY: {
        "proof": BORN_HYBRID_GENESIS_PROOF,
        "published_document_sha256": hashlib.sha256(raw).hexdigest(),
    }}
