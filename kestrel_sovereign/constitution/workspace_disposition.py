"""Separate Sovereign authorization for a preserved legacy filesystem policy."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .amendment_artifact import AmendmentArtifactError, verify_detached_signature
from .reanchor_receipt import is_constitution_hash

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric import ec
    from kestrel_sovereign.features.computer_use.policy import PathPolicy

WORKSPACE_DISPOSITION_FIELD = "workspace_disposition"
PURPOSE = "kestrel.constitution.preserve-workspace-policy.v1"
_FIELDS = {
    "version", "purpose", "signer", "target_did", "prior_sha256", "constitution_sha256",
    "grant_baseline_sha256", "path_policy", "created_at",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class WorkspacePathPolicy:
    allow_sha256: frozenset[str]
    deny_sha256: frozenset[str]
    auto_approve_read: bool

    @classmethod
    def from_policy(cls, policy: PathPolicy) -> WorkspacePathPolicy:
        from kestrel_sovereign.features.computer_use.path_safety import path_pattern_identity

        def fingerprints(patterns, *, require_absolute=False):
            if require_absolute and any(not Path(pattern).expanduser().is_absolute() for pattern in patterns):
                raise AmendmentArtifactError("Signed workspace policy requires absolute native allow selectors")
            return frozenset(_fingerprint(path_pattern_identity(pattern)) for pattern in patterns)

        return cls(
            fingerprints(policy.allow, require_absolute=True),
            fingerprints(policy.deny),
            policy.auto_approve_read,
        )

    @classmethod
    def parse(cls, evidence: Any) -> WorkspacePathPolicy:
        if not isinstance(evidence, dict) or set(evidence) != {
            "allow_sha256", "deny_sha256", "auto_approve_read",
        } or type(evidence["auto_approve_read"]) is not bool:
            raise AmendmentArtifactError("Malformed signed workspace policy evidence")
        for key in ("allow_sha256", "deny_sha256"):
            hashes = evidence[key]
            if (
                not isinstance(hashes, list)
                or any(not is_constitution_hash(value) for value in hashes)
                or hashes != sorted(set(hashes))
            ):
                raise AmendmentArtifactError("Malformed signed workspace selector fingerprints")
        return cls(
            frozenset(evidence["allow_sha256"]), frozenset(evidence["deny_sha256"]),
            evidence["auto_approve_read"],
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "allow_sha256": sorted(self.allow_sha256), "deny_sha256": sorted(self.deny_sha256),
            "auto_approve_read": self.auto_approve_read,
        }

    def permits(self, current: WorkspacePathPolicy) -> bool:
        """A config may narrow signed selectors and policy without new authority."""
        return (
            current.allow_sha256 <= self.allow_sha256
            and current.deny_sha256 >= self.deny_sha256
            and (self.auto_approve_read or not current.auto_approve_read)
        )


@dataclass(frozen=True)
class WorkspaceDisposition:
    prior_sha256: str
    path_policy: WorkspacePathPolicy
    sha256: str
    signer: str

    def same_scope(self, other: WorkspaceDisposition) -> bool:
        return self.prior_sha256 == other.prior_sha256 and self.path_policy == other.path_policy


def workspace_path_policy_evidence(computer_use_config: dict) -> dict[str, Any]:
    """Fingerprint the selectors and approval mode used by native ComputerUse."""
    from kestrel_sovereign.features.computer_use.policy import path_policy_from_config

    return WorkspacePathPolicy.from_policy(path_policy_from_config(computer_use_config)).evidence()


def build_legacy_signed_workspace_disposition(
    *, signer_did: str, private_key: ec.EllipticCurvePrivateKey, target_did: str,
    prior_sha256: str, constitution_sha256: str, grant_baseline: Mapping[str, Any],
    path_policy: dict[str, Any], created_at: str | None = None, kid: str = "keys-1",
) -> dict[str, Any]:
    """Sign the independent decision to retain legacy filesystem policy and queue."""
    from kestrel_sovereign.security.crypto_suite import ALG_ECDSA_SECP256K1_SHA256, get_suite

    proof = {
        "version": 1, "purpose": PURPOSE, "signer": signer_did, "target_did": target_did,
        "prior_sha256": prior_sha256, "constitution_sha256": constitution_sha256,
        "grant_baseline_sha256": _fingerprint(dict(grant_baseline)),
        "path_policy": WorkspacePathPolicy.parse(path_policy).evidence(),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    signature = suite.sign(_canonical(proof), private_key)
    return {**proof, "signature": {
        "alg": ALG_ECDSA_SECP256K1_SHA256, "kid": kid, "sig": signature.hex(),
    }}


def verify_workspace_disposition(
    proof: Any, *, trusted_did_document: Mapping[str, Any], target_did: str,
    constitution_sha256: str, grant_baseline: Mapping[str, Any],
    prior_sha256: str | None = None, artifact_created_at: str | None = None,
) -> WorkspaceDisposition:
    """Verify the separate signature and bind it to the exact migration."""
    if (
        not isinstance(proof, dict)
        or set(proof) not in (_FIELDS | {"signature"}, _FIELDS | {"signatures"})
        or type(proof["version"]) is not int or proof["version"] != 1
        or proof["purpose"] != PURPOSE or proof["target_did"] != target_did
        or proof["signer"] != trusted_did_document.get("id") or proof["signer"] == target_did
        or proof["constitution_sha256"] != constitution_sha256
        or not is_constitution_hash(proof["prior_sha256"])
        or (prior_sha256 is not None and proof["prior_sha256"] != prior_sha256)
        or proof["grant_baseline_sha256"] != _fingerprint(dict(grant_baseline))
        or "filesystem_outside_workspace" in grant_baseline["grants"]
        or not {"filesystem_read", "filesystem_write"}.intersection(grant_baseline["grants"])
    ):
        raise AmendmentArtifactError("Workspace disposition does not match the agent, baseline or governing bytes")
    try:
        created = datetime.fromisoformat(proof["created_at"])
        admitted = datetime.fromisoformat(artifact_created_at) if artifact_created_at is not None else datetime.now(timezone.utc)
        if created.utcoffset() is None or admitted.utcoffset() is None or not (
            created <= admitted <= datetime.now(timezone.utc)
        ):
            raise ValueError("invalid chronology")
    except (TypeError, ValueError) as exc:
        raise AmendmentArtifactError("Invalid timestamp for the signed workspace disposition") from exc

    policy = WorkspacePathPolicy.parse(proof["path_policy"])
    ok, reason = verify_detached_signature(
        _canonical({key: proof[key] for key in _FIELDS}), proof,
        trusted_did_document=trusted_did_document,
    )
    if not ok:
        raise AmendmentArtifactError(f"Workspace disposition failed signature verification: {reason}")
    return WorkspaceDisposition(proof["prior_sha256"], policy, _fingerprint(proof), proof["signer"])
