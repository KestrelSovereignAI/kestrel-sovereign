"""Separate Sovereign authorization for a preserved legacy filesystem policy."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tomllib
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

        def fingerprints(patterns):
            # Use the native identity even for relative selectors. An existing
            # prefix resolves to its absolute path. A missing prefix retains
            # the relative literal until it starts matching an absolute path.
            return frozenset(_fingerprint(path_pattern_identity(pattern)) for pattern in patterns)

        return cls(fingerprints(policy.allow), fingerprints(policy.deny), policy.auto_approve_read)

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


def configured_workspace_policy(storage_path, *, hosted_config_path: Path | None = None) -> dict[str, Any]:
    """Report the disk-selected ComputerUse policy without loading the feature."""
    from kestrel_sovereign.features.computer_use.policy import load_computer_use_config

    if storage_path is None:
        if hosted_config_path is None:
            raise AmendmentArtifactError("Hosted grant cutover requires an operator-selected ComputerUse configuration")
        config = _read_hosted_computer_use_config(hosted_config_path)
    else:
        config = load_computer_use_config(storage_path)
    if not config.get("enabled"):
        return {"enabled": False}
    return {"enabled": True, "path_policy": workspace_path_policy_evidence(config), "backend": config.get("backend", "docker")}


def _read_hosted_computer_use_config(path: Path) -> dict[str, Any]:
    """Read the host's independently selected native TOML policy without fallback."""
    try:
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError("expected an absolute canonical path")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid not in (0, os.geteuid())
                or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise ValueError("expected an operator-owned regular file")
            limit = 1024 * 1024
            raw = source.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("configuration exceeds size limit")
        config = tomllib.loads(raw.decode())["features"]["computer_use"]
        if not isinstance(config, dict) or type(config.get("enabled")) is not bool:
            raise ValueError("expected an explicit ComputerUse enabled setting")
        for key in ("allowed_paths", "deny_paths"):
            if key in config and (
                not isinstance(config[key], list) or any(not isinstance(path, str) for path in config[key])
            ):
                raise ValueError("expected a list of ComputerUse path selectors")
        if "auto_approve_read" in config and type(config["auto_approve_read"]) is not bool:
            raise ValueError("expected a boolean ComputerUse auto-approval setting")
        return config
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise AmendmentArtifactError("Invalid operator-selected hosted ComputerUse configuration") from exc


def running_workspace_policy(agent) -> dict[str, Any]:
    """Read the policy the loaded feature is actually using, not staged TOML."""
    from kestrel_sovereign.features.computer_use.feature import ComputerUseFeature

    features = getattr(agent, "features", None)
    feature = features.get("ComputerUseFeature") if isinstance(features, Mapping) else None
    if not isinstance(feature, ComputerUseFeature) or not feature._cfg.get("enabled"):
        return {"enabled": False}
    if feature._path_policy is None:
        raise AmendmentArtifactError("Cannot establish the running ComputerUse path policy")
    return {
        "enabled": True, "path_policy": WorkspacePathPolicy.from_policy(feature._path_policy).evidence(),
        "backend": feature._backend.name if feature._backend is not None else feature._cfg.get("backend", "docker"),
    }


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
