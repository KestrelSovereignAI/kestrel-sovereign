"""Resolve the Amendment IX inputs shared by ComputerUse and Constitution lookup."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from kestrel_sovereign._regular_file import read_regular_file
from kestrel_sovereign.constitution.hierarchy import (
    DANGEROUS_CAPABILITIES,
    parse_amendment_ix_grants,
)

from .amendment_artifact import AmendmentArtifactError, verify_reanchor_artifact
from .genesis_audit import GenesisAuditError, reconcile_genesis_receipt
from .reanchor_receipt import (
    CONSTITUTION_REANCHOR_HISTORY_KEY,
    CONSTITUTION_REANCHOR_KEY,
    is_constitution_hash,
    validate_constitution_reanchor_evidence,
)
from .structure import constitution_version
from .trust_root import MAX_TRUST_ROOT_BYTES
from .workspace_disposition import (
    WORKSPACE_DISPOSITION_FIELD,
    WorkspaceDisposition,
    configured_workspace_policy,
    verify_workspace_disposition,
)

logger = logging.getLogger(__name__)

BASELINE_FIELD = "grant_baseline"
BASELINE_SCOPE = "computer-use.host"
BASELINE_VERSION = 1
HISTORY_PIN_ENV = "KESTREL_SOVEREIGN_GRANT_HISTORY_PATH"
ENVIRONMENT_KEY = "KESTREL_CAPABILITY_GRANTS"
ENVIRONMENT_SOURCE = f"{ENVIRONMENT_KEY} environment variable"


@dataclass(frozen=True)
class OverlayGrantEvidence:
    """Evidence from the overlay's existing hash verifier."""

    status: str
    verified_hash: str | None = None


@dataclass(frozen=True)
class GrantResolution:
    grants: frozenset[str]
    source: str
    verification: str
    overlay_status: str
    content_hash: str | None = None
    baseline_sha256: str | None = None
    baseline_signer: str | None = None
    environment_narrowed: bool = False
    conflicts: frozenset[str] = frozenset()
    workspace_disposition: WorkspaceDisposition | None = None
    workspace_conflict: bool = False
    verified_governing_anchor: tuple[str, str] | None = None


def resolve_amendment_ix_grants(agent, *, edition: int | None = None) -> GrantResolution:
    """Resolve the one operative source used by the grant gate and viewer."""
    baseline = getattr(agent, "_constitution_host_grant_baseline", None)
    if isinstance(baseline, GrantResolution):
        if baseline.source == ENVIRONMENT_SOURCE:
            narrowed = baseline.grants & _environment_grants(os.environ.get(ENVIRONMENT_KEY, ""))
            return replace(baseline, grants=narrowed, environment_narrowed=narrowed != baseline.grants)
        return baseline
    if edition == 2:
        return unavailable_host_grants()
    return resolve_legacy_grants(agent)


def _environment_grants(value: str) -> frozenset[str]:
    return frozenset(g.strip() for g in value.split(",") if g.strip())


def resolve_legacy_grants(agent, *, environ: Mapping[str, str] | None = None) -> GrantResolution:
    """Resolve v1 sources with the historical decision precedence."""
    overlay_text = getattr(agent, "constitution_text", None)
    overlay_accepted = getattr(agent, "constitution_overlay_verified", False)
    evidence = getattr(agent, "_constitution_overlay_grant_evidence", None)
    if isinstance(evidence, OverlayGrantEvidence):
        overlay_status = evidence.status
        verified_hash = evidence.verified_hash
    else:
        if overlay_text is None:
            overlay_status = "absent"
        else:
            overlay_status = "verification unavailable" if overlay_accepted else "unverified"
        verified_hash = None

    # Legacy agents can supply an accepted overlay without hash evidence.
    # Also avoid attributing a stale verifier result to text refreshed by
    # an in-progress audit. Neither case changes the gate's existing decision.
    if verified_hash is not None:
        if (
            not overlay_accepted
            or overlay_text is None
            or hashlib.sha256(overlay_text.encode("utf-8")).hexdigest() != verified_hash
        ):
            verified_hash = None
            if overlay_status == "hash verified":
                overlay_status = "verification unavailable"

    for attr in ("granted_capabilities", "capability_grants", "amendment_ix"):
        grants = getattr(agent, attr, None)
        if grants:
            return GrantResolution(
                frozenset(grants), f"agent.{attr}", "unavailable", overlay_status
            )

    if overlay_text is not None:
        if overlay_accepted:
            # An accepted overlay controls the decision even if it grants nothing.
            return GrantResolution(
                parse_amendment_ix_grants(overlay_text),
                "per-agent overlay",
                "hash verified" if verified_hash else "unavailable",
                overlay_status,
                verified_hash,
            )
        logger.warning(
            "Ignoring Amendment IX grants from an UNVERIFIED constitution "
            "overlay. Run `kestrel constitution anchor-overlay`."
        )

    from kestrel_sovereign.config import CONSTITUTION_PATH

    try:
        package_bytes = Path(CONSTITUTION_PATH).read_bytes()
        package_text = package_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, UnicodeError) as exc:
        logger.debug("could not read constitution: %s", exc)
        package_status = "unavailable"
    else:
        grants = parse_amendment_ix_grants(package_text)
        if grants:
            return GrantResolution(
                grants,
                "packaged Constitution",
                "unavailable (digest of packaged bytes)",
                overlay_status,
                hashlib.sha256(package_bytes).hexdigest(),
            )
        package_status = "no grants"

    env_grants = (os.environ if environ is None else environ).get(ENVIRONMENT_KEY, "")
    if env_grants:
        return GrantResolution(
            _environment_grants(env_grants),
            ENVIRONMENT_SOURCE,
            "unavailable",
            overlay_status,
        )
    return GrantResolution(
        frozenset(), f"default deny (packaged Constitution: {package_status})",
        "unavailable", overlay_status,
    )

def preview_offline_legacy_grants(
    agent_dir: Path | None,
    *,
    overlay_hash: str | None,
    environ: Mapping[str, str] | None = None,
) -> GrantResolution:
    """Read the on-disk legacy sources through the same precedence as runtime.

    Runtime attributes have no native persistent writer. A deployment injecting
    one must use the live writer, which compares the actual runtime decision.
    """
    overlay = SimpleNamespace()
    if agent_dir is not None:
        try:
            raw = (agent_dir / "CONSTITUTION.md").read_bytes()
        except FileNotFoundError:
            if overlay_hash:
                raise AmendmentArtifactError("Cannot compare grants: anchored overlay is missing") from None
        except OSError as exc:
            raise AmendmentArtifactError("Cannot read legacy grant overlay") from exc
        else:
            try:
                overlay.constitution_text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AmendmentArtifactError("Cannot compare grants: overlay is not UTF-8") from exc
            digest = hashlib.sha256(raw).hexdigest()
            if overlay_hash != digest:
                raise AmendmentArtifactError("Cannot compare grants: overlay is not hash verified")
            overlay.constitution_overlay_verified = True
            overlay._constitution_overlay_grant_evidence = OverlayGrantEvidence("hash verified", digest)
    elif overlay_hash:
        raise AmendmentArtifactError("Cannot compare grants without the agent's anchored overlay")
    return resolve_legacy_grants(overlay, environ=environ)


def build_grant_baseline(target_did: str, resolution: GrantResolution) -> dict[str, Any]:
    """Prepare structured preservation evidence for a forthcoming reanchor."""
    return {
        "version": BASELINE_VERSION,
        "revision": 0,
        "target_did": target_did,
        "scope": BASELINE_SCOPE,
        "grants": sorted(resolution.grants & DANGEROUS_CAPABILITIES),
        "legacy_evidence": {
            "source": resolution.source,
            "verification": resolution.verification,
            "overlay_status": resolution.overlay_status,
            "sha256": resolution.content_hash,
            "signer": None,
        },
    }


def build_offline_grant_observation(
    *, target_did: str, prior_sha256: str, observed_at: str, resolution: GrantResolution,
    workspace_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the running pre-stop grant decision for Sovereign attestation.

    Use the v1 effective-grant viewer's workspace policy observation when
    the legacy file scope requires it and has no separate signed disposition.
    """
    record = {
        "version": 1,
        "prior_sha256": prior_sha256,
        "observed_at": observed_at,
        "baseline": build_grant_baseline(target_did, resolution),
    }

    if workspace_policy is not None:
        record["workspace_policy"] = dict(workspace_policy)
    return record


def require_offline_grant_observation(
    artifact: Mapping[str, Any], *, prior_sha256: str | None,
) -> None:
    """Require the signed running observation in addition to the disk comparison."""
    observation = artifact.get("offline_grant_observation")
    if (
        not is_constitution_hash(prior_sha256)
        or not isinstance(observation, dict)
        or set(observation) - {"workspace_policy"} != {"version", "prior_sha256", "observed_at", "baseline"}
        or type(observation["version"]) is not int or observation["version"] != 1
        or observation["prior_sha256"] != prior_sha256
        or observation["baseline"] != artifact.get(BASELINE_FIELD)
    ):
        raise AmendmentArtifactError("Offline adoption requires a Sovereign-signed pre-stop grant observation")
    try:
        observed = datetime.fromisoformat(observation["observed_at"])
        signed = datetime.fromisoformat(artifact["created_at"])
        if observed.utcoffset() is None or signed.utcoffset() is None or not (
            observed <= signed <= datetime.now(timezone.utc)
        ):
            raise ValueError("expected observation, signing, and verification in chronological order")
    except (KeyError, TypeError, ValueError) as exc:
        raise AmendmentArtifactError("Invalid timestamp for the offline pre-stop grant observation") from exc


def _parse_baseline(target_did: str, artifact: Mapping[str, Any]) -> GrantResolution:
    baseline = artifact.get(BASELINE_FIELD)
    if not isinstance(baseline, dict) or set(baseline) != {
        "version", "revision", "target_did", "scope", "grants", "legacy_evidence",
    }:
        raise AmendmentArtifactError("Edition 2 requires a signed host grant baseline")
    grants = baseline["grants"]
    if (
        type(baseline["version"]) is not int or baseline["version"] != BASELINE_VERSION
        or type(baseline["revision"]) is not int or baseline["revision"] != 0
        or baseline["target_did"] != target_did or baseline["scope"] != BASELINE_SCOPE
        or not isinstance(grants, list)
        or any(not isinstance(grant, str) or grant not in DANGEROUS_CAPABILITIES for grant in grants)
        or grants != sorted(set(grants))
    ):
        raise AmendmentArtifactError("Malformed or mis-targeted edition 2 host grant baseline")
    evidence = baseline["legacy_evidence"]
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"source", "verification", "overlay_status", "sha256", "signer"}
        or any(not isinstance(evidence[key], str) or not evidence[key]
               for key in ("source", "verification", "overlay_status"))
        or (evidence["sha256"] is not None and not is_constitution_hash(evidence["sha256"]))
        or evidence["signer"] is not None
    ):
        raise AmendmentArtifactError("Malformed original evidence in edition 2 host grant baseline")
    return GrantResolution(
        frozenset(grants), evidence["source"], evidence["verification"],
        evidence["overlay_status"], evidence["sha256"],
    )


def validate_grant_cutover(
    *, target_did: str, artifact: Mapping[str, Any], prior_text: str | None,
    new_text: str, legacy_resolution: Callable[[], GrantResolution],
    prior_baseline: GrantResolution | None = None,
    trusted_did_document: Mapping[str, Any] | None = None,
    storage_path: str | os.PathLike | None = None,
    live_workspace_policy: Callable[[], dict] | None = None,
    stopped: bool = False,
) -> GrantResolution | None:
    """Compare the complete observed and proposed decisions before a write.

    The caller has verified the Sovereign artifact. Later reanchors may carry
    forward the baseline but cannot act as a grant revision writer.
    """
    prior_v2 = prior_baseline is not None or (prior_text is not None and constitution_version(prior_text) == 2)
    if constitution_version(new_text) != 2:
        if prior_v2 or BASELINE_FIELD in artifact or WORKSPACE_DISPOSITION_FIELD in artifact:
            raise AmendmentArtifactError("Cannot restore legacy host grant authority through reanchor")
        return None
    proposed = _parse_baseline(target_did, artifact)
    previous = prior_baseline if prior_v2 else legacy_resolution()
    if previous is not None and (previous.conflicts or previous.workspace_conflict):
        raise AmendmentArtifactError("Cannot reanchor with conflicting signed host grant history")
    if previous is None or artifact[BASELINE_FIELD] != build_grant_baseline(target_did, previous):
        raise AmendmentArtifactError(
            "Edition 2 host grant baseline differs from the existing decision or its original evidence"
        )
    previous_scope = previous.workspace_disposition
    proof_present = WORKSPACE_DISPOSITION_FIELD in artifact
    if prior_v2 and (previous_scope is not None) != proof_present:
        raise AmendmentArtifactError("Reanchor cannot add or discard the existing signed workspace disposition")

    disk_policy = None
    preserving_files = not prior_v2 and bool({"filesystem_read", "filesystem_write"} & proposed.grants)
    needs_scope = preserving_files and "filesystem_outside_workspace" not in proposed.grants
    if preserving_files and (live_workspace_policy is not None or needs_scope or proof_present):
        disk_policy = configured_workspace_policy(storage_path)
        if live_workspace_policy is not None and live_workspace_policy() != disk_policy:
            raise AmendmentArtifactError("Running ComputerUse policy differs from the disk-selected policy")
    if needs_scope:
        if disk_policy["enabled"] and not proof_present:
            raise AmendmentArtifactError(
                "Preserving legacy approval-mediated workspace access requires a separate signed workspace disposition"
            )
        if stopped and not proof_present:
            observation = artifact.get("offline_grant_observation")
            observed_policy = observation.get("workspace_policy") if isinstance(observation, dict) else None
            if (
                not isinstance(observed_policy, dict)
                or observed_policy.get("enabled") is not False
                or observed_policy != disk_policy
            ):
                raise AmendmentArtifactError("Offline adoption requires the signed pre-stop ComputerUse policy observation")
    if not proof_present:
        return proposed
    if trusted_did_document is None or prior_text is None:
        raise AmendmentArtifactError("Cannot establish the prior workspace disposition authority")
    disposition = verify_workspace_disposition(
        artifact[WORKSPACE_DISPOSITION_FIELD],
        trusted_did_document=trusted_did_document,
        target_did=target_did,
        constitution_sha256=hashlib.sha256(new_text.encode()).hexdigest(),
        grant_baseline=artifact[BASELINE_FIELD],
        prior_sha256=previous_scope.prior_sha256 if previous_scope else hashlib.sha256(prior_text.encode()).hexdigest(),
        artifact_created_at=artifact.get("created_at"),
    )
    if previous_scope and not previous_scope.same_scope(disposition):
        raise AmendmentArtifactError("Reanchor cannot change the existing signed workspace disposition")
    if not prior_v2 and disk_policy != {
        "enabled": True, "path_policy": disposition.path_policy.evidence(),
    }:
        raise AmendmentArtifactError("Signed workspace disposition differs from the active legacy path policy")
    return replace(proposed, workspace_disposition=disposition)


def unavailable_host_grants() -> GrantResolution:
    return GrantResolution(frozenset(), "edition 2 grant baseline unavailable", "unavailable", "unavailable")


def prior_text_for_grant_repair(
    *, anchored_text: str | None, historical_hash: str | None,
    new_text: str, new_hash: str,
    governance: Mapping[str, Any] | None = None,
) -> str | None:
    """Recover the prior edition from exact, independently signed replacement bytes.

    The two writers have already computed the destination digest from the
    resolved bytes and verified the new artifact against the pinned Sovereign.
    Equality with the historical digest proves the recovered edition even if
    its old blob is missing. An edition 2 repair still needs the prior signed
    grant baseline, independent of the new artifact.
    """
    if anchored_text is not None:
        return anchored_text
    if historical_hash == new_hash:
        return new_text
    if governance is not None:
        receipt = governance["properties"].get(CONSTITUTION_REANCHOR_KEY)
        if (
            isinstance(receipt, Mapping)
            and receipt.get("new_hash") == new_hash
            and governance["governed_by_targets"] == [new_hash]
        ):
            return new_text
    return None


def unknown_signed_prior_edition(prior_text: str | None, properties: Mapping[str, Any]) -> bool:
    receipt = properties.get(CONSTITUTION_REANCHOR_KEY)
    return (
        prior_text is None
        and isinstance(receipt, Mapping)
        and "signed_artifact_hash" in receipt
    )


def sovereign_document_fingerprint(document: Mapping[str, Any]) -> str:
    """Fingerprint an operator-pinned DID document independent of JSON whitespace."""
    payload = json.dumps(dict(document), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_historical_signer(
    trusted_did_document: Mapping[str, Any],
    artifact_hash: str,
    environ: Mapping[str, str] | None,
) -> Mapping[str, Any] | None:
    """Read the independently pinned archived credential for one admitted artifact."""
    env = os.environ if environ is None else environ
    configured = env.get(HISTORY_PIN_ENV, "").strip()
    if not configured:
        return None
    try:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError("archive pin must be absolute")
        raw = read_regular_file(path, max_bytes=MAX_TRUST_ROOT_BYTES)
        if len(raw) > MAX_TRUST_ROOT_BYTES:
            raise ValueError("archive pin too large")
        pin = json.loads(raw)
    except (OSError, TypeError, ValueError) as exc:
        raise AmendmentArtifactError("Cannot load independently pinned historical grant credentials") from exc

    if (
        not isinstance(pin, dict)
        or set(pin) != {"active_document_sha256", "admitted_artifacts"}
        or pin["active_document_sha256"] != sovereign_document_fingerprint(trusted_did_document)
        or not isinstance(pin["admitted_artifacts"], dict)
    ):
        raise AmendmentArtifactError("Historical grant credentials do not bind to this active Sovereign pin")
    signer_doc = pin["admitted_artifacts"].get(artifact_hash)
    if signer_doc is not None and not isinstance(signer_doc, dict):
        raise AmendmentArtifactError("Malformed archived grant signing credential")
    return signer_doc


async def load_anchored_grant_baseline(
    db, *, target_did: str, properties: Mapping[str, Any],
    trusted_did_document: Mapping[str, Any],
    unknown_prior_edition: bool = False,
    recovered_hash: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> GrantResolution | None:
    """Verify the baseline, or identify an older signed artifact by its contents."""
    from .anchored_bytes import read_anchored_constitution

    current_hash = recovered_hash if recovered_hash is not None else properties.get("constitution_hash")
    if not is_constitution_hash(current_hash):
        raise AmendmentArtifactError("Current constitution has no valid anchor digest")
    if (
        not unknown_prior_edition
        and CONSTITUTION_REANCHOR_KEY not in properties
        and CONSTITUTION_REANCHOR_HISTORY_KEY not in properties
        and "genesis_audit_history" not in properties
    ):
        try:
            genesis = reconcile_genesis_receipt(properties, current_hash)
        except GenesisAuditError as exc:
            raise AmendmentArtifactError("Genesis host grant ancestry could not be established") from exc
        if genesis is not None:
            from .publisher import verify_recorded_publisher

            # V2 inception is necessarily dormant and its stored governing
            # digest is the exact digest of the unrendered publisher source.
            verify_recorded_publisher(genesis.get("publisher"), current_hash)
            return GrantResolution(
                frozenset(), "edition 2 default deny at genesis",
                "genesis receipt bound to governing bytes", "not operative", current_hash,
                verified_governing_anchor=(target_did, current_hash),
            )
    receipt = properties.get(CONSTITUTION_REANCHOR_KEY)
    if not isinstance(receipt, Mapping) or receipt.get("new_hash") != current_hash:
        raise AmendmentArtifactError("Current constitution has no matching reanchor receipt")
    current = await _read_signed_baseline(
        db, target_did=target_did, constitution_hash=current_hash,
        receipt=receipt, trusted_did_document=trusted_did_document,
        allow_legacy=unknown_prior_edition, environ=environ,
    )
    if current is None:
        return None

    try:
        history = validate_constitution_reanchor_evidence(properties)
    except ValueError as exc:
        raise AmendmentArtifactError("Cannot establish host grant baseline history") from exc

    conflicts = set()
    workspace_conflict = False
    for entry in history:
        previous_receipt = entry["receipt"]
        previous_hash = previous_receipt["new_hash"]
        if previous_hash != current_hash:
            previous_text, _ = await read_anchored_constitution(db, previous_hash)
            if previous_text is not None and constitution_version(previous_text) == 1:
                continue
        previous = await _read_signed_baseline(
            db, target_did=target_did, constitution_hash=previous_hash,
            receipt=previous_receipt, trusted_did_document=trusted_did_document,
            allow_legacy=False, environ=environ,
        )
        scope, previous_scope = current.workspace_disposition, previous.workspace_disposition
        if (scope is None) != (previous_scope is None) or (
            scope is not None and previous_scope is not None and not scope.same_scope(previous_scope)
        ):
            workspace_conflict = True
        conflicts.update(current.grants ^ previous.grants)
        if build_grant_baseline(target_did, current)["legacy_evidence"] != build_grant_baseline(
            target_did, previous
        )["legacy_evidence"]:
            conflicts.update(current.grants | previous.grants)
    return replace(
        current, grants=current.grants - conflicts, conflicts=frozenset(conflicts),
        workspace_disposition=None if workspace_conflict else current.workspace_disposition,
        workspace_conflict=workspace_conflict,
    )


async def _read_signed_baseline(
    db, *, target_did: str, constitution_hash: str,
    receipt: Mapping[str, Any], trusted_did_document: Mapping[str, Any],
    allow_legacy: bool, environ: Mapping[str, str] | None,
) -> GrantResolution | None:
    from .anchored_bytes import read_anchored_constitution

    artifact_hash = receipt.get("signed_artifact_hash")
    if not is_constitution_hash(artifact_hash):
        raise AmendmentArtifactError("Current constitution has no signed reanchor artifact")
    text, _ = await read_anchored_constitution(db, artifact_hash)
    if text is None:
        raise AmendmentArtifactError("Signed host grant baseline artifact is unavailable")
    try:
        artifact = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AmendmentArtifactError("Signed host grant baseline artifact is invalid JSON") from exc
    if not isinstance(artifact, dict):
        raise AmendmentArtifactError("Signed host grant baseline artifact must be a JSON object")
    credential = trusted_did_document
    verification = verify_reanchor_artifact(
        artifact, trusted_did_document=credential,
        expected_constitution_sha256=constitution_hash,
    )
    if not verification.ok:
        archived = _read_historical_signer(trusted_did_document, artifact_hash, environ)
        if archived is not None and archived.get("id") != target_did:
            credential = archived
            verification = verify_reanchor_artifact(
                artifact, trusted_did_document=credential,
                expected_constitution_sha256=constitution_hash,
            )
    if not verification.ok:
        raise AmendmentArtifactError(f"Host grant baseline failed signature verification: {verification.reason}")
    if allow_legacy and BASELINE_FIELD not in artifact:
        return None
    from .publisher import verify_recorded_publisher

    verify_recorded_publisher(artifact.get("publisher_release"), receipt.get("source_content_sha256"))
    baseline = _parse_baseline(target_did, artifact)
    disposition = None
    if WORKSPACE_DISPOSITION_FIELD in artifact:
        transition = artifact.get("transition") or {}
        disposition = verify_workspace_disposition(
            artifact[WORKSPACE_DISPOSITION_FIELD], trusted_did_document=credential,
            target_did=target_did, constitution_sha256=constitution_hash,
            grant_baseline=artifact[BASELINE_FIELD],
            prior_sha256=(transition.get("intent") or {}).get("prior_sha256"),
            artifact_created_at=artifact.get("created_at"),
        )
    return replace(
        baseline, baseline_sha256=artifact_hash, baseline_signer=verification.signer,
        verified_governing_anchor=(target_did, constitution_hash),
        workspace_disposition=disposition,
    )
