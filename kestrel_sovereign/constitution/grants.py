"""Resolve the Amendment IX inputs shared by ComputerUse and Constitution lookup."""

import asyncio
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
from .genesis_audit import GenesisAuditError, _validated_genesis_history, reconcile_genesis_receipt
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
    hosted_computer_use_config_path: Path | None = None,
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
    observation = artifact.get("offline_grant_observation")
    policy_supplied = not prior_v2 and isinstance(observation, dict) and "workspace_policy" in observation
    # An enabled Docker shell could previously request a host cwd through path
    # approval. Edition two also requires explicit file/mount grants.
    check_sandbox_cwd = (
        not prior_v2 and "shell_execution_sandboxed" in proposed.grants
        and not {"filesystem_read", "filesystem_outside_workspace"} <= proposed.grants
    )
    if check_sandbox_cwd or policy_supplied or (preserving_files and (live_workspace_policy is not None or needs_scope or proof_present)):
        disk_policy = configured_workspace_policy(storage_path, hosted_config_path=hosted_computer_use_config_path)
        if live_workspace_policy is not None and live_workspace_policy() != disk_policy:
            raise AmendmentArtifactError("Running ComputerUse policy differs from the disk-selected policy")
        if policy_supplied:
            from .workspace_disposition import WorkspacePathPolicy

            observed_policy = observation["workspace_policy"]
            if not isinstance(observed_policy, dict) or type(observed_policy.get("enabled")) is not bool:
                raise AmendmentArtifactError("Malformed signed pre-stop ComputerUse policy observation")
            if observed_policy["enabled"]:
                WorkspacePathPolicy.parse(observed_policy.get("path_policy"))
            compared_policy = disk_policy
            if "backend" not in observed_policy and not check_sandbox_cwd:
                # Historical observations signed the path policy only.
                compared_policy = {key: value for key, value in disk_policy.items() if key != "backend"}
            if observed_policy != compared_policy:
                raise AmendmentArtifactError("Signed pre-stop ComputerUse policy differs from the disk-selected policy")
    if check_sandbox_cwd:
        if disk_policy.get("backend") == "docker":
            raise AmendmentArtifactError(
                "Adoption would narrow the existing Docker shell working-directory permission. "
                "Resolve the host mount grants separately before adopting."
            )
        if stopped and not policy_supplied:
            raise AmendmentArtifactError("Offline adoption requires the signed pre-stop ComputerUse policy observation")
    if needs_scope:
        if disk_policy["enabled"] and not proof_present:
            raise AmendmentArtifactError(
                "Preserving legacy approval-mediated workspace access requires a separate signed workspace disposition"
            )
        if stopped and not proof_present and not policy_supplied:
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
    if not prior_v2 and (not disk_policy["enabled"] or disk_policy["path_policy"] != disposition.path_policy.evidence()):
        raise AmendmentArtifactError("Signed workspace disposition differs from the active legacy path policy")
    return replace(proposed, workspace_disposition=disposition)


def unavailable_host_grants() -> GrantResolution:
    return GrantResolution(frozenset(), "edition 2 grant baseline unavailable", "unavailable", "unavailable")


def prior_text_for_grant_repair(
    *, anchored_text: str | None, historical_hash: str | None,
    new_text: str, new_hash: str,
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
    return None


async def _recorded_legacy_restoration(db, governance, pointer: str, destination: str) -> str | None:
    """Identify a damaged pointer separately from the edition that was actually admitted.

    This only restores an already stored published v1 document. Its sole edge,
    signed native receipt and a completed audit must all identify that document.
    Any separate admission of the pointer itself requires restoring its bytes.
    """
    from .anchored_bytes import read_anchored_constitution
    from .genesis_audit import validate_completed_genesis_audit
    from .v1_releases import published_v1_release

    props = governance["properties"]
    receipt = props.get(CONSTITUTION_REANCHOR_KEY)
    release = published_v1_release(destination)
    if (release is None or governance["governed_by_targets"] != [destination]
            or not isinstance(receipt, Mapping) or receipt.get("new_hash") != destination
            or not is_constitution_hash(receipt.get("signed_artifact_hash"))):
        return None
    history = validate_constitution_reanchor_evidence(props)
    for record in (receipt, *(entry["receipt"] for entry in history)):
        if pointer in (record.get("old_hash"), record.get("new_hash"), *record.get("retained_governed_by", ())):
            return None
    audits = [entry["receipt"] for entry in _validated_genesis_history(props)]
    if current := props.get("genesis_audit"):
        audits.append(current)
    confirmed = []
    for audit in audits:
        if not isinstance(audit, Mapping):
            return None
        digest = audit.get("constitution_hash")
        if digest not in (pointer, destination):
            continue
        normalized = reconcile_genesis_receipt({"genesis_audit": audit}, digest)
        if normalized is not None and validate_completed_genesis_audit(normalized, digest) is not None:
            if digest == pointer:
                return None
            confirmed.append(normalized)
    if not confirmed or any(record != confirmed[0] for record in confirmed):
        return None
    stored, _ = await read_anchored_constitution(db, destination)
    return stored if stored == release.authenticated_text() else None


async def load_reanchor_prior_state(
    db, *, target_did: str, governance: Mapping[str, Any], anchored_text: str | None,
    historical_hash: str | None, new_text: str, new_hash: str,
    artifact_bytes: bytes, trusted_did_document: Mapping[str, Any],
    retained_targets: tuple[str, ...] = (),
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None, GrantResolution | None]:
    """Establish prior governing text and grant authority before applying amendment rules."""
    properties = governance["properties"]
    candidate = prior_text_for_grant_repair(
        anchored_text=anchored_text, historical_hash=historical_hash,
        new_text=new_text, new_hash=new_hash,
    )
    if candidate is None and historical_hash:
        from .emancipation import contract_from_json
        from .v1_prior import recover_published_v1

        contract = contract_from_json(properties.get("emancipation_contract"))
        candidate = recover_published_v1(historical_hash) or recover_published_v1(historical_hash, contract)
        if candidate is None and contract is not None and contract.enabled:
            raise AmendmentArtifactError("Restore the exact prior governing bytes before changing a personal agreement")
    if candidate is None or constitution_version(candidate) == 1:
        await asyncio.to_thread(
            require_legacy_governing_history, properties,
            historical_hash=historical_hash, retained_targets=retained_targets,
        )
    if candidate is None and historical_hash and historical_hash != new_hash:
        try:
            restored = await _recorded_legacy_restoration(db, governance, historical_hash, new_hash)
        except GenesisAuditError as exc:
            raise AmendmentArtifactError(f"Cannot establish a completed audit for pointer repair: {exc}") from exc
        if restored is not None:
            return new_hash, restored, None
    unknown = unknown_signed_prior_edition(candidate, properties)
    if not unknown and (candidate is None or constitution_version(candidate) != 2):
        return historical_hash, candidate, None

    recovered_hash = new_hash if anchored_text is None and candidate is not None else historical_hash
    baseline = await load_anchored_grant_baseline(
        db, target_did=target_did, properties=properties,
        trusted_did_document=trusted_did_document, unknown_prior_edition=unknown,
        recovered_hash=recovered_hash, recovered_governing_text=candidate, environ=environ,
        recovered_artifact_bytes=artifact_bytes if recovered_hash == new_hash else None,
    )
    if baseline is not None and baseline.verified_governing_anchor == (target_did, recovered_hash):
        return recovered_hash, candidate, baseline
    if unknown and historical_hash != new_hash:
        raise AmendmentArtifactError("Restore the exact prior governing bytes before revising a signed Constitution")
    return historical_hash, anchored_text, None


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


def _publisher_birth_anchor(original_genesis: Mapping) -> str:
    from .publisher import verify_recorded_publisher

    digest = original_genesis.get("constitution_hash")
    if not is_constitution_hash(digest):
        raise AmendmentArtifactError("Malformed publisher-backed Genesis receipt")
    try:
        genesis = reconcile_genesis_receipt({"genesis_audit": original_genesis}, digest)
    except GenesisAuditError as exc:
        raise AmendmentArtifactError("Genesis host grant ancestry could not be established") from exc
    if genesis is None:
        raise AmendmentArtifactError("Publisher-backed Genesis receipt is unavailable")
    verify_recorded_publisher(genesis.get("publisher"), digest)
    return digest


def require_legacy_governing_history(
    properties: Mapping[str, Any], *, historical_hash: str | None = None, retained_targets: tuple[str, ...] = (),
) -> None:
    """A missing or legacy document cannot supersede evidence of an established later edition."""
    from .publisher import has_packaged_publisher, verify_packaged_publisher

    try:
        history = _validated_genesis_history(properties)
        reanchors = validate_constitution_reanchor_evidence(properties)
    except (GenesisAuditError, ValueError) as exc:
        raise AmendmentArtifactError(str(exc)) from exc
    original = history[0]["receipt"] if history else properties.get("genesis_audit")
    if isinstance(original, Mapping) and (
        "publisher" in original or has_packaged_publisher(original.get("constitution_hash"))
    ):
        _publisher_birth_anchor(original)
        raise AmendmentArtifactError("Restore the exact edition 2 Constitution established at inception")
    # Every native adoption uses a signed descriptor selecting an attested
    # platform source, even when the personal governing digest is different.
    receipts = [entry["receipt"] for entry in reanchors]
    current = properties.get(CONSTITUTION_REANCHOR_KEY)
    if current is not None:
        receipts.append(current)
    from .reanchor_receipt import retained_constitution_hashes

    candidates = retained_constitution_hashes(properties)
    sources = [receipt.get("source_content_sha256") for receipt in receipts]
    sources.extend((properties.get("constitution_hash"), historical_hash))
    sources.extend(retained_targets)
    candidates.update(digest for digest in sources if is_constitution_hash(digest))
    for digest in candidates:
        if has_packaged_publisher(digest):
            verify_packaged_publisher(digest)
            raise AmendmentArtifactError("Restore the exact adopted edition 2 Constitution and its signed evidence")


async def load_anchored_grant_baseline(
    db, *, target_did: str, properties: Mapping[str, Any],
    trusted_did_document: Mapping[str, Any],
    unknown_prior_edition: bool = False,
    recovered_hash: str | None = None,
    recovered_governing_text: str | None = None,
    recovered_artifact_bytes: bytes | None = None,
    environ: Mapping[str, str] | None = None,
) -> GrantResolution | None:
    """Verify the baseline, or identify an older signed artifact by its contents."""
    from .anchored_bytes import read_anchored_constitution
    from .emancipation import EmancipationConfigError, contract_from_json
    from .v1_prior import recover_missing_v1_history

    current_hash = recovered_hash if recovered_hash is not None else properties.get("constitution_hash")
    if not is_constitution_hash(current_hash):
        raise AmendmentArtifactError("Current constitution has no valid anchor digest")
    try:
        history = validate_constitution_reanchor_evidence(properties)
        genesis_history = _validated_genesis_history(properties)
        contract = contract_from_json(properties.get("emancipation_contract"))
    except (ValueError, GenesisAuditError, EmancipationConfigError) as exc:
        raise AmendmentArtifactError("Cannot establish host grant baseline history") from exc

    verified_anchors = set()
    receipt = properties.get(CONSTITUTION_REANCHOR_KEY)
    current_genesis = properties.get("genesis_audit")
    original_genesis = genesis_history[0]["receipt"] if genesis_history else current_genesis
    # Only the original native inception receipt can establish v2 birth.
    # A later audit of adopted bytes cannot stand in for the declaration.
    standalone_genesis = receipt is None and not history and not genesis_history and not unknown_prior_edition
    prior_baselines = []
    if isinstance(original_genesis, Mapping) and ("publisher" in original_genesis or standalone_genesis):
        birth_hash = await asyncio.to_thread(_publisher_birth_anchor, original_genesis)
        verified_anchors.add(birth_hash)
        birth_baseline = GrantResolution(
            frozenset(), "edition 2 default deny at genesis",
            "genesis receipt bound to governing bytes", "not operative", birth_hash,
            verified_governing_anchor=(target_did, birth_hash),
        )
        if standalone_genesis and current_hash == birth_hash:
            return birth_baseline
        prior_baselines.append(birth_baseline)
    if not isinstance(receipt, Mapping) or receipt.get("new_hash") != current_hash:
        raise AmendmentArtifactError("Current constitution has no matching reanchor receipt")

    current_text, current_present = await read_anchored_constitution(db, current_hash)
    if (
        not current_present and recovered_governing_text is not None
        and hashlib.sha256(recovered_governing_text.encode("utf-8")).hexdigest() == current_hash
    ):
        current_text = recovered_governing_text
    if unknown_prior_edition and current_text is None:
        # An unmarked, genuinely legacy artifact can still identify the old
        # repair lane. It never supplies an edition-two grant baseline.
        return await _read_signed_baseline(
            db, target_did=target_did, constitution_hash=current_hash,
            receipt=receipt, trusted_did_document=trusted_did_document,
            allow_legacy=True, environ=environ, governing_text=None,
            contract=contract, verified_anchors=verified_anchors,
            recovered_artifact_bytes=recovered_artifact_bytes,
        )

    for entry in history:
        previous_receipt = entry["receipt"]
        previous_hash = previous_receipt["new_hash"]
        previous_text = current_text if previous_hash == current_hash else None
        if previous_hash != current_hash:
            previous_text, previous_present = await read_anchored_constitution(db, previous_hash)
            if not previous_present:
                previous_text = recover_missing_v1_history(previous_hash, current_text, contract)
        if previous_text is not None and constitution_version(previous_text) == 1:
            continue
        previous = await _read_signed_baseline(
            db, target_did=target_did, constitution_hash=previous_hash,
            receipt=previous_receipt, trusted_did_document=trusted_did_document,
            allow_legacy=False, environ=environ, governing_text=previous_text,
            contract=contract, verified_anchors=verified_anchors,
            recovered_artifact_bytes=recovered_artifact_bytes,
        )
        prior_baselines.append(previous)
        verified_anchors.add(previous_hash)

    current = await _read_signed_baseline(
        db, target_did=target_did, constitution_hash=current_hash,
        receipt=receipt, trusted_did_document=trusted_did_document,
        allow_legacy=unknown_prior_edition, environ=environ, governing_text=current_text,
        contract=contract, verified_anchors=verified_anchors,
        recovered_artifact_bytes=recovered_artifact_bytes,
    )
    if current is None:
        return None

    conflicts = set()
    workspace_conflict = False
    for previous in prior_baselines:
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
    governing_text: str | None, contract, verified_anchors: set[str],
    recovered_artifact_bytes: bytes | None,
) -> GrantResolution | None:
    from .anchored_bytes import read_anchored_constitution

    artifact_hash = receipt.get("signed_artifact_hash")
    if not is_constitution_hash(artifact_hash):
        raise AmendmentArtifactError("Current constitution has no signed reanchor artifact")
    text, present = await read_anchored_constitution(db, artifact_hash)
    if (
        not present and recovered_artifact_bytes is not None
        and hashlib.sha256(recovered_artifact_bytes).hexdigest() == artifact_hash
    ):
        try:
            text = recovered_artifact_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AmendmentArtifactError("Retained host grant baseline artifact is not UTF-8") from exc
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
    source_hash = receipt.get("source_content_sha256")
    await _verify_baseline_ancestry(
        db, artifact_text=text, artifact=artifact, receipt=receipt, target_did=target_did,
        constitution_hash=constitution_hash, governing_text=governing_text, contract=contract,
        source_hash=source_hash, credential=credential, verified_anchors=verified_anchors,
    )
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


async def _verify_baseline_ancestry(
    db, *, artifact_text, artifact, receipt, target_did, constitution_hash,
    governing_text, contract, source_hash, credential, verified_anchors,
):
    from .anchored_bytes import read_anchored_constitution
    from .publisher import verify_recorded_publisher
    from .transition import verify_archived_transition_adoption, verify_reanchor_receipt_chronology
    from .v1_prior import known_published_v1, recover_published_v1, verify_supported_v1_prior

    old_hash = receipt.get("old_hash")
    if not is_constitution_hash(old_hash):
        raise AmendmentArtifactError("Signed edition 2 receipt lacks a prior governing digest")
    if governing_text is None or constitution_version(governing_text) != 2:
        raise AmendmentArtifactError("Canonical edition 2 governing bytes are unavailable")
    if receipt.get("signed_artifact_signer") != credential.get("id"):
        raise AmendmentArtifactError("Native reanchor receipt does not identify the verified signer")
    if old_hash in verified_anchors:
        await asyncio.to_thread(
            verify_recorded_publisher, artifact.get("publisher_release"), source_hash,
            governing_text=governing_text, contract=contract,
        )
        verify_reanchor_receipt_chronology(artifact, receipt)
        from .emancipation import validate_v2_emancipation_activation

        previous = governing_text
        if old_hash != constitution_hash:
            previous, _ = await read_anchored_constitution(db, old_hash)
            if previous is None:
                raise AmendmentArtifactError("Restore the earlier edition 2 bytes to verify the preserved agreement")
        validate_v2_emancipation_activation(
            artifact, target_did=target_did, previous_text=previous, previous_hash=old_hash,
            new_text=governing_text, new_hash=constitution_hash, contract=contract,
        )
        return

    prior, prior_present = await read_anchored_constitution(db, old_hash)
    if not prior_present:
        prior = recover_published_v1(old_hash) or recover_published_v1(old_hash, contract)
    if prior is None or constitution_version(prior) != 1:
        raise AmendmentArtifactError("Signed edition 2 receipt has no verified Genesis adoption ancestry")
    role_bytes = None
    role_hash = receipt.get("prior_role_artifact_sha256")
    if role_hash is not None:
        role, _ = await read_anchored_constitution(db, role_hash)
        if role is not None:
            role_bytes = role.encode("utf-8")

    # A dormant v1 predecessor may have activated after adoption. Prefer
    # independent dormant authority before using the still-anchored active terms.
    prior_contract = None
    if not known_published_v1(old_hash):
        try:
            verify_supported_v1_prior(
                old_hash, None, prior_text=prior, prior_role_artifact_bytes=role_bytes,
                trusted_did_document=credential,
            )
        except AmendmentArtifactError:
            prior_contract = contract

    await asyncio.to_thread(
        verify_archived_transition_adoption,
        artifact_bytes=artifact_text.encode("utf-8"), receipt=receipt,
        trusted_did_document=credential, target_did=target_did,
        prior_hash=old_hash, prior_text=prior, anchored_contract=prior_contract, resulting_contract=contract,
        new_text=governing_text, new_hash=constitution_hash, source_hash=source_hash,
        prior_role_artifact_bytes=role_bytes,
    )
