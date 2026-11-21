"""Resolve the Amendment IX inputs shared by ComputerUse and Constitution lookup."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

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
)
from .structure import constitution_version

logger = logging.getLogger(__name__)

BASELINE_FIELD = "grant_baseline"
BASELINE_SCOPE = "computer-use.host"
BASELINE_VERSION = 1
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
) -> GrantResolution | None:
    """Compare the complete observed and proposed decisions before a write.

    The caller has verified the Sovereign artifact. Later reanchors may carry
    forward the baseline but cannot act as a grant revision writer.
    """
    prior_v2 = prior_baseline is not None or (prior_text is not None and constitution_version(prior_text) == 2)
    if constitution_version(new_text) != 2:
        if prior_v2 or BASELINE_FIELD in artifact:
            raise AmendmentArtifactError("Cannot restore legacy host grant authority through reanchor")
        return None
    proposed = _parse_baseline(target_did, artifact)
    previous = prior_baseline if prior_v2 else legacy_resolution()
    if previous is None or artifact[BASELINE_FIELD] != build_grant_baseline(target_did, previous):
        raise AmendmentArtifactError(
            "Edition 2 host grant baseline differs from the existing decision or its original evidence"
        )
    return proposed


def unavailable_host_grants() -> GrantResolution:
    return GrantResolution(frozenset(), "edition 2 grant baseline unavailable", "unavailable", "unavailable")


def unknown_signed_prior_edition(prior_text: str | None, properties: Mapping[str, Any]) -> bool:
    receipt = properties.get(CONSTITUTION_REANCHOR_KEY)
    return (
        prior_text is None
        and isinstance(receipt, Mapping)
        and "signed_artifact_hash" in receipt
    )


async def load_anchored_grant_baseline(
    db, *, target_did: str, properties: Mapping[str, Any],
    trusted_did_document: Mapping[str, Any],
    unknown_prior_edition: bool = False,
) -> GrantResolution | None:
    """Verify the baseline, or identify an older signed artifact by its contents."""
    from .anchored_bytes import read_anchored_constitution

    current_hash = properties.get("constitution_hash")
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
            return GrantResolution(
                frozenset(), "edition 2 default deny at genesis",
                "genesis receipt bound to governing bytes", "not operative", current_hash,
            )
    receipt = properties.get(CONSTITUTION_REANCHOR_KEY)
    if not isinstance(receipt, Mapping) or receipt.get("new_hash") != current_hash:
        raise AmendmentArtifactError("Current constitution has no matching reanchor receipt")
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
    verification = verify_reanchor_artifact(
        artifact, trusted_did_document=trusted_did_document,
        expected_constitution_sha256=current_hash,
    )
    if not verification.ok:
        raise AmendmentArtifactError(f"Host grant baseline failed signature verification: {verification.reason}")
    if unknown_prior_edition and BASELINE_FIELD not in artifact:
        return None
    return replace(
        _parse_baseline(target_did, artifact),
        baseline_sha256=artifact_hash, baseline_signer=verification.signer,
    )
