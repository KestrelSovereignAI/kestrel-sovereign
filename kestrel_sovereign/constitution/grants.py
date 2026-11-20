"""Resolve the Amendment IX inputs shared by ComputerUse and Constitution lookup."""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from kestrel_sovereign.constitution.hierarchy import parse_amendment_ix_grants

logger = logging.getLogger(__name__)


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


def resolve_amendment_ix_grants(agent) -> GrantResolution:
    """Return the existing grant decision along with the evidence for its source."""
    overlay_text = getattr(agent, "constitution_text", None)
    overlay_accepted = getattr(agent, "constitution_overlay_verified", False)
    evidence = getattr(agent, "_constitution_overlay_grant_evidence", None)
    if isinstance(evidence, OverlayGrantEvidence):
        overlay_status = evidence.status
        verified_hash = evidence.verified_hash
    else:
        overlay_status = "absent" if overlay_text is None else "unverified"
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

    env_grants = os.environ.get("KESTREL_CAPABILITY_GRANTS", "")
    if env_grants:
        return GrantResolution(
            frozenset(g.strip() for g in env_grants.split(",") if g.strip()),
            "KESTREL_CAPABILITY_GRANTS environment variable",
            "unavailable",
            overlay_status,
        )
    return GrantResolution(
        frozenset(), f"default deny (packaged Constitution: {package_status})",
        "unavailable", overlay_status,
    )
