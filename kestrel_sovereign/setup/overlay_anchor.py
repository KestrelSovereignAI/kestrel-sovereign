"""Anchor a per-agent constitution overlay (#1722).

The per-agent overlay ``<agent_dir>/CONSTITUTION.md`` can grant DANGEROUS
Amendment IX capabilities (``shell_execution_host``, ``filesystem_write``). To
stop anyone with file-write next to the agent DB from self-granting host shell,
those grants are honored ONLY when the overlay's sha256 is anchored in the
agent's identity node (``properties["constitution_overlay_hash"]``) and matches
the file at load time.

This module establishes that anchor through a TRUSTED channel — the operator
running ``kestrel constitution anchor-overlay`` against a *stopped* agent's DB,
the same trust boundary as ``constitution_reanchor``. It must never be reachable
from anything an attacker with mere file-write can drive.

It also shares that module's target rule (#2890): the anchor property is read
by the *runtime*, so it must be written to the database the runtime opens. On a
PostgreSQL host, writing the local ``kestrel_prime.db`` reports success while
the agent goes on denying every Amendment IX grant the overlay declares.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kestrel_sovereign.setup.constitution_reanchor import (
    ReanchorTarget,
    ReanchorTargetError,
    resolve_reanchor_target,
    runtime_record_is_pending,
)

logger = logging.getLogger(__name__)

OVERLAY_FILENAME = "CONSTITUTION.md"
OVERLAY_HASH_PROPERTY = "constitution_overlay_hash"


@dataclass
class OverlayAnchorResult:
    agent_name: str
    error: Optional[str] = None
    unchanged: bool = False
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    overlay_path: Optional[Path] = None
    #: The database this run read and wrote, DSN credentials redacted. An
    #: anchor that does not say where it landed cannot be trusted to have
    #: landed anywhere the agent reads (#2890).
    target_label: str = ""
    #: ``"sqlite"`` or ``"postgres"`` — the same fact as ``target_label`` but
    #: as a value rather than prose. The label embeds a filesystem path, so a
    #: caller asserting on it is one directory name away from asserting
    #: nothing; ``ReanchorResult`` carries the pair for the same reason.
    target_backend: str = ""


async def anchor_overlay(
    *,
    agent_name: str,
    agent_dir: Path,
    runtime_backend: Optional[str] = None,
    runtime_dsn: Optional[str] = None,
) -> OverlayAnchorResult:
    """Anchor the agent's current CONSTITUTION.md overlay hash in its DB node.

    Returns an :class:`OverlayAnchorResult`. Idempotent: anchoring an already-
    anchored, unchanged overlay reports ``unchanged=True``.

    ``runtime_backend`` / ``runtime_dsn`` name the database the agent's runtime
    opens; ``None`` resolves them from the environment exactly as boot does.
    """
    overlay_path = agent_dir / OVERLAY_FILENAME
    if not overlay_path.exists():
        return OverlayAnchorResult(
            agent_name=agent_name,
            error=f"no overlay at {overlay_path} — nothing to anchor",
            overlay_path=overlay_path,
        )
    new_hash = hashlib.sha256(overlay_path.read_bytes()).hexdigest()

    try:
        target = await resolve_reanchor_target(
            agent_dir, backend=runtime_backend, dsn=runtime_dsn
        )
    except ReanchorTargetError as exc:
        return OverlayAnchorResult(
            agent_name=agent_name, error=str(exc),
            overlay_path=overlay_path, new_hash=new_hash,
        )

    try:
        # Same rule as the constitution reanchor, from the same predicate: before
        # replication the bytes that will govern this agent are the anchor's, so an
        # overlay anchored anywhere else is anchored where boot will not look.
        #
        # This is *not* a licence to prefer the local file generally — that was the
        # #2890 defect, and it stays refused. An Amendment IX overlay hash is read
        # from the runtime database, so writing it locally on a replicated host
        # reports success while every grant stays denied. Pending is the one state
        # where the anchor is the runtime's future contents.
        if not target.writes_to_anchor and await runtime_record_is_pending(target):
            logger.info(
                "%s has no record for %s yet; anchoring the overlay in the local "
                "anchor, which is what first boot will replicate.",
                target.describe(), target.agent_did,
            )
            target = ReanchorTarget(
                anchor_path=target.anchor_path,
                backend="sqlite",
                agent_did=target.agent_did,
            )

        return await _anchor_overlay_in(
            target,
            agent_name=agent_name,
            overlay_path=overlay_path,
            new_hash=new_hash,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the operator
        logger.exception("Overlay anchor against %s failed", target.describe())
        return OverlayAnchorResult(
            agent_name=agent_name,
            error=(
                f"Could not anchor the overlay in {target.describe()}: "
                f"{exc!r}. Nothing was written."
            ),
            overlay_path=overlay_path, new_hash=new_hash,
            target_label=target.describe(), target_backend=target.backend,
        )


async def _anchor_overlay_in(
    target,
    *,
    agent_name: str,
    overlay_path: Path,
    new_hash: str,
) -> OverlayAnchorResult:
    """Write the overlay hash into ``target``, bound to its agent."""
    async with target.open_storage() as storage:
        # ``open_storage`` binds every store to this agent; the lookup is by
        # DID for the same reason the runtime's own is. One PostgreSQL
        # database holds every local agent, and an unbound read here would
        # anchor a DANGEROUS-capability overlay onto whichever tenant came
        # back first.
        node = await storage.graph.get_node(target.agent_did)
        if node is None or node.node_type != "agent":
            return OverlayAnchorResult(
                agent_name=agent_name,
                error=(
                    f"no agent identity node for {target.agent_did} in "
                    f"{target.describe()}"
                ),
                overlay_path=overlay_path, new_hash=new_hash,
                target_label=target.describe(), target_backend=target.backend,
            )
        old_hash = node.properties.get(OVERLAY_HASH_PROPERTY)
        if old_hash == new_hash:
            return OverlayAnchorResult(
                agent_name=agent_name, unchanged=True,
                old_hash=old_hash, new_hash=new_hash, overlay_path=overlay_path,
                target_label=target.describe(), target_backend=target.backend,
            )
        node.properties[OVERLAY_HASH_PROPERTY] = new_hash
        await storage.graph.add_node(node)  # upsert
        logger.info(
            "Anchored constitution overlay for %s in %s: %s",
            agent_name, target.describe(), new_hash[:16],
        )
    return OverlayAnchorResult(
        agent_name=agent_name, old_hash=old_hash, new_hash=new_hash,
        overlay_path=overlay_path, target_label=target.describe(), target_backend=target.backend,
    )
