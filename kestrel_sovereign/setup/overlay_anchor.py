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
    ReanchorTargetError,
    resolve_reanchor_target,
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
        target = resolve_reanchor_target(
            agent_dir, backend=runtime_backend, dsn=runtime_dsn
        )
    except ReanchorTargetError as exc:
        return OverlayAnchorResult(
            agent_name=agent_name, error=str(exc),
            overlay_path=overlay_path, new_hash=new_hash,
        )

    if target.writes_to_anchor and not target.anchor_path.exists():
        return OverlayAnchorResult(
            agent_name=agent_name,
            error=f"agent DB not found at {target.anchor_path}",
            overlay_path=overlay_path, new_hash=new_hash,
            target_label=target.describe(),
        )

    async with target.open_storage() as storage:
        agent_nodes = await storage.graph.get_nodes_by_type("agent")
        if not agent_nodes:
            return OverlayAnchorResult(
                agent_name=agent_name,
                error=f"no agent identity node in {target.describe()}",
                overlay_path=overlay_path, new_hash=new_hash,
                target_label=target.describe(),
            )
        node = agent_nodes[0]
        old_hash = node.properties.get(OVERLAY_HASH_PROPERTY)
        if old_hash == new_hash:
            return OverlayAnchorResult(
                agent_name=agent_name, unchanged=True,
                old_hash=old_hash, new_hash=new_hash, overlay_path=overlay_path,
                target_label=target.describe(),
            )
        node.properties[OVERLAY_HASH_PROPERTY] = new_hash
        await storage.graph.add_node(node)  # upsert
        logger.info(
            "Anchored constitution overlay for %s in %s: %s",
            agent_name, target.describe(), new_hash[:16],
        )
    return OverlayAnchorResult(
        agent_name=agent_name, old_hash=old_hash, new_hash=new_hash,
        overlay_path=overlay_path, target_label=target.describe(),
    )
