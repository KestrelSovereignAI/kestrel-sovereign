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
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kestrel_sovereign.storage import AsyncStorage

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


async def anchor_overlay(*, agent_name: str, agent_dir: Path) -> OverlayAnchorResult:
    """Anchor the agent's current CONSTITUTION.md overlay hash in its DB node.

    Returns an :class:`OverlayAnchorResult`. Idempotent: anchoring an already-
    anchored, unchanged overlay reports ``unchanged=True``.
    """
    overlay_path = agent_dir / OVERLAY_FILENAME
    if not overlay_path.exists():
        return OverlayAnchorResult(
            agent_name=agent_name,
            error=f"no overlay at {overlay_path} — nothing to anchor",
            overlay_path=overlay_path,
        )
    new_hash = hashlib.sha256(overlay_path.read_bytes()).hexdigest()

    db_path = agent_dir / "kestrel_prime.db"
    if not db_path.exists():
        return OverlayAnchorResult(
            agent_name=agent_name, error=f"agent DB not found at {db_path}",
            overlay_path=overlay_path, new_hash=new_hash,
        )

    async with AsyncStorage(str(db_path)) as storage:
        agent_nodes = await storage.graph.get_nodes_by_type("agent")
        if not agent_nodes:
            return OverlayAnchorResult(
                agent_name=agent_name, error="no agent identity node in DB",
                overlay_path=overlay_path, new_hash=new_hash,
            )
        node = agent_nodes[0]
        old_hash = node.properties.get(OVERLAY_HASH_PROPERTY)
        if old_hash == new_hash:
            return OverlayAnchorResult(
                agent_name=agent_name, unchanged=True,
                old_hash=old_hash, new_hash=new_hash, overlay_path=overlay_path,
            )
        node.properties[OVERLAY_HASH_PROPERTY] = new_hash
        await storage.graph.add_node(node)  # upsert
        logger.info(
            "Anchored constitution overlay for %s: %s", agent_name, new_hash[:16],
        )
    return OverlayAnchorResult(
        agent_name=agent_name, old_hash=old_hash, new_hash=new_hash,
        overlay_path=overlay_path,
    )
