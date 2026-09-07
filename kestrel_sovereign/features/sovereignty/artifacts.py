"""The export artifacts an agent durably owns.

A sovereignty export or backup writes two things: bytes into the host's
storage cache (one ``storage_cache/`` directory, shared by every agent on
the host and by every producer that ever ran here) and a receipt into the
agent's *own* storage — a ``backup_artifact`` node whose id is the content
hash, and a ``sovereignty_receipt`` node naming the same hash and CID.

Only the second is an ownership record. The cache directory carries a
filename and a ``.meta`` sidecar, and the sidecar names an agent, but a
shared directory is data anybody with the host's filesystem can write; the
receipt is written through the bound, owner-scoped graph, so on a shared
PostgreSQL backend it still answers only for the agent that wrote it. The
agent-routed file browser (#3225) and pin view (#3226) therefore ask this
module, never the filename, the CID, the selected agent's display name, or
the sidecar, whether an artifact is the routed agent's.

A cache entry with no receipt in the routed agent's storage — legacy
content, a test's leftovers, another agent's export, an identity package
(which writes no receipt) — is not attributed to whoever asks first. It is
invisible through these surfaces; the operator has the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

BACKUP_ARTIFACT_NODE_TYPE = "backup_artifact"
SOVEREIGNTY_RECEIPT_NODE_TYPE = "sovereignty_receipt"


@dataclass(frozen=True)
class OwnedArtifact:
    """One export/backup artifact the bound agent recorded as its own."""

    content_hash: Optional[str]
    cid: Optional[str]


async def owned_artifacts(storage: Any) -> List[OwnedArtifact]:
    """Every artifact the storage's bound agent durably owns.

    Reads the two receipt node types through ``storage.get_nodes_by_type``,
    which the bound graph store scopes to its owner. An agent whose privacy
    mode hides persisted rows owns nothing *visible* — the same rule
    ``GET /api/storage/stats`` applies — so it sees no artifacts rather
    than persisted ones its current mode says it cannot see.

    Raises whatever the storage raises: a receipt read that fails is a
    fault in the agent's own database, not an empty set.
    """
    # Function-local: ``endpoints/__init__`` imports the sovereignty router,
    # which imports this module, so a module-level import here is a cycle
    # for anyone importing ``artifacts`` first (a script, a feature package).
    from kestrel_sovereign.endpoints.agent_helpers import privacy_hides_persisted

    if storage is None or privacy_hides_persisted(storage):
        return []

    artifacts: List[OwnedArtifact] = []
    for node in await storage.get_nodes_by_type(BACKUP_ARTIFACT_NODE_TYPE):
        properties = getattr(node, "properties", None) or {}
        artifacts.append(
            OwnedArtifact(
                content_hash=_text(getattr(node, "node_id", None)),
                cid=_text(properties.get("ipfs_cid")),
            )
        )
    for node in await storage.get_nodes_by_type(SOVEREIGNTY_RECEIPT_NODE_TYPE):
        properties = getattr(node, "properties", None) or {}
        artifacts.append(
            OwnedArtifact(
                content_hash=_text(properties.get("content_hash")),
                cid=_text(properties.get("ipfs_cid") or properties.get("cid")),
            )
        )
    return artifacts


def owned_content_hashes(artifacts: List[OwnedArtifact]) -> set:
    return {a.content_hash for a in artifacts if a.content_hash}


def owned_cids(artifacts: List[OwnedArtifact]) -> set:
    return {a.cid for a in artifacts if a.cid}


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None
