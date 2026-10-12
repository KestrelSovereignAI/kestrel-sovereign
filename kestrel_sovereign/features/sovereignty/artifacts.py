"""The export artifacts an agent durably owns.

A sovereignty export or backup writes two things: bytes into the host's
storage cache (one ``storage_cache/`` directory, shared by every agent on
the host and by every producer that ever ran here) and a receipt into the
agent's *own* storage — a ``backup_artifact`` node whose id is the content
hash, and a ``sovereignty_receipt`` node naming the same hash and CID.
Both record the name the cache holds the bytes under.

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

from kestrel_sovereign.filecoin_adapter import is_cache_name_of

BACKUP_ARTIFACT_NODE_TYPE = "backup_artifact"
SOVEREIGNTY_RECEIPT_NODE_TYPE = "sovereignty_receipt"


@dataclass(frozen=True)
class OwnedArtifact:
    """One export/backup artifact the bound agent recorded as its own.

    ``cache_name`` is the host-cache name the receipt records the artifact
    under (#2873), as recorded, or None for a receipt written before the
    name was recorded. A receipt that records one, valid or not, is not a
    legacy receipt.
    """

    content_hash: Optional[str]
    cid: Optional[str]
    cache_name: Any = None


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
                cache_name=properties.get("cache_name"),
            )
        )
    for node in await storage.get_nodes_by_type(SOVEREIGNTY_RECEIPT_NODE_TYPE):
        properties = getattr(node, "properties", None) or {}
        artifacts.append(
            OwnedArtifact(
                content_hash=_text(properties.get("content_hash")),
                cid=_text(properties.get("ipfs_cid") or properties.get("cid")),
                cache_name=properties.get("cache_name"),
            )
        )
    return artifacts


def owned_cache_names(artifacts: List[OwnedArtifact]) -> set:
    """The host-cache names the artifacts are cached under.

    A receipt that records a cache name confers that name and nothing else
    (#2873). An encrypted envelope is cached under ``{hash}.{owner_id}``,
    keyed by the agent it was sealed for, while another agent's export of
    the same plaintext may sit under the bare content hash: a plaintext
    export, or an encrypted one from before the envelope, with its sidecar
    and perhaps its key. The content hash alone is shared, so it is not
    this receipt's to confer. A recorded name that is not a cache name of
    the receipt's own content hash confers nothing.

    A receipt that records no cache name was written before the name was
    recorded, when ``store_content`` cached every export under its content
    hash, so it confers its content hash. That hash must be a lowercase
    SHA-256, the name ``store_content`` gives content: a receipt naming
    anything else names no cache entry, in particular not another agent's
    envelope, whose name a receipt could otherwise hold as its content hash.
    """
    names = set()
    for artifact in artifacts:
        name = artifact.content_hash if artifact.cache_name is None else artifact.cache_name
        if artifact.content_hash and is_cache_name_of(name, artifact.content_hash):
            names.add(name)
    return names


def owned_legacy_content_hashes(artifacts: List[OwnedArtifact]) -> set:
    """The content hashes of the artifacts whose receipts record no cache name.

    Only such an export, cached under its bare content hash, can have the
    sidecar ``key_{hash}.key`` that content encrypted before F187 was
    written with. A receipt that records a cache name confers no such
    sidecar: one beside the bare hash belongs to another export of the same
    plaintext, perhaps another agent's (#2873).
    """
    return {
        artifact.content_hash
        for artifact in artifacts
        if artifact.cache_name is None
        and artifact.content_hash
        and is_cache_name_of(artifact.content_hash, artifact.content_hash)
    }


def owned_cids(artifacts: List[OwnedArtifact]) -> set:
    return {a.cid for a in artifacts if a.cid}


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None
