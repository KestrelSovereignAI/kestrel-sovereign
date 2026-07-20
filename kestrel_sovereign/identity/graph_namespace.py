"""Stable, collision-resistant namespaces for imported graph records."""

from __future__ import annotations

import hashlib


_GRAPH_IMPORT_PREFIX = "kestrel-import-v2:"


def imported_graph_node_prefix(agent_id: str) -> str:
    """Return the namespace owned by one complete agent identifier.

    The legacy importer used ``agent_id[:20]``. For ordinary ``did:pkh``
    identifiers those characters are the DID method and chain prefix, not the
    account, so unrelated agents collided. Hashing the complete identifier
    makes the namespace fixed-size while preserving every distinguishing byte.
    """
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("Imported graph namespaces require a non-empty agent_id")
    digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
    return f"{_GRAPH_IMPORT_PREFIX}{digest}:"


def namespace_imported_graph_node(agent_id: str, raw_id: object) -> str:
    """Namespace one package-supplied graph id for ``agent_id``."""
    return f"{imported_graph_node_prefix(agent_id)}{raw_id}"


def strip_imported_graph_namespace(agent_id: str, node_id: str) -> str:
    """Strip this agent's current or legacy import namespace.

    Recognizing the legacy ``agent_id[:20]_`` form keeps packages exported
    from pre-v2 databases stable across the first migration after upgrade.
    Prefixes belonging to any other agent are left untouched.
    """
    if not node_id:
        return node_id

    current_prefix = imported_graph_node_prefix(agent_id)
    if node_id.startswith(current_prefix):
        return node_id[len(current_prefix):]

    legacy_prefix = f"{agent_id[:20]}_"
    if node_id.startswith(legacy_prefix):
        return node_id[len(legacy_prefix):]
    return node_id
