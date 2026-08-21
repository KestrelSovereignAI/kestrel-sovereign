"""Project STRATEGY.yaml decisions into the knowledge graph as an index.

STRATEGY.yaml is canonical. It is human-editable, diffable, reviewable, and it
survives loss of the database — which is what makes an agent's decision record
its own rather than the storage layer's. The graph holds a *derived* copy so the
decision-reasoning surface has something to operate on: ``recall_decisions``
reads ``decision``-typed nodes, and ``mark_superseded`` needs real nodes to link
with ``supersedes`` edges.

Before this, the two stores never met — `strategy_add_decision` appended to YAML
and returned, while `recall_decisions` queried a node type nothing ever wrote.
The graph reported a truthful zero for a question whose answer was sitting in a
66 KB file (#2851).

Two rules keep the direction of truth honest:

1. **The projection never writes back to YAML.** A derived index that edits its
   own source is no longer derived.
2. **Re-projecting is idempotent and non-destructive.** Node ids are derived
   from the entry's own content, so the same decision always lands on the same
   node, and graph-only state that YAML cannot express — chiefly supersession —
   is carried across rather than flattened. Rebuilding the index must never
   silently un-supersede a decision.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DECISION_NODE_TYPE = "decision"

#: Marks a node as created by THIS projection. Reconciliation deletes only
#: rows carrying it, so a decision node written by anything else is never
#: removed on our behalf.
_PROJECTION_SOURCE = "strategic_memory"

#: Node properties owned by the graph, not by STRATEGY.yaml. They are written
#: by `mark_superseded` and have no representation in the YAML entry, so a
#: rebuild must preserve them or it would quietly revive a superseded decision.
_GRAPH_OWNED_PROPERTIES = (
    "superseded_by",
    "superseded_at",
    "superseded_reason",
)


def strategy_decision_node_id(agent_id: str, entry: Dict[str, Any]) -> str:
    """Stable node id for one STRATEGY.yaml decision entry.

    Derived from the entry's own content — its date and decision text — so the
    same entry always projects onto the same node and re-projection upserts
    rather than duplicating. Deliberately NOT a random UUID: the index has to be
    rebuildable from YAML alone, which is only true if identity is a function of
    the source.

    Rationale and impact are excluded from the digest so that correcting a
    typo in either edits the existing node instead of orphaning it and minting
    a second.
    """
    date_part = str(entry.get("date") or "")
    text_part = str(entry.get("decision") or "").strip().lower()
    digest = hashlib.sha1(
        f"{date_part}\x00{text_part}".encode("utf-8")
    ).hexdigest()[:16]
    return f"decision:{agent_id}:strategy:{digest}"


def _entry_properties(agent_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """The node properties a decision entry projects to.

    ``agent_id`` and ``created_at`` are load-bearing, not decoration:
    ``recall_decisions`` filters on the former and orders on the latter, so a
    node missing either is written but unreachable.
    """
    text = str(entry.get("decision") or "").strip()
    return {
        "agent_id": agent_id,
        "text": text,
        "created_at": str(entry.get("date") or ""),
        "rationale": str(entry.get("rationale") or ""),
        "impact": str(entry.get("impact") or ""),
        "session": str(entry.get("session") or ""),
        # Provenance: this node is a projection, and the file is the original.
        # Anything reasoning over it should know it cannot be edited here.
        "claim_source": "strategy_yaml",
        "source": _PROJECTION_SOURCE,
    }


def _label_for(entry: Dict[str, Any], limit: int = 120) -> str:
    text = str(entry.get("decision") or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


async def project_decisions(
    graph_store,
    agent_id: str,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Upsert STRATEGY.yaml decision entries as graph nodes.

    Best-effort by design: the graph is an index, so a failure to write it must
    never fail the strategic-memory write that YAML already persisted. The
    canonical record is on disk either way, and the next projection reconciles.

    Returns a content-free report — counts and reasons only, never decision
    text, since callers surface it to logs.
    """
    report: Dict[str, Any] = {"projected": 0, "skipped": 0, "failed": 0}
    if graph_store is None:
        report["skipped_reason"] = "no_graph_store"
        report["skipped"] = len(entries)
        return report

    try:
        from kestrel_sovereign.storage.async_graph_store import (
            GraphNode,
            NodeSwapResult,
        )
    except Exception as e:  # noqa: BLE001 - never break a YAML write on an import
        logger.debug("decision projection unavailable: %s", e)
        report["skipped"] = len(entries)
        report["skipped_reason"] = "graph_node_unavailable"
        return report

    # Membership in YAML, not success of the write. Deriving the keep-set from
    # what projected would make a transient read or write failure look like a
    # deletion from the canonical file, and reconciliation would then delete a
    # node that YAML still contains — turning a recoverable blip into data loss.
    expected_ids: Set[str] = {
        strategy_decision_node_id(agent_id, entry)
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("decision") or "").strip()
    }
    for entry in entries:
        if not isinstance(entry, dict):
            report["skipped"] += 1
            continue
        text = str(entry.get("decision") or "").strip()
        if not text:
            # An entry with no decision text has nothing to reason over.
            report["skipped"] += 1
            continue

        node_id = strategy_decision_node_id(agent_id, entry)
        properties = _entry_properties(agent_id, entry)

        try:
            existing = await graph_store.get_node(node_id)
        except Exception as e:  # noqa: BLE001
            # A failed READ must not be treated as "no node". Doing so drops
            # the graph-owned properties below and the write then revives a
            # superseded decision — the exact loss this projection exists to
            # prevent, caused by an error it merely failed to hear.
            logger.debug("decision projection could not read %s: %s", node_id, e)
            report["failed"] += 1
            continue

        expected = existing.properties if existing is not None else None
        if existing is not None:
            existing_props = existing.properties or {}
            # Carry across what the graph owns and YAML cannot express.
            # Without this, re-projecting a superseded decision would drop its
            # supersession and quietly restore it to recall.
            for key in _GRAPH_OWNED_PROPERTIES:
                if key in existing_props:
                    properties[key] = existing_props[key]

        node = GraphNode(
            node_id=node_id,
            node_type=DECISION_NODE_TYPE,
            label=_label_for(entry),
            properties=properties,
        )
        try:
            # Conditional, not a clobber. add_node is a whole-row upsert, so a
            # mark_superseded landing between the read above and this write
            # would be overwritten by the stale snapshot we just read — and the
            # decision would silently come back. CAS makes the check and the
            # write one serialized unit; a lost race means someone else changed
            # the node, and the next reindex projects the newer state.
            outcome = await graph_store.compare_and_swap_node(
                node_id, expected, node
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("decision projection failed for %s: %s", node_id, e)
            report["failed"] += 1
            continue
        if outcome == NodeSwapResult.SWAPPED:
            report["projected"] += 1
        else:
            logger.debug(
                "decision projection for %s did not land (%s)", node_id, outcome
            )
            report["failed"] += 1

    # Reconcile: YAML is canonical, so a decision removed or edited there must
    # stop being reachable. Upserting current entries alone leaves the old node
    # behind — an edited decision changes its content-derived id, so the
    # pre-edit node would linger and recall_decisions would return a decision
    # absent from the canonical file.
    report["removed"] = await _remove_orphans(
        graph_store, agent_id, expected_ids
    )
    return report


async def _remove_orphans(
    graph_store: Any, agent_id: str, keep: Set[str]
) -> int:
    """Delete this agent's projected nodes that YAML no longer contains."""
    try:
        nodes = await graph_store.get_nodes_by_type(DECISION_NODE_TYPE)
    except Exception as e:  # noqa: BLE001
        logger.debug("decision reconcile could not list nodes: %s", e)
        return 0
    removed = 0
    for node in nodes or []:
        properties = node.properties or {}
        # Only this agent's rows, and only rows this projection created —
        # a decision node written by anything else is not ours to delete.
        if properties.get("agent_id") != agent_id:
            continue
        if properties.get("source") != _PROJECTION_SOURCE:
            continue
        if node.node_id in keep:
            continue
        try:
            await graph_store.delete_node(node.node_id)
            removed += 1
        except Exception as e:  # noqa: BLE001
            logger.debug("decision reconcile could not delete %s: %s", node.node_id, e)
    return removed


def decision_entries(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The decision entries in a loaded STRATEGY.yaml, or an empty list."""
    if not isinstance(data, dict):
        return []
    entries = data.get("decisions")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]
