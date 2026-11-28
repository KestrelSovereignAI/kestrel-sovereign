"""Project STRATEGY.yaml decisions into the knowledge graph as an index.

STRATEGY.yaml is canonical. It is human-editable, diffable, reviewable, and it
survives loss of the database — which is what makes an agent's decision record
its own rather than the storage layer's. The graph holds a *derived* copy so the
decision-reasoning surface has something to operate on: ``recall_decisions``
reads the copies, typed ``strategy_decision``, beside the decisions
``schema_router`` extracts from conversation, typed ``decision``; and
``mark_superseded`` needs real nodes to link with ``supersedes`` edges.

The copies were typed ``decision`` too until rv-4244-1 (#4244) split the type,
because only the extracted ones have the graph as their home. The v2 registry
seed migration retyped the copies already in the graph; ids and edges did not
change (``node_type_registry.apply_seed_v2``).

Before this, the two stores never met — `strategy_add_decision` appended to YAML
and returned, while `recall_decisions` queried a node type nothing ever wrote.
The graph reported a truthful zero for a question whose answer was sitting in a
66 KB file (#2851).

Three rules keep the direction of truth honest:

1. **The projection never writes back to YAML.** A derived index that edits its
   own source is no longer derived.
2. **Re-projecting is idempotent.** Node ids are derived from the entry's own
   content, so the same decision always lands on the same node.
3. **Supersession is the file's, like everything else about a decision.** An
   entry's ``superseded_by``/``superseded_at``/``superseded_reason`` project to
   the node, and their absence projects as absence. Superseding a decision
   through the graph writes the file first and re-projects the node from it
   (:func:`supersede_entries`), so the graph is never the only record. When the
   node and the file disagree, the file wins and the disagreement is logged
   (#4154). This used to be the other way round: the graph owned supersession
   and a rebuild carried it across, so a decision the file marked superseded
   still came back from ``recall_decisions`` as current.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional, Set

from kestrel_sovereign.storage.node_type_registry import (
    DECISION_NODE_TYPE,
    STRATEGY_DECISION_NODE_TYPE,
    STRATEGY_DECISION_SOURCE,
    is_pre_split_strategy_decision,
    strategy_decision_id_prefix,
)

from .timestamps import stamp_created_at

logger = logging.getLogger(__name__)

#: The types this projection's own rows have: the one it writes, and the one
#: it wrote before rv-4244-1. A row of the older type appears only from a
#: writer or a restore that predates the split, and reconciliation covers it.
PROJECTED_NODE_TYPES = (STRATEGY_DECISION_NODE_TYPE, DECISION_NODE_TYPE)

#: Marks a node as created by THIS projection. Reconciliation deletes only
#: rows carrying it, so a decision node written by anything else is never
#: removed on our behalf.
_PROJECTION_SOURCE = STRATEGY_DECISION_SOURCE

#: A decision entry's supersession, under the names the strategy ledger uses
#: for a superseded pattern. Each is read from the STRATEGY.yaml entry and
#: projected to the node property of the same name; an entry without one
#: projects a node without it (#4154).
SUPERSESSION_FIELDS = (
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
    return f"{strategy_decision_id_prefix(agent_id)}{digest}"


def _entry_properties(agent_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """The node properties a decision entry projects to.

    ``agent_id`` is load-bearing: ``recall_decisions`` filters on it, so a
    node missing it is written but unreachable. ``created_at`` orders the
    recall; it is stamped under the graph contract from the entry's date and
    left ABSENT when the entry has none (#3255), and an unstamped node sorts
    last on both backends, so it can be pushed off a limited page by stamped
    ones but is never unreachable.
    """
    text = str(entry.get("decision") or "").strip()
    properties = {
        "agent_id": agent_id,
        "text": text,
        "rationale": str(entry.get("rationale") or ""),
        "impact": str(entry.get("impact") or ""),
        "session": str(entry.get("session") or ""),
        # Provenance: this node is a projection, and the file is the original.
        # Anything reasoning over it should know it cannot be edited here.
        "claim_source": "strategy_yaml",
        "source": _PROJECTION_SOURCE,
    }
    # Supersession comes from the entry, so ``recall_decisions`` excludes a
    # decision exactly when the file says it was superseded (#4154). A field
    # the entry does not carry is left absent, not empty.
    properties.update(supersession_of(entry))
    # The entry's day at midnight UTC, in the graph contract; absent when the
    # entry has no date (#3255).
    return stamp_created_at(properties, entry.get("date"))


def supersession_of(record: Dict[str, Any]) -> Dict[str, str]:
    """The supersession fields a YAML entry or node carries, as text, omitting
    empty ones -- one reading for both, so comparing them compares meaning.

    Text because the graph stores JSON and a hand-written ``superseded_at:
    2026-07-05`` loads as a ``date``. ``str`` of it is the day as written, the
    same rule the ledger projection applies to a superseded pattern.
    """
    supersession: Dict[str, str] = {}
    for key in SUPERSESSION_FIELDS:
        value = record.get(key)
        text = "" if value is None else str(value).strip()
        if text:
            supersession[key] = text
    return supersession


def entries_for_decision(
    entries: List[Any], agent_id: str, node_id: str
) -> List[Dict[str, Any]]:
    """The STRATEGY.yaml entries that project to the decision node ``node_id``,
    in file order. Entries sharing a date and text share a node."""
    return [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and str(entry.get("decision") or "").strip()
        and strategy_decision_node_id(agent_id, entry) == node_id
    ]


def agreed_supersession(entries: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """The supersession every one of ``entries`` records (see
    :func:`supersession_of`), or None when they record different ones -- as
    hand-edited entries sharing a node can, and then no node can show both."""
    found = [supersession_of(entry) for entry in entries]
    first = found[0] if found else {}
    return first if all(other == first for other in found) else None


def supersede_entries(
    entries: List[Any],
    agent_id: str,
    node_id: str,
    *,
    superseded_by: str,
    superseded_at: str,
    reason: Optional[str] = None,
) -> int:
    """Record in STRATEGY.yaml's entries that the decision ``node_id`` was superseded.

    Mutates the entries that project to ``node_id`` and returns how many did:
    zero means the file holds no such decision, and the caller must refuse
    rather than record a supersession the canonical file cannot hold. Entries
    sharing a date and text share a node, so every one of them is marked. A
    reason replaces a previous one, and no reason removes it, so a decision
    superseded again does not keep the earlier supersession's reason.
    """
    matched = 0
    for entry in entries_for_decision(entries, agent_id, node_id):
        entry["superseded_by"] = superseded_by
        entry["superseded_at"] = superseded_at
        if reason:
            entry["superseded_reason"] = reason
        else:
            entry.pop("superseded_reason", None)
        matched += 1
    return matched


def is_strategy_decision(node: Any) -> bool:
    """Whether ``node`` is a decision this projection derived from STRATEGY.yaml.

    Such a node is a copy, so a change to it belongs in the file: superseding
    one through the graph alone would be undone by the next reindex.

    Its registered type decides: a ``strategy_decision`` is one, whatever its
    properties hold. So, for now, is a node still typed ``decision`` that
    carries this projection's marker (rv-4244-1 §4): one from a writer or a
    restore that predates the split, which would otherwise get a graph-only
    supersession the next reindex undoes. That stays until every path that
    brings graph rows back applies the split's retype (au-4244-1 C4, #4249).
    """
    if node is None:
        return False
    node_type = getattr(node, "node_type", None)
    if node_type == STRATEGY_DECISION_NODE_TYPE:
        return True
    return is_pre_split_strategy_decision(node_type, getattr(node, "properties", None))


def _is_own_projection(node: Any) -> bool:
    """Whether this projection wrote ``node``: either of its types, and its marker."""
    return (
        getattr(node, "node_type", None) in PROJECTED_NODE_TYPES
        and (getattr(node, "properties", None) or {}).get("source") == _PROJECTION_SOURCE
    )


@dataclass(frozen=True)
class DecisionSupersession:
    """What superseding a STRATEGY.yaml decision did, for the caller to report.

    ``recorded`` is the canonical fact: STRATEGY.yaml now says the decision
    was superseded. When it is false nothing was written anywhere, and
    ``conflict`` says whether that was a refusal to write over a file that
    changed on disk or no longer holds the decision. ``reread`` says the file
    had changed since it was read, and has been read again and its decisions
    re-projected, so the caller can see the change before it supersedes
    again. ``projected`` says the node was re-projected from the file; when
    it is false the file still holds the supersession and the next reindex
    projects it.
    """

    recorded: bool
    error: Optional[str] = None
    conflict: bool = False
    reread: bool = False
    projected: bool = False


def _label_for(entry: Dict[str, Any], limit: int = 120) -> str:
    text = str(entry.get("decision") or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


async def project_decisions(
    graph_store,
    agent_id: str,
    entries: List[Dict[str, Any]],
    *,
    changed: Collection[str] = (),
) -> Dict[str, Any]:
    """Upsert STRATEGY.yaml decision entries as graph nodes.

    Best-effort by design: the graph is an index, so a failure to write it must
    never fail the strategic-memory write that YAML already persisted. The
    canonical record is on disk either way, and the next projection reconciles.

    A node whose supersession differs from its entry's is rewritten to the
    entry's and the difference is logged as a disagreement (#4154). ``changed``
    names the nodes whose entries the caller has just rewritten itself: there
    the node trailing the file is the change being projected, not a
    disagreement.

    Returns a content-free report — counts and reasons only, never decision
    text, since callers surface it to logs.
    """
    report: Dict[str, Any] = {
        "projected": 0, "skipped": 0, "failed": 0, "disagreements": 0,
    }
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
            # A failed READ must not be treated as "no node": the write below
            # is conditional on the state it read, and a write made without
            # knowing that state cannot be conditional on it.
            logger.debug("decision projection could not read %s: %s", node_id, e)
            report["failed"] += 1
            continue

        expected = existing.properties if existing is not None else None
        if existing is not None:
            graph_says = supersession_of(existing.properties or {})
            file_says = supersession_of(entry)
            if graph_says != file_says and node_id not in changed:
                # The file wins. Ids and field names only: a reason is
                # free text, and this goes to the log.
                report["disagreements"] += 1
                logger.warning(
                    "decision %s: the graph and STRATEGY.yaml disagree on its "
                    "supersession (%s differ; graph superseded_by=%r, file "
                    "superseded_by=%r). STRATEGY.yaml is canonical, so the "
                    "node now follows the file.",
                    node_id,
                    ", ".join(
                        sorted(
                            key for key in SUPERSESSION_FIELDS
                            if graph_says.get(key) != file_says.get(key)
                        )
                    ),
                    graph_says.get("superseded_by"),
                    file_says.get("superseded_by"),
                )

        # A swap writes properties only, so a node that predates the split
        # keeps its type here; reconciliation covers it under either.
        node = GraphNode(
            node_id=node_id,
            node_type=STRATEGY_DECISION_NODE_TYPE,
            label=_label_for(entry),
            properties=properties,
        )
        try:
            # Conditional, not a clobber. add_node is a whole-row upsert, so a
            # write landing between the read above and this one would be
            # overwritten by the stale snapshot we just read. CAS makes the
            # check and the write one serialized unit; a lost race means
            # someone else changed the node, and the next reindex projects
            # the file onto whatever they left.
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
    nodes: List[Any] = []
    try:
        for node_type in PROJECTED_NODE_TYPES:
            nodes.extend(await graph_store.get_nodes_by_type(node_type) or [])
    except Exception as e:  # noqa: BLE001
        logger.debug("decision reconcile could not list nodes: %s", e)
        return 0
    removed = 0
    for node in nodes:
        properties = node.properties or {}
        # Only this agent's rows, and only rows this projection created —
        # a decision node written by anything else is not ours to delete.
        if properties.get("agent_id") != agent_id:
            continue
        if not _is_own_projection(node):
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
