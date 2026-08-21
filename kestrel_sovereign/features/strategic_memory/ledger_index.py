"""Project the strategy ledger into the knowledge graph as an index.

Same shape as :mod:`decision_index`, and deliberately so — #2851 settled the
direction of truth for strategic memory and landing a different one here would
contradict it. ``STRATEGY_LEDGER.yaml`` is canonical; the graph holds a derived
copy so patterns and blockers are reachable by query instead of by being
poured into the system prompt and truncated at a byte offset (#2954).

The same two rules apply:

1. **The projection never writes back to the ledger.** A derived index that
   edits its own source is no longer derived.
2. **Re-projecting is idempotent and non-destructive.** Node ids are derived
   from the row's own ``id``, so the same row always lands on the same node,
   and graph-only state is carried across rather than flattened.

Where this differs from decisions: the ledger *can* express supersession, so
the canonical file wins whenever it says anything. Graph-owned values are only
carried across for rows the ledger is silent about — otherwise un-superseding a
pattern in YAML could never take effect in the index.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from .ledger import (
    BLOCKERS_KEY,
    PATTERNS_KEY,
    blocker_row_id,
    is_active_blocker,
    is_active_pattern,
    pattern_row_id,
)

logger = logging.getLogger(__name__)

PATTERN_NODE_TYPE = "strategy_pattern"
BLOCKER_NODE_TYPE = "strategy_blocker"

#: Marks a node as created by THIS projection. Reconciliation deletes only
#: rows carrying it, so a node written by anything else is never removed on
#: our behalf.
_PROJECTION_SOURCE = "strategic_memory"

#: Properties the graph may own that the ledger has no field for. The ledger
#: expresses supersession itself, so these are only preserved when the
#: canonical row is silent — see the module docstring.
_GRAPH_OWNED_PROPERTIES = (
    "superseded_by",
    "superseded_at",
    "superseded_reason",
)


def ledger_node_id(node_type: str, agent_id: str, row_id: str) -> str:
    """Stable node id for one ledger row.

    A function of the row's own ``id``, which is itself content-derived when
    the ledger mints it. Deliberately NOT a random UUID: the index has to be
    rebuildable from the file alone, which is only true if identity is a
    function of the source.
    """
    return f"{node_type}:{agent_id}:{row_id}"


def _label(text: str, limit: int = 120) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _pattern_properties(agent_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """The node properties a pattern row projects to.

    ``agent_id`` and ``created_at`` are load-bearing, not decoration: scoped
    queries filter on the former and order on the latter, so a node missing
    either is written but unreachable.
    """
    return {
        "agent_id": agent_id,
        "row_id": str(row.get("id") or ""),
        "text": str(row.get("pattern") or "").strip(),
        "implication": str(row.get("implication") or ""),
        "origin": str(row.get("source") or ""),
        "created_at": str(row.get("recorded_at") or ""),
        "status": "active" if is_active_pattern(row) else "superseded",
        "superseded_at": str(row.get("superseded_at") or ""),
        "superseded_by": str(row.get("superseded_by") or ""),
        "superseded_reason": str(row.get("superseded_reason") or ""),
        # Provenance: this node is a projection, and the file is the original.
        # Anything reasoning over it should know it cannot be edited here.
        "claim_source": "strategy_ledger_yaml",
        "source": _PROJECTION_SOURCE,
    }


def _blocker_properties(agent_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "agent_id": agent_id,
        "row_id": str(row.get("id") or ""),
        "text": str(row.get("title") or "").strip(),
        "issue": str(row.get("issue") or ""),
        "severity": str(row.get("severity") or ""),
        "owner": str(row.get("owner") or ""),
        "notes": str(row.get("notes") or ""),
        "created_at": str(row.get("blocked_since") or ""),
        "status": "active" if is_active_blocker(row) else "resolved",
        "resolved_at": str(row.get("resolved_at") or ""),
        "resolution": str(row.get("resolution") or ""),
        "claim_source": "strategy_ledger_yaml",
        "source": _PROJECTION_SOURCE,
    }


def _row_id(row: Dict[str, Any], minter: Callable[[Dict[str, Any]], str]) -> str:
    return str(row.get("id") or "").strip() or minter(row)


async def project_ledger(
    graph_store,
    agent_id: str,
    ledger_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Upsert ledger rows as graph nodes.

    Best-effort by design: the graph is an index, so a failure to write it must
    never fail the ledger write that YAML already persisted. The canonical
    record is on disk either way, and the next projection reconciles.

    Returns a content-free report — counts and reasons only, never pattern or
    blocker text, since callers surface it to logs.
    """
    report: Dict[str, Any] = {"projected": 0, "skipped": 0, "failed": 0, "removed": 0}
    patterns = _dict_rows(ledger_data, PATTERNS_KEY)
    blockers = _dict_rows(ledger_data, BLOCKERS_KEY)
    total = len(patterns) + len(blockers)

    if graph_store is None:
        report["skipped_reason"] = "no_graph_store"
        report["skipped"] = total
        return report

    try:
        from kestrel_sovereign.storage.async_graph_store import (
            GraphNode,
            NodeSwapResult,
        )
    except Exception as e:  # noqa: BLE001 - never break a YAML write on an import
        logger.debug("ledger projection unavailable: %s", e)
        report["skipped"] = total
        report["skipped_reason"] = "graph_node_unavailable"
        return report

    for node_type, rows, minter, properties_fn, text_key in (
        (PATTERN_NODE_TYPE, patterns, pattern_row_id, _pattern_properties, "pattern"),
        (BLOCKER_NODE_TYPE, blockers, blocker_row_id, _blocker_properties, "title"),
    ):
        await _project_section(
            graph_store,
            GraphNode,
            NodeSwapResult,
            agent_id,
            node_type,
            rows,
            minter,
            properties_fn,
            text_key,
            report,
        )
    return report


def _dict_rows(data: Optional[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    value = data.get(key)
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


async def _project_section(
    graph_store,
    GraphNode,
    NodeSwapResult,
    agent_id: str,
    node_type: str,
    rows: List[Dict[str, Any]],
    minter: Callable[[Dict[str, Any]], str],
    properties_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    text_key: str,
    report: Dict[str, Any],
) -> None:
    # Membership in the ledger, not success of the write. Deriving the keep-set
    # from what projected would make a transient read or write failure look
    # like a deletion from the canonical file, and reconciliation would then
    # delete a node the ledger still contains — turning a recoverable blip into
    # data loss.
    expected_ids: Set[str] = {
        ledger_node_id(node_type, agent_id, _row_id(row, minter))
        for row in rows
        if str(row.get(text_key) or "").strip()
    }

    for row in rows:
        if not str(row.get(text_key) or "").strip():
            # A row with no text has nothing to reason over.
            report["skipped"] += 1
            continue

        node_id = ledger_node_id(node_type, agent_id, _row_id(row, minter))
        properties = properties_fn(agent_id, row)

        try:
            existing = await graph_store.get_node(node_id)
        except Exception as e:  # noqa: BLE001
            # A failed READ must not be treated as "no node". Doing so drops
            # the graph-owned properties below and the write then revives a
            # superseded row — the exact loss this projection exists to
            # prevent, caused by an error it merely failed to hear.
            logger.debug("ledger projection could not read %s: %s", node_id, e)
            report["failed"] += 1
            continue

        expected = existing.properties if existing is not None else None
        if existing is not None:
            existing_props = existing.properties or {}
            for key in _GRAPH_OWNED_PROPERTIES:
                # The canonical file wins when it says anything; the graph's
                # value survives only where the ledger is silent.
                if properties.get(key):
                    continue
                if existing_props.get(key):
                    properties[key] = existing_props[key]
                    properties["status"] = "superseded"

        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            label=_label(row.get(text_key)),
            properties=properties,
        )
        try:
            # Conditional, not a clobber. add_node is a whole-row upsert, so a
            # concurrent write landing between the read above and this one
            # would be overwritten by the stale snapshot we just read. CAS
            # makes the check and the write one serialized unit; a lost race
            # means someone else changed the node, and the next reindex
            # projects the newer state.
            outcome = await graph_store.compare_and_swap_node(
                node_id, expected, node
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("ledger projection failed for %s: %s", node_id, e)
            report["failed"] += 1
            continue
        if outcome == NodeSwapResult.SWAPPED:
            report["projected"] += 1
        else:
            logger.debug(
                "ledger projection for %s did not land (%s)", node_id, outcome
            )
            report["failed"] += 1

    # Reconcile: the ledger is canonical, so a row removed there must stop
    # being reachable. Upserting current rows alone leaves the old node behind.
    report["removed"] += await _remove_orphans(
        graph_store, agent_id, node_type, expected_ids
    )


async def _remove_orphans(
    graph_store: Any, agent_id: str, node_type: str, keep: Set[str]
) -> int:
    """Delete this agent's projected nodes that the ledger no longer contains."""
    try:
        nodes = await graph_store.get_nodes_by_type(node_type)
    except Exception as e:  # noqa: BLE001
        logger.debug("ledger reconcile could not list %s nodes: %s", node_type, e)
        return 0
    removed = 0
    for node in nodes or []:
        properties = node.properties or {}
        # Only this agent's rows, and only rows this projection created —
        # a node written by anything else is not ours to delete.
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
            logger.debug(
                "ledger reconcile could not delete %s: %s", node.node_id, e
            )
    return removed


def search_rows(
    ledger_data: Dict[str, Any],
    query: str,
    kind: str = "all",
    limit: int = 10,
    include_retired: bool = False,
) -> List[Dict[str, Any]]:
    """Keyword-rank ledger rows for the query layer.

    Deliberately named for what it is: term-overlap scoring over the ledger's
    own rows, not embedding search. The graph projection makes the same content
    reachable to the semantic surfaces that read typed nodes; this function is
    the direct, always-available path that does not depend on an embedding
    provider being configured. Reporting it as "semantic" when no provider is
    present is exactly the kind of flattering claim the honesty layer exists to
    catch.
    """
    terms = [t for t in str(query or "").lower().split() if t]
    if not terms:
        return []

    sections: List[tuple] = []
    if kind in ("all", "patterns", "pattern"):
        sections.append((PATTERNS_KEY, "pattern", is_active_pattern))
    if kind in ("all", "blockers", "blocker"):
        sections.append((BLOCKERS_KEY, "blocker", is_active_blocker))

    scored: List[Dict[str, Any]] = []
    for key, row_kind, is_active in sections:
        for row in _dict_rows(ledger_data, key):
            if not include_retired and not is_active(row):
                continue
            row_id = str(row.get("id") or "").lower()
            # ``id`` is deliberately kept out of the haystack. It is a hex
            # digest, so substring-matching it scores unrelated rows full
            # marks -- "pattern 357" matched a row whose id merely contained
            # "357". An id is an address, so it is matched exactly instead,
            # which still lets a caller paste one straight back in.
            haystack = " ".join(
                str(value)
                for name, value in row.items()
                if name != "id" and isinstance(value, (str, int, float))
            ).lower()
            hits = sum(1 for t in terms if t in haystack or t == row_id)
            if hits == 0:
                continue
            scored.append(
                {
                    "id": str(row.get("id") or ""),
                    "kind": row_kind,
                    "score": round(hits / len(terms), 3),
                    "active": is_active(row),
                    "row": row,
                }
            )

    scored.sort(key=lambda m: (-m["score"], m["id"]))
    return scored[: max(0, limit)]
