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

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Mapping

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
        # ``#42`` names an issue only relative to a repository. The ledger
        # records ``repo`` for exactly that reason, so dropping it here hands
        # every consumer of the index an ambiguous reference and leaves
        # disambiguation to whichever repo happens to contain a number 42.
        "repo": str(row.get("repo") or ""),
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


#: Properties the digest below must ignore, because the graph may legitimately
#: hold a value for them that the ledger does not -- see
#: :data:`_GRAPH_OWNED_PROPERTIES` and the ``status`` they imply.
_DIGEST_EXCLUDED = frozenset(_GRAPH_OWNED_PROPERTIES) | {"status"}


def properties_digest(properties: Mapping[str, Any]) -> str:
    """Fingerprint of the ledger-owned half of a projected node.

    Membership answers "is there a node for this row?", which is a strictly
    weaker question than "does the index still say what the file says". An
    id-stable edit -- a blocker's repo or severity, a pattern's implication --
    leaves membership and status identical while the indexed row goes stale,
    and a check that stopped at membership certified that as clean (#3064).

    A node written before a property was ADDED to the projection digests
    differently from one written after, which is the intended answer: such a
    node is stale until the next reprojection, and saying so is what makes the
    upgrade visible instead of silent.
    """
    material = {
        str(key): str(value)
        for key, value in (properties or {}).items()
        if key not in _DIGEST_EXCLUDED
    }
    return hashlib.blake2s(
        json.dumps(material, sort_keys=True).encode("utf-8"), digest_size=8
    ).hexdigest()



@dataclass(frozen=True)
class LedgerSection:
    """One kind of ledger row, and everything both halves of the index need.

    The projection and the recall each have to answer "which rows belong in
    the index?", and #3064 was filed because they answered it separately: the
    writer skipped text-less rows and the reader compared against a list of
    *active* rows the caller had computed, so ``include_superseded=True``
    measured itself against the wrong baseline. Two call sites deriving one
    fact is the bug; naming the fact once is the fix.
    """

    node_type: str
    ledger_key: str
    noun: str
    include_flag_name: str
    text_key: str
    minter: Callable[[Dict[str, Any]], str]
    is_active: Callable[[Dict[str, Any]], bool]
    properties: Callable[[str, Dict[str, Any]], Dict[str, Any]]
    #: Whether the INDEX may hold a retirement the ledger does not.
    #:
    #: True for patterns: :data:`_GRAPH_OWNED_PROPERTIES` is carried across for
    #: rows the ledger is silent about, so a node marked superseded beside an
    #: unsuperseded YAML row is the documented design and reprojection keeps it
    #: that way. False for blockers, whose ``status`` is a pure function of the
    #: ledger's ``resolved_at`` -- there, the same shape is a projection that
    #: has not landed, and an operator reopening a blocker by hand would
    #: otherwise get a certified-clean empty recall (#3064).
    graph_may_retire: bool

    def rows(self, ledger_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _dict_rows(ledger_data, self.ledger_key)

    def is_projectable(self, row: Dict[str, Any]) -> bool:
        """A row with no text has nothing to reason over, so it is not indexed.

        The recall's completeness check has to apply the same rule or a
        text-less row reads as permanently missing from the index.
        """
        return bool(str(row.get(self.text_key) or "").strip())

    def row_id(self, row: Dict[str, Any]) -> str:
        return _row_id(row, self.minter)

    def node_properties(self, agent_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """The node's properties, with ``row_id`` normalized to the node's own.

        The properties functions read ``row["id"]`` verbatim while
        :func:`_row_id` strips it, so a hand-edited id carrying whitespace
        addressed the node under one spelling and advertised another. The
        membership check reads the property, so a freshly and healthily
        rebuilt node came back reported as missing AND orphaned, forever, and
        no restart could clear it. One rule, computed once (#3064).
        """
        properties = self.properties(agent_id, row)
        properties["row_id"] = self.row_id(row)
        return properties

    def content_digest(self, agent_id: str, row: Dict[str, Any]) -> str:
        """Digest of the properties the LEDGER owns for this row."""
        return properties_digest(self.node_properties(agent_id, row))

    def node_id(self, agent_id: str, row: Dict[str, Any]) -> str:
        return ledger_node_id(self.node_type, agent_id, self.row_id(row))

    def expected_rows(
        self, ledger_data: Optional[Dict[str, Any]], *, include_retired: bool
    ) -> List[Dict[str, Any]]:
        """The canonical rows a recall in this mode should find in the index.

        ``include_retired`` is the query's own flag, so the baseline follows
        the question that was asked. Computing it at the call site is what let
        a ledger of only superseded rows report a clean zero.
        """
        rows = [row for row in self.rows(ledger_data) if self.is_projectable(row)]
        if include_retired:
            return rows
        return [row for row in rows if self.is_active(row)]

    def expected_row_ids(
        self, ledger_data: Optional[Dict[str, Any]], *, include_retired: bool
    ) -> Set[str]:
        return set(
            self.expected_row_id_list(
                ledger_data, include_retired=include_retired
            )
        )

    def expected_row_id_list(
        self, ledger_data: Optional[Dict[str, Any]], *, include_retired: bool
    ) -> List[str]:
        """The ids as a LIST, so a caller can see multiplicity.

        The ledger is hand-editable and ``normalize`` only disambiguates ids it
        mints, so two rows can end up sharing one. Both then project to the
        same node and the second overwrites the first -- a set of ids hides
        that entirely, and the index reports itself complete while one
        canonical row is unreachable (#3064).
        """
        return [
            self.row_id(row)
            for row in self.expected_rows(
                ledger_data, include_retired=include_retired
            )
        ]


PATTERN_SECTION = LedgerSection(
    node_type=PATTERN_NODE_TYPE,
    ledger_key=PATTERNS_KEY,
    noun="patterns",
    include_flag_name="include_superseded",
    text_key="pattern",
    minter=pattern_row_id,
    is_active=is_active_pattern,
    properties=_pattern_properties,
    graph_may_retire=True,
)

BLOCKER_SECTION = LedgerSection(
    node_type=BLOCKER_NODE_TYPE,
    ledger_key=BLOCKERS_KEY,
    noun="blockers",
    include_flag_name="include_resolved",
    text_key="title",
    minter=blocker_row_id,
    is_active=is_active_blocker,
    properties=_blocker_properties,
    graph_may_retire=False,
)


async def project_ledger(
    graph_store,
    agent_id: str,
    ledger: Any,
) -> Dict[str, Any]:
    """Upsert ledger rows as graph nodes.

    Takes the LEDGER, not its ``data``. Reconciliation derives its keep-set
    from the rows it is given, so a failed parse — which leaves the sections
    empty — reads as "every row was deleted" and takes the derived index with
    it. A bare mapping cannot express the difference between "no rows" and
    "could not be read", so the guard had to live at the call site, where the
    next call site added would silently omit it. Passing the ledger moves the
    question into the value: readability travels with the rows.

    Best-effort by design: the graph is an index, so a failure to write it must
    never fail the ledger write that YAML already persisted. The canonical
    record is on disk either way, and the next projection reconciles.

    Returns a content-free report — counts and reasons only, never pattern or
    blocker text, since callers surface it to logs.
    """
    report: Dict[str, Any] = {"projected": 0, "skipped": 0, "failed": 0, "removed": 0}
    # A mapping is still accepted, but only when it cannot be an unreadable
    # ledger — i.e. a caller that has already established readability, or a
    # test constructing rows directly.
    if isinstance(ledger, Mapping):
        ledger_data: Dict[str, Any] = dict(ledger)
    else:
        if not getattr(ledger, "readable", True):
            report["skipped_reason"] = "ledger_unavailable"
            return report
        ledger_data = getattr(ledger, "data", {}) or {}
    sections = (PATTERN_SECTION, BLOCKER_SECTION)
    total = sum(len(section.rows(ledger_data)) for section in sections)

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

    for section in sections:
        await _project_section(
            graph_store,
            GraphNode,
            NodeSwapResult,
            agent_id,
            section,
            section.rows(ledger_data),
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
    section: LedgerSection,
    rows: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    # Membership in the ledger, not success of the write. Deriving the keep-set
    # from what projected would make a transient read or write failure look
    # like a deletion from the canonical file, and reconciliation would then
    # delete a node the ledger still contains — turning a recoverable blip into
    # data loss.
    expected_ids: Set[str] = {
        section.node_id(agent_id, row)
        for row in rows
        if section.is_projectable(row)
    }

    for row in rows:
        if not section.is_projectable(row):
            # A row with no text has nothing to reason over.
            report["skipped"] += 1
            continue

        node_id = section.node_id(agent_id, row)
        properties = section.node_properties(agent_id, row)

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
            node_type=section.node_type,
            label=_label(row.get(section.text_key)),
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
        graph_store, agent_id, section.node_type, expected_ids
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


#: The status :func:`_pattern_properties` / :func:`_blocker_properties` project
#: for a row the canonical file still holds. Named once so the recall consumers
#: filter on the same vocabulary the projection writes — a reader that invents
#: its own spelling silently returns everything, or nothing.
ACTIVE_STATUS = "active"


#: Ceiling on the membership read below. ``query_nodes_by_type_and_property``
#: clamps its limit to 10000, so a saturated read is indistinguishable from a
#: complete one -- the caller reports an unrun check rather than guessing.
MEMBERSHIP_READ_CAP = 10000


@dataclass(frozen=True)
class IndexedRow:
    """What the index says about one projected row."""

    status: str
    digest: str


async def index_membership(
    graph_store: Any, agent_id: str, node_type: str
) -> tuple[Dict[str, IndexedRow], bool]:
    """Every row this projection wrote for one agent: ``row_id -> status``.

    Deliberately status-agnostic and deliberately NOT the caller's page.
    Membership is a property of the index, and answering it from a ``LIMIT``-ed
    result is how a divergence older than the page goes unreported: an orphan
    that sorts behind every canonical row is invisible to the page and present
    in the database, so the page certifies a clean index and a larger limit
    then returns deleted guidance (#3064).

    Each entry carries the node's status AND a digest of its ledger-owned
    properties, because "there is a node for this row" is a strictly weaker
    claim than "the index still says what the file says".

    Returns ``(membership, complete)``. ``complete`` is False when the read
    saturated :data:`MEMBERSHIP_READ_CAP`, which the caller must surface as a
    check that did not run.
    """
    if graph_store is None:
        raise RuntimeError("Graph store not available")
    nodes = await graph_store.query_nodes_by_type_and_property(
        node_type,
        filters={"agent_id": agent_id, "source": _PROJECTION_SOURCE},
        order_by_created=False,
        limit=MEMBERSHIP_READ_CAP,
    ) or []
    membership = {
        str((node.properties or {}).get("row_id") or ""): IndexedRow(
            status=str((node.properties or {}).get("status") or ""),
            digest=properties_digest(node.properties or {}),
        )
        for node in nodes
    }
    return membership, len(nodes) < MEMBERSHIP_READ_CAP


async def recall_nodes(
    graph_store: Any,
    agent_id: str,
    node_type: str,
    *,
    include_retired: bool = False,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Read projected ledger rows back out of the graph.

    This is the consumer half of the projection, and it exists because a
    derived index nothing reads is not an index — it is a write nobody
    checked. The tools in :mod:`feature` call it instead of reading the YAML
    they already hold, so the graph path is the one actually exercised: an
    index that stops populating fails a query here rather than going unnoticed
    until someone opens the database.

    Every predicate is an exact equality and therefore goes into SQL, not into
    a loop over the result. Post-filtering a ``LIMIT``-ed page under-reports:
    ask for 25 active patterns when the 25 most recent rows all happen to be
    superseded and the honest-looking answer is zero, with a hundred active
    rows sitting just past the page boundary. Pushing the predicates down makes
    the limit mean what the caller asked for, and lets the JSON-path partial
    indexes do the work.

    Raises rather than returning ``[]`` when the graph cannot be queried. An
    empty list is an answer ("no such rows"); a failure is not, and the whole
    ticket exists because a truthful-looking zero was returned for a question
    with a non-empty answer.
    """
    if graph_store is None:
        raise RuntimeError("Graph store not available")

    # ``source`` keeps a hand-written node of the same type out of an answer
    # that claims to describe the ledger; ``agent_id`` is the tenant boundary.
    filters: Dict[str, Any] = {
        "agent_id": agent_id,
        "source": _PROJECTION_SOURCE,
    }
    if not include_retired:
        filters["status"] = ACTIVE_STATUS

    nodes = await graph_store.query_nodes_by_type_and_property(
        node_type,
        filters=filters,
        order_by_created=True,
        limit=max(1, int(limit)),
    )
    return [
        {"node_id": node.node_id, "label": node.label, **(node.properties or {})}
        for node in (nodes or [])
    ]


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
