"""The generic paged read of a registered node type (#4155, rv-4135-1 step 2).

Architect ruling rv-4135-1 (#4135, ruling (c), "Read and write surface")
extends the existing ``recall`` family instead of adding a ``knowledge_*``
one: ``recall`` and ``recall_get`` accept a registered node type, and
``recall_types`` lists the registry. This module is that read. The tools
live on ``SaveFeature``; the five fixed recall tools that predate it are
aliases of it (:data:`FIXED_RECALL_ALIASES`).

What every read does, in order:

1. **Through ``PrivacyEnforcingStorage``.** Any other storage object is
   refused, and a volatile privacy mode (EPHEMERAL, ISOLATED, DEIDENTIFIED)
   denies the read, the same predicate the Save feature uses.
2. **Agent-scoped.** The graph store must be bound to the reading agent, so
   ``graph_node_owners`` scopes every row, and ``agent_id`` is pushed into SQL
   besides.
3. **Paged, with a hard maximum.** Bounded keyset queries
   (``query_nodes_by_type_and_property(keyset=True, after=...)``), never the
   unbounded ``get_nodes_by_type``. A request above :data:`MAX_PAGE_SIZE` is
   clamped. A page withholds rows (below) after the query, so the read runs
   further bounded queries, at most :data:`MAX_SCAN_QUERIES`, until the page
   is full or the type runs out; a page that stopped at that bound says so.
4. **Governed types by canonical status.** A governed type's graph node is
   returned while its canonical assertion is active, and never when that
   assertion is deleted, quarantined or absent. Erasure removes an assertion
   physically and keeps no record of it, so an absent assertion cannot be
   told from an erased one and is never admitted; it is withheld under its
   own reason, ``canonical_absent``. The bounded producer that makes the type
   governed maps the node to its assertion (``graph_node_assertion_ids``); a
   governed type whose producer has no such mapping is withheld.
5. **Source trash state.** A ``message`` node is returned only while the
   conversation row it stands for is live: present and not in Trash. An
   archived row is live (the consolidator's decay archives it).
6. **Home and projection.** Every row names its canonical home and whether
   the graph node is the original or a projection.

Which registered types are served is :data:`_SERVED`, and every other type
is withheld with its reason (:func:`withheld_reason`). The served set is the
types the five fixed tools already exposed, so the generic read exposes no
graph content that was not model-visible before it. Other types stay off
until each is reviewed: some hold operator configuration, possibly API keys
(``feature_config``), some are projections of tables whose trash or erasure
state this read does not consult.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.storage_access import (
    config_hides_persisted_user_content,
)
from kestrel_sovereign.features.strategic_memory.ledger_index import (
    ACTIVE_STATUS,
    BLOCKER_NODE_TYPE,
    PATTERN_NODE_TYPE,
    PROJECTION_SOURCE,
)
from kestrel_sovereign.storage.async_graph_store import GraphNode, NodePagePosition
from kestrel_sovereign.storage.conversation_ids import coerce_persistent_message_id
from kestrel_sovereign.storage.node_type_registry import (
    DECISION_NODE_TYPE,
    STRATEGY_DECISION_NODE_TYPE,
    Governance,
    NodeTypeEntry,
    derive_governance,
    get_node_type,
    is_pre_split_strategy_decision,
    list_node_types,
)
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

#: The largest page one read returns. A larger request is clamped to it. The
#: five fixed tools already allowed limits up to 200.
MAX_PAGE_SIZE = 200

#: The page a read returns when the caller names no limit.
DEFAULT_PAGE_SIZE = 25

#: How many bounded queries one read may run to fill a page whose rows were
#: withheld. The first asks for the page itself, each later one for
#: :data:`MAX_PAGE_SIZE` rows, so one read scans at most about a thousand.
MAX_SCAN_QUERIES = 5

#: The release that removes the five fixed recall tools (rv-4135-1, Tool
#: budget). Today's release line is 0.66; the aliases ship through it.
ALIAS_REMOVAL_RELEASE = "0.67.0"

#: The property a retired graph-original row carries (``mark_superseded``,
#: and the STRATEGY.yaml supersession ``decision_index`` projects).
RETIRED_MARKER = "superseded_by"

#: The edge a message node has to a person concept it mentions.
MENTIONS_EDGE = "mentions"

MESSAGE_NODE_TYPE = "message"
ACTION_ITEM_NODE_TYPE = "action_item"

#: The attribute a governed type's bounded producer module exposes to map its
#: graph nodes to their assertions: ``(tenant_id, nodes) -> {node_id: id}``.
ASSERTION_MAPPER = "graph_node_assertion_ids"

_EMPTY: Mapping[str, str] = MappingProxyType({})

#: Why the read withheld a row its query returned, by the code ``withheld``
#: counts and ``recall_get`` reports as ``missing``.
WITHHELD_REASONS: Mapping[str, str] = MappingProxyType(
    {
        "foreign": (
            "it is another agent's or another node type's, or another writer's "
            "row of the type rather than its producer's"
        ),
        "retired": "it is superseded or resolved; include_retired=True returns it",
        "canonical_inactive": (
            "its canonical assertion is not in an admitted state: retracted or "
            "superseded (admitted with include_retired=True), quarantined or "
            "deleted"
        ),
        "canonical_absent": (
            "the assertion store holds no canonical assertion for it: the "
            "assertion was erased, or none was written, because the privacy "
            "mode it was written in writes no canonical assertions (anonymous) "
            "or its producer has not yet run or failed. Erasure keeps no record, "
            "so an absent assertion is never admitted"
        ),
        "source_trashed": "its source conversation message is in Trash or purged",
    }
)


class RecallRefused(ValueError):
    """The read was refused before any row was returned.

    ``reason`` is a stable code: ``unregistered``, ``withheld``,
    ``not_privacy_wrapped``, ``privacy``, ``unscoped``, ``unavailable``,
    ``bad_cursor``, ``bad_limit`` or ``bad_request``.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ReadPolicy:
    """How the generic read filters one served type."""

    #: Exact property filters every read of the type pushes into SQL.
    filters: Mapping[str, str] = _EMPTY
    #: Exact property filters a current read (not ``include_retired``) adds.
    current_filters: Mapping[str, str] = _EMPTY
    #: Whether each row's source conversation message must still be live.
    requires_live_message: bool = False


_LEDGER_PROJECTION = ReadPolicy(
    filters=MappingProxyType({"source": PROJECTION_SOURCE}),
    current_filters=MappingProxyType({"status": ACTIVE_STATUS}),
)

#: The registered types the generic read serves.
_SERVED: Mapping[str, ReadPolicy] = MappingProxyType(
    {
        ACTION_ITEM_NODE_TYPE: ReadPolicy(),
        DECISION_NODE_TYPE: ReadPolicy(),
        STRATEGY_DECISION_NODE_TYPE: ReadPolicy(),
        PATTERN_NODE_TYPE: _LEDGER_PROJECTION,
        BLOCKER_NODE_TYPE: _LEDGER_PROJECTION,
        MESSAGE_NODE_TYPE: ReadPolicy(requires_live_message=True),
    }
)

_SYSTEM_RECORD = (
    "a system record (identity, custody, lifecycle or release evidence), "
    "not agent memory; no recall tool exposed it before #4155"
)
_TABLE_PROJECTION = (
    "a projection of {location}, whose trash and erasure state the generic "
    "read does not consult, so a row removed there could still be returned"
)
_SAVED_ITEM = (
    "a saved item: the saved_items table holds it and no graph node is "
    "written for it; read it with recall(query, item_type), recall_list or "
    "recall_get(item_id)"
)

#: Why each registered type the generic read does not serve stays off it.
_WITHHELD: Mapping[str, str] = MappingProxyType(
    {
        "fact": (
            "governed, but no graph projection of its assertions exists yet "
            "(#4159), so its producer maps no graph node to its assertion and "
            "canonical status cannot be checked; read facts through save_fact "
            "/ forget_fact's assertion store"
        ),
        "note": _SAVED_ITEM,
        "excerpt": _SAVED_ITEM,
        "stash": _SAVED_ITEM,
        "learned_fact": (
            "legacy rows that take no new writes; recall's keyword fallback "
            "still serves them until the retire verb (#4156) settles them"
        ),
        "concept": (
            "derived from message text, and not filtered when the messages "
            "it came from are trashed"
        ),
        "todo_item": "served by TodoFeature's own tools; not reviewed for the generic read",
        "skill": "served by SkillsFeature's own tools; not reviewed for the generic read",
        "feature_config": (
            "holds operator feature configuration, which can include API keys"
        ),
        "agent": _SYSTEM_RECORD,
        "SovereignAgent": _SYSTEM_RECORD,
        "backup_artifact": _SYSTEM_RECORD,
        "lifecycle_event": _SYSTEM_RECORD,
        "migration_record": _SYSTEM_RECORD,
        "retirement_event": _SYSTEM_RECORD,
        "semantic_benchmark_payload": _SYSTEM_RECORD,
        "sovereignty_receipt": _SYSTEM_RECORD,
    }
)

_NOT_REVIEWED = (
    "not reviewed for the generic read; a type is served only once its trash, "
    "erasure and exposure are checked (#4155)"
)


@dataclass(frozen=True)
class FixedRecallAlias:
    """One fixed recall tool that is now an alias of the generic read."""

    tool: str
    feature: str
    node_types: Tuple[str, ...]
    removal_release: str = ALIAS_REMOVAL_RELEASE

    def notice(self) -> Dict[str, Any]:
        """The ``deprecated_alias`` block each alias result carries."""
        return {
            "alias_of": "recall",
            "node_types": list(self.node_types),
            "removal_release": self.removal_release,
        }


#: The fixed recall tools rv-4135-1 makes aliases, with the release that
#: removes them. ``recall_decisions`` reads both decision types (rv-4244-1).
FIXED_RECALL_ALIASES: Tuple[FixedRecallAlias, ...] = (
    FixedRecallAlias(
        "recall_decisions",
        "MemoryFeature",
        (DECISION_NODE_TYPE, STRATEGY_DECISION_NODE_TYPE),
    ),
    FixedRecallAlias("recall_action_items", "MemoryFeature", (ACTION_ITEM_NODE_TYPE,)),
    FixedRecallAlias("recall_interactions", "MemoryFeature", (MESSAGE_NODE_TYPE,)),
    FixedRecallAlias("recall_patterns", "StrategicMemoryFeature", (PATTERN_NODE_TYPE,)),
    FixedRecallAlias("recall_blockers", "StrategicMemoryFeature", (BLOCKER_NODE_TYPE,)),
)

_ALIASES_BY_TOOL = {alias.tool: alias for alias in FIXED_RECALL_ALIASES}


def fixed_recall_alias(tool: str) -> FixedRecallAlias:
    return _ALIASES_BY_TOOL[tool]


def alias_result(
    alias: FixedRecallAlias,
    page: "RecallPage",
    *,
    confirmation: str,
    data: Mapping[str, Any],
) -> ToolResult:
    """A fixed alias's result: its own rows, plus what the read withheld.

    A page the read could not fill before its scan bound is partial, not ok:
    a short page there is not the end of the type (#4155).
    """
    merged = {**data, **page.withheld_data(), "deprecated_alias": alias.notice()}
    note = page.stopped_short_note()
    if note is None:
        return ToolResult.ok(confirmation=confirmation, data=merged)
    return ToolResult.partial(confirmation=confirmation, error=note, data=merged)


def withheld_reason(entry: NodeTypeEntry, governance: Governance) -> Optional[str]:
    """Why ``entry``'s type is off the generic read, or ``None`` when served."""
    if entry.node_type not in _SERVED:
        reason = _WITHHELD.get(entry.node_type)
        if reason is not None:
            return reason
        if entry.is_projection and entry.canonical_location:
            return _TABLE_PROJECTION.format(location=entry.canonical_location)
        return _NOT_REVIEWED
    if governance.governed and _assertion_mapper(governance) is None:
        return (
            f"governed, but its producer {governance.producer.module} maps no "
            "graph node to its assertion, so canonical status cannot be checked"
        )
    return None


@dataclass(frozen=True)
class RecallType:
    """One registered type as ``recall_types`` lists it."""

    entry: NodeTypeEntry
    governance: Governance
    withheld: Optional[str]

    @property
    def served(self) -> bool:
        return self.withheld is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.entry.to_dict(),
            "governed": self.governance.governed,
            "governed_reason": self.governance.reason,
            "origin": "projection" if self.entry.is_projection else "original",
            "generic_read": self.served,
            "withheld_reason": self.withheld,
        }


def describe_type(entry: NodeTypeEntry) -> RecallType:
    governance = derive_governance(entry.node_type)
    return RecallType(entry, governance, withheld_reason(entry, governance))


async def list_recall_types(registry_db: Any) -> List[RecallType]:
    """Every registered type, with whether the generic read serves it."""
    return [describe_type(entry) for entry in await list_node_types(registry_db)]


def _assertion_mapper(governance: Governance):
    if governance.producer is None:
        return None
    module = importlib.import_module(governance.producer.module)
    mapper = getattr(module, ASSERTION_MAPPER, None)
    return mapper if callable(mapper) else None


@dataclass(frozen=True)
class RecallRow:
    """One node the generic read returned, with the entry that describes it."""

    node: GraphNode
    entry: NodeTypeEntry
    canonical_status: Optional[str] = None

    def home(self) -> Dict[str, Any]:
        return {
            "registered_type": self.entry.node_type,
            "owning_feature": self.entry.owning_feature,
            "canonical_home": self.entry.canonical_home.value,
            "canonical_location": self.entry.canonical_location,
            "is_projection": self.entry.is_projection,
            "origin": "projection" if self.entry.is_projection else "original",
        }

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "node_id": self.node.node_id,
            "node_type": self.node.node_type,
            "label": self.node.label,
            "properties": dict(self.node.properties or {}),
            "home": self.home(),
        }
        if self.canonical_status is not None:
            row["canonical_status"] = self.canonical_status
        return row


@dataclass(frozen=True)
class RecallPage:
    """One page of the generic read."""

    node_types: Tuple[str, ...]
    rows: Tuple[RecallRow, ...]
    next_cursor: Optional[str]
    limit: int
    limit_requested: Any
    include_retired: bool
    #: Rows the bounded queries returned but the read withheld, by a reason
    #: code of :data:`WITHHELD_REASONS`. Only rows before ``next_cursor``.
    withheld: Mapping[str, int] = field(default_factory=dict)

    @property
    def limit_clamped(self) -> bool:
        """Whether the request asked for a page outside ``[1, MAX_PAGE_SIZE]``."""
        return int(self.limit_requested) != self.limit

    @property
    def stopped_short(self) -> bool:
        """Whether the read hit :data:`MAX_SCAN_QUERIES` before filling the page.

        Then rows past ``next_cursor`` were not read, and a short page is not
        the end of the type.
        """
        return self.next_cursor is not None and len(self.rows) < self.limit

    def withheld_data(self) -> Dict[str, Any]:
        """What the read withheld and where it stopped, for every caller's result."""
        return {
            "withheld": dict(self.withheld),
            "withheld_reasons": {
                reason: WITHHELD_REASONS[reason] for reason in sorted(self.withheld)
            },
            "next_cursor": self.next_cursor,
            "stopped_short": self.stopped_short,
        }

    def stopped_short_note(self) -> Optional[str]:
        """Why a short page is not the end, or ``None`` when it is not short."""
        if not self.stopped_short:
            return None
        withheld = sum(self.withheld.values())
        return (
            f"Returned {len(self.rows)} of the {self.limit} asked for: the read "
            f"withheld {withheld} row(s) and stopped after {MAX_SCAN_QUERIES} "
            "bounded queries, so older rows past next_cursor were not read."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_types": list(self.node_types),
            "count": len(self.rows),
            "results": [row.to_dict() for row in self.rows],
            "limit": self.limit,
            "limit_requested": self.limit_requested,
            "limit_clamped": self.limit_clamped,
            "max_page_size": MAX_PAGE_SIZE,
            "include_retired": self.include_retired,
            **self.withheld_data(),
        }


@dataclass(frozen=True)
class RecallLookup:
    """``recall_get``'s answer: the row, or why there is none."""

    row: Optional[RecallRow]
    #: ``not_found`` or a reason code of :data:`WITHHELD_REASONS`.
    missing: Optional[str] = None


@dataclass(frozen=True)
class _Outcome:
    """What the read decided for one node its query returned."""

    node: GraphNode
    row: Optional[RecallRow] = None
    #: A reason code of :data:`WITHHELD_REASONS` when ``row`` is ``None``.
    withheld: Optional[str] = None


def parse_node_types(value: Any) -> Tuple[str, ...]:
    """The node types a ``node_type`` argument names: one, or several by comma."""
    if not isinstance(value, str):
        raise RecallRefused("bad_request", "node_type must be a string")
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise RecallRefused("bad_request", "node_type names no registered type")
    if len(set(names)) != len(names):
        raise RecallRefused("bad_request", "node_type names a type twice")
    return names


def clamp_limit(limit: Any) -> int:
    """``limit`` as an int in ``[1, MAX_PAGE_SIZE]``."""
    if isinstance(limit, bool):
        raise RecallRefused("bad_limit", f"limit must be an integer, got {limit!r}")
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise RecallRefused("bad_limit", f"limit must be an integer, got {limit!r}") from None
    return max(1, min(value, MAX_PAGE_SIZE))


def encode_cursor(node_types: Sequence[str], node: GraphNode) -> str:
    position = NodePagePosition.after(node)
    payload = {
        "v": 1,
        "types": sorted(node_types),
        "c": position.created_at,
        "n": position.node_id,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_cursor(node_types: Sequence[str], cursor: str) -> NodePagePosition:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (UnicodeEncodeError, binascii.Error, ValueError):
        raise RecallRefused("bad_cursor", "cursor is not one this read issued") from None
    if not isinstance(payload, dict) or payload.get("v") != 1:
        raise RecallRefused("bad_cursor", "cursor is not one this read issued")
    if payload.get("types") != sorted(node_types):
        raise RecallRefused(
            "bad_cursor", "cursor was issued for another node_type; start again without it"
        )
    try:
        return NodePagePosition(created_at=payload.get("c"), node_id=payload.get("n"))
    except ValueError:
        raise RecallRefused("bad_cursor", "cursor is not one this read issued") from None


def _require_governed_storage(storage: Any, agent_id: Any) -> Any:
    """The scoped graph view, after the privacy and scope gates."""
    if not isinstance(storage, PrivacyEnforcingStorage):
        raise RecallRefused(
            "not_privacy_wrapped",
            "the generic recall read goes only through PrivacyEnforcingStorage",
        )
    if config_hides_persisted_user_content(storage.privacy_config):
        raise RecallRefused(
            "privacy",
            "recalled graph content is unavailable in the current privacy mode "
            f"({storage.privacy_mode.value})",
        )
    if not isinstance(agent_id, str) or not agent_id:
        raise RecallRefused("unscoped", "the reading agent has no identity to scope to")
    graph = storage.graph
    if graph.agent_id != agent_id:
        raise RecallRefused(
            "unscoped",
            "the graph store is not bound to the reading agent, so its rows "
            "cannot be scoped to it",
        )
    return graph


async def _served_entries(
    registry_db: Any, node_types: Sequence[str]
) -> Dict[str, Tuple[NodeTypeEntry, Governance]]:
    if registry_db is None:
        raise RecallRefused("unregistered", "the node type registry is not available")
    entries: Dict[str, Tuple[NodeTypeEntry, Governance]] = {}
    for node_type in node_types:
        entry = await get_node_type(registry_db, node_type)
        if entry is None:
            raise RecallRefused(
                "unregistered",
                f"{node_type!r} is not a registered node type; recall_types lists them",
            )
        described = describe_type(entry)
        if not described.served:
            raise RecallRefused(
                "withheld",
                f"{node_type!r} is not on the generic read: {described.withheld}",
            )
        entries[node_type] = (entry, described.governance)
    return entries


def _policy_for(node_types: Sequence[str]) -> ReadPolicy:
    first, *rest = (_SERVED[node_type] for node_type in node_types)
    if any(policy != first for policy in rest):
        raise RecallRefused(
            "bad_request",
            "these node types are filtered differently, so they cannot be read "
            "as one page; read each on its own",
        )
    return first


def _sql_filters(
    agent_id: str,
    policy: ReadPolicy,
    include_retired: bool,
    filters: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"agent_id": agent_id, **policy.filters}
    if not include_retired:
        merged.update(policy.current_filters)
    for key, value in (filters or {}).items():
        if key in merged:
            raise RecallRefused("bad_request", f"the read already filters on {key!r}")
        merged[key] = value
    return merged


def _describing_entry(
    node: GraphNode,
    entries: Mapping[str, Tuple[NodeTypeEntry, Governance]],
    strategy_decision: Optional[NodeTypeEntry],
) -> Tuple[NodeTypeEntry, Governance]:
    """The entry whose home a node has.

    A marked ``decision`` row from before the split (rv-4244-1) is a
    ``STRATEGY.yaml`` projection; for one release ``mark_superseded`` treats
    it as one, and so does this read's label.
    """
    entry, governance = entries[node.node_type]
    if strategy_decision is not None and is_pre_split_strategy_decision(
        node.node_type, node.properties
    ):
        return strategy_decision, derive_governance(strategy_decision.node_type)
    return entry, governance


async def _canonical_outcomes(
    storage: PrivacyEnforcingStorage,
    agent_id: str,
    governed: Sequence[Tuple[GraphNode, Governance]],
    include_retired: bool,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """``node_id -> (admitted status, withheld reason)`` of each governed node.

    Exactly one of the pair is set. Every status is asked for, so an assertion
    in a state the read does not admit (``canonical_inactive``) is told apart
    from one the store does not hold (``canonical_absent``). Erasure deletes
    an assertion and keeps no record, so the second can be an erased row and
    is withheld just the same.
    """
    from kestrel_sovereign.knowledge import AssertionQuery, AssertionStatus

    admitted = {AssertionStatus.ACTIVE}
    if include_retired:
        # Retired in its canonical home: a superseded pattern and a resolved
        # blocker retract their assertion. Deleted and quarantined assertions
        # are never admitted.
        admitted |= {AssertionStatus.RETRACTED, AssertionStatus.SUPERSEDED}
    by_producer: Dict[str, List[GraphNode]] = {}
    mappers = {}
    for node, governance in governed:
        mapper = _assertion_mapper(governance)
        if mapper is None:
            continue
        by_producer.setdefault(governance.producer.module, []).append(node)
        mappers[governance.producer.module] = mapper
    statuses: Dict[str, Any] = {}
    mapped: Dict[str, str] = {}
    for module, nodes in by_producer.items():
        mapping = mappers[module](agent_id, nodes)
        mapped.update(mapping)
        assertion_ids = tuple(sorted(set(mapping.values())))
        if not assertion_ids:
            continue
        found = await storage.query_assertions(
            AssertionQuery(
                assertion_ids=assertion_ids,
                statuses=tuple(AssertionStatus),
                limit=len(assertion_ids),
            )
        )
        statuses.update({assertion.assertion_id: assertion.status for assertion in found})
    outcomes: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for node, _ in governed:
        # A node its producer cannot map names no assertion, so the store
        # holds none for it.
        status = statuses.get(mapped.get(node.node_id))
        if status is None:
            outcomes[node.node_id] = (None, "canonical_absent")
        elif AssertionStatus(status) in admitted:
            outcomes[node.node_id] = (AssertionStatus(status).value, None)
        else:
            outcomes[node.node_id] = (None, "canonical_inactive")
    return outcomes


def _message_row_id(node: GraphNode) -> Optional[int]:
    return coerce_persistent_message_id((node.properties or {}).get("message_id"))


async def _live_message_nodes(conversation_store: Any, nodes: Sequence[GraphNode]) -> Set[str]:
    """The ids of the nodes in ``nodes`` whose conversation row is live."""
    row_ids = {node.node_id: _message_row_id(node) for node in nodes}
    wanted = {row_id for row_id in row_ids.values() if row_id is not None}
    live = await conversation_store.live_message_ids(wanted) if wanted else set()
    return {node_id for node_id, row_id in row_ids.items() if row_id in live}


async def _admit(
    storage: PrivacyEnforcingStorage,
    conversation_store: Any,
    agent_id: str,
    nodes: Sequence[GraphNode],
    entries: Mapping[str, Tuple[NodeTypeEntry, Governance]],
    strategy_decision: Optional[NodeTypeEntry],
    policy: ReadPolicy,
    include_retired: bool,
) -> List[_Outcome]:
    """One outcome per node, in the order the query returned them."""
    decided: Dict[str, _Outcome] = {}
    candidates: List[Tuple[GraphNode, NodeTypeEntry, Governance]] = []
    for node in nodes:
        if node.node_type not in entries:
            decided[node.node_id] = _Outcome(node, withheld="foreign")
            continue
        if not include_retired and (node.properties or {}).get(RETIRED_MARKER):
            decided[node.node_id] = _Outcome(node, withheld="retired")
            continue
        entry, governance = _describing_entry(node, entries, strategy_decision)
        candidates.append((node, entry, governance))

    governed = [(node, gov) for node, _, gov in candidates if gov.governed]
    canonical = (
        await _canonical_outcomes(storage, agent_id, governed, include_retired)
        if governed
        else {}
    )
    statuses: Dict[str, Optional[str]] = {}
    remaining: List[Tuple[GraphNode, NodeTypeEntry]] = []
    for node, entry, governance in candidates:
        if governance.governed:
            status, reason = canonical[node.node_id]
            if reason is not None:
                decided[node.node_id] = _Outcome(node, withheld=reason)
                continue
            statuses[node.node_id] = status
        remaining.append((node, entry))

    live = (
        await _live_message_nodes(conversation_store, [node for node, _ in remaining])
        if policy.requires_live_message and remaining
        else None
    )
    for node, entry in remaining:
        if live is not None and node.node_id not in live:
            decided[node.node_id] = _Outcome(node, withheld="source_trashed")
            continue
        decided[node.node_id] = _Outcome(
            node,
            row=RecallRow(node=node, entry=entry, canonical_status=statuses.get(node.node_id)),
        )
    return [decided[node.node_id] for node in nodes]


async def _strategy_decision_entry(
    registry_db: Any, node_types: Sequence[str]
) -> Optional[NodeTypeEntry]:
    if DECISION_NODE_TYPE not in node_types:
        return None
    return await get_node_type(registry_db, STRATEGY_DECISION_NODE_TYPE)


def _require_conversation_store(policy: ReadPolicy, conversation_store: Any) -> None:
    if policy.requires_live_message and conversation_store is None:
        raise RecallRefused(
            "unavailable",
            "the conversation store is not available, so whether each message "
            "is in Trash cannot be checked",
        )


async def read_page(
    *,
    storage: Any,
    registry_db: Any,
    agent_id: str,
    node_types: Sequence[str],
    conversation_store: Any = None,
    limit: Any = DEFAULT_PAGE_SIZE,
    cursor: Optional[str] = None,
    include_retired: bool = False,
    filters: Optional[Mapping[str, Any]] = None,
    created_since: Optional[str] = None,
    linked_to: Optional[Tuple[str, str]] = None,
) -> RecallPage:
    """One page of registered nodes, newest first.

    ``filters``, ``created_since`` and ``linked_to`` narrow the bounded query
    for the fixed aliases' own arguments. ``conversation_store`` answers
    whether a ``message`` node's row is in Trash; it is required for that
    type. Rows withheld after the query do not shorten the page: the read
    queries on, up to :data:`MAX_SCAN_QUERIES` times, and ``withheld`` counts
    every row it passed over. ``next_cursor`` is set unless the type ran out;
    :attr:`RecallPage.stopped_short` says a short page stopped at the bound.
    Raises :class:`RecallRefused` for a read refused before any row.
    """
    if not isinstance(include_retired, bool):
        raise RecallRefused("bad_request", "include_retired must be a boolean")
    node_types = tuple(node_types)
    graph = _require_governed_storage(storage, agent_id)
    entries = await _served_entries(registry_db, node_types)
    policy = _policy_for(node_types)
    _require_conversation_store(policy, conversation_store)
    page_limit = clamp_limit(limit)
    position = decode_cursor(node_types, cursor) if cursor else None
    sql_filters = _sql_filters(agent_id, policy, include_retired, filters)
    strategy_decision = await _strategy_decision_entry(registry_db, node_types)

    rows: List[RecallRow] = []
    withheld: Dict[str, int] = {}
    last: Optional[GraphNode] = None
    exhausted = False
    for query_index in range(MAX_SCAN_QUERIES):
        requested = page_limit if query_index == 0 else MAX_PAGE_SIZE
        nodes = list(
            await graph.query_nodes_by_type_and_property(
                node_types if len(node_types) > 1 else node_types[0],
                filters=sql_filters,
                created_since=created_since,
                order_by_created=True,
                keyset=True,
                limit=requested,
                after=position,
                linked_to=linked_to,
            )
            or []
        )
        outcomes = await _admit(
            storage,
            conversation_store,
            agent_id,
            nodes,
            entries,
            strategy_decision,
            policy,
            include_retired,
        )
        consumed = 0
        for outcome in outcomes:
            if len(rows) == page_limit:
                break
            consumed += 1
            last = outcome.node
            if outcome.row is not None:
                rows.append(outcome.row)
            else:
                withheld[outcome.withheld] = withheld.get(outcome.withheld, 0) + 1
        if len(nodes) < requested and consumed == len(nodes):
            exhausted = True
            break
        if len(rows) == page_limit:
            break
        position = NodePagePosition.after(last)
    return RecallPage(
        node_types=node_types,
        rows=tuple(rows),
        next_cursor=None if exhausted or last is None else encode_cursor(node_types, last),
        limit=page_limit,
        limit_requested=limit,
        include_retired=include_retired,
        withheld=withheld,
    )


async def read_node(
    *,
    storage: Any,
    registry_db: Any,
    agent_id: str,
    node_type: str,
    node_id: str,
    conversation_store: Any = None,
    include_retired: bool = False,
) -> RecallLookup:
    """One registered node by id, through the same gates as :func:`read_page`."""
    if not isinstance(include_retired, bool):
        raise RecallRefused("bad_request", "include_retired must be a boolean")
    if not isinstance(node_id, str) or not node_id:
        raise RecallRefused("bad_request", "node_id must be a non-empty string")
    graph = _require_governed_storage(storage, agent_id)
    entries = await _served_entries(registry_db, (node_type,))
    policy = _SERVED[node_type]
    _require_conversation_store(policy, conversation_store)
    node = await graph.get_node(node_id)
    if node is None or node.node_type != node_type:
        return RecallLookup(row=None, missing="not_found")
    # The same SQL filters a page applies, checked on the one row: a row
    # another agent names, or another writer's row of the type, is foreign; a
    # row a current read leaves out by status is retired.
    properties = node.properties or {}
    scoped = _sql_filters(agent_id, policy, True, None)
    if any(properties.get(key) != value for key, value in scoped.items()):
        return RecallLookup(row=None, missing="foreign")
    current = _sql_filters(agent_id, policy, include_retired, None)
    if any(properties.get(key) != value for key, value in current.items()):
        return RecallLookup(row=None, missing="retired")
    (outcome,) = await _admit(
        storage,
        conversation_store,
        agent_id,
        (node,),
        entries,
        await _strategy_decision_entry(registry_db, (node_type,)),
        policy,
        include_retired,
    )
    return RecallLookup(row=outcome.row, missing=outcome.withheld)
