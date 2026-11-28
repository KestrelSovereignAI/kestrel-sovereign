"""The generic paged read of a registered node type (#4155, rv-4135-1 step 2).

Architect ruling rv-4135-1 (#4135, ruling (c), "Read and write surface")
extends the existing ``recall`` family instead of adding a ``knowledge_*``
one: ``recall`` and ``recall_get`` accept a registered node type, and
``recall_types`` lists the registry. This module is that read. The tools
live on ``SaveFeature``.

What every read does, in order:

1. **Through ``PrivacyEnforcingStorage``.** Any other storage object is
   refused, and a volatile privacy mode (EPHEMERAL, ISOLATED, DEIDENTIFIED)
   denies the read, the same predicate the Save feature uses.
2. **Agent-scoped by the owner table.** The graph store must be bound to the
   reading agent, so ``graph_node_owners`` scopes every row; an unbound store
   is refused. Each served type also declares the scope carrier its rows
   hold (:class:`ScopeCarrier`), and a row whose carrier names another agent
   is withheld as ``foreign``.
3. **Paged, with a hard maximum.** Bounded keyset queries
   (``query_nodes_by_type_and_property(keyset=True, after=...)``), never the
   unbounded ``get_nodes_by_type``. A request above :data:`MAX_PAGE_SIZE` is
   clamped. A page withholds rows (below) after the query, so the read runs
   further bounded queries, at most :data:`MAX_SCAN_QUERIES`, until the page
   is full or the type runs out; a page that stopped at that bound says so.
4. **Governed types by canonical status.** A type that declares canonical
   status required has its graph node returned while its canonical
   assertion is active, and never when that assertion is deleted,
   quarantined or absent. Erasure removes an assertion physically and keeps
   no record of it, so an absent assertion cannot be told from an erased one
   and is never admitted; it is withheld under its own reason,
   ``canonical_absent``. The bounded producer that makes the type governed
   maps the node to its assertion (``graph_node_assertion_ids``).
5. **Source trash state.** Each served type declares its source
   (:class:`SourceCheck`). A ``message`` node is returned only while the
   conversation row it stands for is live; an ``action_item`` or
   ``decision`` only while the message it was extracted from is. Live means
   present and not in Trash. An archived row is live (the consolidator's
   decay archives it).
6. **Home and projection.** Every row names its canonical home and whether
   the graph node is the original or a projection.

A declared check that cannot be made at read time refuses the type with a
named reason; no type is served unchecked.

Which registered types are served is :data:`_SERVED`, and every other type
is withheld with its reason (:func:`withheld_reason`). The served set is the
types the fixed recall tools (``recall_decisions``, ``recall_action_items``,
``recall_interactions``, ``recall_patterns``, ``recall_blockers``) already
exposed, so the generic read exposes no graph content that was not
model-visible before it. Other types stay off until each is reviewed: some
hold operator configuration, possibly API keys (``feature_config``), some
are projections of tables whose trash or erasure state this read does not
consult.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
#: fixed recall tools already allowed limits up to 200.
MAX_PAGE_SIZE = 200

#: The page a read returns when the caller names no limit.
DEFAULT_PAGE_SIZE = 25

#: How many bounded queries one read may run to fill a page whose rows were
#: withheld. The first asks for the page itself, each later one for
#: :data:`MAX_PAGE_SIZE` rows, so one read scans at most about a thousand.
MAX_SCAN_QUERIES = 5

#: The property a retired graph-original row carries (``mark_superseded``,
#: and the STRATEGY.yaml supersession ``decision_index`` projects).
RETIRED_MARKER = "superseded_by"

#: The property an extracted row names its source conversation message by.
SOURCE_MESSAGE_PROPERTY = "source_message_id"

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
        "source_unverifiable": (
            "the source conversation message it names is not a conversation row "
            "id, so whether that message is in Trash cannot be checked"
        ),
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


class ScopeCarrier(str, Enum):
    """What on a served type's row names its agent, besides the owner table."""

    #: ``properties.agent_id``, pushed into SQL and checked on a single row.
    AGENT_ID_PROPERTY = "agent_id_property"
    #: The node id, which begins ``{node_type}:{agent_id}:``.
    NODE_ID_PREFIX = "node_id_prefix"


class SourceCheck(str, Enum):
    """Which conversation row a served type's row stands on, if any."""

    #: None: the row's own canonical home settles whether it is current.
    NONE = "none"
    #: The row is a projection of the conversation row its id names
    #: (``{node_type}:{agent_id}:{row id}``), which must be live.
    OWN_ROW = "own_row"
    #: ``properties.source_message_id``, when set, names a row that must be
    #: live. A row with none was not extracted from a message and is served.
    SOURCE_MESSAGE_ID = "source_message_id"


@dataclass(frozen=True)
class ServedType:
    """How the generic read checks one served type.

    The first three fields have no default on purpose: a type is served only
    once its scope carrier, source and canonical-status requirement are
    declared (#4155).
    """

    scope: ScopeCarrier
    source: SourceCheck
    #: Whether a row is returned only while its canonical assertion is in an
    #: admitted state. A governed type must declare it; a type that declares
    #: it must be governed with a mapping producer, or it is withheld.
    canonical_required: bool
    #: Exact property filters every read of the type pushes into SQL.
    filters: Mapping[str, str] = _EMPTY
    #: Exact property filters a current read (not ``include_retired``) adds.
    current_filters: Mapping[str, str] = _EMPTY

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ScopeCarrier):
            raise TypeError("a served type declares its scope carrier")
        if not isinstance(self.source, SourceCheck):
            raise TypeError("a served type declares its source")
        if not isinstance(self.canonical_required, bool):
            raise TypeError("a served type declares whether canonical status is required")
        if self.source is SourceCheck.OWN_ROW and self.scope is not ScopeCarrier.NODE_ID_PREFIX:
            raise ValueError("a row that stands for its own conversation row is named by its id")

    @property
    def needs_conversation_store(self) -> bool:
        return self.source is not SourceCheck.NONE


_LEDGER_PROJECTION = ServedType(
    scope=ScopeCarrier.AGENT_ID_PROPERTY,
    source=SourceCheck.NONE,
    canonical_required=True,
    filters=MappingProxyType({"source": PROJECTION_SOURCE}),
    current_filters=MappingProxyType({"status": ACTIVE_STATUS}),
)

#: The registered types the generic read serves, each with its declaration.
#: The property filters are the ones the fixed tool reading the type already
#: applies; ``message`` has none, because a message node written before
#: #2649 carries no ``properties.agent_id`` (its id prefix names the agent).
_SERVED: Mapping[str, ServedType] = MappingProxyType(
    {
        ACTION_ITEM_NODE_TYPE: ServedType(
            scope=ScopeCarrier.AGENT_ID_PROPERTY,
            source=SourceCheck.SOURCE_MESSAGE_ID,
            canonical_required=False,
        ),
        DECISION_NODE_TYPE: ServedType(
            scope=ScopeCarrier.AGENT_ID_PROPERTY,
            source=SourceCheck.SOURCE_MESSAGE_ID,
            canonical_required=False,
        ),
        STRATEGY_DECISION_NODE_TYPE: ServedType(
            scope=ScopeCarrier.AGENT_ID_PROPERTY,
            source=SourceCheck.NONE,
            canonical_required=False,
        ),
        PATTERN_NODE_TYPE: _LEDGER_PROJECTION,
        BLOCKER_NODE_TYPE: _LEDGER_PROJECTION,
        MESSAGE_NODE_TYPE: ServedType(
            scope=ScopeCarrier.NODE_ID_PREFIX,
            source=SourceCheck.OWN_ROW,
            canonical_required=False,
        ),
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


def withheld_reason(entry: NodeTypeEntry, governance: Governance) -> Optional[str]:
    """Why ``entry``'s type is off the generic read, or ``None`` when served.

    A served type whose declared canonical-status check cannot be made is
    withheld, with the reason, rather than served unchecked.
    """
    declaration = _SERVED.get(entry.node_type)
    if declaration is None:
        reason = _WITHHELD.get(entry.node_type)
        if reason is not None:
            return reason
        if entry.is_projection and entry.canonical_location:
            return _TABLE_PROJECTION.format(location=entry.canonical_location)
        return _NOT_REVIEWED
    if declaration.canonical_required:
        if not governance.governed:
            return (
                "its canonical status is required, but governance reports it "
                f"ungoverned ({governance.reason}), so that status cannot be checked"
            )
        if _assertion_mapper(governance) is None:
            producer = (
                "names no module" if governance.producer is None
                else governance.producer.module
            )
            return (
                f"governed, but its producer ({producer}) maps no graph node to "
                "its assertion, so canonical status cannot be checked"
            )
    elif governance.governed:
        return (
            "governed, but its generic-read declaration does not require "
            "canonical status, so that status would go unchecked"
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


@dataclass(frozen=True)
class _Candidate:
    """A node past the scope and retired checks, with what describes it."""

    node: GraphNode
    entry: NodeTypeEntry
    governance: Governance
    declaration: ServedType


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


def decode_cursor(node_types: Sequence[str], cursor: Any) -> NodePagePosition:
    if not isinstance(cursor, str):
        raise RecallRefused("bad_cursor", "cursor is not one this read issued")
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (UnicodeEncodeError, binascii.Error, ValueError, RecursionError):
        # RecursionError: a deeply nested JSON body, which this read never issues.
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


def _require_served(entry: NodeTypeEntry) -> Governance:
    described = describe_type(entry)
    if not described.served:
        raise RecallRefused(
            "withheld",
            f"{entry.node_type!r} is not on the generic read: {described.withheld}",
        )
    return described.governance


async def _served_entries(
    registry_db: Any, node_types: Sequence[str]
) -> Dict[str, Tuple[NodeTypeEntry, Governance]]:
    """Each requested type's entry and governance, every one of them served.

    A ``decision`` read also describes its pre-split ``STRATEGY.yaml`` rows by
    the ``strategy_decision`` entry (:func:`_describing_entry`), so that entry
    is checked as served too.
    """
    if registry_db is None:
        raise RecallRefused("unregistered", "the node type registry is not available")
    wanted = list(node_types)
    if DECISION_NODE_TYPE in wanted and STRATEGY_DECISION_NODE_TYPE not in wanted:
        wanted.append(STRATEGY_DECISION_NODE_TYPE)
    entries: Dict[str, Tuple[NodeTypeEntry, Governance]] = {}
    for node_type in wanted:
        entry = await get_node_type(registry_db, node_type)
        if entry is None:
            raise RecallRefused(
                "unregistered",
                f"{node_type!r} is not a registered node type; recall_types lists them",
            )
        entries[node_type] = (entry, _require_served(entry))
    return entries


def _sql_filters(
    agent_id: str, declaration: ServedType, include_retired: bool
) -> Dict[str, Any]:
    """The exact property filters a read of the type pushes into SQL."""
    merged: Dict[str, Any] = {}
    if declaration.scope is ScopeCarrier.AGENT_ID_PROPERTY:
        merged["agent_id"] = agent_id
    merged.update(declaration.filters)
    if not include_retired:
        merged.update(declaration.current_filters)
    return merged


def _common_sql_filters(
    agent_id: str, node_types: Sequence[str], include_retired: bool
) -> Dict[str, Any]:
    """The one SQL filter set a page of several types is queried with."""
    first, *rest = (
        _sql_filters(agent_id, _SERVED[node_type], include_retired) for node_type in node_types
    )
    if any(filters != first for filters in rest):
        raise RecallRefused(
            "bad_request",
            "these node types are filtered differently, so they cannot be read "
            "as one page; read each on its own",
        )
    return first


def _require_mode_can_check_canonical(
    storage: PrivacyEnforcingStorage, node_types: Sequence[str]
) -> None:
    """Refuse a governed read in a mode that writes no canonical assertions."""
    governed = [node_type for node_type in node_types if _SERVED[node_type].canonical_required]
    if governed and storage.privacy_config.requires_anonymization():
        raise RecallRefused(
            "privacy",
            f"{', '.join(governed)} is not read in {storage.privacy_mode.value} mode: "
            "that mode writes no canonical assertions, so the canonical status "
            "of a row written in it cannot be checked",
        )


def _require_conversation_store(
    node_types: Sequence[str], conversation_store: Any, agent_id: str
) -> None:
    """Refuse a read whose source check has no store, or another agent's."""
    sourced = [node_type for node_type in node_types if _SERVED[node_type].needs_conversation_store]
    if not sourced:
        return
    if conversation_store is None:
        raise RecallRefused(
            "unavailable",
            f"the conversation store is not available, so whether the message "
            f"each {', '.join(sourced)} row stands on is in Trash cannot be checked",
        )
    if getattr(conversation_store, "agent_id", None) != agent_id:
        raise RecallRefused(
            "unscoped",
            "the conversation store is not bound to the reading agent, so the "
            "messages its rows stand on cannot be checked",
        )


def _describing_entry(
    node: GraphNode, entries: Mapping[str, Tuple[NodeTypeEntry, Governance]]
) -> Tuple[NodeTypeEntry, Governance]:
    """The entry whose home a node has.

    A marked ``decision`` row from before the split (rv-4244-1) is a
    ``STRATEGY.yaml`` projection; for one release ``mark_superseded`` treats
    it as one, and so does this read.
    """
    if is_pre_split_strategy_decision(node.node_type, node.properties):
        return entries[STRATEGY_DECISION_NODE_TYPE]
    return entries[node.node_type]


def _scope_prefix(node_type: str, agent_id: str) -> str:
    return f"{node_type}:{agent_id}:"


def _carries_scope(node: GraphNode, declaration: ServedType, agent_id: str) -> bool:
    """Whether the node's scope carrier names the reading agent."""
    if declaration.scope is ScopeCarrier.NODE_ID_PREFIX:
        return node.node_id.startswith(_scope_prefix(node.node_type, agent_id))
    return (node.properties or {}).get("agent_id") == agent_id


def _source_row_id(
    node: GraphNode, declaration: ServedType, agent_id: str
) -> Tuple[bool, Optional[int]]:
    """``(stands on a row, its id)``; the id is ``None`` when it cannot be read."""
    if declaration.source is SourceCheck.OWN_ROW:
        suffix = node.node_id[len(_scope_prefix(node.node_type, agent_id)):]
        return True, coerce_persistent_message_id(suffix)
    if declaration.source is SourceCheck.SOURCE_MESSAGE_ID:
        named = (node.properties or {}).get(SOURCE_MESSAGE_PROPERTY)
        if named is None:
            return False, None
        return True, coerce_persistent_message_id(named)
    return False, None


async def _canonical_outcomes(
    storage: PrivacyEnforcingStorage,
    agent_id: str,
    governed: Sequence[_Candidate],
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
    for candidate in governed:
        mapper = _assertion_mapper(candidate.governance)
        if mapper is None:
            continue
        module = candidate.governance.producer.module
        by_producer.setdefault(module, []).append(candidate.node)
        mappers[module] = mapper
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
    for candidate in governed:
        # A node its producer cannot map names no assertion, so the store
        # holds none for it.
        status = statuses.get(mapped.get(candidate.node.node_id))
        if status is None:
            outcomes[candidate.node.node_id] = (None, "canonical_absent")
        elif AssertionStatus(status) in admitted:
            outcomes[candidate.node.node_id] = (AssertionStatus(status).value, None)
        else:
            outcomes[candidate.node.node_id] = (None, "canonical_inactive")
    return outcomes


async def _admit(
    storage: PrivacyEnforcingStorage,
    conversation_store: Any,
    agent_id: str,
    nodes: Sequence[GraphNode],
    entries: Mapping[str, Tuple[NodeTypeEntry, Governance]],
    include_retired: bool,
) -> List[_Outcome]:
    """One outcome per node, in the order the query returned them.

    Each node is checked by the declaration of the entry that describes it:
    scope carrier, retired marker, canonical status, then source.
    """
    decided: Dict[str, _Outcome] = {}
    candidates: List[_Candidate] = []
    for node in nodes:
        if node.node_type not in entries:
            decided[node.node_id] = _Outcome(node, withheld="foreign")
            continue
        entry, governance = _describing_entry(node, entries)
        declaration = _SERVED[entry.node_type]
        if not _carries_scope(node, declaration, agent_id):
            decided[node.node_id] = _Outcome(node, withheld="foreign")
            continue
        if not include_retired and (node.properties or {}).get(RETIRED_MARKER):
            decided[node.node_id] = _Outcome(node, withheld="retired")
            continue
        candidates.append(_Candidate(node, entry, governance, declaration))

    governed = [c for c in candidates if c.declaration.canonical_required]
    canonical = (
        await _canonical_outcomes(storage, agent_id, governed, include_retired)
        if governed
        else {}
    )
    statuses: Dict[str, Optional[str]] = {}
    sourced: Dict[str, int] = {}
    remaining: List[_Candidate] = []
    for candidate in candidates:
        node_id = candidate.node.node_id
        if candidate.declaration.canonical_required:
            status, reason = canonical[node_id]
            if reason is not None:
                decided[node_id] = _Outcome(candidate.node, withheld=reason)
                continue
            statuses[node_id] = status
        stands_on_row, row_id = _source_row_id(candidate.node, candidate.declaration, agent_id)
        if stands_on_row and row_id is None:
            decided[node_id] = _Outcome(candidate.node, withheld="source_unverifiable")
            continue
        if stands_on_row:
            sourced[node_id] = row_id
        remaining.append(candidate)

    live = (
        await conversation_store.live_message_ids(set(sourced.values())) if sourced else set()
    )
    for candidate in remaining:
        node = candidate.node
        if node.node_id in sourced and sourced[node.node_id] not in live:
            decided[node.node_id] = _Outcome(node, withheld="source_trashed")
            continue
        decided[node.node_id] = _Outcome(
            node,
            row=RecallRow(
                node=node, entry=candidate.entry, canonical_status=statuses.get(node.node_id)
            ),
        )
    return [decided[node.node_id] for node in nodes]


async def _gate(
    *,
    storage: Any,
    registry_db: Any,
    agent_id: str,
    node_types: Sequence[str],
    conversation_store: Any,
    include_retired: Any,
) -> Tuple[Any, Dict[str, Tuple[NodeTypeEntry, Governance]]]:
    """Every check a read makes before it queries: ``(graph, entries)``."""
    if not isinstance(include_retired, bool):
        raise RecallRefused("bad_request", "include_retired must be a boolean")
    graph = _require_governed_storage(storage, agent_id)
    entries = await _served_entries(registry_db, node_types)
    _require_mode_can_check_canonical(storage, node_types)
    _require_conversation_store(node_types, conversation_store, agent_id)
    return graph, entries


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
) -> RecallPage:
    """One page of registered nodes, newest first.

    ``conversation_store`` answers whether the message a row stands on is in
    Trash; it is required for a type whose declared source is a message. Rows
    withheld after the query do not shorten the page: the read queries on, up
    to :data:`MAX_SCAN_QUERIES` times, and ``withheld`` counts every row it
    passed over. ``next_cursor`` is set unless the type ran out;
    :attr:`RecallPage.stopped_short` says a short page stopped at the bound.
    Raises :class:`RecallRefused` for a read refused before any row.
    """
    node_types = tuple(node_types)
    graph, entries = await _gate(
        storage=storage,
        registry_db=registry_db,
        agent_id=agent_id,
        node_types=node_types,
        conversation_store=conversation_store,
        include_retired=include_retired,
    )
    sql_filters = _common_sql_filters(agent_id, node_types, include_retired)
    page_limit = clamp_limit(limit)
    position = decode_cursor(node_types, cursor) if cursor else None

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
                order_by_created=True,
                keyset=True,
                limit=requested,
                after=position,
            )
            or []
        )
        outcomes = await _admit(
            storage, conversation_store, agent_id, nodes, entries, include_retired
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
    if not isinstance(node_id, str) or not node_id:
        raise RecallRefused("bad_request", "node_id must be a non-empty string")
    if not isinstance(node_type, str):
        raise RecallRefused("bad_request", "node_type must be a string")
    graph, entries = await _gate(
        storage=storage,
        registry_db=registry_db,
        agent_id=agent_id,
        node_types=(node_type,),
        conversation_store=conversation_store,
        include_retired=include_retired,
    )
    node = await graph.get_node(node_id)
    if node is None or node.node_type != node_type:
        return RecallLookup(row=None, missing="not_found")
    # The SQL filters a page applies, checked on the one row: another
    # writer's row of the type is foreign; a row a current read leaves out by
    # status is retired. The scope carrier itself is checked by _admit, as
    # on a page.
    properties = node.properties or {}
    declaration = _SERVED[node_type]
    scoped = _sql_filters(agent_id, declaration, True)
    if any(properties.get(key) != value for key, value in scoped.items()):
        return RecallLookup(row=None, missing="foreign")
    current = _sql_filters(agent_id, declaration, include_retired)
    if any(properties.get(key) != value for key, value in current.items()):
        return RecallLookup(row=None, missing="retired")
    (outcome,) = await _admit(
        storage, conversation_store, agent_id, (node,), entries, include_retired
    )
    return RecallLookup(row=outcome.row, missing=outcome.withheld)
