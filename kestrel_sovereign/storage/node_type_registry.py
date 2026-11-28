"""The durable registry of graph node types (#4153, step 1 of rv-4135-1).

Architect ruling rv-4135-1 (#4135) gives every ``node_type`` one registry
entry: one owning feature, one canonical home, a schema and a writer, its
retire behaviour, its retention and decay, and whether its graph node is the
original or a projection. This module holds that entry, the table it lives in,
the entries seeded per the ruling's Types (d) table, and the derivation of
``governed``.

Three rules from the ruling shape it:

* **``owning_feature``, never ``owner``.** ``owner`` already means the owning
  *agent*, in ``graph_node_owners``. The registry names a feature.
* **Durable, not rebuilt at feature load.** The table is created and seeded by
  database initialization, and nothing reads the loaded feature set. A type
  whose feature is unloaded stays registered, readable and exportable. Refusing
  its writes is #4135-5 and is not done here: nothing in this module refuses or
  changes a graph write.
* **``governed`` is derived, never stored.** A type is governed when a bounded
  producer module produces it under a vocabulary term pinned in the semantic
  registry (:func:`derive_governance`). The table has no ``governed`` column
  and :class:`NodeTypeEntry` has no ``governed`` field, so no entry can claim
  it.

The census that counts each type's rows and lists its writers is
:mod:`kestrel_sovereign.storage.node_type_census`.

Architect ruling rv-4244-1 (#4244) split ``decision``, which rv-4135-1 had
given one home although two writers write it: the copies strategic memory
projects from ``STRATEGY.yaml`` are ``strategy_decision``, and the decisions
``schema_router`` extracts from conversation, whose graph row is the original,
keep ``decision``. :func:`apply_seed_v2` moves a database seeded before the
split across it.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

TABLE = "node_type_registry"

#: ``schema_backfills`` marker of the one-time seeding of :data:`SEED_ENTRIES`.
#: It registers this release's seeds in a database that has none. A database
#: seeded by an earlier release is brought forward by a later marker's
#: migration instead (:data:`SEED_V2_MARKER`); seeding never overwrites an
#: entry that is already registered.
SEED_MARKER = "node_type_registry_seed_v1"

#: ``schema_backfills`` marker of :func:`apply_seed_v2`, the migration that
#: splits ``decision`` (rv-4244-1 §3).
SEED_V2_MARKER = "node_type_registry_seed_v2"

#: The ``owning_feature`` of a type written by core code rather than by a
#: feature package. Core cannot be unloaded.
CORE_OWNER = "core"

#: A decision ``schema_router`` extracts from a user message. Its graph row is
#: the original (rv-4244-1).
DECISION_NODE_TYPE = "decision"

#: The graph copy of a ``STRATEGY.yaml`` decision, which strategic memory's
#: ``decision_index`` projects. Typed ``decision`` before rv-4244-1.
STRATEGY_DECISION_NODE_TYPE = "strategy_decision"

#: ``properties.source`` on every node ``decision_index`` projects. It is what
#: told a projected decision from an extracted one while both were typed
#: ``decision``.
STRATEGY_DECISION_SOURCE = "strategic_memory"


def strategy_decision_id_prefix(agent_id: str) -> str:
    """The start of the node id of every decision projected for ``agent_id``.

    Matched as a whole, never by splitting an id on ``:``: an agent id is a
    DID, which has colons of its own (au-4244-1 C3).
    """
    return f"decision:{agent_id}:strategy:"


def is_pre_split_strategy_decision(node_type: Optional[str], properties: Any) -> bool:
    """Whether a node is a projected decision still typed ``decision``.

    These are the rows :func:`apply_seed_v2` retypes. One can still appear
    after it ran: from a writer that predates the split, or from a restore of
    a database taken before it.
    """
    return (
        node_type == DECISION_NODE_TYPE
        and isinstance(properties, dict)
        and properties.get("source") == STRATEGY_DECISION_SOURCE
    )


class CanonicalHome(str, Enum):
    """Where a type's canonical record lives (rv-4135-1, Registry)."""

    #: The governed semantic assertion store.
    ASSERTION_STORE = "assertion_store"
    #: A file in the agent's data directory, such as ``STRATEGY.yaml``.
    FILE = "file"
    #: The graph itself: the graph row is the original.
    GRAPH = "graph"
    #: A table of its owner's, core's or a feature's.
    FEATURE_TABLE = "feature_table"


class RetireBehaviour(str, Enum):
    """How a record of this type is retired."""

    #: The graph node is a projection. Retiring routes to the canonical home,
    #: and the graph copy is re-derived from it, never edited (rv-4135-1 (c)).
    CANONICAL_HOME = "canonical_home"
    #: The graph row is the original. Retiring supersedes it in place and never
    #: hard-deletes it; hard deletion stays on the erasure path.
    GRAPH_SUPERSEDE = "graph_supersede"
    #: Only the owning feature's own lifecycle retires it; there is no generic
    #: retire route.
    OWNING_FEATURE = "owning_feature"


class LifecyclePolicy(str, Enum):
    """Where a type's retention or decay policy is defined."""

    #: The canonical home's own policy.
    CANONICAL_HOME = "canonical_home"
    #: The owning feature's code; the registry records no policy of its own.
    OWNING_FEATURE = "owning_feature"
    #: None applies: the record is kept until it is retired or erased.
    NONE = "none"


class NodeTypeRegistrationConflict(ValueError):
    """A node type is already registered with a different entry."""


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"node type registry: {field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class NodeTypeEntry:
    """One registered node type.

    ``owning_feature`` is the feature's class name, the identity ``Feature.name``
    and an agent's ``features`` allowlist use, or :data:`CORE_OWNER`.

    ``writer`` names the owning feature's modules that write the type's
    canonical record. It is empty for a legacy type that takes no new writes.
    A module of another owner that writes it is a foreign writer: the census
    reports it and #4135-5 settles it. For a type whose graph node is a
    projection, the modules that write the graph copy are listed by the
    census, not here.

    ``property_schema`` is ``module:attribute`` of the object that defines the
    record's shape, or ``None`` when no such object is declared.
    """

    node_type: str
    owning_feature: str
    canonical_home: CanonicalHome
    canonical_location: str
    writer: Tuple[str, ...]
    property_schema: Optional[str]
    retire_behaviour: RetireBehaviour
    retention: LifecyclePolicy
    decay: LifecyclePolicy
    is_projection: bool

    def __post_init__(self) -> None:
        _require_text(self.node_type, "node_type")
        _require_text(self.owning_feature, "owning_feature")
        _require_text(self.canonical_location, "canonical_location")
        object.__setattr__(self, "canonical_home", CanonicalHome(self.canonical_home))
        object.__setattr__(self, "retire_behaviour", RetireBehaviour(self.retire_behaviour))
        object.__setattr__(self, "retention", LifecyclePolicy(self.retention))
        object.__setattr__(self, "decay", LifecyclePolicy(self.decay))
        if isinstance(self.writer, str):
            raise TypeError("node type registry: writer is a sequence of module names")
        writer = tuple(_require_text(module, "writer") for module in self.writer)
        if len(set(writer)) != len(writer):
            raise ValueError(f"node type registry: {self.node_type} names a writer twice")
        object.__setattr__(self, "writer", tuple(sorted(writer)))
        if self.property_schema is not None:
            _require_text(self.property_schema, "property_schema")
        if not isinstance(self.is_projection, bool):
            raise TypeError("node type registry: is_projection must be a bool")
        if self.is_projection != (self.canonical_home is not CanonicalHome.GRAPH):
            raise ValueError(
                f"node type registry: {self.node_type}'s graph node is a projection "
                "exactly when its canonical home is not the graph"
            )
        if self.is_projection != (self.retire_behaviour is RetireBehaviour.CANONICAL_HOME):
            raise ValueError(
                f"node type registry: {self.node_type} retires through its canonical "
                "home exactly when its graph node is a projection"
            )

    @property
    def accepts_new_writes(self) -> bool:
        """Whether anything writes this type: ``False`` for a legacy type."""
        return bool(self.writer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type,
            "owning_feature": self.owning_feature,
            "canonical_home": self.canonical_home.value,
            "canonical_location": self.canonical_location,
            "writer": list(self.writer),
            "property_schema": self.property_schema,
            "retire_behaviour": self.retire_behaviour.value,
            "retention": self.retention.value,
            "decay": self.decay.value,
            "is_projection": self.is_projection,
        }


def _values(enum: type[Enum]) -> str:
    return ", ".join(f"'{member.value}'" for member in enum)


#: Kept out of ``CORE_SCHEMA`` for the reason given above
#: ``_MODEL_USAGE_PERIODS_DDL`` in ``async_database``:
#: ``AsyncDatabase._ensure_node_type_registry`` creates it through probe ->
#: migration lock -> re-probe. ``writer`` holds a JSON array of module names.
DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    node_type TEXT PRIMARY KEY,
    owning_feature TEXT NOT NULL,
    canonical_home TEXT NOT NULL CHECK (canonical_home IN ({_values(CanonicalHome)})),
    canonical_location TEXT NOT NULL,
    writer TEXT NOT NULL,
    property_schema TEXT,
    retire_behaviour TEXT NOT NULL CHECK (retire_behaviour IN ({_values(RetireBehaviour)})),
    retention TEXT NOT NULL CHECK (retention IN ({_values(LifecyclePolicy)})),
    decay TEXT NOT NULL CHECK (decay IN ({_values(LifecyclePolicy)})),
    is_projection INTEGER NOT NULL CHECK (is_projection IN (0, 1)),
    registered_at TEXT NOT NULL,
    CHECK (
        (canonical_home = '{CanonicalHome.GRAPH.value}' AND is_projection = 0)
        OR (canonical_home <> '{CanonicalHome.GRAPH.value}' AND is_projection = 1)
    ),
    CHECK (
        (is_projection = 1 AND retire_behaviour = '{RetireBehaviour.CANONICAL_HOME.value}')
        OR (is_projection = 0 AND retire_behaviour <> '{RetireBehaviour.CANONICAL_HOME.value}')
    )
)
"""

_COLUMNS = (
    "node_type",
    "owning_feature",
    "canonical_home",
    "canonical_location",
    "writer",
    "property_schema",
    "retire_behaviour",
    "retention",
    "decay",
    "is_projection",
)


def _graph_original(
    node_type: str, owning_feature: str, writer: Sequence[str]
) -> NodeTypeEntry:
    """A type the census found with the graph as its home (rv-4135-1: "everything else")."""
    return NodeTypeEntry(
        node_type=node_type,
        owning_feature=owning_feature,
        canonical_home=CanonicalHome.GRAPH,
        canonical_location="graph_nodes",
        writer=tuple(writer),
        property_schema=None,
        retire_behaviour=RetireBehaviour.OWNING_FEATURE,
        retention=LifecyclePolicy.OWNING_FEATURE,
        decay=LifecyclePolicy.OWNING_FEATURE,
        is_projection=False,
    )


def _table_projection(
    node_type: str, owning_feature: str, table: str, writer: Sequence[str]
) -> NodeTypeEntry:
    """A type the census found whose graph node indexes a row of its owner's table."""
    return NodeTypeEntry(
        node_type=node_type,
        owning_feature=owning_feature,
        canonical_home=CanonicalHome.FEATURE_TABLE,
        canonical_location=table,
        writer=tuple(writer),
        property_schema=None,
        retire_behaviour=RetireBehaviour.CANONICAL_HOME,
        retention=LifecyclePolicy.CANONICAL_HOME,
        decay=LifecyclePolicy.CANONICAL_HOME,
        is_projection=True,
    )


def _saved_item(node_type: str) -> NodeTypeEntry:
    """``note``, ``excerpt``, ``stash``: saved items, for now (rv-4135-1, Types (d))."""
    return NodeTypeEntry(
        node_type=node_type,
        owning_feature="SaveFeature",
        canonical_home=CanonicalHome.FEATURE_TABLE,
        canonical_location="saved_items",
        writer=("kestrel_sovereign.storage.saved_items_store",),
        property_schema="kestrel_sovereign.storage.saved_items_store:SavedItem",
        retire_behaviour=RetireBehaviour.CANONICAL_HOME,
        retention=LifecyclePolicy.CANONICAL_HOME,
        decay=LifecyclePolicy.CANONICAL_HOME,
        is_projection=True,
    )


def _strategy_ledger_row(node_type: str, section: str) -> NodeTypeEntry:
    return NodeTypeEntry(
        node_type=node_type,
        owning_feature="StrategicMemoryFeature",
        canonical_home=CanonicalHome.FILE,
        canonical_location="STRATEGY_LEDGER.yaml",
        writer=("kestrel_sovereign.features.strategic_memory.ledger",),
        property_schema=f"kestrel_sovereign.features.strategic_memory.ledger_index:{section}",
        retire_behaviour=RetireBehaviour.CANONICAL_HOME,
        retention=LifecyclePolicy.CANONICAL_HOME,
        decay=LifecyclePolicy.CANONICAL_HOME,
        is_projection=True,
    )


#: The entries database initialization registers, per rv-4135-1's Types (d)
#: table. The ruled types come first. The rest are every type the census finds
#: an in-tree writer for, with the owner it identifies; the census test fails
#: when the two disagree. Types the census finds no in-tree owner for
#: (``parametric_self_runs``, ``health_check``, written by external features)
#: are reported as unowned and are not registered here.
SEED_ENTRIES: Tuple[NodeTypeEntry, ...] = (
    # ``save_fact``/``forget_fact`` stay the door; nothing writes a ``fact``
    # graph node yet (its projector is #4159).
    NodeTypeEntry(
        node_type="fact",
        owning_feature="MemoryAgencyFeature",
        canonical_home=CanonicalHome.ASSERTION_STORE,
        canonical_location="semantic_assertions",
        writer=("kestrel_sovereign.features.memory_agency.semantic_facts",),
        property_schema="kestrel_sovereign.features.memory_agency.semantic_facts:FactMapping",
        retire_behaviour=RetireBehaviour.CANONICAL_HOME,
        retention=LifecyclePolicy.CANONICAL_HOME,
        decay=LifecyclePolicy.CANONICAL_HOME,
        is_projection=True,
    ),
    _strategy_ledger_row("strategy_pattern", "PATTERN_SECTION"),
    _strategy_ledger_row("strategy_blocker", "BLOCKER_SECTION"),
    # The row rv-4135-1 gave ``decision``, under the name rv-4244-1 gave the
    # projection. Not governed until rv-4135-1's preconditions are met.
    NodeTypeEntry(
        node_type=STRATEGY_DECISION_NODE_TYPE,
        owning_feature="StrategicMemoryFeature",
        canonical_home=CanonicalHome.FILE,
        canonical_location="STRATEGY.yaml",
        writer=("kestrel_sovereign.features.strategic_memory.feature",),
        property_schema=None,
        retire_behaviour=RetireBehaviour.CANONICAL_HOME,
        retention=LifecyclePolicy.CANONICAL_HOME,
        decay=LifecyclePolicy.CANONICAL_HOME,
        is_projection=True,
    ),
    # Legacy: readable and retirable, no new writes, not migrated to ``fact``.
    # ``memory_agency``'s ``save_fact`` wrote it until #2765.
    NodeTypeEntry(
        node_type="learned_fact",
        owning_feature="MemoryAgencyFeature",
        canonical_home=CanonicalHome.GRAPH,
        canonical_location="graph_nodes",
        writer=(),
        property_schema=None,
        retire_behaviour=RetireBehaviour.GRAPH_SUPERSEDE,
        retention=LifecyclePolicy.NONE,
        decay=LifecyclePolicy.NONE,
        is_projection=False,
    ),
    _saved_item("note"),
    _saved_item("excerpt"),
    _saved_item("stash"),
    # Everything else: application resources, ungoverned.
    # Most of the agent root's writers read it, change it and write it back,
    # so the census lists them as dynamic sites rather than ``agent`` ones.
    _graph_original(
        "agent",
        CORE_OWNER,
        (
            "kestrel_sovereign.agent.constitution",
            "kestrel_sovereign.agent.doctrine_bundle",
            "kestrel_sovereign.identity.birth_record",
            "kestrel_sovereign.inception_service",
            "kestrel_sovereign.kestrel_agent",
            "kestrel_sovereign.knowledge.kite_release_evidence",
            "kestrel_sovereign.setup.constitution_reanchor",
            "kestrel_sovereign.storage.async_file_store",
            "kestrel_sovereign.storage.identity_metadata",
        ),
    ),
    _graph_original("SovereignAgent", CORE_OWNER, ("kestrel_sovereign.kestrel_agent",)),
    _graph_original("action_item", CORE_OWNER, ("kestrel_sovereign.storage.schema_router",)),
    # An extracted claim with a confidence, not a candidate for governance
    # (rv-4244-1). ``mark_superseded`` supersedes it in place. No retention
    # sweep or decay reads it: it is kept until it is erased.
    NodeTypeEntry(
        node_type=DECISION_NODE_TYPE,
        owning_feature=CORE_OWNER,
        canonical_home=CanonicalHome.GRAPH,
        canonical_location="graph_nodes",
        writer=("kestrel_sovereign.storage.schema_router",),
        property_schema=None,
        retire_behaviour=RetireBehaviour.GRAPH_SUPERSEDE,
        retention=LifecyclePolicy.NONE,
        decay=LifecyclePolicy.NONE,
        is_projection=False,
    ),
    _graph_original("backup_artifact", CORE_OWNER, ("kestrel_sovereign.storage.async_storage",)),
    _graph_original("concept", CORE_OWNER, ("kestrel_sovereign.storage.associative_linker",)),
    _graph_original(
        "feature_config",
        CORE_OWNER,
        ("kestrel_sovereign.features.base", "kestrel_sovereign.features.isolated_runtime"),
    ),
    _graph_original("lifecycle_event", CORE_OWNER, ("kestrel_sovereign.graduate_service",)),
    _graph_original(
        "migration_record",
        CORE_OWNER,
        ("kestrel_sovereign.identity.continuity_verifier", "kestrel_sovereign.identity.importer"),
    ),
    _graph_original("retirement_event", CORE_OWNER, ("kestrel_sovereign.retirement_service",)),
    _graph_original(
        "semantic_benchmark_payload",
        CORE_OWNER,
        ("kestrel_sovereign.knowledge.release_evidence_benchmarks",),
    ),
    _graph_original("skill", "SkillsFeature", ("kestrel_sovereign.features.skills.feature",)),
    _graph_original(
        "sovereignty_receipt", "SovereigntyFeature", ("kestrel_sovereign.features.sovereignty.feature",)
    ),
    _graph_original("todo_item", "TodoFeature", ("kestrel_sovereign.features.todo.feature",)),
    _table_projection(
        "audit_anchor",
        "AuditAnchorFeature",
        "audit_anchors",
        ("kestrel_sovereign.features.audit_anchor.feature",),
    ),
    _table_projection(
        "agent_identity_resource",
        CORE_OWNER,
        "agent_identity_resources",
        ("kestrel_sovereign.storage.agent_resource_store",),
    ),
    _table_projection("avatar", CORE_OWNER, "files", ("kestrel_sovereign.storage.async_file_store",)),
    _table_projection(
        "constitution_amendment_artifact",
        CORE_OWNER,
        "files",
        ("kestrel_sovereign.storage.async_file_store",),
    ),
    _table_projection("document", CORE_OWNER, "files", ("kestrel_sovereign.storage.async_file_store",)),
    _table_projection(
        "episode", CORE_OWNER, "memory_episodes", ("kestrel_sovereign.storage.memory_consolidator",)
    ),
    _table_projection(
        "message",
        CORE_OWNER,
        "conversation_history",
        ("kestrel_sovereign.storage.async_conversation_store",),
    ),
)


def _row(entry: NodeTypeEntry) -> Tuple[Any, ...]:
    return (
        entry.node_type,
        entry.owning_feature,
        entry.canonical_home.value,
        entry.canonical_location,
        json.dumps(list(entry.writer)),
        entry.property_schema,
        entry.retire_behaviour.value,
        entry.retention.value,
        entry.decay.value,
        1 if entry.is_projection else 0,
    )


def _entry_from_row(row: Sequence[Any]) -> NodeTypeEntry:
    writer = json.loads(row[4])
    if not isinstance(writer, list):
        raise ValueError(f"node type registry: {row[0]}'s writer is not a JSON array")
    return NodeTypeEntry(
        node_type=row[0],
        owning_feature=row[1],
        canonical_home=CanonicalHome(row[2]),
        canonical_location=row[3],
        writer=tuple(writer),
        property_schema=row[5],
        retire_behaviour=RetireBehaviour(row[6]),
        retention=LifecyclePolicy(row[7]),
        decay=LifecyclePolicy(row[8]),
        is_projection=bool(row[9]),
    )


_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM {TABLE}"
_INSERT_IF_ABSENT = (
    f"INSERT INTO {TABLE} ({', '.join(_COLUMNS)}, registered_at) "
    f"VALUES ({', '.join('?' for _ in _COLUMNS)}, ?) "
    "ON CONFLICT (node_type) DO NOTHING"
)


async def _insert_if_absent(db, entry: NodeTypeEntry) -> None:
    await db.execute(
        _INSERT_IF_ABSENT, _row(entry) + (datetime.now(timezone.utc).isoformat(),)
    )


async def seed_node_type_registry(db, entries: Iterable[NodeTypeEntry] = SEED_ENTRIES) -> None:
    """Register ``entries``, leaving any type that is already registered as it is."""
    for entry in entries:
        await _insert_if_absent(db, entry)


async def get_node_type(db, node_type: str) -> Optional[NodeTypeEntry]:
    """The registered entry for ``node_type``, or ``None`` when it is unregistered."""
    row = await db.fetchone(f"{_SELECT} WHERE node_type = ?", (node_type,))
    return None if row is None else _entry_from_row(row)


async def list_node_types(db) -> List[NodeTypeEntry]:
    """Every registered entry, in code-point order of node type on every backend.

    Sorted here rather than by ``ORDER BY``, whose order follows the database's
    collation: PostgreSQL's locale sorts ``SovereignAgent`` among the ``s``
    types, SQLite's binary collation before every lowercase one.
    """
    rows = await db.fetchall(_SELECT)
    return sorted((_entry_from_row(row) for row in rows), key=lambda entry: entry.node_type)


async def register_node_type(db, entry: NodeTypeEntry) -> NodeTypeEntry:
    """Register ``entry``; a type keeps one entry.

    Registering the entry a type already has is a no-op. Registering a
    different one raises :class:`NodeTypeRegistrationConflict` and changes
    nothing: one type has one owning feature and one canonical home.
    """
    if not isinstance(entry, NodeTypeEntry):
        raise TypeError("register_node_type takes a NodeTypeEntry")
    await _insert_if_absent(db, entry)
    registered = await get_node_type(db, entry.node_type)
    if registered is None:
        raise RuntimeError(
            f"node type registry: {entry.node_type} was not registered"
        )
    if registered != entry:
        raise NodeTypeRegistrationConflict(
            f"node type {entry.node_type!r} is already registered to "
            f"{registered.owning_feature!r} with home "
            f"{registered.canonical_home.value}:{registered.canonical_location}; "
            f"it cannot be registered again as {entry.owning_feature!r}'s"
        )
    return registered


#: ``decision`` exactly as :data:`SEED_MARKER` registered it before rv-4244-1:
#: one home, ``STRATEGY.yaml``, for a type with two writers. The only stored
#: ``decision`` entry :func:`apply_seed_v2` replaces.
SEED_V1_DECISION = NodeTypeEntry(
    node_type=DECISION_NODE_TYPE,
    owning_feature="StrategicMemoryFeature",
    canonical_home=CanonicalHome.FILE,
    canonical_location="STRATEGY.yaml",
    writer=("kestrel_sovereign.features.strategic_memory.feature",),
    property_schema=None,
    retire_behaviour=RetireBehaviour.CANONICAL_HOME,
    retention=LifecyclePolicy.CANONICAL_HOME,
    decay=LifecyclePolicy.CANONICAL_HOME,
    is_projection=True,
)

#: Rows :func:`apply_seed_v2` reads at a time.
_SEED_V2_PAGE = 500


class SeedV2Refused(RuntimeError):
    """The v2 seed migration found a registry it will not change.

    Raised before it writes anything, so the database is left as it was and
    the migration is tried again at the next open.
    """


@dataclass(frozen=True)
class SeedV2Report:
    """What :func:`apply_seed_v2` did (rv-4244-1 §3)."""

    #: Projected decisions retyped from ``decision`` to ``strategy_decision``.
    retyped: int
    #: Rows still typed ``decision``: none carries the projection's marker.
    left: int
    #: Rows among ``left`` whose node id has a projected decision's shape. Each
    #: stays ``decision``: nothing guesses it is a projection, and nothing
    #: deletes it.
    id_shape_mismatches: int
    #: Whether this run replaced the v1 ``decision`` entry. False when an
    #: earlier, interrupted run already had.
    decision_entry_replaced: bool

    def describe(self) -> str:
        return (
            f"retyped {self.retyped} projected decision row(s) to "
            f"{STRATEGY_DECISION_NODE_TYPE}; left {self.left} row(s) typed "
            f"{DECISION_NODE_TYPE}, {self.id_shape_mismatches} of them with a "
            "projected decision's node id and no projection marker; the "
            f"{DECISION_NODE_TYPE} entry was "
            + ("replaced" if self.decision_entry_replaced else "already the v2 entry")
        )


def _seeded(node_type: str) -> NodeTypeEntry:
    return next(entry for entry in SEED_ENTRIES if entry.node_type == node_type)


async def _stored_entry_or_refuse(db, node_type: str) -> Optional[NodeTypeEntry]:
    try:
        return await get_node_type(db, node_type)
    except (ValueError, TypeError) as error:
        raise SeedV2Refused(
            f"{SEED_V2_MARKER}: the stored {node_type!r} entry is not a valid "
            f"registry entry ({error}); nothing was changed"
        ) from error


async def _replace_entry(db, entry: NodeTypeEntry) -> None:
    columns = _COLUMNS[1:]
    assignments = ", ".join(f"{column} = ?" for column in (*columns, "registered_at"))
    await db.execute(
        f"UPDATE {TABLE} SET {assignments} WHERE node_type = ?",
        _row(entry)[1:] + (datetime.now(timezone.utc).isoformat(), entry.node_type),
    )


def _properties_of(raw: Any) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        # Unreadable properties carry no marker, so the row is left as it is.
        return None


async def _owners_of(db, node_ids: Sequence[str]) -> Dict[str, Set[str]]:
    placeholders = ", ".join("?" for _ in node_ids)
    rows = await db.fetchall(
        f"SELECT node_id, agent_id FROM graph_node_owners WHERE node_id IN ({placeholders})",
        tuple(node_ids),
    )
    owners: Dict[str, Set[str]] = {}
    for node_id, agent_id in rows:
        owners.setdefault(node_id, set()).add(agent_id)
    return owners


def _has_projected_id_shape(node_id: str, agent_ids: Iterable[Any]) -> bool:
    return any(
        isinstance(agent_id, str)
        and agent_id
        and node_id.startswith(strategy_decision_id_prefix(agent_id))
        for agent_id in agent_ids
    )


async def apply_seed_v2(db) -> SeedV2Report:
    """Split ``decision`` in a database seeded before rv-4244-1 (§3).

    Registers ``strategy_decision``, retypes every row
    :func:`is_pre_split_strategy_decision` matches to it, and replaces the
    ``decision`` entry with the v2 one. Node ids and edges do not change, so a
    ``supersedes`` edge and a ``superseded_by`` in ``STRATEGY.yaml`` stay
    valid, and no file is touched. A row without the marker stays
    ``decision``.

    The caller runs it in one transaction where the backend has them. Where it
    does not, an interrupted run leaves some prefix of the writes, in the
    order: register, retype, replace. The stored entries this accepts are
    therefore v1 or v2 ``decision``, and no ``strategy_decision`` or the v2
    one: every prefix resumes, and a completed run changes nothing when run
    again (au-4244-1 C2). Any other stored entry stops the migration before
    it writes, with :class:`SeedV2Refused`.
    """
    decision_v2 = _seeded(DECISION_NODE_TYPE)
    strategy_v2 = _seeded(STRATEGY_DECISION_NODE_TYPE)
    stored_decision = await _stored_entry_or_refuse(db, DECISION_NODE_TYPE)
    if stored_decision not in (SEED_V1_DECISION, decision_v2):
        raise SeedV2Refused(
            f"{SEED_V2_MARKER}: the stored {DECISION_NODE_TYPE!r} entry is neither "
            f"the v1 seed nor the v2 one, so it is not replaced and no row is "
            f"retyped. Stored: {stored_decision.to_dict() if stored_decision else None}; "
            f"v1 seed: {SEED_V1_DECISION.to_dict()}; v2 seed: {decision_v2.to_dict()}"
        )
    stored_strategy = await _stored_entry_or_refuse(db, STRATEGY_DECISION_NODE_TYPE)
    if stored_strategy not in (None, strategy_v2):
        raise SeedV2Refused(
            f"{SEED_V2_MARKER}: {STRATEGY_DECISION_NODE_TYPE!r} is already registered "
            f"with another entry, so no row is retyped to it. Stored: "
            f"{stored_strategy.to_dict()}; v2 seed: {strategy_v2.to_dict()}"
        )

    await _insert_if_absent(db, strategy_v2)

    retyped = left = mismatches = 0
    after = ""
    while True:
        page = await db.fetchall(
            "SELECT node_id, properties FROM graph_nodes "
            "WHERE node_type = ? AND node_id > ? ORDER BY node_id LIMIT ?",
            (DECISION_NODE_TYPE, after, _SEED_V2_PAGE),
        )
        if not page:
            break
        after = page[-1][0]
        unmarked: List[Tuple[str, Any]] = []
        for node_id, raw in page:
            properties = _properties_of(raw)
            if not is_pre_split_strategy_decision(DECISION_NODE_TYPE, properties):
                unmarked.append((node_id, properties))
                continue
            retyped += await db.execute(
                "UPDATE graph_nodes SET node_type = ? WHERE node_id = ? AND node_type = ?",
                (STRATEGY_DECISION_NODE_TYPE, node_id, DECISION_NODE_TYPE),
            )
        if not unmarked:
            continue
        left += len(unmarked)
        owners = await _owners_of(db, [node_id for node_id, _ in unmarked])
        for node_id, properties in unmarked:
            claimed = properties.get("agent_id") if isinstance(properties, dict) else None
            if _has_projected_id_shape(node_id, {claimed, *owners.get(node_id, ())}):
                mismatches += 1

    decision_entry_replaced = stored_decision == SEED_V1_DECISION
    if decision_entry_replaced:
        await _replace_entry(db, decision_v2)
    return SeedV2Report(
        retyped=retyped,
        left=left,
        id_shape_mismatches=mismatches,
        decision_entry_replaced=decision_entry_replaced,
    )


@dataclass(frozen=True)
class BoundedProducer:
    """A bounded producer module and the pinned terms it produces a type under."""

    node_type: str
    module: str
    ontology_identifier: str
    ontology_version: str
    terms: Tuple[str, ...]


#: The bounded producer modules core recognizes. A type is governed only when
#: its producer here is importable, binds the same ontology pin, and every term
#: is declared in that pin; :func:`derive_governance` checks all three. A
#: feature cannot add itself: this tuple is core's, and the vocabulary is
#: closed (a new term is a vocabulary release).
BOUNDED_PRODUCERS: Tuple[BoundedProducer, ...] = (
    BoundedProducer(
        node_type="fact",
        module="kestrel_sovereign.features.memory_agency.semantic_facts",
        ontology_identifier="kestrel-vocab",
        ontology_version="1.0.0",
        terms=("kestrel:preferredDeployRegion",),
    ),
    BoundedProducer(
        node_type="strategy_pattern",
        module="kestrel_sovereign.features.strategic_memory.ledger_assertions",
        ontology_identifier="kestrel-vocab",
        ontology_version="1.2.0",
        terms=("kestrel:strategicPattern",),
    ),
    BoundedProducer(
        node_type="strategy_blocker",
        module="kestrel_sovereign.features.strategic_memory.ledger_assertions",
        ontology_identifier="kestrel-vocab",
        ontology_version="1.2.0",
        terms=("kestrel:strategicBlocker",),
    ),
)


@dataclass(frozen=True)
class Governance:
    """Whether a type is governed, and why."""

    node_type: str
    governed: bool
    reason: str
    producer: Optional[BoundedProducer] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "governed": self.governed,
            "reason": self.reason,
            "producer": None if self.producer is None else self.producer.module,
            "terms": [] if self.producer is None else list(self.producer.terms),
        }


def _module_is_missing(error: ModuleNotFoundError, module: str) -> bool:
    """True when ``module`` itself, not something it imports, is absent."""
    return error.name is not None and (module == error.name or module.startswith(error.name + "."))


def derive_governance(
    node_type: str, *, producers: Sequence[BoundedProducer] = BOUNDED_PRODUCERS
) -> Governance:
    """Derive whether ``node_type`` is governed. Nothing stored decides it."""
    from kestrel_sovereign.knowledge.registry import (
        KnowledgeRegistryError,
        get_knowledge_registry,
    )

    declared = [producer for producer in producers if producer.node_type == node_type]
    if not declared:
        return Governance(node_type, False, "no bounded producer module produces this type")
    if len(declared) > 1:
        modules = ", ".join(sorted(producer.module for producer in declared))
        return Governance(node_type, False, f"more than one producer is declared: {modules}")
    producer = declared[0]
    if not producer.terms:
        return Governance(node_type, False, f"{producer.module} declares no vocabulary term")
    try:
        module = importlib.import_module(producer.module)
    except ModuleNotFoundError as error:
        if not _module_is_missing(error, producer.module):
            raise
        return Governance(node_type, False, f"producer module {producer.module} does not exist")
    pin = (producer.ontology_identifier, producer.ontology_version)
    bound = (getattr(module, "ONTOLOGY_IDENTIFIER", None), getattr(module, "ONTOLOGY_VERSION", None))
    if bound != pin:
        return Governance(
            node_type,
            False,
            f"{producer.module} binds ontology {bound[0]}@{bound[1]}, not {pin[0]}@{pin[1]}",
        )
    try:
        resource = get_knowledge_registry().resolve(*pin)
    except KnowledgeRegistryError as error:
        return Governance(node_type, False, f"ontology {pin[0]}@{pin[1]} is not pinned: {error}")
    unpinned = [term for term in producer.terms if term not in resource.selected_terms]
    if unpinned:
        return Governance(
            node_type,
            False,
            f"{', '.join(unpinned)} is not a term of {pin[0]}@{pin[1]}",
        )
    return Governance(
        node_type,
        True,
        f"{producer.module} produces it under {', '.join(producer.terms)} "
        f"pinned in {pin[0]}@{pin[1]}",
        producer,
    )
