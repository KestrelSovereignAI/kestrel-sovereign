"""#4153: the durable node-type registry (rv-4135-1, Registry and Types (d)).

The registry is created and seeded by database initialization, names an
``owning_feature`` (never ``owner``), derives ``governed`` rather than storing
it, and is not rebuilt from the loaded feature set. Nothing here refuses a
graph write; that is #4135-5.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sqlite3

import pytest

from kestrel_sovereign.storage import node_type_registry as registry
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.node_type_registry import (
    BOUNDED_PRODUCERS,
    CORE_OWNER,
    SEED_ENTRIES,
    BoundedProducer,
    CanonicalHome,
    LifecyclePolicy,
    NodeTypeEntry,
    NodeTypeRegistrationConflict,
    RetireBehaviour,
    derive_governance,
    get_node_type,
    list_node_types,
    register_node_type,
)
from tests.shared.agent_store_schema import load, snapshot_path

#: rv-4135-1's Types (d) table, as the ruling states it, with its ``decision``
#: row amended by rv-4244-1: the projection is ``strategy_decision``, and
#: ``decision`` is an original whose home is the graph.
RULED = {
    "fact": (CanonicalHome.ASSERTION_STORE, "semantic_assertions", True),
    "strategy_pattern": (CanonicalHome.FILE, "STRATEGY_LEDGER.yaml", True),
    "strategy_blocker": (CanonicalHome.FILE, "STRATEGY_LEDGER.yaml", True),
    "strategy_decision": (CanonicalHome.FILE, "STRATEGY.yaml", False),
    "decision": (CanonicalHome.GRAPH, "graph_nodes", False),
    "learned_fact": (CanonicalHome.GRAPH, "graph_nodes", False),
    "note": (CanonicalHome.FEATURE_TABLE, "saved_items", False),
    "excerpt": (CanonicalHome.FEATURE_TABLE, "saved_items", False),
    "stash": (CanonicalHome.FEATURE_TABLE, "saved_items", False),
}


def _entry(node_type: str = "probe_note", **changes) -> NodeTypeEntry:
    fields = dict(
        node_type=node_type,
        owning_feature="ProbeFeature",
        canonical_home=CanonicalHome.GRAPH,
        canonical_location="graph_nodes",
        writer=("probe.writer",),
        property_schema=None,
        retire_behaviour=RetireBehaviour.OWNING_FEATURE,
        retention=LifecyclePolicy.OWNING_FEATURE,
        decay=LifecyclePolicy.OWNING_FEATURE,
        is_projection=False,
    )
    fields.update(changes)
    return NodeTypeEntry(**fields)


@pytest.fixture
async def db(tmp_path):
    database = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    try:
        yield database
    finally:
        await database.close()


async def test_registry_persists_across_reload(tmp_path):
    """Entries outlive a restart onto a feature set without their owner."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
    from kestrel_sovereign.testing.native_shutdown_budget import shutdown_fixture_agent
    from tests.shared.booted_agent import booted_agent

    db_path = tmp_path / "kestrel_prime.db"
    async with booted_agent(
        tmp_path, name="RegistryReload", provenance="test-4153", features={"TodoFeature"}
    ) as agent:
        did = agent.did
        assert "TodoFeature" in {type(f).__name__ for f in agent.features.values()}

    probe = _entry("todo_probe", owning_feature="TodoFeature")
    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        await register_node_type(database, probe)
        seeded = await get_node_type(database, "todo_item")
    finally:
        await database.close()
    assert seeded is not None and seeded.owning_feature == "TodoFeature"

    llm_service = LLMService()
    restarted = KestrelAgent(
        did=did, storage_path=str(db_path), llm_service=llm_service, allowed_features=set()
    )
    try:
        await restarted.initialize()
        loaded = {type(feature).__name__ for feature in restarted.features.values()}
        assert loaded == MANDATORY_FEATURES
        assert "TodoFeature" not in loaded
    finally:
        await shutdown_fixture_agent(restarted)
        await llm_service.close()

    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        assert await get_node_type(database, "todo_probe") == probe
        assert await get_node_type(database, "todo_item") == seeded
        assert {entry.node_type for entry in await list_node_types(database)} >= {
            "todo_probe",
            *(entry.node_type for entry in SEED_ENTRIES),
        }
    finally:
        await database.close()


async def test_reopening_never_reseeds_over_a_registered_entry(tmp_path):
    db_path = str(tmp_path / "kestrel_prime.db")
    database = await AsyncDatabase.sqlite(db_path)
    try:
        await database.execute(
            f"UPDATE {registry.TABLE} SET owning_feature = ? WHERE node_type = ?",
            ("ReassignedFeature", "concept"),
        )
    finally:
        await database.close()

    database = await AsyncDatabase.sqlite(db_path)
    try:
        concept = await get_node_type(database, "concept")
        marker = await database.fetchone(
            "SELECT 1 FROM schema_backfills WHERE name = ?", (registry.SEED_MARKER,)
        )
    finally:
        await database.close()
    assert concept is not None and concept.owning_feature == "ReassignedFeature"
    assert marker is not None


async def test_seeding_runs_once_per_database(tmp_path):
    """The seed is a one-time migration, not a rebuild on every open."""
    db_path = str(tmp_path / "kestrel_prime.db")
    database = await AsyncDatabase.sqlite(db_path)
    try:
        await database.execute(f"DELETE FROM {registry.TABLE} WHERE node_type = ?", ("concept",))
    finally:
        await database.close()

    database = await AsyncDatabase.sqlite(db_path)
    try:
        assert await get_node_type(database, "concept") is None
    finally:
        await database.close()


async def test_seeding_leaves_a_registered_entry_as_it_is(db):
    await db.execute(
        f"UPDATE {registry.TABLE} SET owning_feature = ? WHERE node_type = ?",
        ("ReassignedFeature", "concept"),
    )

    await registry.seed_node_type_registry(db)

    concept = await get_node_type(db, "concept")
    assert concept is not None and concept.owning_feature == "ReassignedFeature"


async def test_released_database_gains_the_seeded_registry(tmp_path):
    """The migration: a database born on 0.64.1 has no registry until it opens."""
    db_path = tmp_path / "kestrel_prime.db"
    load(db_path, snapshot_path("0.64.1"))
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (registry.TABLE,)
        ).fetchone() is None

    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        entries = await list_node_types(database)
    finally:
        await database.close()
    assert entries == sorted(SEED_ENTRIES, key=lambda entry: entry.node_type)


async def test_registry_has_owning_feature_not_owner(db):
    columns = {row[1] for row in await db.fetchall(f"PRAGMA table_info({registry.TABLE})")}

    assert "owning_feature" in columns
    assert "owner" not in columns
    assert {
        "node_type",
        "canonical_home",
        "canonical_location",
        "writer",
        "property_schema",
        "retire_behaviour",
        "retention",
        "decay",
        "is_projection",
    } <= columns
    assert "owner" not in {field.name for field in dataclasses.fields(NodeTypeEntry)}
    assert "owning_feature" in {field.name for field in dataclasses.fields(NodeTypeEntry)}


async def test_governed_is_derived(db):
    # Pinned term + bounded producer module: governed.
    for node_type in ("fact", "strategy_pattern", "strategy_blocker"):
        assert derive_governance(node_type).governed, node_type

    # An entry that tries to claim it gets nowhere: no field, no column.
    with pytest.raises(TypeError):
        _entry("claims_governance", governed=True)
    columns = {row[1] for row in await db.fetchall(f"PRAGMA table_info({registry.TABLE})")}
    assert "governed" not in columns
    assert not hasattr(_entry(), "governed")
    claimant = _entry(
        "claims_governance",
        canonical_home=CanonicalHome.ASSERTION_STORE,
        canonical_location="semantic_assertions",
        retire_behaviour=RetireBehaviour.CANONICAL_HOME,
        is_projection=True,
    )
    await register_node_type(db, claimant)
    governance = derive_governance("claims_governance")
    assert not governance.governed
    assert governance.reason == "no bounded producer module produces this type"

    # A producer module with a pinned term governs the type it produces.
    ledger = "kestrel_sovereign.features.strategic_memory.ledger_assertions"
    pinned = BoundedProducer("probe", ledger, "kestrel-vocab", "1.2.0", ("kestrel:strategicPattern",))
    assert derive_governance("probe", producers=(pinned,)).governed

    # Without a pinned term, a missing module, or a mismatched pin, it does not.
    unpinned = dataclasses.replace(pinned, terms=("kestrel:strategicDecision",))
    missing = dataclasses.replace(pinned, module="kestrel_sovereign.features.no_such_producer")
    mismatched = dataclasses.replace(pinned, ontology_version="1.0.0", terms=("kestrel:preferredDeployRegion",))
    termless = dataclasses.replace(pinned, terms=())
    for producer, reason in (
        (unpinned, "kestrel:strategicDecision is not a term of kestrel-vocab@1.2.0"),
        (missing, "producer module kestrel_sovereign.features.no_such_producer does not exist"),
        (mismatched, f"{ledger} binds ontology kestrel-vocab@1.2.0, not kestrel-vocab@1.0.0"),
        (termless, f"{ledger} declares no vocabulary term"),
    ):
        governance = derive_governance("probe", producers=(producer,))
        assert not governance.governed
        assert governance.reason == reason


def test_governance_import_failure_inside_producer_is_not_absence(monkeypatch):
    real = importlib.import_module

    def broken(name, *args):
        if name == "probe_producer":
            raise ModuleNotFoundError("No module named 'dependency'", name="dependency")
        return real(name, *args)

    monkeypatch.setattr(registry.importlib, "import_module", broken)
    producer = BoundedProducer("probe", "probe_producer", "kestrel-vocab", "1.0.0", ("kestrel:x",))
    with pytest.raises(ModuleNotFoundError):
        derive_governance("probe", producers=(producer,))


def test_bounded_producers_match_the_producer_modules():
    """The declared terms are the ones each producer module actually maps."""
    from kestrel_sovereign.features.memory_agency import semantic_facts
    from kestrel_sovereign.features.strategic_memory import ledger_assertions, ledger_index

    declared = {producer.node_type: producer.terms for producer in BOUNDED_PRODUCERS}
    assert declared["fact"] == tuple(
        f"kestrel:{term}" for term in semantic_facts._SUPPORTED_PREDICATES.values()
    )
    assert declared[ledger_index.PATTERN_NODE_TYPE] == (f"kestrel:{ledger_assertions._PATTERN_TERM}",)
    assert declared[ledger_index.BLOCKER_NODE_TYPE] == (f"kestrel:{ledger_assertions._BLOCKER_TERM}",)


async def test_seed_types_match_ruling(db):
    entries = {entry.node_type: entry for entry in await list_node_types(db)}

    for node_type, (home, location, governed) in RULED.items():
        entry = entries[node_type]
        assert entry.canonical_home is home, node_type
        assert entry.canonical_location == location, node_type
        assert entry.is_projection is (home is not CanonicalHome.GRAPH), node_type
        assert derive_governance(node_type).governed is governed, node_type

    learned = entries["learned_fact"]
    assert learned.writer == ()
    assert not learned.accepts_new_writes
    assert learned.retire_behaviour is RetireBehaviour.GRAPH_SUPERSEDE
    assert all(
        entries[t].accepts_new_writes for t in RULED if t != "learned_fact"
    )
    assert entries["fact"].owning_feature == "MemoryAgencyFeature"
    assert entries["fact"].writer == ("kestrel_sovereign.features.memory_agency.semantic_facts",)
    for node_type in ("strategy_pattern", "strategy_blocker", "strategy_decision"):
        assert entries[node_type].owning_feature == "StrategicMemoryFeature"
    decision = entries["decision"]
    assert decision.owning_feature == CORE_OWNER
    assert decision.writer == ("kestrel_sovereign.storage.schema_router",)
    assert decision.retire_behaviour is RetireBehaviour.GRAPH_SUPERSEDE
    # Nothing sweeps or decays an extracted decision (rv-4244-1 §2).
    assert (decision.retention, decision.decay) == (LifecyclePolicy.NONE, LifecyclePolicy.NONE)
    for node_type in ("note", "excerpt", "stash"):
        assert entries[node_type].owning_feature == "SaveFeature"
    for entry in entries.values():
        if entry.node_type not in RULED:
            assert entry.canonical_home in (CanonicalHome.GRAPH, CanonicalHome.FEATURE_TABLE)
            assert not derive_governance(entry.node_type).governed


def test_seed_locations_match_their_owners_constants():
    from kestrel_sovereign.features.strategic_memory.feature import StrategicMemoryFeature
    from kestrel_sovereign.features.strategic_memory.ledger import LEDGER_FILENAME

    entries = {entry.node_type: entry for entry in SEED_ENTRIES}
    assert entries["strategy_pattern"].canonical_location == LEDGER_FILENAME
    assert entries["strategy_decision"].canonical_location == StrategicMemoryFeature.STRATEGY_FILENAME


def test_seed_references_resolve():
    """Every writer is a module and every property schema names a real object."""
    for entry in SEED_ENTRIES:
        for module in entry.writer:
            assert importlib.util.find_spec(module) is not None, (entry.node_type, module)
        if entry.property_schema is not None:
            module, _, attribute = entry.property_schema.partition(":")
            assert hasattr(importlib.import_module(module), attribute), entry.property_schema
    assert len({entry.node_type for entry in SEED_ENTRIES}) == len(SEED_ENTRIES)


async def test_register_is_idempotent_and_refuses_a_second_owner(db):
    entry = _entry()
    assert await register_node_type(db, entry) == entry
    assert await register_node_type(db, entry) == entry

    with pytest.raises(NodeTypeRegistrationConflict, match="already registered to 'ProbeFeature'"):
        await register_node_type(db, _entry(owning_feature="OtherFeature"))
    assert await get_node_type(db, "probe_note") == entry


@pytest.mark.parametrize(
    "changes",
    (
        {"canonical_home": CanonicalHome.FILE},
        {"is_projection": True},
        {"retire_behaviour": RetireBehaviour.CANONICAL_HOME},
        {"owning_feature": ""},
        {"writer": "probe.writer"},
        {"writer": ("probe.writer", "probe.writer")},
    ),
)
def test_entry_rejects_an_incoherent_entry(changes):
    with pytest.raises((ValueError, TypeError)):
        _entry(**changes)


async def test_table_rejects_an_incoherent_row(db):
    """The table's own checks hold for a writer that bypasses the dataclass."""
    with pytest.raises(Exception) as excinfo:
        await db.execute(
            f"INSERT INTO {registry.TABLE} (node_type, owning_feature, canonical_home, "
            "canonical_location, writer, property_schema, retire_behaviour, retention, "
            "decay, is_projection, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("bad", CORE_OWNER, "graph", "graph_nodes", "[]", None,
             "owning_feature", "none", "none", 1, "2026-11-27T00:00:00+00:00"),
        )
    assert "CHECK" in str(excinfo.value).upper()


async def _no_schema(db):
    """Schema initializer for the admin connection: run no DDL."""


# Parametrized with "postgres" so the #3381 guard fails this case, rather than
# letting it skip, in a job that provides PostgreSQL.
@pytest.fixture(params=["postgres"])
def postgres_url(request):
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    return url


async def _registry_types_in(admin, schema: str):
    """The node types in ``schema``'s own registry, or ``None`` when it has none."""
    from tests.utils.postgres_schema import quoted_search_path

    table = f"{quoted_search_path(schema)}.{registry.TABLE}"
    if await admin.fetchval("SELECT to_regclass(?)::text", (table,)) is None:
        return None
    return sorted(row[0] for row in await admin.fetchall(f"SELECT node_type FROM {table}"))


async def test_registry_and_census_on_postgres(postgres_url):
    """The same table, seed and census on PostgreSQL, in a schema of its own.

    Another schema on the search path already holds a seeded registry, as
    ``public`` does once a boot has run there (#4153 review P2-b). The boot
    creates and seeds its own registry and reads and writes that one; the
    other schema's is left as it was. No other table of a booted database is
    on the path, so none can resolve outside these two schemas (#4239).

    pgvector's schema is off the path too: the case needs no vector, and the
    boot's ``embedding_vec`` migrations fail without one, non-fatally. The
    extension is installed first, so the boot's ``CREATE EXTENSION IF NOT
    EXISTS vector`` never puts it in a schema this case then removes.
    """
    from kestrel_sovereign.storage.node_type_census import run_census
    from tests.utils.postgres_schema import (
        disposable_postgres_schema,
        pgvector_schema,
        quoted_search_path,
        with_search_path,
    )

    sentinel = _entry("other_schema_sentinel")
    admin = await AsyncDatabase.postgres(postgres_url, schema_initializer=_no_schema)
    try:
        await pgvector_schema(admin)
        async with disposable_postgres_schema(admin, "node_type_registry_other") as other:
            async with disposable_postgres_schema(admin, "node_type_registry") as schema:
                seeded = await AsyncDatabase.postgres(
                    with_search_path(postgres_url, quoted_search_path(other)),
                    schema_initializer=_no_schema,
                )
                try:
                    await seeded.execute(registry.DDL)
                    await registry.seed_node_type_registry(seeded, (*SEED_ENTRIES, sentinel))
                finally:
                    await seeded.close()
                other_before = await _registry_types_in(admin, other)

                url = with_search_path(postgres_url, quoted_search_path(schema, other))
                for _boot in range(2):
                    database = await AsyncDatabase.postgres(url)
                    await database.close()
                database = await AsyncDatabase.postgres(url, schema_initializer=_no_schema)
                try:
                    assert await list_node_types(database) == sorted(
                        SEED_ENTRIES, key=lambda entry: entry.node_type
                    )
                    probe = _entry()
                    assert await register_node_type(database, probe) == probe
                    with pytest.raises(NodeTypeRegistrationConflict):
                        await register_node_type(database, _entry(owning_feature="OtherFeature"))
                    store = AsyncGraphStore(database, agent_id="did:example:pg")
                    await store.add_node(
                        GraphNode(node_id="pg:1", node_type="health_check", label="h", properties={})
                    )
                    report = await run_census(database, write_sites=())
                finally:
                    await database.close()
                own = await _registry_types_in(admin, schema)
                other_after = await _registry_types_in(admin, other)
    finally:
        await admin.close()
    seeds = sorted(entry.node_type for entry in SEED_ENTRIES)
    assert own == sorted([*seeds, probe.node_type])
    assert other_after == other_before == sorted([*seeds, sentinel.node_type])
    health = report.row("health_check")
    assert (health.total_rows, dict(health.rows_by_agent)) == (1, {"did:example:pg": 1})
    assert health.owning_feature is None
    assert report.row("probe_note").registered == probe
    assert sentinel.node_type not in {row.node_type for row in report.types}


async def test_unregistered_types_still_write(db):
    """Nothing is refused: the writers SDK#140 names keep working."""
    store = AsyncGraphStore(db, agent_id="did:example:writer")
    for node_type in ("feature_config", "parametric_self_runs", "health_check"):
        await store.add_node(
            GraphNode(node_id=f"{node_type}:1", node_type=node_type, label=node_type, properties={})
        )
        assert (await store.get_node(f"{node_type}:1")).node_type == node_type
    assert await get_node_type(db, "parametric_self_runs") is None
    assert await get_node_type(db, "health_check") is None
