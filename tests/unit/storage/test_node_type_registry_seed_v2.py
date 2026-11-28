"""#4248: the v2 seed migration that splits ``decision`` (rv-4244-1 §3).

A database seeded before the split holds ``decision`` rows of two kinds: the
copies strategic memory projected from STRATEGY.yaml, marked
``source = 'strategic_memory'``, and the decisions extracted from
conversation, unmarked. The migration retypes the marked ones to
``strategy_decision`` and leaves everything else as it was.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from kestrel_sovereign.features.strategic_memory.decision_index import (
    strategy_decision_node_id,
)
from kestrel_sovereign.storage import node_type_registry as registry
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.node_type_registry import (
    DECISION_NODE_TYPE,
    SEED_ENTRIES,
    SEED_V1_DECISION,
    SEED_V2_MARKER,
    STRATEGY_DECISION_NODE_TYPE,
    SeedV2Refused,
    apply_seed_v2,
    get_node_type,
    register_node_type,
)

#: Agent ids are DIDs, with colons of their own (au-4244-1 C3).
ALICE = "did:pkh:eip155:1:0xA11CE"
BOB = "did:test:bob"
#: A DID with a ``strategy`` segment of its own: its conversation decisions
#: contain ``:strategy:`` without being projections.
CAROL = "did:web:strategy:example"

SEEDS = {entry.node_type: entry for entry in SEED_ENTRIES}


def _entry(decision: str, date: str = "2026-07-01") -> dict:
    return {"date": date, "decision": decision}


def _projected(agent: str, decision: str, date: str = "2026-07-01") -> GraphNode:
    """A STRATEGY.yaml decision as the projection wrote it before the split."""
    entry = _entry(decision, date)
    return GraphNode(
        node_id=strategy_decision_node_id(agent, entry),
        node_type=DECISION_NODE_TYPE,
        label=decision,
        properties={
            "agent_id": agent,
            "text": decision,
            "claim_source": "strategy_yaml",
            "source": "strategic_memory",
            "created_at": f"{date}T00:00:00+00:00",
        },
    )


def _extracted(node_id: str, agent: str, text: str, created_at: str) -> GraphNode:
    """A decision ``schema_router`` extracted from a message: no marker."""
    return GraphNode(
        node_id=node_id,
        node_type=DECISION_NODE_TYPE,
        label=text,
        properties={
            "agent_id": agent,
            "text": text,
            "source_message_id": "17",
            "confidence": 0.8,
            "created_at": created_at,
        },
    )


ALICE_PROJECTED = _projected(ALICE, "adopt durable leases")
BOB_PROJECTED = _projected(BOB, "stop using cron", "2026-07-02")
ALICE_EXTRACTED = _extracted(f"decision:{ALICE}:0123456789abcdef", ALICE, "move to brooklyn", "2026-07-03T10:00:00")
#: The strategy id shape with no marker: stays ``decision`` and is counted.
ALICE_MISMATCH = _extracted(f"decision:{ALICE}:strategy:deadbeefdeadbeef", ALICE, "unmarked", "2026-07-04T10:00:00")
#: ``:strategy:`` inside the id, but not after this agent's prefix: not a mismatch.
CAROL_EXTRACTED = _extracted(f"decision:{CAROL}:fedcba9876543210", CAROL, "keep the domain", "2026-07-05T10:00:00")
ALICE_FOREIGN_SHAPE = _extracted(
    "decision:did:test:other:strategy:0000000000000000", ALICE, "foreign shape", "2026-07-06T10:00:00"
)
ACTION = GraphNode(
    node_id=f"action:{ALICE}:1", node_type="action_item", label="call mom",
    properties={"agent_id": ALICE, "text": "call mom", "status": "pending"},
)

PROJECTED = (ALICE_PROJECTED, BOB_PROJECTED)
UNMARKED = (ALICE_EXTRACTED, ALICE_MISMATCH, CAROL_EXTRACTED, ALICE_FOREIGN_SHAPE)


async def rewind_to_v1(database: AsyncDatabase) -> None:
    """Put a database's registry back as the v1 seed left it, before #4248."""
    await database.execute(
        f"DELETE FROM {registry.TABLE} WHERE node_type IN (?, ?)",
        (DECISION_NODE_TYPE, STRATEGY_DECISION_NODE_TYPE),
    )
    await register_node_type(database, SEED_V1_DECISION)
    await database.execute("DELETE FROM schema_backfills WHERE name = ?", (SEED_V2_MARKER,))


async def test_an_unreadable_decision_entry_is_refused_with_its_cause(db_path):
    """The refusal keeps the parse error as its cause, and the chain ends."""
    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        await database.execute(
            f"UPDATE {registry.TABLE} SET writer = ? WHERE node_type = ?",
            ('{"not": "an array"}', DECISION_NODE_TYPE),
        )
        await database.execute("DELETE FROM schema_backfills WHERE name = ?", (SEED_V2_MARKER,))
    finally:
        await database.close()

    with pytest.raises(SeedV2Refused, match="not a valid registry entry") as raised:
        await AsyncDatabase.sqlite(str(db_path))

    assert isinstance(raised.value.__cause__, ValueError)
    seen, error = set(), raised.value
    while error is not None:
        assert id(error) not in seen, "the cause chain is a cycle"
        seen.add(id(error))
        error = error.__cause__


async def _undo_migration(database: AsyncDatabase) -> None:
    """Put an opened (so migrated) pre-split database back as it was."""
    await rewind_to_v1(database)
    await database.execute(
        "UPDATE graph_nodes SET node_type = ? WHERE node_type = ?",
        (DECISION_NODE_TYPE, STRATEGY_DECISION_NODE_TYPE),
    )


async def write_pre_split_graph(database: AsyncDatabase) -> None:
    """Both kinds of decision, a mismatch, and edges across the two kinds."""
    for node in (*PROJECTED, *UNMARKED, ACTION):
        store = AsyncGraphStore(database, agent_id=node.properties["agent_id"])
        await store.add_node(node)
    alice = AsyncGraphStore(database, agent_id=ALICE)
    # An extracted decision superseding a projected one, and the reverse.
    await alice.add_edge(ALICE_EXTRACTED.node_id, ALICE_PROJECTED.node_id, "supersedes", {"reason": "r"})
    await alice.add_edge(ALICE_PROJECTED.node_id, ALICE_MISMATCH.node_id, "supersedes", {"reason": "s"})


async def pre_split_database(path: Path) -> None:
    database = await AsyncDatabase.sqlite(str(path))
    try:
        await rewind_to_v1(database)
        await write_pre_split_graph(database)
    finally:
        await database.close()


def _snapshot(path: Path) -> dict:
    """Every row of the tables the migration could touch."""
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        return {
            table: sorted(connection.execute(f'SELECT * FROM "{table}"').fetchall(), key=repr)
            for table in (
                "graph_nodes", "graph_edges", "graph_node_owners",
                "graph_edge_owners", registry.TABLE, "schema_backfills",
            )
        }
    finally:
        connection.close()


def _types(snapshot: dict) -> dict:
    return {row[0]: row[1] for row in snapshot["graph_nodes"]}


def _without_type(snapshot: dict, table: str) -> list:
    """``graph_nodes`` rows with the type column left out."""
    assert table == "graph_nodes"
    return sorted(((row[0],) + tuple(row[2:]) for row in snapshot[table]), key=repr)


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "kestrel_prime.db"


async def test_migration_retypes_projections_and_leaves_the_rest(db_path):
    await pre_split_database(db_path)
    before = _snapshot(db_path)

    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        assert await get_node_type(database, DECISION_NODE_TYPE) == SEEDS[DECISION_NODE_TYPE]
        assert (
            await get_node_type(database, STRATEGY_DECISION_NODE_TYPE)
            == SEEDS[STRATEGY_DECISION_NODE_TYPE]
        )
    finally:
        await database.close()
    after = _snapshot(db_path)

    types = _types(after)
    for node in PROJECTED:
        assert types[node.node_id] == STRATEGY_DECISION_NODE_TYPE, node.node_id
    for node in UNMARKED:
        assert types[node.node_id] == DECISION_NODE_TYPE, node.node_id
    assert types[ACTION.node_id] == "action_item"
    # Nothing but the type changed: ids, labels, properties, owners, edges.
    assert _without_type(after, "graph_nodes") == _without_type(before, "graph_nodes")
    for table in ("graph_edges", "graph_node_owners", "graph_edge_owners"):
        assert after[table] == before[table], table
    assert (SEED_V2_MARKER,) in {(row[0],) for row in after["schema_backfills"]}


async def test_report_counts_retyped_left_and_mismatched(db_path):
    """The three counts; a mismatch is matched by prefix, not by splitting on ':'."""
    await pre_split_database(db_path)
    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        await _undo_migration(database)
        report = await apply_seed_v2(database)
    finally:
        await database.close()

    assert (report.retyped, report.left, report.id_shape_mismatches) == (2, len(UNMARKED), 1)
    assert report.decision_entry_replaced
    described = report.describe()
    assert "retyped 2" in described and f"left {len(UNMARKED)}" in described


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records: list = []

    def emit(self, record):
        self.records.append(record)


async def test_open_reports_the_counts(db_path):
    await pre_split_database(db_path)
    # On the module's own logger: opening a database can reconfigure the root
    # logging handlers, which drops caplog's.
    database_logger = logging.getLogger("kestrel_sovereign.storage.async_database")
    handler = _Records()
    level = database_logger.level
    database_logger.addHandler(handler)
    database_logger.setLevel(logging.INFO)
    try:
        database = await AsyncDatabase.sqlite(str(db_path))
        await database.close()
    finally:
        database_logger.removeHandler(handler)
        database_logger.setLevel(level)

    lines = [r for r in handler.records if r.getMessage().startswith(SEED_V2_MARKER)]
    assert len(lines) == 1
    # A mismatch row is left for an operator to look at.
    assert lines[0].levelno == logging.WARNING
    assert "1 of them with a projected decision's node id" in lines[0].getMessage()


async def test_a_second_run_changes_nothing(db_path):
    await pre_split_database(db_path)
    database = await AsyncDatabase.sqlite(str(db_path))
    await database.close()
    migrated = _snapshot(db_path)

    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        # Run again by hand, past the marker: still nothing to do.
        report = await apply_seed_v2(database)
    finally:
        await database.close()

    assert (report.retyped, report.decision_entry_replaced) == (0, False)
    assert (report.left, report.id_shape_mismatches) == (len(UNMARKED), 1)
    assert _snapshot(db_path) == migrated


async def test_a_fresh_database_needs_no_retype(db_path):
    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        entries = {entry.node_type: entry for entry in await registry.list_node_types(database)}
        assert entries == SEEDS
        marker = await database.fetchone(
            "SELECT 1 FROM schema_backfills WHERE name = ?", (SEED_V2_MARKER,)
        )
        assert marker is not None
    finally:
        await database.close()


async def test_a_changed_decision_entry_stops_the_migration(db_path):
    """Neither v1 nor v2: nothing is replaced, retyped or registered."""
    await pre_split_database(db_path)
    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        await _undo_migration(database)
        await database.execute(
            f"UPDATE {registry.TABLE} SET owning_feature = ? WHERE node_type = ?",
            ("ReassignedFeature", DECISION_NODE_TYPE),
        )
    finally:
        await database.close()
    before = _snapshot(db_path)

    # By name, not wrapped in the migration transaction's TransactionError.
    with pytest.raises(SeedV2Refused, match="neither the v1 seed nor the v2 one"):
        await AsyncDatabase.sqlite(str(db_path))

    assert _snapshot(db_path) == before


async def test_a_conflicting_strategy_decision_entry_stops_the_migration(db_path):
    await pre_split_database(db_path)
    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        await _undo_migration(database)
        other = SEEDS[STRATEGY_DECISION_NODE_TYPE].to_dict()
        await register_node_type(
            database,
            registry.NodeTypeEntry(**{**other, "owning_feature": "OtherFeature"}),
        )
        with pytest.raises(SeedV2Refused, match="already registered with another entry"):
            await apply_seed_v2(database)
        assert await get_node_type(database, DECISION_NODE_TYPE) == SEED_V1_DECISION
        assert {
            row[0] for row in await database.fetchall(
                "SELECT node_type FROM graph_nodes WHERE node_id IN (?, ?)",
                tuple(node.node_id for node in PROJECTED),
            )
        } == {DECISION_NODE_TYPE}
    finally:
        await database.close()


class _SimulatedCrash(Exception):
    """The process died here."""


class _CrashingDatabase:
    """A database with no transactions that dies before its Nth matching write.

    Every write before it has committed on its own, as on a backend that
    cannot hold the migration in one transaction (au-4244-1 C2).
    """

    def __init__(self, database: AsyncDatabase, prefix: str, nth: int = 1):
        self._database = database
        self._prefix = prefix
        self._remaining = nth
        self.backend_type = database.backend_type

    async def execute(self, sql, params=()):
        if " ".join(sql.split()).startswith(self._prefix):
            self._remaining -= 1
            if self._remaining == 0:
                raise _SimulatedCrash(sql)
        return await self._database.execute(sql, params)

    async def fetchone(self, sql, params=()):
        return await self._database.fetchone(sql, params)

    async def fetchall(self, sql, params=()):
        return await self._database.fetchall(sql, params)


@pytest.mark.parametrize(
    "crash_at",
    (
        pytest.param(("UPDATE graph_nodes", 1, 0, False), id="before-any-retype"),
        pytest.param(("UPDATE graph_nodes", 2, 1, False), id="between-two-retypes"),
        pytest.param((f"UPDATE {registry.TABLE}", 1, 2, False), id="before-the-entry-is-replaced"),
    ),
)
async def test_a_run_that_crashed_midway_resumes(tmp_path, crash_at):
    """Each prefix of the writes resumes to the state an uninterrupted run leaves."""
    clean_path = tmp_path / "clean.db"
    await pre_split_database(clean_path)
    database = await AsyncDatabase.sqlite(str(clean_path))
    await database.close()
    clean = _snapshot(clean_path)

    crashed_path = tmp_path / "crashed.db"
    await pre_split_database(crashed_path)
    database = await AsyncDatabase.sqlite(str(crashed_path))
    try:
        # Undo what the open just did, then rerun it outside any transaction.
        await _undo_migration(database)
        prefix, nth, retyped_before_crash, replaced_before_crash = crash_at
        with pytest.raises(_SimulatedCrash):
            await apply_seed_v2(_CrashingDatabase(database, prefix, nth))
        # The writes before the crash stayed: this is a partial application.
        assert (
            await get_node_type(database, STRATEGY_DECISION_NODE_TYPE)
            == SEEDS[STRATEGY_DECISION_NODE_TYPE]
        )
        projected_types = [
            row[0] for row in await database.fetchall(
                "SELECT node_type FROM graph_nodes WHERE node_id IN (?, ?)",
                tuple(node.node_id for node in PROJECTED),
            )
        ]
        assert projected_types.count(STRATEGY_DECISION_NODE_TYPE) == retyped_before_crash
        assert (await get_node_type(database, DECISION_NODE_TYPE) == SEEDS[DECISION_NODE_TYPE]) is (
            replaced_before_crash
        )
    finally:
        await database.close()
    partial = _snapshot(crashed_path)
    assert (SEED_V2_MARKER,) not in {(row[0],) for row in partial["schema_backfills"]}

    database = await AsyncDatabase.sqlite(str(crashed_path))
    await database.close()

    resumed = _snapshot(crashed_path)
    assert _types(resumed) == _types(clean)
    assert _without_type(resumed, "graph_nodes") == _without_type(clean, "graph_nodes")
    for table in ("graph_edges", "graph_node_owners", "graph_edge_owners"):
        assert resumed[table] == clean[table], table
    database = await AsyncDatabase.sqlite(str(crashed_path))
    try:
        assert await get_node_type(database, DECISION_NODE_TYPE) == SEEDS[DECISION_NODE_TYPE]
        assert (
            await get_node_type(database, STRATEGY_DECISION_NODE_TYPE)
            == SEEDS[STRATEGY_DECISION_NODE_TYPE]
        )
    finally:
        await database.close()


async def test_a_resumed_run_after_the_entry_was_replaced_accepts_the_v2_entry(db_path):
    """The case C2 names: v2 ``decision`` entry, rows still to retype."""
    await pre_split_database(db_path)
    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        await database.execute(
            "UPDATE graph_nodes SET node_type = ? WHERE node_type = ?",
            (DECISION_NODE_TYPE, STRATEGY_DECISION_NODE_TYPE),
        )
        await database.execute(f"DELETE FROM {registry.TABLE} WHERE node_type = ?", (STRATEGY_DECISION_NODE_TYPE,))
        assert await get_node_type(database, DECISION_NODE_TYPE) == SEEDS[DECISION_NODE_TYPE]

        report = await apply_seed_v2(database)

        assert (report.retyped, report.decision_entry_replaced) == (2, False)
        assert (
            await get_node_type(database, STRATEGY_DECISION_NODE_TYPE)
            == SEEDS[STRATEGY_DECISION_NODE_TYPE]
        )
    finally:
        await database.close()


def test_v1_decision_entry_is_the_seed_v1_registered():
    """The precondition compares against the entry #4153 actually seeded."""
    assert SEED_V1_DECISION.to_dict() == {
        "node_type": "decision",
        "owning_feature": "StrategicMemoryFeature",
        "canonical_home": "file",
        "canonical_location": "STRATEGY.yaml",
        "writer": ["kestrel_sovereign.features.strategic_memory.feature"],
        "property_schema": None,
        "retire_behaviour": "canonical_home",
        "retention": "canonical_home",
        "decay": "canonical_home",
        "is_projection": True,
    }
    # The projection keeps the home the v1 row gave it, under its new name.
    assert SEEDS[STRATEGY_DECISION_NODE_TYPE].to_dict() == {
        **SEED_V1_DECISION.to_dict(), "node_type": STRATEGY_DECISION_NODE_TYPE,
    }


async def _no_schema(db):
    """Schema initializer for the admin connection: run no DDL."""


# Parametrized with "postgres" so the #3381 guard fails this case, rather than
# letting it skip, in a job that provides PostgreSQL.
@pytest.fixture(params=["postgres"])
def postgres_url(request):
    import os

    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    return url


async def _graph_rows(database: AsyncDatabase) -> dict:
    return {
        "nodes": sorted(await database.fetchall(
            "SELECT node_id, node_type, label, properties FROM graph_nodes"
        )),
        "edges": sorted(await database.fetchall(
            "SELECT source_id, target_id, label, properties FROM graph_edges"
        )),
        "owners": sorted(await database.fetchall(
            "SELECT node_id, agent_id FROM graph_node_owners"
        )),
    }


async def test_migration_on_postgres(postgres_url):
    """The same split on PostgreSQL, in a schema of its own (as the #4153 case)."""
    from tests.utils.postgres_schema import (
        disposable_postgres_schema,
        pgvector_schema,
        quoted_search_path,
        with_search_path,
    )

    admin = await AsyncDatabase.postgres(postgres_url, schema_initializer=_no_schema)
    try:
        await pgvector_schema(admin)
        async with disposable_postgres_schema(admin, "node_type_seed_v2") as schema:
            url = with_search_path(postgres_url, quoted_search_path(schema))
            database = await AsyncDatabase.postgres(url)
            try:
                await rewind_to_v1(database)
                await write_pre_split_graph(database)
                before = await _graph_rows(database)
            finally:
                await database.close()

            database = await AsyncDatabase.postgres(url)
            try:
                after = await _graph_rows(database)
                assert await get_node_type(database, DECISION_NODE_TYPE) == SEEDS[DECISION_NODE_TYPE]
                assert (
                    await get_node_type(database, STRATEGY_DECISION_NODE_TYPE)
                    == SEEDS[STRATEGY_DECISION_NODE_TYPE]
                )
                again = await apply_seed_v2(database)
                assert await _graph_rows(database) == after
            finally:
                await database.close()
    finally:
        await admin.close()

    types = {row[0]: row[1] for row in after["nodes"]}
    for node in PROJECTED:
        assert types[node.node_id] == STRATEGY_DECISION_NODE_TYPE, node.node_id
    for node in UNMARKED:
        assert types[node.node_id] == DECISION_NODE_TYPE, node.node_id
    assert [(r[0], r[2], r[3]) for r in after["nodes"]] == [(r[0], r[2], r[3]) for r in before["nodes"]]
    assert (after["edges"], after["owners"]) == (before["edges"], before["owners"])
    assert (again.retyped, again.left, again.id_shape_mismatches) == (0, len(UNMARKED), 1)
    assert not again.decision_entry_replaced
