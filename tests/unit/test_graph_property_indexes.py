"""Tests for graph property indexes and the sortable-timestamp invariant.

Covers:
- Backend-specific JSON property index creation (SQLite + PostgreSQL SQL)
- Structural verification that indexes are used via EXPLAIN QUERY PLAN
- query_nodes_by_type_and_property with equality and range filters
- The created_at sortable-timestamp invariant (format + round-trip)

See issue #669 for context on why timing-based benchmarks were replaced
with structural EXPLAIN checks.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import List

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import (
    AsyncGraphStore,
    GraphNode,
    _POSTGRES_JSON_INDEXES,
    _SQLITE_JSON_INDEXES,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def db(tmp_path):
    """In-memory SQLite database with schema initialized."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "test.db"))
    yield db
    await db.close()


@pytest_asyncio.fixture
async def graph(db):
    """Graph store with property indexes created."""
    store = AsyncGraphStore(db)
    await store.ensure_property_indexes()
    return store


# =============================================================================
# 1. Index creation and PostgreSQL parity
# =============================================================================


class TestPropertyIndexCreation:
    """Verify that JSON property indexes exist for both backends."""

    @pytest.mark.asyncio
    async def test_sqlite_indexes_created(self, graph: AsyncGraphStore):
        """All three SQLite JSON indexes should exist after ensure_property_indexes."""
        rows = await graph.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_graph_nodes_%'"
        )
        index_names = {row[0] for row in rows}
        assert "idx_graph_nodes_agent" in index_names
        assert "idx_graph_nodes_action_status" in index_names
        assert "idx_graph_nodes_action_created" in index_names

    @pytest.mark.asyncio
    async def test_ensure_indexes_idempotent(self, graph: AsyncGraphStore):
        """Calling ensure_property_indexes twice must not raise."""
        await graph.ensure_property_indexes()  # second call
        rows = await graph.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_graph_nodes_%'"
        )
        assert len(rows) >= 3

    def test_postgres_indexes_include_created_at(self):
        """PostgreSQL index list must contain a created_at expression index."""
        created_at_indexes = [
            idx for idx in _POSTGRES_JSON_INDEXES if "created_at" in idx
        ]
        assert len(created_at_indexes) >= 1, (
            "PostgreSQL must have a created_at expression index for parity"
        )
        # Verify it uses the correct jsonb extraction syntax
        assert any("(properties::jsonb)->>'created_at'" in idx for idx in created_at_indexes)

    def test_postgres_indexes_include_agent_id(self):
        """PostgreSQL index list must contain an agent_id expression index."""
        agent_indexes = [idx for idx in _POSTGRES_JSON_INDEXES if "agent_id" in idx]
        assert len(agent_indexes) >= 1

    def test_postgres_indexes_include_status(self):
        """PostgreSQL index list must contain a status expression index."""
        status_indexes = [idx for idx in _POSTGRES_JSON_INDEXES if "'status'" in idx]
        assert len(status_indexes) >= 1

    def test_sqlite_and_postgres_index_count_parity(self):
        """Both backends should have the same number of property indexes
        (excluding the Postgres-only GIN index)."""
        pg_non_gin = [idx for idx in _POSTGRES_JSON_INDEXES if "gin" not in idx.lower()]
        assert len(_SQLITE_JSON_INDEXES) == len(pg_non_gin), (
            f"SQLite has {len(_SQLITE_JSON_INDEXES)} indexes but Postgres has "
            f"{len(pg_non_gin)} (excluding GIN) — parity broken"
        )


# =============================================================================
# 2. Structural index verification (replaces timing-based benchmark)
# =============================================================================


class TestIndexUsedStructurally:
    """Use EXPLAIN QUERY PLAN to prove indexes are hit.

    This replaces the <50ms timing benchmark from PR #663 which was
    flake-prone on slow CI runners.  A structural check actually proves
    index usage regardless of hardware speed.
    """

    @pytest.mark.asyncio
    async def test_agent_id_query_uses_index(self, graph: AsyncGraphStore):
        """Filtering by node_type + agent_id should hit idx_graph_nodes_agent."""
        plan = await _explain_query(
            graph,
            node_type="action_item",
            filters={"agent_id": "test-agent"},
        )
        assert _plan_uses_index(plan, "idx_graph_nodes_agent"), (
            f"Expected idx_graph_nodes_agent in query plan, got: {plan}"
        )

    @pytest.mark.asyncio
    async def test_status_filter_uses_index(self, graph: AsyncGraphStore):
        """Filtering by status on action_items should hit idx_graph_nodes_action_status."""
        plan = await _explain_query(
            graph,
            node_type="action_item",
            filters={"agent_id": "test-agent", "status": "pending"},
        )
        # Either the agent index or the status index may be chosen
        assert _plan_uses_any_property_index(plan), (
            f"Expected a property index in query plan, got: {plan}"
        )

    @pytest.mark.asyncio
    async def test_created_since_uses_index(self, graph: AsyncGraphStore):
        """Range filter on created_at should hit idx_graph_nodes_action_created."""
        plan = await _explain_query(
            graph,
            node_type="action_item",
            filters={"agent_id": "test-agent"},
            created_since="2026-01-01T00:00:00+00:00",
        )
        assert _plan_uses_any_property_index(plan), (
            f"Expected a property index in query plan, got: {plan}"
        )


async def _explain_query(
    graph: AsyncGraphStore,
    node_type: str,
    filters: dict | None = None,
    created_since: str | None = None,
) -> List[str]:
    """Build the same SQL as query_nodes_by_type_and_property and EXPLAIN it."""
    clauses = ["node_type = ?"]
    params: list = [node_type]

    for key, value in (filters or {}).items():
        clauses.append(f"{graph._json_extract('properties', key)} = ?")
        params.append(value)

    if created_since:
        clauses.append(f"{graph._json_extract('properties', 'created_at')} >= ?")
        params.append(created_since)

    sql = (
        "SELECT node_id, node_type, label, properties FROM graph_nodes"
        f" WHERE {' AND '.join(clauses)}"
        f" ORDER BY {graph._json_extract('properties', 'created_at')} DESC"
        f" LIMIT ?"
    )
    params.append(100)

    rows = await graph.db.fetchall(f"EXPLAIN QUERY PLAN {sql}", tuple(params))
    return [str(row) for row in rows]


def _plan_uses_index(plan: List[str], index_name: str) -> bool:
    """Check if EXPLAIN QUERY PLAN output references a specific index."""
    combined = " ".join(plan)
    return index_name in combined


def _plan_uses_any_property_index(plan: List[str]) -> bool:
    """Check if any of the graph_nodes property indexes are used."""
    combined = " ".join(plan)
    return any(
        name in combined
        for name in [
            "idx_graph_nodes_agent",
            "idx_graph_nodes_action_status",
            "idx_graph_nodes_action_created",
        ]
    )


# =============================================================================
# 3. query_nodes_by_type_and_property functional tests
# =============================================================================


class TestQueryNodesByTypeAndProperty:

    @pytest.mark.asyncio
    async def test_basic_query(self, graph: AsyncGraphStore):
        """Query returns nodes matching type and property filters."""
        await _insert_action(graph, "a1", "agent-1", "pending", "2026-04-01T00:00:00+00:00")
        await _insert_action(graph, "a2", "agent-1", "done", "2026-04-02T00:00:00+00:00")
        await _insert_action(graph, "a3", "agent-2", "pending", "2026-04-03T00:00:00+00:00")

        results = await graph.query_nodes_by_type_and_property(
            "action_item",
            filters={"agent_id": "agent-1", "status": "pending"},
        )
        assert len(results) == 1
        assert results[0].node_id == "a1"

    @pytest.mark.asyncio
    async def test_created_since_filter(self, graph: AsyncGraphStore):
        """created_since filters out older nodes."""
        await _insert_action(graph, "old", "ag", "pending", "2026-01-01T00:00:00+00:00")
        await _insert_action(graph, "new", "ag", "pending", "2026-06-01T00:00:00+00:00")

        results = await graph.query_nodes_by_type_and_property(
            "action_item",
            filters={"agent_id": "ag"},
            created_since="2026-03-01T00:00:00+00:00",
        )
        assert len(results) == 1
        assert results[0].node_id == "new"

    @pytest.mark.asyncio
    async def test_order_by_created_desc(self, graph: AsyncGraphStore):
        """Results are ordered by created_at descending by default."""
        await _insert_action(graph, "first", "ag", "pending", "2026-01-01T00:00:00+00:00")
        await _insert_action(graph, "second", "ag", "pending", "2026-06-01T00:00:00+00:00")
        await _insert_action(graph, "third", "ag", "pending", "2026-03-01T00:00:00+00:00")

        results = await graph.query_nodes_by_type_and_property(
            "action_item",
            filters={"agent_id": "ag"},
        )
        ids = [r.node_id for r in results]
        assert ids == ["second", "third", "first"]

    @pytest.mark.asyncio
    async def test_limit(self, graph: AsyncGraphStore):
        """Limit caps the number of returned nodes."""
        for i in range(5):
            await _insert_action(graph, f"n{i}", "ag", "pending", f"2026-0{i+1}-01T00:00:00+00:00")

        results = await graph.query_nodes_by_type_and_property(
            "action_item",
            filters={"agent_id": "ag"},
            limit=2,
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_agent_isolation(self, graph: AsyncGraphStore):
        """Nodes from other agents are not returned."""
        await _insert_action(graph, "mine", "agent-1", "pending", "2026-04-01T00:00:00+00:00")
        await _insert_action(graph, "theirs", "agent-2", "pending", "2026-04-01T00:00:00+00:00")

        results = await graph.query_nodes_by_type_and_property(
            "action_item",
            filters={"agent_id": "agent-1"},
        )
        assert len(results) == 1
        assert results[0].node_id == "mine"


async def _insert_action(
    graph: AsyncGraphStore,
    node_id: str,
    agent_id: str,
    status: str,
    created_at: str,
) -> None:
    """Helper to insert an action_item node."""
    await graph.add_node(GraphNode(
        node_id=node_id,
        node_type="action_item",
        label=f"Action {node_id}",
        properties={
            "agent_id": agent_id,
            "status": status,
            "created_at": created_at,
            "text": f"Do something for {node_id}",
        },
    ))


# =============================================================================
# 4. Sortable-timestamp invariant for created_at
# =============================================================================


class TestCreatedAtTimestampInvariant:
    """Verify the created_at format documented in schema_router.py.

    The sortable-timestamp invariant requires created_at values to be:
    1. Produced by datetime.now(timezone.utc).isoformat()
    2. Parseable by datetime.fromisoformat()
    3. Contain a UTC offset (+00:00)
    4. Lexicographically sortable
    """

    def test_utc_isoformat_is_parseable(self):
        """datetime.now(timezone.utc).isoformat() round-trips through fromisoformat."""
        ts = datetime.now(timezone.utc).isoformat()
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_utc_isoformat_contains_offset(self):
        """The produced timestamp must contain +00:00."""
        ts = datetime.now(timezone.utc).isoformat()
        assert "+00:00" in ts, f"Expected +00:00 in timestamp, got: {ts}"

    def test_utc_isoformat_is_lexicographically_sortable(self):
        """Earlier timestamps must sort before later ones via plain string comparison."""
        earlier = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
        later = datetime(2026, 12, 31, tzinfo=timezone.utc).isoformat()
        assert earlier < later, f"{earlier} should sort before {later}"

    def test_utc_isoformat_matches_expected_pattern(self):
        """Timestamp must match YYYY-MM-DDTHH:MM:SS.ffffff+00:00 or without microseconds."""
        ts = datetime.now(timezone.utc).isoformat()
        # ISO format: 2026-04-21T09:00:00.123456+00:00 or 2026-04-21T09:00:00+00:00
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$"
        assert re.match(pattern, ts), f"Timestamp {ts} doesn't match expected ISO-8601 pattern"

    @pytest.mark.asyncio
    async def test_stored_action_items_have_valid_created_at(self, graph: AsyncGraphStore):
        """Action items persisted via graph store must have parseable created_at."""
        ts = datetime.now(timezone.utc).isoformat()
        await _insert_action(graph, "ts-test", "ag", "pending", ts)

        node = await graph.get_node("ts-test")
        stored_ts = node.properties["created_at"]

        # Must round-trip
        parsed = datetime.fromisoformat(stored_ts)
        assert parsed.tzinfo is not None
        assert "+00:00" in stored_ts


# =============================================================================
# 5. _json_extract backend SQL generation
# =============================================================================


class TestJsonExtract:

    def test_sqlite_json_extract(self, graph: AsyncGraphStore):
        """SQLite backend should use json_extract(col, '$.key') syntax."""
        sql = graph._json_extract("properties", "agent_id")
        assert sql == "json_extract(properties, '$.agent_id')"

    def test_sqlite_created_at_extract(self, graph: AsyncGraphStore):
        """SQLite backend should correctly extract created_at."""
        sql = graph._json_extract("properties", "created_at")
        assert sql == "json_extract(properties, '$.created_at')"
