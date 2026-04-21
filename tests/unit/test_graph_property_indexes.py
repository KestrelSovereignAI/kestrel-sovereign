"""Tests for JSON-path property indexes and query_nodes_by_type_and_property.

Covers:
- Backend-aware JSON-path index creation (SQLite)
- query_nodes_by_type_and_property with equality / range filters
- _json_extract helper for SQLite and PostgreSQL SQL generation
- Benchmark: 50k action_item nodes, per-agent pending query < 50ms
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode


@pytest_asyncio.fixture
async def db(tmp_path):
    """Create an in-memory SQLite database with full schema."""
    db_path = str(tmp_path / "test_graph.db")
    database = await AsyncDatabase.sqlite(db_path)
    yield database
    await database.close()


@pytest_asyncio.fixture
async def graph(db):
    return AsyncGraphStore(db)


# =====================================================================
# _json_extract helper
# =====================================================================


class TestJsonExtract:

    def test_sqlite_syntax(self, graph):
        assert graph._json_extract("properties", "agent_id") == \
            "json_extract(properties, '$.agent_id')"

    def test_sqlite_nested_path(self, graph):
        assert graph._json_extract("col", "foo") == \
            "json_extract(col, '$.foo')"


class TestJsonExtractPostgres:
    """Verify PostgreSQL SQL generation without needing a real PG backend."""

    def test_postgres_syntax(self):
        from unittest.mock import MagicMock
        mock_db = MagicMock()
        mock_db.backend_type = "postgres"
        store = AsyncGraphStore(mock_db)
        assert store._json_extract("properties", "agent_id") == \
            "(properties::jsonb)->>'agent_id'"


# =====================================================================
# Index creation (SQLite)
# =====================================================================


class TestJsonPathIndexes:

    @pytest.mark.asyncio
    async def test_indexes_exist_after_init(self, db):
        """The three JSON-path indexes must be created during _init_schema."""
        rows = await db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_graph_nodes_%'",
        )
        names = {r[0] for r in rows}
        assert "idx_graph_nodes_agent" in names
        assert "idx_graph_nodes_action_status" in names
        assert "idx_graph_nodes_action_created" in names

    @pytest.mark.asyncio
    async def test_idempotent_schema_init(self, db):
        """Running _init_schema again must not fail (IF NOT EXISTS)."""
        await db._init_schema()
        rows = await db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND name = 'idx_graph_nodes_agent'",
        )
        assert len(rows) == 1


# =====================================================================
# query_nodes_by_type_and_property
# =====================================================================


def _make_action(agent_id, status="pending", created_at=None, assignee=None):
    nid = f"action:{agent_id}:{uuid.uuid4().hex[:8]}"
    return GraphNode(
        node_id=nid,
        node_type="action_item",
        label=f"Item {nid[-8:]}",
        properties={
            "agent_id": agent_id,
            "status": status,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "assignee_concept_id": assignee,
            "text": f"Do something ({nid[-8:]})",
        },
    )


class TestQueryNodesByTypeAndProperty:

    @pytest.mark.asyncio
    async def test_filters_by_agent_id(self, graph):
        await graph.add_node(_make_action("agent-A"))
        await graph.add_node(_make_action("agent-A"))
        await graph.add_node(_make_action("agent-B"))

        results = await graph.query_nodes_by_type_and_property(
            "action_item", filters={"agent_id": "agent-A"},
        )
        assert len(results) == 2
        assert all(n.properties["agent_id"] == "agent-A" for n in results)

    @pytest.mark.asyncio
    async def test_filters_by_status(self, graph):
        await graph.add_node(_make_action("agent-A", status="pending"))
        await graph.add_node(_make_action("agent-A", status="done"))

        results = await graph.query_nodes_by_type_and_property(
            "action_item", filters={"agent_id": "agent-A", "status": "pending"},
        )
        assert len(results) == 1
        assert results[0].properties["status"] == "pending"

    @pytest.mark.asyncio
    async def test_created_since_filter(self, graph):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        recent = datetime.now(timezone.utc).isoformat()
        await graph.add_node(_make_action("agent-A", created_at=old))
        await graph.add_node(_make_action("agent-A", created_at=recent))

        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        results = await graph.query_nodes_by_type_and_property(
            "action_item",
            filters={"agent_id": "agent-A"},
            created_since=since,
        )
        assert len(results) == 1
        assert results[0].properties["created_at"] == recent

    @pytest.mark.asyncio
    async def test_respects_limit(self, graph):
        for _ in range(10):
            await graph.add_node(_make_action("agent-A"))

        results = await graph.query_nodes_by_type_and_property(
            "action_item", filters={"agent_id": "agent-A"}, limit=3,
        )
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_ordered_by_created_at_desc(self, graph):
        t1 = "2026-01-01T00:00:00+00:00"
        t2 = "2026-06-01T00:00:00+00:00"
        await graph.add_node(_make_action("agent-A", created_at=t1))
        await graph.add_node(_make_action("agent-A", created_at=t2))

        results = await graph.query_nodes_by_type_and_property(
            "action_item", filters={"agent_id": "agent-A"},
        )
        assert results[0].properties["created_at"] == t2
        assert results[1].properties["created_at"] == t1

    @pytest.mark.asyncio
    async def test_empty_result(self, graph):
        results = await graph.query_nodes_by_type_and_property(
            "action_item", filters={"agent_id": "nonexistent"},
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_node_type_isolation(self, graph):
        """Nodes of different types must not leak through."""
        await graph.add_node(_make_action("agent-A"))
        decision = GraphNode(
            node_id="decision:agent-A:xyz",
            node_type="decision",
            label="Some decision",
            properties={"agent_id": "agent-A", "created_at": datetime.now(timezone.utc).isoformat()},
        )
        await graph.add_node(decision)

        results = await graph.query_nodes_by_type_and_property(
            "action_item", filters={"agent_id": "agent-A"},
        )
        assert len(results) == 1
        assert results[0].node_type == "action_item"

    @pytest.mark.asyncio
    async def test_assignee_filter(self, graph):
        await graph.add_node(_make_action("agent-A", assignee="concept:alice"))
        await graph.add_node(_make_action("agent-A", assignee="concept:bob"))

        results = await graph.query_nodes_by_type_and_property(
            "action_item",
            filters={"agent_id": "agent-A", "assignee_concept_id": "concept:alice"},
        )
        assert len(results) == 1
        assert results[0].properties["assignee_concept_id"] == "concept:alice"

    @pytest.mark.asyncio
    async def test_no_filters_returns_all_of_type(self, graph):
        await graph.add_node(_make_action("agent-A"))
        await graph.add_node(_make_action("agent-B"))

        results = await graph.query_nodes_by_type_and_property("action_item")
        assert len(results) == 2


# =====================================================================
# Benchmark: 50k action_item nodes across 10 agents
# =====================================================================


class TestBenchmark:

    @pytest.mark.asyncio
    async def test_50k_nodes_per_agent_query_under_50ms(self, graph):
        """Insert 50k action_item nodes across 10 agents, then query
        per-agent pending items and verify < 50ms.
        """
        agents = [f"agent-{i}" for i in range(10)]
        statuses = ["pending", "done", "cancelled"]

        # Bulk insert 50k nodes (5k per agent)
        batch_sql = graph._upsert_node_sql()
        rows = []
        for i in range(50_000):
            agent = agents[i % 10]
            status = statuses[i % 3]
            nid = f"action:{agent}:{i:06d}"
            props = json.dumps({
                "agent_id": agent,
                "status": status,
                "created_at": f"2026-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00+00:00",
                "text": f"Task {i}",
            })
            rows.append((nid, "action_item", f"Task {i}"[:120], props))

        # Use execute_many for performance
        await graph.db._backend.execute_many(batch_sql, rows)

        # Warm up SQLite query planner
        await graph.query_nodes_by_type_and_property(
            "action_item", filters={"agent_id": "agent-0", "status": "pending"}, limit=25,
        )

        # Timed query
        start = time.perf_counter()
        results = await graph.query_nodes_by_type_and_property(
            "action_item", filters={"agent_id": "agent-0", "status": "pending"}, limit=25,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) > 0
        assert all(r.properties["agent_id"] == "agent-0" for r in results)
        assert all(r.properties["status"] == "pending" for r in results)
        assert elapsed_ms < 50, f"Query took {elapsed_ms:.1f}ms, expected < 50ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
