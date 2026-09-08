"""Unstamped graph rows sort last, and a midnight-UTC stamp is timed, on both backends (#3255).

The strategic-memory projections now leave ``created_at`` absent for a
dateless row and write ``YYYY-MM-DDT00:00:00+00:00`` for a dated one. Two
things must hold on SQLite and Postgres alike: a row with no stamp trails a
created-ordered recall (Postgres put NULL first under DESC), and the scoped
EPHEMERAL purge compares the midnight shape as an instant.
"""

from __future__ import annotations

import uuid

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode


async def _database(db_backend) -> AsyncDatabase:
    db = AsyncDatabase(db_backend)
    await db._init_schema()
    db._initialized = True
    return db


def _node(agent, node_id, node_type, **properties):
    return GraphNode(node_id=node_id, node_type=node_type, label=node_id, properties={"agent_id": agent, **properties})


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_unstamped_rows_sort_last_under_created_order(db_backend):
    db = await _database(db_backend)
    agent = f"did:test:order-{uuid.uuid4().hex}"
    node_type = f"stamp_order_{uuid.uuid4().hex[:8]}"
    graph = AsyncGraphStore(db, agent_id=agent)
    await graph.add_node(_node(agent, f"{agent}:unstamped", node_type))
    await graph.add_node(_node(agent, f"{agent}:older", node_type, created_at="2026-07-01T00:00:00+00:00"))
    await graph.add_node(_node(agent, f"{agent}:newer", node_type, created_at="2026-07-02T00:00:00+00:00"))

    nodes = await graph.query_nodes_by_type_and_property(node_type, filters={"agent_id": agent}, order_by_created=True, limit=10)
    assert [n.node_id.rsplit(":", 1)[1] for n in nodes] == ["newer", "older", "unstamped"]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_midnight_stamp_is_timed_by_the_scoped_purge(db_backend):
    db = await _database(db_backend)
    agent = f"did:test:purge-{uuid.uuid4().hex}"
    node_type = f"stamp_purge_{uuid.uuid4().hex[:8]}"
    graph = AsyncGraphStore(db, agent_id=agent)
    await graph.add_node(_node(agent, f"{agent}:dated", node_type, created_at="2026-07-01T00:00:00+00:00"))

    # A watermark later than midnight keeps the row; one earlier purges it.
    assert await graph.purge_agent_nodes(agent, since_iso="2026-07-01 00:00:01") == 0
    assert await graph.get_node(f"{agent}:dated") is not None
    assert await graph.purge_agent_nodes(agent, since_iso="2026-06-30 23:59:59") == 1
    assert await graph.get_node(f"{agent}:dated") is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "node_type, family",
    [("action_item", "idx_graph_nodes_action_created"), ("todo_item", "idx_graph_nodes_todo_created")],
)
async def test_created_ordered_reads_stay_an_index_walk(db_backend, node_type, family):
    """``DESC NULLS LAST`` must keep the created-at partial indexes in play.
    On Postgres a plain DESC is a backward scan of an ASC index, but DESC
    NULLS LAST matches neither direction of it and the planner falls back
    to a sequential scan; the index is therefore stored DESC NULLS LAST, for
    the action-item and the todo reads alike. SQLite's DESC is already NULLS
    LAST and its index serves both ways."""
    db = await _database(db_backend)
    agent = f"did:test:index-{uuid.uuid4().hex}"
    graph = AsyncGraphStore(db, agent_id=agent)
    for n in range(30):
        await graph.add_node(_node(agent, f"{agent}:a{n}", node_type, status="pending", created_at=f"2026-07-{1 + n % 28:02d}T00:00:00+00:00"))
    je = graph._json_extract
    # The partial index's own predicate and the store's ordering, nothing
    # else, so the only index that can serve the ORDER BY is the created-at
    # one: a plan that sorts is a plan the index did not serve.
    body = (
        "SELECT node_id, node_type, label, properties FROM graph_nodes "
        f"WHERE node_type = '{node_type}' "
        f"ORDER BY {je('properties', 'created_at')} DESC NULLS LAST LIMIT 25"
    )
    # Statistics on both engines: without them SQLite's planner serves the
    # node_type predicate from an unrelated index and sorts, which is the
    # plan this test exists to reject.
    await db.execute("ANALYZE graph_nodes" if db.backend_type == "postgres" else "ANALYZE")
    if db.backend_type == "postgres":
        async with db.transaction():
            # A tiny table makes the planner prefer a scan-and-sort even
            # with a perfect index; forbid the scans and the sort so the
            # plan says whether an index CAN serve the ordering at all.
            # With the ASC index of round 1 the planner still has to sort
            # (a Sort node survives even with enable_sort off); with the
            # DESC NULLS LAST index it walks the index in order.
            for setting in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
                await db.execute(f"SET LOCAL {setting} = off")
            rows = await db.fetchall("EXPLAIN " + body)
        plan = " ".join(str(r) for r in rows)
        # The name carries ensure_index's definition fingerprint; the family
        # prefix is the stable part.
        assert f"Index Scan using {family}_" in plan and "Sort" not in plan, plan
    else:
        # SQLite spells a full ordered index walk "SCAN <table> USING INDEX
        # <name>"; the plan to reject is the one that serves node_type from
        # another index and sorts with a temp b-tree.
        rows = await db.fetchall("EXPLAIN QUERY PLAN " + body)
        plan = " ".join(str(r) for r in rows)
        assert f"{family}_" in plan and "TEMP B-TREE" not in plan, plan
