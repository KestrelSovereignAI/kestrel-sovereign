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
async def test_created_ordered_action_items_stay_an_index_walk(db_backend):
    """``DESC NULLS LAST`` must keep the created-at partial index in play.
    On Postgres a plain DESC is a backward scan of an ASC index, but DESC
    NULLS LAST matches neither direction of it and the planner falls back
    to a sequential scan; the index is therefore stored DESC NULLS LAST.
    SQLite's DESC is already NULLS LAST and its index serves both ways."""
    db = await _database(db_backend)
    agent = f"did:test:index-{uuid.uuid4().hex}"
    graph = AsyncGraphStore(db, agent_id=agent)
    for n in range(30):
        await graph.add_node(_node(agent, f"{agent}:a{n}", "action_item", status="pending", created_at=f"2026-07-{1 + n % 28:02d}T00:00:00+00:00"))
    je = graph._json_extract
    # The partial index's own predicate and the store's ordering, nothing
    # else, so the only index that can serve the ORDER BY is the created-at
    # one: a plan that sorts is a plan the index did not serve.
    body = (
        "SELECT node_id, node_type, label, properties FROM graph_nodes "
        "WHERE node_type = 'action_item' "
        f"ORDER BY {je('properties', 'created_at')} DESC NULLS LAST LIMIT 25"
    )
    if db.backend_type == "postgres":
        await db.execute("ANALYZE graph_nodes")
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
        assert "Index Scan using idx_graph_nodes_action_created_desc" in plan and "Sort" not in plan, plan
    else:
        # SQLite's DESC is already NULLS LAST, so the words must not change
        # the plan: the same index (whichever the planner picks at this
        # size) and no table scan, with and without them.
        rows = await db.fetchall("EXPLAIN QUERY PLAN " + body)
        plan = " ".join(str(r) for r in rows).upper()
        rows_plain = await db.fetchall("EXPLAIN QUERY PLAN " + body.replace(" NULLS LAST", ""))
        plan_plain = " ".join(str(r) for r in rows_plain).upper()
        assert "USING INDEX" in plan and "SCAN GRAPH_NODES" not in plan, plan
        assert plan == plan_plain, (plan, plan_plain)
