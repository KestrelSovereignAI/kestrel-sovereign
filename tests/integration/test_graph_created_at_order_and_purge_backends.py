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
