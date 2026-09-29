"""Person-concept listing is tenant-scoped on both backends (#3228).

Shared-table shape: several stores over ONE database. A tenant-bound store
must never list another agent's people, and where the store is unbound the
``concept:{agent_id}:`` prefix is the only fence, so it must be an exact
prefix and not a containment test.
"""

from __future__ import annotations

import uuid

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.schema_router import PersonResolver


async def _database(db_backend) -> AsyncDatabase:
    db = AsyncDatabase(db_backend)
    await db._init_schema()
    db._initialized = True
    return db


async def _seed_person(graph, agent, label, *, node_id=None):
    await graph.add_node(GraphNode(
        node_id=node_id or f"concept:{agent}:{label.lower()}", node_type="concept",
        label=label, properties={"agent_id": agent, "mention_count": 1},
    ))


async def _listed(store, agent):
    return {cid for cid, _ in await PersonResolver(store)._list_person_concepts(agent)}


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_bound_stores_over_one_database_never_list_each_others_people(db_backend):
    db = await _database(db_backend)
    agent_a, agent_b = f"did:test:a-{uuid.uuid4().hex}", f"did:test:b-{uuid.uuid4().hex}"
    graph_a, graph_b = AsyncGraphStore(db, agent_id=agent_a), AsyncGraphStore(db, agent_id=agent_b)
    await _seed_person(graph_a, agent_a, "Alice")
    await _seed_person(graph_b, agent_b, "Alice")
    await _seed_person(graph_b, agent_b, "Bob")

    assert await _listed(graph_a, agent_a) == {f"concept:{agent_a}:alice"}
    assert await _listed(graph_b, agent_b) == {f"concept:{agent_b}:alice", f"concept:{agent_b}:bob"}
    # Even asked about the other agent's prefix, a bound store yields nothing.
    assert await _listed(graph_a, agent_b) == set()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_unbound_store_lists_by_exact_prefix_not_containment(db_backend):
    """With no owner binding the prefix is the only fence. Neither a label
    that embeds another agent's prefix nor an agent id that is a strict
    prefix of another's (the delimiter case) may leak across it."""
    db = await _database(db_backend)
    agent_a, agent_b = f"did:test:a-{uuid.uuid4().hex}", f"did:test:b-{uuid.uuid4().hex}"
    sibling = f"{agent_a}-suffix"  # agent_a is a strict prefix of this id
    unbound = AsyncGraphStore(db)
    await _seed_person(unbound, agent_a, "Alice")
    await _seed_person(unbound, agent_b, "Bob")
    await _seed_person(unbound, sibling, "Mallory")
    hostile_label = f"concept:{agent_a}:mallory"
    await _seed_person(unbound, agent_b, hostile_label, node_id=f"concept:{agent_b}:{hostile_label}")

    assert await _listed(unbound, agent_a) == {f"concept:{agent_a}:alice"}
    assert await _listed(unbound, sibling) == {f"concept:{sibling}:mallory"}
    assert await _listed(unbound, agent_b) == {f"concept:{agent_b}:bob", f"concept:{agent_b}:{hostile_label}"}
