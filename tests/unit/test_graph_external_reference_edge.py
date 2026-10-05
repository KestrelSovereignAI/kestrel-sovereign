"""Edges from an owned node to a reference outside the graph (#3091).

``todo_link_task`` projects each link as a ``linked_to`` edge whose target is
``"<link_type>:<target>"`` — a GitHub issue or URL that no agent owns and no
graph node represents. Ordinary bound admission requires the writer to own
both endpoints, so it refused every such edge with "Graph edge endpoints are
not both owned by the bound agent".

``add_external_reference_edge`` checks ownership on the agent-owned endpoint
only, and only when the target is outside the graph. A target that is in the
graph — a node or an ownership reservation — gets the ordinary rule, so the
writer never connects two tenants' nodes.

Runs on both backends; the PostgreSQL pass needs ``TEST_POSTGRES_URL``.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

import kestrel_sovereign.storage.async_graph_store as graph_store_module
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode


def _nid(prefix: str) -> str:
    """Unique id, so a reused PostgreSQL database never collides."""
    return f"{prefix}:{uuid.uuid4().hex}"


def _agent() -> str:
    return f"agent:{uuid.uuid4().hex}"


async def _owned_node(store: AsyncGraphStore, node_id: str) -> None:
    await store.add_node(
        GraphNode(
            node_id=node_id,
            node_type="todo_item",
            label="Owned",
            properties={"agent_id": store.agent_id},
        )
    )


async def _edge_row(db, source_id: str, target_id: str, label: str):
    return await db.fetchone(
        "SELECT 1 FROM graph_edges "
        "WHERE source_id = ? AND target_id = ? AND label = ?",
        (source_id, target_id, label),
    )


@pytest_asyncio.fixture
async def db(db_backend):
    database = AsyncDatabase(db_backend)
    await database._init_schema()
    database._initialized = True
    return database


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_edge_to_a_target_outside_the_graph_is_admitted(db):
    """The #3091 shape: an owned todo linked to a GitHub issue."""
    store = AsyncGraphStore(db, agent_id=_agent())
    todo_id = _nid("todo")
    issue_ref = f"github_issue:KestrelSovereignAI/kestrel-sovereign#{uuid.uuid4().hex}"
    await _owned_node(store, todo_id)

    with pytest.raises(Exception, match="not both owned by the bound agent"):
        await store.add_edge(todo_id, issue_ref, "linked_to")

    await store.add_external_reference_edge(
        todo_id, issue_ref, "linked_to", {"agent_id": store.agent_id}
    )

    edges = await store.get_edges(todo_id, direction="out")
    assert [(e.target_id, e.label) for e in edges] == [(issue_ref, "linked_to")]
    owners = await db.fetchall(
        "SELECT agent_id FROM graph_edge_owners "
        "WHERE source_id = ? AND target_id = ? AND label = ?",
        (todo_id, issue_ref, "linked_to"),
    )
    assert {row[0] for row in owners} == {store.agent_id}
    # Still no node for the reference: the edge names it, nothing creates it.
    assert await store.get_node(issue_ref) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_repeated_external_reference_edge_is_one_edge(db):
    store = AsyncGraphStore(db, agent_id=_agent())
    todo_id = _nid("todo")
    ref = _nid("url")
    await _owned_node(store, todo_id)

    await store.add_external_reference_edge(todo_id, ref, "linked_to", {"n": 1})
    await store.add_external_reference_edge(todo_id, ref, "linked_to", {"n": 2})

    edges = await store.get_edges(todo_id, direction="out")
    assert len(edges) == 1
    assert edges[0].properties == {"n": 2}


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_target_owned_by_the_writer_is_admitted(db):
    store = AsyncGraphStore(db, agent_id=_agent())
    source_id = _nid("todo")
    target_id = _nid("todo")
    await _owned_node(store, source_id)
    await _owned_node(store, target_id)

    await store.add_external_reference_edge(source_id, target_id, "linked_to")

    assert await _edge_row(db, source_id, target_id, "linked_to") is not None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_edge_between_two_tenants_nodes_is_still_refused(db):
    """Neither writer connects an owned node to another agent's node."""
    store = AsyncGraphStore(db, agent_id=_agent())
    foreign = AsyncGraphStore(db, agent_id=_agent())
    source_id = _nid("todo")
    foreign_id = _nid("todo")
    await _owned_node(store, source_id)
    await _owned_node(foreign, foreign_id)

    with pytest.raises(Exception, match="not both owned by the bound agent"):
        await store.add_edge(source_id, foreign_id, "linked_to")
    with pytest.raises(Exception, match="not both owned by the bound agent"):
        await store.add_external_reference_edge(source_id, foreign_id, "linked_to")

    assert await _edge_row(db, source_id, foreign_id, "linked_to") is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_another_tenants_reservation_is_not_an_external_reference(db):
    """An id with an owner but no node yet is in the graph, not outside it."""
    store = AsyncGraphStore(db, agent_id=_agent())
    source_id = _nid("todo")
    reserved_id = _nid("reserved")
    await _owned_node(store, source_id)
    await graph_store_module.record_graph_node_owner(db, reserved_id, _agent())

    with pytest.raises(Exception, match="not both owned by the bound agent"):
        await store.add_external_reference_edge(source_id, reserved_id, "linked_to")

    assert await _edge_row(db, source_id, reserved_id, "linked_to") is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_ownerless_legacy_node_is_not_an_external_reference(db):
    """A node row with no ownership witness is still in the graph."""
    store = AsyncGraphStore(db, agent_id=_agent())
    source_id = _nid("todo")
    legacy_id = _nid("legacy")
    await _owned_node(store, source_id)
    await db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, ?, ?, ?)",
        (legacy_id, "concept", "Legacy", "{}"),
    )

    with pytest.raises(Exception, match="not both owned by the bound agent"):
        await store.add_external_reference_edge(source_id, legacy_id, "linked_to")

    assert await _edge_row(db, source_id, legacy_id, "linked_to") is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_source_owned_by_another_tenant_is_refused(db):
    store = AsyncGraphStore(db, agent_id=_agent())
    foreign = AsyncGraphStore(db, agent_id=_agent())
    foreign_source = _nid("todo")
    ref = _nid("url")
    await _owned_node(foreign, foreign_source)

    with pytest.raises(Exception, match="not both owned by the bound agent"):
        await store.add_external_reference_edge(foreign_source, ref, "linked_to")

    assert await _edge_row(db, foreign_source, ref, "linked_to") is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_absent_source_is_refused(db):
    store = AsyncGraphStore(db, agent_id=_agent())
    source_id = _nid("todo")
    ref = _nid("url")

    with pytest.raises(Exception, match="not both owned by the bound agent"):
        await store.add_external_reference_edge(source_id, ref, "linked_to")

    assert await _edge_row(db, source_id, ref, "linked_to") is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_external_reference_edge_requires_a_bound_store(db):
    with pytest.raises(ValueError, match="require a bound graph store"):
        await AsyncGraphStore(db).add_external_reference_edge(
            _nid("todo"), _nid("url"), "linked_to"
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_deleting_the_source_removes_the_external_reference_edge(db):
    """The edge has no target row to anchor it; it must not outlive its todo."""
    store = AsyncGraphStore(db, agent_id=_agent())
    todo_id = _nid("todo")
    ref = _nid("url")
    await _owned_node(store, todo_id)
    await store.add_external_reference_edge(todo_id, ref, "linked_to")

    await store.delete_node(todo_id)

    assert await _edge_row(db, todo_id, ref, "linked_to") is None
    assert await db.fetchone(
        "SELECT 1 FROM graph_edge_owners WHERE source_id = ?", (todo_id,)
    ) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_external_reference_preflights_foreign_target_before_lock(
    db, monkeypatch
):
    """A refused edge never queues behind another tenant's graph row."""
    if db.backend_type != "postgres":
        pytest.skip("PostgreSQL has per-row locks; SQLite has one writer slot")
    store = AsyncGraphStore(db, agent_id=_agent())
    foreign = AsyncGraphStore(db, agent_id=_agent())
    source_id = _nid("todo")
    foreign_id = _nid("todo")
    await _owned_node(store, source_id)
    await _owned_node(foreign, foreign_id)

    locked = []

    async def record_lock(*args, **kwargs):
        locked.append(args)
        return []

    monkeypatch.setattr(graph_store_module, "lock_graph_nodes_for_update", record_lock)

    with pytest.raises(Exception, match="not both owned by the bound agent"):
        await store.add_external_reference_edge(source_id, foreign_id, "linked_to")

    assert locked == []


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_external_reference_reserves_its_absent_target(
    db, monkeypatch
):
    """The target is locked with the source, so a writer creating a node with
    that id cannot slip in between the "outside the graph" check and the
    edge write."""
    if db.backend_type != "postgres":
        pytest.skip("PostgreSQL has per-row locks; SQLite has one writer slot")
    store = AsyncGraphStore(db, agent_id=_agent())
    source_id = _nid("todo")
    ref = _nid("url")
    await _owned_node(store, source_id)

    original_lock = graph_store_module.lock_graph_nodes_for_update
    locked = []

    async def observe_lock(database, node_ids, *, agent_id=""):
        locked.append(sorted(node_ids))
        return await original_lock(database, node_ids, agent_id=agent_id)

    monkeypatch.setattr(graph_store_module, "lock_graph_nodes_for_update", observe_lock)

    await store.add_external_reference_edge(source_id, ref, "linked_to")

    assert locked == [sorted([source_id, ref])]
