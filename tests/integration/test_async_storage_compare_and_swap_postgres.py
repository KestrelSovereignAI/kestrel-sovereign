"""compare_and_swap_node atomicity on the real production DB — Postgres (#2661).

The comprehensive CAS suite lives in ``tests/unit`` and runs against SQLite for
fast feedback; its Postgres parametrization *skips* in the unit CI job, which
provides neither Postgres nor ``DATABASE_URL`` (#2661 review P2). This module
re-exercises the load-bearing **atomicity** guarantees through the
``db_backend`` fixture from the integration job, where the pgvector service and
``DATABASE_URL`` are live — so the concurrent-writer race, the
predicate-failed-leaves-untouched safety property, and the properties-only
non-clobber contract are all proven on the actual Postgres backend, under its
row-lock concurrency (not just SQLite's per-connection write lock).

Runs on both backends via ``db_backend``; the SQLite pass is a cheap redundancy,
the Postgres pass is the point. Postgres is skipped only when the fixture finds
no ``DATABASE_URL`` (local runs without the service) — in CI's integration job
it always runs.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import (
    AsyncGraphStore,
    GraphNode,
    NodeSwapResult,
)


def _nid(prefix: str = "cas-pg") -> str:
    """Unique node id so the shared Postgres backend never collides."""
    return f"{prefix}:{uuid.uuid4().hex}"


def _node(node_id: str, properties: dict, *, label: str = "L", node_type: str = "cas_node") -> GraphNode:
    return GraphNode(node_id=node_id, node_type=node_type, label=label, properties=properties)


@pytest_asyncio.fixture
async def graph_store(db_backend):
    """AsyncGraphStore over the parametrized backend (SQLite + Postgres in CI)."""
    db = AsyncDatabase(db_backend)
    await db._init_schema()
    db._initialized = True
    return AsyncGraphStore(db)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("n_writers", [8, 16])
async def test_exactly_one_swap_wins_on_backend(graph_store, n_writers):
    """Fire N compare_and_swap_node calls in parallel on one node, all with the
    SAME last-read snapshot. Exactly one must win; the rest must report
    predicate_failed. On Postgres this exercises the row-lock path where
    concurrent swaps block and re-evaluate the predicate against the committed
    row — the guarantee a hand-rolled retry loop can never make."""
    nid = _nid("race")
    await graph_store.add_node(_node(nid, {"status": "pending", "v": 0}))
    snapshot = (await graph_store.get_node(nid)).properties

    async def swap(i: int) -> NodeSwapResult:
        return await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"status": "won", "winner": i})
        )

    results = await asyncio.gather(*(swap(i) for i in range(n_writers)))

    swapped = [r for r in results if r == NodeSwapResult.SWAPPED]
    failed = [r for r in results if r == NodeSwapResult.PREDICATE_FAILED]
    assert len(swapped) == 1, f"expected exactly one winner, got {results}"
    assert len(failed) == n_writers - 1
    winner_idx = results.index(NodeSwapResult.SWAPPED)
    after = await graph_store.get_node(nid)
    assert after.properties == {"status": "won", "winner": winner_idx}


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_concurrent_create_exactly_one_wins_on_backend(graph_store):
    """Compare-and-create under concurrency: N parallel creators of the same
    absent node, exactly one inserts (ON CONFLICT DO NOTHING on Postgres)."""
    nid = _nid("create-race")

    async def create(i: int) -> NodeSwapResult:
        return await graph_store.compare_and_swap_node(
            nid, None, _node(nid, {"creator": i})
        )

    results = await asyncio.gather(*(create(i) for i in range(8)))
    assert sum(1 for r in results if r == NodeSwapResult.SWAPPED) == 1
    assert sum(1 for r in results if r == NodeSwapResult.PREDICATE_FAILED) == 7


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_predicate_failed_leaves_post_read_update_untouched_on_backend(graph_store):
    """The core safety property on the production DB: a swap that loses the race
    must not clobber the winner's write."""
    nid = _nid()
    await graph_store.add_node(_node(nid, {"status": "pending"}))
    snapshot = (await graph_store.get_node(nid)).properties

    # A concurrent writer lands a real decision before the caller's swap fires.
    await graph_store.add_node(_node(nid, {"status": "failed", "audited": True}))

    result = await graph_store.compare_and_swap_node(
        nid, snapshot, _node(nid, {"status": "passed"})
    )
    assert result == NodeSwapResult.PREDICATE_FAILED
    after = await graph_store.get_node(nid)
    assert after.properties == {"status": "failed", "audited": True}


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_swap_does_not_clobber_concurrent_type_or_label_change_on_backend(graph_store):
    """P1 regression on Postgres (jsonb predicate + properties-only SET): a
    properties swap must not revert a concurrent ``label`` change. The relabel
    survives and our properties swap still lands."""
    nid = _nid()
    await graph_store.add_node(
        _node(nid, {"status": "pending"}, node_type="t", label="Before")
    )
    snapshot = (await graph_store.get_node(nid)).properties

    await graph_store.add_node(
        _node(nid, {"status": "pending"}, node_type="t", label="After")
    )

    result = await graph_store.compare_and_swap_node(
        nid, snapshot, _node(nid, {"status": "done"}, node_type="t", label="Ours")
    )
    assert result == NodeSwapResult.SWAPPED
    after = await graph_store.get_node(nid)
    assert after.label == "After"  # concurrent relabel survives; our label ignored
    assert after.properties == {"status": "done"}
