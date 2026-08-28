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
    release_graph_node_owners,
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


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_expected_identity_refuses_post_read_relabel_on_backend(graph_store):
    """The wider predicate must be one SQL write predicate on PostgreSQL too."""
    nid = _nid("identity-label")
    await graph_store.add_node(
        _node(nid, {"status": "pending"}, node_type="owned", label="Before")
    )
    snapshot = (await graph_store.get_node(nid)).properties
    await graph_store.add_node(
        _node(nid, dict(snapshot), node_type="owned", label="After")
    )

    result = await graph_store.compare_and_swap_node(
        nid,
        snapshot,
        _node(nid, {"status": "done"}, node_type="owned", label="Before"),
        expected_node_type="owned",
        expected_label="Before",
    )

    assert result == NodeSwapResult.PREDICATE_FAILED
    after = await graph_store.get_node(nid)
    assert after.label == "After"
    assert after.properties == {"status": "pending"}


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_compare_and_delete_preserves_post_read_replacement_on_backend(
    graph_store,
):
    """The identity check and delete share one serialized backend transaction."""
    nid = _nid("delete-identity")
    await graph_store.add_node(
        _node(nid, {"status": "stale"}, node_type="owned", label="Before")
    )
    observed = await graph_store.get_node(nid)
    assert observed.label == "Before"
    await graph_store.add_node(
        _node(
            nid,
            {"status": "replacement", "sentinel": "must survive"},
            node_type="owned",
            label="After",
        )
    )

    result = await graph_store.compare_and_delete_node(
        nid, expected_node_type="owned", expected_label="Before"
    )

    assert result == "predicate_failed"
    after = await graph_store.get_node(nid)
    assert after is not None
    assert after.label == "After"
    assert after.properties["sentinel"] == "must survive"


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_compare_and_delete_locks_identity_until_delete_commits_on_backend(
    graph_store, monkeypatch
):
    """A writer arriving after the identity read must wait, then recreate.

    Pausing immediately before the delete makes the transaction boundary
    observable. Without PostgreSQL's row lock, the replacement lands first and
    is then deleted; with the lock, it lands only after the original deletion
    commits and therefore survives.
    """
    nid = _nid("delete-lock")
    await graph_store.add_node(
        _node(nid, {"status": "stale"}, node_type="owned", label="Before")
    )
    delete_entered = asyncio.Event()
    release_delete = asyncio.Event()
    original_delete = graph_store._delete_node_in_transaction

    async def pause_before_delete(candidate_id):
        delete_entered.set()
        await release_delete.wait()
        return await original_delete(candidate_id)

    monkeypatch.setattr(
        graph_store, "_delete_node_in_transaction", pause_before_delete
    )
    deletion = asyncio.create_task(
        graph_store.compare_and_delete_node(
            nid,
            expected_node_type="owned",
            expected_label="Before",
        )
    )
    await asyncio.wait_for(delete_entered.wait(), timeout=5)
    replacement = asyncio.create_task(
        graph_store.add_node(
            _node(
                nid,
                {"status": "replacement", "sentinel": "must survive"},
                node_type="owned",
                label="After",
            )
        )
    )
    await asyncio.sleep(0.1)
    replacement_waited = not replacement.done()
    release_delete.set()
    try:
        result, _ = await asyncio.gather(deletion, replacement)
    finally:
        release_delete.set()
        await asyncio.gather(deletion, replacement, return_exceptions=True)

    assert replacement_waited, "replacement passed an in-flight conditional delete"
    assert result == "deleted"
    after = await graph_store.get_node(nid)
    assert after is not None
    assert after.label == "After"
    assert after.properties["sentinel"] == "must survive"


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_conditional_and_ordinary_bound_deletes_share_lock_order_on_backend(
    graph_store, monkeypatch
):
    """Both delete doors lock graph identity before releasing ownership.

    If ordinary deletion takes the owner row first while conditional deletion
    holds the graph row, PostgreSQL deadlocks when each path reaches the other
    row. Completing both operations under a short bound proves the shared lock
    order rather than relying on the database deadlock detector.
    """
    agent_id = f"agent:{uuid.uuid4().hex}"
    bound = AsyncGraphStore(graph_store.db, agent_id=agent_id)
    nid = _nid("delete-order")
    await bound.add_node(
        _node(
            nid,
            {"agent_id": agent_id, "status": "stale"},
            node_type="owned",
            label="Before",
        )
    )
    delete_entered = asyncio.Event()
    release_delete = asyncio.Event()
    original_delete = bound._delete_node_in_transaction

    async def pause_before_delete(candidate_id):
        delete_entered.set()
        await release_delete.wait()
        return await original_delete(candidate_id)

    monkeypatch.setattr(bound, "_delete_node_in_transaction", pause_before_delete)
    conditional = asyncio.create_task(
        bound.compare_and_delete_node(
            nid,
            expected_node_type="owned",
            expected_label="Before",
        )
    )
    await asyncio.wait_for(delete_entered.wait(), timeout=5)
    ordinary = asyncio.create_task(bound.delete_node(nid))
    await asyncio.sleep(0.1)
    ordinary_waited = not ordinary.done()
    release_delete.set()
    try:
        conditional_result, _ = await asyncio.wait_for(
            asyncio.gather(conditional, ordinary), timeout=5
        )
    finally:
        release_delete.set()
        if not conditional.done():
            conditional.cancel()
        if not ordinary.done():
            ordinary.cancel()
        await asyncio.gather(conditional, ordinary, return_exceptions=True)

    assert ordinary_waited, "ordinary deletion passed the held graph-row lock"
    assert conditional_result == "deleted"
    assert await bound.get_node(nid) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_owner_release_and_conditional_delete_share_lock_order_on_backend(
    graph_store, monkeypatch
):
    """The reusable ownership-release helper follows graph-before-owner order."""
    agent_id = f"agent:{uuid.uuid4().hex}"
    bound = AsyncGraphStore(graph_store.db, agent_id=agent_id)
    nid = _nid("release-order")
    await bound.add_node(
        _node(
            nid,
            {"agent_id": agent_id, "status": "stale"},
            node_type="owned",
            label="Before",
        )
    )
    delete_entered = asyncio.Event()
    release_delete = asyncio.Event()
    original_delete = bound._delete_node_in_transaction

    async def pause_before_delete(candidate_id):
        delete_entered.set()
        await release_delete.wait()
        return await original_delete(candidate_id)

    monkeypatch.setattr(bound, "_delete_node_in_transaction", pause_before_delete)
    conditional = asyncio.create_task(
        bound.compare_and_delete_node(
            nid,
            expected_node_type="owned",
            expected_label="Before",
        )
    )
    await asyncio.wait_for(delete_entered.wait(), timeout=5)

    async def release_owner():
        async with graph_store.db.transaction(immediate=True):
            return await release_graph_node_owners(
                graph_store.db, [nid], agent_id
            )

    release = asyncio.create_task(release_owner())
    await asyncio.sleep(0.1)
    release_waited = not release.done()
    release_delete.set()
    try:
        conditional_result, _ = await asyncio.wait_for(
            asyncio.gather(conditional, release), timeout=5
        )
    finally:
        release_delete.set()
        if not conditional.done():
            conditional.cancel()
        if not release.done():
            release.cancel()
        await asyncio.gather(conditional, release, return_exceptions=True)

    assert release_waited, "ownership release passed the held graph-row lock"
    assert conditional_result == "deleted"
    assert await bound.get_node(nid) is None
