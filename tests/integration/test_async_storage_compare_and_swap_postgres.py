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
import hashlib
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

import kestrel_sovereign.storage.async_graph_store as graph_store_module
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_graph_store import (
    AsyncGraphStore,
    GraphNode,
    NodeSwapResult,
    release_graph_node_owners,
    reserve_provisional_agent_owner,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage


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
async def test_compare_and_create_waits_for_absent_node_reservation(graph_store):
    """CAS creation shares the lock protocol used by composed graph writers."""

    nid = _nid("create-reservation")
    reservation_acquired = asyncio.Event()
    release_reservation = asyncio.Event()

    async def reserve():
        async with graph_store.db.transaction():
            await graph_store.lock_nodes_for_update([nid])
            reservation_acquired.set()
            await release_reservation.wait()

    holder = asyncio.create_task(reserve())
    creator = None
    try:
        await asyncio.wait_for(reservation_acquired.wait(), timeout=5)
        creator = asyncio.create_task(
            graph_store.compare_and_swap_node(
                nid,
                None,
                _node(nid, {"creator": "cas"}),
            )
        )
        await asyncio.sleep(0.1)
        assert not creator.done(), (
            "compare-and-create bypassed the absent-node reservation"
        )
        release_reservation.set()
        assert await asyncio.wait_for(creator, timeout=5) == NodeSwapResult.SWAPPED
    finally:
        release_reservation.set()
        pending = [task for task in (holder, creator) if task is not None]
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


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


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_shared_owner_admission_waits_for_final_owner_delete_on_backend(
    graph_store, monkeypatch
):
    """Adding a shared witness locks the row against final-owner deletion.

    Without the PostgreSQL row lock, the joining tenant can validate the
    existing row, pause before recording its witness, and then record that
    witness after the final owner has deleted the physical row.  The add call
    reports success but leaves only a dangling owner record.
    """

    agent_a = f"agent:{uuid.uuid4().hex}"
    agent_b = f"agent:{uuid.uuid4().hex}"
    store_a = AsyncGraphStore(graph_store.db, agent_id=agent_a)
    store_b = AsyncGraphStore(graph_store.db, agent_id=agent_b)
    content = f"# shared constitution\n{uuid.uuid4().hex}\n".encode()
    node_id = await AsyncFileStore(graph_store.db, agent_a).store_file(
        content, "KESTREL_CONSTITUTION.md"
    )
    assert await AsyncFileStore(graph_store.db, agent_b).store_file(
        content, "KESTREL_CONSTITUTION.md"
    ) == node_id
    node = GraphNode(
        node_id=node_id,
        node_type="document",
        label="KESTREL_CONSTITUTION",
        properties={
            "hash": node_id,
            "type": "Constitution",
            "created_at": "2026-08-28T00:00:00+00:00",
        },
    )
    await store_a.add_node(node)

    delete_entered = asyncio.Event()
    release_delete = asyncio.Event()
    original_delete = store_a._delete_node_in_transaction

    async def pause_before_delete(candidate_id):
        delete_entered.set()
        await release_delete.wait()
        return await original_delete(candidate_id)

    monkeypatch.setattr(store_a, "_delete_node_in_transaction", pause_before_delete)

    owner_record_entered = asyncio.Event()
    release_owner_record = asyncio.Event()
    original_record_owner = graph_store_module.record_graph_node_owner

    async def pause_before_owner_record(db, candidate_id, candidate_agent):
        if candidate_id == node_id and candidate_agent == agent_b:
            owner_record_entered.set()
            await release_owner_record.wait()
        await original_record_owner(db, candidate_id, candidate_agent)

    monkeypatch.setattr(
        graph_store_module, "record_graph_node_owner", pause_before_owner_record
    )

    deletion = asyncio.create_task(
        store_a.compare_and_delete_node(
            node_id,
            expected_node_type="document",
            expected_label="KESTREL_CONSTITUTION",
        )
    )
    admission = None
    try:
        await asyncio.wait_for(delete_entered.wait(), timeout=5)
        admission = asyncio.create_task(store_b.add_node(node))
        await asyncio.sleep(0.1)
        assert not owner_record_entered.is_set(), (
            "shared-owner admission passed an in-flight final-owner delete"
        )

        release_delete.set()
        assert await asyncio.wait_for(deletion, timeout=5) == "deleted"
        await asyncio.wait_for(owner_record_entered.wait(), timeout=5)
        release_owner_record.set()
        await asyncio.wait_for(admission, timeout=5)
    finally:
        release_delete.set()
        release_owner_record.set()
        pending = [task for task in (deletion, admission) if task is not None]
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    stored = await store_b.get_node(node_id)
    assert stored is not None
    assert stored.label == "KESTREL_CONSTITUTION"
    owners = await graph_store.db.fetchall(
        "SELECT agent_id FROM graph_node_owners WHERE node_id = ?", (node_id,)
    )
    assert {row[0] for row in owners} == {agent_b}


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_edge_admission_waits_for_endpoint_delete_on_backend(
    graph_store, monkeypatch
):
    """An ordinary edge cannot be admitted from a soon-deleted endpoint."""

    agent_id = f"agent:{uuid.uuid4().hex}"
    bound = AsyncGraphStore(graph_store.db, agent_id=agent_id)
    source_id = _nid("edge-source")
    target_id = _nid("edge-target")
    await bound.add_node(
        _node(
            source_id,
            {"agent_id": agent_id},
            node_type="owned",
            label="Source",
        )
    )
    await bound.add_node(
        _node(
            target_id,
            {"agent_id": agent_id},
            node_type="owned",
            label="Target",
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

    edge_owner_entered = asyncio.Event()
    release_edge_owner = asyncio.Event()
    original_record_edge = graph_store_module.record_graph_edge_owner

    async def pause_before_edge_owner(
        db, candidate_source, candidate_target, candidate_label, candidate_agent
    ):
        if (
            candidate_source == source_id
            and candidate_target == target_id
            and candidate_agent == agent_id
        ):
            edge_owner_entered.set()
            await release_edge_owner.wait()
        await original_record_edge(
            db,
            candidate_source,
            candidate_target,
            candidate_label,
            candidate_agent,
        )

    monkeypatch.setattr(
        graph_store_module, "record_graph_edge_owner", pause_before_edge_owner
    )

    deletion = asyncio.create_task(
        bound.compare_and_delete_node(
            source_id,
            expected_node_type="owned",
            expected_label="Source",
        )
    )
    admission = None
    try:
        await asyncio.wait_for(delete_entered.wait(), timeout=5)
        admission = asyncio.create_task(
            bound.add_edge(source_id, target_id, "references")
        )
        await asyncio.sleep(0.1)
        assert not edge_owner_entered.is_set(), (
            "edge admission passed an in-flight endpoint delete"
        )

        release_delete.set()
        assert await asyncio.wait_for(deletion, timeout=5) == "deleted"
        with pytest.raises(Exception, match="endpoints"):
            await asyncio.wait_for(admission, timeout=5)
    finally:
        release_delete.set()
        release_edge_owner.set()
        pending = [task for task in (deletion, admission) if task is not None]
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    assert await graph_store.db.fetchone(
        "SELECT 1 FROM graph_edges "
        "WHERE source_id = ? AND target_id = ? AND label = ?",
        (source_id, target_id, "references"),
    ) is None
    assert await graph_store.db.fetchone(
        "SELECT 1 FROM graph_edge_owners "
        "WHERE source_id = ? AND target_id = ? AND label = ?",
        (source_id, target_id, "references"),
    ) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_bound_edge_preflights_foreign_target_before_lock(
    graph_store, monkeypatch
):
    """A rejected tenant edge never queues behind a foreign graph row."""

    if graph_store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL has per-row locks; SQLite has one writer slot")
    owner = f"agent:{uuid.uuid4().hex}"
    foreign_owner = f"agent:{uuid.uuid4().hex}"
    source_id = _nid("owned-edge-source")
    foreign_id = _nid("foreign-edge-target")
    bound = AsyncGraphStore(graph_store.db, agent_id=owner)
    foreign = AsyncGraphStore(graph_store.db, agent_id=foreign_owner)
    await bound.add_node(
        _node(source_id, {"agent_id": owner}, node_type="owned")
    )
    await foreign.add_node(
        _node(foreign_id, {"agent_id": foreign_owner}, node_type="owned")
    )

    lock = AsyncMock(return_value=[source_id, foreign_id])
    monkeypatch.setattr(graph_store_module, "lock_graph_nodes_for_update", lock)

    with pytest.raises(Exception, match="endpoints"):
        await bound.add_edge(source_id, foreign_id, "references")

    lock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_bound_add_preflights_foreign_ordinary_node_before_lock(
    graph_store, monkeypatch
):
    """A known-invalid tenant node write never queues on a foreign row."""

    if graph_store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL has per-row locks; SQLite has one writer slot")
    owner = f"agent:{uuid.uuid4().hex}"
    foreign_owner = f"agent:{uuid.uuid4().hex}"
    foreign_id = _nid("foreign-ordinary-node")
    bound = AsyncGraphStore(graph_store.db, agent_id=owner)
    foreign = AsyncGraphStore(graph_store.db, agent_id=foreign_owner)
    await foreign.add_node(
        _node(foreign_id, {"agent_id": foreign_owner}, node_type="owned")
    )

    lock = AsyncMock(return_value=[foreign_id])
    monkeypatch.setattr(graph_store_module, "lock_graph_nodes_for_update", lock)

    with pytest.raises(Exception, match="owned by another agent"):
        await bound.add_node(
            _node(foreign_id, {"agent_id": owner}, node_type="owned")
        )

    lock.assert_not_awaited()
    assert (await foreign.get_node(foreign_id)).properties == {
        "agent_id": foreign_owner
    }


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_bound_prelock_refuses_foreign_ordinary_node_before_lock(
    graph_store, monkeypatch
):
    """The public composed-write surface cannot lock another tenant's row."""

    if graph_store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL has per-row locks; SQLite has one writer slot")
    owner = f"agent:{uuid.uuid4().hex}"
    foreign_owner = f"agent:{uuid.uuid4().hex}"
    foreign_id = _nid("foreign-prelock-node")
    bound = AsyncGraphStore(graph_store.db, agent_id=owner)
    foreign = AsyncGraphStore(graph_store.db, agent_id=foreign_owner)
    await foreign.add_node(
        _node(foreign_id, {"agent_id": foreign_owner}, node_type="owned")
    )

    lock = AsyncMock(return_value=[foreign_id])
    monkeypatch.setattr(graph_store_module, "lock_graph_nodes_for_update", lock)

    with pytest.raises(ValueError, match="outside the bound agent"):
        await bound.lock_nodes_for_update([foreign_id])

    lock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_trusted_edge_locks_only_owned_source(
    graph_store, monkeypatch
):
    """Trusted lineage never locks the deliberately foreign target row."""

    if graph_store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL has per-row locks; SQLite has one writer slot")
    owner = f"agent:{uuid.uuid4().hex}"
    foreign_owner = f"agent:{uuid.uuid4().hex}"
    source_id = _nid("trusted-edge-source")
    foreign_id = _nid("trusted-edge-target")
    bound = AsyncGraphStore(graph_store.db, agent_id=owner)
    foreign = AsyncGraphStore(graph_store.db, agent_id=foreign_owner)
    await bound.add_node(
        _node(source_id, {"agent_id": owner}, node_type="owned")
    )
    await foreign.add_node(
        _node(foreign_id, {"agent_id": foreign_owner}, node_type="owned")
    )

    original_lock = graph_store_module.lock_graph_nodes_for_update
    locked = []

    async def observe_lock(db, node_ids, *, agent_id=""):
        locked.append(list(node_ids))
        return await original_lock(db, node_ids, agent_id=agent_id)

    monkeypatch.setattr(
        graph_store_module,
        "lock_graph_nodes_for_update",
        observe_lock,
    )

    await bound.add_trusted_cross_agent_edge(
        source_id,
        foreign_id,
        "spawned_by",
    )

    assert locked == [[source_id]]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_complete_write_set_locks_absent_node_ids_on_backend(graph_store):
    """Canonical prelocking serializes IDs before either row is inserted."""

    first_id = _nid("absent-lock-a")
    second_id = _nid("absent-lock-b")
    first_acquired = asyncio.Event()
    release_first = asyncio.Event()
    second_acquired = asyncio.Event()

    async def first_writer():
        async with graph_store.db.transaction():
            await graph_store.lock_nodes_for_update([second_id, first_id])
            first_acquired.set()
            await release_first.wait()

    async def second_writer():
        async with graph_store.db.transaction():
            await graph_store.lock_nodes_for_update([first_id, second_id])
            second_acquired.set()

    first = asyncio.create_task(first_writer())
    second = None
    try:
        await asyncio.wait_for(first_acquired.wait(), timeout=5)
        second = asyncio.create_task(second_writer())
        await asyncio.sleep(0.1)
        assert not second_acquired.is_set(), (
            "a complete write-set prelock did not cover absent graph IDs"
        )
        release_first.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
        assert second_acquired.is_set()
    finally:
        release_first.set()
        pending = [task for task in (first, second) if task is not None]
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_bulk_existing_write_set_uses_bounded_advisory_shards(
    graph_store,
):
    """Bulk locking uses a fixed reservation-shard budget, not one lock per row."""

    if graph_store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL exposes transaction advisory locks")

    prefix = _nid("bulk-existing-lock") + ":"
    count = 1500
    node_ids = [f"{prefix}{index}" for index in range(count)]
    await graph_store.db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "SELECT ? || series::text, 'owned', 'Bulk', '{}' "
        "FROM generate_series(0, ?) AS series",
        (prefix, count - 1),
    )

    try:
        async with graph_store.db.transaction():
            await graph_store.lock_nodes_for_update(node_ids)
            held = await graph_store.db.fetchone(
                "SELECT COUNT(*) FROM pg_locks "
                "WHERE pid = pg_backend_pid() AND locktype = 'advisory'"
            )
            assert 0 < held[0] <= 128
    finally:
        await graph_store.db.execute(
            "DELETE FROM graph_nodes WHERE node_id LIKE ?",
            (f"{prefix}%",),
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_large_absent_write_set_uses_bounded_advisory_shards(
    graph_store,
):
    """A valid large replication fits within a bounded shared-lock footprint."""

    if graph_store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL exposes transaction advisory locks")

    absent_ids = [_nid(f"absent-shard-{index}") for index in range(1500)]
    async with graph_store.db.transaction():
        assert await graph_store.lock_nodes_for_update(absent_ids) == sorted(
            absent_ids
        )
        held = await graph_store.db.fetchone(
            "SELECT COUNT(*) FROM pg_locks "
            "WHERE pid = pg_backend_pid() AND locktype = 'advisory'"
        )
        assert 0 < held[0] <= 128


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_nested_single_node_writes_share_bounded_advisory_shards(
    graph_store,
):
    """A complete prelock lets nested writes reuse a bounded shard set."""

    if graph_store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL exposes transaction advisory locks")

    prefix = _nid("nested-advisory-cap") + ":"
    node_ids = [f"{prefix}{index}" for index in range(150)]
    try:
        async with graph_store.db.transaction():
            await graph_store.lock_nodes_for_update(node_ids)
            for index, node_id in enumerate(node_ids):
                await graph_store.add_node(
                    _node(node_id, {"index": index})
                )
            held = await graph_store.db.fetchone(
                "SELECT COUNT(*) FROM pg_locks "
                "WHERE pid = pg_backend_pid() AND locktype = 'advisory'"
            )
            assert 0 < held[0] <= 128
    finally:
        await graph_store.db.execute(
            "DELETE FROM graph_nodes WHERE node_id LIKE ?",
            (f"{prefix}%",),
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_reverse_incremental_shard_order_is_rejected_before_wait(
    graph_store,
):
    """Disjoint nested writes cannot turn advisory collisions into deadlock."""

    if graph_store.db.backend_type != "postgres":
        pytest.skip("PostgreSQL exposes transaction advisory locks")

    by_key = {}
    for index in range(10000):
        node_id = _nid(f"shard-order-{index}")
        shard = int.from_bytes(
            hashlib.sha256(
                f"kestrel:graph-node:{node_id}".encode("utf-8")
            ).digest()[:8],
            "big",
        ) % 128
        key = int.from_bytes(
            hashlib.sha256(
                f"kestrel:graph-node-reservation-shard:{shard}".encode(
                    "utf-8"
                )
            ).digest()[:8],
            "big",
            signed=True,
        )
        by_key.setdefault(key, []).append(node_id)
        complete = [
            (candidate, ids)
            for candidate, ids in by_key.items()
            if len(ids) >= 2
        ]
        if len(complete) >= 2:
            break

    ordered = sorted(complete, key=lambda item: item[0])
    low, high = ordered[0], ordered[-1]
    low_ids = low[1][:2]
    high_ids = high[1][:2]
    low_ready = asyncio.Event()
    high_ready = asyncio.Event()

    async def high_then_low():
        async with graph_store.db.transaction():
            await graph_store.lock_nodes_for_update([high_ids[0]])
            high_ready.set()
            await low_ready.wait()
            try:
                await graph_store.lock_nodes_for_update([low_ids[0]])
            except ValueError as exc:
                assert "complete graph write set" in str(exc)
                return "rejected"
            return "acquired"

    async def low_then_high():
        async with graph_store.db.transaction():
            await graph_store.lock_nodes_for_update([low_ids[1]])
            low_ready.set()
            await high_ready.wait()
            await graph_store.lock_nodes_for_update([high_ids[1]])
            return "acquired"

    results = await asyncio.wait_for(
        asyncio.gather(high_then_low(), low_then_high()), timeout=5
    )
    assert results == ["rejected", "acquired"]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_provisional_owner_adopts_valid_ownerless_agent_root(graph_store):
    """Legacy agent roots reach validation before their self-owner is repaired."""

    agent_id = _nid("ownerless-agent-root")
    await graph_store.db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'agent', 'Legacy agent', ?)",
        (agent_id, '{"agent_id": "' + agent_id + '"}'),
    )

    try:
        async with graph_store.db.transaction():
            await reserve_provisional_agent_owner(graph_store.db, agent_id)

        assert await graph_store.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
            (agent_id,),
        ) == [(agent_id,)]
    finally:
        await graph_store.db.execute(
            "DELETE FROM graph_node_owners WHERE node_id = ?", (agent_id,)
        )
        await graph_store.db.execute(
            "DELETE FROM graph_nodes WHERE node_id = ?", (agent_id,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("operation", ["avatar", "backup"])
async def test_bootstrap_writer_adopts_valid_ownerless_agent_root(
    db_backend, operation
):
    """Complete bootstrap callers can repair a legacy root before graph writes."""

    agent_id = _nid(f"ownerless-{operation}-root")
    payload = f"{operation}:{uuid.uuid4().hex}".encode()
    content_hash = hashlib.sha256(payload).hexdigest()
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()
    await storage.db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'agent', 'Legacy agent', ?)",
        (agent_id, '{"agent_id": "' + agent_id + '"}'),
    )

    class BackupResult:
        storage_tier = type("Tier", (), {"value": "local"})()
        ipfs_cid = None
        filecoin_deal_id = None
        encrypted = False
        encryption_key_hash = None

        def __init__(self, node_id):
            self.content_hash = node_id

    try:
        if operation == "avatar":
            await storage.files.store_avatar(payload, agent_id, "primary")
            artifact_node_id = storage.files._avatar_node_id(
                agent_id, "primary", content_hash
            )
        else:
            await storage.record_backup_artifact(
                agent_id, BackupResult(content_hash)
            )
            artifact_node_id = content_hash

        assert await storage.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
            (agent_id,),
        ) == [(agent_id,)]
    finally:
        await storage.db.execute(
            "DELETE FROM graph_edge_owners WHERE source_id = ?", (agent_id,)
        )
        await storage.db.execute(
            "DELETE FROM graph_edges WHERE source_id = ?", (agent_id,)
        )
        await storage.db.execute(
            "DELETE FROM graph_node_owners WHERE node_id = ? OR agent_id = ?",
            (agent_id, agent_id),
        )
        await storage.db.execute(
            "DELETE FROM graph_nodes WHERE node_id = ? OR node_id = ?",
            (agent_id, artifact_node_id),
        )
        await storage.db.execute(
            "DELETE FROM file_owners WHERE content_hash = ?", (content_hash,)
        )
        await storage.db.execute(
            "DELETE FROM files WHERE content_hash = ?", (content_hash,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_provisional_owner_refuses_valid_root_owned_by_another_agent(
    graph_store
):
    """The legacy-root allowance never adopts a foreign ownership witness."""

    agent_id = _nid("foreign-owned-agent-root")
    foreign_owner = _nid("foreign-root-owner")
    additional_id = _nid("foreign-root-additional")
    await graph_store.db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'agent', 'Foreign-owned agent', ?)",
        (agent_id, '{"agent_id": "' + agent_id + '"}'),
    )
    await graph_store.db.execute(
        "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
        (agent_id, foreign_owner),
    )

    try:
        with pytest.raises(Exception, match="owned|outside the bound agent"):
            async with graph_store.db.transaction():
                await reserve_provisional_agent_owner(
                    graph_store.db,
                    agent_id,
                    additional_graph_node_ids=[additional_id],
                )

        assert await graph_store.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
            (agent_id,),
        ) == [(foreign_owner,)]
        assert await graph_store.db.fetchone(
            "SELECT 1 FROM graph_nodes WHERE node_id = ?", (additional_id,)
        ) is None
    finally:
        await graph_store.db.execute(
            "DELETE FROM graph_node_owners WHERE node_id = ?", (agent_id,)
        )
        await graph_store.db.execute(
            "DELETE FROM graph_nodes WHERE node_id = ?", (agent_id,)
        )
