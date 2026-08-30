"""Inception must record an agent atomically with its governing edge (#2867).

Inception writes the constitution node, the agent node and the ``governed_by``
edge that binds the agent to the constitution it is governed by. Historically
each was an *independently* atomic write with no atomicity across them, so a
crash or cancellation between them could leave an agent node recorded as
**existing but not governed** — and because the node is present, every later
boot treats inception as done and never repairs the missing edge (the same
durable-record-without-its-guarantee class as #2774 / #2804).

The invariant these tests guard: *an agent must never be recorded as existing
without its governing edge — either all three writes commit or none do.*

Mutation note (per the issue's mutation requirement): a test that only asserts
the happy-path end state (all three rows present) passes even without the
transaction wrapper, because the three writes succeed individually when nothing
interrupts them. The load-bearing test injects a failure between the agent node
and the edge and asserts **no** agent node survives. Reverting the
``async with db.transaction():`` wrapper in ``create_kestrel_identity_async``
makes ``test_failure_between_agent_node_and_edge_leaves_no_agent_node`` fail
(the independently-committed agent node survives the rollback of the edge), and
makes ``test_rag_and_embedding_work_stays_outside_the_transaction_span`` fail
if the span is widened to cover step 8's embedding work. Both mutants killed.
"""

import contextlib

import pytest

from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore


@pytest.fixture
async def external_db(tmp_path):
    """A caller-owned SQLite database, passed to inception as ``database=``.

    Inception does not close an externally provided database (that path is the
    multi-tenant PostgreSQL contract), so the same handle stays usable for
    assertions *after* an injected mid-inception failure — no re-open race
    against a connection the function left open.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "external_prime.db"))
    try:
        yield db
    finally:
        await db.close()


async def _incept(db, tmp_path, **kwargs):
    return await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        constitution_path=None,  # packaged authoritative governing source
        database=db,
        is_test_instance=True,
        agent_name=kwargs.pop("agent_name", "AtomicBird"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_constitution_agent_and_edge_all_commit(external_db, tmp_path):
    """Happy path: all three identity rows are durable after inception.

    This is the end-state check the issue warns is NOT sufficient on its own —
    it passes with or without the transaction wrapper. It exists to prove the
    atomic commit does not *lose* any of the three writes on the success path.
    """
    creds = await _incept(external_db, tmp_path)

    agent_rows = await external_db.fetchall(
        "SELECT node_id FROM graph_nodes WHERE node_type = 'agent'"
    )
    assert [row[0] for row in agent_rows] == [creds.agent_did]

    const_rows = await external_db.fetchall(
        "SELECT node_id FROM graph_nodes WHERE label = 'KESTREL_CONSTITUTION'"
    )
    assert len(const_rows) == 1
    constitution_id = const_rows[0][0]

    edge_rows = await external_db.fetchall(
        "SELECT source_id, target_id FROM graph_edges WHERE label = 'governed_by'"
    )
    assert edge_rows == [(creds.agent_did, constitution_id)]


@pytest.mark.asyncio
async def test_inception_prelocks_complete_identity_graph_write_set(
    external_db, tmp_path, monkeypatch
):
    """Inception joins replication's canonical multi-node lock order."""
    events = []
    original_lock = AsyncGraphStore.lock_nodes_for_update
    original_add_node = AsyncGraphStore.add_node

    async def observe_lock(self, node_ids):
        materialized = tuple(node_ids)
        events.append(("lock", materialized))
        return await original_lock(self, materialized)

    async def observe_add_node(self, node):
        events.append(("add_node", node.node_id))
        return await original_add_node(self, node)

    monkeypatch.setattr(
        AsyncGraphStore, "lock_nodes_for_update", observe_lock
    )
    monkeypatch.setattr(AsyncGraphStore, "add_node", observe_add_node)

    creds = await _incept(external_db, tmp_path)
    constitution = await external_db.fetchone(
        "SELECT node_id FROM graph_nodes WHERE label = 'KESTREL_CONSTITUTION'"
    )

    assert events[0][0] == "lock"
    assert set(events[0][1]) == {creds.agent_did, constitution[0]}
    assert [event[0] for event in events].count("lock") == 1


@pytest.mark.asyncio
async def test_failure_between_agent_node_and_edge_leaves_no_agent_node(
    external_db, tmp_path, monkeypatch
):
    """A crash between the agent node and its governing edge must leave NO agent
    node behind — not an agent node without its edge.

    Failure is injected at the ``governed_by`` ``add_edge`` call, which runs
    immediately after ``add_node(agent_node)``. With the writes wrapped in one
    transaction the whole unit rolls back, so no agent (or constitution) node
    survives. WITHOUT the wrapper the agent node — committed by its own
    ``add_node`` transaction — would survive the edge failure, and this test
    would find it. That is the mutant this test kills.
    """
    original_add_edge = AsyncGraphStore.add_edge

    async def failing_add_edge(self, source_id, target_id, label, properties=None):
        if label == "governed_by":
            raise RuntimeError(
                "injected failure between agent node and governing edge"
            )
        return await original_add_edge(self, source_id, target_id, label, properties)

    monkeypatch.setattr(AsyncGraphStore, "add_edge", failing_add_edge)

    with pytest.raises(Exception) as exc_info:
        await _incept(external_db, tmp_path)
    assert "injected failure between agent node and governing edge" in str(
        exc_info.value
    )

    # The load-bearing assertion: the agent node did NOT survive the rollback.
    agent_rows = await external_db.fetchall(
        "SELECT node_id FROM graph_nodes WHERE node_type = 'agent'"
    )
    assert agent_rows == [], (
        "an agent node was recorded without its governed_by edge — inception is "
        "not atomic across the three identity writes"
    )

    # The whole unit rolled back: neither the constitution node nor the edge
    # persisted either.
    const_rows = await external_db.fetchall(
        "SELECT node_id FROM graph_nodes WHERE label = 'KESTREL_CONSTITUTION'"
    )
    assert const_rows == []
    edge_rows = await external_db.fetchall(
        "SELECT source_id FROM graph_edges WHERE label = 'governed_by'"
    )
    assert edge_rows == []


@pytest.mark.asyncio
async def test_rag_and_embedding_work_stays_outside_the_transaction_span(
    external_db, tmp_path, monkeypatch
):
    """Structural guard: no RAG / embedding work runs inside the identity
    transaction span (acceptance criterion 3).

    On SQLite ``transaction()`` holds the connection write lock for the whole
    ``BEGIN..COMMIT`` span (#1675). Step 8 indexes the constitution for RAG with
    ``compute_embeddings=True`` — a provider round-trip or local model
    inference. If a later edit widened the identity transaction to cover that
    work, every other writer on the database would block behind it.

    We instrument ``db.transaction`` with a depth counter and record the depth
    at the entry of ``chunk_document`` (before it opens its own transaction to
    write chunks). Correctly bounded, the identity transaction is already closed
    by then, so the depth is 0. Widening the span makes the depth >= 1 here.
    """
    depth = {"current": 0, "depth_at_chunk_entry": [], "chunk_calls": 0}
    original_transaction = external_db.transaction

    @contextlib.asynccontextmanager
    async def counting_transaction(*args, **kwargs):
        depth["current"] += 1
        try:
            async with original_transaction(*args, **kwargs):
                yield
        finally:
            depth["current"] -= 1

    # Instance-level shadow: the graph store, file store and RAG store all hold
    # this same db instance, so every self.db.transaction() call is counted.
    monkeypatch.setattr(external_db, "transaction", counting_transaction)

    original_chunk = AsyncRAGStore.chunk_document

    async def spy_chunk_document(self, *args, **kwargs):
        # Record the OUTER transaction depth before chunk_document opens its own.
        depth["chunk_calls"] += 1
        depth["depth_at_chunk_entry"].append(depth["current"])
        return await original_chunk(self, *args, **kwargs)

    monkeypatch.setattr(AsyncRAGStore, "chunk_document", spy_chunk_document)

    await _incept(external_db, tmp_path)

    assert depth["chunk_calls"] >= 1, (
        "RAG indexing never ran — the structural guard is vacuous; inception "
        "must reach step 8 for this test to be meaningful"
    )
    assert all(d == 0 for d in depth["depth_at_chunk_entry"]), (
        "RAG / embedding work ran inside the identity transaction span "
        f"(observed transaction depths at chunk_document entry: "
        f"{depth['depth_at_chunk_entry']}) — the span was widened past the "
        "three identity writes (#2867)"
    )
