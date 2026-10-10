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
import asyncio

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
async def test_inception_refuses_corrupt_actual_conflict_winner(external_db, tmp_path, monkeypatch):
    """A generic store's returned input hash is not proof of persisted bytes."""
    import hashlib
    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    content = resolve_governing_constitution_bytes(None)
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    digest = hashlib.sha256(content).hexdigest()
    await external_db.execute_commit(
        "INSERT INTO files (content_hash,original_name,content) VALUES (?,?,?)",
        (digest, "preexisting corruption", b"not the governing bytes"),
    )
    await external_db.execute_commit(
        "INSERT INTO file_owners (content_hash,agent_id,original_name) VALUES (?,?,?)",
        (digest, "did:test:prior-file-owner", "preexisting corruption"),
    )
    with pytest.raises(Exception, match="do not verify"):
        await _incept(external_db, tmp_path, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug="retry-atomic")
    assert not list(tmp_path.glob("*_*.pem"))
    assert not list(tmp_path.glob("*_*.json"))
    assert not list(tmp_path.glob("*_*.key.enc"))
    assert (
        await external_db.fetchall(
            "SELECT node_id FROM graph_nodes WHERE node_type='agent'"
        )
        == []
    )
    assert (
        await external_db.fetchall(
            "SELECT source_id FROM graph_edges WHERE label='governed_by'"
        )
        == []
    )
    assert await external_db.fetchall(
        "SELECT agent_id FROM file_owners WHERE content_hash=?", (digest,)
    ) == [("did:test:prior-file-owner",)]
    assert (
        await external_db.fetchone(
            "SELECT content FROM files WHERE content_hash=?", (digest,)
        )
    )[0] == b"not the governing bytes"
    await external_db.execute_commit("UPDATE files SET content=? WHERE content_hash=?", (content, digest))
    # The exact explicit born-hybrid slug is reusable after pre-commit failure.
    credentials = await _incept(external_db, tmp_path, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug="retry-atomic")
    assert credentials.agent_did.endswith(":retry-atomic")


@pytest.mark.asyncio
async def test_owned_database_publication_failure_closes_and_cleans_before_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    native_sqlite = AsyncDatabase.sqlite
    opened = []

    async def database_with_native_refusal(path, **kwargs):
        db = await native_sqlite(path, **kwargs)
        opened.append(db)
        await db.execute_script("CREATE TRIGGER refuse_publication BEFORE INSERT ON files BEGIN SELECT RAISE(ABORT, 'native publication refusal'); END;")
        return db

    monkeypatch.setattr(AsyncDatabase, "sqlite", database_with_native_refusal)
    try:
        with pytest.raises(Exception, match="native publication refusal"):
            await create_kestrel_identity_async(output_dir=str(tmp_path), is_test_instance=True, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug="retry-owned")
        assert opened[0]._backend._connection is None
        assert not (tmp_path / "kestrel_prime.db").exists()
        assert not list(tmp_path.glob("retry-owned_*"))
        monkeypatch.setattr(AsyncDatabase, "sqlite", native_sqlite)
        credentials = await create_kestrel_identity_async(output_dir=str(tmp_path), is_test_instance=True, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug="retry-owned")
        assert credentials.agent_did.endswith(":retry-owned")
    finally:
        # Counterproofs must not leave old-code failures' worker threads alive.
        for db in opened:
            await db.close()


@pytest.mark.asyncio
async def test_cancelled_native_publication_cleans_owned_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    native_sqlite = AsyncDatabase.sqlite
    native_add = AsyncGraphStore.add_node
    opened = []
    reached = asyncio.Event()
    proceed = asyncio.Event()

    async def capture_database(path, **kwargs):
        db = await native_sqlite(path, **kwargs)
        opened.append(db)
        return db

    async def pause_before_identity(graph, node):
        if node.node_type == "agent":
            reached.set()
            await proceed.wait()
        return await native_add(graph, node)

    monkeypatch.setattr(AsyncDatabase, "sqlite", capture_database)
    monkeypatch.setattr(AsyncGraphStore, "add_node", pause_before_identity)
    task = asyncio.create_task(create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, identity_method="did:web",
        did_web_domain="agents.kestrel-sovereign.test", did_web_slug="cancel-owned",
    ))
    try:
        await asyncio.wait_for(reached.wait(), 15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert opened[0]._backend._connection is None
        assert not (tmp_path / "kestrel_prime.db").exists()
        assert not list(tmp_path.glob("cancel-owned_*"))
        monkeypatch.setattr(AsyncGraphStore, "add_node", native_add)
        credentials = await create_kestrel_identity_async(
            output_dir=str(tmp_path), is_test_instance=True, identity_method="did:web",
            did_web_domain="agents.kestrel-sovereign.test", did_web_slug="cancel-owned",
        )
        assert credentials.agent_did.endswith(":cancel-owned")
    finally:
        proceed.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        for db in opened:
            await db.close()


@pytest.mark.asyncio
async def test_cancelled_genesis_auditor_cleans_owned_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    native_sqlite = AsyncDatabase.sqlite
    opened = []
    reached = asyncio.Event()
    proceed = asyncio.Event()

    async def capture_database(path, **kwargs):
        db = await native_sqlite(path, **kwargs)
        opened.append(db)
        return db

    async def auditor(prompt):
        reached.set()
        await proceed.wait()
        return {"risk_level": 1, "reasoning": "Synthetic cancelled inception"}

    monkeypatch.setattr(AsyncDatabase, "sqlite", capture_database)
    task = asyncio.create_task(create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, identity_method="did:web",
        did_web_domain="agents.kestrel-sovereign.test", did_web_slug="cancel-auditor", genesis_auditor=auditor,
    ))
    try:
        await asyncio.wait_for(reached.wait(), 15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert opened[0]._backend._connection is None
        assert not (tmp_path / "kestrel_prime.db").exists()
        assert not list(tmp_path.glob("cancel-auditor_*"))
        async def auditor_after_retry(prompt):
            return {"risk_level": 1, "reasoning": "Synthetic retry"}
        credentials = await create_kestrel_identity_async(
            output_dir=str(tmp_path), is_test_instance=True, identity_method="did:web",
            did_web_domain="agents.kestrel-sovereign.test", did_web_slug="cancel-auditor", genesis_auditor=auditor_after_retry,
        )
        assert credentials.agent_did.endswith(":cancel-auditor")
    finally:
        proceed.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        for db in opened:
            await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("damage_after_commit", [None, "owner", "identity"])
async def test_commit_delivery_error_preserves_committed_identity(
    external_db, tmp_path, monkeypatch, owned, damage_after_commit,
):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    identity = "did:web:agents.kestrel-sovereign.test:committed-owned"
    native_sqlite = AsyncDatabase.sqlite
    opened = []

    def lose_commit_delivery(db):
        native_transaction = db.transaction
        lost = False

        @contextlib.asynccontextmanager
        async def transaction():
            nonlocal lost
            async with native_transaction():
                yield db
            if not lost and not db.owns_open_transaction and await db.fetchone(
                "SELECT node_id FROM graph_nodes WHERE node_id=?", (identity,),
            ) is not None:
                lost = True
                if damage_after_commit == "owner":
                    await db.execute_commit("DELETE FROM graph_node_owners WHERE node_id=?", (identity,))
                elif damage_after_commit == "identity":
                    await db.execute_commit("DELETE FROM graph_nodes WHERE node_id=?", (identity,))
                raise RuntimeError("committed publication delivery lost")

        db.transaction = transaction

    async def capture_database(path, **kwargs):
        db = await native_sqlite(path, **kwargs)
        opened.append(db)
        lose_commit_delivery(db)
        return db

    if owned:
        monkeypatch.setattr(AsyncDatabase, "sqlite", capture_database)
    else:
        lose_commit_delivery(external_db)
    try:
        with pytest.raises(RuntimeError, match="committed publication delivery lost"):
            await create_kestrel_identity_async(
                output_dir=str(tmp_path), is_test_instance=True, identity_method="did:web",
                did_web_domain="agents.kestrel-sovereign.test", did_web_slug="committed-owned",
                database=None if owned else external_db,
            )
        assert len(list(tmp_path.glob("committed-owned_*"))) == 5
        if owned:
            assert opened[0]._backend._connection is None
            check = await native_sqlite(str(tmp_path / "kestrel_prime.db"))
        else:
            assert external_db._backend._connection is not None
            check = external_db
        try:
            node = await AsyncGraphStore(check).get_node(identity)
            if damage_after_commit == "identity":
                assert node is None
                return
            assert node is not None
            assert await check.fetchone(
                "SELECT target_id FROM graph_edges WHERE source_id=? AND label='governed_by'",
                (identity,),
            ) == (node.properties["constitution_hash"],)
        finally:
            if owned:
                await check.close()
    finally:
        for db in opened:
            await db.close()


@pytest.mark.asyncio
async def test_inception_refuses_caller_owned_transaction_before_mint(external_db, tmp_path):
    async with external_db.transaction():
        with pytest.raises(RuntimeError, match="top-level commit"):
            await create_kestrel_identity_async(
                output_dir=str(tmp_path), database=external_db, is_test_instance=True,
                identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test",
                did_web_slug="outer-owned",
            )
        assert await external_db.fetchall("SELECT node_id FROM graph_nodes WHERE node_type='agent'") == []
        assert not list(tmp_path.glob("outer-owned_*"))
    assert external_db._backend._connection is not None


@pytest.mark.asyncio
async def test_committed_inception_survives_native_genesis_notice_failure(
    external_db, tmp_path
):
    """The post-commit observation cannot turn a completed birth into failure."""
    await external_db.execute_script("""
        CREATE TRIGGER refuse_genesis_notice BEFORE INSERT ON conversation_history
        BEGIN SELECT RAISE(ABORT, 'native post-commit notice failure'); END;
    """)

    async def auditor(prompt):
        assert not external_db.owns_open_transaction
        return {"risk_level": 1, "reasoning": "Synthetic provider seam"}

    creds = await _incept(external_db, tmp_path, genesis_auditor=auditor)
    node = await AsyncGraphStore(external_db, agent_id=creds.agent_did).get_node(
        creds.agent_did
    )
    assert node.properties["genesis_audit"]["status"] == "passed"
    assert (
        node.properties["genesis_audit"]["constitution_hash"]
        == node.properties["constitution_hash"]
    )
    assert (
        await external_db.fetchone(
            "SELECT source_id FROM graph_edges WHERE source_id=? AND label='governed_by'",
            (creds.agent_did,),
        )
        is not None
    )
    assert await AsyncRAGStore(
        external_db, agent_id=creds.agent_did
    ).read_indexed_chunks(node.properties["constitution_hash"])


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

    monkeypatch.setattr(AsyncGraphStore, "lock_nodes_for_update", observe_lock)
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
            raise RuntimeError("injected failure between agent node and governing edge")
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
