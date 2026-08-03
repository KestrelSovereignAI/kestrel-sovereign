"""#2871 — the birth record must reach the database the runtime reads.

``kestrel create`` writes the birth record into the SQLite it opens in the
agent's directory. A host configured for PostgreSQL boots the agent against
PostgreSQL, where none of it exists, so the agent came up unnamed with zero
constitution chunks while ``/health`` reported ok.

These tests use two SQLite files rather than PostgreSQL: the defect is
"inception's database is not the runtime's database", which two files reproduce
exactly, and SQLite is the supported default that every feature must work on.

Mutation traps this file is written against:

* Asserting "an agent node exists in the runtime database" PASSES on the broken
  code — boot fabricates exactly such a node. Every assertion here names the
  real label, the inception properties, or a count.
* Asserting "replication returned N chunks" tests the return value, not the
  database. Counts are read back out of the target.
* An idempotency test that inserts into an empty target twice proves nothing.
  The second pass here runs against an already-populated target.
"""

import pytest

from kestrel_sovereign.identity.birth_record import (
    diagnose_runtime_birth_record,
    diagnose_birth_record,
    is_fabricated_placeholder,
    local_anchor_path,
    replicate_birth_record,
    runtime_database_is_the_anchor,
)
from kestrel_sovereign.inception_service import (
    DID_WEB_DOMAIN_ENV,
    IDENTITY_METHOD_ENV,
    create_kestrel_identity_async,
)
from kestrel_sovereign.storage import GraphNode
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
from kestrel_sovereign.storage.async_rag_store import (
    AsyncRAGStore,
    IndexedChunk,
)

TEST_DOMAIN = "agents.kestrel-sovereign.test"
TEST_DATA_KEY = "test-master-key-for-encryption-32chars!"


@pytest.fixture
def hybrid_env(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", TEST_DATA_KEY)
    monkeypatch.setenv(DID_WEB_DOMAIN_ENV, TEST_DOMAIN)
    monkeypatch.delenv(IDENTITY_METHOD_ENV, raising=False)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)


async def _incept(tmp_path, name="Replica Bird"):
    """Run a real inception and return (creds, anchor_db)."""
    creds = await create_kestrel_identity_async(
        str(tmp_path), None, agent_name=name,
    )
    anchor_db = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    return creds, anchor_db


async def _fresh_runtime(tmp_path):
    """A second database standing in for the configured runtime backend."""
    return await AsyncDatabase.sqlite(str(tmp_path / "runtime.db"))


async def _chunk_count(db, agent_did):
    row = await db.fetchone(
        "SELECT COUNT(*) FROM document_chunk_owners WHERE agent_id = ?",
        (agent_did,),
    )
    return int(row[0])


# ---------------------------------------------------------------------------
# The core defect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replication_puts_the_real_birth_record_in_the_runtime_database(
    tmp_path, hybrid_env,
):
    """The runtime database must end up holding inception's record — the real
    name, the constitution hash, the bootstrap state, the governing edge and the
    constitution chunks. A fabricated placeholder satisfies none of these."""
    creds, anchor = await _incept(tmp_path, name="Replica Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        anchor_chunks = await _chunk_count(anchor, creds.agent_did)
        assert anchor_chunks > 0, "inception indexed no constitution chunks"
        assert await _chunk_count(runtime, creds.agent_did) == 0

        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )

        node = await AsyncGraphStore(runtime).get_node(creds.agent_did)
        assert node is not None
        assert node.label == "Replica Bird"
        assert node.properties["name"] == "Replica Bird"
        assert node.properties["bootstrap_state"] == "pending"
        assert node.properties["constitution_hash"]
        assert not is_fabricated_placeholder(node, creds.agent_did)

        edges = await AsyncGraphStore(runtime, agent_id=creds.agent_did).get_edges(
            creds.agent_did, direction="out",
        )
        governed = [e for e in edges if e.label == "governed_by"]
        assert len(governed) == 1
        assert governed[0].target_id == node.properties["constitution_hash"]

        # The constitution node itself, not just the edge pointing at it.
        constitution = await AsyncGraphStore(runtime).get_node(
            governed[0].target_id,
        )
        assert constitution is not None
        assert constitution.label == "KESTREL_CONSTITUTION"

        assert await _chunk_count(runtime, creds.agent_did) == anchor_chunks
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_replicated_constitution_is_retrievable_from_the_runtime_database(
    tmp_path, hybrid_env,
):
    """The point of the chunks is retrieval. Text carried across must be
    searchable in the runtime database, not merely counted there."""
    creds, anchor = await _incept(tmp_path, name="Quotable Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        source_text = "".join(
            await AsyncRAGStore(
                anchor, agent_id=creds.agent_did,
            ).get_chunks_for_file(
                (
                    await AsyncGraphStore(anchor).get_node(creds.agent_did)
                ).properties["constitution_hash"]
            )
        )
        copied_text = "".join(
            await AsyncRAGStore(
                runtime, agent_id=creds.agent_did,
            ).get_chunks_for_file(
                (
                    await AsyncGraphStore(runtime).get_node(creds.agent_did)
                ).properties["constitution_hash"]
            )
        )
        assert copied_text == source_text
        assert "Kestrel" in copied_text
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_replication_carries_the_embeddings_it_does_not_recompute(
    tmp_path, hybrid_env,
):
    """The embedding service is not available at boot on every host, and a
    re-embed under a changed model would land the copy in a different
    coordinate space. The vectors must be carried across verbatim.

    The unit environment has no embedding service, so inception's own chunks
    come out unembedded — this seeds vectors onto the anchor first, otherwise
    dropping every embedding during replication would go unnoticed here.
    """
    creds, anchor = await _incept(tmp_path, name="Vector Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        constitution_hash = (
            await AsyncGraphStore(anchor).get_node(creds.agent_did)
        ).properties["constitution_hash"]
        seeded = [
            IndexedChunk("first section", [0.5, -0.25, 0.125], "profile-x"),
            IndexedChunk("second section", [-1.0, 0.75, 0.0], "profile-x"),
        ]
        await AsyncRAGStore(
            anchor, agent_id=creds.agent_did,
        ).store_precomputed_chunks(constitution_hash, seeded)

        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )

        copied = await AsyncRAGStore(
            runtime, agent_id=creds.agent_did,
        ).read_indexed_chunks(constitution_hash)
        assert [c.content for c in copied] == [c.content for c in seeded]
        assert [c.embedding for c in copied] == [
            pytest.approx(c.embedding) for c in seeded
        ]
        assert [c.profile_id for c in copied] == ["profile-x", "profile-x"]
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_replication_repairs_an_existing_fabricated_placeholder(
    tmp_path, hybrid_env,
):
    """An agent that already booted against the wrong database carries a
    fabricated ``Agent <did>`` node. Repair is the same code path as
    prevention — the placeholder must be overwritten by the real record."""
    creds, anchor = await _incept(tmp_path, name="Damaged Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        placeholder = GraphNode(
            node_id=creds.agent_did,
            node_type="agent",
            label=f"Agent {creds.agent_did}",
            properties={"initialBalance": "100.0"},
        )
        await AsyncGraphStore(runtime, agent_id=creds.agent_did).add_node(
            placeholder,
        )
        assert is_fabricated_placeholder(
            await AsyncGraphStore(runtime).get_node(creds.agent_did),
            creds.agent_did,
        )

        divergence = await diagnose_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        assert divergence, "a placeholder must be reported as divergent"

        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )

        repaired = await AsyncGraphStore(runtime).get_node(creds.agent_did)
        assert repaired.label == "Damaged Bird"
        assert not is_fabricated_placeholder(repaired, creds.agent_did)
        assert not await diagnose_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_replication_is_idempotent_against_a_populated_target(
    tmp_path, hybrid_env,
):
    """Every boot retries, so the second pass runs against a target that
    already holds the record. It must converge, not accumulate."""
    creds, anchor = await _incept(tmp_path, name="Twice Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        after_first = await _chunk_count(runtime, creds.agent_did)
        first_edges = len(
            await AsyncGraphStore(
                runtime, agent_id=creds.agent_did,
            ).get_edges(creds.agent_did, direction="out")
        )

        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )

        assert await _chunk_count(runtime, creds.agent_did) == after_first
        assert len(
            await AsyncGraphStore(
                runtime, agent_id=creds.agent_did,
            ).get_edges(creds.agent_did, direction="out")
        ) == first_edges
        node = await AsyncGraphStore(runtime).get_node(creds.agent_did)
        assert node.label == "Twice Bird"
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_repair_never_reverts_durable_post_inception_state(
    tmp_path, hybrid_env,
):
    """The anchor is frozen at inception; the runtime node keeps living.

    A completed genesis audit, a reanchored ``constitution_hash``, an avatar
    hash — all are written to the runtime node after inception and none of them
    exist in the anchor. Replication repairs what is MISSING; it must never
    copy the anchor's stale snapshot over a node that is already a genuine
    birth record, or a boot whose only complaint was a chunk count would undo a
    signed constitutional amendment.
    """
    creds, anchor = await _incept(tmp_path, name="Living Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        graph = AsyncGraphStore(runtime, agent_id=creds.agent_did)
        live = await graph.get_node(creds.agent_did)
        assert live.properties.get("genesis_audit", {}).get("status") == "pending"

        # What the runtime legitimately accumulates after inception.
        live.properties["genesis_audit"] = {"status": "passed", "audited": True}
        live.properties["avatar_hash"] = "deadbeef"
        live.properties["constitution_reanchor"] = {"receipt": "signed"}
        await graph.add_node(live)

        # Any divergence at all re-triggers repair — a chunk shortfall is enough.
        constitution_hash = live.properties["constitution_hash"]
        await AsyncRAGStore(
            runtime, agent_id=creds.agent_did,
        ).delete_chunks_for_file(constitution_hash)
        assert await diagnose_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )

        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )

        after = await graph.get_node(creds.agent_did)
        assert after.properties["genesis_audit"] == {"status": "passed", "audited": True}
        assert after.properties["avatar_hash"] == "deadbeef"
        assert after.properties["constitution_reanchor"] == {"receipt": "signed"}
        # ...and the thing that was actually missing did get repaired.
        assert await _chunk_count(runtime, creds.agent_did) == 47
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_replication_converges_after_an_interrupted_pass(
    tmp_path, hybrid_env,
):
    """A pass that died partway must be finished by the next one. Simulated by
    replicating, then deleting the chunks the way a failed copy would leave
    them — the retry has to restore the full count."""
    creds, anchor = await _incept(tmp_path, name="Resumed Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        full = await _chunk_count(runtime, creds.agent_did)
        constitution_hash = (
            await AsyncGraphStore(runtime).get_node(creds.agent_did)
        ).properties["constitution_hash"]
        await AsyncRAGStore(
            runtime, agent_id=creds.agent_did,
        ).delete_chunks_for_file(constitution_hash)
        partial = await _chunk_count(runtime, creds.agent_did)
        assert partial < full

        assert await diagnose_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        ), "a record missing its chunks must be reported as divergent"

        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        assert await _chunk_count(runtime, creds.agent_did) == full
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_two_agents_replicate_into_one_shared_runtime_database(
    tmp_path, hybrid_env,
):
    """PostgreSQL exists here to host many agents in one database, and the
    constitution node is content-addressed — every agent under the same
    constitution shares that one row.

    ``AsyncGraphStore.add_node`` lets a second tenant attach its ownership
    witness to a shared content node only when it already owns the underlying
    file. So the file rows have to be written BEFORE the nodes, which is the
    order inception uses. With nodes first, agent two cannot boot at all, and
    no later boot repairs it.
    """
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    creds_a = await create_kestrel_identity_async(
        str(a_dir), None, agent_name="Bird A",
    )
    creds_b = await create_kestrel_identity_async(
        str(b_dir), None, agent_name="Bird B",
    )
    anchor_a = await AsyncDatabase.sqlite(str(a_dir / "kestrel_prime.db"))
    anchor_b = await AsyncDatabase.sqlite(str(b_dir / "kestrel_prime.db"))
    shared = await AsyncDatabase.sqlite(str(tmp_path / "shared_runtime.db"))
    try:
        hash_a = (
            await AsyncGraphStore(anchor_a).get_node(creds_a.agent_did)
        ).properties["constitution_hash"]
        hash_b = (
            await AsyncGraphStore(anchor_b).get_node(creds_b.agent_did)
        ).properties["constitution_hash"]
        assert hash_a == hash_b, "same constitution must be one shared node"

        await replicate_birth_record(
            runtime_db=shared, anchor_db=anchor_a, agent_did=creds_a.agent_did,
        )
        await replicate_birth_record(
            runtime_db=shared, anchor_db=anchor_b, agent_did=creds_b.agent_did,
        )

        for did, name in ((creds_a.agent_did, "Bird A"), (creds_b.agent_did, "Bird B")):
            node = await AsyncGraphStore(shared).get_node(did)
            assert node is not None and node.label == name
            assert await _chunk_count(shared, did) > 0
        owners = {
            row[0]
            for row in await shared.fetchall(
                "SELECT agent_id FROM graph_node_owners WHERE node_id = ?",
                (hash_a,),
            )
        }
        assert owners == {creds_a.agent_did, creds_b.agent_did}
    finally:
        for db in (anchor_a, anchor_b, shared):
            await db.close()


@pytest.mark.asyncio
async def test_replication_carries_a_child_edge_without_claiming_the_parent(
    tmp_path, hybrid_env,
):
    """A spawned child's ``spawned_by`` points at the parent's node, which
    belongs to another tenant. The edge is part of the birth record and must
    cross; the parent's node is not this agent's to copy or overwrite."""
    parent_did = f"did:web:{TEST_DOMAIN}:the-parent"
    creds = await create_kestrel_identity_async(
        str(tmp_path), None, agent_name="Child Bird", parent_did=parent_did,
    )
    anchor = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    runtime = await _fresh_runtime(tmp_path)
    try:
        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )

        edges = await AsyncGraphStore(
            runtime, agent_id=creds.agent_did,
        ).get_edges(creds.agent_did, direction="out")
        spawned = [e for e in edges if e.label == "spawned_by"]
        assert len(spawned) == 1
        assert spawned[0].target_id == parent_did
        # The parent's node was neither copied nor claimed by the child.
        assert await AsyncGraphStore(runtime).get_node(parent_did) is None
        assert not await diagnose_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
    finally:
        await anchor.close()
        await runtime.close()


# ---------------------------------------------------------------------------
# diagnose: no false positives on a database that already agrees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnose_reports_nothing_when_the_record_is_already_present(
    tmp_path, hybrid_env,
):
    """A healthy runtime database must not be re-replicated on every boot."""
    creds, anchor = await _incept(tmp_path, name="Healthy Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        assert not await diagnose_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_diagnose_is_silent_when_the_anchor_holds_no_record(
    tmp_path, hybrid_env,
):
    """A Cloud Run style host whose ceremony wrote straight to the runtime
    database has an empty (or absent) anchor. There is nothing to copy, and
    reporting divergence here would refuse a correctly-created agent."""
    anchor = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    runtime = await _fresh_runtime(tmp_path)
    try:
        assert not await diagnose_birth_record(
            runtime_db=runtime,
            anchor_db=anchor,
            agent_did=f"did:web:{TEST_DOMAIN}:no-anchor-record",
        )
    finally:
        await anchor.close()
        await runtime.close()


# ---------------------------------------------------------------------------
# Damaged anchors: refuse with the reason, never commit half a record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_file_row_without_bytes_refuses_and_commits_nothing(
    tmp_path, hybrid_env,
):
    """A ``file_owners`` row whose file is gone cannot be replicated by this or
    any later boot. Say that, rather than committing a record whose chunks can
    never arrive and letting the post-copy check reject it with a vaguer
    reason."""
    creds, anchor = await _incept(tmp_path, name="Dangling Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await anchor.execute("DELETE FROM files", ())

        with pytest.raises(ValueError, match="holds no bytes"):
            await replicate_birth_record(
                runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
            )

        # Nothing half-written: no agent node, no chunks.
        assert await AsyncGraphStore(runtime).get_node(creds.agent_did) is None
        assert await _chunk_count(runtime, creds.agent_did) == 0
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_an_edge_without_an_ownership_witness_refuses(tmp_path, hybrid_env):
    """The copier writes through a bound store, so an anchor edge with no
    ownership witness is invisible to it. Copying the rest would record the
    agent with a ``constitution_hash`` naming a node that was never written —
    "recorded but not governed" (#2867), and silent. Refuse instead."""
    creds, anchor = await _incept(tmp_path, name="Unwitnessed Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await anchor.execute("DELETE FROM graph_edge_owners", ())

        with pytest.raises(ValueError, match="no ownership witness"):
            await replicate_birth_record(
                runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
            )
        assert await AsyncGraphStore(runtime).get_node(creds.agent_did) is None
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_diagnosis_measures_only_what_replication_can_write(
    tmp_path, hybrid_env,
):
    """Verifier and copier must agree on the row set. A chunk-owner row whose
    file this agent does not own is outside the RAG store's tenant scope, so
    replication can never move it — counting it would demand a row nothing can
    produce and refuse the boot on every future attempt."""
    creds, anchor = await _incept(tmp_path, name="Scoped Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        assert not await diagnose_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )

        # An orphan chunk-owner row on the anchor: owned chunk, unowned file.
        await anchor.execute(
            "INSERT INTO document_chunks (file_hash, content) VALUES (?, ?)",
            ("a-file-this-agent-does-not-own", "orphan"),
        )
        row = await anchor.fetchone("SELECT last_insert_rowid()")
        await anchor.execute(
            "INSERT INTO document_chunk_owners (chunk_id, agent_id) VALUES (?, ?)",
            (int(row[0]), creds.agent_did),
        )

        assert not await diagnose_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        ), "an unreachable chunk must not produce a refusal no retry can clear"
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_a_write_that_did_not_land_rolls_the_whole_copy_back(
    tmp_path, hybrid_env, monkeypatch,
):
    """Verify before committing, not after. A store that reports a write it did
    not make is this defect class exactly, so the copy re-reads its own work
    inside the transaction — and half a birth record must not survive."""
    creds, anchor = await _incept(tmp_path, name="Silent Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        async def _swallow_the_edge(self, *args, **kwargs):
            return None

        monkeypatch.setattr(AsyncGraphStore, "add_edge", _swallow_the_edge)

        with pytest.raises(Exception, match="edges missing after writing them"):
            await replicate_birth_record(
                runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
            )

        # Not "agent recorded but not governed" (#2867) — nothing at all.
        assert await AsyncGraphStore(runtime).get_node(creds.agent_did) is None
        assert await _chunk_count(runtime, creds.agent_did) == 0
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_a_dangling_governing_target_is_refused_not_committed(
    tmp_path, hybrid_env, monkeypatch,
):
    """The edge is not the record — the node it names is.

    A bound ``get_node`` returns None for a node with no ownership witness as
    well as for one that does not exist, so a target can be skipped while its
    edge still lands (``add_trusted_cross_agent_edge`` needs only the source).
    That commits an agent whose ``constitution_hash`` points at nothing, and a
    check that only counted edges would then call it healthy forever.
    """
    creds, anchor = await _incept(tmp_path, name="Dangling Target Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        real_add_node = AsyncGraphStore.add_node

        async def _lose_the_document_row(self, node, *args, **kwargs):
            # Report the write, keep the ownership witness, lose the row —
            # a store claiming a write it did not durably make.
            await real_add_node(self, node, *args, **kwargs)
            if node.node_type == "document":
                await self.db.execute(
                    "DELETE FROM graph_nodes WHERE node_id = ?", (node.node_id,),
                )

        monkeypatch.setattr(AsyncGraphStore, "add_node", _lose_the_document_row)

        with pytest.raises(Exception, match="not readable in the runtime database"):
            await replicate_birth_record(
                runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
            )
        assert await AsyncGraphStore(runtime).get_node(creds.agent_did) is None
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_diagnosis_catches_a_governing_edge_with_no_target(
    tmp_path, hybrid_env,
):
    """The same damage, already committed by an older build, must be reported
    as incomplete so it gets repaired rather than blessed."""
    creds, anchor = await _incept(tmp_path, name="Blessed Bird")
    runtime = await _fresh_runtime(tmp_path)
    try:
        await replicate_birth_record(
            runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
        )
        assert not await diagnose_runtime_birth_record(
            runtime_db=runtime, agent_did=creds.agent_did,
        )

        constitution_hash = (
            await AsyncGraphStore(runtime).get_node(creds.agent_did)
        ).properties["constitution_hash"]
        await runtime.execute(
            "DELETE FROM graph_nodes WHERE node_id = ?", (constitution_hash,),
        )

        shortfall = await diagnose_runtime_birth_record(
            runtime_db=runtime, agent_did=creds.agent_did,
        )
        assert shortfall
        assert "no such node is readable" in shortfall.describe()
    finally:
        await anchor.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_a_complete_runtime_record_never_opens_the_anchor(
    tmp_path, hybrid_env,
):
    """A healthy PostgreSQL host must not depend on a leftover SQLite file it
    no longer reads. Opening the anchor runs its migrations and ownership
    backfills, so a corrupt or read-only one would refuse a boot that is
    entirely fine — the file deciding whether the database may start."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from kestrel_sovereign.agent.boot import BootContext

    creds, agent, runtime, storage = await _storage_phase_agent(
        tmp_path, "Independent Bird",
    )
    try:
        anchor = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
        try:
            await replicate_birth_record(
                runtime_db=runtime, anchor_db=anchor, agent_did=creds.agent_did,
            )
        finally:
            await anchor.close()

        # Corrupt the anchor: opening it now raises.
        (tmp_path / "kestrel_prime.db").write_bytes(b"not a database at all")

        with patch("kestrel_sovereign.kestrel_agent.AsyncStorage", return_value=storage), \
                patch("kestrel_sovereign.storage.db.postgres.PostgresBackend"), \
                patch(
                    "kestrel_sovereign.identity.birth_record.diagnose_birth_record"
                ) as anchor_diagnose:
            await agent._boot_phase_storage_privacy(BootContext())

        anchor_diagnose.assert_not_called()
        assert agent._agent_name == "Independent Bird"
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# The placeholder predicate
# ---------------------------------------------------------------------------


def test_placeholder_predicate_does_not_flag_a_sparse_genuine_node():
    """Matched on the exact shape boot fabricates, not on missing properties —
    an old but genuine node lacking ``name`` must still boot."""
    did = f"did:web:{TEST_DOMAIN}:sparse"
    sparse_but_real = GraphNode(
        node_id=did,
        node_type="agent",
        label="Sparse Bird",
        properties={"initialBalance": "100.0"},
    )
    assert not is_fabricated_placeholder(sparse_but_real, did)

    same_label_more_properties = GraphNode(
        node_id=did,
        node_type="agent",
        label=f"Agent {did}",
        properties={"initialBalance": "100.0", "name": "Real"},
    )
    assert not is_fabricated_placeholder(same_label_more_properties, did)

    fabricated = GraphNode(
        node_id=did,
        node_type="agent",
        label=f"Agent {did}",
        properties={"initialBalance": "100.0"},
    )
    assert is_fabricated_placeholder(fabricated, did)
    assert not is_fabricated_placeholder(None, did)


# ---------------------------------------------------------------------------
# Inert on an ordinary SQLite deployment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_database_is_the_anchor_on_a_sqlite_host(tmp_path):
    """The SQLite deployment opens the anchor as its runtime database, so
    reconciliation must be entirely inert there."""
    path = tmp_path / "kestrel_prime.db"
    db = await AsyncDatabase.sqlite(str(path))
    try:
        assert runtime_database_is_the_anchor(db, path)
        assert runtime_database_is_the_anchor(db, tmp_path / "." / "kestrel_prime.db")
        assert not runtime_database_is_the_anchor(db, tmp_path / "other.db")
    finally:
        await db.close()


def test_local_anchor_path_requires_the_file_to_exist(tmp_path):
    assert local_anchor_path(None) is None
    assert local_anchor_path(str(tmp_path / "missing.db")) is None
    present = tmp_path / "kestrel_prime.db"
    present.write_bytes(b"")
    assert local_anchor_path(str(present)) == present


# ---------------------------------------------------------------------------
# The boot wiring
# ---------------------------------------------------------------------------


async def _storage_phase_agent(tmp_path, name):
    """A PostgreSQL-shaped agent whose runtime database is NOT its anchor.

    ``storage.db`` is a real ``AsyncDatabase`` and ``get_node`` reads through
    it, so the phase sees the runtime database the way production does. A
    duck-typed double here would answer ``get_node`` from thin air and hide the
    very ordering these tests exist to pin.
    """
    from unittest.mock import AsyncMock, MagicMock

    from kestrel_sovereign.kestrel_agent import KestrelAgent

    creds = await create_kestrel_identity_async(
        str(tmp_path), None, agent_name=name,
    )
    runtime = await _fresh_runtime(tmp_path)
    agent = KestrelAgent(
        did=creds.agent_did,
        storage_path=str(tmp_path / "kestrel_prime.db"),
        db_backend="postgres",
        pg_pool=MagicMock(),
        database_url="postgresql://birth-record-test/kestrel",
        llm_service=MagicMock(),
    )
    assert agent.identity is not None

    storage = AsyncMock()
    storage.initialize = AsyncMock()
    storage.db = runtime
    storage.close = AsyncMock()
    async def _read_through(node_id):
        return await AsyncGraphStore(runtime).get_node(node_id)

    storage.get_node = AsyncMock(side_effect=_read_through)
    return creds, agent, runtime, storage


@pytest.mark.asyncio
async def test_storage_phase_reconciles_before_the_agent_node_is_read(
    tmp_path, hybrid_env,
):
    """Ordering is the whole point. The storage phase reads the agent node
    immediately after this — to decide whether the identity is new enough to
    establish a fresh constitution anchor, and to set the agent's name.
    Reconciling any later leaves the agent anchored to the wrong constitution
    and running as "Unnamed Agent", which is exactly what was observed."""
    from unittest.mock import patch

    from kestrel_sovereign.agent.boot import BootContext

    creds, agent, runtime, storage = await _storage_phase_agent(
        tmp_path, "Boot Bird",
    )
    try:
        with patch("kestrel_sovereign.kestrel_agent.AsyncStorage", return_value=storage), \
                patch("kestrel_sovereign.storage.db.postgres.PostgresBackend"):
            await agent._boot_phase_storage_privacy(BootContext())

        assert agent._agent_name == "Boot Bird"
        node = await AsyncGraphStore(runtime).get_node(creds.agent_did)
        assert node is not None and not is_fabricated_placeholder(
            node, creds.agent_did,
        )
        assert await _chunk_count(runtime, creds.agent_did) > 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_storage_phase_refuses_when_replication_leaves_it_incomplete(
    tmp_path, hybrid_env,
):
    """Boot must not proceed on the claim that a copy happened. With
    replication neutered, the post-copy re-check has to stop the boot."""
    from unittest.mock import patch

    from kestrel_sovereign.agent.boot import BootContext
    from kestrel_sovereign.identity.birth_record import ReplicationResult
    from kestrel_sovereign.identity.runtime_identity import IdentityReadinessError

    _creds, agent, runtime, storage = await _storage_phase_agent(
        tmp_path, "Doomed Bird",
    )
    try:
        async def _wrote_nothing(**_kwargs):
            return ReplicationResult()

        with patch("kestrel_sovereign.kestrel_agent.AsyncStorage", return_value=storage), \
                patch("kestrel_sovereign.storage.db.postgres.PostgresBackend"), \
                patch(
                    "kestrel_sovereign.identity.birth_record.replicate_birth_record",
                    _wrote_nothing,
                ):
            with pytest.raises(IdentityReadinessError) as exc_info:
                await agent._boot_phase_storage_privacy(BootContext())

        assert exc_info.value.failure == "birth_record"
        assert exc_info.value.cause_type == "BirthRecordReplicationIncomplete"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_storage_phase_does_not_touch_the_anchor_on_a_sqlite_host(
    tmp_path, hybrid_env,
):
    """The ordinary deployment must not pay for this at all: the runtime
    database IS the anchor, so no diagnosis and no copy may run."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from kestrel_sovereign.agent.boot import BootContext
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    creds = await create_kestrel_identity_async(
        str(tmp_path), None, agent_name="Plain Bird",
    )
    anchor = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    agent = KestrelAgent(
        did=creds.agent_did,
        storage_path=str(tmp_path / "kestrel_prime.db"),
        llm_service=MagicMock(),
    )
    storage = AsyncMock()
    storage.initialize = AsyncMock()
    storage.db = anchor
    storage.close = AsyncMock()
    async def _read_through(node_id):
        return await AsyncGraphStore(anchor).get_node(node_id)

    storage.get_node = AsyncMock(side_effect=_read_through)
    try:
        with patch("kestrel_sovereign.kestrel_agent.AsyncStorage", return_value=storage), \
                patch(
                    "kestrel_sovereign.identity.birth_record.diagnose_birth_record"
                ) as diagnose:
            await agent._boot_phase_storage_privacy(BootContext())
        diagnose.assert_not_called()
        assert agent._agent_name == "Plain Bird"
    finally:
        await anchor.close()


# ---------------------------------------------------------------------------
# The chunk primitive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_precomputed_chunks_round_trip_with_their_vectors(tmp_path):
    """Embeddings are carried, not recomputed: a copy whose vectors were
    dropped would leave semantic recall silently empty."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "chunks.db"))
    try:
        from kestrel_sovereign.storage.async_file_store import AsyncFileStore

        agent = f"did:web:{TEST_DOMAIN}:chunky"
        files = AsyncFileStore(db, agent_id=agent)
        file_hash = await files.store_file(b"alpha beta gamma", "doc.md")
        rag = AsyncRAGStore(db, agent_id=agent)

        written = await rag.store_precomputed_chunks(
            file_hash,
            [
                IndexedChunk("alpha", [0.5, -0.25, 0.125], "profile-a"),
                IndexedChunk("beta", [1.0, 0.0, -1.0], "profile-a"),
            ],
        )
        assert written == 2

        read_back = await rag.read_indexed_chunks(file_hash)
        assert [c.content for c in read_back] == ["alpha", "beta"]
        assert read_back[0].embedding == pytest.approx([0.5, -0.25, 0.125])
        assert read_back[1].embedding == pytest.approx([1.0, 0.0, -1.0])
        assert {c.profile_id for c in read_back} == {"profile-a"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_precomputed_chunks_replace_rather_than_accumulate(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "chunks.db"))
    try:
        from kestrel_sovereign.storage.async_file_store import AsyncFileStore

        agent = f"did:web:{TEST_DOMAIN}:chunky"
        files = AsyncFileStore(db, agent_id=agent)
        file_hash = await files.store_file(b"alpha beta", "doc.md")
        rag = AsyncRAGStore(db, agent_id=agent)
        payload = [IndexedChunk("alpha", [0.5], "p"), IndexedChunk("beta", [0.25], "p")]

        await rag.store_precomputed_chunks(file_hash, payload)
        await rag.store_precomputed_chunks(file_hash, payload)

        assert await _chunk_count(db, agent) == 2
        assert [c.content for c in await rag.read_indexed_chunks(file_hash)] == [
            "alpha", "beta",
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_precomputed_chunks_store_text_when_there_is_no_vector(tmp_path):
    """Inception on a host with no embedding service produces chunks with no
    vectors. Those still have to cross — the text is what BM25 and the
    in-Python fallback search."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "chunks.db"))
    try:
        from kestrel_sovereign.storage.async_file_store import AsyncFileStore

        agent = f"did:web:{TEST_DOMAIN}:unembedded"
        files = AsyncFileStore(db, agent_id=agent)
        file_hash = await files.store_file(b"plain text only", "doc.md")
        rag = AsyncRAGStore(db, agent_id=agent)

        written = await rag.store_precomputed_chunks(
            file_hash, [IndexedChunk("alpha"), IndexedChunk("beta")],
        )
        assert written == 2

        read_back = await rag.read_indexed_chunks(file_hash)
        assert [c.content for c in read_back] == ["alpha", "beta"]
        assert [c.embedding for c in read_back] == [[], []]
        row = await db.fetchone(
            "SELECT COUNT(*) FROM document_chunks "
            "WHERE file_hash = ? AND embedding IS NULL",
            (file_hash,),
        )
        assert int(row[0]) == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_precomputed_chunks_refuse_a_file_outside_the_bound_agent(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "chunks.db"))
    try:
        rag = AsyncRAGStore(db, agent_id=f"did:web:{TEST_DOMAIN}:intruder")
        with pytest.raises(ValueError, match="outside the bound agent"):
            await rag.store_precomputed_chunks(
                "not-a-file-this-agent-owns", [IndexedChunk("x", [0.1], "p")],
            )
    finally:
        await db.close()
