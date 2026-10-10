"""Birth replay is not permission to claim metadata or reset a used lifetime."""

import hashlib
import asyncio
from contextlib import suppress
from uuid import uuid4

import pytest

from kestrel_sovereign.constitution.anchored_bytes import read_anchored_constitution
from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.identity.birth_record import replicate_birth_record
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode
from tests.integration.test_constitution_refusal_races import _agent


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_birth_reserves_ordinary_blob_set_before_any_publication(db_backend, tmp_path, monkeypatch):
    if db_backend.backend_type != "postgres":
        pytest.skip("native PostgreSQL advisory reservation contention")
    import asyncpg
    from kestrel_sovereign.storage.async_graph_store import (
        _GRAPH_NODE_RESERVATION_SHARDS, _GRAPH_NODE_RESERVATION_KEY_BY_SHARD,
    )

    identity = "did:test:complete-birth-blobs:" + uuid4().hex
    source = AsyncStorage(str(tmp_path / "complete-birth.db"), agent_id=identity)
    target = AsyncStorage(backend=db_backend, agent_id=identity)
    await source.initialize()
    await target.initialize()
    peer = await asyncpg.connect(db_backend._dsn)
    task = transaction = None

    def shard(node_id):
        return int.from_bytes(hashlib.sha256(f"kestrel:graph-node:{node_id}".encode()).digest()[:8], "big") % _GRAPH_NODE_RESERVATION_SHARDS

    try:
        governing = ("governing " + uuid4().hex).encode()
        digest = await source.store_file(governing, "constitution.md")
        await source.add_node(GraphNode(node_id=identity, node_type="agent", label="birth", properties={"constitution_hash": digest}))
        await source.add_node(GraphNode(node_id=digest, node_type="document", label="KESTREL_CONSTITUTION", properties={"hash": digest, "type": "Constitution"}))
        await source.add_edge(identity, digest, "governed_by")
        ordinary = ("ordinary birth file " + uuid4().hex).encode()
        ordinary_hash = hashlib.sha256(ordinary).hexdigest()
        while shard(ordinary_hash) in {shard(identity), shard(digest)}:
            ordinary += b"x"
            ordinary_hash = hashlib.sha256(ordinary).hexdigest()
        await source.store_file(ordinary, "ordinary.txt")
        transaction = peer.transaction()
        await transaction.start()
        await peer.fetchval("SELECT pg_advisory_xact_lock($1::bigint)", _GRAPH_NODE_RESERVATION_KEY_BY_SHARD[shard(ordinary_hash)])
        native_fetch = db_backend.fetch_all
        reached = asyncio.Event()
        writer_pid = None

        async def observed(query, params=()):
            nonlocal writer_pid
            if asyncio.current_task() is task and "SELECT pg_advisory_xact_lock" in query:
                writer_pid = await db_backend.fetch_val("SELECT pg_backend_pid()")
                reached.set()
            return await native_fetch(query, params)

        monkeypatch.setattr(db_backend, "fetch_all", observed)
        task = asyncio.create_task(replicate_birth_record(runtime_db=target.db, anchor_db=source.db, agent_did=identity))
        await asyncio.wait_for(reached.wait(), 5)
        async with asyncio.timeout(5):
            while await peer.fetchval("SELECT wait_event_type FROM pg_stat_activity WHERE pid=$1", writer_pid) != "Lock":
                assert not task.done(), "birth finished without reserving its ordinary blob set"
                await asyncio.sleep(0.02)
        assert await peer.fetchval("SELECT COUNT(*) FROM files WHERE content_hash IN ($1,$2)", digest, ordinary_hash) == 0
        await transaction.rollback()
        transaction = None
        await asyncio.wait_for(task, 10)
        assert await target.files.retrieve_file(ordinary_hash) == ordinary
    finally:
        if transaction is not None:
            await transaction.rollback()
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await peer.close()
        await source.close()
        await target.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "damage",
    [
        "private-target",
        "source-bytes",
        "destination-bytes",
        "owned-destination-bytes",
        "consumed-lifetime",
        "lost-consumed-state",
    ],
)
async def test_birth_refuses_unproven_native_custody(db_backend, tmp_path, damage):
    identity = "did:test:replay-custody:" + uuid4().hex
    source = AsyncStorage(str(tmp_path / "birth.db"), agent_id=identity)
    target = AsyncStorage(backend=db_backend, agent_id=identity)
    await source.initialize()
    await target.initialize()
    content = ("unique native birth " + uuid4().hex).encode()
    digest = hashlib.sha256(content).hexdigest()
    try:
        await source.store_file(content, "constitution.md")
        await source.add_node(
            GraphNode(
                node_id=identity,
                node_type="agent",
                label="birth",
                properties={
                    "constitution_hash": digest,
                    "genesis_audit": {"status": "pending", "constitution_hash": digest},
                },
            )
        )
        await source.add_node(
            GraphNode(
                node_id=digest,
                node_type="document",
                label="KESTREL_CONSTITUTION",
                properties={"hash": digest, "type": "Constitution"},
            )
        )
        await source.add_edge(identity, digest, "governed_by")
        if damage == "private-target":
            foreign = "did:test:private-owner:" + uuid4().hex
            other = AsyncStorage(backend=db_backend, agent_id=foreign)
            await other.initialize()
            await other.add_node(
                GraphNode(
                    node_id=digest,
                    node_type="episode",
                    label="private",
                    properties={"owner": foreign, "private": "never disclose"},
                )
            )
        elif damage == "source-bytes":
            await source.db.execute_commit(
                "UPDATE files SET content=?,metadata=NULL WHERE content_hash=?",
                (b"bad birth source", digest),
            )
        elif damage in {"destination-bytes", "owned-destination-bytes"}:
            await target.db.execute_commit(
                "INSERT INTO files (content_hash,original_name,content) VALUES (?,?,?)",
                (digest, "corrupt destination", b"bad destination"),
            )
            if damage == "owned-destination-bytes":
                await target.db.execute_commit(
                    "INSERT INTO file_owners (content_hash,agent_id,original_name) VALUES (?,?,?)",
                    (digest, identity, "owned corrupt destination"),
                )
        else:
            agent = await _agent(target)
            await target.add_node(
                GraphNode(
                    node_id=identity, node_type="agent", label="consumed", properties={}
                )
            )
            assert not (
                await ConstitutionMixin._get_governing_constitution(agent)
            ).startswith("Error:")
            await target.db.execute_commit(
                "DELETE FROM graph_nodes WHERE node_id=?", (identity,)
            )
            before = await agent._constitution_state_store.load(identity)
            if damage == "lost-consumed-state":
                await target.db.execute_commit(
                    "DELETE FROM constitution_runtime_state WHERE agent_id=?", (identity,)
                )

        with pytest.raises(
            Exception,
            match="foreign|private|share|hash|verify|consumed|claim|another agent|lifetime history",
        ):
            await replicate_birth_record(
                runtime_db=target.db, anchor_db=source.db, agent_did=identity
            )
        assert await target.get_node(identity) is None
        assert (
            await target.db.fetchone(
                "SELECT node_id FROM graph_node_owners WHERE node_id=? AND agent_id=?",
                (digest, identity),
            )
            is None
        )
        if damage == "private-target":
            assert (await other.get_node(digest)).properties[
                "private"
            ] == "never disclose"
        if damage == "consumed-lifetime":
            assert await agent._constitution_state_store.load(identity) == before
        if damage == "lost-consumed-state":
            assert await agent._constitution_state_store.load(identity) is None
            assert await agent._constitution_state_store.list_events(identity)
    finally:
        await source.close()
        await target.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_readable_corrupt_historical_plaintext_is_not_rights_evidence(db_backend):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:historical:" + uuid4().hex
    )
    await storage.initialize()
    try:
        digest = await storage.store_file(
            b"original active rights " + uuid4().hex.encode(), "historical.md"
        )
        await storage.db.execute_commit(
            "UPDATE files SET content=?,metadata=NULL WHERE content_hash=?",
            (b"readable replacement dormant constitution", digest),
        )
        assert await read_anchored_constitution(storage.db, digest) == (None, True)
    finally:
        await storage.close()
