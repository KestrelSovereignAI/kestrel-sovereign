"""Native commit, physical close, and consumer cancellation regressions."""

import asyncio
import sqlite3
from types import SimpleNamespace

import httpx
import openai
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, ExecutionCustody, ProcessRuntimeExecutionFence,
    _bind_captured_custody, _host_identity_anchor_access, execution_commit_outcome,
)
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.llm.lifecycle import UnpublishedLLMOwner
from kestrel_sovereign.llm.modality_recording import ModalityCall
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review30 import service_owner
from tests.unit.test_execution_custody_review32 import receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["ack_error", "ack_cancel", "post_ack_retirement"])
@pytest.mark.parametrize("route", ["write", "transaction", "script"])
async def test_local_anchor_native_commit_preserves_original_scopes(
    tmp_path, monkeypatch, boundary, route,
):
    scopes = tuple(ExecutionCustody(ProcessRuntimeExecutionFence(str(i))) for i in range(2))
    path = tmp_path / "kestrel_prime.db"
    backend = None
    with _bind_captured_custody(scopes), _host_identity_anchor_access(path):
        backend = SQLiteBackend(path)
        await backend.connect()
        await backend.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
        conn = backend._connection
        original_commit = conn.commit

        async def commit_then_lost_ack():
            await original_commit()
            if boundary == "post_ack_retirement":
                scopes[-1].fence.retire()
            elif boundary == "ack_cancel":
                raise asyncio.CancelledError("native COMMIT acknowledgement lost")
            else:
                raise OSError("native COMMIT acknowledgement lost")

        monkeypatch.setattr(conn, "commit", commit_then_lost_ack)
        async def invoke():
            if route == "transaction":
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES(1)")
            elif route == "script":
                await backend.execute_script("INSERT INTO effects VALUES(1);")
            else:
                await backend.execute("INSERT INTO effects VALUES(1)")

        try:
            outcome = await await_owned_task(asyncio.create_task(invoke()))
            expected = "committed" if boundary == "post_ack_retirement" else "unknown"
            with sqlite3.connect(path) as observer:
                assert observer.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 1
            assert execution_commit_outcome(outcome.error) == expected, "native anchor commit lost its outcome"
            for scope in scopes:
                with pytest.raises(ExecutionCommitOutcomeError) as caught:
                    scope.require_work()
                assert execution_commit_outcome(caught.value) == expected
        finally:
            await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_slot", ["client", "adapter"])
async def test_native_tcp_close_is_not_cancelled_before_physical_eof(
    tmp_path, monkeypatch, provider_slot,
):
    from kestrel_sovereign.llm import service as module
    service = service_owner()
    db = await AsyncDatabase.sqlite(str(tmp_path / "borrowed.db"))
    entered, release, peer_eof, peer_done = (asyncio.Event() for _ in range(4))
    async def peer(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK")
            await writer.drain()
            assert await reader.read(1) == b""
            peer_eof.set()
        finally:
            writer.close()
            await writer.wait_closed()
            peer_done.set()

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = httpx.AsyncClient()
    sdk = openai.AsyncOpenAI(api_key="offline-local-only", http_client=client, base_url=f"http://127.0.0.1:{port}")
    await client.get(f"http://127.0.0.1:{port}/")
    pool = client._transport._pool
    connections = list(pool.connections)
    original = pool._close_connections
    async def physical_close(conns):
        entered.set()
        await release.wait()
        await original(conns)
    monkeypatch.setattr(pool, "_close_connections", physical_close)
    service.providers = [{provider_slot: sdk if provider_slot == "client" else client}]
    monkeypatch.setattr(module, "CLIENT_CLOSE_TIMEOUT", 0.01)
    waiter = None
    async def host_retirement():
        await UnpublishedLLMOwner(service, ()).retire(join_native=True)
        await db.close()
    try:
        waiter = asyncio.create_task(host_retirement())
        await asyncio.wait_for(entered.wait(), 5)
        done, _ = await asyncio.wait({waiter}, timeout=0.3)
        assert not done, "host retired a native closer that had only been cooperatively cancelled"
        assert not peer_eof.is_set() and db.backend._connection is not None
        assert service.has_pending_close_owners()
        release.set()
        outcome = await await_owned_task(waiter)
        assert outcome.error is None
        await asyncio.wait_for(peer_eof.wait(), 5)
        assert db.backend._connection is None
    finally:
        release.set()
        if waiter is not None:
            await await_owned_task(waiter)
        await service.wait_for_close_completion()
        # httpx marks CLOSED before transport close: clean the captured real
        # pool connections even when the before-fix close cancelled that await.
        for connection in connections:
            await connection.aclose()
        await sdk.close()
        await db.close()
        server.close()
        await server.wait_closed()
        await asyncio.wait_for(peer_done.wait(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_unbound_modality_invocation_returns_original_native_receipt(tmp_path, monkeypatch, kind):
    service = service_owner()
    db = await AsyncDatabase.sqlite(str(tmp_path / "modality.db"))
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    failure = receipt(kind)
    async def write(_call):
        await db.execute("INSERT INTO effects VALUES(1)")
        raise failure
    monkeypatch.setattr(service, "_write_modality_record", write)
    call = ModalityCall("embedding", "offline", "native", 1, True, LLMInvocationContext())
    try:
        outcome = await await_owned_task(asyncio.create_task(service.record_modality_call(call)))
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert outcome.error is failure, "shield replaced the writer's control-bearing cancellation"
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["rag_vector", "conversation_vector"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_native_vector_commit_receipt_cannot_authorize_fallback_replay(tmp_path, monkeypatch, surface, kind):
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
    path = tmp_path / "vector.db"
    db = await AsyncDatabase.sqlite(str(path))
    failure = receipt(kind)
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    if surface == "rag_vector":
        chunk_id = await AsyncRAGStore(db)._insert_chunk("native", "committed chunk")
        await db.execute("CREATE TRIGGER count_vector_effect AFTER UPDATE ON document_chunks BEGIN INSERT INTO effects VALUES(NULL); END")
    conn = db.backend._connection
    original = conn.commit
    async def commit_then_receipt():
        await original()
        raise failure
    monkeypatch.setattr(conn, "commit", commit_then_receipt)
    async def invoke():
        if surface == "rag_vector":
            return await AsyncRAGStore(db)._write_embedding_vec(chunk_id, [0.1], "native-profile")
        return await AsyncConversationStore(db)._insert_message(
            role="user", content="native", rendered_content=None, metadata=None,
            embedding=[0.1], embedding_profile_id="native-profile",
        )
    try:
        outcome = await await_owned_task(asyncio.create_task(invoke()))
        assert outcome.error is failure, "native vector receipt authorized legacy replay"
        with sqlite3.connect(path) as observer:
            table = "effects" if surface == "rag_vector" else "conversation_history"
            assert observer.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_manager_create_agent_does_not_wrap_native_inception_control(tmp_path, monkeypatch, kind):
    from kestrel_sovereign import inception_service
    manager = AgentManager(base_data_dir=tmp_path)
    failure = receipt(kind)
    db = await AsyncDatabase.sqlite(str(tmp_path / "inception-effect.db"))
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    async def inception(**_kwargs):
        await db.execute("INSERT INTO effects VALUES(1)")
        raise failure
    monkeypatch.setattr(manager, "_data_key_custody_conflict", lambda: None)
    monkeypatch.setattr(inception_service, "create_kestrel_identity_async", inception)
    try:
        outcome = await await_owned_task(asyncio.create_task(manager.create_agent("Unborn")))
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert outcome.error is failure, "manager inception converted native control to ordinary ValueError"
        assert not manager.list_agents()
        assert not manager._reserved_ports
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_batch_second_cancellation_still_closes_unpublished_native_agent(tmp_path, monkeypatch):
    # Recovery now joins original initializer Tasks directly, so an artificial
    # second gather is no longer a production boundary. Hold the ACTUAL native
    # SQLite close ACK instead and send the second cancellation at that owner.
    from tests.unit.test_cleanup_task_admission import assert_cancelled_batch_native_cleanup

    await assert_cancelled_batch_native_cleanup(tmp_path, monkeypatch, cancel_again=True)

@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["rag_resolver", "conversation_resolver", "rag_batch"])
@pytest.mark.parametrize("kind", ["unknown", "group"])
async def test_embedding_fallback_never_suppresses_native_control(tmp_path, monkeypatch, surface, kind):
    from kestrel_sovereign.llm import embedding_service
    from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
    db = await AsyncDatabase.sqlite(str(tmp_path / "embedding.db"))
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    await db.execute("CREATE TABLE IF NOT EXISTS document_chunks(chunk_id INTEGER PRIMARY KEY, file_hash TEXT, content TEXT, chunk_index INTEGER, created_at TEXT)")
    failure = receipt(kind)
    if kind == "group":
        failure = ExceptionGroup("original native embedding group", [ExecutionCommitOutcomeError("unknown")])
    async def batch(_chunks):
        await db.execute("INSERT INTO effects VALUES(1)")
        raise failure
    def resolver(*_args):
        with sqlite3.connect(tmp_path / "embedding.db") as native:
            native.execute("INSERT INTO effects VALUES(1)")
        raise failure
    rag = AsyncRAGStore(db)
    conversation = AsyncConversationStore.__new__(AsyncConversationStore)
    conversation._llm_service = None
    monkeypatch.delenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", raising=False)
    monkeypatch.setattr(embedding_service, "get_provider_embedding_service", resolver)
    if surface == "rag_batch":
        monkeypatch.setattr(rag, "_get_embedding_service", lambda: SimpleNamespace(aembed_batch=batch))
    async def invoke():
        if surface == "rag_resolver":
            return rag._get_embedding_service()
        if surface == "conversation_resolver":
            return conversation._lazy_embedding_service()
        return await rag.chunk_document("native", "native document")
    try:
        outcome = await await_owned_task(asyncio.create_task(invoke()))
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert outcome.error is failure, "embedding fallback treated native control as ordinary provider outage"
        assert await db.fetchval("SELECT COUNT(*) FROM document_chunks") == 0
    finally:
        await db.close()
