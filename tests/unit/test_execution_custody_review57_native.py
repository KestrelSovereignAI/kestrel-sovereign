"""Review57 proofs using real SDK pools, loopback sockets and original owners."""
import asyncio
import json

import ollama
import httpx
import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["constructor", "removed", "retargeted"])
@pytest.mark.parametrize("cleanup", ["loaded", "cold"])
async def test_preinitialize_and_historical_immutable_artifacts_survive_offboarding(tmp_path, monkeypatch, selection, cleanup):
    from unittest.mock import Mock
    from kestrel_sovereign.features.isolated_runtime import (
        ProxyFeature, IsolatedRuntimePreparationError, remove_agent_runtime_namespace,
        remove_runtime_namespace,
    )
    from tests.unit.test_isolated_feature_runtime import (
        _configure_idle_lifecycle, _idle_test_runtime, _write_prebuilt_venv_shape, FakeIsolatedClient,
    )
    agent = Mock(did="did:test:original-native-artifact", features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    seed = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    workspace = seed._prepare_runtime_workspace()
    retained = workspace / "operator-owned-venv"
    _write_prebuilt_venv_shape(retained)
    key = "KESTREL_FEATURE_TESTFEATURE_VENV"
    monkeypatch.setenv(key, str(retained))
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    # Never initialize this newly accepted feature: constructor validation
    # alone has already accepted the operator's immutable native artifact.
    if selection != "constructor":
        feature.resolve_runtime_paths()
        if selection == "removed":
            monkeypatch.delenv(key)
        else:
            external = tmp_path / "new-operator-venv"
            _write_prebuilt_venv_shape(external)
            monkeypatch.setenv(key, str(external))
        feature.resolve_runtime_paths()
    with pytest.raises(IsolatedRuntimePreparationError, match="immutable"):
        if cleanup == "loaded":
            remove_agent_runtime_namespace(agent)
        else:
            # Cold AgentManager deletion has only the namespace/DID. Clear all
            # proxy/settings witnesses: canonical disk preflight must retain
            # historical selections even after a process restart.
            scope = feature._isolated_runtime_scope
            agent._isolated_runtime_cleanup_owners.clear()
            monkeypatch.delenv(key, raising=False)
            remove_runtime_namespace(scope, agent.did)
    assert (retained / "pyvenv.cfg").is_file()
    assert (retained / "bin/python").is_file()


@pytest.mark.parametrize("earlier", ["revoked", "committed"])
def test_multiple_original_scopes_keep_strongest_native_commit_receipt(earlier):
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody, bind_execution_cleanup, require_execution_work,
        require_execution_backend, execution_commit_outcome,
    )
    from tests.unit.test_execution_custody_review39_native import Fence
    first, last = ExecutionCustody(Fence()), ExecutionCustody(Fence())
    if earlier == "revoked":
        first.revoke("first admission retired")
    else:
        first.preserve_commit_uncertainty("committed")
    last.preserve_commit_uncertainty("unknown")
    # Cleanup binding presents the exact accepted scopes without admission;
    # ordinary work remains forbidden and may not erase the durable receipt.
    with bind_execution_cleanup(None, (first, last)):
        with pytest.raises(BaseException) as caught:
            require_execution_work()
        assert execution_commit_outcome(caught.value) == "unknown"
        with pytest.raises(BaseException) as caught:
            require_execution_backend("postgres", (first, last))
        assert execution_commit_outcome(caught.value) == "unknown"
    with pytest.raises(BaseException) as caught:
        require_execution_backend("postgres", (first, last))
    assert execution_commit_outcome(caught.value) == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("observer", ["request", "response"])
@pytest.mark.parametrize("status", [200, 400, 500])
async def test_native_httpx_late_observer_cannot_dispatch_or_read_after_retirement(tmp_path, observer, status):
    import aiosqlite
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody, bind_execution_custody_snapshot,
        retain_native_execution_http_client_close, is_execution_control_error,
    )
    from tests.unit.test_execution_custody_review39_native import Fence
    async with aiosqlite.connect(tmp_path / "native-observer.db") as ledger:
        await ledger.execute("CREATE TABLE effects(kind TEXT)")
        original = ExecutionCustody(Fence())
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                await ledger.execute("INSERT INTO effects VALUES ('body-read')")
                await ledger.commit()
                yield b"private-native-result"
            async def aclose(self):
                await ledger.execute("INSERT INTO effects VALUES ('body-close')")
                await ledger.commit()
        async def transport(request):
            await ledger.execute("INSERT INTO effects VALUES ('dispatch')")
            await ledger.commit()
            return httpx.Response(status, stream=Body())
        async def late_observer(value):
            await asyncio.sleep(0)
            original.revoke("original admission retired inside late native HTTPX observer")
        client = httpx.AsyncClient(transport=httpx.MockTransport(transport), event_hooks={observer: [late_observer]})
        retain_native_execution_http_client_close(client)
        try:
            with bind_execution_custody_snapshot((original,)):
                with pytest.raises(BaseException) as caught:
                    await client.get("https://offline.invalid/private")
            assert is_execution_control_error(caught.value)
            rows = await (await ledger.execute("SELECT kind FROM effects")).fetchall()
            assert rows == ([] if observer == "request" else [("dispatch",), ("body-close",)])
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_early_signed_reanchor_native_committed_receipt_is_not_an_unbound_local(tmp_path, monkeypatch):
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, execution_commit_outcome
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from tests.integration.test_constitution_turn_admission import _ready_turn
    from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
    async with AsyncStorage(str(tmp_path / "early-reanchor.db"), agent_id="did:test:early-reanchor57") as storage:
        agent, content = await _ready_turn(storage)
        replacement = content + b"\nOffline signed review57 amendment.\n"
        path = tmp_path / "KESTREL_CONSTITUTION.md"
        path.write_bytes(replacement)
        import kestrel_sovereign.config as config
        monkeypatch.setattr(config, "CONSTITUTION_PATH", str(path))
        artifact, root = _write_authority_files(tmp_path, replacement)
        agent._sovereign_trust_root_path = root
        native_lock = storage.lock_nodes_for_update
        original = OSError("native pre-publication graph lock acknowledgement")
        original.__cause__ = ExecutionCommitOutcomeError("committed")
        async def fail_after_native_lock(nodes):
            await native_lock(nodes)
            raise original
        monkeypatch.setattr(storage, "lock_nodes_for_update", fail_after_native_lock)
        with pytest.raises(BaseException) as caught:
            await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))
        assert caught.value is original
        assert execution_commit_outcome(caught.value) == "committed"
        assert agent._safe_mode and "reconciliation" in agent._safe_mode_reason


@pytest.mark.asyncio
async def test_subsequent_original_sdk_pull_retires_suspended_native_response(tmp_path):
    import httpx2 as httpx
    import aiosqlite
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_custody_snapshot, execution_stream
    from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
    from tests.unit.test_execution_custody_review39_native import Fence, openai_chunk
    import openai
    async with aiosqlite.connect(tmp_path / "pull-close.db") as ledger:
        await ledger.execute("CREATE TABLE effects(kind TEXT)")
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield openai_chunk()
                pytest.fail("retired original admission advanced native body again")
            async def aclose(self):
                await ledger.execute("INSERT INTO effects VALUES ('close')")
                await ledger.commit()
        client = openai.AsyncOpenAI(api_key="offline-transport-only", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
                httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body()))))
        scope = ExecutionCustody(Fence())
        with bind_execution_custody_snapshot((scope,)):
            stream = execution_stream(None, OpenAIAdapter().get_streaming_response(
                client=client, model="offline", messages=[{"role": "user", "content": "offline"}]))
        try:
            assert await anext(stream) == "hello"
            scope.revoke("original retired between native SDK pulls")
            with pytest.raises(BaseException, match="between native SDK pulls"):
                await anext(stream)
            assert await (await ledger.execute("SELECT kind FROM effects")).fetchall() == [("close",)]
        finally:
            await stream.aclose()
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("discovery", ["list_models", "list_embedding_models"])
@pytest.mark.parametrize("close_failure", [False, True])
async def test_temporary_ollama_pool_has_physical_retirement_owner(monkeypatch, discovery, close_failure):
    peer_done = asyncio.Event()
    connections = []
    body = json.dumps({"models": []}).encode()
    async def peer(reader, writer):
        connections.append(writer)
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                         + b"\r\nContent-Type: application/json\r\n\r\n" + body)
            await writer.drain()
            await reader.read(1)
        finally:
            writer.close()
            await writer.wait_closed()
            peer_done.set()
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    sdk = ollama.AsyncClient(host=f"http://127.0.0.1:{port}")
    monkeypatch.setattr(ollama, "AsyncClient", lambda: sdk)
    pool = sdk._client._transport._pool
    native_close = pool._close_connections
    retained_connections = []
    calls = []
    original = OSError("discovery native socket retirement failed")
    async def failed_close(native_connections):
        if not native_connections:
            return await native_close(native_connections)
        calls.append(True)
        retained_connections.extend(native_connections)
        raise original
    if close_failure:
        monkeypatch.setattr(pool, "_close_connections", failed_close)
    adapter = OllamaAdapter()
    try:
        if close_failure:
            with pytest.raises(BaseException) as caught:
                await getattr(adapter, discovery)()
            assert original in tuple(_explicit_control_chain(caught.value))
            assert not execution_cleanup_debts(original)
            first = adapter._discovery_clients[sdk._client]
            assert first.done() and not peer_done.is_set()
            with pytest.raises(BaseException) as caught:
                await adapter.aclose()
            assert original in tuple(_explicit_control_chain(caught.value))
            assert not execution_cleanup_debts(original)
            assert execution_cleanup_debts(caught.value)
            assert all(resource is sdk._client and task.done() for resource, task in execution_cleanup_debts(caught.value))
            assert adapter._discovery_clients[sdk._client] is first
            assert calls == [True]
        else:
            assert await getattr(adapter, discovery)() == []
            await asyncio.wait_for(peer_done.wait(), 2)
            assert sdk._client.is_closed
            assert not adapter._discovery_clients
            await adapter.aclose()
    finally:
        monkeypatch.undo()
        if retained_connections:
            await native_close(retained_connections)
        await sdk._client.aclose()
        for writer in connections:
            writer.close()
            await writer.wait_closed()
        server.close()
        await server.wait_closed()
        if connections:
            await asyncio.wait_for(peer_done.wait(), 2)


@pytest.mark.asyncio
async def test_inception_transfers_failed_native_provider_and_birth_database(monkeypatch, tmp_path):
    from kestrel_sovereign import inception_service
    from kestrel_sovereign.llm import service as service_module
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
    from tests.unit.test_execution_custody_review30 import service_owner

    peer_done = asyncio.Event()
    async def peer(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
            await writer.drain()
            await reader.read(1)
        finally:
            writer.close()
            await writer.wait_closed()
            peer_done.set()
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = httpx.AsyncClient()
    await client.get(f"http://127.0.0.1:{port}/")
    pool = client._transport._pool
    connections, native_close = list(pool.connections), pool._close_connections
    original, calls = OSError("inception transport did not retire"), []
    source = OriginalFailure.capture(original)
    async def failed_close(native_connections):
        calls.append(True)
        raise original
    monkeypatch.setattr(pool, "_close_connections", failed_close)
    provider = service_owner()
    provider._owner_agent_did = None
    provider.providers = [{"adapter": client}]
    born_databases = []
    def original_provider(*, usage_db):
        born_databases.append(usage_db)
        provider._usage_db = usage_db
        return provider
    monkeypatch.setattr(service_module, "LLMService", original_provider)
    native_chunk = AsyncRAGStore.chunk_document
    async def offline_chunk(self, *args, **kwargs):
        kwargs["compute_embeddings"] = False
        return await native_chunk(self, *args, **kwargs)
    monkeypatch.setattr(AsyncRAGStore, "chunk_document", offline_chunk)
    async def offline_auditor(_prompt):
        return {"risk_level": 1, "reasoning": "Offline native retirement test"}
    owner = None
    try:
        with pytest.raises(BaseException) as caught:
            await inception_service.create_kestrel_identity_async(
                output_dir=str(tmp_path), is_test_instance=True,
                genesis_auditor=offline_auditor,
                identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test",
                did_web_slug="review57-retirement",
            )
        source.assert_retained(caught.value, resource=client)
        # Prove the physical birth DB and socket stay owned before inspecting
        # the new retention record. Old-source failure must be native custody,
        # not the absence of a newly added helper attribute.
        assert born_databases[0].backend._connection is not None
        assert not peer_done.is_set()
        owner = caught.value.inception_retirement_owner
        assert owner.provider_owner.provider is provider
        assert owner.database is born_databases[0]
        assert owner.database.backend._connection is not None
        assert not peer_done.is_set()
        first = owner.provider_owner.task
        with pytest.raises(BaseException) as caught:
            await owner.retire()
        source.assert_retained(caught.value, resource=client)
        assert caught.value.inception_retirement_owner is owner
        assert owner.provider_owner.task is first
        assert calls == [True]
        with pytest.raises(Exception, match="inception|lock|busy|held"):
            await inception_service.create_kestrel_identity_async(
                output_dir=str(tmp_path), is_test_instance=True, force=True,
                identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test",
                did_web_slug="review57-retirement",
            )
        assert len(born_databases) == 1
    finally:
        monkeypatch.undo()
        await native_close(connections)
        await asyncio.wait_for(peer_done.wait(), 2)
        await client.aclose()
        for database in born_databases:
            await database.close()
        if owner is not None and owner.directory_custody is not None:
            owner.directory_custody.close()
        server.close()
        await server.wait_closed()
