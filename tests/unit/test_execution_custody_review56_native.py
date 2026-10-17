"""Physical filesystem, SQLite, SDK and TCP regression proofs for review56."""

import asyncio
import sqlite3
from unittest.mock import Mock

import aiosqlite
import anthropic
import httpx2 as httpx
import openai
import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionCustody,
    bind_execution_custody_snapshot, execution_commit_outcome,
)
from kestrel_sovereign.features.isolated_runtime import (
    IsolatedRuntimePreparationError, ProxyFeature, remove_agent_runtime_namespace,
)
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.lifecycle import UnpublishedLLMOwner
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_execution_custody_review30 import service_owner
from tests.unit.test_execution_custody_review39_native import Fence, openai_chunk
from tests.unit.test_execution_custody_review43_native import carries
from tests.unit.test_isolated_feature_runtime import (
    FakeIsolatedClient, _configure_idle_lifecycle, _idle_test_runtime,
    _write_prebuilt_venv_shape,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", ["bin", "python", "pyvenv.cfg"])
@pytest.mark.parametrize("cleanup", ["idle", "offboard"])
async def test_external_venv_internal_artifact_is_not_deleted(tmp_path, monkeypatch, artifact, cleanup):
    agent = Mock(did="did:test:isolated-runtime", features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    workspace = feature._prepare_runtime_workspace()
    external = tmp_path / "external-venv"
    _write_prebuilt_venv_shape(external)
    source = external / ("bin/python" if artifact == "python" else artifact)
    retained = workspace / "operator-artifact"
    source.rename(retained)
    source.symlink_to(retained, target_is_directory=artifact == "bin")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_VENV", str(external))
    feature.resolve_runtime_paths()
    feature._idle_retired, feature._client = True, None
    # The public feature map can be empty after quarantine/offboarding.
    assert agent.features == {}
    if cleanup == "idle":
        assert not feature.runtime_telemetry_snapshot().cleanup_eligible
    with pytest.raises(IsolatedRuntimePreparationError, match="immutable"):
        if cleanup == "idle":
            await feature.reclaim_idle_workspace()
        else:
            remove_agent_runtime_namespace(agent)
    assert retained.exists() and source.resolve() == retained.resolve()
    assert workspace.is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [False, True])
async def test_public_service_stream_retains_creator_before_first_pull(tools):
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.streaming import RoutingResolution
    from tests.unit.test_streaming_usage_metering import _RoutingService
    requests = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=openai_chunk(),
        )))
    client = openai.AsyncOpenAI(api_key="offline-transport-only", http_client=http, max_retries=0)
    adapter = OpenAIAdapter()
    class NativeService(_RoutingService):
        disabled = False
        _check_policy = LLMService._check_policy
        def resolve_provider_routing(self, **kwargs):
            return RoutingResolution(
                [{"name": "openai", "client": client, "adapter": adapter, "is_local": False}],
                "offline", self._compute_route_authorization(**kwargs),
            )
    service = NativeService(adapter)
    original = ExecutionAuthorityError("original creator retired before deferred pull")
    creator, consumer = Fence(), Fence()
    method = service.stream_with_tool_detection if tools else service.stream_with_messages
    creator_scope, consumer_scope = ExecutionCustody(creator), ExecutionCustody(consumer)
    with bind_execution_custody_snapshot((creator_scope,)):
        stream = method(messages=[{"role": "user", "content": "hello"}])
    creator.denial = original
    try:
        with bind_execution_custody_snapshot((consumer_scope,)):
            with pytest.raises(BaseException) as caught:
                await anext(stream)
        assert carries(caught.value, original)
        assert requests == [], "a foreign live consumer spent the retired creator's authority"
    finally:
        await stream.aclose()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk", ["openai", "anthropic"])
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_failed_native_sdk_stream_enter_joins_original_response(tmp_path, sdk, carrier):
    async with aiosqlite.connect(tmp_path / "hidden-response.db") as ledger:
        await ledger.execute("CREATE TABLE effects(id INTEGER)")
        entered, release = asyncio.Event(), asyncio.Event()
        original = OSError("native hidden response close ACK") if carrier == "error" else asyncio.CancelledError("native hidden response close ACK")
        original.__cause__ = ExecutionCommitOutcomeError("unknown")
        class Body(httpx.AsyncByteStream):
            calls = 0
            async def __aiter__(self):
                raise httpx.DecodingError("native malformed error body")
                yield b"unreachable"
            async def aclose(self):
                self.calls += 1
                entered.set()
                await release.wait()
                await ledger.execute("INSERT INTO effects VALUES (1)")
                await ledger.commit()
                raise original
        body = Body()
        http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(400, stream=body)))
        client_class = openai.AsyncOpenAI if sdk == "openai" else anthropic.AsyncAnthropic
        client = client_class(api_key="offline-transport-only", http_client=http, max_retries=0)
        adapter = OpenAIAdapter() if sdk == "openai" else AnthropicAdapter()
        stream = adapter.get_streaming_response(client=client, model="offline", messages=[{"role": "user", "content": "hello"}])
        task = create_owned_task(anext(stream))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "failed SDK acquisition abandoned its physical response"
            release.set()
            outcome = await await_owned_task(task)
            assert carries(outcome.error, original)
            assert body.calls == 1
            assert await (await ledger.execute("SELECT COUNT(*) FROM effects")).fetchone() == (1,)
        finally:
            release.set()
            await await_owned_task(task)
            try:
                await stream.aclose()
            except BaseException as error:
                assert carries(error, original)
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel"])
async def test_standalone_sqlite_commit_keeps_uncertainty(tmp_path, monkeypatch, carrier):
    path = tmp_path / "standalone.db"
    backend = SQLiteBackend(path)
    await backend.connect()
    await backend.execute("CREATE TABLE effects(id INTEGER)")
    native = aiosqlite.Connection.commit
    original = OSError("standalone commit ACK") if carrier == "error" else asyncio.CancelledError("standalone commit ACK")
    async def lost_ack(connection):
        await native(connection)
        if connection is backend._connection:
            raise original
    monkeypatch.setattr(aiosqlite.Connection, "commit", lost_ack)
    try:
        with pytest.raises(BaseException) as caught:
            await backend.execute("INSERT INTO effects VALUES (1)")
        assert execution_commit_outcome(caught.value) == "unknown"
        assert carries(caught.value, original)
        with sqlite3.connect(path) as observer:
            assert observer.execute("SELECT COUNT(*) FROM effects").fetchone() == (1,)
    finally:
        monkeypatch.undo()
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel", "authority", "unknown", "committed", "wrapped", "group"])
async def test_failed_native_tcp_close_retains_exact_owner(monkeypatch, carrier):
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
    connections, close_native = list(pool.connections), pool._close_connections
    original = (OSError("native transport could not retire socket") if carrier == "error"
                else asyncio.CancelledError("native transport close was cancelled"))
    if carrier == "authority":
        original = ExecutionAuthorityError("native close authority lost before socket retirement")
    elif carrier in {"unknown", "committed"}:
        original = ExecutionCommitOutcomeError(carrier)
    elif carrier in {"wrapped", "group"}:
        receipt = ExecutionCommitOutcomeError("unknown")
        if carrier == "wrapped":
            original = OSError("native transport failed before retiring socket")
            original.__cause__ = receipt
        else:
            original = BaseExceptionGroup("native close receipt", [original, receipt])
    calls = []
    async def failed_close(connections):
        calls.append(True)
        raise original
    monkeypatch.setattr(pool, "_close_connections", failed_close)
    service = service_owner()
    service.providers = [{"adapter": client}]
    owner = UnpublishedLLMOwner(service, ())
    source = OriginalFailure.capture(original)
    try:
        with pytest.raises(BaseException) as caught:
            await owner.retire()
        first = service._llm_close_tasks[("adapter", id(client))]
        source.assert_retained(caught.value, resource=client, task=first)
        assert service.has_pending_close_owners() and not peer_done.is_set()
        with pytest.raises(BaseException) as caught:
            await asyncio.wait_for(owner.retire(join_native=True), 2)
        source.assert_retained(caught.value, resource=client, task=first)
        assert service._llm_close_tasks[("adapter", id(client))] is first
        assert calls == [True], "failed physical owner was replaced by a speculative close"
        assert service.has_pending_close_owners()
    finally:
        monkeypatch.undo()
        await close_native(connections)
        await asyncio.wait_for(peer_done.wait(), 2)
        await client.aclose()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "cancel", "group"])
async def test_distributed_completion_cannot_retry_native_control_into_success(tmp_path, monkeypatch, carrier):
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.stop import DistributedInvocationRegistry, DistributedInvocationStore
    from tests.unit.test_distributed_stop_invocations import _ReplicaAgent
    db = await AsyncDatabase.sqlite(str(tmp_path / "distributed-control.db"))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(store, poll_seconds=0.01, owner_lease_seconds=1)
    agent = _ReplicaAgent("did:test:distributed56")
    assert await registry.register(agent, "original-turn", 1)
    ticket = await registry.request_turn(agent.agent_id, "original-turn")
    original = OSError("native DELETE committed acknowledgement") if carrier == "error" else asyncio.CancelledError("native DELETE committed acknowledgement")
    original.__cause__ = ExecutionCommitOutcomeError("unknown")
    if carrier == "group":
        original = BaseExceptionGroup("native settlement", [ValueError("ordinary"), original])
    native_commit, commits = aiosqlite.Connection.commit, []
    connection = db.backend._connection
    async def lost_ack(target):
        await native_commit(target)
        if target is connection and not commits:
            commits.append(True)
            raise original
    monkeypatch.setattr(aiosqlite.Connection, "commit", lost_ack)
    try:
        registry.complete_soon(agent, "original-turn", 1)
        tasks = tuple(registry._cleanup_tasks)
        assert len(tasks) == 1
        outcome = await await_owned_task(tasks[0])
        assert carries(outcome.error, original)
        await asyncio.sleep(0)
        assert not registry._cleanup_tasks
        assert commits == [True]
        assert await db.fetchval("SELECT COUNT(*) FROM stop_active_invocations") == 0
        with pytest.raises(BaseException) as caught:
            await registry.wait_for_stop(ticket)
        assert carries(caught.value, original)
        with pytest.raises(BaseException) as caught:
            await registry.close()
        assert carries(caught.value, original)
    finally:
        monkeypatch.undo()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["cancel", "group"])
async def test_signed_reanchor_reconciles_native_baseexception_receipts(tmp_path, monkeypatch, carrier):
    import hashlib
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from tests.integration.test_constitution_turn_admission import _ready_turn
    from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
    async with AsyncStorage(str(tmp_path / "baseexception-reanchor.db"), agent_id="did:test:reanchor56") as storage:
        agent, content = await _ready_turn(storage)
        replacement = content + b"\nOffline signed review56 amendment.\n"
        path = tmp_path / "KESTREL_CONSTITUTION.md"
        path.write_bytes(replacement)
        import kestrel_sovereign.config as config
        monkeypatch.setattr(config, "CONSTITUTION_PATH", str(path))
        artifact, root = _write_authority_files(tmp_path, replacement)
        agent._sovereign_trust_root_path = root
        original = asyncio.CancelledError("native signed reanchor COMMIT ACK")
        original.__cause__ = ExecutionCommitOutcomeError("unknown")
        if carrier == "group":
            original = BaseExceptionGroup("native signed reanchor", [ValueError("ordinary"), original])
        connection, native = storage.db.backend._connection, aiosqlite.Connection.commit
        commits = []
        async def lost_ack(target):
            await native(target)
            if target is connection and not commits:
                commits.append(True)
                raise original
        monkeypatch.setattr(aiosqlite.Connection, "commit", lost_ack)
        try:
            with pytest.raises(BaseException) as caught:
                await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))
            assert carries(caught.value, original) and commits == [True]
            node = await storage.get_node(storage.agent_id)
            assert node.properties["constitution_hash"] == hashlib.sha256(replacement).hexdigest()
            assert agent._safe_mode is True
            assert "reconciliation" in agent._safe_mode_reason
        finally:
            monkeypatch.undo()
