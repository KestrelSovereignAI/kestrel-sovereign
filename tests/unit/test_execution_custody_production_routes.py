"""Regression proofs for review4's actual production normalization routes."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ExecutionCustody,
    bind_execution_custody,
    execution_commit_outcome,
    require_execution_work,
)
from tests.unit.test_execution_custody import Authority


def uncertain(*args, **kwargs):
    try:
        raise ExecutionCommitOutcomeError("unknown")
    except ExecutionCommitOutcomeError as error:
        raise RuntimeError("native storage wrapper") from error


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_supplied", [False, True])
async def test_a2a_boot_reuses_guarded_native_backend_without_new_pool(monkeypatch, pool_supplied):
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.agent.boot import BootContext
    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    import kestrel_sovereign.kestrel_agent as agent_module

    scope = ExecutionCustody(Authority())
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._execution_custody = scope
    stores = []

    class ReachedStores(Exception):
        pass

    class Manager:
        def __init__(self, **kwargs):
            stores.extend(kwargs[name] for name in (
                "task_store", "session_service", "observability_store",
                "memory_service", "feedback_store",
            ))

        async def initialize(self):
            raise ReachedStores

    monkeypatch.setattr(agent_module, "TaskManager", Manager)
    owner = KestrelAgent.__new__(KestrelAgent)
    owner._db_backend = "postgres"
    owner.pg_pool = object() if pool_supplied else None
    owner._raw_storage = SimpleNamespace(_backend=backend)
    owner.storage_path = ":memory:"
    owner.did = "did:test:guarded-a2a"
    owner.hooks_manager = None
    with pytest.raises(ReachedStores):
        await owner._boot_phase_a2a_observability_signals(BootContext())
    assert len(stores) == 5
    assert all(store.backend is backend for store in stores)
    scope.revoke("original runtime retired")
    for store in stores:
        with pytest.raises(ExecutionAuthorityError, match="original runtime"):
            require_execution_work(store.backend)
        # Store closure must not close the storage-owned backend.
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["inline", "subagent"])
async def test_feature_subagent_preserves_wrapped_commit_uncertainty(boundary):
    from tests.unit.test_feature_subagent_tool_executor import _FakeTool, _make_feature_with_agent_capture
    feature, _ = _make_feature_with_agent_capture([_FakeTool("effect", {})])
    feature._fake_tools[0].execute = AsyncMock(side_effect=uncertain)
    if boundary == "inline":
        operation = lambda: feature._make_feature_inline_tool_executor()("effect", {})
    else:
        feature.agent.llm_service.generate = AsyncMock(side_effect=uncertain)
        operation = lambda: feature.execute_as_subagent(task="effect")
    with pytest.raises(RuntimeError) as caught:
        await operation()
    assert execution_commit_outcome(caught.value) == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["invoke", "send"])
@pytest.mark.parametrize("loss", ["authorization", "client_entry"])
async def test_peer_send_retains_runtime_and_rechecks_actual_rpc(route, loss):
    from kestrel_sovereign.features.peers.directory import LocalHostPeerDirectory, PeerIdentity, PeerRequester

    scope = ExecutionCustody(Authority())
    sends = AsyncMock(return_value=SimpleNamespace(
        status_code=200, raise_for_status=lambda: None, json=lambda: {},
    ))

    @asynccontextmanager
    async def client():
        await asyncio.sleep(0)
        if loss == "client_entry":
            scope.revoke("retired during client entry")
        yield SimpleNamespace(post=sends)

    router = LocalHostPeerDirectory("http://synthetic.test", client_factory=client)
    router._execution_custody = scope
    peer = PeerIdentity("did:test:peer", "peer", "peer", "Peer", "online", "")
    requester = PeerRequester("did:test:source", object())

    async def authorize(*args):
        await asyncio.sleep(0)
        if loss == "authorization":
            scope.revoke("retired during authorization")
        return peer

    router._authorize_peer = authorize
    with pytest.raises(ExecutionAuthorityError, match="retired during"):
        if route == "invoke":
            await router.invoke(requester, peer, "message")
        else:
            await router.send_a2a_task(requester, peer, {})
    sends.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_a2a_operation_binds_original_runtime_through_authorization(monkeypatch):
    from kestrel_sovereign.endpoints import agent as endpoint
    from kestrel_sovereign.execution_custody import current_execution_custody

    scope = ExecutionCustody(Authority())
    owner = SimpleNamespace(_execution_custody=scope)
    committed = []

    async def verified_operation(agent, *args):
        # This continuation represents the asynchronous verifier/authorizer
        # which also writes a replay nonce on native storage.
        assert scope in current_execution_custody()
        await asyncio.sleep(0)
        scope.revoke("retired during inbound authorization")
        require_execution_work()
        committed.append(True)

    monkeypatch.setattr(endpoint, "_create_a2a_task_under_lifecycle_lease", verified_operation)
    with pytest.raises(ExecutionAuthorityError, match="inbound authorization"):
        await endpoint._create_verified_a2a_task(owner, None, None, None, None)
    assert committed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["resolve", "list", "invoke"])
@pytest.mark.parametrize("wrapper", ["runtime", "peer"])
async def test_peer_feature_preserves_control_evidence_at_provider_boundaries(route, wrapper):
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from kestrel_sovereign.features.peers.directory import PeerIdentity, PeerRequester, PeerTransportError

    async def fail(*args):
        try:
            uncertain()
        except RuntimeError as error:
            if wrapper == "peer":
                raise PeerTransportError("transport extension") from error
            raise

    peer = PeerIdentity("did:test:peer", "peer", "peer", "Peer", "online", "")
    owner = SimpleNamespace(did="did:test:source", _agent_name="source")
    feature = PeersFeature(owner)
    router = SimpleNamespace(
        resolve_peer=AsyncMock(return_value=peer),
        list_peers=AsyncMock(return_value=[]),
        invoke=AsyncMock(return_value={}),
    )
    getattr(router, {"resolve": "resolve_peer", "list": "list_peers", "invoke": "invoke"}[route]).side_effect = fail
    feature._peer_router = router
    feature._peer_requester = PeerRequester(owner.did, object())
    with pytest.raises(Exception) as caught:
        if route == "list":
            await feature.list_peers()
        else:
            await feature.ask_agent("peer", "message")
    assert execution_commit_outcome(caught.value) == "unknown"


@pytest.mark.asyncio
async def test_dynamic_tool_binds_original_runtime_for_unbound_provider():
    from kestrel_sovereign.features.base import Feature, tool
    from kestrel_sovereign.execution_custody import current_execution_custody

    scope = ExecutionCustody(Authority())

    class Effects(Feature):
        @property
        def tool_description(self):
            return "synthetic custody tools"

        async def initialize(self):
            pass

        @tool(name="effect", description="synthetic effect")
        async def effect(self):
            assert scope in current_execution_custody()
            return {"success": True}

    feature = Effects(SimpleNamespace(_execution_custody=scope))
    result = await feature.get_tools()[0].execute()
    assert result["success"] is True


@pytest.mark.asyncio
async def test_provider_failure_keeps_unknown_evidence_after_runtime_retirement():
    from kestrel_sovereign.execution_custody import await_execution_work
    scope = ExecutionCustody(Authority())

    async def fail():
        scope.revoke("runtime retired after effect")
        uncertain()

    with pytest.raises(RuntimeError) as caught:
        await await_execution_work(SimpleNamespace(_execution_custody=scope), fail)
    assert execution_commit_outcome(caught.value) == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["dynamic", "direct"])
async def test_native_tool_wrapped_uncertainty_is_not_retryable_envelope(boundary):
    from kestrel_sovereign.features.base import Feature, tool
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
    from kestrel_sovereign.hooks.manager import HooksManager

    class Effects(Feature):
        async def initialize(self):
            pass

        @property
        def tool_description(self):
            return "synthetic effect"

        @tool(name="effect", description="synthetic effect")
        async def effect(self):
            uncertain()

    feature = Effects.__new__(Effects)
    feature.agent = None
    feature.disabled_skills = set()
    native_tool = next(t for t in feature.get_tools() if t.name == "effect")

    class Agent(OrchestratorEngineMixin):
        hooks_manager = HooksManager()
        _direct_tools = {"effect": native_tool}
        observability_store = SimpleNamespace(log_tool_response=AsyncMock())

    if boundary == "dynamic":
        operation = native_tool.execute
    else:
        operation = lambda: Agent()._dispatch_direct_tool(
            None, "effect", {}, 0, "event", tool=native_tool, feature_name="Effects"
        )
    with pytest.raises(RuntimeError) as caught:
        await operation()
    assert execution_commit_outcome(caught.value) == "unknown"


@pytest.mark.asyncio
async def test_readiness_denial_unwinds_committed_boot_phases():
    from kestrel_sovereign.agent.boot import (
        BootContext,
        BootPhase,
        BootPhaseState,
        run_boot_sequence,
    )
    from kestrel_sovereign.agent.custody import ReleaseOutcome

    states, released = [], []
    custody = ExecutionCustody(Authority())

    async def undo():
        released.append(True)
        return ReleaseOutcome.RELEASED

    async def phase(ctx):
        ctx.on_rollback("owned fixture", undo)
        custody.revoke("last boot await lost authority")

    def set_state(state):
        if state is BootPhaseState.READY:
            custody.require_work()
        states.append(state)

    with pytest.raises(ExecutionAuthorityError, match="last boot await"):
        await run_boot_sequence([BootPhase("last", phase)], BootContext(), set_state)
    assert states == [BootPhaseState.IN_PROGRESS, BootPhaseState.FAILED]
    assert released == [True]


@pytest.mark.asyncio
async def test_stream_close_uncertainty_preserves_abandoned_settlement():
    from kestrel_sovereign.agent.invocation import bind_async_generator_invocation
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition

    dispositions = []

    class Owner:
        def register_active_request(self, request_id, **kwargs):
            pass

        async def await_durable_request_admission(self, request_id):
            return True

        def _cleanup_cancelled_request(self, request_id, **kwargs):
            dispositions.append(kwargs.get("disposition"))

        @bind_async_generator_invocation("request_id", track_request_lifecycle=True)
        async def stream(self, request_id=None):
            try:
                yield "first"
            finally:
                uncertain()

    stream = Owner().stream(request_id="fixture")
    assert await anext(stream) == "first"
    with pytest.raises(RuntimeError):
        await stream.aclose()
    assert dispositions == [RequestCompletionDisposition.ABANDONED]


@pytest.mark.asyncio
async def test_codex_inline_uncertainty_latches_turn_and_reaches_owner():
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter
    from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
    from tests.unit.test_codex_app_server import TestDispatchLogic

    client = TestDispatchLogic()._client()
    adapter = CodexAdapter.__new__(CodexAdapter)
    effects = []

    async def execute(*args):
        effects.append(True)
        uncertain()

    handler = adapter._make_tool_call_handler(execute, "turn", frozenset({"effect"}))
    sink = client.open_turn_sink("turn")
    client.register_server_request_handler("item/tool/call", handler, thread_id="turn")
    params = {"threadId": "turn", "tool": "effect", "arguments": {}}
    await client._handle_server_request(1, "item/tool/call", params)
    with pytest.raises(RuntimeError) as caught:
        await anext(client.iter_turn_events(sink, thread_id="turn"))
    assert execution_commit_outcome(caught.value) == "unknown"
    with pytest.raises(RuntimeError):
        await handler(params)
    assert effects == [True]
    assert (
        client._sent[0]["error"]["message"]
        == "execution requires authority/commit reconciliation"
    )


@pytest.mark.asyncio
async def test_codex_send_rechecks_after_startup_await():
    from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
    from tests.unit.test_codex_app_server import TestDispatchLogic

    client = TestDispatchLogic()._client()
    client._next_id = 1

    async def start():
        await asyncio.sleep(0)
        scope.revoke("retired during startup")

    client.ensure_started = start
    with bind_execution_custody(Authority()) as scope:
        with pytest.raises(ExecutionAuthorityError, match="during startup"):
            await client.request("turn/start", {"threadId": "turn"})
    assert client._sent == []
    assert client._pending == {}


@pytest.mark.asyncio
async def test_codex_shutdown_retains_reader_normalized_terminal_error():
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter
    from tests.unit.test_codex_app_server import TestDispatchLogic

    started, release, sink_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    app = TestDispatchLogic()._client()
    adapter = CodexAdapter()
    adapter._client = app
    readers = []
    original_close = app.close_turn_sink

    def close(key):
        sink_closed.set()
        original_close(key)

    app.close_turn_sink = close

    async def request(method, params=None, **kwargs):
        if method == "thread/start":
            return {"thread": {"id": "synthetic-thread"}}
        if method == "turn/start":
            readers.append(asyncio.create_task(app._handle_server_request(
                1, "item/tool/call", {"threadId": "synthetic-thread", "tool": "effect", "arguments": {}},
            )))
            await started.wait()
            return {"turn": {"id": "synthetic-turn"}}
        return {}

    async def events(*args, **kwargs):
        await asyncio.Future()
        if False:
            yield {}

    async def execute(*args):
        started.set()
        await release.wait()
        uncertain()

    app.request = request
    app.ensure_started = AsyncMock()
    app.iter_turn_events = events
    operation = asyncio.create_task(adapter.get_response(
        None, "auto", [{"role": "user", "content": "effect"}],
        tools=[{"type": "function", "function": {"name": "effect", "description": "synthetic", "parameters": {"type": "object"}}}],
        tool_executor=execute,
    ))
    try:
        await asyncio.wait_for(started.wait(), 2)
        operation.cancel()
        await asyncio.sleep(0.01)
        assert not operation.done()
        assert not sink_closed.is_set()
        release.set()
        with pytest.raises(BaseException) as caught:
            await operation
        assert execution_commit_outcome(caught.value) == "unknown"
        assert sink_closed.is_set()
        await asyncio.gather(*readers)
        assert all(not lock.locked() for lock in adapter._thread_locks.values())
    finally:
        release.set()
        await asyncio.gather(operation, *readers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["plain", "tools"])
async def test_public_codex_stream_joins_every_forwarder_on_close(surface):
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    adapter = CodexAdapter.__new__(CodexAdapter)
    started, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def turn(*args, **kwargs):
        try:
            yield {"text": "first"}
        finally:
            started.set()
            await release.wait()
            closed.set()
            uncertain()

    adapter._run_turn = turn
    method = adapter.get_streaming_response if surface == "plain" else adapter.get_streaming_response_with_tools
    stream = method(None, "synthetic", [], tools=[{"name": "effect"}] if surface == "tools" else None)
    try:
        assert await anext(stream) == "first"
        closer = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(started.wait(), 1)
        assert not closer.done()
        closer.cancel()
        await asyncio.sleep(0)
        assert not closer.done()
        release.set()
        with pytest.raises(BaseException) as caught:
            await closer
        assert execution_commit_outcome(caught.value) == "unknown"
        assert closed.is_set()
    finally:
        release.set()
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
async def test_hosted_modality_writer_is_joined_before_invocation_settles(monkeypatch, interruption):
    from kestrel_sovereign.llm import modality_recording as recording
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext

    started, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Recorder(recording.ModalityRecordingMixin):
        async def _write_modality_record(self, call):
            started.set()
            try:
                await asyncio.Future()
            finally:
                await release.wait()
                closed.set()
                uncertain()

    recorder = Recorder()
    recorder._execution_custody = ExecutionCustody(Authority())
    monkeypatch.setattr(recording, "USAGE_RECORD_TIMEOUT", 0.01)
    call = recording.ModalityCall("embedding", "local-test", "synthetic", 1, True, LLMInvocationContext())
    operation = asyncio.create_task(recorder.record_modality_call(call))
    try:
        await started.wait()
        if interruption == "cancel":
            operation.cancel()
        await asyncio.sleep(0.03)
        assert not operation.done()
        release.set()
        with pytest.raises(BaseException) as caught:
            await operation
        assert execution_commit_outcome(caught.value) == "unknown"
        assert closed.is_set()
        assert not recorder._pending_modality_records()
    finally:
        release.set()
        for task in tuple(recorder._pending_modality_records()):
            task.cancel()
        await asyncio.gather(operation, *recorder._pending_modality_records(), return_exceptions=True)


@pytest.mark.asyncio
async def test_interactive_isolated_rpc_rechecks_after_traffic_gate():
    from kestrel_sovereign.features.isolated_runtime import ProxyFeature

    custody = ExecutionCustody(Authority())
    proxy = ProxyFeature.__new__(ProxyFeature)
    proxy.agent = SimpleNamespace(_execution_custody=custody)
    effects = AsyncMock(return_value={"status": "ok"})
    proxy._client = SimpleNamespace(call_tool=effects)
    proxy._config_fence_allows_live_turn_tool = False
    proxy._record_runtime_activity = lambda: None

    @asynccontextmanager
    async def gate(**kwargs):
        await asyncio.sleep(0)
        custody.revoke("retired while waiting for traffic")
        yield

    proxy._traffic_gate = SimpleNamespace(admit=gate)
    with pytest.raises(ExecutionAuthorityError, match="waiting for traffic"):
        await proxy.call_isolated_tool("effect", {})
    effects.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["embedding", "decision"])
async def test_real_modality_writer_cannot_account_after_runtime_loss(modality):
    from kestrel_sovereign.llm.service import LLMService
    from tests.unit.test_embedding_usage_recording import (
        UsageReportingAdapter,
        _provider,
    )
    from tests.unit.test_decisions_service import (
        FakeDecisionAdapter,
        _service,
        _route,
        _request,
        _answer_body,
    )

    custody = ExecutionCustody(Authority())
    if modality == "decision":

        class Adapter(FakeDecisionAdapter):
            async def adecide(self, *args, **kwargs):
                custody.revoke("provider retired its runtime")
                return _answer_body(input_tokens=42)

        service = _service([_route("ollama:local", Adapter(["test"]), local=True)])
        operation = lambda: service.decide(_request(), caller="test", timeout_seconds=5)
    else:
        adapter = UsageReportingAdapter()
        adapter.during_call = lambda: custody.revoke("provider retired its runtime")
        service = LLMService.__new__(LLMService)
        from kestrel_sovereign.llm.invocation_context import LLMInvocationContext

        service.snapshot_invocation_context = lambda: LLMInvocationContext()
        embedding = service._new_embedding_service(_provider(adapter))
        operation = lambda: embedding.aembed("synthetic fixture")
    service._execution_custody = custody
    service._record_model_usage = AsyncMock()
    service._log_llm_call = AsyncMock()
    with pytest.raises(ExecutionAuthorityError):
        await operation()
    service._record_model_usage.assert_not_awaited()
    service._log_llm_call.assert_not_awaited()
    assert not service.__dict__.get("_modality_record_tasks")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["stream_with_messages", "stream_with_tool_detection"]
)
async def test_stream_provider_cannot_normalize_wrapped_commit_error(
    route, monkeypatch
):
    from tests.unit.test_streaming_usage_metering import _RoutingService

    class Adapter:
        async def get_streaming_response_with_tools(self, **kwargs):
            uncertain()
            yield "unreachable"

    service = _RoutingService(Adapter())
    disabled = []
    service._maybe_disable_route = lambda *args: disabled.append(True)
    monkeypatch.setattr(
        "kestrel_sovereign.llm.streaming.provider_cache_body", lambda _: None
    )
    stream = getattr(service, route)(messages=[{"role": "user", "content": "fixture"}])
    with pytest.raises(RuntimeError) as caught:
        await anext(stream)
    assert execution_commit_outcome(caught.value) == "unknown"
    assert not disabled


@pytest.mark.asyncio
async def test_native_stream_close_joins_adapter_cleanup_before_return():
    from tests.unit.test_streaming_usage_metering import _FakeService

    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    service = _FakeService()
    service._execution_custody = ExecutionCustody(Authority())

    class Adapter:
        async def get_streaming_response(self, **kwargs):
            try:
                yield "chunk"
                yield "later"
            finally:
                with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
                    require_execution_work()
                entered.set()
                await release.wait()
                closed.set()

    stream = service._stream_adapter_with_usage(
        adapter=Adapter(),
        client=None,
        model="test",
        messages=[],
        provider_name="local",
        path="test",
        invocation_context=None,
        expose_protocol_events=False,
    )
    assert await anext(stream) == "chunk"
    closer = asyncio.create_task(stream.aclose())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert not closer.done()
        release.set()
        await closer
        assert closed.is_set()
    finally:
        release.set()
        await asyncio.gather(closer, return_exceptions=True)


@pytest.mark.asyncio
async def test_top_level_stream_checks_authority_on_final_eof():
    from kestrel_sovereign.agent.invocation import bind_async_generator_invocation

    class Owner:
        _execution_custody = ExecutionCustody(Authority())

        @bind_async_generator_invocation("request_id")
        async def stream(self, request_id=None):
            yield "first"
            await asyncio.sleep(0)
            self._execution_custody.revoke("retired during final hook")

    owner = Owner()
    stream = owner.stream(request_id="test")
    assert await anext(stream) == "first"
    with pytest.raises(ExecutionAuthorityError, match="final hook"):
        await anext(stream)


@pytest.mark.asyncio
async def test_deferred_readiness_keeps_retained_scope_and_rejects_hook_loss():
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.execution_custody import current_execution_custody

    agent = KestrelAgent.__new__(KestrelAgent)
    agent._execution_custody = ExecutionCustody(Authority())
    agent._agent_ready_hooks_deferred = True
    agent._agent_ready_hooks_completed = False
    agent._agent_readiness_host_owned = True

    async def ready(owner):
        assert current_execution_custody() == (agent._execution_custody,)
        await asyncio.sleep(0)
        agent._execution_custody.revoke("ready hook lost authority")

    agent.features = {"test": SimpleNamespace(on_agent_ready=ready)}
    with pytest.raises(ExecutionAuthorityError, match="ready hook"):
        await agent.complete_deferred_agent_readiness()
    assert agent._agent_ready_hooks_deferred
    assert not agent._agent_ready_hooks_completed


@pytest.mark.asyncio
@pytest.mark.parametrize("hosted", [False, True])
async def test_public_stream_rename_retains_privacy_lock_reentry(monkeypatch, hosted):
    from tests.unit.test_streaming_audit import _make_streaming_audit_agent
    from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock
    from kestrel_sovereign.features.bootstrap.feature import rename_agent_core
    import kestrel_sovereign.features.storage_access as storage_access

    agent, _ = _make_streaming_audit_agent([], mode="warn", register_hook=False)
    lock = ReentrantTransitionLock()
    agent._get_privacy_transition_lock.return_value = lock
    agent._agent_name = "Before"
    if hosted:
        agent._execution_custody = ExecutionCustody(Authority())
    monkeypatch.setattr(storage_access, "hides_persisted_user_content", lambda owner: True)

    async def body(*args, **kwargs):
        outcome = await rename_agent_core(agent, "After")
        assert outcome.skipped_privacy
        yield "renamed"

    agent._process_input_streaming_traced_locked = body
    stream = agent.process_input_streaming("rename")
    try:
        assert await asyncio.wait_for(anext(stream), timeout=0.5) == "renamed"
        assert agent._agent_name == "After"
    finally:
        await stream.aclose()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_advancing_stream_close_denies_source_finalizer_and_its_child():
    from kestrel_sovereign.execution_custody import owned_execution_stream
    owner = SimpleNamespace(_execution_custody=ExecutionCustody(Authority()))
    advancing = asyncio.Event()
    closed = []

    async def source():
        try:
            advancing.set()
            await asyncio.Event().wait()
            yield "never"
        finally:
            for check in (lambda: require_execution_work(owner), require_execution_work):
                with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
                    check()
            async def child():
                with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
                    require_execution_work()
            await asyncio.create_task(child())
            closed.append(True)

    async with owned_execution_stream(owner, source()) as stream:
        advance = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(advancing.wait(), timeout=1)
            await stream.aclose()
            await asyncio.gather(advance, return_exceptions=True)
        finally:
            if not advance.done():
                advance.cancel()
                await asyncio.gather(advance, return_exceptions=True)
    assert closed == [True]
    require_execution_work(owner)  # Closing this stream did not revoke the runtime.


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [False, True])
async def test_class_provider_attempt_preserves_wrapped_control_before_finalization(retired):
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
    service = LLMService.__new__(LLMService)
    service._execution_custody = ExecutionCustody(Authority())
    service._finalize_failed_invocation = AsyncMock()

    async def provider():
        if retired:
            service._execution_custody.revoke("provider owner retired")
        uncertain()

    with pytest.raises(RuntimeError) as caught:
        await service._run_provider_attempt(provider(), "test", "test", path="test", invocation_context=LLMInvocationContext())
    assert execution_commit_outcome(caught.value) == "unknown"
    service._finalize_failed_invocation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("normalizer", ["classify", "retry", "fallback"])
async def test_provider_error_decorators_do_not_normalize_or_retry_control(normalizer):
    from kestrel_sovereign.llm.error_handling import handle_llm_errors, with_retry, handle_provider_fallback
    calls = []
    async def provider(*args, **kwargs):
        calls.append(True)
        uncertain()
    decorators = {"classify": handle_llm_errors(), "retry": with_retry(delay=0), "fallback": handle_provider_fallback(["one", "two"])}
    with pytest.raises(RuntimeError) as caught:
        await decorators[normalizer](provider)()
    assert execution_commit_outcome(caught.value) == "unknown"
    assert calls == [True]
