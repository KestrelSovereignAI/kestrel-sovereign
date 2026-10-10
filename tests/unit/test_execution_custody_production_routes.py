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


def uncertain():
    try:
        raise ExecutionCommitOutcomeError("unknown")
    except ExecutionCommitOutcomeError as error:
        raise RuntimeError("native storage wrapper") from error


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
