"""Round7: real outer consumers must preserve inner custody evidence."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionCustody,
    execution_commit_outcome,
)
from tests.unit.test_execution_custody_production_routes import Authority


def wrapped(error_type=RuntimeError):
    error = error_type("503 unavailable after effect")
    error.__cause__ = ExecutionCommitOutcomeError("unknown")
    return error


@pytest.mark.asyncio
async def test_adapter_retry_does_not_redispatch_wrapped_unknown():
    from kestrel_sovereign.llm.retry import with_retry
    error = wrapped()
    provider = AsyncMock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        await with_retry(provider, max_retries=2, base_delay=0.001, max_delay=1)
    assert caught.value is error
    assert execution_commit_outcome(caught.value) == "unknown"
    provider.assert_awaited_once()


@pytest.mark.asyncio
async def test_public_stream_source_receives_the_requested_session():
    from tests.unit.test_turn_outcome_spans import _real_agent, _drain
    from kestrel_sovereign.agent.turn_lifecycle import capture_turn_session_binding
    agent = _real_agent()
    sessions = []

    async def body(*_a, **_k):
        sessions.append(capture_turn_session_binding(agent).session_id)
        yield "tick"

    agent._process_input_streaming_traced_locked = body
    session = "9dfce7ad-86bb-47c6-ac38-b75a76c1d5a5"
    assert await _drain(agent.process_input_streaming("test", session_id=session)) == ["tick"]
    assert sessions == [session]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["discovery", "canary_transport", "canary_protocol", "timeout"])
async def test_decision_boundaries_preserve_wrapped_controls(phase):
    from tests.unit.test_decisions_service import FakeDecisionAdapter, _service, _route, _request
    from kestrel_sdk.llm.decisions import DecisionTransportError, DecisionProtocolError
    error = wrapped({"canary_transport": DecisionTransportError,
                     "canary_protocol": DecisionProtocolError,
                     "timeout": TimeoutError}.get(phase, RuntimeError))
    adapter = FakeDecisionAdapter(["test"],
        discovery_error=error if phase == "discovery" else None,
        error=error if phase != "discovery" else None)
    route = _route("ollama:local", adapter, local=True,
                   pin="test" if phase != "timeout" else None)
    service = _service([route])
    with pytest.raises(Exception) as caught:
        if phase == "timeout":
            await service.decide(_request(), caller="test", timeout_seconds=5)
        else:
            await service._discover_decision_route(route, force=True)
    assert caught.value is error
    assert execution_commit_outcome(caught.value) == "unknown"
    assert len(adapter.decide_calls) == (0 if phase == "discovery" else 1)
    service._track_model_usage.assert_not_awaited()
    service._observability_store.log_llm_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["decision", "embedding"])
async def test_ordinary_hosted_modality_failure_keeps_real_accounting(modality):
    from tests.unit.test_decisions_service import FakeDecisionAdapter, _service, _route, _request, _logged
    from tests.unit.test_embedding_usage_recording import UsageReportingAdapter, _provider
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
    if modality == "decision":
        adapter = FakeDecisionAdapter(["test"], body={"invalid": True, "usage": {"input_tokens": 23}})
        service = _service([_route("ollama:local", adapter, local=True)])
        operation = lambda: service.decide(_request(), caller="test", timeout_seconds=5)
    else:
        class FailingAdapter(UsageReportingAdapter):
            async def _respond(self, payload, usage_sink):
                if usage_sink is not None:
                    usage_sink.add(input_tokens=23)
                raise ValueError("ordinary provider failure")
        service = LLMService.__new__(LLMService)
        service.snapshot_invocation_context = lambda: LLMInvocationContext()
        service._record_model_usage = AsyncMock()
        service._log_llm_call = AsyncMock()
        embedding = service._new_embedding_service(_provider(FailingAdapter()))
        operation = lambda: embedding.aembed("synthetic fixture")
    service._execution_custody = ExecutionCustody(Authority())
    with pytest.raises(Exception) as caught:
        await operation()
    assert execution_commit_outcome(caught.value) is None
    if modality == "decision":
        [row] = _logged(service)
        assert not row["success"]
        service._track_model_usage.assert_awaited_once_with("test", "ollama:local", tokens=23)
    else:
        service._log_llm_call.assert_awaited_once()
        assert not service._log_llm_call.await_args.kwargs["success"]
        service._record_model_usage.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["body", "close"])
async def test_stream_checkpoint_denial_cannot_replace_unknown(phase):
    from kestrel_sovereign.agent.invocation import bind_async_generator_invocation, mark_current_invocation_effect_completed
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    dispositions = []
    error = wrapped()

    class Owner:
        _execution_custody = ExecutionCustody(Authority())
        def register_active_request(self, request_id):
            pass
        async def await_durable_request_admission(self, request_id):
            return True
        def _cleanup_cancelled_request(self, request_id, **kwargs):
            dispositions.append(kwargs.get("disposition"))
        async def _persist_completed_tool_stop_checkpoint(self, **kwargs):
            raise ExecutionAuthorityError("checkpoint cleanup denied")
        @bind_async_generator_invocation("request_id", track_request_lifecycle=True)
        async def stream(self, request_id=None):
            mark_current_invocation_effect_completed("session")
            if phase == "body":
                raise error
            try:
                yield "tick"
            finally:
                raise error

    stream = Owner().stream(request_id="uncertain-stream")
    with pytest.raises(RuntimeError) as caught:
        if phase == "body":
            await anext(stream)
        else:
            assert await anext(stream) == "tick"
            await stream.aclose()
    assert caught.value is error
    assert execution_commit_outcome(caught.value) == "unknown"
    assert dispositions == [RequestCompletionDisposition.ABANDONED]


@pytest.mark.asyncio
async def test_unbound_embedding_control_does_not_start_ordinary_accounting():
    from tests.unit.test_embedding_usage_recording import UsageReportingAdapter, _provider
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
    service = LLMService.__new__(LLMService)
    service.snapshot_invocation_context = lambda: LLMInvocationContext()
    service._record_model_usage = AsyncMock()
    service._log_llm_call = AsyncMock()
    error = wrapped()
    embedding = service._new_embedding_service(_provider(UsageReportingAdapter(error=error)))
    with pytest.raises(RuntimeError) as caught:
        await embedding.aembed("synthetic fixture")
    assert caught.value is error
    service._record_model_usage.assert_not_awaited()
    service._log_llm_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["decision", "embedding"])
async def test_failed_provider_accounting_cannot_hide_unknown(modality):
    from tests.unit.test_decisions_service import FakeDecisionAdapter, _service, _route, _request
    from tests.unit.test_embedding_usage_recording import UsageReportingAdapter, _provider
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
    if modality == "decision":
        service = _service([_route("ollama:local", FakeDecisionAdapter(["test"], error=ValueError("provider failed")), local=True)])
        operation = lambda: service.decide(_request(), caller="test", timeout_seconds=5)
    else:
        service = LLMService.__new__(LLMService)
        service.snapshot_invocation_context = lambda: LLMInvocationContext()
        embedding = service._new_embedding_service(_provider(UsageReportingAdapter(error=ValueError("provider failed"))))
        operation = lambda: embedding.aembed("synthetic fixture")
    service._execution_custody = ExecutionCustody(Authority())
    # Drive the real guarded recorder/writer; only the terminal sink fails.
    service._write_modality_record = AsyncMock(side_effect=ExecutionCommitOutcomeError("unknown"))
    with pytest.raises(Exception) as caught:
        await operation()
    assert execution_commit_outcome(caught.value) == "unknown"
    service._write_modality_record.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["named", "chat"])
async def test_named_subagent_outer_boundary_keeps_unknown(boundary):
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
    from kestrel_sovereign.hooks.manager import HooksManager
    error = wrapped()

    class Agent(OrchestratorEngineMixin):
        hooks_manager = HooksManager()
        _get_denied_tools = AsyncMock(return_value=set())
        _fire_post_subagent_hook = AsyncMock()
        observability_store = SimpleNamespace(log_tool_response=AsyncMock())

    feature = SimpleNamespace(execute_as_subagent=AsyncMock(side_effect=error), tool_name="test")
    agent = Agent()
    with pytest.raises(RuntimeError) as caught:
        if boundary == "named":
            await agent._execute_named_subagent(feature, tool_name="test", args={"task": "effect"}, session_id="session", source="test")
        else:
            await agent._dispatch_feature_tool(SimpleNamespace(name="test"), feature,
                {"task": "effect"}, 0, "event", None, session_id="session")
    assert caught.value is error
    assert execution_commit_outcome(caught.value) == "unknown"
    agent._fire_post_subagent_hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_isolated_failed_wake_does_not_flatten_unknown(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from tests.unit.test_isolated_feature_runtime import (
        _TEST_AGENT_DID, _configure_idle_lifecycle, _idle_test_runtime,
        ProxyFeature, FakeIsolatedClient,
    )
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    # Real immutable operator executable; FakeIsolatedClient never launches it.
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/echo")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    try:
        feature._last_used_monotonic -= 7200
        assert await feature._retire_idle_generation(
            expected_activity_generation=feature._activity_generation,
            expected_last_used=feature._last_used_monotonic,
        )
        monkeypatch.setattr(feature, "_prepare_runtime_workspace", Mock(side_effect=wrapped()))
        with pytest.raises(Exception) as caught:
            await feature.call_isolated_tool("ping", {"message": "wake"})
        assert execution_commit_outcome(caught.value) == "unknown"
        assert feature._idle_retired
    finally:
        await feature.shutdown()
