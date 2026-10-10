"""Provider-free probes of review11's remaining production consumers."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionCustody, bind_execution_custody,
    require_execution_work,
)
from tests.unit.test_execution_custody import Authority
from tests.unit.test_execution_custody_review10 import runtime_owner
from tests.unit.test_streaming_usage_metering import _FakeService


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
async def test_isolated_invocation_preserves_cancel_carried_commit_evidence(outcome):
    from kestrel_sovereign.agent.invocation import bind_async_invocation
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    dispositions = []
    error = asyncio.CancelledError("effect acknowledgement raced Stop")
    error.__cause__ = ExecutionCommitOutcomeError(outcome)

    class Owner:
        def register_active_request(self, request_id, *, nested=False):
            pass
        async def await_durable_request_admission(self, request_id):
            return True
        def bind_request_operation(self, request_id, operation):
            pass
        def _cleanup_cancelled_request(self, request_id, **kwargs):
            dispositions.append(kwargs.get("disposition"))
        @bind_async_invocation("request_id", track_request_lifecycle=True)
        async def turn(self, request_id=None):
            raise error

    with pytest.raises(asyncio.CancelledError) as caught:
        await Owner().turn(request_id="irreversible")
    assert caught.value is error
    assert dispositions == [RequestCompletionDisposition.ABANDONED]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("partial", [False, True])
async def test_unbound_stream_control_bypasses_ordinary_accounting(outcome, carrier, partial):
    service = _FakeService()
    error = ExecutionAuthorityError("denied") if outcome == "denied" else ExecutionCommitOutcomeError(outcome)
    if carrier != "direct":
        wrapper = asyncio.CancelledError("Stop") if carrier == "cancel" else RuntimeError("wrapper")
        wrapper.__cause__ = error
        error = wrapper

    class Adapter:
        supports_partial_usage_flush = True
        async def get_streaming_response_with_tools(self, *, usage_sink, **kwargs):
            if partial:
                usage_sink["input_tokens"] = 23
                yield "partial"
            raise error
            yield

    stream = service._stream_adapter_with_usage(
        adapter=Adapter(), client=None, model="synthetic", messages=[],
        provider_name="local-test", path="stream", invocation_context=None,
        expose_protocol_events=False,
    )
    try:
        with pytest.raises(type(error)) as caught:
            async for _ in stream:
                pass
        assert caught.value is error
        service._log_llm_call.assert_not_awaited()
        service._track_model_usage.assert_not_awaited()
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_ordinary_live_stream_failure_keeps_native_accounting(partial):
    service = _FakeService()
    service._execution_custody = ExecutionCustody(Authority())
    error = ValueError("ordinary provider failure with live authority")

    class Adapter:
        supports_partial_usage_flush = True

        async def get_streaming_response_with_tools(self, *, usage_sink, **kwargs):
            if partial:
                usage_sink["input_tokens"] = 23
                yield "partial"
            raise error
            yield

    stream = service._stream_adapter_with_usage(
        adapter=Adapter(), client=None, model="synthetic", messages=[],
        provider_name="local-test", path="stream", invocation_context=None,
        expose_protocol_events=False,
    )
    try:
        with pytest.raises(ValueError) as caught:
            async for _ in stream:
                pass
        assert caught.value is error
        service._log_llm_call.assert_awaited_once()
        if partial:
            service._track_model_usage.assert_awaited_once()
        require_execution_work(service)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_boot_peer_replay_and_sweep_survive_only_original_runtime():
    from kestrel_sovereign.features.peers.feature import PeersFeature
    owner = runtime_owner()
    observed = asyncio.Event()
    release = asyncio.Event()
    row = SimpleNamespace(
        task_id="restored", recipient="peer", original_question="question",
        origin_session_id="session", recipient_agent_id="did:test:peer",
        deadline=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        retry_state=None,
    )
    store = SimpleNamespace(
        list_waiting=AsyncMock(return_value=[row]),
        list_waiting_past_deadline=AsyncMock(
            side_effect=lambda: require_execution_work(owner) or [],
        ),
    )
    owner.pending_a2a_questions = store
    feature = PeersFeature(owner)
    feature._peer_directory_context = lambda: object()
    feature.EXPIRY_SWEEP_INTERVAL_SECONDS = 0.001

    async def supervisor(**kwargs):
        await owner._runtime_publication_ready.wait()
        require_execution_work(owner)
        observed.set()
        await release.wait()
        require_execution_work(owner)

    feature._supervise_a2a_question = supervisor
    try:
        with bind_execution_custody(Authority()):
            await feature.post_all_features_loaded(owner)
        owner._runtime_publication_ready.set()
        for _ in range(100):
            await asyncio.sleep(0.001)
            if observed.is_set() and store.list_waiting_past_deadline.await_count:
                break
        assert observed.is_set(), "restored peer inherited retired cold-boot claim"
        assert store.list_waiting_past_deadline.await_count, "sweep inherited boot claim"
        tasks = tuple(owner._background_tasks)
        assert len(tasks) == 2
        owner._execution_custody.revoke("original runtime replaced")
        release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
        assert all(isinstance(error, ExecutionAuthorityError) for error in outcomes)
        assert not feature._owned_background_tasks
    finally:
        release.set()
        await feature._cancel_owned_background_tasks()
        tasks = tuple(owner._background_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
