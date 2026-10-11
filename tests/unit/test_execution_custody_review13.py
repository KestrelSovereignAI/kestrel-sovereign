"""Actual parallel dispatch must retain every joined terminal-control carrier."""

import asyncio

import pytest

from kestrel_sovereign.agent.invocation import bind_async_invocation, bind_async_generator_invocation
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, execution_commit_outcome,
)
from tests.unit.test_tool_concurrency import FakeTool, FakeToolCall


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["cancel", "ordinary"])
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
async def test_parallel_late_control_survives_actual_invocation(first, control, carrier):
    started = asyncio.Event()
    joined = asyncio.Event()
    dispositions = []
    evidence = (
        ExecutionAuthorityError("late original authority denied")
        if control == "denied" else ExecutionCommitOutcomeError(control)
    )
    expected = evidence
    if carrier != "direct":
        expected = (
            asyncio.CancelledError("late cancelled carrier")
            if carrier == "cancel" else RuntimeError("late wrapped carrier")
        )
        expected.__cause__ = evidence

    class Agent(OrchestratorEngineMixin):
        _direct_tools = {"first": FakeTool(safe=True), "late": FakeTool(safe=True)}
        _tool_to_feature = {}
        features = {}

        def register_active_request(self, request_id, *, nested=False):
            pass

        async def await_durable_request_admission(self, request_id):
            return True

        def bind_request_operation(self, request_id, operation):
            pass

        def _cleanup_cancelled_request(self, request_id, **kwargs):
            dispositions.append(kwargs.get("disposition"))

        async def _dispatch_tool_call(self, call, *args, **kwargs):
            if call.name == "first":
                await started.wait()
                if first == "cancel":
                    raise asyncio.CancelledError("ordinary first cancellation")
                raise ValueError("ordinary first failure")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                joined.set()
                raise expected

        @bind_async_invocation("request_id", track_request_lifecycle=True)
        async def turn(self, request_id=None):
            await self._execute_tool_batch(
                [FakeToolCall("first", "first", {}), FakeToolCall("late", "late", {})],
                {}, set(), [], 0, "provider-free probe",
            )

    with pytest.raises(type(expected)) as caught:
        await Agent().turn(request_id="parallel-control")
    assert caught.value is expected
    assert joined.is_set(), "parallel sibling was not joined"
    assert execution_commit_outcome(caught.value) == (None if control == "denied" else control)
    assert dispositions == [RequestCompletionDisposition.ABANDONED]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["advance", "close"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
async def test_stream_authority_denial_never_acknowledges_request(boundary, carrier):
    dispositions = []
    evidence = ExecutionAuthorityError("original stream authority denied")
    expected = evidence
    if carrier != "direct":
        expected = asyncio.CancelledError("cancelled stream") if carrier == "cancel" else RuntimeError("wrapped stream")
        expected.__cause__ = evidence

    class Owner:
        def register_active_request(self, request_id, *, nested=False):
            pass

        async def await_durable_request_admission(self, request_id):
            return True

        def _cleanup_cancelled_request(self, request_id, **kwargs):
            dispositions.append(kwargs.get("disposition"))

        @bind_async_generator_invocation("request_id", track_request_lifecycle=True)
        async def stream(self, request_id=None):
            if boundary == "advance":
                raise expected
            try:
                yield "already-disclosed"
            finally:
                raise expected

    stream = Owner().stream(request_id="original-stream")
    if boundary == "close":
        assert await anext(stream) == "already-disclosed"
    with pytest.raises(type(expected)) as caught:
        if boundary == "advance":
            await anext(stream)
        else:
            await stream.aclose()
    assert caught.value is expected
    assert dispositions == [RequestCompletionDisposition.ABANDONED]
