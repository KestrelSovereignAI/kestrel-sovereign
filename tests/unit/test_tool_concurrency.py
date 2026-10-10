"""Tests for tool concurrency batching (#562 v2).

Verifies that consecutive concurrency-safe direct tools are batched
for parallel execution, while feature dispatches and unsafe tools
remain sequential. All tools go through _dispatch_tool_call —
nothing bypasses hooks, observability, or context_stats.
"""

import asyncio
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock, patch

from kestrel_sovereign.agent.orchestrator_engine import (
    OrchestratorEngineMixin,
    MAX_TOOL_CONCURRENCY,
)


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: dict


class FakeToolSchema:
    def __init__(self, is_concurrency_safe=False):
        self.is_concurrency_safe = is_concurrency_safe


class FakeTool:
    def __init__(self, safe=False):
        self.schema = FakeToolSchema(safe)

    async def execute(self, **kwargs):
        return {"success": True}


class FakeFeature:
    def __init__(self, name):
        self.tool_name = name


def _make_mixin(**direct_tools):
    """Create a minimal OrchestratorEngineMixin with mocked state."""
    mixin = object.__new__(OrchestratorEngineMixin)
    mixin._direct_tools = direct_tools
    mixin._tool_to_feature = {}
    mixin.features = {}
    return mixin


class TestPartitionToolCalls:
    def test_all_serial_feature_tools(self):
        mixin = _make_mixin()
        features = {"feat_a": FakeFeature("feat_a"), "feat_b": FakeFeature("feat_b")}
        calls = [FakeToolCall("1", "feat_a", {}), FakeToolCall("2", "feat_b", {})]

        batches = mixin._partition_tool_calls(calls, features)
        assert len(batches) == 2
        assert all(not is_par for is_par, _ in batches)

    def test_all_parallel_direct_tools(self):
        mixin = _make_mixin(
            read_a=FakeTool(safe=True),
            read_b=FakeTool(safe=True),
        )
        calls = [FakeToolCall("1", "read_a", {}), FakeToolCall("2", "read_b", {})]

        batches = mixin._partition_tool_calls(calls, {})
        assert len(batches) == 1
        assert batches[0][0] is True  # parallel
        assert len(batches[0][1]) == 2

    def test_mixed_serial_and_parallel(self):
        mixin = _make_mixin(
            read_a=FakeTool(safe=True),
            read_b=FakeTool(safe=True),
            write_c=FakeTool(safe=False),
        )
        calls = [
            FakeToolCall("1", "read_a", {}),
            FakeToolCall("2", "read_b", {}),
            FakeToolCall("3", "write_c", {}),  # breaks the batch
            FakeToolCall("4", "read_a", {}),
        ]

        batches = mixin._partition_tool_calls(calls, {})
        assert len(batches) == 3
        assert batches[0] == (True, [calls[0], calls[1]])   # parallel
        assert batches[1] == (False, [calls[2]])              # serial
        assert batches[2] == (True, [calls[3]])               # parallel (single)

    def test_feature_tool_never_parallel(self):
        """Feature dispatches are never parallel even if a direct tool with same name exists."""
        mixin = _make_mixin(my_tool=FakeTool(safe=True))
        features = {"my_tool": FakeFeature("my_tool")}
        calls = [FakeToolCall("1", "my_tool", {})]

        batches = mixin._partition_tool_calls(calls, features)
        assert len(batches) == 1
        assert batches[0][0] is False  # serial — feature wins

    def test_unsafe_direct_tool_serial(self):
        mixin = _make_mixin(bash_tool=FakeTool(safe=False))
        calls = [FakeToolCall("1", "bash_tool", {})]

        batches = mixin._partition_tool_calls(calls, {})
        assert len(batches) == 1
        assert batches[0][0] is False

    def test_single_tool_no_overhead(self):
        mixin = _make_mixin(read_a=FakeTool(safe=True))
        calls = [FakeToolCall("1", "read_a", {})]

        batches = mixin._partition_tool_calls(calls, {})
        # Single safe tool still gets batched as parallel but with 1 item
        assert len(batches) == 1

    def test_empty_tool_calls(self):
        mixin = _make_mixin()
        batches = mixin._partition_tool_calls([], {})
        assert batches == []


class TestIsConcurrencySafe:
    def test_safe_tool(self):
        mixin = _make_mixin(read_file=FakeTool(safe=True))
        assert mixin._is_direct_tool_concurrency_safe("read_file") is True

    def test_unsafe_tool(self):
        mixin = _make_mixin(bash=FakeTool(safe=False))
        assert mixin._is_direct_tool_concurrency_safe("bash") is False

    def test_unknown_tool(self):
        mixin = _make_mixin()
        assert mixin._is_direct_tool_concurrency_safe("nonexistent") is False

    def test_tool_without_schema(self):
        tool = MagicMock()
        del tool.schema  # no schema attribute
        mixin = _make_mixin(weird_tool=tool)
        assert mixin._is_direct_tool_concurrency_safe("weird_tool") is False


class TestMaxConcurrency:
    def test_env_default(self):
        assert MAX_TOOL_CONCURRENCY == 10


@pytest.mark.asyncio
async def test_parallel_failure_joins_cancellation_resistant_sibling_before_return():
    """Exercise the native batch owner, not a detached gather substitute."""
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    drained = asyncio.Event()
    finished = []

    class Agent(OrchestratorEngineMixin):
        _direct_tools = {"fail": FakeTool(safe=True), "slow": FakeTool(safe=True)}
        _tool_to_feature = {}
        features = {}

        async def _dispatch_tool_call(self, call, *args, **kwargs):
            if call.name == "fail":
                await entered.wait()
                raise ValueError("first child failed")
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await drained.wait()
            finished.append("slow child joined")

    agent = Agent()
    task = asyncio.create_task(agent._execute_tool_batch(
        [FakeToolCall("1", "fail", {}), FakeToolCall("2", "slow", {})],
        {}, set(), [], 0, "message",
    ))
    try:
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        assert not task.done()
        drained.set()
        with pytest.raises(ValueError, match="first child failed"):
            await task
        assert finished == ["slow child joined"]
    finally:
        drained.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
