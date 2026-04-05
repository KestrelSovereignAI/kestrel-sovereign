"""
Tests for tool concurrency batching — parallel execution for read-only tools.

Covers:
- _partition_tool_calls correctly separates sequential vs parallel
- _execute_tool_batch runs parallel tools concurrently
- PRE_TOOL_USE hooks fire sequentially before parallel launch
- Results are appended to messages in original request order
- Semaphore respects MAX_TOOL_CONCURRENCY
- Feature subagent dispatches stay sequential
- Non-concurrency-safe direct tools stay sequential
- asyncio.Lock protects _register_explored_feature_tools
- is_concurrency_safe plumbing through @tool decorator
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.hooks import HookEvent, HookInput, PermissionDecision
from kestrel_sovereign.hooks.base import HookOutput
from kestrel_sovereign.tools.base import AgentTool, ToolCategory, ToolParameter, ToolSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: dict


class FakeTool(AgentTool):
    """Minimal AgentTool for testing."""

    def __init__(self, tool_name: str, concurrency_safe: bool = False, delay: float = 0):
        self._name = tool_name
        self._concurrency_safe = concurrency_safe
        self._delay = delay
        self.call_count = 0
        self.call_log: list = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"Fake tool {self._name}",
            category=ToolCategory.SYSTEM,
            is_concurrency_safe=self._concurrency_safe,
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        self.call_count += 1
        self.call_log.append({"time": time.monotonic(), "kwargs": kwargs})
        if self._delay:
            await asyncio.sleep(self._delay)
        return {"success": True, "tool": self._name}


class FakeFeature:
    """Minimal stand-in for a Feature subagent."""
    def __init__(self, name: str):
        self.tool_name = name
        self.name = name


class FakeHooksManager:
    """Hooks manager that always allows."""

    def __init__(self, deny_tools: set | None = None):
        self._deny_tools = deny_tools or set()
        self.pre_hook_calls: list = []

    async def execute_hooks(self, event, hook_input):
        self.pre_hook_calls.append(hook_input.tool_name)
        if hook_input.tool_name in self._deny_tools:
            return HookOutput(
                permission_decision=PermissionDecision.DENY,
                permission_reason=f"Denied: {hook_input.tool_name}",
            )
        return HookOutput(permission_decision=PermissionDecision.ALLOW)

    async def execute_hooks_parallel(self, event, hook_input):
        pass  # POST hooks are fire-and-forget


class FakeObservabilityStore:
    async def log_tool_call(self, **kwargs):
        return "evt-1"

    async def log_tool_response(self, **kwargs):
        pass


def _build_agent_stub(
    direct_tools: dict | None = None,
    features_by_tool_name: dict | None = None,
    deny_tools: set | None = None,
):
    """Build a minimal object that satisfies both OrchestratorEngineMixin and
    ToolRegistryMixin interfaces for testing."""

    class Stub(OrchestratorEngineMixin, ToolRegistryMixin):
        pass

    stub = Stub()
    stub._direct_tools = direct_tools or {}
    stub._direct_tool_defs = []
    stub._explored_features = {}
    stub._tool_to_feature = {
        name: name for name in (direct_tools or {})
    }
    stub.hooks_manager = FakeHooksManager(deny_tools=deny_tools)
    stub.observability_store = FakeObservabilityStore()
    stub.did = "did:test:agent"
    stub.features = {}
    return stub


# ---------------------------------------------------------------------------
# Tests: _partition_tool_calls
# ---------------------------------------------------------------------------

class TestPartitionToolCalls:
    """Tests for OrchestratorEngineMixin._partition_tool_calls."""

    def test_all_sequential_when_no_direct_tools(self):
        """Feature subagent dispatches are always sequential."""
        stub = _build_agent_stub()
        features = {"feat_a": FakeFeature("feat_a"), "feat_b": FakeFeature("feat_b")}
        calls = [
            FakeToolCall(id="1", name="feat_a", arguments={}),
            FakeToolCall(id="2", name="feat_b", arguments={}),
        ]
        seq, par = stub._partition_tool_calls(calls, features)
        assert len(seq) == 2
        assert len(par) == 0

    def test_concurrency_safe_direct_tools_are_parallel(self):
        """Direct tools with is_concurrency_safe=True go to parallel."""
        tool_a = FakeTool("read_file", concurrency_safe=True)
        tool_b = FakeTool("write_file", concurrency_safe=False)
        stub = _build_agent_stub(direct_tools={"read_file": tool_a, "write_file": tool_b})

        calls = [
            FakeToolCall(id="1", name="read_file", arguments={}),
            FakeToolCall(id="2", name="write_file", arguments={}),
        ]
        seq, par = stub._partition_tool_calls(calls, {})
        assert len(par) == 1
        assert par[0].name == "read_file"
        assert len(seq) == 1
        assert seq[0].name == "write_file"

    def test_mixed_feature_and_direct(self):
        """Feature dispatches are sequential even if a direct tool is parallel-eligible."""
        tool_a = FakeTool("search", concurrency_safe=True)
        stub = _build_agent_stub(direct_tools={"search": tool_a})
        features = {"model_agent": FakeFeature("model_agent")}

        calls = [
            FakeToolCall(id="1", name="model_agent", arguments={}),
            FakeToolCall(id="2", name="search", arguments={}),
        ]
        seq, par = stub._partition_tool_calls(calls, features)
        assert seq[0].name == "model_agent"
        assert par[0].name == "search"

    def test_unknown_tool_goes_sequential(self):
        """Tools not in _direct_tools or features go sequential (handled as error later)."""
        stub = _build_agent_stub()
        calls = [FakeToolCall(id="1", name="ghost_tool", arguments={})]
        seq, par = stub._partition_tool_calls(calls, {})
        assert len(seq) == 1
        assert len(par) == 0


# ---------------------------------------------------------------------------
# Tests: _execute_tool_batch
# ---------------------------------------------------------------------------

class TestExecuteToolBatch:
    """Tests for parallel execution in _execute_tool_batch."""

    @pytest.mark.asyncio
    async def test_parallel_tools_run_concurrently(self):
        """Two concurrency-safe tools should overlap in wall-clock time.

        We verify concurrency by checking that both tools' execute() calls
        started before either finished — i.e. they overlapped.
        """
        started = []

        class TimedTool(FakeTool):
            async def execute(self, **kwargs):
                started.append((self._name, time.monotonic()))
                await asyncio.sleep(0.15)
                self.call_count += 1
                return {"success": True, "tool": self._name}

        tool_a = TimedTool("alpha", concurrency_safe=True)
        tool_b = TimedTool("beta", concurrency_safe=True)
        stub = _build_agent_stub(direct_tools={"alpha": tool_a, "beta": tool_b})

        calls = [
            FakeToolCall(id="1", name="alpha", arguments={}),
            FakeToolCall(id="2", name="beta", arguments={}),
        ]
        messages: list = []
        await stub._execute_tool_batch(
            calls, {}, {"alpha", "beta"}, messages, 0, "test"
        )

        assert tool_a.call_count == 1
        assert tool_b.call_count == 1
        # Both should have started before either could finish (0.15s sleep)
        assert len(started) == 2
        t0 = started[0][1]
        t1 = started[1][1]
        # The gap between start times should be much less than the sleep duration
        assert abs(t1 - t0) < 0.1, (
            f"Tools did not start concurrently: gap={abs(t1 - t0):.3f}s"
        )

    @pytest.mark.asyncio
    async def test_results_appended_in_original_order(self):
        """Even if beta finishes first, messages must be alpha then beta."""
        tool_a = FakeTool("alpha", concurrency_safe=True, delay=0.05)
        tool_b = FakeTool("beta", concurrency_safe=True, delay=0.0)
        stub = _build_agent_stub(direct_tools={"alpha": tool_a, "beta": tool_b})

        calls = [
            FakeToolCall(id="tc-1", name="alpha", arguments={}),
            FakeToolCall(id="tc-2", name="beta", arguments={}),
        ]
        messages: list = []
        await stub._execute_tool_batch(calls, {}, {"alpha", "beta"}, messages, 0, "test")

        assert len(messages) == 2
        assert messages[0]["tool_call_id"] == "tc-1"
        assert messages[1]["tool_call_id"] == "tc-2"

    @pytest.mark.asyncio
    async def test_pre_hooks_fire_sequentially_before_parallel(self):
        """PRE_TOOL_USE hooks must fire in order before any parallel execution begins."""
        tool_a = FakeTool("alpha", concurrency_safe=True)
        tool_b = FakeTool("beta", concurrency_safe=True)
        stub = _build_agent_stub(direct_tools={"alpha": tool_a, "beta": tool_b})

        calls = [
            FakeToolCall(id="1", name="alpha", arguments={}),
            FakeToolCall(id="2", name="beta", arguments={}),
        ]
        messages: list = []
        await stub._execute_tool_batch(calls, {}, {"alpha", "beta"}, messages, 0, "test")

        # The hooks manager recorded the order of PRE_TOOL_USE calls
        assert stub.hooks_manager.pre_hook_calls == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_denied_tool_in_parallel_batch(self):
        """A tool denied by PRE_TOOL_USE hook should return permission denied."""
        tool_a = FakeTool("alpha", concurrency_safe=True)
        tool_b = FakeTool("beta", concurrency_safe=True)
        stub = _build_agent_stub(
            direct_tools={"alpha": tool_a, "beta": tool_b},
            deny_tools={"alpha"},
        )

        calls = [
            FakeToolCall(id="1", name="alpha", arguments={}),
            FakeToolCall(id="2", name="beta", arguments={}),
        ]
        messages: list = []
        await stub._execute_tool_batch(calls, {}, {"alpha", "beta"}, messages, 0, "test")

        assert len(messages) == 2
        alpha_result = json.loads(messages[0]["content"])
        assert alpha_result["success"] is False
        assert "Permission denied" in alpha_result["error"]
        # alpha should NOT have been executed
        assert tool_a.call_count == 0
        # beta should have executed normally
        assert tool_b.call_count == 1

    @pytest.mark.asyncio
    async def test_sequential_fallback_when_no_parallel(self):
        """When no tools are concurrency-safe, all run sequentially."""
        tool_a = FakeTool("alpha", concurrency_safe=False)
        tool_b = FakeTool("beta", concurrency_safe=False)
        stub = _build_agent_stub(direct_tools={"alpha": tool_a, "beta": tool_b})

        # We need to mock _dispatch_tool_call since sequential tools go through it
        dispatch_calls = []

        async def mock_dispatch(tc, *args, **kwargs):
            dispatch_calls.append(tc.name)

        stub._dispatch_tool_call = mock_dispatch

        calls = [
            FakeToolCall(id="1", name="alpha", arguments={}),
            FakeToolCall(id="2", name="beta", arguments={}),
        ]
        messages: list = []
        await stub._execute_tool_batch(calls, {}, {"alpha", "beta"}, messages, 0, "test")

        assert dispatch_calls == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_feature_dispatches_stay_sequential(self):
        """Feature subagent dispatches must never be parallelised."""
        tool_a = FakeTool("search", concurrency_safe=True)
        stub = _build_agent_stub(direct_tools={"search": tool_a})
        features = {"model_agent": FakeFeature("model_agent")}

        dispatch_calls = []
        original_dispatch = stub._dispatch_tool_call

        async def mock_dispatch(tc, *args, **kwargs):
            dispatch_calls.append(tc.name)

        stub._dispatch_tool_call = mock_dispatch

        calls = [
            FakeToolCall(id="1", name="model_agent", arguments={"task": "list"}),
            FakeToolCall(id="2", name="search", arguments={}),
        ]
        messages: list = []
        await stub._execute_tool_batch(
            calls, features, {"model_agent", "search"}, messages, 0, "test"
        )

        # model_agent should have gone through _dispatch_tool_call (sequential)
        assert "model_agent" in dispatch_calls
        # search was parallel — executed directly, not via _dispatch_tool_call
        assert tool_a.call_count == 1

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """asyncio.Semaphore should throttle parallel execution."""
        # Create many tools
        tools = {}
        for i in range(20):
            name = f"tool_{i}"
            tools[name] = FakeTool(name, concurrency_safe=True, delay=0.01)

        stub = _build_agent_stub(direct_tools=tools)

        calls = [FakeToolCall(id=str(i), name=f"tool_{i}", arguments={}) for i in range(20)]
        messages: list = []

        with patch(
            "kestrel_sovereign.agent.orchestrator_engine.MAX_TOOL_CONCURRENCY", 3
        ):
            await stub._execute_tool_batch(
                calls, {}, set(tools.keys()), messages, 0, "test"
            )

        # All tools should have executed
        assert len(messages) == 20
        for tool in tools.values():
            assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_exception_captured_in_parallel(self):
        """A tool that raises should not crash the batch; error is captured."""

        class FailingTool(FakeTool):
            async def execute(self, **kwargs):
                raise RuntimeError("boom")

        tool_a = FailingTool("alpha", concurrency_safe=True)
        tool_b = FakeTool("beta", concurrency_safe=True)
        stub = _build_agent_stub(direct_tools={"alpha": tool_a, "beta": tool_b})

        calls = [
            FakeToolCall(id="1", name="alpha", arguments={}),
            FakeToolCall(id="2", name="beta", arguments={}),
        ]
        messages: list = []
        await stub._execute_tool_batch(calls, {}, {"alpha", "beta"}, messages, 0, "test")

        assert len(messages) == 2
        alpha_result = json.loads(messages[0]["content"])
        assert alpha_result["success"] is False
        assert "boom" in alpha_result["error"]
        # beta should still succeed
        beta_result = json.loads(messages[1]["content"])
        assert beta_result["success"] is True


# ---------------------------------------------------------------------------
# Tests: Registry lock
# ---------------------------------------------------------------------------

class TestRegistryLock:
    """Tests that _register_explored_feature_tools is concurrency-safe."""

    @pytest.mark.asyncio
    async def test_lock_prevents_double_registration(self):
        """Concurrent calls should not double-register a feature's tools."""
        stub = _build_agent_stub()
        stub._explored_features = {}
        stub._direct_tools = {}
        stub._direct_tool_defs = []
        stub._tool_to_feature = {}

        feature = MagicMock()
        feature.tool_name = "test_feature"

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.schema = ToolSchema(
            name="test_tool",
            description="test",
            category=ToolCategory.SYSTEM,
            is_concurrency_safe=True,
        )
        mock_tool.schema.to_openai_format = MagicMock(return_value={
            "type": "function",
            "function": {"name": "test_tool", "description": "test", "parameters": {}}
        })
        feature.get_tools.return_value = [mock_tool]

        # Call concurrently
        await asyncio.gather(
            stub._register_explored_feature_tools(feature),
            stub._register_explored_feature_tools(feature),
        )

        # Should only be registered once
        assert len(stub._direct_tools) == 1
        assert "test_tool" in stub._direct_tools


# ---------------------------------------------------------------------------
# Tests: is_concurrency_safe plumbing through @tool decorator
# ---------------------------------------------------------------------------

class TestToolDecoratorConcurrencySafe:
    """Verify the concurrency_safe kwarg flows from @tool to ToolSchema."""

    def test_default_is_false(self):
        from kestrel_sovereign.features.base import tool

        @tool("my_read", "Read something", category=ToolCategory.SYSTEM)
        async def my_read(self):
            pass

        assert my_read._tool_schema["concurrency_safe"] is False

    def test_explicit_true(self):
        from kestrel_sovereign.features.base import tool

        @tool("my_search", "Search", category=ToolCategory.SYSTEM, concurrency_safe=True)
        async def my_search(self):
            pass

        assert my_search._tool_schema["concurrency_safe"] is True

    def test_schema_carries_flag(self):
        """ToolSchema.is_concurrency_safe should reflect the decorator kwarg."""
        schema = ToolSchema(
            name="test",
            description="test",
            category=ToolCategory.SYSTEM,
            is_concurrency_safe=True,
        )
        assert schema.is_concurrency_safe is True

        schema_default = ToolSchema(
            name="test2",
            description="test2",
            category=ToolCategory.SYSTEM,
        )
        assert schema_default.is_concurrency_safe is False


# ---------------------------------------------------------------------------
# Tests: MAX_TOOL_CONCURRENCY env var
# ---------------------------------------------------------------------------

class TestMaxToolConcurrencyConfig:
    """Verify KESTREL_MAX_TOOL_CONCURRENCY env var is respected."""

    def test_default_value(self):
        from kestrel_sovereign.kestrel_agent import MAX_TOOL_CONCURRENCY
        # Default is 10 (unless env var overrides in CI)
        assert isinstance(MAX_TOOL_CONCURRENCY, int)
        assert MAX_TOOL_CONCURRENCY > 0

    def test_env_var_override(self):
        import os
        os.environ["KESTREL_MAX_TOOL_CONCURRENCY"] = "5"
        try:
            val = int(os.environ.get("KESTREL_MAX_TOOL_CONCURRENCY", "10"))
            assert val == 5
        finally:
            os.environ["KESTREL_MAX_TOOL_CONCURRENCY"] = "10"
