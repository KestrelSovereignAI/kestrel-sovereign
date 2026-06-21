from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookOutput
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin


class _MutatingHooksManager:
    def __init__(self, rewritten_args):
        self.rewritten_args = rewritten_args
        self.pre_tool_inputs = []
        self.post_tool_inputs = []

    async def execute_hooks(self, event, hook_input):
        if event == HookEvent.PRE_TOOL_USE:
            self.pre_tool_inputs.append(hook_input)
            hook_input.tool_input = self.rewritten_args
        return HookOutput.allow()

    async def execute_hooks_parallel(self, event, hook_input):
        if event == HookEvent.POST_TOOL_USE:
            self.post_tool_inputs.append(hook_input)


class _Agent(OrchestratorEngineMixin):
    pass


@pytest.mark.asyncio
async def test_dispatch_direct_tool_executes_with_pre_tool_use_rewrite():
    hooks_manager = _MutatingHooksManager({"message": "SANITIZED"})
    agent = _Agent()
    agent.hooks_manager = hooks_manager
    agent._direct_tools = {}
    agent._tool_to_feature = {}
    agent.features = {}
    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_response = AsyncMock()

    tool = MagicMock()
    tool.execute = AsyncMock(return_value={"success": True})
    agent._direct_tools["echo"] = tool

    result = await agent._dispatch_direct_tool(
        tool_call=SimpleNamespace(name="echo"),
        tool_name="echo",
        args={"message": "original pii"},
        dispatch_start=0.0,
        dispatch_event_id="evt-direct",
        session_id="session-direct",
    )

    assert result == {"success": True}
    tool.execute.assert_awaited_once_with(message="SANITIZED")
    assert hooks_manager.post_tool_inputs[0].tool_input == {"message": "SANITIZED"}


@pytest.mark.asyncio
async def test_dispatch_feature_tool_executes_with_pre_tool_use_rewrite():
    hooks_manager = _MutatingHooksManager({
        "task": "SANITIZED",
        "context": "clean context",
    })
    agent = _Agent()
    agent.hooks_manager = hooks_manager
    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_response = AsyncMock()
    agent._get_denied_tools = AsyncMock(return_value=set())
    agent._register_explored_feature_tools = MagicMock()

    feature = MagicMock()
    feature.tool_name = "test_feature"
    feature.execute_as_subagent = AsyncMock(return_value={"success": True})

    result = await agent._dispatch_feature_tool(
        tool_call=SimpleNamespace(name="test_feature"),
        feature=feature,
        args={"task": "original pii", "context": "original context"},
        dispatch_start=0.0,
        dispatch_event_id="evt-feature",
        user_message="original user message",
        session_id="session-feature",
    )

    assert result == {"success": True}
    feature.execute_as_subagent.assert_awaited_once_with(
        task="SANITIZED",
        context="clean context",
        denied_tools=set(),
    )
    assert hooks_manager.post_tool_inputs[0].tool_input == {
        "task": "SANITIZED",
        "context": "clean context",
    }
