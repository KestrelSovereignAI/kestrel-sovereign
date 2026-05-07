"""Verify session_id is correctly threaded to hook calls during tool dispatch (#885)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sdk.hooks.base import HookEvent, HookInput, HookOutput, PermissionDecision
from kestrel_sovereign.llm.adapter import ToolCall


@pytest.mark.asyncio
class TestSessionIdInHooks:
    """Verify session_id propagates to hooks in _dispatch_feature_tool and _dispatch_direct_tool."""

    async def test_dispatch_feature_tool_uses_real_session_id(self):
        """PRE/POST_SUBAGENT_CALL hooks should receive the real session_id, not 'orchestrator'."""

        # Create a mock agent with the orchestrator mixin
        agent = MagicMock()
        agent.hooks_manager = MagicMock()
        agent.hooks_manager.execute_hooks = AsyncMock(
            return_value=HookOutput(permission_decision=PermissionDecision.ALLOW)
        )
        agent.hooks_manager.execute_hooks_parallel = AsyncMock()
        agent.observability_store = MagicMock()
        agent.observability_store.log_tool_response = AsyncMock()
        agent._get_denied_tools = AsyncMock(return_value=set())
        agent._register_explored_feature_tools = MagicMock()
        agent._execute_tool_with_hooks = AsyncMock(return_value={"result": "success"})

        # Mock feature
        feature = MagicMock()
        feature.tool_name = "test_feature"
        feature.execute_as_subagent = AsyncMock(return_value={"result": "success"})
        type(feature).__name__ = "TestFeature"

        # Mock tool call
        tool_call = MagicMock()
        tool_call.name = "test_tool"

        # Bind the method to our mock agent
        dispatch = OrchestratorEngineMixin._dispatch_feature_tool.__get__(agent)

        # Execute with a real session_id
        await dispatch(
            tool_call=tool_call,
            feature=feature,
            args={"task": "do something"},
            dispatch_start=0.0,
            dispatch_event_id="evt_123",
            user_message="test message",
            session_id="real-session-abc",
        )

        # Verify PRE_SUBAGENT_CALL hook was called with the real session_id
        pre_call = agent.hooks_manager.execute_hooks.await_args_list[0]
        pre_hook_input = pre_call[0][1]  # Second positional arg is the HookInput
        assert isinstance(pre_hook_input, HookInput)
        assert pre_hook_input.session_id == "real-session-abc", \
            f"Expected session_id='real-session-abc', got '{pre_hook_input.session_id}'"

        # Verify POST_SUBAGENT_CALL hook was called with the real session_id
        post_call = agent.hooks_manager.execute_hooks_parallel.await_args_list[0]
        post_hook_input = post_call[0][1]  # Second positional arg is the HookInput
        assert isinstance(post_hook_input, HookInput)
        assert post_hook_input.session_id == "real-session-abc", \
            f"Expected session_id='real-session-abc', got '{post_hook_input.session_id}'"

    async def test_dispatch_direct_tool_uses_real_session_id(self):
        """Direct tool dispatch should use the real session_id in _execute_tool_with_hooks."""

        # Create a mock agent
        agent = MagicMock()
        agent._direct_tools = {}
        agent._tool_to_feature = {}
        agent._execute_tool_with_hooks = AsyncMock(return_value={"result": "success"})
        agent.observability_store = MagicMock()
        agent.observability_store.log_tool_response = AsyncMock()

        # Mock direct tool
        tool = MagicMock()
        tool.execute = AsyncMock(return_value={"result": "direct"})
        agent._direct_tools["direct_tool"] = tool
        agent._tool_to_feature["direct_tool"] = "DirectFeature"

        # Mock tool call
        tool_call = MagicMock()
        tool_call.name = "direct_tool"

        # Bind the method to our mock agent
        dispatch = OrchestratorEngineMixin._dispatch_direct_tool.__get__(agent)

        # Execute with a real session_id
        await dispatch(
            tool_call=tool_call,
            tool_name="direct_tool",
            args={"arg": "value"},
            dispatch_start=0.0,
            dispatch_event_id="evt_456",
            session_id="direct-session-xyz",
        )

        # Verify _execute_tool_with_hooks was called with the real session_id
        agent._execute_tool_with_hooks.assert_awaited_once()
        kwargs = agent._execute_tool_with_hooks.await_args.kwargs
        assert kwargs["session_id"] == "direct-session-xyz", \
            f"Expected session_id='direct-session-xyz', got '{kwargs['session_id']}'"

    async def test_default_session_id_still_works(self):
        """When session_id is not provided, it should default to 'orchestrator'."""

        # Create a mock agent
        agent = MagicMock()
        agent.hooks_manager = MagicMock()
        agent.hooks_manager.execute_hooks = AsyncMock(
            return_value=HookOutput(permission_decision=PermissionDecision.ALLOW)
        )
        agent.hooks_manager.execute_hooks_parallel = AsyncMock()
        agent.observability_store = MagicMock()
        agent.observability_store.log_tool_response = AsyncMock()
        agent._get_denied_tools = AsyncMock(return_value=set())
        agent._register_explored_feature_tools = MagicMock()
        agent._execute_tool_with_hooks = AsyncMock(return_value={"result": "success"})

        # Mock feature
        feature = MagicMock()
        feature.tool_name = "test_feature"
        feature.execute_as_subagent = AsyncMock(return_value={"result": "success"})
        type(feature).__name__ = "TestFeature"

        # Mock tool call
        tool_call = MagicMock()
        tool_call.name = "test_tool"

        # Bind the method to our mock agent
        dispatch = OrchestratorEngineMixin._dispatch_feature_tool.__get__(agent)

        # Execute WITHOUT providing session_id (should default to "orchestrator")
        await dispatch(
            tool_call=tool_call,
            feature=feature,
            args={"task": "do something"},
            dispatch_start=0.0,
            dispatch_event_id="evt_789",
            user_message="test message",
            # session_id not provided - should default
        )

        # Verify hooks were called with the default "orchestrator" session_id
        pre_call = agent.hooks_manager.execute_hooks.await_args_list[0]
        pre_hook_input = pre_call[0][1]
        assert pre_hook_input.session_id == "orchestrator"

        post_call = agent.hooks_manager.execute_hooks_parallel.await_args_list[0]
        post_hook_input = post_call[0][1]
        assert post_hook_input.session_id == "orchestrator"
