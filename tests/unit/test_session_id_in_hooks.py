"""Actual registered dispatch threads session IDs into hooks (#885)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookInput
from tests.unit.test_orchestrator_hook_arg_rewrites import RewriteFeature
from tests.utils.native_tool_effects import drive_native


@pytest.mark.asyncio
class TestSessionIdInHooks:
    @pytest.mark.parametrize("session_id", ["real-session-abc", None])
    async def test_dispatch_feature_tool_uses_real_session_id(
        self, native_tool_effect_agent, session_id,
    ):
        agent = await native_tool_effect_agent(did="did:test:subagent-hook-session")
        hooks = agent.hooks_manager
        hooks.execute_hooks = AsyncMock(wraps=hooks.execute_hooks)
        hooks.execute_hooks_parallel = AsyncMock(wraps=hooks.execute_hooks_parallel)
        feature = RewriteFeature(agent)
        feature.execute_as_subagent = AsyncMock(return_value={"success": True})
        agent.features[feature.name] = feature
        event = await agent.observability_store.log_tool_call(agent.did, feature.tool_name)
        supplied = {} if session_id is None else {"session_id": session_id}

        result = await drive_native(agent, lambda: agent._dispatch_feature_tool(
            tool_call=SimpleNamespace(name=feature.tool_name), feature=feature,
            args={"task": "do something"}, dispatch_start=0.0,
            dispatch_event_id=event, user_message="test message", **supplied,
        ))

        assert result == {"success": True}
        feature.execute_as_subagent.assert_awaited_once()
        expected = "orchestrator" if session_id is None else session_id
        for spy, target_event in (
            (hooks.execute_hooks, HookEvent.PRE_SUBAGENT_CALL),
            (hooks.execute_hooks_parallel, HookEvent.POST_SUBAGENT_CALL),
        ):
            inputs = [args.args[1] for args in spy.await_args_list if args.args[0] == target_event]
            assert len(inputs) == 1
            assert isinstance(inputs[0], HookInput)
            assert inputs[0].session_id == expected
        assert await agent._raw_storage.db.fetchone(
            "SELECT state FROM tool_effect_claims"
        ) == ("RETURNED_OK",)

    async def test_dispatch_direct_tool_uses_real_session_id(self, native_tool_effect_agent):
        agent = await native_tool_effect_agent(did="did:test:direct-hook-session")
        feature = RewriteFeature(agent)
        feature.observed = []
        agent.features[feature.name] = feature
        selected, = feature.get_tools()
        agent._direct_tools[selected.name] = selected
        agent._tool_to_feature[selected.name] = feature.tool_name
        agent._execute_tool_with_hooks = AsyncMock(wraps=agent._execute_tool_with_hooks)
        event = await agent.observability_store.log_tool_call(agent.did, selected.name)

        result = await drive_native(agent, lambda: agent._dispatch_direct_tool(
            tool_call=SimpleNamespace(name=selected.name), tool_name=selected.name,
            args={"message": "value"}, dispatch_start=0.0,
            dispatch_event_id=event, session_id="direct-session-xyz",
        ))

        assert result == {"success": True, "tool": "echo", "result": {"success": True}}
        assert feature.observed == ["value"]
        agent._execute_tool_with_hooks.assert_awaited_once()
        assert agent._execute_tool_with_hooks.await_args.kwargs["session_id"] == "direct-session-xyz"
        assert await agent._raw_storage.db.fetchone(
            "SELECT state FROM tool_effect_claims"
        ) == ("RETURNED_OK",)
