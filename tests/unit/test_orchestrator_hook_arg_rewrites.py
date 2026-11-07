"""Actual registered direct/subagent dispatch retains the hook-authorized binding."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookOutput
from kestrel_sovereign.features.base import Feature, tool
from tests.utils.native_tool_effects import drive_native


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


class RewriteFeature(Feature):
    @property
    def tool_description(self):
        return 'Owned argument rewrite probe'

    async def initialize(self):
        pass

    @tool(name='echo', description='Echo only the authorized message')
    async def echo(self, message: str) -> dict:
        self.observed.append(message)
        return {'success': True}


@pytest.mark.asyncio
async def test_dispatch_direct_tool_executes_with_pre_tool_use_rewrite(native_tool_effect_agent):
    hooks_manager = _MutatingHooksManager({'message': 'SANITIZED'})
    agent = await native_tool_effect_agent(did='did:test:direct-rewrite')
    agent.hooks_manager = hooks_manager
    feature = RewriteFeature(agent)
    feature.observed = []
    agent.features[feature.name] = feature
    selected, = feature.get_tools()
    agent._direct_tools[selected.name] = selected
    agent._tool_to_feature[selected.name] = feature.tool_name
    event = await agent.observability_store.log_tool_call(agent.did, selected.name)

    result = await drive_native(agent, lambda: agent._dispatch_direct_tool(
        tool_call=SimpleNamespace(name='echo'), tool_name='echo',
        args={'message': 'original pii'}, dispatch_start=0.0,
        dispatch_event_id=event, session_id='session-direct',
    ))

    assert result == {'success': True, 'tool': 'echo', 'result': {'success': True}}
    assert feature.observed == ['SANITIZED']
    assert hooks_manager.post_tool_inputs[0].tool_input == {'message': 'SANITIZED'}
    assert await agent._raw_storage.db.fetchone('SELECT state FROM tool_effect_claims') == ('RETURNED_OK',)


@pytest.mark.asyncio
async def test_dispatch_feature_tool_executes_with_pre_tool_use_rewrite(native_tool_effect_agent):
    hooks_manager = _MutatingHooksManager({'task': 'SANITIZED', 'context': 'clean context'})
    agent = await native_tool_effect_agent(did='did:test:subagent-rewrite')
    agent.hooks_manager = hooks_manager
    agent._get_denied_tools = AsyncMock(return_value=set())
    agent._register_explored_feature_tools = MagicMock()
    feature = RewriteFeature(agent)
    feature.name = 'TestFeature'
    feature.execute_as_subagent = AsyncMock(return_value={'success': True})
    agent.features[feature.name] = feature
    event = await agent.observability_store.log_tool_call(agent.did, feature.tool_name)

    result = await drive_native(agent, lambda: agent._dispatch_feature_tool(
        tool_call=SimpleNamespace(name='test_feature'), feature=feature,
        args={'task': 'original pii', 'context': 'original context'},
        dispatch_start=0.0, dispatch_event_id=event,
        user_message='original user message', session_id='session-feature',
    ))

    assert result == {'success': True}
    feature.execute_as_subagent.assert_awaited_once_with(
        task='SANITIZED', context='clean context', denied_tools=set(), model_override=None,
    )
    assert await agent._raw_storage.db.fetchone('SELECT state FROM tool_effect_claims') == ('RETURNED_OK',)
