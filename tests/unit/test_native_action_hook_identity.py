"""Actual sequential and parallel hook runners pin native action identity."""

from dataclasses import replace
import asyncio

import pytest
from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput, PermissionDecision
from kestrel_sovereign.hooks.manager import HooksManager
from tests.unit.test_native_action_contract import context as context


class Policy(Hook):
    def __init__(self, change):
        super().__init__('owned-policy', [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE])
        self.change = change

    async def execute(self, input):
        return self.change(input)


def mutate(input, kind):
    if kind == 'replace':
        input.native_action = replace(input.native_action, task_handle='different')
    elif kind == 'same-values':
        input.native_action = replace(input.native_action)
    elif kind == 'clear':
        input.native_action = None
    elif kind == 'in-place':
        object.__setattr__(input.native_action, 'task_handle', 'different')
    elif kind == 'tool':
        input.tool_name = 'different'
    elif kind == 'feature':
        input.feature_name = 'different'
    return HookOutput.allow()


@pytest.mark.asyncio
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('kind', ['replace', 'same-values', 'clear', 'in-place', 'tool', 'feature'])
@pytest.mark.parametrize('raises', [False, True])
async def test_identity_changes_deny_even_when_advisory_hook_raises(context, parallel, kind, raises):
    def change(input):
        result = mutate(input, kind)
        if raises:
            raise RuntimeError('Advisory hook failed after changing identity')
        return result
    manager = HooksManager()
    manager.register(Policy(change))
    value = HookInput('owned-session', HookEvent.PRE_TOOL_USE.value, tool_name='owned_tool',
                      feature_name='mcp:owned', native_action=context, tool_input={})
    if parallel:
        outputs = await manager.execute_hooks_parallel(HookEvent.POST_TOOL_USE, value)
        assert len(outputs) == 1
        output = outputs[0]
    else:
        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, value)
    assert output.permission_decision is PermissionDecision.DENY
    assert output.permission_reason == 'Native action identity changed during governance'
    assert value.native_action.task_handle == context.task_handle == 'owned-local'
    assert value.tool_name == 'owned_tool' and value.feature_name == 'mcp:owned'


@pytest.mark.asyncio
async def test_empty_rewrite_is_preserved_and_context_is_detached_for_each_hook(context):
    manager = HooksManager()
    seen = []
    def first(input):
        seen.append(input.native_action)
        return HookOutput.modify({})
    def second(input):
        seen.append(input.native_action)
        assert input.tool_input == {}
        return HookOutput.allow()
    one, two = Policy(first), Policy(second)
    two.name = 'second-policy'
    manager.register(one)
    manager.register(two)
    value = HookInput('owned-session', HookEvent.PRE_TOOL_USE.value, tool_name='owned_tool',
                      feature_name='mcp:owned', native_action=context,
                      tool_input={'inputResponses': {'key': 'secret'}})
    output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, value)
    assert output.updated_input == {} and value.tool_input == {}
    assert all(item == context and item is not context for item in seen)
    assert seen[0] is not seen[1]


@pytest.mark.asyncio
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('raises', [False, True])
async def test_ordinary_hook_cannot_add_host_owned_native_identity(context, parallel, raises):
    def inject(input):
        input.native_action = context
        if raises:
            raise RuntimeError('Advisory failed after injecting identity')
        return HookOutput.allow()
    manager = HooksManager()
    manager.register(Policy(inject))
    value = HookInput('owned-session', HookEvent.PRE_TOOL_USE.value, tool_name='ordinary',
                      feature_name='ordinary', tool_input={})
    if parallel:
        outputs = await manager.execute_hooks_parallel(HookEvent.POST_TOOL_USE, value)
        assert len(outputs) == 1
        output = outputs[0]
    else:
        output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, value)
    assert output.permission_decision is PermissionDecision.DENY
    assert value.native_action is None


@pytest.mark.asyncio
@pytest.mark.parametrize('reader_restores_original', [False, True])
async def test_parallel_native_observers_share_result_replacement_without_sharing_identity_or_params(context, reader_restores_original):
    started, replaced = asyncio.Event(), asyncio.Event()
    original = {'private': 'original-payload'}
    seen = []
    class Observer(Hook):
        def __init__(self, writer):
            super().__init__('writer' if writer else 'reader', [HookEvent.POST_TOOL_USE])
            self.writer = writer
        async def execute(self, input):
            seen.append(input)
            if self.writer:
                await started.wait()
                input.tool_response = None
                replaced.set()
            else:
                started.set()
                await replaced.wait()
                assert input.tool_response is None, 'A retained observer must see the current shared result'
                if reader_restores_original:
                    input.tool_response = original
            return HookOutput.allow()
    manager = HooksManager()
    manager.register(Observer(True))
    manager.register(Observer(False))
    value = HookInput('owned-session', HookEvent.POST_TOOL_USE.value, tool_name='owned_tool',
        feature_name='mcp:owned', native_action=context, tool_input={'inputResponses': {'key': 'answer'}},
        tool_response=original)
    outputs = await manager.execute_hooks_parallel(HookEvent.POST_TOOL_USE, value, require_complete=True)
    assert len(outputs) == 2 and all(output.permission_decision is PermissionDecision.ALLOW for output in outputs)
    assert value.tool_response is (original if reader_restores_original else None)
    assert seen[0].native_action is not seen[1].native_action
    assert seen[0].tool_input is not seen[1].tool_input and seen[0].tool_input is not value.tool_input
    assert all('_response_owner' not in item.to_dict() for item in seen)


@pytest.mark.asyncio
async def test_native_post_hook_view_preserves_sdk_dataclass_copy_contract(context):
    original_response = {'original': True}
    copies = []
    def copy(input):
        try:
            detached = replace(input, tool_input={'reviewed': True})
        except TypeError as error:
            raise AssertionError('Native hook views must preserve the SDK dataclass copy contract') from error
        detached.tool_response = {'copy': True}
        assert input.tool_response is original_response
        assert detached.to_dict()['native_action'] == context.to_dict()
        copies.append(detached)
        return HookOutput.allow()
    manager = HooksManager()
    manager.register(Policy(copy))
    value = HookInput('owned-session', HookEvent.POST_TOOL_USE.value, tool_name='owned_tool',
        feature_name='mcp:owned', native_action=context, tool_input={}, tool_response=original_response)
    outputs = await manager.execute_hooks_parallel(HookEvent.POST_TOOL_USE, value, require_complete=True)
    assert len(outputs) == 1 and outputs[0].permission_decision is PermissionDecision.ALLOW
    assert len(copies) == 1 and copies[0].tool_input == {'reviewed': True}
    assert value.tool_response is original_response
