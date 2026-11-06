"""Actual named, subagent and native A2A leaves claim before file effects."""

import pytest
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_task
from kestrel_sdk.native_a2a.types import TaskState
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sdk.features.base import Feature as SDKFeature
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.resources import ResourceConflict

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
from tests.unit.test_tool_effect_execution import owner
from tests.unit.test_tool_effect_invocation import governed


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest.mark.parametrize('lane,feature_base', [
    ('named', Feature), ('named', SDKFeature), ('subagent', Feature), ('task', Feature),
], ids=['core-named', 'sdk-named', 'core-subagent', 'core-task'])
@pytest.mark.parametrize('uncertain', [False, True])
async def test_actual_leaf_cannot_execute_before_claim_ack_or_repeat_after_root_exit(owner, tmp_path, lane, feature_base, uncertain):
    witness = tmp_path / 'actual-leaf-effect'
    store = ObservabilityStore(owner.tool_effect_store.database.backend, owns_backend=False)
    await store.initialize()
    owner.observability_store = store

    class OwnedFileFeature(feature_base):
        @property
        def tool_description(self):
            return 'Append an owned file effect'

        async def initialize(self):
            pass

        @tool(name='owned_leaf_effect', description='Append one owned marker')
        async def owned_leaf_effect(self, marker: str) -> dict:
            assert await owner.tool_effect_store.database.fetchone(
                'SELECT state FROM tool_effect_claims') == ('STARTED',), 'Leaf bypassed durable effect admission'
            assert owner.tool_effect_store.database.owns_open_transaction is False
            with witness.open('a') as stream:
                stream.write(marker + '\n')
            if uncertain:
                raise ResourceOutcomeUnknown('Owned remote reply lost after file effect')
            return {'success': True, 'marker': marker}

    feature = OwnedFileFeature(owner)
    owner.features[feature.name] = feature
    registered = feature.get_tools()[0]
    entered = []

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        entered.append(lane)
        if lane == 'named':
            return await agent.execute_named_tool(registered.name, {'marker': 'one'},
                session_id='owned-leaf-turn')
        if lane == 'subagent':
            return await feature._execute_subagent_tool(tool_name=registered.name,
                args={'marker': 'one'}, tools_by_name={registered.name: registered},
                session_id='owned-leaf-turn')
        task = new_task('owned-task', 'owned-context', TaskState.TASK_STATE_SUBMITTED)
        task.metadata.update({'skill': registered.name, 'args': {'marker': 'one'}})
        return await feature.handle_task(task)

    try:
        if uncertain:
            with pytest.raises(ResourceOutcomeUnknown):
                await invoke(owner, 'owned intent', invocation_id='original-leaf-intent')
        else:
            await invoke(owner, 'owned intent', invocation_id='original-leaf-intent')
        assert witness.read_text() == 'one\n'
        assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_claims') == (
            'UNCONFIRMED' if uncertain else 'RETURNED_OK',)
        with pytest.raises(ToolEffectReconciliationRequired):
            await invoke(owner, 'owned intent', invocation_id='original-leaf-intent')
        assert entered == [lane]
        assert witness.read_text() == 'one\n'
    finally:
        await store.close()


@pytest.mark.parametrize('withdrawal', ['provider-registration', 'registry-generation', 'provider-after-claim'])
async def test_task_manager_cannot_execute_a_registration_withdrawn_during_approval(owner, tmp_path, monkeypatch, withdrawal):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.agent.tool_allowlist import bump_registry_generation
    from tests.utils.native_a2a import manager_for

    witness = tmp_path / 'withdrawn-task-manager-effect'
    manager = manager_for(owner.tool_effect_store.database.backend,
        NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    manager.hooks_manager = owner.hooks_manager

    class FileFeature(Feature):
        @property
        def tool_description(self):
            return 'Owned task file witness'

        async def initialize(self):
            pass

        @tool(name='withdrawn_task_effect', description='Must retain original task provider')
        async def withdrawn_task_effect(self) -> dict:
            witness.write_text('withdrawn task provider executed')
            return {'success': True}

    feature = FileFeature(owner)
    owner.features[feature.name] = feature
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)

    class Withdraw(Hook):
        def __init__(self):
            super().__init__('withdraw-task-provider', [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            if withdrawal == 'provider-registration':
                manager.unregister_provider(feature.name)
            elif withdrawal == 'registry-generation':
                bump_registry_generation(owner)
            return HookOutput.allow()

    owner.hooks_manager.register(Withdraw())
    if withdrawal == 'provider-after-claim':
        original_claim = owner.tool_effect_store.claim

        async def withdraw_after_ack(*args, **kwargs):
            claim = await original_claim(*args, **kwargs)
            assert owner.tool_effect_store.database.owns_open_transaction is False
            manager.unregister_provider(feature.name)
            return claim

        monkeypatch.setattr(owner.tool_effect_store, 'claim', withdraw_after_ack)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        return await manager.execute_skill(feature.name, 'withdrawn_task_effect', {})

    try:
        try:
            await invoke(owner, 'original task intention', invocation_id='original-task-intent')
        except ResourceConflict:
            pass
        assert not witness.exists(), 'TaskManager executed a provider withdrawn during PRE_TOOL_USE'
        assert all(row == ('NOT_EXECUTED',) for row in
            await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims'))
    finally:
        await manager.close()


@pytest.mark.parametrize('lane', ['named', 'direct', 'subagent'])
async def test_original_registration_withdrawn_during_pre_hook_cannot_execute(owner, tmp_path, lane):
    from kestrel_sovereign.agent.tool_allowlist import bump_registry_generation
    from kestrel_sovereign.llm.adapter import ToolCall

    witness = tmp_path / 'withdrawn-effect'
    store = ObservabilityStore(owner.tool_effect_store.database.backend, owns_backend=False)
    await store.initialize()
    owner.observability_store = store

    class FileFeature(Feature):
        @property
        def tool_description(self):
            return 'Owned file witness'

        async def initialize(self):
            pass

        @tool(name='withdrawn_effect', description='Must not execute after withdrawal')
        async def withdrawn_effect(self) -> dict:
            witness.write_text('stale registration executed')
            return {'success': True}

    feature = FileFeature(owner)
    owner.features[feature.name] = feature
    registered = feature.get_tools()[0]

    class Withdraw(Hook):
        def __init__(self):
            super().__init__('withdraw-original-registration', [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            # The real registry's withdrawal changes the epoch while approval
            # is suspended. The selected object must retain its old epoch.
            owner.features.pop(feature.name)
            bump_registry_generation(owner)
            return HookOutput.allow()

    owner.hooks_manager.register(Withdraw())

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        if lane == 'named':
            return await agent.execute_named_tool(registered.name, {}, session_id='withdrawn-turn')
        if lane == 'subagent':
            return await feature._execute_subagent_tool(tool_name=registered.name,
                args={}, tools_by_name={registered.name: registered}, session_id='withdrawn-turn')
        call = ToolCall(id='withdrawn-call', name=registered.name, arguments={})
        return await agent._dispatch_direct_tool(call, registered.name, {}, 0,
            'withdrawn-dispatch', session_id='withdrawn-turn', tool=registered,
            feature_name=feature.name)

    try:
        try:
            await invoke(owner, 'owned withdrawal intent', invocation_id=f'withdrawal-{lane}')
        except ResourceConflict:
            pass
        assert not witness.exists(), 'PRE_TOOL_USE withdrawal executed the stale selected tool'
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT state FROM tool_effect_claims') == ('NOT_EXECUTED',)
    finally:
        await store.close()
