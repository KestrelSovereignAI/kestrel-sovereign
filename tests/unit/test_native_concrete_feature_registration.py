"""Every concrete Feature route retains its original admitted owner and registration."""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from kestrel_sdk.features.base import Feature as SDKFeature
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.resources import ResourceConflict
from kestrel_sdk.tools.base import ToolSchema, ToolCategory
from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.llm.adapter import ToolCall
from tests.utils.native_tool_effects import drive_native

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest_asyncio.fixture(params=[(Feature, 'decorated'), (Feature, 'plain'), (SDKFeature, 'decorated'), (SDKFeature, 'plain')], ids=['core-decorated', 'core-plain', 'sdk-decorated', 'sdk-plain'])
async def concrete_feature(native_tool_effect_agent, tmp_path, request):
    host = await native_tool_effect_agent(did='did:test:concrete-original')
    foreign = await native_tool_effect_agent(did=host.did)
    host.native_effect_target = tmp_path / 'original-effect'
    foreign.native_effect_target = tmp_path / 'foreign-effect'

    feature_base, tool_style = request.param

    class ConcreteFeature(feature_base):
        tool_description = 'Retain the original concrete tool owner'

        async def initialize(self):
            pass

        @tool(name='concrete_effect', description='Write one original host marker')
        async def concrete_effect(self):
            self.agent.native_effect_target.write_text('concrete feature effect')
            return {'success': True}

        def get_tools(self):
            if tool_style == 'decorated':
                return super().get_tools()
            original_feature = self
            class PlainTool:
                name = 'concrete_effect'
                schema = ToolSchema(name, 'Use the original feature host', ToolCategory.UTILITY)
                async def execute(self):
                    return await original_feature.concrete_effect()
            return [PlainTool()]

    feature = ConcreteFeature(host)
    host.features[feature.name] = feature
    bindings = []
    host.register_dynamic_tools(feature.name, feature.get_tools(),
        on_registered=lambda offered, binding: bindings.append(binding))
    return SimpleNamespace(host=host, foreign=foreign, feature=feature,
        binding=bindings[0], replacement=ConcreteFeature(host))


@pytest.mark.parametrize('lane', ['chat', 'named', 'bound'])
@pytest.mark.parametrize('timing', ['permission', 'claim-ack'])
@pytest.mark.parametrize('change', ['host', 'disabled', 'replacement'])
async def test_concrete_feature_keeps_original_registration_through_native_admission(
    concrete_feature, monkeypatch, lane, timing, change,
):
    p = concrete_feature
    def withdraw():
        if change == 'host':
            p.feature.agent = p.foreign
        elif change == 'disabled':
            p.feature.enabled = False
        else:
            p.host.features[p.feature.name] = p.replacement

    if timing == 'permission':
        class Withdraw(Hook):
            def __init__(self):
                super().__init__('withdraw-concrete-original', [HookEvent.PRE_TOOL_USE])
            async def execute(self, value):
                withdraw()
                return HookOutput.allow()
        p.host.hooks_manager.register(Withdraw())
    else:
        claim = p.host.tool_effect_store.claim
        async def after_ack(*args, **kwargs):
            receipt = await claim(*args, **kwargs)
            withdraw()
            return receipt
        monkeypatch.setattr(p.host.tool_effect_store, 'claim', after_ack)

    async def operation():
        if lane == 'named':
            return await p.host.execute_named_tool(p.binding.name, {},
                session_id='concrete-session', operation_id='concrete-original')
        if lane == 'bound':
            return await p.binding.invoke({}, session_id='concrete-session',
                operation_id='concrete-original')
        return await p.host._dispatch_tool_call(
            ToolCall(id='concrete-call', name=p.binding.name, arguments={}), {},
            p.host._known_tool_names(), [], 0, 'one concrete effect', session_id='concrete-session')
    with caller_context_scope(CallerContext.sovereign()):
        try:
            await drive_native(p.host, operation)
        except ResourceConflict:
            pass
    assert not p.foreign.native_effect_target.exists(), 'Concrete dispatch entered a replacement host'
    assert not p.host.native_effect_target.exists(), 'Concrete dispatch executed a withdrawn feature'
    states = await p.host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims')
    assert all(state == ('NOT_EXECUTED',) for state in states)


@pytest.mark.parametrize('timing', ['permission', 'claim-ack'])
@pytest.mark.parametrize('change', ['host', 'disabled', 'handler'])
async def test_async_native_acceptance_retains_original_provider_feature(
    native_tool_effect_agent, tmp_path, monkeypatch, timing, change,
):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from tests.utils.native_a2a import manager_for

    host = await native_tool_effect_agent(did='did:test:original-async-feature')
    foreign = await native_tool_effect_agent(did=host.did)
    witness = tmp_path / 'withdrawn-native-provider-effect'
    created = []
    withdrawals = []

    class NativeProvider(Feature):
        tool_description = 'Original native provider ownership'
        async def initialize(self):
            pass
        @tool(name='original_provider_effect', description='Original native provider effect')
        async def original_provider_effect(self):
            witness.write_text('original native provider effect')
            return {'success': True}
        async def handle_task(self, task, *, arguments=None):
            # Native Feature subclasses can supply their own task handler. Its
            # admission belongs to TaskManager, independently of a tool leaf.
            await self.original_provider_effect()
            return task

    provider = NativeProvider(host)
    host.features[provider.name] = provider
    manager = manager_for(host.tool_effect_store.backend,
        NativeRecipientScope(host.did), owns_backend=False)
    await manager.initialize()
    manager.hooks_manager = host.hooks_manager
    manager.register_provider(provider.name, provider.get_agent_skills(), provider)
    original_create = manager.create_task
    async def record_create(*args, **kwargs):
        task = await original_create(*args, **kwargs)
        created.append(task.id)
        return task
    monkeypatch.setattr(manager, 'create_task', record_create)

    def withdraw():
        if change == 'host':
            provider.agent = foreign
        elif change == 'disabled':
            provider.enabled = False
        else:
            async def replacement_handler(task, *, arguments=None):
                witness.write_text('replacement native provider effect')
                return task
            provider.handle_task = replacement_handler
        withdrawals.append(True)
    if timing == 'permission':
        class Withdraw(Hook):
            def __init__(self):
                super().__init__('withdraw-original-async-feature', [HookEvent.PRE_TOOL_USE])
            async def execute(self, value):
                withdraw()
                return HookOutput.allow()
        host.hooks_manager.register(Withdraw())
    else:
        original_claim = host.tool_effect_store.claim
        async def after_ack(*args, **kwargs):
            receipt = await original_claim(*args, **kwargs)
            withdraw()
            return receipt
        monkeypatch.setattr(host.tool_effect_store, 'claim', after_ack)

    async def operation():
        return await manager.execute_skill(provider.name, 'original_provider_effect', {}, sync=False)
    try:
        with caller_context_scope(CallerContext.sovereign()):
            try:
                await drive_native(host, operation)
            except ResourceConflict:
                pass
        assert withdrawals, 'Original asynchronous producer never reached the withdrawal boundary'
        assert not witness.exists(), 'Native asynchronous acceptance executed a withdrawn Feature provider'
        assert not created, 'Native asynchronous acceptance persisted a withdrawn Feature provider'
    finally:
        await manager.close()


@pytest.mark.parametrize('change', ['host', 'disabled', 'replacement'])
async def test_mounted_feature_alias_retains_its_original_concrete_owner(
    concrete_feature, monkeypatch, change,
):
    p = concrete_feature
    p.host.unregister_dynamic_tools(p.feature.name)
    bindings = []
    p.host.register_dynamic_tools(p.feature.tool_name, p.feature.get_tools(),
        on_registered=lambda offered, binding: bindings.append(binding))
    original_claim = p.host.tool_effect_store.claim
    async def after_ack(*args, **kwargs):
        receipt = await original_claim(*args, **kwargs)
        if change == 'host':
            p.feature.agent = p.foreign
        elif change == 'disabled':
            p.feature.enabled = False
        else:
            p.host.features[p.feature.name] = p.replacement
        return receipt
    monkeypatch.setattr(p.host.tool_effect_store, 'claim', after_ack)
    with caller_context_scope(CallerContext.sovereign()):
        try:
            await bindings[0].invoke({}, session_id='original-alias-session',
                operation_id='original-alias-operation')
        except ResourceConflict:
            pass
    assert not p.foreign.native_effect_target.exists(), 'Mounted Feature alias entered a replacement host'
    assert not p.host.native_effect_target.exists(), 'Mounted Feature alias executed a withdrawn owner'
    assert await p.host.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_claims') == ('NOT_EXECUTED',)


@pytest.mark.parametrize('feature_base', [Feature, SDKFeature], ids=['core', 'sdk'])
@pytest.mark.parametrize('lane', ['direct', 'workflow', 'async'])
@pytest.mark.parametrize('error_type', [TimeoutError, ValueError], ids=['timeout', 'ordinary-error'])
async def test_custom_sync_native_provider_fences_its_original_uncertain_effect(
    native_tool_effect_agent, tmp_path, lane, error_type, feature_base,
):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.tasks.feature import TaskFeature
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from tests.utils.native_a2a import manager_for

    host = await native_tool_effect_agent(did='did:test:custom-sync-provider')
    witness = tmp_path / 'custom-sync-effects'
    original = error_type('Original custom handler reply unavailable')
    task_ids = []
    class CustomProvider(feature_base):
        tool_description = 'Opaque native provider original effect'
        async def initialize(self):
            pass
        @tool(name='custom_native_effect', description='Original opaque native effect')
        async def custom_native_effect(self):
            raise AssertionError('Custom task handler unexpectedly entered a canonical leaf')
        async def handle_task(self, task, *, arguments=None):
            task_ids.append(task.id)
            with witness.open('a') as stream:
                stream.write('once\n')
            raise original

    provider = CustomProvider(host)
    host.features[provider.name] = provider
    manager = manager_for(host.tool_effect_store.backend,
        NativeRecipientScope(host.did), owns_backend=False)
    await manager.initialize()
    manager.register_provider(provider.name, provider.get_agent_skills(), provider)
    workflow = TaskFeature(host)
    workflow.set_task_manager(manager)
    host.features[workflow.name] = workflow
    observed = []
    async def operation():
        if lane in {'direct', 'async'}:
            return await manager.execute_skill(provider.name, 'custom_native_effect', {}, sync=lane == 'direct')
        return await workflow.run_workflow([{'feature': provider.name,
            'skill': 'custom_native_effect', 'max_retries': 1, 'retry_delay_ms': 0}])
    try:
        with caller_context_scope(CallerContext.sovereign()):
            try:
                await drive_native(host, operation)
            except (TimeoutError, ValueError, ResourceOutcomeUnknown) as error:
                observed.extend(execution_evidence_carriers(error))
        assert witness.read_text().splitlines() == ['once'], 'Workflow repeated an opaque native Feature effect'
        assert any(isinstance(error, ResourceOutcomeUnknown) for error in observed), \
            'Opaque native Feature handler lost its original effect uncertainty'
        assert any(error is original for error in observed), 'Opaque native Feature handler replaced its original failure'
        expected_claims = [('RETURNED_OK',), ('UNCONFIRMED',)] if lane == 'async' else [('UNCONFIRMED',)]
        assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == expected_claims
        from kestrel_sdk.native_a2a.types import TaskState
        from kestrel_sovereign.a2a.lifecycle import task_outcome_unknown
        assert len(task_ids) == 1
        persisted = await manager.task_store.get_for_recipient(task_ids[0], manager.did)
        assert persisted.status.state == TaskState.TASK_STATE_FAILED and task_outcome_unknown(persisted)
    finally:
        await manager.close()


@pytest.mark.parametrize('lane', ['direct', 'workflow'])
async def test_custom_native_reply_contract_fences_its_original_effect(concrete_feature, lane):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sdk.native_a2a.types import TaskState
    from kestrel_sovereign.a2a.lifecycle import task_outcome_unknown
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.tasks.feature import TaskFeature
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from tests.utils.native_a2a import manager_for

    p = concrete_feature
    task_ids = []
    async def malformed_handler(task, *, arguments=None):
        task_ids.append(task.id)
        with p.host.native_effect_target.open('a') as stream:
            stream.write('once\n')
        return {'success': True}
    p.feature.handle_task = malformed_handler
    manager = manager_for(p.host.tool_effect_store.backend,
        NativeRecipientScope(p.host.did), owns_backend=False)
    await manager.initialize()
    manager.register_provider(p.feature.name, p.feature.get_agent_skills(), p.feature)
    workflow = TaskFeature(p.host)
    workflow.set_task_manager(manager)
    p.host.features[workflow.name] = workflow
    observed = []
    async def operation():
        if lane == 'direct':
            return await manager.execute_skill(p.feature.name, 'concrete_effect', {}, sync=True)
        return await workflow.run_workflow([{'feature': p.feature.name,
            'skill': 'concrete_effect', 'max_retries': 1, 'retry_delay_ms': 0}])
    try:
        with caller_context_scope(CallerContext.sovereign()):
            try:
                await drive_native(p.host, operation)
            except (AttributeError, TypeError, ResourceOutcomeUnknown) as error:
                observed.extend(execution_evidence_carriers(error))
        assert p.host.native_effect_target.read_text().splitlines() == ['once'], \
            'Workflow repeated an effect after a malformed native Task reply'
        assert any(isinstance(error, ResourceOutcomeUnknown) for error in observed), \
            'Malformed native Task reply lost its original effect uncertainty'
        assert any(isinstance(error, TypeError) for error in observed)
        assert await p.host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('UNCONFIRMED',)]
        assert len(task_ids) == 1
        persisted = await manager.task_store.get_for_recipient(task_ids[0], manager.did)
        assert persisted.status.state == TaskState.TASK_STATE_FAILED and task_outcome_unknown(persisted)
    finally:
        await manager.close()
