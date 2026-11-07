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


@pytest.mark.parametrize('lane', ['direct', 'workflow'])
async def test_canonical_native_encoding_failure_keeps_original_effect_receipt(
    native_tool_effect_agent, tmp_path, lane,
):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.tasks.feature import TaskFeature
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from tests.utils.native_a2a import manager_for

    host = await native_tool_effect_agent(did='did:test:original-native-encoding')
    witness = tmp_path / 'native-encoding-effects'
    class NativeEncodingFeature(Feature):
        tool_description = 'Original native result encoding'
        async def initialize(self):
            pass
        @tool(name='native_encoding_effect', description='Write before native encoding')
        async def native_encoding_effect(self):
            with witness.open('a') as stream:
                stream.write('once\n')
            return {'value': float('nan')}
    provider = NativeEncodingFeature(host)
    host.features[provider.name] = provider
    manager = manager_for(host.tool_effect_store.backend, NativeRecipientScope(host.did), owns_backend=False)
    await manager.initialize()
    manager.register_provider(provider.name, provider.get_agent_skills(), provider)
    workflow = TaskFeature(host)
    workflow.set_task_manager(manager)
    host.features[workflow.name] = workflow
    observed = []
    async def operation():
        if lane == 'direct':
            return await manager.execute_skill(provider.name, 'native_encoding_effect', {}, sync=True)
        return await workflow.run_workflow([{'feature': provider.name,
            'skill': 'native_encoding_effect', 'max_retries': 1, 'retry_delay_ms': 0}])
    try:
        with caller_context_scope(CallerContext.sovereign()):
            try:
                await drive_native(host, operation)
            except ResourceOutcomeUnknown as error:
                observed.extend(execution_evidence_carriers(error))
        assert witness.read_text().splitlines() == ['once'], 'Workflow repeated an acknowledged native effect after encoding failed'
        assert any(isinstance(error, ResourceOutcomeUnknown) for error in observed), 'Native encoding failure erased completion uncertainty'
        assert any(isinstance(error, ValueError) for error in observed)
        assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)], 'Completion failure rewrote an acknowledged effect claim'
    finally:
        await manager.close()


@pytest.mark.parametrize('lane', ['direct', 'workflow'])
@pytest.mark.parametrize('malformation', ['duplicate-artifact', 'missing-artifact-parts'])
async def test_native_task_invariant_failure_keeps_one_original_effect(concrete_feature, lane, malformation):
    from kestrel_sdk.native_a2a.types import Artifact, Part, TaskState
    from kestrel_sdk.resources import ResourceOutcomeUnknown
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
        task.status.state = TaskState.TASK_STATE_COMPLETED
        if malformation == 'duplicate-artifact':
            for _ in range(2):
                task.artifacts.append(Artifact(artifact_id='duplicate', parts=[Part(text='done')]))
        else:
            task.artifacts.append(Artifact(artifact_id='missing-parts'))
        return task
    p.feature.handle_task = malformed_handler
    manager = manager_for(p.host.tool_effect_store.backend, NativeRecipientScope(p.host.did), owns_backend=False)
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
            except (ValueError, ResourceOutcomeUnknown) as error:
                observed.extend(execution_evidence_carriers(error))
        assert p.host.native_effect_target.read_text().splitlines() == ['once'], 'Workflow repeated an effect after native Task invariants failed'
        assert any(isinstance(error, ResourceOutcomeUnknown) for error in observed), 'Native Task invariant failure erased completion uncertainty'
        assert len(task_ids) == 1
        persisted = await manager.task_store.get_for_recipient(task_ids[0], manager.did)
        assert persisted.status.state == TaskState.TASK_STATE_FAILED and task_outcome_unknown(persisted), 'Malformed native Task remained submitted after its effect'
        assert await p.host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('UNCONFIRMED',)]
    finally:
        await manager.close()


@pytest.mark.parametrize('error_type', [RuntimeError, ResourceConflict], ids=['ordinary-error', 'known-refusal'])
async def test_result_classification_failure_settles_and_fences_its_original_claim(native_tool_effect_agent, tmp_path, error_type):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.tools.effect_journal import execute_tool_effect

    host = await native_tool_effect_agent(did='did:test:original-result-classification')
    witness = tmp_path / 'classification-effects'
    original = error_type('Original result classifier refused')
    class UnreadableResult(dict):
        def get(self, *args, **kwargs):
            raise original
    async def effect():
        with witness.open('a') as stream:
            stream.write('first\n')
        return UnreadableResult()
    async def second():
        with witness.open('a') as stream:
            stream.write('repeated\n')
        return {'success': True}
    observed = []
    async def operation():
        try:
            await execute_tool_effect(host, registration='original-classification', arguments={}, execute=effect)
        except (RuntimeError, ResourceConflict, ResourceOutcomeUnknown) as error:
            observed.extend(execution_evidence_carriers(error))
        try:
            await execute_tool_effect(host, registration='unsafe-repeat', arguments={}, execute=second)
        except ResourceOutcomeUnknown:
            pass
    with caller_context_scope(CallerContext.sovereign()):
        try:
            await drive_native(host, operation)
        except ResourceOutcomeUnknown as error:
            observed.extend(execution_evidence_carriers(error))
    assert witness.read_text().splitlines() == ['first'], 'Result classification escaped the original live uncertainty fence'
    assert any(error is original for error in observed)
    assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('UNCONFIRMED',)], 'Result classification abandoned its original STARTED claim'


@pytest.mark.parametrize('lane', ['direct', 'workflow', 'async'])
async def test_native_completion_conflict_preserves_acknowledgement_and_fences_retry(concrete_feature, lane):
    from kestrel_sdk.native_a2a.types import Artifact, Part, TaskState
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.a2a.outcomes import TaskOutcomeConflictError
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.tasks.feature import TaskFeature
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from tests.utils.native_a2a import manager_for

    p = concrete_feature
    manager = manager_for(p.host.tool_effect_store.backend, NativeRecipientScope(p.host.did), owns_backend=False)
    await manager.initialize()
    task_ids = []
    async def conflicting_handler(task, *, arguments=None):
        task_ids.append(task.id)
        with p.host.native_effect_target.open('a') as stream:
            stream.write('once\n')
        # A real concurrent durable artifact must survive our incompatible
        # final result. Both values are independently valid native artifacts.
        await manager.task_store.add_artifact(task.id,
            Artifact(artifact_id='shared', parts=[Part(text='committed other value')]),
            recipient_agent_id=manager.did)
        task.status.state = TaskState.TASK_STATE_COMPLETED
        task.artifacts.append(Artifact(artifact_id='shared', parts=[Part(text='our incompatible value')]))
        return task
    p.feature.handle_task = conflicting_handler
    manager.register_provider(p.feature.name, p.feature.get_agent_skills(), p.feature)
    workflow = TaskFeature(p.host)
    workflow.set_task_manager(manager)
    p.host.features[workflow.name] = workflow
    observed = []
    persisted = None
    async def operation():
        if lane in {'direct', 'async'}:
            return await manager.execute_skill(p.feature.name, 'concrete_effect', {}, sync=lane == 'direct')
        return await workflow.run_workflow([{'feature': p.feature.name,
            'skill': 'concrete_effect', 'max_retries': 1, 'retry_delay_ms': 0}])
    try:
        with caller_context_scope(CallerContext.sovereign()):
            try:
                await drive_native(p.host, operation)
            except (ValueError, ResourceOutcomeUnknown, BaseExceptionGroup) as error:
                observed.extend(execution_evidence_carriers(error))
        if task_ids:
            persisted = await manager.task_store.get_for_recipient(task_ids[0], manager.did)
    finally:
        try:
            await manager.close()
        except (ValueError, ResourceOutcomeUnknown, BaseExceptionGroup) as error:
            observed.extend(execution_evidence_carriers(error))
    assert p.host.native_effect_target.read_text().splitlines() == ['once'], 'Workflow repeated an acknowledged effect after native completion conflicted'
    assert any(isinstance(error, ResourceOutcomeUnknown) for error in observed), 'Native completion conflict erased delivery uncertainty'
    assert any(isinstance(error, TaskOutcomeConflictError) for error in observed)
    assert len(task_ids) == 1
    assert persisted.artifacts[0].parts[0].text == 'committed other value'
    assert await p.host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)] * (2 if lane == 'async' else 1), 'Native completion failure rewrote acknowledged effect receipts'


@pytest.mark.parametrize('lane', ['direct', 'catcher', 'workflow'])
@pytest.mark.parametrize('failure', ['ordinary', 'conflict', 'unknown'])
async def test_native_task_post_completion_failure_cannot_replay_an_acknowledged_effect(
    concrete_feature, monkeypatch, lane, failure,
):
    from kestrel_sdk.native_a2a.types import TaskState
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.features.tasks.feature import TaskFeature
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn
    from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
    from tests.utils.native_a2a import manager_for

    p = concrete_feature
    class CompletingFeature(type(p.feature)):
        @tool(name='concrete_effect', description='Append one acknowledged native effect')
        async def concrete_effect(self):
            with self.agent.native_effect_target.open('a') as stream:
                stream.write('once\n')
            return {'success': True}
    feature = ensure_subagent_dispatch(CompletingFeature)(p.host)
    p.host.features[feature.name] = feature
    original = {'ordinary': ValueError, 'conflict': ResourceConflict,
        'unknown': ResourceOutcomeUnknown}[failure]('private-native-task-post-matcher')
    class FailingMatcher(Hook):
        calls = 0
        def __init__(self):
            super().__init__('original-native-task-post-matcher', [HookEvent.POST_TOOL_USE])
        def matches(self, tool_name):
            if tool_name != 'concrete_effect':
                return False
            self.calls += 1
            if self.calls == 1:
                raise original
            return True
        async def execute(self, value):
            return HookOutput.allow()
    matcher = FailingMatcher()
    p.host.hooks_manager.register(matcher)
    manager = manager_for(p.host.tool_effect_store.backend,
        NativeRecipientScope(p.host.did), owns_backend=False,
        hooks_manager=p.host.hooks_manager)
    await manager.initialize()
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)
    workflow = TaskFeature(p.host)
    workflow.set_task_manager(manager)
    p.host.features[workflow.name] = workflow
    task_ids, seen = [], {}
    create = manager.create_task
    async def record_create(*args, **kwargs):
        task = await create(*args, **kwargs)
        task_ids.append(task.id)
        return task
    monkeypatch.setattr(manager, 'create_task', record_create)
    async def operation():
        async def invoke():
            return await manager.execute_skill(feature.name, 'concrete_effect', {}, sync=True)
        try:
            if lane == 'workflow':
                seen['result'] = await workflow.run_workflow([{
                    'feature': feature.name, 'skill': 'concrete_effect',
                    'max_retries': 1, 'retry_delay_ms': 0}])
            else:
                seen['result'] = await invoke()
        except Exception as error:
            seen['first_error'] = error
        seen['unconfirmed'] = tuple(current_tool_effect_turn(p.host).unconfirmed)
        if lane == 'catcher':
            try:
                seen['second_result'] = await invoke()
            except Exception as error:
                seen['second_error'] = error
    try:
        with caller_context_scope(CallerContext.sovereign()):
            try:
                await drive_native(p.host, operation)
            except ResourceOutcomeUnknown as error:
                seen['root_error'] = error
        persisted = [await manager.task_store.get_for_recipient(task_id, manager.did)
            for task_id in task_ids]
    finally:
        await manager.close()
    assert p.host.native_effect_target.read_text().splitlines() == ['once'], (
        'Native Task post-completion failure replayed its acknowledged physical effect')
    assert any(original in execution_evidence_carriers(error) for error in seen['unconfirmed']), (
        'Native Task post-completion failure lost original live uncertainty')
    assert 'root_error' in seen and original in execution_evidence_carriers(seen['root_error'])
    assert len(task_ids) == matcher.calls == 1, 'Uncertain native completion admitted another Task'
    assert all(task.status.state == TaskState.TASK_STATE_COMPLETED for task in persisted), (
        'Native completion failure rewrote its acknowledged Task outcome')
    assert await p.host.tool_effect_store.database.fetchall(
        'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]


@pytest.mark.parametrize('consumer', ['workflow', 'command'])
@pytest.mark.parametrize('owner_change', [False, True], ids=['original-host', 'owner-changed'])
async def test_native_task_result_readers_cannot_replay_after_decode_failure(
    concrete_feature, consumer, owner_change,
):
    from kestrel_sdk.native_a2a.helpers.proto_helpers import new_raw_artifact
    from kestrel_sdk.native_a2a.types import TaskState
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.features.tasks.feature import TaskFeature
    from tests.utils.native_a2a import manager_for

    p = concrete_feature
    task_ids, seen = [], {}
    async def native_handler(task, *, arguments=None):
        task_ids.append(task.id)
        with p.host.native_effect_target.open('a') as stream:
            stream.write('once\n')
        task.status.state = TaskState.TASK_STATE_COMPLETED
        task.artifacts.append(new_raw_artifact('unreadable-native-result',
            b'not-json', media_type='application/json'))
        if owner_change:
            p.feature.agent = p.foreign
            workflow.agent = p.foreign
        return task
    p.feature.handle_task = native_handler
    manager = manager_for(p.host.tool_effect_store.backend,
        NativeRecipientScope(p.host.did), owns_backend=False)
    await manager.initialize()
    manager.register_provider(p.feature.name, p.feature.get_agent_skills(), p.feature,
        command_prefixes={'!completed-native': 'concrete_effect'})
    workflow = TaskFeature(p.host)
    workflow.set_task_manager(manager)
    p.host.features[workflow.name] = workflow
    async def operation():
        if consumer == 'workflow':
            seen['result'] = await workflow.run_workflow([{
                'feature': p.feature.name, 'skill': 'concrete_effect',
                'max_retries': 1, 'retry_delay_ms': 0}])
        else:
            for attempt in range(2):
                try:
                    seen[f'result-{attempt}'] = await manager.execute_command('!completed-native')
                except Exception as error:
                    seen[f'error-{attempt}'] = error
    try:
        with caller_context_scope(CallerContext.sovereign()):
            try:
                await drive_native(p.host, operation)
            except ResourceOutcomeUnknown as error:
                seen['root_error'] = error
        persisted = [await manager.task_store.get_for_recipient(task_id, manager.did)
            for task_id in task_ids]
    finally:
        await manager.close()
    assert p.host.native_effect_target.read_text().splitlines() == ['once'], (
        'Native Task result reader replayed its completed effect after decode failed')
    assert len(task_ids) == 1
    assert 'root_error' in seen
    assert any(isinstance(error, ValueError)
        for error in execution_evidence_carriers(seen['root_error']))
    assert persisted[0].status.state == TaskState.TASK_STATE_COMPLETED
    assert await manager.task_store.backend.fetch_all('SELECT COUNT(*) FROM a2a_tasks') == [(1,)]
    assert await p.foreign.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == []
    assert await p.host.tool_effect_store.database.fetchall(
        'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]


@pytest.mark.parametrize('lane', ['direct', 'catcher', 'workflow', 'async'])
@pytest.mark.parametrize('state', ['failed', 'completed'])
async def test_opaque_native_unknown_reply_fences_its_original_input(
    concrete_feature, lane, state,
):
    from kestrel_sdk.native_a2a.helpers.proto_helpers import new_raw_artifact
    from kestrel_sdk.native_a2a.types import TaskState
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.a2a.lifecycle import mark_outcome_unknown, task_outcome_unknown
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.tasks.feature import TaskFeature
    from tests.utils.native_a2a import manager_for

    p = concrete_feature
    task_ids, seen = [], {}
    expected_state = (TaskState.TASK_STATE_FAILED if state == 'failed'
        else TaskState.TASK_STATE_COMPLETED)
    async def native_handler(task, *, arguments=None):
        task_ids.append(task.id)
        with p.host.native_effect_target.open('a') as stream:
            stream.write('once\n')
        task.status.state = expected_state
        task.artifacts.append(new_raw_artifact('original-native-reply',
            b'{"partial":"receipt"}', media_type='application/json'))
        mark_outcome_unknown(task)
        return task
    p.feature.handle_task = native_handler
    manager = manager_for(p.host.tool_effect_store.backend,
        NativeRecipientScope(p.host.did), owns_backend=False)
    await manager.initialize()
    manager.register_provider(p.feature.name, p.feature.get_agent_skills(), p.feature)
    workflow = TaskFeature(p.host)
    workflow.set_task_manager(manager)
    p.host.features[workflow.name] = workflow
    async def operation():
        if lane == 'workflow':
            seen['result'] = await workflow.run_workflow([{
                'feature': p.feature.name, 'skill': 'concrete_effect',
                'max_retries': 1, 'retry_delay_ms': 0}])
        else:
            seen['result'] = await manager.execute_skill(p.feature.name,
                'concrete_effect', {}, sync=lane != 'async')
            if lane == 'catcher':
                try:
                    seen['second_result'] = await manager.execute_skill(
                        p.feature.name, 'concrete_effect', {}, sync=True)
                except Exception as error:
                    seen['second_error'] = error
    try:
        with caller_context_scope(CallerContext.sovereign()):
            try:
                await drive_native(p.host, operation)
            except ResourceOutcomeUnknown as error:
                seen['root_error'] = error
        persisted = [await manager.task_store.get_for_recipient(task_id, manager.did)
            for task_id in task_ids]
    finally:
        await manager.close()
    assert p.host.native_effect_target.read_text().splitlines() == ['once'], (
        'Explicit native unknown Task reply permitted another physical effect')
    assert 'root_error' in seen, 'Explicit native unknown Task reply completed its original input'
    assert len(task_ids) == 1
    assert persisted[0].status.state == expected_state and task_outcome_unknown(persisted[0]), (
        'Native explicit unknown reply lost its validated Task state')
    assert persisted[0].artifacts[0].parts[0].raw == b'{"partial":"receipt"}', (
        'Native explicit unknown reply lost its validated artifact receipt')
    states = await p.host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims')
    assert states == ([('RETURNED_OK',)] if lane == 'async' else []) + [('UNCONFIRMED',)]
