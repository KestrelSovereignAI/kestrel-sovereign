"""Actual named, subagent and native A2A leaves claim before file effects."""

import asyncio
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
from tests.unit.test_tool_effect_bound_operation import bound_operation


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest.mark.parametrize('feature_base', [Feature, SDKFeature], ids=['core', 'sdk'])
@pytest.mark.parametrize('shape', ['task-group', 'explicit-cause', 'attribute-hook', 'attribute-group-hook'])
async def test_composite_unknown_cannot_be_flattened_by_named_tool_wrappers(owner, tmp_path, feature_base, shape):
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.tools.effect_journal import ToolEffectOutcomeUnconfirmed

    witness = tmp_path / 'composite-unknown-effects'
    unknown = ResourceOutcomeUnknown('Original grouped or chained remote reply was lost')

    class OpaqueConflict(ResourceConflict):
        @property
        def __class__(self):
            raise RuntimeError('Application class hook replaced the original Core carrier')

    class OpaqueGroup(ExceptionGroup):
        @property
        def __class__(self):
            raise RuntimeError('Application group class hook replaced the original Core carrier')
    observability = ObservabilityStore(owner.tool_effect_store.backend, owns_backend=False)
    await observability.initialize()
    owner.observability_store = observability

    async def lose_reply():
        raise unknown

    class CompoundFeature(feature_base):
        @property
        def tool_description(self):
            return 'Original composite outcome'

        async def initialize(self):
            pass

        @tool(name='compound_effect', description='Write once before a composite lost reply')
        async def compound_effect(self, marker: str) -> dict:
            with witness.open('a') as stream:
                stream.write(marker + '\n')
            if shape == 'task-group':
                async with asyncio.TaskGroup() as children:
                    children.create_task(lose_reply())
            if shape == 'attribute-group-hook':
                raise OpaqueGroup('Original opaque group', [unknown])
            carrier = (OpaqueConflict('Original opaque wrapper') if shape == 'attribute-hook'
                       else ResourceConflict('Original wrapper precondition'))
            raise carrier from unknown

    feature = CompoundFeature(owner)
    owner.features[feature.name] = feature
    wrapped = next(tool for tool in feature.get_tools() if tool.name == 'compound_effect')

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        with pytest.raises(BaseException) as original:
            await agent.execute_named_tool(wrapped.name, {'marker': 'first'}, session_id='owned-composite-session')
        assert any(error is unknown for error in execution_evidence_carriers(original.value)), \
            'Native dispatch erased the original unknown before fencing the live turn'
        with pytest.raises(ToolEffectOutcomeUnconfirmed):
            await agent.execute_named_tool(wrapped.name, {'marker': 'unsafe second effect'}, session_id='owned-composite-session')

    try:
        with pytest.raises((ResourceOutcomeUnknown, BaseExceptionGroup)) as closed:
            await invoke(owner, 'original composite intent', invocation_id='owned-composite-intent')
        assert any(error is unknown for error in execution_evidence_carriers(closed.value))
        assert witness.read_text() == 'first\n'
        assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('UNCONFIRMED',)]
    finally:
        await observability.close()


async def test_async_registration_refusal_settles_the_native_task_without_dispatch(owner, tmp_path):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sdk.native_a2a.types import AgentSkill
    from tests.utils.native_a2a import manager_for

    witness = tmp_path / 'withdrawn-async-provider'
    manager = manager_for(owner.tool_effect_store.backend,
        NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()

    class Provider:
        async def handle_task(self, task, *, arguments=None):
            witness.write_text('withdrawn provider executed')
            return task

    manager.register_provider('owned-provider', [AgentSkill(id='owned-skill', name='Owned skill',
        description='Original asynchronous registration', tags=['test'])], Provider())
    denials = []
    try:
        accepted = await manager.execute_skill('owned-provider', 'owned-skill', {}, sync=False)
        manager.unregister_provider('owned-provider')
        try:
            await manager.drain_execution_tasks()
        except ResourceConflict as error:
            denials.append(error)
        persisted = await manager.task_store.get_for_recipient(accepted.id, manager.did)
    finally:
        try:
            await manager.close()
        except ResourceConflict as error:
            denials.append(error)
    assert not witness.exists()
    assert persisted.status.state == TaskState.TASK_STATE_FAILED, 'Withdrawn provider left its native task SUBMITTED'
    assert not denials, 'Known pre-dispatch refusal escaped its original task settlement'


@pytest.mark.parametrize('uncertain', [False, True])
async def test_async_feature_task_retains_its_original_journal_until_native_completion(owner, tmp_path, uncertain):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from tests.utils.native_a2a import manager_for

    witness = tmp_path / 'original-async-feature-effect'
    entered, resume, accepted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    background = []
    accepted_tasks = []
    lost_reply = ResourceOutcomeUnknown('Owned asynchronous reply lost after file effect')
    manager = manager_for(owner.tool_effect_store.backend,
        NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()

    class AsyncFileFeature(Feature):
        @property
        def tool_description(self):
            return 'Owned asynchronous feature task'

        async def initialize(self):
            pass

        async def handle_task(self, task, *, arguments=None):
            entered.set()
            await resume.wait()
            return await super().handle_task(task, arguments=arguments)

        @tool(name='owned_async_task_effect', description='Append the original task marker')
        async def owned_async_task_effect(self) -> dict:
            rows = await owner.tool_effect_store.database.fetchall(
                'SELECT parent_claim_id,state FROM tool_effect_claims')
            assert len(rows) == 2 and any(parent is not None and state == 'STARTED'
                for parent, state in rows), 'Async task did not retain its original dispatch claim'
            witness.write_text('one original asynchronous effect')
            if uncertain:
                raise lost_reply
            return {'success': True}

    feature = AsyncFileFeature(owner)
    owner.features[feature.name] = feature
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        task = await manager.execute_skill(feature.name, 'owned_async_task_effect', {}, sync=False)
        background.extend(manager._execution_tasks)
        accepted_tasks.append(task)
        accepted.set()
        return task

    invocation = asyncio.create_task(invoke(owner, 'original async intention', invocation_id='original-async-task-intent'))
    try:
        await asyncio.wait({invocation}, timeout=0.5)
        assert accepted.is_set(), 'Native asynchronous acceptance waited for a whole-worker privacy span'
        assert entered.is_set(), 'Accepted native asynchronous worker never entered its original handler'
        assert not invocation.done(), 'Originating journal retired before its accepted worker'
        assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
        resume.set()
        if uncertain:
            from kestrel_sovereign.execution_custody import execution_evidence_carriers

            with pytest.raises((ResourceOutcomeUnknown, BaseExceptionGroup)) as denied:
                await invocation
            assert any(error is lost_reply for error in execution_evidence_carriers(denied.value))
            task = accepted_tasks[0]
        else:
            task = await invocation
        await asyncio.gather(*background)
        assert witness.exists(), 'Accepted async task inherited a retired parent journal'
        assert witness.read_text() == 'one original asynchronous effect'
        persisted = await manager.task_store.get_for_recipient(task.id, manager.did)
        assert persisted.status.state == (TaskState.TASK_STATE_FAILED if uncertain else TaskState.TASK_STATE_COMPLETED)
        if uncertain:
            from kestrel_sovereign.a2a.lifecycle import task_outcome_unknown

            assert task_outcome_unknown(persisted)
        assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('CLOSED',)
        assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [
            ('RETURNED_OK',), ('UNCONFIRMED' if uncertain else 'RETURNED_OK',)]
        with pytest.raises(ToolEffectReconciliationRequired):
            await invoke(owner, 'original async intention', invocation_id='original-async-task-intent')
        assert witness.read_text() == 'one original asynchronous effect'
    finally:
        resume.set()
        await asyncio.gather(invocation, *background, return_exceptions=True)
        await manager.close()


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
    created = []
    create_native_task = manager.create_task

    async def capture_original_task(*args, **kwargs):
        task = await create_native_task(*args, **kwargs)
        created.append(task.id)
        return task

    monkeypatch.setattr(manager, 'create_task', capture_original_task)
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
        assert len(created) == 1
        persisted = await manager.task_store.get_for_recipient(created[0], manager.did)
        assert persisted.status.state == TaskState.TASK_STATE_FAILED, 'Synchronous withdrawn provider left a native task SUBMITTED'
    finally:
        await manager.close()


@pytest.mark.parametrize('lane,checkpoint,mutation', [
    ('task', 'approval', 'caller'), ('task', 'task-ack', 'caller'),
    ('task', 'claim-ack', 'caller'), ('task', 'task-ack', 'hook'),
    ('task', 'claim-ack', 'hook'), ('named', 'approval', 'caller'),
    ('named', 'claim-ack', 'caller'), ('named', 'claim-ack', 'hook'),
])
async def test_approved_binding_survives_native_admission_without_borrowed_arguments(
    owner, tmp_path, monkeypatch, lane, checkpoint, mutation,
):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from tests.utils.native_a2a import manager_for

    witness = tmp_path / 'approved-binding-effect'
    entered, resume = asyncio.Event(), asyncio.Event()
    supplied = {'payload': {'marker': 'approved'}}
    retained = []
    observability = ObservabilityStore(owner.tool_effect_store.backend, owns_backend=False)
    await observability.initialize()
    owner.observability_store = observability
    manager = manager_for(owner.tool_effect_store.backend,
        NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    manager.hooks_manager = owner.hooks_manager

    class SnapshotFeature(Feature):
        @property
        def tool_description(self):
            return 'Original approved binding'

        async def initialize(self):
            pass

        @tool(name='owned_snapshot_effect', description='Write only the original approved binding')
        async def owned_snapshot_effect(self, payload: dict) -> dict:
            witness.write_text(payload['marker'])
            return {'success': True}

    feature = SnapshotFeature(owner)
    owner.features[feature.name] = feature
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)

    class RetainApproval(Hook):
        def __init__(self):
            super().__init__('retain-original-approval', [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            if value.tool_name == 'owned_snapshot_effect':
                retained.append(value.tool_input)
                if checkpoint == 'approval':
                    entered.set()
                    await resume.wait()
            return HookOutput.allow()

    owner.hooks_manager.register(RetainApproval())
    if checkpoint == 'task-ack':
        create_task = manager.create_task

        async def hold_task_ack(*args, **kwargs):
            task = await create_task(*args, **kwargs)
            entered.set()
            await resume.wait()
            return task

        monkeypatch.setattr(manager, 'create_task', hold_task_ack)
    elif checkpoint == 'claim-ack':
        claim = owner.tool_effect_store.claim

        async def hold_claim_ack(*args, **kwargs):
            accepted = await claim(*args, **kwargs)
            assert not owner.tool_effect_store.database.owns_open_transaction
            entered.set()
            await resume.wait()
            return accepted

        monkeypatch.setattr(owner.tool_effect_store, 'claim', hold_claim_ack)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        if lane == 'task':
            return await manager.execute_skill(feature.name, 'owned_snapshot_effect', supplied)
        return await agent.execute_named_tool('owned_snapshot_effect', supplied, session_id='owned-snapshot')

    calling = asyncio.create_task(invoke(owner, 'original approved intent', invocation_id='owned-approved-intent'))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        (supplied if mutation == 'caller' else retained[-1])['payload']['marker'] = 'unapproved mutation'
        resume.set()
        result = await calling
        assert witness.read_text() == 'approved', 'Native effect executed a borrowed binding after approval'
        if lane == 'task':
            assert result.status.state == TaskState.TASK_STATE_COMPLETED
        assert await owner.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    finally:
        resume.set()
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await manager.close()
        await observability.close()


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
        states = await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims')
        assert states == ([] if lane == 'subagent' else [('NOT_EXECUTED',)])
    finally:
        await store.close()


@pytest.mark.parametrize('producer', ['direct', 'gather'])
async def test_accepted_async_worker_can_finish_its_original_nested_bound_operation(bound_operation, producer):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign._async_ownership import await_owned_task
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn
    from tests.utils.native_a2a import manager_for

    owner, binding, witness = bound_operation
    manager = manager_for(owner.tool_effect_store.backend,
        NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    accepted, resume = asyncio.Event(), asyncio.Event()
    turns = []

    class NestedAsyncFeature(Feature):
        tool_description = 'Finish the original accepted nested effect'

        async def initialize(self):
            pass

        async def handle_task(self, task, *, arguments=None):
            await resume.wait()
            return await super().handle_task(task, arguments=arguments)

        @tool(name='finish_original_nested_effect', description='Join the original native bound effect')
        async def finish_original_nested_effect(self) -> dict:
            async def original_effect():
                return await binding.invoke({'marker': 'accepted-original-nested-effect'},
                    session_id='accepted-original-nested-session',
                    operation_id='accepted-original-nested-intent')

            if producer == 'direct':
                return await original_effect()
            from kestrel_sovereign._async_ownership import gather_owned
            results = await gather_owned([original_effect],
                work_factory=owner._get_privacy_transition_lock().capture_native_work_factory(acquire_span=False),
                retain_originals=True)
            return results[0]

    feature = NestedAsyncFeature(owner)
    owner.features[feature.name] = feature
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        turns.append(current_tool_effect_turn(agent))
        task = await manager.execute_skill(feature.name, 'finish_original_nested_effect', {}, sync=False)
        accepted.set()
        return task

    calling = asyncio.create_task(invoke(owner, 'Original asynchronous nested intention',
        invocation_id='original-async-nested-intention'))
    try:
        await asyncio.wait({calling}, timeout=.5)
        assert accepted.is_set() and turns[0].closing and not calling.done()
        resume.set()
        result = await await_owned_task(calling)
        assert result.error is None, 'accepted asynchronous executor lost its original nested continuation while closing'
        persisted = await manager.task_store.get_for_recipient(result.result.id, owner.did)
        assert persisted.status.state == TaskState.TASK_STATE_COMPLETED
        assert witness.read_text() == 'accepted-original-nested-effect\n'
        assert await owner.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)] * 3
    finally:
        resume.set()
        await asyncio.gather(calling, return_exceptions=True)
        await manager.close()
