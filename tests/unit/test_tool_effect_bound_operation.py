"""Actual bound-tool dispatch, native lifecycle, SQL claims and file effects."""

import asyncio
import pytest
import pytest_asyncio
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sovereign.agent.invocation import current_invocation_id
from kestrel_sovereign.tool_effect_store import (
    ToolEffectOperationConflict, ToolEffectReconciliationRequired,
)
from tests.unit.test_tool_effect_execution import owner as owner


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest_asyncio.fixture
async def bound_operation(owner, tmp_path):
    store = ObservabilityStore(owner.tool_effect_store.database.backend, owns_backend=False)
    await store.initialize()
    owner.observability_store = store
    witness = tmp_path / 'bound-operation-effects'

    class FileTool:
        name = 'owned.effect'
        schema = ToolSchema(name, 'Append the owned marker', ToolCategory.UTILITY,
            parameters=[ToolParameter('marker', 'string', 'Owned marker', required=True)],
            is_concurrency_safe=True)

        async def execute(self, *, marker):
            assert owner._request_generation_for_current_task(current_invocation_id()) is not None
            claims = await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims')
            assert claims[-1] == ('STARTED',), 'Native dispatcher bypassed durable effect admission'
            assert owner.tool_effect_store.database.owns_open_transaction is False
            async with owner._get_privacy_transition_lock():
                with witness.open('a') as stream:
                    stream.write(marker + '\n')
            return {'success': True, 'marker': marker}

    mounted = []
    owner.register_dynamic_tools('mcp:owned-operation', [FileTool()],
        on_registered=lambda tool, binding: mounted.append(binding))
    binding, = mounted
    yield owner, binding, witness
    await store.close()


async def _foreign_effect(owner, operation):
    class RelayTool:
        name = 'owned.foreign_relay'
        schema = ToolSchema(name, 'Join an original foreign continuation', ToolCategory.UTILITY)

        async def execute(self):
            return await operation()

    mounted = []
    owner.register_dynamic_tools('owned-foreign-continuation', [RelayTool()],
        on_registered=lambda tool, binding: mounted.append(binding))
    relay, = mounted
    async with owner._get_privacy_transition_lock():
        return await relay.invoke({}, session_id='owned-foreign-session', operation_id='owned-foreign-intent')


async def _retire_foreign_scenario(calling, names):
    children = [task for task in asyncio.all_tasks() if task is not calling
        and any(name in task.get_name() for name in names)] if not calling.done() else []
    for child in children:
        child.cancel()
    if not calling.done():
        calling.cancel()
    await asyncio.gather(calling, *children, return_exceptions=True)


async def test_parallel_batch_retains_original_privacy_ownership_for_its_native_leaves(bound_operation):
    await _parallel_scenario(bound_operation, foreign=False)


async def test_nested_foreign_parallel_batch_retains_its_original_privacy_frame(bound_operation):
    await _parallel_scenario(bound_operation, foreign=True)


async def _parallel_scenario(bound_operation, *, foreign):
    from kestrel_sovereign.llm.adapter import ToolCall
    from tests.unit.test_tool_effect_invocation import governed

    owner, binding, witness = bound_operation
    messages, results = [], []

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        async def batch():
            await agent._execute_tool_batch_at_stop_boundary(
                [ToolCall(id=str(index), name=binding.name, arguments={'marker': marker})
                    for index, marker in enumerate(('first', 'second'))],
                {}, {binding.name}, messages, 0, user_input,
                tool_results=results, session_id='owned-parallel-session')
            return {'success': True}

        if foreign:
            return await _foreign_effect(agent, batch)
        async with agent._get_privacy_transition_lock():
            return await batch()

    calling = asyncio.create_task(invoke(owner, 'original parallel intention',
        invocation_id='owned-parallel-intent'))
    try:
        _done, pending = await asyncio.wait({calling}, timeout=0.5)
        assert not pending, 'Parallel native tools lost the privacy frame held by their joining batch'
        await calling
        assert sorted(witness.read_text().splitlines()) == ['first', 'second']
        assert [row['name'] for row in results] == [binding.name, binding.name]
        assert [row['tool_call_id'] for row in results] == ['0', '1']
        assert await owner.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)] * (3 if foreign else 2)
        assert not owner._active_request_ids
    finally:
        # Retire the actual batch children first if the regression deadlocks;
        # their parent deliberately shields admitted effects from caller Stop.
        children = [task for task in asyncio.all_tasks()
            if task.get_name().startswith('orchestrator-parallel-tool:')]
        for task in children:
            task.cancel()
        await asyncio.gather(*children, return_exceptions=True)
        await _retire_foreign_scenario(calling, (binding.name, 'owned.foreign_relay'))


async def test_external_async_a2a_worker_retains_original_privacy_ownership(bound_operation):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.base import Feature, tool
    from tests.utils.native_a2a import manager_for

    owner, _binding, witness = bound_operation
    manager = manager_for(owner.tool_effect_store.backend,
        NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    accepted = []

    class BackgroundFeature(Feature):
        @property
        def tool_description(self):
            return 'Native asynchronous file effect'

        async def initialize(self):
            pass

        @tool(name='owned_background_file', description='Write the accepted task marker')
        async def owned_background_file(self) -> dict:
            async with owner._get_privacy_transition_lock():
                witness.write_text('one accepted background effect')
            return {'success': True}

    feature = BackgroundFeature(owner)
    owner.features[feature.name] = feature
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)

    class DispatchTool:
        name = 'owned.async_dispatch'
        schema = ToolSchema(name, 'Accept an asynchronous native task', ToolCategory.UTILITY)

        async def execute(self):
            task = await manager.execute_skill(feature.name, 'owned_background_file', {}, sync=False)
            accepted.append(task)
            return {'success': True, 'task_id': task.id}

    mounted = []
    owner.register_dynamic_tools('owned-background', [DispatchTool()],
        on_registered=lambda tool, binding: mounted.append(binding))
    binding, = mounted
    calling = asyncio.create_task(binding.invoke({}, session_id='owned-background-session',
        operation_id='owned-background-intent'))
    try:
        _done, pending = await asyncio.wait({calling}, timeout=0.5)
        assert not pending, 'Asynchronous A2A worker lost its joining external lifecycle privacy frame'
        assert (await calling)['success'] is True
        task, = accepted
        from kestrel_sdk.native_a2a.types import TaskState

        persisted = await manager.task_store.get_for_recipient(task.id, manager.did)
        assert persisted.status.state == TaskState.TASK_STATE_COMPLETED
        assert witness.read_text() == 'one accepted background effect'
        assert await owner.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)] * 3
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT state FROM tool_effect_operations') == ('CLOSED',)
        assert not owner._active_request_ids
    finally:
        workers = tuple(manager._execution_tasks)
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await manager.close()


@pytest.mark.parametrize('lane', ['bound', 'named', 'audit'])
async def test_nested_external_operation_retains_its_original_privacy_frame(bound_operation, lane):
    await _nested_scenario(bound_operation, lane, foreign=False)


@pytest.mark.parametrize('lane', ['bound', 'named', 'audit'])
async def test_nested_foreign_operation_retains_its_original_privacy_frame(bound_operation, lane):
    await _nested_scenario(bound_operation, lane, foreign=True)


async def _nested_scenario(bound_operation, lane, *, foreign):
    from tests.unit.test_tool_effect_invocation import governed
    from kestrel_sovereign.tools.effect_journal import execute_tool_effect

    owner, binding, witness = bound_operation

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        async def outer_effect():
            context = dict(session_id='owned-nested-session', operation_id='owned-nested-intent')
            if lane == 'bound':
                return await binding.invoke({'marker': 'nested'}, **context)
            if lane == 'audit':
                from kestrel_sovereign._async_ownership import raise_owned_outcome
                from kestrel_sovereign.tools.effect_invocation import ToolEffectOperation

                async def audit(outcome):
                    # Bound the native acquisition itself so a broken handoff
                    # can retire and report a failed audit without hanging its
                    # original owner during the regression's cleanup.
                    async with asyncio.timeout(0.2), agent._get_privacy_transition_lock():
                        with witness.open('a') as stream:
                            stream.write('audit\n')

                outcome = await agent._execute_tool_operation_at_stop_boundary(
                    lambda: agent.execute_named_tool(binding.name, {'marker': 'nested'},
                        session_id='owned-nested-session'),
                    name='owned-nested-audit', cancel_on_caller_cancellation=True,
                    finalize=audit,
                    tool_operation=ToolEffectOperation('native_audit', 'owned-nested-intent',
                        'owned-registry', binding.name, {'marker': 'nested'}, 'owned-nested-session'))
                return raise_owned_outcome(outcome, operation='original nested audit', retain_originals=True)
            return await agent.execute_named_tool(binding.name, {'marker': 'nested'}, **context)

        return await execute_tool_effect(agent, registration='owned-parent-effect',
            arguments={}, execute=outer_effect)

    async def call():
        operation = lambda: invoke(owner, 'original nested intention', invocation_id='owned-parent-intent')
        return await _foreign_effect(owner, operation) if foreign else await operation()

    calling = asyncio.create_task(call())
    try:
        _done, pending = await asyncio.wait({calling}, timeout=0.5)
        assert not pending, 'Nested external tool lost its original held privacy frame'
        if lane == 'audit':
            from kestrel_sovereign._async_ownership import await_owned_task

            outcome = await await_owned_task(calling)
            healthy = outcome.error is None
            assert healthy, 'Mandatory native audit lost its original held privacy frame'
            result = outcome.result
        else:
            result = await calling
        assert result['success'] is True
        assert witness.read_text() == ('nested\naudit\n' if lane == 'audit' else 'nested\n')
        assert await owner.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)] * (3 if foreign else 2)
        assert not owner._active_request_ids
    finally:
        await _retire_foreign_scenario(calling, (binding.name, 'owned.foreign_relay', 'owned-nested-audit'))


async def test_bound_operation_fences_retry_before_another_native_effect(bound_operation):
    owner, binding, witness = bound_operation
    context = dict(session_id='owned-app-session', operation_id='original-intent', source='mcp-app')
    with caller_context_scope(CallerContext.sovereign(credential='original-key')):
        result = await binding.invoke({'marker': 'one'}, **context)
        assert result == {'success': True, 'marker': 'one'}
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('CLOSED',)
    with caller_context_scope(CallerContext.sovereign(credential='rotated-key')):
        with pytest.raises(ToolEffectReconciliationRequired):
            await binding.invoke({'marker': 'one'}, **context)
        with pytest.raises(ToolEffectOperationConflict):
            await binding.invoke({'marker': 'changed'}, **context)
        await binding.invoke({'marker': 'fresh'}, **dict(context, operation_id='new-intent'))
    with caller_context_scope(CallerContext.authenticated('another-person')):
        await binding.invoke({'marker': 'other principal'}, **context)
    assert witness.read_text() == 'one\nfresh\nother principal\n'
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (3,)
    assert not owner._active_request_ids


@pytest.mark.parametrize('operation_id', ['', None, 3, 'x' * 257])
async def test_invalid_operation_id_cannot_reach_native_admission(bound_operation, operation_id):
    owner, binding, witness = bound_operation
    with pytest.raises(ValueError, match='invocation id'):
        await binding.invoke({'marker': 'forbidden'}, session_id='owned-session', operation_id=operation_id)
    assert not witness.exists()
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (0,)


async def test_missing_operation_id_is_a_required_interface_failure(bound_operation):
    _owner, binding, witness = bound_operation
    with pytest.raises(TypeError, match='operation_id'):
        await binding.invoke({'marker': 'forbidden'}, session_id='owned-session')
    assert not witness.exists()


@pytest.mark.parametrize('lane', ['bound', 'in-turn'])
async def test_external_tool_and_real_privacy_transition_follow_one_lock_order(bound_operation, monkeypatch, lane):
    owner, binding, witness = bound_operation
    acknowledged, resume = asyncio.Event(), asyncio.Event()
    original_claim = owner.tool_effect_store.claim

    async def pause_after_native_ack(*args, **kwargs):
        claim = await original_claim(*args, **kwargs)
        acknowledged.set()
        await resume.wait()
        return claim

    monkeypatch.setattr(owner.tool_effect_store, 'claim', pause_after_native_ack)
    transitioned = asyncio.Event()

    async def transition():
        async with owner.privacy_transition():
            transitioned.set()

    from tests.unit.test_tool_effect_invocation import governed

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        return await agent.execute_named_tool(binding.name, {'marker': 'ordered'},
            session_id='owned-lock-order')

    calling = asyncio.create_task(
        binding.invoke({'marker': 'ordered'}, session_id='owned-lock-order',
            operation_id='owned-lock-order-intent') if lane == 'bound' else
        invoke(owner, 'original in-turn intent', invocation_id='owned-lock-order-intent'))
    changing = None
    try:
        await asyncio.wait_for(acknowledged.wait(), 2)
        changing = asyncio.create_task(transition())
        await asyncio.sleep(0.02)
        resume.set()
        _done, pending = await asyncio.wait({calling, changing}, timeout=0.3)
        assert not pending, 'Tool/privacy lock order created a native mutex/gate cycle'
        assert (await calling)['success'] is True
        await changing
        assert transitioned.is_set() and witness.read_text() == 'ordered\n'
        assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_claims') == ('RETURNED_OK',)
    finally:
        resume.set()
        for task in (calling, changing):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(calling, *([changing] if changing is not None else []), return_exceptions=True)


async def test_external_admission_rechecks_privacy_before_dispatch(bound_operation, monkeypatch):
    from kestrel_sovereign._async_ownership import await_owned_task
    from kestrel_sovereign.features.privacy.feature import PrivacyAgent
    from kestrel_sovereign.privacy import PrivacyMode

    owner, binding, witness = bound_operation
    acknowledged, resume = asyncio.Event(), asyncio.Event()
    admit = owner.tool_effect_store.admit

    async def hold_native_admission(*args, **kwargs):
        attempt = await admit(*args, **kwargs)
        assert not owner.tool_effect_store.database.owns_open_transaction
        acknowledged.set()
        await resume.wait()
        return attempt

    async def revoke():
        async with owner.privacy_transition():
            owner.privacy_agent = PrivacyAgent(owner.storage, PrivacyMode.EPHEMERAL)

    monkeypatch.setattr(owner.tool_effect_store, 'admit', hold_native_admission)
    calling = asyncio.create_task(binding.invoke({'marker': 'forbidden'},
        session_id='owned-lifecycle-policy', operation_id='owned-lifecycle-policy-intent'))
    changing = None
    try:
        await asyncio.wait_for(acknowledged.wait(), 2)
        changing = asyncio.create_task(revoke())
        await asyncio.sleep(0.02)
        resume.set()
        await changing
        outcome = await await_owned_task(calling)
        assert outcome.error is None
        assert outcome.result == {'success': False, 'error': 'tool_effect_privacy_mode_refused'}
        assert not witness.exists(), 'Revoked privacy policy admitted the original effect'
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT COUNT(*) FROM tool_effect_claims') == (0,)
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT state FROM tool_effect_operations') == ('OPEN',)
        assert not owner._active_request_ids
    finally:
        resume.set()
        for task in (calling, changing):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(calling, *([changing] if changing is not None else []), return_exceptions=True)


async def test_external_named_dispatch_owns_its_explicit_operation_without_a_model_turn(bound_operation):
    owner, binding, witness = bound_operation
    context = dict(session_id='owned-voice-session', operation_id='owned-voice-intent', source='voice_realtime')
    with caller_context_scope(CallerContext.authenticated('owned-voice-principal')):
        result = await owner.execute_named_tool(binding.name, {'marker': 'one'}, **context)
        assert result['success'] is True
        with pytest.raises(ToolEffectReconciliationRequired):
            await owner.execute_named_tool(binding.name, {'marker': 'one'}, **context)
    assert witness.read_text() == 'one\n'
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('CLOSED',)
    assert not owner._active_request_ids


async def test_external_named_intent_and_effect_share_the_same_owned_argument_snapshot(bound_operation, monkeypatch):
    owner, binding, witness = bound_operation
    arguments = {'marker': 'original'}
    original_admit = owner.tool_effect_store.admit

    async def mutate_caller_arguments_after_admission(**options):
        attempt = await original_admit(**options)
        arguments['marker'] = 'caller mutation after admission'
        return attempt

    monkeypatch.setattr(owner.tool_effect_store, 'admit', mutate_caller_arguments_after_admission)
    await owner.execute_named_tool(binding.name, arguments, session_id='owned-mutable-caller',
        operation_id='original-mutable-intent', source='voice_realtime')
    assert witness.read_text() == 'original\n', 'Caller mutation changed the effect after its original intent was journaled'
    assert arguments == {'marker': 'caller mutation after admission'}


async def test_independent_external_approvals_do_not_own_the_effect_mutex(bound_operation):
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput

    owner, binding, witness = bound_operation
    requests, release = asyncio.Queue(), asyncio.Event()

    class AwaitApproval(Hook):
        def __init__(self):
            super().__init__('owned-concurrent-approvals', [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            if value.tool_name == binding.name:
                requests.put_nowait(value.tool_input['marker'])
                await release.wait()
            return HookOutput.allow()

    owner.hooks_manager.register(AwaitApproval())
    calls = [asyncio.create_task(binding.invoke({'marker': str(index)},
        session_id='owned-concurrent-approval-session', operation_id=f'owned-approval-{index}'))
        for index in range(2)]
    try:
        await asyncio.wait(calls, timeout=0.5)
        assert requests.qsize() == 2, 'An independent external approval waited behind another approval privacy span'
        assert not witness.exists(), 'Native effect executed before its original approval'
        release.set()
        assert all(result['success'] for result in await asyncio.gather(*calls))
        assert sorted(witness.read_text().splitlines()) == ['0', '1']
        assert await owner.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',), ('RETURNED_OK',)]
    finally:
        release.set()
        await asyncio.gather(*calls, return_exceptions=True)


async def test_privacy_revocation_during_external_approval_refuses_the_original_effect(bound_operation):
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
    from kestrel_sovereign.features.privacy.feature import PrivacyAgent
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign._async_ownership import await_owned_task

    owner, binding, witness = bound_operation
    requested, release = asyncio.Event(), asyncio.Event()

    class AwaitApproval(Hook):
        def __init__(self):
            super().__init__('owned-revoked-approval', [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            if value.tool_name == binding.name:
                requested.set()
                await release.wait()
            return HookOutput.allow()

    async def revoke():
        async with owner.privacy_transition():
            owner.privacy_agent = PrivacyAgent(owner.storage, PrivacyMode.EPHEMERAL)

    owner.hooks_manager.register(AwaitApproval())
    calling = asyncio.create_task(binding.invoke({'marker': 'forbidden'},
        session_id='owned-revoked-approval-session', operation_id='owned-revoked-approval'))
    changing = None
    try:
        await asyncio.wait_for(requested.wait(), 2)
        changing = asyncio.create_task(revoke())
        _, pending = await asyncio.wait({changing}, timeout=0.5)
        assert not pending, 'Pending external approval prevented the original privacy revocation'
        await changing
        release.set()
        outcome = await await_owned_task(calling)
        assert outcome.error is None
        assert outcome.result == {'success': False, 'error': 'tool_effect_privacy_mode_refused'}
        assert not witness.exists()
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT COUNT(*) FROM tool_effect_claims') == (0,)
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT state FROM tool_effect_operations') == ('OPEN',)
    finally:
        release.set()
        await asyncio.gather(calling, *([changing] if changing is not None else []), return_exceptions=True)


@pytest.mark.parametrize('lane', ['external', 'normal-turn', 'nested-normal-turn', 'async-parent', 'parallel-parent'])
async def test_async_native_sibling_effects_exclude_each_other_without_delaying_acceptance(bound_operation, lane):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.base import Feature, tool
    from kestrel_sdk.native_a2a.types import TaskState
    from tests.utils.native_a2a import manager_for

    owner, _binding, witness = bound_operation
    manager = manager_for(owner.tool_effect_store.backend, NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    entered, accepted = [], []
    first_started, all_accepted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SiblingFeature(Feature):
        @property
        def tool_description(self):
            return 'Mutually exclusive original background effects'

        async def initialize(self):
            pass

        @tool(name='owned_sibling_effect', description='Write under the native privacy mutex')
        async def owned_sibling_effect(self, marker: str) -> dict:
            async with owner._get_privacy_transition_lock():
                entered.append(marker)
                if len(entered) == 1:
                    first_started.set()
                    await release.wait()
                with witness.open('a') as stream:
                    stream.write(marker + '\n')
            return {'success': True}

    feature = SiblingFeature(owner)
    owner.features[feature.name] = feature
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)

    class DispatchTool:
        name = 'owned.sibling_dispatch'
        schema = ToolSchema(name, 'Accept two asynchronous native effects', ToolCategory.UTILITY,
            is_concurrency_safe=True)

        async def execute(self):
            for marker in ('first', 'second'):
                accepted.append(await manager.execute_skill(feature.name, 'owned_sibling_effect',
                    {'marker': marker}, sync=False))
            all_accepted.set()
            return {'success': True}

    mounted = []
    owner.register_dynamic_tools('owned-sibling-background', [DispatchTool()],
        on_registered=lambda tool, binding: mounted.append(binding))
    binding, = mounted

    class ParentDispatchFeature(Feature):
        @property
        def tool_description(self):
            return 'Original nested asynchronous workflow'

        async def initialize(self):
            pass

        @tool(name='owned_nested_sibling_dispatch', description='Join a nested native sibling dispatch')
        async def owned_nested_sibling_dispatch(self) -> dict:
            return await binding.invoke({}, session_id='owned-async-parent-session',
                operation_id='owned-async-parent-child')

    parent_feature = ParentDispatchFeature(owner)
    owner.features[parent_feature.name] = parent_feature
    manager.register_provider(parent_feature.name, parent_feature.get_agent_skills(), parent_feature)

    class ParentAsyncTool:
        name = 'owned.async_parent_dispatch'
        schema = ToolSchema(name, 'Accept an asynchronous parent', ToolCategory.UTILITY)

        async def execute(self):
            await manager.execute_skill(parent_feature.name, 'owned_nested_sibling_dispatch', {}, sync=False)
            return {'success': True}

    parents = []
    owner.register_dynamic_tools('owned-async-parent', [ParentAsyncTool()],
        on_registered=lambda tool, binding: parents.append(binding))
    parent_binding, = parents

    class ObserverTool:
        name = 'owned.parallel_observer'
        schema = ToolSchema(name, 'Observe exclusion beneath another parallel parent',
            ToolCategory.UTILITY, is_concurrency_safe=True)

        async def execute(self):
            async with owner._get_privacy_transition_lock():
                entered.append('observer')
                with witness.open('a') as stream:
                    stream.write('observer\n')
            return {'success': True}

    observers = []
    owner.register_dynamic_tools('owned-parallel-observer', [ObserverTool()],
        on_registered=lambda tool, binding: observers.append(binding))
    observer_binding, = observers

    if lane == 'parallel-parent':
        from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput

        class ObserveAfterChildStarts(Hook):
            def __init__(self):
                super().__init__('native-subtree-observer-order', [HookEvent.PRE_TOOL_USE])

            async def execute(self, value):
                if value.tool_name == observer_binding.name:
                    await first_started.wait()
                return HookOutput.allow()

        owner.hooks_manager.register(ObserveAfterChildStarts())

    class RelayTool:
        name = 'owned.sibling_relay'
        schema = ToolSchema(name, 'Join the original nested native dispatch', ToolCategory.UTILITY)

        async def execute(self):
            if lane == 'parallel-parent':
                from kestrel_sovereign.llm.adapter import ToolCall
                await owner._execute_tool_batch_at_stop_boundary([
                    ToolCall(id='parent', name=binding.name, arguments={}),
                    ToolCall(id='observer', name=observer_binding.name, arguments={}),
                ], {}, {binding.name, observer_binding.name}, [], 0,
                    'Original parallel native parent and observer', tool_results=[],
                    session_id='owned-parallel-parent')
                return {'success': True}
            return await binding.invoke({}, session_id='owned-nested-sibling-session',
                operation_id='owned-nested-sibling-child')

    relays = []
    owner.register_dynamic_tools('owned-sibling-relay', [RelayTool()],
        on_registered=lambda tool, binding: relays.append(binding))
    relay, = relays

    async def call():
        if lane == 'async-parent':
            async with owner._get_privacy_transition_lock():
                return await parent_binding.invoke({}, session_id='owned-async-parent-session',
                    operation_id='owned-async-parent-intent')
        if lane == 'parallel-parent':
            from kestrel_sovereign.llm.adapter import ToolCall
            from tests.unit.test_tool_effect_invocation import governed

            @governed
            async def invoke_parallel(agent, user_input, *, invocation_id=None):
                async with agent._get_privacy_transition_lock().policy_span():
                    await agent._execute_tool_batch_at_stop_boundary([
                        ToolCall(id='parent', name=binding.name, arguments={}),
                        ToolCall(id='observer', name=observer_binding.name, arguments={}),
                    ], {}, {binding.name, observer_binding.name}, [], 0,
                        user_input, tool_results=[], session_id='owned-parallel-parent')
                return {'success': True}

            return await invoke_parallel(owner, 'Native parallel subtree exclusion',
                invocation_id='owned-parallel-parent-intent')
        if lane == 'nested-normal-turn':
            async with owner._get_privacy_transition_lock():
                return await relay.invoke({}, session_id='owned-nested-sibling-session',
                    operation_id='owned-nested-sibling-parent')
        if lane == 'normal-turn':
            async with owner._get_privacy_transition_lock():
                return await binding.invoke({}, session_id='owned-sibling-session', operation_id='owned-sibling-intent')
        return await binding.invoke({}, session_id='owned-sibling-session', operation_id='owned-sibling-intent')

    calling = asyncio.create_task(call())
    try:
        milestone = asyncio.create_task(first_started.wait())
        try:
            await asyncio.wait({milestone, calling}, timeout=2.0,
                return_when=asyncio.FIRST_COMPLETED)
        finally:
            milestone.cancel()
            await asyncio.gather(milestone, return_exceptions=True)
        lock = owner._get_privacy_transition_lock()
        diagnostic = {
            'writer': lock._writer.get_name() if lock._writer is not None else None,
            'closing': lock._token.closing if lock._token is not None else None,
            'parked': [(frame.task.get_name(), [getattr(target, 'get_name', lambda: type(target).__name__)() for target in frame.targets]) for frame in lock._parked],
            'tasks': [(task.get_name(), task.done(), [(frame.f_code.co_name, frame.f_lineno) for frame in task.get_stack()]) for task in lock._admissions],
        }
        assert first_started.is_set(), f'Native child has not entered: {diagnostic}'
        assert all_accepted.is_set(), 'Native background acceptance waited for its sibling effect'
        assert len(entered) == 1 and entered[0] in {'first', 'second'}, 'Native sibling effects bypassed their shared privacy mutex'
        if lane != 'external':
            admitted = sum(worker._kestrel_native_privacy_state['entered'] for worker in manager._execution_tasks)
            assert admitted == (2 if lane == 'async-parent' else 1), 'Queued native worker was classified as already entered'
        release.set()
        assert (await calling)['success'] is True
        if lane == 'parallel-parent':
            assert sorted(witness.read_text().splitlines()) == ['first', 'observer', 'second']
        else:
            assert sorted(witness.read_text().splitlines()) == ['first', 'second']
        for task in accepted:
            persisted = await manager.task_store.get_for_recipient(task.id, manager.did)
            assert persisted.status.state == TaskState.TASK_STATE_COMPLETED
        states = await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims')
        assert states == [('RETURNED_OK',)] * ({'nested-normal-turn': 6, 'async-parent': 8, 'parallel-parent': 6}.get(lane, 5))
    finally:
        release.set()
        workers = tuple(manager._execution_tasks)
        if not first_started.is_set():
            for worker in workers:
                worker.cancel()
        await asyncio.gather(calling, *workers, return_exceptions=True)
        await manager.close()


async def test_nested_foreign_parallel_effects_exclude_siblings_inside_the_native_mutex(bound_operation):
    from kestrel_sovereign.llm.adapter import ToolCall
    from tests.unit.test_tool_effect_invocation import governed

    owner, _binding, witness = bound_operation
    entered = []
    first_started, release = asyncio.Event(), asyncio.Event()

    class CriticalTool:
        name = 'owned.parallel_critical'
        schema = ToolSchema(name, 'Concurrent tools share the native privacy mutex',
            ToolCategory.UTILITY, parameters=[ToolParameter('marker', 'string', 'Marker', required=True)],
            is_concurrency_safe=True)

        async def execute(self, *, marker):
            async with owner._get_privacy_transition_lock():
                entered.append(marker)
                if marker == 'first':
                    first_started.set()
                    await release.wait()
                with witness.open('a') as stream:
                    stream.write(marker + '\n')
            return {'success': True}

    mounted = []
    owner.register_dynamic_tools('owned-parallel-critical', [CriticalTool()],
        on_registered=lambda tool, binding: mounted.append(binding))
    binding, = mounted

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        async def batch():
            await agent._execute_tool_batch_at_stop_boundary(
                [ToolCall(id=str(index), name=binding.name, arguments={'marker': marker})
                 for index, marker in enumerate(('first', 'second'))],
                {}, {binding.name}, [], 0, user_input, tool_results=[],
                session_id='owned-parallel-critical-session')
            return {'success': True}
        return await _foreign_effect(agent, batch)

    calling = asyncio.create_task(invoke(owner, 'Original parallel critical sections',
        invocation_id='owned-parallel-critical-intent'))
    try:
        await asyncio.wait({calling}, timeout=0.2)
        assert first_started.is_set(), 'Nested parallel children deadlocked behind their parent frame'
        assert entered == ['first'], 'Parallel siblings bypassed their original privacy mutex'
        release.set()
        assert (await calling)['success'] is True
        assert witness.read_text() == 'first\nsecond\n'
    finally:
        release.set()
        await _retire_foreign_scenario(calling, (binding.name, 'owned.foreign_relay',
            'orchestrator-parallel-tool:'))


@pytest_asyncio.fixture
async def response_loss_dispatch(bound_operation, monkeypatch):
    """Retain one real file effect and every entry point's original intention."""
    from kestrel_sovereign.llm.adapter import ToolCall
    from tests.unit.test_tool_effect_invocation import governed

    owner, binding, witness = bound_operation
    original = binding.tool.execute
    failure = TimeoutError('owned file completed before original response loss')

    async def lose_response(**arguments):
        await original(**arguments)
        raise failure

    monkeypatch.setattr(binding.tool, 'execute', lose_response)

    @governed
    async def chat(agent, user_input, *, invocation_id=None):
        return await agent._dispatch_tool_call(
            ToolCall(id='original-native-call', name=binding.name, arguments={'marker': 'once'}),
            {}, {binding.name}, [], 0, user_input, session_id='unknown-native-audit')

    async def invoke(lane):
        intent = 'unknown-native-audit-' + lane
        if lane == 'chat':
            return await chat(owner, 'one original intention', invocation_id=intent)
        if lane == 'named':
            return await owner.execute_named_tool(binding.name, {'marker': 'once'},
                session_id='unknown-native-audit', operation_id=intent)
        return await binding.invoke({'marker': 'once'}, session_id='unknown-native-audit', operation_id=intent)

    return owner, binding, witness, failure, original, invoke


@pytest.mark.parametrize('lane', ['chat', 'named', 'bound'])
async def test_unknown_effect_keeps_one_required_dispatch_row_without_replay(response_loss_dispatch, lane):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import execution_evidence_carriers

    owner, _binding, witness, failure, _original, invoke = response_loss_dispatch
    with caller_context_scope(CallerContext.sovereign()):
        with pytest.raises(ResourceOutcomeUnknown) as caught:
            await invoke(lane)
        assert any(error is failure for error in execution_evidence_carriers(caught.value))
        assert witness.read_text().splitlines() == ['once']
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT state FROM tool_effect_claims') == ('UNCONFIRMED',)
        rows = await owner.tool_effect_store.database.fetchall(
            'SELECT result_status, error_class FROM a2a_tool_dispatches')
        assert rows == [('error', 'ResourceOutcomeUnknown')], 'original unknown effect lost its required dispatch audit row'
        with pytest.raises(ToolEffectReconciliationRequired):
            await invoke(lane)
        assert witness.read_text().splitlines() == ['once']
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)


@pytest.mark.parametrize('lane', ['chat', 'named', 'bound'])
@pytest.mark.parametrize('failure_mode,delivery', [('refused', 'ordinary'), ('lost_ack', 'ordinary'), ('lost_ack', 'deferred')])
async def test_dispatch_audit_failure_preserves_original_receipts_without_another_write(
    response_loss_dispatch, monkeypatch, lane, failure_mode, delivery,
):
    from kestrel_sdk.resources import observed_outcome_unknown
    from kestrel_sdk.storage.database import QueryError
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError, execution_commit_outcome, execution_evidence_carriers,
    )

    owner, binding, witness, body_error, original, invoke = response_loss_dispatch
    database = owner.tool_effect_store.database
    if failure_mode == 'refused':
        await database.execute("CREATE TRIGGER refuse_dispatch_audit BEFORE INSERT ON a2a_tool_dispatches BEGIN SELECT RAISE(ABORT, 'owned audit insertion refusal'); END")
    else:
        if delivery == 'deferred':
            from kestrel_sdk.tools import DeferredToolResult

            async def completed_deferred(**arguments):
                result = await original(**arguments)
                return DeferredToolResult(result, publish=lambda: publications.append('published') or [])

            monkeypatch.setattr(binding.tool, 'execute', completed_deferred)
        else:
            monkeypatch.setattr(binding.tool, 'execute', original)
    publications = []
    write = owner.observability_store.log_tool_dispatch
    audit_errors = []

    async def observe_original_write(*args, **kwargs):
        try:
            result = await write(*args, **kwargs)
        except BaseException as error:
            audit_errors.append(error)
            raise
        error = QueryError('original audit commit acknowledgement lost')
        error.__cause__ = ExecutionCommitOutcomeError('unknown')
        audit_errors.append(error)
        raise error

    monkeypatch.setattr(owner.observability_store, 'log_tool_dispatch', observe_original_write)
    with caller_context_scope(CallerContext.sovereign()):
        with pytest.raises(BaseException) as caught:
            await invoke(lane)
        carriers = execution_evidence_carriers(caught.value)
        assert audit_errors and any(error is audit_errors[0] for error in carriers), 'dispatch discarded its original audit write failure'
        assert len(audit_errors) == 1, 'dispatch retried an original audit write whose completion was not confirmed'
        assert not any(type(error) is RuntimeError and 'audit is still pending' in str(error) for error in carriers), 'settled audit failure lost its retained completion error'
        expected_rows = 0 if failure_mode == 'refused' else 1
        assert await database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (expected_rows,), 'one original audit write produced another physical row'
        assert witness.read_text().splitlines() == ['once']
        if failure_mode == 'refused':
            assert any(error is body_error for error in carriers)
            assert observed_outcome_unknown(caught.value)
        else:
            assert execution_commit_outcome(caught.value) == 'unknown'
        with pytest.raises(ToolEffectReconciliationRequired):
            await invoke(lane)
        assert len(audit_errors) == 1
        assert witness.read_text().splitlines() == ['once']
        assert publications == [], 'unconfirmed native audit published a private UI factory'


@pytest_asyncio.fixture(params=['core', 'sdk'])
async def custom_feature_dispatch(bound_operation, request):
    kind = request.param
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.features.base import Feature
    from kestrel_sdk.features.base import Feature as SDKFeature
    from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
    from kestrel_sovereign.llm.adapter import ToolCall
    from tests.unit.test_tool_effect_invocation import governed
    from kestrel_sovereign.execution_custody import execution_evidence_carriers

    owner, _binding, witness = bound_operation
    original = TimeoutError('custom feature lost its original reply after a file effect')

    class CustomFeature(Feature if kind == "core" else SDKFeature):
        tool_description = 'Owned custom feature effect'

        async def initialize(self):
            pass

        async def execute_as_subagent(self, **kwargs):
            with witness.open('a') as stream:
                stream.write('once\n')
            raise original

    feature = ensure_subagent_dispatch(CustomFeature)(owner)
    owner.features[feature.name] = feature

    @governed
    async def chat(agent, user_input, *, invocation_id=None):
        return await agent._dispatch_tool_call(
            ToolCall(id='custom-feature-call', name=feature.tool_name, arguments={'task': 'one effect'}),
            {feature.tool_name: feature}, {feature.tool_name}, [], 0, user_input,
            session_id='owned-custom-feature')

    async def invoke(lane):
        if lane == 'named':
            return await owner.execute_named_tool(feature.tool_name, {'task': 'one effect'},
                session_id='owned-custom-feature', operation_id='custom-feature-original-intent')
        return await chat(owner, 'one original custom feature input',
            invocation_id='custom-feature-original-intent')

    return owner, feature, witness, original, invoke


@pytest.mark.parametrize('lane', ['chat', 'named'])
async def test_custom_feature_dispatch_cannot_repeat_an_unconfirmed_file_effect(custom_feature_dispatch, lane):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import execution_evidence_carriers

    owner, feature, witness, original, invoke = custom_feature_dispatch
    outcomes = []
    for delivery in range(2):
        try:
            outcomes.append(await invoke(lane))
        except BaseException as error:
            outcomes.append(error)
    assert witness.read_text().splitlines() == ['once'], 'Original custom feature uncertainty allowed a repeated file effect'
    assert isinstance(outcomes[0], ResourceOutcomeUnknown), 'Custom feature failure was flattened into a business reply'
    assert any(error is original for error in execution_evidence_carriers(outcomes[0]))
    assert isinstance(outcomes[1], ToolEffectReconciliationRequired)
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state FROM tool_effect_claims') == ('UNCONFIRMED',)


@pytest.mark.parametrize('lane', ['chat', 'named'])
@pytest.mark.parametrize('change', ['feature', 'method', 'disabled', 'foreign-owner', 'registry'])
async def test_custom_feature_revalidates_its_original_registration_after_permission(
    custom_feature_dispatch, native_tool_effect_agent, lane, change,
):
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
    from kestrel_sdk.resources import ResourceConflict

    owner, feature, witness, original, invoke = custom_feature_dispatch
    replacement = type(feature)(owner)
    foreign = await native_tool_effect_agent(did=owner.did) if change == 'foreign-owner' else None

    class ReplaceAfterPermission(Hook):
        def __init__(self):
            super().__init__('original-custom-feature-approval', [HookEvent.PRE_SUBAGENT_CALL])
            self.calls = 0

        async def execute(self, value):
            self.calls += 1
            if change == 'feature':
                owner.features[feature.name] = replacement
            elif change == 'method':
                feature.execute_as_subagent = replacement.execute_as_subagent
            elif change == 'disabled':
                feature.enabled = False
            elif change == 'registry':
                assert owner.register_dynamic_tools('owned-permission-generation',
                    list(owner._direct_tools.values())[:1]) == 1
            else:
                feature.agent = foreign
            return HookOutput.allow()

    policy = ReplaceAfterPermission()
    owner.hooks_manager.register(policy)
    if lane == 'named':
        with pytest.raises(ResourceConflict):
            await invoke(lane)
    else:
        result = await invoke(lane)
        assert result['success'] is False
    assert policy.calls == 1, 'The permission hook never reached the selected custom feature'
    assert not witness.exists(), 'A replaced or foreign feature entered its physical effect'
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state FROM tool_effect_claims') == ('NOT_EXECUTED',)


@pytest.mark.parametrize('lane', ['chat', 'named', 'bound'])
async def test_post_approval_argument_owner_cannot_change_the_admitted_effect(bound_operation, monkeypatch, lane):
    import json
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
    from kestrel_sovereign.llm.adapter import ToolCall
    from tests.unit.test_tool_effect_invocation import governed

    owner, binding, witness = bound_operation
    approved = {'marker': 'approved'}

    class Rewrite(Hook):
        def __init__(self):
            super().__init__('owned-argument-rewrite', [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            return HookOutput.modify(approved)

    owner.hooks_manager.register(Rewrite())
    claim = owner.tool_effect_store.claim

    async def change_after_actual_admission(*args, **kwargs):
        receipt = await claim(*args, **kwargs)
        approved['marker'] = 'unapproved'
        return receipt

    monkeypatch.setattr(owner.tool_effect_store, 'claim', change_after_actual_admission)

    @governed
    async def chat(agent, user_input, *, invocation_id=None):
        return await agent._dispatch_tool_call(
            ToolCall(id='owned-argument-call', name=binding.name, arguments={'marker': 'requested'}),
            {}, {binding.name}, [], 0, user_input, session_id='owned-argument-session')

    if lane == 'chat':
        result = await chat(owner, 'one approved file effect', invocation_id='owned-argument-intent')
    elif lane == 'named':
        result = await owner.execute_named_tool(binding.name, {'marker': 'requested'},
            session_id='owned-argument-session', operation_id='owned-argument-intent')
    else:
        result = await binding.invoke({'marker': 'requested'},
            session_id='owned-argument-session', operation_id='owned-argument-intent')
    assert result['success'] is True
    assert approved == {'marker': 'unapproved'}, 'The external argument owner never changed its data'
    assert witness.read_text().splitlines() == ['approved'], 'An argument changed after approval and durable effect admission'
    rows = await owner.tool_effect_store.database.fetchall('SELECT args_redacted FROM a2a_tool_dispatches')
    assert [json.loads(row[0]) for row in rows] == [{'marker': 'approved'}]
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state FROM tool_effect_claims') == ('RETURNED_OK',)


@pytest.mark.parametrize('lane', ['chat', 'named', 'bound'])
async def test_original_required_audit_joins_before_repeated_cancellation_retires_its_owner(
    response_loss_dispatch, monkeypatch, lane,
):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import execution_evidence_carriers

    owner, binding, witness, _, original, invoke = response_loss_dispatch
    monkeypatch.setattr(binding.tool, 'execute', original)
    entered, release = asyncio.Event(), asyncio.Event()
    writes = []
    write = owner.observability_store.log_tool_dispatch

    async def pending_original_write(entry, **kwargs):
        writes.append(entry.dispatch_id)
        entered.set()
        await release.wait()
        return await write(entry, **kwargs)

    monkeypatch.setattr(owner.observability_store, 'log_tool_dispatch', pending_original_write)
    calling = None
    with caller_context_scope(CallerContext.sovereign()):
        try:
            calling = asyncio.create_task(invoke(lane))
            await asyncio.wait_for(entered.wait(), 5)
            assert witness.read_text().splitlines() == ['once']
            for _ in range(2):
                calling.cancel()
                await asyncio.sleep(0.02)
                assert not calling.done(), 'Original required audit was abandoned before its physical SQL acknowledgement'
            release.set()
            with pytest.raises((asyncio.CancelledError, ResourceOutcomeUnknown)) as caught:
                await calling
            assert any(isinstance(error, asyncio.CancelledError)
                for error in execution_evidence_carriers(caught.value))
            assert len(writes) == 1, 'Cancellation issued another physical audit attempt'
            assert await owner.tool_effect_store.database.fetchone(
                'SELECT result_status FROM a2a_tool_dispatches') == ('success',)
            with pytest.raises(ToolEffectReconciliationRequired):
                await invoke(lane)
            assert len(writes) == 1 and witness.read_text().splitlines() == ['once']
        finally:
            release.set()
            if calling is not None:
                await asyncio.gather(calling, return_exceptions=True)


@pytest.mark.parametrize('route', ['named', 'bound', 'nested-invocation', 'native-admit'])
@pytest.mark.parametrize('uncertain', [False, True], ids=['known-refusal', 'uncertain'])
async def test_explicit_nested_intent_cannot_replace_original_uncertainty_fence(bound_operation, route, uncertain):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.tools.effect_journal import bind_tool_effect_turn, current_tool_effect_turn, execute_tool_effect
    from tests.unit.test_tool_effect_execution import admit
    from tests.unit.test_tool_effect_invocation import governed
    host, binding, witness = bound_operation
    original, seen = ResourceOutcomeUnknown('original input effect reply unavailable'), {}

    async def first():
        with witness.open('a') as stream:
            stream.write('original\n')
        raise original

    async def later():
        with witness.open('a') as stream:
            stream.write('nested\n')
        return 'acknowledged nested effect'

    @governed
    async def nested(agent, user_input, *, invocation_id=None):
        return await execute_tool_effect(agent, registration='nested-input-effect', arguments={}, execute=later)

    async def invoke_nested():
        if route == 'named':
            return await host.execute_named_tool(binding.name, {'marker': 'nested'},
                session_id='original-session', operation_id='unused-nested-named-intent')
        if route == 'bound':
            return await binding.invoke({'marker': 'nested'}, session_id='original-session',
                operation_id='unused-nested-bound-intent')
        if route == 'nested-invocation':
            return await nested(host, 'nested input', invocation_id='unused-nested-input')
        turn = await admit(host, key='unused-native-nested-intent')
        try:
            with bind_tool_effect_turn((turn, None)):
                return await execute_tool_effect(host, registration='nested-direct-effect', arguments={}, execute=later)
        finally:
            await turn.close()

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        try:
            await execute_tool_effect(agent, registration='original-input-effect',
                arguments=({} if uncertain else {'invalid_json': object()}), execute=first)
        except BaseException as error:
            seen['first'] = error
        assert bool(current_tool_effect_turn(agent).unconfirmed) is uncertain
        try:
            seen['nested_result'] = await invoke_nested()
        except BaseException as error:
            seen['nested_error'] = error

    try:
        await invoke(host, 'one original input', invocation_id='original-before-explicit-intent')
    except BaseException as error:
        seen['root_error'] = error
    lines = witness.read_text().splitlines() if witness.exists() else []
    if uncertain:
        assert lines == ['original'], 'a new explicit operation identity bypassed the original live uncertainty fence'
        assert 'nested_error' in seen and 'root_error' in seen
        assert any(node is original for node in execution_evidence_carriers(seen['root_error']))
        assert await host.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (1,), 'a fenced parent admitted a replacement durable operation'
        assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('UNCONFIRMED',)]
    else:
        assert lines == ['nested'] and 'nested_error' not in seen and 'root_error' not in seen


@pytest.mark.parametrize('uncertain', [False, True], ids=['known-refusal', 'uncertain'])
async def test_nested_admitted_input_retains_one_original_live_uncertainty_fence(bound_operation, uncertain):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect
    from tests.unit.test_tool_effect_invocation import governed
    host, _binding, witness = bound_operation
    original, seen = ResourceOutcomeUnknown('nested original effect reply unavailable'), {}

    async def first():
        witness.write_text('original\n')
        raise original

    async def later():
        with witness.open('a') as stream:
            stream.write('later\n')
        return 'acknowledged'

    @governed
    async def nested(agent, user_input, *, invocation_id=None):
        await execute_tool_effect(agent, registration='original-nested-input',
            arguments=({} if uncertain else {'invalid_json': object()}), execute=first)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        seen['parent'] = current_tool_effect_turn(agent)
        try:
            await nested(agent, 'original nested input', invocation_id='original-nested-input')
        except BaseException as error:
            seen['nested_error'] = error
        try:
            await execute_tool_effect(agent, registration='later-parent-effect', arguments={}, execute=later)
        except BaseException as error:
            seen['later_error'] = error

    try:
        await invoke(host, 'original owning input', invocation_id='original-parent-input')
    except BaseException as error:
        seen['root_error'] = error
    assert 'nested_error' in seen
    if uncertain:
        assert witness.read_text().splitlines() == ['original'], 'nested uncertainty escaped its original owning input fence'
        assert 'later_error' in seen and 'root_error' in seen
        assert any(node is original for node in execution_evidence_carriers(seen['root_error']))
        assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('UNCONFIRMED',)]
    else:
        assert witness.read_text().splitlines() == ['later']
        assert 'later_error' not in seen and 'root_error' not in seen


async def test_fresh_intent_requires_original_parent_native_membership(bound_operation):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from tests.unit.test_tool_effect_execution import admit
    from tests.unit.test_tool_effect_invocation import governed
    host, _binding, witness = bound_operation

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        # A raw Task inherits the business context, but has no original native
        # admission. A new intent must refuse before creating another SQL row.
        with pytest.raises(ExecutionAuthorityError, match='original native privacy lifetime'):
            await asyncio.create_task(admit(agent, key='unadmitted-copied-parent'))
        assert await host.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (1,)

    await invoke(host, 'original admitted input', invocation_id='original-parent-native-membership')
    assert not witness.exists()


@pytest.mark.parametrize('uncertain', [False, True], ids=['known-refusal', 'uncertain'])
async def test_nested_claim_free_terminal_publishes_original_input_uncertainty(bound_operation, uncertain):
    from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect
    from tests.unit.test_tool_effect_invocation import governed
    host, _binding, witness = bound_operation
    original = (ResourceOutcomeUnknown if uncertain else ResourceConflict)('original claim-free nested terminal')
    seen = {}

    async def nested(agent):
        from kestrel_sovereign.tools.effect_invocation import ToolEffectOperation, journal_tool_operation
        operation = ToolEffectOperation('claim-free-native-operation', 'claim-free-nested-input',
            'original-host', 'original-pre-body-input', {}, 'original-session')
        async with journal_tool_operation(agent, operation):
            child = current_tool_effect_turn(agent)
            assert child.native_root is seen['parent'].native_root is asyncio.current_task()
            assert child.unconfirmed is seen['parent'].unconfirmed
            assert await host.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
            raise original

    async def later():
        witness.write_text('later parent effect')
        return 'acknowledged'

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        seen['parent'] = current_tool_effect_turn(agent)
        try:
            await nested(agent)
        except BaseException as error:
            seen['nested_error'] = error
        assert await host.tool_effect_store.database.fetchall(
            'SELECT state,latched FROM tool_effect_operations ORDER BY rowid') == [('OPEN', 0), ('CLOSED', int(uncertain))]
        try:
            await execute_tool_effect(agent, registration='later-parent-after-claim-free', arguments={}, execute=later)
        except BaseException as error:
            seen['later_error'] = error

    try:
        await invoke(host, 'original parent input', invocation_id='parent-before-claim-free-terminal')
    except BaseException as error:
        seen['root_error'] = error
    assert 'nested_error' in seen and any(node is original for node in execution_evidence_carriers(seen['nested_error']))
    if uncertain:
        assert not witness.exists(), 'claim-free nested terminal escaped the original input live fence'
        assert len(seen['parent'].unconfirmed) == 1, 'the same original terminal was published more than once'
        assert 'later_error' in seen and 'root_error' in seen
        assert any(node is original for node in execution_evidence_carriers(seen['root_error']))
        assert await host.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
        assert await host.tool_effect_store.database.fetchall(
            'SELECT state,latched FROM tool_effect_operations') == [('CLOSED', 1), ('CLOSED', 1)]
    else:
        assert witness.read_text() == 'later parent effect'
        assert 'later_error' not in seen and 'root_error' not in seen
        assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]


@pytest.mark.parametrize('after', [False, True], ids=['before-commit', 'after-commit'])
async def test_nested_close_commit_uncertainty_fences_original_parent(bound_operation, monkeypatch, after):
    import aiosqlite
    from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_evidence_carriers
    from kestrel_sovereign.tools.effect_invocation import ToolEffectOperation, journal_tool_operation
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect
    from tests.unit.test_native_completion_observations import sqlite_commit_ack_loss
    from tests.unit.test_tool_effect_invocation import governed
    host, _binding, witness = bound_operation
    armed = []
    original, lost, lose_ack = sqlite_commit_ack_loss(host.tool_effect_store.database.backend._connection,
        after=after, ready=lambda: bool(armed))
    seen = {}

    async def later():
        witness.write_text('later effect after nested close commit uncertainty')
        return 'acknowledged'

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        operation = ToolEffectOperation('close-commit-operation', 'original-nested-close',
            'original-host', 'original-input', {}, 'original-session')
        try:
            async with journal_tool_operation(agent, operation):
                # Arm only after real admission. The same original native
                # callback is captured before either journal is admitted; no
                # extra SQL is issued inside the sealed terminal program.
                armed.append(True)
        except BaseException as error:
            seen['nested_error'] = error
            seen['published'] = tuple(current_tool_effect_turn(agent).unconfirmed)
        try:
            await execute_tool_effect(agent, registration='later-after-nested-close', arguments={}, execute=later)
        except BaseException as error:
            seen['later_error'] = error

    monkeypatch.setattr(aiosqlite.Connection, 'commit', lose_ack)
    try:
        await invoke(host, 'original parent before close commit', invocation_id='parent-before-close-commit')
    except BaseException as error:
        seen['root_error'] = error
    assert not witness.exists(), 'nested terminal COMMIT uncertainty escaped the original parent live fence'
    assert lost == [True], [(type(error).__name__, str(error)) for error in execution_evidence_carriers(seen.get('nested_error'))]
    assert all(name in seen for name in ['nested_error', 'later_error', 'root_error'])
    assert seen['published'], 'nested terminal writer did not publish uncertainty before consumer handoff'
    assert execution_commit_outcome(seen['nested_error']) == 'unknown'
    assert any(node is original for node in execution_evidence_carriers(seen['root_error']))
    assert await host.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)


@pytest.mark.parametrize('uncertain', [False, True], ids=['known-refusal', 'uncertain'])
async def test_nested_terminal_fences_parent_before_native_retirement(bound_operation, monkeypatch, uncertain):
    from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.tools.effect_invocation import ToolEffectOperation, journal_tool_operation
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect
    from tests.unit.test_native_completion_observations import write_required_denial_audit
    from tests.unit.test_tool_effect_invocation import governed
    host, _binding, witness = bound_operation
    original = (ResourceOutcomeUnknown if uncertain else ResourceConflict)('original nested terminal before retirement')
    entered, release, joining = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen, lock = {}, host._get_privacy_transition_lock()
    join = lock.join_native_scope

    async def observe_join(scope, pending=None, *, root=None):
        if root is seen.get('child_root') and pending is original:
            joining.set()
        return await join(scope, pending, root=root)

    monkeypatch.setattr(lock, 'join_native_scope', observe_join)

    async def audit():
        entered.set()
        await release.wait()
        await write_required_denial_audit(host)

    async def later():
        witness.write_text('parent effect while original nested work is retiring')
        return 'acknowledged'

    async def nested():
        operation = ToolEffectOperation('nested-draining-operation', 'original-nested-drain',
            'original-host', 'original-input', {}, 'original-session')
        async with journal_tool_operation(host, operation):
            child = current_tool_effect_turn(host)
            seen['child_root'] = child.native_root
            lock.start_owned_native_work(audit, acquire_span=False)
            await entered.wait()
            raise original

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        child = create_owned_task(nested(), retain_originals=True)
        try:
            await asyncio.wait_for(joining.wait(), 5)
            assert not child.done() and seen['child_root'] is not asyncio.current_task()
            try:
                await execute_tool_effect(agent, registration='parent-during-original-retirement', arguments={}, execute=later)
            except BaseException as error:
                seen['later_error'] = error
        finally:
            release.set()
            seen['nested_outcome'] = await await_owned_task(child)

    try:
        await invoke(host, 'original parent during native retirement', invocation_id='parent-before-nested-native-retirement')
    except BaseException as error:
        seen['root_error'] = error
    assert any(node is original for node in execution_evidence_carriers(seen['nested_outcome'].error))
    if uncertain:
        assert not witness.exists(), 'a known nested terminal admitted a parent effect while native retirement was pending'
        assert 'later_error' in seen and 'root_error' in seen
    else:
        assert witness.read_text() == 'parent effect while original nested work is retiring'
        assert 'later_error' not in seen and 'root_error' not in seen
    assert await host.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)


async def test_copied_request_generation_cannot_detach_original_journal(bound_operation):
    from contextvars import copy_context
    from kestrel_sovereign.agent.invocation import invocation_scope
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.tools.effect_invocation import journal_async_invocation
    from kestrel_sovereign.tools.effect_journal import execute_tool_effect
    from tests.unit.test_tool_effect_invocation import governed
    host, _binding, witness = bound_operation
    copied_id = 'copied-generation-without-original-input-task'

    async def effect():
        witness.write_text('copied generation forged a fresh journal')
        return 'unadmitted'

    @journal_async_invocation
    async def unregistered(agent, user_input, *, invocation_id=None):
        return await execute_tool_effect(agent, registration='forged-independent-input', arguments={}, execute=effect)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        copied = copy_context()
        # The genuine registrar is THIS Task. The raw child inherits a real
        # active generation but owns neither its registration nor an admitted
        # operation in that generation's canonical native inventory.
        generation = copied.run(agent.register_active_request, copied_id, nested=False)

        async def raw_child():
            with invocation_scope(copied_id):
                return await unregistered(agent, 'copied input', invocation_id=copied_id)

        child = copied.run(asyncio.create_task, raw_child())
        try:
            with pytest.raises(ExecutionAuthorityError, match='original native privacy lifetime'):
                await child
        finally:
            agent._cleanup_cancelled_request(copied_id, generation=generation)
        assert not witness.exists()
        assert await host.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (1,)

    await invoke(host, 'original parent input', invocation_id='original-before-copied-generation')


@pytest.mark.parametrize('changed', [False, True], ids=['same-input', 'changed-input'])
async def test_independent_copied_input_key_cannot_reuse_live_parent_journal(bound_operation, changed):
    from tests.unit.test_tool_effect_invocation import governed
    host, _binding, witness = bound_operation
    entered, original_owner = [], []
    same_key = 'original-live-input-key'

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        if original_owner:
            entered.append(user_input)
            return 'unadmitted duplicate body'
        original_owner.append(asyncio.current_task())
        # A genuine new request operation must readmit this retry key, even
        # when it inherited the parent's matching id/generation ContextVars.
        child = asyncio.create_task(invoke(agent, 'changed input' if changed else user_input,
            invocation_id=same_key))
        with pytest.raises(ToolEffectOperationConflict if changed else ToolEffectReconciliationRequired):
            await child
        assert entered == [] and not witness.exists()
        assert await host.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (1,)

    await invoke(host, 'original input', invocation_id=same_key)


@pytest.mark.parametrize('after', [False, True], ids=['before-commit', 'after-commit'])
async def test_root_terminal_commit_loss_cannot_replay_after_database_restart(
    bound_operation, native_tool_effect_agent, monkeypatch, after,
):
    import aiosqlite
    from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_evidence_carriers
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.tools.effect_journal import execute_tool_effect
    from tests.unit.test_native_completion_observations import sqlite_commit_ack_loss
    from tests.unit.test_tool_effect_invocation import governed

    host, _binding, witness = bound_operation
    database = host.tool_effect_store.database
    path = (await database.fetchone('PRAGMA database_list'))[2]
    assert path, 'restart probe requires the actual file-backed database'
    armed, entered = [], []
    original, lost, lose_ack = sqlite_commit_ack_loss(
        database.backend._connection, after=after, ready=lambda: bool(armed))
    replaying = False

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        entered.append(invocation_id)
        if not replaying:
            armed.append(True)
            return 'claim-free original result'

        async def physical_effect():
            witness.write_text('effect after original terminal uncertainty')
            return 'acknowledged'

        return await execute_tool_effect(agent, registration='root-restart-effect',
            arguments={}, execute=physical_effect)

    monkeypatch.setattr(aiosqlite.Connection, 'commit', lose_ack)
    with pytest.raises(BaseException) as failed:
        await invoke(host, 'original claim-free input', invocation_id='root-terminal-restart')
    assert lost == [True]
    assert execution_commit_outcome(failed.value) == 'unknown'
    assert any(error is original for error in execution_evidence_carriers(failed.value))
    assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == (
        'CLOSED' if after else 'OPEN', 0)
    assert await database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    await host.tool_effect_store.close()
    await host.observability_store.close()
    await host.llm_service.close()
    await host._raw_storage.close()
    backend = SQLiteBackend(path)
    await backend.connect()
    restarted = await native_tool_effect_agent(did=host.did, connected_backend=backend)
    replaying = True
    refused = False
    try:
        await invoke(restarted, 'original claim-free input', invocation_id='root-terminal-restart')
    except ToolEffectReconciliationRequired:
        refused = True
    assert not witness.exists(), 'root terminal COMMIT uncertainty replayed a physical effect after restart'
    assert refused and entered == ['root-terminal-restart'], 'original input reentered cognition after restart'
    assert await invoke(restarted, 'original claim-free input', invocation_id='fresh-root-terminal-intent') == 'acknowledged'
    assert witness.read_text() == 'effect after original terminal uncertainty'
    assert await restarted.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (2,)
