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
from tests.unit.test_tool_effect_execution import owner


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


async def test_parallel_batch_retains_original_privacy_ownership_for_its_native_leaves(bound_operation):
    from kestrel_sovereign.llm.adapter import ToolCall
    from tests.unit.test_tool_effect_invocation import governed

    owner, binding, witness = bound_operation
    messages, results = [], []

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        async with agent._get_privacy_transition_lock():
            await agent._execute_tool_batch_at_stop_boundary(
                [ToolCall(id=str(index), name=binding.name, arguments={'marker': marker})
                    for index, marker in enumerate(('first', 'second'))],
                {}, {binding.name}, messages, 0, user_input,
                tool_results=results, session_id='owned-parallel-session')

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
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',), ('RETURNED_OK',)]
        assert not owner._active_request_ids
    finally:
        # Retire the actual batch children first if the regression deadlocks;
        # their parent deliberately shields admitted effects from caller Stop.
        children = [task for task in asyncio.all_tasks()
            if task.get_name().startswith('orchestrator-parallel-tool:')]
        for task in children:
            task.cancel()
        await asyncio.gather(*children, return_exceptions=True)
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)


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

    calling = asyncio.create_task(invoke(owner, 'original nested intention', invocation_id='owned-parent-intent'))
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
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',), ('RETURNED_OK',)]
        assert not owner._active_request_ids
    finally:
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)


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


async def test_external_lifecycle_holds_privacy_from_native_admission_to_dispatch(bound_operation, monkeypatch):
    from kestrel_sovereign._async_ownership import await_owned_task

    owner, binding, witness = bound_operation
    acknowledged, resume, transitioned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    admit = owner.tool_effect_store.admit
    tool = owner._direct_tools[binding.name]
    execute = tool.execute

    async def hold_native_admission(*args, **kwargs):
        attempt = await admit(*args, **kwargs)
        assert not owner.tool_effect_store.database.owns_open_transaction
        acknowledged.set()
        await resume.wait()
        return attempt

    async def observe_original_execution(**kwargs):
        assert not transitioned.is_set(), 'Privacy transition overtook an already admitted lifecycle'
        return await execute(**kwargs)

    async def transition():
        async with owner.privacy_transition():
            transitioned.set()

    monkeypatch.setattr(owner.tool_effect_store, 'admit', hold_native_admission)
    monkeypatch.setattr(tool, 'execute', observe_original_execution)
    calling = asyncio.create_task(binding.invoke({'marker': 'ordered'},
        session_id='owned-lifecycle-span', operation_id='owned-lifecycle-span-intent'))
    changing = None
    try:
        await asyncio.wait_for(acknowledged.wait(), 2)
        changing = asyncio.create_task(transition())
        await asyncio.sleep(0.02)
        resume.set()
        outcome = await await_owned_task(calling)
        healthy = outcome.error is None
        assert healthy, 'External lifecycle lost its privacy span before native dispatch'
        await changing
        assert transitioned.is_set() and witness.read_text() == 'ordered\n'
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
