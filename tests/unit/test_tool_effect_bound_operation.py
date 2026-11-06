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
            parameters=[ToolParameter('marker', 'string', 'Owned marker', required=True)])

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


async def test_external_tool_and_real_privacy_transition_follow_one_lock_order(bound_operation, monkeypatch):
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

    calling = asyncio.create_task(binding.invoke({'marker': 'ordered'},
        session_id='owned-lock-order', operation_id='owned-lock-order-intent'))
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
