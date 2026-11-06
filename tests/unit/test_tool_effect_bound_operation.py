"""Actual bound-tool dispatch, native lifecycle, SQL claims and file effects."""

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
            with witness.open('a') as stream:
                stream.write(marker + '\n')
            return {'success': True, 'marker': marker}

    mounted = []
    owner.register_dynamic_tools('mcp:owned-operation', [FileTool()],
        on_registered=lambda tool, binding: mounted.append(binding))
    binding, = mounted
    yield owner, binding, witness
    await store.close()


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
