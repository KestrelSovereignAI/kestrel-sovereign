"""Canonical Core and adopted SDK subagent effects retain their original approval."""

import pytest
import pytest_asyncio
from kestrel_sdk.features.base import Feature as SDKFeature
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.agent.invocation import current_invocation_effect_checkpoint
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from tests.unit.test_tool_effect_bound_operation import bound_operation
from tests.unit.test_tool_effect_execution import owner
from tests.unit.test_tool_effect_invocation import governed

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest_asyncio.fixture(params=[Feature, SDKFeature], ids=['core', 'sdk'])
async def canonical_subagent(bound_operation, request):
    host, binding, witness = bound_operation
    failure = TimeoutError('canonical native leaf lost its original reply')
    checkpoints = []

    class NativeFeature(request.param):
        tool_description = 'One original file effect'

        async def initialize(self):
            pass

        @tool(name='native_leaf', description='Write one original marker')
        async def native_leaf(self, marker: str):
            checkpoints.append(current_invocation_effect_checkpoint())
            with witness.open('a') as stream:
                stream.write(marker + '\n')
            if self.lose_reply:
                raise failure
            return {'success': True, 'marker': marker}

    feature = ensure_subagent_dispatch(NativeFeature)(host)
    feature.lose_reply = False
    host.features[feature.name] = feature

    class Model:
        calls = 0

        async def generate(self, **kwargs):
            self.calls += 1
            return LLMResponse(content=None, tool_calls=[
                ToolCall(id='native-leaf-call', name='native_leaf', arguments={'marker': 'original'})])

        async def generate_with_messages(self, **kwargs):
            return LLMResponse(content='done', tool_calls=None)

    model = host.llm_service = Model()

    @governed
    async def chat(agent, user_input, *, invocation_id=None):
        return await agent._dispatch_tool_call(
            ToolCall(id='canonical-dispatch', name=feature.tool_name, arguments={'task': 'one effect'}),
            {feature.tool_name: feature}, {feature.tool_name}, [], 0, user_input,
            session_id='canonical-session')

    async def invoke(lane):
        if lane == 'named':
            return await host.execute_named_tool(feature.tool_name, {'task': 'one effect'},
                session_id='canonical-session', operation_id='canonical-original-intent')
        return await chat(host, 'one original canonical input', invocation_id='canonical-original-intent')

    return host, feature, witness, model, failure, checkpoints, invoke


@pytest.mark.parametrize('lane', ['chat', 'named'])
@pytest.mark.parametrize('change', ['feature', 'method', 'disabled', 'foreign-owner', 'registry'])
async def test_canonical_subagent_revalidates_original_permission_registration(
    canonical_subagent, native_tool_effect_agent, lane, change,
):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    replacement = type(feature)(host)
    foreign = await native_tool_effect_agent(did=host.did) if change == 'foreign-owner' else None

    class ReplaceAfterPermission(Hook):
        def __init__(self):
            super().__init__('canonical-original-approval', [HookEvent.PRE_SUBAGENT_CALL])
            self.calls = 0

        async def execute(self, value):
            self.calls += 1
            if change == 'feature':
                host.features[feature.name] = replacement
            elif change == 'method':
                feature.execute_as_subagent = replacement.execute_as_subagent
            elif change == 'disabled':
                feature.enabled = False
            elif change == 'foreign-owner':
                feature.agent = foreign
            else:
                host.register_dynamic_tools('canonical-generation-change', list(host._direct_tools.values())[:1])
            return HookOutput.allow()

    policy = ReplaceAfterPermission()
    host.hooks_manager.register(policy)
    if lane == 'named':
        with pytest.raises(ResourceConflict):
            await invoke(lane)
    else:
        result = await invoke(lane)
        assert result['success'] is False
    assert policy.calls == 1
    assert model.calls == 0, 'The stale canonical dispatcher entered its model loop'
    assert not witness.exists(), 'A stale canonical feature entered its physical effect'


@pytest.mark.parametrize('lane', ['chat', 'named'])
async def test_canonical_leaf_detaches_the_post_approval_binding(canonical_subagent, monkeypatch, lane):
    host, _, witness, _, _, _, invoke = canonical_subagent
    approved = {'marker': 'approved'}

    class Rewrite(Hook):
        def __init__(self):
            super().__init__('canonical-original-binding', [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            return HookOutput.modify(approved)

    host.hooks_manager.register(Rewrite())
    claim = host.tool_effect_store.claim

    async def mutate_after_claim(*args, **kwargs):
        receipt = await claim(*args, **kwargs)
        approved['marker'] = 'unapproved'
        return receipt

    monkeypatch.setattr(host.tool_effect_store, 'claim', mutate_after_claim)
    result = await invoke(lane)
    assert result['success'] is True
    assert approved['marker'] == 'unapproved'
    assert witness.read_text().splitlines() == ['approved'], 'The effect used a hook-owned mutable binding'
    rows = await host.tool_effect_store.database.fetchall(
        "SELECT args_redacted FROM a2a_tool_dispatches WHERE tool_name='native_leaf'")
    assert len(rows) == 1 and 'approved' in rows[0][0] and 'unapproved' not in rows[0][0]


@pytest.mark.parametrize('lane', ['chat', 'named'])
async def test_uncertain_canonical_leaf_retains_audit_checkpoint_and_replay_fence(canonical_subagent, lane):
    host, feature, witness, _, original, checkpoints, invoke = canonical_subagent
    feature.lose_reply = True
    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await invoke(lane)
    assert any(error is original for error in execution_evidence_carriers(caught.value))
    assert witness.read_text().splitlines() == ['original']
    assert len(checkpoints) == 1 and checkpoints[0].completed and checkpoints[0].checkpointed
    rows = await host.tool_effect_store.database.fetchall(
        "SELECT result_status, error_class, parent_dispatch_id FROM a2a_tool_dispatches WHERE tool_name='native_leaf'")
    assert len(rows) == 1 and rows[0][:2] == ('error', 'ResourceOutcomeUnknown')
    assert rows[0][2], 'The native leaf lost the dispatch that delegated its effect'
    with pytest.raises(ToolEffectReconciliationRequired):
        await invoke(lane)
    assert witness.read_text().splitlines() == ['original']
