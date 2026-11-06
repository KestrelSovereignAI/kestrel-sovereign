"""Actual named, subagent and native A2A leaves claim before file effects."""

import pytest
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_task
from kestrel_sdk.native_a2a.types import TaskState
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sdk.features.base import Feature as SDKFeature

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
