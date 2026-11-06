"""Real Agent generations, native SQL admission and external file effects."""

import pytest

from kestrel_sovereign.agent.invocation import bind_async_invocation, request_provenance
from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.tool_effect_store import (
    ToolEffectOperationConflict, ToolEffectReconciliationRequired,
)
from kestrel_sovereign.tools.effect_invocation import journal_async_invocation
from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect
from tests.unit.test_tool_effect_execution import owner


pytestmark = pytest.mark.timeout(30)


def governed(function):
    return bind_async_invocation('invocation_id', track_request_lifecycle=True)(
        journal_async_invocation(function))


@pytest.mark.asyncio
async def test_root_admission_precedes_body_and_rejects_retry_before_another_effect(owner, tmp_path):
    entered = []
    witness = tmp_path / 'effects'

    @governed
    async def invoke(agent, user_input, *, session_id=None, caller=None,
                     invocation_id=None, invocation_provenance=None, user_passphrase=None):
        entered.append(user_input)
        turn = current_tool_effect_turn(agent)
        assert turn is not None and turn.generation == agent._request_generation_for_current_task(invocation_id)

        async def effect():
            with witness.open('a') as stream:
                stream.write(user_input + '\n')
            return 'acknowledged'

        return await execute_tool_effect(agent, registration='owned-file', arguments={}, execute=effect)

    provenance = request_provenance(source_kind='http_request', source_locator='/owned/invoke')
    first = dict(invocation_id='same-intent', invocation_provenance=provenance,
                 caller=CallerContext.sovereign(credential='owned-first-key'))
    assert await invoke(owner, 'one effect', **first) == 'acknowledged'
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('CLOSED',)
    rotated = dict(first, caller=CallerContext.sovereign(credential='owned-rotated-key'))
    with pytest.raises(ToolEffectReconciliationRequired):
        await invoke(owner, 'one effect', user_passphrase='different-secret', **rotated)
    with pytest.raises(ToolEffectOperationConflict):
        await invoke(owner, 'changed input', **first)
    assert entered == ['one effect']
    assert witness.read_text() == 'one effect\n'
    assert await invoke(owner, 'new intention', **dict(first, invocation_id='fresh-intent')) == 'acknowledged'
    assert witness.read_text() == 'one effect\nnew intention\n'


@pytest.mark.asyncio
async def test_nested_call_reuses_only_the_same_original_generation(owner):
    turns = []

    @governed
    async def invoke(agent, user_input, *, invocation_id=None, nested=False):
        turn = current_tool_effect_turn(agent)
        turns.append(turn)
        if not nested:
            await invoke(agent, user_input, invocation_id=invocation_id, nested=True)
        assert turn.live and not turn.closing
        return 'done'

    assert await invoke(owner, 'owned input', invocation_id='nested-intent') == 'done'
    assert len(turns) == 2 and turns[0] is turns[1]
    assert turns[0].live is False
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (1,)


@pytest.mark.asyncio
async def test_retry_namespace_keeps_distinct_authenticated_people_separate(owner):
    entered = []

    @governed
    async def invoke(agent, user_input, *, invocation_id=None, caller=None):
        entered.append(caller.identity)

        async def effect():
            return user_input

        return await execute_tool_effect(agent, registration='owned-effect', arguments={}, execute=effect)

    for identity in ['owned-alice', 'owned-bob']:
        assert await invoke(owner, 'owned input', invocation_id='same-client-key',
            caller=CallerContext.authenticated(identity)) == 'owned input'
    assert entered == ['owned-alice', 'owned-bob']
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (2,)


@pytest.mark.asyncio
async def test_nested_invocation_keeps_its_original_parent_effect_claim(owner, tmp_path):
    witness = tmp_path / 'nested-effect'

    @governed
    async def invoke(agent, user_input, *, invocation_id=None, nested=False):
        async def effect():
            if not nested:
                return await invoke(agent, user_input, invocation_id=invocation_id, nested=True)
            witness.write_text('one physical effect')
            return 'done'

        return await execute_tool_effect(agent, registration='child' if nested else 'parent',
            arguments={}, execute=effect)

    assert await invoke(owner, 'owned input', invocation_id='parent-intent') == 'done'
    claims = await owner.tool_effect_store.database.fetchall('SELECT claim_id, parent_claim_id FROM tool_effect_claims')
    roots = [row[0] for row in claims if row[1] is None]
    children = [row for row in claims if row[1] is not None]
    assert len(roots) == len(children) == 1, 'Nested invocation detached the original effect claim'
    assert children[0][1] == roots[0]
    assert witness.read_text() == 'one physical effect'
