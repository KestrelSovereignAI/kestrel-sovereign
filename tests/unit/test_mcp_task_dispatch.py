"""Live registered-handler proofs and bounded owner retention on native Core."""

import asyncio
import contextvars
import json

import pytest
import pytest_asyncio
from kestrel_sdk.resources import ResourceConflict
from kestrel_sdk.tools import DeferredToolResult
from kestrel_sdk.tools.result import ToolResult
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.llm.adapter import ToolCall
from kestrel_sovereign.tools.effect_journal import (
    ToolEffectJournalUnavailable, bind_tool_effect_turn, current_tool_effect_execution,
    execute_tool_effect,
)
from kestrel_sovereign.turn_scope import capture_turn_scope
from tests.unit.test_tool_effect_execution import owner as owner, admit
from tests.unit.test_tool_effect_invocation import governed

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


class ProbeTool:
    def __init__(self, name, handler):
        self.name, self.handler = name, handler
        self.schema = ToolSchema(name, 'Observe the actual registered handler', ToolCategory.UTILITY,
            parameters=[ToolParameter('value', 'string', 'Decided value', required=True)])

    async def execute(self, *, value):
        return await self.handler(value)


def mount(owner, name, handler, *, count=1):
    mounted = []
    owner.register_dynamic_tools(name, [ProbeTool(name + str(i), handler) for i in range(count)],
        on_registered=lambda tool, binding: mounted.append(binding))
    return mounted


@pytest_asyncio.fixture
async def dispatch(owner):
    store = ObservabilityStore(owner.tool_effect_store.database.backend, owns_backend=False)
    await store.initialize()
    owner.observability_store = store
    try:
        yield owner
    finally:
        await store.close()


async def invoke(owner, binding, lane):
    if lane == 'bound':
        return await binding.invoke({'value': 'approved'}, session_id='real-session', operation_id='bound-intent')
    if lane in {'named', 'named-capture'}:
        return await owner.execute_named_tool(binding.name, {'value': 'approved'},
            session_id='real-session', operation_id='named-intent', source='inline',
            _capture={} if lane == 'named-capture' else None)

    @governed
    async def model(agent, user_input, *, invocation_id=None):
        async with agent._get_privacy_transition_lock():
            await agent._execute_tool_batch_at_stop_boundary(
                [ToolCall(id='one-native-call', name=binding.name, arguments={'value': 'approved'})],
                {}, {binding.name}, [], 0, user_input, tool_results=[], session_id='real-session')
        return {'success': True}

    return await model(owner, 'A real native registered tool', invocation_id='model-intent')


def available(owner):
    try:
        current_tool_effect_execution(owner)
    except ToolEffectJournalUnavailable:
        return False
    return True


@pytest.mark.parametrize('lane', ['model', 'bound', 'named', 'named-capture'])
async def test_every_mounted_dispatch_producer_exposes_its_actual_live_selection(dispatch, lane):
    owner = dispatch
    seen = []

    async def handler(value):
        frame = current_tool_effect_execution(owner)
        seen.append(frame)
        assert frame.witness.started and not frame.witness.returned
        assert frame.witness.tool is binding.tool
        assert frame.witness.tool_name == binding.name and frame.witness.owner == binding.owner
        assert frame.witness.registration is binding.registration
        assert json.loads(frame.arguments_json) == {'value': value}
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT state FROM tool_effect_claims') == ('STARTED',)
        return {'success': True}

    binding, = mount(owner, 'owned.dispatch', handler)
    outcome, = await asyncio.gather(invoke(owner, binding, lane), return_exceptions=True)
    assert not isinstance(outcome, BaseException), 'Registered dispatch lost its live source proof'
    assert len(seen) == 1 and not seen[0].live
    with pytest.raises(ToolEffectJournalUnavailable):
        current_tool_effect_execution(owner)
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state FROM tool_effect_claims') == ('RETURNED_OK',)


async def test_a_copied_child_task_cannot_use_the_running_handlers_proof(dispatch):
    owner = dispatch
    attempts = []

    async def handler(value):
        current_tool_effect_execution(owner)

        async def child():
            attempts.append(available(owner))

        await asyncio.create_task(child())
        return {'success': True}

    binding, = mount(owner, 'owned.parent', handler)
    await invoke(owner, binding, 'bound')
    assert attempts == [False], 'A copied child used the original handler proof'


async def test_turn_rebinding_cannot_revive_a_parent_inside_a_witnessless_handler(owner):
    turn = await admit(owner)
    witness = object()
    visited = []

    async def outer():
        original = current_tool_effect_execution(owner)
        scope = capture_turn_scope(owner)

        async def child():
            with scope.bind():
                visited.append(available(owner))
            return {'success': True}

        await execute_tool_effect(owner, registration='witnessless-child', arguments={}, execute=child)
        assert current_tool_effect_execution(owner) is original
        return {'success': True}

    with bind_tool_effect_turn((turn, None)):
        await execute_tool_effect(owner, registration='parent', arguments={}, execute=outer, witness=witness)
    await turn.close()
    assert visited == [False], 'Turn rebinding restored a masked handler proof'


async def test_saved_context_on_the_same_task_cannot_revive_a_returned_handler(owner):
    turn = await admit(owner)
    contexts = []

    async def handler():
        contexts.append(contextvars.copy_context())
        return {'success': True}

    with bind_tool_effect_turn((turn, None)):
        await execute_tool_effect(owner, registration='original', arguments={}, execute=handler, witness=object())
        restored = contexts[0].run(available, owner)
    await turn.close()
    assert restored is False, 'Saved context revived a returned handler'


async def test_decided_arguments_are_sealed_before_native_claim_admission_await(owner, monkeypatch):
    turn = await admit(owner)
    arguments = {'value': 'original'}
    captured = []
    claim = owner.tool_effect_store.claim

    async def admitted(*args, **kwargs):
        result = await claim(*args, **kwargs)
        arguments['value'] = 'caller changed after admission'
        return result

    monkeypatch.setattr(owner.tool_effect_store, 'claim', admitted)

    async def handler():
        captured.append(json.loads(current_tool_effect_execution(owner).arguments_json))
        return {'success': True}

    with bind_tool_effect_turn((turn, None)):
        await execute_tool_effect(owner, registration='original', arguments=arguments,
            execute=handler, witness=object())
    await turn.close()
    assert captured == [{'value': 'original'}], 'Caller mutation replaced the admitted argument snapshot'


async def test_owner_retention_uses_the_actual_epoch_and_releases_back_to_lru(dispatch):
    owner = dispatch
    owner.MAX_DIRECT_TOOLS = 3

    async def handler(value):
        return {'success': True}

    first, = mount(owner, 'oldest', handler)
    hold = owner._acquire_task_owner_hold(first, max_holds=2, max_held_tools=1)
    second, = mount(owner, 'second', handler)
    mount(owner, 'pressure', handler, count=2)
    assert first.is_current and not second.is_current
    owner._release_task_owner_hold(hold)
    owner._release_task_owner_hold(hold)
    mount(owner, 'next-pressure', handler)
    assert not first.is_current and hold.released


async def test_hold_and_tool_headroom_limits_refuse_without_changing_the_registry(dispatch):
    owner = dispatch
    owner.MAX_DIRECT_TOOLS = 4

    async def handler(value):
        return {'success': True}

    first, = mount(owner, 'first', handler)
    second, = mount(owner, 'second', handler)
    hold = owner._acquire_task_owner_hold(first, max_holds=1, max_held_tools=1)
    with pytest.raises(ResourceConflict, match='hold capacity'):
        owner._acquire_task_owner_hold(second, max_holds=1, max_held_tools=2)
    with pytest.raises(ResourceConflict, match='registry headroom'):
        owner._acquire_task_owner_hold(second, max_holds=2, max_held_tools=1)
    assert first.is_current and second.is_current
    assert first.registration.task_holds == {hold} and not second.registration.task_holds
    owner._release_task_owner_hold(hold)


async def test_explicit_unmount_leaves_old_holds_inert_and_uncounted(dispatch):
    owner = dispatch
    owner.MAX_DIRECT_TOOLS = 4

    async def handler(value):
        return {'success': True}

    first, = mount(owner, 'replace', handler)
    original = owner._acquire_task_owner_hold(first, max_holds=1, max_held_tools=1)
    owner.unregister_dynamic_tools(first.owner)
    second, = mount(owner, 'replace', handler)
    with pytest.raises(ResourceConflict, match='expired'):
        owner._acquire_task_owner_hold(first, max_holds=1, max_held_tools=1)
    current = owner._acquire_task_owner_hold(second, max_holds=1, max_held_tools=1)
    assert not first.is_current and second.is_current
    assert original.registration is not current.registration
    owner._release_task_owner_hold(original)
    assert second.registration.task_holds == {current}
    owner._release_task_owner_hold(current)


@pytest.mark.parametrize('lane', ['model', 'bound', 'named', 'named-capture'])
async def test_completed_owner_lease_survives_real_deferred_publication_and_releases_after_audit(dispatch, lane):
    owner = dispatch
    owner.MAX_DIRECT_TOOLS = 3
    published = []
    leases = []

    async def handler(value):
        frame = current_tool_effect_execution(owner)
        lease = owner._acquire_task_owner_hold(binding, max_holds=1, max_held_tools=1)
        leases.append(lease)
        frame.witness.owner_lease = lease

        def publish():
            mount(owner, 'competing-registration', ordinary, count=3)
            assert binding.is_current, 'Final publication lost its original mounted owner'
            assert not lease.released, 'Owner lease ended before native deferred publication'
            published.append('original publication')
            return []

        return DeferredToolResult(ToolResult.ok('Actual deferred result').to_dict(), publish=publish)

    async def ordinary(value):
        return {'success': True}

    binding, = mount(owner, 'original', handler)
    outcome, = await asyncio.gather(invoke(owner, binding, lane), return_exceptions=True)
    assert not isinstance(outcome, BaseException), 'Owner retention failed during actual deferred publication'
    assert published == ['original publication']
    assert leases[0].released and not binding.registration.task_holds
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)


@pytest.mark.parametrize('max_holds,max_held_tools', [(0, 1), (1, 0), (True, 1), (1, True), (1, 4)])
async def test_invalid_owner_budgets_never_acquire_retention(dispatch, max_holds, max_held_tools):
    owner = dispatch
    owner.MAX_DIRECT_TOOLS = 4

    async def handler(value):
        return {'success': True}

    binding, = mount(owner, 'budget', handler)
    with pytest.raises(ValueError, match='headroom'):
        owner._acquire_task_owner_hold(binding, max_holds=max_holds, max_held_tools=max_held_tools)
    assert binding.is_current and not binding.registration.task_holds


async def test_owner_holds_cannot_be_acquired_or_released_through_foreign_descriptions(dispatch):
    from dataclasses import replace
    owner = dispatch
    owner.MAX_DIRECT_TOOLS = 4

    async def handler(value):
        return {'success': True}

    binding, = mount(owner, 'owned', handler)
    for foreign in (object(), replace(binding, agent=object())):
        with pytest.raises(ResourceConflict, match='original registered tool'):
            owner._acquire_task_owner_hold(foreign, max_holds=1, max_held_tools=1)
    hold = owner._acquire_task_owner_hold(binding, max_holds=1, max_held_tools=1)
    with pytest.raises(TypeError, match='another registry'):
        owner._release_task_owner_hold(object())
    assert binding.registration.task_holds == {hold}
    owner._release_task_owner_hold(hold)
