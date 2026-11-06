"""Real Agent generations, native SQL admission and external file effects."""

import asyncio
import inspect
import pytest

from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign.agent.invocation import (
    admit_request_operation, bind_async_generator_invocation, bind_async_invocation,
    request_provenance,
)
from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.api_errors import tool_effect_api_error
from kestrel_sovereign.llm.streaming_errors import bridge_sse_error_event
from kestrel_sovereign.tool_effect_store import (
    ToolEffectOperationConflict, ToolEffectReconciliationRequired,
)
from kestrel_sovereign.tools.effect_invocation import (
    journal_async_generator_invocation, journal_async_invocation,
)
from kestrel_sovereign.tools.effect_journal import (
    ToolEffectOutcomeUnconfirmed, current_tool_effect_turn, execute_tool_effect,
)
from tests.unit.test_tool_effect_execution import owner


pytestmark = pytest.mark.timeout(30)


def governed(function):
    return bind_async_invocation('invocation_id', track_request_lifecycle=True)(
        journal_async_invocation(function))


def governed_stream(function):
    return bind_async_generator_invocation('request_id', track_request_lifecycle=True)(
        journal_async_generator_invocation(function))


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
    with pytest.raises(ToolEffectReconciliationRequired) as blocked:
        await invoke(owner, 'one effect', user_passphrase='different-secret', **rotated)
    public = tool_effect_api_error(blocked.value)
    assert public.status_code == 409 and public.code == 'tool_effect_reconciliation_required'
    assert public.details == [{'counts': dict(returned_ok=1, returned_error=0, unconfirmed=0, unsettled=0, not_executed=0)}]
    assert public.headers is None
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
@pytest.mark.parametrize('depth', [1, 3])
async def test_nested_invocation_keeps_its_original_parent_effect_claim(owner, tmp_path, depth):
    witness = tmp_path / 'nested-effect'

    @governed
    async def invoke(agent, user_input, *, invocation_id=None, nested=0):
        async def effect():
            if nested < depth:
                return await invoke(agent, user_input, invocation_id=invocation_id, nested=nested + 1)
            async with agent._get_privacy_transition_lock():
                witness.write_text('one physical effect')
            return 'done'

        return await execute_tool_effect(agent, registration=f'child-{nested}' if nested else 'parent',
            arguments={}, execute=effect)

    assert await invoke(owner, 'owned input', invocation_id='parent-intent') == 'done'
    claims = await owner.tool_effect_store.database.fetchall('SELECT claim_id, parent_claim_id FROM tool_effect_claims')
    roots = [row[0] for row in claims if row[1] is None]
    children = [row for row in claims if row[1] is not None]
    assert len(roots) == 1 and len(children) == depth, 'Nested invocation detached the original effect claim'
    parents = {row[0]: row[1] for row in children}
    for child in parents:
        ancestor = child
        while ancestor in parents:
            ancestor = parents[ancestor]
        assert ancestor == roots[0]
    assert witness.read_text() == 'one physical effect'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['allocation', 'binding'])
async def test_privacy_owned_invocation_refusal_never_constructs_its_lazy_body(
    owner, tmp_path, monkeypatch, failure,
):
    from kestrel_sovereign._async_ownership import await_owned_task
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from tests.unit.test_host_context_native_cleanup import _actual_operation

    witness = tmp_path / 'refused-lazy-effect'
    constructed, refused = [], []
    refusal = RuntimeError('Original privacy-owned invocation refused')
    lock = owner._get_privacy_transition_lock()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    actual_bind = type(owner).bind_request_operation
    request_id = 'owned-lazy-refusal'
    owner.register_active_request(request_id)

    async def body():
        witness.write_text('must not execute')

    def lazy_body():
        constructed.append(True)
        return body()

    def bind(self, rid, task):
        if self is owner and rid == request_id and failure == 'binding':
            raise refusal
        return actual_bind(self, rid, task)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, 'cr_code', None)
        if failure == 'allocation' and code is not None and code.co_qualname.endswith(
            'ReentrantTransitionLock.start_owned_native_work.<locals>.run'
        ):
            refused.extend((coroutine, operation))
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    monkeypatch.setattr(type(owner), 'bind_request_operation', bind)
    try:
        async with lock:
            loop.set_task_factory(factory)
            try:
                with pytest.raises(BaseException) as observed:
                    task = await admit_request_operation(owner, request_id, lazy_body,
                        name='owned-lazy-request', privacy_work_factory=lock.capture_native_work_factory())
                    outcome = await await_owned_task(task)
                    assert outcome.error is not None
                    raise outcome.error
            finally:
                loop.set_task_factory(previous)
            assert refusal in execution_evidence_carriers(observed.value)
        assert not constructed and not witness.exists()
        assert not lock._native_work
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused)
    finally:
        loop.set_task_factory(previous)
        owner._cleanup_cancelled_request(request_id)


@pytest.mark.asyncio
async def test_stream_never_holds_binding_tokens_across_outward_yields(owner):
    turns = []

    @governed_stream
    async def stream(agent, user_input, *, request_id=None):
        turn = current_tool_effect_turn(agent)
        turns.append(turn)
        yield 'first'
        assert current_tool_effect_turn(agent) is turn
        yield 'second'

    iterator = stream(owner, 'owned input', request_id='stream-intent')
    try:
        assert await anext(iterator) == 'first'
        assert current_tool_effect_turn(owner) is None
        assert await asyncio.create_task(anext(iterator)) == 'second'
        assert current_tool_effect_turn(owner) is None
        with pytest.raises(StopAsyncIteration):
            await asyncio.create_task(anext(iterator))
    finally:
        await iterator.aclose()
    assert len(turns) == 1 and turns[0].live is False
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('CLOSED',)


@pytest.mark.asyncio
async def test_stream_cannot_publish_narration_after_a_swallowed_unknown_effect(owner, tmp_path):
    witness = tmp_path / 'uncertain-effect'

    @governed_stream
    async def stream(agent, user_input, *, request_id=None):
        yield 'known prefix'

        async def effect():
            witness.write_text('one physical effect')
            raise ResourceOutcomeUnknown('owned result lost after effect')

        try:
            await execute_tool_effect(agent, registration='owned-file', arguments={}, execute=effect)
        except ResourceOutcomeUnknown:
            pass
        yield 'incorrect success narration'

    iterator = stream(owner, 'owned input', request_id='uncertain-intent')
    try:
        assert await anext(iterator) == 'known prefix'
        with pytest.raises(ToolEffectOutcomeUnconfirmed, match='tool_effect_outcome_unconfirmed') as blocked:
            await anext(iterator)
        public = tool_effect_api_error(blocked.value)
        assert public.status_code == 502 and public.code == 'tool_effect_outcome_unconfirmed'
        assert public.headers is None
        event = bridge_sse_error_event(blocked.value)
        assert '"code": "tool_effect_outcome_unconfirmed"' in event
        assert 'owned result lost' not in event and 'incorrect success' not in event
    finally:
        await iterator.aclose()
    assert witness.read_text() == 'one physical effect'
    assert current_tool_effect_turn(owner) is None
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_claims') == ('UNCONFIRMED',)
