"""Actual registering/bound Tasks distinguish input ownership from copied ids."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import create_owned_task
from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin


def test_synchronous_registration_has_no_native_input_owner():
    owner = RequestLifecycleMixin()
    generation = owner.register_active_request('synchronous-bookkeeping')
    try:
        assert owner._request_generation_for_current_task('synchronous-bookkeeping') == generation
        assert owner._request_generation_for_current_task('synchronous-bookkeeping', original_only=True) is None
    finally:
        owner._cleanup_cancelled_request('synchronous-bookkeeping', generation=generation)


@pytest.mark.asyncio
async def test_original_request_generation_requires_live_native_owner():
    owner = RequestLifecycleMixin()
    request_id = 'actual-original-request-ownership'
    generation = owner.register_active_request(request_id)
    assert owner._request_generation_for_current_task(request_id, original_only=True) == generation
    observed, entered, admitted = [], asyncio.Event(), asyncio.Event()

    async def operation():
        observed.append(owner._request_generation_for_current_task(request_id, original_only=True))
        entered.set()
        await admitted.wait()
        observed.append(owner._request_generation_for_current_task(request_id, original_only=True))

    child = create_owned_task(operation(), retain_originals=True)
    try:
        await entered.wait()
        owner.bind_request_operation(request_id, child)
    finally:
        admitted.set()
        await child
        owner._cleanup_cancelled_request(request_id, generation=generation)
    assert observed[0] is None, 'copied generation granted original request ownership'
    assert observed[1] == generation, 'bound original operation lost request ownership'
    assert owner._request_generation_for_current_task(request_id, original_only=True) is None, (
        'retired generation granted original request ownership')


def test_active_projection_does_not_prove_another_request_owner():
    original, other = RequestLifecycleMixin(), RequestLifecycleMixin()
    generation = original.register_active_request('active-projection-only')
    other_generation = other.register_active_request('different-registration-context')
    try:
        assert original._request_generation_for_current_task('active-projection-only') == generation
        assert original._request_generation_for_current_task('active-projection-only', original_only=True) is None, (
            'active generation projection granted original request ownership')
    finally:
        other._cleanup_cancelled_request('different-registration-context', generation=other_generation)
        original._cleanup_cancelled_request('active-projection-only', generation=generation)


@pytest.mark.asyncio
async def test_loop_callback_registration_has_no_original_task_owner():
    owner, loop = RequestLifecycleMixin(), asyncio.get_running_loop()
    result = loop.create_future()

    def callback():
        generation = owner.register_active_request('callback-bookkeeping-only')
        try:
            result.set_result(owner._request_generation_for_current_task(
                'callback-bookkeeping-only', original_only=True))
        except BaseException as error:
            result.set_exception(error)
        finally:
            owner._cleanup_cancelled_request('callback-bookkeeping-only', generation=generation)

    loop.call_soon(callback)
    assert await result is None, 'taskless callback granted original request ownership'


@pytest.mark.asyncio
async def test_unowned_requester_cannot_publish_operation_binding():
    owner = RequestLifecycleMixin()
    request_id = 'original-operation-binding-publisher'
    generation = owner.register_active_request(request_id)

    async def copied_requester():
        try:
            owner.bind_request_operation(request_id, asyncio.current_task())
        except RuntimeError:
            return False
        return True

    child = create_owned_task(copied_requester(), retain_originals=True)
    try:
        published = await child
    finally:
        owner._cleanup_cancelled_request(request_id, generation=generation)
    assert published is False, 'unowned requester published original operation binding'


@pytest.mark.asyncio
async def test_captured_binding_delegates_only_its_original_operation():
    owner = RequestLifecycleMixin()
    request_id = 'delegated-original-operation-binding'
    generation = owner.register_active_request(request_id)
    publish = owner.capture_request_operation_binding(request_id)
    entered, released = asyncio.Event(), asyncio.Event()
    observed = []

    async def operation():
        entered.set()
        await released.wait()
        observed.append(owner._request_generation_for_current_task(request_id, original_only=True))

    target = create_owned_task(operation(), retain_originals=True)
    await entered.wait()

    async def driver():
        before = owner._request_generation_for_current_task(request_id, original_only=True)
        publish(target)
        after = owner._request_generation_for_current_task(request_id, original_only=True)
        return before, after

    driving = create_owned_task(driver(), retain_originals=True)
    try:
        assert await driving == (None, None), 'captured publication granted its foreign driver input ownership'
    finally:
        released.set()
        await target
        owner._cleanup_cancelled_request(request_id, generation=generation)
    assert observed == [generation], 'captured original operation did not receive its native binding'


@pytest.mark.asyncio
async def test_captured_binding_observes_its_original_cancellation_generation():
    owner, request_id = RequestLifecycleMixin(), 'captured-original-stop-generation'
    original_generation = owner.register_active_request(request_id)
    publish = owner.capture_request_operation_binding(request_id)
    fresh_generation = owner.register_active_request(request_id, nested=False)
    assert fresh_generation != original_generation
    assert owner.cancel_current_request(request_id, generation=original_generation)
    calls = []

    async def operation():
        calls.append('payload dispatched after original Stop')

    target = create_owned_task(operation(), retain_originals=True)
    try:
        publish(target)
        await asyncio.gather(target, return_exceptions=True)
        assert target.cancelled() and not calls, 'captured original Stop was replaced by the driver generation'
        assert owner._request_generation_for_current_task(request_id) == fresh_generation
        assert owner.is_request_cancelled(request_id) is False
    finally:
        owner._cleanup_cancelled_request(request_id, generation=original_generation)
        owner._cleanup_cancelled_request(request_id, generation=fresh_generation)


@pytest.mark.asyncio
async def test_retired_captured_binding_cannot_dispatch_an_owned_stream():
    from kestrel_sovereign._async_ownership import create_owned_iterator
    owner, request_id = RequestLifecycleMixin(), 'retired-original-stream-binding'
    generation = owner.register_active_request(request_id)
    publish = owner.capture_request_operation_binding(request_id)
    owner._cleanup_cancelled_request(request_id, generation=generation)
    fresh_generation = owner.register_active_request(request_id, nested=False)
    calls, stream = [], None

    async def source():
        calls.append('unadmitted original source ran')
        yield 'unadmitted'

    try:
        with pytest.raises(RuntimeError, match='binding publication is retired'):
            stream = await create_owned_iterator(source, operation='retired request binding',
                owner_started=lambda _iterator, task: publish(task))
            await anext(stream)
    finally:
        if stream is not None:
            await stream.aclose()
        owner._cleanup_cancelled_request(request_id, generation=fresh_generation)
    assert not calls, 'retired binding dispatched the original stream payload'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_stage', [None, 'capture', 'publish'])
async def test_http_stream_binds_original_producer_before_payload(tmp_path, failure_stage):
    import httpx
    from fastapi import FastAPI
    from unittest.mock import AsyncMock, MagicMock
    from kestrel_sovereign.endpoints.agent import stream_agent_response
    from kestrel_sovereign.streams.tap import AgentStreamTap

    payload = tmp_path / 'original-stream-payload'
    observations = []

    class Agent(RequestLifecycleMixin):
        def __init__(self):
            self.storage = MagicMock()
            self.storage.resolve_session_id = AsyncMock(return_value=None)

        def capture_request_operation_binding(self, request_id):
            publish = super().capture_request_operation_binding(request_id)
            if failure_stage == 'capture':
                raise OSError('original publisher capture refused')

            def captured(operation, **kwargs):
                observations.append('publication')
                publish(operation, **kwargs)
                if failure_stage == 'publish':
                    raise OSError('original stream publication refused')

            return captured

        async def process_input_streaming(self, *_args, request_id, **_kwargs):
            assert self._request_generation_for_current_task(request_id, original_only=True) is not None
            observations.append('payload')
            payload.write_text('actual dispatched source')
            yield 'original source payload'

    agent, app = Agent(), FastAPI()
    app.state.agent = agent
    endpoint = getattr(stream_agent_response, '__wrapped__', stream_agent_response)
    app.add_api_route('/api/agent/stream', endpoint, methods=['POST'])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/agent/stream', json={'input': 'work', 'request_id': 'original-http-producer'})
    # A successful tap retains the generation for its final voice drain.
    delivery = response.headers.get('x-stream-delivery-id')
    if delivery is not None:
        AgentStreamTap.get_instance().unregister(delivery)
    if failure_stage == 'capture':
        assert response.status_code == 500
        assert not observations and not payload.exists()
    elif failure_stage == 'publish':
        assert response.status_code == 200
        assert observations == ['publication'], 'failed publication dispatched original source'
        assert not payload.exists(), 'unpublished producer performed its physical effect'
        assert 'could not be completed' in response.text
    else:
        assert response.status_code == 200
        assert observations == ['publication', 'payload']
        assert payload.read_text() == 'actual dispatched source'
        assert response.text == 'original source payload'
    assert not agent._active_request_ids
    assert not vars(agent).get('_request_operation_tasks', {})


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', ['operation', 'cancellation'])
async def test_captured_binding_rejects_non_task_targets(invalid):
    owner, request_id = RequestLifecycleMixin(), 'invalid-original-task-target'
    generation = owner.register_active_request(request_id)
    publish = owner.capture_request_operation_binding(request_id)

    async def operation():
        return None

    task = create_owned_task(operation(), retain_originals=True)
    try:
        with pytest.raises(TypeError, match='must be an asyncio task'):
            if invalid == 'operation':
                publish(object())
            else:
                publish(task, cancel_via=object())
        assert not vars(owner).get('_request_operation_tasks', {})
    finally:
        await task
        owner._cleanup_cancelled_request(request_id, generation=generation)
