"""Unit tests for the generic agent stream tap."""

import pytest

from kestrel_sovereign.streams.tap import AgentStreamTap


@pytest.mark.asyncio
async def test_tap_native_admission_and_completed_drain_use_their_actual_lifetimes(native_tool_effect_agent, tmp_path):
    import asyncio
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.storage.privacy_wrapper import (
        bind_transition_lock_reentry, current_bound_reentry_token,
    )

    agent = await native_tool_effect_agent(did='did:test:original-native-tap')
    lock, tap = agent._get_privacy_transition_lock(), AgentStreamTap()
    ready, release = asyncio.Event(), asyncio.Event()
    request, delivery = 'native-tap-input', 'native-tap-delivery'
    path = tmp_path / 'native-tap-effects'
    recorded, policy = [], []
    from kestrel_sovereign.telemetry import bind_current_turn_id, current_turn_id
    expected_turn = 'live-voice-consumer'

    async def effect():
        policy.append((lock.current_native_scope(), current_bound_reentry_token(), current_turn_id()))
        assert current_turn_id() == expected_turn, 'tap imported a different consumer turn context'
        with path.open('a') as file:
            file.write('original effect\n')
        return 'acknowledged'

    async def nested_provider():
        async with lock:
            child = lock.start_owned_native_work(effect)
            result = await await_owned_task(child)
            recorded.append((lock._writer is asyncio.current_task(), lock._depth))
            return result

    async def produce():
        generation = agent.register_active_request(request)
        try:
            tap.register(delivery, owner=agent, request_id=request)
            async with lock.policy_span():
                tap.capture_request_privacy(agent, request, generation)
                ready.set()
                await release.wait()
            await tap.finish(delivery, completed=True)
        finally:
            agent._cleanup_cancelled_request(request, generation=generation)

    producer = create_owned_task(produce())
    try:
        await asyncio.wait_for(ready.wait(), 1)
        token = tap._requests[delivery].privacy_token
        original = tap._requests[delivery].native_work_factory
        with bind_transition_lock_reentry(token):
            with pytest.raises(ExecutionAuthorityError, match='original native producer'):
                tap.capture_request_privacy(agent, request, tap._requests[delivery].generation)
        assert tap._requests[delivery].native_work_factory is original
        consumer_context = object()
        with bind_transition_lock_reentry(consumer_context), bind_current_turn_id(expected_turn):
            async with tap.producer_privacy(delivery, agent) as factory:
                assert current_bound_reentry_token() is consumer_context, (
                    'tap replaced consumer turn context instead of admitting original native work')
                assert factory is original
                assert callable(factory), 'tap did not retain original native work admission'
                outcome = await await_owned_task(factory(nested_provider))
        assert outcome.error is None and outcome.result.error is None
        assert outcome.result.result == 'acknowledged' and recorded == [(True, 2)]
        release.set()
        assert (await await_owned_task(producer)).error is None
        # A final drain admits its own native policy decision, while keeping
        # the consumer's complete turn context outside that owned worker.
        expected_turn = 'final-voice-consumer'
        with bind_transition_lock_reentry(token), bind_current_turn_id(expected_turn):
            async with tap.producer_privacy(delivery, agent) as factory:
                assert current_bound_reentry_token() is token, 'tap changed its consumer turn context'
                assert callable(factory), 'completed tap drain has no native work factory'
                final = await await_owned_task(factory(effect))
        assert final.error is None and final.result == 'acknowledged'
        assert policy[0] == (token, token, 'live-voice-consumer')
        assert policy[1][0] is not None and policy[1][0] is not token, (
            'completed tap drain reused its retired text span')
        assert policy[1][1:] == (None, 'final-voice-consumer')
        assert path.read_text() == 'original effect\noriginal effect\n'
    finally:
        release.set()
        await await_owned_task(producer)
        tap.unregister(delivery)
    assert not lock.locked() and not lock._native_scopes


@pytest.mark.asyncio
async def test_tap_drain_waits_for_the_original_native_lifetime_to_close(native_tool_effect_agent, tmp_path):
    import asyncio
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task

    agent = await native_tool_effect_agent(did='did:test:closing-native-tap')
    lock, tap = agent._get_privacy_transition_lock(), AgentStreamTap()
    ready, entered, leave, release = (asyncio.Event() for _ in range(4))
    request, delivery = 'closing-native-input', 'closing-native-delivery'
    path = tmp_path / 'closing-native-effects'
    tasks = {}

    async def first_effect():
        entered.set()
        await release.wait()
        path.write_text('original effect\n')

    async def final_effect():
        with path.open('a') as file:
            file.write('final drain\n')

    async def produce():
        generation = agent.register_active_request(request)
        try:
            tap.register(delivery, owner=agent, request_id=request)
            async with lock.policy_span():
                tap.capture_request_privacy(agent, request, generation)
                tasks['first'] = lock.start_owned_native_work(first_effect)
                ready.set()
                await leave.wait()
            await tap.finish(delivery, completed=True)
        finally:
            agent._cleanup_cancelled_request(request, generation=generation)

    async def read_final():
        async with tap.producer_privacy(delivery, agent) as factory:
            assert callable(factory), 'completed native drain did not receive its own factory'
            child = factory(final_effect)
            return await await_owned_task(child)

    producer, reader = create_owned_task(produce()), None
    try:
        await asyncio.wait_for(ready.wait(), 1)
        await asyncio.wait_for(entered.wait(), 1)
        leave.set()
        async with asyncio.timeout(1):
            while not tap._requests[delivery].privacy_token.closing:
                await asyncio.sleep(0)
        reader = create_owned_task(read_final())
        _, pending = await asyncio.wait({reader}, timeout=.03)
        blocked = bool(pending)
    finally:
        leave.set()
        release.set()
        produced = await await_owned_task(producer)
        first = await await_owned_task(tasks['first']) if 'first' in tasks else None
        observed = await await_owned_task(reader) if reader is not None else None
        tap.unregister(delivery)
    assert blocked, 'tap drain did not wait for original native lifetime closure'
    assert produced.error is None and first.error is None
    assert observed.error is None and observed.result.error is None
    assert path.read_text() == 'original effect\nfinal drain\n'
    assert not lock.locked() and not lock._native_scopes


@pytest.mark.asyncio
@pytest.mark.parametrize('refusal', ['stopped', 'foreign-owner', 'incomplete'])
async def test_tap_native_admission_refuses_unowned_or_unacknowledged_work(native_tool_effect_agent, refusal):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    agent = await native_tool_effect_agent(did='did:test:refused-native-tap')
    request, delivery = 'refused-native-tap-input', 'refused-native-tap-delivery'
    generation = agent.register_active_request(request)
    tap = AgentStreamTap()
    tap.register(delivery, owner=agent, request_id=request)
    consumer = agent
    try:
        if refusal == 'foreign-owner':
            consumer = await native_tool_effect_agent(did=agent.did)
        elif refusal == 'stopped':
            agent.cancel_current_request(request, generation=generation)
        else:
            await tap.finish(delivery, completed=False)
        with pytest.raises(ExecutionAuthorityError):
            async with tap.producer_privacy(delivery, consumer):
                pytest.fail('unowned or unacknowledged tap admitted native work')
    finally:
        tap.unregister(delivery)
        agent._cleanup_cancelled_request(request, generation=generation)


@pytest.mark.asyncio
async def test_stream_tap_publishes_chunks_and_cleans_up():
    AgentStreamTap.reset()
    tap = AgentStreamTap.get_instance()
    request_id = "req-1"

    tap.register(request_id)
    await tap.publish(request_id, "hello")
    await tap.publish(request_id, " world")
    await tap.finish(request_id)

    chunks = [chunk async for chunk in tap.subscribe(request_id)]

    assert chunks == ["hello", " world"]
    assert not tap.has_stream(request_id)


@pytest.mark.asyncio
async def test_stream_tap_delivery_queues_are_isolated_for_the_same_retry():
    """Finishing one delivery must not tear down a concurrent retry's tap."""
    AgentStreamTap.reset()
    tap = AgentStreamTap.get_instance()
    first_delivery = "stream:first-delivery"
    retry_delivery = "stream:retry-delivery"

    tap.register(first_delivery)
    tap.register(retry_delivery)
    await tap.publish(first_delivery, "first")
    await tap.finish(first_delivery)

    assert [chunk async for chunk in tap.subscribe(first_delivery)] == ["first"]
    assert not tap.has_stream(first_delivery)
    assert tap.has_stream(retry_delivery)

    await tap.publish(retry_delivery, "retry")
    await tap.finish(retry_delivery)
    assert [chunk async for chunk in tap.subscribe(retry_delivery)] == ["retry"]
    assert not tap.has_stream(retry_delivery)


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_original_stopped_tap_body_physically_retires_without_abandonment(tmp_path, started):
    import asyncio
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    from kestrel_sovereign.execution_custody import (
        execution_cleanup_debts, execution_stream, is_execution_control_error,
    )
    from tests.unit.test_execution_custody_review29 import native_owner, retire

    agent, db = await native_owner(tmp_path)
    tap = AgentStreamTap()
    request_id, delivery = "original-native-stop", "original-native-delivery"
    generation = agent.register_active_request(request_id)
    tap.register(delivery, owner=agent, request_id=request_id)
    await tap.finish(delivery, completed=True)
    agent._cleanup_cancelled_request(request_id, generation=generation)
    producer = tap.producer_custody(delivery, agent)
    async def audio():
        yield await db.fetchval("SELECT CAST('audio' AS BLOB)")
        await asyncio.Event().wait()
    stream = tap.own_response(delivery, agent,
        execution_stream(agent, audio(), captured_custody=producer))
    try:
        if started:
            assert await anext(stream) == b"audio"
        assert agent.cancel_current_request(request_id, generation=generation)
        with pytest.raises(BaseException) as stopped:
            await anext(stream)
        assert is_execution_control_error(stopped.value)
        assert not execution_cleanup_debts(stopped.value)
        tap.unregister(delivery)
        assert await asyncio.wait_for(agent.wait_for_request_completion(request_id), 2) is RequestCompletionDisposition.COMPLETED
    finally:
        tap.unregister(delivery)
        try:
            await stream.aclose()
        except BaseException as error:
            assert is_execution_control_error(error)
        await retire(agent)
        await db.close()
