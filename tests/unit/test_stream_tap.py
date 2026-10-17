"""Unit tests for the generic agent stream tap."""

import pytest

from kestrel_sovereign.streams.tap import AgentStreamTap


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
