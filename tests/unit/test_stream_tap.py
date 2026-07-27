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
