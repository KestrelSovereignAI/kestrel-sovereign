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
