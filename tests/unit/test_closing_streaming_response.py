"""Transport interruption still closes owned streaming bodies (#3152)."""

import asyncio

import pytest
from starlette.requests import ClientDisconnect

from kestrel_sovereign.endpoints.closing_streaming_response import (
    ClosingStreamingResponse,
)


@pytest.mark.asyncio
async def test_send_failure_after_first_chunk_closes_body() -> None:
    finalized = asyncio.Event()

    async def body():
        try:
            yield "first"
            await asyncio.Event().wait()
        finally:
            finalized.set()

    response = ClosingStreamingResponse(body())

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            raise OSError("client disconnected")

    with pytest.raises(ClientDisconnect):
        await response(
            {"type": "http", "asgi": {"spec_version": "2.4"}},
            receive,
            send,
        )

    assert finalized.is_set()
