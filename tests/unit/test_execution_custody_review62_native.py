"""Review62 original native retirement and session-only metadata proofs."""
import asyncio

import httpx
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import close_native_execution_stream, NativeExecutionStreamOwner
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


@pytest.mark.asyncio
async def test_isolated_native_audio_keeps_metadata_without_persistent_file(tmp_path):
    async with AsyncStorage(str(tmp_path / "isolated.db"), agent_id="did:test:review62-metadata") as storage:
        isolated = PrivacyEnforcingStorage(storage, PrivacyMode.ISOLATED)
        metadata = {"format": "pcm", "type": "tts_audio"}
        digest = await isolated.store_file(b"\x00\x01" * 480, "speech.pcm", metadata)
        metadata["format"] = "wrong"
        assert await isolated.retrieve_file(digest) == b"\x00\x01" * 480
        assert await isolated.get_file_metadata(digest) == {"format": "pcm", "type": "tts_audio"}
        assert await storage.get_file_metadata(digest) is None
        isolated.clear_session()
        assert await isolated.get_file_metadata(digest) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_kind", ["direct", "inventory"])
async def test_cancel_during_successful_native_http_close_keeps_cancellation_not_attribute_error(monkeypatch, owner_kind):
    native_closed, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    peers = set()
    async def peer(reader, writer):
        peers.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            native_closed.set()
            peers.discard(asyncio.current_task())
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    client = httpx.AsyncClient()
    closing = None
    try:
        response = await client.send(client.build_request("GET",
            f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"), stream=True)
        native_close = response.aclose
        async def held_native_close():
            await native_close()
            entered.set()
            await release.wait()
        monkeypatch.setattr(response, "aclose", held_native_close)
        inventory = NativeExecutionStreamOwner()
        inventory.retain(response)
        closing = create_owned_task(close_native_execution_stream(response)
            if owner_kind == "direct" else inventory.close())
        await asyncio.wait_for(entered.wait(), 2)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        result = await await_owned_task(closing)
        assert isinstance(result.error, asyncio.CancelledError)
        assert not isinstance(result.error, AttributeError)
        await asyncio.wait_for(native_closed.wait(), 2)
        if owner_kind == "inventory":
            assert inventory.handles == [] and inventory.retirement_error is None
    finally:
        release.set()
        if closing is not None:
            await await_owned_task(closing)
        await client.aclose()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)
