"""Decision HTTP normalization cannot erase original native close ownership."""

import asyncio

import httpx
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.llm.decisions.http import post_systemone


@pytest.mark.asyncio
async def test_actual_decision_http_close_keeps_original_native_debt(monkeypatch):
    peers, received, closed = set(), [], asyncio.Event()

    async def peer(reader, writer):
        task = asyncio.current_task()
        peers.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            lengths = [
                line.split(b":", 1)[1].strip()
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            ]
            body = await reader.readexactly(int(lengths[0]))
            received.append(body)
            payload = b'{"action":"noop"}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\n\r\n"
                + payload
            )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            closed.set()
            peers.discard(task)

    listener = await asyncio.start_server(peer, "127.0.0.1", 0)
    native_close, clients = httpx.AsyncClient.aclose, []
    original = httpx.ReadError("private original physical-close ACK lost")
    source_graph = tuple(_explicit_control_chain(original))
    source_debts = tuple(execution_cleanup_debts(original))

    async def native_close_then_lose_ack(client):
        clients.append(client)
        await native_close(client)
        raise original

    monkeypatch.setattr(httpx.AsyncClient, "aclose", native_close_then_lose_ack)
    try:
        with pytest.raises(BaseException) as caught:
            await post_systemone(
                f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}/decision",
                b'{"model":"local-native-proof"}',
                timeout=5,
                route="native-loopback",
            )
        await asyncio.wait_for(closed.wait(), 2)
        assert received == [b'{"model":"local-native-proof"}']
        assert len(clients) == 1 and clients[0].is_closed
        assert original in tuple(_explicit_control_chain(caught.value)) and is_execution_control_error(caught.value), (
            "decision transport normalization discarded original native control"
        )
        assert tuple(_explicit_control_chain(original)) == source_graph
        assert tuple(execution_cleanup_debts(original)) == source_debts
        debts = execution_cleanup_debts(caught.value)
        assert any(resource is clients[0] and owner.done() for resource, owner in debts)
    finally:
        monkeypatch.undo()
        for client in clients:
            await native_close(client)
        listener.close()
        await listener.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)
