"""Native HTTP dispatch gates against real owned loopback connections."""

import asyncio
from types import SimpleNamespace

import httpx
import httpx2
import pytest
import pytest_asyncio

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    bind_native_execution_admission,
    bind_native_http_execution_admission,
    owned_native_http_client,
)


@pytest_asyncio.fixture
async def native_http_server():
    hits = []
    tasks = []

    async def serve(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            hits.append(headers.split(b"\r\n", 1)[0])
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    def accepted(reader, writer):
        tasks.append(asyncio.create_task(serve(reader, writer)))

    server = await asyncio.start_server(accepted, "127.0.0.1", 0)
    try:
        yield SimpleNamespace(
            url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", hits=hits
        )
    finally:
        server.close()
        await server.wait_closed()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(result is None for result in results), results


@pytest.mark.parametrize("library", [httpx, httpx2])
async def test_targeted_gate_does_not_recursively_gate_its_lookup_client(
    native_http_server, library
):
    server = native_http_server
    ordinary_checks = []
    peer_checks = []

    async def ordinary():
        ordinary_checks.append(True)

    async with (
        owned_native_http_client(library.AsyncClient(trust_env=False)) as peer,
        owned_native_http_client(library.AsyncClient(trust_env=False)) as lookup,
    ):

        async def authorize():
            peer_checks.append(True)
            assert len(peer_checks) == 1, (
                "Peer gate must not recursively authorize its own lookup"
            )
            assert (await lookup.get(server.url)).text == "ok"

        with (
            bind_native_execution_admission(ordinary),
            bind_native_http_execution_admission(peer, authorize),
        ):
            assert (await peer.get(server.url)).text == "ok"
    assert len(peer_checks) == 1
    assert len(ordinary_checks) >= 2, "Other execution gates apply to both real clients"
    assert len(server.hits) == 2


@pytest.mark.parametrize("library", [httpx, httpx2])
async def test_targeted_gate_runs_after_request_observers_before_socket_dispatch(
    native_http_server, library
):
    state = SimpleNamespace(allowed=True)
    denial = ExecutionAuthorityError("Authority revoked by observer")

    async def observer(request):
        state.allowed = False

    async def authorize():
        if not state.allowed:
            raise denial

    async with owned_native_http_client(
        library.AsyncClient(
            trust_env=False,
            event_hooks={"request": [observer]},
        )
    ) as client:
        with bind_native_http_execution_admission(client, authorize):
            with pytest.raises(ExecutionAuthorityError) as caught:
                await client.get(native_http_server.url)
    assert caught.value is denial
    assert not native_http_server.hits


@pytest.mark.parametrize("library", [httpx, httpx2])
async def test_copied_child_keeps_original_targeted_gate_after_parent_scope_resets(
    native_http_server, library
):
    start = asyncio.Event()
    denial = ExecutionAuthorityError("Original peer scope revoked")

    async def authorize():
        raise denial

    async with owned_native_http_client(library.AsyncClient(trust_env=False)) as client:

        async def child():
            await start.wait()
            await client.get(native_http_server.url)

        with bind_native_http_execution_admission(client, authorize):
            task = asyncio.create_task(child())
        start.set()
        with pytest.raises(ExecutionAuthorityError) as caught:
            await task
    assert caught.value is denial
    assert not native_http_server.hits


def test_targeted_gate_refuses_non_native_client():
    with (
        pytest.raises(TypeError, match="actual native client"),
        bind_native_http_execution_admission(object(), lambda: None),
    ):
        pass
