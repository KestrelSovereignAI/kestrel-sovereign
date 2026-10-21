"""Successful original pool/socket closes are not retried for observer failures."""
import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.phoenix_supervisor import PhoenixSupervisor
from kestrel_sovereign.server import _shutdown_phoenix
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_native_completion_vs_observation import observation_factory
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg


@asynccontextmanager
async def original_proxy(tmp_path):
    peers, closed = set(), asyncio.Event()
    async def peer(reader, writer):
        task = asyncio.current_task()
        peers.add(task)
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            closed.set()
            peers.discard(task)
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    client = httpx.AsyncClient(trust_env=False)
    supervisor = PhoenixSupervisor(working_dir=tmp_path)
    supervisor._client = client
    supervisor.pid_file.write_text("declared fixture; no launched or adopted process")
    try:
        assert supervisor.process is None and supervisor._adopted_pid is None
        assert (await client.get("http://127.0.0.1:" + str(server.sockets[0].getsockname()[1]) + "/original")).status_code == 200
        yield supervisor, client, closed
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("observer", ["ordinary", "native", "native-cancel"])
@pytest.mark.parametrize("target", ["postgres", "phoenix"])
async def test_successful_original_close_reports_observer_without_false_physical_debt(
    native_pg, tmp_path, monkeypatch, eager, observer, target,
):
    backend = native_pg[0]
    primary = backend._pool
    # The shared fixture's primary pool is BORROWED. Exercise its actual
    # independently-owned advisory pool, never claim borrowed detachment ran
    # an original close or transfer somebody else's primary ownership.
    pool = await backend._ensure_advisory_pool() if target == "postgres" else None
    if pool is not None:
        async with pool.acquire() as connection:
            assert await connection.fetchval("SELECT 1") == 1
    snapshot = None
    if observer == "ordinary":
        failure = OSError("actual close completion removal failed")
    else:
        snapshot = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, observer == "native-cancel"))
        failure = snapshot.error
    attached, accepted = asyncio.Event(), []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    owner = None
    async with original_proxy(tmp_path) as (supervisor, client, peer_closed):
        app = FastAPI()
        app.state.phoenix = supervisor
        if target == "postgres":
            operation = backend.close
            target_code = type(pool).close.__code__
        else:
            operation = lambda: _shutdown_phoenix(app)
            target_code = PhoenixSupervisor.aclose.__code__
        factory = observation_factory(
            lambda coroutine: getattr(coroutine, "cr_code", None) is target_code,
            failure, attached, accepted, previous, eager=eager,
        )
        owner = create_owned_task(operation(), retain_originals=True)
        try:
            loop.set_task_factory(factory)
            result = await await_owned_task(owner)
            assert accepted and all(task.done() for task in accepted)
            assert failure in tuple(_explicit_control_chain(result.error))
            if snapshot is not None:
                snapshot.assert_retained(result.error)
            assert tuple(execution_cleanup_debts(result.error)) == (() if snapshot is None else snapshot.debts)
            if target == "postgres":
                assert backend._advisory_pool is None and pool._closed
                assert backend._pool is None and not primary._closed
                assert not backend._pool_close_owners
            else:
                assert client.is_closed and supervisor._client is None
                await asyncio.wait_for(peer_closed.wait(), 5)
                assert app.state.phoenix is None
                assert app.state.phoenix_proxy_close_task is None
                assert getattr(app.state, "phoenix_proxy_close_error", None) is None
                assert not supervisor.pid_file.exists()
            count = len(accepted)
            later = await operation()
            assert later is (False if target == "phoenix" else None)
            assert len(accepted) == count
            if target == "postgres":
                assert backend._pool is None
            else:
                assert app.state.phoenix is None
        finally:
            loop.set_task_factory(previous)
            if owner is not None:
                await await_owned_task(owner)
            for task in accepted:
                await await_owned_task(task)
