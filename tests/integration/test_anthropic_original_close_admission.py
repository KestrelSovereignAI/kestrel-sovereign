"""Actual SDK stream retirement under closer refusal/observer withdrawal.

Only loopback HTTP and a disposable local PG schema are used. The SSE bytes
are protocol input, not provider acceptance, token output or Agent READY.
"""

import asyncio
import inspect

import aiosqlite
import anthropic
import pytest

from kestrel_sovereign import _async_ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.llm.anthropic_adapter import _anthropic_stream_with_retry
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_cleanup_task_admission import _operation_name


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_open", [False, True])
@pytest.mark.parametrize("mode", ["refusal", "body-cancel", "observer", "closer-cancel", "caused-closer-cancel"])
async def test_actual_anthropic_closer_keeps_original_stream_and_receipts(
    native_pg, monkeypatch, retry_open, mode
):
    backend, _, _ = native_pg
    async with backend.advisory_locks([(3569, 17828)]) as lease:
        lease.require_live()
    peers, requests = set(), []
    disconnected = asyncio.Event()

    async def peer(reader, writer):
        task = asyncio.current_task()
        peers.add(task)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            assert head.startswith(b"POST /v1/messages ")
            length = next(
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            requests.append(await reader.readexactly(length))
            # Keep the native response live; no complete message is fabricated.
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Content-Length: 1000000\r\n\r\nevent: ping\ndata: {}\n\n"
            )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            disconnected.set()
            peers.discard(task)

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    client = anthropic.AsyncAnthropic(
        api_key="inert-loopback-fixture-not-provider-credential",
        base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(trust_env=False),
    )
    loop, factory = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal = RuntimeError("actual Anthropic stream closer Task refused")
    refused, sources, streams, cancellations, closer_tasks = [], [], [], [], []
    original_graphs = []
    entered, release = asyncio.Event(), asyncio.Event()
    actual_join = _async_ownership.await_owned_task
    original_exit = None
    manager_type = None
    managers = []
    caused_cancellations, ordinary_causes = [], []
    closed_db = None
    if mode == "caused-closer-cancel":
        closed_db = await aiosqlite.connect(":memory:")
        await closed_db.close()
    actual_stream = client.messages.stream

    def observed_stream(*args, **kwargs):
        manager = actual_stream(*args, **kwargs)
        managers.append(manager)
        return manager

    monkeypatch.setattr(client.messages, "stream", observed_stream)

    def refuse(loop, coroutine, **kwargs):
        if _operation_name(coroutine).endswith(".close_acquired.<locals>.owned_close"):
            refused.append(coroutine)
            raise refusal
        return factory(loop, coroutine, **kwargs) if factory else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def observed_join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if task in closer_tasks and outcome.cancellation is not None:
            cancellations.append(outcome.cancellation)
        return outcome

    monkeypatch.setattr(_async_ownership, "await_owned_task", observed_join)

    async def run():
        nonlocal original_exit, manager_type
        async with _anthropic_stream_with_retry(
            client,
            {"model": "local-protocol-input", "max_tokens": 1, "messages": [{"role": "user", "content": "fixture"}]},
            retry_open=retry_open,
        ) as stream:
            streams.append(stream)
            assert not stream.response.is_closed
            if mode in {"refusal", "body-cancel"}:
                loop.set_task_factory(refuse)
            else:
                # Transparent SDK context exit: hold the actual physical close,
                # then execute it. No replacement stream or close ACK is used.
                manager_type = type(managers[0])
                original_exit = manager_type.__aexit__

                async def held_exit(manager, *args):
                    closer_tasks.append(asyncio.current_task())
                    entered.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError as cancellation:
                        if closed_db is not None:
                            try:
                                await closed_db.execute("SELECT 1")
                            except ValueError as ordinary:
                                ordinary_causes.append(ordinary)
                                cancellation.__cause__ = ordinary
                            caused_cancellations.append(cancellation)
                        raise
                    finally:
                        await original_exit(manager, *args)

                monkeypatch.setattr(manager_type, "__aexit__", held_exit)
            try:
                lease.require_live()
            except BaseException as original:
                if mode == "body-cancel":
                    asyncio.current_task().cancel("actual SDK consumer withdrawal")
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as cancelled:
                        cancelled.__cause__ = original
                        sources.append(cancelled)
                        original_graphs.append(tuple(_explicit_control_chain(cancelled)))
                        raise
                sources.append(original)
                original_graphs.append(tuple(_explicit_control_chain(original)))
                raise

    operation = create_owned_task(run(), retain_originals=True)
    try:
        if mode in {"observer", "closer-cancel", "caused-closer-cancel"}:
            await asyncio.wait_for(entered.wait(), 5)
            assert not streams[0].response.is_closed
            target = operation if mode == "observer" else closer_tasks[0]
            target.cancel("actual SDK retirement observer withdrawn")
            await asyncio.sleep(0)
            release.set()
        outcome = await asyncio.wait_for(await_owned_task(operation), 5)
        loop.set_task_factory(factory)
        assert outcome.error is not None and len(requests) == 1
        chain = tuple(_explicit_control_chain(outcome.error))
        assert sources and all(original in chain for original in sources), (
            "Anthropic closer admission/retirement replaced the original native stream failure"
        )
        assert all(tuple(_explicit_control_chain(error)) == before for error, before in zip(sources, original_graphs))
        assert is_execution_control_error(outcome.error)
        if mode in {"refusal", "body-cancel"}:
            assert refusal in chain and refused
            assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused)
            assert (streams[0], None) in execution_cleanup_debts(outcome.error), (
                "refused closer lost its actual acquired SDK stream"
            )
            assert not streams[0].response.is_closed and not disconnected.is_set()
        else:
            assert streams[0].response.is_closed and all(task.done() for task in closer_tasks)
            if mode == "observer":
                assert cancellations and all(cancelled in chain for cancelled in cancellations)
            else:
                assert any(isinstance(error, asyncio.CancelledError) and error.args == ("actual SDK retirement observer withdrawn",) for error in chain)
                if mode == "caused-closer-cancel":
                    assert caused_cancellations and ordinary_causes
                    assert all(error in chain for error in (*caused_cancellations, *ordinary_causes))
    finally:
        loop.set_task_factory(factory)
        release.set()
        await await_owned_task(operation)
        if manager_type is not None and original_exit is not None:
            monkeypatch.setattr(manager_type, "__aexit__", original_exit)
        for coroutine in refused:
            coroutine.close()
        for stream in streams:
            await stream.close()
        await client.close()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)
