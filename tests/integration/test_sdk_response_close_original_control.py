"""Actual SDK-acquired HTTP response close keeps native PG originals.

Only a loopback SDK protocol peer and disposable PG schema are used. This is
response-ownership evidence, not authenticated provider or Agent acceptance.
"""
import asyncio
import inspect

import anthropic
import httpx2 as httpx
import openai
import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    NativeExecutionStreamOwner, _explicit_control_chain, bind_execution_custody,
    close_native_execution_stream, execution_cleanup_debts, execution_commit_outcome,
    retain_native_execution_http_client_close,
)
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("route", ["implicit", "inventory", "helper"])
@pytest.mark.parametrize("mode", ["admission", "observer", "native-failure"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_sdk_native_response_close_preserves_original_admission_and_observers(
    native_pg, monkeypatch, provider, route, mode, source_cancel,
):
    backend, observer, schema = native_pg
    originals = []
    async def original_native():
        with bind_execution_custody(GenerationFence()), monkeypatch.context() as ack:
            receipts = lose_native_ack(ack, boundary="commit", original=OSError("actual earlier SDK operation COMMIT ACK lost"))
            try:
                await backend.execute("INSERT INTO effects VALUES (224, 'original SDK close body')")
            except BaseException as error:
                assert receipts == [True]
                if source_cancel:
                    asyncio.current_task().cancel("actual original SDK body withdrawal")
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as cancellation:
                        cancellation.__cause__ = error
                        error = cancellation
                originals.append(error)
                raise error
    source = create_owned_task(original_native(), retain_originals=True)
    source_outcome = await await_owned_task(source)
    assert source_outcome.error is not None and len(originals) == 1
    original = originals[0]
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    entered, release, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
    peers, responses, close_tasks, withdrawals, close_sources, refused = set(), [], [], [], [], []
    close_graphs, close_debts = [], []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal = RuntimeError("actual original HTTP response native closer admission refused")
    native_close = httpx.Response.aclose
    actual_join = ownership.await_owned_task
    owner = sdk = response = manager = None

    async def peer(reader, writer):
        peers.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000000\r\n\r\n{")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            disconnected.set()
            peers.discard(asyncio.current_task())

    async def held_close(response):
        if any(response is selected for selected in responses):
            close_tasks.append(asyncio.current_task())
            await native_close(response)
            entered.set()
            await release.wait()
            if mode == "native-failure":
                with bind_execution_custody(GenerationFence()), monkeypatch.context() as ack:
                    committed = lose_native_ack(ack, boundary="commit", original=OSError("actual SDK response retirement COMMIT ACK lost"))
                    try:
                        await backend.execute("INSERT INTO effects VALUES (225, 'original SDK close tail')")
                    except BaseException as error:
                        assert committed == [True]
                        close_sources.append(error)
                        close_graphs.append(tuple(_explicit_control_chain(error)))
                        close_debts.append(tuple(execution_cleanup_debts(error)))
                        raise
        else:
            await native_close(response)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if (getattr(operation, "cr_code", None) is held_close.__code__
                and any(operation.cr_frame.f_locals.get("response") is item for item in responses)):
            refused.append((coroutine, operation))
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def observed_join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        # Inventory close has an outer accepted physical-close owner as well
        # as the response's native closer. Either finite join can observe C.
        if outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    async def run():
        try:
            raise original
        except BaseException:
            if route == "implicit":
                await response.aclose()
            elif route == "helper":
                await close_native_execution_stream(response)
            else:
                inventory = NativeExecutionStreamOwner()
                inventory.retain(response)
                await inventory.close(original)
            raise
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    http = httpx.AsyncClient(trust_env=False)
    try:
        monkeypatch.setattr(httpx.Response, "aclose", held_close)
        monkeypatch.setattr(ownership, "await_owned_task", observed_join)
        retain_native_execution_http_client_close(http)
        sdk_type = openai.AsyncOpenAI if provider == "openai" else anthropic.AsyncAnthropic
        sdk = sdk_type(api_key="inert-localhost-not-provider-credential",
                       base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                       http_client=http, max_retries=0)
        manager = sdk.models.with_streaming_response.retrieve("offline")
        acquired = await manager.__aenter__()
        response = acquired.http_response
        assert not response.is_closed
        responses.append(response)
        if mode == "admission":
            loop.set_task_factory(factory)
        owner = create_owned_task(run(), retain_originals=True)
        if mode != "admission":
            await asyncio.wait_for(entered.wait(), 5)
            if mode == "observer":
                owner.cancel("actual SDK response close observer withdrew")
            release.set()
        outcome = await await_owned_task(owner)
        loop.set_task_factory(previous)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain, "native HTTP close replaced original SDK body control"
        assert execution_commit_outcome(outcome.error) == "unknown"
        if mode == "admission":
            assert refusal in chain and refused
            assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "refused original native closer coroutine leaked"
            assert any(resource is response and task is None for resource, task in execution_cleanup_debts(outcome.error)), "refused native closer lost SAME acquired response"
            assert not response.is_closed
        elif mode == "observer":
            assert withdrawals and all(cancellation in chain for cancellation in withdrawals), "native close lost actual observer C beside original SDK control"
            assert response.is_closed and all(task.done() for task in close_tasks)
        else:
            assert close_sources and all(error in chain for error in close_sources), "native closer failure replaced or lost original body evidence"
            assert any(resource is response and task in close_tasks for resource, task in execution_cleanup_debts(outcome.error))
        assert tuple(_explicit_control_chain(original)) == graph, "native close mutated shared original source graph"
        assert tuple(execution_cleanup_debts(original)) == debts, "native close added private retirement debt to shared original"
        assert all(tuple(_explicit_control_chain(error)) == before for error, before in zip(close_sources, close_graphs))
        assert all(tuple(execution_cleanup_debts(error)) == before for error, before in zip(close_sources, close_debts)), "private native close debt mutated original retirement source"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=224') == 1
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=225') == (1 if mode == "native-failure" else 0)
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        if response is not None:
            # The measurement is over. Retire the exact native response
            # directly if its deliberately-refused admission never started.
            await native_close(response)
        if sdk is not None:
            await sdk.close()
        else:
            await http.aclose()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)
