"""Real Ollama catalog pools use canonical retirement, never paid inference."""
import asyncio
import inspect
import json

import httpx
import ollama
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.llm import ollama_adapter as adapter_module
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.fixture
async def catalog_peer(monkeypatch):
    clients, peers, requests = [], set(), []
    response_control = []
    async def response(response):
        for selected in response_control:
            if isinstance(selected, tuple):
                entered, release = selected
                entered.set()
                await release.wait()
            else:
                raise selected
    async def serve(reader, writer):
        peers.add(asyncio.current_task())
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            requests.append(head.split(b"\r\n", 1)[0])
            body = json.dumps({"models": []}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\n" +
                         f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(asyncio.current_task())
    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    actual_sdk, actual_close = ollama.AsyncClient, httpx.AsyncClient.aclose
    class LocalOllama(actual_sdk):
        def __init__(self, *args, **kwargs):
            kwargs.update(host=base, trust_env=False, event_hooks={"response": [response]})
            super().__init__(*args, **kwargs)
            clients.append(self)
    monkeypatch.setattr(ollama, "AsyncClient", LocalOllama)
    try:
        yield clients, requests, response_control, actual_close
    finally:
        for sdk in clients:
            await actual_close(sdk._client)
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("fault", ["before", "publication-before", "publication-after"])
async def test_ollama_catalog_original_outer_admission_keeps_pool_and_body(
    native_pg, catalog_peer, monkeypatch, embedding, source_cancel, fault,
):
    clients, requests, control, _ = catalog_peer
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    control.append(original)
    adapter = OllamaAdapter()
    refusal = RuntimeError("actual Ollama outer retirement admission/publication refused")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    accepted, refused = [], []
    class OriginalTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *, context=None):
            if self.failed or getattr(callback, "__name__", None) != "close_unstarted":
                return super().add_done_callback(callback, context=context)
            self.failed = True
            if fault == "publication-after":
                super().add_done_callback(callback, context=context)
            raise refusal
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        name = getattr(getattr(operation, "cr_code", None), "co_name", "")
        frame = getattr(operation, "cr_frame", None)
        direct_pool = bool(clients and frame is not None and frame.f_locals.get("self") is clients[0]._client and name == "aclose")
        if name == "close_native_execution_stream" or direct_pool:
            if fault == "before":
                refused.append((coroutine, operation))
                raise refusal
            task = OriginalTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    invoke = adapter.list_embedding_models if embedding else adapter.list_models
    owner = create_owned_task(invoke(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert len(clients) == len(requests) == 1
        native = clients[0]._client
        assert original in chain and refusal in chain, "Ollama catalog retirement lost independent body or admission evidence"
        assert not native.is_closed, "unstarted outer retirement spent native pool closure"
        expected = None if fault == "before" else accepted[0]
        assert any(resource is native and task is expected for resource, task in execution_cleanup_debts(outcome.error)), "Ollama outer admission lost exact unretired pool/Task debt"
        if expected is not None:
            assert expected.done(), "Ollama abandoned accepted original outer retirement Task"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "Ollama leaked refused outer retirement coroutine"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
        loop.set_task_factory(previous)
        repeated = await await_owned_task(create_owned_task(adapter.aclose(), retain_originals=True))
        assert refusal in tuple(_explicit_control_chain(repeated.error))
        assert not native.is_closed, "Ollama retried an originally refused/failed retirement"
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for pair in refused:
            for coro in pair:
                coro.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
@pytest.mark.parametrize("publication", ["normal", "before", "after"])
async def test_ollama_actual_pool_close_keeps_body_failure_and_observer_originals(
    native_pg, catalog_peer, monkeypatch, embedding, source_cancel, withdraw, publication,
):
    clients, requests, control, actual_close = catalog_peer
    original = await _original_control(native_pg[0], monkeypatch, source_cancel)
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    control.append(original)
    adapter = OllamaAdapter()
    close_error = OSError("actual original Ollama pool-close observer failed")
    close_graph = tuple(_explicit_control_chain(close_error))
    publication_error = RuntimeError("actual Ollama completed-close publication refused")
    entered, release = asyncio.Event(), asyncio.Event()
    closers, withdrawals = [], []
    actual_join = adapter_module.await_owned_task
    async def close(native):
        await actual_close(native)
        if clients and native is clients[0]._client:
            closers.append(asyncio.current_task())
            entered.set()
            await release.wait()
            raise close_error
    async def join(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    monkeypatch.setattr(adapter_module, "await_owned_task", join)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class OriginalTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *, context=None):
            if self.failed or getattr(callback, "__name__", None) != "completed":
                return super().add_done_callback(callback, context=context)
            self.failed = True
            if publication == "after":
                super().add_done_callback(callback, context=context)
            raise publication_error
    def factory(loop, coroutine, **kwargs):
        if publication != "normal" and getattr(_actual_operation(coroutine), "cr_code", None) is close.__code__:
            return OriginalTask(coroutine, loop=loop, **kwargs)
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    invoke = adapter.list_embedding_models if embedding else adapter.list_models
    owner = create_owned_task(invoke(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if withdraw:
            owner.cancel("actual Ollama catalog retirement observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done()
        release.set()
        outcome = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain and close_error in chain, "Ollama catalog dropped independent body or pool-close original"
        if publication != "normal":
            assert publication_error in chain, "Ollama lost completed-close publication evidence"
        assert any(resource is clients[0]._client and task is not None and task.done()
                   for resource, task in execution_cleanup_debts(outcome.error)), "Ollama direct harvester lost original failed pool/Task debt"
        if withdraw:
            assert withdrawals and all(error in chain for error in withdrawals), "Ollama catalog dropped actual independent observer cancellation"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts, "Ollama catalog appended private close debt to shared original"
        assert tuple(_explicit_control_chain(close_error)) == close_graph
        repeated = await await_owned_task(create_owned_task(adapter.aclose(), retain_originals=True))
        assert close_error in tuple(_explicit_control_chain(repeated.error))
        assert len(closers) == 1 and closers[0].done(), "Ollama repeated original native pool close"
        assert clients[0]._client.is_closed and len(requests) == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding", [False, True])
@pytest.mark.parametrize("borrowed", [False, True])
async def test_ollama_healthy_discovery_closes_only_private_pool(catalog_peer, embedding, borrowed):
    clients, requests, _, _ = catalog_peer
    adapter = OllamaAdapter()
    supplied = ollama.AsyncClient() if borrowed else None
    invoke = adapter.list_embedding_models if embedding else adapter.list_models
    try:
        assert await invoke(supplied) == []
        assert len(clients) == len(requests) == 1
        assert clients[0]._client.is_closed is not borrowed
        await adapter.aclose()
        assert clients[0]._client.is_closed is not borrowed
    finally:
        await adapter.aclose()



@pytest.mark.asyncio
async def test_ollama_shutdown_keeps_borrower_withdrawal_and_both_original_closers(
    native_pg, catalog_peer, monkeypatch,
):
    clients, requests, control, actual_close = catalog_peer
    original = await _original_control(native_pg[0], monkeypatch)
    other = OSError("second original Ollama native close failed")
    errors = (original, other)
    graphs = tuple(tuple(_explicit_control_chain(error)) for error in errors)
    debts = tuple(tuple(execution_cleanup_debts(error)) for error in errors)
    entered, resume, both_closing, release = (asyncio.Event() for _ in range(4))
    control.append((entered, resume))
    adapter = OllamaAdapter()
    closers, withdrawals = [], []
    actual_join = adapter_module.await_owned_task
    async def join(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    async def close(native):
        await actual_close(native)
        selected = next((index for index, sdk in enumerate(clients) if sdk._client is native), None)
        if selected is not None:
            closers.append((native, asyncio.current_task()))
            if len(closers) == 2:
                both_closing.set()
            await release.wait()
            raise errors[selected]
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close)
    monkeypatch.setattr(adapter_module, "await_owned_task", join)
    owners = [create_owned_task(method(), retain_originals=True)
              for method in (adapter.list_models, adapter.list_embedding_models)]
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert len(adapter._discovery_borrowers) == 2
        closing = create_owned_task(adapter.aclose(), retain_originals=True)
        await asyncio.sleep(0)
        closing.cancel("actual Ollama shutdown borrower observer withdrew")
        await asyncio.sleep(0)
        assert not closing.done()
        resume.set()
        await asyncio.wait_for(both_closing.wait(), 5)
        release.set()
        outcome = await await_owned_task(closing)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert all(error in chain for error in errors), "Ollama shutdown lost an independent original closer"
        assert withdrawals and all(error in chain for error in withdrawals), "Ollama shutdown dropped actual borrower withdrawal"
        assert len(closers) == 2 and all(task.done() for _, task in closers)
        assert len(requests) == 2 and all(sdk._client.is_closed for sdk in clients)
        repeated = await await_owned_task(create_owned_task(adapter.aclose(), retain_originals=True))
        assert all(error in tuple(_explicit_control_chain(repeated.error)) for error in errors)
        assert len(closers) == 2
        for error, graph, debt in zip(errors, graphs, debts):
            assert tuple(_explicit_control_chain(error)) == graph
            assert tuple(execution_cleanup_debts(error)) == debt
    finally:
        resume.set()
        release.set()
        for owner in owners:
            await await_owned_task(owner)
        if closing is not None:
            await await_owned_task(closing)
