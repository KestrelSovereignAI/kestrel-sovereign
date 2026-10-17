"""Actual localhost SDK pools have an explicit owning lifetime; no paid I/O."""
import asyncio
import json

import httpx
import openai
import pytest
from google import genai
from google.genai import types

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.llm.genai_custody import close_genai_client, prepare_genai_client
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["response", "stream", "tools", "google", "vertex"])
@pytest.mark.parametrize("borrowed", [False, True])
async def test_actual_private_sdk_is_retired_but_supplied_route_is_borrowed(monkeypatch, door, borrowed):
    peers, clients, requests = set(), [], []
    async def peer(reader, writer):
        peers.add(asyncio.current_task())
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            length = next((int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")), 0)
            payload = json.loads(await reader.readexactly(length)) if length else {}
            requests.append(head.split(b"\r\n", 1)[0])
            if door in ("google", "vertex"):
                body = json.dumps({"models": [{"name": "models/offline", "displayName": "Offline"}]}).encode()
                kind = b"application/json"
            elif payload.get("stream"):
                body = (b'data: {"id":"offline","object":"chat.completion.chunk","created":1,"model":"offline",'
                        b'"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                        b'data: {"id":"offline","object":"chat.completion.chunk","created":1,"model":"offline",'
                        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                        b'data: [DONE]\n\n')
                kind = b"text/event-stream"
            else:
                body = json.dumps({"id": "offline", "object": "chat.completion", "created": 1, "model": "offline",
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                "finish_reason": "stop"}]}).encode()
                kind = b"application/json"
            writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: " + kind +
                         f"\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(asyncio.current_task())
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    actual_openai, actual_genai = openai.AsyncOpenAI, genai.Client
    class LocalOpenAI(actual_openai):
        def __init__(self, *args, **kwargs):
            kwargs.update(base_url=base, api_key="inert-localhost-only", max_retries=0)
            super().__init__(*args, **kwargs)
            clients.append(self)
    class LocalGenai(actual_genai):
        def __init__(self, *args, **kwargs):
            kwargs["http_options"] = types.HttpOptions(base_url=base, api_version="v1beta",
                                                       retry_options=types.HttpRetryOptions(attempts=1))
            super().__init__(*args, **kwargs)
            clients.append(self)
    monkeypatch.setattr(openai, "AsyncOpenAI", LocalOpenAI)
    monkeypatch.setattr(genai, "Client", LocalGenai)
    monkeypatch.setenv("GOOGLE_API_KEY", "inert-localhost-only")
    monkeypatch.setenv("OPENROUTER_API_KEY", "inert-localhost-only")
    google = door in ("google", "vertex")
    adapter = GoogleAdapter() if door == "google" else VertexAIAdapter(project_id="offline") if door == "vertex" else OpenRouterAdapter()
    supplied = (LocalGenai(api_key="inert-localhost-only") if google else LocalOpenAI()) if borrowed else None
    try:
        if google:
            assert len(await adapter.list_models(supplied)) == 1
        elif door == "response":
            assert (await adapter.get_response(supplied, "offline", [{"role": "user", "content": "offline"}])).content == "ok"
        else:
            method = adapter.get_streaming_response if door == "stream" else adapter.get_streaming_response_with_tools
            stream = method(supplied, "offline", [{"role": "user", "content": "offline"}])
            try:
                assert [item async for item in stream]
            finally:
                await stream.aclose()
        assert len(requests) == 1 and len(clients) == 1
        client = clients[0]
        http = client._api_client._async_httpx_client if google else client._client
        if door == "vertex" and not borrowed:
            assert not http.is_closed, "cached Vertex client must remain reusable before adapter shutdown"
            await adapter.aclose()
            await adapter.aclose()
        assert http.is_closed is not borrowed, "private SDK pool leaked or borrowed route was closed"
    finally:
        for client in clients:
            if google:
                await close_genai_client(client)
            else:
                await client.close()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_actual_genai_pool_close_is_one_original_owner_across_withdrawal(native_pg, monkeypatch, source_cancel):
    backend, observer, schema = native_pg
    async def source_control():
        with bind_execution_custody(GenerationFence()), monkeypatch.context() as fault:
            receipts = lose_native_ack(fault, boundary="commit", original=OSError("original private-pool COMMIT ACK lost"))
            try:
                await backend.execute("INSERT INTO effects VALUES (229, 'private pool close')")
            except BaseException as error:
                assert receipts == [True]
                if source_cancel:
                    asyncio.current_task().cancel("original private-pool source cancelled")
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as cancellation:
                        cancellation.__cause__ = error
                        error = cancellation
                return error
        pytest.fail("lost-ACK fixture did not run")
    source = await await_owned_task(create_owned_task(source_control(), retain_originals=True))
    original = source.result
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    client = genai.Client(api_key="inert-no-requests")
    prepare_genai_client(client)
    http = client._kestrel_genai_httpx_owned
    actual_close = httpx.AsyncClient.aclose
    entered, release = asyncio.Event(), asyncio.Event()
    closers = []
    async def held_close(self):
        if self is not http:
            return await actual_close(self)
        closers.append(asyncio.current_task())
        await actual_close(self)
        entered.set()
        await release.wait()
        raise original
    monkeypatch.setattr(httpx.AsyncClient, "aclose", held_close)
    owner = create_owned_task(close_genai_client(client), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual private-close observer withdrew")
        await asyncio.sleep(0)
        assert not owner.done(), "private pool close returned before original physical retirement"
        release.set()
        first = await await_owned_task(owner)
        again = await await_owned_task(create_owned_task(close_genai_client(client), retain_originals=True))
        assert len(closers) == 1 and closers[0].done(), "repeated private close resubmitted the physical SDK pool closer"
        for outcome in (first, again):
            assert original in tuple(_explicit_control_chain(outcome.error))
            assert execution_commit_outcome(outcome.error) == "unknown"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
        assert http.is_closed and await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=229') == 1
    finally:
        release.set()
        await await_owned_task(owner)
        # The intentionally failed original close remains debt; cleanup does
        # not ask it to retry a second native effect.
        await actual_close(http)
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_control", [False, True])
async def test_genai_before_task_refusal_preserves_actual_catalog_withdrawal(native_pg, monkeypatch, source_control):
    from kestrel_sovereign.llm.genai_custody import _close_genai_layers
    from tests.integration.test_maintenance_original_retirement import _original_control
    native = await _original_control(native_pg[0], monkeypatch) if source_control else None
    entered = asyncio.Event()
    peers, clients, originals, refused = set(), [], [], []
    async def peer(reader, writer):
        peers.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            entered.set()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(asyncio.current_task())
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    actual = genai.Client
    class LocalGenai(actual):
        def __init__(self, *args, **kwargs):
            kwargs["http_options"] = types.HttpOptions(
                base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", api_version="v1beta",
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            super().__init__(*args, **kwargs)
            clients.append(self)
            request = self._api_client._async_request
            async def observed(*args, **kwargs):
                try:
                    return await request(*args, **kwargs)
                except asyncio.CancelledError as error:
                    if native is not None:
                        error.__cause__ = native
                    originals.append((error, tuple(_explicit_control_chain(error)), tuple(execution_cleanup_debts(error))))
                    raise
            self._api_client._async_request = observed
    monkeypatch.setattr(genai, "Client", LocalGenai)
    monkeypatch.setenv("GOOGLE_API_KEY", "inert-localhost-only")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal = RuntimeError("original private GenAI closer allocation refused")
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is _close_genai_layers.__code__:
            refused.append((coroutine, operation))
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = create_owned_task(GoogleAdapter().list_models(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        loop.set_task_factory(factory)
        owner.cancel("actual catalog socket acquisition withdrew")
        result = await await_owned_task(owner)
        assert len(originals) == 1 and len(refused) == 1
        opening, graph, debts = originals[0]
        chain = tuple(_explicit_control_chain(result.error))
        assert opening in chain, "GenAI close admission erased actual opening catalog cancellation"
        assert refusal in chain
        assert any(resource is clients[0] and task is None for resource, task in execution_cleanup_debts(result.error))
        assert tuple(_explicit_control_chain(opening)) == graph
        assert tuple(execution_cleanup_debts(opening)) == debts
        if source_control:
            assert native in chain and execution_commit_outcome(result.error) == "unknown"
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for client in clients:
            await _close_genai_layers(client)
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
async def test_genai_close_publication_failure_still_joins_original_pool(monkeypatch, after):
    from kestrel_sovereign.llm.genai_custody import _close_genai_layers
    client = genai.Client(api_key="inert-no-requests")
    prepare_genai_client(client)
    http = client._kestrel_genai_httpx_owned
    entered, release = asyncio.Event(), asyncio.Event()
    actual_close = httpx.AsyncClient.aclose
    original = RuntimeError("actual original SDK closer callback publication failed")
    closers, accepted = [], []
    async def held_close(self):
        if self is http:
            closers.append(asyncio.current_task())
            await actual_close(self)
            entered.set()
            await release.wait()
        else:
            await actual_close(self)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class OriginalTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *, context=None):
            if self.failed or getattr(callback, "__name__", None) != "completed":
                return super().add_done_callback(callback, context=context)
            self.failed = True
            if after:
                super().add_done_callback(callback, context=context)
            raise original
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is _close_genai_layers.__code__:
            task = OriginalTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", held_close)
    loop.set_task_factory(factory)
    owner = create_owned_task(close_genai_client(client), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not owner.done(), "SDK close publication failure released an accepted private pool owner"
        release.set()
        outcome = await await_owned_task(owner)
        assert original in tuple(_explicit_control_chain(outcome.error))
        again = await await_owned_task(create_owned_task(close_genai_client(client), retain_originals=True))
        assert original in tuple(_explicit_control_chain(again.error))
        assert len(closers) == 1 and all(task.done() for task in accepted) and http.is_closed
    finally:
        release.set()
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        await actual_close(http)
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
async def test_genai_canonical_task_publication_retains_unstarted_native_debt(monkeypatch, after):
    from kestrel_sovereign.llm.genai_custody import _close_genai_layers
    client = genai.Client(api_key="inert-no-requests")
    prepare_genai_client(client)
    http = client._kestrel_genai_httpx_owned
    original = RuntimeError("canonical SDK close receipt publication failed")
    accepted = []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class OriginalTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *, context=None):
            if self.failed or getattr(callback, "__name__", None) != "close_unstarted":
                return super().add_done_callback(callback, context=context)
            self.failed = True
            if after:
                super().add_done_callback(callback, context=context)
            raise original
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is _close_genai_layers.__code__:
            task = OriginalTask(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    try:
        first = await await_owned_task(create_owned_task(close_genai_client(client), retain_originals=True))
        again = await await_owned_task(create_owned_task(close_genai_client(client), retain_originals=True))
        assert len(accepted) == 1 and accepted[0].done(), "original accepted SDK close Task escaped its owner"
        assert not http.is_closed, "unstarted original retirement was mislabeled physical success"
        for outcome in (first, again):
            assert original in tuple(_explicit_control_chain(outcome.error))
            assert any(resource is client and closer is accepted[0] for resource, closer in execution_cleanup_debts(outcome.error))
        assert client._llm_native_retirement_resources["genai-native-client"] is client
    finally:
        loop.set_task_factory(previous)
        for task in accepted:
            await await_owned_task(task)
        # Explicit fixture reconciliation of the retained unstarted resource,
        # after fault removal; production never retries the failed owner.
        await http.aclose()
        client.close()
