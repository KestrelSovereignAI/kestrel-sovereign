"""Real SDK retries against localhost must not consume original PG control."""
import asyncio
import inspect
import json

import anthropic
import httpx
import httpx2
import openai
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome, native_execution_response_owner,
)
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("close_refusal", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize(("provider", "boundary"), [
    ("openai", "explicit-owner"), ("anthropic", "explicit-owner"),
    ("openai", "raw-adapter"), ("anthropic", "raw-adapter"),
    ("openai", "chat"), ("openai", "embedding"), ("openai", "catalog"),
    ("openai", "batch-poll"), ("openai", "file-get"),
    ("anthropic", "count"), ("anthropic", "output-ceiling"), ("anthropic", "chat"),
])
async def test_sdk_retry_cannot_consume_original_native_observer_failure(native_pg, monkeypatch, provider, boundary, source_cancel, independent, close_refusal):
    backend, observer, schema = native_pg
    requests, peers, originals, graphs, debts = [], set(), [], [], []
    owner = None
    responses, refused = [], []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal = RuntimeError("actual native SDK HTTPX observer cleanup admission refused")
    http_family = httpx2 if provider == "anthropic" else httpx
    native_close = http_family.Response.aclose

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if (not refused and getattr(operation, "cr_code", None) is native_close.__code__
                and any(operation.cr_frame.f_locals.get("self") is response for response in responses)):
            refused.append((coroutine, operation))
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    async def peer(reader, writer):
        peers.add(asyncio.current_task())
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            body = json.dumps({"id": "offline", "object": "model", "created": 0,
                               "owned_by": "localhost", "type": "model",
                               "display_name": "offline", "created_at": "2026-10-17T00:00:00Z"}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n" +
                         f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(asyncio.current_task())

    async def original_native_failure():
        with monkeypatch.context() as ack:
            receipts = lose_native_ack(ack, boundary="commit", original=OSError("actual SDK observer COMMIT ACK lost"))
            try:
                with bind_execution_custody(GenerationFence()):
                    await backend.execute("INSERT INTO effects VALUES (222, 'original SDK observer effect')")
            except BaseException as error:
                assert receipts == [True] and execution_commit_outcome(error) == "unknown"
                return error
        pytest.fail("actual native COMMIT did not lose its acknowledgement")

    # In the independent variant the native receipt exists BEFORE the SDK
    # call, in its own actual operation/scope. Reporting that original receipt
    # from a response observer must not rely on revoking the SDK caller's
    # otherwise healthy generation as a proxy for terminal control.
    prior = None
    if independent:
        prior_task = create_owned_task(original_native_failure(), retain_originals=True)
        prior_outcome = await await_owned_task(prior_task)
        assert prior_outcome.error is None
        prior = prior_outcome.result

    async def observed_response(response):
        if originals:
            return
        responses.append(response)
        error = prior if prior is not None else await original_native_failure()
        if source_cancel:
            asyncio.current_task().cancel("actual native SDK observer withdrew after COMMIT ACK loss")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                cancellation.__cause__ = error
                error = cancellation
        originals.append(error)
        graphs.append(tuple(_explicit_control_chain(error)))
        debts.append(tuple(execution_cleanup_debts(error)))
        raise error

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    http = http_family.AsyncClient(event_hooks={"response": [observed_response]})
    sdk = openai.AsyncOpenAI if provider == "openai" else anthropic.AsyncAnthropic
    client = sdk(api_key="localhost-only-not-a-credential", base_url=base, http_client=http, max_retries=1)
    adapter = OpenAIAdapter() if provider == "openai" else AnthropicAdapter()

    async def run():
        with bind_execution_custody(GenerationFence()):
            if boundary == "explicit-owner":
                async with native_execution_response_owner(http):
                    return await client.models.retrieve("offline")
            if boundary == "chat":
                return await adapter.get_response(client, model="offline", messages=[{"role": "user", "content": "offline"}])
            if boundary == "embedding":
                return await adapter.aembed(client, "offline", model="offline")
            if boundary == "catalog":
                return await adapter.list_models(client)
            if boundary == "batch-poll":
                return await adapter.batch_poll(client, "offline")
            if boundary == "file-get":
                return await adapter.file_get(client, "offline")
            if boundary == "count":
                return await adapter.count_tokens(client, "offline", [{"role": "user", "content": "offline"}])
            if boundary == "output-ceiling":
                return await adapter._model_output_ceiling(client, "offline")
            payload = {"model" if provider == "openai" else "model_id": "offline"}
            return await adapter.raw_request(client, "models.retrieve", payload)

    try:
        if close_refusal:
            loop.set_task_factory(factory)
        owner = create_owned_task(run(), retain_originals=True)
        outcome = await await_owned_task(owner)
        loop.set_task_factory(previous)
        assert len(originals) == 1
        assert len(requests) == 1, "SDK retried after original native UNKNOWN without any new authority"
        assert originals[0] in tuple(_explicit_control_chain(outcome.error)), "SDK normalization consumed original native control"
        if close_refusal:
            assert refused and refusal in tuple(_explicit_control_chain(outcome.error))
            assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
            assert any(resource is responses[0] and task is None for resource, task in execution_cleanup_debts(outcome.error)), "SDK retry lost refused native HTTP response resource"
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert tuple(_explicit_control_chain(originals[0])) == graphs[0]
        assert tuple(execution_cleanup_debts(originals[0])) == debts[0]
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=222') == 1
        # Retention is per operation, not sticky client-wide poison. A fresh
        # caller scope can still use this SAME shared SDK/transport normally.
        with bind_execution_custody(GenerationFence()):
            async with native_execution_response_owner(http):
                healthy = await client.models.retrieve("offline")
        assert healthy.id == "offline" and len(requests) == 2
        assert tuple(_explicit_control_chain(originals[0])) == graphs[0]
        assert tuple(execution_cleanup_debts(originals[0])) == debts[0]
    finally:
        loop.set_task_factory(previous)
        if owner is not None:
            await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        for response in responses:
            await native_close(response)
        await client.close()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)
