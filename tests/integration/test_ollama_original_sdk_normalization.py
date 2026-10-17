"""Actual Ollama SDK normalization may not erase native PG originals.

Only a localhost protocol peer and disposable native PostgreSQL are used.
Responses are protocol inputs, not real inference/model capability acceptance.
"""
import asyncio
import json

import httpx
import ollama
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack


CASES = [(name, False) for name in (
    "embedding", "batch", "chat", "raw", "tool-capability", "embedding-capability",
    "catalog", "embedding-catalog", "decision-catalog", "inner-chat", "inner-embedding", "inner-decision",
)] + [(name, True) for name in ("catalog", "embedding-catalog", "decision-catalog", "inner-chat", "inner-embedding", "inner-decision")]


@pytest.mark.asyncio
@pytest.mark.parametrize(("boundary", "late_show"), CASES)
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_ollama_sdk_normalization_cannot_turn_original_control_into_absence(
    native_pg, monkeypatch, boundary, late_show, source_cancel,
):
    backend, observer, schema = native_pg
    peers, requests, carriers, originals, graphs, debts = set(), [], [], [], [], []
    owner = None
    response_body = {
        "models": [{"name": name, "model": name, "size": 1, "digest": "localhost",
                    "modified_at": "2026-10-17T00:00:00Z"} for name in ("offline", "unreached")],
        "model": "offline", "message": {"role": "assistant", "content": "offline"},
        "done": True, "done_reason": "stop", "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        "capabilities": ["embedding", "tools", "completion", "decision"],
        "parameters": "num_ctx 4096", "model_info": {"embedding_length": 2},
    }

    async def native_original():
        with bind_execution_custody(GenerationFence()), monkeypatch.context() as ack:
            receipts = lose_native_ack(ack, boundary="commit", original=OSError("actual earlier Ollama observer COMMIT ACK lost"))
            try:
                await backend.execute("INSERT INTO effects VALUES (226, 'original Ollama observer effect')")
            except BaseException as error:
                assert receipts == [True]
                if source_cancel:
                    asyncio.current_task().cancel("actual original Ollama observer withdrew")
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as cancellation:
                        cancellation.__cause__ = error
                        error = cancellation
                return error
        pytest.fail("actual native fixture did not produce original COMMIT control")

    original_task = create_owned_task(native_original(), retain_originals=True)
    original_outcome = await await_owned_task(original_task)
    assert original_outcome.error is None
    original = original_outcome.result
    originals.append(original)
    graphs.append(tuple(_explicit_control_chain(original)))
    debts.append(tuple(execution_cleanup_debts(original)))

    async def peer(reader, writer):
        peers.add(asyncio.current_task())
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            requests.append(head.split(b"\r\n", 1)[0])
            length = next((int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")), 0)
            if length:
                await reader.readexactly(length)
            body = json.dumps(response_body).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n" +
                         f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(asyncio.current_task())

    async def observed_response(response):
        if carriers or (late_show and response.request.url.path != "/api/show"):
            return
        carrier = httpx.ConnectError("original supported Ollama HTTP observer failed", request=response.request)
        carrier.__cause__ = original
        carriers.append(carrier)
        raise carrier

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    client = ollama.AsyncClient(
        host=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        event_hooks={"response": [observed_response]}, trust_env=False,
    )
    adapter = OllamaAdapter()

    async def invoke():
        if boundary == "embedding":
            return await adapter.aembed(client, "offline", model="offline")
        if boundary == "batch":
            return await adapter.aembed_batch(client, ["offline", "unreached"], model="offline")
        if boundary == "chat":
            return await adapter.get_response(client, model="offline", messages=[{"role": "user", "content": "offline"}])
        if boundary == "raw":
            return await adapter.raw_request(client, "show", {"model": "offline"})
        if boundary == "tool-capability":
            return await adapter._check_tool_support(client, "offline")
        if boundary == "embedding-capability":
            return await adapter._check_embedding_support(client, "offline")
        if boundary == "catalog":
            return await adapter.list_models(client)
        if boundary == "embedding-catalog":
            return await adapter.list_embedding_models(client)
        if boundary == "decision-catalog":
            return await adapter.list_decision_models(client)
        if boundary == "inner-chat":
            return await adapter._list_chat_models(client)
        if boundary == "inner-embedding":
            return await adapter._list_embedding_models(client)
        return await adapter._list_decision_models(client)

    async def run():
        # The source receipt belongs to an earlier independent scope. This
        # SDK caller's otherwise healthy scope must not be a proxy for it.
        with bind_execution_custody(GenerationFence()):
            return await invoke()
    try:
        owner = create_owned_task(run(), retain_originals=True)
        outcome = await await_owned_task(owner)
        assert len(carriers) == 1
        chain = tuple(_explicit_control_chain(outcome.error))
        assert carriers[0] in chain and original in chain, "Ollama SDK normalized original native control into ordinary absence/failure"
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert len(requests) == (2 if late_show else 1), "Ollama issued another native request after original control"
        assert tuple(_explicit_control_chain(original)) == graphs[0]
        assert tuple(execution_cleanup_debts(original)) == debts[0]
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=226') == 1
        before = len(requests)
        with bind_execution_custody(GenerationFence()):
            healthy = await invoke()
        assert healthy is not None and len(requests) > before, "shared Ollama client remained poisoned for an independent healthy operation"
        assert tuple(_explicit_control_chain(original)) == graphs[0]
        assert tuple(execution_cleanup_debts(original)) == debts[0]
    finally:
        if owner is not None:
            await await_owned_task(owner)
        await adapter.aclose()
        await client._client.aclose()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)
