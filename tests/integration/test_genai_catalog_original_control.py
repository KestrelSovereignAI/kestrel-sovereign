"""Actual GenAI catalog pagination cannot consume independent native control.

Only a localhost SDK protocol peer and disposable PostgreSQL are used. Catalog
records are inputs, not authenticated Vertex/Gemini model capability acceptance.
Private-client lifetime retirement is a separate finding, not proved here.
"""
import asyncio
import json

import httpx
import pytest
from google import genai
from google.genai import types

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.llm.genai_custody import close_genai_client
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "vertex"])
@pytest.mark.parametrize("door", ["supplied", "private", "service"])
@pytest.mark.parametrize("late_page", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_genai_catalog_keeps_original_control_across_sdk_and_later_pages(
    native_pg, tmp_path, monkeypatch, provider, door, late_page, source_cancel,
):
    backend, observer, schema = native_pg
    async def native_original():
        with bind_execution_custody(GenerationFence()), monkeypatch.context() as fault:
            receipts = lose_native_ack(fault, boundary="commit", original=OSError("actual earlier catalog COMMIT ACK lost"))
            try:
                await backend.execute("INSERT INTO effects VALUES (228, 'original GenAI catalog observer')")
            except BaseException as error:
                assert receipts == [True]
                if source_cancel:
                    asyncio.current_task().cancel("actual original catalog observer withdrew")
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as cancellation:
                        cancellation.__cause__ = error
                        error = cancellation
                return error
        pytest.fail("native catalog fixture did not produce lost COMMIT ACK")
    source = await await_owned_task(create_owned_task(native_original(), retain_originals=True))
    assert source.error is None and source.result is not None
    original = source.result
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    peers, requests, carriers, clients = set(), [], [], []
    async def peer(reader, writer):
        peers.add(asyncio.current_task())
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            request = head.split(b"\r\n", 1)[0]
            requests.append(request)
            next_page = b"pageToken=" in request
            payload = {"models": [{"name": "models/offline-next" if next_page else "models/offline",
                                    "displayName": "Local protocol record", "inputTokenLimit": 4096}]}
            if not next_page:
                payload["nextPageToken"] = "native-next"
            body = json.dumps(payload).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n" +
                         f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            peers.discard(asyncio.current_task())
    async def observed_response(response):
        if carriers or (late_page and len(requests) < 2):
            return
        carrier = httpx.ConnectError("original GenAI response observer failed", request=response.request)
        carrier.__cause__ = original
        carriers.append(carrier)
        raise carrier
    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    http = httpx.AsyncClient(event_hooks={"response": [observed_response]}, trust_env=False)
    actual_client = genai.Client
    class LocalGenaiClient(actual_client):
        def __init__(self, *args, **kwargs):
            kwargs["http_options"] = types.HttpOptions(
                base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                api_version="v1beta", httpx_async_client=http,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            super().__init__(*args, **kwargs)
            clients.append(self)
    monkeypatch.setattr(genai, "Client", LocalGenaiClient)
    monkeypatch.setenv("GOOGLE_API_KEY", "inert-loopback-catalog-not-provider-credential")
    monkeypatch.setattr("kestrel_sovereign.llm.service.ProviderRegistry.initialize_providers", lambda self: [])
    service = LLMService(agent_data_dir=tmp_path)
    adapter = GoogleAdapter() if provider == "google" else VertexAIAdapter(project_id="offline-protocol")
    client = None if door == "private" else LocalGenaiClient(api_key="inert-loopback-catalog-not-provider-credential")
    if door == "service":
        service.providers = [{"adapter": adapter, "client": client}]
    async def invoke():
        with bind_execution_custody(GenerationFence()):
            if door == "service":
                return await service._safe_list_models(provider, adapter, client)
            return await adapter.list_models(client)
    owner = None
    try:
        owner = create_owned_task(invoke(), retain_originals=True)
        result = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(result.error))
        assert len(carriers) == 1 and carriers[0] in chain and original in chain, "GenAI catalog normalized original native control into availability"
        assert execution_commit_outcome(result.error) == "unknown"
        assert len(requests) == (2 if late_page else 1), "GenAI catalog fetched another page after original control"
        if door != "private":
            assert clients == [client], "catalog ignored original supplied route client"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=228') == 1
        before = len(requests)
        healthy = await invoke()
        assert len(healthy) == 2 and len(requests) == before + 2, "fresh GenAI catalog operation inherited another scope's terminal control"
    finally:
        if owner is not None:
            await await_owned_task(owner)
        await service.close()
        for item in clients:
            await close_genai_client(item)
        await http.aclose()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)
