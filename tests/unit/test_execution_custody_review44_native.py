"""Actual SDK/HTTPX close and mTLS configuration; never network or credentials."""

import asyncio

import httpx
import httpx2
import pytest
from google import genai
from google.auth.credentials import AnonymousCredentials

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["anthropic", "openrouter"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("close_failure", [False, True])
async def test_temporary_client_close_cannot_erase_original_control(
    monkeypatch, adapter, kind, close_failure,
):
    native_http = httpx2 if adapter == "anthropic" else httpx
    original = receipt(kind)
    entered, release = asyncio.Event(), asyncio.Event()
    retired = []

    class Transport(native_http.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise original

        async def aclose(self):
            entered.set()
            await release.wait()
            retired.append(True)
            if close_failure:
                raise RuntimeError("ordinary native transport close failure")

    native_client = native_http.AsyncClient
    client = native_client(transport=Transport())
    monkeypatch.setattr(native_http, "AsyncClient", lambda *a, **k: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-transport-only")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-transport-only")
    instance = AnthropicAdapter() if adapter == "anthropic" else OpenRouterAdapter()
    operation = instance.list_models if adapter == "anthropic" else instance.list_embedding_models
    task = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done(), "temporary client's physical transport close escaped"
        release.set()
        outcome = await await_owned_task(task)
        assert retired == [True]
        assert carries(outcome.error, original), "cleanup erased the original native receipt"
    finally:
        release.set()
        await await_owned_task(task)
        await client.aclose()


@pytest.mark.asyncio
async def test_native_vertex_automatic_mtls_is_not_silently_disabled(monkeypatch):
    from google.genai import _api_client as native
    from kestrel_sovereign.llm.genai_custody import prepare_genai_client

    monkeypatch.setattr(native, "has_aiohttp", True)
    monkeypatch.setattr(native.mtls, "should_use_client_cert", lambda: True, raising=False)
    monkeypatch.setattr(native.mtls, "has_default_client_cert_source", lambda: True, raising=False)
    client = genai.Client(
        vertexai=True, project="offline-native", location="us-central1",
        credentials=AnonymousCredentials(),
    )
    try:
        assert client._api_client._use_google_auth_async()
        with pytest.raises(RuntimeError, match="automatic Vertex mTLS"):
            prepare_genai_client(client)
        assert client._api_client._http_options.httpx_async_client is None
        assert client._api_client._use_google_auth_async()
    finally:
        await client.aio.aclose()
        client.close()
