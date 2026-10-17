"""Real SDK transports, never credentials/network: independent review43."""

import asyncio
import json

import anthropic
import httpx
import openai
import pytest
from google import genai

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    bind_execution_custody,
    execution_commit_outcome,
    execution_terminal_error,
    is_execution_control_error,
    retain_native_execution_http_client_close,
)
from kestrel_sovereign.llm.anthropic_adapter import (
    AnthropicAdapter,
    _anthropic_stream_with_retry,
)
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.provider_registry import ProviderRegistry
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
from tests.unit.test_execution_custody import Authority
from tests.unit.test_execution_custody_review30 import service_owner
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_execution_custody_review39_native import Body
from tests.unit.test_execution_custody_review40_native import EndingBody


class ReceiptBody(EndingBody):
    def __init__(self, data, original):
        super().__init__(data)
        self.original = original
        self.close_calls = 0

    async def aclose(self):
        self.close_calls += 1
        await super().aclose()
        if self.original is not None:
            raise self.original


def carries(error, original):
    seen, pending = set(), [error]
    while pending:
        current = pending.pop()
        if current is original:
            return True
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        pending.append(current.__cause__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stream_open", "ceiling", "tokens", "plan_models", "raw"])
@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.parametrize("kind", [None, "unknown", "cancel_unknown", "group"])
async def test_anthropic_owns_native_body_before_sdk_reads(operation, status, kind):
    import httpx2 as httpx
    original = receipt(kind) if kind else None
    payload = {"id": "offline", "display_name": "Offline", "max_tokens": 1024,
               "created_at": "2026-01-01T00:00:00Z"}
    if operation in {"tokens", "raw"}:
        payload = {"input_tokens": 1}
    if operation == "plan_models":
        payload = {"data": [payload], "has_more": False}
    if status == 503:
        payload = {"type": "error", "error": {"type": "overloaded_error", "message": "offline"}}
    body = ReceiptBody(json.dumps(payload).encode(), original)
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(status, headers={"content-type": "application/json"}, stream=body)
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = anthropic.AsyncAnthropic(api_key="offline-transport-only", http_client=http, max_retries=0)
    adapter = AnthropicAdapter()
    async def invoke():
        if operation == "stream_open":
            async with _anthropic_stream_with_retry(client, {"model": "offline", "max_tokens": 1, "messages": []}, retry_open=False) as stream:
                async for _ in stream:
                    pass
        elif operation == "ceiling":
            await adapter._model_output_ceiling(client, "offline")
        elif operation == "plan_models":
            await ClaudeMaxAdapter().list_models(client)
        elif operation == "raw":
            await adapter.raw_request(client, "messages.count_tokens", model="offline", messages=[])
        else:
            await adapter.count_tokens(client, "offline", [])
    task = asyncio.create_task(invoke())
    try:
        await asyncio.wait_for(body.close_entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done(), "cancelled SDK discarded a native close still running"
        body.release_close.set()
        result = await await_owned_task(task)
        terminal = execution_terminal_error(result.error, result.cancellation)
        if original:
            assert carries(terminal, original)
            assert execution_commit_outcome(terminal) == "unknown"
        else:
            assert isinstance(terminal, asyncio.CancelledError)
        assert body.closed and body.close_calls == 1
        assert len(requests) == 1
    finally:
        body.release_close.set()
        await await_owned_task(task)
        await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "vertex"])
@pytest.mark.parametrize("tools_contract", [False, True])
@pytest.mark.parametrize("kind", [None, "unknown", "cancel_unknown", "group"])
async def test_genai_joins_all_real_sdk_stream_layers(provider, tools_contract, kind):
    original = receipt(kind) if kind else None
    payload = {"candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]}}]}
    class StreamingBody(ReceiptBody):
        __aiter__ = Body.__aiter__
    body = StreamingBody(b"data: " + json.dumps(payload).encode() + b"\n\n", original)
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = genai.Client(api_key="offline-transport-only", http_options={"httpx_async_client": http})
    adapter = GoogleAdapter() if provider == "google" else VertexAIAdapter(project_id="offline")
    method = adapter.get_streaming_response_with_tools if tools_contract else adapter.get_streaming_response
    iterator = method(client=client, model="gemini-offline", messages=[{"role": "user", "parts": [{"text": "hello"}]}], max_tokens=1)
    task = None
    try:
        assert await asyncio.wait_for(anext(iterator), 2) == "hello"
        task = asyncio.create_task(iterator.aclose())
        await asyncio.wait_for(body.close_entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done(), "outer forwarding close escaped before native SDK retirement"
        body.release_close.set()
        result = await await_owned_task(task)
        terminal = execution_terminal_error(result.error, result.cancellation)
        if original:
            assert carries(terminal, original)
            assert execution_commit_outcome(terminal) == "unknown"
        else:
            assert isinstance(terminal, asyncio.CancelledError)
        assert body.closed and body.close_calls == 1
        assert len(requests) == 1
    finally:
        body.release_close.set()
        if task is not None:
            await await_owned_task(task)
        try:
            await iterator.aclose()
        except BaseException:
            pass
        await client.aio.aclose()
        client.close()
        await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [False, True])
async def test_service_retires_real_genai_async_transport_before_success(prepared):
    client = genai.Client(api_key="offline-transport-only")
    http = client._api_client._async_httpx_client
    if prepared:
        from kestrel_sovereign.llm.genai_custody import prepare_genai_client
        prepare_genai_client(client)
    service = service_owner()
    service.providers = [{"adapter": GoogleAdapter(), "client": client}]
    await service.close()
    assert http.is_closed, "sync-only close claimed retirement while async transport lived"
    assert not service.has_pending_close_owners()


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding", [False, True])
async def test_legacy_discovery_actual_openai_constructor_disables_sdk_retry(monkeypatch, embedding):
    import httpx2 as httpx
    native = openai.AsyncOpenAI
    clients = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [], "object": "list"}),
    ))
    def construct(**kwargs):
        client = native(http_client=http, **kwargs)
        clients.append(client)
        return client
    monkeypatch.setenv("OPENAI_API_KEY", "offline-transport-only")
    monkeypatch.setattr(openai, "AsyncOpenAI", construct)
    try:
        method = OpenAIAdapter().list_embedding_models if embedding else OpenAIAdapter().list_models
        assert await method(None) == []
        assert len(clients) == 1 and clients[0].max_retries == 0
    finally:
        await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "plan"])
async def test_registry_native_client_has_no_sdk_owned_retries(monkeypatch, provider):
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry._resolve_secret = lambda config, env_name, value_name: "offline-transport-only" if value_name == "api_key" else None
    if provider == "plan":
        from kestrel_sovereign.llm.anthropic_oauth import ClaudeOAuthTokenManager
        monkeypatch.setattr(ClaudeOAuthTokenManager, "from_sources", lambda **kwargs: None)
    cls = AnthropicAdapter if provider == "anthropic" else ClaudeMaxAdapter
    client, _ = registry._build_client_and_adapter("anthropic", "api", cls, {}, {})
    try:
        assert client.max_retries == 0, "SDK backoff may issue a request after Core authority expires"
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk", ["anthropic", "openai"])
async def test_even_external_sdk_retry_cannot_reuse_revoked_request_authority(sdk):
    import httpx2 as httpx
    requests = []
    scope = None
    def respond(request):
        requests.append(request)
        scope.revoke("retired during native SDK backoff")
        return httpx.Response(503, json={"error": {"type": "overloaded_error", "message": "offline"}}, headers={"retry-after-ms": "1"})
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    retain_native_execution_http_client_close(http)
    cls = anthropic.AsyncAnthropic if sdk == "anthropic" else openai.AsyncOpenAI
    client = cls(api_key="offline-transport-only", http_client=http, max_retries=1)
    try:
        with bind_execution_custody(Authority()) as scope:
            with pytest.raises(BaseException) as caught:
                await client.models.retrieve("offline")
        assert is_execution_control_error(caught.value)
        assert len(requests) == 1, "native retry spent retired authority"
        assert carries(caught.value, next(error for error in [caught.value, caught.value.__cause__] if isinstance(error, ExecutionAuthorityError)))
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_real_catalog_transport_control_is_not_empty_success(provider, kind):
    import httpx2

    native_http = httpx if provider == "ollama" else httpx2
    original = receipt(kind)
    body = ReceiptBody(json.dumps({"data": [], "models": []}).encode(), original)
    body.release_close.set()
    def respond(request):
        return native_http.Response(200, headers={"content-type": "application/json"}, stream=body)
    http = native_http.AsyncClient(base_url="http://127.0.0.1:11434", transport=native_http.MockTransport(respond))
    previous = None
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key="offline-transport-only", http_client=http, max_retries=0)
        method = OpenAIAdapter().list_models
    else:
        import ollama
        client = ollama.AsyncClient()
        previous, client._client = client._client, http
        method = OllamaAdapter().list_embedding_models
    try:
        with bind_execution_custody(Authority()):
            with pytest.raises(BaseException) as caught:
                await method(client)
        assert is_execution_control_error(caught.value) and carries(caught.value, original)
    finally:
        await http.aclose()
        if previous is not None:
            await previous.aclose()
