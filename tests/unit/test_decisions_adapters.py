"""OpenRouter and Ollama decision routes (#3424): wire dialect, discovery,
error typing, and Ollama's chat-listing exclusion. HTTP and the Ollama client
are faked; nothing leaves the process."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
import pytest

from kestrel_sdk.llm.decisions import (
    DecisionProtocolError,
    DecisionRequest,
    DecisionTransportError,
    NoulQuestion,
    validate_decision_request,
)

import kestrel_sovereign.llm.decisions.http as decision_http
import kestrel_sovereign.llm.ollama_adapter as ollama_mod
import kestrel_sovereign.llm.openrouter_adapter as openrouter_mod
from kestrel_sovereign.llm.decisions.http import DecisionHTTPError
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter, _num_ctx
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter

SNAPSHOT = validate_decision_request(
    DecisionRequest(state={"text": "hello"}, questions={"greets": NoulQuestion(instructions="A greeting?")})
)


class _FakeResponse:
    def __init__(self, status: int, payload: Any = None, text: Optional[str] = None):
        self.status_code = status
        self._payload = payload
        self._text = text

    def json(self):
        if self._text is not None:
            return json.loads(self._text)
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]


class _FakeHTTP:
    """Stands in for ``httpx.AsyncClient``; records every request."""

    calls: List[Dict[str, Any]] = []
    response: Any = None
    error: Optional[BaseException] = None

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, *, content, headers):
        _FakeHTTP.calls.append({"method": "POST", "url": url, "body": json.loads(content),
                                "headers": headers, "timeout": self.kwargs.get("timeout")})
        if _FakeHTTP.error is not None:
            raise _FakeHTTP.error
        return _FakeHTTP.response

    async def get(self, url, *, params=None, headers=None, timeout=None):
        _FakeHTTP.calls.append({"method": "GET", "url": url, "params": params, "headers": headers})
        return _FakeHTTP.response


@pytest.fixture(autouse=True)
def fake_http(monkeypatch):
    _FakeHTTP.calls = []
    _FakeHTTP.response = None
    _FakeHTTP.error = None
    monkeypatch.setattr(decision_http.httpx, "AsyncClient", _FakeHTTP)
    monkeypatch.setattr(openrouter_mod.httpx, "AsyncClient", _FakeHTTP)
    return _FakeHTTP


def _openrouter(monkeypatch) -> OpenRouterAdapter:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    adapter = OpenRouterAdapter()
    adapter.base_url = "https://openrouter.example/api/v1"
    return adapter


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_posts_the_canonical_systemone_request(monkeypatch, fake_http) -> None:
    adapter = _openrouter(monkeypatch)
    fake_http.response = _FakeResponse(200, {"answers": {"greets": {"type": "noul", "noul": 0.9}}})

    body = await adapter.adecide(None, "typesafe/jev-1.13", SNAPSHOT, timeout=7.5)

    [call] = fake_http.calls
    assert call["url"] == "https://openrouter.example/api/v1/systemone"
    assert call["body"] == {"model": "typesafe/jev-1.13", **json.loads(SNAPSHOT.canonical_json)}
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["timeout"] == 7.5
    assert body["answers"]["greets"]["noul"] == 0.9


@pytest.mark.asyncio
async def test_openrouter_discovery_filters_the_decisions_modality(monkeypatch, fake_http) -> None:
    adapter = _openrouter(monkeypatch)
    fake_http.response = _FakeResponse(200, {"data": [
        {"id": "typesafe/jev-1.13", "context_length": 32000, "created": 1758000000,
         "architecture": {"modality": "text->decisions"}},
        {"id": "respan/span-01", "context_length": 0, "architecture": {"modality": "text->decisions"}},
        {"id": "some/chat-model", "context_length": 128000, "architecture": {"modality": "text->text"}},
    ]})

    models = await adapter.list_decision_models(None)

    [call] = fake_http.calls
    assert call["params"] == {"output_modalities": "decisions"}
    assert [(m.id, m.context_limit) for m in models] == [
        ("typesafe/jev-1.13", 32000), ("respan/span-01", None)
    ]
    assert adapter.provider_capabilities().supports_decisions is True


@pytest.mark.asyncio
async def test_openrouter_without_a_key_is_a_transport_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    adapter = OpenRouterAdapter()
    with pytest.raises(DecisionTransportError, match="OPENROUTER_API_KEY"):
        await adapter.adecide(None, "m", SNAPSHOT, timeout=1)
    assert await adapter.list_decision_models(None) == []


# ---------------------------------------------------------------------------
# Shared transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_errors_are_typed_and_never_echo_the_body(monkeypatch, fake_http) -> None:
    adapter = _openrouter(monkeypatch)
    fake_http.response = _FakeResponse(404, text='{"error": "model typesafe/x not found for state hello"}')
    with pytest.raises(DecisionHTTPError) as excinfo:
        await adapter.adecide(None, "typesafe/x", SNAPSHOT, timeout=1)
    assert excinfo.value.status_code == 404 and "hello" not in str(excinfo.value)

    fake_http.error = httpx.ConnectError("refused")
    with pytest.raises(DecisionTransportError, match="ConnectError"):
        await adapter.adecide(None, "m", SNAPSHOT, timeout=1)


@pytest.mark.asyncio
async def test_non_json_and_non_object_bodies_are_protocol_errors(monkeypatch, fake_http) -> None:
    adapter = _openrouter(monkeypatch)
    fake_http.response = _FakeResponse(200, text="<html>")
    with pytest.raises(DecisionProtocolError):
        await adapter.adecide(None, "m", SNAPSHOT, timeout=1)
    fake_http.response = _FakeResponse(200, payload=[1, 2])
    with pytest.raises(DecisionProtocolError):
        await adapter.adecide(None, "m", SNAPSHOT, timeout=1)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def _ollama_client(models: Dict[str, Dict[str, Any]], host="http://ollama.test:11434"):
    async def list_():
        return SimpleNamespace(models=[SimpleNamespace(model=name, size=1, modified_at="2026-10-01")
                                       for name in models])

    async def show(name):
        return SimpleNamespace(**{"template": "", "parameters": None, "capabilities": [], **models[name]})

    return SimpleNamespace(list=list_, show=show, _client=SimpleNamespace(base_url=host))


def test_num_ctx_is_read_from_parameters_not_the_base_model() -> None:
    assert _num_ctx("num_ctx 8194\nstop <x>") == 8194
    assert _num_ctx("stop <x>") is None
    assert _num_ctx(None) is None
    assert _num_ctx("num_ctx zero") is None


@pytest.mark.asyncio
async def test_ollama_posts_to_the_routes_daemon_with_a_timeout(fake_http) -> None:
    client = _ollama_client({})
    fake_http.response = _FakeResponse(200, {"answers": {}})
    await OllamaAdapter().adecide(client, "nimble", SNAPSHOT, timeout=3)
    [call] = fake_http.calls
    assert call["url"] == "http://ollama.test:11434/v1/systemone"
    assert call["body"]["model"] == "nimble" and call["timeout"] == 3


@pytest.mark.asyncio
async def test_ollama_too_old_for_systemone_is_a_404(fake_http) -> None:
    fake_http.response = _FakeResponse(404, payload={"error": "404 page not found"})
    with pytest.raises(DecisionHTTPError) as excinfo:
        await OllamaAdapter().adecide(_ollama_client({}), "nimble", SNAPSHOT, timeout=3)
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_ollama_discovery_uses_capabilities_and_served_context(monkeypatch) -> None:
    monkeypatch.setattr(ollama_mod, "OLLAMA_AVAILABLE", True)
    client = _ollama_client({
        "nimble:9b": {"capabilities": ["decision", "tools", "thinking"], "parameters": "num_ctx 8194"},
        "tev1:0.8b": {"capabilities": ["decision"], "parameters": "num_ctx 2050"},
        "qwen3.8:latest": {"capabilities": ["completion", "tools"], "parameters": "num_ctx 32768"},
        "decision-ish-name": {"capabilities": ["completion"]},
    })
    models = await OllamaAdapter().list_decision_models(client)
    assert [(m.id, m.context_limit, m.max_questions, m.max_options, m.parallel_questions)
            for m in models] == [
        ("nimble:9b", 8194, 64, 26, False),
        ("tev1:0.8b", 2050, 64, 26, False),
    ]
    assert all(m.max_request_bytes == 64 * 1024 for m in models)


@pytest.mark.asyncio
async def test_ollama_chat_listing_drops_decision_only_models(monkeypatch) -> None:
    monkeypatch.setattr(ollama_mod, "OLLAMA_AVAILABLE", True)
    client = _ollama_client({
        "tev1:0.8b": {"capabilities": ["decision"]},
        "both:latest": {"capabilities": ["decision", "completion"]},
        "qwen3.8:latest": {"capabilities": ["completion", "tools"]},
    })
    monkeypatch.setattr(ollama_mod.ollama, "AsyncClient", lambda *a, **k: client)
    listed = [m.id for m in await OllamaAdapter().list_models()]
    assert listed == ["both:latest", "qwen3.8:latest"]
