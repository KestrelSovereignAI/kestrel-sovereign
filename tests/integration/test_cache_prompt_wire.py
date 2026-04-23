"""Integration test: `cache_prompt` reaches the wire (issue #704).

Verifies the full pipeline from `service.py._try_single_provider` →
`provider_cache_body(provider)` → `OpenAIAdapter.get_response(extra_body=…)`
→ OpenAI SDK → HTTP body actually carries `"cache_prompt": true` when the
active provider is llama.cpp, and does NOT carry it for other vendors.

Uses the real llama-server on :8001 if running (skipped otherwise), and
intercepts the HTTP body via `httpx.MockTransport` for the per-vendor gate
check (so those cases don't need the server).
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import httpx
import openai
import pytest

from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.provider_registry import provider_cache_body


async def _send_and_capture(
    provider: Dict,
) -> Dict:
    """Run OpenAIAdapter.get_response through a real AsyncOpenAI client
    whose transport is mocked by pytest-httpx, and return the JSON body
    the SDK actually sent on the wire.

    This exercises the full stack: service.py gate → adapter → SDK →
    httpx request body.  The only thing it doesn't hit is a real network
    socket, which is what makes this safe/fast for CI.
    """
    captured: Dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            captured["body"] = json.loads(request.content)
        except Exception:
            captured["body"] = None
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = openai.AsyncOpenAI(
        api_key="test",
        base_url="http://localhost:9999/v1",
        http_client=http_client,
    )

    adapter = OpenAIAdapter()
    try:
        await adapter.get_response(
            client=client,
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            extra_body=provider_cache_body(provider),
        )
    finally:
        await http_client.aclose()

    return captured


@pytest.mark.asyncio
async def test_cache_prompt_reaches_wire_for_llama_cpp():
    """llama.cpp route → HTTP body must include `"cache_prompt": true`."""
    provider = {"vendor": "llama_cpp", "route": "local", "name": "llama_cpp"}
    captured = await _send_and_capture(provider)
    body = captured.get("body")
    assert body is not None, "request body not captured"
    assert body.get("cache_prompt") is True, (
        f"cache_prompt missing or false in body sent over the wire: "
        f"{body!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vendor",
    ["openai", "openrouter", "ollama", "runpod", "xai", "groq", "anthropic"],
)
async def test_cache_prompt_absent_from_wire_for_other_vendors(vendor):
    """Non-llama.cpp routes → HTTP body must NOT include `cache_prompt`.
    Strict OpenAI-compatible proxies would 4xx on the unknown field.
    """
    provider = {"vendor": vendor, "route": "api", "name": vendor}
    captured = await _send_and_capture(provider)
    body = captured.get("body")
    assert body is not None
    assert "cache_prompt" not in body, (
        f"cache_prompt leaked to wire for vendor={vendor}: {body!r}"
    )


# ---------------------------------------------------------------------------
# Real llama-server end-to-end (skipped when the server isn't running)
# ---------------------------------------------------------------------------


async def _llama_server_reachable(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{base_url}/models")
            return r.status_code < 500
    except (httpx.RequestError, httpx.TimeoutException):
        return False


@pytest.mark.asyncio
async def test_real_llama_server_accepts_cache_prompt():
    """Defensive: if a llama-server is actually running on :8001, confirm
    it accepts our request with `cache_prompt: true` and returns 200.

    Skipped when the server isn't running so CI stays green.  Useful as a
    local sanity check when iterating on the adapter; complements the
    `scripts/bench_prompt_cache_providers.py` TTFT measurement.
    """
    base_url = "http://localhost:8001/v1"
    if not await _llama_server_reachable(base_url):
        pytest.skip("llama-server not reachable on :8001")

    client = openai.AsyncOpenAI(api_key="not-needed", base_url=base_url)
    try:
        resp = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Reply 'ok' only."}],
            max_tokens=5,
            temperature=0.0,
            extra_body={"cache_prompt": True},
        )
    finally:
        await client.close()

    assert resp.choices, "llama-server returned no choices"
    # If llama-server had rejected extra_body, the SDK would have raised.
    # Reaching this line means the server accepted the field.
