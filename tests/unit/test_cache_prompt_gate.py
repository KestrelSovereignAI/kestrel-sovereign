"""Tests for the llama.cpp `cache_prompt` extension gate (issue #704).

Two layers of coverage:

1. `provider_cache_body()` helper in provider_registry — returns the right
   extra_body dict per vendor. Tight whitelist; easy to regression-test.

2. `OpenAIAdapter.get_response()` passes `extra_body` through to the SDK's
   chat completions call when it's set, and does NOT inject the field
   when the caller passes None or omits it.

The service-layer integration (is the helper actually called from each
call site with the real provider dict?) is covered by the existing
`test_adapter_cache_stability` family plus the per-call-site service
tests — those would have regressed if I'd missed a call site.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.provider_registry import provider_cache_body


# ---------------------------------------------------------------------------
# provider_cache_body gate
# ---------------------------------------------------------------------------


def test_provider_cache_body_returns_cache_prompt_for_llama_cpp():
    """llama.cpp (llama-server) is the one vendor that understands this
    extension today. Helper must return the cache_prompt flag.
    """
    provider = {"vendor": "llama_cpp", "route": "local", "name": "llama_cpp"}
    assert provider_cache_body(provider) == {"cache_prompt": True}


@pytest.mark.parametrize(
    "vendor",
    [
        "openai",        # official OpenAI cloud
        "openai_plan",   # OpenAI OAuth wrapper, strict
        "anthropic",     # different cache mechanism (cache_control)
        "claude_max",    # Anthropic OAuth wrapper
        "openrouter",    # proxy, routes to various backends
        "ollama",        # OpenAI-compat but different KV cache plumbing
        "xai",           # strict OpenAI-compat
        "runpod",        # vLLM under the hood
        "groq",
        "cerebras",
        "google",
        "vertex_ai",
    ],
)
def test_provider_cache_body_returns_none_for_non_llama_cpp_vendors(vendor):
    """Every other vendor must get None — sending cache_prompt to a strict
    OpenAI-compatible endpoint can return 4xx. Tight whitelist protects
    against accidental broadening.
    """
    provider = {"vendor": vendor, "route": "api", "name": vendor}
    assert provider_cache_body(provider) is None


def test_provider_cache_body_returns_none_when_vendor_missing():
    """Defensive: missing vendor key (shouldn't happen in practice) must
    not crash — return None so the call proceeds without extra_body.
    """
    assert provider_cache_body({}) is None
    assert provider_cache_body({"name": "something", "route": "api"}) is None


def test_provider_cache_body_is_fresh_dict_per_call():
    """The helper must return a fresh dict each call so callers mutating
    it (e.g. merging other body extensions) can't poison later calls.
    """
    a = provider_cache_body({"vendor": "llama_cpp"})
    b = provider_cache_body({"vendor": "llama_cpp"})
    assert a == b
    assert a is not b, "helper returned the same dict instance twice"


# ---------------------------------------------------------------------------
# OpenAIAdapter extra_body passthrough
# ---------------------------------------------------------------------------


async def _capture_create_kwargs(extra_body_arg) -> Dict[str, Any]:
    """Run OpenAIAdapter.get_response with the given extra_body kwarg and
    return the kwargs the OpenAI SDK's create() call received.
    """
    fake_client = MagicMock()
    create_call = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    fake_client.chat.completions.create = create_call

    adapter = OpenAIAdapter()
    await adapter.get_response(
        client=fake_client,
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        extra_body=extra_body_arg,
    )
    return create_call.call_args.kwargs


@pytest.mark.asyncio
async def test_openai_adapter_forwards_extra_body_when_set():
    """Caller passes `extra_body={"cache_prompt": True}` → adapter must
    forward it verbatim to the SDK's create() call.  The OpenAI SDK
    serializes extra_body into the HTTP request body, which is how
    llama-server sees the cache_prompt flag.
    """
    captured = await _capture_create_kwargs({"cache_prompt": True})
    assert captured.get("extra_body") == {"cache_prompt": True}


@pytest.mark.asyncio
async def test_openai_adapter_omits_extra_body_when_none():
    """Caller passes `extra_body=None` (the typical case for non-llama_cpp
    providers) → adapter must NOT include extra_body in the create() call.
    Forwarding None could cause some SDK versions to serialize a null body
    which strict endpoints reject.
    """
    captured = await _capture_create_kwargs(None)
    assert "extra_body" not in captured, (
        f"extra_body leaked into create() kwargs when caller passed None: "
        f"{captured.get('extra_body')!r}"
    )


@pytest.mark.asyncio
async def test_openai_adapter_omits_extra_body_when_empty_dict():
    """An empty dict is also a falsy value; the adapter should skip it
    rather than sending `extra_body: {}` on the wire.
    """
    captured = await _capture_create_kwargs({})
    assert "extra_body" not in captured
