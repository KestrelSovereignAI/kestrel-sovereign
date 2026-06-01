"""Regression guard for #656.

When a local LLM server (llama.cpp, Ollama, generic OpenAI-compatible)
isn't running, connection attempts should fail FAST — no silent SDK
retries. The OpenAI Python SDK defaults to 2 retries with exponential
backoff (~8s total for a connection-refused error), which compounds
with any Kestrel-level fallback chain and turns every user message
into a noticeable stall.

The fix is ``max_retries=0`` on every ``openai.AsyncOpenAI(...)``
constructed by ``ProviderRegistry`` — the single retry owner is
``llm/retry.py`` at the Kestrel layer, which explicitly doesn't retry
connection errors. These tests lock that in so a future cleanup that
removes "unused" constructor kwargs doesn't silently re-introduce the
8-second wall.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import openai
import pytest

from kestrel_sovereign.llm.provider_registry import ProviderRegistry
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter


def _empty_registry() -> ProviderRegistry:
    """Instantiate without running through config parsing; we're only
    exercising the private client-builder."""
    reg = ProviderRegistry.__new__(ProviderRegistry)
    reg.config = {}
    return reg


class TestOpenAICompatibleClientsDisableSdkRetries:
    """Every OpenAI-compatible client the registry constructs must have
    ``max_retries=0``. Generic OpenAI, llama.cpp (OpenAI-compatible),
    Ollama's OpenAI shim, xAI, Groq, RunPod — all go through the same
    adapter-class branch in ``_build_client_and_adapter``."""

    def test_generic_openai_compatible_client_has_zero_retries(self):
        """Covers llama.cpp, xAI, Groq, RunPod, OpenAI itself — any vendor
        that uses OpenAIAdapter."""
        reg = _empty_registry()
        route_cfg = {
            "base_url": "http://localhost:8001/v1",
            "api_key": "local",
            "local": True,
        }
        client, adapter = reg._build_client_and_adapter(
            vendor="llama_cpp", route="local",
            adapter_cls=OpenAIAdapter,
            vendor_cfg={}, route_cfg=route_cfg,
        )
        assert isinstance(client, openai.AsyncOpenAI)
        assert client.max_retries == 0, (
            "Generic OpenAI-compatible client is retrying on SDK level — "
            "this regresses #656 (connection-refused triggers ~8s of "
            "silent retries before the Kestrel fallback chain advances)."
        )

    def test_openrouter_client_has_zero_retries(self):
        """OpenRouter uses its own adapter but constructs the client
        with the same SDK; the fix note in the adapter says so explicitly."""
        reg = _empty_registry()
        route_cfg = {"api_key": "sk-or-test"}
        client, adapter = reg._build_client_and_adapter(
            vendor="openrouter", route="api",
            adapter_cls=OpenRouterAdapter,
            vendor_cfg={}, route_cfg=route_cfg,
        )
        assert isinstance(client, openai.AsyncOpenAI)
        assert client.max_retries == 0


class TestConnectionErrorFailsFast:
    """End-to-end check: with ``max_retries=0``, an
    ``APIConnectionError`` surfaces after EXACTLY ONE attempt — no
    silent SDK retries.

    Previous version asserted ``elapsed < 1.0s`` which was both
    flaky (CI runner load) and weakly load-bearing (codex P2: OpenAI
    2.x's default ``max_retries=2`` backoff can complete in 1-1.5s
    on a fast host, so any threshold high enough for CI stability is
    high enough to silently pass a retries-back-on regression). Count
    attempts directly via an ``httpx.AsyncHTTPTransport`` spy — that's
    both flake-free and precisely measures the thing we actually care
    about: number of HTTP attempts == 1.
    """

    @pytest.mark.asyncio
    async def test_connection_refused_makes_exactly_one_attempt(self):
        """The SDK must NOT retry. One ``httpx.send`` call per request."""
        import httpx

        attempt_count = 0
        original_handle_async_request = (
            httpx.AsyncHTTPTransport.handle_async_request
        )

        async def counting_handle(self, request):
            nonlocal attempt_count
            attempt_count += 1
            # Defer to the real transport so the connection-refused
            # error is raised authentically (instead of mocking out the
            # whole stack, which could mask future bugs in how the SDK
            # propagates transport-level errors).
            return await original_handle_async_request(self, request)

        # Port 1 is reserved; nothing should be listening.
        client = openai.AsyncOpenAI(
            api_key="local",
            base_url="http://localhost:1/v1",
            max_retries=0,
        )
        try:
            with patch.object(
                httpx.AsyncHTTPTransport,
                "handle_async_request",
                counting_handle,
            ):
                with pytest.raises(openai.APIConnectionError):
                    await client.chat.completions.create(
                        model="anything",
                        messages=[{"role": "user", "content": "x"}],
                    )
        finally:
            await client.close()

        # SDK default ``max_retries=2`` would produce 3 attempts (1
        # initial + 2 retries). With ``max_retries=0`` we expect
        # exactly 1.
        assert attempt_count == 1, (
            f"Connection refused triggered {attempt_count} HTTP attempts — "
            f"SDK retries appear to be back on (expected exactly 1 with "
            f"max_retries=0)."
        )
