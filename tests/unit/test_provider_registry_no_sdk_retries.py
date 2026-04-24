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
    """End-to-end timing check: with ``max_retries=0``, an OpenAI
    ``APIConnectionError`` (nothing listening on the port) surfaces in
    under a second — well below the ~8s the SDK default would produce."""

    @pytest.mark.asyncio
    async def test_connection_refused_fails_in_under_one_second(self):
        # Port 1 is reserved; nothing should be listening.
        client = openai.AsyncOpenAI(
            api_key="local",
            base_url="http://localhost:1/v1",
            max_retries=0,
        )
        try:
            start = time.monotonic()
            with pytest.raises(openai.APIConnectionError):
                await client.chat.completions.create(
                    model="anything",
                    messages=[{"role": "user", "content": "x"}],
                )
            elapsed = time.monotonic() - start
            # Default SDK retries would land around 8s. 1s is a generous
            # ceiling that still clearly fails if retries come back on.
            assert elapsed < 1.0, (
                f"Connection refused took {elapsed:.2f}s — SDK retries "
                f"appear to be back on (expected <1s with max_retries=0)."
            )
        finally:
            await client.close()
