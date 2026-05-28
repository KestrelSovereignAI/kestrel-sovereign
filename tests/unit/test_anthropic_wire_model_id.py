"""Unit tests for AnthropicAdapter wire model-id normalization (#1420).

Stored model ids may carry an ``anthropic/`` vendor prefix (per the
Vendor/Route/Model design), but the canonical Anthropic Messages API
rejects prefixed ids. ``AnthropicAdapter._resolve_wire_model_id`` strips
the prefix at the transport boundary; ``ClaudeMaxAdapter`` inherits the
same behavior because it also routes through api.anthropic.com.

OpenRouter and other proxies treat the prefix as meaningful routing
information, so they must NOT use this helper — they live in different
adapter classes and are exercised by their own tests.

Mirrors openclaw commit ``aa0a29099f`` (#87181) at the Python layer.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter


# ---------------------------------------------------------------------------
# Helper-level
# ---------------------------------------------------------------------------


def test_resolve_wire_model_id_strips_anthropic_prefix():
    assert (
        AnthropicAdapter._resolve_wire_model_id("anthropic/claude-opus-4-5-20251101")
        == "claude-opus-4-5-20251101"
    )


def test_resolve_wire_model_id_is_case_insensitive():
    assert (
        AnthropicAdapter._resolve_wire_model_id("Anthropic/claude-haiku-4-5")
        == "claude-haiku-4-5"
    )
    assert (
        AnthropicAdapter._resolve_wire_model_id("ANTHROPIC/claude-haiku-4-5")
        == "claude-haiku-4-5"
    )


def test_resolve_wire_model_id_passes_through_bare_id():
    """Bare ids (the form Anthropic actually accepts) survive unchanged."""
    assert (
        AnthropicAdapter._resolve_wire_model_id("claude-sonnet-4-6")
        == "claude-sonnet-4-6"
    )


def test_resolve_wire_model_id_passes_through_other_vendor_prefix():
    """Only ``anthropic/`` is stripped. Other prefixes (e.g. ``openai/``,
    ``google/``) belong to model ids that should never reach this adapter
    in the first place — if they do, surface the misroute rather than
    silently mutating the id.
    """
    assert (
        AnthropicAdapter._resolve_wire_model_id("openai/gpt-5")
        == "openai/gpt-5"
    )
    assert (
        AnthropicAdapter._resolve_wire_model_id("openrouter/anthropic/claude-3.5-sonnet")
        == "openrouter/anthropic/claude-3.5-sonnet"
    )


def test_resolve_wire_model_id_handles_empty_and_none_gracefully():
    """Defensive: the helper should not raise on falsy input. The caller
    is responsible for validating that a model id was actually resolved
    upstream; this helper just normalizes the prefix.
    """
    assert AnthropicAdapter._resolve_wire_model_id("") == ""
    assert AnthropicAdapter._resolve_wire_model_id(None) is None  # type: ignore[arg-type]


def test_resolve_wire_model_id_does_not_strip_prefix_inside_id():
    """``anthropic/`` must be a *prefix*, not a substring. If it appears
    later in the id (unlikely in practice but worth pinning), leave it.
    """
    assert (
        AnthropicAdapter._resolve_wire_model_id("foo-anthropic/claude")
        == "foo-anthropic/claude"
    )


def test_resolve_wire_model_id_inherited_by_claude_max():
    """ClaudeMaxAdapter routes to api.anthropic.com too — same strip."""
    assert (
        ClaudeMaxAdapter._resolve_wire_model_id("anthropic/claude-haiku-4-5")
        == "claude-haiku-4-5"
    )


# ---------------------------------------------------------------------------
# Wire-format integration: confirm the helper is applied at the SDK call
# site (not just available on the class).
# ---------------------------------------------------------------------------


async def _capture_messages_create_kwargs(
    adapter: AnthropicAdapter,
    model: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(type="text", text="ok")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=1),
        )
    )
    await adapter.get_response(
        client=fake_client,
        model=model,
        messages=messages,
    )
    return fake_client.messages.create.call_args.kwargs


@pytest.mark.asyncio
async def test_get_response_sends_bare_model_id_when_prefixed():
    captured = await _capture_messages_create_kwargs(
        AnthropicAdapter(),
        model="anthropic/claude-opus-4-5-20251101",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert captured["model"] == "claude-opus-4-5-20251101"


@pytest.mark.asyncio
async def test_get_response_passes_bare_model_id_through_unchanged():
    captured = await _capture_messages_create_kwargs(
        AnthropicAdapter(),
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert captured["model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_claude_max_get_response_also_strips_prefix():
    """ClaudeMaxAdapter's wire endpoint is also api.anthropic.com, so the
    same prefix-strip rule applies to OAuth-backed routes.
    """
    captured = await _capture_messages_create_kwargs(
        ClaudeMaxAdapter(),
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert captured["model"] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Streaming-path coverage: confirm the strip is applied at the streaming
# SDK call sites as well, not just the non-streaming path.
# ---------------------------------------------------------------------------


class _FakeStreamContext:
    """Async context manager that yields no events — enough to exercise
    the api_params construction without simulating real streaming.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


async def _capture_messages_stream_kwargs(
    adapter: AnthropicAdapter,
    model: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fake_client = MagicMock()
    fake_client.messages.stream = MagicMock(return_value=_FakeStreamContext())
    chunks: List[Any] = []
    async for chunk in adapter.get_streaming_response(
        client=fake_client,
        model=model,
        messages=messages,
    ):
        chunks.append(chunk)
    assert fake_client.messages.stream.called, "stream() was not invoked"
    return fake_client.messages.stream.call_args.kwargs


@pytest.mark.asyncio
async def test_get_streaming_response_sends_bare_model_id_when_prefixed():
    captured = await _capture_messages_stream_kwargs(
        AnthropicAdapter(),
        model="anthropic/claude-opus-4-5-20251101",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert captured["model"] == "claude-opus-4-5-20251101"


@pytest.mark.asyncio
async def test_get_streaming_response_with_tools_sends_bare_model_id_when_prefixed():
    """The tool-streaming path constructs its own ``api_params`` dict and
    must also apply the strip.
    """
    fake_client = MagicMock()
    fake_client.messages.stream = MagicMock(return_value=_FakeStreamContext())
    adapter = AnthropicAdapter()
    chunks: List[Any] = []
    async for chunk in adapter.get_streaming_response_with_tools(
        client=fake_client,
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "noop", "description": "x", "input_schema": {"type": "object"}}],
    ):
        chunks.append(chunk)
    assert fake_client.messages.stream.called
    captured = fake_client.messages.stream.call_args.kwargs
    assert captured["model"] == "claude-sonnet-4-6"
