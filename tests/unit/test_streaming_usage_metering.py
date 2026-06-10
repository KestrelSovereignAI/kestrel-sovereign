"""Streamed turns must be metered (token usage + billing).

The streaming path never reached the non-streaming usage chokepoint
(_track_model_usage / _log_llm_call), so every streamed turn silently
bypassed metering. Two halves of the fix are pinned here:

1. The anthropic adapter emits a terminal LLMResponse carrying token usage
   even for a text-only stream (previously only when tool calls were present),
   so the service layer has usage to record.
2. StreamingMixin._record_streamed_usage records that usage via the same
   _track_model_usage / _log_llm_call calls the non-streaming path uses.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.streaming import StreamingMixin


# --------------------------------------------------------------------------
# 1. Adapter: a text-only stream emits a terminal LLMResponse with usage
# --------------------------------------------------------------------------


class _FakeAnthropicStream:
    def __init__(self, events: List[Any]):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for ev in self._events:
            yield ev


def _ev(event_type: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **fields)


def _drive(adapter, events) -> List[Any]:
    fake_messages = MagicMock()
    fake_messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    fake_client = SimpleNamespace(messages=fake_messages)
    items: List[Any] = []

    async def _run():
        async for item in adapter.get_streaming_response_with_tools(
            client=fake_client,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        ):
            items.append(item)

    asyncio.run(_run())
    return items


def test_text_only_stream_emits_terminal_llmresponse_with_usage():
    from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

    events = [
        _ev("message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=12))),
        _ev("content_block_start", index=0, content_block=SimpleNamespace(type="text")),
        _ev("content_block_delta", index=0,
            delta=SimpleNamespace(type="text_delta", text="hello world")),
        _ev("content_block_stop", index=0),
        _ev("message_delta", usage=SimpleNamespace(output_tokens=7)),
    ]

    items = _drive(AnthropicAdapter(), events)

    # The visible text streamed as str chunks.
    assert "".join(i for i in items if isinstance(i, str)) == "hello world"

    # And a terminal LLMResponse carries the token usage — even with no tools.
    terminals = [i for i in items if isinstance(i, LLMResponse)]
    assert len(terminals) == 1, items
    resp = terminals[0]
    assert resp.input_tokens == 12
    assert resp.output_tokens == 7
    assert not resp.tool_calls  # text-only: no tool calls


# --------------------------------------------------------------------------
# 2. Service: _record_streamed_usage meters from the terminal response
# --------------------------------------------------------------------------


class _FakeService(StreamingMixin):
    """Minimal StreamingMixin host with the recording sinks stubbed."""

    def __init__(self):
        self._track_model_usage = AsyncMock()
        self._log_llm_call = AsyncMock()


def test_record_streamed_usage_meters_terminal_response():
    svc = _FakeService()
    resp = LLMResponse(content="hi", tool_calls=None, raw=None,
                       input_tokens=10, output_tokens=20, total_tokens=30)

    asyncio.run(svc._record_streamed_usage(resp, "claude-x", "anthropic", duration_ms=42))

    svc._track_model_usage.assert_awaited_once()
    args, kwargs = svc._track_model_usage.await_args
    assert args[0] == "claude-x" and args[1] == "anthropic"
    assert kwargs.get("tokens") == 30  # input + output

    svc._log_llm_call.assert_awaited_once()
    _, lk = svc._log_llm_call.await_args
    assert lk["provider"] == "anthropic" and lk["model"] == "claude-x"
    assert lk["input_tokens"] == 10 and lk["output_tokens"] == 20
    assert lk["success"] is True
    assert lk["metadata"] == {"streamed": True}
    assert lk["tools_used"] is False


def test_record_streamed_usage_ignores_non_llmresponse():
    svc = _FakeService()
    asyncio.run(svc._record_streamed_usage("just a string", "m", "p", duration_ms=1))
    svc._track_model_usage.assert_not_awaited()
    svc._log_llm_call.assert_not_awaited()


def test_record_streamed_usage_swallows_recording_errors():
    """A metering failure must never break the stream the user is consuming."""
    svc = _FakeService()
    svc._track_model_usage = AsyncMock(side_effect=RuntimeError("usage db down"))
    resp = LLMResponse(content="hi", tool_calls=None, raw=None,
                       input_tokens=1, output_tokens=2, total_tokens=3)
    # Must not raise.
    asyncio.run(svc._record_streamed_usage(resp, "m", "p", duration_ms=1))


# --------------------------------------------------------------------------
# 3. End-to-end: stream_with_tool_detection meters the terminal response once
# --------------------------------------------------------------------------


class _FakeStreamAdapter:
    async def get_streaming_response_with_tools(
        self, *, client, model, messages, tools, **kwargs
    ):
        yield "hello "
        yield "world"
        yield LLMResponse(content="hello world", tool_calls=None, raw=None,
                          input_tokens=11, output_tokens=22, total_tokens=33)


class _RoutingService(StreamingMixin):
    """Minimal service that routes stream_with_tool_detection to a fake adapter."""

    def __init__(self, adapter):
        self._track_model_usage = AsyncMock()
        self._log_llm_call = AsyncMock()
        self._backend = None          # not REMOTE_GPU -> skip remote branch
        self._remote_client = None
        self._adapter = adapter

    def _check_policy(self):
        pass

    def resolve_provider_routing(self, **kwargs):
        return (
            [{"name": "anthropic", "client": object(),
              "adapter": self._adapter, "is_local": False}],
            "claude-x",
        )

    def _check_model_tool_support(self, providers, tools, model_override):
        return tools

    def _resolve_concrete_model(self, target_model, provider):
        return target_model


def test_stream_with_tool_detection_records_usage_exactly_once(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.streaming.provider_cache_body", lambda provider: None
    )
    svc = _RoutingService(_FakeStreamAdapter())

    async def _run():
        items = []
        async for item in svc.stream_with_tool_detection(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
        ):
            items.append(item)
        return items

    items = asyncio.run(_run())

    # Visible text streamed; usage metered once from the terminal response.
    assert "".join(i for i in items if isinstance(i, str)) == "hello world"
    svc._track_model_usage.assert_awaited_once()
    svc._log_llm_call.assert_awaited_once()
    _, lk = svc._log_llm_call.await_args
    assert lk["input_tokens"] == 11 and lk["output_tokens"] == 22
    assert lk["metadata"] == {"streamed": True}
