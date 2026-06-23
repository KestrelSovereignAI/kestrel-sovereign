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
        self._disabled_routes = {}

    def _stamp_response_identity(self, response, *, model, provider):
        """Stub for #1370 model/provider stamping."""
        pass

    def _maybe_disable_route(self, provider, exc):
        """Stub for route disabling logic."""
        pass


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
        self._disabled_routes = {}

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

    def _stamp_response_identity(self, response, *, model, provider):
        """Stub for #1370 model/provider stamping."""
        pass

    def _maybe_disable_route(self, provider, exc):
        """Stub for route disabling logic."""
        pass


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


# --------------------------------------------------------------------------
# 4. Other adapters emit a terminal LLMResponse with usage too (#1684)
# --------------------------------------------------------------------------


class _FakeAsyncIter:
    """Wrap a list of chunks as an async iterator (what ``with_retry`` returns)."""

    def __init__(self, chunks: List[Any]):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


def _collect(agen) -> List[Any]:
    items: List[Any] = []

    async def _run():
        async for item in agen:
            items.append(item)

    asyncio.run(_run())
    return items


def test_openai_text_only_stream_emits_terminal_llmresponse_with_usage(monkeypatch):
    """The hoist out of `if tool_calls_accumulator` — OpenRouter inherits it
    via super() delegation, so this one test pins both."""
    from kestrel_sovereign.llm import openai_adapter as oa_mod
    from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

    content_chunk = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content="hello world", tool_calls=None),
            finish_reason=None,
        )],
    )
    usage_chunk = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19),
        choices=[],
    )

    async def fake_with_retry(fn, **kwargs):
        return _FakeAsyncIter([content_chunk, usage_chunk])

    monkeypatch.setattr(oa_mod, "with_retry", fake_with_retry)

    # `client.chat.completions.create` is evaluated to pass into with_retry
    # before the monkeypatched with_retry runs, so the attribute must exist.
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: None))
    )
    items = _collect(OpenAIAdapter().get_streaming_response_with_tools(
        client=fake_client,
        model="gpt-x",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    ))

    assert "".join(i for i in items if isinstance(i, str)) == "hello world"
    terminals = [i for i in items if isinstance(i, LLMResponse)]
    assert len(terminals) == 1, items
    assert terminals[0].input_tokens == 12
    assert terminals[0].output_tokens == 7
    assert not terminals[0].tool_calls


def test_vertex_text_stream_emits_terminal_llmresponse_with_usage(monkeypatch):
    from kestrel_sovereign.llm import vertex_adapter as vx_mod
    from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

    chunks = [
        SimpleNamespace(text="hello ", usage_metadata=None),
        SimpleNamespace(text="world", usage_metadata=None),
        SimpleNamespace(text="", usage_metadata=SimpleNamespace(
            prompt_token_count=5, candidates_token_count=3, total_token_count=8)),
    ]

    async def fake_with_retry(fn, **kwargs):
        return _FakeAsyncIter(chunks)

    monkeypatch.setattr(vx_mod, "with_retry", fake_with_retry)

    # `genai_client.aio.models.generate_content_stream` is evaluated to pass
    # into with_retry before the monkeypatch runs, so the chain must exist.
    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=lambda **k: None))
    )

    # No-tools branch routes through _stream_with_usage and must forward the
    # terminal LLMResponse.
    items = _collect(VertexAIAdapter().get_streaming_response_with_tools(
        client=fake_client,
        model="gemini-x",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    ))

    assert "".join(i for i in items if isinstance(i, str)) == "hello world"
    terminals = [i for i in items if isinstance(i, LLMResponse)]
    assert len(terminals) == 1, items
    assert terminals[0].input_tokens == 5
    assert terminals[0].output_tokens == 3

    # And the public text-only contract still drops the terminal response.
    text_items = _collect(VertexAIAdapter().get_streaming_response(
        client=fake_client,
        model="gemini-x",
        messages=[{"role": "user", "content": "hi"}],
    ))
    assert all(isinstance(i, str) for i in text_items)
    assert "".join(text_items) == "hello world"


def test_vertex_text_with_tools_path_emits_terminal_response(monkeypatch):
    """The common agent path: tools provided, model answers in text (no tool
    call). The non-streaming probe's usage must still reach the meter (#1684)."""
    from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

    adapter = VertexAIAdapter()
    probe = LLMResponse(content="just text", tool_calls=None, raw=None,
                        input_tokens=9, output_tokens=4, total_tokens=13)
    adapter.get_response = AsyncMock(return_value=probe)

    items = _collect(adapter.get_streaming_response_with_tools(
        client=SimpleNamespace(),
        model="gemini-x",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
    ))

    assert "just text" in [i for i in items if isinstance(i, str)]
    terminals = [i for i in items if isinstance(i, LLMResponse)]
    assert len(terminals) == 1
    assert terminals[0].input_tokens == 9 and terminals[0].output_tokens == 4


def test_ollama_text_stream_emits_terminal_llmresponse_with_usage(monkeypatch):
    pytest.importorskip("ollama")
    from kestrel_sovereign.llm import ollama_adapter as ol_mod
    from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter

    # Ollama reports usage on the final (done) chunk.
    chunks = [
        {"message": {"content": "hello "}},
        {"message": {"content": "world"}},
        {"message": {"content": ""}, "prompt_eval_count": 14, "eval_count": 6},
    ]

    async def fake_with_retry(fn, **kwargs):
        return _FakeAsyncIter(chunks)

    monkeypatch.setattr(ol_mod, "with_retry", fake_with_retry)
    monkeypatch.setattr(ol_mod, "OLLAMA_AVAILABLE", True)

    # `client.chat` is evaluated to pass into with_retry before the
    # monkeypatch runs, so the attribute must exist.
    fake_client = SimpleNamespace(chat=lambda **k: None)
    items = _collect(OllamaAdapter().get_streaming_response_with_tools(
        client=fake_client,
        model="llama-x",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    ))

    assert "".join(i for i in items if isinstance(i, str)) == "hello world"
    terminals = [i for i in items if isinstance(i, LLMResponse)]
    assert len(terminals) == 1, items
    assert terminals[0].input_tokens == 14
    assert terminals[0].output_tokens == 6


# --------------------------------------------------------------------------
# 5. Partial-abort flush: usage_sink meters an aborted stream (#1684)
# --------------------------------------------------------------------------


def test_record_streamed_usage_tags_partial_abort():
    svc = _FakeService()
    resp = LLMResponse(content=None, tool_calls=None, raw=None,
                       input_tokens=100, output_tokens=5, total_tokens=105)
    asyncio.run(svc._record_streamed_usage(
        resp, "claude-x", "anthropic", duration_ms=9, partial=True))
    _, lk = svc._log_llm_call.await_args
    assert lk["metadata"] == {"streamed": True, "partial_abort": True}
    assert lk["input_tokens"] == 100 and lk["output_tokens"] == 5


def test_anthropic_populates_usage_sink_as_events_arrive():
    """The sink lets the service flush partial usage if the stream aborts
    before the terminal response."""
    from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

    assert AnthropicAdapter.supports_partial_usage_flush is True

    events = [
        _ev("message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=42))),
        _ev("message_delta", usage=SimpleNamespace(output_tokens=8)),
    ]
    sink: dict = {}
    fake_messages = MagicMock()
    fake_messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
    fake_client = SimpleNamespace(messages=fake_messages)

    async def _run():
        async for _ in AnthropicAdapter().get_streaming_response_with_tools(
            client=fake_client,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            usage_sink=sink,
        ):
            pass

    asyncio.run(_run())
    assert sink == {"input_tokens": 42, "output_tokens": 8}


class _AbortingAdapter:
    """Populates usage_sink mid-stream, then aborts before the terminal."""

    supports_partial_usage_flush = True

    async def get_streaming_response_with_tools(
        self, *, client, model, messages, tools, **kwargs
    ):
        sink = kwargs.get("usage_sink")
        yield "partial "
        if sink is not None:
            sink["input_tokens"] = 100
            sink["output_tokens"] = 3
        raise asyncio.CancelledError()


def test_stream_with_tool_detection_flushes_partial_on_abort(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.streaming.provider_cache_body", lambda provider: None
    )
    svc = _RoutingService(_AbortingAdapter())

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            async for _ in svc.stream_with_tool_detection(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            ):
                pass

    asyncio.run(_run())

    # The aborted stream still recorded the partial usage the provider billed.
    svc._track_model_usage.assert_awaited_once()
    svc._log_llm_call.assert_awaited_once()
    _, lk = svc._log_llm_call.await_args
    assert lk["input_tokens"] == 100 and lk["output_tokens"] == 3
    assert lk["metadata"] == {"streamed": True, "partial_abort": True}


def test_non_flushing_adapter_records_nothing_on_abort(monkeypatch):
    """Adapters without incremental usage leave the sink empty, so an abort
    records nothing (there is nothing to record) — no spurious zero rows."""
    monkeypatch.setattr(
        "kestrel_sovereign.llm.streaming.provider_cache_body", lambda provider: None
    )

    class _NoSinkAbort:
        # no supports_partial_usage_flush -> service won't pass a sink
        async def get_streaming_response_with_tools(
            self, *, client, model, messages, tools, **kwargs
        ):
            assert "usage_sink" not in kwargs
            yield "partial "
            raise asyncio.CancelledError()

    svc = _RoutingService(_NoSinkAbort())

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            async for _ in svc.stream_with_tool_detection(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            ):
                pass

    asyncio.run(_run())
    svc._track_model_usage.assert_not_awaited()
    svc._log_llm_call.assert_not_awaited()
