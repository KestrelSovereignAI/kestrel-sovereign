"""Tests for ToolCallStarted emission in the streaming-with-tools path.

Wave 4B of kestrel-sovereign #1048. Pins:

* Both anthropic and openai adapters emit ``ToolCallStarted`` exactly
  once per distinct tool-call index, in the order the corresponding
  entries appear in the final ``LLMResponse.tool_calls``.
* OpenAI's first-delta case where id/name may not yet be populated
  is honored: ``ToolCallStarted.id`` / ``.name`` are ``None`` rather
  than an empty string.
* Anthropic's ``content_block_start`` case populates both id and name.
* Multiple concurrent tool calls preserve index order.
* Text chunks may interleave with ``ToolCallStarted`` events
  (text-before-tool and text-during-tool).
* Malformed JSON in tool arguments at end-of-stream falls back to
  the SDK 0.7.0 ``{"_raw": "<accumulated>"}`` sentinel in the final
  ``LLMResponse``, rather than raising — so a bad model output
  surfaces as a tool-result error rather than crashing the turn.

These tests use mock provider streams. End-to-end propagation through
the framework's streaming pipeline lives in Wave 5.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.llm import LLMResponse, ToolCall, ToolCallStarted


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class _FakeAnthropicStream:
    """Async-context-manager stream stub that yields a scripted event sequence.

    Anthropic's SDK exposes ``client.messages.stream(...)`` as an async
    context manager whose ``async for event in stream`` produces typed
    events. We mimic just the surface our adapter touches.
    """

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


def _anthropic_event(event_type: str, **fields: Any) -> SimpleNamespace:
    """Build an Anthropic-shaped SDK event SimpleNamespace."""
    return SimpleNamespace(type=event_type, **fields)


def _drive(adapter, events) -> List[Any]:
    """Run get_streaming_response_with_tools to completion and return
    every yielded item in order."""
    fake_stream = _FakeAnthropicStream(events)
    fake_messages = MagicMock()
    fake_messages.stream = MagicMock(return_value=fake_stream)
    fake_client = SimpleNamespace(messages=fake_messages)

    items: List[Any] = []

    async def _run():
        async for item in adapter.get_streaming_response_with_tools(
            client=fake_client,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        ):
            items.append(item)

    asyncio.run(_run())
    return items


class TestAnthropicEmitsToolCallStarted:
    """AnthropicAdapter.get_streaming_response_with_tools must emit
    ToolCallStarted at every content_block_start with type='tool_use',
    populating both id and name from the block."""

    def test_emits_at_content_block_start_with_tool_use(self):
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

        events = [
            _anthropic_event(
                "message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10)),
            ),
            _anthropic_event(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(
                    type="tool_use", id="toolu_abc", name="lookup"
                ),
            ),
            _anthropic_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"q":'),
            ),
            _anthropic_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json='"hi"}'),
            ),
            _anthropic_event("content_block_stop", index=0),
            _anthropic_event(
                "message_delta",
                usage=SimpleNamespace(output_tokens=5),
            ),
        ]

        items = _drive(AnthropicAdapter(), events)
        starts = [i for i in items if isinstance(i, ToolCallStarted)]
        assert len(starts) == 1
        assert starts[0] == ToolCallStarted(index=0, id="toolu_abc", name="lookup")

        finals = [i for i in items if isinstance(i, LLMResponse)]
        assert len(finals) == 1
        assert finals[0].has_tool_calls
        assert finals[0].tool_calls[0].arguments == {"q": "hi"}

    def test_text_before_tool_then_tool_emits_in_order(self):
        """Anthropic frequently emits a text block before a tool_use
        block. ToolCallStarted must arrive AFTER the leading text
        chunks, in stream order."""
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

        events = [
            _anthropic_event(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(type="text"),
            ),
            _anthropic_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Let me check"),
            ),
            _anthropic_event("content_block_stop", index=0),
            _anthropic_event(
                "content_block_start",
                index=1,
                content_block=SimpleNamespace(
                    type="tool_use", id="toolu_x", name="check"
                ),
            ),
            _anthropic_event(
                "content_block_delta",
                index=1,
                delta=SimpleNamespace(type="input_json_delta", partial_json="{}"),
            ),
            _anthropic_event("content_block_stop", index=1),
        ]

        items = _drive(AnthropicAdapter(), events)
        # Order: text chunk, then ToolCallStarted, then final LLMResponse.
        assert items[0] == "Let me check"
        starts = [i for i in items if isinstance(i, ToolCallStarted)]
        assert len(starts) == 1
        assert starts[0].index == 1
        assert items.index("Let me check") < items.index(starts[0])

    def test_multiple_concurrent_tool_calls_emit_in_index_order(self):
        """Codex finding #2: multiple concurrent tool_use blocks emit
        ToolCallStarted in the same order as their indices appear in
        the final LLMResponse.tool_calls."""
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

        events = []
        for idx, (tid, name) in enumerate([
            ("toolu_a", "fn_a"),
            ("toolu_b", "fn_b"),
            ("toolu_c", "fn_c"),
        ]):
            events.extend([
                _anthropic_event(
                    "content_block_start",
                    index=idx,
                    content_block=SimpleNamespace(
                        type="tool_use", id=tid, name=name
                    ),
                ),
                _anthropic_event(
                    "content_block_delta",
                    index=idx,
                    delta=SimpleNamespace(
                        type="input_json_delta", partial_json="{}"
                    ),
                ),
                _anthropic_event("content_block_stop", index=idx),
            ])

        items = _drive(AnthropicAdapter(), events)
        starts = [i for i in items if isinstance(i, ToolCallStarted)]
        assert [s.index for s in starts] == [0, 1, 2]
        assert [s.name for s in starts] == ["fn_a", "fn_b", "fn_c"]

        final = next(i for i in items if isinstance(i, LLMResponse))
        # ToolCallStarted index aligns with final.tool_calls position.
        for i, tc in enumerate(final.tool_calls or []):
            assert starts[i].index == i
            assert starts[i].id == tc.id

    def test_malformed_json_falls_back_to_underscore_raw(self):
        """Codex finding #1: malformed argument JSON yields
        {"_raw": "<accumulated>"} in the final LLMResponse, not an
        exception. Renamed from "raw" to "_raw" in SDK 0.7.0 to
        signal "sentinel, not real data"."""
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

        events = [
            _anthropic_event(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(
                    type="tool_use", id="toolu_x", name="lookup"
                ),
            ),
            _anthropic_event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    type="input_json_delta", partial_json='{"q": "hi'  # truncated
                ),
            ),
            _anthropic_event("content_block_stop", index=0),
        ]

        items = _drive(AnthropicAdapter(), events)
        final = next(i for i in items if isinstance(i, LLMResponse))
        assert final.tool_calls is not None
        assert final.tool_calls[0].arguments == {"_raw": '{"q": "hi'}


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _openai_chunk(*, content=None, tool_calls=None, usage=None):
    """Build an OpenAI streaming chunk SimpleNamespace."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )
    choices = [SimpleNamespace(delta=delta)]
    return SimpleNamespace(choices=choices, usage=usage)


def _openai_tc_delta(index, *, id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


class _FakeOpenAIStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for c in self._chunks:
            yield c


def _drive_openai(adapter, chunks):
    """Run OpenAI streaming-with-tools to completion."""
    fake_stream = _FakeOpenAIStream(chunks)

    async def _create(*args, **kwargs):
        return fake_stream

    fake_completions = SimpleNamespace(create=_create)
    fake_chat = SimpleNamespace(completions=fake_completions)
    fake_client = SimpleNamespace(chat=fake_chat)

    items: List[Any] = []

    async def _run():
        async for item in adapter.get_streaming_response_with_tools(
            client=fake_client,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        ):
            items.append(item)

    asyncio.run(_run())
    return items


class TestOpenAIEmitsToolCallStarted:
    """OpenAIAdapter.get_streaming_response_with_tools must emit
    ToolCallStarted exactly once per distinct index, on the FIRST
    delta carrying that index. id/name are populated when the first
    delta surfaces them; ``None`` otherwise — the first-delta case
    pins the contract's MAY-BE-NONE rule."""

    def test_emits_with_id_and_name_when_first_delta_carries_them(self):
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        chunks = [
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(
                        index=0, id="call_1", name="lookup", arguments=""
                    )
                ]
            ),
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=0, arguments='{"q":')
                ]
            ),
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=0, arguments='"hi"}')
                ]
            ),
        ]

        items = _drive_openai(OpenAIAdapter(), chunks)
        starts = [i for i in items if isinstance(i, ToolCallStarted)]
        assert len(starts) == 1
        assert starts[0] == ToolCallStarted(index=0, id="call_1", name="lookup")

    def test_emits_with_none_when_first_delta_lacks_id_and_name(self):
        """The OPEN spec question codex flagged: when does
        ToolCallStarted fire if OpenAI's first delta carries only the
        index? Answer (locked into the contract): fire at first
        delta, with id=None and name=None. The final LLMResponse is
        the source of truth for the assembled call."""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        chunks = [
            # First delta: only index, no id, no name (rare but possible).
            _openai_chunk(tool_calls=[_openai_tc_delta(index=0)]),
            # Subsequent delta delivers id + name.
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=0, id="call_x", name="lookup")
                ]
            ),
            _openai_chunk(
                tool_calls=[_openai_tc_delta(index=0, arguments="{}")]
            ),
        ]

        items = _drive_openai(OpenAIAdapter(), chunks)
        starts = [i for i in items if isinstance(i, ToolCallStarted)]
        assert len(starts) == 1
        # First delta had no id/name, so the marker carries None for both
        # — even though subsequent deltas filled them in.
        assert starts[0].index == 0
        assert starts[0].id is None
        assert starts[0].name is None

        # The final LLMResponse, however, is the source of truth.
        final = next(i for i in items if isinstance(i, LLMResponse))
        assert final.tool_calls[0].id == "call_x"
        assert final.tool_calls[0].name == "lookup"

    def test_emits_exactly_once_per_index_even_across_many_deltas(self):
        """A second delta for the same index does NOT trigger a
        second ToolCallStarted. Pinned because the audit hook may
        rely on event-count-equals-tool-call-count."""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        chunks = [
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=0, id="c", name="f", arguments="")
                ]
            ),
            _openai_chunk(
                tool_calls=[_openai_tc_delta(index=0, arguments="{}")]
            ),
            _openai_chunk(
                tool_calls=[_openai_tc_delta(index=0, arguments="")]
            ),
        ]
        items = _drive_openai(OpenAIAdapter(), chunks)
        starts = [i for i in items if isinstance(i, ToolCallStarted)]
        assert len(starts) == 1

    def test_text_during_tool_emits_text_chunks_alongside_marker(self):
        """OpenAI streams may emit text and tool deltas in the same
        chunk (delta has BOTH content and tool_calls). The adapter
        yields the text chunk and the ToolCallStarted marker for the
        same provider chunk; consumers must handle interleaving."""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        chunks = [
            # Leading text-only chunk.
            _openai_chunk(content="Let me look "),
            # Mixed chunk: text continues AND first tool delta arrives.
            _openai_chunk(
                content="that up. ",
                tool_calls=[
                    _openai_tc_delta(
                        index=0, id="c", name="lookup", arguments=""
                    )
                ],
            ),
            _openai_chunk(
                tool_calls=[_openai_tc_delta(index=0, arguments="{}")]
            ),
        ]
        items = _drive_openai(OpenAIAdapter(), chunks)
        # Text appears in stream order; ToolCallStarted appears after
        # the text chunks that preceded its first delta in the same
        # provider chunk.
        text_indices = [i for i, x in enumerate(items) if isinstance(x, str)]
        marker_indices = [
            i for i, x in enumerate(items) if isinstance(x, ToolCallStarted)
        ]
        assert items[text_indices[0]] == "Let me look "
        assert items[text_indices[1]] == "that up. "
        # The marker is yielded AFTER the text in its own provider chunk.
        assert marker_indices[0] > text_indices[1]

    def test_multiple_concurrent_indices_emit_in_arrival_order(self):
        """Codex finding #2 + #3: multiple concurrent tool calls.
        OpenAI may stream them with interleaved deltas (tool 0 first,
        then tool 1, then more of tool 0). The marker fires once per
        index in the order each index's FIRST delta arrives, which
        matches the order in the assembled tool_calls list (sorted
        by index in the adapter)."""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        chunks = [
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=0, id="c0", name="fn0")
                ]
            ),
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=1, id="c1", name="fn1")
                ]
            ),
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=0, arguments="{}"),
                    _openai_tc_delta(index=1, arguments="{}"),
                ]
            ),
        ]
        items = _drive_openai(OpenAIAdapter(), chunks)
        starts = [i for i in items if isinstance(i, ToolCallStarted)]
        assert [s.index for s in starts] == [0, 1]
        final = next(i for i in items if isinstance(i, LLMResponse))
        assert [tc.id for tc in (final.tool_calls or [])] == ["c0", "c1"]

    def test_malformed_json_falls_back_to_underscore_raw(self):
        """Codex finding #1: same fallback contract as Anthropic."""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        chunks = [
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=0, id="c", name="fn", arguments="")
                ]
            ),
            _openai_chunk(
                tool_calls=[
                    _openai_tc_delta(index=0, arguments='{"q": "h')
                ]
            ),
            # Stream ends mid-JSON — final accumulated args are invalid.
        ]
        items = _drive_openai(OpenAIAdapter(), chunks)
        final = next(i for i in items if isinstance(i, LLMResponse))
        assert final.tool_calls[0].arguments == {"_raw": '{"q": "h'}


# ---------------------------------------------------------------------------
# Codex (OpenAI Plan via Responses API) — added in response to codex
# review of Wave 4B which flagged this as a missed adapter.
# ---------------------------------------------------------------------------


class TestCodexAdapterEmissionLogic:
    """The codex_adapter now drives the ``codex app-server``. Tool calls
    surface as ``item/completed`` events with a function-call item type;
    ``get_streaming_response_with_tools`` must emit exactly one
    ``ToolCallStarted`` per such item (id/name from the item), then a
    final ``LLMResponse`` carrying the assembled ``ToolCall``s.

    Behavioral test against the real adapter with a scripted in-memory
    app-server — no source-grep, no live binary.
    """

    @pytest.mark.asyncio
    async def test_emits_one_toolcallstarted_per_app_server_tool_item(self):
        from kestrel_sovereign.llm.adapter import LLMResponse
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter
        from kestrel_sdk.llm import ToolCallStarted

        class _ScriptedApp:
            def __init__(self):
                self.registered = {}

            async def ensure_started(self):
                pass

            async def request(self, method, params=None, *, timeout=120):
                if method == "thread/start":
                    return {"thread": {"id": "t1"}}
                return {}

            def register_server_request_handler(self, m, h):
                self.registered[m] = h
                return lambda: self.registered.pop(m, None)

            def open_turn_sink(self, key):
                return key

            def close_turn_sink(self, key):
                pass

            async def iter_turn_events(self, sink, *, idle_timeout=120):
                for ev in [
                    {"method": "item/agentMessage/delta",
                     "params": {"delta": "calling"}},
                    {"method": "item/completed", "params": {"item": {
                        "type": "functionCall", "id": "c1",
                        "name": "alpha", "arguments": '{"x": 1}'}}},
                    {"method": "item/completed", "params": {"item": {
                        "type": "functionCall", "id": "c2",
                        "name": "beta", "arguments": {"y": 2}}}},
                    # duplicate id must NOT re-emit
                    {"method": "item/completed", "params": {"item": {
                        "type": "functionCall", "id": "c1",
                        "name": "alpha", "arguments": '{"x": 1}'}}},
                    {"method": "turn/completed", "params": {}},
                ]:
                    yield ev

        async def exe(name, args):
            return {"success": True, "result": "ok"}

        adapter = CodexAdapter()
        adapter._client = _ScriptedApp()
        out = [
            c async for c in adapter.get_streaming_response_with_tools(
                client=None, model="auto",
                messages=[{"role": "user", "content": "go"}],
                tools=[{"type": "function", "function": {
                    "name": "alpha", "description": "d",
                    "parameters": {"type": "object"}}}],
                session_id="s", tool_executor=exe,
            )
        ]
        starts = [c for c in out if isinstance(c, ToolCallStarted)]
        finals = [c for c in out if isinstance(c, LLMResponse)]
        assert [(s.id, s.name) for s in starts] == [
            ("c1", "alpha"), ("c2", "beta")
        ]
        assert finals and len(finals[-1].tool_calls) == 2
        assert finals[-1].tool_calls[0].arguments == {"x": 1}
        assert finals[-1].tool_calls[1].arguments == {"y": 2}
