"""
Wave 5E in-band revising sentinel — kestrel-sovereign #1086.

Wave 5C ships the chat client with two redundant signals to retract
pre-tool prose: the parallel /api/agent/notifications/sse `revising`
event AND the in-band sentinel embedded in the /api/agent/stream
text/plain channel itself. The in-band sentinel is the
ordering-correct primary signal — strictly serialized with the
chunks it bounds — and the SSE event is the reliability backup.

These tests pin the SERVER side: the sentinel must be yielded
through the chat stream after the first ToolCallStarted marker, with
the correct wire format and JSON payload. Client-side stripping is
covered in tests/frontend/.
"""
import json

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.llm import ToolCallStarted

from kestrel_sovereign.agent.streaming import (
    REVISE_SENTINEL_PREFIX,
    REVISE_SENTINEL_SUFFIX,
    _build_revise_sentinel,
)


class TestSentinelConstruction:
    """``_build_revise_sentinel`` produces the documented wire format."""

    def test_wire_format_round_trip(self):
        marker = ToolCallStarted(index=2, id="tc-7", name="save_fact")
        s = _build_revise_sentinel(marker)
        assert s.startswith(REVISE_SENTINEL_PREFIX)
        assert s.endswith(REVISE_SENTINEL_SUFFIX)
        # Strip bookends and parse the JSON payload.
        payload_str = s[len(REVISE_SENTINEL_PREFIX):-len(REVISE_SENTINEL_SUFFIX)]
        payload = json.loads(payload_str)
        assert payload == {
            "index": 2,
            "tool_call_id": "tc-7",
            "tool_name": "save_fact",
        }

    def test_handles_none_id_and_name(self):
        """Pre-Wave-4B adapters may yield ToolCallStarted before the
        provider has surfaced id/name (#1063 covered this for OpenAI).
        Sentinel must serialize None as JSON null without crashing."""
        marker = ToolCallStarted(index=0, id=None, name=None)
        s = _build_revise_sentinel(marker)
        payload_str = s[len(REVISE_SENTINEL_PREFIX):-len(REVISE_SENTINEL_SUFFIX)]
        payload = json.loads(payload_str)
        assert payload["tool_call_id"] is None
        assert payload["tool_name"] is None
        assert payload["index"] == 0

    def test_special_chars_in_tool_name_escaped(self):
        """JSON escaping handles weird tool names — defensive against
        plugin authors who use unusual characters."""
        marker = ToolCallStarted(index=0, id="tc1", name='quote"and\\back')
        s = _build_revise_sentinel(marker)
        payload_str = s[len(REVISE_SENTINEL_PREFIX):-len(REVISE_SENTINEL_SUFFIX)]
        payload = json.loads(payload_str)
        assert payload["tool_name"] == 'quote"and\\back'

    def test_sentinel_does_not_contain_inner_record_separator(self):
        """The closing \\x1e is the unique terminator. Any embedded
        \\x1e in the payload would break the client's parser. JSON
        encoding ensures this can't happen — \\x1e isn't a valid bare
        char in JSON strings without escaping. Pin the invariant."""
        marker = ToolCallStarted(index=0, id="tc1", name="weird\x1ename")
        s = _build_revise_sentinel(marker)
        # Exactly two \\x1e: open + close. Inner \\x1e from the name
        # got JSON-escaped to \\u001e.
        assert s.count("\x1e") == 2


@asynccontextmanager
async def _passthrough():
    yield


def _build_mock_agent():
    """Reusable mock agent — same shape as the Wave 5B test."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "did:test"
    agent.extension = None
    agent._cached_features_prompt = ""
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._maybe_audit = AsyncMock()
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = None
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"
    agent._current_request_id = "req-abc"
    agent.emit_event = AsyncMock()

    ctx = MagicMock()
    ctx.system_prompt = "system"
    ctx.dynamic_user_context = ""
    ctx.messages = []
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=ctx)

    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()

    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    agent._emit_revising_event = StreamingMixin._emit_revising_event.__get__(agent)
    agent._fire_post_response_hook = (
        StreamingMixin._fire_post_response_hook.__get__(agent)
    )

    # Disable the post-tool path so these tests focus on pre-tool +
    # marker handling. Tests that need post-tool synthesis override.
    async def post_tool(*, tool_results=None, **kw):
        if tool_results is not None:
            tool_results.append({
                "tool_call_id": "tc1", "name": "github",
                "result": {"status": "ok"},
            })
        yield "post-tool synthesis"
    agent._handle_orchestrator_response_streaming = post_tool
    return agent


@pytest.mark.asyncio
async def test_inband_sentinel_yielded_through_chat_stream():
    """The sentinel must arrive on the chat stream (via the agent
    generator) after the pre-tool chunks but before the post-tool
    synthesis chunks — strict ordering is the whole point of going
    in-band."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()

    async def stream():
        yield "Saving that now"
        yield "..."
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="github", arguments={})],
        )
    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    chunks = []
    async for c in agent.process_input_streaming("save", session_id="s"):
        chunks.append(c)

    # Find the sentinel in the chunk stream.
    sentinel_indices = [
        i for i, c in enumerate(chunks)
        if REVISE_SENTINEL_PREFIX in c
    ]
    assert len(sentinel_indices) == 1, (
        f"expected exactly one sentinel chunk, got chunks={chunks}"
    )
    si = sentinel_indices[0]
    # Pre-tool chunks come BEFORE the sentinel.
    pre_chunks = chunks[:si]
    assert "Saving that now" in "".join(pre_chunks)
    # Post-tool chunks come AFTER the sentinel.
    post_chunks = chunks[si + 1:]
    assert "post-tool synthesis" in "".join(post_chunks)
    # The sentinel chunk itself parses as the documented wire format.
    sentinel = chunks[si]
    assert sentinel.startswith(REVISE_SENTINEL_PREFIX)
    assert sentinel.endswith(REVISE_SENTINEL_SUFFIX)
    payload = json.loads(
        sentinel[len(REVISE_SENTINEL_PREFIX):-len(REVISE_SENTINEL_SUFFIX)]
    )
    assert payload == {"index": 0, "tool_call_id": "tc1", "tool_name": "github"}


@pytest.mark.asyncio
async def test_sentinel_not_appended_to_persisted_assistant_text():
    """Wire-protocol bytes must NEVER reach storage. If the sentinel
    leaks into the persisted assistant turn, the next turn's history
    loader feeds it back to the LLM and the LLM would see raw control
    chars in its context — corruption + token waste + audit nightmare.
    """
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()
    persisted = []
    agent.privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: persisted.append({"role": role, "content": content}),
    )

    async def stream():
        yield "Saving that now"
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="github", arguments={})],
        )
    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async for _ in agent.process_input_streaming("save", session_id="s"):
        pass

    # Find the assistant-row insert.
    assistant_rows = [r for r in persisted if r["role"] == "assistant"]
    assert len(assistant_rows) == 1
    persisted_text = assistant_rows[0]["content"]
    assert REVISE_SENTINEL_PREFIX not in persisted_text, (
        f"Sentinel leaked into storage! persisted: {persisted_text!r}"
    )
    assert "\x1e" not in persisted_text


@pytest.mark.asyncio
async def test_no_sentinel_when_no_marker_fires():
    """A turn with no tool calls must not emit the sentinel. The
    sentinel is a tool-call boundary signal, not a heartbeat."""
    agent = _build_mock_agent()

    async def stream():
        yield "Plain answer, no tools."
    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    chunks = []
    async for c in agent.process_input_streaming("hi", session_id="s"):
        chunks.append(c)
    assert all(REVISE_SENTINEL_PREFIX not in c for c in chunks)


@pytest.mark.asyncio
async def test_multiple_markers_yield_multiple_sentinels():
    """Two ToolCallStarted markers in the same LLM turn → two
    sentinels in stream order. Each is a distinct tool boundary."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()

    async def stream():
        yield "thinking..."
        yield ToolCallStarted(index=0, id="tc-a", name="search")
        yield ToolCallStarted(index=1, id="tc-b", name="fetch")
        yield LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc-a", name="search", arguments={}),
                ToolCall(id="tc-b", name="fetch", arguments={}),
            ],
        )
    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    chunks = []
    async for c in agent.process_input_streaming("go", session_id="s"):
        chunks.append(c)
    sentinels = [c for c in chunks if REVISE_SENTINEL_PREFIX in c]
    assert len(sentinels) == 2
    payloads = [
        json.loads(s[len(REVISE_SENTINEL_PREFIX):-len(REVISE_SENTINEL_SUFFIX)])
        for s in sentinels
    ]
    assert payloads[0]["tool_name"] == "search"
    assert payloads[1]["tool_name"] == "fetch"
