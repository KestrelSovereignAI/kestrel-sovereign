"""Tests for the streaming pre-tool buffer (issue #1042 layer 2).

The constitutional honesty failure caught at v0.9.0 was: the LLM
emits "Saved:" alongside a save_fact tool_use, the SSE client
streams "Saved:" live, only the post-tool turn observes the actual
tool result. By then the user has received a confident lie.

The fix: buffer text chunks during the first stream loop. Decide
flush-or-drop after the LLMResponse marker arrives at end-of-stream:
- No tool_calls → flush (the buffered text IS the final reply).
- tool_calls    → drop from the user-facing stream (the post-tool
                  synthesis is the answer); preserve in persistence
                  so the next-turn history loader sees the model's
                  reasoning (Meridian #877).

These tests are tighter than ``test_streaming_persists_full_assistant_text``
— they assert the buffer/flush contract directly instead of inferring
it via persistence side-effects.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest


@asynccontextmanager
async def _passthrough():
    yield


def _build_mock_agent(stream_with_tool_detection_impl, orchestrator_streaming_impl=None):
    """Stand up a MagicMock-backed agent shell with the smallest set of
    attributes ``StreamingMixin.process_input_streaming`` actually
    reads. Returns (agent, add_convo_calls)."""
    add_convo_calls: list[dict] = []

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: add_convo_calls.append(
            {"role": role, "content": content, **kw}
        )
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.privacy_agent = privacy_agent
    mock_agent.features = {}
    mock_agent.did = "test-did"
    mock_agent.extension = None
    mock_agent._cached_features_prompt = ""
    mock_agent.is_request_cancelled = MagicMock(return_value=False)
    mock_agent._maybe_audit = AsyncMock()
    mock_agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    mock_agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    mock_agent.hooks_manager = None  # skip lifecycle hooks
    mock_agent._get_governing_constitution = AsyncMock(return_value="")
    mock_agent.check_solvency = AsyncMock(return_value="test-model")
    mock_agent._build_all_tools = MagicMock(return_value=[])
    mock_agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid: text)
    mock_agent.user_prompt_template = MagicMock()
    mock_agent.user_prompt_template.format.return_value = "rendered"

    ctx = MagicMock()
    ctx.system_prompt = "system"
    ctx.dynamic_user_context = ""
    ctx.messages = []
    mock_agent.context_manager = MagicMock()
    mock_agent.context_manager.build_context = AsyncMock(return_value=ctx)

    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = stream_with_tool_detection_impl
    if orchestrator_streaming_impl is not None:
        mock_agent._handle_orchestrator_response_streaming = orchestrator_streaming_impl

    # Bind the actual mixin methods so the code under test runs
    # unchanged.
    from kestrel_sovereign.agent.streaming import StreamingMixin

    mock_agent.process_input_streaming = (
        StreamingMixin.process_input_streaming.__get__(mock_agent)
    )
    mock_agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(mock_agent)
    )
    mock_agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(mock_agent)
    )
    return mock_agent, add_convo_calls


@pytest.mark.asyncio
async def test_pre_tool_text_does_not_reach_user_when_tool_called():
    """Load-bearing #1042 honesty assertion: pre-tool prose like
    "Saved:" is suppressed from the SSE stream when tool_calls are
    detected. The post-tool synthesis IS what the user sees."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    async def stream_impl(**_):
        yield "Saved: your favorite color is teal. "
        yield "Pulling memory_status now."
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="save_fact", arguments={})],
        )

    async def orchestrator_impl(**_):
        # Honest post-tool synthesis (after observing the tool result).
        for piece in ["I tried to save that ", "but couldn't confirm it persisted."]:
            yield piece

    agent, _ = _build_mock_agent(stream_impl, orchestrator_impl)

    yielded = []
    async for chunk in agent.process_input_streaming("save my color", session_id="s1"):
        yielded.append(chunk)

    visible = "".join(yielded)
    # The dishonest pre-tool claim does NOT reach the user.
    assert "Saved: your favorite color is teal." not in visible, (
        "pre-tool text must be suppressed when tool_calls are detected; "
        "letting it through is the #1042 confident-lie failure"
    )
    # The honest post-tool synthesis DOES reach the user.
    assert "couldn't confirm it persisted" in visible


@pytest.mark.asyncio
async def test_pre_tool_text_persisted_for_self_recall():
    """Companion to the assertion above: the suppressed pre-tool
    text is still persisted in the assistant turn, so the model can
    see its own reasoning on the next turn (Meridian #877)."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    async def stream_impl(**_):
        yield "Reasoning step one. "
        yield "Reasoning step two."
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="x", arguments={})],
        )

    async def orchestrator_impl(**_):
        yield "Final answer."

    agent, add_convo_calls = _build_mock_agent(stream_impl, orchestrator_impl)

    async for _ in agent.process_input_streaming("q", session_id="s2"):
        pass

    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1
    persisted = assistant_inserts[0]["content"]

    # Both halves are persisted.
    assert "Reasoning step one." in persisted
    assert "Reasoning step two." in persisted
    assert "Final answer." in persisted


@pytest.mark.asyncio
async def test_no_tool_calls_path_unchanged_text_streams_normally():
    """Regression guard: the no-tool-calls path's UX must not change.
    When the model responds without calling tools, the buffered
    chunks are flushed to the SSE client immediately after the
    stream loop ends — same end result as the pre-#1042 behaviour
    from the user's perspective."""
    async def stream_impl(**_):
        for piece in ["Hello ", "world."]:
            yield piece
        # No LLMResponse with tool_calls.

    agent, add_convo_calls = _build_mock_agent(stream_impl)

    yielded = []
    async for chunk in agent.process_input_streaming("hi", session_id="s3"):
        yielded.append(chunk)

    visible = "".join(yielded)
    assert visible == "Hello world."

    persisted = [c["content"] for c in add_convo_calls if c["role"] == "assistant"]
    assert persisted == ["Hello world."]


@pytest.mark.asyncio
async def test_pre_tool_buffer_with_no_text_chunks_is_safe():
    """Edge case: model jumps straight to a tool call without any
    pre-tool prose. Buffer is empty; nothing to suppress; nothing
    to persist on the pre-tool side."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    async def stream_impl(**_):
        # No text chunks at all — straight to LLMResponse with tools.
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="x", arguments={})],
        )

    async def orchestrator_impl(**_):
        yield "Direct synthesis."

    agent, add_convo_calls = _build_mock_agent(stream_impl, orchestrator_impl)

    yielded = []
    async for chunk in agent.process_input_streaming("q", session_id="s4"):
        yielded.append(chunk)

    visible = "".join(yielded)
    assert visible == "Direct synthesis."

    persisted = [c["content"] for c in add_convo_calls if c["role"] == "assistant"]
    # Empty pre-tool buffer + post-tool synthesis = just the synthesis.
    assert persisted == ["Direct synthesis."]
