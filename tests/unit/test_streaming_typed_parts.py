"""Typed component parts through the streaming persist path (#1914).

A PART sentinel emitted into the post-tool stream must (a) pass through to the
live client verbatim and (b) be stripped from the persisted assistant content
and recorded — position-stamped — in ``metadata['parts']`` so a reload can
re-render the component bubble. Mirrors the harness in
``test_streaming_persists_full_assistant_text`` (the persist contract this
extends).
"""
import json
from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.agent.parts import build_part_sentinel


@asynccontextmanager
async def _passthrough():
    yield


def _make_agent(add_convo_calls):
    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: add_convo_calls.append({
            "role": role, "content": content, **kw,
        })
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "test-did"
    agent.extension = None
    agent._cached_features_prompt = ""
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent.emit_event = AsyncMock()
    agent._maybe_audit = AsyncMock()
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = None
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered prompt"

    context_result = MagicMock()
    context_result.system_prompt = "system"
    context_result.dynamic_user_context = "ctx"
    context_result.messages = []
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=context_result)

    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()
    return agent


def _bind(agent):
    from kestrel_sovereign.agent.streaming import StreamingMixin

    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    agent._emit_revising_event = (
        StreamingMixin._emit_revising_event.__get__(agent)
    )


@pytest.mark.asyncio
async def test_post_tool_part_passes_through_and_persists_to_metadata():
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    add_convo_calls = []
    agent = _make_agent(add_convo_calls)

    async def mock_stream_with_tool_detection(**kwargs):
        yield "Updating the list. "
        yield ToolCallStarted(index=0, id="tc1", name="todo_add")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = mock_stream_with_tool_detection

    part_sentinel = build_part_sentinel(
        {"type": "todo", "data": {"title": "ship #1914", "status": "open"}, "id": "t1"}
    )

    # The real orchestrator drains emit_part() into a PART sentinel after the
    # tool batch; the mock yields it directly at the same point — between the
    # post-tool prose halves — so the streaming persist path sees the identical
    # stream.
    async def mock_orchestrator_streaming(**kwargs):
        yield "Done — "
        yield part_sentinel
        yield "added one item."

    agent._handle_orchestrator_response_streaming = mock_orchestrator_streaming
    _bind(agent)

    yielded = []
    async for chunk in agent.process_input_streaming("update todos", session_id="s1"):
        yielded.append(chunk)

    # (a) The PART sentinel reaches the live client verbatim.
    assert any(part_sentinel in c for c in yielded)

    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1
    persisted = assistant_inserts[0]["content"]
    metadata = assistant_inserts[0].get("metadata") or {}

    # (b) The sentinel is stripped from persisted content.
    assert "\x1eKESTREL:PART:" not in persisted
    assert persisted == "Done — added one item."

    # (c) The part is recorded in metadata, position-stamped at its clean-text
    # offset ("Done — " == 7 chars into the post-tool synthesis).
    assert "parts" in metadata
    assert len(metadata["parts"]) == 1
    part = metadata["parts"][0]
    assert part["type"] == "todo"
    assert part["data"] == {"title": "ship #1914", "status": "open"}
    assert part["id"] == "t1"
    assert part["pos"] == len("Done — ")


@pytest.mark.asyncio
async def test_no_tool_turn_persists_emitted_part():
    """A turn with no tool calls can still emit a component part (e.g. from a
    hook); it is stripped from content and persisted with position 0."""
    from kestrel_sovereign.llm.adapter import LLMResponse

    add_convo_calls = []
    agent = _make_agent(add_convo_calls)

    part_sentinel = build_part_sentinel({"type": "notice", "data": {"body": "hello"}})

    async def mock_stream_with_tool_detection(**kwargs):
        yield part_sentinel
        yield "All set."
        yield LLMResponse(content="", tool_calls=[])

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = mock_stream_with_tool_detection
    _bind(agent)

    yielded = []
    async for chunk in agent.process_input_streaming("hi", session_id="s2"):
        yielded.append(chunk)

    assert any(part_sentinel in c for c in yielded)
    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1
    persisted = assistant_inserts[0]["content"]
    metadata = assistant_inserts[0].get("metadata") or {}
    assert persisted == "All set."
    assert "\x1eKESTREL:PART:" not in persisted
    assert metadata.get("parts") == [
        {"type": "notice", "data": {"body": "hello"}, "pos": 0}
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
