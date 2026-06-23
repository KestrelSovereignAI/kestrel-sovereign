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


@pytest.mark.asyncio
async def test_inline_executed_tool_emitted_part_drained_and_persisted():
    """On the codex app-server inline path, a feature tool runs INSIDE the LLM
    call and emits a part via ``emit_part`` (buffered on the collector). The
    streaming layer must drain it into a live PART sentinel AND persist it to
    metadata — otherwise inline-tool parts (the openai:plan path, e.g. todo
    tools on Emma) silently vanish.
    """
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.agent.parts import emit_part
    from kestrel_sovereign.llm.adapter import LLMResponse

    add_convo_calls = []
    agent = _make_agent(add_convo_calls)
    agent._make_inline_tool_executor = MagicMock(return_value=None)
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent.context_manager.build_context.return_value.degraded_mode = False

    async def stream(**kwargs):
        yield "Updating. "
        # The inline tool fires its marker, then (simulating its body) emits a
        # component part into the active per-turn collector.
        yield ToolCallStarted(index=0, id="tc1", name="todo_add")
        emit_part("todo", {"title": "ship #1914"}, part_id="t1")
        yield "Added it."
        resp = LLMResponse(content="", tool_calls=None)
        resp.executed_tool_calls = [
            {"id": "tc1", "name": "todo_add", "arguments": {}, "result": {"ok": True}}
        ]
        yield resp

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()
    _bind(agent)

    yielded = []
    async for chunk in agent.process_input_streaming("update todos", session_id="s3"):
        yielded.append(chunk)

    # The drained part reaches the live client as a PART sentinel.
    assert any("\x1eKESTREL:PART:" in c for c in yielded)

    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1
    persisted = assistant_inserts[0]["content"]
    metadata = assistant_inserts[0].get("metadata") or {}
    assert "\x1eKESTREL:PART:" not in persisted
    assert "parts" in metadata
    assert len(metadata["parts"]) == 1
    assert metadata["parts"][0]["type"] == "todo"
    assert metadata["parts"][0]["data"] == {"title": "ship #1914"}
    assert metadata["parts"][0]["id"] == "t1"
    assert isinstance(metadata["parts"][0].get("pos"), int)


@pytest.mark.asyncio
async def test_parts_dropped_when_post_response_hook_blocks_text():
    """If a POST_RESPONSE hook rewrites/blocks the assistant text (audit denial),
    the component parts must NOT persist — otherwise reload would render the
    structured bubbles next to the blocked placeholder, leaking what the hook
    removed."""
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    add_convo_calls = []
    agent = _make_agent(add_convo_calls)
    # The hook BLOCKS: it replaces the post-tool text with a denial placeholder.
    agent._fire_post_response_hook = AsyncMock(
        side_effect=lambda text, sid, **_: "[Response blocked by audit]"
    )

    async def mock_stream_with_tool_detection(**kwargs):
        yield "Updating. "
        yield ToolCallStarted(index=0, id="tc1", name="todo_add")
        yield LLMResponse(
            content="", tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = mock_stream_with_tool_detection

    part_sentinel = build_part_sentinel({"type": "todo", "data": {"secret": "x"}})

    async def mock_orchestrator_streaming(**kwargs):
        yield "Done "
        yield part_sentinel
        yield "now."

    agent._handle_orchestrator_response_streaming = mock_orchestrator_streaming
    _bind(agent)

    async for _ in agent.process_input_streaming("update", session_id="s4"):
        pass

    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1
    metadata = assistant_inserts[0].get("metadata") or {}
    assert assistant_inserts[0]["content"] == "[Response blocked by audit]"
    # The blocked turn must NOT carry the structured part the hook scrubbed.
    assert "parts" not in metadata


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
