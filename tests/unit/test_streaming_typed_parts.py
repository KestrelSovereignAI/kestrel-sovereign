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
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
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
    parts_md = metadata.get("parts")
    assert len(parts_md) == 1
    assert {k: v for k, v in parts_md[0].items() if k != "seq"} == {
        "type": "notice", "data": {"body": "hello"}, "pos": 0,
    }


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

    # The drained part reaches the live client as a PART sentinel, and it lands
    # at the tool boundary — BEFORE the post-tool prose ("Added it."), not after.
    part_idx = next(i for i, c in enumerate(yielded) if "\x1eKESTREL:PART:" in c)
    added_idx = next(i for i, c in enumerate(yielded) if "Added it." in c)
    assert part_idx < added_idx

    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1
    persisted = assistant_inserts[0]["content"]
    metadata = assistant_inserts[0].get("metadata") or {}
    assert "\x1eKESTREL:PART:" not in persisted
    # The honesty layer retracts the pre-tool prose ("Updating. ") to metadata;
    # the persisted content is the post-tool answer, and the part sits at its
    # start (pos 0) — the tool boundary, matching the live stream order.
    assert persisted == "Added it."
    assert "parts" in metadata
    assert len(metadata["parts"]) == 1
    assert metadata["parts"][0]["type"] == "todo"
    assert metadata["parts"][0]["data"] == {"title": "ship #1914"}
    assert metadata["parts"][0]["id"] == "t1"
    assert metadata["parts"][0]["pos"] == 0


@pytest.mark.asyncio
async def test_inline_part_lands_after_tool_done_sentinel_before_text():
    """A Codex inline tool emits its part while running, then the adapter yields
    the tool's done sentinel, then answer text. The part must land AFTER the
    done sentinel (so it renders below the completed tool card) but BEFORE the
    answer prose."""
    from kestrel_sovereign.agent.parts import emit_part
    from kestrel_sovereign.agent.streaming import _build_tool_sentinel
    from kestrel_sovereign.llm.adapter import LLMResponse

    add_convo_calls = []
    agent = _make_agent(add_convo_calls)

    done_sentinel = _build_tool_sentinel("done", "todo_add", ms=5)

    async def mock_stream_with_tool_detection(**kwargs):
        yield "Working. "
        # Tool ran inside the call: emit its part, THEN the adapter surfaces the
        # tool's done sentinel as its own string item.
        emit_part("todo", {"title": "x"})
        yield done_sentinel
        yield "All done."
        yield LLMResponse(content="", tool_calls=[])

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = mock_stream_with_tool_detection
    _bind(agent)

    yielded = []
    async for chunk in agent.process_input_streaming("go", session_id="s7"):
        yielded.append(chunk)

    joined = "".join(yielded)
    done_at = joined.index("\x1eKESTREL:TOOL:")
    part_at = joined.index("\x1eKESTREL:PART:")
    text_at = joined.index("All done.")
    assert done_at < part_at < text_at

    # Persisted: the tool card's seq precedes the part's seq (same position).
    metadata = [c for c in add_convo_calls if c["role"] == "assistant"][0]["metadata"]
    tool_seq = metadata["tool_events"][0]["seq"]
    part_seq = metadata["parts"][0]["seq"]
    assert tool_seq < part_seq


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


@pytest.mark.asyncio
async def test_part_position_uses_utf16_offset_after_emoji():
    """A part after a non-BMP char (emoji) must persist a UTF-16 offset, since
    history reload feeds ``pos`` to JS ``String.slice`` (UTF-16 code units).
    "\U0001F422 " is 2 code points but 3 UTF-16 units (surrogate pair + space)."""
    from kestrel_sovereign.llm.adapter import LLMResponse

    add_convo_calls = []
    agent = _make_agent(add_convo_calls)

    part_sentinel = build_part_sentinel({"type": "notice", "data": {"body": "x"}})

    async def mock_stream_with_tool_detection(**kwargs):
        yield "\U0001F422 "  # turtle emoji + space
        yield part_sentinel
        yield "tail"
        yield LLMResponse(content="", tool_calls=[])

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = mock_stream_with_tool_detection
    _bind(agent)

    async for _ in agent.process_input_streaming("hi", session_id="s5"):
        pass

    metadata = [c for c in add_convo_calls if c["role"] == "assistant"][0]["metadata"]
    # Code-point offset would be 2 ("🐢 "); UTF-16 offset is 3.
    assert metadata["parts"][0]["pos"] == 3


@pytest.mark.asyncio
async def test_post_response_hook_emitted_part_is_persisted():
    """A part emitted by a POST_RESPONSE hook (after streaming) is drained and
    persisted at the end of the text, rather than silently lost."""
    from kestrel_sovereign.agent.parts import emit_part
    from kestrel_sovereign.llm.adapter import LLMResponse

    add_convo_calls = []
    agent = _make_agent(add_convo_calls)

    def _hook(text, sid, **_):
        emit_part("notice", {"body": "from hook"})
        return text

    agent._fire_post_response_hook = AsyncMock(side_effect=_hook)

    async def mock_stream_with_tool_detection(**kwargs):
        yield "Hello there."
        yield LLMResponse(content="", tool_calls=[])

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = mock_stream_with_tool_detection
    _bind(agent)

    async for _ in agent.process_input_streaming("hi", session_id="s6"):
        pass

    metadata = [c for c in add_convo_calls if c["role"] == "assistant"][0]["metadata"]
    assert metadata["parts"] == [
        {"type": "notice", "data": {"body": "from hook"}, "pos": len("Hello there.")}
    ]


@pytest.mark.asyncio
async def test_bang_command_emitted_part_reaches_stream():
    """#1894: a ``!todo`` command delegates to the non-streaming handler, which
    runs OUTSIDE the normal per-turn collector. The streaming layer must bind a
    collector around the delegation so a todo tool's ``emit_part`` still surfaces
    as a live PART sentinel; otherwise ``!todo add`` mutates the graph but the
    typed bubble silently vanishes (``emit_part`` no-ops with no collector)."""
    from kestrel_sovereign.agent.parts import current_part_collector, emit_part

    add_convo_calls = []
    agent = _make_agent(add_convo_calls)

    async def fake_process_input(user_input, *args, **kwargs):
        # Simulate the command handler running the todo_add tool, which emits a
        # typed part. It only lands if the streaming layer bound a collector.
        assert current_part_collector() is not None, "command path bound no collector"
        emit_part("todo", {"title": "ship #1894", "status": "open"}, part_id="t9")
        return "Added todo: ship #1894"

    agent.process_input = fake_process_input
    _bind(agent)

    yielded = []
    async for chunk in agent.process_input_streaming("!todo add ship #1894", session_id="s8"):
        yielded.append(chunk)

    # The command's text result and the emitted todo part both reach the client,
    # with the PART sentinel following the command result.
    assert any("Added todo: ship #1894" in c for c in yielded)
    part_chunks = [c for c in yielded if "\x1eKESTREL:PART:" in c]
    assert len(part_chunks) == 1
    payload = part_chunks[0].split("\x1eKESTREL:PART:", 1)[1].rstrip("\x1e")
    part = json.loads(payload)
    assert part["type"] == "todo"
    assert part["data"] == {"title": "ship #1894", "status": "open"}
    assert part["id"] == "t9"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
