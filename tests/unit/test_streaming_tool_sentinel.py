"""Typed tool-activity sentinels (#1659).

Tool activity now rides the same in-band sentinel channel as REVISE/THINK
(`\\x1eKESTREL:TOOL:{json}\\x1e`) instead of regex-scraped emoji text. These
tests pin the builder, the single-pass strip+extract parser (with position),
and the strip helpers used by the persist + TTS paths.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.parts import build_part_sentinel
from kestrel_sovereign.agent.streaming import (
    _build_tool_sentinel,
    _parse_stream_sentinels,
    _rebase_events_onto_persisted_text,
    _strip_and_weld_revise_sentinels,
    _build_revise_sentinel,
    _tool_parts_to_events,
    _stamp_tool_event_positions,
    strip_revise_sentinels,
    is_only_sentinels,
    TOOL_SENTINEL_PREFIX,
)
from tests.unit.test_streaming_typed_parts import _bind, _make_agent


def test_build_tool_sentinel_shape():
    s = _build_tool_sentinel("start", "web_search", index=0)
    assert s.startswith(TOOL_SENTINEL_PREFIX)
    assert s.endswith("\x1e")
    payload = json.loads(s[len(TOOL_SENTINEL_PREFIX):-1])
    assert payload == {"phase": "start", "name": "web_search",
                       "index": 0, "ms": None, "detail": None}


def test_parse_strips_tool_and_records_position():
    text = (
        "Let me search."
        + _build_tool_sentinel("start", "web_search", index=0)
        + "Here is the answer."
    )
    clean, parts, _ = _parse_stream_sentinels(text)
    assert clean == "Let me search.Here is the answer."
    assert len(parts) == 1
    assert parts[0]["phase"] == "start"
    assert parts[0]["name"] == "web_search"
    # pos == offset of the sentinel in the cleaned text (after "Let me search.")
    assert parts[0]["pos"] == len("Let me search.")


def test_parse_preserves_ms_and_detail_and_error():
    text = (
        _build_tool_sentinel("start", "shell", detail="git status")
        + _build_tool_sentinel("done", "shell", ms=42)
        + _build_tool_sentinel("error", "deploy", detail="boom")
    )
    clean, parts, _ = _parse_stream_sentinels(text)
    assert clean == ""
    assert [p["phase"] for p in parts] == ["start", "done", "error"]
    assert parts[0]["detail"] == "git status"
    assert parts[1]["ms"] == 42
    assert parts[2]["detail"] == "boom"
    # All three sat at offset 0 (no prose between them).
    assert all(p["pos"] == 0 for p in parts)


def test_base_offset_shifts_positions():
    text = "post" + _build_tool_sentinel("start", "t")
    _clean, parts, _ = _parse_stream_sentinels(text, base_offset=100)
    assert parts[0]["pos"] == 100 + len("post")


def test_tool_and_revise_coexist_weld_preserved():
    # A revise sentinel still welds a paragraph break; a tool sentinel does not.
    text = (
        "thinking"
        + _build_revise_sentinel(_M("search", 0))
        + _build_tool_sentinel("start", "search", index=0)
        + "answer"
    )
    clean, parts, _ = _parse_stream_sentinels(text)
    assert clean == "thinking\n\nanswer"  # revise welded; tool stripped, no weld
    assert len(parts) == 1
    assert parts[0]["name"] == "search"


def test_strip_helpers_remove_tool_sentinels():
    chunk = "say" + _build_tool_sentinel("done", "x", ms=1) + "more"
    assert strip_revise_sentinels(chunk) == "saymore"
    assert _strip_and_weld_revise_sentinels(chunk) == "saymore"


def test_is_only_sentinels():
    assert is_only_sentinels(_build_tool_sentinel("start", "x")) is True
    assert is_only_sentinels("hello") is False
    assert is_only_sentinels("") is False
    assert is_only_sentinels(_build_tool_sentinel("start", "x") + "hi") is False


def test_tool_parts_to_events_maps_to_metadata_shape():
    # Round-trip: sentinels → parts (with pos) → persisted tool_events shape.
    stream = (
        "ran"
        + _build_tool_sentinel("start", "shell", detail="ls")
        + _build_tool_sentinel("done", "shell", ms=9)
        + _build_tool_sentinel("error", "deploy", detail="boom")
    )
    _clean, parts, _ = _parse_stream_sentinels(stream)
    events = _tool_parts_to_events(parts)
    assert events[0]["type"] == "start" and events[0]["tool"] == "shell"
    assert events[0]["pos"] == len("ran")
    # #1914: the shared wire-order ``seq`` is carried through (start=0, done=1,
    # error=2) so reload can interleave same-position tool cards and parts.
    assert events[1] == {
        "type": "complete", "tool": "shell", "pos": len("ran"), "ms": 9, "seq": 1,
    }
    assert events[2]["type"] == "error" and events[2]["error"] == "boom"
    assert [e["seq"] for e in events] == [0, 1, 2]


def test_stamp_positions_matches_by_phase_and_name():
    # Sentinels arrive start-first then batch-terminal (different order than
    # the by-ref tool_events); matching by (phase,name) keeps them aligned.
    tool_events = [
        {"type": "start", "tool": "shell"},
        {"type": "complete", "tool": "shell"},
        {"type": "start", "tool": "git"},
        {"type": "complete", "tool": "git"},
    ]
    parts = [
        {"phase": "start", "name": "shell", "pos": 10},
        {"phase": "start", "name": "git", "pos": 10},
        {"phase": "done", "name": "shell", "pos": 10},
        {"phase": "done", "name": "git", "pos": 10},
    ]
    _stamp_tool_event_positions(tool_events, parts)
    assert all(ev["pos"] == 10 for ev in tool_events)


def test_stamp_positions_startless_error_keeps_own_pos():
    # A follow-up-timeout error for "llm" has no matching start — it must keep
    # its OWN sentinel position, not inherit the previous tool's.
    tool_events = [
        {"type": "start", "tool": "shell"},
        {"type": "complete", "tool": "shell"},
        {"type": "error", "tool": "llm"},
    ]
    parts = [
        {"phase": "start", "name": "shell", "pos": 5},
        {"phase": "done", "name": "shell", "pos": 5},
        {"phase": "error", "name": "llm", "pos": 40},
    ]
    _stamp_tool_event_positions(tool_events, parts)
    assert [ev["pos"] for ev in tool_events] == [5, 5, 40]


def test_rebase_event_in_retracted_pre_tool_text_pins_to_start():
    events = [{"type": "start", "tool": "lookup", "pos": 4}]

    rebased = _rebase_events_onto_persisted_text(events, 10, "answer")

    assert rebased[0]["pos"] == 0
    assert events[0]["pos"] == 4


async def _persist_multi_iteration_tool_turn(*, with_part=False, emoji=False):
    """Drive the orchestrated-tool persistence branch with position sentinels."""
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    persisted = []
    agent = _make_agent(persisted)
    pre_tool = "I need to inspect this."
    boundary = (
        "Only one permission is missing: administration: read.\n\n"
        if not emoji else "🐢 Result ready.\n\n"
    )
    tail = "The header names it directly."

    async def llm_stream(**kwargs):
        yield pre_tool
        yield ToolCallStarted(index=0, id="tc1", name="strategy_resolve_blocker")
        yield LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc1", name="strategy_resolve_blocker", arguments={}),
            ],
        )

    async def orchestrator_stream(tool_events, **kwargs):
        tool_events.extend([
            {"type": "start", "tool": "strategy_resolve_blocker"},
            {"type": "complete", "tool": "strategy_resolve_blocker", "ms": 7},
        ])
        yield boundary
        yield _build_tool_sentinel("start", "strategy_resolve_blocker", index=0)
        yield _build_tool_sentinel(
            "done", "strategy_resolve_blocker", index=0, ms=7,
        )
        if with_part:
            yield build_part_sentinel({"type": "notice", "data": {"body": "same"}})
        yield tail

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = llm_stream
    agent._handle_orchestrator_response_streaming = orchestrator_stream
    _bind(agent)

    async for _ in agent.process_input_streaming("inspect", session_id="multi"):
        pass

    assistant = [row for row in persisted if row["role"] == "assistant"]
    assert len(assistant) == 1
    return assistant[0], boundary


@pytest.mark.asyncio
async def test_tool_positions_without_parts_index_persisted_post_tool_content():
    """Non-empty retracted prose must not drift card offsets on persistence."""
    assistant, boundary = await _persist_multi_iteration_tool_turn()
    events = assistant["metadata"]["tool_events"]

    assert assistant["content"] == boundary + "The header names it directly."
    assert events
    for event in events:
        pos = event["pos"]
        assert 0 <= pos <= len(assistant["content"])
        assert assistant["content"][:pos] == boundary


@pytest.mark.asyncio
async def test_tool_positions_are_identical_with_and_without_component_parts():
    without_parts, _ = await _persist_multi_iteration_tool_turn(with_part=False)
    with_parts, _ = await _persist_multi_iteration_tool_turn(with_part=True)

    assert [e["pos"] for e in without_parts["metadata"]["tool_events"]] == [
        e["pos"] for e in with_parts["metadata"]["tool_events"]
    ]


@pytest.mark.asyncio
async def test_tool_positions_use_utf16_without_component_parts():
    assistant, boundary = await _persist_multi_iteration_tool_turn(emoji=True)

    expected = len(boundary.encode("utf-16-le")) // 2
    assert [e["pos"] for e in assistant["metadata"]["tool_events"]] == [
        expected,
        expected,
    ]


@pytest.mark.asyncio
async def test_inline_executed_tool_positions_rebase_without_component_parts():
    """Codex inline execution removes ``inline_pre_len`` from card offsets."""
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.llm.adapter import LLMResponse

    persisted = []
    agent = _make_agent(persisted)
    agent.observability_store.log_tool_dispatch = AsyncMock()
    agent._make_inline_tool_executor = MagicMock(return_value=None)
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent.context_manager.build_context.return_value.degraded_mode = False
    pre_tool = "I will inspect it."
    boundary = "Inspection finished.\n\n"

    async def stream(**kwargs):
        yield pre_tool
        yield ToolCallStarted(index=0, id="tc-inline", name="inspect")
        yield boundary
        yield _build_tool_sentinel("start", "inspect", index=0)
        yield _build_tool_sentinel("done", "inspect", index=0, ms=3)
        yield "The setting is correct."
        response = LLMResponse(content="", tool_calls=None)
        response.executed_tool_calls = [
            {
                "id": "tc-inline",
                "name": "inspect",
                "arguments": {},
                "result": {"ok": True},
            },
        ]
        yield response

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = stream
    _bind(agent)

    async for _ in agent.process_input_streaming("inspect", session_id="inline"):
        pass

    assistant = [row for row in persisted if row["role"] == "assistant"][0]
    assert assistant["content"] == boundary + "The setting is correct."
    assert [e["pos"] for e in assistant["metadata"]["tool_events"]] == [
        len(boundary),
        len(boundary),
    ]
    agent.observability_store.log_tool_dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_codex_native_tool_positions_use_utf16_without_component_parts():
    """The no-tool-call branch still converts native sentinel offsets."""
    from kestrel_sovereign.llm.adapter import LLMResponse

    persisted = []
    agent = _make_agent(persisted)
    boundary = "🐢 Native result.\n\n"

    async def stream(**kwargs):
        yield boundary
        yield _build_tool_sentinel("start", "web_search", index=0)
        yield _build_tool_sentinel("done", "web_search", index=0, ms=4)
        yield "Details follow."
        yield LLMResponse(content="", tool_calls=[])

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = stream
    _bind(agent)

    async for _ in agent.process_input_streaming("search", session_id="native"):
        pass

    assistant = [row for row in persisted if row["role"] == "assistant"][0]
    expected = len(boundary.encode("utf-16-le")) // 2
    assert assistant["content"] == boundary + "Details follow."
    assert [e["pos"] for e in assistant["metadata"]["tool_events"]] == [
        expected,
        expected,
    ]


class _M:
    """Minimal ToolCallStarted stand-in for _build_revise_sentinel."""
    def __init__(self, name, index, id="id1"):
        self.name = name
        self.index = index
        self.id = id
