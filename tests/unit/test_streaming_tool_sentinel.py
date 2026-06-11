"""Typed tool-activity sentinels (#1659).

Tool activity now rides the same in-band sentinel channel as REVISE/THINK
(`\\x1eKESTREL:TOOL:{json}\\x1e`) instead of regex-scraped emoji text. These
tests pin the builder, the single-pass strip+extract parser (with position),
and the strip helpers used by the persist + TTS paths.
"""
from kestrel_sovereign.agent.streaming import (
    _build_tool_sentinel,
    _parse_stream_sentinels,
    _strip_and_weld_revise_sentinels,
    _build_revise_sentinel,
    _tool_parts_to_events,
    _stamp_tool_event_positions,
    strip_revise_sentinels,
    is_only_sentinels,
    TOOL_SENTINEL_PREFIX,
)
import json


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
    clean, parts = _parse_stream_sentinels(text)
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
    clean, parts = _parse_stream_sentinels(text)
    assert clean == ""
    assert [p["phase"] for p in parts] == ["start", "done", "error"]
    assert parts[0]["detail"] == "git status"
    assert parts[1]["ms"] == 42
    assert parts[2]["detail"] == "boom"
    # All three sat at offset 0 (no prose between them).
    assert all(p["pos"] == 0 for p in parts)


def test_base_offset_shifts_positions():
    text = "post" + _build_tool_sentinel("start", "t")
    _clean, parts = _parse_stream_sentinels(text, base_offset=100)
    assert parts[0]["pos"] == 100 + len("post")


def test_tool_and_revise_coexist_weld_preserved():
    # A revise sentinel still welds a paragraph break; a tool sentinel does not.
    text = (
        "thinking"
        + _build_revise_sentinel(_M("search", 0))
        + _build_tool_sentinel("start", "search", index=0)
        + "answer"
    )
    clean, parts = _parse_stream_sentinels(text)
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
    _clean, parts = _parse_stream_sentinels(stream)
    events = _tool_parts_to_events(parts)
    assert events[0]["type"] == "start" and events[0]["tool"] == "shell"
    assert events[0]["pos"] == len("ran")
    assert events[1] == {"type": "complete", "tool": "shell", "pos": len("ran"), "ms": 9}
    assert events[2]["type"] == "error" and events[2]["error"] == "boom"


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


class _M:
    """Minimal ToolCallStarted stand-in for _build_revise_sentinel."""
    def __init__(self, name, index, id="id1"):
        self.name = name
        self.index = index
        self.id = id
