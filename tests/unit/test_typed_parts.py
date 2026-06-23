"""Typed component parts (#1914).

Parts are typed, structured components the agent stream renders as their own
first-class console bubbles (todo cards, citations, A2A artifacts, …). They ride
the same in-band sentinel channel as TOOL/THINK/REVISE:
``\\x1eKESTREL:PART:{json}\\x1e``. These tests pin the emit/collect side
(``parts.py``), the wire builder, and the streaming parser's part strip +
position extraction — including the roundtrip that proves the builder and the
parser agree on the wire contract.
"""
import json

import pytest

from kestrel_sovereign.agent.parts import (
    MAX_PART_PAYLOAD_BYTES,
    PART_SENTINEL_PREFIX,
    PART_SENTINEL_SUFFIX,
    build_part_sentinel,
    drain_parts,
    emit_part,
    part_collector,
)
from kestrel_sovereign.agent.streaming import _parse_stream_sentinels


# --------------------------------------------------------------------------
# emit_part / collector / drain
# --------------------------------------------------------------------------

def test_emit_part_no_collector_is_noop():
    # Outside a turn there is no active collector: emit is a no-op (False),
    # never an error — a tool that emits a part still works off the stream path.
    assert emit_part("todo", {"title": "x"}) is False
    assert drain_parts() == []


def test_emit_and_drain_roundtrip():
    with part_collector():
        assert emit_part("todo", {"title": "buy milk"}, part_id="t1") is True
        assert emit_part("notice", {"body": "hi"}) is True
        drained = drain_parts()
        assert drained == [
            {"type": "todo", "data": {"title": "buy milk"}, "id": "t1"},
            {"type": "notice", "data": {"body": "hi"}},
        ]
        # Drained buffer is cleared — a second drain yields nothing.
        assert drain_parts() == []


def test_collector_resets_between_turns():
    with part_collector():
        emit_part("todo", {"a": 1})
    # Outside the with-block the collector is gone; the prior turn's part must
    # not leak into the next.
    assert drain_parts() == []
    with part_collector():
        assert drain_parts() == []


def test_emit_part_rejects_invalid_type():
    with part_collector():
        assert emit_part("", {"x": 1}) is False
        assert emit_part(None, {"x": 1}) is False  # type: ignore[arg-type]
        # A control char (incl. the 0x1e framing byte) in the type is rejected.
        assert emit_part("to\x1edo", {"x": 1}) is False
        assert emit_part("x" * 65, {"x": 1}) is False
        assert drain_parts() == []


def test_emit_part_rejects_non_serializable():
    with part_collector():
        assert emit_part("todo", {"bad": object()}) is False
        assert drain_parts() == []


def test_emit_part_rejects_oversized():
    with part_collector():
        big = "z" * (MAX_PART_PAYLOAD_BYTES + 100)
        assert emit_part("todo", {"body": big}) is False
        assert drain_parts() == []


def test_emit_part_rejects_non_finite_numbers():
    # NaN/Infinity serialize to non-standard JSON tokens the browser's
    # JSON.parse rejects, which would make the live component silently vanish.
    # Reject them at emit time instead.
    with part_collector():
        assert emit_part("chart", {"v": float("nan")}) is False
        assert emit_part("chart", {"v": float("inf")}) is False
        assert emit_part("chart", {"v": float("-inf")}) is False
        assert drain_parts() == []


def test_build_part_sentinel_rejects_non_finite():
    assert build_part_sentinel({"type": "chart", "data": {"v": float("inf")}}) is None


# --------------------------------------------------------------------------
# build_part_sentinel — wire format
# --------------------------------------------------------------------------

def test_build_part_sentinel_shape():
    part = {"type": "todo", "data": {"title": "x"}, "id": "t1"}
    s = build_part_sentinel(part)
    assert s.startswith(PART_SENTINEL_PREFIX)
    assert s.endswith(PART_SENTINEL_SUFFIX)
    payload = json.loads(s[len(PART_SENTINEL_PREFIX):-len(PART_SENTINEL_SUFFIX)])
    assert payload == part


def test_build_part_sentinel_escapes_record_separator():
    # A 0x1e smuggled into data must be JSON-escaped so it can't break framing:
    # the serialized sentinel contains exactly one literal RS on each side.
    part = {"type": "notice", "data": {"body": "a\x1eb"}}
    s = build_part_sentinel(part)
    assert s.count("\x1e") == 2  # the two bookends only
    payload = json.loads(s[len(PART_SENTINEL_PREFIX):-len(PART_SENTINEL_SUFFIX)])
    assert payload["data"]["body"] == "a\x1eb"


def test_build_part_sentinel_oversized_returns_none():
    part = {"type": "todo", "data": {"body": "z" * (MAX_PART_PAYLOAD_BYTES + 100)}}
    assert build_part_sentinel(part) is None


# --------------------------------------------------------------------------
# _parse_stream_sentinels — part strip + position
# --------------------------------------------------------------------------

def test_parser_strips_part_and_extracts_with_position():
    sentinel = build_part_sentinel({"type": "todo", "data": {"title": "x"}, "id": "t1"})
    text = "before " + sentinel + "after"
    clean, tool_parts, parts = _parse_stream_sentinels(text)
    assert clean == "before after"  # sentinel removed from visible prose
    assert tool_parts == []
    assert len(parts) == 1
    assert parts[0]["type"] == "todo"
    assert parts[0]["data"] == {"title": "x"}
    assert parts[0]["id"] == "t1"
    # pos == clean-text offset where the sentinel sat ("before " == 7 chars).
    assert parts[0]["pos"] == len("before ")


def test_parser_base_offset_applied_to_parts():
    sentinel = build_part_sentinel({"type": "notice", "data": 1})
    clean, _tools, parts = _parse_stream_sentinels("ab" + sentinel, base_offset=100)
    assert clean == "ab"
    assert parts[0]["pos"] == 100 + 2


def test_parser_drops_malformed_part():
    # A PART sentinel whose payload isn't a dict-with-string-type is dropped,
    # and the bytes never reach the visible prose.
    bad = PART_SENTINEL_PREFIX + "not json{" + PART_SENTINEL_SUFFIX
    no_type = PART_SENTINEL_PREFIX + json.dumps({"data": 1}) + PART_SENTINEL_SUFFIX
    clean, _tools, parts = _parse_stream_sentinels("x" + bad + no_type + "y")
    assert clean == "xy"
    assert parts == []


def test_parser_handles_tool_and_part_together():
    from kestrel_sovereign.agent.streaming import _build_tool_sentinel

    tool = _build_tool_sentinel("start", "web_search", index=0)
    part = build_part_sentinel({"type": "todo", "data": {"t": 1}})
    text = "a" + tool + "b" + part + "c"
    clean, tool_parts, parts = _parse_stream_sentinels(text)
    assert clean == "abc"
    assert len(tool_parts) == 1 and tool_parts[0]["name"] == "web_search"
    assert len(parts) == 1 and parts[0]["type"] == "todo"
    assert parts[0]["pos"] == 2  # after "ab"


def test_parser_stamps_shared_wire_order_seq():
    # A shared monotonic ``seq`` across TOOL and PART sentinels records wire
    # order, so the reload renderer can interleave same-position items. Here a
    # tool-done, a PART, and a tool-start all sit at offset 0 (no prose between).
    from kestrel_sovereign.agent.streaming import _build_tool_sentinel

    done = _build_tool_sentinel("done", "todo_add", ms=5)
    part = build_part_sentinel({"type": "todo", "data": {"t": 1}})
    start = _build_tool_sentinel("start", "web_search", index=0)
    clean, tool_parts, parts = _parse_stream_sentinels(done + part + start)
    assert clean == ""
    # Wire order: done (0) < part (1) < start (2), all at pos 0.
    assert [tp["seq"] for tp in tool_parts] == [0, 2]
    assert parts[0]["seq"] == 1
    assert all(tp["pos"] == 0 for tp in tool_parts) and parts[0]["pos"] == 0


def test_roundtrip_emit_to_parse():
    # End-to-end wire contract: a part emitted + built into a sentinel is
    # recovered intact (minus the added ``pos``) by the streaming parser.
    with part_collector():
        emit_part("todo", {"title": "ship #1914"}, part_id="z9")
        (part,) = drain_parts()
    sentinel = build_part_sentinel(part)
    clean, _tools, parsed = _parse_stream_sentinels("p" + sentinel + "q")
    assert clean == "pq"
    # ``pos`` (clean-text offset) and ``seq`` (wire order) are stamped by the
    # parser; the rest must round-trip intact.
    recovered = {k: v for k, v in parsed[0].items() if k not in ("pos", "seq")}
    assert recovered == {"type": "todo", "data": {"title": "ship #1914"}, "id": "z9"}


# --------------------------------------------------------------------------
# contextvar propagation across nested async generators
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collector_visible_across_nested_async_generators():
    """The collector is set by an OUTER async generator (mirroring
    ``process_input_streaming``'s ``with part_collector()``) and must be visible
    to ``emit_part`` called deep inside an INNER async generator (a tool
    dispatched by the orchestrator) and to ``drain_parts`` in between — the core
    assumption behind the streaming→orchestrator wiring.
    """
    drained_seen = []

    async def inner_tool_stream():
        # Stands in for the orchestrator + a tool: emit a part, then later
        # drain it (as the orchestrator does after the tool batch).
        emit_part("todo", {"title": "from inner"})
        yield "chunk-1"
        drained_seen.extend(drain_parts())
        yield "chunk-2"

    async def outer_turn():
        with part_collector():
            async for chunk in inner_tool_stream():
                yield chunk

    chunks = [c async for c in outer_turn()]
    assert chunks == ["chunk-1", "chunk-2"]
    assert drained_seen == [{"type": "todo", "data": {"title": "from inner"}}]
    # After the outer turn exits, the collector is reset — no leak.
    assert drain_parts() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
