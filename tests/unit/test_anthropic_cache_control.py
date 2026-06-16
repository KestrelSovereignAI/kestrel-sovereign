"""Unit tests for Anthropic cache_control markers (issue #705).

Covers both the marker-placement helpers and the full
``AnthropicAdapter.get_response`` integration with a mocked SDK client.
Also verifies ``ClaudeMaxAdapter`` inherits the behavior unchanged.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.anthropic_adapter import (
    CACHE_CONTROL_EPHEMERAL,
    CLAUDE_CODE_IDENTITY,
    AnthropicAdapter,
    _attach_cache_control,
    _messages_with_penultimate_cache_marker,
    _system_as_cacheable_array,
    _tools_with_final_cache_marker,
)
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def test_attach_cache_control_returns_copy():
    """Helper must not mutate input — important when callers pass shared
    structures (tool schemas are often reused across requests).
    """
    original = {"type": "text", "text": "hello"}
    result = _attach_cache_control(original)
    assert result is not original
    assert "cache_control" not in original
    assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL


def test_system_as_cacheable_array_wraps_string():
    """Plain-string system → single-block array with cache_control on the
    sole block.  Preserves the original text exactly.
    """
    blocks = _system_as_cacheable_array("You are a helpful assistant.")
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "You are a helpful assistant."
    assert blocks[0]["cache_control"] == CACHE_CONTROL_EPHEMERAL


def test_tools_with_final_cache_marker_marks_last_only():
    """Only the LAST tool gets a marker — one breakpoint covers the entire
    tools block for cache purposes.
    """
    tools = [
        {"name": "t1", "description": "first"},
        {"name": "t2", "description": "second"},
        {"name": "t3", "description": "third"},
    ]
    marked = _tools_with_final_cache_marker(tools)
    assert "cache_control" not in marked[0]
    assert "cache_control" not in marked[1]
    assert marked[2]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    # Input not mutated.
    assert "cache_control" not in tools[2]


def test_tools_with_final_cache_marker_empty_list_passthrough():
    """No tools → returned unchanged (no empty-list corner case)."""
    assert _tools_with_final_cache_marker([]) == []


def test_messages_with_penultimate_marker_marks_second_to_last():
    """Last-before-current-user gets marked.  The current user turn (the
    final entry) NEVER gets cache_control — it's new content every call.
    """
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "response 1"},
        {"role": "user", "content": "turn 2 (current)"},
    ]
    marked = _messages_with_penultimate_cache_marker(messages)
    # Final message (current user turn) untouched.
    assert marked[-1] == {"role": "user", "content": "turn 2 (current)"}
    # Penultimate (assistant response 1) got cache_control.
    penult = marked[-2]
    assert penult["role"] == "assistant"
    assert isinstance(penult["content"], list)
    assert penult["content"][-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    assert penult["content"][-1]["text"] == "response 1"


def test_messages_with_penultimate_marker_no_history():
    """Single message (== current user turn only) → no marker inserted
    anywhere.  There's no history to cache.
    """
    messages = [{"role": "user", "content": "turn 1 (current)"}]
    marked = _messages_with_penultimate_cache_marker(messages)
    assert marked == messages
    assert "cache_control" not in marked[0]


def test_messages_with_penultimate_marker_list_content_preserved():
    """If the penultimate message already has content-list format (e.g.
    a tool-result response), the marker attaches to the final block
    without dropping earlier blocks.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
            ],
        },
        {"role": "user", "content": "current"},
    ]
    marked = _messages_with_penultimate_cache_marker(messages)
    penult_content = marked[-2]["content"]
    assert len(penult_content) == 2
    assert "cache_control" not in penult_content[0]
    assert penult_content[1]["cache_control"] == CACHE_CONTROL_EPHEMERAL


def test_messages_with_trailing_system_marks_last_stable_turn():
    """A newly appended inline system turn is fresh input, so the marker
    stays on the stable assistant response before current-user + system.
    """
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2 current"},
        {"role": "system", "content": [{"type": "text", "text": "auto on"}]},
    ]

    marked = _messages_with_penultimate_cache_marker(
        messages,
        volatile_tail_size=2,
    )

    assert marked[-1] == messages[-1]
    assert marked[-2] == messages[-2]
    stable = marked[1]["content"]
    assert isinstance(stable, list)
    assert stable[-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL


def test_messages_with_stable_inline_system_compounds_to_prior_marker():
    """On the next turn, the latest assistant marker includes the stable
    inline system in its prefix while the second marker still queries the
    breakpoint written before the system was introduced.
    """
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "system", "content": [{"type": "text", "text": "auto on"}]},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3 current"},
    ]

    marked = _messages_with_penultimate_cache_marker(messages)

    assert marked[4]["content"][-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    assert marked[1]["content"][-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    assert marked[3] == messages[3]


def test_convert_messages_preserves_supported_inline_system():
    adapter = AnthropicAdapter()
    messages = [
        {"role": "system", "content": "prefix"},
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "operator fact"},
    ]

    converted, system = adapter._convert_messages_to_anthropic(
        messages,
        keep_trailing_system=True,
        model="claude-opus-4-8-20260501",
    )

    assert system == "prefix"
    assert converted == [
        {"role": "user", "content": "q1"},
        {
            "role": "system",
            "content": [{"type": "text", "text": "operator fact"}],
        },
    ]


def test_convert_messages_demotes_unsupported_inline_system(caplog):
    adapter = AnthropicAdapter()
    messages = [
        {"role": "system", "content": "prefix"},
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "operator fact"},
    ]

    with caplog.at_level("INFO"):
        converted, system = adapter._convert_messages_to_anthropic(
            messages,
            keep_trailing_system=True,
            model="claude-sonnet-4-6-20260101",
        )

    assert converted == [{"role": "user", "content": "q1"}]
    assert system == "prefix\n\noperator fact"
    assert "demoting non-leading system" in caplog.text


def test_inline_system_validator_rejects_illegal_positions():
    adapter = AnthropicAdapter()

    with pytest.raises(ValueError, match="cannot be the first"):
        adapter._validate_inline_system_messages([
            {"role": "system", "content": "bad"},
            {"role": "user", "content": "q"},
        ])

    with pytest.raises(ValueError, match="cannot be consecutive"):
        adapter._validate_inline_system_messages([
            {"role": "user", "content": "q"},
            {"role": "system", "content": "one"},
            {"role": "system", "content": "two"},
        ])

    with pytest.raises(ValueError, match="must be last"):
        adapter._validate_inline_system_messages([
            {"role": "user", "content": "q"},
            {"role": "system", "content": "operator"},
            {"role": "user", "content": "bad next"},
        ])

    with pytest.raises(ValueError, match="must immediately follow"):
        adapter._validate_inline_system_messages([
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1"}],
            },
            {"role": "system", "content": "bad"},
        ])


# ---------------------------------------------------------------------------
# Adapter-level _apply_cache_control
# ---------------------------------------------------------------------------


def test_apply_cache_control_marks_all_three_positions():
    """System + tools + >2-message history → all three canonical marker
    positions set.  Total markers: 3, well under Anthropic's cap of 4.
    """
    api_params = {
        "model": "claude-sonnet-4-5",
        "system": "You are helpful.",
        "tools": [{"name": "t1"}, {"name": "t2"}],
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2 (current)"},
        ],
        "max_tokens": 100,
    }
    result = AnthropicAdapter._apply_cache_control(api_params)

    # system → array with marker on the single block
    assert isinstance(result["system"], list)
    assert result["system"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL

    # tools → last tool marked
    assert result["tools"][-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    assert "cache_control" not in result["tools"][0]

    # messages → second-to-last marked
    penult_content = result["messages"][-2]["content"]
    assert isinstance(penult_content, list)
    assert penult_content[-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    # Current user turn NOT marked.
    assert result["messages"][-1]["content"] == "q2 (current)"


def test_apply_cache_control_counts_at_most_4_markers():
    """Hard invariant: however the input is shaped, we never emit more
    than 4 cache_control markers.  Anthropic rejects requests that
    exceed the breakpoint cap.
    """
    api_params = {
        "system": "sys",
        "tools": [{"name": f"t{i}"} for i in range(10)],
        "messages": [
            {"role": "user", "content": f"m{i}"}
            for i in range(20)
        ],
    }
    result = AnthropicAdapter._apply_cache_control(api_params)

    def count_markers(obj):
        if isinstance(obj, dict):
            n = 1 if obj.get("cache_control") == CACHE_CONTROL_EPHEMERAL else 0
            return n + sum(count_markers(v) for v in obj.values())
        if isinstance(obj, list):
            return sum(count_markers(v) for v in obj)
        return 0

    assert count_markers(result) <= 4


def test_apply_cache_control_does_not_mutate_input():
    """Caller's api_params dict must be unchanged after the call.  Retry
    logic re-sends the original request; mutation would compound markers
    on retry and break the cap.
    """
    api_params = {
        "system": "sys",
        "tools": [{"name": "t1"}],
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
    }
    import copy
    snapshot = copy.deepcopy(api_params)
    AnthropicAdapter._apply_cache_control(api_params)
    assert api_params == snapshot


def test_apply_cache_control_no_system_no_tools_short_history():
    """Minimum viable input (bare system + single-turn conversation):
    still produces a valid request; the marker on system covers the
    small prefix.  History marker skipped because there's no history.
    """
    api_params = {
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = AnthropicAdapter._apply_cache_control(api_params)
    # Single-message conversation: no history to mark.
    assert result["messages"][0].get("content") == "hi"
    # No system / tools to mark either.
    assert "system" not in result


def test_apply_cache_control_empty_system_string_no_array_upgrade():
    """Empty system string stays a string (or is removed upstream).  The
    helper must not upgrade `""` into a content-block array — Anthropic
    would reject the empty text block.
    """
    api_params = {"system": "", "messages": [{"role": "user", "content": "hi"}]}
    result = AnthropicAdapter._apply_cache_control(api_params)
    assert result["system"] == ""


# ---------------------------------------------------------------------------
# Full AnthropicAdapter.get_response integration (SDK mocked)
# ---------------------------------------------------------------------------


async def _call_anthropic_and_capture(messages: List[Dict[str, Any]]) -> Dict:
    """Run AnthropicAdapter.get_response with a mocked SDK client and
    return the kwargs sent to messages.create.
    """
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(type="text", text="ok")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=1),
        )
    )
    adapter = AnthropicAdapter()
    await adapter.get_response(
        client=fake_client,
        model="claude-sonnet-4-5-20250929",
        messages=messages,
    )
    return fake_client.messages.create.call_args.kwargs


@pytest.mark.asyncio
async def test_get_response_sends_cacheable_system_block():
    """End to end: the kwargs messages.create receives have the system
    parameter in cache-block form with the cache_control marker.
    """
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "hi"},
    ]
    captured = await _call_anthropic_and_capture(messages)
    system = captured.get("system")
    assert isinstance(system, list)
    assert system[-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL


@pytest.mark.asyncio
async def test_get_response_marks_history_before_current_user_turn():
    """Multi-turn conversation → penultimate message is cache-marked,
    current user turn is not.
    """
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2 (current)"},
    ]
    captured = await _call_anthropic_and_capture(messages)
    anthropic_messages = captured["messages"]
    # Penultimate (assistant a1) gets marked.
    penult_content = anthropic_messages[-2]["content"]
    assert isinstance(penult_content, list)
    assert penult_content[-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    # Current user turn stays as-is (string content, no marker).
    assert anthropic_messages[-1] == {
        "role": "user",
        "content": "q2 (current)",
    }


@pytest.mark.asyncio
async def test_get_response_first_turn_only_marks_system():
    """First turn of a brand-new conversation (system + one user) marks
    system only — no history to mark, no tools.  That's 1 of 4
    breakpoints used; Anthropic's cache docs note this is the expected
    minimum-viable cache config.
    """
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "first turn"},
    ]
    captured = await _call_anthropic_and_capture(messages)
    # System marked.
    assert isinstance(captured["system"], list)
    assert captured["system"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    # No tools in this request.
    assert "tools" not in captured or not captured.get("tools")
    # Single message, no history marker.
    assert len(captured["messages"]) == 1
    assert captured["messages"][0]["content"] == "first turn"


# ---------------------------------------------------------------------------
# Subclass inheritance
# ---------------------------------------------------------------------------


async def _call_claude_max_and_capture(messages):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(type="text", text="ok")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=1),
        )
    )
    adapter = ClaudeMaxAdapter()
    await adapter.get_response(
        client=fake_client,
        model="claude-sonnet-4-5-20250929",
        messages=messages,
    )
    return fake_client.messages.create.call_args.kwargs


@pytest.mark.asyncio
async def test_claude_max_inherits_cache_control_behavior():
    """ClaudeMaxAdapter subclasses AnthropicAdapter and inherits the cache
    markers. The OAuth route additionally prepends the Claude Code identity
    block (required by Anthropic's subscription endpoint), so the cache
    breakpoint lands on the TRAILING real-system block — which still covers
    the whole system prefix, identity included. History markers are unchanged.
    See tests/unit/test_anthropic_oauth_shaping.py for the identity-shaping
    assertions.
    """
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2 (current)"},
    ]
    captured = await _call_claude_max_and_capture(messages)
    assert isinstance(captured["system"], list)
    # Identity prepended first (no marker); real system trailing (marked).
    assert captured["system"][0]["text"] == CLAUDE_CODE_IDENTITY
    assert captured["system"][-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    penult_content = captured["messages"][-2]["content"]
    assert isinstance(penult_content, list)
    assert penult_content[-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
