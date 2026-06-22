"""Revise-boundary weld — server side (#1547).

The chat client strips in-band revise/think sentinels and welds a
paragraph break wherever removing a sentinel would otherwise glue two
non-whitespace prose segments ("...let me check.The answer is 42.").
The persisted assistant turn must match what the user saw, so the server
reproduces the same strip-and-weld. These tests pin:

  * ``_strip_and_weld_revise_sentinels`` — the string transform applied
    to the post-tool half before persistence.
  * the inline-execution (codex app-server) path — where a
    ``ToolCallStarted`` marker records the boundary as ``\\n\\n`` in the
    accumulated visible text so plain-text-then-tool-then-plain-text
    doesn't glue.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.agent.streaming import (
    StreamingMixin,
    _strip_and_weld_revise_sentinels,
    REVISE_SENTINEL_PREFIX,
    REVISE_SENTINEL_SUFFIX,
    THINKING_SENTINEL_PREFIX,
    THINKING_SENTINEL_SUFFIX,
)


def _revise(payload='{"index":0,"tool_call_id":"t","tool_name":"x"}'):
    return f"{REVISE_SENTINEL_PREFIX}{payload}{REVISE_SENTINEL_SUFFIX}"


def _think(content="reasoning"):
    payload = '{"content":"%s","provider":"openai"}' % content
    return f"{THINKING_SENTINEL_PREFIX}{payload}{THINKING_SENTINEL_SUFFIX}"


class TestStripAndWeld:
    def test_plain_text_is_untouched(self):
        assert _strip_and_weld_revise_sentinels("just prose") == "just prose"

    def test_revise_between_nonspace_welds_paragraph(self):
        text = "Let me check." + _revise() + "The answer is 42."
        assert _strip_and_weld_revise_sentinels(text) == (
            "Let me check.\n\nThe answer is 42."
        )

    def test_existing_whitespace_seam_not_doubled(self):
        # Post half already opens with a newline — the weld is a no-op.
        text = "Checking now." + _revise() + "\nDone."
        assert _strip_and_weld_revise_sentinels(text) == "Checking now.\nDone."

    def test_leading_whitespace_on_right_not_doubled(self):
        text = "Checking now." + _revise() + " Done."
        assert _strip_and_weld_revise_sentinels(text) == "Checking now. Done."

    def test_thinking_sentinel_stripped_without_weld(self):
        # Thinking is reasoning removed from the stream; it is NOT a prose
        # boundary, so the surrounding visible text joins verbatim.
        text = "Answer part one " + _think() + "and part two."
        assert _strip_and_weld_revise_sentinels(text) == (
            "Answer part one and part two."
        )

    def test_multiple_revise_sentinels_each_weld(self):
        text = "before" + _revise() + "between" + _revise() + "after"
        assert _strip_and_weld_revise_sentinels(text) == (
            "before\n\nbetween\n\nafter"
        )

    def test_revise_before_tool_marker_welds(self):
        # The orchestrator emits "🔧 Calling ..." right after a follow-up
        # revise sentinel — the weld lands before the marker line.
        text = "Working on it." + _revise() + "🔧 Calling github...\n"
        assert _strip_and_weld_revise_sentinels(text) == (
            "Working on it.\n\n🔧 Calling github...\n"
        )

    def test_unterminated_sentinel_tail_dropped(self):
        text = "visible" + REVISE_SENTINEL_PREFIX + "no-close-here"
        assert _strip_and_weld_revise_sentinels(text) == "visible"

    def test_no_wire_bytes_survive(self):
        text = "a" + _revise() + "b" + _think() + "c"
        out = _strip_and_weld_revise_sentinels(text)
        assert "\x1e" not in out
        assert "KESTREL:REVISE" not in out
        assert "KESTREL:THINK" not in out


def _passthrough():
    class _Ctx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    return _Ctx()


def _build_inline_agent(persisted):
    """A mock agent wired for the inline-execution (codex) path: the LLM
    stream yields plain text, a ToolCallStarted marker, then more plain
    text, and the final LLMResponse carries executed_tool_calls (tools
    already ran inside the call) rather than tool_calls."""
    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: persisted.append(
            {"role": role, "content": content, **kw}
        )
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "did"
    agent.extension = None
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._maybe_audit = AsyncMock()
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = None
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent._emit_revising_event = AsyncMock()
    agent._make_inline_tool_executor = MagicMock(return_value=None)
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"

    context_result = MagicMock()
    context_result.system_prompt = "system"
    context_result.dynamic_user_context = "ctx"
    context_result.messages = []
    context_result.degraded_mode = False
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=context_result)

    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()

    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    return agent


@pytest.mark.asyncio
async def test_inline_execution_path_welds_pre_and_post_tool_prose():
    from kestrel_sovereign.llm.adapter import LLMResponse

    persisted = []
    agent = _build_inline_agent(persisted)

    async def stream(**kw):
        yield "Let me check that."
        # Inline tools execute inside the call; the marker fires for the
        # honesty layer but the tool is NOT re-dispatched.
        from kestrel_sdk.llm import ToolCallStarted
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield "The answer is 42."
        resp = LLMResponse(content="", tool_calls=None)
        resp.executed_tool_calls = [
            {"id": "tc1", "name": "github", "arguments": {}, "result": {"ok": True}}
        ]
        yield resp

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    chunks = []
    async for c in agent.process_input_streaming("go", session_id="s"):
        chunks.append(c)

    assistant_rows = [r for r in persisted if r["role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == "The answer is 42."
    assert assistant_rows[0]["metadata"]["pre_tool_reasoning"] == {
        "content": "Let me check that.",
        "seam": "\n\n",
        "context_replay": "Let me check that.\n\nThe answer is 42.",
    }
    assert "\x1e" not in assistant_rows[0]["content"]


@pytest.mark.asyncio
async def test_cancel_right_after_marker_persists_no_dangling_boundary():
    """Codex review: the revise boundary is armed lazily. If the user
    cancels after the marker but before any post-tool text arrives, the
    persisted turn must NOT carry a trailing ``\\n\\n`` the client never
    rendered (the client leaves its pendingReviseBoundary unconsumed)."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
    from kestrel_sdk.llm import ToolCallStarted

    persisted = []
    agent = _build_inline_agent(persisted)

    # is_request_cancelled flips True only once the stream is exhausted —
    # i.e. the user hits Stop after the LLM finished but before tools run.
    cancel_state = {"cancelled": False}
    agent.is_request_cancelled = MagicMock(
        side_effect=lambda _rid: cancel_state["cancelled"]
    )

    async def stream(**kw):
        yield "Let me check."
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="", tool_calls=[ToolCall(id="tc1", name="github", arguments={})]
        )
        cancel_state["cancelled"] = True  # Stop pressed post-stream

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async for _ in agent.process_input_streaming("go", session_id="s", request_id="r"):
        pass

    assistant_rows = [r for r in persisted if r["role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == "Let me check.", (
        "a turn cancelled right after the marker must persist exactly the "
        "pre-tool prose — no dangling boundary the client never drew"
    )
