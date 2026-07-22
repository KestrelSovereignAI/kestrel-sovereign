"""#2674 finding 4 — the provenance boundary for command results that invoke an
LLM.

A POST_RESPONSE response audit governs the agent's user-visible CONVERSATIONAL
output, not every internal LLM computation. Memory-maintenance commands
(``!compact`` / ``!context compact`` / ``!context query``) invoke an LLM to
produce an INTERNAL summary/answer that is persisted as a memory artifact, and
they return a DETERMINISTIC status — the LLM-generated prose is NOT released as
the assistant response. These tests pin BOTH sides of that boundary:

  * maintenance commands surface a deterministic status, never raw LLM prose;
  * a command that falls through to a real assistant turn IS audited (covered by
    test_streaming_audit.TestCommandRealProcessInputFailClosed).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.command_handler import CommandHandler


class TestCompactCommandDeterministicStatus:
    @pytest.mark.asyncio
    async def test_compact_does_not_echo_llm_summary(self):
        """``!compact`` returns a deterministic maintenance status; the raw LLM
        summary stays an internal marker and is NOT leaked as assistant prose."""
        agent = MagicMock()
        agent.context_manager = MagicMock()
        agent.llm_service = MagicMock()
        leaky_summary = "SENSITIVE unaudited model-written summary text"
        agent.context_manager.compact_session = AsyncMock(return_value={
            "success": True,
            "messages_compacted": 12,
            "messages_preserved": 10,
            "tokens_saved": 4000,
            "tokens_before": 9000,
            "tokens_after": 5000,
            "summary_preview": leaky_summary,
        })
        agent.context_stats = MagicMock()
        handler = CommandHandler(agent)

        result = await handler._cmd_compact("!compact --force")

        assert "Session compacted" in result
        assert "Messages compacted: 12" in result
        # The raw LLM summary must NOT appear in the user-visible command result.
        assert leaky_summary not in result
        assert "SENSITIVE" not in result


class TestToolResultEnvelopeDoesNotSurfaceLlmAnswer:
    def test_recursive_query_answer_not_rendered_to_user(self):
        """``!context query`` (recursive_query) puts the LLM answer in
        ``data.answer`` (a scalar); the command renderer surfaces ONLY the
        deterministic confirmation — the answer is not released as assistant
        prose. (As a mid-turn tool the answer feeds the model, whose synthesis
        IS audited — the audited path.)"""
        envelope = {
            "status": "ok",
            "confirmation": "Answered query against last_5 (500 chars, model=cheap)",
            "data": {
                "answer": "UNAUDITED model answer prose about the context",
                "context_source": "last_5",
                "query": "what happened?",
                "model_used": "cheap",
                "context_chars": 500,
            },
        }
        rendered = CommandHandler._format_tool_result_envelope(envelope)
        assert "Answered query against last_5" in rendered
        assert "UNAUDITED model answer prose" not in rendered

    def test_compact_context_confirmation_is_deterministic(self):
        """``!context compact`` surfaces a fixed confirmation; the compaction
        summary is internal (scalar data), never rendered."""
        envelope = {
            "status": "ok",
            "confirmation": "Compacted context, preserved last 10",
            "data": {"messages_compacted": 12, "tokens_after": 5000},
        }
        rendered = CommandHandler._format_tool_result_envelope(envelope)
        assert rendered == "Compacted context, preserved last 10"
