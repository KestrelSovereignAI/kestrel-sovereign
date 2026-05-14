"""
Per-turn reflection + fact-save phase (#1238, child of epic #1235).

After the orchestrator finalizes its visible response for a turn, this phase
issues one additional LLM call with a restricted tool set (`save_fact`,
`strategy_add_pattern`, `strategy_add_blocker`) and a reflection-focused
system prompt. The model decides which structural facts learned this turn
are worth persisting.

The phase is opt-out via `KESTREL_REFLECTION_DISABLED=1` (default on).

Design notes:
- Single round of tool calls (no follow-up LLM call) — we don't want a
  loop here, just a one-shot save opportunity.
- Failures are logged and swallowed; reflection must never break the
  main turn (it is post-yield work).
- The transcript passed to the reflection LLM is built from the
  orchestrator's full message history — assistant turns, tool results,
  the final synthesis — so the model sees what actually happened.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Tool names eligible during the reflection phase. The reflection LLM call
# is invoked with this filtered subset of the agent's tools, so the model
# cannot e.g. fire off unrelated subagent dispatches.
RESERVED_FACT_TOOL_NAMES: frozenset[str] = frozenset({
    "save_fact",
    "strategy_add_pattern",
    "strategy_add_blocker",
})


REFLECTION_SYSTEM_PROMPT = """You just finished a turn. Before yielding to the user, take one structured moment to capture what you learned.

Look back at the conversation transcript below and ask yourself two questions:
1. What structural facts did this turn surface that are worth persisting beyond this conversation? (Examples: a file or package was renamed, a tool lives at a non-obvious location, a previously-broken assumption was corrected, a peer agent uses a specific convention.)
2. What patterns or failure modes did you observe? (Examples: a category of bug, a repeating user preference, a workflow that consistently fails.)

For each fact worth persisting, call `save_fact` with subject/predicate/value/confidence.
For each pattern worth recording, call `strategy_add_pattern`.
For each open blocker the user surfaced, call `strategy_add_blocker`.

Rules:
- One tool call per distinct fact/pattern/blocker. Do not over-save: only persist what would be useful to a future you, in a future conversation, with no access to today's transcript.
- If nothing structural was learned, emit no tool calls and return an empty response. That is a valid outcome.
- Do not narrate, summarize, or address the user. Your output text after the tool calls is discarded; only the tool calls have effect.
- Confidence should reflect how certain you are: 1.0 for things you directly verified, 0.7-0.9 for things you inferred from strong evidence, lower for guesses.
"""


def reflection_disabled() -> bool:
    """Return True if reflection phase is globally disabled via env var."""
    val = os.environ.get("KESTREL_REFLECTION_DISABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def filter_reflection_tools(all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only the tool schemas eligible during the reflection phase."""
    out: List[Dict[str, Any]] = []
    for tool in all_tools or []:
        name = _tool_schema_name(tool)
        if name in RESERVED_FACT_TOOL_NAMES:
            out.append(tool)
    return out


def _tool_schema_name(tool: Dict[str, Any]) -> Optional[str]:
    """Extract the tool name from an OpenAI-format function schema."""
    fn = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(fn, dict):
        return fn.get("name")
    return tool.get("name") if isinstance(tool, dict) else None


def format_turn_transcript(
    messages: List[Dict[str, Any]],
    final_content: Optional[str],
    *,
    max_chars: int = 12_000,
) -> str:
    """Build a compact transcript of the turn for the reflection LLM call.

    The orchestrator's `messages` list already includes the system prompt,
    user message, assistant responses, and tool results. We strip the
    system prompt (the reflection LLM gets its own) and trim long tool
    outputs to keep the reflection call cheap.
    """
    lines: List[str] = []
    for msg in messages or []:
        role = msg.get("role", "")
        if role == "system":
            continue
        content = msg.get("content") or ""
        if role == "tool":
            content = content[:1500]
            lines.append(f"[tool-result] {content}")
            continue
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                call_summary = ", ".join(
                    _summarize_tool_call(tc) for tc in tool_calls
                )
                lines.append(f"[assistant tool-calls] {call_summary}")
            if content:
                lines.append(f"[assistant] {content[:2500]}")
            continue
        if role == "user":
            lines.append(f"[user] {content[:2000]}")
            continue
        lines.append(f"[{role}] {content[:1000]}")

    if final_content:
        lines.append(f"[assistant final] {final_content[:2500]}")

    transcript = "\n\n".join(lines)
    if len(transcript) > max_chars:
        # Keep tail — most recent turn content is what reflection cares about.
        transcript = "[transcript truncated]\n" + transcript[-max_chars:]
    return transcript


def _summarize_tool_call(tc: Dict[str, Any]) -> str:
    fn = tc.get("function") if isinstance(tc, dict) else None
    if not isinstance(fn, dict):
        return "<unknown-tool>"
    name = fn.get("name", "?")
    args = fn.get("arguments")
    if isinstance(args, dict):
        arg_repr = json.dumps(args, default=str)[:200]
    elif isinstance(args, str):
        arg_repr = args[:200]
    else:
        arg_repr = ""
    return f"{name}({arg_repr})" if arg_repr else name


class ReflectionOutcome:
    """Lightweight result holder for tests + observability."""

    __slots__ = ("skipped", "skip_reason", "tool_calls_executed", "duration_ms", "error")

    def __init__(
        self,
        *,
        skipped: bool = False,
        skip_reason: Optional[str] = None,
        tool_calls_executed: int = 0,
        duration_ms: int = 0,
        error: Optional[str] = None,
    ) -> None:
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.tool_calls_executed = tool_calls_executed
        self.duration_ms = duration_ms
        self.error = error

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "tool_calls_executed": self.tool_calls_executed,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


def now_monotonic_ms() -> int:
    return int(time.monotonic() * 1000)
