"""Turn-completion repair shared by the orchestrator and feature subagents.

#1237: a model can end a message by announcing a tool call it never makes
("Let me check the issue.") and the loop, seeing no tool call, hands that
announcement to the user as the answer. The repair gives the model one more
step to make the call.

Two rules keep the repair from damaging a message that was already finished:

* Only the message's final paragraph is read for the announcement. The #1237
  failure is a message that *ends* by promising a call. A plan inside a
  finished answer ("when CI passes, I will run the review") is not that
  failure, and reading the whole body fired the repair on nearly every
  orchestrator status report.
* A repair that makes no tool call does not replace the message it repaired.
  The model has confirmed the message was its answer, so the answer is the
  original plus anything the model adds, never the model's reply to the
  runtime's check alone.
"""

from __future__ import annotations

import re
from typing import Optional, Pattern

#: What the model replies when its message was already its complete answer.
REPAIR_COMPLETE_MARKER = "[answer complete]"

_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")


def final_paragraph(content: Optional[str]) -> str:
    """The last non-empty paragraph of ``content`` ("" when there is none)."""
    if not content:
        return ""
    paragraphs = [p for p in _PARAGRAPH_BREAK_RE.split(content.strip()) if p.strip()]
    return paragraphs[-1] if paragraphs else ""


def ends_with_unfinished_intent(content: Optional[str], pattern: Pattern[str]) -> bool:
    """True when the final paragraph of ``content`` announces a tool call."""
    return bool(pattern.search(final_paragraph(content)))


def turn_completion_repair_prompt(unit: str) -> str:
    """The runtime's check, worded for a ``unit`` of work ("turn" or "task").

    It reaches the model in the user role, so it says it is the runtime and not
    the user. It also leaves the model a truthful way to say the message was
    finished: a message whose steps wait for a later event is a complete answer,
    and pressing the model to act on it now would push work ahead of the
    condition it waits for.
    """
    return (
        "This is an automatic check by the agent runtime, not a message from the user.\n\n"
        f"Your last message ends as if you were about to use a tool in this {unit}, "
        "but it made no tool call.\n"
        "- If you meant to act now, make the tool call now.\n"
        "- If that message was already your complete answer (for example, the steps "
        f"it describes wait for a later event or {unit}), it will be delivered exactly "
        f"as written. Reply with {REPAIR_COMPLETE_MARKER} and nothing else, unless you "
        "have something to add. Do not repeat the message."
    )


def repair_addition(repaired: Optional[str]) -> str:
    """What a no-tool repair reply adds to the message it repaired."""
    return (repaired or "").replace(REPAIR_COMPLETE_MARKER, "").strip()


def settle_repaired_content(original: Optional[str], repaired: Optional[str]) -> str:
    """The answer after a repair that made no tool call.

    The original message, followed by whatever the repair reply adds.
    """
    kept = (original or "").rstrip()
    addition = repair_addition(repaired)
    if not addition:
        return kept
    if not kept:
        return addition
    return f"{kept}\n\n{addition}"
