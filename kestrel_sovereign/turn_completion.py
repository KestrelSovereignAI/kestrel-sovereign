"""Turn-completion repair shared by the orchestrator and feature subagents.

#1237: a model can end a message by announcing a tool call it never makes
("Let me check the issue.") and the loop, seeing no tool call, hands that
announcement to the user as the answer. The repair gives the model one more
step to make the call.

The announcement is matched by a pattern, and a pattern cannot tell "I will
check the issue now" from a finished answer's plan ("when CI passes, I will
run the review"). The repair is therefore built so that firing on a finished
answer costs one model call and nothing else (#3397):

* It asks rather than orders. The check says it is the runtime, not the user,
  and gives the model a truthful way to say the message was complete.
* A repair that confirms the message was complete does not replace it. The
  answer is the original plus anything the model adds, never the model's reply
  to the runtime's check alone. A repair that answers without confirming has
  written a new answer, and that answer stands on its own: an announcement
  followed by "I cannot do that" would contradict itself.
"""

from __future__ import annotations

from typing import Optional

#: What the model replies when its message was already its complete answer.
REPAIR_COMPLETE_MARKER = "[answer complete]"

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


def confirms_complete(repaired: Optional[str]) -> bool:
    """True when a repair reply says the repaired message was the answer."""
    return REPAIR_COMPLETE_MARKER in (repaired or "")


def repair_addition(repaired: Optional[str]) -> str:
    """What a confirming repair reply adds to the message it repaired."""
    return (repaired or "").replace(REPAIR_COMPLETE_MARKER, "").strip()


def settle_repaired_content(original: Optional[str], repaired: Optional[str]) -> str:
    """The answer after a repair that made no tool call.

    A confirming reply keeps the original message, followed by whatever the
    reply adds. Any other reply is the model's new answer.
    """
    if not confirms_complete(repaired):
        return repaired or ""
    kept = (original or "").rstrip()
    addition = repair_addition(repaired)
    if not addition:
        return kept
    if not kept:
        return addition
    return f"{kept}\n\n{addition}"
