"""Structured outcomes returned by scheduled task executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
)

#: What a reader returns in place of a stored ``result_text`` it may not show.
REDACTED_RESULT_TEXT = "[redacted: volatile privacy mode]"


def redact_stored_result_text(agent: Any, result_text: Any) -> Any:
    """Withhold a stored ``task_execution_log.result_text`` when privacy forbids it.

    The column is free-form output: a ``self_followup``'s complete cognition
    response, or whatever a scheduled tool returned. Under EPHEMERAL /
    ISOLATED / DEIDENTIFIED no reader returns it, whatever task produced it.

    This was first keyed on ``task_name == self_followup`` (#3112 gate-6 P1),
    and the next review found the same content one row over (gate-2 P1): a
    scheduled ``schedule_list`` or ``schedule_self_followups`` run under
    durable storage stores its output -- follow-up intents included -- in its
    OWN row, and a scheduled ``schedule_history`` copies earlier results into
    its row the same way. Any tool that reads conversation history can do it
    too, because a follow-up's rendered prompt and response live there. A task
    name says which task wrote the text, not what the text contains, so no
    name list can certify a result content-free.

    Every reader of the column calls this, rather than each re-deriving the
    rule. Status, timing and outcome signal stay visible; only the text is
    withheld, and it is replaced by a marker rather than ``None`` so a
    redacted result is not mistaken for a task that produced no output.
    """
    if not result_text:
        return result_text
    if not hides_persisted_user_content(agent):
        return result_text
    return REDACTED_RESULT_TEXT


@dataclass(frozen=True)
class ScheduledTaskOutcome:
    """A non-success task outcome that the runner must handle explicitly.

    Pausing is reserved for ``blocked`` outcomes, which travel through the
    dispatcher as expected policy states. Failed outcomes deliberately raise at
    the signal-handler boundary so signal and scheduler audit rows agree; they
    therefore cannot carry a pause instruction that the failed SignalResult
    has no structured channel to preserve.
    """

    status: str
    result_text: str
    pause_schedule: bool = False

    def __post_init__(self) -> None:
        if self.pause_schedule and self.status != "blocked":
            raise ValueError(
                "pause_schedule=True is only valid for blocked scheduled outcomes"
            )

    @classmethod
    def blocked(
        cls,
        *,
        task_name: str,
        decision: str,
        reason: str,
    ) -> "ScheduledTaskOutcome":
        detail = reason.strip() or "operator approval is required"
        return cls(
            status="blocked",
            result_text=(
                f"blocked: {task_name} was denied by the {decision} permission "
                f"gate: {detail}. Set this tool's permission to Auto, then "
                "resume the schedule."
            ),
            pause_schedule=True,
        )
