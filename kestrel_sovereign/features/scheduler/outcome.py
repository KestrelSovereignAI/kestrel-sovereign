"""Structured outcomes returned by scheduled task executors."""

from __future__ import annotations

from dataclasses import dataclass


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
    # A short controlled token naming WHY a failed outcome failed (for example
    # ``SLEEP_FAILED``). Unlike ``result_text`` it may cross into the
    # exception the signal boundary raises, because it is a code the feature
    # owning the task DECLARED (``Feature.tool_reason_codes``), never
    # free-form prose; the raise site bounds it by membership in that
    # declaration and drops — and logs — anything else.
    reason_code: str = ""

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
