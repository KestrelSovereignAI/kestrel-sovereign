"""Structured outcomes returned by scheduled task executors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledTaskOutcome:
    """A non-success task outcome that the runner must handle explicitly."""

    status: str
    result_text: str
    pause_schedule: bool = False

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
