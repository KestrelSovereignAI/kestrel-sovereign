"""Structured outcomes returned by scheduled task executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kestrel_sovereign.features.scheduler.constants import (
    HOOK_FAILED_STATUS,
    TERMINAL_CAUSE_MAX_CHARS,
)
from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
)

#: What a reader returns in place of a stored ``result_text`` it may not show.
REDACTED_RESULT_TEXT = "[redacted: volatile privacy mode]"

#: A permission gate's decision (DENY or ASK) blocked the tool.
BLOCKED_STATUS = "blocked"

#: The gate decisions that block a scheduled tool: the
#: ``PermissionDecision`` values a :class:`ScheduledTaskOutcome.blocked`
#: outcome may name. Its stored cause is built from these alone, so the cause
#: is content-free whatever the hook's reason says (#3873).
BLOCKING_GATE_DECISIONS = frozenset({"deny", "ask"})

#: Outcomes the signal-handler boundary returns to the runner rather than
#: raising: each is an expected state of the permission gate, and the runner
#: needs its structure (the pause instruction, or the hook-failure backoff).
GATE_OUTCOME_STATUSES = frozenset({BLOCKED_STATUS, HOOK_FAILED_STATUS})


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


def withhold_stored_cause_detail(withhold: bool, detail: Any) -> Any:
    """Withhold a stored ``scheduled_tasks.terminal_cause_detail`` (#3873).

    The detail is the free-form reason a gate gave when it blocked a scheduled
    tool, and a gate's reason can repeat the call's arguments: a spawn
    mandate's ``MandateRestrictionHook`` names the rejected argument value. It
    is therefore held to the same rule as ``result_text``
    (:func:`redact_stored_result_text`), decided by the reader at read time so
    a detail stored under a durable mode is withheld once the agent enters a
    volatile one. ``withhold`` is that rule's verdict,
    ``hides_persisted_user_content(agent)``, computed by a reader that has the
    agent; status readers that do not, receive it as a parameter. The
    content-free ``terminal_cause`` beside it is never withheld.
    """
    if not detail or not withhold:
        return detail
    return REDACTED_RESULT_TEXT


@dataclass(frozen=True)
class ScheduledTaskOutcome:
    """A non-success task outcome that the runner must handle explicitly.

    Pausing is reserved for ``blocked`` outcomes, which travel through the
    dispatcher as expected policy states. A ``hook_failed`` outcome travels the
    same way but never pauses: the gate's enforcing hook failed rather than
    decided, so the runner keeps the schedule enabled and backs off its next
    attempt (#3749). Failed outcomes deliberately raise at the signal-handler
    boundary so signal and scheduler audit rows agree; they therefore cannot
    carry a pause instruction that the failed SignalResult has no structured
    channel to preserve.
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
    # Why a pausing outcome paused the schedule, in two parts the runner
    # stores on the schedule row beside ``terminal_status`` (#3873), so the
    # pause still names its cause once the execution's ``result_text`` has
    # aged out of the log. ``pause_cause`` is content-free, built only from
    # controlled values (the gate's decision), and every reader shows it.
    # ``pause_cause_detail`` is the reason the gate gave: free-form hook text
    # that can repeat the call's arguments, so readers withhold it in a
    # volatile privacy mode exactly as they withhold ``result_text``.
    pause_cause: str = ""
    pause_cause_detail: str = ""

    def __post_init__(self) -> None:
        if self.pause_schedule and self.status != BLOCKED_STATUS:
            raise ValueError(
                "pause_schedule=True is only valid for blocked scheduled outcomes"
            )
        if self.pause_schedule and not self.pause_cause.strip():
            raise ValueError("a pausing scheduled outcome must record its cause")

    @classmethod
    def blocked(
        cls,
        *,
        task_name: str,
        decision: str,
        reason: str,
    ) -> "ScheduledTaskOutcome":
        if decision not in BLOCKING_GATE_DECISIONS:
            raise ValueError(
                f"a blocked scheduled outcome needs a blocking gate decision "
                f"({', '.join(sorted(BLOCKING_GATE_DECISIONS))}), not {decision!r}"
            )
        detail = reason.strip() or "operator approval is required"
        return cls(
            status=BLOCKED_STATUS,
            result_text=(
                f"blocked: {task_name} was denied by the {decision} permission "
                f"gate: {detail}. Set this tool's permission to Auto, then "
                "resume the schedule."
            ),
            pause_schedule=True,
            pause_cause=f"the permission gate decided {decision}",
            pause_cause_detail=detail[:TERMINAL_CAUSE_MAX_CHARS],
        )

    @classmethod
    def hook_failed(
        cls,
        *,
        task_name: str,
        reason: str,
    ) -> "ScheduledTaskOutcome":
        """The permission gate failed closed because an enforcing hook failed.

        Not a policy verdict, so the schedule is not paused: nothing an
        operator changes would make the next tick different, and a transient
        failure (``database is locked``) clears by itself (#3749).
        """
        detail = reason.strip().rstrip(".") or "an enforcing permission hook failed"
        return cls(
            status=HOOK_FAILED_STATUS,
            result_text=(
                f"hook_failed: {task_name} was not run on this tick: {detail}. "
                "The scheduler failed closed for this tick only; the schedule "
                "stays enabled and retries on a later tick."
            ),
        )
