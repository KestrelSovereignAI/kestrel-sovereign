"""Structured outcomes returned by scheduled task executors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from kestrel_sovereign.features.scheduler.constants import HOOK_FAILED_STATUS
from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
)

#: What a reader returns in place of a stored ``result_text`` it may not show.
REDACTED_RESULT_TEXT = "[redacted: volatile privacy mode]"

#: A permission gate's decision (DENY or ASK) blocked the tool.
BLOCKED_STATUS = "blocked"

#: Outcomes the signal-handler boundary returns to the runner rather than
#: raising: each is an expected state of the permission gate, and the runner
#: needs its structure (the pause instruction, or the hook-failure backoff).
GATE_OUTCOME_STATUSES = frozenset({BLOCKED_STATUS, HOOK_FAILED_STATUS})

#: The permission decisions that block a scheduled tool, as
#: ``PermissionDecision`` values: the decision a :class:`PauseCause` records.
BLOCKING_DECISIONS = frozenset({"deny", "ask"})

#: Additive ``scheduled_tasks`` columns holding a pause's :class:`PauseCause`,
#: in its field order. Written with ``terminal_status`` by the finalizer and
#: cleared by ``schedule_resume`` (#3873).
PAUSE_CAUSE_COLUMNS = {
    "terminal_cause_hook": "TEXT",
    "terminal_cause_decision": "TEXT",
    "terminal_cause_tool": "TEXT",
}

#: Projection appended to a ``... FROM scheduled_tasks`` read, decoded by
#: :meth:`PauseCause.from_row`.
PAUSE_CAUSE_PROJECTION = ", ".join(PAUSE_CAUSE_COLUMNS)

#: Number of values :data:`PAUSE_CAUSE_PROJECTION` yields.
PAUSE_CAUSE_PROJECTION_WIDTH = len(PAUSE_CAUSE_COLUMNS)

#: What a hook or tool name must look like to be recorded: a name set in code.
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def _identifier(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


@dataclass(frozen=True)
class PauseCause:
    """Why a permission gate paused a schedule, as structured fields (#3873).

    The hook whose decision paused it, that decision (``deny`` or ``ask``),
    and the tool it gated: identifiers set in code, never text. The reason the
    hook gave is deliberately not part of it. A hook's reason is free-form and
    can repeat the call's arguments (a spawn mandate's hook names the argument
    value it rejected), so nothing the scheduler stores, logs or surfaces
    carries it, the blocked outcome's ``result_text`` included. This cause
    needs no privacy withholding, because nothing in it came from the call. A
    name that is not a plain identifier is recorded as unknown (``None``)
    rather than kept as text.
    """

    hook: Optional[str]
    decision: str
    tool: Optional[str]

    def __post_init__(self) -> None:
        if self.decision not in BLOCKING_DECISIONS:
            raise ValueError(
                "a pause cause needs a blocking decision "
                f"({', '.join(sorted(BLOCKING_DECISIONS))}), not {self.decision!r}"
            )
        for field_name in ("hook", "tool"):
            value = getattr(self, field_name)
            if value is not None and _identifier(value) is None:
                raise ValueError(
                    f"a pause cause's {field_name} must be an identifier or None"
                )

    @classmethod
    def of(
        cls, *, hook: Any, decision: str, tool: Any
    ) -> "PauseCause":
        """Build a cause, recording a name that is not an identifier as unknown."""

        return cls(hook=_identifier(hook), decision=decision, tool=_identifier(tool))

    @classmethod
    def from_row(cls, values: Optional[Sequence[Any]]) -> Optional["PauseCause"]:
        """Decode :data:`PAUSE_CAUSE_PROJECTION`; ``None`` when none is stored.

        A row whose stored decision is not a blocking one names no cause.
        """

        if values is None or len(values) != PAUSE_CAUSE_PROJECTION_WIDTH:
            return None
        hook, decision, tool = values
        if decision not in BLOCKING_DECISIONS:
            return None
        return cls.of(hook=hook, decision=decision, tool=tool)

    def as_params(self) -> tuple[Any, ...]:
        """Values for the :data:`PAUSE_CAUSE_COLUMNS` in declaration order."""

        return (self.hook, self.decision, self.tool)

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {"hook": self.hook, "decision": self.decision, "tool": self.tool}

    def describe(self) -> str:
        """The cause as every surface words it, from its fields alone."""

        return (
            f"hook {self.hook or 'unknown'} decided {self.decision} "
            f"for tool {self.tool or 'unknown'}"
        )


#: :data:`PAUSE_CAUSE_COLUMNS` values for a row the finalizer did not pause.
NO_PAUSE_CAUSE_PARAMS: tuple[Any, ...] = (None,) * PAUSE_CAUSE_PROJECTION_WIDTH


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
    # Why a pausing outcome paused, which the runner stores on the schedule
    # row beside ``terminal_status`` so status can name it after the
    # execution's ``result_text`` has aged out of the log (#3873). Structured
    # fields only; a blocked ``result_text`` is worded from them too.
    pause_cause: Optional[PauseCause] = None

    def __post_init__(self) -> None:
        if self.pause_schedule and self.status != BLOCKED_STATUS:
            raise ValueError(
                "pause_schedule=True is only valid for blocked scheduled outcomes"
            )
        if self.pause_schedule and self.pause_cause is None:
            raise ValueError("a pausing scheduled outcome must record its cause")

    @classmethod
    def blocked(
        cls,
        *,
        task_name: str,
        decision: str,
        hook: Optional[str] = None,
    ) -> "ScheduledTaskOutcome":
        """A permission gate's DENY or ASK blocked ``task_name``; pause it.

        ``hook`` names the hook that decided, when the gate says. The outcome
        takes no reason: a hook's reason is free-form and can repeat the
        call's arguments, and ``result_text`` is persisted in
        ``task_execution_log``. So the text is worded from the
        :class:`PauseCause` alone, as every other surface words it (#3873).
        """
        cause = PauseCause.of(hook=hook, decision=decision, tool=task_name)
        return cls(
            status=BLOCKED_STATUS,
            result_text=(
                f"blocked: the permission gate stopped this tick "
                f"({cause.describe()}). Set this tool's permission to Auto, "
                "then resume the schedule."
            ),
            pause_schedule=True,
            pause_cause=cause,
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
