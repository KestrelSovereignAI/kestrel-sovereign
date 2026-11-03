"""Shared scheduler state constants without runner/status import cycles."""

ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE = "rollout_ambiguous_legacy_occurrence"

SCHEDULER_SAFETY_DISABLEMENT_REASONS = frozenset({
    "execution_log_inconsistent",
    "invalid_idempotency_key",
    ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE,
})

# task_execution_log status for a COGNITION occurrence the dispatcher dropped
# before the turn ran (rate limit, quiet hours, coalescing). Deliberately not
# 'success' and not 'skipped': the schedule promised a turn and produced none
# (#3101).
MISSED_COGNITION_STATUS = "missed"

# task_execution_log status for a permission-gated tool the scheduler did not
# run because an enforcing PRE_TOOL_USE hook FAILED (raised or timed out) and
# the hook manager failed closed. Deliberately not 'blocked': no policy decided
# anything, so there is nothing for an operator to change and the schedule is
# not paused. The tick fails closed, the row stays enabled, and the next
# attempt backs off (#3749).
HOOK_FAILED_STATUS = "hook_failed"

# Longest ``scheduled_tasks.terminal_cause_detail`` the runner stores: the
# free-form reason behind the cause the scheduler recorded when it disabled a
# row, beside ``terminal_status`` and ``terminal_cause`` (#3873).
TERMINAL_CAUSE_MAX_CHARS = 500

__all__ = [
    "HOOK_FAILED_STATUS",
    "MISSED_COGNITION_STATUS",
    "ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE",
    "SCHEDULER_SAFETY_DISABLEMENT_REASONS",
    "TERMINAL_CAUSE_MAX_CHARS",
]
