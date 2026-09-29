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

__all__ = [
    "MISSED_COGNITION_STATUS",
    "ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE",
    "SCHEDULER_SAFETY_DISABLEMENT_REASONS",
]
