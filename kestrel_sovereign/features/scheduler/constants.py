"""Shared scheduler state constants without runner/status import cycles."""

ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE = "rollout_ambiguous_legacy_occurrence"

SCHEDULER_SAFETY_DISABLEMENT_REASONS = frozenset({
    "execution_log_inconsistent",
    "invalid_idempotency_key",
    ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE,
})

__all__ = [
    "ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE",
    "SCHEDULER_SAFETY_DISABLEMENT_REASONS",
]
