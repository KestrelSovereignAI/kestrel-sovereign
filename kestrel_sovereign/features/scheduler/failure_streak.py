"""Consecutive-failure tracking for scheduled tasks (#3280).

A scheduled task that failed every time it ran used to be visible only as rows
in ``task_execution_log``. Emma's ``signal_dispatch`` failed 23 days running
and nothing a reader looks at said so, while ``morning_signal`` succeeding
beside it made the pair read as healthy.

The runner keeps a durable streak on each schedule row, written in the same
transaction that finalizes the execution. When the streak reaches the
configured threshold it records a marker, and readers (``schedule_list`` and
the ``scheduled_task_failures`` health check) report that marker. The marker
is visibility only: it retries, pauses, or remediates nothing.

Only a ``failed`` or ``hook_failed`` execution extends a streak and only a
``success`` ends one. Every other outcome (a misfire skip, a dropped cognition
turn, a permission block) is a run that did not happen or did not fail, so it
leaves the streak as it was. A paused schedule runs nothing and so changes
nothing.

A ``hook_failed`` tick is a failure the scheduler also acts on. It did not
run the tool because the permission gate's enforcing hook failed, so it failed
closed for that tick and kept the schedule enabled (#3749). The runner delays
that schedule's next attempt by :func:`hook_failure_backoff`, which grows with
the streak, so a hook that keeps failing is not retried every minute while the
marker reports it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Dict, Optional, Sequence

from kestrel_sovereign.features.scheduler.constants import HOOK_FAILED_STATUS
from kestrel_sovereign.features.scheduler.outcome import (
    redact_stored_result_text,
)

#: Consecutive failures of one schedule before the runner records a marker.
DEFAULT_FAILURE_ALERT_THRESHOLD = 3

#: The execution statuses that move a streak. Anything else leaves it alone.
FAILED_STATUS = "failed"
SUCCESS_STATUS = "success"
FAILURE_STATUSES = frozenset({FAILED_STATUS, HOOK_FAILED_STATUS})

#: Delay before the first retry after a ``hook_failed`` tick; each further
#: consecutive failure doubles it, up to the maximum.
DEFAULT_HOOK_FAILURE_BACKOFF_BASE_SECONDS = 60
DEFAULT_HOOK_FAILURE_BACKOFF_MAX_SECONDS = 15 * 60

#: How much of the last failure's stored result a marker repeats.
FAILURE_SUMMARY_MAX_CHARS = 500

#: Additive ``scheduled_tasks`` columns holding the streak.
FAILURE_STREAK_COLUMNS = {
    "failure_streak_count": "INTEGER NOT NULL DEFAULT 0",
    "failure_streak_first_at": "TEXT",
    "failure_streak_last_at": "TEXT",
    "failure_streak_last_execution_id": "TEXT",
    "failure_streak_flagged_at": "TEXT",
}

#: Projection appended to a ``... FROM scheduled_tasks`` read, decoded by
#: :func:`failure_streak_view`. The last error is read from the execution the
#: streak names rather than copied onto the schedule row, so the execution log
#: stays the one record of what a run returned.
FAILURE_STREAK_PROJECTION = """
    failure_streak_count, failure_streak_first_at, failure_streak_last_at,
    failure_streak_flagged_at,
    (SELECT failed_run.result_text FROM task_execution_log failed_run
      WHERE failed_run.id = scheduled_tasks.failure_streak_last_execution_id
        AND failed_run.agent_id = scheduled_tasks.agent_id)
"""

#: Number of values :data:`FAILURE_STREAK_PROJECTION` yields.
FAILURE_STREAK_PROJECTION_WIDTH = 5


@dataclass(frozen=True)
class FailureStreak:
    """The persisted streak of one schedule row."""

    count: int = 0
    first_at: Optional[str] = None
    last_at: Optional[str] = None
    last_execution_id: Optional[str] = None
    flagged_at: Optional[str] = None

    @classmethod
    def from_row(cls, row: Optional[Sequence[Any]]) -> "FailureStreak":
        """Decode the five ``FAILURE_STREAK_COLUMNS`` in declaration order."""

        if row is None:
            return cls()
        return cls(
            count=int(row[0] or 0),
            first_at=row[1],
            last_at=row[2],
            last_execution_id=row[3],
            flagged_at=row[4],
        )

    def as_params(self) -> tuple[Any, ...]:
        """Values for the ``FAILURE_STREAK_COLUMNS`` in declaration order."""

        return (
            self.count,
            self.first_at,
            self.last_at,
            self.last_execution_id,
            self.flagged_at,
        )


@dataclass(frozen=True)
class FailureStreakTransition:
    """What one finalized execution did to its schedule's streak."""

    prior: FailureStreak
    streak: FailureStreak

    @property
    def flagged(self) -> bool:
        """This execution raised the marker."""

        return self.prior.flagged_at is None and self.streak.flagged_at is not None

    @property
    def recovered(self) -> bool:
        """This execution cleared a raised marker."""

        return self.prior.flagged_at is not None and self.streak.flagged_at is None


def next_failure_streak(
    prior: FailureStreak,
    *,
    status: str,
    at: str,
    execution_id: str,
    threshold: int,
) -> FailureStreakTransition:
    """Return the streak after one execution finalized with ``status``."""

    if status == SUCCESS_STATUS:
        return FailureStreakTransition(prior=prior, streak=FailureStreak())
    if status not in FAILURE_STATUSES:
        return FailureStreakTransition(prior=prior, streak=prior)
    count = prior.count + 1
    flagged_at = prior.flagged_at
    if flagged_at is None and count >= threshold:
        flagged_at = at
    return FailureStreakTransition(
        prior=prior,
        streak=replace(
            prior,
            count=count,
            first_at=prior.first_at if prior.count and prior.first_at else at,
            last_at=at,
            last_execution_id=execution_id,
            flagged_at=flagged_at,
        ),
    )


def hook_failure_backoff(
    consecutive_failures: int,
    *,
    base_seconds: int = DEFAULT_HOOK_FAILURE_BACKOFF_BASE_SECONDS,
    max_seconds: int = DEFAULT_HOOK_FAILURE_BACKOFF_MAX_SECONDS,
) -> timedelta:
    """How long after a ``hook_failed`` tick the schedule may next run.

    ``consecutive_failures`` is the streak including that tick. The first
    failure waits ``base_seconds``, so an every-minute schedule simply runs on
    its next tick; each further one doubles the wait, capped at
    ``max_seconds``.
    """

    base = max(0, int(base_seconds))
    if base == 0:
        return timedelta(0)
    ceiling = max(base, int(max_seconds))
    exponent = max(0, int(consecutive_failures) - 1)
    # Past the cap the doubling no longer matters; bound the exponent so a
    # long streak cannot build an enormous integer.
    if exponent >= ceiling.bit_length():
        return timedelta(seconds=ceiling)
    return timedelta(seconds=min(ceiling, base * (2 ** exponent)))


def failure_streak_view(
    values: Sequence[Any], *, agent: Any
) -> tuple[int, Optional[Dict[str, Any]]]:
    """Decode a :data:`FAILURE_STREAK_PROJECTION` into (count, marker).

    The marker is ``None`` until the runner has recorded one. Its
    ``last_error`` passes through :func:`redact_stored_result_text`, because it
    is the stored result of a run and every reader of that column redacts it.
    """

    count = int(values[0] or 0)
    first_at, last_at, flagged_at, last_result = values[1:5]
    if flagged_at is None:
        return count, None
    last_error = redact_stored_result_text(agent, last_result)
    if isinstance(last_error, str) and len(last_error) > FAILURE_SUMMARY_MAX_CHARS:
        last_error = last_error[: FAILURE_SUMMARY_MAX_CHARS - 3].rstrip() + "..."
    return count, {
        "consecutive_failures": count,
        "first_failure_at": first_at,
        "last_failure_at": last_at,
        "last_error": last_error,
        "flagged_at": flagged_at,
    }


async def load_failure_markers(
    db: Any, agent_id: str, *, agent: Any
) -> list[Dict[str, Any]]:
    """Return every schedule of ``agent_id`` carrying a failure marker."""

    rows = await db.fetchall(
        f"""
        SELECT id, task_name, enabled, {FAILURE_STREAK_PROJECTION}
        FROM scheduled_tasks
        WHERE agent_id = ? AND failure_streak_flagged_at IS NOT NULL
        ORDER BY failure_streak_first_at, id
        """,
        (agent_id,),
    )
    markers = []
    for row in rows:
        _count, marker = failure_streak_view(row[3:], agent=agent)
        markers.append({
            "task_id": row[0],
            "task_name": row[1],
            "enabled": bool(row[2]),
            **marker,
        })
    return markers


__all__ = [
    "DEFAULT_FAILURE_ALERT_THRESHOLD",
    "DEFAULT_HOOK_FAILURE_BACKOFF_BASE_SECONDS",
    "DEFAULT_HOOK_FAILURE_BACKOFF_MAX_SECONDS",
    "FAILED_STATUS",
    "FAILURE_STATUSES",
    "FAILURE_STREAK_COLUMNS",
    "FAILURE_STREAK_PROJECTION",
    "FAILURE_STREAK_PROJECTION_WIDTH",
    "FAILURE_SUMMARY_MAX_CHARS",
    "FailureStreak",
    "FailureStreakTransition",
    "SUCCESS_STATUS",
    "failure_streak_view",
    "hook_failure_backoff",
    "load_failure_markers",
    "next_failure_streak",
]
