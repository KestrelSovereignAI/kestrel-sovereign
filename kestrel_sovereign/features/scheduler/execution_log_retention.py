"""Age out old ``task_execution_log`` rows (#3785).

Every scheduled run leaves a row, and the every-minute built-ins
(``wait_reconcile``, ``restart_coordinator``) leave ~2,500 a day per agent,
almost all of them runs that found nothing to do. Nothing removed them: 288k
rows and ~250 MB on one agent. The ``trash_retention`` maintenance rail calls
:func:`purge_execution_log` with the ``[telemetry_retention]`` window.

The log is not only history, so an old row survives the window when anything
still reads it:

* a run the runner still owns (``claimed`` or ``executing``), or one a
  schedule row names as its current claim;
* the run a schedule's failure streak names, whose result the streak reports;
* the newest run of each schedule, which ``!schedule self-followups`` and the
  reflection status read as that schedule's outcome;
* a run the Sovereign rated (``outcome_signal``), which ``!schedule
  engagement`` averages.
"""

from __future__ import annotations

from kestrel_sovereign.storage.retention import delete_rows_in_batches

_EXPIRED_EXECUTION = """
    candidate.agent_id = ?
    AND candidate.executed_at < ?
    AND candidate.status NOT IN ('claimed', 'executing')
    AND candidate.outcome_signal IS NULL
    AND EXISTS (
        SELECT 1 FROM task_execution_log newer
        WHERE newer.task_id = candidate.task_id
          AND newer.agent_id = candidate.agent_id
          AND newer.executed_at > candidate.executed_at
    )
    AND NOT EXISTS (
        SELECT 1 FROM scheduled_tasks schedule
        WHERE schedule.agent_id = candidate.agent_id
          AND (schedule.claim_execution_id = candidate.id
               OR schedule.failure_streak_last_execution_id = candidate.id)
    )
"""


async def purge_execution_log(
    db, *, agent_id: str, older_than: str, max_rows: int,
) -> int:
    """Delete this agent's expired execution rows; see the module docstring.

    *older_than* is an ISO-8601 instant compared with ``executed_at``. Returns
    the number of rows deleted, at most *max_rows*.
    """
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("agent_id must identify a durable agent principal")
    return await delete_rows_in_batches(
        db,
        table="task_execution_log",
        predicate=_EXPIRED_EXECUTION,
        params=(agent_id, older_than),
        max_rows=max_rows,
    )
