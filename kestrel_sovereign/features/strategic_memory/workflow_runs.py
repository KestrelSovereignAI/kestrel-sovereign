"""How the agent's workflow runs ended, for the Morning Signal (#3519).

``stalled_work_rescue`` failed on every run from 09-29 to 10-08, four times a
day, and nothing said so: not the morning signal, not the dispatch report, not
the agent's own turns. It was found by reading ``workflow_runs`` by hand. This
module is how the briefing reads it.

Core cannot import kestrel-feature-workflows, and that package publishes no
read-only reporter of run outcomes: its ``workflow_list_runs`` and
``workflow_history`` are governed tool calls, one run at a time, and neither
answers "how did each definition do since yesterday". So this reads the
Workflows store tables in the agent's own feature database, and only these
columns of them:

``workflow_runs``
    ``run_id``, ``workflow_name``, ``status``, ``started_by_did``,
    ``started_at``, ``finished_at``, ``deleted_at``
``workflow_stage_links``
    ``run_id``, ``gate_outcome``, ``stage_name``, ``gate_reason``,
    ``occurred_at``, ``attempt_number``

A run belongs to the agent whose DID the Workflows runner stamped into
``started_by_did``: the agent's stable DID (the legacy ``did:pkh`` of a
rotated agent, the ``did:web`` of a born-hybrid one). A shared PostgreSQL
database holds every hosted agent's runs, so the read is scoped to it.

Only ``completed`` and ``failed`` runs are outcomes here. A run that is still
going has no outcome yet, and a cancelled one was stopped, not failed or
succeeded, so neither counts toward, or breaks, a run of failures.

A store that cannot be read is reported as unreadable, never as a quiet
section: a briefing that says nothing about workflow runs is the silence this
module exists to end.

A gate reason is free text: Workflows stores a failing stage's error there,
and that can carry conversation-derived content. Under EPHEMERAL, ISOLATED or
DEIDENTIFIED no reader returns persisted user content, so in those modes each
reason is replaced by :data:`WITHHELD_GATE_REASON` as it is read, and a store
read error is reported by its type alone (SQLite's decode error quotes the
row's text). The counts, the failing stage, and the persistent-failure
call-out with its last-success date are run structure, not content, and stay:
withholding them too would bring back the silence.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from kestrel_sovereign.features.storage_access import (
    AgentIdentityUnavailable,
    hides_persisted_user_content,
    resolve_feature_database,
    resolve_scoped_agent_did,
)
from kestrel_sovereign.storage.db.timestamp import TimestamptzParameter

from .timestamps import parse_instant

logger = logging.getLogger(__name__)

#: The Workflows store tables this module reads (kestrel-feature-workflows
#: ``WorkflowStore.RUNS_TABLE`` and ``STAGE_LINKS_TABLE``).
RUNS_TABLE = "workflow_runs"
STAGE_LINKS_TABLE = "workflow_stage_links"

#: The heading the Morning Signal gives this section.
SECTION_TITLE = "Workflow runs (24h)"

#: How far back the counts reach.
WINDOW = timedelta(hours=24)

#: A definition whose last this-many outcomes were all failures is called out
#: as persistently failing. ``morning_signal_config.persistent_failure_runs``
#: overrides it.
DEFAULT_PERSISTENT_FAILURE_RUNS = 3
PERSISTENT_FAILURE_RUNS_KEY = "persistent_failure_runs"

COMPLETED = "completed"
FAILED = "failed"

#: What a recorded gate reason reads as in a volatile privacy mode. A marker,
#: not ``None``, so a withheld reason is not mistaken for one never recorded.
WITHHELD_GATE_REASON = "reason withheld (privacy mode)"

#: A gate reason can carry a whole CI snapshot. The briefing shows its start;
#: ``workflow_history`` on the named run has the rest.
_REASON_LIMIT = 300
#: Distinct failures listed per definition before the rest are counted.
_FAILURE_GROUP_LIMIT = 5

_WHITESPACE = re.compile(r"\s+")

# Every query reads the store as of the window's end, so a report built for
# an earlier moment does not count the runs that ended after it.

# Each terminal run in the window, joined to its failing stage links. A run
# whose gate failed more than once (a retried stage) has one row per failing
# link; ordered by ``occurred_at`` so the last row is the link it ended on.
_WINDOW_RUNS_SQL = f"""
    SELECT r.run_id, r.workflow_name, r.status,
           COALESCE(r.finished_at, r.started_at),
           l.stage_name, l.gate_reason
    FROM {RUNS_TABLE} r
    LEFT JOIN {STAGE_LINKS_TABLE} l
        ON l.run_id = r.run_id AND l.gate_outcome = 'fail'
    WHERE r.started_by_did = ?
      AND r.deleted_at IS NULL
      AND r.status IN ('{COMPLETED}', '{FAILED}')
      AND COALESCE(r.finished_at, r.started_at) >= ?
      AND COALESCE(r.finished_at, r.started_at) <= ?
    ORDER BY r.run_id, l.occurred_at, l.attempt_number
"""

# When each definition last completed.
_LAST_SUCCESS_SQL = f"""
    SELECT workflow_name, MAX(COALESCE(finished_at, started_at))
    FROM {RUNS_TABLE}
    WHERE started_by_did = ?
      AND deleted_at IS NULL
      AND status = '{COMPLETED}'
      AND COALESCE(finished_at, started_at) <= ?
    GROUP BY workflow_name
"""

# How many runs of each definition have failed since it last completed (every
# failed run, for one that never has).
_FAILURES_SINCE_SUCCESS_SQL = f"""
    SELECT f.workflow_name, COUNT(*)
    FROM {RUNS_TABLE} f
    LEFT JOIN (
        SELECT workflow_name,
               MAX(COALESCE(finished_at, started_at)) AS succeeded_at
        FROM {RUNS_TABLE}
        WHERE started_by_did = ?
          AND deleted_at IS NULL
          AND status = '{COMPLETED}'
          AND COALESCE(finished_at, started_at) <= ?
        GROUP BY workflow_name
    ) s ON s.workflow_name = f.workflow_name
    WHERE f.started_by_did = ?
      AND f.deleted_at IS NULL
      AND f.status = '{FAILED}'
      AND COALESCE(f.finished_at, f.started_at) <= ?
      AND (s.succeeded_at IS NULL
           OR COALESCE(f.finished_at, f.started_at) > s.succeeded_at)
    GROUP BY f.workflow_name
"""


class WorkflowRunsNotAssessed(Exception):
    """The agent's workflow runs could not be assessed.

    Raised rather than reporting no runs: a store that could not be read may
    hold exactly the failures the section exists to report.
    """


@dataclass(frozen=True)
class FailedRun:
    """One failed run, and the stage it failed at."""

    run_id: str
    ended_at: Optional[datetime]
    #: ``None`` when no stage of the run recorded a failed gate.
    stage_name: Optional[str]
    #: ``None`` when none was recorded; :data:`WITHHELD_GATE_REASON` when one
    #: was and the agent's privacy mode forbids showing it.
    gate_reason: Optional[str]


@dataclass(frozen=True)
class WorkflowOutcomes:
    """One definition's outcomes in the window, and its run of failures."""

    workflow_name: str
    completed: int
    #: Newest first.
    failed_runs: Tuple[FailedRun, ...]
    #: Failed runs since the definition last completed, at any time.
    consecutive_failures: int
    #: ``None`` when no completed run is on record.
    last_success_at: Optional[datetime]

    @property
    def failed(self) -> int:
        return len(self.failed_runs)


@dataclass(frozen=True)
class WorkflowRunReport:
    """The agent's workflow run outcomes in ``[since, until]``."""

    since: datetime
    until: datetime
    persistent_failure_runs: int
    #: Every definition with an outcome in the window, by name.
    workflows: Tuple[WorkflowOutcomes, ...]
    #: Whether the Workflows store exists on this agent at all.
    store_present: bool

    def persistently_failing(self) -> Tuple[WorkflowOutcomes, ...]:
        """Definitions whose last ``persistent_failure_runs`` outcomes failed.

        Only a definition with a failure in the window: the three reads are
        not one snapshot, and a failure that ended between them is counted
        in its run of failures without being one the window can name.
        """
        return tuple(
            outcomes
            for outcomes in self.workflows
            if outcomes.failed_runs
            and outcomes.consecutive_failures >= self.persistent_failure_runs
        )


WorkflowRunsAssessment = Union[WorkflowRunReport, WorkflowRunsNotAssessed]


def persistent_failure_runs_setting(data: Mapping[str, Any]) -> int:
    """``morning_signal_config.persistent_failure_runs``, or the default.

    Raises :class:`WorkflowRunsNotAssessed` for a value that is not a positive
    integer rather than guessing what the operator meant.
    """
    config = data.get("morning_signal_config")
    value = config.get(PERSISTENT_FAILURE_RUNS_KEY) if isinstance(config, Mapping) else None
    if value is None:
        return DEFAULT_PERSISTENT_FAILURE_RUNS
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkflowRunsNotAssessed(
            f"morning_signal_config.{PERSISTENT_FAILURE_RUNS_KEY} must be a "
            f"positive integer, not {value!r}"
        )
    return value


async def assess_workflow_runs(
    agent: Any,
    data: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> WorkflowRunsAssessment:
    """The report for ``agent``, or why there is none, for the briefing."""
    try:
        return await read_workflow_run_report(
            agent,
            persistent_failure_runs=persistent_failure_runs_setting(data),
            now=now,
        )
    except WorkflowRunsNotAssessed as exc:
        logger.warning("Morning Signal could not assess workflow runs: %s", exc)
        return exc


async def read_workflow_run_report(
    agent: Any,
    *,
    persistent_failure_runs: int = DEFAULT_PERSISTENT_FAILURE_RUNS,
    now: Optional[datetime] = None,
) -> WorkflowRunReport:
    """Outcomes of ``agent``'s workflow runs that ended in the 24 hours to ``now``.

    An agent whose database has no Workflows store has run no workflows: the
    report is empty. Raises :class:`WorkflowRunsNotAssessed` when the agent has
    no DID or database, when only one of the two store tables exists, or when
    reading them fails. Gate reasons, and the text of a store read error, are
    withheld when the agent's privacy mode hides persisted user content.
    """
    if isinstance(persistent_failure_runs, bool) or persistent_failure_runs < 1:
        raise ValueError("persistent_failure_runs must be a positive integer")
    try:
        owner_did = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable as exc:
        raise WorkflowRunsNotAssessed(
            "the agent has no DID to scope its workflow runs to"
        ) from exc
    db = resolve_feature_database(agent)
    if db is None:
        raise WorkflowRunsNotAssessed(
            "the agent has no database to read workflow runs from"
        )
    withhold_content = hides_persisted_user_content(agent)
    until = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    since = until - WINDOW
    try:
        backend = db.backend
        present = {
            table: await backend.table_exists(table)
            for table in (RUNS_TABLE, STAGE_LINKS_TABLE)
        }
        if not any(present.values()):
            return WorkflowRunReport(
                since=since,
                until=until,
                persistent_failure_runs=persistent_failure_runs,
                workflows=(),
                store_present=False,
            )
        missing = [table for table, exists in present.items() if not exists]
        if missing:
            raise WorkflowRunsNotAssessed(
                f"the Workflows store is incomplete: {', '.join(missing)} is missing"
            )
        start = _timestamp_param(backend, since)
        end = _timestamp_param(backend, until)
        window_rows = await backend.fetch_all(
            _WINDOW_RUNS_SQL, (owner_did, start, end)
        )
        success_rows = await backend.fetch_all(_LAST_SUCCESS_SQL, (owner_did, end))
        streak_rows = await backend.fetch_all(
            _FAILURES_SINCE_SUCCESS_SQL, (owner_did, end, owner_did, end)
        )
    except WorkflowRunsNotAssessed:
        raise
    except Exception as exc:  # noqa: BLE001 - store boundary; reported, not swallowed
        detail = (
            f"{type(exc).__name__} (details withheld in this privacy mode)"
            if withhold_content
            else str(exc)
        )
        raise WorkflowRunsNotAssessed(
            f"reading the Workflows store failed: {detail}"
        ) from exc

    last_success = {str(name): _instant(at) for name, at in success_rows}
    streaks = {str(name): int(count) for name, count in streak_rows}
    return WorkflowRunReport(
        since=since,
        until=until,
        persistent_failure_runs=persistent_failure_runs,
        workflows=_outcomes(
            window_rows,
            last_success,
            streaks,
            withhold_reasons=withhold_content,
        ),
        store_present=True,
    )


def _outcomes(
    rows: List[Tuple[Any, ...]],
    last_success: Dict[str, Optional[datetime]],
    streaks: Dict[str, int],
    *,
    withhold_reasons: bool,
) -> Tuple[WorkflowOutcomes, ...]:
    # One entry per run; a later row for the same run is a later failing link.
    runs: Dict[str, Tuple[str, str, Optional[datetime], Any, Any]] = {}
    for run_id, name, status, ended_at, stage_name, gate_reason in rows:
        runs[str(run_id)] = (str(name), str(status), _instant(ended_at), stage_name, gate_reason)

    completed: Dict[str, int] = {}
    failed: Dict[str, List[FailedRun]] = {}
    for run_id, (name, status, ended_at, stage_name, gate_reason) in runs.items():
        completed.setdefault(name, 0)
        failed.setdefault(name, [])
        if status == COMPLETED:
            completed[name] += 1
        else:
            failed[name].append(
                FailedRun(
                    run_id=run_id,
                    ended_at=ended_at,
                    stage_name=_text(stage_name),
                    gate_reason=_gate_reason(gate_reason, withhold=withhold_reasons),
                )
            )
    return tuple(
        WorkflowOutcomes(
            workflow_name=name,
            completed=completed[name],
            failed_runs=tuple(sorted(failed[name], key=_newest_first)),
            consecutive_failures=streaks.get(name, 0),
            last_success_at=last_success.get(name),
        )
        for name in sorted(completed)
    )


def render_workflow_runs_section(assessment: WorkflowRunsAssessment) -> List[str]:
    """The Morning Signal's "Workflow runs (24h)" section, as markdown lines."""
    lines = [f"## {SECTION_TITLE}"]
    if isinstance(assessment, WorkflowRunsNotAssessed):
        lines.append(f"Workflow runs could not be assessed: {assessment}")
        return lines
    if not assessment.workflows:
        if assessment.store_present:
            lines.append("No workflow runs finished in the last 24 hours.")
        else:
            lines.append(
                "No workflow runs finished in the last 24 hours: this agent "
                "has no Workflows run store."
            )
        return lines
    for outcomes in assessment.persistently_failing():
        lines.append(persistent_failure_line(outcomes))
    for outcomes in assessment.workflows:
        lines.append(
            f"- `{outcomes.workflow_name}`: {outcomes.completed} completed, "
            f"{outcomes.failed} failed"
        )
        lines.extend(_failure_lines(outcomes))
    return lines


def persistent_failure_line(outcomes: WorkflowOutcomes) -> str:
    """The call-out for a definition whose recent runs all failed."""
    latest = outcomes.failed_runs[0]
    return (
        f"- **PERSISTENTLY FAILING** `{outcomes.workflow_name}`: its last "
        f"{outcomes.consecutive_failures} runs failed. Latest run "
        f"`{latest.run_id}` failed {_failure_point(latest)}. "
        f"Last success: {last_success_date(outcomes)}."
    )


def last_success_date(outcomes: WorkflowOutcomes) -> str:
    """The UTC date the definition last completed, or that none is recorded."""
    if outcomes.last_success_at is None:
        return "none on record"
    return outcomes.last_success_at.date().isoformat()


def _failure_lines(outcomes: WorkflowOutcomes) -> List[str]:
    groups: Dict[Tuple[Optional[str], Optional[str]], List[FailedRun]] = {}
    for run in outcomes.failed_runs:
        groups.setdefault((run.stage_name, _reason_text(run.gate_reason)), []).append(run)
    # Most frequent first; a tie goes to the most recent. Runs are newest
    # first, so each group's first run is its latest.
    ordered = sorted(
        groups.values(),
        key=lambda runs: (-len(runs), _newest_first(runs[0])),
    )
    lines = []
    for runs in ordered[:_FAILURE_GROUP_LIMIT]:
        latest = runs[0]
        when = f" at {_format_instant(latest.ended_at)}" if latest.ended_at else ""
        lines.append(
            f"  - {len(runs)} failed {_failure_point(latest)} "
            f"(latest run `{latest.run_id}`{when})"
        )
    hidden = ordered[_FAILURE_GROUP_LIMIT:]
    if hidden:
        lines.append(
            f"  - {len(hidden)} more distinct failure(s) across "
            f"{sum(len(runs) for runs in hidden)} run(s) not listed"
        )
    return lines


def _failure_point(run: FailedRun) -> str:
    reason = _reason_text(run.gate_reason) or "no gate reason recorded"
    if run.stage_name is None:
        return f"with no failing stage recorded: {reason}"
    return f"at stage `{run.stage_name}`: {reason}"


def _reason_text(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    text = _WHITESPACE.sub(" ", reason).strip()
    if not text:
        return None
    if len(text) > _REASON_LIMIT:
        return text[:_REASON_LIMIT].rstrip() + "…"
    return text


def _gate_reason(value: Any, *, withhold: bool) -> Optional[str]:
    reason = _text(value)
    if reason is not None and withhold:
        return WITHHELD_GATE_REASON
    return reason


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _newest_first(run: FailedRun) -> Tuple[bool, float]:
    # Runs with no readable end time sort after every dated one.
    if run.ended_at is None:
        return (True, 0.0)
    return (False, -run.ended_at.timestamp())


def _format_instant(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def _instant(value: Any) -> Optional[datetime]:
    """A stored timestamp as an aware UTC ``datetime``, or ``None``.

    PostgreSQL returns ``TIMESTAMPTZ`` values as datetimes; SQLite stores the
    ISO-8601 text the Workflows store wrote.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return parse_instant(value)


def _timestamp_param(backend: Any, moment: datetime) -> Any:
    """``moment`` bound the way each backend stores a Workflows timestamp."""
    if backend.backend_type == "postgres":
        return TimestamptzParameter(moment)
    return moment.isoformat()
