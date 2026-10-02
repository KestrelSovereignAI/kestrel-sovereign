"""What Talon's job registry says about each issue's most recent run (#3398).

``signal_dispatch`` sent #3093 to Talon on 09-16, 09-23 and 09-24. Every run
blocked at iteration 1 with no diff and no PR, each saying the same thing:
this epic has nothing to implement here. Selection only asked GitHub whether
the issue was open, unlabelled and free of an open PR, so a label cleared to
mean "acknowledged" re-armed the same run. The registry already knew how the
last run ended and when; this module is how selection reads it.

Core cannot import kestrel-feature-talon. It reads the registry through the
provider Talon registers for the ``talon`` wait kind on
``agent.wait_registry``, and only through one operation of it:

``finished_runs()``
    Returns ``{"complete": bool, "runs": [run, ...], "reason": str}`` where
    each run is ``{"job_id", "repo", "issue", "disposition",
    "completed_at"}``: one entry per terminal job, ``completed_at`` the
    ISO-8601 moment Talon terminalized it. ``complete`` is ``True`` only when
    every persisted job was read; an absent registry is complete and empty,
    while an unreadable, oversized, corrupt or partly parsed one is not, and
    the optional ``reason`` says why. It must be read-only: no reaping, no
    pushing of preserved work, no registry write.

Two properties of that operation are load-bearing, and the wait provider's
other methods have neither. ``active_handles()`` reloads the registry and
returns every job it holds, but a registry it cannot read logs and returns an
empty list, so "no previous runs" and "could not read the runs" look alike.
That is the #3093 failure again. ``poll()`` reaps jobs, pushes preserved work
to the remote and rewrites the registry, which is right for the reconciler's
tick and wrong for a read that ``signal_dispatch(mode='suggest')`` makes
while previewing. A provider without ``finished_runs()`` therefore has an
unconfirmed history, never an empty one, and neither of the other two is
called here.

Only the ``talon`` kind is read. A sibling provider such as ``A2AWaitable``
spreads peer-returned data into what it reports, and a peer must not be able
to stand down dispatch of an issue by naming it in a payload.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

from .timestamps import parse_instant

#: The wait kind kestrel-feature-talon registers (``TalonWaitable.kind``).
TALON_WAIT_KIND = "talon"

#: The provider operation run history is read from (see the module docstring).
FINISHED_RUNS_METHOD = "finished_runs"

#: kestrel-feature-talon ``run_disposition`` values for a run that stopped to
#: ask a question rather than open a PR: ``BLOCKED`` (Talon also labels the
#: issue ``agent-blocked``) and ``CLARIFYING`` (``agent-clarifying``). Core
#: cannot import that package, so this is a copy of the vocabulary;
#: ``test_the_retry_vocabulary_matches_talons`` pins it.
QUESTION_DISPOSITIONS = frozenset({"blocked", "clarifying"})

_ISSUE_NUMBER = re.compile(r"#?[0-9]+")


class RunHistoryUnreadable(Exception):
    """Talon's run history is not known to be complete.

    Raised rather than returning a partial history: a job that could not be
    read may be exactly the run that asked the question, and an issue whose
    last run is unknown is not an issue whose last run is known to be fine.
    """


@dataclass(frozen=True)
class TalonRun:
    """One finished Talon run against one issue."""

    job_id: str
    repo: str
    issue_number: int
    disposition: str
    completed_at: datetime

    @property
    def ended_with_question(self) -> bool:
        """Whether the run stopped to ask something instead of opening a PR."""
        return self.disposition in QUESTION_DISPOSITIONS


@dataclass(frozen=True)
class RunHistory:
    """The most recent finished Talon run per issue.

    Keyed by ``(repo.lower(), issue_number)``: GitHub repository names are
    case-insensitive, and the ledger and Talon's registry need not agree on
    case.
    """

    runs: Dict[Tuple[str, int], TalonRun] = field(default_factory=dict)

    @classmethod
    def from_runs(cls, runs: Iterable[TalonRun]) -> "RunHistory":
        latest: Dict[Tuple[str, int], TalonRun] = {}
        for run in runs:
            key = (run.repo.lower(), run.issue_number)
            current = latest.get(key)
            if current is None or run.completed_at > current.completed_at:
                latest[key] = run
        return cls(latest)

    def latest(self, repo: str, issue_number: int) -> Optional[TalonRun]:
        return self.runs.get((repo.lower(), issue_number))


async def read_run_history(agent: Any) -> RunHistory:
    """The most recent finished Talon run per issue on ``agent``.

    An agent with no ``talon`` wait provider has no Talon registry, so no
    issue on it has a last run: the history is empty. A registered provider
    whose ``finished_runs()`` is missing, fails, or does not report a complete
    read raises :class:`RunHistoryUnreadable`.
    """
    registry = getattr(agent, "wait_registry", None)
    provider = registry.get(TALON_WAIT_KIND) if registry is not None else None
    if provider is None:
        return RunHistory()
    read = getattr(provider, FINISHED_RUNS_METHOD, None)
    if not callable(read):
        raise RunHistoryUnreadable(
            f"the {TALON_WAIT_KIND!r} wait provider has no read-only "
            f"{FINISHED_RUNS_METHOD}() that reports whether it read every job"
        )
    try:
        report = read()
        if inspect.isawaitable(report):
            report = await report
    except Exception as exc:  # noqa: BLE001 - provider boundary; reported, not swallowed
        raise RunHistoryUnreadable(f"reading Talon's finished runs failed: {exc}") from exc
    if not isinstance(report, Mapping):
        raise RunHistoryUnreadable(
            f"Talon's {FINISHED_RUNS_METHOD}() returned {type(report).__name__}, "
            "not a report"
        )
    if report.get("complete") is not True:
        reason = report.get("reason")
        raise RunHistoryUnreadable(
            "Talon could not read its whole job registry"
            + (f": {reason}" if isinstance(reason, str) and reason.strip() else "")
        )
    entries = report.get("runs")
    if not isinstance(entries, list):
        raise RunHistoryUnreadable(
            f"Talon's {FINISHED_RUNS_METHOD}() reported no list of runs"
        )
    runs = []
    for entry in entries:
        run = _issue_run(entry)
        if run is not None:
            runs.append(run)
    return RunHistory.from_runs(runs)


def _issue_run(entry: Any) -> Optional[TalonRun]:
    """The run on an issue ``entry`` records, or ``None`` when it is not one.

    A job that names no issue (a batch, an ``iterate`` on a PR) is not a run
    on an issue. One that names an issue it does not identify, or a run with
    no completion time, cannot be placed: it may be the very run that asked
    the question, so the history is unreadable.
    """
    if not isinstance(entry, Mapping):
        raise RunHistoryUnreadable(
            f"Talon reported a run that is not a record ({type(entry).__name__})"
        )
    job_id = entry.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RunHistoryUnreadable("Talon reported a run with no job id")
    issue = entry.get("issue")
    if issue is None or issue == "":
        return None
    repo = entry.get("repo")
    issue_number = _issue_number(issue)
    if not isinstance(repo, str) or not repo.strip() or issue_number is None:
        raise RunHistoryUnreadable(
            f"Talon job {job_id} names issue {issue!r} in repository {repo!r}, "
            "which does not identify an issue"
        )
    completed_at = parse_instant(entry.get("completed_at"))
    if completed_at is None:
        # Talon stamps every job it terminalizes. Without the stamp nothing
        # can say whether a later comment or edit came after this run.
        raise RunHistoryUnreadable(
            f"Talon job {job_id} on {repo}#{issue_number} finished with no "
            "readable completion time"
        )
    disposition = entry.get("disposition")
    return TalonRun(
        job_id=job_id,
        repo=repo.strip(),
        issue_number=issue_number,
        disposition=disposition if isinstance(disposition, str) else "unknown",
        completed_at=completed_at,
    )


def _issue_number(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and _ISSUE_NUMBER.fullmatch(value.strip()):
        number = int(value.strip().lstrip("#"))
        return number if number > 0 else None
    return None
