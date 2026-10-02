"""What Talon's job registry says about each issue's most recent run (#3398).

``signal_dispatch`` sent #3093 to Talon on 09-16, 09-23 and 09-24. Every run
blocked at iteration 1 with no diff and no PR, each saying the same thing:
this epic has nothing to implement here. Selection only asked GitHub whether
the issue was open, unlabelled and free of an open PR, so a label cleared to
mean "acknowledged" re-armed the same run. The registry already knew how the
last run ended and when; this module is how selection reads it.

Core cannot import kestrel-feature-talon. It reads the registry through the
seam the wait reconciler already uses: the provider Talon registers for the
``talon`` wait kind on ``agent.wait_registry``. Its ``active_handles()``
enumerates every dispatched job in the durable registry, terminal ones
included, and ``poll()`` reports each job's repo, issue, disposition and
completion time -- the same fields the ``talon.job_complete`` wake carries.
That poll is the per-job read the reconciler already performs on every job
each tick, so reading the history adds no effect of its own.

Only that kind is read. The provider's poll data comes from Talon's own job
record; a sibling provider such as ``A2AWaitable`` spreads peer-returned data
into its poll, and a peer must not be able to stand down dispatch of an issue
by naming it in a payload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

from .timestamps import parse_instant

#: The wait kind kestrel-feature-talon registers (``TalonWaitable.kind``).
TALON_WAIT_KIND = "talon"

#: kestrel-feature-talon ``run_disposition`` values for a run that stopped to
#: ask a question rather than open a PR: ``BLOCKED`` (Talon also labels the
#: issue ``agent-blocked``) and ``CLARIFYING`` (``agent-clarifying``). Core
#: cannot import that package, so this is a copy of the vocabulary;
#: ``test_the_retry_vocabulary_matches_talons`` pins it.
QUESTION_DISPOSITIONS = frozenset({"blocked", "clarifying"})

_ISSUE_NUMBER = re.compile(r"#?[0-9]+")


class RunHistoryUnreadable(Exception):
    """Talon's job registry is present but could not be read completely.

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
    that cannot be read completely raises :class:`RunHistoryUnreadable`.
    """
    registry = getattr(agent, "wait_registry", None)
    provider = registry.get(TALON_WAIT_KIND) if registry is not None else None
    if provider is None:
        return RunHistory()
    list_jobs = getattr(provider, "active_handles", None)
    if not callable(list_jobs):
        raise RunHistoryUnreadable(
            f"the {TALON_WAIT_KIND!r} wait provider cannot list its jobs"
        )
    try:
        handles = await list_jobs()
    except Exception as exc:  # noqa: BLE001 - provider boundary; reported, not swallowed
        raise RunHistoryUnreadable(f"listing Talon jobs failed: {exc}") from exc
    runs = []
    for handle in handles:
        try:
            status = await provider.poll(handle)
        except Exception as exc:  # noqa: BLE001 - provider boundary; reported, not swallowed
            raise RunHistoryUnreadable(
                f"reading Talon job {handle} failed: {exc}"
            ) from exc
        run = _finished_issue_run(str(handle), status)
        if run is not None:
            runs.append(run)
    return RunHistory.from_runs(runs)


def _finished_issue_run(job_id: str, status: Any) -> Optional[TalonRun]:
    """The finished run ``status`` describes, or ``None`` when it is not one.

    Still running is not a finished run, and a job with no repository and
    issue (a batch, an ``iterate`` on a PR, a job the registry no longer
    holds) is not a run on an issue.
    """
    outcome = getattr(status, "outcome", None)
    if outcome is None or not outcome.is_terminal():
        return None
    data = getattr(status, "data", None) or {}
    repo = data.get("repo")
    issue_number = _issue_number(data.get("issue"))
    if not isinstance(repo, str) or not repo.strip() or issue_number is None:
        return None
    completed_at = parse_instant(data.get("completed_at"))
    if completed_at is None:
        # Talon stamps every job it terminalizes. Without the stamp nothing
        # can say whether a later comment or edit came after this run.
        raise RunHistoryUnreadable(
            f"Talon job {job_id} on {repo}#{issue_number} finished with no "
            "readable completion time"
        )
    disposition = data.get("disposition")
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
