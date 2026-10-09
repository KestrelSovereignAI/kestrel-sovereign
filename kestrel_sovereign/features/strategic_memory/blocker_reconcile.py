"""Reconcile ledger blockers against live GitHub issue state.

57 of Emma's 110 blocker rows referenced issues GitHub had already closed
(#2954). Nothing was wrong with the rows when they were written — there was
simply no path from "the issue closed" back to "the blocker is stale", so the
list only ever grew and the agent reasoned over a backlog that had partly
ceased to exist.

This module supplies that path. It reads GitHub and reports; applying the
result is a separate, explicit decision by the caller, because closing a
GitHub issue is not by itself proof that the strategic blocker it stood for is
gone. The scheduler makes that decision once a day: the
``strategy_reconcile_blockers`` cron source runs with ``apply='yes'`` (#3537).

A row can only be reconciled if it can be read as one issue in one
repository, and rows are written in several shapes: ``repo: self``, a bare
repository name (``repo: kestrel-feature-talon`` with ``issue:
kestrel-feature-talon#46``), and bare numbers with no repository at all. 70 of
Emma's 122 active rows were in one of those shapes and could never close
(#3537). :func:`resolve_blocker_reference` is the one reading of every shape,
used both when a row is written and when it is reconciled.

A reading that has to choose between two repositories refuses instead (#3540).
``repo: self`` with ``issue: owner/other#46`` names two repositories, and
reading the number in the declared one retired a blocker on the home
repository's closed issue 46 while ``owner/other#46`` was still open. A short
name that more than one configured repository has does not say whose it is.
:func:`issue_repository_conflict` is the rule a row's repository has to pass,
and issue dispatch applies the same function.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .github_integration import get_github_self_repo, get_github_token, github_api_get
from .ledger import active_blockers
from .timestamps import parse_instant

logger = logging.getLogger(__name__)

#: Returned as ``reason`` when a row simply cannot be checked. Kept distinct
#: from "open" so a lookup failure is never reported as a live blocker
#: confirmed still blocking.
UNRESOLVABLE = "unresolvable"

#: Returned as ``reason`` when the row names an issue number but not which
#: repository it lives in, more than one is configured, and the agent's own
#: repository is not among them. Distinct from ``UNRESOLVABLE`` because the
#: row is perfectly well-formed -- what is missing is the identity of the
#: project, and no amount of retrying supplies it.
AMBIGUOUS_REPO = "ambiguous_repository"

#: Returned as ``reason`` when the row names its repository by a short name
#: (``widgets``) that more than one configured repository has (``Acme/widgets``
#: and ``Other/widgets``). Which owner the row meant is unknown, so neither is
#: read and the owner of the agent's own repository is not assumed (#3540).
AMBIGUOUS_REPO_NAME = "ambiguous_repository_name"

#: Returned as ``reason`` when the issue reference is written in a repository
#: that is not the row's: ``repo: self`` with ``issue: owner/other#46``. The
#: number belongs to one of them, and reading it in the other reaches an
#: unrelated issue that happens to share it (#3540).
CONFLICTING_REPOS = "conflicting_repositories"

#: Returned as ``reason`` when the row names no repository, one was assumed,
#: and GitHub's answer shows the assumed issue cannot be the one the row
#: meant: it was closed before the blocker was recorded, or opened after.
INFERRED_REPO_MISMATCH = "inferred_repository_mismatch"

#: ``reason_code`` values a reconcile that could not run carries, so a
#: scheduled run that failed names its cause in ``signal_log`` (#3184).
NO_REPOSITORY_REASON_CODE = "NO_BLOCKER_REPOSITORY"
NO_TOKEN_REASON_CODE = "NO_GITHUB_TOKEN"

#: The repository alias for the agent's own repository (``GITHUB_SELF_REPO``),
#: as the GitHub feature spells it.
SELF_REPO_ALIAS = "self"

#: Where a reference's repository came from. Only the first two are the row's
#: own statement; the other two are inferred, and an inference is checked
#: against GitHub's answer before anything is resolved on it.
REPO_FROM_ROW = "row"
REPO_FROM_ISSUE = "issue"
REPO_FROM_LONE_SCAN_REPO = "lone_scan_repo"
REPO_FROM_SELF_DEFAULT = "self_default"
_INFERRED_SOURCES = frozenset({REPO_FROM_LONE_SCAN_REPO, REPO_FROM_SELF_DEFAULT})

#: ``owner/name`` in GitHub's allowed character set. A reference prefix that
#: does not match this (or :data:`REPO_NAME_SHAPE`) is prose, not a repository.
REPO_SHAPE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")

#: A repository name without its owner: ``kestrel-feature-talon``.
REPO_NAME_SHAPE = re.compile(r"[A-Za-z0-9._-]*[A-Za-z0-9][A-Za-z0-9._-]*")

#: The repository written immediately before the number: ``kestrel-talon#252``,
#: or ``owner/kestrel-talon#252`` inside prose that :func:`split_issue_reference`
#: does not read as a reference. Text before a spaced ``#`` is prose
#: ("Issue #123"), so ``kestrel-talon #252`` cannot be told from it and names
#: no repository.
_WRITTEN_REPOSITORY = re.compile(
    r"(?:([A-Za-z0-9._-]+)/)?([A-Za-z0-9._-]+)#\s*[0-9]+\s*$"
)

#: How far apart a blocker's recorded date and an issue's open/close instant
#: may be before an inferred repository is judged wrong. ``blocked_since`` is
#: the recording host's local calendar date; GitHub stamps UTC.
_RECORDING_DATE_SLACK = timedelta(days=1)

#: The ledger key holding the outcome of the last applied reconcile, which the
#: morning briefing reports so a gap stays visible after the run (#3537).
RECONCILIATION_KEY = "blocker_reconciliation"

#: The reconcile is scheduled daily; a last run older than this means the
#: schedule stopped, and the briefing says so instead of repeating old counts.
RECONCILIATION_STALE_AFTER = timedelta(hours=36)


def split_issue_reference(value: object) -> Tuple[Optional[str], Optional[int]]:
    """Split an issue reference into the repository it names and its number.

    Returns ``(repository, number)`` with the repository as written:
    ``owner/repo`` from ``owner/repo#12``, ``name`` from ``name#12``, or
    ``None`` when the reference names none (``#12``, ``12``). A bare name is
    only read as one when it touches the ``#``: text before a spaced ``#`` is
    prose ("Issue #123"), not a repository. Either half may be ``None``.
    """
    text = str(value or "").strip()
    if not text:
        return None, None
    written: Optional[str] = None
    if "#" in text:
        raw_head, _, tail = text.partition("#")
        head = raw_head.strip().strip("/")
        # Only an owner/repo (or name) shape names a repository. Treating ANY
        # non-empty prefix as one turned "Issue #123" and "see FIXME #7" into
        # the repos "Issue" and "see FIXME".
        if REPO_SHAPE.fullmatch(head):
            written = head
        elif raw_head == raw_head.rstrip() and REPO_NAME_SHAPE.fullmatch(head):
            written = head
        text = tail.strip()
    text = text.lstrip("#").strip()
    try:
        return written, int(text)
    except (TypeError, ValueError):
        return written, None


def parse_issue_ref(value: object) -> Tuple[Optional[str], Optional[int]]:
    """Split an issue reference into its ``owner/repo`` and its number.

    ``strategy_add_blocker`` documents that it accepts a qualified reference
    (``owner/repo#123``), so every consumer has to be able to read one back.
    Doing that at each call site is how ``int("owner/repo#123")`` ended up in
    the dispatch path: the string carries two facts and the reader wanted one.

    Returns ``(repo, number)``. Either may be ``None``: a bare ``#123`` has no
    repository, a short ``name#123`` names none this function can vouch for
    (see :func:`resolve_blocker_reference` for that), and an unparseable
    reference has no number.
    """
    written, number = split_issue_reference(value)
    if written is not None and not REPO_SHAPE.fullmatch(written):
        written = None
    return written, number


def configured_repos(strategy_data: Dict[str, Any]) -> List[str]:
    """The repositories STRATEGY.yaml says this agent scans.

    Each once, under the first spelling ``scan_repos`` gives it.
    """
    if not isinstance(strategy_data, dict):
        return []
    config = strategy_data.get("morning_signal_config", {})
    if not isinstance(config, dict):
        return []
    repos = config.get("scan_repos", [])
    if not isinstance(repos, list):
        return []
    return _distinct_repositories(
        str(r).strip() for r in repos if str(r).strip()
    )


def _distinct_repositories(repos: Iterable[str]) -> List[str]:
    """``repos`` with each repository once, under its first spelling.

    GitHub's names are case-insensitive, so ``Acme/widgets`` and
    ``acme/Widgets`` are one repository. Counted as two, a repeated or
    differently cased ``scan_repos`` entry made ``widgets#46`` ambiguous
    between a repository and itself, and a bare number ambiguous on an agent
    that scans one repository (#3542).
    """
    seen: Dict[str, str] = {}
    for repo in repos:
        seen.setdefault(repo.lower(), repo)
    return list(seen.values())


def normalize_repository(
    value: object, configured: List[str], self_repo: str
) -> Optional[str]:
    """``owner/repo`` for a repository as a row writes it, or ``None``.

    - ``self`` is the agent's own repository, ``GITHUB_SELF_REPO``.
    - ``owner/repo`` is already qualified.
    - A bare name is the configured scan repository of that name when exactly
      one has it. When none has it, it is the same name under the owner of
      the agent's own repository: ``kestrel-feature-talon`` is
      ``KestrelSovereignAI/kestrel-feature-talon`` on a Kestrel host. When
      several have it, it names none of them: ``widgets`` with
      ``Acme/widgets`` and ``Other/widgets`` both configured does not say
      whose it is. The owner fallback used to pick one anyway, and a
      reconcile then resolved the row on Acme's closed issue while Other's
      stayed open (#3540).
    - Anything else (prose, a URL) names no repository.
    """
    return _read_repository(value, configured, self_repo)[0]


def _read_repository(
    value: object, configured: List[str], self_repo: str
) -> Tuple[Optional[str], List[str]]:
    """``(owner/repo, [])``; ``(None, candidates)`` for an ambiguous name."""
    text = str(value or "").strip().strip("/")
    if not text:
        return None, []
    home = _home_repository(self_repo)
    if text.lower() == SELF_REPO_ALIAS:
        return home, []
    if REPO_SHAPE.fullmatch(text):
        return text, []
    if not REPO_NAME_SHAPE.fullmatch(text):
        return None, []
    named = _distinct_repositories(
        r for r in configured if _repository_name(r) == text.lower()
    )
    if len(named) > 1:
        return None, named
    if named:
        return named[0], []
    if home is None:
        return None, []
    return f"{home.split('/', 1)[0]}/{text}", []


def _home_repository(self_repo: str) -> Optional[str]:
    """The agent's own repository when it is ``owner/repo``, else ``None``."""
    return self_repo if REPO_SHAPE.fullmatch(self_repo or "") else None


def _repository_name(repo: str) -> str:
    """A repository's name without its owner, lowercased."""
    return repo.rsplit("/", 1)[-1].lower()


def issue_repository_conflict(
    issue: object, repo: str, self_repo: str
) -> Optional[str]:
    """The repository ``issue`` is written in, when that is not ``repo``.

    ``repo`` is the ``owner/repo`` a blocker row is read in. ``None`` when
    the reference names no repository, or names one that can be ``repo``:

    - ``owner/name`` is ``repo`` only when it is ``repo`` (any case);
    - ``self`` is ``repo`` only when ``repo`` is the agent's own repository;
    - a bare ``name`` is ``repo`` when ``repo`` has that name. The row's own
      repository identifies the owner, so ``repo: owner/other`` with
      ``issue: other#46`` is ``owner/other#46``.

    Both readings of the reference are checked: the one
    :func:`split_issue_reference` makes, and the repository written just
    before a trailing number inside prose (``blocked by other/core#5``).

    One rule for every reader that binds a number to a repository (#3540):
    the blocker reconciler refuses a conflicting row rather than resolve it,
    and issue dispatch refuses it rather than start work on it.
    """
    text = str(issue or "")
    written = [split_issue_reference(text)[0]]
    match = _WRITTEN_REPOSITORY.search(text)
    if match is not None:
        owner, name = match.groups()
        written.append(f"{owner}/{name}" if owner else name)
    for candidate in written:
        if candidate is not None and not _names_repository(candidate, repo, self_repo):
            return candidate
    return None


def _names_repository(written: str, repo: str, self_repo: str) -> bool:
    """Whether a repository as an issue reference writes it can be ``repo``."""
    if "/" in written:
        return written.lower() == repo.lower()
    if written.lower() == SELF_REPO_ALIAS:
        home = _home_repository(self_repo)
        return home is not None and home.lower() == repo.lower()
    return written.lower() == _repository_name(repo)


@dataclass(frozen=True)
class BlockerReference:
    """One blocker row read as one issue in one repository, or why it isn't.

    ``problem`` is ``None`` only when both ``repo`` and ``number`` are set. A
    row refused as :data:`CONFLICTING_REPOS` keeps the repository it is read
    in, so the report can name both; ``conflicting`` is the other one, as the
    issue reference writes it. ``candidates`` are the configured repositories
    an :data:`AMBIGUOUS_REPO_NAME` row could mean. ``source`` says where
    ``repo`` came from (``REPO_FROM_*``).
    """

    repo: Optional[str]
    number: Optional[int]
    source: Optional[str]
    problem: Optional[str] = None
    conflicting: Optional[str] = None
    candidates: Tuple[str, ...] = ()

    @property
    def inferred(self) -> bool:
        """Whether the repository was assumed rather than stated by the row."""
        return self.source in _INFERRED_SOURCES


def resolve_blocker_reference(
    row: Dict[str, Any], configured: List[str], self_repo: str
) -> BlockerReference:
    """The one issue this row names, read the same way at write and at reconcile.

    The repository comes from, in order: the row's ``repo``; the repository
    written in its ``issue`` (``owner/repo#N``, ``name#N``, ``self#N``); the
    lone configured scan repository; and, when several are configured, the
    agent's own repository if it is one of them. A row naming a bare number
    on an agent that scans several repositories means its home repository,
    the convention GitHub itself applies to an unqualified ``#N``.

    There is still no "try them all" branch. ``#42`` searched across every
    configured repo and bound to the first hit resolved a blocker whose issue
    42 was open in one project because a *different* project had closed its
    own issue 42. When the home repository is not among several configured
    ones, the row is ambiguous and is reported unchecked.

    Nor is there a "pick one" branch (#3540). A short name that several
    configured repositories have is :data:`AMBIGUOUS_REPO_NAME`, and a row
    whose issue reference is written in a repository other than the one it
    is read in is :data:`CONFLICTING_REPOS` (see
    :func:`issue_repository_conflict`). Both are reported unchecked.
    """
    issue = row.get("issue")
    written, number = split_issue_reference(issue)
    declared = str(row.get("repo") or "").strip()
    if declared or written:
        repo, candidates = _read_repository(declared or written, configured, self_repo)
        if candidates:
            return BlockerReference(
                None, number, None, AMBIGUOUS_REPO_NAME, candidates=tuple(candidates)
            )
        source = REPO_FROM_ROW if declared else REPO_FROM_ISSUE
    else:
        repo, source, problem = _infer_repository(configured, self_repo)
        if problem is not None:
            return BlockerReference(None, number, None, problem)
    if repo is None or number is None:
        return BlockerReference(repo, number, source, UNRESOLVABLE)
    conflicting = issue_repository_conflict(issue, repo, self_repo)
    if conflicting is not None:
        return BlockerReference(
            repo, number, source, CONFLICTING_REPOS, conflicting=conflicting
        )
    return BlockerReference(repo, number, source)


def _infer_repository(
    configured: List[str], self_repo: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(repo, source, problem)`` for a row that names no repository."""
    configured = _distinct_repositories(configured)
    if len(configured) == 1:
        # Exactly one configured repository: unqualified is unambiguous.
        return configured[0], REPO_FROM_LONE_SCAN_REPO, None
    if not configured:
        # Nothing says this agent's blockers are GitHub issues at all.
        return None, None, UNRESOLVABLE
    home = next((r for r in configured if r.lower() == self_repo.lower()), None)
    if home is None:
        return None, None, AMBIGUOUS_REPO
    return home, REPO_FROM_SELF_DEFAULT, None


def _calendar_date(value: Any) -> Optional[date]:
    """The calendar date a ledger field records, or ``None``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def contradicts_inference(row: Dict[str, Any], issue: Dict[str, Any]) -> Optional[str]:
    """Why GitHub's issue cannot be the one an inferred row meant, or ``None``.

    A blocker waits on an issue that is open when it is recorded. An issue
    closed before the row was written, or opened after it, is some other
    project's issue with the same number -- low numbers exist in every
    repository. A row without ``blocked_since`` cannot be checked this way.
    """
    recorded = _calendar_date(row.get("blocked_since"))
    if recorded is None:
        return None
    closed = parse_instant(issue.get("closed_at"))
    if closed is not None and closed.date() < recorded - _RECORDING_DATE_SLACK:
        return (
            f"closed on {closed.date()}, before this blocker was recorded on "
            f"{recorded}"
        )
    created = parse_instant(issue.get("created_at"))
    if created is not None and created.date() > recorded + _RECORDING_DATE_SLACK:
        return (
            f"opened on {created.date()}, after this blocker was recorded on "
            f"{recorded}"
        )
    return None


async def check_blockers(
    ledger_data: Dict[str, Any],
    strategy_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Look up each active blocker's issue and classify it.

    Returns a report with ``closed``/``open``/``unresolvable`` row lists and a
    ``reason`` (with a ``reason_code``) when the check could not run at all. A
    missing token, or no ``scan_repos`` and no row naming a repository, is a
    *skipped* check, never an empty result set — reporting "0 stale blockers"
    because nothing was queried would be the same lie the ticket was filed
    about. With ``scan_repos`` configured, a row that names no usable
    repository (an ambiguous one, or two) is listed unresolvable with its
    reason rather than reported as a missing configuration.
    ``checked`` counts the rows whose live state was established.
    """
    rows = active_blockers(
        ledger_data.get("blockers", []) if isinstance(ledger_data, dict) else []
    )
    report: Dict[str, Any] = {
        "checked": 0,
        "closed": [],
        "open": [],
        "unresolvable": [],
        "ran": False,
    }
    if not rows:
        report["ran"] = True
        return report

    configured = configured_repos(strategy_data)
    self_repo = get_github_self_repo()
    references = [
        (row, resolve_blocker_reference(row, configured, self_repo)) for row in rows
    ]
    if not configured and not any(reference.repo for _, reference in references):
        report["reason"] = (
            "No scan_repos configured in morning_signal_config and no blocker "
            "carries an explicit repo -- nothing could be looked up."
        )
        report["reason_code"] = NO_REPOSITORY_REASON_CODE
        return report

    token = get_github_token()
    if not token:
        report["reason"] = (
            "No GITHUB_TOKEN found -- live blocker state could not be checked."
        )
        report["reason_code"] = NO_TOKEN_REASON_CODE
        return report

    report["ran"] = True
    for row, reference in references:
        entry: Dict[str, Any] = {
            "id": row.get("id"),
            "issue": row.get("issue"),
            "title": row.get("title"),
            "repo": reference.repo,
            "number": reference.number,
            "repo_source": reference.source,
        }
        if reference.problem is not None:
            entry["reason"] = reference.problem
            if reference.problem == AMBIGUOUS_REPO:
                entry["candidate_repos"] = list(configured)
            elif reference.problem == AMBIGUOUS_REPO_NAME:
                entry["candidate_repos"] = list(reference.candidates)
            elif reference.problem == CONFLICTING_REPOS:
                entry["conflicting_repo"] = reference.conflicting
            report["unresolvable"].append(entry)
            continue

        issue = await github_api_get(
            f"/repos/{reference.repo}/issues/{reference.number}", token
        )
        state = (
            str(issue["state"]).lower()
            if isinstance(issue, dict) and issue.get("state")
            else None
        )
        if state not in ("open", "closed"):
            entry["reason"] = UNRESOLVABLE
            report["unresolvable"].append(entry)
            continue
        entry["state"] = state
        for field in ("state_reason", "closed_at", "html_url"):
            if issue.get(field):
                entry[field] = issue[field]
        if reference.inferred:
            mismatch = contradicts_inference(row, issue)
            if mismatch is not None:
                entry["reason"] = INFERRED_REPO_MISMATCH
                entry["detail"] = mismatch
                report["unresolvable"].append(entry)
                continue
        report["checked"] += 1
        report[state].append(entry)
    return report


def closing_resolution(entry: Dict[str, Any], today: date) -> str:
    """The resolution note for a row resolved because its issue closed.

    Cites what GitHub said -- which issue, when it closed and why -- so a
    reader can audit an unattended resolution without re-asking GitHub.
    """
    issue = f"{entry.get('repo')}#{entry.get('number')}"
    closed = _calendar_date(entry.get("closed_at"))
    note = f"GitHub reports {issue} closed"
    if closed is not None:
        note += f" on {closed}"
    if entry.get("state_reason"):
        note += f" ({entry['state_reason']})"
    note += f"; resolved by blocker reconciliation on {today}."
    if entry.get("repo_source") in _INFERRED_SOURCES:
        note += (
            " The row names no repository; "
            f"{entry.get('repo')} was assumed."
        )
    return note


def reconciliation_summary(
    report: Dict[str, Any], resolved_ids: List[str], ran_at: datetime
) -> Dict[str, Any]:
    """What an applied reconcile records for the morning briefing."""
    summary: Dict[str, Any] = {
        "ran_at": ran_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "ran": bool(report.get("ran")),
        "checked": int(report.get("checked", 0)),
        "closed": len(report.get("closed", [])),
        "resolved": len(resolved_ids),
        "open": len(report.get("open", [])),
        "unresolvable": len(report.get("unresolvable", [])),
    }
    if not summary["ran"]:
        summary["reason"] = str(report.get("reason") or "")
    return summary


def describe_last_reconciliation(summary: Any, now: datetime) -> str:
    """One line for the briefing: what the last applied reconcile found.

    Says so when none is recorded, when it could not run, and when it is
    older than :data:`RECONCILIATION_STALE_AFTER`, so a stopped schedule does
    not keep presenting old counts as today's.
    """
    if not isinstance(summary, dict):
        return (
            "No blocker reconciliation has been recorded -- nothing has "
            "checked these rows against GitHub."
        )
    ran_at = parse_instant(summary.get("ran_at"))
    if ran_at is None:
        return (
            "The last blocker reconciliation recorded no time it ran, so its "
            "counts cannot be dated -- run !strategy-reconcile yes."
        )
    when = ran_at.strftime("%Y-%m-%d %H:%M UTC")
    if not summary.get("ran"):
        line = (
            f"Blocker reconciliation could not run at {when}: "
            f"{summary.get('reason') or 'no reason recorded'}"
        )
    else:
        line = (
            f"Blocker reconciliation at {when}: {summary.get('checked', 0)} "
            f"checked, {summary.get('resolved', 0)} resolved as closed, "
            f"{summary.get('unresolvable', 0)} could not be checked"
        )
        if summary.get("unresolvable"):
            line += " (run !strategy-reconcile to list them)"
        line += "."
    if now - ran_at > RECONCILIATION_STALE_AFTER:
        line += " This is more than a day old: the reconcile schedule has not run since."
    return line
