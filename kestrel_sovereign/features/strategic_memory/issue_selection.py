"""Select the highest-priority actionable GitHub issue.

Issue selection is strategic-memory behavior. Dispatching the selected issue
belongs to an independently installed coding feature or another orchestrator.

Selection is an allow-list, applied before anything is ranked (#3464). An
issue is a candidate only when GitHub serves it, open, from the scanned
repository the candidate names, labelled ``agent-ready``. The strategy ledger
only orders the issues that pass. Refusing bad picks one at a time did not
converge: the run-history gate (#3398) withheld #3093, and the next morning a
high-severity ledger row picked #3319, a ``bug``-only issue whose fix belonged
to another repository.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.features.strategic_memory.blocker_reconcile import (
    configured_repos,
)
from kestrel_sovereign.features.strategic_memory.github_integration import (
    get_github_token,
    github_api_get,
    github_api_post,
)
from kestrel_sovereign.features.strategic_memory.run_history import (
    RunHistory,
    TalonRun,
)
from kestrel_sovereign.features.strategic_memory.timestamps import parse_instant

logger = logging.getLogger(__name__)

#: ``owner/name`` in GitHub's allowed character set. A reference prefix that
#: does not match this is prose, not a repository.
_REPO_SHAPE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")

#: The repository written immediately before the number: ``kestrel-talon#252``,
#: or ``owner/kestrel-talon#252`` inside prose that :func:`parse_issue_ref`
#: does not read as a reference. Text before a spaced ``#`` is prose
#: ("Issue #123"), so ``kestrel-talon #252`` cannot be told from it and names
#: no repository; the allow-list still governs what that row can reach.
_WRITTEN_REPOSITORY = re.compile(
    r"(?:([A-Za-z0-9._-]+)/)?([A-Za-z0-9._-]+)#\s*[0-9]+\s*$"
)

#: Labels Talon puts on an issue while a run owns it or is waiting on a
#: human: ``kestreltalon/config.py`` ``label_analyzing`` / ``label_clarifying``
#: / ``label_in_progress`` / ``label_blocked`` / ``label_failed``. Talon's own
#: failure comment says "remove the label to allow a retry": the label is the
#: retry protocol, and re-dispatching over it re-runs the same claim every
#: morning (#3294 -- #3051 four days running). ``agent-ready`` and
#: ``agent-complete`` are deliberately absent: the first is the authorization
#: selection requires (:data:`AGENT_READY_LABEL`), the second is a finished
#: run. A finished run whose pull request is still open is withheld by the PR
#: itself (#3317), not by the label: the label outlives a PR that was closed
#: unmerged. Core cannot import kestreltalon
#: (features are entry points, never imports), so this is a copy of that
#: vocabulary; ``test_talon_state_labels_match_talons_vocabulary`` pins it.
TALON_STATE_LABELS = frozenset(
    {
        "agent-analyzing",
        "agent-clarifying",
        "agent-claimed",
        "agent-blocked",
        "agent-failed",
    }
)

#: The label that authorizes agent work on an issue, and the only thing that
#: makes one a candidate (#3464). The orchestrator applies it during triage,
#: so it is the orchestrator's own authorization, not a human sign-off. A
#: strategy-ledger row is a note about an issue: it never makes one eligible.
AGENT_READY_LABEL = "agent-ready"

#: How blocker severity orders eligible blockers, most urgent first. Severity
#: orders; it never authorizes. A row of any other severity is not a blocker
#: here, though the milestone or backlog pass may still reach its issue.
_BLOCKER_SEVERITY_ORDER = {"critical": 0, "high": 1}

#: Exclusion reasons reported in ``diagnostics["eligibility_exclusions"]``:
#: why a candidate is not on the allow-list (#3464). Decided before ranking.
EXCLUDED_WRONG_REPO = "wrong_repo"
EXCLUDED_ISSUE_UNREADABLE = "issue_unreadable"
EXCLUDED_NOT_AN_ISSUE = "not_an_issue"
EXCLUDED_CLOSED = "closed"
EXCLUDED_TALON_OWNED = "talon_owned"
EXCLUDED_NOT_AGENT_READY = "not_agent_ready"


#: Days an open pull request may go without an update before an exclusion
#: names it stalled. A stalled PR is still not a fresh claim -- a second run
#: derives the same worktree and branch as the one in flight (#3101, #3108) --
#: but it is a candidate for rescue, and the dispatch output says so.
#: Overridable with ``morning_signal_config.stalled_pr_days``.
DEFAULT_STALLED_PR_DAYS = 3

#: GitHub's own issue-to-PR link: a ``Fixes #N`` / ``Closes #N`` keyword in a
#: PR, or a PR linked from the issue's sidebar, including cross-repository
#: PRs. Open PRs only; a merged one has already closed the issue.
_LINKED_PULL_REQUESTS_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      closedByPullRequestsReferences(first: 20, includeClosedPrs: false) {
        nodes { number state isDraft updatedAt url repository { nameWithOwner } }
      }
    }
  }
}
"""

#: Exclusion reasons reported in ``diagnostics["open_pr_exclusions"]``.
EXCLUDED_OPEN_PR = "open_pr"
EXCLUDED_STALLED_PR = "stalled_pr"
EXCLUDED_PR_LINKAGE_UNREADABLE = "pr_linkage_unreadable"

#: Exclusion reasons reported in ``diagnostics["run_exclusions"]`` by the
#: run-history gate (#3398), which only eligible candidates reach: the last
#: Talon run asked a question nothing since has answered, or GitHub could not
#: say whether anything has.
EXCLUDED_RUN_HISTORY = "run_history"
EXCLUDED_RUN_HISTORY_UNCONFIRMED = "run_history_unconfirmed"

#: Exclusions that withhold an issue because GitHub could not answer, not
#: because it answered "busy". Counted as unreadable so an outage renders as
#: one rather than as nothing to do.
_UNREADABLE_EXCLUSIONS = frozenset(
    {EXCLUDED_PR_LINKAGE_UNREADABLE, EXCLUDED_RUN_HISTORY_UNCONFIRMED}
)

#: Applied after a run that asked a question, :data:`AGENT_READY_LABEL` is a
#: deliberate go-ahead. Removing ``agent-blocked`` is not: it has been used
#: both for "retry" and for "acknowledged, stop re-picking it" (#3398), so
#: removal authorizes nothing.
RETRY_READINESS_LABELS = frozenset({AGENT_READY_LABEL})

#: Comment authors who speak for the repository: its maintainers, and the
#: orchestrator, whose account can label issues and so is at least a
#: collaborator. A stranger's comment on a public repository must not be able
#: to re-arm a dispatch that writes code.
_AUTHORIZING_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

#: How every comment Talon posts begins: ``**Kestrel Talon``
#: (``kestreltalon/models.py`` ``TALON_COMMENT_MARKER``, #185) or a
#: ``<!-- kestrel-talon:`` claim-record marker (``claim_record.py``). Talon
#: posts under the operator's own account, so author association cannot tell
#: its comments about a run from a maintainer's answer to it; its markers can.
#: ``test_the_retry_vocabulary_matches_talons`` pins the copy.
TALON_COMMENT_PREFIXES = ("**Kestrel Talon", "<!-- kestrel-talon:")

#: What happened on an issue since a given moment that could authorize a
#: retry: comments, labels applied, title renames, and the body's last edit.
#: The most recent items only -- anything that authorizes a retry is newer
#: than the run it follows, and a window full of newer items that authorize
#: nothing withholds, which is the safe direction.
_RETRY_EVIDENCE_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      lastEditedAt
      timelineItems(last: 50, itemTypes: [ISSUE_COMMENT, LABELED_EVENT, RENAMED_TITLE_EVENT]) {
        nodes {
          __typename
          ... on IssueComment { createdAt authorAssociation body }
          ... on LabeledEvent { createdAt label { name } }
          ... on RenamedTitleEvent { createdAt }
        }
      }
    }
  }
}
"""


def parse_issue_ref(value: object) -> tuple[Optional[str], Optional[int]]:
    """Split an issue reference into its repository and its number.

    ``strategy_add_blocker`` documents that it accepts a qualified reference
    (``owner/repo#123``), so every consumer has to be able to read one back.
    Doing that at each call site is how ``int("owner/repo#123")`` ended up in
    the dispatch path: the string carries two facts and the reader wanted one.
    Parsing once, here, keeps the repository and the number separable wherever
    a reference is recorded, reconciled or selected.

    Returns ``(repo, number)``. Either may be ``None``: a bare ``#123`` has no
    repository, and an unparseable reference has no number.
    """
    text = str(value or "").strip()
    if not text:
        return None, None
    repo: Optional[str] = None
    if "#" in text:
        head, _, tail = text.partition("#")
        head = head.strip().strip("/")
        # Only an owner/repo shape names a repository. Treating ANY non-empty
        # prefix as one turned "Issue #123" and "see FIXME #7" into the repos
        # "Issue" and "see FIXME", which pick_top_issue would then dispatch
        # against — and, because it returns on the first candidate, a
        # handwritten reference like that masked every valid blocker behind it.
        # This closed one wrong-repository path by opening another.
        if _REPO_SHAPE.fullmatch(head):
            repo = head
        text = tail.strip()
    text = text.lstrip("#").strip()
    try:
        return repo, int(text)
    except (TypeError, ValueError):
        return repo, None


async def pick_top_issue(
    data: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
    run_history: Optional[RunHistory] = None,
) -> Optional[Dict[str, Any]]:
    """Return the highest-priority eligible issue represented by strategic memory.

    Eligibility is an allow-list, decided for every candidate before any is
    ranked (#3464): GitHub serves the issue, open, from the scanned
    repository the candidate names, and it is labelled
    :data:`AGENT_READY_LABEL` with no Talon state label. Blocker severity, and
    every other ledger signal, only orders the issues that pass -- a ledger
    row alone never makes one eligible. When none passes the answer is
    ``None``, never the top ledger row. ``diagnostics["eligibility_exclusions"]``
    names each refused candidate and why, so a ``suggest`` run is evidence
    that the allow-list held.

    ``None`` has two meanings a caller must be able to tell apart: nothing is
    actionable, or GitHub could not confirm anything. Pass ``diagnostics`` to
    have ``blockers_checked`` and ``blockers_unreadable`` filled in -- when
    every blocker checked was unreadable, "no actionable issue" would be a
    claim about a ledger nobody actually looked at.

    Eligible candidates then meet two more gates, in rank order.
    ``diagnostics["open_pr_exclusions"]`` lists every one passed over
    because an open pull request already works it (or GitHub could not say),
    so the dispatch can report what it skipped instead of staying silent.

    ``diagnostics["run_exclusions"]`` does the same for candidates whose most
    recent Talon run in ``run_history`` stopped to ask a question, with
    nothing on the issue since that authorizes a retry (#3398). ``None``
    means the caller has no Talon registry, so no issue has a last run.

    ``candidates_checked`` and ``candidates_unreadable`` do the same for the
    milestone and backlog passes: how many distinct candidates had their PR
    linkage looked up, and how many of those GitHub could not answer for. A
    candidate withheld only because its linkage was unreadable was never
    confirmed busy, so "no actionable issue" is not the answer (#3367).
    """
    if diagnostics is None:
        diagnostics = {}
    diagnostics.setdefault("blockers_checked", 0)
    diagnostics.setdefault("blockers_unreadable", 0)
    diagnostics.setdefault("blockers_talon_owned", 0)
    diagnostics.setdefault("candidates_checked", 0)
    diagnostics.setdefault("candidates_unreadable", 0)
    ineligible = diagnostics.setdefault("eligibility_exclusions", [])
    exclusions = diagnostics.setdefault("open_pr_exclusions", [])
    run_exclusions = diagnostics.setdefault("run_exclusions", [])
    token = get_github_token()
    if not token:
        logger.info("No GITHUB_TOKEN — cannot pick top issue")
        return None

    config = data.get("morning_signal_config", {})
    # Each scanned repository under the spelling scan_repos gives it. GitHub
    # names are case-insensitive; one spelling per repository is what lets the
    # per-issue reads below recognise the same issue across passes.
    scanned: Dict[str, str] = {}
    for scanned_repo in configured_repos(data):
        scanned.setdefault(scanned_repo.lower(), scanned_repo)
    stalled_after_days = _stalled_pr_days(config)

    reported: set = set()

    def refuse(exclusion: Dict[str, Any]) -> None:
        """Report an allow-list refusal, once per issue and reason."""
        key = (
            str(exclusion.get("repo") or "").lower(),
            exclusion["issue_number"],
            exclusion["reason"],
        )
        if key in reported:
            return
        reported.add(key)
        ineligible.append(exclusion)
        logger.info("Not dispatching: %s", describe_exclusion(exclusion))

    # One linkage read per distinct (repo, number) across all three passes:
    # a blocker's issue can reappear in its milestone and in the backlog scan.
    linkage: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}

    async def in_flight(repo: str, issue_number: int) -> Optional[Dict[str, Any]]:
        """The exclusion for an issue a pull request already works, or None.

        An open issue with an open PR against it is claimed and in flight: it
        stays open by design until the PR merges, so neither open-state nor
        labels can tell it apart from idle work. Re-dispatching it derives
        the same worktree as the in-flight run (#3317 -- #3310 re-picked
        while its PR #3311 was open and green).
        """
        key = (repo, issue_number)
        if key not in linkage:
            prs = await _fetch_open_linked_pull_requests(repo, issue_number, token)
            linkage[key] = _open_pr_exclusion(
                repo, issue_number, prs, stalled_after_days
            )
            if linkage[key] is not None:
                exclusions.append(linkage[key])
                logger.info(
                    "Not dispatching %s#%s: %s", repo, issue_number,
                    describe_exclusion(linkage[key]),
                )
        return linkage[key]

    # One post-run activity read per distinct (repo, number), like linkage.
    awaiting: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}

    async def awaiting_new_input(
        repo: str, issue_number: int
    ) -> Optional[Dict[str, Any]]:
        """The exclusion for an issue whose last Talon run is still the answer.

        A run that ended blocked or clarifying has said what it needs. Until
        something newer than that run -- a maintainer or orchestrator comment,
        an issue edit, ``agent-ready`` applied -- changes its input, a new
        dispatch is the same run again (#3398: #3093 dispatched three times,
        blocked three times, nothing to implement in this repository).
        Labels are not consulted for this: removing ``agent-blocked`` has
        meant both "retry" and "stop".
        """
        run = (
            run_history.latest(repo, issue_number)
            if run_history is not None
            else None
        )
        if run is None or not run.ended_with_question:
            return None
        key = (repo, issue_number)
        if key not in awaiting:
            evidence = await _fetch_retry_evidence(repo, issue_number, token)
            authorized_by = (
                None
                if evidence is None
                else _retry_authorization(evidence, run.completed_at)
            )
            if authorized_by is not None:
                logger.info(
                    "%s#%s: Talon job %s ended %s; retry authorized by %s",
                    repo, issue_number, run.job_id, run.disposition, authorized_by,
                )
                awaiting[key] = None
            else:
                awaiting[key] = _run_exclusion(
                    repo,
                    issue_number,
                    run,
                    EXCLUDED_RUN_HISTORY_UNCONFIRMED
                    if evidence is None
                    else EXCLUDED_RUN_HISTORY,
                )
                run_exclusions.append(awaiting[key])
                logger.info(
                    "Not dispatching %s#%s: %s", repo, issue_number,
                    describe_exclusion(awaiting[key]),
                )
        return awaiting[key]

    async def withheld(repo: str, issue_number: int) -> Optional[Dict[str, Any]]:
        """Why an eligible candidate must not be dispatched now, or ``None``."""
        return await in_flight(repo, issue_number) or await awaiting_new_input(
            repo, issue_number
        )

    # Distinct (repo, number) per milestone/backlog candidate: the same issue
    # can sit in a milestone and in the backlog scan, and is one candidate.
    candidates_checked: set = set()
    candidates_unreadable: set = set()

    async def first_free(repo: str, issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        eligible = []
        for issue in issues:
            number = issue.get("number")
            if isinstance(number, bool) or not isinstance(number, int):
                continue
            refusal = _ineligibility(repo, number, issue)
            if refusal is not None:
                refuse(refusal)
                continue
            eligible.append(issue)
        for candidate in _ranked_candidates(eligible):
            key = (repo, candidate["number"])
            exclusion = await withheld(*key)
            candidates_checked.add(key)
            diagnostics["candidates_checked"] = len(candidates_checked)
            if exclusion is None:
                return candidate
            if exclusion["reason"] in _UNREADABLE_EXCLUSIONS:
                # Withheld, correctly, but not confirmed busy: counted so an
                # outage cannot render as an empty backlog (#3367), as the
                # blocker pass does for its own unreadable reads.
                candidates_unreadable.add(key)
                diagnostics["candidates_unreadable"] = len(candidates_unreadable)
        return None

    # The most urgent row naming each distinct (repo, number). One GitHub read
    # per distinct target, not per row: the ledger only grows, and on the
    # live host 12 of the 33 qualifying rows named the same pull request -- a
    # guaranteed miss, read twelve times every run.
    rows: Dict[Tuple[str, int], Tuple[int, int, Dict[str, Any]]] = {}
    for index, blocker in enumerate(data.get("blockers", [])):
        severity = blocker.get("severity")
        rank = (
            _BLOCKER_SEVERITY_ORDER.get(severity) if isinstance(severity, str) else None
        )
        if rank is None or not blocker.get("issue"):
            continue
        ref_repo, issue_number = parse_issue_ref(blocker["issue"])
        if issue_number is None:
            # An unparseable reference is not dispatchable, and guessing a
            # number from it would dispatch against the wrong issue.
            continue
        repo, problem = _blocker_repository(blocker, ref_repo, issue_number, scanned)
        if problem is not None:
            refuse(_eligibility_exclusion(
                repo, issue_number, EXCLUDED_WRONG_REPO, detail=problem
            ))
            continue
        best = rows.get((repo, issue_number))
        if best is None or rank < best[0]:
            rows[(repo, issue_number)] = (rank, index, blocker)

    # Allow-list every blocker before ranking any. Eligibility is decided
    # against GitHub, not the ledger: the ledger only grows unless someone
    # reconciles it, and nothing schedules that. On the live host 161 blocker
    # rows were all unresolved, several naming issues closed for weeks, and
    # the dispatch path picked one of them -- a ticket closed on 2026-07-28,
    # under a title that was not the issue's own -- every morning.
    eligible_blockers = []
    for (repo, issue_number), (rank, index, blocker) in rows.items():
        issue = await _fetch_issue(repo, issue_number, token)
        diagnostics["blockers_checked"] += 1
        if issue is None:
            diagnostics["blockers_unreadable"] += 1
        refusal = _ineligibility(repo, issue_number, issue)
        if refusal is not None:
            if refusal["reason"] == EXCLUDED_TALON_OWNED:
                diagnostics["blockers_talon_owned"] += 1
            refuse(refusal)
            continue
        eligible_blockers.append((rank, index, repo, issue_number, issue, blocker))

    # Severity orders the eligible blockers; the ledger's order breaks ties.
    eligible_blockers.sort(key=lambda candidate: candidate[:2])
    for _, _, repo, issue_number, issue, blocker in eligible_blockers:
        exclusion = await withheld(repo, issue_number)
        if exclusion is not None:
            if exclusion["reason"] in _UNREADABLE_EXCLUSIONS:
                # Not confirmed free of in-flight work (or of a run still
                # waiting on an answer) is not confirmed at all: it counts
                # with the unreadable issues, so a GitHub outage still
                # renders as one rather than as "nothing to do".
                diagnostics["blockers_unreadable"] += 1
            continue
        return {
            "repo": repo,
            "issue_number": issue_number,
            # The issue's own title, which is what Talon will work from. The
            # ledger row's title is a note someone wrote about it, and on the
            # live host it described a different problem than the issue did.
            "issue_title": issue.get("title") or blocker.get("title", "Blocker"),
            "priority": "high",
            "context": (
                f"Blocker (severity: {blocker.get('severity')}): "
                f"{blocker.get('title', '')}. {blocker.get('notes', '')}"
            ).strip(),
        }

    # #2813: the retired handoff fetched the full morning-signal projection
    # here but never consumed it, adding network/auth failure modes without
    # changing a candidate. Deliberately do not restore that no-op call. Issue
    # selection below owns targeted milestone/open-issue reads, while the YAML
    # strategic state owns priority/context; ``morning_signal`` separately owns
    # its broad briefing projection.

    milestones = [
        milestone
        for milestone in data.get("milestones", [])
        if milestone.get("status") in ("at_risk", "in_progress")
    ]
    for milestone in milestones:
        for milestone_repo in milestone.get("repos", []):
            repo = scanned.get(str(milestone_repo).lower())
            if repo is None:
                continue
            milestone_name = milestone.get("name", "")
            issues = await _fetch_milestone_issues(repo, milestone_name, token)
            pick = await first_free(repo, issues)
            if pick:
                return {
                    "repo": repo,
                    "issue_number": pick["number"],
                    "issue_title": pick["title"],
                    "priority": (
                        "high" if milestone.get("status") == "at_risk" else "normal"
                    ),
                    "context": (
                        f"Milestone: {milestone_name} ({milestone.get('status')}). "
                        f"{milestone.get('critical_path', '')}"
                    ),
                }

    for repo in scanned.values():
        pick = await first_free(repo, await _fetch_open_issues(repo, token, limit=5))
        if pick:
            return {
                "repo": repo,
                "issue_number": pick["number"],
                "issue_title": pick["title"],
                "priority": "normal",
                "context": "Open issue from backlog scan",
            }
    return None


def _blocker_repository(
    blocker: Dict[str, Any],
    ref_repo: Optional[str],
    issue_number: int,
    scanned: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    """The scanned repository a blocker row dispatches against, or why none.

    Returns ``(repo, problem)``. Without a problem, ``repo`` is spelled as
    ``scan_repos`` spells it. With one, ``repo`` is the repository the row
    would have dispatched against (``None`` when it names none), for the
    report.

    A row naming an unscanned repository used to be dispatched anyway, on the
    reasoning that it was the least ambiguous kind of row. It is the kind that
    sends work to a repository this agent was never asked to work in (#3464).
    """
    declared = str(blocker.get("repo") or "").strip()
    if declared and ref_repo and declared.lower() != ref_repo.lower():
        # The number belongs to the reference's repository. Dispatching it
        # against the declared one reaches an unrelated issue that happens to
        # share its number.
        return declared, (
            f"the ledger row's issue reference is {ref_repo}#{issue_number}, "
            "in another repository"
        )
    named = declared or ref_repo
    if not named:
        # A bare number names no project when several are scanned. This used
        # to walk scan_repos and take the FIRST repository that had any issue
        # with that number -- with fourteen repos, low numbers collide
        # everywhere. The blocker reconciler and ``_resolve_blocker_repo``
        # both refuse this guess. A lone scanned repository is not a guess.
        if len(scanned) != 1:
            return None, (
                f"the ledger row names no repository, and {len(scanned)} are "
                "scanned"
            )
        named = next(iter(scanned.values()))
    repo = scanned.get(named.lower())
    if repo is None:
        return named, f"{named} is not in morning_signal_config.scan_repos"
    written = _WRITTEN_REPOSITORY.search(str(blocker.get("issue") or ""))
    if written is not None:
        owner, name = written.groups()
        repo_owner, _, repo_name = repo.partition("/")
        if name.lower() != repo_name.lower() or (
            owner is not None and owner.lower() != repo_owner.lower()
        ):
            # ``talon#5`` is Talon's issue 5. The row's repository -- bound
            # when it was written, or the lone scanned one -- is not where the
            # number came from.
            return repo, (
                "the ledger row's issue reference names "
                f"{f'{owner}/{name}' if owner else name}, not {repo}"
            )
    return repo, None


async def _fetch_issue(
    repo: str, issue_number: int, token: str
) -> Optional[Dict[str, Any]]:
    """One issue, or ``None`` when it cannot be read. Never raises."""
    try:
        issue = await github_api_get(f"/repos/{repo}/issues/{issue_number}", token)
    except Exception as exc:  # noqa: BLE001 - selection must not crash dispatch
        logger.debug("Could not read %s#%s: %s", repo, issue_number, exc)
        return None
    return issue if isinstance(issue, dict) else None


def _ineligibility(
    repo: str, issue_number: int, issue: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Why GitHub's answer for ``repo#issue_number`` keeps it off the allow-list.

    ``None`` only when GitHub served exactly that issue, from that repository,
    open, labelled :data:`AGENT_READY_LABEL` and carrying no Talon state
    label. What GitHub did not state is refused: this decides what a dispatch
    that writes code may touch.
    """
    if not isinstance(issue, dict):
        # Unreadable is not open: a lookup failure must not be read as a live
        # target.
        return _eligibility_exclusion(repo, issue_number, EXCLUDED_ISSUE_UNREADABLE)
    served_repo = _served_repository(issue)
    served_number = issue.get("number")
    if (
        served_repo is None
        or served_repo.lower() != repo.lower()
        or served_number != issue_number
    ):
        # GitHub follows a transferred issue's redirect, so asking this
        # repository for the number can answer with the issue it moved to.
        return _eligibility_exclusion(
            repo,
            issue_number,
            EXCLUDED_WRONG_REPO,
            detail=(
                "GitHub did not say which repository serves it"
                if served_repo is None
                else f"GitHub serves it as {served_repo}#{served_number}"
            ),
        )
    if issue.get("pull_request"):
        # GitHub serves pull requests from the issues endpoint too, and a
        # blocker's number can name one.
        return _eligibility_exclusion(repo, issue_number, EXCLUDED_NOT_AN_ISSUE)
    if issue.get("state") != "open":
        return _eligibility_exclusion(
            repo, issue_number, EXCLUDED_CLOSED, state=issue.get("state")
        )
    owned_by = _talon_state_labels(issue)
    if owned_by:
        # Open, but Talon has already claimed it, is waiting for an answer on
        # it, or failed on it and asked for the label to be cleared.
        # Dispatching again is the same run again; the decision belongs to
        # whoever reads Talon's comment (#3294).
        return _eligibility_exclusion(
            repo, issue_number, EXCLUDED_TALON_OWNED, labels=sorted(owned_by)
        )
    if AGENT_READY_LABEL not in _label_names(issue):
        return _eligibility_exclusion(repo, issue_number, EXCLUDED_NOT_AGENT_READY)
    return None


def _served_repository(issue: Dict[str, Any]) -> Optional[str]:
    """The ``owner/name`` GitHub's ``repository_url`` names, or ``None``."""
    url = issue.get("repository_url")
    if not isinstance(url, str):
        return None
    _, marker, repo = url.rstrip("/").rpartition("/repos/")
    return repo if marker and repo else None


def _eligibility_exclusion(
    repo: Optional[str], issue_number: int, reason: str, **facts: Any
) -> Dict[str, Any]:
    return {"repo": repo, "issue_number": issue_number, "reason": reason, **facts}


def _stalled_pr_days(config: Dict[str, Any]) -> int:
    value = config.get("stalled_pr_days", DEFAULT_STALLED_PR_DAYS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        logger.warning(
            "Ignoring morning_signal_config.stalled_pr_days=%r; using %s",
            value, DEFAULT_STALLED_PR_DAYS,
        )
        return DEFAULT_STALLED_PR_DAYS
    return value


async def _fetch_open_linked_pull_requests(
    repo: str, issue_number: int, token: str
) -> Optional[List[Dict[str, Any]]]:
    """Open PRs GitHub links to ``issue_number``, or ``None`` when it cannot say.

    ``[]`` is an answer (nothing links it); ``None`` is not. Never raises.
    """
    owner, _, name = repo.partition("/")
    try:
        response = await github_api_post(
            "/graphql",
            token,
            {
                "query": _LINKED_PULL_REQUESTS_QUERY,
                "variables": {"owner": owner, "name": name, "number": issue_number},
            },
        )
    except Exception as exc:  # noqa: BLE001 - selection must not crash dispatch
        logger.debug("Could not read PR linkage for %s#%s: %s", repo, issue_number, exc)
        return None
    if not isinstance(response, dict) or response.get("errors"):
        return None
    try:
        nodes = response["data"]["repository"]["issue"][
            "closedByPullRequestsReferences"
        ]["nodes"]
    except (KeyError, TypeError):
        return None
    if not isinstance(nodes, list):
        return None
    return [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("state") == "OPEN"
    ]


async def _fetch_retry_evidence(
    repo: str, issue_number: int, token: str
) -> Optional[Dict[str, Any]]:
    """The issue's recent activity, or ``None`` when GitHub cannot say.

    ``{"last_edited_at": <body's last edit or None>, "timeline": [nodes]}``.
    Never raises.
    """
    owner, _, name = repo.partition("/")
    try:
        response = await github_api_post(
            "/graphql",
            token,
            {
                "query": _RETRY_EVIDENCE_QUERY,
                "variables": {"owner": owner, "name": name, "number": issue_number},
            },
        )
    except Exception as exc:  # noqa: BLE001 - selection must not crash dispatch
        logger.debug(
            "Could not read post-run activity for %s#%s: %s", repo, issue_number, exc
        )
        return None
    if not isinstance(response, dict) or response.get("errors"):
        return None
    try:
        issue = response["data"]["repository"]["issue"]
        nodes = issue["timelineItems"]["nodes"]
    except (KeyError, TypeError):
        return None
    if not isinstance(nodes, list):
        return None
    return {
        "last_edited_at": issue.get("lastEditedAt"),
        "timeline": [node for node in nodes if isinstance(node, dict)],
    }


def _retry_authorization(evidence: Dict[str, Any], since: datetime) -> Optional[str]:
    """What on the issue, strictly newer than ``since``, authorizes a retry.

    ``None`` when nothing does. A body edit or title change is new input to
    the next run; so is a maintainer's or the orchestrator's comment --
    Talon reads the issue's comments, so a run after one is not the same run
    again. Talon's own comments about the run it just finished, a stranger's
    comment, and any label other than :data:`RETRY_READINESS_LABELS` are not.
    """
    edited = parse_instant(evidence.get("last_edited_at"))
    if edited is not None and edited > since:
        return f"an issue edit at {edited.isoformat()}"
    for item in reversed(evidence.get("timeline") or ()):
        created = parse_instant(item.get("createdAt"))
        if created is None or created <= since:
            continue
        kind = item.get("__typename")
        if kind == "IssueComment" and _comment_speaks_for_repository(item):
            return f"a comment at {created.isoformat()}"
        if kind == "LabeledEvent":
            label = item.get("label")
            name = label.get("name") if isinstance(label, dict) else None
            if isinstance(name, str) and name.lower() in RETRY_READINESS_LABELS:
                return f"{name} applied at {created.isoformat()}"
        if kind == "RenamedTitleEvent":
            return f"a title change at {created.isoformat()}"
    return None


def _comment_speaks_for_repository(comment: Dict[str, Any]) -> bool:
    if comment.get("authorAssociation") not in _AUTHORIZING_ASSOCIATIONS:
        return False
    body = comment.get("body")
    return not (isinstance(body, str) and body.lstrip().startswith(TALON_COMMENT_PREFIXES))


def _run_exclusion(
    repo: str, issue_number: int, run: TalonRun, reason: str
) -> Dict[str, Any]:
    return {
        "repo": repo,
        "issue_number": issue_number,
        "reason": reason,
        "job_id": run.job_id,
        "disposition": run.disposition,
        "completed_at": run.completed_at.isoformat(),
    }


def _days_since(timestamp: object, now: datetime) -> Optional[int]:
    moment = parse_instant(timestamp)
    if moment is None:
        return None
    return max(0, (now - moment).days)


def _open_pr_exclusion(
    repo: str,
    issue_number: int,
    pull_requests: Optional[List[Dict[str, Any]]],
    stalled_after_days: int,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Why ``repo#issue_number`` must not be freshly dispatched, or ``None``.

    Unreadable linkage withholds too: a dispatch writes code into a worktree
    derived from the issue, and "could not tell whether a run already owns
    it" is not "free".
    """
    if pull_requests is None:
        return {
            "repo": repo,
            "issue_number": issue_number,
            "reason": EXCLUDED_PR_LINKAGE_UNREADABLE,
            "pull_requests": [],
        }
    if not pull_requests:
        return None
    now = now or datetime.now(timezone.utc)
    prs = []
    for pr in pull_requests:
        pr_repo = (pr.get("repository") or {}).get("nameWithOwner") or repo
        prs.append({
            "repo": pr_repo,
            "number": pr.get("number"),
            "url": pr.get("url"),
            "draft": bool(pr.get("isDraft")),
            "updated_at": pr.get("updatedAt"),
            "days_idle": _days_since(pr.get("updatedAt"), now),
        })
    idle = [pr["days_idle"] for pr in prs]
    # Stalled only when EVERY open PR is known to be idle past the threshold:
    # one PR still moving means the work is in flight, and an unreadable
    # timestamp is not evidence of neglect.
    stalled = all(days is not None and days >= stalled_after_days for days in idle)
    return {
        "repo": repo,
        "issue_number": issue_number,
        "reason": EXCLUDED_STALLED_PR if stalled else EXCLUDED_OPEN_PR,
        "pull_requests": prs,
        "stalled_after_days": stalled_after_days,
    }


def describe_exclusion(exclusion: Dict[str, Any]) -> str:
    """One line an orchestrator can read: ``skipped o/r#3310 -- PR #3311 open``."""
    target = f"{exclusion.get('repo') or ''}#{exclusion['issue_number']}"
    reason = exclusion.get("reason")
    if reason == EXCLUDED_WRONG_REPO:
        return f"skipped {target} -- {exclusion.get('detail')}"
    if reason == EXCLUDED_ISSUE_UNREADABLE:
        return f"skipped {target} -- GitHub could not return the issue"
    if reason == EXCLUDED_NOT_AN_ISSUE:
        return f"skipped {target} -- a pull request, not an issue"
    if reason == EXCLUDED_CLOSED:
        state = exclusion.get("state")
        if state == "closed":
            return f"skipped {target} -- closed"
        return f"skipped {target} -- GitHub does not report it open (state {state!r})"
    if reason == EXCLUDED_TALON_OWNED:
        return (
            f"skipped {target} -- Talon labels "
            f"{', '.join(exclusion.get('labels') or ())}; the decision belongs to "
            "whoever reads Talon's comment"
        )
    if reason == EXCLUDED_NOT_AGENT_READY:
        return f"skipped {target} -- not labelled {AGENT_READY_LABEL}"
    if reason in (EXCLUDED_RUN_HISTORY, EXCLUDED_RUN_HISTORY_UNCONFIRMED):
        completed = parse_instant(exclusion.get("completed_at"))
        when = (
            completed.strftime("%Y-%m-%d %H:%M UTC")
            if completed is not None
            else exclusion.get("completed_at")
        )
        run = (
            f"Talon job {str(exclusion.get('job_id'))[:8]} ended "
            f"{exclusion.get('disposition')} at {when} without a PR"
        )
        if reason == EXCLUDED_RUN_HISTORY_UNCONFIRMED:
            return (
                f"skipped {target} -- {run}, and GitHub could not say whether "
                "anything since authorizes a retry"
            )
        return (
            f"skipped {target} -- {run}, and nothing since authorizes a retry "
            "(a maintainer or orchestrator comment, an issue edit, or agent-ready)"
        )
    if reason == EXCLUDED_PR_LINKAGE_UNREADABLE:
        return (
            f"skipped {target} -- GitHub could not say whether an open PR "
            "already works it"
        )
    refs = []
    for pr in exclusion.get("pull_requests") or ():
        ref = f"PR #{pr['number']}"
        if pr.get("repo") and pr["repo"] != exclusion["repo"]:
            ref = f"PR {pr['repo']}#{pr['number']}"
        if pr.get("draft"):
            ref += " (draft)"
        refs.append(ref)
    prs = ", ".join(refs)
    if reason == EXCLUDED_STALLED_PR:
        idle = min(pr["days_idle"] for pr in exclusion["pull_requests"])
        return (
            f"skipped {target} -- {prs} open but untouched for {idle} day(s); "
            "a rescue, not a fresh claim"
        )
    return f"skipped {target} -- {prs} open"


def _label_names(issue: Dict[str, Any]) -> set:
    """The lower-cased label names on ``issue``."""
    names = set()
    for label in issue.get("labels") or ():
        name = label.get("name") if isinstance(label, dict) else label
        if isinstance(name, str):
            names.add(name.lower())
    return names


def _talon_state_labels(issue: Dict[str, Any]) -> set:
    """The Talon state labels on ``issue`` -- non-empty means not dispatchable."""
    return _label_names(issue) & TALON_STATE_LABELS


async def _fetch_milestone_issues(
    repo: str, milestone_name: str, token: str
) -> List[Dict[str, Any]]:
    """The milestone's open issues labelled :data:`AGENT_READY_LABEL`.

    GitHub filters by the label so the page holds candidates rather than the
    issues nobody authorized; selection still checks every one it returns.
    """
    try:
        milestones = await github_api_get(
            f"/repos/{repo}/milestones?state=open&per_page=20", token
        )
        if not isinstance(milestones, list):
            return []
        milestone_number = next(
            (
                milestone["number"]
                for milestone in milestones
                if milestone_name.lower() in milestone.get("title", "").lower()
            ),
            None,
        )
        if milestone_number is None:
            return []
        issues = await github_api_get(
            f"/repos/{repo}/issues?milestone={milestone_number}"
            f"&state=open&labels={AGENT_READY_LABEL}&per_page=10&sort=updated",
            token,
        )
        return [issue for issue in (issues or []) if not issue.get("pull_request")]
    except Exception as exc:
        logger.debug("Failed to fetch milestone issues for %s/%s: %s", repo, milestone_name, exc)
        return []


async def _fetch_open_issues(
    repo: str, token: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """The repository's ``limit`` most recently updated agent-ready issues.

    Filtered by GitHub for the reason :func:`_fetch_milestone_issues` is: the
    five most recently updated issues of any kind are rarely ones the
    orchestrator has authorized.
    """
    try:
        issues = await github_api_get(
            f"/repos/{repo}/issues?state=open&labels={AGENT_READY_LABEL}"
            f"&per_page={limit}&sort=updated",
            token,
        )
        return [issue for issue in (issues or []) if not issue.get("pull_request")]
    except Exception:
        return []


def _ranked_candidates(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer unassigned, low-comment issues and skip blocked outcomes.

    Ranks eligible issues only: the allow-list (:func:`_ineligibility`), which
    also refuses Talon-owned issues, runs first.
    """
    skip_labels = {"blocked", "wontfix", "won't fix", "duplicate", "invalid"}
    candidates = [issue for issue in issues if not _label_names(issue) & skip_labels]

    def sort_key(issue: Dict[str, Any]) -> Tuple[int, int]:
        return (1 if issue.get("assignees") else 0, issue.get("comments", 0))

    candidates.sort(key=sort_key)
    return candidates
