"""Select the highest-priority actionable GitHub issue.

Issue selection is strategic-memory behavior. Dispatching the selected issue
belongs to an independently installed coding feature or another orchestrator.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.features.strategic_memory.github_integration import (
    get_github_token,
    github_api_get,
    github_api_post,
)

logger = logging.getLogger(__name__)

#: ``owner/name`` in GitHub's allowed character set. A reference prefix that
#: does not match this is prose, not a repository.
_REPO_SHAPE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")

#: Labels Talon puts on an issue while a run owns it or is waiting on a
#: human: ``kestreltalon/config.py`` ``label_analyzing`` / ``label_clarifying``
#: / ``label_in_progress`` / ``label_blocked`` / ``label_failed``. Talon's own
#: failure comment says "remove the label to allow a retry": the label is the
#: retry protocol, and re-dispatching over it re-runs the same claim every
#: morning (#3294 -- #3051 four days running). ``agent-ready`` and
#: ``agent-complete`` are deliberately absent: the first is an instruction to
#: Talon, the second is a finished run. A finished run whose pull request is
#: still open is withheld by the PR itself (#3317), not by the label: the
#: label outlives a PR that was closed unmerged. Core cannot import kestreltalon
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
    data: Dict[str, Any], diagnostics: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Return the highest-priority issue represented by strategic memory.

    ``None`` has two meanings a caller must be able to tell apart: nothing is
    actionable, or GitHub could not confirm anything. Pass ``diagnostics`` to
    have ``blockers_checked`` and ``blockers_unreadable`` filled in -- when
    every blocker checked was unreadable, "no actionable issue" would be a
    claim about a ledger nobody actually looked at.

    ``diagnostics["open_pr_exclusions"]`` lists every candidate passed over
    because an open pull request already works it (or GitHub could not say),
    so the dispatch can report what it skipped instead of staying silent.
    """
    if diagnostics is None:
        diagnostics = {}
    diagnostics.setdefault("blockers_checked", 0)
    diagnostics.setdefault("blockers_unreadable", 0)
    diagnostics.setdefault("blockers_talon_owned", 0)
    exclusions = diagnostics.setdefault("open_pr_exclusions", [])
    token = get_github_token()
    if not token:
        logger.info("No GITHUB_TOKEN — cannot pick top issue")
        return None

    config = data.get("morning_signal_config", {})
    repos = config.get("scan_repos", [])
    stalled_after_days = _stalled_pr_days(config)
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

    async def first_free(repo: str, issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for candidate in _ranked_candidates(issues):
            if await in_flight(repo, candidate["number"]) is None:
                return candidate
        return None

    # One GitHub read per DISTINCT (repo, number), not per row: the ledger only
    # grows, and on the live host 12 of the 33 qualifying rows named the same
    # pull request -- a guaranteed miss, read twelve times every run. That is
    # the bound on this walk: one read per distinct target named by a
    # high/critical blocker.
    checked: set = set()
    for blocker in data.get("blockers", []):
        if blocker.get("severity") not in ("critical", "high") or not blocker.get("issue"):
            continue
        ref_repo, issue_number = parse_issue_ref(blocker["issue"])
        if issue_number is None:
            # An unparseable reference is not dispatchable, and guessing a
            # number from it would dispatch against the wrong issue.
            continue
        # A blocker that names its own repository does not need the scan list.
        # Returning early on an empty scan_repos skipped exactly those, which
        # are the ones whose target is least ambiguous.
        repo = blocker.get("repo") or ref_repo
        if not repo:
            # A bare number names no project when several are configured.
            # This used to walk scan_repos and take the FIRST repository that
            # had any issue with that number -- with fourteen repos, low
            # numbers collide everywhere. The blocker reconciler and
            # ``_resolve_blocker_repo`` both refuse this guess ("the guess is
            # what made reconciliation resolve a blocker against the wrong
            # project's issue 42"); dispatch, which is irreversible, guessed.
            # A lone configured repository is not a guess.
            if len(repos) != 1:
                continue
            repo = repos[0]

        # Selection is where "this blocker is still live" gets decided, so it
        # is decided against GitHub, not the ledger. The ledger only grows
        # unless someone reconciles it, and nothing schedules that: on the
        # live host 161 blocker rows were all unresolved, several naming
        # issues closed for weeks, and the dispatch path picked one of them --
        # a ticket closed on 2026-07-28, under a title that was not the
        # issue's own -- every morning. Talon would have written code for it.
        if (repo, issue_number) in checked:
            continue
        checked.add((repo, issue_number))
        issue = await _fetch_issue(repo, issue_number, token)
        diagnostics["blockers_checked"] += 1
        if issue is None:
            diagnostics["blockers_unreadable"] += 1
        if not _is_open_issue(issue):
            continue
        owned_by = _talon_state_labels(issue)
        if owned_by:
            # Open, but Talon has already claimed it, is waiting for an answer
            # on it, or failed on it and asked for the label to be cleared.
            # Dispatching again is the same run again; the decision belongs
            # to whoever reads Talon's comment (#3294).
            diagnostics["blockers_talon_owned"] += 1
            logger.info(
                "Not dispatching %s#%s: Talon labels %s",
                repo, issue_number, sorted(owned_by),
            )
            continue
        exclusion = await in_flight(repo, issue_number)
        if exclusion is not None:
            if exclusion["reason"] == EXCLUDED_PR_LINKAGE_UNREADABLE:
                # Not confirmed free of in-flight work is not confirmed at
                # all: it counts with the unreadable issues, so a GitHub
                # outage still renders as one rather than as "nothing to do".
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
        for repo in milestone.get("repos", []):
            if repo not in repos:
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

    for repo in repos:
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


def _is_open_issue(issue: Optional[Dict[str, Any]]) -> bool:
    """Dispatchable only when GitHub says it is an open issue.

    Unreadable is not open: a lookup failure must not be read as a live
    target for work that writes code. And GitHub serves pull requests from
    the issues endpoint too, which a blocker's number can name.
    """
    return (
        isinstance(issue, dict)
        and issue.get("state") == "open"
        and not issue.get("pull_request")
    )


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


def _days_since(timestamp: object, now: datetime) -> Optional[int]:
    if not isinstance(timestamp, str) or not timestamp.strip():
        return None
    try:
        moment = datetime.fromisoformat(timestamp.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
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
    target = f"{exclusion['repo']}#{exclusion['issue_number']}"
    reason = exclusion.get("reason")
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


def _talon_state_labels(issue: Dict[str, Any]) -> set:
    """The Talon state labels on ``issue`` -- non-empty means not dispatchable."""
    names = set()
    for label in issue.get("labels") or ():
        name = label.get("name") if isinstance(label, dict) else label
        if isinstance(name, str):
            names.add(name.lower())
    return names & TALON_STATE_LABELS


async def _fetch_milestone_issues(
    repo: str, milestone_name: str, token: str
) -> List[Dict[str, Any]]:
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
            "&state=open&per_page=10&sort=updated",
            token,
        )
        return [issue for issue in (issues or []) if not issue.get("pull_request")]
    except Exception as exc:
        logger.debug("Failed to fetch milestone issues for %s/%s: %s", repo, milestone_name, exc)
        return []


async def _fetch_open_issues(
    repo: str, token: str, limit: int = 5
) -> List[Dict[str, Any]]:
    try:
        issues = await github_api_get(
            f"/repos/{repo}/issues?state=open&per_page={limit}&sort=updated", token
        )
        return [issue for issue in (issues or []) if not issue.get("pull_request")]
    except Exception:
        return []


def _ranked_candidates(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer unassigned, low-comment issues and skip blocked outcomes."""
    skip_labels = {"blocked", "wontfix", "won't fix", "duplicate", "invalid"}
    candidates = [
        issue
        for issue in issues
        if not ({label["name"].lower() for label in issue.get("labels", [])} & skip_labels)
        and not _talon_state_labels(issue)
    ]

    def sort_key(issue: Dict[str, Any]) -> Tuple[int, int]:
        return (1 if issue.get("assignees") else 0, issue.get("comments", 0))

    candidates.sort(key=sort_key)
    return candidates
