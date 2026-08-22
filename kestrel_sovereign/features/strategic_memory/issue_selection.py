"""Select the highest-priority actionable GitHub issue.

Issue selection is strategic-memory behavior. Dispatching the selected issue
belongs to an independently installed coding feature or another orchestrator.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.features.strategic_memory.github_integration import (
    get_github_token,
    github_api_get,
)

logger = logging.getLogger(__name__)


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
        if head:
            repo = head
        text = tail.strip()
    text = text.lstrip("#").strip()
    try:
        return repo, int(text)
    except (TypeError, ValueError):
        return repo, None


async def pick_top_issue(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the highest-priority issue represented by strategic memory."""
    token = get_github_token()
    if not token:
        logger.info("No GITHUB_TOKEN — cannot pick top issue")
        return None

    config = data.get("morning_signal_config", {})
    repos = config.get("scan_repos", [])

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
            if not repos:
                continue
            repo = await _find_issue_repo(str(issue_number), repos, token)
        if repo:
            return {
                "repo": repo,
                "issue_number": issue_number,
                "issue_title": blocker.get("title", "Blocker"),
                "priority": "high",
                "context": (
                    f"Blocker (severity: {blocker.get('severity')}): "
                    f"{blocker.get('notes', '')}"
                ),
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
            pick = _select_best_candidate(issues)
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
        pick = _select_best_candidate(await _fetch_open_issues(repo, token, limit=5))
        if pick:
            return {
                "repo": repo,
                "issue_number": pick["number"],
                "issue_title": pick["title"],
                "priority": "normal",
                "context": "Open issue from backlog scan",
            }
    return None


async def _find_issue_repo(issue_number: str, repos: List[str], token: str) -> Optional[str]:
    for repo in repos:
        try:
            issue = await github_api_get(f"/repos/{repo}/issues/{issue_number}", token)
            if issue and not issue.get("pull_request"):
                return repo
        except Exception:
            continue
    return None


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


def _select_best_candidate(issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prefer unassigned, low-comment issues and skip blocked outcomes."""
    skip_labels = {"blocked", "wontfix", "won't fix", "duplicate", "invalid"}
    candidates = [
        issue
        for issue in issues
        if not ({label["name"].lower() for label in issue.get("labels", [])} & skip_labels)
    ]
    if not candidates:
        return None

    def sort_key(issue: Dict[str, Any]) -> Tuple[int, int]:
        return (1 if issue.get("assignees") else 0, issue.get("comments", 0))

    candidates.sort(key=sort_key)
    return candidates[0]
