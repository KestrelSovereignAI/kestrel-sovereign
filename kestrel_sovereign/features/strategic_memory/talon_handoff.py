"""Signal -> Talon Dispatch.

After any signal runs (morning, hygiene, event-driven, on-demand),
this module picks the highest-priority actionable issue and sends
a mesh 'assign' message to Talon for autonomous implementation.

Selection criteria (in priority order):
1. Critical/high blockers with assigned owner matching this agent
2. At-risk milestone items on the critical path
3. Open issues from in-progress milestones

Reference: sovereign #302, #307
"""

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.features.peers.mesh import make_assign_message
from kestrel_sovereign.features.strategic_memory.github_integration import (
    fetch_github_signal,
    get_github_token,
    github_api_get,
)

logger = logging.getLogger(__name__)


async def pick_top_issue(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Analyze strategic memory + live GitHub data to pick the single
    highest-priority issue to assign to Talon.

    Returns:
        Dict with keys: repo, issue_number, issue_title, priority, context
        or None if no suitable issue was found.
    """
    token = get_github_token()
    if not token:
        logger.info("No GITHUB_TOKEN — cannot pick top issue")
        return None

    config = data.get("morning_signal_config", {})
    repos = config.get("scan_repos", [])
    if not repos:
        return None

    # 1. Check blockers first — highest priority
    blockers = data.get("blockers", [])
    for b in blockers:
        if b.get("severity") in ("critical", "high") and b.get("issue"):
            issue_ref = str(b["issue"]).lstrip("#")
            # Find which repo this blocker belongs to
            repo = b.get("repo")
            if not repo:
                # Try to find the issue across repos
                repo = await _find_issue_repo(issue_ref, repos, token)
            if repo:
                return {
                    "repo": repo,
                    "issue_number": int(issue_ref),
                    "issue_title": b.get("title", "Blocker"),
                    "priority": "high",
                    "context": f"Blocker (severity: {b.get('severity')}): {b.get('notes', '')}",
                }

    # 2. Fetch live signal for at-risk milestones
    github_data = await fetch_github_signal(data)

    # 3. Find open issues from at-risk or in-progress milestones
    at_risk_milestones = [
        m for m in data.get("milestones", [])
        if m.get("status") in ("at_risk", "in_progress")
    ]

    for milestone in at_risk_milestones:
        for repo in milestone.get("repos", []):
            if repo not in repos:
                continue
            rd = github_data.get(repo, {})
            # Look for open issues in this milestone
            milestone_name = milestone.get("name", "")
            issues = await _fetch_milestone_issues(repo, milestone_name, token)
            if issues:
                # Pick the first unassigned or least-worked issue
                pick = _select_best_candidate(issues)
                if pick:
                    return {
                        "repo": repo,
                        "issue_number": pick["number"],
                        "issue_title": pick["title"],
                        "priority": "high" if milestone.get("status") == "at_risk" else "normal",
                        "context": f"Milestone: {milestone_name} ({milestone.get('status')}). {milestone.get('critical_path', '')}",
                    }

    # 4. Fall back to any open issue from scan repos
    for repo in repos:
        issues = await _fetch_open_issues(repo, token, limit=5)
        pick = _select_best_candidate(issues)
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
    """Try to find which repo an issue number belongs to."""
    for repo in repos:
        try:
            data = await github_api_get(f"/repos/{repo}/issues/{issue_number}", token)
            if data and not data.get("pull_request"):
                return repo
        except Exception:
            continue
    return None


async def _fetch_milestone_issues(
    repo: str, milestone_name: str, token: str
) -> List[Dict[str, Any]]:
    """Fetch open issues for a named milestone."""
    try:
        milestones = await github_api_get(
            f"/repos/{repo}/milestones?state=open&per_page=20", token
        )
        if not isinstance(milestones, list):
            return []

        milestone_number = None
        for ms in milestones:
            if milestone_name.lower() in ms.get("title", "").lower():
                milestone_number = ms["number"]
                break

        if milestone_number is None:
            return []

        issues = await github_api_get(
            f"/repos/{repo}/issues?milestone={milestone_number}&state=open&per_page=10&sort=updated",
            token,
        )
        return [i for i in (issues or []) if not i.get("pull_request")]

    except Exception as e:
        logger.debug(f"Failed to fetch milestone issues for {repo}/{milestone_name}: {e}")
        return []


async def _fetch_open_issues(
    repo: str, token: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """Fetch recent open issues from a repo."""
    try:
        issues = await github_api_get(
            f"/repos/{repo}/issues?state=open&per_page={limit}&sort=updated",
            token,
        )
        return [i for i in (issues or []) if not i.get("pull_request")]
    except Exception:
        return []


def _select_best_candidate(issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Pick the best issue to assign from a list.

    Prefers: unassigned > fewer comments > recently updated.
    Skips issues with 'blocked' or 'wontfix' labels.
    """
    skip_labels = {"blocked", "wontfix", "won't fix", "duplicate", "invalid"}

    candidates = []
    for issue in issues:
        labels = {l["name"].lower() for l in issue.get("labels", [])}
        if labels & skip_labels:
            continue
        candidates.append(issue)

    if not candidates:
        return None

    # Sort: unassigned first, then fewest comments, then most recently updated
    def sort_key(i: Dict) -> Tuple:
        assigned = 1 if i.get("assignees") else 0
        comments = i.get("comments", 0)
        return (assigned, comments)

    candidates.sort(key=sort_key)
    return candidates[0]


def _discover_host_url() -> Optional[str]:
    """Discover the rookery host URL (same logic as PeersFeature)."""
    host_url = os.environ.get("KESTREL_HOST_URL")
    if host_url:
        return host_url.rstrip("/")

    for candidate in [
        Path.cwd() / "rookery.toml",
        Path(__file__).resolve().parents[3] / "rookery.toml",
    ]:
        if candidate.exists():
            try:
                import toml
                data = toml.load(candidate)
                port = data.get("host", {}).get("port", 8888)
                return f"http://localhost:{port}"
            except Exception as e:
                logger.debug(f"Could not read {candidate}: {e}")
    return None


async def dispatch_to_talon(
    data: Dict[str, Any],
    sender: str = "kestrel",
    recipient: str = "talon",
) -> str:
    """
    Pick the top issue and dispatch it to Talon via mesh protocol.

    Returns a status message describing what was dispatched (or why not).
    """
    issue = await pick_top_issue(data)
    if not issue:
        return "No actionable issue found to dispatch."

    host_url = _discover_host_url()
    if not host_url:
        return (
            f"Found issue to dispatch ({issue['repo']}#{issue['issue_number']}: "
            f"{issue['issue_title']}) but no rookery host URL configured."
        )

    # Build the mesh assign message
    msg = make_assign_message(
        sender=sender,
        recipient=recipient,
        repo=issue["repo"],
        issue_number=issue["issue_number"],
        issue_title=issue["issue_title"],
        priority=issue["priority"],
        context=issue.get("context", ""),
    )

    # POST to recipient's mesh endpoint via rookery
    url = f"{host_url}/api/agents/{recipient}/agent/mesh"
    payload = json.dumps(msg.to_dict()).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "kestrel-agent"},
    )
    try:
        resp = await asyncio.to_thread(
            lambda: urllib.request.urlopen(req, timeout=10).read()
        )
        result = json.loads(resp)
        return (
            f"Dispatched to {recipient}: {issue['repo']}#{issue['issue_number']} "
            f"({issue['issue_title']}) — priority: {issue['priority']}. "
            f"Mesh ID: {msg.id}"
        )
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return (
            f"Failed to dispatch {issue['repo']}#{issue['issue_number']} "
            f"to {recipient}: {e}"
        )
