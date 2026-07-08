"""GitHub API integration helpers for Strategic Memory.

Provides low-level GitHub API access (GET/POST) and signal fetching
for Morning Signal, Backlog Hygiene, and Session Log features.
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Prerequisite states for live GitHub signal fetching. These let callers
# surface the correct remediation instead of always blaming the token.
GITHUB_SIGNAL_READY = "ready"
GITHUB_SIGNAL_NO_SCAN_REPOS = "no_scan_repos"
GITHUB_SIGNAL_NO_TOKEN = "no_token"


class GitHubAuthError(Exception):
    """GitHub rejected the supplied credentials (HTTP 401/403).

    Raised from ``github_api_get`` only when ``raise_on_auth=True`` so that
    live-signal callers can tell "token present but invalid" apart from a
    transient network failure and surface the token remediation (#2271).
    """


def github_signal_prerequisite(data: Dict[str, Any]) -> str:
    """Classify whether live GitHub scanning can run, and if not, why.

    Returns one of ``GITHUB_SIGNAL_READY``, ``GITHUB_SIGNAL_NO_SCAN_REPOS``,
    or ``GITHUB_SIGNAL_NO_TOKEN``. ``scan_repos`` is checked before the token
    so an empty repo list is not misreported as a missing token (#2271).
    """
    config = data.get("morning_signal_config", {})
    repos = config.get("scan_repos", [])
    if not repos:
        return GITHUB_SIGNAL_NO_SCAN_REPOS
    if not get_github_token():
        return GITHUB_SIGNAL_NO_TOKEN
    return GITHUB_SIGNAL_READY


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or .env file."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    # Try .env in project root
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


async def github_api_get(path: str, token: str, *, raise_on_auth: bool = False) -> Any:
    """Make a GitHub API GET request. Returns parsed JSON or None on error.

    When ``raise_on_auth`` is set, a 401/403 response raises
    ``GitHubAuthError`` instead of being flattened to ``None`` -- this lets
    live-signal callers distinguish an invalid token from a transient failure.
    Callers that leave it ``False`` keep the historical "None on any error"
    contract.
    """
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "kestrel-agent",
    })
    try:
        resp = await asyncio.to_thread(
            lambda: urllib.request.urlopen(req, timeout=10).read()
        )
        return json.loads(resp)
    except urllib.error.HTTPError as e:
        if raise_on_auth and e.code in (401, 403):
            logger.warning(f"GitHub API auth error ({e.code}) for {path}")
            raise GitHubAuthError(str(e.code)) from e
        logger.warning(f"GitHub API error for {path}: {e}")
        return None
    except (urllib.error.URLError, Exception) as e:
        logger.warning(f"GitHub API error for {path}: {e}")
        return None


async def github_api_post(path: str, token: str, body: dict) -> Any:
    """Make a GitHub API POST request. Returns parsed JSON or None on error."""
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
        "User-Agent": "kestrel-agent",
    })
    try:
        resp = await asyncio.to_thread(
            lambda: urllib.request.urlopen(req, timeout=10).read()
        )
        return json.loads(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        logger.warning(f"GitHub API POST error for {path}: {e}")
        return None


def short_repo(repo: str, all_repos: list) -> str:
    """Shorten repo name, using full owner/repo when names collide."""
    name = repo.split("/")[-1]
    if sum(1 for r in all_repos if r.split("/")[-1] == name) > 1:
        return repo
    return name


async def fetch_github_signal(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch live GitHub data for Morning Signal repos.

    Args:
        data: The strategic memory data dict (needs morning_signal_config).

    Returns a dict with per-repo issue counts, open PRs, and recent comments.

    Raises ``GitHubAuthError`` when a token is configured but GitHub rejects
    it (401/403), so callers can surface the token remediation rather than
    treating placeholder repo shells as live data (#2271).
    """
    prereq = github_signal_prerequisite(data)
    if prereq == GITHUB_SIGNAL_NO_SCAN_REPOS:
        return {}
    if prereq == GITHUB_SIGNAL_NO_TOKEN:
        logger.info("No GITHUB_TOKEN found -- Morning Signal will use YAML data only")
        return {}

    config = data.get("morning_signal_config", {})
    repos = config.get("scan_repos", [])
    token = get_github_token()

    result: Dict[str, Any] = {}
    since = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    for repo in repos:
        repo_data: Dict[str, Any] = {"issues": [], "prs": [], "comments_24h": 0}

        # Fetch open issues (not PRs, up to 100)
        issues = await github_api_get(
            f"/repos/{repo}/issues?state=open&per_page=100", token,
            raise_on_auth=True,
        )
        if isinstance(issues, list):
            # Filter out PRs (they appear in /issues too)
            real_issues = [i for i in issues if "pull_request" not in i]
            repo_data["issue_count"] = len(real_issues)

            # Count by milestone
            by_milestone: Dict[str, int] = {}
            for i in real_issues:
                ms_name = i.get("milestone", {}).get("title", "No Milestone") if i.get("milestone") else "No Milestone"
                by_milestone[ms_name] = by_milestone.get(ms_name, 0) + 1
            repo_data["by_milestone"] = by_milestone

            # Collect issues with labels containing "blocked" or severity
            blocked = [
                {"number": i["number"], "title": i["title"]}
                for i in real_issues
                if any("block" in l["name"].lower() for l in i.get("labels", []))
            ]
            repo_data["blocked_issues"] = blocked
        else:
            repo_data["issue_count"] = "?"

        # Fetch open PRs
        prs = await github_api_get(
            f"/repos/{repo}/pulls?state=open&per_page=20", token,
            raise_on_auth=True,
        )
        if isinstance(prs, list):
            repo_data["prs"] = [
                {
                    "number": p["number"],
                    "title": p["title"],
                    "author": p["user"]["login"],
                    "updated": p["updated_at"][:10],
                }
                for p in prs
            ]

        # Fetch recent comments (since midnight UTC)
        comments = await github_api_get(
            f"/repos/{repo}/issues/comments?since={since}&sort=updated&direction=desc&per_page=5",
            token,
            raise_on_auth=True,
        )
        if isinstance(comments, list):
            repo_data["comments_24h"] = len(comments)
            repo_data["recent_comments"] = [
                {
                    "issue_url": c.get("issue_url", "").split("/")[-1],
                    "author": c["user"]["login"],
                    "snippet": (c.get("body") or "")[:80],
                }
                for c in comments[:3]
            ]

        result[repo] = repo_data

    return result
