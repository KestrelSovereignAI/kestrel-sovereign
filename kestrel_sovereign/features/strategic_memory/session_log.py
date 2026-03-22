"""Session Log collector for Strategic Memory.

Scans GitHub repos for today's activity (issues closed, PRs merged,
comments, commits) and generates a structured session summary with
outcomes, contributor scoreboard, and activity highlights.
"""

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from .github_integration import get_github_token, github_api_get, short_repo

logger = logging.getLogger(__name__)


async def collect_session_log(
    data: Dict[str, Any],
    session_id: str = "",
    focus: str = "",
) -> str:
    """Collect end-of-day session log from GitHub activity.

    Args:
        data: The strategic memory data dict (needs morning_signal_config).
        session_id: Session number (e.g. '020'). Auto-generated if empty.
        focus: Brief description of today's focus area.

    Returns:
        Formatted markdown session log.
    """
    config = data.get("morning_signal_config", {})
    repos = config.get("scan_repos", [])
    if not repos:
        return "No scan_repos configured in morning_signal_config."

    token = get_github_token()
    if not token:
        return "No GITHUB_TOKEN found. Set GITHUB_TOKEN environment variable or add to .env file."

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    since = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Collect data across all repos
    issues_closed: List[Dict] = []
    prs_merged: List[Dict] = []
    prs_opened: List[Dict] = []
    comments_posted: List[Dict] = []
    all_repos = repos[:]
    contributors: Dict[str, Dict] = {}

    for repo in repos:
        short = short_repo(repo, all_repos)

        # Closed issues today
        closed = await github_api_get(
            f"/repos/{repo}/issues?state=closed&since={since}&per_page=50&sort=updated", token,
        )
        if isinstance(closed, list):
            for i in closed:
                if "pull_request" not in i and i.get("closed_at", "")[:10] == today_str:
                    closer = (i.get("closed_by") or i.get("user") or {}).get("login", "unknown")
                    issues_closed.append({
                        "repo": short, "number": i["number"],
                        "title": i["title"], "closer": closer,
                    })
                    contributors.setdefault(closer, {"closed": 0, "prs": 0, "comments": 0})
                    contributors[closer]["closed"] += 1

        # Merged PRs today
        merged = await github_api_get(
            f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc&per_page=20", token,
        )
        if isinstance(merged, list):
            for p in merged:
                if p.get("merged_at") and p["merged_at"][:10] == today_str:
                    author = (p.get("user") or {}).get("login", "unknown")
                    prs_merged.append({
                        "repo": short, "number": p["number"],
                        "title": p["title"], "author": author,
                    })
                    contributors.setdefault(author, {"closed": 0, "prs": 0, "comments": 0})
                    contributors[author]["prs"] += 1

        # PRs opened today
        opened = await github_api_get(
            f"/repos/{repo}/pulls?state=open&sort=created&direction=desc&per_page=10", token,
        )
        if isinstance(opened, list):
            for p in opened:
                if p.get("created_at", "")[:10] == today_str:
                    author = (p.get("user") or {}).get("login", "unknown")
                    prs_opened.append({
                        "repo": short, "number": p["number"],
                        "title": p["title"], "author": author,
                    })

        # Comments today
        comments = await github_api_get(
            f"/repos/{repo}/issues/comments?since={since}&sort=updated&direction=desc&per_page=30",
            token,
        )
        if isinstance(comments, list):
            for c in comments:
                author = (c.get("user") or {}).get("login", "unknown")
                issue_num = (c.get("issue_url") or "").split("/")[-1]
                comments_posted.append({
                    "repo": short, "issue": issue_num, "author": author,
                    "snippet": (c.get("body") or "")[:60],
                })
                contributors.setdefault(author, {"closed": 0, "prs": 0, "comments": 0})
                contributors[author]["comments"] += 1

    # Build session report
    sid = session_id or today.strftime("%Y%m%d")
    lines = [
        f"# Session Log -- {today.strftime('%B %d, %Y')} (#{sid})",
        "",
    ]
    if focus:
        lines.append(f"**Focus:** {focus}")
        lines.append("")

    # Outcomes summary
    lines.append("## Outcomes")
    lines.append(f"- **{len(issues_closed)}** issues closed")
    lines.append(f"- **{len(prs_merged)}** PRs merged")
    lines.append(f"- **{len(prs_opened)}** PRs opened")
    lines.append(f"- **{len(comments_posted)}** comments posted")
    lines.append("")

    # Issues closed
    if issues_closed:
        lines.append("## Issues Closed")
        for i in issues_closed:
            lines.append(f"- {i['repo']}#{i['number']}: {i['title']} (@{i['closer']})")
        lines.append("")

    # PRs merged
    if prs_merged:
        lines.append("## PRs Merged")
        for p in prs_merged:
            lines.append(f"- {p['repo']}#{p['number']}: {p['title']} (@{p['author']})")
        lines.append("")

    # PRs opened
    if prs_opened:
        lines.append("## PRs Opened")
        for p in prs_opened:
            lines.append(f"- {p['repo']}#{p['number']}: {p['title']} (@{p['author']})")
        lines.append("")

    # Contributor scoreboard
    if contributors:
        lines.append("## Contributor Scoreboard")
        lines.append("| Contributor | Issues Closed | PRs Merged | Comments | Score |")
        lines.append("|------------|--------------|------------|----------|-------|")
        sorted_c = sorted(
            contributors.items(),
            key=lambda x: x[1]["closed"] + x[1]["prs"] * 2,
            reverse=True,
        )
        for name, counts in sorted_c:
            score = counts["closed"] + counts["prs"] * 2
            lines.append(
                f"| @{name} | {counts['closed']} | {counts['prs']} | {counts['comments']} | {score} |"
            )
        lines.append("")

    # Activity feed (last 5 comments)
    if comments_posted:
        lines.append("## Activity Highlights")
        for c in comments_posted[:5]:
            lines.append(f"- {c['repo']}#{c['issue']} (@{c['author']}): {c['snippet']}...")
        lines.append("")

    if not issues_closed and not prs_merged and not comments_posted:
        lines.append("*No GitHub activity detected today. Activity may be happening locally.*")
        lines.append("")

    lines.append(f"*Auto-collected at {datetime.now().strftime('%H:%M')} from {len(repos)} repos.*")

    return "\n".join(lines)
