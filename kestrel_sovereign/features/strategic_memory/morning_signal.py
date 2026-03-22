"""Morning Signal briefing generator for Strategic Memory.

Builds the Morning Signal report from STRATEGY.yaml data
enriched with live GitHub data.
"""

import os
from datetime import date, datetime
from typing import Any, Dict, List

from .github_integration import fetch_github_signal, short_repo


async def generate_morning_signal(data: Dict[str, Any]) -> str:
    """Generate the Morning Signal briefing from strategic memory + live GitHub data.

    Args:
        data: The strategic memory data dict.

    Returns:
        Formatted markdown briefing string.
    """
    if not data:
        return "No strategic memory loaded. Create a STRATEGY.yaml first."

    today = date.today()
    lines = [f"# Morning Signal -- {today.strftime('%B %d, %Y')}", ""]

    # Fetch live GitHub data (non-blocking, graceful on failure)
    github_data = await fetch_github_signal(data)
    has_live = bool(github_data)

    if has_live:
        lines.append("*Live data from GitHub*")
        lines.append("")

    # Repo overview (live data)
    if has_live:
        lines.append("## Repo Overview")
        lines.append("| Repo | Open Issues | Open PRs | Activity (24h) |")
        lines.append("|------|-------------|----------|----------------|")
        total_issues = 0
        total_prs = 0
        total_comments = 0
        for repo, rd in github_data.items():
            ic = rd.get("issue_count", "?")
            pc = len(rd.get("prs", []))
            cc = rd.get("comments_24h", 0)
            short = short_repo(repo, list(github_data.keys()))
            lines.append(f"| {short} | {ic} | {pc} | {cc} comments |")
            if isinstance(ic, int):
                total_issues += ic
            total_prs += pc
            total_comments += cc
        lines.append(f"| **Total** | **{total_issues}** | **{total_prs}** | **{total_comments}** |")
        lines.append("")

    # Milestones (YAML + enriched with live counts)
    lines.append("## Milestones")
    for m in data.get("milestones", []):
        status_icon = {
            "on_track": "ON TRACK",
            "in_progress": "IN PROGRESS",
            "at_risk": "AT RISK",
            "complete": "COMPLETE",
            "data_collection": "COLLECTING DATA",
            "active": "ACTIVE",
        }.get(m.get("status", ""), m.get("status", "unknown"))

        due = m.get("due", "no date")
        if due and str(due) not in ("no date", "ongoing", "null", "None"):
            try:
                due_date = datetime.strptime(str(due), "%Y-%m-%d").date()
                days_left = (due_date - today).days
                due_str = f"due {due} ({days_left} days)"
            except (ValueError, TypeError):
                due_str = f"due: {due}"
        else:
            due_str = str(due) if due else "no date"

        owner = m.get("owner", "unassigned")

        # Enrich with live issue count from GitHub
        live_count = None
        milestone_name = m.get("name", "")
        if has_live:
            for repo in m.get("repos", []):
                rd = github_data.get(repo, {})
                by_ms = rd.get("by_milestone", {})
                for ms_key, count in by_ms.items():
                    if milestone_name.lower() in ms_key.lower():
                        live_count = (live_count or 0) + count

        open_str = f"{live_count} open issues" if live_count is not None else f"{m.get('open_items', '?')} open items"
        lines.append(f"- **{milestone_name}**: [{status_icon}] {due_str}, owner: {owner}, {open_str}")

        cp = m.get("critical_path")
        if cp:
            lines.append(f"  - Critical path: {cp}")

    # Open PRs (live data)
    all_prs: List[Dict] = []
    if has_live:
        all_repos = list(github_data.keys())
        for repo, rd in github_data.items():
            for pr in rd.get("prs", []):
                pr["repo"] = short_repo(repo, all_repos)
                all_prs.append(pr)
        if all_prs:
            lines.append("")
            lines.append("## Open Pull Requests")
            for pr in all_prs:
                lines.append(f"- **{pr['repo']}#{pr['number']}**: {pr['title']} (@{pr['author']}, updated {pr['updated']})")

    # Blockers (YAML + live blocked issues)
    blockers = data.get("blockers", [])
    live_blocked: List[Dict] = []
    if has_live:
        all_repos = list(github_data.keys())
        for repo, rd in github_data.items():
            for b in rd.get("blocked_issues", []):
                live_blocked.append({"repo": short_repo(repo, all_repos), **b})

    if blockers or live_blocked:
        lines.append("")
        lines.append("## Blockers")
        for b in blockers:
            severity = b.get("severity", "?").upper()
            since = b.get("blocked_since", "?")
            lines.append(f"- [{severity}] {b.get('issue', '?')}: {b.get('title', '?')} (since {since}, owner: {b.get('owner', 'unassigned')})")
            notes = b.get("notes")
            if notes:
                lines.append(f"  - {notes}")
        # Add any GitHub-labeled blocked issues not already in YAML
        yaml_issues = {b.get("issue", "").replace("#", "") for b in blockers}
        for lb in live_blocked:
            if str(lb["number"]) not in yaml_issues:
                lines.append(f"- [GITHUB] {lb['repo']}#{lb['number']}: {lb['title']}")

    # Recent activity highlights (live data)
    if has_live:
        active_repos = [
            (repo, rd) for repo, rd in github_data.items()
            if rd.get("comments_24h", 0) > 0
        ]
        if active_repos:
            lines.append("")
            lines.append("## Recent Activity")
            for repo, rd in active_repos:
                short = short_repo(repo, list(github_data.keys()))
                lines.append(f"- **{short}**: {rd['comments_24h']} comments today")
                for c in rd.get("recent_comments", []):
                    lines.append(f"  - #{c['issue_url']} (@{c['author']}): {c['snippet']}...")

    # Suggested work items (based on milestones + blockers)
    lines.append("")
    lines.append("## Suggested Work Items (by impact)")
    suggestions: List[str] = []

    # Suggest unblocking blockers first
    for b in blockers:
        if b.get("severity") in ("high", "critical"):
            suggestions.append(f"Unblock {b.get('issue', '?')}: {b.get('title', '?')}")

    # Suggest work on in-progress milestones
    for m in data.get("milestones", []):
        if m.get("status") in ("in_progress", "on_track", "at_risk"):
            cp = m.get("critical_path")
            if cp:
                suggestions.append(f"{m.get('name', '?')}: {cp}")

    # Suggest reviewing open PRs
    if has_live and all_prs:
        suggestions.append(f"Review {len(all_prs)} open PR(s)")

    for i, s in enumerate(suggestions[:5], 1):
        lines.append(f"{i}. {s}")

    if not suggestions:
        lines.append("No urgent items detected. Review backlog for next priorities.")

    lines.append("")
    if not has_live:
        lines.append("*Set GITHUB_TOKEN to enable live repo scanning.*")
        lines.append("")
    lines.append("**Your call. What resonates?**")

    return "\n".join(lines)


async def generate_portfolio_dashboard(data: Dict[str, Any]) -> str:
    """Generate the Portfolio Dashboard summary.

    Args:
        data: The strategic memory data dict.

    Returns:
        Formatted markdown dashboard string.
    """
    config = data.get("morning_signal_config", {})
    repos = config.get("scan_repos", [])

    # Try to detect the host port from environment or default
    host_port = os.environ.get("KESTREL_HOST_PORT", "8888")

    lines = ["# Portfolio Dashboard", ""]
    lines.append(f"**Open in browser:** http://localhost:{host_port}/static/dashboard.html")
    lines.append("")

    # Quick summary from live data
    from .github_integration import get_github_token, github_api_get

    token = get_github_token()
    if token and repos:
        total_issues = 0
        total_prs = 0
        for repo in repos:
            issues = await github_api_get(
                f"/repos/{repo}/issues?state=open&per_page=100", token,
            )
            if isinstance(issues, list):
                real = [i for i in issues if "pull_request" not in i]
                total_issues += len(real)

            prs = await github_api_get(
                f"/repos/{repo}/pulls?state=open&per_page=20", token,
            )
            if isinstance(prs, list):
                total_prs += len(prs)

        lines.append(f"**Quick snapshot:** {total_issues} open issues, {total_prs} open PRs across {len(repos)} repos")
    else:
        lines.append("*Set GITHUB_TOKEN to see live data in the dashboard.*")

    lines.append("")
    lines.append("The dashboard has 4 tabs:")
    lines.append("1. **Operational** -- open issues, PRs, blockers, backlog health, activity")
    lines.append("2. **Strategic** -- milestones, velocity trend, decision log")
    lines.append("3. **Scoreboard** -- outcome rankings by contributor")
    lines.append("4. **Budget** -- engineering cost savings, ceremony cost avoided")

    return "\n".join(lines)
