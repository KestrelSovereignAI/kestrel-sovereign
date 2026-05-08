"""Backlog Hygiene scanner for Strategic Memory.

Scans GitHub repos for issues missing assignees, milestones,
labels, or stale issues, and optionally auto-fixes by adding labels.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List

from .github_integration import (
    get_github_token,
    github_api_get,
    github_api_post,
    short_repo,
)

logger = logging.getLogger(__name__)


def is_auto_fix(fix: str) -> bool:
    """Truthy check matching run_backlog_hygiene's predicate.

    Exposed so the @tool wrapper in feature.py can use the same rule
    when deciding whether to surface a dry-run PARTIAL caveat — the
    runner accepts "yes", "true", "1" (case-insensitive); anything
    else means report-only.
    """
    return fix.lower() in ("yes", "true", "1")


async def run_backlog_hygiene(data: Dict[str, Any], fix: str = "no") -> str:
    """Scan repos for backlog hygiene issues and optionally auto-fix.

    Args:
        data: The strategic memory data dict (needs morning_signal_config).
        fix: Set to 'yes' to auto-fix issues where possible (add labels). Default 'no'.

    Returns:
        Formatted markdown hygiene report.
    """
    config = data.get("morning_signal_config", {})
    repos = config.get("scan_repos", [])
    if not repos:
        return "No scan_repos configured in morning_signal_config."

    token = get_github_token()
    if not token:
        return "No GITHUB_TOKEN found. Set GITHUB_TOKEN environment variable or add to .env file."

    auto_fix = is_auto_fix(fix)
    today_str = date.today().strftime("%B %d, %Y")
    lines = [f"# Backlog Hygiene Report -- {today_str}", ""]

    total_issues = 0
    missing_assignee: List[Dict] = []
    missing_milestone: List[Dict] = []
    missing_labels: List[Dict] = []
    stale_issues: List[Dict] = []
    needs_review: List[Dict] = []
    fixes_applied: List[str] = []

    all_repos = repos[:]

    for repo in repos:
        issues = await github_api_get(
            f"/repos/{repo}/issues?state=open&per_page=100", token,
        )
        if not isinstance(issues, list):
            lines.append(f"- **{repo}**: API error, skipped")
            continue

        real_issues = [i for i in issues if "pull_request" not in i]
        short = short_repo(repo, all_repos)
        total_issues += len(real_issues)

        for issue in real_issues:
            num = issue["number"]
            title = issue["title"]
            labels = [l["name"] for l in issue.get("labels", [])]
            assignee = issue.get("assignee")
            milestone = issue.get("milestone")
            updated = issue.get("updated_at", "")[:10]

            entry = {
                "repo": repo,
                "short_repo": short,
                "number": num,
                "title": title,
                "labels": labels,
                "updated": updated,
            }

            # Check: missing assignee
            if not assignee:
                missing_assignee.append(entry)

            # Check: missing milestone
            if not milestone:
                missing_milestone.append(entry)

            # Check: missing status label
            status_labels = {"status:blocked", "status:in-progress",
                             "status:ready", "status:done", "status:backlog"}
            has_status = any(l.lower() in status_labels or
                            l.lower().startswith("status:") for l in labels)
            has_any_label = len(labels) > 0
            if not has_any_label:
                missing_labels.append(entry)

            # Check: stale (not updated in 14+ days)
            if updated:
                try:
                    updated_date = datetime.strptime(updated, "%Y-%m-%d").date()
                    days_stale = (date.today() - updated_date).days
                    if days_stale >= 14:
                        entry["days_stale"] = days_stale
                        stale_issues.append(entry)
                except ValueError:
                    pass

            # Flag for human review: no assignee AND no milestone
            if not assignee and not milestone:
                needs_review.append(entry)

    # Build report
    lines.append("## Summary")
    lines.append(f"- **{total_issues}** open issues across {len(repos)} repos")
    lines.append(f"- **{len(missing_assignee)}** missing assignee")
    lines.append(f"- **{len(missing_milestone)}** missing milestone")
    lines.append(f"- **{len(missing_labels)}** missing labels")
    lines.append(f"- **{len(stale_issues)}** stale (14+ days without update)")
    lines.append(f"- **{len(needs_review)}** need human review (no assignee + no milestone)")
    lines.append("")

    if missing_assignee:
        lines.append("## Missing Assignee")
        for e in missing_assignee:
            lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']}")
        lines.append("")

    if missing_milestone:
        lines.append("## Missing Milestone")
        for e in missing_milestone:
            lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']}")
        lines.append("")

    if missing_labels:
        lines.append("## Missing Labels")
        for e in missing_labels:
            lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']}")

        # Auto-fix: add 'needs-triage' label
        if auto_fix:
            lines.append("")
            lines.append("### Auto-fix: Adding 'needs-triage' label")
            for e in missing_labels:
                result = await github_api_post(
                    f"/repos/{e['repo']}/issues/{e['number']}/labels",
                    token,
                    {"labels": ["needs-triage"]},
                )
                if result is not None:
                    fixes_applied.append(
                        f"Added 'needs-triage' to {e['short_repo']}#{e['number']}"
                    )
                    lines.append(f"  - Added to {e['short_repo']}#{e['number']}")
                else:
                    lines.append(f"  - FAILED: {e['short_repo']}#{e['number']}")
        lines.append("")

    if stale_issues:
        lines.append("## Stale Issues (14+ days)")
        for e in stale_issues:
            lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']} ({e['days_stale']} days)")
        lines.append("")

    if needs_review:
        lines.append("## Needs Human Review")
        lines.append("*These issues have no assignee AND no milestone -- the AI agent cannot determine who owns them or which workstream they belong to.*")
        lines.append("")
        for e in needs_review:
            lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']}")
        lines.append("")

    # Health score
    if total_issues > 0:
        issues_ok = total_issues - len(missing_assignee) - len(missing_milestone)
        health_pct = round((issues_ok / total_issues) * 100)
        lines.append(f"## Backlog Health Score: {health_pct}%")
        if health_pct >= 90:
            lines.append("Backlog is clean.")
        elif health_pct >= 70:
            lines.append("Backlog needs minor cleanup.")
        else:
            lines.append("Backlog needs attention -- too many unowned or untracked issues.")
    lines.append("")

    if fixes_applied:
        lines.append(f"**{len(fixes_applied)} auto-fixes applied.**")
    elif auto_fix:
        lines.append("**No auto-fixes were needed.**")
    else:
        lines.append("*Run `!hygiene fix=yes` to auto-apply labels where possible.*")

    return "\n".join(lines)
