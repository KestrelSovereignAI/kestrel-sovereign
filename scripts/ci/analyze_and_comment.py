#!/usr/bin/env python3
"""Analyze test feedback and post PR comment with insights.

This script:
1. Reads test feedback from SQLite database
2. Analyzes patterns using TestResultAnalyzer
3. Formats results as a markdown PR comment
4. Posts the comment to the PR via GitHub API
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def get_feedback_entries(db_path: str) -> list[dict]:
    """Read feedback entries from SQLite database."""
    if not Path(db_path).exists():
        return []

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT category, severity, content, context FROM feedback ORDER BY created_at DESC"
        )
        entries = []
        for row in cursor:
            entries.append({
                "category": row[0],
                "severity": row[1],
                "content": row[2],
                "context": json.loads(row[3]) if row[3] else {}
            })
        return entries
    except sqlite3.OperationalError:
        # Table doesn't exist
        return []
    finally:
        conn.close()


def analyze_patterns(entries: list[dict]) -> list:
    """Analyze feedback entries for patterns."""
    try:
        from tests.utils.test_result_analyzer import TestResultAnalyzer, InsightType
        from tests.utils.feedback_bridge import TestResult, TestOutcome

        # Convert to TestResult format for analyzer
        results = []
        for e in entries:
            ctx = e.get("context", {})
            outcome = TestOutcome.FAILED if e["category"] == "bug" else TestOutcome.ERROR
            results.append(TestResult(
                node_id=ctx.get("test_node_id", "unknown"),
                outcome=outcome,
                duration=ctx.get("duration", 0),
                error_message=e["content"],
            ))

        if not results:
            return []

        analyzer = TestResultAnalyzer()
        return analyzer.analyze(results)
    except ImportError:
        # Analyzer not available
        return []


def format_pr_comment(insights: list, failures: list[dict]) -> str:
    """Format insights as markdown PR comment."""
    if not failures and not insights:
        return "## ✅ All Tests Passed\n\nNo issues detected."

    lines = ["## 🔍 Test Analysis\n"]

    if failures:
        lines.append(f"**{len(failures)} failure(s) detected**\n")

    if not insights:
        # Just list failures without pattern analysis
        lines.append("### Failures\n")
        for f in failures[:10]:  # Limit to 10
            lines.append(f"- `{f.get('context', {}).get('test_node_id', 'unknown')}`: {f['content'][:100]}...")
        if len(failures) > 10:
            lines.append(f"\n*...and {len(failures) - 10} more*")
        return "\n".join(lines)

    # Import InsightType for icon mapping
    try:
        from tests.utils.test_result_analyzer import InsightType
        icon_map = {
            InsightType.PATTERN: "🔄",
            InsightType.REGRESSION: "📉",
            InsightType.FLAKY: "🎲",
            InsightType.HOTSPOT: "🔥",
            InsightType.PERFORMANCE: "⏱️",
        }
    except ImportError:
        icon_map = {}

    # Group insights by type
    for insight in sorted(insights, key=lambda x: -x.confidence):
        icon = icon_map.get(insight.insight_type, "💡")

        lines.append(f"### {icon} {insight.title}")
        lines.append(f"**Confidence:** {insight.confidence:.0%} | **Severity:** {insight.severity}\n")
        lines.append(insight.description)
        if insight.suggested_action:
            lines.append(f"\n**Suggested Action:** {insight.suggested_action}")
        if insight.evidence:
            lines.append("\n<details><summary>Evidence</summary>\n")
            for e in insight.evidence[:5]:
                lines.append(f"- `{e}`")
            lines.append("\n</details>\n")

    return "\n".join(lines)


def post_pr_comment(comment: str) -> None:
    """Post comment to PR using GitHub API."""
    try:
        import httpx
    except ImportError:
        print("httpx not available, printing comment instead:")
        print(comment)
        return

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("PR_NUMBER")

    if not all([token, repo, pr_number]):
        print("Missing required environment variables (GITHUB_TOKEN, GITHUB_REPOSITORY, PR_NUMBER)")
        print("Printing comment instead:")
        print(comment)
        return

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = httpx.post(url, json={"body": comment}, headers=headers, timeout=30.0)
        response.raise_for_status()
        print(f"Posted PR comment: {response.json()['html_url']}")
    except Exception as e:
        print(f"Failed to post PR comment: {e}")
        print("Comment content:")
        print(comment)


def main():
    db_path = os.environ.get("KESTREL_FEEDBACK_DB", "/tmp/test_feedback.db")

    print(f"Looking for feedback database at: {db_path}")

    if not Path(db_path).exists():
        print("No feedback database found - assuming all tests passed!")
        comment = "## ✅ All Tests Passed\n\nNo issues detected in feedback database."
        if os.environ.get("PR_NUMBER"):
            post_pr_comment(comment)
        else:
            print(comment)
        return

    entries = get_feedback_entries(db_path)
    print(f"Found {len(entries)} feedback entries")

    failures = [e for e in entries if e["category"] in ("bug", "error")]
    print(f"Found {len(failures)} failures")

    # Analyze patterns
    insights = analyze_patterns(entries)
    print(f"Generated {len(insights)} insights")

    comment = format_pr_comment(insights, failures)

    if os.environ.get("PR_NUMBER"):
        post_pr_comment(comment)
    else:
        print("No PR_NUMBER set, printing comment:")
        print(comment)


if __name__ == "__main__":
    main()
