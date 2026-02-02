#!/usr/bin/env python3
"""Create GitHub issues for high-confidence recurring patterns.

This script:
1. Reads test feedback from SQLite database
2. Analyzes patterns using TestResultAnalyzer
3. Creates GitHub issues for high-confidence (>80%) actionable patterns
4. Avoids duplicates by checking existing issues

Run manually or via weekly-analysis.yml workflow.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

MIN_CONFIDENCE = 0.8  # Only create issues for high-confidence patterns
ISSUE_LABEL = "automated-test-analysis"


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
        return []
    finally:
        conn.close()


def analyze_patterns(entries: list[dict]) -> list:
    """Analyze feedback entries for patterns."""
    try:
        from tests.utils.test_result_analyzer import TestResultAnalyzer
        from tests.utils.feedback_bridge import TestResult, TestOutcome

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
    except ImportError as e:
        print(f"Could not import analyzer: {e}")
        return []


def issue_exists(title: str, repo: str, token: str) -> bool:
    """Check if issue with same title already exists."""
    try:
        import httpx
    except ImportError:
        return False

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"state": "all", "labels": ISSUE_LABEL, "per_page": 100}

    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30.0)
        response.raise_for_status()
        issues = response.json()
        return any(issue["title"] == title for issue in issues)
    except Exception as e:
        print(f"Error checking existing issues: {e}")
        return False


def create_issue(insight, repo: str, token: str) -> bool:
    """Create GitHub issue from insight. Returns True if created."""
    try:
        import httpx
    except ImportError:
        print("httpx not available, cannot create issues")
        return False

    title = f"[Test Analysis] {insight.title}"

    if issue_exists(title, repo, token):
        print(f"Issue already exists: {title}")
        return False

    # Format evidence as bullet list
    evidence_list = "\n".join(f"- `{e}`" for e in insight.evidence[:10])

    body = f"""## Automated Test Analysis

**Type:** {insight.insight_type.value}
**Confidence:** {insight.confidence:.0%}
**Severity:** {insight.severity}

### Description
{insight.description}

### Suggested Action
{insight.suggested_action or "Review and investigate"}

### Evidence
{evidence_list}

---
*This issue was automatically created by the test analysis system.*
"""

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body,
        "labels": [ISSUE_LABEL, f"severity-{insight.severity}"]
    }

    try:
        response = httpx.post(url, json=data, headers=headers, timeout=30.0)
        response.raise_for_status()
        print(f"Created issue: {response.json()['html_url']}")
        return True
    except Exception as e:
        print(f"Failed to create issue: {e}")
        return False


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    db_path = os.environ.get("KESTREL_FEEDBACK_DB", "/tmp/test_feedback.db")

    if not token:
        print("GITHUB_TOKEN not set, cannot create issues")
        sys.exit(1)

    if not repo:
        print("GITHUB_REPOSITORY not set, cannot create issues")
        sys.exit(1)

    print(f"Looking for feedback database at: {db_path}")

    if not Path(db_path).exists():
        print("No feedback database found - nothing to analyze")
        return

    entries = get_feedback_entries(db_path)
    print(f"Found {len(entries)} feedback entries")

    if not entries:
        print("No feedback entries to analyze")
        return

    insights = analyze_patterns(entries)
    print(f"Generated {len(insights)} insights")

    # Filter to high-confidence actionable insights
    actionable = [i for i in insights if i.confidence >= MIN_CONFIDENCE and i.actionable]
    print(f"Found {len(actionable)} high-confidence actionable insights")

    created_count = 0
    for insight in actionable:
        if create_issue(insight, repo, token):
            created_count += 1

    print(f"Created {created_count} new issues")


if __name__ == "__main__":
    main()
