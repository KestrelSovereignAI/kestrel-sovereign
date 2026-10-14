"""Missing-status-label check in ``run_backlog_hygiene`` (#2812).

The check used to branch on "has any label", so an issue labelled only
``bug`` was never reported as missing a status label. These tests drive the
real runner over a stubbed GitHub API and read the rendered report.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List
from unittest.mock import AsyncMock, patch

from kestrel_sovereign.features.strategic_memory import backlog_hygiene
from kestrel_sovereign.features.strategic_memory.backlog_hygiene import (
    run_backlog_hygiene,
)

REPO = "acme/widgets"
MISSING_STATUS_SECTION = "## Missing Status Label"


def _issue(number: int, labels: List[str]) -> Dict:
    return {
        "number": number,
        "title": f"Issue {number}",
        "labels": [{"name": name} for name in labels],
        "assignee": {"login": "owner"},
        "milestone": {"title": "M1"},
        "updated_at": f"{date.today().isoformat()}T00:00:00Z",
    }


def _section(report: str, header: str) -> List[str]:
    """The report lines under ``header``, up to the next ``## `` header."""
    lines = report.splitlines()
    if header not in lines:
        return []
    body = []
    for line in lines[lines.index(header) + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return body


def _reported(report: str, number: int) -> bool:
    entry = f"- widgets#{number}: Issue {number}"
    return entry in _section(report, MISSING_STATUS_SECTION)


async def _run(issues: List[Dict], fix: str = "no"):
    post = AsyncMock(return_value=[{"name": "needs-triage"}])
    with patch.object(
        backlog_hygiene, "get_github_token", return_value="token",
    ), patch.object(
        backlog_hygiene, "github_api_get", new=AsyncMock(return_value=issues),
    ), patch.object(backlog_hygiene, "github_api_post", new=post):
        report = await run_backlog_hygiene(
            {"morning_signal_config": {"scan_repos": [REPO]}}, fix=fix,
        )
    return report, post


async def test_non_status_label_is_reported_missing_status():
    report, _ = await _run([_issue(1, ["bug"])])

    assert _reported(report, 1)
    assert "- **1** missing status label" in report


async def test_status_label_is_not_reported():
    report, _ = await _run([
        _issue(2, ["status:ready"]),
        _issue(3, ["bug", "Status:Blocked"]),
    ])

    assert not _reported(report, 2)
    assert not _reported(report, 3)
    assert MISSING_STATUS_SECTION not in report
    assert "- **0** missing status label" in report


async def test_unlabelled_issue_is_reported_missing_status():
    report, _ = await _run([_issue(4, [])])

    assert _reported(report, 4)
    assert "- **1** missing status label" in report


async def test_fix_applies_the_same_label_to_non_status_and_unlabelled():
    report, post = await _run([_issue(1, ["bug"]), _issue(4, [])], fix="yes")

    assert [c.args for c in post.await_args_list] == [
        (f"/repos/{REPO}/issues/1/labels", "token", {"labels": ["needs-triage"]}),
        (f"/repos/{REPO}/issues/4/labels", "token", {"labels": ["needs-triage"]}),
    ]
    assert "  - Added to widgets#1" in report
    assert "  - Added to widgets#4" in report
    assert "**2 auto-fixes applied.**" in report


async def test_report_only_run_performs_no_write():
    report, post = await _run([_issue(1, ["bug"]), _issue(4, [])], fix="no")

    post.assert_not_awaited()
    assert _reported(report, 1)
    assert _reported(report, 4)
    assert "auto-fixes applied" not in report


async def test_fix_does_not_claim_to_add_a_label_already_present():
    """``needs-triage`` is not a status label, so a triaged issue is still
    reported. Re-adding the label would change nothing, so the run must not
    POST it or count it as a fix."""
    report, post = await _run([_issue(5, ["Needs-Triage"])], fix="yes")

    assert _reported(report, 5)
    post.assert_not_awaited()
    assert "  - Added to widgets#5" not in report
    assert "  - Already labelled: widgets#5" in report
    assert "**No auto-fixes were needed.**" in report
