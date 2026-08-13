"""Tests for provider-neutral strategic-memory issue selection."""

from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.features.strategic_memory import github_integration
from kestrel_sovereign.features.strategic_memory import issue_selection


@pytest.mark.asyncio
async def test_pick_top_issue_requires_github_token(monkeypatch):
    monkeypatch.setattr(issue_selection, "get_github_token", lambda: None)

    assert await issue_selection.pick_top_issue({}) is None


@pytest.mark.asyncio
async def test_pick_top_issue_requires_scan_repositories(monkeypatch):
    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")

    assert await issue_selection.pick_top_issue({"morning_signal_config": {}}) is None


def test_select_best_candidate_skips_blocked_and_prefers_unassigned_low_comment():
    issues = [
        {
            "number": 1,
            "labels": [{"name": "blocked"}],
            "assignees": [],
            "comments": 0,
        },
        {"number": 2, "labels": [], "assignees": [{"login": "owner"}], "comments": 0},
        {"number": 3, "labels": [], "assignees": [], "comments": 4},
        {"number": 4, "labels": [], "assignees": [], "comments": 1},
    ]

    assert issue_selection._select_best_candidate(issues)["number"] == 4


@pytest.mark.asyncio
async def test_pick_top_issue_does_not_fetch_unused_morning_projection(monkeypatch):
    """#2813: selection uses targeted issue reads, not a no-op broad fetch."""

    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")
    broad_fetch = AsyncMock(side_effect=AssertionError("unused broad fetch"))
    monkeypatch.setattr(github_integration, "fetch_github_signal", broad_fetch)
    monkeypatch.setattr(
        issue_selection,
        "_fetch_open_issues",
        AsyncMock(return_value=[{
            "number": 17,
            "title": "Targeted issue",
            "labels": [],
            "assignees": [],
            "comments": 0,
        }]),
    )

    picked = await issue_selection.pick_top_issue({
        "morning_signal_config": {"scan_repos": ["owner/repo"]},
    })

    assert picked["issue_number"] == 17
    broad_fetch.assert_not_awaited()
