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


class TestReferencePrefixMustLookLikeARepository:
    """A prose prefix is not a repository.

    ``parse_issue_ref`` treated any non-empty prefix as one, so a handwritten
    ``Issue #123`` produced the repository ``Issue``. ``pick_top_issue``
    returns on its first candidate, so such a row did not merely dispatch
    against an invalid target — it masked every valid blocker behind it. The
    fix for one wrong-repository path had opened another.
    """

    def test_prose_prefixes_do_not_become_repositories(self):
        from kestrel_sovereign.features.strategic_memory.issue_selection import (
            parse_issue_ref,
        )

        for text in ("Issue #123", "not-a-repo#123", "see FIXME #7"):
            repo, number = parse_issue_ref(text)
            assert repo is None, f"{text!r} must not yield a repository"
            assert number is not None, f"{text!r} still names an issue number"

    def test_owner_repo_shapes_are_still_recognised(self):
        from kestrel_sovereign.features.strategic_memory.issue_selection import (
            parse_issue_ref,
        )

        assert parse_issue_ref("owner/repo#123") == ("owner/repo", 123)
        assert parse_issue_ref("Kestrel.AI/kestrel-x#9") == ("Kestrel.AI/kestrel-x", 9)
        assert parse_issue_ref("#42") == (None, 42)

    @pytest.mark.asyncio
    async def test_a_prose_blocker_does_not_mask_the_valid_one_behind_it(
        self, monkeypatch
    ):
        monkeypatch.setenv("GITHUB_TOKEN", "dummy")
        from kestrel_sovereign.features.strategic_memory.issue_selection import (
            pick_top_issue,
        )

        data = {
            "morning_signal_config": {"scan_repos": []},
            "blockers": [
                {"severity": "critical", "issue": "Issue #123", "title": "prose"},
                {"severity": "high", "issue": "owner/repo#7", "title": "valid"},
            ],
        }

        picked = await pick_top_issue(data)

        assert picked is not None, "the valid blocker must still be reachable"
        assert picked["repo"] == "owner/repo"
        assert picked["issue_number"] == 7
