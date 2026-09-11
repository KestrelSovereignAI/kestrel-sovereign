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
        # Selection now confirms the target is an open issue before handing it
        # to an irreversible dispatch. This test used to pass with no GitHub
        # read at all -- which is the defect #3280 describes.
        _stub_github(monkeypatch, {"/repos/owner/repo/issues/7": _open(7, "valid")})

        picked = await pick_top_issue(data)

        assert picked is not None, "the valid blocker must still be reachable"
        assert picked["repo"] == "owner/repo"
        assert picked["issue_number"] == 7



# ---------------------------------------------------------------------------
# #3280: a blocker is dispatched only if GitHub says it is an open issue, in a
# repository the row actually names
# ---------------------------------------------------------------------------


def _open(number, title="t"):
    return {"number": number, "title": title, "state": "open", "labels": []}


def _closed(number, title="t"):
    return {"number": number, "title": title, "state": "closed", "labels": []}


def _stub_github(monkeypatch, responses, calls=None):
    """Answer github_api_get from ``responses``; anything else is a 404."""
    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")

    async def fake(path, token):
        if calls is not None:
            calls.append(path)
        if path in responses:
            value = responses[path]
            if isinstance(value, Exception):
                raise value
            return value
        # The backlog scan's list endpoints: nothing open, so a test observes
        # the blocker path's decision rather than a fallback's.
        if "/issues?" in path:
            return []
        raise RuntimeError(f"404 {path}")

    monkeypatch.setattr(issue_selection, "github_api_get", fake)


FOURTEEN = [f"org/repo{i}" for i in range(14)]


@pytest.mark.asyncio
async def test_the_live_host_shape_a_closed_issue_is_not_dispatched(monkeypatch):
    """What the live host did every morning: a bare '#2665' among fourteen scan
    repos, resolved to the first repo that had ANY issue 2665 -- closed since
    2026-07-28 -- and dispatched under a title that was not the issue's."""
    calls = []
    _stub_github(
        monkeypatch,
        {"/repos/org/repo0/issues/2665": _closed(2665, "Restart coordinator busy-tracker")},
        calls,
    )
    data = {
        "morning_signal_config": {"scan_repos": FOURTEEN},
        "blockers": [
            {"severity": "high", "issue": "#2665",
             "title": "Cross-restart re-arm of durable a2a: waits still unverified"},
        ],
    }

    assert await issue_selection.pick_top_issue(data) is None
    # Not merely filtered after a guess: an ambiguous row is never looked up.
    assert not any("/issues/2665" in c for c in calls), calls


@pytest.mark.asyncio
async def test_a_closed_blocker_gives_way_to_the_open_one_behind_it(monkeypatch):
    _stub_github(monkeypatch, {
        "/repos/o/r/issues/1": _closed(1),
        "/repos/o/r/issues/2": _open(2, "still live"),
    })
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [
            {"severity": "critical", "issue": "o/r#1", "title": "old"},
            {"severity": "high", "issue": "o/r#2", "title": "new"},
        ],
    }

    picked = await issue_selection.pick_top_issue(data)

    assert picked is not None and picked["issue_number"] == 2


@pytest.mark.asyncio
async def test_an_ambiguous_number_is_never_guessed_into_a_repository(monkeypatch):
    """Even when the number exists, open, in one of the repos. With several
    configured, a bare number names no project; the reconciler already refuses
    this guess, and dispatch is the irreversible reader of the same row."""
    calls = []
    _stub_github(monkeypatch, {"/repos/a/b/issues/5": _open(5)}, calls)
    data = {
        "morning_signal_config": {"scan_repos": ["a/b", "c/d"]},
        "blockers": [{"severity": "high", "issue": "5", "title": "x"}],
    }

    assert await issue_selection.pick_top_issue(data) is None
    assert "/repos/a/b/issues/5" not in calls


@pytest.mark.asyncio
async def test_a_bare_number_with_one_repository_is_not_a_guess(monkeypatch):
    _stub_github(monkeypatch, {"/repos/only/one/issues/9": _open(9, "real")})
    data = {
        "morning_signal_config": {"scan_repos": ["only/one"]},
        "blockers": [{"severity": "high", "issue": "#9", "title": "x"}],
    }

    picked = await issue_selection.pick_top_issue(data)

    assert picked is not None
    assert (picked["repo"], picked["issue_number"]) == ("only/one", 9)


@pytest.mark.asyncio
async def test_an_unreadable_issue_is_not_a_live_target(monkeypatch):
    """A lookup failure must not be read as 'still blocking' for work that
    writes code."""
    _stub_github(monkeypatch, {"/repos/o/r/issues/3": RuntimeError("502")})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#3", "title": "x"}],
    }

    assert await issue_selection.pick_top_issue(data) is None


@pytest.mark.asyncio
async def test_a_pull_request_number_is_not_an_issue_to_dispatch(monkeypatch):
    pr = {**_open(4), "pull_request": {"url": "..."}}
    _stub_github(monkeypatch, {"/repos/o/r/issues/4": pr})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#4", "title": "x"}],
    }

    assert await issue_selection.pick_top_issue(data) is None


@pytest.mark.asyncio
async def test_the_dispatched_title_is_the_issues_own(monkeypatch):
    """Talon works from this title. The ledger row's title is a note about the
    issue, and on the live host it described a different problem."""
    _stub_github(monkeypatch, {"/repos/o/r/issues/8": _open(8, "What GitHub says")})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#8", "title": "A note"}],
    }

    picked = await issue_selection.pick_top_issue(data)

    assert picked["issue_title"] == "What GitHub says"
    assert "A note" in picked["context"]
