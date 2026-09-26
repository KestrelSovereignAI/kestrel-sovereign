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


def test_ranked_candidates_skip_blocked_and_prefer_unassigned_low_comment():
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

    assert issue_selection._ranked_candidates(issues)[0]["number"] == 4


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
    monkeypatch.setattr(
        issue_selection, "_fetch_open_linked_pull_requests", AsyncMock(return_value=[])
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


def _stub_github(monkeypatch, responses, calls=None, linked_prs=None):
    """Answer github_api_get from ``responses``; anything else is a 404.

    The PR-linkage GraphQL read answers from ``linked_prs`` keyed by
    ``(repo, number)``: a list of PR nodes, ``None`` for an unreadable read,
    or an exception to raise. Absent means no PR links the issue.
    """
    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")
    linked_prs = linked_prs or {}

    async def fake_post(path, token, body):
        assert path == "/graphql", path
        variables = body["variables"]
        key = (f"{variables['owner']}/{variables['name']}", variables["number"])
        if calls is not None:
            calls.append(("graphql",) + key)
        value = linked_prs.get(key, [])
        if isinstance(value, Exception):
            raise value
        if value is None:
            return None
        return {"data": {"repository": {"issue": {
            "closedByPullRequestsReferences": {"nodes": value},
        }}}}

    monkeypatch.setattr(issue_selection, "github_api_post", fake_post)

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


# ---------------------------------------------------------------------------
# #3280 review round 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unreadable_issue_in_its_production_shape_is_not_open(monkeypatch):
    """[P2] github_api_get does not raise on a 404, a 410, a rate limit or a
    network error: it returns None. The first "unreadable" test raised, which
    only exercised _fetch_issue's except branch -- so a mutant turning an
    unreadable read into a phantom OPEN issue, the exact defect this ticket is
    about, passed all of it."""
    _stub_github(monkeypatch, {"/repos/o/r/issues/3": None})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#3", "title": "x"}],
    }

    assert await issue_selection.pick_top_issue(data) is None


@pytest.mark.asyncio
async def test_a_blocker_that_names_its_repo_in_a_field_is_not_ambiguous(monkeypatch):
    """[P3] The repo FIELD is how 31 of the live host's 33 qualifying blockers
    name their repository -- a bare number plus repo: owner/name -- and that
    leg of the contract had no test. With fourteen scan repos a bare number
    would otherwise be skipped as ambiguous."""
    _stub_github(monkeypatch, {"/repos/org/repo3/issues/51": _open(51, "named")})
    data = {
        "morning_signal_config": {"scan_repos": FOURTEEN},
        "blockers": [{"severity": "high", "issue": "51", "repo": "org/repo3", "title": "x"}],
    }

    picked = await issue_selection.pick_top_issue(data)

    assert picked is not None
    assert (picked["repo"], picked["issue_number"]) == ("org/repo3", 51)


@pytest.mark.asyncio
async def test_a_target_named_by_many_rows_is_read_once(monkeypatch):
    """[P3] The walk is one GitHub read per distinct (repo, number). On the live
    host 12 of 33 qualifying rows named the same pull request -- a guaranteed
    miss, read twelve times every run -- and the ledger only grows."""
    calls = []
    pr = {**_open(3112), "pull_request": {"url": "..."}}
    _stub_github(monkeypatch, {
        "/repos/o/r/issues/3112": pr,
        "/repos/o/r/issues/7": _open(7, "live"),
    }, calls)
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": (
            [{"severity": "high", "issue": "3112", "repo": "o/r", "title": f"pr {i}"} for i in range(12)]
            + [{"severity": "high", "issue": "o/r#7", "title": "live"}]
        ),
    }

    picked = await issue_selection.pick_top_issue(data)

    assert picked["issue_number"] == 7
    assert calls.count("/repos/o/r/issues/3112") == 1, calls


@pytest.mark.asyncio
async def test_diagnostics_say_when_nothing_could_be_confirmed(monkeypatch):
    """[P3] None means either "nothing is actionable" or "GitHub could not
    confirm anything". The caller can tell them apart only if selection says
    how many blockers it checked and how many were unreadable."""
    _stub_github(monkeypatch, {"/repos/o/r/issues/1": None, "/repos/o/r/issues/2": None})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [
            {"severity": "high", "issue": "o/r#1", "title": "a"},
            {"severity": "critical", "issue": "o/r#2", "title": "b"},
        ],
    }
    diagnostics: dict = {}

    assert await issue_selection.pick_top_issue(data, diagnostics) is None
    assert diagnostics == {
        "blockers_checked": 2,
        "blockers_unreadable": 2,
        "blockers_talon_owned": 0,
        "open_pr_exclusions": [],
    }


@pytest.mark.asyncio
async def test_a_closed_blocker_is_checked_but_not_unreadable(monkeypatch):
    """The control: a blocker GitHub answered for, closed, is a real answer --
    it must not be counted with the unreadable ones, or an all-closed ledger
    would be reported as an outage."""
    _stub_github(monkeypatch, {"/repos/o/r/issues/1": _closed(1)})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#1", "title": "a"}],
    }
    diagnostics: dict = {}

    assert await issue_selection.pick_top_issue(data, diagnostics) is None
    assert diagnostics == {
        "blockers_checked": 1,
        "blockers_unreadable": 0,
        "blockers_talon_owned": 0,
        "open_pr_exclusions": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [None, "", "unknown", "OPEN"])
async def test_only_an_explicitly_open_issue_is_dispatched(monkeypatch, state):
    """The suite pinned only that a CLOSED issue is refused. ``state != "closed"``
    would pass all of that while dispatching an issue whose state GitHub did
    not state -- the permissive reading, for the one action that writes code."""
    issue = {"number": 5, "title": "t", "labels": []}
    if state is not None:
        issue["state"] = state
    _stub_github(monkeypatch, {"/repos/o/r/issues/5": issue})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#5", "title": "x"}],
    }

    assert await issue_selection.pick_top_issue(data) is None


def _labelled(number, *names, title="t"):
    return {
        "number": number, "title": title, "state": "open",
        "labels": [{"name": n} for n in names],
    }


@pytest.mark.asyncio
async def test_a_blocker_talon_owns_gives_way_to_the_next_one(monkeypatch):
    """#3051 on the live host, 09-12 through 09-15: open, top blocker, and
    already carrying Talon's answer -- clarification asked, then failed. The
    dispatcher never looked at the label and re-ran the same claim daily."""
    _stub_github(monkeypatch, {
        "/repos/o/r/issues/3051": _labelled(3051, "tech-debt", "agent-clarifying"),
        "/repos/o/r/issues/3052": _open(3052, "next"),
    })
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [
            {"severity": "high", "issue": "o/r#3051", "title": "asked"},
            {"severity": "high", "issue": "o/r#3052", "title": "free"},
        ],
    }
    diagnostics = {}

    picked = await issue_selection.pick_top_issue(data, diagnostics)

    assert picked is not None and picked["issue_number"] == 3052
    assert diagnostics["blockers_talon_owned"] == 1
    assert diagnostics["blockers_checked"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label",
    ["agent-analyzing", "agent-clarifying", "agent-claimed", "agent-blocked",
     "agent-failed", "Agent-Failed"],
)
async def test_every_talon_state_label_withholds_dispatch(monkeypatch, label):
    _stub_github(monkeypatch, {"/repos/o/r/issues/1": _labelled(1, label)})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "critical", "issue": "o/r#1", "title": "t"}],
    }
    diagnostics = {}

    assert await issue_selection.pick_top_issue(data, diagnostics) is None
    assert diagnostics["blockers_talon_owned"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["agent-ready", "agent-complete", "tech-debt"])
async def test_an_instruction_or_a_finished_run_does_not_withhold(monkeypatch, label):
    """``agent-ready`` tells Talon to skip clarification; it must still dispatch.

    ``agent-complete`` with no open PR (one closed unmerged, say) is not in
    flight. An open PR withholds the issue by itself -- see the #3317 tests."""
    _stub_github(monkeypatch, {"/repos/o/r/issues/1": _labelled(1, label)})
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "critical", "issue": "o/r#1", "title": "t"}],
    }

    picked = await issue_selection.pick_top_issue(data)

    assert picked is not None and picked["issue_number"] == 1


def test_backlog_scan_skips_talon_owned_issues_too():
    issues = [
        _labelled(1, "agent-failed"),
        _labelled(2, "agent-claimed"),
        _open(3),
    ]
    assert [i["number"] for i in issue_selection._ranked_candidates(issues)] == [3]


def test_talon_state_labels_match_talons_vocabulary():
    """A copy of ``kestreltalon/config.py``'s state labels, pinned here because
    core cannot import the Talon package. Change both or neither."""
    assert issue_selection.TALON_STATE_LABELS == frozenset({
        "agent-analyzing",   # label_analyzing
        "agent-clarifying",  # label_clarifying
        "agent-claimed",     # label_in_progress
        "agent-blocked",     # label_blocked
        "agent-failed",      # label_failed
    })



# ---------------------------------------------------------------------------
# #3317: an issue an open pull request already works is in flight, not idle
# ---------------------------------------------------------------------------


def _pr(number, *, updated_at="2026-09-22T12:00:00Z", repo=None, draft=False):
    node = {
        "number": number,
        "state": "OPEN",
        "isDraft": draft,
        "updatedAt": updated_at,
        "url": f"https://github.com/o/r/pull/{number}",
    }
    if repo is not None:
        node["repository"] = {"nameWithOwner": repo}
    return node


def _fresh_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@pytest.mark.asyncio
async def test_the_live_shape_a_blocker_with_an_open_pr_is_not_re_picked(monkeypatch):
    """09-22: #3310 was selected while PR #3311 -- ``Fixes #3310``, CI green,
    awaiting review -- was open. The issue stays open by design until the PR
    merges, so open-state and labels could not tell it from idle work."""
    _stub_github(
        monkeypatch,
        {
            "/repos/o/r/issues/3310": _labelled(3310, "bug", "agent-ready", "agent-complete"),
            "/repos/o/r/issues/3312": _open(3312, "idle"),
        },
        linked_prs={("o/r", 3310): [_pr(3311, updated_at=_fresh_now())]},
    )
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [
            {"severity": "critical", "issue": "o/r#3310", "title": "in flight"},
            {"severity": "high", "issue": "o/r#3312", "title": "free"},
        ],
    }
    diagnostics = {}

    picked = await issue_selection.pick_top_issue(data, diagnostics)

    assert picked is not None and picked["issue_number"] == 3312
    [exclusion] = diagnostics["open_pr_exclusions"]
    assert (exclusion["repo"], exclusion["issue_number"]) == ("o/r", 3310)
    assert exclusion["reason"] == issue_selection.EXCLUDED_OPEN_PR
    assert [pr["number"] for pr in exclusion["pull_requests"]] == [3311]
    assert issue_selection.describe_exclusion(exclusion) == (
        "skipped o/r#3310 -- PR #3311 open"
    )
    # A real GitHub answer, not an outage.
    assert diagnostics["blockers_unreadable"] == 0


@pytest.mark.asyncio
async def test_the_backlog_scan_skips_an_issue_with_an_open_pr(monkeypatch):
    """The fallback path ranks by assignees and comments; the best-ranked issue
    is exactly the one a fresh PR is most likely to be working."""
    _stub_github(
        monkeypatch,
        {
            "/repos/o/r/issues?state=open&per_page=5&sort=updated": [
                {**_open(1), "comments": 0, "assignees": []},
                {**_open(2), "comments": 3, "assignees": []},
            ],
        },
        linked_prs={("o/r", 1): [_pr(9, updated_at=_fresh_now())]},
    )
    diagnostics = {}

    picked = await issue_selection.pick_top_issue(
        {"morning_signal_config": {"scan_repos": ["o/r"]}}, diagnostics
    )

    assert picked is not None and picked["issue_number"] == 2
    assert [e["issue_number"] for e in diagnostics["open_pr_exclusions"]] == [1]


@pytest.mark.asyncio
async def test_the_milestone_scan_skips_an_issue_with_an_open_pr(monkeypatch):
    _stub_github(
        monkeypatch,
        {
            "/repos/o/r/milestones?state=open&per_page=20": [
                {"number": 4, "title": "Extraction"},
            ],
            "/repos/o/r/issues?milestone=4&state=open&per_page=10&sort=updated": [
                _open(11), _open(12),
            ],
        },
        linked_prs={("o/r", 11): [_pr(13, updated_at=_fresh_now())]},
    )
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "milestones": [
            {"name": "Extraction", "status": "at_risk", "repos": ["o/r"]},
        ],
    }

    picked = await issue_selection.pick_top_issue(data)

    assert picked is not None and picked["issue_number"] == 12


@pytest.mark.asyncio
async def test_a_board_whose_only_candidate_has_an_open_pr_selects_nothing(monkeypatch):
    """The acceptance criterion: not selected, and the reason is recorded."""
    _stub_github(
        monkeypatch,
        {"/repos/o/r/issues/3310": _open(3310)},
        linked_prs={("o/r", 3310): [_pr(3311, updated_at=_fresh_now())]},
    )
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#3310", "title": "x"}],
    }
    diagnostics = {}

    assert await issue_selection.pick_top_issue(data, diagnostics) is None
    # Read once, reported once -- the backlog pass does not see it again
    # because the list endpoint is empty here, and the cache covers repeats.
    assert len(diagnostics["open_pr_exclusions"]) == 1


@pytest.mark.asyncio
async def test_linkage_is_read_once_per_issue_across_passes(monkeypatch):
    calls = []
    _stub_github(
        monkeypatch,
        {
            "/repos/o/r/issues/5": _open(5),
            "/repos/o/r/issues?state=open&per_page=5&sort=updated": [_open(5)],
        },
        calls,
        linked_prs={("o/r", 5): [_pr(6, updated_at=_fresh_now())]},
    )
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#5", "title": "x"}],
    }
    diagnostics = {}

    assert await issue_selection.pick_top_issue(data, diagnostics) is None
    assert calls.count(("graphql", "o/r", 5)) == 1, calls
    assert len(diagnostics["open_pr_exclusions"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("502")])
async def test_unreadable_pr_linkage_withholds_and_counts_as_unreadable(
    monkeypatch, failure
):
    """Dispatch writes code into a worktree derived from the issue. "Could not
    tell whether a run already owns it" is not "free", and a linkage outage
    must still render as an outage rather than as nothing to do."""
    _stub_github(
        monkeypatch,
        {"/repos/o/r/issues/7": _open(7)},
        linked_prs={("o/r", 7): failure},
    )
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"]},
        "blockers": [{"severity": "high", "issue": "o/r#7", "title": "x"}],
    }
    diagnostics = {}

    assert await issue_selection.pick_top_issue(data, diagnostics) is None
    assert diagnostics["blockers_checked"] == 1
    assert diagnostics["blockers_unreadable"] == 1
    [exclusion] = diagnostics["open_pr_exclusions"]
    assert exclusion["reason"] == issue_selection.EXCLUDED_PR_LINKAGE_UNREADABLE
    assert "could not say" in issue_selection.describe_exclusion(exclusion)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"errors": [{"message": "Could not resolve to a Repository"}]},
        # A partial answer: GraphQL returns what it could alongside errors,
        # and an empty ``nodes`` next to an error is not "no PRs".
        {
            "data": {"repository": {"issue": {
                "closedByPullRequestsReferences": {"nodes": []},
            }}},
            "errors": [{"message": "Resource not accessible by integration"}],
        },
        {"data": {"repository": None}},
        {"data": {"repository": {"issue": None}}},
        [],
    ],
)
async def test_a_graphql_answer_without_linkage_is_unreadable_not_empty(
    monkeypatch, response
):
    """GraphQL reports failure inside a 200. Reading a missing ``nodes`` as
    "no PRs" would be the permissive reading for the action that writes code."""
    monkeypatch.setattr(
        issue_selection, "github_api_post", AsyncMock(return_value=response)
    )

    assert await issue_selection._fetch_open_linked_pull_requests("o/r", 1, "t") is None


@pytest.mark.asyncio
async def test_only_open_linked_prs_withhold(monkeypatch):
    nodes = [{**_pr(1), "state": "MERGED"}, {**_pr(2), "state": "CLOSED"}]
    monkeypatch.setattr(
        issue_selection,
        "github_api_post",
        AsyncMock(return_value={"data": {"repository": {"issue": {
            "closedByPullRequestsReferences": {"nodes": nodes},
        }}}}),
    )

    assert await issue_selection._fetch_open_linked_pull_requests("o/r", 1, "t") == []


@pytest.mark.asyncio
async def test_the_linkage_query_names_the_issue_it_is_asked_about(monkeypatch):
    post = AsyncMock(return_value={"data": {"repository": {"issue": {
        "closedByPullRequestsReferences": {"nodes": []},
    }}}})
    monkeypatch.setattr(issue_selection, "github_api_post", post)

    assert await issue_selection._fetch_open_linked_pull_requests(
        "Kestrel.AI/kestrel-x", 3310, "t"
    ) == []
    path, token, body = post.await_args.args
    assert (path, token) == ("/graphql", "t")
    assert body["variables"] == {"owner": "Kestrel.AI", "name": "kestrel-x", "number": 3310}
    assert "closedByPullRequestsReferences" in body["query"]


def test_a_pr_untouched_past_the_threshold_is_named_stalled():
    """Still not a fresh claim -- a second run derives the colliding worktree
    -- but the output says rescue, not silence."""
    from datetime import datetime, timezone

    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    exclusion = issue_selection._open_pr_exclusion(
        "o/r", 3310, [_pr(3311, updated_at="2026-09-20T00:00:00Z")], 3, now=now
    )

    assert exclusion["reason"] == issue_selection.EXCLUDED_STALLED_PR
    assert exclusion["pull_requests"][0]["days_idle"] == 5
    text = issue_selection.describe_exclusion(exclusion)
    assert text.startswith("skipped o/r#3310 -- PR #3311 open but untouched for 5 day(s)")
    assert "rescue" in text


def test_one_moving_pr_or_an_unknown_timestamp_is_not_stalled():
    from datetime import datetime, timezone

    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    moving = issue_selection._open_pr_exclusion(
        "o/r", 1,
        [_pr(2, updated_at="2026-09-01T00:00:00Z"), _pr(3, updated_at="2026-09-24T00:00:00Z")],
        3, now=now,
    )
    unknown = issue_selection._open_pr_exclusion(
        "o/r", 1, [_pr(2, updated_at=None)], 3, now=now
    )
    fresh = issue_selection._open_pr_exclusion(
        "o/r", 1, [_pr(2, updated_at="2026-09-23T00:00:00Z")], 3, now=now
    )

    assert moving["reason"] == unknown["reason"] == fresh["reason"] == (
        issue_selection.EXCLUDED_OPEN_PR
    )


def test_no_linked_pr_is_no_exclusion():
    assert issue_selection._open_pr_exclusion("o/r", 1, [], 3) is None


def test_a_cross_repository_or_draft_pr_is_named_in_full():
    exclusion = issue_selection._open_pr_exclusion(
        "o/core", 1, [_pr(8, repo="o/feature", draft=True, updated_at=_fresh_now())], 3
    )

    assert issue_selection.describe_exclusion(exclusion) == (
        "skipped o/core#1 -- PR o/feature#8 (draft) open"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("configured, expected", [(7, 7), (0, 3), ("5", 3), (True, 3)])
async def test_the_stalled_threshold_is_configurable(monkeypatch, configured, expected):
    _stub_github(
        monkeypatch,
        {"/repos/o/r/issues/1": _open(1)},
        linked_prs={("o/r", 1): [_pr(2)]},
    )
    data = {
        "morning_signal_config": {"scan_repos": ["o/r"], "stalled_pr_days": configured},
        "blockers": [{"severity": "high", "issue": "o/r#1", "title": "x"}],
    }
    diagnostics = {}

    await issue_selection.pick_top_issue(data, diagnostics)

    assert diagnostics["open_pr_exclusions"][0]["stalled_after_days"] == expected
