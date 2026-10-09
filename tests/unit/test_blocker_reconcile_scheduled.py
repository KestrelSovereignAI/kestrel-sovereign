"""Blocker reconciliation runs on a schedule and reads every row shape (#3537).

Observed on Emma (2026-10-09): #3519 closed on 10-08, but two blocker rows
referencing it stayed active until resolved by hand. Two causes:

* Nothing called the reconciler. ``strategy_reconcile_blockers`` was only
  reachable as ``!strategy-reconcile`` and defaulted to report-only.
* 70 of 122 active rows could not be checked at all: ``repo: self``, a bare
  repository name (``repo: kestrel-feature-talon`` with ``issue:
  kestrel-feature-talon#46``), and bare numbers with no repository.

The acceptance bar: a blocker whose issue closes is resolved by the next
scheduled run without manual action, and a dry run reports 0 unresolvable
rows for those shapes.
"""

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml as _yaml

from kestrel_sovereign.features.strategic_memory import blocker_reconcile
from kestrel_sovereign.features.strategic_memory.blocker_reconcile import (
    INFERRED_REPO_MISMATCH,
    RECONCILIATION_KEY,
    REPO_FROM_ISSUE,
    REPO_FROM_ROW,
    REPO_FROM_SELF_DEFAULT,
    describe_last_reconciliation,
    resolve_blocker_reference,
    split_issue_reference,
)
from kestrel_sovereign.features.strategic_memory.feature import (
    StrategicMemoryFeature,
)
from kestrel_sovereign.features.strategic_memory.ledger import (
    BLOCKERS_KEY,
    LEDGER_FILENAME,
)
from kestrel_sovereign.features.strategic_memory.morning_signal import (
    generate_morning_signal,
)
from kestrel_sovereign.features.strategic_memory.workflow_runs import (
    WorkflowRunsNotAssessed,
)

SELF_REPO = "KestrelSovereignAI/kestrel-sovereign"
SCAN_REPOS = [
    SELF_REPO,
    "KestrelSovereignAI/kestrel-feature-talon",
    "KestrelSovereignAI/kestrel-feature-workflows",
]
_MOD = "kestrel_sovereign.features.strategic_memory.blocker_reconcile"


@pytest.fixture(autouse=True)
def _self_repo(monkeypatch):
    """Pin ``GITHUB_SELF_REPO``; never read the developer's own ``.env``."""
    monkeypatch.setenv("GITHUB_SELF_REPO", SELF_REPO)


def _issue(repo, number, state, *, closed_at=None, created_at="2026-01-01T00:00:00Z"):
    issue = {
        "number": number,
        "state": state,
        "created_at": created_at,
        "html_url": f"https://github.com/{repo}/issues/{number}",
    }
    if state == "closed":
        issue["closed_at"] = closed_at or "2026-10-08T17:04:00Z"
        issue["state_reason"] = "completed"
    return issue


class _FakeGitHub:
    """Serves exactly the issues it holds; records every path asked for."""

    def __init__(self, issues):
        self.issues = {
            f"/repos/{repo}/issues/{number}": body
            for (repo, number), body in issues.items()
        }
        self.paths = []

    async def get(self, path, token, **kwargs):
        self.paths.append(path)
        return self.issues.get(path)


def _patched(github, token="t"):
    return (
        patch(f"{_MOD}.get_github_token", return_value=token),
        patch(f"{_MOD}.github_api_get", new=github.get),
    )


async def _feature(tmp_path, blockers, *, scan_repos=SCAN_REPOS):
    agent = MagicMock()
    agent.agent_id = "did:test:reconcile"
    agent.agent_data_dir = str(tmp_path)
    agent.storage = MagicMock()
    agent.storage.graph = None
    (tmp_path / "STRATEGY.yaml").write_text(
        _yaml.dump(
            {
                "version": 1,
                "morning_signal_config": {"scan_repos": list(scan_repos)},
                BLOCKERS_KEY: blockers,
            }
        ),
        encoding="utf-8",
    )
    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    return feature


def _ledger_on_disk(tmp_path):
    return _yaml.safe_load((tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Every shape in the ticket resolves to one issue in one repository
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "repo", "number", "source"),
    [
        pytest.param(
            {"repo": "self", "issue": "#3517"}, SELF_REPO, 3517, REPO_FROM_ROW,
            id="repo-self",
        ),
        pytest.param(
            {"repo": "SELF", "issue": "3517"}, SELF_REPO, 3517, REPO_FROM_ROW,
            id="repo-self-any-case-bare-number",
        ),
        pytest.param(
            {"repo": "kestrel-feature-talon", "issue": "kestrel-feature-talon#46"},
            "KestrelSovereignAI/kestrel-feature-talon",
            46,
            REPO_FROM_ROW,
            id="short-repo-and-short-issue",
        ),
        pytest.param(
            {"issue": "kestrel-feature-talon#46"},
            "KestrelSovereignAI/kestrel-feature-talon",
            46,
            REPO_FROM_ISSUE,
            id="repo-hash-n-in-issue",
        ),
        pytest.param(
            {"issue": "KestrelSovereignAI/kestrel-feature-workflows#30"},
            "KestrelSovereignAI/kestrel-feature-workflows",
            30,
            REPO_FROM_ISSUE,
            id="owner-repo-hash-n-in-issue",
        ),
        pytest.param(
            {"issue": "self#3518"}, SELF_REPO, 3518, REPO_FROM_ISSUE,
            id="self-hash-n-in-issue",
        ),
        pytest.param(
            {"issue": "#3519"}, SELF_REPO, 3519, REPO_FROM_SELF_DEFAULT,
            id="bare-hash-number",
        ),
        pytest.param(
            {"issue": "3519"}, SELF_REPO, 3519, REPO_FROM_SELF_DEFAULT,
            id="bare-number-string",
        ),
        pytest.param(
            {"issue": 3519}, SELF_REPO, 3519, REPO_FROM_SELF_DEFAULT,
            id="bare-number-unquoted-yaml-int",
        ),
        pytest.param(
            {"repo": "", "issue": "#3519"}, SELF_REPO, 3519, REPO_FROM_SELF_DEFAULT,
            id="empty-repo-bare-number",
        ),
    ],
)
def test_every_ticket_shape_names_one_issue(row, repo, number, source):
    reference = resolve_blocker_reference(row, SCAN_REPOS, SELF_REPO)

    assert (reference.repo, reference.number, reference.problem) == (
        repo, number, None
    )
    assert reference.source == source


def test_a_short_name_prefers_the_configured_repository_of_that_name():
    """``widgets`` on an agent scanning ``Acme/widgets`` is Acme's widgets."""
    reference = resolve_blocker_reference(
        {"repo": "widgets", "issue": "#5"}, ["Acme/widgets", SELF_REPO], SELF_REPO
    )
    assert reference.repo == "Acme/widgets"


def test_a_bare_number_on_an_agent_without_github_config_is_not_assumed():
    """No scan_repos: nothing says this agent's blockers are GitHub issues."""
    reference = resolve_blocker_reference({"issue": "#12"}, [], SELF_REPO)
    assert reference.repo is None
    assert reference.problem == blocker_reconcile.UNRESOLVABLE


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("owner/repo#12", ("owner/repo", 12)),
        ("kestrel-talon#252", ("kestrel-talon", 252)),
        ("#42", (None, 42)),
        ("42", (None, 42)),
        # Text before a spaced ``#`` is prose, never a repository.
        ("Issue #123", (None, 123)),
        ("see FIXME #7", (None, 7)),
        ("not-a-number", (None, None)),
    ],
)
def test_the_reference_grammar(text, expected):
    assert split_issue_reference(text) == expected


# ---------------------------------------------------------------------------
# The dry run: 0 unresolvable rows for the shapes listed in the ticket
# ---------------------------------------------------------------------------


_SHAPE_ROWS = [
    {"id": "blk_self", "repo": "self", "issue": "#3517", "title": "self",
     "blocked_since": "2026-10-06"},
    {"id": "blk_short", "repo": "kestrel-feature-talon",
     "issue": "kestrel-feature-talon#46", "title": "short",
     "blocked_since": "2026-10-01"},
    {"id": "blk_bare", "issue": "#3519", "title": "bare",
     "blocked_since": "2026-10-05"},
    {"id": "blk_bare_int", "issue": 3401, "title": "bare int",
     "blocked_since": "2026-10-05"},
    {"id": "blk_open", "issue": "self#3600", "title": "still open",
     "blocked_since": "2026-10-07"},
]


def _shape_github():
    return _FakeGitHub(
        {
            (SELF_REPO, 3517): _issue(SELF_REPO, 3517, "closed"),
            ("KestrelSovereignAI/kestrel-feature-talon", 46): _issue(
                "KestrelSovereignAI/kestrel-feature-talon", 46, "closed"
            ),
            (SELF_REPO, 3519): _issue(SELF_REPO, 3519, "closed"),
            (SELF_REPO, 3401): _issue(SELF_REPO, 3401, "closed"),
            (SELF_REPO, 3600): _issue(SELF_REPO, 3600, "open"),
        }
    )


@pytest.mark.asyncio
async def test_a_dry_run_checks_every_ticket_shape(tmp_path):
    feature = await _feature(tmp_path, _SHAPE_ROWS)
    github = _shape_github()
    token, get = _patched(github)

    with token, get:
        result = await feature.strategy_reconcile_blockers()

    report = result.data["report"]
    assert report["unresolvable"] == []
    assert result.data["unresolvable_count"] == 0
    assert report["checked"] == 5
    assert {entry["id"] for entry in report["closed"]} == {
        "blk_self", "blk_short", "blk_bare", "blk_bare_int"
    }
    assert [entry["id"] for entry in report["open"]] == ["blk_open"]
    assert sorted(github.paths) == sorted(
        [
            f"/repos/{SELF_REPO}/issues/3517",
            "/repos/KestrelSovereignAI/kestrel-feature-talon/issues/46",
            f"/repos/{SELF_REPO}/issues/3519",
            f"/repos/{SELF_REPO}/issues/3401",
            f"/repos/{SELF_REPO}/issues/3600",
        ]
    )
    # Report-only: nothing resolved, nothing recorded.
    assert result.status.value == "partial"
    assert not any(row.get("resolved_at") for row in feature._ledger.blockers)
    assert RECONCILIATION_KEY not in _ledger_on_disk(tmp_path)


@pytest.mark.asyncio
async def test_apply_cites_the_closing_issue_state(tmp_path):
    feature = await _feature(tmp_path, _SHAPE_ROWS)
    token, get = _patched(_shape_github())

    with token, get:
        result = await feature.strategy_reconcile_blockers(apply="yes")

    assert result.status.value == "ok"
    assert result.data["persisted"] is True
    rows = {row["id"]: row for row in _ledger_on_disk(tmp_path)[BLOCKERS_KEY]}
    assert rows["blk_bare"]["resolved_at"] == str(date.today())
    assert rows["blk_bare"]["resolution"].startswith(
        f"GitHub reports {SELF_REPO}#3519 closed on 2026-10-08 (completed); "
        "resolved by blocker reconciliation"
    )
    assert "names no repository" in rows["blk_bare"]["resolution"], (
        "an assumed repository must say it was assumed"
    )
    assert rows["blk_short"]["resolution"].startswith(
        "GitHub reports KestrelSovereignAI/kestrel-feature-talon#46 closed"
    )
    assert "names no repository" not in rows["blk_short"]["resolution"]
    assert not rows["blk_open"].get("resolved_at")


@pytest.mark.asyncio
async def test_a_row_resolved_during_the_lookup_keeps_its_own_resolution(tmp_path):
    """The GitHub read runs outside the ledger lock; a resolution another
    call wrote meanwhile is not overwritten by the reconcile's note."""
    feature = await _feature(
        tmp_path,
        [{"id": "blk_hand", "repo": "self", "issue": "#3519", "title": "t"}],
    )
    github = _FakeGitHub({(SELF_REPO, 3519): _issue(SELF_REPO, 3519, "closed")})
    real_get = github.get

    async def resolved_meanwhile(path, token, **kwargs):
        await feature.strategy_resolve_blocker("blk_hand", resolution="by hand")
        return await real_get(path, token, **kwargs)

    with patch(f"{_MOD}.get_github_token", return_value="t"), patch(
        f"{_MOD}.github_api_get", new=resolved_meanwhile
    ):
        result = await feature.strategy_reconcile_blockers(apply="yes")

    assert result.data["resolved_ids"] == []
    [row] = _ledger_on_disk(tmp_path)[BLOCKERS_KEY]
    assert row["resolution"] == "by hand"


# ---------------------------------------------------------------------------
# An assumed repository is checked against GitHub's dates before resolving
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_assumed_issue_closed_before_the_blocker_existed_is_left_alone(
    tmp_path,
):
    """``#46`` written for Talon's issue 46 is not kestrel-sovereign#46.

    Low numbers exist in every repository and are long closed in the agent's
    own. An issue that closed before the row was recorded cannot be what the
    row was waiting on, so an unattended apply must not resolve it.
    """
    feature = await _feature(
        tmp_path,
        [{"id": "blk_46", "issue": "#46", "title": "talon's 46",
          "blocked_since": "2026-09-20"}],
    )
    token, get = _patched(
        _FakeGitHub(
            {(SELF_REPO, 46): _issue(
                SELF_REPO, 46, "closed", closed_at="2025-03-02T10:00:00Z"
            )}
        )
    )

    with token, get:
        result = await feature.strategy_reconcile_blockers(apply="yes")

    report = result.data["report"]
    assert report["closed"] == []
    [entry] = report["unresolvable"]
    assert entry["reason"] == INFERRED_REPO_MISMATCH
    assert "2025-03-02" in entry["detail"]
    assert "Set repo on the row" in result.confirmation
    assert not _ledger_on_disk(tmp_path)[BLOCKERS_KEY][0].get("resolved_at")


@pytest.mark.asyncio
async def test_an_assumed_issue_opened_after_the_blocker_is_not_its_issue(tmp_path):
    feature = await _feature(
        tmp_path,
        [{"id": "blk_new", "issue": "#3700", "title": "t",
          "blocked_since": "2026-09-01"}],
    )
    token, get = _patched(
        _FakeGitHub(
            {(SELF_REPO, 3700): _issue(
                SELF_REPO, 3700, "open", created_at="2026-10-05T00:00:00Z"
            )}
        )
    )

    with token, get:
        result = await feature.strategy_reconcile_blockers()

    assert result.data["report"]["open"] == []
    assert result.data["report"]["unresolvable"][0]["reason"] == (
        INFERRED_REPO_MISMATCH
    )


@pytest.mark.asyncio
async def test_a_stated_repository_is_resolved_whatever_the_dates(tmp_path):
    """The date check guards an assumption, not a row that said its repo."""
    feature = await _feature(
        tmp_path,
        [{"id": "blk_stated", "repo": "self", "issue": "#46", "title": "t",
          "blocked_since": "2026-09-20"}],
    )
    token, get = _patched(
        _FakeGitHub(
            {(SELF_REPO, 46): _issue(
                SELF_REPO, 46, "closed", closed_at="2025-03-02T10:00:00Z"
            )}
        )
    )

    with token, get:
        await feature.strategy_reconcile_blockers(apply="yes")

    assert _ledger_on_disk(tmp_path)[BLOCKERS_KEY][0]["resolved_at"]


# ---------------------------------------------------------------------------
# The scheduled run resolves without manual action
# ---------------------------------------------------------------------------


async def _seeded_reconcile(agent):
    """The (cron, args_json) core seeds for the reconciler, or ``None``."""
    from kestrel_sdk.tools.result import ToolResult
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature

    agent.wait_registry = None
    scheduler = SchedulerFeature(agent)
    # Every durable write is stubbed below; the database only has to exist.
    scheduler._db = MagicMock()
    scheduler.schedule_list = AsyncMock(
        return_value=ToolResult.ok(confirmation="ok", data={"tasks": []})
    )
    scheduler._ensure_builtin_schedule = AsyncMock(
        return_value=ToolResult.ok(confirmation="added", data={"next_run_at": None})
    )
    await scheduler.post_all_features_loaded(agent)
    for call in scheduler._ensure_builtin_schedule.await_args_list:
        if call.kwargs["task_name"] == "strategy_reconcile_blockers":
            return call.kwargs["cron_expression"], call.kwargs["args_json"]
    return None


@pytest.mark.asyncio
async def test_core_seeds_a_daily_applying_reconcile_before_the_briefing(tmp_path):
    feature = await _feature(tmp_path, [])
    agent = feature.agent
    agent.features = {"StrategicMemoryFeature": feature}

    seeded = await _seeded_reconcile(agent)

    assert seeded == ("30 7 * * *", '{"apply": "yes"}'), (
        "daily at 07:30 UTC, before the 08:00 morning_signal, applying"
    )


@pytest.mark.asyncio
async def test_no_reconcile_is_seeded_without_strategic_memory():
    agent = MagicMock()
    agent.agent_id = "did:test:no-strategy"
    agent.features = {}

    assert await _seeded_reconcile(agent) is None


@pytest.mark.asyncio
async def test_the_scheduled_run_resolves_a_closed_issue_without_manual_action(
    tmp_path,
):
    """The registered cron source, with the seeded args, through the real
    scheduler tool lookup and the real tool: the row is resolved on disk."""
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    from kestrel_sovereign.signals.sources.scheduler import (
        build_cron_registrations,
        cron_source_name,
    )

    feature = await _feature(
        tmp_path,
        [{"id": "blk_3519", "issue": "#3519", "title": "morning signal gap",
          "severity": "high", "blocked_since": "2026-10-02"}],
    )
    agent = feature.agent
    agent.features = {"StrategicMemoryFeature": feature}
    agent.hooks_manager = None
    scheduler = SchedulerFeature(agent)
    [registration] = [
        r
        for r in build_cron_registrations(
            tool_lookup=scheduler._lookup_raw_tool_result,
            reason_codes_lookup=scheduler._declared_reason_codes,
        )
        if r.name == cron_source_name("strategy_reconcile_blockers")
    ]
    seeded = await _seeded_reconcile(agent)
    token, get = _patched(
        _FakeGitHub({(SELF_REPO, 3519): _issue(SELF_REPO, 3519, "closed")})
    )

    with token, get:
        outcome = await registration.handler(json.loads(seeded[1]))

    assert json.loads(outcome)["status"] == "ok"
    [row] = _ledger_on_disk(tmp_path)[BLOCKERS_KEY]
    assert row["resolved_at"] == str(date.today())
    assert f"{SELF_REPO}#3519 closed" in row["resolution"]


@pytest.mark.asyncio
async def test_a_scheduled_run_that_cannot_check_names_why(tmp_path):
    """A failed run must say which prerequisite was missing, in signal_log."""
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    from kestrel_sovereign.signals.sources.scheduler import (
        build_cron_registrations,
        cron_source_name,
    )

    feature = await _feature(tmp_path, [{"id": "blk_1", "issue": "#1", "title": "t"}])
    agent = feature.agent
    agent.features = {"StrategicMemoryFeature": feature}
    agent.hooks_manager = None
    scheduler = SchedulerFeature(agent)
    [registration] = [
        r
        for r in build_cron_registrations(
            tool_lookup=scheduler._lookup_raw_tool_result,
            reason_codes_lookup=scheduler._declared_reason_codes,
        )
        if r.name == cron_source_name("strategy_reconcile_blockers")
    ]

    with patch(f"{_MOD}.get_github_token", return_value=None):
        with pytest.raises(RuntimeError, match=r"failed \(NO_GITHUB_TOKEN\)"):
            await registration.handler({"apply": "yes"})

    recorded = _ledger_on_disk(tmp_path)[RECONCILIATION_KEY]
    assert recorded["ran"] is False
    assert "GITHUB_TOKEN" in recorded["reason"]


# ---------------------------------------------------------------------------
# The morning briefing keeps the gap visible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_briefing_reports_rows_the_last_run_could_not_check(tmp_path):
    feature = await _feature(
        tmp_path,
        [
            {"id": "blk_closed", "issue": "#3519", "title": "closed",
             "blocked_since": "2026-10-02"},
            {"id": "blk_open", "issue": "#3600", "title": "open one",
             "severity": "high", "blocked_since": "2026-10-02"},
            {"id": "blk_lost", "repo": "no-such-repo", "issue": "#9",
             "title": "lost"},
        ],
    )
    token, get = _patched(
        _FakeGitHub(
            {
                (SELF_REPO, 3519): _issue(SELF_REPO, 3519, "closed"),
                (SELF_REPO, 3600): _issue(SELF_REPO, 3600, "open"),
            }
        )
    )
    with token, get:
        await feature.strategy_reconcile_blockers(apply="yes")

    recorded = _ledger_on_disk(tmp_path)[RECONCILIATION_KEY]
    assert (recorded["checked"], recorded["resolved"], recorded["unresolvable"]) == (
        2, 1, 1
    )

    with patch(
        "kestrel_sovereign.features.strategic_memory.morning_signal."
        "fetch_github_signal",
        new=AsyncMock(return_value={}),
    ), patch(
        "kestrel_sovereign.features.strategic_memory.feature.assess_workflow_runs",
        new=AsyncMock(return_value=WorkflowRunsNotAssessed("not installed")),
    ):
        briefing = (await feature.morning_signal()).confirmation

    assert "#3600: open one" in briefing
    assert "#3519: closed" not in briefing, "a resolved row is not a blocker"
    assert (
        "2 checked, 1 resolved as closed, 1 could not be checked "
        "(run !strategy-reconcile to list them)."
    ) in briefing


def test_no_recorded_run_is_said_not_implied():
    line = describe_last_reconciliation(None, datetime.now(timezone.utc))
    assert "No blocker reconciliation has been recorded" in line


def test_a_stopped_schedule_is_called_out():
    now = datetime(2026, 10, 9, 8, 0, tzinfo=timezone.utc)
    summary = {
        "ran_at": (now - timedelta(days=3)).isoformat(),
        "ran": True,
        "checked": 4,
        "resolved": 0,
        "unresolvable": 0,
    }
    line = describe_last_reconciliation(summary, now)
    assert "2026-10-06 08:00 UTC" in line
    assert "more than a day old" in line
    fresh = describe_last_reconciliation(
        {**summary, "ran_at": (now - timedelta(minutes=30)).isoformat()}, now
    )
    assert "more than a day old" not in fresh


def test_a_run_that_could_not_check_anything_says_so():
    now = datetime(2026, 10, 9, 8, 0, tzinfo=timezone.utc)
    line = describe_last_reconciliation(
        {
            "ran_at": (now - timedelta(minutes=30)).isoformat(),
            "ran": False,
            "reason": "No GITHUB_TOKEN found",
        },
        now,
    )
    assert line.startswith("Blocker reconciliation could not run at 2026-10-09 07:30 UTC")
    assert "No GITHUB_TOKEN found" in line


@pytest.mark.asyncio
async def test_generate_morning_signal_renders_the_line_under_blockers():
    data = {
        "morning_signal_config": {"scan_repos": []},
        BLOCKERS_KEY: [{"issue": "#1", "title": "t", "severity": "low"}],
    }
    briefing = await generate_morning_signal(
        data, WorkflowRunsNotAssessed("not installed"), blocker_reconciliation=None
    )
    blockers = briefing.split("## Blockers", 1)[1].split("\n## ", 1)[0]
    assert "No blocker reconciliation has been recorded" in blockers


# ---------------------------------------------------------------------------
# Rows are normalized when written
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("issue", "repo", "stored_issue", "stored_repo"),
    [
        pytest.param("#3517", "self", "#3517", SELF_REPO, id="repo-self"),
        pytest.param(
            "kestrel-feature-talon#46",
            "kestrel-feature-talon",
            "KestrelSovereignAI/kestrel-feature-talon#46",
            "KestrelSovereignAI/kestrel-feature-talon",
            id="short-repo-and-short-issue",
        ),
        pytest.param(
            "kestrel-feature-talon#46",
            "",
            "KestrelSovereignAI/kestrel-feature-talon#46",
            "KestrelSovereignAI/kestrel-feature-talon",
            id="repo-hash-n",
        ),
        pytest.param(
            "self#3518", "", f"{SELF_REPO}#3518", SELF_REPO, id="self-hash-n"
        ),
        pytest.param(
            "KestrelSovereignAI/kestrel-feature-workflows#30",
            "",
            "KestrelSovereignAI/kestrel-feature-workflows#30",
            "KestrelSovereignAI/kestrel-feature-workflows",
            id="owner-repo-hash-n",
        ),
    ],
)
async def test_add_blocker_stores_the_normalized_reference(
    tmp_path, issue, repo, stored_issue, stored_repo
):
    feature = await _feature(tmp_path, [])

    result = await feature.strategy_add_blocker(issue=issue, title="t", repo=repo)

    assert result.status.value == "ok"
    [row] = _ledger_on_disk(tmp_path)[BLOCKERS_KEY]
    assert (row["issue"], row["repo"]) == (stored_issue, stored_repo)
    # What was written is exactly what the reconciler reads back.
    reference = resolve_blocker_reference(row, SCAN_REPOS, SELF_REPO)
    assert (reference.repo, reference.problem) == (stored_repo, None)


@pytest.mark.asyncio
async def test_add_blocker_still_refuses_to_assume_a_repository(tmp_path):
    """The caller is present: a bare number on a multi-repo agent is refused
    with the way to say which, rather than bound to an assumption."""
    feature = await _feature(tmp_path, [])

    result = await feature.strategy_add_blocker(issue="#46", title="t")

    assert result.status.value == "error"
    assert "repo='self'" in result.error
    assert _ledger_on_disk(tmp_path)[BLOCKERS_KEY] == []


@pytest.mark.asyncio
async def test_add_blocker_refuses_a_repo_that_is_not_one(tmp_path):
    feature = await _feature(tmp_path, [])

    result = await feature.strategy_add_blocker(
        issue="#46", title="t", repo="the talon repo"
    )

    assert result.status.value == "error"
    assert "not 'owner/repo'" in result.error


ALPHA = "owner/alpha"
BETA = "owner/beta"
TALON = "KestrelSovereignAI/kestrel-feature-talon"


async def _briefing_beside_live_blocked(rows, scan_repos, number=77):
    """The briefing when every scanned repo has a ``blocked`` issue ``number``.

    Each live issue is titled ``"<owner/repo> labelled"`` so a test can tell
    which repository's issue was listed.
    """
    data = {"morning_signal_config": {"scan_repos": scan_repos}, BLOCKERS_KEY: rows}
    live = {
        repo: {
            "issue_count": 1,
            "prs": [],
            "comments_24h": 0,
            "blocked_issues": [{"number": number, "title": f"{repo} labelled"}],
        }
        for repo in scan_repos
    }
    with patch(
        "kestrel_sovereign.features.strategic_memory.morning_signal."
        "fetch_github_signal",
        new=AsyncMock(return_value=live),
    ):
        return await generate_morning_signal(
            data, WorkflowRunsNotAssessed("n/a"), blocker_reconciliation=None
        )


def _listed_live_blockers(briefing):
    """The repositories whose live blocked issue the briefing listed."""
    return {
        line.rsplit(": ", 1)[1].removesuffix(" labelled")
        for line in briefing.splitlines()
        if line.startswith("- [GITHUB] ")
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "hidden"),
    [
        # A row states its issue: it hides that repository's issue, no other.
        pytest.param({"issue": "owner/alpha#77"}, {ALPHA}, id="owner-repo-hash-n"),
        pytest.param({"issue": "OWNER/Alpha#77"}, {ALPHA}, id="other-case"),
        pytest.param({"issue": "#77", "repo": ALPHA}, {ALPHA}, id="declared-repo"),
        pytest.param({"issue": "alpha#77"}, {ALPHA}, id="scanned-short-name"),
        pytest.param({"issue": "#77", "repo": "self"}, {SELF_REPO}, id="repo-self"),
        pytest.param({"issue": "self#77"}, {SELF_REPO}, id="self-hash-n"),
        pytest.param(
            {"issue": "kestrel-feature-talon#77", "repo": "kestrel-feature-talon"},
            {TALON},
            id="short-repo-and-short-issue",
        ),
        pytest.param({"issue": "owner/alpha#78"}, set(), id="other-number"),
        pytest.param({"issue": "owner/gamma#77"}, set(), id="unscanned-repo"),
        # A row whose repository is only inferred, or that names none at all,
        # hides nothing: a bare 77 on this agent is assumed to be home's 77.
        pytest.param({"issue": "#77"}, set(), id="bare-inferred"),
        pytest.param({"issue": 77}, set(), id="yaml-int-inferred"),
        pytest.param(
            {"issue": "#77", "repo": "the talon repo"}, set(), id="unresolvable"
        ),
    ],
)
async def test_a_ledger_row_hides_only_the_live_blocker_it_names(row, hidden):
    """Live blocked issues are matched against ledger rows by repository AND
    number. Matching by number alone let ``owner/alpha#77`` hide a different
    project's ``owner/beta#77`` from the briefing."""
    scan_repos = [*SCAN_REPOS, ALPHA, BETA]

    briefing = await _briefing_beside_live_blocked(
        [{**row, "title": "ledger row", "severity": "low"}], scan_repos
    )

    assert "ledger row" in briefing
    assert _listed_live_blockers(briefing) == set(scan_repos) - hidden


@pytest.mark.asyncio
async def test_an_inferred_row_does_not_hide_the_lone_scanned_repositorys_issue():
    """Even on a single-repository agent a bare number is an assumption -- the
    reconciler checks GitHub's dates before trusting it -- so the briefing
    lists the live issue too rather than risk hiding a different one."""
    briefing = await _briefing_beside_live_blocked(
        [{"issue": "#77", "title": "ledger row", "severity": "low"}], [SELF_REPO]
    )

    assert "ledger row" in briefing
    assert _listed_live_blockers(briefing) == {SELF_REPO}


def test_self_repo_comes_from_the_environment_then_dotenv_then_the_default(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.features.strategic_memory.github_integration import (
        DEFAULT_GITHUB_SELF_REPO,
        get_github_self_repo,
    )

    monkeypatch.chdir(tmp_path)
    assert get_github_self_repo() == SELF_REPO

    monkeypatch.delenv("GITHUB_SELF_REPO")
    assert get_github_self_repo() == DEFAULT_GITHUB_SELF_REPO

    # ``.env.example`` writes the setting with a trailing comment.
    (tmp_path / ".env").write_text(
        "GITHUB_SELF_REPO=Acme/home  # Agent's own source repository\n",
        encoding="utf-8",
    )
    assert get_github_self_repo() == "Acme/home"


# ---------------------------------------------------------------------------
# A row that names two repositories, or a short name several repositories
# have, is refused rather than read in one of them (#3540)
# ---------------------------------------------------------------------------


OTHER = "owner/other"


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"repo": "self", "issue": f"{OTHER}#46"}, id="self-vs-qualified"),
        pytest.param({"repo": "self", "issue": f"{OTHER} #46"}, id="spaced-reference"),
        pytest.param({"repo": OTHER, "issue": "self#46"}, id="qualified-vs-self"),
        pytest.param({"repo": OTHER, "issue": "talon#46"}, id="other-short-name"),
        pytest.param(
            {"repo": "kestrel-feature-talon", "issue": f"{OTHER}#46"},
            id="short-repo-vs-qualified",
        ),
        pytest.param(
            {"repo": "self", "issue": "blocked by owner/other#46"},
            id="repository-inside-prose",
        ),
        # No repo: the assumed home repository is not the one the prose names.
        pytest.param({"issue": "blocked by owner/other#46"}, id="inferred-vs-prose"),
    ],
)
def test_a_row_naming_two_repositories_is_refused(row):
    reference = resolve_blocker_reference(row, SCAN_REPOS, SELF_REPO)

    assert reference.problem == blocker_reconcile.CONFLICTING_REPOS
    assert reference.conflicting is not None
    assert not _issues_stated(row), "a refused row states no issue"


@pytest.mark.parametrize(
    ("row", "configured"),
    [
        pytest.param({"repo": OTHER, "issue": "other#46"}, SCAN_REPOS, id="unscanned"),
        pytest.param(
            {"repo": OTHER, "issue": "other#46"},
            [*SCAN_REPOS, OTHER, "else/other"],
            id="the-row-names-the-owner-a-short-name-would-not",
        ),
        pytest.param({"repo": OTHER, "issue": "OWNER/Other#46"}, SCAN_REPOS, id="case"),
        pytest.param({"repo": "self", "issue": "kestrel-sovereign#46"}, SCAN_REPOS,
                     id="self-and-its-name"),
        pytest.param({"repo": "self", "issue": f"{SELF_REPO}#46"}, SCAN_REPOS,
                     id="self-and-its-full-name"),
    ],
)
def test_a_row_whose_reference_names_its_own_repository_is_read(row, configured):
    reference = resolve_blocker_reference(row, configured, SELF_REPO)

    expected = SELF_REPO if row["repo"] == "self" else OTHER
    assert (reference.repo, reference.number, reference.problem) == (
        expected, 46, None
    )
    assert reference.source == REPO_FROM_ROW


ACME_WIDGETS = "Acme/widgets"
OTHER_WIDGETS = "Other/widgets"
WIDGET_REPOS = [ACME_WIDGETS, OTHER_WIDGETS]


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"issue": "widgets#46"}, id="short-name-in-issue"),
        pytest.param({"repo": "widgets", "issue": "#46"}, id="short-name-as-repo"),
        pytest.param({"repo": "Widgets", "issue": "widgets#46"}, id="both"),
    ],
)
def test_a_short_name_several_configured_repositories_have_is_refused(row):
    """The owner fallback (``Acme`` from ``GITHUB_SELF_REPO=Acme/home``) used
    to pick ``Acme/widgets`` and mark it stated by the row."""
    reference = resolve_blocker_reference(row, WIDGET_REPOS, "Acme/home")

    assert reference.repo is None
    assert reference.problem == blocker_reconcile.AMBIGUOUS_REPO_NAME
    assert reference.source is None, "never marked as stated by the row"
    assert reference.candidates == (ACME_WIDGETS, OTHER_WIDGETS)


def test_a_qualified_repo_settles_a_short_name_several_repositories_have():
    reference = resolve_blocker_reference(
        {"repo": OTHER_WIDGETS, "issue": "widgets#46"}, WIDGET_REPOS, "Acme/home"
    )
    assert (reference.repo, reference.problem) == (OTHER_WIDGETS, None)


def test_a_short_name_no_configured_repository_has_takes_the_home_owner():
    reference = resolve_blocker_reference(
        {"issue": "kestrel-feature-eye#4"}, SCAN_REPOS, SELF_REPO
    )
    assert (reference.repo, reference.problem, reference.source) == (
        "KestrelSovereignAI/kestrel-feature-eye", None, REPO_FROM_ISSUE
    )


def test_a_short_name_one_configured_repository_has_is_that_one():
    reference = resolve_blocker_reference(
        {"issue": "widgets#46"}, [ACME_WIDGETS, SELF_REPO], "Other/home"
    )
    assert (reference.repo, reference.problem, reference.source) == (
        ACME_WIDGETS, None, REPO_FROM_ISSUE
    )


def _issues_stated(row):
    from kestrel_sovereign.features.strategic_memory.morning_signal import (
        _issues_stated_by_ledger,
    )

    return _issues_stated_by_ledger([row], SCAN_REPOS, SELF_REPO)


@pytest.mark.asyncio
async def test_apply_does_not_retire_a_blocker_on_the_declared_repositorys_issue(
    tmp_path,
):
    """``repo: self`` + ``issue: owner/other#46``: home's 46 is closed and
    ``owner/other#46`` is open. Neither is the row's issue to resolve on."""
    feature = await _feature(
        tmp_path,
        [{"id": "blk_two", "repo": "self", "issue": f"{OTHER}#46", "title": "t",
          "blocked_since": "2026-10-01"}],
    )
    github = _FakeGitHub(
        {
            (SELF_REPO, 46): _issue(SELF_REPO, 46, "closed"),
            (OTHER, 46): _issue(OTHER, 46, "open"),
        }
    )
    token, get = _patched(github)

    with token, get:
        result = await feature.strategy_reconcile_blockers(apply="yes")

    report = result.data["report"]
    assert report["closed"] == [] and report["open"] == []
    [entry] = report["unresolvable"]
    assert entry["reason"] == blocker_reconcile.CONFLICTING_REPOS
    assert (entry["repo"], entry["conflicting_repo"]) == (SELF_REPO, OTHER)
    assert github.paths == [], "a refused row is not looked up"
    assert (
        f"its issue reference names {OTHER}, but the row is read in {SELF_REPO}"
    ) in result.confirmation
    assert not _ledger_on_disk(tmp_path)[BLOCKERS_KEY][0].get("resolved_at")
    assert _ledger_on_disk(tmp_path)[RECONCILIATION_KEY]["unresolvable"] == 1


@pytest.mark.asyncio
async def test_apply_does_not_retire_an_ambiguous_short_name(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SELF_REPO", "Acme/home")
    feature = await _feature(
        tmp_path,
        [{"id": "blk_w", "issue": "widgets#46", "title": "t",
          "blocked_since": "2026-10-01"}],
        scan_repos=WIDGET_REPOS,
    )
    github = _FakeGitHub(
        {
            (ACME_WIDGETS, 46): _issue(ACME_WIDGETS, 46, "closed"),
            (OTHER_WIDGETS, 46): _issue(OTHER_WIDGETS, 46, "open"),
        }
    )
    token, get = _patched(github)

    with token, get:
        result = await feature.strategy_reconcile_blockers(apply="yes")

    [entry] = result.data["report"]["unresolvable"]
    assert entry["reason"] == blocker_reconcile.AMBIGUOUS_REPO_NAME
    assert entry["candidate_repos"] == WIDGET_REPOS
    assert github.paths == []
    assert f"could be any of {ACME_WIDGETS}, {OTHER_WIDGETS}" in result.confirmation
    assert not _ledger_on_disk(tmp_path)[BLOCKERS_KEY][0].get("resolved_at")


@pytest.mark.asyncio
async def test_apply_resolves_a_short_reference_in_the_declared_repository(tmp_path):
    feature = await _feature(
        tmp_path,
        [{"id": "blk_o", "repo": OTHER, "issue": "other#46", "title": "t",
          "blocked_since": "2026-10-01"}],
    )
    github = _FakeGitHub({(OTHER, 46): _issue(OTHER, 46, "closed")})
    token, get = _patched(github)

    with token, get:
        await feature.strategy_reconcile_blockers(apply="yes")

    assert github.paths == [f"/repos/{OTHER}/issues/46"]
    [row] = _ledger_on_disk(tmp_path)[BLOCKERS_KEY]
    assert row["resolution"].startswith(f"GitHub reports {OTHER}#46 closed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("issue", "repo", "scan_repos", "message"),
    [
        pytest.param(
            f"{OTHER}#46", "self", SCAN_REPOS,
            f"names {OTHER}, but the row's repository is {SELF_REPO}",
            id="two-repositories",
        ),
        pytest.param(
            "widgets#46", "", WIDGET_REPOS,
            f"called 'widgets' ({ACME_WIDGETS}, {OTHER_WIDGETS})",
            id="ambiguous-short-issue",
        ),
        pytest.param(
            "#46", "widgets", WIDGET_REPOS,
            f"called 'widgets' ({ACME_WIDGETS}, {OTHER_WIDGETS})",
            id="ambiguous-short-repo",
        ),
    ],
)
async def test_add_blocker_refuses_a_row_reconcile_would_refuse(
    tmp_path, issue, repo, scan_repos, message
):
    feature = await _feature(tmp_path, [], scan_repos=scan_repos)

    result = await feature.strategy_add_blocker(issue=issue, title="t", repo=repo)

    assert result.status.value == "error"
    assert message in result.error
    assert _ledger_on_disk(tmp_path)[BLOCKERS_KEY] == []


@pytest.mark.asyncio
async def test_add_blocker_qualifies_a_short_reference_with_the_declared_owner(
    tmp_path,
):
    feature = await _feature(tmp_path, [])

    result = await feature.strategy_add_blocker(
        issue="other#46", title="t", repo=OTHER
    )

    assert result.status.value == "ok"
    [row] = _ledger_on_disk(tmp_path)[BLOCKERS_KEY]
    assert (row["issue"], row["repo"]) == (f"{OTHER}#46", OTHER)


@pytest.mark.parametrize(
    ("issue", "repo", "self_repo", "expected"),
    [
        ("o/talon#5", "o/core", "o/core", "o/talon"),
        ("talon#5", "o/core", "o/core", "talon"),
        ("blocked by other/core#5", "o/core", "o/core", "other/core"),
        ("self#5", "o/core", "o/home", "self"),
        ("self#5", "o/core", "o/core", None),
        ("O/Core#5", "o/core", "o/home", None),
        ("core#5", "o/core", "o/home", None),
        ("#5", "o/core", "o/home", None),
        ("Issue #5", "o/core", "o/home", None),
    ],
)
def test_the_shared_conflict_rule(issue, repo, self_repo, expected):
    """One function decides for the reconciler and for dispatch."""
    assert blocker_reconcile.issue_repository_conflict(issue, repo, self_repo) == (
        expected
    )

