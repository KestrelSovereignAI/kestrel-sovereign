"""ToolResult contract tests for StrategicMemoryFeature (#1061 wave 12).

Pins the honesty edges introduced by the migration:
  - strategy_view 'vision' falls back to placeholder when the field is
    null/empty (not just missing) — ToolResult.ok rejects empty
    confirmations
  - strategy_view unknown section -> ERROR
  - strategy_resolve_blocker on missing issue -> ERROR
  - backlog_hygiene with prereq failures -> ERROR
  - backlog_hygiene fix='no' / 'yes' -> PARTIAL / OK matching the
    runner's truthy predicate (yes/true/1)
  - session_log with prereq failures -> ERROR
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pathlib import Path

from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature
from kestrel_sovereign.features.strategic_memory.feature import _SaveOutcome
from kestrel_sovereign.features.strategic_memory.ledger import (
    BLOCKERS_KEY,
    PATTERNS_KEY,
    StrategyLedger,
)


def _make_feature(
    data: dict | None = None,
    *,
    agent=None,
    blockers: list | None = None,
    patterns: list | None = None,
) -> StrategicMemoryFeature:
    feat = StrategicMemoryFeature(agent=agent if agent is not None else MagicMock())
    feat._data = data if data is not None else {}
    # A strategy path IS configured; ``_save`` is stubbed to report a
    # successful persist so happy-path mutating tools return OK. Tests
    # exercising the no-path / write-failure edges (F291) override these.
    feat._strategy_path = Path("/tmp/kestrel-test/STRATEGY.yaml")
    feat._save = MagicMock(return_value=_SaveOutcome(persisted=True))
    # Blockers and patterns live in STRATEGY_LEDGER.yaml (#2954), which has
    # its own path and its own write. Stub it the same way.
    feat._ledger = StrategyLedger(Path("/tmp/kestrel-test/STRATEGY_LEDGER.yaml"))
    feat._ledger.data[BLOCKERS_KEY] = list(blockers or [])
    feat._ledger.data[PATTERNS_KEY] = list(patterns or [])
    feat._ledger.normalize()
    feat._ledger.save = MagicMock(return_value=None)
    return feat


# ---------------------------------------------------------------------------
# strategy_view
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_strategy_view_no_data_returns_error():
    feat = _make_feature({})
    result = await feat.strategy_view()
    assert result.status is ToolResultStatus.ERROR


@pytest.mark.asyncio
async def test_strategy_view_unknown_section_returns_error():
    feat = _make_feature({"vision": "ok"})
    result = await feat.strategy_view(section="garbage")
    assert result.status is ToolResultStatus.ERROR
    assert "garbage" in result.error


@pytest.mark.asyncio
async def test_strategy_view_vision_falls_back_when_empty():
    """STRATEGY.yaml may have ``vision:`` (null) or ``vision: ""``.

    ToolResult.ok requires a non-empty confirmation, so the renderer
    must fall back to the placeholder text on falsy values, not just
    missing keys.
    """
    for empty_vision in (None, ""):
        feat = _make_feature({"vision": empty_vision})
        result = await feat.strategy_view(section="vision")
        assert result.status is ToolResultStatus.OK
        assert "No vision defined" in result.confirmation


@pytest.mark.asyncio
async def test_strategy_view_vision_present():
    feat = _make_feature({"vision": "Build the best agent"})
    result = await feat.strategy_view(section="vision")
    assert result.status is ToolResultStatus.OK
    assert result.confirmation == "Build the best agent"


# ---------------------------------------------------------------------------
# strategy_resolve_blocker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_blocker_missing_returns_error():
    feat = _make_feature(blockers=[{"issue": "OTHER"}])
    result = await feat.strategy_resolve_blocker(issue="GONE")
    assert result.status is ToolResultStatus.ERROR
    assert "GONE" in result.error


@pytest.mark.asyncio
async def test_resolve_blocker_present_returns_ok():
    feat = _make_feature(blockers=[{"issue": "X-1", "title": "fix"}])
    result = await feat.strategy_resolve_blocker(issue="X-1")
    assert result.status is ToolResultStatus.OK
    assert result.data["removed_count"] == 1


# ---------------------------------------------------------------------------
# backlog_hygiene
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backlog_hygiene_prereq_failure_returns_error():
    feat = _make_feature({})
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.run_backlog_hygiene",
        new=AsyncMock(return_value="No GITHUB_TOKEN found. Set GITHUB_TOKEN ..."),
    ):
        result = await feat.backlog_hygiene(fix="yes")
    assert result.status is ToolResultStatus.ERROR
    assert "GITHUB_TOKEN" in result.error
    assert result.data["applied"] is False


@pytest.mark.asyncio
async def test_backlog_hygiene_dry_run_returns_partial():
    feat = _make_feature({})
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.run_backlog_hygiene",
        new=AsyncMock(return_value="# Backlog Hygiene Report -- ...\nclean."),
    ):
        result = await feat.backlog_hygiene(fix="no")
    assert result.status is ToolResultStatus.PARTIAL
    assert "report-only" in result.error
    assert result.data["applied"] is False


@pytest.mark.asyncio
async def test_backlog_hygiene_truthy_aliases_apply():
    """The runner accepts 'yes', 'true', '1' as auto-fix; the wrapper
    must agree so the envelope doesn't contradict the side effects."""
    for truthy in ("yes", "true", "1", "YES", "True"):
        feat = _make_feature({})
        with patch(
            "kestrel_sovereign.features.strategic_memory.feature.run_backlog_hygiene",
            new=AsyncMock(return_value="# Hygiene\nfixes applied."),
        ):
            result = await feat.backlog_hygiene(fix=truthy)
        assert result.status is ToolResultStatus.OK, f"fix={truthy!r}"
        assert result.data["applied"] is True


# ---------------------------------------------------------------------------
# session_log
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_log_prereq_failure_returns_error():
    feat = _make_feature({})
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.collect_session_log",
        new=AsyncMock(return_value="No scan_repos configured in morning_signal_config."),
    ):
        result = await feat.session_log()
    assert result.status is ToolResultStatus.ERROR
    assert "scan_repos" in result.error


@pytest.mark.asyncio
async def test_session_log_success():
    feat = _make_feature({})
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.collect_session_log",
        new=AsyncMock(return_value="# Session Log\n..."),
    ):
        result = await feat.session_log(session_id="020", focus="testing")
    assert result.status is ToolResultStatus.OK
    assert result.data["session_id"] == "020"




def test_strategic_memory_exposes_provider_neutral_dispatch():
    feature = _make_feature({})
    tool_names = {tool.name for tool in feature.get_tools()}
    assert "signal_dispatch" in tool_names


_TOP_ISSUE = {
    "repo": "owner/repo",
    "issue_number": 42,
    "issue_title": "Repair the boundary",
    "priority": "high",
    "context": "Milestone: extraction",
}


def _agent_ready_issue(number, title, repo="o/r"):
    """An open issue as GitHub's REST API returns it, on the allow-list."""
    return {
        "number": number,
        "title": title,
        "state": "open",
        "repository_url": f"https://api.github.com/repos/{repo}",
        "labels": [{"name": "agent-ready"}],
    }


def _dispatch_agent(*, registration=None, runner_result=None):
    operator_registry = SimpleNamespace(
        get_workflow_registration=lambda name: registration
    )
    execute_named_tool = AsyncMock(
        return_value=(
            runner_result
            if runner_result is not None
            else ToolResult.ok(
                "started",
                data={"run_id": "run-42", "status": "pending"},
            )
        )
    )
    return SimpleNamespace(
        operator_registry=operator_registry,
        execute_named_tool=execute_named_tool,
        get_turn_bound_session_id=lambda: "chat-7",
    )


@pytest.mark.asyncio
async def test_signal_dispatch_suggest_never_requires_or_runs_capability():
    agent = _dispatch_agent(registration=None)
    feat = _make_feature({}, agent=agent)
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=AsyncMock(return_value=_TOP_ISSUE),
    ):
        result = await feat.signal_dispatch(mode="preview")

    assert result.status is ToolResultStatus.OK
    assert result.data["mode"] == "suggest"
    assert result.data["dispatched"] is False
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_dispatch_fails_closed_when_capability_is_absent():
    agent = _dispatch_agent(registration=None)
    feat = _make_feature({}, agent=agent)
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=AsyncMock(return_value=_TOP_ISSUE),
    ):
        result = await feat.signal_dispatch()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "DISPATCH_CAPABILITY_UNAVAILABLE"
    assert result.data["dispatched"] is False
    assert "Install and enable a feature" in result.error
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_dispatch_routes_contributed_workflow_through_governed_runner():
    registration = SimpleNamespace(owner="feature:fixture-dispatch")
    agent = _dispatch_agent(registration=registration)
    feat = _make_feature({}, agent=agent)
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=AsyncMock(return_value=_TOP_ISSUE),
    ):
        result = await feat.signal_dispatch()

    assert result.status is ToolResultStatus.OK
    assert result.data["workflow_run_id"] == "run-42"
    assert result.data["capability_owner"] == "feature:fixture-dispatch"
    assert result.data["dispatched"] is True
    agent.execute_named_tool.assert_awaited_once_with(
        "workflow_run",
        {
            "name": "fleet_coding_pipeline",
            "params": {
                "repo": "owner/repo",
                "issue": 42,
                "issue_title": "Repair the boundary",
                "priority": "high",
                "context": "Milestone: extraction",
            },
        },
        session_id="chat-7",
        source="strategic_memory.signal_dispatch",
    )


@pytest.mark.asyncio
async def test_signal_dispatch_surfaces_runner_rejection_as_error():
    registration = SimpleNamespace(owner="feature:fixture-dispatch")
    agent = _dispatch_agent(
        registration=registration,
        runner_result=ToolResult.failed("definition is not loaded"),
    )
    feat = _make_feature({}, agent=agent)
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=AsyncMock(return_value=_TOP_ISSUE),
    ):
        result = await feat.signal_dispatch()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORKFLOW_RUN_REJECTED"
    assert result.data["dispatched"] is False
    assert "definition is not loaded" in result.error


@pytest.mark.asyncio
async def test_signal_dispatch_identifies_missing_governed_runner_by_public_error():
    registration = SimpleNamespace(owner="feature:fixture-dispatch")
    agent = _dispatch_agent(registration=registration)
    agent.execute_named_tool.side_effect = ToolNotRegisteredError(
        "workflow_run is not registered with any enabled feature"
    )
    feat = _make_feature({}, agent=agent)
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=AsyncMock(return_value=_TOP_ISSUE),
    ):
        result = await feat.signal_dispatch()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORKFLOW_RUNNER_UNAVAILABLE"
    assert result.data["dispatched"] is False


@pytest.mark.asyncio
async def test_signal_dispatch_does_not_misclassify_provider_value_error():
    """A registered runner's own validation failure is not tool absence."""

    registration = SimpleNamespace(owner="feature:fixture-dispatch")
    agent = _dispatch_agent(registration=registration)
    agent.execute_named_tool.side_effect = ValueError("invalid workflow params")
    feat = _make_feature({}, agent=agent)
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=AsyncMock(return_value=_TOP_ISSUE),
    ):
        result = await feat.signal_dispatch()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORKFLOW_RUNNER_FAILED"
    assert result.data["dispatched"] is False


@pytest.mark.asyncio
async def test_signal_dispatch_invalid_mode_never_selects_or_dispatches():
    agent = _dispatch_agent(registration=SimpleNamespace(owner="feature:x"))
    feat = _make_feature({}, agent=agent)
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=AsyncMock(),
    ) as select:
        result = await feat.signal_dispatch(mode="run")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["dispatched"] is False
    select.assert_not_awaited()
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_strategy_add_blocker_invalid_severity_rejected():
    feat = _make_feature({})
    result = await feat.strategy_add_blocker(issue="42", title="x", severity="sev1")
    assert result.status is ToolResultStatus.ERROR
    assert "Must be one of: low, medium, high, critical" in result.error
    # Nothing persisted on rejection.
    assert not feat._ledger.blockers


@pytest.mark.asyncio
async def test_strategy_add_blocker_severity_normalized():
    feat = _make_feature({})
    result = await feat.strategy_add_blocker(issue="42", title="x", severity="HIGH")
    assert result.status is ToolResultStatus.OK
    assert feat._ledger.blockers[-1]["severity"] == "high"


# ---------------------------------------------------------------------------
# F291: mutating tools must not report OK when _save silently no-ops
# ---------------------------------------------------------------------------


def _mutating_calls(feat):
    """Every mutating tool paired with a coroutine factory + happy label."""
    return [
        ("strategy_add_decision", lambda: feat.strategy_add_decision("d", "r")),
        ("strategy_add_blocker", lambda: feat.strategy_add_blocker("42", "t")),
        ("strategy_add_pattern", lambda: feat.strategy_add_pattern("p")),
        (
            "strategy_resolve_blocker",
            lambda: feat.strategy_resolve_blocker("42"),
        ),
    ]


@pytest.mark.asyncio
async def test_mutating_tools_no_strategy_path_return_error():
    """No strategy path configured -> feature is not active. Persisting is
    impossible, so a mutating tool must ERROR, never report OK."""
    for name, _ in _mutating_calls(None):
        feat = _make_feature(blockers=[{"issue": "42", "title": "x"}])
        # No path AND the real _save so it detects the no-path condition.
        feat._strategy_path = None
        del feat._save  # drop the stub; use the real method
        # Same for the ledger: no path means nothing could be persisted.
        feat._ledger.path = None
        del feat._ledger.save
        call = dict(_mutating_calls(feat))[name]
        result = await call()
        assert result.status is ToolResultStatus.ERROR, name
        assert result.data["persisted"] is False, name


@pytest.mark.asyncio
async def test_mutating_tools_write_failure_return_partial():
    """When the write raises, the in-memory update stands but nothing was
    persisted -> PARTIAL with the error surfaced, never OK."""
    for name, _ in _mutating_calls(None):
        feat = _make_feature(blockers=[{"issue": "42", "title": "x"}])
        feat._save = MagicMock(
            return_value=_SaveOutcome(persisted=False, error="disk full")
        )
        feat._ledger.save = MagicMock(return_value="disk full")
        call = dict(_mutating_calls(feat))[name]
        result = await call()
        assert result.status is ToolResultStatus.PARTIAL, name
        assert "disk full" in result.error, name
        assert result.data["persisted"] is False, name


@pytest.mark.asyncio
async def test_mutating_tools_happy_path_return_ok():
    """Path present + write succeeds -> OK, persisted flag true."""
    for name, _ in _mutating_calls(None):
        feat = _make_feature(blockers=[{"issue": "42", "title": "x"}])
        call = dict(_mutating_calls(feat))[name]
        result = await call()
        assert result.status is ToolResultStatus.OK, name
        assert result.data["persisted"] is True, name


def test_save_no_path_reports_no_op():
    """_save must return a truthful outcome, not silently no-op (F291)."""
    feat = StrategicMemoryFeature(agent=MagicMock())
    feat._data = {"decisions": []}
    feat._strategy_path = None
    outcome = feat._save()
    assert outcome.persisted is False
    assert outcome.no_path is True
    assert outcome.error


def test_save_write_error_is_reported(tmp_path):
    """A write exception is surfaced in the outcome, not swallowed."""
    feat = StrategicMemoryFeature(agent=MagicMock())
    feat._data = {"decisions": []}
    feat._strategy_path = tmp_path / "STRATEGY.yaml"
    with patch.object(
        Path, "write_text", side_effect=OSError("no space left on device")
    ):
        outcome = feat._save()
    assert outcome.persisted is False
    assert outcome.no_path is False
    assert "no space left on device" in outcome.error


def test_save_happy_path_persists(tmp_path):
    feat = StrategicMemoryFeature(agent=MagicMock())
    feat._data = {"decisions": [{"decision": "ship"}]}
    feat._strategy_path = tmp_path / "STRATEGY.yaml"
    outcome = feat._save()
    assert outcome.persisted is True
    assert feat._strategy_path.exists()


# ---------------------------------------------------------------------------
# Contract: every @tool annotated -> ToolResult
# ---------------------------------------------------------------------------

def test_strategic_memory_passes_toolresult_contract():
    from kestrel_sovereign.tools.result_contract import (
        assert_feature_returns_tool_result,
    )

    feat = StrategicMemoryFeature(agent=MagicMock())
    assert_feature_returns_tool_result(feat)


@pytest.mark.asyncio
async def test_signal_dispatch_invalid_mode_names_a_reason_code():
    """The sixth failed return of the tool #3184 is about. The other five
    carry a reason_code; a scheduled row whose args_json holds a bad mode
    failed with none, so its dispatch failure read as cause-free."""
    feat = _make_feature({}, agent=_dispatch_agent(registration=None))

    result = await feat.signal_dispatch(mode="bogus")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "INVALID_DISPATCH_MODE"
    assert result.data["dispatched"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_signal_dispatch_does_not_call_an_unreachable_github_nothing_to_do(mode):
    """#3280 review. Selection now asks GitHub before dispatching a blocker, so
    it can fail on a network fault where it never could before. When every
    blocker it tried came back unreadable, "No actionable issue found" would be
    a claim about a ledger nobody looked at -- the same lie blocker_reconcile
    is written to avoid."""
    agent = _dispatch_agent(registration=None)
    feat = _make_feature({}, agent=agent)

    async def unreachable(view, diagnostics=None, run_history=None):
        diagnostics.update(blockers_checked=3, blockers_unreadable=3)
        return None

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=unreachable,
    ):
        result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "BLOCKERS_UNCONFIRMED"
    assert result.data["dispatched"] is False
    assert "No actionable issue found" not in result.confirmation
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_dispatch_still_says_nothing_to_do_when_github_answered():
    """The control: blockers GitHub answered for -- closed, say -- are a real
    answer, and an empty result then really is nothing to do."""
    agent = _dispatch_agent(registration=None)
    feat = _make_feature({}, agent=agent)

    async def all_closed(view, diagnostics=None, run_history=None):
        diagnostics.update(blockers_checked=3, blockers_unreadable=0)
        return None

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=all_closed,
    ):
        result = await feat.signal_dispatch()

    assert result.status is ToolResultStatus.OK
    assert "No actionable issue found" in result.confirmation


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_signal_dispatch_does_not_call_unreadable_candidate_linkage_nothing_to_do(
    mode, monkeypatch
):
    """#3367 acceptance, through the real selector: every backlog candidate's
    PR-linkage read returns None. Nothing may be dispatched, and an outage
    must not render as "No actionable issue found" with an OK status."""
    from kestrel_sovereign.features.strategic_memory import issue_selection

    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")

    async def fake_get(path, token):
        if path == (
            "/repos/o/r/issues?state=open&labels=agent-ready&per_page=5&sort=updated"
        ):
            return [
                _agent_ready_issue(1, "a"),
                _agent_ready_issue(2, "b"),
            ]
        raise RuntimeError(f"404 {path}")

    async def unreadable_linkage(path, token, body):
        return None

    monkeypatch.setattr(issue_selection, "github_api_get", fake_get)
    monkeypatch.setattr(issue_selection, "github_api_post", unreadable_linkage)
    agent = _dispatch_agent(registration=SimpleNamespace(owner="feature:x"))
    feat = _make_feature(
        {"morning_signal_config": {"scan_repos": ["o/r"]}}, agent=agent
    )

    result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "CANDIDATES_UNCONFIRMED"
    assert result.data["candidates_checked"] == 2
    assert result.data["candidates_unreadable"] == 2
    assert result.data["dispatched"] is False
    assert "No actionable issue found" not in result.confirmation
    assert "could not say" in result.confirmation
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_dispatch_one_unreadable_candidate_is_still_unconfirmed():
    """Candidates GitHub answered for (open PRs) plus one it could not: the
    unreadable one is why nothing was picked, so the result is not OK."""
    agent = _dispatch_agent(registration=None)
    feat = _make_feature({}, agent=agent)

    async def mixed(view, diagnostics=None, run_history=None):
        diagnostics.update(
            blockers_checked=0, candidates_checked=3, candidates_unreadable=1,
        )
        return None

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=mixed,
    ):
        result = await feat.signal_dispatch()

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "CANDIDATES_UNCONFIRMED"
    assert "1 of 3" in result.confirmation


@pytest.mark.asyncio
async def test_signal_dispatch_says_nothing_to_do_when_candidate_linkage_answered():
    """The control: every candidate checked, GitHub answered for each (they
    are in flight), so the empty result really is nothing to do."""
    agent = _dispatch_agent(registration=None)
    feat = _make_feature({}, agent=agent)

    async def all_in_flight(view, diagnostics=None, run_history=None):
        diagnostics.update(candidates_checked=2, candidates_unreadable=0)
        return None

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=all_in_flight,
    ):
        result = await feat.signal_dispatch()

    assert result.status is ToolResultStatus.OK
    assert "No actionable issue found" in result.confirmation


_IN_FLIGHT = {
    "repo": "o/r",
    "issue_number": 3310,
    "reason": "open_pr",
    "pull_requests": [{"repo": "o/r", "number": 3311, "draft": False, "days_idle": 0}],
    "stalled_after_days": 3,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_signal_dispatch_says_why_an_in_flight_issue_was_not_selected(mode):
    """#3317 acceptance: a board whose only issue has an open linked PR selects
    nothing, and the output names the skip -- "skipped o/r#3310 -- PR #3311
    open" -- rather than leaving an orchestrator to infer it from silence."""
    agent = _dispatch_agent(registration=SimpleNamespace(owner="feature:x"))
    feat = _make_feature({}, agent=agent)

    async def only_in_flight(view, diagnostics=None, run_history=None):
        diagnostics.update(
            blockers_checked=1, blockers_unreadable=0,
            open_pr_exclusions=[dict(_IN_FLIGHT)],
        )
        return None

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=only_in_flight,
    ):
        result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    assert result.data["skipped"] == [_IN_FLIGHT]
    assert "No actionable issue found." in result.confirmation
    assert "skipped o/r#3310 -- PR #3311 open" in result.confirmation
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_dispatch_reports_skips_alongside_the_issue_it_dispatched():
    registration = SimpleNamespace(owner="feature:fixture-dispatch")
    agent = _dispatch_agent(registration=registration)
    feat = _make_feature({}, agent=agent)

    async def skip_then_pick(view, diagnostics=None, run_history=None):
        diagnostics.update(open_pr_exclusions=[dict(_IN_FLIGHT)])
        return dict(_TOP_ISSUE)

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=skip_then_pick,
    ):
        result = await feat.signal_dispatch()

    assert result.data["dispatched"] is True
    assert result.data["skipped"] == [_IN_FLIGHT]
    assert "skipped o/r#3310 -- PR #3311 open" in result.confirmation


@pytest.mark.asyncio
async def test_signal_dispatch_without_skips_reports_an_empty_list():
    agent = _dispatch_agent(registration=None)
    feat = _make_feature({}, agent=agent)
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue",
        new=AsyncMock(return_value=_TOP_ISSUE),
    ):
        result = await feat.signal_dispatch(mode="suggest")

    assert result.data["skipped"] == []
    assert "Skipped" not in result.confirmation


# ---------------------------------------------------------------------------
# #3398: the pick reads Talon's run history, and says so when it cannot
# ---------------------------------------------------------------------------


class _TalonJobs:
    """A ``talon`` wait provider reporting its finished runs read-only.

    On the real provider ``poll()`` reaps jobs, pushes preserved work to the
    remote and rewrites the registry, and ``active_handles()`` cannot say
    whether it read the whole registry. A dispatch, and above all a
    ``mode='suggest'`` preview, must use neither. Each records the call in
    ``effects`` before failing, so a test can assert none happened even where
    the failure itself is caught.
    """

    kind = "talon"
    signal = None

    def __init__(self, runs, complete=True, reason="", error=None):
        self._report = {"complete": complete, "runs": list(runs), "reason": reason}
        self._error = error
        self.reads = 0
        self.effects = []

    async def finished_runs(self):
        self.reads += 1
        if self._error is not None:
            raise self._error
        return self._report

    async def active_handles(self):
        self.effects.append("active_handles")
        raise AssertionError("signal_dispatch must not enumerate through active_handles()")

    async def poll(self, handle):
        self.effects.extend([f"reap:{handle}", f"push:{handle}", "persist"])
        raise AssertionError("signal_dispatch must not poll: poll() reaps, pushes and writes")


class _SilentlyUnreadableTalon:
    """The Talon provider as it stands without ``finished_runs()``, over a
    corrupt ``jobs.json``: the reload logs the failure and returns, so
    ``active_handles()`` answers ``[]``; ``poll()`` would reap, push and
    persist."""

    kind = "talon"
    signal = None

    def __init__(self):
        self.effects = []

    async def active_handles(self):
        self.effects.append("active_handles")
        return []

    async def poll(self, handle):
        self.effects.extend([f"reap:{handle}", f"push:{handle}", "persist"])
        raise AssertionError("signal_dispatch must not poll: poll() reaps, pushes and writes")


def _with_talon(agent, provider):
    from kestrel_sovereign.waits.engine import WaitRegistry

    agent.wait_registry = WaitRegistry()
    agent.wait_registry.register(provider)
    return agent


def _talon_run(disposition="clarifying", job_id="9a63c66b2b71"):
    """One ``finished_runs()`` entry: #3093's last run, as Talon records it."""
    return {
        "job_id": job_id,
        "repo": "o/r",
        "issue": 3093,
        "completed_at": "2026-09-24T08:41:00+00:00",
        "disposition": disposition,
    }


def _github_with_open_3093(monkeypatch):
    """GitHub as the selector sees it: #3093 open, labelled ``agent-ready``
    and no Talon state label, no PR on it, and nothing on the issue newer
    than its last Talon run."""
    from kestrel_sovereign.features.strategic_memory import issue_selection

    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")

    async def fake_get(path, token):
        if path == "/repos/o/r/issues/3093":
            return _agent_ready_issue(3093, "epic")
        if "/issues?" in path:
            return []
        raise RuntimeError(f"404 {path}")

    async def fake_post(path, token, body):
        if "timelineItems" in body["query"]:
            return {"data": {"repository": {"issue": {
                "lastEditedAt": "2026-08-24T10:00:00Z",
                "timelineItems": {"nodes": []},
            }}}}
        return {"data": {"repository": {"issue": {
            "closedByPullRequestsReferences": {"nodes": []},
        }}}}

    monkeypatch.setattr(issue_selection, "github_api_get", fake_get)
    monkeypatch.setattr(issue_selection, "github_api_post", fake_post)


def _feature_with_3093_blocker(agent):
    return _make_feature(
        {"morning_signal_config": {"scan_repos": ["o/r"]}},
        agent=agent,
        blockers=[{"severity": "high", "issue": "o/r#3093", "title": "epic"}],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_signal_dispatch_does_not_select_without_talons_run_history(mode):
    """Selecting without the history could dispatch the very run that is
    waiting on an answer. That is not "nothing to do" either."""
    agent = _with_talon(
        _dispatch_agent(registration=SimpleNamespace(owner="feature:x")),
        _TalonJobs([], error=RuntimeError("jobs.json unreadable")),
    )
    feat = _make_feature({}, agent=agent)
    pick = AsyncMock(return_value=_TOP_ISSUE)

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue", new=pick
    ):
        result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "RUN_HISTORY_UNCONFIRMED"
    assert result.data["dispatched"] is False
    assert result.data["requirement"] is None
    assert "jobs.json unreadable" in result.confirmation
    assert "kestrel-feature-talon>=" not in result.confirmation
    assert "No actionable issue found" not in result.confirmation
    pick.assert_not_awaited()
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_signal_dispatch_names_the_talon_release_it_needs(mode):
    """#3446: on kestrel-feature-talon <=0.2.9 the ``talon`` provider has no
    ``finished_runs()``, so every dispatch is refused even over a healthy,
    empty registry. The refusal names the release that fixes it, in the text
    and in the data an orchestrator reads, and still dispatches nothing."""
    provider = _SilentlyUnreadableTalon()
    agent = _with_talon(
        _dispatch_agent(registration=SimpleNamespace(owner="feature:x")), provider
    )
    feat = _make_feature({}, agent=agent)
    pick = AsyncMock(return_value=_TOP_ISSUE)

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue", new=pick
    ):
        result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "RUN_HISTORY_UNCONFIRMED"
    assert result.data["requirement"] == "kestrel-feature-talon>=0.2.10"
    assert result.data["dispatched"] is False
    assert "kestrel-feature-talon>=0.2.10" in result.confirmation
    assert "kestrel-feature-talon>=0.2.10" in result.error
    assert provider.effects == []
    pick.assert_not_awaited()
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_signal_dispatch_does_not_dispatch_over_a_silently_unreadable_registry(
    mode, monkeypatch
):
    """Review P1 on #3432: a corrupt ``jobs.json`` makes the provider's
    enumeration answer ``[]`` without raising. Read as a history, that is "no
    previous runs" and #3093 is dispatched again. Through the real selector:
    the history is unconfirmed, nothing is selected or dispatched, and
    nothing on the provider is touched."""
    _github_with_open_3093(monkeypatch)
    provider = _SilentlyUnreadableTalon()
    agent = _with_talon(
        _dispatch_agent(registration=SimpleNamespace(owner="feature:x")), provider
    )
    feat = _feature_with_3093_blocker(agent)

    result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "RUN_HISTORY_UNCONFIRMED"
    assert result.data["dispatched"] is False
    assert result.data["issue"] is None
    assert "finished_runs()" in result.confirmation
    assert "No actionable issue found" not in result.confirmation
    assert provider.effects == []
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_dispatch_does_not_dispatch_over_an_incomplete_read(monkeypatch):
    """The runs Talon could read say #3093's last run completed; the one it
    could not read may be a later run that asked a question."""
    _github_with_open_3093(monkeypatch)
    agent = _with_talon(
        _dispatch_agent(registration=SimpleNamespace(owner="feature:x")),
        _TalonJobs(
            [_talon_run("completed")],
            complete=False,
            reason="jobs.json: JSONDecodeError",
        ),
    )
    feat = _feature_with_3093_blocker(agent)

    result = await feat.signal_dispatch(mode="execute")

    assert result.data["reason_code"] == "RUN_HISTORY_UNCONFIRMED"
    assert result.data["dispatched"] is False
    assert "jobs.json: JSONDecodeError" in result.confirmation
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_dispatch_hands_the_history_it_read_to_selection():
    agent = _with_talon(_dispatch_agent(registration=None), _TalonJobs([_talon_run()]))
    feat = _make_feature({}, agent=agent)
    pick = AsyncMock(return_value=None)

    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.pick_top_issue", new=pick
    ):
        await feat.signal_dispatch(mode="suggest")

    history = pick.await_args.kwargs["run_history"]
    assert history.latest("o/r", 3093).disposition == "clarifying"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_signal_dispatch_does_not_redispatch_an_unanswered_run(mode, monkeypatch):
    """#3398 acceptance, through the real selector and the real history
    read: #3093's last run ended clarifying, its label has since been
    cleared, and nothing on the issue is newer than the run. It is not
    dispatched, and the output says why and what would re-arm it."""
    _github_with_open_3093(monkeypatch)
    provider = _TalonJobs([_talon_run("clarifying")])
    agent = _with_talon(
        _dispatch_agent(registration=SimpleNamespace(owner="feature:x")), provider
    )
    feat = _feature_with_3093_blocker(agent)

    result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    assert result.data["issue"] is None
    [skip] = result.data["skipped"]
    assert skip["reason"] == "run_history"
    assert "No actionable issue found." in result.confirmation
    assert (
        "skipped o/r#3093 -- Talon job 9a63c66b ended clarifying at "
        "2026-09-24 08:41 UTC without a PR, and nothing since authorizes a retry"
    ) in result.confirmation
    assert provider.reads == 1
    assert provider.effects == []
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["suggest", "execute"])
async def test_signal_dispatch_reads_talons_history_without_polling(mode, monkeypatch):
    """Review P2 on #3432: the real ``poll()`` reaps jobs, pushes preserved
    work (``git push origin``) and persists the registry. A preview must not
    publish work, and the history read is the same in both modes, so neither
    polls. #3093's last run completed, so it is picked."""
    _github_with_open_3093(monkeypatch)
    provider = _TalonJobs([_talon_run("completed")])
    agent = _with_talon(
        _dispatch_agent(registration=SimpleNamespace(owner="feature:x")), provider
    )
    feat = _feature_with_3093_blocker(agent)

    result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.OK
    assert result.data["issue"]["issue_number"] == 3093
    assert result.data["dispatched"] is (mode == "execute")
    assert provider.reads == 1
    assert provider.effects == []


# ---------------------------------------------------------------------------
# #3464: the allow-list, through signal_dispatch and the real selector
# ---------------------------------------------------------------------------


def _github_board(monkeypatch, issues, *, unreadable_activity=()):
    """GitHub serving ``issues`` by ``(repo, number)``: no PR links any of
    them, nothing on any is newer than its last Talon run, and the post-run
    activity of each issue in ``unreadable_activity`` cannot be read."""
    from kestrel_sovereign.features.strategic_memory import issue_selection

    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")
    reads = []

    async def fake_get(path, token):
        reads.append(path)
        for (repo, number), issue in issues.items():
            if path == f"/repos/{repo}/issues/{number}":
                return issue
        if "/issues?" in path:
            return []
        raise RuntimeError(f"404 {path}")

    async def fake_post(path, token, body):
        variables = body["variables"]
        key = (f"{variables['owner']}/{variables['name']}", variables["number"])
        if "timelineItems" in body["query"]:
            if key in unreadable_activity:
                return None
            return {"data": {"repository": {"issue": {
                "lastEditedAt": None, "timelineItems": {"nodes": []},
            }}}}
        return {"data": {"repository": {"issue": {
            "closedByPullRequestsReferences": {"nodes": []},
        }}}}

    monkeypatch.setattr(issue_selection, "github_api_get", fake_get)
    monkeypatch.setattr(issue_selection, "github_api_post", fake_post)
    return reads


def _run_on(issue, disposition):
    return {**_talon_run(disposition, job_id=f"job{issue}abcdef"), "issue": issue}


@pytest.mark.asyncio
async def test_signal_dispatch_suggest_names_every_gate_that_refused_a_candidate(
    monkeypatch,
):
    """#3464 acceptance: a suggest run is evidence that the gates work. Each
    refused candidate is reported with its reason, and the pick is the
    agent-ready issue in the scanned repository that owns it -- not the
    critical row, which names a repository this agent does not scan."""
    _github_board(
        monkeypatch,
        {
            ("o/r", 3319): {
                **_agent_ready_issue(3319, "bug only"), "labels": [{"name": "bug"}],
            },
            ("o/r", 3093): {**_agent_ready_issue(3093, "epic"), "state": "closed"},
            ("o/r", 3398): _agent_ready_issue(3398, "asked a question"),
            ("o/r", 3400): _agent_ready_issue(3400, "asked, unreadable since"),
            ("o/r", 3464): _agent_ready_issue(3464, "allow-list the selector"),
            ("o/talon", 35): _agent_ready_issue(35, "elsewhere", repo="o/talon"),
        },
        unreadable_activity={("o/r", 3400)},
    )
    agent = _with_talon(
        _dispatch_agent(registration=SimpleNamespace(owner="feature:x")),
        _TalonJobs([_run_on(3398, "clarifying"), _run_on(3400, "blocked")]),
    )
    feat = _make_feature(
        {"morning_signal_config": {"scan_repos": ["o/r"]}},
        agent=agent,
        blockers=[
            {"severity": "critical", "issue": "o/talon#35", "title": "loudest"},
            {"severity": "high", "issue": "o/r#3319", "title": "bug"},
            {"severity": "high", "issue": "o/r#3093", "title": "epic"},
            {"severity": "high", "issue": "o/r#3398", "title": "asked"},
            {"severity": "high", "issue": "o/r#3400", "title": "asked"},
            {"severity": "high", "issue": "o/r#3464", "title": "ready"},
        ],
    )

    result = await feat.signal_dispatch(mode="suggest")

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    assert (result.data["issue"]["repo"], result.data["issue"]["issue_number"]) == (
        "o/r", 3464,
    )
    assert [
        (skip["repo"], skip["issue_number"], skip["reason"])
        for skip in result.data["skipped"]
    ] == [
        ("o/talon", 35, "wrong_repo"),
        ("o/r", 3319, "not_agent_ready"),
        ("o/r", 3093, "closed"),
        ("o/r", 3398, "run_history"),
        ("o/r", 3400, "run_history_unconfirmed"),
    ]
    for line in (
        "skipped o/talon#35 -- o/talon is not in morning_signal_config.scan_repos",
        "skipped o/r#3319 -- not labelled agent-ready",
        "skipped o/r#3093 -- closed",
        "skipped o/r#3398 -- Talon job job3398a ended clarifying",
        "skipped o/r#3400 -- Talon job job3400a ended blocked",
    ):
        assert line in result.confirmation, line
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_dispatch_with_no_eligible_candidate_dispatches_nothing(monkeypatch):
    """When nothing passes the allow-list, execute is a no-op that says why:
    there is no fallback to the top ledger row."""
    reads = _github_board(
        monkeypatch,
        {
            ("o/r", 3319): {
                **_agent_ready_issue(3319, "bug only"), "labels": [{"name": "bug"}],
            },
        },
    )
    agent = _dispatch_agent(registration=SimpleNamespace(owner="feature:x"))
    feat = _make_feature(
        {"morning_signal_config": {"scan_repos": ["o/r"]}},
        agent=agent,
        blockers=[
            {"severity": "critical", "issue": "o/r#3319", "title": "top row"},
            {"severity": "high", "issue": "o/talon#35", "title": "elsewhere"},
        ],
    )

    result = await feat.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    assert result.data["issue"] is None
    assert "No actionable issue found." in result.confirmation
    assert sorted(skip["reason"] for skip in result.data["skipped"]) == [
        "not_agent_ready", "wrong_repo",
    ]
    assert "/repos/o/talon/issues/35" not in reads
    agent.execute_named_tool.assert_not_awaited()
