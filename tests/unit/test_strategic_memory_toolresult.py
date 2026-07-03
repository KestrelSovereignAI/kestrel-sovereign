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

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature
from kestrel_sovereign.features.strategic_memory.feature import _SaveOutcome


def _make_feature(data: dict | None = None) -> StrategicMemoryFeature:
    feat = StrategicMemoryFeature(agent=MagicMock())
    feat._data = data if data is not None else {}
    # A strategy path IS configured; ``_save`` is stubbed to report a
    # successful persist so happy-path mutating tools return OK. Tests
    # exercising the no-path / write-failure edges (F291) override these.
    feat._strategy_path = Path("/tmp/kestrel-test/STRATEGY.yaml")
    feat._save = MagicMock(return_value=_SaveOutcome(persisted=True))
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
    feat = _make_feature({"blockers": [{"issue": "OTHER"}]})
    result = await feat.strategy_resolve_blocker(issue="GONE")
    assert result.status is ToolResultStatus.ERROR
    assert "GONE" in result.error


@pytest.mark.asyncio
async def test_resolve_blocker_present_returns_ok():
    feat = _make_feature({"blockers": [{"issue": "X-1", "title": "fix"}]})
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


# ---------------------------------------------------------------------------
# signal_dispatch fallback failure detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_signal_dispatch_fallback_failure_returns_partial():
    """When no TalonCoordinatorFeature is wired and dispatch_to_talon
    returns a "Failed to dispatch ..." or "Found issue to dispatch ...
    but no multi_agent host URL" body, the wrapper must surface PARTIAL
    so the LLM cannot narrate "dispatched" off a failure body."""
    for failure_msg in (
        "Failed to dispatch foo/bar#1 to talon: connection refused",
        "Found issue to dispatch (foo/bar#1: bug) but no multi_agent host URL configured.",
    ):
        feat = _make_feature({})
        with patch(
            "kestrel_sovereign.features.strategic_memory.feature.dispatch_to_talon",
            new=AsyncMock(return_value=failure_msg),
        ):
            result = await feat.signal_dispatch(mode="execute")
        assert result.status is ToolResultStatus.PARTIAL, failure_msg
        assert "fallback dispatch" in result.error


@pytest.mark.asyncio
async def test_signal_dispatch_fallback_success_returns_ok():
    feat = _make_feature({})
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.dispatch_to_talon",
        new=AsyncMock(return_value="Dispatched to talon: foo/bar#1 ..."),
    ):
        result = await feat.signal_dispatch(mode="execute")
    assert result.status is ToolResultStatus.OK
    assert result.data["fallback"] is True


@pytest.mark.asyncio
async def test_signal_dispatch_invalid_mode_rejected_not_dispatched():
    """A typo'd/unknown mode must error, NOT fall through to live dispatch
    (#1925). dispatch_to_talon is patched to blow up if it's ever reached."""
    feat = _make_feature({})
    boom = AsyncMock(side_effect=AssertionError("must not dispatch on bad mode"))
    with patch(
        "kestrel_sovereign.features.strategic_memory.feature.dispatch_to_talon",
        new=boom,
    ):
        # Genuine typos / ambiguous words must error (and list valid values).
        # "run"/"apply"/"go" are deliberately NOT aliased to execute — a live
        # dispatch requires the literal canonical value (#1923 asymmetry).
        for bad in ("suggst", "run", "apply", "go", ""):
            result = await feat.signal_dispatch(mode=bad)
            assert result.status is ToolResultStatus.ERROR, bad
            assert "Must be one of: execute, suggest" in result.error
    boom.assert_not_called()


@pytest.mark.asyncio
async def test_signal_dispatch_dryrun_synonyms_map_to_suggest_no_dispatch():
    """Safe preview synonyms (dry-run, plan, preview) normalize to 'suggest'
    and never reach live dispatch (#1923)."""
    boom = AsyncMock(side_effect=AssertionError("suggest must not dispatch"))
    for syn in ("dry-run", "dry_run", "plan", "preview", "PLAN"):
        feat = _make_feature({})
        with patch(
            "kestrel_sovereign.features.strategic_memory.feature.dispatch_to_talon",
            new=boom,
        ):
            result = await feat.signal_dispatch(mode=syn)
        assert result.status is ToolResultStatus.OK, syn
        assert result.data["mode"] == "suggest", syn
    boom.assert_not_called()


@pytest.mark.asyncio
async def test_signal_dispatch_suggest_is_case_insensitive():
    feat = _make_feature({})
    result = await feat.signal_dispatch(mode="SUGGEST")
    assert result.status is ToolResultStatus.OK
    assert result.data["mode"] == "suggest"


@pytest.mark.asyncio
async def test_strategy_add_blocker_invalid_severity_rejected():
    feat = _make_feature({})
    result = await feat.strategy_add_blocker(issue="42", title="x", severity="sev1")
    assert result.status is ToolResultStatus.ERROR
    assert "Must be one of: low, medium, high, critical" in result.error
    # Nothing persisted on rejection.
    assert not feat._data.get("blockers")


@pytest.mark.asyncio
async def test_strategy_add_blocker_severity_normalized():
    feat = _make_feature({})
    result = await feat.strategy_add_blocker(issue="42", title="x", severity="HIGH")
    assert result.status is ToolResultStatus.OK
    assert feat._data["blockers"][-1]["severity"] == "high"


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
        feat = _make_feature({"blockers": [{"issue": "42", "title": "x"}]})
        # No path AND the real _save so it detects the no-path condition.
        feat._strategy_path = None
        del feat._save  # drop the stub; use the real method
        call = dict(_mutating_calls(feat))[name]
        result = await call()
        assert result.status is ToolResultStatus.ERROR, name
        assert result.data["persisted"] is False, name


@pytest.mark.asyncio
async def test_mutating_tools_write_failure_return_partial():
    """When the write raises, the in-memory update stands but nothing was
    persisted -> PARTIAL with the error surfaced, never OK."""
    for name, _ in _mutating_calls(None):
        feat = _make_feature({"blockers": [{"issue": "42", "title": "x"}]})
        feat._save = MagicMock(
            return_value=_SaveOutcome(persisted=False, error="disk full")
        )
        call = dict(_mutating_calls(feat))[name]
        result = await call()
        assert result.status is ToolResultStatus.PARTIAL, name
        assert "disk full" in result.error, name
        assert result.data["persisted"] is False, name


@pytest.mark.asyncio
async def test_mutating_tools_happy_path_return_ok():
    """Path present + write succeeds -> OK, persisted flag true."""
    for name, _ in _mutating_calls(None):
        feat = _make_feature({"blockers": [{"issue": "42", "title": "x"}]})
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
