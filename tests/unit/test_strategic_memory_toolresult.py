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

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature


def _make_feature(data: dict | None = None) -> StrategicMemoryFeature:
    feat = StrategicMemoryFeature(agent=MagicMock())
    feat._data = data if data is not None else {}
    feat._strategy_path = None
    feat._save = MagicMock()
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


# ---------------------------------------------------------------------------
# Contract: every @tool annotated -> ToolResult
# ---------------------------------------------------------------------------

def test_strategic_memory_passes_toolresult_contract():
    from kestrel_sovereign.tools.result_contract import (
        assert_feature_returns_tool_result,
    )

    feat = StrategicMemoryFeature(agent=MagicMock())
    assert_feature_returns_tool_result(feat)
