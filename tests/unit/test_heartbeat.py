"""
Tests for the heartbeat system (#151).

Verifies HeartbeatConfig loading, HeartbeatRunner scheduling,
response normalization, dispatcher translation, and API integration.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kestrel_sovereign.heartbeat import (
    HeartbeatConfig,
    HeartbeatRunner,
)
from kestrel_sovereign.heartbeat_response import (
    HEARTBEAT_OK_PATTERN,
    HeartbeatResponseClassification,
    classify_heartbeat_response,
)
from kestrel_sovereign.config import parse_duration


# --- Config Tests ---


class TestParseDuration:
    """Tests for the duration parsing utility."""

    def test_seconds(self):
        assert parse_duration("30s") == 30

    def test_minutes(self):
        assert parse_duration("5m") == 300

    def test_hours(self):
        assert parse_duration("1h") == 3600

    def test_compound(self):
        assert parse_duration("1h30m") == 5400

    def test_full_compound(self):
        assert parse_duration("2h15m30s") == 8130

    def test_plain_integer_as_minutes(self):
        assert parse_duration("30") == 1800

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_duration("")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_duration("abc")


class TestHeartbeatConfig:
    """Tests for HeartbeatConfig construction."""

    def test_defaults(self):
        cfg = HeartbeatConfig()
        assert cfg.enabled is False
        assert cfg.interval_seconds == 1800
        assert cfg.timezone == "UTC"
        assert cfg.target == "log"
        assert cfg.suppress_ok is True

    def test_explicit_false_suppress_ok_remains_constructor_compatible(self):
        cfg = HeartbeatConfig(suppress_ok=False)
        assert cfg.suppress_ok is False

    @patch("kestrel_sovereign.heartbeat.load_section")
    def test_from_config_enabled(self, mock_load):
        mock_load.return_value = {
            "enabled": True,
            "interval": "15m",
            "active_hours_start": "09:00",
            "active_hours_end": "22:00",
            "timezone": "America/New_York",
            "target": "last_session",
            "suppress_ok": False,
        }
        cfg = HeartbeatConfig.from_config()
        assert cfg.enabled is True
        assert cfg.interval_seconds == 900  # 15 minutes
        assert cfg.active_hours_start == "09:00"
        assert cfg.timezone == "America/New_York"
        assert cfg.suppress_ok is False

    @patch("kestrel_sovereign.heartbeat.load_section")
    def test_from_config_empty(self, mock_load):
        mock_load.return_value = {}
        cfg = HeartbeatConfig.from_config()
        assert cfg.enabled is False
        assert cfg.interval_seconds == 1800


# --- Response Classification Tests ---


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "   \n\t",
        "HEARTBEAT_OK",
        "HEARTBEAT_OK!",
        "**HEARTBEAT_OK**",
        "<b>HEARTBEAT_OK</b>",
        "heartbeat_ok",
        "`HEARTBEAT_OK`",
        '"HEARTBEAT_OK"',
        "```HEARTBEAT_OK```",
        "(HEARTBEAT_OK)",
    ],
)
def test_classifier_recognizes_exact_all_clear_formatting(response):
    assert classify_heartbeat_response(response) == HeartbeatResponseClassification(
        is_all_clear=True
    )


@pytest.mark.parametrize(
    "response",
    [
        "HEARTBEAT_OK\nDisk low",
        "Disk low\nHEARTBEAT_OK",
    ],
)
def test_classifier_surfaces_short_alert_in_either_token_order(response):
    assert classify_heartbeat_response(response) == HeartbeatResponseClassification(
        is_all_clear=False,
        alert_text="Disk low",
    )


def test_classifier_preserves_full_alert_without_token():
    response = "  Disk low  "
    assert classify_heartbeat_response(response) == HeartbeatResponseClassification(
        is_all_clear=False,
        alert_text=response,
    )


def test_classifier_treats_non_string_zero_as_alert():
    assert classify_heartbeat_response(0) == HeartbeatResponseClassification(
        is_all_clear=False,
        alert_text="0",
    )


def test_classifier_preserves_non_ascii_symbol_alert_beside_token():
    assert classify_heartbeat_response("HEARTBEAT_OK ⚠️") == (
        HeartbeatResponseClassification(
            is_all_clear=False,
            alert_text="⚠️",
        )
    )


@pytest.mark.parametrize(
    "response",
    [
        "prefixHEARTBEAT_OK",
        "HEARTBEAT_OK_suffix",
    ],
)
def test_classifier_does_not_match_token_inside_identifier(response):
    assert HEARTBEAT_OK_PATTERN.search(response) is None
    assert classify_heartbeat_response(response) == HeartbeatResponseClassification(
        is_all_clear=False,
        alert_text=response,
    )


# --- HeartbeatRunner Tests ---


@pytest.fixture
def mock_agent(tmp_path):
    """Create a mock KestrelAgent.

    Phase 3 of #889 routed heartbeat through the SignalDispatcher; the
    mock now provides a `dispatcher.dispatch_signal` that wraps the
    legacy `process_input` return-value plumbing in a SignalResult. Tests
    keep setting `agent.process_input.return_value` as before — the
    dispatcher wrapper reads it and packages it as `SignalResult.artifact`.
    """
    from kestrel_sdk.signals import SignalResult, Status, SignalMode

    agent = Mock()
    agent.did = "did:test:heartbeat"
    agent.storage_path = str(tmp_path / "kestrel_prime.db")
    agent.process_input = AsyncMock(return_value="HEARTBEAT_OK")

    async def fake_dispatch(signal):
        # Re-use process_input for the test's return-value/side-effect
        # configuration, then package the result the way the dispatcher
        # would: artifact carries the LLM response text.
        try:
            response = await agent.process_input(signal.payload.get("heartbeat_md", ""))
        except Exception as exc:
            return SignalResult(
                signal_id=signal.id,
                status=Status.FAILED,
                mode=SignalMode.COGNITION,
                duration_ms=1,
                error=f"{type(exc).__name__}: {exc}",
            )
        return SignalResult(
            signal_id=signal.id,
            status=Status.OK,
            mode=SignalMode.COGNITION,
            duration_ms=1,
            artifact=response,
        )

    agent.dispatcher = Mock()
    agent.dispatcher.dispatch_signal = AsyncMock(side_effect=fake_dispatch)
    return agent


@pytest.fixture
def default_config():
    return HeartbeatConfig(enabled=True, interval_seconds=60)


class TestHeartbeatRunner:
    """Tests for HeartbeatRunner."""

    def test_initialization(self, mock_agent, default_config):
        runner = HeartbeatRunner(mock_agent, default_config)
        assert not runner.is_running
        assert runner.history == []

    @pytest.mark.asyncio
    async def test_run_once_ok(self, mock_agent, default_config):
        """Run once with HEARTBEAT_OK response."""
        runner = HeartbeatRunner(mock_agent, default_config)
        result = await runner.run_once()
        assert result.status == "ok"
        assert result.message is None
        assert result.duration_ms >= 0
        mock_agent.process_input.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_suppress_ok_false_remains_behaviorally_inert(self, mock_agent):
        config = HeartbeatConfig(
            enabled=True,
            interval_seconds=60,
            suppress_ok=False,
        )
        runner = HeartbeatRunner(mock_agent, config)

        result = await runner.run_once()

        assert result.status == "ok"
        assert result.message is None

    @pytest.mark.asyncio
    async def test_run_once_alert(self, mock_agent, default_config):
        """Run once with alert response (no OK token)."""
        mock_agent.process_input.return_value = "Warning: disk space low!"
        runner = HeartbeatRunner(mock_agent, default_config)
        result = await runner.run_once()
        assert result.status == "alert"
        assert "disk space low" in result.message

    @pytest.mark.asyncio
    async def test_run_once_error(self, mock_agent, default_config):
        """Run once with process_input raising an exception."""
        mock_agent.process_input.side_effect = RuntimeError("LLM unavailable")
        runner = HeartbeatRunner(mock_agent, default_config)
        result = await runner.run_once()
        assert result.status == "error"
        assert "LLM unavailable" in result.message

    @pytest.mark.asyncio
    async def test_run_once_ok_with_extra_text(self, mock_agent, default_config):
        """Any content alongside HEARTBEAT_OK is an alert."""
        mock_agent.process_input.return_value = "HEARTBEAT_OK\nAll good."
        runner = HeartbeatRunner(mock_agent, default_config)
        result = await runner.run_once()
        assert result.status == "alert"
        assert result.message == "All good."

    @pytest.mark.asyncio
    async def test_run_once_non_string_zero_is_alert(self, mock_agent, default_config):
        mock_agent.process_input.return_value = 0
        runner = HeartbeatRunner(mock_agent, default_config)
        result = await runner.run_once()
        assert result.status == "alert"
        assert result.message == "0"

    @pytest.mark.asyncio
    async def test_history_tracking(self, mock_agent, default_config):
        """Results are stored in history."""
        runner = HeartbeatRunner(mock_agent, default_config)
        await runner.run_once()
        await runner.run_once()
        assert len(runner.history) == 2

    @pytest.mark.asyncio
    async def test_heartbeat_prompt_includes_content(self, mock_agent, default_config):
        """Heartbeat prompt includes HEARTBEAT.md content."""
        agent_dir = Path(mock_agent.storage_path).parent
        (agent_dir / "HEARTBEAT.md").write_text("Check emails")
        runner = HeartbeatRunner(mock_agent, default_config)
        await runner.run_once()
        call_args = mock_agent.process_input.call_args[0][0]
        assert "Check emails" in call_args

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_agent, default_config):
        """Start and stop the heartbeat loop."""
        default_config.interval_seconds = 3600  # Long interval so it doesn't fire
        runner = HeartbeatRunner(mock_agent, default_config)
        await runner.start()
        assert runner.is_running

        await runner.stop()
        assert not runner.is_running

    @pytest.mark.asyncio
    async def test_start_disabled(self, mock_agent):
        """Start does nothing when disabled."""
        config = HeartbeatConfig(enabled=False)
        runner = HeartbeatRunner(mock_agent, config)
        await runner.start()
        assert not runner.is_running

    def test_get_status(self, mock_agent, default_config):
        runner = HeartbeatRunner(mock_agent, default_config)
        status = runner.get_status()
        assert status["enabled"] is True
        assert status["running"] is False
        assert status["interval_seconds"] == 60
        assert status["history_count"] == 0

    def test_ensure_heartbeat_file(self, mock_agent, default_config):
        """Default HEARTBEAT.md created when missing."""
        agent_dir = Path(mock_agent.storage_path).parent
        runner = HeartbeatRunner(mock_agent, default_config)
        runner._ensure_heartbeat_file()
        heartbeat_path = agent_dir / "HEARTBEAT.md"
        assert heartbeat_path.exists()
        content = heartbeat_path.read_text()
        assert "Heartbeat Checklist" in content

    def test_ensure_heartbeat_file_no_overwrite(self, mock_agent, default_config):
        """Don't overwrite existing HEARTBEAT.md."""
        agent_dir = Path(mock_agent.storage_path).parent
        (agent_dir / "HEARTBEAT.md").write_text("Custom checklist")
        runner = HeartbeatRunner(mock_agent, default_config)
        runner._ensure_heartbeat_file()
        content = (agent_dir / "HEARTBEAT.md").read_text()
        assert content == "Custom checklist"


class TestDispatcherTranslation:
    """Tests for dispatcher statuses translated into heartbeat history."""

    @pytest.mark.asyncio
    async def test_skipped_outside_hours(self, mock_agent):
        """Heartbeat skipped when outside active hours."""
        from kestrel_sdk.signals import SignalMode, SignalResult, Status

        config = HeartbeatConfig(
            enabled=True,
            active_hours_start="09:00",
            active_hours_end="17:00",
            timezone="UTC",
        )
        runner = HeartbeatRunner(mock_agent, config)

        # Phase 3 of #889 moved active-hours enforcement out of heartbeat
        # and into the dispatcher's per-source attention_policy. Outside
        # active hours, the dispatcher returns DROPPED_QUIET_HOURS and
        # heartbeat translates that into status="skipped".
        async def quiet_hours_dispatch(signal):
            return SignalResult(
                signal_id=signal.id,
                status=Status.DROPPED_QUIET_HOURS,
                mode=SignalMode.COGNITION,
                duration_ms=0,
                error="outside attention window",
            )

        mock_agent.dispatcher.dispatch_signal.side_effect = quiet_hours_dispatch
        result = await runner.run_once()
        assert result.status == "skipped"
        assert "dropped_quiet_hours" in result.reason


class TestResponseNormalization:
    """Tests for response normalization logic."""

    @pytest.fixture
    def runner(self, mock_agent, default_config):
        return HeartbeatRunner(mock_agent, default_config)

    def test_plain_ok(self, runner):
        result = runner._normalize_response("HEARTBEAT_OK", "2026-01-01T00:00:00Z", 100)
        assert result.status == "ok"
        assert result.message is None

    def test_bold_ok(self, runner):
        result = runner._normalize_response("**HEARTBEAT_OK**", "2026-01-01T00:00:00Z", 100)
        assert result.status == "ok"

    def test_ok_with_exclamation(self, runner):
        result = runner._normalize_response("HEARTBEAT_OK!", "2026-01-01T00:00:00Z", 100)
        assert result.status == "ok"

    def test_alert_message(self, runner):
        result = runner._normalize_response(
            "Found 3 overdue tasks!", "2026-01-01T00:00:00Z", 100
        )
        assert result.status == "alert"
        assert "overdue tasks" in result.message

    def test_empty_response(self, runner):
        result = runner._normalize_response("", "2026-01-01T00:00:00Z", 100)
        assert result.status == "ok"

    def test_ok_with_substantial_extra_content(self, runner):
        result = runner._normalize_response(
            "HEARTBEAT_OK\nAll systems nominal. No urgent items. Memory usage at 42%.",
            "2026-01-01T00:00:00Z",
            100,
        )
        assert result.status == "alert"
        assert result.message == (
            "All systems nominal. No urgent items. Memory usage at 42%."
        )

    @pytest.mark.parametrize(
        ("response", "expected_status", "expected_message"),
        [
            (None, "ok", None),
            ("HEARTBEAT_OK", "ok", None),
            ("HEARTBEAT_OK\nDisk low", "alert", "Disk low"),
            ("Disk low\nHEARTBEAT_OK", "alert", "Disk low"),
            ("Disk low", "alert", "Disk low"),
            (0, "alert", "0"),
        ],
    )
    def test_runner_status_matches_canonical_classifier(
        self,
        runner,
        response,
        expected_status,
        expected_message,
    ):
        result = runner._normalize_response(
            response,
            "2026-01-01T00:00:00Z",
            100,
        )
        classification = classify_heartbeat_response(response)

        assert result.status == expected_status
        assert result.message == expected_message
        assert (result.status == "ok") is classification.is_all_clear
        assert result.message == classification.alert_text
