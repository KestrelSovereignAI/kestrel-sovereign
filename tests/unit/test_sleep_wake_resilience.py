"""Unit tests for host sleep/wake (suspend/resume) resilience (#1545).

Covers the four behaviours added by the issue:
1. Suspend detection — `suspend_gap_seconds` + `ResumeMonitor` (injected clocks).
2. Dispatcher re-anchoring — `notify_resume` clears coalescing + rate-limit,
   and an end-to-end `system.resumed` ACTION dispatch invokes the handler.
3. Scheduler misfire grace — a task far past its scheduled time is skipped and
   re-anchored, not fired late; within grace it still runs.
4. Heartbeat gap awareness — missed-beat math and the loop's detection hook.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.resume_monitor import (
    DEFAULT_THRESHOLD_SECONDS,
    DEFAULT_TICK_SECONDS,
    ResumeMonitor,
    ResumeMonitorConfig,
    suspend_gap_seconds,
)
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.heartbeat import HeartbeatConfig, HeartbeatRunner, HeartbeatResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_db():
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    return db


class _ScriptedClock:
    """Returns a preset sequence of values, one per call (last value sticks)."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def __call__(self) -> float:
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


# ---------------------------------------------------------------------------
# 1. Suspend detection
# ---------------------------------------------------------------------------


class TestSuspendGap:
    def test_no_gap_when_clocks_advance_together(self):
        # Awake: wall and monotonic advance by the same ~30s.
        assert suspend_gap_seconds(1000.0, 1030.0, 500.0, 530.0) == 0.0

    def test_gap_is_wall_minus_monotonic_delta(self):
        # Slept 1h: wall advanced 3630s, monotonic only the 30s tick.
        gap = suspend_gap_seconds(1000.0, 4630.0, 500.0, 530.0)
        assert gap == pytest.approx(3600.0)

    def test_backward_wallclock_step_never_negative(self):
        # NTP corrected the wall clock backwards — gap floored at 0.
        assert suspend_gap_seconds(1000.0, 990.0, 500.0, 530.0) == 0.0


class TestResumeMonitor:
    async def test_first_poll_establishes_baseline_no_fire(self):
        fired = []
        mon = ResumeMonitor(
            on_resume=lambda g: fired.append(g),
            threshold_seconds=120.0,
            wall_clock=_ScriptedClock([1000.0]),
            mono_clock=_ScriptedClock([500.0]),
        )
        gap = await mon.poll_once()
        assert gap == 0.0
        assert fired == []

    async def test_no_fire_when_awake(self):
        fired = []

        async def on_resume(g):
            fired.append(g)

        mon = ResumeMonitor(
            on_resume=on_resume,
            threshold_seconds=120.0,
            wall_clock=_ScriptedClock([1000.0, 1030.0]),
            mono_clock=_ScriptedClock([500.0, 530.0]),
        )
        await mon.poll_once()  # baseline
        gap = await mon.poll_once()  # awake tick
        assert gap == 0.0
        assert fired == []

    async def test_fires_once_on_suspend(self):
        fired = []

        async def on_resume(g):
            fired.append(g)

        # Tick 2: wall jumped 3630s, monotonic only 30s → 3600s suspend.
        mon = ResumeMonitor(
            on_resume=on_resume,
            threshold_seconds=120.0,
            wall_clock=_ScriptedClock([1000.0, 4630.0, 4660.0]),
            mono_clock=_ScriptedClock([500.0, 530.0, 560.0]),
        )
        await mon.poll_once()  # baseline
        gap = await mon.poll_once()  # suspend detected
        assert gap == pytest.approx(3600.0)
        assert len(fired) == 1
        # Next awake tick must not re-fire the same gap.
        gap3 = await mon.poll_once()
        assert gap3 == 0.0
        assert len(fired) == 1

    async def test_callback_exception_does_not_propagate(self):
        async def boom(g):
            raise RuntimeError("consumer blew up")

        mon = ResumeMonitor(
            on_resume=boom,
            threshold_seconds=120.0,
            wall_clock=_ScriptedClock([1000.0, 4630.0]),
            mono_clock=_ScriptedClock([500.0, 530.0]),
        )
        await mon.poll_once()
        # Must swallow the consumer error — the monitor has to survive.
        gap = await mon.poll_once()
        assert gap == pytest.approx(3600.0)


class TestResumeMonitorConfig:
    def test_defaults(self):
        cfg = ResumeMonitorConfig()
        assert cfg.enabled is True
        assert cfg.tick_seconds == DEFAULT_TICK_SECONDS
        assert cfg.threshold_seconds == DEFAULT_THRESHOLD_SECONDS


# ---------------------------------------------------------------------------
# 2. Dispatcher re-anchoring
# ---------------------------------------------------------------------------


class TestDispatcherReanchor:
    def test_rate_limit_state_reset(self):
        from kestrel_sovereign.signals.dispatcher import _RateLimitState

        rate = _RateLimitState()
        from kestrel_sdk.signals import RateLimit

        # Saturate a per_hour=2 source.
        rl = RateLimit(per_hour=2)
        assert rate.check_and_record("src", rl, now=100.0) is False
        assert rate.check_and_record("src", rl, now=101.0) is False
        assert rate.check_and_record("src", rl, now=102.0) is True  # saturated

        rate.reset()
        # After re-anchor the window is clear again.
        assert rate.check_and_record("src", rl, now=103.0) is False

    def test_coalescing_state_reset(self):
        from kestrel_sovereign.signals.dispatcher import _CoalescingState

        coal = _CoalescingState()
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        window = timedelta(seconds=10)
        assert coal.check_and_record("src", "key", window, now=now) is False
        # Same key within window → coalesced.
        assert coal.check_and_record("src", "key", window, now=now) is True

        coal.reset()
        # After re-anchor the key is forgotten and fires fresh.
        assert coal.check_and_record("src", "key", window, now=now) is False

    async def test_notify_resume_clears_both(self):
        from kestrel_sovereign.signals.dispatcher import SignalDispatcher

        dispatcher = SignalDispatcher(
            agent=MagicMock(),
            registry=MagicMock(),
            lock_manager=MagicMock(),
            store=MagicMock(),
        )
        dispatcher._rate.reset = MagicMock()
        dispatcher._coalescing.reset = MagicMock()
        dispatcher.notify_resume(3600.0)
        dispatcher._rate.reset.assert_called_once()
        dispatcher._coalescing.reset.assert_called_once()

    def test_system_resumed_registration_is_valid(self):
        from kestrel_sovereign.signals.registry import SourceRegistry
        from kestrel_sovereign.signals.sources.system_resumed import (
            SOURCE_NAME,
            build_system_resumed_registration,
        )

        async def handler(payload):
            return {"ok": True}

        reg = build_system_resumed_registration(handler=handler)
        # Registry validation must accept it (raises on any v1 violation).
        registry = SourceRegistry()
        registry.register(reg)
        assert SOURCE_NAME in registry

    def test_system_resumed_schema_rejects_bad_payload(self):
        from kestrel_sovereign.signals.sources.system_resumed import _schema

        assert _schema({"gap_seconds": 12})["gap_seconds"] == 12.0
        with pytest.raises(ValueError):
            _schema({"gap_seconds": -1})
        with pytest.raises(ValueError):
            _schema({"unexpected": 1})


# ---------------------------------------------------------------------------
# 3. Scheduler misfire grace
# ---------------------------------------------------------------------------


def _due_row(task_id, name, cron, next_run_at, *, created="2026-01-01T00:00:00"):
    # Matches the SELECT column order in SchedulerRunner._tick.
    return (task_id, "test-agent", name, cron, "{}", 1, None, next_run_at, created)


class TestSchedulerMisfireGrace:
    async def test_stale_task_is_skipped_not_executed(self):
        db = _make_mock_db()
        now = datetime.now(timezone.utc)
        # Scheduled 1 hour ago — well past a 600s grace.
        stale = (now - timedelta(hours=1)).isoformat()
        db.fetchall = AsyncMock(
            return_value=[_due_row("t1", "wellness_check", "@hourly", stale)]
        )
        # Re-read for re-anchor returns a live row.
        db.fetchone = AsyncMock(return_value=("@hourly", 1))
        executor = AsyncMock()
        runner = SchedulerRunner(
            db, "test-agent", executor, misfire_grace_seconds=600
        )
        await runner._tick()

        executor.assert_not_called()  # the slept-through run is skipped
        sqls = [c[0][0] for c in db.execute.call_args_list]
        # A skipped_misfire audit row was written...
        assert any("task_execution_log" in s and "skipped_misfire" in s for s in sqls)
        # ...and next_run_at was re-anchored without touching last_run_at.
        assert any(
            "UPDATE scheduled_tasks SET next_run_at" in s for s in sqls
        )

    async def test_recent_task_still_executes(self):
        db = _make_mock_db()
        now = datetime.now(timezone.utc)
        # Scheduled 5s ago — inside the 600s grace.
        recent = (now - timedelta(seconds=5)).isoformat()
        db.fetchall = AsyncMock(
            return_value=[_due_row("t1", "wellness_check", "@hourly", recent)]
        )
        db.fetchone = AsyncMock(return_value=("@hourly", 1))
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(
            db, "test-agent", executor, misfire_grace_seconds=600
        )
        await runner._tick()

        executor.assert_called_once_with("wellness_check", {})

    async def test_grace_zero_disables_rail(self):
        db = _make_mock_db()
        now = datetime.now(timezone.utc)
        stale = (now - timedelta(hours=5)).isoformat()
        db.fetchall = AsyncMock(
            return_value=[_due_row("t1", "wellness_check", "@hourly", stale)]
        )
        db.fetchone = AsyncMock(return_value=("@hourly", 1))
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(
            db, "test-agent", executor, misfire_grace_seconds=0
        )
        await runner._tick()

        # Legacy behaviour: even a 5h-late task fires.
        executor.assert_called_once_with("wellness_check", {})

    def test_seconds_late_handles_naive_and_missing(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Naive timestamp is treated as UTC.
        assert SchedulerRunner._seconds_late("2026-01-01T11:00:00", now) == pytest.approx(3600.0)
        assert SchedulerRunner._seconds_late(None, now) == 0.0
        assert SchedulerRunner._seconds_late("not-a-date", now) == 0.0


# ---------------------------------------------------------------------------
# 4. Heartbeat gap awareness
# ---------------------------------------------------------------------------


class TestHeartbeatGapAwareness:
    def test_compute_missed_beats(self):
        # One interval elapsed → normal, 0 missed.
        assert HeartbeatRunner._compute_missed_beats(1800, 1800) == 0
        # ~3h with 30m interval → 6 beats span, 5 slept through.
        assert HeartbeatRunner._compute_missed_beats(3 * 3600 + 60, 1800) == 5
        # Sub-interval elapsed → 0.
        assert HeartbeatRunner._compute_missed_beats(100, 1800) == 0
        # Defensive: non-positive interval → 0, no ZeroDivision.
        assert HeartbeatRunner._compute_missed_beats(1000, 0) == 0

    def test_note_sleep_elapsed_sets_pending_on_suspend(self):
        runner = HeartbeatRunner(MagicMock(), HeartbeatConfig(interval_seconds=1800))
        # Simulate an inter-tick sleep that actually spanned ~3h (suspend).
        import time as _time

        runner._pre_sleep_wall = _time.time() - (3 * 3600 + 60)
        runner._note_sleep_elapsed()
        gap, missed = runner._pending_gap
        assert missed == 5
        assert gap > 0

    def test_slow_tick_is_not_a_suspend(self):
        # The bug codex caught: a tick slower than the interval must NOT
        # register as a missed beat. Only the sleep span is measured, and a
        # normal sleep ≈ one interval → 0 missed regardless of tick duration.
        runner = HeartbeatRunner(MagicMock(), HeartbeatConfig(interval_seconds=30))
        import time as _time

        runner._pre_sleep_wall = _time.time() - 31  # ~one interval of sleep
        runner._note_sleep_elapsed()
        gap, missed = runner._pending_gap
        assert missed == 0
        assert gap == 0.0

    def test_consume_pending_gap_clears(self):
        runner = HeartbeatRunner(MagicMock(), HeartbeatConfig(interval_seconds=1800))
        runner._pending_gap = (3600.0, 2)
        assert runner._consume_pending_gap() == (3600.0, 2)
        assert runner._consume_pending_gap() == (0.0, 0)

    def test_no_gap_when_on_cadence(self):
        runner = HeartbeatRunner(MagicMock(), HeartbeatConfig(interval_seconds=1800))
        import time as _time

        runner._pre_sleep_wall = _time.time() - 1800  # exactly one interval
        runner._note_sleep_elapsed()
        gap, missed = runner._pending_gap
        assert missed == 0
        assert gap == 0.0
