"""Shared scheduler reliability states across worker, health, and recovery."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.features.health.checks import check_scheduler_liveness
from kestrel_sovereign.features.restart_coordinator.feature import (
    MAX_IDLE_ONLY_DEFERRAL_SECONDS,
    RestartCoordinatorFeature,
)
from kestrel_sovereign.features.restart_coordinator.store import (
    get_request,
    insert_request,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import (
    ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE,
    SCHEDULER_PROTOCOL_VERSION,
    SchedulerRunner,
)
from kestrel_sovereign.features.scheduler.status import (
    RUNTIME_STATUS_RETENTION_SECONDS,
    emit_runtime_status,
    ensure_runtime_status_table,
    scheduler_status,
    scheduler_status_parameters,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.database_clock import database_clock
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

scheduler_database_clock = database_clock


async def _database(path) -> AsyncDatabase:
    backend = SQLiteBackend(str(path))
    await backend.connect()
    return AsyncDatabase(backend)


@pytest.mark.asyncio
async def test_runtime_report_distinguishes_missing_zero_and_system_disabled(tmp_path):
    db = await _database(tmp_path / "scheduler-status.db")
    runner = SchedulerRunner(db, "agent-1", lambda *_: None, owner_id="status-test")
    try:
        await runner._ensure_tables()

        missing = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert missing["state"] == "no_telemetry"
        assert missing["telemetry_received"] is False
        assert missing["enabled_count"] == 0

        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="status-test",
            worker_state="running",
            last_tick_started_at=datetime.now(timezone.utc).isoformat(),
            last_tick_completed_at=datetime.now(timezone.utc).isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        zero = await scheduler_status(db, agent_id="agent-1", poll_interval=30)
        assert zero["state"] == "running_zero_schedules"
        assert zero["telemetry_received"] is True
        assert zero["reported_enabled_count"] == 0

        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, scheduler_protocol_version)
            VALUES ('operator-paused', 'agent-1', 'wait_reconcile',
                    '* * * * *', '{}', 0, ?, ?, ?)
            """,
            (now, now, SCHEDULER_PROTOCOL_VERSION),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="status-test",
            worker_state="running",
            last_tick_started_at=now,
            last_tick_completed_at=now,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        paused = await scheduler_status(db, agent_id="agent-1", poll_interval=30)
        assert paused["state"] == "running_only_operator_paused_schedules"
        assert paused["status"] == "pass"
        assert paused["disabled_reasons"] == {"operator_paused": 1}

        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, scheduler_protocol_version)
            VALUES ('enabled', 'agent-1', 'wait_reconcile', '* * * * *',
                    '{}', 1, ?, ?, ?)
            """,
            (future, now, SCHEDULER_PROTOCOL_VERSION),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="status-test",
            worker_state="running",
            last_tick_started_at=now,
            last_tick_completed_at=now,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        active = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert active["state"] == "running"
        assert active["configured_enabled_count"] == 1
        assert active["enabled_count"] == 1
        assert active["reported_configured_enabled_count"] == 1
        assert active["reported_enabled_count"] == 1
        await db.execute("DELETE FROM scheduled_tasks WHERE id = 'enabled'")

        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, terminal_status,
                 scheduler_protocol_version)
            VALUES (?, ?, 'restart_coordinator', '* * * * *', '{}', 0,
                    ?, ?, ?, ?)
            """,
            (
                "ambiguous",
                "agent-1",
                now,
                now,
                ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE,
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="status-test",
            worker_state="running",
            last_tick_started_at=now,
            last_tick_completed_at=now,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        disabled = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert disabled["state"] == "system_disabled_schedules"
        assert disabled["enabled_count"] == 0
        assert disabled["system_disabled_count"] == 1
        assert disabled["disabled_reasons"] == {
            "operator_paused": 1,
            ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE: 1,
        }

        await db.execute(
            """
            UPDATE scheduler_runtime_status
            SET reported_at = ?
            WHERE agent_id = ?
            """,
            (
                (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                "agent-1",
            ),
        )
        health_agent = SimpleNamespace(
            did="agent-1",
            features={
                "SchedulerFeature": SimpleNamespace(
                    enabled=True,
                    _runner=SimpleNamespace(_poll_interval=30),
                    _initialized_monotonic=0.0,
                )
            },
        )
        health = await check_scheduler_liveness(health_agent, db)
        assert health["status"] == "fail"
        assert health["details"]["state"] == "stale"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_enabled_schedules_without_valid_next_run_are_not_runnable(tmp_path):
    db = await _database(tmp_path / "scheduler-non-runnable.db")
    agent = SimpleNamespace(
        did="agent-1",
        agent_id="agent-1",
        storage=SimpleNamespace(db=db),
        signal_registry=None,
        wait_registry=None,
        features={},
    )
    feature = SchedulerFeature(agent)
    try:
        await feature.initialize()
        agent.features["SchedulerFeature"] = feature
        now = await scheduler_database_clock(db)
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, scheduler_protocol_version)
            VALUES ('invalid-cron-resume', 'agent-1', 'wait_reconcile',
                    'not a cron', '{}', 0, NULL, ?, ?)
            """,
            (now.isoformat(), SCHEDULER_PROTOCOL_VERSION),
        )

        resumed = await feature.schedule_resume("invalid-cron-resume")
        assert resumed.status.value == "partial"
        assert resumed.data["next_run_at"] is None
        assert await db.fetchone(
            "SELECT enabled, next_run_at FROM scheduled_tasks WHERE id = ?",
            ("invalid-cron-resume",),
        ) == (1, None)

        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, scheduler_protocol_version)
            VALUES ('malformed-legacy-time', 'agent-1', 'wait_reconcile',
                    '* * * * *', '{}', 1, 'not-a-timestamp', ?, ?)
            """,
            (now.isoformat(), SCHEDULER_PROTOCOL_VERSION),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=feature._runner._owner_id,
            worker_state="running",
            last_tick_started_at=now.isoformat(),
            last_tick_completed_at=now.isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        status = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            expected_owner_id=feature._runner._owner_id,
        )
        assert status["configured_enabled_count"] == 2
        assert status["enabled_count"] == 0
        assert status["non_runnable_count"] == 2
        assert status["non_runnable_reasons"] == {
            "missing_next_run_at": 1,
            "invalid_next_run_at": 1,
        }
        assert status["reported_configured_enabled_count"] == 2
        assert status["reported_enabled_count"] == 0
        assert status["next_run_at"] is None
        assert status["state"] == "non_runnable_schedules"
        assert status["status"] == "fail"

        health = await check_scheduler_liveness(agent, db)
        assert health["status"] == "fail"
        assert health["details"]["state"] == "non_runnable_schedules"
        assert "without a valid next_run_at" in health["message"]
    finally:
        await feature.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_multi_owner_health_uses_fresh_healthy_worker_without_peer_flap(
    tmp_path,
):
    db = await _database(tmp_path / "scheduler-multi-owner.db")
    runner = SchedulerRunner(db, "agent-1", lambda *_: None, owner_id="healthy")
    now = datetime.now(timezone.utc).isoformat()
    try:
        await runner._ensure_tables()
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="healthy",
            worker_state="running",
            last_tick_started_at=now,
            last_tick_completed_at=now,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="peer",
            worker_state="restarting",
            last_tick_started_at=now,
            last_tick_completed_at=now,
            restart_count=3,
            consecutive_failures=2,
            last_error_type="RuntimeError",
        )

        degraded_peer = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert degraded_peer["status"] == "pass"
        assert degraded_peer["owner_id"] == "healthy"
        assert degraded_peer["fresh_worker_count"] == 2
        assert degraded_peer["running_worker_count"] == 1

        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="peer",
            worker_state="stopped",
            last_tick_started_at=now,
            last_tick_completed_at=now,
            restart_count=3,
            consecutive_failures=2,
            last_error_type="RuntimeError",
        )
        stopping_peer = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert stopping_peer["status"] == "pass"
        assert stopping_peer["owner_id"] == "healthy"

        await db.execute(
            "UPDATE scheduler_runtime_status SET reported_at = ? "
            "WHERE agent_id = ? AND owner_id = ?",
            (
                (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                "agent-1",
                "healthy",
            ),
        )
        only_stopped_is_fresh = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert only_stopped_is_fresh["status"] == "fail"
        assert only_stopped_is_fresh["state"] == "stopped"
        assert only_stopped_is_fresh["fresh_worker_count"] == 1
        assert only_stopped_is_fresh["running_worker_count"] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_runtime_status_bounds_historical_owners_and_reaps_expired_rows(
    tmp_path,
):
    db = await _database(tmp_path / "scheduler-owner-retention.db")
    runner = SchedulerRunner(db, "agent-1", lambda *_: None, owner_id="healthy")
    try:
        await runner._ensure_tables()
        now = datetime.now(timezone.utc)
        for index in range(20):
            await db.execute(
                """INSERT INTO scheduler_runtime_status
                       (agent_id, owner_id, worker_state, reported_at)
                   VALUES (?, ?, 'stopped', ?)""",
                (
                    "agent-1",
                    f"historical-{index}",
                    (now - timedelta(minutes=10, seconds=index)).isoformat(),
                ),
            )
        await db.execute(
            """INSERT INTO scheduler_runtime_status
                   (agent_id, owner_id, worker_state, reported_at)
               VALUES (?, ?, 'stopped', ?)""",
            ("agent-1", "expired", (now - timedelta(days=2)).isoformat()),
        )

        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="healthy",
            worker_state="running",
            last_tick_started_at=now.isoformat(),
            last_tick_completed_at=now.isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        assert await db.fetchone(
            "SELECT COUNT(*) FROM scheduler_runtime_status "
            "WHERE owner_id = 'expired'"
        ) == (0,)
        status = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert status["status"] == "pass"
        assert status["owner_id"] == "healthy"
        assert status["worker_reports_received"] == 1
        assert status["stale_worker_count"] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_runtime_status_reaps_revoked_tenants_without_deleting_peer_rows(
    tmp_path,
):
    db = await _database(tmp_path / "scheduler-revoked-tenant-status.db")
    authorized = {"tenant-a", "tenant-b"}
    runner = SchedulerRunner(
        db,
        None,
        lambda *_: None,
        owner_id="dynamic-runner",
        authorized_agent_ids=authorized,
        authorized_agent_ids_provider=lambda: authorized,
    )
    now = datetime.now(timezone.utc)
    try:
        await runner._ensure_tables()
        await runner._publish_runtime_status("running")
        await emit_runtime_status(
            db,
            agent_ids=("tenant-a", "tenant-b"),
            owner_id="live-peer",
            worker_state="running",
            last_tick_started_at=now.isoformat(),
            last_tick_completed_at=now.isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        authorized.remove("tenant-a")
        await runner._publish_runtime_status("running")
        assert await db.fetchall(
            "SELECT agent_id, owner_id FROM scheduler_runtime_status "
            "ORDER BY agent_id, owner_id"
        ) == [
            ("tenant-a", "live-peer"),
            ("tenant-b", "dynamic-runner"),
            ("tenant-b", "live-peer"),
        ]

        # Re-creating and removing the same DID does not accumulate this
        # runner's old UUID rows, including when its live fleet becomes empty.
        authorized.add("tenant-a")
        await runner._publish_runtime_status("running")
        authorized.clear()
        await runner._publish_runtime_status("running")
        assert await db.fetchall(
            "SELECT agent_id, owner_id FROM scheduler_runtime_status "
            "ORDER BY agent_id, owner_id"
        ) == [
            ("tenant-a", "live-peer"),
            ("tenant-b", "live-peer"),
        ]

        # Any active publisher also bounds orphaned reports from dead runners,
        # regardless of whether that retired DID is in its current scope.
        await db.execute(
            "UPDATE scheduler_runtime_status SET reported_at = ? "
            "WHERE agent_id = ? AND owner_id = ?",
            (
                (
                    now
                    - timedelta(seconds=RUNTIME_STATUS_RETENTION_SECONDS + 1)
                ).isoformat(),
                "tenant-a",
                "live-peer",
            ),
        )
        authorized.add("tenant-b")
        await runner._publish_runtime_status("running")
        assert await db.fetchall(
            "SELECT agent_id, owner_id FROM scheduler_runtime_status "
            "ORDER BY agent_id, owner_id"
        ) == [
            ("tenant-b", "dynamic-runner"),
            ("tenant-b", "live-peer"),
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_runtime_status_migrates_agent_primary_key_without_losing_report(
    tmp_path,
):
    db = await _database(tmp_path / "scheduler-runtime-owner-migration.db")
    try:
        await db.execute(
            """
            CREATE TABLE scheduler_runtime_status (
                agent_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                worker_state TEXT NOT NULL,
                reported_at TEXT NOT NULL,
                last_tick_started_at TEXT,
                last_tick_completed_at TEXT,
                restart_count INTEGER NOT NULL DEFAULT 0,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                last_error_type TEXT,
                configured_enabled_count INTEGER NOT NULL DEFAULT 0,
                enabled_count INTEGER NOT NULL DEFAULT 0,
                executing_count INTEGER NOT NULL DEFAULT 0,
                disabled_count INTEGER NOT NULL DEFAULT 0,
                fenced_count INTEGER NOT NULL DEFAULT 0,
                system_disabled_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        await db.execute(
            """INSERT INTO scheduler_runtime_status
                   (agent_id, owner_id, worker_state, reported_at)
               VALUES ('agent-1', 'legacy-owner', 'running', ?)""",
            (datetime.now(timezone.utc).isoformat(),),
        )

        await ensure_runtime_status_table(db)
        await ensure_runtime_status_table(db)

        key_columns = [
            row[1]
            for row in sorted(
                await db.fetchall("PRAGMA table_info(scheduler_runtime_status)"),
                key=lambda row: int(row[5]),
            )
            if row[5]
        ]
        assert key_columns == ["agent_id", "owner_id"]
        assert await db.fetchone(
            "SELECT owner_id, worker_state FROM scheduler_runtime_status "
            "WHERE agent_id = ?",
            ("agent-1",),
        ) == ("legacy-owner", "running")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_terminal_one_shot_is_history_not_recoverable_disablement(tmp_path):
    db = await _database(tmp_path / "scheduler-terminal-history.db")
    agent = SimpleNamespace(
        did="agent-1",
        agent_id="agent-1",
        storage=SimpleNamespace(db=db),
        signal_registry=None,
        wait_registry=None,
        features={},
    )
    feature = SchedulerFeature(agent)
    now = datetime.now(timezone.utc).isoformat()
    try:
        await feature.initialize()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, schedule_kind, run_at,
                 terminal_status, terminal_at, scheduler_protocol_version)
            VALUES ('completed-deadline', 'agent-1', 'wait_reconcile',
                    '* * * * *', '{}', 0, NULL, ?, 'one_shot', ?,
                    'success', ?, ?)
            """,
            (now, now, now, SCHEDULER_PROTOCOL_VERSION),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=feature._runner._owner_id,
            worker_state="running",
            last_tick_started_at=now,
            last_tick_completed_at=now,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        status = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert status["status"] == "pass"
        assert status["state"] == "running_only_terminal_schedules"
        assert status["terminal_count"] == 1
        assert status["disabled_count"] == 0
        assert status["system_disabled_count"] == 0
        assert status["disabled_reasons"] == {}

        listed = await feature.schedule_list()
        task = listed.data["tasks"][0]
        assert task["disablement"] == {
            "state": "terminal",
            "source": "scheduler_history",
            "reason": "success",
            "recoverable": False,
            "recovery_action": None,
        }
        resume = await feature.schedule_resume("completed-deadline")
        assert resume.status.value == "error"
        assert "one-shot" in resume.error.lower()
    finally:
        await feature.shutdown()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    [
        "execution_log_inconsistent",
        "invalid_idempotency_key",
        ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE,
    ],
)
async def test_one_shot_safety_disablement_is_not_terminal_history(
    tmp_path, terminal_status,
):
    db = await _database(
        tmp_path / f"scheduler-one-shot-safety-{terminal_status}.db"
    )
    agent = SimpleNamespace(
        did="agent-1",
        agent_id="agent-1",
        storage=SimpleNamespace(db=db),
        signal_registry=None,
        wait_registry=None,
        features={},
    )
    feature = SchedulerFeature(agent)
    now = datetime.now(timezone.utc)
    run_at = (now + timedelta(hours=1)).isoformat()
    try:
        await feature.initialize()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, schedule_kind, run_at,
                 terminal_status, terminal_at, scheduler_protocol_version)
            VALUES ('safety-disabled-deadline', 'agent-1', 'wait_reconcile',
                    '* * * * *', '{}', 0, NULL, ?, 'one_shot', ?, ?, ?, ?)
            """,
            (
                now.isoformat(),
                run_at,
                terminal_status,
                now.isoformat(),
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=feature._runner._owner_id,
            worker_state="running",
            last_tick_started_at=now.isoformat(),
            last_tick_completed_at=now.isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        status = await scheduler_status(
            db, agent_id="agent-1", poll_interval=30
        )
        assert status["status"] == "warn"
        assert status["state"] == "system_disabled_schedules"
        assert status["terminal_count"] == 0
        assert status["system_disabled_count"] == 1
        assert status["disabled_reasons"] == {terminal_status: 1}

        listed = await feature.schedule_list()
        disablement = listed.data["tasks"][0]["disablement"]
        assert disablement["state"] == "disabled"
        assert disablement["source"] == "scheduler"
        assert disablement["reason"] == terminal_status

        if terminal_status == "invalid_idempotency_key":
            refused = await feature.schedule_resume("safety-disabled-deadline")
            assert refused.status.value == "error"
            assert "repair or recreate" in refused.error
        elif terminal_status == ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE:
            refused = await feature.schedule_resume("safety-disabled-deadline")
            assert refused.status.value == "error"
            assert "acknowledge_ambiguous_effect=true" in refused.error
            resumed = await feature.schedule_resume(
                "safety-disabled-deadline",
                acknowledge_ambiguous_effect=True,
            )
            assert resumed.status.value == "ok"
        else:
            assert disablement["recoverable"] is False
            assert "recreate the one-shot" in disablement["recovery_action"]
            refused = await feature.schedule_resume("safety-disabled-deadline")
            assert refused.status.value == "error"
            assert "recreate the deadline" in refused.error
            persisted = await db.fetchone(
                "SELECT enabled, terminal_status, terminal_at FROM scheduled_tasks "
                "WHERE id = ?",
                ("safety-disabled-deadline",),
            )
            assert persisted[0] == 0
            assert persisted[1] == "execution_log_inconsistent"
            assert persisted[2]
    finally:
        await feature.shutdown()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("producer_skew_hours", [-12, 12])
async def test_runtime_freshness_uses_database_time_despite_producer_clock_skew(
    monkeypatch, tmp_path, producer_skew_hours,
):
    from kestrel_sovereign.storage import database_clock as scheduler_clock

    real_datetime = datetime

    class SkewedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            current = real_datetime.now(tz or timezone.utc)
            return current + timedelta(hours=producer_skew_hours)

    monkeypatch.setattr(scheduler_clock, "datetime", SkewedDateTime)
    db = await _database(
        tmp_path / f"scheduler-clock-skew-{producer_skew_hours}.db"
    )
    runner = SchedulerRunner(
        db,
        "agent-1",
        lambda *_: None,
        owner_id="skewed",
        poll_interval=0.01,
    )
    try:
        await runner._ensure_tables()

        async def one_tick():
            runner._running = False

        monkeypatch.setattr(runner, "_tick", one_tick)
        runner._running = True
        await runner._loop()

        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id="skewed",
            worker_state="running",
            last_tick_started_at=runner._last_tick_started_at,
            last_tick_completed_at=runner._last_tick_completed_at,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        status = await scheduler_status(
            db, agent_id="agent-1", poll_interval=1, stale_after_ticks=1
        )
        assert status["status"] == "pass"
        assert status["report_age_seconds"] < 1
        assert abs(
            (_parse_reported_at(status["reported_at"]) - real_datetime.now(timezone.utc))
            .total_seconds()
        ) < 5
        assert abs(
            (
                _parse_reported_at(runner._last_tick_started_at)
                - real_datetime.now(timezone.utc)
            ).total_seconds()
        ) < 5
        assert abs(
            (
                _parse_reported_at(runner._last_tick_completed_at)
                - real_datetime.now(timezone.utc)
            ).total_seconds()
        ) < 5
    finally:
        await db.close()


def _parse_reported_at(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.asyncio
async def test_supervisor_restarts_failed_worker_and_keeps_polling(monkeypatch, tmp_path):
    db = await _database(tmp_path / "scheduler-supervision.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="supervision-test",
        poll_interval=0.01,
    )
    completed = asyncio.Event()
    attempts = 0

    async def tick():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic worker death")
        completed.set()

    monkeypatch.setattr(runner, "_tick", tick)
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner.SUPERVISOR_MIN_BACKOFF_SECONDS",
        0.01,
    )
    try:
        await runner.start()
        await asyncio.wait_for(completed.wait(), timeout=2)
        await asyncio.sleep(0.02)
        assert attempts >= 2
        assert runner._worker_restart_count == 1
        assert runner.worker_available is True
        assert isinstance(runner.readiness_failure, RuntimeError)
        report = await scheduler_status(db, agent_id="agent-1", poll_interval=30)
        assert report["worker_state"] == "running"
        assert report["restart_count"] == 1
    finally:
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_supervisor_propagates_its_own_cancellation_without_resurrection(
    monkeypatch, tmp_path,
):
    db = await _database(tmp_path / "scheduler-supervisor-cancel.db")
    runner = SchedulerRunner(
        db,
        "agent-1",
        lambda *_: None,
        owner_id="supervisor-cancel",
        poll_interval=0.01,
    )
    tick_started = asyncio.Event()

    async def blocked_tick():
        tick_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_tick", blocked_tick)
    try:
        await runner.start()
        await asyncio.wait_for(tick_started.wait(), timeout=2)
        supervisor = runner._task
        supervisor.cancel("external lifecycle cancellation")
        with pytest.raises(asyncio.CancelledError):
            await supervisor
        await asyncio.sleep(0)

        assert supervisor.done()
        assert runner._running is False
        assert runner._worker_task is None
        assert runner._worker_restart_count == 0
        assert runner.readiness_failure is None
    finally:
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_closed_storage_terminates_supervisor_without_restart_loop(
    monkeypatch, tmp_path,
):
    db = await _database(tmp_path / "scheduler-closed-storage.db")
    runner = SchedulerRunner(
        db,
        "agent-1",
        lambda *_: None,
        owner_id="closed-storage",
        poll_interval=0.01,
    )
    tick_started = asyncio.Event()
    release_tick = asyncio.Event()

    async def fail_after_close():
        tick_started.set()
        await release_tick.wait()
        await db.fetchval("SELECT 1")

    monkeypatch.setattr(runner, "_tick", fail_after_close)
    await runner.start()
    supervisor = runner._task
    sqlite_connection = db.backend._connection
    sqlite_worker = getattr(sqlite_connection, "_thread", sqlite_connection)
    await asyncio.wait_for(tick_started.wait(), timeout=2)
    await db.close()
    release_tick.set()
    await asyncio.wait_for(asyncio.shield(supervisor), timeout=2)

    assert runner._running is False
    assert runner._runtime_worker_state == "storage_unavailable"
    assert runner._worker_restart_count == 0
    await runner.stop()
    assert runner._task is None
    assert runner._worker_task is None
    assert runner._telemetry_task is None
    is_alive = getattr(sqlite_worker, "is_alive", None)
    assert not callable(is_alive) or not is_alive()


@pytest.mark.asyncio
async def test_runtime_telemetry_failure_cannot_prevent_or_kill_polling(
    monkeypatch, tmp_path,
):
    db = await _database(tmp_path / "scheduler-telemetry-isolation.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="telemetry-isolation",
        poll_interval=0.01,
    )
    completed = asyncio.Event()

    async def tick():
        completed.set()

    async def reporting_failure(_state):
        raise RuntimeError("synthetic telemetry write failure")

    monkeypatch.setattr(runner, "_tick", tick)
    monkeypatch.setattr(runner, "_publish_runtime_status", reporting_failure)
    try:
        await runner.start()
        await asyncio.wait_for(completed.wait(), timeout=2)
        await asyncio.sleep(0.02)
        assert runner._arm_requested is True
        assert runner._running is True
        assert runner.worker_available is True
        assert runner._worker_restart_count == 0
        assert runner.readiness_failure is None
    finally:
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_slow_runtime_telemetry_remains_owned_without_cancellation(
    monkeypatch, tmp_path,
):
    db = await _database(tmp_path / "scheduler-slow-telemetry.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="slow-telemetry",
    )
    publish_started = asyncio.Event()
    release_publish = asyncio.Event()

    async def slow_publish(_state):
        publish_started.set()
        await release_publish.wait()

    monkeypatch.setattr(runner, "_publish_runtime_status", slow_publish)
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner."
        "RUNTIME_STATUS_PUBLISH_TIMEOUT_SECONDS",
        0.01,
    )

    try:
        await runner._publish_runtime_status_best_effort()
        await asyncio.wait_for(publish_started.wait(), timeout=1)
        publish_task = runner._runtime_status_publish_task

        assert publish_task is not None
        assert publish_task.done() is False
        assert publish_task.cancelled() is False

        release_publish.set()
        await asyncio.wait_for(publish_task, timeout=1)
        assert publish_task.cancelled() is False
    finally:
        release_publish.set()
        pending = runner._runtime_status_publish_task
        if pending is not None:
            await pending
        await db.close()


@pytest.mark.asyncio
async def test_stop_drains_or_cancels_owned_runtime_telemetry(
    monkeypatch, tmp_path,
):
    db = await _database(tmp_path / "scheduler-stop-telemetry.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="stop-telemetry",
    )
    publish_started = asyncio.Event()
    release_publish = asyncio.Event()

    async def slow_publish(_state):
        publish_started.set()
        await release_publish.wait()

    monkeypatch.setattr(runner, "_publish_runtime_status", slow_publish)
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner."
        "RUNTIME_STATUS_PUBLISH_TIMEOUT_SECONDS",
        0.01,
    )

    try:
        await runner._publish_runtime_status_best_effort()
        await asyncio.wait_for(publish_started.wait(), timeout=1)
        publish_task = runner._runtime_status_publish_task

        await runner.stop()

        assert publish_task is not None
        assert publish_task.done() is True
        assert publish_task.cancelled() is True
        assert runner._runtime_status_publish_task is None
    finally:
        release_publish.set()
        await db.close()


@pytest.mark.asyncio
async def test_stop_drains_pending_telemetry_then_publishes_stopped(
    monkeypatch, tmp_path,
):
    db = await _database(tmp_path / "scheduler-stop-drain-telemetry.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="stop-drain-telemetry",
    )
    publish_started = asyncio.Event()
    release_publish = asyncio.Event()
    published = []

    async def delayed_publish(state):
        if state != "stopped":
            publish_started.set()
            await release_publish.wait()
        published.append(state)

    monkeypatch.setattr(runner, "_publish_runtime_status", delayed_publish)
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner."
        "RUNTIME_STATUS_PUBLISH_TIMEOUT_SECONDS",
        0.05,
    )

    try:
        await runner._publish_runtime_status_best_effort("running")
        await asyncio.wait_for(publish_started.wait(), timeout=1)
        asyncio.get_running_loop().call_later(0.01, release_publish.set)
        started = asyncio.get_running_loop().time()

        await runner.stop()

        elapsed = asyncio.get_running_loop().time() - started
        assert published == ["running", "stopped"]
        assert runner._runtime_status_publish_task is None
        assert elapsed < 0.2
    finally:
        release_publish.set()
        await db.close()


@pytest.mark.asyncio
async def test_stop_cancellation_does_not_wait_out_publish_deadline(
    monkeypatch, tmp_path,
):
    db = await _database(tmp_path / "scheduler-stop-cancel-telemetry.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="stop-cancel-telemetry",
    )
    publish_started = asyncio.Event()

    async def blocked_publish():
        publish_started.set()
        await asyncio.Event().wait()

    publish_task = asyncio.create_task(blocked_publish())
    runner._runtime_status_publish_task = publish_task
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner."
        "RUNTIME_STATUS_PUBLISH_TIMEOUT_SECONDS",
        0.5,
    )

    try:
        await asyncio.wait_for(publish_started.wait(), timeout=1)
        stop_task = asyncio.create_task(runner.stop())
        await asyncio.sleep(0.01)
        started = asyncio.get_running_loop().time()
        stop_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await stop_task

        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.2
        assert publish_task.done() is True
        assert publish_task.cancelled() is True
        assert runner._runtime_status_publish_task is None
    finally:
        if not publish_task.done():
            publish_task.cancel()
            with suppress(asyncio.CancelledError):
                await publish_task
        await db.close()


@pytest.mark.asyncio
async def test_long_running_tick_keeps_independent_worker_heartbeat(
    monkeypatch, tmp_path,
):
    db = await _database(tmp_path / "scheduler-long-tick.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="long-tick",
        poll_interval=0.01,
    )
    tick_started = asyncio.Event()
    release_tick = asyncio.Event()

    async def long_tick():
        tick_started.set()
        await release_tick.wait()

    monkeypatch.setattr(runner, "_tick", long_tick)
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner."
        "RUNTIME_STATUS_MIN_HEARTBEAT_SECONDS",
        0.05,
    )
    try:
        await runner.start()
        await asyncio.wait_for(tick_started.wait(), timeout=2)
        # The ordinary tick-boundary report is now stale against this explicit
        # one-second diagnostic window. Only the independent heartbeat can
        # keep the report current while the claimed work is still executing.
        await asyncio.sleep(1.1)
        status = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=1,
            stale_after_ticks=1,
            expected_owner_id=runner._owner_id,
        )
        assert status["status"] == "pass"
        assert status["state"] == "running_zero_schedules"
        assert status["tick_in_progress"] is True
        assert status["tick_stalled"] is False
        assert status["report_age_seconds"] < 0.5
        assert runner.worker_available is True
    finally:
        release_tick.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_live_claimed_execution_is_not_a_stalled_poller(
    monkeypatch,
    tmp_path,
):
    db = await _database(tmp_path / "scheduler-long-claimed-execution.db")
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()

    async def execute(_task_name, _args):
        execution_started.set()
        await release_execution.wait()

    runner = SchedulerRunner(
        db,
        "agent-1",
        execute,
        owner_id="long-claimed-execution",
        poll_interval=0.01,
        lease_seconds=1,
    )
    runner._tick_in_progress_limit_seconds = 0.05
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner."
        "RUNTIME_STATUS_MIN_HEARTBEAT_SECONDS",
        0.01,
    )
    try:
        await runner._ensure_tables()
        now = await scheduler_database_clock(db)
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, scheduler_protocol_version)
            VALUES ('long-running', 'agent-1', 'wait_reconcile', '* * * * *',
                    '{}', 1, ?, ?, ?)
            """,
            (
                (now - timedelta(seconds=1)).isoformat(),
                now.isoformat(),
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        await runner.arm()
        await asyncio.wait_for(execution_started.wait(), timeout=2)

        # Exercise the production state after both the in-process and durable
        # hard bounds, without spending real seconds waiting in this test.
        runner._tick_started_monotonic = time.monotonic() - 1
        runner._last_tick_started_at = (now - timedelta(seconds=5)).isoformat()
        await runner._publish_runtime_status_best_effort()

        assert runner._active_claimed_execution_count == 1
        assert runner.tick_stalled is False
        assert runner.worker_available is True
        status = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=1,
            stale_after_ticks=1,
            lease_seconds=1,
            expected_owner_id=runner._owner_id,
        )
        assert status["tick_age_seconds"] >= 5
        assert status["tick_in_progress_limit_seconds"] == 2
        assert status["tick_stalled"] is False
        assert status["worker_state"] == "executing"
        assert status["executing_count"] == 1
        assert status["status"] == "pass"

        agent = SimpleNamespace(
            did="agent-1",
            features={
                "SchedulerFeature": SimpleNamespace(
                    enabled=True,
                    _runner=runner,
                    _initialized_monotonic=0.0,
                )
            },
        )
        health = await check_scheduler_liveness(agent, db)
        assert health["status"] == "pass"
        assert "executing schedule" in health["message"]

        # A live execution exempts only itself. Work that became due after the
        # current batch was selected still surfaces once the bounded in-flight
        # grace expires.
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, misfire_grace_seconds,
                 scheduler_protocol_version)
            VALUES ('starved', 'agent-1', 'wait_reconcile', '* * * * *',
                    '{}', 1, ?, ?, 0, ?)
            """,
            (
                (now - timedelta(seconds=5)).isoformat(),
                now.isoformat(),
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        await runner._publish_runtime_status_best_effort()
        overdue = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=1,
            stale_after_ticks=1,
            lease_seconds=1,
            expected_owner_id=runner._owner_id,
        )
        assert overdue["worker_state"] == "executing"
        assert overdue["tick_stalled"] is False
        assert overdue["state"] == "overdue"
        assert overdue["status"] == "fail"
    finally:
        release_execution.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_blocked_tick_fails_closed_and_recovers_after_completion(
    monkeypatch,
    tmp_path,
):
    db = await _database(tmp_path / "scheduler-blocked-tick.db")
    runner = SchedulerRunner(
        db,
        "agent-1",
        lambda *_: None,
        owner_id="blocked-tick",
        poll_interval=0.01,
    )
    runner._tick_in_progress_limit_seconds = 0.05
    tick_started = asyncio.Event()
    release_tick = asyncio.Event()
    attempts = 0

    async def blocked_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            tick_started.set()
            await release_tick.wait()

    async def wait_for_worker_state(expected):
        while True:
            row = await db.fetchone(
                "SELECT worker_state FROM scheduler_runtime_status "
                "WHERE agent_id = ? AND owner_id = ?",
                ("agent-1", runner._owner_id),
            )
            if row == (expected,):
                return
            await asyncio.sleep(0.01)

    async def wait_for_recovered_live_view():
        """Wait for one instant where the live view is coherently recovered.

        ``tick_stalled`` is derived from the clock on every read: it re-arms at
        the top of each tick and reads True again whenever the tick in flight
        outlives ``_tick_in_progress_limit_seconds`` — squeezed to 50ms above so
        the stall under test is detectable quickly. Every later tick still makes
        two ``scheduler_database_clock`` round trips, so on a loaded runner an
        ordinary healthy tick crosses that 50ms and the property legitimately
        reads True again *after* the durable row has already reported
        "running". Sampling it once, at an instant this test does not control,
        asserted a scheduling race rather than recovery: measured True in 33/60
        samples with 80ms round trips and 0/60 when they are fast, which is why
        it passed everywhere except one contended CI runner (#3099).

        So wait for the event. Recovery is "the runner returns to a coherent
        healthy view", and a regression that never clears the stall never
        presents one — the bound below is a hang bound, not a coordination
        window.
        """
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            # Read both halves at one instant: the pair is the claim, and
            # reading them apart re-opens the same race in miniature.
            if not runner.tick_stalled and runner.worker_available:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(
            "runner never presented a recovered live view after the blocked "
            f"tick completed (tick_stalled={runner.tick_stalled}, "
            f"worker_available={runner.worker_available})"
        )

    monkeypatch.setattr(runner, "_tick", blocked_once)
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner."
        "RUNTIME_STATUS_MIN_HEARTBEAT_SECONDS",
        0.01,
    )
    try:
        await runner.start()
        await asyncio.wait_for(tick_started.wait(), timeout=2)
        await asyncio.wait_for(wait_for_worker_state("stalled"), timeout=2)

        assert runner.tick_stalled is True
        assert runner.worker_available is False
        stalled = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=1,
            expected_owner_id=runner._owner_id,
        )
        assert stalled["state"] == "tick_stalled"
        assert stalled["status"] == "fail"

        release_tick.set()
        await asyncio.wait_for(wait_for_worker_state("running"), timeout=2)
        await wait_for_recovered_live_view()
    finally:
        release_tick.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_initial_database_clock_stall_is_covered_by_tick_watchdog(
    monkeypatch,
    tmp_path,
):
    db = await _database(tmp_path / "scheduler-clock-stall.db")
    runner = SchedulerRunner(
        db,
        "agent-1",
        lambda *_: None,
        owner_id="clock-stall",
        poll_interval=0.01,
    )
    runner._tick_in_progress_limit_seconds = 0.05
    clock_read_started = asyncio.Event()
    release_clock = asyncio.Event()
    clock_reads = 0

    async def block_first_tick_clock(clock_db):
        nonlocal clock_reads
        clock_reads += 1
        if clock_reads == 1:
            return await scheduler_database_clock(clock_db)
        clock_read_started.set()
        await release_clock.wait()
        return await scheduler_database_clock(clock_db)

    async def wait_for_worker_state(expected):
        while True:
            row = await db.fetchone(
                "SELECT worker_state FROM scheduler_runtime_status "
                "WHERE agent_id = ? AND owner_id = ?",
                ("agent-1", runner._owner_id),
            )
            if row == (expected,):
                return
            await asyncio.sleep(0.01)

    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner.scheduler_database_clock",
        block_first_tick_clock,
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner."
        "RUNTIME_STATUS_MIN_HEARTBEAT_SECONDS",
        0.01,
    )
    try:
        await runner.start()
        await asyncio.wait_for(clock_read_started.wait(), timeout=2)
        await asyncio.wait_for(wait_for_worker_state("stalled"), timeout=2)

        assert runner._last_tick_started_at is None
        assert runner._tick_started_monotonic is not None
        assert runner.tick_stalled is True
        assert runner.worker_available is False
        status = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=1,
            expected_owner_id=runner._owner_id,
        )
        assert status["worker_state"] == "stalled"
        assert status["state"] == "tick_stalled"
        assert status["status"] == "fail"
    finally:
        release_clock.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_live_claim_is_executing_not_protocol_fenced(tmp_path):
    db = await _database(tmp_path / "scheduler-live-claim.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="claim-owner"
    )
    try:
        await runner._ensure_tables()
        now = datetime.now(timezone.utc)
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, lease_owner, lease_expires_at,
                 scheduler_protocol_version, scheduler_claim_fenced)
            VALUES (?, ?, ?, ?, '{}', 0, ?, ?, ?, ?, ?, 1)
            """,
            (
                "executing",
                "agent-1",
                "wait_reconcile",
                "* * * * *",
                now.isoformat(),
                now.isoformat(),
                runner._owner_id,
                (now + timedelta(minutes=2)).isoformat(),
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=runner._owner_id,
            worker_state="running",
            last_tick_started_at=now.isoformat(),
            last_tick_completed_at=None,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        status = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            expected_owner_id=runner._owner_id,
        )
        assert status["status"] == "pass"
        assert status["state"] == "running"
        assert status["configured_enabled_count"] == 1
        assert status["enabled_count"] == 0
        assert status["executing_count"] == 1
        assert status["fenced_count"] == 0

        agent = SimpleNamespace(
            did="agent-1",
            agent_id="agent-1",
            storage=SimpleNamespace(db=db),
            signal_registry=None,
            wait_registry=None,
            features={},
        )
        feature = SchedulerFeature(agent)
        feature._db = db
        feature._agent_id = "agent-1"
        feature._runner = runner
        feature._initialized_monotonic = 0.0
        listed = await feature.schedule_list()
        task = next(item for item in listed.data["tasks"] if item["id"] == "executing")
        assert task["disablement"]["state"] == "executing"
        assert task["disablement"]["source"] == "scheduler"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_prior_boot_report_uses_startup_grace_for_current_owner(tmp_path):
    db = await _database(tmp_path / "scheduler-restart-grace.db")
    prior = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="prior-owner"
    )
    current = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="current-owner"
    )
    try:
        await prior._ensure_tables()
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=prior._owner_id,
            worker_state="stopped",
            last_tick_started_at=None,
            last_tick_completed_at=None,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        during_grace = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            startup_grace_remaining=60,
            expected_owner_id=current._owner_id,
        )
        assert during_grace["state"] == "awaiting_telemetry"
        assert during_grace["status"] == "warn"
        assert during_grace["worker_state"] == "stopped"
        assert during_grace["current_owner_telemetry_received"] is False

        after_grace = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            expected_owner_id=current._owner_id,
        )
        assert after_grace["state"] == "no_current_telemetry"
        assert after_grace["status"] == "fail"

        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=current._owner_id,
            worker_state="restarting",
            last_tick_started_at=None,
            last_tick_completed_at=None,
            restart_count=1,
            consecutive_failures=1,
            last_error_type="RuntimeError",
        )
        current_failure = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            startup_grace_remaining=60,
            expected_owner_id=current._owner_id,
        )
        assert current_failure["state"] == "restarting"
        assert current_failure["status"] == "fail"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_pre_arm_startup_grace_expires_from_feature_initialization(tmp_path):
    db = await _database(tmp_path / "scheduler-arm-grace.db")
    agent = SimpleNamespace(
        did="agent-1",
        agent_id="agent-1",
        storage=SimpleNamespace(db=db),
        signal_registry=None,
        wait_registry=None,
        features={},
    )
    feature = SchedulerFeature(agent)
    try:
        await feature.initialize()
        feature._initialized_monotonic = 0.0

        # A swallowed on_agent_ready failure cannot leave this pre-arm state
        # warned forever after its finite initialization grace expires.
        before_ready = await check_scheduler_liveness(
            SimpleNamespace(
                did="agent-1",
                features={"SchedulerFeature": feature},
            ),
            db,
        )
        assert before_ready["status"] == "fail"
        assert before_ready["details"]["state"] == "no_telemetry"
        parameters = scheduler_status_parameters(feature)
        assert parameters["startup_grace_remaining"] == 0.0

        # Once arm is requested, its own timestamp is the finite grace anchor.
        feature._runner._arm_requested = True
        feature._runner._arm_requested_monotonic = 0.0
        feature._runner._arm_requested_at = datetime.now(timezone.utc).isoformat()
        after_arm_grace = await check_scheduler_liveness(
            SimpleNamespace(
                did="agent-1",
                features={"SchedulerFeature": feature},
            ),
            db,
        )
        assert after_arm_grace["status"] == "fail"
        assert after_arm_grace["details"]["state"] == "no_telemetry"
    finally:
        await feature.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_same_owner_pre_arm_report_is_not_current_worker_telemetry(tmp_path):
    db = await _database(tmp_path / "scheduler-same-owner-rearm.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="retained-owner"
    )
    try:
        await runner._ensure_tables()
        prior_reported_at = datetime.now(timezone.utc)
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=runner._owner_id,
            worker_state="stopped",
            last_tick_started_at=None,
            last_tick_completed_at=None,
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        arm_started_at = (prior_reported_at + timedelta(seconds=1)).isoformat()

        during_grace = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            startup_grace_remaining=60,
            expected_owner_id=runner._owner_id,
            expected_started_at=arm_started_at,
        )
        assert during_grace["state"] == "awaiting_telemetry"
        assert during_grace["status"] == "warn"
        assert during_grace["report_predates_expected_start"] is True

        after_grace = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            expected_owner_id=runner._owner_id,
            expected_started_at=arm_started_at,
        )
        assert after_grace["state"] == "no_current_telemetry"
        assert after_grace["status"] == "fail"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_overdue_health_respects_misfire_grace_and_active_tick(tmp_path):
    db = await _database(tmp_path / "scheduler-overdue-grace.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="overdue-owner"
    )
    try:
        await runner._ensure_tables()
        now = datetime.now(timezone.utc)
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, scheduler_protocol_version)
            VALUES ('queued', 'agent-1', 'wait_reconcile', '* * * * *',
                    '{}', 1, ?, ?, ?)
            """,
            (
                (now - timedelta(seconds=120)).isoformat(),
                now.isoformat(),
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=runner._owner_id,
            worker_state="running",
            last_tick_started_at=now.isoformat(),
            last_tick_completed_at=now.isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        within_misfire_grace = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            expected_owner_id=runner._owner_id,
        )
        assert within_misfire_grace["status"] == "pass"
        assert within_misfire_grace["state"] == "running"

        very_late = (now - timedelta(seconds=700)).isoformat()
        await db.execute(
            "UPDATE scheduled_tasks SET next_run_at = ?, "
            "misfire_grace_seconds = 0 WHERE id = 'queued'",
            (very_late,),
        )
        tick_started = (await scheduler_database_clock(db)).isoformat()
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=runner._owner_id,
            worker_state="running",
            last_tick_started_at=tick_started,
            last_tick_completed_at=now.isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        queued_behind_semaphore = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            expected_owner_id=runner._owner_id,
        )
        assert queued_behind_semaphore["tick_in_progress"] is True
        assert queued_behind_semaphore["status"] == "pass"
        assert queued_behind_semaphore["state"] == "running"

        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=runner._owner_id,
            worker_state="running",
            last_tick_started_at=tick_started,
            last_tick_completed_at=datetime.now(timezone.utc).isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )
        genuinely_overdue = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            expected_owner_id=runner._owner_id,
        )
        assert genuinely_overdue["tick_in_progress"] is False
        assert genuinely_overdue["status"] == "fail"
        assert genuinely_overdue["state"] == "overdue"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_wedged_tick_cannot_hide_overdue_queue_forever(tmp_path):
    db = await _database(tmp_path / "scheduler-wedged-tick.db")
    runner = SchedulerRunner(
        db,
        "agent-1",
        lambda *_: None,
        owner_id="wedged-tick",
        lease_seconds=120,
    )
    now = datetime.now(timezone.utc)
    try:
        await runner._ensure_tables()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, misfire_grace_seconds,
                 scheduler_protocol_version)
            VALUES ('overdue', 'agent-1', 'wait_reconcile', '* * * * *',
                    '{}', 1, ?, ?, 0, ?)
            """,
            (
                (now - timedelta(minutes=10)).isoformat(),
                now.isoformat(),
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=runner._owner_id,
            worker_state="running",
            last_tick_started_at=(now - timedelta(minutes=5)).isoformat(),
            last_tick_completed_at=(now - timedelta(minutes=6)).isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        status = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            lease_seconds=120,
            expected_owner_id=runner._owner_id,
        )
        assert status["telemetry_received"] is True
        assert status["report_age_seconds"] < 1
        assert status["tick_in_progress"] is False
        assert status["tick_in_progress_limit_seconds"] == 210
        assert status["tick_stalled"] is True
        assert status["state"] == "tick_stalled"
        assert status["status"] == "fail"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_wedged_tick_with_zero_schedules_fails_detailed_health(tmp_path):
    db = await _database(tmp_path / "scheduler-wedged-empty-tick.db")
    runner = SchedulerRunner(
        db,
        "agent-1",
        lambda *_: None,
        owner_id="wedged-empty-tick",
        lease_seconds=120,
    )
    try:
        await runner._ensure_tables()
        now = await scheduler_database_clock(db)
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=runner._owner_id,
            worker_state="running",
            last_tick_started_at=(now - timedelta(seconds=211)).isoformat(),
            last_tick_completed_at=(now - timedelta(seconds=212)).isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        status = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            lease_seconds=120,
            expected_owner_id=runner._owner_id,
        )
        assert status["configured_enabled_count"] == 0
        assert status["tick_in_progress_limit_seconds"] == 210
        assert status["tick_stalled"] is True
        assert status["state"] == "tick_stalled"
        assert status["status"] == "fail"

        agent = SimpleNamespace(
            did="agent-1",
            features={
                "SchedulerFeature": SimpleNamespace(
                    enabled=True,
                    _runner=runner,
                    _initialized_monotonic=0.0,
                )
            },
        )
        health = await check_scheduler_liveness(agent, db)
        assert health["status"] == "fail"
        assert health["details"]["state"] == "tick_stalled"
        assert "liveness bound" in health["message"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_overdue_age_uses_row_with_earliest_misfire_deadline(tmp_path):
    db = await _database(tmp_path / "scheduler-mixed-grace-overdue.db")
    runner = SchedulerRunner(
        db, "agent-1", lambda *_: None, owner_id="mixed-grace"
    )
    try:
        await runner._ensure_tables()
        now = await scheduler_database_clock(db)
        await db.execute_many(
            """INSERT INTO scheduled_tasks
                   (id, agent_id, task_name, cron_expression, args_json,
                    enabled, next_run_at, created_at, misfire_grace_seconds,
                    scheduler_protocol_version)
               VALUES (?, 'agent-1', 'wait_reconcile', '* * * * *', '{}',
                       1, ?, ?, ?, ?)""",
            [
                (
                    "older-inside-long-grace",
                    (now - timedelta(minutes=35)).isoformat(),
                    now.isoformat(),
                    3600,
                    SCHEDULER_PROTOCOL_VERSION,
                ),
                (
                    "newer-past-short-grace",
                    (now - timedelta(minutes=5)).isoformat(),
                    now.isoformat(),
                    0,
                    SCHEDULER_PROTOCOL_VERSION,
                ),
            ],
        )
        await emit_runtime_status(
            db,
            agent_ids=("agent-1",),
            owner_id=runner._owner_id,
            worker_state="running",
            last_tick_started_at=now.isoformat(),
            last_tick_completed_at=now.isoformat(),
            restart_count=0,
            consecutive_failures=0,
            last_error_type=None,
        )

        status = await scheduler_status(
            db,
            agent_id="agent-1",
            poll_interval=30,
            expected_owner_id=runner._owner_id,
        )

        assert status["state"] == "overdue"
        assert 240 <= status["overdue_seconds"] <= 360
        overdue_run = _parse_reported_at(
            status["oldest_unclaimed_overdue_run_at"]
        )
        assert abs((overdue_run - (now - timedelta(minutes=5))).total_seconds()) < 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_schedule_list_survives_unavailable_runtime_telemetry(tmp_path):
    db = await _database(tmp_path / "scheduler-list-telemetry.db")
    agent = SimpleNamespace(
        did="agent-1",
        agent_id="agent-1",
        storage=SimpleNamespace(db=db),
        signal_registry=None,
        wait_registry=None,
        features={},
    )
    feature = SchedulerFeature(agent)
    try:
        await feature.initialize()
        await db.execute("DROP TABLE scheduler_runtime_status")
        listed = await feature.schedule_list()
        assert listed.status.value == "ok"
        assert listed.data["tasks"] == []
        assert listed.data["scheduler_status"]["state"] == "inspection_failed"
        assert listed.data["scheduler_status"]["status"] == "fail"
        assert listed.data["scheduler_status"]["enabled_count"] is None
    finally:
        await ensure_runtime_status_table(db)
        await feature.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_restart_scope_and_bounded_escalation_keep_blocker_evidence(tmp_path):
    db = await _database(tmp_path / "restart-scope.db")
    requester = SimpleNamespace(
        did="agent-1",
        agent_id="agent-1",
        storage=SimpleNamespace(db=db),
        dispatcher=SimpleNamespace(),
        signal_registry=None,
        features={},
        _active_request_ids=set(),
        _background_tasks=set(),
        emit_event=AsyncMock(),
    )
    sibling = SimpleNamespace(
        did="agent-2",
        name="Sibling",
        dispatcher=SimpleNamespace(),
        _active_request_ids={"busy"},
        _background_tasks=set(),
    )
    requester._cohosted_agents_provider = lambda: [requester, sibling]
    feature = RestartCoordinatorFeature(requester)
    try:
        await feature.initialize()
        fresh = await insert_request(
            db, requested_by_agent="agent-1", reason="requester scoped"
        )
        fresh, decision = await feature._evaluate_and_track_safety(fresh)
        assert decision["safe"] is False
        assert decision["blocker"]["scope"] == "cohosted_agent"
        assert decision["blocker"]["count"] is None
        assert decision["blocker"]["oldest_age_seconds"] is None
        assert "1 active request" not in decision["reason"]
        assert fresh.first_blocked_at
        assert decision["request_age_seconds"] < 5
        assert decision["deferral_age_seconds"] < 5

        aged = (
            datetime.now(timezone.utc)
            - timedelta(seconds=MAX_IDLE_ONLY_DEFERRAL_SECONDS + 1)
        ).isoformat()
        await db.execute(
            "UPDATE restart_requests SET first_blocked_at = ? WHERE id = ?",
            (aged, fresh.id),
        )
        fresh.first_blocked_at = aged
        escalated = feature._evaluate_safety(
            fresh, database_now=await scheduler_database_clock(db)
        )
        assert escalated["safe"] is True
        assert escalated["escalated"] is True
        assert escalated["blocker"]["scope"] == "cohosted_agent"
        assert escalated["blocker"]["kind"] == "active_requests"
        assert escalated["deferral_age_seconds"] >= MAX_IDLE_ONLY_DEFERRAL_SECONDS

        await feature._emit_status_event(
            fresh,
            state="escalated",
            deferral_reason=escalated["reason"],
            blocker=escalated["blocker"],
            request_age_seconds=escalated["request_age_seconds"],
            deferral_age_seconds=escalated["deferral_age_seconds"],
            escalated=True,
        )
        event = await db.fetchone(
            """
            SELECT state, payload_json
            FROM restart_status_events
            WHERE request_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (fresh.id,),
        )
        payload = json.loads(event[1])
        assert event[0] == "escalated"
        assert payload["escalated"] is True
        assert payload["blocker"]["kind"] == "active_requests"
        assert payload["request_age_seconds"] < 5
        assert payload["deferral_age_seconds"] >= MAX_IDLE_ONLY_DEFERRAL_SECONDS
        requester.emit_event.assert_awaited_once()

        sibling._active_request_ids = set()
        fresh, idle = await feature._evaluate_and_track_safety(fresh)
        assert idle["safe"] is True
        assert fresh.first_blocked_at == ""
        assert (await get_request(db, fresh.id)).first_blocked_at == ""
        sibling._active_request_ids = {"busy"}

        # A migrated backlog row starts a fresh deferral clock on this boot and
        # cannot escalate until its one-time acknowledgement is recorded.
        legacy = await insert_request(
            db, requested_by_agent="agent-1", reason="pre-upgrade backlog"
        )
        ancient_request = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).isoformat()
        await db.execute(
            "UPDATE restart_requests SET requested_at = ?, "
            "first_blocked_at = '', escalation_acknowledged = 0 WHERE id = ?",
            (ancient_request, legacy.id),
        )
        legacy = await get_request(db, legacy.id)
        legacy, initial = await feature._evaluate_and_track_safety(legacy)
        assert initial["safe"] is False
        assert initial["request_age_seconds"] > 60 * 60 * 24
        assert initial["deferral_age_seconds"] < 5
        await db.execute(
            "UPDATE restart_requests SET first_blocked_at = ? WHERE id = ?",
            (aged, legacy.id),
        )
        legacy = await get_request(db, legacy.id)
        needs_ack = feature._evaluate_safety(
            legacy, database_now=await scheduler_database_clock(db)
        )
        assert needs_ack["safe"] is False
        assert "acknowledgement required" in needs_ack["reason"]
        acknowledged = await feature.acknowledge_restart_escalation(legacy.id)
        assert acknowledged.data["acknowledged"] is True
        legacy = await get_request(db, legacy.id)
        assert feature._evaluate_safety(
            legacy, database_now=await scheduler_database_clock(db)
        )["safe"] is True
    finally:
        await feature.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_restart_deferral_escalation_uses_database_time_under_host_skew(
    monkeypatch, tmp_path,
):
    from kestrel_sovereign.features.restart_coordinator import (
        feature as restart_feature_module,
    )
    from kestrel_sovereign.features.restart_coordinator import (
        store as restart_store_module,
    )

    db = await _database(tmp_path / "restart-deferral-clock.db")
    agent = SimpleNamespace(
        did="agent-1",
        agent_id="agent-1",
        storage=SimpleNamespace(db=db),
        dispatcher=SimpleNamespace(),
        signal_registry=None,
        features={},
        _active_request_ids={"busy"},
        _background_tasks=set(),
        emit_event=AsyncMock(),
    )
    feature = RestartCoordinatorFeature(agent)
    real_datetime = datetime

    class FastHostDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz or timezone.utc) + timedelta(hours=12)

    class SlowHostDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz or timezone.utc) - timedelta(hours=12)

    try:
        await feature.initialize()
        request = await insert_request(
            db, requested_by_agent="agent-1", reason="clock skew"
        )
        monkeypatch.setattr(restart_feature_module, "datetime", FastHostDateTime)
        monkeypatch.setattr(restart_store_module, "datetime", FastHostDateTime)

        request, initial = await feature._evaluate_and_track_safety(request)
        assert initial["safe"] is False
        assert initial["escalated"] is False
        assert initial["deferral_age_seconds"] < 5
        database_now = await scheduler_database_clock(db)
        blocked_at = datetime.fromisoformat(request.first_blocked_at)
        assert abs((database_now - blocked_at).total_seconds()) < 5

        aged = (
            database_now
            - timedelta(seconds=MAX_IDLE_ONLY_DEFERRAL_SECONDS + 1)
        ).isoformat()
        await db.execute(
            "UPDATE restart_requests SET first_blocked_at = ? WHERE id = ?",
            (aged, request.id),
        )
        request = await get_request(db, request.id)
        monkeypatch.setattr(restart_feature_module, "datetime", SlowHostDateTime)
        monkeypatch.setattr(restart_store_module, "datetime", SlowHostDateTime)

        request, escalated = await feature._evaluate_and_track_safety(request)
        assert escalated["safe"] is True
        assert escalated["escalated"] is True
        assert (
            escalated["deferral_age_seconds"]
            >= MAX_IDLE_ONLY_DEFERRAL_SECONDS
        )
    finally:
        await feature.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_ambiguous_legacy_disablement_is_visible_and_explicitly_recoverable(tmp_path):
    db = await _database(tmp_path / "scheduler-recovery.db")
    agent = SimpleNamespace(
        did="agent-1",
        agent_id="agent-1",
        storage=SimpleNamespace(db=db),
        signal_registry=None,
        wait_registry=None,
        features={},
    )
    feature = SchedulerFeature(agent)
    try:
        await feature.initialize()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, terminal_status,
                 scheduler_protocol_version)
            VALUES ('legacy-ambiguous', 'agent-1', 'restart_coordinator',
                    '* * * * *', '{}', 0, ?, ?, ?, ?)
            """,
            (now, now, ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE, SCHEDULER_PROTOCOL_VERSION),
        )

        listed = await feature.schedule_list()
        row = next(
            task for task in listed.data["tasks"]
            if task["id"] == "legacy-ambiguous"
        )
        assert row["enabled"] is False
        assert row["disablement"]["source"] == "scheduler"
        assert row["disablement"]["reason"] == ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE
        assert "schedule_resume" in row["disablement"]["recovery_action"]

        refused = await feature.schedule_resume("legacy-ambiguous")
        assert refused.status.value == "error"
        assert (
            refused.data["disabled_reason"]
            == ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE
        )
        assert await db.fetchone(
            "SELECT enabled, terminal_status FROM scheduled_tasks WHERE id = ?",
            ("legacy-ambiguous",),
        ) == (0, ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE)

        recovered = await feature.schedule_resume(
            "legacy-ambiguous", acknowledge_ambiguous_effect=True
        )
        assert recovered.data["recovered_from"] == ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE
        assert await db.fetchone(
            "SELECT enabled, terminal_status FROM scheduled_tasks WHERE id = ?",
            ("legacy-ambiguous",),
        ) == (1, None)
    finally:
        await feature.shutdown()
        await db.close()
