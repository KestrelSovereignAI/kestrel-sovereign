"""Shared scheduler reliability states across worker, health, and recovery."""

from __future__ import annotations

import asyncio
import json
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
    emit_runtime_status,
    ensure_runtime_status_table,
    scheduler_status,
    scheduler_status_parameters,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


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
        assert status["report_age_seconds"] < 0.5
        assert runner.worker_available is True
    finally:
        release_tick.set()
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
async def test_startup_grace_begins_when_standalone_runner_is_armed(tmp_path):
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

        # Feature-heavy boot can spend longer than three poll intervals before
        # Phase 6 calls on_agent_ready. Polling is not expected before then, so
        # construction age must not turn this into a liveness failure.
        before_ready = await check_scheduler_liveness(
            SimpleNamespace(
                did="agent-1",
                features={"SchedulerFeature": feature},
            ),
            db,
        )
        assert before_ready["status"] == "warn"
        assert before_ready["details"]["state"] == "awaiting_telemetry"
        parameters = scheduler_status_parameters(feature)
        assert parameters["startup_grace_remaining"] == 90.0

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
        tick_started = datetime.now(timezone.utc).isoformat()
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
        assert listed.data["scheduler_status"]["state"] == "unavailable"
        assert listed.data["scheduler_status"]["status"] == "warn"
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
        escalated = feature._evaluate_safety(fresh)
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
        needs_ack = feature._evaluate_safety(legacy)
        assert needs_ack["safe"] is False
        assert "acknowledgement required" in needs_ack["reason"]
        acknowledged = await feature.acknowledge_restart_escalation(legacy.id)
        assert acknowledged.data["acknowledged"] is True
        legacy = await get_request(db, legacy.id)
        assert feature._evaluate_safety(legacy)["safe"] is True
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
