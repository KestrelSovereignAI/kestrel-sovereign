"""Repeated failure of one scheduled task is visible (#3280).

Emma's ``signal_dispatch`` failed 23 days running and the only record was in
``task_execution_log``. These tests drive a real ``SchedulerRunner`` over a
real SQLite database, so every streak value asserted here was written by the
same finalize transaction production uses, and read back by the two readers a
person or the agent actually looks at: ``schedule_list`` and the health check.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI

from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign import server
from kestrel_sovereign.features.health import checks as health_checks
from kestrel_sovereign.features.scheduler.constants import (
    MISSED_COGNITION_STATUS,
)
from kestrel_sovereign.features.scheduler.failure_streak import (
    DEFAULT_FAILURE_ALERT_THRESHOLD,
    FAILURE_STREAK_COLUMNS,
    FAILURE_SUMMARY_MAX_CHARS,
    FailureStreak,
    next_failure_streak,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.outcome import (
    REDACTED_RESULT_TEXT,
    ScheduledTaskOutcome,
)
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SchedulerRunner,
)
from kestrel_sovereign.privacy import PrivacyConfig
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.utils.scheduler_ticks import tick_and_settle

AGENT = "did:test:failure-streak"
TASK = "streak-task"
TASK_NAME = "signal_dispatch"
RUNNER_LOGGER = "kestrel_sovereign.features.scheduler.runner"


class _Script:
    """An executor that plays one scripted outcome per fire."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.fired = 0

    async def __call__(self, _task_name, _args):
        outcome = self._outcomes[self.fired]
        self.fired += 1
        if outcome == "fail":
            raise RuntimeError(f"dispatch rejected #{self.fired}")
        if outcome == "refuse":
            # A returned refusal, the shape signal_dispatch produced for Emma.
            return ToolResult.failed(f"workflow definition not found #{self.fired}")
        if outcome == "ok":
            return "done"
        return outcome


@pytest_asyncio.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "failure-streak.db"))
    await backend.connect()
    database = AsyncDatabase(backend)
    try:
        yield database
    finally:
        await database.close()


async def _runner(db, executor, *, threshold=DEFAULT_FAILURE_ALERT_THRESHOLD):
    runner = SchedulerRunner(
        db,
        AGENT,
        executor,
        owner_id="failure-streak-worker",
        failure_alert_threshold=threshold,
    )
    await runner._ensure_tables()
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at, schedule_kind, timezone_name,
             misfire_policy, idempotency_key, scheduler_protocol_version)
        VALUES (?, ?, ?, '5 8 * * *', '{}', 1, ?, ?, 'cron', 'UTC', 'skip',
                'streak-effect', ?)
        """,
        (TASK, AGENT, TASK_NAME, now, now, SCHEDULER_PROTOCOL_VERSION),
    )
    return runner


async def _fire(runner, db, *, late_seconds=2):
    """Make the schedule due and run the occurrence the next poll claims."""

    due = datetime.now(timezone.utc) - timedelta(seconds=late_seconds)
    await db.execute(
        "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
        (due.isoformat(), TASK),
    )
    await tick_and_settle(runner)


async def _fire_times(runner, db, times):
    for _ in range(times):
        await _fire(runner, db)


async def _streak(db) -> FailureStreak:
    return FailureStreak.from_row(
        await db.fetchone(
            f"SELECT {', '.join(FAILURE_STREAK_COLUMNS)} "
            "FROM scheduled_tasks WHERE id = ?",
            (TASK,),
        )
    )


async def _executions(db):
    """Every finalized execution of the task, oldest first."""

    return await db.fetchall(
        """
        SELECT id, status, claimed_at, completed_at FROM task_execution_log
        WHERE task_id = ? ORDER BY rowid
        """,
        (TASK,),
    )


def _at(value) -> datetime:
    return datetime.fromisoformat(str(value))


def _streak_warnings(caplog):
    return [
        record for record in caplog.records
        if record.name == RUNNER_LOGGER
        and record.levelno == logging.WARNING
        and "consecutive" in record.getMessage()
    ]


def _feature(db, **agent_attrs) -> SchedulerFeature:
    agent = SimpleNamespace(did=AGENT, **agent_attrs)
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = AGENT
    agent.features = {"SchedulerFeature": feature}
    return feature


def _listed(result) -> dict:
    [task] = [task for task in result.data["tasks"] if task["id"] == TASK]
    return task


# ---------------------------------------------------------------------------
# The runner records the streak and the marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failures_below_the_threshold_record_no_marker(db, caplog):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    runner = await _runner(db, _Script("fail", "fail"))

    await _fire_times(runner, db, 2)

    streak = await _streak(db)
    assert streak.count == 2
    assert streak.flagged_at is None
    assert _streak_warnings(caplog) == []
    feature = _feature(db)
    listed = _listed(await feature.schedule_list())
    assert listed["consecutive_failures"] == 2
    assert listed["failure_marker"] is None
    check = await health_checks.check_scheduled_task_failures(feature.agent, db)
    assert check["status"] == "pass"
    assert check["details"]["failing"] == []


@pytest.mark.asyncio
async def test_reaching_the_threshold_records_the_marker_and_warns_once(
    db, caplog
):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    runner = await _runner(db, _Script("fail", "refuse", "fail"))

    await _fire_times(runner, db, 3)

    executions = await _executions(db)
    assert [row[1] for row in executions] == ["failed"] * 3
    streak = await _streak(db)
    assert streak.count == 3
    assert streak.last_execution_id == executions[-1][0]
    # Each failure time is read from the database clock inside the finalize
    # that recorded it: after that execution was claimed, before it completed.
    first, last = executions[0], executions[-1]
    assert _at(first[2]) <= _at(streak.first_at) <= _at(first[3])
    assert _at(last[2]) <= _at(streak.last_at) <= _at(last[3])
    assert streak.flagged_at == streak.last_at
    [warning] = _streak_warnings(caplog)
    message = warning.getMessage()
    assert TASK in message and TASK_NAME in message
    assert "failed 3 consecutive times" in message
    assert executions[-1][0] in message


@pytest.mark.asyncio
async def test_further_failures_count_up_without_another_warning(db, caplog):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    runner = await _runner(db, _Script("fail", "fail", "fail", "fail", "fail"))

    await _fire_times(runner, db, 3)
    flagged = await _streak(db)
    await _fire_times(runner, db, 2)

    streak = await _streak(db)
    executions = await _executions(db)
    assert streak.count == 5
    assert streak.first_at == flagged.first_at
    assert streak.flagged_at == flagged.flagged_at
    assert _at(executions[-1][2]) <= _at(streak.last_at) <= _at(executions[-1][3])
    assert streak.last_execution_id == executions[-1][0]
    assert len(_streak_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_one_success_clears_the_marker_and_logs_recovery(db, caplog):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    runner = await _runner(db, _Script("fail", "fail", "fail", "fail", "ok"))

    await _fire_times(runner, db, 4)
    assert (await _streak(db)).flagged_at is not None
    await _fire(runner, db)

    assert await _streak(db) == FailureStreak()
    raised, recovered = _streak_warnings(caplog)
    assert "failed 3 consecutive times" in raised.getMessage()
    assert "succeeded after 4 consecutive failures" in recovered.getMessage()
    assert "marker is cleared" in recovered.getMessage()
    listed = _listed(await _feature(db).schedule_list())
    assert listed["consecutive_failures"] == 0
    assert listed["failure_marker"] is None


@pytest.mark.asyncio
async def test_failures_interleaved_with_successes_never_reach_the_threshold(
    db, caplog
):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    script = ("fail", "fail", "ok", "fail", "fail", "ok", "fail", "fail")
    runner = await _runner(db, _Script(*script))

    for _ in script:
        await _fire(runner, db)
        assert (await _streak(db)).flagged_at is None

    assert (await _streak(db)).count == 2
    assert _streak_warnings(caplog) == []


@pytest.mark.asyncio
async def test_the_threshold_is_the_runners_configured_value(db, caplog):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    runner = await _runner(db, _Script("fail", "fail"), threshold=2)

    await _fire(runner, db)
    assert (await _streak(db)).flagged_at is None
    await _fire(runner, db)

    assert (await _streak(db)).flagged_at is not None
    assert len(_streak_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# Runs that did not happen, or did not fail, leave the streak alone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_paused_schedule_is_not_flagged_for_runs_that_never_happened(
    db, caplog
):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    script = _Script("fail", "fail", "fail")
    runner = await _runner(db, script)
    await _fire_times(runner, db, 2)

    await db.execute(
        "UPDATE scheduled_tasks SET enabled = 0, next_run_at = ? WHERE id = ?",
        ((datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(), TASK),
    )
    for _ in range(4):
        await tick_and_settle(runner)

    assert script.fired == 2
    assert (await _streak(db)).count == 2
    assert (await _streak(db)).flagged_at is None
    assert _streak_warnings(caplog) == []

    # Pausing neither extends nor ends the streak: the next run that fails
    # is the third consecutive failure.
    await db.execute("UPDATE scheduled_tasks SET enabled = 1 WHERE id = ?", (TASK,))
    await _fire(runner, db)
    assert (await _streak(db)).count == 3
    assert (await _streak(db)).flagged_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        ScheduledTaskOutcome(
            status=MISSED_COGNITION_STATUS, result_text="missed: dropped"
        ),
        ScheduledTaskOutcome(status="blocked", result_text="blocked: gate"),
    ],
    ids=["missed", "blocked"],
)
async def test_outcomes_that_are_neither_failure_nor_success_leave_the_streak(
    db, outcome
):
    runner = await _runner(db, _Script("fail", "fail", outcome, "fail"))

    await _fire_times(runner, db, 2)
    before = await _streak(db)
    await _fire(runner, db)

    assert [row[1] for row in await _executions(db)][-1] == outcome.status
    assert await _streak(db) == before
    await _fire(runner, db)
    assert (await _streak(db)).count == 3
    assert (await _streak(db)).flagged_at is not None


@pytest.mark.asyncio
async def test_a_misfire_skip_is_not_a_failure(db):
    script = _Script("fail", "fail", "fail")
    runner = await _runner(db, script)
    await _fire_times(runner, db, 2)
    before = await _streak(db)

    await _fire(runner, db, late_seconds=2 * 3600)

    assert script.fired == 2
    assert [row[1] for row in await _executions(db)][-1] == "skipped_misfire"
    assert await _streak(db) == before


# ---------------------------------------------------------------------------
# Where a reader looks: schedule_list and the health result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_list_reports_the_marker_for_that_task(db):
    runner = await _runner(db, _Script("fail", "fail", "refuse"))
    await _fire_times(runner, db, 3)
    streak = await _streak(db)

    result = await _feature(db).schedule_list()

    listed = _listed(result)
    assert listed["consecutive_failures"] == 3
    assert listed["failure_marker"] == {
        "consecutive_failures": 3,
        "first_failure_at": streak.first_at,
        "last_failure_at": streak.last_at,
        "last_error": "workflow definition not found #3",
        "flagged_at": streak.flagged_at,
    }
    assert result.data["failing_count"] == 1
    assert "1 failing repeatedly" in result.confirmation


@pytest.mark.asyncio
async def test_health_flags_the_failing_task(db):
    runner = await _runner(db, _Script("fail", "fail", "fail"))
    await _fire_times(runner, db, 3)
    feature = _feature(db)

    check = await health_checks.check_scheduled_task_failures(feature.agent, db)

    assert check["name"] == "scheduled_task_failures"
    assert check["status"] == "warn"
    [failing] = check["details"]["failing"]
    assert failing["task_id"] == TASK
    assert failing["task_name"] == TASK_NAME
    assert failing["enabled"] is True
    assert failing["consecutive_failures"] == 3
    assert failing["last_error"] == "RuntimeError: dispatch rejected #3"
    assert TASK_NAME in check["message"]
    assert "3 consecutive failures" in check["message"]
    assert health_checks.derive_overall_status([check]) == "degraded"


@pytest.mark.asyncio
async def test_health_passes_once_the_task_recovers(db):
    runner = await _runner(db, _Script("fail", "fail", "fail", "ok"))
    await _fire_times(runner, db, 4)
    feature = _feature(db)

    check = await health_checks.check_scheduled_task_failures(feature.agent, db)

    assert check["status"] == "pass"
    assert check["details"]["failing"] == []


@pytest.mark.asyncio
async def test_the_standard_health_checks_include_the_failing_task(db):
    runner = await _runner(db, _Script("fail", "fail", "fail"))
    await _fire_times(runner, db, 3)
    feature = _feature(
        db,
        bootstrap_service=None,
        context_manager=None,
        dispatcher=None,
        llm_service=None,
        memory_retriever=None,
        memory_consolidator=None,
        storage=None,
    )

    checks = await health_checks.run_standard_checks(feature.agent, db)

    [check] = [c for c in checks if c["name"] == "scheduled_task_failures"]
    assert check["status"] == "warn"
    assert check["details"]["failing"][0]["task_id"] == TASK


@pytest.mark.asyncio
async def test_health_passes_without_a_scheduler_feature(db):
    agent = SimpleNamespace(did=AGENT, features={})

    check = await health_checks.check_scheduled_task_failures(agent, db)

    assert check["status"] == "pass"
    assert check["details"]["state"] == "not_configured"


@pytest.mark.asyncio
async def test_health_fails_when_failures_cannot_be_read(db):
    feature = _feature(db)

    no_storage = await health_checks.check_scheduled_task_failures(
        feature.agent, None
    )
    no_identity = await health_checks.check_scheduled_task_failures(
        SimpleNamespace(did="", features=feature.agent.features), db
    )
    broken = SimpleNamespace(
        fetchall=AsyncMock(side_effect=RuntimeError("no such column"))
    )
    unreadable = await health_checks.check_scheduled_task_failures(
        feature.agent, broken
    )

    assert (no_storage["status"], no_storage["details"]["state"]) == (
        "fail", "storage_unavailable"
    )
    assert (no_identity["status"], no_identity["details"]["state"]) == (
        "fail", "identity_unavailable"
    )
    assert (unreadable["status"], unreadable["details"]["state"]) == (
        "fail", "inspection_failed"
    )
    assert "no such column" in unreadable["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["none", "temp", "deidentified"])
async def test_a_volatile_privacy_mode_withholds_the_last_error(db, storage):
    runner = await _runner(db, _Script("fail", "fail", "fail"))
    await _fire_times(runner, db, 3)
    feature = _feature(db, privacy_config=PrivacyConfig(storage=storage))

    listed = _listed(await feature.schedule_list())
    check = await health_checks.check_scheduled_task_failures(feature.agent, db)

    assert listed["failure_marker"]["last_error"] == REDACTED_RESULT_TEXT
    assert listed["failure_marker"]["consecutive_failures"] == 3
    assert check["details"]["failing"][0]["last_error"] == REDACTED_RESULT_TEXT


@pytest.mark.asyncio
async def test_the_last_error_is_a_bounded_summary(db):
    long_error = "x" * (FAILURE_SUMMARY_MAX_CHARS * 3)
    runner = await _runner(db, _Script("fail", "fail", "fail"))
    await _fire_times(runner, db, 3)
    # The marker names the latest failure, so make that the long one.
    await db.execute(
        """
        UPDATE task_execution_log SET result_text = ?
        WHERE id = (SELECT failure_streak_last_execution_id
                    FROM scheduled_tasks WHERE id = ?)
        """,
        (long_error, TASK),
    )

    marker = _listed(await _feature(db).schedule_list())["failure_marker"]

    assert len(marker["last_error"]) == FAILURE_SUMMARY_MAX_CHARS
    assert marker["last_error"].endswith("...")


# ---------------------------------------------------------------------------
# The transition rule and its configuration
# ---------------------------------------------------------------------------


def test_only_failure_and_success_move_a_streak():
    prior = FailureStreak(
        count=2, first_at="t1", last_at="t2", last_execution_id="e2"
    )
    for status in ("skipped_misfire", MISSED_COGNITION_STATUS, "blocked", "cancelled"):
        transition = next_failure_streak(
            prior, status=status, at="t3", execution_id="e3", threshold=3
        )
        assert transition.streak == prior
        assert not transition.flagged and not transition.recovered

    flagged = next_failure_streak(
        prior, status="failed", at="t3", execution_id="e3", threshold=3
    )
    assert flagged.streak == FailureStreak(
        count=3, first_at="t1", last_at="t3", last_execution_id="e3",
        flagged_at="t3",
    )
    assert flagged.flagged and not flagged.recovered

    recovered = next_failure_streak(
        flagged.streak, status="success", at="t4", execution_id="e4", threshold=3
    )
    assert recovered.streak == FailureStreak()
    assert recovered.recovered and not recovered.flagged


@pytest.mark.parametrize(
    ("configured", "expected"),
    [({}, DEFAULT_FAILURE_ALERT_THRESHOLD),
     ({"failure_alert_threshold": 5}, 5),
     ({"failure_alert_threshold": 0}, 1),
     ({"failure_alert_threshold": "three"}, DEFAULT_FAILURE_ALERT_THRESHOLD)],
)
def test_threshold_is_read_from_scheduler_config(monkeypatch, configured, expected):
    monkeypatch.setattr(
        "kestrel_sovereign.config.load_section",
        lambda section: configured if section == "scheduler" else {},
    )

    assert SchedulerFeature._load_failure_alert_threshold() == expected


@pytest.mark.asyncio
async def test_the_agent_runner_uses_the_configured_threshold(monkeypatch):
    from tests.unit.test_scheduler_feature import _make_mock_agent

    monkeypatch.setattr(
        SchedulerFeature, "_load_failure_alert_threshold", staticmethod(lambda: 6)
    )
    feature = SchedulerFeature(_make_mock_agent())
    monkeypatch.setattr(SchedulerRunner, "start", AsyncMock())

    await feature.initialize()

    assert feature._runner._failure_alert_threshold == 6


@pytest.mark.asyncio
async def test_the_host_runner_uses_the_configured_threshold(monkeypatch):
    runner_kwargs = []

    class _Storage:
        def __init__(self, **_kwargs) -> None:
            self.db = object()

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class _Runner:
        def __init__(self, *_args, **kwargs) -> None:
            runner_kwargs.append(kwargs)

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(
            return_value={"did:pkh:warm": ("Warm", object())}
        ),
        cold_scheduler_identity_failures=[],
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _Storage
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner.SchedulerRunner", _Runner
    )
    monkeypatch.setattr(
        SchedulerFeature, "_load_failure_alert_threshold", staticmethod(lambda: 7)
    )

    try:
        await server._start_host_scheduler(app, manager, object())
    finally:
        await server._shutdown_host_scheduler(app)

    assert runner_kwargs[0]["failure_alert_threshold"] == 7
