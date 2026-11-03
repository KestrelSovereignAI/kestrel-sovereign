"""A failing enforcing hook fails one tick closed; it does not pause (#3749).

``security_guard`` raised ``OperationalError: database is locked``, the hook
manager failed closed with a DENY, and the scheduler read that DENY as a
policy block: it paused ``restart_coordinator`` (``enabled=0``,
``terminal_status='blocked'``) on Emma, Claw and Meridian, three times, and
pending restart requests waited days for someone to notice.

These tests drive the production path end to end: a real ``SchedulerRunner``
over a real SQLite database, the real ``SignalDispatcher`` and cron source
registration, a real ``HooksManager``, and the real ``SecurityHook`` over a
permission store that raises the way the live one did. The tool the schedule
names is a stub; no test reaches a real restart (see ``_no_real_restart``).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sovereign import cli_lifecycle
from kestrel_sovereign.features.health import checks as health_checks
from kestrel_sovereign.features.restart_coordinator.feature import (
    RestartCoordinatorFeature,
)
from kestrel_sovereign.features.scheduler.constants import HOOK_FAILED_STATUS
from kestrel_sovereign.features.scheduler.failure_streak import (
    FAILURE_STREAK_COLUMNS,
    FailureStreak,
    hook_failure_backoff,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SchedulerRunner,
)
from kestrel_sovereign.features.scheduler.status import (
    emit_runtime_status,
    scheduler_status,
)
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionLevel
from kestrel_sovereign.hooks import HooksManager, evaluate_blocking_decision
from kestrel_sovereign.hooks.manager import FailedHookDenial
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.scheduler import build_cron_registrations
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_signals_scheduler_source import _FakeAgent
from tests.utils.scheduler_ticks import tick_and_settle

TASK = "restart-task"
TASK_NAME = "restart_coordinator"
RUNNER_LOGGER = "kestrel_sovereign.features.scheduler.runner"
LOCKED = "database is locked"


@pytest.fixture(autouse=True)
def _no_real_restart(monkeypatch):
    """Every restart/spawn seam fails the test if anything reaches it.

    Installed before any test body runs, so before any scheduler tick. The
    schedules here name ``restart_coordinator`` but resolve to a stub tool;
    this makes certain a wiring mistake cannot run a real restart instead.
    """

    def refuse(*_args, **_kwargs):
        raise AssertionError("a #3749 test reached a real restart seam")

    async def refuse_async(*_args, **_kwargs):
        refuse()

    monkeypatch.setattr(cli_lifecycle, "cmd_start", refuse)
    monkeypatch.setattr(cli_lifecycle, "cmd_terminate", refuse)
    monkeypatch.setattr(
        RestartCoordinatorFeature, "_spawn_restart_subprocess", refuse
    )
    monkeypatch.setattr(
        RestartCoordinatorFeature, "restart_coordinator", refuse_async
    )


class _StubRestartFeature:
    """Exposes a ``restart_coordinator`` tool that only counts its calls."""

    def __init__(self):
        self.enabled = True
        self.calls = 0

        async def execute(**_kwargs):
            self.calls += 1
            return {"success": True, "processed": 0}

        tool = MagicMock()
        tool.name = TASK_NAME
        tool.execute = AsyncMock(side_effect=execute)
        self._tool = tool

    def get_tools(self):
        return [self._tool]


class _PermissionStore:
    """The SecurityHook's store, answering from a script.

    An exception in the script is raised, as the live store raised
    ``OperationalError: database is locked``; a level is returned.
    """

    def __init__(self, *answers):
        self._answers = list(answers)
        self.reads = 0
        self.log_decision = AsyncMock()

    async def get_permission(self, _feature_name, _tool_name):
        answer = self._answers[min(self.reads, len(self._answers) - 1)]
        self.reads += 1
        if isinstance(answer, BaseException):
            raise answer
        return answer


class _Harness:
    def __init__(self, db, runner, feature, tool_feature, store):
        self.db = db
        self.runner = runner
        self.feature = feature
        self.tool_feature = tool_feature
        self.store = store

    async def fire(self):
        """Make the schedule due now and run the occurrence the poll claims."""

        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await self.db.execute(
            "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
            (due, TASK),
        )
        await tick_and_settle(self.runner)

    async def schedule(self):
        return await self.db.fetchone(
            "SELECT enabled, terminal_status, next_run_at FROM scheduled_tasks "
            "WHERE id = ?",
            (TASK,),
        )

    async def executions(self):
        return await self.db.fetchall(
            "SELECT status, result_text, completed_at FROM task_execution_log "
            "WHERE task_id = ? ORDER BY rowid",
            (TASK,),
        )

    async def streak(self) -> FailureStreak:
        return FailureStreak.from_row(
            await self.db.fetchone(
                f"SELECT {', '.join(FAILURE_STREAK_COLUMNS)} "
                "FROM scheduled_tasks WHERE id = ?",
                (TASK,),
            )
        )

    async def retry_delay(self) -> timedelta:
        """How far after the last execution completed the next run is due."""

        (_enabled, _terminal, next_run_at) = await self.schedule()
        completed_at = (await self.executions())[-1][2]
        return datetime.fromisoformat(next_run_at) - datetime.fromisoformat(
            completed_at
        )


@pytest_asyncio.fixture
async def harness(tmp_path):
    """Build the production path for one ``* * * * *`` restart_coordinator row.

    ``request.param``-free: each test calls ``build(*answers)`` with the
    permission store's script, and ``kind`` for a one-shot deadline.
    """

    opened = []

    async def build(*answers, kind="cron"):
        signal_backend = SQLiteBackend(str(tmp_path / f"signals-{len(opened)}.db"))
        await signal_backend.connect()
        opened.append(signal_backend)
        store = SignalLogStore(signal_backend)
        await store.initialize()
        registry = SourceRegistry()

        agent = _FakeAgent()
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )
        permissions = _PermissionStore(*answers)
        approvals = SimpleNamespace(
            request_approval=AsyncMock(return_value=(False, "no_approver"))
        )
        hooks = HooksManager()
        hooks.register(SecurityHook(permissions, approvals))
        tool_feature = _StubRestartFeature()
        agent.hooks_manager = hooks
        agent.features = {"StubRestartFeature": tool_feature}
        agent.dispatcher = dispatcher
        agent.signal_registry = registry

        scheduler_backend = SQLiteBackend(
            str(tmp_path / f"scheduler-{len(opened)}.db")
        )
        await scheduler_backend.connect()
        db = AsyncDatabase(scheduler_backend)
        opened.append(db)

        feature = SchedulerFeature(agent)
        feature._db = db
        feature._agent_id = agent.did
        agent.features["SchedulerFeature"] = feature
        for registration in build_cron_registrations(
            tool_lookup=feature._lookup_raw_tool_result,
            reason_codes_lookup=feature._declared_reason_codes,
            builtin_handlers=feature._builtin_cron_handlers(),
        ):
            registry.register(registration)

        runner = SchedulerRunner(
            db,
            agent.did,
            feature._dispatch_scheduled_task,
            owner_id="hook-failure-worker",
        )
        await runner._ensure_tables()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, schedule_kind, run_at,
                 timezone_name, misfire_policy, idempotency_key,
                 scheduler_protocol_version)
            VALUES (?, ?, ?, '* * * * *', '{}', 1, ?, ?, ?, ?, 'UTC', 'skip',
                    'hook-failure-effect', ?)
            """,
            (
                TASK, agent.did, TASK_NAME, now, now, kind,
                now if kind == "one_shot" else None,
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        return _Harness(db, runner, feature, tool_feature, permissions)

    try:
        yield build
    finally:
        for resource in reversed(opened):
            await resource.close()


def _locked() -> sqlite3.OperationalError:
    return sqlite3.OperationalError(LOCKED)


# ---------------------------------------------------------------------------
# The hook manager says when its DENY is a failure, not a decision
# ---------------------------------------------------------------------------


class _EnforcingHook(Hook):
    def __init__(self, behaviour):
        super().__init__(
            name="enforcing_guard",
            events=[HookEvent.PRE_TOOL_USE],
            priority=10,
            timeout=0.05,
        )
        self.fail_closed = True
        self._behaviour = behaviour

    async def execute(self, input):
        if self._behaviour == "raise":
            raise _locked()
        if self._behaviour == "timeout":
            await asyncio.Event().wait()
        return HookOutput.deny("Blocked by policy: x.y")


def _pre_tool_input() -> HookInput:
    return HookInput(
        session_id="scheduler",
        hook_event_name=HookEvent.PRE_TOOL_USE.value,
        tool_name="restart_coordinator",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("behaviour", ["raise", "timeout"])
async def test_a_failed_enforcing_hook_denies_and_names_itself(behaviour):
    manager = HooksManager()
    manager.register(_EnforcingHook(behaviour))

    output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, _pre_tool_input())

    # Still a fail-closed DENY to every caller that only reads the decision.
    assert isinstance(output, FailedHookDenial)
    assert output.continue_execution is False
    blocked = evaluate_blocking_decision(output)
    assert blocked is not None
    assert blocked.failed_hook == "enforcing_guard"
    assert "fail-closed policy" in blocked.reason


@pytest.mark.asyncio
async def test_a_deny_the_hook_decided_is_not_a_hook_failure():
    manager = HooksManager()
    manager.register(_EnforcingHook("deny"))

    output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, _pre_tool_input())

    assert not isinstance(output, FailedHookDenial)
    blocked = evaluate_blocking_decision(output)
    assert blocked is not None
    assert blocked.failed_hook is None


@pytest.mark.asyncio
async def test_a_failed_parallel_enforcing_hook_is_marked_too():
    manager = HooksManager()
    manager.register(_EnforcingHook("raise"))

    [output] = await manager.execute_hooks_parallel(
        HookEvent.PRE_TOOL_USE, _pre_tool_input()
    )

    assert isinstance(output, FailedHookDenial)
    assert output.failed_hook == "enforcing_guard"


# ---------------------------------------------------------------------------
# The scheduled tick fails closed once and the schedule keeps running
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hook_that_raises_once_leaves_the_row_enabled_and_the_next_tick_runs(
    harness, caplog
):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    h = await harness(_locked(), PermissionLevel.AUTO)

    await h.fire()

    # Failed closed: the tool did not run on the tick whose hook failed.
    assert h.tool_feature.calls == 0
    [(status, text, _completed)] = await h.executions()
    assert status == HOOK_FAILED_STATUS
    assert f"Enforcing hook 'security_guard' failed (raised OperationalError: {LOCKED})" in text
    assert "stays enabled" in text
    assert f"Schedule id: {TASK}" in text
    enabled, terminal_status, _next = await h.schedule()
    assert (enabled, terminal_status) == (1, None)
    assert await h.retry_delay() >= timedelta(seconds=59)
    assert (await h.streak()).count == 1
    [warning] = [
        r for r in caplog.records
        if r.name == RUNNER_LOGGER and "enforcing permission hook failed" in r.getMessage()
    ]
    assert TASK in warning.getMessage() and "stays enabled" in warning.getMessage()

    await h.fire()

    assert h.tool_feature.calls == 1
    assert [row[0] for row in await h.executions()] == [HOOK_FAILED_STATUS, "success"]
    enabled, terminal_status, _next = await h.schedule()
    assert (enabled, terminal_status) == (1, None)
    assert await h.streak() == FailureStreak()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level", [PermissionLevel.DENY, PermissionLevel.ASK], ids=["deny", "ask"],
)
async def test_a_genuine_policy_block_still_pauses_the_schedule(harness, level):
    h = await harness(level)

    await h.fire()

    assert h.tool_feature.calls == 0
    [(status, text, _completed)] = await h.executions()
    assert status == "blocked"
    assert "permission to Auto" in text
    enabled, terminal_status, _next = await h.schedule()
    assert (enabled, terminal_status) == (0, "blocked")
    # A policy block is not a failure: the streak is untouched (#3280).
    assert await h.streak() == FailureStreak()

    # Paused: a later poll does not dispatch the row again.
    await tick_and_settle(h.runner)
    assert h.store.reads == 1


@pytest.mark.asyncio
async def test_repeated_hook_failures_back_off_and_are_reported_without_pausing(
    harness, caplog
):
    caplog.set_level(logging.WARNING, logger=RUNNER_LOGGER)
    h = await harness(_locked())
    delays = []

    for _ in range(4):
        await h.fire()
        delays.append(await h.retry_delay())

    assert h.tool_feature.calls == 0
    assert [row[0] for row in await h.executions()] == [HOOK_FAILED_STATUS] * 4
    # Each consecutive failure doubles the wait before the next attempt.
    for delay, failures in zip(delays, range(1, 5)):
        expected = hook_failure_backoff(failures)
        assert expected - timedelta(seconds=1) <= delay <= expected + timedelta(seconds=1)
    enabled, terminal_status, _next = await h.schedule()
    assert (enabled, terminal_status) == (1, None)

    # Escalation at the configured threshold (3): one WARNING, the
    # schedule_list marker, and the health check.
    streak = await h.streak()
    assert streak.count == 4 and streak.flagged_at is not None
    [flagged] = [
        r for r in caplog.records
        if r.name == RUNNER_LOGGER and "failed 3 consecutive times" in r.getMessage()
    ]
    assert TASK in flagged.getMessage()
    listed = await h.feature.schedule_list()
    [task] = listed.data["tasks"]
    assert task["enabled"] is True
    assert task["failure_marker"]["consecutive_failures"] == 4
    assert LOCKED in task["failure_marker"]["last_error"]
    check = await health_checks.check_scheduled_task_failures(h.feature.agent, h.db)
    assert check["status"] == "warn"
    assert check["details"]["failing"][0]["task_id"] == TASK


@pytest.mark.asyncio
async def test_a_one_shot_deadline_whose_hook_failed_is_retried_not_consumed(harness):
    h = await harness(_locked(), PermissionLevel.AUTO, kind="one_shot")

    await h.fire()

    assert h.tool_feature.calls == 0
    enabled, terminal_status, _next = await h.schedule()
    assert (enabled, terminal_status) == (1, None)
    assert await h.retry_delay() >= timedelta(seconds=59)

    await h.fire()

    assert h.tool_feature.calls == 1
    enabled, terminal_status, _next = await h.schedule()
    assert (enabled, terminal_status) == (0, "success")


def test_the_backoff_doubles_from_its_base_up_to_its_cap():
    assert [hook_failure_backoff(n).total_seconds() for n in range(1, 8)] == [
        60, 120, 240, 480, 900, 900, 900,
    ]
    assert hook_failure_backoff(10**9) == timedelta(minutes=15)
    assert hook_failure_backoff(5, base_seconds=0) == timedelta(0)


def test_a_hook_failure_outcome_cannot_pause():
    outcome = ScheduledTaskOutcome.hook_failed(task_name="t", reason="boom.")
    assert outcome.status == HOOK_FAILED_STATUS
    assert outcome.pause_schedule is False
    assert "boom. The scheduler failed closed" in outcome.result_text
    with pytest.raises(ValueError, match="only valid for blocked"):
        ScheduledTaskOutcome(
            status=HOOK_FAILED_STATUS, result_text="x", pause_schedule=True
        )


# ---------------------------------------------------------------------------
# A schedule the scheduler paused is not reported as an operator pause
# ---------------------------------------------------------------------------


async def _status(h):
    now = datetime.now(timezone.utc).isoformat()
    await emit_runtime_status(
        h.db,
        agent_ids=(h.feature._agent_id,),
        owner_id=h.runner._owner_id,
        worker_state="running",
        last_tick_started_at=now,
        last_tick_completed_at=now,
        restart_count=0,
        consecutive_failures=0,
        last_error_type=None,
    )
    status = await scheduler_status(
        h.db, agent_id=h.feature._agent_id, poll_interval=30
    )
    [task] = (await h.feature.schedule_list()).data["tasks"]
    return status, task["disablement"]


def _assert_scheduler_paused(status, disablement):
    assert disablement["source"] == "scheduler"
    assert disablement["reason"] == "blocked"
    assert "permission" in disablement["recovery_action"]
    assert disablement["recoverable"] is True
    assert status["disabled_reasons"] == {"blocked": 1}
    assert status["system_disabled_count"] == 1
    assert status["state"] == "system_disabled_schedules"


@pytest.mark.asyncio
async def test_a_policy_pause_is_labelled_the_schedulers(harness):
    h = await harness(PermissionLevel.DENY)
    await h.fire()

    _assert_scheduler_paused(*await _status(h))


LEGACY_BLOCKED_AT = "2026-07-14T12:00:00.123456+00:00"


async def _execution(h, execution_id, status, executed_at, *, claimed=False):
    """Insert an execution as a #2457-era runner wrote it, or a claimed one.

    That runner predates claims and wrote no ``claimed_at``; every runner
    since claims an occurrence before recording it.
    """

    await h.db.execute(
        "INSERT INTO task_execution_log (id, task_id, agent_id, status, "
        "result_text, duration_ms, executed_at, claimed_at) "
        "VALUES (?, ?, ?, ?, 'text', 3, ?, ?)",
        (
            execution_id, TASK, h.feature._agent_id, status, executed_at,
            executed_at if claimed else None,
        ),
    )


async def _disable(h, *, last_run_at):
    await h.db.execute(
        "UPDATE scheduled_tasks SET enabled = 0, terminal_status = NULL, "
        "last_run_at = ? WHERE id = ?",
        (last_run_at, TASK),
    )


def _assert_operator_paused(status, disablement):
    assert disablement["source"] == "operator"
    assert disablement["reason"] == "operator_paused"
    assert status["disabled_reasons"] == {"operator_paused": 1}
    assert status["system_disabled_count"] == 0


@pytest.mark.asyncio
async def test_a_legacy_policy_pause_without_terminal_status_is_the_schedulers(
    harness,
):
    """The #2457-era pause: the row disabled, ``last_run_at`` set to the
    blocked execution's instant, and nothing else written."""

    h = await harness(PermissionLevel.AUTO)
    await _execution(h, "legacy-block", "blocked", LEGACY_BLOCKED_AT)
    await _disable(h, last_run_at=LEGACY_BLOCKED_AT)

    _assert_scheduler_paused(*await _status(h))


@pytest.mark.asyncio
async def test_an_operator_pause_after_a_later_run_stays_an_operator_pause(harness):
    """Resumed after the legacy block, ticked (a misfire skip, which leaves
    ``last_run_at`` alone), then paused by an operator."""

    h = await harness(PermissionLevel.AUTO)
    await _execution(h, "legacy-block", "blocked", LEGACY_BLOCKED_AT)
    await _execution(
        h, "later-skip", "skipped_misfire", "2026-07-15T12:00:00.000+00:00",
        claimed=True,
    )
    await _disable(h, last_run_at=LEGACY_BLOCKED_AT)

    _assert_operator_paused(*await _status(h))


@pytest.mark.asyncio
async def test_an_operator_pause_that_won_the_race_stays_an_operator_pause(harness):
    """An operator paused while the legacy blocked tick ran: that runner's
    pause matched no enabled row, so it never wrote ``last_run_at``."""

    h = await harness(PermissionLevel.AUTO)
    await _execution(h, "legacy-block", "blocked", LEGACY_BLOCKED_AT)
    await _disable(h, last_run_at="2026-07-13T12:00:00.000000+00:00")

    _assert_operator_paused(*await _status(h))


@pytest.mark.asyncio
async def test_an_operator_pause_after_a_claimed_block_stays_an_operator_pause(
    harness,
):
    """A current runner's non-pausing block (training_cycle's semantic gate)
    writes ``last_run_at`` and ``executed_at`` from the same millisecond clock.
    An operator pause after it is still the operator's."""

    h = await harness(PermissionLevel.AUTO)
    await _execution(h, "claimed-block", "blocked", LEGACY_BLOCKED_AT, claimed=True)
    await _disable(h, last_run_at=LEGACY_BLOCKED_AT)

    _assert_operator_paused(*await _status(h))


@pytest.mark.asyncio
async def test_a_plain_operator_pause_is_still_an_operator_pause(harness):
    h = await harness(PermissionLevel.AUTO)
    await h.fire()
    paused = await h.feature.schedule_pause(TASK)
    assert paused.status.value == "ok"

    status, disablement = await _status(h)

    assert disablement["source"] == "operator"
    assert status["disabled_reasons"] == {"operator_paused": 1}
