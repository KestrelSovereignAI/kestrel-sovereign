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
import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sovereign import cli_lifecycle
from kestrel_sovereign.features.health import checks as health_checks
from kestrel_sovereign.features.health.feature import HealthFeature
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
from kestrel_sovereign.features.scheduler.outcome import (
    PAUSE_CAUSE_COLUMNS,
    PauseCause,
    ScheduledTaskOutcome,
)
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SchedulerRunner,
)
from kestrel_sovereign.features.scheduler.status import (
    SUPERSESSION_INDEX,
    emit_runtime_status,
    scheduler_status,
    superseding_schedules,
)
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionLevel
from kestrel_sovereign.hooks import HooksManager, evaluate_blocking_decision
from kestrel_sovereign.hooks.manager import FailedHookDenial, deciding_hook
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.scheduler import build_cron_registrations
from kestrel_sovereign.spawn.mandate_hook import MandateRestrictionHook
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
    # A hook that failed decided nothing (#3873).
    assert blocked.deciding_hook is None


@pytest.mark.asyncio
async def test_a_deny_the_hook_decided_is_not_a_hook_failure():
    manager = HooksManager()
    manager.register(_EnforcingHook("deny"))

    output = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, _pre_tool_input())

    assert not isinstance(output, FailedHookDenial)
    blocked = evaluate_blocking_decision(output)
    assert blocked is not None
    assert blocked.failed_hook is None
    # The manager records which hook decided (#3873).
    assert blocked.deciding_hook == "enforcing_guard"
    assert deciding_hook(output) == "enforcing_guard"


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


# ---------------------------------------------------------------------------
# A scheduler pause records a structured cause (#3873)
# ---------------------------------------------------------------------------

#: The hook's own words in ``MandateRestrictionHook``'s argument denial.
MANDATE_REASON = "per this agent's spawn mandate"
SECRET_ARGUMENT = "private-target-7f3a"


async def _stored_cause(h, task_id=TASK):
    return tuple(
        await h.db.fetchone(
            f"SELECT {', '.join(PAUSE_CAUSE_COLUMNS)} FROM scheduled_tasks "
            "WHERE id = ?",
            (task_id,),
        )
    )


async def _report_worker(h):
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


async def _liveness(h):
    await _report_worker(h)
    return await health_checks.check_scheduler_liveness(h.feature.agent, h.db)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level", [PermissionLevel.DENY, PermissionLevel.ASK], ids=["deny", "ask"],
)
async def test_a_policy_pause_stores_its_structured_cause_and_liveness_names_it(
    harness, level
):
    h = await harness(level)
    await h.fire()

    # A tick has no approver, so the security hook answers ASK with a DENY.
    assert await _stored_cause(h) == ("security_guard", "deny", TASK_NAME)
    cause = {"hook": "security_guard", "decision": "deny", "tool": TASK_NAME}
    status, disablement = await _status(h)
    assert disablement["cause"] == cause
    [named] = status["system_disabled_schedules"]
    assert named["task_id"] == TASK and named["task_name"] == TASK_NAME
    assert (named["reason"], named["cause"]) == ("blocked", cause)
    assert named["terminal_at"] is not None

    check = await _liveness(h)
    assert check["status"] == "warn"
    assert check["details"]["state"] == "system_disabled_schedules"
    assert (
        f"{TASK_NAME} ({TASK[:8]}) blocked: hook security_guard decided deny "
        f"for tool {TASK_NAME}"
    ) in check["message"]

    # The cause survives the execution log row the pause was recorded in.
    await h.db.execute("DELETE FROM task_execution_log WHERE task_id = ?", (TASK,))
    assert (await _status(h))[1]["cause"] == cause

    # Resuming clears it with the pause.
    resumed = await h.feature.schedule_resume(TASK)
    assert resumed.status.value == "ok"
    assert await _stored_cause(h) == (None, None, None)


@pytest.mark.asyncio
async def test_the_hook_reason_and_arguments_are_never_stored_or_surfaced(
    harness, monkeypatch, caplog
):
    """A spawn mandate's hook names the argument value it rejected.

    Under a durable privacy mode, where no reader withholds anything, the
    pause records only which hook decided what about which tool: no column
    of the schedule row or of its ``task_execution_log`` row (the
    execution's ``result_text`` included), no scheduler status, no
    ``schedule_history`` entry, no health check, and none of HealthFeature's
    ``health_log`` history, latest result or in-memory cache holds the hook's
    words or the argument.
    """

    caplog.set_level(logging.DEBUG)
    h = await harness(PermissionLevel.AUTO)
    mandate = MandateRestrictionHook(
        (), restricted_tool_args={TASK_NAME: {"target": ["approved"]}},
    )
    # The hook's reason really does carry both, so their absence below is
    # the scheduler leaving them out.
    denial = await mandate.execute(
        HookInput(
            session_id="scheduler",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name=TASK_NAME,
            tool_input={"target": SECRET_ARGUMENT},
        )
    )
    assert MANDATE_REASON in denial.permission_reason
    assert SECRET_ARGUMENT in denial.permission_reason
    h.feature.agent.hooks_manager.register(mandate)
    await h.db.execute(
        "UPDATE scheduled_tasks SET args_json = ? WHERE id = ?",
        (json.dumps({"target": SECRET_ARGUMENT}), TASK),
    )
    await h.fire()

    assert await _stored_cause(h) == ("mandate_restriction", "deny", TASK_NAME)

    [(status, text, _completed)] = await h.executions()
    assert status == "blocked"
    assert text == (
        "blocked: the permission gate stopped this tick (hook "
        f"mandate_restriction decided deny for tool {TASK_NAME}). Set this "
        f"tool's permission to Auto, then resume the schedule. Schedule id: "
        f"{TASK}."
    )

    async def holding(table, where):
        columns = [
            row[0]
            for row in await h.db.fetchall(
                f"SELECT name FROM pragma_table_info('{table}')"
            )
        ]
        return {
            name
            for row in await h.db.fetchall(
                f"SELECT * FROM {table} WHERE {where} = ?", (TASK,)
            )
            for name, value in zip(columns, row)
            if isinstance(value, str)
            and (MANDATE_REASON in value or SECRET_ARGUMENT in value)
        }

    # Only the schedule's own arguments, which it was created with.
    assert await holding("scheduled_tasks", "id") == {"args_json"}
    assert await holding("task_execution_log", "task_id") == set()

    async def scheduler_checks_only(agent, db):
        return [await health_checks.check_scheduler_liveness(agent, db)]

    monkeypatch.setattr(health_checks, "run_standard_checks", scheduler_checks_only)
    agent = h.feature.agent
    agent.storage = SimpleNamespace(db=h.db)
    health = HealthFeature(agent)
    await health.initialize()
    try:
        await _report_worker(h)
        result = await health.run_once()
        history = (await health.health_history()).data
        latest = await health.get_latest()
        cached = health.get_status()
        status = await scheduler_status(
            h.db, agent_id=h.feature._agent_id, poll_interval=30
        )
        listed = (await h.feature.schedule_list()).data
        executions = (await h.feature.schedule_history()).data
    finally:
        await health.shutdown()

    assert result["status"] == "degraded"
    [check] = result["checks"]
    assert (
        f"blocked: hook mandate_restriction decided deny for tool {TASK_NAME}"
        in check["message"]
    )
    [(logged_id, checks_json)] = await h.db.fetchall(
        "SELECT id, checks_json FROM health_log"
    )
    assert logged_id == result["id"] == history["history"][0]["id"]
    surfaces = {
        "health_log row": checks_json,
        "health check": result,
        "health_history": history,
        "get_latest": latest,
        "cache": cached,
        "scheduler status": status,
        "schedule_list status": listed["scheduler_status"],
        "schedule_list disablement": [t["disablement"] for t in listed["tasks"]],
        "schedule_history": executions,
    }
    assert "mandate_restriction decided deny" in json.dumps(executions)
    for name, surface in surfaces.items():
        rendered = json.dumps(surface, default=str)
        assert MANDATE_REASON not in rendered, name
        assert SECRET_ARGUMENT not in rendered, name
    logged = [
        record.getMessage() for record in caplog.records
        if record.name.startswith((
            "kestrel_sovereign.features.scheduler",
            "kestrel_sovereign.features.health",
        ))
    ]
    assert any("hook 'mandate_restriction' decided deny" in m for m in logged)
    assert not any(
        MANDATE_REASON in m or SECRET_ARGUMENT in m for m in logged
    )


@pytest.mark.asyncio
async def test_a_hook_failure_records_no_cause_and_no_disablement(harness):
    """#3749's distinction stands: a hook that failed decided nothing."""

    h = await harness(_locked(), PermissionLevel.AUTO)
    await h.fire()

    enabled, terminal_status, _next = await h.schedule()
    assert (enabled, terminal_status) == (1, None)
    assert await _stored_cause(h) == (None, None, None)
    status, disablement = await _status(h)
    assert disablement["state"] == "enabled"
    assert status["system_disabled_count"] == 0
    assert status["system_disabled_schedules"] == []


@pytest.mark.asyncio
async def test_a_pause_stored_without_a_cause_says_so(harness):
    """A pause from before causes were stored is still named."""

    h = await harness(PermissionLevel.DENY)
    await h.fire()
    await h.db.execute(
        "UPDATE scheduled_tasks SET "
        + ", ".join(f"{column} = NULL" for column in PAUSE_CAUSE_COLUMNS)
        + " WHERE id = ?",
        (TASK,),
    )

    check = await _liveness(h)

    assert check["details"]["system_disabled_schedules"][0]["cause"] is None
    assert f"{TASK_NAME} ({TASK[:8]}) blocked, no cause recorded" in check["message"]


def test_a_pause_cause_holds_identifiers_only():
    with pytest.raises(ValueError, match="blocking decision"):
        PauseCause(hook="security_guard", decision="allow", tool=TASK_NAME)
    with pytest.raises(ValueError, match="identifier"):
        PauseCause(hook="Blocked: secret", decision="deny", tool=TASK_NAME)
    # A name that is not an identifier is recorded as unknown, never as text.
    assert PauseCause.of(
        hook="reason with spaces", decision="ask", tool=TASK_NAME
    ) == PauseCause(hook=None, decision="ask", tool=TASK_NAME)
    assert PauseCause.from_row(("h", "not-a-decision", "t")) is None
    assert PauseCause.from_row((None, None, None)) is None
    with pytest.raises(ValueError, match="must record its cause"):
        ScheduledTaskOutcome(
            status="blocked", result_text="x", pause_schedule=True
        )
    outcome = ScheduledTaskOutcome.blocked(
        task_name=TASK_NAME, decision="ask", hook="security_guard"
    )
    assert outcome.pause_cause == PauseCause(
        hook="security_guard", decision="ask", tool=TASK_NAME
    )


# ---------------------------------------------------------------------------
# A pause a re-registration superseded stops warning (#3873)
# ---------------------------------------------------------------------------


async def _reregister(h, *, idempotency_key):
    added = await h.feature.schedule_add(
        cron_expression="* * * * *",
        task_name=TASK_NAME,
        idempotency_key=idempotency_key,
    )
    assert added.status.value == "ok", added.error
    return added.data["task_id"]


@pytest.mark.asyncio
async def test_a_superseded_blocked_row_does_not_hold_liveness_at_warn(harness):
    """Emma's ``workflow_trigger_cron`` was paused, and its owner registered a
    replacement under the same idempotency key under a second later. The
    replacement ran; the old row held ``scheduler_liveness`` at warn on every
    tick until an operator deleted it."""

    h = await harness(PermissionLevel.DENY)
    await h.fire()
    assert (await _liveness(h))["status"] == "warn"

    replacement = await _reregister(h, idempotency_key="hook-failure-effect")

    check = await _liveness(h)
    assert check["status"] == "pass"
    details = check["details"]
    assert details["system_disabled_count"] == 0
    assert details["superseded_count"] == 1
    [superseded] = details["superseded_schedules"]
    assert (superseded["task_id"], superseded["superseded_by"]) == (TASK, replacement)
    assert superseded["cause"] == {
        "hook": "security_guard", "decision": "deny", "tool": TASK_NAME,
    }

    # Nothing rewrote the paused row; it is reported as history.
    enabled, terminal_status, _next = await h.schedule()
    assert (enabled, terminal_status) == (0, "blocked")
    assert await _stored_cause(h) == ("security_guard", "deny", TASK_NAME)
    tasks = {
        task["id"]: task for task in (await h.feature.schedule_list()).data["tasks"]
    }
    old = tasks[TASK]["disablement"]
    assert (old["state"], old["superseded_by"]) == ("superseded", replacement)
    assert replacement in old["recovery_action"]
    assert tasks[replacement]["disablement"]["state"] == "enabled"


@pytest.mark.asyncio
async def test_a_schedule_under_another_key_supersedes_nothing(harness):
    h = await harness(PermissionLevel.DENY)
    await h.fire()

    await _reregister(h, idempotency_key="some-other-effect")

    check = await _liveness(h)
    assert check["status"] == "warn"
    assert check["details"]["system_disabled_count"] == 1
    assert check["details"]["superseded_count"] == 0


@pytest.mark.asyncio
async def test_a_replacement_created_at_the_same_instant_supersedes_nothing(harness):
    """Equal creation times say nothing about which row replaced which."""

    h = await harness(PermissionLevel.DENY)
    await h.fire()
    replacement = await _reregister(h, idempotency_key="hook-failure-effect")
    await h.db.execute(
        "UPDATE scheduled_tasks SET created_at = "
        "(SELECT created_at FROM scheduled_tasks WHERE id = ?) WHERE id = ?",
        (TASK, replacement),
    )

    assert (await _liveness(h))["details"]["system_disabled_count"] == 1


@pytest.mark.asyncio
async def test_only_a_permission_pause_is_superseded(harness):
    """Any other scheduler disablement names work an operator must still
    reconcile, so a replacement does not retire it."""

    h = await harness(PermissionLevel.AUTO)
    # A cause left from an earlier pause names no other disablement.
    await h.db.execute(
        "UPDATE scheduled_tasks SET enabled = 0, "
        "terminal_status = 'execution_log_inconsistent', "
        "terminal_cause_hook = 'security_guard', "
        "terminal_cause_decision = 'deny', terminal_cause_tool = ? WHERE id = ?",
        (TASK_NAME, TASK),
    )
    await _reregister(h, idempotency_key="hook-failure-effect")

    check = await _liveness(h)
    assert check["status"] == "warn"
    assert check["details"]["disabled_reasons"] == {"execution_log_inconsistent": 1}
    assert check["details"]["superseded_count"] == 0
    [named] = check["details"]["system_disabled_schedules"]
    assert named["cause"] is None
    assert "security_guard" not in check["message"]


async def _paused_rows(h, count, *, replaced):
    """Insert ``count`` permission-paused rows, each under its own key, and
    a later replacement for the first ``replaced`` of them."""

    created = "2026-10-25T18:00:00.000000+00:00"
    later = "2026-10-25T18:00:01.000000+00:00"
    for index in range(count):
        await h.db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, enabled, created_at,
                 idempotency_key, terminal_status, scheduler_protocol_version,
                 terminal_cause_hook, terminal_cause_decision, terminal_cause_tool)
            VALUES (?, ?, ?, '* * * * *', 0, ?, ?, 'blocked', ?,
                    'security_guard', 'deny', ?)
            """,
            (
                f"paused-{index}", h.feature._agent_id, TASK_NAME, created,
                f"effect-{index}", SCHEDULER_PROTOCOL_VERSION, TASK_NAME,
            ),
        )
        if index < replaced:
            await h.db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, enabled,
                     next_run_at, created_at, idempotency_key,
                     scheduler_protocol_version)
                VALUES (?, ?, ?, '* * * * *', 1, ?, ?, ?, ?)
                """,
                (
                    f"replacement-{index}", h.feature._agent_id, TASK_NAME,
                    "2999-01-01T00:00:00+00:00", later, f"effect-{index}",
                    SCHEDULER_PROTOCOL_VERSION,
                ),
            )


def _count_supersession_queries(monkeypatch, db):
    """Record each supersession lookup: its SQL, parameters and row count."""

    seen = []
    fetchall = db.fetchall

    async def counting(sql, params=()):
        rows = await fetchall(sql, params)
        if "AS superseded_by" in sql and not sql.startswith("EXPLAIN"):
            seen.append((sql, params, len(rows)))
        return rows

    monkeypatch.setattr(db, "fetchall", counting)
    return seen


async def _assert_indexed_lookup(db, sql, params):
    """The lookup seeks each paused row by its primary key and reads the
    composite index in order: no table scan, and no sort."""

    plan = [
        str(row[-1])
        for row in await db.fetchall(f"EXPLAIN QUERY PLAN {sql}", params)
    ]
    uses_index = re.compile(rf"USING (COVERING )?INDEX {SUPERSESSION_INDEX} ")
    assert any(uses_index.search(step) for step in plan), plan
    assert not any(step.startswith("SCAN") for step in plan), plan
    assert not any("TEMP B-TREE" in step for step in plan), plan


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [1, 40])
async def test_the_supersession_lookup_is_one_indexed_query(
    harness, monkeypatch, paused
):
    """However many rows are paused: one lookup per read, through the
    composite index, and none when nothing is paused."""

    h = await harness(PermissionLevel.AUTO)
    seen = _count_supersession_queries(monkeypatch, h.db)

    # An enabled schedule, nothing paused.
    await _report_worker(h)
    await scheduler_status(h.db, agent_id=h.feature._agent_id, poll_interval=30)
    assert seen == []

    await _paused_rows(h, paused, replaced=paused // 2)
    await _report_worker(h)
    assert len(seen) == 1  # the heartbeat's inventory
    status = await scheduler_status(
        h.db, agent_id=h.feature._agent_id, poll_interval=30
    )
    assert len(seen) == 2
    assert status["superseded_count"] == paused // 2
    assert status["system_disabled_count"] == paused - paused // 2

    sql, params, returned = seen[-1]
    assert returned == len(params) == paused
    await _assert_indexed_lookup(h.db, sql, params)


async def _blocked_registrations(h, count, *, idempotency_key):
    """Insert ``count`` permission-paused registrations under one key, each
    created a millisecond after the last. Their ids run the other way, so
    only ordering by ``created_at`` finds the newest. Returns the ids,
    oldest first."""

    created = datetime(2026, 10, 25, 18, 0, 35, tzinfo=timezone.utc)
    ids = []
    for index in range(count):
        task_id = f"blocked-{count - index:04d}"
        await h.db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, enabled, created_at,
                 idempotency_key, terminal_status, scheduler_protocol_version,
                 terminal_cause_hook, terminal_cause_decision, terminal_cause_tool)
            VALUES (?, ?, ?, '* * * * *', 0, ?, ?, 'blocked', ?,
                    'security_guard', 'deny', ?)
            """,
            (
                task_id, h.feature._agent_id, TASK_NAME,
                (created + timedelta(milliseconds=index)).isoformat(
                    timespec="microseconds"
                ),
                idempotency_key, SCHEDULER_PROTOCOL_VERSION, TASK_NAME,
            ),
        )
        ids.append(task_id)
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize("registrations", [2, 200])
async def test_many_blocked_registrations_under_one_key_cost_one_row_each(
    harness, monkeypatch, registrations
):
    """A key re-registered and blocked again and again: every pause but the
    newest is superseded by the newest, and the lookup returns one row per
    paused row. Joining each pause to every later row under its key returned
    N(N-1)/2 rows, 19,900 for 200, on every heartbeat."""

    h = await harness(PermissionLevel.AUTO)
    ids = await _blocked_registrations(
        h, registrations, idempotency_key="shared-effect"
    )
    newest = ids[-1]
    seen = _count_supersession_queries(monkeypatch, h.db)

    assert await superseding_schedules(h.db, ids) == {
        task_id: newest for task_id in ids[:-1]
    }
    [(sql, params, returned)] = seen
    assert returned == len(params) == registrations
    await _assert_indexed_lookup(h.db, sql, params)

    await _report_worker(h)
    status = await scheduler_status(
        h.db, agent_id=h.feature._agent_id, poll_interval=30
    )
    assert [returned for _sql, _params, returned in seen[1:]] == [
        registrations, registrations,
    ]  # the heartbeat's inventory, then the read's
    # The newest registration was blocked too, and still holds the warn.
    assert status["system_disabled_count"] == 1
    [named] = status["system_disabled_schedules"]
    assert named["task_id"] == newest
    assert status["superseded_count"] == registrations - 1
    assert {
        superseded["superseded_by"]
        for superseded in status["superseded_schedules"]
    } == {newest}


@pytest.mark.asyncio
async def test_superseding_schedules_names_the_newest_replacement(harness):
    h = await harness(PermissionLevel.AUTO)
    await _paused_rows(h, 1, replaced=1)
    await h.db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, enabled, next_run_at,
             created_at, idempotency_key, scheduler_protocol_version)
        VALUES ('replacement-newest', ?, ?, '* * * * *', 1, ?, ?, 'effect-0', ?)
        """,
        (
            h.feature._agent_id, TASK_NAME, "2999-01-01T00:00:00+00:00",
            "2026-10-25T18:00:02.000000+00:00", SCHEDULER_PROTOCOL_VERSION,
        ),
    )

    assert await superseding_schedules(h.db, ["paused-0"]) == {
        "paused-0": "replacement-newest"
    }
    assert await superseding_schedules(h.db, []) == {}

    # Two replacements created at the same instant are told apart by id.
    await h.db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, enabled, next_run_at,
             created_at, idempotency_key, scheduler_protocol_version)
        VALUES ('replacement-newest-twin', ?, ?, '* * * * *', 1, ?, ?,
                'effect-0', ?)
        """,
        (
            h.feature._agent_id, TASK_NAME, "2999-01-01T00:00:00+00:00",
            "2026-10-25T18:00:02.000000+00:00", SCHEDULER_PROTOCOL_VERSION,
        ),
    )
    assert await superseding_schedules(h.db, ["paused-0"]) == {
        "paused-0": "replacement-newest-twin"
    }
