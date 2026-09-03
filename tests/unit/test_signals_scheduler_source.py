"""Phase 4 of #889: scheduler source registrations + executor end-to-end
through the real SignalDispatcher.

The legacy scheduler's tool-search behavior is covered by
`test_scheduler_feature.py::TestTaskExecutor` (now hitting the renamed
`_lookup_and_run_tool`). These tests verify the new layer: each cron
task is registered with the right mode, the dispatch path translates
SignalResult back into the runner's expected shape, and the dispatcher
is what actually invokes the registered handlers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kestrel_sdk.signals import (
    ResourceLock,
    Signal,
    SignalMode,
    Status,
    Trust,
    Visibility,
)
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.agent.sleep import SleepMixin
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.scheduler import (
    CRON_TASKS,
    build_cron_registrations,
    cron_source_name,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeAgent(SleepMixin):
    did = "did:test:scheduler-phase4"

    def __init__(self):
        self.background_tasks = []
        self.sleep_hooks = []

    async def process_input(self, prompt):  # unused for scheduler tests
        return ""

    def _track_background_task(self, coro, *, name):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def dispatcher_components(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "scheduler_e2e.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    locks = OrderedLockManager()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=locks, store=store,
    )
    yield (agent, registry, dispatcher, backend)
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


# ---------------------------------------------------------------------------
# Per-task classification
# ---------------------------------------------------------------------------


def test_all_cron_tasks_are_classified():
    """Canonical cron-capable task set, including user-scheduled tools.

    If a new one is added or one removed, this test fails loudly so
    the classification table stays in sync with the scheduler's supported
    dispatch surface. Not every source here is auto-seeded.
    """
    names = [t[0] for t in CRON_TASKS]
    assert sorted(names) == sorted([
        "backup_snapshot",
        "bootstrap_timeout_check",  # #378 — bootstrap watchdog
        "morning_signal",
        "signal_dispatch",  # user-schedulable; not a core auto-seed
        "trash_retention",
        "training_cycle",
        "reflect",
        "memory_consolidate",
        "sleep",  # #1674 P3 — nightly memory-maintenance cycle
        "wait_reconcile",  # #1860 Wave 2 — generic wait→signal reconciler
        "restart_coordinator",  # #1512
        "github_pr_watch",  # #1618
        "ecosystem_discovery_watch",  # #2281
        "self_followup",  # #3101 — agent-authored one-shot follow-up turn
    ])


def test_action_vs_artifact_split_matches_design():
    """ACTION = side effect, no LLM follow-up. ARTIFACT = produces a
    result via feature workflow that may use an LLM internally but
    doesn't enter conversation history. COGNITION = a real turn that
    enters conversation history."""
    by_mode = {
        SignalMode.ACTION: set(),
        SignalMode.ARTIFACT: set(),
        SignalMode.COGNITION: set(),
    }
    for name, mode, _ in CRON_TASKS:
        by_mode[mode].add(name)
    assert by_mode[SignalMode.ACTION] == {
        "backup_snapshot",
        "bootstrap_timeout_check",  # #378 — bootstrap watchdog
        "signal_dispatch",  # provider-neutral user-scheduled dispatch
        "trash_retention",
        "training_cycle",
        "sleep",  # #1674 P3 — built-in handler (_handle_sleep); ACTION so the
                  # handler is wired (builtin_handlers are ACTION-only)
        "wait_reconcile",  # #1860 Wave 2 — polls waitables, emits signals, no LLM
        "restart_coordinator",  # #1512 — scans, spawns subprocess
        "github_pr_watch",  # #1618 — polls a PR, emits signal on change
        "ecosystem_discovery_watch",  # #2281 — polls discovery, emits signal on findings
    }
    assert by_mode[SignalMode.ARTIFACT] == {
        "morning_signal",
        "reflect",
        "memory_consolidate",
    }
    # #3101 — the only cron task that is a genuine agent turn. It carries an
    # intention the agent formed in an earlier turn across the turn boundary,
    # so it must enter conversation history rather than produce an artifact.
    assert by_mode[SignalMode.COGNITION] == {"self_followup"}


def test_no_cron_source_declares_conversation():
    """CONVERSATION is owned solely by the turn lifecycle (Phase 2,
    #891). Cron sources must never declare it — registry would reject
    at registration time anyway, but the classification table is the
    source of truth and worth checking explicitly."""
    for _, _, resources in CRON_TASKS:
        assert ResourceLock.CONVERSATION not in resources


def test_state_mutating_tasks_declare_memory():
    """Sources that touch shared storage declare MEMORY so they
    serialize against each other (and against any future feature that
    declares MEMORY too). v1 uses a coarse `MEMORY` lock; finer
    granularity is a follow-up.

    `reflect` is in this set despite not being in CRON_TASKS as ACTION
    — ReflectionFeature.reflect() persists each session via
    _persist_reflection() (writes session + insights rows), so it
    shares storage state with memory_consolidate. Caught in #904 review
    P2; the prior empty-resources classification was wrong.
    """
    by_name = {name: resources for name, _, resources in CRON_TASKS}
    assert ResourceLock.MEMORY in by_name["trash_retention"]
    assert ResourceLock.MEMORY in by_name["training_cycle"]
    assert ResourceLock.MEMORY in by_name["memory_consolidate"]
    assert ResourceLock.MEMORY in by_name["reflect"]
    assert ResourceLock.MEMORY in by_name["sleep"]  # #1674 P3
    assert ResourceLock.MEMORY in by_name["bootstrap_timeout_check"]  # #378


# ---------------------------------------------------------------------------
# Registration shape
# ---------------------------------------------------------------------------


def test_build_cron_registrations_match_cron_tasks_table():
    async def _lookup(name, args):
        return None

    regs = build_cron_registrations(tool_lookup=_lookup)
    assert len(regs) == len(CRON_TASKS)
    names = [r.name for r in regs]
    assert all(n.startswith("cron.") for n in names)
    assert names == [cron_source_name(t[0]) for t in CRON_TASKS]


def test_action_registrations_have_handler_artifact_have_artifact_handler():
    async def _lookup(name, args):
        return f"lookup:{name}"

    regs = build_cron_registrations(tool_lookup=_lookup)
    for reg in regs:
        if reg.default_mode == SignalMode.ACTION:
            assert reg.handler is not None, f"{reg.name} ACTION needs handler"
            assert reg.artifact_handler is None
        elif reg.default_mode == SignalMode.COGNITION:
            # A COGNITION cron task has no handler of either kind: the
            # dispatcher renders the prompt template and runs a turn. A
            # registration with neither a handler nor a template would accept
            # a dispatch and produce nothing, which is the silent no-op #3101
            # exists to prevent.
            assert reg.handler is None, f"{reg.name} COGNITION takes no handler"
            assert reg.artifact_handler is None
            assert reg.prompt_template is not None, (
                f"{reg.name} COGNITION needs a prompt_template to reach a turn"
            )
            assert reg.prompt_template.exists(), (
                f"{reg.name} prompt_template {reg.prompt_template} is missing"
            )
        else:
            assert reg.artifact_handler is not None, (
                f"{reg.name} ARTIFACT needs artifact_handler"
            )
            assert reg.handler is None


def test_builtin_handlers_override_tool_lookup():
    """SchedulerFeature passes builtin_handlers for tasks that don't go
    through tool lookup (backup_snapshot calls sync.force_snapshot
    directly). The factory must wire those instead of a lookup-based
    handler."""
    captured = []

    async def lookup(name, args):
        raise AssertionError("lookup should not be called for builtins")

    async def fake_backup(args):
        captured.append(("backup", args))
        return "backup-ok"

    regs = build_cron_registrations(
        tool_lookup=lookup,
        builtin_handlers={"backup_snapshot": fake_backup},
    )
    backup_reg = next(r for r in regs if r.name == "cron.backup_snapshot")

    async def _run():
        return await backup_reg.handler({"foo": "bar"})

    result = asyncio.run(_run())
    assert result == "backup-ok"
    assert captured == [("backup", {"foo": "bar"})]


# ---------------------------------------------------------------------------
# End-to-end through the real dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_scheduled_signal_dispatch_uses_cron_action_source(
    dispatcher_components,
):
    """A custom dispatch schedule retains cron signal logging/dispatch.

    Core no longer auto-seeds this task, but a user-created row still emits
    ``cron.signal_dispatch`` and reaches the provider-neutral feature tool via
    the ordinary ACTION handler.
    """
    agent, registry, dispatcher, backend = dispatcher_components
    captured = []

    async def fake_lookup(name, args):
        captured.append((name, args))
        return f"ran:{name}"

    for reg in build_cron_registrations(tool_lookup=fake_lookup):
        registry.register(reg)

    signal = Signal(
        source=cron_source_name("signal_dispatch"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={"mode": "execute"},
        target_agent=agent.did,
    )
    result = await dispatcher.dispatch_signal(signal)
    assert result.status == Status.OK
    assert result.action_result == "ran:signal_dispatch"
    assert captured == [("signal_dispatch", {"mode": "execute"})]

    pending = [task for task in agent.background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending)
    rows = await backend.fetch_all(
        "SELECT source, status FROM signal_log WHERE source=?",
        (cron_source_name("signal_dispatch"),),
    )
    assert rows == [("cron.signal_dispatch", Status.OK.value)]


@pytest.mark.asyncio
async def test_artifact_task_dispatches_through_artifact_handler(
    dispatcher_components,
):
    """ARTIFACT cron source → dispatcher invokes artifact_handler with
    the Signal envelope → result.artifact carries the workflow output."""
    agent, registry, dispatcher, _ = dispatcher_components

    async def fake_lookup(name, args):
        return f"briefing:{name}"

    for reg in build_cron_registrations(tool_lookup=fake_lookup):
        registry.register(reg)

    signal = Signal(
        source=cron_source_name("morning_signal"),
        kind="run",
        mode=SignalMode.ARTIFACT,
        payload={},
        target_agent=agent.did,
    )
    result = await dispatcher.dispatch_signal(signal)
    assert result.status == Status.OK
    assert result.artifact == "briefing:morning_signal"
    assert result.action_result is None


@pytest.mark.asyncio
async def test_json_shaped_string_artifact_is_not_a_scheduler_envelope(
    dispatcher_components,
):
    """Only documented built-in JSON results carry scheduler semantics."""
    agent, registry, dispatcher, _ = dispatcher_components
    artifact = json.dumps({"error": "quoted error text from a report"})

    async def fake_lookup(name, args):
        return artifact

    for reg in build_cron_registrations(tool_lookup=fake_lookup):
        registry.register(reg)

    signal = Signal(
        source=cron_source_name("morning_signal"),
        kind="run",
        mode=SignalMode.ARTIFACT,
        payload={},
        target_agent=agent.did,
    )
    result = await dispatcher.dispatch_signal(signal)

    assert result.status == Status.OK
    assert result.artifact == artifact


@pytest.mark.asyncio
async def test_handler_exception_becomes_failed_status(
    dispatcher_components, caplog,
):
    """Raised tool details stay in trusted logs, not durable audit errors."""
    agent, registry, dispatcher, _ = dispatcher_components

    async def lookup_raises(name, args):
        raise RuntimeError("tool blew up")

    for reg in build_cron_registrations(tool_lookup=lookup_raises):
        registry.register(reg)

    signal = Signal(
        source=cron_source_name("training_cycle"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={"iterations": 3},
        target_agent=agent.did,
    )
    with caplog.at_level(logging.ERROR):
        result = await dispatcher.dispatch_signal(signal)
    assert result.status == Status.FAILED
    assert "scheduled tool training_cycle raised" in (result.error or "")
    assert "tool blew up" not in (result.error or "")
    trusted_record = next(
        record
        for record in caplog.records
        if record.message == "Scheduled tool training_cycle raised"
    )
    assert trusted_record.exc_info is not None
    assert "tool blew up" in str(trusted_record.exc_info[1])


@pytest.mark.asyncio
async def test_failed_tool_result_becomes_failed_status(
    dispatcher_components, caplog,
):
    """The production lookup rejects failure before its JSON boundary."""
    agent, registry, dispatcher, backend = dispatcher_components
    private_marker = "private-consolidation-detail-2907"
    failed_tool = MagicMock()
    failed_tool.name = "memory_consolidate"
    failed_tool.execute = AsyncMock(
        return_value=ToolResult.failed(private_marker).to_dict()
    )
    memory_feature = MagicMock()
    memory_feature.enabled = True
    memory_feature.get_tools.return_value = [failed_tool]
    agent.features = {"MemoryFeature": memory_feature}
    scheduler_feature = SchedulerFeature(agent)

    for reg in build_cron_registrations(
        tool_lookup=scheduler_feature._lookup_raw_tool_result
    ):
        registry.register(reg)

    signal = Signal(
        source=cron_source_name("memory_consolidate"),
        kind="run",
        mode=SignalMode.ARTIFACT,
        payload={},
        target_agent=agent.did,
    )
    with caplog.at_level(
        logging.ERROR,
        logger="kestrel_sovereign.signals.sources.scheduler",
    ):
        result = await dispatcher.dispatch_signal(signal)

    assert result.status == Status.FAILED
    assert "scheduled tool memory_consolidate failed" in (result.error or "")
    assert private_marker not in (result.error or "")
    row = await backend.fetch_one(
        "SELECT error FROM signal_log WHERE id = ?",
        (signal.id,),
    )
    assert row is not None
    assert private_marker not in (row[0] or "")
    assert any(
        private_marker in record.getMessage()
        and "returned a failed result" in record.getMessage()
        for record in caplog.records
        if record.name == "kestrel_sovereign.signals.sources.scheduler"
    )


@pytest.mark.asyncio
async def test_permission_block_is_expected_outcome_without_dispatcher_traceback(
    dispatcher_components, caplog,
):
    """#2430: headless permission blocks must not enter ERROR exception flow."""
    agent, registry, dispatcher, _ = dispatcher_components
    blocked = ScheduledTaskOutcome.blocked(
        task_name="restart_coordinator",
        decision="ask",
        reason="operator approval required",
    )

    async def lookup_blocked(name, args):
        return blocked

    for reg in build_cron_registrations(tool_lookup=lookup_blocked):
        registry.register(reg)

    signal = Signal(
        source=cron_source_name("restart_coordinator"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent=agent.did,
    )
    with caplog.at_level(logging.ERROR, logger="kestrel_sovereign.signals.dispatcher"):
        result = await dispatcher.dispatch_signal(signal)

    assert result.status == Status.OK
    assert result.action_result is blocked
    assert not caplog.records


@pytest.mark.parametrize(
    ("case", "task_name", "signal_status", "execution_status", "enabled"),
    [
        ("failed", "sleep", Status.FAILED, "failed", 1),
        ("exception", "sleep", Status.FAILED, "failed", 1),
        ("successful", "sleep", Status.OK, "success", 1),
        ("blocked", "restart_coordinator", Status.OK, "blocked", 0),
    ],
)
@pytest.mark.asyncio
async def test_dispatch_audit_and_scheduler_history_agree_end_to_end(
    dispatcher_components,
    case,
    task_name,
    signal_status,
    execution_status,
    enabled,
):
    """A cron result has one status meaning across dispatch and scheduling."""
    agent, registry, dispatcher, backend = dispatcher_components
    agent.dispatcher = dispatcher

    if case == "failed":
        sleep_report = {
            "success": False,
            "error": "consolidation deadline expired",
        }
    else:
        sleep_report = {"success": True}

    class _SleepReport:
        def to_dict(self):
            return sleep_report

    async def sleep(**kwargs):
        if case == "exception":
            raise RuntimeError("sleep exploded")
        return _SleepReport()

    blocked = ScheduledTaskOutcome.blocked(
        task_name="restart_coordinator",
        decision="ask",
        reason="operator approval required",
    )

    async def lookup(name, args):
        if name == "restart_coordinator":
            return blocked
        raise AssertionError(f"unexpected lookup for {name}")

    agent.sleep = sleep
    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did
    for registration in build_cron_registrations(
        tool_lookup=lookup,
        builtin_handlers={"sleep": feature._handle_sleep},
    ):
        registry.register(registration)

    db = AsyncDatabase(backend)
    runner = SchedulerRunner(db, agent.did, feature._dispatch_scheduled_task)
    await runner._ensure_tables()
    due_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    schedule_id = f"{case}-task"
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json,
             enabled, next_run_at, created_at, scheduler_protocol_version)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?, 2)
        """,
        (
            schedule_id,
            agent.did,
            task_name,
            "@daily",
            '{"skip_reflection": true}',
            due_at,
            due_at,
        ),
    )

    await runner._tick()
    pending = [task for task in agent.background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending)

    signal_row = await backend.fetch_one(
        "SELECT status, error FROM signal_log WHERE source = ?",
        (cron_source_name(task_name),),
    )
    execution_row = await db.fetchone(
        "SELECT status, result_text FROM task_execution_log WHERE task_id = ?",
        (schedule_id,),
    )
    schedule_row = await db.fetchone(
        "SELECT enabled FROM scheduled_tasks WHERE id = ?",
        (schedule_id,),
    )

    assert signal_row is not None
    assert signal_row[0] == signal_status.value
    assert execution_row is not None
    assert execution_row[0] == execution_status
    assert schedule_row == (enabled,)
    if case == "failed":
        assert "scheduled task sleep returned failed" in (signal_row[1] or "")
        assert "scheduled task sleep returned failed" in (execution_row[1] or "")
        assert "consolidation deadline expired" not in (signal_row[1] or "")
    elif case == "exception":
        assert "scheduled task sleep returned failed" in (signal_row[1] or "")
        assert "scheduled task sleep returned failed" in (execution_row[1] or "")
        assert "sleep exploded" not in (signal_row[1] or "")
    elif case == "blocked":
        assert signal_row[1] is None
        assert "operator approval required" in (execution_row[1] or "")


@pytest.mark.parametrize(
    ("case", "task_name", "signal_status", "execution_status"),
    [
        ("error", "trash_retention", Status.FAILED, "failed"),
        ("success", "trash_retention", Status.OK, "success"),
        ("blocked", "github_pr_watch", Status.OK, "success"),
    ],
)
@pytest.mark.asyncio
async def test_builtin_json_envelopes_follow_scheduler_result_contract(
    dispatcher_components,
    case,
    task_name,
    signal_status,
    execution_status,
):
    """Exercise the real built-ins that return JSON object strings.

    A trash sweep exception is a failed occurrence, while successful sweep
    summaries and expected watcher blocks remain ordinary recorded results.
    """
    agent, registry, dispatcher, backend = dispatcher_components
    agent.dispatcher = dispatcher
    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did

    class _TrashStorage:
        async def purge_trash_older_than(self, *args, **kwargs):
            if case == "error":
                raise RuntimeError("DB locked")
            return 4

    agent.storage = _TrashStorage()

    async def run_trash_retention(args):
        with patch(
            "kestrel_sovereign.storage.retention.load_trash_config",
            return_value={"conversation_history_days": 30},
        ):
            return await feature._run_trash_retention(args)

    async def fetch_blocked(*args, **kwargs):
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            PRWatchNetworkError,
        )

        raise PRWatchNetworkError("GitHub timed out")

    async def run_github_pr_watch(args):
        with patch(
            "kestrel_sovereign.features.strategic_memory.github_integration.get_github_token",
            return_value="token",
        ), patch(
            "kestrel_sovereign.signals.sources.github_pr_watch.fetch_pr_state",
            new=fetch_blocked,
        ):
            return await feature._run_github_pr_watch(args)

    handler = (
        run_github_pr_watch if task_name == "github_pr_watch"
        else run_trash_retention
    )

    async def unused_lookup(name, args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=unused_lookup,
        builtin_handlers={task_name: handler},
    ):
        registry.register(registration)

    db = AsyncDatabase(backend)
    runner = SchedulerRunner(db, agent.did, feature._dispatch_scheduled_task)
    await runner._ensure_tables()
    due_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    schedule_id = f"json-envelope-{case}"
    args_json = (
        '{"repo": "owner/repo", "pr": 2907}'
        if case == "blocked"
        else "{}"
    )
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json,
             enabled, next_run_at, created_at, scheduler_protocol_version)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?, 2)
        """,
        (
            schedule_id,
            agent.did,
            task_name,
            "@daily",
            args_json,
            due_at,
            due_at,
        ),
    )

    await runner._tick()
    pending = [task for task in agent.background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending)

    signal_row = await backend.fetch_one(
        "SELECT status, error FROM signal_log WHERE source = ?",
        (cron_source_name(task_name),),
    )
    execution_row = await db.fetchone(
        "SELECT status, result_text FROM task_execution_log WHERE task_id = ?",
        (schedule_id,),
    )

    assert signal_row is not None
    assert signal_row[0] == signal_status.value
    assert execution_row is not None
    assert execution_row[0] == execution_status
    if case == "error":
        assert "scheduled tool trash_retention failed" in (signal_row[1] or "")
        assert "scheduled tool trash_retention failed" in (execution_row[1] or "")
        assert "DB locked" not in (signal_row[1] or "")
    elif case == "success":
        assert signal_row[1] is None
        assert json.loads(execution_row[1])["rows_purged"] == 4
    else:
        assert signal_row[1] is None
        blocked_result = json.loads(execution_row[1])
        assert blocked_result["blocked"] == "network"
        assert "GitHub timed out" in blocked_result["error"]


@pytest.mark.asyncio
async def test_backup_without_sync_service_is_a_successful_skipped_dispatch(
    dispatcher_components,
):
    """The stock local install must not record its four-hour backup as failed."""
    agent, registry, dispatcher, _ = dispatcher_components
    agent._sync_service = None
    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did

    async def unused_lookup(name, args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=unused_lookup,
        builtin_handlers={"backup_snapshot": feature._handle_backup_snapshot},
    ):
        registry.register(registration)

    result = await dispatcher.dispatch_signal(Signal(
        source=cron_source_name("backup_snapshot"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent=agent.did,
    ))

    assert result.status == Status.OK
    assert json.loads(result.action_result) == {
        "skipped": True,
        "reason": "no sync service configured",
    }


@pytest.mark.asyncio
async def test_backup_without_targets_is_a_successful_skipped_dispatch(
    dispatcher_components,
):
    """A live sync service with no targets is a no-op, not a failed backup."""
    agent, registry, dispatcher, _ = dispatcher_components
    agent._sync_service = SimpleNamespace(
        snapshot_if_changed=AsyncMock(return_value={})
    )
    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did

    async def unused_lookup(name, args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=unused_lookup,
        builtin_handlers={"backup_snapshot": feature._handle_backup_snapshot},
    ):
        registry.register(registration)

    result = await dispatcher.dispatch_signal(Signal(
        source=cron_source_name("backup_snapshot"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent=agent.did,
    ))

    assert result.status == Status.OK
    assert json.loads(result.action_result) == {
        "skipped": True,
        "reason": "no sync targets configured",
    }


@pytest.mark.asyncio
async def test_backup_with_failed_targets_is_a_failed_dispatch(
    dispatcher_components,
):
    """A configured backup is successful only when every target succeeds."""
    agent, registry, dispatcher, _ = dispatcher_components
    agent._sync_service = SimpleNamespace(
        snapshot_if_changed=AsyncMock(return_value={
            "gcs": SimpleNamespace(success=False, bytes_synced=0),
            "ipfs": SimpleNamespace(success=False, bytes_synced=0),
        })
    )
    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did

    async def unused_lookup(name, args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=unused_lookup,
        builtin_handlers={"backup_snapshot": feature._handle_backup_snapshot},
    ):
        registry.register(registration)

    result = await dispatcher.dispatch_signal(Signal(
        source=cron_source_name("backup_snapshot"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent=agent.did,
    ))

    assert result.status == Status.FAILED
    assert "scheduled tool backup_snapshot failed" in (result.error or "")


@pytest.mark.asyncio
async def test_failed_sleep_audit_uses_bounded_error_not_raw_report(
    dispatcher_components, caplog,
):
    """Failure routing cannot copy governed/hook result maps into audit error."""
    agent, registry, dispatcher, backend = dispatcher_components
    private_marker = "private-semantic-assertion-2907"

    class _IncompleteReport:
        def to_dict(self):
            return {
                "success": False,
                "error": None,
                "semantic_maintenance": {
                    "status": "partial",
                    "raw_assertion": private_marker,
                },
                "hook_results": [{"third_party_payload": private_marker}],
            }

    agent.sleep = AsyncMock(return_value=_IncompleteReport())
    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did

    async def unused_lookup(name, args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=unused_lookup,
        builtin_handlers={"sleep": feature._handle_sleep},
    ):
        registry.register(registration)

    signal = Signal(
        source=cron_source_name("sleep"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={"skip_consolidation": True, "skip_reflection": True},
        target_agent=agent.did,
    )
    with caplog.at_level(
        logging.ERROR,
        logger="kestrel_sovereign.signals.sources.scheduler",
    ):
        result = await dispatcher.dispatch_signal(signal)

    assert result.status == Status.FAILED
    assert (result.error or "").endswith(
        "scheduled task sleep returned failed"
    )
    assert private_marker not in (result.error or "")
    row = await backend.fetch_one(
        "SELECT error FROM signal_log WHERE id = ?",
        (signal.id,),
    )
    assert row is not None
    assert row[0].endswith("scheduled task sleep returned failed")
    assert private_marker not in row[0]
    assert any(
        private_marker in record.getMessage()
        and "returned failed" in record.getMessage()
        for record in caplog.records
        if record.name == "kestrel_sovereign.signals.sources.scheduler"
    )


@pytest.mark.parametrize(
    "case",
    ["privacy_skip", "export_failure", "explicit_noop"],
)
@pytest.mark.asyncio
async def test_real_sleep_nonterminal_reports_remain_successful_cron_dispatches(
    dispatcher_components,
    case,
):
    """Drive SleepMixin through the real built-in registration wrapper."""
    agent, registry, dispatcher, _ = dispatcher_components
    if case == "privacy_skip":
        agent._consolidate_memories = AsyncMock(return_value={
            "skipped": True,
            "privacy_blocked": True,
        })
        args = {"skip_reflection": True}
    elif case == "export_failure":
        agent._consolidate_memories = AsyncMock(return_value={
            "episodes_created": 1,
        })
        agent._export_sovereignty = AsyncMock(
            side_effect=RuntimeError("remote backup unavailable")
        )
        args = {"skip_reflection": True, "skip_export": False}
    else:
        sweep = AsyncMock()
        agent.storage = SimpleNamespace(
            sweep_expired_governed_semantic_artifacts=sweep
        )
        agent._consolidate_memories = AsyncMock(
            side_effect=AssertionError("consolidation must be skipped")
        )
        args = {"skip_reflection": True, "skip_consolidation": True}

    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did

    async def unused_lookup(name, task_args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=unused_lookup,
        builtin_handlers={"sleep": feature._handle_sleep},
    ):
        registry.register(registration)

    result = await dispatcher.dispatch_signal(Signal(
        source=cron_source_name("sleep"),
        kind="run",
        mode=SignalMode.ACTION,
        payload=args,
        target_agent=agent.did,
    ))

    assert result.status == Status.OK
    payload = json.loads(result.action_result)
    if case == "privacy_skip":
        assert payload["success"] is False
        assert payload["skipped"] is True
        assert payload["error"] == "consolidation_skipped"
    elif case == "export_failure":
        assert payload["success"] is True
        assert "Export failed: remote backup unavailable" in payload["error"]
    else:
        assert payload["skipped"] is True
        assert payload["reason"] == "consolidation and export were both skipped"
        assert payload["skip_reflection"] is True
        agent._consolidate_memories.assert_not_awaited()
        sweep.assert_awaited_once()


@pytest.mark.asyncio
async def test_privacy_skip_does_not_mask_artifact_sweep_failure(
    dispatcher_components,
):
    """A privacy no-op is nonterminal only when no other sleep phase failed."""
    agent, registry, dispatcher, _ = dispatcher_components

    class _FailingSweepStorage:
        async def sweep_expired_governed_semantic_artifacts(self):
            raise RuntimeError("private storage detail")

    agent.storage = _FailingSweepStorage()
    agent._consolidate_memories = AsyncMock(return_value={
        "skipped": True,
        "privacy_blocked": True,
    })
    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did

    async def unused_lookup(name, task_args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=unused_lookup,
        builtin_handlers={"sleep": feature._handle_sleep},
    ):
        registry.register(registration)

    result = await dispatcher.dispatch_signal(Signal(
        source=cron_source_name("sleep"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={"skip_reflection": True},
        target_agent=agent.did,
    ))

    assert result.status == Status.FAILED
    assert (result.error or "").endswith(
        "scheduled task sleep returned failed"
    )
    assert "private storage detail" not in (result.error or "")


@pytest.mark.asyncio
async def test_signal_log_writes_redacted_args(dispatcher_components):
    """Cron payloads are config args. Redaction stores them
    JSON-encoded with a 200-char cap. Verifies args land in the log
    in a debuggable form without blowing the column."""
    agent, registry, dispatcher, backend = dispatcher_components

    async def lookup(name, args):
        return "ok"

    for reg in build_cron_registrations(tool_lookup=lookup):
        registry.register(reg)

    signal = Signal(
        source=cron_source_name("reflect"),
        kind="run",
        mode=SignalMode.ARTIFACT,
        payload={"scope": "all", "depth": "normal"},
        target_agent=agent.did,
    )
    await dispatcher.dispatch_signal(signal)

    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending)

    rows = await backend.fetch_all(
        "SELECT payload_redacted FROM signal_log WHERE source=?",
        (cron_source_name("reflect"),),
    )
    assert len(rows) == 1
    redacted = rows[0][0]
    assert "args=" in redacted
    assert "scope" in redacted
    assert "depth" in redacted


@pytest.mark.asyncio
async def test_concurrent_memory_tasks_serialize(dispatcher_components):
    """Two tasks both declaring MEMORY (e.g. trash_retention while
    memory_consolidate is mid-flight) must serialize. The dispatcher
    acquires their declared resources via the OrderedLockManager."""
    agent, registry, dispatcher, _ = dispatcher_components
    order = []
    order_lock = asyncio.Lock()

    async def lookup(name, args):
        async with order_lock:
            order.append(f"start:{name}")
        await asyncio.sleep(0.05)
        async with order_lock:
            order.append(f"end:{name}")
        return None

    for reg in build_cron_registrations(tool_lookup=lookup):
        registry.register(reg)

    sig_a = Signal(
        source=cron_source_name("trash_retention"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent=agent.did,
    )
    sig_b = Signal(
        source=cron_source_name("memory_consolidate"),
        kind="run",
        mode=SignalMode.ARTIFACT,
        payload={},
        target_agent=agent.did,
    )
    await asyncio.gather(
        dispatcher.dispatch_signal(sig_a),
        dispatcher.dispatch_signal(sig_b),
    )

    # Strict alternation — neither task overlapped the other's critical section.
    assert order in (
        ["start:trash_retention", "end:trash_retention",
         "start:memory_consolidate", "end:memory_consolidate"],
        ["start:memory_consolidate", "end:memory_consolidate",
         "start:trash_retention", "end:trash_retention"],
    ), order
