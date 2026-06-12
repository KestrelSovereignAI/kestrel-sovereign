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

import pytest

from kestrel_sdk.signals import (
    ResourceLock,
    Signal,
    SignalMode,
    Status,
    Trust,
    Visibility,
)
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
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeAgent:
    did = "did:test:scheduler-phase4"

    def __init__(self):
        self.background_tasks = []

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
    """Spec from #893 + #1510 + #1512: the canonical built-in cron task set.
    If a new one is added or one removed, this test fails loudly so
    the classification table stays in sync with the scheduler's
    defaults."""
    names = [t[0] for t in CRON_TASKS]
    assert sorted(names) == sorted([
        "backup_snapshot",
        "morning_signal",
        "signal_dispatch",
        "trash_retention",
        "training_cycle",
        "reflect",
        "memory_consolidate",
        "sleep",  # #1674 P3 — nightly memory-maintenance cycle
        "talon_monitor",  # #1510
        "restart_coordinator",  # #1512
        "github_pr_watch",  # #1618
    ])


def test_action_vs_artifact_split_matches_design():
    """ACTION = side effect, no LLM follow-up. ARTIFACT = produces a
    result via feature workflow that may use an LLM internally but
    doesn't enter conversation history."""
    by_mode = {SignalMode.ACTION: set(), SignalMode.ARTIFACT: set()}
    for name, mode, _ in CRON_TASKS:
        by_mode[mode].add(name)
    assert by_mode[SignalMode.ACTION] == {
        "backup_snapshot",
        "signal_dispatch",
        "trash_retention",
        "training_cycle",
        "talon_monitor",  # #1510 — polls jobs, emits signals, no LLM
        "restart_coordinator",  # #1512 — scans, spawns subprocess
        "github_pr_watch",  # #1618 — polls a PR, emits signal on change
    }
    assert by_mode[SignalMode.ARTIFACT] == {
        "morning_signal",
        "reflect",
        "memory_consolidate",
        "sleep",  # #1674 P3 — returns a SleepReport, no follow-up cognition
    }


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
async def test_action_task_dispatches_through_handler(dispatcher_components):
    """ACTION cron source → dispatcher invokes handler with payload →
    result.action_result carries the return value."""
    agent, registry, dispatcher, _ = dispatcher_components
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
async def test_handler_exception_becomes_failed_status(dispatcher_components):
    """If a tool raises, the dispatcher captures it as Status.FAILED
    with the error message — the runner translates this to
    status='failed' in task_execution_log (verified separately by the
    SchedulerFeature._translate_signal_result test)."""
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
    result = await dispatcher.dispatch_signal(signal)
    assert result.status == Status.FAILED
    assert "tool blew up" in (result.error or "")


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
