"""
Unit tests for the SchedulerFeature and SchedulerRunner.

Tests:
- Feature initialization and tool registration
- Adding, listing, removing, pausing, and resuming scheduled tasks
- Execution history retrieval
- SchedulerRunner tick logic and execution recording
- Error handling for missing DB, invalid cron, etc.
"""

import asyncio
import json
import logging
import pytest
import pytest_asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.signals import (
    CausationFrame,
    SignalHandle,
    SignalMode,
    SignalResult,
    Status,
)
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.agent.sleep import SleepMixin
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, ScheduledTask
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


# =========================================================================
# Helpers
# =========================================================================


def _make_mock_db():
    """Create a mock AsyncDatabase with standard methods."""
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])

    async def _fetchone(sql, *args):
        if "FROM scheduler_protocol_schema" in sql:
            if "provenance" in sql:
                return ("fresh-v2", 2)
            return (2,)
        if "FROM scheduler_protocol_rollout" in sql:
            if "activation_nonce" in sql:
                return (2, "active", None)
            return (2, "active")
        # The runner's exact effect-entry fence is deliberately a real
        # token-guarded read. This generic, storageless double models its
        # otherwise-valid claimed row while leaving unrelated lookups absent.
        if (
            "SELECT 1 FROM scheduled_tasks" in sql
            and "lease_owner = ?" in sql
            and "claim_token = ?" in sql
        ):
            return (1,)
        return None

    db.fetchone = AsyncMock(side_effect=_fetchone)
    db.fetchval = AsyncMock(return_value=0)
    # This generic unit double models a newly-created scheduler schema. Tests
    # that need a legacy table set this explicitly; a preexisting table now
    # correctly requires the durable rollout acknowledgement.
    db.table_exists = AsyncMock(return_value=False)
    return db


def _make_mock_agent(db=None):
    """Create a mock agent with storage.db."""
    agent = MagicMock()
    agent.agent_id = "did:test:scheduler-agent"
    agent.features = {}

    mock_db = db or _make_mock_db()
    agent.storage = MagicMock()
    agent.storage.db = mock_db

    # A watcher's fingerprint checkpoint is committed by a delivery supervisor
    # owned through ``Feature._track_owned_background_task`` → this (#2532).
    # Start REAL tasks and record them so a test can drive a wake to its
    # terminal state; a MagicMock tracker would silently drop the coroutine and
    # the checkpoint assertions would prove nothing.
    agent.tracked_tasks = []

    def _track_background_task(coro, *, name=""):
        task = asyncio.ensure_future(coro)
        agent.tracked_tasks.append((task, name))
        return task

    agent._track_background_task = _track_background_task
    return agent


def _watch_handle(status: Status = Status.OK, *, error=None) -> SignalHandle:
    """A real ``SignalHandle`` resolving to a real ``SignalResult``.

    ``enqueue_signal`` hands the handle back at *acceptance*; the terminal
    status only arrives via ``await handle.wait()``. Watch tests drive the real
    object because a truthy handle mock is unconditionally successful — it can
    only show the happy path is reachable, never that a ``FAILED`` or dropped
    wake leaves the fingerprint un-advanced (#2532).
    """

    async def _terminal() -> SignalResult:
        return SignalResult(
            signal_id="sig-watch",
            status=status,
            mode=SignalMode.COGNITION,
            duration_ms=1,
            error=error,
        )

    return SignalHandle(
        signal_id="sig-watch", task=asyncio.ensure_future(_terminal()),
    )


def _cancelled_watch_handle() -> SignalHandle:
    """A handle whose dispatch task was cancelled out from under the watcher."""

    async def _never() -> SignalResult:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    task = asyncio.ensure_future(_never())
    task.cancel()
    return SignalHandle(signal_id="sig-watch", task=task)


def _wire_watch_dispatcher(feature, handle_factory=_watch_handle):
    """Point the feature's dispatcher at real handles from ``handle_factory``."""
    feature.agent.dispatcher = MagicMock()

    async def _enqueue(_signal):
        return handle_factory()

    feature.agent.dispatcher.enqueue_signal = AsyncMock(side_effect=_enqueue)
    return feature.agent.dispatcher.enqueue_signal


async def _settle_watch_deliveries(feature):
    """Run every watch-checkpoint supervisor this feature owns to completion.

    The watchers deliberately do NOT await delivery inline (they run inside a
    dispatcher worker holding a scheduler lease), so the checkpoint lands in a
    supervisor task. Tests must drive that task before asserting on the
    fingerprint — that boundary is the thing under test.
    """
    supervisors = [
        task for task, name in feature.agent.tracked_tasks
        if name.startswith("watch_checkpoint:")
    ]
    for task in supervisors:
        await asyncio.wait_for(task, timeout=5)
    return supervisors


def _use_postgres_clock(feature, database_now, *, scheduled_row=None):
    """Make one feature exercise the concrete DB-clock path in order."""

    events = []

    @asynccontextmanager
    async def _transaction():
        events.append("transaction_begin")
        try:
            yield
        finally:
            events.append("transaction_end")

    async def _fetchone(sql, params=()):
        if "FROM scheduler_protocol_schema" in sql:
            events.append("schema_lock")
            return (2,)
        if "FROM scheduler_protocol_rollout" in sql:
            events.append("rollout_lock")
            return (2, "active")
        if "SELECT scheduler_protocol_version" in sql:
            events.append("schedule_lock")
            return (2, None)
        if "FROM scheduled_tasks" in sql:
            events.append("schedule_read")
            return scheduled_row
        return None

    async def _fetchval(sql, params=()):
        events.append("database_clock")
        assert sql == "SELECT clock_timestamp()"
        return database_now

    async def _execute(sql, params=()):
        if "INSERT INTO scheduled_tasks" in sql:
            events.append("schedule_insert")
        elif "UPDATE scheduled_tasks" in sql:
            events.append("schedule_update")
        elif "UPDATE task_execution_log" in sql:
            events.append("execution_terminal")
        return 1

    feature._db.backend_type = "postgres"
    feature._db.transaction = _transaction
    feature._db.fetchone = AsyncMock(side_effect=_fetchone)
    feature._db.fetchval = AsyncMock(side_effect=_fetchval)
    feature._db.execute = AsyncMock(side_effect=_execute)
    return events


class _StubJobFeature:
    """Minimal feature stand-in for the disabled-feature scheduler tests.

    Exposes one tool named ``job``, a REAL boolean ``enabled`` flag (a
    MagicMock's ``.enabled`` is a truthy Mock, which would mask the gate), and
    an execution counter so a skipped tick is observable.
    """

    def __init__(self):
        self.name = "JobFeature"
        self.enabled = True
        self.calls = 0

        def _execute(**kwargs):
            self.calls += 1
            return {"success": True, "ran": self.calls}

        tool = MagicMock()
        tool.name = "job"
        tool.execute = AsyncMock(side_effect=_execute)
        self._tool = tool

    def get_tools(self):
        return [self._tool]


# =========================================================================
# Fixtures
# =========================================================================


@pytest_asyncio.fixture
async def feature():
    """Create and initialize a SchedulerFeature with mocked agent/db."""
    agent = _make_mock_agent()
    f = SchedulerFeature(agent)
    # Patch the runner so it does not actually start a background task
    with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
        await f.initialize()
    return f


@pytest_asyncio.fixture
async def feature_no_db():
    """SchedulerFeature with no database available."""
    agent = MagicMock(spec=["agent_id", "did", "features"])
    agent.agent_id = "did:test:no-db"
    agent.features = {}
    f = SchedulerFeature(agent)
    await f.initialize()
    return f


# =========================================================================
# SchedulerFeature tool registration
# =========================================================================


class TestSchedulerTools:
    """Test that the feature exposes the expected tools."""

    @pytest.mark.asyncio
    async def test_feature_has_correct_tools(self, feature):
        tools = feature.get_tools()
        tool_names = {t.name for t in tools}
        assert "schedule_list" in tool_names
        assert "schedule_add" in tool_names
        assert "schedule_remove" in tool_names
        assert "schedule_pause" in tool_names
        assert "schedule_resume" in tool_names
        assert "schedule_history" in tool_names
        assert "schedule_update" in tool_names
        assert "schedule_record_outcome" in tool_names
        assert "schedule_engagement" in tool_names

    @pytest.mark.asyncio
    async def test_tool_description(self, feature):
        desc = feature.tool_description
        assert "scheduled" in desc.lower() or "schedule" in desc.lower()

    @pytest.mark.asyncio
    async def test_shutdown_unregisters_sources_when_runner_propagates_cancellation(
        self, feature,
    ):
        feature._runner.stop = AsyncMock(side_effect=asyncio.CancelledError())
        base_shutdown = AsyncMock()

        with patch.object(Feature, "shutdown", base_shutdown):
            with pytest.raises(asyncio.CancelledError):
                await feature.shutdown()

        base_shutdown.assert_awaited_once()


# =========================================================================
# schedule_list
# =========================================================================


class TestScheduleList:

    @pytest.mark.asyncio
    async def test_list_empty(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        result = await feature.schedule_list()
        assert result.status is ToolResultStatus.OK
        assert result.data["tasks"] == []
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_returns_tasks(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("id-1", "wellness_check", "@daily", "{}", 1, None, "2026-03-06T00:00:00", "2026-03-05T00:00:00"),
            ("id-2", "audit_anchor", "0 */6 * * *", '{"force": true}', 0, "2026-03-05T06:00:00", "2026-03-05T12:00:00", "2026-03-04T00:00:00"),
        ])
        result = await feature.schedule_list()
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 2
        assert result.data["tasks"][0]["task_name"] == "wellness_check"
        assert result.data["tasks"][1]["enabled"] is False
        assert result.data["tasks"][1]["args"] == {"force": True}

    @pytest.mark.asyncio
    async def test_list_exposes_durable_builtin_identity(self, feature):
        """Boot migrations can distinguish a core row from a user row."""

        feature._db.fetchall = AsyncMock(return_value=[(
            "builtin-1",
            "signal_dispatch",
            "5 8 * * *",
            "{}",
            1,
            None,
            "2026-08-14T08:05:00+00:00",
            "2026-08-13T08:00:00+00:00",
            "cron",
            None,
            "UTC",
            "skip",
            None,
            "scheduler:builtin:v1:signal_dispatch",
            None,
            None,
            0,
            None,
            None,
        )])

        result = await feature.schedule_list()

        assert result.status is ToolResultStatus.OK
        assert result.data["tasks"][0]["idempotency_key"] == (
            "scheduler:builtin:v1:signal_dispatch"
        )

    @pytest.mark.asyncio
    async def test_list_no_db(self, feature_no_db):
        result = await feature_no_db.schedule_list()
        assert result.status is ToolResultStatus.ERROR
        assert "not available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_handles_malformed_args_json_as_partial(self, feature):
        """Regression: a single row with malformed args_json must not
        abort the whole list. The migrated code surfaces the bad rows
        as load_errors with a PARTIAL caveat (codex round 2 P2)."""
        feature._db.fetchall = AsyncMock(return_value=[
            ("id-1", "wellness_check", "@daily", "{}", 1, None, None, "2026-03-05T00:00:00"),
            ("id-2", "broken", "@hourly", "not-json", 1, None, None, "2026-03-05T00:00:00"),
            ("id-3", "list-args", "@daily", "[1,2,3]", 1, None, None, "2026-03-05T00:00:00"),
        ])
        result = await feature.schedule_list()

        assert result.status is ToolResultStatus.PARTIAL
        # All 3 tasks still listed (no row dropped).
        assert result.data["count"] == 3
        # The good one keeps its args; the bad ones get empty {}.
        names_to_args = {t["task_name"]: t["args"] for t in result.data["tasks"]}
        assert names_to_args["wellness_check"] == {}
        assert names_to_args["broken"] == {}
        assert names_to_args["list-args"] == {}
        # load_errors carries both bad rows.
        bad_ids = {e["task_id"] for e in result.data["load_errors"]}
        assert bad_ids == {"id-2", "id-3"}

    @pytest.mark.asyncio
    async def test_list_exposes_durable_execution_and_misfire_state(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            (
                "deadline-1", "workflow_run", "", "{}", 0, None, None,
                "2026-07-24T00:00:00+00:00", "one_shot",
                "2026-07-24T01:00:00+00:00", "UTC", "fire_once", 30,
                "workflow-deadline", None, None, 2, "success",
                "2026-07-24T01:00:00+00:00",
            ),
        ])

        result = await feature.schedule_list()
        task = result.data["tasks"][0]
        assert task["schedule_kind"] == "one_shot"
        assert task["misfire_policy"] == "fire_once"
        assert task["idempotency_key"] == "workflow-deadline"
        assert task["attempt_count"] == 2
        assert task["terminal_status"] == "success"


# =========================================================================
# post_all_features_loaded — retired-cron cutover cleanup (#1674)
# =========================================================================


class TestRetiredCronCleanup:
    @pytest.mark.asyncio
    async def test_post_load_removes_retired_builtin_schedules(self):
        """Persisted rows for removed core sources must be deleted on upgrade."""
        from kestrel_sdk.tools.result import ToolResult

        agent = _make_mock_agent()
        f = SchedulerFeature(agent)
        with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
            await f.initialize()

        # Mirror schedule_list's real envelope: the row id is under "id".
        f.schedule_list = AsyncMock(return_value=ToolResult.ok(
            confirmation="ok",
            data={"tasks": [
                {"task_name": "cognition_retention", "id": "orphan-1"},
                {"task_name": "backup_snapshot", "id": "keep-1"},
            ]},
        ))
        f.schedule_remove = AsyncMock(return_value=ToolResult.ok(confirmation="removed"))
        f._ensure_builtin_schedule = AsyncMock(return_value=ToolResult.ok(
            confirmation="added", data={"next_run_at": None}))

        await f.post_all_features_loaded(agent)

        # The orphaned built-in was removed by id...
        removed_ids = {call.args[0] for call in f.schedule_remove.await_args_list}
        assert removed_ids == {"orphan-1"}
        # ...and never re-seeded (it's no longer a default).
        readded = [
            c.kwargs.get("task_name")
            for c in f._ensure_builtin_schedule.await_args_list
        ]
        assert "cognition_retention" not in readded
        # Existing defaults still take the authoritative transaction path so
        # a second pending host adopts a first host's registration-owned row.
        assert "backup_snapshot" in readded

    @pytest.mark.asyncio
    async def test_post_load_pauses_only_legacy_discovery_watches(self, caplog):
        """Rows from the implicit-provider era cannot keep failing each tick.

        Missing/blank ``args.tool`` rows are paused with an actionable
        migration diagnostic. A watch that already names a feature-owned tool
        is left intact.
        """
        from kestrel_sdk.tools.result import ToolResult

        agent = _make_mock_agent()
        f = SchedulerFeature(agent)
        with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
            await f.initialize()

        f.schedule_list = AsyncMock(return_value=ToolResult.ok(
            confirmation="ok",
            data={"tasks": [
                {
                    "task_name": "ecosystem_discovery_watch",
                    "id": "legacy-missing",
                    "cron_expression": "*/15 * * * *",
                    "args": {"repo": "owner/repo"},
                    "enabled": True,
                },
                {
                    "task_name": "ecosystem_discovery_watch",
                    "id": "legacy-blank",
                    "cron_expression": "*/20 * * * *",
                    "args": {"tool": "   ", "org": "owner"},
                    "enabled": True,
                },
                {
                    "task_name": "ecosystem_discovery_watch",
                    "id": "legacy-already-paused",
                    "cron_expression": "*/25 * * * *",
                    "args": {"repo": "owner/repo"},
                    "enabled": False,
                },
                {
                    "task_name": "ecosystem_discovery_watch",
                    "id": "configured",
                    "cron_expression": "*/30 * * * *",
                    "args": {
                        "tool": "discover_ecosystem",
                        "tool_args": {"repo": "owner/repo"},
                    },
                    "enabled": True,
                },
            ]},
        ))
        f.schedule_remove = AsyncMock(
            return_value=ToolResult.ok(confirmation="removed")
        )
        f.schedule_pause = AsyncMock(
            return_value=ToolResult.ok(confirmation="paused")
        )
        f._ensure_builtin_schedule = AsyncMock(
            return_value=ToolResult.ok(
                confirmation="added", data={"next_run_at": None}
            )
        )

        with caplog.at_level(logging.WARNING):
            await f.post_all_features_loaded(agent)

        paused_ids = {call.args[0] for call in f.schedule_pause.await_args_list}
        assert paused_ids == {"legacy-missing", "legacy-blank"}
        assert "legacy-already-paused" not in paused_ids
        assert "configured" not in paused_ids
        assert "core no longer supplies an implicit discovery tool" in caplog.text
        assert 'args_json containing an explicit enabled feature-owned tool' in caplog.text

    @pytest.mark.asyncio
    async def test_post_load_removes_autoseeded_consolidate_reflect_keeps_custom(self):
        """#1674 P3: the nightly `sleep` cycle supersedes the auto-seeded
        memory_consolidate + reflect crons. post_all_features_loaded removes
        rows that EXACTLY match the old auto-seed (name+cron+args) but leaves a
        user-customized schedule intact, and seeds `sleep` when MemoryFeature
        is present."""
        from kestrel_sdk.tools.result import ToolResult

        agent = _make_mock_agent()
        agent.features = {"MemoryFeature"}
        f = SchedulerFeature(agent)
        with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
            await f.initialize()

        f.schedule_list = AsyncMock(return_value=ToolResult.ok(
            confirmation="ok",
            data={"tasks": [
                # Auto-seeded defaults (exact match) → removed.
                {"task_name": "memory_consolidate", "id": "mc-auto",
                 "cron_expression": "0 4 * * *", "args": {}},
                {"task_name": "reflect", "id": "rf-auto",
                 "cron_expression": "0 */4 * * *",
                 "args": {"scope": "all", "depth": "normal"}},
                # User-customized memory_consolidate (different cron) → kept.
                {"task_name": "memory_consolidate", "id": "mc-custom",
                 "cron_expression": "30 2 * * *", "args": {}},
            ]},
        ))
        f.schedule_remove = AsyncMock(return_value=ToolResult.ok(confirmation="removed"))
        f._ensure_builtin_schedule = AsyncMock(return_value=ToolResult.ok(
            confirmation="added", data={"next_run_at": None}))

        await f.post_all_features_loaded(agent)

        removed_ids = {c.args[0] for c in f.schedule_remove.await_args_list}
        assert "mc-auto" in removed_ids and "rf-auto" in removed_ids
        assert "mc-custom" not in removed_ids  # user schedule preserved
        seeded = [
            c.kwargs.get("task_name")
            for c in f._ensure_builtin_schedule.await_args_list
        ]
        assert "sleep" in seeded                # the one memory-maintenance cron
        assert "memory_consolidate" not in seeded
        assert "reflect" not in seeded

    @pytest.mark.asyncio
    async def test_post_load_removes_only_builtin_signal_dispatch_identity_once(self):
        """The former built-in row is removed without retiring the live tool.

        A user-created row at the identical 08:05 cadence and payload has its
        own durable identity and survives this and repeated boots unchanged.
        """
        from kestrel_sdk.tools.result import ToolResult

        agent = _make_mock_agent()
        f = SchedulerFeature(agent)
        with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
            await f.initialize()

        legacy_builtin = {
            "task_name": "signal_dispatch",
            "id": "dispatch-old-autoseed",
            "cron_expression": "5 8 * * *",
            "args": {},
            "idempotency_key": "scheduler:builtin:v1:signal_dispatch",
        }
        same_cadence_user_row = {
            "task_name": "signal_dispatch",
            "id": "dispatch-user-same-cadence",
            "cron_expression": "5 8 * * *",
            "args": {},
            "idempotency_key": "schedule:user-created-dispatch",
        }
        f.schedule_list = AsyncMock(side_effect=[
            ToolResult.ok(
                confirmation="first boot",
                data={"tasks": [legacy_builtin, same_cadence_user_row]},
            ),
            ToolResult.ok(
                confirmation="second boot",
                data={"tasks": [same_cadence_user_row]},
            ),
        ])
        f.schedule_remove = AsyncMock(
            return_value=ToolResult.ok(confirmation="removed")
        )
        f._ensure_builtin_schedule = AsyncMock(
            return_value=ToolResult.ok(
                confirmation="added", data={"next_run_at": None}
            )
        )

        await f.post_all_features_loaded(agent)
        await f.post_all_features_loaded(agent)

        f.schedule_remove.assert_awaited_once_with("dispatch-old-autoseed")
        seeded = {
            call.kwargs["task_name"]
            for call in f._ensure_builtin_schedule.await_args_list
        }
        assert "signal_dispatch" not in seeded


class TestSleepCronHandler:
    @pytest.mark.asyncio
    async def test_handle_sleep_runs_maintenance_when_core_work_is_disabled(self):
        class _Report:
            def to_dict(self):
                return {"success": False, "error": None}

        agent = _make_mock_agent()
        agent.sleep = AsyncMock(return_value=_Report())
        f = SchedulerFeature(agent)

        out = await f._handle_sleep({
            "skip_consolidation": True,
            "skip_reflection": False,
        })

        assert json.loads(out) == {
            "success": False,
            "error": None,
            "skipped": True,
            "reason": "consolidation and export were both skipped",
            "skip_reflection": False,
        }
        agent.sleep.assert_awaited_once_with(
            skip_export=True,
            skip_consolidation=True,
            skip_reflection=False,
        )

    @pytest.mark.asyncio
    async def test_handle_sleep_fails_incomplete_maintenance_only_cycle(self):
        class _Report:
            def to_dict(self):
                return {
                    "success": False,
                    "error": None,
                    "semantic_maintenance": {"status": "partial"},
                }

        agent = _make_mock_agent()
        agent.sleep = AsyncMock(return_value=_Report())
        feature = SchedulerFeature(agent)

        outcome = await feature._handle_sleep({
            "skip_consolidation": True,
            "skip_reflection": True,
        })

        assert isinstance(outcome, ScheduledTaskOutcome)
        assert outcome.status == "failed"
        assert json.loads(outcome.result_text)["semantic_maintenance"] == {
            "status": "partial"
        }

    @pytest.mark.asyncio
    async def test_handle_sleep_logs_raised_cycle_locally_with_traceback(
        self, caplog
    ):
        agent = _make_mock_agent()
        agent.sleep = AsyncMock(side_effect=RuntimeError("sleep exploded"))
        feature = SchedulerFeature(agent)
        feature._agent_id = "did:test:sleep-diagnostics"

        with caplog.at_level(logging.ERROR):
            outcome = await feature._handle_sleep({"skip_reflection": True})

        assert isinstance(outcome, ScheduledTaskOutcome)
        assert json.loads(outcome.result_text) == {"error": "sleep_failed"}
        record = next(
            record
            for record in caplog.records
            if record.message
            == "[sleep] agent=did:test:sleep-diagnostics cycle failed"
        )
        assert record.exc_info is not None
        assert "sleep exploded" in str(record.exc_info[1])

    @pytest.mark.asyncio
    async def test_handle_sleep_calls_agent_sleep_skip_export(self):
        """The sleep cron handler runs the agent's sleep cycle with
        skip_export=True (backups own DR) and surfaces the report."""
        class _Report:
            def to_dict(self):
                return {"success": True, "consolidation": {"episodes_deleted": 2}}

        agent = _make_mock_agent()
        agent.sleep = AsyncMock(return_value=_Report())
        agent.storage = MagicMock()
        # No messages at all → genuinely idle → reflection gated off. The gate
        # queries conversation MAX(created_at) first; None → idle (early return).
        agent.storage.db = MagicMock()
        agent.storage.db.fetchval = AsyncMock(side_effect=[None])
        f = SchedulerFeature(agent)
        with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
            await f.initialize()

        out = await f._handle_sleep({})

        agent.sleep.assert_awaited_once()
        kwargs = agent.sleep.await_args.kwargs
        assert kwargs["skip_export"] is True
        assert kwargs["skip_reflection"] is True  # idle → reflection skipped
        assert '"success": true' in out

    @pytest.mark.asyncio
    async def test_handle_sleep_reports_structured_consolidation_failure(self):
        """The default no-export cron must not report a failed pass as success."""
        class _SleepAgent(SleepMixin):
            pass

        agent = _SleepAgent()
        agent.agent_id = "did:test:scheduler-agent"
        agent.did = agent.agent_id
        agent.features = {}
        agent.storage = MagicMock()
        agent.storage.sweep_expired_governed_semantic_artifacts = None
        agent.storage.db = _make_mock_db()
        agent.sleep_hooks = []
        agent._consolidate_memories = AsyncMock(
            return_value={"error": "provider unavailable"}
        )
        f = SchedulerFeature(agent)
        with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
            await f.initialize()

        outcome = await f._handle_sleep({"skip_reflection": True})

        agent._consolidate_memories.assert_awaited_once()
        assert isinstance(outcome, ScheduledTaskOutcome)
        assert outcome.status == "failed"
        assert outcome.pause_schedule is False
        payload = json.loads(outcome.result_text)
        assert payload["success"] is False
        assert payload["error"] == "consolidation_failed"

    @pytest.mark.asyncio
    async def test_handle_sleep_exception_is_failed_without_pausing_schedule(self):
        agent = _make_mock_agent()
        agent.sleep = AsyncMock(side_effect=RuntimeError("sleep exploded"))
        agent.storage = MagicMock()
        agent.storage.db = MagicMock()
        agent.storage.db.fetchval = AsyncMock(return_value=None)
        f = SchedulerFeature(agent)
        with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
            await f.initialize()

        outcome = await f._handle_sleep({"skip_reflection": True})

        assert isinstance(outcome, ScheduledTaskOutcome)
        assert outcome.status == "failed"
        assert outcome.pause_schedule is False
        assert json.loads(outcome.result_text) == {"error": "sleep_failed"}

    @pytest.mark.asyncio
    async def test_handle_sleep_reflects_when_active(self):
        agent = _make_mock_agent()

        class _Report:
            def to_dict(self):
                return {"success": True}

        agent.sleep = AsyncMock(return_value=_Report())
        agent.storage = MagicMock()
        agent.storage.db = MagicMock()
        # Gate queries conversation MAX first, then episode MAX. Newest message
        # (Jun 12, space-format) is newer than the last episode (Jun 1, ISO/tz)
        # → active even across the two on-disk timestamp formats.
        agent.storage.db.fetchval = AsyncMock(
            side_effect=["2026-06-12 10:00:00", "2026-06-01T00:00:00+00:00"])
        f = SchedulerFeature(agent)
        with patch.object(SchedulerRunner, "start", new_callable=AsyncMock):
            await f.initialize()

        await f._handle_sleep({})
        assert agent.sleep.await_args.kwargs["skip_reflection"] is False


class TestSleepActivityGateSoftDelete:
    """The activity gate must measure *live* conversation activity only.

    Soft-deleted rows (deleted_at IS NOT NULL) are not real activity, so a
    soft-delete as the sole change since the last episode must read as idle
    (#2061). The gate biases toward "active" on uncertainty, so the only way
    to get this wrong is to count deleted rows.
    """

    @staticmethod
    def _make_feature(rows):
        """Build a SchedulerFeature whose db.fetchval honors the soft-delete
        filter the way real SQLite would.

        ``rows`` is a list of (created_at, deleted_at) conversation rows. The
        fake fetchval returns MAX(created_at) over the rows that match the
        query's WHERE clause — respecting ``deleted_at IS NULL`` when present.
        """

        async def fetchval(query, params=()):
            if "memory_episodes" in query:
                return None  # no prior episode → "newest msg" decides
            candidates = rows
            if "deleted_at IS NULL" in query:
                candidates = [r for r in rows if r[1] is None]
            stamps = [r[0] for r in candidates]
            return max(stamps) if stamps else None

        agent = _make_mock_agent()
        agent.storage = MagicMock()
        agent.storage.db = MagicMock()
        agent.storage.db.fetchval = AsyncMock(side_effect=fetchval)
        f = SchedulerFeature(agent)
        f._agent_id = "did:test:scheduler-agent"
        return f

    @pytest.mark.asyncio
    async def test_soft_deleted_only_message_is_idle(self):
        # The only row newer than the (absent) last episode is soft-deleted.
        f = self._make_feature([("2026-06-12 10:00:00", "2026-06-12 11:00:00")])
        assert await f._sleep_had_activity() is False

    @pytest.mark.asyncio
    async def test_live_message_is_active(self):
        # A live (non-deleted) message → active, even alongside a deleted one.
        f = self._make_feature(
            [
                ("2026-06-11 09:00:00", "2026-06-11 09:30:00"),
                ("2026-06-12 10:00:00", None),
            ]
        )
        assert await f._sleep_had_activity() is True

    @pytest.mark.asyncio
    async def test_query_filters_deleted_at(self):
        # Guard the literal filter so the fix can't silently regress.
        captured = []

        async def fetchval(query, params=()):
            captured.append(query)
            return None

        agent = _make_mock_agent()
        agent.storage = MagicMock()
        agent.storage.db = MagicMock()
        agent.storage.db.fetchval = AsyncMock(side_effect=fetchval)
        f = SchedulerFeature(agent)
        f._agent_id = "did:test:scheduler-agent"
        await f._sleep_had_activity()

        conv_query = next(q for q in captured if "conversation_history" in q)
        assert "deleted_at IS NULL" in conv_query


# =========================================================================
# schedule_add
# =========================================================================


class TestScheduleAdd:

    @pytest.mark.asyncio
    async def test_add_valid_task(self, feature):
        # memory_consolidate is a registered built-in cron source, so it
        # passes the #1618 task-name validation.
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="memory_consolidate",
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["task_name"] == "memory_consolidate"
        assert result.data["cron_expression"] == "@daily"
        assert result.data["task_id"] is not None
        assert result.data["next_run_at"] is not None

        # Verify DB insert was called
        feature._db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_add_with_args(self, feature):
        result = await feature.schedule_add(
            cron_expression="*/15 * * * *",
            task_name="memory_consolidate",
            args_json='{"threshold": 100}',
        )
        assert result.status is ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_add_anchors_first_cron_occurrence_to_database_clock(self, feature):
        """A skewed API host cannot choose a different first minute than PG."""

        database_now = datetime(2026, 7, 25, 8, 0, 30, tzinfo=timezone.utc)
        events = _use_postgres_clock(feature, database_now)

        # If schedule_add used the API process wall clock, this would select a
        # 2040 occurrence instead of 08:01 on the scheduler's database clock.
        with patch(
            "kestrel_sovereign.features.scheduler.feature.datetime"
        ) as host_datetime:
            host_datetime.now.return_value = datetime(
                2040, 1, 1, 0, 0, tzinfo=timezone.utc
            )
            result = await feature.schedule_add(
                cron_expression="* * * * *",
                task_name="memory_consolidate",
            )

        assert result.status is ToolResultStatus.OK
        assert result.data["next_run_at"] == "2026-07-25T08:01:00+00:00"
        assert result.data["created_at"] == database_now.isoformat()
        assert events == [
            "transaction_begin",
            "schema_lock",
            "rollout_lock",
            "database_clock",
            "schedule_insert",
            "transaction_end",
        ]

    @pytest.mark.asyncio
    async def test_add_timezone_aware_cron_persists_policy_and_identity(self, feature):
        result = await feature.schedule_add(
            cron_expression="30 9 * * *",
            task_name="memory_consolidate",
            timezone_name="America/Chicago",
            misfire_policy="fire_once",
            idempotency_key="daily-memory",
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["timezone"] == "America/Chicago"
        assert result.data["misfire_policy"] == "fire_once"
        assert result.data["idempotency_key"] == "daily-memory"

    @pytest.mark.asyncio
    async def test_add_accepts_idempotency_base_at_447_utf8_byte_boundary(self, feature):
        """447 + ':' + SHA-256 hex exactly fits the SDK's 512-byte cap."""

        ascii_boundary = "a" * 447
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="memory_consolidate",
            idempotency_key=ascii_boundary,
        )

        assert result.status is ToolResultStatus.OK
        assert result.data["idempotency_key"] == ascii_boundary

        # A multibyte key with the same UTF-8 byte length is accepted too;
        # validation is deliberately bytes, not Python character count.
        multibyte_boundary = ("é" * 223) + "a"  # 446 + 1 bytes
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="memory_consolidate",
            idempotency_key=multibyte_boundary,
        )
        assert result.status is ToolResultStatus.OK
        assert len(result.data["idempotency_key"].encode("utf-8")) == 447

    @pytest.mark.asyncio
    async def test_add_rejects_idempotency_base_over_447_utf8_bytes(self, feature):
        for key in ("a" * 448, "é" * 224):
            result = await feature.schedule_add(
                cron_expression="@daily",
                task_name="memory_consolidate",
                idempotency_key=key,
            )
            assert result.status is ToolResultStatus.ERROR
            assert "at most 447 bytes" in result.error
        feature._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_deadline_is_one_shot(self, feature):
        result = await feature.schedule_add_deadline(
            run_at="2026-12-01T12:00:00+00:00",
            task_name="memory_consolidate",
            idempotency_key="workflow-deadline-9",
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["schedule_kind"] == "one_shot"
        assert result.data["run_at"] == "2026-12-01T12:00:00+00:00"
        assert result.data["idempotency_key"] == "workflow-deadline-9"

    @pytest.mark.asyncio
    async def test_add_rejects_unknown_iana_timezone(self, feature):
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="memory_consolidate",
            timezone_name="Mars/Olympus_Mons",
        )
        assert result.status is ToolResultStatus.ERROR
        assert "timezone" in result.error.lower()

    @pytest.mark.asyncio
    async def test_add_invalid_cron(self, feature):
        result = await feature.schedule_add(
            cron_expression="bad cron",
            task_name="test_task",
        )
        assert result.status is ToolResultStatus.ERROR
        assert "invalid cron" in result.error.lower() or "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_add_invalid_args_json(self, feature):
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="test_task",
            args_json="not valid json",
        )
        assert result.status is ToolResultStatus.ERROR
        assert "args_json" in result.error.lower() or "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_add_args_json_must_be_object(self, feature):
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="test_task",
            args_json='[1, 2, 3]',
        )
        assert result.status is ToolResultStatus.ERROR
        assert "object" in result.error.lower()

    @pytest.mark.asyncio
    async def test_add_no_db(self, feature_no_db):
        result = await feature_no_db.schedule_add(
            cron_expression="@daily",
            task_name="test_task",
        )
        assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_add_rejects_unknown_task_name(self, feature):
        """#1618: an unregistered task name must be rejected at creation
        time instead of silently entering the schedule and failing every
        tick with 'Unknown task' (the github_pr_watch incident)."""
        result = await feature.schedule_add(
            cron_expression="*/15 * * * *",
            task_name="totally_made_up_task",
        )
        assert result.status is ToolResultStatus.ERROR
        assert "unknown scheduled task" in result.error.lower()
        assert "totally_made_up_task" in result.error
        # The caller gets the list of valid names to fix the typo.
        assert "totally_made_up_task" not in result.data["valid_task_names"]
        assert "memory_consolidate" in result.data["valid_task_names"]
        # And nothing was inserted into the schedule.
        feature._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_github_pr_watch_is_valid(self, feature):
        """#1618: github_pr_watch is now a registered built-in cron task,
        so scheduling it succeeds (the original bug was that it was not
        registered)."""
        result = await feature.schedule_add(
            cron_expression="*/15 * * * *",
            task_name="github_pr_watch",
            args_json='{"repo": "owner/name", "pr": 1614}',
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["task_name"] == "github_pr_watch"

    @pytest.mark.asyncio
    async def test_add_accepts_loaded_feature_tool(self, feature):
        """A tool exposed by a loaded feature is a valid scheduled task
        even though it isn't a built-in cron source."""
        mock_tool = MagicMock()
        mock_tool.name = "wellness_check"
        mock_feature = MagicMock()
        mock_feature.get_tools = MagicMock(return_value=[mock_tool])
        feature.agent.features = {"WellnessFeature": mock_feature}

        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="wellness_check",
        )
        assert result.status is ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_add_rejects_authority_bound_restart_tool(self, feature):
        """An unattended scheduler tick cannot supply sovereign authority."""

        restart_tool = MagicMock()
        restart_tool.name = "request_restart"
        restart_feature = MagicMock()
        restart_feature.get_tools = MagicMock(return_value=[restart_tool])
        feature.agent.features = {"RestartCoordinatorFeature": restart_feature}

        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="request_restart",
        )

        assert result.status is ToolResultStatus.ERROR
        assert "unknown scheduled task" in result.error.lower()
        assert "request_restart" not in result.data["valid_task_names"]
        feature._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_rejects_disabled_feature_tool(self, feature):
        """#2522: a soft-disabled feature's tool is not executable, so it is
        not schedulable — ``schedule_add`` rejects it as an unknown task (a
        persisted row would otherwise be skipped every tick). Re-enabling the
        feature makes the same name schedulable again."""
        job_feature = _StubJobFeature()
        job_feature.enabled = False
        feature.agent.features = {"JobFeature": job_feature}

        result = await feature.schedule_add(
            cron_expression="@daily", task_name="job",
        )
        assert result.status is ToolResultStatus.ERROR
        assert "unknown scheduled task" in result.error.lower()
        assert "job" not in set(result.data["valid_task_names"])
        feature._db.execute.assert_not_called()

        # Re-enable → the same tool name is now a valid scheduled task.
        job_feature.enabled = True
        result = await feature.schedule_add(
            cron_expression="@daily", task_name="job",
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["task_name"] == "job"

    @pytest.mark.asyncio
    async def test_add_rejects_deny_listed_tool(self, feature):
        """F245: a tool the SecurityFeature permission store has set to
        DENY must be rejected at creation time — persisting it would just
        guarantee the tick-path PRE_TOOL_USE gate blocks every fire. The
        schedule row must NOT be inserted."""
        from kestrel_sovereign.features.security.permissions import PermissionLevel

        mock_tool = MagicMock()
        mock_tool.name = "dangerous_op"
        mock_feature = MagicMock()
        mock_feature.get_tools = MagicMock(return_value=[mock_tool])

        # SecurityFeature exposing a permission_store that DENYs the tool.
        security_feature = MagicMock()
        security_feature.permission_store = MagicMock()
        security_feature.permission_store.get_permission = AsyncMock(
            return_value=PermissionLevel.DENY
        )

        feature.agent.features = {
            "DangerFeature": mock_feature,
            "SecurityFeature": security_feature,
        }

        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="dangerous_op",
        )

        assert result.status is ToolResultStatus.ERROR
        assert "deny" in result.error.lower()
        assert result.data["denied_by_policy"] is True
        # The permission store was consulted under the registered feature name.
        security_feature.permission_store.get_permission.assert_awaited_once_with(
            "DangerFeature", "dangerous_op"
        )
        # And nothing was inserted into the schedule.
        feature._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_allows_non_deny_tool_with_security_present(self, feature):
        """An ASK/ALLOW-gated tool is still schedulable — only DENY is a
        hard creation-time block. Guards against the DENY check
        over-rejecting."""
        from kestrel_sovereign.features.security.permissions import PermissionLevel

        mock_tool = MagicMock()
        mock_tool.name = "wellness_check"
        mock_feature = MagicMock()
        mock_feature.get_tools = MagicMock(return_value=[mock_tool])

        security_feature = MagicMock()
        security_feature.permission_store = MagicMock()
        security_feature.permission_store.get_permission = AsyncMock(
            return_value=PermissionLevel.ASK
        )

        feature.agent.features = {
            "WellnessFeature": mock_feature,
            "SecurityFeature": security_feature,
        }

        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="wellness_check",
        )
        assert result.status is ToolResultStatus.OK
        feature._db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_feature_get_tools_failure_is_logged_not_swallowed(
        self, feature, caplog
    ):
        """#1640: if a loaded feature's get_tools() raises while the
        scheduler collects valid task names, the failure must be logged
        (not silently swallowed) and must not block validation of the
        other features. A silent skip drops that feature's names from the
        'every currently-valid name' set schedule_add's rejection error
        promises, so a legitimate name would be rejected with no trace of
        why."""
        bad_feature = MagicMock()
        bad_feature.name = "BadFeature"
        bad_feature.get_tools = MagicMock(side_effect=RuntimeError("boom"))

        good_tool = MagicMock()
        good_tool.name = "wellness_check"
        good_feature = MagicMock()
        good_feature.get_tools = MagicMock(return_value=[good_tool])

        feature.agent.features = {
            "BadFeature": bad_feature,
            "GoodFeature": good_feature,
        }

        with caplog.at_level(
            logging.WARNING,
            logger="kestrel_sovereign.features.scheduler.feature",
        ):
            # The healthy feature's tool is still collected despite the
            # broken one -- one bad feature can't block the rest.
            result = await feature.schedule_add(
                cron_expression="@daily",
                task_name="wellness_check",
            )

        assert result.status is ToolResultStatus.OK
        # The broken feature was reported, not silently swallowed, and the
        # warning names the offending feature so the omission is traceable.
        assert any(
            rec.levelno == logging.WARNING and "BadFeature" in rec.getMessage()
            for rec in caplog.records
        ), caplog.text


# =========================================================================
# schedule_remove
# =========================================================================


class TestScheduleRemove:

    @pytest.mark.asyncio
    async def test_remove_existing(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id",))
        result = await feature.schedule_remove(task_id="task-id")
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "removed"

    @pytest.mark.asyncio
    async def test_remove_locks_schedule_before_execution_log(self, feature):
        """Match runner finalization's schedule-row-then-log lock order."""
        feature._db.fetchone = AsyncMock(return_value=("task-id",))

        result = await feature.schedule_remove(task_id="task-id")

        assert result.status is ToolResultStatus.OK
        statements = [call.args[0] for call in feature._db.execute.await_args_list]
        schedule_mutation = next(
            index
            for index, statement in enumerate(statements)
            if "DELETE FROM scheduled_tasks" in statement
        )
        log_terminalization = next(
            index
            for index, statement in enumerate(statements)
            if "UPDATE task_execution_log" in statement
        )
        assert schedule_mutation < log_terminalization

    @pytest.mark.asyncio
    async def test_remove_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_remove(task_id="nonexistent")
        assert result.status is ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_remove_no_db(self, feature_no_db):
        result = await feature_no_db.schedule_remove(task_id="any")
        assert result.status is ToolResultStatus.ERROR


# =========================================================================
# schedule_pause
# =========================================================================


class TestSchedulePause:

    @pytest.mark.asyncio
    async def test_pause_active_task(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id", 1))
        result = await feature.schedule_pause(task_id="task-id")
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "paused"

    @pytest.mark.asyncio
    async def test_pause_already_paused(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id", 0))
        result = await feature.schedule_pause(task_id="task-id")
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "already_paused"

    @pytest.mark.asyncio
    async def test_pause_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_pause(task_id="nonexistent")
        assert result.status is ToolResultStatus.ERROR


# =========================================================================
# schedule_resume
# =========================================================================


class TestScheduleResume:

    @pytest.mark.asyncio
    async def test_resume_paused_task(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id", 0, "@daily"))
        result = await feature.schedule_resume(task_id="task-id")
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "resumed"
        assert result.data["next_run_at"] is not None

    @pytest.mark.asyncio
    async def test_resume_already_running(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id", 1, "@daily"))
        result = await feature.schedule_resume(task_id="task-id")
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "already_running"

    @pytest.mark.asyncio
    async def test_resume_reanchors_cron_occurrence_to_database_clock(self, feature):
        """A resumed schedule must use the runner's clock, not host time."""

        database_now = datetime(2026, 7, 25, 8, 0, 30, tzinfo=timezone.utc)
        events = _use_postgres_clock(
            feature,
            database_now,
            scheduled_row=(
                "task-id", 0, "* * * * *", "cron", None, "UTC", None, 0, 0,
            ),
        )

        with patch(
            "kestrel_sovereign.features.scheduler.feature.datetime"
        ) as host_datetime:
            host_datetime.now.return_value = datetime(
                2040, 1, 1, 0, 0, tzinfo=timezone.utc
            )
            result = await feature.schedule_resume(task_id="task-id")

        assert result.status is ToolResultStatus.OK
        assert result.data["next_run_at"] == "2026-07-25T08:01:00+00:00"
        assert events[:6] == [
            "transaction_begin",
            "schema_lock",
            "rollout_lock",
            "schedule_lock",
            "schedule_read",
            "database_clock",
        ]
        assert events[-1] == "transaction_end"

    @pytest.mark.asyncio
    async def test_resume_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_resume(task_id="nonexistent")
        assert result.status is ToolResultStatus.ERROR


# =========================================================================
# schedule_history
# =========================================================================


class TestScheduleHistory:

    @pytest.mark.asyncio
    async def test_history_empty(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        result = await feature.schedule_history()
        assert result.status is ToolResultStatus.OK
        assert result.data["executions"] == []
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_history_returns_records(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("exec-1", "task-1", "success", '{"result": "ok"}', 150, "2026-03-05T10:00:00", "wellness_check", 0.9),
            ("exec-2", "task-2", "failed", "Connection timeout", 5000, "2026-03-05T09:00:00", "audit_anchor", None),
        ])
        result = await feature.schedule_history()
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 2
        assert result.data["executions"][0]["status"] == "success"
        assert result.data["executions"][0]["outcome_signal"] == 0.9
        assert result.data["executions"][1]["task_name"] == "audit_anchor"
        assert result.data["executions"][1]["outcome_signal"] is None

    @pytest.mark.asyncio
    async def test_history_respects_limit(self, feature):
        await feature.schedule_history(limit=5)
        # Verify the limit was passed to the DB query
        call_args = feature._db.fetchall.call_args
        assert call_args[0][1][1] == 5  # second positional param tuple


# =========================================================================
# SchedulerRunner tests
# =========================================================================


class TestSchedulerRunner:

    @pytest.mark.asyncio
    async def test_ensure_tables_creates_tables(self):
        db = _make_mock_db()
        executor = AsyncMock(return_value='{"ok": true}')
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._ensure_tables()

        # Expected DDL: scheduled_tasks table + index, task_execution_log
        # table + additive ALTER for outcome_signal + index = 5 statements.
        sqls = [call[0][0] for call in db.execute.call_args_list]
        assert any("CREATE TABLE IF NOT EXISTS scheduled_tasks" in s for s in sqls)
        assert any("CREATE TABLE IF NOT EXISTS task_execution_log" in s for s in sqls)
        assert any("ALTER TABLE task_execution_log" in s for s in sqls)
        assert any("outcome_signal" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_ensure_tables_is_idempotent_when_column_exists(self):
        """PostgreSQL migrations use native duplicate-safe DDL.

        Catching a duplicate-column error is not safe here: PostgreSQL marks
        the caller's transaction failed before Python could catch it.
        """
        db = _make_mock_db()
        db.backend_type = "postgres"
        base_fetchone = db.fetchone.side_effect

        async def _fetchone(sql, *args):
            if "FROM pg_constraint con" in sql:
                return ("scheduler_runtime_status_pkey",)
            return await base_fetchone(sql, *args)

        async def _fetchall(sql, *args):
            if "JOIN pg_attribute attribute" in sql:
                return [("agent_id",), ("owner_id",)]
            return []

        async def _exec(sql, *args):
            if "ALTER TABLE" in sql:
                assert "ADD COLUMN IF NOT EXISTS" in sql

        db.fetchone = AsyncMock(side_effect=_fetchone)
        db.fetchall = AsyncMock(side_effect=_fetchall)
        db.execute = AsyncMock(side_effect=_exec)
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._ensure_tables()
        assert not any(
            "LOCK TABLE scheduler_runtime_status" in call.args[0]
            for call in db.execute.call_args_list
        )

    @pytest.mark.asyncio
    async def test_arm_rejects_unprepared_protocol_before_database_clock(self):
        db = _make_mock_db()
        db.fetchval = AsyncMock(
            side_effect=AssertionError("database clock must not be queried")
        )
        runner = SchedulerRunner(db, "test-agent", AsyncMock(return_value="ok"))

        with pytest.raises(RuntimeError, match="protocol preparation"):
            await runner.arm()

        db.fetchval.assert_not_awaited()
        assert runner._arm_requested is False

    def test_database_connectivity_probe_tolerates_protected_backend_wrapper(self):
        class ProtectedDatabase:
            @property
            def backend(self):
                raise RuntimeError("backend access denied")

        runner = SchedulerRunner(
            ProtectedDatabase(), "test-agent", AsyncMock(return_value="ok")
        )
        assert runner._database_is_connected() is True

    @pytest.mark.asyncio
    async def test_ensure_tables_skips_existing_sqlite_columns_before_alter(self):
        """SQLite checks pragma metadata instead of relying on ALTER errors."""
        db = _make_mock_db()
        db.backend_type = "sqlite"

        async def _fetchone(sql, *args):
            if "pragma_table_info" in sql:
                return (1,)
            return None

        db.fetchone = AsyncMock(side_effect=_fetchone)
        runner = SchedulerRunner(db, "test-agent", AsyncMock(return_value="ok"))

        await runner._ensure_tables()

        assert not any(
            "ALTER TABLE" in call.args[0] for call in db.execute.call_args_list
        )

    @pytest.mark.asyncio
    async def test_ensure_tables_reraises_unexpected_migration_error(self):
        """The migration must NOT swallow non-duplicate errors — a locked DB,
        permission failure, or schema corruption must surface."""
        db = _make_mock_db()
        db.backend_type = "sqlite"

        async def _exec(sql, *args):
            if "ALTER TABLE" in sql:
                raise Exception("database is locked")

        db.execute = AsyncMock(side_effect=_exec)
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        with pytest.raises(Exception, match="database is locked"):
            await runner._ensure_tables()

    @pytest.mark.asyncio
    async def test_tick_no_due_tasks(self):
        db = _make_mock_db()
        db.fetchall = AsyncMock(return_value=[])
        executor = AsyncMock()
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()
        executor.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_executes_due_task(self):
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "wellness_check", "@daily", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        executor = AsyncMock(return_value='{"status": "ok"}')
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        executor.assert_called_once_with("wellness_check", {})

    @pytest.mark.asyncio
    async def test_tick_runs_due_tasks_concurrently_bounded(self):
        """#1675: due tasks run concurrently, capped at max_concurrent_tasks."""
        import asyncio

        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            (f"task-{i}", "test-agent", "t", "@hourly", "{}",
             1, None, now_iso, "2026-03-04T00:00:00")
            for i in range(6)
        ])

        live = 0
        peak = 0
        ran = []

        async def executor(name, args):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.02)  # force overlap
            ran.append(name)
            live -= 1
            return "ok"

        runner = SchedulerRunner(db, "test-agent", executor, max_concurrent_tasks=2)
        await runner._tick()

        assert len(ran) == 6          # every due task executed
        assert peak == 2              # never exceeded the cap
        assert peak > 1               # genuinely concurrent (not serial)

    @pytest.mark.asyncio
    async def test_tick_serial_when_cap_is_one(self):
        """max_concurrent_tasks=1 restores strictly-serial execution."""
        import asyncio

        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            (f"task-{i}", "test-agent", "t", "@hourly", "{}",
             1, None, now_iso, "2026-03-04T00:00:00")
            for i in range(4)
        ])

        live = 0
        peak = 0

        async def executor(name, args):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return "ok"

        runner = SchedulerRunner(db, "test-agent", executor, max_concurrent_tasks=1)
        await runner._tick()

        assert peak == 1              # strictly serial

    @pytest.mark.asyncio
    async def test_tick_one_failure_does_not_block_siblings(self):
        """A task raising must not cancel its concurrent siblings in the tick."""
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            (f"task-{i}", "test-agent", "t", "@hourly", "{}",
             1, None, now_iso, "2026-03-04T00:00:00")
            for i in range(3)
        ])

        ran = []

        async def executor(name, args):
            if name == "t" and len(ran) == 0:
                ran.append("boom")
                raise RuntimeError("kaboom")
            ran.append(name)
            return "ok"

        runner = SchedulerRunner(db, "test-agent", executor, max_concurrent_tasks=3)
        await runner._tick()

        # All three were attempted despite the first raising.
        assert len(ran) == 3

    @pytest.mark.asyncio
    async def test_tick_records_execution(self):
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "wellness_check", "@daily", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        # A durable scheduler first claims the row, records a ``claimed``
        # execution identity, and only then commits the terminal outcome.
        execute_calls = db.execute.call_args_list
        claim_call = next(c for c in execute_calls if "INSERT INTO task_execution_log" in c[0][0])
        claim_params = claim_call[0][1]
        assert claim_params[1] == "task-1"  # task_id
        assert "'claimed'" in claim_call[0][0]
        outcome_call = next(
            c for c in execute_calls
            if "UPDATE task_execution_log" in c[0][0]
        )
        assert outcome_call[0][1][0] == "success"

    @pytest.mark.asyncio
    async def test_tick_records_failure(self):
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "bad_task", "@hourly", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        executor = AsyncMock(side_effect=ValueError("task not found"))
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        # The durable log starts claimed then transitions to failed via CAS.
        outcome_call = next(
            c for c in db.execute.call_args_list
            if "UPDATE task_execution_log" in c[0][0]
        )
        assert outcome_call[0][1][0] == "failed"

    def test_failed_outcome_cannot_silently_request_pause(self):
        """Failed dispatcher results have no structured pause channel."""
        with pytest.raises(ValueError, match="only valid for blocked"):
            ScheduledTaskOutcome(
                status="failed",
                result_text="failed",
                pause_schedule=True,
            )

    @pytest.mark.asyncio
    async def test_tick_records_blocked_reason_and_pauses_schedule(self):
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("restart-task", "test-agent", "restart_coordinator", "* * * * *", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        executor = AsyncMock(
            return_value=ScheduledTaskOutcome.blocked(
                task_name="restart_coordinator",
                decision="ask",
                reason="headless scheduler cannot approve this operation",
            )
        )
        runner = SchedulerRunner(db, "test-agent", executor)

        await runner._tick()

        outcome_call = next(
            c for c in db.execute.call_args_list
            if "UPDATE task_execution_log" in c[0][0]
        )
        outcome_params = outcome_call[0][1]
        assert outcome_params[0] == "blocked"
        assert "headless scheduler" in outcome_params[1]
        assert "Schedule id: restart-task" in outcome_params[1]
        assert "resume the schedule" in outcome_params[1]

        pause_call = next(
            c for c in db.execute.call_args_list
            if "UPDATE scheduled_tasks" in c[0][0] and "terminal_status" in c[0][0]
        )
        assert "enabled = ?" in pause_call[0][0]
        # Protocol/lease CAS parameters intentionally evolve; assert the
        # schedule identity semantically rather than a brittle SQL offset.
        assert "restart-task" in pause_call[0][1]
        # A successful CAS rereads the authoritative claim metadata, then the
        # final effect-entry guard verifies the exact live token.
        reads = [call.args[0] for call in db.fetchone.call_args_list]
        assert any("SELECT claim_execution_id" in sql for sql in reads)
        assert any(
            "SELECT 1 FROM scheduled_tasks" in sql
            and "claim_token = ?" in sql
            for sql in reads
        )

    @pytest.mark.asyncio
    async def test_blocked_schedule_is_durable_and_not_retried(self, tmp_path):
        """#2430: pause and unblock guidance survive the current process."""
        raw_db = SQLiteBackend(str(tmp_path / "scheduler.db"))
        await raw_db.connect()
        db = AsyncDatabase(raw_db)
        executor = AsyncMock(
            return_value=ScheduledTaskOutcome.blocked(
                task_name="restart_coordinator",
                decision="ask",
                reason="operator approval required",
            )
        )
        runner = SchedulerRunner(db, "test-agent", executor)

        try:
            await runner._ensure_tables()
            due_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json,
                     enabled, next_run_at, created_at,
                     scheduler_protocol_version)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, 2)
                """,
                (
                    "restart-task",
                    "test-agent",
                    "restart_coordinator",
                    "* * * * *",
                    "{}",
                    due_at,
                    due_at,
                ),
            )

            await runner._tick()

            schedule = await db.fetchone(
                "SELECT enabled, last_run_at FROM scheduled_tasks WHERE id = ?",
                ("restart-task",),
            )
            assert schedule is not None
            assert schedule[0] == 0
            assert schedule[1] is not None

            history = await db.fetchone(
                """
                SELECT status, result_text
                FROM task_execution_log
                WHERE task_id = ?
                """,
                ("restart-task",),
            )
            assert history is not None
            assert history[0] == "blocked"
            assert "operator approval required" in history[1]
            assert "permission to Auto" in history[1]
            assert "resume the schedule" in history[1]

            # A later poll cannot redispatch the disabled schedule.
            await runner._tick()
            executor.assert_awaited_once()
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_tick_updates_next_run(self):
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "test_task", "*/5 * * * *", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        executor = AsyncMock(return_value="done")
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        # Locate the claim-CAS completion update (the first writes are claim
        # metadata and the durable ``claimed`` execution log).
        update_call = next(
            c for c in db.execute.call_args_list
            if "UPDATE scheduled_tasks" in c[0][0] and "terminal_status" in c[0][0]
        )
        update_params = update_call[0][1]
        # next_run_at should be set (not None)
        assert update_params[1] is not None  # next_run_at

    @pytest.mark.asyncio
    async def test_persisted_schedule_skips_disabled_feature_end_to_end(
        self, tmp_path,
    ):
        """E2E (#2522): a schedule persisted BEFORE its feature is disabled must
        not invoke the tool on a tick while the feature is disabled, and must
        resume executing it once the feature is re-enabled.

        Drives the whole scheduler path — real SQLite ``scheduled_tasks`` row →
        ``SchedulerRunner._tick`` → ``_dispatch_scheduled_task`` →
        ``_lookup_and_run_tool`` → the feature tool — with only the enablement
        state changing between ticks.
        """
        raw_db = SQLiteBackend(str(tmp_path / "scheduler.db"))
        await raw_db.connect()
        db = AsyncDatabase(raw_db)

        agent = _make_mock_agent(db)
        agent.did = "did:test:scheduler-agent"
        # Force the direct tool-execution path (no dispatcher) so the tick runs
        # exactly the production `_lookup_and_run_tool` body under test.
        agent.dispatcher = None
        job_feature = _StubJobFeature()
        agent.features = {"JobFeature": job_feature}
        agent.hooks_manager = TestTaskExecutor._passthrough_hooks_manager()

        sched = SchedulerFeature(agent)
        sched._db = db
        sched._agent_id = agent.did

        runner = SchedulerRunner(db, agent.did, sched._dispatch_scheduled_task)

        async def _make_due():
            """Reset the persisted task so the next tick considers it due."""
            due_at = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
            await db.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
                (due_at, "job-task"),
            )

        async def _latest_log_status():
            row = await db.fetchone(
                "SELECT status, result_text FROM task_execution_log "
                "WHERE task_id = ? ORDER BY rowid DESC LIMIT 1",
                ("job-task",),
            )
            return row

        try:
            await runner._ensure_tables()
            due_at = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
            await db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json,
                     enabled, next_run_at, created_at,
                     scheduler_protocol_version)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, 2)
                """,
                (
                    "job-task", agent.did, "job", "* * * * *", "{}",
                    due_at, due_at,
                ),
            )

            # --- Phase 1: feature enabled → the tool executes on the tick.
            await runner._tick()
            assert job_feature.calls == 1
            status, _text = await _latest_log_status()
            assert status == "success"

            # --- Phase 2: soft-disable, tick again → the tool is NOT invoked.
            job_feature.enabled = False
            await _make_due()
            await runner._tick()
            assert job_feature.calls == 1  # never ran while disabled
            status, text = await _latest_log_status()
            # Benign skip recorded as success, NOT a failure — no per-tick spam.
            assert status == "success"
            assert "disabled" in text
            # The schedule itself stays enabled (a disable must not pause it).
            row = await db.fetchone(
                "SELECT enabled FROM scheduled_tasks WHERE id = ?", ("job-task",),
            )
            assert row[0] == 1

            # --- Phase 3: re-enable, tick again → execution is restored.
            job_feature.enabled = True
            await _make_due()
            await runner._tick()
            assert job_feature.calls == 2
            status, _text = await _latest_log_status()
            assert status == "success"
        finally:
            await db.close()


# =========================================================================
# ScheduledTask dataclass
# =========================================================================


class TestScheduledTaskDataclass:

    def test_args_parses_valid_json(self):
        task = ScheduledTask(
            id="t1", agent_id="a1", task_name="test", cron_expression="@daily",
            args_json='{"key": "value"}', enabled=True,
            last_run_at=None, next_run_at=None, created_at="2026-01-01",
        )
        assert task.args == {"key": "value"}

    def test_args_returns_empty_for_invalid_json(self):
        task = ScheduledTask(
            id="t1", agent_id="a1", task_name="test", cron_expression="@daily",
            args_json="not json", enabled=True,
            last_run_at=None, next_run_at=None, created_at="2026-01-01",
        )
        assert task.args == {}

    def test_args_returns_empty_for_none(self):
        task = ScheduledTask(
            id="t1", agent_id="a1", task_name="test", cron_expression="@daily",
            args_json="", enabled=True,
            last_run_at=None, next_run_at=None, created_at="2026-01-01",
        )
        assert task.args == {}


# =========================================================================
# Feature initialization
# =========================================================================


class TestSchedulerInit:

    @pytest.mark.asyncio
    async def test_initialize_without_storage(self):
        agent = MagicMock(spec=["agent_id", "did", "features"])
        agent.agent_id = "did:test:no-storage"
        agent.features = {}
        f = SchedulerFeature(agent)
        # Should not raise
        await f.initialize()
        assert f._db is None
        assert f._runner is None

    @pytest.mark.asyncio
    async def test_shutdown_stops_runner(self, feature):
        feature._runner = MagicMock()
        feature._runner.stop = AsyncMock()
        await feature.shutdown()
        feature._runner.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_prepares_but_agent_ready_arms_polling(self):
        agent = _make_mock_agent()
        agent.did = agent.agent_id
        scheduler = SchedulerFeature(agent)
        with (
            patch.object(
                SchedulerRunner,
                "start",
                new_callable=AsyncMock,
            ) as prepare,
            patch.object(
                SchedulerRunner,
                "arm",
                new_callable=AsyncMock,
            ) as arm,
        ):
            await scheduler.initialize()
            prepare.assert_awaited_once_with(polling=False)
            arm.assert_not_awaited()

            await scheduler.on_agent_ready(agent)
            arm.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_host_managed_postgres_agent_never_creates_scoped_runner(self):
        agent = _make_mock_agent()
        agent.did = agent.agent_id
        agent._scheduler_polling_managed_by_host = True
        scheduler = SchedulerFeature(agent)
        with patch.object(
            SchedulerRunner,
            "start",
            new_callable=AsyncMock,
        ) as start:
            await scheduler.initialize()
            await scheduler.on_agent_ready(agent)

        assert scheduler._polling_managed_by_host is True
        assert scheduler._runner is None
        start.assert_not_awaited()


# =========================================================================
# Task executor callback
# =========================================================================


class TestTaskExecutor:

    @staticmethod
    def _passthrough_hooks_manager():
        """A hooks_manager whose PRE_TOOL_USE hook allows the tool through
        (no DENY/ASK): execute_hooks returns an output with no blocking
        decision, execute_hooks_parallel is a no-op. Mirrors the real
        HooksManager surface the tick-path executor (F245) now routes
        through."""
        from types import SimpleNamespace

        hm = MagicMock()
        hm.execute_hooks = AsyncMock(
            return_value=SimpleNamespace(
                permission_decision=None,
                updated_input=None,
                continue_execution=True,
            )
        )
        hm.execute_hooks_parallel = AsyncMock(return_value=None)
        return hm

    @pytest.mark.asyncio
    async def test_execute_known_task(self, feature):
        # Mock a feature with a matching tool. Phase 4 of #889 renamed
        # `_execute_scheduled_task` → `_lookup_and_run_tool` (the
        # tool-search body) when the dispatcher took over the executor
        # role. Scheduled dispatch now routes through the PRE/POST_TOOL_USE
        # hook gate (F245), so a passthrough hooks_manager stands in for the
        # real one; the lookup + execute behavior is otherwise unchanged.
        mock_tool = MagicMock()
        mock_tool.name = "wellness_check"
        mock_tool.execute = AsyncMock(return_value={"success": True, "score": 0.85})

        mock_feature = MagicMock()
        mock_feature.get_tools = MagicMock(return_value=[mock_tool])

        feature.agent.features = {"WellnessFeature": mock_feature}
        feature.agent.hooks_manager = self._passthrough_hooks_manager()

        result = await feature._lookup_and_run_tool("wellness_check", {})
        parsed = json.loads(result)
        assert parsed["success"] is True
        mock_tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_signal_dispatch_schedule_routes_through_cron_source(
        self, feature,
    ):
        """The supported source contract is independent of auto-seeding."""

        feature.agent.dispatcher = MagicMock()
        feature.agent.dispatcher.dispatch_signal = AsyncMock(
            return_value=SignalResult(
                signal_id="sig-dispatch",
                status=Status.OK,
                mode=SignalMode.ACTION,
                duration_ms=1,
                action_result="started",
            )
        )

        result = await feature._dispatch_scheduled_task(
            "signal_dispatch", {"mode": "execute"}
        )

        assert result == "started"
        signal = feature.agent.dispatcher.dispatch_signal.await_args.args[0]
        assert signal.source == "cron.signal_dispatch"
        assert signal.mode is SignalMode.ACTION
        assert signal.payload == {"mode": "execute"}

    @pytest.mark.asyncio
    async def test_training_cycle_requires_current_durable_semantic_maintenance(
        self, feature,
    ):
        """The generic scheduler path cannot bypass the sleep-hook boundary."""
        mock_tool = MagicMock()
        mock_tool.name = "training_cycle"
        mock_tool.execute = AsyncMock(return_value={"success": True})
        mock_feature = MagicMock()
        mock_feature.get_tools = MagicMock(return_value=[mock_tool])

        readiness = MagicMock(
            ready=False,
            reason="semantic_maintenance_partial",
            using_prior_verified_snapshot=False,
        )
        storage = MagicMock()
        storage.semantic_maintenance_training_readiness = AsyncMock(
            return_value=readiness
        )
        feature.agent.features = {"ReflectionFeature": mock_feature}
        feature.agent.storage = storage
        feature.agent.semantic_inference_profile = None
        feature.agent.semantic_inference_limits = None
        feature.agent.semantic_maintenance_limits = None
        feature.agent.semantic_capabilities = None
        feature.agent.semantic_inference_configured = False
        feature.agent.semantic_maintenance_configured = True
        feature.agent.semantic_maintenance_allow_prior_verified_snapshot = False

        # Drive the scheduler callback rather than calling the lookup helper
        # directly. The no-dispatcher branch preserves the direct callback
        # path used by partially initialized scheduler hosts.
        feature.agent.dispatcher = None
        result = await feature._dispatch_scheduled_task("training_cycle", {})

        assert isinstance(result, ScheduledTaskOutcome)
        assert result.status == "blocked"
        assert result.pause_schedule is False
        assert "semantic_maintenance_partial" in result.result_text
        mock_tool.execute.assert_not_awaited()
        storage.semantic_maintenance_training_readiness.assert_awaited_once_with(
            None,
            inference_limits=None,
            maintenance_limits=None,
            semantic_capabilities=None,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("decision", ["deny", "ask"])
    async def test_scheduled_permission_hook_returns_blocked_outcome(
        self, feature, decision,
    ):
        """F245: a PRE_TOOL_USE DENY on a scheduler tick must block the
        tool. #2430: DENY/ASK is an expected headless state, so return a
        structured outcome instead of raising through the dispatcher."""
        from types import SimpleNamespace

        from kestrel_sdk.hooks.base import PermissionDecision

        mock_tool = MagicMock()
        mock_tool.name = "dangerous_op"
        mock_tool.execute = AsyncMock(return_value={"success": True})

        mock_feature = MagicMock()
        mock_feature.get_tools = MagicMock(return_value=[mock_tool])

        feature.agent.features = {"DangerFeature": mock_feature}
        hm = MagicMock()
        hm.execute_hooks = AsyncMock(
            return_value=SimpleNamespace(
                permission_decision=PermissionDecision(decision),
                permission_reason="policy forbids it",
                updated_input=None,
                continue_execution=False,
            )
        )
        hm.execute_hooks_parallel = AsyncMock(return_value=None)
        feature.agent.hooks_manager = hm

        result = await feature._lookup_and_run_tool("dangerous_op", {})

        assert isinstance(result, ScheduledTaskOutcome)
        assert result.status == "blocked"
        assert result.pause_schedule is True
        assert decision in result.result_text
        assert "policy forbids it" in result.result_text
        assert "permission to Auto" in result.result_text
        # The tool never ran — the gate blocked it before execute().
        mock_tool.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_unknown_task_raises(self, feature):
        feature.agent.features = {}
        with pytest.raises(ValueError, match="Unknown task"):
            await feature._lookup_and_run_tool("nonexistent_task", {})

    @pytest.mark.asyncio
    async def test_failed_tool_result_raises_before_scheduler_serialization(
        self, feature,
    ):
        """The production lookup cannot JSON-encode failure into success."""
        failed_tool = MagicMock()
        failed_tool.name = "memory_consolidate"
        failed_tool.execute = AsyncMock(
            return_value={
                **ToolResult.failed(
                    "consolidation deadline expired"
                ).to_dict(),
                "tool": "memory_consolidate",
            }
        )
        memory_feature = MagicMock()
        memory_feature.get_tools = MagicMock(return_value=[failed_tool])
        feature.agent.features = {"MemoryFeature": memory_feature}
        feature.agent.hooks_manager = self._passthrough_hooks_manager()

        with pytest.raises(
            RuntimeError, match="scheduled tool memory_consolidate failed"
        ):
            await feature._lookup_and_run_tool("memory_consolidate", {})

    @pytest.mark.asyncio
    async def test_legacy_tool_exception_envelope_also_raises(self, feature):
        failed_tool = MagicMock()
        failed_tool.name = "job"
        failed_tool.execute = AsyncMock(
            return_value={
                "success": False,
                "error": "job exploded",
                "tool": "job",
            }
        )
        job_feature = MagicMock()
        job_feature.get_tools = MagicMock(return_value=[failed_tool])
        feature.agent.features = {"JobFeature": job_feature}
        feature.agent.hooks_manager = self._passthrough_hooks_manager()

        with pytest.raises(RuntimeError, match="scheduled tool job failed"):
            await feature._lookup_and_run_tool("job", {})

    @pytest.mark.asyncio
    async def test_builtin_cron_task_skipped_when_feature_not_loaded(
        self, feature,
    ):
        """A persisted built-in cron task (e.g. restart_coordinator) can
        fire on the first scheduler tick after a restart before its owning
        feature has registered the tool — a transient startup-order race
        (#1796). It must skip benignly, NOT raise 'Unknown task' (which
        would record a spurious one-time failure)."""
        feature.agent.features = {}
        result = await feature._lookup_and_run_tool("restart_coordinator", {})
        assert result.startswith("skipped:")
        assert "restart_coordinator" in result

    @pytest.mark.asyncio
    async def test_disabled_feature_tool_is_not_executed(self, feature):
        """#2522: a soft-disabled feature stays in ``agent.features`` with
        ``enabled=False`` but its tools are detached from every other surface.
        A persisted schedule that names one of its tools must NOT execute on a
        tick — the scheduler skips it benignly (no failure spam) — and
        re-enabling the feature restores execution."""
        job_feature = _StubJobFeature()
        feature.agent.features = {"JobFeature": job_feature}
        feature.agent.hooks_manager = self._passthrough_hooks_manager()

        # Enabled → the tool runs.
        result = await feature._lookup_and_run_tool("job", {})
        assert job_feature.calls == 1
        assert json.loads(result)["success"] is True

        # Soft-disabled → the SAME tool is skipped, not executed, not raised.
        job_feature.enabled = False
        result = await feature._lookup_and_run_tool("job", {})
        assert job_feature.calls == 1  # unchanged — never ran
        assert result.startswith("skipped:")
        assert "JobFeature" in result and "disabled" in result

        # Re-enabled → execution is restored on the next tick.
        job_feature.enabled = True
        result = await feature._lookup_and_run_tool("job", {})
        assert job_feature.calls == 2
        assert json.loads(result)["success"] is True

    @pytest.mark.asyncio
    async def test_enabled_owner_wins_over_disabled_same_name(self, feature):
        """If both a disabled and an enabled feature expose the same tool name,
        the enabled owner executes (the disabled scan runs only as a fallback)."""
        disabled = _StubJobFeature()
        disabled.name = "OldJobFeature"
        disabled.enabled = False
        enabled = _StubJobFeature()
        enabled.name = "NewJobFeature"
        feature.agent.features = {
            "OldJobFeature": disabled,
            "NewJobFeature": enabled,
        }
        feature.agent.hooks_manager = self._passthrough_hooks_manager()

        result = await feature._lookup_and_run_tool("job", {})
        assert enabled.calls == 1
        assert disabled.calls == 0
        assert json.loads(result)["success"] is True


class TestTranslateSignalResult:
    """Regression for #904 review P1: misconfiguration drops
    (DROPPED_VALIDATION, DROPPED_CYCLE) must surface as failures so the
    runner records status='failed', matching the legacy behavior where
    bad args_json would have raised at `**args` / `.get`. Benign drops
    (rate_limit, quiet_hours, coalesced) keep flowing as success rows
    with descriptive text in result_text."""

    @staticmethod
    def _make_result(status, *, mode=None, action_result=None, artifact=None, error=None):
        from kestrel_sdk.signals import SignalMode, SignalResult

        return SignalResult(
            signal_id="sig-test",
            status=status,
            mode=mode or SignalMode.ACTION,
            duration_ms=1,
            action_result=action_result,
            artifact=artifact,
            error=error,
        )

    def test_validation_drop_raises(self, feature):
        """Bad args (e.g. args_json decoded to a non-dict) → schema
        validation in the dispatcher → DROPPED_VALIDATION → must
        bubble up so runner records 'failed'."""
        from kestrel_sdk.signals import Status

        with pytest.raises(RuntimeError, match="dropped_validation"):
            feature._translate_signal_result(
                self._make_result(Status.DROPPED_VALIDATION, error="payload not dict"),
                "morning_signal",
            )

    def test_cycle_drop_raises(self, feature):
        """Causation cycle = misconfiguration; surface as failure."""
        from kestrel_sdk.signals import Status

        with pytest.raises(RuntimeError, match="dropped_cycle"):
            feature._translate_signal_result(
                self._make_result(Status.DROPPED_CYCLE, error="agent in chain"),
                "reflect",
            )

    def test_failed_status_raises(self, feature):
        """Tool exception inside the handler → Status.FAILED → raise."""
        from kestrel_sdk.signals import Status

        with pytest.raises(RuntimeError, match="failed"):
            feature._translate_signal_result(
                self._make_result(Status.FAILED, error="tool blew up"),
                "training_cycle",
            )

    def test_rate_limit_drop_returns_skipped_string(self, feature):
        """Rate limit is a benign throttle; record as success with
        'skipped: ...' text."""
        from kestrel_sdk.signals import Status

        result = feature._translate_signal_result(
            self._make_result(Status.DROPPED_RATE_LIMIT, error="too many"),
            "backup_snapshot",
        )
        assert isinstance(result, str)
        assert result.startswith("skipped:")
        assert "rate_limit" in result

    def test_quiet_hours_drop_returns_skipped_string(self, feature):
        from kestrel_sdk.signals import Status

        result = feature._translate_signal_result(
            self._make_result(Status.DROPPED_QUIET_HOURS, error="quiet"),
            "morning_signal",
        )
        assert result.startswith("skipped:")
        assert "quiet_hours" in result

    def test_coalesced_returns_skipped_string(self, feature):
        from kestrel_sdk.signals import Status

        result = feature._translate_signal_result(
            self._make_result(Status.COALESCED, error=None),
            "morning_signal",
        )
        assert result.startswith("skipped:")

    def test_ok_action_returns_action_result(self, feature):
        from kestrel_sdk.signals import SignalMode, Status

        result = feature._translate_signal_result(
            self._make_result(
                Status.OK, mode=SignalMode.ACTION, action_result="ran-it"
            ),
            "backup_snapshot",
        )
        assert result == "ran-it"

    def test_ok_action_preserves_structured_blocked_outcome(self, feature):
        from kestrel_sdk.signals import SignalMode, Status

        blocked = ScheduledTaskOutcome.blocked(
            task_name="restart_coordinator",
            decision="ask",
            reason="approval is unavailable in a headless tick",
        )
        result = feature._translate_signal_result(
            self._make_result(
                Status.OK,
                mode=SignalMode.ACTION,
                action_result=blocked,
            ),
            "restart_coordinator",
        )

        assert result is blocked

    def test_ok_artifact_dict_is_json_encoded(self, feature):
        """ARTIFACT tools commonly return Dicts. Preserve the legacy
        result_text contract (JSON-encoded for the runner)."""
        from kestrel_sdk.signals import SignalMode, Status

        result = feature._translate_signal_result(
            self._make_result(
                Status.OK, mode=SignalMode.ARTIFACT, artifact={"score": 0.9}
            ),
            "reflect",
        )
        assert isinstance(result, str)
        assert json.loads(result) == {"score": 0.9}

    def test_ok_action_tuple_preserved_for_outcome_signal(self, feature):
        """Tools may return (text, outcome_signal) — runner extracts
        the second element. Translation must preserve the tuple shape."""
        from kestrel_sdk.signals import SignalMode, Status

        result = feature._translate_signal_result(
            self._make_result(
                Status.OK,
                mode=SignalMode.ACTION,
                action_result=("dispatched", 0.7),
            ),
            "backup_snapshot",
        )
        assert result == ("dispatched", 0.7)


# =========================================================================
# schedule_update
# =========================================================================


class TestScheduleUpdate:

    @pytest.mark.asyncio
    async def test_update_existing(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("@daily", 1))
        result = await feature.schedule_update(
            task_id="task-id", cron_expression="@hourly"
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "updated"
        assert result.data["old_cron"] == "@daily"
        assert result.data["cron_expression"] == "@hourly"
        assert result.data["next_run_at"] is not None

    @pytest.mark.asyncio
    async def test_update_unchanged_is_noop(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("@hourly", 1))
        result = await feature.schedule_update(
            task_id="task-id", cron_expression="@hourly"
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["status"] == "unchanged"
        # The durable rollout control row is locked before the in-transaction
        # read, but an unchanged definition must not write scheduled_tasks.
        assert all(
            "UPDATE scheduled_tasks" not in call.args[0]
            for call in feature._db.execute.call_args_list
        )

    @pytest.mark.asyncio
    async def test_update_reanchors_cron_occurrence_to_database_clock(self, feature):
        """Changing cadence cannot create a host-clock-skewed next run."""

        database_now = datetime(2026, 7, 25, 8, 0, 30, tzinfo=timezone.utc)
        events = _use_postgres_clock(
            feature,
            database_now,
            scheduled_row=("@daily", 1, "cron", "UTC", 0, 0),
        )

        with patch(
            "kestrel_sovereign.features.scheduler.feature.datetime"
        ) as host_datetime:
            host_datetime.now.return_value = datetime(
                2040, 1, 1, 0, 0, tzinfo=timezone.utc
            )
            result = await feature.schedule_update(
                task_id="task-id",
                cron_expression="* * * * *",
            )

        assert result.status is ToolResultStatus.OK
        assert result.data["next_run_at"] == "2026-07-25T08:01:00+00:00"
        assert events[:6] == [
            "transaction_begin",
            "schema_lock",
            "rollout_lock",
            "schedule_lock",
            "schedule_read",
            "database_clock",
        ]
        assert events[-1] == "transaction_end"

    @pytest.mark.asyncio
    async def test_update_terminalizes_claim_with_database_clock_after_schedule_write(
        self, feature
    ):
        """Superseding a claim must not publish a skewed API-host audit time."""

        database_now = datetime(2026, 7, 25, 8, 0, 30, tzinfo=timezone.utc)
        events = _use_postgres_clock(
            feature,
            database_now,
            scheduled_row=("@daily", 1, "cron", "UTC", 0, 0),
        )

        with patch(
            "kestrel_sovereign.features.scheduler.feature.datetime"
        ) as host_datetime:
            host_datetime.now.return_value = datetime(
                2040, 1, 1, 0, 0, tzinfo=timezone.utc
            )
            result = await feature.schedule_update(
                task_id="task-id",
                cron_expression="* * * * *",
            )

        assert result.status is ToolResultStatus.OK
        terminal_call = next(
            call
            for call in feature._db.execute.call_args_list
            if "UPDATE task_execution_log" in call.args[0]
        )
        assert "clock_timestamp()" in terminal_call.args[0]
        assert terminal_call.args[1] == (
            "superseded",
            "schedule definition updated before outcome commit",
            "task-id",
            feature._agent_id,
        )
        assert events == [
            "transaction_begin",
            "schema_lock",
            "rollout_lock",
            "schedule_lock",
            "schedule_read",
            "database_clock",
            "schedule_update",
            "execution_terminal",
            "transaction_end",
        ]

    @pytest.mark.asyncio
    async def test_update_invalid_cron(self, feature):
        result = await feature.schedule_update(
            task_id="task-id", cron_expression="not a cron"
        )
        assert result.status is ToolResultStatus.ERROR
        assert "invalid cron" in result.error.lower()

    @pytest.mark.asyncio
    async def test_update_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_update(
            task_id="missing", cron_expression="@daily"
        )
        assert result.status is ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_update_disabled_task_leaves_next_run_null(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("@daily", 0))
        result = await feature.schedule_update(
            task_id="task-id", cron_expression="@hourly"
        )
        assert result.status is ToolResultStatus.OK
        # Paused task must not compute a next_run
        assert result.data["next_run_at"] is None

    @pytest.mark.asyncio
    async def test_update_no_db(self, feature_no_db):
        result = await feature_no_db.schedule_update(
            task_id="any", cron_expression="@daily"
        )
        assert result.status is ToolResultStatus.ERROR


# =========================================================================
# schedule_record_outcome
# =========================================================================


class TestRecordOutcome:

    @pytest.mark.asyncio
    async def test_record_signal(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("exec-id",))
        result = await feature.schedule_record_outcome(
            execution_id="exec-id", signal=0.8
        )
        assert result.status is ToolResultStatus.OK
        assert result.data["signal"] == 0.8

    @pytest.mark.asyncio
    async def test_signal_clamped_upper(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("exec-id",))
        result = await feature.schedule_record_outcome(
            execution_id="exec-id", signal=2.5
        )
        # Honesty: out-of-range input was silently clamped → PARTIAL.
        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["signal"] == 1.0
        assert result.data["signal_clamped"] is True
        assert "clamped" in result.error

    @pytest.mark.asyncio
    async def test_signal_clamped_lower(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("exec-id",))
        result = await feature.schedule_record_outcome(
            execution_id="exec-id", signal=-0.5
        )
        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["signal"] == 0.0
        assert result.data["signal_clamped"] is True

    @pytest.mark.asyncio
    async def test_record_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_record_outcome(
            execution_id="missing", signal=0.5
        )
        assert result.status is ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_non_numeric_signal_returns_error_not_exception(self, feature):
        """A bad signal must produce a structured error, not an uncaught exception."""
        result = await feature.schedule_record_outcome(
            execution_id="exec-1", signal="high"
        )
        assert result.status is ToolResultStatus.ERROR
        assert "numeric" in result.error.lower()

    @pytest.mark.asyncio
    async def test_none_signal_returns_error(self, feature):
        result = await feature.schedule_record_outcome(
            execution_id="exec-1", signal=None
        )
        assert result.status is ToolResultStatus.ERROR


# =========================================================================
# schedule_engagement
# =========================================================================


class TestScheduleEngagement:

    @pytest.mark.asyncio
    async def test_empty(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        result = await feature.schedule_engagement(days=7)
        assert result.status is ToolResultStatus.OK
        assert result.data["window_days"] == 7
        assert result.data["tasks"] == []
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_aggregates(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("task-1", "morning_signal", "0 8 * * *", 7, 5, 0.62),
            ("task-2", "reflect", "0 */4 * * *", 42, 0, None),
        ])
        result = await feature.schedule_engagement(days=7)
        assert result.status is ToolResultStatus.OK
        assert result.data["count"] == 2
        assert result.data["tasks"][0]["mean_signal"] == 0.62
        assert result.data["tasks"][0]["signals"] == 5
        # Second task has executions but zero signals — downstream isn't
        # reporting back. mean_signal must be None, not 0.
        assert result.data["tasks"][1]["mean_signal"] is None
        assert result.data["tasks"][1]["signals"] == 0

    @pytest.mark.asyncio
    async def test_rejects_zero_or_negative_days(self, feature):
        result = await feature.schedule_engagement(days=0)
        assert result.status is ToolResultStatus.ERROR
        result = await feature.schedule_engagement(days=-1)
        assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_rejects_absurdly_large_days(self, feature):
        result = await feature.schedule_engagement(days=10_000)
        assert result.status is ToolResultStatus.ERROR
        assert "365" in result.error


# =========================================================================
# Runner outcome signal + mid-flight cron reload
# =========================================================================


class TestRunnerOutcomeSignal:

    @pytest.mark.asyncio
    async def test_tuple_return_captures_signal(self):
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "morning_signal", "0 8 * * *", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        db.fetchone = AsyncMock(return_value=("0 8 * * *", 1))
        executor = AsyncMock(return_value=("dispatched", 0.75))
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        outcome_call = next(
            c for c in db.execute.call_args_list
            if "UPDATE task_execution_log" in c[0][0]
        )
        # outcome_signal is committed with the terminal CAS outcome.
        assert outcome_call[0][1][4] == 0.75

    @pytest.mark.asyncio
    async def test_plain_string_return_has_null_signal(self):
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "backup_snapshot", "0 */4 * * *", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        db.fetchone = AsyncMock(return_value=("0 */4 * * *", 1))
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        outcome_call = next(c for c in db.execute.call_args_list if "UPDATE task_execution_log" in c[0][0])
        assert outcome_call[0][1][4] is None

    @pytest.mark.asyncio
    async def test_non_numeric_tuple_signal_is_dropped(self):
        """Runner must reject garbage signal values rather than writing them."""
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "morning_signal", "0 8 * * *", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        db.fetchone = AsyncMock(return_value=("0 8 * * *", 1))
        executor = AsyncMock(return_value=("dispatched", "high"))
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        outcome_call = next(c for c in db.execute.call_args_list if "UPDATE task_execution_log" in c[0][0])
        assert outcome_call[0][1][4] is None

    @pytest.mark.asyncio
    async def test_tuple_signal_is_clamped(self):
        """Signals above 1.0 are clamped, not written as-is."""
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "morning_signal", "0 8 * * *", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        db.fetchone = AsyncMock(return_value=("0 8 * * *", 1))
        executor = AsyncMock(return_value=("dispatched", 2.5))
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        outcome_call = next(c for c in db.execute.call_args_list if "UPDATE task_execution_log" in c[0][0])
        assert outcome_call[0][1][4] == 1.0


class TestRunnerCronReload:

    @pytest.mark.asyncio
    async def test_next_run_at_reflects_fresh_cron(self):
        """Behavior test — not just that re-fetch happened, but that the
        persisted next_run_at was computed from the updated cron, not the
        stale in-memory value. Schedules daily → updated to every-5-min
        mid-flight; the persisted next_run_at must be within the next 5
        minutes, not tomorrow.
        """
        db = _make_mock_db()
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "test_task", "@daily", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        # Simulate schedule_update changing cron to every-5-min mid-execution.
        db.fetchone = AsyncMock(return_value=("*/5 * * * *", 1))
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        # Find the UPDATE scheduled_tasks call that sets next_run_at
        update_call = next(
            c for c in db.execute.call_args_list
            if "UPDATE scheduled_tasks" in c[0][0] and "next_run_at" in c[0][0]
        )
        next_run_iso = update_call[0][1][1]
        assert next_run_iso is not None
        next_run_dt = datetime.fromisoformat(next_run_iso)
        # Must fire within the next ~6 minutes (proves */5 cron, not @daily)
        delta_seconds = (next_run_dt - now).total_seconds()
        assert 0 <= delta_seconds <= 6 * 60, (
            f"next_run_at should reflect */5 cron but is {delta_seconds}s away"
        )

    @pytest.mark.asyncio
    async def test_completion_is_guarded_by_claim_token(self):
        """A pause/update clears the claim token, so an old worker's final
        write has a CAS guard instead of being able to resurrect the schedule."""
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "test_task", "@daily", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        completion = next(
            c for c in db.execute.call_args_list
            if "UPDATE scheduled_tasks" in c[0][0] and "terminal_status" in c[0][0]
        )
        assert "claim_token = ?" in completion[0][0]
        assert "claim_execution_id = ?" in completion[0][0]

    @pytest.mark.asyncio
    async def test_deleted_before_effect_entry_does_not_execute(self):
        """A vanished claim loses the final token/live admission check."""
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "test_task", "@daily", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        # The claim was selected and recorded, but an administrative delete
        # wins before the exact effect-entry read.
        db.fetchone = AsyncMock(return_value=None)
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        executor.assert_not_awaited()
        assert not any(
            "UPDATE task_execution_log" in call.args[0]
            for call in db.execute.call_args_list
        )


# =========================================================================
# github_pr_watch handler (#1618)
# =========================================================================


def _discovery_clean():
    return {
        "summary": "clean",
        "findings": [],
    }


def _discovery_finding(**overrides):
    base = {
        "repo": "owner/name",
        "kind": "red_ci",
        "pr": 2281,
        "branch": "feature/wake-discovery",
        "check": "unit",
        "severity": "high",
        "status": "failure",
        "title": "Unit tests are red",
        "html_url": "https://github.com/owner/name/actions/runs/1",
    }
    base.update(overrides)
    return {"summary": "1 finding", "findings": [base]}


class TestEcosystemDiscoveryWatchHandler:

    @pytest.mark.asyncio
    async def test_missing_discovery_tool_fails_closed(self, feature):
        out = await feature._run_ecosystem_discovery_watch(
            {"repo": "owner/name"}
        )

        data = json.loads(out)
        assert data["signaled"] is False
        assert data["blocked"] == "configuration"
        assert "requires a non-empty tool" in data["error"]

    @pytest.mark.asyncio
    async def test_clean_scan_does_not_signal(self, feature):
        feature._lookup_and_run_tool = AsyncMock(return_value=_discovery_clean())
        _wire_watch_dispatcher(feature)

        out = await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        )

        data = json.loads(out)
        assert data["signaled"] is False
        assert data["reason"] == "clean"
        assert data["findings_count"] == 0
        feature.agent.dispatcher.enqueue_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_discovery_tool_resolves_external_feature(self, feature):
        external = _StubJobFeature()
        external._tool.name = "discover_ecosystem"
        external._tool.execute = AsyncMock(return_value=_discovery_finding())
        feature.agent.features = {"ExternalDiscoveryFeature": external}
        feature.agent.hooks_manager = None
        feature._load_ecosystem_discovery_state = AsyncMock(return_value=(None, None))
        feature._save_ecosystem_discovery_state = AsyncMock()
        _wire_watch_dispatcher(feature)

        out = await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        )

        data = json.loads(out)
        assert data["signaled"] is True
        assert data["reason"] == "new_findings"
        assert data["findings_count"] == 1
        signal = feature.agent.dispatcher.enqueue_signal.call_args.args[0]
        findings = json.loads(signal.payload["findings"])
        assert findings[0]["repo"] == "owner/name"
        assert findings[0]["kind"] == "red_ci"
        assert findings[0]["number"] == "2281"

    @pytest.mark.asyncio
    async def test_roster_args_forwarded_to_scan_tool(self, feature):
        """Scheduler forwards org/allowlist/prefix roster args to the tool (#2269)."""
        feature._lookup_and_run_tool = AsyncMock(return_value=_discovery_clean())
        feature._load_ecosystem_discovery_state = AsyncMock(return_value=(None, None))
        feature._save_ecosystem_discovery_state = AsyncMock()
        _wire_watch_dispatcher(feature)

        out = await feature._run_ecosystem_discovery_watch({
            "tool": "discover_ecosystem",
            "org": "KestrelSovereignAI",
            "repos": ["KestrelSovereignAI/kestrel-feature-*"],
            "repo_prefix": "KestrelSovereignAI/kestrel-",
            "watch_key": "ecosystem-roster",
        })

        data = json.loads(out)
        assert data["watch_key"] == "ecosystem-roster"
        tool_name, tool_args = feature._lookup_and_run_tool.call_args.args
        assert tool_name == "discover_ecosystem"
        assert tool_args["org"] == "KestrelSovereignAI"
        assert tool_args["repos"] == ["KestrelSovereignAI/kestrel-feature-*"]
        assert tool_args["repo_prefix"] == "KestrelSovereignAI/kestrel-"
        # Watcher control keys are never leaked into the tool kwargs.
        assert "watch_key" not in tool_args
        assert "tool" not in tool_args

    @pytest.mark.asyncio
    async def test_new_finding_emits_signal_with_payload(self, feature):
        feature._lookup_and_run_tool = AsyncMock(return_value=_discovery_finding())
        feature._load_ecosystem_discovery_state = AsyncMock(return_value=(None, None))
        feature._save_ecosystem_discovery_state = AsyncMock()
        _wire_watch_dispatcher(feature)

        out = await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        )

        data = json.loads(out)
        assert data["signaled"] is True
        assert data["reason"] == "new_findings"
        assert data["findings_count"] == 1
        feature.agent.dispatcher.enqueue_signal.assert_called_once()
        signal = feature.agent.dispatcher.enqueue_signal.call_args.args[0]
        assert signal.source == "ecosystem.discovery_findings"
        findings = json.loads(signal.payload["findings"])
        assert findings[0]["repo"] == "owner/name"
        assert findings[0]["number"] == "2281"
        assert findings[0]["severity"] == "high"
        assert findings[0]["suggested_gate"] == "verify_ci_then_dispatch_fix"
        # Acceptance is not delivery: the fingerprint must still be un-advanced
        # here, and the handler must say so rather than claim a checkpoint.
        feature._save_ecosystem_discovery_state.assert_not_awaited()
        assert data["checkpoint"] == "pending_delivery"

        await _settle_watch_deliveries(feature)

        feature._save_ecosystem_discovery_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_notify_cannot_forge_discovery_wake_target(self, feature):
        feature.agent.did = "did:test:watch-owner"
        feature._agent_id = feature.agent.did
        feature._lookup_and_run_tool = AsyncMock(return_value=_discovery_finding())
        feature._load_ecosystem_discovery_state = AsyncMock(return_value=(None, None))
        feature._save_ecosystem_discovery_state = AsyncMock()
        _wire_watch_dispatcher(feature)

        out = await feature._run_ecosystem_discovery_watch(
            {
                "tool": "discover_ecosystem",
                "repo": "owner/name",
                "notify": "did:test:forged-peer",
            }
        )

        assert json.loads(out)["signaled"] is True
        signal = feature.agent.dispatcher.enqueue_signal.call_args.args[0]
        assert signal.target_agent == "did:test:watch-owner"
        await _settle_watch_deliveries(feature)

    @pytest.mark.asyncio
    async def test_unchanged_finding_does_not_signal(self, feature):
        from kestrel_sovereign.signals.sources.ecosystem_discovery import (
            normalize_discovery_result,
            state_to_json,
        )

        state = normalize_discovery_result(_discovery_finding())
        feature._lookup_and_run_tool = AsyncMock(return_value=_discovery_finding())
        feature._load_ecosystem_discovery_state = AsyncMock(
            return_value=(state.fingerprint, json.loads(state_to_json(state)))
        )
        feature._save_ecosystem_discovery_state = AsyncMock()
        _wire_watch_dispatcher(feature)

        out = await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        )

        data = json.loads(out)
        assert data["signaled"] is False
        assert data["reason"] == "no_change"
        feature.agent.dispatcher.enqueue_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_changed_finding_emits_signal(self, feature):
        from kestrel_sovereign.signals.sources.ecosystem_discovery import (
            normalize_discovery_result,
            state_to_json,
        )

        previous = normalize_discovery_result(_discovery_finding(status="failure"))
        feature._lookup_and_run_tool = AsyncMock(
            return_value=_discovery_finding(status="timed_out", job="e2e")
        )
        feature._load_ecosystem_discovery_state = AsyncMock(
            return_value=(previous.fingerprint, json.loads(state_to_json(previous)))
        )
        feature._save_ecosystem_discovery_state = AsyncMock()
        _wire_watch_dispatcher(feature)

        out = await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        )

        data = json.loads(out)
        assert data["signaled"] is True
        assert data["reason"] == "changed_findings"
        signal = feature.agent.dispatcher.enqueue_signal.call_args.args[0]
        findings = json.loads(signal.payload["findings"])
        assert findings[0]["status"] == "timed_out"
        assert findings[0]["job"] == "e2e"

    @pytest.mark.asyncio
    async def test_resolved_finding_emits_one_resolution_signal(self, feature):
        from kestrel_sovereign.signals.sources.ecosystem_discovery import (
            normalize_discovery_result,
            state_to_json,
        )

        previous = normalize_discovery_result(_discovery_finding())
        feature._lookup_and_run_tool = AsyncMock(return_value=_discovery_clean())
        feature._load_ecosystem_discovery_state = AsyncMock(
            return_value=(previous.fingerprint, json.loads(state_to_json(previous)))
        )
        feature._save_ecosystem_discovery_state = AsyncMock()
        _wire_watch_dispatcher(feature)

        out = await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        )

        data = json.loads(out)
        assert data["signaled"] is True
        assert data["reason"] == "resolved_findings"
        assert data["findings_count"] == 0
        signal = feature.agent.dispatcher.enqueue_signal.call_args.args[0]
        previous_findings = json.loads(signal.payload["previous_findings"])
        assert previous_findings[0]["repo"] == "owner/name"

        await _settle_watch_deliveries(feature)

        feature._save_ecosystem_discovery_state.assert_awaited_once()


class TestEcosystemDiscoveryDeliveryGate:
    """#2532: the fingerprint checkpoint may only advance on terminal ``OK``.

    A watcher that advances on enqueue *acceptance* does not retry — the next
    poll sees no change — so a failed or dropped wake loses the finding
    silently and permanently. Every non-``OK`` terminal state below must leave
    the prior fingerprint in place so the next poll re-detects and re-dispatches.
    """

    async def _run_with(self, feature, handle_factory):
        feature._lookup_and_run_tool = AsyncMock(return_value=_discovery_finding())
        feature._load_ecosystem_discovery_state = AsyncMock(
            return_value=(None, None)
        )
        feature._save_ecosystem_discovery_state = AsyncMock()
        _wire_watch_dispatcher(feature, handle_factory)

        out = await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        )
        await _settle_watch_deliveries(feature)
        return json.loads(out)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [
        Status.FAILED,
        Status.DROPPED_RATE_LIMIT,
        Status.DROPPED_QUIET_HOURS,
        Status.DROPPED_CYCLE,
        Status.DROPPED_VALIDATION,
        # COALESCED is deliberately not checkpoint-grade: the dedupe key is
        # recorded before the turn runs, so a wake that died inside the
        # resuming turn still coalesces a fast retry.
        Status.COALESCED,
    ])
    async def test_non_ok_terminal_state_does_not_advance_fingerprint(
        self, feature, status,
    ):
        await self._run_with(feature, lambda: _watch_handle(status))
        feature._save_ecosystem_discovery_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ok_terminal_state_advances_fingerprint(self, feature):
        await self._run_with(feature, lambda: _watch_handle(Status.OK))
        feature._save_ecosystem_discovery_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancelled_dispatch_does_not_advance_fingerprint(self, feature):
        """The dispatch task died (shutdown reaped it). Nothing was delivered,
        and this watcher is healthy — retain the fingerprint, do not crash."""
        await self._run_with(feature, _cancelled_watch_handle)
        feature._save_ecosystem_discovery_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unobservable_handle_does_not_advance_fingerprint(self, feature):
        """A dispatcher that hands back something with no awaitable ``wait``
        makes delivery unobservable. Unobservable is treated exactly like
        undelivered — never like success."""
        await self._run_with(feature, lambda: object())
        feature._save_ecosystem_discovery_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wait_error_does_not_advance_fingerprint(self, feature):
        """A tooling error inside ``wait()`` is not evidence of delivery."""

        class _ExplodingHandle:
            async def wait(self):
                raise RuntimeError("dispatcher internals blew up")

        await self._run_with(feature, _ExplodingHandle)
        feature._save_ecosystem_discovery_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delivery_in_flight_suppresses_duplicate_wake(self, feature):
        """While a wake is settling the baseline still holds the PRIOR
        fingerprint, so every tick re-detects the same change. Without an
        in-flight guard that dispatches a duplicate wake per tick for the whole
        length of the cognition turn — the #2738 storm shape."""
        feature._lookup_and_run_tool = AsyncMock(return_value=_discovery_finding())
        feature._load_ecosystem_discovery_state = AsyncMock(
            return_value=(None, None)
        )
        feature._save_ecosystem_discovery_state = AsyncMock()

        gate = asyncio.Event()

        def _slow_handle():
            async def _terminal():
                await gate.wait()
                return SignalResult(
                    signal_id="sig-watch",
                    status=Status.OK,
                    mode=SignalMode.COGNITION,
                    duration_ms=1,
                )
            return SignalHandle(
                signal_id="sig-watch", task=asyncio.ensure_future(_terminal()),
            )

        enqueue = _wire_watch_dispatcher(feature, _slow_handle)

        first = json.loads(await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        ))
        second = json.loads(await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        ))

        assert first["signaled"] is True
        assert second["signaled"] is False
        assert second["blocked"] == "delivery_in_flight"
        assert enqueue.await_count == 1

        gate.set()
        await _settle_watch_deliveries(feature)

        feature._save_ecosystem_discovery_state.assert_awaited_once()
        # Settled — the guard released, so a later change can dispatch again.
        third = json.loads(await feature._run_ecosystem_discovery_watch(
            {"tool": "discover_ecosystem", "repo": "owner/name"}
        ))
        assert third["signaled"] is True
        assert enqueue.await_count == 2

        gate.set()
        await _settle_watch_deliveries(feature)


_GH_TOKEN = (
    "kestrel_sovereign.features.strategic_memory."
    "github_integration.get_github_token"
)
_GH_FETCH = "kestrel_sovereign.signals.sources.github_pr_watch.fetch_pr_state"


def _pr_payload(**overrides):
    base = {
        "state": "open",
        "merged": False,
        "comments": 2,
        "review_comments": 1,
        "updated_at": "2026-06-09T16:00:00Z",
        "head": {"sha": "abc123"},
        "checks_status": "success",
        "mergeable_state": "clean",
        "html_url": "https://github.com/owner/name/pull/1614",
    }
    base.update(overrides)
    return base


class TestGitHubPRWatchHandler:

    @pytest.mark.asyncio
    async def test_missing_args_reports_error(self, feature):
        out = await feature._run_github_pr_watch({"repo": "owner/name"})
        data = json.loads(out)
        assert data["signaled"] is False
        assert "requires" in data["error"]

    @pytest.mark.asyncio
    async def test_no_token_blocked_auth(self, feature):
        with patch(_GH_TOKEN, return_value=None):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )
        data = json.loads(out)
        assert data["signaled"] is False
        assert data["blocked"] == "auth"

    @pytest.mark.asyncio
    async def test_network_failure_blocked_network(self, feature):
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            PRWatchNetworkError,
        )

        feature._db.fetchone = AsyncMock(return_value=None)
        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH,
            new=AsyncMock(side_effect=PRWatchNetworkError("timeout")),
        ):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )
        data = json.loads(out)
        assert data["signaled"] is False
        assert data["blocked"] == "network"

    @pytest.mark.asyncio
    async def test_first_observation_does_not_signal(self, feature):
        # No persisted fingerprint → first observation → no wake.
        feature._db.fetchone = AsyncMock(return_value=None)
        _wire_watch_dispatcher(feature)
        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH, new=AsyncMock(return_value=_pr_payload())
        ):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )
        data = json.loads(out)
        assert data["signaled"] is False
        assert data["reason"] == "first_observation"
        feature.agent.dispatcher.enqueue_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_change_does_not_signal(self, feature):
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            compute_fingerprint,
            normalize_pr_state,
        )

        norm = normalize_pr_state(_pr_payload())
        fp = compute_fingerprint(norm)
        feature._db.fetchone = AsyncMock(return_value=(fp, json.dumps(norm)))
        _wire_watch_dispatcher(feature)
        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH, new=AsyncMock(return_value=_pr_payload())
        ):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )
        data = json.loads(out)
        assert data["signaled"] is False
        assert data["reason"] == "no_change"
        feature.agent.dispatcher.enqueue_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_poll_holds_the_baseline_while_a_wake_is_in_flight(
        self, feature
    ):
        """"This poll dispatched nothing" is not "no delivery is pending".

        The sequence: a comment is added (signal-worthy, wake dispatched,
        baseline correctly held pending Status.OK), then removed —
        ``updated_at`` moves, so the next poll is a no-op change. If that
        no-op advances the baseline and the in-flight wake then fails or is
        dropped, the state its retry needed is already gone and the matching
        event is lost permanently.

        That is exactly the loss #2532 exists to prevent, reached through a
        second poll instead of through the dispatch itself.
        """
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            compute_fingerprint,
            normalize_pr_state,
        )

        norm = normalize_pr_state(_pr_payload(comments=2))
        fp = compute_fingerprint(norm)
        feature._db.fetchone = AsyncMock(return_value=(fp, json.dumps(norm)))
        _wire_watch_dispatcher(feature)
        # A wake dispatched by an earlier poll has not settled yet.
        feature._inflight_watch_deliveries.add("github_pr:owner/name#1614")

        saved: list = []
        feature._save_pr_watch_state = AsyncMock(
            side_effect=lambda *a, **kw: saved.append(a)
        )

        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH,
            new=AsyncMock(
                return_value=_pr_payload(
                    comments=2, updated_at="2026-06-09T17:00:00Z"
                )
            ),
        ):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )

        data = json.loads(out)
        assert data["signaled"] is False
        assert saved == [], (
            "the baseline must be held while a prior wake is still in "
            f"flight; got {saved}"
        )
        assert data["checkpoint"] == "held_delivery_in_flight"

    @pytest.mark.asyncio
    async def test_new_comment_emits_signal(self, feature):
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            compute_fingerprint,
            normalize_pr_state,
        )

        prev = normalize_pr_state(_pr_payload(comments=2))
        prev_fp = compute_fingerprint(prev)
        feature._db.fetchone = AsyncMock(
            return_value=(prev_fp, json.dumps(prev))
        )
        feature._save_pr_watch_state = AsyncMock()
        _wire_watch_dispatcher(feature)
        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH, new=AsyncMock(return_value=_pr_payload(comments=3))
        ):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )
        data = json.loads(out)
        assert data["signaled"] is True
        assert "comments" in data["changed"]
        feature.agent.dispatcher.enqueue_signal.assert_called_once()
        # Accepted, not yet delivered — the baseline still holds the old
        # fingerprint until the wake lands (#2532).
        assert data["checkpoint"] == "pending_delivery"
        feature._save_pr_watch_state.assert_not_awaited()

        await _settle_watch_deliveries(feature)

        feature._save_pr_watch_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_notify_cannot_forge_pr_causation_or_evade_cycle_detection(
        self, feature
    ):
        from kestrel_sovereign.signals.dispatcher import SignalDispatcher
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            SOURCE_NAME,
            build_github_pr_activity_registration,
            compute_fingerprint,
            normalize_pr_state,
        )

        feature.agent.did = "did:test:watch-owner"
        feature._agent_id = feature.agent.did
        previous = normalize_pr_state(_pr_payload(comments=2))
        feature._load_pr_watch_state = AsyncMock(
            return_value=(compute_fingerprint(previous), previous)
        )
        feature._save_pr_watch_state = AsyncMock()
        _wire_watch_dispatcher(feature)
        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH,
            new=AsyncMock(return_value=_pr_payload(comments=3)),
        ):
            out = await feature._run_github_pr_watch(
                {
                    "repo": "owner/name",
                    "pr": 1614,
                    "notify": "did:test:forged-peer",
                }
            )

        assert json.loads(out)["signaled"] is True
        signal = feature.agent.dispatcher.enqueue_signal.call_args.args[0]
        assert signal.target_agent == "did:test:watch-owner"

        signal.causation_chain.append(
            CausationFrame(
                agent_id="did:test:watch-owner",
                source=SOURCE_NAME,
                signal_id="prior-watch-wake",
                turn_id="turn-prior",
                depth=1,
                emitted_at=datetime.now(timezone.utc),
            )
        )
        dispatcher = SimpleNamespace(
            _ttl=5,
            _clock=lambda: datetime.now(timezone.utc),
        )
        frame, cycle = SignalDispatcher._compute_frame_and_check_cycle(
            dispatcher,
            signal,
            build_github_pr_activity_registration(),
        )
        assert frame.agent_id == "did:test:watch-owner"
        assert cycle is not None and "Cycle detected" in cycle
        await _settle_watch_deliveries(feature)

    @pytest.mark.asyncio
    async def test_dispatch_error_does_not_advance_watch_state(self, feature):
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            compute_fingerprint,
            normalize_pr_state,
        )

        prev = normalize_pr_state(_pr_payload(comments=2))
        prev_fp = compute_fingerprint(prev)
        feature._load_pr_watch_state = AsyncMock(
            return_value=(prev_fp, prev)
        )
        feature._save_pr_watch_state = AsyncMock()
        feature.agent.dispatcher = MagicMock()
        feature.agent.dispatcher.enqueue_signal = AsyncMock(
            side_effect=RuntimeError("queue unavailable")
        )

        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH, new=AsyncMock(return_value=_pr_payload(comments=3))
        ):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )

        data = json.loads(out)
        assert data["signaled"] is False
        assert data["blocked"] == "dispatch_error"
        feature._save_pr_watch_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_dispatcher_does_not_advance_watch_state(self, feature):
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            compute_fingerprint,
            normalize_pr_state,
        )

        prev = normalize_pr_state(_pr_payload(comments=2))
        prev_fp = compute_fingerprint(prev)
        feature._load_pr_watch_state = AsyncMock(
            return_value=(prev_fp, prev)
        )
        feature._save_pr_watch_state = AsyncMock()
        feature.agent.dispatcher = None

        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH, new=AsyncMock(return_value=_pr_payload(comments=3))
        ):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )

        data = json.loads(out)
        assert data["signaled"] is False
        assert data["blocked"] == "no_dispatcher"
        feature._save_pr_watch_state.assert_not_awaited()

    async def _watch_with_handle(self, feature, handle_factory):
        """Drive one signal-worthy poll whose wake settles per the factory."""
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            compute_fingerprint,
            normalize_pr_state,
        )

        prev = normalize_pr_state(_pr_payload(comments=2))
        feature._load_pr_watch_state = AsyncMock(
            return_value=(compute_fingerprint(prev), prev)
        )
        feature._save_pr_watch_state = AsyncMock()
        _wire_watch_dispatcher(feature, handle_factory)

        with patch(_GH_TOKEN, return_value="tok"), patch(
            _GH_FETCH, new=AsyncMock(return_value=_pr_payload(comments=3))
        ):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "pr": 1614}
            )
        await _settle_watch_deliveries(feature)
        return json.loads(out)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [
        Status.FAILED,
        Status.DROPPED_RATE_LIMIT,
        Status.DROPPED_CYCLE,
        Status.DROPPED_VALIDATION,
        Status.COALESCED,
    ])
    async def test_non_ok_delivery_does_not_advance_watch_state(
        self, feature, status,
    ):
        """#2532: the comment/merge/check delta must stay re-detectable. A
        checkpoint advanced on a wake the dispatcher then failed or dropped
        marks the event handled forever — the watcher never retries."""
        await self._watch_with_handle(feature, lambda: _watch_handle(status))
        feature._save_pr_watch_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ok_delivery_advances_watch_state(self, feature):
        await self._watch_with_handle(feature, lambda: _watch_handle(Status.OK))
        feature._save_pr_watch_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancelled_dispatch_does_not_advance_watch_state(self, feature):
        await self._watch_with_handle(feature, _cancelled_watch_handle)
        feature._save_pr_watch_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unobservable_handle_does_not_advance_watch_state(self, feature):
        await self._watch_with_handle(feature, lambda: object())
        feature._save_pr_watch_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pr_arg_fetches_pulls_endpoint(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        fetch = AsyncMock(return_value=_pr_payload())
        with patch(_GH_TOKEN, return_value="tok"), patch(_GH_FETCH, new=fetch):
            await feature._run_github_pr_watch({"repo": "owner/name", "pr": 1614})
        assert fetch.call_args.kwargs["kind"] == "pr"

    @pytest.mark.asyncio
    async def test_issue_arg_fetches_issues_endpoint(self, feature):
        # An explicit issue number must hit /issues, not /pulls (which would
        # 404 for an issue and falsely report blocked: network every tick).
        feature._db.fetchone = AsyncMock(return_value=None)
        fetch = AsyncMock(return_value={"state": "open", "comments": 0})
        with patch(_GH_TOKEN, return_value="tok"), patch(_GH_FETCH, new=fetch):
            out = await feature._run_github_pr_watch(
                {"repo": "owner/name", "issue": 1618}
            )
        assert fetch.call_args.kwargs["kind"] == "issue"
        data = json.loads(out)
        assert data["signaled"] is False
        assert data["reason"] == "first_observation"


def test_fetch_url_selects_endpoint_by_kind():
    """fetch_pr_state must build /pulls vs /issues from the kind arg."""
    import asyncio
    from unittest.mock import patch as _patch

    from kestrel_sovereign.signals.sources import github_pr_watch as gpw

    captured = {}

    class _FakeResp:
        def read(self):
            return b'{"state": "open"}'

    def _fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        return _FakeResp()

    with _patch.object(gpw.urllib.request, "urlopen", _fake_urlopen):
        asyncio.run(gpw.fetch_pr_state("owner/name", 1614, token="t", kind="pr"))
        assert captured["url"].endswith("/repos/owner/name/pulls/1614")
        asyncio.run(gpw.fetch_pr_state("owner/name", 1618, token="t", kind="issue"))
        assert captured["url"].endswith("/repos/owner/name/issues/1618")


# =========================================================================
# Run tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
