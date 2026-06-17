"""
Unit tests for the SchedulerFeature and SchedulerRunner.

Tests:
- Feature initialization and tool registration
- Adding, listing, removing, pausing, and resuming scheduled tasks
- Execution history retrieval
- SchedulerRunner tick logic and execution recording
- Error handling for missing DB, invalid cron, etc.
"""

import json
import logging
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, ScheduledTask


# =========================================================================
# Helpers
# =========================================================================


def _make_mock_db():
    """Create a mock AsyncDatabase with standard methods."""
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    db.fetchval = AsyncMock(return_value=0)
    db.table_exists = AsyncMock(return_value=True)
    return db


def _make_mock_agent(db=None):
    """Create a mock agent with storage.db."""
    agent = MagicMock()
    agent.agent_id = "did:test:scheduler-agent"
    agent.features = {}

    mock_db = db or _make_mock_db()
    agent.storage = MagicMock()
    agent.storage.db = mock_db

    return agent


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


# =========================================================================
# post_all_features_loaded — retired-cron cutover cleanup (#1674)
# =========================================================================


class TestRetiredCronCleanup:
    @pytest.mark.asyncio
    async def test_post_load_removes_orphaned_cognition_retention(self):
        """An agent upgraded from #1715 has a persisted cognition_retention
        schedule. After #1674 removed its handler/source, post_all_features_loaded
        must delete that orphan row so it doesn't fire forever as 'Unknown task'."""
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
        f.schedule_add = AsyncMock(return_value=ToolResult.ok(
            confirmation="added", data={"next_run_at": None}))

        await f.post_all_features_loaded(agent)

        # The orphaned built-in was removed by id...
        f.schedule_remove.assert_awaited_once_with("orphan-1")
        # ...and never re-seeded (it's no longer a default).
        readded = [c.kwargs.get("task_name") for c in f.schedule_add.await_args_list]
        assert "cognition_retention" not in readded
        # An already-present live default is not duplicated.
        assert "backup_snapshot" not in readded

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
        f.schedule_add = AsyncMock(return_value=ToolResult.ok(
            confirmation="added", data={"next_run_at": None}))

        await f.post_all_features_loaded(agent)

        removed_ids = {c.args[0] for c in f.schedule_remove.await_args_list}
        assert "mc-auto" in removed_ids and "rf-auto" in removed_ids
        assert "mc-custom" not in removed_ids  # user schedule preserved
        seeded = [c.kwargs.get("task_name") for c in f.schedule_add.await_args_list]
        assert "sleep" in seeded                # the one memory-maintenance cron
        assert "memory_consolidate" not in seeded
        assert "reflect" not in seeded


class TestSleepCronHandler:
    @pytest.mark.asyncio
    async def test_handle_sleep_calls_agent_sleep_skip_export(self):
        """The sleep cron handler runs the agent's sleep cycle with
        skip_export=True (backups own DR) and surfaces the report."""
        from types import SimpleNamespace

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
        result = await feature.schedule_history(limit=5)
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
        """ALTER TABLE ADD COLUMN fails if the column already exists — that
        specific error must be swallowed so re-running _ensure_tables stays safe."""
        db = _make_mock_db()

        async def _exec(sql, *args):
            if "ALTER TABLE" in sql:
                raise Exception("duplicate column name: outcome_signal")

        db.execute = AsyncMock(side_effect=_exec)
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        # Must not raise even though ALTER fails with duplicate-column
        await runner._ensure_tables()

    @pytest.mark.asyncio
    async def test_ensure_tables_reraises_unexpected_migration_error(self):
        """The migration must NOT swallow non-duplicate errors — a locked DB,
        permission failure, or schema corruption must surface."""
        db = _make_mock_db()

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

        # Should have recorded execution (INSERT into task_execution_log)
        # and updated the task (UPDATE scheduled_tasks)
        execute_calls = db.execute.call_args_list
        assert len(execute_calls) >= 2  # INSERT + UPDATE

        # Check the INSERT call
        insert_call = execute_calls[0]
        assert "task_execution_log" in insert_call[0][0]
        insert_params = insert_call[0][1]
        assert insert_params[1] == "task-1"  # task_id
        assert insert_params[3] == "success"  # status

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

        # Should record "failed" status
        insert_call = db.execute.call_args_list[0]
        insert_params = insert_call[0][1]
        assert insert_params[3] == "failed"

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

        # Check the UPDATE call
        update_call = db.execute.call_args_list[1]
        assert "UPDATE scheduled_tasks" in update_call[0][0]
        update_params = update_call[0][1]
        # next_run_at should be set (not None)
        assert update_params[1] is not None  # next_run_at


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


# =========================================================================
# Task executor callback
# =========================================================================


class TestTaskExecutor:

    @pytest.mark.asyncio
    async def test_execute_known_task(self, feature):
        # Mock a feature with a matching tool. Phase 4 of #889 renamed
        # `_execute_scheduled_task` → `_lookup_and_run_tool` (the
        # tool-search body) when the dispatcher took over the executor
        # role. The lookup behavior tested here is unchanged.
        mock_tool = MagicMock()
        mock_tool.name = "wellness_check"
        mock_tool.execute = AsyncMock(return_value={"success": True, "score": 0.85})

        mock_feature = MagicMock()
        mock_feature.get_tools = MagicMock(return_value=[mock_tool])

        feature.agent.features = {"WellnessFeature": mock_feature}

        result = await feature._lookup_and_run_tool("wellness_check", {})
        parsed = json.loads(result)
        assert parsed["success"] is True

    @pytest.mark.asyncio
    async def test_execute_unknown_task_raises(self, feature):
        feature.agent.features = {}
        with pytest.raises(ValueError, match="Unknown task"):
            await feature._lookup_and_run_tool("nonexistent_task", {})

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
            "signal_dispatch",
        )
        assert result == "ran-it"

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
            "signal_dispatch",
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
        # Must not UPDATE when nothing changed
        feature._db.execute.assert_not_called()

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

        insert_call = db.execute.call_args_list[0]
        insert_params = insert_call[0][1]
        # outcome_signal is the 8th positional arg in the INSERT
        assert insert_params[7] == 0.75

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

        insert_params = db.execute.call_args_list[0][0][1]
        assert insert_params[7] is None

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

        insert_params = db.execute.call_args_list[0][0][1]
        assert insert_params[7] is None

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

        insert_params = db.execute.call_args_list[0][0][1]
        assert insert_params[7] == 1.0


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
    async def test_paused_mid_flight_does_not_reschedule(self):
        """If schedule_pause runs while the task is executing, the runner
        must NOT set a new next_run_at — that would silently undo the pause."""
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "test_task", "@daily", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        # Re-fetch shows enabled=0 — pause happened between SELECT and recompute.
        db.fetchone = AsyncMock(return_value=("@daily", 0))
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        await runner._tick()

        # The UPDATE must only touch last_run_at, not next_run_at.
        update_calls = [
            c for c in db.execute.call_args_list
            if "UPDATE scheduled_tasks" in c[0][0]
        ]
        assert len(update_calls) == 1
        assert "next_run_at" not in update_calls[0][0][0]
        # Only last_run_at + task id should be in params
        assert len(update_calls[0][0][1]) == 2

    @pytest.mark.asyncio
    async def test_deleted_mid_flight_does_not_crash(self):
        """If the task row was deleted mid-execution, only last_run_at gets
        written (for the audit trail); no next_run_at resurrection."""
        db = _make_mock_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        db.fetchall = AsyncMock(return_value=[
            ("task-1", "test-agent", "test_task", "@daily", "{}",
             1, None, now_iso, "2026-03-04T00:00:00"),
        ])
        db.fetchone = AsyncMock(return_value=None)
        executor = AsyncMock(return_value="ok")
        runner = SchedulerRunner(db, "test-agent", executor)
        # Must not raise
        await runner._tick()

        update_calls = [
            c for c in db.execute.call_args_list
            if "UPDATE scheduled_tasks" in c[0][0]
        ]
        # Either no UPDATE (row gone, DELETE will fail with no-op) or
        # last_run_at only. Current code takes the last_run_at-only path.
        if update_calls:
            assert "next_run_at" not in update_calls[0][0][0]


# =========================================================================
# github_pr_watch handler (#1618)
# =========================================================================


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
        feature.agent.dispatcher = MagicMock()
        feature.agent.dispatcher.enqueue_signal = AsyncMock()
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
        feature.agent.dispatcher = MagicMock()
        feature.agent.dispatcher.enqueue_signal = AsyncMock()
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
        feature.agent.dispatcher = MagicMock()
        feature.agent.dispatcher.enqueue_signal = AsyncMock(
            return_value=MagicMock()
        )
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
