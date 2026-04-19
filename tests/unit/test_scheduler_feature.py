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
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
        assert result["tasks"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_returns_tasks(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("id-1", "wellness_check", "@daily", "{}", 1, None, "2026-03-06T00:00:00", "2026-03-05T00:00:00"),
            ("id-2", "audit_anchor", "0 */6 * * *", '{"force": true}', 0, "2026-03-05T06:00:00", "2026-03-05T12:00:00", "2026-03-04T00:00:00"),
        ])
        result = await feature.schedule_list()
        assert result["count"] == 2
        assert result["tasks"][0]["task_name"] == "wellness_check"
        assert result["tasks"][1]["enabled"] is False
        assert result["tasks"][1]["args"] == {"force": True}

    @pytest.mark.asyncio
    async def test_list_no_db(self, feature_no_db):
        result = await feature_no_db.schedule_list()
        assert result["success"] is False
        assert "not available" in result["error"].lower()


# =========================================================================
# schedule_add
# =========================================================================


class TestScheduleAdd:

    @pytest.mark.asyncio
    async def test_add_valid_task(self, feature):
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="wellness_check",
        )
        assert result["success"] is True
        assert result["task_name"] == "wellness_check"
        assert result["cron_expression"] == "@daily"
        assert result["task_id"] is not None
        assert result["next_run_at"] is not None

        # Verify DB insert was called
        feature._db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_add_with_args(self, feature):
        result = await feature.schedule_add(
            cron_expression="*/15 * * * *",
            task_name="memory_consolidation",
            args_json='{"threshold": 100}',
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_add_invalid_cron(self, feature):
        result = await feature.schedule_add(
            cron_expression="bad cron",
            task_name="test_task",
        )
        assert result["success"] is False
        assert "invalid cron" in result["error"].lower() or "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_add_invalid_args_json(self, feature):
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="test_task",
            args_json="not valid json",
        )
        assert result["success"] is False
        assert "args_json" in result["error"].lower() or "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_add_args_json_must_be_object(self, feature):
        result = await feature.schedule_add(
            cron_expression="@daily",
            task_name="test_task",
            args_json='[1, 2, 3]',
        )
        assert result["success"] is False
        assert "object" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_add_no_db(self, feature_no_db):
        result = await feature_no_db.schedule_add(
            cron_expression="@daily",
            task_name="test_task",
        )
        assert result["success"] is False


# =========================================================================
# schedule_remove
# =========================================================================


class TestScheduleRemove:

    @pytest.mark.asyncio
    async def test_remove_existing(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id",))
        result = await feature.schedule_remove(task_id="task-id")
        assert result["success"] is True
        assert result["status"] == "removed"

    @pytest.mark.asyncio
    async def test_remove_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_remove(task_id="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_remove_no_db(self, feature_no_db):
        result = await feature_no_db.schedule_remove(task_id="any")
        assert result["success"] is False


# =========================================================================
# schedule_pause
# =========================================================================


class TestSchedulePause:

    @pytest.mark.asyncio
    async def test_pause_active_task(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id", 1))
        result = await feature.schedule_pause(task_id="task-id")
        assert result["success"] is True
        assert result["status"] == "paused"

    @pytest.mark.asyncio
    async def test_pause_already_paused(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id", 0))
        result = await feature.schedule_pause(task_id="task-id")
        assert result["success"] is True
        assert result["status"] == "already_paused"

    @pytest.mark.asyncio
    async def test_pause_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_pause(task_id="nonexistent")
        assert result["success"] is False


# =========================================================================
# schedule_resume
# =========================================================================


class TestScheduleResume:

    @pytest.mark.asyncio
    async def test_resume_paused_task(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id", 0, "@daily"))
        result = await feature.schedule_resume(task_id="task-id")
        assert result["success"] is True
        assert result["status"] == "resumed"
        assert result["next_run_at"] is not None

    @pytest.mark.asyncio
    async def test_resume_already_running(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("task-id", 1, "@daily"))
        result = await feature.schedule_resume(task_id="task-id")
        assert result["success"] is True
        assert result["status"] == "already_running"

    @pytest.mark.asyncio
    async def test_resume_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_resume(task_id="nonexistent")
        assert result["success"] is False


# =========================================================================
# schedule_history
# =========================================================================


class TestScheduleHistory:

    @pytest.mark.asyncio
    async def test_history_empty(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        result = await feature.schedule_history()
        assert result["executions"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_history_returns_records(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("exec-1", "task-1", "success", '{"result": "ok"}', 150, "2026-03-05T10:00:00", "wellness_check", 0.9),
            ("exec-2", "task-2", "failed", "Connection timeout", 5000, "2026-03-05T09:00:00", "audit_anchor", None),
        ])
        result = await feature.schedule_history()
        assert result["count"] == 2
        assert result["executions"][0]["status"] == "success"
        assert result["executions"][0]["outcome_signal"] == 0.9
        assert result["executions"][1]["task_name"] == "audit_anchor"
        assert result["executions"][1]["outcome_signal"] is None

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
        # Mock a feature with a matching tool
        mock_tool = MagicMock()
        mock_tool.name = "wellness_check"
        mock_tool.execute = AsyncMock(return_value={"success": True, "score": 0.85})

        mock_feature = MagicMock()
        mock_feature.get_tools = MagicMock(return_value=[mock_tool])

        feature.agent.features = {"WellnessFeature": mock_feature}

        result = await feature._execute_scheduled_task("wellness_check", {})
        parsed = json.loads(result)
        assert parsed["success"] is True

    @pytest.mark.asyncio
    async def test_execute_unknown_task_raises(self, feature):
        feature.agent.features = {}
        with pytest.raises(ValueError, match="Unknown task"):
            await feature._execute_scheduled_task("nonexistent_task", {})


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
        assert result["success"] is True
        assert result["status"] == "updated"
        assert result["old_cron"] == "@daily"
        assert result["cron_expression"] == "@hourly"
        assert result["next_run_at"] is not None

    @pytest.mark.asyncio
    async def test_update_unchanged_is_noop(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("@hourly", 1))
        result = await feature.schedule_update(
            task_id="task-id", cron_expression="@hourly"
        )
        assert result["success"] is True
        assert result["status"] == "unchanged"
        # Must not UPDATE when nothing changed
        feature._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_invalid_cron(self, feature):
        result = await feature.schedule_update(
            task_id="task-id", cron_expression="not a cron"
        )
        assert result["success"] is False
        assert "invalid cron" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_update_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_update(
            task_id="missing", cron_expression="@daily"
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_update_disabled_task_leaves_next_run_null(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("@daily", 0))
        result = await feature.schedule_update(
            task_id="task-id", cron_expression="@hourly"
        )
        assert result["success"] is True
        # Paused task must not compute a next_run
        assert result["next_run_at"] is None

    @pytest.mark.asyncio
    async def test_update_no_db(self, feature_no_db):
        result = await feature_no_db.schedule_update(
            task_id="any", cron_expression="@daily"
        )
        assert result["success"] is False


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
        assert result["success"] is True
        assert result["signal"] == 0.8

    @pytest.mark.asyncio
    async def test_signal_clamped_upper(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("exec-id",))
        result = await feature.schedule_record_outcome(
            execution_id="exec-id", signal=2.5
        )
        assert result["signal"] == 1.0

    @pytest.mark.asyncio
    async def test_signal_clamped_lower(self, feature):
        feature._db.fetchone = AsyncMock(return_value=("exec-id",))
        result = await feature.schedule_record_outcome(
            execution_id="exec-id", signal=-0.5
        )
        assert result["signal"] == 0.0

    @pytest.mark.asyncio
    async def test_record_not_found(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.schedule_record_outcome(
            execution_id="missing", signal=0.5
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_non_numeric_signal_returns_error_not_exception(self, feature):
        """A bad signal must produce a structured error, not an uncaught exception."""
        result = await feature.schedule_record_outcome(
            execution_id="exec-1", signal="high"
        )
        assert result["success"] is False
        assert "numeric" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_none_signal_returns_error(self, feature):
        result = await feature.schedule_record_outcome(
            execution_id="exec-1", signal=None
        )
        assert result["success"] is False


# =========================================================================
# schedule_engagement
# =========================================================================


class TestScheduleEngagement:

    @pytest.mark.asyncio
    async def test_empty(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        result = await feature.schedule_engagement(days=7)
        assert result["window_days"] == 7
        assert result["tasks"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_aggregates(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("task-1", "morning_signal", "0 8 * * *", 7, 5, 0.62),
            ("task-2", "reflect", "0 */4 * * *", 42, 0, None),
        ])
        result = await feature.schedule_engagement(days=7)
        assert result["count"] == 2
        assert result["tasks"][0]["mean_signal"] == 0.62
        assert result["tasks"][0]["signals"] == 5
        # Second task has executions but zero signals — downstream isn't
        # reporting back. mean_signal must be None, not 0.
        assert result["tasks"][1]["mean_signal"] is None
        assert result["tasks"][1]["signals"] == 0

    @pytest.mark.asyncio
    async def test_rejects_zero_or_negative_days(self, feature):
        result = await feature.schedule_engagement(days=0)
        assert result["success"] is False
        result = await feature.schedule_engagement(days=-1)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_rejects_absurdly_large_days(self, feature):
        result = await feature.schedule_engagement(days=10_000)
        assert result["success"] is False
        assert "365" in result["error"]


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
# Run tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
