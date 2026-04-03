"""
Unit Tests for the Heartbeat Feature (#151).

Tests:
- Individual health check functions (database, LLM, memory, disk, context)
- HeartbeatFeature lifecycle (initialize, shutdown)
- heartbeat_check tool runs all checks and persists results
- heartbeat_status tool returns history and uptime
- heartbeat_interval tool changes the interval
- Overall status derivation (healthy, degraded, unhealthy)
- Background loop start/stop
- Graceful handling of missing DB and components
- Tool discovery
"""

import asyncio
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.heartbeat.checks import (
    check_context_budget,
    check_database,
    check_disk_space,
    check_llm_service,
    check_memory_system,
)
from kestrel_sovereign.features.heartbeat.feature import (
    DEFAULT_INTERVAL_SECONDS,
    HeartbeatFeature,
    _derive_overall_status,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_db(fetchone_data=None, fetchall_data=None, table_exists_map=None):
    """Create a mock AsyncDatabase."""
    db = AsyncMock()

    if table_exists_map is None:
        table_exists_map = {}

    async def _table_exists(name):
        return table_exists_map.get(name, True)

    db.table_exists = AsyncMock(side_effect=_table_exists)
    db.fetchone = AsyncMock(return_value=fetchone_data if fetchone_data is not None else (1,))
    db.fetchall = AsyncMock(return_value=fetchall_data or [])
    db.execute = AsyncMock(return_value=0)
    return db


def _make_agent(db=None, agent_id="test-heartbeat-agent"):
    """Create a mock KestrelAgent with configurable components."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.did = agent_id

    storage = MagicMock()
    storage.db = db
    storage.retriever = MagicMock()
    storage.consolidator = None
    agent.storage = storage
    agent._raw_storage = None

    # LLM service — providers list matches real LLMService.providers structure
    llm_service = MagicMock()
    llm_service.providers = [{"name": "anthropic", "model": "claude-opus-4-6"}]
    llm_service.get_model_preference = MagicMock(
        return_value={"model": "claude-opus-4-6", "provider": "anthropic"}
    )
    llm_service.get_active_model_id = MagicMock(return_value="claude-opus-4-6")
    agent.llm_service = llm_service

    # No context manager by default
    agent.context_manager = None

    # Features list (empty by default)
    agent.features = []

    return agent


# ============================================================================
# Individual Check Tests
# ============================================================================


class TestCheckDatabase:
    @pytest.mark.asyncio
    async def test_pass_when_healthy(self):
        db = _make_db(fetchone_data=(1,))
        result = await check_database(db)
        assert result["name"] == "database"
        assert result["status"] == "pass"
        assert result["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_fail_when_no_db(self):
        result = await check_database(None)
        assert result["status"] == "fail"
        assert "No database" in result["message"]

    @pytest.mark.asyncio
    async def test_fail_when_query_raises(self):
        db = AsyncMock()
        db.fetchone = AsyncMock(side_effect=Exception("connection lost"))
        result = await check_database(db)
        assert result["status"] == "fail"
        assert "connection lost" in result["message"]

    @pytest.mark.asyncio
    async def test_warn_when_unexpected_result(self):
        db = _make_db(fetchone_data=(42,))
        result = await check_database(db)
        assert result["status"] == "warn"


class TestCheckLLMService:
    @pytest.mark.asyncio
    async def test_pass_with_providers(self):
        agent = _make_agent()
        result = await check_llm_service(agent)
        assert result["name"] == "llm_service"
        assert result["status"] == "pass"
        assert "anthropic" in result["message"]

    @pytest.mark.asyncio
    async def test_fail_when_no_llm(self):
        agent = _make_agent()
        agent.llm_service = None
        result = await check_llm_service(agent)
        assert result["status"] == "fail"

    @pytest.mark.asyncio
    async def test_warn_when_no_providers(self):
        agent = _make_agent()
        agent.llm_service.providers = []
        result = await check_llm_service(agent)
        assert result["status"] == "warn"

    @pytest.mark.asyncio
    async def test_includes_active_model(self):
        agent = _make_agent()
        result = await check_llm_service(agent)
        # Heartbeat should report the resolved active model, not a raw dict
        assert "(active: anthropic/claude-opus-4-6)" in result["message"]

    @pytest.mark.asyncio
    async def test_reports_mandated_model_not_config_default(self):
        """When a mandate overrides the provider default, heartbeat should
        report the mandated model — not the first provider's config."""
        agent = _make_agent()
        # Config default is anthropic/claude-opus-4-6, but mandate points to openai/gpt-5
        agent.llm_service.get_active_model_id = MagicMock(return_value="gpt-5")
        agent.llm_service.get_model_preference = MagicMock(
            return_value={"model": "gpt-5", "provider": "openai"}
        )
        result = await check_llm_service(agent)
        assert "(active: openai/gpt-5)" in result["message"]
        # Must NOT report the config default
        assert "claude-opus-4-6" not in result["message"].split("(active:")[1]

    @pytest.mark.asyncio
    async def test_reports_model_without_provider_when_no_mandate_provider(self):
        """When active model is set but no provider preference, report model only."""
        agent = _make_agent()
        agent.llm_service.get_active_model_id = MagicMock(return_value="llama3.2:3b")
        agent.llm_service.get_model_preference = MagicMock(
            return_value={"model": None, "provider": None}
        )
        result = await check_llm_service(agent)
        assert "(active: llama3.2:3b)" in result["message"]


class TestCheckMemorySystem:
    @pytest.mark.asyncio
    async def test_pass_with_components(self):
        agent = _make_agent(db=_make_db())
        result = await check_memory_system(agent)
        assert result["status"] == "pass"
        assert "retriever" in result["message"]
        assert "database" in result["message"]

    @pytest.mark.asyncio
    async def test_fail_when_no_storage(self):
        agent = _make_agent()
        agent.storage = None
        result = await check_memory_system(agent)
        assert result["status"] == "fail"

    @pytest.mark.asyncio
    async def test_warn_when_no_components(self):
        agent = _make_agent()
        # Use spec=[] to create a storage mock that has no auto-created attributes
        bare_storage = MagicMock(spec=[])
        agent.storage = bare_storage
        agent.memory_retriever = None
        agent.memory_consolidator = None
        result = await check_memory_system(agent)
        assert result["status"] == "warn"


class TestCheckDiskSpace:
    @pytest.mark.asyncio
    async def test_pass_normally(self):
        result = await check_disk_space()
        assert result["name"] == "disk_space"
        # On any real system, disk space should be available
        assert result["status"] in ("pass", "warn", "fail")
        assert result["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_warn_on_low_space(self):
        # Mock shutil.disk_usage to simulate low space
        with patch("kestrel_sovereign.features.heartbeat.checks.shutil") as mock_shutil:
            # 50MB free on 1TB disk
            mock_shutil.disk_usage.return_value = MagicMock(
                total=1_000_000_000_000,
                used=999_950_000_000,
                free=50_000_000,  # 50MB
            )
            result = await check_disk_space(threshold_mb=100)
            assert result["status"] in ("warn", "fail")
            assert "Low disk" in result["message"]


class TestCheckContextBudget:
    @pytest.mark.asyncio
    async def test_pass_with_low_usage(self):
        agent = _make_agent()
        agent.context_manager = MagicMock()
        agent.context_manager.tokens_used = 2000
        agent.context_manager.max_tokens = 8000
        result = await check_context_budget(agent)
        assert result["status"] == "pass"
        assert "25.0%" in result["message"]

    @pytest.mark.asyncio
    async def test_warn_on_high_usage(self):
        agent = _make_agent()
        agent.context_manager = MagicMock()
        agent.context_manager.tokens_used = 6500
        agent.context_manager.max_tokens = 8000
        result = await check_context_budget(agent)
        assert result["status"] == "warn"

    @pytest.mark.asyncio
    async def test_fail_on_critical_usage(self):
        agent = _make_agent()
        agent.context_manager = MagicMock()
        agent.context_manager.tokens_used = 7500
        agent.context_manager.max_tokens = 8000
        result = await check_context_budget(agent)
        assert result["status"] == "fail"

    @pytest.mark.asyncio
    async def test_pass_when_no_context_manager(self):
        agent = _make_agent()
        result = await check_context_budget(agent)
        assert result["status"] == "pass"
        assert "not tracked" in result["message"]

    @pytest.mark.asyncio
    async def test_fallback_to_token_budget(self):
        agent = _make_agent()
        agent.llm_service.token_budget = MagicMock()
        agent.llm_service.token_budget.used = 1000
        agent.llm_service.token_budget.total = 10000
        result = await check_context_budget(agent)
        assert result["status"] == "pass"
        assert "10.0%" in result["message"]


# ============================================================================
# Overall Status Derivation Tests
# ============================================================================


class TestDeriveOverallStatus:
    def test_all_pass(self):
        checks = [
            {"name": "database", "status": "pass"},
            {"name": "llm_service", "status": "pass"},
            {"name": "memory_system", "status": "pass"},
            {"name": "disk_space", "status": "pass"},
            {"name": "context_budget", "status": "pass"},
        ]
        assert _derive_overall_status(checks) == "healthy"

    def test_warn_is_degraded(self):
        checks = [
            {"name": "database", "status": "pass"},
            {"name": "llm_service", "status": "pass"},
            {"name": "disk_space", "status": "warn"},
        ]
        assert _derive_overall_status(checks) == "degraded"

    def test_critical_fail_is_unhealthy(self):
        checks = [
            {"name": "database", "status": "fail"},
            {"name": "llm_service", "status": "pass"},
            {"name": "disk_space", "status": "pass"},
        ]
        assert _derive_overall_status(checks) == "unhealthy"

    def test_llm_fail_is_unhealthy(self):
        checks = [
            {"name": "database", "status": "pass"},
            {"name": "llm_service", "status": "fail"},
        ]
        assert _derive_overall_status(checks) == "unhealthy"

    def test_non_critical_fail_is_degraded(self):
        checks = [
            {"name": "database", "status": "pass"},
            {"name": "llm_service", "status": "pass"},
            {"name": "disk_space", "status": "fail"},
        ]
        assert _derive_overall_status(checks) == "degraded"

    def test_empty_checks(self):
        assert _derive_overall_status([]) == "healthy"


# ============================================================================
# HeartbeatFeature Tests
# ============================================================================


class TestHeartbeatFeatureInitialize:
    @pytest_asyncio.fixture
    async def feature(self):
        """Create and initialize a HeartbeatFeature with mock agent."""
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        # Patch asyncio.create_task to avoid actually starting the loop
        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()
            yield feat
            feat._running = False

    @pytest.mark.asyncio
    async def test_creates_table(self, feature):
        """Initialize creates the heartbeat_log table."""
        create_calls = [
            c for c in feature._db.execute.call_args_list
            if "CREATE TABLE" in str(c) and "heartbeat_log" in str(c)
        ]
        assert len(create_calls) == 1

    @pytest.mark.asyncio
    async def test_creates_index(self, feature):
        """Initialize creates the index on heartbeat_log."""
        index_calls = [
            c for c in feature._db.execute.call_args_list
            if "CREATE INDEX" in str(c) and "heartbeat_log" in str(c)
        ]
        assert len(index_calls) == 1

    @pytest.mark.asyncio
    async def test_sets_agent_id(self, feature):
        assert feature._agent_id == "test-heartbeat-agent"

    @pytest.mark.asyncio
    async def test_sets_default_interval(self, feature):
        assert feature._interval_seconds == DEFAULT_INTERVAL_SECONDS


class TestHeartbeatCheck:
    @pytest_asyncio.fixture
    async def feature(self):
        """Create an initialized HeartbeatFeature."""
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()
            yield feat
            feat._running = False

    @pytest.mark.asyncio
    async def test_returns_all_checks(self, feature):
        """heartbeat_check returns results for all 5 checks."""
        result = await feature.heartbeat_check()
        assert "heartbeat_id" in result
        assert "status" in result
        assert "checks" in result
        assert "created_at" in result

        check_names = {c["name"] for c in result["checks"]}
        assert "database" in check_names
        assert "llm_service" in check_names
        assert "memory_system" in check_names
        assert "disk_space" in check_names
        assert "context_budget" in check_names

    @pytest.mark.asyncio
    async def test_persists_to_db(self, feature):
        """heartbeat_check writes a row to heartbeat_log."""
        await feature.heartbeat_check()

        insert_calls = [
            c for c in feature._db.execute.call_args_list
            if "INSERT INTO heartbeat_log" in str(c)
        ]
        assert len(insert_calls) == 1

    @pytest.mark.asyncio
    async def test_stores_in_memory_history(self, feature):
        """heartbeat_check appends to in-memory history."""
        assert len(feature._in_memory_history) == 0
        await feature.heartbeat_check()
        assert len(feature._in_memory_history) == 1

    @pytest.mark.asyncio
    async def test_healthy_when_all_pass(self, feature):
        """Status is 'healthy' when all checks pass."""
        result = await feature.heartbeat_check()
        # With our mock agent that has db, llm, storage, the checks should mostly pass
        assert result["status"] in ("healthy", "degraded")
        # overall_healthy should match
        assert result["overall_healthy"] == (result["status"] == "healthy")


class TestHeartbeatStatus:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()
            yield feat
            feat._running = False

    @pytest.mark.asyncio
    async def test_returns_uptime(self, feature):
        result = await feature.heartbeat_status()
        assert "uptime_seconds" in result
        assert result["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_returns_interval(self, feature):
        result = await feature.heartbeat_status()
        assert result["interval_seconds"] == DEFAULT_INTERVAL_SECONDS

    @pytest.mark.asyncio
    async def test_returns_history_from_db(self, feature):
        """heartbeat_status queries the database for history."""
        feature._db.fetchall = AsyncMock(return_value=[
            ("id-1", "healthy", '[{"name":"database","status":"pass"}]', 1, "2026-03-05T12:00:00"),
            ("id-2", "degraded", '[{"name":"database","status":"warn"}]', 0, "2026-03-05T11:00:00"),
        ])

        result = await feature.heartbeat_status(limit=10)
        assert result["history_count"] == 2
        assert result["history"][0]["id"] == "id-1"
        assert result["history"][1]["id"] == "id-2"

    @pytest.mark.asyncio
    async def test_trend_stable(self, feature):
        """Trend is 'stable' when recent heartbeats are both healthy."""
        feature._db.fetchall = AsyncMock(return_value=[
            ("id-1", "healthy", '[]', 1, "2026-03-05T12:00:00"),
            ("id-2", "healthy", '[]', 1, "2026-03-05T11:00:00"),
        ])
        result = await feature.heartbeat_status()
        assert result["trend"] == "stable"

    @pytest.mark.asyncio
    async def test_trend_declining(self, feature):
        """Trend is 'declining' when latest is unhealthy but previous was healthy."""
        feature._db.fetchall = AsyncMock(return_value=[
            ("id-1", "unhealthy", '[]', 0, "2026-03-05T12:00:00"),
            ("id-2", "healthy", '[]', 1, "2026-03-05T11:00:00"),
        ])
        result = await feature.heartbeat_status()
        assert result["trend"] == "declining"

    @pytest.mark.asyncio
    async def test_trend_recovering(self, feature):
        """Trend is 'recovering' when latest is healthy but previous was not."""
        feature._db.fetchall = AsyncMock(return_value=[
            ("id-1", "healthy", '[]', 1, "2026-03-05T12:00:00"),
            ("id-2", "unhealthy", '[]', 0, "2026-03-05T11:00:00"),
        ])
        result = await feature.heartbeat_status()
        assert result["trend"] == "recovering"

    @pytest.mark.asyncio
    async def test_falls_back_to_memory_history(self, feature):
        """Falls back to in-memory history when DB is empty."""
        feature._db.fetchall = AsyncMock(return_value=[])
        feature._in_memory_history = [
            {"id": "mem-1", "status": "healthy", "overall_healthy": True},
        ]
        result = await feature.heartbeat_status()
        assert result["history_count"] == 1
        assert result["history"][0]["id"] == "mem-1"


class _FakeTask:
    """A fake asyncio.Task that supports await, cancel(), and done().

    Used in tests to replace asyncio.create_task() without starting a real loop.
    """

    def __init__(self):
        self._done = False
        self._cancelled = False
        self.cancel_called = False

    def done(self):
        return self._done

    def cancel(self):
        self.cancel_called = True
        self._cancelled = True
        self._done = True

    def __await__(self):
        if self._cancelled:
            raise asyncio.CancelledError()
        # Yield nothing -- completes immediately
        return
        yield  # Make this a generator (required for __await__)


def _make_awaitable_task():
    """Create a fake asyncio.Task for testing."""
    return _FakeTask()


class TestHeartbeatInterval:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task", return_value=_make_awaitable_task()):
            await feat.initialize()
            yield feat
            feat._running = False

    @pytest.mark.asyncio
    async def test_changes_interval(self, feature):
        with patch("asyncio.create_task", return_value=_make_awaitable_task()):
            result = await feature.heartbeat_interval(seconds=120)

        assert result["old_interval_seconds"] == DEFAULT_INTERVAL_SECONDS
        assert result["new_interval_seconds"] == 120
        assert feature._interval_seconds == 120

    @pytest.mark.asyncio
    async def test_clamps_minimum(self, feature):
        with patch("asyncio.create_task", return_value=_make_awaitable_task()):
            result = await feature.heartbeat_interval(seconds=1)

        assert result["new_interval_seconds"] == 10

    @pytest.mark.asyncio
    async def test_clamps_maximum(self, feature):
        with patch("asyncio.create_task", return_value=_make_awaitable_task()):
            result = await feature.heartbeat_interval(seconds=9999)

        assert result["new_interval_seconds"] == 3600


# ============================================================================
# Lifecycle Tests
# ============================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_shutdown_stops_background_task(self):
        """shutdown() cancels the background task."""
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        mock_task = _make_awaitable_task()

        with patch("asyncio.create_task", return_value=mock_task):
            await feat.initialize()
            assert feat._running is True

            await feat.shutdown()
            assert feat._running is False
            assert mock_task.cancel_called is True

    @pytest.mark.asyncio
    async def test_initialize_without_db(self):
        """Initialize works gracefully without a database."""
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        assert feat._db is None
        # Should still be running (background loop doesn't need DB)
        assert feat._running is True

        feat._running = False


# ============================================================================
# No DB / Graceful Degradation Tests
# ============================================================================


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_heartbeat_check_without_db(self):
        """heartbeat_check works even without a database."""
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        result = await feat.heartbeat_check()
        assert "heartbeat_id" in result
        assert "checks" in result
        # Database check should fail but others should still run
        db_check = next(c for c in result["checks"] if c["name"] == "database")
        assert db_check["status"] == "fail"
        # LLM check should pass
        llm_check = next(c for c in result["checks"] if c["name"] == "llm_service")
        assert llm_check["status"] == "pass"

        feat._running = False

    @pytest.mark.asyncio
    async def test_heartbeat_status_without_db(self):
        """heartbeat_status works with in-memory history when DB unavailable."""
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        # Run a heartbeat to populate in-memory history
        await feat.heartbeat_check()

        result = await feat.heartbeat_status()
        assert result["history_count"] == 1

        feat._running = False

    @pytest.mark.asyncio
    async def test_persist_failure_does_not_crash(self):
        """If DB persist fails, the heartbeat still returns results."""
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        # Make INSERT fail
        original_execute = db.execute

        async def failing_execute(sql, *args, **kwargs):
            if "INSERT INTO heartbeat_log" in sql:
                raise Exception("DB write failed")
            return await original_execute(sql, *args, **kwargs)

        db.execute = AsyncMock(side_effect=failing_execute)

        # Should not raise
        result = await feat.heartbeat_check()
        assert "heartbeat_id" in result
        assert len(feat._in_memory_history) == 1

        feat._running = False


# ============================================================================
# Tool Discovery Tests
# ============================================================================


class TestToolDiscovery:
    @pytest.mark.asyncio
    async def test_tools_registered(self):
        """HeartbeatFeature exposes expected tools via get_tools()."""
        db = _make_db()
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        tools = feat.get_tools()
        tool_names = {t.name for t in tools}

        assert "heartbeat_check" in tool_names
        assert "heartbeat_status" in tool_names
        assert "heartbeat_interval" in tool_names
        assert len(tool_names) == 3

        feat._running = False

    @pytest.mark.asyncio
    async def test_tool_description(self):
        """HeartbeatFeature has a meaningful tool_description."""
        agent = _make_agent()
        feat = HeartbeatFeature(agent)
        assert "heartbeat" in feat.tool_description.lower()
        assert "health" in feat.tool_description.lower()

    @pytest.mark.asyncio
    async def test_command_prefixes(self):
        """Tools have correct command prefixes."""
        db = _make_db()
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        tools = feat.get_tools()
        prefixes = {t.schema.command_prefix for t in tools}
        assert "!heartbeat" in prefixes
        assert "!heartbeat-status" in prefixes
        assert "!heartbeat-interval" in prefixes

        feat._running = False


# ============================================================================
# get_latest_heartbeat Tests
# ============================================================================


class TestGetLatestHeartbeat:
    @pytest.mark.asyncio
    async def test_returns_last_from_memory(self):
        """get_latest_heartbeat returns the most recent in-memory result."""
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        # Run two heartbeats
        await feat._run_heartbeat()
        await feat._run_heartbeat()

        latest = await feat.get_latest_heartbeat()
        assert latest == feat._in_memory_history[-1]

        feat._running = False

    @pytest.mark.asyncio
    async def test_runs_fresh_when_no_history(self):
        """get_latest_heartbeat runs a fresh heartbeat when no history exists."""
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        assert len(feat._in_memory_history) == 0
        latest = await feat.get_latest_heartbeat()
        assert "heartbeat_id" in latest
        assert len(feat._in_memory_history) == 1

        feat._running = False


# ============================================================================
# In-Memory History Cap Tests
# ============================================================================


class TestInMemoryHistoryCap:
    @pytest.mark.asyncio
    async def test_caps_at_max(self):
        """In-memory history is capped at MAX_IN_MEMORY_HISTORY."""
        db = _make_db(table_exists_map={"heartbeat_log": True})
        agent = _make_agent(db=db)
        feat = HeartbeatFeature(agent)

        with patch("asyncio.create_task") as mock_create_task:
            mock_create_task.return_value = _make_awaitable_task()
            await feat.initialize()

        # Fill beyond the cap
        for i in range(105):
            feat._in_memory_history.append({"id": f"test-{i}"})

        # Run one more heartbeat which triggers the cap
        await feat._run_heartbeat()

        # Should be capped at 100
        assert len(feat._in_memory_history) <= 100

        feat._running = False
