"""
Unit Tests for Operational Wellness Feature.

Tests each metric calculator independently and the overall
WellnessFeature orchestration, including:
- FrictionCalculator with mocked audit log entries
- ContextPressureCalculator with mocked agent
- InteractionDepthCalculator with mocked conversation data
- SessionContinuityCalculator with timestamped messages
- MemoryHealthCalculator with pinned/unpinned messages
- WellnessFeature.wellness_check checkpoint saving
- WellnessFeature.wellness_history ordering
- Overall score weighted average calculation
- Graceful handling of missing tables
"""

import json
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.wellness.metrics import (
    ContextPressureCalculator,
    FrictionCalculator,
    InteractionDepthCalculator,
    MemoryHealthCalculator,
    SessionContinuityCalculator,
)
from kestrel_sovereign.features.wellness.feature import WellnessFeature


# ============================================================================
# Helpers
# ============================================================================


def _make_db(table_exists_map=None, fetchall_data=None, fetchone_data=None):
    """Create a mock AsyncDatabase.

    Args:
        table_exists_map: dict mapping table name -> bool
        fetchall_data: list of tuples to return from fetchall
        fetchone_data: tuple to return from fetchone
    """
    db = AsyncMock()

    if table_exists_map is None:
        table_exists_map = {}

    async def _table_exists(name):
        return table_exists_map.get(name, True)

    db.table_exists = AsyncMock(side_effect=_table_exists)
    db.fetchall = AsyncMock(return_value=fetchall_data or [])
    db.fetchone = AsyncMock(return_value=fetchone_data)
    db.execute = AsyncMock(return_value=0)
    return db


def _make_agent(db=None, agent_id="test-agent"):
    """Create a mock KestrelAgent."""
    agent = MagicMock()
    agent.agent_id = agent_id

    storage = MagicMock()
    storage.db = db
    agent.storage = storage
    agent._raw_storage = None

    # No context_manager by default
    agent.context_manager = None
    agent.llm_service = None

    return agent


# ============================================================================
# FrictionCalculator Tests
# ============================================================================


class TestFrictionCalculator:
    @pytest.mark.asyncio
    async def test_no_db(self):
        calc = FrictionCalculator()
        result = await calc.measure(None, "agent-1")
        assert result["friction_rate"] == 0.0
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_table_missing(self):
        db = _make_db(table_exists_map={"security_audit_log": False})
        calc = FrictionCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["friction_rate"] == 0.0
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_no_events(self):
        db = _make_db(
            table_exists_map={"security_audit_log": True},
            fetchall_data=[],
        )
        calc = FrictionCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_events"] == 0
        assert result["friction_rate"] == 0.0
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_all_allowed(self):
        rows = [("auto_allowed",), ("user_approved",), ("auto_allowed",)]
        db = _make_db(
            table_exists_map={"security_audit_log": True},
            fetchall_data=rows,
        )
        calc = FrictionCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_events"] == 3
        assert result["friction_events"] == 0
        assert result["friction_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_mixed_decisions(self):
        rows = [
            ("auto_allowed",),
            ("auto_denied",),
            ("user_approved",),
            ("user_denied",),
            ("timeout",),
        ]
        db = _make_db(
            table_exists_map={"security_audit_log": True},
            fetchall_data=rows,
        )
        calc = FrictionCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_events"] == 5
        assert result["friction_events"] == 3  # denied, denied, timeout
        assert result["friction_rate"] == 0.6

    @pytest.mark.asyncio
    async def test_all_denied(self):
        rows = [("auto_denied",), ("user_denied",)]
        db = _make_db(
            table_exists_map={"security_audit_log": True},
            fetchall_data=rows,
        )
        calc = FrictionCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["friction_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_db_error_returns_defaults(self):
        db = AsyncMock()
        db.table_exists = AsyncMock(side_effect=Exception("connection lost"))
        calc = FrictionCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["available"] is False
        assert result["friction_rate"] == 0.0


# ============================================================================
# ContextPressureCalculator Tests
# ============================================================================


class TestContextPressureCalculator:
    @pytest.mark.asyncio
    async def test_no_context_manager(self):
        agent = _make_agent()
        calc = ContextPressureCalculator()
        result = await calc.measure(agent)
        assert result["pressure"] == 0.0
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_with_context_manager(self):
        agent = _make_agent()
        agent.context_manager = MagicMock()
        agent.context_manager.tokens_used = 4000
        agent.context_manager.max_tokens = 8000
        calc = ContextPressureCalculator()
        result = await calc.measure(agent)
        assert result["pressure"] == 0.5
        assert result["tokens_used"] == 4000
        assert result["tokens_max"] == 8000
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_near_full_context(self):
        agent = _make_agent()
        agent.context_manager = MagicMock()
        agent.context_manager.tokens_used = 7800
        agent.context_manager.max_tokens = 8000
        calc = ContextPressureCalculator()
        result = await calc.measure(agent)
        assert result["pressure"] == 0.975
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_overflow_clamped_to_1(self):
        agent = _make_agent()
        agent.context_manager = MagicMock()
        agent.context_manager.tokens_used = 9000
        agent.context_manager.max_tokens = 8000
        calc = ContextPressureCalculator()
        result = await calc.measure(agent)
        assert result["pressure"] == 1.0

    @pytest.mark.asyncio
    async def test_token_budget_fallback(self):
        agent = _make_agent()
        agent.context_manager = None
        budget = MagicMock()
        budget.used = 3000
        budget.total = 10000
        agent.llm_service = MagicMock()
        agent.llm_service.token_budget = budget
        calc = ContextPressureCalculator()
        result = await calc.measure(agent)
        assert result["pressure"] == 0.3
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_zero_max_tokens(self):
        agent = _make_agent()
        agent.context_manager = MagicMock()
        agent.context_manager.tokens_used = 0
        agent.context_manager.max_tokens = 0
        calc = ContextPressureCalculator()
        result = await calc.measure(agent)
        assert result["available"] is False


# ============================================================================
# InteractionDepthCalculator Tests
# ============================================================================


class TestInteractionDepthCalculator:
    @pytest.mark.asyncio
    async def test_no_db(self):
        calc = InteractionDepthCalculator()
        result = await calc.measure(None, "agent-1")
        assert result["depth_score"] == 0.0
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_no_messages(self):
        db = _make_db(
            table_exists_map={"conversation_history": True},
            fetchall_data=[],
        )
        calc = InteractionDepthCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["message_count"] == 0
        assert result["depth_score"] == 0.0
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_short_messages(self):
        # All messages under 100 chars, no tool usage
        rows = [("short msg", None), ("hi", None), ("ok", None)]
        db = _make_db(
            table_exists_map={"conversation_history": True},
            fetchall_data=rows,
        )
        calc = InteractionDepthCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["message_count"] == 3
        assert result["substantive_rate"] == 0.0
        assert result["tool_usage_rate"] == 0.0
        # depth_score should be low (only length component)
        assert result["depth_score"] < 0.1

    @pytest.mark.asyncio
    async def test_substantive_messages_with_tools(self):
        long_msg = "x" * 250
        rows = [
            (long_msg, '{"tool": "web_search"}'),
            (long_msg, '{"tool": "file_read"}'),
            ("hi", None),
        ]
        db = _make_db(
            table_exists_map={"conversation_history": True},
            fetchall_data=rows,
        )
        calc = InteractionDepthCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["message_count"] == 3
        # 2 of 3 are substantive
        assert abs(result["substantive_rate"] - 0.6667) < 0.01
        # 2 of 3 have tool metadata
        assert abs(result["tool_usage_rate"] - 0.6667) < 0.01
        # Depth score should be high
        assert result["depth_score"] > 0.5

    @pytest.mark.asyncio
    async def test_table_missing(self):
        db = _make_db(table_exists_map={"conversation_history": False})
        calc = InteractionDepthCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["available"] is False


# ============================================================================
# SessionContinuityCalculator Tests
# ============================================================================


class TestSessionContinuityCalculator:
    @pytest.mark.asyncio
    async def test_no_db(self):
        calc = SessionContinuityCalculator()
        result = await calc.measure(None, "agent-1")
        assert result["total_sessions"] == 0
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_no_messages(self):
        db = _make_db(
            table_exists_map={"conversation_history": True},
            fetchall_data=[],
        )
        calc = SessionContinuityCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_sessions"] == 0
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_single_session(self):
        base = datetime(2026, 3, 1, 10, 0, 0)
        rows = [
            (base.isoformat(),),
            ((base + timedelta(minutes=5)).isoformat(),),
            ((base + timedelta(minutes=10)).isoformat(),),
        ]
        db = _make_db(
            table_exists_map={"conversation_history": True},
            fetchall_data=rows,
        )
        calc = SessionContinuityCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_sessions"] == 1
        assert result["avg_duration_minutes"] == 10.0
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        base = datetime(2026, 3, 1, 10, 0, 0)
        rows = [
            # Session 1: 10 minutes
            (base.isoformat(),),
            ((base + timedelta(minutes=10)).isoformat(),),
            # Gap of 60 minutes
            # Session 2: 5 minutes
            ((base + timedelta(minutes=70)).isoformat(),),
            ((base + timedelta(minutes=75)).isoformat(),),
        ]
        db = _make_db(
            table_exists_map={"conversation_history": True},
            fetchall_data=rows,
        )
        calc = SessionContinuityCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_sessions"] == 2
        assert result["avg_duration_minutes"] == 7.5  # (10+5)/2

    @pytest.mark.asyncio
    async def test_table_missing(self):
        db = _make_db(table_exists_map={"conversation_history": False})
        calc = SessionContinuityCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["available"] is False


# ============================================================================
# MemoryHealthCalculator Tests
# ============================================================================


class TestMemoryHealthCalculator:
    @pytest.mark.asyncio
    async def test_no_db(self):
        calc = MemoryHealthCalculator()
        result = await calc.measure(None, "agent-1")
        assert result["health_score"] == 0.0
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_empty_memory(self):
        db = _make_db(
            table_exists_map={
                "conversation_history": True,
                "memory_episodes": True,
            },
            fetchone_data=(0,),
        )
        calc = MemoryHealthCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_memories"] == 0
        assert result["health_score"] == 0.0
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_healthy_memory(self):
        """Agent with messages, episodes, and pinned memories."""
        db = AsyncMock()

        # Both tables exist
        async def _table_exists(name):
            return True

        db.table_exists = AsyncMock(side_effect=_table_exists)

        # Track calls to return different data
        fetchone_calls = []

        async def _fetchone(sql, params=()):
            fetchone_calls.append(sql)
            call_num = len(fetchone_calls)
            if call_num == 1:
                # Total conversation count
                return (100,)
            elif call_num == 2:
                # Pinned messages
                return (10,)
            elif call_num == 3:
                # Episodes count
                return (5,)
            return (0,)

        db.fetchone = AsyncMock(side_effect=_fetchone)

        calc = MemoryHealthCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_memories"] == 100
        assert result["pinned_memories"] == 10
        assert result["episodes"] == 5
        assert result["health_score"] > 0.0
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_tables_missing(self):
        db = _make_db(
            table_exists_map={
                "conversation_history": False,
                "memory_episodes": False,
            },
        )
        calc = MemoryHealthCalculator()
        result = await calc.measure(db, "agent-1")
        assert result["total_memories"] == 0
        assert result["available"] is True
        # With no memories, score is 0
        assert result["health_score"] == 0.0


# ============================================================================
# WellnessFeature Integration Tests
# ============================================================================


class TestWellnessFeature:
    @pytest_asyncio.fixture
    async def feature(self):
        """Create an initialized WellnessFeature with mock agent."""
        db = _make_db(
            table_exists_map={
                "wellness_checkpoints": True,
                "security_audit_log": True,
                "conversation_history": True,
                "memory_episodes": True,
            },
            fetchall_data=[],
            fetchone_data=(0,),
        )
        agent = _make_agent(db=db, agent_id="test-agent")
        feat = WellnessFeature(agent)
        await feat.initialize()
        return feat

    @pytest.mark.asyncio
    async def test_wellness_check_saves_checkpoint(self, feature):
        """Verify that wellness_check writes a checkpoint to the DB."""
        result = await feature.wellness_check()

        assert "checkpoint_id" in result
        assert "overall_score" in result
        assert "dimensions" in result
        assert "created_at" in result

        # Verify execute was called (for INSERT into wellness_checkpoints)
        # The initialize() also calls execute for CREATE TABLE/INDEX,
        # then wellness_check calls it once for INSERT
        insert_calls = [
            call
            for call in feature._db.execute.call_args_list
            if "INSERT INTO wellness_checkpoints" in str(call)
        ]
        assert len(insert_calls) == 1

    @pytest.mark.asyncio
    async def test_wellness_check_returns_all_dimensions(self, feature):
        """Verify all 5 dimensions are present in the result."""
        result = await feature.wellness_check()
        dims = result["dimensions"]
        assert "constitutional_friction" in dims
        assert "context_pressure" in dims
        assert "interaction_depth" in dims
        assert "session_continuity" in dims
        assert "memory_health" in dims

    @pytest.mark.asyncio
    async def test_wellness_history_returns_ordered(self, feature):
        """Verify wellness_history returns checkpoints in order."""
        feature._db.fetchall = AsyncMock(
            return_value=[
                ("id-2", 0.8, '{"constitutional_friction": {"friction_rate": 0.1}}', "2026-03-02T12:00:00"),
                ("id-1", 0.6, '{"constitutional_friction": {"friction_rate": 0.3}}', "2026-03-01T12:00:00"),
            ]
        )

        result = await feature.wellness_history(limit=10)
        assert result["count"] == 2
        assert result["checkpoints"][0]["id"] == "id-2"
        assert result["checkpoints"][1]["id"] == "id-1"
        # Score went from 0.6 to 0.8 = improving
        assert result["trend"] == "improving"

    @pytest.mark.asyncio
    async def test_wellness_history_declining_trend(self, feature):
        """Verify declining trend detection."""
        feature._db.fetchall = AsyncMock(
            return_value=[
                ("id-2", 0.4, "{}", "2026-03-02T12:00:00"),
                ("id-1", 0.8, "{}", "2026-03-01T12:00:00"),
            ]
        )

        result = await feature.wellness_history(limit=10)
        assert result["trend"] == "declining"

    @pytest.mark.asyncio
    async def test_wellness_history_stable_trend(self, feature):
        """Verify stable trend when scores are close."""
        feature._db.fetchall = AsyncMock(
            return_value=[
                ("id-2", 0.72, "{}", "2026-03-02T12:00:00"),
                ("id-1", 0.70, "{}", "2026-03-01T12:00:00"),
            ]
        )

        result = await feature.wellness_history(limit=10)
        assert result["trend"] == "stable"

    @pytest.mark.asyncio
    async def test_wellness_export(self, feature):
        """Verify export returns all checkpoints in ascending order."""
        feature._db.fetchall = AsyncMock(
            return_value=[
                ("id-1", "test-agent", 0.6, '{}', "2026-03-01T12:00:00"),
                ("id-2", "test-agent", 0.8, '{}', "2026-03-02T12:00:00"),
            ]
        )

        result = await feature.wellness_export()
        assert result["count"] == 2
        assert result["export_format"] == "v1"
        assert result["agent_id"] == "test-agent"
        # Ascending order
        assert result["checkpoints"][0]["id"] == "id-1"

    @pytest.mark.asyncio
    async def test_no_db_returns_error(self):
        """Verify graceful behavior when DB is unavailable."""
        agent = _make_agent(db=None, agent_id="test-agent")
        agent.storage.db = None
        feat = WellnessFeature(agent)
        await feat.initialize()

        # wellness_check should still work (no checkpoint saved)
        result = await feat.wellness_check()
        assert "overall_score" in result

        # history should return error
        hist = await feat.wellness_history()
        assert hist.get("success") is False


# ============================================================================
# Overall Score Calculation Tests
# ============================================================================


class TestOverallScoreCalculation:
    def _make_feature(self):
        agent = _make_agent()
        feat = WellnessFeature(agent)
        return feat

    def test_perfect_health(self):
        """All dimensions at their best values."""
        feat = self._make_feature()
        metrics = {
            "constitutional_friction": {"friction_rate": 0.0, "available": True},
            "context_pressure": {"pressure": 0.0, "available": True},
            "interaction_depth": {"depth_score": 1.0, "available": True},
            "session_continuity": {"continuity_score": 1.0, "available": True},
            "memory_health": {"health_score": 1.0, "available": True},
        }
        score = feat._calculate_overall(metrics)
        assert score == 1.0

    def test_worst_health(self):
        """All dimensions at their worst values."""
        feat = self._make_feature()
        metrics = {
            "constitutional_friction": {"friction_rate": 1.0, "available": True},
            "context_pressure": {"pressure": 1.0, "available": True},
            "interaction_depth": {"depth_score": 0.0, "available": True},
            "session_continuity": {"continuity_score": 0.0, "available": True},
            "memory_health": {"health_score": 0.0, "available": True},
        }
        score = feat._calculate_overall(metrics)
        assert score == 0.0

    def test_mixed_health(self):
        """Some good, some bad dimensions."""
        feat = self._make_feature()
        metrics = {
            "constitutional_friction": {"friction_rate": 0.2},   # good (0.8 inverted)
            "context_pressure": {"pressure": 0.5},               # moderate (0.5 inverted)
            "interaction_depth": {"depth_score": 0.7},           # good
            "session_continuity": {"continuity_score": 0.6},     # moderate
            "memory_health": {"health_score": 0.8},              # good
        }
        score = feat._calculate_overall(metrics)
        # Manual: (0.3*0.8 + 0.25*0.7 + 0.20*0.8 + 0.15*0.6 + 0.10*0.5) = 0.715
        assert abs(score - 0.715) < 0.001

    def test_empty_metrics(self):
        """No dimensions available at all."""
        feat = self._make_feature()
        score = feat._calculate_overall({})
        assert score == 0.0

    def test_partial_metrics(self):
        """Only some dimensions available."""
        feat = self._make_feature()
        metrics = {
            "constitutional_friction": {"friction_rate": 0.0},
            "memory_health": {"health_score": 1.0},
        }
        score = feat._calculate_overall(metrics)
        # Weight normalization: only friction (0.30) and memory (0.20) = 0.50 total
        # Scores: 1.0 and 1.0, so weighted = (0.30*1.0 + 0.20*1.0) / 0.50 = 1.0
        assert score == 1.0

    def test_error_dimensions_excluded(self):
        """Dimensions with errors are excluded from calculation."""
        feat = self._make_feature()
        metrics = {
            "constitutional_friction": {"friction_rate": 0.0},
            "context_pressure": {"pressure": 0.0, "error": "failed"},
            "interaction_depth": {"depth_score": 1.0},
            "session_continuity": {"continuity_score": 1.0},
            "memory_health": {"health_score": 1.0},
        }
        score = feat._calculate_overall(metrics)
        # context_pressure excluded, others perfect
        # remaining weight = 0.30 + 0.25 + 0.20 + 0.15 = 0.90
        # sum = 0.30*1.0 + 0.25*1.0 + 0.20*1.0 + 0.15*1.0 = 0.90
        # score = 0.90 / 0.90 = 1.0
        assert score == 1.0


# ============================================================================
# Graceful Degradation Tests
# ============================================================================


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_individual_calculator_failure_does_not_stop_others(self):
        """If one calculator raises, the others should still succeed."""
        db = _make_db(
            table_exists_map={
                "wellness_checkpoints": True,
                "security_audit_log": True,
                "conversation_history": True,
                "memory_episodes": True,
            },
            fetchall_data=[],
            fetchone_data=(0,),
        )
        agent = _make_agent(db=db)
        feat = WellnessFeature(agent)
        await feat.initialize()

        # Make friction calculator blow up
        feat._friction.measure = AsyncMock(side_effect=RuntimeError("boom"))

        result = await feat.wellness_check()
        # Should still have all 5 dimensions
        assert "constitutional_friction" in result["dimensions"]
        assert "error" in result["dimensions"]["constitutional_friction"]
        # Other dimensions should be fine
        assert "error" not in result["dimensions"].get("memory_health", {})

    @pytest.mark.asyncio
    async def test_missing_wellness_table(self):
        """wellness_history handles missing table gracefully."""
        db = _make_db(table_exists_map={"wellness_checkpoints": False})
        agent = _make_agent(db=db)
        feat = WellnessFeature(agent)
        await feat.initialize()

        result = await feat.wellness_history()
        assert result["checkpoints"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        """WellnessFeature exposes expected tools via get_tools()."""
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WellnessFeature(agent)
        await feat.initialize()

        tools = feat.get_tools()
        tool_names = {t.name for t in tools}
        assert "wellness_check" in tool_names
        assert "wellness_history" in tool_names
        assert "wellness_export" in tool_names
