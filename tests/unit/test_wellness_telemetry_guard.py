"""
Telemetry-Only Guard Tests for Operational Wellness.

COUNCIL CONDITION (Session 82ce894a):
Wellness metrics must be telemetry-only by default. They must NEVER be
injected into the agent's system prompt or context window. The agent can
observe its own metrics via tools, but metrics do not influence LLM
reasoning directly. This enforces the observation/action boundary.

These tests ENFORCE that boundary. If any of these tests fail, it means
wellness data has leaked into the agent's context -- a council violation.
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

from kestrel_sovereign.features.wellness.feature import (
    WELLNESS_TELEMETRY_ONLY,
    WellnessFeature,
)
from kestrel_sovereign.agent.context_builder import ContextBuilder


# ============================================================================
# Helpers
# ============================================================================


def _make_db(table_exists_map=None, fetchall_data=None, fetchone_data=None):
    """Create a mock AsyncDatabase."""
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

    agent.context_manager = None
    agent.llm_service = None

    return agent


# Wellness-related keywords that should NOT appear in context
WELLNESS_KEYWORDS = [
    "wellness_check",
    "wellness_history",
    "wellness_export",
    "overall_score",
    "constitutional_friction",
    "context_pressure",
    "interaction_depth",
    "session_continuity",
    "memory_health",
    "wellness_checkpoints",
    "friction_rate",
    "depth_score",
    "continuity_score",
    "health_score",
    "operational wellness",
]


# ============================================================================
# Test: WELLNESS_TELEMETRY_ONLY flag
# ============================================================================


class TestWellnessTelemetryOnlyFlag:
    def test_wellness_telemetry_only_flag_exists(self):
        """Verify WELLNESS_TELEMETRY_ONLY constant is True.

        Council Session 82ce894a mandated that wellness metrics are
        telemetry-only. This constant encodes that decision.
        """
        assert WELLNESS_TELEMETRY_ONLY is True

    def test_wellness_telemetry_only_is_module_level(self):
        """Verify the flag is importable at module level."""
        from kestrel_sovereign.features.wellness import feature as wellness_mod

        assert hasattr(wellness_mod, "WELLNESS_TELEMETRY_ONLY")
        assert wellness_mod.WELLNESS_TELEMETRY_ONLY is True


# ============================================================================
# Test: Wellness NOT in system prompt
# ============================================================================


class TestWellnessNotInSystemPrompt:
    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage for ContextBuilder."""
        storage = Mock()
        storage.search_chunks = Mock(return_value=[])
        return storage

    @pytest.fixture
    def context_builder(self, mock_storage):
        """Create a ContextBuilder with mock storage."""
        return ContextBuilder(mock_storage)

    def test_wellness_not_in_system_prompt_basic(self, context_builder):
        """Build a system prompt and verify no wellness data appears.

        The system prompt is the primary injection point for context.
        Wellness data must never appear here.
        """
        constitution = "Article 1: Protect the sovereign."
        prompt = context_builder.build_system_prompt(constitution)

        prompt_lower = prompt.lower()
        for keyword in WELLNESS_KEYWORDS:
            assert keyword.lower() not in prompt_lower, (
                f"Wellness keyword '{keyword}' found in system prompt. "
                f"COUNCIL VIOLATION: Wellness must be telemetry-only."
            )

    def test_wellness_not_in_system_prompt_with_briefing(self, context_builder):
        """Verify no wellness data in prompt when briefing is included."""
        constitution = "Article 1: Protect the sovereign."
        prompt = context_builder.build_system_prompt(
            constitution, include_briefing=True
        )

        prompt_lower = prompt.lower()
        for keyword in WELLNESS_KEYWORDS:
            assert keyword.lower() not in prompt_lower, (
                f"Wellness keyword '{keyword}' found in system prompt with briefing. "
                f"COUNCIL VIOLATION: Wellness must be telemetry-only."
            )

    def test_wellness_not_in_system_prompt_with_additional_context(self, context_builder):
        """Verify no wellness data even when additional context is provided."""
        constitution = "Article 1: Protect the sovereign."
        prompt = context_builder.build_system_prompt(
            constitution,
            additional_context="The user is discussing technical topics.",
        )

        prompt_lower = prompt.lower()
        for keyword in WELLNESS_KEYWORDS:
            assert keyword.lower() not in prompt_lower, (
                f"Wellness keyword '{keyword}' found in system prompt with additional context. "
                f"COUNCIL VIOLATION: Wellness must be telemetry-only."
            )

    def test_wellness_not_in_system_prompt_with_state_of_mind(self, context_builder):
        """Verify no wellness data when StateOfMind is provided.

        StateOfMind is the closest adjacent feature to wellness (it
        tracks governance mode). Ensure wellness does not piggyback.
        """
        constitution = "Article 1: Protect the sovereign."

        state_of_mind = MagicMock()
        state_of_mind.governance_mode = "standard"
        state_of_mind.active_conflicts = []
        state_of_mind.delegated_principles = []

        prompt = context_builder.build_system_prompt(
            constitution,
            state_of_mind=state_of_mind,
        )

        prompt_lower = prompt.lower()
        for keyword in WELLNESS_KEYWORDS:
            assert keyword.lower() not in prompt_lower, (
                f"Wellness keyword '{keyword}' found in system prompt with state_of_mind. "
                f"COUNCIL VIOLATION: Wellness must be telemetry-only."
            )


# ============================================================================
# Test: Wellness NOT in full context
# ============================================================================


class TestWellnessNotInContext:
    @pytest.fixture
    def async_mock_storage(self):
        """Create an async mock storage for ContextBuilder."""
        storage = Mock()
        storage.search_chunks = AsyncMock(return_value=[])
        return storage

    @pytest.fixture
    def context_builder(self, async_mock_storage):
        """Create a ContextBuilder with async mock storage."""
        return ContextBuilder(async_mock_storage)

    @pytest.mark.asyncio
    async def test_wellness_not_in_full_context(self, context_builder):
        """Build full context and verify no wellness metrics in any part.

        This tests the complete context assembly pipeline: system prompt,
        formatted history messages, and budget summary.
        """
        constitution = "Article 1: Protect the sovereign."
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = await context_builder.build_full_context(
            query="How are you?",
            history=history,
            constitution=constitution,
        )

        # Check system prompt
        prompt_lower = result["system_prompt"].lower()
        for keyword in WELLNESS_KEYWORDS:
            assert keyword.lower() not in prompt_lower, (
                f"Wellness keyword '{keyword}' found in full context system prompt. "
                f"COUNCIL VIOLATION: Wellness must be telemetry-only."
            )

        # Check all messages in the context
        for msg in result["messages"]:
            content_lower = msg.get("content", "").lower()
            for keyword in WELLNESS_KEYWORDS:
                assert keyword.lower() not in content_lower, (
                    f"Wellness keyword '{keyword}' found in context message. "
                    f"COUNCIL VIOLATION: Wellness must be telemetry-only."
                )

    @pytest.mark.asyncio
    async def test_wellness_not_in_context_with_long_history(self, context_builder):
        """Verify no wellness leakage even with extensive conversation history."""
        constitution = "Article 1: Protect the sovereign."
        history = [
            {"role": "user", "content": f"Message number {i}"}
            for i in range(50)
        ]

        result = await context_builder.build_full_context(
            query="What happened?",
            history=history,
            constitution=constitution,
            message_count=50,
        )

        all_text = result["system_prompt"]
        for msg in result["messages"]:
            all_text += " " + msg.get("content", "")

        all_text_lower = all_text.lower()
        for keyword in WELLNESS_KEYWORDS:
            assert keyword.lower() not in all_text_lower, (
                f"Wellness keyword '{keyword}' found in full context with long history. "
                f"COUNCIL VIOLATION: Wellness must be telemetry-only."
            )


# ============================================================================
# Test: Wellness tools are read-only (no context injection)
# ============================================================================


class TestWellnessToolsAreReadOnly:
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
    async def test_wellness_check_returns_dict_not_modifies_agent(self, feature):
        """Verify wellness_check returns data as a dict, does not modify agent state.

        The tool must return data to the caller (tool response channel) and
        must not set any attribute on the agent that would be picked up by
        the context builder.
        """
        # Track attribute assignments on the agent mock
        agent = feature.agent
        assigned_attrs = []
        original_setattr = type(agent).__setattr__

        def tracking_setattr(self_mock, name, value):
            assigned_attrs.append(name)
            original_setattr(self_mock, name, value)

        type(agent).__setattr__ = tracking_setattr
        try:
            result = await feature.wellness_check()
        finally:
            type(agent).__setattr__ = original_setattr

        # Returns a dict (tool response)
        assert isinstance(result, dict)
        assert "overall_score" in result
        assert "dimensions" in result

        # Agent should not have any wellness-related attribute set on it
        wellness_attrs = [a for a in assigned_attrs if "wellness" in a.lower()]
        assert wellness_attrs == [], (
            f"Wellness attributes were set on the agent: {wellness_attrs}. "
            f"COUNCIL VIOLATION: Wellness must be telemetry-only."
        )

    @pytest.mark.asyncio
    async def test_wellness_history_returns_dict_not_modifies_agent(self, feature):
        """Verify wellness_history returns data without modifying agent."""
        agent = feature.agent
        assigned_attrs = []
        original_setattr = type(agent).__setattr__

        def tracking_setattr(self_mock, name, value):
            assigned_attrs.append(name)
            original_setattr(self_mock, name, value)

        type(agent).__setattr__ = tracking_setattr
        try:
            result = await feature.wellness_history()
        finally:
            type(agent).__setattr__ = original_setattr

        assert isinstance(result, dict)

        wellness_attrs = [a for a in assigned_attrs if "wellness" in a.lower()]
        assert wellness_attrs == [], (
            f"Wellness attributes were set on the agent: {wellness_attrs}. "
            f"COUNCIL VIOLATION: Wellness must be telemetry-only."
        )

    @pytest.mark.asyncio
    async def test_wellness_tools_execute_as_tool_responses(self, feature):
        """Verify that wellness tools, when executed via get_tools(), return
        data wrapped in tool response format (success + result).

        Tool responses go to the tool result channel, NOT the system prompt.
        """
        tools = feature.get_tools()
        tool_by_name = {t.name: t for t in tools}

        # Execute wellness_check via the tool interface
        check_tool = tool_by_name["wellness_check"]
        result = await check_tool.execute()

        # Tool framework wraps in {"success": True, "result": ..., "tool": ...}
        assert result["success"] is True
        assert "result" in result
        assert result["tool"] == "wellness_check"

        # The inner result is the metrics dict
        inner = result["result"]
        assert "overall_score" in inner
        assert "dimensions" in inner


# ============================================================================
# Test: wellness_export does not inject into context
# ============================================================================


class TestWellnessExportDoesNotInject:
    @pytest_asyncio.fixture
    async def feature(self):
        """Create a WellnessFeature with mock export data."""
        db = _make_db(
            table_exists_map={
                "wellness_checkpoints": True,
                "security_audit_log": True,
                "conversation_history": True,
                "memory_episodes": True,
            },
            fetchall_data=[
                ("id-1", "test-agent", 0.6, '{"memory_health": {"health_score": 0.8}}', "2026-03-01T12:00:00"),
                ("id-2", "test-agent", 0.8, '{"memory_health": {"health_score": 0.9}}', "2026-03-02T12:00:00"),
            ],
            fetchone_data=(0,),
        )
        agent = _make_agent(db=db, agent_id="test-agent")
        feat = WellnessFeature(agent)
        await feat.initialize()
        return feat

    @pytest.mark.asyncio
    async def test_wellness_export_returns_to_tool_only(self, feature):
        """Verify wellness_export returns data as a tool response only.

        The export data must flow through the tool response channel,
        not be injected into context.
        """
        agent = feature.agent
        assigned_attrs = []
        original_setattr = type(agent).__setattr__

        def tracking_setattr(self_mock, name, value):
            assigned_attrs.append(name)
            original_setattr(self_mock, name, value)

        type(agent).__setattr__ = tracking_setattr
        try:
            result = await feature.wellness_export()
        finally:
            type(agent).__setattr__ = original_setattr

        # Returns a dict (tool response)
        assert isinstance(result, dict)
        assert "checkpoints" in result
        assert "count" in result
        assert result["export_format"] == "v1"

        # Agent should not store export data for context injection
        wellness_attrs = [a for a in assigned_attrs if "wellness" in a.lower()]
        assert wellness_attrs == [], (
            f"Wellness attributes were set on the agent during export: {wellness_attrs}. "
            f"COUNCIL VIOLATION: Wellness must be telemetry-only."
        )

    @pytest.mark.asyncio
    async def test_wellness_export_via_tool_interface(self, feature):
        """Verify wellness_export through the tool execution interface."""
        tools = feature.get_tools()
        tool_by_name = {t.name: t for t in tools}

        export_tool = tool_by_name["wellness_export"]
        result = await export_tool.execute()

        # Wrapped in tool response format
        assert result["success"] is True
        assert result["tool"] == "wellness_export"

        # The inner result has the export data
        inner = result["result"]
        assert "checkpoints" in inner
        assert inner["export_format"] == "v1"

    @pytest.mark.asyncio
    async def test_wellness_export_does_not_inject_into_context_builder(self, feature):
        """End-to-end: run wellness_export, then build context -- verify no leakage.

        This simulates the real-world scenario: the agent calls wellness_export
        via a tool, gets back data, and then we build a new context. The export
        data must not appear in the context.
        """
        # Run the export
        export_result = await feature.wellness_export()
        assert export_result["count"] == 2

        # Now build a system prompt (simulating the next turn)
        storage = Mock()
        storage.search_chunks = Mock(return_value=[])
        builder = ContextBuilder(storage)

        prompt = builder.build_system_prompt(
            constitution="Article 1: Protect the sovereign."
        )

        prompt_lower = prompt.lower()
        for keyword in WELLNESS_KEYWORDS:
            assert keyword.lower() not in prompt_lower, (
                f"Wellness keyword '{keyword}' leaked into system prompt after export. "
                f"COUNCIL VIOLATION: Wellness must be telemetry-only."
            )


# ============================================================================
# Test: Context builder docstrings reference council condition
# ============================================================================


class TestCouncilConditionDocumented:
    def test_build_system_prompt_docstring_references_council(self):
        """Verify build_system_prompt docstring contains the council condition."""
        docstring = ContextBuilder.build_system_prompt.__doc__
        assert docstring is not None
        assert "COUNCIL CONDITION" in docstring
        assert "telemetry-only" in docstring.lower()
        assert "82ce894a" in docstring

    def test_build_full_context_docstring_references_council(self):
        """Verify build_full_context docstring contains the council condition."""
        docstring = ContextBuilder.build_full_context.__doc__
        assert docstring is not None
        assert "COUNCIL CONDITION" in docstring
        assert "telemetry-only" in docstring.lower()

    def test_wellness_feature_class_docstring_references_council(self):
        """Verify WellnessFeature class docstring contains the council condition."""
        docstring = WellnessFeature.__doc__
        assert docstring is not None
        assert "COUNCIL CONDITION" in docstring
        assert "82ce894a" in docstring
        assert "telemetry-only" in docstring.lower()

    def test_wellness_check_docstring_references_council(self):
        """Verify wellness_check docstring contains the council condition."""
        docstring = WellnessFeature.wellness_check.__doc__
        assert docstring is not None
        assert "COUNCIL CONDITION" in docstring
        assert "tool caller only" in docstring.lower()

    def test_wellness_export_docstring_references_council(self):
        """Verify wellness_export docstring contains the council condition."""
        docstring = WellnessFeature.wellness_export.__doc__
        assert docstring is not None
        assert "COUNCIL CONDITION" in docstring
