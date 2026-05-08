"""
Integration tests for dynamic tool loading.

Tests the full explore-then-direct-call flow with a real KestrelAgent and
real features, mocking only the LLM responses to be deterministic.

The key insight: we mock LLMService.generate_with_messages to return
scripted tool calls, but the features and their tools are real. This
verifies the full dispatch pipeline.
"""

import logging
import logging.handlers
from contextlib import contextmanager

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.privacy import PrivacyMode

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

@contextmanager
def capture_logs(level=logging.DEBUG):
    """Context manager to capture log records."""
    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    collector = Collector()
    collector.setLevel(level)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(level)
    root.addHandler(collector)
    try:
        yield records
    finally:
        root.removeHandler(collector)
        root.setLevel(old_level)


# =============================================================================
# Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def agent(temp_db):
    """Create a real KestrelAgent with all features loaded."""
    llm_service = LLMService()
    a = KestrelAgent(
        did="did:test:dynamic-tools",
        storage_path=str(temp_db),
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await a.initialize()

    from tests.integration.conftest import complete_bootstrap, grant_permissions
    await complete_bootstrap(a)

    # The orchestrator-loop tests in TestExploreThenDirectFlow
    # exercise the SecurityHook chain: the LLM mock dispatches
    # ``model_agent`` as a subagent (PRE_SUBAGENT_CALL) and then
    # ``list_models`` as a direct tool (PRE_TOOL_USE through
    # _dispatch_direct_tool, which keys by lowercase feature
    # tool_name).  Grant ALLOW for exactly those pairs — anything
    # else stays at the default ASK so an unexpected dispatch hangs
    # and surfaces the regression.  See conftest.grant_permissions
    # for the full background (#879).
    await grant_permissions(
        a,
        ("ModelAgent", "model_agent"),    # subagent dispatch
        ("ModelAgent", "list_models"),    # direct feature tool
        ("model_agent", "list_models"),   # direct-tool path via _tool_to_feature
        reason="dynamic-tool-loading-e2e",
    )

    yield a

    await a.shutdown()
    await llm_service.close()


# =============================================================================
# Explore-then-direct flow
# =============================================================================

class TestExploreThenDirectFlow:
    """Test the full lifecycle: subagent dispatch -> tool registration -> direct call."""

    @pytest.mark.asyncio
    async def test_subagent_dispatch_registers_tools(self, agent):
        """After a subagent dispatch, the feature's individual tools are registered."""
        features = {f.tool_name: f for f in agent.features.values()}
        assert "model_agent" in features, f"Expected model_agent in {list(features.keys())}"

        # Before exploration: record existing direct tools count
        initial_count = len(agent._direct_tools)

        # Register (same as what _handle_orchestrator_response does after dispatch)
        feature = features["model_agent"]
        agent._register_explored_feature_tools(feature)

        # After exploration: model_agent tools are registered (count increased)
        assert len(agent._direct_tools) > initial_count
        assert "model_agent" in agent._explored_features
        model_tools = [name for name, feat in agent._tool_to_feature.items() if feat == "model_agent"]
        assert len(model_tools) > 0, "Expected at least one tool registered from model_agent"
        assert "list_models" in agent._direct_tools, (
            f"Expected list_models in direct tools, got: {list(agent._direct_tools.keys())}"
        )

    @pytest.mark.asyncio
    async def test_direct_tool_executes_real_feature_method(self, agent):
        """Direct tool calls execute the real feature method (no subagent LLM)."""
        features = {f.tool_name: f for f in agent.features.values()}
        feature = features["model_agent"]

        # Register tools
        agent._register_explored_feature_tools(feature)

        # Execute list_models directly through _direct_tools
        tool = agent._direct_tools["list_models"]
        result = await tool.execute(use_cache=False)

        # DynamicTool.execute wraps in {"success": ..., "result": ..., "tool": ...}.
        # Post-#1061 wave 10, list_models returns a ToolResult, so DynamicTool
        # serializes its envelope to a dict under result["result"] with
        # {"status": "ok", "confirmation": ..., "data": {"models": [...], "count": N}}.
        assert result["success"] is True, f"Direct tool execution failed: {result}"
        assert result["tool"] == "list_models"
        envelope = result["result"]
        assert isinstance(envelope, dict)
        assert envelope.get("status") == "ok"
        assert isinstance(envelope.get("data", {}).get("models"), list)

    @pytest.mark.asyncio
    async def test_build_all_tools_grows_after_exploration(self, agent):
        """_build_all_tools() includes direct tools after exploration."""
        initial_tools = agent._build_all_tools()
        initial_count = len(initial_tools)
        initial_names = {t["function"]["name"] for t in initial_tools}

        # Explore model_agent
        features = {f.tool_name: f for f in agent.features.values()}
        agent._register_explored_feature_tools(features["model_agent"])

        after_tools = agent._build_all_tools()
        after_names = {t["function"]["name"] for t in after_tools}

        # Should have more tools now
        assert len(after_tools) > initial_count
        # New tools should include list_models
        new_tools = after_names - initial_names
        assert "list_models" in new_tools, f"Expected list_models in new tools: {new_tools}"

    @pytest.mark.asyncio
    async def test_orchestrator_loop_explore_then_direct(self, agent):
        """
        Full orchestrator loop: LLM dispatches to model_agent (subagent),
        then calls list_models directly in the next iteration.

        We mock only the LLM to return scripted responses.
        """
        call_count = 0

        async def mock_generate(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # After subagent dispatch result comes back, LLM sees
                # direct tools and calls list_models directly
                if tools:
                    tool_names = [t["function"]["name"] for t in tools]
                    assert "list_models" in tool_names, (
                        f"list_models should be in tool list after exploration. "
                        f"Got: {tool_names}"
                    )
                return LLMResponse(
                    content="Now let me get the model list directly.",
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="list_models",
                            arguments={"use_cache": False},
                        )
                    ],
                )
            else:
                # Final summary (no more tool calls)
                return LLMResponse(
                    content="Here are the available models.",
                    tool_calls=None,
                )

        # Mock subagent execution (avoids needing real LLM credentials)
        features = {f.tool_name: f for f in agent.features.values()}
        model_feature = features["model_agent"]
        original_execute = model_feature.execute_as_subagent
        model_feature.execute_as_subagent = AsyncMock(return_value={
            "success": True,
            "result": "Found 3 models: gpt-4, claude-3, llama-3",
        })

        try:
            with patch.object(agent.llm_service, "generate_with_messages", side_effect=mock_generate):
                result = await agent._handle_orchestrator_response(
                    response=LLMResponse(
                        content="Let me check.",
                        tool_calls=[
                            ToolCall(
                                id="call_0",
                                name="model_agent",
                                arguments={"task": "list models"},
                            )
                        ],
                    ),
                    feature_tools=agent._build_all_tools(),
                    system_prompt="test",
                    force_local_only=False,
                    effective_model="test-model",
                    user_message="list all models",
                )
        finally:
            model_feature.execute_as_subagent = original_execute

        # Verify: subagent was called, tools were registered, direct call happened
        assert call_count >= 2, f"Expected at least 2 LLM calls, got {call_count}"
        assert "model_agent" in agent._explored_features
        assert "list_models" in agent._direct_tools

    @pytest.mark.asyncio
    async def test_direct_call_logged_as_direct_tool(self, agent):
        """Direct tool calls are logged with [DIRECT-TOOL]."""
        features = {f.tool_name: f for f in agent.features.values()}
        agent._register_explored_feature_tools(features["model_agent"])

        with capture_logs() as logs:
            call_count = 0

            async def mock_generate(messages, tools=None, **kwargs):
                nonlocal call_count
                call_count += 1
                return LLMResponse(content="Done.", tool_calls=None)

            with patch.object(agent.llm_service, "generate_with_messages", side_effect=mock_generate):
                await agent._handle_orchestrator_response(
                    response=LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="call_0", name="list_models", arguments={"use_cache": False})
                        ],
                    ),
                    feature_tools=agent._build_all_tools(),
                    system_prompt="test",
                    force_local_only=False,
                    effective_model="test-model",
                    user_message="list models",
                )

        direct_logs = [r for r in logs if "[DIRECT-TOOL]" in r.getMessage()]
        assert len(direct_logs) > 0, "Expected [DIRECT-TOOL] log entry for direct tool call"


# =============================================================================
# Multi-feature exploration
# =============================================================================

class TestMultiFeatureExploration:

    @pytest.mark.asyncio
    async def test_explore_multiple_features(self, agent):
        """Exploring multiple features registers all their tools."""
        features = {f.tool_name: f for f in agent.features.values()}

        explored = []
        for name in ["model_agent", "memory_feature"]:
            if name in features:
                agent._register_explored_feature_tools(features[name])
                explored.append(name)

        assert len(explored) >= 1, f"Expected at least model_agent in features: {list(features.keys())}"

        # All explored features' tools should be in direct_tools
        for feat_name in explored:
            feat_tools = [k for k, v in agent._tool_to_feature.items() if v == feat_name]
            assert len(feat_tools) > 0, f"Expected tools from {feat_name}"

    @pytest.mark.asyncio
    async def test_explored_tools_persist_across_calls(self, agent):
        """Tools explored in one call persist for the next (session-scoped)."""
        features = {f.tool_name: f for f in agent.features.values()}
        agent._register_explored_feature_tools(features["model_agent"])

        # First call: verify tools are available
        tools_first = agent._build_all_tools()
        names_first = {t["function"]["name"] for t in tools_first}
        assert "list_models" in names_first

        # Simulate a later orchestrator call
        tools_second = agent._build_all_tools()
        names_second = {t["function"]["name"] for t in tools_second}
        assert "list_models" in names_second, "Direct tools should persist across calls"
