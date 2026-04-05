"""
Unit tests for dynamic tool loading.

Tests the explore-then-direct-call optimization where features explored
via subagent dispatch expose their individual tools for direct calling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.tools.base import ToolSchema, ToolParameter, ToolCategory


# =============================================================================
# Helpers
# =============================================================================

def _make_mock_tool(name: str, description: str = "test tool"):
    """Create a mock AgentTool with a proper schema."""
    tool = MagicMock()
    tool.name = name
    tool.schema = ToolSchema(
        name=name,
        description=description,
        category=ToolCategory.SYSTEM,
        parameters=[
            ToolParameter(name="query", type="string", description="test param"),
        ],
    )
    tool.execute = AsyncMock(return_value={"success": True, "result": "ok", "tool": name})
    return tool


def _make_mock_feature(tool_name: str, tools: list):
    """Create a mock Feature with given tool_name and tools."""
    feature = MagicMock()
    feature.tool_name = tool_name
    feature.get_tools.return_value = tools
    feature.name = tool_name.replace("_", " ").title()
    feature.tool_description = f"Mock feature: {tool_name}"
    feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Dispatch to {tool_name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What to do"},
                },
                "required": ["task"],
            },
        },
    }
    return feature


@pytest.fixture
def agent():
    """Create a KestrelAgent with mocked dependencies for testing."""
    with patch("kestrel_sovereign.kestrel_agent.LLMService"):
        a = KestrelAgent(did="did:test:agent")
    # Add a minimal features dict for _build_feature_tools
    a.features = {}
    return a


# =============================================================================
# State initialization
# =============================================================================

class TestDynamicToolLoadingInit:

    def test_initial_state_empty(self, agent):
        """New agent has empty dynamic tool state."""
        assert agent._explored_features == {}
        assert agent._direct_tools == {}
        assert agent._direct_tool_defs == []
        assert agent._tool_to_feature == {}


# =============================================================================
# Tool registration
# =============================================================================

class TestRegisterExploredFeatureTools:

    @pytest.mark.asyncio
    async def test_register_populates_direct_tools(self, agent):
        """After registration, tools appear in _direct_tools and _direct_tool_defs."""
        tools = [_make_mock_tool("list_models"), _make_mock_tool("get_current_model")]
        feature = _make_mock_feature("model_agent", tools)

        await agent._register_explored_feature_tools(feature)

        assert "model_agent" in agent._explored_features
        assert "list_models" in agent._direct_tools
        assert "get_current_model" in agent._direct_tools
        assert len(agent._direct_tool_defs) == 2
        assert agent._tool_to_feature["list_models"] == "model_agent"
        assert agent._tool_to_feature["get_current_model"] == "model_agent"

    @pytest.mark.asyncio
    async def test_register_idempotent(self, agent):
        """Registering same feature twice doesn't double tools."""
        tools = [_make_mock_tool("list_models")]
        feature = _make_mock_feature("model_agent", tools)

        await agent._register_explored_feature_tools(feature)
        await agent._register_explored_feature_tools(feature)

        assert len(agent._direct_tools) == 1
        assert len(agent._direct_tool_defs) == 1

    @pytest.mark.asyncio
    async def test_tool_defs_in_openai_format(self, agent):
        """Registered tool defs match OpenAI function calling format."""
        tools = [_make_mock_tool("list_models")]
        feature = _make_mock_feature("model_agent", tools)

        await agent._register_explored_feature_tools(feature)

        tool_def = agent._direct_tool_defs[0]
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "list_models"
        assert "parameters" in tool_def["function"]
        assert tool_def["function"]["parameters"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_register_multiple_features(self, agent):
        """Multiple features register their tools independently."""
        model_tools = [_make_mock_tool("list_models")]
        memory_tools = [_make_mock_tool("search_memory"), _make_mock_tool("memory_status")]
        model_feature = _make_mock_feature("model_agent", model_tools)
        memory_feature = _make_mock_feature("memory_feature", memory_tools)

        await agent._register_explored_feature_tools(model_feature)
        await agent._register_explored_feature_tools(memory_feature)

        assert len(agent._explored_features) == 2
        assert len(agent._direct_tools) == 3
        assert len(agent._direct_tool_defs) == 3


# =============================================================================
# Name collision handling
# =============================================================================

class TestNameCollision:

    @pytest.mark.asyncio
    async def test_collision_prefixed_with_feature_name(self, agent):
        """Colliding tool names get feature_name__tool_name prefix."""
        tools_a = [_make_mock_tool("status")]
        tools_b = [_make_mock_tool("status")]
        feature_a = _make_mock_feature("model_agent", tools_a)
        feature_b = _make_mock_feature("memory_feature", tools_b)

        await agent._register_explored_feature_tools(feature_a)
        await agent._register_explored_feature_tools(feature_b)

        assert "status" in agent._direct_tools
        assert "memory_feature__status" in agent._direct_tools
        assert len(agent._direct_tools) == 2

    @pytest.mark.asyncio
    async def test_collision_tool_def_uses_prefixed_name(self, agent):
        """Prefixed tool's OpenAI def uses the prefixed name."""
        tools_a = [_make_mock_tool("status")]
        tools_b = [_make_mock_tool("status")]
        feature_a = _make_mock_feature("model_agent", tools_a)
        feature_b = _make_mock_feature("memory_feature", tools_b)

        await agent._register_explored_feature_tools(feature_a)
        await agent._register_explored_feature_tools(feature_b)

        names = [d["function"]["name"] for d in agent._direct_tool_defs]
        assert "status" in names
        assert "memory_feature__status" in names


# =============================================================================
# Build all tools
# =============================================================================

class TestBuildAllTools:

    @pytest.mark.asyncio
    async def test_explored_feature_replaces_dispatcher_with_direct_tools(self, agent):
        """Once explored, feature dispatch tool is replaced by direct tools."""
        # Add a feature so _build_feature_tools returns something
        feature = _make_mock_feature("model_agent", [_make_mock_tool("list_models")])
        agent.features = {"ModelAgent": feature}

        # Register direct tools (promotes tools, skips dispatcher)
        await agent._register_explored_feature_tools(feature)

        all_tools = agent._build_all_tools()

        # Should have only direct tools — dispatcher is skipped for explored features
        names = [t["function"]["name"] for t in all_tools]
        assert "model_agent" not in names  # dispatcher skipped
        assert "list_models" in names  # direct tool

    def test_empty_when_no_features(self, agent):
        """_build_all_tools returns empty list with no features or direct tools."""
        assert agent._build_all_tools() == []

    def test_only_feature_tools_when_none_explored(self, agent):
        """Before any exploration, only feature dispatch tools returned."""
        feature = _make_mock_feature("model_agent", [])
        agent.features = {"ModelAgent": feature}

        all_tools = agent._build_all_tools()
        assert len(all_tools) == 1
        assert all_tools[0]["function"]["name"] == "model_agent"


# =============================================================================
# Direct tool execution
# =============================================================================

class TestDirectToolExecution:

    @pytest.mark.asyncio
    async def test_direct_tool_executes_with_args(self, agent):
        """Direct tool .execute() is called with the provided arguments."""
        tool = _make_mock_tool("list_models")
        agent._direct_tools["list_models"] = tool

        result = await agent._direct_tools["list_models"].execute(query="openai")

        tool.execute.assert_called_once_with(query="openai")
        assert result["success"] is True


# =============================================================================
# Eviction
# =============================================================================

class TestEviction:

    @pytest.mark.asyncio
    async def test_eviction_at_capacity(self, agent):
        """Oldest feature's tools evicted when over MAX_DIRECT_TOOLS."""
        agent.MAX_DIRECT_TOOLS = 5  # Low cap for testing

        # Register feature A with 3 tools
        tools_a = [_make_mock_tool(f"tool_a_{i}") for i in range(3)]
        feature_a = _make_mock_feature("feature_a", tools_a)
        await agent._register_explored_feature_tools(feature_a)
        assert len(agent._direct_tools) == 3

        # Register feature B with 3 tools -> total 6, exceeds cap of 5
        tools_b = [_make_mock_tool(f"tool_b_{i}") for i in range(3)]
        feature_b = _make_mock_feature("feature_b", tools_b)
        await agent._register_explored_feature_tools(feature_b)

        # Feature A (oldest) should be evicted
        assert "feature_a" not in agent._explored_features
        assert all(f"tool_a_{i}" not in agent._direct_tools for i in range(3))
        # Feature B tools remain
        assert "feature_b" in agent._explored_features
        assert all(f"tool_b_{i}" in agent._direct_tools for i in range(3))
        assert len(agent._direct_tools) == 3

    @pytest.mark.asyncio
    async def test_eviction_removes_tool_defs(self, agent):
        """Evicted tools are also removed from _direct_tool_defs."""
        agent.MAX_DIRECT_TOOLS = 3

        tools_a = [_make_mock_tool("old_tool_1"), _make_mock_tool("old_tool_2")]
        feature_a = _make_mock_feature("old_feature", tools_a)
        await agent._register_explored_feature_tools(feature_a)

        tools_b = [_make_mock_tool("new_tool_1"), _make_mock_tool("new_tool_2")]
        feature_b = _make_mock_feature("new_feature", tools_b)
        await agent._register_explored_feature_tools(feature_b)

        def_names = [d["function"]["name"] for d in agent._direct_tool_defs]
        assert "old_tool_1" not in def_names
        assert "old_tool_2" not in def_names
        assert "new_tool_1" in def_names
        assert "new_tool_2" in def_names

    @pytest.mark.asyncio
    async def test_no_eviction_under_capacity(self, agent):
        """No eviction when total tools are under the cap."""
        agent.MAX_DIRECT_TOOLS = 60

        tools = [_make_mock_tool(f"tool_{i}") for i in range(5)]
        feature = _make_mock_feature("small_feature", tools)
        await agent._register_explored_feature_tools(feature)

        assert len(agent._direct_tools) == 5
        assert "small_feature" in agent._explored_features


# =============================================================================
# Feature dispatch still works alongside direct tools
# =============================================================================

class TestFeatureDispatchUnaffected:

    @pytest.mark.asyncio
    async def test_feature_dispatch_removed_after_exploration(self, agent):
        """Feature dispatch tool is removed when direct tools are registered."""
        feature = _make_mock_feature("model_agent", [_make_mock_tool("list_models")])
        agent.features = {"ModelAgent": feature}

        # Before exploration — dispatch tool present
        tools_before = agent._build_feature_tools()
        assert len(tools_before) == 1
        assert tools_before[0]["function"]["name"] == "model_agent"

        # After exploration — dispatch tool removed (direct tools replace it)
        await agent._register_explored_feature_tools(feature)
        tools_after = agent._build_feature_tools()
        assert len(tools_after) == 0
