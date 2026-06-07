"""
Unit tests for dynamic tool loading.

Tests the explore-then-direct-call optimization where features explored
via subagent dispatch expose their individual tools for direct calling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sdk.tools.base import ToolSchema, ToolParameter, ToolCategory


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


class ToolOnlyFeature:
    name = "ToolOnlyFeature"
    tool_name = "tool_only_feature"
    tool_description = "SDK-only tool feature"

    def __init__(self, tools):
        self._tools = tools

    def get_tools(self):
        return self._tools


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

    def test_register_populates_direct_tools(self, agent):
        """After registration, tools appear in _direct_tools and _direct_tool_defs."""
        tools = [_make_mock_tool("list_models"), _make_mock_tool("get_current_model")]
        feature = _make_mock_feature("model_agent", tools)

        agent._register_explored_feature_tools(feature)

        assert "model_agent" in agent._explored_features
        assert "list_models" in agent._direct_tools
        assert "get_current_model" in agent._direct_tools
        assert len(agent._direct_tool_defs) == 2
        assert agent._tool_to_feature["list_models"] == "model_agent"
        assert agent._tool_to_feature["get_current_model"] == "model_agent"

    def test_register_idempotent(self, agent):
        """Registering same feature twice doesn't double tools."""
        tools = [_make_mock_tool("list_models")]
        feature = _make_mock_feature("model_agent", tools)

        agent._register_explored_feature_tools(feature)
        agent._register_explored_feature_tools(feature)

        assert len(agent._direct_tools) == 1
        assert len(agent._direct_tool_defs) == 1

    def test_tool_defs_in_openai_format(self, agent):
        """Registered tool defs match OpenAI function calling format."""
        tools = [_make_mock_tool("list_models")]
        feature = _make_mock_feature("model_agent", tools)

        agent._register_explored_feature_tools(feature)

        tool_def = agent._direct_tool_defs[0]
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "list_models"
        assert "parameters" in tool_def["function"]
        assert tool_def["function"]["parameters"]["type"] == "object"

    def test_register_multiple_features(self, agent):
        """Multiple features register their tools independently."""
        model_tools = [_make_mock_tool("list_models")]
        memory_tools = [_make_mock_tool("search_memory"), _make_mock_tool("memory_status")]
        model_feature = _make_mock_feature("model_agent", model_tools)
        memory_feature = _make_mock_feature("memory_feature", memory_tools)

        agent._register_explored_feature_tools(model_feature)
        agent._register_explored_feature_tools(memory_feature)

        assert len(agent._explored_features) == 2
        assert len(agent._direct_tools) == 3
        assert len(agent._direct_tool_defs) == 3


# =============================================================================
# Name collision handling
# =============================================================================

class TestNameCollision:

    def test_collision_prefixed_with_feature_name(self, agent):
        """Colliding tool names get feature_name__tool_name prefix."""
        tools_a = [_make_mock_tool("status")]
        tools_b = [_make_mock_tool("status")]
        feature_a = _make_mock_feature("model_agent", tools_a)
        feature_b = _make_mock_feature("memory_feature", tools_b)

        agent._register_explored_feature_tools(feature_a)
        agent._register_explored_feature_tools(feature_b)

        assert "status" in agent._direct_tools
        assert "memory_feature__status" in agent._direct_tools
        assert len(agent._direct_tools) == 2

    def test_collision_tool_def_uses_prefixed_name(self, agent):
        """Prefixed tool's OpenAI def uses the prefixed name."""
        tools_a = [_make_mock_tool("status")]
        tools_b = [_make_mock_tool("status")]
        feature_a = _make_mock_feature("model_agent", tools_a)
        feature_b = _make_mock_feature("memory_feature", tools_b)

        agent._register_explored_feature_tools(feature_a)
        agent._register_explored_feature_tools(feature_b)

        names = [d["function"]["name"] for d in agent._direct_tool_defs]
        assert "status" in names
        assert "memory_feature__status" in names


# =============================================================================
# Build all tools
# =============================================================================

class TestBuildAllTools:

    def test_explored_feature_replaces_dispatcher_with_direct_tools(self, agent):
        """Once explored, feature dispatch tool is replaced by direct tools."""
        # Add a feature so _build_feature_tools returns something
        feature = _make_mock_feature("model_agent", [_make_mock_tool("list_models")])
        agent.features = {"ModelAgent": feature}

        # Register direct tools (promotes tools, skips dispatcher)
        agent._register_explored_feature_tools(feature)

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

    def test_sdk_tool_only_feature_is_not_exposed_as_subagent(self, agent):
        """SDK-only features with direct tools are not advertised as dispatchers."""
        feature = ToolOnlyFeature([_make_mock_tool("fetch_issue")])
        agent.features = {"ToolOnlyFeature": feature}

        assert agent._build_all_tools() == []
        assert agent._visible_features_by_tool_name() == {}
        assert agent._visible_known_tool_names() == set()

    def test_context_hidden_feature_dispatcher_is_not_exposed(self, agent):
        """Context profile can hide a feature dispatcher without unloading it."""
        feature = _make_mock_feature("model_agent", [])
        agent.features = {"ModelAgent": feature}
        agent._tool_context_hidden_features = {"model_agent"}

        assert agent._build_all_tools() == []

    def test_context_hidden_direct_tool_is_not_exposed(self, agent):
        """Context profile can hide direct tools without deleting registry state."""
        feature = _make_mock_feature("model_agent", [_make_mock_tool("list_models")])
        agent.features = {"ModelAgent": feature}
        agent._register_explored_feature_tools(feature)
        agent._tool_context_hidden_tools = {"list_models"}

        all_tools = agent._build_all_tools()

        assert all_tools == []
        assert "list_models" in agent._direct_tools

    def test_context_hidden_feature_hides_its_direct_tools(self, agent):
        """Hiding a feature also hides its promoted direct tools from context."""
        feature = _make_mock_feature("model_agent", [_make_mock_tool("list_models")])
        agent.features = {"ModelAgent": feature}
        agent._register_explored_feature_tools(feature)
        agent._tool_context_hidden_features = {"model_agent"}

        assert agent._build_all_tools() == []

    def test_context_hidden_direct_tool_is_not_listed_in_prompt(self, agent):
        """Hidden direct tools are removed from loaded-feature prompt text too."""
        feature = _make_mock_feature(
            "model_agent",
            [
                _make_mock_tool("list_models"),
                _make_mock_tool("get_current_model"),
            ],
        )
        agent.features = {"ModelAgent": feature}
        agent._tool_context_hidden_tools = {"get_current_model"}

        prompt = agent._build_features_prompt_section()

        assert "list_models" in prompt
        assert "get_current_model" not in prompt

    def test_sdk_tool_only_feature_is_not_listed_as_active_subagent(self, agent):
        """Loaded SDK-only tool features are not described as subagents."""
        feature = ToolOnlyFeature([_make_mock_tool("fetch_issue")])
        agent.features = {"ToolOnlyFeature": feature}

        prompt = agent._build_features_prompt_section()

        assert prompt == ""
        assert "ToolOnlyFeature" not in prompt
        assert "fetch_issue" not in prompt

    def test_visible_known_tool_names_excludes_hidden_context_tools(self, agent):
        """Orchestrator guardrails use only currently visible context tools."""
        feature = _make_mock_feature(
            "model_agent",
            [
                _make_mock_tool("list_models"),
                _make_mock_tool("get_current_model"),
            ],
        )
        agent.features = {"ModelAgent": feature}
        agent._register_explored_feature_tools(feature)
        agent._tool_context_hidden_tools = {"get_current_model"}

        known_tools = agent._visible_known_tool_names()

        assert "list_models" in known_tools
        assert "get_current_model" not in known_tools

    def test_visible_features_by_tool_name_excludes_hidden_features(self, agent):
        """Hidden feature dispatchers are not valid orchestrator targets."""
        feature = _make_mock_feature("model_agent", [])
        agent.features = {"ModelAgent": feature}
        agent._tool_context_hidden_features = {"model_agent"}

        assert agent._visible_features_by_tool_name() == {}

    @pytest.mark.asyncio
    async def test_disable_feature_removes_runtime_tool_registrations(self, agent):
        """Runtime feature removal clears dispatchers, direct tools, and context hides."""
        feature = _make_mock_feature("model_agent", [_make_mock_tool("list_models")])
        feature.name = "ModelAgent"
        feature.on_disable = AsyncMock()
        feature.shutdown = AsyncMock()
        feature.get_hooks.return_value = []
        feature.get_agent_card.return_value.name = "ModelAgent"
        agent.features = {"ModelAgent": feature}
        agent.task_manager = MagicMock()
        agent._register_explored_feature_tools(feature)
        agent._tool_context_hidden_features = {"model_agent"}
        agent._tool_context_hidden_tools = {"list_models"}

        await agent._disable_feature("model_agent")

        assert "ModelAgent" not in agent.features
        assert "model_agent" not in agent._explored_features
        assert "list_models" not in agent._direct_tools
        assert "list_models" not in agent._tool_to_feature
        assert agent._direct_tool_defs == []
        assert agent._tool_context_hidden_features == set()
        assert agent._tool_context_hidden_tools == set()
        feature.shutdown.assert_awaited_once()
        agent.task_manager.unregister_agent.assert_called_once_with("ModelAgent")


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

    def test_eviction_at_capacity(self, agent):
        """Oldest feature's tools evicted when over MAX_DIRECT_TOOLS."""
        agent.MAX_DIRECT_TOOLS = 5  # Low cap for testing

        # Register feature A with 3 tools
        tools_a = [_make_mock_tool(f"tool_a_{i}") for i in range(3)]
        feature_a = _make_mock_feature("feature_a", tools_a)
        agent._register_explored_feature_tools(feature_a)
        assert len(agent._direct_tools) == 3

        # Register feature B with 3 tools -> total 6, exceeds cap of 5
        tools_b = [_make_mock_tool(f"tool_b_{i}") for i in range(3)]
        feature_b = _make_mock_feature("feature_b", tools_b)
        agent._register_explored_feature_tools(feature_b)

        # Feature A (oldest) should be evicted
        assert "feature_a" not in agent._explored_features
        assert all(f"tool_a_{i}" not in agent._direct_tools for i in range(3))
        # Feature B tools remain
        assert "feature_b" in agent._explored_features
        assert all(f"tool_b_{i}" in agent._direct_tools for i in range(3))
        assert len(agent._direct_tools) == 3

    def test_eviction_removes_tool_defs(self, agent):
        """Evicted tools are also removed from _direct_tool_defs."""
        agent.MAX_DIRECT_TOOLS = 3

        tools_a = [_make_mock_tool("old_tool_1"), _make_mock_tool("old_tool_2")]
        feature_a = _make_mock_feature("old_feature", tools_a)
        agent._register_explored_feature_tools(feature_a)

        tools_b = [_make_mock_tool("new_tool_1"), _make_mock_tool("new_tool_2")]
        feature_b = _make_mock_feature("new_feature", tools_b)
        agent._register_explored_feature_tools(feature_b)

        def_names = [d["function"]["name"] for d in agent._direct_tool_defs]
        assert "old_tool_1" not in def_names
        assert "old_tool_2" not in def_names
        assert "new_tool_1" in def_names
        assert "new_tool_2" in def_names

    def test_no_eviction_under_capacity(self, agent):
        """No eviction when total tools are under the cap."""
        agent.MAX_DIRECT_TOOLS = 60

        tools = [_make_mock_tool(f"tool_{i}") for i in range(5)]
        feature = _make_mock_feature("small_feature", tools)
        agent._register_explored_feature_tools(feature)

        assert len(agent._direct_tools) == 5
        assert "small_feature" in agent._explored_features


# =============================================================================
# Feature dispatch still works alongside direct tools
# =============================================================================

class TestFeatureDispatchUnaffected:

    def test_feature_dispatch_removed_after_exploration(self, agent):
        """Feature dispatch tool is removed when direct tools are registered."""
        feature = _make_mock_feature("model_agent", [_make_mock_tool("list_models")])
        agent.features = {"ModelAgent": feature}

        # Before exploration — dispatch tool present
        tools_before = agent._build_feature_tools()
        assert len(tools_before) == 1
        assert tools_before[0]["function"]["name"] == "model_agent"

        # After exploration — dispatch tool removed (direct tools replace it)
        agent._register_explored_feature_tools(feature)
        tools_after = agent._build_feature_tools()
        assert len(tools_after) == 0


# =============================================================================
# #1580 (D) — pin tier + evicted-name logging
# =============================================================================

class TestPinTier:

    def test_pinned_feature_survives_eviction(self, agent):
        """A pinned (startup-promoted) feature's tools must NOT be
        evicted when the cap is exceeded — even when it's the oldest.
        Otherwise long sessions silently drop operationally-critical
        tools like get_peer_task_result / save_item."""
        agent.MAX_DIRECT_TOOLS = 5

        # Pinned feature with 3 tools, registered first (would
        # normally be LRU-oldest).
        pinned_tools = [_make_mock_tool(f"pinned_{i}") for i in range(3)]
        pinned = _make_mock_feature("pinned_feature", pinned_tools)
        agent._register_explored_feature_tools(pinned)
        agent._pinned_features.add("pinned_feature")

        # Then an unpinned feature with 3 tools → total 6 > cap 5.
        # The pinned one would have been oldest, but must NOT be
        # evicted.
        evictable_tools = [_make_mock_tool(f"evict_{i}") for i in range(3)]
        evictable = _make_mock_feature("evictable_feature", evictable_tools)
        agent._register_explored_feature_tools(evictable)

        # The unpinned feature must have been the one evicted (it
        # was the only candidate); the pinned one survives.
        assert "pinned_feature" in agent._explored_features
        assert all(f"pinned_{i}" in agent._direct_tools for i in range(3))

    def test_all_pinned_logs_warning_and_stops(self, agent, caplog):
        """If every explored feature is pinned and the cap is
        exceeded, eviction is impossible. The loop must bail with a
        warning rather than spin forever or evict a pinned feature.

        Build the over-cap state directly (rather than via
        sequential register+pin, which would evict the second feature
        before its pin lands) to isolate the all-pinned eviction
        contract."""
        import logging
        agent.MAX_DIRECT_TOOLS = 2

        # Stuff state with two pinned features, total 4 tools.
        for tag in ("a", "b"):
            for i in range(2):
                name = f"{tag}_{i}"
                agent._direct_tools[name] = MagicMock()
                agent._direct_tool_defs.append({
                    "type": "function",
                    "function": {"name": name, "description": "",
                                 "parameters": {"type": "object"}},
                })
                agent._tool_to_feature[name] = f"{tag}_feature"
            agent._explored_features[f"{tag}_feature"] = True
            agent._pinned_features.add(f"{tag}_feature")

        assert len(agent._direct_tools) == 4  # over cap 2

        with caplog.at_level(logging.WARNING):
            agent._maybe_evict_direct_tools()

        # Nothing was evicted — both pinned features survived.
        assert "a_feature" in agent._explored_features
        assert "b_feature" in agent._explored_features
        # Warning surfaced naming the pinned features.
        assert any(
            "all" in r.message.lower() and "pinned" in r.message.lower()
            for r in caplog.records
        )

    def test_eviction_log_includes_tool_names(self, agent, caplog):
        """#1580 (D) acceptance: eviction must log the actual tool
        names, not just a count, so regressions are visible in audit."""
        import logging
        agent.MAX_DIRECT_TOOLS = 2

        tools_old = [_make_mock_tool("old_named_tool")]
        agent._register_explored_feature_tools(
            _make_mock_feature("old_feat", tools_old)
        )
        tools_new = [
            _make_mock_tool(f"new_tool_{i}") for i in range(2)
        ]

        with caplog.at_level(logging.INFO):
            agent._register_explored_feature_tools(
                _make_mock_feature("new_feat", tools_new)
            )

        evict_logs = [
            r.message for r in caplog.records
            if "Evicted" in r.message and "old_feat" in r.message
        ]
        assert evict_logs, "eviction was not logged"
        assert "old_named_tool" in evict_logs[0], (
            "evicted tool name must appear in the log line (not just count)"
        )


# =============================================================================
# #1577 (A) — end-to-end: build_all_tools → codex per-turn handler
# =============================================================================

class TestAllToolsReachCodexHandler:
    """Pin the contract: tool names returned by ``_build_all_tools()``
    end up in the codex adapter's per-turn ``allowed_tools`` set, so
    a tool the agent advertised is actually callable for that turn.

    Without this pin, the "Kestrel did not register an
    item/tool/call handler for this turn." failure Emma hit could
    regress silently — the RPC layer and the registry layer are each
    tested in isolation but no test wires the seam."""

    @pytest.mark.asyncio
    async def test_advertised_tools_reach_handler_allowed_set(self):
        """An agent's ``_build_all_tools()`` output, when threaded
        through the codex adapter via ``_run_turn``, must produce a
        handler whose ``allowed_tools`` includes every advertised
        tool name."""
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter
        from tests.unit.test_codex_adapter import (
            _FakeAppServer, _TEXT_TURN,
        )

        adapter = CodexAdapter()
        adapter._client = _FakeAppServer(list(_TEXT_TURN))

        # Build a fake "advertised" tool list — the shape
        # ``_build_all_tools`` returns: OpenAI function-tool envelopes.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"do {name}",
                    "parameters": {"type": "object"},
                },
            }
            for name in (
                "save_item", "strategy_add_decision",
                "get_peer_task_result",
            )
        ]

        async def exe(name, args):
            return {"success": True, "result": "x"}

        captured = {}
        orig_register = adapter._make_tool_call_handler

        def _spy_make_handler(executor, thread_id, allowed_tools, executed_log=None):
            captured["allowed_tools"] = set(allowed_tools)
            return orig_register(executor, thread_id, allowed_tools, executed_log)

        adapter._make_tool_call_handler = _spy_make_handler

        await adapter.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            session_id="s",
            tool_executor=exe,
        )

        assert captured["allowed_tools"] == {
            "save_item", "strategy_add_decision", "get_peer_task_result",
        }, "every advertised tool must reach the per-turn allowed set"

    @pytest.mark.asyncio
    async def test_text_only_turn_registers_no_handler(self):
        """Negative pin: a turn with ``tools=None`` must NOT register
        an item/tool/call handler — that's the defense-in-depth path
        Emma's failure surfaced (a stale tool call on a text-only
        turn would otherwise execute against the orchestrator's full
        registry)."""
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter
        from tests.unit.test_codex_adapter import (
            _FakeAppServer, _TEXT_TURN,
        )

        adapter = CodexAdapter()
        adapter._client = _FakeAppServer(list(_TEXT_TURN))

        async def exe(name, args):
            return {"success": True, "result": "x"}

        await adapter.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "hi"}],
            session_id="text-only",
            tool_executor=exe,
        )
        # Append-only history — the bridge handlers may register
        # globally (#1575) but item/tool/call must NOT, because no
        # tools were advertised this turn.
        registered_tool_call = [
            k for k in adapter._client.registered_history
            if k[0] == "item/tool/call"
        ]
        assert registered_tool_call == [], (
            "text-only turn must not register an item/tool/call handler"
        )
