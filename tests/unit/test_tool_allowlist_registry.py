"""Regression tests for the tool allowlist (#2929).

The orchestrator's guardrail allowlist used to be the set of tool names
currently *advertised* to the LLM. That view shrinks mid-turn — exploration
replaces a feature's dispatcher with its individual tools, LRU eviction sheds
direct tools, a context profile hides them — so a name that dispatched
successfully one call ago got bounced on the next as "unknown", *before* the
permission layer and therefore with no security-audit row at all.

These tests pin the two contracts that close it:

* allowlist membership is DERIVED from the live feature registry, so a loaded
  feature's tool is known by construction (and stays known across exploration,
  eviction, and context hiding);
* every rejection writes a ``security_audit_log`` entry.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.hooks.base import HookOutput
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.security.tool_audit import (
    ACTION_TOOL_RESOLUTION,
    ACTION_TOOL_VALIDATION,
    REJECTION_DECISION,
    REJECTION_FEATURE_NAME,
    record_tool_rejection,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_tool(name: str):
    tool = MagicMock()
    tool.name = name
    tool.schema = ToolSchema(
        name=name,
        description=f"{name} tool",
        category=ToolCategory.SYSTEM,
        parameters=[
            ToolParameter(name="query", type="string", description="p"),
        ],
    )
    tool.execute = AsyncMock(return_value={"success": True, "tool": name})
    return tool


def _make_feature(tool_name: str, tools: list, *, name: str | None = None):
    feature = MagicMock()
    feature.tool_name = tool_name
    feature.name = name or "".join(p.title() for p in tool_name.split("_"))
    feature.enabled = True
    feature.tool_description = f"Mock feature: {tool_name}"
    feature.get_tools.return_value = tools
    feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Dispatch to {tool_name}",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string", "description": "d"}},
                "required": ["task"],
            },
        },
    }
    feature.execute_as_subagent = AsyncMock(
        return_value={"success": True, "result": f"{tool_name} ran"}
    )
    return feature


class _RecordingPermissionStore:
    """Minimal PermissionStore stand-in capturing ``log_decision`` rows."""

    def __init__(self):
        self.rows = []

    async def log_decision(self, **kwargs):
        self.rows.append(kwargs)


def _attach_security_feature(agent):
    store = _RecordingPermissionStore()
    security = MagicMock()
    security.permission_store = store
    agent.features["SecurityFeature"] = security
    return store


@pytest.fixture
def agent():
    with patch("kestrel_sovereign.kestrel_agent.LLMService"):
        a = KestrelAgent(did="did:test:allowlist")
    a.features = {}
    return a


@pytest.fixture
def dispatch_agent(agent):
    """An agent wired for ``_dispatch_tool_call`` (hooks allow everything)."""
    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_error = AsyncMock()
    agent.observability_store.log_tool_dispatch = AsyncMock(return_value="d-1")
    agent.hooks_manager = MagicMock()
    agent.hooks_manager.execute_hooks = AsyncMock(return_value=HookOutput())
    agent.hooks_manager.execute_hooks_parallel = AsyncMock(return_value=None)
    return agent


def _tool_call(name: str, arguments: dict | None = None):
    call = MagicMock()
    call.id = "tc-1"
    call.name = name
    call.arguments = arguments if arguments is not None else {"query": "x"}
    return call


# =============================================================================
# The allowlist is derived from the live registry
# =============================================================================

class TestAllowlistDerivation:
    def test_unpromoted_feature_tools_are_known(self, agent):
        """A loaded feature's @tool is known before it is ever promoted."""
        feature = _make_feature(
            "github", [_make_tool("get_github_issue"), _make_tool("create_github_issue")]
        )
        agent.features = {"GitHubFeature": feature}

        advertised = {t["function"]["name"] for t in agent._build_all_tools()}
        known = agent._known_tool_names()

        # Only the dispatcher is advertised...
        assert advertised == {"github"}
        # ...but every registered name is known.
        assert {"github", "get_github_issue", "create_github_issue"} <= known

    def test_explored_feature_dispatcher_stays_known(self, agent):
        """The mid-turn bounce: exploration drops the dispatcher from the
        advertised view, and the same name must not become "unknown"."""
        feature = _make_feature("github", [_make_tool("get_github_issue")])
        agent.features = {"GitHubFeature": feature}

        assert "github" in agent._known_tool_names()

        # First successful dispatch promotes the feature's tools.
        agent._register_explored_feature_tools(feature)

        advertised = {t["function"]["name"] for t in agent._build_all_tools()}
        assert "github" not in advertised  # dispatcher no longer advertised
        assert "github" in agent._known_tool_names()  # ...but still known

    def test_context_hidden_tools_stay_known(self, agent):
        """Progressive disclosure is a prompt-budget decision, not a
        capability boundary — hidden is not unknown."""
        feature = _make_feature(
            "model_agent", [_make_tool("list_models"), _make_tool("get_current_model")]
        )
        agent.features = {"ModelAgent": feature}
        agent._register_explored_feature_tools(feature)
        agent._tool_context_hidden_tools = {"get_current_model"}
        agent._tool_context_hidden_features = {"model_agent"}

        known = agent._known_tool_names()

        assert "get_current_model" in known
        assert "list_models" in known
        assert "model_agent" in known

    def test_evicted_direct_tools_stay_known(self, agent):
        """LRU eviction unmounts direct tools; the owning feature still
        exposes them, so they remain known."""
        feature = _make_feature("model_agent", [_make_tool("list_models")])
        agent.features = {"ModelAgent": feature}
        agent._register_explored_feature_tools(feature)
        assert "list_models" in agent._direct_tools

        agent.unregister_dynamic_tools("model_agent")

        assert "list_models" not in agent._direct_tools
        assert "list_models" in agent._known_tool_names()

    def test_dynamically_mounted_tools_are_known(self, agent):
        """Tools mounted by a non-feature owner (MCP) belong to no feature's
        ``get_tools()`` but are still registered."""
        agent.register_dynamic_tools("mcp:files", [_make_tool("mcp__files__read")])

        assert "mcp__files__read" in agent._known_tool_names()

    def test_disabled_feature_tools_are_not_known(self, agent):
        """A soft-disabled feature IS a capability boundary (#2522)."""
        feature = _make_feature("github", [_make_tool("get_github_issue")])
        feature.enabled = False
        agent.features = {"GitHubFeature": feature}

        known = agent._known_tool_names()

        assert "github" not in known
        assert "get_github_issue" not in known

    def test_allowlist_stays_consistent_with_the_live_registry(self, agent):
        """The audit: membership is exactly what the live registry exposes.

        Recomputed independently here from ``agent.features`` so a future
        hand-written addition to the allowlist fails this test.
        """
        github = _make_feature(
            "github", [_make_tool("get_github_issue"), _make_tool("create_github_issue")]
        )
        talon = _make_feature(
            "talon_coordinator_feature", [_make_tool("talon_status")]
        )
        agent.features = {"GitHubFeature": github, "TalonCoordinatorFeature": talon}
        agent._register_explored_feature_tools(github)
        agent._tool_context_hidden_features = {"talon_coordinator_feature"}

        expected = set(agent._direct_tools)
        for feature in agent.features.values():
            expected.add(feature.tool_name)
            expected.update(t.name for t in feature.get_tools())

        assert agent._known_tool_names() == expected
        # The advertised view is a strict subset — never the allowlist itself.
        assert agent._visible_known_tool_names() < agent._known_tool_names()


# =============================================================================
# Registered tools dispatch instead of being reported unknown
# =============================================================================

class TestRegisteredToolDispatch:
    @pytest.mark.asyncio
    async def test_unpromoted_feature_tool_dispatches(self, dispatch_agent):
        """``create_github_issue`` before exploration: registered, so it runs."""
        tool = _make_tool("create_github_issue")
        feature = _make_feature("github", [tool], name="GitHubFeature")
        dispatch_agent.features = {"GitHubFeature": feature}
        messages = []

        result = await dispatch_agent._dispatch_tool_call(
            _tool_call("create_github_issue", {"title": "t"}),
            {},  # nothing visible this turn
            {"create_github_issue"},
            messages,
            0,
            None,
        )

        assert result == {"success": True, "tool": "create_github_issue"}
        tool.execute.assert_awaited_once_with(title="t")
        # The permission lookup names the OWNING feature, not the tool.
        pre_call = dispatch_agent.hooks_manager.execute_hooks.await_args_list[0]
        assert pre_call.args[1].feature_name == "GitHubFeature"

    @pytest.mark.asyncio
    async def test_context_hidden_feature_dispatcher_dispatches(self, dispatch_agent):
        """A hidden dispatcher is absent from the visible map but still live."""
        feature = _make_feature("github", [_make_tool("get_github_issue")])
        dispatch_agent.features = {"GitHubFeature": feature}
        dispatch_agent._tool_context_hidden_features = {"github"}
        messages = []

        result = await dispatch_agent._dispatch_tool_call(
            _tool_call("github", {"task": "read #1"}),
            dispatch_agent._visible_features_by_tool_name(),
            dispatch_agent._known_tool_names(),
            messages,
            0,
            None,
        )

        assert result == {"success": True, "result": "github ran"}
        feature.execute_as_subagent.assert_awaited_once()


# =============================================================================
# Every rejection is audited
# =============================================================================

class TestRejectionAudit:
    @pytest.mark.asyncio
    async def test_allowlist_rejection_writes_audit_entry(self, dispatch_agent):
        store = _attach_security_feature(dispatch_agent)
        messages = []

        await dispatch_agent._dispatch_tool_call(
            _tool_call("ghost_tool", {"query": "x"}),
            {},
            {"some_other_tool"},
            messages,
            0,
            None,
        )

        assert len(store.rows) == 1
        row = store.rows[0]
        assert row["tool_name"] == "ghost_tool"
        assert row["feature_name"] == REJECTION_FEATURE_NAME
        assert row["action"] == ACTION_TOOL_VALIDATION
        assert row["decision"] == REJECTION_DECISION
        assert "not in the known tool allowlist" in row["args_summary"]

    @pytest.mark.asyncio
    async def test_unresolvable_tool_writes_audit_entry(self, dispatch_agent):
        """Passing the allowlist but resolving to nothing is also audited."""
        store = _attach_security_feature(dispatch_agent)
        messages = []

        await dispatch_agent._dispatch_tool_call(
            _tool_call("ghost_tool", {"query": "x"}),
            {},
            {"ghost_tool"},  # allowlisted, but no feature exposes it
            messages,
            0,
            None,
        )

        assert len(store.rows) == 1
        assert store.rows[0]["action"] == ACTION_TOOL_RESOLUTION
        assert store.rows[0]["decision"] == REJECTION_DECISION

    @pytest.mark.asyncio
    async def test_audit_masks_sensitive_arguments(self):
        agent = MagicMock()
        store = _RecordingPermissionStore()
        security = MagicMock()
        security.permission_store = store
        agent.features = {"SecurityFeature": security}

        wrote = await record_tool_rejection(
            agent,
            tool_name="ghost_tool",
            reason="nope",
            args={"api_key": "sk-live-secret", "repo": "owner/repo"},
        )

        assert wrote is True
        summary = store.rows[0]["args_summary"]
        assert "sk-live-secret" not in summary
        assert "owner/repo" in summary

    @pytest.mark.asyncio
    async def test_audit_is_best_effort_without_a_security_feature(self, caplog):
        agent = MagicMock()
        agent.features = {}

        wrote = await record_tool_rejection(
            agent, tool_name="ghost_tool", reason="nope",
        )

        assert wrote is False
        assert "ghost_tool" in caplog.text

    @pytest.mark.asyncio
    async def test_audit_failure_never_raises(self):
        agent = MagicMock()
        security = MagicMock()
        security.permission_store = MagicMock()
        security.permission_store.log_decision = AsyncMock(
            side_effect=RuntimeError("db locked")
        )
        agent.features = {"SecurityFeature": security}

        assert await record_tool_rejection(
            agent, tool_name="ghost_tool", reason="nope",
        ) is False
