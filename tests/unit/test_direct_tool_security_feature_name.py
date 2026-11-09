"""Direct-tool dispatch must use the PascalCase feature name for security
lookups (#1427).

``_tool_to_feature`` stores ``feature.tool_name`` (snake_case) so the
feature_features inventory can map back to a feature object. The security
permission store, however, keys on ``feature.name`` (PascalCase, =
``type(feature).__name__``) — every other call site already uses that form.

Without this translation, every direct-tool call (e.g.
``respond_to_a2a_task``) bypassed the existing ``set_permission("TaskFeature",
"respond_to_a2a_task", ALLOW)`` row and defaulted to ASK, queueing approvals
that no user was watching for in signal-driven cognition turns.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.result import ToolResult
from tests.utils.native_tool_effects import drive_native
from tests.unit.test_tool_concurrency import FakeToolCall

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin


class _Feature:
    """Minimal stand-in for a Kestrel ``Feature`` carrying both names."""

    def __init__(self, klass_name: str, snake_name: str):
        self._klass_name = klass_name
        self.tool_name = snake_name

    @property
    def name(self) -> str:
        return self._klass_name


def _bind_helper(agent) -> str:
    """Return the bound helper method for tests to call directly."""
    agent._feature_for_tool_owner = ToolRegistryMixin._feature_for_tool_owner.__get__(agent)
    return OrchestratorEngineMixin._security_feature_name_for_tool.__get__(agent)


class TestSecurityFeatureNameForTool:
    def test_direct_tool_resolves_to_pascalcase_feature_name(self):
        agent = MagicMock()
        agent._tool_to_feature = {"respond_to_a2a_task": "task_feature"}
        agent.features = {
            "TaskFeature": _Feature("TaskFeature", "task_feature"),
        }
        helper = _bind_helper(agent)

        assert helper("respond_to_a2a_task") == "TaskFeature"

    def test_unknown_tool_falls_through(self):
        agent = MagicMock()
        agent._tool_to_feature = {}
        agent.features = {}
        helper = _bind_helper(agent)

        assert helper("some_unmapped_tool") == "some_unmapped_tool"

    def test_tool_without_matching_feature_falls_back_to_snake(self):
        """If the snake-case mapping exists but no feature with that
        ``tool_name`` is loaded any more (e.g. evicted), return the
        snake-case form rather than raising or returning the tool name —
        callers downstream can still log it for diagnosis."""
        agent = MagicMock()
        agent._tool_to_feature = {"orphaned_tool": "ghost_feature"}
        agent.features = {}
        helper = _bind_helper(agent)

        assert helper("orphaned_tool") == "ghost_feature"

    def test_multiple_features_picks_the_matching_one(self):
        agent = MagicMock()
        agent._tool_to_feature = {"send_a2a_question": "peers_feature"}
        agent.features = {
            "TaskFeature": _Feature("TaskFeature", "task_feature"),
            "PeersFeature": _Feature("PeersFeature", "peers_feature"),
            "MemoryFeature": _Feature("MemoryFeature", "memory_feature"),
        }
        helper = _bind_helper(agent)

        assert helper("send_a2a_question") == "PeersFeature"


@pytest.mark.asyncio
class TestDirectToolDispatchUsesPascalcaseFeatureName:
    async def test_dispatch_threads_pascalcase_to_hooks(self, native_tool_effect_agent, monkeypatch):
        """Real registered tool, journal and hooks retain the canonical owner name."""
        observed = []

        class TaskFeature(Feature):
            @property
            def tool_description(self):
                return "Owned direct security-name probe"

            async def initialize(self):
                pass

            @tool(name="respond_to_a2a_task", description="Record one owned response")
            async def respond_to_a2a_task(self, task_id: str, content: str, state: str) -> ToolResult:
                observed.append((task_id, content, state))
                return ToolResult.ok(confirmation="Recorded the response")

        agent = await native_tool_effect_agent(did="did:test:security-name")
        feature = TaskFeature(agent)
        agent.features[feature.name] = feature
        selected, = feature.get_tools()
        agent._direct_tools[selected.name] = selected
        agent._tool_to_feature[selected.name] = feature.tool_name
        monkeypatch.setattr(agent, "_execute_tool_with_hooks",
            AsyncMock(wraps=agent._execute_tool_with_hooks))
        args = {"task_id": "abc", "content": "ok", "state": "completed"}
        event = await agent.observability_store.log_tool_call(
            agent_name=agent.did, tool_name=selected.name, session_id="any-session")
        await drive_native(agent, lambda: agent._dispatch_direct_tool(
            tool_call=FakeToolCall("owned-call", selected.name, args),
            tool_name=selected.name, args=args, dispatch_start=0.0,
            dispatch_event_id=event, session_id="any-session",
        ))
        agent._execute_tool_with_hooks.assert_awaited_once()
        assert agent._execute_tool_with_hooks.await_args.kwargs["feature_name"] == "TaskFeature"
        assert observed == [("abc", "ok", "completed")]
        assert await agent._raw_storage.db.fetchone("SELECT state FROM tool_effect_claims") == ("RETURNED_OK",)
