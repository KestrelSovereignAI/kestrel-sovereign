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
    async def test_dispatch_threads_pascalcase_to_hooks(self):
        """End-to-end on the dispatch path: ``_execute_tool_with_hooks`` must
        receive the PascalCase ``feature_name`` so the security hook's
        ``permission_store.get_permission("TaskFeature", ...)`` lookup hits
        the historical row."""

        agent = MagicMock()
        agent._direct_tools = {}
        agent._tool_to_feature = {"respond_to_a2a_task": "task_feature"}
        agent.features = {
            "TaskFeature": _Feature("TaskFeature", "task_feature"),
        }
        agent._execute_tool_with_hooks = AsyncMock(return_value={"result": "ok"})
        agent.observability_store = MagicMock()
        agent.observability_store.log_tool_response = AsyncMock()

        tool = MagicMock()
        tool.execute = AsyncMock(return_value={"result": "ok"})
        agent._direct_tools["respond_to_a2a_task"] = tool

        tool_call = MagicMock()
        tool_call.name = "respond_to_a2a_task"

        # Bind the real helper too — otherwise the MagicMock auto-attribute
        # returns a MagicMock instead of the PascalCase string, masking the
        # fix this test is here to enforce.
        agent._security_feature_name_for_tool = (
            OrchestratorEngineMixin._security_feature_name_for_tool.__get__(agent)
        )
        dispatch = OrchestratorEngineMixin._dispatch_direct_tool.__get__(agent)
        await dispatch(
            tool_call=tool_call,
            tool_name="respond_to_a2a_task",
            args={"task_id": "abc", "content": "ok", "state": "completed"},
            dispatch_start=0.0,
            dispatch_event_id="evt_1427",
            session_id="any-session",
        )

        agent._execute_tool_with_hooks.assert_awaited_once()
        kwargs = agent._execute_tool_with_hooks.await_args.kwargs
        assert kwargs["feature_name"] == "TaskFeature", (
            f"Expected feature_name='TaskFeature' (PascalCase, matching "
            f"the historical permission DB rows), got "
            f"{kwargs['feature_name']!r}. See #1427 — when this regresses, "
            "signal-driven cognition turns queue approvals indefinitely."
        )
