"""
Tests for the ObservabilityHook and ObservabilityFeature.

Covers:
1. Hook registers on all events
2. Hook always returns ALLOW (never blocks)
3. Hook logs to ObservabilityStore on each event type
4. Hook handles missing ObservabilityStore gracefully
5. Hook handles exceptions without propagating
6. Feature registers hook during initialize()
7. Privacy: user_message content is NOT logged, only length
8. Privacy: tool_response error truncated to 200 chars
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.hooks.base import (
    HookEvent,
    HookInput,
    HookOutput,
    PermissionDecision,
)
from kestrel_sovereign.features.observability.hook import ObservabilityHook
from kestrel_sovereign.features.observability.feature import ObservabilityFeature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(with_store=True, with_hooks_manager=True):
    """Create a mock agent with optional observability_store and hooks_manager."""
    agent = MagicMock()
    agent.agent_name = "test-agent"

    if with_store:
        store = AsyncMock()
        store.log_metric = AsyncMock(return_value="event-id-123")
        store.query_events = AsyncMock(return_value=[])
        agent.observability_store = store
    else:
        agent.observability_store = None

    if with_hooks_manager:
        manager = MagicMock()
        manager.register = MagicMock()
        manager.unregister = MagicMock()
        agent.hooks_manager = manager
    else:
        agent.hooks_manager = None

    return agent


def _make_input(event_name="PreToolUse", **overrides):
    """Create a HookInput for testing."""
    defaults = {
        "session_id": "sess-1",
        "hook_event_name": event_name,
    }
    defaults.update(overrides)
    return HookInput(**defaults)


# ---------------------------------------------------------------------------
# 1. Hook registers on all events
# ---------------------------------------------------------------------------

class TestHookRegistration:
    def test_registers_on_all_hook_events(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        assert set(hook.events) == set(HookEvent)

    def test_priority_is_999(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        assert hook.priority == 999

    def test_name_is_observability(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        assert hook.name == "observability"

    def test_timeout_is_5_seconds(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        assert hook.timeout == 5.0


# ---------------------------------------------------------------------------
# 2. Hook always returns ALLOW
# ---------------------------------------------------------------------------

class TestHookAlwaysAllows:
    @pytest.mark.asyncio
    async def test_returns_allow_on_pre_tool_use(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("PreToolUse", tool_name="some_tool"))
        assert result.continue_execution is True
        assert result.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_returns_allow_on_session_start(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("SessionStart"))
        assert result.continue_execution is True
        assert result.permission_decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_returns_allow_on_stop(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("Stop"))
        assert result.continue_execution is True

    @pytest.mark.asyncio
    async def test_returns_allow_when_store_missing(self):
        agent = _make_agent(with_store=False)
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("PreToolUse"))
        assert result.continue_execution is True

    @pytest.mark.asyncio
    async def test_returns_allow_when_store_raises(self):
        agent = _make_agent()
        agent.observability_store.log_metric.side_effect = RuntimeError("DB down")
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("PreToolUse"))
        assert result.continue_execution is True
        assert result.permission_decision == PermissionDecision.ALLOW


# ---------------------------------------------------------------------------
# 3. Hook logs to ObservabilityStore on each event type
# ---------------------------------------------------------------------------

class TestHookLogsEvents:
    @pytest.mark.asyncio
    async def test_logs_metric_for_pre_tool_use(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        await hook.execute(_make_input("PreToolUse", tool_name="web_search"))

        agent.observability_store.log_metric.assert_called_once()
        call_kwargs = agent.observability_store.log_metric.call_args
        assert call_kwargs.kwargs["agent_name"] == "test-agent"
        assert call_kwargs.kwargs["metric_name"] == "hook.PreToolUse"
        assert call_kwargs.kwargs["metric_value"] == 1
        assert call_kwargs.kwargs["metadata"]["tool_name"] == "web_search"

    @pytest.mark.asyncio
    async def test_logs_metric_for_session_start(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        await hook.execute(_make_input("SessionStart"))

        call_kwargs = agent.observability_store.log_metric.call_args
        assert call_kwargs.kwargs["metric_name"] == "hook.SessionStart"

    @pytest.mark.asyncio
    async def test_logs_metric_for_user_prompt_submit(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        await hook.execute(_make_input("UserPromptSubmit", user_message="hello world"))

        call_kwargs = agent.observability_store.log_metric.call_args
        assert call_kwargs.kwargs["metric_name"] == "hook.UserPromptSubmit"
        assert call_kwargs.kwargs["metadata"]["user_message_length"] == 11

    @pytest.mark.asyncio
    async def test_logs_all_seven_event_types(self):
        """Verify each HookEvent value can be logged without error."""
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)

        for event in HookEvent:
            agent.observability_store.log_metric.reset_mock()
            inp = _make_input(event.value)
            result = await hook.execute(inp)
            assert result.continue_execution is True
            agent.observability_store.log_metric.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_execution_time_ms(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        await hook.execute(_make_input("PostToolUse", tool_name="test", execution_time_ms=42))

        metadata = agent.observability_store.log_metric.call_args.kwargs["metadata"]
        assert metadata["execution_time_ms"] == 42

    @pytest.mark.asyncio
    async def test_logs_feature_name(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        await hook.execute(_make_input("PreToolUse", tool_name="t", feature_name="SecurityFeature"))

        metadata = agent.observability_store.log_metric.call_args.kwargs["metadata"]
        assert metadata["feature_name"] == "SecurityFeature"

    @pytest.mark.asyncio
    async def test_logs_tool_response_success(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        await hook.execute(
            _make_input(
                "PostToolUse",
                tool_name="t",
                tool_response={"success": True, "result": "ok"},
            )
        )

        metadata = agent.observability_store.log_metric.call_args.kwargs["metadata"]
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_logs_tool_response_failure(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        await hook.execute(
            _make_input(
                "PostToolUse",
                tool_name="t",
                tool_response={"success": False, "error": "something broke"},
            )
        )

        metadata = agent.observability_store.log_metric.call_args.kwargs["metadata"]
        assert metadata["success"] is False
        assert metadata["error"] == "something broke"


# ---------------------------------------------------------------------------
# 4. Hook handles missing ObservabilityStore gracefully
# ---------------------------------------------------------------------------

class TestHookMissingStore:
    @pytest.mark.asyncio
    async def test_no_store_attribute(self):
        agent = MagicMock(spec=[])  # No attributes at all
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("PreToolUse"))
        assert result.continue_execution is True

    @pytest.mark.asyncio
    async def test_store_is_none(self):
        agent = _make_agent(with_store=False)
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("Stop"))
        assert result.continue_execution is True


# ---------------------------------------------------------------------------
# 5. Hook handles exceptions without propagating
# ---------------------------------------------------------------------------

class TestHookExceptionHandling:
    @pytest.mark.asyncio
    async def test_store_log_metric_raises(self):
        agent = _make_agent()
        agent.observability_store.log_metric.side_effect = Exception("DB error")
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("PreToolUse", tool_name="test"))
        assert result.continue_execution is True

    @pytest.mark.asyncio
    async def test_agent_name_missing(self):
        agent = _make_agent()
        del agent.agent_name
        hook = ObservabilityHook(agent=agent)
        result = await hook.execute(_make_input("PreToolUse"))
        assert result.continue_execution is True


# ---------------------------------------------------------------------------
# 6. Feature registers hook during initialize()
# ---------------------------------------------------------------------------

class TestFeatureInitialization:
    @pytest.mark.asyncio
    async def test_feature_provides_hook_via_get_hooks(self):
        agent = _make_agent()
        feature = ObservabilityFeature(agent)
        await feature.initialize()

        hooks = feature.get_hooks()
        assert len(hooks) == 1
        assert isinstance(hooks[0], ObservabilityHook)
        assert hooks[0].name == "observability"
        assert hooks[0].priority == 999

    @pytest.mark.asyncio
    async def test_feature_handles_no_hooks_manager(self):
        agent = _make_agent(with_hooks_manager=False)
        feature = ObservabilityFeature(agent)
        # Should not raise
        await feature.initialize()

    @pytest.mark.asyncio
    async def test_feature_clears_hook_on_shutdown(self):
        agent = _make_agent()
        feature = ObservabilityFeature(agent)
        await feature.initialize()
        await feature.shutdown()

        # After shutdown, get_hooks() returns empty (hook reference cleared)
        assert feature.get_hooks() == []

    def test_feature_tool_description(self):
        agent = _make_agent()
        feature = ObservabilityFeature(agent)
        assert "observability" in feature.tool_description.lower()

    @pytest.mark.asyncio
    async def test_feature_has_obs_status_tool(self):
        agent = _make_agent()
        feature = ObservabilityFeature(agent)
        tool_names = [t.name for t in feature.get_tools()]
        assert "obs_status" in tool_names

    @pytest.mark.asyncio
    async def test_feature_has_obs_events_tool(self):
        agent = _make_agent()
        feature = ObservabilityFeature(agent)
        tool_names = [t.name for t in feature.get_tools()]
        assert "obs_events" in tool_names


# ---------------------------------------------------------------------------
# 7. Privacy: user_message content NOT logged, only length
# ---------------------------------------------------------------------------

class TestPrivacy:
    @pytest.mark.asyncio
    async def test_user_message_not_in_metadata(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        secret_message = "my password is hunter2"
        await hook.execute(
            _make_input("UserPromptSubmit", user_message=secret_message)
        )

        metadata = agent.observability_store.log_metric.call_args.kwargs["metadata"]
        # Length is logged
        assert metadata["user_message_length"] == len(secret_message)
        # Actual content must not be in any metadata value
        for key, value in metadata.items():
            if isinstance(value, str):
                assert secret_message not in value

    @pytest.mark.asyncio
    async def test_tool_input_not_logged(self):
        """Tool input args should not appear in metadata (privacy)."""
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        await hook.execute(
            _make_input(
                "PreToolUse",
                tool_name="web_search",
                tool_input={"query": "sensitive query", "api_key": "secret123"},
            )
        )

        metadata = agent.observability_store.log_metric.call_args.kwargs["metadata"]
        # tool_input should not be directly in metadata
        assert "tool_input" not in metadata
        # The values should not leak
        for key, value in metadata.items():
            if isinstance(value, str):
                assert "sensitive query" not in value
                assert "secret123" not in value


# ---------------------------------------------------------------------------
# 8. Error truncation
# ---------------------------------------------------------------------------

class TestErrorTruncation:
    @pytest.mark.asyncio
    async def test_long_error_truncated_to_200_chars(self):
        agent = _make_agent()
        hook = ObservabilityHook(agent=agent)
        long_error = "x" * 500
        await hook.execute(
            _make_input(
                "PostToolUse",
                tool_name="t",
                tool_response={"success": False, "error": long_error},
            )
        )

        metadata = agent.observability_store.log_metric.call_args.kwargs["metadata"]
        assert len(metadata["error"]) == 200
