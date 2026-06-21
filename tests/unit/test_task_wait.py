"""Tests for ``TaskFeature.wait`` — the generic bounded wait primitive.

#1541: agents were shelling out to ``sleep`` between polls during
autonomous work loops. ``wait`` is the native replacement: a bounded,
audited pause that enforces a conservative maximum duration and reports
the observed elapsed time.
"""

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.waits import WaitRegistry


class _StubAgent:
    def __init__(self):
        self.wait_registry = WaitRegistry()


class TestGenericWait:
    @pytest.mark.asyncio
    async def test_zero_duration_returns_immediately(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds=0, reason="probe")
        assert result.status is ToolResultStatus.OK
        assert result.data["requested_seconds"] == 0
        assert result.data["reason"] == "probe"
        assert result.data["elapsed_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_short_wait_reports_elapsed(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds=1)
        assert result.status is ToolResultStatus.OK
        # Observed elapsed should be at least the requested duration.
        assert result.data["elapsed_seconds"] >= 1
        assert result.data["requested_seconds"] == 1

    @pytest.mark.asyncio
    async def test_max_duration_rejected(self):
        feature = TaskFeature(agent=None)
        too_long = TaskFeature._MAX_WAIT_SECONDS + 1
        result = await feature.wait(duration_seconds=too_long)
        assert result.status is ToolResultStatus.ERROR
        assert "exceeds the maximum" in result.error
        assert result.data["requested_seconds"] == too_long
        assert result.data["max_seconds"] == TaskFeature._MAX_WAIT_SECONDS

    @pytest.mark.asyncio
    async def test_negative_duration_rejected(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds=-5)
        assert result.status is ToolResultStatus.ERROR
        assert "must be >= 0" in result.error

    @pytest.mark.asyncio
    async def test_non_integer_duration_rejected(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds="soon")
        assert result.status is ToolResultStatus.ERROR
        assert "must be an integer" in result.error


class TestUnifiedWaitTarget:
    """The single `wait` tool dispatches `target="<kind>:<handle>"` to the
    agent's wait_registry — the unified interface replacing per-feature
    waiters."""

    @pytest.mark.asyncio
    async def test_target_dispatches_to_registry(self, monkeypatch):
        agent = _StubAgent()
        feature = TaskFeature(agent=None)
        feature.agent = agent

        # Register a TaskFeature provider whose status read is stubbed
        # to "completed" so the engine returns OK immediately.
        await feature.post_all_features_loaded(agent)

        async def fake_status(task_id):
            return {
                "ok": True, "task_id": task_id, "status": "completed",
                "task_type": "demo", "artifacts": [], "message": "done",
            }

        monkeypatch.setattr(feature, "_get_task_status_data", fake_status)
        # The provider registered above wraps THIS feature instance, so
        # the monkeypatched method is the one the engine polls.
        result = await feature.wait(target="task:abc123", timeout_seconds=5)
        assert result.status is ToolResultStatus.OK
        assert result.data["ref"] == "task:abc123"

    @pytest.mark.asyncio
    async def test_target_without_registry_errors(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(target="task:abc", timeout_seconds=5)
        assert result.status is ToolResultStatus.ERROR
        assert "wait engine unavailable" in result.error

    @pytest.mark.asyncio
    async def test_unknown_kind_errors(self):
        agent = _StubAgent()
        feature = TaskFeature(agent=None)
        feature.agent = agent
        result = await feature.wait(target="bogus:xyz", timeout_seconds=5)
        assert result.status is ToolResultStatus.ERROR
        assert "no wait provider for kind 'bogus'" in result.error

    @pytest.mark.asyncio
    async def test_post_load_registers_task_provider(self):
        agent = _StubAgent()
        feature = TaskFeature(agent=None)
        feature.agent = agent
        await feature.post_all_features_loaded(agent)
        assert "task" in agent.wait_registry.kinds()
