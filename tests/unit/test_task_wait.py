"""Tests for ``TaskFeature.wait`` — the generic bounded wait primitive.

#1541: agents were shelling out to ``sleep`` between polls during
autonomous work loops. ``wait`` is the native replacement: a bounded,
audited pause that enforces a conservative maximum duration and reports
the observed elapsed time.
"""

from types import SimpleNamespace

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.waits import WaitRegistry


class _StubAgent:
    def __init__(self):
        self.wait_registry = WaitRegistry()


async def _make_db_agent(tmp_path):
    """A stub agent with a real sqlite DB so register_wait_watch can persist
    a watch row (the signal-mode path writes to WaitSignalStore)."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
    return SimpleNamespace(
        did="did:test:agent",
        agent_id="did:test:agent",
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=WaitRegistry(),
    )


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
    async def test_numeric_target_is_bounded_sleep(self):
        """`!wait 5` binds the token positionally to `target`; a bare
        number must still mean a bounded pause, not a malformed ref."""
        feature = TaskFeature(agent=None)
        result = await feature.wait(target="0", reason="legacy positional")
        assert result.status is ToolResultStatus.OK
        assert result.data["requested_seconds"] == 0
        assert result.data["reason"] == "legacy positional"

    @pytest.mark.asyncio
    async def test_post_load_registers_task_provider(self):
        agent = _StubAgent()
        feature = TaskFeature(agent=None)
        feature.agent = agent
        await feature.post_all_features_loaded(agent)
        assert "task" in agent.wait_registry.kinds()


class TestWaitModeSignal:
    """`wait(target=..., mode="signal")` registers a watch and returns
    immediately — the explicit half of "every async waitable is wakeable"
    that also works for poll-only providers like TaskWaitable."""

    @pytest.mark.asyncio
    async def test_signal_mode_registers_watch_and_returns_immediately(self, tmp_path):
        agent = await _make_db_agent(tmp_path)
        feature = TaskFeature(agent=None)
        feature.agent = agent
        # Register the task provider so the kind validates.
        await feature.post_all_features_loaded(agent)

        result = await feature.wait(target="task:abc123", mode="signal")
        assert result.status is ToolResultStatus.OK
        assert result.data["mode"] == "signal"
        assert result.data["watching"] is True
        assert result.data["ref"] == "task:abc123"

        # A durable watch row was persisted (watching=1, not yet signaled).
        store = agent._wait_reconciler._store
        watched = await store.list_watched()
        keys = {(w.kind, w.handle) for w in watched}
        assert ("task", "abc123") in keys

    @pytest.mark.asyncio
    async def test_signal_mode_unknown_kind_errors(self, tmp_path):
        agent = await _make_db_agent(tmp_path)
        feature = TaskFeature(agent=None)
        feature.agent = agent
        result = await feature.wait(target="bogus:xyz", mode="signal")
        assert result.status is ToolResultStatus.ERROR
        assert "no wait provider for kind 'bogus'" in result.error

    @pytest.mark.asyncio
    async def test_signal_mode_requires_target(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(duration_seconds=5, mode="signal")
        assert result.status is ToolResultStatus.ERROR
        assert "requires a target" in result.error

    @pytest.mark.asyncio
    async def test_signal_mode_numeric_target_is_not_a_handle(self):
        """A bare numeric token is a bounded sleep, not a handle — so
        mode='signal' with it must fail the no-target validation."""
        feature = TaskFeature(agent=None)
        result = await feature.wait(target="5", mode="signal")
        assert result.status is ToolResultStatus.ERROR
        assert "requires a target" in result.error

    @pytest.mark.asyncio
    async def test_invalid_mode_errors(self):
        feature = TaskFeature(agent=None)
        result = await feature.wait(target="task:abc", mode="bogus")
        assert result.status is ToolResultStatus.ERROR
        assert "mode must be 'block' or 'signal'" in result.error

    @pytest.mark.asyncio
    async def test_block_mode_is_default(self, tmp_path):
        """Default mode stays blocking — no watch row is created."""
        agent = await _make_db_agent(tmp_path)
        feature = TaskFeature(agent=None)
        feature.agent = agent
        await feature.post_all_features_loaded(agent)

        async def fake_status(task_id):
            return {
                "ok": True, "task_id": task_id, "status": "completed",
                "task_type": "demo", "artifacts": [], "message": "done",
            }

        feature._get_task_status_data = fake_status
        result = await feature.wait(target="task:abc123", timeout_seconds=5)
        assert result.status is ToolResultStatus.OK
        # Blocking path doesn't lazily build a reconciler/watch row.
        assert getattr(agent, "_wait_reconciler", None) is None
