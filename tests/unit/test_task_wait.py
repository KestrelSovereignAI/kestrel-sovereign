"""Tests for the single generic ``wait`` tool (WaitFeature) and its
dispatch to feature-registered Waitable providers (e.g. TaskFeature).

#1541: agents shelled out to ``sleep`` between polls; ``wait`` is the
native bounded pause. #1860: ``wait`` became the ONE generic waiter —
``wait("<kind>:<handle>")`` dispatches to whichever feature registered
that kind. The tool lives on its own MANDATORY ``WaitFeature`` (not on
TaskFeature) so it is present even for agent profiles that don't load
tasks.
"""

from types import SimpleNamespace

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.features.wait.feature import WaitFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.waits import WaitRegistry


class _StubAgent:
    def __init__(self):
        self.wait_registry = WaitRegistry()


def _wait_feature(agent=None):
    """A WaitFeature bound to ``agent`` (the tool reads agent.wait_registry)."""
    f = WaitFeature(agent=None)
    f.agent = agent
    return f


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


async def _register_task_provider(agent, *, status="completed"):
    """Register a TaskFeature's ``task`` provider on ``agent`` with a stubbed
    status read, and return the TaskFeature (so the provider polls the stub)."""
    task_feature = TaskFeature(agent=None)
    task_feature.agent = agent
    await task_feature.post_all_features_loaded(agent)

    async def fake_status(task_id):
        return {
            "ok": True, "task_id": task_id, "status": status,
            "task_type": "demo", "artifacts": [], "message": "done",
        }

    task_feature._get_task_status_data = fake_status
    return task_feature


class TestGenericWaitSleep:
    @pytest.mark.asyncio
    async def test_zero_duration_returns_immediately(self):
        result = await _wait_feature().wait(duration_seconds=0, reason="probe")
        assert result.status is ToolResultStatus.OK
        assert result.data["requested_seconds"] == 0
        assert result.data["reason"] == "probe"
        assert result.data["elapsed_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_short_wait_reports_elapsed(self):
        result = await _wait_feature().wait(duration_seconds=1)
        assert result.status is ToolResultStatus.OK
        assert result.data["elapsed_seconds"] >= 1
        assert result.data["requested_seconds"] == 1

    @pytest.mark.asyncio
    async def test_max_duration_rejected(self):
        too_long = WaitFeature._MAX_WAIT_SECONDS + 1
        result = await _wait_feature().wait(duration_seconds=too_long)
        assert result.status is ToolResultStatus.ERROR
        assert "exceeds the maximum" in result.error
        assert result.data["requested_seconds"] == too_long
        assert result.data["max_seconds"] == WaitFeature._MAX_WAIT_SECONDS

    @pytest.mark.asyncio
    async def test_negative_duration_rejected(self):
        result = await _wait_feature().wait(duration_seconds=-5)
        assert result.status is ToolResultStatus.ERROR
        assert "must be >= 0" in result.error

    @pytest.mark.asyncio
    async def test_non_integer_duration_rejected(self):
        result = await _wait_feature().wait(duration_seconds="soon")
        assert result.status is ToolResultStatus.ERROR
        assert "must be an integer" in result.error


class TestUnifiedWaitTarget:
    """`wait("<kind>:<handle>")` dispatches to the agent's wait_registry."""

    @pytest.mark.asyncio
    async def test_target_dispatches_to_registered_provider(self):
        agent = _StubAgent()
        await _register_task_provider(agent, status="completed")
        result = await _wait_feature(agent).wait(target="task:abc123", timeout_seconds=5)
        assert result.status is ToolResultStatus.OK
        assert result.data["ref"] == "task:abc123"

    @pytest.mark.asyncio
    async def test_target_without_registry_errors(self):
        result = await _wait_feature(agent=None).wait(target="task:abc", timeout_seconds=5)
        assert result.status is ToolResultStatus.ERROR
        assert "wait engine unavailable" in result.error

    @pytest.mark.asyncio
    async def test_unknown_kind_errors(self):
        result = await _wait_feature(_StubAgent()).wait(target="bogus:xyz", timeout_seconds=5)
        assert result.status is ToolResultStatus.ERROR
        assert "no wait provider for kind 'bogus'" in result.error

    @pytest.mark.asyncio
    async def test_numeric_target_is_bounded_sleep(self):
        """`!wait 5` binds the token positionally to `target`; a bare
        number must still mean a bounded pause, not a malformed ref."""
        result = await _wait_feature().wait(target="0", reason="legacy positional")
        assert result.status is ToolResultStatus.OK
        assert result.data["requested_seconds"] == 0
        assert result.data["reason"] == "legacy positional"

    @pytest.mark.asyncio
    async def test_task_feature_registers_task_provider(self):
        """The task provider registration lives on TaskFeature (the kind is
        only available when tasks is loaded), but the wait TOOL does not."""
        agent = _StubAgent()
        task_feature = TaskFeature(agent=None)
        task_feature.agent = agent
        await task_feature.post_all_features_loaded(agent)
        assert "task" in agent.wait_registry.kinds()


class TestWaitModeSignal:
    """`wait(target=..., mode="signal")` registers a watch and returns
    immediately — the explicit half of "every async waitable is wakeable"
    that also works for poll-only providers like TaskWaitable."""

    @pytest.mark.asyncio
    async def test_signal_mode_registers_watch_and_returns_immediately(self, tmp_path):
        agent = await _make_db_agent(tmp_path)
        await _register_task_provider(agent)
        result = await _wait_feature(agent).wait(target="task:abc123", mode="signal")
        assert result.status is ToolResultStatus.OK
        assert result.data["mode"] == "signal"
        assert result.data["watching"] is True
        assert result.data["ref"] == "task:abc123"

        store = agent._wait_reconciler._store
        keys = {(w.kind, w.handle) for w in await store.list_watched()}
        assert ("task", "abc123") in keys

    @pytest.mark.asyncio
    async def test_signal_mode_unknown_kind_errors(self, tmp_path):
        agent = await _make_db_agent(tmp_path)
        result = await _wait_feature(agent).wait(target="bogus:xyz", mode="signal")
        assert result.status is ToolResultStatus.ERROR
        assert "no wait provider for kind 'bogus'" in result.error

    @pytest.mark.asyncio
    async def test_signal_mode_requires_target(self):
        result = await _wait_feature().wait(duration_seconds=5, mode="signal")
        assert result.status is ToolResultStatus.ERROR
        assert "requires a target" in result.error

    @pytest.mark.asyncio
    async def test_signal_mode_numeric_target_is_not_a_handle(self):
        result = await _wait_feature().wait(target="5", mode="signal")
        assert result.status is ToolResultStatus.ERROR
        assert "requires a target" in result.error

    @pytest.mark.asyncio
    async def test_invalid_mode_errors(self):
        result = await _wait_feature().wait(target="task:abc", mode="bogus")
        assert result.status is ToolResultStatus.ERROR
        assert "mode must be 'block' or 'signal'" in result.error

    @pytest.mark.asyncio
    async def test_block_mode_is_default(self, tmp_path):
        """Default mode stays blocking — no watch row is created."""
        agent = await _make_db_agent(tmp_path)
        await _register_task_provider(agent)
        result = await _wait_feature(agent).wait(target="task:abc123", timeout_seconds=5)
        assert result.status is ToolResultStatus.OK
        # Blocking path doesn't lazily build a reconciler/watch row.
        assert getattr(agent, "_wait_reconciler", None) is None
