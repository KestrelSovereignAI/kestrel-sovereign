"""Unit tests for SpawnedAgentLifecycle and hook events."""

import asyncio
import os
import tempfile
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.hooks.base import HookEvent, HookInput, HookOutput
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.spawn.lifecycle import (
    SpawnedAgentLifecycle,
    SpawnMode,
    SpawnResult,
    SpawnStatus,
)


def _make_mock_manager():
    """Create a mock AgentManager with terminate_child."""
    manager = MagicMock()
    manager.terminate_child = AsyncMock(return_value=True)
    manager.get_children = MagicMock(return_value=[])
    return manager


class TestSpawnResult:
    """SpawnResult dataclass basics."""

    def test_defaults(self):
        result = SpawnResult(
            child_name="worker",
            child_did="did:child",
            status=SpawnStatus.COMPLETED,
        )
        assert result.child_name == "worker"
        assert result.status == SpawnStatus.COMPLETED
        assert result.output_artifacts == {}
        assert result.budget_consumed == Decimal("0")
        assert result.ended_at  # non-empty

    def test_with_artifacts(self):
        result = SpawnResult(
            child_name="worker",
            child_did="did:child",
            status=SpawnStatus.COMPLETED,
            output_artifacts={"summary": "done", "files": ["out.txt"]},
            budget_consumed=Decimal("3.50"),
        )
        assert result.output_artifacts["summary"] == "done"
        assert result.budget_consumed == Decimal("3.50")


class TestSpawnStatus:
    """SpawnStatus enum values."""

    def test_all_statuses(self):
        assert SpawnStatus.RUNNING == "running"
        assert SpawnStatus.COMPLETED == "completed"
        assert SpawnStatus.TERMINATED == "terminated"
        assert SpawnStatus.TIMED_OUT == "timed_out"
        assert SpawnStatus.FAILED == "failed"


class TestSpawnMode:
    """SpawnMode enum values."""

    def test_modes(self):
        assert SpawnMode.EPHEMERAL == "ephemeral"
        assert SpawnMode.PERSISTENT == "persistent"


class TestLifecycleRegistration:
    """Test child registration and tracking."""

    @pytest.mark.asyncio
    async def test_register_tracks_child(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
            purpose="test task",
        )

        assert lifecycle.is_tracked("worker")
        assert "worker" in lifecycle.get_tracked_children()

        # Clean up
        await lifecycle.shutdown()

    @pytest.mark.asyncio
    async def test_register_starts_ttl_task(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        tracked = lifecycle._tracked["worker"]
        assert tracked.ttl_task is not None
        assert not tracked.ttl_task.done()

        await lifecycle.shutdown()


class TestTTLExpiration:
    """TTL expiration triggers auto-termination."""

    @pytest.mark.asyncio
    async def test_ttl_expiry_auto_terminates(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        # Use a very short TTL
        await lifecycle.register(
            child_name="ephemeral",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.1,  # 100ms
        )

        # Wait for TTL to expire
        await asyncio.sleep(0.3)

        # Child should have been terminated
        assert not lifecycle.is_tracked("ephemeral")

        # Result should be stored with TIMED_OUT status
        result = lifecycle.get_result("ephemeral")
        assert result is not None
        assert result.status == SpawnStatus.TIMED_OUT
        assert result.child_name == "ephemeral"

        # AgentManager.terminate_child should have been called
        manager.terminate_child.assert_awaited_once_with("did:parent", "ephemeral")

    @pytest.mark.asyncio
    async def test_ttl_cancelled_on_early_completion(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="quick",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=100,  # Long TTL
        )

        ttl_task = lifecycle._tracked["quick"].ttl_task

        # Report result before TTL
        await lifecycle.report_result(
            child_name="quick",
            output_artifacts={"answer": 42},
            status=SpawnStatus.COMPLETED,
        )

        # Let the cancellation propagate
        await asyncio.sleep(0.05)

        # TTL task should be cancelled or done
        assert ttl_task.done()

        # Child should be cleaned up
        assert not lifecycle.is_tracked("quick")

        result = lifecycle.get_result("quick")
        assert result.status == SpawnStatus.COMPLETED


class TestResultCollection:
    """Result reporting from child to parent."""

    @pytest.mark.asyncio
    async def test_report_result_stores_result(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        result = await lifecycle.report_result(
            child_name="worker",
            output_artifacts={"data": [1, 2, 3]},
            budget_consumed=Decimal("1.25"),
            status=SpawnStatus.COMPLETED,
        )

        assert result is not None
        assert result.status == SpawnStatus.COMPLETED
        assert result.output_artifacts == {"data": [1, 2, 3]}
        assert result.budget_consumed == Decimal("1.25")

    @pytest.mark.asyncio
    async def test_report_result_untracked_returns_none(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        result = await lifecycle.report_result(
            child_name="ghost",
            output_artifacts={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pop_result_removes_it(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        await lifecycle.report_result(child_name="worker")

        result = lifecycle.pop_result("worker")
        assert result is not None
        assert lifecycle.get_result("worker") is None

    @pytest.mark.asyncio
    async def test_report_failed_status(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="broken",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        result = await lifecycle.report_result(
            child_name="broken",
            status=SpawnStatus.FAILED,
        )

        assert result.status == SpawnStatus.FAILED


class TestEphemeralCleanup:
    """Ephemeral cleanup — no leftover temp files."""

    @pytest.mark.asyncio
    async def test_ephemeral_cleanup_removes_temp_dir(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        # Create a real temp directory
        temp_dir = tempfile.mkdtemp(prefix="kestrel_test_")
        # Put a file in it to verify cleanup
        with open(os.path.join(temp_dir, "test.txt"), "w") as f:
            f.write("ephemeral data")

        assert os.path.exists(temp_dir)

        await lifecycle.register(
            child_name="temp_worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=temp_dir,
        )

        # Terminate triggers cleanup
        await lifecycle.terminate("temp_worker", reason="test cleanup")

        # Temp dir should be gone
        assert not os.path.exists(temp_dir)

    @pytest.mark.asyncio
    async def test_persistent_mode_keeps_data(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        temp_dir = tempfile.mkdtemp(prefix="kestrel_test_persist_")

        await lifecycle.register(
            child_name="persist_worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
            mode=SpawnMode.PERSISTENT,
            temp_dir=temp_dir,
        )

        await lifecycle.terminate("persist_worker")

        # Persistent mode should NOT clean up the directory
        assert os.path.exists(temp_dir)

        # Manual cleanup
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_ephemeral_ttl_cleanup(self):
        """TTL expiry also cleans up ephemeral resources."""
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        temp_dir = tempfile.mkdtemp(prefix="kestrel_test_ttl_")

        await lifecycle.register(
            child_name="ttl_worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.1,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=temp_dir,
        )

        await asyncio.sleep(0.3)

        assert not os.path.exists(temp_dir)

    @pytest.mark.asyncio
    async def test_create_ephemeral_dir(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        temp_dir = lifecycle.create_ephemeral_dir()
        assert os.path.isdir(temp_dir)
        assert "kestrel_spawn_" in temp_dir

        # Cleanup
        os.rmdir(temp_dir)


class TestExplicitTermination:
    """Explicit terminate() method."""

    @pytest.mark.asyncio
    async def test_terminate_returns_result(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="doomed",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        result = await lifecycle.terminate("doomed", reason="no longer needed")

        assert result is not None
        assert result.status == SpawnStatus.TERMINATED
        assert not lifecycle.is_tracked("doomed")

    @pytest.mark.asyncio
    async def test_terminate_untracked_returns_none(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        result = await lifecycle.terminate("ghost")
        assert result is None


class TestCascadingShutdown:
    """Parent termination terminates all children."""

    @pytest.mark.asyncio
    async def test_shutdown_terminates_all_children(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="child1",
            child_did="did:c1",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        await lifecycle.register(
            child_name="child2",
            child_did="did:c2",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        assert len(lifecycle.get_tracked_children()) == 2

        await lifecycle.shutdown()

        assert len(lifecycle.get_tracked_children()) == 0
        assert manager.terminate_child.await_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_stores_results(self):
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="child1",
            child_did="did:c1",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        await lifecycle.shutdown()

        result = lifecycle.get_result("child1")
        assert result is not None
        assert result.status == SpawnStatus.TERMINATED


class TestHookEvents:
    """Hook events fire correctly for spawn and terminate."""

    @pytest.mark.asyncio
    async def test_spawn_fires_agent_spawn_hook(self):
        manager = _make_mock_manager()
        hooks = HooksManager()

        received_inputs = []

        class SpawnHook:
            def __init__(self):
                self.name = "test_spawn_hook"
                self.events = [HookEvent.AGENT_SPAWN]
                self.matcher = None
                self.priority = 100
                self.timeout = 5.0
                self.enabled = True
                self._compiled_matcher = None

            def matches(self, tool_name):
                return True

            async def execute(self, input: HookInput) -> HookOutput:
                received_inputs.append(input)
                return HookOutput.allow()

        hooks.register(SpawnHook())

        lifecycle = SpawnedAgentLifecycle(manager, hooks_manager=hooks)

        await lifecycle.register(
            child_name="hooked_child",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
            purpose="hook test",
        )

        assert len(received_inputs) == 1
        inp = received_inputs[0]
        assert inp.hook_event_name == "AgentSpawn"
        assert inp.parent_did == "did:parent"
        assert inp.child_did == "did:child"
        assert inp.child_name == "hooked_child"
        assert inp.spawn_purpose == "hook test"

        await lifecycle.shutdown()

    @pytest.mark.asyncio
    async def test_terminate_fires_agent_terminate_hook(self):
        manager = _make_mock_manager()
        hooks = HooksManager()

        received_inputs = []

        class TermHook:
            def __init__(self):
                self.name = "test_term_hook"
                self.events = [HookEvent.AGENT_TERMINATE]
                self.matcher = None
                self.priority = 100
                self.timeout = 5.0
                self.enabled = True
                self._compiled_matcher = None

            def matches(self, tool_name):
                return True

            async def execute(self, input: HookInput) -> HookOutput:
                received_inputs.append(input)
                return HookOutput.allow()

        hooks.register(TermHook())

        lifecycle = SpawnedAgentLifecycle(manager, hooks_manager=hooks)

        await lifecycle.register(
            child_name="hooked_child",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        await lifecycle.terminate("hooked_child", reason="done")

        assert len(received_inputs) == 1
        inp = received_inputs[0]
        assert inp.hook_event_name == "AgentTerminate"
        assert inp.child_name == "hooked_child"
        assert inp.termination_reason == "done"

    @pytest.mark.asyncio
    async def test_no_hooks_manager_no_error(self):
        """Lifecycle works fine without a HooksManager."""
        manager = _make_mock_manager()
        lifecycle = SpawnedAgentLifecycle(manager, hooks_manager=None)

        await lifecycle.register(
            child_name="worker",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )

        # Should not raise
        await lifecycle.terminate("worker")

    @pytest.mark.asyncio
    async def test_ttl_fires_terminate_hook(self):
        """TTL expiry should also fire the AGENT_TERMINATE hook."""
        manager = _make_mock_manager()
        hooks = HooksManager()

        received = []

        class TermHook:
            def __init__(self):
                self.name = "ttl_term_hook"
                self.events = [HookEvent.AGENT_TERMINATE]
                self.matcher = None
                self.priority = 100
                self.timeout = 5.0
                self.enabled = True
                self._compiled_matcher = None

            def matches(self, tool_name):
                return True

            async def execute(self, input: HookInput) -> HookOutput:
                received.append(input)
                return HookOutput.allow()

        hooks.register(TermHook())

        lifecycle = SpawnedAgentLifecycle(manager, hooks_manager=hooks)

        await lifecycle.register(
            child_name="ttl_child",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.1,
        )

        await asyncio.sleep(0.3)

        assert len(received) == 1
        assert received[0].termination_reason == "TTL expired"


class TestHookEventEnum:
    """Verify the new HookEvent values exist."""

    def test_agent_spawn_event(self):
        assert HookEvent.AGENT_SPAWN.value == "AgentSpawn"

    def test_agent_terminate_event(self):
        assert HookEvent.AGENT_TERMINATE.value == "AgentTerminate"

    def test_hooks_manager_initializes_new_events(self):
        """HooksManager should have registries for the new events."""
        hooks = HooksManager()
        assert HookEvent.AGENT_SPAWN in hooks._hooks
        assert HookEvent.AGENT_TERMINATE in hooks._hooks
