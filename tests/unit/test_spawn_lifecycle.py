"""Unit tests for SpawnedAgentLifecycle and hook events."""

import asyncio
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.hooks.base import HookEvent, HookInput, HookOutput

from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.multi_agent.agent_manager import (
    AgentManager,
    RuntimeOffboardingRetainedError,
)
from kestrel_sovereign.spawn.lifecycle import (
    SpawnedAgentLifecycle,
    SpawnMode,
    SpawnResult,
    SpawnStatus,
)
from kestrel_sovereign.spawn.mandate import SpawnMandate


def _make_mock_manager():
    """Create a mock AgentManager with terminate_child."""
    manager = MagicMock()
    manager.terminate_child = AsyncMock(return_value=True)
    manager.get_children = MagicMock(return_value=[])
    manager.get_agent = MagicMock(return_value=None)
    return manager


def test_restored_ephemeral_ttl_rearms_after_sync_construction() -> None:
    """A lifecycle first built without a loop must arm its timer later."""

    manager = AgentManager()
    mandate = SpawnMandate(
        parent_did="did:test:parent",
        child_did="did:test:child",
        ttl_seconds=3600,
    )
    manager._child_mandates["Restored"] = mandate
    lifecycle = SpawnedAgentLifecycle(manager)
    assert lifecycle._tracked["Restored"].ttl_task is None

    async def rearm() -> None:
        lifecycle.restore_from_manager()
        ttl_task = lifecycle._tracked["Restored"].ttl_task
        assert ttl_task is not None
        lifecycle.withdraw_persisted_child("Restored")
        await asyncio.sleep(0)
        assert ttl_task.cancelled()

    asyncio.run(rearm())


@pytest.mark.asyncio
async def test_manager_prune_cancels_removed_child_ttl_before_name_reuse() -> None:
    manager = AgentManager()
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    old_did = "did:test:removed-child"
    await lifecycle.register(
        "Reusable",
        old_did,
        "did:test:parent",
        ttl_seconds=3600,
    )
    old_task = lifecycle._tracked["Reusable"].ttl_task
    manager._parent_children["did:test:parent"] = ["Reusable"]
    manager._child_mandates["Reusable"] = SpawnMandate(
        parent_did="did:test:parent",
        child_did=old_did,
    )

    manager._prune_child_relationship_and_mandate(
        "did:test:parent",
        "Reusable",
    )
    assert not lifecycle.is_tracked("Reusable")
    await asyncio.sleep(0)
    await lifecycle.register(
        "Reusable",
        "did:test:replacement-child",
        "did:test:other-parent",
        ttl_seconds=3600,
    )

    assert old_task is not None and old_task.cancelled()
    assert lifecycle._tracked["Reusable"].child_did == "did:test:replacement-child"
    await lifecycle.shutdown()


@pytest.mark.asyncio
async def test_manager_prune_does_not_cancel_lifecycle_task_terminating_itself() -> None:
    manager = AgentManager()
    lifecycle = SpawnedAgentLifecycle(manager)
    manager._lifecycle = lifecycle
    child_did = "did:test:self-terminating-child"
    await lifecycle.register(
        "SelfTerminating",
        child_did,
        "did:test:parent",
        ttl_seconds=3600,
    )
    original_ttl = lifecycle._tracked["SelfTerminating"].ttl_task
    assert original_ttl is not None
    original_ttl.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original_ttl
    lifecycle._tracked["SelfTerminating"].ttl_task = asyncio.current_task()
    manager._parent_children["did:test:parent"] = ["SelfTerminating"]
    manager._child_mandates["SelfTerminating"] = SpawnMandate(
        parent_did="did:test:parent",
        child_did=child_did,
    )

    manager._prune_child_relationship_and_mandate(
        "did:test:parent",
        "SelfTerminating",
    )

    assert not lifecycle.is_tracked("SelfTerminating")
    assert not asyncio.current_task().cancelling()


@pytest.mark.asyncio
async def test_stale_ttl_monitor_cannot_terminate_same_name_replacement() -> None:
    manager = _make_mock_manager()
    lifecycle = SpawnedAgentLifecycle(manager)
    await lifecycle.register(
        "Reusable",
        "did:test:replacement-child",
        "did:test:parent",
        ttl_seconds=3600,
    )

    await lifecycle._ttl_monitor(
        "Reusable",
        "did:test:removed-child",
        0,
    )

    manager.terminate_child.assert_not_awaited()
    assert lifecycle._tracked["Reusable"].child_did == "did:test:replacement-child"
    await lifecycle.shutdown()


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
        assert result.finalized_from_absence is False

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

    @pytest.mark.asyncio
    async def test_ttl_reaper_reconciles_grouped_retained_offboarding(self):
        manager = _make_mock_manager()
        retained = RuntimeOffboardingRetainedError(
            agent_name="ephemeral",
            agent_id="did:child",
            runtime_path=Path("operator/runtime/child"),
            cause=OSError("retained"),
        )
        manager.terminate_child.side_effect = BaseExceptionGroup(
            "cancelled retained cleanup",
            [asyncio.CancelledError(), retained],
        )
        lifecycle = SpawnedAgentLifecycle(manager)

        await lifecycle.register(
            child_name="ephemeral",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.01,
        )
        ttl_task = lifecycle._tracked["ephemeral"].ttl_task
        await asyncio.sleep(0.1)

        assert ttl_task.done()
        assert ttl_task.exception() is None
        assert not lifecycle.is_tracked("ephemeral")
        assert lifecycle.get_result("ephemeral").status == SpawnStatus.TIMED_OUT

    @pytest.mark.asyncio
    async def test_refused_ttl_termination_rearms_and_publishes_only_on_retry(self):
        manager = _make_mock_manager()
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["retry-ttl"]
        first_refused = asyncio.Event()
        allow_retry = asyncio.Event()
        attempts = 0

        async def terminate_child(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                first_refused.set()
                return False
            await allow_retry.wait()
            return True

        manager.terminate_child.side_effect = terminate_child
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name="retry-ttl",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=0.01,
        )
        first_ttl = lifecycle._tracked["retry-ttl"].ttl_task

        await asyncio.wait_for(first_refused.wait(), timeout=1)
        await asyncio.sleep(0)

        assert lifecycle.is_tracked("retry-ttl")
        assert lifecycle.get_result("retry-ttl") is None
        retry_ttl = lifecycle._tracked["retry-ttl"].ttl_task
        assert retry_ttl is not first_ttl
        assert retry_ttl is not None and not retry_ttl.done()

        allow_retry.set()
        for _ in range(100):
            if not lifecycle.is_tracked("retry-ttl"):
                break
            await asyncio.sleep(0.01)

        assert not lifecycle.is_tracked("retry-ttl")
        assert manager.terminate_child.await_count == 2
        assert lifecycle.get_result("retry-ttl").status is SpawnStatus.TIMED_OUT

    @pytest.mark.asyncio
    async def test_ttl_finalizes_child_already_removed_by_real_manager(self, tmp_path):
        """A pruned parent edge is already-gone evidence, not a refusal."""

        manager = AgentManager()
        child = MagicMock()
        child.agent_id = "did:child:already-gone"
        child.shutdown = AsyncMock()
        manager._agents["already-gone"] = child
        manager._agent_names[child.agent_id] = "already-gone"
        manager._parent_children["did:parent"] = ["already-gone"]
        lifecycle = SpawnedAgentLifecycle(manager)
        ephemeral_dir = tmp_path / "already-gone"
        ephemeral_dir.mkdir()
        (ephemeral_dir / "artifact").write_text("temporary")

        await lifecycle.register(
            child_name="already-gone",
            child_did=child.agent_id,
            parent_did="did:parent",
            ttl_seconds=0.05,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=str(ephemeral_dir),
        )
        lifecycle._fire_hook = AsyncMock()
        assert await manager.remove_agent("already-gone") is True
        assert manager.get_agent("already-gone") is None
        assert manager.get_children("did:parent") == []

        for _ in range(100):
            if not lifecycle.is_tracked("already-gone"):
                break
            await asyncio.sleep(0.01)

        assert not lifecycle.is_tracked("already-gone")
        assert lifecycle.get_result("already-gone").status is SpawnStatus.TIMED_OUT
        assert not ephemeral_dir.exists()
        lifecycle._fire_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ttl_refusal_retries_are_bounded_and_explicitly_retryable(
        self, tmp_path, caplog
    ):
        """A live refused child stops auto-looping but remains operator-retryable."""

        manager = _make_mock_manager()
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["bounded-refusal"]
        manager.terminate_child.return_value = False
        lifecycle = SpawnedAgentLifecycle(manager)
        ephemeral_dir = tmp_path / "bounded-refusal"
        ephemeral_dir.mkdir()

        await lifecycle.register(
            child_name="bounded-refusal",
            child_did="did:child:bounded-refusal",
            parent_did="did:parent",
            ttl_seconds=0.01,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=str(ephemeral_dir),
        )
        lifecycle._fire_hook = AsyncMock()
        for _ in range(100):
            refusal = lifecycle.get_termination_refusal("bounded-refusal")
            if refusal is not None:
                break
            await asyncio.sleep(0.01)

        refusal = lifecycle.get_termination_refusal("bounded-refusal")
        assert refusal is not None
        recorded_at = refusal.pop("recorded_at")
        assert recorded_at
        assert refusal == {
            "termination_not_performed": True,
            "automatic_termination_attempts": 3,
            "automatic_retries_exhausted": True,
            "operator_action_required": True,
            "retry_termination": True,
            "requested_status": "timed_out",
        }
        assert lifecycle.get_result("bounded-refusal") is None
        assert lifecycle._tracked["bounded-refusal"].result is None
        assert manager.terminate_child.await_count == 3
        await asyncio.sleep(0.05)
        assert manager.terminate_child.await_count == 3
        assert lifecycle.is_tracked("bounded-refusal")
        assert ephemeral_dir.exists()
        lifecycle._fire_hook.assert_not_awaited()
        assert sum(
            "periodic retry remain active" in record.getMessage()
            for record in caplog.records
        ) == 2
        assert sum(
            "automatic retries stopped" in record.getMessage()
            for record in caplog.records
        ) == 1

        manager.terminate_child.return_value = True
        retried = await lifecycle.terminate("bounded-refusal")

        assert retried is not None
        assert retried.status is SpawnStatus.TERMINATED
        assert manager.terminate_child.await_count == 4
        assert not lifecycle.is_tracked("bounded-refusal")
        assert lifecycle.get_termination_refusal("bounded-refusal") is None
        assert not ephemeral_dir.exists()
        lifecycle._fire_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_real_result_supersedes_bounded_ttl_refusal(self, tmp_path):
        """Operator refusal state cannot consume child artifacts or budget."""

        manager = _make_mock_manager()
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["completed-after-refusal"]
        manager.terminate_child.return_value = False
        lifecycle = SpawnedAgentLifecycle(manager)
        ephemeral_dir = tmp_path / "completed-after-refusal"
        ephemeral_dir.mkdir()
        await lifecycle.register(
            child_name="completed-after-refusal",
            child_did="did:child:completed-after-refusal",
            parent_did="did:parent",
            ttl_seconds=0.01,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=str(ephemeral_dir),
        )
        lifecycle._fire_hook = AsyncMock()
        for _ in range(100):
            if lifecycle.get_termination_refusal("completed-after-refusal"):
                break
            await asyncio.sleep(0.01)

        assert lifecycle.get_termination_refusal("completed-after-refusal")
        assert lifecycle.get_result("completed-after-refusal") is None
        manager.terminate_child.return_value = True

        result = await lifecycle.report_result(
            "completed-after-refusal",
            output_artifacts={"answer": "42"},
            budget_consumed=Decimal("1.75"),
        )

        assert result is not None
        assert result.status is SpawnStatus.COMPLETED
        assert result.output_artifacts == {"answer": "42"}
        assert result.budget_consumed == Decimal("1.75")
        assert lifecycle.get_result("completed-after-refusal") is result
        assert lifecycle.get_termination_refusal("completed-after-refusal") is None
        assert not lifecycle.is_tracked("completed-after-refusal")
        assert not ephemeral_dir.exists()
        lifecycle._fire_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refused_result_report_preserves_operator_state(self, tmp_path):
        """A work result cannot erase a still-live child's refusal witness."""

        manager = _make_mock_manager()
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["completed-but-live"]
        manager.terminate_child.return_value = False
        lifecycle = SpawnedAgentLifecycle(manager)
        ephemeral_dir = tmp_path / "completed-but-live"
        ephemeral_dir.mkdir()
        await lifecycle.register(
            child_name="completed-but-live",
            child_did="did:child:completed-but-live",
            parent_did="did:parent",
            ttl_seconds=0.01,
            mode=SpawnMode.EPHEMERAL,
            temp_dir=str(ephemeral_dir),
        )
        lifecycle._fire_hook = AsyncMock()
        for _ in range(100):
            refusal = lifecycle.get_termination_refusal("completed-but-live")
            if refusal is not None:
                break
            await asyncio.sleep(0.01)

        refusal = lifecycle.get_termination_refusal("completed-but-live")
        assert refusal is not None
        assert refusal["automatic_retries_exhausted"] is True

        result = await lifecycle.report_result(
            "completed-but-live",
            output_artifacts={"answer": "not-finalized"},
            budget_consumed=Decimal("2.25"),
        )

        assert result is None
        assert lifecycle.get_termination_refusal("completed-but-live") == refusal
        assert lifecycle.get_result("completed-but-live") is None
        assert lifecycle.is_tracked("completed-but-live")
        assert ephemeral_dir.exists()
        lifecycle._fire_hook.assert_not_awaited()

        manager.terminate_child.return_value = True
        finalized = await lifecycle.terminate("completed-but-live")
        assert finalized is not None
        assert lifecycle.get_termination_refusal("completed-but-live") is None


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

    @pytest.mark.asyncio
    async def test_terminate_finalizes_authoritatively_already_gone_child(self):
        manager = _make_mock_manager()
        manager.terminate_child.return_value = False
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name="already-gone",
            child_did="did:child:already-gone",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        lifecycle._fire_hook = AsyncMock()

        result = await lifecycle.terminate("already-gone")

        assert result is not None
        assert result.status is SpawnStatus.TERMINATED
        assert result.finalized_from_absence is True
        assert not lifecycle.is_tracked("already-gone")
        lifecycle._fire_hook.assert_awaited_once()


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

    @pytest.mark.asyncio
    async def test_shutdown_refusal_keeps_tracking_ttl_and_no_terminal_result(self):
        from kestrel_sovereign.multi_agent.agent_manager import (
            ChildTerminationNotPerformedError,
        )

        manager = _make_mock_manager()
        manager.terminate_child.return_value = False
        manager.get_agent.return_value = object()
        manager.get_children.return_value = ["refused"]
        lifecycle = SpawnedAgentLifecycle(manager)
        await lifecycle.register(
            child_name="refused",
            child_did="did:refused",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        ttl_task = lifecycle._tracked["refused"].ttl_task
        lifecycle._fire_hook = AsyncMock()

        with pytest.raises(ChildTerminationNotPerformedError):
            await lifecycle.shutdown()

        assert lifecycle.is_tracked("refused")
        assert lifecycle.get_result("refused") is None
        assert lifecycle._tracked["refused"].ttl_task is ttl_task
        assert ttl_task is not None and not ttl_task.done()
        lifecycle._fire_hook.assert_not_awaited()
        ttl_task.cancel()
        await asyncio.gather(ttl_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_shutdown_continues_after_retained_offboarding(self):
        manager = _make_mock_manager()
        retained = RuntimeOffboardingRetainedError(
            agent_name="child1",
            agent_id="did:c1",
            runtime_path=Path("operator/runtime/child1"),
            cause=OSError("retained"),
        )
        manager.terminate_child.side_effect = [retained, True]
        lifecycle = SpawnedAgentLifecycle(manager)
        for child_name, child_did in (("child1", "did:c1"), ("child2", "did:c2")):
            await lifecycle.register(
                child_name=child_name,
                child_did=child_did,
                parent_did="did:parent",
                ttl_seconds=3600,
            )

        with pytest.raises(RuntimeOffboardingRetainedError):
            await lifecycle.shutdown()

        assert manager.terminate_child.await_count == 2
        assert lifecycle.get_tracked_children() == []


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
