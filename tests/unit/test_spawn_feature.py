"""Unit tests for SpawnFeature and AgentManager spawn extensions."""

import asyncio
from pathlib import Path
import pytest
from kestrel_sdk.tools.result import ToolResultStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.spawn.feature import SpawnFeature
from kestrel_sovereign.features.isolated_runtime import (
    derive_isolated_runtime_namespace,
    prepare_isolated_runtime_namespace,
    resolve_isolated_runtime_namespace,
)
from kestrel_sovereign.multi_agent.agent_manager import (
    AgentManager,
    ChildTerminationReconciliationError,
    RuntimeOffboardingRetainedError,
)
from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle
from kestrel_sovereign.spawn.mandate import SpawnMandate


def _make_mock_agent(agent_id: str = "did:pkh:eip155:1:0xPARENT"):
    """Create a mock KestrelAgent."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()
    agent.process_input = AsyncMock(return_value="task completed")
    agent._private_key = None  # No signing in unit tests
    agent.identity = None
    return agent


def _make_spawn_feature(parent_agent=None, manager=None):
    """Create a SpawnFeature with a mock agent and optional manager.

    When manager is None, disables lazy resolution by setting a sentinel.
    """
    if parent_agent is None:
        parent_agent = _make_mock_agent()
        # Ensure lazy resolution doesn't find a manager on the mock
        parent_agent._agent_manager = None
        parent_agent.agent_manager = None
    feature = SpawnFeature(parent_agent)
    # Manually set initialized state
    feature._agent_manager = manager
    feature._child_results = {}
    feature._child_tasks = {}
    return feature


class TestSpawnFeatureTools:
    """Verify SpawnFeature exposes the correct tools."""

    def test_has_five_tools(self):
        feature = _make_spawn_feature()
        tools = feature.get_tools()
        tool_names = {t.name for t in tools}
        assert tool_names == {
            "spawn_agent",
            "list_children",
            "delegate_task",
            "get_child_result",
            "terminate_child",
        }

    def test_tool_description(self):
        feature = _make_spawn_feature()
        assert "child agents" in feature.tool_description

    def test_tool_name(self):
        feature = _make_spawn_feature()
        assert feature.tool_name == "spawn_feature"


class TestSpawnFeatureAutoManager:
    """In single-agent mode, SpawnFeature auto-creates an AgentManager."""

    @pytest.mark.asyncio
    async def test_auto_creates_manager(self):
        parent = _make_mock_agent()
        parent._agent_manager = None
        parent.agent_manager = None
        parent.storage_path = "/tmp/test/kestrel_prime.db"
        feature = SpawnFeature(parent)
        feature._agent_manager = None
        feature._child_results = {}
        feature._child_tasks = {}
        feature._lifecycle = None
        manager = feature._get_agent_manager()
        assert manager is not None
        assert parent._agent_manager is manager  # attached back

    @pytest.mark.asyncio
    async def test_list_children_auto_manager(self):
        parent = _make_mock_agent()
        parent._agent_manager = None
        parent.agent_manager = None
        parent.storage_path = "/tmp/test/kestrel_prime.db"
        feature = SpawnFeature(parent)
        feature._agent_manager = None
        feature._child_results = {}
        feature._child_tasks = {}
        feature._lifecycle = None
        envelope = await feature.list_children()
        assert envelope.data["children"] == []
        assert envelope.data["count"] == 0

    @pytest.mark.asyncio
    async def test_delegate_without_children(self):
        feature = _make_spawn_feature(manager=MagicMock())
        envelope = await feature.delegate_task(child_name="child1", task="do stuff")
        assert envelope.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_terminate_without_manager(self):
        feature = _make_spawn_feature(manager=None)
        envelope = await feature.terminate_child(child_name="child1")
        assert envelope.status is ToolResultStatus.ERROR


class TestSpawnFeatureWithManager:
    """Test SpawnFeature tools with a mock AgentManager."""

    @pytest.mark.asyncio
    async def test_spawn_agent_success(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")

        manager = MagicMock()
        manager.spawn_agent = AsyncMock(return_value=child)

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.spawn_agent(
            name="helper",
            purpose="assist with research",
            ttl=1800,
            constraints="max_tokens=1000,no_web",
            features="memory,web_search",
        )

        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["child_name"] == "helper"
        assert envelope.data["child_did"] == "did:child"

        # Verify mandate was constructed correctly
        call_args = manager.spawn_agent.call_args
        mandate = call_args.kwargs["mandate"]
        assert mandate.parent_did == "did:parent"
        assert mandate.purpose == "assist with research"
        assert mandate.ttl_seconds == 1800
        # max_tokens is coerced to int so ScopedConstitution.validate_constraints
        # (which type-checks it as int/float) accepts it (#2138); flags stay str.
        assert mandate.additional_constraints == {"max_tokens": 1000, "no_web": "true"}
        # Shorthand feature names are canonicalized to their class names so the
        # child's feature loader (which filters by cls.__name__) can match them
        # (#1946).
        assert mandate.features_allowed == ["MemoryFeature", "WebSearchFeature"]

    @pytest.mark.asyncio
    async def test_spawn_agent_failure(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.spawn_agent = AsyncMock(side_effect=ValueError("already exists"))

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.spawn_agent(name="dup", purpose="test")

        assert envelope.status is ToolResultStatus.ERROR
        assert "already exists" in envelope.error

    @pytest.mark.asyncio
    async def test_list_children(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")

        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=child)

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.list_children()

        assert envelope.data["count"] == 1
        assert envelope.data["children"][0]["name"] == "helper"
        assert envelope.data["children"][0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_delegate_task_success(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")

        manager = MagicMock()
        manager.get_agent = MagicMock(return_value=child)
        manager.get_children = MagicMock(return_value=["helper"])
        manager._lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle.report_result = AsyncMock()

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.delegate_task(child_name="helper", task="analyze data")

        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["child_name"] == "helper"

        # Wait for the async task to finish
        await asyncio.sleep(0.1)
        assert "helper" in feature._child_tasks
        # #F279: completing a delegated task records the result but must NOT
        # finalize/terminate the child. report_result (which runs
        # _terminate_and_cleanup) is NOT called per task, so the child stays
        # alive for the next delegate_task / get_child_result.
        manager._lifecycle.report_result.assert_not_awaited()
        assert feature._child_results["helper"]["success"] is True

    @pytest.mark.asyncio
    async def test_delegate_twice_keeps_child_alive(self):
        """#F279: the documented spawn → delegate → get_result → delegate-again
        flow. A second delegate must succeed (first task didn't kill the child)."""
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")
        manager = MagicMock()
        manager.get_agent = MagicMock(return_value=child)
        manager.get_children = MagicMock(return_value=["helper"])
        manager._lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle.report_result = AsyncMock()
        manager.terminate_child = AsyncMock()

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        first = await feature.delegate_task(child_name="helper", task="task 1")
        await asyncio.sleep(0.05)
        second = await feature.delegate_task(child_name="helper", task="task 2")
        await asyncio.sleep(0.05)

        assert first.status is ToolResultStatus.OK
        assert second.status is ToolResultStatus.OK  # child still alive
        manager._lifecycle.report_result.assert_not_awaited()
        manager.terminate_child.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_spawn_forwards_budget_to_mandate(self):
        """#2113: a per-child budget is now ENFORCED by AgentManager (hold from
        the parent wallet + ceiling'd DelegatedWallet), so the feature forwards it
        on the mandate rather than rejecting it. Refusal of an unbacked budget is
        the manager's job (validated in test_spawn_budget_enforcement)."""
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.spawn_agent = AsyncMock(return_value=_make_mock_agent("did:child"))

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.spawn_agent(name="helper", purpose="x", budget=5.0)

        assert envelope.status is ToolResultStatus.OK
        manager.spawn_agent.assert_awaited_once()
        mandate = manager.spawn_agent.call_args.kwargs["mandate"]
        assert mandate.budget_allocation == 5.0

    @pytest.mark.asyncio
    async def test_delegate_task_not_our_child(self):
        parent = _make_mock_agent("did:parent")
        other = _make_mock_agent("did:other")

        manager = MagicMock()
        manager.get_agent = MagicMock(return_value=other)
        manager.get_children = MagicMock(return_value=[])  # not our child

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.delegate_task(child_name="stranger", task="hack")

        assert envelope.status is ToolResultStatus.ERROR
        assert "not a child" in envelope.error

    @pytest.mark.asyncio
    async def test_get_child_result_after_delegation(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")
        child.process_input = AsyncMock(return_value="analysis complete: 42")

        manager = MagicMock()
        manager.get_agent = MagicMock(return_value=child)
        manager.get_children = MagicMock(return_value=["helper"])

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        # Delegate and wait for completion
        await feature.delegate_task(child_name="helper", task="analyze")
        await feature._child_tasks["helper"]

        envelope = await feature.get_child_result(child_name="helper")
        assert envelope.data["ready"] is True
        assert envelope.data["result"] == "analysis complete: 42"

    @pytest.mark.asyncio
    async def test_get_child_result_still_running(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")

        # Make the child take a long time
        async def slow_chat(msg):
            await asyncio.sleep(10)
            return "done"

        child.process_input = slow_chat

        manager = MagicMock()
        manager.get_agent = MagicMock(return_value=child)
        manager.get_children = MagicMock(return_value=["helper"])

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        await feature.delegate_task(child_name="helper", task="slow work")

        envelope = await feature.get_child_result(child_name="helper")
        assert envelope.data["ready"] is False
        assert "still running" in envelope.data["note"]

        # Clean up
        feature._child_tasks["helper"].cancel()

    @pytest.mark.asyncio
    async def test_get_child_result_no_task(self):
        feature = _make_spawn_feature(manager=MagicMock())
        envelope = await feature.get_child_result(child_name="nobody")
        assert envelope.data["ready"] is False

    @pytest.mark.asyncio
    async def test_terminate_child_success(self):
        parent = _make_mock_agent("did:parent")

        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager._lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle.terminate = AsyncMock(return_value=SimpleNamespace())

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.terminate_child(child_name="helper")

        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["runtime_offboarded"] is False
        assert envelope.data["runtime_retained_for_restart"] is True
        manager._lifecycle.terminate.assert_awaited_once_with(
            child_name="helper",
            reason="explicit termination",
        )

    @pytest.mark.asyncio
    async def test_terminate_child_explicit_offboard_threads_destructive_intent(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager._lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle.terminate = AsyncMock(return_value=SimpleNamespace())
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        envelope = await feature.terminate_child(
            child_name="helper",
            offboard_runtime=True,
        )

        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["runtime_offboarded"] is True
        assert envelope.data["runtime_retained_for_restart"] is False
        manager._lifecycle.terminate.assert_awaited_once_with(
            child_name="helper",
            reason="explicit termination",
            offboard_runtime=True,
        )

    @pytest.mark.asyncio
    async def test_terminate_child_reports_stopped_with_retained_runtime(self, caplog):
        """The tool separates successful stop from retained runtime custody."""

        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager.terminate_child = AsyncMock()
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        await lifecycle.register(
            child_name="helper",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        secret_path = Path("/operator/private/runtime/helper")
        secret = "retained-cause-secret"
        manager.terminate_child.side_effect = RuntimeOffboardingRetainedError(
            agent_name="helper",
            agent_id="did:child",
            runtime_path=secret_path,
            cause=OSError(secret),
        )
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        with caplog.at_level("ERROR"):
            envelope = await feature.terminate_child(
                child_name="helper",
                offboard_runtime=True,
            )

        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.confirmation == "Terminated child 'helper'."
        assert "runtime custody was retained" in envelope.error
        assert "Do not retry termination" in envelope.error
        assert envelope.data == {
            "terminated": True,
            "child_name": "helper",
            "agent_removed": True,
            "runtime_offboard_requested": True,
            "runtime_offboarded": False,
            "runtime_retained": True,
            "runtime_retained_for_restart": False,
            "named_child_runtime_retained": True,
            "named_child_runtime_removed": False,
            "runtime_cleanup_pending": False,
            "runtime_cleanup_state": "retained",
            "operator_action_required": True,
            "retry_termination": False,
            "retained_outcome_count": 1,
            "retained_agents": ["helper"],
            "retained_agent": "helper",
            "additional_outcome_count": 0,
            "additional_outcome_types": [],
            "runtime_custody_code": "runtime_offboarding_retained",
            "retained_cause_types": ["OSError"],
            "retained_cause_type": "OSError",
        }
        serialized = str(envelope.to_dict()) + caplog.text
        assert str(secret_path) not in serialized
        assert secret not in serialized
        assert lifecycle.is_tracked("helper") is False
        assert lifecycle.get_result("helper").status.value == "terminated"
        manager.terminate_child.assert_awaited_once_with(
            "did:parent",
            "helper",
            offboard_runtime=True,
        )

    @pytest.mark.asyncio
    async def test_terminate_child_maps_real_shutdown_handoff_to_pending_partial(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT",
            0.01,
        )

        class HostileChild:
            agent_id = "did:child:handoff"

            def __init__(self):
                self.shutdown_entered = asyncio.Event()
                self.allow_shutdown_finish = asyncio.Event()

            async def shutdown(self):
                self.shutdown_entered.set()
                while not self.allow_shutdown_finish.is_set():
                    try:
                        await self.allow_shutdown_finish.wait()
                    except asyncio.CancelledError:
                        pass

            def handoff_shutdown_to_reaper(self, shutdown_task):
                async def reap():
                    await asyncio.shield(shutdown_task)

                return asyncio.create_task(reap())

        parent = _make_mock_agent("did:parent")
        child = HostileChild()
        manager = AgentManager()
        manager._agents["helper"] = child
        manager._agent_names[child.agent_id] = "helper"
        manager._parent_children[parent.agent_id] = ["helper"]
        manager._offboard_agent_runtime_namespace = AsyncMock(return_value=(False, None))
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        await lifecycle.register(
            child_name="helper",
            child_did=child.agent_id,
            parent_did=parent.agent_id,
            ttl_seconds=3600,
        )
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        try:
            envelope = await feature.terminate_child(
                child_name="helper",
                offboard_runtime=True,
            )

            assert envelope.status is ToolResultStatus.PARTIAL
            assert envelope.data["runtime_cleanup_state"] == "pending"
            assert envelope.data["retry_termination"] is False
            assert envelope.data["retained_agent"] == "helper"
            assert manager.get_agent("helper") is None
            manager._offboard_agent_runtime_namespace.assert_not_awaited()

            child.allow_shutdown_finish.set()
            assert await manager.drain_quarantined_shutdowns() is False
        finally:
            child.allow_shutdown_finish.set()

        manager._offboard_agent_runtime_namespace.assert_awaited_once_with(child)

    @pytest.mark.asyncio
    async def test_terminate_child_flattens_grouped_retained_agents(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager.terminate_child = AsyncMock()
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        await lifecycle.register(
            child_name="helper",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        manager.terminate_child.side_effect = ExceptionGroup(
            "retained descendants",
            [
                RuntimeOffboardingRetainedError(
                    agent_name="helper",
                    agent_id="did:child",
                    runtime_path=Path("/private/helper"),
                    cause=OSError("helper-secret"),
                ),
                ExceptionGroup(
                    "nested",
                    [
                        RuntimeOffboardingRetainedError(
                            agent_name="grandchild",
                            agent_id="did:grandchild",
                            runtime_path=Path("/private/grandchild"),
                            cause=PermissionError("grandchild-secret"),
                        )
                    ],
                ),
            ],
        )
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        envelope = await feature.terminate_child(
            child_name="helper",
            offboard_runtime=True,
        )

        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.data["retained_agents"] == ["grandchild", "helper"]
        assert "retained_agent" not in envelope.data
        assert envelope.data["retained_outcome_count"] == 2
        assert envelope.data["named_child_runtime_retained"] is True
        assert envelope.data["named_child_runtime_removed"] is False
        assert envelope.data["additional_outcome_count"] == 0
        assert envelope.data["retained_cause_types"] == [
            "OSError",
            "PermissionError",
        ]
        serialized = str(envelope.to_dict())
        assert "/private" not in serialized
        assert "helper-secret" not in serialized
        assert "grandchild-secret" not in serialized

    @pytest.mark.asyncio
    async def test_terminate_child_reports_descendant_custody_truthfully(self):
        parent = _make_mock_agent("did:parent")
        manager = AgentManager()
        child = _make_mock_agent("did:child")
        grandchild = _make_mock_agent("did:grandchild")
        manager._agents.update({"Child": child, "Grandchild": grandchild})
        manager._agent_names.update(
            {child.agent_id: "Child", grandchild.agent_id: "Grandchild"}
        )
        manager._parent_children["did:parent"] = ["Child"]
        manager._parent_children[child.agent_id] = ["Grandchild"]
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        await lifecycle.register(
            child_name="Child",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        descendant_retained = RuntimeOffboardingRetainedError(
            agent_name="Grandchild",
            agent_id="did:grandchild",
            runtime_path=Path("/private/grandchild"),
            cause=OSError("descendant-secret"),
        )

        async def remove_agent(name: str, *, offboard_runtime: bool) -> bool:
            assert offboard_runtime is True
            removed = manager._agents.pop(name, None)
            if removed is None:
                return False
            manager._agent_names.pop(removed.agent_id, None)
            if name == "Grandchild":
                raise descendant_retained
            return True

        manager.remove_agent = AsyncMock(side_effect=remove_agent)
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        envelope = await feature.terminate_child(
            child_name="Child",
            offboard_runtime=True,
        )

        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.data["retained_agent"] == "Grandchild"
        assert envelope.data["retained_agents"] == ["Grandchild"]
        assert envelope.data["named_child_runtime_retained"] is False
        assert envelope.data["named_child_runtime_removed"] is True
        assert "Child" not in envelope.error
        assert "Grandchild" in envelope.error
        assert "descendant-secret" not in str(envelope.to_dict())
        assert manager.remove_agent.await_args_list == [
            (("Grandchild",), {"offboard_runtime": True}),
            (("Child",), {"offboard_runtime": True}),
        ]
        assert manager.get_agent("Child") is None
        assert manager.get_agent("Grandchild") is None
        assert manager.get_children("did:parent") == []

    @pytest.mark.asyncio
    async def test_terminate_child_flattens_retained_and_cancellation(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager.terminate_child = AsyncMock()
        lifecycle = SpawnedAgentLifecycle(manager)
        manager._lifecycle = lifecycle
        await lifecycle.register(
            child_name="helper",
            child_did="did:child",
            parent_did="did:parent",
            ttl_seconds=3600,
        )
        manager.terminate_child.side_effect = BaseExceptionGroup(
            "cancelled cleanup",
            [
                asyncio.CancelledError("private-cancel-text"),
                RuntimeOffboardingRetainedError(
                    agent_name="helper",
                    agent_id="did:child",
                    runtime_path=Path("/private/helper"),
                    cause=TimeoutError("private-timeout-text"),
                    cleanup_pending=True,
                ),
            ],
        )
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        envelope = await feature.terminate_child(
            child_name="helper",
            offboard_runtime=True,
        )

        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.data["additional_outcome_count"] == 1
        assert envelope.data["additional_outcome_types"] == ["CancelledError"]
        assert envelope.data["runtime_cleanup_pending"] is True
        assert envelope.data["runtime_cleanup_state"] == "pending"
        assert "may complete" in envelope.error
        serialized = str(envelope.to_dict())
        assert "private-cancel-text" not in serialized
        assert "private-timeout-text" not in serialized
        assert "/private/helper" not in serialized
        manager.terminate_child.assert_awaited_once_with(
            "did:parent",
            "helper",
            offboard_runtime=True,
        )

    @pytest.mark.asyncio
    async def test_terminate_child_preserves_active_cancellation_group(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager._lifecycle = None
        retained = RuntimeOffboardingRetainedError(
            agent_name="helper",
            agent_id="did:child",
            runtime_path=Path("/private/helper"),
            cause=OSError("private-custody-text"),
        )

        async def actively_cancelled(*_args, **_kwargs):
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            raise BaseExceptionGroup(
                "active cancellation and custody",
                [asyncio.CancelledError(), retained],
            )

        manager.terminate_child = AsyncMock(side_effect=actively_cancelled)
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        task = asyncio.current_task()
        assert task is not None

        try:
            with pytest.raises(BaseExceptionGroup) as raised:
                await feature.terminate_child(child_name="helper")
            assert any(
                isinstance(item, asyncio.CancelledError)
                for item in raised.value.exceptions
            )
        finally:
            while task.cancelling():
                task.uncancel()

    @pytest.mark.asyncio
    async def test_terminate_child_supports_typed_reconciliation_outcome(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager._lifecycle = None
        manager.terminate_child = AsyncMock(
            side_effect=ChildTerminationReconciliationError(
                child_name="helper",
                cause=OSError("private-reconcile-text"),
            )
        )
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        envelope = await feature.terminate_child(child_name="helper")

        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.data["runtime_offboard_requested"] is False
        assert envelope.data["runtime_offboarded"] is False
        assert envelope.data["runtime_retained"] is True
        assert envelope.data["runtime_retained_for_restart"] is True
        assert envelope.data["named_child_runtime_retained"] is True
        assert envelope.data["named_child_runtime_removed"] is False
        assert envelope.data["runtime_cleanup_state"] == "not_requested"
        assert envelope.data["tracking_reconciled"] is False
        assert envelope.data["additional_outcome_types"] == [
            "ChildTerminationReconciliationError"
        ]
        assert "private-reconcile-text" not in str(envelope.to_dict())

    @pytest.mark.asyncio
    async def test_destructive_reconciliation_partial_reports_runtime_removed(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager._lifecycle = None
        manager.terminate_child = AsyncMock(
            side_effect=ChildTerminationReconciliationError(
                child_name="helper",
                cause=OSError("private-reconcile-text"),
            )
        )
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        envelope = await feature.terminate_child(
            child_name="helper",
            offboard_runtime=True,
        )

        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.data["runtime_offboard_requested"] is True
        assert envelope.data["runtime_offboarded"] is True
        assert envelope.data["runtime_retained"] is False
        assert envelope.data["runtime_retained_for_restart"] is False
        assert envelope.data["named_child_runtime_retained"] is False
        assert envelope.data["named_child_runtime_removed"] is True
        assert envelope.data["runtime_cleanup_state"] == "removed"
        assert envelope.data["tracking_reconciled"] is False
        assert "private-reconcile-text" not in str(envelope.to_dict())
        manager.terminate_child.assert_awaited_once_with(
            parent.agent_id,
            "helper",
            offboard_runtime=True,
        )

    @pytest.mark.asyncio
    async def test_terminate_child_flattens_retained_and_reconciliation(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager._lifecycle = None
        manager.terminate_child = AsyncMock(
            side_effect=ExceptionGroup(
                "custody and reconciliation",
                [
                    RuntimeOffboardingRetainedError(
                        agent_name="grandchild",
                        agent_id="did:grandchild",
                        runtime_path=Path("/private/grandchild"),
                        cause=OSError("private-custody-text"),
                    ),
                    ChildTerminationReconciliationError(
                        child_name="helper",
                        cause=OSError("private-reconcile-text"),
                    ),
                ],
            )
        )
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        envelope = await feature.terminate_child(child_name="helper")

        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.data["retained_agent"] == "grandchild"
        assert envelope.data["runtime_offboard_requested"] is False
        assert envelope.data["runtime_offboarded"] is False
        assert envelope.data["runtime_retained_for_restart"] is True
        assert envelope.data["runtime_cleanup_state"] == "not_requested"
        assert envelope.data["named_child_runtime_retained"] is True
        assert envelope.data["named_child_runtime_removed"] is False
        assert envelope.data["tracking_reconciled"] is False
        assert envelope.data["additional_outcome_count"] == 1
        assert envelope.data["additional_outcome_types"] == [
            "ChildTerminationReconciliationError"
        ]
        serialized = str(envelope.to_dict())
        assert "/private/grandchild" not in serialized
        assert "private-custody-text" not in serialized
        assert "private-reconcile-text" not in serialized

    @pytest.mark.asyncio
    async def test_terminate_child_reraises_unsupported_grouped_failure(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager._lifecycle = None
        retained = RuntimeOffboardingRetainedError(
            agent_name="helper",
            agent_id="did:child",
            runtime_path=Path("/private/helper"),
            cause=OSError("private-custody-text"),
        )
        programmer_failure = AssertionError("programmer invariant")
        manager.terminate_child = AsyncMock(
            side_effect=BaseExceptionGroup(
                "mixed unsupported outcome",
                [retained, programmer_failure],
            )
        )
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        with pytest.raises(BaseExceptionGroup) as raised:
            await feature.terminate_child(child_name="helper")

        assert programmer_failure in raised.value.exceptions

    @pytest.mark.asyncio
    async def test_terminate_child_reraises_untyped_operational_failure(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=None)
        manager._lifecycle = None
        failure = RuntimeError("untyped lifecycle failure")
        manager.terminate_child = AsyncMock(side_effect=failure)
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        with pytest.raises(RuntimeError) as raised:
            await feature.terminate_child(child_name="helper")

        assert raised.value is failure

    @pytest.mark.asyncio
    async def test_terminate_child_falls_back_without_lifecycle(self):
        parent = _make_mock_agent("did:parent")

        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager._lifecycle = None
        manager.terminate_child = AsyncMock(return_value=True)

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.terminate_child(child_name="helper")

        assert envelope.status is ToolResultStatus.OK
        manager.terminate_child.assert_awaited_once_with("did:parent", "helper")

    @pytest.mark.asyncio
    async def test_terminate_child_false_result_remains_error(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager._lifecycle = None
        manager.terminate_child = AsyncMock(return_value=False)
        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        envelope = await feature.terminate_child(child_name="helper")

        assert envelope.status is ToolResultStatus.ERROR
        assert envelope.error == "Failed to terminate 'helper'"
        manager.terminate_child.assert_awaited_once_with("did:parent", "helper")

    @pytest.mark.asyncio
    async def test_terminate_child_not_ours(self):
        parent = _make_mock_agent("did:parent")

        manager = MagicMock()
        manager.get_children = MagicMock(return_value=[])

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        envelope = await feature.terminate_child(child_name="stranger")

        assert envelope.status is ToolResultStatus.ERROR
        assert "not a child" in envelope.error

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tasks(self):
        feature = _make_spawn_feature(manager=MagicMock())

        # Simulate a running task
        async def long_running():
            await asyncio.sleep(100)

        feature._child_tasks["helper"] = asyncio.create_task(long_running())
        feature._child_results["helper"] = {"some": "data"}

        await feature.shutdown()

        assert len(feature._child_tasks) == 0
        assert len(feature._child_results) == 0


class TestSpawnFeatureDefaultPermissions:
    """Verify default permission levels."""

    def test_spawn_agent_requires_ask(self):
        feature = _make_spawn_feature()
        assert feature.default_permissions["spawn_agent"] == "ask"

    def test_delegate_task_defaults_to_allow(self):
        feature = _make_spawn_feature()
        assert feature.default_permissions["delegate_task"] == "allow"

    def test_terminate_child_requires_approval_for_destructive_variant(self):
        feature = _make_spawn_feature()
        assert feature.default_permissions["terminate_child"] == "ask"


class TestAgentManagerSpawn:
    """Test AgentManager spawn extensions."""

    def test_parent_children_tracking_initialized(self):
        manager = AgentManager()
        assert manager._parent_children == {}
        assert manager._child_mandates == {}

    def test_get_children_empty(self):
        manager = AgentManager()
        assert manager.get_children("did:nonexistent") == []

    @pytest.mark.asyncio
    async def test_spawn_agent_creates_and_tracks(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")

        manager = AgentManager()

        # No budget here — this test covers spawn tracking, not budget
        # enforcement (which requires a funded parent wallet; see
        # test_spawn_budget_enforcement). #2113.
        mandate = SpawnMandate(
            parent_did="did:parent",
            purpose="test spawning",
            ttl_seconds=600,
        )

        async def create_and_publish(name, **_kwargs):
            # Public spawn commits its mandate only for the exact child already
            # published by create/load; this fake keeps the test on that
            # production contract.
            manager._agents[name] = child
            manager._agent_names[child.agent_id] = name
            return child

        with patch.object(manager, "create_agent", side_effect=create_and_publish):
            result = await manager.spawn_agent("helper", parent, mandate)

        assert result is child
        assert "helper" in manager.get_children("did:parent")
        assert manager.get_mandate("helper") is mandate
        assert mandate.child_did == "did:child"

    @pytest.mark.asyncio
    async def test_spawn_agent_duplicate_raises(self):
        parent = _make_mock_agent("did:parent")

        manager = AgentManager()
        manager._agents["helper"] = _make_mock_agent("did:existing")

        mandate = SpawnMandate(parent_did="did:parent", purpose="test")

        with pytest.raises(ValueError, match="already exists"):
            await manager.spawn_agent("helper", parent, mandate)

    @pytest.mark.asyncio
    async def test_failed_spawn_surfaces_live_uncommitted_child_rollback_failure(self):
        """A refused rollback cannot masquerade as a completed failed spawn."""

        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")
        child.shutdown.side_effect = RuntimeError("shutdown refused")
        manager = AgentManager()
        mandate = SpawnMandate(parent_did="did:parent", purpose="test")

        async def create_and_publish(name, **_kwargs):
            manager._agents[name] = child
            manager._agent_names[child.agent_id] = name
            return child

        manager._apply_delegated_budget = AsyncMock(
            side_effect=RuntimeError("budget setup failed")
        )
        with patch.object(manager, "create_agent", side_effect=create_and_publish):
            with pytest.raises(ExceptionGroup) as exc_info:
                await manager.spawn_agent("helper", parent, mandate)

        assert any(
            "did not remove its live routable child" in str(error)
            for error in exc_info.value.exceptions
        )
        # The failed rollback is surfaced instead of returning a child whose
        # parent edge/mandate never committed.  The admission and cap slot still
        # retire so a later explicit cleanup/retry is not itself stranded.
        assert manager.get_agent("helper") is child
        assert manager.get_children("did:parent") == []
        assert manager.get_mandate("helper") is None
        assert manager._pending_spawns == 0
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_cancelled_spawn_surfaces_live_uncommitted_child_rollback_failure(self):
        """Cancellation cannot hide a rollback that left a child routable."""

        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")
        child.shutdown.side_effect = RuntimeError("shutdown refused")
        manager = AgentManager()
        mandate = SpawnMandate(parent_did="did:parent", purpose="test")

        async def create_and_publish(name, **_kwargs):
            manager._agents[name] = child
            manager._agent_names[child.agent_id] = name
            return child

        budget_started = asyncio.Event()

        async def wait_for_cancellation(*_args, **_kwargs):
            budget_started.set()
            await asyncio.Event().wait()

        manager._apply_delegated_budget = wait_for_cancellation
        with patch.object(manager, "create_agent", side_effect=create_and_publish):
            spawn = asyncio.create_task(manager.spawn_agent("helper", parent, mandate))
            await asyncio.wait_for(budget_started.wait(), timeout=1.0)
            spawn.cancel()
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await spawn

        assert any(
            "did not remove its live routable child" in str(error)
            for error in exc_info.value.exceptions
        )
        assert any(
            isinstance(error, asyncio.CancelledError)
            for error in exc_info.value.exceptions
        )
        assert manager.get_agent("helper") is child
        assert manager._pending_spawns == 0
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_cancelled_spawn_preserves_rollback_refund_failure_group(self):
        """Final slot retirement cannot replace rollback evidence with cancellation."""

        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")
        manager = AgentManager()
        mandate = SpawnMandate(parent_did="did:parent", purpose="test")
        budget_entry = (object(), object())
        budget_started = asyncio.Event()

        async def create_and_publish(name, **_kwargs):
            manager._agents[name] = child
            manager._agent_names[child.agent_id] = name
            return child

        async def allocate_then_wait(name, *_args, **_kwargs):
            manager._child_budgets[name] = budget_entry
            budget_started.set()
            await asyncio.Event().wait()

        async def fail_refund(name: str) -> bool:
            assert name == "helper"
            assert manager._child_budgets[name] is budget_entry
            raise RuntimeError("rollback refund failed")

        manager._apply_delegated_budget = allocate_then_wait
        manager._release_child_budget_cancellation_safe = fail_refund
        with patch.object(manager, "create_agent", side_effect=create_and_publish):
            spawn = asyncio.create_task(manager.spawn_agent("helper", parent, mandate))
            await asyncio.wait_for(budget_started.wait(), timeout=1.0)
            spawn.cancel()
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await asyncio.wait_for(spawn, timeout=1.0)

        assert any(
            isinstance(error, asyncio.CancelledError)
            for error in exc_info.value.exceptions
        )
        assert any(
            "rollback refund failed" in str(error)
            for error in exc_info.value.exceptions
        )
        assert manager.get_agent("helper") is None
        assert manager._child_budgets["helper"] is budget_entry
        assert manager._pending_spawns == 0
        assert manager._agent_operations == {}

    @pytest.mark.asyncio
    async def test_terminate_child(self):
        child = _make_mock_agent("did:child")

        manager = AgentManager()
        manager._agents["helper"] = child
        manager._agent_names["did:child"] = "helper"
        manager._parent_children["did:parent"] = ["helper"]
        manager._child_mandates["helper"] = SpawnMandate(
            parent_did="did:parent", purpose="test"
        )

        removed = await manager.terminate_child("did:parent", "helper")

        assert removed is True
        assert manager.get_agent("helper") is None
        assert "helper" not in manager.get_children("did:parent")
        assert manager.get_mandate("helper") is None
        child.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_terminate_child_default_retains_hosted_runtime_for_restart(
        self,
        tmp_path,
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        child = _make_mock_agent("did:child:retained")
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(child.agent_id),
        )
        prepare_isolated_runtime_namespace(
            scope,
            child.agent_id,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        credential = scope.path / "feature_venvs" / "feature-safe" / "credential"
        credential.write_text("retain-on-stop")
        child.isolated_runtime_scope = scope
        manager._agents["helper"] = child
        manager._agent_names[child.agent_id] = "helper"
        manager._parent_children["did:parent"] = ["helper"]

        assert await manager.terminate_child("did:parent", "helper") is True

        assert credential.read_text() == "retain-on-stop"
        assert scope.path.is_dir()

    @pytest.mark.asyncio
    async def test_terminate_child_explicit_offboard_deletes_hosted_runtime(
        self,
        tmp_path,
    ):
        manager = AgentManager(base_data_dir=tmp_path)
        child = _make_mock_agent("did:child:offboard")
        scope = resolve_isolated_runtime_namespace(
            manager._isolated_runtime_root,
            derive_isolated_runtime_namespace(child.agent_id),
        )
        prepare_isolated_runtime_namespace(
            scope,
            child.agent_id,
            relative_directories=(("feature_venvs", "feature-safe"),),
        )
        (scope.path / "feature_venvs" / "feature-safe" / "credential").write_text(
            "delete-on-offboard"
        )
        child.isolated_runtime_scope = scope
        manager._agents["helper"] = child
        manager._agent_names[child.agent_id] = "helper"
        manager._parent_children["did:parent"] = ["helper"]

        assert await manager.terminate_child(
            "did:parent",
            "helper",
            offboard_runtime=True,
        ) is True

        assert not scope.path.exists()

    @pytest.mark.asyncio
    async def test_terminate_child_not_found(self):
        manager = AgentManager()
        removed = await manager.terminate_child("did:parent", "ghost")
        assert removed is False

    @pytest.mark.asyncio
    async def test_terminate_children_cascading(self):
        """Terminating children should cascade to grandchildren."""
        child = _make_mock_agent("did:child")
        grandchild = _make_mock_agent("did:grandchild")

        manager = AgentManager()
        manager._agents["child1"] = child
        manager._agents["grandchild1"] = grandchild
        manager._agent_names["did:child"] = "child1"
        manager._agent_names["did:grandchild"] = "grandchild1"

        # parent -> child1 -> grandchild1
        manager._parent_children["did:parent"] = ["child1"]
        manager._parent_children["did:child"] = ["grandchild1"]
        manager._child_mandates["child1"] = SpawnMandate(
            parent_did="did:parent", purpose="child"
        )
        manager._child_mandates["grandchild1"] = SpawnMandate(
            parent_did="did:child", purpose="grandchild"
        )

        count = await manager.terminate_children("did:parent")

        assert count == 1  # 1 direct child terminated
        assert manager.get_agent("child1") is None
        assert manager.get_agent("grandchild1") is None
        assert manager.get_children("did:parent") == []
        assert manager.get_children("did:child") == []
        child.shutdown.assert_awaited_once()
        grandchild.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_all_clears_tracking(self):
        manager = AgentManager()
        manager._parent_children["did:p"] = ["c1"]
        manager._child_mandates["c1"] = SpawnMandate(parent_did="did:p", purpose="test")

        agent = _make_mock_agent("did:c1")
        manager._agents["c1"] = agent
        manager._agent_names["did:c1"] = "c1"

        await manager.shutdown_all()

        assert manager._parent_children == {}
        assert manager._child_mandates == {}
        assert len(manager._agents) == 0


class TestSpawnLifecycle:
    """End-to-end lifecycle: spawn → delegate → get result → terminate."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")
        child.process_input = AsyncMock(return_value="result: 42")

        manager = MagicMock()
        manager.spawn_agent = AsyncMock(return_value=child)
        manager.get_agent = MagicMock(return_value=child)
        manager.get_children = MagicMock(return_value=["worker"])
        manager.terminate_child = AsyncMock(return_value=True)

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)

        # 1. Spawn
        spawn_envelope = await feature.spawn_agent(name="worker", purpose="compute")
        assert spawn_envelope.status is ToolResultStatus.OK

        # 2. Delegate
        delegate_envelope = await feature.delegate_task(
            child_name="worker", task="compute 6*7"
        )
        assert delegate_envelope.status is ToolResultStatus.OK

        # 3. Wait for result
        await asyncio.sleep(0.2)

        # 4. Get result
        get_envelope = await feature.get_child_result(child_name="worker")
        assert get_envelope.status is ToolResultStatus.OK
        assert get_envelope.data["ready"] is True
        assert get_envelope.data["result"] == "result: 42"

        # 5. Terminate
        term_envelope = await feature.terminate_child(child_name="worker")
        assert term_envelope.status is ToolResultStatus.OK
