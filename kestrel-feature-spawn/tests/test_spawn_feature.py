"""Unit tests for SpawnFeature and AgentManager spawn extensions."""

import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from kestrel_feature_spawn.spawn.feature import SpawnFeature
from kestrel_sovereign.rookery.agent_manager import AgentManager
from kestrel_feature_spawn.spawn.mandate import SpawnMandate


def _make_mock_agent(agent_id: str = "did:pkh:eip155:1:0xPARENT"):
    """Create a mock KestrelAgent."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()
    agent.process_input = AsyncMock(return_value="task completed")
    agent._private_key = None  # No signing in unit tests
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
        result = await feature.list_children()
        assert result["children"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_delegate_without_children(self):
        feature = _make_spawn_feature(manager=MagicMock())
        result = await feature.delegate_task(child_name="child1", task="do stuff")
        assert result["delegated"] is False

    @pytest.mark.asyncio
    async def test_terminate_without_manager(self):
        feature = _make_spawn_feature(manager=None)
        result = await feature.terminate_child(child_name="child1")
        assert result["terminated"] is False


class TestSpawnFeatureWithManager:
    """Test SpawnFeature tools with a mock AgentManager."""

    @pytest.mark.asyncio
    async def test_spawn_agent_success(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")

        manager = MagicMock()
        manager.spawn_agent = AsyncMock(return_value=child)

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        result = await feature.spawn_agent(
            name="helper",
            purpose="assist with research",
            budget=10.0,
            ttl=1800,
            constraints="max_tokens=1000,no_web",
            features="memory,web_search",
        )

        assert result["spawned"] is True
        assert result["child_name"] == "helper"
        assert result["child_did"] == "did:child"

        # Verify mandate was constructed correctly
        call_args = manager.spawn_agent.call_args
        mandate = call_args.kwargs["mandate"]
        assert mandate.parent_did == "did:parent"
        assert mandate.purpose == "assist with research"
        assert mandate.budget_allocation == 10.0
        assert mandate.ttl_seconds == 1800
        assert mandate.additional_constraints == {"max_tokens": "1000", "no_web": "true"}
        assert mandate.features_allowed == ["memory", "web_search"]

    @pytest.mark.asyncio
    async def test_spawn_agent_failure(self):
        parent = _make_mock_agent("did:parent")
        manager = MagicMock()
        manager.spawn_agent = AsyncMock(side_effect=ValueError("already exists"))

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        result = await feature.spawn_agent(name="dup", purpose="test")

        assert result["spawned"] is False
        assert "already exists" in result["error"]

    @pytest.mark.asyncio
    async def test_list_children(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")

        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.get_agent = MagicMock(return_value=child)

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        result = await feature.list_children()

        assert result["count"] == 1
        assert result["children"][0]["name"] == "helper"
        assert result["children"][0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_delegate_task_success(self):
        parent = _make_mock_agent("did:parent")
        child = _make_mock_agent("did:child")

        manager = MagicMock()
        manager.get_agent = MagicMock(return_value=child)
        manager.get_children = MagicMock(return_value=["helper"])

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        result = await feature.delegate_task(child_name="helper", task="analyze data")

        assert result["delegated"] is True
        assert result["child_name"] == "helper"

        # Wait briefly for the async task to start
        await asyncio.sleep(0.1)
        assert "helper" in feature._child_tasks

    @pytest.mark.asyncio
    async def test_delegate_task_not_our_child(self):
        parent = _make_mock_agent("did:parent")
        other = _make_mock_agent("did:other")

        manager = MagicMock()
        manager.get_agent = MagicMock(return_value=other)
        manager.get_children = MagicMock(return_value=[])  # not our child

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        result = await feature.delegate_task(child_name="stranger", task="hack")

        assert result["delegated"] is False
        assert "not a child" in result["error"]

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
        await asyncio.sleep(0.2)  # Let the task complete

        result = await feature.get_child_result(child_name="helper")
        assert result["ready"] is True
        assert result["result"] == "analysis complete: 42"

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

        result = await feature.get_child_result(child_name="helper")
        assert result["ready"] is False
        assert "still running" in result["note"]

        # Clean up
        feature._child_tasks["helper"].cancel()

    @pytest.mark.asyncio
    async def test_get_child_result_no_task(self):
        feature = _make_spawn_feature(manager=MagicMock())
        result = await feature.get_child_result(child_name="nobody")
        assert result["ready"] is False

    @pytest.mark.asyncio
    async def test_terminate_child_success(self):
        parent = _make_mock_agent("did:parent")

        manager = MagicMock()
        manager.get_children = MagicMock(return_value=["helper"])
        manager.terminate_child = AsyncMock(return_value=True)

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        result = await feature.terminate_child(child_name="helper")

        assert result["terminated"] is True
        manager.terminate_child.assert_awaited_once_with("did:parent", "helper")

    @pytest.mark.asyncio
    async def test_terminate_child_not_ours(self):
        parent = _make_mock_agent("did:parent")

        manager = MagicMock()
        manager.get_children = MagicMock(return_value=[])

        feature = _make_spawn_feature(parent_agent=parent, manager=manager)
        result = await feature.terminate_child(child_name="stranger")

        assert result["terminated"] is False
        assert "not a child" in result["error"]

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

    def test_terminate_child_defaults_to_allow(self):
        feature = _make_spawn_feature()
        assert feature.default_permissions["terminate_child"] == "allow"


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

        mandate = SpawnMandate(
            parent_did="did:parent",
            purpose="test spawning",
            budget_allocation=5.0,
            ttl_seconds=600,
        )

        with patch.object(manager, "create_agent", new_callable=AsyncMock, return_value=child):
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
    async def test_terminate_child(self):
        parent = _make_mock_agent("did:parent")
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
    async def test_terminate_child_not_found(self):
        manager = AgentManager()
        removed = await manager.terminate_child("did:parent", "ghost")
        assert removed is False

    @pytest.mark.asyncio
    async def test_terminate_children_cascading(self):
        """Terminating children should cascade to grandchildren."""
        parent = _make_mock_agent("did:parent")
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
        spawn_result = await feature.spawn_agent(name="worker", purpose="compute")
        assert spawn_result["spawned"] is True

        # 2. Delegate
        delegate_result = await feature.delegate_task(
            child_name="worker", task="compute 6*7"
        )
        assert delegate_result["delegated"] is True

        # 3. Wait for result
        await asyncio.sleep(0.2)

        # 4. Get result
        get_result = await feature.get_child_result(child_name="worker")
        assert get_result["ready"] is True
        assert get_result["result"] == "result: 42"

        # 5. Terminate
        term_result = await feature.terminate_child(child_name="worker")
        assert term_result["terminated"] is True
