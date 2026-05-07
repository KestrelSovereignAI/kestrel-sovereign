"""Tests for kestrel_sovereign.testing module (MockAgent, FeatureTestCase, fixtures)."""

import pytest
from unittest.mock import MagicMock

from kestrel_sovereign.testing import MockAgent, FeatureTestCase
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sovereign.privacy import PrivacyMode


# ---------------------------------------------------------------------------
# Test Feature — a minimal feature for exercising MockAgent
# ---------------------------------------------------------------------------


class SampleFeature(Feature):
    """Minimal feature for testing."""

    def __init__(self, agent):
        super().__init__(agent)
        self.initialized = False

    @property
    def tool_description(self) -> str:
        return "Sample feature for tests"

    async def initialize(self):
        self.initialized = True

    @tool("greet", "Greet a user", category=ToolCategory.UTILITY)
    async def greet(self, name: str = "World"):
        """Greet someone.

        Args:
            name: The name to greet.
        """
        return {"success": True, "message": f"Hello, {name}!"}


class StorageFeature(Feature):
    """Feature that requires storage access."""

    @property
    def tool_description(self) -> str:
        return "Feature that uses storage"

    async def initialize(self):
        pass

    @tool("store_note", "Store a note in the graph", category=ToolCategory.MEMORY)
    async def store_note(self, text: str = ""):
        """Store a note.

        Args:
            text: Note content.
        """
        if self.agent.storage is None:
            return {"success": False, "error": "No storage available"}
        import uuid
        node = GraphNode(node_id=str(uuid.uuid4()), node_type="note", label=text, properties={"text": text})
        await self.agent.storage.graph.add_node(node)
        return {"success": True}


# ---------------------------------------------------------------------------
# MockAgent tests
# ---------------------------------------------------------------------------


class TestMockAgent:
    """Test MockAgent construction and service stubs."""

    def test_default_construction(self):
        agent = MockAgent()
        assert agent.did == "did:test:mock-agent"
        assert agent.agent_id == "did:test:mock-agent"
        assert agent.features == {}
        assert agent.storage is None
        assert agent.wallet is None

    def test_custom_did(self):
        agent = MockAgent(did="did:test:custom")
        assert agent.did == "did:test:custom"

    def test_privacy_mode(self):
        agent = MockAgent(privacy_mode=PrivacyMode.EPHEMERAL)
        assert agent.privacy_agent.privacy_mode == PrivacyMode.EPHEMERAL

    async def test_llm_service_default_response(self):
        agent = MockAgent(default_llm_response="test response")
        resp = await agent.llm_service.get_response([{"role": "user", "content": "hi"}])
        assert resp.text == "test response"

    async def test_llm_service_queued_responses(self):
        agent = MockAgent(llm_responses=["first", "second"])
        r1 = await agent.llm_service.get_response([])
        r2 = await agent.llm_service.get_response([])
        r3 = await agent.llm_service.get_response([])  # falls back to default
        assert r1.text == "first"
        assert r2.text == "second"
        assert r3.text == "Mock LLM response"

    async def test_llm_service_call_history(self):
        agent = MockAgent()
        messages = [{"role": "user", "content": "hello"}]
        await agent.llm_service.get_response(messages, temperature=0.5)
        assert len(agent.llm_service._call_history) == 1
        assert agent.llm_service._call_history[0]["messages"] == messages
        assert agent.llm_service._call_history[0]["temperature"] == 0.5

    async def test_llm_service_streaming(self):
        agent = MockAgent(llm_responses=["streamed"])
        stream = await agent.llm_service.get_streaming_response([])
        chunks = [chunk async for chunk in stream]
        assert len(chunks) == 1
        assert chunks[0].text == "streamed"

    def test_hooks_manager_register(self):
        agent = MockAgent()
        hook = MagicMock()
        agent.hooks_manager.register(hook)
        assert hook in agent.hooks_manager.registered_hooks

    def test_hooks_manager_unregister(self):
        agent = MockAgent()
        hook = MagicMock()
        agent.hooks_manager.register(hook)
        agent.hooks_manager.unregister(hook)
        assert hook not in agent.hooks_manager.registered_hooks

    async def test_hooks_manager_execute(self):
        agent = MockAgent()
        result = await agent.hooks_manager.execute_hooks("PRE_TOOL_USE")
        assert result.decision == "ALLOW"
        assert len(agent.hooks_manager.executed_hooks) == 1

    async def test_create_with_storage(self):
        agent = await MockAgent.create()
        assert agent.storage is not None
        assert agent.storage._initialized
        await agent.shutdown()

    async def test_feature_instantiation(self):
        """MockAgent can instantiate any Feature subclass without real services."""
        agent = MockAgent()
        feature = SampleFeature(agent)
        await feature.initialize()
        assert feature.initialized
        result = await feature.greet(name="Kestrel")
        assert result == {"success": True, "message": "Hello, Kestrel!"}

    async def test_feature_with_storage(self):
        """Features can use in-memory storage from MockAgent.create()."""
        agent = await MockAgent.create()
        feature = StorageFeature(agent)
        await feature.initialize()
        result = await feature.store_note(text="test note")
        assert result["success"]
        await agent.shutdown()


# ---------------------------------------------------------------------------
# FeatureTestCase tests
# ---------------------------------------------------------------------------


class TestSampleFeature(FeatureTestCase):
    """Demonstrate FeatureTestCase usage with SampleFeature."""

    feature_class = SampleFeature

    async def test_feature_initialized(self):
        assert self.feature.initialized

    async def test_feature_in_agent(self):
        assert "SampleFeature" in self.agent.features

    async def test_greet_default(self):
        result = await self.feature.greet()
        assert result == {"success": True, "message": "Hello, World!"}

    async def test_greet_custom(self):
        result = await self.feature.greet(name="Tester")
        assert result == {"success": True, "message": "Hello, Tester!"}

    async def test_tool_discovery(self):
        tools = self.feature.get_tools()
        tool_names = [t.schema.name for t in tools]
        assert "greet" in tool_names


class TestStorageFeature(FeatureTestCase):
    """Demonstrate FeatureTestCase with storage-dependent feature."""

    feature_class = StorageFeature
    use_storage = True

    async def test_store_note(self):
        result = await self.feature.store_note(text="hello world")
        assert result["success"]


class TestFeatureTestCaseWithLLMResponses(FeatureTestCase):
    """Demonstrate FeatureTestCase with custom LLM responses."""

    feature_class = SampleFeature
    llm_responses = ["custom response"]
    default_llm_response = "fallback"

    async def test_llm_responses_configured(self):
        resp = await self.agent.llm_service.get_response([])
        assert resp.text == "custom response"
        # Now falls back to default
        resp2 = await self.agent.llm_service.get_response([])
        assert resp2.text == "fallback"


# ---------------------------------------------------------------------------
# Pytest fixture tests
# ---------------------------------------------------------------------------


class TestPytestFixtures:
    """Test that the pytest fixtures work correctly."""

    async def test_mock_agent_fixture(self, mock_agent):
        assert mock_agent.did == "did:test:mock-agent"
        feature = SampleFeature(mock_agent)
        await feature.initialize()
        assert feature.initialized

    async def test_mock_agent_with_storage_fixture(self, mock_agent_with_storage):
        assert mock_agent_with_storage.storage is not None
        feature = StorageFeature(mock_agent_with_storage)
        await feature.initialize()
        result = await feature.store_note(text="fixture test")
        assert result["success"]

    async def test_mock_agent_factory_fixture(self, mock_agent_factory):
        agent1 = mock_agent_factory(did="did:test:one")
        agent2 = mock_agent_factory(
            did="did:test:two",
            llm_responses=["hello"],
        )
        assert agent1.did == "did:test:one"
        assert agent2.did == "did:test:two"
        resp = await agent2.llm_service.get_response([])
        assert resp.text == "hello"
