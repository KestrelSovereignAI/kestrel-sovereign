"""
Pytest fixtures for Kestrel feature testing.

These fixtures provide lightweight MockAgent instances for feature tests.
Feature packages can use them by depending on kestrel-sovereign[test].

Usage in conftest.py or test files:
    from kestrel_sovereign.testing.fixtures import mock_agent, mock_agent_with_storage

Or register as a pytest plugin in conftest.py:
    pytest_plugins = ["kestrel_sovereign.testing.fixtures"]

Then use fixtures directly:
    async def test_my_feature(mock_agent):
        feature = MyFeature(mock_agent)
        await feature.initialize()
        ...
"""

import pytest

from kestrel_sovereign.testing.mock_agent import MockAgent


@pytest.fixture
def mock_agent():
    """
    Provide a MockAgent without storage.

    Use this for features that don't need database access.
    The agent has stubbed llm_service, hooks_manager, privacy_agent,
    and an empty features dict.

    Usage:
        async def test_my_feature(mock_agent):
            feature = MyFeature(mock_agent)
            await feature.initialize()
            result = await feature.some_tool()
            assert result["success"]
    """
    return MockAgent()


@pytest.fixture
async def mock_agent_with_storage():
    """
    Provide a MockAgent with in-memory SQLite storage.

    Use this for features that need real database operations (graph store,
    conversation history, file storage, RAG).

    Usage:
        async def test_feature_with_storage(mock_agent_with_storage):
            feature = MyFeature(mock_agent_with_storage)
            await feature.initialize()
            # feature can read/write to agent.storage
    """
    agent = await MockAgent.create()
    yield agent
    await agent.shutdown()


@pytest.fixture
def mock_agent_factory():
    """
    Factory fixture for creating customized MockAgents.

    Use this when you need multiple agents or agents with specific
    configurations in a single test.

    Usage:
        async def test_with_custom_agent(mock_agent_factory):
            agent = mock_agent_factory(
                llm_responses=["Hello!", "Goodbye!"],
                did="did:test:custom",
            )
            feature = MyFeature(agent)
            await feature.initialize()
            # First LLM call returns "Hello!", second returns "Goodbye!"
    """
    agents = []

    def _create(**kwargs):
        agent = MockAgent(**kwargs)
        agents.append(agent)
        return agent

    yield _create

    # No async cleanup needed for non-storage agents
