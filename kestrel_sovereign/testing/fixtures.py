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
import pytest_asyncio

from kestrel_sovereign.testing.mock_agent import MockAgent


@pytest_asyncio.fixture
async def native_tool_effect_agent(tmp_path, monkeypatch):
    """Create partial native Agents for governed effect component tests.

    SQLite, journal keys, privacy, hooks and request lifecycle are real. No
    READY publication, model response or provider acceptance is fabricated.
    Each created Agent owns only its test directory and native resources. A
    supplied connected backend transfers ownership to that Agent, allowing the
    same primary SQLite/PostgreSQL store to serve journal and task contracts.
    An explicitly borrowed AsyncDatabase retains its caller's ownership.
    """
    from contextlib import AsyncExitStack
    from uuid import uuid4
    from cryptography.fernet import Fernet
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.privacy.feature import PrivacyAgent
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm import service as service_module
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from kestrel_sovereign.tool_effect_store import ToolEffectStore

    monkeypatch.setenv('KESTREL_DATA_KEY', Fernet.generate_key().decode())
    # Explicitly unconfigured real provider service: no ambient operator
    # routes or credentials are inputs to this component fixture.
    monkeypatch.setattr(service_module, 'load_section', lambda _section: {})
    async with AsyncExitStack() as resources:
        async def create(*, did, connected_backend=None, borrowed_database=None):
            if connected_backend is not None and borrowed_database is not None:
                raise ValueError("A native test Agent must have one primary database owner")
            root = tmp_path / ('native-tool-effect-' + uuid4().hex)
            root.mkdir()
            if borrowed_database is not None:
                database = borrowed_database
            else:
                database = (await AsyncDatabase.sqlite(str(root / 'primary.db'))
                    if connected_backend is None else
                    await AsyncDatabase.from_connected_backend(connected_backend))
                resources.push_async_callback(database.close)
            service = service_module.LLMService(usage_db=database, agent_data_dir=root)
            resources.push_async_callback(service.close)
            agent = KestrelAgent(did=did, storage_path=str(root / 'agent.db'),
                llm_service=service, allowed_features=[], sync_enabled=False)
            storage = AsyncStorage(backend=database.backend)
            storage.db = database
            agent.storage = agent._raw_storage = storage
            agent.privacy_agent = PrivacyAgent(storage)
            journal = ToolEffectStore(database, scope=NativeRecipientScope(did))
            await journal.initialize()
            resources.push_async_callback(journal.close)
            agent.tool_effect_store = journal
            assert not agent._runtime_publication_ready.is_set()
            return agent

        yield create


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
