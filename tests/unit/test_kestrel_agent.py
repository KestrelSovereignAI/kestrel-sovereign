"""
Unit tests for KestrelAgent core methods.

Tests the main agent orchestrator without requiring running LLM services.
Uses mocks for LLMService, AsyncStorage, and external dependencies.
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from datetime import datetime, timezone

from kestrel_sovereign.kestrel_agent import KestrelAgent, _load_prompt_file
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


# =============================================================================
# Mock Classes
# =============================================================================

class MockDB:
    """Mock database for testing."""

    def __init__(self):
        self.data = {}
        self.executed_queries = []

    async def fetchall(self, query: str, params: tuple = None):
        """Mock fetchall."""
        self.executed_queries.append((query, params))
        key = (params[0], params[1]) if params and len(params) >= 2 else None
        if key and key in self.data:
            return [(self.data[key],)]
        return []

    async def execute(self, query: str, params: tuple = None):
        """Mock execute."""
        self.executed_queries.append((query, params))
        if params and len(params) >= 4:
            key = (params[0], params[1])
            self.data[key] = params[2]

    async def executemany(self, query: str, params: list):
        """Mock executemany."""
        self.executed_queries.append((query, params))


class MockAsyncStorage:
    """Mock AsyncStorage for testing."""

    def __init__(self):
        self.initialized = False
        self.closed = False
        self.db = MockDB()
        self.nodes = {}
        self.conversations = []

    async def initialize(self):
        """Mock initialize."""
        self.initialized = True

    async def close(self):
        """Mock close."""
        self.closed = True

    async def get_node(self, node_id: str):
        """Mock get_node."""
        return self.nodes.get(node_id)

    async def add_node(self, node):
        """Mock add_node."""
        self.nodes[node.node_id] = node

    async def get_conversation_history(self, limit: int = 50, session_id: str = None):
        """Mock get conversation history."""
        return self.conversations[-limit:] if self.conversations else []

    async def add_conversation(self, role: str, content: str, session_id: str = None):
        """Mock add conversation."""
        self.conversations.append({
            "role": role,
            "content": content,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, responses=None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.providers = [{"name": "mock", "model": "mock-model"}]
        self.generate_calls = []

    async def generate(self, messages, temperature=None):
        """Mock generate."""
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        self.generate_calls.append(messages)

        class MockResponse:
            content = response

        return MockResponse()

    async def generate_with_messages(self, messages, **kwargs):
        """Mock generate_with_messages."""
        response_text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        self.generate_calls.append(messages)

        # Return LLMResponse object (no tool calls)
        return LLMResponse(content=response_text, tool_calls=None)

    def get_model_preference(self):
        """Mock get_model_preference."""
        return {"provider": "mock", "model": "mock-model"}

    def get_default_model(self):
        """Mock get_default_model."""
        return "mock-model"

    async def get_audit_response(self, text: str):
        """Mock audit response."""
        return {
            "integrity_score": 95,
            "reasoning": "Mock audit passed"
        }

    async def close(self):
        """Mock close."""
        pass


class MockTaskManager:
    """Mock TaskManager for testing."""

    def __init__(self):
        self.initialized = False
        self.closed = False

    async def initialize(self):
        """Mock initialize."""
        self.initialized = True

    async def close(self):
        """Mock close."""
        self.closed = True

    def register_agent(self, agent_card, handler, command_prefixes=None):
        """Mock register_agent."""
        pass


class MockWalletAgent:
    """Mock WalletAgent for testing."""

    def __init__(self, agent_id, initial_balance, db_path):
        self.agent_id = agent_id
        self.balance = initial_balance

    async def initialize(self):
        """Mock initialize."""
        pass

    def get_balance(self):
        """Mock get_balance."""
        from decimal import Decimal
        return self.balance

    def can_afford_audit(self, cost):
        """Mock can_afford_audit."""
        from decimal import Decimal
        return self.balance >= Decimal(str(cost))


class MockMemorySystem:
    """Mock MemorySystem for testing."""

    def __init__(self, storage, agent_id):
        self.storage = storage
        self.agent_id = agent_id
        self.retriever = MagicMock()

    async def initialize(self):
        """Mock initialize."""
        pass


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_storage():
    """Create a mock AsyncStorage."""
    return MockAsyncStorage()


@pytest.fixture
def mock_llm():
    """Create a mock LLM service."""
    return MockLLMService()


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    db_path = tmp_path / "test_agent.db"
    return str(db_path)


@pytest.fixture
def mock_agent_did():
    """Return a mock agent DID."""
    return "did:pkh:eip155:1:0xTestAgent123"


# =============================================================================
# Tests for _load_prompt_file()
# =============================================================================

class TestLoadPromptFile:
    """Tests for _load_prompt_file function."""

    def test_load_existing_file(self, tmp_path):
        """Test loading a prompt from an existing file."""
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text("Test prompt content", encoding="utf-8")

        result = _load_prompt_file(prompt_file, fallback="fallback")
        assert result == "Test prompt content"

    def test_load_missing_file(self, tmp_path):
        """Test loading a prompt when file doesn't exist."""
        prompt_file = tmp_path / "nonexistent.txt"

        result = _load_prompt_file(prompt_file, fallback="fallback content")
        assert result == "fallback content"

    def test_load_file_read_error(self, tmp_path):
        """Test handling file read errors."""
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text("content")

        # Make file unreadable (Unix-only)
        if os.name != 'nt':
            os.chmod(prompt_file, 0o000)

            result = _load_prompt_file(prompt_file, fallback="fallback")
            assert result == "fallback"

            # Restore permissions for cleanup
            os.chmod(prompt_file, 0o644)


# =============================================================================
# Tests for KestrelAgent Initialization
# =============================================================================

class TestKestrelAgentInit:
    """Tests for KestrelAgent initialization."""

    def test_init_with_required_params(self, mock_agent_did, temp_db_path, mock_llm):
        """Test initialization with required parameters."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path,
            llm_service=mock_llm
        )

        assert agent.did == mock_agent_did
        assert agent.storage_path == temp_db_path
        assert agent.llm_service == mock_llm
        assert agent._privacy_mode == PrivacyMode.NORMAL

    def test_init_with_privacy_mode(self, mock_agent_did, temp_db_path):
        """Test initialization with custom privacy mode."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path,
            privacy_mode=PrivacyMode.ISOLATED
        )

        assert agent._privacy_mode == PrivacyMode.ISOLATED

    def test_init_with_database_url(self, mock_agent_did):
        """Test initialization with PostgreSQL database URL."""
        agent = KestrelAgent(
            did=mock_agent_did,
            database_url="postgresql://user:pass@localhost/kestrel",
            db_backend="postgres"
        )

        assert agent._db_backend == "postgres"
        assert agent._database_url == "postgresql://user:pass@localhost/kestrel"

    def test_init_defaults_to_sqlite(self, mock_agent_did, temp_db_path):
        """Test that initialization defaults to SQLite backend."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        assert agent._db_backend == "sqlite"

    def test_init_creates_llm_service_if_not_provided(self, mock_agent_did, temp_db_path):
        """Test that LLMService is created if not provided."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        assert agent.llm_service is not None
        assert hasattr(agent.llm_service, 'generate_with_messages')


# =============================================================================
# Tests for KestrelAgent.initialize()
# =============================================================================

class TestKestrelAgentInitialize:
    """Tests for async initialization."""

    @pytest.mark.asyncio
    async def test_initialize_sets_up_storage(self, mock_agent_did, temp_db_path, mock_llm):
        """Test that initialize() sets up storage."""
        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            mock_storage_instance = MockAsyncStorage()
            MockStorage.return_value = mock_storage_instance

            agent = KestrelAgent(
                did=mock_agent_did,
                storage_path=temp_db_path,
                llm_service=mock_llm
            )

            # Mock feature discovery to prevent real feature loading
            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent', return_value=MockWalletAgent(mock_agent_did, 100, temp_db_path)):
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem', return_value=MockMemorySystem(mock_storage_instance, mock_agent_did)):
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager', return_value=MockTaskManager()):
                                with patch('kestrel_sovereign.kestrel_agent.CommandHandler'):
                                    with patch('kestrel_sovereign.kestrel_agent.ContextBuilder'):
                                        with patch('kestrel_sovereign.kestrel_agent.ContextManager'):
                                            with patch('kestrel_sovereign.kestrel_agent.BootstrapService'):
                                                await agent.initialize()

            assert agent._raw_storage is not None
            assert agent.storage is not None
            assert mock_storage_instance.initialized

    @pytest.mark.asyncio
    async def test_initialize_creates_agent_node_if_missing(self, mock_agent_did, temp_db_path, mock_llm):
        """Test that initialize creates agent node if it doesn't exist."""
        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            mock_storage_instance = MockAsyncStorage()
            MockStorage.return_value = mock_storage_instance

            agent = KestrelAgent(
                did=mock_agent_did,
                storage_path=temp_db_path,
                llm_service=mock_llm
            )

            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent', return_value=MockWalletAgent(mock_agent_did, 100, temp_db_path)):
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem', return_value=MockMemorySystem(mock_storage_instance, mock_agent_did)):
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager', return_value=MockTaskManager()):
                                with patch('kestrel_sovereign.kestrel_agent.CommandHandler'):
                                    with patch('kestrel_sovereign.kestrel_agent.ContextBuilder'):
                                        with patch('kestrel_sovereign.kestrel_agent.ContextManager'):
                                            with patch('kestrel_sovereign.kestrel_agent.BootstrapService'):
                                                await agent.initialize()

            # Check that agent node was created
            assert mock_agent_did in mock_storage_instance.nodes


# =============================================================================
# Tests for Privacy Mode
# =============================================================================

class TestPrivacyMode:
    """Tests for privacy mode getter/setter."""

    def test_privacy_mode_getter(self, mock_agent_did, temp_db_path):
        """Test privacy mode getter."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path,
            privacy_mode=PrivacyMode.ISOLATED
        )

        assert agent.privacy_mode == PrivacyMode.ISOLATED

    @pytest.mark.asyncio
    async def test_privacy_mode_setter(self, mock_agent_did, temp_db_path, mock_llm):
        """Test privacy mode setter."""
        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            mock_storage_instance = MockAsyncStorage()
            MockStorage.return_value = mock_storage_instance

            agent = KestrelAgent(
                did=mock_agent_did,
                storage_path=temp_db_path,
                llm_service=mock_llm,
                privacy_mode=PrivacyMode.NORMAL
            )

            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent', return_value=MockWalletAgent(mock_agent_did, 100, temp_db_path)):
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem', return_value=MockMemorySystem(mock_storage_instance, mock_agent_did)):
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager', return_value=MockTaskManager()):
                                with patch('kestrel_sovereign.kestrel_agent.CommandHandler'):
                                    with patch('kestrel_sovereign.kestrel_agent.ContextBuilder'):
                                        with patch('kestrel_sovereign.kestrel_agent.ContextManager'):
                                            with patch('kestrel_sovereign.kestrel_agent.BootstrapService'):
                                                await agent.initialize()

            # Change privacy mode
            agent.set_privacy_mode(PrivacyMode.EPHEMERAL)

            assert agent._privacy_mode == PrivacyMode.EPHEMERAL
            assert agent.privacy_mode == PrivacyMode.EPHEMERAL


# =============================================================================
# Tests for Model Selection
# =============================================================================

class TestModelSelection:
    """Tests for model selection methods."""

    @pytest.mark.asyncio
    async def test_set_model(self, mock_agent_did, temp_db_path):
        """Test setting a model."""
        mock_model_agent = MagicMock()
        mock_model_agent.set_model_preference.return_value = "Model set to gpt-5"

        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )
        agent.model_agent = mock_model_agent

        result = agent.set_model("gpt-5")

        assert "Model set" in result
        mock_model_agent.set_model_preference.assert_called_once_with("gpt-5")

    def test_get_current_model_with_preference(self, mock_agent_did, temp_db_path):
        """Test getting current model when preference is set."""
        mock_llm = MockLLMService()
        mock_llm.get_model_preference = lambda: {"provider": "openai", "model": "gpt-5"}

        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path,
            llm_service=mock_llm
        )

        result = agent.get_current_model()

        assert result == "openai/gpt-5"

    def test_get_current_model_fallback_to_first_provider(self, mock_agent_did, temp_db_path):
        """Test getting current model falls back to first provider."""
        mock_llm = MockLLMService()
        mock_llm.get_model_preference = lambda: {"provider": None, "model": None}
        mock_llm.providers = [{"name": "anthropic", "model": "claude-sonnet-4-5"}]

        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path,
            llm_service=mock_llm
        )

        result = agent.get_current_model()

        assert result == "anthropic/claude-sonnet-4-5"


# =============================================================================
# Tests for process_input()
# =============================================================================

class TestProcessInput:
    """Tests for process_input method."""

    @pytest.mark.asyncio
    async def test_process_input_basic(self, mock_agent_did, temp_db_path):
        """Test basic process_input with mocked response."""
        mock_llm = MockLLMService(responses=["Hello, how can I help?"])

        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            mock_storage_instance = MockAsyncStorage()
            MockStorage.return_value = mock_storage_instance

            agent = KestrelAgent(
                did=mock_agent_did,
                storage_path=temp_db_path,
                llm_service=mock_llm
            )

            # Mock all dependencies
            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent', return_value=MockWalletAgent(mock_agent_did, 100, temp_db_path)):
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem', return_value=MockMemorySystem(mock_storage_instance, mock_agent_did)):
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager', return_value=MockTaskManager()):
                                with patch('kestrel_sovereign.kestrel_agent.CommandHandler'):
                                    mock_context_builder = MagicMock()
                                    with patch('kestrel_sovereign.kestrel_agent.ContextBuilder', return_value=mock_context_builder):
                                        mock_context_manager = AsyncMock()
                                        mock_context_result = MagicMock()
                                        mock_context_result.system_prompt = "System prompt"
                                        mock_context_result.messages = []
                                        mock_context_result.warnings = []
                                        mock_context_result.budget_summary = {}
                                        mock_context_result.total_tokens = 100
                                        mock_context_result.episode_count = 0
                                        mock_context_result.memory_count = 0
                                        mock_context_result.rag_chunks = 0
                                        mock_context_manager.build_context.return_value = mock_context_result
                                        mock_context_manager.create_episode_if_needed = AsyncMock()

                                        with patch('kestrel_sovereign.kestrel_agent.ContextManager', return_value=mock_context_manager):
                                            with patch('kestrel_sovereign.kestrel_agent.BootstrapService') as MockBootstrap:
                                                mock_bootstrap = AsyncMock()
                                                mock_bootstrap.is_bootstrap_needed.return_value = False
                                                MockBootstrap.return_value = mock_bootstrap

                                                await agent.initialize()

                                                # Mock privacy agent
                                                agent.privacy_agent = AsyncMock()
                                                agent.privacy_agent.add_conversation = AsyncMock()
                                                agent.privacy_agent.get_conversation_history = AsyncMock(return_value=[])
                                                agent.privacy_agent.privacy_config = MagicMock()
                                                agent.privacy_agent.privacy_config.allows_cloud_llm.return_value = True

                                                # Mock observability store
                                                agent.observability_store = AsyncMock()
                                                agent.observability_store.log_metric = AsyncMock()
                                                agent.observability_store.log_tool_call = AsyncMock(return_value="event-123")
                                                agent.observability_store.log_tool_response = AsyncMock()
                                                agent.observability_store.log_error = AsyncMock()

                                                # Ensure prompt templates are set (they should be set in __init__ but verify)
                                                if not hasattr(agent, 'user_prompt_template'):
                                                    agent.user_prompt_template = agent._get_default_user_prompt()
                                                if not hasattr(agent, 'prompt_template'):
                                                    agent.prompt_template = agent._get_default_system_prompt()

                                                # Set _current_model_preference (used in check_solvency)
                                                if not hasattr(agent, '_current_model_preference'):
                                                    agent._current_model_preference = None

                                                response = await agent.process_input("Hello")

                                                assert "Hello, how can I help?" in response
                                                assert mock_llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_process_input_with_session_id(self, mock_agent_did, temp_db_path):
        """Test process_input with session_id parameter."""
        mock_llm = MockLLMService(responses=["Response for session"])

        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            mock_storage_instance = MockAsyncStorage()
            MockStorage.return_value = mock_storage_instance

            agent = KestrelAgent(
                did=mock_agent_did,
                storage_path=temp_db_path,
                llm_service=mock_llm
            )

            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent', return_value=MockWalletAgent(mock_agent_did, 100, temp_db_path)):
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem', return_value=MockMemorySystem(mock_storage_instance, mock_agent_did)):
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager', return_value=MockTaskManager()):
                                with patch('kestrel_sovereign.kestrel_agent.CommandHandler'):
                                    with patch('kestrel_sovereign.kestrel_agent.ContextBuilder'):
                                        mock_context_manager = AsyncMock()
                                        mock_context_result = MagicMock()
                                        mock_context_result.system_prompt = "System"
                                        mock_context_result.messages = []
                                        mock_context_result.warnings = []
                                        mock_context_result.budget_summary = {}
                                        mock_context_result.total_tokens = 50
                                        mock_context_result.episode_count = 0
                                        mock_context_result.memory_count = 0
                                        mock_context_result.rag_chunks = 0
                                        mock_context_manager.build_context.return_value = mock_context_result
                                        mock_context_manager.create_episode_if_needed = AsyncMock()

                                        with patch('kestrel_sovereign.kestrel_agent.ContextManager', return_value=mock_context_manager):
                                            with patch('kestrel_sovereign.kestrel_agent.BootstrapService') as MockBootstrap:
                                                mock_bootstrap = AsyncMock()
                                                mock_bootstrap.is_bootstrap_needed.return_value = False
                                                MockBootstrap.return_value = mock_bootstrap

                                                await agent.initialize()

                                                agent.privacy_agent = AsyncMock()
                                                agent.privacy_agent.add_conversation = AsyncMock()
                                                agent.privacy_agent.get_conversation_history = AsyncMock(return_value=[])
                                                agent.privacy_agent.privacy_config = MagicMock()
                                                agent.privacy_agent.privacy_config.allows_cloud_llm.return_value = True

                                                agent.observability_store = AsyncMock()
                                                agent.observability_store.log_metric = AsyncMock()
                                                agent.observability_store.log_tool_call = AsyncMock(return_value="event-123")
                                                agent.observability_store.log_tool_response = AsyncMock()

                                                # Ensure prompt templates are set
                                                if not hasattr(agent, 'user_prompt_template'):
                                                    agent.user_prompt_template = agent._get_default_user_prompt()
                                                if not hasattr(agent, 'prompt_template'):
                                                    agent.prompt_template = agent._get_default_system_prompt()
                                                if not hasattr(agent, '_current_model_preference'):
                                                    agent._current_model_preference = None

                                                response = await agent.process_input("Test", session_id="session-123")

                                                # Verify session_id was passed to privacy agent
                                                calls = agent.privacy_agent.get_conversation_history.call_args_list
                                                assert any('session_id' in str(call) for call in calls)


# =============================================================================
# Tests for Cancellation
# =============================================================================

class TestCancellation:
    """Tests for request cancellation."""

    def test_cancel_current_request_with_active_request(self, mock_agent_did, temp_db_path):
        """Test canceling an active request."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        # Simulate active request
        agent._current_request_id = "request-123"

        result = agent.cancel_current_request()

        assert result is True
        assert "request-123" in agent._cancelled_requests

    def test_cancel_current_request_with_no_active_request(self, mock_agent_did, temp_db_path):
        """Test canceling when no request is active."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        result = agent.cancel_current_request()

        assert result is False

    def test_is_request_cancelled(self, mock_agent_did, temp_db_path):
        """Test checking if request is cancelled."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        agent._current_request_id = "request-456"
        agent._cancelled_requests.add("request-456")

        assert agent.is_request_cancelled() is True
        assert agent.is_request_cancelled("request-456") is True
        assert agent.is_request_cancelled("request-789") is False


# =============================================================================
# Tests for Lifecycle Methods
# =============================================================================

class TestLifecycle:
    """Tests for lifecycle methods."""

    def test_close(self, mock_agent_did, temp_db_path):
        """Test synchronous close method."""
        mock_storage = MagicMock()
        mock_storage.close = MagicMock()

        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )
        agent.storage = mock_storage

        agent.close()

        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_agent_did, temp_db_path):
        """Test async shutdown method."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        # Mock all components that need cleanup
        mock_security = AsyncMock()
        mock_security.shutdown = AsyncMock()
        agent.features = {"SecurityFeature": mock_security}

        mock_mcp = AsyncMock()
        mock_mcp.shutdown = AsyncMock()
        agent.mcp_agent = mock_mcp

        mock_llm = AsyncMock()
        mock_llm.close = AsyncMock()
        agent.llm_service = mock_llm

        mock_task_manager = AsyncMock()
        mock_task_manager.close = AsyncMock()
        agent.task_manager = mock_task_manager

        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        await agent.shutdown()

        # Verify all cleanup methods were called
        mock_security.shutdown.assert_called_once()
        mock_mcp.shutdown.assert_called_once()
        mock_llm.close.assert_called_once()
        mock_task_manager.close.assert_called_once()
        mock_storage.close.assert_called_once()


# =============================================================================
# Tests for Event System
# =============================================================================

class TestEventSystem:
    """Tests for event listener system."""

    @pytest.mark.asyncio
    async def test_emit_event(self, mock_agent_did, temp_db_path):
        """Test emitting events to listeners."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        # Create mock listener
        listener_called = []

        async def mock_listener(event_type, data):
            listener_called.append((event_type, data))

        agent.add_event_listener(mock_listener)

        await agent.emit_event("test_event", {"key": "value"})

        assert len(listener_called) == 1
        assert listener_called[0][0] == "test_event"
        assert listener_called[0][1]["key"] == "value"

    def test_add_event_listener(self, mock_agent_did, temp_db_path):
        """Test adding event listeners."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        listener = MagicMock()
        agent.add_event_listener(listener)

        assert listener in agent._event_listeners

    def test_remove_event_listener(self, mock_agent_did, temp_db_path):
        """Test removing event listeners."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        listener = MagicMock()
        agent.add_event_listener(listener)
        agent.remove_event_listener(listener)

        assert listener not in agent._event_listeners


# =============================================================================
# Tests for Notifications
# =============================================================================

class TestNotifications:
    """Tests for pending task notifications."""

    def test_get_pending_notifications_empty(self, mock_agent_did, temp_db_path):
        """Test getting notifications when none are pending."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        notifications = agent.get_pending_notifications()

        assert notifications == []

    def test_get_pending_notifications_clears_queue(self, mock_agent_did, temp_db_path):
        """Test that getting notifications clears the queue."""
        agent = KestrelAgent(
            did=mock_agent_did,
            storage_path=temp_db_path
        )

        agent._pending_task_notifications.append("Task completed")
        agent._pending_task_notifications.append("Task failed")

        notifications = agent.get_pending_notifications()

        assert len(notifications) == 2
        assert "Task completed" in notifications
        assert "Task failed" in notifications

        # Queue should be empty now
        assert len(agent._pending_task_notifications) == 0
