"""
Unit tests for KestrelAgent core methods.

Tests actual behavior, error handling, and side effects.
NO mock-returns-mock tests - each test verifies real logic.
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from datetime import datetime, timezone
from decimal import Decimal

from kestrel_sovereign.kestrel_agent import KestrelAgent, _load_prompt_file
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


# =============================================================================
# Tests for _load_prompt_file() - pure file I/O logic
# =============================================================================

class TestLoadPromptFile:
    """Tests for _load_prompt_file function."""

    def test_load_existing_file_returns_content(self, tmp_path):
        """File exists → returns exact content."""
        prompt_file = tmp_path / "test_prompt.txt"
        expected_content = "Test prompt content\nwith multiple lines"
        prompt_file.write_text(expected_content, encoding="utf-8")

        result = _load_prompt_file(prompt_file, fallback="should not use this")

        assert result == expected_content

    def test_load_missing_file_returns_fallback(self, tmp_path):
        """File doesn't exist → returns fallback."""
        prompt_file = tmp_path / "nonexistent.txt"
        fallback = "fallback content"

        result = _load_prompt_file(prompt_file, fallback=fallback)

        assert result == fallback

    def test_load_file_with_unicode(self, tmp_path):
        """File with unicode → handles correctly."""
        prompt_file = tmp_path / "unicode.txt"
        unicode_content = "Hello 世界 🌍"
        prompt_file.write_text(unicode_content, encoding="utf-8")

        result = _load_prompt_file(prompt_file, fallback="fallback")

        assert result == unicode_content

    def test_load_unreadable_file_returns_fallback(self, tmp_path):
        """File exists but unreadable → returns fallback."""
        if os.name == 'nt':
            pytest.skip("Permission test not reliable on Windows")

        prompt_file = tmp_path / "unreadable.txt"
        prompt_file.write_text("content")
        os.chmod(prompt_file, 0o000)

        try:
            result = _load_prompt_file(prompt_file, fallback="fallback")
            assert result == "fallback"
        finally:
            os.chmod(prompt_file, 0o644)

    def test_load_empty_file_returns_empty_string(self, tmp_path):
        """Empty file → returns empty string (not fallback)."""
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("")

        result = _load_prompt_file(prompt_file, fallback="fallback")

        # Empty string stripped becomes empty, should use fallback? Let's verify actual behavior
        # Reading the code: filepath.read_text().strip() - empty file becomes ""
        assert result == ""


# =============================================================================
# Tests for KestrelAgent.__init__() - initialization state
# =============================================================================

class TestKestrelAgentInit:
    """Tests for KestrelAgent initialization (sync part)."""

    def test_init_sets_basic_attributes(self, tmp_path):
        """Init sets DID, storage_path, and privacy mode."""
        did = "did:pkh:eip155:1:0xTest123"
        db_path = str(tmp_path / "test.db")

        agent = KestrelAgent(
            did=did,
            storage_path=db_path,
            privacy_mode=PrivacyMode.ISOLATED
        )

        assert agent.did == did
        assert agent.storage_path == db_path
        assert agent._privacy_mode == PrivacyMode.ISOLATED
        assert agent.agent_id == did

    def test_init_defaults_to_normal_privacy_mode(self, tmp_path):
        """Default privacy mode is NORMAL."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        assert agent._privacy_mode == PrivacyMode.NORMAL
        assert agent.privacy_mode == PrivacyMode.NORMAL

    def test_init_creates_llm_service_if_not_provided(self, tmp_path):
        """LLMService created automatically if not provided."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        assert agent.llm_service is not None
        assert hasattr(agent.llm_service, 'generate_with_messages')

    def test_init_uses_provided_llm_service(self, tmp_path):
        """Uses provided LLM service instead of creating new one."""
        mock_llm = MagicMock()

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=mock_llm
        )

        assert agent.llm_service is mock_llm

    def test_init_with_postgres_backend(self):
        """PostgreSQL backend configuration stored correctly."""
        agent = KestrelAgent(
            did="did:test:123",
            database_url="postgresql://user:pass@localhost/kestrel",
            db_backend="postgres"
        )

        assert agent._db_backend == "postgres"
        assert agent._database_url == "postgresql://user:pass@localhost/kestrel"

    def test_init_defaults_to_sqlite_backend(self, tmp_path):
        """Defaults to SQLite backend."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        assert agent._db_backend == "sqlite"

    def test_init_initializes_empty_event_listeners(self, tmp_path):
        """Event listeners list starts empty."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        assert agent._event_listeners == []

    def test_init_initializes_empty_cancelled_requests(self, tmp_path):
        """Cancelled requests set starts empty."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        assert agent._cancelled_requests == set()
        assert agent._current_request_id is None


# =============================================================================
# Tests for Privacy Mode - getter/setter behavior
# =============================================================================

class TestPrivacyMode:
    """Tests for privacy mode getter/setter."""

    def test_privacy_mode_getter_returns_current_mode(self, tmp_path):
        """privacy_mode property returns current mode."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            privacy_mode=PrivacyMode.ANONYMOUS
        )

        assert agent.privacy_mode == PrivacyMode.ANONYMOUS

    @pytest.mark.asyncio
    async def test_set_privacy_mode_changes_internal_state(self, tmp_path):
        """set_privacy_mode() updates internal _privacy_mode."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            privacy_mode=PrivacyMode.NORMAL
        )

        # Mock storage and privacy agent (since not initialized)
        agent.storage = MagicMock()
        agent.storage.set_privacy_mode = MagicMock()
        agent.privacy_agent = MagicMock()
        agent.privacy_agent.set_mode = MagicMock()

        await agent.set_privacy_mode(PrivacyMode.EPHEMERAL)

        assert agent._privacy_mode == PrivacyMode.EPHEMERAL
        assert agent.privacy_mode == PrivacyMode.EPHEMERAL

    @pytest.mark.asyncio
    async def test_set_privacy_mode_updates_storage_wrapper(self, tmp_path):
        """set_privacy_mode() updates storage wrapper if initialized."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        # Mock storage to verify call
        mock_storage = MagicMock()
        mock_storage.set_privacy_mode = MagicMock()
        agent.storage = mock_storage

        # Mock privacy agent
        mock_privacy_agent = MagicMock()
        mock_privacy_agent.set_mode = MagicMock()
        agent.privacy_agent = mock_privacy_agent

        await agent.set_privacy_mode(PrivacyMode.ISOLATED)

        mock_storage.set_privacy_mode.assert_called_once_with(PrivacyMode.ISOLATED)
        mock_privacy_agent.set_mode.assert_called_once_with(PrivacyMode.ISOLATED)

    @pytest.mark.asyncio
    async def test_set_privacy_mode_returns_privacy_agent_message(self, tmp_path):
        """set_privacy_mode() returns the canonical privacy-agent status message."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            privacy_mode=PrivacyMode.NORMAL,
        )

        mock_storage = MagicMock()
        mock_storage.set_privacy_mode = MagicMock()
        agent.storage = mock_storage

        mock_privacy_agent = MagicMock()
        mock_privacy_agent.set_mode = MagicMock(return_value="Privacy mode changed from normal to isolated.")
        agent.privacy_agent = mock_privacy_agent

        result = await agent.set_privacy_mode(PrivacyMode.ISOLATED)

        assert result == "Privacy mode changed from normal to isolated."
        mock_storage.set_privacy_mode.assert_called_once_with(PrivacyMode.ISOLATED)
        mock_privacy_agent.set_mode.assert_called_once_with(PrivacyMode.ISOLATED)


# =============================================================================
# Tests for Model Selection
# =============================================================================

class TestModelSelection:
    """Tests for model selection methods."""

    def test_set_model_delegates_to_model_agent(self, tmp_path):
        """set_model() delegates to model_agent.set_model_preference()."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        mock_model_agent = MagicMock()
        mock_model_agent.set_model_preference.return_value = "Model set to gpt-5"
        agent.model_agent = mock_model_agent

        result = agent.set_model("gpt-5")

        # Verify delegation happened with correct argument
        mock_model_agent.set_model_preference.assert_called_once_with("gpt-5")
        # Verify result contains success message
        assert "Model set" in result


class TestTrustedAgentCreation:
    """Tests for trusted-agent creation boundaries."""

    @pytest.mark.asyncio
    async def test_create_trusted_agent_awaits_graph_store_write(self, tmp_path):
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        graph_store = MagicMock()
        graph_store.add_node = AsyncMock()
        agent.storage = MagicMock(graph_store=graph_store)

        with patch(
            "kestrel_sovereign.inception_service.generate_kestrel_identity",
            return_value=({"id": "did:test:new"}, {"private_key": "secret"}),
        ), patch(
            "kestrel_sovereign.inception_service.save_kestrel_identity",
        ):
            result = await agent.create_trusted_agent("new-friend")

        assert "Created trusted agent 'new-friend'" in result
        graph_store.add_node.assert_awaited_once()
        created_node = graph_store.add_node.await_args.args[0]
        assert created_node.node_id == "did:test:new"
        assert created_node.label == "new-friend"


class TestMemoryAnchoring:
    """Tests for async anchoring boundaries."""

    @pytest.mark.asyncio
    async def test_anchor_memory_state_awaits_wallet_transfer(self, tmp_path):
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        conversation = MagicMock()
        conversation.get_conversation_history_hash.return_value = "hash-123"
        conversation.add_log_anchor = MagicMock()
        agent.storage = MagicMock(conversation=conversation)

        wallet = MagicMock()
        wallet.can_afford.return_value = True
        wallet.transfer = AsyncMock()
        agent.wallet = wallet

        notary = MagicMock()
        notary.estimate_cost.return_value = Decimal("0.5")
        notary.publish_anchor.return_value = "tx-abc"
        agent.notary_service = notary

        privacy_config = MagicMock()
        privacy_config.is_ephemeral.return_value = False
        agent.privacy_agent = MagicMock(privacy_config=privacy_config)

        result = await agent.anchor_memory_state()

        assert "Successfully anchored memory state." in result
        wallet.transfer.assert_awaited_once_with(Decimal("0.5"), "Notary Service for memory anchor")
        conversation.add_log_anchor.assert_called_once_with("hash-123", "tx-abc")

    def test_get_current_model_with_preference_set(self, tmp_path):
        """get_current_model() returns provider/model when preference set."""
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "gpt-5"
        mock_llm.get_model_preference.return_value = {
            "provider": "openai",
            "model": "gpt-5"
        }

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=mock_llm
        )

        result = agent.get_current_model()

        assert result == "openai/gpt-5"

    def test_get_current_model_falls_back_to_first_provider(self, tmp_path):
        """get_current_model() falls back to first provider when no preference."""
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "claude-sonnet-4-5"
        mock_llm.get_model_preference.return_value = {
            "provider": None,
            "model": None
        }
        mock_llm.providers = [
            {"name": "anthropic", "model": "claude-sonnet-4-5"}
        ]

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=mock_llm
        )

        result = agent.get_current_model()

        assert result == "anthropic/claude-sonnet-4-5"

    def test_get_current_model_returns_auto_when_no_providers(self, tmp_path):
        """get_current_model() returns 'auto' when no providers available."""
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "auto"
        mock_llm.get_model_preference.return_value = {
            "provider": None,
            "model": None
        }
        mock_llm.providers = []

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=mock_llm
        )

        result = agent.get_current_model()

        assert result == "auto"


# =============================================================================
# Tests for Event System
# =============================================================================

class TestEventSystem:
    """Tests for event listener system."""

    def test_add_event_listener_adds_to_list(self, tmp_path):
        """add_event_listener() adds listener to internal list."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        listener1 = MagicMock()
        listener2 = MagicMock()

        agent.add_event_listener(listener1)
        agent.add_event_listener(listener2)

        assert listener1 in agent._event_listeners
        assert listener2 in agent._event_listeners
        assert len(agent._event_listeners) == 2

    def test_remove_event_listener_removes_from_list(self, tmp_path):
        """remove_event_listener() removes listener from list."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        listener = MagicMock()
        agent.add_event_listener(listener)

        agent.remove_event_listener(listener)

        assert listener not in agent._event_listeners

    def test_remove_nonexistent_listener_does_not_crash(self, tmp_path):
        """Removing non-existent listener doesn't crash."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        listener = MagicMock()

        # Should not raise exception
        agent.remove_event_listener(listener)

    @pytest.mark.asyncio
    async def test_emit_event_calls_all_listeners(self, tmp_path):
        """emit_event() calls all registered listeners with correct args."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        # Track calls to listeners
        listener1_calls = []
        listener2_calls = []

        async def listener1(event_type, data):
            listener1_calls.append((event_type, data))

        async def listener2(event_type, data):
            listener2_calls.append((event_type, data))

        agent.add_event_listener(listener1)
        agent.add_event_listener(listener2)

        event_data = {"key": "value", "count": 42}
        await agent.emit_event("test_event", event_data)

        # Verify both listeners received the event
        assert len(listener1_calls) == 1
        assert listener1_calls[0] == ("test_event", event_data)
        assert len(listener2_calls) == 1
        assert listener2_calls[0] == ("test_event", event_data)

    @pytest.mark.asyncio
    async def test_emit_event_handles_listener_exceptions(self, tmp_path):
        """emit_event() continues calling listeners even if one raises exception."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        successful_calls = []

        async def failing_listener(event_type, data):
            raise ValueError("Listener error")

        async def successful_listener(event_type, data):
            successful_calls.append((event_type, data))

        agent.add_event_listener(failing_listener)
        agent.add_event_listener(successful_listener)

        # Should not crash despite failing listener
        await agent.emit_event("test_event", {"test": "data"})

        # Successful listener should still have been called
        assert len(successful_calls) == 1
        assert successful_calls[0] == ("test_event", {"test": "data"})


# =============================================================================
# Tests for Cancellation
# =============================================================================

class TestCancellation:
    """Tests for request cancellation."""

    def test_cancel_current_request_with_active_request_returns_true(self, tmp_path):
        """cancel_current_request() returns True when request is active."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent.register_active_request("request-123")
        agent._current_request_id = "request-123"

        result = agent.cancel_current_request()

        assert result is True
        assert "request-123" in agent._cancelled_requests

    def test_cancel_current_request_with_no_active_request_returns_false(self, tmp_path):
        """cancel_current_request() returns False when no active request."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent._current_request_id = None

        result = agent.cancel_current_request()

        assert result is False
        assert len(agent._cancelled_requests) == 0

    def test_is_request_cancelled_for_current_request(self, tmp_path):
        """is_request_cancelled() returns True for cancelled current request."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent.register_active_request("request-456")
        agent._current_request_id = "request-456"
        agent._cancelled_requests.add("request-456")

        assert agent.is_request_cancelled() is True

    def test_is_request_cancelled_for_specific_request_id(self, tmp_path):
        """is_request_cancelled(request_id) checks specific request."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent.register_active_request("request-456")
        agent._cancelled_requests.add("request-456")

        assert agent.is_request_cancelled("request-456") is True
        assert agent.is_request_cancelled("request-789") is False

    def test_is_request_cancelled_returns_false_when_not_cancelled(self, tmp_path):
        """is_request_cancelled() returns False when request not cancelled."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent.register_active_request("request-999")
        agent._current_request_id = "request-999"

        assert agent.is_request_cancelled() is False

    def test_cancel_current_request_specific_id_returns_true(self, tmp_path):
        """cancel_current_request(request_id) cancels the targeted active request."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent.register_active_request("request-1")
        agent.register_active_request("request-2")

        result = agent.cancel_current_request("request-1")

        assert result is True
        assert "request-1" in agent._cancelled_requests


# =============================================================================
# Tests for Notifications
# =============================================================================

class TestNotifications:
    """Tests for pending task notifications."""

    def test_get_pending_notifications_returns_empty_list_initially(self, tmp_path):
        """get_pending_notifications() returns [] when no notifications."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        notifications = agent.get_pending_notifications()

        assert notifications == []

    def test_get_pending_notifications_returns_all_notifications(self, tmp_path):
        """get_pending_notifications() returns all accumulated notifications."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent._pending_task_notifications.append("Task 1 completed")
        agent._pending_task_notifications.append("Task 2 failed")
        agent._pending_task_notifications.append("Task 3 started")

        notifications = agent.get_pending_notifications()

        assert len(notifications) == 3
        assert "Task 1 completed" in notifications
        assert "Task 2 failed" in notifications
        assert "Task 3 started" in notifications

    def test_get_pending_notifications_clears_queue(self, tmp_path):
        """get_pending_notifications() clears queue after returning."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent._pending_task_notifications.append("Notification 1")
        agent._pending_task_notifications.append("Notification 2")

        # First call returns notifications
        first_call = agent.get_pending_notifications()
        assert len(first_call) == 2

        # Second call returns empty (queue cleared)
        second_call = agent.get_pending_notifications()
        assert second_call == []
        assert len(agent._pending_task_notifications) == 0


# =============================================================================
# Tests for Lifecycle Methods
# =============================================================================

class TestLifecycle:
    """Tests for lifecycle methods."""

    @pytest.mark.asyncio
    async def test_shutdown_does_not_crash_when_storage_none(self, tmp_path):
        """shutdown() doesn't crash when storage is None."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        agent.features = {}
        agent.mcp_agent = None
        agent.llm_service = None
        agent.task_manager = None
        agent.storage = None

        # Should not raise exception
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_closes_all_components(self, tmp_path):
        """shutdown() calls close/shutdown on all components."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        # Mock all components
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

    @pytest.mark.asyncio
    async def test_shutdown_handles_missing_components_gracefully(self, tmp_path):
        """shutdown() doesn't crash when components not initialized."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        # Don't initialize any components
        agent.features = {}
        agent.mcp_agent = None
        agent.llm_service = None
        agent.task_manager = None
        agent.storage = None

        # Should not raise exception
        await agent.shutdown()


# =============================================================================
# Tests for Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_shutdown_handles_component_shutdown_exceptions(self, tmp_path):
        """shutdown() continues closing other components if one fails."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        # Mock components where one fails
        mock_llm = AsyncMock()
        mock_llm.close.side_effect = RuntimeError("LLM close failed")
        agent.llm_service = mock_llm

        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()  # This should still be called
        agent.storage = mock_storage

        agent.features = {}
        agent.mcp_agent = None
        agent.task_manager = None

        # Should not crash
        await agent.shutdown()

        # Storage close should still have been attempted
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_event_logs_listener_exception(self, tmp_path):
        """emit_event() logs exception from failing listener."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        async def failing_listener(event_type, data):
            raise ValueError("Listener failed")

        agent.add_event_listener(failing_listener)

        # Should not crash - exception is caught and logged at warning level
        with patch('logging.warning') as mock_log:
            await agent.emit_event("test_event", {"data": "value"})

            # Verify warning was logged
            assert mock_log.called
            # Verify the exception message was logged
            call_args = str(mock_log.call_args)
            assert "Listener failed" in call_args or "Failed to emit event" in call_args


# =============================================================================
# Tests for initialize() - Side Effects
# =============================================================================

class TestInitialize:
    """Tests for async initialize() method."""

    @pytest.mark.asyncio
    async def test_initialize_creates_storage(self, tmp_path):
        """initialize() creates and initializes storage."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent') as MockWallet:
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                                mock_storage_instance = AsyncMock()
                                mock_storage_instance.initialize = AsyncMock()
                                mock_storage_instance.get_node = AsyncMock(return_value=None)
                                mock_storage_instance.add_node = AsyncMock()
                                mock_storage_instance.db = MagicMock()
                                MockStorage.return_value = mock_storage_instance

                                mock_wallet = AsyncMock()
                                mock_wallet.initialize = AsyncMock()
                                MockWallet.return_value = mock_wallet

                                mock_memory_system = AsyncMock()
                                mock_memory_system.initialize = AsyncMock()
                                mock_memory_system.retriever = MagicMock()
                                MockMemorySystem.return_value = mock_memory_system

                                mock_task_manager = AsyncMock()
                                mock_task_manager.initialize = AsyncMock()
                                mock_task_manager.register_agent = MagicMock()
                                MockTaskManager.return_value = mock_task_manager

                                await agent.initialize()

                                # Verify storage was created and initialized
                                assert agent._raw_storage is not None
                                assert agent.storage is not None
                                mock_storage_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_creates_memory_system(self, tmp_path):
        """initialize() creates memory_system."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent') as MockWallet:
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                                mock_storage = AsyncMock()
                                mock_storage.initialize = AsyncMock()
                                mock_storage.get_node = AsyncMock(return_value=None)
                                mock_storage.add_node = AsyncMock()
                                mock_storage.db = MagicMock()
                                MockStorage.return_value = mock_storage

                                mock_wallet = AsyncMock()
                                mock_wallet.initialize = AsyncMock()
                                MockWallet.return_value = mock_wallet

                                mock_memory_system = AsyncMock()
                                mock_memory_system.initialize = AsyncMock()
                                mock_memory_system.retriever = MagicMock()
                                MockMemorySystem.return_value = mock_memory_system

                                mock_task_manager = AsyncMock()
                                mock_task_manager.initialize = AsyncMock()
                                mock_task_manager.register_agent = MagicMock()
                                MockTaskManager.return_value = mock_task_manager

                                await agent.initialize()

                                # Verify memory system created and initialized
                                assert agent.memory_system is not None
                                mock_memory_system.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_creates_agent_node_if_missing(self, tmp_path):
        """initialize() creates agent node in storage if it doesn't exist."""
        agent = KestrelAgent(
            did="did:test:agent123",
            storage_path=str(tmp_path / "test.db")
        )

        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent') as MockWallet:
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                                mock_storage = AsyncMock()
                                mock_storage.initialize = AsyncMock()
                                mock_storage.get_node = AsyncMock(return_value=None)  # No existing node
                                mock_storage.add_node = AsyncMock()
                                mock_storage.db = MagicMock()
                                MockStorage.return_value = mock_storage

                                mock_wallet = AsyncMock()
                                mock_wallet.initialize = AsyncMock()
                                MockWallet.return_value = mock_wallet

                                mock_memory_system = AsyncMock()
                                mock_memory_system.initialize = AsyncMock()
                                mock_memory_system.retriever = MagicMock()
                                MockMemorySystem.return_value = mock_memory_system

                                mock_task_manager = AsyncMock()
                                mock_task_manager.initialize = AsyncMock()
                                mock_task_manager.register_agent = MagicMock()
                                MockTaskManager.return_value = mock_task_manager

                                await agent.initialize()

                                # Verify add_node was called to create the agent node
                                mock_storage.add_node.assert_called_once()
                                call_args = mock_storage.add_node.call_args[0][0]
                                assert call_args.node_id == "did:test:agent123"
                                assert call_args.node_type == "agent"

    @pytest.mark.asyncio
    async def test_initialize_registers_features(self, tmp_path):
        """initialize() registers discovered features."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        mock_feature = MagicMock()
        mock_feature.name = "TestFeature"
        mock_feature.initialize = AsyncMock()
        mock_feature.on_enable = AsyncMock()
        mock_feature.post_all_features_loaded = AsyncMock()
        mock_feature.get_hooks.return_value = []
        mock_feature.get_tools.return_value = []
        mock_feature.get_agent_card.return_value = MagicMock(name="TestFeature", skills=[])

        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[mock_feature]):
                with patch('kestrel_sovereign.kestrel_agent.WalletAgent') as MockWallet:
                    with patch('kestrel_sovereign.kestrel_agent.MemoryConsolidator'):
                        with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                            with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                                mock_storage = AsyncMock()
                                mock_storage.initialize = AsyncMock()
                                mock_storage.get_node = AsyncMock(return_value=None)
                                mock_storage.add_node = AsyncMock()
                                mock_storage.db = MagicMock()
                                MockStorage.return_value = mock_storage

                                mock_wallet = AsyncMock()
                                mock_wallet.initialize = AsyncMock()
                                MockWallet.return_value = mock_wallet

                                mock_memory_system = AsyncMock()
                                mock_memory_system.initialize = AsyncMock()
                                mock_memory_system.retriever = MagicMock()
                                MockMemorySystem.return_value = mock_memory_system

                                mock_task_manager = AsyncMock()
                                mock_task_manager.initialize = AsyncMock()
                                mock_task_manager.register_agent = MagicMock()
                                MockTaskManager.return_value = mock_task_manager

                                await agent.initialize()

                                # Verify feature was registered
                                assert "TestFeature" in agent.features
                                assert agent.features["TestFeature"] is mock_feature
                                mock_feature.initialize.assert_called_once()
