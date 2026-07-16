"""
Unit tests for KestrelAgent core methods.

Tests actual behavior, error handling, and side effects.
NO mock-returns-mock tests - each test verifies real logic.
"""

import pytest
import asyncio
import contextlib
import os
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from kestrel_sovereign.kestrel_agent import KestrelAgent, _load_prompt_file
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.features.privacy.feature import PrivacyTransitionDecision


def _no_confirm_evaluate():
    # evaluate_transition mock for non-destructive transitions (never
    # PUBLIC→EPHEMERAL): the agent applies rather than staging pending.
    return MagicMock(side_effect=lambda m: PrivacyTransitionDecision(target=m, requires_confirmation=False))


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

    def test_init_defaults_bootstrap_service_to_none(self, tmp_path):
        """bootstrap_service defaults to None before initialize() runs (#1632).

        Regression: a COGNITION signal dispatch (e.g. talon.job_complete)
        reaches process_input's bootstrap check, which evaluates
        ``self.bootstrap_service``. Before this fix the attribute only existed
        after initialize(), so an early/partial dispatch raised
        ``AttributeError: 'KestrelAgent' object has no attribute
        'bootstrap_service'`` and the signal was never marked delivered.
        Accessing the attribute must yield None rather than raising.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        # Must not raise AttributeError; the bootstrap-needed guard in
        # process_input relies on this short-circuiting to False.
        assert agent.bootstrap_service is None
        assert bool(agent.bootstrap_service) is False

    def test_init_defaults_sync_service_enabled(self, tmp_path):
        """Sync service is enabled by default for production behavior."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=MagicMock(),
        )

        assert agent._sync_enabled is True

    def test_init_accepts_explicit_sync_disabled(self, tmp_path):
        """Callers can explicitly disable sync side effects."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=MagicMock(),
            sync_enabled=False,
        )

        assert agent._sync_enabled is False

    def test_init_reads_sync_disabled_env(self, tmp_path, monkeypatch):
        """Environment can disable sync without removing provider credentials."""
        monkeypatch.setenv("KESTREL_SYNC_ENABLED", "false")

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=MagicMock(),
        )

        assert agent._sync_enabled is False


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
        agent.privacy_agent.evaluate_transition = _no_confirm_evaluate()

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
        mock_privacy_agent.evaluate_transition = _no_confirm_evaluate()
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
        mock_privacy_agent.evaluate_transition = _no_confirm_evaluate()
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
        agent.features["ModelAgent"] = mock_model_agent

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
        """get_current_model() returns vendor/model when preference set."""
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "gpt-5"
        mock_llm.get_model_preference.return_value = {
            "vendor": "openai",
            "model": "gpt-5",
            "route": None,
        }
        mock_llm.providers = [
            {"name": "openai:api", "vendor": "openai", "route": "api", "model": "gpt-5"},
        ]

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=mock_llm
        )

        result = agent.get_current_model()

        assert result == "openai/gpt-5"

    def test_get_current_model_with_route_preference_set(self, tmp_path):
        """get_current_model() returns vendor:route/model when route mandated."""
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "claude-sonnet-4-6"
        mock_llm.get_model_preference.return_value = {
            "vendor": "anthropic",
            "model": "claude-sonnet-4-6",
            "route": "plan",
        }
        mock_llm.providers = [
            {"name": "anthropic:plan", "vendor": "anthropic", "route": "plan", "model": "claude-sonnet-4-6"},
        ]

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=mock_llm
        )

        result = agent.get_current_model()

        assert result == "anthropic:plan/claude-sonnet-4-6"

    def test_get_current_model_falls_back_to_first_provider(self, tmp_path):
        """get_current_model() falls back to first route when no preference."""
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "claude-sonnet-4-5"
        mock_llm.get_model_preference.return_value = {
            "vendor": None,
            "model": None,
            "route": None,
        }
        mock_llm.providers = [
            {"name": "anthropic:api", "vendor": "anthropic", "route": "api", "model": "claude-sonnet-4-5"},
        ]

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            llm_service=mock_llm
        )

        result = agent.get_current_model()

        assert result == "anthropic:api/claude-sonnet-4-5"

    def test_get_current_model_returns_auto_when_no_providers(self, tmp_path):
        """get_current_model() returns 'auto' when no providers available."""
        mock_llm = MagicMock()
        mock_llm.get_active_model_id.return_value = "auto"
        mock_llm.get_model_preference.return_value = {
            "vendor": None,
            "model": None,
            "route": None,
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

    @pytest.mark.asyncio
    async def test_emit_event_with_no_listeners_buffers_for_replay(self, tmp_path):
        """An event emitted while no listener is connected must be buffered
        and returned by get_pending_events — the host-startup reality the
        restart `completed` status straddles (#1551).
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        # No listeners connected — the boot reality.
        assert agent._event_listeners == []
        await agent.emit_event("restart_status", {"status": "completed"})

        drained = agent.get_pending_events()
        assert drained == [("restart_status", {"status": "completed"})]
        # Drain-once: a second drain is empty.
        assert agent.get_pending_events() == []

    @pytest.mark.asyncio
    async def test_emit_event_with_listener_does_not_buffer(self, tmp_path):
        """When a listener IS connected the event delivers live and is NOT
        also buffered (no double-delivery on a later reconnect).
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        calls = []

        async def listener(event_type, data):
            calls.append((event_type, data))

        agent.add_event_listener(listener)
        await agent.emit_event("restart_status", {"status": "pending"})

        assert calls == [("restart_status", {"status": "pending"})]
        assert agent.get_pending_events() == []

    @pytest.mark.asyncio
    async def test_pending_events_buffer_is_bounded(self, tmp_path):
        """A headless host that never opens an SSE stream must not grow the
        buffer without bound — oldest events drop past the cap.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        cap = agent._MAX_PENDING_EVENTS
        for i in range(cap + 25):
            await agent.emit_event("restart_status", {"n": i})

        drained = agent.get_pending_events()
        assert len(drained) == cap
        # Oldest dropped; the most recent event is retained.
        assert drained[-1] == ("restart_status", {"n": cap + 24})
        assert drained[0] == ("restart_status", {"n": 25})


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
    async def test_shutdown_closes_all_components(self, tmp_path):
        """shutdown() calls close/shutdown on all components."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        # Mock all components
        mock_security = AsyncMock()
        mock_security.shutdown = AsyncMock()

        mock_mcp = AsyncMock()
        mock_mcp.shutdown = AsyncMock()
        agent.features = {"SecurityFeature": mock_security, "MCPAgent": mock_mcp}

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
        agent.llm_service = None
        agent.task_manager = None
        agent.storage = None

        # Should not raise exception
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_background_task_removed_after_completion(self, tmp_path):
        """Agent-owned background tasks are removed when they finish."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        async def complete():
            return None

        task = agent._track_background_task(complete(), name="test_complete")

        await task
        await asyncio.sleep(0)

        assert agent._background_tasks == set()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_background_tasks_before_storage_close(self, tmp_path):
        """shutdown() cancels agent-owned background tasks before closing storage."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )
        started = asyncio.Event()

        async def never_finishes():
            started.set()
            await asyncio.Event().wait()

        task = agent._track_background_task(never_finishes(), name="test_pending")
        await started.wait()

        agent.features = {}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        await agent.shutdown()

        assert task.done()
        assert agent._background_tasks == set()
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_stops_memory_system_before_storage_close(self, tmp_path):
        """Memory-owned background work is stopped before storage closes."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )
        call_order = []

        async def shutdown_memory():
            call_order.append("memory")

        async def close_storage():
            call_order.append("storage")

        agent.features = {}
        agent.llm_service = None
        agent.task_manager = None
        agent.memory_system = MagicMock()
        agent.memory_system.shutdown = AsyncMock(side_effect=shutdown_memory)
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock(side_effect=close_storage)
        agent.storage = mock_storage

        await agent.shutdown()

        agent.memory_system.shutdown.assert_called_once()
        mock_storage.close.assert_called_once()
        assert call_order == ["memory", "storage"]

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_feature_owned_workers(self, tmp_path):
        """Whole-agent shutdown stops feature-owned background workers (#2409)."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        class _WorkerFeature:
            """Representative feature that owns a background worker task."""

            def __init__(self, name):
                self.name = name
                self.worker = None
                self._started = asyncio.Event()

            async def start(self):
                async def _run():
                    self._started.set()
                    await asyncio.Event().wait()

                self.worker = asyncio.create_task(_run())
                await self._started.wait()

            async def shutdown(self):
                if self.worker and not self.worker.done():
                    self.worker.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.worker

        health = _WorkerFeature("HealthFeature")
        delivery = _WorkerFeature("DeliveryFeature")
        scheduler = _WorkerFeature("SchedulerFeature")
        await health.start()
        await delivery.start()
        await scheduler.start()

        agent.features = {
            "HealthFeature": health,
            "DeliveryFeature": delivery,
            "SchedulerFeature": scheduler,
        }
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        await agent.shutdown()

        # No feature worker survives normal whole-agent shutdown.
        assert health.worker.done()
        assert delivery.worker.done()
        assert scheduler.worker.done()
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_stops_real_health_feature_background_task(self, tmp_path):
        """A real HealthFeature's owned background task does not survive (#2409).

        The fake-worker tests prove the sweep calls shutdown(); this proves
        the contract against a real in-tree feature that owns an
        ``asyncio.create_task`` background loop, so the actual
        ``HealthFeature.shutdown`` cancellation path is exercised end to end.
        """
        from kestrel_sovereign.features.health.feature import HealthFeature

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        # Drive the real feature's background loop without touching the DB:
        # a large interval means the loop parks in its first sleep and is
        # torn down purely by HealthFeature.shutdown()'s cancel path.
        health = HealthFeature(agent)
        health._interval_seconds = 3600
        health._running = False
        health._background_task = None
        health._start_background_loop()
        assert health._background_task is not None
        assert not health._background_task.done()

        agent.features = {"HealthFeature": health}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        await agent.shutdown()

        # The real feature's owned task is cancelled and cleared.
        assert health._background_task is None
        assert health._running is False
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_does_not_double_stop_security_feature(self, tmp_path):
        """SecurityFeature is shut down exactly once, not by the feature sweep (#2409)."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        mock_security = AsyncMock()
        mock_security.shutdown = AsyncMock()
        other = AsyncMock()
        other.shutdown = AsyncMock()

        agent.features = {"SecurityFeature": mock_security, "OtherFeature": other}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        await agent.shutdown()

        mock_security.shutdown.assert_called_once()
        other.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_feature_sweep_runs_before_storage_close(self, tmp_path):
        """Feature-owned workers stop before storage teardown, never racing it (#2409)."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )
        call_order = []

        feature = MagicMock()

        async def shutdown_feature():
            call_order.append("feature")

        feature.shutdown = AsyncMock(side_effect=shutdown_feature)

        async def close_storage():
            call_order.append("storage")

        agent.features = {"WorkerFeature": feature}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock(side_effect=close_storage)
        agent.storage = mock_storage

        await agent.shutdown()

        assert call_order == ["feature", "storage"]

    @pytest.mark.asyncio
    async def test_shutdown_twice_is_safe_with_stateful_feature(self, tmp_path):
        """A second whole-agent shutdown is safe against real feature state (#2409).

        A bare AsyncMock called twice proves nothing about idempotency — it
        cannot fail. This uses a stateful feature whose shutdown() operates
        on a real worker task: the first shutdown cancels+awaits it, and the
        second must observe the already-stopped state and return cleanly
        (no double-cancel error, no await on a consumed task).
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        class _StatefulFeature:
            """Owns a real worker; shutdown() must be safe to call twice."""

            def __init__(self):
                self.worker = None
                self.shutdown_calls = 0
                self._started = asyncio.Event()

            async def start(self):
                async def _run():
                    self._started.set()
                    await asyncio.Event().wait()

                self.worker = asyncio.create_task(_run())
                await self._started.wait()

            async def shutdown(self):
                self.shutdown_calls += 1
                # Guard on real state: only cancel/await a live worker.
                # A naive impl that unconditionally `await self.worker`
                # would raise on the second call (task already consumed).
                if self.worker is not None and not self.worker.done():
                    self.worker.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.worker
                self.worker = None

        feature = _StatefulFeature()
        await feature.start()

        agent.features = {"WorkerFeature": feature}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        await agent.shutdown()
        # A second whole-agent shutdown must not raise despite the worker
        # already being stopped, and must not double-cancel a consumed task.
        await agent.shutdown()

        assert feature.shutdown_calls == 2
        assert feature.worker is None

    @pytest.mark.asyncio
    async def test_shutdown_continues_when_one_feature_shutdown_fails(self, tmp_path):
        """A failing feature shutdown does not block the rest or storage close (#2409)."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        failing = AsyncMock()
        failing.shutdown = AsyncMock(side_effect=RuntimeError("worker boom"))
        healthy = AsyncMock()
        healthy.shutdown = AsyncMock()

        agent.features = {"FailingFeature": failing, "HealthyFeature": healthy}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        await agent.shutdown()

        failing.shutdown.assert_called_once()
        healthy.shutdown.assert_called_once()
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_runs_durable_tail_then_propagates_cancellation(
        self, tmp_path
    ):
        """Cancellation runs the durable tail, then propagates (#2409).

        agent.shutdown() is wrapped in asyncio.wait_for() by the CLI/server
        shutdown paths. If a feature shutdown is cancelled by the outer
        timeout, the sweep must NOT report a successful shutdown — but it
        must also NOT skip the safety-critical durable cleanup tail
        (background-task cleanup, memory shutdown, final sync snapshot,
        storage close). So storage MUST still close, and CancelledError
        MUST still propagate so wait_for surfaces the timeout.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        cancelled = AsyncMock()
        cancelled.shutdown = AsyncMock(side_effect=asyncio.CancelledError())

        agent.features = {"SlowFeature": cancelled}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        with pytest.raises(asyncio.CancelledError):
            await agent.shutdown()

        cancelled.shutdown.assert_called_once()
        # Durable cleanup MUST run despite cancellation — data safety wins.
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_prefix_cancellation_still_runs_durable_tail(
        self, tmp_path
    ):
        """Cancellation in the fallible PREFIX cannot bypass the durable tail (#2409).

        The heartbeat runner, resume monitor, salvage worker, and
        SecurityFeature run *before* the feature sweep. Before this fix they
        sat outside the try/finally, so a cancellation there (the outer
        wait_for timeout firing early) propagated straight out and skipped the
        durable cleanup tail entirely — leaking data/storage. They now live
        inside the protected region: cancellation anywhere in the prefix still
        runs the durable tail, then re-raises.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        # Cancellation strikes while stopping the heartbeat — the very first
        # prefix step, well before any feature sweep.
        heartbeat = MagicMock()
        heartbeat.stop = AsyncMock(side_effect=asyncio.CancelledError())
        agent.heartbeat_runner = heartbeat

        later_feature = AsyncMock()
        later_feature.shutdown = AsyncMock()
        agent.features = {"LaterFeature": later_feature}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        with pytest.raises(asyncio.CancelledError):
            await agent.shutdown()

        heartbeat.stop.assert_called_once()
        # The durable tail MUST still run even though cancellation hit the
        # prefix before the feature sweep.
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_bounds_a_hung_feature(self, tmp_path):
        """A feature that never returns from shutdown() is bounded (#2409).

        A single hung feature must not stall the sweep or starve the durable
        cleanup tail — it is abandoned after the per-feature timeout and the
        rest of shutdown (including storage close) still completes.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        hung_started = asyncio.Event()

        async def _never_returns():
            hung_started.set()
            await asyncio.Event().wait()

        hung = MagicMock()
        hung.shutdown = _never_returns
        healthy = AsyncMock()
        healthy.shutdown = AsyncMock()

        agent.features = {"HungFeature": hung, "HealthyFeature": healthy}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        with patch(
            "kestrel_sovereign.kestrel_agent.KESTREL_FEATURE_SHUTDOWN_TIMEOUT_S",
            0.05,
        ):
            await agent.shutdown()

        assert hung_started.is_set()
        # The hung feature was abandoned but the sweep and teardown finished.
        healthy.shutdown.assert_called_once()
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_hung_early_feature_does_not_starve_later_feature_or_storage(
        self, tmp_path
    ):
        """Coherent deadline composition under the production outer wait_for (#2409).

        The CLI/server/AgentManager paths all wrap agent.shutdown() in
        ``asyncio.wait_for(agent.shutdown(), timeout=SHUTDOWN_TIMEOUT)``. If a
        single early feature hangs forever, the per-feature bound must be
        composed against that outer deadline (not a larger fixed per-feature
        timeout) so the hung feature consumes only its fair slice and a LATER
        feature — plus the durable storage close — still run *inside* the
        outer deadline. This is the regression the review asked for: it drives
        shutdown through the real production-style outer wait_for rather than
        awaiting shutdown() directly.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        hung_started = asyncio.Event()
        late_started = asyncio.Event()

        async def _never_returns():
            hung_started.set()
            await asyncio.Event().wait()

        async def _late_shutdown():
            late_started.set()

        hung = MagicMock()
        hung.shutdown = _never_returns
        late = MagicMock()
        late.shutdown = _late_shutdown

        # Order matters: the hung feature is FIRST so a naive sweep would
        # never reach the late feature within the outer deadline.
        agent.features = {"HungFeature": hung, "LateFeature": late}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        # Shrink the internal budget so the test is fast, but keep the shape
        # identical to production: the internal budget is composed BELOW the
        # outer wait_for deadline, and each feature fair-divides it.
        outer_deadline = 2.0
        with patch(
            "kestrel_sovereign.kestrel_agent.KESTREL_AGENT_SHUTDOWN_TIMEOUT_S",
            0.6,
        ), patch(
            "kestrel_sovereign.kestrel_agent.KESTREL_SHUTDOWN_DURABLE_RESERVE_S",
            0.2,
        ):
            # Production-style outer wait_for. It must NOT be what saves us —
            # shutdown() must complete on its own well within the deadline.
            await asyncio.wait_for(agent.shutdown(), timeout=outer_deadline)

        assert hung_started.is_set()
        # The hung early feature did not starve the later feature...
        assert late_started.is_set()
        # ...nor the durable storage close.
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_hung_prefix_component_does_not_starve_later_feature(
        self, tmp_path
    ):
        """A hung PREFIX component (heartbeat) cannot starve a later feature (#2409).

        This reproduces the exact blocker the reviewer demonstrated on
        ``4bb9e425``: with a tiny internal budget/reserve, a
        cancellation-cooperative hung *heartbeat* (a prefix op, not a feature)
        followed by a real ``LateFeature`` and MCP/LLM/TaskManager sentinels,
        all wrapped in the production-style outer ``asyncio.wait_for``. The
        old fair-share denominator covered only the feature sweep, so the hung
        heartbeat consumed the whole prefix window and the later feature's
        body never ran (``late_shutdown_calls == 0``). The unified per-op
        budget must give the heartbeat only its fair slice so every later
        sentinel's coroutine BODY actually runs.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        heartbeat_started = asyncio.Event()
        late_started = asyncio.Event()
        mcp_started = asyncio.Event()
        llm_started = asyncio.Event()
        task_mgr_started = asyncio.Event()

        async def _hung_heartbeat():
            heartbeat_started.set()
            # Cancellation-cooperative: an infinite wait that yields to the
            # loop so wait_for's cancellation actually lands.
            await asyncio.Event().wait()

        heartbeat = MagicMock()
        heartbeat.stop = _hung_heartbeat
        agent.heartbeat_runner = heartbeat

        async def _late_shutdown():
            late_started.set()

        late = MagicMock()
        late.shutdown = _late_shutdown

        async def _mcp_shutdown():
            mcp_started.set()

        mcp = MagicMock()
        mcp.shutdown = _mcp_shutdown
        # ``mcp_agent`` is a read-only property backed by features["MCPAgent"];
        # the dedicated MCP block handles it and the sweep skips it.
        agent.features = {"LateFeature": late, "MCPAgent": mcp}

        async def _llm_close():
            llm_started.set()

        llm = MagicMock()
        llm.close = _llm_close
        agent.llm_service = llm

        async def _task_mgr_close():
            task_mgr_started.set()

        task_mgr = MagicMock()
        task_mgr.close = _task_mgr_close
        agent.task_manager = task_mgr

        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        with patch(
            "kestrel_sovereign.kestrel_agent.KESTREL_AGENT_SHUTDOWN_TIMEOUT_S",
            0.20,
        ), patch(
            "kestrel_sovereign.kestrel_agent.KESTREL_SHUTDOWN_DURABLE_RESERVE_S",
            0.05,
        ), patch(
            "kestrel_sovereign.kestrel_agent.KESTREL_SHUTDOWN_TAIL_MIN_STEP_S",
            0.01,
        ):
            await asyncio.wait_for(agent.shutdown(), timeout=5.0)

        assert heartbeat_started.is_set()
        # Every later op's coroutine BODY must have executed — proven by events
        # set from inside the bodies, not by AsyncMock call assertions (a
        # timeout-zero coroutine can be "called" without its body starting).
        assert late_started.is_set(), "later feature body was starved"
        assert mcp_started.is_set(), "MCP shutdown body was starved"
        assert llm_started.is_set(), "LLM close body was starved"
        assert task_mgr_started.is_set(), "TaskManager close body was starved"
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_durable_tail_bounds_cancellation_suppressing_step(
        self, tmp_path
    ):
        """A tail step that suppresses cancellation cannot hang the tail (#2409).

        ``asyncio.wait_for`` cancels once and then waits for completion. If the
        durable tail performed fresh unbounded awaits, a step that swallows
        ``CancelledError`` would make the outer timeout / CLI "forcing exit"
        branch unreachable forever. The tail must ABANDON such a step past its
        guard and continue to the data-critical storage close.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        storage_started = asyncio.Event()

        async def _suppresses_cancel():
            # Swallow cancellation and keep running forever.
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    continue

        async def _close_storage():
            storage_started.set()

        # Force the memory step to be the cancellation-suppressing hang.
        agent.memory_system = MagicMock()
        agent.memory_system.shutdown = _suppresses_cancel
        agent.features = {}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock(side_effect=_close_storage)
        agent.storage = mock_storage

        with patch(
            "kestrel_sovereign.kestrel_agent.KESTREL_SHUTDOWN_TAIL_MIN_STEP_S",
            0.05,
        ):
            # Must finish well within a generous outer bound despite the
            # suppressing step; the tail's own guard bounds it.
            await asyncio.wait_for(agent.shutdown(), timeout=3.0)

        # Storage close (data-critical) still ran after the hung step was
        # abandoned.
        assert storage_started.is_set()
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancellation_during_force_snapshot_completes_it_and_stops(
        self, tmp_path
    ):
        """Cancellation during force_snapshot: snapshot completes, stop() runs (#2409).

        The final snapshot is shielded so cancellation neither aborts it nor
        skips the following ``stop()`` (which releases the sync worker), while
        cancellation still propagates out of ``shutdown()``.
        """
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        snapshot_completed = asyncio.Event()
        stop_called = asyncio.Event()

        async def _slow_snapshot():
            await asyncio.sleep(0.1)
            snapshot_completed.set()

        async def _stop():
            stop_called.set()

        sync = MagicMock()
        sync.is_running = True
        sync.force_snapshot = _slow_snapshot
        sync.stop = _stop
        agent._sync_service = sync

        # A feature that cancels the whole shutdown, so cancellation is in
        # flight when the durable tail (and its force_snapshot) runs.
        cancelling = MagicMock()
        cancelling.shutdown = AsyncMock(side_effect=asyncio.CancelledError())
        agent.features = {"Canceller": cancelling}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        with pytest.raises(asyncio.CancelledError):
            await agent.shutdown()

        # Snapshot was NOT aborted by the in-flight cancellation...
        assert snapshot_completed.is_set()
        # ...and stop() still ran...
        assert stop_called.is_set()
        # ...and storage still closed...
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_abandoned_tail_step_is_not_reported_as_completed(
        self, tmp_path, caplog
    ):
        """No 'durable cleanup completed' when a tail step was abandoned (#2409)."""
        import logging as _logging

        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
        )

        async def _hangs():
            await asyncio.Event().wait()

        agent.memory_system = MagicMock()
        agent.memory_system.shutdown = _hangs
        agent.features = {}
        agent.llm_service = None
        agent.task_manager = None
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        agent.storage = mock_storage

        with patch(
            "kestrel_sovereign.kestrel_agent.KESTREL_SHUTDOWN_TAIL_MIN_STEP_S",
            0.05,
        ), caplog.at_level(_logging.INFO):
            await asyncio.wait_for(agent.shutdown(), timeout=3.0)

        text = caplog.text
        # The abandoned step must be surfaced as DEGRADED, never "completed".
        assert "DEGRADED" in text
        assert "async shutdown complete." not in text or "DEGRADED" in text
        # Storage still closed despite the degraded step.
        mock_storage.close.assert_called_once()

    def test_resolve_shutdown_budget_clamps_reserve_over_budget(self):
        """reserve >= budget is an explicit, safe clamp (#2409)."""
        from kestrel_sovereign import kestrel_agent as ka

        with patch.object(ka, "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", 4.0), patch.object(
            ka, "KESTREL_SHUTDOWN_DURABLE_RESERVE_S", 10.0
        ):
            prefix, reserve = ka._resolve_shutdown_budget()

        # Prefix keeps a nonzero majority; reserve clamped to at most half.
        assert prefix > 0
        assert reserve > 0
        assert reserve <= 4.0 / 2.0
        assert abs((prefix + reserve) - 4.0) < 1e-9

    def test_resolve_shutdown_budget_clamps_timeout_over_outer(self):
        """An internal timeout above the production outer deadline is clamped (#2409)."""
        from kestrel_sovereign import kestrel_agent as ka
        from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT

        with patch.object(
            ka, "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", float(SHUTDOWN_TIMEOUT) + 100.0
        ), patch.object(ka, "KESTREL_SHUTDOWN_DURABLE_RESERVE_S", 1.0):
            prefix, reserve = ka._resolve_shutdown_budget()

        # Total internal budget never exceeds the outer deadline.
        assert prefix + reserve <= float(SHUTDOWN_TIMEOUT) + 1e-9


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
                with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                    with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                        mock_storage_instance = AsyncMock()
                        mock_storage_instance.initialize = AsyncMock()
                        mock_storage_instance.get_node = AsyncMock(return_value=None)
                        mock_storage_instance.add_node = AsyncMock()
                        mock_storage_instance.db = MagicMock()
                        MockStorage.return_value = mock_storage_instance

                        mock_memory_system = AsyncMock()
                        mock_memory_system.initialize = AsyncMock()
                        mock_memory_system.retriever = MagicMock()
                        mock_memory_system.consolidator = MagicMock()
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
    async def test_initialize_wires_observability_store_into_llm_service(self, tmp_path):
        """initialize() attaches the observability store to the LLMService (#2236).

        Without this attach, LLMService._log_llm_call is a silent no-op and
        the LLM Calls panel only shows rows from features that log directly
        to the store (e.g. per-turn reflection) — never real chat calls."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                    with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                        mock_storage_instance = AsyncMock()
                        mock_storage_instance.initialize = AsyncMock()
                        mock_storage_instance.get_node = AsyncMock(return_value=None)
                        mock_storage_instance.add_node = AsyncMock()
                        mock_storage_instance.db = MagicMock()
                        MockStorage.return_value = mock_storage_instance

                        mock_memory_system = AsyncMock()
                        mock_memory_system.initialize = AsyncMock()
                        mock_memory_system.retriever = MagicMock()
                        mock_memory_system.consolidator = MagicMock()
                        MockMemorySystem.return_value = mock_memory_system

                        mock_task_manager = AsyncMock()
                        mock_task_manager.initialize = AsyncMock()
                        mock_task_manager.register_agent = MagicMock()
                        MockTaskManager.return_value = mock_task_manager

                        await agent.initialize()

                        assert agent.observability_store is not None
                        assert agent.llm_service._observability_store is agent.observability_store

    @pytest.mark.asyncio
    async def test_initialize_creates_memory_system(self, tmp_path):
        """initialize() creates memory_system."""
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db")
        )

        with patch('kestrel_sovereign.kestrel_agent.AsyncStorage') as MockStorage:
            with patch('kestrel_sovereign.kestrel_agent.discover_features', return_value=[]):
                with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                    with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                        mock_storage = AsyncMock()
                        mock_storage.initialize = AsyncMock()
                        mock_storage.get_node = AsyncMock(return_value=None)
                        mock_storage.add_node = AsyncMock()
                        mock_storage.db = MagicMock()
                        MockStorage.return_value = mock_storage

                        mock_memory_system = AsyncMock()
                        mock_memory_system.initialize = AsyncMock()
                        mock_memory_system.retriever = MagicMock()
                        mock_memory_system.consolidator = MagicMock()
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
                with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                    with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                        mock_storage = AsyncMock()
                        mock_storage.initialize = AsyncMock()
                        mock_storage.get_node = AsyncMock(return_value=None)  # No existing node
                        mock_storage.add_node = AsyncMock()
                        mock_storage.db = MagicMock()
                        MockStorage.return_value = mock_storage

                        mock_memory_system = AsyncMock()
                        mock_memory_system.initialize = AsyncMock()
                        mock_memory_system.retriever = MagicMock()
                        mock_memory_system.consolidator = MagicMock()
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
                with patch('kestrel_sovereign.kestrel_agent.MemorySystem') as MockMemorySystem:
                    with patch('kestrel_sovereign.kestrel_agent.TaskManager') as MockTaskManager:
                        mock_storage = AsyncMock()
                        mock_storage.initialize = AsyncMock()
                        mock_storage.get_node = AsyncMock(return_value=None)
                        mock_storage.add_node = AsyncMock()
                        mock_storage.db = MagicMock()
                        MockStorage.return_value = mock_storage

                        mock_memory_system = AsyncMock()
                        mock_memory_system.initialize = AsyncMock()
                        mock_memory_system.retriever = MagicMock()
                        mock_memory_system.consolidator = MagicMock()
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


# =============================================================================
# Tests for injectable PayerPolicy + host db (#1649, multi-tenant embedding)
# =============================================================================

class TestPayerPolicyInjection:
    """A host can inject PayerPolicy + host_db instead of reading them off disk."""

    def test_injected_policy_overrides_toml(self):
        """payer_policy injected → _resolve_payer_policy returns it, no toml."""
        sentinel = object()
        agent = KestrelAgent(did="did:test:inject-policy", payer_policy=sentinel)
        assert agent._resolve_payer_policy() is sentinel

    def test_no_policy_falls_back_to_toml_default(self):
        """No injection → loads the toml policy (host_env default sans kestrel.toml)."""
        from kestrel_sdk.payer_policy import PayerPolicy

        agent = KestrelAgent(did="did:test:fallback-policy")
        assert isinstance(agent._resolve_payer_policy(), PayerPolicy)

    @pytest.mark.asyncio
    async def test_injected_host_db_overrides_disk(self):
        """host_db injected → _resolve_host_db returns it, no on-disk lookup."""
        sentinel = object()
        agent = KestrelAgent(did="did:test:inject-hostdb", host_db=sentinel)
        assert await agent._resolve_host_db() is sentinel

    @pytest.mark.asyncio
    async def test_no_host_db_returns_none_without_host_file(self):
        """No injection + no on-disk host.db → None (resolver falls back to agent db)."""
        agent = KestrelAgent(did="did:test:no-hostdb", storage_path=None)
        assert await agent._resolve_host_db() is None
