"""
Tests for the Privacy-Enforcing Storage Wrapper.
"""

import warnings
import pytest
from unittest.mock import Mock, AsyncMock, PropertyMock
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import (
    PrivacyEnforcingStorage,
    PrivacyViolationError,
    PrivacyPolicy,
    wrap_storage_with_privacy,
)


class TestPrivacyPolicy:
    """Tests for PrivacyPolicy configuration."""

    def test_ephemeral_policy(self):
        """EPHEMERAL mode should block persistent writes."""
        policy = PrivacyPolicy.for_mode(PrivacyMode.EPHEMERAL)
        assert policy.allow_persistent_write is False
        assert policy.allow_persistent_read is True
        assert policy.use_session_storage is False
        assert policy.allow_cloud_backup is False

    def test_isolated_policy(self):
        """ISOLATED mode should use session storage."""
        policy = PrivacyPolicy.for_mode(PrivacyMode.ISOLATED)
        assert policy.allow_persistent_write is False
        assert policy.use_session_storage is True
        assert policy.allow_cloud_backup is False

    def test_anonymous_policy(self):
        """ANONYMOUS mode should require anonymization."""
        policy = PrivacyPolicy.for_mode(PrivacyMode.ANONYMOUS)
        assert policy.allow_persistent_write is True
        assert policy.require_anonymization is True
        assert policy.allow_cloud_backup is True

    def test_normal_policy(self):
        """NORMAL mode should allow everything without anonymization."""
        policy = PrivacyPolicy.for_mode(PrivacyMode.NORMAL)
        assert policy.allow_persistent_write is True
        assert policy.require_anonymization is False
        assert policy.allow_cloud_backup is True


class TestEphemeralMode:
    """Tests for EPHEMERAL privacy mode enforcement."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = Mock()
        storage.add_conversation = AsyncMock()
        storage.get_conversation_history = AsyncMock(return_value=[])
        storage.store_file = AsyncMock(return_value="hash123")
        storage.create_backup_blob = AsyncMock(return_value=b"backup")
        return storage

    @pytest.fixture
    def ephemeral_storage(self, mock_storage):
        """Create privacy wrapper in EPHEMERAL mode."""
        return PrivacyEnforcingStorage(mock_storage, PrivacyMode.EPHEMERAL)

    @pytest.mark.asyncio
    async def test_blocks_conversation_storage(self, ephemeral_storage):
        """EPHEMERAL mode should block conversation storage."""
        with pytest.raises(PrivacyViolationError) as exc_info:
            await ephemeral_storage.add_conversation("user", "Hello")

        assert "ephemeral" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_blocks_file_storage(self, ephemeral_storage):
        """EPHEMERAL mode should block file storage."""
        with pytest.raises(PrivacyViolationError):
            await ephemeral_storage.store_file(b"content", "test.txt")

    @pytest.mark.asyncio
    async def test_blocks_backup(self, ephemeral_storage):
        """EPHEMERAL mode should block backups."""
        with pytest.raises(PrivacyViolationError):
            await ephemeral_storage.create_backup_blob()

    @pytest.mark.asyncio
    async def test_allows_reads(self, ephemeral_storage, mock_storage):
        """EPHEMERAL mode should allow reading existing data."""
        mock_storage.get_conversation_history.return_value = [{"role": "user", "content": "old"}]

        history = await ephemeral_storage.get_conversation_history()
        assert len(history) == 1


class TestIsolatedMode:
    """Tests for ISOLATED privacy mode enforcement."""

    @pytest.fixture
    def mock_storage(self):
        storage = Mock()
        storage.add_conversation = AsyncMock()
        storage.get_conversation_history = AsyncMock(return_value=[])
        storage.store_file = AsyncMock(return_value="hash123")
        return storage

    @pytest.fixture
    def isolated_storage(self, mock_storage):
        return PrivacyEnforcingStorage(mock_storage, PrivacyMode.ISOLATED)

    @pytest.mark.asyncio
    async def test_stores_in_session(self, isolated_storage, mock_storage):
        """ISOLATED mode should store in session, not persistent storage."""
        await isolated_storage.add_conversation("user", "Hello")
        await isolated_storage.add_conversation("assistant", "Hi there")

        # Should NOT call underlying storage
        mock_storage.add_conversation.assert_not_called()

        # Should have in session
        history = await isolated_storage.get_conversation_history()
        assert len(history) == 2
        assert history[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_session_clear(self, isolated_storage):
        """Should be able to clear session storage."""
        await isolated_storage.add_conversation("user", "Hello")
        await isolated_storage.add_conversation("assistant", "Hi")

        count = isolated_storage.clear_session()
        assert count == 2
        history = await isolated_storage.get_conversation_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_session_save_to_persistent(self, isolated_storage, mock_storage):
        """Should be able to promote session to persistent storage."""
        await isolated_storage.add_conversation("user", "Hello")
        await isolated_storage.add_conversation("assistant", "Hi")

        count = await isolated_storage.save_session_to_persistent()
        assert count == 2
        assert mock_storage.add_conversation.call_count == 2
        history = await isolated_storage.get_conversation_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_file_stored_in_session(self, isolated_storage, mock_storage):
        """Files should be stored in session in ISOLATED mode."""
        content = b"test file content"
        hash_result = await isolated_storage.store_file(content, "test.txt")

        # Should NOT call underlying storage
        mock_storage.store_file.assert_not_called()

        # Should be retrievable from session
        retrieved = await isolated_storage.retrieve_file(hash_result)
        assert retrieved == content


class TestAnonymousMode:
    """Tests for ANONYMOUS privacy mode enforcement."""

    @pytest.fixture
    def mock_storage(self):
        storage = Mock()
        storage.add_conversation = AsyncMock()
        return storage

    @pytest.fixture
    def anonymous_storage(self, mock_storage):
        return PrivacyEnforcingStorage(mock_storage, PrivacyMode.ANONYMOUS)

    @pytest.mark.asyncio
    async def test_anonymizes_pii(self, anonymous_storage, mock_storage):
        """ANONYMOUS mode should scrub PII before storage."""
        await anonymous_storage.add_conversation(
            "user",
            "My email is john@example.com and phone is 555-123-4567"
        )

        # Check what was actually stored
        mock_storage.add_conversation.assert_called_once()
        args = mock_storage.add_conversation.call_args
        stored_content = args[0][1]  # Second positional arg

        assert "john@example.com" not in stored_content
        assert "555-123-4567" not in stored_content
        assert "[EMAIL_REDACTED]" in stored_content
        assert "[PHONE_REDACTED]" in stored_content

    @pytest.mark.asyncio
    async def test_adds_privacy_mode_metadata(self, anonymous_storage, mock_storage):
        """Should add privacy_mode to metadata."""
        await anonymous_storage.add_conversation("user", "Hello")

        args = mock_storage.add_conversation.call_args
        metadata = args[0][2]  # Third positional arg

        assert metadata["privacy_mode"] == "anonymous"


class TestNormalMode:
    """Tests for NORMAL privacy mode (pass-through)."""

    @pytest.fixture
    def mock_storage(self):
        storage = Mock()
        storage.add_conversation = AsyncMock()
        storage.store_file = AsyncMock(return_value="hash123")
        storage.create_backup_blob = AsyncMock(return_value=b"backup")
        return storage

    @pytest.fixture
    def normal_storage(self, mock_storage):
        return PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

    @pytest.mark.asyncio
    async def test_passes_through_conversation(self, normal_storage, mock_storage):
        """NORMAL mode should pass through without modification."""
        await normal_storage.add_conversation("user", "My email is john@example.com")

        args = mock_storage.add_conversation.call_args
        stored_content = args[0][1]

        # Should NOT anonymize in NORMAL mode
        assert "john@example.com" in stored_content

    @pytest.mark.asyncio
    async def test_allows_file_storage(self, normal_storage, mock_storage):
        """NORMAL mode should allow file storage."""
        result = await normal_storage.store_file(b"content", "test.txt")
        assert result == "hash123"
        mock_storage.store_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_backup(self, normal_storage, mock_storage):
        """NORMAL mode should allow backups."""
        result = await normal_storage.create_backup_blob()
        assert result == b"backup"


class TestModeTransitions:
    """Tests for changing privacy modes."""

    @pytest.fixture
    def mock_storage(self):
        storage = Mock()
        storage.add_conversation = AsyncMock()
        return storage

    def test_can_change_mode(self, mock_storage):
        """Should be able to change privacy mode."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)
        assert wrapper.privacy_mode == PrivacyMode.NORMAL

        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        assert wrapper.privacy_mode == PrivacyMode.EPHEMERAL

    @pytest.mark.asyncio
    async def test_mode_change_affects_operations(self, mock_storage):
        """Mode change should immediately affect operations."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

        # Should work in NORMAL
        await wrapper.add_conversation("user", "Hello")
        assert mock_storage.add_conversation.call_count == 1

        # Switch to EPHEMERAL
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)

        # Should now block
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_conversation("user", "Hello again")


class TestWrapperFactory:
    """Tests for the factory function."""

    def test_wrap_storage_with_privacy(self):
        """Factory should create properly configured wrapper."""
        mock_storage = Mock()
        wrapper = wrap_storage_with_privacy(mock_storage, PrivacyMode.ANONYMOUS)

        assert isinstance(wrapper, PrivacyEnforcingStorage)
        assert wrapper.privacy_mode == PrivacyMode.ANONYMOUS


class TestDeprecationWarnings:
    """Tests that direct property access triggers deprecation warnings."""

    @pytest.fixture
    def mock_storage(self):
        storage = Mock()
        storage.db = Mock()
        storage.conversation = Mock()
        storage.files = Mock()
        return storage

    @pytest.fixture
    def normal_wrapper(self, mock_storage):
        return PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

    def test_db_access_warns(self, normal_wrapper):
        """Accessing .db should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = normal_wrapper.db
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "db" in str(w[0].message)
            assert "privacy" in str(w[0].message).lower()

    def test_conversation_access_warns(self, normal_wrapper):
        """Accessing .conversation should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = normal_wrapper.conversation
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "conversation" in str(w[0].message)

    def test_files_access_warns(self, normal_wrapper):
        """Accessing .files should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = normal_wrapper.files
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "files" in str(w[0].message)

    def test_db_path_no_warning(self, normal_wrapper):
        """Accessing .db_path should NOT emit a warning (non-sensitive)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = normal_wrapper.db_path
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

    def test_graph_no_warning(self, normal_wrapper):
        """Accessing .graph should NOT emit a warning (structural, not PII)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = normal_wrapper.graph
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

    def test_rag_no_warning(self, normal_wrapper):
        """Accessing .rag should NOT emit a warning (structural)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = normal_wrapper.rag
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0


class TestPrivacyAwareQueries:
    """Tests for privacy-aware query methods that replace direct db access."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage with db sub-object."""
        storage = Mock()
        storage.db = Mock()
        storage.db.fetchall = AsyncMock(return_value=[])
        storage.db.fetchone = AsyncMock(return_value=None)
        storage.db.execute_commit = AsyncMock(return_value=Mock(rowcount=1))
        storage.add_conversation = AsyncMock()
        storage.get_conversation_history = AsyncMock(return_value=[])
        storage.conversation = Mock()
        storage.conversation.encryption_enabled = True
        return storage

    # --- query_conversations ---

    @pytest.mark.asyncio
    async def test_query_conversations_normal_mode(self, mock_storage):
        """NORMAL mode should query the persistent database."""
        mock_storage.db.fetchall.return_value = [
            (1, "user", "Hello", None, "2026-01-01 12:00:00"),
            (2, "assistant", "Hi", None, "2026-01-01 12:00:01"),
        ]
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

        rows = await wrapper.query_conversations("agent-1")
        assert len(rows) == 2
        assert rows[0][1] == "user"
        mock_storage.db.fetchall.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_conversations_ephemeral_returns_empty(self, mock_storage):
        """EPHEMERAL mode should return empty list, not query db."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.EPHEMERAL)

        rows = await wrapper.query_conversations("agent-1")
        assert rows == []
        mock_storage.db.fetchall.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_conversations_isolated_returns_session(self, mock_storage):
        """ISOLATED mode should return session-local data."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.ISOLATED)

        await wrapper.add_conversation("user", "Hello")
        await wrapper.add_conversation("assistant", "Hi")

        rows = await wrapper.query_conversations("agent-1")
        assert len(rows) == 2
        assert rows[0][1] == "user"
        assert rows[1][1] == "assistant"
        # Should NOT touch persistent storage
        mock_storage.db.fetchall.assert_not_called()

    # --- query_conversation_start ---

    @pytest.mark.asyncio
    async def test_query_conversation_start_normal(self, mock_storage):
        """NORMAL mode should query the database for session start."""
        mock_storage.db.fetchone.return_value = ("2026-01-01 12:00:00",)
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

        result = await wrapper.query_conversation_start("1", "agent-1")
        assert result == ("2026-01-01 12:00:00",)
        mock_storage.db.fetchone.assert_called_once()
        assert mock_storage.db.fetchone.call_args.args[1] == (1, "agent-1")

    @pytest.mark.asyncio
    async def test_query_conversation_start_rejects_invalid_message_id(self, mock_storage):
        """NORMAL mode should not send non-row ids to persistent storage."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

        result = await wrapper.query_conversation_start("not-a-row-id", "agent-1")
        assert result is None
        mock_storage.db.fetchone.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_conversation_start_ephemeral_returns_none(self, mock_storage):
        """EPHEMERAL mode should return None."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.EPHEMERAL)

        result = await wrapper.query_conversation_start("1", "agent-1")
        assert result is None
        mock_storage.db.fetchone.assert_not_called()

    # --- query_conversation_messages ---

    @pytest.mark.asyncio
    async def test_query_conversation_messages_normal(self, mock_storage):
        """NORMAL mode should query messages from database."""
        mock_storage.db.fetchall.return_value = [
            (1, "user", "Hello", None, "2026-01-01 12:00:00"),
        ]
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

        rows = await wrapper.query_conversation_messages("agent-1", "2026-01-01 12:00:00", limit=50)
        assert len(rows) == 1
        mock_storage.db.fetchall.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_conversation_messages_ephemeral_returns_empty(self, mock_storage):
        """EPHEMERAL mode should return empty list."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.EPHEMERAL)

        rows = await wrapper.query_conversation_messages("agent-1", "2026-01-01 12:00:00")
        assert rows == []
        mock_storage.db.fetchall.assert_not_called()

    # --- query_last_conversation_row ---

    @pytest.mark.asyncio
    async def test_query_last_conversation_row_normal(self, mock_storage):
        """NORMAL mode should query for most recent row."""
        mock_storage.db.fetchone.return_value = (42, "2026-01-01 12:00:00")
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

        result = await wrapper.query_last_conversation_row("agent-1")
        assert result == (42, "2026-01-01 12:00:00")

    @pytest.mark.asyncio
    async def test_query_last_conversation_row_ephemeral(self, mock_storage):
        """EPHEMERAL mode should return None."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.EPHEMERAL)

        result = await wrapper.query_last_conversation_row("agent-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_last_conversation_row_isolated(self, mock_storage):
        """ISOLATED mode should return last session entry."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.ISOLATED)

        await wrapper.add_conversation("user", "Hello")
        await wrapper.add_conversation("assistant", "Hi")

        result = await wrapper.query_last_conversation_row("agent-1")
        assert result is not None
        assert result[0] == 1  # index of last entry

    # --- delete_conversation_message ---

    @pytest.mark.asyncio
    async def test_delete_message_normal_mode(self, mock_storage):
        """NORMAL mode should delete from persistent database and clean up pins."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)

        result = await wrapper.delete_conversation_message(42, "agent-1")
        assert result is True
        # Two execute_commit calls: one for the message deletion,
        # one for the sovereign pin cleanup.
        assert mock_storage.db.execute_commit.call_count == 2
        # First call deletes from conversation_history
        first_call = mock_storage.db.execute_commit.call_args_list[0]
        assert "conversation_history" in first_call[0][0]
        # Second call cleans up memory_pins
        second_call = mock_storage.db.execute_commit.call_args_list[1]
        assert "memory_pins" in second_call[0][0]

    @pytest.mark.asyncio
    async def test_delete_message_ephemeral_raises(self, mock_storage):
        """EPHEMERAL mode should raise PrivacyViolationError."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.EPHEMERAL)

        with pytest.raises(PrivacyViolationError):
            await wrapper.delete_conversation_message(42, "agent-1")

    @pytest.mark.asyncio
    async def test_delete_message_isolated_removes_from_session(self, mock_storage):
        """ISOLATED mode should remove from session storage."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.ISOLATED)

        await wrapper.add_conversation("user", "Hello")
        await wrapper.add_conversation("assistant", "Hi")
        assert len(wrapper._session_conversations) == 2

        result = await wrapper.delete_conversation_message(0, "agent-1")
        assert result is True
        assert len(wrapper._session_conversations) == 1
        assert wrapper._session_conversations[0]["content"] == "Hi"

    # --- encryption_enabled ---

    def test_encryption_enabled_returns_true(self, mock_storage):
        """Should return encryption status from underlying conversation store."""
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)
        assert wrapper.encryption_enabled is True

    def test_encryption_enabled_returns_false_when_missing(self):
        """Should return False when conversation store has no encryption."""
        mock_storage = Mock()
        mock_storage.conversation = Mock(spec=[])  # No encryption_enabled attr
        wrapper = PrivacyEnforcingStorage(mock_storage, PrivacyMode.NORMAL)
        assert wrapper.encryption_enabled is False
