"""
Tests for the Privacy-Enforcing Storage Wrapper.
"""

import pytest
from unittest.mock import Mock, AsyncMock
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
