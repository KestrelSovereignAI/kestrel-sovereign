"""
Tests for key resolution security patterns.

Verifies fail-closed behavior: after max retries, requests are DENIED (not allowed).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


class MockStorage:
    """Mock storage for testing."""

    def __init__(self, fail_count: int = 0, should_allow: bool = True):
        self.fail_count = fail_count
        self.should_allow = should_allow
        self.call_count = 0

    async def check_quota(self, provider_id: str, units: int) -> bool:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise ConnectionError("Database connection failed")
        return self.should_allow


@pytest.fixture
def mock_storage():
    """Create a mock storage."""
    return MockStorage()


class TestQuotaCheckRetryBehavior:
    """Tests for quota check retry-then-deny pattern."""

    @pytest.mark.asyncio
    async def test_quota_check_succeeds_on_first_try(self):
        """Quota check should succeed if storage works on first try."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        mock = MockStorage(fail_count=0, should_allow=True)
        service._storage = mock

        result = await service.check_quota("openai", units=1)

        assert result is True
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_quota_check_succeeds_after_retry(self):
        """Quota check should succeed after transient failures."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        # Fail twice, then succeed
        mock = MockStorage(fail_count=2, should_allow=True)
        service._storage = mock

        result = await service.check_quota("openai", units=1, max_retries=3)

        assert result is True
        assert mock.call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_quota_check_denies_after_max_retries(self):
        """Quota check should DENY after max retries exhausted (fail-closed)."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        # Fail all 3 times
        mock = MockStorage(fail_count=3, should_allow=True)
        service._storage = mock

        result = await service.check_quota("openai", units=1, max_retries=3)

        # Should be denied (fail-closed), not allowed
        assert result is False
        assert mock.call_count == 3

    @pytest.mark.asyncio
    async def test_quota_check_denies_after_max_retries_even_with_more_failures(self):
        """Should stop retrying after max_retries and deny."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        # Fail 100 times (but we only retry 3)
        mock = MockStorage(fail_count=100, should_allow=True)
        service._storage = mock

        result = await service.check_quota("openai", units=1, max_retries=3)

        assert result is False
        assert mock.call_count == 3  # Should stop at 3, not continue

    @pytest.mark.asyncio
    async def test_quota_check_allows_when_no_storage(self):
        """When no storage is configured, quota enforcement is disabled."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        service._storage = None  # No storage

        result = await service.check_quota("openai", units=1)

        # No storage = no enforcement, allow the request
        assert result is True

    @pytest.mark.asyncio
    async def test_quota_check_denies_when_quota_exceeded(self):
        """Should deny when quota is actually exceeded (not just on error)."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        mock = MockStorage(fail_count=0, should_allow=False)  # Quota exceeded
        service._storage = mock

        result = await service.check_quota("openai", units=1)

        assert result is False
        assert mock.call_count == 1


class TestQuotaCheckLogging:
    """Tests for quota check logging behavior."""

    @pytest.mark.asyncio
    async def test_logs_warning_on_transient_failure(self):
        """Should log warning on each retry attempt."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        mock = MockStorage(fail_count=1, should_allow=True)
        service._storage = mock

        with patch("kestrel_sovereign.services.key_resolution.logger") as mock_logger:
            await service.check_quota("openai", units=1, max_retries=3)

            # Should have logged 1 warning for the failed attempt
            assert mock_logger.warning.call_count == 1

    @pytest.mark.asyncio
    async def test_logs_error_when_denying_after_retries(self):
        """Should log error when denying after max retries."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        mock = MockStorage(fail_count=3, should_allow=True)
        service._storage = mock

        with patch("kestrel_sovereign.services.key_resolution.logger") as mock_logger:
            result = await service.check_quota("openai", units=1, max_retries=3)

            assert result is False
            # Should have logged warnings for attempts + error for denial
            assert mock_logger.warning.call_count == 3
            assert mock_logger.error.call_count == 1


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    @pytest.mark.asyncio
    async def test_backoff_delay_between_retries(self):
        """Should wait with exponential backoff between retries."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        service = KeyResolutionService()
        mock = MockStorage(fail_count=2, should_allow=True)
        service._storage = mock

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await service.check_quota("openai", units=1, max_retries=3)

            # Should have called sleep twice (after attempt 1 and 2)
            assert mock_sleep.call_count == 2

            # First delay: 0.5 * (2^0) = 0.5s
            # Second delay: 0.5 * (2^1) = 1.0s
            calls = [call.args[0] for call in mock_sleep.call_args_list]
            assert calls[0] == pytest.approx(0.5, rel=0.1)
            assert calls[1] == pytest.approx(1.0, rel=0.1)
