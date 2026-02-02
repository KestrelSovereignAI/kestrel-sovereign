"""
Unit tests for LayeredKeyResolver.

Tests three-tier key resolution priority: Agent → User → Platform.
Each tier has different billing modes.
"""

import pytest
import pytest_asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.services.layered_key_resolver import (
    LayeredKeyResolver,
    KeyResolutionResult,
    KeyNotConfiguredError,
    resolve_key,
)


class MockAsyncContextManager:
    """Async context manager for mocking pool.acquire()."""

    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_mock_pool(conn=None):
    """Create a mock database pool with proper async context manager."""
    if conn is None:
        conn = MagicMock()
        conn.fetchrow = AsyncMock(return_value=None)
        conn.fetch = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value="INSERT 1")
        conn.fetchval = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire.return_value = MockAsyncContextManager(conn)
    return pool, conn


class TestKeyResolutionResult:
    """Tests for KeyResolutionResult dataclass."""

    def test_agent_result_fields(self):
        """Test agent key result has correct billing mode."""
        result = KeyResolutionResult(
            api_key="sk-agent-key",
            source="agent",
            source_id="did:example:123",
            billing_mode="agent_wallet",
            margin_pct=Decimal("0"),
        )

        assert result.api_key == "sk-agent-key"
        assert result.source == "agent"
        assert result.source_id == "did:example:123"
        assert result.billing_mode == "agent_wallet"
        assert result.margin_pct == Decimal("0")

    def test_user_result_fields(self):
        """Test user BYOK result has no_charge billing mode."""
        result = KeyResolutionResult(
            api_key="sk-user-byok",
            source="user",
            source_id="user-uuid-123",
            billing_mode="no_charge",
            margin_pct=Decimal("0"),
        )

        assert result.source == "user"
        assert result.billing_mode == "no_charge"
        assert result.margin_pct == Decimal("0")

    def test_platform_result_fields(self):
        """Test platform key result has wallet_debit billing mode with margin."""
        result = KeyResolutionResult(
            api_key="sk-platform-pool",
            source="platform",
            source_id=None,
            billing_mode="wallet_debit",
            margin_pct=Decimal("0.15"),
        )

        assert result.source == "platform"
        assert result.source_id is None
        assert result.billing_mode == "wallet_debit"
        assert result.margin_pct == Decimal("0.15")


class TestKeyNotConfiguredError:
    """Tests for KeyNotConfiguredError exception."""

    def test_error_includes_provider(self):
        """Test exception stores provider name."""
        error = KeyNotConfiguredError("openrouter")
        assert error.provider == "openrouter"
        assert "openrouter" in str(error)

    def test_error_custom_message(self):
        """Test exception with custom message."""
        error = KeyNotConfiguredError("openai", "Custom error message")
        assert error.provider == "openai"
        assert "Custom error message" in str(error)


class TestLayeredKeyResolverAgentTier:
    """Tests for Tier 1: Agent key resolution."""

    @pytest.mark.asyncio
    async def test_agent_key_found(self):
        """Test agent key is returned when available."""
        pool, conn = create_mock_pool()

        # Mock agent storage
        mock_agent_storage = MagicMock()
        mock_agent_storage.get_key = AsyncMock(return_value="sk-agent-key-123")

        resolver = LayeredKeyResolver(pool, agent_storage=mock_agent_storage)

        result = await resolver.resolve(
            provider="openrouter",
            agent_did="did:example:agent1",
        )

        assert result is not None
        assert result.api_key == "sk-agent-key-123"
        assert result.source == "agent"
        assert result.source_id == "did:example:agent1"
        assert result.billing_mode == "agent_wallet"
        assert result.margin_pct == Decimal("0")

    @pytest.mark.asyncio
    async def test_agent_key_not_found_falls_through(self):
        """Test resolution falls through when agent has no key."""
        pool, conn = create_mock_pool()

        # Mock agent storage returning None
        mock_agent_storage = MagicMock()
        mock_agent_storage.get_key = AsyncMock(return_value=None)

        resolver = LayeredKeyResolver(pool, agent_storage=mock_agent_storage)

        # Should raise KeyNotConfiguredError since no other tiers available
        with pytest.raises(KeyNotConfiguredError) as exc_info:
            await resolver.resolve(
                provider="openrouter",
                agent_did="did:example:agent1",
            )

        assert exc_info.value.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_no_agent_storage_skips_tier(self):
        """Test tier 1 is skipped when no agent storage configured."""
        pool, conn = create_mock_pool()

        # No agent_storage provided
        resolver = LayeredKeyResolver(pool, agent_storage=None)

        # Should skip agent tier and fail (no other tiers configured)
        with pytest.raises(KeyNotConfiguredError):
            await resolver.resolve(
                provider="openrouter",
                agent_did="did:example:agent1",
            )


class TestLayeredKeyResolverUserTier:
    """Tests for Tier 2: User BYOK key resolution."""

    @pytest.mark.asyncio
    async def test_user_byok_found_with_passphrase(self):
        """Test user BYOK is returned when passphrase provided."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.user_key_storage.UserKeyStorage"
        ) as MockUserStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=True)
            mock_instance.get_key = AsyncMock(return_value="sk-user-byok-key")
            MockUserStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            result = await resolver.resolve(
                provider="openai",
                user_id="user-uuid-123",
                user_passphrase="my-secret-passphrase",
            )

            assert result is not None
            assert result.api_key == "sk-user-byok-key"
            assert result.source == "user"
            assert result.source_id == "user-uuid-123"
            assert result.billing_mode == "no_charge"
            assert result.margin_pct == Decimal("0")

    @pytest.mark.asyncio
    async def test_user_byok_no_passphrase_skips(self):
        """Test user BYOK is skipped when passphrase not provided."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.user_key_storage.UserKeyStorage"
        ) as MockUserStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=True)
            MockUserStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            # No passphrase provided - should skip user tier
            with pytest.raises(KeyNotConfiguredError):
                await resolver.resolve(
                    provider="openai",
                    user_id="user-uuid-123",
                    user_passphrase=None,  # No passphrase
                )

    @pytest.mark.asyncio
    async def test_user_byok_wrong_passphrase_skips(self):
        """Test user BYOK is skipped on wrong passphrase."""
        pool, conn = create_mock_pool()

        from kestrel_sovereign.security.user_key_storage import DecryptionError

        with patch(
            "kestrel_sovereign.security.user_key_storage.UserKeyStorage"
        ) as MockUserStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=True)
            mock_instance.get_key = AsyncMock(
                side_effect=DecryptionError("Wrong passphrase")
            )
            MockUserStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            # Wrong passphrase should cause decryption error, skip to next tier
            with pytest.raises(KeyNotConfiguredError):
                await resolver.resolve(
                    provider="openai",
                    user_id="user-uuid-123",
                    user_passphrase="wrong-passphrase",
                )


class TestLayeredKeyResolverPlatformTier:
    """Tests for Tier 3: Platform vending machine key resolution."""

    @pytest.mark.asyncio
    async def test_platform_key_found(self):
        """Test platform key is returned when available."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=True)
            mock_instance.get_key = AsyncMock(return_value="sk-platform-pool-key")
            mock_instance.get_margin = AsyncMock(return_value=Decimal("0.12"))
            MockPlatformStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            result = await resolver.resolve(provider="openrouter")

            assert result is not None
            assert result.api_key == "sk-platform-pool-key"
            assert result.source == "platform"
            assert result.source_id is None
            assert result.billing_mode == "wallet_debit"
            assert result.margin_pct == Decimal("0.12")

    @pytest.mark.asyncio
    async def test_platform_key_not_found(self):
        """Test KeyNotConfiguredError when no platform key."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=False)
            MockPlatformStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            with pytest.raises(KeyNotConfiguredError) as exc_info:
                await resolver.resolve(provider="nonexistent")

            assert exc_info.value.provider == "nonexistent"

    @pytest.mark.asyncio
    async def test_platform_key_master_not_configured(self):
        """Test graceful handling when platform master key not set."""
        pool, conn = create_mock_pool()

        from kestrel_sovereign.security.platform_key_storage import MasterKeyNotConfiguredError

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=True)
            mock_instance.get_key = AsyncMock(
                side_effect=MasterKeyNotConfiguredError("Not configured")
            )
            MockPlatformStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            with pytest.raises(KeyNotConfiguredError):
                await resolver.resolve(provider="openrouter")


class TestLayeredKeyResolverPriority:
    """Tests for key resolution priority order."""

    @pytest.mark.asyncio
    async def test_agent_takes_priority_over_user(self):
        """Test agent key is used even when user BYOK available."""
        pool, conn = create_mock_pool()

        # Mock agent storage with key
        mock_agent_storage = MagicMock()
        mock_agent_storage.get_key = AsyncMock(return_value="sk-agent-key")

        with patch(
            "kestrel_sovereign.security.user_key_storage.UserKeyStorage"
        ) as MockUserStorage:
            mock_user = MagicMock()
            mock_user.has_key = AsyncMock(return_value=True)
            mock_user.get_key = AsyncMock(return_value="sk-user-key")
            MockUserStorage.return_value = mock_user

            resolver = LayeredKeyResolver(pool, agent_storage=mock_agent_storage)

            result = await resolver.resolve(
                provider="openrouter",
                agent_did="did:example:agent1",
                user_id="user-123",
                user_passphrase="passphrase",
            )

            assert result.source == "agent"
            assert result.api_key == "sk-agent-key"

    @pytest.mark.asyncio
    async def test_user_takes_priority_over_platform(self):
        """Test user BYOK is used even when platform key available."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.user_key_storage.UserKeyStorage"
        ) as MockUserStorage:
            mock_user = MagicMock()
            mock_user.has_key = AsyncMock(return_value=True)
            mock_user.get_key = AsyncMock(return_value="sk-user-key")
            MockUserStorage.return_value = mock_user

            with patch(
                "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
            ) as MockPlatformStorage:
                mock_platform = MagicMock()
                mock_platform.has_key = AsyncMock(return_value=True)
                mock_platform.get_key = AsyncMock(return_value="sk-platform-key")
                mock_platform.get_margin = AsyncMock(return_value=Decimal("0.15"))
                MockPlatformStorage.return_value = mock_platform

                resolver = LayeredKeyResolver(pool)

                result = await resolver.resolve(
                    provider="openrouter",
                    user_id="user-123",
                    user_passphrase="passphrase",
                )

                assert result.source == "user"
                assert result.api_key == "sk-user-key"
                assert result.billing_mode == "no_charge"

    @pytest.mark.asyncio
    async def test_fallback_to_platform(self):
        """Test fallback to platform when no agent or user key."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.user_key_storage.UserKeyStorage"
        ) as MockUserStorage:
            mock_user = MagicMock()
            mock_user.has_key = AsyncMock(return_value=False)
            MockUserStorage.return_value = mock_user

            with patch(
                "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
            ) as MockPlatformStorage:
                mock_platform = MagicMock()
                mock_platform.has_key = AsyncMock(return_value=True)
                mock_platform.get_key = AsyncMock(return_value="sk-platform-key")
                mock_platform.get_margin = AsyncMock(return_value=Decimal("0.15"))
                MockPlatformStorage.return_value = mock_platform

                resolver = LayeredKeyResolver(pool)

                result = await resolver.resolve(
                    provider="openrouter",
                    user_id="user-123",
                    user_passphrase="passphrase",
                )

                assert result.source == "platform"
                assert result.billing_mode == "wallet_debit"


class TestLayeredKeyResolverOptionalRequire:
    """Tests for require=False behavior."""

    @pytest.mark.asyncio
    async def test_require_false_returns_none(self):
        """Test require=False returns None instead of raising."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=False)
            MockPlatformStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            result = await resolver.resolve(
                provider="nonexistent",
                require=False,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_require_true_raises(self):
        """Test require=True (default) raises KeyNotConfiguredError."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=False)
            MockPlatformStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            with pytest.raises(KeyNotConfiguredError):
                await resolver.resolve(provider="nonexistent")


class TestLayeredKeyResolverHasAnyKey:
    """Tests for has_any_key method."""

    @pytest.mark.asyncio
    async def test_has_any_key_agent(self):
        """Test has_any_key returns True when agent has key."""
        pool, conn = create_mock_pool()

        mock_agent_storage = MagicMock()
        mock_agent_storage.has_key = AsyncMock(return_value=True)

        resolver = LayeredKeyResolver(pool, agent_storage=mock_agent_storage)

        result = await resolver.has_any_key(
            provider="openrouter",
            agent_did="did:example:agent1",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_has_any_key_platform(self):
        """Test has_any_key returns True when platform has key."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=True)
            MockPlatformStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            result = await resolver.has_any_key(provider="openrouter")

            assert result is True

    @pytest.mark.asyncio
    async def test_has_any_key_none(self):
        """Test has_any_key returns False when no source has key."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=False)
            MockPlatformStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            result = await resolver.has_any_key(provider="nonexistent")

            assert result is False


class TestLayeredKeyResolverGetAvailableSources:
    """Tests for get_available_sources method."""

    @pytest.mark.asyncio
    async def test_get_available_sources_all(self):
        """Test get_available_sources returns all source statuses."""
        pool, conn = create_mock_pool()

        mock_agent_storage = MagicMock()
        mock_agent_storage.has_key = AsyncMock(return_value=True)

        with patch(
            "kestrel_sovereign.security.user_key_storage.UserKeyStorage"
        ) as MockUserStorage:
            mock_user = MagicMock()
            mock_user.has_key = AsyncMock(return_value=True)
            MockUserStorage.return_value = mock_user

            with patch(
                "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
            ) as MockPlatformStorage:
                mock_platform = MagicMock()
                mock_platform.has_key = AsyncMock(return_value=True)
                mock_platform.get_margin = AsyncMock(return_value=Decimal("0.15"))
                MockPlatformStorage.return_value = mock_platform

                resolver = LayeredKeyResolver(pool, agent_storage=mock_agent_storage)

                result = await resolver.get_available_sources(
                    provider="openrouter",
                    user_id="user-123",
                    agent_did="did:example:agent1",
                )

                assert result["agent"] is True
                assert result["user"] is True
                assert result["platform"] is True
                assert result["platform_margin"] == Decimal("0.15")

    @pytest.mark.asyncio
    async def test_get_available_sources_partial(self):
        """Test get_available_sources with only some sources available."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_platform = MagicMock()
            mock_platform.has_key = AsyncMock(return_value=True)
            mock_platform.get_margin = AsyncMock(return_value=Decimal("0.20"))
            MockPlatformStorage.return_value = mock_platform

            resolver = LayeredKeyResolver(pool)

            result = await resolver.get_available_sources(
                provider="openrouter",
            )

            assert result["agent"] is False
            assert result["user"] is False
            assert result["platform"] is True
            assert result["platform_margin"] == Decimal("0.20")


class TestResolveKeyConvenienceFunction:
    """Tests for resolve_key convenience function."""

    @pytest.mark.asyncio
    async def test_resolve_key_function(self):
        """Test convenience function delegates to LayeredKeyResolver."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=True)
            mock_instance.get_key = AsyncMock(return_value="sk-platform-key")
            mock_instance.get_margin = AsyncMock(return_value=Decimal("0.10"))
            MockPlatformStorage.return_value = mock_instance

            result = await resolve_key(
                pool=pool,
                provider="openrouter",
            )

            assert result is not None
            assert result.api_key == "sk-platform-key"
            assert result.source == "platform"


class TestProviderNormalization:
    """Tests for provider name normalization."""

    @pytest.mark.asyncio
    async def test_provider_lowercased(self):
        """Test provider names are lowercased for consistency."""
        pool, conn = create_mock_pool()

        with patch(
            "kestrel_sovereign.security.platform_key_storage.PlatformKeyStorage"
        ) as MockPlatformStorage:
            mock_instance = MagicMock()
            mock_instance.has_key = AsyncMock(return_value=True)
            mock_instance.get_key = AsyncMock(return_value="sk-key")
            mock_instance.get_margin = AsyncMock(return_value=Decimal("0.15"))
            MockPlatformStorage.return_value = mock_instance

            resolver = LayeredKeyResolver(pool)

            # Use uppercase provider
            result = await resolver.resolve(provider="OPENROUTER")

            # Should still work (internally lowercased)
            assert result is not None
            mock_instance.has_key.assert_called_with("openrouter")
