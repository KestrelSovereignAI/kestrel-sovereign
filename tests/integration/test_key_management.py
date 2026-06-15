"""
Integration Tests for API Key Management System.

Tests ServiceKeyStorage, KeyManagementFeature, and KeyResolutionService
with real SQLite database operations.

All key storage is agent-scoped - each agent has isolated key storage.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from kestrel_sdk.tools.result import ToolResultStatus

# Mark all tests as integration tests
pytestmark = pytest.mark.integration

# Test constants
TEST_AGENT_DID = "did:pkh:eip155:1:0xTestAgent123"
OTHER_AGENT_DID = "did:pkh:eip155:1:0xOtherAgent456"


@pytest.fixture
async def temp_db():
    """Create a temporary SQLite database."""
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_keys.db"
        db = await AsyncDatabase.sqlite(str(db_path))
        yield db
        await db.close()


@pytest.fixture
def data_key():
    """Return a test data key."""
    return "test-encryption-key-for-unit-tests"


@pytest.fixture
async def key_storage(temp_db, data_key):
    """Create ServiceKeyStorage for test agent."""
    from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

    with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
        storage = ServiceKeyStorage(temp_db, TEST_AGENT_DID)
        yield storage


class TestServiceKeyStorage:
    """Tests for ServiceKeyStorage class."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_key(self, key_storage):
        """Test storing and retrieving an API key."""
        # Store key
        key_id = await key_storage.store_key(
            provider_id="openai",
            api_key="sk-test-12345",
        )

        assert key_id is not None

        # Retrieve key
        retrieved = await key_storage.get_key(provider_id="openai")

        assert retrieved == "sk-test-12345"

    @pytest.mark.asyncio
    async def test_key_encryption(self, key_storage, temp_db):
        """Test that keys are encrypted in the database."""
        await key_storage.store_key(
            provider_id="lighthouse",
            api_key="secret-lighthouse-key",
        )

        # Query raw database - key should be encrypted
        row = await temp_db.fetchone(
            "SELECT encrypted_key FROM agent_service_keys WHERE agent_did = ?",
            (TEST_AGENT_DID,),
        )

        assert row is not None
        encrypted_b64 = row[0]
        # Encrypted data should not contain the plaintext
        assert "secret-lighthouse-key" not in encrypted_b64

    @pytest.mark.asyncio
    async def test_key_not_found(self, key_storage):
        """Test KeyNotConfiguredError when key not found."""
        from kestrel_sovereign.security.service_key_storage import KeyNotConfiguredError

        with pytest.raises(KeyNotConfiguredError):
            await key_storage.get_key(provider_id="nonexistent")

    @pytest.mark.asyncio
    async def test_agent_isolation(self, temp_db, data_key):
        """Test that different agents have isolated key storage."""
        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage, KeyNotConfiguredError

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            storage1 = ServiceKeyStorage(temp_db, TEST_AGENT_DID)
            storage2 = ServiceKeyStorage(temp_db, OTHER_AGENT_DID)

            # Store key for agent 1
            await storage1.store_key(provider_id="openai", api_key="agent1-key")

            # Agent 1 can retrieve it
            key1 = await storage1.get_key(provider_id="openai")
            assert key1 == "agent1-key"

            # Agent 2 cannot access it
            with pytest.raises(KeyNotConfiguredError):
                await storage2.get_key(provider_id="openai")

    @pytest.mark.asyncio
    async def test_list_keys_no_secrets(self, key_storage):
        """Test listing keys does not expose secrets."""
        await key_storage.store_key(
            provider_id="openai",
            api_key="sk-secret-key",
            quota_limit=1000,
        )

        await key_storage.store_key(
            provider_id="lighthouse",
            api_key="lh-secret-key",
        )

        keys = await key_storage.list_keys()

        assert len(keys) == 2

        for key in keys:
            # Check no secret is exposed
            assert not hasattr(key, "api_key")
            assert not hasattr(key, "decrypted_key")
            # Check metadata is present
            assert key.provider_id in ["openai", "lighthouse"]
            assert key.is_active is True

    @pytest.mark.asyncio
    async def test_has_key(self, key_storage):
        """Test has_key method."""
        # Initially no key
        assert await key_storage.has_key(provider_id="openai") is False

        # Store key
        await key_storage.store_key(provider_id="openai", api_key="sk-test")

        # Now has key
        assert await key_storage.has_key(provider_id="openai") is True

    @pytest.mark.asyncio
    async def test_quota_tracking(self, key_storage):
        """Test quota tracking and enforcement."""
        await key_storage.store_key(
            provider_id="openai",
            api_key="sk-test",
            quota_limit=100,
        )

        # Check quota allows operation
        allowed = await key_storage.check_quota(
            provider_id="openai",
            units=50,
        )
        assert allowed is True

        # Record usage
        await key_storage.record_usage(
            provider_id="openai",
            operation="inference",
            units=50,
        )

        # Still within quota
        allowed = await key_storage.check_quota(
            provider_id="openai",
            units=50,
        )
        assert allowed is True

        # Record more usage
        await key_storage.record_usage(
            provider_id="openai",
            operation="inference",
            units=50,
        )

        # Now exceed quota
        allowed = await key_storage.check_quota(
            provider_id="openai",
            units=10,
        )
        assert allowed is False

    @pytest.mark.asyncio
    async def test_deactivate_key(self, key_storage):
        """Test key deactivation."""
        from kestrel_sovereign.security.service_key_storage import KeyNotConfiguredError

        await key_storage.store_key(
            provider_id="github",
            api_key="ghp_test",
        )

        # Deactivate
        await key_storage.deactivate_key(provider_id="github")

        # Should not be retrievable
        with pytest.raises(KeyNotConfiguredError):
            await key_storage.get_key(provider_id="github")

    @pytest.mark.asyncio
    async def test_delete_key(self, key_storage):
        """Test key hard deletion."""
        from kestrel_sovereign.security.service_key_storage import KeyNotConfiguredError

        await key_storage.store_key(
            provider_id="github",
            api_key="ghp_test",
        )

        # Delete
        await key_storage.delete_key(provider_id="github")

        # Should not be retrievable
        with pytest.raises(KeyNotConfiguredError):
            await key_storage.get_key(provider_id="github")

        # Should not appear in list
        keys = await key_storage.list_keys()
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_get_usage_history(self, key_storage):
        """Test retrieving usage history."""
        await key_storage.store_key(
            provider_id="lighthouse",
            api_key="lh-key",
        )

        # Record multiple usages
        for i in range(5):
            await key_storage.record_usage(
                provider_id="lighthouse",
                operation="upload",
                units=10,
                cost_estimate=0.01 * (i + 1),
            )

        usage = await key_storage.get_usage(
            provider_id="lighthouse",
            days=30,
        )

        assert len(usage) == 5
        assert all(u.operation == "upload" for u in usage)


class TestKeyResolutionService:
    """Tests for KeyResolutionService."""

    @pytest.mark.asyncio
    async def test_resolve_from_storage(self, key_storage, data_key):
        """Test resolving key from kestrel_sovereign.storage."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            await key_storage.store_key(
                provider_id="anthropic",
                api_key="sk-ant-test",
            )

            resolver = KeyResolutionService(
                storage=key_storage,
                agent_did=TEST_AGENT_DID,
            )

            key = await resolver.resolve_key("anthropic")
            assert key == "sk-ant-test"

    @pytest.mark.asyncio
    async def test_resolve_from_env_fallback(self):
        """Test fallback to environment variable."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        resolver = KeyResolutionService(storage=None, agent_did=TEST_AGENT_DID)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}):
            key = await resolver.resolve_key("openai")
            assert key == "sk-env-key"

    @pytest.mark.asyncio
    async def test_resolve_not_found_raises(self):
        """Test that missing key raises error when required."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService, KeyNotConfiguredError

        resolver = KeyResolutionService(storage=None, agent_did=TEST_AGENT_DID)

        # Clear env
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(KeyNotConfiguredError):
                await resolver.resolve_key("nonexistent_provider")

    @pytest.mark.asyncio
    async def test_resolve_not_found_returns_none(self):
        """Test that missing key returns None when not required."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        resolver = KeyResolutionService(storage=None, agent_did=TEST_AGENT_DID)

        with patch.dict(os.environ, {}, clear=True):
            key = await resolver.resolve_key("nonexistent_provider", require=False)
            assert key is None

    @pytest.mark.asyncio
    async def test_get_key_info(self, key_storage, data_key):
        """Test getting key metadata."""
        from kestrel_sovereign.services.key_resolution import KeyResolutionService

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            await key_storage.store_key(
                provider_id="runpod",
                api_key="rp-key",
                quota_limit=500,
            )

            resolver = KeyResolutionService(
                storage=key_storage,
                agent_did=TEST_AGENT_DID,
            )

            info = await resolver.get_key_info("runpod")

            assert info is not None
            assert info["provider"] == "runpod"
            assert info["quota_limit"] == 500
            assert info["source"] == "storage"


class TestKeyManagementFeature:
    """Tests for KeyManagementFeature agent tools."""

    @pytest.fixture
    def mock_agent(self, temp_db):
        """Create a mock agent with storage."""
        agent = MagicMock()
        agent.storage = MagicMock()
        agent.storage.db = temp_db
        agent.did = TEST_AGENT_DID
        return agent

    @pytest.mark.asyncio
    async def test_add_service_key_tool(self, mock_agent, temp_db, data_key):
        """Test the add_service_key tool."""
        from kestrel_sovereign.features.keys import KeyManagementFeature

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            feature = KeyManagementFeature(mock_agent)
            await feature.initialize()

            result = await feature.add_service_key(
                provider="openai",
                api_key="sk-tool-test",
                quota_limit=1000,
            )

            assert result.status is ToolResultStatus.OK
            assert result.data["provider"] == "openai"
            assert result.data["quota_limit"] == 1000

    @pytest.mark.asyncio
    async def test_list_service_keys_tool(self, mock_agent, data_key):
        """Test the list_service_keys tool."""
        from kestrel_sovereign.features.keys import KeyManagementFeature

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            feature = KeyManagementFeature(mock_agent)
            await feature.initialize()

            # Add some keys first
            await feature.add_service_key(provider="openai", api_key="sk-1")
            await feature.add_service_key(provider="anthropic", api_key="sk-2")

            result = await feature.list_service_keys()

            assert result.status is ToolResultStatus.OK
            assert result.data["total"] == 2
            assert len(result.data["keys"]) == 2

    @pytest.mark.asyncio
    async def test_list_providers_tool(self, mock_agent, data_key):
        """Test the list_providers tool."""
        from kestrel_sovereign.features.keys import KeyManagementFeature

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            feature = KeyManagementFeature(mock_agent)
            await feature.initialize()

            result = await feature.list_providers()

            assert result.status is ToolResultStatus.OK
            assert result.data["total"] >= 6  # openrouter, lighthouse, openai, anthropic, github, runpod, vastai
            provider_ids = [p["id"] for p in result.data["providers"]]
            assert "openrouter" in provider_ids
            assert "openai" in provider_ids

    @pytest.mark.asyncio
    async def test_remove_service_key_tool(self, mock_agent, data_key):
        """Test the remove_service_key tool."""
        from kestrel_sovereign.features.keys import KeyManagementFeature

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            feature = KeyManagementFeature(mock_agent)
            await feature.initialize()

            # Add then remove
            await feature.add_service_key(provider="github", api_key="ghp-test")
            result = await feature.remove_service_key(provider="github")

            assert result.status is ToolResultStatus.OK
            assert "deactivated" in result.confirmation.lower()

    @pytest.mark.asyncio
    async def test_get_key_internal_api(self, mock_agent, data_key):
        """Test the internal get_key API for other features."""
        from kestrel_sovereign.features.keys import KeyManagementFeature

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            feature = KeyManagementFeature(mock_agent)
            await feature.initialize()

            await feature.add_service_key(
                provider="lighthouse",
                api_key="lh-internal-test",
            )

            # Use internal API
            key = await feature.get_key("lighthouse")
            assert key == "lh-internal-test"


class TestKnownProviders:
    """Tests for known provider definitions."""

    def test_all_providers_have_required_fields(self):
        """Test that all known providers have required metadata."""
        from kestrel_sovereign.security.service_key_storage import KNOWN_PROVIDERS

        required_fields = ["name"]

        for provider_id, info in KNOWN_PROVIDERS.items():
            for field in required_fields:
                assert field in info, f"Provider {provider_id} missing {field}"

    def test_provider_names_are_lowercase(self):
        """Test that provider IDs are lowercase."""
        from kestrel_sovereign.security.service_key_storage import KNOWN_PROVIDERS

        for provider_id in KNOWN_PROVIDERS:
            assert provider_id == provider_id.lower()


class TestMasterKeyNotConfigured:
    """Tests for behavior when master key is not configured."""

    @pytest.mark.asyncio
    async def test_storage_requires_agent_did(self, temp_db, data_key):
        """Test that ServiceKeyStorage requires agent_did."""
        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            with pytest.raises(ValueError, match="agent_did is required"):
                ServiceKeyStorage(temp_db, "")

    @pytest.mark.asyncio
    async def test_storage_fails_without_master_key(self, temp_db):
        """Test that encryption fails without master key."""
        from kestrel_sovereign.security.agent_encryption import MasterKeyNotConfiguredError, encrypt

        # Clear the env var
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MasterKeyNotConfiguredError):
                encrypt(TEST_AGENT_DID, "service-keys", b"test")

    @pytest.mark.asyncio
    async def test_feature_degrades_gracefully(self, data_key):
        """Test that feature degrades when storage not available."""
        from kestrel_sovereign.features.keys import KeyManagementFeature

        # Mock agent without proper storage
        agent = MagicMock()
        agent.storage = None
        agent.did = None  # No DID

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            feature = KeyManagementFeature(agent)
            await feature.initialize()

            result = await feature.list_service_keys()

            assert result.status is ToolResultStatus.ERROR
            assert "not available" in result.error


@pytest.fixture(params=["sqlite", "postgres"])
async def dual_backend_db(request, tmp_path):
    """An ``AsyncDatabase`` on SQLite and (when available) PostgreSQL.

    Regression coverage for #1779: ``ServiceKeyStorage.list_keys`` /
    ``get_usage`` previously used SQLite-only SQL and value handling, so they
    500'd on Postgres. The original suite only ran on SQLite, hiding the bug.
    PostgreSQL is skipped when no test DSN is configured.
    """
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    if request.param == "sqlite":
        db = await AsyncDatabase.sqlite(str(tmp_path / "keys.db"))
        try:
            yield db
        finally:
            await db.close()
    else:
        dsn = (
            os.environ.get("TEST_POSTGRES_URL")
            or os.environ.get("KESTREL_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
        )
        if not dsn:
            pytest.skip("TEST_POSTGRES_URL/KESTREL_DATABASE_URL/DATABASE_URL required for PostgreSQL")
        try:
            db = await AsyncDatabase.postgres(dsn)
        except Exception as e:  # pragma: no cover - infra dependent
            pytest.skip(f"PostgreSQL not available: {e}")
        # PG (unlike the per-test SQLite tmpfile) may be a shared DB, so clear
        # only THIS synthetic test agent's rows — never truncate the tables.
        await db.execute(
            "DELETE FROM service_key_usage WHERE key_id IN "
            "(SELECT id FROM agent_service_keys WHERE agent_did = ?)",
            (TEST_AGENT_DID,),
        )
        await db.execute("DELETE FROM agent_service_keys WHERE agent_did = ?", (TEST_AGENT_DID,))
        try:
            yield db
        finally:
            await db.close()


@pytest.mark.integration
@pytest.mark.dual_backend
class TestServiceKeyStorageBackendCompat:
    """list_keys/get_usage must work on both SQLite and PostgreSQL (#1779)."""

    @pytest.mark.asyncio
    async def test_list_keys_returns_datetimes(self, dual_backend_db, data_key):
        from datetime import datetime
        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            storage = ServiceKeyStorage(dual_backend_db, TEST_AGENT_DID)
            await storage.store_key("runpod", "sk-test-runpod-123")

            # On PG this raised TypeError: fromisoformat: argument must be str
            keys = await storage.list_keys()

        assert len(keys) == 1
        assert keys[0].provider_id == "runpod"
        assert isinstance(keys[0].created_at, datetime)

    @pytest.mark.asyncio
    async def test_get_usage_window(self, dual_backend_db, data_key):
        from datetime import datetime
        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        with patch.dict(os.environ, {"KESTREL_DATA_KEY": data_key}):
            storage = ServiceKeyStorage(dual_backend_db, TEST_AGENT_DID)
            await storage.store_key("runpod", "sk-test-runpod-123")
            await storage.record_usage("runpod", "inference", units=3, cost_estimate=0.01)

            # On PG this raised: function datetime(unknown, unknown) does not exist
            usage = await storage.get_usage("runpod", days=30)

        assert len(usage) == 1
        assert usage[0].operation == "inference"
        assert usage[0].units_consumed == 3
        assert isinstance(usage[0].recorded_at, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
