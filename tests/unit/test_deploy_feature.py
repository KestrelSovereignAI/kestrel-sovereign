"""
Unit tests for Deploy Feature.

Tests the DeployFeature class with config loading, profile management,
and graceful degradation patterns.
"""

import os
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.deploy.core import DeployManagerCore
from kestrel_sovereign.features.deploy.feature import DeployFeature
from kestrel_sovereign.features.deploy.manager import DeployManager
from kestrel_sovereign.features.deploy.models import (
    DeploymentProfile,
    DeployProviderType,
    DeployStatus,
    DeploymentSession,
)


@pytest.fixture
def sample_config():
    """Sample deploy config for testing."""
    return {
        "manager": {
            "default_provider": "cloudrun",
            "gcp_project_id": "test-project",
            "image_name": "kestrel",
            "build_strategy": "prebuilt",
            "health_check_timeout_seconds": 120,
            "health_check_path": "/health",
        },
        "profiles": {
            "dev": {
                "name": "Cloud Run Development",
                "provider": "cloudrun",
                "service_name": "kestrel-dev",
                "region": "us-central1",
                "min_instances": 0,
                "max_instances": 10,
                "memory": "2Gi",
                "cpu": 2,
                "port": 8080,
                "timeout": 300,
                "concurrency": 80,
                "env_vars": {
                    "KESTREL_ENV": "development",
                    "KESTREL_DB_BACKEND": "sqlite",
                },
                "secrets": {
                    "OPENAI_API_KEY": "kestrel-openai-key:latest",
                },
            },
            "prod": {
                "name": "Cloud Run Production",
                "provider": "cloudrun",
                "service_name": "kestrel-prod",
                "region": "us-central1",
                "min_instances": 1,
                "max_instances": 100,
                "memory": "2Gi",
                "cpu": 2,
                "port": 8080,
                "timeout": 300,
                "concurrency": 80,
                "env_vars": {
                    "KESTREL_ENV": "production",
                },
                "secrets": {
                    "OPENAI_API_KEY": "kestrel-openai-key:latest",
                },
            },
        },
    }


class TestDeployManagerCoreInit:
    """Test DeployManagerCore initialization."""

    def test_init_with_config(self, sample_config):
        """Test initialization with explicit config."""
        # Clear GCP_PROJECT_ID env var to test config-only initialization
        with patch.dict(os.environ, {}, clear=True):
            manager = DeployManagerCore(config=sample_config)

            assert manager.default_provider == "cloudrun"
            assert manager.gcp_project_id == "test-project"
            assert manager.image_name == "kestrel"
            assert manager.build_strategy == "prebuilt"
            assert manager.health_check_timeout == 120
            assert manager.health_check_path == "/health"

    def test_init_with_env_var(self, sample_config):
        """Test initialization with GCP_PROJECT_ID env var override."""
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "env-project"}):
            manager = DeployManagerCore(config=sample_config)
            assert manager.gcp_project_id == "env-project"

    def test_init_without_config(self):
        """Test initialization without config falls back to load_config."""
        with patch("kestrel_sovereign.features.deploy.core.load_config") as mock_load:
            mock_load.return_value = {}
            manager = DeployManagerCore()
            mock_load.assert_called_once_with("deploy_config.toml")


class TestDeployManagerCoreProfiles:
    """Test profile loading and management."""

    def test_load_profiles(self, sample_config):
        """Test profile loading from config."""
        manager = DeployManagerCore(config=sample_config)

        assert len(manager.profiles) == 2
        assert "dev" in manager.profiles
        assert "prod" in manager.profiles

        dev_profile = manager.profiles["dev"]
        assert dev_profile.service_name == "kestrel-dev"
        assert dev_profile.provider == DeployProviderType.CLOUD_RUN
        assert dev_profile.min_instances == 0
        assert dev_profile.max_instances == 10
        assert dev_profile.is_scale_to_zero is True

        prod_profile = manager.profiles["prod"]
        assert prod_profile.service_name == "kestrel-prod"
        assert prod_profile.min_instances == 1
        assert prod_profile.is_scale_to_zero is False

    def test_get_profile_valid(self, sample_config):
        """Test getting a valid profile."""
        manager = DeployManagerCore(config=sample_config)
        profile = manager.get_profile("dev")

        assert profile.service_name == "kestrel-dev"
        assert profile.region == "us-central1"

    def test_get_profile_invalid(self, sample_config):
        """Test getting an invalid profile raises error."""
        from kestrel_sovereign.features.deploy.models import DeployManagerError

        manager = DeployManagerCore(config=sample_config)

        with pytest.raises(DeployManagerError) as exc_info:
            manager.get_profile("nonexistent")

        assert "Unknown profile 'nonexistent'" in str(exc_info.value)
        assert "dev, prod" in str(exc_info.value)

    def test_env_var_expansion(self, sample_config):
        """Test environment variable expansion in config."""
        # Add env var reference to config
        sample_config["profiles"]["dev"]["env_vars"]["API_KEY"] = "${TEST_API_KEY}"

        with patch.dict(os.environ, {"TEST_API_KEY": "secret-123"}):
            manager = DeployManagerCore(config=sample_config)
            dev_profile = manager.profiles["dev"]

            assert dev_profile.env_vars["API_KEY"] == "secret-123"

    def test_env_var_expansion_missing(self, sample_config):
        """Test env var expansion when var is missing."""
        sample_config["profiles"]["dev"]["env_vars"]["API_KEY"] = "${MISSING_VAR}"

        with patch.dict(os.environ, {}, clear=True):
            manager = DeployManagerCore(config=sample_config)
            dev_profile = manager.profiles["dev"]

            # Should keep the placeholder when var not found
            assert dev_profile.env_vars["API_KEY"] == "${MISSING_VAR}"


class TestDeployManagerCoreProvider:
    """Test provider registry."""

    def test_get_cloudrun_provider(self, sample_config):
        """Test getting Cloud Run provider."""
        manager = DeployManagerCore(config=sample_config)

        provider = manager._get_provider(DeployProviderType.CLOUD_RUN)

        assert provider is not None
        from kestrel_sovereign.features.deploy.providers.cloudrun import CloudRunProvider
        assert isinstance(provider, CloudRunProvider)

    def test_get_provider_caching(self, sample_config):
        """Test provider instances are cached."""
        manager = DeployManagerCore(config=sample_config)

        provider1 = manager._get_provider(DeployProviderType.CLOUD_RUN)
        provider2 = manager._get_provider(DeployProviderType.CLOUD_RUN)

        assert provider1 is provider2

    def test_get_azure_provider_not_implemented(self, sample_config):
        """Test Azure provider raises not implemented."""
        from kestrel_sovereign.features.deploy.models import DeployManagerError

        manager = DeployManagerCore(config=sample_config)

        with pytest.raises(DeployManagerError) as exc_info:
            manager._get_provider(DeployProviderType.AZURE_CONTAINER_APPS)

        assert "not yet implemented" in str(exc_info.value).lower()


class TestDeployManagerCoreSessions:
    """Test session management."""

    @pytest.mark.asyncio
    async def test_add_and_get_session(self, sample_config):
        """Test adding and retrieving sessions."""
        manager = DeployManagerCore(config=sample_config)
        profile = manager.get_profile("dev")

        session = DeploymentSession(
            service_name="kestrel-dev",
            provider=DeployProviderType.CLOUD_RUN,
            profile=profile,
            status=DeployStatus.ACTIVE,
            started_at=datetime.now(),
            service_url="https://kestrel-dev-abc.run.app",
        )

        await manager.add_session(session)

        retrieved = await manager.get_session("kestrel-dev")
        assert retrieved is not None
        assert retrieved.service_name == "kestrel-dev"
        assert retrieved.service_url == "https://kestrel-dev-abc.run.app"

    @pytest.mark.asyncio
    async def test_remove_session(self, sample_config):
        """Test removing a session."""
        manager = DeployManagerCore(config=sample_config)
        profile = manager.get_profile("dev")

        session = DeploymentSession(
            service_name="kestrel-dev",
            provider=DeployProviderType.CLOUD_RUN,
            profile=profile,
            status=DeployStatus.ACTIVE,
            started_at=datetime.now(),
        )

        await manager.add_session(session)
        await manager.remove_session("kestrel-dev")

        retrieved = await manager.get_session("kestrel-dev")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, sample_config):
        """Test listing all sessions."""
        manager = DeployManagerCore(config=sample_config)
        dev_profile = manager.get_profile("dev")
        prod_profile = manager.get_profile("prod")

        dev_session = DeploymentSession(
            service_name="kestrel-dev",
            provider=DeployProviderType.CLOUD_RUN,
            profile=dev_profile,
            status=DeployStatus.ACTIVE,
            started_at=datetime.now(),
        )

        prod_session = DeploymentSession(
            service_name="kestrel-prod",
            provider=DeployProviderType.CLOUD_RUN,
            profile=prod_profile,
            status=DeployStatus.ACTIVE,
            started_at=datetime.now(),
        )

        await manager.add_session(dev_session)
        await manager.add_session(prod_session)

        sessions = await manager.list_sessions()
        assert len(sessions) == 2
        assert "kestrel-dev" in sessions
        assert "kestrel-prod" in sessions


class TestDeployFeature:
    """Test DeployFeature initialization and tool methods."""

    @pytest.mark.asyncio
    async def test_initialize_with_profiles(self, sample_config):
        """Test feature initialization with valid config."""
        with patch("kestrel_sovereign.features.deploy.feature.DeployManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.profiles = {"dev": MagicMock(), "prod": MagicMock()}
            mock_mgr.return_value = mock_instance

            feature = DeployFeature(agent=MagicMock())
            await feature.initialize()

            assert feature.disabled is False
            assert feature.manager is not None

    @pytest.mark.asyncio
    async def test_initialize_without_profiles(self):
        """Test feature gracefully degrades when no profiles configured."""
        with patch("kestrel_sovereign.features.deploy.feature.DeployManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.profiles = {}  # No profiles
            mock_mgr.return_value = mock_instance

            feature = DeployFeature(agent=MagicMock())
            await feature.initialize()

            assert feature.disabled is True
            assert "No deployment profiles" in feature.disabled_reason

    @pytest.mark.asyncio
    async def test_deploy_agent_when_disabled(self):
        """Test deploy_agent returns error when feature is disabled."""
        feature = DeployFeature(agent=MagicMock())
        feature.disabled = True
        feature.disabled_reason = "No profiles configured"

        result = await feature.deploy_agent(action="status")

        assert result["error"] == "Deploy feature is disabled"
        assert result["reason"] == "No profiles configured"

    @pytest.mark.asyncio
    async def test_status_no_sessions(self, sample_config):
        """Test status command with no active sessions."""
        with patch("kestrel_sovereign.features.deploy.feature.DeployManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.profiles = {"dev": MagicMock()}
            mock_instance.list_sessions = AsyncMock(return_value={})
            mock_mgr.return_value = mock_instance

            feature = DeployFeature(agent=MagicMock())
            await feature.initialize()

            result = await feature.deploy_agent(action="status")

            assert result["success"] is True
            assert result["active_deployments"] == 0
            assert "No active deployment sessions" in result["message"]

    @pytest.mark.asyncio
    async def test_deploy_without_profile(self, sample_config):
        """Test deploy command without profile returns error."""
        with patch("kestrel_sovereign.features.deploy.feature.DeployManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.profiles = {"dev": MagicMock()}
            mock_mgr.return_value = mock_instance

            feature = DeployFeature(agent=MagicMock())
            await feature.initialize()

            result = await feature.deploy_agent(action="deploy", profile="")

            assert result["success"] is False
            assert "Profile required" in result["error"]
            assert "available_profiles" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, sample_config):
        """Test unknown action returns error."""
        with patch("kestrel_sovereign.features.deploy.feature.DeployManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.profiles = {"dev": MagicMock()}
            mock_mgr.return_value = mock_instance

            feature = DeployFeature(agent=MagicMock())
            await feature.initialize()

            result = await feature.deploy_agent(action="invalid")

            assert result["success"] is False
            assert "Unknown action" in result["error"]
            assert "available_actions" in result

    def test_build_image_reference(self, sample_config):
        """Test image reference building."""
        with patch("kestrel_sovereign.features.deploy.feature.DeployManager") as mock_mgr:
            mock_instance = MagicMock()
            mock_instance.profiles = {"dev": MagicMock()}
            mock_instance.gcp_project_id = "test-project"
            mock_instance.image_name = "kestrel"
            mock_mgr.return_value = mock_instance

            feature = DeployFeature(agent=MagicMock())
            feature.manager = mock_instance

            # Test with default tag
            ref = feature._build_image_reference("dev", "latest")
            assert ref == "gcr.io/test-project/kestrel:latest"

            # Test with custom tag
            ref = feature._build_image_reference("dev", "v1.2.3")
            assert ref == "gcr.io/test-project/kestrel:v1.2.3"


class TestDeployManager:
    """Test DeployManager composition layer."""

    def test_manager_inherits_core(self, sample_config):
        """Test DeployManager inherits from DeployManagerCore."""
        manager = DeployManager(config=sample_config)

        assert isinstance(manager, DeployManagerCore)
        assert hasattr(manager, "profiles")
        assert hasattr(manager, "get_profile")
        assert len(manager.profiles) == 2
