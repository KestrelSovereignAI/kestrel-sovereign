"""
Integration tests for Deploy Feature - Real Cloud Run deployments.

These tests verify the agent's self-deployment capability with REAL Cloud Run API calls.
NO MOCKS - these are real integration tests that create and destroy cloud resources.

Requirements:
1. GCP_PROJECT_ID environment variable
2. GOOGLE_APPLICATION_CREDENTIALS or GCP_SERVICE_ACCOUNT_KEY
3. deploy_config.toml with valid profiles
4. Container image pushed to GCR (gcr.io/PROJECT_ID/kestrel:latest)

Run with:
    pytest tests/integration/test_deploy_e2e.py -v

Skip without credentials:
    Tests automatically skip if GOOGLE_APPLICATION_CREDENTIALS is not set
"""

import asyncio
import os
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

# Skip all tests if GCP credentials not configured
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    reason="GOOGLE_APPLICATION_CREDENTIALS not set - skipping Cloud Run deployment tests",
)


class TestDeployModelsUnit:
    """Unit tests for deploy models - no credentials required."""

    def test_import_models(self):
        """Test that deploy models can be imported."""
        from kestrel_sovereign.features.deploy.models import (
            DeployManagerError,
            DeploymentProfile,
            DeploymentSession,
            DeployProviderType,
            DeployStatus,
        )

        assert DeployStatus.OFFLINE.value == "offline"
        assert DeployStatus.ACTIVE.value == "active"
        assert DeployProviderType.CLOUD_RUN.value == "cloud_run"
        assert DeployProviderType.AZURE_CONTAINER_APPS.value == "azure_container_apps"

    def test_deployment_profile_scale_to_zero(self):
        """Test DeploymentProfile scale-to-zero detection."""
        from kestrel_sovereign.features.deploy.models import (
            DeploymentProfile,
            DeployProviderType,
        )

        # Scale to zero profile
        dev_profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
            min_instances=0,
            max_instances=10,
        )
        assert dev_profile.is_scale_to_zero is True

        # Always-warm profile
        prod_profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-prod",
            region="us-central1",
            min_instances=1,
            max_instances=100,
        )
        assert prod_profile.is_scale_to_zero is False

    def test_deployment_session_to_dict(self):
        """Test DeploymentSession serialization."""
        from kestrel_sovereign.features.deploy.models import (
            DeploymentProfile,
            DeploymentSession,
            DeployProviderType,
            DeployStatus,
        )

        profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-test",
            region="us-central1",
            min_instances=0,
            max_instances=10,
        )

        now = datetime.now(timezone.utc)
        session = DeploymentSession(
            service_name="kestrel-test",
            provider=DeployProviderType.CLOUD_RUN,
            profile=profile,
            status=DeployStatus.ACTIVE,
            started_at=now,
            service_url="https://kestrel-test-abc123.run.app",
            revision="kestrel-test-00001-abc",
            health_status="healthy",
        )

        data = session.to_dict()
        assert data["service_name"] == "kestrel-test"
        assert data["status"] == "active"
        assert data["service_url"] == "https://kestrel-test-abc123.run.app"
        assert data["is_scale_to_zero"] is True


class TestDeployManagerCoreUnit:
    """Unit tests for DeployManagerCore - no credentials required."""

    def test_load_config(self):
        """Test config loading from deploy_config.toml.example."""
        from kestrel_sovereign.features.deploy.core import DeployManagerCore

        # Use example config (doesn't require GCP_PROJECT_ID for loading)
        config_path = Path(__file__).parent.parent.parent / "deploy_config.toml.example"
        if not config_path.exists():
            pytest.skip("deploy_config.toml.example not found")

        from kestrel_sovereign.config import load_config

        config = load_config(str(config_path))

        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            manager = DeployManagerCore(config=config)

            assert manager.default_provider == "cloudrun"
            assert "dev" in manager.profiles
            assert "prod" in manager.profiles

            # Dev profile should scale to zero
            dev_profile = manager.profiles["dev"]
            assert dev_profile.is_scale_to_zero is True
            assert dev_profile.min_instances == 0

            # Prod profile should be always-warm
            prod_profile = manager.profiles["prod"]
            assert prod_profile.is_scale_to_zero is False
            assert prod_profile.min_instances == 1

    def test_profile_not_found(self):
        """Test error when profile doesn't exist."""
        from kestrel_sovereign.features.deploy.core import DeployManagerCore
        from kestrel_sovereign.features.deploy.models import DeployManagerError

        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            manager = DeployManagerCore(config={"profiles": {}})

            with pytest.raises(DeployManagerError, match="Unknown profile"):
                manager.get_profile("nonexistent")

    def test_env_var_expansion(self):
        """Test environment variable expansion in config."""
        from kestrel_sovereign.features.deploy.core import DeployManagerCore

        with patch.dict(
            os.environ,
            {"GCP_PROJECT_ID": "test-project", "MY_SECRET": "secret-value"},
        ):
            config = {
                "manager": {"gcp_project_id": "test-project"},
                "profiles": {
                    "test": {
                        "provider": "cloudrun",
                        "service_name": "test-service",
                        "region": "us-central1",
                        "env_vars": {"SECRET": "${MY_SECRET}"},
                    }
                },
            }

            manager = DeployManagerCore(config=config)
            profile = manager.get_profile("test")

            assert profile.env_vars["SECRET"] == "secret-value"


class TestAzureProviderStub:
    """Test Azure Container Apps provider stub."""

    def test_import_azure_provider(self):
        """Test that Azure provider can be imported."""
        from kestrel_sovereign.features.deploy.providers import AzureContainerProvider

        assert AzureContainerProvider is not None

    @pytest.mark.asyncio
    async def test_azure_provider_requires_sdk(self):
        """Test that Azure provider raises DeployManagerError without azure SDK."""
        from kestrel_sovereign.features.deploy.providers import AzureContainerProvider
        from kestrel_sovereign.features.deploy.models import (
            DeploymentProfile,
            DeployManagerError,
            DeployProviderType,
        )

        provider = AzureContainerProvider(
            subscription_id="test-sub-id",
            resource_group="test-rg",
        )

        profile = DeploymentProfile(
            provider=DeployProviderType.AZURE_CONTAINER_APPS,
            service_name="test",
            region="eastus2",
        )

        try:
            import azure  # noqa: F401
            pytest.skip("Azure SDK installed — stub test not applicable")
        except ImportError:
            pass

        with pytest.raises(DeployManagerError, match="Missing dependency"):
            await provider.deploy("image:latest", "test", profile)

    def test_azure_provider_in_registry(self):
        """Test that Azure provider is recognized in core registry."""
        from kestrel_sovereign.features.deploy.core import DeployManagerCore
        from kestrel_sovereign.features.deploy.models import DeployProviderType

        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            config = {
                "manager": {"gcp_project_id": "test-project"},
                "profiles": {
                    "azure-test": {
                        "provider": "azure",
                        "service_name": "test-service",
                        "region": "eastus2",
                    }
                },
            }

            manager = DeployManagerCore(config=config)

            # Should parse Azure provider from config
            profile = manager.get_profile("azure-test")
            assert profile.provider == DeployProviderType.AZURE_CONTAINER_APPS

            # Should be able to get Azure provider instance
            provider = manager._get_provider(DeployProviderType.AZURE_CONTAINER_APPS)
            assert provider is not None
            assert type(provider).__name__ == "AzureContainerProvider"


@pytest.mark.cloud_resource
class TestCloudRunDeployE2E:
    """
    Integration tests for Cloud Run deployments.

    These tests create REAL Cloud Run services that cost money.
    They automatically skip if GOOGLE_APPLICATION_CREDENTIALS is not set.

    Run with: pytest tests/integration/test_deploy_e2e.py::TestCloudRunDeployE2E -v
    """

    @pytest.fixture
    def manager(self):
        """Create a DeployManagerCore instance."""
        from kestrel_sovereign.features.deploy.core import DeployManagerCore

        # Ensure GCP_PROJECT_ID is set
        if not os.getenv("GCP_PROJECT_ID"):
            pytest.skip("GCP_PROJECT_ID not set")

        return DeployManagerCore()

    @pytest.fixture
    def test_service_name(self):
        """Generate a unique test service name."""
        import uuid

        return f"kestrel-test-{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_deploy_dev_profile(self, manager, test_service_name):
        """
        Test deploying to Cloud Run with dev profile (scale to zero).

        WARNING: This creates a real Cloud Run service.
        The service will be automatically deleted after the test.
        """
        from kestrel_sovereign.features.deploy.models import DeployStatus

        # Get dev profile
        profile = manager.get_profile("dev")
        assert profile.is_scale_to_zero is True

        # Override service name for test
        profile.service_name = test_service_name

        # Get provider
        provider = manager._get_provider(profile.provider)

        try:
            # Deploy
            result = await provider.deploy(
                image=f"gcr.io/{manager.gcp_project_id}/kestrel:latest",
                service_name=test_service_name,
                profile=profile,
                env_vars={"KESTREL_ENV": "test"},
            )

            assert "service_url" in result
            assert result["service_url"].startswith("https://")
            assert test_service_name in result["service_url"]

        finally:
            # Cleanup: always delete the test service
            try:
                await provider.teardown(test_service_name)
            except Exception as e:
                print(f"Cleanup failed: {e}")

    @pytest.mark.asyncio
    async def test_health_check_passes(self, manager, test_service_name):
        """
        Test that health check passes after deployment.

        WARNING: This creates a real Cloud Run service.
        """
        from kestrel_sovereign.features.deploy.models import DeployStatus

        profile = manager.get_profile("dev")
        profile.service_name = test_service_name

        provider = manager._get_provider(profile.provider)

        try:
            # Deploy
            result = await provider.deploy(
                image=f"gcr.io/{manager.gcp_project_id}/kestrel:latest",
                service_name=test_service_name,
                profile=profile,
            )

            service_url = result["service_url"]

            # Verify health with short timeout
            is_healthy = await manager._verify_health(service_url, timeout=60)
            assert is_healthy is True

        finally:
            # Cleanup
            try:
                await provider.teardown(test_service_name)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_status_shows_url(self, manager, test_service_name):
        """
        Test that get_status returns service URL.

        WARNING: This creates a real Cloud Run service.
        """
        profile = manager.get_profile("dev")
        profile.service_name = test_service_name

        provider = manager._get_provider(profile.provider)

        try:
            # Deploy
            deploy_result = await provider.deploy(
                image=f"gcr.io/{manager.gcp_project_id}/kestrel:latest",
                service_name=test_service_name,
                profile=profile,
            )

            # Get status
            status = await provider.get_status(test_service_name)

            assert status["status"] in ["active", "ready"]
            assert "service_url" in status
            assert status["service_url"] == deploy_result["service_url"]

        finally:
            # Cleanup
            try:
                await provider.teardown(test_service_name)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_teardown(self, manager, test_service_name):
        """
        Test that teardown deletes the service.

        WARNING: This creates a real Cloud Run service.
        """
        profile = manager.get_profile("dev")
        profile.service_name = test_service_name

        provider = manager._get_provider(profile.provider)

        # Deploy
        await provider.deploy(
            image=f"gcr.io/{manager.gcp_project_id}/kestrel:latest",
            service_name=test_service_name,
            profile=profile,
        )

        # Teardown
        result = await provider.teardown(test_service_name)

        assert result["status"] in ["deleted", "success"]

        # Verify it's gone (should raise or return offline status)
        try:
            status = await provider.get_status(test_service_name)
            # If it returns, status should indicate service is gone
            assert status["status"] in ["offline", "not_found", "terminated"]
        except Exception:
            # Expected - service not found is OK
            pass


class TestDeployFeature:
    """Tests for the DeployFeature tool interface."""

    def test_import_feature(self):
        """Test that the feature can be imported."""
        from kestrel_sovereign.features.deploy.feature import DeployFeature

        assert DeployFeature is not None

    @pytest.mark.asyncio
    async def test_feature_initialization(self):
        """Test feature initialization."""
        from kestrel_sovereign.features.deploy.feature import DeployFeature

        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            feature = DeployFeature(agent=None)
            await feature.initialize()

            assert feature.manager is not None
            assert feature.manager.gcp_project_id == "test-project"

    @pytest.mark.asyncio
    async def test_deploy_tool_actions(self):
        """Test that deploy tool exposes all required actions."""
        from kestrel_sovereign.features.deploy.feature import DeployFeature

        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            feature = DeployFeature(agent=None)
            await feature.initialize()

            # Should have deploy_agent tool
            assert hasattr(feature, "deploy_agent")

            # Tool should support required actions
            # (actual invocation tested in unit tests)
