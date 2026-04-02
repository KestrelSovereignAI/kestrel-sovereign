"""
Unit tests for Cloud Run deployment provider.

Tests the Cloud Run provider with mocked google-cloud-run SDK.
"""

import os
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

pytest.importorskip("google.cloud.run_v2", reason="google-cloud-run not installed (cloud extras)")

from kestrel_sovereign.features.deploy.models import (
    DeploymentProfile,
    DeployProviderType,
    DeployManagerError,
)
from kestrel_sovereign.features.deploy.providers.cloudrun import CloudRunProvider


@pytest.fixture
def mock_services_client():
    """Mock google-cloud-run ServicesClient."""
    with patch("google.cloud.run_v2.ServicesClient") as mock:
        yield mock


@pytest.fixture
def mock_logging_client():
    """Mock google-cloud-logging Client."""
    with patch("google.cloud.logging.Client") as mock:
        yield mock


@pytest.fixture
def deployment_profile():
    """Create a test deployment profile."""
    return DeploymentProfile(
        provider=DeployProviderType.CLOUD_RUN,
        service_name="kestrel-dev",
        region="us-central1",
        min_instances=0,
        max_instances=10,
        memory="2Gi",
        cpu=2,
        port=8080,
        timeout=300,
        concurrency=80,
        env_vars={
            "KESTREL_ENV": "development",
            "KESTREL_DB_BACKEND": "sqlite",
        },
        secrets={
            "OPENAI_API_KEY": "kestrel-openai-key:latest",
            "KESTREL_API_KEY": "kestrel-api-key:latest",
        },
    )


class TestCloudRunProviderInit:
    """Test CloudRunProvider initialization."""

    def test_init_with_project_id(self):
        """Test initialization with explicit project ID."""
        provider = CloudRunProvider(project_id="my-project-123")
        assert provider.project_id == "my-project-123"

    def test_init_with_env_var(self):
        """Test initialization with GCP_PROJECT_ID env var."""
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "env-project-456"}):
            provider = CloudRunProvider()
            assert provider.project_id == "env-project-456"

    def test_init_without_project_id(self):
        """Test initialization fails without project ID."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(DeployManagerError) as exc_info:
                CloudRunProvider()
            assert "GCP_PROJECT_ID is required" in str(exc_info.value)


class TestCloudRunProviderDeploy:
    """Test CloudRunProvider.deploy() method."""

    @pytest.mark.asyncio
    async def test_deploy_new_service(self, mock_services_client, deployment_profile):
        """Test deploying a new service."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client

        # Mock get_service to raise exception (service doesn't exist)
        mock_client.get_service.side_effect = Exception("404 not found")

        # Mock create_service operation
        mock_operation = MagicMock()
        mock_result = MagicMock()
        mock_result.uri = "https://kestrel-dev-abc123.run.app"
        mock_result.latest_ready_revision = "kestrel-dev-00001-abc"
        mock_operation.result.return_value = mock_result
        mock_client.create_service.return_value = mock_operation

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        result = await provider.deploy(
            image="gcr.io/test-project/kestrel:latest",
            service_name="kestrel-dev",
            profile=deployment_profile,
        )

        assert result["service_url"] == "https://kestrel-dev-abc123.run.app"
        assert result["revision"] == "kestrel-dev-00001-abc"
        assert result["status"] == "active"

        # Verify create_service was called
        assert mock_client.create_service.called

    @pytest.mark.asyncio
    async def test_deploy_update_existing(self, mock_services_client, deployment_profile):
        """Test updating an existing service."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client

        # Mock get_service to return existing service
        mock_existing = MagicMock()
        mock_client.get_service.return_value = mock_existing

        # Mock update_service operation
        mock_operation = MagicMock()
        mock_result = MagicMock()
        mock_result.uri = "https://kestrel-dev-xyz789.run.app"
        mock_result.latest_ready_revision = "kestrel-dev-00002-xyz"
        mock_operation.result.return_value = mock_result
        mock_client.update_service.return_value = mock_operation

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        result = await provider.deploy(
            image="gcr.io/test-project/kestrel:latest",
            service_name="kestrel-dev",
            profile=deployment_profile,
        )

        assert result["service_url"] == "https://kestrel-dev-xyz789.run.app"
        assert result["revision"] == "kestrel-dev-00002-xyz"
        assert result["status"] == "active"

        # Verify update_service was called instead of create
        assert mock_client.update_service.called
        assert not mock_client.create_service.called

    @pytest.mark.asyncio
    async def test_deploy_with_custom_env_vars(self, mock_services_client, deployment_profile):
        """Test deployment with additional environment variables."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client
        mock_client.get_service.side_effect = Exception("404 not found")

        mock_operation = MagicMock()
        mock_result = MagicMock()
        mock_result.uri = "https://kestrel-dev-abc123.run.app"
        mock_result.latest_ready_revision = "kestrel-dev-00001-abc"
        mock_operation.result.return_value = mock_result
        mock_client.create_service.return_value = mock_operation

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        # Deploy with additional env vars
        result = await provider.deploy(
            image="gcr.io/test-project/kestrel:latest",
            service_name="kestrel-dev",
            profile=deployment_profile,
            env_vars={"CUSTOM_VAR": "custom_value"},
        )

        assert result["status"] == "active"


class TestCloudRunProviderStatus:
    """Test CloudRunProvider.get_status() method."""

    @pytest.mark.asyncio
    async def test_get_status_active(self, mock_services_client):
        """Test getting status of active service."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client

        # Mock service with ready condition
        mock_service = MagicMock()
        mock_service.name = "projects/test-project/locations/us-central1/services/kestrel-dev"
        mock_service.uri = "https://kestrel-dev-abc123.run.app"
        mock_service.latest_ready_revision = "kestrel-dev-00001-abc"

        mock_condition = MagicMock()
        mock_condition.type = "Ready"
        mock_condition.state = "CONDITION_SUCCEEDED"
        mock_service.conditions = [mock_condition]

        mock_client.list_services.return_value = [mock_service]

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        result = await provider.get_status("kestrel-dev")

        assert result["status"] == "active"
        assert result["service_url"] == "https://kestrel-dev-abc123.run.app"
        assert result["revision"] == "kestrel-dev-00001-abc"
        assert result["health"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_status_deploying(self, mock_services_client):
        """Test getting status of deploying service."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client

        # Mock service with non-ready condition
        mock_service = MagicMock()
        mock_service.name = "projects/test-project/locations/us-central1/services/kestrel-dev"
        mock_service.uri = "https://kestrel-dev-abc123.run.app"
        mock_service.latest_ready_revision = None

        mock_condition = MagicMock()
        mock_condition.type = "Ready"
        mock_condition.state = "PENDING"
        mock_service.conditions = [mock_condition]

        mock_client.list_services.return_value = [mock_service]

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        result = await provider.get_status("kestrel-dev")

        assert result["status"] == "deploying"
        assert result["health"] == "unknown"

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, mock_services_client):
        """Test getting status of non-existent service."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client
        mock_client.list_services.return_value = []

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        result = await provider.get_status("kestrel-nonexistent")

        assert result["status"] == "offline"
        assert result["service_url"] is None
        assert result["revision"] is None


class TestCloudRunProviderTeardown:
    """Test CloudRunProvider.teardown() method."""

    @pytest.mark.asyncio
    async def test_teardown_success(self, mock_services_client):
        """Test successful service teardown."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client

        # Mock list_services to find the service
        mock_service = MagicMock()
        mock_service.name = "projects/test-project/locations/us-central1/services/kestrel-dev"
        mock_client.list_services.return_value = [mock_service]

        # Mock delete operation
        mock_operation = MagicMock()
        mock_operation.result.return_value = None
        mock_client.delete_service.return_value = mock_operation

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        result = await provider.teardown("kestrel-dev")

        assert result["status"] == "deleted"
        assert "successfully" in result["message"]
        assert mock_client.delete_service.called

    @pytest.mark.asyncio
    async def test_teardown_not_found(self, mock_services_client):
        """Test teardown of non-existent service."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client
        mock_client.list_services.return_value = []

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        result = await provider.teardown("kestrel-nonexistent")

        assert result["status"] == "not_found"
        assert "not found" in result["message"]
        assert not mock_client.delete_service.called


class TestCloudRunProviderList:
    """Test CloudRunProvider.list_deployments() method."""

    @pytest.mark.asyncio
    async def test_list_deployments(self, mock_services_client):
        """Test listing Kestrel deployments."""
        # Setup mocks
        mock_client = MagicMock()
        mock_services_client.return_value = mock_client

        # Mock services
        mock_service1 = MagicMock()
        mock_service1.name = "projects/test-project/locations/us-central1/services/kestrel-dev"
        mock_service1.uri = "https://kestrel-dev-abc123.run.app"
        mock_service1.create_time = datetime(2026, 2, 15, 12, 0, 0)

        mock_condition1 = MagicMock()
        mock_condition1.type = "Ready"
        mock_condition1.state = "CONDITION_SUCCEEDED"
        mock_service1.conditions = [mock_condition1]

        mock_service2 = MagicMock()
        mock_service2.name = "projects/test-project/locations/us-west1/services/kestrel-prod"
        mock_service2.uri = "https://kestrel-prod-xyz789.run.app"
        mock_service2.create_time = datetime(2026, 2, 14, 12, 0, 0)

        mock_condition2 = MagicMock()
        mock_condition2.type = "Ready"
        mock_condition2.state = "CONDITION_SUCCEEDED"
        mock_service2.conditions = [mock_condition2]

        # Non-kestrel service should be filtered out
        mock_service3 = MagicMock()
        mock_service3.name = "projects/test-project/locations/us-central1/services/other-app"

        mock_client.list_services.return_value = [mock_service1, mock_service2, mock_service3]

        provider = CloudRunProvider(project_id="test-project")
        provider._services_client = mock_client

        result = await provider.list_deployments()

        assert len(result) == 2
        assert result[0]["name"] == "kestrel-dev"
        assert result[0]["status"] == "active"
        assert result[0]["url"] == "https://kestrel-dev-abc123.run.app"
        assert result[1]["name"] == "kestrel-prod"


class TestCloudRunProviderHealthCheck:
    """Test CloudRunProvider.health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check of healthy service."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check("https://kestrel-dev-abc123.run.app")

            assert result["healthy"] is True
            assert result["status_code"] == 200
            assert "response_time" in result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check of unhealthy service."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check("https://kestrel-dev-abc123.run.app")

            assert result["healthy"] is False
            assert result["status_code"] == 503

    @pytest.mark.asyncio
    async def test_health_check_error(self):
        """Test health check when service is unreachable."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection refused")

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check("https://kestrel-dev-abc123.run.app")

            assert result["healthy"] is False
            assert result["status_code"] is None
            assert "error" in result
