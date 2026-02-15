"""
Unit tests for deploy models.

Tests the data models, enums, and validation logic for agent self-deployment.
"""

import pytest
from datetime import datetime, timezone

from kestrel_sovereign.features.deploy.models import (
    DeployStatus,
    DeployProviderType,
    DeploymentProfile,
    DeploymentSession,
    DeployManagerError,
)


class TestDeployStatus:
    """Test DeployStatus enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert DeployStatus.OFFLINE.value == "offline"
        assert DeployStatus.BUILDING.value == "building"
        assert DeployStatus.DEPLOYING.value == "deploying"
        assert DeployStatus.ACTIVE.value == "active"
        assert DeployStatus.FAILED.value == "failed"
        assert DeployStatus.TEARING_DOWN.value == "tearing_down"
        assert DeployStatus.TERMINATED.value == "terminated"


class TestDeployProviderType:
    """Test DeployProviderType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert DeployProviderType.CLOUD_RUN.value == "cloud_run"
        assert DeployProviderType.AZURE_CONTAINER_APPS.value == "azure_container_apps"


class TestDeploymentProfile:
    """Test DeploymentProfile dataclass."""

    def test_default_profile(self):
        """Test profile creation with minimal required fields."""
        profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
        )

        assert profile.provider == DeployProviderType.CLOUD_RUN
        assert profile.service_name == "kestrel-dev"
        assert profile.region == "us-central1"
        assert profile.min_instances == 0  # Scale to zero by default
        assert profile.max_instances == 10
        assert profile.memory == "2Gi"
        assert profile.cpu == 2
        assert profile.port == 8080
        assert profile.timeout == 300
        assert profile.concurrency == 80
        assert profile.env_vars == {}
        assert profile.secrets == {}

    def test_scale_to_zero_config(self):
        """Test scale-to-zero configuration."""
        profile_dev = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
            min_instances=0,
        )
        assert profile_dev.is_scale_to_zero is True

        profile_prod = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-prod",
            region="us-central1",
            min_instances=1,
        )
        assert profile_prod.is_scale_to_zero is False

    def test_custom_resources(self):
        """Test custom resource configuration."""
        profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-high-mem",
            region="us-central1",
            memory="4Gi",
            cpu=4,
            max_instances=100,
        )

        assert profile.memory == "4Gi"
        assert profile.cpu == 4
        assert profile.max_instances == 100

    def test_env_vars_and_secrets(self):
        """Test environment variables and secrets."""
        profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
            env_vars={
                "KESTREL_ENV": "development",
                "KESTREL_DB_BACKEND": "sqlite",
            },
            secrets={
                "OPENAI_API_KEY": "kestrel-openai-key:latest",
                "KESTREL_API_KEY": "kestrel-api-key:latest",
            },
        )

        assert len(profile.env_vars) == 2
        assert profile.env_vars["KESTREL_ENV"] == "development"
        assert len(profile.secrets) == 2
        assert profile.secrets["OPENAI_API_KEY"] == "kestrel-openai-key:latest"

    def test_provider_specific_fields(self):
        """Test provider-specific fields."""
        profile_gcp = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
            gcp_project_id="my-project-123",
        )
        assert profile_gcp.gcp_project_id == "my-project-123"
        assert profile_gcp.azure_resource_group is None

        profile_azure = DeploymentProfile(
            provider=DeployProviderType.AZURE_CONTAINER_APPS,
            service_name="kestrel-dev",
            region="eastus",
            azure_resource_group="my-resource-group",
        )
        assert profile_azure.azure_resource_group == "my-resource-group"
        assert profile_azure.gcp_project_id is None


class TestDeploymentSession:
    """Test DeploymentSession dataclass."""

    def test_session_creation(self):
        """Test session creation."""
        profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
        )

        started_at = datetime.now(timezone.utc)
        session = DeploymentSession(
            service_name="kestrel-dev",
            provider=DeployProviderType.CLOUD_RUN,
            profile=profile,
            status=DeployStatus.DEPLOYING,
            started_at=started_at,
        )

        assert session.service_name == "kestrel-dev"
        assert session.provider == DeployProviderType.CLOUD_RUN
        assert session.status == DeployStatus.DEPLOYING
        assert session.started_at == started_at
        assert session.service_url is None
        assert session.revision is None
        assert session.health_status is None

    def test_session_to_dict(self):
        """Test session serialization to dict."""
        profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
            min_instances=0,
            max_instances=10,
        )

        started_at = datetime.now(timezone.utc)
        session = DeploymentSession(
            service_name="kestrel-dev",
            provider=DeployProviderType.CLOUD_RUN,
            profile=profile,
            status=DeployStatus.ACTIVE,
            started_at=started_at,
            service_url="https://kestrel-dev-abc123.run.app",
            revision="kestrel-dev-00001-abc",
            health_status="healthy",
        )

        result = session.to_dict()

        assert result["service_name"] == "kestrel-dev"
        assert result["provider"] == "cloud_run"
        assert result["status"] == "active"
        assert result["service_url"] == "https://kestrel-dev-abc123.run.app"
        assert result["revision"] == "kestrel-dev-00001-abc"
        assert result["health_status"] == "healthy"
        assert result["is_scale_to_zero"] is True
        assert result["min_instances"] == 0
        assert result["max_instances"] == 10
        assert "started_at" in result

    def test_status_transitions(self):
        """Test status transitions through deployment lifecycle."""
        profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
        )

        session = DeploymentSession(
            service_name="kestrel-dev",
            provider=DeployProviderType.CLOUD_RUN,
            profile=profile,
            status=DeployStatus.OFFLINE,
            started_at=datetime.now(timezone.utc),
        )

        # Test state transitions
        assert session.status == DeployStatus.OFFLINE

        session.status = DeployStatus.BUILDING
        assert session.status == DeployStatus.BUILDING

        session.status = DeployStatus.DEPLOYING
        assert session.status == DeployStatus.DEPLOYING

        session.status = DeployStatus.ACTIVE
        assert session.status == DeployStatus.ACTIVE

        session.status = DeployStatus.TEARING_DOWN
        assert session.status == DeployStatus.TEARING_DOWN

        session.status = DeployStatus.TERMINATED
        assert session.status == DeployStatus.TERMINATED

    def test_failed_deployment(self):
        """Test failed deployment with error message."""
        profile = DeploymentProfile(
            provider=DeployProviderType.CLOUD_RUN,
            service_name="kestrel-dev",
            region="us-central1",
        )

        session = DeploymentSession(
            service_name="kestrel-dev",
            provider=DeployProviderType.CLOUD_RUN,
            profile=profile,
            status=DeployStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            error_message="Deployment quota exceeded",
        )

        assert session.status == DeployStatus.FAILED
        assert session.error_message == "Deployment quota exceeded"

        result = session.to_dict()
        assert result["status"] == "failed"
        assert result["error_message"] == "Deployment quota exceeded"


class TestDeployManagerError:
    """Test DeployManagerError exception."""

    def test_exception_creation(self):
        """Test exception can be raised and caught."""
        with pytest.raises(DeployManagerError) as exc_info:
            raise DeployManagerError("Deployment failed: insufficient quota")

        assert "Deployment failed" in str(exc_info.value)
