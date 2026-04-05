"""
Unit tests for deploy feature bug fixes (#101).

Bug 1: azure_resource_group must be populated from profile config.
Bug 2: Health check must reject 4xx status codes (not treat them as healthy).
Bug 3: Temp credential file must be cleaned up via explicit cleanup() method.
"""

import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.features.deploy.core import DeployManagerCore
from kestrel_sovereign.features.deploy.models import (
    DeploymentProfile,
    DeployProviderType,
)
from kestrel_sovereign.features.deploy.providers.cloudrun import CloudRunProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config_with_azure():
    """Config that includes an Azure profile with azure_resource_group."""
    return {
        "manager": {
            "default_provider": "cloudrun",
            "gcp_project_id": "test-project",
            "image_name": "kestrel",
        },
        "profiles": {
            "azure-dev": {
                "provider": "azure",
                "service_name": "kestrel-azure-dev",
                "region": "eastus2",
                "azure_resource_group": "my-rg-from-config",
            },
            "gcp-dev": {
                "provider": "cloudrun",
                "service_name": "kestrel-gcp-dev",
                "region": "us-central1",
                "gcp_project_id": "profile-project",
            },
        },
    }


@pytest.fixture
def sample_config_cloudrun_only():
    """Config with only Cloud Run profiles for health check tests."""
    return {
        "manager": {
            "default_provider": "cloudrun",
            "gcp_project_id": "test-project",
            "health_check_timeout_seconds": 5,
            "health_check_path": "/health",
        },
        "profiles": {
            "dev": {
                "provider": "cloudrun",
                "service_name": "kestrel-dev",
                "region": "us-central1",
            },
        },
    }


# ---------------------------------------------------------------------------
# Bug 1: azure_resource_group populated from config
# ---------------------------------------------------------------------------


class TestBug1AzureResourceGroup:
    """Verify azure_resource_group is loaded from profile config data."""

    def test_azure_resource_group_loaded_from_config(self, sample_config_with_azure):
        """azure_resource_group should be set from the profile's config entry."""
        manager = DeployManagerCore(config=sample_config_with_azure)
        profile = manager.profiles["azure-dev"]

        assert profile.azure_resource_group == "my-rg-from-config"

    def test_azure_resource_group_none_when_absent(self, sample_config_with_azure):
        """azure_resource_group should be None when not specified in config."""
        manager = DeployManagerCore(config=sample_config_with_azure)
        profile = manager.profiles["gcp-dev"]

        assert profile.azure_resource_group is None

    def test_gcp_project_id_fallback_to_manager(self, sample_config_with_azure):
        """gcp_project_id should fall back to manager-level value when absent in profile."""
        # Remove the profile-level gcp_project_id
        del sample_config_with_azure["profiles"]["gcp-dev"]["gcp_project_id"]

        # Clear GCP_PROJECT_ID env var so it does not override the config value
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCP_PROJECT_ID", None)
            manager = DeployManagerCore(config=sample_config_with_azure)

        profile = manager.profiles["gcp-dev"]

        # Falls back to manager.gcp_project_id
        assert profile.gcp_project_id == "test-project"

    def test_gcp_project_id_profile_overrides_manager(self, sample_config_with_azure):
        """Profile-level gcp_project_id takes precedence over manager-level."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCP_PROJECT_ID", None)
            manager = DeployManagerCore(config=sample_config_with_azure)

        profile = manager.profiles["gcp-dev"]

        assert profile.gcp_project_id == "profile-project"


# ---------------------------------------------------------------------------
# Bug 2: Health check must reject 4xx as unhealthy
# ---------------------------------------------------------------------------


class TestBug2HealthCheckStatusCodes:
    """Verify that 4xx status codes are NOT treated as healthy."""

    @pytest.mark.asyncio
    async def test_health_check_rejects_404(self):
        """A 404 response must not be considered healthy."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check(
                "https://kestrel-dev-abc.run.app"
            )

            assert result["healthy"] is False
            assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_health_check_rejects_403(self):
        """A 403 response must not be considered healthy."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check(
                "https://kestrel-dev-abc.run.app"
            )

            assert result["healthy"] is False
            assert result["status_code"] == 403

    @pytest.mark.asyncio
    async def test_health_check_rejects_401(self):
        """A 401 response must not be considered healthy."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check(
                "https://kestrel-dev-abc.run.app"
            )

            assert result["healthy"] is False
            assert result["status_code"] == 401

    @pytest.mark.asyncio
    async def test_health_check_accepts_200(self):
        """A 200 response must be considered healthy."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check(
                "https://kestrel-dev-abc.run.app"
            )

            assert result["healthy"] is True
            assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_health_check_accepts_301_redirect(self):
        """A 301 redirect should be treated as healthy (service is reachable)."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 301
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check(
                "https://kestrel-dev-abc.run.app"
            )

            assert result["healthy"] is True
            assert result["status_code"] == 301

    @pytest.mark.asyncio
    async def test_health_check_rejects_500(self):
        """A 500 response must not be considered healthy."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            provider = CloudRunProvider(project_id="test-project")
            result = await provider.health_check(
                "https://kestrel-dev-abc.run.app"
            )

            assert result["healthy"] is False
            assert result["status_code"] == 500

    @pytest.mark.asyncio
    async def test_verify_health_rejects_404(self, sample_config_cloudrun_only):
        """_verify_health() in core must also reject 4xx as unhealthy."""
        manager = DeployManagerCore(config=sample_config_cloudrun_only)

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            # Short timeout so it fails fast
            result = await manager._verify_health(
                "https://kestrel-dev-abc.run.app",
                timeout=2,
                poll_interval=1,
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_verify_health_accepts_200(self, sample_config_cloudrun_only):
        """_verify_health() in core must accept 200 as healthy."""
        manager = DeployManagerCore(config=sample_config_cloudrun_only)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await manager._verify_health(
                "https://kestrel-dev-abc.run.app",
                timeout=5,
                poll_interval=1,
            )

            assert result is True


# ---------------------------------------------------------------------------
# Bug 3: Temp credential file cleanup
# ---------------------------------------------------------------------------


class TestBug3TempFileCleanup:
    """Verify temporary credential files are properly cleaned up."""

    def test_cleanup_removes_temp_file(self):
        """cleanup() must remove the temp credentials file."""
        # Create a real temp file to simulate what _setup_auth does
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"type": "service_account"}')
            temp_path = f.name

        assert os.path.exists(temp_path)

        provider = CloudRunProvider(project_id="test-project")
        provider._temp_cred_file = temp_path

        provider.cleanup()

        assert not os.path.exists(temp_path)
        assert provider._temp_cred_file is None

    def test_cleanup_safe_when_no_temp_file(self):
        """cleanup() must not raise when there is no temp file."""
        provider = CloudRunProvider(project_id="test-project")
        assert provider._temp_cred_file is None

        # Should not raise
        provider.cleanup()

    def test_cleanup_safe_when_file_already_deleted(self):
        """cleanup() must not raise when the temp file is already gone."""
        provider = CloudRunProvider(project_id="test-project")
        provider._temp_cred_file = "/tmp/nonexistent-cred-file-12345.json"

        # Should not raise
        provider.cleanup()

    def test_cleanup_idempotent(self):
        """Calling cleanup() multiple times must be safe."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"type": "service_account"}')
            temp_path = f.name

        provider = CloudRunProvider(project_id="test-project")
        provider._temp_cred_file = temp_path

        provider.cleanup()
        assert not os.path.exists(temp_path)

        # Second call should be a no-op
        provider.cleanup()

    def test_del_cleans_up_temp_file(self):
        """__del__ must clean up the temp file as a safety net."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"type": "service_account"}')
            temp_path = f.name

        provider = CloudRunProvider(project_id="test-project")
        provider._temp_cred_file = temp_path

        # Simulate garbage collection calling __del__
        provider.__del__()

        assert not os.path.exists(temp_path)

    def test_setup_auth_with_inline_key_registers_cleanup(self):
        """When GCP_SERVICE_ACCOUNT_KEY is set, _setup_auth must create a temp file
        and register atexit cleanup."""
        fake_key_json = '{"type": "service_account", "project_id": "test"}'

        with patch.dict(
            os.environ,
            {
                "GCP_SERVICE_ACCOUNT_KEY": fake_key_json,
                "GCP_PROJECT_ID": "test-project",
            },
            clear=False,
        ):
            # Remove GOOGLE_APPLICATION_CREDENTIALS to hit the inline key path
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

            with patch("atexit.register") as mock_atexit:
                provider = CloudRunProvider(project_id="test-project")

                # Temp file should exist
                assert provider._temp_cred_file is not None
                temp_path = provider._temp_cred_file
                assert os.path.exists(temp_path)

                # atexit should have been registered
                mock_atexit.assert_called_once_with(provider._cleanup_temp_creds)

                # Clean up
                provider.cleanup()
                assert not os.path.exists(temp_path)
                assert provider._temp_cred_file is None

    def test_cleanup_resets_temp_cred_file_to_none(self):
        """After cleanup, _temp_cred_file should be None to prevent double-delete."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"type": "service_account"}')
            temp_path = f.name

        provider = CloudRunProvider(project_id="test-project")
        provider._temp_cred_file = temp_path

        provider.cleanup()

        assert provider._temp_cred_file is None

    def test_base_provider_cleanup_is_noop(self):
        """DeployProvider.cleanup() should be a safe no-op by default."""
        from kestrel_sovereign.features.deploy.providers.base import DeployProvider

        # Can't instantiate abstract class directly, but we can check the method exists
        assert hasattr(DeployProvider, "cleanup")
