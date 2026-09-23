"""
Unit tests for deploy feature bug fixes (#101).

Bug 1: azure_resource_group must be populated from profile config.
Bug 2: Health check must reject 4xx status codes (not treat them as healthy).
Bug 3: Temp credential file must be cleaned up via explicit cleanup() method.
#2473: A deploy whose revision fails the readiness gate must not report success.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import httpx

import pytest

from kestrel_sovereign.features.deploy.core import DeployManagerCore
from kestrel_sovereign.features.deploy.models import (
    DeployManagerError,
    ReadinessCheck,
    ReadinessStatus,
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


class TestBug2ReadinessStatusCodes:
    """Verify readiness polling delegates the shared status contract."""

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

            assert result.status is ReadinessStatus.UNREADY
            assert result.failure == "http_status"
            assert result.last_status_code == 404

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

            assert result.status is ReadinessStatus.READY
            assert result.ready is True


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
        and register atexit cleanup.

        Auth setup was extracted to ``deploy/_gcp_auth.py`` (PR #1057
        post-codex), so the atexit registration target is now the
        module-level ``_cleanup_temp_creds`` keyed on the temp path.
        We assert the contract — temp file exists + an atexit handler
        was registered that can clean it up — without binding the test
        to the exact callable identity.
        """
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

            with patch(
                "kestrel_sovereign.features.deploy._gcp_auth.atexit.register"
            ) as mock_atexit:
                provider = CloudRunProvider(project_id="test-project")

                # Temp file should exist
                assert provider._temp_cred_file is not None
                temp_path = provider._temp_cred_file
                assert os.path.exists(temp_path)

                # atexit should have been registered with a callable that
                # cleans up the same temp file we just created.
                mock_atexit.assert_called_once()
                registered_callable, *registered_args = mock_atexit.call_args.args
                assert callable(registered_callable)
                assert registered_args == [temp_path]

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


# ---------------------------------------------------------------------------
# #2473: readiness gates deploy success
# ---------------------------------------------------------------------------

_SERVICE_URL = "https://kestrel-dev-abc.run.app"
_SECRET_IN_ERROR = "token=sk-live-SECRET-should-not-leak"


class _FakeProvider:
    """Stands in for Cloud Run: the control plane always creates a revision."""

    def __init__(self, operation: str, *, service_url: str | None = _SERVICE_URL):
        self.operation = operation
        self.service_url = service_url
        self.deploy_calls = 0
        self.teardown_calls = 0

    async def deploy(self, *, image, service_name, profile):
        self.deploy_calls += 1
        return {
            "service_url": self.service_url,
            "revision": f"{service_name}-00007-abc",
            "status": "active",
            "operation": self.operation,
            "warnings": [],
        }

    async def teardown(self, service_name):
        self.teardown_calls += 1
        return {"status": "deleted"}


def _readiness_config() -> dict:
    return {
        "manager": {
            "gcp_project_id": "test-project",
            "health_check_timeout_seconds": 5,
            "health_check_path": "/health",
        },
        "profiles": {
            "dev": {
                "provider": "cloudrun",
                "service_name": "kestrel-dev",
                "region": "us-central1",
                "max_instances": 1,
                "persistence_mode": "ephemeral_demo",
                "env_vars": {
                    "KESTREL_ENV": "development",
                    "KESTREL_DB_BACKEND": "sqlite",
                    "KESTREL_DEPLOYMENT_PERSISTENCE": "ephemeral_demo",
                },
            },
        },
    }


def _manager_with(provider: _FakeProvider) -> DeployManagerCore:
    manager = DeployManagerCore(config=_readiness_config())
    manager._get_provider = lambda *a, **kw: provider
    # A failing gate sleeps at most the remaining deadline; keep it short.
    manager.health_check_timeout = 0.2
    return manager


def _serve(handler):
    """Route every probe through real httpx using ``handler``."""
    real_client = httpx.AsyncClient

    def client_factory(*, timeout):
        return real_client(transport=httpx.MockTransport(handler), timeout=timeout)

    return patch("httpx.AsyncClient", side_effect=client_factory)


def _status(code: int, body: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL(f"{_SERVICE_URL}/health")
        if body is None:
            return httpx.Response(code, text="")
        return httpx.Response(code, json=body)

    return handler


def _raises(exc_type):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_type(f"{_SECRET_IN_ERROR} via {request.url}", request=request)

    return handler


_UNREADY_CASES = {
    "http_503": (_status(503), "http_status", 503),
    "bad_auth_403": (_status(403), "auth_rejected", 403),
    "bad_auth_401": (_status(401), "auth_rejected", 401),
    "zero_agent_503": (
        _status(503, {"status": "degraded", "agent_initialized": False}),
        "agent_not_initialized",
        503,
    ),
    "zero_agent_200": (
        _status(200, {"status": "ok", "agent_initialized": False}),
        "agent_not_initialized",
        200,
    ),
    "timeout": (_raises(httpx.ConnectTimeout), "probe_timeout", None),
    "unreachable": (_raises(httpx.ConnectError), "unreachable", None),
}


class TestIssue2473ReadinessGatesDeploySuccess:
    """A created revision is not a successful deploy until it is ready."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["create", "update"])
    @pytest.mark.parametrize("case", sorted(_UNREADY_CASES))
    async def test_unready_revision_is_not_success(self, case, operation):
        handler, failure, status_code = _UNREADY_CASES[case]
        provider = _FakeProvider(operation)
        manager = _manager_with(provider)

        with _serve(handler):
            result = await manager.deploy_profile("dev", tag="v1.2.3")

        assert result["success"] is False
        assert result["control_plane_status"] == "succeeded"
        assert result["readiness_status"] == "unready"
        assert result["operation"] == operation
        # The operator can find what was deployed and which gate failed.
        assert result["service"] == "kestrel-dev"
        assert result["revision"] == "kestrel-dev-00007-abc"
        assert result["service_url"] == _SERVICE_URL
        readiness = result["readiness"]
        assert readiness["status"] == "unready"
        assert readiness["failure"] == failure
        assert readiness["last_status_code"] == status_code
        assert readiness["attempts"] >= 1
        assert readiness["health_url"] == f"{_SERVICE_URL}/health"
        assert f"{_SERVICE_URL}/health" in readiness["gate"]
        assert "kestrel-dev-00007-abc" in result["error"]
        assert readiness["detail"] in result["error"]
        # The session reflects readiness honestly, not "unknown".
        assert result["session"]["health_status"] == "unready"
        assert result["session"]["revision"] == "kestrel-dev-00007-abc"
        # Sanitized: transport exception text never reaches the result.
        assert "SECRET" not in repr(result)
        # Not rolled back: the revision is left for inspection.
        assert provider.teardown_calls == 0
        assert (await manager.get_session("kestrel-dev")) is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["create", "update"])
    async def test_ready_revision_is_success(self, operation):
        provider = _FakeProvider(operation)
        manager = _manager_with(provider)

        with _serve(_status(200, {"status": "ok", "agent_initialized": True})):
            result = await manager.deploy_profile("dev", tag="v1.2.3")

        assert result["success"] is True
        assert result["control_plane_status"] == "succeeded"
        assert result["readiness_status"] == "ready"
        assert result["operation"] == operation
        assert result["readiness"]["failure"] is None
        assert result["readiness"]["last_status_code"] == 200
        assert result["session"]["health_status"] == "ready"
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_timeout_error_type_names_the_failure_without_its_message(self):
        manager = _manager_with(_FakeProvider("create"))

        with _serve(_raises(httpx.ConnectTimeout)):
            result = await manager.deploy_profile("dev", tag="v1.2.3")

        assert "ConnectTimeout" in result["readiness"]["detail"]
        assert "SECRET" not in result["error"]

    @pytest.mark.asyncio
    async def test_missing_service_url_is_unknown_and_not_success(self):
        manager = _manager_with(_FakeProvider("create", service_url=None))

        with patch(
            "kestrel_sovereign.features.deploy.core.probe_http_health"
        ) as probe:
            result = await manager.deploy_profile("dev", tag="v1.2.3")

        probe.assert_not_called()
        assert result["success"] is False
        assert result["control_plane_status"] == "succeeded"
        assert result["readiness_status"] == "unknown"
        assert result["readiness"]["failure"] == "no_service_url"

    @pytest.mark.asyncio
    async def test_gate_that_never_probed_is_unknown(self):
        manager = _manager_with(_FakeProvider("create"))
        manager.health_check_timeout = 0

        with patch(
            "kestrel_sovereign.features.deploy.core.probe_http_health"
        ) as probe:
            result = await manager.deploy_profile("dev", tag="v1.2.3")

        probe.assert_not_called()
        assert result["success"] is False
        assert result["readiness_status"] == "unknown"
        assert result["readiness"]["failure"] == "not_probed"

    @pytest.mark.asyncio
    async def test_control_plane_failure_is_reported_separately(self):
        provider = _FakeProvider("create")

        async def failing_deploy(**_kwargs):
            raise DeployManagerError("Deployment failed: quota exceeded")

        provider.deploy = failing_deploy
        manager = _manager_with(provider)

        result = await manager.deploy_profile("dev", tag="v1.2.3")

        assert result["success"] is False
        assert result["control_plane_status"] == "failed"
        assert result["readiness_status"] == "unknown"

    @pytest.mark.asyncio
    async def test_refusal_before_provider_is_not_a_control_plane_failure(self):
        manager = _manager_with(_FakeProvider("create"))

        result = await manager.deploy_profile("dev", tag="latest")

        assert result["success"] is False
        assert result["control_plane_status"] == "not_started"

    @pytest.mark.asyncio
    async def test_provider_iam_warning_reaches_the_result(self):
        """The Cloud Run IAM grant is warning-only; an unreachable service it
        leaves behind is reported as unready with the warning alongside."""
        provider = _FakeProvider("create")
        original = provider.deploy

        async def deploy_with_warning(**kwargs):
            result = await original(**kwargs)
            result["warnings"] = ["allUsers/run.invoker could not be granted"]
            return result

        provider.deploy = deploy_with_warning
        manager = _manager_with(provider)

        with _serve(_status(403)):
            result = await manager.deploy_profile("dev", tag="v1.2.3")

        assert result["success"] is False
        assert result["readiness"]["failure"] == "auth_rejected"
        assert result["warnings"] == ["allUsers/run.invoker could not be granted"]

    def test_readiness_check_has_no_truth_value(self):
        check = ReadinessCheck(status=ReadinessStatus.UNREADY, gate="g")
        with pytest.raises(TypeError, match="no truth value"):
            bool(check)
