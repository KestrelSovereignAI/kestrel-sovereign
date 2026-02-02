"""
Integration tests for GCP Compute Engine GPU management.

These tests require:
1. GCP_PROJECT_ID environment variable
2. GOOGLE_APPLICATION_CREDENTIALS or GCP_SERVICE_ACCOUNT_KEY
3. SSH key at ~/.ssh/kestrel_gcp (for SSH-based tests)
4. GPU quota in the specified zone

Run with:
    pytest tests/integration/test_gcp_compute_e2e.py -v -x

Skip GPU-creating tests (expensive):
    pytest tests/integration/test_gcp_compute_e2e.py -v -x -k "not create_instance"
"""

import asyncio
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Skip all tests if GCP not configured
pytestmark = pytest.mark.skipif(
    not os.getenv("GCP_PROJECT_ID"),
    reason="GCP_PROJECT_ID not set"
)


class TestGCPComputeManagerUnit:
    """Unit tests that don't require GCP credentials."""

    def test_import(self):
        """Test that the module can be imported."""
        from kestrel_sovereign.features.gcp_compute import (
            GCPComputeManager,
            GCPComputeManagerError,
            GCPComputeSession,
            GPUProfile,
            InstanceStatus,
        )
        assert GCPComputeManager is not None
        assert InstanceStatus.RUNNING.value == "running"

    def test_load_config(self):
        """Test config loading from gcp_compute_config.toml."""
        from kestrel_sovereign.features.gcp_compute.manager import GCPComputeEngineManager as GCPComputeManager

        # Create with mock to avoid actual GCP calls
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            manager = GCPComputeManager()

            assert manager.project_id == "test-project"
            assert "training" in manager.profiles
            assert "inference" in manager.profiles
            assert manager.profiles["training"].gpu_type == "nvidia-tesla-a100"

    def test_profile_attributes(self):
        """Test GPUProfile dataclass attributes."""
        from kestrel_sovereign.features.gcp_compute.models import GPUProfile

        profile = GPUProfile(
            id="test",
            name="Test Profile",
            task_type="training",
            machine_type="a2-highgpu-1g",
            gpu_type="nvidia-tesla-a100",
            gpu_count=1,
            boot_disk_size_gb=100,
            image_name="gcr.io/test/image:latest",
            cost_per_hr_spot=1.10,
        )

        assert profile.id == "test"
        assert profile.cost_per_hr_spot == 1.10
        assert profile.readiness_timeout_seconds == 600  # default

    def test_instance_status_from_gcp(self):
        """Test GCP status conversion."""
        from kestrel_sovereign.features.gcp_compute.models import InstanceStatus

        assert InstanceStatus.from_gcp_status("RUNNING") == InstanceStatus.RUNNING
        assert InstanceStatus.from_gcp_status("TERMINATED") == InstanceStatus.TERMINATED
        assert InstanceStatus.from_gcp_status("STAGING") == InstanceStatus.STAGING
        assert InstanceStatus.from_gcp_status("UNKNOWN") == InstanceStatus.ERROR

    def test_session_to_dict(self):
        """Test session serialization."""
        from kestrel_sovereign.features.gcp_compute.manager import GCPComputeEngineManager as GCPComputeManager
        from kestrel_sovereign.features.gcp_compute.models import (
            GCPComputeSession,
            GPUProfile,
            InstanceStatus,
        )

        profile = GPUProfile(
            id="training",
            name="Test",
            task_type="training",
            machine_type="a2-highgpu-1g",
            gpu_type="nvidia-tesla-a100",
            gpu_count=1,
            boot_disk_size_gb=100,
            image_name="gcr.io/test/image:latest",
        )

        now = datetime.now(timezone.utc)
        session = GCPComputeSession(
            instance_name="test-instance",
            zone="us-central1-a",
            profile=profile,
            task_profile="training",
            model_name="flux2-dev",
            status=InstanceStatus.RUNNING,
            ttl_seconds=3600,
            started_at=now,
            expires_at=now,
            external_ip="35.1.2.3",
        )

        data = session.to_dict()
        assert data["instance_name"] == "test-instance"
        assert data["status"] == "running"
        assert data["external_ip"] == "35.1.2.3"


@pytest.mark.cloud_resource
class TestGCPComputeManagerIntegration:
    """Integration tests that require GCP credentials.

    Run with: pytest --run-cloud tests/integration/test_gcp_compute_e2e.py -v
    """

    @pytest.fixture
    def manager(self):
        """Create a GCPComputeManager instance."""
        from kestrel_sovereign.features.gcp_compute.manager import GCPComputeEngineManager as GCPComputeManager
        return GCPComputeManager()

    @pytest.mark.asyncio
    async def test_get_status_no_session(self, manager):
        """Test getting status when no session exists."""
        status = await manager.get_status()
        assert status["status"] == "offline"

    @pytest.mark.asyncio
    async def test_list_instances(self, manager):
        """Test listing Kestrel instances."""
        instances = await manager.list_instances()
        assert isinstance(instances, list)
        # May be empty if no instances running

    @pytest.mark.asyncio
    async def test_list_disks(self, manager):
        """Test listing persistent disks."""
        disks = await manager.list_disks()
        assert isinstance(disks, list)
        # May be empty if no disks created

    @pytest.mark.asyncio
    async def test_ensure_persistent_disk(self, manager):
        """Test creating a persistent disk."""
        # WARNING: This creates a real disk that costs money
        result = await manager.ensure_persistent_disk(
            disk_name="kestrel-test-disk",
            zone="us-central1-a"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_start_stop_session(self, manager):
        """Test starting and stopping a GPU session."""
        # WARNING: This creates a real GPU instance that costs ~$1/hour
        try:
            # Start with spot for lower cost
            result = await manager.start_session(
                task_profile="budget",  # T4 is cheapest
                ttl_seconds=300,  # 5 min
                use_spot=True,
            )

            assert result["status"] in ["provisioning", "staging", "running"]
            assert result["instance_name"].startswith("kestrel-")

            # Wait for running
            for _ in range(30):
                status = await manager.get_status()
                if status["status"] == "running":
                    break
                await asyncio.sleep(10)

            # Stop
            final = await manager.stop_session()
            assert final["status"] == "terminated"

        finally:
            # Cleanup: ensure instance is deleted
            if manager._session:
                await manager.stop_session()


class TestGCPComputeFeature:
    """Tests for the GCPComputeFeature tool interface."""

    def test_import_feature(self):
        """Test that the feature can be imported."""
        from kestrel_sovereign.features.gcp_compute.feature import GCPComputeFeature
        assert GCPComputeFeature is not None

    @pytest.mark.asyncio
    async def test_feature_initialization(self):
        """Test feature initialization."""
        from kestrel_sovereign.features.gcp_compute.feature import GCPComputeFeature

        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project"}):
            feature = GCPComputeFeature(agent=None)
            await feature.initialize()

            assert feature.manager is not None
            assert feature.manager.project_id == "test-project"


@pytest.mark.cloud_resource
class TestVisualIdentityGCPIntegration:
    """Test GCP integration in VisualIdentityFeature.

    Run with: pytest --run-cloud tests/integration/test_gcp_compute_e2e.py -v
    """

    @pytest.mark.asyncio
    async def test_gcp_provider_detection(self, monkeypatch):
        """Test that GCP is detected as a LoRA training provider."""
        from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

        # Clear any existing env vars using monkeypatch
        for k in ["VASTAI_API_KEY", "RUNPOD_API_KEY", "GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"]:
            monkeypatch.delenv(k, raising=False)

        # Set only GCP credentials
        monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

        feature = VisualIdentityFeature(agent=None)
        await feature.initialize()

        # Check that GCP would be initialized
        result = feature._ensure_lora_services()

        # Should have initialized GCP manager
        assert feature.gcp_manager is not None
        assert feature.vastai_manager is None
        assert feature.runpod_manager is None
