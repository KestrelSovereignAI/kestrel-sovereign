"""Unit tests for Vast.ai GPU feature."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta, timezone

from kestrel_sovereign.features.vastai.manager import VastAIManager
from kestrel_sovereign.features.vastai.models import (
    VastAIManagerError,
    VastAISession,
    InstanceStatus,
    GPUProfile,
)


@pytest.fixture
def mock_vastai_sdk():
    """Mock the vastai_sdk module."""
    mock_sdk = MagicMock()
    with patch.dict("sys.modules", {"vastai_sdk": MagicMock(VastAI=MagicMock(return_value=mock_sdk))}):
        yield mock_sdk


@pytest.fixture
def sample_profile():
    """Create a sample GPU profile for testing."""
    return GPUProfile(
        id="test",
        name="Test Profile",
        task_type="test",
        image_name="pytorch/pytorch",
        disk_gb=50,
        gpu_ram_min=24,
        num_gpus=1,
        reliability_min=0.9,
        compute_cap_min=800,
        cuda_vers_min=12.0,
        ports=["8000/http"],
        inference_port=8000,
        inference_protocol="http",
        inference_base_path="/v1",
        default_model="test-model",
    )


@pytest.fixture
def sample_session(sample_profile):
    """Create a sample session for testing."""
    now = datetime.now(timezone.utc)
    return VastAISession(
        instance_id=12345,
        profile=sample_profile,
        task_profile="test",
        model_name="test-model",
        status=InstanceStatus.RUNNING,
        ttl_seconds=3600,
        started_at=now,
        expires_at=now + timedelta(hours=1),
        ssh_host="1.2.3.4",
        ssh_port=22,
        backend_base_url="http://1.2.3.4:8000",
        inference_url="http://1.2.3.4:8000/v1",
        actual_cost_per_hr=0.35,
        gpu_name="RTX 3090",
    )


class TestGPUProfile:
    """Tests for GPUProfile dataclass."""

    def test_profile_creation(self, sample_profile):
        """Test basic profile creation."""
        assert sample_profile.id == "test"
        assert sample_profile.gpu_ram_min == 24
        assert sample_profile.reliability_min == 0.9
        assert sample_profile.compute_cap_min == 800

    def test_profile_defaults(self):
        """Test profile default values."""
        profile = GPUProfile(
            id="minimal",
            name="Minimal Profile",
            task_type="test",
            image_name="test:latest",
            disk_gb=20,
            gpu_ram_min=8,
        )
        assert profile.num_gpus == 1
        assert profile.reliability_min == 0.9
        assert profile.ports == ["8888/http"]


class TestVastAISession:
    """Tests for VastAISession dataclass."""

    def test_session_to_dict(self, sample_session):
        """Test session serialization."""
        data = sample_session.to_dict()
        assert data["instance_id"] == 12345
        assert data["status"] == "running"
        assert data["gpu_name"] == "RTX 3090"
        assert data["actual_cost_per_hr"] == 0.35

    def test_session_remaining_ttl(self, sample_session):
        """Test remaining TTL calculation."""
        # Session expires in 1 hour, so remaining should be ~3600
        assert 3500 < sample_session.remaining_ttl_seconds <= 3600

    def test_session_is_active(self, sample_session):
        """Test is_active property."""
        assert sample_session.is_active is True

        sample_session.status = InstanceStatus.EXITED
        assert sample_session.is_active is False

        sample_session.status = InstanceStatus.CREATING
        assert sample_session.is_active is True


class TestVastAIManager:
    """Tests for VastAIManager class."""

    def test_build_search_query(self, mock_vastai_sdk):
        """Test search query construction from profile."""
        with patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"}):
            manager = VastAIManager(config={
                "manager": {},
                "profiles": {
                    "test": {
                        "name": "Test",
                        "image_name": "test:latest",
                        "disk_gb": 50,
                        "gpu_ram_min": 24,
                        "reliability_min": 0.95,
                        "compute_cap_min": 800,
                        "cost_per_hr_max": 0.50,
                    }
                }
            })

        profile = manager.profiles["test"]
        query = manager._build_search_query(profile)

        assert "gpu_ram >= 24" in query
        assert "reliability > 0.95" in query
        assert "compute_cap > 800" in query
        assert "dph <= 0.5" in query
        assert "rentable = true" in query

    def test_map_status(self):
        """Test status string mapping."""
        assert VastAIManager._map_status("running") == InstanceStatus.RUNNING
        assert VastAIManager._map_status("RUNNING") == InstanceStatus.RUNNING
        assert VastAIManager._map_status("creating") == InstanceStatus.CREATING
        assert VastAIManager._map_status("loading") == InstanceStatus.LOADING
        assert VastAIManager._map_status("exited") == InstanceStatus.EXITED
        assert VastAIManager._map_status("error") == InstanceStatus.ERROR
        assert VastAIManager._map_status("unknown") == InstanceStatus.OFFLINE
        assert VastAIManager._map_status(None) == InstanceStatus.OFFLINE

    def test_validate_ttl(self, mock_vastai_sdk):
        """Test TTL validation."""
        with patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"}):
            manager = VastAIManager(config={
                "manager": {
                    "default_ttl_seconds": 3600,
                    "max_ttl_seconds": 7200,
                },
                "profiles": {}
            })

        # Default TTL
        assert manager._validate_ttl(None) == 3600

        # Custom TTL within limits
        assert manager._validate_ttl(1800) == 1800

        # TTL exceeds max
        with pytest.raises(VastAIManagerError, match="exceeds max"):
            manager._validate_ttl(10000)

    def test_select_profile_valid(self, mock_vastai_sdk):
        """Test profile selection with valid profile."""
        with patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"}):
            manager = VastAIManager(config={
                "manager": {},
                "profiles": {
                    "training": {
                        "name": "Training",
                        "image_name": "test:latest",
                        "disk_gb": 50,
                        "gpu_ram_min": 24,
                    }
                }
            })

        profile = manager._select_profile("training")
        assert profile.id == "training"
        assert profile.name == "Training"

    def test_select_profile_invalid(self, mock_vastai_sdk):
        """Test profile selection with invalid profile."""
        with patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"}):
            manager = VastAIManager(config={
                "manager": {},
                "profiles": {
                    "training": {
                        "name": "Training",
                        "image_name": "test:latest",
                        "disk_gb": 50,
                        "gpu_ram_min": 24,
                    }
                }
            })

        with pytest.raises(VastAIManagerError, match="Unknown task_profile"):
            manager._select_profile("nonexistent")

    @pytest.mark.asyncio
    async def test_get_status_no_session(self, mock_vastai_sdk):
        """Test get_status when no session is active."""
        with patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"}):
            manager = VastAIManager(config={"manager": {}, "profiles": {}})

        status = await manager.get_status()
        assert status["active"] is False
        assert status["status"] == "offline"

    @pytest.mark.asyncio
    async def test_search_offers(self, mock_vastai_sdk):
        """Test searching for GPU offers."""
        mock_vastai_sdk.search_offers.return_value = {
            "offers": [
                {
                    "id": 1,
                    "gpu_name": "RTX 3090",
                    "gpu_ram": 24,
                    "dph_total": 0.35,
                    "reliability": 0.95,
                },
                {
                    "id": 2,
                    "gpu_name": "RTX 4090",
                    "gpu_ram": 24,
                    "dph_total": 0.45,
                    "reliability": 0.98,
                },
            ]
        }

        with patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"}):
            manager = VastAIManager(config={"manager": {}, "profiles": {}})
            # Pre-initialize the SDK
            manager._sdk = mock_vastai_sdk

        offers = await manager.search_offers(query="gpu_ram >= 24", limit=2)

        assert len(offers) == 2
        # Should be sorted by price ascending
        assert offers[0]["dph_total"] == 0.35
        assert offers[1]["dph_total"] == 0.45

    @pytest.mark.asyncio
    async def test_show_instances(self, mock_vastai_sdk):
        """Test listing all instances."""
        mock_vastai_sdk.show_instances.return_value = {
            "instances": [
                {"id": 1, "actual_status": "running"},
                {"id": 2, "actual_status": "exited"},
            ]
        }

        with patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"}):
            manager = VastAIManager(config={"manager": {}, "profiles": {}})
            manager._sdk = mock_vastai_sdk

        instances = await manager.show_instances()
        assert len(instances) == 2


class TestVastAIManagerErrors:
    """Tests for error handling."""

    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove VASTAI_API_KEY if present
            import os
            os.environ.pop("VASTAI_API_KEY", None)

            manager = VastAIManager(config={"manager": {}, "profiles": {}})
            # Should not raise until SDK is actually used
            assert manager.api_key is None

            with pytest.raises(VastAIManagerError, match="VASTAI_API_KEY is required"):
                manager._get_sdk()

    def test_incomplete_profile(self):
        """Test error with incomplete profile config."""
        with patch.dict("os.environ", {"VASTAI_API_KEY": "test-key"}):
            with pytest.raises(VastAIManagerError, match="missing"):
                VastAIManager(config={
                    "manager": {},
                    "profiles": {
                        "broken": {
                            "name": "Broken Profile",
                            # Missing required fields
                        }
                    }
                })


class TestInstanceStatusMapping:
    """Tests for status mapping edge cases."""

    def test_all_known_statuses(self):
        """Test all known Vast.ai status strings."""
        mappings = {
            "running": InstanceStatus.RUNNING,
            "ready": InstanceStatus.RUNNING,
            "creating": InstanceStatus.CREATING,
            "starting": InstanceStatus.CREATING,
            "provisioning": InstanceStatus.CREATING,
            "loading": InstanceStatus.LOADING,
            "pulling": InstanceStatus.LOADING,
            "stopping": InstanceStatus.EXITED,
            "exited": InstanceStatus.EXITED,
            "stopped": InstanceStatus.EXITED,
            "error": InstanceStatus.ERROR,
            "failed": InstanceStatus.ERROR,
        }

        for raw, expected in mappings.items():
            assert VastAIManager._map_status(raw) == expected
            # Also test uppercase
            assert VastAIManager._map_status(raw.upper()) == expected
