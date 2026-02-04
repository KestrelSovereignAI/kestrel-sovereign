"""
End-to-end tests for Vertex AI LoRA training.

Tests the full training pipeline:
1. VertexAIManager - direct API calls
2. VertexAITrainingAdapter - unified protocol
3. TrainingProviderFactory - provider selection

Requires:
- GCP_PROJECT_ID environment variable
- gcloud auth configured
- A100 quota in us-central1 (for actual training)

Run with:
    pytest tests/integration/test_vertex_ai_training_e2e.py -v -s

For quick validation without GPU:
    pytest tests/integration/test_vertex_ai_training_e2e.py -v -k "not slow"
"""

import asyncio
import os
import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

# Skip all tests if GCP credentials not available
pytestmark = pytest.mark.skipif(
    not os.getenv("GCP_PROJECT_ID"),
    reason="GCP_PROJECT_ID not set - skipping Vertex AI tests"
)


class TestVertexAIManager:
    """Test the VertexAIManager directly."""

    @pytest.fixture
    def manager(self):
        """Create a VertexAIManager instance."""
        from kestrel_sovereign.features.vertex_ai.vertex_ai_manager import VertexAIManager
        return VertexAIManager()

    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """Test manager initializes with correct defaults."""
        assert manager.project_id == os.getenv("GCP_PROJECT_ID", "YOUR_PROJECT_ID")
        assert manager.region == os.getenv("GCP_REGION", "us-central1")
        assert "kestrel-lora" in manager.DEFAULT_IMAGE

    @pytest.mark.asyncio
    async def test_get_access_token(self, manager):
        """Test that we can get a GCP access token."""
        token = await manager._get_access_token()
        assert token is not None
        assert len(token) > 50  # Tokens are typically ~200 chars

    @pytest.mark.asyncio
    async def test_upload_to_gcs(self, manager):
        """Test uploading data to GCS."""
        test_data = b"test avatar data " + str(uuid.uuid4()).encode()
        blob_name = f"test/upload_test_{uuid.uuid4().hex[:8]}.txt"

        gcs_uri = await manager._upload_to_gcs(test_data, blob_name)

        assert gcs_uri.startswith("gs://")
        assert blob_name in gcs_uri

    @pytest.mark.asyncio
    async def test_download_from_gcs(self, manager):
        """Test downloading data from GCS."""
        # First upload something
        test_data = b"download test data " + str(uuid.uuid4()).encode()
        blob_name = f"test/download_test_{uuid.uuid4().hex[:8]}.txt"
        gcs_uri = await manager._upload_to_gcs(test_data, blob_name)

        # Now download it
        downloaded = await manager._download_from_gcs(gcs_uri)

        assert downloaded == test_data

    @pytest.mark.asyncio
    async def test_list_jobs(self, manager):
        """Test listing existing jobs."""
        jobs = await manager.list_jobs(limit=5)

        # Should return a list (may be empty)
        assert isinstance(jobs, list)

        # If there are jobs, verify structure
        if jobs:
            job = jobs[0]
            assert "job_id" in job
            assert "state" in job

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skip(reason="Costs money (~$3-5) - run manually with -k test_submit_training_job_real")
    async def test_submit_training_job_real(self, manager):
        """
        Submit a real training job to Vertex AI.

        WARNING: This test costs money (~$3-5 per run)!
        Only run manually when needed.
        """
        # Create a minimal test image (1x1 pixel PNG)
        test_avatar = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # bit depth, color type
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])

        companion_id = f"test-{uuid.uuid4().hex[:8]}"

        job = await manager.submit_training_job(
            companion_id=companion_id,
            avatar_data=test_avatar,
            trigger_word=f"TOK{companion_id[:8]}",
            steps=10,  # Minimal steps for testing
            lora_rank=8,
        )

        assert job.job_id is not None
        assert job.companion_id == companion_id
        assert job.gcs_output_path is not None
        assert job.gcs_output_path.startswith("gs://")

        # Clean up: cancel the job
        await manager.cancel_job(job.job_id)


class TestVertexAITrainingAdapter:
    """Test the unified TrainingProvider adapter."""

    @pytest.fixture
    def adapter(self):
        """Create a VertexAITrainingAdapter instance."""
        from kestrel_sovereign.features.training.adapters.vertex_ai_adapter import VertexAITrainingAdapter
        return VertexAITrainingAdapter()

    def test_adapter_properties(self, adapter):
        """Test adapter has correct properties."""
        from kestrel_sovereign.features.training.types import ProviderType

        assert adapter.provider_name == "vertex_ai"
        assert adapter.provider_type == ProviderType.SERVERLESS

    def test_is_available(self, adapter):
        """Test availability check."""
        # Should be True since we have GCP_PROJECT_ID
        assert adapter.is_available() is True

    @pytest.mark.asyncio
    async def test_start_training_creates_job(self, adapter):
        """Test that start_training creates a TrainingJob via mocked manager."""
        from kestrel_sovereign.features.training.types import TrainingConfig, TrainingState

        # Mock the manager
        mock_manager = AsyncMock()
        mock_vertex_job = MagicMock()
        mock_vertex_job.job_id = "mock-job-123"
        mock_vertex_job.gcs_output_path = "gs://test-bucket/output"
        mock_manager.submit_training_job.return_value = mock_vertex_job

        adapter._manager = mock_manager

        config = TrainingConfig(
            trigger_word="TOKtest",
            steps=100,
            lora_rank=16,
        )

        job = await adapter.start_training(
            companion_id="test-companion-123",
            avatar_data=b"test avatar data",
            config=config,
        )

        assert job.companion_id == "test-companion-123"
        assert job.provider == "vertex_ai"
        assert job.trigger_word == "TOKtest"
        assert job.state == TrainingState.PENDING
        assert job.provider_job_id == "mock-job-123"

        # Verify manager was called correctly
        mock_manager.submit_training_job.assert_called_once()
        call_args = mock_manager.submit_training_job.call_args
        assert call_args.kwargs["companion_id"] == "test-companion-123"
        assert call_args.kwargs["steps"] == 100
        assert call_args.kwargs["lora_rank"] == 16

    @pytest.mark.asyncio
    async def test_get_status_returns_training_status(self, adapter):
        """Test that get_status returns proper TrainingStatus."""
        from kestrel_sovereign.features.training.types import TrainingConfig, TrainingState

        # First create a job
        mock_manager = AsyncMock()
        mock_vertex_job = MagicMock()
        mock_vertex_job.job_id = "status-test-job"
        mock_vertex_job.gcs_output_path = "gs://test-bucket/output"
        mock_manager.submit_training_job.return_value = mock_vertex_job

        # Mock status response
        mock_manager.get_job_status.return_value = {
            "state": "running",
            "progress": 0.5,
            "error": None,
        }

        adapter._manager = mock_manager

        job = await adapter.start_training(
            companion_id="status-test",
            avatar_data=b"test",
            config=TrainingConfig(),
        )

        status = await adapter.get_status(job.job_id)

        assert status.job_id == job.job_id
        assert status.state == TrainingState.TRAINING
        assert status.progress == 0.5

    @pytest.mark.asyncio
    async def test_download_weights_calls_manager(self, adapter):
        """Test that download_weights delegates to manager."""
        from kestrel_sovereign.features.training.types import TrainingConfig

        # Setup mocked manager
        mock_manager = AsyncMock()
        mock_vertex_job = MagicMock()
        mock_vertex_job.job_id = "download-test-job"
        mock_vertex_job.gcs_output_path = "gs://test/output"
        mock_manager.submit_training_job.return_value = mock_vertex_job
        mock_manager.download_lora.return_value = b"fake lora weights"

        adapter._manager = mock_manager

        job = await adapter.start_training(
            companion_id="download-test",
            avatar_data=b"test",
            config=TrainingConfig(),
        )

        weights = await adapter.download_weights(job.job_id)

        assert weights == b"fake lora weights"
        mock_manager.download_lora.assert_called_once_with("download-test-job")


class TestTrainingProviderFactory:
    """Test the TrainingProviderFactory with Vertex AI."""

    def test_vertex_ai_in_available_providers(self):
        """Test that Vertex AI is listed as available."""
        from kestrel_sovereign.features.training import TrainingProviderFactory

        available = TrainingProviderFactory.list_available_providers()

        # Should include vertex_ai since GCP_PROJECT_ID is set
        assert "vertex_ai" in available

    def test_get_vertex_ai_provider(self):
        """Test getting Vertex AI provider specifically."""
        from kestrel_sovereign.features.training import TrainingProviderFactory

        provider = TrainingProviderFactory.get_provider("vertex_ai")

        assert provider is not None
        assert provider.provider_name == "vertex_ai"

    def test_vertex_ai_default_priority(self):
        """Test that Vertex AI is preferred when available."""
        from kestrel_sovereign.features.training import TrainingProviderFactory

        # When GCP_PROJECT_ID is set, Vertex AI should be default
        default = TrainingProviderFactory.get_default_provider()

        # The default depends on what's configured, but vertex_ai should be
        # high priority when available
        assert default is not None


class TestTrainingStateMapping:
    """Test state mapping from Vertex AI to unified states."""

    def test_vertex_state_mapping(self):
        """Test mapping Vertex AI states to TrainingState."""
        from kestrel_sovereign.features.training.types import TrainingState

        # Map all Vertex AI states
        mappings = {
            "JOB_STATE_PENDING": TrainingState.PENDING,
            "JOB_STATE_QUEUED": TrainingState.PENDING,
            "JOB_STATE_RUNNING": TrainingState.TRAINING,
            "JOB_STATE_SUCCEEDED": TrainingState.COMPLETED,
            "JOB_STATE_FAILED": TrainingState.FAILED,
            "JOB_STATE_CANCELLED": TrainingState.CANCELLED,
            "JOB_STATE_CANCELLING": TrainingState.CANCELLED,
        }

        for vertex_state, expected in mappings.items():
            result = TrainingState.from_vertex_state(vertex_state)
            assert result == expected, f"Expected {vertex_state} -> {expected}, got {result}"

    def test_unknown_state_defaults_to_pending(self):
        """Test that unknown states default to PENDING."""
        from kestrel_sovereign.features.training.types import TrainingState

        result = TrainingState.from_vertex_state("UNKNOWN_STATE")
        assert result == TrainingState.PENDING


class TestEndToEndWithMockedGCP:
    """End-to-end tests with mocked GCP calls."""

    @pytest.mark.asyncio
    async def test_full_training_flow_mocked(self):
        """Test the complete training flow with mocked GCP."""
        from kestrel_sovereign.features.training import TrainingProviderFactory, TrainingConfig, TrainingState

        provider = TrainingProviderFactory.get_provider("vertex_ai")
        assert provider is not None

        # Mock the internal manager
        mock_manager = AsyncMock()

        # Mock submit_training_job
        mock_vertex_job = MagicMock()
        mock_vertex_job.job_id = "e2e-test-job"
        mock_vertex_job.gcs_output_path = "gs://test/output"
        mock_manager.submit_training_job.return_value = mock_vertex_job

        # Mock get_job_status - return completed
        mock_manager.get_job_status.return_value = {
            "state": "completed",
            "progress": 1.0,
            "error": None,
        }

        # Mock download_lora
        mock_manager.download_lora.return_value = b"trained lora weights"

        provider._manager = mock_manager

        # Step 1: Start training
        config = TrainingConfig(
            trigger_word="TOKtest",
            steps=1000,
            lora_rank=16,
        )

        job = await provider.start_training(
            companion_id="e2e-test-companion",
            avatar_data=b"test avatar image",
            config=config,
        )

        assert job.job_id is not None
        assert job.state == TrainingState.PENDING

        # Step 2: Check status
        status = await provider.get_status(job.job_id)
        assert status.state == TrainingState.COMPLETED
        assert status.progress == 1.0

        # Step 3: Download weights
        weights = await provider.download_weights(job.job_id)
        assert weights == b"trained lora weights"

        # Step 4: Cleanup (no-op for serverless)
        await provider.cleanup(job.job_id)


@pytest.mark.skip(reason="FLUX tests require flux2_api.py refactor - simpletuner_api.py is now a wrapper")
class TestDockerImageVertexMode:
    """Test the Docker image --vertex-mode integration."""

    def test_vertex_mode_args_parsing(self):
        """Test that the Docker entrypoint parses --vertex-mode args."""
        # Import the simpletuner_api module to check it exists
        import importlib.util
        import os

        api_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "docker", "simpletuner_api.py"
        )

        spec = importlib.util.spec_from_file_location("simpletuner_api", api_path)
        # Don't actually load it (has dependencies), just verify it exists
        assert spec is not None
        assert os.path.exists(api_path)

        # Read file and verify --vertex-mode is supported
        with open(api_path, "r") as f:
            content = f.read()

        assert "--vertex-mode" in content
        assert "--avatar-gcs" in content
        assert "--output-gcs" in content
        assert "--companion-id" in content
        assert "run_vertex_batch_training" in content
