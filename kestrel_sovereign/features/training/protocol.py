"""
TrainingProvider Protocol for unified LoRA training across providers.

This protocol abstracts the differences between:
- Serverless providers (Vertex AI, Replicate) - job runs to completion
- Session-based providers (GCP Compute, RunPod, Vast.ai) - requires instance lifecycle

All providers expose the same interface; session management is hidden internally.

Usage:
    from kestrel_sovereign.features.training import TrainingProviderFactory, TrainingConfig

    # Get default provider (auto-detected from environment)
    provider = TrainingProviderFactory.get_default_provider()

    # Or get specific provider
    provider = TrainingProviderFactory.get_provider("vertex_ai")

    # Start training
    job = await provider.start_training(
        companion_id="abc123",
        avatar_data=image_bytes,
        config=TrainingConfig(steps=1000, lora_rank=16)
    )

    # Poll for completion
    while True:
        status = await provider.get_status(job.job_id)
        if status.state.is_terminal():
            break
        await asyncio.sleep(30)

    # Download weights
    if status.state == TrainingState.COMPLETED:
        lora_bytes = await provider.download_weights(job.job_id)
        # Store lora_bytes to database

    # Cleanup (important for session-based providers)
    await provider.cleanup(job.job_id)
"""

from typing import Optional, Protocol, runtime_checkable

from .types import TrainingJob, TrainingStatus, TrainingConfig, ProviderType


@runtime_checkable
class TrainingProvider(Protocol):
    """
    Unified interface for LoRA training providers.

    Lifecycle models are abstracted away:
    - Serverless: start_training() submits job directly
    - Session-based: start_training() starts instance, submits job

    All cleanup is handled internally or via cleanup() method.
    """

    @property
    def provider_name(self) -> str:
        """
        Unique provider identifier.

        Returns one of:
        - 'vertex_ai' (Google Vertex AI Custom Jobs)
        - 'gcp_compute' (Google Compute Engine VMs)
        - 'runpod' (RunPod managed pods)
        - 'vastai' (Vast.ai GPU marketplace)
        - 'replicate' (Replicate managed API)
        """
        ...

    @property
    def provider_type(self) -> ProviderType:
        """
        Provider lifecycle type.

        Returns:
        - ProviderType.SERVERLESS: Jobs run to completion (Vertex AI, Replicate)
        - ProviderType.SESSION_BASED: Requires instance management (GCP, RunPod, Vast.ai)
        """
        ...

    def is_available(self) -> bool:
        """
        Check if provider is configured and available.

        Returns True if required environment variables are set and
        the provider can accept training jobs.
        """
        ...

    async def start_training(
        self,
        companion_id: str,
        avatar_data: bytes,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """
        Start a LoRA training job.

        For serverless providers: Submits job directly.
        For session-based: Starts instance, then submits job.

        Args:
            companion_id: Companion UUID for tracking
            avatar_data: Avatar image bytes (JPEG/PNG)
            config: Optional training configuration (defaults used if None)

        Returns:
            TrainingJob with job_id and initial status

        Raises:
            TrainingProviderError: If training cannot be started
        """
        ...

    async def get_status(self, job_id: str) -> TrainingStatus:
        """
        Get current status of a training job.

        Args:
            job_id: Training job ID returned from start_training()

        Returns:
            TrainingStatus with state, progress (0.0-1.0), error info
        """
        ...

    async def download_weights(self, job_id: str) -> Optional[bytes]:
        """
        Download trained LoRA weights from completed job.

        Should only be called after job reaches COMPLETED state.

        Args:
            job_id: Completed training job ID

        Returns:
            LoRA weights as bytes (.safetensors format), or None if not ready
        """
        ...

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a running training job.

        For session-based providers, also terminates the instance.

        Args:
            job_id: Training job ID to cancel

        Returns:
            True if cancelled successfully
        """
        ...

    async def cleanup(self, job_id: str) -> None:
        """
        Cleanup resources after job completion.

        For serverless: Usually no-op
        For session-based: Terminates instance if still running

        Should be called after download_weights() for session-based providers
        to avoid paying for idle instances.

        Args:
            job_id: Job ID to cleanup
        """
        ...


class TrainingProviderError(Exception):
    """Base exception for training provider errors."""

    def __init__(self, message: str, provider: str = "unknown", details: dict = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class ProviderNotAvailableError(TrainingProviderError):
    """Provider is not configured or unavailable."""
    pass


class TrainingSubmissionError(TrainingProviderError):
    """Failed to submit training job."""
    pass


class TrainingStatusError(TrainingProviderError):
    """Failed to get training status."""
    pass


class DownloadError(TrainingProviderError):
    """Failed to download trained weights."""
    pass


class GenerationError(TrainingProviderError):
    """Failed to generate image."""
    pass
