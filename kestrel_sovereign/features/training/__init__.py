"""
Training Provider Module for Kestrel

Provides a unified interface for LoRA training across multiple providers:
- Vertex AI (Google serverless Custom Jobs)
- GCP Compute Engine (Google VM instances)
- RunPod (managed GPU pods)
- Vast.ai (GPU marketplace)
- Replicate (managed training API)

Usage:
    from kestrel_sovereign.features.training import (
        TrainingProviderFactory,
        TrainingProvider,
        TrainingJob,
        TrainingStatus,
        TrainingConfig,
        TrainingState,
    )

    # Get best available provider
    provider = TrainingProviderFactory.get_default_provider()

    # Start training
    job = await provider.start_training(
        companion_id="abc123",
        avatar_data=avatar_bytes,
        config=TrainingConfig(steps=1000)
    )

    # Poll status
    status = await provider.get_status(job.job_id)

    # Download weights when complete
    if status.state == TrainingState.COMPLETED:
        weights = await provider.download_weights(job.job_id)
"""

from .types import (
    TrainingState,
    ProviderType,
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
    # Generation types
    GenerationState,
    GenerationConfig,
    GenerationResult,
)

from .protocol import (
    TrainingProvider,
    TrainingProviderError,
    ProviderNotAvailableError,
    TrainingSubmissionError,
    TrainingStatusError,
    DownloadError,
    GenerationError,
)

from .factory import TrainingProviderFactory

__all__ = [
    # Training Types
    "TrainingState",
    "ProviderType",
    "TrainingConfig",
    "TrainingJob",
    "TrainingStatus",
    # Generation Types
    "GenerationState",
    "GenerationConfig",
    "GenerationResult",
    # Protocol
    "TrainingProvider",
    "TrainingProviderError",
    "ProviderNotAvailableError",
    "TrainingSubmissionError",
    "TrainingStatusError",
    "DownloadError",
    "GenerationError",
    # Factory
    "TrainingProviderFactory",
]
