"""
Training Provider Module for Kestrel

Provides the TrainingProvider protocol and shared data types for LoRA training.
Adapter implementations (LocalMPS, RunPod, Vertex AI, GCP Compute, Replicate,
Vast.ai) and TrainingProviderFactory are available in the kestrel-training
private package.

Usage:
    from kestrel_sovereign.features.training import (
        TrainingProvider,
        TrainingJob,
        TrainingStatus,
        TrainingConfig,
        TrainingState,
    )

    # Implement the protocol for custom providers:
    class MyProvider(TrainingProvider):
        ...

    # Factory and adapters are in the kestrel-training package:
    # pip install kestrel-training
    try:
        from kestrel_training import TrainingProviderFactory
        provider = TrainingProviderFactory.get_default_provider()
    except ImportError:
        provider = None  # No adapters installed
"""

from .types import (
    TrainingState,
    ProviderType,
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
    ProviderCapabilities,
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

# Factory is no longer in core — available from kestrel-training package
try:
    from kestrel_training import TrainingProviderFactory  # type: ignore[import-not-found]
    TRAINING_FACTORY_AVAILABLE = True
except ImportError:
    TrainingProviderFactory = None  # type: ignore[assignment,misc]
    TRAINING_FACTORY_AVAILABLE = False

__all__ = [
    # Training Types
    "TrainingState",
    "ProviderType",
    "TrainingConfig",
    "TrainingJob",
    "TrainingStatus",
    "ProviderCapabilities",
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
    # Factory (None if kestrel-training not installed)
    "TrainingProviderFactory",
    "TRAINING_FACTORY_AVAILABLE",
]
