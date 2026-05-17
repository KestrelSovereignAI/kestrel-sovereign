"""
Factory for creating TrainingProvider instances.

Handles auto-detection of available providers based on environment
variables and provides a consistent way to get providers.

Usage:
    # Get default (best available) provider
    provider = TrainingProviderFactory.get_default_provider()

    # Get specific provider
    provider = TrainingProviderFactory.get_provider("vertex_ai")

    # List available providers
    available = TrainingProviderFactory.list_available_providers()

    # Get first available provider without additional content filters
    provider = TrainingProviderFactory.get_unfiltered_provider()

    # Check provider capabilities
    caps = TrainingProviderFactory.get_capabilities("replicate")
"""

import logging
import os
from typing import Dict, List, Optional

from .protocol import TrainingProvider
from .types import ProviderCapabilities

logger = logging.getLogger(__name__)


class TrainingProviderFactory:
    """
    Factory for creating and managing TrainingProvider instances.

    Supports auto-detection of available providers based on environment.
    Caches instances for reuse.
    """

    # Provider priority for auto-selection
    # Local MPS first: zero cloud cost, Apple Silicon with MPS backend
    # RunPod second: persistent pod with FLUX.2, supports both training AND generation
    # Vertex AI third: serverless, supports both training AND generation (batch mode)
    PROVIDER_PRIORITY = [
        "local_mps",     # Local Apple Silicon (MPS backend), zero cloud cost
        "runpod",        # Persistent pod, FLUX.2, training + generation
        "vertex_ai",     # Serverless, FLUX.2, training + generation (batch)
        "replicate",     # Serverless, FLUX.1 only, censored
        "gcp_compute",   # VM-based, good reliability
        "vastai",        # Marketplace, cheapest but variable
    ]

    # Environment variables that enable each provider
    PROVIDER_ENV_VARS = {
        "local_mps": [],   # No API keys - local hardware
        "vertex_ai": ["GCP_PROJECT_ID"],
        "runpod": ["RUNPOD_API_KEY"],
        "gcp_compute": ["GCP_PROJECT_ID"],
        "replicate": ["REPLICATE_API_TOKEN"],
        "vastai": ["VASTAI_API_KEY"],
    }

    # Provider capabilities - what each provider supports
    # Used for intelligent routing based on content requirements
    PROVIDER_CAPABILITIES: Dict[str, ProviderCapabilities] = {
        "local_mps": ProviderCapabilities(
            training=True,
            generation=True,  # SDXL pipeline on MPS
            unfiltered_generation=True,
            flux_version="sdxl",  # Using SDXL (commercial-friendly license)
            supports_lora_download=True,
        ),
        "runpod": ProviderCapabilities(
            training=True,
            generation=True,
            unfiltered_generation=True,
            flux_version="2.x",
            supports_lora_download=True,
        ),
        "vertex_ai": ProviderCapabilities(
            training=True,
            generation=True,
            unfiltered_generation=True,
            flux_version="2.x",
            supports_lora_download=True,
        ),
        "replicate": ProviderCapabilities(
            training=True,
            generation=True,
            unfiltered_generation=False,  # Replicate applies content safety filters
            flux_version="1.x",  # Uses FLUX.1-dev, not FLUX.2
            supports_lora_download=True,  # Weights can be downloaded and used elsewhere
        ),
        "gcp_compute": ProviderCapabilities(
            training=True,
            generation=True,
            unfiltered_generation=True,
            flux_version="2.x",
            supports_lora_download=True,
        ),
        "vastai": ProviderCapabilities(
            training=True,
            generation=True,
            unfiltered_generation=True,
            flux_version="2.x",
            supports_lora_download=True,
        ),
    }

    # Cached instances
    _instances: Dict[str, TrainingProvider] = {}

    @classmethod
    def get_provider(cls, name: str) -> Optional[TrainingProvider]:
        """
        Get a specific provider by name.

        Args:
            name: Provider name ('vertex_ai', 'runpod', 'gcp_compute', 'replicate', 'vastai')

        Returns:
            TrainingProvider instance or None if not available
        """
        if name in cls._instances:
            return cls._instances[name]

        provider = cls._create_provider(name)
        if provider and provider.is_available():
            cls._instances[name] = provider
            return provider

        logger.warning(f"Provider '{name}' not available")
        return None

    @classmethod
    def get_default_provider(cls) -> Optional[TrainingProvider]:
        """
        Get the best available provider based on priority.

        Tries providers in order: vertex_ai > replicate > gcp_compute > vastai

        Returns:
            First available TrainingProvider or None

        Environment:
            GENERATION_PROVIDER: Force a specific provider (e.g., "vertex_ai", "runpod")
        """
        # Check for forced provider via env var
        forced_provider = os.getenv("GENERATION_PROVIDER")
        if forced_provider:
            provider = cls.get_provider(forced_provider)
            if provider:
                logger.info(f"Using forced training provider: {forced_provider} (GENERATION_PROVIDER env)")
                return provider
            logger.warning(f"Forced provider '{forced_provider}' not available")

        for name in cls.PROVIDER_PRIORITY:
            provider = cls.get_provider(name)
            if provider:
                logger.info(f"Using training provider: {name}")
                return provider

        logger.warning("No training providers available")
        return None

    @classmethod
    def list_available_providers(cls) -> List[str]:
        """List names of all available providers."""
        available = []
        for name in cls.PROVIDER_PRIORITY:
            if cls._check_provider_available(name):
                available.append(name)
        return available

    @classmethod
    def get_capabilities(cls, name: str) -> Optional[ProviderCapabilities]:
        """
        Get capabilities for a specific provider.

        Args:
            name: Provider name

        Returns:
            ProviderCapabilities or None if provider unknown
        """
        return cls.PROVIDER_CAPABILITIES.get(name)

    @classmethod
    def get_unfiltered_provider(cls) -> Optional[TrainingProvider]:
        """
        Get first available provider without additional generation filters.

        Useful when an operator needs direct access to the model output.
        Priority follows PROVIDER_PRIORITY order.

        Returns:
            First available unfiltered provider, or None if none available
        """
        for name in cls.PROVIDER_PRIORITY:
            caps = cls.PROVIDER_CAPABILITIES.get(name)
            if caps and caps.unfiltered_generation:
                provider = cls.get_provider(name)
                if provider:
                    logger.info(f"Using unfiltered provider: {name}")
                    return provider

        logger.warning("No unfiltered providers available")
        return None

    @classmethod
    def get_generation_provider(cls, unfiltered: bool = False) -> Optional[TrainingProvider]:
        """
        Get a provider that supports image generation.

        Args:
            unfiltered: If True, only return providers without additional generation filters

        Returns:
            TrainingProvider with generation capability, or None

        Environment:
            GENERATION_PROVIDER: Force a specific provider (e.g., "vertex_ai", "runpod")
        """
        # Check for forced provider via env var
        forced_provider = os.getenv("GENERATION_PROVIDER")
        if forced_provider:
            caps = cls.PROVIDER_CAPABILITIES.get(forced_provider)
            if caps and caps.generation:
                provider = cls.get_provider(forced_provider)
                if provider:
                    logger.info(f"Using forced generation provider: {forced_provider} (GENERATION_PROVIDER env)")
                    return provider
            logger.warning(f"Forced provider '{forced_provider}' not available or doesn't support generation")

        for name in cls.PROVIDER_PRIORITY:
            caps = cls.PROVIDER_CAPABILITIES.get(name)
            if caps and caps.generation:
                if unfiltered and not caps.unfiltered_generation:
                    continue
                provider = cls.get_provider(name)
                if provider:
                    logger.info(
                        f"Using generation provider: {name} "
                        f"(unfiltered={caps.unfiltered_generation})"
                    )
                    return provider

        logger.warning(f"No generation providers available (unfiltered={unfiltered})")
        return None

    @classmethod
    def _check_provider_available(cls, name: str) -> bool:
        """Check if provider's required env vars are set."""
        env_vars = cls.PROVIDER_ENV_VARS.get(name, [])
        return all(os.getenv(var) for var in env_vars)

    @classmethod
    def get_local_provider(cls) -> Optional[TrainingProvider]:
        """
        Get local MPS training provider.

        Returns:
            LocalMPSTrainingAdapter if available, None otherwise
        """
        return cls.get_provider("local_mps")

    @classmethod
    def _create_provider(cls, name: str) -> Optional[TrainingProvider]:
        """Create a provider instance by name."""
        try:
            if name == "local_mps":
                from .adapters.local_mps_adapter import LocalMPSTrainingAdapter
                return LocalMPSTrainingAdapter()

            elif name == "vertex_ai":
                from .adapters.vertex_ai_adapter import VertexAITrainingAdapter
                return VertexAITrainingAdapter()

            elif name == "gcp_compute":
                from .adapters.gcp_compute_adapter import GCPComputeTrainingAdapter
                return GCPComputeTrainingAdapter()

            elif name == "replicate":
                from .adapters.replicate_adapter import ReplicateTrainingAdapter
                return ReplicateTrainingAdapter()

            elif name == "vastai":
                from .adapters.vastai_adapter import VastAITrainingAdapter
                return VastAITrainingAdapter()

            elif name == "runpod":
                from .adapters.runpod_adapter import RunPodTrainingAdapter
                return RunPodTrainingAdapter()

            else:
                logger.warning(f"Unknown provider: {name}")
                return None

        except ImportError as e:
            logger.error(f"Failed to import provider {name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create provider {name}: {e}")
            return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached provider instances."""
        cls._instances.clear()
