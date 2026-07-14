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

    # Get first available uncensored provider
    provider = TrainingProviderFactory.get_uncensored_provider()

    # Check provider capabilities
    caps = TrainingProviderFactory.get_capabilities("replicate")
"""

import logging
import os
from typing import Dict, List, Optional

from .protocol import DEFAULT_PROVIDER_PRIORITY, TrainingProvider
from .types import ProviderCapabilities

logger = logging.getLogger(__name__)

# Entry-point group external packages register generation/training backends
# under. Mirrors ``kestrel_sovereign.llm_providers`` (see
# kestrel_sovereign/llm/provider_registry.py) so a package can contribute a
# provider without editing this factory::
#
#     [project.entry-points."kestrel_sovereign.training_providers"]
#     catalog_worker = "frinz.services.catalog_worker_provider:CatalogWorkerProvider"
TRAINING_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.training_providers"

# Numeric spacing between successive built-in providers. Each built-in's
# priority is its index in PROVIDER_PRIORITY times this step, so an entry-point
# provider can slot *between* two built-ins by declaring an in-between priority
# (e.g. priority=5 lands between local_mps=0 and runpod=10).
_BUILTIN_PRIORITY_STEP = 10


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
        "runpod",        # Persistent pod, FLUX.2, uncensored, training + generation
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
            uncensored=True,  # SDXL is not content-filtered
            flux_version="sdxl",  # Using SDXL (commercial-friendly license)
            supports_lora_download=True,
        ),
        "runpod": ProviderCapabilities(
            training=True,
            generation=True,
            uncensored=True,
            flux_version="2.x",
            supports_lora_download=True,
        ),
        "vertex_ai": ProviderCapabilities(
            training=True,
            generation=True,
            uncensored=True,
            flux_version="2.x",
            supports_lora_download=True,
        ),
        "replicate": ProviderCapabilities(
            training=True,
            generation=True,
            uncensored=False,  # Replicate applies content safety filters
            flux_version="1.x",  # Uses FLUX.1-dev, not FLUX.2
            supports_lora_download=True,  # Weights can be downloaded and used elsewhere
        ),
        "gcp_compute": ProviderCapabilities(
            training=True,
            generation=True,
            uncensored=True,
            flux_version="2.x",
            supports_lora_download=True,
        ),
        "vastai": ProviderCapabilities(
            training=True,
            generation=True,
            uncensored=True,
            flux_version="2.x",
            supports_lora_download=True,
        ),
    }

    # Cached instances
    _instances: Dict[str, TrainingProvider] = {}

    # Methods a provider MUST expose given the capabilities it declares. Used
    # at entry-point discovery to drop plugins whose method surface doesn't
    # match what their `capabilities` promise (codex round-2 on #2445).
    _REQUIRED_METHODS_BY_CAPABILITY = {
        "training": (
            "start_training", "get_status", "download_weights",
            "cancel", "cleanup",
        ),
        "generation": ("generate_image",),
    }

    @classmethod
    def _required_methods_missing(
        cls, obj: type, caps: ProviderCapabilities,
    ) -> set:
        """Return the set of methods a plugin promised via capabilities but
        didn't implement. Empty set means the contract holds."""
        missing = set()
        if getattr(caps, "training", False):
            for name in cls._REQUIRED_METHODS_BY_CAPABILITY["training"]:
                if not callable(getattr(obj, name, None)):
                    missing.add(name)
        if getattr(caps, "generation", False):
            for name in cls._REQUIRED_METHODS_BY_CAPABILITY["generation"]:
                if not callable(getattr(obj, name, None)):
                    missing.add(name)
        return missing

    # Providers discovered via the entry-point group (populated lazily on first
    # use). ``_ep_provider_classes`` maps entry-point name -> provider class;
    # the two parallel dicts hold each provider's declared capabilities and
    # priority so routing can consult them without re-instantiating.
    _entry_points_loaded: bool = False
    _ep_provider_classes: Dict[str, type] = {}
    _ep_provider_capabilities: Dict[str, ProviderCapabilities] = {}
    _ep_provider_priorities: Dict[str, int] = {}

    @classmethod
    def _discover_entry_point_providers(cls) -> None:
        """Load every provider registered under the training-provider entry-point
        group and record its class, capabilities, and priority.

        Follows the same discovery pattern as the LLM provider registry
        (``kestrel_sovereign.llm_providers``). Failures loading any single
        entry point are logged and skipped so one broken package never breaks
        discovery for the rest. A built-in provider name is never shadowed by
        an entry point.
        """
        from kestrel_sovereign.entrypoints import discover_entry_point_callables

        for name, obj in discover_entry_point_callables(
            TRAINING_PROVIDER_ENTRY_POINT_GROUP
        ):
            if not isinstance(obj, type):
                logger.warning(
                    "Training provider entry point '%s' is not a class, skipping",
                    name,
                )
                continue
            if name in cls.PROVIDER_PRIORITY or name in cls.PROVIDER_ENV_VARS:
                logger.warning(
                    "Training provider entry point '%s' collides with a built-in "
                    "provider name, skipping",
                    name,
                )
                continue
            if name in cls._ep_provider_classes:
                logger.warning(
                    "Duplicate training provider entry point '%s', keeping first",
                    name,
                )
                continue

            caps = getattr(obj, "capabilities", None)
            if not isinstance(caps, ProviderCapabilities):
                caps = ProviderCapabilities()

            # Codex round-2 P1/P2 on #2445: validate the class implements
            # the methods its capability declaration promises. Otherwise a
            # plugin advertising `generation=True` without `generate_image`
            # gets selected and crashes with AttributeError at call time
            # (and a `training=True` plugin without training methods breaks
            # default routing). Drop violators at discovery so the factory
            # never sees them.
            missing = cls._required_methods_missing(obj, caps)
            if missing:
                logger.warning(
                    "Training provider entry point '%s' declares "
                    "capabilities %s but is missing required method(s): %s "
                    "— skipping (would crash at call time)",
                    name, caps, ", ".join(sorted(missing)),
                )
                continue

            cls._ep_provider_classes[name] = obj
            cls._ep_provider_capabilities[name] = caps

            priority = getattr(obj, "priority", None)
            if not isinstance(priority, int) or isinstance(priority, bool):
                priority = DEFAULT_PROVIDER_PRIORITY
            cls._ep_provider_priorities[name] = priority

    @classmethod
    def _ensure_entry_points_loaded(cls) -> None:
        """Run entry-point discovery exactly once (lazy-init)."""
        if cls._entry_points_loaded:
            return
        cls._entry_points_loaded = True
        try:
            cls._discover_entry_point_providers()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Training provider entry-point discovery failed: %s", e)

    @classmethod
    def _effective_priority(cls) -> List[str]:
        """Return all provider names ordered by effective priority.

        Built-in providers keep their PROVIDER_PRIORITY order (index * step);
        entry-point providers are interleaved by their declared ``priority``.
        Ties break by registration order (built-ins before entry points).
        """
        cls._ensure_entry_points_loaded()

        entries: List[tuple] = []
        for index, name in enumerate(cls.PROVIDER_PRIORITY):
            entries.append((index * _BUILTIN_PRIORITY_STEP, index, name))

        base = len(cls.PROVIDER_PRIORITY)
        for order, (name, priority) in enumerate(cls._ep_provider_priorities.items()):
            entries.append((priority, base + order, name))

        entries.sort(key=lambda entry: (entry[0], entry[1]))
        return [name for _, _, name in entries]

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
        if not provider:
            logger.warning(f"Provider '{name}' not available")
            return None

        # Codex round-2 P1 on #2445: a plugin's `is_available()` might raise
        # (e.g. reachability probe against a service that's down). Treat any
        # exception as "unavailable" for entry-point providers so one broken
        # plugin can't brick `list_available_providers()` or default routing.
        # Built-in providers keep the pre-fix behavior so their own bugs
        # aren't silently swallowed.
        cls._ensure_entry_points_loaded()
        is_entry_point = name in cls._ep_provider_classes
        try:
            available = provider.is_available()
        except Exception as e:
            if is_entry_point:
                logger.warning(
                    "Entry-point provider '%s' is_available() raised: %s "
                    "— treating as unavailable",
                    name, e,
                )
                return None
            raise

        if available:
            cls._instances[name] = provider
            return provider

        logger.warning(f"Provider '{name}' not available")
        return None

    @classmethod
    def get_default_provider(cls) -> Optional[TrainingProvider]:
        """
        Get the best available provider that supports training, by priority.

        Tries providers in order: vertex_ai > replicate > gcp_compute > vastai

        A provider is only selected if its capabilities declare ``training`` —
        entry-point packages may register generation-only backends, which must
        never be routed into a training flow.

        Returns:
            First available training-capable TrainingProvider or None

        Environment:
            GENERATION_PROVIDER: Force a specific provider (e.g., "vertex_ai", "runpod")
        """
        # Check for forced provider via env var
        forced_provider = os.getenv("GENERATION_PROVIDER")
        if forced_provider:
            caps = cls.get_capabilities(forced_provider)
            if caps and caps.training:
                provider = cls.get_provider(forced_provider)
                if provider:
                    logger.info(f"Using forced training provider: {forced_provider} (GENERATION_PROVIDER env)")
                    return provider
            logger.warning(f"Forced provider '{forced_provider}' not available or doesn't support training")

        for name in cls._effective_priority():
            caps = cls.get_capabilities(name)
            if not (caps and caps.training):
                continue
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
        for name in cls._effective_priority():
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
        if name in cls.PROVIDER_CAPABILITIES:
            return cls.PROVIDER_CAPABILITIES[name]
        cls._ensure_entry_points_loaded()
        return cls._ep_provider_capabilities.get(name)

    @classmethod
    def get_uncensored_provider(cls) -> Optional[TrainingProvider]:
        """
        Get first available provider with uncensored generation.

        Useful when content requires no safety filtering. The provider must
        support *both* generation and uncensored output — a training-only
        backend that happens to declare ``uncensored=True`` (e.g. an
        entry-point provider) would fail when generation code calls it, so it
        is skipped here.
        Priority follows PROVIDER_PRIORITY order.

        Returns:
            First available uncensored generation provider, or None if none available
        """
        for name in cls._effective_priority():
            caps = cls.get_capabilities(name)
            if caps and caps.generation and caps.uncensored:
                provider = cls.get_provider(name)
                if provider:
                    logger.info(f"Using uncensored provider: {name}")
                    return provider

        logger.warning("No uncensored providers available")
        return None

    @classmethod
    def get_generation_provider(cls, uncensored: bool = False) -> Optional[TrainingProvider]:
        """
        Get a provider that supports image generation.

        Args:
            uncensored: If True, only return providers with uncensored generation

        Returns:
            TrainingProvider with generation capability, or None

        Environment:
            GENERATION_PROVIDER: Force a specific provider (e.g., "vertex_ai", "runpod")
        """
        # Check for forced provider via env var
        forced_provider = os.getenv("GENERATION_PROVIDER")
        if forced_provider:
            caps = cls.get_capabilities(forced_provider)
            if caps and caps.generation:
                provider = cls.get_provider(forced_provider)
                if provider:
                    logger.info(f"Using forced generation provider: {forced_provider} (GENERATION_PROVIDER env)")
                    return provider
            logger.warning(f"Forced provider '{forced_provider}' not available or doesn't support generation")

        for name in cls._effective_priority():
            caps = cls.get_capabilities(name)
            if caps and caps.generation:
                if uncensored and not caps.uncensored:
                    continue
                provider = cls.get_provider(name)
                if provider:
                    logger.info(f"Using generation provider: {name} (uncensored={caps.uncensored})")
                    return provider

        logger.warning(f"No generation providers available (uncensored={uncensored})")
        return None

    @classmethod
    def _check_provider_available(cls, name: str) -> bool:
        """Check whether a provider is available.

        Built-in providers are cheap-checked against their declared env vars
        (avoids importing heavy adapters). Entry-point providers have no
        registered env vars, so their own ``is_available()`` is the source of
        truth — resolve them through ``get_provider`` (which instantiates,
        checks ``is_available()``, and caches on success).
        """
        cls._ensure_entry_points_loaded()
        if name in cls._ep_provider_classes:
            return cls.get_provider(name) is not None
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
        cls._ensure_entry_points_loaded()
        if name in cls._ep_provider_classes:
            try:
                return cls._ep_provider_classes[name]()
            except Exception as e:
                logger.error(f"Failed to create entry-point provider {name}: {e}")
                return None

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
        """Clear cached provider instances and re-run entry-point discovery.

        Resets the lazy entry-point discovery state as well, so a test that
        installs (or removes) a provider entry point and then calls
        ``clear_cache()`` sees the updated registration on next use.
        """
        cls._instances.clear()
        cls._entry_points_loaded = False
        cls._ep_provider_classes = {}
        cls._ep_provider_capabilities = {}
        cls._ep_provider_priorities = {}
