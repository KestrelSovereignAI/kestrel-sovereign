"""
Shared Model Discovery Cache

Process-wide singleton that holds the in-memory model discovery cache.
Multiple LLMService instances (e.g., per-agent in multi-agent deployments)
share this cache to avoid redundant API discovery calls and memory duplication.

The disk cache (model_discovery_cache.json) is managed by ModelCatalogService.
This module manages the in-memory layer on top of that.
"""
import logging
import time
import threading
from typing import List, Optional

from .model_metadata import ModelInfo

logger = logging.getLogger(__name__)

# Default cache TTL: 5 minutes
DEFAULT_CACHE_TTL_SECONDS = 300


class SharedModelCache:
    """Process-wide model discovery cache shared across LLMService instances.

    Thread-safe singleton that stores discovered ModelInfo objects in memory.
    All LLMService instances read from and write to this shared cache,
    eliminating duplicate API discovery calls in multi-agent scenarios.
    """

    def __init__(self, cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS):
        self._models: Optional[List[ModelInfo]] = None
        self._timestamp: Optional[float] = None
        self._ttl = cache_ttl
        self._lock = threading.Lock()

    def get(self) -> Optional[List[ModelInfo]]:
        """Return cached models if fresh, None if stale or empty."""
        with self._lock:
            if self._models is None or self._timestamp is None:
                return None
            age = time.time() - self._timestamp
            if age >= self._ttl:
                return None
            return self._models

    def get_any(self) -> Optional[List[ModelInfo]]:
        """Return cached models regardless of freshness (for pre-discovery fallback).

        Used by _load_from_disk_cache to check if another instance already
        populated the cache.
        """
        with self._lock:
            return self._models

    def set(self, models: List[ModelInfo]) -> None:
        """Update cache with newly discovered models."""
        with self._lock:
            self._models = models
            self._timestamp = time.time()
            logger.debug(f"Shared model cache updated: {len(models)} models")

    def set_stale(self, models: List[ModelInfo]) -> None:
        """Populate cache with stale data (e.g., from disk cache).

        Sets timestamp to 0 so the next discover_all_models() call
        will refresh via API, but models are available immediately.
        """
        with self._lock:
            if self._models is not None:
                return  # Don't overwrite fresh data with stale
            self._models = models
            self._timestamp = 0.0
            logger.debug(f"Shared model cache pre-populated (stale): {len(models)} models")

    def has_data(self) -> bool:
        """Check if any data exists (fresh or stale)."""
        with self._lock:
            return self._models is not None

    def clear(self) -> None:
        """Invalidate cache to force rediscovery."""
        with self._lock:
            self._models = None
            self._timestamp = None
            logger.info("Shared model cache cleared")


# Module-level singleton
_shared_cache: Optional[SharedModelCache] = None
_singleton_lock = threading.Lock()


def get_shared_model_cache() -> SharedModelCache:
    """Get or create the process-wide shared model cache."""
    global _shared_cache
    if _shared_cache is None:
        with _singleton_lock:
            if _shared_cache is None:
                _shared_cache = SharedModelCache()
    return _shared_cache
