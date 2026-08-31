"""
Shared Model Discovery Cache

Process-wide singleton that holds the in-memory model discovery cache.
Multiple LLMService instances (e.g., per-agent in multi-agent deployments)
share this cache to avoid redundant API discovery calls and memory duplication.

The disk cache (model_discovery_cache.json) is managed by ModelCatalogService.
This module manages the in-memory layer on top of that.
"""
import asyncio
import logging
import time
import threading
from typing import Awaitable, Callable, List, Optional

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
        self._refresh_task: Optional[asyncio.Task] = None
        self._refresh_loop: Optional[asyncio.AbstractEventLoop] = None

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

    def refresh_in_background(
        self, refresh_factory: Callable[[], Awaitable[object]]
    ) -> bool:
        """Start at most one process-wide async refresh.

        The stale catalog remains readable while the refresh runs. Returning a
        boolean lets callers log whether they started the shared work or joined
        an already-running refresh without awaiting provider network calls.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._refresh_task is not None and not self._refresh_task.done():
                owner_loop = self._refresh_loop
                # Test harnesses and embedded hosts can replace their event
                # loop while the process-wide cache survives. A task owned by
                # a closed loop can never complete, so it must not wedge model
                # refresh forever on the replacement loop.
                if owner_loop is None or not owner_loop.is_closed():
                    return False
                self._refresh_task = None
                self._refresh_loop = None
            async def _deferred_refresh() -> object:
                # Let the latency-sensitive response serialize before provider
                # discovery starts doing synchronous SDK/catalog setup.
                await asyncio.sleep(0.05)
                return await refresh_factory()

            task = loop.create_task(_deferred_refresh())
            self._refresh_task = task
            self._refresh_loop = loop

        def _finished(done: asyncio.Task) -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("Background model discovery refresh failed", exc_info=True)
            finally:
                with self._lock:
                    if self._refresh_task is done:
                        self._refresh_task = None
                        self._refresh_loop = None

        task.add_done_callback(_finished)
        return True

    async def wait_for_refresh(self) -> bool:
        """Join the process-wide refresh when it belongs to this event loop.

        Stale-while-revalidate callers return immediately, but a subsequent
        routing/cognition path may require a fresh catalog before that refresh
        completes. Joining the existing task prevents the latency-sensitive
        request from launching a duplicate full provider discovery. Cross-loop
        tasks cannot be awaited safely, so embedded multi-loop hosts fall back
        to their normal foreground discovery path.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            task = self._refresh_task
            owner_loop = self._refresh_loop
            if task is None or task.done() or owner_loop is not loop:
                return False

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Explicit cache invalidation may cancel the shared refresh. Do not
            # turn that internal cancellation into cancellation of the caller;
            # cancellation of the caller itself must still propagate.
            if task.cancelled():
                return False
            raise
        except Exception:
            # The refresh callback logs the provider failure. The caller can
            # now perform its normal foreground discovery/fallback.
            return False
        return True

    def clear(self) -> None:
        """Invalidate cache to force rediscovery."""
        with self._lock:
            self._models = None
            self._timestamp = None
            refresh_task = self._refresh_task
            refresh_loop = self._refresh_loop
            if (
                refresh_task is not None
                and not refresh_task.done()
                and (refresh_loop is None or refresh_loop.is_closed())
            ):
                self._refresh_task = None
                self._refresh_loop = None
            logger.info("Shared model cache cleared")
        # A refresh that started before explicit invalidation must not repopulate
        # the cache with the snapshot the caller meant to discard.
        if refresh_task is not None and not refresh_task.done():
            if refresh_loop is not None and not refresh_loop.is_closed():
                try:
                    refresh_loop.call_soon_threadsafe(refresh_task.cancel)
                except RuntimeError:
                    # The loop may close between is_closed() and scheduling.
                    # In that state the task cannot repopulate this cache, so
                    # detaching it is both safe and necessary for re-arming.
                    with self._lock:
                        if self._refresh_task is refresh_task:
                            self._refresh_task = None
                            self._refresh_loop = None


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
