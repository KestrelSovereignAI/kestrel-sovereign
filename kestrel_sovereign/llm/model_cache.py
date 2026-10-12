"""
Shared Model Discovery Cache

Process-wide singleton that holds the in-memory model discovery cache.
Multiple LLMService instances (e.g., per-agent in multi-agent deployments)
share this cache to avoid redundant API discovery calls and memory duplication.

The disk cache (model_discovery_cache.json) is managed by ModelCatalogService.
This module manages the in-memory layer on top of that.

The catalog and the per-vendor discovery outcomes that go with it are one
:class:`ModelCatalogSnapshot`, published whole under the cache's lock (#3197).
Before that, the outcome was recorded on whichever ``LLMService`` happened to
run discovery, so a sibling service reading the same catalog through a cache
hit reported discovery healthy while the service that ran it knew the catalog
was reduced, and a later successful refresh by a sibling could not clear the
first service's warning.
"""
import asyncio
import dataclasses
import enum
import logging
import time
import threading
from types import MappingProxyType
from typing import Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from .model_metadata import ModelInfo

logger = logging.getLogger(__name__)

# Default cache TTL: 5 minutes
DEFAULT_CACHE_TTL_SECONDS = 300


class DiscoveryStatus(str, enum.Enum):
    """What a snapshot records for one vendor's model discovery."""

    #: A catalog request completed. An empty catalog is still a completed one.
    OK = "ok"
    #: The catalog request raised.
    FAILED = "failed"
    #: No catalog request was issued: no ``base_url``, no credential, an
    #: adapter that publishes no catalog, or a publication that was not a
    #: refresh at all.
    NOT_ATTEMPTED = "not_attempted"


@dataclasses.dataclass(frozen=True)
class VendorDiscoveryOutcome:
    """One vendor's discovery outcome as recorded in a snapshot.

    ``error`` is the failure this snapshot reports for the vendor. On a
    ``FAILED`` outcome the refresh observed it. On a ``NOT_ATTEMPTED`` one it
    is the failure the previous snapshot reported, carried because nothing
    since has issued a request that could have contradicted it (#3190 r10): a
    skipped request is not a successful one. ``observed_version`` is the
    version of the snapshot whose refresh actually made the observation.
    """

    status: DiscoveryStatus
    error: Optional[str] = None
    observed_version: Optional[int] = None

    @classmethod
    def ok(cls) -> "VendorDiscoveryOutcome":
        return cls(DiscoveryStatus.OK)

    @classmethod
    def failed(cls, error: str) -> "VendorDiscoveryOutcome":
        return cls(DiscoveryStatus.FAILED, error=error)

    @classmethod
    def not_attempted(cls) -> "VendorDiscoveryOutcome":
        return cls(DiscoveryStatus.NOT_ATTEMPTED)


@dataclasses.dataclass(frozen=True)
class ModelCatalogSnapshot:
    """A model catalog and the discovery outcomes that go with it.

    Frozen, with a tuple of models and a read-only outcome map, and replaced
    whole by every publication, so a reader holding a snapshot can never see a
    catalog from one publication with outcomes from another. (The ``ModelInfo``
    records themselves are shared, as they always were.)

    A refresh publishes outcomes for exactly the vendors it covered, so a
    vendor it did not cover has no outcome at all. A publication that is not a
    refresh — a catalog restored from disk, one set without outcomes, or the
    invalidation by ``clear()`` — observed no vendor, so it carries every
    failure the previous snapshot reported, as not attempted. A failure
    therefore stands until a refresh either issues a request for its vendor or
    does not cover it.

    ``models`` is ``None`` once ``clear()`` has invalidated the catalog.
    ``timestamp`` is ``0.0`` for a catalog restored from disk, which is
    therefore always stale.
    """

    version: int
    models: Optional[Tuple[ModelInfo, ...]]
    outcomes: Mapping[str, VendorDiscoveryOutcome]
    timestamp: float

    @property
    def discovery_failures(self) -> Dict[str, str]:
        """Vendor -> the discovery failure this snapshot reports for it."""
        return {
            vendor: outcome.error
            for vendor, outcome in self.outcomes.items()
            if outcome.error is not None
        }


def _settle_outcomes(
    observed: Optional[Mapping[str, VendorDiscoveryOutcome]],
    previous: Optional[ModelCatalogSnapshot],
    version: int,
) -> Mapping[str, VendorDiscoveryOutcome]:
    """The outcomes a snapshot at ``version`` publishes.

    ``observed`` is what a refresh observed, or ``None`` for a publication that
    is not a refresh. Each vendor's record is replaced by what the refresh
    observed, stamped with ``version``. A vendor it did not attempt keeps the
    failure the previous snapshot reported for it and nothing else; a vendor it
    did not cover at all is not carried. A publication that is not a refresh
    attempted nothing, so every failure the previous snapshot reported stands.
    """
    prior_outcomes: Mapping[str, VendorDiscoveryOutcome] = (
        previous.outcomes if previous is not None else {}
    )
    if observed is None:
        observed = {
            vendor: VendorDiscoveryOutcome.not_attempted()
            for vendor, outcome in prior_outcomes.items()
            if outcome.error is not None
        }
    settled: Dict[str, VendorDiscoveryOutcome] = {}
    for vendor, outcome in observed.items():
        if outcome.status is DiscoveryStatus.NOT_ATTEMPTED:
            prior = prior_outcomes.get(vendor)
            if prior is not None and prior.error is not None:
                outcome = VendorDiscoveryOutcome(
                    DiscoveryStatus.NOT_ATTEMPTED,
                    error=prior.error,
                    observed_version=prior.observed_version,
                )
            else:
                outcome = VendorDiscoveryOutcome.not_attempted()
        else:
            outcome = dataclasses.replace(outcome, observed_version=version)
        settled[vendor] = outcome
    return MappingProxyType(settled)


class SharedModelCache:
    """Process-wide model discovery cache shared across LLMService instances.

    Thread-safe singleton holding the current :class:`ModelCatalogSnapshot`.
    All LLMService instances read from and publish to this shared cache,
    eliminating duplicate API discovery calls in multi-agent scenarios, and
    every one of them reads the same discovery outcomes for the same catalog.
    """

    def __init__(self, cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS):
        self._snapshot: Optional[ModelCatalogSnapshot] = None
        # Monotonic for the cache's lifetime, so a version names one
        # publication.
        self._version = 0
        self._ttl = cache_ttl
        self._lock = threading.Lock()
        self._refresh_task: Optional[asyncio.Task] = None
        self._refresh_loop: Optional[asyncio.AbstractEventLoop] = None

    def snapshot(self) -> Optional[ModelCatalogSnapshot]:
        """Return the current snapshot regardless of freshness, or None if
        nothing has been published."""
        with self._lock:
            return self._snapshot

    def get(self) -> Optional[List[ModelInfo]]:
        """Return cached models if fresh, None if stale or empty."""
        snapshot = self.snapshot()
        if snapshot is None or snapshot.models is None:
            return None
        if time.time() - snapshot.timestamp >= self._ttl:
            return None
        return list(snapshot.models)

    def get_any(self) -> Optional[List[ModelInfo]]:
        """Return cached models regardless of freshness (for pre-discovery fallback).

        Used by _load_from_disk_cache to check if another instance already
        populated the cache.
        """
        snapshot = self.snapshot()
        if snapshot is None or snapshot.models is None:
            return None
        return list(snapshot.models)

    def set(
        self,
        models: List[ModelInfo],
        outcomes: Optional[Mapping[str, VendorDiscoveryOutcome]] = None,
    ) -> None:
        """Publish newly discovered models with the outcomes that produced them.

        ``outcomes`` maps each vendor the refresh discovered to what it
        observed. Omitted, the catalog was not produced by a refresh, and the
        snapshot carries the previous one's failures.
        """
        with self._lock:
            self._publish(models, outcomes, time.time())
            logger.debug(
                "Shared model cache updated: %d models (snapshot %d)",
                len(models), self._version,
            )

    def set_stale(self, models: List[ModelInfo]) -> None:
        """Populate cache with stale data (e.g., from disk cache).

        Sets timestamp to 0 so the next discover_all_models() call
        will refresh via API, but models are available immediately. The disk
        cache holds no discovery outcomes, so this snapshot carries the
        previous one's failures.
        """
        with self._lock:
            if self._snapshot is not None and self._snapshot.models is not None:
                return  # Don't overwrite fresh data with stale
            self._publish(models, None, 0.0)
            logger.debug(f"Shared model cache pre-populated (stale): {len(models)} models")

    def _publish(
        self,
        models: Optional[List[ModelInfo]],
        outcomes: Optional[Mapping[str, VendorDiscoveryOutcome]],
        timestamp: float,
    ) -> None:
        """Replace the snapshot. The caller holds ``self._lock``."""
        self._version += 1
        self._snapshot = ModelCatalogSnapshot(
            version=self._version,
            models=None if models is None else tuple(models),
            outcomes=_settle_outcomes(outcomes, self._snapshot, self._version),
            timestamp=timestamp,
        )

    def has_data(self) -> bool:
        """Check if any data exists (fresh or stale)."""
        with self._lock:
            return self._snapshot is not None and self._snapshot.models is not None

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
        """Invalidate cache to force rediscovery.

        Publishes a snapshot with no catalog. It carries the previous one's
        failures: invalidating the catalog observes nothing about a vendor, and
        it runs on unrelated events such as an Ollama pull (#3190 r10).
        """
        with self._lock:
            if self._snapshot is not None:
                self._publish(None, None, 0.0)
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
