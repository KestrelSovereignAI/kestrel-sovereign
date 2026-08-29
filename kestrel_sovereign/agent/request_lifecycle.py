"""
Request lifecycle mixin for KestrelAgent.

Extracted from kestrel_agent.py — tracks active requests and supports
cancellation (stop button functionality) for streaming responses.

Each active request is stamped with a monotonic registration time so
the restart coordinator can distinguish a genuinely in-flight stream
from a stale marker whose endpoint cleanup never ran (client
disconnect, crashed generator). Without that, an abandoned request id
permanently blocks ``idle_agents_only`` restarts (#1558).
"""

import asyncio
import logging
import time
from contextvars import ContextVar
from enum import Enum
from typing import Dict, List, Optional

from kestrel_sovereign.agent.invocation import invocation_log_correlation


_PENDING_REQUEST_CANCELLATION_TTL_SECONDS = 30.0
_PENDING_REQUEST_CANCELLATION_LIMIT = 1024


class RequestCompletionDisposition(str, Enum):
    """What released a cooperative Stop lifecycle waiter."""

    COMPLETED = "completed"
    ABANDONED = "abandoned"


_current_request_generation: ContextVar[tuple[int, str, int] | None] = ContextVar(
    "kestrel_current_request_generation",
    default=None,
)


class RequestLifecycleMixin:
    """Mixin providing request tracking and cancellation for KestrelAgent."""

    def register_active_request(self, request_id: str) -> int:
        """Track an active delivery and bind its generation to this task."""
        if not hasattr(self, "_active_request_ids"):
            self._active_request_ids = set()
        if not isinstance(getattr(self, "_cancelled_requests", None), set):
            self._cancelled_requests = set()
        if not isinstance(
            getattr(self, "_cancelled_request_generations", None),
            set,
        ):
            self._cancelled_request_generations = set()
        if not isinstance(getattr(self, "_active_request_counts", None), dict):
            self._active_request_counts = {}
        counts = self._active_request_counts
        was_inactive = counts.get(request_id, 0) == 0
        generations = getattr(self, "_active_request_generations", None)
        if not isinstance(generations, dict):
            generations = {}
            self._active_request_generations = generations
        generation = generations.get(request_id)
        if was_inactive or not isinstance(generation, int):
            next_generation = getattr(self, "_next_request_generation", 0)
            if not isinstance(next_generation, int):
                next_generation = 0
            generation = next_generation + 1
            self._next_request_generation = generation
            generations[request_id] = generation
        counts[request_id] = counts.get(request_id, 0) + 1
        self._active_request_ids.add(request_id)
        # Stamp the registration time (monotonic) so abandoned request
        # ids can be aged out as stale (#1558).
        if not hasattr(self, "_active_request_started_at"):
            self._active_request_started_at = {}
        if was_inactive:
            self._active_request_started_at[request_id] = time.monotonic()
        # Preserve the legacy "current request" fallback for callers that
        # do not yet pass an explicit request ID.
        self._current_request_id = request_id
        _current_request_generation.set((id(self), request_id, generation))
        self._consume_pending_request_cancellation(request_id, generation)
        return generation

    def _prune_pending_request_cancellations(self, now: float) -> None:
        pending = getattr(self, "_pending_request_cancellations", None)
        if not isinstance(pending, dict):
            self._pending_request_cancellations = {}
            return
        cutoff = now - _PENDING_REQUEST_CANCELLATION_TTL_SECONDS
        for request_id, reserved_at in tuple(pending.items()):
            if not isinstance(reserved_at, (int, float)) or reserved_at <= cutoff:
                pending.pop(request_id, None)

    def reserve_request_cancellation(self, request_id: str) -> None:
        """Fence an exact request ID that may still be entering registration.

        Stop and request registration are separate HTTP requests.  A Stop can
        therefore arrive after the client dispatched a request but before its
        handler has registered the lifecycle.  This short-lived tombstone
        fences every same-ID redelivery in the race window; a transport retry
        cannot resurrect work after an exact Stop was acknowledged.
        """

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("pending request cancellation requires an exact id")
        now = time.monotonic()
        self._prune_pending_request_cancellations(now)
        pending = self._pending_request_cancellations
        pending[request_id] = now
        while len(pending) > _PENDING_REQUEST_CANCELLATION_LIMIT:
            oldest = min(pending, key=pending.__getitem__)
            pending.pop(oldest, None)

    def _consume_pending_request_cancellation(
        self,
        request_id: str,
        generation: int,
    ) -> bool:
        now = time.monotonic()
        self._prune_pending_request_cancellations(now)
        pending = self._pending_request_cancellations
        if pending.get(request_id) is None:
            return False
        cancelled_generations = getattr(
            self,
            "_cancelled_request_generations",
            None,
        )
        if not isinstance(cancelled_generations, set):
            cancelled_generations = set()
            self._cancelled_request_generations = cancelled_generations
        cancelled_generations.add((request_id, generation))
        cancelled_requests = getattr(self, "_cancelled_requests", None)
        if not isinstance(cancelled_requests, set):
            cancelled_requests = set()
            self._cancelled_requests = cancelled_requests
        cancelled_requests.add(request_id)
        logging.info(
            "Consumed pre-registration cancellation for request lifecycle: %s",
            invocation_log_correlation(request_id),
        )
        return True

    def bind_request_operation(
        self,
        request_id: str,
        operation: asyncio.Task,
    ) -> None:
        """Bind cancellable turn work to the current request generation."""

        if not isinstance(operation, asyncio.Task):
            raise TypeError("request operation must be an asyncio task")
        generation = self._request_generation_for_current_task(request_id)
        if generation is None:
            raise RuntimeError("request operation requires an active generation")
        operations = getattr(self, "_request_operation_tasks", None)
        if not isinstance(operations, dict):
            operations = {}
            self._request_operation_tasks = operations
        key = (request_id, generation)
        owners = operations.setdefault(key, set())
        owners.add(operation)

        def release(completed: asyncio.Task) -> None:
            current = operations.get(key)
            if not isinstance(current, set):
                return
            current.discard(completed)
            if not current:
                operations.pop(key, None)

        operation.add_done_callback(release)
        if self.is_request_cancelled(request_id):
            operation.cancel()

    def _request_generation_for_current_task(
        self,
        request_id: str,
    ) -> int | None:
        """Resolve this task's delivery generation, then the active fallback."""

        bound = _current_request_generation.get()
        if (
            bound is not None
            and bound[0] == id(self)
            and bound[1] == request_id
        ):
            return bound[2]
        generations = getattr(self, "_active_request_generations", None)
        if not isinstance(generations, dict):
            return None
        generation = generations.get(request_id)
        return generation if isinstance(generation, int) else None

    def _request_generation_for_cleanup(self, request_id: str) -> int | None:
        """Resolve the generation whose endpoint cleanup is now running.

        A normally registered delivery carries an exact task-local generation.
        Legacy/foreign deliveries do not.  Once one of those is pruned from the
        active projection, retain its generations in cleanup order so its
        eventual ``finally`` cannot accidentally clean a fresh redelivery (or
        fail to clean anything at all).
        """

        bound = _current_request_generation.get()
        if (
            bound is not None
            and bound[0] == id(self)
            and bound[1] == request_id
        ):
            return bound[2]
        cleanup_generations = getattr(
            self,
            "_legacy_request_cleanup_generations",
            None,
        )
        queued = (
            cleanup_generations.get(request_id)
            if isinstance(cleanup_generations, dict)
            else None
        )
        if isinstance(queued, list) and queued:
            return queued[0]
        generations = getattr(self, "_active_request_generations", None)
        if not isinstance(generations, dict):
            return None
        generation = generations.get(request_id)
        return generation if isinstance(generation, int) else None

    def _remember_pruned_cleanup_generation(
        self,
        request_id: str,
        generation: int,
    ) -> None:
        cleanup_generations = getattr(
            self,
            "_legacy_request_cleanup_generations",
            None,
        )
        if not isinstance(cleanup_generations, dict):
            cleanup_generations = {}
            self._legacy_request_cleanup_generations = cleanup_generations
        queued = cleanup_generations.setdefault(request_id, [])
        if generation not in queued:
            queued.append(generation)

    def _forget_pruned_cleanup_generation(
        self,
        request_id: str,
        generation: int,
    ) -> None:
        cleanup_generations = getattr(
            self,
            "_legacy_request_cleanup_generations",
            None,
        )
        if not isinstance(cleanup_generations, dict):
            return
        queued = cleanup_generations.get(request_id)
        if not isinstance(queued, list):
            return
        try:
            queued.remove(generation)
        except ValueError:
            return
        if not queued:
            cleanup_generations.pop(request_id, None)

    def _abandoned_generations(self, request_id: str) -> set[int]:
        tombstones = getattr(self, "_abandoned_request_generations", None)
        if not isinstance(tombstones, dict):
            return set()
        generations = tombstones.get(request_id)
        return set(generations) if isinstance(generations, set) else set()

    def cancel_current_request(self, request_id: Optional[str] = None) -> bool:
        """
        Cancel the current streaming request.

        Returns:
            True if a request was cancelled, False if no request was active.
        """
        active_request_ids = getattr(self, "_active_request_ids", None)
        if not isinstance(active_request_ids, set):
            active_request_ids = set()
            self._active_request_ids = active_request_ids
        target_request_id = request_id or self._current_request_id
        if not target_request_id:
            return False
        # Install the exact-ID fence even for a currently active generation.
        # Same-ID transport redeliveries must not resume after Stop returns.
        self.reserve_request_cancellation(target_request_id)

        generations: set[int] = self._abandoned_generations(target_request_id)
        if (
            target_request_id in active_request_ids
            or target_request_id == self._current_request_id
        ):
            active_generation = self._request_generation_for_current_task(
                target_request_id
            )
            if active_generation is None:
                active_generations = getattr(
                    self,
                    "_active_request_generations",
                    None,
                )
                if not isinstance(active_generations, dict):
                    active_generations = {}
                    self._active_request_generations = active_generations
                next_generation = getattr(self, "_next_request_generation", 0)
                if not isinstance(next_generation, int):
                    next_generation = 0
                active_generation = next_generation + 1
                self._next_request_generation = active_generation
                active_generations[target_request_id] = active_generation
            # A legacy caller may expose liveness only through
            # ``_current_request_id``. The synthesized generation is still
            # executing until that caller's eventual cleanup; project it as
            # active so Stop cannot acknowledge completion immediately.
            active_request_ids.add(target_request_id)
            generations.add(active_generation)
        if not generations:
            return False

        cancelled_generations = getattr(
            self,
            "_cancelled_request_generations",
            None,
        )
        if not isinstance(cancelled_generations, set):
            cancelled_generations = set()
            self._cancelled_request_generations = cancelled_generations
        cancelled_generations.update(
            (target_request_id, generation) for generation in generations
        )
        self._cancelled_requests.add(target_request_id)
        operations = getattr(self, "_request_operation_tasks", None)
        if isinstance(operations, dict):
            for generation in generations:
                for operation in tuple(
                    operations.get((target_request_id, generation), set())
                ):
                    operation.cancel()
        logging.info(
            "Cancelled request lifecycle: %s",
            invocation_log_correlation(target_request_id),
        )
        return True

    def is_request_cancelled(self, request_id: Optional[str] = None) -> bool:
        """Check if a request has been cancelled."""
        rid = request_id or self._current_request_id
        if not rid:
            return False
        generation = self._request_generation_for_current_task(rid)
        cancelled_generations = getattr(
            self,
            "_cancelled_request_generations",
            None,
        )
        if isinstance(cancelled_generations, set) and generation is not None:
            return (rid, generation) in cancelled_generations
        # Compatibility for legacy/test registrations which never received a
        # generation. Once a generation exists, a bare old ID is deliberately
        # insufficient to poison a fresh delivery.
        return generation is None and rid in self._cancelled_requests

    async def wait_for_request_completion(
        self,
        request_id: Optional[str] = None,
    ) -> RequestCompletionDisposition:
        """Wait until every cancelled generation has left the live lifecycle.

        Marking ``_cancelled_requests`` is only a cooperative request to stop.
        A caller that needs a Stop acknowledgement must wait for endpoint
        cleanup before it may report success or start replacement work. A
        request ID can name a pruned old delivery and a fresh redelivery at the
        same time, so waiting only for the active projection is insufficient.
        The waiters for all snapshotted cancelled generations are installed
        before yielding, which closes the check-then-sleep cleanup race.
        """
        target_request_id = request_id or self._current_request_id
        if target_request_id is None:
            return RequestCompletionDisposition.COMPLETED
        cancelled = getattr(self, "_cancelled_request_generations", None)
        generations = {
            generation
            for rid, generation in (
                cancelled if isinstance(cancelled, set) else set()
            )
            if rid == target_request_id
        }
        abandoned = self._abandoned_generations(target_request_id)
        if not generations:
            return (
                RequestCompletionDisposition.ABANDONED
                if abandoned
                else RequestCompletionDisposition.COMPLETED
            )

        active_request_ids = getattr(self, "_active_request_ids", set())
        active_generations = getattr(self, "_active_request_generations", None)
        active_generation = (
            active_generations.get(target_request_id)
            if isinstance(active_generations, dict)
            else None
        )
        abandoned_counts = getattr(self, "_abandoned_request_counts", None)
        waiters = getattr(self, "_request_completion_events", None)
        if not isinstance(waiters, dict):
            waiters = {}
            self._request_completion_events = waiters
        completions: list[asyncio.Future[object]] = []
        terminally_abandoned = False
        for generation in sorted(generations):
            abandoned_key = (target_request_id, generation)
            is_pruned_and_running = (
                isinstance(abandoned_counts, dict)
                and abandoned_counts.get(abandoned_key, 0) > 0
            )
            is_active = (
                generation == active_generation
                and target_request_id in active_request_ids
            )
            if generation in abandoned and not is_pruned_and_running:
                terminally_abandoned = True
                continue
            if not is_active and not is_pruned_and_running:
                # The generation completed between cancellation and this
                # synchronous snapshot. Its cleanup removed execution state.
                continue
            completion = waiters.get(abandoned_key)
            if completion is None:
                completion = asyncio.get_running_loop().create_future()
                waiters[abandoned_key] = completion
            completions.append(completion)

        dispositions = (
            await asyncio.gather(
                *(asyncio.shield(completion) for completion in completions)
            )
            if completions
            else ()
        )
        if terminally_abandoned or any(
            disposition is RequestCompletionDisposition.ABANDONED
            for disposition in dispositions
        ):
            return RequestCompletionDisposition.ABANDONED
        return RequestCompletionDisposition.COMPLETED

    def _resolve_request_completion(
        self,
        request_id: str,
        disposition: RequestCompletionDisposition = (
            RequestCompletionDisposition.COMPLETED
        ),
        *,
        generation: int | None = None,
    ) -> None:
        """Terminally release and forget waiters for one request lifecycle."""

        waiters = getattr(self, "_request_completion_events", None)
        if not isinstance(waiters, dict):
            return
        generation = (
            self._request_generation_for_current_task(request_id)
            if generation is None
            else generation
        )
        if generation is None:
            return
        completion = waiters.pop((request_id, generation), None)
        if completion is not None and not completion.done():
            completion.set_result(disposition)

    def _cleanup_cancelled_request(
        self,
        request_id: str,
        *,
        disposition: RequestCompletionDisposition = (
            RequestCompletionDisposition.COMPLETED
        ),
    ) -> None:
        """Release one delivery after completed or failed nested cleanup."""

        if not isinstance(disposition, RequestCompletionDisposition):
            raise TypeError("request completion disposition must be typed")
        active_request_ids = getattr(self, "_active_request_ids", None)
        counts = getattr(self, "_active_request_counts", None)
        generation = self._request_generation_for_cleanup(request_id)
        active_generations = getattr(self, "_active_request_generations", None)
        active_generation = (
            active_generations.get(request_id)
            if isinstance(active_generations, dict)
            else None
        )
        tombstones = getattr(self, "_abandoned_request_generations", None)
        if (
            generation is not None
            and disposition is RequestCompletionDisposition.ABANDONED
        ):
            if not isinstance(tombstones, dict):
                tombstones = {}
                self._abandoned_request_generations = tombstones
            tombstones.setdefault(request_id, set()).add(generation)
        generation_is_abandoned = (
            generation is not None
            and isinstance(tombstones, dict)
            and generation in tombstones.get(request_id, set())
        )
        effective_disposition = (
            RequestCompletionDisposition.ABANDONED
            if generation_is_abandoned
            else disposition
        )

        # Pruning removes an aged generation from the active projection so a
        # redelivery of the same request ID receives a fresh generation.  Its
        # still-running delivery count lives here until every old endpoint
        # finally exits; no one old delivery may clear the shared Stop marker.
        abandoned_counts = getattr(self, "_abandoned_request_counts", None)
        abandoned_key = (request_id, generation)
        if (
            generation is not None
            and isinstance(abandoned_counts, dict)
            and abandoned_key in abandoned_counts
        ):
            abandoned_dispositions = getattr(
                self,
                "_abandoned_request_dispositions",
                None,
            )
            if not isinstance(abandoned_dispositions, dict):
                abandoned_dispositions = {}
                self._abandoned_request_dispositions = abandoned_dispositions
            prior_disposition = abandoned_dispositions.get(
                abandoned_key,
                RequestCompletionDisposition.COMPLETED,
            )
            if disposition is RequestCompletionDisposition.ABANDONED:
                abandoned_dispositions[abandoned_key] = disposition
            else:
                abandoned_dispositions.setdefault(
                    abandoned_key,
                    prior_disposition,
                )
            remaining = abandoned_counts[abandoned_key]
            if remaining > 1:
                abandoned_counts[abandoned_key] = remaining - 1
                return
            abandoned_counts.pop(abandoned_key, None)
            self._forget_pruned_cleanup_generation(request_id, generation)
            final_disposition = abandoned_dispositions.pop(
                abandoned_key,
                disposition,
            )
            if isinstance(tombstones, dict):
                abandoned = tombstones.get(request_id)
                if isinstance(abandoned, set):
                    if (
                        final_disposition
                        is RequestCompletionDisposition.COMPLETED
                    ):
                        abandoned.discard(generation)
                    if not abandoned:
                        tombstones.pop(request_id, None)
            self._release_cancelled_generation(request_id, generation)
            self._resolve_request_completion(
                request_id,
                final_disposition,
                generation=generation,
            )
            return

        cleans_active_generation = (
            generation is not None and generation == active_generation
        ) or (generation is None and active_generation is None)
        if (
            cleans_active_generation
            and isinstance(counts, dict)
            and counts.get(request_id, 0) > 1
        ):
            counts[request_id] -= 1
            return
        if cleans_active_generation:
            if isinstance(counts, dict):
                counts.pop(request_id, None)
            if active_request_ids is not None:
                active_request_ids.discard(request_id)
            if isinstance(active_generations, dict):
                active_generations.pop(request_id, None)
        started = getattr(self, "_active_request_started_at", None)
        if cleans_active_generation and started is not None:
            started.pop(request_id, None)
        if generation is not None:
            if (
                effective_disposition is RequestCompletionDisposition.COMPLETED
                and isinstance(tombstones, dict)
            ):
                abandoned = tombstones.get(request_id)
                if isinstance(abandoned, set):
                    abandoned.discard(generation)
                    if not abandoned:
                        tombstones.pop(request_id, None)
        self._release_cancelled_generation(request_id, generation)
        if cleans_active_generation and self._current_request_id == request_id:
            self._current_request_id = (
                next(iter(active_request_ids), None)
                if active_request_ids
                else None
            )
        if cleans_active_generation:
            self._resolve_request_completion(
                request_id,
                effective_disposition,
                generation=generation,
            )

    def _release_cancelled_generation(
        self,
        request_id: str,
        generation: int | None,
    ) -> None:
        """Forget one completed generation without clearing live siblings."""

        cancelled_generations = getattr(
            self,
            "_cancelled_request_generations",
            None,
        )
        if isinstance(cancelled_generations, set) and generation is not None:
            cancelled_generations.discard((request_id, generation))
            if not any(rid == request_id for rid, _ in cancelled_generations):
                self._cancelled_requests.discard(request_id)
        else:
            self._cancelled_requests.discard(request_id)

    def active_request_ages(self) -> Dict[str, float]:
        """Return ``{request_id: age_seconds}`` for each active request.

        Ages are measured from ``register_active_request``. Request ids
        without a recorded registration time (foreign/legacy
        registrations) are omitted. Used for restart-deferral
        observability (#1558).
        """
        active = getattr(self, "_active_request_ids", None)
        started = getattr(self, "_active_request_started_at", None)
        if not active or not started:
            return {}
        now = time.monotonic()
        return {
            rid: max(0.0, now - ts)
            for rid, ts in started.items()
            if rid in active
        }

    def prune_stale_active_requests(self, max_age_seconds: float) -> List[str]:
        """Drop active request ids older than ``max_age_seconds``.

        A streaming request that finishes or is abandoned should be
        cleared by the endpoint's ``finally`` (``_cleanup_cancelled_request``).
        A client disconnect or crashed generator can leave a request id
        registered indefinitely, which permanently blocks
        ``idle_agents_only`` restarts (#1558). This sweeper removes such
        abandoned markers and returns the request ids it pruned.

        Request ids with no recorded registration time get stamped
        ``now`` instead of being pruned blind — the staleness clock
        starts on first observation rather than removing an id we can't
        date.
        """
        active = getattr(self, "_active_request_ids", None)
        if not active:
            return []
        if not hasattr(self, "_active_request_started_at"):
            self._active_request_started_at = {}
        started = self._active_request_started_at
        now = time.monotonic()
        stale: List[str] = []
        for rid in list(active):
            ts = started.get(rid)
            if ts is None:
                # Unknown registration time — start the clock now.
                started[rid] = now
                continue
            if now - ts >= max_age_seconds:
                stale.append(rid)
        for rid in stale:
            generations = getattr(self, "_active_request_generations", None)
            generation = (
                generations.get(rid)
                if isinstance(generations, dict)
                else None
            )
            if not isinstance(generation, int):
                next_generation = getattr(self, "_next_request_generation", 0)
                if not isinstance(next_generation, int):
                    next_generation = 0
                generation = next_generation + 1
                self._next_request_generation = generation
            tombstones = getattr(self, "_abandoned_request_generations", None)
            if not isinstance(tombstones, dict):
                tombstones = {}
                self._abandoned_request_generations = tombstones
            tombstones.setdefault(rid, set()).add(generation)
            counts = getattr(self, "_active_request_counts", None)
            delivery_count = (
                counts.get(rid, 1)
                if isinstance(counts, dict)
                else 1
            )
            abandoned_counts = getattr(self, "_abandoned_request_counts", None)
            if not isinstance(abandoned_counts, dict):
                abandoned_counts = {}
                self._abandoned_request_counts = abandoned_counts
            abandoned_counts[(rid, generation)] = max(1, delivery_count)
            self._remember_pruned_cleanup_generation(rid, generation)
            active.discard(rid)
            if isinstance(counts, dict):
                counts.pop(rid, None)
            if isinstance(generations, dict):
                generations.pop(rid, None)
            started.pop(rid, None)
            # If Stop already marked this genuinely live request, retain the
            # cooperative marker until the real endpoint finally runs. Age is
            # evidence that bookkeeping may be abandoned, never evidence that
            # execution completed or permission to let it continue.
            self._resolve_request_completion(
                rid,
                RequestCompletionDisposition.ABANDONED,
                generation=generation,
            )
        if stale and getattr(self, "_current_request_id", None) in stale:
            self._current_request_id = (
                next(iter(active), None) if active else None
            )
        return stale
