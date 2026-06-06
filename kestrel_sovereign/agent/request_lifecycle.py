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

import logging
import time
from typing import Dict, List, Optional


class RequestLifecycleMixin:
    """Mixin providing request tracking and cancellation for KestrelAgent."""

    def register_active_request(self, request_id: str) -> None:
        """Track an active request for later cancellation and cleanup."""
        if not hasattr(self, "_active_request_ids"):
            self._active_request_ids = set()
        self._active_request_ids.add(request_id)
        # Stamp the registration time (monotonic) so abandoned request
        # ids can be aged out as stale (#1558).
        if not hasattr(self, "_active_request_started_at"):
            self._active_request_started_at = {}
        self._active_request_started_at[request_id] = time.monotonic()
        # Preserve the legacy "current request" fallback for callers that
        # do not yet pass an explicit request ID.
        self._current_request_id = request_id

    def cancel_current_request(self, request_id: Optional[str] = None) -> bool:
        """
        Cancel the current streaming request.

        Returns:
            True if a request was cancelled, False if no request was active.
        """
        active_request_ids = getattr(self, "_active_request_ids", set())
        target_request_id = request_id or self._current_request_id
        if target_request_id and (
            target_request_id in active_request_ids
            or target_request_id == self._current_request_id
        ):
            self._cancelled_requests.add(target_request_id)
            logging.info(f"Cancelled request: {target_request_id}")
            return True
        return False

    def is_request_cancelled(self, request_id: Optional[str] = None) -> bool:
        """Check if a request has been cancelled."""
        rid = request_id or self._current_request_id
        return rid in self._cancelled_requests if rid else False

    def _cleanup_cancelled_request(self, request_id: str):
        """Remove a request from the cancelled set after it's been handled."""
        active_request_ids = getattr(self, "_active_request_ids", None)
        if active_request_ids is not None:
            active_request_ids.discard(request_id)
        started = getattr(self, "_active_request_started_at", None)
        if started is not None:
            started.pop(request_id, None)
        self._cancelled_requests.discard(request_id)
        if self._current_request_id == request_id:
            self._current_request_id = next(iter(active_request_ids), None) if active_request_ids else None

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
            active.discard(rid)
            started.pop(rid, None)
            cancelled = getattr(self, "_cancelled_requests", None)
            if cancelled is not None:
                cancelled.discard(rid)
        if stale and getattr(self, "_current_request_id", None) in stale:
            self._current_request_id = (
                next(iter(active), None) if active else None
            )
        return stale
