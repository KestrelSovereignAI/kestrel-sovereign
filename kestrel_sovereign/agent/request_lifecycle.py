"""
Request lifecycle mixin for KestrelAgent.

Extracted from kestrel_agent.py — tracks active requests and supports
cancellation (stop button functionality) for streaming responses.
"""

import logging
from typing import Optional


class RequestLifecycleMixin:
    """Mixin providing request tracking and cancellation for KestrelAgent."""

    def register_active_request(self, request_id: str) -> None:
        """Track an active request for later cancellation and cleanup."""
        if not hasattr(self, "_active_request_ids"):
            self._active_request_ids = set()
        self._active_request_ids.add(request_id)
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
        self._cancelled_requests.discard(request_id)
        if self._current_request_id == request_id:
            self._current_request_id = next(iter(active_request_ids), None) if active_request_ids else None
