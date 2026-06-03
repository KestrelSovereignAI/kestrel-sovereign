"""Restart Coordinator Feature (#1512).

Durable agent-side surface for requesting a safe host restart. An
agent calls ``request_restart`` to file a row in ``restart_requests``;
the periodic ``restart_coordinator`` cron task evaluates safety, marks
the row as ``executing``, and spawns a detached subprocess that
outlives the current Kestrel host so the restart can land cleanly.

On the next agent boot, ``RestartCoordinatorFeature.initialize`` sweeps
the table for any rows in ``executing`` state filed by this agent,
marks them ``completed``, and emits a ``restart.completed`` COGNITION
signal so the requester wakes and can verify post-restart runtime
state.
"""

from .feature import RestartCoordinatorFeature

__all__ = ["RestartCoordinatorFeature"]
