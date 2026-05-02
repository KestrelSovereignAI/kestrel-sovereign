"""
Event manager mixin for KestrelAgent.

Extracted from kestrel_agent.py — provides SSE event emission,
listener management, and background task notification queuing.
"""

import logging
from typing import Any, Dict, List


class EventManagerMixin:
    """Mixin providing event/notification methods for KestrelAgent."""

    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered listeners (for SSE notifications).

        Args:
            event_type: Type of event (e.g., 'approval_request')
            data: Event data to send
        """
        for listener in self._event_listeners:
            try:
                await listener(event_type, data)
            except (TypeError, AttributeError, ConnectionError) as e:
                logging.warning(f"Failed to emit event to listener: {e}")
            except Exception as e:
                logging.warning(f"Failed to emit event to listener: {e}", exc_info=True)

    def add_event_listener(self, listener) -> None:
        """Add an event listener for SSE notifications."""
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener) -> None:
        """Remove an event listener."""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def _on_background_task_complete(self, task) -> None:
        """
        Callback invoked when a background task completes.

        Two outputs (parallel, neither replaces the other):

        1. SSE notification \u2014 appended to `_pending_task_notifications`
           for the next chat response. Browser-facing surface, unchanged
           since before #889.
        2. Cognition signal (Phase 5 of #889) \u2014 enqueued via the
           dispatcher so the bird wakes up and decides what to do with
           the result. Carries the causation chain from
           `task.metadata["causation_chain"]` so A\u2192B\u2192A ping-pong loops
           are caught by Phase 1's cycle detection.

        Called by TaskManager when tasks reach terminal states
        (COMPLETED, FAILED, CANCELED).
        """
        from kestrel_sovereign.a2a.types import TaskState

        state = task.status.state
        task_id = task.id

        # ---- 1. SSE notification (legacy path; unchanged) ----------------

        # Get task description from metadata
        agent_id = task.metadata.get("agent_id", "unknown") if task.metadata else "unknown"
        skill_id = task.metadata.get("skill", "task") if task.metadata else "task"

        # Format notification based on state
        if state == TaskState.COMPLETED:
            msg = f"\u2705 Background task completed: {agent_id}/{skill_id} (task: {task_id[:8]})"
        elif state == TaskState.FAILED:
            error_msg = ""
            if task.status.message and task.status.message.parts:
                for part in task.status.message.parts:
                    if hasattr(part, 'text'):
                        error_msg = f": {part.text}"
                        break
            msg = f"\u274c Background task failed: {agent_id}/{skill_id}{error_msg} (task: {task_id[:8]})"
        elif state == TaskState.CANCELED:
            msg = f"\u26a0\ufe0f Background task canceled: {agent_id}/{skill_id} (task: {task_id[:8]})"
        else:
            return  # Don't notify for non-terminal states

        self._pending_task_notifications.append(msg)
        logging.info(f"Queued task notification: {msg}")

        # ---- 2. Cognition signal via dispatcher (Phase 5 of #889) -------

        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is None:
            # Pre-Phase-5 agents (or tests with mocked agents lacking a
            # dispatcher) just get the SSE notification \u2014 backward compat.
            return

        try:
            from kestrel_sovereign.signals.sources.a2a import (
                build_signal_for_completed_task,
            )

            signal = build_signal_for_completed_task(
                task=task, target_agent=self.did
            )

            # enqueue_signal is async (returns SignalHandle). We're in a
            # sync callback (TaskManager._notify_status_update calls us
            # synchronously). Wrap the await in a tracked coroutine \u2014
            # the agent's background task tracker supervises the work
            # so exceptions land in logs and shutdown drains it cleanly.
            async def _enqueue():
                await dispatcher.enqueue_signal(signal)

            self._track_background_task(
                _enqueue(), name=f"a2a_complete:{task_id[:8]}",
            )
        except Exception as e:
            # Never let a dispatcher failure break the SSE notification
            # path (browser users still get the green check). Log and
            # continue.
            logging.warning(
                "Failed to enqueue a2a.task_complete signal for %s: %s",
                task_id, e, exc_info=True,
            )

    def get_pending_notifications(self) -> List[str]:
        """
        Get and clear pending task completion notifications.

        Called by the chat endpoint to include notifications in responses.
        """
        notifications = self._pending_task_notifications.copy()
        self._pending_task_notifications.clear()
        return notifications
