"""
Event manager mixin for KestrelAgent.

Extracted from kestrel_agent.py — provides SSE event emission,
listener management, and background task notification queuing.
"""

import logging
from typing import Any, Dict, List, Tuple


def describe_background_task(task) -> Tuple[str, str]:
    """Derive a human-meaningful ``source/name`` label for a terminal
    background task from its metadata.

    Only keys that a real producer actually stamps on ``Task.metadata``
    are read here — reading speculative keys no one writes would give a
    false sense of coverage (#1526). The genuine producers are:

    - ``TaskManager.execute_skill`` stamps ``agent_id`` + ``skill`` when
      a peer agent runs one of our skills (``a2a/task_manager.py``).
    - The inbound A2A submit endpoint passes the caller-supplied
      ``sender`` (and optionally ``task_type``) straight through
      ``create_task`` (``endpoints/agent.py`` → ``TaskSendParams.metadata``).
    - ``TaskManager.create_task`` attaches a ``causation_chain`` whenever
      a task is spawned during a signal-driven turn.

    Scheduled tasks themselves never reach this callback directly: the
    scheduler dispatches a *signal* (``cron.<task_name>``), not an A2A
    task (``features/scheduler/feature.py``). The only thread connecting
    a later spawned task back to that schedule is the causation chain, so
    that is the canonical path for labelling scheduler-originated work —
    a task whose chain records ``cron.restart_coordinator`` renders as
    ``cron/restart_coordinator`` instead of the historical
    ``unknown/task``.

    Preferring the most specific identifiers available keeps completion
    notifications from collapsing to ``unknown/task`` whenever a richer
    field is present.

    Returns a ``(source, name)`` tuple; the caller joins them as
    ``source/name``.
    """
    md = getattr(task, "metadata", None)
    if not isinstance(md, dict):
        md = {}

    source = md.get("agent_id") or md.get("sender")
    name = md.get("skill") or md.get("task_type")

    if source and name:
        return str(source), str(name)
    if source:
        return str(source), "task"
    if name:
        return "unknown", str(name)

    # Fall back to the originating signal source recorded in the
    # causation chain — e.g. a cron task (``cron.restart_coordinator``)
    # that woke a turn which then spawned this task. The most recent
    # frame is the proximate cause. ``cron.restart_coordinator`` renders
    # as ``cron/restart_coordinator``; a single-segment source renders
    # as ``<source>/task``.
    frame_source = _causation_chain_source(md)
    if frame_source:
        head, sep, tail = frame_source.partition(".")
        if sep and tail:
            return head, tail
        return frame_source, "task"

    return "unknown", "task"


def _causation_chain_source(metadata: Dict[str, Any]) -> str:
    """Return the source of the most recent causation-chain frame, or ''.

    Defensive against the wire-serialized chain shape produced by
    ``signals.sources.a2a.serialize_chain_for_metadata`` (a list of
    dicts, each with a ``source`` key).
    """
    chain = metadata.get("causation_chain")
    if not isinstance(chain, list) or not chain:
        return ""
    frame = chain[-1]
    if isinstance(frame, dict):
        src = frame.get("source")
        if src:
            return str(src)
    return ""


def background_task_identifiers(task) -> str:
    """Build the identifier suffix for a completion notification.

    Always exposes the FULL task id (not a truncated prefix) so the toast
    text is directly resolvable via ``check_task_status`` and correlates
    with task-registry records (#1526). The historical bug truncated this
    to an 8-char prefix that ``check_task_status`` could not resolve.

    No scheduler ``execution_id`` is surfaced: that id is the
    ``task_execution_log`` row id, generated *after* the scheduled run
    completes (``features/scheduler/runner.py``), so it never exists on
    the A2A task this callback receives. Correlation with scheduler
    history instead flows through the ``cron/<task_name>`` label that
    :func:`describe_background_task` derives from the causation chain.
    """
    return f"task: {getattr(task, 'id', 'unknown')}"


class EventManagerMixin:
    """Mixin providing event/notification methods for KestrelAgent."""

    # Cap on events buffered for replay to a reconnecting listener. Bounds
    # memory on a headless host that never opens an SSE stream; the oldest
    # buffered events drop first once the cap is exceeded.
    _MAX_PENDING_EVENTS = 100

    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered listeners (for SSE notifications).

        When NO listener is connected, the event is buffered and replayed
        to the first listener that connects (see ``get_pending_events``).
        Without this, events emitted while the browser's SSE stream is
        momentarily absent are silently lost — notably the restart
        ``completed`` status emitted from ``feature.initialize()`` during
        host startup, BEFORE the browser reconnects its notifications
        stream. That is the one transition that straddles the restart, so
        losing it defeated the issue's primary acceptance criterion (#1551).

        Args:
            event_type: Type of event (e.g., 'approval_request')
            data: Event data to send
        """
        if not self._event_listeners:
            self._buffer_pending_event(event_type, data)
            return
        for listener in self._event_listeners:
            try:
                await listener(event_type, data)
            except (TypeError, AttributeError, ConnectionError) as e:
                logging.warning(f"Failed to emit event to listener: {e}")
            except Exception as e:
                logging.warning(f"Failed to emit event to listener: {e}", exc_info=True)

    def _buffer_pending_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Buffer an event emitted while no listener was connected.

        Drained by ``get_pending_events`` when a client reconnects. The
        attribute is lazily created so minimal agent stand-ins that only
        set ``_event_listeners`` keep working.
        """
        buf = getattr(self, "_pending_events", None)
        if buf is None:
            buf = []
            self._pending_events = buf
        buf.append((event_type, data))
        overflow = len(buf) - self._MAX_PENDING_EVENTS
        if overflow > 0:
            del buf[:overflow]

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

        # ---- 1. SSE notification (legacy path) ---------------------------

        # Derive a meaningful source/name label and identifier suffix from
        # whatever metadata the producer stamped. Falls back to the
        # historical "unknown/task" only when nothing better is present, and
        # always exposes the full task id so the toast is resolvable (#1526).
        source, name = describe_background_task(task)
        label = f"{source}/{name}"
        identifiers = background_task_identifiers(task)

        # Format notification based on state
        if state == TaskState.COMPLETED:
            msg = f"\u2705 Background task completed: {label} ({identifiers})"
        elif state == TaskState.FAILED:
            error_msg = ""
            if task.status.message and task.status.message.parts:
                for part in task.status.message.parts:
                    if hasattr(part, 'text'):
                        error_msg = f": {part.text}"
                        break
            msg = f"\u274c Background task failed: {label}{error_msg} ({identifiers})"
        elif state == TaskState.CANCELED:
            msg = f"\u26a0\ufe0f Background task canceled: {label} ({identifiers})"
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

    def _on_task_submitted(self, task) -> None:
        """
        Callback invoked when a peer agent creates a new A2A task
        addressed to this agent.

        Closes the inbound wakeup gap: the matching counterpart to
        ``_on_background_task_complete`` for the SUBMITTED direction.
        Without this, a peer-submitted task sits in the local task
        store with no one acting on it until the next user-driven chat
        turn — the Emma/Meridian symptom (#645 missing piece).

        Mirrors `channels.feature.py:425` — the proven inbound-signal
        pattern: dispatch a COGNITION signal and let the dispatcher
        wake the cognition loop. The new turn sees the task in
        context and decides what to do.

        Called by TaskManager from ``create_task`` (synchronous
        callback). The actual ``dispatcher.enqueue_signal`` await is
        wrapped in a tracked background task so exceptions land in
        logs and shutdown drains it cleanly.
        """
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is None:
            # Pre-dispatcher agents (or test fixtures without one) get
            # the persisted task but no wake. The TaskStore row still
            # exists, so a subsequent reading turn can see it; just no
            # autonomous trigger. Same backward-compat posture as
            # ``_on_background_task_complete``.
            return

        task_id = getattr(task, "id", "<unknown>")
        try:
            from kestrel_sovereign.signals.sources.a2a_task_submitted import (
                build_signal_for_submitted_task,
            )

            # ``sender`` lives in task.metadata when the create came
            # via the inter-agent HTTP send path. Local self-spawn
            # paths leave it blank, which is fine — the signal still
            # fires (allow_self_loops=False prevents the degenerate
            # case at dispatch time).
            metadata = getattr(task, "metadata", None) or {}
            sender = str(metadata.get("sender", "") or "") if isinstance(metadata, dict) else ""

            signal = build_signal_for_submitted_task(
                task=task,
                target_agent=getattr(self, "did", ""),
                sender=sender,
            )

            async def _enqueue():
                await dispatcher.enqueue_signal(signal)

            self._track_background_task(
                _enqueue(), name=f"a2a_submitted:{str(task_id)[:8]}",
            )
        except Exception as e:
            # Same posture as task_complete: never let a dispatcher
            # failure break task creation. Log and continue — the
            # task IS persisted; only the wake was missed.
            logging.warning(
                "Failed to enqueue a2a.task_submitted signal for %s: %s",
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

    def get_pending_events(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Drain events buffered while no SSE listener was connected.

        Structured counterpart to ``get_pending_notifications`` for the
        ``emit_event`` bus. The notifications SSE generator calls this
        right after registering its listener so a client reconnecting
        after a host restart still receives events emitted during startup
        — notably the restart ``completed`` status (#1551). Drain-once
        semantics, same as task notifications: the first reconnecting
        client consumes the buffer.
        """
        buf = getattr(self, "_pending_events", None)
        if not buf:
            return []
        drained = list(buf)
        buf.clear()
        return drained
