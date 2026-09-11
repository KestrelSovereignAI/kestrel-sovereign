"""
Event manager mixin for KestrelAgent.

Extracted from kestrel_agent.py — provides SSE event emission,
listener management, and background task notification queuing.
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# emit_event delivery receipt (#2922)
# ---------------------------------------------------------------------------
#
# ``emit_event`` used to return ``None`` unconditionally, and it swallows every
# listener exception. A caller therefore could not tell "every SSE forwarder
# failed" from "the event went out" — so the wait reconciler recorded a bare
# ``ok`` for wakes that reached nobody, which is the self-reporting failure
# #2877/#2922 exist to remove.
#
# The receipt reports the three outcomes an emitter can actually distinguish.
# NONE of them means a human saw anything: ``ACCEPTED`` means a listener
# callback returned without raising, which for the ``/notifications/sse``
# forwarder means the event entered a server-side queue. The browser can still
# discard it (``chat.js`` drops a wake whose ``session_id`` is not the pane's
# open conversation). Server-side truth stops at acceptance; callers must not
# promote it to "rendered".

EVENT_BUFFERED = "buffered"
EVENT_ACCEPTED = "accepted"
EVENT_REJECTED = "rejected"

# Cross-worker cancellation cannot rely on a process-local Event.  Keep the
# durable fallback responsive at first, then cap its steady-state database
# load for long LLM/tool turns.
A2A_CANCELLATION_POLL_INITIAL_SECONDS = 0.1
A2A_CANCELLATION_POLL_MAX_SECONDS = 2.0
A2A_WAKE_RETRY_INITIAL_SECONDS = 0.1
A2A_WAKE_RETRY_MAX_SECONDS = 5.0


@dataclass(frozen=True)
class EventDeliveryReceipt:
    """Aggregate outcome of one :meth:`EventManagerMixin.emit_event` call.

    Attributes:
        listeners: Listeners the event was offered to (0 when buffered).
        accepted: Listeners whose callback returned without raising.
        rejected: Listeners whose callback raised (the failure is logged and
            swallowed, so this counter is the only way a caller learns of it).
        buffered: True when no listener was connected and the event was
            buffered for replay to the next one (see ``get_pending_events``).
    """

    listeners: int
    accepted: int
    rejected: int
    buffered: bool

    @property
    def outcome(self) -> str:
        """``buffered`` / ``accepted`` / ``rejected`` — in that precedence.

        ``accepted`` requires at least one listener to have taken the event;
        an emit whose every listener raised is ``rejected``, never ``accepted``.
        """
        if self.buffered:
            return EVENT_BUFFERED
        if self.accepted > 0:
            return EVENT_ACCEPTED
        return EVENT_REJECTED


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

    No scheduler ``execution_id`` is surfaced. It is now claimed before
    scheduler dispatch so target tools can use it for idempotency, but it
    deliberately remains scheduler-local context rather than untrusted A2A
    task metadata. Correlation with scheduler history instead flows through
    the ``cron/<task_name>`` label that :func:`describe_background_task`
    derives from the causation chain.
    """
    return f"task: {getattr(task, 'id', 'unknown')}"


class EventManagerMixin:
    """Mixin providing event/notification methods for KestrelAgent."""

    # Cap on events buffered for replay to a reconnecting listener. Bounds
    # memory on a headless host that never opens an SSE stream; the oldest
    # buffered events drop first once the cap is exceeded.
    _MAX_PENDING_EVENTS = 100

    async def emit_event(
        self, event_type: str, data: Dict[str, Any]
    ) -> EventDeliveryReceipt:
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

        A listener that raises is logged and skipped — one broken forwarder
        must not deny the event to the others, and a UI notification is never
        worth failing the work that produced it. But swallowing the failure
        and returning ``None`` also left the caller unable to tell a total
        delivery failure from a success (#2922): the wait reconciler read
        "emit returned" as "the user can see it" and recorded a bare ``ok``
        for wakes that reached nobody. So the aggregate outcome is RETURNED
        rather than only logged.

        Args:
            event_type: Type of event (e.g., 'approval_request')
            data: Event data to send

        Returns:
            An :class:`EventDeliveryReceipt`. ``accepted`` counts listeners
            that took the event without raising — for the SSE forwarder that
            is server-side queue admission, NOT proof that anything rendered.
            Callers must not report an accepted emit as "seen by the user".
        """
        if not self._event_listeners:
            self._buffer_pending_event(event_type, data)
            return EventDeliveryReceipt(
                listeners=0, accepted=0, rejected=0, buffered=True
            )
        accepted = 0
        rejected = 0
        listeners = list(self._event_listeners)
        for listener in listeners:
            try:
                await listener(event_type, data)
            except (TypeError, AttributeError, ConnectionError) as e:
                rejected += 1
                logging.warning(f"Failed to emit event to listener: {e}")
            except Exception as e:
                rejected += 1
                logging.warning(f"Failed to emit event to listener: {e}", exc_info=True)
            else:
                accepted += 1
        return EventDeliveryReceipt(
            listeners=len(listeners),
            accepted=accepted,
            rejected=rejected,
            buffered=False,
        )

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

    async def _deliver_a2a_wake_until_durable(
        self,
        *,
        signal_factory,
        task_id: str,
        consumer_id: str,
        label: str,
        cancellation_aware: bool,
    ) -> None:
        """Retry the task->signal outbox handoff until it is durable.

        The task row committed before this coroutine was scheduled. A transient
        dispatcher/storage failure therefore cannot be treated as completion:
        retry in-process, while boot reconciliation covers process loss.
        """

        from kestrel_sovereign.signals import DurableAdmissionDisposition
        from kestrel_sovereign.signals.delivery import (
            ACCEPTED_STATUSES,
            await_terminal_delivery,
        )

        delay = A2A_WAKE_RETRY_INITIAL_SECONDS
        pending = vars(self).setdefault("_a2a_submitted_signal_handles", {})
        # A retry loop must terminate on a condition retrying cannot change.
        # Durable admission needs a registered durable cognition consumer;
        # an agent that never registered one (the ordinary non-durable
        # configuration) can never admit this wake, and NOT_ADMITTED there is
        # permanent, not transient. Before #3163 this path was a plain
        # enqueue_signal wake -- "nothing durable advances on delivery here,
        # the task row is already persisted, this signal is only the wake."
        # Keep that contract for those agents instead of spinning at the 5s
        # cap forever.
        has_durable = getattr(self.dispatcher, "has_durable_consumer", None)
        if callable(has_durable) and not await has_durable(consumer_id):
            signal = signal_factory()
            handle = await self.dispatcher.enqueue_signal(signal)
            outcome = await await_terminal_delivery(
                handle,
                label=label,
                delivered_statuses=ACCEPTED_STATUSES,
            )
            if not outcome.delivered:
                logging.warning(
                    "%s: non-durable wake was not delivered (%s)",
                    label,
                    outcome.describe(),
                )
            return
        while True:
            if cancellation_aware:
                try:
                    snapshot = await self.task_manager.get_task_cancellation_snapshot(
                        task_id,
                        recipient_agent_id=self.did,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logging.warning(
                        "%s: task-state read failed; retrying durable wake: %s",
                        label,
                        exc,
                        exc_info=True,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, A2A_WAKE_RETRY_MAX_SECONDS)
                    continue
                if snapshot is None or snapshot.state not in {"submitted", "working"}:
                    return
            try:
                # Dispatch mutates normalization and causation state in-place.
                # A persistence retry must therefore start from a fresh source
                # envelope rather than appending a second frame to the failed
                # attempt's object.
                signal = signal_factory()
                handle = await self.dispatcher.enqueue_durable_cognition(
                    signal,
                    source_event_id=task_id,
                    consumer_id=consumer_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logging.warning(
                    "%s: durable enqueue failed; retrying: %s",
                    label,
                    exc,
                    exc_info=True,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, A2A_WAKE_RETRY_MAX_SECONDS)
                continue

            if cancellation_aware:
                pending[task_id] = handle
            retry = False
            try:
                if cancellation_aware:
                    try:
                        snapshot = (
                            await self.task_manager.get_task_cancellation_snapshot(
                                task_id,
                                recipient_agent_id=self.did,
                            )
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # The dispatcher's execution-time validation reads the
                        # same authoritative row. Keep the durable handoff live
                        # and observable rather than abandoning it here.
                        logging.warning(
                            "%s: post-enqueue task-state read failed: %s",
                            label,
                            exc,
                            exc_info=True,
                        )
                    else:
                        if snapshot is None or snapshot.state not in {
                            "submitted",
                            "working",
                        }:
                            dispatch_task = getattr(handle, "task", None)
                            if dispatch_task is not None and not dispatch_task.done():
                                dispatch_task.cancel()

                admission_waiter = getattr(handle, "wait_for_durable_admission", None)
                admission = (
                    await admission_waiter()
                    if callable(admission_waiter)
                    else None
                )
                outcome = await await_terminal_delivery(
                    handle,
                    label=label,
                    delivered_statuses=ACCEPTED_STATUSES,
                )
                if not outcome.delivered:
                    logging.warning(
                        "%s: signal was accepted but never delivered (%s)",
                        label,
                        outcome.describe(),
                    )
                retry = (
                    admission is not None
                    and admission.disposition
                    is DurableAdmissionDisposition.NOT_ADMITTED
                )
            finally:
                if cancellation_aware and pending.get(task_id) is handle:
                    pending.pop(task_id, None)
                if cancellation_aware:
                    self_declines = vars(self).get("_a2a_self_declining_task_ids")
                    if isinstance(self_declines, set):
                        self_declines.discard(task_id)
            if not retry:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, A2A_WAKE_RETRY_MAX_SECONDS)

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
                DURABLE_COGNITION_CONSUMER_ID,
                build_signal_for_completed_task,
            )

            # enqueue_durable_cognition is async and returns a SignalHandle at
            # *acceptance*; the terminal result only arrives via
            # handle.wait(). We're in a sync callback (TaskManager
            # ._notify_status_update calls us synchronously). The task row is
            # already terminal, so this signal is only its wake; the selected
            # durable consumer preserves that wake across Hold and restart.
            # The agent's tracker owns the initial enqueue and harvests its
            # terminal result for observability (#2532).
            self._track_background_task(
                self._deliver_a2a_wake_until_durable(
                    signal_factory=lambda: build_signal_for_completed_task(
                        task=task,
                        target_agent=self.did,
                    ),
                    task_id=str(task_id),
                    consumer_id=DURABLE_COGNITION_CONSUMER_ID,
                    label=f"a2a.task_complete[{task_id}]",
                    cancellation_aware=False,
                ),
                name=f"a2a_complete:{task_id[:8]}",
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
                DURABLE_COGNITION_CONSUMER_ID,
                build_signal_for_submitted_task,
            )

            # ``sender`` lives in task.metadata when the create came
            # via the inter-agent HTTP send path. Local self-spawn
            # paths leave it blank, which is fine — the signal still
            # fires (allow_self_loops=False prevents the degenerate
            # case at dispatch time).
            metadata = getattr(task, "metadata", None) or {}
            sender = str(metadata.get("sender", "") or "") if isinstance(metadata, dict) else ""

            self._track_background_task(
                self._deliver_a2a_wake_until_durable(
                    signal_factory=lambda: build_signal_for_submitted_task(
                        task=task,
                        target_agent=getattr(self, "did", ""),
                        sender=sender,
                    ),
                    task_id=str(task_id),
                    consumer_id=DURABLE_COGNITION_CONSUMER_ID,
                    label=f"a2a.task_submitted[{task_id}]",
                    cancellation_aware=True,
                ),
                name=f"a2a_submitted:{str(task_id)[:8]}",
            )
        except Exception as e:
            # Same posture as task_complete: never let a dispatcher
            # failure break task creation. Log and continue — the
            # task IS persisted; only the wake was missed.
            logging.warning(
                "Failed to enqueue a2a.task_submitted signal for %s: %s",
                task_id, e, exc_info=True,
            )

    async def rehydrate_durable_cognition_signal(
        self,
        event,
        *,
        dispatch_signal,
    ):
        """Rebuild a privacy-elided A2A wake from its authoritative task row."""

        source = getattr(event, "source", None)
        if source not in {"a2a.task_submitted", "a2a.task_complete"}:
            return None
        # WORKING recovery deliberately uses a revision-specific source event
        # identity so it cannot deduplicate against the original SUBMITTED
        # wake. The canonical task id remains in the durable dedupe field,
        # which privacy projection retains without storing authored content.
        task_id = (
            getattr(event, "dedupe_key", None)
            if source == "a2a.task_submitted"
            else getattr(event, "source_event_id", None)
        )
        if not isinstance(task_id, str) or not task_id:
            task_id = getattr(event, "source_event_id", None)
        if not isinstance(task_id, str) or not task_id:
            return None
        if getattr(event, "target_agent", None) != getattr(self, "did", None):
            return None
        task = await self.task_manager.get_task_for_recipient(task_id, self.did)
        if task is None:
            return None

        if source == "a2a.task_submitted":
            from kestrel_sovereign.signals.sources.a2a_task_submitted import (
                build_signal_for_submitted_task,
            )

            metadata = getattr(task, "metadata", None) or {}
            sender = (
                str(metadata.get("sender", "") or "")
                if isinstance(metadata, dict)
                else ""
            )
            recovered = build_signal_for_submitted_task(
                task,
                target_agent=self.did,
                sender=sender,
            )
        else:
            from kestrel_sovereign.signals.sources.a2a import (
                build_signal_for_completed_task,
            )

            recovered = build_signal_for_completed_task(
                task,
                target_agent=self.did,
            )
        return replace(
            recovered,
            id=dispatch_signal.id,
            arrived_at=dispatch_signal.arrived_at,
        )

    async def reconcile_a2a_cognition_wakes(self) -> None:
        """Recreate any A2A wake lost after its authoritative task commit.

        Task callbacks cannot be atomic with the asynchronous signal ledger.
        Boot therefore treats TaskStore as the outbox authority and idempotently
        replays candidates before durable drainers are allowed to consume
        privacy-elided marker rows.
        """

        from kestrel_sovereign.a2a.types import TaskState
        from kestrel_sovereign.signals import DurableAdmissionDisposition
        from kestrel_sovereign.signals.sources.a2a import (
            DURABLE_COGNITION_CONSUMER_ID as COMPLETE_CONSUMER,
            build_signal_for_completed_task,
        )
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            DURABLE_COGNITION_CONSUMER_ID as SUBMITTED_CONSUMER,
            build_signal_for_submitted_task,
        )

        tasks = await self.task_manager.list_cognition_wake_candidates(
            recipient_agent_id=self.did,
            terminal_updated_since=datetime.now(timezone.utc) - timedelta(days=14),
        )
        for candidate in tasks:
            task = candidate.task
            state = task.status.state
            if state in {TaskState.SUBMITTED, TaskState.WORKING}:
                metadata = getattr(task, "metadata", None) or {}
                sender = (
                    str(metadata.get("sender", "") or "")
                    if isinstance(metadata, dict)
                    else ""
                )
                signal = build_signal_for_submitted_task(
                    task,
                    target_agent=self.did,
                    sender=sender,
                )
                consumer_id = SUBMITTED_CONSUMER
                # SUBMITTED owns the callback's original source identity.
                # WORKING is a later durable lifecycle revision: if the
                # original wake already ACKed before a process died, reusing
                # the task ID would deduplicate against that completed wake.
                source_event_id = (
                    str(task.id)
                    if state is TaskState.SUBMITTED
                    else f"{task.id}:working:{candidate.lifecycle_revision}"
                )
            else:
                signal = build_signal_for_completed_task(
                    task,
                    target_agent=self.did,
                )
                consumer_id = COMPLETE_CONSUMER
                source_event_id = str(task.id)

            handle = await self.dispatcher.enqueue_durable_cognition(
                signal,
                source_event_id=source_event_id,
                consumer_id=consumer_id,
            )
            receipt = await handle.wait_for_durable_admission()
            if receipt.disposition is DurableAdmissionDisposition.NOT_ADMITTED:
                raise RuntimeError(
                    "A2A cognition outbox reconciliation could not durably "
                    f"admit task {task.id}"
                )

    async def validate_cognition_signal_execution(self, signal) -> str | None:
        """Fail closed when durable state withdraws an inbound A2A wake.

        The queued signal handle is process-local, while PostgreSQL-backed A2A
        tasks and their cancellation authority are shared across workers.  The
        dispatcher therefore calls this hook in the execution worker directly
        before ``process_input``.  A cancellation committed by another worker
        can no longer leave a stale local wake free to execute.
        """

        if getattr(signal, "source", None) != "a2a.task_submitted":
            return None
        payload = getattr(signal, "payload", None)
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        if not isinstance(task_id, str) or not task_id:
            return "a2a.task_submitted has no concrete durable task id"

        snapshot = await self.task_manager.get_task_cancellation_snapshot(
            task_id,
            recipient_agent_id=self.did,
        )
        if snapshot is None:
            return f"A2A task {task_id!r} no longer exists"
        durable_state = snapshot.state
        if durable_state not in {"submitted", "working"}:
            return (
                f"A2A task {task_id!r} is already {durable_state!r}; "
                "its submission wake is no longer executable"
            )
        return None

    async def monitor_cognition_signal_execution(self, signal) -> str | None:
        """Watch durable A2A cancellation throughout a cognition turn.

        Validation closes the stale-before-start case. This monitor closes the
        cross-worker race after validation by polling the shared task row until
        either cognition finishes or cancellation commits. A decline issued by
        this exact signal turn is exempt so its status projection and tool
        response can finish; other workers do not share that process-local
        exemption and therefore stop their stale execution.
        """

        if getattr(signal, "source", None) != "a2a.task_submitted":
            return None
        payload = getattr(signal, "payload", None)
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        if not isinstance(task_id, str) or not task_id:
            return "a2a.task_submitted has no concrete durable task id"

        poll_delay = A2A_CANCELLATION_POLL_INITIAL_SECONDS
        while True:
            snapshot = await self.task_manager.get_task_cancellation_snapshot(
                task_id,
                recipient_agent_id=self.did,
            )
            if snapshot is None:
                return f"A2A task {task_id!r} no longer exists"
            durable_state = snapshot.state
            if durable_state == "canceled":
                self_declines = vars(self).get(
                    "_a2a_self_declining_task_ids",
                )
                if (
                    isinstance(self_declines, set)
                    and task_id in self_declines
                ):
                    self_declines.discard(task_id)
                    # The marker is provisional while the recipient's CAS is
                    # in flight. A creator on another worker may win first;
                    # only the durable receipt actor proves this wake owns the
                    # self-decline exemption.
                    if snapshot.actor_agent_id == self.did:
                        return None
                return (
                    f"A2A task {task_id!r} was canceled while its "
                    "submission wake was executing"
                )
            await asyncio.sleep(poll_delay)
            poll_delay = min(
                poll_delay * 2,
                A2A_CANCELLATION_POLL_MAX_SECONDS,
            )

    def finish_cognition_signal_execution(self, signal) -> None:
        """Release any process-local exemption owned by a completed wake."""

        if getattr(signal, "source", None) != "a2a.task_submitted":
            return
        payload = getattr(signal, "payload", None)
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        self_declines = vars(self).get("_a2a_self_declining_task_ids")
        if isinstance(task_id, str) and isinstance(self_declines, set):
            self_declines.discard(task_id)

    def _on_task_cancellation_started(
        self,
        task_id: str,
        actor_agent_id: str,
    ):
        """Mark this exact signal turn's recipient decline before DB await."""

        from kestrel_sovereign.signals.context import get_current_signal

        current_signal = get_current_signal()
        payload = getattr(current_signal, "payload", None)
        if (
            actor_agent_id != getattr(self, "did", None)
            or getattr(current_signal, "source", None) != "a2a.task_submitted"
            or not isinstance(payload, dict)
            or payload.get("task_id") != task_id
        ):
            return None
        self_declines = vars(self).setdefault(
            "_a2a_self_declining_task_ids",
            set(),
        )
        self_declines.add(task_id)

        def rollback() -> None:
            self_declines.discard(task_id)

        return rollback

    def _on_task_cancelled(self, task) -> None:
        """Cancel an admitted task-submission wake after durable cancellation."""

        task_id = str(getattr(task, "id", ""))
        pending = vars(self).get("_a2a_submitted_signal_handles", {})
        handle = pending.pop(task_id, None)
        dispatch_task = getattr(handle, "task", None)
        from kestrel_sovereign.signals.context import get_current_signal

        current_signal = get_current_signal()
        current_payload = getattr(current_signal, "payload", None)
        is_current_signal_decline = (
            getattr(current_signal, "source", None) == "a2a.task_submitted"
            and isinstance(current_payload, dict)
            and current_payload.get("task_id") == task_id
        )
        is_current_dispatch = dispatch_task is asyncio.current_task()
        # A recipient may decline from inside the cognition dispatch that this
        # very handle represents. Cancelling it here would interrupt
        # TaskManager.cancel_task at its next await, after the durable state
        # transition but before status projection and the tool response. Only
        # suppress a queued/different delivery; the current dispatch is already
        # consuming the cancellation and must finish publishing it.
        if is_current_signal_decline or is_current_dispatch:
            self_declines = vars(self).setdefault(
                "_a2a_self_declining_task_ids",
                set(),
            )
            self_declines.add(task_id)
        elif (
            dispatch_task is not None
            and not dispatch_task.done()
        ):
            dispatch_task.cancel()

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

    # ------------------------------------------------------------------
    # Sticky aux events (current-state replay)
    # ------------------------------------------------------------------
    #
    # Some aux renders are CURRENT STATE, not one-shot: a channel pairing QR
    # should appear in any chat session opened while the channel is unlinked,
    # not only for the one client that happened to be connected when it was
    # produced (``emit_event``'s pending buffer is drain-once). These are
    # replayed to EVERY new SSE connection until cleared.

    def set_sticky_event(self, key: str, event_type: str, data: Dict[str, Any]) -> None:
        """Record a current-state aux event, replayed to every new SSE client."""
        sticky = getattr(self, "_sticky_events", None)
        if sticky is None:
            sticky = {}
            self._sticky_events = sticky
        sticky[key] = (event_type, data)

    def clear_sticky_event(self, key: str) -> None:
        sticky = getattr(self, "_sticky_events", None)
        if sticky:
            sticky.pop(key, None)

    def get_sticky_events(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return current-state aux events for replay on SSE connect (not drained)."""
        return list(getattr(self, "_sticky_events", {}).values())
