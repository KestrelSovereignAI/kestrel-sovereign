"""Terminal-delivery accounting for durable signal producers (#2532).

``SignalDispatcher.enqueue_signal`` returns a :class:`SignalHandle` the moment a
signal is **accepted onto the queue**. Acceptance is not delivery: only the
:class:`SignalResult` produced by ``await handle.wait()`` reports the terminal
``OK`` / ``FAILED`` / dropped outcome.

The rule this module exists to enforce — the same one #2774 applied to
``wake_delivered`` and #2530 applied to the operator-notice audit:

    **A checkpoint may only advance when the thing it records is beyond loss.**

A producer that advances a durable fingerprint (or retires a durable
``WAITING`` row) on acceptance records a delivery nobody observed. That is
worse than a redelivery bug: the advanced checkpoint means the next poll sees
no change, so nothing ever retries and the finding is lost *silently and
permanently*.

Classification
--------------
Only ``Status.OK`` is checkpoint-grade. ``COALESCED`` deliberately is **not**:
the dedupe key is recorded *before* the turn runs, so a wake that failed inside
the resuming turn still suppresses a fast retry as ``COALESCED``. Advancing on
that would lock in a delivery that never produced a turn — the identical
reasoning ``RestartCoordinatorFeature._spawn_ack_supervisor`` documents for
``wake_delivered``. Callers that only want *observability* (a detached callback
with no durable checkpoint) can pass :data:`ACCEPTED_STATUSES` instead, so a
legitimately coalesced duplicate is not logged as a failure.

Ownership boundary
------------------
Awaiting a terminal result is only safe from a task that is not itself blocking
the thing the dispatch must get past:

* **Safe to await inline** — a feature-owned background task
  (``Feature._track_owned_background_task``), i.e. any caller that can block
  for the full length of a COGNITION turn without holding something else up.
* **NOT safe to await inline** — a cron ACTION handler running inside a
  dispatcher worker (it holds a scheduler lease and one of the runner's
  bounded concurrency slots; a minutes-long LLM turn would starve other due
  tasks and let the lease lapse into a duplicate execution), and anything on
  the boot path such as ``Feature.initialize`` / startup replay, where the
  COGNITION turn being awaited cannot run until boot finishes.

Producers in the second group hand the wait to a supervisor task via
:func:`supervise_terminal_delivery` — exactly the shape
``RestartCoordinatorFeature`` uses for ``restart.completed``.
:class:`~kestrel_sovereign.waits.reconciler.WaitReconciler` is the other
sanctioned shape (enqueue on one tick, harvest at the top of the next, backed
by a durable pending row). Do not invent a third.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, FrozenSet, Optional

logger = logging.getLogger(__name__)


# Terminal statuses that may advance a durable checkpoint. ``COALESCED`` is
# excluded on purpose — see the module docstring.
DELIVERED_STATUSES: FrozenSet[str] = frozenset({"ok"})

# Terminal statuses that are "not a problem" for a producer that keeps no
# durable checkpoint. A coalesced duplicate means an equivalent signal was
# already dispatched, which is the intended outcome for a detached callback.
ACCEPTED_STATUSES: FrozenSet[str] = frozenset({"ok", "coalesced"})

# Synthetic statuses. None of these come from a dispatcher ``SignalResult`` —
# they record that no terminal result was ever observed, which is exactly the
# state a checkpoint must not advance through.
STATUS_NO_HANDLE = "no_handle"
STATUS_DISPATCH_CANCELLED = "dispatch_cancelled"
STATUS_TOOLING_ERROR = "tooling_error"
# This supervisor was cancelled (feature shutdown, soft disable, boot
# rollback) before a terminal result arrived. Distinct from
# ``dispatch_cancelled``: there the caller is healthy and the dispatch died,
# here the caller itself is going away.
STATUS_SUPERVISOR_CANCELLED = "supervisor_cancelled"


@dataclass(frozen=True)
class DeliveryOutcome:
    """Normalized terminal outcome of one ``enqueue_signal`` dispatch."""

    status: str
    delivered: bool
    error: Optional[str] = None

    def describe(self) -> str:
        """Single-line ``status`` (+ error) for log lines."""
        return f"{self.status}: {self.error}" if self.error else self.status


def _outcome(
    status: str, delivered_statuses: FrozenSet[str], error: Optional[str] = None
) -> DeliveryOutcome:
    return DeliveryOutcome(
        status=status, delivered=status in delivered_statuses, error=error
    )


async def await_terminal_delivery(
    handle: Any,
    *,
    label: str,
    delivered_statuses: FrozenSet[str] = DELIVERED_STATUSES,
) -> DeliveryOutcome:
    """Await one dispatch to a terminal state and classify it.

    Never raises for a *signal* outcome — ``FAILED`` and every ``DROPPED_*``
    come back as a non-delivered :class:`DeliveryOutcome` so the caller can
    retain its checkpoint and keep going.

    Cancellation is split, because the two cases mean opposite things:

    * **The dispatch task was cancelled** (agent shutdown reaped it, the
      handle was cancelled) — reported as ``dispatch_cancelled`` and not
      delivered. This caller is healthy and must still run its retain path.
    * **This caller is being cancelled** — re-raised. Swallowing it would
      turn a shutdown into a silent normal return.

      Re-raising is NOT on the grounds that there is nothing to restore. That
      is only true for a producer which retains by *not advancing*; one which
      optimistically *retires* durable state before dispatching (A2A question
      completion) still has a row to put back. Callers that supervise through
      :func:`supervise_terminal_delivery` get that restore on cancellation —
      see its handler. A caller awaiting this directly owns its own.

    A handle that exposes no awaitable ``wait`` (legacy/stub dispatcher)
    yields ``no_handle``; anything else going wrong in the wait itself yields
    ``tooling_error``. Neither is delivered — an unobservable dispatch is
    treated exactly like a failed one.
    """
    waiter = getattr(handle, "wait", None)
    if not callable(waiter):
        return _outcome(
            STATUS_NO_HANDLE,
            delivered_statuses,
            error="dispatcher returned no awaitable handle",
        )

    try:
        result = await waiter()
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            # We are the one being cancelled — propagate.
            raise
        # The dispatch task died, not us.
        return _outcome(
            STATUS_DISPATCH_CANCELLED,
            delivered_statuses,
            error="signal dispatch task was cancelled",
        )
    except Exception as e:  # noqa: BLE001 - a broken wait is not a delivery
        logger.warning("%s: awaiting signal delivery raised: %s", label, e)
        return _outcome(
            STATUS_TOOLING_ERROR, delivered_statuses, error=f"{type(e).__name__}: {e}"
        )

    status = getattr(result, "status", None)
    status_value = getattr(status, "value", None)
    if not isinstance(status_value, str):
        # A result the dispatcher contract says cannot happen. Treat it as
        # undelivered rather than guessing the signal landed.
        return _outcome(
            STATUS_TOOLING_ERROR,
            delivered_statuses,
            error=f"dispatch returned no usable status: {result!r}",
        )

    return _outcome(
        status_value, delivered_statuses, error=getattr(result, "error", None) or None
    )


def supervise_terminal_delivery(
    feature: Any,
    handle: Any,
    *,
    label: str,
    task_name: str,
    on_delivered: Optional[Callable[[], Awaitable[Any]]] = None,
    on_undelivered: Optional[Callable[[DeliveryOutcome], Awaitable[Any]]] = None,
    on_settled: Optional[Callable[[], None]] = None,
    delivered_statuses: FrozenSet[str] = DELIVERED_STATUSES,
) -> Any:
    """Spawn a feature-owned task that awaits ``handle`` and settles the
    checkpoint, for producers that cannot await inline.

    ``on_delivered`` runs only on a terminal delivered status — that is the
    *only* place a durable checkpoint may advance. ``on_undelivered`` runs for
    every other terminal state (``FAILED``, any ``DROPPED_*``, a cancelled
    dispatch, a tooling error) **and on cancellation of this supervisor
    itself** — that last one matters, because shutdown is exactly when a
    producer which optimistically retired a durable row would otherwise leave
    it retired forever.

    ``on_settled`` is a synchronous finalizer for in-flight bookkeeping. It
    runs on every exit path *including* cancellation of this supervisor, so a
    de-dupe guard cannot wedge a watch permanently.

    The task is registered through ``Feature._track_owned_background_task`` so
    feature shutdown / boot rollback / soft disable cancel exactly this work,
    while the agent's global reap set still covers full shutdown.
    """

    async def _await_and_settle() -> None:
        try:
            outcome = await await_terminal_delivery(
                handle, label=label, delivered_statuses=delivered_statuses
            )
            if outcome.delivered:
                if on_delivered is not None:
                    await on_delivered()
                return
            logger.warning(
                "%s: signal was accepted but never delivered (%s); durable "
                "state stays un-advanced so the next pass retries",
                label,
                outcome.describe(),
            )
            if on_undelivered is not None:
                await on_undelivered(outcome)
        except asyncio.CancelledError:
            # This supervisor is being cancelled — feature shutdown, soft
            # disable, or boot rollback. Awaiting the handle cancels the
            # dispatch with it, so the wake will never land.
            #
            # ``await_terminal_delivery`` re-raises on the reasoning that a
            # checkpoint which never advanced is already correct. That holds
            # for a producer which retains by NOT advancing (the watchers),
            # and is false for one which optimistically RETIRES durable state
            # before dispatching: A2A question completion claims the row
            # ``RESOLVED``/``EXPIRED`` up front. Skipping the restore leaves
            # that row terminal forever and startup replay never resumes the
            # asker — the same permanent silent loss this module exists to
            # prevent, arriving through the shutdown path instead.
            #
            # Restore only. No retry is scheduled: teardown is not the time to
            # start new work, and the restored row is what makes the next boot
            # pick this up.
            if on_undelivered is not None:
                await on_undelivered(
                    _outcome(STATUS_SUPERVISOR_CANCELLED, delivered_statuses)
                )
            raise
        finally:
            if on_settled is not None:
                on_settled()

    return feature._track_owned_background_task(_await_and_settle(), name=task_name)


def harvest_detached_delivery(
    track: Callable[..., Any],
    enqueue: Callable[[], Awaitable[Any]],
    *,
    label: str,
    task_name: str,
    delivered_statuses: FrozenSet[str] = ACCEPTED_STATUSES,
) -> Any:
    """Own and harvest a detached (non-checkpointing) dispatch.

    For call sites where nothing durable advances on delivery, so retrying is
    not this producer's job — but the dispatch must still be *owned* (so
    shutdown drains it) and its terminal result *harvested* (so a signal that
    silently failed shows up in the log rather than vanishing).

    ``enqueue`` is the coroutine factory that performs the ``enqueue_signal``;
    it runs inside the tracked task so a synchronous callback can call this
    without an event-loop hop of its own. ``track`` is the owning tracker —
    ``agent._track_background_task`` or ``Feature._track_owned_background_task``.
    """

    async def _enqueue_and_harvest() -> None:
        try:
            handle = await enqueue()
        except Exception as e:  # noqa: BLE001 - mirrors the pre-harvest posture
            logger.warning("%s: enqueue_signal raised: %s", label, e, exc_info=True)
            return
        outcome = await await_terminal_delivery(
            handle, label=label, delivered_statuses=delivered_statuses
        )
        if not outcome.delivered:
            logger.warning(
                "%s: signal was accepted but never delivered (%s)",
                label,
                outcome.describe(),
            )

    return track(_enqueue_and_harvest(), name=task_name)
