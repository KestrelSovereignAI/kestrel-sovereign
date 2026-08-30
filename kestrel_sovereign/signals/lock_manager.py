"""Single ordered lock manager for the signal dispatcher.

Per SIGNAL_DISPATCHER.md §Concern 2: any caller acquiring multiple named
locks acquires them in a single global order. `CONVERSATION` is the
highest-order acquisition (always last) and is owned solely by the turn
lifecycle — the dispatcher never pre-acquires it for COGNITION sources.

The order is defined by `lock_sort_key` below: `CONVERSATION` is pinned to
the end via a (rank, name) tuple; everything else lex-orders on its enum
value within the lower rank. We do NOT sort by raw enum value because
"conversation" sorts before "memory"/"scheduler"/"wallet" alphabetically,
which would put CONVERSATION first — directly contradicting the design.

Acquisitions are instrumented (#2770). A turn that stalls while holding
`CONVERSATION` makes an agent look completely dead — every later turn queues
behind a bare `asyncio.Lock` — while its non-turn endpoints keep answering in
milliseconds. Previously that produced no log line at all on either side of the
blocking call: the only INFO record was the turn's own "process_input called",
and `turn_lifecycle` logged begin/end at DEBUG. Diagnosing one incident meant
bisecting the locked region by hand. So both *waiting for* and *holding* a lock
now emit periodic WARNINGs that name the holder and the elapsed time.

These are diagnostics, not enforcement: acquisition still blocks indefinitely
rather than failing. Turns legitimately run for minutes (the Anthropic SDK's
default read timeout alone is 600s, and generation is retried up to 5 times), so
a hard acquire deadline would cancel healthy work. A blocked hold that crosses
the escalation threshold emits ERROR once and appears in authenticated detailed
health; the goal is that the next stall is legible without a manual bisect.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from contextvars import Context, ContextVar
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable, Optional

from kestrel_sdk.signals import ResourceLock

logger = logging.getLogger(__name__)


# How long to wait before reporting, and then re-reporting, a contended
# acquisition. Contention itself is normal — turns serialize by design — so this
# is set well above a healthy turn's queueing time to stay quiet in normal
# operation.
SLOW_WAIT_WARN_SECONDS = 30.0

# How long a single holder may hold a lock before it is reported, and then
# re-reported. Far below the ~600s-per-attempt ceiling an LLM call can reach, so
# a genuine stall surfaces while it is still happening rather than only in
# hindsight.
#
# A long hold is NOT by itself a fault: a healthy generation can legitimately run
# for minutes with nothing else queued. So the severity of a hold report depends
# on whether anyone is actually blocked (see ``_warn_while_holding``) — otherwise
# every slow-but-normal turn would emit WARNINGs for its whole duration and
# WARNING-keyed alerting would learn to ignore exactly this signal.
SLOW_HOLD_WARN_SECONDS = 60.0

# A blocked hold past this point is no longer merely a slow operation. Do not
# cancel it here — some operations intentionally retain task-owned locks — but
# emit one ERROR per hold and expose the same threshold to health reporting.
BLOCKED_HOLD_ERROR_SECONDS = 60 * 60.0


@dataclass(frozen=True)
class LockHolder:
    """Who holds a lock, and since when (``time.monotonic``)."""

    label: str
    acquired_at: float
    owner_task: Optional[asyncio.Task[object]] = field(
        default=None, repr=False, compare=False
    )
    owner_token: object = field(default_factory=object, repr=False, compare=False)

    def held_seconds(self) -> float:
        return time.monotonic() - self.acquired_at


def lock_sort_key(name: ResourceLock) -> tuple[int, str]:
    """Global lock acquisition order.

    `CONVERSATION` is pinned last (rank 1). Every other lock acquires in
    lex order of its enum value (rank 0). Exposed at module scope so tests
    and other components can assert against the canonical ordering.
    """
    if name == ResourceLock.CONVERSATION:
        return (1, "")
    return (0, name.value)


class OrderedLockManager:
    """Acquires a set of named locks in a single sorted pass.

    The single global order (see `lock_sort_key`) is the entire deadlock-
    freedom invariant: with one order, two callers cannot acquire the same
    pair of locks in opposite directions, so the deadlock surface is empty.
    """

    def __init__(self) -> None:
        self._locks: dict[ResourceLock, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._holders: dict[ResourceLock, LockHolder] = {}
        # Blocked acquirers per lock. ``asyncio.Lock`` exposes no public waiter
        # count, and a holder cannot otherwise tell whether it is blocking
        # anyone — which is the difference between "slow" and "harmful".
        self._waiters: dict[ResourceLock, int] = {}
        # Task isolation is explicit: normal ``create_task`` calls must not
        # inherit lock authority, while the top-level invocation wrapper may
        # hand the exact current hold generations to its owned child context.
        self._inherited_ownership: ContextVar[
            tuple[tuple[ResourceLock, object], ...]
        ] = ContextVar(
            f"kestrel_resource_lock_ownership_{id(self)}",
            default=(),
        )

    async def _get(self, name: ResourceLock) -> asyncio.Lock:
        # Lazy creation; protected so first-use races don't double-create.
        if name in self._locks:
            return self._locks[name]
        async with self._registry_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            return self._locks[name]

    @asynccontextmanager
    async def acquire(
        self, names: Iterable[ResourceLock], *, label: Optional[str] = None
    ) -> AsyncIterator[None]:
        """Acquire all named locks in canonical order; release in reverse
        on exit. Empty iterable is a no-op (yields immediately).

        ``label`` identifies the acquirer in slow-wait/slow-hold diagnostics —
        pass something a human can act on, e.g. ``"Nellie turn_ab12cd34ef56"``.
        Without it the lock's own name is reported, which tells an operator that
        *something* is stuck but not what (#2770).
        """
        ordered = sorted(set(names), key=lock_sort_key)
        acquired: list[tuple[ResourceLock, asyncio.Lock]] = []
        watchdogs: list[asyncio.Task] = []
        try:
            for name in ordered:
                lock = await self._get(name)
                await self._acquire_one(name, lock, label)
                acquired.append((name, lock))
                # One hold watchdog per lock actually taken, contended or not —
                # a hold cannot be reported from the wait side, since a stall
                # with no waiter produces no waiter to report it. The cost is a
                # Task allocation per acquisition; it does not reach the genuine
                # hot path because the frequent dispatcher/scheduler sources
                # declare empty resource sets, and an empty acquire never enters
                # this loop at all.
                watchdogs.append(
                    asyncio.ensure_future(self._warn_while_holding(name))
                )
            yield
        finally:
            # Release FIRST, and synchronously. Nothing may await before this
            # loop: a release placed after a suspension point is not
            # cancellation-safe, and a second cancellation arriving during that
            # await (aggressive shutdown, or a wait_for timeout composed with an
            # outer cancel) would skip the release entirely — stranding
            # CONVERSATION and wedging every later turn for the agent. That is
            # precisely the incident this instrumentation exists to diagnose, so
            # the diagnostics must not be able to cause it.
            for name, lock in reversed(acquired):
                self._holders.pop(name, None)
                lock.release()
            # Then retire the diagnostics. Cancelling after release is safe for
            # log correctness because a watchdog that fires in the gap reads an
            # already-popped holder and returns without logging.
            for watchdog in reversed(watchdogs):
                watchdog.cancel()
            if watchdogs:
                # Observe the cancellations rather than just requesting them, so
                # an interpreter/loop teardown mid-hold cannot surface "Task was
                # destroyed but it is pending!" noise from these diagnostics.
                await asyncio.gather(*watchdogs, return_exceptions=True)

    async def _acquire_one(
        self,
        name: ResourceLock,
        lock: asyncio.Lock,
        label: Optional[str],
    ) -> None:
        """Acquire one lock, reporting the wait if it is contended.

        The watchdog runs as a separate task rather than wrapping the acquire in
        ``wait_for``: cancelling a timed-out ``Lock.acquire`` that has already
        succeeded would drop the lock on the floor. ``asyncio.Lock.acquire``
        already handles its own cancellation correctly, so this keeps the
        acquisition path untouched and bolts observability alongside it.
        """
        if not lock.locked():
            # Uncontended fast path: skip the wait watchdog entirely. A lock
            # taken between this check and the acquire below simply costs a
            # missed wait warning, never correctness.
            await lock.acquire()
            self._holders[name] = LockHolder(
                label or name.value,
                time.monotonic(),
                owner_task=asyncio.current_task(),
            )
            return

        watchdog = asyncio.ensure_future(self._warn_while_waiting(name, label))
        self._waiters[name] = self._waiters.get(name, 0) + 1
        try:
            await lock.acquire()
        finally:
            # Decrement on EVERY exit, including cancellation — a cancelled
            # waiter that stayed counted would make later holders report phantom
            # queued turns forever.
            remaining = self._waiters.get(name, 1) - 1
            if remaining > 0:
                self._waiters[name] = remaining
            else:
                self._waiters.pop(name, None)
            watchdog.cancel()
        self._holders[name] = LockHolder(
            label or name.value,
            time.monotonic(),
            owner_task=asyncio.current_task(),
        )

    async def _warn_while_waiting(
        self, name: ResourceLock, label: Optional[str]
    ) -> None:
        """Periodically report a blocked acquisition, naming the holder."""
        waited = 0.0
        while True:
            await asyncio.sleep(SLOW_WAIT_WARN_SECONDS)
            waited += SLOW_WAIT_WARN_SECONDS
            holder = self._holders.get(name)
            if holder is None:
                held = "holder unknown"
            else:
                held = f"held {holder.held_seconds():.0f}s by {holder.label}"
            logger.warning(
                "%s has been waiting %.0fs for the %s lock (%s). The holder is "
                "still holding; nothing is being dropped.",
                label or "an acquirer",
                waited,
                name.value,
                held,
            )

    async def _warn_while_holding(self, name: ResourceLock) -> None:
        """Periodically report a long-running holder.

        This is the signal that was missing in #2770: it fires while the stall
        is happening, from the holder's side, so an operator does not have to
        infer a wedge from the absence of logs.

        Severity tracks actual harm rather than elapsed time. A long hold with
        nobody waiting is a slow turn — real, worth seeing, but not a fault, and
        emitting WARNING for it would train operators to ignore this exact line.
        A long hold with blocked acquirers is the incident shape, and only then
        may this claim that work is queued — the holder has no other way to know,
        so the count comes from the manager's own waiter bookkeeping.
        """
        escalated = False
        while True:
            await asyncio.sleep(SLOW_HOLD_WARN_SECONDS)
            holder = self._holders.get(name)
            if holder is None:
                return
            waiting = self._waiters.get(name, 0)
            if waiting:
                held_seconds = holder.held_seconds()
                if held_seconds >= BLOCKED_HOLD_ERROR_SECONDS and not escalated:
                    logger.error(
                        "%s has held the %s lock for %.0fs with %d acquirer(s) "
                        "blocked behind it; the hold exceeded the %.0fs "
                        "escalation threshold and is degrading readiness.",
                        holder.label,
                        name.value,
                        held_seconds,
                        waiting,
                        BLOCKED_HOLD_ERROR_SECONDS,
                    )
                    escalated = True
                else:
                    logger.warning(
                        "%s has held the %s lock for %.0fs with %d acquirer(s) "
                        "blocked behind it.",
                        holder.label,
                        name.value,
                        held_seconds,
                        waiting,
                    )
            else:
                logger.info(
                    "%s has held the %s lock for %.0fs (nothing is waiting on "
                    "it).",
                    holder.label,
                    name.value,
                    holder.held_seconds(),
                )

    def is_held(self, name: ResourceLock) -> bool:
        """Best-effort introspection for tests/debug. Not for routing logic."""
        lock = self._locks.get(name)
        return bool(lock and lock.locked())

    def is_owned_by_current_task(self, name: ResourceLock) -> bool:
        """Whether ``name`` is held by the calling coroutine's task.

        This narrow ownership check lets a tool invoked inside a dispatch that
        already owns a non-reentrant resource avoid acquiring that same lock a
        second time. It must not be used to skip acquisition merely because a
        different task currently holds the resource.
        """
        holder = self._holders.get(name)
        if holder is None or holder.owner_task is None:
            return False
        if holder.owner_task is asyncio.current_task():
            return True
        return any(
            inherited_name is name and inherited_token is holder.owner_token
            for inherited_name, inherited_token in self._inherited_ownership.get()
        )

    def bind_current_task_ownership_to_context(self, context: Context) -> None:
        """Grant ``context`` only the lock generations this task now owns.

        ``ContextVar`` values normally copy into every child task, which would
        make accidental background work an owner. This explicit handoff is
        called only by the request-lifecycle invocation task boundary. Tokens
        are matched against the live holder, so a delayed child cannot reuse a
        stale grant after release and reacquisition.
        """

        if not isinstance(context, Context):
            raise TypeError("lock ownership handoff requires a Context")
        current = asyncio.current_task()
        inherited = self._inherited_ownership.get()
        inherited_by_name = dict(inherited)
        grants = tuple(
            (name, holder.owner_token)
            for name, holder in sorted(
                self._holders.items(), key=lambda item: lock_sort_key(item[0])
            )
            if holder.owner_task is current
            or inherited_by_name.get(name) is holder.owner_token
        )
        context.run(self._inherited_ownership.set, grants)

    def holder(self, name: ResourceLock) -> Optional[LockHolder]:
        """Who currently holds ``name``, for diagnostics and tests.

        Best-effort like :meth:`is_held` — never use it for routing.
        """
        return self._holders.get(name)

    def active_hold_diagnostics(self) -> list[dict[str, object]]:
        """Return a health-safe snapshot of every currently held resource.

        The task object is intentionally excluded. Labels are operator-authored
        runtime identifiers and this snapshot is consumed only by authenticated
        detailed health surfaces; the public readiness probe remains aggregate.
        """
        return [
            {
                "resource": name.value,
                "label": holder.label,
                "held_seconds": holder.held_seconds(),
                "blocked_acquirers": self._waiters.get(name, 0),
            }
            for name, holder in sorted(
                self._holders.items(), key=lambda item: lock_sort_key(item[0])
            )
        ]
