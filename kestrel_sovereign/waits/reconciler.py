"""The generic wait reconciler — Wave 2 of the unified-wait epic (#1860).

The principle the SDK's :class:`~kestrel_sdk.tools.MonitorableWaitable`
encodes: *every async waitable should be wakeable*. The legacy
``talon_monitor`` proved the pattern for one kind — it polled the durable
talon job registry every minute, detected terminal-state transitions, and
enqueued one COGNITION signal per transition so the agent woke without the
user polling. This module generalizes that to EVERY registered provider.

The reconciler enumerates each ``MonitorableWaitable`` provider in
``agent.wait_registry``, asks it for its in-flight handles, polls each, and
emits one signal per terminal-state transition — reproducing talon_monitor's
proven TWO-PHASE delivery semantics generically:

  Phase 0 (harvest)  — confirm signals enqueued on a PRIOR tick. The
      reconciler uses fire-and-forget ``enqueue_signal`` so a slow COGNITION
      dispatch never starves other due cron tasks; the handle's task is
      checked at the top of the NEXT tick. Delivered/hard-fail lock the
      transition (``last_signaled_outcome`` set); soft-fail leaves it unset so
      the next tick re-detects + retries with a fresh attempt counter.

  Phase 1 (detect + enqueue) — for each provider's active handle, poll it;
      if terminal and not already signaled (and not in flight), enqueue a
      fresh signal and stash the handle for the next tick's harvest.

The dedup/delivery ledger lives in the ``wait_signal_state`` table
(:class:`WaitSignalStore`), the generic successor to the per-job fields
talon_monitor stashed inside ``jobs.json``. The reconciler instance is held
as a singleton on the agent (``agent._wait_reconciler``) so the in-memory
``_pending_signal_tasks`` map survives across cron ticks.

PERSISTED IS NOT SURFACED (#2922). The ledger used to record the bare
dispatcher status, so a wake that ran, wrote its turn, and reached no human
still read ``ok``. That is the self-reporting failure that hid #2877 for
months: the system claimed success while the observer's chat stayed blank.
Phase 0 now composes each accepted wake's dispatch status with a VISIBILITY
verdict drawn from what the dispatcher observed of the ``signal_completed``
emit — ``ok_queued``, ``ok_unsurfaced``, ``ok_unbound``, or
``ok_visibility_unknown`` — and there is no longer any path that writes a
bare ``ok``.

The ceiling on those verdicts is deliberate. ``queued`` means the SSE event
was accepted by at least one live listener, which is server-side queue
admission; the browser still discards a wake bound to a session other than
the open pane's. No server-side observation can prove a render, so nothing
here claims one, and anything unobserved is reported as unknown rather than
guessed in the flattering direction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sdk.signals import Signal, SignalMode, Visibility
from kestrel_sdk.tools import MonitorableWaitable, ToolResult

from kestrel_sovereign.signals.dispatcher import (
    SURFACE_QUEUED,
    SURFACE_UNSURFACED_STATES,
)
from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore

logger = logging.getLogger(__name__)

# Cap repeated soft-fail retries so a deterministically-broken signal does
# not produce unbounded LLM turns (mirrors talon_monitor's cap). After this
# many attempts, lock the transition and surface a synthetic
# ``max_attempts_exceeded`` delivery status for operator review.
MAX_DELIVERY_ATTEMPTS = 10

# Dispatcher result statuses that mean the wake was ACCEPTED and its turn ran
# — i.e. it was PERSISTED. Formerly ``_DELIVERED_STATES``, and the rename is
# the point of #2922: acceptance by the dispatcher says nothing about whether
# a human can see the result, and calling it "delivered" is what let a
# stranded wake report success for months (#2877).
_PERSISTED_STATES = {"ok", "coalesced"}
_HARD_FAIL_STATES = {"dropped_validation", "dropped_cycle"}

# ---------------------------------------------------------------------------
# Visibility verdicts (#2922)
# ---------------------------------------------------------------------------
#
# ``last_delivery_status`` used to be the bare dispatcher status, so a wake
# with no resolvable origin session — or one whose SSE emit reached nobody —
# still read ``ok``. Persistence and visibility are now composed into ONE
# recorded state (``ok_queued``, ``ok_unsurfaced``, ``ok_unbound``,
# ``ok_visibility_unknown``) so the ledger can never again say "ok" about a
# wake the user never saw.
#
# On the ceiling of these claims: ``QUEUED`` is the strongest verdict that
# exists, and it means only that the ``signal_completed`` event was accepted
# by at least one live listener — server-side queue admission. The browser
# still discards a wake bound to a session other than the open pane's
# (``chat.js``). Nothing here asserts a render, which is why no verdict is
# named "surfaced": when the server cannot tell, it says so.

# The emit reached at least one live consumer's queue. NOT proof of render.
VISIBILITY_QUEUED = "queued"
# The emit demonstrably reached no live consumer (buffered with nobody
# connected, every listener rejected it, the emit raised, or no emitter).
VISIBILITY_UNSURFACED = "unsurfaced"
# No origin session resolved, so the wake was built INTERNAL and there was no
# chat window to surface into. Correct behavior, but not a delivery.
VISIBILITY_UNBOUND = "unbound"
# The dispatcher reported no verdict for this signal — a foreign dispatcher, a
# stand-in predating #2922, or a record lost to a restart. Explicitly unknown
# rather than assumed either way.
VISIBILITY_UNKNOWN = "visibility_unknown"


def compose_delivery_status(dispatch_status: str, visibility: str) -> str:
    """Join a dispatch status and a visibility verdict into one ledger state.

    ``ok`` + ``queued`` -> ``ok_queued``. Deliberately NOT a bare dispatch
    status: the whole failure #2922 fixes is a ledger that recorded ``ok`` for
    a wake nobody could see.
    """
    return f"{dispatch_status}_{visibility}"


class WaitReconciler:
    """Generic two-phase wait→signal reconciler over ``agent.wait_registry``.

    Constructed once per agent (cached on ``agent._wait_reconciler`` via
    :func:`run_wait_reconcile`). Holds the in-memory pending-task map so a
    fire-and-forget enqueue on one tick is harvested on the next.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        # (kind, handle) -> SignalHandle from a prior tick's enqueue_signal,
        # awaiting harvest. Survives across ticks because the reconciler is a
        # singleton on the agent.
        self._pending_signal_tasks: Dict[Tuple[str, str], Any] = {}
        # (kind, handle) -> whether the in-flight wake was bound to an origin
        # chat session (#2922). Kept beside the task map rather than derived at
        # harvest time: the reconciler is the component that decided to build
        # the signal INTERNAL for want of an origin, so it is the honest source
        # for "this wake never had a window to surface into". Populated and
        # dropped in lockstep with ``_pending_signal_tasks`` — an entry missing
        # from that map is already a ``lost_at_restart`` soft-fail that never
        # reaches the visibility accounting.
        self._pending_signal_bindings: Dict[Tuple[str, str], bool] = {}
        agent_id = (
            getattr(agent, "did", None)
            or getattr(agent, "agent_id", None)
            or ""
        )
        # The agent's AsyncDatabase lives behind _raw_storage.db (same path
        # PendingA2AQuestionStore is wired through in kestrel_agent.py).
        raw_storage = getattr(agent, "_raw_storage", None)
        db = getattr(raw_storage, "db", None)
        self._store = WaitSignalStore(db, str(agent_id))
        # Serialize reconcile ticks so the TWO drivers that can call this — the
        # scheduler's ``wait_reconcile`` cron AND the mandatory WaitFeature
        # fallback loop (#2729) — can never race on the shared
        # ``_pending_signal_tasks`` map and both enqueue a wake for the same
        # terminal transition before either records it pending.
        self._reconcile_lock = asyncio.Lock()
        # Monotonic timestamp of the last reconcile tick's start. The
        # WaitFeature fallback driver reads it (via
        # :meth:`seconds_since_last_reconcile`) to stand down while the cron
        # keeps the reconciler fresh. ``None`` until the first tick.
        self._last_reconcile_monotonic: Optional[float] = None

    # ------------------------------------------------------------------

    async def reconcile(self) -> ToolResult:
        """Run one reconcile tick. ACTION — no LLM cost (the COGNITION wake
        comes from the downstream signal, not from this task).

        Serialized (#2729): both the scheduler's ``wait_reconcile`` cron and
        the mandatory WaitFeature fallback loop may drive this, so the lock
        guarantees exactly one tick runs at a time. Two overlapping ticks
        could otherwise both detect the same terminal handle and enqueue a
        duplicate wake before either recorded it pending.
        """
        async with self._reconcile_lock:
            self._last_reconcile_monotonic = time.monotonic()
            return await self._reconcile_once()

    def seconds_since_last_reconcile(self) -> Optional[float]:
        """Monotonic seconds since the last reconcile tick began, or ``None``
        if none has run yet.

        The WaitFeature fallback driver reads this to decide whether to drive a
        tick: while the scheduler ``wait_reconcile`` cron keeps it fresh the
        fallback stands down, so an agent that HAS a scheduler is not
        double-driven, and one that does not still gets reconciled (#2729).
        """
        if self._last_reconcile_monotonic is None:
            return None
        return time.monotonic() - self._last_reconcile_monotonic

    async def _reconcile_once(self) -> ToolResult:
        store = self._store
        transitions: List[Dict[str, Any]] = []
        # persisted = dispatcher returned OK or COALESCED on a PRIOR enqueue,
        # i.e. the wake ran and its turn was written down. NOT a visibility
        # claim — the four counters below split that out (#2922).
        signals_persisted = 0
        signals_queued = 0
        signals_unsurfaced = 0
        signals_unbound = 0
        signals_visibility_unknown = 0
        # hard_fail = permanent rejection (dropped_validation/cycle, cap).
        signals_hard_failed = 0
        # soft_fail = retriable (rate_limit/quiet_hours/failed/raised/lost).
        signals_soft_failed = 0
        signals_enqueued = 0
        signals_skipped_no_dispatcher = 0
        scanned = 0

        dispatcher = getattr(self._agent, "dispatcher", None)

        # --- Phase 0: harvest prior-tick enqueues -----------------------
        # Build the harvest set from BOTH the in-memory tasks AND the durable
        # pending rows (a row with no in-memory task means we restarted
        # mid-flight). Union the keys so neither side is missed.
        harvest_keys: set[Tuple[str, str]] = set(self._pending_signal_tasks)
        for row in await store.list_pending():
            harvest_keys.add((row.kind, row.handle))

        for kind, handle in harvest_keys:
            handle_obj = self._pending_signal_tasks.get((kind, handle))
            now = datetime.now(timezone.utc)

            if handle_obj is None:
                # Durable pending row but no in-memory task — Kestrel
                # restarted mid-flight. The background task died with the
                # parent; we can't know whether the cognition turn fired.
                # Soft-fail (DON'T set signaled_outcome) so the next tick
                # re-detects + re-emits. record_delivery clears pending.
                self._pending_signal_bindings.pop((kind, handle), None)
                await store.record_delivery(
                    kind, handle,
                    delivery_status="lost_at_restart",
                    attempt_at=now,
                )
                signals_soft_failed += 1
                continue

            if not handle_obj.task.done():
                # Still in flight; check again next tick.
                continue

            state = await store.get(kind, handle)
            target = state.pending_signaled_target if state else None

            try:
                result = handle_obj.task.result()
                status_value = result.status.value
                delivery_error = result.error or None
            except Exception as e:  # task raised
                logger.warning(
                    "wait_reconcile: pending signal task raised for "
                    "%s:%s: %s", kind, handle, e,
                )
                status_value = "dispatcher_raised"
                delivery_error = f"{type(e).__name__}: {e}"

            self._pending_signal_tasks.pop((kind, handle), None)
            bound = self._pending_signal_bindings.pop((kind, handle), None)

            if status_value in _PERSISTED_STATES:
                # The dispatcher accepted the wake and its turn ran, so the
                # transition is locked against re-emit. Whether anyone can SEE
                # it is a separate question, answered by the dispatcher's
                # record of the actual ``signal_completed`` emit (#2922) — not
                # inferred from this status, which is what made a stranded
                # wake report ``ok`` for months.
                visibility, surface_status = self._resolve_visibility(
                    dispatcher,
                    signal_id=getattr(handle_obj, "signal_id", None),
                    bound=bound,
                )
                recorded_status = compose_delivery_status(status_value, visibility)
                await store.record_delivery(
                    kind, handle,
                    delivery_status=recorded_status,
                    delivery_error=delivery_error,
                    signaled_outcome=target,
                    attempt_at=now,
                    surface_status=surface_status,
                )
                signals_persisted += 1
                if visibility == VISIBILITY_QUEUED:
                    signals_queued += 1
                elif visibility == VISIBILITY_UNSURFACED:
                    signals_unsurfaced += 1
                elif visibility == VISIBILITY_UNBOUND:
                    signals_unbound += 1
                else:
                    signals_visibility_unknown += 1
                if visibility != VISIBILITY_QUEUED:
                    logger.info(
                        "wait_reconcile: %s:%s persisted (%s) but not shown to "
                        "a live consumer — visibility=%s surface=%s",
                        kind, handle, status_value, visibility,
                        surface_status or "none",
                    )
                transitions.append({
                    "kind": kind, "handle": handle, "outcome": target,
                    "delivery_status": recorded_status,
                    "dispatch_status": status_value,
                    "visibility": visibility,
                    "surface_status": surface_status or "",
                })
            elif status_value in _HARD_FAIL_STATES:
                # Permanent rejection — lock signaled to stop re-emit loops.
                await store.record_delivery(
                    kind, handle,
                    delivery_status=status_value,
                    delivery_error=delivery_error,
                    signaled_outcome=target,
                    attempt_at=now,
                )
                signals_hard_failed += 1
                transitions.append({
                    "kind": kind, "handle": handle, "outcome": target,
                    "delivery_status": status_value,
                    "delivery_error": delivery_error or "",
                })
            else:
                # Soft fail (rate_limit/quiet_hours/failed/dispatcher_raised).
                # Don't set signaled_outcome — next tick re-detects and
                # re-emits with a fresh attempt counter so the dispatcher's
                # coalescing window doesn't swallow the retry as COALESCED.
                await store.record_delivery(
                    kind, handle,
                    delivery_status=status_value,
                    delivery_error=delivery_error,
                    attempt_at=now,
                )
                signals_soft_failed += 1

        # --- Phase 1: detect terminal transitions + enqueue -------------
        # Two complementary wake sources, both routed through the SAME
        # per-handle emit logic (:meth:`_process_handle`) so they never
        # diverge:
        #
        #   (a) IMPLICIT auto-wake — every MonitorableWaitable provider's
        #       active_handles() (talon). Drives unattended completions
        #       without the agent asking.
        #   (b) EXPLICIT watched-waits — handles the agent registered via
        #       wait(target, mode="signal"). Polled via the base provider so
        #       even a poll-only Waitable (e.g. TaskWaitable, which has no
        #       active_handles) is wakeable. This is the half that makes
        #       "every waitable wakeable if it is async" true WITHOUT
        #       auto-waking all tasks (which would self-wake on inbound work).
        registry = getattr(self._agent, "wait_registry", None)
        counters: Dict[str, int] = {
            "signals_enqueued": 0,
            "signals_hard_failed": 0,
            "signals_soft_failed": 0,
            "signals_skipped_no_dispatcher": 0,
        }
        # (kind, handle) processed this tick so a handle that is BOTH
        # monitorable-active AND explicitly watched isn't polled/emitted twice.
        processed: set[Tuple[str, str]] = set()

        # (a) Implicit auto-wake over MonitorableWaitable providers.
        providers = self._monitorable_providers(registry)
        for provider in providers:
            kind = provider.kind
            try:
                handles = await provider.active_handles()
            except Exception as e:
                logger.warning(
                    "wait_reconcile: active_handles() raised for kind %s: %s",
                    kind, e,
                )
                continue

            for handle in handles:
                scanned += 1
                processed.add((kind, handle))
                await self._process_handle(
                    provider, kind, handle, dispatcher, store,
                    transitions, counters,
                )

        # (b) Explicit watched-waits over ANY registered provider.
        if registry is not None:
            for row in await store.list_watched():
                kind, handle = row.kind, row.handle
                if (kind, handle) in processed:
                    # Already handled by the active_handles loop this tick.
                    continue
                provider = registry.get(kind)
                if provider is None:
                    # Provider not registered (feature unloaded) — skip; the
                    # watch row stays so it resumes if the provider returns.
                    continue
                scanned += 1
                processed.add((kind, handle))
                await self._process_handle(
                    provider, kind, handle, dispatcher, store,
                    transitions, counters,
                )

        signals_enqueued = counters["signals_enqueued"]
        signals_hard_failed += counters["signals_hard_failed"]
        signals_soft_failed += counters["signals_soft_failed"]
        signals_skipped_no_dispatcher += counters["signals_skipped_no_dispatcher"]

        parts = [
            f"persisted={signals_persisted}",
            f"enqueued={signals_enqueued}",
        ]
        # Report the visibility split whenever anything was persisted, INCLUDING
        # the zeros. "persisted=1" alone is the sentence that hid #2877; the
        # reader needs to see "queued=0, unsurfaced=1" in the same breath.
        if signals_persisted:
            parts.append(f"queued={signals_queued}")
            if signals_unsurfaced:
                parts.append(f"unsurfaced={signals_unsurfaced}")
            if signals_unbound:
                parts.append(f"unbound={signals_unbound}")
            if signals_visibility_unknown:
                parts.append(
                    f"visibility_unknown={signals_visibility_unknown}"
                )
        if signals_hard_failed:
            parts.append(f"hard_failed={signals_hard_failed}")
        if signals_soft_failed:
            parts.append(f"soft_failed={signals_soft_failed}")
        if signals_skipped_no_dispatcher:
            parts.append(
                f"skipped_no_dispatcher={signals_skipped_no_dispatcher}"
            )
        return ToolResult.ok(
            confirmation=(
                f"Wait reconcile: scanned {scanned} handle(s), "
                + ", ".join(parts)
            ),
            data={
                "scanned": scanned,
                # signals_persisted = wakes CONFIRMED accepted this tick (from
                # a prior tick's enqueues), i.e. the turn ran and was written
                # down. signals_enqueued = this tick's NEW emits awaiting
                # confirmation.
                "signals_persisted": signals_persisted,
                # Back-compat alias. It always counted persistence; #2922 only
                # stopped the name from implying the user saw anything.
                "signals_emitted": signals_persisted,
                # The visibility split of signals_persisted. signals_queued is
                # the CEILING of what the server can attest — the event reached
                # a live consumer's queue, not necessarily a rendered pane.
                "signals_queued": signals_queued,
                "signals_unsurfaced": signals_unsurfaced,
                "signals_unbound": signals_unbound,
                "signals_visibility_unknown": signals_visibility_unknown,
                "signals_enqueued": signals_enqueued,
                "signals_hard_failed": signals_hard_failed,
                "signals_soft_failed": signals_soft_failed,
                "signals_skipped_no_dispatcher": signals_skipped_no_dispatcher,
                "pending_deliveries": len(self._pending_signal_tasks),
                "transitions": transitions,
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_visibility(
        dispatcher: Any,
        *,
        signal_id: Optional[str],
        bound: Optional[bool],
    ) -> Tuple[str, Optional[str]]:
        """Decide what this reconciler may honestly claim about one wake's
        visibility, and return ``(verdict, raw dispatcher surface status)``.

        Two independent facts feed the verdict, and neither is the dispatch
        status:

        * ``bound`` — whether THIS reconciler resolved an origin chat session
          when it built the signal. Unbound wakes are built ``INTERNAL`` on
          purpose (unattended cron/CLI work has no window), so they were never
          candidates for surfacing and must not be counted as delivered to a
          user.
        * the dispatcher's :meth:`~SignalDispatcher.surface_record` — what the
          ``signal_completed`` emit actually did.

        Anything the dispatcher did not observe — no ledger method, no record,
        or a receipt-less ``emit_event`` — is :data:`VISIBILITY_UNKNOWN`. The
        temptation to read a missing record as success is precisely the bug
        (#2877 attempt-2 P2: report unknown, do not assert from ledger
        contents alone).
        """
        record = None
        lookup = getattr(dispatcher, "surface_record", None)
        if callable(lookup) and signal_id:
            try:
                record = lookup(signal_id)
            except Exception as exc:  # a broken ledger is not a verdict
                logger.debug(
                    "surface_record(%r) raised on the dispatcher: %s",
                    signal_id, exc,
                )
                record = None
        raw = getattr(record, "status", None)
        raw = str(raw) if raw is not None else None

        if bound is False:
            return VISIBILITY_UNBOUND, raw
        if raw == SURFACE_QUEUED:
            return VISIBILITY_QUEUED, raw
        if raw in SURFACE_UNSURFACED_STATES:
            return VISIBILITY_UNSURFACED, raw
        # Includes ``not_applicable`` (the dispatcher saw an INTERNAL signal
        # while this reconciler believed it bound one — a contradiction we
        # report as unknown rather than resolve in either direction),
        # ``unknown``, and no record at all.
        return VISIBILITY_UNKNOWN, raw

    # ------------------------------------------------------------------

    async def _process_handle(
        self,
        provider: Any,
        kind: str,
        handle: str,
        dispatcher: Any,
        store: WaitSignalStore,
        transitions: List[Dict[str, Any]],
        counters: Dict[str, int],
    ) -> None:
        """Poll one handle and apply the terminal/dedup/retry/enqueue logic.

        Shared by BOTH the active_handles (implicit auto-wake) loop and the
        watched-handles (explicit signal-wait) loop so the two wake sources
        can never diverge. ``provider`` need only be a base ``Waitable`` —
        the watched path passes poll-only providers here. Mutates ``counters``
        and appends to ``transitions``; a non-terminal handle is a no-op.
        """
        try:
            status = await provider.poll(handle)
        except Exception as e:
            logger.warning(
                "wait_reconcile: poll(%s:%s) raised: %s",
                kind, handle, e,
            )
            return

        if not status.outcome.is_terminal():
            # Non-terminal — leave it (watched handles stay watched).
            return

        # Dedup token: the generic Outcome alone is too coarse — a provider
        # can map several distinct native states onto one Outcome (talon's
        # ``finished_unknown`` and ``failed`` both -> FAILED). The legacy
        # talon_monitor dedup'd on the native status and explicitly allowed a
        # corrected ``finished_unknown -> failed`` to re-signal. So we dedup on
        # the outcome PLUS the provider's native ``status`` (when it exposes one
        # in WaitStatus.data) — preserving one-signal-per-real-transition
        # (codex Wave 2 P2).
        signaled_token = self._signaled_token(status)
        state = await store.get(kind, handle)

        # Application-level dedup: already signaled this transition.
        if state and state.last_signaled_outcome == signaled_token:
            return
        # A prior emit for this handle is still in flight — wait for the
        # next tick's Phase 0 to confirm it before re-emitting.
        if (kind, handle) in self._pending_signal_tasks:
            return

        attempts_so_far = state.last_delivery_attempts if state else 0
        if attempts_so_far >= MAX_DELIVERY_ATTEMPTS:
            # Retry cap reached — lock signaled and surface a synthetic
            # delivery status for operator review.
            await store.record_delivery(
                kind, handle,
                delivery_status="max_attempts_exceeded",
                signaled_outcome=signaled_token,
                attempt_at=datetime.now(timezone.utc),
            )
            counters["signals_hard_failed"] += 1
            transitions.append({
                "kind": kind, "handle": handle, "outcome": status.outcome.value,
                "delivery_status": "max_attempts_exceeded",
                "delivery_attempts": attempts_so_far,
            })
            return

        attempts = attempts_so_far + 1
        origin_session_id = await _provider_origin_session(provider, handle)
        signal = self._build_signal(
            provider, kind, handle, status, attempts,
            origin_session_id=origin_session_id,
            prior_state=state,
        )

        if dispatcher is None or not hasattr(dispatcher, "enqueue_signal"):
            # Without a dispatcher we have NOT woken anyone — don't record
            # signaled so the next tick retries.
            counters["signals_skipped_no_dispatcher"] += 1
            return

        now = datetime.now(timezone.utc)
        await store.record_pending(
            kind, handle,
            signal_id=signal.id,
            target=signaled_token,
            attempts=attempts,
            attempt_at=now,
        )

        try:
            enq = dispatcher.enqueue_signal(signal)
            handle_obj = await enq if asyncio.iscoroutine(enq) else enq
        except Exception as e:
            logger.warning(
                "wait_reconcile: enqueue_signal raised for %s:%s: %s",
                kind, handle, e,
            )
            await store.record_delivery(
                kind, handle,
                delivery_status="dispatcher_raised",
                delivery_error=f"{type(e).__name__}: {e}",
                attempt_at=now,
            )
            counters["signals_soft_failed"] += 1
            return

        self._pending_signal_tasks[(kind, handle)] = handle_obj
        # Remember whether this wake had a chat window to surface into, so the
        # next tick's harvest can tell "nobody was listening" from "there was
        # nowhere to listen" (#2922) instead of collapsing both into ``ok``.
        self._pending_signal_bindings[(kind, handle)] = (
            signal.visibility != Visibility.INTERNAL
        )
        counters["signals_enqueued"] += 1

    @staticmethod
    def _signaled_token(status: Any) -> str:
        """Dedup token for a terminal poll.

        The generic ``Outcome`` plus the provider's native ``status`` (when it
        exposes one in ``WaitStatus.data``), so providers that collapse several
        native terminal states into one Outcome still get one signal per real
        transition (talon ``finished_unknown`` -> ``failed``). Falls back to
        the bare outcome value for providers that expose no native status.
        """
        native = (status.data or {}).get("status")
        if native:
            return f"{status.outcome.value}:{native}"
        return status.outcome.value

    @staticmethod
    def _monitorable_providers(registry: Any) -> List[MonitorableWaitable]:
        """Return the registry's providers that support enumeration.

        Detected structurally per the SDK's protocol contract — a poll-only
        provider (no ``active_handles``) is valid against base ``Waitable``
        and is simply skipped here.
        """
        if registry is None:
            return []
        providers: List[MonitorableWaitable] = []
        for kind in registry.kinds():
            provider = registry.get(kind)
            if provider is not None and isinstance(provider, MonitorableWaitable):
                providers.append(provider)
        return providers

    def _build_signal(
        self,
        provider: Any,
        kind: str,
        handle: str,
        status: Any,
        attempts: int,
        *,
        origin_session_id: Optional[str] = None,
        prior_state: Any = None,
    ) -> Signal:
        """Build a COGNITION signal envelope for a terminal transition.

        The provider's own ``signal`` name is used when it declares one
        (e.g. TalonWaitable -> ``talon.job_complete``); otherwise the
        generic ``wait.complete`` source. The provider's WaitStatus.data is
        spread underneath the generic kind/handle/outcome/summary keys so
        kind-specific templates (talon's) still find their fields.

        ``prior_state`` is this handle's ledger row as read at the top of
        :meth:`_process_handle` — the source of the delivery-provenance keys
        (#3105). A wake carries its subject's state AND its own: without the
        attempt count in the payload, a retry after a soft-failed dispatch is
        byte-identical to a first delivery, and an orchestrator woken 90
        minutes after a job ended reasonably reads it as news.

        ``origin_session_id`` is the session that REGISTERED the work,
        resolved by :func:`_provider_origin_session` from the provider's own
        local record — never read out of ``status.data`` (#2877). It routes
        the wake back into that chat window instead of opening a fresh
        session for it. The spread payload's own ``origin_session_id`` key,
        if a provider set one, is OVERWRITTEN with the resolved value so a
        provider that forwards untrusted third-party data (``A2AWaitable``
        spreads a peer's task result verbatim) cannot smuggle routing
        authority through the payload, nor leave a misleading value in the
        signal log.

        Binding the session is only half of "the user can see it". A bound
        wake is also built ``USER_VISIBLE`` so the dispatcher emits the
        ``signal_completed`` SSE event after the turn logs; at the default
        ``INTERNAL`` the dispatcher log-only's it and the open chat stays
        blank until a manual refresh — the persisted-but-unsurfaced half of
        the same bug. The rendered body comes from the source's
        ``result_summary`` callback (the frontend requires BOTH), which
        ``talon.job_complete`` and ``restart.completed`` supply.

        An origin-less wake stays ``INTERNAL``: unattended cron/CLI dispatch
        has no chat window to surface into, and the notifications SSE stream
        is pinned to the agent rather than to a session, so emitting would
        paint a turn into whichever pane happens to be open.
        """
        source = getattr(provider, "signal", None) or "wait.complete"
        prior_status = getattr(prior_state, "last_delivery_status", None)
        prior_at = getattr(prior_state, "last_delivery_attempt_at", None)
        payload: Dict[str, Any] = {
            **(status.data or {}),
            "kind": kind,
            "handle": handle,
            "outcome": status.outcome.value,
            "summary": status.summary,
            "origin_session_id": origin_session_id or "",
            # DELIVERY PROVENANCE (#3105). A retry after a soft-failed
            # dispatch re-describes the same terminal transition, so its
            # payload is byte-identical to the first attempt's — the reader
            # cannot tell "this job just finished" from "this wake's third
            # dispatch". The attempt count already existed here; it went into
            # ``dedupe_key`` for the DISPATCHER and was withheld from the
            # AGENT, which is the only party that has to decide whether the
            # wake is news. These keys are the same fact, addressed to the
            # reader. Written AFTER the ``status.data`` spread for the same
            # reason ``origin_session_id`` is: a provider that forwards
            # third-party data must not be able to understate an attempt
            # count and make a retry read as a first delivery.
            "delivery_attempt": attempts,
            "delivery_max_attempts": MAX_DELIVERY_ATTEMPTS,
            # Empty on a first attempt; on a retry, how the PREVIOUS dispatch
            # of this same transition ended and when it was tried.
            "delivery_previous_status": str(prior_status or "") if attempts > 1 else "",
            "delivery_previous_attempt_at": (
                str(prior_at or "") if attempts > 1 else ""
            ),
        }
        target_agent = (
            getattr(self._agent, "did", None)
            or getattr(self._agent, "agent_id", None)
            or ""
        )
        return Signal(
            source=source,
            kind="inbound",
            mode=SignalMode.COGNITION,
            payload=payload,
            target_agent=str(target_agent),
            session_id=origin_session_id or None,
            visibility=(
                Visibility.USER_VISIBLE
                if origin_session_id
                else Visibility.INTERNAL
            ),
            # Unique per attempt so a retry after a soft failure isn't
            # swallowed by the dispatcher's coalescing window as COALESCED
            # against the prior failed attempt (talon_monitor codex round 1
            # P1). Application-level dedup via last_signaled_outcome still
            # prevents redundant emits across ticks.
            dedupe_key=(
                f"{kind}:{handle}:{self._signaled_token(status)}:attempt-{attempts}"
            ),
        )


async def run_wait_reconcile(agent: Any) -> ToolResult:
    """Lazily build/cache the agent's singleton reconciler and run one tick.

    This is the entry point the ``cron.wait_reconcile`` ACTION handler calls.
    Caching on ``agent._wait_reconciler`` preserves the in-memory pending-task
    map across cron ticks so a fire-and-forget enqueue is harvested next tick.
    """
    reconciler = getattr(agent, "_wait_reconciler", None)
    if reconciler is None:
        reconciler = WaitReconciler(agent)
        agent._wait_reconciler = reconciler
    return await reconciler.reconcile()


def _get_reconciler(agent: Any) -> "WaitReconciler":
    """Lazily build/cache the agent's singleton reconciler WITHOUT running a
    tick — the watch-registration path needs its store but not a reconcile."""
    reconciler = getattr(agent, "_wait_reconciler", None)
    if reconciler is None:
        reconciler = WaitReconciler(agent)
        agent._wait_reconciler = reconciler
    return reconciler


async def _provider_owns_handle(provider: Any, handle: str) -> Optional[bool]:
    """Ask ``provider`` whether ``handle`` belongs to its namespace.

    Ownership validation is OPTIONAL and structural (mirrors how
    ``MonitorableWaitable`` is detected): a provider that can cheaply tell a
    real handle from a foreign one exposes an async ``owns_handle(handle)``
    returning

      * ``True``  — the handle is definitely this provider's,
      * ``False`` — the handle definitely is NOT (reject the watch),
      * ``None``  — cannot determine right now (backend unavailable, etc.),
        so the caller fails OPEN and allows the watch.

    A provider with no ``owns_handle`` is treated as ``None`` (unverifiable).
    A provider bug is swallowed to ``None`` — an ownership check must never be
    the thing that blocks an otherwise-valid watch.
    """
    verify = getattr(provider, "owns_handle", None)
    if not callable(verify):
        return None
    try:
        result = verify(handle)
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as exc:  # provider bug — don't block on a broken check
        logger.debug(
            "owns_handle(%r) raised on provider %s: %s",
            handle, getattr(provider, "kind", "?"), exc,
        )
        return None
    if result is None:
        return None
    return bool(result)


async def _provider_origin_session(provider: Any, handle: str) -> Optional[str]:
    """Ask ``provider`` which chat session registered ``handle`` (#2877).

    The reconciler's ONLY trusted source for wake routing. Optional and
    structural, exactly like ``owns_handle``: a provider that records the
    dispatching session on its own local job/task record exposes
    ``origin_session_id(handle)`` returning that session id, or ``None`` when
    the work was registered unattended (CLI, cron, scheduler) or predates the
    field. A provider with no such method returns ``None`` here.

    Why a method and not a ``WaitStatus.data`` key: the reconciler spreads a
    provider's poll data into the signal payload verbatim, and
    :class:`~kestrel_sovereign.features.peers.wait_provider.A2AWaitable`
    spreads a *peer's* returned task result into that same dict. A payload key
    would therefore let a remote peer choose which local chat session a
    COGNITION wake resumes into — and, since a bound wake is built
    USER_VISIBLE, get its text painted into that window. Routing authority
    stays with locally-owned provider state.

    A provider bug is swallowed to ``None``: failing to resolve an origin must
    degrade to the pre-#2877 behavior (a system-initiated wake), never block
    the wake itself.
    """
    resolve = getattr(provider, "origin_session_id", None)
    if not callable(resolve):
        return None
    try:
        result = resolve(handle)
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as exc:  # provider bug — wake system-initiated instead
        logger.debug(
            "origin_session_id(%r) raised on provider %s: %s",
            handle, getattr(provider, "kind", "?"), exc,
        )
        return None
    if not isinstance(result, str) or not result.strip():
        return None
    return result.strip()


async def register_wait_watch(agent: Any, ref: str) -> None:
    """Register an explicit watch on ``ref`` so the reconciler wakes the agent
    when that handle reaches a terminal state.

    This is the durable backing for ``wait(target, mode="signal")``: it parses
    the ``"<kind>:<handle>"`` ref, validates the kind is registered in
    ``agent.wait_registry`` (so a typo'd or unloaded kind fails LOUD rather
    than silently never waking), validates provider/handle OWNERSHIP whenever
    the provider can (#2729), and records ``watching=1`` on the reconciler's
    store. The reconciler's watched-handles loop then polls it every tick —
    works for ANY Waitable, including poll-only providers (TaskWaitable) that
    have no ``active_handles`` for the implicit auto-wake path.

    Provider *availability* (a kind is registered) and provider *ownership*
    (the handle actually belongs to that provider) are distinct. A registered
    kind re-arms across restart; a handle that a foreign provider owns is a
    mismatch that must fail synchronously HERE rather than becoming a durable
    watch that the reconciler later converts into a misleading terminal
    ``wait.complete`` failure. The canonical example (#2729): an outbound A2A
    task id registered as ``task:<id>`` — the ``task`` provider is the LOCAL
    background TaskStore, does not own the outbound id, and its poll would
    read "not found" and emit a false terminal failure. Ownership validation
    rejects it up front and, when another registered provider DOES own the
    handle, names it so the caller can retry with the right kind.

    Raises:
        ValueError: on a malformed ref, a missing ``wait_registry``, a
            ``kind`` with no registered provider, or a handle the named
            provider affirmatively does not own.
    """
    from kestrel_sovereign.waits.engine import parse_ref

    kind, handle = parse_ref(ref)
    registry = getattr(agent, "wait_registry", None)
    if registry is None:
        raise ValueError(
            "wait engine unavailable: no wait_registry on the agent"
        )
    provider = registry.get(kind)
    if provider is None:
        known = ", ".join(registry.kinds()) or "(none registered)"
        raise ValueError(
            f"no wait provider for kind {kind!r}; known kinds: {known}"
        )

    # Ownership validation (#2729): reject a handle the named provider does
    # not own BEFORE it becomes a durable watch. Only an affirmative False
    # (the provider is sure the handle is foreign) blocks — an unverifiable
    # None fails open so providers that can't cheaply check stay permissive.
    owned = await _provider_owns_handle(provider, handle)
    if owned is False:
        hint = ""
        for other in registry.kinds():
            if other == kind:
                continue
            other_provider = registry.get(other)
            if other_provider is None:
                continue
            if await _provider_owns_handle(other_provider, handle) is True:
                hint = f" — did you mean {other}:{handle}?"
                break
        raise ValueError(
            f"handle {handle!r} is not a valid {kind!r} wait handle "
            f"(the {kind!r} provider does not own it){hint}"
        )

    reconciler = _get_reconciler(agent)
    await reconciler._store.start_watch(kind, handle)
