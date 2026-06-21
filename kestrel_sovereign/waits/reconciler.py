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
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sdk.signals import Signal, SignalMode
from kestrel_sdk.tools import MonitorableWaitable, ToolResult

from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore

logger = logging.getLogger(__name__)

# Cap repeated soft-fail retries so a deterministically-broken signal does
# not produce unbounded LLM turns (mirrors talon_monitor's cap). After this
# many attempts, lock the transition and surface a synthetic
# ``max_attempts_exceeded`` delivery status for operator review.
MAX_DELIVERY_ATTEMPTS = 10

# Dispatcher result statuses, classified exactly as talon_monitor did.
_DELIVERED_STATES = {"ok", "coalesced"}
_HARD_FAIL_STATES = {"dropped_validation", "dropped_cycle"}


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

    # ------------------------------------------------------------------

    async def reconcile(self) -> ToolResult:
        """Run one reconcile tick. ACTION — no LLM cost (the COGNITION wake
        comes from the downstream signal, not from this task)."""
        store = self._store
        transitions: List[Dict[str, Any]] = []
        # delivered = dispatcher returned OK or COALESCED on a PRIOR enqueue.
        signals_delivered = 0
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

            if status_value in _DELIVERED_STATES:
                # Lock the transition so we don't re-emit it.
                await store.record_delivery(
                    kind, handle,
                    delivery_status=status_value,
                    delivery_error=delivery_error,
                    signaled_outcome=target,
                    attempt_at=now,
                )
                signals_delivered += 1
                transitions.append({
                    "kind": kind, "handle": handle, "outcome": target,
                    "delivery_status": status_value,
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
            f"delivered={signals_delivered}",
            f"enqueued={signals_enqueued}",
        ]
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
                # signals_emitted = deliveries CONFIRMED this tick (from a
                # prior tick's enqueues). signals_enqueued = this tick's NEW
                # emits awaiting confirmation.
                "signals_emitted": signals_delivered,
                "signals_enqueued": signals_enqueued,
                "signals_hard_failed": signals_hard_failed,
                "signals_soft_failed": signals_soft_failed,
                "signals_skipped_no_dispatcher": signals_skipped_no_dispatcher,
                "pending_deliveries": len(self._pending_signal_tasks),
                "transitions": transitions,
            },
        )

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
        signal = self._build_signal(provider, kind, handle, status, attempts)

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
    ) -> Signal:
        """Build a COGNITION signal envelope for a terminal transition.

        The provider's own ``signal`` name is used when it declares one
        (e.g. TalonWaitable -> ``talon.job_complete``); otherwise the
        generic ``wait.complete`` source. The provider's WaitStatus.data is
        spread underneath the generic kind/handle/outcome/summary keys so
        kind-specific templates (talon's) still find their fields.
        """
        source = getattr(provider, "signal", None) or "wait.complete"
        payload: Dict[str, Any] = {
            **(status.data or {}),
            "kind": kind,
            "handle": handle,
            "outcome": status.outcome.value,
            "summary": status.summary,
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


async def register_wait_watch(agent: Any, ref: str) -> None:
    """Register an explicit watch on ``ref`` so the reconciler wakes the agent
    when that handle reaches a terminal state.

    This is the durable backing for ``wait(target, mode="signal")``: it parses
    the ``"<kind>:<handle>"`` ref, validates the kind is registered in
    ``agent.wait_registry`` (so a typo'd or unloaded kind fails LOUD rather
    than silently never waking), and records ``watching=1`` on the reconciler's
    store. The reconciler's watched-handles loop then polls it every tick —
    works for ANY Waitable, including poll-only providers (TaskWaitable) that
    have no ``active_handles`` for the implicit auto-wake path.

    Raises:
        ValueError: on a malformed ref, a missing ``wait_registry``, or a
            ``kind`` with no registered provider.
    """
    from kestrel_sovereign.waits.engine import parse_ref

    kind, handle = parse_ref(ref)
    registry = getattr(agent, "wait_registry", None)
    if registry is None:
        raise ValueError(
            "wait engine unavailable: no wait_registry on the agent"
        )
    if registry.get(kind) is None:
        known = ", ".join(registry.kinds()) or "(none registered)"
        raise ValueError(
            f"no wait provider for kind {kind!r}; known kinds: {known}"
        )
    reconciler = _get_reconciler(agent)
    await reconciler._store.start_watch(kind, handle)
