"""SignalDispatcher — the runtime engine.

# Signals vs hooks — read this before adding either

Signals ORIGINATE work. Hooks INTERCEPT work. They sit on opposite
sides of the cognition lifecycle and conflating them is the original
sin behind half the bugs in the early signal-dispatcher epic.

* A **signal** wakes the bird (COGNITION) or runs a side effect
  (ACTION/ARTIFACT). It enters the dispatcher pipeline below and
  ends in either a turn (COGNITION) or a registered handler
  (ACTION / ARTIFACT). Heartbeat ticks, scheduled cron firings,
  A2A task-completion notifications, and external webhooks are all
  signals.
* A **hook** is invoked DURING work the bird is already doing —
  PRE_TOOL_USE, POST_TOOL_USE, USER_PROMPT_SUBMIT, STOP. Hooks live
  in `kestrel_sovereign/hooks/manager.py` and modify the flow of an
  in-flight turn (e.g. asking the user to approve a tool call
  before it runs).

The arrow is one-way: a signal can produce a turn that fires hooks;
a hook never produces a signal. If you find yourself reaching for
the dispatcher from inside a hook, you almost certainly want to
modify the in-flight turn instead. If you find yourself reaching
for the hook system from inside a signal handler, you almost
certainly want to add a downstream signal source.

The textbook example of getting this wrong: approval. Approval is
"the bird is paused mid-turn waiting for the user to click yes/no."
That is a HOOK (it suspends an in-flight turn until a gate
releases), not a SIGNAL (which would start a new turn — defeating
the point and racing the original turn). Approval lives in the
hook system; the dispatcher does not own it.

# The pipeline

Per SIGNAL_DISPATCHER.md §"The dispatcher contract":

    1. validate against registration
    2. append-and-cycle-check  (worked examples in SIGNAL_SOURCES_GUIDE.md)
    3. quiet-hours check
    4. coalescing check
    5. rate-limit check
    6. acquire registered resource locks (lex order via OrderedLockManager)
    7. route by mode
       - ACTION    → handler(payload)
       - ARTIFACT  → artifact_handler(signal)
       - COGNITION → render prompt_template → process_input(prompt)
                     CONVERSATION lock acquired by the turn lifecycle,
                     NOT here. Sources are FORBIDDEN from declaring
                     CONVERSATION in their `resources` (registry rejects
                     at register time).
    8. release locks (reverse order; handled by lock manager context)
    9. log to signal_log; if visibility != INTERNAL also emit a
       `signal_completed` SSE event AFTER the log write succeeds
       (consumers correlate by signal_id; emitting before/without
       the log entry would mislead them).

`enqueue_signal` returns an awaitable handle backed by the agent's
existing background task tracker — NEVER a raw `asyncio.create_task`.
The lint test at `tests/unit/test_signals_no_raw_create_task.py`
enforces this for the whole repo.

# Where to learn more

- `docs/architecture/SIGNAL_DISPATCHER.md` — the design spec
  (architecture decisions, concerns, classification table).
- `docs/architecture/SIGNAL_SOURCES_GUIDE.md` — the operator
  guide for adding a new signal source (registration walkthrough,
  cycle-detection worked examples, redaction patterns).
- `kestrel_sovereign/signals/sources/` — every cron task, A2A
  completion, webhook source the agent listens to. `grep` here to
  see exactly what wakes the bird.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, List, Optional, Protocol
from zoneinfo import ZoneInfo

from kestrel_sdk.signals import (
    AttentionPolicy,
    CausationFrame,
    ResourceLock,
    Signal,
    SignalHandle,
    SignalMode,
    SignalResult,
    SourceRegistration,
    Status,
    Urgency,
    Visibility,
)
from kestrel_sovereign.signals.constitution_canary import (
    CanaryStatus,
    build_canary_instruction,
    derive_canary,
)
from kestrel_sovereign.signals.constitution_metrics import (
    record_doctrine_bundle_drift,
    record_echo_missing,
    record_echo_verified,
)
from kestrel_sovereign.features.storage_access import resolve_agent_privacy_config
from kestrel_sovereign.storage.privacy_wrapper import (
    _resolve_transition_lock,
    optional_transition_lock,
)
from kestrel_sovereign.signals.lock_manager import OrderedLockManager
from kestrel_sovereign.signals.registry import RegistrationError, SourceRegistry
from kestrel_sovereign.signals.durable import (
    FAILED,
    DurableConsumerRegistration,
    DurableDelivery,
    DurableSignalStore,
)
from kestrel_sovereign.signals.store import SignalLogStore
from kestrel_sovereign.storage.db.write_audit import (
    capture_write_queries,
    requested_handler_write_audit_callback,
    suppress_write_audit,
)
from kestrel_sovereign.telemetry import (
    KESTREL_AGENT_NAME,
    OI_SPAN_KIND,
    OI_SPAN_KIND_CHAIN,
    optional_span,
)

logger = logging.getLogger(__name__)


# Durable event payloads are intentionally distinct from the runtime signal
# envelope.  This marker preserves an observable event/consumer handoff while
# proving that a volatile privacy mode did not retain user-authored content.
_DURABLE_PRIVACY_GATED_MARKER = "_privacy_gated"


_PROMPT_TEMPLATE_HASH_ATTR = "_kestrel_prompt_template_hash"


# ---------------------------------------------------------------------------
# Constitutional injection — kestrel-sovereign#1137 chunk 1G
# ---------------------------------------------------------------------------


@dataclass
class _ConstitutionAudit:
    """Per-dispatch constitutional-injection audit record.

    Threaded from `_route_under_locks` through `_success` / `_fail` to
    `_log_safe`, where its fields land in `signal_log` (chunk 1C).
    Defaults reflect "ACTION/ARTIFACT or COGNITION with
    `constitution_injection='none'`": all None, status NOT_REQUIRED.
    """

    constitution_hash: Optional[str] = None
    doctrine_bundle_hash: Optional[str] = None
    echo_canary_status: CanaryStatus = CanaryStatus.NOT_REQUIRED
    injected_clauses: Optional[List[str]] = None
    dropped_clauses: Optional[List[str]] = None
    # Set when bundle-hash drift was detected; the dispatcher returns
    # DROPPED_VALIDATION but still writes the (mismatched) hashes to
    # signal_log so an auditor can see exactly what diverged.
    drift_error: Optional[str] = None
    # Pre-derived canary value (populated before the LLM call when
    # `require_constitution_echo=True`) so the post-dispatch verifier
    # checks against the same token the model saw embedded in its
    # prompt. The dispatcher derives once and reuses; deriving again
    # post-dispatch with a fresh nonce would race the model against
    # a moving target.
    canary: Optional[str] = None


@dataclass(frozen=True)
class _DurableSignalProjection:
    """The durable event representation and its initial selector contract.

    Payload-eliding privacy modes deliberately retain no selector-visible
    payload in their durable event.  Existing waits still need their
    normalized in-memory event to be materialized atomically at dispatch
    time; later registrations and restart backfill must see only the durable
    representation.  Anonymized payloads, conversely, are replayable and so
    selectors must use that stored projection from the first delivery.
    """

    signal: Signal
    payload_elided: bool = False


@dataclass
class _TransientDurableHandoff:
    """Process-local payload for one initially matched volatile delivery.

    Volatile privacy modes retain only a marker in ``durable_signal_events``.
    This handoff lets a worker in the same live dispatcher receive the
    normalized payload that selected it, while retries after a lease expiry or
    a process restart correctly fall back to the durable marker.
    """

    payload: Any
    retention_until: datetime
    expires_at: datetime


def _agent_accepts_kwarg(callable_: Any, name: str) -> bool:
    """Return True iff `callable_` declares `name` as a parameter or
    accepts arbitrary kwargs (`**kwargs`). Used to feature-detect
    optional process_input kwargs without try/except TypeError, which
    would also swallow runtime errors raised inside the call.
    """
    try:
        sig = inspect.signature(callable_)
    except (TypeError, ValueError):
        # Builtins or C-extension callables: assume permissive.
        return True
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name:
            return True
    return False


def _audit_to_log_kwargs(audit: Optional[_ConstitutionAudit]) -> dict:
    """Convert audit to the kwargs `SignalLogStore.append` expects.

    A None audit (the legacy ACTION/ARTIFACT path that doesn't go
    through constitutional injection) maps to all-None kwargs, which
    `store.append` documented as "no system prompt path"."""
    if audit is None:
        return {}
    return {
        "constitution_hash": audit.constitution_hash,
        "doctrine_bundle_hash": audit.doctrine_bundle_hash,
        "echo_canary_status": audit.echo_canary_status.value,
        "injected_clauses": audit.injected_clauses,
        "dropped_clauses": audit.dropped_clauses,
    }


DEFAULT_TTL = 5
DEFAULT_COALESCING_WINDOW = timedelta(seconds=5)


# ---------------------------------------------------------------------------
# Agent seam
# ---------------------------------------------------------------------------


class DispatcherAgent(Protocol):
    """The slice of `KestrelAgent` the dispatcher needs.

    Defined as a Protocol so tests can pass a duck-typed stand-in without
    constructing a full agent. Sovereign's `KestrelAgent` satisfies this
    structurally (it has `did`, `process_input`, `_track_background_task`).
    """

    @property
    def did(self) -> str: ...

    async def process_input(self, prompt: str) -> Any: ...

    def _track_background_task(
        self, coro: Coroutine, *, name: str
    ) -> asyncio.Task: ...


# ---------------------------------------------------------------------------
# Coalescing / rate-limit state (in-memory, per-source)
# ---------------------------------------------------------------------------


class _CoalescingState:
    """Last-seen `dedupe_key` per source within the coalescing window."""

    def __init__(self) -> None:
        self._seen: dict[str, dict[str, datetime]] = {}

    def check_and_record(
        self,
        source: str,
        dedupe_key: str,
        window: timedelta,
        *,
        now: datetime,
    ) -> bool:
        """Returns True if this signal should be coalesced (i.e. dropped
        because a same-keyed signal was dispatched within `window`).
        """
        per_source = self._seen.setdefault(source, {})
        last = per_source.get(dedupe_key)
        if last is not None and (now - last) < window:
            return True
        per_source[dedupe_key] = now
        # Opportunistic prune to bound memory.
        cutoff = now - window
        for key, t in list(per_source.items()):
            if t < cutoff:
                del per_source[key]
        return False

    def reset(self) -> None:
        """Drop all remembered dedupe keys. Called on host-resume: the
        wall-clock windows expired while the process was suspended, so the
        stored timestamps are stale and the next same-keyed signal should be
        treated as fresh."""
        self._seen.clear()


class _RateLimitState:
    """Sliding window of dispatch timestamps per source."""

    def __init__(self) -> None:
        self._times: dict[str, deque[float]] = {}

    def check_and_record(
        self,
        source: str,
        rate_limit,
        *,
        now: float,
    ) -> bool:
        """Returns True if this signal should be dropped for rate limit."""
        times = self._times.setdefault(source, deque())
        # Prune anything older than 1 hour.
        cutoff_hour = now - 3600
        while times and times[0] < cutoff_hour:
            times.popleft()

        if rate_limit.per_hour is not None and len(times) >= rate_limit.per_hour:
            return True

        cutoff_minute = now - 60
        recent_minute = sum(1 for t in times if t >= cutoff_minute)
        if rate_limit.per_minute is not None and recent_minute >= rate_limit.per_minute:
            return True

        if rate_limit.burst is not None:
            cutoff_burst = now - 1.0  # 1-second burst window
            recent_burst = sum(1 for t in times if t >= cutoff_burst)
            if recent_burst >= rate_limit.burst:
                return True

        times.append(now)
        return False

    def reset(self) -> None:
        """Drop all recorded dispatch timestamps. Called on host-resume:
        these are ``time.monotonic()`` values, which DON'T advance during
        system suspend, so after a long sleep the sliding window is frozen
        and would wrongly report quotas as saturated. The real hour elapsed
        while asleep, so starting the window fresh is the correct state."""
        self._times.clear()


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------


class SignalDispatcher:
    """Single instance per agent. Holds the registry, lock manager, store,
    and per-source state for coalescing and rate limiting.

    The dispatcher does NOT pre-acquire `CONVERSATION` for COGNITION sources.
    The shared turn lifecycle (Phase 2) is the sole owner; sources are
    rejected at registration time if they declare it.
    """

    def __init__(
        self,
        *,
        agent: DispatcherAgent,
        registry: SourceRegistry,
        lock_manager: OrderedLockManager,
        store: SignalLogStore,
        durable_store: Optional[DurableSignalStore] = None,
        ttl: int = DEFAULT_TTL,
        coalescing_window_default: timedelta = DEFAULT_COALESCING_WINDOW,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._agent = agent
        self._registry = registry
        self._locks = lock_manager
        self._store = store
        # Keep outcome/audit persistence and pending-delivery persistence
        # distinct.  Existing embeddings/tests construct only SignalLogStore;
        # deriving the durable store from its backend preserves that seam while
        # retaining one database transaction domain for each agent.
        self._durable_store = durable_store or DurableSignalStore(store.backend)
        self._durable_initialized = False
        self._durable_init_lock = asyncio.Lock()
        # This is intentionally dispatcher-local.  It is not a second durable
        # queue and must disappear on shutdown/restart; ``delivery_id`` is
        # globally unique and every dispatcher owns exactly one agent scope.
        self._transient_durable_handoffs: dict[str, _TransientDurableHandoff] = {}
        self._transient_durable_handoff_timers: dict[str, asyncio.TimerHandle] = {}
        self._ttl = ttl
        self._default_window = coalescing_window_default
        self._clock = clock
        self._coalescing = _CoalescingState()
        self._rate = _RateLimitState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_resume(self, gap_seconds: float) -> None:
        """Re-anchor throttling state after a host suspend/resume (#1545).

        Coalescing keys off wall-clock and rate-limiting off ``monotonic``,
        so across a suspend they disagree: coalescing windows expire (a
        repeat signal fires again) while the monotonic rate-limit window is
        frozen (quotas look saturated though the hour really elapsed).
        Clearing both restores a consistent, correct post-sleep baseline.

        Invoked by the ``system.resumed`` ACTION handler, which the
        ResumeMonitor dispatches once per detected suspend.
        """
        self._rate.reset()
        self._coalescing.reset()
        logger.info(
            "Dispatcher throttling windows re-anchored after ~%.0fs host suspend",
            gap_seconds,
        )

    async def dispatch_signal(
        self, signal: Signal, *, source_event_id: Optional[str] = None
    ) -> SignalResult:
        """Awaits the full lifecycle. Used by callers that need the result
        (scheduler, heartbeat). Always returns a `SignalResult` — failures
        are encoded as `Status.FAILED` with `error` set, never raised."""
        start = time.monotonic()
        # Publish the in-flight signal for the duration of this dispatch so
        # code running inside handlers/turns (e.g. the Talon coordinator's
        # orchestrator/workflow correlation stamping, kestrel-talon#53) can
        # observe the envelope that drove it. Per-task ContextVar —
        # concurrent enqueue_signal dispatches stay isolated.
        from kestrel_sovereign.signals.context import (
            reset_current_signal,
            set_current_signal,
        )

        ctx_token = set_current_signal(signal)
        try:
            return await self._run(
                signal,
                start,
                source_event_id=(
                    source_event_id
                    if source_event_id is not None
                    else getattr(signal, "source_event_id", None)
                ),
            )
        except Exception as e:
            # Defensive — every failure path inside _run should already
            # produce a SignalResult. If we land here, log it loudly.
            logger.exception(
                "Dispatcher pipeline raised for signal %s (source=%s); "
                "this is a bug — failures should be captured as SignalResult",
                signal.id,
                signal.source,
            )
            return SignalResult(
                signal_id=signal.id,
                status=Status.FAILED,
                mode=signal.mode,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            reset_current_signal(ctx_token)

    async def enqueue_signal(
        self, signal: Signal, *, source_event_id: Optional[str] = None
    ) -> SignalHandle:
        """Returns immediately with a tracked handle. The dispatch runs as
        an agent-owned background task; exceptions are logged not swallowed,
        and the task is cancellable via the agent's shutdown path."""
        coro = self.dispatch_signal(signal, source_event_id=source_event_id)
        task = self._agent._track_background_task(
            coro, name=f"signal_dispatch:{signal.source}:{signal.id}"
        )
        return SignalHandle(signal_id=signal.id, task=task)

    async def initialize_durable_delivery(self) -> None:
        """Initialize the durable consumer ledger exactly once.

        Agent boot calls this eagerly, while the guard in ``_run`` preserves
        compatibility for embedding/test dispatchers that predate the ledger.
        Initialization is complete before the first event is persisted.
        """
        if self._durable_initialized:
            return
        async with self._durable_init_lock:
            if not self._durable_initialized:
                await self._durable_store.initialize()
                self._durable_initialized = True

    async def register_durable_consumer(
        self, registration: DurableConsumerRegistration
    ) -> None:
        """Register a scoped durable consumer for normalized signals.

        Agent scope is authorization, not an advisory filter: a dispatcher
        owns one DID and refuses to register a consumer for any other tenant.
        """
        if registration.agent_id != self._agent.did:
            raise PermissionError(
                "Durable consumer agent_id must match this dispatcher's agent"
            )
        await self.initialize_durable_delivery()
        await self._durable_store.register_consumer(registration)

    async def claim_durable_delivery(
        self, *, consumer_id: str, executor_id: str
    ) -> Optional[DurableDelivery]:
        """Atomically claim one delivery for this agent-scoped consumer."""
        await self.initialize_durable_delivery()
        self._discard_expired_transient_durable_handoffs()
        delivery = await self._durable_store.claim_delivery(
            agent_id=self._agent.did,
            consumer_id=consumer_id,
            executor_id=executor_id,
        )
        if delivery is None:
            return None
        handoff = self._transient_durable_handoffs.get(delivery.delivery_id)
        if handoff is None:
            return delivery
        # A claim is the live consumer handoff.  Retain the payload through a
        # retry, but never past the lease it was handed to this executor.
        if delivery.lease_expires_at is not None:
            handoff.expires_at = min(
                handoff.retention_until, delivery.lease_expires_at
            )
            self._schedule_transient_durable_handoff_expiry(
                delivery.delivery_id, handoff.expires_at
            )
        return replace(
            delivery,
            event=replace(delivery.event, payload=copy.deepcopy(handoff.payload)),
        )

    async def ack_durable_delivery(
        self, *, consumer_id: str, delivery_id: str, lease_token: str
    ) -> bool:
        """Acknowledge a claimed delivery owned by this dispatcher."""
        await self.initialize_durable_delivery()
        acknowledged = await self._durable_store.ack_delivery(
            agent_id=self._agent.did,
            consumer_id=consumer_id,
            delivery_id=delivery_id,
            lease_token=lease_token,
        )
        if acknowledged:
            self._discard_transient_durable_handoff(delivery_id)
        return acknowledged

    async def nack_durable_delivery(
        self,
        *,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        error: str,
        retry_delay: timedelta = timedelta(),
        terminal: bool = False,
    ) -> Optional[DurableDelivery]:
        """Release a claimed delivery for a bounded retry or terminal failure."""
        await self.initialize_durable_delivery()
        delivery = await self._durable_store.nack_delivery(
            agent_id=self._agent.did,
            consumer_id=consumer_id,
            delivery_id=delivery_id,
            lease_token=lease_token,
            error=error,
            retry_delay=retry_delay,
            terminal=terminal,
        )
        if delivery is not None and delivery.status == FAILED:
            self._discard_transient_durable_handoff(delivery_id)
        return delivery

    async def list_durable_deliveries(
        self,
        *,
        consumer_id: Optional[str] = None,
        statuses: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[DurableDelivery]:
        """Observe durable delivery state for this agent only."""
        await self.initialize_durable_delivery()
        return await self._durable_store.list_deliveries(
            agent_id=self._agent.did,
            consumer_id=consumer_id,
            statuses=statuses,
            limit=limit,
        )

    async def purge_expired_durable_deliveries(self) -> int:
        """Run the durable-ledger retention sweep (terminal history only)."""
        await self.initialize_durable_delivery()
        purged = await self._durable_store.purge_expired(agent_id=self._agent.did)
        self._discard_expired_transient_durable_handoffs()
        return purged

    def shutdown(self) -> None:
        """Discard non-durable live payload handoffs before agent teardown."""
        for timer in self._transient_durable_handoff_timers.values():
            timer.cancel()
        self._transient_durable_handoff_timers.clear()
        self._transient_durable_handoffs.clear()

    def _discard_expired_transient_durable_handoffs(self) -> None:
        """Drop raw payloads after their live lease or retention deadline."""
        now = datetime.now(timezone.utc)
        expired = [
            delivery_id
            for delivery_id, handoff in self._transient_durable_handoffs.items()
            if handoff.expires_at <= now
        ]
        for delivery_id in expired:
            self._discard_transient_durable_handoff(delivery_id)

    def _schedule_transient_durable_handoff_expiry(
        self, delivery_id: str, expires_at: datetime
    ) -> None:
        """Schedule removal without retaining the payload in a task closure."""
        previous = self._transient_durable_handoff_timers.pop(delivery_id, None)
        if previous is not None:
            previous.cancel()
        delay = max(0.0, (expires_at - datetime.now(timezone.utc)).total_seconds())
        self._transient_durable_handoff_timers[delivery_id] = (
            asyncio.get_running_loop().call_later(
                delay,
                self._expire_transient_durable_handoff,
                delivery_id,
                expires_at,
            )
        )

    def _expire_transient_durable_handoff(
        self, delivery_id: str, expires_at: datetime
    ) -> None:
        """Remove only the handoff whose currently scheduled deadline fired."""
        handoff = self._transient_durable_handoffs.get(delivery_id)
        if handoff is not None and handoff.expires_at == expires_at:
            self._discard_transient_durable_handoff(delivery_id)

    def _discard_transient_durable_handoff(self, delivery_id: str) -> None:
        self._transient_durable_handoffs.pop(delivery_id, None)
        timer = self._transient_durable_handoff_timers.pop(delivery_id, None)
        if timer is not None:
            timer.cancel()

    def _signal_for_durable_persistence(
        self, signal: Signal
    ) -> _DurableSignalProjection:
        """Return the privacy-safe event projection for the durable ledger.

        Signal handlers and cognition run against ``signal`` itself: source
        schemas have already normalized it, and changing it here would make a
        storage policy silently alter live delivery.  The ledger instead gets
        a copy with either an elided payload (EPHEMERAL, ISOLATED, and the
        not-yet-supported DEIDENTIFIED safe-harbor mode) or the same PII
        anonymization primitive used by channel storage (ANONYMOUS).

        A failure while reading or applying the privacy policy must never
        downgrade into a plaintext durable write, so this boundary fails
        closed by persisting only the marker.
        """
        config = resolve_agent_privacy_config(self._agent)
        if config is None:
            return _DurableSignalProjection(signal=signal)
        try:
            if any(
                getattr(config, name)()
                for name in (
                    "is_ephemeral",
                    "uses_temp_storage",
                    "requires_deidentification",
                )
            ):
                return _DurableSignalProjection(
                    signal=replace(
                        signal,
                        payload={
                            _DURABLE_PRIVACY_GATED_MARKER: getattr(
                                config, "storage", "restricted"
                            )
                        },
                    ),
                    payload_elided=True,
                )
            if config.requires_anonymization():
                from kestrel_sovereign.features.privacy.pii_detector import (
                    anonymize_text,
                )

                return _DurableSignalProjection(
                    signal=replace(
                        signal,
                        payload=self._anonymize_durable_value(
                            signal.payload, anonymize_text
                        ),
                    ),
                )
        except Exception as exc:  # Privacy persistence must fail closed.
            logger.warning(
                "Failed to project signal %s for durable privacy storage; "
                "persisting only a privacy marker: %s",
                signal.id,
                exc,
            )
            return _DurableSignalProjection(
                signal=replace(
                    signal,
                    payload={_DURABLE_PRIVACY_GATED_MARKER: "projection_error"},
                ),
                payload_elided=True,
            )
        return _DurableSignalProjection(signal=signal)

    @classmethod
    def _anonymize_durable_value(
        cls, value: Any, anonymize: Callable[[str], str]
    ) -> Any:
        """Apply the channel PII anonymizer to every persisted string value.

        The source payload is a nested JSON document.  Retaining its shape
        keeps durable correlation selectors replayable while applying the
        policy to content in metadata as well as the obvious ``content``
        field.  Overly deep input is elided rather than left unredacted.
        """
        return cls._anonymize_durable_value_at_depth(value, anonymize, depth=0)

    @classmethod
    def _anonymize_durable_value_at_depth(
        cls, value: Any, anonymize: Callable[[str], str], *, depth: int
    ) -> Any:
        if depth > 8:
            return _DURABLE_PRIVACY_GATED_MARKER
        if isinstance(value, dict):
            return {
                anonymize(str(key)): cls._anonymize_durable_value_at_depth(
                    item, anonymize, depth=depth + 1
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                cls._anonymize_durable_value_at_depth(
                    item, anonymize, depth=depth + 1
                )
                for item in value
            ]
        if isinstance(value, str):
            return anonymize(value)
        return value

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def _run(
        self,
        signal: Signal,
        start: float,
        *,
        source_event_id: Optional[str],
    ) -> SignalResult:
        # Step 1: validation
        registration = self._registry.get(signal.source)
        if registration is None:
            return self._fail(
                signal,
                start,
                Status.DROPPED_VALIDATION,
                error=f"Unknown source: '{signal.source}'",
            )

        if signal.mode not in registration.allowed_modes:
            return self._fail(
                signal,
                start,
                Status.DROPPED_VALIDATION,
                error=(
                    f"Mode {signal.mode.value} not in allowed modes for "
                    f"source '{signal.source}'"
                ),
                registration=registration,
            )

        # Trust is the MINIMUM of the registration ceiling and any per-signal
        # downgrade. A source registered TRUSTED still lets an individual signal
        # mark ITSELF untrusted (e.g. an inbound A2A wake whose sender signature
        # didn't verify, #1721) — but a caller can never RAISE trust above the
        # registration ceiling, so callers still can't lie upward. Concretely:
        # keep a self-downgraded UNTRUSTED signal, otherwise inherit the
        # registration's trust.
        if (
            registration.trust.value != "untrusted"
            and getattr(signal.origin_trust, "value", None) == "untrusted"
        ):
            pass  # honor the per-signal downgrade
        else:
            signal.origin_trust = registration.trust

        # UNTRUSTED → run sanitizer for non-ACTION modes
        if (
            signal.origin_trust.value == "untrusted"
            and signal.mode != SignalMode.ACTION
            and registration.sanitizer is not None
        ):
            try:
                signal.payload = registration.sanitizer(signal.payload)
            except Exception as e:
                return self._fail(
                    signal,
                    start,
                    Status.DROPPED_VALIDATION,
                    error=f"Sanitizer raised: {type(e).__name__}: {e}",
                    registration=registration,
                )

        # Schema validation runs AFTER sanitization so the schema validates
        # the canonical, scrubbed form that downstream handlers/templates
        # will actually see. The validated value replaces the payload —
        # schemas may normalize (defaults, type coercion) without surprise.
        try:
            signal.payload = registration.schema(signal.payload)
        except Exception as e:
            return self._fail(
                signal,
                start,
                Status.DROPPED_VALIDATION,
                error=f"Schema rejected payload: {type(e).__name__}: {e}",
                registration=registration,
            )

        # Step 2: append-and-cycle-check
        new_frame, cycle_reason = self._compute_frame_and_check_cycle(
            signal, registration
        )
        if cycle_reason is not None:
            return self._fail(
                signal, start, Status.DROPPED_CYCLE, error=cycle_reason,
                registration=registration,
            )
        signal.causation_chain.append(new_frame)

        # Durable delivery has a stricter boundary than signal_log: commit the
        # normalized/sanitized envelope and materialized consumer deliveries
        # before any handler, cognition turn, or external workflow executor can
        # observe it.  Thus a process loss after this point is replayable.
        try:
            await self.initialize_durable_delivery()
            # A transition and a durable write must share the same critical
            # section.  Otherwise a NORMAL projection can be computed, the
            # mode can change to EPHEMERAL while persistence is blocked, and
            # the stale plaintext projection can commit after the transition.
            # KestrelAgent provides a task-reentrant lock; lightweight
            # embeddings with no transition machinery intentionally run
            # unguarded through ``optional_transition_lock``.
            async with optional_transition_lock(
                _resolve_transition_lock(self._agent)
            ):
                durable_projection = self._signal_for_durable_persistence(signal)
                # Snapshot the normalized payload before the durable commit so
                # a deepcopy failure cannot leave a committed marker with no
                # corresponding live handoff.
                transient_payload = (
                    copy.deepcopy(signal.payload)
                    if durable_projection.payload_elided
                    else None
                )
                transient_selector_payload = (
                    {"transient_selector_payload": signal.payload}
                    if durable_projection.payload_elided
                    else {}
                )
                persisted = await self._durable_store.persist_signal(
                    durable_projection.signal,
                    agent_id=self._agent.did,
                    source_event_id=source_event_id,
                    retention_days=registration.retention_days,
                    # A volatile privacy projection must never make a
                    # pre-existing payload-correlated wait miss its normalized
                    # live signal.  DurableSignalStore consumes this only while
                    # materializing the initial deliveries; its event row and
                    # every later replay retain the projected payload above.
                    **transient_selector_payload,
                )
                if persisted.created and durable_projection.payload_elided:
                    # The delivery IDs come from the same transaction that
                    # matched selectors.  The pre-commit snapshot keeps later
                    # handler mutation from altering live worker input.
                    assert transient_payload is not None
                    retention_until = persisted.retention_until or datetime.now(
                        timezone.utc
                    )
                    for delivery_id in persisted.delivery_ids:
                        self._transient_durable_handoffs[delivery_id] = (
                            _TransientDurableHandoff(
                                payload=transient_payload,
                                retention_until=retention_until,
                                expires_at=retention_until,
                            )
                        )
                        self._schedule_transient_durable_handoff_expiry(
                            delivery_id,
                            self._transient_durable_handoffs[delivery_id].expires_at,
                        )
        except Exception as exc:
            logger.exception(
                "Failed to persist durable signal event %s (source=%s)",
                signal.id,
                signal.source,
            )
            return self._fail(
                signal,
                start,
                Status.FAILED,
                error=f"Durable signal persistence failed: {type(exc).__name__}: {exc}",
                registration=registration,
            )
        if not persisted.created:
            return self._fail(
                signal,
                start,
                Status.COALESCED,
                error=(
                    "Duplicate source event ID already accepted as "
                    f"{persisted.event_id}"
                ),
                registration=registration,
            )

        # Step 3: quiet-hours
        if self._in_quiet_hours(signal, registration.attention_policy):
            return self._fail(
                signal,
                start,
                Status.DROPPED_QUIET_HOURS,
                error="Source is in quiet hours and urgency is below override threshold",
                registration=registration,
            )

        # Step 4: coalescing
        if signal.dedupe_key is not None:
            window = registration.coalescing_window or self._default_window
            if self._coalescing.check_and_record(
                signal.source,
                signal.dedupe_key,
                window,
                now=self._clock(),
            ):
                return self._fail(
                    signal,
                    start,
                    Status.COALESCED,
                    error=None,
                    registration=registration,
                )

        # Step 5: rate limit
        if self._rate.check_and_record(
            signal.source, registration.rate_limit, now=time.monotonic()
        ):
            return self._fail(
                signal,
                start,
                Status.DROPPED_RATE_LIMIT,
                error="Per-source rate limit exceeded",
                registration=registration,
            )

        # Step 6 + 7: acquire locks and route
        return await self._route_under_locks(signal, registration, start)

    # ------------------------------------------------------------------
    # Cycle detection (precise rules — see SIGNAL_DISPATCHER.md §6)
    # ------------------------------------------------------------------

    def _compute_frame_and_check_cycle(
        self, signal: Signal, registration: SourceRegistration
    ) -> tuple[CausationFrame, Optional[str]]:
        existing_max_depth = max((f.depth for f in signal.causation_chain), default=0)
        new_depth = existing_max_depth + 1

        new_frame = CausationFrame(
            agent_id=signal.target_agent,
            source=signal.source,
            signal_id=signal.id,
            turn_id=None,  # set later if COGNITION succeeds
            depth=new_depth,
            emitted_at=self._clock(),
        )

        if new_depth > self._ttl:
            return new_frame, (
                f"Causation chain depth {new_depth} exceeds TTL {self._ttl}"
            )

        if not registration.allow_self_loops:
            for frame in signal.causation_chain:
                if (
                    frame.agent_id == signal.target_agent
                    and frame.source == signal.source
                ):
                    return new_frame, (
                        f"Cycle detected: agent={signal.target_agent} "
                        f"source={signal.source} appears earlier in chain "
                        f"at depth {frame.depth}"
                    )

        return new_frame, None

    # ------------------------------------------------------------------
    # Quiet hours
    # ------------------------------------------------------------------

    def _in_quiet_hours(
        self, signal: Signal, policy: AttentionPolicy
    ) -> bool:
        if policy.quiet_hours is None:
            return False
        if signal.mode not in policy.modes_governed:
            return False
        if signal.urgency.at_or_above(policy.urgency_override):
            return False

        try:
            tz = ZoneInfo(policy.tz)
        except Exception:
            logger.warning(
                "Invalid timezone '%s' on attention policy; treating as UTC",
                policy.tz,
            )
            tz = ZoneInfo("UTC")

        now_local = self._clock().astimezone(tz).time()
        start, end = policy.quiet_hours
        return _time_in_window(now_local, start, end)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def _route_under_locks(
        self,
        signal: Signal,
        registration: SourceRegistration,
        start: float,
    ) -> SignalResult:
        # CONVERSATION must never be in registration.resources — registry
        # validates this at registration time. The guard here is defense in
        # depth, in case someone constructs a SourceRegistration without
        # going through the registry.
        if ResourceLock.CONVERSATION in registration.resources:
            return self._fail(
                signal,
                start,
                Status.DROPPED_VALIDATION,
                error=(
                    "CONVERSATION resource lock found in registered resources. "
                    "It is owned solely by the turn lifecycle; reject."
                ),
                registration=registration,
            )

        async with self._locks.acquire(registration.resources):
            try:
                if signal.mode == SignalMode.ACTION:
                    assert registration.handler is not None
                    write_audit_callback = requested_handler_write_audit_callback()
                    if write_audit_callback is None:
                        action_result = await registration.handler(signal.payload)
                    else:
                        with capture_write_queries(write_audit_callback):
                            action_result = await registration.handler(signal.payload)
                    return self._success(
                        signal, start, registration, action_result=action_result
                    )

                if signal.mode == SignalMode.ARTIFACT:
                    assert registration.artifact_handler is not None
                    artifact = await registration.artifact_handler(signal)
                    return self._success(
                        signal, start, registration, artifact=artifact
                    )

                if signal.mode == SignalMode.COGNITION:
                    return await self._dispatch_cognition(
                        signal, registration, start
                    )

                # Unreachable — validation step rejected unknown modes.
                return self._fail(
                    signal,
                    start,
                    Status.FAILED,
                    error=f"Unhandled mode: {signal.mode}",
                    registration=registration,
                )

            except Exception as e:
                logger.exception(
                    "Handler raised for signal %s (source=%s, mode=%s)",
                    signal.id,
                    signal.source,
                    signal.mode.value,
                )
                return self._fail(
                    signal,
                    start,
                    Status.FAILED,
                    error=f"{type(e).__name__}: {e}",
                    registration=registration,
                )

    # ------------------------------------------------------------------
    # COGNITION dispatch — extracted so the constitutional-injection
    # audit (kestrel-sovereign#1137 chunk 1G) has a clear ownership
    # boundary and can pre-flight + post-flight without complicating
    # the ACTION/ARTIFACT branches.
    # ------------------------------------------------------------------

    async def _dispatch_cognition(
        self,
        signal: Signal,
        registration: SourceRegistration,
        start: float,
    ) -> SignalResult:
        assert registration.prompt_template is not None

        # Codex round-15 P2: clear the ContextVar at dispatch start
        # so an early-return process_input (safe mode, bootstrap, !
        # command) doesn't leak previous-turn injection tracking
        # into this dispatch's signal_log row.
        from kestrel_sovereign.agent.context_manager import (
            reset_injection_tracking,
        )

        reset_injection_tracking()

        # Step A: build the constitutional audit BEFORE the dispatch
        # runs. For sources with `constitution_injection="full"` this
        # resolves the operative constitution_hash, computes the live
        # doctrine_bundle_hash, and compares to the anchored value.
        # Drift → DROPPED_VALIDATION (the bundle is what would be
        # injected; refusing the dispatch is safer than dispatching
        # under tampered doctrine).
        audit = await self._build_constitution_audit(signal, registration)
        # OTel span (#1137 chunk 1H): emit a span for every COGNITION
        # dispatch with constitutional-injection attributes. No-op
        # when tracing is disabled. The span attributes are set as
        # the audit fills out so a partial dispatch (e.g. drift
        # refusal) still records what was resolved.
        span_attrs = {
            OI_SPAN_KIND: OI_SPAN_KIND_CHAIN,
            # The dispatcher owns one agent; stamp its name so this COGNITION
            # dispatch span groups under the right agent lane in Phoenix rather
            # than "(none)" (#2699). None (duck-typed/test agent) is dropped by
            # ``optional_span``, so this stays safe for minimal agents.
            KESTREL_AGENT_NAME: getattr(self._agent, "agent_name", None),
            "kestrel.signal.source": signal.source,
            "kestrel.signal.id": signal.id,
            "kestrel.constitution.injection": registration.constitution_injection,
            "kestrel.constitution.format": registration.prompt_template_format,
            "kestrel.constitution.echo_required": registration.require_constitution_echo,
        }
        if audit.constitution_hash:
            span_attrs["kestrel.constitution.hash"] = audit.constitution_hash[:16]
        if audit.doctrine_bundle_hash:
            span_attrs["kestrel.doctrine_bundle.hash"] = audit.doctrine_bundle_hash[:16]
        try:
            with optional_span(
                "signal.dispatch.cognition", span_attrs
            ) as span:
                result = await self._run_cognition_with_audit(
                    signal, registration, start, audit
                )
                if span is not None:
                    span.set_attribute(
                        "kestrel.constitution.echo_status",
                        audit.echo_canary_status.value,
                    )
                    span.set_attribute(
                        "kestrel.signal.status", result.status.value
                    )
                return result
        except Exception as e:
            # Codex round-3 P2: if process_input raises, the audit
            # would otherwise be lost when the outer try/except in
            # `_route_under_locks` calls `_fail` without it. Catch
            # here so the per-dispatch forensic trail (constitution
            # hash, bundle hash, MISSING canary stamp) lands in
            # signal_log even for LLM/API failure cases.
            logger.exception(
                "COGNITION dispatch raised for signal %s "
                "(source=%s) — preserving audit on _fail",
                signal.id,
                signal.source,
            )
            return self._fail(
                signal,
                start,
                Status.FAILED,
                error=f"{type(e).__name__}: {e}",
                registration=registration,
                audit=audit,
            )

    async def _run_cognition_with_audit(
        self,
        signal: Signal,
        registration: SourceRegistration,
        start: float,
        audit: "_ConstitutionAudit",
    ) -> SignalResult:
        if audit.drift_error is not None:
            record_doctrine_bundle_drift(signal.source)
            return self._fail(
                signal,
                start,
                Status.DROPPED_VALIDATION,
                error=audit.drift_error,
                registration=registration,
                audit=audit,
            )

        # Resolve the constitution body for full-injection sources.
        # Codex round-5 P1 fix: the dispatcher UNCONDITIONALLY
        # prepends a fenced constitution block to the rendered
        # prompt for `constitution_injection="full"`. A previous
        # round exposed the constitution via a `{constitution}`
        # template placeholder, but that made injection
        # template-author-dependent — a source whose template
        # forgot the placeholder would still pass canary verification
        # without the model ever seeing the constitution. Unconditional
        # prepend is auditable, deterministic, and impossible to
        # bypass via template error.
        constitution_text: Optional[str] = None
        if registration.constitution_injection == "full":
            getter = getattr(self._agent, "_get_governing_constitution", None)
            if callable(getter):
                try:
                    value = getter()
                    if asyncio.iscoroutine(value):
                        value = await value
                    if isinstance(value, str):
                        constitution_text = value
                except Exception:
                    logger.exception(
                        "_get_governing_constitution raised for signal %s; "
                        "the dispatch will refuse to proceed without the "
                        "constitution body for a full-injection source",
                        signal.id,
                    )
            # Codex round-7 P2: the existing
            # `ConstitutionMixin._get_governing_constitution` returns
            # error-sentinel strings like
            # "Error: Could not retrieve constitution..." on storage
            # failures. Those are non-empty strings but emphatically
            # NOT a constitution body. Treat them as missing — refuse
            # the dispatch rather than prepend the error message and
            # pretend it counts as a constitution receipt.
            if (
                isinstance(constitution_text, str)
                and constitution_text.lstrip().startswith("Error:")
            ):
                logger.warning(
                    "_get_governing_constitution returned an error "
                    "sentinel for signal %s; refusing dispatch "
                    "rather than injecting the error string as if it "
                    "were the constitution",
                    signal.id,
                )
                constitution_text = None
            if not constitution_text:
                # Refuse the dispatch — full injection without an
                # injectable constitution would log VERIFIED while
                # the model saw no constitution; that defeats the
                # entire point of `require_constitution_echo`.
                return self._fail(
                    signal,
                    start,
                    Status.DROPPED_VALIDATION,
                    error=(
                        "constitution_injection='full' requested but "
                        "agent could not produce a constitution body"
                    ),
                    registration=registration,
                    audit=audit,
                )

            # Codex round-6 P2 #1: `_get_governing_constitution()` may
            # have lazily anchored on first call (writing
            # constitution_hash to agent_node). The audit recorded
            # whatever was there at audit-build time; refresh it now
            # so the canary derivation below uses the post-anchor
            # value, not the pre-anchor None.
            get_const = getattr(self._agent, "get_constitution_hash", None)
            if callable(get_const) and not audit.constitution_hash:
                try:
                    refreshed = get_const()
                    if asyncio.iscoroutine(refreshed):
                        refreshed = await refreshed
                    if isinstance(refreshed, str) and refreshed:
                        audit.constitution_hash = refreshed
                except Exception:
                    logger.exception(
                        "Post-anchor constitution_hash refresh raised for "
                        "signal %s; canary derivation may use stale value",
                        signal.id,
                    )

        prompt = self._render_prompt(signal, registration)

        # Constitution delivery is format-conditional:
        #
        # - `claude_code` (in-agent): the agent's `build_system_prompt`
        #   already places the constitution in the system prompt every
        #   turn (cached, stable). The dispatcher does NOT duplicate
        #   it into the user prompt — that would pollute conversation
        #   history AND break the cache-stable prefix.
        #
        # - `codex` / `local` (external reviewer): the dispatch goes
        #   to a non-in-agent reviewer that does NOT have the agent's
        #   system-prompt assembly path. The "prompt" IS the entire
        #   message to that reviewer. The dispatcher MUST inline the
        #   constitution into the prompt, otherwise the canary will
        #   verify but the model never received the constitution body
        #   (codex round-11 P1 finding).
        #
        # - `bare`: caller-responsibility. Operator constructs whatever
        #   prompt shape they need; dispatcher provides the canary
        #   primitive but does not inject anything.
        if constitution_text is not None:
            audit.injected_clauses = ["KESTREL_CONSTITUTION"]
            fmt = registration.prompt_template_format
            if fmt in ("codex", "local"):
                # External reviewer paths: the entire prompt IS the
                # message. Inline the constitution AND any anchored
                # doctrine so the reviewer actually sees the doctrine
                # the canary will verify (codex round-23 P2 fix —
                # without this, drift-checking the bundle hash claims
                # coverage the reviewer never received).
                from kestrel_sovereign.agent.system_prompt_assembler import (
                    section_name_for_anchored_file,
                )

                doctrine_blocks: List[str] = []
                getter = getattr(
                    self._agent, "get_anchored_doctrine_files", None
                )
                if callable(getter):
                    try:
                        files = getter()
                        if asyncio.iscoroutine(files):
                            files = await files
                        if files:
                            for fname, body in files.items():
                                label = section_name_for_anchored_file(fname)
                                doctrine_blocks.append(
                                    f"--- {label} ---\n{body}\n--- END {label} ---"
                                )
                                if audit.injected_clauses is None:
                                    audit.injected_clauses = ["KESTREL_CONSTITUTION"]
                                if fname not in audit.injected_clauses:
                                    audit.injected_clauses.append(fname)
                    except Exception:
                        logger.exception(
                            "Inline-format doctrine resolution failed for "
                            "signal %s; reviewer will see only the "
                            "constitution body",
                            signal.id,
                        )

                doctrine_section = (
                    "\n\n".join(doctrine_blocks) + "\n\n"
                    if doctrine_blocks
                    else ""
                )
                prompt = (
                    "--- GOVERNING CONSTITUTION ---\n"
                    f"{constitution_text}\n"
                    "--- END CONSTITUTION ---\n\n"
                    f"{doctrine_section}"
                    f"{prompt}"
                )
            # For claude_code, no inline prepend — the constitution
            # and doctrine arrive via the agent's system-prompt path
            # (build_system_prompt_with_tracking handles
            # anchored_doctrine when budget is set). For bare,
            # caller-responsibility.

        # Derive canary + build the format-appropriate instruction
        # BEFORE the LLM call. Pass it as `system_prompt_addendum` to
        # process_input so it lands in the system prompt (where
        # operational directives belong) rather than the user prompt
        # (where they would persist into conversation history). Cache:
        # the addendum changes per dispatch (canary is fresh), so
        # echo-required dispatches don't benefit from prompt-prefix
        # cache hits — that's expected; verification is the priority.
        # Cache stability for the COMMON path (no echo) is unchanged
        # because no addendum is supplied.
        addendum: Optional[str] = None
        if registration.require_constitution_echo:
            # Codex round-21 P2: refuse pre-execution when echo is
            # required but no canary can be derived (no
            # constitution_hash). Running the turn anyway would
            # incur side effects only to fail
            # constitution_not_received afterward.
            if not audit.constitution_hash:
                return self._fail(
                    signal,
                    start,
                    Status.DROPPED_VALIDATION,
                    error=(
                        "require_constitution_echo=True but agent "
                        "could not provide a constitution_hash; canary "
                        "cannot be derived. Anchor the constitution "
                        "before enabling echo on this source."
                    ),
                    registration=registration,
                    audit=audit,
                )
            canary = derive_canary(
                signal_id=signal.id,
                constitution_hash=audit.constitution_hash,
                engine_nonce=secrets.token_hex(16),
            )
            audit.canary = canary
            instruction = build_canary_instruction(
                canary, registration.prompt_template_format
            )
            if instruction is not None:
                addendum = instruction

        # Codex round-9 P2 #3: don't catch TypeError around
        # process_input — that would also catch errors raised inside
        # the LLM/tool path and retry, duplicating side effects.
        # Inspect the signature once up-front so we know whether the
        # kwargs are accepted; if not, log and skip silently.
        accepts_addendum = _agent_accepts_kwarg(
            self._agent.process_input, "system_prompt_addendum"
        )
        accepts_budget = _agent_accepts_kwarg(
            self._agent.process_input, "system_prompt_budget_bytes"
        )
        # Codex round-19 P2 + round-20 P2: refuse pre-execution when
        # the canary directive can't reach the model. CRITICAL: this
        # check runs BEFORE `_set_current_chain` so failing here does
        # NOT leak the in-flight causation chain ContextVar (which
        # would corrupt subsequent task lineage).
        if addendum is not None and not accepts_addendum:
            return self._fail(
                signal,
                start,
                Status.DROPPED_VALIDATION,
                error=(
                    "require_constitution_echo=True but "
                    "agent.process_input does not accept "
                    "system_prompt_addendum kwarg; canary directive "
                    "cannot be delivered. Upgrade the agent to plumb "
                    "the kwarg through context_manager.build_context."
                ),
                registration=registration,
                audit=audit,
            )

        receipt_tool_registered = False
        if (
            registration.require_constitution_echo
            and registration.prompt_template_format == "claude_code"
            and audit.canary is not None
        ):
            register_receipt = getattr(
                self._agent, "register_constitution_receipt_tool", None
            )
            if not callable(register_receipt):
                return self._fail(
                    signal,
                    start,
                    Status.DROPPED_VALIDATION,
                    error=(
                        "require_constitution_echo=True for claude_code "
                        "but agent cannot register the ephemeral "
                        "_constitution_receipt tool"
                    ),
                    registration=registration,
                    audit=audit,
                )
            value = register_receipt(canary=audit.canary, signal_id=signal.id)
            if asyncio.iscoroutine(value):
                await value
            receipt_tool_registered = True

        # Set the in-flight turn's causation chain (already extended
        # with this hop's frame in step 2 of the pipeline) so outbound
        # A2A tasks created during the turn carry the lineage forward
        # (#905 review P1). The chain lives in a ContextVar so
        # concurrent COGNITION dispatches stay isolated — agent-level
        # mutable state would race here (#906 review P1: dispatcher
        # set/clear runs OUTSIDE the CONVERSATION lock that
        # process_input acquires inside its body).
        set_chain = getattr(self._agent, "_set_current_chain", None)
        clear_chain = getattr(self._agent, "_clear_current_chain", None)
        token = None
        if set_chain is not None:
            token = set_chain(signal.causation_chain)

        budget = registration.system_prompt_budget_bytes
        accepts_anchored = _agent_accepts_kwarg(
            self._agent.process_input, "anchored_doctrine"
        )
        # Resolve anchored doctrine for full-injection sources so the
        # agent's system-prompt assembler can deliver doctrine content
        # (TORTOISE_DOCTRINE.md, AGENTS.md). Codex round-16 P2 fix —
        # without this, the audit recorded the bundle hash but the
        # model never received the doctrine.
        anchored_doctrine = None
        if (
            registration.constitution_injection == "full"
            and accepts_anchored
        ):
            getter = getattr(self._agent, "get_anchored_doctrine_files", None)
            if callable(getter):
                try:
                    value = getter()
                    if asyncio.iscoroutine(value):
                        value = await value
                    if value:
                        anchored_doctrine = value
                except Exception:
                    logger.exception(
                        "agent.get_anchored_doctrine_files raised for "
                        "signal %s; proceeding without anchored doctrine",
                        signal.id,
                    )
        process_input_kwargs: dict[str, Any] = {}
        if addendum is not None and accepts_addendum:
            process_input_kwargs["system_prompt_addendum"] = addendum
        if budget is not None and accepts_budget:
            process_input_kwargs["system_prompt_budget_bytes"] = budget
        if anchored_doctrine is not None:
            process_input_kwargs["anchored_doctrine"] = anchored_doctrine
        # Route the cognition turn into the signal's originating session when one
        # is set (e.g. the restart.completed wake carries the session the
        # restart was requested from, #1809), so the turn lands in that chat
        # window instead of a fresh implicit session. Guarded by signature
        # inspection like the other optional kwargs.
        if signal.session_id and _agent_accepts_kwarg(
            self._agent.process_input, "session_id"
        ):
            process_input_kwargs["session_id"] = signal.session_id

        # Tag the persisted wake turn so the transcript renderer collapses this
        # internal COGNITION prompt to an "Autonomous wake" chip on reload
        # instead of surfacing the raw instruction template as a user message.
        # Guarded by signature inspection like the other optional kwargs.
        if _agent_accepts_kwarg(self._agent.process_input, "signal_wake"):
            process_input_kwargs["signal_wake"] = {
                "source": signal.source,
                "mode": signal.mode.value,
            }

        try:
            if process_input_kwargs:
                result = await self._agent.process_input(
                    prompt, **process_input_kwargs
                )
            else:
                result = await self._agent.process_input(prompt)
        except Exception:
            if receipt_tool_registered:
                clear_receipt = getattr(
                    self._agent, "clear_constitution_receipt_tool", None
                )
                if callable(clear_receipt):
                    clear_receipt()
            raise
        finally:
            if clear_chain is not None:
                clear_chain(token)

        # Codex round-13/14 P2: surface the agent's actual
        # injected/dropped clause tracking from the budget-aware
        # assembler into the signal_log audit. The tracking is
        # published per-async-task via a ContextVar in
        # `kestrel_sovereign.agent.context_manager`, so concurrent
        # COGNITION dispatches don't race on a shared attribute.
        from kestrel_sovereign.agent.context_manager import (
            get_current_injection_tracking,
        )

        injection_tracking = get_current_injection_tracking()
        if injection_tracking is not None:
            tracked_injected, tracked_dropped = injection_tracking
            if tracked_injected is not None:
                # Merge: dispatcher's KESTREL_CONSTITUTION marker
                # plus the assembler's own clause list. Use a
                # de-duplicated, order-preserving merge.
                merged = list(audit.injected_clauses or [])
                for clause in tracked_injected:
                    if clause not in merged:
                        merged.append(clause)
                audit.injected_clauses = merged
            if tracked_dropped:
                audit.dropped_clauses = list(tracked_dropped)

        # Step B: post-dispatch echo verification. If the source
        # opted in to `require_constitution_echo=True` the dispatcher
        # asks the agent for the format-specific receipt — checking
        # against the SAME canary that was injected pre-dispatch
        # (audit.canary, derived in Step A.5). Anything other than
        # VERIFIED flips the dispatch to FAILED with
        # `error="constitution_not_received"` per design §3 — codex
        # round-3 P2 fix: NOT_REQUIRED returned by a verifier when
        # echo IS required is a contract violation, not a pass.
        if registration.require_constitution_echo:
            await self._verify_canary_post_dispatch(
                signal, registration, audit, response=result
            )
            if receipt_tool_registered:
                clear_receipt = getattr(
                    self._agent, "clear_constitution_receipt_tool", None
                )
                if callable(clear_receipt):
                    clear_receipt()
                receipt_tool_registered = False
            if audit.echo_canary_status is CanaryStatus.VERIFIED:
                record_echo_verified(signal.source)
            else:
                record_echo_missing(signal.source)
                return self._fail(
                    signal,
                    start,
                    Status.FAILED,
                    error="constitution_not_received",
                    registration=registration,
                    audit=audit,
                )

        if receipt_tool_registered:
            clear_receipt = getattr(
                self._agent, "clear_constitution_receipt_tool", None
            )
            if callable(clear_receipt):
                clear_receipt()

        return self._success(
            signal,
            start,
            registration,
            audit=audit,
            cognition_result=result,
        )

    async def _ensure_doctrine_bundle_anchored(self) -> None:
        """Codex round-18 P1: ensure the doctrine bundle is anchored
        on the agent_node BEFORE the first drift check. Without this,
        agents upgraded to Phase 1 would have no anchored hash and
        every dispatch would skip drift detection, accepting any
        edits to AGENTS.md / TORTOISE_DOCTRINE.md silently.

        Calls the optional `agent.ensure_doctrine_bundle_anchored()`
        hook (idempotent — returns the existing anchor if set,
        otherwise writes a new one). Failures are logged but
        non-fatal; the dispatch proceeds with whatever state the
        agent ended up in."""
        ensure = getattr(self._agent, "ensure_doctrine_bundle_anchored", None)
        if not callable(ensure):
            return
        try:
            value = ensure()
            if asyncio.iscoroutine(value):
                await value
        except Exception:
            logger.exception(
                "agent.ensure_doctrine_bundle_anchored raised; "
                "drift detection may be skipped for this dispatch"
            )

    async def _build_constitution_audit(
        self, signal: Signal, registration: SourceRegistration
    ) -> _ConstitutionAudit:
        """Resolve the per-dispatch audit before the LLM call.

        For sources with `constitution_injection="none"` this returns
        a default audit (all None, status=NOT_REQUIRED) — the legacy
        path. For `"full"` it consults optional agent hooks
        (`get_constitution_hash`, `get_anchored_doctrine_bundle_hash`,
        `compute_live_doctrine_bundle_hash`) via `getattr` so the
        dispatcher remains usable with minimal/test agents that
        don't provide them; the audit fields are populated where
        possible and drift is detected only when both anchored and
        live hashes are available."""
        audit = _ConstitutionAudit()

        if registration.constitution_injection != "full":
            return audit

        # Codex round-18 P1: ensure the bundle is anchored before
        # we read anchored vs live hashes — first-time dispatch on
        # a fresh agent establishes the anchor; subsequent dispatches
        # detect drift normally.
        await self._ensure_doctrine_bundle_anchored()

        get_const = getattr(self._agent, "get_constitution_hash", None)
        if callable(get_const):
            try:
                value = get_const()
                if asyncio.iscoroutine(value):
                    value = await value
                audit.constitution_hash = value
            except Exception:
                logger.exception(
                    "agent.get_constitution_hash raised for signal %s; "
                    "leaving constitution_hash NULL",
                    signal.id,
                )

        anchored_hash: Optional[str] = None
        live_hash: Optional[str] = None

        get_anchored = getattr(
            self._agent, "get_anchored_doctrine_bundle_hash", None
        )
        if callable(get_anchored):
            try:
                value = get_anchored()
                if asyncio.iscoroutine(value):
                    value = await value
                anchored_hash = value
            except Exception:
                logger.exception(
                    "agent.get_anchored_doctrine_bundle_hash raised for "
                    "signal %s",
                    signal.id,
                )

        get_live = getattr(self._agent, "compute_live_doctrine_bundle_hash", None)
        if callable(get_live):
            try:
                value = get_live()
                if asyncio.iscoroutine(value):
                    value = await value
                live_hash = value
            except Exception:
                logger.exception(
                    "agent.compute_live_doctrine_bundle_hash raised for "
                    "signal %s",
                    signal.id,
                )

        # The audit records the LIVE hash (what would actually be
        # injected); auditors comparing across dispatches see what
        # ran. The anchored hash is in agent_node.properties already.
        audit.doctrine_bundle_hash = live_hash

        if (
            anchored_hash is not None
            and live_hash is not None
            and anchored_hash != live_hash
        ):
            audit.drift_error = (
                f"doctrine_bundle_drift: anchored={anchored_hash[:16]}... "
                f"live={live_hash[:16]}..."
            )
            # Pin the anchored hash on the audit so the signal_log row
            # reflects what was expected, not just what the bundle was
            # at fail time.
            audit.doctrine_bundle_hash = anchored_hash

        # Echo defaults to NOT_REQUIRED unless the registration opts
        # in. The post-dispatch verifier flips it to VERIFIED or
        # MISSING based on the agent's receipt channel.
        if registration.require_constitution_echo:
            # Tentatively MISSING until the post-dispatch verifier
            # confirms; this way a dispatch that crashes mid-flight
            # is recorded as MISSING (operator sees it; safer default).
            audit.echo_canary_status = CanaryStatus.MISSING

        return audit

    async def _verify_canary_post_dispatch(
        self,
        signal: Signal,
        registration: SourceRegistration,
        audit: _ConstitutionAudit,
        *,
        response: Any,
    ) -> None:
        """Post-dispatch hook to verify the format-specific receipt.

        The canary was derived BEFORE `process_input` and stored on
        `audit.canary` so the model actually saw the directive in its
        prompt. Here we ask `agent.verify_constitution_echo(canary,
        format, signal_id)` whether the model honored it via the
        format-specific channel.

        Outcomes:
        - `audit.canary is None` (couldn't derive: no constitution
          hash) → MISSING.
        - No verifier hook → MISSING (Phase 1 contract; sources
          shipping echo=True before agents implement the hook fail
          loudly so the gap is visible, not silent).
        - Verifier raises → MISSING (safe degradation).
        - Verifier returns non-CanaryStatus / non-string → MISSING.
        - Otherwise → whatever the verifier reports."""
        if not registration.require_constitution_echo:
            audit.echo_canary_status = CanaryStatus.NOT_REQUIRED
            return

        if audit.canary is None:
            # Pre-dispatch derivation skipped (no constitution hash);
            # the model never saw a canary so verification is moot.
            audit.echo_canary_status = CanaryStatus.MISSING
            return

        verifier = getattr(self._agent, "verify_constitution_echo", None)
        if not callable(verifier):
            audit.echo_canary_status = CanaryStatus.MISSING
            return

        try:
            verifier_kwargs = {
                "canary": audit.canary,
                "prompt_template_format": registration.prompt_template_format,
                "signal_id": signal.id,
            }
            # Pass the response only if the verifier accepts it. The
            # default ConstitutionMixin verifier needs it to scan
            # codex/local format reviewer responses for the canary.
            if _agent_accepts_kwarg(verifier, "response"):
                verifier_kwargs["response"] = response
            value = verifier(**verifier_kwargs)
            if asyncio.iscoroutine(value):
                value = await value
        except Exception:
            logger.exception(
                "agent.verify_constitution_echo raised for signal %s",
                signal.id,
            )
            audit.echo_canary_status = CanaryStatus.MISSING
            return

        if isinstance(value, CanaryStatus):
            audit.echo_canary_status = value
        elif isinstance(value, str):
            try:
                audit.echo_canary_status = CanaryStatus(value)
            except ValueError:
                audit.echo_canary_status = CanaryStatus.MISSING
        else:
            audit.echo_canary_status = CanaryStatus.MISSING

    def _render_prompt(
        self,
        signal: Signal,
        registration: SourceRegistration,
    ) -> str:
        # Minimal template render for v1: read the file and substitute
        # known placeholders. Fenced UNTRUSTED payload is the source's
        # responsibility via the prompt template content (the design says
        # "templates live under prompts/signals/" with explicit fences).
        # A richer template engine (jinja, etc.) is a follow-up.
        #
        # `{constitution}` is accepted as a placeholder for backward
        # compatibility with templates from an earlier chunk-1G round
        # but is ALWAYS substituted with an empty string. Constitution
        # injection happens via the dispatcher's unconditional prepend
        # for `constitution_injection="full"` sources (see
        # `_run_cognition_with_audit`). This avoids the
        # template-forgot-the-placeholder bypass codex round-5 caught.
        template_path = self._select_prompt_template(signal, registration)
        assert template_path is not None
        template = template_path.read_text(encoding="utf-8")
        setattr(
            signal,
            _PROMPT_TEMPLATE_HASH_ATTR,
            hashlib.sha256(template.encode("utf-8")).hexdigest(),
        )
        return template.format(
            source=signal.source,
            kind=signal.kind,
            target_agent=signal.target_agent,
            payload=signal.payload,
            urgency=signal.urgency.value,
            arrived_at=signal.arrived_at.isoformat(),
            constitution="",
        )

    @staticmethod
    def _select_prompt_template(
        signal: Signal,
        registration: SourceRegistration,
    ) -> Optional[Path]:
        template_path = registration.prompt_template
        override = getattr(signal, "prompt_template_override", None)
        if override is None:
            return template_path
        if not getattr(registration, "allow_prompt_override", False):
            return template_path
        if isinstance(override, Path):
            return override
        if isinstance(override, os.PathLike):
            return Path(override)
        raise TypeError(
            "Signal.prompt_template_override must be pathlib.Path or os.PathLike"
        )

    # ------------------------------------------------------------------
    # Result construction + logging
    # ------------------------------------------------------------------

    def _success(
        self,
        signal: Signal,
        start: float,
        registration: SourceRegistration,
        *,
        artifact: Any = None,
        action_result: Any = None,
        cognition_result: Any = None,
        audit: Optional[_ConstitutionAudit] = None,
    ) -> SignalResult:
        # SignalResult has separate fields for side-effect action results and
        # generated artifacts. A cognition dispatch produces user-visible turn
        # output, so we return it through artifact: callers already consume that
        # field as the generic signal output payload for non-ACTION modes.
        result = SignalResult(
            signal_id=signal.id,
            status=Status.OK,
            mode=signal.mode,
            duration_ms=int((time.monotonic() - start) * 1000),
            artifact=artifact if artifact is not None else cognition_result,
            action_result=action_result,
        )
        # Fire-and-forget log write; if logging fails it's logged but does
        # not fail the dispatch.
        self._agent._track_background_task(
            self._log_safe(signal, registration, result, audit=audit),
            name=f"signal_log:{signal.source}:{signal.id}",
        )
        return result

    def _fail(
        self,
        signal: Signal,
        start: float,
        status: Status,
        *,
        error: Optional[str],
        registration: Optional[SourceRegistration] = None,
        audit: Optional[_ConstitutionAudit] = None,
    ) -> SignalResult:
        result = SignalResult(
            signal_id=signal.id,
            status=status,
            mode=signal.mode,
            duration_ms=int((time.monotonic() - start) * 1000),
            error=error,
        )
        if registration is not None:
            self._agent._track_background_task(
            self._log_safe(signal, registration, result, audit=audit),
            name=f"signal_log:{signal.source}:{signal.id}",
        )
        return result

    async def _log_safe(
        self,
        signal: Signal,
        registration: SourceRegistration,
        result: SignalResult,
        *,
        audit: Optional[_ConstitutionAudit] = None,
    ) -> None:
        try:
            with suppress_write_audit():
                result_summary = await self._store.append(
                    signal,
                    registration,
                    result,
                    prompt_template_hash=getattr(
                        signal, _PROMPT_TEMPLATE_HASH_ATTR, None
                    ),
                    **_audit_to_log_kwargs(audit),
                )
        except Exception:
            logger.exception(
                "Failed to write signal_log entry for %s", signal.id
            )
            # #907 review P2: contract says SSE event fires AFTER the
            # log write. If the write failed, the audit trail is
            # broken — emitting a signal_completed event whose
            # signal_id can't be looked up in signal_log would mislead
            # consumers that try to correlate. Better to be silent and
            # surface the failure via the logger.exception above.
            return

        # Phase 7 of #889: emit a UI-side-channel SSE event for non-
        # INTERNAL signals. Three rendering tiers in the design:
        #   - INTERNAL              → log only, no UI emit
        #   - USER_VISIBLE          → side channel OR inline (per session_id)
        #   - ADMIN_VISIBLE         → side channel for admin tools
        # Existing sources (heartbeat, cron, a2a.task_complete, stripe)
        # default to INTERNAL — none of them surprise-emit to the UI.
        # Sources opt in by constructing signals with an explicit
        # visibility argument.
        if signal.visibility == Visibility.INTERNAL:
            return
        emit = getattr(self._agent, "emit_event", None)
        if emit is None:
            return
        payload = _build_ui_event_payload(signal, result, result_summary)
        try:
            await emit("signal_completed", payload)
        except Exception:
            logger.exception(
                "Failed to emit signal_completed UI event for %s", signal.id
            )


def _build_ui_event_payload(
    signal: Signal,
    result: SignalResult,
    result_summary: Optional[str],
) -> dict:
    """Construct the JSON payload shape for the UI side-channel event.

    Routing fields (`session_id`, `visibility`, `target_agent`,
    `source`, `caller`) tell the consumer where to render the turn.
    Result fields (`status`, `duration_ms`, `error`) summarize the
    outcome.

    `result_summary` (#907 review P1 fix) is the bounded text body
    produced by the source's `result_summary` callback, capped in the
    store at MAX_RESULT_SUMMARY_BYTES. Sources that don't set the
    callback get None here — consumers see metadata only and must
    fetch the body from a source-specific surface (chat history,
    task_execution_log, etc.). The same value lands in
    signal_log.result_summary; the UI can scroll back via the log if
    it needs the historical text after the SSE event.

    The full raw artifact / action_result is NEVER included — the
    per-source `result_summary` callback controls what becomes
    user-visible, with a hard byte cap as defense in depth.
    """
    return {
        "signal_id": signal.id,
        "source": signal.source,
        "kind": signal.kind,
        "mode": signal.mode.value,
        "target_agent": signal.target_agent,
        "session_id": signal.session_id,
        "caller": signal.caller,
        "visibility": signal.visibility.value,
        "urgency": signal.urgency.value,
        "status": result.status.value,
        "duration_ms": result.duration_ms,
        "error": result.error,
        "result_summary": result_summary,
        "arrived_at": signal.arrived_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_in_window(now: dtime, start: dtime, end: dtime) -> bool:
    """True if `now` is inside the [start, end) window. Wraps midnight if
    end < start (e.g. 22:00 → 06:00)."""
    if start == end:
        # Degenerate; treat as "never quiet"
        return False
    if start < end:
        return start <= now < end
    # Wrap-around window
    return now >= start or now < end
