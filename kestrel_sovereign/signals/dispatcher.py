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
import contextvars
import copy
import hashlib
import hmac
import inspect
import json
import logging
import os
import re
import secrets
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional, Protocol
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
    Trust,
    Urgency,
    Visibility,
)

from kestrel_sovereign.features.storage_access import resolve_agent_privacy_config
from kestrel_sovereign.security.encryption import (
    DecryptionError,
    MasterKeyNotConfiguredError,
    get_agent_fernet,
    get_agent_key,
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
from kestrel_sovereign.signals.durable import (
    ACKNOWLEDGED,
    FAILED,
    PENDING,
    RETRY,
    TERMINAL_ACKABLE,
    DurableConsumerRegistration,
    DurableDelivery,
    DurableSourceBoundary,
    DurableSignalStore,
)
from kestrel_sovereign.signals.lock_manager import OrderedLockManager
from kestrel_sovereign.signals.registry import SourceRegistry
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_CONSUMER_ID,
    DURABLE_COGNITION_MARKER,
    DURABLE_COGNITION_MARKER_VALUE,
    DURABLE_TERMINAL_CONSUMER_ID,
)
from kestrel_sovereign.signals.store import SignalLogStore
from kestrel_sovereign.storage.db.write_audit import (
    capture_write_queries,
    requested_handler_write_audit_callback,
    suppress_write_audit,
)
from kestrel_sovereign.storage.privacy_wrapper import (
    _resolve_transition_lock,
    optional_transition_lock,
)
from kestrel_sovereign.telemetry import (
    KESTREL_AGENT_NAME,
    OI_SPAN_KIND,
    OI_SPAN_KIND_CHAIN,
    optional_span,
    session_span_attributes,
)

logger = logging.getLogger(__name__)


# Durable event payloads are intentionally distinct from the runtime signal
# envelope.  This marker preserves an observable event/consumer handoff while
# proving that a volatile privacy mode did not retain user-authored content.
_DURABLE_PRIVACY_GATED_MARKER = "_privacy_gated"
_DURABLE_CALLER_IDENTITY_NONE = "v1:none"
_DURABLE_CALLER_IDENTITY_PREFIX = "v1:"
_DURABLE_CALLER_IDENTITY_AAD_PREFIX = b"kestrel:durable-signal-caller:v1:"
# Keyless installations retain NORMAL durable payloads in plaintext by the
# documented storage policy.  A caller identity still must not add a second,
# raw identifier column merely because no at-rest key is configured.  This
# process-generated, non-secret opaque label is stable for the retained event
# (including provider redelivery) without encoding or exposing the caller.
_DURABLE_CALLER_IDENTITY_KEYLESS_PREFIX = "v2:opaque:"
_DURABLE_CALLER_IDENTITY_KEYLESS_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_DURABLE_INTEGRITY_CONTEXT = b"kestrel:durable-signal-integrity:v1"
# A cancellation-resistant cognition turn must never pin the ingress dispatch
# forever.  The surviving task remains dispatcher-owned after this bounded
# join, and its exact delivery is released only from its done callback.
_DURABLE_COGNITION_CANCELLATION_GRACE = 0.25
# A recovered cursor-owned delivery must remain retryable without a busy loop.
# The timer is only a wake-up for the persisted ``next_attempt_at`` value; it
# is not a second queue and process loss is recovered by the next owner scan.
_DURABLE_COGNITION_RETRY_DELAY = timedelta(seconds=1)


_PROMPT_TEMPLATE_HASH_ATTR = "_kestrel_prompt_template_hash"


class _DurableDeliveryShuttingDownError(RuntimeError):
    """Raised when a durable operation arrives after teardown begins.

    This is deliberately a distinct failure from a database error.  Callers
    can safely retry against a replacement dispatcher, while a dispatch maps
    it to its normal ``SignalResult`` failure contract.
    """


class DurableAdmissionDisposition(str, Enum):
    """The durable-ledger disposition an ingress producer may checkpoint."""

    COMMITTED = "committed"
    DUPLICATE = "duplicate"
    TERMINAL = "terminal"
    NOT_ADMITTED = "not_admitted"


@dataclass(frozen=True)
class DurableAdmissionResult:
    """An explicit durable-admission result for one enqueued signal."""

    disposition: DurableAdmissionDisposition
    signal_id: str

    @property
    def acknowledged(self) -> bool:
        """Whether an external source may advance its stable cursor."""

        return self.disposition in {
            DurableAdmissionDisposition.COMMITTED,
            DurableAdmissionDisposition.DUPLICATE,
            DurableAdmissionDisposition.TERMINAL,
        }


@dataclass
class SignalDispatchHandle(SignalHandle):
    """SDK terminal handle plus the earlier durable-admission receipt.

    ``SignalHandle.wait()`` remains the released SDK 0.35.1 terminal dispatch
    contract.  ACK-bearing Core ingress uses this explicit receipt instead of
    mistaking background-task creation for a durable commit.
    """

    durable_admission: asyncio.Future[DurableAdmissionResult]

    async def wait_for_durable_admission(self) -> DurableAdmissionResult:
        return await asyncio.shield(self.durable_admission)


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
class SignalLogWriteFailure:
    """One dropped ``signal_log`` audit row (#2660).

    Public because the health surface reports it: an audit row that failed to
    persist is a loss, and a loss nobody can observe is operationally the same
    as no loss.  Carries enough to act on — which signal, what failed, when —
    without retaining the payload the row would have redacted anyway.
    """

    signal_id: str
    error: str
    failed_at: datetime


# ---------------------------------------------------------------------------
# UI surface accounting (#2922)
# ---------------------------------------------------------------------------
#
# Persisting a turn and SURFACING it are different facts, and conflating them
# is what made #2877 take months to diagnose: the wait reconciler read a
# dispatch-level ``ok`` as "the user saw the wake" while the observer's chat
# stayed blank. The dispatcher is the only component that watches the
# ``signal_completed`` emit, so it records what it observed, per signal, and
# the reconciler reads that instead of inferring visibility from the dispatch
# status alone.
#
# The CEILING of what these verdicts can claim is deliberate. ``QUEUED`` means
# at least one live listener took the event — for ``/notifications/sse`` that
# is admission to a server-side ``asyncio.Queue``. The browser can still drop
# it: ``chat.js`` discards a wake whose ``session_id`` is not the open pane's
# conversation, and it requires a non-empty ``result_summary``. No server-side
# state can prove a render, so none of these values says "surfaced" or "seen".
# Anything the dispatcher could not observe is ``UNKNOWN``, never an optimistic
# guess.

# INTERNAL signal: no UI emit is attempted, by design.
SURFACE_NOT_APPLICABLE = "not_applicable"
# The agent exposes no ``emit_event`` at all (minimal/legacy stand-ins).
SURFACE_NO_EMITTER = "no_emitter"
# The emit call itself raised.
SURFACE_EMIT_FAILED = "emit_failed"
# No listener was connected; the event was buffered for replay on reconnect.
SURFACE_BUFFERED = "buffered"
# Every connected listener raised — the event reached no consumer.
SURFACE_REJECTED = "rejected"
# At least one listener accepted the event into its server-side queue. The
# strongest verdict available, and still NOT proof that anything rendered.
SURFACE_QUEUED = "queued"
# ``emit_event`` returned no receipt (a stand-in predating #2922) — the
# outcome is genuinely unobservable rather than assumed good.
SURFACE_UNKNOWN = "unknown"

# Verdicts that mean the event definitively reached no live consumer.
SURFACE_UNSURFACED_STATES = frozenset(
    {SURFACE_NO_EMITTER, SURFACE_EMIT_FAILED, SURFACE_BUFFERED, SURFACE_REJECTED}
)


@dataclass(frozen=True)
class SignalSurfaceRecord:
    """What the dispatcher observed of one signal's UI side-channel emit.

    Public because the wait reconciler consumes it to distinguish "persisted"
    from "reached a live consumer" in its delivery ledger (#2922).
    """

    signal_id: str
    status: str
    listeners: int = 0
    accepted: int = 0
    rejected: int = 0

    @property
    def reached_a_consumer(self) -> bool:
        """True only for :data:`SURFACE_QUEUED` — server-side acceptance.

        Never interpret this as "the user saw it"; see the module notes above.
        """
        return self.status == SURFACE_QUEUED

    @property
    def definitely_unsurfaced(self) -> bool:
        """True when the emit demonstrably reached no live consumer."""
        return self.status in SURFACE_UNSURFACED_STATES


def _receipt_count(receipt: Any, field: str) -> int:
    """Read one integer counter off an ``emit_event`` receipt, or 0."""
    try:
        return int(getattr(receipt, field, 0) or 0)
    except (TypeError, ValueError):
        return 0


def classify_event_receipt(receipt: Any) -> str:
    """Map an ``emit_event`` return value onto a ``SURFACE_*`` verdict.

    Structural rather than ``isinstance``-based: the agent is a protocol here,
    and a host/feature stand-in may supply its own compatible receipt. A return
    value that does not carry the receipt's counters is ``SURFACE_UNKNOWN`` —
    a caller that cannot report delivery must not be credited with it.
    """
    if receipt is None:
        return SURFACE_UNKNOWN
    buffered = getattr(receipt, "buffered", None)
    accepted = getattr(receipt, "accepted", None)
    if buffered is None or accepted is None:
        return SURFACE_UNKNOWN
    if buffered:
        return SURFACE_BUFFERED
    try:
        accepted_count = int(accepted)
    except (TypeError, ValueError):
        return SURFACE_UNKNOWN
    return SURFACE_QUEUED if accepted_count > 0 else SURFACE_REJECTED


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
    consumer_id: str
    created_at: datetime
    retention_until: datetime
    expires_at: datetime
    # Kept as the historical field name for in-process test/embedding
    # compatibility. It is a reservation capability until post-commit
    # activation, never a pre-commit lease token.
    initial_lease_token: Optional[str] = None


@dataclass(frozen=True)
class _DeferredOutcomeLog:
    """One route-level outcome held until its durable lease is finalized."""

    signal: Signal
    registration: SourceRegistration
    result: SignalResult
    audit: Optional[_ConstitutionAudit]


@dataclass
class _DurableAdmissionReservation:
    """One synchronous claim on the dispatcher lifecycle gate.

    Releasing a reservation must never await: cancellation is delivered only
    at await boundaries, so a synchronous, idempotent release cannot strand
    the active-admission count after an operation's ``finally`` begins.
    """

    released: bool = False


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

    # Cap on retained per-signal UI-emit verdicts (#2922). The only consumer
    # reads a record on the reconciler tick after the one that enqueued it, so
    # a small window is sufficient; bounding it keeps a long-lived host from
    # accumulating one entry per signal forever.
    _MAX_SURFACE_RECORDS = 512

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
        runtime_owner_stale_after: timedelta = timedelta(minutes=2),
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._agent = agent
        self._registry = registry
        self._locks = lock_manager
        self._store = store
        # Dropped-audit-row accounting (#2660). See ``_record_log_write_failure``
        # for why this is in-memory rather than persisted.
        self._log_write_failures = 0
        self._last_log_write_failure: Optional[SignalLogWriteFailure] = None
        # Keep outcome/audit persistence and pending-delivery persistence
        # distinct.  Existing embeddings/tests construct only SignalLogStore;
        # deriving the durable store from its backend preserves that seam while
        # retaining one database transaction domain for each agent.
        self._durable_store = durable_store or DurableSignalStore(store.backend)
        self._durable_initialized = False
        # ``initialize_durable_delivery`` registers this owner before its
        # startup recovery runs.  Keep that fact separate from full
        # initialization: recovery itself can fail, and boot rollback must
        # still mark the partly-registered owner stopped before storage closes.
        self._durable_runtime_owner_registered = False
        # A cancelled ``register_runtime_owner`` await is ambiguous on real
        # async database drivers: its commit can finish on the driver worker
        # immediately before cancellation is delivered to this task.  Record
        # that the registration *may* have committed before awaiting it, so
        # teardown always issues the owner-scoped release for that generation.
        # A release against a row that never committed is an intentional
        # no-op; leaving a committed row live is not.
        self._durable_runtime_owner_registration_started = False
        self._durable_init_lock = asyncio.Lock()
        # Durable shutdown is a state transition, not a best-effort flag.
        # Every public durable operation (and every dispatch) takes an
        # admission before it can touch the ledger.  Shutdown closes admission
        # under the same lock, waits for all already-admitted operations, then
        # releases the runtime owner and permits storage to close.  A task-local
        # depth makes the guard re-entrant: a dispatch owns one admission while
        # its internal ``initialize_durable_delivery`` call shares it rather
        # than deadlocking or reopening the gate after shutdown has begun.
        self._durable_lifecycle_lock = asyncio.Lock()
        self._durable_admissions_drained = asyncio.Event()
        self._durable_admissions_drained.set()
        self._durable_active_admissions = 0
        # ContextVars are copied into child tasks.  Keep the actual outer
        # admission owners separately so shutdown can distinguish a stale
        # copied context from one whose parent dispatch is still live.
        self._durable_admission_owners: set[asyncio.Task] = set()
        self._durable_admission: contextvars.ContextVar[
            Optional[tuple[asyncio.Task, int]]
        ] = contextvars.ContextVar(
            f"durable_delivery_admission:{id(self)}", default=None
        )
        # Outcome logging is accepted as part of the dispatch that formed the
        # result.  These tasks deliberately do not live in the agent's general
        # background-task set: that set is cancelled before durable teardown,
        # while a log write sharing this backend must be drained (or its
        # cancelled-before-start reservation reconciled) before storage closes.
        self._outcome_log_tasks: set[asyncio.Task] = set()
        self._outcome_log_tasks_by_signal: dict[str, set[asyncio.Task]] = {}
        # Cursor-owned cognition must not write a route-level outcome before
        # its exact durable ACK/NACK decision.  The task-local collector lets
        # the normal routing code continue constructing its audited result
        # while its sole log entry is committed only at that outer boundary.
        self._deferred_outcome_logs: contextvars.ContextVar[
            Optional[list[_DeferredOutcomeLog]]
        ] = contextvars.ContextVar(
            f"deferred_durable_outcome_logs:{id(self)}", default=None
        )
        # A durable retry is the same persisted delivery, not a competing
        # source event. The child route task inherits this narrow context flag
        # so legacy test/subclass seams retain their three-argument signature.
        self._durable_retry_skips_coalescing: contextvars.ContextVar[bool] = (
            contextvars.ContextVar(
                f"durable_retry_skips_coalescing:{id(self)}", default=False
            )
        )
        # Cancellation-resistant cognition turns are intentionally retained
        # rather than abandoned.  Both sets are bounded by the one in-flight
        # cursor delivery per dispatcher/consumer and expose exact ownership
        # for shutdown and test/health inspection.
        self._retained_durable_cognition_tasks: set[asyncio.Task[SignalResult]] = set()
        self._retained_durable_cognition_cleanup_tasks: set[asyncio.Task[None]] = set()
        # Channel ingress ACKs after its selected delivery is durable, not
        # after cognition completes.  These owners make that promise useful
        # after the provider has advanced its cursor: every restarted
        # dispatcher scans and drains the persisted consumer itself.
        self._durable_cognition_drainers: dict[str, asyncio.Task[None]] = {}
        self._durable_cognition_drain_timers: dict[str, asyncio.TimerHandle] = {}
        self._started_durable_cognition_consumers: set[str] = set()
        # A live provider callback owns the first exact claim for its own
        # event. The recovery drainer may execute only rows that survived that
        # admission attempt; otherwise it could steal a just-persisted PENDING
        # delivery before the callback obtains the receipt it needs to ACK.
        self._live_durable_cognition_event_ids: set[str] = set()
        # A process may be asked to unload while a cognition turn keeps
        # suppressing cancellation.  Ordinary shutdown stays bounded, but its
        # managed runtime owner must remain live until that exact turn settles;
        # otherwise a peer can requeue the same cursor delivery while side
        # effects from the original task are still running.
        self._durable_shutdown_owner_fenced = False
        self._fenced_durable_shutdown_completion: Optional[asyncio.Task[None]] = None
        self._runtime_owner_fence_lock = asyncio.Lock()
        # What the UI side-channel emit actually did, per signal (#2922).
        # Deliberately in-memory and bounded: it is diagnostic provenance for
        # the wait reconciler's next harvest, not a second audit trail, and an
        # entry that did not survive a restart must read as UNKNOWN rather
        # than resurrect as an optimistic verdict. ``dispatch_signal`` drains
        # its own outcome writers before returning, so the record for a signal
        # is always present by the time its handle resolves.
        self._surface_records: "OrderedDict[str, SignalSurfaceRecord]" = OrderedDict()
        # This task owns teardown, rather than whichever public caller first
        # requested it.  Callers await it through ``shield`` so cancellation
        # of an agent's bounded shutdown wrapper cannot abandon a committed
        # reservation or poison a later retry.
        self._durable_shutdown_completion: Optional[asyncio.Task[None]] = None
        if runtime_owner_stale_after.total_seconds() <= 0:
            raise ValueError("runtime_owner_stale_after must be positive")
        self._runtime_owner_stale_after = runtime_owner_stale_after
        self._runtime_owner_heartbeat_timer: Optional[asyncio.TimerHandle] = None
        self._runtime_owner_heartbeat_task: Optional[asyncio.Task] = None
        # A transient ledger outage must not silently orphan this runtime's
        # ownership heartbeat.  Backoff is capped so repeated errors retain a
        # bounded future retry rather than either hot-looping or giving up.
        self._runtime_owner_heartbeat_failures = 0
        self._durable_shutdown = False
        # This is intentionally dispatcher-local.  It is not a second durable
        # queue and must disappear on shutdown/restart; ``delivery_id`` is
        # globally unique and every dispatcher owns exactly one agent scope.
        self._transient_durable_handoffs: dict[str, _TransientDurableHandoff] = {}
        self._transient_durable_handoff_timers: dict[str, asyncio.TimerHandle] = {}
        # Post-commit reservation repair must outlive the agent-wide
        # best-effort background-task sweep.  A repair created immediately
        # before shutdown may not get a first event-loop turn before that
        # sweep cancels ordinary work; cancelling it would strand a committed
        # ``LEASED`` or ``INITIAL_RESERVED`` row.  The dispatcher owns these
        # critical tasks and drains them before it releases its runtime owner
        # or lets storage close.
        self._post_commit_reservation_repairs: set[asyncio.Task] = set()
        # The durable lease transfer is atomic, but its raw sidecar is shared
        # by local claimants. A payload-elided persistence holds this lock
        # from pre-commit sidecar installation through commit; a local initial
        # claim then takes the same lock before it can transfer the lease. A
        # PostgreSQL claimant can otherwise query on a separate transaction
        # between those points, not see the uncommitted row, and mistakenly
        # discard the valid raw sidecar.
        self._transient_durable_initial_claim_lock = asyncio.Lock()
        # This opaque runtime generation is persisted as the initial
        # reservation owner. Its matching capability remains in the live
        # sidecar, so another dispatcher sharing this database cannot consume
        # an initial payload-elided delivery before this instance activates and
        # hands it to its worker. A separate runtime heartbeat makes startup
        # recovery distinguish a crashed owner from a concurrent live one.
        self._durable_delivery_owner = f"dispatcher:{secrets.token_urlsafe(24)}"
        self._ttl = ttl
        self._default_window = coalescing_window_default
        self._clock = clock
        self._coalescing = _CoalescingState()
        self._rate = _RateLimitState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _admit_durable_operation(self):
        """Admit one public durable operation until its work is reconciled.

        Checking a shutdown flag at method entry is insufficient: an
        initializer can be awaiting owner registration, or a dispatch can be
        between initialization and its event transaction when shutdown starts.
        This gate linearizes admission with shutdown.  Once closing is marked,
        no new operation can reach durable storage; work admitted before that
        point remains valid and shutdown waits for it to finish.
        """
        current_task = asyncio.current_task()
        if current_task is None:  # pragma: no cover - async APIs run in Tasks
            raise RuntimeError("Durable signal delivery requires an asyncio task")
        admission = self._durable_admission.get()
        if admission is not None and admission[0] is current_task:
            token = self._durable_admission.set((current_task, admission[1] + 1))
            try:
                yield
            finally:
                self._durable_admission.reset(token)
            return

        await self._durable_lifecycle_lock.acquire()
        try:
            if self._durable_shutdown:
                raise _DurableDeliveryShuttingDownError(
                    "Durable signal delivery is shutting down"
                )
            reservation = self._reserve_durable_admission()
            self._durable_admission_owners.add(current_task)
        finally:
            # ``Lock.release`` is synchronous.  Keeping this out of an
            # ``async with`` avoids another cancellation boundary between
            # deciding admission and publishing its reservation.
            self._durable_lifecycle_lock.release()

        token = self._durable_admission.set((current_task, 1))
        try:
            yield
        finally:
            self._durable_admission.reset(token)
            # This finalizer is intentionally await-free.  A cancellation may
            # arrive at every other exit boundary, but cannot interrupt the
            # exactly-once counter release and leave shutdown waiting forever.
            self._durable_admission_owners.discard(current_task)
            self._release_durable_admission(reservation)

    def _reserve_durable_admission(self) -> _DurableAdmissionReservation:
        """Synchronously reserve one unit of dispatcher/storage lifetime.

        The caller has already linearized admission with lifecycle shutdown.
        This is also used while a dispatch is still admitted to transfer that
        lifetime to an outcome-log task before the dispatch can return.
        """
        self._durable_active_admissions += 1
        self._durable_admissions_drained.clear()
        return _DurableAdmissionReservation()

    def _release_durable_admission(
        self, reservation: _DurableAdmissionReservation
    ) -> None:
        """Release one lifetime reservation without an await boundary."""
        if reservation.released:
            return
        reservation.released = True
        if self._durable_active_admissions <= 0:
            raise RuntimeError("Durable admission accounting underflow")
        self._durable_active_admissions -= 1
        if self._durable_active_admissions == 0:
            self._durable_admissions_drained.set()

    def has_live_durable_admission_in_current_context(self) -> bool:
        """Whether this task inherited a still-live durable admission.

        ``asyncio.create_task`` copies ContextVars.  A shutdown child created
        by a signal handler therefore sees its parent's admission tuple even
        though the child is not the owner.  That context is unsafe exactly
        while the parent remains an active admission: waiting for admission
        drain would then wait on the parent that is awaiting shutdown.  Once
        the parent releases the admission, its copied tuple is stale and a
        deferred child may safely perform teardown.
        """
        admission = self._durable_admission.get()
        return (
            admission is not None
            and admission[0] in self._durable_admission_owners
        )

    def _durable_shutdown_signal_result(
        self, signal: Signal, start: float
    ) -> SignalResult:
        """Return the dispatch contract's safe failure after closure starts."""
        return SignalResult(
            signal_id=signal.id,
            status=Status.FAILED,
            mode=signal.mode,
            duration_ms=int((time.monotonic() - start) * 1000),
            error="Durable signal delivery is shutting down",
        )

    def notify_resume(self, gap_seconds: float) -> None:
        """Re-anchor throttling state after a host suspend/resume (#1545).

        Coalescing keys off wall-clock and rate-limiting off ``monotonic``,
        so across a suspend they disagree: coalescing windows expire (a
        repeat signal fires again) while the monotonic rate-limit window is
        frozen (quotas look saturated though the hour really elapsed).
        Clearing both restores a consistent, correct post-sleep baseline.
        Any volatile durable-delivery sidecars are also reconciled against
        their wall-clock lease deadlines because their ``call_later`` timers
        use the suspended monotonic clock.

        Invoked directly by the ``ResumeMonitor`` callback before it emits a
        ``system.resumed`` audit signal for each detected suspend.
        """
        self._rate.reset()
        self._coalescing.reset()
        # ``call_later`` runs against a monotonic clock, which pauses during
        # system suspend. Reconcile raw sidecars against UTC immediately, then
        # rebuild every live timer from its wall-clock deadline.
        self._discard_expired_transient_durable_handoffs()
        for delivery_id, handoff in list(self._transient_durable_handoffs.items()):
            self._schedule_transient_durable_handoff_expiry(
                delivery_id, handoff.expires_at
            )
        logger.info(
            "Dispatcher throttling windows re-anchored after ~%.0fs host suspend",
            gap_seconds,
        )

    async def dispatch_signal(
        self,
        signal: Signal,
        *,
        source_event_id: Optional[str] = None,
        _durable_admission: asyncio.Future[DurableAdmissionResult] | None = None,
        _durable_delivery_consumer_id: Optional[str] = None,
        _durable_terminal_consumer_id: Optional[str] = None,
    ) -> SignalResult:
        """Awaits the full lifecycle. Used by callers that need the result
        (scheduler, heartbeat). Always returns a `SignalResult` — failures
        are encoded as `Status.FAILED` with `error` set, never raised."""
        start = time.monotonic()
        # A durable cognition route defers *its own* outcome until the exact
        # delivery ACK/NACK boundary. ContextVars are copied into awaited and
        # background child dispatches, so inheriting that mutable list would
        # make a nested signal append to its parent's audit and lose its own
        # final outcome. Every public dispatch begins with a private collector;
        # a durable route installs a fresh one for its one final outcome below.
        deferred_token = self._deferred_outcome_logs.set(None)
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
            async with self._admit_durable_operation():
                result = await self._run(
                    signal,
                    start,
                    source_event_id=(
                        source_event_id
                        if source_event_id is not None
                        else getattr(signal, "source_event_id", None)
                    ),
                    durable_admission=_durable_admission,
                    durable_delivery_consumer_id=_durable_delivery_consumer_id,
                    durable_terminal_consumer_id=_durable_terminal_consumer_id,
                )
                if (
                    _durable_admission is not None
                    and not _durable_admission.done()
                ):
                    # A validation/cycle refusal is a proven terminal no-op for
                    # this exact signal.  Every other unresolved outcome,
                    # notably persistence failure, leaves the source cursor
                    # unchanged.
                    disposition = (
                        DurableAdmissionDisposition.TERMINAL
                        if result.status
                        in {Status.DROPPED_VALIDATION, Status.DROPPED_CYCLE}
                        else DurableAdmissionDisposition.NOT_ADMITTED
                    )
                    _durable_admission.set_result(
                        DurableAdmissionResult(disposition, signal.id)
                    )
                # The public dispatch contract returns only after its own
                # outcome entry has reached a terminal write result.  The task
                # remains separately admitted so a caller cancellation in
                # this await cannot create a storage-close gap.
                await self._drain_outcome_logs_for_signal(signal.id)
                return result
        except _DurableDeliveryShuttingDownError:
            return self._durable_shutdown_signal_result(signal, start)
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
            if _durable_admission is not None and not _durable_admission.done():
                # Cancellation and unexpected pipeline failure are not a
                # durable receipt.  Resolve the waiter so its source can retry
                # rather than waiting forever on an abandoned task.
                _durable_admission.set_result(
                    DurableAdmissionResult(
                        DurableAdmissionDisposition.NOT_ADMITTED,
                        signal.id,
                    )
                )
            reset_current_signal(ctx_token)
            self._deferred_outcome_logs.reset(deferred_token)

    async def enqueue_signal(
        self, signal: Signal, *, source_event_id: Optional[str] = None
    ) -> SignalDispatchHandle:
        """Returns immediately with a tracked handle. The dispatch runs as
        an agent-owned background task; exceptions are logged not swallowed,
        and the task is cancellable via the agent's shutdown path."""
        durable_admission: asyncio.Future[DurableAdmissionResult] = (
            asyncio.get_running_loop().create_future()
        )
        coro = self.dispatch_signal(
            signal,
            source_event_id=source_event_id,
            _durable_admission=durable_admission,
        )
        task = self._agent._track_background_task(
            coro, name=f"signal_dispatch:{signal.source}:{signal.id}"
        )
        return SignalDispatchHandle(
            signal_id=signal.id,
            task=task,
            durable_admission=durable_admission,
        )

    async def enqueue_durable_cognition(
        self,
        signal: Signal,
        *,
        source_event_id: Optional[str],
        consumer_id: str,
    ) -> SignalDispatchHandle:
        """Enqueue cursor-owning cognition behind one durable delivery lease.

        Its admission receipt resolves once the event and its selected durable
        delivery have committed.  Cognition runs in the dispatcher-owned
        durable executor afterwards; an external cursor must never wait for a
        full LLM turn merely to learn that its work survived process loss.
        """
        durable_admission: asyncio.Future[DurableAdmissionResult] = (
            asyncio.get_running_loop().create_future()
        )
        coro = self.dispatch_signal(
            signal,
            source_event_id=source_event_id,
            _durable_admission=durable_admission,
            _durable_delivery_consumer_id=consumer_id,
        )
        self._live_durable_cognition_event_ids.add(signal.id)
        task = self._agent._track_background_task(
            coro, name=f"durable_cognition:{consumer_id}:{signal.id}"
        )
        def release_live_admission(_task: asyncio.Task[SignalResult]) -> None:
            self._live_durable_cognition_event_ids.discard(signal.id)
            # A NACK can have committed just before the task becomes terminal.
            # Re-scan only after releasing this local first-claim fence.
            if consumer_id in self._started_durable_cognition_consumers:
                self._schedule_durable_cognition_drain(consumer_id, delay=0.0)

        task.add_done_callback(release_live_admission)
        return SignalDispatchHandle(
            signal_id=signal.id,
            task=task,
            durable_admission=durable_admission,
        )

    async def enqueue_durable_terminal(
        self,
        signal: Signal,
        *,
        source_event_id: Optional[str],
        consumer_id: str = DURABLE_TERMINAL_CONSUMER_ID,
    ) -> SignalDispatchHandle:
        """Persist a selected terminal ingress disposition without cognition."""

        durable_admission: asyncio.Future[DurableAdmissionResult] = (
            asyncio.get_running_loop().create_future()
        )
        coro = self.dispatch_signal(
            signal,
            source_event_id=source_event_id,
            _durable_admission=durable_admission,
            _durable_terminal_consumer_id=consumer_id,
        )
        task = self._agent._track_background_task(
            coro, name=f"durable_terminal:{consumer_id}:{signal.id}"
        )
        return SignalDispatchHandle(
            signal_id=signal.id,
            task=task,
            durable_admission=durable_admission,
        )

    async def start_durable_cognition_consumer(self, consumer_id: str) -> None:
        """Start the durable owner for a cursor-owning cognition consumer.

        This is intentionally explicit rather than making every durable
        subscription executable: a consumer still needs its registered source
        contract.  Calling it at feature boot is idempotent and immediately
        scans persisted retry work left behind by a dead process.
        """

        if not isinstance(consumer_id, str) or not consumer_id:
            raise ValueError("consumer_id must be a non-empty string")
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            self._started_durable_cognition_consumers.add(consumer_id)
            self._start_durable_cognition_drain(consumer_id)

    def _start_durable_cognition_drain(self, consumer_id: str) -> None:
        if self._durable_shutdown:
            return
        existing = self._durable_cognition_drainers.get(consumer_id)
        if existing is not None and not existing.done():
            return
        timer = self._durable_cognition_drain_timers.pop(consumer_id, None)
        if timer is not None:
            timer.cancel()
        task = asyncio.create_task(
            self._drain_durable_cognition_consumer(consumer_id),
            name=f"durable_cognition_drain:{consumer_id}:{self._agent.did}",
        )
        self._durable_cognition_drainers[consumer_id] = task

        def complete(completed: asyncio.Task[None]) -> None:
            if self._durable_cognition_drainers.get(consumer_id) is completed:
                self._durable_cognition_drainers.pop(consumer_id, None)
            if completed.cancelled():
                return
            exc = completed.exception()
            if exc is not None:
                logger.exception(
                    "Durable cognition drainer stopped unexpectedly: consumer=%s",
                    consumer_id,
                    exc_info=exc,
                )
                self._schedule_durable_cognition_drain(consumer_id, delay=1.0)

        task.add_done_callback(complete)

    async def _drain_durable_cognition_consumer(self, consumer_id: str) -> None:
        """Claim and route all currently-due work for one durable consumer."""

        while not self._durable_shutdown:
            candidates = await self.list_durable_deliveries(
                consumer_id=consumer_id,
                statuses=[PENDING, RETRY],
                limit=100,
            )
            delivery = None
            blocked_by_live_admission = False
            for candidate in candidates:
                if candidate.event_id in self._live_durable_cognition_event_ids:
                    blocked_by_live_admission = True
                    continue
                delivery = await self.claim_durable_delivery_for_event(
                    consumer_id=consumer_id,
                    event_id=candidate.event_id,
                    executor_id=self._durable_delivery_owner,
                )
                if delivery is not None:
                    break
            if delivery is None:
                if blocked_by_live_admission:
                    # The live callback will either ACK, NACK (which schedules
                    # us from the persisted retry deadline), or die; a later
                    # runtime owner restart then discovers its row.
                    return
                await self._schedule_next_durable_cognition_drain(consumer_id)
                return
            registration = self._registry.get(delivery.event.source)
            if registration is None:
                # A feature can only drain a source it currently owns.  Keep
                # the ledger row retryable until boot has registered the
                # source again rather than converting it into an ACKable loss.
                await self.nack_durable_delivery(
                    consumer_id=consumer_id,
                    delivery_id=delivery.delivery_id,
                    lease_token=delivery.lease_token or "",
                    error="Durable cognition source registration is unavailable",
                    retry_delay=_DURABLE_COGNITION_RETRY_DELAY,
                )
                await self._schedule_next_durable_cognition_drain(consumer_id)
                return
            seed = Signal(
                source=delivery.event.source,
                kind=delivery.event.kind,
                mode=SignalMode(delivery.event.mode),
                payload={},
                target_agent=delivery.event.target_agent,
            )
            await self._route_durable_cognition_delivery(
                seed,
                registration,
                time.monotonic(),
                persisted_event_id=delivery.event_id,
                consumer_id=consumer_id,
                durable_admission=None,
                durable_created=False,
                use_live_signal=False,
                claimed_delivery=delivery,
                retry_delay=_DURABLE_COGNITION_RETRY_DELAY,
            )

    async def _schedule_next_durable_cognition_drain(self, consumer_id: str) -> None:
        pending = await self.list_durable_deliveries(
            consumer_id=consumer_id,
            statuses=[PENDING, RETRY],
            limit=1,
        )
        if not pending:
            return
        next_attempt = pending[0].next_attempt_at
        delay = 0.0
        if next_attempt is not None:
            delay = max(0.0, (next_attempt - datetime.now(timezone.utc)).total_seconds())
        self._schedule_durable_cognition_drain(consumer_id, delay=delay)

    def _schedule_durable_cognition_drain(self, consumer_id: str, *, delay: float) -> None:
        if self._durable_shutdown:
            return
        old_timer = self._durable_cognition_drain_timers.pop(consumer_id, None)
        if old_timer is not None:
            old_timer.cancel()

        def wake() -> None:
            self._durable_cognition_drain_timers.pop(consumer_id, None)
            self._start_durable_cognition_drain(consumer_id)

        self._durable_cognition_drain_timers[consumer_id] = (
            asyncio.get_running_loop().call_later(delay, wake)
        )

    async def initialize_durable_delivery(self) -> None:
        """Initialize the durable consumer ledger exactly once.

        Agent boot calls this eagerly, while the guard in ``_run`` preserves
        compatibility for embedding/test dispatchers that predate the ledger.
        Initialization is complete before the first event is persisted.
        """
        async with self._admit_durable_operation():
            if self._durable_initialized:
                return
            async with self._durable_init_lock:
                if not self._durable_initialized:
                    await self._durable_store.initialize()
                    self._durable_runtime_owner_registration_started = True
                    await self._durable_store.register_runtime_owner(
                        agent_id=self._agent.did,
                        owner_id=self._durable_delivery_owner,
                    )
                    self._durable_runtime_owner_registered = True
                    # Registration can wait behind another database writer.  Do
                    # not carry a pre-wait timestamp into recovery: that could
                    # immediately classify an otherwise live owner as stale.
                    await self._recover_abandoned_initial_reservations()
                    await self._recover_abandoned_leases()
                    self._durable_initialized = True
                    self._schedule_runtime_owner_heartbeat()

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
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            await self._durable_store.register_consumer(registration)

    async def capture_durable_source_boundary(
        self, *, source: str
    ) -> DurableSourceBoundary:
        """Capture this agent's current durable commit boundary for ``source``.

        Call immediately before dispatching an external effect whose later
        workflow wake must exclude previously committed history. The workflow
        must durably commit ``boundary.to_dict()`` before external dispatch and
        rehydrate it after restart; an in-memory cursor is not crash-safe. The
        API has no ``agent_id`` argument by design: dispatcher ownership is the
        tenant authority, and A2A identity is a separate boundary.
        """

        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            return await self._durable_store.capture_source_boundary(
                agent_id=self._agent.did,
                source=source,
            )

    async def deactivate_durable_consumer(self, *, consumer_id: str) -> bool:
        """Deactivate one of this agent's durable consumers.

        This is the public durable-consumer lifecycle boundary.  It returns
        ``False`` only when this dispatcher has no registration with
        ``consumer_id``; an already-inactive registration returns ``True``.
        The durable store atomically preserves its historical rows while
        terminalizing every live delivery, so no post-deactivation claim or
        stale executor follow-up can recreate work.
        """
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            deactivated = await self._durable_store.deactivate_consumer(
                agent_id=self._agent.did,
                consumer_id=consumer_id,
            )
            if not deactivated:
                return False
            # Payload-eliding privacy modes retain a process-local sidecar
            # until a delivery reaches a terminal state.  The store has just
            # made this consumer's nonterminal rows terminal, so discard that
            # no-longer-authorized live payload state too.
            for delivery_id, handoff in list(self._transient_durable_handoffs.items()):
                if handoff.consumer_id == consumer_id:
                    self._discard_transient_durable_handoff(delivery_id)
            self._started_durable_cognition_consumers.discard(consumer_id)
            timer = self._durable_cognition_drain_timers.pop(consumer_id, None)
            if timer is not None:
                timer.cancel()
            return True

    async def claim_durable_delivery(
        self, *, consumer_id: str, executor_id: str
    ) -> Optional[DurableDelivery]:
        """Atomically claim one delivery for this agent-scoped consumer."""
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            self._discard_expired_transient_durable_handoffs()
            delivery = await self._durable_store.claim_delivery(
                agent_id=self._agent.did,
                consumer_id=consumer_id,
                executor_id=executor_id,
                runtime_owner_stale_before=(
                    datetime.now(timezone.utc) - self._runtime_owner_stale_after
                ),
            )
            if delivery is not None:
                return self._delivery_with_transient_handoff(delivery)

            # A payload-elided event is initially an unclaimable reservation in
            # the transaction that writes its durable privacy marker. It becomes a
            # real owner lease only after that transaction commits. Ordinary claims
            # above therefore cannot race ahead of the emitting dispatcher while
            # its process-local sidecar is installed. Only this dispatcher, which
            # holds the reservation capability outside the durable payload, may
            # transfer the activated lease to the requested worker.
            async with self._transient_durable_initial_claim_lock:
                # A prior local claimant may have transferred the reservation while
                # this claimant waited. Re-read the sidecar state inside the lock;
                # a cleared token means there is no initial reservation left here.
                self._discard_expired_transient_durable_handoffs()
                reservations = sorted(
                    (
                        (delivery_id, handoff)
                        for delivery_id, handoff in self._transient_durable_handoffs.items()
                        if (
                            handoff.consumer_id == consumer_id
                            and handoff.initial_lease_token is not None
                        )
                    ),
                    key=lambda item: (item[1].created_at, item[0]),
                )
                for delivery_id, handoff in reservations:
                    assert handoff.initial_lease_token is not None
                    delivery = await self._durable_store.claim_initial_delivery(
                        agent_id=self._agent.did,
                        consumer_id=consumer_id,
                        delivery_id=delivery_id,
                        initial_lease_owner=self._durable_delivery_owner,
                        initial_lease_token=handoff.initial_lease_token,
                        executor_id=executor_id,
                    )
                    if delivery is None:
                        # A reservation can fail only after it was released,
                        # expired, or otherwise became terminal. Retaining raw
                        # data in that case would violate its lease-bound lifetime.
                        self._discard_transient_durable_handoff(delivery_id)
                        continue
                    handoff.initial_lease_token = None
                    return self._delivery_with_transient_handoff(delivery)
            return None

    async def claim_durable_delivery_for_event(
        self, *, consumer_id: str, event_id: str, executor_id: str
    ) -> Optional[DurableDelivery]:
        """Claim this consumer's delivery for exactly one persisted event.

        Payload-eliding privacy modes initially reserve their delivery to this
        dispatcher, rather than publishing a claimable marker row. The exact
        event path used by cursor-owning channel ingress must perform that same
        reservation transfer as the general polling path; otherwise a first
        EPHEMERAL/ISOLATED/DEIDENTIFIED delivery is durable but unclaimable.
        """
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            self._discard_expired_transient_durable_handoffs()
            delivery = await self._durable_store.claim_delivery_for_event(
                agent_id=self._agent.did,
                consumer_id=consumer_id,
                event_id=event_id,
                executor_id=executor_id,
                runtime_owner_stale_before=(
                    datetime.now(timezone.utc) - self._runtime_owner_stale_after
                ),
            )
            if delivery is not None:
                return self._delivery_with_transient_handoff(delivery)

            async with self._transient_durable_initial_claim_lock:
                reserved = await self._durable_store.get_delivery_for_event(
                    agent_id=self._agent.did,
                    consumer_id=consumer_id,
                    event_id=event_id,
                )
                if reserved is None:
                    return None
                handoff = self._transient_durable_handoffs.get(reserved.delivery_id)
                if handoff is None or handoff.initial_lease_token is None:
                    return None
                delivery = await self._durable_store.claim_initial_delivery(
                    agent_id=self._agent.did,
                    consumer_id=consumer_id,
                    delivery_id=reserved.delivery_id,
                    initial_lease_owner=self._durable_delivery_owner,
                    initial_lease_token=handoff.initial_lease_token,
                    executor_id=executor_id,
                )
                if delivery is None:
                    # If the transfer can no longer happen, raw data must not
                    # outlive the reservation capability that protected it.
                    self._discard_transient_durable_handoff(reserved.delivery_id)
                    return None
                handoff.initial_lease_token = None
                return self._delivery_with_transient_handoff(delivery)

    async def get_durable_delivery_for_event(
        self, *, consumer_id: str, event_id: str
    ) -> Optional[DurableDelivery]:
        """Read one consumer delivery without changing its ownership."""
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            return await self._durable_store.get_delivery_for_event(
                agent_id=self._agent.did,
                consumer_id=consumer_id,
                event_id=event_id,
            )

    def _delivery_with_transient_handoff(
        self, delivery: DurableDelivery
    ) -> DurableDelivery:
        """Attach a still-live payload sidecar to this dispatcher's claim."""
        handoff = self._transient_durable_handoffs.get(delivery.delivery_id)
        if handoff is None:
            return delivery
        if handoff.expires_at <= datetime.now(timezone.utc):
            self._discard_transient_durable_handoff(delivery.delivery_id)
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
        async with self._admit_durable_operation():
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
        terminal_ackable: bool = False,
    ) -> Optional[DurableDelivery]:
        """Release a claimed delivery for a bounded retry or terminal failure."""
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            delivery = await self._durable_store.nack_delivery(
                agent_id=self._agent.did,
                consumer_id=consumer_id,
                delivery_id=delivery_id,
                lease_token=lease_token,
                error=error,
                retry_delay=retry_delay,
                terminal=terminal,
                terminal_ackable=terminal_ackable,
            )
            if delivery is not None and delivery.status in {FAILED, TERMINAL_ACKABLE}:
                self._discard_transient_durable_handoff(delivery_id)
            if (
                delivery is not None
                and delivery.status == RETRY
                and consumer_id == DURABLE_COGNITION_CONSUMER_ID
                and consumer_id in self._started_durable_cognition_consumers
            ):
                # The provider may already have advanced after the initial
                # durable admission. Wake the dispatcher-owned scanner from
                # the row's persisted deadline so a later cognition NACK does
                # not depend on an imaginary provider redelivery.
                next_attempt = delivery.next_attempt_at or datetime.now(timezone.utc)
                self._schedule_durable_cognition_drain(
                    consumer_id,
                    delay=max(
                        0.0,
                        (next_attempt - datetime.now(timezone.utc)).total_seconds(),
                    ),
                )
            return delivery

    async def release_durable_delivery_after_task(
        self,
        *,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        error: str,
        terminal: bool = False,
        terminal_ackable: bool = False,
    ) -> Optional[DurableDelivery]:
        """Release this dispatcher's exact lease after retained work settles."""

        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            delivery = await self._durable_store.release_managed_delivery_after_task(
                agent_id=self._agent.did,
                consumer_id=consumer_id,
                delivery_id=delivery_id,
                lease_token=lease_token,
                owner_id=self._durable_delivery_owner,
                error=error,
                terminal=terminal,
                terminal_ackable=terminal_ackable,
            )
            if delivery is not None and delivery.status in {FAILED, TERMINAL_ACKABLE}:
                self._discard_transient_durable_handoff(delivery_id)
            return delivery

    async def renew_durable_delivery_lease(
        self, *, consumer_id: str, delivery_id: str, lease_token: str
    ) -> Optional[DurableDelivery]:
        """Renew a still-owned cursor delivery while its cognition turn runs."""

        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            return await self._durable_store.renew_delivery_lease(
                agent_id=self._agent.did,
                consumer_id=consumer_id,
                delivery_id=delivery_id,
                lease_token=lease_token,
            )

    async def list_durable_deliveries(
        self,
        *,
        consumer_id: Optional[str] = None,
        statuses: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[DurableDelivery]:
        """Observe durable delivery state for this agent only."""
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            return await self._durable_store.list_deliveries(
                agent_id=self._agent.did,
                consumer_id=consumer_id,
                statuses=statuses,
                limit=limit,
            )

    async def purge_expired_durable_deliveries(self) -> int:
        """Run the durable-ledger retention sweep (terminal history only)."""
        async with self._admit_durable_operation():
            await self.initialize_durable_delivery()
            purged = await self._durable_store.purge_expired(agent_id=self._agent.did)
            self._discard_expired_transient_durable_handoffs()
            return purged

    async def shutdown_durable_delivery(self) -> bool:
        """Close admission, drain committed work, then release this owner.

        The state change is linearizable with every durable public API and
        dispatch.  In particular, an owner registration or event transaction
        that started before this method wins admission and is fully reconciled
        before owner release; all later calls fail before they touch storage.
        """
        if self.has_live_durable_admission_in_current_context():
            raise RuntimeError(
                "Cannot shut down durable signal delivery from a live admitted operation"
            )

        async with self._durable_lifecycle_lock:
            completion = self._start_durable_shutdown_completion()

        # A public caller may be cancelled by the bounded agent shutdown
        # wrapper.  Shield the owned teardown so the next caller can either
        # join the same cleanup or retry a genuine cleanup failure.
        await asyncio.shield(completion)
        return not self._durable_shutdown_owner_fenced

    async def wait_for_durable_shutdown_release(self) -> None:
        """Join the eventual owner release after a bounded shutdown fence.

        Normal shutdown has no second task.  A live-process unload can instead
        return promptly with a retained cognition fence, while its owner keeps
        heartbeating until the task settles.  Storage lifecycle owners use this
        explicit join before closing the shared backend.
        """
        completion = self._durable_shutdown_completion
        if completion is not None:
            await asyncio.shield(completion)
        fenced = self._fenced_durable_shutdown_completion
        if fenced is not None:
            await asyncio.shield(fenced)

    def _start_durable_shutdown_completion(self) -> asyncio.Task[None]:
        """Close durable admission and return the one owned teardown task.

        Async callers hold ``_durable_lifecycle_lock`` before calling this
        helper.  The synchronous compatibility seam runs atomically on the
        same event-loop thread, so it can use this exact transition without
        inserting an await between the admission-close decision and task
        ownership.  Keeping that transition here prevents the two public
        shutdown forms from drifting into independent release paths.
        """
        completion = self._durable_shutdown_completion
        if completion is None or self._durable_shutdown_needs_retry(completion):
            self._durable_shutdown = True
            completion = asyncio.create_task(
                self._complete_durable_shutdown(),
                name=f"durable_signal_shutdown:{self._agent.did}",
            )
            completion.add_done_callback(self._observe_durable_shutdown_completion)
            self._durable_shutdown_completion = completion
        return completion

    @staticmethod
    def _durable_shutdown_needs_retry(completion: asyncio.Task[None]) -> bool:
        """Return whether a completed teardown task failed before completion."""
        if not completion.done():
            return False
        if completion.cancelled():
            return True
        return completion.exception() is not None

    @staticmethod
    def _observe_durable_shutdown_completion(task: asyncio.Task[None]) -> None:
        """Harvest detached teardown failures without changing retry semantics."""
        if task.cancelled():
            return
        task.exception()

    async def _complete_durable_shutdown(self) -> None:
        """Reconcile durable state once, leaving failures available for retry."""
        # A drainer is an owner, not a provider callback. Stop it before the
        # admission gate drains so it cannot begin another lease while this
        # dispatcher is giving up its runtime-owner generation.
        for timer in self._durable_cognition_drain_timers.values():
            timer.cancel()
        self._durable_cognition_drain_timers.clear()
        self._started_durable_cognition_consumers.clear()
        drainers = tuple(self._durable_cognition_drainers.values())
        for task in drainers:
            if not task.done():
                task.cancel()
        if drainers:
            await asyncio.gather(*drainers, return_exceptions=True)
        # A retained route has already detached from its dispatch so it does
        # not hold an admission open. Ask it to stop again before draining
        # storage operations; a task that keeps running is fenced below rather
        # than making this public shutdown wait forever.
        for task in tuple(self._retained_durable_cognition_tasks):
            if not task.done():
                task.cancel()
        # No operation that reaches durable storage can still be running
        # beyond this point.  The final repair drain remains necessary for a
        # committed reservation task created immediately before its parent
        # completed.
        await self._durable_admissions_drained.wait()
        await self._drain_outcome_log_tasks()
        await self._drain_post_commit_reservation_repairs()
        retained = await self._wait_for_retained_durable_cognition_cancellation()
        if retained:
            await self._activate_retained_durable_shutdown_fence()
            return

        await self._stop_runtime_owner_heartbeat()
        await self._release_runtime_owner_after_shutdown(mark_owner_stopped=True)

    async def _wait_for_retained_durable_cognition_cancellation(self) -> bool:
        """Request a bounded stop and report whether retained work remains."""
        tasks = tuple(
            task for task in self._retained_durable_cognition_tasks if not task.done()
        )
        for task in tasks:
            task.cancel()
        if not tasks:
            return False
        _done, pending = await asyncio.wait(
            tasks, timeout=_DURABLE_COGNITION_CANCELLATION_GRACE
        )
        return bool(pending)

    async def _activate_retained_durable_shutdown_fence(self) -> None:
        """Keep the managed owner live until cancellation-resistant work settles."""
        self._durable_shutdown_owner_fenced = True
        # Refresh liveness after closing admission. This is deliberately not a
        # public durable operation: public work is closed, but the retained
        # task's exact owner must remain non-reclaimable while it can act.
        async with self._runtime_owner_fence_lock:
            await self._durable_store.heartbeat_runtime_owner(
                agent_id=self._agent.did,
                owner_id=self._durable_delivery_owner,
            )
        self._schedule_runtime_owner_heartbeat()
        await self._release_runtime_owner_after_shutdown(mark_owner_stopped=False)
        if (
            self._fenced_durable_shutdown_completion is None
            or self._fenced_durable_shutdown_completion.done()
        ):
            completion = asyncio.create_task(
                self._complete_fenced_durable_shutdown(),
                name=f"durable_signal_shutdown_fence:{self._agent.did}",
            )
            completion.add_done_callback(self._observe_durable_shutdown_completion)
            self._fenced_durable_shutdown_completion = completion

    async def _complete_fenced_durable_shutdown(self) -> None:
        """Repeatedly request cancellation, then stop the owner only when safe."""
        while self._retained_durable_cognition_tasks:
            tasks = tuple(self._retained_durable_cognition_tasks)
            for task in tasks:
                if not task.done():
                    task.cancel()
            _done, pending = await asyncio.wait(
                tasks, timeout=_DURABLE_COGNITION_CANCELLATION_GRACE
            )
            if pending:
                logger.warning(
                    "Durable cognition still suppresses cancellation during shutdown; "
                    "keeping runtime owner fenced: agent=%s tasks=%s",
                    self._agent.did,
                    len(pending),
                )

        # Once every retained task is terminal, no original cognition can make
        # another external effect. Stop the fence heartbeat before atomically
        # marking the owner stopped, so a delayed heartbeat cannot revive it.
        self._durable_shutdown_owner_fenced = False
        await self._stop_runtime_owner_heartbeat()
        async with self._runtime_owner_fence_lock:
            pass
        await self._drain_retained_durable_cognition_cleanup_tasks()
        await self._release_runtime_owner_after_shutdown(mark_owner_stopped=True)

    async def _release_runtime_owner_after_shutdown(
        self, *, mark_owner_stopped: bool
    ) -> None:
        """Drop initial handoffs and, only when safe, stop this runtime owner."""
        async with self._transient_durable_initial_claim_lock:
            # Drop the raw capability before the durable state becomes retry
            # work. A concurrent worker can therefore receive only the marker
            # once shutdown releases an unactivated reservation.
            for timer in self._transient_durable_handoff_timers.values():
                timer.cancel()
            self._transient_durable_handoff_timers.clear()
            self._transient_durable_handoffs.clear()
            if self._durable_runtime_owner_registration_started:
                await self._durable_store.release_initial_reservations(
                    agent_id=self._agent.did,
                    owner_id=self._durable_delivery_owner,
                    mark_owner_stopped=mark_owner_stopped,
                )
                if mark_owner_stopped:
                    self._durable_runtime_owner_registered = False
                    self._durable_runtime_owner_registration_started = False

    async def _drain_retained_durable_cognition_cleanup_tasks(self) -> None:
        """Harvest cleanup callbacks before their owner can become reclaimable."""
        while self._retained_durable_cognition_cleanup_tasks:
            tasks = tuple(self._retained_durable_cognition_cleanup_tasks)
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )

    async def _drain_outcome_log_tasks(self) -> None:
        """Join every accepted outcome log before releasing shared storage.

        Each task owns a lifecycle reservation, so normally the admission
        event above is set only after this set is empty.  Keep this explicit
        drain as a defensive second half of that contract: a task can complete
        between the event wake-up and this check, and a task cancelled before
        its first turn still runs its done callback to reconcile the
        reservation without ever touching storage.
        """
        while self._outcome_log_tasks:
            tasks = tuple(self._outcome_log_tasks)
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )

    async def _drain_outcome_logs_for_signal(self, signal_id: str) -> None:
        """Join this dispatch's outcome writers before returning its result."""
        while tasks := self._outcome_log_tasks_by_signal.get(signal_id):
            await asyncio.gather(
                *(asyncio.shield(task) for task in tuple(tasks)),
                return_exceptions=True,
            )

    def shutdown(self) -> None:
        """Compatibility teardown for embeddings that cannot await shutdown.

        Production agent shutdown uses :meth:`shutdown_durable_delivery` and
        awaits the owner-scoped durable release.  A synchronous embedding can
        only initiate that same owned state machine: it returns the work to
        the dispatcher-owned joinable task and never races an admitted
        operation by releasing reservations or clearing raw sidecars itself.
        """
        completion = self._durable_shutdown_completion
        if (
            completion is not None
            and completion.done()
            and not self._durable_shutdown_needs_retry(completion)
        ):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "SignalDispatcher.shutdown requires a running event loop; "
                "await shutdown_durable_delivery() before closing storage"
            ) from exc

        # A synchronous method cannot await the lifecycle lock.  On one
        # running asyncio loop it nevertheless executes without an interleave,
        # so this creates the same linearization point and the same owned task
        # as the async method above.  Any already-admitted work is drained by
        # ``_complete_durable_shutdown`` before it releases the runtime owner.
        self._start_durable_shutdown_completion()

    async def _activate_transient_reservations(self, persisted) -> None:
        """Activate every post-commit reservation into its first live lease.

        The caller owns failure cleanup.  Keeping activation free of partial
        recovery is deliberate: if any await below raises after the event has
        committed, every reservation for that event must be returned to the
        marker-only retry path together, including leases whose UPDATE
        succeeded before their readback failed.
        """
        for reservation in persisted.initial_reservations:
            handoff = self._transient_durable_handoffs.get(reservation.delivery_id)
            if handoff is None:
                raise RuntimeError("initial reservation sidecar is unavailable")
            delivery = await self._durable_store.activate_initial_delivery(
                agent_id=self._agent.did,
                consumer_id=reservation.consumer_id,
                delivery_id=reservation.delivery_id,
                initial_lease_owner=self._durable_delivery_owner,
                initial_lease_token=reservation.reservation_token,
            )
            if delivery is None or delivery.lease_expires_at is None:
                raise RuntimeError("initial reservation could not be activated")
            handoff.expires_at = min(
                handoff.retention_until, delivery.lease_expires_at
            )
            self._schedule_transient_durable_handoff_expiry(
                reservation.delivery_id, handoff.expires_at
            )

    async def _requeue_committed_initial_reservations(self, persisted) -> None:
        """Synchronously erase raw sidecars and requeue a failed handoff.

        This coroutine runs in a separate task once an event transaction may
        have committed.  The parent dispatch task can then be cancelled as
        often as needed without interrupting the durable repair.  A failed
        activation is all-or-nothing: an earlier reservation may already be
        ``LEASED`` when a later activation or readback raises, but its
        owner/token capability still lets ``abandon_initial_reservation`` turn
        it into marker-only retry work safely.
        """
        reservations = persisted.initial_reservations
        for reservation in reservations:
            self._discard_transient_durable_handoff(reservation.delivery_id)

        for reservation in reservations:
            while True:
                try:
                    abandoned = await self._durable_store.abandon_initial_reservation(
                        agent_id=self._agent.did,
                        consumer_id=reservation.consumer_id,
                        delivery_id=reservation.delivery_id,
                        owner_id=self._durable_delivery_owner,
                        reservation_token=reservation.reservation_token,
                    )
                except asyncio.CancelledError:
                    # The repair task is normally protected by its parent's
                    # shield, but a direct cancellation (for example during
                    # event-loop teardown) can still arrive at this await.
                    # Retrying the owner/token-conditional release is safe and
                    # ensures the cancellation cannot leave an INITIAL_RESERVED
                    # row behind.
                    continue
                if not abandoned:
                    # A transaction that definitely rolled back reaches this
                    # path too: the dispatcher deliberately keeps its opaque
                    # capability until conditional repair proves there is no
                    # matching committed row.  This is an expected no-op, not
                    # an ownership violation.
                    logger.debug(
                        "Committed-reservation repair found no matching row for %s",
                        reservation.delivery_id,
                    )
                break

    async def _repair_post_commit_reservations(self, persisted) -> None:
        """Finish committed-reservation repair despite repeated cancellation.

        Raw sidecars are discarded before this method creates its task, so
        there is no scheduling window in which cancellation can retain user
        payload.  The separate repair task is shielded from cancellation of
        the dispatch task; repeated cancellations are observed and deferred
        until the repair is done, then the original triggering exception is
        re-raised by the caller.
        """
        for reservation in persisted.initial_reservations:
            self._discard_transient_durable_handoff(reservation.delivery_id)

        repair_task = asyncio.create_task(
            self._requeue_committed_initial_reservations(persisted),
            name=f"durable_signal_post_commit_repair:{self._agent.did}",
        )
        self._post_commit_reservation_repairs.add(repair_task)
        repair_task.add_done_callback(self._post_commit_reservation_repairs.discard)
        while not repair_task.done():
            try:
                await asyncio.shield(repair_task)
            except asyncio.CancelledError:
                # The original failure is retained by the caller.  A repeated
                # cancellation must not strand a committed reservation.
                continue
        # Do not catch a database repair error here.  The caller that is
        # already preserving a persistence/activation exception records the
        # cleanup fault explicitly, while normal callers must see the repair
        # error rather than silently proceeding with an unsafe handoff.
        repair_task.result()

    async def _drain_post_commit_reservation_repairs(self) -> None:
        """Await every dispatcher-owned committed-reservation repair.

        Unlike general agent background work, a reservation repair is part of
        the durable state transition already committed to storage.  It must
        therefore run to completion before the owner release below makes any
        remaining work visible to a restart worker.
        """
        while self._post_commit_reservation_repairs:
            repairs = tuple(self._post_commit_reservation_repairs)
            await asyncio.gather(
                *(asyncio.shield(repair) for repair in repairs),
            )

    def _schedule_runtime_owner_heartbeat(self, *, retry: bool = False) -> None:
        """Keep owner liveness separate from delivery lease countdowns.

        ``retry`` uses an exponential-but-capped delay after a storage error.
        Crucially it still creates a future timer: an exception is observable,
        but never changes into a silent end to runtime ownership heartbeats.
        """
        if self._durable_shutdown and not self._durable_shutdown_owner_fenced:
            return
        if self._runtime_owner_heartbeat_timer is not None:
            self._runtime_owner_heartbeat_timer.cancel()
        interval = max(0.01, self._runtime_owner_stale_after.total_seconds() / 3)
        if retry:
            # Cap retries at the normal interval. This gives a failing backend
            # breathing room without permitting one outage to leave an owner
            # unsupervised for longer than its ordinary heartbeat cadence.
            interval = min(
                interval,
                max(0.01, interval / 8)
                * (2 ** min(self._runtime_owner_heartbeat_failures, 3)),
            )
        self._runtime_owner_heartbeat_timer = asyncio.get_running_loop().call_later(
            interval, self._start_runtime_owner_heartbeat
        )

    async def _stop_runtime_owner_heartbeat(self) -> None:
        """Cancel and drain owner liveness work before closing its store.

        A timer may already have spawned its agent-owned task when boot starts
        rolling back.  Merely cancelling the next ``TimerHandle`` leaves that
        in-flight task free to touch storage after the rollback has closed it.
        """
        if self._runtime_owner_heartbeat_timer is not None:
            self._runtime_owner_heartbeat_timer.cancel()
            self._runtime_owner_heartbeat_timer = None
        task = self._runtime_owner_heartbeat_task
        self._runtime_owner_heartbeat_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _start_runtime_owner_heartbeat(self) -> None:
        self._runtime_owner_heartbeat_timer = None
        if self._durable_shutdown and not self._durable_shutdown_owner_fenced:
            return
        task = self._agent._track_background_task(
            self._heartbeat_runtime_owner(),
            name=f"durable_signal_owner_heartbeat:{self._agent.did}",
        )
        self._runtime_owner_heartbeat_task = task
        task.add_done_callback(self._finish_runtime_owner_heartbeat)

    async def _heartbeat_runtime_owner(self) -> None:
        # The timer can have queued this coroutine immediately before
        # shutdown flips the lifecycle state.  Take the same admission as
        # public APIs so it either completes before teardown or performs no
        # post-close storage work at all.
        if self._durable_shutdown:
            if not self._durable_shutdown_owner_fenced:
                raise _DurableDeliveryShuttingDownError(
                    "Durable signal delivery is shutting down"
                )
            async with self._runtime_owner_fence_lock:
                if not self._durable_shutdown_owner_fenced:
                    return
                await self._durable_store.heartbeat_runtime_owner(
                    agent_id=self._agent.did,
                    owner_id=self._durable_delivery_owner,
                )
            return

        async with self._admit_durable_operation():
            await self._durable_store.heartbeat_runtime_owner(
                agent_id=self._agent.did,
                owner_id=self._durable_delivery_owner,
            )
            # Startup cannot know whether a recently crashed runtime will cross
            # the stale threshold moments later.  Sweep on every owner heartbeat
            # so such unactivated reservations eventually become marker-only
            # retry work without another process restart.
            await self._recover_abandoned_initial_reservations()
            await self._recover_abandoned_leases()

    async def _recover_abandoned_initial_reservations(self) -> int:
        """Requeue stale foreign initial reservations for this tenant."""
        recovery_now = datetime.now(timezone.utc)
        return await self._durable_store.recover_abandoned_initial_reservations(
            agent_id=self._agent.did,
            recovering_owner_id=self._durable_delivery_owner,
            stale_before=recovery_now - self._runtime_owner_stale_after,
            now=recovery_now,
        )

    async def _recover_abandoned_leases(self) -> int:
        """Requeue stopped/stale foreign leases before source redelivery."""
        recovery_now = datetime.now(timezone.utc)
        return await self._durable_store.recover_abandoned_leases(
            agent_id=self._agent.did,
            recovering_owner_id=self._durable_delivery_owner,
            stale_before=recovery_now - self._runtime_owner_stale_after,
            now=recovery_now,
        )

    def _finish_runtime_owner_heartbeat(self, task: asyncio.Task) -> None:
        if self._runtime_owner_heartbeat_task is task:
            self._runtime_owner_heartbeat_task = None
        if task.cancelled():
            if not self._durable_shutdown or self._durable_shutdown_owner_fenced:
                self._runtime_owner_heartbeat_failures += 1
                logger.warning(
                    "Durable signal runtime-owner heartbeat was cancelled; "
                    "scheduling bounded retry"
                )
                self._schedule_runtime_owner_heartbeat(retry=True)
            return
        try:
            task.result()
        except _DurableDeliveryShuttingDownError:
            # A queued timer callback can lose the lifecycle race after
            # shutdown closes admission.  That rejection is normal, but its
            # exception must still be retrieved before returning.
            return
        except Exception:
            logger.exception("Durable signal runtime-owner heartbeat failed")
            if not self._durable_shutdown or self._durable_shutdown_owner_fenced:
                self._runtime_owner_heartbeat_failures += 1
                self._schedule_runtime_owner_heartbeat(retry=True)
            return
        if self._durable_shutdown and not self._durable_shutdown_owner_fenced:
            return
        self._runtime_owner_heartbeat_failures = 0
        self._schedule_runtime_owner_heartbeat()

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

    def _volatile_source_integrity_binding(
        self, signal: Signal, source_event_id: Optional[str]
    ) -> str:
        """Bind a live privacy-mode retry to its normalized original input.

        The durable event intentionally elides user content in volatile privacy
        modes. A per-agent keyed MAC gives a restart-safe equality proof for a
        provider redelivery without persisting sender/content or another raw
        payload projection. The event UUID is deliberately excluded because a
        provider retry creates a fresh signal object for the same source event.

        ``conversations`` is the existing per-agent HKDF branch.  We derive a
        purpose-separated MAC key from it rather than using either a public DID
        or an unkeyed content hash.  If that key hierarchy is unavailable, a
        payload-elided event cannot safely become durable retry work and fails
        closed at the persistence boundary.
        """

        agent_id = self._durable_agent_id()
        envelope = {
            "agent_id": agent_id,
            "source_event_id": source_event_id,
            "source": signal.source,
            "kind": signal.kind,
            "mode": signal.mode.value,
            "target_agent": signal.target_agent,
            "caller": getattr(signal, "caller", None),
            "session_id": signal.session_id,
            "visibility": signal.visibility.value,
            "urgency": signal.urgency.value,
            "origin_trust": getattr(signal.origin_trust, "value", None),
            "dedupe_key": signal.dedupe_key,
            "payload": signal.payload,
        }
        canonical = json.dumps(
            envelope,
            default=lambda value: (
                value.isoformat()
                if isinstance(value, datetime)
                else getattr(value, "value", str(value))
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            base_key = get_agent_key(agent_id, "conversations")
        except (MasterKeyNotConfiguredError, ValueError) as exc:
            raise RuntimeError(
                "Payload-elided durable signal delivery requires the configured "
                "per-agent key hierarchy"
            ) from exc
        mac_key = hmac.new(base_key, _DURABLE_INTEGRITY_CONTEXT, hashlib.sha256).digest()
        return hmac.new(mac_key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    def _durable_agent_id(self) -> str:
        """Return the sole tenant identity allowed to bind durable data."""

        agent_id = getattr(self._agent, "did", None)
        if type(agent_id) is not str or not agent_id:
            raise RuntimeError("Durable signal delivery requires a non-empty agent DID")
        return agent_id

    def _durable_caller_identity_aad(self, event_id: str) -> bytes:
        """Bind encrypted caller identity to this tenant and durable event."""

        return (
            _DURABLE_CALLER_IDENTITY_AAD_PREFIX
            + self._durable_agent_id().encode("utf-8")
            + b":"
            + event_id.encode("utf-8")
        )

    @staticmethod
    def _canonical_caller_identity(caller: Any) -> Optional[str]:
        """Validate the opaque caller identity before it enters a MAC/cipher.

        Caller IDs come from source-owned normalizers.  Treating arbitrary
        objects as strings here would make object ``__str__`` behavior part of
        the security boundary and would make retry identity nondeterministic.
        """

        if caller is None:
            return None
        if type(caller) is not str or not caller or len(caller) > 2048:
            raise ValueError("Signal caller must be a non-empty opaque string")
        return caller

    def _protect_durable_caller_identity(self, signal: Signal) -> str:
        """Return tenant/event-bound encrypted caller storage for a normal row."""

        return self._protect_durable_caller_identity_for_event(
            signal.caller, event_id=signal.id
        )

    def _protect_durable_caller_identity_for_event(
        self, caller: Any, *, event_id: str
    ) -> str:
        """Seal one canonical caller, or retain a keyless opaque label.

        The keyless representation is intentionally not reversible.  Sources
        that support keyless durable persistence already retain their own
        replayable payload identity (for Telegram, ``payload.sender``), while
        the dispatcher preserves only an event-local opaque caller label for
        routing/audit surfaces.  This avoids putting a raw caller in a second
        durable column when no encryption hierarchy is configured.
        """

        caller = self._canonical_caller_identity(caller)
        if caller is None:
            return _DURABLE_CALLER_IDENTITY_NONE
        cipher = get_agent_fernet(self._durable_agent_id())
        if cipher is None:
            return _DURABLE_CALLER_IDENTITY_KEYLESS_PREFIX + secrets.token_urlsafe(32)
        return _DURABLE_CALLER_IDENTITY_PREFIX + cipher.encrypt(
            caller.encode("utf-8"), aad=self._durable_caller_identity_aad(event_id)
        ).decode("ascii")

    def _recover_durable_caller_identity(self, event) -> Optional[str]:
        """Recover a protected caller, refusing legacy/unbound raw state."""

        stored = event.caller_identity
        if stored == _DURABLE_CALLER_IDENTITY_NONE:
            return None
        if isinstance(stored, str) and stored.startswith(
            _DURABLE_CALLER_IDENTITY_KEYLESS_PREFIX
        ):
            token = stored.removeprefix(_DURABLE_CALLER_IDENTITY_KEYLESS_PREFIX)
            if _DURABLE_CALLER_IDENTITY_KEYLESS_TOKEN_RE.fullmatch(token) is None:
                raise RuntimeError(
                    "Durable source event has an invalid keyless caller identity"
                )
            # This is a non-secret opaque caller label, never the source's
            # raw caller identifier.  Returning the complete versioned value
            # keeps callers distinguishable within the durable event contract.
            return stored
        if type(stored) is not str or not stored.startswith(
            _DURABLE_CALLER_IDENTITY_PREFIX
        ):
            raise RuntimeError(
                "Durable source event lacks a protected canonical caller identity"
            )
        cipher = get_agent_fernet(self._durable_agent_id())
        if cipher is None:
            raise RuntimeError(
                "Durable caller identity recovery requires the configured "
                "per-agent key hierarchy"
            )
        try:
            caller = cipher.decrypt(
                stored.removeprefix(_DURABLE_CALLER_IDENTITY_PREFIX).encode("ascii"),
                aad=self._durable_caller_identity_aad(event.event_id),
            ).decode("utf-8")
        except (DecryptionError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                "Durable source event caller identity could not be verified"
            ) from exc
        return self._canonical_caller_identity(caller)

    def _signal_from_durable_event(
        self, event, *, dispatch_signal: Signal
    ) -> Signal:
        """Reconstruct one canonical retry with this dispatch's fresh ID.

        The durable event/source identity stays in the ledger (and in its
        immutable ``source_event_id``); a retry attempt must nevertheless get
        a new signal/outcome identity.  Reusing ``event_id`` here made a retry
        report a result that disagreed with its public handle and collapsed
        distinct outcome-log rows into one ID.
        """

        chain = [
            CausationFrame(
                agent_id=str(frame["agent_id"]),
                source=str(frame["source"]),
                signal_id=str(frame["signal_id"]),
                turn_id=(str(frame["turn_id"]) if frame.get("turn_id") is not None else None),
                depth=int(frame["depth"]),
                emitted_at=datetime.fromisoformat(str(frame["emitted_at"])),
            )
            for frame in event.causation_chain
        ]
        return Signal(
            source=event.source,
            kind=event.kind,
            mode=SignalMode(event.mode),
            payload=copy.deepcopy(event.payload),
            target_agent=event.target_agent,
            visibility=Visibility(event.visibility),
            session_id=event.session_id,
            caller=self._recover_durable_caller_identity(event),
            urgency=Urgency(event.urgency),
            dedupe_key=event.dedupe_key,
            # Source registration reasserts its trust ceiling before routing.
            origin_trust=Trust.UNTRUSTED,
            causation_chain=chain,
            id=dispatch_signal.id,
            arrived_at=dispatch_signal.arrived_at,
        )

    @asynccontextmanager
    async def _renew_durable_cognition_lease(self, delivery: DurableDelivery):
        """Keep a cursor-owned cognition lease alive until its turn settles."""

        stop = asyncio.Event()
        lost: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def report_loss(reason: str) -> None:
            if not lost.done():
                lost.set_result(reason)

        if delivery.lease_expires_at is None or delivery.lease_token is None:
            report_loss("Durable cognition delivery has no live lease")
            yield lost
            return

        async def renew() -> None:
            try:
                expires_at = delivery.lease_expires_at
                while not stop.is_set():
                    remaining = max(
                        0.01, (expires_at - datetime.now(timezone.utc)).total_seconds()
                    )
                    try:
                        await asyncio.wait_for(
                            stop.wait(), timeout=max(0.01, remaining / 3)
                        )
                        return
                    except asyncio.TimeoutError:
                        pass
                    try:
                        renewed = await self.renew_durable_delivery_lease(
                            consumer_id=delivery.consumer_id,
                            delivery_id=delivery.delivery_id,
                            lease_token=delivery.lease_token,
                        )
                    except Exception:
                        logger.exception(
                            "Could not renew durable cognition lease: delivery=%s",
                            delivery.delivery_id,
                        )
                        report_loss("Durable cognition lease renewal failed")
                        return
                    if (
                        renewed is None
                        or renewed.lease_expires_at is None
                        or renewed.lease_token is None
                        or not secrets.compare_digest(
                            renewed.lease_token, delivery.lease_token
                        )
                    ):
                        logger.error(
                            "Lost durable cognition lease while turn was running: delivery=%s",
                            delivery.delivery_id,
                        )
                        report_loss("Durable cognition lease ownership was lost")
                        return
                    expires_at = renewed.lease_expires_at
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Durable cognition lease renewal task failed: delivery=%s",
                    delivery.delivery_id,
                )
                report_loss("Durable cognition lease renewal failed")

        task = asyncio.create_task(
            renew(), name=f"durable_cognition_lease_renewal:{delivery.delivery_id}"
        )
        try:
            yield lost
        finally:
            stop.set()
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.shield(task)

    @staticmethod
    def _legacy_channel_cognition_signal(
        signal: Signal, *, consumer_id: str
    ) -> Signal | None:
        """Return the precise pre-marker envelope eligible for one upgrade.

        Only a live channel redelivery addressed to Core's cursor-owning
        consumer can enter the legacy path.  Removing the one new selector
        marker gives the store an exact normalized envelope to compare with
        the old retained event; it does not make historical rows eligible on
        their own.
        """

        if (
            consumer_id != DURABLE_COGNITION_CONSUMER_ID
            or signal.source != "channel.message"
            or type(signal.payload) is not dict
            or signal.payload.get(DURABLE_COGNITION_MARKER)
            != DURABLE_COGNITION_MARKER_VALUE
        ):
            return None
        legacy_payload = dict(signal.payload)
        legacy_payload.pop(DURABLE_COGNITION_MARKER)
        return replace(signal, payload=legacy_payload)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def _run(
        self,
        signal: Signal,
        start: float,
        *,
        source_event_id: Optional[str],
        durable_admission: asyncio.Future[DurableAdmissionResult] | None = None,
        durable_delivery_consumer_id: Optional[str] = None,
        durable_terminal_consumer_id: Optional[str] = None,
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
                # Normalize the opaque caller once before either the protected
                # normal-row representation or an elided row's keyed MAC sees
                # it. This makes caller identity stable across retries and
                # prevents a user-defined ``__str__`` from entering either
                # security boundary.
                signal.caller = self._canonical_caller_identity(signal.caller)
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
                volatile_integrity_binding = (
                    self._volatile_source_integrity_binding(signal, source_event_id)
                    if durable_projection.payload_elided
                    else None
                )
                # Persisted caller identity is independent of the audit log's
                # redacted display field.  It is encrypted and AAD-bound to
                # this tenant/event for normal rows; privacy-elided rows retain
                # no caller at all and instead bind it through the live MAC.
                committed_reservations = None

                def install_transient_handoffs(persisted) -> None:
                    """Install raw sidecars before the event transaction commits."""
                    nonlocal committed_reservations
                    assert transient_payload is not None
                    retention_until = persisted.retention_until or datetime.now(
                        timezone.utc
                    )
                    for reservation in persisted.initial_reservations:
                        self._transient_durable_handoffs[reservation.delivery_id] = (
                            _TransientDurableHandoff(
                                payload=copy.deepcopy(transient_payload),
                                consumer_id=reservation.consumer_id,
                                created_at=reservation.created_at,
                                retention_until=retention_until,
                                # Do not start any volatile lease countdown
                                # before the event commit. Post-commit
                                # activation replaces this retention bound with
                                # the actual lease deadline.
                                expires_at=retention_until,
                                initial_lease_token=(
                                    reservation.reservation_token
                                ),
                            )
                        )
                    # ``persist_signal`` may be cancelled after its transaction
                    # commits but before its await returns. Retain the opaque
                    # reservation capabilities here so that boundary can still
                    # be repaired. ``on_rollback`` cannot distinguish a real
                    # rollback from cancellation after the driver committed,
                    # so conditional repair below resolves either outcome.
                    committed_reservations = persisted

                def discard_rolled_back_handoffs(persisted) -> None:
                    for reservation in persisted.initial_reservations:
                        self._discard_transient_durable_handoff(
                            reservation.delivery_id
                        )
                    # Keep ``committed_reservations`` even though the callback
                    # convention calls this ``on_rollback``. SQLite can raise
                    # CancelledError from ``await conn.commit()`` after its
                    # worker has committed; clearing the owner/token
                    # capabilities here would strand that visible
                    # INITIAL_RESERVED row. A confirmed rollback makes the
                    # later conditional repair a harmless no-op.

                persistence_kwargs = {
                    "agent_id": self._agent.did,
                    "source_event_id": source_event_id,
                    "retention_days": registration.retention_days,
                    # A volatile privacy projection must never make a
                    # pre-existing payload-correlated wait miss its normalized
                    # live signal. DurableSignalStore consumes this only while
                    # materializing the initial deliveries; its event row and
                    # every later replay retain the projected payload above.
                    **transient_selector_payload,
                    "initial_lease_owner": (
                        self._durable_delivery_owner
                        if durable_projection.payload_elided
                        else None
                    ),
                    "integrity_binding": volatile_integrity_binding,
                    # A duplicate reuses its retained, AAD-bound caller.  Do
                    # not require current key material before the store has
                    # determined whether this attempt actually creates an
                    # event; otherwise a lost/rotated key masks the claimed
                    # delivery recovery path below.
                    "caller_identity_factory": (
                        None
                        if durable_projection.payload_elided
                        else lambda: self._protect_durable_caller_identity(signal)
                    ),
                    "before_commit": (
                        install_transient_handoffs
                        if durable_projection.payload_elided
                        else None
                    ),
                    "on_rollback": (
                        discard_rolled_back_handoffs
                        if durable_projection.payload_elided
                        else None
                    ),
                }
                if durable_projection.payload_elided:
                    # Keep a local initial claimant outside the store's
                    # uncommitted-transaction window. The synchronous
                    # ``before_commit`` callback has already installed raw
                    # sidecars by the time this await reaches commit, and the
                    # same lock remains held until the row is visible.
                    async with self._transient_durable_initial_claim_lock:
                        try:
                            persisted = await self._durable_store.persist_signal(
                                durable_projection.signal,
                                **persistence_kwargs,
                            )
                        except BaseException:
                            # The before-commit hook already gave us the
                            # capability set. If cancellation arrives after
                            # commit but before this await returns, use it to
                            # erase raw state and requeue every reservation.
                            # A confirmed rollback leaves no matching rows, so
                            # this is harmless before commit too.
                            if committed_reservations is not None:
                                try:
                                    await self._repair_post_commit_reservations(
                                        committed_reservations
                                    )
                                except BaseException as cleanup_exc:
                                    # The event-persistence failure remains
                                    # authoritative, but a failed repair must
                                    # never disappear into a broad exception
                                    # handler. The exception log provides the
                                    # operator the exact cleanup failure.
                                    logger.exception(
                                        "Durable post-commit reservation repair "
                                        "failed while preserving persistence "
                                        "failure: %s",
                                        cleanup_exc,
                                    )
                            raise
                        if persisted.created:
                            try:
                                await self._activate_transient_reservations(persisted)
                            except BaseException as exc:
                                # The event transaction has already committed.
                                # Activation can have written a first lease
                                # before failing its readback. Recover the
                                # entire committed batch before preserving the
                                # original failure/cancellation below.
                                try:
                                    await self._repair_post_commit_reservations(
                                        persisted
                                    )
                                except BaseException as cleanup_exc:
                                    # Preserve the activation failure below,
                                    # but surface any independently failed
                                    # repair rather than swallowing it.
                                    logger.exception(
                                        "Durable post-commit reservation repair "
                                        "failed while preserving activation "
                                        "failure: %s",
                                        cleanup_exc,
                                    )
                                if isinstance(exc, Exception):
                                    logger.error(
                                        "Durable signal event %s committed but "
                                        "initial delivery activation failed",
                                        signal.id,
                                        exc_info=(
                                            type(exc), exc, exc.__traceback__
                                        ),
                                    )
                                    return self._fail(
                                        signal,
                                        start,
                                        Status.FAILED,
                                        error=(
                                            "Durable signal delivery activation "
                                            "failed after commit: "
                                            f"{type(exc).__name__}: {exc}"
                                        ),
                                        registration=registration,
                                    )
                                raise
                else:
                    persisted = await self._durable_store.persist_signal(
                        durable_projection.signal,
                        **persistence_kwargs,
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
        if (
            durable_admission is not None
            and durable_delivery_consumer_id is None
            and durable_terminal_consumer_id is None
            and not durable_admission.done()
        ):
            durable_admission.set_result(
                DurableAdmissionResult(
                    (
                        DurableAdmissionDisposition.COMMITTED
                        if persisted.created
                        else DurableAdmissionDisposition.DUPLICATE
                    ),
                    signal.id,
                )
            )

        # The row that already exists controls replay authority.  A retry can
        # cross a privacy transition (EPHEMERAL -> NORMAL or back again), so
        # the retry's *current* projection cannot decide whether live content
        # is safe to use.  Presence of the persisted integrity row means the
        # original event deliberately elided content and requires this exact
        # keyed-MAC-verified live envelope.
        recorded_binding = None
        use_verified_live_signal = False
        if not persisted.created:
            recorded_binding = await self._durable_store.get_event_integrity(
                agent_id=self._agent.did, event_id=persisted.event_id
            )
            if recorded_binding is not None:
                live_integrity_binding = (
                    volatile_integrity_binding
                    or self._volatile_source_integrity_binding(signal, source_event_id)
                )
                if not secrets.compare_digest(recorded_binding, live_integrity_binding):
                    return self._fail(
                        signal,
                        start,
                        Status.FAILED,
                        error=(
                            "Privacy-elided durable source event did not match its "
                            "original normalized integrity binding"
                        ),
                        registration=registration,
                    )
                use_verified_live_signal = True
            elif durable_projection.payload_elided:
                # A new privacy-elided retry must never attach live caller or
                # content to a legacy/non-elided event with no MAC row.
                return self._fail(
                    signal,
                    start,
                    Status.FAILED,
                    error=(
                        "Privacy-elided durable source event lacks its required "
                        "integrity binding"
                    ),
                    registration=registration,
                )

            # Before the marker-selected cognition consumer existed, normal
            # channel rows were retained without this delivery or a protected
            # caller field.  Upgrade one only when its provider redelivers the
            # exact canonical envelope.  This is intentionally after the
            # privacy-integrity branch above: payload-elided rows never take
            # this path and no historical scan/backfill is performed.
            legacy_signal = self._legacy_channel_cognition_signal(
                signal, consumer_id=durable_delivery_consumer_id or ""
            )
            if legacy_signal is not None and source_event_id is not None:
                try:
                    upgraded_legacy_delivery = await self._durable_store.upgrade_legacy_delivery_for_redelivery(
                        agent_id=self._agent.did,
                        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
                        event_id=persisted.event_id,
                        source_event_id=source_event_id,
                        expected_signal=legacy_signal,
                        caller_identity_factory=(
                            lambda: self._protect_durable_caller_identity_for_event(
                                signal.caller, event_id=persisted.event_id
                            )
                        ),
                    )
                    if not upgraded_legacy_delivery:
                        # ``False`` deliberately carries no admission meaning:
                        # the row may be absent, mismatched, expired, already
                        # terminal, or an ordinary FAILED delivery. The exact
                        # selected-delivery read/claim below is the only
                        # cursor-receipt proof.
                        logger.debug(
                            "Legacy durable channel redelivery was not upgraded; "
                            "requiring selected delivery proof: event=%s",
                            persisted.event_id,
                        )
                except Exception:
                    logger.exception(
                        "Could not upgrade legacy durable channel redelivery: event=%s",
                        persisted.event_id,
                    )
                    return self._fail(
                        signal,
                        start,
                        Status.FAILED,
                        error="Legacy durable channel redelivery upgrade failed",
                        registration=registration,
                    )

        if durable_delivery_consumer_id is not None:
            # Do not publish a cursor receipt merely because the source event
            # deduplicated. Legacy rows can lack the selected delivery, and an
            # ordinary FAILED delivery is explicitly non-ACKable. The route
            # below first proves an ACKable terminal row or claims an exact
            # Core-owned retry lease, then resolves durable_admission.
            return await self._route_durable_cognition_delivery(
                signal,
                registration,
                start,
                persisted_event_id=persisted.event_id,
                consumer_id=durable_delivery_consumer_id,
                durable_admission=durable_admission,
                durable_created=persisted.created,
                use_live_signal=(
                    durable_projection.payload_elided and persisted.created
                )
                or use_verified_live_signal,
            )

        if durable_terminal_consumer_id is not None:
            return await self._route_durable_terminal_delivery(
                signal,
                registration,
                start,
                persisted_event_id=persisted.event_id,
                consumer_id=durable_terminal_consumer_id,
                durable_admission=durable_admission,
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

        return await self._route_after_durable_persistence(
            signal, registration, start
        )

    async def _route_durable_terminal_delivery(
        self,
        signal: Signal,
        registration: SourceRegistration,
        start: float,
        *,
        persisted_event_id: str,
        consumer_id: str,
        durable_admission: asyncio.Future[DurableAdmissionResult] | None,
    ) -> SignalResult:
        """Mark one selected malformed-ingress delivery terminal and ACKable.

        No channel content reaches cognition on this path. The durable row is
        nevertheless claimed and terminally settled so a polling provider can
        advance only after an idempotent, restart-visible disposition exists.
        """

        delivery = await self.claim_durable_delivery_for_event(
            consumer_id=consumer_id,
            event_id=persisted_event_id,
            executor_id=self._durable_delivery_owner,
        )
        if delivery is None:
            existing = await self.get_durable_delivery_for_event(
                consumer_id=consumer_id,
                event_id=persisted_event_id,
            )
            if existing is not None and existing.status == TERMINAL_ACKABLE:
                if durable_admission is not None and not durable_admission.done():
                    durable_admission.set_result(
                        DurableAdmissionResult(DurableAdmissionDisposition.TERMINAL, signal.id)
                    )
                return self._success(signal, start, registration)
            return self._fail(
                signal,
                start,
                Status.FAILED,
                error="Durable terminal ingress delivery is unavailable; source must retry",
                registration=registration,
            )

        settled = await self.nack_durable_delivery(
            consumer_id=consumer_id,
            delivery_id=delivery.delivery_id,
            lease_token=delivery.lease_token or "",
            error="Malformed Telegram update recorded as terminal ingress disposition",
            terminal=True,
            terminal_ackable=True,
        )
        if settled is None or settled.status != TERMINAL_ACKABLE:
            return self._fail(
                signal,
                start,
                Status.FAILED,
                error="Durable terminal ingress disposition could not be settled",
                registration=registration,
            )
        if durable_admission is not None and not durable_admission.done():
            durable_admission.set_result(
                DurableAdmissionResult(DurableAdmissionDisposition.TERMINAL, signal.id)
            )
        return self._success(signal, start, registration)

    async def _route_durable_cognition_delivery(
        self,
        signal: Signal,
        registration: SourceRegistration,
        start: float,
        *,
        persisted_event_id: str,
        consumer_id: str,
        durable_admission: asyncio.Future[DurableAdmissionResult] | None,
        durable_created: bool,
        use_live_signal: bool,
        claimed_delivery: DurableDelivery | None = None,
        retry_delay: timedelta = timedelta(),
    ) -> SignalResult:
        """Route cursor-owned work and ACK only its durable cognition lease."""
        delivery = claimed_delivery
        if delivery is None:
            delivery = await self.claim_durable_delivery_for_event(
                consumer_id=consumer_id,
                event_id=persisted_event_id,
                executor_id=self._durable_delivery_owner,
            )
        if delivery is None:
            existing = await self.get_durable_delivery_for_event(
                consumer_id=consumer_id,
                event_id=persisted_event_id,
            )
            if existing is not None and existing.status in {
                ACKNOWLEDGED,
                TERMINAL_ACKABLE,
            }:
                if durable_admission is not None and not durable_admission.done():
                    durable_admission.set_result(
                        DurableAdmissionResult(
                            (
                                DurableAdmissionDisposition.DUPLICATE
                                if existing.status == ACKNOWLEDGED
                                else DurableAdmissionDisposition.TERMINAL
                            ),
                            signal.id,
                        )
                    )
                return self._fail(
                    signal,
                    start,
                    Status.COALESCED,
                    error=(
                        (
                            "Duplicate source event ID already acknowledged by "
                            f"durable consumer {consumer_id}"
                            if existing.status == ACKNOWLEDGED
                            else "Duplicate source event ID already completed as a "
                            f"terminal no-op by durable consumer {consumer_id}"
                        )
                    ),
                    registration=registration,
                )
            return self._fail(
                signal,
                start,
                Status.FAILED,
                error=(
                    "Durable cognition delivery is unavailable; source must "
                    "retry without advancing its cursor"
                ),
                registration=registration,
            )

        # This exact selected delivery is now leased to the current dispatcher
        # owner.  That durable lease is the retry authority a cursor-owning
        # producer needs; later cognition/NACK work can no longer make an
        # absent or ordinary-FAILED row look ACKable. Resolve only after the
        # claim succeeds, not at source-event deduplication above.
        if durable_admission is not None and not durable_admission.done():
            durable_admission.set_result(
                DurableAdmissionResult(
                    (
                        DurableAdmissionDisposition.COMMITTED
                        if durable_created
                        else DurableAdmissionDisposition.DUPLICATE
                    ),
                    signal.id,
                )
            )

        # A retried persistence-allowed event must execute the canonical
        # durable envelope, not whatever a duplicate provider callback happens
        # to carry. Volatile privacy modes retain no content in the ledger, so
        # their live callback is usable only after the integrity check above.
        try:
            routing_signal = (
                signal
                if use_live_signal
                else self._signal_from_durable_event(
                    delivery.event, dispatch_signal=signal
                )
            )
        except Exception:
            # A claim without a recoverable canonical caller must never wait
            # for its lease to expire.  Release this exact token before
            # producing the ordinary audited failure so cursor-owning ingress
            # can NACK/retry rather than treating persistence as an ACK.
            try:
                await self.nack_durable_delivery(
                    consumer_id=consumer_id,
                    delivery_id=delivery.delivery_id,
                    lease_token=delivery.lease_token or "",
                    error="Durable cognition caller recovery failed",
                    retry_delay=retry_delay,
                )
            except Exception:
                logger.exception(
                    "Could not release durable cognition lease after caller recovery failure: "
                    "delivery=%s",
                    delivery.delivery_id,
                )
            return self._fail(
                signal,
                start,
                Status.FAILED,
                error="Durable cognition caller recovery failed",
                registration=registration,
            )

        routing_task: asyncio.Task[SignalResult] | None = None
        deferred_outcomes: list[_DeferredOutcomeLog] = []
        lease_loss_reason: Optional[str] = None
        try:
            async with self._renew_durable_cognition_lease(delivery) as lease_lost:
                # The routing task inherits this task-local collector.  Its
                # ordinary _success/_fail calls still build the complete
                # audit, but no route-level outcome reaches signal_log before
                # the exact delivery ACK/NACK decision below.
                deferred_token = self._deferred_outcome_logs.set(deferred_outcomes)
                coalescing_token = self._durable_retry_skips_coalescing.set(
                    delivery.attempts > 1
                )
                try:
                    routing_task = asyncio.create_task(
                        self._route_after_durable_persistence(
                            routing_signal,
                            registration,
                            start,
                        ),
                        name=f"durable_cognition_route:{delivery.delivery_id}",
                    )
                finally:
                    self._durable_retry_skips_coalescing.reset(coalescing_token)
                    self._deferred_outcome_logs.reset(deferred_token)

                completed, _ = await asyncio.wait(
                    {routing_task, lease_lost},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if lease_lost in completed:
                    lease_loss_reason = lease_lost.result()
                    if routing_task not in completed:
                        settled = await self._cancel_durable_cognition_routing_task(
                            routing_task,
                            delivery=delivery,
                        )
                        if not settled:
                            result = self._failure_result(
                                signal,
                                start,
                                error=(
                                    f"{lease_loss_reason}; cognition cancellation "
                                    "is still draining under the live dispatcher owner"
                                ),
                            )
                            self._finalize_deferred_durable_outcome(
                                deferred_outcomes,
                                fallback_signal=routing_signal,
                                fallback_registration=registration,
                                result=result,
                            )
                            return result
                assert routing_task is not None
                result = self._completed_durable_cognition_result(
                    routing_task,
                    signal=routing_signal,
                    start=start,
                )
        except asyncio.CancelledError:
            if routing_task is not None:
                # A caller cancellation is not a durable receipt.  Never join
                # an uncooperative child here: retain it and release the exact
                # lease only after its done callback has harvested it.
                if not routing_task.done():
                    routing_task.cancel()
                self._retain_durable_cognition_task(routing_task, delivery=delivery)
            raise

        if result is None:
            # A cooperative cancellation has no route-level result.  Its
            # exact lease may still be valid, so release it now rather than
            # waiting for expiry.  The final audit remains one lease-loss row.
            failure = self._failure_result(
                signal,
                start,
                error=lease_loss_reason or "Durable cognition routing was cancelled",
            )
            try:
                released = await self.nack_durable_delivery(
                    consumer_id=consumer_id,
                    delivery_id=delivery.delivery_id,
                    lease_token=delivery.lease_token or "",
                    error=failure.error or "Durable cognition routing was cancelled",
                    retry_delay=retry_delay,
                )
                if released is None:
                    await self.release_durable_delivery_after_task(
                        consumer_id=consumer_id,
                        delivery_id=delivery.delivery_id,
                        lease_token=delivery.lease_token or "",
                        error=failure.error or "Durable cognition routing was cancelled",
                    )
            except Exception:
                logger.exception(
                    "Could not release durable cognition lease after cancellation: "
                    "delivery=%s",
                    delivery.delivery_id,
                )
            self._finalize_deferred_durable_outcome(
                deferred_outcomes,
                fallback_signal=routing_signal,
                fallback_registration=registration,
                result=failure,
            )
            return failure

        if result.status is Status.OK:
            # A route and renewal-loss notification can finish in the same
            # event-loop turn.  The completed turn has already performed its
            # effects, so try this exact token ACK before treating ownership
            # as lost.  An ACK rejection is an honest at-least-once outcome,
            # not a contradictory NACK claiming those effects never ran.
            try:
                acknowledged = await self.ack_durable_delivery(
                    consumer_id=consumer_id,
                    delivery_id=delivery.delivery_id,
                    lease_token=delivery.lease_token or "",
                )
            except Exception:
                logger.exception(
                    "Could not acknowledge completed durable cognition lease: "
                    "delivery=%s",
                    delivery.delivery_id,
                )
                acknowledged = False
            if acknowledged:
                if durable_admission is not None and not durable_admission.done():
                    durable_admission.set_result(
                        DurableAdmissionResult(
                            DurableAdmissionDisposition.COMMITTED, signal.id
                        )
                    )
                self._finalize_deferred_durable_outcome(
                    deferred_outcomes,
                    fallback_signal=routing_signal,
                    fallback_registration=registration,
                    result=result,
                )
                return result

            ack_rejected = self._failure_result(
                signal,
                start,
                error=(
                    "Cognition completed, but its exact durable delivery ACK "
                    "was not accepted; effects may already have occurred"
                ),
            )
            try:
                # If the ACK missed only because the nominal lease deadline
                # passed, this exact managed token can now safely requeue: the
                # route task is terminal.  A true ownership transfer makes
                # this a no-op, so we never overwrite another executor or
                # invent a contradictory provider NACK.
                await self.release_durable_delivery_after_task(
                    consumer_id=consumer_id,
                    delivery_id=delivery.delivery_id,
                    lease_token=delivery.lease_token or "",
                    error="Completed cognition delivery ACK was not accepted",
                )
            except Exception:
                logger.exception(
                    "Could not release completed durable cognition after ACK rejection: "
                    "delivery=%s",
                    delivery.delivery_id,
                )
            self._finalize_deferred_durable_outcome(
                deferred_outcomes,
                fallback_signal=routing_signal,
                fallback_registration=registration,
                result=ack_rejected,
            )
            return ack_rejected

        # Rate limits, quiet hours, coalescing, and cognition failures are
        # recoverable for cursor-owning ingress. Validation/cycle refusal is a
        # proven terminal no-op and may be acknowledged idempotently.
        terminal = result.status in {Status.DROPPED_VALIDATION, Status.DROPPED_CYCLE}
        released = await self.nack_durable_delivery(
            consumer_id=consumer_id,
            delivery_id=delivery.delivery_id,
            lease_token=delivery.lease_token or "",
            error=result.error or f"Cognition delivery returned {result.status.value}",
            retry_delay=retry_delay,
            terminal=terminal,
            terminal_ackable=terminal,
        )
        if released is None:
            # An ordinary NACK correctly refuses an expired lease.  This
            # dispatcher still owns a live managed token, however, so it can
            # make one narrow owner/token-conditional transition after the
            # route task has settled. Preserve a proven terminal no-op in that
            # same atomic update; never turn it into an ordinary retry/failed
            # row and then tell the provider it may advance its cursor.
            released = await self.release_durable_delivery_after_task(
                consumer_id=consumer_id,
                delivery_id=delivery.delivery_id,
                lease_token=delivery.lease_token or "",
                error=result.error or f"Cognition delivery returned {result.status.value}",
                terminal=terminal,
                terminal_ackable=terminal,
            )
        # TERMINAL is an external receipt, not a route-level interpretation.
        # Publish it only after this exact delivery row confirms the durable
        # terminal-ackable state. A lost lease/ownership transfer remains
        # non-ACKable and the provider must redeliver.
        terminal_persisted = (
            terminal and released is not None and released.status == TERMINAL_ACKABLE
        )
        if (
            terminal_persisted
            and durable_admission is not None
            and not durable_admission.done()
        ):
            durable_admission.set_result(
                DurableAdmissionResult(DurableAdmissionDisposition.TERMINAL, signal.id)
            )
        self._finalize_deferred_durable_outcome(
            deferred_outcomes,
            fallback_signal=routing_signal,
            fallback_registration=registration,
            result=result,
        )
        return result

    async def _cancel_durable_cognition_routing_task(
        self,
        task: asyncio.Task[SignalResult],
        *,
        delivery: DurableDelivery,
    ) -> bool:
        """Request cancellation, but never let an uncooperative turn pin ingress.

        ``asyncio.wait`` observes completion without propagating a child
        ``CancelledError`` into this dispatch.  If the bounded grace expires,
        the task remains under dispatcher ownership and retains its managed
        owner heartbeat until it finally settles.
        """

        if not task.done():
            task.cancel()
        try:
            completed, _ = await asyncio.wait(
                {task}, timeout=_DURABLE_COGNITION_CANCELLATION_GRACE
            )
        except asyncio.CancelledError:
            self._retain_durable_cognition_task(task, delivery=delivery)
            raise
        if task in completed:
            return True
        self._retain_durable_cognition_task(task, delivery=delivery)
        logger.warning(
            "Durable cognition ignored cancellation after lease loss; retained task: "
            "delivery=%s",
            delivery.delivery_id,
        )
        return False

    def _retain_durable_cognition_task(
        self,
        task: asyncio.Task[SignalResult],
        *,
        delivery: DurableDelivery,
    ) -> None:
        """Own and harvest a route task that outlives its ingress dispatch."""

        if task in self._retained_durable_cognition_tasks:
            return
        self._retained_durable_cognition_tasks.add(task)

        def completed(completed_task: asyncio.Task[SignalResult]) -> None:
            self._retained_durable_cognition_tasks.discard(completed_task)
            try:
                completed_task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                # The route's final disposition was already made visible when
                # retention began.  Harvest this late exception so it cannot
                # become an unobserved task failure or a second audit row.
                logger.exception(
                    "Retained durable cognition task failed after lease loss: "
                    "delivery=%s",
                    delivery.delivery_id,
                )

            if self._durable_shutdown:
                # Graceful shutdown marks this managed owner stopped.  A
                # replacement's owner-aware recovery then owns requeueing;
                # touching the closing backend from this late callback would
                # violate the lifecycle admission boundary.
                return
            cleanup = asyncio.create_task(
                self._release_retained_durable_cognition_task(delivery),
                name=(
                    "durable_cognition_retained_cleanup:"
                    f"{delivery.delivery_id}"
                ),
            )
            self._retained_durable_cognition_cleanup_tasks.add(cleanup)

            def harvest_cleanup(cleanup_task: asyncio.Task[None]) -> None:
                self._retained_durable_cognition_cleanup_tasks.discard(cleanup_task)
                if cleanup_task.cancelled():
                    return
                try:
                    cleanup_task.result()
                except Exception:
                    logger.exception(
                        "Retained durable cognition cleanup failed: delivery=%s",
                        delivery.delivery_id,
                    )

            cleanup.add_done_callback(harvest_cleanup)

        task.add_done_callback(completed)

    async def _release_retained_durable_cognition_task(
        self, delivery: DurableDelivery
    ) -> None:
        """Requeue one exact retained lease only after its task is terminal."""

        try:
            released = await self.release_durable_delivery_after_task(
                consumer_id=delivery.consumer_id,
                delivery_id=delivery.delivery_id,
                lease_token=delivery.lease_token or "",
                error="Cognition task settled after durable lease loss",
            )
        except _DurableDeliveryShuttingDownError:
            return
        except Exception:
            logger.exception(
                "Could not release retained durable cognition lease: delivery=%s",
                delivery.delivery_id,
            )
            return
        if released is None:
            logger.warning(
                "Retained durable cognition lease was no longer owned at cleanup: "
                "delivery=%s",
                delivery.delivery_id,
            )

    @staticmethod
    def _completed_durable_cognition_result(
        task: asyncio.Task[SignalResult],
        *,
        signal: Signal,
        start: float,
    ) -> Optional[SignalResult]:
        """Return a settled route result without leaking task cancellation."""

        try:
            return task.result()
        except asyncio.CancelledError:
            return None
        except Exception as exc:
            logger.exception(
                "Durable cognition route task failed outside its result contract: "
                "signal=%s",
                signal.id,
            )
            return SignalResult(
                signal_id=signal.id,
                status=Status.FAILED,
                mode=signal.mode,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _finalize_deferred_durable_outcome(
        self,
        outcomes: list[_DeferredOutcomeLog],
        *,
        fallback_signal: Signal,
        fallback_registration: SourceRegistration,
        result: SignalResult,
    ) -> None:
        """Write exactly one durable-cognition outcome after lease finality."""

        if len(outcomes) == 1:
            outcome = outcomes[0]
            self._schedule_outcome_log(
                outcome.signal,
                outcome.registration,
                result,
                audit=outcome.audit,
            )
            return
        if outcomes:
            logger.error(
                "Durable cognition route produced multiple deferred outcomes; "
                "writing one final outcome: signal=%s count=%s",
                result.signal_id,
                len(outcomes),
            )
        self._schedule_outcome_log(
            fallback_signal,
            fallback_registration,
            result,
        )

    async def _route_after_durable_persistence(
        self,
        signal: Signal,
        registration: SourceRegistration,
        start: float,
    ) -> SignalResult:
        """Run the post-commit policy and cognition stages for a signal."""
        # Step 3: quiet-hours
        if self._in_quiet_hours(signal, registration.attention_policy):
            return self._fail(
                signal,
                start,
                Status.DROPPED_QUIET_HOURS,
                error="Source is in quiet hours and urgency is below override threshold",
                registration=registration,
            )

        # Step 4: coalescing. A second durable lease is a retry of this exact
        # delivery, so reapplying the process-local duplicate cache would make
        # a failed cognition turn COALESCED forever. The durable lease remains
        # its idempotency boundary.
        if (
            signal.dedupe_key is not None
            and not self._durable_retry_skips_coalescing.get()
        ):
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

        async with self._locks.acquire(
            registration.resources,
            label=f"{signal.source} {signal.kind}",
        ):
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
            # A signal that carries a session was filed FROM that chat window and
            # its wake is routed back there (#1809), so the dispatch span joins
            # that session's Timeline band instead of opening one of its own
            # (#2940). ``Signal.session_id`` is None for system-initiated work
            # (cron ticks, webhooks) — that and any sentinel stamp nothing.
            **session_span_attributes(signal.session_id),
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

        # Some COGNITION sources are volatile wakeups over durable work.  The
        # producer's process-local task handle cannot prove that another worker
        # has not withdrawn that work, so let the execution worker revalidate
        # against the source's durable authority immediately before it builds
        # or runs a cognition turn.  Missing hooks preserve existing sources.
        validate_execution = getattr(
            self._agent,
            "validate_cognition_signal_execution",
            None,
        )
        if callable(validate_execution):
            validation_error = validate_execution(signal)
            if asyncio.iscoroutine(validation_error):
                validation_error = await validation_error
            if validation_error is not None:
                return self._fail(
                    signal,
                    start,
                    Status.DROPPED_VALIDATION,
                    error=str(validation_error),
                    registration=registration,
                    audit=audit,
                )

        async def await_monitored_execution(execution):
            """Race cognition against source-owned durable withdrawal."""

            async def execute_with_tracking():
                result = await execution
                # A monitored cognition turn runs in ``execution_task``.
                # ContextVar writes belong to that task and are invisible to
                # the dispatcher's parent task, so carry the audit value back
                # explicitly with the result.
                from kestrel_sovereign.agent.context_manager import (
                    get_current_injection_tracking,
                )

                return result, get_current_injection_tracking()

            monitor_execution = getattr(
                self._agent,
                "monitor_cognition_signal_execution",
                None,
            )
            if not callable(monitor_execution):
                result, tracking = await execute_with_tracking()
                return result, None, tracking

            try:
                monitor = monitor_execution(signal)
            except BaseException:
                if inspect.iscoroutine(execution):
                    execution.close()
                raise
            if not inspect.isawaitable(monitor):
                if monitor is not None:
                    if inspect.iscoroutine(execution):
                        execution.close()
                    return None, str(monitor), None
                result, tracking = await execute_with_tracking()
                return result, None, tracking

            execution_task = asyncio.create_task(
                execute_with_tracking(),
                name=f"signal_cognition:{signal.id}",
            )
            monitor_task = asyncio.create_task(
                monitor,
                name=f"signal_cognition_monitor:{signal.id}",
            )
            try:
                done, _ = await asyncio.wait(
                    {execution_task, monitor_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if monitor_task in done:
                    withdrawal = await monitor_task
                    if withdrawal is not None:
                        if not execution_task.done():
                            execution_task.cancel()
                        await asyncio.gather(
                            execution_task,
                            return_exceptions=True,
                        )
                        return None, str(withdrawal), None
                result, tracking = await execution_task
                return result, None, tracking
            finally:
                for task in (execution_task, monitor_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    execution_task,
                    monitor_task,
                    return_exceptions=True,
                )
                finish_execution = getattr(
                    self._agent,
                    "finish_cognition_signal_execution",
                    None,
                )
                if callable(finish_execution):
                    finished = finish_execution(signal)
                    if inspect.isawaitable(finished):
                        await finished

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

        execution_withdrawal = None
        injection_tracking = None
        try:
            if process_input_kwargs:
                result, execution_withdrawal, injection_tracking = (
                    await await_monitored_execution(
                    self._agent.process_input(prompt, **process_input_kwargs)
                    )
                )
            else:
                result, execution_withdrawal, injection_tracking = (
                    await await_monitored_execution(
                    self._agent.process_input(prompt)
                    )
                )
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

        if execution_withdrawal is not None:
            if receipt_tool_registered:
                clear_receipt = getattr(
                    self._agent, "clear_constitution_receipt_tool", None
                )
                if callable(clear_receipt):
                    clear_receipt()
            return self._fail(
                signal,
                start,
                Status.DROPPED_VALIDATION,
                error=execution_withdrawal,
                registration=registration,
                audit=audit,
            )

        # Codex round-13/14 P2: surface the agent's actual
        # injected/dropped clause tracking from the budget-aware
        # assembler into the signal_log audit. The tracking is
        # published per-async-task via a ContextVar in
        # `kestrel_sovereign.agent.context_manager`, so concurrent
        # COGNITION dispatches don't race on a shared attribute.
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
        self._schedule_outcome_log(signal, registration, result, audit=audit)
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
        result = self._failure_result(signal, start, error=error, status=status)
        if registration is not None:
            self._schedule_outcome_log(signal, registration, result, audit=audit)
        return result

    @staticmethod
    def _failure_result(
        signal: Signal,
        start: float,
        *,
        error: Optional[str],
        status: Status = Status.FAILED,
    ) -> SignalResult:
        """Build a failure result without committing an outcome yet."""

        return SignalResult(
            signal_id=signal.id,
            status=status,
            mode=signal.mode,
            duration_ms=int((time.monotonic() - start) * 1000),
            error=error,
        )

    def _schedule_outcome_log(
        self,
        signal: Signal,
        registration: SourceRegistration,
        result: SignalResult,
        *,
        audit: Optional[_ConstitutionAudit] = None,
    ) -> None:
        """Transfer an admitted dispatch's lifetime to its outcome writer.

        ``_success`` and ``_fail`` run before their parent dispatch releases
        its admission.  Reserve the writer synchronously *before* scheduling
        it, so shutdown cannot observe zero admitted work in the gap before a
        queued task gets its first event-loop turn.  The task's done callback
        is the sole release point, which also covers cancellation before its
        coroutine body starts.
        """
        deferred = self._deferred_outcome_logs.get()
        if deferred is not None:
            deferred.append(
                _DeferredOutcomeLog(
                    signal=signal,
                    registration=registration,
                    result=result,
                    audit=audit,
                )
            )
            return
        reservation = self._reserve_durable_admission()
        try:
            task = asyncio.create_task(
                self._log_safe(signal, registration, result, audit=audit),
                name=f"durable_signal_log:{signal.source}:{signal.id}",
            )
        except BaseException:
            self._release_durable_admission(reservation)
            raise

        self._outcome_log_tasks.add(task)
        self._outcome_log_tasks_by_signal.setdefault(signal.id, set()).add(task)

        def _complete(task: asyncio.Task) -> None:
            self._outcome_log_tasks.discard(task)
            signal_tasks = self._outcome_log_tasks_by_signal.get(signal.id)
            if signal_tasks is not None:
                signal_tasks.discard(task)
                if not signal_tasks:
                    self._outcome_log_tasks_by_signal.pop(signal.id, None)
            self._release_durable_admission(reservation)
            if task.cancelled():
                return
            try:
                task.result()
            except Exception:
                # _log_safe reports storage/UI errors itself.  This is a
                # defensive harvest for an unexpected task-body failure.
                logger.error(
                    "Signal outcome-log task failed for %s",
                    signal.id,
                    exc_info=True,
                )

        task.add_done_callback(_complete)

    async def _log_safe(
        self,
        signal: Signal,
        registration: SourceRegistration,
        result: SignalResult,
        *,
        audit: Optional[_ConstitutionAudit] = None,
    ) -> None:
        # A direct cancellation can land while SQLite has queued the append
        # but before its worker returns.  Keep the real writer in a child task
        # and shield it until it settles; otherwise the outer task's done
        # callback could release the lifecycle reservation and let shutdown
        # close the backend ahead of that queued write.
        writer = asyncio.create_task(
            self._write_outcome_log(signal, registration, result, audit=audit),
            name=f"durable_signal_log_writer:{signal.source}:{signal.id}",
        )
        cancelled = False
        while not writer.done():
            try:
                await asyncio.shield(writer)
            except asyncio.CancelledError:
                # Preserve cancellation for the caller after the accepted
                # write has reached a terminal outcome.  Repeated
                # cancellation cannot create a post-close SQLite operation.
                cancelled = True
                continue
            except Exception:
                # ``asyncio.shield`` re-raises the writer's own failure here.
                # Without this branch it escaped ``_log_safe`` entirely, so the
                # handler below never ran: from #2713 (2026-07-24) until now a
                # dropped signal_log row produced NO log line at all, because
                # ``_track_background_task`` only discards and shutdown's
                # ``gather(return_exceptions=True)`` swallows the rest.
                #
                # The failure is reported once, from ``writer.result()`` below,
                # which is also where the loss is counted.  Break rather than
                # handle it here so there is exactly one reporting path.
                break

        try:
            writer.result()
        except Exception as exc:
            # A dropped audit row is a loss, and a loss that leaves no record
            # anyone reads is indistinguishable from no loss at all (#2660):
            # 3,323 of these accumulated in a log file over two months before
            # anyone noticed. The ERROR line stays for the traceback; the
            # counter is what makes the loss observable at /health/detailed.
            #
            # Deliberately in-memory: the write that just failed was the
            # durable one, so recording the failure durably would need the
            # backend that is unavailable. This is an observability signal,
            # not a second audit trail.
            self._record_log_write_failure(signal.id, exc)
            logger.exception("Failed to write signal_log entry for %s", signal.id)
        if cancelled:
            raise asyncio.CancelledError()

    def _record_log_write_failure(
        self, signal_id: str, error: BaseException
    ) -> None:
        """Count one dropped signal_log row and remember its most recent cause."""
        self._log_write_failures += 1
        self._last_log_write_failure = SignalLogWriteFailure(
            signal_id=signal_id,
            error=f"{type(error).__name__}: {error}",
            failed_at=self._clock(),
        )

    @property
    def log_write_failure_count(self) -> int:
        """Number of signal_log rows dropped since this dispatcher was built.

        Monotonic by design.  It does not reset when a later write succeeds:
        the rows lost earlier are still lost, and an operator acknowledging
        that is the point of surfacing it.
        """
        return self._log_write_failures

    @property
    def lock_manager(self) -> OrderedLockManager:
        """The shared resource-lock manager, for ownership and health checks."""
        return self._locks

    @property
    def last_log_write_failure(self) -> Optional["SignalLogWriteFailure"]:
        """The most recent dropped-row cause, or None if none have dropped."""
        return self._last_log_write_failure

    @property
    def retained_durable_cognition_task_count(self) -> int:
        """Number of cancellation-resistant cognition tasks still quarantined."""

        return len(self._retained_durable_cognition_tasks)

    @property
    def retained_durable_cognition_cleanup_task_count(self) -> int:
        """Number of owned exact-lease cleanup tasks still in progress."""

        return len(self._retained_durable_cognition_cleanup_tasks)

    @property
    def durable_shutdown_owner_fenced(self) -> bool:
        """Whether shutdown returned while live cognition still owns this lease."""

        return self._durable_shutdown_owner_fenced

    async def _write_outcome_log(
        self,
        signal: Signal,
        registration: SourceRegistration,
        result: SignalResult,
        *,
        audit: Optional[_ConstitutionAudit] = None,
    ) -> None:
        """Persist one outcome and emit its UI side channel after commit."""
        with suppress_write_audit():
            result_summary = await self._store.append(
                signal,
                registration,
                result,
                prompt_template_hash=getattr(signal, _PROMPT_TEMPLATE_HASH_ATTR, None),
                **_audit_to_log_kwargs(audit),
            )

        # Phase 7 of #889: emit a UI-side-channel SSE event for non-
        # INTERNAL signals. Three rendering tiers in the design:
        #   - INTERNAL              → log only, no UI emit
        #   - USER_VISIBLE          → side channel OR inline (per session_id)
        #   - ADMIN_VISIBLE         → side channel for admin tools
        # Existing sources (heartbeat, cron, a2a.task_complete, stripe)
        # default to INTERNAL — none of them surprise-emit to the UI.
        # Sources opt in by constructing signals with an explicit
        # visibility argument.
        #
        # Every branch below records what it observed (#2922). A silent return
        # would leave the wait reconciler to infer visibility from the dispatch
        # status, which is exactly the "persisted therefore seen" conflation
        # this accounting exists to break.
        if signal.visibility == Visibility.INTERNAL:
            self._record_surface(signal.id, SURFACE_NOT_APPLICABLE)
            return
        emit = getattr(self._agent, "emit_event", None)
        if emit is None:
            self._record_surface(signal.id, SURFACE_NO_EMITTER)
            return
        payload = _build_ui_event_payload(signal, result, result_summary)
        try:
            receipt = await emit("signal_completed", payload)
        except Exception:
            self._record_surface(signal.id, SURFACE_EMIT_FAILED)
            logger.exception(
                "Failed to emit signal_completed UI event for %s", signal.id
            )
            return
        self._record_surface(
            signal.id, classify_event_receipt(receipt), receipt=receipt
        )

    def _record_surface(
        self, signal_id: str, status: str, *, receipt: Any = None
    ) -> None:
        """Record one signal's UI-emit verdict in the bounded surface ledger.

        Counter coercion is defensive because ``receipt`` comes from the agent
        protocol: bookkeeping about a delivery must never be the thing that
        fails the outcome writer that already persisted the audit row.
        """
        record = SignalSurfaceRecord(
            signal_id=signal_id,
            status=status,
            listeners=_receipt_count(receipt, "listeners"),
            accepted=_receipt_count(receipt, "accepted"),
            rejected=_receipt_count(receipt, "rejected"),
        )
        records = self._surface_records
        records.pop(signal_id, None)
        records[signal_id] = record
        while len(records) > self._MAX_SURFACE_RECORDS:
            records.popitem(last=False)

    def surface_record(self, signal_id: str) -> Optional["SignalSurfaceRecord"]:
        """What the UI side-channel emit did for ``signal_id``, if observed.

        ``None`` means the dispatcher has no record — the signal never reached
        the outcome writer, this process did not dispatch it, or the entry aged
        out of the bounded ledger. Callers must treat ``None`` as *visibility
        unknown* and never as either success or failure (#2922).
        """
        return self._surface_records.get(signal_id)


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
