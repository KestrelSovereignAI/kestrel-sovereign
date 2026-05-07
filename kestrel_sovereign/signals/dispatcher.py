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
import logging
import time
from collections import deque
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Awaitable, Callable, Coroutine, Optional, Protocol
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
from kestrel_sovereign.signals.lock_manager import OrderedLockManager
from kestrel_sovereign.signals.registry import RegistrationError, SourceRegistry
from kestrel_sovereign.signals.store import SignalLogStore

logger = logging.getLogger(__name__)


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
        ttl: int = DEFAULT_TTL,
        coalescing_window_default: timedelta = DEFAULT_COALESCING_WINDOW,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._agent = agent
        self._registry = registry
        self._locks = lock_manager
        self._store = store
        self._ttl = ttl
        self._default_window = coalescing_window_default
        self._clock = clock
        self._coalescing = _CoalescingState()
        self._rate = _RateLimitState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch_signal(self, signal: Signal) -> SignalResult:
        """Awaits the full lifecycle. Used by callers that need the result
        (scheduler, heartbeat). Always returns a `SignalResult` — failures
        are encoded as `Status.FAILED` with `error` set, never raised."""
        start = time.monotonic()
        try:
            return await self._run(signal, start)
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

    async def enqueue_signal(self, signal: Signal) -> SignalHandle:
        """Returns immediately with a tracked handle. The dispatch runs as
        an agent-owned background task; exceptions are logged not swallowed,
        and the task is cancellable via the agent's shutdown path."""
        coro = self.dispatch_signal(signal)
        task = self._agent._track_background_task(
            coro, name=f"signal_dispatch:{signal.source}:{signal.id}"
        )
        return SignalHandle(signal_id=signal.id, task=task)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def _run(self, signal: Signal, start: float) -> SignalResult:
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

        # Inherit trust from registration so callers can't lie.
        signal.origin_trust = registration.trust

        # UNTRUSTED → run sanitizer for non-ACTION modes
        if (
            registration.trust.value == "untrusted"
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
                    assert registration.prompt_template is not None
                    prompt = self._render_prompt(signal, registration)
                    # Set the in-flight turn's causation chain (already
                    # extended with this hop's frame in step 2 of the
                    # pipeline) so outbound A2A tasks created during
                    # the turn carry the lineage forward (#905 review
                    # P1). The chain lives in a ContextVar so
                    # concurrent COGNITION dispatches stay isolated —
                    # agent-level mutable state would race here
                    # (#906 review P1: dispatcher set/clear runs
                    # OUTSIDE the CONVERSATION lock that process_input
                    # acquires inside its body).
                    set_chain = getattr(self._agent, "_set_current_chain", None)
                    clear_chain = getattr(self._agent, "_clear_current_chain", None)
                    token = None
                    if set_chain is not None:
                        token = set_chain(signal.causation_chain)
                    try:
                        result = await self._agent.process_input(prompt)
                    finally:
                        if clear_chain is not None:
                            clear_chain(token)
                    return self._success(
                        signal,
                        start,
                        registration,
                        artifact=None,
                        action_result=None,
                        cognition_result=result,
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

    def _render_prompt(
        self, signal: Signal, registration: SourceRegistration
    ) -> str:
        # Minimal template render for v1: read the file and substitute
        # known placeholders. Fenced UNTRUSTED payload is the source's
        # responsibility via the prompt template content (the design says
        # "templates live under prompts/signals/" with explicit fences).
        # A richer template engine (jinja, etc.) is a follow-up.
        template_path = registration.prompt_template
        assert template_path is not None
        template = template_path.read_text(encoding="utf-8")
        return template.format(
            source=signal.source,
            kind=signal.kind,
            target_agent=signal.target_agent,
            payload=signal.payload,
            urgency=signal.urgency.value,
            arrived_at=signal.arrived_at.isoformat(),
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
            self._log_safe(signal, registration, result),
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
            self._log_safe(signal, registration, result),
            name=f"signal_log:{signal.source}:{signal.id}",
        )
        return result

    async def _log_safe(
        self,
        signal: Signal,
        registration: SourceRegistration,
        result: SignalResult,
    ) -> None:
        try:
            result_summary = await self._store.append(
                signal, registration, result
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
