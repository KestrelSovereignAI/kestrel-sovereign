"""Signal Dispatcher data model — public contract.

Lives in `kestrel_sdk` so feature packages can construct `Signal` envelopes
and read `SignalResult` objects without taking a hard dependency on
`kestrel_sovereign`. The runtime (dispatcher, registry, lock manager,
signal_log store) stays in sovereign.

Frozen `CausationFrame` is intentional — chains are immutable audit data
that propagate across A2A hops. `Signal` itself is mutable so the dispatcher
can append the new frame in place during the append-and-cycle-check step
(see SIGNAL_DISPATCHER.md §6 in the sovereign repo).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalMode(str, Enum):
    ACTION = "action"
    ARTIFACT = "artifact"
    COGNITION = "cognition"


class Trust(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class Urgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

    def at_or_above(self, other: "Urgency") -> bool:
        order = {Urgency.LOW: 0, Urgency.NORMAL: 1, Urgency.HIGH: 2}
        return order[self] >= order[other]


class Visibility(str, Enum):
    INTERNAL = "internal"
    USER_VISIBLE = "user_visible"
    ADMIN_VISIBLE = "admin_visible"


class Status(str, Enum):
    OK = "ok"
    COALESCED = "coalesced"
    DROPPED_RATE_LIMIT = "dropped_rate_limit"
    DROPPED_QUIET_HOURS = "dropped_quiet_hours"
    DROPPED_CYCLE = "dropped_cycle"
    DROPPED_VALIDATION = "dropped_validation"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------


class ResourceLock(str, Enum):
    """Named locks for the single ordered lock manager.

    `CONVERSATION` is the highest-order acquisition system-wide and is owned
    SOLELY by the turn lifecycle (see SIGNAL_DISPATCHER.md §Concern 1, 2).
    Sources MUST NOT declare it in their `resources` set — the registry
    validator enforces this.
    """

    CONVERSATION = "conversation"
    MEMORY = "memory"
    WALLET = "wallet"
    SCHEDULER = "scheduler"
    A2A = "a2a"
    SIGNAL_LOG = "signal_log"


# ---------------------------------------------------------------------------
# Causation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausationFrame:
    """One hop in a Signal's causation chain. Immutable audit data."""

    agent_id: str
    source: str
    signal_id: str
    turn_id: Optional[str]
    depth: int
    emitted_at: datetime


# ---------------------------------------------------------------------------
# Throttling / attention policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimit:
    """Throttling. `burst` is the immediate ceiling; per_minute/per_hour are
    rolling windows. None means unlimited on that axis (don't use that for
    UNTRUSTED sources)."""

    per_minute: Optional[int] = None
    per_hour: Optional[int] = None
    burst: Optional[int] = None


@dataclass(frozen=True)
class AttentionPolicy:
    """When the bird is reachable for ARTIFACT/COGNITION from this source.

    `quiet_hours` is a (start, end) tuple in `tz`. If `end < start`, the
    window wraps midnight. ACTION is never gated by quiet hours (only modes
    in `modes_governed` are).
    """

    quiet_hours: Optional[tuple[time, time]] = None
    tz: str = "UTC"
    modes_governed: frozenset[SignalMode] = frozenset(
        {SignalMode.COGNITION, SignalMode.ARTIFACT}
    )
    urgency_override: Urgency = Urgency.HIGH


# ---------------------------------------------------------------------------
# Privacy / logging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedactionPolicy:
    """Required for every source registration; no defaults.

    `summarize` runs against the validated payload and returns a redacted
    string suitable for `signal_log.payload_redacted`. Raw UNTRUSTED payloads
    are NEVER stored — only digest + summary.

    `store_raw_trusted` is opt-in and only honored when source.trust is
    TRUSTED. Defaults to False for safety.
    """

    summarize: Callable[[dict], str]
    store_raw_trusted: bool = False
    redact_caller_identifier: bool = True


# ---------------------------------------------------------------------------
# Source registration
# ---------------------------------------------------------------------------


# Forward references — handlers receive a Signal and return mode-specific
# results. ARTIFACT result is opaque (whatever the feature workflow returns).
ActionHandler = Callable[[dict], Awaitable[Any]]
ArtifactHandler = Callable[["Signal"], Awaitable[Any]]
PayloadSchema = Callable[[dict], dict]
"""Schema validator. Receives the raw (or post-sanitizer) payload, returns
the validated/normalized form, raises on invalid. Sources that use pydantic
should wrap: `lambda p: MyModel(**p).model_dump()`. Sources with no schema
needs can pass `dict` (no-op copy) — but writing an explicit validator is
preferred since it's the source's contract with downstream handlers."""


@dataclass
class SourceRegistration:
    """v1 boundary — no default-deny escape hatch. Every field documented
    in SIGNAL_DISPATCHER.md §"The Source Registry".
    """

    # Identity
    name: str
    schema: PayloadSchema

    # Mode
    default_mode: SignalMode
    allowed_modes: frozenset[SignalMode]

    # Behavior contracts (mode-specific; exactly one required per allowed mode)
    handler: Optional[ActionHandler] = None  # required if ACTION in allowed_modes
    artifact_handler: Optional[ArtifactHandler] = None  # required if ARTIFACT
    prompt_template: Optional[Path] = None  # required if COGNITION

    # Trust & sanitization
    trust: Trust = Trust.TRUSTED
    sanitizer: Optional[Callable[[dict], dict]] = None

    # Throttling
    rate_limit: RateLimit = field(default_factory=RateLimit)
    coalescing_window: Optional[timedelta] = None  # None = global default (5s)

    # Attention
    attention_policy: AttentionPolicy = field(default_factory=AttentionPolicy)

    # Concurrency — CONVERSATION FORBIDDEN here (turn lifecycle owns it)
    resources: frozenset[ResourceLock] = frozenset()
    allow_self_loops: bool = False

    # Privacy & audit — both required (registry rejects None for log_redaction)
    log_redaction: Optional[RedactionPolicy] = None
    retention_days: int = 30


# ---------------------------------------------------------------------------
# Signal envelope
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """The thing that wakes the bird (or runs an action / produces an
    artifact). Built by the dispatcher's caller from registered-source data.

    `id` and `arrived_at` default to fresh values; everything else is
    explicit because guessing routing fields is the road back to the
    accretion mess this dispatcher fixes.
    """

    # Identity
    source: str
    kind: str
    mode: SignalMode
    payload: dict

    # Routing
    target_agent: str
    visibility: Visibility = Visibility.INTERNAL  # opt into user visibility
    session_id: Optional[str] = None  # None = system-initiated
    caller: Optional[str] = None  # opaque identity string; redacted by default

    # Behavior
    urgency: Urgency = Urgency.NORMAL
    dedupe_key: Optional[str] = None
    origin_trust: Trust = Trust.TRUSTED  # set by dispatcher from registration

    # Causation — appended-to during dispatch
    causation_chain: list[CausationFrame] = field(default_factory=list)

    # Audit (auto-filled)
    id: str = field(default_factory=lambda: f"sig_{uuid4().hex}")
    arrived_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SignalResult:
    """Returned by `dispatch_signal`. Sources that need outcomes (scheduler,
    heartbeat) read this; fire-and-forget callers ignore it.
    """

    signal_id: str
    status: Status
    mode: SignalMode
    duration_ms: int
    turn_id: Optional[str] = None
    artifact: Any = None
    action_result: Any = None
    error: Optional[str] = None


@dataclass
class SignalHandle:
    """Returned by `enqueue_signal`. Wraps the supervised background task
    so callers can await/cancel without touching the underlying asyncio
    primitives.
    """

    signal_id: str
    task: asyncio.Task

    async def wait(self) -> SignalResult:
        return await self.task

    def cancel(self) -> bool:
        return self.task.cancel()

    @property
    def done(self) -> bool:
        return self.task.done()
