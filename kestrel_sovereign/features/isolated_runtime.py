"""Isolated feature runtime proxy and per-agent venv provisioning.

A feature distribution opts into out-of-venv execution via its pyproject:

    [tool.kestrel.feature]
    runtime = "isolated-venv"
    service = "kestrel_whatsapp.service:main" # console name or module:callable
    project = "service"                # install target for the venv (path/dist); defaults to the distribution
    # venv  = "/abs/path/.venv"        # optional explicit venv-path override

`service` is the thing to RUN: either a bare portable console-script name
resolved from the per-agent venv's bin/, or a validated Python
``module:callable`` target launched through that venv's interpreter. `project`
is the thing to INSTALL. They are deliberately distinct so the runnable is
never mistaken for a pip target or a ``python -m`` module. Console artifacts
are verified at the exact path later launched; callables do not use a wrapper.
"""

import asyncio
import base64
import errno
import hashlib
import hmac
import inspect
import json
import keyword
import logging
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
import weakref
from concurrent.futures import Executor, Future
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from queue import Empty, Full, Queue
from types import MappingProxyType, TracebackType
from typing import Any, Awaitable, Callable, Dict, List, Mapping, NoReturn, Optional
from uuid import uuid4

import psutil
from kestrel_sdk.channels import ChannelAdapter
from kestrel_sdk.isolated_feature import (
    CONFIG_TRANSITION_APPLIED,
    MAX_HOST_INGRESS_PAYLOAD_BYTES,
    ConfigTransitionResult,
    HostIngressCapabilities,
    HostIngressError,
    HostIngressPayload,
    HostIngressUnknownNameError,
    HostIngressUnsupportedError,
    ProtocolError,
    validate_host_ingress_name,
    validate_host_ingress_payload,
)
from kestrel_sdk.tools.base import AgentTool, ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.base import Feature, UIContributions
from kestrel_sovereign.features.channels.route_ownership import (
    ChannelRouteClaim,
    ChannelRouteOwnershipStore,
)

logger = logging.getLogger(__name__)
_CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class IsolatedRuntimeTelemetrySnapshot:
    """Sanitized lifecycle telemetry for one agent-bound isolated feature."""

    feature: str
    distribution: str
    state: str
    lifecycle_generation: int
    active_processes: int
    idle_processes: int
    restart_count: int
    idle_wake_count: int
    last_used_at: datetime | None
    cold_start_seconds: float | None
    warm_start_seconds: float | None
    rss_bytes: int | None
    cpu_seconds: float | None
    open_fds: int | None
    process_count: int | None
    environment_bytes: int | None = None
    private_writable_bytes: int | None = None
    downloaded_bytes: int | None = None
    provision_seconds: float | None = None
    cache_hit: bool | None = None
    cleanup_eligible: bool = False
    disk_telemetry_status: str | None = None


def configure_hosted_isolated_runtime_lifecycle(
    agent: Any,
    *,
    idle_timeout_seconds: float | None = None,
    idle_timeouts: Mapping[str, float | None] | None = None,
    telemetry_observer: Callable[[IsolatedRuntimeTelemetrySnapshot], Any] | None = None,
) -> None:
    """Install per-agent hosted lifecycle policy before feature discovery."""

    if _agent_runtime_scope(agent) is None:
        raise ValueError(
            "isolated runtime lifecycle policy requires an explicit hosted scope"
        )
    _agent_runtime_owner(agent)
    agent_attributes = getattr(agent, "__dict__", {})
    if (
        isinstance(agent_attributes, dict)
        and agent_attributes.get("_isolated_runtime_features_constructed") is True
    ):
        raise RuntimeError(
            "isolated runtime lifecycle policy must be configured before feature discovery"
        )

    def validate_timeout(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("isolated runtime idle timeout must be a number or None")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("isolated runtime idle timeout must be finite and positive")
        return float(value)

    idle_timeout_seconds = validate_timeout(idle_timeout_seconds)
    resolved_timeouts: dict[str, float | None] = {}
    if idle_timeouts is not None:
        if not isinstance(idle_timeouts, Mapping):
            raise TypeError("isolated runtime feature idle timeouts must be a mapping")
        for feature, timeout in idle_timeouts.items():
            if type(feature) is not str or _ISOLATED_FEATURE_CLASS_NAME.fullmatch(feature) is None:
                raise ValueError("isolated runtime feature idle timeout key is invalid")
            resolved_timeouts[feature] = validate_timeout(timeout)
    if telemetry_observer is not None and not callable(telemetry_observer):
        raise TypeError("isolated runtime telemetry observer must be callable")
    agent.isolated_runtime_idle_timeout_seconds = idle_timeout_seconds
    agent.isolated_runtime_idle_timeouts = MappingProxyType(resolved_timeouts)
    agent.isolated_runtime_telemetry_observer = telemetry_observer


def canonical_telegram_bot_id(value: object) -> str:
    """Normalize one Telegram bot ID for Core's route-ownership boundary.

    Telegram accepts decimal IDs, while route ownership must not let a host
    accidentally split one bot across textual aliases.  In particular,
    ``000123`` normalizes to ``123`` and provider-prefixed strings such as
    ``telegram-bot:123`` are rejected rather than treated as already
    canonical.
    """

    if type(value) is not str or not value.isascii() or not value.isdecimal():
        raise ValueError("Telegram bot ID must be a positive decimal string")
    normalized = value.lstrip("0")
    if not normalized:
        raise ValueError("Telegram bot ID must be positive")
    return normalized


@dataclass(frozen=True, slots=True)
class HostedTelegramRouteAttestation:
    """Host-supplied route evidence consumed before a Telegram child starts.

    The host (for example Frinz) remains responsible for concrete provider and
    HTTP provisioning.  Core only receives this typed, provider-neutral
    ownership input and establishes its own durable ledger fence before the
    child handshake can start polling or receive the hosted-ingress capability.
    """

    ownership_store: ChannelRouteOwnershipStore
    bot_id: str


def set_hosted_telegram_route_attestation_resolver(
    agent: Any, resolver: Callable[["ProxyFeature"], Any]
) -> None:
    """Inject a host's pre-initialize Telegram route resolver.

    This is Core's generic boot seam: a host installs it while constructing an
    agent, before feature discovery invokes :meth:`ProxyFeature.initialize`.
    The resolver supplies only typed route evidence; Core performs the durable
    ownership claim itself. Concrete webhook and provider provisioning remain
    outside Core.
    """

    if not callable(resolver):
        raise TypeError("hosted Telegram route attestation resolver must be callable")
    setattr(agent, "hosted_telegram_route_attestation_resolver", resolver)

# Upper bound on a single supervision health probe. A wedged child that never
# answers health() must not silently kill supervision forever (F013) — treat a
# probe that exceeds this as unhealthy and fall through to the restart path.
_HEALTH_PROBE_TIMEOUT = 5.0
# Host telemetry is advisory. An async observer that wedges must never acquire
# ownership of child startup, retirement, reload, or shutdown progress.
_TELEMETRY_OBSERVER_TIMEOUT = 1.0
_TELEMETRY_EMIT_MIN_INTERVAL = 5.0
_DISK_TELEMETRY_ENTRY_BUDGET = 250_000
_DISK_TELEMETRY_TIME_BUDGET_SECONDS = 1.0
_DISK_TELEMETRY_DEPTH_BUDGET = 64


class _BoundedDaemonExecutor(Executor):
    """Small daemon pool reserved for advisory host callbacks.

    Standard ``ThreadPoolExecutor`` workers are non-daemon and can let a
    hostile synchronous observer acquire interpreter-exit ownership. These
    workers deliberately cannot do that, and the bounded queue prevents an
    unbounded retained-snapshot backlog when every observer slot is wedged.
    """

    _STOP = object()

    def __init__(self, *, max_workers: int, queue_capacity: int) -> None:
        self._max_workers = max_workers
        self._work: Queue[object] = Queue(maxsize=queue_capacity)
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def _start_workers(self) -> None:
        with self._lock:
            if self._threads or self._shutdown:
                return
            for index in range(self._max_workers):
                thread = threading.Thread(
                    target=self._worker,
                    name=f"kestrel-telemetry-observer-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    def _worker(self) -> None:
        while True:
            item = self._work.get()
            try:
                if item is self._STOP:
                    return
                future, fn, args, kwargs = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:  # noqa: BLE001 - Future owns outcome
                    future.set_exception(exc)
            finally:
                self._work.task_done()

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        self._start_workers()
        with self._lock:
            if self._shutdown:
                future.set_exception(RuntimeError("telemetry observer executor stopped"))
                return future
            try:
                self._work.put_nowait((future, fn, args, kwargs))
            except Full:
                future.set_exception(RuntimeError("telemetry observer executor saturated"))
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Reject new callbacks and abandon queued advisory work.

        ``wait`` is intentionally ignored: a hostile observer must not regain
        process-lifecycle ownership through explicit executor shutdown. Running
        callbacks remain isolated on daemon threads and queued callbacks are
        always cancelled so sentinel delivery cannot block on a full queue.
        """

        queued_futures: list[Future[Any]] = []
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            threads = tuple(self._threads)
            while True:
                try:
                    item = self._work.get_nowait()
                except Empty:
                    break
                try:
                    if item is not self._STOP:
                        future, _fn, _args, _kwargs = item
                        queued_futures.append(future)
                finally:
                    self._work.task_done()
            for _thread in threads:
                try:
                    self._work.put_nowait(self._STOP)
                except Full:
                    # A bounded queue may be smaller than the worker count.
                    # Unsignalled workers are daemon-only and will remain
                    # harmlessly parked after the signalled workers exit.
                    break
        for future in queued_futures:
            future.cancel()


_TELEMETRY_OBSERVER_EXECUTOR = _BoundedDaemonExecutor(
    max_workers=4,
    queue_capacity=8,
)


def _measure_directory_tree_bytes(
    path: Path, *, deadline: float | None = None
) -> tuple[int | None, str]:
    """Measure one no-follow tree and preserve why measurement degraded."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if (
        nofollow is None
        or directory is None
        or os.open not in os.supports_dir_fd
        or os.scandir not in os.supports_fd
    ):
        return None, "unavailable"
    flags = os.O_RDONLY | nofollow | directory
    stack: list[tuple[int, Any]] = []
    total = 0
    entries_seen = 0
    if deadline is None:
        deadline = time.monotonic() + _DISK_TELEMETRY_TIME_BUDGET_SECONDS
    try:
        root_fd = os.open(path, flags)
        try:
            root_entries = os.scandir(root_fd)
        except BaseException:
            os.close(root_fd)
            raise
        stack.append((root_fd, root_entries))
        while stack:
            current_fd, entries = stack[-1]
            try:
                entry = next(entries)
            except StopIteration:
                entries.close()
                os.close(current_fd)
                stack.pop()
                continue
            entries_seen += 1
            if (
                entries_seen > _DISK_TELEMETRY_ENTRY_BUDGET
                or time.monotonic() >= deadline
            ):
                return None, "budget-exceeded"
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                if len(stack) >= _DISK_TELEMETRY_DEPTH_BUDGET:
                    return None, "budget-exceeded"
                child_fd = os.open(entry.name, flags, dir_fd=current_fd)
                try:
                    child_entries = os.scandir(child_fd)
                except BaseException:
                    os.close(child_fd)
                    raise
                stack.append((child_fd, child_entries))
            elif stat.S_ISREG(metadata.st_mode):
                total += int(metadata.st_size)
        return total, "complete"
    except OSError:
        return None, "unavailable"
    finally:
        for descriptor, entries in reversed(stack):
            entries.close()
            try:
                os.close(descriptor)
            except OSError:
                pass


# ``ToolExecutionTrigger`` validates identifiers by UTF-8 byte length, not
# Python character count.  Schedule IDs normally fit unchanged, but legacy
# imports can contain arbitrarily long IDs while the SDK context permits at
# most 512 bytes.  Keep the full ID in SchedulerExecution/storage and expose a
# deterministic, plainly tagged provenance surrogate only at that bounded wire
# boundary.
_SDK_CONTEXT_IDENTIFIER_MAX_BYTES = 512
_SCHEDULE_TRIGGER_SOURCE_HASH_PREFIX = "schedule-sha256:"

# Transition metadata lives beside the existing ``config`` property on the
# feature-config graph node.  The keys are deliberately implementation-private:
# callers continue to read the long-standing ``config`` dictionary, while the
# proxy uses the opaque generation token to prove that a later promotion is its
# own write rather than a same-shaped write from another hosted replica.
_CONFIG_GENERATION_KEY = "_isolated_config_generation"
_PENDING_GENERATION_KEY = "_isolated_pending_generation"
_PENDING_OWNER_KEY = "_isolated_pending_owner"
_PENDING_LEASE_EXPIRES_AT_KEY = "_isolated_pending_lease_expires_at"

# Optional, capability-negotiated lifecycle callbacks for isolated services
# that acknowledge external input independently of Core's stdio event path.
# A service only advertises these names when it can stop its producer and reap
# any in-flight callback before acknowledging quiescence. The opaque transition
# id lets a later resume prove it belongs to this exact config attempt.
_EXTERNAL_INGRESS_QUIESCE = "external-ingress-quiesce"
_EXTERNAL_INGRESS_RESUME = "external-ingress-resume"
_EXTERNAL_INGRESS_TRANSITION_TOKEN_BYTES = 32

# A service-to-host event is a JSON-RPC notification and therefore has no
# response channel. An opted-in producer may wrap a channel inbound payload in
# this private descriptor; after the host has completed the inbound handler it
# calls the advertised host-ingress name with the exact detached payload. The
# callback must run outside the SDK event reader because that same reader owns
# the response stream for host-ingress RPCs.
_EVENT_HOST_INGRESS_ACK_FIELD = "_host_ingress_ack"
_EVENT_HOST_INGRESS_RETRY_FIELD = "_host_ingress_retry"
_EVENT_HOST_INGRESS_MESSAGE_FIELD = "message"
_EVENT_TELEGRAM_TERMINAL_DISPOSITION_FIELD = "_telegram_terminal_disposition"
_TELEGRAM_TERMINAL_DISPOSITIONS = frozenset(
    {
        "malformed_update",
        "unsupported_update",
        "senderless_update",
        "unauthorized_sender",
    }
)
_EVENT_INGRESS_ATTEMPT_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_TELEGRAM_POLLING_ACK = "telegram-polling-ack"
_TELEGRAM_POLLING_NACK = "telegram-polling-nack"
# Polling is sequential: an acknowledged source cannot emit its next update
# until Core acknowledges the current one. Retaining one current-child event
# during a finite gate close therefore preserves no-loss startup without
# turning a malicious notification flood into unbounded host memory.
_MAX_DEFERRED_ACKNOWLEDGED_EVENTS = 1
# An ACK is an idempotent private RPC, but it must not become an unbounded task
# flood or retain a Telegram poller forever after one lost response.
_EVENT_INGRESS_ACK_ATTEMPTS = 3
_EVENT_INGRESS_ACK_TIMEOUT = 3.0
_EVENT_INGRESS_ACK_CANCELLATION_GRACE = 1.0
_EVENT_INGRESS_ACK_BACKOFF = 0.1
# This is an initialize-handshake capability, not persisted feature config.
# SDK 0.35.1 forwards the host-controlled config object but has no reverse
# capability field on its subprocess wrapper yet.  Keeping this identifier in
# one Core-owned place gives services an explicit opt-in contract without
# guessing Core versions from an arbitrary client name.
_HOST_RUNTIME_CAPABILITIES_FIELD = "_kestrel_host_runtime_capabilities"
_ACKNOWLEDGED_CHANNEL_INBOUND_CAPABILITY = "channel-inbound-acknowledgement-v1"
# A host-only startup fence for an externally provisioned Telegram webhook.
# It is deliberately injected at process launch rather than stored in feature
# config: a user config row must never be able to claim a host route is live.
_HOSTED_TELEGRAM_INGRESS_OWNER_CAPABILITY = "telegram-hosted-ingress-owner-v1"
# A non-cursor producer has no child-side durable cursor to retain a callback
# while Core's gate is closed.  Keep one item in the queue while its serial
# worker owns one other item; the SDK notification reader naturally applies
# backpressure to the producer before a third item can allocate host memory.
_MAX_PENDING_NON_CURSOR_INGRESS_EVENTS = 1
_event_source_client: ContextVar[Any | None] = ContextVar(
    "isolated_event_source_client", default=None
)
# This value is set only around a host-validated Telegram polling callback.
# It is task-local rather than part of the child's JSON payload so an isolated
# service cannot promote an arbitrary notification into the cursor-owning
# protocol by adding a field to its message.
_cursor_owned_inbound_protocol: ContextVar[bool] = ContextVar(
    "isolated_cursor_owned_inbound_protocol", default=False
)
_telegram_terminal_inbound_disposition: ContextVar[str | None] = ContextVar(
    "isolated_telegram_terminal_inbound_disposition", default=None
)

# A staged config must survive a short process pause, but it must not turn an
# interrupted deploy or process death into a permanent write lock.  Readers
# wait an additional skew allowance before takeover: a replica whose clock is
# ahead cannot steal a healthy writer's lease merely because its wall clock is
# fast.  The durable timestamp is always UTC and is parsed fail-closed.
_PENDING_CONFIG_LEASE_TTL = timedelta(minutes=2)
_PENDING_CONFIG_CLOCK_SKEW = timedelta(seconds=30)
_PENDING_CLEANUP_WRITE_ATTEMPTS = 2
_TERMINAL_TRAFFIC_ERROR = "isolated feature traffic is unavailable"

# The locked SDK 0.36.0 has no public stop-budget export.  Its private
# ``SupervisedIsolatedFeatureClient.stop()`` path observes a single facade's
# lifecycle serially: two startup-settlement observations, eight child
# retirement observations (spawn/startup, graceful shutdown, natural exit,
# TERM, KILL, close, and the post-close process observation), and one retired
# operation observation.  Every one uses its three-second phase limit.  Keep
# this derivation named and adjacent to the SDK pin so an SDK lifecycle change
# has one auditable host deadline to update rather than a misleading literal.
#
# A facade with more than one unresolved historical retirement is still fenced
# after this one-child complete sequence.  That preserves whole-agent fair
# share and leaves the exact facade owned for a later bounded retry instead of
# making host cleanup unbounded.
_SDK_STOP_PHASE_TIMEOUT = 3.0
_SDK_STOP_STARTUP_SETTLEMENT_OBSERVATIONS = 2
_SDK_STOP_RETIREMENT_OBSERVATIONS = 8
_SDK_STOP_RETIRED_OPERATION_OBSERVATIONS = 1
_SDK_SUBPROCESS_STOP_BUDGET = _SDK_STOP_PHASE_TIMEOUT * (
    _SDK_STOP_STARTUP_SETTLEMENT_OBSERVATIONS
    + _SDK_STOP_RETIREMENT_OBSERVATIONS
    + _SDK_STOP_RETIRED_OPERATION_OBSERVATIONS
)
# Account for host task scheduling and the SDK's lock handoff independently of
# the audited subprocess observations above.  Cancellation acknowledgement is
# deliberately separate: a timed-out operation must first be asked to stop and
# have its task outcome consumed before it is marked uncertain.  Tests shorten
# both constants to exercise the fence without waiting for the production stop
# budget.
_FACADE_LIFECYCLE_SCHEDULING_ALLOWANCE = 3.0
_FACADE_LIFECYCLE_OPERATION_TIMEOUT = (
    _SDK_SUBPROCESS_STOP_BUDGET + _FACADE_LIFECYCLE_SCHEDULING_ALLOWANCE
)
_FACADE_LIFECYCLE_CANCELLATION_GRACE = 1.0
# A health call is external facade work too.  Give a cancelled probe a short
# acknowledgement window before retaining it as terminally incomplete work.
# This intentionally shares the lifecycle-operation grace: both awaitables can
# retain an SDK facade and must never be detached when they suppress cancel.
_HEALTH_PROBE_CANCELLATION_GRACE = _FACADE_LIFECYCLE_CANCELLATION_GRACE
# A stop result proves only that the facade has completed its process-retirement
# contract.  It cannot prove that a legacy facade also completed every RPC it
# admitted before reporting success.  Keep that second, host-side boundary
# independently bounded; a timeout leaves the gate sealed and its exact drain
# task owned for eventual observation.
_TERMINAL_TRAFFIC_DRAIN_TIMEOUT = 3.0

# KestrelAgent's fair-share ``wait_for`` owns the caller deadline.  Direct
# feature callers retain the historical interruption contract: cancellation
# waits through the terminal boundary.  The agent invokes the explicit wrapper
# below so the proxy can hand its already-owned cleanup task back at that
# deadline without cancelling a SDK coroutine which may hold a process handle.
_AGENT_SHUTDOWN_DEADLINE_ACTIVE: ContextVar[bool] = ContextVar(
    "isolated_agent_shutdown_deadline_active", default=False
)

# Isolated feature config used to share the in-process feature key
# ``feature_config:<class-name>``. Graph-node IDs are globally unique even
# though each AsyncGraphStore is tenant-bound, so fresh agents need a DID
# scoped key. A visible pre-scoping key is *adopted in place* by a new proxy:
# copying it would leave old replicas writing one CAS authority while new
# replicas write another.
_SCOPED_CONFIG_NODE_PREFIX = "feature_config:v2"
_DID_IDENTITY_RE = re.compile(r"^did:[a-z0-9]+:[^\s\x00]+$")


class _ConfigTransitionLeaseLost(RuntimeError):
    """The durable lease for an in-flight lifecycle transition was not renewed."""


class _ConfigAuthorityChanged(RuntimeError):
    """A visible legacy config row superseded scoped rolling-upgrade authority."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # A stage can commit before its post-CAS authority fence reports the
        # legacy row.  The owning lifecycle method has not received the local
        # transition return value yet, so preserve it on the signal.
        self.transition: Optional["_ConfigTransition"] = None


class _TrafficGateTerminalError(RuntimeError):
    """The proxy has entered a terminal no-admission lifecycle state."""

    def __init__(self) -> None:
        super().__init__(_TERMINAL_TRAFFIC_ERROR)


class _TrafficGateClosedError(RuntimeError):
    """A non-waiting admission arrived during a finite transition."""


class _TerminalLifecyclePermitRevoked(RuntimeError):
    """A terminal transition superseded an initializer before it acquired ownership."""


class _FacadeLifecycleOperationTimedOut(RuntimeError):
    """An externally supplied lifecycle coroutine exceeded its ownership fence."""


class _TerminalTrafficDrainTimedOut(RuntimeError):
    """A sealed admitted call outlived terminal traffic-drain ownership."""


class _CrossLoopFacadeOperationError(_FacadeLifecycleOperationTimedOut):
    """A facade returned a Future owned by a different event loop."""


class _HostOwnedFacadeOperation:
    """The one canonical ownership record for a facade-returned operation.

    A Task/Future returned by a facade is already an operation, not merely an
    awaitable recipe.  The old adapter created an outer task which did not
    begin awaiting that operation until the next scheduling turn.  Cancelling
    the outer task in that gap orphaned the real operation.  This record claims
    the original Future synchronously and directs every cancellation, outcome,
    and late-retention decision to that original object.

    Coroutines and immediate values retain their compatibility path: the named
    host task *is* their operation.  A same-loop Future/Task additionally gets
    a named observer task for diagnostics, but the Future remains authoritative
    for ownership.  Cross-loop Futures cannot safely be awaited here; they are
    retained and cancelled/observed through their owning loop, while callers
    fail closed without starting a competing lifecycle operation.
    """

    def __init__(self, operation: Any, *, name: str) -> None:
        self._name = name
        self._host_loop = asyncio.get_running_loop()
        self._source: asyncio.Future[Any] | None = None
        self._source_loop: asyncio.AbstractEventLoop | None = None
        self._host_task: asyncio.Task[Any] | None = None
        self._foreign_loop = False
        self._done_callbacks: list[Callable[["_HostOwnedFacadeOperation"], None]] = []
        # A foreign Future may only be consumed by its owning loop.  Until that
        # loop has run our observer, its raw ``done()`` bit is not a safe
        # lifecycle settlement signal: a stopped (but open) loop otherwise
        # strands both the observer and a requested cancellation forever while
        # the host retries beside the original facade operation.
        self._foreign_settlement_acknowledged = False
        # Foreign outcomes are consumed only by their owner loop.  Retain the
        # resulting neutral classification, never the facade value, exception,
        # or traceback, so host lifecycle ownership can distinguish success.
        self._foreign_settlement_disposition: str | None = None
        self._settlement_notified = False
        # ``done()`` for a foreign source means its owner loop has safely
        # consumed the private result.  It does *not* mean the host lifecycle
        # callbacks which release/fence the exact client have run yet.  Keep
        # that final host-side delivery state explicit so a fresh shutdown
        # cannot discard lifecycle ownership in the gap.
        self._settlement_callbacks_pending = 0

        if isinstance(operation, asyncio.Future):
            self._source = operation
            self._source_loop = operation.get_loop()
            self._foreign_loop = self._source_loop is not self._host_loop
            if not self._foreign_loop:
                self._host_task = asyncio.create_task(
                    self._observe_source(operation), name=name
                )
                self._host_task.add_done_callback(_consume_late_lifecycle_task_outcome)
            self._install_source_settlement_observer()
            return

        async def await_operation() -> Any:
            return await _maybe_await(operation)

        self._host_task = asyncio.create_task(await_operation(), name=name)
        self._host_task.add_done_callback(self._notify_settled)

    async def _observe_source(self, source: asyncio.Future[Any]) -> Any:
        return await asyncio.shield(source)

    @property
    def foreign_loop(self) -> bool:
        return self._foreign_loop

    def get_name(self) -> str:
        """Expose the historic task-like diagnostic name to ownership users."""

        return self._name

    def __await__(self):
        """Keep task-like compatibility for internal late-operation owners."""

        return self.wait().__await__()

    def done(self) -> bool:
        if self._foreign_loop:
            return self._foreign_settlement_acknowledged
        target = self._source if self._source is not None else self._host_task
        return target is not None and target.done()

    def cancelled(self) -> bool:
        # Do not inspect a foreign Future's terminal outcome from the host
        # loop.  Foreign callers fail closed before they need this distinction.
        if self._foreign_loop:
            return False
        target = self._source if self._source is not None else self._host_task
        return target is not None and target.cancelled()

    @property
    def foreign_settlement_disposition(self) -> str | None:
        """Return the owner-loop's sanitized foreign terminal classification."""

        return self._foreign_settlement_disposition

    def result(self) -> Any:
        if self._foreign_loop:
            raise _CrossLoopFacadeOperationError(
                "isolated facade returned a Future owned by another event loop"
            )
        target = self._source if self._source is not None else self._host_task
        if target is None:
            raise RuntimeError("host-owned facade operation lost its completion target")
        return target.result()

    async def wait(self) -> Any:
        return await self.shield()

    def shield(self) -> asyncio.Future[Any]:
        """Return the host-loop shield used by bounded ownership waits."""

        if self._foreign_loop:
            raise _CrossLoopFacadeOperationError(
                "isolated facade returned a Future owned by another event loop"
            )
        target = self._source if self._source is not None else self._host_task
        if target is None:
            raise RuntimeError("host-owned facade operation lost its completion target")
        return asyncio.shield(target)

    def cancel(self) -> None:
        """Cancel the real facade operation, including before an observer runs."""

        target = self._source if self._source is not None else self._host_task
        if target is None or self.done():
            return
        if self._foreign_loop:
            self.retry_owner_loop_cancellation_and_observation()
            return
        target.cancel()

    def retry_owner_loop_cancellation_and_observation(self) -> bool:
        """Retry foreign cancellation and outcome observation on its owner loop.

        A lifecycle owner can retain this exact operation while its foreign
        loop is stopped.  Once that loop restarts, a later cleanup pass must
        be able to re-request cancellation and acknowledgement without ever
        calling the facade again.  Both owner-loop actions are idempotent:
        cancellation ignores an already-settled source and settlement
        acknowledgement is guarded by ``_foreign_settlement_acknowledged``.

        ``False`` means the loop remains stopped or closed (or this operation
        has already settled), so callers stay fail-closed and retain ownership.
        """

        if not self._foreign_loop or self.done():
            return False
        # Cancellation, like result consumption, belongs exclusively to the
        # source loop.  A stopped/closing loop cannot acknowledge this request,
        # so leave the durable owner fenced instead of queuing work there.
        return self._dispatch_to_foreign_loop(
            self._cancel_and_observe_source_in_owner_loop
        )

    def add_done_callback(
        self, callback: Callable[["_HostOwnedFacadeOperation"], None]
    ) -> None:
        """Run ``callback`` in the host loop after the authoritative settles."""

        if self._settlement_notified:
            # Notification has already drained the registration list.  Queue
            # this one callback through the same delivery path; do not retain
            # it in the list (or invoke it synchronously during reentrancy).
            self._schedule_settlement_callback(callback)
            return
        self._done_callbacks.append(callback)

    @property
    def settlement_delivery_complete(self) -> bool:
        """Whether host lifecycle callbacks have consumed settlement state."""

        return self._settlement_notified and self._settlement_callbacks_pending == 0

    def _schedule_settlement_callback(
        self, callback: Callable[["_HostOwnedFacadeOperation"], None]
    ) -> None:
        """Deliver a settlement callback and account for its host-side fence."""

        self._settlement_callbacks_pending += 1

        def deliver() -> None:
            try:
                callback(self)
            finally:
                self._settlement_callbacks_pending -= 1

        try:
            self._host_loop.call_soon(deliver)
        except RuntimeError:
            # A closing host loop cannot make another lifecycle attempt.  Do
            # not claim a callback was delivered when it was not.
            self._settlement_callbacks_pending -= 1

    def _dispatch_to_foreign_loop(self, callback: Callable[[], None]) -> bool:
        """Request owner-loop work only while that loop can acknowledge it.

        ``is_running`` avoids putting observer/cancel handles onto an already
        stopped-open loop.  The check and dispatch necessarily race loop
        shutdown; a ``RuntimeError`` or a handle stranded by that race leaves
        this record durably unsettled, which is the fail-closed ownership
        state consumed by lifecycle retention.
        """

        source_loop = self._source_loop
        if (
            source_loop is None
            or source_loop.is_closed()
            or not source_loop.is_running()
        ):
            return False
        try:
            source_loop.call_soon_threadsafe(callback)
        except RuntimeError:
            return False
        return True

    def _cancel_source_in_owner_loop(self) -> None:
        """Cancel the source only from the loop which owns it."""

        source = self._source
        if source is not None and not source.done():
            source.cancel()

    def _cancel_and_observe_source_in_owner_loop(self) -> None:
        """Retry a fenced cancellation and settlement acknowledgement together."""

        self._cancel_source_in_owner_loop()
        self._observe_foreign_source_in_owner_loop()

    def _observe_foreign_source_in_owner_loop(self) -> None:
        """Install or run foreign settlement observation in the source loop.

        ``Future.add_done_callback`` queues a later turn even for a completed
        Future.  An owner loop that stops immediately after registration would
        strand that callback, so inspect a settled source inline instead.
        """

        source = self._source
        if source is None:
            return
        if source.done():
            self._observe_foreign_source_settlement(source)
            return
        source.add_done_callback(self._observe_foreign_source_settlement)

    def _observe_foreign_source_settlement(self, source: asyncio.Future[Any]) -> None:
        """Consume one foreign outcome and acknowledge it idempotently."""

        if self._foreign_settlement_acknowledged:
            return
        try:
            source.result()
        except asyncio.CancelledError:
            disposition = "cancelled"
        except BaseException:  # noqa: BLE001 - foreign outcomes stay private
            disposition = "failed"
        else:
            disposition = "succeeded"
        self._foreign_settlement_disposition = disposition
        # Set acknowledgement only after retrieving the terminal outcome.
        # Callback registration/cancellation can race in the owner loop, so
        # this guard makes either order safe and prevents duplicate release.
        self._foreign_settlement_acknowledged = True
        # The source can retain a result, exception, and traceback supplied by
        # a tenant facade.  Once its owning loop has consumed that terminal
        # state, this ownership record needs only the neutral disposition and
        # the host delivery fence.  In particular, host-loop shutdown can
        # strand the callback below, so retaining ``source`` until delivery is
        # complete would keep private terminal state alive indefinitely.
        if self._source is source:
            self._source = None
        try:
            self._host_loop.call_soon_threadsafe(self._notify_settled)
        except RuntimeError:
            # Host shutdown cannot turn owner-loop acknowledgement into a
            # retry, but it must not undo this terminal acknowledgement.
            pass

    def _install_source_settlement_observer(self) -> None:
        source = self._source
        if source is None:
            return

        def observed(_: asyncio.Future[Any]) -> None:
            # A foreign Task must have its exception retrieved in its own loop
            # before this record drops the last strong reference to it.
            if self._foreign_loop:
                self._observe_foreign_source_settlement(source)
                return
            try:
                self._host_loop.call_soon_threadsafe(self._notify_settled)
            except RuntimeError:
                # Host shutdown cannot turn an unobserved foreign outcome into
                # a retry.  The acknowledgement above remains the only safe
                # terminal state, and retained owners may prune it later.
                pass

        if self._foreign_loop:
            # ``Future.result()`` is an owner-loop operation even after a
            # Future has settled: subclasses may enforce affinity or perform
            # bookkeeping there.  Retain every foreign source until its owner
            # runs this observer.  A stopped or closed loop therefore remains
            # deliberately fenced instead of exposing tenant data or treating
            # a pre-settled result as a completed lifecycle operation.
            self._dispatch_to_foreign_loop(self._observe_foreign_source_in_owner_loop)
        else:
            source.add_done_callback(observed)

    def _notify_settled(self, _: asyncio.Future[Any] | None = None) -> None:
        if self._settlement_notified:
            return
        self._settlement_notified = True
        callbacks, self._done_callbacks = self._done_callbacks, []
        for callback in callbacks:
            # Each callback gets its own loop handle, matching asyncio's
            # callback isolation: one consumer failure must not strand a
            # later lifecycle-release callback or retain this facade.
            self._schedule_settlement_callback(callback)


def _consume_late_lifecycle_task_outcome(
    operation: asyncio.Future[Any] | _HostOwnedFacadeOperation,
) -> None:
    """Consume a fenced operation's eventual result without retaining failure."""

    # Cross-loop results are normally retrieved by ``observed`` in their
    # source loop.  The only exception is an already-terminal source at
    # construction time, whose immutable outcome is consumed there so a
    # stopped owner loop cannot strand it.  Do not retrieve a later foreign
    # result here: that would retain secret-bearing errors on host traceback.
    if isinstance(operation, _HostOwnedFacadeOperation) and operation.foreign_loop:
        return
    try:
        operation.result()
    except BaseException:  # noqa: BLE001 - terminal facade outcomes stay private
        pass


def _facade_settlement_event(operation: _HostOwnedFacadeOperation) -> asyncio.Event:
    """Return a neutral host-loop signal for one facade operation settling.

    Awaiting a shielded facade Future can raise that Future's
    ``CancelledError`` in the lifecycle owner.  An event carries settlement as
    data instead, so a caught ``CancelledError`` while waiting is unambiguously
    delivery to the lifecycle owner itself.
    """

    settled = asyncio.Event()
    if operation.done():
        settled.set()
    else:
        operation.add_done_callback(lambda _completed: settled.set())
    return settled


def _create_host_owned_facade_task(
    operation: Any, *, name: str
) -> _HostOwnedFacadeOperation:
    """Synchronously establish canonical ownership of one facade operation."""

    return _HostOwnedFacadeOperation(operation, name=name)


async def _await_task_until_complete(
    task: asyncio.Task[Any],
    *,
    preserve_cancellation: bool,
    settle_on_cancellation: bool = True,
) -> Any:
    """Wait for a shielded task without letting a later cancellation orphan it.

    Lifecycle cleanup, including the traffic-gate boundary, must settle before
    its owner releases ``_reload_lock``.  ``Task.result()`` after completion is
    intentional: it avoids one final cancellation point after the task has
    already changed durable/client/gate state.
    ``settle_on_cancellation=False`` is for a caller with an externally owned
    deadline.  The task must already be retained by a durable object field;
    the caller may then honour its deadline without cancelling a child that
    owns a subprocess or an active traffic boundary.
    """

    cancellation_args: tuple[Any, ...] | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            # A cancellation request is not a reason to abandon the child
            # before it records its own SDK fence or cleanup outcome.
            if task.done() and task.cancelled():
                break
            if not settle_on_cancellation:
                # The shared task remains attached to its owner (for example
                # ``_terminal_cleanup_task``).  A caller with a smaller
                # whole-agent budget can therefore move on without cancelling
                # an SDK stop coroutine that may own the sole process handle.
                raise
            # Preserve the first cancellation only.  A caller can receive
            # further cancellation requests while this shielded cleanup
            # drains; letting a later request replace the original loses the
            # cancellation which actually interrupted the lifecycle owner and
            # retains an exception traceback longer than necessary.
            if cancellation_args is None:
                cancellation_args = tuple(exc.args)
            continue
    # A child cancellation is a result of the child operation, not an empty
    # successful result.  In particular, lifecycle callers must not mistake a
    # cancelled terminal cleanup for a completed one and release ownership
    # while its child or gate state remains uncertain.
    if task.cancelled():
        raise asyncio.CancelledError()
    result = task.result()
    if cancellation_args is not None and not preserve_cancellation:
        raise asyncio.CancelledError(*cancellation_args)
    return result


async def _await_owned_facade_lifecycle_operation(
    operation: Any,
    *,
    name: str,
    on_completed: Callable[[], None] | None = None,
    on_timeout: Callable[[], None] | None = None,
    on_late_task: Callable[[_HostOwnedFacadeOperation], None],
) -> Any:
    """Own a facade operation without allowing it to hold lifecycle forever.

    The SDK's subprocess ``stop()`` detaches its only process handle before it
    awaits graceful termination.  Cancelling that exact coroutine therefore
    cannot be repaired by a later ``stop()`` call: the later call sees no
    process to kill.  Own the operation in a task and observe caller
    cancellation outside facade code. A real SDK 0.36.0 stop receives its full
    documented graceful/terminate/kill budget; a hostile legacy facade which
    does not settle by then is cancelled, drained for a short acknowledgement
    window, and reported as an *uncertain* outcome through ``on_timeout``.

    A Python coroutine that actively suppresses every cancellation and keeps
    running cannot be force-killed by asyncio. In that pathological case the
    required ``on_late_task`` takes ownership *before* this helper can release
    its caller.  The owner keeps the exact client sealed and refuses a second
    ``stop()`` until the original coroutine has settled and its outcome is
    consumed.  Cooperative awaitables (including ``asyncio.Event.wait``)
    settle during the acknowledgement window and leave no task behind.
    """

    task = _create_host_owned_facade_task(operation, name=name)
    if task.foreign_loop:
        # The original Future is retained before this failure is visible. Its
        # owner records both the exact operation and the facade fence, so no
        # retry can race a cross-loop stop/start which is still running.  A
        # later owner-loop acknowledgement of success is a real completed
        # stop: release that exact ownership once, while failed/cancelled
        # outcomes remain fenced for a fresh bounded retirement attempt.
        foreign_fenced = False
        foreign_success_reported = False

        def apply_foreign_disposition(
            completed: _HostOwnedFacadeOperation,
        ) -> None:
            nonlocal foreign_fenced, foreign_success_reported

            if completed.foreign_settlement_disposition == "succeeded":
                if not foreign_success_reported and on_completed is not None:
                    foreign_success_reported = True
                    on_completed()
                return
            if not foreign_fenced and on_timeout is not None:
                foreign_fenced = True
                on_timeout()

        task.add_done_callback(apply_foreign_disposition)
        apply_foreign_disposition(task)
        on_late_task(task)
        task.cancel()
        raise _CrossLoopFacadeOperationError(
            "isolated facade lifecycle Future belongs to another event loop"
        )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _FACADE_LIFECYCLE_OPERATION_TIMEOUT
    owner = asyncio.current_task()
    # ``Task.cancelling()`` is cumulative.  A cancellation caught before this
    # lifecycle operation was admitted is historical state, not a new request
    # which may reclassify a facade result.
    owner_cancellation_count = owner.cancelling() if owner is not None else 0
    settled = _facade_settlement_event(task)
    cancellation_args: tuple[Any, ...] | None = None

    def parent_cancellation_arrived() -> bool:
        current_owner = asyncio.current_task()
        return (
            current_owner is owner
            and current_owner is not None
            and current_owner.cancelling() > owner_cancellation_count
        )

    def remember_cancellation(exc: asyncio.CancelledError) -> None:
        nonlocal cancellation_args

        if cancellation_args is None and parent_cancellation_arrived():
            # This exception came from waiting on a neutral settlement event,
            # never from the facade operation.  Its args therefore preserve
            # Python's first-delivered semantics, including an empty first
            # cancellation followed by a later reason.
            cancellation_args = tuple(exc.args)

    async def observe_delivered_parent_cancellation() -> None:
        """Receive a cancellation accepted as a child settled concurrently."""

        if cancellation_args is not None or not parent_cancellation_arrived():
            return
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            remember_cancellation(exc)

    def replay_remembered_cancellation() -> NoReturn:
        """Raise the caller cancellation without retaining facade state."""

        nonlocal operation, task, on_completed, on_timeout, on_late_task, owner

        # A CancelledError keeps this helper's traceback.  Clear every local
        # which can lead back to a facade task, a bound facade callback, or an
        # external coroutine before it becomes the public exception.
        operation = None
        task = None
        on_completed = None
        on_timeout = None
        on_late_task = None
        owner = None
        raise asyncio.CancelledError(*cancellation_args)

    def raise_authoritative_timeout() -> NoReturn:
        """Raise the host timeout without retaining a completed facade task."""

        nonlocal operation, task, on_completed, on_timeout, on_late_task, owner

        # Like the cancellation path above, this public exception keeps this
        # helper's traceback.  A cooperative task can raise a secret-bearing
        # facade exception while acknowledging our own timeout cancellation;
        # clear every route from that traceback back to the consumed task.
        operation = None
        task = None
        on_completed = None
        on_timeout = None
        on_late_task = None
        owner = None
        raise _FacadeLifecycleOperationTimedOut(
            "isolated facade lifecycle operation exceeded its settlement budget"
        )

    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(settled.wait(), timeout=remaining)
        except asyncio.CancelledError as exc:
            # ``settled`` reports child cancellation as data.  A caught
            # cancellation can only be injected into this lifecycle owner.
            remember_cancellation(exc)
        except asyncio.TimeoutError:
            break
        except BaseException:  # noqa: BLE001 - facade outcome stays private after cancel
            # The task can fail in the scheduling turn immediately after a
            # caller cancellation was recorded.  Do not let that external
            # exception replace the caller's authoritative interruption (or
            # retain its traceback through the public CancelledError).
            if cancellation_args is not None:
                _consume_late_lifecycle_task_outcome(task)
                # Leave this ``except`` before replaying cancellation: raising
                # inside it would chain the facade exception as the public
                # CancelledError's ``__context__``.
                break
            raise

    if not task.done():
        # Never deliver a parent cancellation into the real SDK operation.
        # This is the explicit fencing cancellation after its full stop budget
        # elapsed; even a facade which returns ``None`` after receiving it has
        # an uncertain process outcome and must not be treated as success.
        if on_timeout is not None:
            on_timeout()
        task.cancel()
        acknowledgement_deadline = (
            loop.time() + _FACADE_LIFECYCLE_CANCELLATION_GRACE
        )
        while not task.done():
            remaining = acknowledgement_deadline - loop.time()
            if remaining <= 0:
                # The handoff is deliberately mandatory.  A timeout must not
                # detach a still-running SDK stop coroutine: it may own the
                # sole subprocess handle after its public facade has detached
                # it.  Record that exact task before releasing this caller.
                on_late_task(task)
                if cancellation_args is not None:
                    replay_remembered_cancellation()
                raise_authoritative_timeout()
            try:
                await asyncio.wait_for(settled.wait(), timeout=remaining)
            except asyncio.CancelledError as exc:
                # The timeout fence cancels only ``task``.  It can set the
                # event, but it cannot inject cancellation into this owner.
                remember_cancellation(exc)
            except asyncio.TimeoutError:
                continue
            except BaseException:  # noqa: BLE001 - timed-out facade outcome is private
                # The timeout is the authoritative lifecycle outcome once we
                # fenced the exact operation.  A late facade failure must not
                # escape this acknowledgement window or become the context of
                # the public timeout/cancellation.
                _consume_late_lifecycle_task_outcome(task)
                break

        # Consume the post-fence terminal result even when a facade swallowed
        # cancellation and returned success.  The timeout itself remains the
        # authoritative lifecycle outcome.
        _consume_late_lifecycle_task_outcome(task)
        await observe_delivered_parent_cancellation()
        if cancellation_args is not None:
            replay_remembered_cancellation()
        raise_authoritative_timeout()

    # A pre-settled source can skip the wait loop entirely.  If the owner was
    # cancelled in the same scheduling turn, receive that delivery before
    # classifying the child terminal state.  Historical cancellation counts do
    # not pass this admission fence.
    await observe_delivered_parent_cancellation()

    if cancellation_args is not None:
        # A cancellation already observed by this owner is its public result.
        # The facade task may have completed in the same scheduling turn, and
        # may itself have failed with data from an external service.  Consume
        # that private result before replaying the caller cancellation; do not
        # let a late facade exception replace the interruption or attach its
        # traceback to the cancellation that leaves this boundary.  A nominal
        # success still invokes ``on_completed`` so callers retain exact-stop
        # knowledge even though their own cancellation wins publicly.
        try:
            task.result()
        except BaseException:  # noqa: BLE001 - facade outcome remains private
            pass
        else:
            if on_completed is not None:
                on_completed()
        replay_remembered_cancellation()
    if task.cancelled():
        raise asyncio.CancelledError()
    result = task.result()
    if on_completed is not None:
        on_completed()
    return result


async def _await_owned_health_probe(
    operation: Any,
    *,
    name: str,
    on_started: Callable[[_HostOwnedFacadeOperation], None],
    on_late_task: Callable[[_HostOwnedFacadeOperation], None],
) -> Any:
    """Await one health probe without orphaning cancellation-resistant work.

    ``asyncio.wait_for(coro, ...)`` creates an inner task implicitly.  When a
    facade's ``health()`` suppresses its timeout or parent cancellation, that
    task can outlive a cancelled supervisor while retaining the facade and its
    credentials.  Explicitly own it instead.  A cooperative probe is still
    cancelled at the health deadline; a hostile one is handed to the proxy
    before this helper releases its supervisor so terminal cleanup can stay
    sealed and honestly incomplete without blocking an agent-wide shutdown.
    """

    task = _create_host_owned_facade_task(operation, name=name)
    on_started(task)
    owner = asyncio.current_task()
    # ``Task.cancelling()`` is cumulative, so its absolute value cannot tell
    # a historical, caught cancellation from one delivered while this probe
    # was running.  Snapshot the public count at ownership admission; only a
    # later increment can make a parent interruption authoritative here.
    owner_cancellation_count = owner.cancelling() if owner is not None else 0

    def parent_cancellation_arrived() -> bool:
        current_owner = asyncio.current_task()
        return (
            current_owner is owner
            and current_owner is not None
            and current_owner.cancelling() > owner_cancellation_count
        )

    if task.foreign_loop:
        # Health must not treat a foreign-loop Future as an ordinary failed
        # probe and restart beside it.  Retain it first, request cancellation
        # through its owning loop, and let supervision terminally fence the
        # facade while the settlement callback owns eventual release.
        on_late_task(task)
        task.cancel()
        raise _CrossLoopFacadeOperationError(
            "isolated facade health Future belongs to another event loop"
        )

    settled = _facade_settlement_event(task)
    cancellation_args: tuple[Any, ...] | None = None

    def remember_parent_cancellation(exc: asyncio.CancelledError) -> None:
        """Keep the first newly delivered parent cancellation payload."""

        nonlocal cancellation_args

        if cancellation_args is None and parent_cancellation_arrived():
            cancellation_args = tuple(exc.args)

    def replay_parent_cancellation(cancellation_args: tuple[Any, ...]) -> NoReturn:
        """Raise cancellation without retaining a facade task in its traceback."""

        nonlocal operation, task, on_started, on_late_task, owner, parent_cancellation_arrived

        operation = None
        task = None
        on_started = None
        on_late_task = None
        owner = None
        parent_cancellation_arrived = None
        raise asyncio.CancelledError(*cancellation_args)

    def raise_authoritative_timeout() -> NoReturn:
        """Publish a neutral timeout after dropping the facade task reference."""

        nonlocal operation, task, on_started, on_late_task, owner, parent_cancellation_arrived

        operation = None
        task = None
        on_started = None
        on_late_task = None
        owner = None
        parent_cancellation_arrived = None
        raise asyncio.TimeoutError()

    async def fence_parent_cancellation() -> None:
        """Fence the facade and allow queued cooperative cancellation to settle.

        A parent cancellation and a child's own cancellation can be queued in
        the same loop turn.  Calling ``on_late_task`` synchronously from the
        parent handler races the child's next cancellation checkpoint and
        incorrectly detaches an operation which is already settling.  One
        cooperative turn is enough to observe that acknowledgement; a child
        that remains pending after it still transfers exact ownership before
        this helper releases its parent.
        """

        task.cancel()
        if not task.done():
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as exc:
                # A later cancellation must not replace the first payload,
                # but its cumulative count remains on the parent task.
                remember_parent_cancellation(exc)
        if task.done():
            _consume_late_lifecycle_task_outcome(task)
            return
        on_late_task(task)

    async def receive_parent_delivery_after_settlement() -> None:
        """Observe a parent cancel accepted in the child's terminal turn."""

        if cancellation_args is not None or not parent_cancellation_arrived():
            return
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            remember_parent_cancellation(exc)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _HEALTH_PROBE_TIMEOUT
    timed_out = False
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            timed_out = True
            break
        try:
            await asyncio.wait_for(settled.wait(), timeout=remaining)
        except asyncio.CancelledError as exc:
            remember_parent_cancellation(exc)
            if cancellation_args is not None:
                break
        except asyncio.TimeoutError:
            timed_out = True
            break

    if not timed_out:
        await receive_parent_delivery_after_settlement()
        if cancellation_args is not None:
            await fence_parent_cancellation()
            replay_parent_cancellation(cancellation_args)
        if task.cancelled():
            operation = None
            task = None
            on_started = None
            on_late_task = None
            raise RuntimeError("isolated facade health probe was cancelled")
        return task.result()

    task.cancel()
    acknowledgement_deadline = loop.time() + _HEALTH_PROBE_CANCELLATION_GRACE
    while not task.done():
        remaining = acknowledgement_deadline - loop.time()
        if remaining <= 0:
            on_late_task(task)
            raise asyncio.TimeoutError()
        try:
            await asyncio.wait_for(settled.wait(), timeout=remaining)
        except asyncio.CancelledError as exc:
            remember_parent_cancellation(exc)
            if cancellation_args is not None:
                break
        except asyncio.TimeoutError:
            continue

    # A timeout-fenced child can settle in the same scheduling turn as a
    # parent cancellation.  The neutral event keeps the two outcomes separate.
    await receive_parent_delivery_after_settlement()
    if cancellation_args is not None:
        await fence_parent_cancellation()
        replay_parent_cancellation(cancellation_args)
    _consume_late_lifecycle_task_outcome(task)
    raise_authoritative_timeout()


class _TrafficGate:
    """A small reader/writer gate around externally visible child traffic.

    Normal tool and event delivery takes the shared side only long enough to
    keep its selected child stable.  A config transition takes the exclusive
    side: it closes admission first, drains calls that were already executing,
    and does not reopen until the caller has reconciled client and durable
    state.  This is deliberately not the reload lock: serialising every normal
    tool call would both hurt throughput and make a long tool execution block
    unrelated calls when no transition is taking place.
    """

    def __init__(self, *, before_reset: Callable[[], None] | None = None) -> None:
        self._condition = asyncio.Condition()
        self._closed = False
        self._sealed = False
        self._active = 0
        self._before_reset = before_reset

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def sealed(self) -> bool:
        return self._sealed

    async def close_and_drain(self) -> None:
        async with self._condition:
            if self._sealed:
                raise _TrafficGateTerminalError()
            self._closed = True
            while self._active:
                await self._condition.wait()
            if self._sealed:
                raise _TrafficGateTerminalError()

    async def close(self) -> None:
        """Stop admitting new work without waiting for existing work.

        Unhealthy-child recovery must first make the selected child unreachable
        and then ask the SDK wrapper to terminate it.  Waiting for an admitted
        RPC here would make that bounded terminate/kill path unreachable when
        the RPC itself is wedged in the child.
        """

        async with self._condition:
            if self._sealed:
                raise _TrafficGateTerminalError()
            self._closed = True

    async def close_if_idle(self) -> bool:
        """Atomically close admission only when no work is already admitted."""

        async with self._condition:
            if self._sealed:
                raise _TrafficGateTerminalError()
            if self._active:
                return False
            self._closed = True
            return True

    async def seal(self) -> None:
        """Make admission terminal without waiting for admitted calls.

        Terminal cleanup must make the child unreachable before it tells the
        SDK wrapper to stop it.  Waiting for active traffic first can deadlock
        an ingress RPC whose child callback will never return: the bounded
        wrapper termination path is precisely what makes that RPC terminal.
        """
        async with self._condition:
            self._sealed = True
            self._closed = True
            self._condition.notify_all()

    async def drain(self) -> None:
        """Wait for work admitted before a close/seal boundary to finish."""

        async with self._condition:
            while self._active:
                await self._condition.wait()

    async def seal_and_drain(self) -> None:
        """Compatibility composition for callers that require both phases."""

        await self.seal()
        await self.drain()

    async def reopen(self) -> None:
        async with self._condition:
            # A terminal proxy may only be reset by an explicit successful
            # initialize.  Cleanup for a cancelled transition must never turn
            # a quarantine/shutdown into an accidental reopen.
            if not self._sealed:
                self._closed = False
                self._condition.notify_all()

    async def reset_and_reopen(self) -> None:
        """Reset terminal admission after a durable child initialization."""

        async with self._condition:
            # The lifecycle predicate must be checked while the gate lock is
            # held.  Checking only before scheduling this coroutine leaves a
            # window where shutdown can latch while this call awaits the
            # condition and the stale initializer then reopens traffic.
            if self._before_reset is not None:
                self._before_reset()
            self._sealed = False
            self._closed = False
            self._condition.notify_all()

    async def _release_admission(self) -> None:
        async with self._condition:
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()

    @asynccontextmanager
    async def admit(self, *, wait_for_open: bool = True):
        async with self._condition:
            while self._closed and not self._sealed:
                if not wait_for_open:
                    raise _TrafficGateClosedError()
                await self._condition.wait()
            if self._sealed:
                raise _TrafficGateTerminalError()
            self._active += 1
        try:
            yield
        finally:
            release = asyncio.create_task(self._release_admission())
            await _await_task_until_complete(release, preserve_cancellation=False)


_HOST_INGRESS_SUCCESS = "success"
_HOST_INGRESS_GENERIC_FAILURE = "generic-failure"
_HOST_INGRESS_UNSUPPORTED = "unsupported"
_HOST_INGRESS_UNKNOWN_NAME = "unknown-name"
_HOST_INGRESS_TERMINAL = "terminal"
_HOST_INGRESS_CANCELLED = "cancelled"

# A valid JSON tree whose encoded form fits the SDK's 64 KiB wire limit can
# contain at most this many nodes: a flat list of one-character JSON scalars
# is the densest possible tree (one byte per scalar plus one separator per
# sibling).  Keep the host-only snapshot bounded *before* allocating a copy.
# The SDK remains the canonical validator for the exact encoded-size and
# nesting limits below; this is solely a traversal/allocation safety fence.
_HOST_INGRESS_SNAPSHOT_NODE_BUDGET = MAX_HOST_INGRESS_PAYLOAD_BYTES // 2
_HOST_INGRESS_SNAPSHOT_DEPTH_BUDGET = 32


@dataclass(frozen=True)
class _HostIngressRequest:
    """A detached, exact-JSON request safe to hand to an SDK facade."""

    name: str
    payload: HostIngressPayload = field(repr=False)


@dataclass(frozen=True)
class _HostIngressOutcome:
    """A non-exceptional result from the private ingress worker.

    Ingress boundary failures intentionally travel as data until the public
    method has discarded its request/client locals.  Raising an SDK/facade
    exception directly from a worker keeps its traceback (and its request
    payload) reachable through the public error.
    """

    status: str
    payload: HostIngressPayload = field(default=None, repr=False)


@dataclass
class _HostIngressOutcomeSlot:
    """One-shot result handoff which keeps a worker task payload-free.

    A completed ``asyncio.Task`` retains its return value. Host ingress can
    carry credentials, so the worker returns ``None`` and hands its detached
    result to the caller through this slot. The consumer clears the slot
    before exposing either a public result or cancellation.
    """

    outcome: _HostIngressOutcome = field(
        default_factory=lambda: _HostIngressOutcome(_HOST_INGRESS_GENERIC_FAILURE),
        repr=False,
    )

    def take(self) -> _HostIngressOutcome:
        outcome = self.outcome
        self.outcome = _HostIngressOutcome(_HOST_INGRESS_GENERIC_FAILURE)
        return outcome


@dataclass
class _NonCursorIngressQueue:
    """One bounded, serial legacy ingress queue owned by an exact child."""

    client: Any
    events: asyncio.Queue[Any]
    worker: asyncio.Task[None] | None = None
    accepting: bool = True
    retired: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _TrackedFacadeLifecycleTask:
    """Non-detached ownership for a cancellation-hostile facade operation."""

    task: _HostOwnedFacadeOperation
    client_id: int
    client_ref: weakref.ReferenceType[Any] | None


@dataclass
class _TerminalStopCompletionMarker:
    """An exact stop-completion marker with safe non-weakref fallback.

    Weak references avoid retaining ordinary SDK facades.  A custom facade
    that cannot be weak-referenced is instead held only until its next
    lifecycle disposition (terminal cleanup consumes it or restart revokes
    it); identity, never an ``id`` cache, remains the proof.
    """

    weak_client: weakref.ReferenceType[Any] | None = None
    strong_client: Any = field(default=None, repr=False)
    # The supervisor registers its completion marker *before* awaiting stop.
    # Terminal cleanup can then claim that in-flight completion instead of
    # forgetting a marker which the callback later recreates after retirement.
    completed: bool = False
    terminal_retirement_claimed: bool = False

    def client(self) -> Any:
        return self.weak_client() if self.weak_client is not None else self.strong_client


def _preflight_exact_host_ingress_json(value: Any) -> None:
    """Account for the SDK's exact JSON bytes without serializing a payload.

    ``validate_host_ingress_payload`` remains the wire contract and is called
    below for every snapshot that reaches it.  Its canonical size check,
    however, necessarily materializes ``json.dumps(..., ensure_ascii=True)``.
    An untrusted exact string containing astral characters expands to twelve
    ASCII bytes per code point there, and an oversized dict key is just as
    capable of forcing that allocation.  Charge every emitted byte first so
    the serializer is reached only for inputs already proven to fit.

    This deliberately mirrors the SDK's compact separators and ASCII encoder:
    container delimiters, commas, colons, scalar tokens, object keys, and JSON
    escaping are all included.  Integer text is produced only after a
    bit-length lower bound proves that it can be at most one bounded wire
    payload; it is not a hidden unbounded serialization escape hatch.
    """

    remaining_bytes = MAX_HOST_INGRESS_PAYLOAD_BYTES
    ancestor_containers: set[int] = set()

    def charge(byte_count: int) -> None:
        nonlocal remaining_bytes

        remaining_bytes -= byte_count
        if remaining_bytes < 0:
            raise ProtocolError("host ingress payload exceeds the size limit")

    def charge_string(candidate: str) -> None:
        # JSON quotes surround every string, including object keys.
        charge(2)
        for character in candidate:
            codepoint = ord(character)
            if character == '"' or character == "\\":
                charge(2)
            elif codepoint <= 0x1F:
                # The five short control escapes are the only non-six-byte
                # escapes emitted by Python's JSON encoder.
                if character in {"\b", "\t", "\n", "\f", "\r"}:
                    charge(2)
                else:
                    charge(6)
            elif codepoint <= 0x7F:
                charge(1)
            elif codepoint <= 0xFFFF:
                charge(6)
            else:
                # ``ensure_ascii=True`` emits a surrogate pair for astral
                # Unicode rather than the source code point itself.
                charge(12)

    def charge_integer(candidate: int) -> None:
        # ``str()`` on an arbitrary integer is itself an allocation.  The
        # lower bound below uses a deliberately low rational approximation of
        # log10(2), so rejecting here cannot reject a valid wire payload.
        bit_length = candidate.bit_length()
        if candidate < 0:
            charge(1)
        if bit_length == 0:
            charge(1)
            return
        minimum_digits = ((bit_length - 1) * 30102) // 100000 + 1
        if minimum_digits > remaining_bytes:
            raise ProtocolError("host ingress payload exceeds the size limit")
        # Any integer that reaches this point is bounded to roughly one wire
        # payload.  Preserve the SDK's exact decimal spelling rather than
        # approximating a near-limit value incorrectly.
        charge(len(str(candidate if candidate >= 0 else -candidate)))

    def visit(candidate: Any, *, depth: int) -> None:
        candidate_type = type(candidate)
        if depth > _HOST_INGRESS_SNAPSHOT_DEPTH_BUDGET:
            raise ProtocolError("host ingress payload exceeds the nesting limit")
        if candidate_type is type(None):
            charge(4)
            return
        if candidate_type is bool:
            charge(4 if candidate else 5)
            return
        if candidate_type is int:
            charge_integer(candidate)
            return
        if candidate_type is float:
            if not math.isfinite(candidate):
                raise ProtocolError("host ingress payload must be valid JSON")
            # Python's compact JSON float path uses the finite float repr.
            charge(len(repr(candidate)))
            return
        if candidate_type is str:
            charge_string(candidate)
            return
        if candidate_type not in (list, dict):
            raise TypeError("host ingress payload must be exact JSON")

        identity = id(candidate)
        if identity in ancestor_containers:
            raise TypeError("host ingress payload must not contain a container cycle")
        ancestor_containers.add(identity)
        try:
            if candidate_type is list:
                charge(1)
                for index, item in enumerate(candidate):
                    if index:
                        charge(1)
                    visit(item, depth=depth + 1)
                charge(1)
                return

            charge(1)
            for index, (key, item) in enumerate(candidate.items()):
                if type(key) is not str:
                    raise TypeError("host ingress payload keys must be exact strings")
                if index:
                    charge(1)
                charge_string(key)
                charge(1)
                visit(item, depth=depth + 1)
            charge(1)
        finally:
            ancestor_containers.remove(identity)

    visit(value, depth=0)


def _copy_exact_host_ingress_json(value: Any) -> Any:
    """Return a detached JSON snapshot made only of exact built-in objects.

    The SDK validator intentionally accepts normal Python JSON-compatible
    values and returns its input unchanged.  At this host trust boundary that
    is insufficient: a ``dict``/``str`` subclass can run user code during
    validation, and a mutable input can change between validation and RPC
    dispatch.  Copy before calling the SDK validator and reject every
    subclass, including dictionary keys, so neither behavior is possible.

    JSON has tree, not graph, semantics.  Shared containers therefore become
    independent copied branches; only a container already on the current
    ancestor stack is a cycle.  The traversal and depth budgets are consumed
    for every branch before each output node is allocated, preventing compact
    alias DAGs from expanding without a bound.
    """

    # Do this before allocating a detached container or asking the SDK to
    # serialize.  In particular, a huge astral scalar or object key must never
    # reach ``json.dumps(..., ensure_ascii=True)`` just to discover its size.
    _preflight_exact_host_ingress_json(value)

    ancestor_containers: set[int] = set()
    remaining_nodes = _HOST_INGRESS_SNAPSHOT_NODE_BUDGET

    def copy_value(candidate: Any, *, depth: int) -> Any:
        nonlocal remaining_nodes

        remaining_nodes -= 1
        if remaining_nodes < 0:
            raise TypeError("host ingress payload exceeds the snapshot budget")
        if depth > _HOST_INGRESS_SNAPSHOT_DEPTH_BUDGET:
            raise TypeError("host ingress payload exceeds the snapshot depth budget")

        candidate_type = type(candidate)
        if candidate_type in (type(None), bool, int, float, str):
            return candidate
        if candidate_type not in (list, dict):
            raise TypeError("host ingress payload must be exact JSON")

        identity = id(candidate)
        if identity in ancestor_containers:
            raise TypeError("host ingress payload must not contain a container cycle")
        ancestor_containers.add(identity)
        try:
            if candidate_type is list:
                copied_list: list[Any] = []
                for item in candidate:
                    copied_list.append(copy_value(item, depth=depth + 1))
                return copied_list

            copied_dict: dict[str, Any] = {}
            for key, item in candidate.items():
                if type(key) is not str:
                    raise TypeError("host ingress payload keys must be exact strings")
                copied_dict[key] = copy_value(item, depth=depth + 1)
            return copied_dict
        finally:
            ancestor_containers.remove(identity)

    snapshot = copy_value(value, depth=0)
    # The source container can be changed from another thread between the
    # first pass and the built-in copy loop.  Re-account the detached snapshot
    # so the SDK still never performs an unbounded serialization allocation.
    _preflight_exact_host_ingress_json(snapshot)
    return snapshot


def _snapshot_host_ingress_payload(value: Any) -> HostIngressPayload:
    """Copy and retain the SDK's size, depth, and finite-float validation."""

    snapshot = _copy_exact_host_ingress_json(value)
    # Validate the detached graph, never a caller/client-owned value.  The SDK
    # remains the canonical authority for JSON wire limits and finite floats.
    validate_host_ingress_payload(snapshot)
    return snapshot


def _prepare_host_ingress_request(
    name: Any, payload: Any
) -> _HostIngressRequest | None:
    """Validate/snapshot untrusted ingress arguments without raising outward."""

    if type(name) is not str:
        return None
    try:
        validated_name = validate_host_ingress_name(name)
        return _HostIngressRequest(
            name=validated_name,
            payload=_snapshot_host_ingress_payload(payload),
        )
    except (
        ProtocolError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        RuntimeError,
    ):
        # Input is untrusted and its representation can contain a webhook
        # credential.  The public boundary turns every validation detail into
        # the one generic error after it has cleared its own arguments.
        return None


def _consume_host_ingress_operation(
    operation: asyncio.Task[Any],
) -> bool:
    """Consume a worker terminal exception and report whether it returned."""

    if operation.cancelled():
        return False
    try:
        operation.result()
    except asyncio.CancelledError:
        return False
    except BaseException:  # noqa: BLE001 - consume every terminal worker failure
        # ``result()`` is deliberately called even after ``shield`` observed
        # the error: it consumes every terminal task exception so asyncio never
        # reports an unhandled shield-future failure during caller cancellation.
        return False
    return True


async def _wait_for_host_ingress_operation(
    operation: asyncio.Task[Any],
    outcome_slot: _HostIngressOutcomeSlot,
) -> tuple[_HostIngressOutcome, tuple[Any, ...] | None]:
    """Drain a shielded worker and remember only caller-originated cancel.

    A child-side ``CancelledError`` propagates from the shielded worker without
    marking the caller task as cancelling.  A cancellation directed at the
    public caller does mark it, and must win even if the child later succeeds
    or fails while we drain it.  Repeated cancellation is consumed here so the
    traffic admission cannot be abandoned midway through cleanup.
    """

    caller = asyncio.current_task()
    caller_cancel_args: tuple[Any, ...] | None = None
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError as error:
            if caller is not None and caller.cancelling():
                if caller_cancel_args is None:
                    caller_cancel_args = tuple(error.args)
                continue
            # The worker's child/RPC cancellation is a terminal worker result,
            # not a cancellation request against this public caller.
            break
        except BaseException:  # noqa: BLE001 - discard worker traceback at boundary
            # Always retrieve the terminal exception below rather than letting
            # a task traceback become the public ingress traceback.
            break
    if _consume_host_ingress_operation(operation):
        outcome = outcome_slot.take()
    else:
        # A task which failed outside the worker must not leave a result in a
        # slot held by its public caller.
        outcome_slot.take()
        outcome = _HostIngressOutcome(_HOST_INGRESS_GENERIC_FAILURE)
    if caller_cancel_args is not None:
        # A completed worker can carry a successful response even though a
        # caller cancellation won while its shield was being drained.  Never
        # return that response through the public cancellation path: exception
        # tracebacks retain every caller frame and could otherwise reach
        # ``outcome.payload`` despite the dataclass's redacted repr.
        outcome = _HostIngressOutcome(_HOST_INGRESS_CANCELLED)
    return outcome, caller_cancel_args


def _deliver_host_ingress_outcome(
    outcome: _HostIngressOutcome,
    caller_cancel_args: tuple[Any, ...] | None,
) -> HostIngressPayload:
    """Raise public errors from a frame that has no request/client locals."""

    if caller_cancel_args is not None:
        raise asyncio.CancelledError(*caller_cancel_args)
    if outcome.status == _HOST_INGRESS_SUCCESS:
        return outcome.payload
    if outcome.status == _HOST_INGRESS_UNSUPPORTED:
        raise HostIngressUnsupportedError("host ingress is not supported")
    if outcome.status == _HOST_INGRESS_UNKNOWN_NAME:
        raise HostIngressUnknownNameError("host ingress name is not available")
    if outcome.status == _HOST_INGRESS_TERMINAL:
        raise HostIngressError("host ingress is unavailable")
    if outcome.status == _HOST_INGRESS_CANCELLED:
        raise asyncio.CancelledError()
    raise HostIngressError("host ingress failed")


def _utc_now() -> datetime:
    """Return the UTC wall clock used for durable config lease decisions."""

    return datetime.now(timezone.utc)


@dataclass
class _ConfigState:
    """One authoritative snapshot of an isolated feature's config node."""

    properties: Optional[Dict[str, Any]] = field(repr=False)
    config: Dict[str, Any] = field(repr=False)
    # Every durable read carries the exact graph node identity it observed.
    # A transition pins that identity through every later lease, cleanup, and
    # promotion write instead of re-resolving midway through the lifecycle.
    node_id: Optional[str] = None
    has_pending: bool = False
    pending_generation: Optional[str] = None
    pending_owner: Optional[str] = None
    pending_lease_expires_at: Optional[datetime] = None


@dataclass
class _ConfigTransition:
    """A generation-owned stage → promote transaction.

    ``expected_properties`` and ``staged_properties`` are exact graph-store
    snapshots.  They are never reconstructed from the proxy cache: a hosted
    replica must only promote the pending generation it actually staged.
    """

    active_config: Dict[str, Any] = field(repr=False)
    next_config: Dict[str, Any] = field(repr=False)
    persistent: bool
    storage: Any = field(repr=False)
    expected_properties: Optional[Dict[str, Any]] = field(repr=False)
    staged_properties: Optional[Dict[str, Any]] = field(repr=False)
    promoted_properties: Optional[Dict[str, Any]] = field(repr=False)
    config_node_id: Optional[str] = None
    generation: Optional[str] = None
    owner: Optional[str] = None


@dataclass(frozen=True)
class _ExternalIngressQuiesce:
    """One acknowledged external-producer pause owned by a config transition."""

    client: Any = field(repr=False)
    transition_id: str = field(repr=False)


@dataclass(frozen=True)
class _DeferredAcknowledgedIngress:
    """One detached, bounded polling event held behind a finite gate."""

    message: dict[str, Any]
    acknowledgement: _HostIngressRequest
    source_client: Any = field(repr=False)
    retry: _HostIngressRequest | None = None
    telegram_terminal_disposition: str | None = None


@dataclass
class _ConfigWriteResult:
    """The direct result of one graph write, before ambiguity is reconciled."""

    committed: bool
    error: Optional[BaseException] = field(default=None, repr=False)
    # Keep the storage predicate outcome so PATCH preservation retries only
    # genuine concurrent winners.  A globally colliding foreign scoped node,
    # for example, is reported as ``not_found`` to this tenant and must not
    # spin forever against an unchanged absent read.
    outcome: Optional[str] = None


@dataclass
class _PromotionResolution:
    """Durable outcome of a promotion after any ambiguous write is re-read."""

    state: _ConfigState
    committed: bool
    error: Optional[BaseException] = field(default=None, repr=False)
    storage_error: bool = False


@dataclass
class _PendingCleanupResolution:
    """Outcome of a generation-scoped pending-state cleanup attempt."""

    state: _ConfigState
    cleared: bool


class SchedulerExecutionContextUnavailable(RuntimeError):
    """A scheduled isolated call cannot carry its trusted effect identity.

    A normal interactive isolated-tool invocation remains compatible with
    legacy SDK services.  A scheduler delivery does not: omitting its
    occurrence identity would let an isolated tool perform an effect without
    the idempotency key that makes lease recovery safe.
    """


class SchedulerTerminalAdmissionError(RuntimeError):
    """A scheduled occurrence reached a terminal isolated traffic gate.

    SchedulerRunner treats exceptions as durable failed occurrences.  Returning
    the ordinary direct-call error envelope here would instead serialize that
    envelope as a successful scheduled result, concealing a shut down or
    quarantined effect boundary.
    """

    def __init__(self) -> None:
        super().__init__(_TERMINAL_TRAFFIC_ERROR)


def _scheduler_trigger_source_id(schedule_id: str) -> str:
    """Return an SDK-safe, stable source ID for a scheduler occurrence.

    The SDK's execution context uses a 512-byte UTF-8 identifier limit.  A
    migrated schedule can legitimately have a longer database ID, so passing
    it through would reject an otherwise safe execution *after* it was
    claimed.  Retain fitting IDs verbatim for compatibility; hash only the
    oversized representation.  Invalid persisted IDs fail closed rather than
    being erased or silently replaced.
    """

    if not isinstance(schedule_id, str) or not schedule_id:
        raise SchedulerExecutionContextUnavailable(
            "scheduled occurrence has an invalid schedule_id"
        )
    if len(schedule_id.encode("utf-8")) <= _SDK_CONTEXT_IDENTIFIER_MAX_BYTES:
        return schedule_id
    digest = hashlib.sha256(schedule_id.encode("utf-8")).hexdigest()
    return f"{_SCHEDULE_TRIGGER_SOURCE_HASH_PREFIX}{digest}"


def _scheduled_tool_execution_context() -> Any | None:
    """Translate the active scheduler occurrence into the public SDK context.

    Keep this lookup lazy so the core continues to start with the currently
    published SDK.  Once a scheduled isolated invocation is attempted, the
    missing SDK contract is a safety failure rather than a reason to smuggle
    scheduler fields into user-controlled tool arguments.
    """

    from kestrel_sovereign.features.scheduler.runner import (
        get_current_scheduler_execution,
    )

    execution = get_current_scheduler_execution()
    if execution is None:
        return None

    try:
        from kestrel_sdk.isolated_feature import (
            ToolExecutionContext,
            ToolExecutionTrigger,
        )
    except ImportError as exc:
        raise SchedulerExecutionContextUnavailable(
            "scheduled isolated tool calls require an SDK with "
            "ToolExecutionContext support"
        ) from exc

    try:
        scheduled_for = datetime.fromisoformat(
            execution.scheduled_for.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise SchedulerExecutionContextUnavailable(
            "scheduled occurrence has an invalid scheduled_for timestamp"
        ) from exc
    if scheduled_for.tzinfo is None or scheduled_for.utcoffset() is None:
        raise SchedulerExecutionContextUnavailable(
            "scheduled occurrence has a timezone-naive scheduled_for timestamp"
        )

    return ToolExecutionContext(
        invocation_id=execution.id,
        idempotency_key=execution.idempotency_key,
        attempt=execution.attempt,
        trigger=ToolExecutionTrigger(
            kind="scheduler",
            id=execution.id,
            source_id=_scheduler_trigger_source_id(execution.schedule_id),
            triggered_at=datetime.now(timezone.utc),
            scheduled_for=scheduled_for.astimezone(timezone.utc),
        ),
    )


def _host_sdk_version() -> str:
    """The kestrel-sdk version resolved in the *host* process, used as the
    provisioning stamp for isolated venvs so a host SDK upgrade forces the
    per-agent venv to reprovision instead of pinning a stale wire contract.

    The ``kestrel_sdk`` import package is shipped by the ``kestrel-sovereign-sdk``
    distribution, so resolve the distribution from the import name rather than
    guessing — a hardcoded wrong name would silently stamp ``unknown`` forever
    and defeat stale detection.
    """
    try:
        candidates = importlib_metadata.packages_distributions().get("kestrel_sdk")
    except Exception:  # noqa: BLE001
        candidates = None
    for dist in list(candidates or []) + ["kestrel-sovereign-sdk", "kestrel-sdk"]:
        try:
            return importlib_metadata.version(dist)
        except Exception:  # noqa: BLE001
            continue
    return "unknown"


def _feature_distribution_version(distribution: str, install_target: str) -> str:
    """Return the host-visible feature release used to provision a child.

    The install target is often deliberately unversioned (for example
    ``kestrel-channel-telegram[service]``), so it cannot tell a provisioned
    venv that the host's feature distribution has changed. Explicit local
    project metadata wins for editable targets; installed distribution
    metadata covers normal package targets.

    ``unknown`` is an honest non-version rather than a moving value: callers
    stamp it once and do not reprovision forever when a target has no readable
    version metadata.
    """
    # Editable installs retain their original ``.dist-info`` version until
    # they are reinstalled. For an explicit local project, its source metadata
    # therefore has to win over that stale installed record or a source version
    # bump could not trigger the very reprovision it requires.
    if isinstance(install_target, str) and install_target:
        local_target = install_target.removeprefix("-e ").strip()
        is_local_target = local_target.startswith((".", "/", "~", "file://"))
        if is_local_target:
            if local_target.startswith("file://"):
                local_target = local_target.removeprefix("file://")
            # pip permits extras on a local directory target; they are not
            # part of the filesystem path containing project metadata.
            local_target = re.sub(r"\[[^]]*\]$", "", local_target)
            pyproject = Path(local_target).expanduser() / "pyproject.toml"
            if pyproject.is_file():
                try:
                    try:
                        import tomllib
                    except ImportError:  # pragma: no cover - Python < 3.11 support
                        import tomli as tomllib  # type: ignore[no-redef]

                    data = tomllib.loads(pyproject.read_text())
                    version = data.get("project", {}).get("version")
                    if isinstance(version, str) and version:
                        return version
                except Exception:  # noqa: BLE001 - use installed metadata below
                    pass

    if isinstance(distribution, str) and distribution:
        try:
            return importlib_metadata.version(distribution)
        except Exception:  # noqa: BLE001 - a non-installed target is stable below
            pass
    return "unknown"


# Probe run *inside* a feature venv to report the kestrel-sdk version actually
# installed there — mirrors _host_sdk_version's distribution resolution.
_CHILD_SDK_PROBE = (
    "from importlib import metadata as m\n"
    "def v():\n"
    "    try: c = m.packages_distributions().get('kestrel_sdk')\n"
    "    except Exception: c = None\n"
    "    for d in list(c or []) + ['kestrel-sovereign-sdk', 'kestrel-sdk']:\n"
    "        try: return m.version(d)\n"
    "        except Exception: continue\n"
    "    return 'unknown'\n"
    "print(v())\n"
)
_ISOLATED_PYTHON_SAFE_PATH_FLAG = "-P"
_NO_BYTECODE_FLAG = "-B"
_CALLABLE_TARGET_VERIFICATION_TIMEOUT_S = 10.0
_FRESHNESS_PROBE_TIMEOUT_S = 10.0
_CALLABLE_TARGET_MISSING_MODULE_EXIT = 40
_CALLABLE_TARGET_MISSING_ATTRIBUTE_EXIT = 41
_CALLABLE_TARGET_NOT_CALLABLE_EXIT = 42
_CALLABLE_TARGET_UNVERIFIABLE_EXIT = 43
_CALLABLE_TARGET_UNSUPPORTED_INTERPRETER_EXIT = 44


def _isolated_python_command(python_path: Path, source: str) -> list[str]:
    """Run supported feature Python without cwd injection or bytecode writes.

    The installed SDK requires Python 3.11+, where ``-P`` provides safe-path
    behavior without implying ``-E``. Hosted ``PYTHONUTF8`` and
    ``PYTHONIOENCODING`` therefore retain their documented stdio semantics.
    The caller supplies an environment with ``PYTHONPATH`` and the other
    interpreter-shadowing variables removed.
    """

    return [
        str(python_path),
        _ISOLATED_PYTHON_SAFE_PATH_FLAG,
        _NO_BYTECODE_FLAG,
        "-c",
        source,
    ]


@dataclass(frozen=True)
class _FeatureDistributionProbe:
    """One classified distribution probe from an isolated interpreter.

    ``importlib.metadata.version`` alone conflates a missing distribution, a
    broken probe, and a positively identified editable/versionless
    distribution.  Prebuilt venvs are immutable, so those states must remain
    distinguishable when deciding whether they are safe to run.
    """

    state: str
    version: str | None = None

    @classmethod
    def versioned(cls, version: str) -> "_FeatureDistributionProbe":
        return cls("versioned", version)

    @classmethod
    def present_unversioned(cls) -> "_FeatureDistributionProbe":
        return cls("present-unversioned")

    @classmethod
    def missing(cls) -> "_FeatureDistributionProbe":
        return cls("missing")

    @classmethod
    def failed(cls) -> "_FeatureDistributionProbe":
        return cls("probe-failed")

    @property
    def is_present(self) -> bool:
        return self.state in {"versioned", "present-unversioned"}


def _venv_feature_distribution_probe(
    python_path: Path,
    distribution: str,
    *,
    hosted: bool = False,
) -> _FeatureDistributionProbe:
    """Classify this runtime distribution inside the isolated child.

    A host-visible release is only an installation intent.  The resolver can
    still select an older index package (notably when the host runs an editable
    or pre-release build), so provisioning must verify the distribution that
    actually landed before declaring the venv fresh.
    """
    if type(distribution) is not str or not distribution:
        return _FeatureDistributionProbe.failed()
    bin_dir = python_path.parent
    venv_path = bin_dir.parent if bin_dir.name in {"bin", "Scripts"} else None
    probe = (
        "from importlib import metadata as m\n"
        f"distribution = {json.dumps(distribution)}\n"
        "try:\n"
        "    installed = m.distribution(distribution)\n"
        "except m.PackageNotFoundError:\n"
        "    print('{\"state\": \"missing\"}')\n"
        "except Exception:\n"
        "    print('{\"state\": \"probe-failed\"}')\n"
        "else:\n"
        "    try:\n"
        "        version = installed.version\n"
        "    except Exception:\n"
        "        version = None\n"
        "    if isinstance(version, str) and version:\n"
        "        import json\n"
        "        print(json.dumps({\"state\": \"versioned\", \"version\": version}))\n"
        "    else:\n"
        "        print('{\"state\": \"present-unversioned\"}')\n"
    )
    try:
        result = subprocess.run(
            _isolated_python_command(python_path, probe),
            check=True,
            capture_output=True,
            text=True,
            env=(
                _isolated_provisioning_env(venv_path)
                if hosted
                else _isolated_child_env(venv_path)
            ),
            timeout=_FRESHNESS_PROBE_TIMEOUT_S,
        )
        decoded = json.loads(result.stdout)
        if type(decoded) is not dict:
            return _FeatureDistributionProbe.failed()
        state = decoded.get("state")
        if state == "versioned" and type(decoded.get("version")) is str and decoded["version"]:
            return _FeatureDistributionProbe.versioned(decoded["version"])
        if state == "present-unversioned":
            return _FeatureDistributionProbe.present_unversioned()
        if state == "missing":
            return _FeatureDistributionProbe.missing()
        if state == "probe-failed":
            return _FeatureDistributionProbe.failed()
        return _FeatureDistributionProbe.failed()
    except subprocess.TimeoutExpired:
        return _FeatureDistributionProbe.failed()
    except Exception:  # noqa: BLE001 - the caller applies the safe stale policy
        return _FeatureDistributionProbe.failed()


def _venv_sdk_version(python_path: Path, *, hosted: bool = False) -> str:
    """The kestrel-sdk version resolved *inside* the feature venv (may differ
    from the host when the feature pins the dependency)."""
    # Reuse the exact child-launch environment.  A bare version probe that
    # inherits host PYTHONPATH/PYTHONHOME/VIRTUAL_ENV can report the host SDK,
    # causing stale/mismatch decisions to stamp the wrong wire contract.
    bin_dir = python_path.parent
    venv_path = (
        bin_dir.parent
        if bin_dir.name in {"bin", "Scripts"}
        else None
    )
    try:
        res = subprocess.run(
            _isolated_python_command(python_path, _CHILD_SDK_PROBE),
            check=True,
            capture_output=True,
            text=True,
            env=(
                _isolated_provisioning_env(venv_path)
                if hosted
                else _isolated_child_env(venv_path)
            ),
            timeout=_FRESHNESS_PROBE_TIMEOUT_S,
        )
        return res.stdout.strip() or "unknown"
    except subprocess.TimeoutExpired:
        return "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_key(feature_name: str, suffix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", feature_name).upper()
    return f"KESTREL_FEATURE_{normalized}_{suffix}"


class IsolatedRuntimeNamespaceError(ValueError):
    """A hosted runtime scope violates containment or ownership policy."""


class IsolatedRuntimePreparationError(RuntimeError):
    """A safe runtime scope could not be prepared because the OS failed.

    Unlike :class:`IsolatedRuntimeNamespaceError`, this does not indicate a
    tenant-boundary violation.  Discovery may therefore mark the optional
    isolated feature unavailable without taking down the rest of the agent.
    """


class _IsolatedRuntimeLaunchTargetPreparationError(
    IsolatedRuntimePreparationError
):
    """Core verified that the selected child target cannot be resolved."""


class _IsolatedRuntimeLaunchTargetMissingModuleError(
    _IsolatedRuntimeLaunchTargetPreparationError
):
    """The declared callable module is absent from the selected venv."""


class _IsolatedRuntimeLaunchTargetMissingAttributeError(
    _IsolatedRuntimeLaunchTargetPreparationError
):
    """The declared callable attribute is absent from its module."""


class _IsolatedRuntimeLaunchTargetNotCallableError(
    _IsolatedRuntimeLaunchTargetPreparationError
):
    """The declared callable attribute resolves to a non-callable object."""


class _IsolatedRuntimeLaunchVerificationTimeoutError(
    IsolatedRuntimePreparationError
):
    """Callable verification exceeded Core's bounded startup budget."""


class _IsolatedRuntimeLaunchVerificationInfrastructureError(
    IsolatedRuntimePreparationError
):
    """The host could not obtain a trustworthy callable verification result."""


_CONFIGURATION_UNSAFE_PROCESS_ENVIRONMENT = "unsafe-process-environment"
_CONFIGURATION_HOSTED_CLIENT_FACTORY = "hosted-client-factory"
_CONFIGURATION_HOSTED_PREBUILT_OVERRIDE = "hosted-prebuilt-override"
_CONFIGURATION_FEATURE_IDENTITY = "feature-identity"
_CONFIGURATION_SERVICE_EXECUTABLE = "service-executable"
_CONFIGURATION_FEATURE_INTERPRETER = "feature-interpreter"
_HOSTED_RUNTIME_VENV_SETTING = "runtime.venv"
_SAFE_ENVIRONMENT_KEY_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,255}")


class IsolatedRuntimeConfigurationError(RuntimeError):
    """Hosted feature configuration cannot be forwarded without leaking scope.

    ``safe_diagnostic`` deliberately derives public/log text from a closed
    reason set and sanitized environment-key names. Startup must not log the
    arbitrary exception message: a third-party feature can raise this public
    exception type, and its message may contain credential values.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        reason: str | None = None,
        environment_keys: tuple[str, ...] = (),
    ) -> None:
        self.reason = reason
        self.environment_keys = tuple(environment_keys)
        super().__init__(
            message
            if message is not None
            else IsolatedRuntimeConfigurationError.safe_diagnostic(self)
        )

    def safe_diagnostic(self) -> str:
        """Return a bounded diagnostic containing names, never env values."""

        try:
            state = object.__getattribute__(self, "__dict__")
        except BaseException:  # pragma: no cover - hostile object seam
            state = {}
        reason = state.get("reason") if type(state) is dict else None
        if type(reason) is not str:
            reason = None
        environment_keys = (
            state.get("environment_keys", ()) if type(state) is dict else ()
        )
        if type(environment_keys) is not tuple:
            environment_keys = ()

        if reason == _CONFIGURATION_UNSAFE_PROCESS_ENVIRONMENT:
            safe_names = []
            invalid_name_seen = False
            for key in environment_keys:
                if type(key) is str and _SAFE_ENVIRONMENT_KEY_NAME.fullmatch(key):
                    safe_names.append(key)
                else:
                    invalid_name_seen = True
            if invalid_name_seen:
                safe_names.append("<invalid-environment-key>")
            names = ", ".join(sorted(set(safe_names))) or "<unknown>"
            return (
                "unsafe process-wide environment keys were found "
                f"({names}); move these keys into persisted per-agent feature "
                "configuration"
            )
        if reason == _CONFIGURATION_HOSTED_CLIENT_FACTORY:
            return (
                "the isolated client factory cannot guarantee tenant-scoped "
                "env and cwd delivery; update it to accept both hosted launch "
                "arguments"
            )
        if reason == _CONFIGURATION_HOSTED_PREBUILT_OVERRIDE:
            safe_names = [
                key
                for key in environment_keys
                if type(key) is str
                and (
                    _SAFE_ENVIRONMENT_KEY_NAME.fullmatch(key)
                    or key == _HOSTED_RUNTIME_VENV_SETTING
                )
            ]
            name = safe_names[0] if len(safe_names) == 1 else "<unknown>"
            return (
                f"hosted process-wide prebuilt override {name} must name an "
                "existing operator-owned executable or venv with no Core "
                "provisioning manifest"
            )
        if reason == _CONFIGURATION_FEATURE_IDENTITY:
            return "isolated feature class name is not a safe canonical identifier"
        if reason == _CONFIGURATION_SERVICE_EXECUTABLE:
            return (
                "isolated feature service must be a bare portable console-script "
                "executable name or a safe Python module:callable target"
            )
        if reason == _CONFIGURATION_FEATURE_INTERPRETER:
            return (
                "isolated Python callable services require a feature interpreter "
                "that supports the SDK's Python 3.11 safe-path contract"
            )
        return "the hosted isolated feature configuration is unsafe"


def safe_isolated_runtime_preparation_diagnostic(
    error: BaseException,
) -> str:
    """Classify a preparation failure without reflecting arbitrary text.

    Optional feature packages can raise the public preparation exception or
    attach an ``OSError`` containing paths and credentials.  Only Core-chosen
    errno categories cross the startup log boundary; all messages and path
    attributes on the original exception remain private.
    """

    if type(error) is _IsolatedRuntimeLaunchTargetMissingModuleError:
        return (
            "the isolated feature Python callable module is absent from its "
            "selected venv; verify the installed feature package"
        )
    if type(error) is _IsolatedRuntimeLaunchTargetMissingAttributeError:
        return (
            "the isolated feature Python callable attribute is absent from its "
            "module; verify the service metadata and installed feature package"
        )
    if type(error) is _IsolatedRuntimeLaunchTargetNotCallableError:
        return (
            "the isolated feature Python callable target resolves to a "
            "non-callable object; verify the service metadata"
        )
    if type(error) is _IsolatedRuntimeLaunchVerificationTimeoutError:
        return (
            "isolated feature callable verification exceeded the bounded "
            "startup timeout; inspect feature import side effects and host health"
        )
    if type(error) is _IsolatedRuntimeLaunchVerificationInfrastructureError:
        return (
            "the host could not complete isolated feature callable verification; "
            "inspect the sanitized traceback and host process health"
        )

    current: BaseException | None = error
    visited: set[int] = set()
    error_number: int | None = None
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, OSError) and type(current.errno) is int:
            error_number = current.errno
            break
        current = current.__cause__ or current.__context__

    if error_number == errno.EXDEV:
        return (
            "released runtime state cannot be adopted across filesystems; "
            "move it onto the hosted runtime filesystem and retry"
        )
    if error_number in {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}:
        return (
            "the runtime filesystem has insufficient free space or quota; "
            "restore capacity and retry"
        )
    if error_number in {errno.EACCES, errno.EPERM, errno.EROFS}:
        return (
            "the host filesystem denied runtime preparation; verify the "
            "configured mount ownership and write policy"
        )
    if error_number in {errno.EMFILE, errno.ENFILE}:
        return (
            "the host exhausted its file-descriptor capacity during runtime "
            "preparation; restore capacity and retry"
        )
    return (
        "the agent-scoped runtime could not be prepared; inspect the sanitized "
        "traceback and host filesystem health"
    )


def sanitized_isolated_runtime_preparation_exc_info(
    error: BaseException,
) -> tuple[type[BaseException], BaseException, TracebackType | None]:
    """Return traceback evidence whose exception text and cause are sanitized."""

    core_frames = []
    current = error.__traceback__
    while current is not None:
        try:
            frame_path = Path(current.tb_frame.f_code.co_filename).resolve(
                strict=False
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            frame_path = None
        if frame_path is not None and (
            frame_path == _CORE_PACKAGE_ROOT
            or _CORE_PACKAGE_ROOT in frame_path.parents
        ):
            core_frames.append(current)
        current = current.tb_next
    safe_traceback: TracebackType | None = None
    for frame in reversed(core_frames):
        safe_traceback = TracebackType(
            safe_traceback,
            frame.tb_frame,
            frame.tb_lasti,
            frame.tb_lineno,
        )
    safe_error = IsolatedRuntimePreparationError(
        safe_isolated_runtime_preparation_diagnostic(error)
    )
    safe_error.__traceback__ = safe_traceback
    safe_error.__cause__ = None
    safe_error.__context__ = None
    safe_error.__suppress_context__ = True
    return type(safe_error), safe_error, safe_traceback


class _RuntimeOwnerMarkerMissing(IsolatedRuntimeNamespaceError):
    """Internal signal: creation is allowed, but cleanup must still refuse."""


@dataclass(frozen=True)
class IsolatedRuntimeNamespace:
    """Canonical, agent-owned location for mutable isolated-feature runtime data.

    A hosted factory supplies a host-owned root plus a tenant/agent namespace.
    Resolving both together here keeps the containment rule in one place rather
    than trusting every feature caller to join and sanitize path fragments.
    """

    root: Path
    namespace: Path
    path: Path


class RuntimeNamespaceCleanupOutcome(str, Enum):
    """Exact custody result from a secure runtime deletion primitive."""

    REMOVED = "removed"
    ALREADY_ABSENT = "already_absent"
    NOT_HOSTED = "not_hosted"


_RUNTIME_NAMESPACE_COMPONENT = re.compile(
    r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$"
)
_WINDOWS_RESERVED_COMPONENTS = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)
_RUNTIME_OWNER_MARKER = ".kestrel-runtime-owner"
_RUNTIME_OWNER_TEMP_PREFIX = ".kestrel-runtime-owner.tmp-"
_VENV_RELOCATION_REPAIR_MARKER = ".kestrel-venv-relocation-pending"
_VENV_RELOCATION_REPAIR_TEMP_PREFIX = ".kestrel-venv-relocation-pending.tmp-"
_VENV_RELOCATION_REPAIR_PAYLOAD = b"kestrel-venv-relocation-v1\n"
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_ISOLATED_FEATURE_CLASS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_ISOLATED_SERVICE_EXECUTABLE_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)
_ISOLATED_SERVICE_PYTHON_IDENTIFIER = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"
)
_ISOLATED_SERVICE_CALLABLE_MAX_LENGTH = 512
_HOSTED_FEATURE_RUNTIME_COMPONENT = re.compile(r"^feature-[0-9a-f]{64}$")
_DERIVED_NAMESPACE_DIGEST_HEX_CHARS = 58


def _is_windows_reserved_runtime_component(component: str) -> bool:
    return component.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_COMPONENTS


def _validated_isolated_service_executable(service: object) -> str:
    """Return one portable basename used identically for verify and launch."""

    if (
        type(service) is not str
        or not service
        or service != service.strip()
        or _ISOLATED_SERVICE_EXECUTABLE_NAME.fullmatch(service) is None
        or service.endswith(".")
        or _is_windows_reserved_runtime_component(service)
    ):
        raise IsolatedRuntimeConfigurationError(
            reason=_CONFIGURATION_SERVICE_EXECUTABLE,
        )
    return service


@dataclass(frozen=True)
class _IsolatedServiceTarget:
    """One validated console executable or Python callable launch target."""

    raw: str
    console_executable: str | None = None
    module: str | None = None
    callable_name: str | None = None

    @property
    def is_callable(self) -> bool:
        return self.module is not None


def _is_safe_service_python_identifier(value: str) -> bool:
    return (
        _ISOLATED_SERVICE_PYTHON_IDENTIFIER.fullmatch(value) is not None
        and not keyword.iskeyword(value)
    )


def _validated_isolated_service_target(service: object) -> _IsolatedServiceTarget:
    """Classify one non-reflective, code-injection-safe service declaration."""

    if type(service) is not str or not service or service != service.strip():
        raise IsolatedRuntimeConfigurationError(
            reason=_CONFIGURATION_SERVICE_EXECUTABLE,
        )
    if ":" not in service:
        return _IsolatedServiceTarget(
            raw=service,
            console_executable=_validated_isolated_service_executable(service),
        )
    if (
        len(service) > _ISOLATED_SERVICE_CALLABLE_MAX_LENGTH
        or service.count(":") != 1
    ):
        raise IsolatedRuntimeConfigurationError(
            reason=_CONFIGURATION_SERVICE_EXECUTABLE,
        )
    module, callable_name = service.split(":", 1)
    module_components = module.split(".")
    if (
        not module_components
        or (len(module) == 1 and module.isascii() and module.isalpha())
        or any(
            not _is_safe_service_python_identifier(component)
            for component in module_components
        )
        or not _is_safe_service_python_identifier(callable_name)
    ):
        raise IsolatedRuntimeConfigurationError(
            reason=_CONFIGURATION_SERVICE_EXECUTABLE,
        )
    return _IsolatedServiceTarget(
        raw=service,
        module=module,
        callable_name=callable_name,
    )


def safe_isolated_runtime_exception_type_name(error: BaseException) -> str:
    """Return a bounded identifier for sanitized server-side diagnostics."""

    name = type(error).__name__
    if _ISOLATED_FEATURE_CLASS_NAME.fullmatch(name) is not None:
        return name
    return "RuntimeError"


def derive_isolated_runtime_namespace(*identifiers: str) -> str:
    """Derive one portable, collision-resistant namespace from host identities.

    The host decides which authenticated identities define ownership (for
    example ``user_id, companion_id``). Length-prefixing keeps tuples such as
    ``("ab", "c")`` and ``("a", "bc")`` distinct without putting private or
    path-active identifier text on disk.
    """

    if not identifiers:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime namespace requires an identity."
        )
    digest = hashlib.sha256()
    for identifier in identifiers:
        if type(identifier) is not str or not identifier:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime identities must be non-empty strings."
            )
        encoded = identifier.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    # Keep the opaque component within the same grammar accepted from hosted
    # factories. Fifty-eight hex digits retain a 232-bit digest (a roughly
    # 2^116 birthday bound) while making the complete value exactly 64
    # characters.
    value = f"agent-{digest.hexdigest()[:_DERIVED_NAMESPACE_DIGEST_HEX_CHARS]}"
    if (
        _RUNTIME_NAMESPACE_COMPONENT.fullmatch(value) is None
        or _is_windows_reserved_runtime_component(value)
    ):
        raise AssertionError(
            "derived isolated-runtime namespace no longer satisfies its grammar"
        )
    return value


def _hosted_feature_runtime_component(runtime: InstalledFeatureRuntime) -> str:
    """Return the durable, path-inert identity of one hosted feature.

    Entry-point module paths and service runner spellings are packaging
    details, not feature identity.  Moving either must not relocate mutable
    tenant state or credentials.  Distribution names use their PEP 503
    comparison form so punctuation/case-only metadata changes are inert too.
    Distinct classes in one distribution remain independently scoped.
    """

    digest = hashlib.sha256()
    distribution = re.sub(r"[-_.]+", "-", runtime.distribution.strip()).lower()
    for value in (distribution, runtime.class_name):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"feature-{digest.hexdigest()}"


def _legacy_hosted_feature_runtime_component(
    runtime: InstalledFeatureRuntime,
) -> str:
    """Return the issue-2716 pre-stable hosted directory identity.

    This exists only to migrate a tree created by the earlier implementation.
    New runtime paths must always use :func:`_hosted_feature_runtime_component`.
    """

    digest = hashlib.sha256()
    for value in (
        runtime.distribution,
        runtime.class_name,
        runtime.entry_point,
        runtime.service or "",
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"feature-{digest.hexdigest()}"


def resolve_isolated_runtime_namespace(
    root: str | os.PathLike[str],
    namespace: str | os.PathLike[str],
) -> IsolatedRuntimeNamespace:
    """Validate and canonicalize a hosted agent's runtime namespace.

    ``namespace`` is deliberately relative to the configured host root.  Reject
    traversal components even when normalization would remain inside that root:
    accepting them makes a tenant identifier's meaning depend on path parsing
    and invites future callers to join it unsafely.
    """
    if not isinstance(root, (str, os.PathLike)):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated features require an explicit runtime root."
        )
    if not isinstance(namespace, (str, os.PathLike)):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated features require an explicit runtime namespace."
        )

    try:
        root_text = os.fspath(root)
        namespace_text = os.fspath(namespace)
    except TypeError as exc:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime paths must be filesystem paths."
        ) from exc
    if type(root_text) is not str or not root_text or "\x00" in root_text:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime root must be a non-empty path."
        )
    if (
        type(namespace_text) is not str
        or not namespace_text
        or namespace_text != namespace_text.strip()
        or "\x00" in namespace_text
    ):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime namespace must not be empty."
        )
    if "\\" in namespace_text or namespace_text.startswith("/"):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime namespace must be a canonical "
            "relative path."
        )
    components = namespace_text.split("/")
    if (
        "/".join(components) != namespace_text
        or any(
            component in {"", ".", ".."}
            or _RUNTIME_NAMESPACE_COMPONENT.fullmatch(component) is None
            or _is_windows_reserved_runtime_component(component)
            for component in components
        )
    ):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime namespace must contain only "
            "canonical lowercase path components."
        )
    relative_namespace = Path(*components)

    try:
        canonical_root = Path(root_text).expanduser().resolve()
    except OSError as exc:
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature runtime root could not be resolved."
        ) from exc
    if canonical_root == Path(canonical_root.anchor):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime root must not be a filesystem root."
        )
    if canonical_root.exists() and not canonical_root.is_dir():
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime root must be a directory."
        )
    if not canonical_root.parent.is_dir():
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime root parent must already exist; "
            "refusing to materialize a missing operator volume path."
        )
    canonical_path = canonical_root.joinpath(*components)

    return IsolatedRuntimeNamespace(
        root=canonical_root,
        namespace=relative_namespace,
        path=canonical_path,
    )


def resolve_legacy_isolated_runtime_root(
    root: str | os.PathLike[str],
    scope: IsolatedRuntimeNamespace,
) -> Path:
    """Validate the factory-owned source of the released hosted layout.

    This path is migration input only.  It must name the exact historical
    ``feature_venvs`` directory below an already-existing per-agent data
    directory, and it must be disjoint from the new runtime root.  Filesystem
    identity and ownership are checked again descriptor-relatively when a
    feature is adopted; construction deliberately performs no creation.
    """

    if not isinstance(root, (str, os.PathLike)):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime root must be a path."
        )
    try:
        root_text = os.fspath(root)
    except TypeError as exc:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime root must be a path."
        ) from exc
    if type(root_text) is not str or not root_text or "\x00" in root_text:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime root must be a non-empty path."
        )
    expanded = os.path.expanduser(root_text)
    if not os.path.isabs(expanded):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime root must be absolute."
        )
    canonical = Path(os.path.abspath(expanded))
    if canonical.name != "feature_venvs":
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime root must name feature_venvs."
        )
    if not canonical.parent.is_dir():
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime parent must already exist."
        )
    try:
        resolved_parent = canonical.parent.resolve(strict=True)
    except OSError as exc:
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature legacy runtime parent could not be resolved."
        ) from exc
    if resolved_parent != canonical.parent:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime parent must not contain "
            "path aliases or symlinks."
        )
    if canonical.is_symlink():
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime root must not be a symlink."
        )
    try:
        canonical.relative_to(scope.root)
    except ValueError:
        pass
    else:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy and current runtime roots must be disjoint."
        )
    try:
        scope.root.relative_to(canonical)
    except ValueError:
        pass
    else:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy and current runtime roots must be disjoint."
        )
    return canonical


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _rename_directory_noreplace_at(
    source_parent_fd: int,
    source_component: str,
    target_parent_fd: int,
    target_component: str,
) -> None:
    """Atomically rename a directory while refusing any existing target.

    Plain POSIX ``rename`` may replace an empty directory created after a
    collision pre-check, destroying the competing custody claim. Linux and
    Darwin both expose an exclusive descriptor-relative variant. Other POSIX
    platforms fail closed rather than weakening migration semantics.
    """

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_component)
    target = os.fsencode(target_component)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_parent_fd,
            source,
            target_parent_fd,
            target,
            1,  # RENAME_NOREPLACE
        )
    else:
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory rename is unavailable",
            )
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            source_parent_fd,
            source,
            target_parent_fd,
            target,
            0x00000004,  # RENAME_EXCL
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _secure_dirfd_supported() -> bool:
    required = (os.open, os.mkdir, os.stat)
    return (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and all(function in os.supports_dir_fd for function in required)
    )


def _validate_operator_root_metadata(metadata: os.stat_result) -> None:
    if os.name == "posix" and metadata.st_uid != os.geteuid():
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime root must be owned by the service "
            "account."
        )
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o022:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime root must not be group- or "
            "world-writable."
        )


def _open_or_create_directory_at(
    parent_fd: int,
    component: str,
    *,
    enforce_private: bool = True,
    operator_root: bool = False,
) -> int:
    created = False
    try:
        descriptor = os.open(component, _directory_open_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(component, mode=_PRIVATE_DIRECTORY_MODE, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        descriptor = os.open(component, _directory_open_flags(), dir_fd=parent_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime path contains a non-directory entry."
        )
    try:
        if created or enforce_private:
            os.fchmod(descriptor, _PRIVATE_DIRECTORY_MODE)
        elif operator_root:
            _validate_operator_root_metadata(metadata)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_secure_absolute_directory(path: Path) -> int:
    # The root's parent is operator custody.  It must already exist (validated
    # by ``resolve_isolated_runtime_namespace``); Core may create only the root
    # leaf itself, never a plausible-looking replacement for a missing mount.
    parent_fd = os.open(path.parent, _directory_open_flags())
    try:
        parent_lexical = os.stat(path.parent, follow_symlinks=False)
        if not stat.S_ISDIR(parent_lexical.st_mode) or not _same_file_identity(
            parent_lexical, os.fstat(parent_fd)
        ):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime root parent changed during "
                "validation."
            )
        descriptor = _open_or_create_directory_at(
            parent_fd,
            path.name,
            enforce_private=False,
            operator_root=True,
        )
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        lexical = os.stat(path, follow_symlinks=False)
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(lexical.st_mode) or not _same_file_identity(
            lexical, opened
        ):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime root changed during validation."
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
    finally:
        os.close(parent_fd)


def _runtime_owner_bytes(owner: str) -> bytes:
    return (hashlib.sha256(owner.encode("utf-8")).hexdigest() + "\n").encode()


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("runtime ownership marker write made no progress")
        view = view[written:]


def _read_venv_relocation_repair_marker_at(directory_fd: int) -> bool:
    """Validate a durable relocation marker without following path entries."""

    marker_fd: Optional[int] = None
    try:
        try:
            metadata = os.stat(
                _VENV_RELOCATION_REPAIR_MARKER,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (os.name == "posix" and metadata.st_uid != os.geteuid())
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature relocation repair marker is unsafe."
            )
        if metadata.st_nlink != 1:
            for name in os.listdir(directory_fd):
                if not name.startswith(_VENV_RELOCATION_REPAIR_TEMP_PREFIX):
                    continue
                try:
                    candidate = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(candidate.st_mode) and _same_file_identity(
                    candidate, metadata
                ):
                    try:
                        os.unlink(name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
            try:
                metadata = os.stat(
                    _VENV_RELOCATION_REPAIR_MARKER,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return False
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (os.name == "posix" and metadata.st_uid != os.geteuid())
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature relocation repair marker has an "
                    "unsafe external hard link."
                )
        marker_fd = os.open(
            _VENV_RELOCATION_REPAIR_MARKER,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        if not _same_file_identity(metadata, os.fstat(marker_fd)):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature relocation repair marker changed during "
                "validation."
            )
        payload = os.read(marker_fd, len(_VENV_RELOCATION_REPAIR_PAYLOAD) + 1)
        if payload != _VENV_RELOCATION_REPAIR_PAYLOAD:
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature relocation repair state is corrupt; "
                "tenant state was retained."
            )
        return True
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature relocation repair marker is unsafe."
            ) from exc
        raise
    finally:
        if marker_fd is not None:
            os.close(marker_fd)


def _ensure_venv_relocation_repair_marker_at(directory_fd: int) -> None:
    """Publish repair intent before a feature-directory rename can commit."""

    if _read_venv_relocation_repair_marker_at(directory_fd):
        return
    temporary = (
        f"{_VENV_RELOCATION_REPAIR_TEMP_PREFIX}{os.getpid()}-{uuid4().hex}"
    )
    temporary_fd: Optional[int] = None
    try:
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            _PRIVATE_FILE_MODE,
            dir_fd=directory_fd,
        )
        _write_all(temporary_fd, _VENV_RELOCATION_REPAIR_PAYLOAD)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        try:
            os.link(
                temporary,
                _VENV_RELOCATION_REPAIR_MARKER,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            pass
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.fsync(directory_fd)
        if not _read_venv_relocation_repair_marker_at(directory_fd):
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature relocation repair state could not be "
                "recorded."
            )
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _read_venv_relocation_repair_marker_portable(directory: Path) -> bool:
    """Portable fallback for validating relocation repair intent."""

    marker = directory / _VENV_RELOCATION_REPAIR_MARKER
    try:
        metadata = marker.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or (
            os.name == "posix"
            and (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            )
        )
    ):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature relocation repair marker is unsafe."
        )
    if metadata.st_nlink != 1:
        for candidate in directory.iterdir():
            if not candidate.name.startswith(_VENV_RELOCATION_REPAIR_TEMP_PREFIX):
                continue
            try:
                candidate_metadata = candidate.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(candidate_metadata.st_mode) and _same_file_identity(
                candidate_metadata, metadata
            ):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass
        try:
            metadata = marker.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (
                os.name == "posix"
                and (
                    metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) & 0o077
                )
            )
        ):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature relocation repair marker has an unsafe "
                "external hard link."
            )
    if marker.read_bytes() != _VENV_RELOCATION_REPAIR_PAYLOAD:
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature relocation repair state is corrupt; tenant "
            "state was retained."
        )
    return True


def _ensure_venv_relocation_repair_marker_portable(directory: Path) -> None:
    """Portable fallback for publishing relocation repair intent."""

    if _read_venv_relocation_repair_marker_portable(directory):
        return
    marker = directory / _VENV_RELOCATION_REPAIR_MARKER
    temporary = directory / (
        f"{_VENV_RELOCATION_REPAIR_TEMP_PREFIX}{os.getpid()}-{uuid4().hex}"
    )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _PRIVATE_FILE_MODE,
        )
        _write_all(descriptor, _VENV_RELOCATION_REPAIR_PAYLOAD)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, marker, follow_symlinks=False)
        except FileExistsError:
            pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if not _read_venv_relocation_repair_marker_portable(directory):
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature relocation repair state could not be "
                "recorded."
            )
        if os.name == "posix":
            directory_fd = os.open(directory, _directory_open_flags())
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _recover_runtime_owner_install_link(
    directory_fd: int,
    marker_fd: int,
) -> os.stat_result:
    """Finish the unlink half of an interrupted temp+link marker install."""

    metadata = os.fstat(marker_fd)
    if metadata.st_nlink == 1:
        return metadata
    for name in os.listdir(directory_fd):
        if not name.startswith(_RUNTIME_OWNER_TEMP_PREFIX):
            continue
        try:
            candidate = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            # Another preparer/cleaner may have completed recovery after this
            # directory snapshot was taken.
            continue
        if stat.S_ISREG(candidate.st_mode) and _same_file_identity(candidate, metadata):
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                # The hard-link name is recovery debris, not authoritative
                # state. Concurrent disappearance is the desired outcome.
                pass
    metadata = os.fstat(marker_fd)
    if metadata.st_nlink != 1:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime ownership marker has an unsafe "
            "external hard link."
        )
    os.fsync(directory_fd)
    return metadata


def _read_existing_runtime_owner(directory_fd: int, owner: str) -> None:
    owner_bytes = _runtime_owner_bytes(owner)
    try:
        marker_fd = os.open(
            _RUNTIME_OWNER_MARKER,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except FileNotFoundError as exc:
        raise _RuntimeOwnerMarkerMissing(
            "Hosted isolated feature runtime ownership marker is missing; "
            "refusing to assume tenant ownership."
        ) from exc
    try:
        metadata = os.fstat(marker_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime ownership marker is unsafe."
            )
        metadata = _recover_runtime_owner_install_link(directory_fd, marker_fd)
        existing = os.read(marker_fd, len(owner_bytes) + 1)
        if metadata.st_size != len(owner_bytes) or len(existing) != len(owner_bytes):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime ownership marker is corrupt; "
                "operator verification is required before recovery."
            )
        if not hmac.compare_digest(existing, owner_bytes):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime namespace is already owned by "
                "a different agent."
            )
        os.fchmod(marker_fd, _PRIVATE_FILE_MODE)
    finally:
        os.close(marker_fd)


def _read_or_create_runtime_owner(directory_fd: int, owner: str) -> None:
    """Atomically publish a complete, durable tenant ownership marker."""

    try:
        _read_existing_runtime_owner(directory_fd, owner)
        return
    except _RuntimeOwnerMarkerMissing:
        pass

    owner_bytes = _runtime_owner_bytes(owner)
    temporary = f"{_RUNTIME_OWNER_TEMP_PREFIX}{os.getpid()}-{uuid4().hex}"
    temporary_fd = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        _PRIVATE_FILE_MODE,
        dir_fd=directory_fd,
    )
    try:
        try:
            _write_all(temporary_fd, owner_bytes)
            os.fchmod(temporary_fd, _PRIVATE_FILE_MODE)
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise

    published = False
    try:
        try:
            os.link(
                temporary,
                _RUNTIME_OWNER_MARKER,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError:
            pass
    finally:
        # A concurrent preparer can recover the just-published hard link and
        # remove this temporary name before the creator reaches its cleanup
        # tail.  The marker is already durable; a missing install temp is the
        # expected result of that recovery, not a preparation failure.
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
    if published:
        os.fsync(directory_fd)
    # A concurrent creator may have won.  Re-read the published marker in all
    # cases so collision and corrupt-marker policy has one authoritative path.
    _read_existing_runtime_owner(directory_fd, owner)


def _migrate_runtime_directory_at(
    parent_fd: int,
    legacy_component: str,
    stable_component: str,
) -> bool:
    """Atomically adopt one legacy feature directory under an open parent.

    Both names are opaque Core-derived components.  A target collision is not
    guessed at or merged: either tree can contain credentials, so retaining
    both and making the optional feature unavailable is the only safe policy.
    """

    if legacy_component == stable_component:
        return False
    if (
        _HOSTED_FEATURE_RUNTIME_COMPONENT.fullmatch(legacy_component) is None
        or _HOSTED_FEATURE_RUNTIME_COMPONENT.fullmatch(stable_component) is None
    ):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime migration names are not canonical."
        )
    try:
        legacy_metadata = os.stat(
            legacy_component,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(legacy_metadata.st_mode):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature legacy runtime path is unsafe."
        )
    legacy_fd = os.open(
        legacy_component,
        _directory_open_flags(),
        dir_fd=parent_fd,
    )
    try:
        if not _same_file_identity(legacy_metadata, os.fstat(legacy_fd)):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature legacy runtime changed during migration."
            )
    finally:
        os.close(legacy_fd)

    try:
        stable_metadata = os.stat(
            stable_component,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        stable_metadata = None
    if stable_metadata is not None:
        if not stat.S_ISDIR(stable_metadata.st_mode):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature stable runtime path is unsafe."
            )
        _log_runtime_migration_collision(legacy_component, stable_component)
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature has both legacy and stable runtime state; "
            "operator custody reconciliation is required."
        )

    legacy_fd = os.open(
        legacy_component,
        _directory_open_flags(),
        dir_fd=parent_fd,
    )
    try:
        if not _same_file_identity(legacy_metadata, os.fstat(legacy_fd)):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature legacy runtime changed during migration."
            )
        _ensure_venv_relocation_repair_marker_at(legacy_fd)
    finally:
        os.close(legacy_fd)

    try:
        _rename_directory_noreplace_at(
            parent_fd,
            legacy_component,
            parent_fd,
            stable_component,
        )
    except FileNotFoundError:
        # A concurrent preparer may have completed the same atomic rename.
        # Accept only that exact terminal shape; every other disappearance is
        # retained as an ambiguous preparation failure.
        try:
            post_legacy = os.stat(
                legacy_component,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            post_legacy = None
        try:
            post_stable = os.stat(
                stable_component,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            post_stable = None
        if post_legacy is not None or post_stable is None:
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature legacy runtime migration did not "
                "complete; tenant state was retained."
            ) from None
        if not stat.S_ISDIR(post_stable.st_mode):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature stable runtime path is unsafe."
            )
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        _log_runtime_migration_collision(legacy_component, stable_component)
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature has both legacy and stable runtime state; "
            "operator custody reconciliation is required."
        ) from exc
    os.fsync(parent_fd)
    return True


def _log_runtime_migration_collision(
    legacy_component: str,
    stable_component: str,
    *,
    legacy_parent: str = "feature_venvs",
) -> None:
    """Log opaque namespace-relative custody locations for the operator.

    The exception crossing feature/API boundaries deliberately contains no
    filesystem names.  Operators still need enough server-side evidence to
    reconcile the two retained credential trees, so log only Core-derived
    relative components -- never the configured root or tenant namespace.
    """

    logger.error(
        "Hosted isolated feature runtime migration collision; both trees were "
        "retained and the feature is unavailable pending custody reconciliation "
        "(legacy=%s, stable=%s)",
        f"{legacy_parent}/{legacy_component}",
        f"feature_venvs/{stable_component}",
    )


def _validate_released_legacy_directory_metadata(
    metadata: os.stat_result,
) -> None:
    """Require custody properties Core can prove without mutating old state."""

    if not stat.S_ISDIR(metadata.st_mode):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature released legacy runtime path is unsafe."
        )
    if os.name == "posix" and metadata.st_uid != os.geteuid():
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature released legacy runtime custody could not "
            "be proven; tenant state was retained."
        )
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o022:
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature released legacy runtime custody could not "
            "be proven; tenant state was retained."
        )


def _validate_released_legacy_root_custody(metadata: os.stat_result) -> None:
    """Quarantine populated released state whose custody cannot be proven.

    The released ``feature_venvs`` parent predates Core's private-directory
    contract and may legitimately remain permissive after every class-named
    child has already been migrated.  Its mode is therefore relevant only
    when the requested legacy component still exists.  A non-directory is a
    path-substitution violation; ownership or write-access ambiguity for a
    populated directory is an operational custody problem and quarantines only
    the optional feature without moving either tree.
    """

    if not stat.S_ISDIR(metadata.st_mode):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature released legacy runtime root is unsafe."
        )
    if (
        os.name == "posix"
        and (
            metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        )
    ):
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature released runtime custody could not be "
            "proven; tenant state was retained."
        )


def _revalidate_portable_released_root(
    legacy_root: Path,
    expected: os.stat_result,
) -> None:
    """Detect a portable-path root substitution before any rename."""

    try:
        current = legacy_root.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature released legacy runtime root changed "
            "during validation."
        ) from exc
    if not stat.S_ISDIR(current.st_mode) or not _same_file_identity(expected, current):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature released legacy runtime root changed "
            "during validation."
        )


def _migrate_released_runtime_directory_portable(
    legacy_root: Path,
    scope: IsolatedRuntimeNamespace,
    legacy_component: str,
    stable_component: str,
) -> bool:
    """Best-effort non-dirfd adoption with fail-closed custody checks."""

    try:
        root_metadata = legacy_root.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature released legacy runtime root is unsafe."
        )
    legacy = legacy_root / legacy_component
    stable = scope.path / "feature_venvs" / stable_component
    try:
        legacy_metadata = legacy.stat(follow_symlinks=False)
    except FileNotFoundError:
        _revalidate_portable_released_root(legacy_root, root_metadata)
        return False
    _revalidate_portable_released_root(legacy_root, root_metadata)
    _validate_released_legacy_root_custody(root_metadata)
    if stat.S_ISLNK(legacy_metadata.st_mode):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature released legacy runtime path is unsafe."
        )
    _validate_released_legacy_directory_metadata(legacy_metadata)
    if stable.exists() or stable.is_symlink():
        if stable.is_symlink() or not stable.is_dir():
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature stable runtime path is unsafe."
            )
        _log_runtime_migration_collision(
            legacy_component,
            stable_component,
            legacy_parent="released_feature_venvs",
        )
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature has both released legacy and stable runtime "
            "state; operator custody reconciliation is required."
        )
    _revalidate_portable_released_root(legacy_root, root_metadata)
    _ensure_venv_relocation_repair_marker_portable(legacy)
    try:
        legacy.rename(stable)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        _log_runtime_migration_collision(
            legacy_component,
            stable_component,
            legacy_parent="released_feature_venvs",
        )
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature has both released legacy and stable runtime "
            "state; operator custody reconciliation is required."
        ) from exc
    try:
        migrated = stable.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature released legacy runtime migration did not "
            "complete; tenant state was retained."
        ) from exc
    if not _same_file_identity(legacy_metadata, migrated):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature released legacy runtime changed during migration."
        )
    return True


def migrate_released_hosted_feature_runtime(
    legacy_root: Path,
    scope: IsolatedRuntimeNamespace,
    owner: str,
    legacy_component: str,
    stable_component: str,
) -> bool:
    """Adopt the shipped class-named hosted tree into its tenant namespace.

    The managed factory, rather than feature metadata or process environment,
    supplies ``legacy_root``.  Adoption renames the complete feature directory
    so venv contents and service data (including channel credentials) move as
    one filesystem object.  Cross-device moves fail with both custody trees
    untouched; collisions are never merged or overwritten.
    """

    if (
        not isinstance(legacy_root, Path)
        or _ISOLATED_FEATURE_CLASS_NAME.fullmatch(legacy_component) is None
        or _HOSTED_FEATURE_RUNTIME_COMPONENT.fullmatch(stable_component) is None
    ):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature released runtime migration is not canonical."
        )
    if not _secure_dirfd_supported():  # pragma: no cover - Windows fallback
        try:
            migrated = _migrate_released_runtime_directory_portable(
                legacy_root,
                scope,
                legacy_component,
                stable_component,
            )
        except IsolatedRuntimeNamespaceError:
            raise
        except OSError as exc:
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature released runtime could not be migrated; "
                "tenant state was retained."
            ) from exc
        return migrated

    source_parent_fd: Optional[int] = None
    target_root_fd: Optional[int] = None
    namespace_fd: Optional[int] = None
    target_parent_fd: Optional[int] = None
    try:
        try:
            source_parent_fd = os.open(
                legacy_root.parent,
                _directory_open_flags(),
            )
            lexical_parent = os.stat(
                legacy_root.parent,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(lexical_parent.st_mode) or not _same_file_identity(
                lexical_parent,
                os.fstat(source_parent_fd),
            ):
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature released legacy runtime parent "
                    "changed during validation."
                )
            try:
                source_root_metadata = os.stat(
                    legacy_root.name,
                    dir_fd=source_parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return False
            source_root_fd = os.open(
                legacy_root.name,
                _directory_open_flags(),
                dir_fd=source_parent_fd,
            )
        except IsolatedRuntimeNamespaceError:
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature released legacy runtime contains "
                    "an unsafe path entry."
                ) from exc
            raise

        try:
            if not _same_file_identity(source_root_metadata, os.fstat(source_root_fd)):
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature released legacy runtime root changed "
                    "during validation."
                )

            try:
                legacy_metadata = os.stat(
                    legacy_component,
                    dir_fd=source_root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return False
            _validate_released_legacy_root_custody(source_root_metadata)
            _validate_released_legacy_directory_metadata(legacy_metadata)
            legacy_fd = os.open(
                legacy_component,
                _directory_open_flags(),
                dir_fd=source_root_fd,
            )
            try:
                if not _same_file_identity(legacy_metadata, os.fstat(legacy_fd)):
                    raise IsolatedRuntimeNamespaceError(
                        "Hosted isolated feature released legacy runtime changed "
                        "during migration."
                    )
            finally:
                os.close(legacy_fd)

            target_root_fd = _open_secure_absolute_directory(scope.root)
            namespace_fd = os.dup(target_root_fd)
            for component in scope.namespace.parts:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=namespace_fd,
                )
                os.close(namespace_fd)
                namespace_fd = child
            _read_existing_runtime_owner(namespace_fd, owner)
            target_parent_fd = os.open(
                "feature_venvs",
                _directory_open_flags(),
                dir_fd=namespace_fd,
            )

            try:
                stable_metadata = os.stat(
                    stable_component,
                    dir_fd=target_parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                stable_metadata = None
            if stable_metadata is not None:
                if not stat.S_ISDIR(stable_metadata.st_mode):
                    raise IsolatedRuntimeNamespaceError(
                        "Hosted isolated feature stable runtime path is unsafe."
                    )
                _log_runtime_migration_collision(
                    legacy_component,
                    stable_component,
                    legacy_parent="released_feature_venvs",
                )
                raise IsolatedRuntimePreparationError(
                    "Hosted isolated feature has both released legacy and stable "
                    "runtime state; operator custody reconciliation is required."
                )

            legacy_fd = os.open(
                legacy_component,
                _directory_open_flags(),
                dir_fd=source_root_fd,
            )
            try:
                if not _same_file_identity(legacy_metadata, os.fstat(legacy_fd)):
                    raise IsolatedRuntimeNamespaceError(
                        "Hosted isolated feature released legacy runtime changed "
                        "during migration."
                    )
                _ensure_venv_relocation_repair_marker_at(legacy_fd)
            finally:
                os.close(legacy_fd)

            try:
                _rename_directory_noreplace_at(
                    source_root_fd,
                    legacy_component,
                    target_parent_fd,
                    stable_component,
                )
            except FileNotFoundError:
                try:
                    post_legacy = os.stat(
                        legacy_component,
                        dir_fd=source_root_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    post_legacy = None
                try:
                    post_stable = os.stat(
                        stable_component,
                        dir_fd=target_parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    post_stable = None
                if (
                    post_legacy is not None
                    or post_stable is None
                    or not _same_file_identity(legacy_metadata, post_stable)
                ):
                    raise IsolatedRuntimePreparationError(
                        "Hosted isolated feature released legacy runtime migration "
                        "did not complete; tenant state was retained."
                    ) from None
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                _log_runtime_migration_collision(
                    legacy_component,
                    stable_component,
                    legacy_parent="released_feature_venvs",
                )
                raise IsolatedRuntimePreparationError(
                    "Hosted isolated feature has both released legacy and stable "
                    "runtime state; operator custody reconciliation is required."
                ) from exc
            migrated_fd = os.open(
                stable_component,
                _directory_open_flags(),
                dir_fd=target_parent_fd,
            )
            try:
                if not _same_file_identity(legacy_metadata, os.fstat(migrated_fd)):
                    raise IsolatedRuntimeNamespaceError(
                        "Hosted isolated feature released legacy runtime changed "
                        "during migration."
                    )
            finally:
                os.close(migrated_fd)
            os.fsync(source_root_fd)
            os.fsync(target_parent_fd)
            return True
        finally:
            os.close(source_root_fd)
    except (IsolatedRuntimeNamespaceError, IsolatedRuntimePreparationError):
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature released runtime migration encountered "
                "an unsafe path entry."
            ) from exc
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature released runtime could not be migrated; "
            "tenant state was retained."
        ) from exc
    finally:
        if target_parent_fd is not None:
            os.close(target_parent_fd)
        if namespace_fd is not None:
            os.close(namespace_fd)
        if target_root_fd is not None:
            os.close(target_root_fd)
        if source_parent_fd is not None:
            os.close(source_parent_fd)


def _prepare_runtime_tree_portable(
    scope: IsolatedRuntimeNamespace,
    owner: str,
    relative_directories: tuple[tuple[str, ...], ...],
    directory_migrations: tuple[tuple[tuple[str, ...], str, str], ...],
    migration_results: Optional[set[tuple[tuple[str, ...], str, str]]],
) -> None:
    """Best-effort non-POSIX fallback where dirfd no-follow is unavailable."""

    created_root = False
    try:
        scope.root.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
        created_root = True
    except FileExistsError:
        pass
    if scope.root.is_symlink():
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime root must not be a symlink."
        )
    root_metadata = scope.root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime root must be a directory."
        )
    if created_root:
        scope.root.chmod(_PRIVATE_DIRECTORY_MODE)
    else:
        _validate_operator_root_metadata(root_metadata)
    cursor = scope.root
    for component in scope.namespace.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime namespace must not contain symlinks."
            )
        cursor.mkdir(exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
        cursor.chmod(_PRIVATE_DIRECTORY_MODE)
    marker = cursor / _RUNTIME_OWNER_MARKER
    owner_bytes = _runtime_owner_bytes(owner)

    def validate_existing_marker() -> None:
        if marker.is_symlink():
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime ownership marker is unsafe."
            )
        try:
            metadata = marker.stat(follow_symlinks=False)
            existing = marker.read_bytes()
        except FileNotFoundError as exc:
            raise _RuntimeOwnerMarkerMissing(
                "Hosted isolated feature runtime ownership marker is missing; "
                "refusing to assume tenant ownership."
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime ownership marker is unsafe."
            )
        if metadata.st_nlink != 1:
            for candidate in cursor.iterdir():
                if not candidate.name.startswith(_RUNTIME_OWNER_TEMP_PREFIX):
                    continue
                try:
                    candidate_metadata = candidate.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(candidate_metadata.st_mode) and _same_file_identity(
                    candidate_metadata, metadata
                ):
                    try:
                        candidate.unlink()
                    except FileNotFoundError:
                        pass
            if marker.stat(follow_symlinks=False).st_nlink != 1:
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature runtime ownership marker has an "
                    "unsafe external hard link."
                )
        if len(existing) != len(owner_bytes):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime ownership marker is corrupt; "
                "operator verification is required before recovery."
            )
        if not hmac.compare_digest(existing, owner_bytes):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime namespace is already owned by "
                "a different agent."
            )
        marker.chmod(_PRIVATE_FILE_MODE)

    try:
        validate_existing_marker()
    except _RuntimeOwnerMarkerMissing:
        temporary = cursor / f"{_RUNTIME_OWNER_TEMP_PREFIX}{os.getpid()}-{uuid4().hex}"
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _PRIVATE_FILE_MODE,
        )
        try:
            try:
                _write_all(temporary_fd, owner_bytes)
                os.fsync(temporary_fd)
            finally:
                os.close(temporary_fd)
            temporary.chmod(_PRIVATE_FILE_MODE)
            try:
                os.link(temporary, marker, follow_symlinks=False)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        validate_existing_marker()
    for parent, legacy_component, stable_component in directory_migrations:
        migration_parent = cursor.joinpath(*parent)
        migration_parent.mkdir(
            parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE
        )
        if migration_parent.is_symlink():
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime migration parent must not "
                "contain symlinks."
            )
        legacy = migration_parent / legacy_component
        stable = migration_parent / stable_component
        if not legacy.exists() and not legacy.is_symlink():
            continue
        if legacy.is_symlink() or not legacy.is_dir():
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature legacy runtime path is unsafe."
            )
        if stable.exists() or stable.is_symlink():
            if stable.is_symlink() or not stable.is_dir():
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature stable runtime path is unsafe."
                )
            _log_runtime_migration_collision(legacy_component, stable_component)
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature has both legacy and stable runtime "
                "state; operator custody reconciliation is required."
            )
        _ensure_venv_relocation_repair_marker_portable(legacy)
        legacy.rename(stable)
        if migration_results is not None:
            migration_results.add((parent, legacy_component, stable_component))
    for relative in relative_directories:
        cursor = scope.path
        for component in relative:
            cursor = cursor / component
            if cursor.is_symlink():
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature runtime workspace must not contain "
                    "symlinks."
                )
            cursor.mkdir(exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
            cursor.chmod(_PRIVATE_DIRECTORY_MODE)


def prepare_isolated_runtime_namespace(
    scope: IsolatedRuntimeNamespace,
    owner: str,
    *,
    relative_directories: tuple[tuple[str, ...], ...] = (),
    directory_migrations: tuple[tuple[tuple[str, ...], str, str], ...] = (),
    migration_results: Optional[set[tuple[tuple[str, ...], str, str]]] = None,
) -> Path:
    """Securely bind and prepare one hosted agent's mutable runtime tree.

    On POSIX, every creation/open is descriptor-relative and no-follow. The
    final inode/path bindings are rechecked before the descriptors are released,
    closing the validation-to-create symlink race for tenant-controlled path
    entries. The configured root's parent remains an operator custody boundary.
    """

    if type(owner) is not str or not owner:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime requires a concrete agent identity."
        )
    if any(
        not relative
        or any(
            type(component) is not str
            or not component
            or component in {".", ".."}
            or "/" in component
            or "\\" in component
            for component in relative
        )
        for relative in relative_directories
    ):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime workspace paths must be canonical "
            "relative components."
        )
    if any(
        not parent
        or any(
            type(component) is not str
            or not component
            or component in {".", ".."}
            or "/" in component
            or "\\" in component
            for component in parent
        )
        or _HOSTED_FEATURE_RUNTIME_COMPONENT.fullmatch(legacy_component) is None
        or _HOSTED_FEATURE_RUNTIME_COMPONENT.fullmatch(stable_component) is None
        for parent, legacy_component, stable_component in directory_migrations
    ):
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime migrations must use canonical "
            "relative components."
        )
    if not _secure_dirfd_supported():  # pragma: no cover - Windows fallback
        try:
            _prepare_runtime_tree_portable(
                scope,
                owner,
                relative_directories,
                directory_migrations,
                migration_results,
            )
        except IsolatedRuntimeNamespaceError:
            raise
        except OSError as exc:
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature runtime path could not be prepared."
            ) from exc
        return scope.path

    try:
        root_fd = _open_secure_absolute_directory(scope.root)
    except IsolatedRuntimeNamespaceError:
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime root contains an unsafe path "
                "entry."
            ) from exc
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature runtime root could not be prepared."
        ) from exc
    namespace_fd = root_fd
    owns_namespace_fd = False
    try:
        for component in scope.namespace.parts:
            child = _open_or_create_directory_at(namespace_fd, component)
            if owns_namespace_fd:
                os.close(namespace_fd)
            namespace_fd = child
            owns_namespace_fd = True
        try:
            current = os.stat(scope.path, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime namespace changed during "
                "validation."
            ) from exc
        if not stat.S_ISDIR(current.st_mode) or not _same_file_identity(
            current, os.fstat(namespace_fd)
        ):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime namespace changed during validation."
            )
        _read_or_create_runtime_owner(namespace_fd, owner)
        for parent, legacy_component, stable_component in directory_migrations:
            descriptor = os.dup(namespace_fd)
            try:
                for component in parent:
                    child = _open_or_create_directory_at(descriptor, component)
                    os.close(descriptor)
                    descriptor = child
                migrated = _migrate_runtime_directory_at(
                    descriptor,
                    legacy_component,
                    stable_component,
                )
                if migrated and migration_results is not None:
                    migration_results.add((parent, legacy_component, stable_component))
            finally:
                os.close(descriptor)
        for relative in relative_directories:
            descriptor = os.dup(namespace_fd)
            try:
                for component in relative:
                    child = _open_or_create_directory_at(descriptor, component)
                    os.close(descriptor)
                    descriptor = child
                path = scope.path.joinpath(*relative)
                try:
                    current = os.stat(path, follow_symlinks=False)
                except FileNotFoundError as exc:
                    raise IsolatedRuntimeNamespaceError(
                        "Hosted isolated feature runtime workspace changed during "
                        "validation."
                    ) from exc
                if not stat.S_ISDIR(current.st_mode) or not _same_file_identity(
                    current, os.fstat(descriptor)
                ):
                    raise IsolatedRuntimeNamespaceError(
                        "Hosted isolated feature runtime workspace changed during "
                        "validation."
                    )
            finally:
                os.close(descriptor)
        return scope.path
    except IsolatedRuntimeNamespaceError:
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime path contains an unsafe entry."
            ) from exc
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature runtime path could not be prepared."
        ) from exc
    finally:
        if owns_namespace_fd:
            os.close(namespace_fd)
        os.close(root_fd)


def _assert_no_nested_runtime_owners_at(
    directory_fd: int,
    *,
    allow_owner_marker: bool,
) -> None:
    """Refuse cleanup when another runtime namespace is nested below this one.

    Multi-component namespaces are valid, but allocated namespace leaves must
    remain prefix-free. This descriptor-relative preflight prevents deleting
    any sibling content before discovering that a descendant is independently
    bound to an agent. The deletion walk repeats the check to fail closed if a
    marker appears between preflight and mutation.
    """

    with os.scandir(directory_fd) as entries:
        names = [entry.name for entry in entries]
    if not allow_owner_marker and _RUNTIME_OWNER_MARKER in names:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime cleanup found a nested ownership "
            "marker; allocated namespaces must not contain one another."
        )
    for name in names:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
        try:
            if not _same_file_identity(metadata, os.fstat(child_fd)):
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature runtime cleanup target changed "
                    "during nested-owner validation."
                )
            _assert_no_nested_runtime_owners_at(
                child_fd,
                allow_owner_marker=False,
            )
        finally:
            os.close(child_fd)


def _remove_directory_contents_at(
    directory_fd: int,
    *,
    allow_owner_marker: bool,
) -> None:
    """Delete one already-open tree without following any directory symlink."""

    with os.scandir(directory_fd) as entries:
        names = [entry.name for entry in entries]
    if not allow_owner_marker and _RUNTIME_OWNER_MARKER in names:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime cleanup found a nested ownership "
            "marker; allocated namespaces must not contain one another."
        )
    for name in names:
        # The top-level marker is the custody proof needed to retry a partial
        # cleanup. Preserve it throughout the sweep regardless of filesystem
        # enumeration order. Nested markers remain a hard failure above.
        if allow_owner_marker and name == _RUNTIME_OWNER_MARKER:
            continue
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
            try:
                if not _same_file_identity(metadata, os.fstat(child_fd)):
                    raise IsolatedRuntimeNamespaceError(
                        "Hosted isolated feature runtime cleanup target changed "
                        "during validation."
                    )
                _remove_directory_contents_at(
                    child_fd,
                    allow_owner_marker=False,
                )
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            # Symlinks and special files are removed as directory entries; they
            # are never opened or traversed.
            os.unlink(name, dir_fd=directory_fd)


def _remove_isolated_feature_runtime(
    scope: IsolatedRuntimeNamespace,
    owner: str,
    feature_component: str,
) -> RuntimeNamespaceCleanupOutcome:
    """Securely remove one idle feature tree while retaining its agent scope."""

    if type(owner) is not str or not owner:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature cleanup requires an agent identity."
        )
    if _HOSTED_FEATURE_RUNTIME_COMPONENT.fullmatch(feature_component) is None:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature cleanup component is invalid."
        )
    if not _secure_dirfd_supported():  # pragma: no cover - non-POSIX policy
        raise IsolatedRuntimePreparationError(
            "Secure hosted feature cleanup is unavailable on this platform; "
            "runtime state was retained."
        )

    descriptors: list[int] = []
    try:
        current_fd = os.open(scope.root, _directory_open_flags())
        descriptors.append(current_fd)
        lexical_root = os.stat(scope.root, follow_symlinks=False)
        if not _same_file_identity(lexical_root, os.fstat(current_fd)):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature cleanup root changed during validation."
            )
        _validate_operator_root_metadata(os.fstat(current_fd))
        for component in scope.namespace.parts:
            try:
                current_fd = os.open(
                    component, _directory_open_flags(), dir_fd=current_fd
                )
            except FileNotFoundError:
                return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
            descriptors.append(current_fd)
        _read_existing_runtime_owner(current_fd, owner)
        try:
            feature_parent_fd = os.open(
                "feature_venvs", _directory_open_flags(), dir_fd=current_fd
            )
        except FileNotFoundError:
            return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
        descriptors.append(feature_parent_fd)
        try:
            feature_fd = os.open(
                feature_component,
                _directory_open_flags(),
                dir_fd=feature_parent_fd,
            )
        except FileNotFoundError:
            return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
        descriptors.append(feature_fd)
        lexical_feature = os.stat(
            scope.path / "feature_venvs" / feature_component,
            follow_symlinks=False,
        )
        if not _same_file_identity(lexical_feature, os.fstat(feature_fd)):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature cleanup target changed during validation."
            )
        _assert_no_nested_runtime_owners_at(feature_fd, allow_owner_marker=False)
        _remove_directory_contents_at(feature_fd, allow_owner_marker=False)
        os.rmdir(feature_component, dir_fd=feature_parent_fd)
        os.fsync(feature_parent_fd)
        return RuntimeNamespaceCleanupOutcome.REMOVED
    except (IsolatedRuntimeNamespaceError, IsolatedRuntimePreparationError):
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature cleanup encountered an unsafe path entry."
            ) from exc
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature cleanup could not complete; runtime state "
            "was retained."
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def remove_isolated_runtime_namespace(
    scope: IsolatedRuntimeNamespace,
    owner: str,
) -> RuntimeNamespaceCleanupOutcome:
    """Securely offboard one tenant runtime tree after its agent has stopped.

    Cleanup is deliberately POSIX-dirfd-only.  On a platform without the
    no-follow primitives required to bind every deletion to the verified
    namespace inode, Core retains the tree and reports the unsupported secure
    cleanup instead of falling back to a traversal-prone recursive delete.
    """

    if type(owner) is not str or not owner:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime cleanup requires an agent identity."
        )
    if not _secure_dirfd_supported():  # pragma: no cover - non-POSIX policy
        if scope.path.exists():
            raise IsolatedRuntimePreparationError(
                "Secure isolated-runtime offboarding is unavailable on this "
                "platform; tenant state was retained."
            )
        return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT

    try:
        root_fd = os.open(scope.root, _directory_open_flags())
    except FileNotFoundError:
        return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime cleanup root is unsafe."
            ) from exc
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature runtime cleanup root could not be opened."
        ) from exc

    current_fd = root_fd
    owns_current_fd = False
    parent_fd: Optional[int] = None
    namespace_fd: Optional[int] = None
    leaf = scope.namespace.parts[-1]
    try:
        root_metadata = os.fstat(root_fd)
        lexical_root = os.stat(scope.root, follow_symlinks=False)
        if not _same_file_identity(root_metadata, lexical_root):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime cleanup root changed during "
                "validation."
            )
        _validate_operator_root_metadata(root_metadata)

        for index, component in enumerate(scope.namespace.parts):
            try:
                child_fd = os.open(
                    component, _directory_open_flags(), dir_fd=current_fd
                )
            except FileNotFoundError:
                return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
            if index == len(scope.namespace.parts) - 1:
                parent_fd = current_fd
                namespace_fd = child_fd
                break
            if owns_current_fd:
                os.close(current_fd)
            current_fd = child_fd
            owns_current_fd = True

        assert parent_fd is not None and namespace_fd is not None
        try:
            lexical_namespace = os.stat(scope.path, follow_symlinks=False)
        except FileNotFoundError:
            return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
        if not stat.S_ISDIR(lexical_namespace.st_mode) or not _same_file_identity(
            lexical_namespace, os.fstat(namespace_fd)
        ):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime cleanup namespace changed during "
                "validation."
            )
        _read_existing_runtime_owner(namespace_fd, owner)
        _assert_no_nested_runtime_owners_at(
            namespace_fd,
            allow_owner_marker=True,
        )
        _remove_directory_contents_at(
            namespace_fd,
            allow_owner_marker=True,
        )
        # Revalidate custody after the mutable sweep, then remove its marker as
        # the final directory entry immediately before rmdir.  If the final
        # rmdir itself loses a race, recreate the same atomic marker through
        # the still-open directory descriptor so a later retry is not wedged.
        _read_existing_runtime_owner(namespace_fd, owner)
        try:
            os.unlink(_RUNTIME_OWNER_MARKER, dir_fd=namespace_fd)
        except FileNotFoundError as exc:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime ownership marker changed "
                "during cleanup."
            ) from exc
        try:
            os.rmdir(leaf, dir_fd=parent_fd)
        except FileNotFoundError:
            # A concurrent cleanup holding the same verified custody evidence
            # may have completed the final removal first.  The still-open
            # descriptor refers to the now-unlinked inode; attempting to
            # recreate a marker there would manufacture an unreachable file
            # and turn a successful deletion into a false retained outcome.
            pass
        except OSError as remove_exc:
            try:
                _read_or_create_runtime_owner(namespace_fd, owner)
            except Exception as restore_exc:
                # Do not replace the original deletion diagnosis with the
                # compensation failure.  Both are required for an operator to
                # understand a retained tree whose marker could not be restored.
                causes = ExceptionGroup(
                    "isolated runtime removal and ownership recovery failed",
                    [remove_exc, restore_exc],
                )
                raise IsolatedRuntimePreparationError(
                    "Hosted isolated feature runtime cleanup failed and its "
                    "ownership marker could not be restored; operator custody "
                    "reconciliation is required."
                ) from causes
            raise
        os.fsync(parent_fd)
        os.close(namespace_fd)
        namespace_fd = None
        return RuntimeNamespaceCleanupOutcome.REMOVED
    except IsolatedRuntimeNamespaceError:
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature runtime cleanup encountered an unsafe "
                "path entry."
            ) from exc
        raise IsolatedRuntimePreparationError(
            "Hosted isolated feature runtime cleanup could not complete; tenant "
            "state was retained."
        ) from exc
    finally:
        if namespace_fd is not None:
            os.close(namespace_fd)
        if owns_current_fd:
            os.close(current_fd)
        os.close(root_fd)


def _runtime_scope_value(agent: Any, attribute: str) -> Optional[str | Path]:
    """Read one normalized agent path attribute without mistaking a Mock for one.

    ``KestrelAgent`` canonicalizes general ``os.PathLike`` constructor inputs to
    ``Path`` before exposing them. Restricting this duck-typed compatibility
    boundary to concrete strings/pathlib paths also prevents a MagicMock's
    synthetic ``__fspath__`` from declaring a hosted scope in endpoint tests.
    """
    value = getattr(agent, attribute, None)
    if type(value) is str or isinstance(value, Path):
        return value
    return None


def _agent_runtime_scope(agent: Any) -> Optional[IsolatedRuntimeNamespace]:
    declared = getattr(agent, "isolated_runtime_scope", None)
    if isinstance(declared, IsolatedRuntimeNamespace):
        return declared
    runtime_root = _runtime_scope_value(agent, "isolated_runtime_root")
    runtime_namespace = _runtime_scope_value(agent, "isolated_runtime_namespace")
    if runtime_root is not None or runtime_namespace is not None:
        return resolve_isolated_runtime_namespace(runtime_root, runtime_namespace)
    return None


def _agent_released_legacy_runtime_root(agent: Any) -> Optional[Path]:
    """Return only a factory-validated released-layout migration source."""

    value = _runtime_scope_value(agent, "isolated_runtime_legacy_root")
    return value if isinstance(value, Path) else None


def _agent_runtime_owner(agent: Any) -> str:
    for attribute in ("did", "agent_id"):
        value = getattr(agent, attribute, None)
        if type(value) is str and value:
            return value
    raise IsolatedRuntimeNamespaceError(
        "Hosted isolated feature runtime requires a concrete agent DID."
    )


def resolve_agent_runtime_dir(agent: Any) -> Path:
    """Resolve an agent runtime path without creating or changing the filesystem.

    Standalone agents keep the existing storage-derived layout.  Hosted agents
    must opt into an explicit root plus relative namespace; they never fall back
    to the process CWD because that would collapse tenants onto one directory.
    """
    runtime_scope = _agent_runtime_scope(agent)
    if runtime_scope is not None:
        return runtime_scope.path

    hosted = getattr(agent, "isolated_runtime_hosted", False)
    if isinstance(hosted, bool) and hosted:
        legacy_data_dir = _runtime_scope_value(agent, "isolated_feature_data_dir")
        migration = (
            " The legacy isolated_feature_data_dir value is not a containment "
            "boundary; pass isolated_runtime_root and "
            "isolated_runtime_namespace instead."
            if legacy_data_dir is not None
            else ""
        )
        raise IsolatedRuntimeNamespaceError(
            "Hosted agent has an isolated feature but no explicit runtime "
            f"root and namespace.{migration}"
        )

    isolated_feature_data_dir = _runtime_scope_value(
        agent, "isolated_feature_data_dir"
    )
    if isolated_feature_data_dir is not None:
        return Path(isolated_feature_data_dir).expanduser().resolve()
    storage_path = getattr(agent, "storage_path", None)
    if isinstance(storage_path, (str, os.PathLike)) and storage_path:
        return Path(storage_path).expanduser().resolve().parent
    return (Path.cwd() / "agent_data" / "default").resolve()


def agent_runtime_dir(agent: Any) -> Path:
    """Prepare and return the agent-owned mutable isolated-feature root."""

    runtime_scope = _agent_runtime_scope(agent)
    if runtime_scope is not None:
        return prepare_isolated_runtime_namespace(
            runtime_scope, _agent_runtime_owner(agent)
        )
    return resolve_agent_runtime_dir(agent)


def remove_released_legacy_runtime_root(
    legacy_root: Path,
) -> RuntimeNamespaceCleanupOutcome:
    """Securely remove the released per-agent ``feature_venvs`` tree.

    The released layout has no ownership marker, so custody is intentionally
    narrower than migration: the factory-provided path must be the canonical
    ``feature_venvs`` leaf, its parent and leaf must remain descriptor-bound,
    and the populated leaf must be service-owned and non-writable by group or
    world.  Ambiguous custody is retained and reported; it is never mistaken
    for successful deprovisioning of the new namespace.
    """

    if (
        not isinstance(legacy_root, Path)
        or not legacy_root.is_absolute()
        or legacy_root.name != "feature_venvs"
    ):
        raise IsolatedRuntimeNamespaceError(
            "Hosted released runtime cleanup path is not canonical."
        )
    if not _secure_dirfd_supported():  # pragma: no cover - non-POSIX policy
        if legacy_root.exists() or legacy_root.is_symlink():
            raise IsolatedRuntimePreparationError(
                "Secure released-runtime offboarding is unavailable on this "
                "platform; tenant state was retained."
            )
        return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT

    parent_fd: Optional[int] = None
    legacy_fd: Optional[int] = None
    try:
        parent_fd = os.open(legacy_root.parent, _directory_open_flags())
        lexical_parent = os.stat(legacy_root.parent, follow_symlinks=False)
        if not stat.S_ISDIR(lexical_parent.st_mode) or not _same_file_identity(
            lexical_parent,
            os.fstat(parent_fd),
        ):
            raise IsolatedRuntimeNamespaceError(
                "Hosted released runtime cleanup parent changed during validation."
            )
        try:
            lexical_root = os.stat(
                legacy_root.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
        legacy_fd = os.open(
            legacy_root.name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
        opened_root = os.fstat(legacy_fd)
        if not stat.S_ISDIR(lexical_root.st_mode) or not _same_file_identity(
            lexical_root,
            opened_root,
        ):
            raise IsolatedRuntimeNamespaceError(
                "Hosted released runtime cleanup root changed during validation."
            )
        _validate_released_legacy_root_custody(opened_root)
        _assert_no_nested_runtime_owners_at(
            legacy_fd,
            allow_owner_marker=False,
        )
        _remove_directory_contents_at(
            legacy_fd,
            allow_owner_marker=False,
        )
        if not _same_file_identity(opened_root, os.fstat(legacy_fd)):
            raise IsolatedRuntimeNamespaceError(
                "Hosted released runtime cleanup root changed during removal."
            )
        try:
            os.rmdir(legacy_root.name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.fsync(parent_fd)
        return RuntimeNamespaceCleanupOutcome.REMOVED
    except (IsolatedRuntimeNamespaceError, IsolatedRuntimePreparationError):
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Hosted released runtime cleanup encountered an unsafe path entry."
            ) from exc
        raise IsolatedRuntimePreparationError(
            "Hosted released runtime cleanup could not complete; tenant state "
            "was retained."
        ) from exc
    finally:
        if legacy_fd is not None:
            os.close(legacy_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def remove_runtime_namespace(
    scope: Optional[IsolatedRuntimeNamespace],
    owner: Optional[str],
    released_legacy_root: Optional[Path] = None,
) -> RuntimeNamespaceCleanupOutcome:
    """Delete all known hosted runtime layouts with exact custody semantics."""

    if scope is None:
        return RuntimeNamespaceCleanupOutcome.NOT_HOSTED
    if type(owner) is not str or not owner:
        raise IsolatedRuntimeNamespaceError(
            "Hosted isolated feature runtime cleanup requires an agent identity."
        )
    current_outcome = remove_isolated_runtime_namespace(scope, owner)
    if released_legacy_root is None:
        return current_outcome
    legacy_outcome = remove_released_legacy_runtime_root(released_legacy_root)
    if RuntimeNamespaceCleanupOutcome.REMOVED in {
        current_outcome,
        legacy_outcome,
    }:
        return RuntimeNamespaceCleanupOutcome.REMOVED
    return RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT


def remove_agent_runtime_namespace(agent: Any) -> RuntimeNamespaceCleanupOutcome:
    """Remove current and released hosted state, or report storage custody."""

    runtime_scope = _agent_runtime_scope(agent)
    owner = _agent_runtime_owner(agent) if runtime_scope is not None else None
    legacy_root = (
        _agent_released_legacy_runtime_root(agent)
        if runtime_scope is not None
        else None
    )
    return remove_runtime_namespace(runtime_scope, owner, legacy_root)


def _venv_bin_dir(venv_path: Path) -> Path:
    return venv_path / ("Scripts" if os.name == "nt" else "bin")


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _console_script_path(venv_path: Path, service: str) -> Path:
    """Return the exact executable path used for a console entry point."""

    executable = _validated_isolated_service_executable(service)
    if os.name == "nt" and not executable.casefold().endswith(".exe"):
        executable = f"{executable}.exe"
    return _venv_bin_dir(venv_path) / executable


# Interpreter-behavior env vars that would let the HOST Python installation
# shadow the isolated venv's packages, defeating the isolation the runtime
# exists for (F023). Standalone feature config/secrets ride through the general
# environment for backwards compatibility, so only these interpreter variables
# are stripped there; hosted mode uses the restrictive allowlist below.
_SHADOWING_ENV_VARS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV")

# Hosted services must not inherit a co-hosted agent's process-wide credentials
# or feature configuration. These are the only host variables a child needs to
# execute; agent-specific configuration (including secrets) travels through the
# isolated-feature initialize handshake instead.
_HOSTED_CHILD_ENV_BASE_KEYS = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)

# Package resolution is the only provisioning-specific authority inherited by
# hosted uv/build subprocesses.  URLs may carry private-index credentials, and
# named uv indexes conventionally use UV_INDEX_<NAME>_{USERNAME,PASSWORD};
# those values are intentionally available only during provisioning, never to
# the long-lived feature service or read-only venv probes.  Config-file,
# keyring, SSH-agent, cloud, API, channel, and tenant variables are excluded.
_HOSTED_PROVISIONING_PACKAGE_ENV_KEYS = (
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_TRUSTED_HOST",
    "UV_DEFAULT_INDEX",
    "UV_INDEX",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX_STRATEGY",
    "UV_INSECURE_HOST",
    "UV_NATIVE_TLS",
)
_HOSTED_PROVISIONING_NAMED_INDEX_CREDENTIAL = re.compile(
    r"^UV_INDEX_[A-Z0-9_]+_(?:USERNAME|PASSWORD)$"
)


def _isolated_provisioning_env(
    venv_path: Optional[Path],
    *,
    include_package_index: bool = False,
    uv_cache_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Build a narrow environment for hosted probes and provisioning.

    A feature venv can execute ``sitecustomize`` during a probe and arbitrary
    build-backend code during installation.  Both therefore start from the
    same host execution/locale/CA/proxy allowlist.  Only the installation path
    opts into explicitly documented package-index settings and credentials.
    """

    source = os.environ
    env = {key: source[key] for key in _HOSTED_CHILD_ENV_BASE_KEYS if key in source}
    if include_package_index:
        if uv_cache_dir is None:
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature provisioning requires an explicit "
                "private uv cache."
            )
        env.update(
            {
                key: source[key]
                for key in _HOSTED_PROVISIONING_PACKAGE_ENV_KEYS
                if key in source
            }
        )
        env.update(
            {
                key: value
                for key, value in source.items()
                if _HOSTED_PROVISIONING_NAMED_INDEX_CREDENTIAL.fullmatch(key)
            }
        )
        # Do not let uv discover an operator/home/cwd config file that was not
        # explicitly admitted above.
        env["UV_NO_CONFIG"] = "1"
        env["UV_CACHE_DIR"] = str(uv_cache_dir)
    elif uv_cache_dir is not None:
        raise ValueError("uv_cache_dir is only valid for package provisioning")
    if venv_path is not None:
        env["VIRTUAL_ENV"] = str(venv_path)
        bin_dir = str(_venv_bin_dir(venv_path))
        env["PATH"] = os.pathsep.join([bin_dir, env.get("PATH", "")]).rstrip(os.pathsep)
    return env


def _trusted_host_executable(
    executable: str,
    *,
    excluded_venv: Path,
) -> tuple[str, str]:
    """Resolve an operator executable without consulting feature-owned bins.

    Package-index credentials are visible to the provisioning process and its
    build backend.  The command which receives them must therefore be resolved
    from the host PATH after removing every entry inside the mutable feature
    venv.  Both PATH directories and the selected executable are resolved to
    defeat symlink aliases back into that venv.

    Returns the trusted absolute executable and the filtered absolute PATH used
    for selection.  The latter is also given to the subprocess so a child
    cannot resolve a helper through the excluded feature bin.
    """

    if (
        type(executable) is not str
        or not executable
        or Path(executable).name != executable
    ):
        raise RuntimeError("Hosted provisioning executable must be a bare name")

    excluded = excluded_venv.expanduser().resolve(strict=False)
    trusted_entries: list[str] = []
    for raw_entry in os.environ.get("PATH", "").split(os.pathsep):
        # An empty PATH entry means the process cwd.  It is mutable application
        # state, not an operator executable directory, so never admit it here.
        if not raw_entry:
            continue
        unresolved_entry = Path(raw_entry).expanduser()
        if not unresolved_entry.is_absolute():
            continue
        try:
            resolved_entry = unresolved_entry.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved_entry == excluded or excluded in resolved_entry.parents:
            continue
        trusted_entries.append(str(resolved_entry))

    if not trusted_entries:
        raise RuntimeError(f"Required executable not found: {executable}")
    trusted_path = os.pathsep.join(trusted_entries)
    selected = shutil.which(executable, path=trusted_path)
    if selected is None:
        raise RuntimeError(f"Required executable not found: {executable}")
    try:
        selected_path = Path(selected).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Required executable could not be resolved safely: {executable}"
        ) from exc
    if selected_path == excluded or excluded in selected_path.parents:
        raise RuntimeError(
            "Hosted provisioning executable resolved inside the feature venv"
        )
    return str(selected_path), trusted_path


def _unsafe_hosted_feature_env_keys(
    feature_name: Optional[str],
    feature_distribution: Optional[str],
) -> tuple[str, ...]:
    """Return legacy process-wide config that cannot be assigned to a tenant."""

    identity = f"{feature_name or ''} {feature_distribution or ''}".casefold()
    relevant_legacy_keys: set[str] = set()
    if "telegram" in identity:
        relevant_legacy_keys.update(
            {"KESTREL_TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN"}
        )
    if "whatsapp" in identity or "twilio" in identity:
        relevant_legacy_keys.update(
            {
                "KESTREL_WHATSAPP_PROVIDER",
                "KESTREL_WHATSAPP_SESSION_DB",
                "TWILIO_ACCOUNT_SID",
                "TWILIO_AUTH_TOKEN",
                "TWILIO_WHATSAPP_FROM",
            }
        )
    unsafe = {key for key in relevant_legacy_keys if key in os.environ}
    if feature_name:
        prefix = _env_key(feature_name, "")
        host_only = {_env_key(feature_name, "BIN"), _env_key(feature_name, "VENV")}
        unsafe.update(
            key for key in os.environ if key.startswith(prefix) and key not in host_only
        )
    return tuple(sorted(unsafe))


def _assert_hosted_feature_env_is_scoped(
    feature_name: Optional[str],
    feature_distribution: Optional[str],
) -> None:
    unsafe_keys = _unsafe_hosted_feature_env_keys(feature_name, feature_distribution)
    if not unsafe_keys:
        return
    raise IsolatedRuntimeConfigurationError(
        reason=_CONFIGURATION_UNSAFE_PROCESS_ENVIRONMENT,
        environment_keys=unsafe_keys,
    )


def _hosted_prebuilt_override_error(key: str) -> IsolatedRuntimeConfigurationError:
    return IsolatedRuntimeConfigurationError(
        reason=_CONFIGURATION_HOSTED_PREBUILT_OVERRIDE,
        environment_keys=(key,),
    )


def _hosted_immutable_metadata_is_unsafe(metadata: os.stat_result) -> bool:
    """Return whether an immutable artifact has unsafe POSIX custody/mode."""

    return os.name == "posix" and (
        metadata.st_uid not in {0, os.geteuid()}
        or bool(stat.S_IMODE(metadata.st_mode) & 0o022)
    )


@dataclass(frozen=True)
class _ValidatedHostedPrebuiltVenv:
    """Canonical root and bin directory proven safe for immutable use."""

    root_path: Path
    bin_path: Path


def _validate_hosted_prebuilt_venv(
    value: str,
    *,
    setting: str,
    require_absolute: bool,
) -> _ValidatedHostedPrebuiltVenv:
    """Resolve and validate one immutable hosted venv selection.

    ``runtime.venv`` comes from installed package metadata.  Unlike a
    standalone declaration, a relative value would resolve against Core's
    process CWD and become one shared mutable location for every tenant, so it
    is never a valid hosted selection.  Environment overrides retain their
    established absolute-normalization behavior but are subject to the same
    immutable artifact checks. Operator-facing venv and interpreter symlinks
    remain supported: every chain is resolved once and the canonical venv root,
    configuration target, bin directory, and interpreter target are validated.
    """

    try:
        candidate = Path(value).expanduser()
        if require_absolute and not candidate.is_absolute():
            raise ValueError("hosted runtime venv must be absolute")
        venv_path = Path(os.path.abspath(candidate)).resolve(strict=True)
        venv_metadata = venv_path.stat(follow_symlinks=False)
        manifest = venv_path / ".kestrel_provision.json"
        try:
            manifest.stat(follow_symlinks=False)
        except FileNotFoundError:
            manifest_present = False
        else:
            manifest_present = True
        config_path = (venv_path / "pyvenv.cfg").resolve(strict=True)
        config_metadata = config_path.stat(follow_symlinks=False)
        bin_path = _venv_bin_dir(venv_path).resolve(strict=True)
        bin_metadata = bin_path.stat(follow_symlinks=False)
        python_path = _venv_python(venv_path).resolve(strict=True)
        python_metadata = python_path.stat(follow_symlinks=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _hosted_prebuilt_override_error(setting) from None
    if (
        not stat.S_ISDIR(venv_metadata.st_mode)
        or _hosted_immutable_metadata_is_unsafe(venv_metadata)
        or manifest_present
        or not stat.S_ISREG(config_metadata.st_mode)
        or _hosted_immutable_metadata_is_unsafe(config_metadata)
        or not stat.S_ISDIR(bin_metadata.st_mode)
        or _hosted_immutable_metadata_is_unsafe(bin_metadata)
        or not stat.S_ISREG(python_metadata.st_mode)
        or _hosted_immutable_metadata_is_unsafe(python_metadata)
        or (os.name == "posix" and not os.access(python_path, os.X_OK))
    ):
        raise _hosted_prebuilt_override_error(setting)
    return _ValidatedHostedPrebuiltVenv(
        root_path=venv_path,
        bin_path=bin_path,
    )


def _validate_hosted_prebuilt_executable(
    value: str,
    *,
    setting: str,
    containment_root: Optional[Path] = None,
) -> Path:
    """Pin one executable after optional canonical containment validation.

    The returned canonical target is the only path the caller may publish or
    launch. A later replacement of the operator-facing symlink therefore
    cannot redirect an already-resolved feature to a file Core did not inspect.
    Root-owned system artifacts and service-uid-owned artifacts are accepted;
    mutable group/world-writable targets are not.
    """

    try:
        candidate = Path(value).expanduser()
        absolute = Path(os.path.abspath(candidate))
        resolved = absolute.resolve(strict=True)
        if (
            containment_root is not None
            and containment_root not in resolved.parents
        ):
            raise ValueError("hosted prebuilt executable escaped containment")
        metadata = resolved.stat(follow_symlinks=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _hosted_prebuilt_override_error(setting) from None
    unsafe_posix_custody = _hosted_immutable_metadata_is_unsafe(metadata) or (
        os.name == "posix" and not os.access(resolved, os.X_OK)
    )
    if not stat.S_ISREG(metadata.st_mode) or unsafe_posix_custody:
        raise _hosted_prebuilt_override_error(setting)
    return resolved


def _validate_hosted_prebuilt_bin(value: str, *, setting: str) -> Path:
    """Pin one hosted BIN symlink chain to its immutable executable target."""

    return _validate_hosted_prebuilt_executable(value, setting=setting)


def _validate_hosted_prebuilt_console(
    venv_path: Path,
    venv_bin_path: Path,
    service: str,
    *,
    setting: str,
) -> Path:
    """Pin one immutable console target contained by its validated venv bin."""

    script_name = _console_script_path(venv_path, service).name
    return _validate_hosted_prebuilt_executable(
        str(venv_bin_path / script_name),
        setting=setting,
        containment_root=venv_bin_path,
    )


@dataclass(frozen=True)
class _ValidatedHostedPrebuiltOverrides:
    """Canonical immutable artifacts selected by hosted operator settings."""

    venv_path: Optional[Path]
    venv_bin_path: Optional[Path]
    bin_path: Optional[Path]


def _validate_hosted_process_prebuilt_overrides(
    feature_name: str,
    *,
    runtime_venv: Optional[str] = None,
) -> _ValidatedHostedPrebuiltOverrides:
    """Accept only existing immutable-shape hosted launch overrides.

    Process variables and installed ``runtime.venv`` metadata are host
    configuration rather than tenant config, so they may be shared only as
    prebuilt artifacts. Core must never create, upgrade, or stamp either path:
    doing so lets concurrent hosted agents race over mutable provisioning
    state.
    """

    venv_key = _env_key(feature_name, "VENV")
    venv_value = os.environ.get(venv_key)
    validated_process_venv: Optional[_ValidatedHostedPrebuiltVenv] = None
    if venv_value:
        validated_process_venv = _validate_hosted_prebuilt_venv(
            venv_value,
            setting=venv_key,
            require_absolute=False,
        )

    validated_runtime_venv: Optional[_ValidatedHostedPrebuiltVenv] = None
    if runtime_venv is not None:
        if type(runtime_venv) is not str or not runtime_venv:
            raise _hosted_prebuilt_override_error(_HOSTED_RUNTIME_VENV_SETTING)
        validated_runtime_venv = _validate_hosted_prebuilt_venv(
            runtime_venv,
            setting=_HOSTED_RUNTIME_VENV_SETTING,
            require_absolute=True,
        )
    selected_venv = validated_process_venv or validated_runtime_venv
    selected_venv_path = (
        selected_venv.root_path if selected_venv is not None else None
    )
    selected_venv_bin_path = (
        selected_venv.bin_path if selected_venv is not None else None
    )

    bin_key = _env_key(feature_name, "BIN")
    bin_value = os.environ.get(bin_key)
    if bin_value:
        return _ValidatedHostedPrebuiltOverrides(
            venv_path=selected_venv_path,
            venv_bin_path=selected_venv_bin_path,
            bin_path=_validate_hosted_prebuilt_bin(bin_value, setting=bin_key),
        )
    return _ValidatedHostedPrebuiltOverrides(
        venv_path=selected_venv_path,
        venv_bin_path=selected_venv_bin_path,
        bin_path=None,
    )


def _isolated_child_env(
    venv_path: Optional[Path],
    *,
    runtime_dir: Optional[Path] = None,
    hosted: bool = False,
    feature_name: Optional[str] = None,
    feature_distribution: Optional[str] = None,
) -> Dict[str, str]:
    """Build the launch environment for the isolated service subprocess.

    Standalone services inherit feature variables for backwards compatibility.
    Hosted services deliberately drop those process-wide variables and receive
    agent-scoped configuration through their initialize handshake. In both
    modes interpreter-behavior vars in ``_SHADOWING_ENV_VARS`` are stripped so
    a stray host ``PYTHONPATH``/``VIRTUAL_ENV`` cannot defeat venv isolation.
    Hosted runtime workspaces also own child homes, cache, data, and temp
    paths. Standalone launches preserve their historical process environment;
    their feature packages may already rely on HOME, TMPDIR, XDG, or legacy
    Kestrel data-directory precedence.
    """
    env = dict(os.environ)
    for var in _SHADOWING_ENV_VARS:
        env.pop(var, None)
    if hosted:
        _assert_hosted_feature_env_is_scoped(feature_name, feature_distribution)
        # A process environment is host scoped, so inherited variables cannot
        # safely represent a tenant's configuration or credentials. Keep only
        # execution/locale/CA variables; feature config and secrets arrive in
        # the per-agent initialize handshake.
        env = {key: env[key] for key in _HOSTED_CHILD_ENV_BASE_KEYS if key in env}
    if venv_path is not None:
        env["VIRTUAL_ENV"] = str(venv_path)
        bin_dir = str(_venv_bin_dir(venv_path))
        env["PATH"] = os.pathsep.join([bin_dir, env.get("PATH", "")]).rstrip(os.pathsep)
    if hosted and runtime_dir is not None:
        # These are location contracts, not capabilities. Hosted feature code
        # runs under Core's service UID unless the operator supplies a real
        # container/UID sandbox; POSIX 0700 cannot isolate hostile same-UID
        # siblings. Expose only this child's required workspace paths, never
        # the configured operator root or a sibling namespace.
        # Child libraries commonly write relative files, XDG state, or temp
        # files. Point all of those at the feature's agent-owned workspace.
        # ``KESTREL_ISOLATED_RUNTIME_DIR`` is Kestrel's generic canonical
        # variable. ``KESTREL_ISOLATED_FEATURE_DATA_DIR`` is the established
        # isolated-service contract (including kestrel-channel-whatsapp), so
        # export both while feature packages migrate to the canonical spelling.
        env["KESTREL_ISOLATED_RUNTIME_DIR"] = str(runtime_dir)
        env["KESTREL_ISOLATED_FEATURE_DATA_DIR"] = str(runtime_dir)
        env["HOME"] = str(runtime_dir / "home")
        env["TMPDIR"] = str(runtime_dir / "tmp")
        env["TMP"] = str(runtime_dir / "tmp")
        env["TEMP"] = str(runtime_dir / "tmp")
        env["USERPROFILE"] = str(runtime_dir / "home")
        env["APPDATA"] = str(runtime_dir / "config")
        env["LOCALAPPDATA"] = str(runtime_dir / "data")
        env["XDG_CONFIG_HOME"] = str(runtime_dir / "config")
        env["XDG_DATA_HOME"] = str(runtime_dir / "data")
        env["XDG_CACHE_HOME"] = str(runtime_dir / "cache")
    return env


_MAX_PRIVATE_ARTIFACT_BYTES = 4 * 1024 * 1024
_ASCII_BASE64_WHITESPACE = b" \t\n\r\v\f"


def _write_private_artifact(path: Path, payload: bytes) -> None:
    """Atomically replace one file in an existing, no-follow runtime directory."""

    if len(payload) > _MAX_PRIVATE_ARTIFACT_BYTES:
        raise IsolatedRuntimeNamespaceError(
            "Agent runtime artifact exceeds the safe write limit."
        )

    if not _secure_dirfd_supported():  # pragma: no cover - Windows fallback
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise IsolatedRuntimeNamespaceError(
                "Agent runtime artifact directory is unavailable or unsafe."
            )
        temporary_path = path.parent / (f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _PRIVATE_FILE_MODE,
        )
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temporary_path.chmod(_PRIVATE_FILE_MODE)
        try:
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return

    directory_fd = os.open(path.parent, _directory_open_flags())
    temporary = f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    temporary_fd: Optional[int] = None
    try:
        lexical = os.stat(path.parent, follow_symlinks=False)
        if not stat.S_ISDIR(lexical.st_mode) or not _same_file_identity(
            lexical, os.fstat(directory_fd)
        ):
            raise IsolatedRuntimeNamespaceError(
                "Agent runtime artifact directory changed during validation."
            )
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            _PRIVATE_FILE_MODE,
            dir_fd=directory_fd,
        )
        _write_all(temporary_fd, payload)
        os.fchmod(temporary_fd, _PRIVATE_FILE_MODE)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        os.replace(
            temporary,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        finally:
            # Cleanup failure must never strand the directory descriptor.
            os.close(directory_fd)


def _read_bounded_descriptor(descriptor: int, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise IsolatedRuntimeNamespaceError(
                "Agent runtime artifact exceeds the safe read limit."
            )


def _validate_private_artifact_metadata(
    metadata: os.stat_result, *, max_bytes: int
) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise IsolatedRuntimeNamespaceError(
            "Agent runtime artifact is not a private regular file."
        )
    if metadata.st_size < 0 or metadata.st_size > max_bytes:
        raise IsolatedRuntimeNamespaceError(
            "Agent runtime artifact exceeds the safe read limit."
        )


def read_private_artifact(
    path: Path, *, max_bytes: int = _MAX_PRIVATE_ARTIFACT_BYTES
) -> Optional[bytes]:
    """Read one bounded artifact without following a tenant-controlled link.

    This function is intentionally synchronous: HTTP/event callers must run
    the entire descriptor-relative open and read in a worker thread. ``None``
    means the artifact or its parent is absent. Unsafe path entries and file
    types fail closed with :class:`IsolatedRuntimeNamespaceError`.
    """

    path = Path(path)
    if max_bytes < 0 or path.name in {"", ".", ".."}:
        raise IsolatedRuntimeNamespaceError(
            "Agent runtime artifact path is invalid."
        )

    if not _secure_dirfd_supported():  # pragma: no cover - Windows fallback
        try:
            parent_metadata = path.parent.stat(follow_symlinks=False)
            entry_metadata = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise IsolatedRuntimeNamespaceError(
                "Agent runtime artifact directory is unavailable or unsafe."
            )
        _validate_private_artifact_metadata(entry_metadata, max_bytes=max_bytes)
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
        except FileNotFoundError:
            return None
        try:
            opened = os.fstat(descriptor)
            _validate_private_artifact_metadata(opened, max_bytes=max_bytes)
            if not _same_file_identity(entry_metadata, opened):
                raise IsolatedRuntimeNamespaceError(
                    "Agent runtime artifact changed during validation."
                )
            return _read_bounded_descriptor(descriptor, max_bytes)
        finally:
            os.close(descriptor)

    try:
        directory_fd = os.open(path.parent, _directory_open_flags())
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise IsolatedRuntimeNamespaceError(
                "Agent runtime artifact directory is unavailable or unsafe."
            ) from exc
        raise IsolatedRuntimePreparationError(
            "Agent runtime artifact directory could not be opened."
        ) from exc
    try:
        try:
            lexical_parent = os.stat(path.parent, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(lexical_parent.st_mode) or not _same_file_identity(
            lexical_parent, os.fstat(directory_fd)
        ):
            raise IsolatedRuntimeNamespaceError(
                "Agent runtime artifact directory changed during validation."
            )
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise IsolatedRuntimeNamespaceError(
                    "Agent runtime artifact is unavailable or unsafe."
                ) from exc
            raise IsolatedRuntimePreparationError(
                "Agent runtime artifact could not be opened."
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            _validate_private_artifact_metadata(metadata, max_bytes=max_bytes)
            return _read_bounded_descriptor(descriptor, max_bytes)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)


def _unlink_private_artifact(path: Path) -> None:
    """Unlink one artifact relative to its existing no-follow directory."""

    if not _secure_dirfd_supported():  # pragma: no cover - Windows fallback
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise IsolatedRuntimeNamespaceError(
                "Agent runtime artifact directory is unavailable or unsafe."
            )
        path.unlink(missing_ok=True)
        return

    directory_fd = os.open(path.parent, _directory_open_flags())
    try:
        os.unlink(path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except FileNotFoundError:
        pass
    finally:
        os.close(directory_fd)


def _coerce_category(value: Any) -> ToolCategory:
    if isinstance(value, ToolCategory):
        return value
    if value is None:
        return ToolCategory.SYSTEM
    text = str(value)
    for category in ToolCategory:
        if text == category.value or text.upper() == category.name:
            return category
    return ToolCategory.SYSTEM


def _meta_get(metadata: Any, key: str, default: Any = None) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _input_schema_to_parameters(input_schema: Any) -> List[ToolParameter]:
    """Convert an isolated service's advertised JSON-Schema ``input_schema``
    (the SDK wire contract, ``ToolMetadata.input_schema``) into the
    ``List[ToolParameter]`` a host ``ToolSchema`` expects.

    Without this the proxied tool reaches the LLM with an empty parameter list,
    so the model cannot supply arguments (F004). Passing the raw dict through is
    also wrong: ``ToolSchema.to_openai_format`` iterates ``ToolParameter``
    objects and would crash on a dict/string.
    """
    if not isinstance(input_schema, dict):
        return []
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return []
    raw_required = input_schema.get("required")
    required = set(raw_required) if isinstance(raw_required, (list, tuple, set)) else set()

    params: List[ToolParameter] = []
    for pname, pdef in properties.items():
        pdef = pdef if isinstance(pdef, dict) else {}
        ptype = pdef.get("type")
        if isinstance(ptype, (list, tuple)):
            # JSON-Schema union, e.g. ["string", "null"] for an Optional — take
            # the first non-null member.
            ptype = next((t for t in ptype if t != "null"), None)
        params.append(
            ToolParameter(
                name=str(pname),
                type=str(ptype or "string"),
                description=str(pdef.get("description", "")),
                required=pname in required,
                default=pdef.get("default"),
                enum=pdef.get("enum"),
                items=pdef.get("items"),
            )
        )
    return params


def _maybe_await(value: Any) -> Awaitable[Any]:
    if inspect.isawaitable(value):
        return value

    async def _wrapped() -> Any:
        return value

    return _wrapped()


class IsolatedFeatureTool(AgentTool):
    """Tool wrapper that forwards execution to an isolated feature service."""

    def __init__(self, feature: "ProxyFeature", metadata: Any):
        self._feature = feature
        self._metadata = metadata

    @property
    def name(self) -> str:
        return str(_meta_get(self._metadata, "name", ""))

    @property
    def schema(self) -> ToolSchema:
        input_schema = _meta_get(self._metadata, "input_schema", None)
        if input_schema is None:
            # camelCase spelling tolerated on the wire (see protocol.from_dict)
            input_schema = _meta_get(self._metadata, "inputSchema", None)
        return ToolSchema(
            name=self.name,
            description=str(_meta_get(self._metadata, "description", "")),
            category=_coerce_category(_meta_get(self._metadata, "category")),
            parameters=_input_schema_to_parameters(input_schema),
            command_prefix=_meta_get(self._metadata, "command_prefix"),
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        return await self._feature.call_isolated_tool(self.name, kwargs)


class ProxyFeature(Feature):
    """Feature contract adapter backed by an SDK isolated-feature client."""

    def __init__(
        self,
        agent: Any,
        runtime: InstalledFeatureRuntime,
        *,
        client_factory: Optional[Callable[..., Any]] = None,
    ):
        super().__init__(agent)
        agent_attributes = getattr(agent, "__dict__", None)
        if isinstance(agent_attributes, dict):
            agent_attributes["_isolated_runtime_features_constructed"] = True
        self.runtime = runtime
        self.name = runtime.class_name
        # Resolve hosted containment before classifying optional feature
        # metadata.  A malformed third-party class may be quarantined, but it
        # must never mask a real missing/invalid hosted namespace.
        self._isolated_runtime_scope = _agent_runtime_scope(agent)
        if (
            self._isolated_runtime_scope is None
            and getattr(agent, "isolated_runtime_hosted", False) is True
        ):
            # This read-only resolver owns the explicit missing-scope error;
            # it cannot create a standalone fallback for a declared host.
            resolve_agent_runtime_dir(agent)
        if _ISOLATED_FEATURE_CLASS_NAME.fullmatch(self.name) is None:
            raise IsolatedRuntimeConfigurationError(
                reason=_CONFIGURATION_FEATURE_IDENTITY,
            )
        # An explicit BIN override is the complete runnable and historically
        # does not require service metadata. Defer target validation until the
        # constructor has validated any hosted process-wide override below.
        self._service_target: _IsolatedServiceTarget | None = None
        self._runtime_directory_name = (
            _hosted_feature_runtime_component(runtime)
            if self._isolated_runtime_scope is not None
            else self.name
        )
        self._legacy_runtime_directory_name = (
            _legacy_hosted_feature_runtime_component(runtime)
            if self._isolated_runtime_scope is not None
            else None
        )
        self._released_legacy_runtime_root = (
            _agent_released_legacy_runtime_root(agent)
            if self._isolated_runtime_scope is not None
            else None
        )
        self._client_factory = client_factory
        self._client: Any = None
        self._tools: List[AgentTool] = []
        # UI capabilities are initialize-handshake metadata, not live child
        # traffic. Preserve the last parsed contribution while an idle child is
        # deliberately absent so the console manifest does not flap off even
        # though registry-derived feature capabilities remain enabled.
        self._idle_ui_contributions: Optional[UIContributions] = None
        self._supervision_task: Optional[asyncio.Task] = None
        # A terminal cleanup owns sealing, lifecycle serialization, and child
        # retirement as one transaction.  Keep one shared task so repeated
        # caller cancellation cannot strand a second cleanup behind
        # ``_reload_lock`` after the first caller has unwound.
        self._terminal_cleanup_task: Optional[asyncio.Task[None]] = None
        # Terminal retirement has a narrower ownership domain than a reload:
        # an owner which already holds ``_reload_lock`` must be able to finish
        # its terminal fence without waiting for a shared cleanup that is
        # itself waiting for that lock.  This lock therefore serializes only
        # exact-client ``stop()`` calls.  It is deliberately never held while
        # awaiting ``_reload_lock``, gate drain, or a supervisor task; that
        # ordering prevents a shutdown/supervisor lock cycle while ensuring
        # two cleanup owners cannot stop the same facade concurrently.
        self._terminal_retirement_lock = asyncio.Lock()
        # A client is unpublished before terminal retirement starts.  If a
        # non-SDK facade raises from ``stop()`` before it has actually
        # terminated a wedged RPC, keeping only the public client slot would
        # lose the sole retry/fencing handle.  Retain exact identities here
        # until their stop path reports completion; they never become traffic
        # visible again.
        self._terminal_retirement_clients: list[Any] = []
        # A facade that actively suppresses the timeout cancellation cannot be
        # force-killed by asyncio. Keep that still-running task explicitly
        # owned until its done callback consumes the outcome; this is a
        # fail-closed fence, never detached background work.
        self._terminal_lifecycle_tasks: list[_TrackedFacadeLifecycleTask] = []
        # A health supervisor can complete an exact stop, then be cancelled in
        # the narrow backoff window before it restarts that same facade.  Keep
        # that completion ownership by identity so terminal cleanup unpublishes
        # and drains the known-stopped facade without issuing a second stop.
        # A completion marker proves that a supervisor already stopped this
        # exact facade in its backoff window.  It is consumed before a later
        # start or terminal cleanup.  Ordinary facades use weak markers;
        # non-weakrefable facades use a short-lived exact strong marker rather
        # than an ``id`` map that could confuse object-identity reuse.
        self._terminal_stop_completed_client_markers: list[
            _TerminalStopCompletionMarker
        ] = []
        # A dishonest legacy facade can report stop success while a previously
        # admitted RPC never settles.  Keep the one sealed drain task owned so
        # its eventual outcome is consumed; never claim terminal success or
        # reopen/release the retained lifecycle state before then.
        self._terminal_traffic_drain_task: Optional[asyncio.Task[None]] = None
        # ``health()`` is also facade work.  The supervisor owns its exact task
        # from creation, letting a concurrent terminal cleanup cancel/fence it
        # before the supervisor itself runs its cancellation handler. A
        # cancellation-resistant probe retains tenant credentials and request
        # state until this completion callback consumes its outcome. One
        # supervisor issues probes serially, hence one slot.
        self._terminal_health_probe_task: Optional[_HostOwnedFacadeOperation] = None
        # A neutral cleanup task never exports an adapter/facade exception.
        # Record only that its state transition became uncertain so an explicit
        # caller cannot report success until a later clean attempt settles.
        self._terminal_cleanup_uncertain = False
        # Shutdown and quarantine make the current enable cycle terminal.  A
        # durable config repair remains allowed while soft-disabled, but no
        # normal reconciliation may build or publish another child until an
        # explicit later ``initialize()`` begins a fresh cycle.
        self._terminal_lifecycle_latched = False
        self._stopping = False
        # Every terminal request invalidates an initializer that has not yet
        # acquired ``_reload_lock``.  A re-enable may clear only the terminal
        # cycle it observed *after* cleanup; a shutdown racing in that queue
        # must keep its newer seal rather than being overwritten by stale
        # initialization state.
        self._terminal_lifecycle_generation = 0

        # Coordinate ``set_config``'s reload with the health supervisor so they
        # never stop/start the client concurrently. ``_reloading`` skips probes
        # during a reload; ``_reload_lock`` serializes the actual stop/start of
        # reload vs. a supervisor restart; ``_reload_gen`` lets the supervisor
        # detect that a reload cycled the client around its (now-stale) probe and
        # skip restarting the freshly launched one.
        self._reloading = False
        self._reload_lock = asyncio.Lock()
        self._reload_gen = 0
        # Resolve at construction, before feature startup/discovery can turn a
        # missing hosted scope into an optional-feature warning and continue.
        self._agent_runtime_dir = agent_runtime_dir(agent)
        if (
            self._isolated_runtime_scope is not None
            or getattr(agent, "isolated_runtime_hosted", False) is True
        ):
            _assert_hosted_feature_env_is_scoped(self.name, self.runtime.distribution)
            _validate_hosted_process_prebuilt_overrides(
                self.name,
                runtime_venv=self.runtime.venv,
            )
        if not os.environ.get(_env_key(self.name, "BIN")):
            self._service_target = _validated_isolated_service_target(
                self.runtime.service
            )
        self._traffic_gate = _TrafficGate(before_reset=self._assert_child_start_allowed)
        agent_attributes = getattr(agent, "__dict__", {})
        default_idle_timeout: float | None = (
            agent_attributes.get("isolated_runtime_idle_timeout_seconds")
            if isinstance(agent_attributes, dict)
            else None
        )
        feature_idle_timeouts = (
            agent_attributes.get("isolated_runtime_idle_timeouts", {})
            if isinstance(agent_attributes, dict)
            else {}
        )
        self._idle_timeout_seconds = feature_idle_timeouts.get(
            self.name, default_idle_timeout
        )
        self._telemetry_observer = (
            agent_attributes.get("isolated_runtime_telemetry_observer")
            if isinstance(agent_attributes, dict)
            else None
        )
        self._idle_monitor_task: asyncio.Task[None] | None = None
        self._idle_retired = False
        self._activity_generation = 0
        self._last_used_monotonic: float | None = None
        self._last_used_at: datetime | None = None
        self._health_restart_count = 0
        self._idle_wake_count = 0
        self._last_cold_start_seconds: float | None = None
        self._last_warm_start_seconds: float | None = None
        self._last_provision_seconds: float | None = None
        self._last_cache_hit: bool | None = None
        self._process_identity: tuple[int, float] | None = None
        self._environment_bytes: int | None = None
        self._private_writable_bytes: int | None = None
        self._downloaded_bytes: int | None = None
        self._disk_telemetry_status: str | None = None
        self._disk_budget_warning_emitted = False
        self._workspace_reclaim_generation = 0
        self._last_telemetry_emit_monotonic: float | None = None
        self._telemetry_observer_tasks: set[asyncio.Future[Any]] = set()
        self._telemetry_emit_tasks: set[asyncio.Task[None]] = set()
        self._telemetry_emit_pending = False
        self._telemetry_emit_force_pending = False
        self._telemetry_observer_emit_pending = False
        self._telemetry_observer_force_pending = False
        self._telemetry_disk_refresh_pending = False
        self._telemetry_environment_refresh_pending = False
        self._telemetry_disk_lock = asyncio.Lock()
        self._idle_resume_event = asyncio.Event()
        self._observed_inbound_producer = False
        # Event acknowledgement requests are intentionally detached from the
        # SDK read loop (which cannot await a response it must itself read).
        # Keep exact task ownership so terminal cleanup can cancel them rather
        # than leaving a raw client RPC alive after the proxy is retired.
        self._event_ack_tasks: set[asyncio.Task[None]] = set()
        self._event_ack_clients: list[tuple[Any, asyncio.Task[None]]] = []
        # Full inbound routing can run cognition. Keep it outside the SDK's
        # serial notification reader. Cursor-owning providers retain their own
        # next callback; legacy providers use the separate bounded serial queue
        # below because they have no cursor/NACK contract to prevent loss.
        self._event_ingress_tasks: set[asyncio.Task[None]] = set()
        self._event_ingress_clients: list[tuple[Any, asyncio.Task[None]]] = []
        self._non_cursor_event_ingress_queues: list[_NonCursorIngressQueue] = []
        # Serial SDK readers normally invoke event handlers one at a time, but
        # the queue handoff must stay correct for a concurrent/custom facade as
        # well.  It closes the only race between a worker deciding it is idle
        # and the next notification reserving that worker's bounded queue.
        self._non_cursor_event_ingress_lock = asyncio.Lock()
        self._deferred_acknowledged_event_tasks: set[asyncio.Task[None]] = set()
        self._fenced_recovery_failed = False
        self._venv_path: Optional[Path] = None
        self._bin_path: Optional[Path] = None
        # Hosted immutable console wrappers may themselves be operator-facing
        # symlinks. Preparation resolves and validates one exact regular target;
        # child construction must execute that target rather than re-following
        # the public link after the custody check.
        self._validated_hosted_console_path: Optional[Path] = None
        # Set only when Core observes a custody-preserving directory rename in
        # the current enable attempt. A missing manifest stamp is not evidence
        # of relocation; older, unmoved venvs use this distinction to start
        # offline and atomically backfill their canonical path.
        self._venv_relocated_this_startup = False
        self._host_config: Dict[str, Any] = {}
        self._hosted_telegram_startup_attested = False
        self._hosted_telegram_route_identity: Optional[str] = None
        self._hosted_telegram_route_claim: Optional[ChannelRouteClaim] = None
        self._hosted_telegram_ownership_store: Optional[ChannelRouteOwnershipStore] = None
        # Process-local identity for the durable pending lease.  A new proxy
        # instance (including one after a crash/restart) never impersonates an
        # earlier writer; it may only reclaim that writer after its lease has
        # conservatively expired.
        self._config_transition_owner = uuid4().hex
        # ``{}`` is a valid, fully-loaded config.  Keep its loaded state
        # separate from its truthiness so a concurrent read never falls back to
        # durable transition state while a service is still running the empty
        # config.
        self._host_config_loaded = False
        # A visible legacy row wins during rolling overlap.  Scoped authority
        # is cached only while that legacy row remains absent; every scoped
        # operation revalidates the absence before it reads or writes.
        self._resolved_config_node_id: Optional[str] = None
        self._config_identity_lock = asyncio.Lock()
        self._channel_adapter: Optional["ProxyChannelAdapter"] = None
        # Channel-link plumbing (#2081): the bridged channel type and the name of
        # its pairing tool. When that tool runs on the streaming turn, the host
        # emits a persisted ``channel_link`` typed part so the pairing card rides
        # the conversation that asked for it (survives refresh) instead of
        # orphaning as a live SSE bubble.
        self._channel_type: Optional[str] = None
        self._link_tool: Optional[str] = None

    @property
    def contribution_owner(self) -> str:
        """Return the durable contribution identity of this isolated runtime.

        Every isolated feature is represented in-process by ``ProxyFeature``,
        so the SDK's implementation-class default would give every proxy the
        same owner.  Hash only installed runtime metadata: unlike process or
        object identity, this remains stable across agent restarts while
        distinguishing independently installed feature implementations.  The
        fixed-format digest also remains within the SDK stable-token contract
        regardless of distribution or entry-point length and character set.
        """

        identity = json.dumps(
            (
                self.runtime.distribution,
                self.runtime.class_name,
                self.runtime.entry_point,
            ),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"isolated-runtime:{digest}"

    @property
    def tool_description(self) -> str:
        if self.runtime.description:
            return self.runtime.description
        return f"Isolated feature service for {self.name}"

    def _runtime_telemetry_snapshot_inputs(
        self,
    ) -> tuple[dict[str, Any], Any, tuple[int, float] | None]:
        """Freeze event-loop-owned lifecycle state before process sampling."""

        retirement_clients = tuple(self._terminal_retirement_clients)
        uncertain_retirement = self._retirement_is_uncertain()
        client = self._client
        if client is None and retirement_clients:
            client = retirement_clients[0]
        idle = self._idle_retired and client is None
        if uncertain_retirement:
            state = "retirement-uncertain"
        elif self._terminal_lifecycle_latched or self._stopping:
            state = "stopping" if client is not None else "stopped"
        elif idle:
            state = "idle"
        elif client is not None:
            state = "running"
        else:
            state = "starting"
        values = {
            "feature": self.name,
            "distribution": self.runtime.distribution,
            "state": state,
            "lifecycle_generation": self._reload_gen,
            "active_processes": int(client is not None or uncertain_retirement),
            "idle_processes": int(idle and not uncertain_retirement),
            "restart_count": self._health_restart_count,
            "idle_wake_count": self._idle_wake_count,
            "last_used_at": self._last_used_at,
            "cold_start_seconds": self._last_cold_start_seconds,
            "warm_start_seconds": self._last_warm_start_seconds,
            "environment_bytes": self._environment_bytes,
            "private_writable_bytes": self._private_writable_bytes,
            "downloaded_bytes": self._downloaded_bytes,
            "provision_seconds": self._last_provision_seconds,
            "cache_hit": self._last_cache_hit,
            "cleanup_eligible": (
                idle
                and not uncertain_retirement
                and not self._terminal_lifecycle_latched
                and not self._stopping
            ),
            "disk_telemetry_status": self._disk_telemetry_status,
        }
        return values, client, self._process_identity

    def _retirement_is_uncertain(self) -> bool:
        """Return the predicate shared by telemetry and workspace reclaim."""

        live_evidence = bool(
            self._terminal_retirement_clients
            or self._terminal_lifecycle_tasks
        )
        # A terminal cleanup can lose its last concrete facade handle while
        # still being unable to prove completion. Non-terminal recovery does
        # not inherit that sticky terminal meaning: once all exact evidence is
        # gone, a republished/cleanly-idle child must report its current truth.
        return live_evidence or bool(
            self._terminal_cleanup_uncertain
            and (self._terminal_lifecycle_latched or self._stopping)
        )

    @staticmethod
    def _build_runtime_telemetry_snapshot(
        values: dict[str, Any],
        client: Any,
        process_identity: tuple[int, float] | None,
    ) -> IsolatedRuntimeTelemetrySnapshot:
        """Perform only OS process sampling after lifecycle state is frozen."""

        rss_bytes: int | None = None
        cpu_seconds: float | None = None
        open_fds: int | None = None
        process_count: int | None = None
        process = getattr(client, "process", None) if client is not None else None
        pid = getattr(process, "pid", None)
        if type(pid) is int and getattr(process, "returncode", None) is None:
            try:
                observed = psutil.Process(pid)
                if process_identity != (pid, observed.create_time()):
                    raise psutil.NoSuchProcess(pid)
                children = observed.children(recursive=True)
                memory = observed.memory_info().rss
                cpu = observed.cpu_times()
                rss_bytes = int(memory) + sum(
                    int(child.memory_info().rss) for child in children
                )
                cpu_seconds = float(cpu.user + cpu.system) + sum(
                    float(times.user + times.system)
                    for times in (child.cpu_times() for child in children)
                )
                num_fds = getattr(observed, "num_fds", None)
                open_fds = int(num_fds()) if callable(num_fds) else None
                process_count = 1 + len(children)
            except (OSError, RuntimeError, psutil.Error):
                pass
        return IsolatedRuntimeTelemetrySnapshot(
            **values,
            rss_bytes=rss_bytes,
            cpu_seconds=cpu_seconds,
            open_fds=open_fds,
            process_count=process_count,
        )

    def runtime_telemetry_snapshot(self) -> IsolatedRuntimeTelemetrySnapshot:
        """Return a path-free, non-mutating snapshot for this exact proxy."""

        return self._build_runtime_telemetry_snapshot(
            *self._runtime_telemetry_snapshot_inputs()
        )

    async def _refresh_disk_telemetry(
        self,
        *,
        refresh_environment: bool = False,
        expected_reclaim_generation: int | None = None,
    ) -> None:
        """Refresh owned workspace byte counters outside the event loop."""

        if self._telemetry_observer is None:
            return

        runtime_dir = self._feature_runtime_dir()
        venv = self._venv_path

        def measure() -> tuple[int | None, int | None, int | None, str]:
            deadline = time.monotonic() + _DISK_TELEMETRY_TIME_BUDGET_SECONDS
            statuses: list[str] = []
            if (
                refresh_environment
                and venv is not None
                and venv == runtime_dir / ".venv"
                and self._bin_path is None
            ):
                environment, environment_status = _measure_directory_tree_bytes(
                    venv, deadline=deadline
                )
                statuses.append(environment_status)
            else:
                environment = self._environment_bytes
            private_measurements = [
                _measure_directory_tree_bytes(
                    runtime_dir / component,
                    deadline=deadline,
                )
                for component in ("work", "home", "tmp", "config", "data", "cache")
            ]
            private_sizes = [size for size, _status in private_measurements]
            statuses.extend(status for _size, status in private_measurements)
            private = (
                sum(size for size in private_sizes if size is not None)
                if all(size is not None for size in private_sizes)
                else None
            )
            downloaded, downloaded_status = _measure_directory_tree_bytes(
                runtime_dir / "provisioning_cache",
                deadline=deadline,
            )
            statuses.append(downloaded_status)
            status = (
                "budget-exceeded"
                if "budget-exceeded" in statuses
                else "unavailable"
                if "unavailable" in statuses
                else "complete"
            )
            return environment, private, downloaded, status

        async with self._telemetry_disk_lock:
            if (
                expected_reclaim_generation is not None
                and expected_reclaim_generation != self._workspace_reclaim_generation
            ):
                return
            (
                self._environment_bytes,
                self._private_writable_bytes,
                self._downloaded_bytes,
                self._disk_telemetry_status,
            ) = await asyncio.to_thread(measure)
        if (
            self._disk_telemetry_status == "budget-exceeded"
            and not self._disk_budget_warning_emitted
        ):
            self._disk_budget_warning_emitted = True
            logger.warning(
                "Hosted isolated runtime disk telemetry exceeded its shared "
                "measurement budget for %s",
                self.name,
            )

    async def _emit_runtime_telemetry(self, *, rate_limited: bool = False) -> None:
        observer = self._telemetry_observer
        if observer is None:
            return
        if any(not task.done() for task in self._telemetry_observer_tasks):
            # Coalesce behind the one retained observer rather than dropping a
            # terminal idle/capacity transition that may have no later trigger.
            self._telemetry_observer_emit_pending = True
            self._telemetry_observer_force_pending |= not rate_limited
            return
        loop = asyncio.get_running_loop()
        if (
            rate_limited
            and self._last_telemetry_emit_monotonic is not None
            and loop.time() - self._last_telemetry_emit_monotonic
            < _TELEMETRY_EMIT_MIN_INTERVAL
        ):
            return
        self._last_telemetry_emit_monotonic = loop.time()
        try:
            # Snapshot loop-owned lifecycle references before crossing into the
            # worker.  The synchronous builder is deliberately non-mutating,
            # so telemetry can never consume or clear lifecycle task state.
            inputs = self._runtime_telemetry_snapshot_inputs()
            snapshot = await asyncio.to_thread(
                self._build_runtime_telemetry_snapshot, *inputs
            )
            async def invoke_observer() -> None:
                # A host may supply a normal callable or an async callable. Run
                # the call itself in a dedicated bounded pool so a slow
                # synchronous observer cannot freeze the event loop or occupy
                # the default executor needed by venv/lifecycle work.
                result = await asyncio.get_running_loop().run_in_executor(
                    _TELEMETRY_OBSERVER_EXECUTOR,
                    observer,
                    snapshot,
                )
                if inspect.isawaitable(result):
                    await result

            task = asyncio.create_task(
                invoke_observer(),
                name=f"isolated-runtime-observer:{self.name}",
            )
            self._telemetry_observer_tasks.add(task)

            def consume_observer(completed: asyncio.Future[Any]) -> None:
                self._telemetry_observer_tasks.discard(completed)
                if completed.cancelled():
                    self._telemetry_observer_emit_pending = False
                    self._telemetry_observer_force_pending = False
                    return
                try:
                    completed.result()
                except BaseException:  # noqa: BLE001 - advisory host callback
                    logger.warning(
                        "Hosted isolated runtime telemetry observer failed for %s",
                        self.name,
                    )
                pending = self._telemetry_observer_emit_pending
                force_pending = self._telemetry_observer_force_pending
                self._telemetry_observer_emit_pending = False
                self._telemetry_observer_force_pending = False
                if (
                    pending
                    and not self._terminal_lifecycle_latched
                    and not self._stopping
                ):
                    self._schedule_runtime_telemetry(force=force_pending)

            task.add_done_callback(consume_observer)
            done, _pending = await asyncio.wait(
                (task,), timeout=_TELEMETRY_OBSERVER_TIMEOUT
            )
            if task not in done:
                logger.warning(
                    "Hosted isolated runtime telemetry observer exceeded its "
                    "delivery budget for %s",
                    self.name,
                )
        except asyncio.CancelledError:
            if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                raise
            logger.warning(
                "Hosted isolated runtime telemetry observer cancelled for %s", self.name
            )
        except BaseException:  # noqa: BLE001 - host observer cannot own child lifecycle
            logger.warning(
                "Hosted isolated runtime telemetry observer failed for %s", self.name
            )

    def _schedule_runtime_telemetry(
        self,
        *,
        force: bool = False,
        refresh_disk: bool = False,
        refresh_environment: bool = False,
    ) -> None:
        """Emit hot-path telemetry without retaining traffic admission."""

        if (
            self._telemetry_observer is None
            or self._terminal_lifecycle_latched
            or self._stopping
        ):
            return
        self._telemetry_disk_refresh_pending |= refresh_disk
        self._telemetry_environment_refresh_pending |= refresh_environment
        if any(not task.done() for task in self._telemetry_observer_tasks):
            self._telemetry_observer_emit_pending = True
            self._telemetry_observer_force_pending |= force
            return
        if any(not task.done() for task in self._telemetry_emit_tasks):
            self._telemetry_emit_pending = True
            self._telemetry_emit_force_pending |= force
            return
        refresh_disk = self._telemetry_disk_refresh_pending
        refresh_environment = self._telemetry_environment_refresh_pending
        reclaim_generation = self._workspace_reclaim_generation
        force |= self._telemetry_emit_force_pending
        self._telemetry_emit_pending = False
        self._telemetry_emit_force_pending = False
        self._telemetry_disk_refresh_pending = False
        self._telemetry_environment_refresh_pending = False

        async def deliver() -> None:
            if refresh_disk:
                try:
                    await self._refresh_disk_telemetry(
                        refresh_environment=refresh_environment,
                        expected_reclaim_generation=reclaim_generation,
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException:  # noqa: BLE001 - telemetry stays advisory
                    logger.warning(
                        "Hosted isolated runtime disk telemetry refresh failed for %s",
                        self.name,
                    )
                    async with self._telemetry_disk_lock:
                        if reclaim_generation == self._workspace_reclaim_generation:
                            self._environment_bytes = None
                            self._private_writable_bytes = None
                            self._downloaded_bytes = None
                            self._disk_telemetry_status = "unavailable"
            await self._emit_runtime_telemetry(rate_limited=not force)

        coro = deliver()
        name = f"isolated-runtime-telemetry:{self.name}"
        tracker = getattr(self.agent, "_track_background_task", None)
        task = None
        if callable(tracker):
            try:
                tracked = tracker(coro, name=name)
                if isinstance(tracked, asyncio.Task):
                    task = tracked
            except Exception:  # noqa: BLE001 - test doubles may reject tracking
                task = None
        if task is None:
            task = asyncio.create_task(coro, name=name)
        self._telemetry_emit_tasks.add(task)

        def consume(completed: asyncio.Task[None]) -> None:
            self._telemetry_emit_tasks.discard(completed)
            if completed.cancelled():
                self._telemetry_emit_pending = False
                self._telemetry_emit_force_pending = False
                return
            try:
                completed.result()
            except BaseException:
                pass
            pending = self._telemetry_emit_pending
            force_pending = self._telemetry_emit_force_pending
            if (
                pending
                and not self._terminal_lifecycle_latched
                and not self._stopping
            ):
                self._schedule_runtime_telemetry(force=force_pending)

        task.add_done_callback(consume)

    def _record_runtime_activity(self) -> None:
        self._activity_generation += 1
        self._last_used_monotonic = asyncio.get_running_loop().time()
        self._last_used_at = datetime.now(timezone.utc)

    def _capture_process_identity(self, client: Any) -> None:
        """Pin PID telemetry to the exact process generation just started."""

        self._process_identity = None
        process = getattr(client, "process", None)
        pid = getattr(process, "pid", None)
        if type(pid) is not int or getattr(process, "returncode", None) is not None:
            return
        try:
            self._process_identity = (pid, psutil.Process(pid).create_time())
        except (OSError, psutil.Error):
            pass

    def _is_telegram_runtime(self) -> bool:
        return (
            re.sub(r"[-_.]+", "-", self.runtime.distribution.strip().lower())
            == "kestrel-channel-telegram"
        )

    def _require_telegram_runtime(self) -> None:
        """Reject route attestation for any non-Telegram isolated runtime."""
        if not self._is_telegram_runtime():
            raise RuntimeError(
                "hosted Telegram startup attestation is valid only for the Telegram feature"
            )

    @staticmethod
    def _telegram_route_identity(bot_id: object) -> str:
        """Build the only Telegram route spelling Core places in the ledger."""

        return f"telegram-bot:{canonical_telegram_bot_id(bot_id)}"

    def _clear_hosted_telegram_startup_attestation(self) -> None:
        self._hosted_telegram_startup_attested = False
        self._hosted_telegram_route_identity = None
        self._hosted_telegram_route_claim = None
        self._hosted_telegram_ownership_store = None

    async def _resolve_hosted_telegram_startup_attestation(self) -> None:
        """Resolve host route evidence before a Telegram child handshake.

        A normal Core boot discovers isolated features before the caller can
        obtain the new proxy instance.  Hosts therefore inject the resolver on
        the agent *before* initialization under the explicit
        ``hosted_telegram_route_attestation_resolver`` seam.  Returning
        ``None`` means this is an ordinary polling route.  Returning typed
        hosted-route evidence requires a successful durable claim; any
        conflict aborts startup before the child can touch provider ingress.
        """

        self._require_telegram_runtime()
        # Do not use bare ``getattr`` here: dynamic integration proxies (and
        # unittest mocks) manufacture callable-looking attributes that were
        # never a host injection. Only an explicitly stored instance value or
        # a concrete class-level resolver opens this hosted-route path.
        resolver_name = "hosted_telegram_route_attestation_resolver"
        instance_attributes = getattr(self.agent, "__dict__", None)
        resolver = (
            instance_attributes.get(resolver_name)
            if isinstance(instance_attributes, dict)
            else None
        )
        if resolver is None:
            class_resolver = getattr(type(self.agent), resolver_name, None)
            if class_resolver is not None:
                resolver = getattr(self.agent, resolver_name)
        if resolver is None:
            # Preserve the narrow pre-initialize injection API for hosts that
            # already hold the proxy, but reassert its generation before a new
            # handshake. A stale in-memory boolean is never sufficient.
            if (
                self._hosted_telegram_ownership_store is not None
                and self._hosted_telegram_route_identity is not None
            ):
                bot_id = self._hosted_telegram_route_identity.removeprefix(
                    "telegram-bot:"
                )
                claimed = await self.reconcile_hosted_telegram_route_claim(
                    ownership_store=self._hosted_telegram_ownership_store,
                    bot_id=bot_id,
                )
                if not claimed:
                    raise RuntimeError(
                        "Hosted Telegram route is already owned; refusing to start child"
                    )
            else:
                self._clear_hosted_telegram_startup_attestation()
            return
        if not callable(resolver):
            raise TypeError(
                "hosted_telegram_route_attestation_resolver must be callable"
            )
        resolved = await _maybe_await(resolver(self))
        if resolved is None:
            if self._hosted_telegram_route_claim is not None:
                raise RuntimeError(
                    "cannot remove hosted Telegram route evidence while its "
                    "generation remains claimed; release it first"
                )
            self._clear_hosted_telegram_startup_attestation()
            return
        if not isinstance(resolved, HostedTelegramRouteAttestation):
            raise TypeError(
                "hosted Telegram route resolver must return "
                "HostedTelegramRouteAttestation or None"
            )
        claimed = await self.reconcile_hosted_telegram_route_claim(
            ownership_store=resolved.ownership_store,
            bot_id=resolved.bot_id,
        )
        if not claimed:
            raise RuntimeError(
                "Hosted Telegram route is already owned; refusing to start child"
            )

    async def reconcile_hosted_telegram_route_claim(
        self,
        *,
        ownership_store: ChannelRouteOwnershipStore,
        bot_id: str,
    ) -> bool:
        """Durably claim/reconcile the host-provisioned Telegram route.

        This is a narrow host/provisioner API, not a feature config setting or
        agent tool.  The host calls it before starting a webhook-owned child
        and after replacing one.  A successful result is the *only* way Core
        injects the no-poll startup capability; a config boolean cannot create
        that attestation.  Core intentionally does not fabricate a Telegram
        HTTP endpoint in this branch — the external Frinz provisioner owns
        provider API calls and may invoke the same generic durable store.
        """
        self._require_telegram_runtime()
        if not isinstance(ownership_store, ChannelRouteOwnershipStore):
            raise TypeError(
                "hosted Telegram route attestation requires ChannelRouteOwnershipStore"
            )
        route_identity = self._telegram_route_identity(bot_id)
        if (
            self._hosted_telegram_route_claim is not None
            and (
                self._hosted_telegram_route_identity != route_identity
                or self._hosted_telegram_ownership_store is not ownership_store
            )
        ):
            raise RuntimeError(
                "cannot reconcile a different hosted Telegram route ledger or "
                "identity while the current generation remains claimed; release it first"
            )
        claim = await ownership_store.claim(
            channel_type="telegram",
            canonical_route_identity=route_identity,
            agent_id=self._config_agent_did(),
        )
        if claim is None:
            self._clear_hosted_telegram_startup_attestation()
            return False
        self._hosted_telegram_startup_attested = True
        self._hosted_telegram_route_identity = route_identity
        self._hosted_telegram_route_claim = claim
        self._hosted_telegram_ownership_store = ownership_store
        return True

    async def release_hosted_telegram_route_claim(
        self,
        *,
        ownership_store: ChannelRouteOwnershipStore,
        bot_id: str,
    ) -> bool:
        """Release this proxy's route claim and revoke its launch attestation."""
        self._require_telegram_runtime()
        if not isinstance(ownership_store, ChannelRouteOwnershipStore):
            raise TypeError(
                "hosted Telegram route release requires ChannelRouteOwnershipStore"
            )
        route_identity = self._telegram_route_identity(bot_id)
        claim = self._hosted_telegram_route_claim
        if (
            claim is None
            or self._hosted_telegram_ownership_store is not ownership_store
            or self._hosted_telegram_route_identity != route_identity
        ):
            return False
        released = await ownership_store.release(
            channel_type="telegram",
            canonical_route_identity=route_identity,
            agent_id=self._config_agent_did(),
            claim=claim,
        )
        self._clear_hosted_telegram_startup_attestation()
        return released

    def _config_agent_did(self) -> str:
        """Return the stable DID that scopes this proxy's durable config.

        ``agent_id`` is deliberately not accepted as a fallback.  It is an
        alias in Kestrel today, but accepting an arbitrary display/process ID
        here would turn a durable secret/config namespace back into ambient
        global state when an embedding gets its construction wrong.
        """

        did = getattr(self.agent, "did", None)
        if not isinstance(did, str) or not _DID_IDENTITY_RE.fullmatch(did):
            raise RuntimeError(
                f"Cannot use durable config for isolated feature {self.name}: "
                "agent DID is missing or invalid"
            )
        return did

    def _config_node_id(self) -> str:
        """Return the DID-scoped ID used when no legacy authority is visible.

        This is intentionally deterministic rather than the resolved identity:
        callers that need the actual durable authority must await
        :meth:`_resolve_config_node_id` so an owned legacy row can remain the
        only live CAS target during a rolling upgrade.
        """

        return f"{_SCOPED_CONFIG_NODE_PREFIX}:{self._config_agent_did()}:{self.name}"

    def _legacy_config_node_id(self) -> str:
        """Return the pre-DID-scoping config key for in-place adoption."""

        return super()._config_node_id()

    def _require_config_storage_scope(self, storage: Any) -> None:
        """Prove the config store is bound to this exact agent DID.

        A legacy key has no DID in its global node ID. Reading it through an
        unbound store could expose another agent's config, so identity
        resolution is permitted only through the same agent-bound storage
        capability used by normal graph operations.
        """

        did = self._config_agent_did()
        storage_agent_id = getattr(storage, "agent_id", None)
        if not isinstance(storage_agent_id, str) or storage_agent_id != did:
            raise RuntimeError(
                f"Cannot use durable config for isolated feature {self.name}: "
                "storage is not bound to the current agent DID"
            )

    @staticmethod
    def _validate_config_node(node: Any, *, node_kind: str) -> None:
        """Validate a visible config row before accepting it as authority."""

        if getattr(node, "node_type", None) != Feature._CONFIG_NODE_TYPE:
            raise RuntimeError(f"{node_kind} config node has an invalid type")
        if not isinstance(getattr(node, "properties", None), dict):
            raise RuntimeError(f"{node_kind} config node has invalid properties")

    async def _resolve_config_node_id(
        self,
        storage: Any,
        *,
        fence_cached_scoped_authority: bool = False,
    ) -> str:
        """Resolve the presently visible config authority.

        During a rolling upgrade a same-agent legacy row is the only safe
        authority whenever it is visible, even if an empty DID-scoped row also
        exists.  We therefore check it before every scoped use, including a
        cached identity.  A transition that had already cached scoped identity
        receives :class:`_ConfigAuthorityChanged` rather than silently drifting
        to legacy midway through its lifecycle.

        The final legacy recheck narrows the creation race but cannot make two
        independently-written keys atomic with an old binary.  A scoped CAS is
        consequently revalidated again before any lifecycle hook or traffic can
        proceed; a visible mixed-authority result fails closed.  If an old
        binary writes legacy after that final recheck, an orphaned scoped
        candidate can remain.  Kestrel intentionally does not delete it: after
        the rolling overlap is over, an operator must inspect and remove that
        orphan only after confirming no old replica remains.  All future
        proxies still converge on the visible legacy row.
        """

        self._require_config_storage_scope(storage)
        get_node = getattr(storage, "get_node", None)
        if not callable(get_node):
            raise RuntimeError("storage cannot resolve isolated config identity")

        async with self._config_identity_lock:
            cached = self._resolved_config_node_id
            legacy_node_id = self._legacy_config_node_id()
            legacy = await _maybe_await(get_node(legacy_node_id))
            if legacy is not None:
                self._validate_config_node(legacy, node_kind="legacy")
                self._resolved_config_node_id = legacy_node_id
                if (
                    fence_cached_scoped_authority
                    and cached == self._config_node_id()
                ):
                    raise _ConfigAuthorityChanged(
                        "legacy isolated config authority became visible "
                        "during a rolling upgrade"
                    )
                return legacy_node_id

            if cached is not None:
                return cached

            scoped_node_id = self._config_node_id()
            scoped = await _maybe_await(get_node(scoped_node_id))
            if scoped is not None:
                self._validate_config_node(scoped, node_kind="scoped")

            # This does not claim cross-key atomicity: an old binary can still
            # create legacy immediately afterwards.  It does ensure that a
            # legacy row visible while resolving wins over scoped, and every
            # later scoped read/write repeats the same fence.
            legacy = await _maybe_await(get_node(legacy_node_id))
            if legacy is not None:
                self._validate_config_node(legacy, node_kind="legacy")
                self._resolved_config_node_id = legacy_node_id
                return legacy_node_id

            self._resolved_config_node_id = scoped_node_id
            return scoped_node_id

    async def _resolved_config_node_id_for(
        self,
        storage: Any,
        *,
        expected_node_id: Optional[str] = None,
        fence_cached_scoped_authority: bool = False,
    ) -> str:
        """Return the resolved identity and reject a pinned transition drift.

        A caller that pins ``expected_node_id`` is always fencing a durable
        transition, so a newly visible legacy authority must interrupt it.
        Read-only callers can instead adopt that legacy authority before any
        transition is pinned by leaving both arguments unset.
        """

        node_id = await self._resolve_config_node_id(
            storage,
            fence_cached_scoped_authority=(
                fence_cached_scoped_authority or expected_node_id is not None
            ),
        )
        if expected_node_id is not None and node_id != expected_node_id:
            raise _ConfigAuthorityChanged(
                "isolated config authority changed during transition"
            )
        return node_id

    async def persist_config(self, config: Dict) -> None:
        """Best-effort compatibility persist without clobbering transitions.

        This helper is used to seed boot-time config, not to apply a hosted
        runtime transition.  Its read/modify/write must still use the graph
        store CAS primitive: ``add_node`` can erase a different replica's
        pending generation or newer promotion.  A contention loss is therefore
        left intact as best effort; an invalid identity, policy boundary, or
        missing CAS capability remains a hard error rather than being hidden.
        """

        storage = getattr(self.agent, "storage", None)
        if storage is None:
            logger.debug("No storage available to persist config for %s", self.name)
            return
        get_node = getattr(storage, "get_node", None)
        # Keep Feature's historical absent-storage treatment for loose mocks;
        # a real graph-store contract is asynchronous and therefore must pass
        # the DID-bound identity checks below.
        if not inspect.iscoroutinefunction(get_node):
            return
        if not await self._persistent_config_writes_allowed(storage):
            logger.debug(
                "Skipping durable config persist for %s — persistent writes are disabled",
                self.name,
            )
            return

        compare_and_swap = getattr(storage, "compare_and_swap_node", None)
        if not callable(compare_and_swap):
            raise RuntimeError(
                "persistent isolated config requires compare_and_swap_node"
            )

        state = await self._read_config_state(storage)
        if state.has_pending:
            # ``config`` and ``pending_config`` are one generation-owned state
            # machine.  Updating only the active half would corrupt that
            # relationship even if we preserved the metadata fields verbatim.
            logger.debug(
                "Skipping compatibility config persist for %s — a transition is active",
                self.name,
            )
            return

        properties = dict(state.properties or {})
        properties["config"] = dict(config)
        write = await self._write_config_state(
            storage,
            state.properties,
            properties,
            expected_node_id=state.node_id,
        )
        if write.committed:
            return
        if write.error is not None:
            logger.warning("Failed to persist config for %s", self.name)
            return
        logger.debug(
            "Skipped compatibility config persist for %s — durable config changed",
            self.name,
        )

    async def load_persisted_config(
        self, *, raise_on_error: bool = False
    ) -> Optional[Dict]:
        """Load config from this proxy's resolved durable authority."""

        storage = getattr(self.agent, "storage", None)
        if storage is None:
            return None
        get_node = getattr(storage, "get_node", None)
        if not inspect.iscoroutinefunction(get_node):
            return None
        try:
            node_id = await self._resolve_config_node_id(storage)
            node = await _maybe_await(get_node(node_id))
            if node_id == self._config_node_id():
                # Recheck after the scoped read.  This narrows (but does not
                # close) the old-writer cross-key race described above.
                resolved_node_id = await self._resolve_config_node_id(storage)
                if resolved_node_id != node_id:
                    node_id = resolved_node_id
                    node = await _maybe_await(get_node(node_id))
            if node is None:
                return None
            self._validate_config_node(node, node_kind="stored")
            raw_properties = node.properties
            config = raw_properties.get("config")
            if isinstance(config, str):
                config = json.loads(config)
            if not isinstance(config, dict):
                config = {}
            config = dict(config)
            disabled = config.get("disabled_skills")
            if isinstance(disabled, list):
                self.disabled_skills = set(disabled)
            return config
        except Exception:
            if raise_on_error:
                raise
            logger.warning("Failed to load persisted config for %s", self.name)
            return None

    @property
    def config_schema(self) -> Optional[Dict]:
        return self.runtime.config_schema

    async def initialize(self):
        """Initialize a child as an all-or-terminal lifecycle transaction.

        ``_connect_client`` intentionally publishes before event registration
        completes.  A caller cancellation delivered at the following gate
        reset used to return from here with that published child unsupervised.
        Run the initialization body independently, then terminally retire any
        published state before reporting *any* failure to the caller.
        """

        # A prior terminal stop failure leaves the proxy sealed with a private
        # retirement handle.  Do not begin a fresh enable cycle beside that
        # still-live child: retry its bounded stop path first, or report the
        # outstanding retirement failure honestly to the explicit initializer.
        enable_generation = self._terminal_lifecycle_generation
        if (
            self._terminal_lifecycle_latched
            or self._terminal_cleanup_task is not None
            or self._terminal_retirement_clients
            or self._has_running_terminal_health_probe_task()
            or self._terminal_cleanup_uncertain
        ):
            # Initialization may retry a previously failed retirement, but it
            # never does so as a non-terminal side path.  Revoking a fresh
            # generation before joining the cleanup preserves the same
            # synchronous ordering edge as shutdown/quarantine against another
            # initializer queued on ``_reload_lock``.
            enable_generation = self._latch_terminal_lifecycle()
            await self._complete_terminal_cleanup()

        task = asyncio.create_task(
            self._initialize_uninterrupted(enable_generation),
            name=f"isolated-initialize:{self.name}",
        )
        try:
            await _await_task_until_complete(task, preserve_cancellation=False)
        except BaseException:
            # This also covers a cancellation observed after the inner task
            # successfully reset the gate.  A cancelled initialize is never a
            # successful publication from its caller's perspective.
            await self._quarantine_unreconciled_client()
            raise

    async def _initialize_uninterrupted(self, terminal_generation: int) -> None:
        """Build and publish a fresh child while holding lifecycle ownership."""

        async with self._reload_lock:
            if terminal_generation != self._terminal_lifecycle_generation:
                raise _TerminalLifecyclePermitRevoked(
                    "isolated feature terminal lifecycle changed during initialize"
                )
            # A completed shutdown/quarantine transaction belongs to the old
            # enable cycle.  A later explicit initialize gets a fresh terminal
            # transaction if this new cycle subsequently fails.
            if self._terminal_cleanup_task is not None and self._terminal_cleanup_task.done():
                self._terminal_cleanup_task = None
            # Reset lifecycle state to the fresh-start baseline BEFORE the client or
            # supervisor start. ``shutdown()`` latches ``_stopping=True`` to unwind
            # the health supervisor; runtime re-enable re-runs this SAME instance's
            # ``initialize()`` (``_activate_feature_runtime``), so without the reset
            # the new ``_supervise()`` task sees a stale ``_stopping`` and exits on
            # its first ``while not self._stopping`` check — leaving a re-enabled
            # service with no health supervisor (kestrel-sovereign#2522 P2).
            self._terminal_lifecycle_latched = False
            self._stopping = False
            self._idle_retired = False
            self._process_identity = None
            # A previous enable cycle may have left an intentional empty config (or
            # a stopped client) on this same object. A fresh initialize must never
            # let that in-memory state stand in for the durable read below.
            self._host_config = {}
            self._host_config_loaded = False
            self._prepare_runtime_workspace()
            self._venv_path, self._bin_path = self.resolve_runtime_paths()
            if self._bin_path is None:
                await self._ensure_venv_without_blocking_event_loop()
            # Resolve persisted/UI host config BEFORE building the client so it can be
            # forwarded to the isolated service through the initialize handshake (the
            # service is otherwise launched bare, with only env vars).
            await self._ensure_host_config_loaded()
            if self._is_telegram_runtime():
                await self._resolve_hosted_telegram_startup_attestation()
            self._assert_child_start_allowed()
            await self._connect_client()
            self._assert_child_start_allowed()
            # A previously quarantined instance is only made reachable after its
            # fresh child was initialized from durable config.
            await self._reset_traffic_gate_after_initialize()
            self._assert_child_start_allowed()
            self._supervision_task = self._start_supervision()
            self._last_used_monotonic = asyncio.get_running_loop().time()
            self._start_idle_monitor()
        # Advisory disk/process telemetry never extends lifecycle-lock custody.
        # Capacity telemetry is advisory. Initial publication must not await a
        # disk walk or a host callback, and all emissions must pass through the
        # same serialization/coalescing owner.
        self._schedule_runtime_telemetry(
            force=True,
            refresh_disk=True,
            refresh_environment=True,
        )

    async def _ensure_venv_without_blocking_event_loop(self) -> None:
        """Own synchronous preparation in a worker without orphaning it.

        Fresh callable verification executes feature import code with a finite
        subprocess timeout. Running the surrounding synchronous preparation in
        a worker keeps that budget from blocking every agent on the host loop.
        A cancellation is observed only after the shielded worker settles, so
        venv mutation cannot continue after lifecycle ownership is released.
        """

        started = asyncio.get_running_loop().time()
        task = asyncio.create_task(
            asyncio.to_thread(self.ensure_venv),
            name=f"isolated-venv-prepare:{self.name}",
        )
        environment_mutated = await _await_task_until_complete(
            task, preserve_cancellation=False
        )
        self._last_provision_seconds = asyncio.get_running_loop().time() - started
        self._last_cache_hit = not environment_mutated

    def _latch_terminal_lifecycle(self) -> int:
        """Record one terminal intent and revoke queued initialization permits.

        Latching is synchronous by design: it is the ordering edge between a
        terminal caller and an initializer waiting to acquire ``_reload_lock``.
        Increment even when a prior terminal cycle is already latched, because
        an explicit shutdown in that state is still newer than any permit a
        re-enable captured from the older completed cycle.
        """

        self._terminal_lifecycle_generation += 1
        self._terminal_lifecycle_latched = True
        self._stopping = True
        self._idle_resume_event.set()
        for task in tuple(self._event_ack_tasks):
            task.cancel()
        for task in tuple(self._event_ingress_tasks):
            task.cancel()
        for task in tuple(self._deferred_acknowledged_event_tasks):
            task.cancel()
        for task in tuple(self._telemetry_emit_tasks):
            task.cancel()
        for task in tuple(self._telemetry_observer_tasks):
            task.cancel()
        self._telemetry_emit_pending = False
        self._telemetry_emit_force_pending = False
        self._telemetry_observer_emit_pending = False
        self._telemetry_observer_force_pending = False
        self._telemetry_disk_refresh_pending = False
        self._telemetry_environment_refresh_pending = False
        idle_monitor = self._idle_monitor_task
        if idle_monitor is not None and idle_monitor is not asyncio.current_task():
            idle_monitor.cancel()
        self._idle_monitor_task = None
        return self._terminal_lifecycle_generation

    async def _run_traffic_gate_operation(
        self,
        operation: Awaitable[None],
        *,
        name: str,
        preserve_cancellation: bool = False,
    ) -> None:
        """Complete a gate mutation before its lifecycle owner can proceed.

        A cancellation can arrive while a close/drain waits for an admitted
        tool.  Running the boundary in a shielded task ensures that cancellation
        is reported only after the gate has reached a coherent state, so no task
        remains after ``_reload_lock`` with authority to mutate admission.
        """

        task = asyncio.create_task(operation, name=f"isolated-traffic-{name}:{self.name}")
        await _await_task_until_complete(
            task,
            preserve_cancellation=preserve_cancellation,
        )

    async def _close_traffic_gate(self) -> None:
        await self._run_traffic_gate_operation(
            self._traffic_gate.close_and_drain(),
            name="close",
        )

    async def _close_traffic_gate_admission(self) -> None:
        """Close admission without draining an unhealthy child's RPCs."""

        await self._run_traffic_gate_operation(
            self._traffic_gate.close(),
            name="close-admission",
        )

    async def _reopen_traffic_gate(self) -> None:
        await self._run_traffic_gate_operation(
            self._traffic_gate.reopen(),
            name="reopen",
        )

    async def _seal_traffic_gate(self) -> None:
        await self._run_traffic_gate_operation(
            self._traffic_gate.seal(),
            name="seal",
        )

    async def _drain_traffic_gate(self) -> None:
        """Bound terminal traffic drain without detaching its exact ownership."""

        task = self._terminal_traffic_drain_task
        if task is None:
            task = asyncio.create_task(
                self._traffic_gate.drain(),
                name=f"isolated-traffic-drain:{self.name}",
            )
            self._terminal_traffic_drain_task = task

            def release(completed_task: asyncio.Task[None]) -> None:
                _consume_late_lifecycle_task_outcome(completed_task)
                if self._terminal_traffic_drain_task is completed_task:
                    self._terminal_traffic_drain_task = None

            task.add_done_callback(release)

        try:
            await asyncio.wait_for(
                asyncio.shield(task), timeout=_TERMINAL_TRAFFIC_DRAIN_TIMEOUT
            )
        except asyncio.TimeoutError as exc:
            raise _TerminalTrafficDrainTimedOut(
                "isolated feature traffic did not drain after stop"
            ) from exc

    async def _reset_traffic_gate_after_initialize(self) -> None:
        await self._run_traffic_gate_operation(
            self._traffic_gate.reset_and_reopen(),
            name="initialize",
        )

    async def _connect_client(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        register_channel_bridge: bool = True,
    ) -> None:
        """Build + start the isolated client from its effective config, then
        wire event handling, tools, and the channel bridge.

        Shared by ``initialize`` (first launch) and ``reload`` (re-launch after a
        config change). A forced reconciliation can start a child with an
        explicit authoritative config before ``_host_config`` is updated, so
        the bridge must receive that same effective config rather than reading
        the stale host cache.
        """
        self._assert_child_start_allowed()
        effective_config = self._host_config if config is None else config
        client, tools = await self._start_detached_client(effective_config)
        # A terminal cleanup can latch while a detached start awaits.  Do not
        # publish that child behind a sealed gate; retire it before reporting
        # the terminal lifecycle boundary to the caller.
        if self._terminal_lifecycle_latched:
            await self._retire_detached_client(client)
            self._assert_child_start_allowed()
        self._publish_client(
            client,
            tools,
            register_channel_bridge=register_channel_bridge,
            channel_config=effective_config,
        )
        # Publishing a new generation is the single construction boundary for
        # initialize, reload, supervisor recovery, and idle wake.
        self._idle_retired = False
        self._idle_resume_event.set()
        try:
            await self._register_event_handler(client)
        except BaseException as exc:
            # A client whose event registration failed must not remain
            # reachable through host tools while its caller unwinds.  Shutdown
            # may already have unpublished and retired this exact client while
            # registration was awaited; only the caller that actually removed
            # it from publication owns this additional retirement attempt.
            unpublished_client = self._unpublish_client(client)
            if unpublished_client is client:
                await self._retire_detached_client(unpublished_client)
            if isinstance(
                exc,
                (
                    IsolatedRuntimeNamespaceError,
                    IsolatedRuntimeConfigurationError,
                    IsolatedRuntimePreparationError,
                    _TerminalLifecyclePermitRevoked,
                    asyncio.CancelledError,
                ),
            ):
                raise
            if isinstance(
                exc,
                (OSError, subprocess.SubprocessError, RuntimeError, ProtocolError),
            ):
                raise IsolatedRuntimePreparationError(
                    "Isolated feature child event registration could not be "
                    "prepared."
                ) from exc
            raise
        # Registration is an awaited post-publication operation.  Shutdown may
        # have latched while it was in flight and be waiting for this lifecycle
        # owner to release ``_reload_lock``.  Do not let that terminal cycle
        # reset admission, start supervision, or report initialization success.
        self._assert_child_start_allowed()

    async def _refresh_published_client_inventory(self) -> None:
        """Republish tools and channel capability after a live apply.

        A negotiated ``applied`` hook keeps the same process, but its config can
        change which tools it advertises and which channel adapter should be
        registered.  The caller holds both the reload lock and the closed
        traffic gate, so the replacement inventory becomes visible as one host
        state before any new call or event is admitted.
        """

        client = self._client
        if client is None:
            raise RuntimeError("isolated feature client is unavailable after live config apply")
        advertised_tools = await _maybe_await(client.list_tools())
        refreshed_tools = [IsolatedFeatureTool(self, meta) for meta in advertised_tools]
        if self._client is not client:
            raise RuntimeError("isolated feature client changed during inventory refresh")
        self._unregister_channel_bridge()
        self._tools = refreshed_tools
        self._register_channel_bridge()

    async def _start_detached_client(
        self, config: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, List[AgentTool]]:
        """Start a child without making it reachable through this proxy.

        Keep startup and host publication separate so a newly started child is
        never reachable through tools, channel bridges, or event handlers until
        its caller has established the authoritative lifecycle state.
        """

        child_config = self._host_config if config is None else config
        try:
            client = self._build_client(config=child_config)
        except (
            IsolatedRuntimeNamespaceError,
            IsolatedRuntimeConfigurationError,
            IsolatedRuntimePreparationError,
            _TerminalLifecyclePermitRevoked,
        ):
            raise
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise IsolatedRuntimePreparationError(
                "Isolated feature child process could not be prepared."
            ) from exc
        try:
            started = asyncio.get_running_loop().time()
            await _maybe_await(client.start())
            self._capture_process_identity(client)
            advertised_tools = await _maybe_await(client.list_tools())
            self._last_cold_start_seconds = asyncio.get_running_loop().time() - started
        except BaseException as exc:
            await self._retire_detached_client(client)
            if isinstance(
                exc,
                (
                    IsolatedRuntimeNamespaceError,
                    IsolatedRuntimeConfigurationError,
                    IsolatedRuntimePreparationError,
                    _TerminalLifecyclePermitRevoked,
                    asyncio.CancelledError,
                ),
            ):
                raise
            if isinstance(
                exc,
                (OSError, subprocess.SubprocessError, RuntimeError, ProtocolError),
            ):
                raise IsolatedRuntimePreparationError(
                    "Isolated feature child process could not start or advertise "
                    "its runtime contract."
                ) from exc
            raise
        return client, [IsolatedFeatureTool(self, meta) for meta in advertised_tools]

    def _publish_client(
        self,
        client: Any,
        tools: List[AgentTool],
        *,
        register_channel_bridge: bool,
        channel_config: Dict[str, Any],
    ) -> None:
        """Atomically make a started child available to host traffic.

        Callers hold ``_reload_lock`` whenever replacing a live child.  The
        paired assignments deliberately happen before any event registration:
        once an event can enter the host, the child and its advertised tools are
        already the single live proxy state.
        """

        self._client = client
        self._tools = tools
        self._idle_ui_contributions = self._ui_contributions_from_capabilities(
            self._client_capabilities()
        )
        if not self._terminal_lifecycle_latched and not self._stopping:
            self._terminal_cleanup_uncertain = False
        if register_channel_bridge:
            self._register_channel_bridge(channel_config)

    def _unpublish_client(self, expected_client: Any = None) -> Any:
        """Remove the current child from host-visible proxy state.

        ``expected_client`` prevents an error path for an old detached child
        from removing a newer restored child.
        """

        if expected_client is not None and self._client is not expected_client:
            return None
        self._unregister_channel_bridge()
        client = self._client
        self._client = None
        self._tools = []
        return client

    async def _retire_detached_client(self, client: Any) -> bool:
        """Attempt neutral retirement of a child which was never published.

        Startup and registration failures still need the exact child handle if
        a hostile facade raises before its RPC/process is truly terminal.  The
        caller's original failure remains its own public result; this helper
        only records whether a later terminal cleanup must retry the child.
        """

        self._retain_terminal_retirement_client(client)
        return await self._retire_terminal_clients()

    def _retain_terminal_retirement_client(self, client: Any) -> None:
        """Keep one unpublished client reachable until terminal stop succeeds.

        Identity, rather than equality, is the lifecycle fence: an application
        facade may implement hostile or stateful equality, and terminal cleanup
        must retry the exact selected client rather than whatever a later
        lifecycle operation happens to publish.
        """

        if client is None:
            return
        if not any(
            candidate is client for candidate in self._terminal_retirement_clients
        ):
            self._terminal_retirement_clients.append(client)

    def _release_terminal_retirement_client(self, client: Any) -> None:
        """Forget only the exact client whose terminal stop completed."""

        self._terminal_retirement_clients = [
            candidate
            for candidate in self._terminal_retirement_clients
            if candidate is not client
        ]

    def _new_terminal_stop_completion_marker(
        self, client: Any
    ) -> _TerminalStopCompletionMarker | None:
        """Build one identity-safe supervisor stop-completion marker."""

        if client is None:
            return None
        try:
            return _TerminalStopCompletionMarker(weak_client=weakref.ref(client))
        except TypeError:
            # A non-weakrefable facade needs a strong reference only while its
            # exact completion disposition is unresolved.  The marker is
            # removed on restart, terminal retirement, or failed stop.
            return _TerminalStopCompletionMarker(strong_client=client)

    def _begin_terminal_stop_completion(
        self, client: Any
    ) -> _TerminalStopCompletionMarker | None:
        """Register a supervisor stop before its completion callback can race."""

        marker = self._new_terminal_stop_completion_marker(client)
        if marker is not None:
            self._terminal_stop_completed_client_markers.append(marker)
        return marker

    def _discard_terminal_stop_completion(
        self, marker: _TerminalStopCompletionMarker | None
    ) -> None:
        """Drop one abandoned in-flight completion ownership record."""

        if marker is None:
            return
        self._terminal_stop_completed_client_markers = [
            candidate
            for candidate in self._terminal_stop_completed_client_markers
            if candidate is not marker
        ]

    def _fence_terminal_stop_completion_timeout(
        self,
        client: Any,
        marker: _TerminalStopCompletionMarker | None,
    ) -> None:
        """Discard an abandoned completion record before fencing its facade."""

        self._discard_terminal_stop_completion(marker)
        self._fence_terminal_retirement_timeout(client)

    def _mark_terminal_stop_completed(
        self, marker_or_client: _TerminalStopCompletionMarker | Any,
    ) -> None:
        """Finish a registered stop, or record a direct completed stop for tests.

        A registered marker is the only supervisor callback path.  If terminal
        cleanup claimed it while stop was in flight, completion releases that
        cleanup's retained exact client and disappears; it must not leave a
        marker that could be consumed by a later reuse of the same facade.
        """

        if isinstance(marker_or_client, _TerminalStopCompletionMarker):
            marker = marker_or_client
            if not any(
                candidate is marker
                for candidate in self._terminal_stop_completed_client_markers
            ):
                return
            client = marker.client()
            if marker.terminal_retirement_claimed:
                self._release_terminal_retirement_client(client)
                self._discard_terminal_stop_completion(marker)
                return
            marker.completed = True
            return

        # Preserve the narrow direct helper used by focused identity tests.
        # Production supervisor stops always use the registered path above.
        client = marker_or_client
        self._forget_terminal_stop_completion(client)
        marker = self._new_terminal_stop_completion_marker(client)
        if marker is not None:
            marker.completed = True
            self._terminal_stop_completed_client_markers.append(marker)

    def _forget_terminal_stop_completion(
        self,
        client: Any,
        *,
        terminal_retirement: bool = False,
    ) -> bool:
        """Consume a completed stop or claim its in-flight terminal handoff.

        A terminal caller that arrives while the supervisor owns ``stop()``
        keeps the pending marker alive and retains the exact client.  The
        callback then releases that retention rather than recreating a stale
        completion marker after this cleanup pass has already consumed it.
        Non-terminal restart paths revoke any pending/completed marker so it
        cannot apply to a facade reused by a new lifecycle generation.
        """

        found = False
        retained_markers: list[_TerminalStopCompletionMarker] = []
        for marker in self._terminal_stop_completed_client_markers:
            marked_client = marker.client()
            if marked_client is None:
                # Dead marker cannot identify any future object, including an
                # object which happens to reuse its former id().
                continue
            if marked_client is not client:
                retained_markers.append(marker)
                continue
            if marker.completed:
                found = True
                continue
            if terminal_retirement:
                marker.terminal_retirement_claimed = True
                retained_markers.append(marker)
            # A normal restart revokes an in-flight marker. Its callback may
            # no longer affect this facade's next lifecycle generation.
        self._terminal_stop_completed_client_markers = retained_markers
        return found

    def _fence_terminal_retirement_timeout(self, client: Any) -> None:
        """Retain an exact facade whose bounded stop outcome is uncertain."""

        self._retain_terminal_retirement_client(client)
        self._terminal_cleanup_uncertain = True

    def _retain_terminal_lifecycle_task(
        self,
        task: _HostOwnedFacadeOperation,
        client: Any,
    ) -> None:
        """Own a still-running exact facade task until it is consumed.

        The task is registered before its timeout caller is released. Its own
        stop coroutine retains the exact client while running; this record
        stores only a weak identity marker, avoiding a second secret-bearing
        client reference while still preventing a concurrent retry.
        """

        if any(candidate.task is task for candidate in self._terminal_lifecycle_tasks):
            return
        try:
            client_ref: weakref.ReferenceType[Any] | None = weakref.ref(client)
        except TypeError:
            client_ref = None
        ownership = _TrackedFacadeLifecycleTask(task, id(client), client_ref)
        self._terminal_lifecycle_tasks.append(ownership)

        def release(completed_task: _HostOwnedFacadeOperation) -> None:
            _consume_late_lifecycle_task_outcome(completed_task)
            self._terminal_lifecycle_tasks = [
                candidate
                for candidate in self._terminal_lifecycle_tasks
                if candidate.task is not completed_task
            ]

        task.add_done_callback(release)

    def _own_health_probe_task(self, task: _HostOwnedFacadeOperation) -> None:
        """Own one exact health task from creation until its outcome is consumed.

        The task itself retains the exact facade while its bound ``health``
        coroutine is running.  Retaining only the task avoids a second
        credential-bearing facade reference, while the completion callback
        consumes exceptions and drops that final reference promptly.
        """

        if task.done():
            _consume_late_lifecycle_task_outcome(task)
            return
        if self._terminal_health_probe_task is task:
            return
        if self._terminal_health_probe_task is not None:
            raise RuntimeError("isolated feature already owns a live health probe")
        self._terminal_health_probe_task = task

        def release(completed_task: _HostOwnedFacadeOperation) -> None:
            _consume_late_lifecycle_task_outcome(completed_task)
            if self._terminal_health_probe_task is completed_task:
                self._terminal_health_probe_task = None

        task.add_done_callback(release)

    def _retain_terminal_health_probe_task(self, task: _HostOwnedFacadeOperation) -> None:
        """Mark a still-running owned health task as terminally incomplete."""

        self._own_health_probe_task(task)
        if not task.done():
            self._terminal_cleanup_uncertain = True

    def _has_running_terminal_health_probe_task(self) -> bool:
        """Return whether terminal state still owns an unsettled health call."""

        task = self._terminal_health_probe_task
        if task is None:
            return False
        if not task.done():
            return True
        _consume_late_lifecycle_task_outcome(task)
        if self._terminal_health_probe_task is task:
            self._terminal_health_probe_task = None
        return False

    async def _cancel_terminal_health_probe(self) -> bool:
        """Request and boundedly acknowledge terminal health-probe cancellation.

        This is deliberately outside reload and retirement locks. A direct
        shutdown gets the same short cooperative acknowledgement afforded to
        facade lifecycle work; an agent fair-share cancellation interrupts the
        shared cleanup while the exact task remains attached to this proxy.
        """

        task = self._terminal_health_probe_task
        if task is None:
            return True
        if task.done():
            self._has_running_terminal_health_probe_task()
            return True

        task.cancel()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _HEALTH_PROBE_CANCELLATION_GRACE
        while not task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                self._terminal_cleanup_uncertain = True
                return False
            try:
                await asyncio.wait_for(task.shield(), timeout=remaining)
            except asyncio.CancelledError:
                if task.done() and task.cancelled():
                    break
                self._terminal_cleanup_uncertain = True
                raise
            except asyncio.TimeoutError:
                continue
            except _CrossLoopFacadeOperationError:
                # A foreign health Future can only be settled by its owner
                # loop.  Its cancellation request is merely best effort until
                # that owner acknowledges settlement, so no terminal client
                # stop may race the still-retained health operation.
                self._terminal_cleanup_uncertain = True
                return False
            except BaseException:  # noqa: BLE001 - terminal probe outcome stays private
                # The cancellation request settled the exact probe, but a
                # facade may then publish an ordinary terminal failure instead
                # of CancelledError.  It is not cancellation of this cleanup
                # owner; consume it and continue with client retirement.
                _consume_late_lifecycle_task_outcome(task)
                break

        self._has_running_terminal_health_probe_task()
        return True

    async def _cancel_terminal_event_ingress_tasks(self) -> bool:
        """Fence inbound, ACK, and deferred-ingress workers before retirement.

        A terminal latch may cancel either kind of worker, but cancellation is
        only a request.  Retiring the exact facade before those tasks settle
        could let one resume on a foreign loop and issue a late acknowledgement
        or route a deferred body after terminal ownership was reported.  Keep
        cleanup incomplete when the bounded join cannot prove settlement.
        """

        current = asyncio.current_task()
        tasks = tuple(
            task
            for task in (
                *self._event_ack_tasks,
                *self._event_ingress_tasks,
                *self._deferred_acknowledged_event_tasks,
            )
            if task is not current and not task.done()
        )
        for task in tasks:
            task.cancel()
        if not tasks:
            return True
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=_EVENT_INGRESS_ACK_CANCELLATION_GRACE,
            )
        except asyncio.TimeoutError:
            self._terminal_cleanup_uncertain = True
            return False
        return True

    def _has_running_terminal_lifecycle_task(self, client: Any) -> bool:
        """Whether a prior exact stop still owns this facade's process handle."""

        running = False
        retained_tasks: list[_TrackedFacadeLifecycleTask] = []
        for ownership in self._terminal_lifecycle_tasks:
            task = ownership.task
            if task.done():
                if (
                    task.foreign_loop
                    and not task.settlement_delivery_complete
                ):
                    # Owner-loop acknowledgement has safely classified the
                    # foreign result, but the host callback which releases a
                    # successful client or fences a failed one has not run.
                    # Retaining this exact record closes the window in which a
                    # new shutdown could issue a duplicate ``stop()``.
                    marked_client = (
                        ownership.client_ref()
                        if ownership.client_ref is not None
                        else None
                    )
                    if marked_client is client or (
                        ownership.client_ref is None
                        and ownership.client_id == id(client)
                    ):
                        running = True
                    retained_tasks.append(ownership)
                    continue
                _consume_late_lifecycle_task_outcome(task)
                continue
            marked_client = (
                ownership.client_ref() if ownership.client_ref is not None else None
            )
            # A weak marker gives exact identity. For a non-weakrefable facade,
            # its still-running bound stop coroutine necessarily retains the
            # object, so an equal id cannot yet be reused.
            if marked_client is client or (
                ownership.client_ref is None and ownership.client_id == id(client)
            ):
                if task.foreign_loop:
                    # This exact operation may have been retained while its
                    # owner loop was stopped.  Retry cancellation and outcome
                    # acknowledgement on a loop that has since restarted;
                    # never issue a second facade ``stop()`` beside it.
                    task.retry_owner_loop_cancellation_and_observation()
                running = True
            retained_tasks.append(ownership)
        self._terminal_lifecycle_tasks = retained_tasks
        return running

    async def _retire_terminal_clients(self) -> bool:
        """Attempt exact retained-client retirement without public policy.

        This is the only shared terminal ``stop()`` path.  It catches every
        stop-side ``BaseException`` so the shared task never stores an
        untrusted exception object or lets the first caller's reporting policy
        choose another caller's result.  A failed stop leaves its exact handle
        retained and returns ``False``; no caller may drain traffic from that
        uncertain child.  A later cleanup attempt retries independently.
        """

        async with self._terminal_retirement_lock:
            # A detached startup can retain another client while an existing
            # stop is awaited.  Keep selecting from the live identity list so
            # a successful pass cannot report retirement while that late exact
            # handle is still waiting behind this lock.
            while self._terminal_retirement_clients:
                client = self._terminal_retirement_clients[0]
                if self._has_running_terminal_lifecycle_task(client):
                    # A timeout caller was released only after registering
                    # this exact running stop. Retrying beside it could race
                    # the facade's sole subprocess handle, so remain sealed
                    # and require its completion before a fresh retry.
                    self._terminal_cleanup_uncertain = True
                    return False
                try:
                    await _await_owned_facade_lifecycle_operation(
                        client.stop(),
                        name=f"isolated-terminal-stop:{self.name}",
                        on_completed=lambda client=client: self._release_terminal_retirement_client(
                            client
                        ),
                        on_timeout=lambda client=client: self._fence_terminal_retirement_timeout(
                            client
                        ),
                        on_late_task=lambda task, client=client: self._retain_terminal_lifecycle_task(
                            task, client
                        ),
                    )
                except asyncio.CancelledError:
                    # The owned stop task has already reached its terminal
                    # process state.  Let the caller replay its original
                    # cancellation rather than rewriting it as a stop failure.
                    raise
                except _FacadeLifecycleOperationTimedOut:
                    logger.error(
                        "Isolated feature %s terminal stop exceeded its bounded "
                        "facade lifecycle budget; the proxy remains sealed for retry",
                        self.name,
                    )
                    return False
                except BaseException:  # noqa: BLE001 - hostile stop can raise non-Exception
                    self._terminal_cleanup_uncertain = True
                    logger.error(
                        "Isolated feature %s could not stop its terminal client; "
                        "the proxy remains sealed for retry",
                        self.name,
                    )
                    return False
        return True

    async def reload(self) -> None:
        """Restart the isolated service so the current ``_host_config`` takes
        effect (config is forwarded only at the initialize handshake, so a live
        config change requires a re-launch). Guarded so the health supervisor
        doesn't treat the intentional stop as a crash and double-restart."""
        async with self._reload_lock:
            self._begin_reload()
            # Set this before the await: a cancelled drain still closed the
            # gate and must take the matching final boundary below.
            gate_closed = True
            replacement_started = False
            try:
                await self._close_traffic_gate()
                replacement_started = True
                await self._replace_client()
            except BaseException:
                # ``_replace_client`` retires the old child before starting a
                # candidate.  If the candidate fails or this reload is
                # cancelled, reopening to the old (now stopped) child would
                # make its stale tools callable.  Terminal quarantine is the
                # only honest outcome until a later explicit initialize builds
                # a coherent child from durable config.
                # Cancellation while the gate is merely draining has not
                # touched the old child, so it can still safely reopen.  Once
                # replacement begins, however, publication is removed before
                # the first stop await and only terminal quarantine is honest.
                if replacement_started:
                    await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
                raise
            finally:
                self._end_reload()
                if gate_closed:
                    if self._stopping or self._client is None:
                        await self._seal_traffic_gate()
                    else:
                        await self._reopen_traffic_gate()

    def _begin_reload(self) -> None:
        """Fence health supervision while this proxy owns client lifecycle."""

        self._reloading = True
        self._reload_gen += 1

    def _end_reload(self) -> None:
        self._reloading = False

    async def _replace_client(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        register_channel_bridge: bool = True,
    ) -> None:
        """Stop the current child and start one using ``_host_config``.

        Callers must already hold ``_reload_lock`` and have established whether
        an advertised config-transition lifecycle hook must run first.

        ``config`` selects an explicit effective config for a normal
        replacement. Fenced recovery calls this only after its pending
        generation has been durably promoted (or after it has read another
        authoritative active generation).
        """

        self._assert_child_start_allowed()
        # The old child must lose every host-visible handle before a stop that
        # can succeed while candidate startup fails.  Restoring it only on a
        # stop failure gives terminal quarantine one last best-effort retirement
        # handle; it never restores its tools or channel bridge.
        previous_client = self._unpublish_client()
        previous_stop_completed = False
        if previous_client is not None:
            def remember_previous_stop() -> None:
                nonlocal previous_stop_completed

                previous_stop_completed = True

            try:
                await _await_owned_facade_lifecycle_operation(
                    previous_client.stop(),
                    name=f"isolated-replace-stop:{self.name}",
                    on_completed=remember_previous_stop,
                    on_timeout=lambda previous_client=previous_client: self._fence_terminal_retirement_timeout(
                        previous_client
                    ),
                    on_late_task=lambda task, previous_client=previous_client: self._retain_terminal_lifecycle_task(
                        task, previous_client
                    ),
                )
            except BaseException:
                # The helper replays caller cancellation after recording a
                # successful exact stop.  Such a facade is irreversibly
                # retired: restoring it would republish stale tools and the
                # ensuing quarantine would issue a duplicate stop.  A failed
                # or incomplete stop remains intentionally restorable so the
                # terminal path keeps its exact retry handle.
                if not previous_stop_completed:
                    self._client = previous_client
                raise
        await self._connect_client(
            config,
            register_channel_bridge=register_channel_bridge,
        )

    async def shutdown(self):
        # Set the latch before scheduling the transaction so a health probe
        # cannot decide to restart the child in the tiny interval before seal.
        self._latch_terminal_lifecycle()
        await self._complete_terminal_cleanup()

    def prepare_shutdown_with_agent_deadline(self) -> None:
        """Synchronously establish terminal ownership for agent shutdown.

        ``asyncio.wait_for(..., timeout=0)`` may cancel a coroutine before its
        body runs.  The agent calls this synchronous preparation hook before
        applying its fair-share timeout, so the terminal latch and one owned
        cleanup task exist even when this proxy receives no execution slice.
        """

        self._latch_terminal_lifecycle()
        self._terminal_cleanup_uncertain = False
        task = self._terminal_cleanup_task
        if task is None or task.done():
            self._terminal_cleanup_task = asyncio.create_task(
                self._terminal_cleanup_uninterrupted(lifecycle_lock_held=False),
                name=f"isolated-terminal-cleanup:{self.name}",
            )

    async def shutdown_with_agent_deadline(self) -> None:
        """Shutdown under KestrelAgent's bounded fair-share lifecycle owner.

        The agent's ``wait_for`` remains the sole deadline authority.  If it
        expires, ``_terminal_cleanup_task`` stays owned by this proxy and can
        finish the SDK's documented subprocess retirement path independently;
        the feature sweep can still continue to later features and its durable
        tail without detaching that exact process handle.
        """

        # Direct callers retain the normal shutdown contract.  KestrelAgent
        # has already prepared the shared task synchronously before it applies
        # its fair-share timeout.
        if self._terminal_cleanup_task is None:
            self.prepare_shutdown_with_agent_deadline()
        token = _AGENT_SHUTDOWN_DEADLINE_ACTIVE.set(True)
        try:
            await self._complete_terminal_cleanup()
        finally:
            _AGENT_SHUTDOWN_DEADLINE_ACTIVE.reset(token)

    async def _complete_terminal_cleanup(
        self,
        *,
        best_effort: bool = False,
        lifecycle_lock_held: bool = False,
    ) -> None:
        """Finish neutral teardown, then apply this caller's own policy.

        The shared/owned task records only terminal state: exact retained
        clients, publication fences, and gate state.  It never propagates a
        facade ``stop()`` exception.  Once it settles, explicit lifecycle
        callers independently reject incomplete retirement while best-effort
        quarantine/supervisor callers preserve their original error or
        cancellation.  The shared task remains owned if an external caller
        reaches its deadline, so whole-agent shutdown can continue without
        cancelling an SDK stop coroutine that owns a process handle.
        """

        # An explicit lifecycle caller is a new bounded retirement attempt. A
        # best-effort supervisor/quarantine cleanup that just fenced a facade
        # timeout must instead preserve uncertainty and return promptly.
        if not best_effort:
            self._terminal_cleanup_uncertain = False

        if lifecycle_lock_held:
            # A set-config/reload owner cannot await the shared terminal task:
            # that task may already be waiting behind this very lock (for
            # example when shutdown sealed while a reload was in progress).
            # Complete this terminal portion under the existing ownership in a
            # separate shielded task instead.  It does not acquire the lock, so
            # neither the cleanup nor the original shutdown can be orphaned in
            # a lock cycle.
            task = asyncio.create_task(
                self._terminal_cleanup_uninterrupted(
                    lifecycle_lock_held=True,
                ),
                name=f"isolated-terminal-cleanup-owned:{self.name}",
            )
            # This owner-local task is not stored on the proxy.  It must drain
            # through cancellation; only the shared task in the branch below
            # can safely be handed back to KestrelAgent's deadline owner.
            await _await_task_until_complete(task, preserve_cancellation=False)
        else:
            task = self._terminal_cleanup_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._terminal_cleanup_uninterrupted(
                        lifecycle_lock_held=False,
                    ),
                    name=f"isolated-terminal-cleanup:{self.name}",
                )
                self._terminal_cleanup_task = task
            try:
                await _await_task_until_complete(
                    task,
                    preserve_cancellation=False,
                    settle_on_cancellation=not _AGENT_SHUTDOWN_DEADLINE_ACTIVE.get(),
                )
            finally:
                # A settled shared task has no policy-bearing result and must
                # not become a stale lifecycle handle.  A later explicit call
                # creates a fresh neutral attempt if any exact client remains.
                if self._terminal_cleanup_task is task and task.done():
                    self._terminal_cleanup_task = None

        if not best_effort and (
            self._terminal_cleanup_uncertain
            or self._terminal_retirement_clients
            or self._has_running_terminal_health_probe_task()
            or any(not task.done() for task in self._event_ack_tasks)
            or any(not task.done() for task in self._event_ingress_tasks)
            or any(not task.done() for task in self._deferred_acknowledged_event_tasks)
            or self._traffic_gate._active
            or self._terminal_traffic_drain_task is not None
        ):
            # Do not expose a hostile facade's message/type/traceback.  This
            # fresh public lifecycle error says only that the sealed proxy has
            # not reached a state in which success would be honest.
            raise RuntimeError("isolated feature terminal retirement is incomplete")

    async def _terminal_cleanup_uninterrupted(
        self,
        *,
        lifecycle_lock_held: bool,
    ) -> None:
        """Run one shared cleanup attempt without exporting external failures.

        A task stores its exception object and traceback until its consumer
        retrieves it.  Terminal cleanup is a shared private task, so even an
        unexpected facade/adapter/supervisor failure must become retained
        lifecycle state rather than a secret-bearing task result.  The caller
        decides whether that incomplete state is a public lifecycle failure.
        """

        try:
            await self._terminal_cleanup_attempt(
                lifecycle_lock_held=lifecycle_lock_held,
            )
        except BaseException:  # noqa: BLE001 - shared task must not retain facade failures
            # If unpublication itself failed, the sealed public slot is still
            # the only exact retry handle.  Retaining by identity is harmless
            # when a prior phase already unpublished it.
            self._retain_terminal_retirement_client(self._client)
            self._terminal_cleanup_uncertain = True
            logger.error(
                "Isolated feature %s could not complete terminal cleanup; "
                "the proxy remains sealed for retry",
                self.name,
            )

    async def _terminal_cleanup_attempt(
        self,
        *,
        lifecycle_lock_held: bool,
    ) -> None:
        """Seal, unpublish, stop, and drain without caller interruption.

        A terminal operation differs from reload/config transitions: it must
        not wait for healthy admitted work, because an admitted JSON-RPC call
        can be permanently wedged in the child.  Seal first, detach the exact
        published client, start its bounded SDK stop path, then drain the
        terminal RPC before completing lifecycle serialization.
        """

        # Seal before lifecycle ownership so finite-transition waiters become
        # terminal even while another reload currently holds the lock.
        await self._seal_traffic_gate()

        supervision_task = self._supervision_task
        if supervision_task is not None:
            self._supervision_task = None
            if supervision_task is not asyncio.current_task():
                # Do not wait here: the supervisor may currently hold the
                # reload lock while it drains the very RPC we must terminate.
                supervision_task.cancel()

        # This is intentionally outside ``_reload_lock``. A finite transition
        # can hold that lock while waiting for an admitted call; waiting behind
        # it would recreate the seal-and-drain deadlock. The terminal latch and
        # sealed gate fence publication, while unpublishing is synchronous.
        client = self._unpublish_client()
        already_stopped = self._forget_terminal_stop_completion(
            client, terminal_retirement=True
        )
        if not already_stopped:
            self._retain_terminal_retirement_client(client)
        health_probe_running = self._has_running_terminal_health_probe_task()
        if health_probe_running and not self._terminal_cleanup_uncertain:
            health_probe_running = not await self._cancel_terminal_health_probe()
        ingress_tasks_settled = await self._cancel_terminal_event_ingress_tasks()
        if (
            self._terminal_cleanup_uncertain
            or health_probe_running
            or not ingress_tasks_settled
        ):
            # A bounded facade stop did not establish whether its child retired,
            # or a cancellation-resistant health call still owns the facade.
            # Keep the exact private handle and leave admitted RPCs undrained:
            # waiting for them could recreate the very hang the stop fence
            # avoided. A later explicit lifecycle call gets a fresh bounded
            # retry; this terminal path reports no false success or reopen.
            self._terminal_cleanup_uncertain = True
            return
        if not await self._retire_terminal_clients():
            # Do not drain beneath a stop failure: a hostile facade can raise
            # while its admitted RPC is still wedged.  The retained exact
            # handle makes a later terminal attempt bounded and recoverable.
            return

        async def retire_late_publication() -> bool:
            # A detached candidate checks the terminal latch before publish,
            # but this final fence also covers a client published just before
            # shutdown latched while its registration await was in flight.
            late_client = self._unpublish_client()
            if not self._forget_terminal_stop_completion(
                late_client, terminal_retirement=True
            ):
                self._retain_terminal_retirement_client(late_client)
            return await self._retire_terminal_clients()

        # Keep the retirement lock out of every wait below.  A supervisor can
        # hold ``_reload_lock`` while cancellation sends it through an owned
        # cleanup; that owner may await this narrow stop lock.  This shared
        # cleanup releases it before awaiting ``_reload_lock``, and both paths
        # release it before gate drain or supervisor join, so there is no
        # reload/retirement/drain cycle.
        if lifecycle_lock_held:
            late_retired = await retire_late_publication()
        else:
            async with self._reload_lock:
                late_retired = await retire_late_publication()

        if not late_retired:
            return

        # SDK stop owns the bounded shutdown/terminate/kill path. Only after
        # every pre-existing and late-published client has completed stop may
        # we *boundedly* wait for an admitted host call to release the gate.
        # A legacy facade that lied about stop success leaves the sealed drain
        # task attached and makes this attempt incomplete rather than hanging
        # the whole-agent lifecycle or reporting false success.
        await self._drain_traffic_gate()

        if (
            supervision_task is not None
            and supervision_task is not asyncio.current_task()
        ):
            try:
                await supervision_task
            except asyncio.CancelledError:
                pass
        # A cancelled supervisor may have handed us a health task which keeps
        # the old facade alive.  Do not report terminal completion or permit a
        # re-enable until that exact task has settled and its callback consumed
        # its outcome.  This check stays outside reload/retirement locks.
        self._terminal_cleanup_uncertain = (
            self._has_running_terminal_health_probe_task()
        )

    def get_tools(self) -> List[AgentTool]:
        return list(self._tools)

    def get_router(self):
        if self._client is not None and hasattr(self._client, "get_router"):
            return self._client.get_router()
        return None

    def get_ui_contributions(self) -> Optional[UIContributions]:
        """Forward UI contributions an isolated service reports over the SDK
        init handshake (design option (a) of ticket #2043).

        The out-of-process service advertises its UI assets in its
        ``initialize`` capabilities under ``ui_contributions`` — modules/css
        plus an absolute ``static_dir`` that lives on the same host. The host
        then mounts and serves those assets through the same single asset path
        as in-process features, so isolated-venv features can contribute UI
        without the host proxying every static request.
        """
        if self._client is None:
            return self._idle_ui_contributions if self._idle_retired else None
        contribution = self._ui_contributions_from_capabilities(
            self._client_capabilities()
        )
        self._idle_ui_contributions = contribution
        return contribution

    @staticmethod
    def _ui_contributions_from_capabilities(
        caps: Dict[str, Any],
    ) -> Optional[UIContributions]:
        """Parse one immutable host-owned view of child UI metadata."""

        ui = caps.get("ui_contributions") or caps.get("ui")
        if not isinstance(ui, dict):
            return None
        modules = ui.get("modules")
        if not isinstance(modules, list) or not modules:
            return None
        css = ui.get("css")
        return UIContributions(
            modules=[str(m) for m in modules],
            css=[str(c) for c in css] if isinstance(css, list) else [],
            static_dir=ui.get("static_dir"),
            capability=ui.get("capability"),
        )

    async def get_config(self) -> Dict:
        """Return the feature's current host config.

        The SDK client exposes no ``get_config`` (config only flows host→service
        at initialize), so read from the in-memory host config, falling back to
        the resolved durable config node — NOT an empty passthrough,
        which made the config API/UI show blank and drop write-only secrets on a
        partial PATCH (#2214).
        """
        # Hosted replicas must not answer from their process-local cache: a
        # second replica can commit a credential rotation after this proxy
        # initialized.  In particular, cached ``{}`` is a valid loaded config,
        # not evidence that no durable config exists.  A pending transition
        # deliberately exposes its *active* config here; the candidate remains
        # private until promotion.
        storage = getattr(self.agent, "storage", None)
        get_node = getattr(storage, "get_node", None) if storage is not None else None
        if (
            storage is not None
            and inspect.iscoroutinefunction(get_node)
            and await self._persistent_config_writes_allowed(storage)
        ):
            state = await self._read_config_state(storage)
            # A durable read can describe a winner from another replica while
            # this process still has a child running an older config.  Return
            # the durable value, but retain the locally applied identity until
            # reconciliation replaces or live-applies that child.  Otherwise a
            # later PATCH can mistake the stale child for the durable winner
            # and invoke a transition hook against stale resources.
            if self._client is None:
                self._host_config = dict(state.config)
                self._host_config_loaded = True
            return dict(state.config)

        # Volatile privacy mode intentionally has no durable node.  Its local
        # empty config is still distinct from an unloaded state.
        await self._ensure_host_config_loaded()
        return dict(self._host_config)

    async def set_config(
        self,
        config: Dict,
        *,
        _preserve_secret_fields: set[str] | None = None,
        _validate_effective_config: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        """Persist an effective config and apply it to the running service.

        The previous implementation forwarded to ``self._client.set_config`` —
        which the SDK client does not implement — so config set via the API/UI
        was silently dropped: never persisted (so lost on restart) and never
        applied (#2214). Persist to the resolved durable graph node (the same
        authority ``_load_host_config`` reads at startup). A service that
        advertises the SDK config-transition capability receives the full next
        effective config while it still owns the old one. Its successful typed
        result decides whether to replace the process or retain it after a
        live apply. Legacy services retain the existing safe replacement path.

        The candidate is durably staged as ``pending_config`` while the active
        ``config`` remains unchanged.  This is deliberately not implemented
        through :meth:`Feature.persist_config`: that compatibility helper
        swallows storage failures, whereas an apply may only continue after a
        durable write succeeds (apart from an intentional volatile-privacy
        no-op). A failed hook conditionally removes its own pending generation;
        if storage cannot prove that cleanup, the local proxy is quarantined
        rather than left reachable on an uncertain config.

        Each durable update is a generation-owned ``stage → promote`` protocol:
        a conditional graph write stages the candidate, and a second conditional
        write promotes only that exact pending generation.  A write exception is
        not evidence of rollback — cloud storage can commit before a connection
        breaks or a task is cancelled — so every uncertain promotion is re-read
        and the child is reconciled to that authoritative state.  In particular,
        fenced SDK recovery promotes before it starts a replacement child: child
        startup may create external resources and must never run ahead of the
        durable active configuration.
        """
        cfg = dict(config) if isinstance(config, dict) else {}
        # Idle retirement removes only the process, not the feature's lifecycle
        # contract. Recreate that child before staging so a negotiated
        # transition validator cannot be bypassed by ``_client is None``.
        if self._idle_retired and not self._terminal_lifecycle_latched:
            await self._wake_idle_runtime()
        async with self._reload_lock:
            if self._terminal_lifecycle_latched:
                await self._persist_terminal_config(
                    cfg,
                    preserve_secret_fields=_preserve_secret_fields,
                    validate_effective_config=_validate_effective_config,
                )
                return
            if self._idle_retired:
                # A very short idle deadline can retire again between the wake
                # and this lock acquisition. Refuse this attempt rather than
                # promoting without the child-owned transition hook.
                raise IsolatedRuntimePreparationError(
                    "Isolated feature became idle before config transition."
                )
            self._begin_reload()
            # The gate closes only after an opt-in external producer has
            # acknowledged that it cannot emit another callback. A cancelled
            # close/drain still needs the final reopen/seal below, but staging
            # and producer quiescence deliberately happen while old traffic is
            # still admissible.
            gate_closed = False
            self._fenced_recovery_failed = False
            transition_attempted = False
            transition_succeeded = False
            transition: Optional[_ConfigTransition] = None
            promotion: Optional[_PromotionResolution] = None
            transition_settled = False
            lifecycle_result: Optional[ConfigTransitionResult] = None
            local_authoritative = False
            external_ingress_quiesce: _ExternalIngressQuiesce | None = None
            body_error: BaseException | None = None
            try:
                # A caller may invoke set_config after a failed startup or
                # before normal initialization. Reload the authoritative
                # durable value first; otherwise a partial PATCH could stage
                # an empty config over a write-only secret.
                await self._ensure_host_config_loaded()
                # Stage from a fresh graph snapshot, never the in-memory
                # cache. A Cloud Run replica that read an older config cannot
                # overwrite a newer generation through the old add-node
                # upsert path.
                transition = await self._stage_pending_config(
                    cfg,
                    preserve_secret_fields=_preserve_secret_fields,
                    validate_effective_config=_validate_effective_config,
                )
                # A scoped compare-and-create cannot atomically exclude an old
                # binary's independent legacy write.  Fence the exact staged
                # authority before reconciliation can do any lifecycle work;
                # the post-reconciliation lease renewal below remains the
                # final fence immediately before a live SDK hook.
                await self._assert_staged_transition_authority(transition)
                # A no-op may commit immediately only when the published child
                # is already known to be on the active durable generation. A
                # replica can have a child on A while its fresh stage reads B;
                # treating ``B -> B`` as a no-op there would cache B and leave
                # that child reachable on A indefinitely.
                reconcile_can_retire_client = (
                    self._client is not None
                    and self._host_config != transition.active_config
                )
                if (
                    transition.next_config == transition.active_config
                    and not reconcile_can_retire_client
                ):
                    promotion = await self._promote_config(transition)
                    if not promotion.committed:
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=False,
                            preserve_cancellation=False,
                        )
                        transition_settled = True
                        self._raise_promotion_failure(promotion)
                    transition_settled = True
                    self._host_config = dict(transition.next_config)
                    self._host_config_loaded = True
                    local_authoritative = True
                    if promotion.error is not None:
                        self._raise_promotion_failure(promotion)
                    return

                # Quiesce the exact external producer while Core admission is
                # still open. A callback already written into the bridge may
                # be waiting for Core's durable ACK; closing first deadlocks
                # that callback with the producer quiesce wait. The exact
                # lifecycle RPC remains identity-fenced and bypasses the
                # normal data-plane gate, so it cannot admit new work.
                external_ingress_quiesce = self._new_external_ingress_quiesce()
                if external_ingress_quiesce is not None:
                    await self._quiesce_external_ingress(external_ingress_quiesce)
                # Once the producer has confirmed its pause, close admission
                # and drain every callback that crossed the prior boundary.
                gate_closed = True
                await self._close_traffic_gate_admission()
                await self._drain_traffic_gate()

                await self._reconcile_client_to_authoritative_config(
                    transition.active_config,
                    force=False,
                )
                if (
                    self._client is None
                    or transition.next_config == transition.active_config
                ):
                    promotion = await self._promote_config(transition)
                    if not promotion.committed:
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=False,
                            preserve_cancellation=False,
                        )
                        transition_settled = True
                        self._raise_promotion_failure(promotion)
                    transition_settled = True
                    self._host_config = dict(transition.next_config)
                    self._host_config_loaded = True
                    local_authoritative = True
                    if promotion.error is not None:
                        self._raise_promotion_failure(promotion)
                    return

                if self._supports_config_transition():
                    # Staging precedes local reconciliation so every replica
                    # agrees on the active config it must restore.  That
                    # reconciliation can itself stop/start a stale child and
                    # take longer than the lease.  Re-prove this exact
                    # owner/generation *after* it finishes and before marking
                    # the SDK hook attempted; otherwise a replica that lost
                    # its expired stage could still invoke external live-apply
                    # work against a candidate it no longer owns.
                    await self._renew_transition_lease(transition)
                    transition_attempted = True
                    lifecycle_result = await self._prepare_config_transition_with_lease(
                        transition
                    )
                    transition_succeeded = True

                # A legacy service has no preparation phase; a supported
                # service reaches this point only after its preparation
                # completed.  Either way, make the next config durable before
                # exposing it through the host or launching a normal child.
                promotion = await self._promote_config(transition)
                if not promotion.committed:
                    # A negotiated hook may already have mutated the old
                    # child. Replacing it from the active config after the
                    # generation-owned abort is the only safe response.
                    await self._run_owned_transition_cleanup(
                        transition,
                        force=lifecycle_result is not None,
                        preserve_cancellation=False,
                    )
                    transition_settled = True
                    self._raise_promotion_failure(promotion)

                transition_settled = True
                self._host_config = dict(transition.next_config)
                self._host_config_loaded = True

                if lifecycle_result is None:
                    # Legacy SDK/service: no negotiated hook, so preserve the
                    # established stop-and-replace behavior.
                    await self._replace_client()
                elif lifecycle_result.action == CONFIG_TRANSITION_APPLIED:
                    # The service atomically adopted the config in-process.
                    # Its channel bridge still carries host-side config (enabled
                    # and sender filters), so refresh that forwarding adapter.
                    await self._refresh_published_client_inventory()
                else:
                    # The SDK validates result actions; the non-live outcome is
                    # the normal prepare-then-restart protocol.
                    await self._replace_client()
                local_authoritative = True
                # A connection failure/cancellation may have been raised after
                # the promote committed. The child is now coherent with the
                # durable state, but the original caller still receives its
                # transport outcome rather than a false success.
                if promotion.error is not None:
                    self._raise_promotion_failure(promotion)
            except _ConfigAuthorityChanged as authority_error:
                body_error = authority_error
                staged_transition = transition or getattr(
                    authority_error, "transition", None
                )
                await self._handle_config_authority_change(staged_transition)
                raise RuntimeError(
                    f"Cannot apply config for isolated feature {self.name}: "
                    "legacy config authority became visible during rolling upgrade"
                ) from authority_error
            except asyncio.CancelledError as cancellation_error:
                body_error = cancellation_error
                if transition is not None:
                    # Every await after staging enters this path. The cleanup
                    # task is shielded so a second cancellation cannot strand
                    # this generation; it either proves durable state and
                    # reconciles the child or quarantines the proxy itself.
                    if self._client_requires_replacement():
                        # The SDK fences a cancelled lifecycle RPC precisely
                        # because it may have reached the child.  Do not revert
                        # to the old config first: promote this generation (or
                        # prove another durable winner) before any child is
                        # started, then preserve the caller's cancellation.
                        await self._recover_fenced_transition(
                            transition,
                            asyncio.CancelledError(),
                            preserve_cancellation=True,
                        )
                    else:
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=(not local_authoritative)
                            and (
                                transition_attempted
                                or lifecycle_result is not None
                                or transition_settled
                            ),
                            preserve_cancellation=True,
                        )
                raise
            except BaseException as transition_error:
                body_error = transition_error
                if transition is not None and not transition_settled:
                    if transition_attempted and not transition_succeeded:
                        if (
                            self._client_requires_replacement()
                            or isinstance(transition_error, _ConfigTransitionLeaseLost)
                        ):
                            await self._recover_fenced_transition(
                                transition,
                                transition_error,
                            )
                        else:
                            await self._run_owned_transition_cleanup(
                                transition,
                                force=False,
                                preserve_cancellation=False,
                            )
                    else:
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=lifecycle_result is not None,
                            preserve_cancellation=False,
                        )
                elif (
                    transition is not None
                    and transition_settled
                    and promotion is not None
                    and promotion.committed
                    and not local_authoritative
                ):
                    # An await after the durable promotion (for example child
                    # replacement) failed.  Re-read/reconcile the active
                    # generation before surfacing it; no pending stage remains.
                    await self._run_owned_transition_cleanup(
                        transition,
                        force=True,
                        preserve_cancellation=False,
                    )
                raise
            finally:
                # This finalizer owns the complete resume/reopen boundary.
                # In particular, a cancelled caller cannot reopen the gate and
                # replay a deferred polling update before the exact paused
                # source has either resumed or been terminally quarantined.
                finalizer_error: BaseException | None = None
                try:
                    if self._fenced_recovery_failed:
                        await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
                    # Cancellation can arrive while the exact producer is
                    # quiescing, before the normal body reaches its Core gate
                    # close. Once that lifecycle operation has started, this
                    # finalizer owns the same close/drain -> resume/quarantine
                    # boundary; otherwise a cancelled quiesce can strand the
                    # external producer paused forever.
                    if external_ingress_quiesce is not None and not gate_closed:
                        gate_closed = True
                        await self._close_traffic_gate_admission()
                        await self._drain_traffic_gate()
                    if gate_closed:
                        await self._finalize_external_ingress_transition(
                            external_ingress_quiesce
                        )
                except BaseException as exc:  # noqa: BLE001 - body outcome wins below
                    finalizer_error = exc
                finally:
                    self._end_reload()
                if finalizer_error is not None and body_error is None:
                    raise finalizer_error

    async def _persist_terminal_config(
        self,
        config: Dict[str, Any],
        *,
        preserve_secret_fields: set[str] | None,
        validate_effective_config: Callable[[Dict[str, Any]], None] | None,
    ) -> None:
        """Durably repair config without reviving a terminal enable cycle.

        A loaded soft-disabled feature retains its proxy so the config API can
        rotate credentials before a later re-enable.  Its traffic gate is
        terminal by design, however, so this path performs only the same
        generation-owned stage/promote protocol; it never closes or reopens
        admission, invokes a child hook, or starts a child.
        """

        transition: Optional[_ConfigTransition] = None
        transition_settled = False
        promotion: Optional[_PromotionResolution] = None
        try:
            # The volatile path preserves omitted fields from this cache.  For
            # durable stores the stage always derives from a fresh CAS snapshot,
            # but loading here keeps the two paths equally safe after a failed
            # startup that never reached a child.
            await self._ensure_host_config_loaded()
            transition = await self._stage_pending_config(
                config,
                preserve_secret_fields=preserve_secret_fields,
                validate_effective_config=validate_effective_config,
            )
            promotion = await self._promote_config(transition)
            if not promotion.committed:
                await self._run_owned_transition_cleanup(
                    transition,
                    force=False,
                    preserve_cancellation=False,
                )
                transition_settled = True
                self._raise_promotion_failure(promotion)

            transition_settled = True
            # There is no applied child to keep in sync.  This is the durable
            # config that a later explicit initialize will load.
            self._host_config = dict(transition.next_config)
            self._host_config_loaded = True
            if promotion.error is not None:
                self._raise_promotion_failure(promotion)
        except BaseException:
            if transition is not None and not transition_settled:
                await self._run_owned_transition_cleanup(
                    transition,
                    force=False,
                    preserve_cancellation=False,
                )
            raise

    async def set_config_with_secret_preservation(
        self,
        incoming: Dict[str, Any],
        secret_fields: set[str],
        validate: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Atomically preserve omitted write-only fields for the generic API.

        The endpoint must not read a secret on one hosted replica and later
        stage that stale value over another replica's rotation.  Preservation
        is therefore performed from the very same authoritative snapshot that
        becomes the stage CAS predicate; validation also runs on that effective
        value before the lifecycle hook can observe it.
        """

        await self.set_config(
            incoming,
            _preserve_secret_fields=set(secret_fields),
            _validate_effective_config=validate,
        )

    async def _run_owned_transition_cleanup(
        self,
        transition: _ConfigTransition,
        *,
        force: bool,
        preserve_cancellation: bool,
    ) -> None:
        """Run owned abort/reconciliation while protecting its durable outcome.

        The cleanup task either proves durable state and reconciles the child or
        quarantines the proxy.  Its caller chooses whether a new cancellation
        should immediately propagate (ordinary exception unwinding) or defer to
        an already-caught original ``CancelledError``.
        """

        async def cleanup() -> None:
            try:
                await self._abort_and_reconcile_uncommitted_transition(
                    transition,
                    force=force,
                )
            except BaseException:  # noqa: BLE001 - cancellation cleanup fences the proxy
                self._host_config_loaded = False
                await self._quarantine_unreconciled_client(lifecycle_lock_held=True)

        task = asyncio.create_task(cleanup())
        # Shielding alone is insufficient: a *second* cancellation used to let
        # this method return while ``task`` still held a durable cleanup or
        # quarantine operation, releasing ``_reload_lock`` to the next reload.
        # Keep waiting through every cancellation; the caller either re-raises
        # its original cancellation after cleanup or receives the newly caught
        # one only once the state can no longer mutate in the background.
        await _await_task_until_complete(
            task,
            preserve_cancellation=preserve_cancellation,
        )

    async def _handle_config_authority_change(
        self,
        transition: Optional[_ConfigTransition],
    ) -> None:
        """Reconcile or quarantine after legacy supersedes scoped authority.

        Before staging, the proxy can safely replace a stale scoped *live*
        child from the visible legacy active config and leave traffic closed
        until that reconciliation completes.  A clientless proxy instead
        caches the legacy active config: config-only ``set_config`` must not
        create an external child before explicit initialization.  After a
        scoped stage may have committed, deleting or promoting it would be a
        cross-key guess against an old writer, so the proxy is terminally
        quarantined instead.  The legacy row remains the authority for the
        next proxy; the orphan candidate is an explicit post-rollout
        operator-cleanup concern.
        """

        storage = transition.storage if transition is not None else getattr(
            self.agent, "storage", None
        )
        try:
            if storage is None:
                raise RuntimeError("storage is unavailable after config authority change")
            state = await self._read_config_state(storage)
            if transition is None:
                if self._client is None:
                    self._host_config = dict(state.config)
                    self._host_config_loaded = True
                    return
                await self._reconcile_client_to_authoritative_config(
                    state.config,
                    force=True,
                )
                return

            # A scoped candidate may have committed before the visible legacy
            # row was revalidated.  Cache the legacy active config for a later
            # explicit initialize, then retire the potentially divergent child.
            self._host_config = dict(state.config)
            self._host_config_loaded = True
        except BaseException:
            self._host_config_loaded = False
        await self._quarantine_unreconciled_client(lifecycle_lock_held=True)

    async def _assert_staged_transition_authority(
        self,
        transition: _ConfigTransition,
    ) -> None:
        """Prove a freshly staged generation still owns its pinned authority."""

        if not transition.persistent:
            return
        state = await self._read_config_state(
            transition.storage,
            expected_node_id=transition.config_node_id,
        )
        if not self._state_matches_pending_generation(
            state,
            generation=transition.generation,
            owner=transition.owner,
        ):
            raise _ConfigTransitionLeaseLost(
                "isolated config transition was lost immediately after staging"
            )

    async def _recover_fenced_transition(
        self,
        transition: _ConfigTransition,
        _original_error: BaseException,
        *,
        preserve_cancellation: bool = False,
    ) -> None:
        """Finish fenced recovery before releasing lifecycle ownership.

        The caller may already be unwinding a cancellation.  In that case a
        later cancellation or recovery error must not let a cleanup task escape
        behind ``_reload_lock``; quarantine is complete first and the original
        cancellation remains the public result.
        """

        # Treat recovery as unsafe until its task has completed normally.  This
        # also gives the enclosing finally a fail-closed marker if an exception
        # is raised at any await boundary below.
        self._fenced_recovery_failed = True
        task = asyncio.create_task(
            self._recover_fenced_transition_uninterrupted(transition),
            name=f"isolated-fenced-recovery:{self.name}",
        )
        try:
            await _await_task_until_complete(task, preserve_cancellation=preserve_cancellation)
        except asyncio.CancelledError:
            raise
        except BaseException as recovery_error:
            self._fenced_recovery_failed = True
            # The uninterrupted body has already tried generation-scoped
            # cleanup.  Repeat through the standard fenced cleanup path so a
            # generic failure (including old-client stop failure) cannot leave
            # our pending generation wedged.  That helper never clears another
            # owner's state and quarantines on uncertainty.
            await self._run_owned_transition_cleanup(
                transition,
                force=True,
                preserve_cancellation=True,
            )
            await self._clear_owned_pending_before_quarantine(transition)
            # A recovery operation itself failed (for example an old child
            # refused to stop or the replacement could not start).  Even if
            # durable cleanup succeeded, do not leave a client whose lifecycle
            # outcome is unknown reachable to a later supervisor restart.
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            # ``_quarantine_unreconciled_client`` is deliberately best-effort
            # about process retirement; publication itself is not best-effort.
            # Keep the host boundary closed even if a non-SDK test/client
            # implementation mutates its stop path unexpectedly.
            self._client = None
            self._tools = []
            self._latch_terminal_lifecycle()
            if preserve_cancellation:
                logger.error(
                    "Isolated feature %s fenced recovery failed; proxy was reconciled "
                    "or quarantined before cancellation propagated",
                    self.name,
                )
                return
            raise recovery_error
        self._fenced_recovery_failed = False

    async def _recover_fenced_transition_uninterrupted(
        self,
        transition: _ConfigTransition,
    ) -> None:
        """Recover an SDK-fenced hook outcome without exposing pending config.

        An SDK fence means the hook may have reached the child, so first remove
        that child from host traffic.  Promotion remains generation-scoped; if
        it cannot be proved, abort only this staged generation and restart from
        the durable active config.  No child ever starts from a pending value.
        """

        active_client = self._unpublish_client()
        active_stop_completed = False
        try:
            if active_client is not None:
                def remember_active_stop() -> None:
                    nonlocal active_stop_completed

                    active_stop_completed = True

                try:
                    await _await_owned_facade_lifecycle_operation(
                        active_client.stop(),
                        name=f"isolated-fenced-recovery-stop:{self.name}",
                        on_completed=remember_active_stop,
                        on_timeout=lambda active_client=active_client: self._fence_terminal_retirement_timeout(
                            active_client
                        ),
                        on_late_task=lambda task, active_client=active_client: self._retain_terminal_lifecycle_task(
                            task, active_client
                        ),
                    )
                except BaseException:
                    # Put it back only so the standard cleanup can retire or
                    # quarantine it.  It remains unpublished throughout.
                    if not active_stop_completed:
                        self._client = active_client
                    raise

            promotion = await self._promote_config(transition)
            if promotion.committed:
                target_config = transition.next_config
            else:
                state = await self._abort_and_reconcile_uncommitted_transition(
                    transition,
                    force=False,
                )
                # ``state.config`` is always the active value; pending_config
                # is never used to initialize a recovery child.
                target_config = state.config

            await self._connect_client(target_config)
        except BaseException:
            # A partially restored old client must be visible only to the
            # cleanup/quarantine path, never to traffic (the gate is closed).
            if (
                active_client is not None
                and not active_stop_completed
                and self._client is None
            ):
                self._client = active_client
            # In particular, a failed old-client stop must not prevent the
            # generation's durable cleanup.  This happens before quarantine so
            # a later replica does not inherit a needless pending wedge.
            await self._clear_owned_pending_before_quarantine(transition)
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            raise

        self._host_config = dict(target_config)
        self._host_config_loaded = True

    async def _clear_owned_pending_before_quarantine(
        self,
        transition: _ConfigTransition,
    ) -> None:
        """Make one last client-independent attempt to retire our stage.

        Recovery can fail while stopping the old child.  The normal cleanup
        path also reconciles that child and can therefore abort before its
        post-cleanup result is observable.  Durable pending removal must not
        depend on a process stop: this helper uses only generation-scoped CAS
        and never touches a state owned by another replica.  Any uncertainty is
        left quarantined rather than guessed away.
        """

        if not transition.persistent:
            return
        try:
            state = await self._read_config_state(
                transition.storage,
                expected_node_id=transition.config_node_id,
            )
            if not self._state_matches_pending_generation(
                state,
                generation=transition.generation,
                owner=transition.owner,
            ):
                return
            cleanup = await self._clear_pending_generation(
                transition.storage,
                state,
                generation=transition.generation,
                owner=transition.owner,
            )
            if cleanup.cleared:
                self._host_config = dict(cleanup.state.config)
                self._host_config_loaded = True
        except BaseException:  # noqa: BLE001 - quarantine remains the fallback
            self._host_config_loaded = False

    async def _quarantine_unreconciled_client(
        self,
        *,
        lifecycle_lock_held: bool = False,
    ) -> None:
        """Fail closed when a recovery child cannot be reconciled to storage.

        The SDK subprocess client stops/terminates its child even when graceful
        RPC shutdown fails.  Other client implementations may not provide that
        guarantee, so remove the client and its host-visible tools before
        attempting best-effort retirement.  No config values enter logs or
        exception messages.
        """

        # This shares the terminal transaction with shutdown.  In particular,
        # a cancellation delivered after seal cannot skip unpublication,
        # adapter removal, supervision cancellation, or best-effort retirement
        # while the task is still waiting on another lifecycle owner.
        self._latch_terminal_lifecycle()
        await self._complete_terminal_cleanup(
            best_effort=True,
            lifecycle_lock_held=lifecycle_lock_held,
        )

    async def _stage_pending_config(
        self,
        pending_config: Dict[str, Any],
        *,
        preserve_secret_fields: set[str] | None = None,
        validate_effective_config: Callable[[Dict[str, Any]], None] | None = None,
    ) -> _ConfigTransition:
        """CAS-stage one candidate from a fresh authoritative graph snapshot.

        ``add_node`` is an upsert and is therefore unsafe on hosted replicas.
        Persistent transitions require ``compare_and_swap_node``; a storage
        surface without that atomic contract fails before any lifecycle hook is
        invoked.  Volatile privacy mode remains the intentional non-durable
        path above.
        """

        storage = getattr(self.agent, "storage", None)
        if storage is None:
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: storage is unavailable"
            )

        persistent = await self._persistent_config_writes_allowed(storage)
        if not persistent:
            # Privacy modes intentionally have no durable state to contend
            # over. Keeping the transition in-memory is valid only because the
            # policy forbids a durable config node in the first place.
            effective_pending_config = dict(pending_config)
            for key in preserve_secret_fields or set():
                if key not in effective_pending_config and key in self._host_config:
                    effective_pending_config[key] = self._host_config[key]
            if validate_effective_config is not None:
                validate_effective_config(dict(effective_pending_config))
            return _ConfigTransition(
                active_config=dict(self._host_config),
                next_config=effective_pending_config,
                persistent=False,
                storage=storage,
                expected_properties=None,
                staged_properties=None,
                promoted_properties=None,
            )

        transition: Optional[_ConfigTransition] = None
        try:
            while True:
                # This is the first durable read of a new lifecycle
                # transition.  A cached scoped identity must not silently
                # drift to a legacy row here: surface the rolling-upgrade
                # retry before any scoped candidate is staged.
                state = await self._read_config_state(
                    storage,
                    fence_cached_scoped_authority=True,
                )
                if state.has_pending:
                    if not self._pending_lease_is_expired(state):
                        await self._reconcile_client_to_authoritative_config(
                            state.config,
                            force=False,
                        )
                        raise RuntimeError(
                            f"Cannot apply config for isolated feature {self.name}: "
                            "config transition is already in progress"
                        )

                    # An expired lease is abandoned work, not a candidate we
                    # may promote.  First CAS-remove exactly that generation,
                    # retain its active config, then start over from a fresh
                    # snapshot.  No child sees the abandoned pending config.
                    cleanup = await self._clear_pending_generation(
                        storage,
                        state,
                        generation=state.pending_generation,
                        owner=state.pending_owner,
                    )
                    if cleanup.cleared:
                        continue
                    if self._state_matches_pending_generation(
                        cleanup.state,
                        generation=state.pending_generation,
                        owner=state.pending_owner,
                    ):
                        raise RuntimeError(
                            f"Cannot apply config for isolated feature {self.name}: "
                            "could not clear an expired config transition"
                        )
                    # A concurrent replica changed the node. Re-read before
                    # deciding whether its state is active, pending, or ours.
                    continue

                generation = uuid4().hex
                owner = self._config_transition_owner
                lease_expires_at = _utc_now() + _PENDING_CONFIG_LEASE_TTL
                # Reconstitute omitted write-only fields immediately before the
                # stage CAS from *this* snapshot.  A concurrent credential
                # rotation can only make our predicate fail; the next loop
                # observes its winner and preserves that value instead.
                effective_pending_config = dict(pending_config)
                for key in preserve_secret_fields or set():
                    if key not in effective_pending_config and key in state.config:
                        effective_pending_config[key] = state.config[key]
                if validate_effective_config is not None:
                    validate_effective_config(dict(effective_pending_config))
                staged_properties = dict(state.properties or {})
                staged_properties["config"] = dict(state.config)
                staged_properties["pending_config"] = dict(effective_pending_config)
                staged_properties[_PENDING_GENERATION_KEY] = generation
                staged_properties[_PENDING_OWNER_KEY] = owner
                staged_properties[_PENDING_LEASE_EXPIRES_AT_KEY] = (
                    lease_expires_at.isoformat()
                )

                promoted_properties = self._promoted_properties_from_staged(
                    staged_properties,
                    config=effective_pending_config,
                    generation=generation,
                )

                transition = _ConfigTransition(
                    active_config=dict(state.config),
                    next_config=dict(effective_pending_config),
                    persistent=True,
                    storage=storage,
                    expected_properties=(
                        dict(state.properties) if state.properties is not None else None
                    ),
                    staged_properties=staged_properties,
                    promoted_properties=promoted_properties,
                    config_node_id=state.node_id,
                    generation=generation,
                    owner=owner,
                )
                try:
                    write = await self._write_config_state(
                        storage,
                        transition.expected_properties,
                        staged_properties,
                        expected_node_id=transition.config_node_id,
                    )
                except _ConfigAuthorityChanged as authority_error:
                    # The write can commit before its post-CAS legacy fence
                    # observes an old replica.  ``set_config`` has not yet
                    # received this local transition return value, so carry
                    # its exact scoped candidate with the fence signal; the
                    # lifecycle owner must quarantine rather than treating it
                    # as a pre-stage authority change and reconciling traffic.
                    authority_error.transition = transition
                    raise
                if write.committed:
                    return transition

                # A connection can fail after the conditional write committed.
                # Read before deciding this is a failed stage or a concurrent
                # winner. Cancellation follows the same owned-abort path as
                # every later await boundary.
                observed = await self._read_config_state(
                    storage,
                    expected_node_id=transition.config_node_id,
                )
                if observed.properties == staged_properties:
                    if isinstance(write.error, asyncio.CancelledError):
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=False,
                            preserve_cancellation=True,
                        )
                        raise write.error
                    return transition

                if (
                    write.error is None
                    and write.outcome == "predicate_failed"
                    and observed.properties != transition.expected_properties
                    and preserve_secret_fields
                ):
                    # An atomic PATCH preservation attempt deliberately retries
                    # only after a newer same-tenant durable winner.  Each
                    # loop re-merges omitted write-only fields from the exact
                    # CAS predicate snapshot, so it can never reintroduce the
                    # stale secret that lost.  Other outcomes (notably a
                    # foreign globally-colliding node reported as not_found)
                    # have no newer readable predicate and must fail closed.
                    continue

                await self._reconcile_client_to_authoritative_config(
                    observed.config,
                    force=False,
                )
                if write.error is not None:
                    self._raise_storage_write_error(write.error)
                raise RuntimeError(
                    f"Cannot apply config for isolated feature {self.name}: "
                    "config transition conflicts with a newer durable state"
                )
        except asyncio.CancelledError:
            if transition is not None:
                await self._run_owned_transition_cleanup(
                    transition,
                    force=False,
                    preserve_cancellation=True,
                )
            raise

    def _pending_lease_is_expired(self, state: _ConfigState) -> bool:
        """Return whether a validated pending lease is safely reclaimable."""

        expires_at = state.pending_lease_expires_at
        if expires_at is None:
            # ``_read_config_state`` rejects malformed pending metadata. Keep
            # this guard fail-closed if a future caller constructs a state.
            return False
        return expires_at <= (_utc_now() - _PENDING_CONFIG_CLOCK_SKEW)

    async def _renew_transition_lease(self, transition: _ConfigTransition) -> None:
        """Extend exactly this staged generation's lease with a CAS write.

        The lifecycle hook may legitimately take longer than the original
        takeover interval.  Renewal keeps a healthy owner unstealable without
        relaxing recovery of an actually abandoned stage.  A predicate failure
        is a hard fence: another durable outcome won and this proxy must stop
        treating its child as authoritative.
        """

        if not transition.persistent:
            return
        staged = transition.staged_properties
        generation = transition.generation
        owner = transition.owner
        if (
            not isinstance(staged, dict)
            or not isinstance(generation, str)
            or not generation
            or not isinstance(owner, str)
            or not owner
        ):
            raise _ConfigTransitionLeaseLost("isolated config transition lease is invalid")

        current_state = await self._read_config_state(
            transition.storage,
            expected_node_id=transition.config_node_id,
        )
        if not self._state_matches_pending_generation(
            current_state,
            generation=generation,
            owner=owner,
        ) or current_state.properties != staged:
            raise _ConfigTransitionLeaseLost("isolated config transition lease was lost")

        renewed = dict(staged)
        renewed[_PENDING_LEASE_EXPIRES_AT_KEY] = (
            _utc_now() + _PENDING_CONFIG_LEASE_TTL
        ).isoformat()
        write = await self._write_config_state(
            transition.storage,
            staged,
            renewed,
            expected_node_id=transition.config_node_id,
        )
        if write.committed:
            transition.staged_properties = renewed
            transition.promoted_properties = self._promoted_properties_from_staged(
                renewed,
                config=transition.next_config,
                generation=generation,
            )
            return

        observed = await self._read_config_state(
            transition.storage,
            expected_node_id=transition.config_node_id,
        )
        if observed.properties == renewed:
            # The CAS committed before its caller lost the response.  Preserve
            # the refreshed predicate for a later promotion.
            transition.staged_properties = renewed
            transition.promoted_properties = self._promoted_properties_from_staged(
                renewed,
                config=transition.next_config,
                generation=generation,
            )
            return
        raise _ConfigTransitionLeaseLost("isolated config transition lease was lost")

    @staticmethod
    def _lease_heartbeat_interval() -> float:
        """Renew well before expiry while keeping a minimum testable cadence."""

        return max(0.01, _PENDING_CONFIG_LEASE_TTL.total_seconds() / 3)

    async def _run_transition_lease_heartbeat(
        self,
        transition: _ConfigTransition,
        stop: asyncio.Event,
    ) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self._lease_heartbeat_interval()
                )
                return
            except asyncio.TimeoutError:
                await self._renew_transition_lease(transition)

    async def _await_task_completion(
        self,
        task: asyncio.Task[Any],
        *,
        preserve_cancellation: bool,
    ) -> Any:
        """Wait for a shielded task without leaving it behind on cancellation."""

        return await _await_task_until_complete(
            task,
            preserve_cancellation=preserve_cancellation,
        )

    @staticmethod
    def _state_matches_pending_generation(
        state: _ConfigState,
        *,
        generation: Optional[str],
        owner: Optional[str],
    ) -> bool:
        """Whether ``state`` still names one exact pending owner/generation."""

        return (
            state.has_pending
            and generation is not None
            and owner is not None
            and state.pending_generation == generation
            and state.pending_owner == owner
        )

    @staticmethod
    def _without_pending_metadata(properties: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the active config while removing a completed/abandoned stage."""

        cleared = dict(properties)
        cleared.pop("pending_config", None)
        cleared.pop(_PENDING_GENERATION_KEY, None)
        cleared.pop(_PENDING_OWNER_KEY, None)
        cleared.pop(_PENDING_LEASE_EXPIRES_AT_KEY, None)
        return cleared

    @staticmethod
    def _promoted_properties_from_staged(
        staged_properties: Dict[str, Any],
        *,
        config: Dict[str, Any],
        generation: str,
    ) -> Dict[str, Any]:
        """Build the exact promotion state for one current leased stage."""

        promoted = dict(staged_properties)
        promoted["config"] = dict(config)
        promoted[_CONFIG_GENERATION_KEY] = generation
        promoted.pop("pending_config", None)
        promoted.pop(_PENDING_GENERATION_KEY, None)
        promoted.pop(_PENDING_OWNER_KEY, None)
        promoted.pop(_PENDING_LEASE_EXPIRES_AT_KEY, None)
        return promoted

    async def _clear_pending_generation(
        self,
        storage: Any,
        state: _ConfigState,
        *,
        generation: Optional[str],
        owner: Optional[str],
    ) -> _PendingCleanupResolution:
        """CAS-clear exactly one pending generation and reconcile ambiguity.

        The full properties snapshot is the predicate.  Matching owner and
        generation before every write is defense in depth: a delayed cleanup
        can never erase a newer writer's stage.  A write exception is ambiguous
        and is always followed by a durable read; one retry covers a transient
        pre-commit failure without turning cleanup into an unbounded loop.
        """

        if (
            state.properties is None
            or not self._state_matches_pending_generation(
                state,
                generation=generation,
                owner=owner,
            )
        ):
            return _PendingCleanupResolution(state=state, cleared=False)

        expected_properties = dict(state.properties)
        cleared_properties = self._without_pending_metadata(expected_properties)
        observed = state
        for _ in range(_PENDING_CLEANUP_WRITE_ATTEMPTS):
            write = await self._write_config_state(
                storage,
                expected_properties,
                cleared_properties,
                expected_node_id=state.node_id,
            )
            if write.committed:
                return _PendingCleanupResolution(
                    state=_ConfigState(
                        properties=cleared_properties,
                        config=dict(state.config),
                        node_id=state.node_id,
                    ),
                    cleared=True,
                )

            observed = await self._read_config_state(
                storage,
                expected_node_id=state.node_id,
            )
            if observed.properties == cleared_properties:
                return _PendingCleanupResolution(state=observed, cleared=True)
            if not self._state_matches_pending_generation(
                observed,
                generation=generation,
                owner=owner,
            ):
                return _PendingCleanupResolution(state=observed, cleared=False)
            # A retry remains scoped to the exact initial properties. If
            # anything beyond a retry is needed, leave it for the lease/takeover
            # protocol instead of risking a writer that changed the state.
            if observed.properties != expected_properties:
                return _PendingCleanupResolution(state=observed, cleared=False)

        return _PendingCleanupResolution(state=observed, cleared=False)

    async def _abort_and_reconcile_uncommitted_transition(
        self,
        transition: _ConfigTransition,
        *,
        force: bool,
    ) -> _ConfigState:
        """Abort this pending stage or quarantine if its durable outcome is unknown."""

        if not transition.persistent:
            state = _ConfigState(properties=None, config=dict(transition.active_config))
            await self._reconcile_client_to_authoritative_config(state.config, force=force)
            return state

        try:
            state = await self._read_config_state(
                transition.storage,
                expected_node_id=transition.config_node_id,
            )
            if self._state_matches_pending_generation(
                state,
                generation=transition.generation,
                owner=transition.owner,
            ):
                cleanup = await self._clear_pending_generation(
                    transition.storage,
                    state,
                    generation=transition.generation,
                    owner=transition.owner,
                )
                state = cleanup.state
                if not cleanup.cleared and self._state_matches_pending_generation(
                    state,
                    generation=transition.generation,
                    owner=transition.owner,
                ):
                    raise RuntimeError(
                        f"Cannot apply config for isolated feature {self.name}: "
                        "could not reconcile config transition cleanup"
                    )
            await self._reconcile_client_to_authoritative_config(state.config, force=force)
            return state
        except BaseException:
            self._host_config_loaded = False
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            raise

    async def _promote_config(
        self, transition: _ConfigTransition
    ) -> _PromotionResolution:
        """Promote only ``transition``'s staged generation, then reconcile.

        Every non-successful write result — including ``CancelledError`` and a
        transport exception after commit — is followed by a durable read. The
        caller must use the returned state rather than assuming an exception
        means the old config is still active.
        """

        if not transition.persistent:
            return _PromotionResolution(
                state=_ConfigState(
                    properties=None,
                    config=dict(transition.next_config),
                ),
                committed=True,
            )

        try:
            permitted = await self._persistent_config_writes_allowed(
                transition.storage
            )
        except BaseException as probe_error:
            # A completed SDK hook may already have adopted next_config. Even
            # though no promotion write was attempted, a failed policy probe
            # leaves the staged durable state authoritative, so reconcile it
            # before returning the probe/cancellation outcome.
            observed = await self._read_config_state_after_promotion_failure(
                transition.storage,
                probe_error,
                expected_node_id=transition.config_node_id,
            )
            return _PromotionResolution(
                state=observed,
                committed=False,
                error=probe_error,
                storage_error=True,
            )
        if not permitted:
            observed = await self._read_config_state_after_promotion_failure(
                transition.storage,
                RuntimeError("persistent config writes became unavailable"),
                expected_node_id=transition.config_node_id,
            )
            return _PromotionResolution(
                state=observed,
                committed=False,
                error=RuntimeError(
                    f"Cannot apply config for isolated feature {self.name}: "
                    "persistent config writes became unavailable"
                ),
            )

        write = await self._write_config_state(
            transition.storage,
            transition.staged_properties,
            transition.promoted_properties,
            expected_node_id=transition.config_node_id,
        )
        if write.committed:
            return _PromotionResolution(
                state=_ConfigState(
                    properties=dict(transition.promoted_properties or {}),
                    config=dict(transition.next_config),
                    node_id=transition.config_node_id,
                ),
                committed=True,
            )

        observed = await self._read_config_state_after_promotion_failure(
            transition.storage,
            write.error,
            expected_node_id=transition.config_node_id,
        )
        # The generation stamp makes this proof specific to this transition;
        # matching only ``config`` would let a different replica's same-valued
        # write be mistaken for our promotion.
        committed = observed.properties == transition.promoted_properties
        return _PromotionResolution(
            state=observed,
            committed=committed,
            error=write.error,
            storage_error=write.error is not None,
        )

    async def _write_config_state(
        self,
        storage: Any,
        expected_properties: Optional[Dict[str, Any]],
        properties: Optional[Dict[str, Any]],
        *,
        expected_node_id: Optional[str] = None,
    ) -> _ConfigWriteResult:
        """Conditionally write one complete transition state without swallowing.

        The graph store's atomic compare-and-swap is the durable protocol.
        There is intentionally no ``add_node`` fallback: an upsert cannot prove
        ownership on hosted replicas and would let a stale reader overwrite a
        newer config.
        """

        from kestrel_sovereign.storage.async_graph_store import GraphNode

        node_id = await self._resolved_config_node_id_for(
            storage,
            expected_node_id=expected_node_id,
        )

        node = GraphNode(
            node_id=node_id,
            node_type=self._CONFIG_NODE_TYPE,
            label=f"{self.name} config",
            properties=dict(properties or {}),
        )
        compare_and_swap = getattr(storage, "compare_and_swap_node", None)
        if not callable(compare_and_swap):
            return _ConfigWriteResult(
                committed=False,
                error=RuntimeError(
                    "persistent isolated config transitions require "
                    "compare_and_swap_node"
                ),
            )
        try:
            result = await _maybe_await(
                compare_and_swap(node_id, expected_properties, node)
            )
            if result == "swapped":
                if node_id == self._config_node_id():
                    # The first authority resolution cannot make a scoped key
                    # atomic with an old binary's legacy key.  A legacy row
                    # can appear while this CAS awaits, so every successful
                    # scoped write must fence that outcome *before* lease
                    # renewal, promotion, lifecycle hooks, or traffic can
                    # continue under the scoped authority.
                    await self._resolved_config_node_id_for(
                        storage,
                        expected_node_id=node_id,
                        fence_cached_scoped_authority=True,
                    )
                return _ConfigWriteResult(committed=True, outcome="swapped")
            return _ConfigWriteResult(
                committed=False,
                outcome=result if isinstance(result, str) else None,
            )
        except _ConfigAuthorityChanged:
            # This is a post-commit rolling-upgrade fence, not an ambiguous
            # storage failure.  It must reach the lifecycle owner immediately
            # so it quarantines before any hook or traffic can continue.
            raise
        except BaseException as exc:
            # A write boundary may raise after commit. Its caller performs the
            # authoritative read needed to classify the result.
            return _ConfigWriteResult(committed=False, error=exc)

    async def _persistent_config_writes_allowed(self, storage: Any) -> bool:
        """Return the current privacy-policy permission for config persistence."""

        allows_persistent_writes = getattr(storage, "allows_persistent_writes", None)
        if not callable(allows_persistent_writes):
            return True
        try:
            return bool(await _maybe_await(allows_persistent_writes()))
        except Exception as exc:  # noqa: BLE001 - policy probe is a hard boundary
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "could not determine persistence policy"
            ) from exc

    async def _read_config_state(
        self,
        storage: Any,
        *,
        expected_node_id: Optional[str] = None,
        fence_cached_scoped_authority: bool = False,
    ) -> _ConfigState:
        """Read one snapshot from the visible durable authority.

        A read made before a transition is pinned may safely adopt a newly
        visible legacy row.  Expected-node reads and callers explicitly
        beginning a transition retain the strict cached-scoped authority fence.
        """

        get_node = getattr(storage, "get_node", None)
        if not callable(get_node):
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "storage cannot read config transition state"
            )
        try:
            node_id = await self._resolved_config_node_id_for(
                storage,
                expected_node_id=expected_node_id,
                fence_cached_scoped_authority=fence_cached_scoped_authority,
            )
            node = await _maybe_await(get_node(node_id))
            if node_id == self._config_node_id():
                # A read-only lookup can adopt a legacy row that appeared
                # between its scoped lookup and this recheck. Once an exact
                # node is pinned, keep the strict fence instead.
                pinned_authority = (
                    expected_node_id is not None or fence_cached_scoped_authority
                )
                revalidated_node_id = await self._resolved_config_node_id_for(
                    storage,
                    expected_node_id=node_id if pinned_authority else None,
                    fence_cached_scoped_authority=fence_cached_scoped_authority,
                )
                if revalidated_node_id != node_id:
                    node_id = revalidated_node_id
                    node = await _maybe_await(get_node(node_id))
        except _ConfigAuthorityChanged:
            raise
        except Exception as exc:  # noqa: BLE001 - durable read is a hard boundary
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "failed to load config transition state"
            ) from exc
        if node is None:
            return _ConfigState(properties=None, config={}, node_id=node_id)

        self._validate_config_node(node, node_kind="stored")
        properties = dict(node.properties)
        raw_config = properties.get("config")
        if isinstance(raw_config, str):
            try:
                raw_config = json.loads(raw_config)
            except (TypeError, ValueError):
                raw_config = None
        config = dict(raw_config) if isinstance(raw_config, dict) else {}
        pending_keys = (
            "pending_config",
            _PENDING_GENERATION_KEY,
            _PENDING_OWNER_KEY,
            _PENDING_LEASE_EXPIRES_AT_KEY,
        )
        has_pending = any(key in properties for key in pending_keys)
        if not has_pending:
            return _ConfigState(
                properties=properties,
                config=config,
                node_id=node_id,
            )

        pending_config = properties.get("pending_config")
        generation = properties.get(_PENDING_GENERATION_KEY)
        owner = properties.get(_PENDING_OWNER_KEY)
        raw_expires_at = properties.get(_PENDING_LEASE_EXPIRES_AT_KEY)
        if (
            not isinstance(pending_config, dict)
            or not isinstance(generation, str)
            or not generation
            or not isinstance(owner, str)
            or not owner
            or not isinstance(raw_expires_at, str)
            or not raw_expires_at
        ):
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "stored pending config transition metadata is invalid"
            )
        try:
            expires_at = datetime.fromisoformat(raw_expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "stored pending config lease is invalid"
            ) from exc
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "stored pending config lease is invalid"
            )
        return _ConfigState(
            properties=properties,
            config=config,
            node_id=node_id,
            has_pending=True,
            pending_generation=generation,
            pending_owner=owner,
            pending_lease_expires_at=expires_at.astimezone(timezone.utc),
        )

    async def _read_config_state_after_promotion_failure(
        self,
        storage: Any,
        write_error: Optional[BaseException],
        *,
        expected_node_id: Optional[str] = None,
    ) -> _ConfigState:
        """Read durable state after an ambiguous promotion or quarantine."""

        try:
            return await self._read_config_state(
                storage,
                expected_node_id=expected_node_id,
            )
        except BaseException as read_error:
            # We cannot prove which config won, so no live child may remain.
            self._host_config_loaded = False
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            if isinstance(write_error, asyncio.CancelledError):
                raise write_error
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "could not reconcile durable config after a promotion failure"
            ) from read_error

    async def _reconcile_client_to_authoritative_config(
        self,
        config: Dict[str, Any],
        *,
        force: bool,
    ) -> None:
        """Make the local child match a config freshly read from storage."""

        authoritative_config = dict(config)
        if self._terminal_lifecycle_latched:
            # Terminal cleanup has already made this enable cycle unavailable.
            # A cleanup re-read may recover storage after quarantine, but it
            # must never turn that recovery into a new, unsupervised child.
            # Retain the actual child's config if retirement has not completed;
            # otherwise cache the durable value for the later explicit
            # initialize.
            if self._client is None:
                self._host_config = authoritative_config
                self._host_config_loaded = True
            return
        try:
            # Forced reconciliation owns recovery after a lifecycle operation
            # whose outcome made the current publication unsafe.  In
            # particular, ``_replace_client`` removes the old child before it
            # starts a candidate; if that candidate fails after durable
            # promotion, there is intentionally no client here.  Rebuild from
            # the freshly read *active* config before the finite traffic gate
            # can reopen.  A failed rebuild still follows the quarantine path
            # below while the caller retains the lifecycle lock.
            if force or (
                self._client is not None and self._host_config != authoritative_config
            ):
                await self._replace_client(authoritative_config)
        except BaseException:
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            self._host_config = authoritative_config
            self._host_config_loaded = True
            raise
        self._host_config = authoritative_config
        self._host_config_loaded = True

    def _assert_child_start_allowed(self) -> None:
        """Refuse normal child-lifecycle work after terminal cleanup starts."""

        if self._terminal_lifecycle_latched:
            raise RuntimeError(
                f"Cannot continue isolated feature {self.name}: terminal lifecycle "
                "is latched; explicit initialize is required"
            )

    def _raise_storage_write_error(self, error: BaseException) -> None:
        """Surface storage failure without leaking feature config or secrets."""

        if isinstance(error, asyncio.CancelledError):
            raise error
        raise RuntimeError(
            f"Cannot apply config for isolated feature {self.name}: "
            "failed to persist config"
        ) from error

    def _raise_promotion_failure(self, promotion: _PromotionResolution) -> None:
        """Raise the classified promotion outcome after local reconciliation."""

        if promotion.error is not None:
            if promotion.storage_error:
                self._raise_storage_write_error(promotion.error)
            raise promotion.error
        raise RuntimeError(
            f"Cannot apply config for isolated feature {self.name}: "
            "config transition conflicts with a newer durable state"
        )

    async def _prepare_config_transition(
        self, next_config: Dict[str, Any]
    ) -> ConfigTransitionResult | None:
        """Run the public SDK lifecycle hook when the live client opted in.

        Capability negotiation is intentionally limited to the SDK client's
        typed property and lifecycle method. The host neither knows nor sends
        feature-private RPC method names.
        """

        if self._client is None:
            return None
        if not self._supports_config_transition():
            return None

        prepare = getattr(self._client, "prepare_config_transition", None)
        if not callable(prepare):
            raise RuntimeError(
                f"Isolated feature {self.name} advertised config-transition support "
                "without the SDK lifecycle method"
            )

        result = await _maybe_await(prepare(next_config))
        if not isinstance(result, ConfigTransitionResult):
            raise RuntimeError(
                f"Isolated feature {self.name} returned an invalid config-transition result"
            )
        return result

    async def _prepare_config_transition_with_lease(
        self, transition: _ConfigTransition
    ) -> ConfigTransitionResult | None:
        """Run the external hook while continuously proving stage ownership."""

        if not transition.persistent:
            return await self._prepare_config_transition(transition.next_config)

        stop_heartbeat = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._run_transition_lease_heartbeat(transition, stop_heartbeat),
            name=f"isolated-config-lease:{self.name}",
        )
        hook = asyncio.create_task(
            self._prepare_config_transition(transition.next_config),
            name=f"isolated-config-hook:{self.name}",
        )
        try:
            done, _ = await asyncio.wait(
                {hook, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                # A healthy heartbeat only finishes when asked to stop below;
                # before that, its completion is a lost lease or storage error.
                heartbeat.result()
                raise _ConfigTransitionLeaseLost(
                    "isolated config transition lease heartbeat stopped unexpectedly"
                )
            return hook.result()
        finally:
            # If the lease cannot be renewed, cancellation may reach the SDK
            # child; wait for its cancellation to settle before recovery makes
            # a replacement decision.  No lifecycle task may outlive the
            # reload lock.
            if not hook.done():
                hook.cancel()
                try:
                    await self._await_task_completion(
                        hook, preserve_cancellation=True
                    )
                except asyncio.CancelledError:
                    pass
            stop_heartbeat.set()
            try:
                await self._await_task_completion(
                    heartbeat, preserve_cancellation=True
                )
            except asyncio.CancelledError:
                pass

    def _supports_config_transition(self) -> bool:
        """Whether the initialized SDK client explicitly opted into transitions."""

        return (
            self._client is not None
            and getattr(self._client, "supports_config_transition", False) is True
        )

    def _new_external_ingress_quiesce(self) -> _ExternalIngressQuiesce | None:
        """Return a capability-negotiated producer pause for this exact child.

        The SDK 0.35.1 private host-ingress capability already supplies the
        versioned, typed negotiation and bounded JSON transport this lifecycle
        protocol needs. Legacy SDK/services simply do not advertise both names
        and retain their established replacement behavior.
        """

        client = self._client
        if client is None:
            return None
        capabilities = self._host_ingress_capabilities()
        if capabilities is None or not {
            _EXTERNAL_INGRESS_QUIESCE,
            _EXTERNAL_INGRESS_RESUME,
        }.issubset(capabilities.names):
            return None
        return _ExternalIngressQuiesce(
            client=client,
            transition_id=secrets.token_urlsafe(_EXTERNAL_INGRESS_TRANSITION_TOKEN_BYTES),
        )

    @staticmethod
    def _is_external_ingress_lifecycle_ack(value: Any, *, state: str) -> bool:
        """Accept only the small, deterministic lifecycle acknowledgment."""

        if type(value) is not dict:
            return False
        allowed = {
            "status",
            "http_status",
            "state",
            "already_quiesced",
            "already_resumed",
        }
        if not set(value).issubset(allowed):
            return False
        if (
            value.get("status") != "ok"
            or type(value.get("http_status")) is not int
            or value.get("http_status") != 200
            or value.get("state") != state
        ):
            return False
        return all(
            type(value[key]) is bool
            for key in ("already_quiesced", "already_resumed")
            if key in value
        )

    def _fence_external_ingress_lifecycle_timeout(self) -> None:
        """Fail closed if a quiesce/resume RPC cannot settle within its budget."""

        self._fenced_recovery_failed = True
        self._latch_terminal_lifecycle()

    async def _call_exact_external_ingress_lifecycle(
        self,
        quiesce: _ExternalIngressQuiesce,
        name: str,
    ) -> Any:
        """Call only the paused client's lifecycle RPC outside data admission.

        This private path is deliberately narrower than ``call_host_ingress``:
        it accepts only the negotiated transition names, only the exact client
        captured before the gate closed, and only a freshly validated detached
        token payload. It is not an agent tool or a general ingress bypass.
        """

        if self._stopping or self._client is not quiesce.client:
            raise RuntimeError("isolated feature changed during external ingress lifecycle")
        if name not in {_EXTERNAL_INGRESS_QUIESCE, _EXTERNAL_INGRESS_RESUME}:
            raise RuntimeError("invalid external ingress lifecycle operation")
        capabilities = self._host_ingress_capabilities(quiesce.client)
        if capabilities is None or name not in capabilities.names:
            raise RuntimeError("isolated feature lacks external ingress lifecycle capability")
        request = _prepare_host_ingress_request(
            name, {"transition_id": quiesce.transition_id}
        )
        if request is None:
            raise RuntimeError("invalid external ingress lifecycle request")
        call = getattr(quiesce.client, "call_host_ingress", None)
        if not callable(call):
            raise RuntimeError("isolated feature lacks external ingress lifecycle RPC")
        outcome_slot = _HostIngressOutcomeSlot()
        await self._call_host_ingress_rpc(call, request, outcome_slot)
        outcome = outcome_slot.outcome
        if outcome is None:
            raise HostIngressError("external ingress lifecycle RPC failed")
        if outcome.status == _HOST_INGRESS_CANCELLED:
            raise asyncio.CancelledError()
        if outcome.status != _HOST_INGRESS_SUCCESS:
            raise HostIngressError("external ingress lifecycle RPC failed")
        return outcome.payload

    async def _quiesce_external_ingress(
        self, quiesce: _ExternalIngressQuiesce
    ) -> None:
        """Stop/reap an opt-in external producer before closing Core admission."""

        if self._client is not quiesce.client:
            raise RuntimeError("isolated feature changed before external ingress quiesce")
        result = await _await_owned_facade_lifecycle_operation(
            self._call_exact_external_ingress_lifecycle(
                quiesce, _EXTERNAL_INGRESS_QUIESCE
            ),
            name=f"isolated-external-ingress-quiesce:{self.name}",
            on_timeout=self._fence_external_ingress_lifecycle_timeout,
            on_late_task=lambda task, client=quiesce.client: self._retain_terminal_lifecycle_task(
                task, client
            ),
        )
        if not self._is_external_ingress_lifecycle_ack(result, state="quiesced"):
            raise RuntimeError("isolated feature did not acknowledge external ingress quiesce")

    async def _resume_external_ingress(
        self, quiesce: _ExternalIngressQuiesce
    ) -> None:
        """Resume a failed/live-applied transition only after Core reopens its gate."""

        if self._client is not quiesce.client:
            return
        result = await _await_owned_facade_lifecycle_operation(
            self._call_exact_external_ingress_lifecycle(
                quiesce, _EXTERNAL_INGRESS_RESUME
            ),
            name=f"isolated-external-ingress-resume:{self.name}",
            on_timeout=self._fence_external_ingress_lifecycle_timeout,
            on_late_task=lambda task, client=quiesce.client: self._retain_terminal_lifecycle_task(
                task, client
            ),
        )
        if not self._is_external_ingress_lifecycle_ack(result, state="resumed"):
            raise RuntimeError("isolated feature did not acknowledge external ingress resume")

    async def _finalize_external_ingress_transition(
        self, quiesce: _ExternalIngressQuiesce | None
    ) -> None:
        """Resume the exact paused producer, then reopen traffic atomically.

        Resuming while the gate remains closed is intentional: its first
        acknowledged callback is stored as the single detached deferred
        snapshot. A resume failure seals and quarantines before that snapshot
        can replay.
        """

        if self._stopping:
            await self._seal_traffic_gate()
            return
        if quiesce is not None and self._client is quiesce.client:
            try:
                await self._resume_external_ingress(quiesce)
            except BaseException:
                logger.error(
                    "Isolated feature %s could not resume external ingress; quarantining the proxy",
                    self.name,
                )
                self._latch_terminal_lifecycle()
                await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
                raise
        if self._stopping:
            await self._seal_traffic_gate()
            return
        await self._reopen_traffic_gate()

    def _client_requires_replacement(self) -> bool:
        """Whether the SDK fenced the current child after an unknown outcome."""

        return getattr(self._client, "replacement_required", False) is True

    async def _load_host_config(self) -> Dict[str, Any]:
        """Resolve persisted/UI host config to forward into the service.

        Reads the same graph-store node the in-process Feature base persists to
        (``feature_config:<name>``). An absent node is an intentional empty
        config. A failed read is not: starting with ``{}`` would make that
        transient failure authoritative and could overwrite a write-only secret
        in a later partial update, so initialization fails until storage recovers.
        """
        try:
            persisted = await self.load_persisted_config(raise_on_error=True)
        except Exception as exc:  # noqa: BLE001 - durable read is a hard boundary
            raise RuntimeError(
                f"Cannot initialize isolated feature {self.name}: failed to load persisted config"
            ) from exc
        return persisted if isinstance(persisted, dict) else {}

    async def _ensure_host_config_loaded(self) -> None:
        """Load durable config exactly before it may become host-authoritative."""

        if self._host_config_loaded:
            return
        self._host_config = await self._load_host_config()
        self._host_config_loaded = True

    # ------------------------------------------------------------------
    # Channel bridge
    # ------------------------------------------------------------------

    def _client_capabilities(self) -> Dict[str, Any]:
        # Prefer the wrapper's passthrough; fall back to the inner JSON-RPC
        # client, where SubprocessIsolatedFeatureClient stores capabilities after
        # initialize (covers SDK builds without the wrapper-level property).
        caps = getattr(self._client, "capabilities", None)
        if not isinstance(caps, dict) or not caps:
            inner = getattr(self._client, "client", None)
            inner_caps = getattr(inner, "capabilities", None)
            if isinstance(inner_caps, dict):
                caps = inner_caps
        return caps if isinstance(caps, dict) else {}

    def _host_ingress_capabilities(
        self, client: Any | None = None
    ) -> HostIngressCapabilities | None:
        """Return an immutable snapshot of this client's ingress contract.

        Host ingress deliberately relies on the SDK's typed capability rather
        than parsing its raw initialize metadata here.  That preserves one
        validator for malformed/legacy metadata and, importantly, keeps this
        proxy on the subprocess wrapper for the actual RPC.  Reaching through
        to an inner client would bypass its process-lifecycle accounting.

        The SDK dataclass is frozen but Python does not enforce its ``tuple``
        annotation at runtime: a caller can construct it with a mutable list.
        Also, subclasses can replace ``supports``.  Treat the public property
        as an untrusted facade, then reconstruct only its exact immutable base
        contract.  Callers must compare the returned snapshot's names directly
        rather than invoke behavior on the facade object.
        """

        target = self._client if client is None else client
        capabilities = getattr(target, "host_ingress_capabilities", None)
        if type(capabilities) is not HostIngressCapabilities:
            return None

        names = capabilities.names
        version = capabilities.version
        # A valid SDK response always has these exact immutable runtime types:
        # ``from_dict`` converts the wire list into a tuple, while version and
        # names are ordinary built-in primitives.  Reject anything that could
        # mutate after admission or execute user-defined behavior while we
        # validate the negotiated contract.
        if (
            type(version) is not int
            or type(names) is not tuple
            or not all(type(name) is str for name in names)
        ):
            return None
        try:
            return HostIngressCapabilities(names=tuple(names), version=version)
        except ProtocolError:
            # Malformed values that reached a frozen dataclass through direct
            # construction or mutation are indistinguishable from no support.
            return None

    async def _call_host_ingress_rpc(
        self,
        call: Callable[[str, HostIngressPayload], Any],
        request: _HostIngressRequest,
        outcome_slot: _HostIngressOutcomeSlot,
    ) -> None:
        """Run one RPC and place a detached outcome outside the worker task.

        The operation is shielded by the public method. Returning failures as
        data is intentional: even a freshly constructed ``HostIngressError``
        would otherwise retain this coroutine's request, response, and bound
        SDK method in its traceback.
        """

        try:
            try:
                result = await _maybe_await(call(request.name, request.payload))
            except asyncio.CancelledError:
                outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_CANCELLED)
                return
            except BaseException:  # noqa: BLE001 - external RPC boundary redaction
                # The SDK/custom-client boundary is untrusted. Do not retain its
                # exception object or its potentially secret-bearing message.
                outcome_slot.outcome = _HostIngressOutcome(
                    _HOST_INGRESS_GENERIC_FAILURE
                )
                return
            try:
                # A custom facade must not be able to return a mutable or subclass
                # response into the HTTP integration layer. Revalidate the exact,
                # detached snapshot even though the SDK validates its wire payload.
                outcome_slot.outcome = _HostIngressOutcome(
                    _HOST_INGRESS_SUCCESS,
                    _snapshot_host_ingress_payload(result),
                )
            except asyncio.CancelledError:
                outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_CANCELLED)
            except BaseException:  # noqa: BLE001 - external response boundary redaction
                outcome_slot.outcome = _HostIngressOutcome(
                    _HOST_INGRESS_GENERIC_FAILURE
                )
        finally:
            # This worker may survive a caller cancellation. Drop every
            # request/client-adjacent reference as soon as the RPC settles;
            # only the detached one-shot outcome may cross its boundary.
            result = None
            call = None
            request = None
            self = None  # noqa: F841 - explicitly detach the bound proxy reference

    async def _run_host_ingress(
        self,
        request: _HostIngressRequest,
        outcome_slot: _HostIngressOutcomeSlot,
    ) -> None:
        """Perform an already-snapshotted ingress call inside traffic admission."""

        try:
            while True:
                wake_idle = False
                completed_call = False
                async with self._traffic_gate.admit():
                    client = self._client
                    if client is None and self._idle_retired:
                        wake_idle = True
                    elif client is None:
                        outcome_slot.outcome = _HostIngressOutcome(
                            _HOST_INGRESS_UNSUPPORTED
                        )
                        return
                    else:
                        try:
                            capabilities = self._host_ingress_capabilities()
                        except asyncio.CancelledError:
                            outcome_slot.outcome = _HostIngressOutcome(
                                _HOST_INGRESS_CANCELLED
                            )
                            return
                        except BaseException:  # noqa: BLE001 - untrusted capability facade
                            outcome_slot.outcome = _HostIngressOutcome(
                                _HOST_INGRESS_GENERIC_FAILURE
                            )
                            return
                        if capabilities is None:
                            outcome_slot.outcome = _HostIngressOutcome(
                                _HOST_INGRESS_UNSUPPORTED
                            )
                            return
                        if request.name not in capabilities.names:
                            outcome_slot.outcome = _HostIngressOutcome(
                                _HOST_INGRESS_UNKNOWN_NAME
                            )
                            return

                        try:
                            call = getattr(client, "call_host_ingress", None)
                        except asyncio.CancelledError:
                            outcome_slot.outcome = _HostIngressOutcome(
                                _HOST_INGRESS_CANCELLED
                            )
                            return
                        except BaseException:  # noqa: BLE001 - untrusted descriptor boundary
                            outcome_slot.outcome = _HostIngressOutcome(
                                _HOST_INGRESS_GENERIC_FAILURE
                            )
                            return
                        if not callable(call):
                            outcome_slot.outcome = _HostIngressOutcome(
                                _HOST_INGRESS_UNSUPPORTED
                            )
                            return
                        self._record_runtime_activity()
                        try:
                            await self._call_host_ingress_rpc(
                                call, request, outcome_slot
                            )
                        finally:
                            self._record_runtime_activity()
                        completed_call = True
                if completed_call:
                    self._schedule_runtime_telemetry()
                    return
                if wake_idle:
                    await self._wake_idle_runtime()
        except _TrafficGateTerminalError:
            outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_TERMINAL)
        except asyncio.CancelledError:
            # The public caller shields this worker, so a cancellation here is
            # child/runtime-originated and must remain distinguishable from the
            # caller cancellation handled by ``_wait_for_host_ingress_operation``.
            outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_CANCELLED)
        except BaseException:  # noqa: BLE001 - no internal traceback may escape
            outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_GENERIC_FAILURE)
        finally:
            call = None
            capabilities = None
            client = None
            request = None
            self = None

    def _supports_tool_execution_context(self, context: Any) -> bool:
        """Whether the initialized service accepts this SDK context version.

        New SDK clients expose a typed boolean property.  Reading the raw
        initialize capability as a fallback keeps compatible wrappers usable,
        while malformed or legacy capability data fails closed for scheduler
        delivery.
        """

        supported = getattr(self._client, "supports_tool_execution_context", None)
        if isinstance(supported, bool):
            return supported

        capability = self._client_capabilities().get("tool_execution_context")
        versions = capability.get("versions") if isinstance(capability, dict) else None
        return (
            isinstance(versions, list)
            and not isinstance(getattr(context, "version", None), bool)
            and getattr(context, "version", None) in versions
        )

    def _register_channel_bridge(
        self, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """If the service advertises a channel capability, register a forwarding
        ``ChannelAdapter`` so the generic channels API works against this
        isolated feature (otherwise only the feature's own tools + inbound
        events work, not ``channels_send``/``channels_list``)."""
        channel = self._client_capabilities().get("channel")
        if not isinstance(channel, dict):
            return
        channel_type = channel.get("channel_type")
        send_tool = channel.get("send_tool")
        if not channel_type or not send_tool:
            logger.warning(
                "Isolated feature %s advertised an incomplete channel capability: %r",
                self.name,
                channel,
            )
            return

        channel_feature = self._channel_feature()
        registry = getattr(channel_feature, "registry", None) if channel_feature else None
        if registry is None:
            logger.warning(
                "Isolated channel feature %s cannot bridge: ChannelFeature/registry "
                "unavailable; channels_send will not see channel '%s'",
                self.name,
                channel_type,
            )
            return

        adapter = ProxyChannelAdapter(
            self,
            channel_type=str(channel_type),
            send_tool=str(send_tool),
            status_tool=channel.get("status_tool"),
            config=self._channel_config(str(channel_type), config),
        )
        registry.register(adapter)
        self._channel_adapter = adapter
        # Remember the pairing tool so ``call_isolated_tool`` can emit the
        # persisted ``channel_link`` part when it runs (#2081). Prefer an
        # explicitly advertised ``link_tool``; otherwise fall back to the
        # ``<channel_type>_link`` naming convention (e.g. ``whatsapp_link``).
        self._channel_type = str(channel_type)
        link_tool = channel.get("link_tool")
        self._link_tool = str(link_tool) if link_tool else f"{channel_type}_link"
        logger.info(
            "Bridged isolated feature %s into ChannelFeature.registry as channel '%s'",
            self.name,
            channel_type,
        )

    def _unregister_channel_bridge(self) -> None:
        if self._channel_adapter is None:
            return
        channel_feature = self._channel_feature()
        registry = getattr(channel_feature, "registry", None) if channel_feature else None
        if registry is not None:
            # Only remove our own adapter: a reload or a native adapter may have
            # since replaced this channel_type, and we must not evict it.
            getter = getattr(registry, "get", None)
            if not callable(getter) or getter(self._channel_adapter.channel_type) is self._channel_adapter:
                registry.unregister(self._channel_adapter.channel_type)
        self._channel_adapter = None
        self._channel_type = None
        self._link_tool = None

    def _channel_feature(self) -> Any:
        features = getattr(self.agent, "features", None)
        if isinstance(features, dict):
            return features.get("ChannelFeature")
        getter = getattr(features, "get", None)
        return getter("ChannelFeature") if callable(getter) else None

    def _channel_config(
        self, channel_type: str, config: Optional[Dict[str, Any]] = None
    ):
        """Build a ChannelConfig from host config for sender-filtering / enabled.

        Inbound sender filtering and the disabled-channel guard both read the
        registered adapter's ``config``, so mirror the host-side feature config
        onto the forwarding adapter.
        """
        from kestrel_sdk.channels import ChannelConfig

        from kestrel_sovereign.features.channels.feature import (
            canonical_telegram_allowed_senders,
        )

        cfg = self._host_config if config is None else config
        cfg = cfg if isinstance(cfg, dict) else {}
        allowed_senders = cfg.get("allowed_senders") or []
        if channel_type == "telegram":
            # The child retains legacy @usernames only to explain migration.
            # They are not host authorization data: notifications are
            # untrusted until this proxy has applied immutable numeric IDs.
            allowed_senders = canonical_telegram_allowed_senders(allowed_senders)
        return ChannelConfig(
            channel_type=channel_type,
            agent_id=str(cfg.get("agent_id", "") or ""),
            enabled=bool(cfg.get("enabled", True)),
            allowed_senders=list(allowed_senders),
        )

    async def call_isolated_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        context = _scheduled_tool_execution_context()
        requested_at = asyncio.get_running_loop().time()
        experienced_wake = False
        # The gate keeps a selected client alive through the complete RPC.  It
        # is shared (not a reload mutex), so unrelated calls remain concurrent
        # whenever no config transition is active.
        try:
            while True:
                wake_idle = False
                async with self._traffic_gate.admit():
                    if self._client is None and self._idle_retired:
                        wake_idle = True
                    else:
                        # This preflight belongs *inside* admission. Otherwise a
                        # terminal shutdown with no published client would leak a
                        # scheduler-specific error instead of the stable fail-closed
                        # result used by every other new tool/channel call.
                        if context is not None and not self._supports_tool_execution_context(context):
                            raise SchedulerExecutionContextUnavailable(
                                "scheduled isolated tool calls require a service that advertises "
                                "ToolExecutionContext support"
                            )
                        if not experienced_wake:
                            self._last_warm_start_seconds = (
                                asyncio.get_running_loop().time() - requested_at
                            )
                        self._record_runtime_activity()
                        try:
                            result = await self._call_isolated_tool_admitted(
                                name, args, context
                            )
                        finally:
                            self._record_runtime_activity()
                if not wake_idle:
                    # Admission has been released before this background task
                    # can begin any observer or disk work.
                    self._schedule_runtime_telemetry()
                    return result
                if wake_idle:
                    experienced_wake = True
                    await self._wake_idle_runtime()
        except _TrafficGateTerminalError:
            if context is not None:
                # Scheduled work must surface terminal admission as an
                # exception so SchedulerRunner records a failed occurrence.
                # Direct tools and channel sends retain their established flat
                # error envelope for compatibility.
                raise SchedulerTerminalAdmissionError()
            # Unlike an ordinary RPC error, terminal admission has no client to
            # retry against. Keep the public result stable and secret-free for
            # both direct tools and ProxyChannelAdapter sends.
            return {
                "status": "error",
                "error": _TERMINAL_TRAFFIC_ERROR,
                "tool": name,
                "success": False,
            }
        except IsolatedRuntimePreparationError:
            if context is not None:
                raise
            return {
                "status": "error",
                "error": "isolated feature could not start",
                "tool": name,
                "success": False,
            }

    async def call_host_ingress(
        self,
        name: str,
        payload: HostIngressPayload = None,
    ) -> HostIngressPayload:
        """Invoke a negotiated private host-to-service ingress callback.

        This is intentionally a host-only method: it does not create an
        :class:`AgentTool`, appear in ``get_tools()``, or participate in an
        agent/LLM tool description.  Its caller must have already resolved the
        exact feature proxy for the target agent; this method never performs an
        agent or feature lookup itself.

        Capability validation and the RPC both occur under the same traffic
        admission used by normal tools and channel events.  A reload or config
        transition therefore drains an already admitted ingress call and holds
        later calls until its new child is coherent; terminal shutdown rejects
        new calls without touching a retired child.
        """

        # Snapshot synchronously, before scheduling the worker.  A caller can
        # mutate an ordinary dict in the scheduling gap, so validating inside a
        # task is still a TOCTOU bug even if the task later keeps traffic open.
        request = _prepare_host_ingress_request(name, payload)
        outcome = _HostIngressOutcome(_HOST_INGRESS_GENERIC_FAILURE)
        caller_cancel_args: tuple[Any, ...] | None = None
        operation: asyncio.Task[Any] | None = None
        outcome_slot: _HostIngressOutcomeSlot | None = None
        worker: Any = None
        task_name = f"isolated-host-ingress:{self.name}"
        if request is not None:
            try:
                outcome_slot = _HostIngressOutcomeSlot()
                worker = self._run_host_ingress(request, outcome_slot)
                operation = asyncio.create_task(worker, name=task_name)
            except (RuntimeError, TypeError):
                # Task construction is not a supported external failure mode,
                # but it must not cause a request-bearing public traceback.
                outcome = _HostIngressOutcome(_HOST_INGRESS_GENERIC_FAILURE)
                if worker is not None:
                    worker.close()

        # The public error below is deliberately raised only after all frames
        # that received untrusted input or facades have been scrubbed.  Function
        # arguments themselves are traceback locals in CPython, so ``from
        # None`` alone cannot provide this boundary.
        name = None
        payload = None
        request = None
        worker = None
        self = None

        if operation is not None and outcome_slot is not None:
            outcome, caller_cancel_args = await _wait_for_host_ingress_operation(
                operation, outcome_slot
            )
        operation = None
        outcome_slot = None
        return _deliver_host_ingress_outcome(outcome, caller_cancel_args)

    async def _call_isolated_tool_admitted(
        self,
        name: str,
        args: Dict[str, Any],
        context: Any | None,
    ) -> Dict[str, Any]:
        """Dispatch one tool after :meth:`call_isolated_tool` admitted traffic."""

        try:
            if self._client is None:
                raise RuntimeError("isolated feature client is unavailable")
            if context is None:
                # Preserve the existing wire format for chat/API/legacy calls.
                result = await _maybe_await(self._client.call_tool(name, args))
            else:
                # Context is reserved JSON-RPC metadata.  It must not be merged
                # into ``args``: those remain entirely user-controlled tool
                # input, while the isolated SDK authenticates this envelope.
                result = await _maybe_await(
                    self._client.call_tool(name, args, context=context)
                )
            from kestrel_sovereign.features.base import is_flat_toolresult_envelope
            if is_flat_toolresult_envelope(result):
                # Service returned the flat ToolResult envelope. Pass it through
                # TOP-LEVEL (unified shape #F025) rather than nesting it under
                # ``result`` with a hardcoded ``success: True`` — that hid a
                # service-side ``ToolResult.failed`` behind success and made the
                # honesty layer read the isolated tool as always succeeding
                # (#F018). Derive ``success`` from the service's status.
                envelope = dict(result)
                envelope["tool"] = name
                envelope["success"] = result.get("status") != "error"
                if envelope["success"]:
                    self._maybe_emit_channel_link_part(name)
                return envelope
            # Non-envelope return (a raw value from a legacy service) — keep the
            # wrapped legacy shape so existing readers still see it.
            self._maybe_emit_channel_link_part(name)
            return {
                "success": True,
                "result": result,
                "tool": name,
            }
        except SchedulerExecutionContextUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            if context is not None:
                # No scheduler effect may proceed if the negotiated context was
                # rejected or the context-aware RPC could not be delivered.
                # Raising lets SchedulerRunner persist a failed occurrence and
                # retain its stable idempotency key for recovery.
                raise SchedulerExecutionContextUnavailable(
                    "scheduled isolated tool call rejected its execution context"
                ) from exc
            logger.warning("Isolated feature tool %s.%s failed: %s", self.name, name, exc)
            # Transport/RPC failure — emit the flat error envelope so callers
            # and the honesty layer read a top-level ``status: error``.
            return {
                "status": "error",
                "error": str(exc),
                "tool": name,
                "success": False,
            }

    def _maybe_emit_channel_link_part(self, tool_name: str) -> None:
        """Emit a persisted ``channel_link`` typed part when the bridged
        channel's pairing tool ran on the active streaming turn (#2081).

        The part carries only a reference (``{channel_type}``), not the QR
        bytes: the chat card resolves the current QR state live from
        ``/api/agent/channels/<type>/link-qr.png``. Emitting here — inside the
        tool's host-side execution, which runs within the turn's part-collector
        contextvar — makes the card ride the message that requested the link so
        it persists in that conversation and survives a refresh. ``emit_part``
        is a no-op off a streaming turn, so this is safe on any call path.
        """
        if not self._link_tool or tool_name != self._link_tool or not self._channel_type:
            return
        try:
            from kestrel_sovereign.agent.parts import emit_part

            emit_part("channel_link", {"channel_type": self._channel_type})
        except Exception as exc:  # noqa: BLE001
            logger.debug("channel_link emit_part failed for %s: %s", self.name, exc)

    def resolve_runtime_paths(self) -> tuple[Path, Optional[Path]]:
        # A new resolution invalidates any prior enable cycle's console pin.
        self._validated_hosted_console_path = None
        validated_hosted_overrides: Optional[
            _ValidatedHostedPrebuiltOverrides
        ] = None
        if self._runtime_is_hosted():
            # Revalidate here as well as at construction: tests, embedders, and
            # long-lived hosts can mutate ``os.environ`` between discovery and
            # enable.  A late process-wide path must never acquire provisioning
            # authority merely because the ProxyFeature already exists.
            validated_hosted_overrides = _validate_hosted_process_prebuilt_overrides(
                self.name,
                runtime_venv=self.runtime.venv,
            )
        bin_override = os.environ.get(_env_key(self.name, "BIN"))
        if bin_override:
            # BIN is authoritative for this launch attempt. Do not reflect or
            # forward unused service metadata to the child factory.
            self._service_target = None
            if (
                validated_hosted_overrides is not None
                and validated_hosted_overrides.bin_path is not None
            ):
                return (
                    self._default_venv_path(),
                    validated_hosted_overrides.bin_path,
                )
            return (
                self._default_venv_path(),
                Path(bin_override).expanduser().resolve(),
            )

        # Revalidate at the launch-path mutation boundary. In particular, a
        # BIN override removed after discovery must not expose missing or
        # malformed service metadata to `_service_command`.
        self._service_target = _validated_isolated_service_target(
            self.runtime.service
        )

        venv_override = os.environ.get(_env_key(self.name, "VENV"))
        if venv_override:
            if (
                validated_hosted_overrides is not None
                and validated_hosted_overrides.venv_path is not None
            ):
                return validated_hosted_overrides.venv_path, None
            return Path(venv_override).expanduser().resolve(), None

        if self.runtime.venv:
            if (
                validated_hosted_overrides is not None
                and validated_hosted_overrides.venv_path is not None
            ):
                return validated_hosted_overrides.venv_path, None
            return Path(self.runtime.venv).expanduser().resolve(), None

        return self._default_venv_path(), None

    def _runtime_is_hosted(self) -> bool:
        return (
            self._isolated_runtime_scope is not None
            or getattr(self.agent, "isolated_runtime_hosted", False) is True
        )

    def _required_service_target(self) -> _IsolatedServiceTarget:
        """Return service metadata only on a launch path without BIN."""

        if self._service_target is None:
            self._service_target = _validated_isolated_service_target(
                self.runtime.service
            )
        return self._service_target

    def _probe_feature_distribution(
        self,
        python_path: Path,
    ) -> _FeatureDistributionProbe:
        if self._runtime_is_hosted():
            return _venv_feature_distribution_probe(
                python_path,
                self.runtime.distribution,
                hosted=True,
            )
        return _venv_feature_distribution_probe(
            python_path,
            self.runtime.distribution,
        )

    def _probe_sdk_version(self, python_path: Path) -> str:
        if self._runtime_is_hosted():
            return _venv_sdk_version(python_path, hosted=True)
        return _venv_sdk_version(python_path)

    def _venv_is_overridden(self) -> bool:
        """True when the venv path was supplied by the operator (KESTREL_FEATURE_
        <NAME>_VENV env or the pyproject ``venv =``) rather than provisioned by
        the host at the default path. An operator-supplied venv is NOT ours to
        mutate — see ensure_venv."""
        return bool(
            os.environ.get(_env_key(self.name, "VENV"))
            or self.runtime.venv
        )

    def _process_venv_is_overridden(self) -> bool:
        """Whether this host selected a process-wide immutable venv artifact."""

        return bool(os.environ.get(_env_key(self.name, "VENV")))

    def _hosted_immutable_venv_setting(self) -> Optional[str]:
        """Return the safe setting name selecting an immutable hosted venv."""

        if self._process_venv_is_overridden():
            return _env_key(self.name, "VENV")
        if self.runtime.venv is not None:
            return _HOSTED_RUNTIME_VENV_SETTING
        return None

    def _default_venv_path(self) -> Path:
        return self._feature_runtime_dir() / ".venv"

    def _feature_runtime_dir(self) -> Path:
        """Return this feature's mutable, agent-scoped runtime directory."""
        return (
            self._agent_runtime_dir
            / "feature_venvs"
            / self._runtime_directory_name
        )

    def _prepare_runtime_workspace(self) -> Path:
        """Create the per-feature child-process workspace without touching its venv.

        This intentionally lives beside, rather than inside, ``.venv`` so an
        operator-provided prebuilt environment remains immutable while service
        state, temp files, and user-home/XDG writes stay agent scoped.
        """
        runtime_dir = self._feature_runtime_dir()
        self._venv_relocated_this_startup = False
        if self._isolated_runtime_scope is not None:
            prefix = ("feature_venvs", self._runtime_directory_name)
            legacy_component = self._legacy_runtime_directory_name
            migrations = (
                ((("feature_venvs",), legacy_component, self._runtime_directory_name),)
                if (
                    legacy_component is not None
                    and legacy_component != self._runtime_directory_name
                )
                else ()
            )
            migration_results: set[tuple[tuple[str, ...], str, str]] = set()
            prepare_isolated_runtime_namespace(
                self._isolated_runtime_scope,
                _agent_runtime_owner(self.agent),
                relative_directories=(("feature_venvs",),),
                directory_migrations=migrations,
                migration_results=migration_results,
            )
            self._venv_relocated_this_startup = bool(migration_results)
            if self._released_legacy_runtime_root is not None:
                self._venv_relocated_this_startup = (
                    migrate_released_hosted_feature_runtime(
                        self._released_legacy_runtime_root,
                        self._isolated_runtime_scope,
                        _agent_runtime_owner(self.agent),
                        self.name,
                        self._runtime_directory_name,
                    )
                    or self._venv_relocated_this_startup
                )
            prepare_isolated_runtime_namespace(
                self._isolated_runtime_scope,
                _agent_runtime_owner(self.agent),
                relative_directories=(
                    ("channel_link_artifacts",),
                    prefix,
                    prefix + ("work",),
                    prefix + ("home",),
                    prefix + ("tmp",),
                    prefix + ("config",),
                    prefix + ("data",),
                    prefix + ("cache",),
                    prefix + ("provisioning_cache",),
                ),
            )
            return runtime_dir
        # The storage parent belongs to the standalone operator and may be the
        # process CWD. Never chmod it when it already exists. Core's dedicated
        # child directories below are its private ownership boundary and are
        # created/normalized explicitly so mutable feature state and channel
        # credentials remain inaccessible through a permissive parent.
        if self._agent_runtime_dir.is_symlink():
            raise IsolatedRuntimeNamespaceError(
                "Standalone isolated feature runtime root must not be a symlink."
            )
        try:
            self._agent_runtime_dir.mkdir(
                parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE
            )
        except OSError as exc:
            raise IsolatedRuntimePreparationError(
                "Standalone isolated feature runtime root could not be prepared."
            ) from exc
        for directory in (
            self._agent_runtime_dir / "feature_venvs",
            runtime_dir,
            runtime_dir / "work",
            runtime_dir / "home",
            runtime_dir / "tmp",
            runtime_dir / "config",
            runtime_dir / "data",
            runtime_dir / "cache",
            self._agent_runtime_dir / "channel_link_artifacts",
        ):
            if directory.is_symlink():
                raise IsolatedRuntimeNamespaceError(
                    "Standalone isolated feature runtime workspace must not "
                    "contain symlinks."
                )
            try:
                directory.mkdir(exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
                directory.chmod(_PRIVATE_DIRECTORY_MODE)
            except OSError as exc:
                raise IsolatedRuntimePreparationError(
                    "Standalone isolated feature runtime workspace could not be "
                    "prepared."
                ) from exc
        return runtime_dir

    def _hosted_provisioning_cache_dir(self) -> Path:
        """Return the already-prepared private cache used by hosted uv.

        Workspace creation is descriptor-relative and precedes venv
        provisioning in ``initialize``.  Revalidate the resulting lexical
        path here so a missing mount/workspace becomes a preparation failure,
        while a symlink, foreign owner, or permissive replacement remains a
        containment failure that aborts hosted startup.
        """

        if self._isolated_runtime_scope is None:
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature provisioning has no runtime scope."
            )
        cache_dir = self._feature_runtime_dir() / "provisioning_cache"
        try:
            metadata = cache_dir.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature provisioning cache was not prepared."
            ) from exc
        except OSError as exc:
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature provisioning cache could not be inspected."
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise IsolatedRuntimeNamespaceError(
                "Hosted isolated feature provisioning cache is unsafe."
            )
        if os.name == "posix":
            if metadata.st_uid != os.geteuid():
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature provisioning cache has a foreign owner."
                )
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature provisioning cache is not private."
                )
        return cache_dir

    def _provision_manifest_path(self) -> Path:
        # Inside the venv dir, not its parent: explicit venv overrides can share
        # a parent directory, and a parent-scoped manifest would let sibling
        # features clobber each other's stamp and reinstall on every startup.
        assert self._venv_path is not None
        return self._venv_path / ".kestrel_provision.json"

    def _venv_relocation_repair_pending(self) -> bool:
        """Read durable migration intent from the feature directory."""

        if self._isolated_runtime_scope is None:
            return False
        runtime_dir = self._feature_runtime_dir()
        if not _secure_dirfd_supported():  # pragma: no cover - portable policy
            try:
                return _read_venv_relocation_repair_marker_portable(runtime_dir)
            except IsolatedRuntimeNamespaceError:
                raise
            except OSError as exc:
                raise IsolatedRuntimePreparationError(
                    "Hosted isolated feature relocation repair state could not "
                    "be read."
                ) from exc
        directory_fd: Optional[int] = None
        try:
            directory_fd = os.open(runtime_dir, _directory_open_flags())
            lexical = os.stat(runtime_dir, follow_symlinks=False)
            if not _same_file_identity(lexical, os.fstat(directory_fd)):
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature runtime changed while reading "
                    "relocation repair state."
                )
            return _read_venv_relocation_repair_marker_at(directory_fd)
        except (IsolatedRuntimeNamespaceError, IsolatedRuntimePreparationError):
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature relocation repair path is unsafe."
                ) from exc
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature relocation repair state could not be read."
            ) from exc
        finally:
            if directory_fd is not None:
                os.close(directory_fd)

    def _clear_venv_relocation_repair_marker(self) -> None:
        """Clear migration intent only after launch verification and stamping."""

        if self._isolated_runtime_scope is None:
            return
        runtime_dir = self._feature_runtime_dir()
        if not _secure_dirfd_supported():  # pragma: no cover - portable policy
            marker = runtime_dir / _VENV_RELOCATION_REPAIR_MARKER
            try:
                if not _read_venv_relocation_repair_marker_portable(runtime_dir):
                    return
                marker.unlink()
            except FileNotFoundError:
                return
            except OSError as exc:
                raise IsolatedRuntimePreparationError(
                    "Hosted isolated feature relocation repair state could not "
                    "be finalized."
                ) from exc
            return
        directory_fd: Optional[int] = None
        try:
            directory_fd = os.open(runtime_dir, _directory_open_flags())
            lexical = os.stat(runtime_dir, follow_symlinks=False)
            if not _same_file_identity(lexical, os.fstat(directory_fd)):
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature runtime changed while finalizing "
                    "relocation repair."
                )
            if not _read_venv_relocation_repair_marker_at(directory_fd):
                return
            try:
                os.unlink(
                    _VENV_RELOCATION_REPAIR_MARKER,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                return
            os.fsync(directory_fd)
        except (IsolatedRuntimeNamespaceError, IsolatedRuntimePreparationError):
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise IsolatedRuntimeNamespaceError(
                    "Hosted isolated feature relocation repair path is unsafe."
                ) from exc
            raise IsolatedRuntimePreparationError(
                "Hosted isolated feature relocation repair state could not be "
                "finalized."
            ) from exc
        finally:
            if directory_fd is not None:
                os.close(directory_fd)

    def _read_provision_manifest(self) -> Dict[str, Any]:
        try:
            manifest = json.loads(self._provision_manifest_path().read_text())
        except Exception:  # noqa: BLE001 — missing/corrupt manifest ⇒ reprovision
            return {}
        return manifest if isinstance(manifest, dict) else {}

    def _write_provision_manifest(
        self,
        install_target: str,
        host_sdk: str,
        child_sdk: str,
        feature_distribution_version: str,
        child_feature_distribution: _FeatureDistributionProbe,
    ) -> None:
        self._write_provision_manifest_payload(
            {
                "install_target": install_target,
                # Console-script shebangs embed the venv's absolute interpreter
                # path. A directory adoption therefore makes an otherwise
                # version-current venv stale. Stamp the canonical location so
                # both released-layout and pre-stable moves force exactly one
                # reinstall at their destination.
                "venv_path": str(self._venv_path.resolve()),
                # The host SDK we provisioned AGAINST — staleness keys on a
                # change here, so a genuinely SDK-pinned feature reinstalls
                # once per host bump, not on every startup.
                "provisioned_against_host_sdk": host_sdk,
                # The SDK version that actually landed in the venv (may lag
                # host_sdk if the feature pins it); recorded for diagnosis.
                "child_sdk_version": child_sdk,
                # ``project`` is commonly an unversioned pip target. Keep
                # the host-visible distribution release separately so an
                # upgraded isolated feature cannot keep running an older
                # child merely because its target string did not change.
                "feature_distribution_version": feature_distribution_version,
                # The feature release actually resolved inside the child
                # venv.  It must equal the host-visible desired release
                # whenever that release is known; retaining it makes the
                # successful verification auditable and invalidates old
                # manifests that predate this check once.
                "child_feature_distribution_state": child_feature_distribution.state,
                "child_feature_distribution_version": (
                    child_feature_distribution.version
                ),
            }
        )

    def _write_provision_manifest_payload(self, manifest: Dict[str, Any]) -> None:
        """Atomically publish one private Core provisioning manifest."""

        path = self._provision_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(manifest, indent=2).encode("utf-8")
        temporary = path.with_name(
            f"{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
        )
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                _PRIVATE_FILE_MODE,
            )
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
            path.chmod(_PRIVATE_FILE_MODE)
            if os.name == "posix":
                parent_fd = os.open(path.parent, _directory_open_flags())
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
        except OSError as exc:
            raise IsolatedRuntimePreparationError(
                "Isolated feature provisioning state could not be recorded."
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _console_script_location_state(
        self,
        *,
        validated_script_path: Optional[Path] = None,
    ) -> str:
        """Return ``current``, ``relocated``, ``missing``, or ``not-applicable``.

        A configured console entry point normally embeds the venv's absolute
        path. Python callables launch through the current venv interpreter and
        have no wrapper to verify. Only a positively observed foreign absolute
        shebang proves console relocation when no Core path stamp exists. A
        hosted immutable wrapper symlink is inspected through the already
        resolved and custody-validated target that will be launched.
        """

        if self._bin_path is not None:
            return "not-applicable"
        service_target = self._required_service_target()
        if service_target.console_executable is None:
            return "not-applicable"
        service = service_target.console_executable
        assert self._venv_path is not None
        script = validated_script_path or _console_script_path(
            self._venv_path,
            service,
        )
        executable_name = script.name
        if _secure_dirfd_supported():
            bin_fd: Optional[int] = None
            script_fd: Optional[int] = None
            try:
                bin_fd = os.open(
                    script.parent,
                    _directory_open_flags(),
                )
                metadata = os.stat(
                    executable_name,
                    dir_fd=bin_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(metadata.st_mode):
                    return "missing"
                script_fd = os.open(
                    executable_name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=bin_fd,
                )
                if not _same_file_identity(metadata, os.fstat(script_fd)):
                    return "missing"
                prefix = os.read(script_fd, 8192)
            except OSError:
                return "missing"
            finally:
                if script_fd is not None:
                    os.close(script_fd)
                if bin_fd is not None:
                    os.close(bin_fd)
        else:  # pragma: no cover - portable fallback
            try:
                metadata = script.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    return "missing"
                with script.open("rb") as stream:
                    prefix = stream.read(8192)
            except OSError:
                return "missing"
        if os.name == "nt":  # pragma: no cover - exercised by platform seam tests
            # distlib's Windows console launcher is binary and does not expose
            # a reliably parseable interpreter path. Presence of the exact
            # regular ``.exe`` launch artifact is verifiable; durable rename
            # evidence still forces its reinstall before this check.
            return "current"
        # Console generators embed the lexical venv interpreter path even when
        # ``bin/python`` is itself a symlink to a base interpreter. Resolving
        # that final symlink would falsely classify every repaired uv venv as
        # stale on the next boot.
        current_python = os.fsencode(str(_venv_python(self._venv_path)))
        if current_python in prefix:
            return "current"
        first_line = prefix.splitlines()[0] if prefix else b""
        if first_line.startswith(b"#!"):
            interpreter_parts = first_line[2:].strip().split(maxsplit=1)
            if interpreter_parts:
                interpreter = interpreter_parts[0]
                if interpreter == b"/usr/bin/env":
                    return "current"
                if interpreter.startswith(b"/"):
                    return "relocated"
        # Windows launchers and nonstandard wrappers cannot be safely inferred
        # from an absent text path. A rename observed by Core below remains
        # sufficient evidence to repair them conservatively.
        return "missing"

    def _location_requires_forced_reinstall(
        self,
        manifest: Dict[str, Any],
    ) -> bool:
        assert self._venv_path is not None
        current_path = str(self._venv_path.resolve())
        stamped_path = manifest.get("venv_path")
        stamped_relocation = (
            type(stamped_path) is str
            and bool(stamped_path)
            and stamped_path != current_path
        )
        location_state = self._console_script_location_state()
        relocation_proven = (
            self._venv_relocated_this_startup
            or self._venv_relocation_repair_pending()
            or stamped_relocation
            or location_state == "relocated"
        )
        # A missing or unclassifiable console wrapper is never fresh, even when
        # every manifest field is current. Only ``--reinstall`` is sufficient
        # to recreate an already-satisfied distribution's generated script.
        if location_state == "missing":
            return True
        return relocation_proven and location_state != "not-applicable"

    def _provision_status(
        self,
        install_target: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """Return ``(stale, force_reinstall)`` for a Core-owned venv.

        A location mismatch or an observed foreign shebang proves that an
        entry-point script may still contain the source venv's interpreter.
        A missing stamp alone is legacy metadata, not relocation evidence. If
        a repair crashes before the atomic manifest replacement, either the old
        stamp or the stale wrapper itself causes the repair to be retried.
        """

        if manifest is None:
            manifest = self._read_provision_manifest()
        assert self._venv_path is not None
        if self._location_requires_forced_reinstall(manifest):
            return True, True
        location_stamp_stale = manifest.get("venv_path") != str(
            self._venv_path.resolve()
        )
        return (
            self._provision_is_stale_from_manifest(install_target, manifest)
            or location_stamp_stale,
            False,
        )

    def _adopt_verified_unstamped_venv(
        self,
        install_target: str,
        manifest: Dict[str, Any],
    ) -> bool:
        """Stamp an already-usable Core venv without contacting an index.

        This path applies only when the location stamp is absent or obsolete
        and no stale console wrapper requires repair. Positive child probes
        replace the missing historical stamp; a changed recorded install target
        still takes the ordinary provisioning path.
        """

        assert self._venv_path is not None
        current_path = str(self._venv_path.resolve())
        if manifest.get("venv_path") == current_path:
            return False
        recorded_target = manifest.get("install_target")
        if recorded_target != install_target:
            return False
        if manifest.get("provisioned_against_host_sdk") != _host_sdk_version():
            return False
        # Location metadata is the only field this path may repair. A host SDK,
        # install target, feature release, or child-probe transition must take
        # the ordinary provisioning path rather than being blessed by a fresh
        # manifest assembled from post-hoc probes.
        if self._provision_is_stale_from_manifest(install_target, manifest):
            return False
        if self._console_script_location_state() in {"relocated", "missing"}:
            return False
        desired = _feature_distribution_version(
            self.runtime.distribution,
            install_target,
        )
        child = self._probe_feature_distribution(_venv_python(self._venv_path))
        if not child.is_present:
            return False
        if desired != "unknown" and (
            child.state != "versioned" or child.version != desired
        ):
            return False
        host_sdk = _host_sdk_version()
        child_sdk = self._probe_sdk_version(_venv_python(self._venv_path))
        self._warn_on_sdk_mismatch(
            _venv_python(self._venv_path),
            host_sdk=host_sdk,
            child_sdk=child_sdk,
        )
        # Adoption is still a freshness stamp. Prove that the configured
        # launch target resolves in this exact venv before publishing it.
        self._verify_launch_artifact()
        self._write_provision_manifest(
            install_target,
            host_sdk,
            child_sdk,
            desired,
            child,
        )
        return True

    def _verify_launch_artifact(
        self,
        *,
        validated_console_path: Optional[Path] = None,
    ) -> None:
        """Require the configured launch target to resolve inside this venv."""

        target = self._required_service_target()
        if target.is_callable:
            self._verify_python_callable_target(target)
            return

        state = self._console_script_location_state(
            validated_script_path=validated_console_path,
        )
        if state in {"relocated", "missing"}:
            raise IsolatedRuntimePreparationError(
                "Isolated feature launch artifact could not be verified after "
                "runtime provisioning."
            )

    def _verify_python_callable_target(
        self,
        target: _IsolatedServiceTarget,
    ) -> None:
        """Resolve a callable with the same isolated interpreter used to launch.

        Closed exit codes distinguish an absent module, absent attribute, and
        non-callable object from timeout/spawn/import infrastructure failures.
        Child stdout, stderr, and exception text are deliberately discarded
        because feature import hooks are untrusted and may reflect credentials
        or paths.
        """

        assert target.module is not None
        assert target.callable_name is not None
        assert self._venv_path is not None
        source = (
            "import importlib, os, sys\n"
            "finish = os._exit\n"
            f"module_name = {target.module!r}\n"
            "if sys.version_info < (3, 11) or not sys.flags.safe_path:\n"
            f"    finish({_CALLABLE_TARGET_UNSUPPORTED_INTERPRETER_EXIT})\n"
            "try:\n"
            "    module = importlib.import_module(module_name)\n"
            "except ModuleNotFoundError as error:\n"
            "    missing_name = error.name if type(error) is ModuleNotFoundError "
            "and type(error.name) is str else None\n"
            "    target_missing = missing_name is not None and "
            "(module_name == missing_name or "
            "module_name.startswith(missing_name + '.'))\n"
            f"    finish({_CALLABLE_TARGET_MISSING_MODULE_EXIT} if target_missing "
            f"else {_CALLABLE_TARGET_UNVERIFIABLE_EXIT})\n"
            "except BaseException:\n"
            f"    finish({_CALLABLE_TARGET_UNVERIFIABLE_EXIT})\n"
            "try:\n"
            f"    resolved = getattr(module, {target.callable_name!r})\n"
            "except AttributeError:\n"
            f"    finish({_CALLABLE_TARGET_MISSING_ATTRIBUTE_EXIT})\n"
            "except BaseException:\n"
            f"    finish({_CALLABLE_TARGET_UNVERIFIABLE_EXIT})\n"
            f"finish(0 if callable(resolved) else "
            f"{_CALLABLE_TARGET_NOT_CALLABLE_EXIT})\n"
        )
        runtime_dir = self._feature_runtime_dir()
        hosted = self._runtime_is_hosted()
        try:
            completed = subprocess.run(
                _isolated_python_command(
                    _venv_python(self._venv_path),
                    source,
                ),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_isolated_child_env(
                    self._venv_path,
                    runtime_dir=runtime_dir if hosted else None,
                    hosted=hosted,
                    feature_name=self.name,
                    feature_distribution=self.runtime.distribution,
                ),
                cwd=str(runtime_dir / "work") if hosted else None,
                timeout=_CALLABLE_TARGET_VERIFICATION_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise _IsolatedRuntimeLaunchVerificationTimeoutError(
                "Isolated feature Python callable verification timed out."
            ) from None
        except OSError as exc:
            raise IsolatedRuntimePreparationError(
                "Isolated feature Python callable verification could not start."
            ) from exc
        except subprocess.SubprocessError:
            raise _IsolatedRuntimeLaunchVerificationInfrastructureError(
                "Isolated feature Python callable verification failed."
            ) from None

        target_failures: dict[int, type[_IsolatedRuntimeLaunchTargetPreparationError]] = {
            _CALLABLE_TARGET_MISSING_MODULE_EXIT: (
                _IsolatedRuntimeLaunchTargetMissingModuleError
            ),
            _CALLABLE_TARGET_MISSING_ATTRIBUTE_EXIT: (
                _IsolatedRuntimeLaunchTargetMissingAttributeError
            ),
            _CALLABLE_TARGET_NOT_CALLABLE_EXIT: (
                _IsolatedRuntimeLaunchTargetNotCallableError
            ),
        }
        failure_type = target_failures.get(completed.returncode)
        if failure_type is not None:
            raise failure_type(
                "Isolated feature Python callable target is unavailable."
            )
        if completed.returncode in {
            2,
            _CALLABLE_TARGET_UNSUPPORTED_INTERPRETER_EXIT,
        }:
            raise IsolatedRuntimeConfigurationError(
                reason=_CONFIGURATION_FEATURE_INTERPRETER,
            )
        if completed.returncode != 0:
            raise _IsolatedRuntimeLaunchVerificationInfrastructureError(
                "Isolated feature Python callable verification was inconclusive."
            )

    def _provision_is_stale_from_manifest(
        self,
        install_target: str,
        manifest: Dict[str, Any],
    ) -> bool:
        """Apply version/probe staleness checks to one location-valid stamp."""

        if manifest.get("install_target") != install_target:
            return True
        if manifest.get("provisioned_against_host_sdk") != _host_sdk_version():
            return True
        installed_version = _feature_distribution_version(
            self.runtime.distribution, install_target
        )
        stamped_version = manifest.get("feature_distribution_version")
        # Old manifests predate this stamp. Reprovision once even when the
        # version cannot be observed, then treat an ``unknown`` stamp as stable
        # so local/dynamic targets do not reinstall on every startup.
        if not isinstance(stamped_version, str):
            return True
        if installed_version != "unknown" and stamped_version != installed_version:
            return True
        stamped_child_state = manifest.get("child_feature_distribution_state")
        stamped_child_version = manifest.get("child_feature_distribution_version")
        if stamped_child_state not in {"versioned", "present-unversioned"}:
            return True
        # A fresh manifest proves only what was installed previously. Every
        # unchanged-environment check must probe the child interpreter again:
        # an index repair, manual downgrade, or stale editable metadata can
        # otherwise leave the host running an older service indefinitely.
        child = self._probe_feature_distribution(_venv_python(self._venv_path))
        if not child.is_present:
            return True
        if installed_version != "unknown" and (
            child.state != "versioned" or child.version != installed_version
        ):
            return True
        if (
            stamped_child_state == "versioned"
            and child.version != stamped_child_version
        ):
            return True
        return False

    def _verify_prebuilt_feature_distribution(
        self, python_path: Path, install_target: str
    ) -> None:
        """Fail closed when an immutable override cannot run the desired feature.

        An operator-owned venv must never be installed into or stamped by the
        host, but SDK compatibility alone does not prove that its feature
        distribution is present or current. A positively identified but
        versionless/editable child is accepted only when the desired release is
        genuinely unknown; missing and failed probes are never treated as that
        evidence.
        """

        desired = _feature_distribution_version(
            self.runtime.distribution, install_target
        )
        child = self._probe_feature_distribution(python_path)
        if not child.is_present:
            observed = "missing" if child.state == "missing" else "unverifiable"
            raise IsolatedRuntimePreparationError(
                f"Prebuilt isolated feature {self.name} has {self.runtime.distribution!r} "
                f"version {observed}, but host requires {desired!r}; refusing to run "
                "an unverifiable override venv"
            )
        if desired != "unknown" and (
            child.state != "versioned" or child.version != desired
        ):
            observed = (
                "versionless/editable"
                if child.state == "present-unversioned"
                else repr(child.version)
            )
            raise IsolatedRuntimePreparationError(
                f"Prebuilt isolated feature {self.name} has {self.runtime.distribution!r} "
                f"version {observed}, but host requires {desired!r}; refusing to run "
                "an unverifiable override venv"
            )

    def ensure_venv(self) -> bool:
        """Ensure the runtime environment and report whether Core mutated it."""
        assert self._venv_path is not None
        python_path = _venv_python(self._venv_path)

        # Install the PROJECT (path/dist), never the `service` runnable — the
        # latter is a console-script name or module:callable, not a pip target.
        install_target = self.runtime.project or self.runtime.distribution
        if not install_target:
            raise IsolatedRuntimePreparationError(
                "Isolated feature provisioning metadata has no install target."
            )

        hosted_immutable_setting = (
            self._hosted_immutable_venv_setting()
            if self._runtime_is_hosted()
            else None
        )
        if hosted_immutable_setting is not None:
            self._validated_hosted_console_path = None
            # Revalidate at the mutation boundary, not only at discovery/path
            # resolution. A concurrent host/metadata change or late Core
            # manifest must never make this shared path fall through to uv
            # creation, upgrade, or manifest stamping.
            validated_overrides = _validate_hosted_process_prebuilt_overrides(
                self.name,
                runtime_venv=self.runtime.venv,
            )
            if validated_overrides.venv_path != self._venv_path:
                raise _hosted_prebuilt_override_error(hosted_immutable_setting)
            target = self._required_service_target()
            validated_console_path: Optional[Path] = None
            if target.is_callable:
                # Callable target and infrastructure classifications must stay
                # distinct from selecting-setting custody diagnostics. Verify
                # before the override wrapper and before distribution probing
                # can collapse an unsupported interpreter into a missing
                # package outcome.
                self._verify_launch_artifact()
            try:
                if not target.is_callable:
                    # A console wrapper is part of the selected immutable venv.
                    # Resolve and custody-check the exact launch target first;
                    # missing, mutable, foreign, or relocated artifacts must
                    # name the VENV setting that the operator needs to repair.
                    assert target.console_executable is not None
                    assert validated_overrides.venv_bin_path is not None
                    validated_console_path = _validate_hosted_prebuilt_console(
                        self._venv_path,
                        validated_overrides.venv_bin_path,
                        target.console_executable,
                        setting=hosted_immutable_setting,
                    )
                    self._verify_launch_artifact(
                        validated_console_path=validated_console_path,
                    )
                self._verify_prebuilt_feature_distribution(
                    python_path,
                    install_target,
                )
                self._warn_on_sdk_mismatch(python_path)
            except IsolatedRuntimeConfigurationError:
                raise
            except Exception as exc:
                raise _hosted_prebuilt_override_error(
                    hosted_immutable_setting
                ) from exc
            self._validated_hosted_console_path = validated_console_path
            return False

        exists = python_path.exists()

        # A PREBUILT operator-supplied (override) venv is NOT ours to mutate:
        # running `uv pip install --upgrade` into it would rewrite a prebuilt/
        # pinned environment the operator deliberately provided (and hard-fail
        # the whole feature at startup if the index is unreachable). We recognize
        # a prebuilt override as one that exists at an override path AND carries
        # no provision manifest of ours — i.e. we did not create it. Verify SDK
        # compatibility and warn on a mismatch (See Something Say Something), but
        # leave it untouched and stamp nothing. An override venv WE created
        # earlier (our manifest present) keeps the full reprovision lifecycle, as
        # do host-owned default venvs — both fall through below.
        if (
            exists
            and self._venv_is_overridden()
            and not self._provision_manifest_path().exists()
        ):
            self._verify_launch_artifact()
            self._verify_prebuilt_feature_distribution(python_path, install_target)
            self._warn_on_sdk_mismatch(python_path)
            return False

        # An operator-supplied (override) venv that already exists is NOT ours to
        # mutate: running `uv pip install --upgrade` into it would rewrite a
        # prebuilt/pinned environment the operator deliberately provided (and
        # hard-fail the whole feature at startup if the index is unreachable).
        # Verify SDK compatibility and warn on a mismatch (See Something Say
        # Something), but leave the venv untouched and do not stamp a manifest we
        # don't own. Host-owned default venvs (and a not-yet-created override
        # path we bootstrap below) keep the full reprovision lifecycle.
        force_reinstall = False
        if not exists:
            self._run_provisioning_command(
                ["uv", "venv", str(self._venv_path)]
            )
        else:
            manifest = self._read_provision_manifest()
            stale, force_reinstall = self._provision_status(
                install_target,
                manifest,
            )
            if not stale:
                self._verify_launch_artifact()
                self._clear_venv_relocation_repair_marker()
                return False
            if not force_reinstall and self._adopt_verified_unstamped_venv(
                install_target,
                manifest,
            ):
                self._clear_venv_relocation_repair_marker()
                return False

        # Fresh venv, changed install target, or host SDK upgraded since the
        # venv was provisioned. A moved/unstamped venv requires ``--reinstall``
        # so console scripts are rewritten even when package versions are
        # already satisfied; other stale existing venvs upgrade in place.
        # Stamp only after install and verification both succeed.
        cmd = ["uv", "pip", "install", "--python", str(python_path)]
        if exists:
            cmd.append("--reinstall" if force_reinstall else "--upgrade")
        cmd.append(install_target)
        self._run_provisioning_command(cmd)

        # Verify what actually landed: a feature that pins an older SDK can
        # install "successfully" while keeping the stale wire contract. Surface
        # that rather than silently stamping the venv as fresh (See Something
        # Say Something) — staleness still keys on the host transition so we
        # don't thrash reinstalling a genuinely pinned feature every startup.
        host_sdk = _host_sdk_version()
        child_sdk = self._probe_sdk_version(python_path)
        desired_feature_version = _feature_distribution_version(
            self.runtime.distribution, install_target
        )
        child_feature_distribution = self._probe_feature_distribution(python_path)
        if desired_feature_version != "unknown" and (
            child_feature_distribution.state != "versioned"
            or child_feature_distribution.version != desired_feature_version
        ):
            raise IsolatedRuntimePreparationError(
                "Isolated feature child distribution did not match the host "
                "release after provisioning; the venv was not stamped fresh."
            )
        if not child_feature_distribution.is_present:
            raise IsolatedRuntimePreparationError(
                "Isolated feature child distribution could not be verified after "
                "provisioning; the venv was not stamped fresh."
            )
        # Every install or upgrade must prove the configured launch artifact,
        # not only relocation repairs. Otherwise a resolver can succeed without
        # installing the declared console entry point and Core would stamp that
        # permanently unlaunchable venv as fresh.
        self._verify_launch_artifact()
        self._warn_on_sdk_mismatch(python_path, host_sdk=host_sdk, child_sdk=child_sdk)
        self._write_provision_manifest(
            install_target,
            host_sdk,
            child_sdk,
            desired_feature_version,
            child_feature_distribution,
        )
        self._clear_venv_relocation_repair_marker()
        return True

    def _warn_on_sdk_mismatch(
        self, python_path: Path, *, host_sdk: str = None, child_sdk: str = None
    ) -> None:
        host_sdk = host_sdk if host_sdk is not None else _host_sdk_version()
        child_sdk = (
            child_sdk if child_sdk is not None else self._probe_sdk_version(python_path)
        )
        if child_sdk != host_sdk and "unknown" not in (child_sdk, host_sdk):
            logger.warning(
                "Isolated feature %s venv resolved kestrel-sdk %s but host is %s — "
                "the feature may pin an incompatible wire contract",
                self.name,
                child_sdk,
                host_sdk,
            )

    def _run_provisioning_command(self, cmd: List[str]) -> None:
        """Map expected host provisioning failures to optional quarantine."""

        try:
            self._run(cmd)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise IsolatedRuntimePreparationError(
                "Isolated feature venv provisioning could not be completed."
            ) from exc

    def _run(self, cmd: List[str]) -> None:
        if not cmd:
            raise ValueError("Provisioning command must not be empty")
        if not self._runtime_is_hosted():
            if shutil.which(cmd[0]) is None:
                raise RuntimeError(f"Required executable not found: {cmd[0]}")
            subprocess.run(cmd, check=True)
            return

        assert self._venv_path is not None
        executable, trusted_path = _trusted_host_executable(
            cmd[0],
            excluded_venv=self._venv_path,
        )
        env = _isolated_provisioning_env(
            None,
            include_package_index=True,
            uv_cache_dir=self._hosted_provisioning_cache_dir(),
        )
        env["PATH"] = trusted_path
        subprocess.run([executable, *cmd[1:]], check=True, env=env)

    def _build_client(self, config: Optional[Dict[str, Any]] = None) -> Any:
        factory = self._client_factory
        if factory is None:
            from kestrel_sdk.isolated_feature import SubprocessIsolatedFeatureClient

            factory = SubprocessIsolatedFeatureClient

        runtime_dir = self._feature_runtime_dir()
        hosted = self._runtime_is_hosted()
        child_config = self._host_config if config is None else config
        # SDK 0.35.1's subprocess wrapper does not yet expose a reverse
        # host-capability field in ``clientInfo``.  Its initialize ``config``
        # envelope is nevertheless host-authenticated stdio input. Telegram
        # 0.1.3 needs the acknowledged ingress protocol, so inject its reserved
        # non-persisted marker here and let that service strip it before normal
        # configuration. The copy also means stored user config can never forge
        # or suppress Core's contract.
        child_config = dict(child_config) if isinstance(child_config, dict) else {}
        runtime_distribution = re.sub(
            r"[-_.]+", "-", self.runtime.distribution.strip().lower()
        )
        if runtime_distribution == "kestrel-channel-telegram":
            capabilities = [_ACKNOWLEDGED_CHANNEL_INBOUND_CAPABILITY]
            if self._hosted_telegram_startup_attested:
                capabilities.append(_HOSTED_TELEGRAM_INGRESS_OWNER_CAPABILITY)
            child_config[_HOST_RUNTIME_CAPABILITIES_FIELD] = capabilities
        kwargs = {
            "feature_name": self.name,
            "service": (
                self._service_target.raw
                if self._service_target is not None
                else None
            ),
            "venv_path": str(self._venv_path) if self._venv_path else None,
            "python": str(_venv_python(self._venv_path)) if self._venv_path else None,
            "executable": str(self._bin_path) if self._bin_path else None,
            # Hosted execution is deliberately rooted in its tenant workspace.
            # Standalone execution retains the SDK's historical inherited cwd.
            "cwd": str(runtime_dir / "work") if hosted else None,
            "event_handler": self._handle_event,
            "notification_handler": self._handle_event,
            # An empty object is an explicit effective config: the SDK sends
            # ``config`` only when this value is not ``None``, and its service
            # then calls ``configure({})``. Do not collapse it into a missing
            # config field.
            "config": child_config,
            # Launch env with interpreter-shadowing vars stripped (F023) so the
            # host PYTHONPATH/VIRTUAL_ENV can't defeat the venv isolation.
            "env": _isolated_child_env(
                self._venv_path,
                runtime_dir=runtime_dir if hosted else None,
                hosted=hosted,
                feature_name=self.name,
                feature_distribution=self.runtime.distribution,
            ),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        try:
            signature = inspect.signature(factory)
            params = signature.parameters
        except (TypeError, ValueError) as exc:
            if hosted:
                raise IsolatedRuntimeConfigurationError(
                    reason=_CONFIGURATION_HOSTED_CLIENT_FACTORY,
                ) from exc
            params = {}

        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        )
        if hosted and not accepts_var_kwargs:
            keyword_kinds = {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
            if any(
                name not in params or params[name].kind not in keyword_kinds
                for name in ("env", "cwd")
            ):
                raise IsolatedRuntimeConfigurationError(
                    reason=_CONFIGURATION_HOSTED_CLIENT_FACTORY,
                )

        accepted = (
            dict(kwargs)
            if accepts_var_kwargs
            else {key: value for key, value in kwargs.items() if key in params}
        )

        # Keyword-only factory (named params, no positional `command`): deliver the
        # accepted keyword args directly (config/event handlers/etc.).
        if "command" not in params:
            try:
                if accepted:
                    return factory(**accepted)
                return factory()
            except (TypeError, ValueError) as exc:
                if hosted:
                    raise IsolatedRuntimeConfigurationError(
                        reason=_CONFIGURATION_HOSTED_CLIENT_FACTORY,
                    ) from exc
                raise

        # Positional-command constructor (SubprocessIsolatedFeatureClient): pass the
        # launch argv plus whatever keyword extras the factory accepts (notably
        # `config`, so host config reaches the service via the initialize handshake).
        accepted.pop("command", None)
        command = self._service_command()
        try:
            return factory(command, **accepted)
        except (TypeError, ValueError) as exc:
            if hosted:
                raise IsolatedRuntimeConfigurationError(
                    reason=_CONFIGURATION_HOSTED_CLIENT_FACTORY,
                ) from exc
            return factory(command)

    def _service_command(self) -> List[str]:
        """Build the argv to launch the isolated service.

        Resolution order:
          1. explicit BIN override (``self._bin_path``);
          2. a validated ``module:callable`` through the venv interpreter;
          3. the validated bare console-script name in the venv bin directory.
        Console runnables are never treated as paths or install targets.
        """
        if self._bin_path is not None:
            return [str(self._bin_path)]

        service = self._required_service_target()
        if service.is_callable:
            assert service.module is not None
            assert service.callable_name is not None
            python = (
                str(_venv_python(self._venv_path))
                if self._venv_path is not None
                else "python"
            )
            # The installed SDK requires Python 3.11+, so ``-P`` is the
            # compatible safe-path boundary. Unlike ``-I``, it preserves the
            # hosted stdio encoding variables required by the JSON-RPC pipe;
            # the child environment separately removes Python path injection.
            return _isolated_python_command(
                Path(python),
                (
                    f"from {service.module} import {service.callable_name}; "
                    f"{service.callable_name}()"
                ),
            )

        assert service.console_executable is not None
        if self._validated_hosted_console_path is not None:
            return [str(self._validated_hosted_console_path)]
        if self._venv_path is not None:
            return [
                str(
                    _console_script_path(
                        self._venv_path,
                        service.console_executable,
                    )
                )
            ]
        return [service.console_executable]

    async def _register_event_handler(self, client: Any = None) -> None:
        """Attach the host event handler to a published client.

        Accepting the client explicitly lets fenced recovery keep a started
        candidate detached until durable promotion has completed.
        """

        target = self._client if client is None else client

        async def handle_source_event(event: Any, *, source_client: Any = target) -> None:
            await self._handle_event(event, source_client=source_client)

        register = (
            getattr(target, "set_event_handler", None)
            or getattr(target, "add_event_handler", None)
            or getattr(target, "subscribe", None)
        )
        if register is not None:
            await _maybe_await(register(handle_source_event))
            return

        on_event = getattr(target, "on_event", None)
        if on_event is None:
            return
        try:
            signature = inspect.signature(on_event)
            params = list(signature.parameters.values())
        except (TypeError, ValueError):
            return
        if not params:
            return
        first_name = params[0].name
        if first_name in {"handler", "callback", "event_handler"}:
            await _maybe_await(on_event(handle_source_event))

    def _start_supervision(self) -> asyncio.Task:
        """Start the supervision loop, registered with the agent's background-task
        lifecycle when available so normal agent shutdown cancels it (otherwise
        the child process + task leak — agent shutdown does not call every
        feature's ``shutdown()``). Falls back to a bare task (e.g. under test
        doubles whose ``_track_background_task`` doesn't return a real Task)."""
        name = f"isolated-feature:{self.name}"
        coro = self._supervise()
        tracker = getattr(self.agent, "_track_background_task", None)
        if callable(tracker):
            try:
                task = tracker(coro, name=name)
            except Exception:  # noqa: BLE001
                task = None
            if isinstance(task, asyncio.Task):
                return task
        return asyncio.create_task(coro, name=name)

    def _start_idle_monitor(self) -> None:
        if self._idle_timeout_seconds is None:
            return
        if self._owns_inbound_producer():
            logger.warning(
                "Hosted isolated runtime %s remains resident because it owns "
                "an inbound producer",
                self.name,
            )
            return
        task = self._idle_monitor_task
        if task is not None and not task.done():
            return
        coro = self._monitor_idle_runtime()
        name = f"isolated-feature-idle:{self.name}"
        tracker = getattr(self.agent, "_track_background_task", None)
        tracked = None
        if callable(tracker):
            try:
                tracked = tracker(coro, name=name)
            except Exception:  # noqa: BLE001 - test doubles may reject tracking
                tracked = None
        if isinstance(tracked, asyncio.Task):
            self._idle_monitor_task = tracked
        else:
            self._idle_monitor_task = asyncio.create_task(coro, name=name)

    def _owns_inbound_producer(self) -> bool:
        """Return whether retiring this child would remove its only wake source."""

        if self._hosted_telegram_startup_attested or self._observed_inbound_producer:
            return True
        try:
            capabilities = self._client_capabilities()
            if "channel" in capabilities:
                return True
            return self._new_external_ingress_quiesce() is not None
        except BaseException:  # noqa: BLE001 - ambiguous producer state fails resident
            return True

    async def _monitor_idle_runtime(self) -> None:
        """Retire an inactive child through the existing lifecycle owners."""

        assert self._idle_timeout_seconds is not None
        try:
            while not self._stopping:
                try:
                    baseline = self._last_used_monotonic
                    baseline_generation = self._activity_generation
                    if baseline is None or self._idle_retired:
                        await asyncio.sleep(self._idle_timeout_seconds)
                        continue
                    remaining = (
                        baseline
                        + self._idle_timeout_seconds
                        - asyncio.get_running_loop().time()
                    )
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                        continue
                    retired = await self._retire_idle_generation(
                        expected_activity_generation=baseline_generation,
                        expected_last_used=baseline,
                    )
                    if not retired:
                        if self._owns_inbound_producer():
                            logger.warning(
                                "Hosted isolated runtime %s stopped idle monitoring "
                                "because it now owns an inbound producer",
                                self.name,
                            )
                            return
                        await asyncio.sleep(min(self._idle_timeout_seconds, 1.0))
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - monitor must remain observable/alive
                    logger.warning(
                        "Hosted isolated runtime idle monitor failed for %s; retrying",
                        self.name,
                    )
                    await asyncio.sleep(min(self._idle_timeout_seconds, 1.0))
        except asyncio.CancelledError:
            raise
        finally:
            if self._idle_monitor_task is asyncio.current_task():
                self._idle_monitor_task = None

    async def _retire_idle_generation(
        self, *, expected_activity_generation: int, expected_last_used: float
    ) -> bool:
        """Atomically close admission, recheck the deadline, and retire."""

        assert self._idle_timeout_seconds is not None
        refresh_after_retirement = False
        async with self._reload_lock:
            if self._stopping or self._idle_retired or self._client is None:
                return False
            if self._owns_inbound_producer():
                return False
            self._reloading = True
            reopened = False
            try:
                if not await self._traffic_gate.close_if_idle():
                    return False
                current_last_used = self._last_used_monotonic
                expired = (
                    self._activity_generation == expected_activity_generation
                    and current_last_used == expected_last_used
                    and asyncio.get_running_loop().time()
                    >= expected_last_used + self._idle_timeout_seconds
                )
                if not expired:
                    await self._reopen_traffic_gate()
                    reopened = True
                    return False
                client = self._client
                self._client = None
                retired = await self._retire_detached_client(client)
                if not retired:
                    self._latch_terminal_lifecycle()
                    await self._seal_traffic_gate()
                    # Re-derive uncertainty in terminal cleanup from the exact
                    # retained clients/lifecycle tasks. A failed retirement can
                    # include a still-running bounded facade operation.
                    self._terminal_cleanup_uncertain = False
                    await self._complete_terminal_cleanup(
                        best_effort=True,
                        lifecycle_lock_held=True,
                    )
                    return False
                self._idle_retired = True
                self._idle_resume_event.clear()
                self._terminal_cleanup_uncertain = False
                self._reload_gen += 1
                await self._reopen_traffic_gate()
                reopened = True
                refresh_after_retirement = True
            finally:
                self._reloading = False
                if (
                    not reopened
                    and not self._terminal_lifecycle_latched
                    and self._traffic_gate.closed
                ):
                    await self._reopen_traffic_gate()
        if refresh_after_retirement:
            self._schedule_runtime_telemetry(
                force=True,
                refresh_disk=True,
            )
            return True
        return False

    async def _wake_idle_runtime(self) -> None:
        """Cold-start exactly one generation after idle retirement."""

        task = asyncio.create_task(
            self._wake_idle_runtime_uninterrupted(),
            name=f"isolated-idle-wake:{self.name}",
        )
        try:
            await _await_task_until_complete(task, preserve_cancellation=False)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            raise IsolatedRuntimePreparationError(
                "Isolated feature could not be prepared after idle retirement."
            ) from exc

    async def reclaim_idle_workspace(self) -> RuntimeNamespaceCleanupOutcome:
        """Securely reclaim this exact idle feature tree under the wake lock.

        Hosts must use this seam rather than deleting a path after observing a
        telemetry snapshot. The reload lock turns the eligibility check,
        deletion, and a racing cold wake into one serial transaction.
        """

        if self._isolated_runtime_scope is None:
            return RuntimeNamespaceCleanupOutcome.NOT_HOSTED
        task = asyncio.create_task(
            self._reclaim_idle_workspace_uninterrupted(),
            name=f"isolated-idle-reclaim:{self.name}",
        )
        return await _await_task_until_complete(
            task,
            preserve_cancellation=False,
        )

    async def _reclaim_idle_workspace_uninterrupted(
        self,
    ) -> RuntimeNamespaceCleanupOutcome:
        """Finish deletion and accounting before releasing lifecycle ownership."""

        assert self._isolated_runtime_scope is not None
        # Disk sampling never acquires the lifecycle lock. Taking its lock first
        # prevents deletion beneath a no-follow walk without holding reload
        # ownership while a slow measurement completes.
        async with self._telemetry_disk_lock:
            async with self._reload_lock:
                if (
                    self._terminal_lifecycle_latched
                    or self._stopping
                    or not self._idle_retired
                    or self._client is not None
                    or self._retirement_is_uncertain()
                ):
                    raise IsolatedRuntimePreparationError(
                        "Hosted isolated feature workspace is not idle and reclaimable."
                    )
                outcome = await asyncio.to_thread(
                    _remove_isolated_feature_runtime,
                    self._isolated_runtime_scope,
                    _agent_runtime_owner(self.agent),
                    self._runtime_directory_name,
                )
                self._workspace_reclaim_generation += 1
                self._telemetry_disk_refresh_pending = False
                self._telemetry_environment_refresh_pending = False
                self._environment_bytes = (
                    0
                    if self._bin_path is None and not self._venv_is_overridden()
                    else None
                )
                self._private_writable_bytes = 0
                self._downloaded_bytes = 0
                self._disk_telemetry_status = "complete"
        self._schedule_runtime_telemetry(force=True)
        return outcome

    async def _wake_idle_runtime_uninterrupted(self) -> None:
        """Finish one wake transaction even if its initiating caller cancels."""

        refresh_after_wake = False
        refresh_environment_after_wake = False
        async with self._reload_lock:
            if self._client is not None:
                return
            if not self._idle_retired:
                self._assert_child_start_allowed()
                raise RuntimeError("isolated feature client is unavailable")
            self._assert_child_start_allowed()
            self._reloading = True
            reopened = False
            try:
                await self._close_traffic_gate()
                if self._client is None:
                    self._prepare_runtime_workspace()
                    self._venv_path, self._bin_path = self.resolve_runtime_paths()
                    if self._bin_path is None:
                        await self._ensure_venv_without_blocking_event_loop()
                    # A newly constructed child must read the current durable
                    # generation. Another replica may have changed config while
                    # this process was idle; the memoized prior child config is
                    # never authority for a cold wake.
                    self._host_config = {}
                    self._host_config_loaded = False
                    await self._ensure_host_config_loaded()
                    if self._is_telegram_runtime():
                        await self._resolve_hosted_telegram_startup_attestation()
                    await self._connect_client()
                    self._reload_gen += 1
                    self._idle_wake_count += 1
                    self._last_used_monotonic = asyncio.get_running_loop().time()
                await self._reopen_traffic_gate()
                reopened = True
                refresh_after_wake = True
                refresh_environment_after_wake = (
                    not self._last_cache_hit
                    or self._environment_bytes is None
                )
            except BaseException:
                uncertain = bool(
                    self._terminal_retirement_clients
                    or self._terminal_lifecycle_tasks
                    or self._terminal_cleanup_uncertain
                )
                if uncertain:
                    self._latch_terminal_lifecycle()
                    await self._seal_traffic_gate()
                elif self._client is None:
                    self._idle_retired = True
                    self._idle_resume_event.clear()
                    if self._traffic_gate.closed:
                        await self._reopen_traffic_gate()
                        reopened = True
                else:
                    # The child was published before ancillary telemetry failed.
                    # Keep it running and supervised; never advertise a live
                    # process as idle or safe for workspace reclamation.
                    self._idle_retired = False
                    if self._traffic_gate.closed:
                        await self._reopen_traffic_gate()
                        reopened = True
                raise
            finally:
                self._reloading = False
                if (
                    not reopened
                    and not self._terminal_lifecycle_latched
                    and self._traffic_gate.closed
                ):
                    await self._reopen_traffic_gate()
        if refresh_after_wake:
            self._schedule_runtime_telemetry(
                force=True,
                refresh_disk=True,
                refresh_environment=refresh_environment_after_wake,
            )

    async def _supervise(self) -> None:
        terminal_unwind = False
        try:
            backoff = 1.0
            while not self._stopping:
                await asyncio.sleep(backoff)
                # A ``set_config`` reload intentionally stops/starts the client;
                # don't probe (and "restart") a service that is mid-reload.
                if self._reloading:
                    backoff = 1.0
                    continue
                if self._idle_retired:
                    # Remain quiescent for the whole idle period. Publishing a
                    # fresh child (or terminal shutdown) sets this event.
                    await self._idle_resume_event.wait()
                    backoff = 1.0
                    continue
                # Snapshot the reload generation BEFORE probing: if a reload cycles
                # the client while this (now-stale) probe is in flight, we must not
                # then "restart" the freshly launched client.
                gen = self._reload_gen
                try:
                    client = self._client
                    if client is None:
                        raise RuntimeError(
                            "isolated feature client is unavailable during health probe"
                        )
                    health = await _await_owned_health_probe(
                        client.health(),
                        name=f"isolated-health-probe:{self.name}",
                        on_started=self._own_health_probe_task,
                        on_late_task=self._retain_terminal_health_probe_task,
                    )
                    healthy = self._is_healthy_response(health)
                    if healthy:
                        backoff = 1.0
                        continue
                except asyncio.TimeoutError:
                    logger.warning(
                        "Isolated feature %s health probe exceeded %ss — treating as "
                        "wedged and restarting",
                        self.name,
                        _HEALTH_PROBE_TIMEOUT,
                    )
                except Exception:  # noqa: BLE001 - facade details stay private
                    logger.warning(
                        "Isolated feature %s health check failed", self.name
                    )

                if self._has_running_terminal_health_probe_task():
                    # A stale probe still owns the old facade.  Restarting or
                    # re-enabling beside it would preserve the exact leak this
                    # supervisor exists to prevent, so terminally seal the
                    # cycle.  Cleanup remains bounded: it records incomplete
                    # ownership instead of waiting under lifecycle locks.
                    terminal_unwind = True
                    self._latch_terminal_lifecycle()
                    await self._complete_terminal_cleanup(best_effort=True)
                    break
                if self._stopping:
                    break
                # Serialize the restart against a concurrent reload, and re-check
                # under the lock: a reload may have started during our probe (so
                # the failed probe was expected) or completed with a fresh healthy
                # client. Either way the reload owns the lifecycle — skip.
                async with self._reload_lock:
                    if self._stopping or self._reloading or self._reload_gen != gen:
                        backoff = 1.0
                        continue
                    # Mark ownership before draining. A supervisor cancellation
                    # while an admitted tool is active must still run the final
                    # gate boundary below instead of leaving callers blocked.
                    gate_closed = True
                    stop_completion: _TerminalStopCompletionMarker | None = None
                    try:
                        # A health failure is not a finite reload.  Close new
                        # admission, then let the selected wrapper's bounded
                        # stop/terminate/kill path make its wedged RPC
                        # terminal before we wait for gate drain.  Normal
                        # reload/config transitions deliberately retain the
                        # drain-before-retire order above.
                        await self._close_traffic_gate_admission()
                        client = self._client
                        if client is None:
                            raise RuntimeError(
                                "isolated feature client is unavailable during "
                                "health recovery"
                            )
                        try:
                            # A shutdown can seal/unpublish this same client
                            # while the unhealthy supervisor is already in its
                            # stop await.  Use the terminal retirement lock
                            # even on the healthy-restart path so that race
                            # cannot issue concurrent facade ``stop()`` calls.
                            async with self._terminal_retirement_lock:
                                # Shutdown can latch, unpublish this client,
                                # and queue behind the narrow stop lock while
                                # the supervisor is waiting to acquire it.
                                # Revalidate after acquisition so the stale
                                # recovery owner never stops the terminal
                                # transaction's exact client.
                                if not self._supervisor_owns_client_restart(client, gen):
                                    backoff = 1.0
                                    continue
                                stop_completion = self._begin_terminal_stop_completion(
                                    client
                                )
                                await _await_owned_facade_lifecycle_operation(
                                    client.stop(),
                                    name=f"isolated-supervisor-stop:{self.name}",
                                    on_completed=lambda completion=stop_completion: self._mark_terminal_stop_completed(
                                        completion
                                    ),
                                    on_timeout=lambda client=client, completion=stop_completion: self._fence_terminal_stop_completion_timeout(
                                        client, completion
                                    ),
                                    on_late_task=lambda task, client=client: self._retain_terminal_lifecycle_task(
                                        task, client
                                    ),
                                )
                        except asyncio.CancelledError:
                            # A real caller cancellation can arrive while a
                            # facade stop itself fails/cancels.  Keep only a
                            # callback-proven completion for terminal cleanup;
                            # an uncompleted pending marker would otherwise
                            # retain a non-weak facade after the retry path
                            # has taken ownership of the exact client.
                            if (
                                stop_completion is not None
                                and not stop_completion.completed
                            ):
                                self._discard_terminal_stop_completion(
                                    stop_completion
                                )
                            raise
                        except BaseException:  # noqa: BLE001 - fence hostile stop failures
                            self._discard_terminal_stop_completion(stop_completion)
                            # Never reopen traffic to a client whose stop
                            # outcome is unknown.  The terminal transaction
                            # keeps the exact private retry handle and, unlike
                            # this recovery path, can safely report a failed
                            # retirement to an explicit lifecycle caller.
                            self._latch_terminal_lifecycle()
                            self._retain_terminal_retirement_client(
                                self._unpublish_client(client)
                            )
                            if self._supervision_task is asyncio.current_task():
                                # The shared cleanup runs in its own task. If
                                # it still sees this supervisor as tracked, it
                                # would cancel and later join its own owner
                                # while that owner awaits the cleanup task.
                                self._supervision_task = None
                            await self._complete_terminal_cleanup(
                                best_effort=True,
                                lifecycle_lock_held=True,
                            )
                            break
                        try:
                            await self._drain_traffic_gate()
                        except _TerminalTrafficDrainTimedOut:
                            # A facade can claim stop success while one of its
                            # admitted RPCs remains wedged.  Latch terminal
                            # intent before the recovery finalizer runs, so it
                            # seals rather than briefly reopening admission to
                            # that stopped facade.
                            terminal_unwind = True
                            self._latch_terminal_lifecycle()
                            raise
                        if not self._supervisor_owns_client_restart(client, gen):
                            break
                        await asyncio.sleep(backoff)
                        # Backoff deliberately yields to shutdown/reload.  A
                        # restart decision made before it must not start a
                        # client which has since become terminal or stale.
                        if not self._supervisor_owns_client_restart(client, gen):
                            break
                        try:
                            # A fresh start can spawn another subprocess before
                            # it reports failure, so the prior exact-stop
                            # completion may no longer prove terminal safety.
                            self._forget_terminal_stop_completion(client)
                            started = asyncio.get_running_loop().time()
                            await _await_owned_facade_lifecycle_operation(
                                client.start(),
                                name=f"isolated-supervisor-start:{self.name}",
                                on_timeout=lambda client=client: self._fence_terminal_retirement_timeout(
                                    client
                                ),
                                on_late_task=lambda task, client=client: self._retain_terminal_lifecycle_task(
                                    task, client
                                ),
                            )
                            self._capture_process_identity(client)
                            self._last_cold_start_seconds = (
                                asyncio.get_running_loop().time() - started
                            )
                            self._health_restart_count += 1
                            self._reload_gen += 1
                            self._schedule_runtime_telemetry(force=True)
                            backoff = 1.0
                        except _FacadeLifecycleOperationTimedOut:
                            # A timed-out start can have spawned a child before
                            # its facade stopped responding. It is terminally
                            # uncertain just like a timed-out stop: remove it
                            # from admission and retain its exact handle rather
                            # than reopening traffic behind an unknown process.
                            self._latch_terminal_lifecycle()
                            self._retain_terminal_retirement_client(
                                self._unpublish_client(client)
                            )
                            if self._supervision_task is asyncio.current_task():
                                self._supervision_task = None
                            await self._complete_terminal_cleanup(
                                best_effort=True,
                                lifecycle_lock_held=True,
                            )
                            break
                        except Exception:  # noqa: BLE001 - facade details stay private
                            logger.warning(
                                "Isolated feature %s restart failed", self.name
                            )
                            backoff = min(backoff * 2, 30.0)
                    except asyncio.CancelledError:
                        # This must happen before the gate finalizer.  Without
                        # it, a cancellation during close-admission/recovery
                        # sees ``_stopping`` still false below and briefly
                        # reopens traffic before the outer terminal finally
                        # latches it.
                        terminal_unwind = True
                        self._latch_terminal_lifecycle()
                        raise
                    finally:
                        if gate_closed:
                            if (
                                terminal_unwind
                                or self._stopping
                                or self._terminal_lifecycle_latched
                            ):
                                await self._seal_traffic_gate()
                            else:
                                await self._reopen_traffic_gate()
        finally:
            # If the task is cancelled (e.g. agent shutdown cancelling tracked
            # background tasks) rather than stopped via shutdown(), make sure the
            # child process is still torn down so it can't outlive the agent.
            # This is terminal rather than a finite restart: release pending
            # admissions with the stable fail-closed result, ask the wrapper to
            # terminate a wedged child, and only then drain the old RPC.
            if terminal_unwind or not self._stopping:
                if not terminal_unwind:
                    self._latch_terminal_lifecycle()
                if self._supervision_task is asyncio.current_task():
                    # The shared terminal task may wait for a separately
                    # cancelled supervisor. It must never wait for this task
                    # while this task is waiting for that cleanup to finish.
                    self._supervision_task = None
                shared_cleanup = self._terminal_cleanup_task
                if shared_cleanup is None or shared_cleanup.done():
                    await self._complete_terminal_cleanup(
                        best_effort=True,
                    )
                # A concurrently running shutdown cleanup may already be
                # joining this cancelled supervisor after it fences the exact
                # SDK stop. Awaiting it here would form a supervisor → cleanup
                # → supervisor cycle. The external terminal owner continues to
                # own that shared task; this supervisor can now finish its own
                # cancellation path and let the join settle.

    def _supervisor_owns_client_restart(self, client: Any, generation: int) -> bool:
        """Return whether this health-recovery iteration still owns ``client``.

        This predicate is intentionally reused on both sides of the recovery
        backoff and immediately after terminal-stop lock acquisition.  The
        reload lock serializes normal replacements, while shutdown can latch
        and unpublish outside it to break a drain deadlock; both fences are
        therefore required before the supervisor may touch a facade.
        """

        return (
            not self._stopping
            and not self._terminal_lifecycle_latched
            and self._client is client
            and self._reload_gen == generation
        )

    @staticmethod
    def _is_healthy_response(health: Any) -> bool:
        """Interpret the SDK health envelope without treating an error as ready.

        Legacy client doubles may return a boolean. SDK clients return an
        object, including ``{\"status\": \"restart-required\", \"ready\": false}``
        after a child has been fenced. A non-empty mapping is not evidence of
        readiness, so unknown envelopes fail closed rather than suppressing a
        required replacement.
        """

        if not isinstance(health, dict):
            return bool(health)
        if health.get("replacement_required") is True:
            return False
        if "ready" in health:
            return health["ready"] is True
        if "ok" in health:
            return health["ok"] is True
        if "healthy" in health:
            return health["healthy"] is True
        status = health.get("status")
        if isinstance(status, str):
            return status.lower() in {"ready", "ok", "healthy", "running"}
        return False

    async def _handle_event(self, event: Any, *, source_client: Any = None) -> None:
        # SDK event callbacks are externally visible traffic too: an inbound
        # channel message can wake the agent and trigger effects.  Keep it on
        # the same gate as tools so a candidate hook cannot route an event
        # before its config becomes durable. Events are normally *dropped*
        # rather than queued during a finite close: they originated from the
        # old child and replaying them after promotion could apply stale input
        # under a new configuration. The exceptions below are an acknowledged
        # event from the already-published replacement child, whose producer
        # has retained the source cursor, and a legacy/non-cursor event, which
        # Core holds in a bounded serial queue because dropping it is permanent
        # loss. Terminal drops remain silent
        # because an SDK callback has no caller to handle a deliberate shutdown
        # result.
        source_client = self._client if source_client is None else source_client
        # Channel ingress can run an unbounded cognition turn. The SDK invokes
        # this callback from its one serial notification reader, which is also
        # responsible for receiving private host-ingress RPC responses. Detach
        # the *entire* admission/routing/completion lifecycle before any await
        # that could reach cognition; otherwise cognition calling channels_send
        # can deadlock the reader against its own response stream.
        if self._is_inbound_event(event):
            if source_client is self._client:
                # Event names are part of Core's inbound contract even when a
                # legacy producer omits capability metadata. Once observed,
                # fail resident: this child has no out-of-process wake source.
                self._observed_inbound_producer = True
            await self._schedule_event_ingress_routing(event, source_client)
            return
        try:
            async with self._traffic_gate.admit(wait_for_open=False):
                # The callback retains the client that registered it. Recheck
                # that exact identity *under admission*, before routing or ACK,
                # so a retired child cannot deliver a late message after a
                # replacement has been published.
                if source_client is not self._client:
                    return
                emit_telemetry = await self._handle_event_admitted_from_source(
                    event, source_client
                )
            if emit_telemetry:
                self._schedule_runtime_telemetry()
        except _TrafficGateClosedError:
            # A newly published replacement can begin polling while its parent
            # transition's gate is still closed. Its canonical acknowledged
            # event is already on the durable next config, unlike an old
            # quiesced child event. Hold exactly that one producer update until
            # the gate reopens; never acknowledge an old or unproven source.
            if source_client is self._client:
                self._defer_current_acknowledged_event(event, source_client)
            return
        except _TrafficGateTerminalError:
            return

    @staticmethod
    def _is_inbound_event(event: Any) -> bool:
        kind = _meta_get(event, "type") or _meta_get(event, "event") or _meta_get(event, "kind")
        payload = _meta_get(event, "payload", event)
        if kind == "feature/event":
            event_name = _meta_get(payload, "name") or _meta_get(payload, "event")
        else:
            event_name = kind
        return event_name in {"channel.inbound", "inbound", "message.inbound"}

    @staticmethod
    def _inbound_event_data(event: Any) -> Any:
        """Extract an inbound event payload from either supported wire shape."""

        kind = _meta_get(event, "type") or _meta_get(event, "event") or _meta_get(event, "kind")
        payload = _meta_get(event, "payload", event)
        return _meta_get(payload, "data", payload) if kind == "feature/event" else payload

    async def _schedule_event_ingress_routing(
        self, event: Any, source_client: Any
    ) -> None:
        """Route one inbound callback with its negotiated retention protocol.

        This method deliberately classifies the acknowledgement protocol before
        applying the cursor-source guard. A legacy notification has no cursor
        or NACK to retain it at the child, so dropping a concurrent one is
        permanent loss; it must enter the bounded serial queue instead.
        """

        if self._terminal_lifecycle_latched or source_client is not self._client:
            return
        # The SDK reader owns an untrusted JSON graph. Copy the complete
        # bounded envelope before returning so a child (or a test double) cannot
        # mutate the event between notification receipt and the detached host
        # task's first turn.
        try:
            detached_event = _snapshot_host_ingress_payload(event)
        except (ProtocolError, TypeError, ValueError):
            logger.warning(
                "Ignoring malformed inbound ingress from isolated feature %s",
                self.name,
            )
            return
        data = self._inbound_event_data(detached_event)
        _inbound, acknowledgement = self._split_inbound_event_acknowledgement(data)
        retry = self._inbound_event_retry_completion(data)
        # Telegram descriptors are an all-or-nothing, host-validated cursor
        # protocol.  A malformed pair must not silently fall through to the
        # legacy notification path (where it could bypass the paired
        # ACK/NACK/fencing checks).  Other channels retain their established
        # compatibility handling.
        if (
            self._authoritative_inbound_channel_type() == "telegram"
            and type(data) is dict
            and (
                _EVENT_HOST_INGRESS_ACK_FIELD in data
                or _EVENT_HOST_INGRESS_RETRY_FIELD in data
            )
            and (acknowledgement is None or retry is None)
        ):
            logger.warning(
                "Dropping malformed Telegram polling ingress from isolated feature %s",
                self.name,
            )
            return
        if acknowledgement is None and retry is None:
            await self._enqueue_non_cursor_inbound_event(detached_event, source_client)
            return
        if any(
            client is source_client and not task.done()
            for client, task in self._event_ingress_clients
        ):
            # A cursor-owning source must retain the second update itself. Do
            # not turn a malformed/flooding notification stream into unbounded
            # host tasks or host-memory queueing.
            logger.warning(
                "Ignoring concurrent inbound ingress from isolated feature %s",
                self.name,
            )
            return

        # A legitimate serial provider can emit callback two after callback
        # one's child-side Future is resolved but before Core receives that
        # completion RPC response. Retain exactly that next callback as the
        # single active route task, but do not execute its cognition until the
        # preceding provider completion settles. This keeps an untrusted child
        # from building an unbounded completion queue while preserving the
        # already-emitted callback instead of dropping it.
        completion_predecessors = tuple(
            task
            for client, task in self._event_ack_clients
            if client is source_client and not task.done()
        )

        async def route() -> None:
            retry: _HostIngressRequest | None = None
            try:
                kind = _meta_get(detached_event, "type") or _meta_get(detached_event, "event") or _meta_get(detached_event, "kind")
                payload = _meta_get(detached_event, "payload", detached_event)
                if kind == "feature/event":
                    data = _meta_get(payload, "data", payload)
                else:
                    data = payload
                inbound, acknowledgement = self._split_inbound_event_acknowledgement(data)
                retry = self._inbound_event_retry_completion(data)
                try:
                    async with self._traffic_gate.admit(wait_for_open=False):
                        for predecessor in completion_predecessors:
                            try:
                                await asyncio.shield(predecessor)
                            except asyncio.CancelledError:
                                if (
                                    asyncio.current_task() is not None
                                    and asyncio.current_task().cancelling()
                                ):
                                    raise
                            except BaseException:  # noqa: BLE001 - predecessor audited
                                logger.warning(
                                    "Prior inbound completion from isolated feature %s "
                                    "failed; rechecking the queued callback against "
                                    "source ownership",
                                    self.name,
                                )
                        if source_client is not self._client:
                            return
                        terminal_disposition = self._telegram_terminal_disposition(
                            data,
                            cursor_owned_protocol=(
                                acknowledgement is not None
                                and retry is not None
                                and self._authoritative_inbound_channel_type()
                                == "telegram"
                            ),
                        )
                        admission = await self._route_validated_inbound(
                            inbound,
                            cursor_owned_protocol=(
                                acknowledgement is not None
                                and retry is not None
                                and self._authoritative_inbound_channel_type()
                                == "telegram"
                            ),
                            telegram_terminal_disposition=terminal_disposition,
                        )
                except _TrafficGateClosedError:
                    if source_client is self._client:
                        self._defer_current_acknowledged_event(detached_event, source_client)
                    return
                except _TrafficGateTerminalError:
                    return
                if self._inbound_admission_is_durable(admission):
                    if acknowledgement is not None:
                        self._record_runtime_activity()
                        self._schedule_event_ingress_acknowledgement(
                            source_client, acknowledgement
                        )
                        self._schedule_runtime_telemetry()
                elif retry is not None:
                    self._schedule_event_ingress_retry(source_client, retry)
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - producer retains its cursor
                logger.warning(
                    "Detached inbound ingress from isolated feature %s failed",
                    self.name,
                )
                # A retryable cognitive failure must release the child-side
                # polling callback.  Without this terminal provider response,
                # the update is retained correctly but its serial poll loop is
                # stuck awaiting a completion Core will never send.
                if retry is not None:
                    self._schedule_event_ingress_retry(source_client, retry)

        task = asyncio.create_task(
            route(), name=f"isolated-event-ingress-route:{self.name}"
        )
        self._event_ingress_tasks.add(task)
        self._event_ingress_clients.append((source_client, task))

        def consume_route(completed: asyncio.Task[None]) -> None:
            self._event_ingress_tasks.discard(completed)
            self._event_ingress_clients = [
                (client, owned_task)
                for client, owned_task in self._event_ingress_clients
                if owned_task is not completed
            ]
            if completed.cancelled():
                return
            try:
                completed.result()
            except BaseException:  # noqa: BLE001 - already logged by route
                return

        task.add_done_callback(consume_route)

    async def _enqueue_non_cursor_inbound_event(
        self, detached_event: Any, source_client: Any
    ) -> None:
        """Queue a legacy inbound notification with bounded reader backpressure."""

        while True:
            if (
                self._terminal_lifecycle_latched
                or self._traffic_gate.closed
                or source_client is not self._client
            ):
                return
            async with self._non_cursor_event_ingress_lock:
                queue_entry = next(
                    (
                        entry
                        for entry in self._non_cursor_event_ingress_queues
                        if entry.client is source_client
                        and entry.accepting
                        and entry.worker is not None
                        and not entry.worker.done()
                    ),
                    None,
                )
                if queue_entry is None:
                    queue_entry = _NonCursorIngressQueue(
                        client=source_client,
                        events=asyncio.Queue(
                            maxsize=_MAX_PENDING_NON_CURSOR_INGRESS_EVENTS
                        ),
                    )
                    task = asyncio.create_task(
                        self._route_non_cursor_inbound_events(queue_entry),
                        name=f"isolated-non-cursor-ingress-route:{self.name}",
                    )
                    queue_entry.worker = task
                    self._non_cursor_event_ingress_queues.append(queue_entry)
                    self._event_ingress_tasks.add(task)

                    def consume(
                        completed: asyncio.Task[None],
                        *,
                        entry: _NonCursorIngressQueue = queue_entry,
                    ) -> None:
                        self._event_ingress_tasks.discard(completed)
                        self._non_cursor_event_ingress_queues = [
                            candidate
                            for candidate in self._non_cursor_event_ingress_queues
                            if candidate.worker is not completed
                        ]
                        entry.retired.set()
                        if completed.cancelled():
                            return
                        try:
                            completed.result()
                        except BaseException:  # noqa: BLE001 - worker logs its route errors
                            return

                    task.add_done_callback(consume)

                try:
                    queue_entry.events.put_nowait(detached_event)
                    return
                except asyncio.QueueFull:
                    # A worker owns one current callback and this queue holds
                    # one later callback. Wait outside the lock so the worker
                    # can drain it; no third item allocates host memory.
                    pass

            join = asyncio.create_task(queue_entry.events.join())
            retired = asyncio.create_task(queue_entry.retired.wait())
            done, pending = await asyncio.wait(
                (join, retired), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()

    async def _route_non_cursor_inbound_events(
        self, queue_entry: _NonCursorIngressQueue
    ) -> None:
        """Deliver legacy notifications in source order across finite gate closes."""

        while True:
            detached_event = await queue_entry.events.get()
            retire_worker = False
            try:
                inbound, acknowledgement = self._split_inbound_event_acknowledgement(
                    self._inbound_event_data(detached_event)
                )
                retry = self._inbound_event_retry_completion(
                    self._inbound_event_data(detached_event)
                )
                # The envelope was classified before enqueue. If a bridge
                # changed while it waited, do not accidentally turn this old
                # source into a cursor protocol; source ownership below still
                # prevents it from reaching a replacement child.
                if acknowledgement is not None or retry is not None:
                    logger.warning(
                        "Dropping ingress whose completion protocol changed while queued "
                        "for isolated feature %s",
                        self.name,
                    )
                else:
                    # Legacy ingress is ordered and backpressured while the gate
                    # is open.  Once a live transition closes it, however, this
                    # old-child event is stale and must not replay under the next
                    # configuration.
                    async with self._traffic_gate.admit(wait_for_open=False):
                        if queue_entry.client is not self._client:
                            retire_worker = True
                        else:
                            await self._route_inbound(inbound)
            except (_TrafficGateClosedError, _TrafficGateTerminalError):
                retire_worker = True
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - legacy provider has no retry path
                logger.warning(
                    "Detached legacy inbound ingress from isolated feature %s failed",
                    self.name,
                )
            finally:
                queue_entry.events.task_done()
            async with self._non_cursor_event_ingress_lock:
                if retire_worker or queue_entry.events.empty():
                    # Enqueue checks this flag under the same lock. It either
                    # publishes the next item before this point, or starts a
                    # new worker after this one retires; no notification can
                    # be placed onto an about-to-exit worker and disappear.
                    queue_entry.accepting = False
                    queue_entry.retired.set()
                    return

    async def _handle_event_admitted_from_source(
        self, event: Any, source_client: Any
    ) -> bool:
        """Run one admitted event with its exact client available to the ack path."""

        token = _event_source_client.set(source_client)
        try:
            return await self._handle_event_admitted(event)
        finally:
            _event_source_client.reset(token)

    async def _handle_event_admitted(self, event: Any) -> bool:
        kind = _meta_get(event, "type") or _meta_get(event, "event") or _meta_get(event, "kind")
        payload = _meta_get(event, "payload", event)
        if kind == "feature/event":
            event_name = _meta_get(payload, "name") or _meta_get(payload, "event")
            data = _meta_get(payload, "data", payload)
        else:
            event_name = kind
            data = payload

        if event_name in {"channel.inbound", "inbound", "message.inbound"}:
            inbound, acknowledgement = self._split_inbound_event_acknowledgement(data)
            retry = self._inbound_event_retry_completion(data)
            cursor_owned_protocol = (
                acknowledgement is not None
                and retry is not None
                and self._authoritative_inbound_channel_type() == "telegram"
            )
            admission = await self._route_validated_inbound(
                inbound,
                cursor_owned_protocol=cursor_owned_protocol,
                telegram_terminal_disposition=self._telegram_terminal_disposition(
                    data, cursor_owned_protocol=cursor_owned_protocol
                ),
            )
            source_client = _event_source_client.get()
            if (
                self._inbound_admission_is_durable(admission)
                and acknowledgement is not None
                and source_client is not None
            ):
                self._record_runtime_activity()
                self._schedule_event_ingress_acknowledgement(
                    source_client,
                    acknowledgement,
                )
                return True
            elif retry is not None and source_client is not None:
                self._schedule_event_ingress_retry(source_client, retry)
        elif event_name in {"channel.link_qr", "link_qr", "channel.qr"}:
            await self._route_link_qr(data)
        elif event_name in {"channel.link_cleared", "link_cleared"}:
            await self._route_link_cleared(data)
        return False

    def _split_inbound_event_acknowledgement(
        self, payload: Any
    ) -> tuple[Any, _HostIngressRequest | None]:
        """Extract a bounded post-delivery ingress acknowledgement, if valid.

        A malformed descriptor never causes Core to acknowledge an external
        update. The inbound message can still take the normal ChannelFeature
        path, but its producer will retry from its unchanged durable cursor.
        The acknowledgement payload must name the same stable dedupe key as
        the message itself, preventing a child from acknowledging a different
        bot/update after a successful delivery.
        """

        allowed_payload_keys = {
            frozenset(
                {_EVENT_HOST_INGRESS_MESSAGE_FIELD, _EVENT_HOST_INGRESS_ACK_FIELD}
            ),
            frozenset(
                {
                    _EVENT_HOST_INGRESS_MESSAGE_FIELD,
                    _EVENT_HOST_INGRESS_ACK_FIELD,
                    _EVENT_HOST_INGRESS_RETRY_FIELD,
                }
            ),
        }
        # Telegram's polling protocol adds a bounded, host-validated terminal
        # disposition.  It is intentionally not accepted for generic channel
        # events, where it would otherwise become an unreviewed extension of
        # the acknowledgement envelope.
        if self._authoritative_inbound_channel_type() == "telegram":
            allowed_payload_keys.add(
                frozenset(
                    {
                        _EVENT_HOST_INGRESS_MESSAGE_FIELD,
                        _EVENT_HOST_INGRESS_ACK_FIELD,
                        _EVENT_HOST_INGRESS_RETRY_FIELD,
                        _EVENT_TELEGRAM_TERMINAL_DISPOSITION_FIELD,
                    }
                )
            )
        if type(payload) is not dict or frozenset(payload) not in allowed_payload_keys:
            return payload, None
        message = payload.get(_EVENT_HOST_INGRESS_MESSAGE_FIELD)
        descriptor = payload.get(_EVENT_HOST_INGRESS_ACK_FIELD)
        if type(message) is not dict or type(descriptor) is not dict:
            return message, None
        # Telegram polling is a cursor-owning protocol, not a convention
        # inferred from an arbitrary child descriptor name *or* a child-owned
        # field in an inbound message.  The registered proxy bridge is the
        # only host-negotiated identity that can classify this callback, and
        # therefore the only identity allowed to select its ACK/NACK contract.
        if self._authoritative_inbound_channel_type() == "telegram":
            pair = self._telegram_polling_completion_pair(payload, message)
            return message, pair[0] if pair is not None else None
        if set(descriptor) != {"name", "payload"}:
            return message, None
        request = _prepare_host_ingress_request(
            descriptor.get("name"), descriptor.get("payload")
        )
        if request is None or type(request.payload) is not dict:
            return message, None
        # A polling NACK is meaningful only as the paired retry descriptor;
        # accepting it in the ACK slot could invert a successful delivery.
        if request.name == _TELEGRAM_POLLING_NACK:
            return message, None
        payload_keys = set(request.payload)
        telegram_polling_ack = request.name == _TELEGRAM_POLLING_ACK
        if payload_keys not in ({"dedupe_key"}, {"dedupe_key", "attempt_token"}):
            return message, None
        if telegram_polling_ack and payload_keys != {"dedupe_key", "attempt_token"}:
            return message, None
        message_metadata = message.get("metadata")
        dedupe_key = (
            message_metadata.get("dedupe_key")
            if type(message_metadata) is dict
            else None
        )
        if (
            type(dedupe_key) is not str
            or message.get("id") != dedupe_key
            or request.payload.get("dedupe_key") != dedupe_key
        ):
            return message, None
        attempt_token = request.payload.get("attempt_token")
        if (telegram_polling_ack and type(attempt_token) is not str) or (
            attempt_token is not None
            and (
                type(attempt_token) is not str
                or _EVENT_INGRESS_ATTEMPT_TOKEN_RE.fullmatch(attempt_token) is None
            )
        ):
            return message, None
        return message, request

    def _authoritative_inbound_channel_type(self) -> str | None:
        """Return this proxy's currently registered inbound channel identity.

        The isolated child owns every event field, including ``channel_type``.
        It may not select an allowlist, routing adapter, or cursor-completion
        protocol by changing that field.  A channel is authoritative only
        while this exact proxy adapter remains registered in the host's
        ChannelFeature; a missing/replaced bridge fails closed instead of
        bypassing the sender filter through a generic route.
        """
        channel_type = self._channel_type
        adapter = self._channel_adapter
        if (
            not isinstance(channel_type, str)
            or not channel_type
            or adapter is None
            or getattr(adapter, "channel_type", None) != channel_type
        ):
            return None
        channel_feature = self._channel_feature()
        registry = getattr(channel_feature, "registry", None) if channel_feature else None
        getter = getattr(registry, "get", None)
        if not callable(getter) or getter(channel_type) is not adapter:
            return None
        return channel_type

    @staticmethod
    def _telegram_polling_completion_pair(
        payload: dict[str, Any], message: dict[str, Any]
    ) -> tuple[_HostIngressRequest, _HostIngressRequest] | None:
        """Validate Telegram's exact, attempt-fenced ACK/NACK descriptor pair."""

        if set(payload) not in (
            {
                _EVENT_HOST_INGRESS_MESSAGE_FIELD,
                _EVENT_HOST_INGRESS_ACK_FIELD,
                _EVENT_HOST_INGRESS_RETRY_FIELD,
            },
            {
                _EVENT_HOST_INGRESS_MESSAGE_FIELD,
                _EVENT_HOST_INGRESS_ACK_FIELD,
                _EVENT_HOST_INGRESS_RETRY_FIELD,
                _EVENT_TELEGRAM_TERMINAL_DISPOSITION_FIELD,
            },
        ):
            return None
        acknowledgement_descriptor = payload.get(_EVENT_HOST_INGRESS_ACK_FIELD)
        retry_descriptor = payload.get(_EVENT_HOST_INGRESS_RETRY_FIELD)
        if (
            type(acknowledgement_descriptor) is not dict
            or type(retry_descriptor) is not dict
            or set(acknowledgement_descriptor) != {"name", "payload"}
            or set(retry_descriptor) != {"name", "payload"}
        ):
            return None
        acknowledgement = _prepare_host_ingress_request(
            acknowledgement_descriptor.get("name"),
            acknowledgement_descriptor.get("payload"),
        )
        retry = _prepare_host_ingress_request(
            retry_descriptor.get("name"), retry_descriptor.get("payload")
        )
        if (
            acknowledgement is None
            or retry is None
            or acknowledgement.name != _TELEGRAM_POLLING_ACK
            or retry.name != _TELEGRAM_POLLING_NACK
            or type(acknowledgement.payload) is not dict
            or type(retry.payload) is not dict
            or set(acknowledgement.payload) != {"dedupe_key", "attempt_token"}
            or set(retry.payload) != {"dedupe_key", "attempt_token"}
        ):
            return None
        metadata = message.get("metadata")
        dedupe_key = metadata.get("dedupe_key") if type(metadata) is dict else None
        acknowledgement_token = acknowledgement.payload.get("attempt_token")
        retry_token = retry.payload.get("attempt_token")
        if (
            type(dedupe_key) is not str
            or message.get("id") != dedupe_key
            or acknowledgement.payload.get("dedupe_key") != dedupe_key
            or retry.payload.get("dedupe_key") != dedupe_key
            or type(acknowledgement_token) is not str
            or type(retry_token) is not str
            or _EVENT_INGRESS_ATTEMPT_TOKEN_RE.fullmatch(acknowledgement_token)
            is None
            or _EVENT_INGRESS_ATTEMPT_TOKEN_RE.fullmatch(retry_token) is None
            or not secrets.compare_digest(acknowledgement_token, retry_token)
        ):
            return None
        return acknowledgement, retry

    def _inbound_event_retry_completion(
        self, payload: Any
    ) -> _HostIngressRequest | None:
        """Return a validated retry completion paired to one inbound ACK key."""

        if type(payload) is not dict or set(payload) not in (
            {
                _EVENT_HOST_INGRESS_MESSAGE_FIELD,
                _EVENT_HOST_INGRESS_ACK_FIELD,
                _EVENT_HOST_INGRESS_RETRY_FIELD,
            },
            {
                _EVENT_HOST_INGRESS_MESSAGE_FIELD,
                _EVENT_HOST_INGRESS_ACK_FIELD,
                _EVENT_HOST_INGRESS_RETRY_FIELD,
                _EVENT_TELEGRAM_TERMINAL_DISPOSITION_FIELD,
            },
        ):
            return None
        message, acknowledgement = self._split_inbound_event_acknowledgement(payload)
        descriptor = payload.get(_EVENT_HOST_INGRESS_RETRY_FIELD)
        if (
            acknowledgement is None
            or type(message) is not dict
            or type(descriptor) is not dict
            or set(descriptor) != {"name", "payload"}
        ):
            return None
        request = _prepare_host_ingress_request(
            descriptor.get("name"), descriptor.get("payload")
        )
        if request is None or type(request.payload) is not dict:
            return None
        message_metadata = message.get("metadata")
        dedupe_key = (
            message_metadata.get("dedupe_key")
            if type(message_metadata) is dict
            else None
        )
        telegram_polling_pair = (
            acknowledgement.name == _TELEGRAM_POLLING_ACK
            or request.name == _TELEGRAM_POLLING_NACK
        )
        if (
            type(dedupe_key) is not str
            or set(request.payload) not in (
                {"dedupe_key"}, {"dedupe_key", "attempt_token"}
            )
            or request.payload.get("dedupe_key") != dedupe_key
        ):
            return None
        if telegram_polling_pair and (
            acknowledgement.name != _TELEGRAM_POLLING_ACK
            or request.name != _TELEGRAM_POLLING_NACK
            or set(request.payload) != {"dedupe_key", "attempt_token"}
        ):
            return None
        attempt_token = request.payload.get("attempt_token")
        if (telegram_polling_pair and type(attempt_token) is not str) or (
            attempt_token is not None
            and (
            type(attempt_token) is not str
            or _EVENT_INGRESS_ATTEMPT_TOKEN_RE.fullmatch(attempt_token) is None
            )
        ):
            return None
        acknowledgement_token = acknowledgement.payload.get("attempt_token")
        if telegram_polling_pair and not secrets.compare_digest(
            acknowledgement_token, attempt_token
        ):
            return None
        return request

    @staticmethod
    def _inbound_admission_is_durable(admission: Any) -> bool:
        return getattr(admission, "durably_admitted", False) is True

    def _event_ingress_acknowledgement(
        self, event: Any
    ) -> _HostIngressRequest | None:
        """Return a valid acknowledgement descriptor only for inbound events."""

        kind = _meta_get(event, "type") or _meta_get(event, "event") or _meta_get(event, "kind")
        payload = _meta_get(event, "payload", event)
        if kind == "feature/event":
            event_name = _meta_get(payload, "name") or _meta_get(payload, "event")
            data = _meta_get(payload, "data", payload)
        else:
            event_name = kind
            data = payload
        if event_name not in {"channel.inbound", "inbound", "message.inbound"}:
            return None
        _, acknowledgement = self._split_inbound_event_acknowledgement(data)
        return acknowledgement

    def _defer_current_acknowledged_event(self, event: Any, source_client: Any) -> None:
        """Hold one new-child polling event until a finite gate reopens."""

        kind = _meta_get(event, "type") or _meta_get(event, "event") or _meta_get(event, "kind")
        payload = _meta_get(event, "payload", event)
        if kind == "feature/event":
            event_name = _meta_get(payload, "name") or _meta_get(payload, "event")
            payload = _meta_get(payload, "data", payload)
        else:
            event_name = kind
        if event_name not in {"channel.inbound", "inbound", "message.inbound"}:
            return
        message, acknowledgement = self._split_inbound_event_acknowledgement(payload)
        retry = self._inbound_event_retry_completion(payload)
        if acknowledgement is None or type(message) is not dict:
            return
        try:
            detached_message = _snapshot_host_ingress_payload(message)
        except BaseException:  # noqa: BLE001 - producer keeps its cursor
            logger.warning(
                "Dropping malformed deferred acknowledged ingress from isolated feature %s",
                self.name,
            )
            return
        if type(detached_message) is not dict:
            return
        deferred = _DeferredAcknowledgedIngress(
            message=detached_message,
            acknowledgement=acknowledgement,
            source_client=source_client,
            retry=retry,
            telegram_terminal_disposition=self._telegram_terminal_disposition(
                payload,
                cursor_owned_protocol=(
                    acknowledgement is not None
                    and retry is not None
                    and self._authoritative_inbound_channel_type() == "telegram"
                ),
            ),
        )
        if len(self._deferred_acknowledged_event_tasks) >= _MAX_DEFERRED_ACKNOWLEDGED_EVENTS:
            logger.warning(
                "Dropping excess deferred acknowledged ingress from isolated feature %s",
                self.name,
            )
            return

        async def deliver_after_reopen() -> None:
            try:
                emit_telemetry = False
                async with self._traffic_gate.admit():
                    # A second transition can replace this child before the
                    # first gate reopens. Its event must remain unacknowledged
                    # so the producer restarts from the same durable cursor.
                    if deferred.source_client is not self._client:
                        return
                    admission = await self._route_validated_inbound(
                        deferred.message,
                        cursor_owned_protocol=(
                            deferred.retry is not None
                            and self._authoritative_inbound_channel_type()
                            == "telegram"
                        ),
                        telegram_terminal_disposition=(
                            deferred.telegram_terminal_disposition
                        ),
                    )
                    if self._inbound_admission_is_durable(admission):
                        self._record_runtime_activity()
                        self._schedule_event_ingress_acknowledgement(
                            deferred.source_client,
                            deferred.acknowledgement,
                        )
                        emit_telemetry = True
                    elif deferred.retry is not None:
                        self._schedule_event_ingress_retry(
                            deferred.source_client, deferred.retry
                        )
                if emit_telemetry:
                    self._schedule_runtime_telemetry()
            except (_TrafficGateClosedError, _TrafficGateTerminalError):
                return
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - producer retains its cursor
                logger.warning(
                    "Deferred acknowledged ingress from isolated feature %s failed",
                    self.name,
                )
                # Match detached ingress: a validated NACK reaches the exact
                # still-live facade after a routing exception, while any
                # replacement/terminal lifecycle state retains the provider
                # callback for its normal retry instead.
                if deferred.retry is not None:
                    self._schedule_event_ingress_retry(
                        deferred.source_client, deferred.retry
                    )

        task = asyncio.create_task(
            deliver_after_reopen(),
            name=f"isolated-deferred-ingress:{self.name}",
        )
        self._deferred_acknowledged_event_tasks.add(task)

        def consume_deferred_event(completed: asyncio.Task[None]) -> None:
            self._deferred_acknowledged_event_tasks.discard(completed)
            if completed.cancelled():
                return
            try:
                completed.result()
            except BaseException:  # noqa: BLE001 - failure already logged
                return

        task.add_done_callback(consume_deferred_event)

    def _schedule_event_ingress_acknowledgement(
        self,
        source_client: Any,
        request: _HostIngressRequest,
    ) -> None:
        """Acknowledge durably admitted ingress without blocking the SDK reader."""

        self._schedule_event_ingress_completion(
            source_client, request, expected_state="acknowledged", kind="ack"
        )

    def _schedule_event_ingress_retry(
        self,
        source_client: Any,
        request: _HostIngressRequest,
    ) -> None:
        """Release a retryable provider callback without advancing its cursor."""

        self._schedule_event_ingress_completion(
            source_client, request, expected_state="retrying", kind="retry"
        )

    def _schedule_event_ingress_completion(
        self,
        source_client: Any,
        request: _HostIngressRequest,
        *,
        expected_state: str,
        kind: str,
    ) -> None:
        """Complete a provider callback after detached routing has settled.

        ``IsolatedFeatureClient`` reads notifications and JSON-RPC responses on
        one task. Awaiting ``call_host_ingress`` in that notification handler
        would deadlock waiting for a response the same reader cannot consume.
        Schedule the callback only after :meth:`_route_inbound` completes; the
        task is queued before the gate releases, so a following quiesce sees
        the acknowledgement request before it waits for the producer callback.
        It deliberately calls the exact source client rather than the current
        proxy slot, allowing a pre-gate acknowledgement to finish while a
        lifecycle transition owns ordinary admission.
        """

        if self._terminal_lifecycle_latched or source_client is not self._client:
            return
        # Routing finishes before the provider RPC response comes back. A
        # cross-process child can therefore deliver callback two while callback
        # one is still awaiting its ACK response. Keep the source serialized,
        # but do not discard that already-routed completion: queue it behind
        # the exact active completions in arrival order.
        predecessors = tuple(
            task
            for client, task in self._event_ack_clients
            if client is source_client and not task.done()
        )

        try:
            capabilities = getattr(source_client, "host_ingress_capabilities", None)
            names = getattr(capabilities, "names", ())
            if request.name not in names:
                logger.warning(
                    "Inbound event from isolated feature %s requested an unavailable completion",
                    self.name,
                )
                return
            call = getattr(source_client, "call_host_ingress", None)
        except BaseException:  # noqa: BLE001 - untrusted facade stays unacknowledged
            logger.warning(
                "Inbound event from isolated feature %s could not start completion",
                self.name,
            )
            return
        if not callable(call):
            logger.warning(
                "Inbound event from isolated feature %s requested an unsupported completion",
                self.name,
            )
            return

        async def complete() -> None:
            for predecessor in predecessors:
                try:
                    # Shield the predecessor so shutdown/cancellation of this
                    # queued completion cannot create a second completion RPC
                    # for the same provider callback.
                    await asyncio.shield(predecessor)
                except asyncio.CancelledError:
                    if (
                        asyncio.current_task() is not None
                        and asyncio.current_task().cancelling()
                    ):
                        raise
                    # A predecessor may have been cancelled by an older
                    # lifecycle transition. Recheck the current source below;
                    # if it remains current, this exact callback still needs
                    # its own completion attempt.
                except BaseException:  # noqa: BLE001 - predecessor already audited
                    logger.warning(
                        "Prior inbound completion from isolated feature %s failed; "
                        "continuing the queued provider callback",
                        self.name,
                    )
            for attempt in range(_EVENT_INGRESS_ACK_ATTEMPTS):
                if self._terminal_lifecycle_latched or source_client is not self._client:
                    return
                settled, result = await self._await_event_ingress_ack_attempt(
                    call, request, source_client
                )
                if settled and self._is_event_ingress_completion(result, expected_state):
                    return
                if not settled:
                    # The exact RPC rejected cancellation or belongs to a
                    # stopped loop. It remains lifecycle-owned and the source
                    # is sealed rather than risking a second concurrent ACK.
                    self._fence_event_ingress_ack_source(source_client)
                    return
                if attempt + 1 < _EVENT_INGRESS_ACK_ATTEMPTS:
                    await asyncio.sleep(_EVENT_INGRESS_ACK_BACKOFF * (attempt + 1))
            logger.warning(
                "Inbound event %s completion for isolated feature %s exhausted retries",
                kind,
                self.name,
            )
            # Telegram's pending completion keeps its getUpdates offset
            # unchanged; retiring this exact source makes it redeliver on a
            # fresh process instead of silently losing the update.
            self._fence_event_ingress_ack_source(source_client)

        task = asyncio.create_task(
            complete(),
            name=f"isolated-event-ingress-{kind}:{self.name}",
        )
        self._event_ack_tasks.add(task)
        self._event_ack_clients.append((source_client, task))

        def consume_acknowledgement(completed: asyncio.Task[None]) -> None:
            self._event_ack_tasks.discard(completed)
            self._event_ack_clients = [
                (client, owned_task)
                for client, owned_task in self._event_ack_clients
                if owned_task is not completed
            ]
            if completed.cancelled():
                return
            try:
                completed.result()
            except BaseException:  # noqa: BLE001 - completion already logged
                return

        task.add_done_callback(consume_acknowledgement)

    async def _await_event_ingress_ack_attempt(
        self,
        call: Callable[[str, HostIngressPayload], Any],
        request: _HostIngressRequest,
        source_client: Any,
    ) -> tuple[bool, Any]:
        """Run one ACK with a bounded owned timeout."""

        operation = _create_host_owned_facade_task(
            _maybe_await(call(request.name, request.payload)),
            name=f"isolated-event-ingress-ack-rpc:{self.name}",
        )
        try:
            result = await asyncio.wait_for(
                operation.shield(), timeout=_EVENT_INGRESS_ACK_TIMEOUT
            )
            return True, result
        except asyncio.TimeoutError:
            operation.cancel()
            try:
                await asyncio.wait_for(
                    operation.shield(), timeout=_EVENT_INGRESS_ACK_CANCELLATION_GRACE
                )
            except asyncio.TimeoutError:
                self._retain_terminal_lifecycle_task(operation, source_client)
                return False, None
            except BaseException:  # noqa: BLE001 - failed ACK remains retryable
                _consume_late_lifecycle_task_outcome(operation)
                return True, None
            _consume_late_lifecycle_task_outcome(operation)
            return True, None
        except asyncio.CancelledError:
            operation.cancel()
            if not operation.done():
                self._retain_terminal_lifecycle_task(operation, source_client)
            raise
        except BaseException:  # noqa: BLE001 - producer retries unchanged cursor
            _consume_late_lifecycle_task_outcome(operation)
            return True, None

    def _fence_event_ingress_ack_source(self, source_client: Any) -> None:
        """Terminally remove an ACK source after bounded retries fail."""

        if source_client is not self._client:
            return
        self._latch_terminal_lifecycle()
        # Establish the shared terminal owner synchronously. It unpublishes and
        # retires this exact facade; no successful cleanup can race a later ACK
        # because scheduling is blocked by the latch above.
        self.prepare_shutdown_with_agent_deadline()

    @staticmethod
    def _is_event_ingress_acknowledged(value: Any) -> bool:
        """Accept the narrow, private completion envelope only."""

        return ProxyFeature._is_event_ingress_completion(value, "acknowledged")

    @staticmethod
    def _is_event_ingress_completion(value: Any, expected_state: str) -> bool:
        """Accept the narrow private ACK or retry-completion envelope."""

        if type(value) is not dict:
            return False
        allowed = {
            "status",
            "http_status",
            "state",
            "already_acknowledged",
            "already_retrying",
        }
        if not set(value).issubset(allowed):
            return False
        if (
            value.get("status") != "ok"
            or value.get("http_status") != 200
            or value.get("state") != expected_state
        ):
            return False
        return all(
            type(value[key]) is bool
            for key in ("already_acknowledged", "already_retrying")
            if key in value
        )

    async def _route_link_cleared(self, payload: Any) -> None:
        """Retract a channel pairing QR once the channel is linked.

        Removes the persisted PNG so the ``channel_link`` card (#2081) resolves
        to "expired or already linked" on its next fetch. No sticky/SSE state to
        clear anymore — the card is a persisted typed part, not a live bubble.
        """
        if not isinstance(payload, dict):
            return
        channel_type = str(payload.get("channel_type") or "").strip().lower()
        if not channel_type or not re.fullmatch(r"[a-z0-9_]{1,32}", channel_type):
            return
        png = (
            self._agent_runtime_dir
            / "channel_link_artifacts"
            / f"{channel_type}_link_qr.png"
        )
        try:
            await asyncio.to_thread(_unlink_private_artifact, png)
        except (
            IsolatedRuntimeNamespaceError,
            IsolatedRuntimePreparationError,
            OSError,
        ) as exc:
            logger.warning(
                "Failed to clear channel-link artifact for %s: %s",
                self.name,
                exc,
            )

    async def _route_link_qr(self, payload: Any) -> None:
        """Persist the latest channel pairing-QR PNG under the agent data dir.

        Isolated channel features emit ``channel.link_qr`` with the QR rendered
        as a PNG (base64) each time a pairing code is produced (it rotates
        ~20s). The host writes the latest under the agent data dir, served by
        ``/api/agent/channels/{type}/link-qr.png``. The chat's persisted
        ``channel_link`` card (#2081) fetches that endpoint on render/refresh —
        so the QR is no longer pushed as a live SSE bubble (which orphaned on
        refresh, #1918); the PNG is simply the current state the card resolves.
        """
        if not isinstance(payload, dict):
            return
        channel_type = str(payload.get("channel_type") or "").strip().lower()
        png_b64 = payload.get("png_b64") or payload.get("png")
        if (
            not channel_type
            or not re.fullmatch(r"[a-z0-9_]{1,32}", channel_type)
            or not png_b64
        ):
            logger.warning("Dropping malformed channel.link_qr from %s", self.name)
            return
        if type(png_b64) is not str:
            logger.warning("Dropping malformed channel.link_qr from %s", self.name)
            return
        max_encoded_bytes = ((_MAX_PRIVATE_ARTIFACT_BYTES + 2) // 3) * 4
        # Accept ordinary MIME wrapping without making whitespace an unbounded
        # child-controlled input.  A 6.25% allowance covers CRLF every 76
        # columns, while the normalized alphabet and decoded payload retain
        # their independent exact bounds below.
        max_wrapping_bytes = max(16, max_encoded_bytes // 16)
        if len(png_b64) > max_encoded_bytes + max_wrapping_bytes:
            logger.warning("Dropping oversized channel.link_qr PNG from %s", self.name)
            return
        try:
            encoded_png = png_b64.encode("ascii")
        except (UnicodeEncodeError, ValueError) as exc:
            logger.warning(
                "channel.link_qr from %s had undecodable PNG: %s", self.name, exc
            )
            return
        normalized_png = encoded_png.translate(None, _ASCII_BASE64_WHITESPACE)
        if not normalized_png:
            logger.warning("Dropping malformed channel.link_qr from %s", self.name)
            return
        if len(encoded_png) - len(normalized_png) > max(
            16, len(normalized_png) // 16
        ):
            logger.warning("Dropping oversized channel.link_qr PNG from %s", self.name)
            return
        if len(normalized_png) > max_encoded_bytes:
            logger.warning("Dropping oversized channel.link_qr PNG from %s", self.name)
            return
        try:
            png_bytes = base64.b64decode(normalized_png, validate=True)
        except ValueError as exc:
            logger.warning(
                "channel.link_qr from %s had undecodable PNG: %s", self.name, exc
            )
            return
        if len(png_bytes) > _MAX_PRIVATE_ARTIFACT_BYTES:
            logger.warning("Dropping oversized channel.link_qr PNG from %s", self.name)
            return

        out = (
            self._agent_runtime_dir
            / "channel_link_artifacts"
            / f"{channel_type}_link_qr.png"
        )
        try:
            await asyncio.to_thread(_write_private_artifact, out, png_bytes)
        except (
            IsolatedRuntimeNamespaceError,
            IsolatedRuntimePreparationError,
            OSError,
        ) as exc:
            logger.warning(
                "Failed to persist channel.link_qr PNG for %s: %s", self.name, exc
            )
            return

    def _telegram_terminal_disposition(
        self, data: Any, *, cursor_owned_protocol: bool
    ) -> str | None:
        """Validate one bounded, non-cognitive Telegram terminal descriptor."""

        if not isinstance(data, dict):
            return None
        descriptor = data.get(_EVENT_TELEGRAM_TERMINAL_DISPOSITION_FIELD)
        if descriptor is None:
            return None
        if (
            self._authoritative_inbound_channel_type() != "telegram"
            or not cursor_owned_protocol
            or type(descriptor) is not dict
            or set(descriptor) != {"kind"}
            or descriptor.get("kind") not in _TELEGRAM_TERMINAL_DISPOSITIONS
        ):
            raise ProtocolError("invalid Telegram terminal inbound disposition")
        return descriptor["kind"]

    async def _route_validated_inbound(
        self,
        payload: Any,
        *,
        cursor_owned_protocol: bool,
        telegram_terminal_disposition: str | None = None,
    ) -> Any:
        """Route with a host-only protocol classification.

        The public/legacy ``_route_inbound`` shape remains a one-argument
        method: integrations and test doubles use it as that narrow seam.
        Context-local classification lets a detached callback carry the
        negotiated protocol through that seam without trusting its JSON body
        or widening every legacy ChannelFeature implementation.
        """

        token = _cursor_owned_inbound_protocol.set(cursor_owned_protocol)
        terminal_token = _telegram_terminal_inbound_disposition.set(
            telegram_terminal_disposition
        )
        try:
            return await self._route_inbound(payload)
        finally:
            _cursor_owned_inbound_protocol.reset(token)
            _telegram_terminal_inbound_disposition.reset(terminal_token)

    async def admit_hosted_telegram_ingress(
        self,
        payload: dict[str, Any],
        *,
        terminal_disposition: str | None = None,
    ) -> Any:
        """Durably admit one host-authenticated Telegram webhook result.

        Frinz first gives the authenticated provider update to the isolated
        child, which validates its active binding and normalizes it without
        retaining a bot token.  A hosted delivery has no child-side polling
        cursor/RPC pair, so the host must explicitly drive its resulting event
        through the same Core durable channel boundary before returning HTTP
        success to Telegram.  This narrow host API preserves the proxy-owned
        agent/channel binding and refuses ordinary polling or un-attested
        children from manufacturing hosted admission.
        """

        if (
            not self._is_telegram_runtime()
            or not self._hosted_telegram_startup_attested
            or type(payload) is not dict
        ):
            return None
        if (
            terminal_disposition is not None
            and terminal_disposition not in _TELEGRAM_TERMINAL_DISPOSITIONS
        ):
            raise ProtocolError("invalid hosted Telegram terminal disposition")
        return await self._route_validated_inbound(
            payload,
            cursor_owned_protocol=True,
            telegram_terminal_disposition=terminal_disposition,
        )

    async def _route_inbound(self, payload: Any) -> Any:
        channel = self._channel_feature()
        terminal_disposition = _telegram_terminal_inbound_disposition.get()
        expected_method = (
            "handle_terminal_inbound" if terminal_disposition is not None else "handle_inbound"
        )
        if channel is None or not hasattr(channel, expected_method):
            logger.warning(
                "Inbound notification from %s dropped: ChannelFeature unavailable",
                self.name,
            )
            return None

        authoritative_channel_type = self._authoritative_inbound_channel_type()
        if authoritative_channel_type is None:
            logger.warning(
                "Inbound notification from %s dropped: no registered host channel bridge",
                self.name,
            )
            return None

        from kestrel_sovereign.features.channels.models import ChannelMessage

        message = payload
        if isinstance(payload, dict):
            # from_dict coerces the wire shape (string direction/timestamp) back
            # into typed fields and ignores unknown keys.
            message = ChannelMessage.from_dict(payload)
        authoritative_agent_id = getattr(self.agent, "did", None)
        if not isinstance(authoritative_agent_id, str) or not authoritative_agent_id:
            logger.warning(
                "Inbound notification from %s dropped: host agent identity unavailable",
                self.name,
            )
            return None
        supplied_agent_id = getattr(message, "agent_id", "")
        if supplied_agent_id not in {"", authoritative_agent_id}:
            logger.warning(
                "Inbound notification from %s dropped: child supplied another agent scope",
                self.name,
            )
            return None
        # The host, not the isolated child, owns the tenant/agent boundary.
        # ChannelFeature repeats this binding before its own persistence so a
        # direct caller cannot bypass the proxy's authoritative scope either.
        message.agent_id = authoritative_agent_id
        # Route and sender filtering must use the host-negotiated proxy
        # identity, never the isolated child's mutable wire field.
        message.channel_type = authoritative_channel_type
        # This transient attribute is never serialized by ChannelMessage. It
        # is written after child data is parsed and scoped by the ContextVar
        # above, so only the host-validated Telegram polling path can request
        # cursor-owned durable cognition.
        setattr(
            message,
            "_kestrel_cursor_owned_protocol",
            _cursor_owned_inbound_protocol.get(),
        )
        if terminal_disposition is not None:
            return await channel.handle_terminal_inbound(
                message, disposition=terminal_disposition
            )
        return await channel.handle_inbound(message)


def _send_outcome(envelope: Dict[str, Any], transport_ok: bool):
    """Classify a send tool's envelope into a DeliveryStatus.

    Recognizes two wire shapes a channel send tool may return:
      * a plain ``{"ok": bool, ...}`` envelope, and
      * the framework ``ToolResult`` shape ``{"status": "ok"|"error"|
        "partial", ...}`` (ok->SUCCESS, error->FAILURE, partial->PENDING).
    An explicit error/non-OK envelope must NOT be reported as a successful
    delivery just because the JSON-RPC transport itself succeeded.
    """
    from kestrel_sdk.channels import DeliveryStatus

    if not transport_ok:
        return DeliveryStatus.FAILURE
    if "ok" in envelope:
        return DeliveryStatus.SUCCESS if envelope["ok"] else DeliveryStatus.FAILURE
    status = envelope.get("status")
    if isinstance(status, str):
        token = status.lower()
        if token in {"error", "failure", "failed"}:
            return DeliveryStatus.FAILURE
        if token in {"partial", "pending"}:
            return DeliveryStatus.PENDING
        if token in {"ok", "success"}:
            return DeliveryStatus.SUCCESS
    if "success" in envelope:
        return DeliveryStatus.SUCCESS if envelope["success"] else DeliveryStatus.FAILURE
    # No tool-level signal: trust the transport outcome.
    return DeliveryStatus.SUCCESS


def _delivery_receipt_from_result(channel_type: str, result: Dict[str, Any]):
    """Map a forwarded isolated-tool result onto a ``DeliveryReceipt``.

    ``call_isolated_tool`` returns the tool's ToolResult envelope TOP-LEVEL
    (unified shape #F025): ``{status, confirmation, error, data, tool, success}``.
    The envelope IS the result; a legacy raw return still arrives wrapped as
    ``{success, result}`` and is tolerated below.
    """
    import uuid as _uuid

    from kestrel_sdk.channels import DeliveryReceipt, DeliveryStatus

    result = result if isinstance(result, dict) else {}
    transport_ok = bool(result.get("success", True))
    # Flat envelope (status top-level) is the result itself; tolerate a legacy
    # nested ``result`` payload for any un-migrated raw-return service.
    envelope = result
    if result.get("status") is None and isinstance(result.get("result"), dict):
        envelope = result["result"]
    data = envelope.get("data")
    data = data if isinstance(data, dict) else {}
    receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
    message_id = str(
        envelope.get("message_id")
        or envelope.get("id")
        or data.get("message_id")
        or data.get("id")
        or receipt.get("message_id")
        or _uuid.uuid4()
    )
    status = _send_outcome(envelope, transport_ok)
    if status is DeliveryStatus.FAILURE:
        error = envelope.get("error") or result.get("error") or "send failed"
        return DeliveryReceipt(
            message_id=message_id,
            status=status,
            channel_type=channel_type,
            error=str(error),
        )
    return DeliveryReceipt(message_id=message_id, status=status, channel_type=channel_type)


class ProxyChannelAdapter(ChannelAdapter):
    """Forwarding ``ChannelAdapter`` backed by an isolated feature service.

    Registered into ``ChannelFeature.registry`` for isolated features that
    advertise a channel capability, so the generic channels API routes through
    the out-of-venv service. Sends forward to the service's ``send_tool``;
    inbound flows independently via ``channel.inbound`` events (handled by the
    proxy's event handler), so ``on_message`` is a no-op here.
    """

    def __init__(
        self,
        proxy: "ProxyFeature",
        *,
        channel_type: str,
        send_tool: str,
        status_tool: Optional[str] = None,
        config: Any = None,
    ):
        super().__init__(config)
        self._proxy = proxy
        self._channel_type = channel_type
        self._send_tool = send_tool
        self._status_tool = status_tool
        # Registered only after the service reports ready; treat as connected
        # until told otherwise. The send receipt carries the real per-send truth.
        self._connected = True

    @property
    def channel_type(self) -> str:
        return self._channel_type

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def on_message(self, callback) -> None:
        # Inbound is delivered to ChannelFeature.handle_inbound via the proxy's
        # event handler, not through an adapter-held callback.
        return None

    async def send_message(self, to: str, content: str, **kwargs):
        result = await self._proxy.call_isolated_tool(
            self._send_tool, {"to": to, "message": content}
        )
        return _delivery_receipt_from_result(self._channel_type, result)
