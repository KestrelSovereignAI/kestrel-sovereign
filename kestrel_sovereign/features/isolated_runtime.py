"""Isolated feature runtime proxy and per-agent venv provisioning.

A feature distribution opts into out-of-venv execution via its pyproject:

    [tool.kestrel.feature]
    runtime = "isolated-venv"
    service = "kestrel-whatsapp-web"   # runnable: a console-script name, or "module:func"
    project = "service"                # install target for the venv (path/dist); defaults to the distribution
    # venv  = "/abs/path/.venv"        # optional explicit venv-path override

`service` is the thing to RUN (resolved from the per-agent venv's bin/ as a
console script, or executed as a "module:func" callable). `project` is the
thing to INSTALL. They are deliberately distinct so the runnable is never
mistaken for a pip target or a `python -m` module.
"""

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import weakref
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, NoReturn, Optional
from uuid import uuid4

from kestrel_sdk.channels import ChannelAdapter
from kestrel_sdk.isolated_feature import (
    CONFIG_TRANSITION_APPLIED,
    ConfigTransitionResult,
    HostIngressCapabilities,
    HostIngressError,
    HostIngressPayload,
    HostIngressUnknownNameError,
    HostIngressUnsupportedError,
    MAX_HOST_INGRESS_PAYLOAD_BYTES,
    ProtocolError,
    validate_host_ingress_name,
    validate_host_ingress_payload,
)
from kestrel_sdk.tools.base import AgentTool, ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.base import Feature, UIContributions

logger = logging.getLogger(__name__)

# Upper bound on a single supervision health probe. A wedged child that never
# answers health() must not silently kill supervision forever (F013) — treat a
# probe that exceeds this as unhealthy and fall through to the restart path.
_HEALTH_PROBE_TIMEOUT = 5.0

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


def _venv_sdk_version(python_path: Path) -> str:
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
            [str(python_path), "-c", _CHILD_SDK_PROBE],
            check=True,
            capture_output=True,
            text=True,
            env=_isolated_child_env(venv_path),
        )
        return res.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_key(feature_name: str, suffix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", feature_name).upper()
    return f"KESTREL_FEATURE_{normalized}_{suffix}"


def _agent_data_dir(agent: Any) -> Path:
    storage_path = getattr(agent, "storage_path", None)
    if storage_path:
        return Path(storage_path).expanduser().resolve().parent
    return (Path.cwd() / "agent_data" / "default").resolve()


def _venv_bin_dir(venv_path: Path) -> Path:
    return venv_path / ("Scripts" if os.name == "nt" else "bin")


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


# Interpreter-behavior env vars that would let the HOST Python installation
# shadow the isolated venv's packages, defeating the isolation the runtime
# exists for (F023). Feature config/secrets ride through the general environment
# intentionally (KESTREL_FEATURE_* is the documented config channel), so we STRIP
# these specific interpreter vars rather than allowlisting the whole environment.
_SHADOWING_ENV_VARS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV")


def _isolated_child_env(venv_path: Optional[Path]) -> Dict[str, str]:
    """Build the launch environment for the isolated service subprocess.

    Inherits the host environment (so feature config/secrets pass through) but
    strips the interpreter-behavior vars in ``_SHADOWING_ENV_VARS`` so a stray
    host ``PYTHONPATH``/``VIRTUAL_ENV`` can't resolve the service's imports
    against host site-packages. When a venv is used, re-point ``VIRTUAL_ENV`` at
    it and prepend its bin dir to ``PATH`` so child processes bind to the
    isolated venv.
    """
    env = dict(os.environ)
    for var in _SHADOWING_ENV_VARS:
        env.pop(var, None)
    if venv_path is not None:
        env["VIRTUAL_ENV"] = str(venv_path)
        bin_dir = str(_venv_bin_dir(venv_path))
        env["PATH"] = os.pathsep.join(
            [bin_dir, env.get("PATH", "")]
        ).rstrip(os.pathsep)
    return env


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
        self.runtime = runtime
        self.name = runtime.class_name
        self._client_factory = client_factory
        self._client: Any = None
        self._tools: List[AgentTool] = []
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
        self._traffic_gate = _TrafficGate(before_reset=self._assert_child_start_allowed)
        self._fenced_recovery_failed = False
        self._venv_path: Optional[Path] = None
        self._bin_path: Optional[Path] = None
        self._host_config: Dict[str, Any] = {}
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
            # A previous enable cycle may have left an intentional empty config (or
            # a stopped client) on this same object. A fresh initialize must never
            # let that in-memory state stand in for the durable read below.
            self._host_config = {}
            self._host_config_loaded = False
            self._venv_path, self._bin_path = self.resolve_runtime_paths()
            if self._bin_path is None:
                self.ensure_venv()
            # Resolve persisted/UI host config BEFORE building the client so it can be
            # forwarded to the isolated service through the initialize handshake (the
            # service is otherwise launched bare, with only env vars).
            await self._ensure_host_config_loaded()
            self._assert_child_start_allowed()
            await self._connect_client()
            self._assert_child_start_allowed()
            # A previously quarantined instance is only made reachable after its
            # fresh child was initialized from durable config.
            await self._reset_traffic_gate_after_initialize()
            self._assert_child_start_allowed()
            self._supervision_task = self._start_supervision()

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
        try:
            await self._register_event_handler(client)
        except BaseException:
            # A client whose event registration failed must not remain
            # reachable through host tools while its caller unwinds.  Shutdown
            # may already have unpublished and retired this exact client while
            # registration was awaited; only the caller that actually removed
            # it from publication owns this additional retirement attempt.
            unpublished_client = self._unpublish_client(client)
            if unpublished_client is client:
                await self._retire_detached_client(unpublished_client)
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
        client = self._build_client(config=child_config)
        try:
            await _maybe_await(client.start())
            advertised_tools = await _maybe_await(client.list_tools())
        except BaseException:
            await self._retire_detached_client(client)
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
        if self._terminal_cleanup_uncertain or health_probe_running:
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
        caps = self._client_capabilities()
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
        async with self._reload_lock:
            if self._terminal_lifecycle_latched:
                await self._persist_terminal_config(
                    cfg,
                    preserve_secret_fields=_preserve_secret_fields,
                    validate_effective_config=_validate_effective_config,
                )
                return
            self._begin_reload()
            # This intent marker deliberately precedes the await below. A
            # cancelled close/drain has already made the gate finite-closed,
            # so its finally must perform a cancellation-safe reopen or seal.
            gate_closed = True
            self._fenced_recovery_failed = False
            transition_attempted = False
            transition_succeeded = False
            transition: Optional[_ConfigTransition] = None
            promotion: Optional[_PromotionResolution] = None
            transition_settled = False
            lifecycle_result: Optional[ConfigTransitionResult] = None
            local_authoritative = False
            try:
                # Admission must close before the candidate is staged, not just
                # before a replacement.  A successful in-process hook may have
                # adopted its candidate by the time it returns, so tools,
                # channel sends, and inbound callbacks must all be drained
                # before the hook begins.
                await self._close_traffic_gate()
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
                await self._reconcile_client_to_authoritative_config(
                    transition.active_config,
                    force=False,
                )
                if self._client is None:
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
                staged_transition = transition or getattr(
                    authority_error, "transition", None
                )
                await self._handle_config_authority_change(staged_transition)
                raise RuntimeError(
                    f"Cannot apply config for isolated feature {self.name}: "
                    "legacy config authority became visible during rolling upgrade"
                ) from authority_error
            except asyncio.CancelledError:
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
                try:
                    if self._fenced_recovery_failed:
                        await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
                finally:
                    self._end_reload()
                    # A quarantined proxy remains fail-closed. Every other path
                    # reaches this point only after promotion or owned cleanup
                    # has reconciled the active child with durable state. These
                    # operations are themselves shielded to a final condition
                    # state before this reload releases its lock.
                    if gate_closed:
                        if self._stopping:
                            await self._seal_traffic_gate()
                        else:
                            await self._reopen_traffic_gate()

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

    def _host_ingress_capabilities(self) -> HostIngressCapabilities | None:
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

        capabilities = getattr(self._client, "host_ingress_capabilities", None)
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
            self = None

    async def _run_host_ingress(
        self,
        request: _HostIngressRequest,
        outcome_slot: _HostIngressOutcomeSlot,
    ) -> None:
        """Perform an already-snapshotted ingress call inside traffic admission."""

        try:
            async with self._traffic_gate.admit():
                client = self._client
                if client is None:
                    outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_UNSUPPORTED)
                    return

                try:
                    capabilities = self._host_ingress_capabilities()
                except asyncio.CancelledError:
                    outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_CANCELLED)
                    return
                except BaseException:  # noqa: BLE001 - untrusted capability facade
                    outcome_slot.outcome = _HostIngressOutcome(
                        _HOST_INGRESS_GENERIC_FAILURE
                    )
                    return
                if capabilities is None:
                    outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_UNSUPPORTED)
                    return
                if request.name not in capabilities.names:
                    outcome_slot.outcome = _HostIngressOutcome(
                        _HOST_INGRESS_UNKNOWN_NAME
                    )
                    return

                try:
                    call = getattr(client, "call_host_ingress", None)
                except asyncio.CancelledError:
                    outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_CANCELLED)
                    return
                except BaseException:  # noqa: BLE001 - untrusted descriptor boundary
                    outcome_slot.outcome = _HostIngressOutcome(
                        _HOST_INGRESS_GENERIC_FAILURE
                    )
                    return
                if not callable(call):
                    outcome_slot.outcome = _HostIngressOutcome(_HOST_INGRESS_UNSUPPORTED)
                    return
                await self._call_host_ingress_rpc(call, request, outcome_slot)
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

        cfg = self._host_config if config is None else config
        cfg = cfg if isinstance(cfg, dict) else {}
        return ChannelConfig(
            channel_type=channel_type,
            agent_id=str(cfg.get("agent_id", "") or ""),
            enabled=bool(cfg.get("enabled", True)),
            allowed_senders=list(cfg.get("allowed_senders") or []),
        )

    async def call_isolated_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        context = _scheduled_tool_execution_context()
        # The gate keeps a selected client alive through the complete RPC.  It
        # is shared (not a reload mutex), so unrelated calls remain concurrent
        # whenever no config transition is active.
        try:
            async with self._traffic_gate.admit():
                # This preflight belongs *inside* admission. Otherwise a
                # terminal shutdown with no published client would leak a
                # scheduler-specific error instead of the stable fail-closed
                # result used by every other new tool/channel call.
                if context is not None and not self._supports_tool_execution_context(context):
                    raise SchedulerExecutionContextUnavailable(
                        "scheduled isolated tool calls require a service that advertises "
                        "ToolExecutionContext support"
                    )
                return await self._call_isolated_tool_admitted(name, args, context)
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
        bin_override = os.environ.get(_env_key(self.name, "BIN"))
        if bin_override:
            return self._default_venv_path(), Path(bin_override).expanduser().resolve()

        venv_override = os.environ.get(_env_key(self.name, "VENV"))
        if venv_override:
            return Path(venv_override).expanduser().resolve(), None

        if self.runtime.venv:
            return Path(self.runtime.venv).expanduser().resolve(), None

        return self._default_venv_path(), None

    def _venv_is_overridden(self) -> bool:
        """True when the venv path was supplied by the operator (KESTREL_FEATURE_
        <NAME>_VENV env or the pyproject ``venv =``) rather than provisioned by
        the host at the default path. An operator-supplied venv is NOT ours to
        mutate — see ensure_venv."""
        return bool(
            os.environ.get(_env_key(self.name, "VENV"))
            or self.runtime.venv
        )

    def _default_venv_path(self) -> Path:
        return _agent_data_dir(self.agent) / "feature_venvs" / self.name / ".venv"

    def _provision_manifest_path(self) -> Path:
        # Inside the venv dir, not its parent: explicit venv overrides can share
        # a parent directory, and a parent-scoped manifest would let sibling
        # features clobber each other's stamp and reinstall on every startup.
        assert self._venv_path is not None
        return self._venv_path / ".kestrel_provision.json"

    def _read_provision_manifest(self) -> Dict[str, Any]:
        try:
            return json.loads(self._provision_manifest_path().read_text())
        except Exception:  # noqa: BLE001 — missing/corrupt manifest ⇒ reprovision
            return {}

    def _write_provision_manifest(
        self, install_target: str, host_sdk: str, child_sdk: str
    ) -> None:
        path = self._provision_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "install_target": install_target,
                    # The host SDK we provisioned AGAINST — staleness keys on a
                    # change here, so a genuinely SDK-pinned feature reinstalls
                    # once per host bump, not on every startup.
                    "provisioned_against_host_sdk": host_sdk,
                    # The SDK version that actually landed in the venv (may lag
                    # host_sdk if the feature pins it); recorded for diagnosis.
                    "child_sdk_version": child_sdk,
                },
                indent=2,
            )
        )

    def _provision_is_stale(self, install_target: str) -> bool:
        """A provisioned venv is stale if the install target changed or the host
        has upgraded kestrel-sdk since we last provisioned against it (F019: a
        stale wire contract — e.g. pre-0.28 serial dispatch — must not silently
        survive a host update)."""
        manifest = self._read_provision_manifest()
        if manifest.get("install_target") != install_target:
            return True
        if manifest.get("provisioned_against_host_sdk") != _host_sdk_version():
            return True
        return False

    def ensure_venv(self) -> None:
        assert self._venv_path is not None
        python_path = _venv_python(self._venv_path)

        # Install the PROJECT (path/dist), never the `service` runnable — the
        # latter is a console-script name or "module:func", not a pip target.
        install_target = self.runtime.project or self.runtime.distribution
        if not install_target:
            raise RuntimeError(
                f"Isolated feature {self.name} has no project/distribution to install"
            )

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
            self._warn_on_sdk_mismatch(python_path)
            return

        # An operator-supplied (override) venv that already exists is NOT ours to
        # mutate: running `uv pip install --upgrade` into it would rewrite a
        # prebuilt/pinned environment the operator deliberately provided (and
        # hard-fail the whole feature at startup if the index is unreachable).
        # Verify SDK compatibility and warn on a mismatch (See Something Say
        # Something), but leave the venv untouched and do not stamp a manifest we
        # don't own. Host-owned default venvs (and a not-yet-created override
        # path we bootstrap below) keep the full reprovision lifecycle.
        if not exists:
            self._run(["uv", "venv", str(self._venv_path)])
        elif not self._provision_is_stale(install_target):
            return

        # Fresh venv, changed install target, or host SDK upgraded since the
        # venv was provisioned. On an existing venv, upgrade in place so a stale
        # kestrel-sdk is replaced; then stamp the manifest so the next startup
        # can tell whether another reprovision is due.
        cmd = ["uv", "pip", "install", "--python", str(python_path)]
        if exists:
            cmd.append("--upgrade")
        cmd.append(install_target)
        self._run(cmd)

        # Verify what actually landed: a feature that pins an older SDK can
        # install "successfully" while keeping the stale wire contract. Surface
        # that rather than silently stamping the venv as fresh (See Something
        # Say Something) — staleness still keys on the host transition so we
        # don't thrash reinstalling a genuinely pinned feature every startup.
        host_sdk = _host_sdk_version()
        child_sdk = _venv_sdk_version(python_path)
        self._warn_on_sdk_mismatch(python_path, host_sdk=host_sdk, child_sdk=child_sdk)
        self._write_provision_manifest(install_target, host_sdk, child_sdk)

    def _warn_on_sdk_mismatch(
        self, python_path: Path, *, host_sdk: str = None, child_sdk: str = None
    ) -> None:
        host_sdk = host_sdk if host_sdk is not None else _host_sdk_version()
        child_sdk = child_sdk if child_sdk is not None else _venv_sdk_version(python_path)
        if child_sdk != host_sdk and "unknown" not in (child_sdk, host_sdk):
            logger.warning(
                "Isolated feature %s venv resolved kestrel-sdk %s but host is %s — "
                "the feature may pin an incompatible wire contract",
                self.name,
                child_sdk,
                host_sdk,
            )

    def _run(self, cmd: List[str]) -> None:
        if shutil.which(cmd[0]) is None:
            raise RuntimeError(f"Required executable not found: {cmd[0]}")
        subprocess.run(cmd, check=True)

    def _build_client(self, config: Optional[Dict[str, Any]] = None) -> Any:
        factory = self._client_factory
        if factory is None:
            from kestrel_sdk.isolated_feature import SubprocessIsolatedFeatureClient

            factory = SubprocessIsolatedFeatureClient

        child_config = self._host_config if config is None else config

        kwargs = {
            "feature_name": self.name,
            "service": self.runtime.service,
            "venv_path": str(self._venv_path) if self._venv_path else None,
            "python": str(_venv_python(self._venv_path)) if self._venv_path else None,
            "executable": str(self._bin_path) if self._bin_path else None,
            "event_handler": self._handle_event,
            "notification_handler": self._handle_event,
            # An empty object is an explicit effective config: the SDK sends
            # ``config`` only when this value is not ``None``, and its service
            # then calls ``configure({})``. Do not collapse it into a missing
            # config field.
            "config": child_config,
            # Launch env with interpreter-shadowing vars stripped (F023) so the
            # host PYTHONPATH/VIRTUAL_ENV can't defeat the venv isolation.
            "env": _isolated_child_env(self._venv_path),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        try:
            signature = inspect.signature(factory)
            params = signature.parameters
        except (TypeError, ValueError):
            params = {}

        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return factory(**kwargs)

        accepted = {key: value for key, value in kwargs.items() if key in params}

        # Keyword-only factory (named params, no positional `command`): deliver the
        # accepted keyword args directly (config/event handlers/etc.).
        if "command" not in params:
            if accepted:
                return factory(**accepted)
            return factory()

        # Positional-command constructor (SubprocessIsolatedFeatureClient): pass the
        # launch argv plus whatever keyword extras the factory accepts (notably
        # `config`, so host config reaches the service via the initialize handshake).
        accepted.pop("command", None)
        try:
            return factory(self._service_command(), **accepted)
        except (TypeError, ValueError):
            return factory(self._service_command())

    def _service_command(self) -> List[str]:
        """Build the argv to launch the isolated service.

        Resolution order:
          1. explicit BIN override (``self._bin_path``);
          2. ``service`` of the form ``module:func`` -> ``<venv-python> -c ...``;
          3. ``service`` as a console-script name -> ``<venv>/bin/<script>``.
        The ``service`` runnable is NEVER treated as a ``python -m`` module
        (it may be a path/dist), which is what previously broke startup.
        """
        if self._bin_path is not None:
            return [str(self._bin_path)]

        service = self.runtime.service
        if not service:
            raise RuntimeError(
                f"Isolated feature {self.name} has no `service` runnable configured"
            )

        if ":" in service:  # module:func callable
            module, _, func = service.partition(":")
            python = (
                str(_venv_python(self._venv_path)) if self._venv_path else "python"
            )
            return [python, "-c", f"from {module} import {func}; {func}()"]

        # console-script installed into the venv's bin/Scripts dir
        if self._venv_path is not None:
            return [str(_venv_bin_dir(self._venv_path) / service)]
        return [service]

    async def _register_event_handler(self, client: Any = None) -> None:
        """Attach the host event handler to a published client.

        Accepting the client explicitly lets fenced recovery keep a started
        candidate detached until durable promotion has completed.
        """

        target = self._client if client is None else client
        register = (
            getattr(target, "set_event_handler", None)
            or getattr(target, "add_event_handler", None)
            or getattr(target, "subscribe", None)
        )
        if register is not None:
            await _maybe_await(register(self._handle_event))
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
            await _maybe_await(on_event(self._handle_event))

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

    async def _handle_event(self, event: Any) -> None:
        # SDK event callbacks are externally visible traffic too: an inbound
        # channel message can wake the agent and trigger effects.  Keep it on
        # the same gate as tools so a candidate hook cannot route an event
        # before its config becomes durable. Events are deliberately *dropped*
        # rather than queued during a finite close: they originated from the
        # old child and replaying them after promotion could apply stale input
        # under a new configuration. Terminal drops are also silent because an
        # SDK callback has no caller to handle a deliberate shutdown result.
        try:
            async with self._traffic_gate.admit(wait_for_open=False):
                await self._handle_event_admitted(event)
        except (_TrafficGateClosedError, _TrafficGateTerminalError):
            return

    async def _handle_event_admitted(self, event: Any) -> None:
        kind = _meta_get(event, "type") or _meta_get(event, "event") or _meta_get(event, "kind")
        payload = _meta_get(event, "payload", event)
        if kind == "feature/event":
            event_name = _meta_get(payload, "name") or _meta_get(payload, "event")
            data = _meta_get(payload, "data", payload)
        else:
            event_name = kind
            data = payload

        if event_name in {"channel.inbound", "inbound", "message.inbound"}:
            await self._route_inbound(data)
        elif event_name in {"channel.link_qr", "link_qr", "channel.qr"}:
            await self._route_link_qr(data)
        elif event_name in {"channel.link_cleared", "link_cleared"}:
            await self._route_link_cleared(data)

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
            _agent_data_dir(self.agent)
            / "channel_link_artifacts"
            / f"{channel_type}_link_qr.png"
        )
        try:
            png.unlink(missing_ok=True)
        except OSError:
            pass

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
        import base64

        channel_type = str(payload.get("channel_type") or "").strip().lower()
        png_b64 = payload.get("png_b64") or payload.get("png")
        if (
            not channel_type
            or not re.fullmatch(r"[a-z0-9_]{1,32}", channel_type)
            or not png_b64
        ):
            logger.warning("Dropping malformed channel.link_qr from %s", self.name)
            return
        try:
            png_bytes = base64.b64decode(png_b64)
        except Exception as exc:  # noqa: BLE001
            logger.warning("channel.link_qr from %s had undecodable PNG: %s", self.name, exc)
            return

        out = (
            _agent_data_dir(self.agent)
            / "channel_link_artifacts"
            / f"{channel_type}_link_qr.png"
        )
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(png_bytes)
        except OSError as exc:
            logger.warning("Failed to persist channel.link_qr PNG for %s: %s", self.name, exc)
            return

    async def _route_inbound(self, payload: Any) -> None:
        channel = self._channel_feature()
        if channel is None or not hasattr(channel, "handle_inbound"):
            logger.warning(
                "Inbound notification from %s dropped: ChannelFeature unavailable",
                self.name,
            )
            return

        from kestrel_sovereign.features.channels.models import ChannelMessage

        message = payload
        if isinstance(payload, dict):
            # from_dict coerces the wire shape (string direction/timestamp) back
            # into typed fields and ignores unknown keys.
            message = ChannelMessage.from_dict(payload)
        await channel.handle_inbound(message)


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
