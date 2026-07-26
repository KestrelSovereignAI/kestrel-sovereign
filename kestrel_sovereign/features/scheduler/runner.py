"""Durable scheduler execution with claims, leases, and recovery.

``scheduled_tasks`` owns the next occurrence while ``task_execution_log`` owns
the stable execution identity for that occurrence.  A runner claims before it
dispatches, renews the claim while dispatch is in flight, and finalizes with a
compare-and-set.  This makes concurrent runners safe against one another; a
process death leaves a lease which another runner may recover after expiry.
"""

import asyncio
import contextvars
import hashlib
import hmac
import inspect
import json
import logging
import os
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Collection
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Protocol, Union

from kestrel_sovereign._async_ownership import await_owned_task, run_blocking_operation
from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.features.scheduler.cron import CronParseError, next_run
from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome

logger = logging.getLogger(__name__)


def _scheduler_database_backend_type(db: Any) -> str:
    """Return a concrete scheduler DB type without trusting loose doubles."""

    backend_type = getattr(db, "backend_type", "")
    return backend_type.lower() if isinstance(backend_type, str) else ""


def scheduler_database_now_sql(db: Any) -> str:
    """Return the scheduler's portable statement-time UTC clock expression."""

    backend_type = _scheduler_database_backend_type(db)
    if backend_type == "postgres":
        return (
            "(to_char(clock_timestamp() AT TIME ZONE 'UTC', "
            "'YYYY-MM-DD\"T\"HH24:MI:SS.US') || '+00:00')"
        )
    if backend_type == "sqlite":
        return "strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')"
    raise RuntimeError("scheduler database clock is unavailable for this backend")


async def scheduler_database_clock(db: Any) -> datetime:
    """Read the scheduler's wall clock from its durable database.

    Cron progression must use the same authority as the due-work predicates.
    Concrete PostgreSQL and SQLite backends therefore read their statement-time
    clocks; deliberately minimal test/adaptor doubles retain the historical
    host-clock fallback.
    """

    backend_type = _scheduler_database_backend_type(db)
    if backend_type == "postgres":
        value = await db.fetchval("SELECT clock_timestamp()")
    elif backend_type == "sqlite":
        value = await db.fetchval(
            "SELECT strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')"
        )
    else:
        return datetime.now(timezone.utc)

    if isinstance(value, datetime):
        return (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None
            else value.astimezone(timezone.utc)
        )
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        raise RuntimeError("scheduler database returned an invalid wall-clock timestamp")
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


POLL_INTERVAL = 30
DEFAULT_MISFIRE_GRACE_SECONDS = 600
DEFAULT_MAX_CONCURRENT_TASKS = 4
DEFAULT_LEASE_SECONDS = 120

MISFIRE_SKIP = "skip"
MISFIRE_FIRE_ONCE = "fire_once"
MISFIRE_CATCH_UP = "catch_up"
MISFIRE_POLICIES = frozenset({MISFIRE_SKIP, MISFIRE_FIRE_ONCE, MISFIRE_CATCH_UP})

SCHEDULE_CRON = "cron"
SCHEDULE_ONE_SHOT = "one_shot"

# The SDK's ToolExecutionContext limits idempotency keys to 512 UTF-8 bytes.
# An occurrence key appends ``:`` plus a SHA-256 hex digest to the persisted
# schedule base, so validate the base before the schedule ever becomes due.
MAX_EXECUTION_IDEMPOTENCY_KEY_BYTES = 512
OCCURRENCE_IDEMPOTENCY_SUFFIX_BYTES = 1 + hashlib.sha256().digest_size * 2
MAX_SCHEDULE_IDEMPOTENCY_BASE_BYTES = (
    MAX_EXECUTION_IDEMPOTENCY_KEY_BYTES - OCCURRENCE_IDEMPOTENCY_SUFFIX_BYTES
)

SCHEDULER_PROTOCOL_VERSION = 2
SCHEDULER_ROLLOUT_STATE_ACTIVE = "active"
SCHEDULER_ROLLOUT_STATE_QUIESCING = "quiescing"
SCHEDULER_ROLLOUT_ACK_ENV = "KESTREL_SCHEDULER_ROLLOUT_ACK"

# ``scheduled_tasks`` is shared by every DID in a PostgreSQL fleet.  A table
# existing is therefore not, by itself, evidence that a particular DID is an
# upgrade from the pre-v2 runner: the first v2 DID creates it and a second
# freshly-configured DID must not be spuriously forced through a legacy drain.
# Keep that distinction durable and database-global.
SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2 = "fresh-v2"
SCHEDULER_SCHEMA_PROVENANCE_LEGACY_UNKNOWN = "legacy-unknown"

# A row that was already due when it was hidden from an origin/main selector
# may have been dispatched by that selector just before the fence.  There is
# no safe way to infer that its external effect did not happen, so activation
# deliberately leaves it paused and visible for an explicit operator resume.
ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE = "rollout_ambiguous_legacy_occurrence"

# Distinct from every real DID and used only as the key for the file-backed
# SQLite bootstrap gate. It serializes provenance discovery, additive DDL, and
# rollout seeding across independently opened connections to the same file.
# A NUL cannot occur in a valid DID, so this cannot collide with a per-DID
# rollout gate while still producing a stable hash-based SQLite lock filename.
_SCHEDULER_BOOTSTRAP_LOCK_SCOPE = "\0scheduler-bootstrap"
_SCHEDULER_BOOTSTRAP_ADVISORY_LOCK_SCOPE = 0
# PostgreSQL long-running effects cannot keep a transaction-scoped control-row
# lock without deadlocking a scheduled tool that writes scheduler state through
# another pooled connection.  A distinct per-DID session advisory lock carries
# the *effect versus rollout-transition* exclusion instead.  It deliberately
# shares the issue namespace with the bootstrap lock, with the DID-derived
# signed-32-bit key remapped away from the reserved global key 0.
_SCHEDULER_EFFECT_ADVISORY_LOCK_NAMESPACE = 2715


# SQLite keeps one writer transaction per AsyncDatabase connection.  Holding
# that transaction while an executor runs would prevent the separately-owned
# lease-renewal task (and target tools that persist state) from making any
# progress. A per-DID advisory file lock instead provides the shared/exclusive
# external-effect boundary between v2 dispatch admission and a v2 rollout
# fence. Advisory locks are released by the OS on a process death, unlike a
# create-with-O_EXCL sentinel, so a crashed runner cannot permanently wedge a
# local deployment. Windows uses ``LockFileEx`` for the corresponding
# shared/exclusive byte-range lock.
class _SQLiteRolloutFileLock:
    """Cancellation-safe, writer-preferring advisory lock for one SQLite DID.

    The main lock is shared by admitted effects and exclusive for a rollout
    transition.  A short turnstile lock closes reader admission once a writer
    queues, so a continual stream of due schedules cannot starve a fence.
    Both locks are released by the operating system when a process dies.
    """

    def __init__(self, path: str, *, shared: bool):
        self._path = path
        self._shared = shared
        self._main_fd: Optional[int] = None
        self._main_token: Any = None
        self._turnstile_fd: Optional[int] = None
        self._turnstile_token: Any = None

    @staticmethod
    def _lock_fd(fd: int, *, shared: bool) -> Any:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import ctypes
            import msvcrt
            from ctypes import wintypes

            class _Overlapped(ctypes.Structure):
                _fields_ = [
                    ("Internal", ctypes.c_size_t),
                    ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD),
                    ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE),
                ]

            overlapped = _Overlapped()
            flags = 0 if shared else 0x00000002  # LOCKFILE_EXCLUSIVE_LOCK
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel32.LockFileEx(
                wintypes.HANDLE(msvcrt.get_osfhandle(fd)),
                flags,
                0,
                1,
                0,
                ctypes.byref(overlapped),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return overlapped

        import fcntl

        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        return None

    @staticmethod
    def _unlock_fd(fd: int, token: Any) -> None:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import ctypes
            import msvcrt
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel32.UnlockFileEx(
                wintypes.HANDLE(msvcrt.get_osfhandle(fd)),
                0,
                1,
                0,
                ctypes.byref(token),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return

        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)

    def _acquire_path(self, path: str, *, shared: bool) -> tuple[int, Any]:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            return fd, self._lock_fd(fd, shared=shared)
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _release_path(fd: int, token: Any) -> None:
        try:
            _SQLiteRolloutFileLock._unlock_fd(fd, token)
        finally:
            os.close(fd)

    def acquire(self) -> None:
        # A reader crosses the shared turnstile only while taking the shared
        # main lock. A writer retains the exclusive turnstile through its
        # critical section, closing new reader admission before it drains the
        # effects that were already admitted.
        turnstile_path = f"{self._path}.turnstile"
        try:
            (
                self._turnstile_fd,
                self._turnstile_token,
            ) = self._acquire_path(turnstile_path, shared=self._shared)
            self._main_fd, self._main_token = self._acquire_path(
                self._path, shared=self._shared
            )
            if self._shared:
                self._release_turnstile()
        except BaseException:
            self.release()
            raise

    def _release_turnstile(self) -> None:
        fd, token = self._turnstile_fd, self._turnstile_token
        self._turnstile_fd = None
        self._turnstile_token = None
        if fd is not None:
            self._release_path(fd, token)

    def release(self) -> None:
        fd, token = self._main_fd, self._main_token
        self._main_fd = None
        self._main_token = None
        try:
            if fd is not None:
                self._release_path(fd, token)
        finally:
            self._release_turnstile()


_sqlite_memory_rollout_locks: Dict[tuple[int, str], AsyncReaderWriterLock] = {}


class SchedulerRolloutQuiescenceRequired(RuntimeError):
    """Raised when a legacy scheduler must be drained before v2 can run.

    A legacy runner has no claim CAS and selects a due row from only
    ``agent_id``, ``enabled``, and ``next_run_at``.  A claim-only migration
    therefore cannot protect an occurrence the legacy process already read.
    The durable rollout state makes the upgrade an explicit quiesce/acknowledge
    transition instead of pretending arbitrary mixed-version overlap is safe.
    """


class SchedulerProtocolVersionIncompatible(RuntimeError):
    """Raised when durable scheduler state belongs to a newer protocol.

    The message is deliberately independent of stored version numbers, DIDs,
    and activation nonces. A host readiness endpoint may safely surface the
    error's type without disclosing another scheduler deployment's state.
    """

    def __init__(self) -> None:
        super().__init__(
            "Scheduler protocol is newer than this runner; upgrade the "
            "scheduler binary before starting."
        )


async def adopt_scheduler_registration_ownership(
    db: Any,
    *,
    task_id: str,
    agent_id: str,
    observed_registration_nonce: Optional[str],
    pending_registration_nonce: Optional[str],
) -> bool:
    """Clear a foreign pending-registration marker from one locked schedule.

    Dynamic tenant registration rollback may delete rows carrying its private
    nonce.  Once a different host claims or mutates such a row, that row has
    become shared durable state and must no longer be attributable to the
    original registration.  The caller supplies the nonce read while holding
    the schedule-row/transaction lock; the exact-nonce predicate makes this a
    compare-and-clear operation rather than accidentally adopting a freshly
    replaced row.  A retry by the same pending registration deliberately
    keeps its marker.
    """

    if (
        not observed_registration_nonce
        or observed_registration_nonce == pending_registration_nonce
    ):
        return True
    updated = await db.execute(
        """
        UPDATE scheduled_tasks
        SET scheduler_registration_nonce = NULL
        WHERE id = ? AND agent_id = ? AND scheduler_registration_nonce = ?
        """,
        (task_id, agent_id, observed_registration_nonce),
    )
    if isinstance(updated, bool):
        return updated
    if isinstance(updated, int):
        return updated > 0
    rowcount = getattr(updated, "rowcount", None)
    return not isinstance(rowcount, int) or rowcount > 0


class SchedulerFeatureUnavailable(RuntimeError):
    """A hosted agent loaded without a live SchedulerFeature dispatcher.

    This is not an execution failure: a feature may be intentionally excluded
    from a cold tenant or be soft-disabled while its persisted schedules are
    retained.  The runner therefore leaves the durable claim recoverable
    instead of consuming the occurrence as a failed task.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__(
            f"woken agent {agent_id!r} has no enabled SchedulerFeature dispatcher"
        )


def validate_schedule_idempotency_base(base: str) -> Optional[str]:
    """Return an invariant error when ``base`` cannot form an SDK-safe key."""

    if not isinstance(base, str):
        return "must be text"
    try:
        byte_length = len(base.encode("utf-8"))
    except UnicodeEncodeError:
        return "must be valid UTF-8"
    if byte_length > MAX_SCHEDULE_IDEMPOTENCY_BASE_BYTES:
        return (
            f"is {byte_length} UTF-8 bytes; at most "
            f"{MAX_SCHEDULE_IDEMPOTENCY_BASE_BYTES} bytes are allowed"
        )
    return None


@dataclass(frozen=True)
class SchedulerExecution:
    """Identity and delivery context for one durable scheduled occurrence.

    ``idempotency_key`` is stable across lease recovery of this occurrence.
    Target tools can call :func:`get_current_scheduler_execution` while a
    scheduler dispatch is active instead of receiving undocumented arguments.
    The identity is revoked as soon as that dispatch returns, including from
    child tasks that inherited the parent task's context.
    """

    id: str
    schedule_id: str
    agent_id: str
    task_name: str
    args: dict
    scheduled_for: str
    idempotency_key: str
    attempt: int
    owner: str


@dataclass(frozen=True)
class SchedulerTenantProtocolRegistration:
    """Durable ownership evidence for one dynamic host-tenant registration."""

    agent_id: str
    rollout_preexisting: bool
    registration_nonce: str


@dataclass
class _SchedulerExecutionScope:
    """A revocable scheduler identity shared with copied task contexts.

    ``asyncio.create_task`` and ``asyncio.to_thread`` copy ``ContextVar``
    values.  Storing a raw :class:`SchedulerExecution` would therefore leave
    a trusted occurrence identity usable by a detached child after the runner
    finalized its claim.  A mutable scope is deliberately shared by those
    copied contexts so revoking it closes every copy at once.
    """

    execution: SchedulerExecution
    active: bool = True

    def revoke(self) -> None:
        """Make this occurrence unavailable from every copied context."""

        self.active = False


@dataclass
class _LeaseRenewalState:
    """Shared fail-closed state for an owned claim-renewal task.

    A renewal worker returning is never a benign completion: it either lost
    its exact token/authority or failed to keep it live.  The event lets the
    preparation and effect paths observe that loss immediately instead of
    letting a completed executor attempt a stale terminal compare-and-set.
    """

    lost: asyncio.Event = field(default_factory=asyncio.Event)
    error: Optional[BaseException] = None

    def mark_lost(self, error: Optional[BaseException] = None) -> None:
        if self.error is None and error is not None:
            self.error = error
        self.lost.set()


_current_execution: contextvars.ContextVar[Optional[_SchedulerExecutionScope]] = (
    contextvars.ContextVar("scheduler_execution", default=None)
)


def get_current_scheduler_execution() -> Optional[SchedulerExecution]:
    """Return the active execution identity, or ``None`` outside a schedule.

    Tools that cause externally-visible effects should use
    ``execution.idempotency_key`` at their effect boundary.  The value is not
    added to a tool's normal arguments, which keeps existing tool signatures
    compatible and prevents callers from forging scheduler metadata.
    """

    scope = _current_execution.get()
    if scope is None or not scope.active:
        return None
    return scope.execution


@dataclass
class ScheduledTask:
    """In-memory representation of a durable scheduled task row."""

    id: str
    agent_id: str
    task_name: str
    cron_expression: str
    args_json: str
    enabled: bool
    last_run_at: Optional[str]
    next_run_at: Optional[str]
    created_at: str
    schedule_kind: str = SCHEDULE_CRON
    run_at: Optional[str] = None
    timezone_name: str = "UTC"
    misfire_policy: str = MISFIRE_SKIP
    misfire_grace_seconds: Optional[int] = None
    idempotency_key: Optional[str] = None
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[str] = None
    claim_token: Optional[str] = None
    claim_execution_id: Optional[str] = None
    claim_scheduled_for: Optional[str] = None
    attempt_count: int = 0
    terminal_status: Optional[str] = None
    terminal_at: Optional[str] = None
    scheduler_protocol_version: int = SCHEDULER_PROTOCOL_VERSION
    scheduler_rollout_fenced: bool = False
    scheduler_claim_fenced: bool = False
    scheduler_rollout_fenced_at: Optional[str] = None

    @property
    def args(self) -> dict:
        if not self.args_json:
            return {}
        try:
            parsed = json.loads(self.args_json)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @classmethod
    def from_row(cls, row: tuple) -> "ScheduledTask":
        """Decode current rows and legacy nine-column test/DB rows."""
        values = list(row)
        legacy_defaults = [
            SCHEDULE_CRON, None, "UTC", MISFIRE_SKIP, None, None, None,
            None, None, None, None, 0, None, None,
            SCHEDULER_PROTOCOL_VERSION, 0, 0, None,
        ]
        values.extend(legacy_defaults[len(values) - 9 :])
        return cls(
            id=values[0], agent_id=values[1], task_name=values[2],
            cron_expression=values[3] or "", args_json=values[4] or "{}",
            enabled=bool(values[5]), last_run_at=values[6],
            next_run_at=values[7], created_at=values[8],
            schedule_kind=values[9] or SCHEDULE_CRON, run_at=values[10],
            timezone_name=values[11] or "UTC",
            misfire_policy=values[12] or MISFIRE_SKIP,
            misfire_grace_seconds=values[13], idempotency_key=values[14],
            lease_owner=values[15], lease_expires_at=values[16],
            claim_token=values[17], claim_execution_id=values[18],
            claim_scheduled_for=values[19], attempt_count=int(values[20] or 0),
            terminal_status=values[21], terminal_at=values[22],
            scheduler_protocol_version=int(
                values[23] or SCHEDULER_PROTOCOL_VERSION
            ),
            scheduler_rollout_fenced=bool(values[24]),
            scheduler_claim_fenced=bool(values[25]),
            scheduler_rollout_fenced_at=values[26],
        )


@dataclass
class ExecutionRecord:
    """Result of one terminal task execution."""

    id: str
    task_id: str
    agent_id: str
    status: str
    result_text: Optional[str]
    duration_ms: int
    executed_at: str
    idempotency_key: Optional[str] = None
    occurrence_at: Optional[str] = None
    attempt_count: int = 1


TaskExecutor = Callable[[str, dict], Coroutine[Any, Any, Any]]
PreparedScheduledDispatch = Callable[[], Awaitable[Any]]


class HostedExecutionExecutor(Protocol):
    """Executor contract for a host that runs schedules for cold agents."""

    async def execute_scheduled(self, execution: SchedulerExecution) -> Any:
        """Resolve/wake the target and dispatch ``execution``."""


class PreparedExecutionExecutor(Protocol):
    """Optional structural protocol for a prepared scheduler dispatch.

    ``prepare_scheduled`` performs work which must not happen while the
    runner holds its database admission transaction (for example a cold agent
    load whose feature initialization can migrate scheduler tables).  It
    yields a no-argument dispatcher and owns any lifecycle lock until that
    dispatcher has returned.  The runner intentionally discovers this by
    structure so third-party hosted executors do not need to inherit a private
    base class.
    """

    def prepare_scheduled(
        self, execution: SchedulerExecution
    ) -> Any:
        """Return an async context yielding a prepared dispatch callable."""


def _loaded_agent_did(agent: Any) -> Optional[str]:
    """Return a concrete loaded-agent DID without trusting dynamic proxies."""

    for attribute in ("did", "agent_id"):
        value = getattr(agent, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


class HostedSchedulerExecutor:
    """Adapter for a host-owned scheduler loop.

    ``resolve_agent`` is intentionally host-specific: it may return a loaded
    agent or initialize a cold one.  Keeping that responsibility outside the
    scheduler means a deployment can use a process manager, a serverless wake,
    or a test double without copying the claim/lease implementation.
    """

    is_scheduler_host_executor = True

    def __init__(self, resolve_agent: Callable[[str], Awaitable[Any]]):
        self._resolve_agent = resolve_agent

    @asynccontextmanager
    async def prepare_scheduled(
        self, execution: SchedulerExecution
    ) -> AsyncIterator[PreparedScheduledDispatch]:
        """Resolve once before admission and yield a dispatch-only closure."""

        agent = await self._resolve_agent(execution.agent_id)
        self._require_enabled_scheduler_feature(agent, execution.agent_id)

        async def dispatch() -> Any:
            return await self._dispatch_resolved_agent(agent, execution)

        yield dispatch

    async def execute_scheduled(self, execution: SchedulerExecution) -> Any:
        # Retain the public one-shot hosted-executor contract for callers that
        # do not run through ``SchedulerRunner``. The runner itself enters the
        # preparation context before its database admission boundary.
        async with self.prepare_scheduled(execution) as dispatch:
            return await dispatch()

    @staticmethod
    async def _dispatch_resolved_agent(
        agent: Any, execution: SchedulerExecution
    ) -> Any:
        """Dispatch only after a host has resolved the claimed DID."""

        feature = HostedSchedulerExecutor._enabled_scheduler_feature(agent)
        if feature is not None:
            dispatch = getattr(feature, "_dispatch_scheduled_task", None)
            if callable(dispatch):
                return await dispatch(execution.task_name, execution.args)
        raise SchedulerFeatureUnavailable(execution.agent_id)

    @staticmethod
    def _require_enabled_scheduler_feature(agent: Any, agent_id: str) -> Any:
        """Require a live scheduler feature before admitting a hosted effect."""

        feature = HostedSchedulerExecutor._enabled_scheduler_feature(agent)
        if feature is None:
            raise SchedulerFeatureUnavailable(agent_id)
        return feature

    @staticmethod
    def _enabled_scheduler_feature(agent: Any) -> Optional[Any]:
        """Return this agent's live, enabled SchedulerFeature dispatcher.

        Shared-PG hosts retain an agent object after a feature is soft-disabled.
        Looking only for a private dispatcher method would therefore let a
        disabled scheduler continue to execute persisted custom-tool rows.
        The canonical feature-map key is retained for lightweight host test
        doubles; the class-name check covers normal feature registry wiring.
        """

        features = getattr(agent, "features", {}) or {}
        if not isinstance(features, dict):
            return None
        for name, feature in features.items():
            if name != "SchedulerFeature" and type(feature).__name__ != "SchedulerFeature":
                continue
            if not getattr(feature, "enabled", True):
                return None
            if callable(getattr(feature, "_dispatch_scheduled_task", None)):
                return feature
            return None
        return None


class AgentManagerHostedSchedulerExecutor(HostedSchedulerExecutor):
    """Hosted adapter that loads an unloaded agent through ``AgentManager``.

    Concrete ``AgentManager`` hosts resolve every dispatch from its live
    authority registry. ``agent_configs`` remains only as a compatibility
    fallback for simpler adapters that do not expose that registry. Per-agent
    locks ensure two due schedules cannot cold-start the same target
    concurrently.
    """

    def __init__(
        self,
        agent_manager: Any,
        agent_configs: Optional[Dict[str, tuple[str, Any]]] = None,
    ):
        self._agent_manager = agent_manager
        self._agent_configs = dict(agent_configs or {})
        self._locks: Dict[str, asyncio.Lock] = {}
        super().__init__(self._resolve_or_wake)

    async def _live_config_for(self, agent_id: str) -> Optional[tuple[str, Any]]:
        """Resolve current manager authority, retaining snapshot bootstrap only."""

        authority = getattr(self._agent_manager, "scheduler_authority_for", None)
        if callable(authority):
            config = authority(agent_id)
            if inspect.isawaitable(config):
                config = await config
            return config
        return self._agent_configs.get(agent_id)

    async def scheduler_dispatch_enabled(self, agent_id: str) -> bool:
        """Whether the host can safely claim work for this tenant.

        A warm agent is authoritative for soft-disabled state. For a cold
        tenant, an explicit per-agent feature allowlist can reject an excluded
        SchedulerFeature without waking it or publishing a claim. A ``None``
        allowlist is deliberately admitted: global/runtime disable state is
        only knowable after the cold load, where ``prepare_scheduled`` performs
        the matching check and leaves the durable claim recoverable.
        """

        agent = self._loaded_agent_for(agent_id)
        if agent is not None:
            return self._enabled_scheduler_feature(agent) is not None
        config_entry = await self._live_config_for(agent_id)
        # AgentManager returns ``(name, LocalAgentConfig)`` while lightweight
        # compatibility adapters may return the configuration directly.
        config = (
            config_entry[1]
            if isinstance(config_entry, tuple) and len(config_entry) == 2
            else config_entry
        )
        allowed_features = getattr(config, "features", None)
        return allowed_features is None or "SchedulerFeature" in allowed_features

    def _lifecycle_lock_for(self, agent_id: str) -> Any:
        """Return the manager's exclusive DID lifecycle lock."""

        factory = getattr(self._agent_manager, "scheduler_lifecycle_lock", None)
        if callable(factory):
            lock = factory(agent_id)
            if hasattr(lock, "__aenter__"):
                return lock
        return self._locks.setdefault(agent_id, asyncio.Lock())

    def _execution_lease_for(self, agent_id: str) -> Any:
        """Return a shared manager admission lease, if the host provides one."""

        factory = getattr(self._agent_manager, "scheduler_execution_lease", None)
        if callable(factory):
            lease = factory(agent_id)
            if hasattr(lease, "__aenter__"):
                return lease
        # Compatibility managers predate shared lifecycle admission. Their
        # only safe contract remains the exclusive lock used before this
        # protocol, so retain it rather than silently weakening deletion.
        return self._lifecycle_lock_for(agent_id)

    @asynccontextmanager
    async def prepare_scheduled(
        self, execution: SchedulerExecution
    ) -> AsyncIterator[PreparedScheduledDispatch]:
        """Cold-resolve under the DID lock before database admission.

        Scheduler feature initialization can perform additive schema DDL. A
        PostgreSQL admission transaction holds an ``ACCESS SHARE`` lock while
        it validates the claim, so doing that cold load inside admission can
        self-deadlock when initialization asks for ``ACCESS EXCLUSIVE``. Only
        cold initialization takes the manager's exclusive lifecycle lock.
        Every effect then takes a shared execution lease: sibling schedules
        can dispatch concurrently, while a deletion or authority mutation
        closes admission and drains them first.
        """

        shared_lease_factory = getattr(
            self._agent_manager, "scheduler_execution_lease", None
        )
        supports_shared_leases = callable(shared_lease_factory)
        # A warm agent can immediately join the shared execution lease. A
        # cold agent takes the writer only while it is initialized and made
        # visible to the host, then atomically downgrades to its reader lease.
        agent = self._loaded_agent_for(execution.agent_id)
        retained_cold_lease: Any = None
        if agent is None:
            lifecycle_lock = self._lifecycle_lock_for(execution.agent_id)
            downgrade = getattr(lifecycle_lock, "downgrade", None)
            if not supports_shared_leases or not callable(downgrade):
                # Compatibility managers expose only an exclusive lifecycle
                # lock. Retain their established all-or-nothing safety model
                # instead of assuming a reader/writer API they do not have.
                async with lifecycle_lock:
                    agent = await self._resolve_or_wake(execution.agent_id)
                    if _loaded_agent_did(agent) != execution.agent_id:
                        raise RuntimeError(
                            "Hosted scheduler resolved an agent whose DID does not "
                            f"match the claimed schedule agent: expected "
                            f"{execution.agent_id!r}"
                        )
                    if await self._live_config_for(execution.agent_id) is None:
                        raise LookupError(
                            "Hosted scheduler authority was revoked for "
                            f"{execution.agent_id!r}"
                        )
                    self._require_enabled_scheduler_feature(
                        agent, execution.agent_id
                    )

                    async def dispatch() -> Any:
                        return await self._dispatch_resolved_agent(agent, execution)

                    yield dispatch
                return

            await lifecycle_lock.acquire()
            writer_held = True
            try:
                agent = await self._resolve_or_wake(execution.agent_id)
                if _loaded_agent_did(agent) != execution.agent_id:
                    raise RuntimeError(
                        "Hosted scheduler resolved an agent whose DID does not match "
                        f"the claimed schedule agent: expected {execution.agent_id!r}"
                    )
                retained_cold_lease = downgrade()
                writer_held = False
            except BaseException:
                if writer_held:
                    lifecycle_lock.release()
                raise

        lease = retained_cold_lease or self._execution_lease_for(execution.agent_id)
        async with lease:
            # A writer may have revoked authority after preparation but before
            # this reader was admitted. Fail closed rather than dispatching a
            # stale prepared agent.
            if await self._live_config_for(execution.agent_id) is None:
                raise LookupError(
                    f"Hosted scheduler authority was revoked for {execution.agent_id!r}"
                )
            published_agent = (
                self._loaded_agent_for(execution.agent_id)
                if supports_shared_leases
                else agent
            )
            if published_agent is None:
                raise LookupError(
                    "Hosted scheduler agent was unpublished before dispatch for "
                    f"{execution.agent_id!r}"
                )
            if _loaded_agent_did(published_agent) != execution.agent_id:
                raise RuntimeError(
                    "Hosted scheduler resolved an agent whose DID does not match "
                    f"the claimed schedule agent: expected {execution.agent_id!r}"
                )
            # A same-DID replacement can happen between the initial warm
            # lookup and reader admission. Under the reader, use the live
            # published object rather than invoking a stale/shutting-down one.
            agent = published_agent
            self._require_enabled_scheduler_feature(agent, execution.agent_id)

            async def dispatch() -> Any:
                return await self._dispatch_resolved_agent(agent, execution)

            yield dispatch

    def _loaded_agent_for(self, agent_id: str) -> Optional[Any]:
        """Return an already-published agent for ``agent_id``, if present."""

        agents = self._agent_manager.list_agents()
        for agent in agents.values():
            if _loaded_agent_did(agent) == agent_id:
                return agent

        return None

    async def _resolve_or_wake(self, agent_id: str) -> Any:
        """Return a warm agent or cold-load exactly one under the writer."""

        agent = self._loaded_agent_for(agent_id)
        if agent is not None:
            return agent
        config = await self._live_config_for(agent_id)
        if config is None:
            raise LookupError(f"No hosted agent configuration for {agent_id!r}")
        name, local_config = config
        loader = self._agent_manager.load_agent
        loader_kwargs: dict[str, Any] = {"expected_agent_id": agent_id}
        try:
            signature = inspect.signature(loader)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and (
            "scheduler_lifecycle_lock_held" in signature.parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        ):
            # The manager writer lock is already held for cold initialization.
            # Tell the concrete manager not to recursively acquire its
            # non-reentrant per-DID lifecycle writer while loading this agent.
            loader_kwargs["scheduler_lifecycle_lock_held"] = True
        loaded = await loader(name, local_config, **loader_kwargs)
        loaded_agent_id = _loaded_agent_did(loaded)
        if loaded_agent_id != agent_id:
            # AgentManager enforces this before registration. Keep this
            # assertion at the dispatch boundary too so a nonconforming
            # manager implementation can never route a tenant-A context to
            # a tenant-B agent.
            raise RuntimeError(
                "Hosted scheduler loaded an agent whose DID does not match "
                f"the claimed schedule agent: expected {agent_id!r}, "
                f"got {loaded_agent_id!r}"
            )
        return loaded


class SchedulerRunner:
    """Poll durable schedule rows and execute only rows this runner claims.

    Pass an agent ID for the longstanding standalone per-agent loop.  A host
    may pass ``None`` plus a :class:`HostedExecutionExecutor` and an explicit
    set of locally-authorized agent DIDs to claim rows in a shared PostgreSQL
    database.  A host must never claim another host's fleet rows.
    """

    def __init__(
        self,
        db,
        agent_id: Optional[str],
        executor: Union[
            TaskExecutor,
            HostedExecutionExecutor,
            PreparedExecutionExecutor,
        ],
        poll_interval: int = POLL_INTERVAL,
        misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        owner_id: Optional[str] = None,
        authorized_agent_ids: Optional[Collection[str]] = None,
        authorized_agent_ids_provider: Optional[
            Callable[
                [],
                Union[
                    Collection[str],
                    Awaitable[Collection[str]],
                ],
            ]
        ] = None,
        is_agent_authorized: Optional[
            Callable[[str], Union[bool, Awaitable[bool]]]
        ] = None,
        on_protocol_failure: Optional[Callable[[BaseException], None]] = None,
    ):
        try:
            normalized_lease_seconds = int(lease_seconds)
        except (TypeError, ValueError) as e:
            raise ValueError("lease_seconds must be positive") from e
        if normalized_lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if authorized_agent_ids is not None:
            authorized = tuple(sorted(set(authorized_agent_ids)))
            if any(not isinstance(value, str) or not value for value in authorized):
                raise ValueError(
                    "authorized_agent_ids must contain only non-empty agent IDs"
                )
            if not authorized and authorized_agent_ids_provider is None:
                raise ValueError(
                    "an empty host scheduler scope requires a live "
                    "authorized_agent_ids_provider"
                )
        elif agent_id is None:
            raise ValueError(
                "host SchedulerRunner requires explicit authorized_agent_ids"
            )
        else:
            # Preserve the standalone runner's historical contract: its
            # provided agent identity is its complete authority scope.
            authorized = (agent_id,)
        if agent_id is not None and authorized != (agent_id,):
            raise ValueError(
                "agent-scoped SchedulerRunner may authorize only its agent_id"
            )
        self._db = db
        self._agent_id = agent_id
        self._authorized_agent_ids = authorized
        self._authorized_agent_ids_provider = authorized_agent_ids_provider
        self._executor = executor
        self._poll_interval = poll_interval
        self._misfire_grace_seconds = max(0, int(misfire_grace_seconds))
        self._max_concurrent_tasks = max(1, int(max_concurrent_tasks))
        self._lease_seconds = normalized_lease_seconds
        self._owner_id = owner_id or f"scheduler:{uuid.uuid4()}"
        # Standalone runners retain their fixed one-DID scope. Hosted runners
        # can additionally resolve a fresh manager-owned scope for every
        # rollout/selection/CAS boundary, so a runtime-created tenant becomes
        # executable without rebuilding the runner and a removed tenant drops
        # out immediately.
        self._is_agent_authorized = is_agent_authorized
        self._on_protocol_failure = on_protocol_failure
        self._readiness_failure: Optional[BaseException] = None
        self._schema_provenance: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # Set only after the schema and durable rollout state have both been
        # established. Production starts always go through ``start``; retaining
        # a false default keeps legacy private-tick test doubles from inventing
        # a protocol row without first creating its schema.
        self._protocol_ready = False
        self._rollout_acknowledgements = tuple(
            value.strip()
            for value in os.environ.get(SCHEDULER_ROLLOUT_ACK_ENV, "").split(",")
            if value.strip()
        )

    @property
    def readiness_failure(self) -> Optional[BaseException]:
        """Return a latched scheduler-safety failure, if one occurred.

        Task-level executor failures are represented in their execution logs;
        this latch is only for protocol/schema safety failures which mean the
        scheduler cannot safely provide its advertised fleet service.
        """

        return self._readiness_failure

    def _latch_protocol_failure(self, error: BaseException) -> None:
        """Persist the first safety failure in process state and notify host."""

        if self._readiness_failure is None:
            self._readiness_failure = error
        if self._on_protocol_failure is not None:
            try:
                self._on_protocol_failure(error)
            except Exception:  # pragma: no cover - host observability must not mask safety
                logger.exception("Scheduler protocol-failure callback failed")

    async def _agent_is_currently_authorized(self, agent_id: str) -> bool:
        """Check the current fleet scope and an optional live authority view."""

        if agent_id not in await self._current_authorized_agent_ids():
            return False
        if self._is_agent_authorized is None:
            return True
        result = self._is_agent_authorized(agent_id)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def _current_authorized_agent_ids(self) -> tuple[str, ...]:
        """Return a validated snapshot of this runner's current SQL scope."""

        provider = self._authorized_agent_ids_provider
        if provider is None:
            return self._authorized_agent_ids
        values = provider()
        if inspect.isawaitable(values):
            values = await values
        authorized = tuple(sorted(set(values)))
        if any(not isinstance(value, str) or not value for value in authorized):
            raise ValueError(
                "authorized_agent_ids_provider returned an invalid agent ID"
            )
        if self._agent_id is not None and authorized != (self._agent_id,):
            raise ValueError(
                "agent-scoped SchedulerRunner provider may authorize only its agent_id"
            )
        return authorized

    async def start(self, *, polling: bool = True):
        """Establish protocol state and optionally arm the polling loop.

        Feature-owned runners prepare their durable schema during feature
        initialization, then arm only from the agent's final ready hook. Host
        runners retain the historical one-call start behavior.
        """
        try:
            await self._ensure_tables()
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                self._latch_protocol_failure(error)
            raise
        if not polling:
            logger.info("SchedulerRunner %s prepared without polling", self._owner_id)
            return
        await self.arm()

    async def arm(self) -> None:
        """Start polling after durable protocol preparation has completed."""

        if self._running:
            return
        if not self._protocol_ready:
            raise RuntimeError(
                "SchedulerRunner cannot poll before protocol preparation"
            )
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="scheduler-runner")
        logger.info("SchedulerRunner %s started (poll every %ds)", self._owner_id, self._poll_interval)

    async def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SchedulerRunner %s stopped", self._owner_id)

    async def _loop(self):
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Executor failures are normalized into durable task outcomes
                # inside ``_execute_claim``. Anything escaping a tick is
                # scheduler infrastructure/protocol failure and must make host
                # readiness fail rather than merely emit a log line.
                self._latch_protocol_failure(error)
                logger.exception("SchedulerRunner tick error")
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise

    async def _tick(self):
        # A legacy binary can insert a row after this runner started. Check the
        # durable per-agent protocol state before every claim batch so an
        # unknown/null protocol row is fenced rather than silently adopted.
        if self._protocol_ready:
            try:
                await self._ensure_protocol_rollout(preexisting_schedule_table=True)
            except BaseException as error:
                if not isinstance(error, asyncio.CancelledError):
                    self._latch_protocol_failure(error)
                raise
        now = datetime.now(timezone.utc)
        rows = await self._due_rows(now)
        if not rows:
            return
        semaphore = asyncio.Semaphore(self._max_concurrent_tasks)

        async def run_one(task: ScheduledTask) -> None:
            async with semaphore:
                # ``now`` above is only the polling cutoff. A task can wait
                # behind the semaphore longer than its full lease interval, so
                # begin its lease at the actual compare-and-set transition.
                if not await self._executor_accepts_scheduled_agent(task.agent_id):
                    return
                claimed = await self._claim(task, datetime.now(timezone.utc))
                if claimed is not None:
                    await self._execute_claim(claimed)

        tasks = [ScheduledTask.from_row(row) for row in rows]
        results = await asyncio.gather(
            *(run_one(task) for task in tasks), return_exceptions=True
        )
        for task, result in zip(tasks, results):
            if not isinstance(result, BaseException):
                continue
            if isinstance(result, asyncio.CancelledError):
                logger.warning(
                    "Scheduled occurrence for task %s (%s) was cancelled",
                    task.id,
                    task.task_name,
                )
                continue
            logger.error(
                "Scheduled occurrence for task %s (%s) failed outside normal finalization",
                task.id,
                task.task_name,
                exc_info=(type(result), result, result.__traceback__),
            )
            self._latch_protocol_failure(result)

    async def _executor_accepts_scheduled_agent(self, agent_id: str) -> bool:
        """Return whether an optional hosted executor can admit this DID.

        The normal durable authorization check still runs in :meth:`_claim`.
        This hook only avoids publishing a claim for a warm hosted agent whose
        SchedulerFeature is already soft-disabled. Executors without an
        explicit preflight preserve the established claim-first behavior.
        """

        preflight = getattr(self._executor, "scheduler_dispatch_enabled", None)
        if not callable(preflight):
            return True
        result = preflight(agent_id)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def _due_rows(self, now: datetime) -> List[tuple]:
        authorized_agent_ids = await self._current_authorized_agent_ids()
        if not authorized_agent_ids:
            return []
        authorization_scope = self._authorized_agent_placeholders(
            authorized_agent_ids
        )
        if self._uses_database_clock():
            due_predicate = self._database_due_sql()
            expired = self._database_lease_expired_sql()
            params: tuple = (
                *authorized_agent_ids,
                SCHEDULER_PROTOCOL_VERSION,
            )
        else:
            due_predicate = "next_run_at IS NOT NULL AND next_run_at <= ?"
            expired = "(lease_expires_at IS NULL OR lease_expires_at <= ?)"
            params = (
                *authorized_agent_ids,
                SCHEDULER_PROTOCOL_VERSION,
                now.isoformat(),
                now.isoformat(),
            )
        return await self._db.fetchall(
            f"""
            SELECT id, agent_id, task_name, cron_expression, args_json,
                   enabled, last_run_at, next_run_at, created_at,
                   schedule_kind, run_at, timezone_name, misfire_policy,
                   misfire_grace_seconds, idempotency_key, lease_owner,
                   lease_expires_at, claim_token, claim_execution_id,
                   claim_scheduled_for, attempt_count, terminal_status, terminal_at,
                   scheduler_protocol_version, scheduler_rollout_fenced,
                   scheduler_claim_fenced, scheduler_rollout_fenced_at
            FROM scheduled_tasks
            WHERE agent_id IN ({authorization_scope})
              AND scheduler_protocol_version = ?
              AND scheduler_rollout_fenced = 0
              AND {due_predicate}
              AND (
                    (enabled = 1 AND scheduler_claim_fenced = 0)
                 OR (enabled = 0 AND scheduler_claim_fenced = 1)
              )
              AND {expired}
            ORDER BY next_run_at ASC
            """,
            params,
        )

    @asynccontextmanager
    async def _transaction(self):
        transaction = getattr(self._db, "transaction", None)
        if not callable(transaction):
            yield
            return
        context = transaction()
        if not hasattr(context, "__aenter__"):
            yield
            return
        try:
            async with context:
                yield
        except BaseException as error:
            # Both concrete backends normalize arbitrary exceptions escaping a
            # top-level transaction into TransactionError. Protocol
            # incompatibility is a typed readiness boundary, not a generic
            # query failure: recover the original sanitized exception from the
            # causal chain so callers and /health retain the actionable type.
            current: Optional[BaseException] = error
            seen: set[int] = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                if isinstance(current, SchedulerProtocolVersionIncompatible):
                    raise current from None
                current = current.__cause__ or current.__context__
            raise

    @staticmethod
    def _updated(result: Any) -> bool:
        """Interpret backend row counts without breaking legacy test doubles."""
        if isinstance(result, bool):
            return result
        if isinstance(result, int):
            return result > 0
        rowcount = getattr(result, "rowcount", None)
        if isinstance(rowcount, int):
            return rowcount > 0
        return True

    @staticmethod
    def _authorized_agent_placeholders(agent_ids: Collection[str]) -> str:
        """Return placeholders for one already-validated fleet snapshot."""

        return ", ".join("?" for _ in agent_ids)

    def _database_backend_type(self) -> str:
        """Return the concrete backend type without trusting loose test doubles."""

        return _scheduler_database_backend_type(self._db)

    def _uses_database_clock(self) -> bool:
        """Whether this DB supports the portable statement-time SQL paths."""

        return self._database_backend_type() in {"postgres", "sqlite"}

    def _database_now_sql(self) -> str:
        """Return a textual UTC statement-time clock expression.

        ``now()`` is deliberately not used for PostgreSQL: it is frozen at
        transaction start, so a row-lock wait can publish a lease that was
        already stale before commit.  ``clock_timestamp()`` is evaluated by
        the statement after that wait.  SQLite's ``strftime(..., 'now')`` has
        the same statement-time role and preserves the scheduler's ISO text
        timestamp representation.
        """

        return scheduler_database_now_sql(self._db)

    def _database_lease_expiry_sql(self) -> tuple[str, tuple[Any, ...]]:
        """Return a database-time lease expiry expression and its parameters."""

        if self._database_backend_type() == "postgres":
            return (
                "(to_char((clock_timestamp() + (? * INTERVAL '1 second')) "
                "AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US') || '+00:00')",
                (self._lease_seconds,),
            )
        if self._database_backend_type() == "sqlite":
            return (
                "strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now', ?)",
                (f"+{self._lease_seconds} seconds",),
            )
        raise RuntimeError("scheduler database clock is unavailable for this backend")

    def _database_lease_expired_sql(self, column: str = "lease_expires_at") -> str:
        """Return a backend-safe, database-clock expiry predicate.

        A host-clock string comparison is both skew-prone and wrong for legacy
        spellings such as ``Z`` or SQLite's space separator.  PostgreSQL casts
        valid text to ``timestamptz`` at statement time; malformed non-empty
        state raises visibly instead of being silently adopted. SQLite's
        ``julianday`` returns NULL for malformed text, which deliberately
        leaves the row non-recoverable until an operator repairs it.
        """

        if self._database_backend_type() == "postgres":
            timestamp = self._postgres_utc_timestamp_sql(column)
            return (
                f"({column} IS NULL OR "
                f"{timestamp} <= clock_timestamp())"
            )
        if self._database_backend_type() == "sqlite":
            return (
                f"({column} IS NULL OR "
                f"julianday({column}) <= julianday('now'))"
            )
        raise RuntimeError("scheduler database clock is unavailable for this backend")

    def _database_lease_live_sql(self, column: str = "lease_expires_at") -> str:
        """Return the inverse live-lease predicate using statement-time DB time."""

        if self._database_backend_type() == "postgres":
            timestamp = self._postgres_utc_timestamp_sql(column)
            return (
                f"({column} IS NOT NULL AND "
                f"{timestamp} > clock_timestamp())"
            )
        if self._database_backend_type() == "sqlite":
            return (
                f"({column} IS NOT NULL AND "
                f"julianday({column}) > julianday('now'))"
            )
        raise RuntimeError("scheduler database clock is unavailable for this backend")

    def _database_due_sql(self, column: str = "next_run_at") -> str:
        """Return a database-clock due predicate tolerant of legacy formats."""

        if self._database_backend_type() == "postgres":
            timestamp = self._postgres_utc_timestamp_sql(column)
            return (
                f"({column} IS NOT NULL AND "
                f"{timestamp} <= clock_timestamp())"
            )
        if self._database_backend_type() == "sqlite":
            return (
                f"({column} IS NOT NULL AND "
                f"julianday({column}) <= julianday('now'))"
            )
        raise RuntimeError("scheduler database clock is unavailable for this backend")

    @staticmethod
    def _postgres_utc_timestamp_sql(column: str) -> str:
        """Parse scheduler timestamp text without inheriting session timezone.

        SQLite treats an offset-less ISO timestamp (including its legacy space
        separator) as UTC. PostgreSQL instead interprets such text in the
        connection's ``TimeZone`` when casting directly to ``timestamptz``.
        Preserve an explicit ``Z``/numeric offset, but attach UTC to the
        scheduler's legacy offset-less values so both backends agree.
        """

        value = f"NULLIF({column}, '')"
        # Scheduler-generated values always carry a colonized offset, but
        # accept the common compact numeric form while reading older
        # hand-written rows. Do not match a bare ``+HH`` suffix: a date-only
        # value can end in ``-DD`` and must remain an offset-less UTC value.
        has_explicit_offset = (
            "'(z|[+-][0-9]{2}(:[0-9]{2}|[0-9]{2}))$'"
        )
        return (
            "(CASE "
            f"WHEN {value} IS NULL THEN NULL "
            f"WHEN {value} ~* {has_explicit_offset} THEN {value}::timestamptz "
            f"ELSE ({value}::timestamp AT TIME ZONE 'UTC') END)"
        )

    async def _database_clock(self) -> datetime:
        """Read the database wall clock after a row lock for cron calculations."""

        return await scheduler_database_clock(self._db)

    async def _lock_claim_candidate(self, task: ScheduledTask) -> bool:
        """Acquire the schedule-row lock before taking a durable claim.

        This intentionally does *not* publish a lease.  A PostgreSQL claimant
        can otherwise wait on a row/log lock after calculating its lease and
        commit an already-expired value.  SQLite's no-op write obtains the
        equivalent transaction writer slot.
        """

        if self._database_backend_type() == "postgres":
            row = await self._db.fetchone(
                """
                SELECT id, scheduler_protocol_version, scheduler_registration_nonce
                FROM scheduled_tasks
                WHERE id = ? AND agent_id = ?
                FOR UPDATE
                """,
                (task.id, task.agent_id),
            )
            if row is None:
                return False
            try:
                if int(row[1]) > SCHEDULER_PROTOCOL_VERSION:
                    return False
            except (TypeError, ValueError):
                return False
            return await adopt_scheduler_registration_ownership(
                self._db,
                task_id=task.id,
                agent_id=task.agent_id,
                observed_registration_nonce=row[2] if len(row) > 2 else None,
                pending_registration_nonce=None,
            )
        if self._database_backend_type() == "sqlite":
            result = await self._db.execute(
                """
                UPDATE scheduled_tasks
                SET scheduler_claim_fenced = scheduler_claim_fenced
                WHERE id = ? AND agent_id = ?
                """,
                (task.id, task.agent_id),
            )
            if not self._updated(result):
                return False
            row = await self._db.fetchone(
                """
                SELECT scheduler_protocol_version, scheduler_registration_nonce
                FROM scheduled_tasks
                WHERE id = ? AND agent_id = ?
                """,
                (task.id, task.agent_id),
            )
            if row is None:
                return False
            try:
                if int(row[0]) > SCHEDULER_PROTOCOL_VERSION:
                    return False
            except (TypeError, ValueError):
                return False
            return await adopt_scheduler_registration_ownership(
                self._db,
                task_id=task.id,
                agent_id=task.agent_id,
                observed_registration_nonce=row[1] if len(row) > 1 else None,
                pending_registration_nonce=None,
            )
        return True

    async def _lock_active_rollout_control(self, agent_id: str) -> bool:
        """Lock one DID's active rollout epoch for claim/dispatch admission."""

        if not self._uses_database_clock():
            return True
        updated = await self._db.execute(
            """
            UPDATE scheduler_protocol_rollout
            SET updated_at = updated_at, scheduler_registration_nonce = NULL
            WHERE agent_id = ? AND protocol_version = ? AND state = 'active'
            """,
            (agent_id, SCHEDULER_PROTOCOL_VERSION),
        )
        return self._updated(updated)

    @staticmethod
    def _rollout_effect_advisory_key(agent_id: str) -> int:
        """Return a stable signed-32-bit PostgreSQL advisory key for one DID."""

        digest = hashlib.sha256(
            f"scheduler-rollout-effect\0{agent_id}".encode("utf-8")
        ).digest()
        key = int.from_bytes(digest[:4], byteorder="big", signed=True)
        # ``(2715, 0)`` is the transaction-scoped bootstrap key. A session
        # effect gate at that exact key makes bootstrap wait on itself across
        # its advisory and operational connections. Preserve the long-lived
        # namespace for rollout compatibility, but remap this one sentinel.
        return 1 if key == _SCHEDULER_BOOTSTRAP_ADVISORY_LOCK_SCOPE else key

    @asynccontextmanager
    async def _postgres_rollout_effect_gates(
        self, agent_ids: Collection[str], *, shared: bool = False
    ):
        """Take PostgreSQL rollout readers or an exclusive transition writer.

        This is intentionally a dedicated-session advisory lock, not a
        transaction containing the target effect. Effects hold a shared lock
        for their full external-effect/final-CAS
        span. A rollout transition takes the corresponding exclusive lock
        before opening its transaction. Therefore a transition closes future
        admission, drains every previously-admitted effect, then changes the
        durable epoch. Scheduler tools remain free to mutate ordinary
        scheduler rows while an effect is running.
        """

        if self._database_backend_type() != "postgres":
            yield
            return
        backend = getattr(self._db, "backend", None)
        advisory_locks = getattr(backend, "advisory_locks", None)
        if not callable(advisory_locks):
            raise RuntimeError(
                "PostgreSQL scheduler execution requires backend advisory-lock support"
            )
        keys = tuple(
            (
                _SCHEDULER_EFFECT_ADVISORY_LOCK_NAMESPACE,
                self._rollout_effect_advisory_key(agent_id),
            )
            for agent_id in sorted(set(agent_ids))
        )
        async with advisory_locks(keys, shared=shared):
            yield

    @asynccontextmanager
    async def _postgres_rollout_effect_gate(self, agent_id: str):
        """Hold one shared PostgreSQL effect admission lease."""

        async with self._postgres_rollout_effect_gates((agent_id,), shared=True):
            yield

    @asynccontextmanager
    async def _postgres_rollout_transition_gate(self, agent_id: str):
        """Hold one exclusive PostgreSQL rollout transition lease."""

        async with self._postgres_rollout_effect_gates((agent_id,)):
            yield

    def _sqlite_rollout_lock_path(self, agent_id: str) -> Optional[str]:
        """Return a stable sidecar lock path for a file-backed SQLite database."""

        if self._database_backend_type() != "sqlite":
            return None
        backend = getattr(self._db, "backend", None)
        db_path = getattr(backend, "db_path", None)
        if not isinstance(db_path, str) or not db_path or db_path == ":memory:":
            return None
        # The DID is hashed so an unusual (but valid) DID cannot influence a
        # filesystem path. Resolve aliases before deriving the sidecar so a
        # real SQLite file and a symlink to it coordinate through one rollout
        # lock. ``realpath`` also has useful non-strict behavior for a database
        # that has not been created yet (or a broken leaf symlink): it resolves
        # every existing parent and retains the unresolved suffix.
        canonical = os.path.realpath(os.path.abspath(db_path))
        digest = hashlib.sha256(
            f"{canonical}\0{agent_id}".encode("utf-8")
        ).hexdigest()
        return f"{canonical}.scheduler-rollout-{digest}.lock"

    @asynccontextmanager
    async def _sqlite_rollout_gate(self, agent_id: str, *, shared: bool = False):
        """Take SQLite rollout readers or an exclusive transition writer."""

        if self._database_backend_type() != "sqlite":
            yield
            return
        path = self._sqlite_rollout_lock_path(agent_id)
        if path is None:
            # In-memory SQLite cannot be shared with another process. A
            # process-local gate remains necessary for two runners using the
            # same AsyncDatabase object in tests or embedded hosts.
            lock = _sqlite_memory_rollout_locks.setdefault(
                (id(self._db), agent_id), AsyncReaderWriterLock()
            )
            context = lock.read() if shared else lock
            async with context:
                yield
            return

        lock = _SQLiteRolloutFileLock(path, shared=shared)
        acquired = False
        try:
            # The owned-task wrapper waits for a blocking flock/locking call
            # to reach a terminal state even when shutdown cancels this task.
            # If cancellation races a successful acquire, release below before
            # propagating it so no orphaned advisory lock survives.
            await run_blocking_operation(lock.acquire)
            acquired = True
        except asyncio.CancelledError:
            # ``flock(..., LOCK_UN)`` and ``UnlockFileEx`` are immediate
            # unlock operations: neither waits for another reader or writer.
            # Running them here, rather than queueing onto the default
            # executor, guarantees an admitted effect can release its lock
            # even when that executor is full of blocking acquisitions. It
            # also adds no runner-owned executor that needs shutdown cleanup.
            lock.release()
            raise
        try:
            yield
        finally:
            if acquired:
                lock.release()

    @asynccontextmanager
    async def _bootstrap_serialization_boundary(self) -> AsyncIterator[None]:
        """Hold one backend-wide gate across scheduler bootstrap mutation.

        PostgreSQL first drains every fixed-scope effect through exclusive
        session advisory locks, then keeps its transaction-scoped schema lock
        until all DDL and DID rollout state are durable. File-backed SQLite
        takes a global writer lock *before* its write transaction, then takes
        normal per-DID writer gates in deterministic order. That ordering
        avoids a bootstrap writer holding the database writer while waiting
        for admitted effects to leave their DID leases.
        """

        async with self._sqlite_rollout_gate(_SCHEDULER_BOOTSTRAP_LOCK_SCOPE):
            async with AsyncExitStack() as gates:
                for agent_id in self._authorized_agent_ids:
                    await gates.enter_async_context(
                        self._sqlite_rollout_gate(agent_id)
                    )
                async with self._postgres_rollout_effect_gates(
                    self._authorized_agent_ids
                ):
                    async with self._transaction():
                        await self._acquire_scheduler_schema_lock()
                        yield

    @asynccontextmanager
    async def _active_dispatch_admission(self, task: ScheduledTask):
        """Linearize executor entry against active→quiescing fencing.

        PostgreSQL uses a shared per-DID session advisory lease for each
        effect span and holds the active control-row transaction only long
        enough to record admission. The matching active→quiescing transition
        takes the exclusive lease *before* changing the control row. SQLite
        follows the same shared/exclusive admission rule with OS locks, but
        releases its single-writer transaction immediately after checking the
        active row so renewal and target storage writes can proceed. If an
        effect wins it is linearized before the rollout fence; if fencing wins,
        admission sees ``quiescing`` and never invokes the executor.
        """

        if not self._uses_database_clock():
            yield True
            return
        if self._database_backend_type() == "sqlite":
            # Do not retain SQLite's sole writer transaction across an
            # executor. The advisory gate has the same protocol linearization
            # role while allowing lease renewal and ordinary tool writes.
            async with self._sqlite_rollout_gate(task.agent_id, shared=True):
                async with self._transaction():
                    admitted = await self._lock_active_rollout_control(task.agent_id)
                yield admitted
            return
        async with self._postgres_rollout_effect_gate(task.agent_id):
            async with self._transaction():
                admitted = await self._lock_active_rollout_control(task.agent_id)
            yield admitted

    async def _lock_claim_execution_log(
        self, execution_id: str, task: ScheduledTask
    ) -> None:
        """Lock an existing recovery log before the lease clock is sampled."""

        if self._database_backend_type() != "postgres":
            return
        await self._db.fetchone(
            """
            SELECT id FROM task_execution_log
            WHERE id = ? AND task_id = ? AND agent_id = ?
            FOR UPDATE
            """,
            (execution_id, task.id, task.agent_id),
        )

    async def _locked_claim_metadata(
        self, task: ScheduledTask,
    ) -> Optional[tuple[Optional[str], Optional[str], int]]:
        """Read recovery identity after the schedule row is locked.

        ``_due_rows`` intentionally returns an unlocked snapshot.  A worker
        can therefore select an unclaimed row, wait while another worker
        claims it, and only resume after that claim expires.  Recovery must
        use the claim identity now stored in the durable row, rather than the
        stale snapshot it originally selected; otherwise it creates a second
        execution-log/idempotency identity for one occurrence.

        The caller holds ``_lock_claim_candidate``'s PostgreSQL row lock or
        SQLite writer slot, so this is an authoritative single-row read.
        Lightweight non-database test doubles retain their historical local
        snapshot path and do not call this helper.
        """

        row = await self._db.fetchone(
            """
            SELECT claim_execution_id, claim_scheduled_for, attempt_count
            FROM scheduled_tasks
            WHERE id = ? AND agent_id = ?
            """,
            (task.id, task.agent_id),
        )
        if row is None:
            return None
        if len(row) < 3:
            raise RuntimeError(
                f"scheduler claim {task.id} returned incomplete durable claim metadata"
            )
        execution_id = row[0]
        scheduled_for = row[1]
        return (
            execution_id if isinstance(execution_id, str) and execution_id else None,
            scheduled_for if isinstance(scheduled_for, str) and scheduled_for else None,
            int(row[2] or 0),
        )

    async def _lock_live_claim_for_finalization(
        self, task: ScheduledTask, execution: SchedulerExecution
    ) -> bool:
        """Lock a still-live token before computing its terminal transition."""

        predicate = self._database_lease_live_sql()
        if self._database_backend_type() == "postgres":
            row = await self._db.fetchone(
                f"""
                SELECT id FROM scheduled_tasks
                WHERE id = ? AND agent_id = ?
                  AND scheduler_protocol_version = ?
                  AND scheduler_rollout_fenced = 0
                  AND enabled = 0 AND scheduler_claim_fenced = 1
                  AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
                  AND {predicate}
                FOR UPDATE
                """,
                (
                    task.id, task.agent_id, SCHEDULER_PROTOCOL_VERSION,
                    self._owner_id, task.claim_token, execution.id,
                ),
            )
            return row is not None
        if self._database_backend_type() == "sqlite":
            updated = await self._db.execute(
                f"""
                UPDATE scheduled_tasks
                SET scheduler_claim_fenced = scheduler_claim_fenced
                WHERE id = ? AND agent_id = ?
                  AND scheduler_protocol_version = ?
                  AND scheduler_rollout_fenced = 0
                  AND enabled = 0 AND scheduler_claim_fenced = 1
                  AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
                  AND {predicate}
                """,
                (
                    task.id, task.agent_id, SCHEDULER_PROTOCOL_VERSION,
                    self._owner_id, task.claim_token, execution.id,
                ),
            )
            return self._updated(updated)
        return False

    async def _lock_claim_token_for_renewal(self, task: ScheduledTask) -> bool:
        """Lock the exact claim before sampling its lease clock again.

        The lock predicate deliberately excludes lease liveness. PostgreSQL
        can evaluate a volatile clock predicate before it waits on a row lock,
        so putting ``clock_timestamp()`` in this statement could let a blocked
        renewal carry an already-expired decision forward. The following
        renewal UPDATE runs only after this lock is held and evaluates its
        liveness predicate in a fresh statement.
        """

        if self._database_backend_type() == "postgres":
            row = await self._db.fetchone(
                """
                SELECT id FROM scheduled_tasks
                WHERE id = ? AND agent_id = ?
                  AND scheduler_protocol_version = ?
                  AND scheduler_rollout_fenced = 0
                  AND enabled = 0 AND scheduler_claim_fenced = 1
                  AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
                FOR UPDATE
                """,
                (
                    task.id,
                    task.agent_id,
                    SCHEDULER_PROTOCOL_VERSION,
                    self._owner_id,
                    task.claim_token,
                    task.claim_execution_id,
                ),
            )
            return row is not None
        if self._database_backend_type() == "sqlite":
            # SQLite has no ``FOR UPDATE``. This no-op write obtains its
            # single writer slot inside the short renewal transaction before
            # the next statement samples ``strftime(..., 'now')``.
            updated = await self._db.execute(
                """
                UPDATE scheduled_tasks
                SET scheduler_claim_fenced = scheduler_claim_fenced
                WHERE id = ? AND agent_id = ?
                  AND scheduler_protocol_version = ?
                  AND scheduler_rollout_fenced = 0
                  AND enabled = 0 AND scheduler_claim_fenced = 1
                  AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
                """,
                (
                    task.id,
                    task.agent_id,
                    SCHEDULER_PROTOCOL_VERSION,
                    self._owner_id,
                    task.claim_token,
                    task.claim_execution_id,
                ),
            )
            return self._updated(updated)
        return False

    async def _claim_token_is_live(self, task: ScheduledTask) -> bool:
        """Verify the exact lease token against database time before dispatch."""

        if not task.claim_token or not task.claim_execution_id:
            return False
        if not self._uses_database_clock():
            now_iso = datetime.now(timezone.utc).isoformat()
            row = await self._db.fetchone(
                """
                SELECT 1 FROM scheduled_tasks
                WHERE id = ? AND agent_id = ? AND lease_owner = ?
                  AND claim_token = ? AND claim_execution_id = ?
                  AND lease_expires_at > ?
                """,
                (
                    task.id, task.agent_id, self._owner_id,
                    task.claim_token, task.claim_execution_id, now_iso,
                ),
            )
            return row is not None
        row = await self._db.fetchone(
            f"""
            SELECT 1 FROM scheduled_tasks
            WHERE id = ? AND agent_id = ?
              AND scheduler_protocol_version = ?
              AND scheduler_rollout_fenced = 0
              AND enabled = 0 AND scheduler_claim_fenced = 1
              AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
              AND {self._database_lease_live_sql()}
            """,
            (
                task.id, task.agent_id, SCHEDULER_PROTOCOL_VERSION,
                self._owner_id, task.claim_token, task.claim_execution_id,
            ),
        )
        return row is not None

    async def _terminalize_inconsistent_execution_log(
        self,
        task: ScheduledTask,
        *,
        token: str,
        execution_id: str,
        status: Optional[str],
    ) -> None:
        """Fail closed when recovery finds a terminal log for a live schedule.

        A claimed task row paired with a terminal execution record is corrupt:
        dispatching it can duplicate an already-completed external effect, while
        overwriting the terminal log would erase audit evidence. Disable the
        schedule visibly and leave the historical log intact for repair.
        """

        detail = f"execution log status is {status!r}, expected 'claimed'"
        if self._uses_database_clock():
            terminal_at = self._database_now_sql()
            params: tuple[Any, ...] = (
                task.id,
                task.agent_id,
                self._owner_id,
                token,
                execution_id,
            )
        else:
            terminal_at = "?"
            params = (
                datetime.now(timezone.utc).isoformat(),
                task.id,
                task.agent_id,
                self._owner_id,
                token,
                execution_id,
            )
        updated = await self._db.execute(
            f"""
            UPDATE scheduled_tasks
            SET enabled = 0, scheduler_claim_fenced = 0,
                lease_owner = NULL, lease_expires_at = NULL, claim_token = NULL,
                claim_execution_id = NULL, claim_scheduled_for = NULL,
                terminal_status = 'execution_log_inconsistent',
                terminal_at = {terminal_at}
            WHERE id = ? AND agent_id = ? AND lease_owner = ?
              AND claim_token = ? AND claim_execution_id = ?
            """,
            params,
        )
        if not self._updated(updated):
            raise RuntimeError(
                f"scheduler claim {task.id} changed while handling inconsistent execution log"
            )
        logger.error(
            "Disabled scheduler task %s because execution log %s is inconsistent: %s",
            task.id,
            execution_id,
            detail,
        )

    @staticmethod
    def _legacy_base_idempotency_key(schedule_id: str) -> str:
        """Return a bounded stable base for a row predating user-provided keys."""

        candidate = f"legacy:{schedule_id}"
        if validate_schedule_idempotency_base(candidate) is None:
            return candidate
        digest = hashlib.sha256(schedule_id.encode("utf-8")).hexdigest()
        return f"legacy-sha256:{digest}"

    async def _claim(
        self, task: ScheduledTask, _polled_at: datetime
    ) -> Optional[ScheduledTask]:
        if (
            task.next_run_at is None
            or not await self._agent_is_currently_authorized(task.agent_id)
        ):
            return None
        scheduled_for = task.next_run_at
        base_idempotency = task.idempotency_key or self._legacy_base_idempotency_key(task.id)
        base_error = validate_schedule_idempotency_base(base_idempotency)
        if base_error is not None:
            await self._disable_invalid_idempotency_key(task, base_error)
            return None
        # This provisional identity is used only by the legacy test-double
        # path below. Concrete backends replace it after locking and rereading
        # the schedule row, which is essential for stale due-row recovery.
        execution_id = (
            task.claim_execution_id
            if task.claim_scheduled_for == scheduled_for and task.claim_execution_id
            else str(uuid.uuid4())
        )
        token = str(uuid.uuid4())
        idempotency_key = self._occurrence_idempotency_key(base_idempotency, task.id, scheduled_for)
        lease_expires: Optional[str] = None
        attempt = (
            task.attempt_count + 1
            if task.claim_scheduled_for == scheduled_for
            and task.claim_execution_id is not None
            else 1
        )

        authorized_agent_ids = await self._current_authorized_agent_ids()
        if task.agent_id not in authorized_agent_ids:
            return None
        authorization_scope = self._authorized_agent_placeholders(
            authorized_agent_ids
        )
        uses_database_clock = self._uses_database_clock()
        # Claim publication is a reader too: it must not cross a transition,
        # but it can overlap an already-admitted sibling effect.
        async with self._sqlite_rollout_gate(task.agent_id, shared=True):
            async with self._transaction():
                if uses_database_clock:
                    # Serialize claim publication with the same per-DID epoch
                    # boundary used by executor admission (a PostgreSQL row
                    # lock or SQLite advisory gate). A recovery cannot steal
                    # an expired lease while an already-admitted effect is in
                    # flight, and a quiescing transition wins before any new
                    # claim is made.
                    if not await self._lock_active_rollout_control(task.agent_id):
                        return None
                    # Acquire both rows which can block before sampling the
                    # lease clock. In particular, recovery can wait on an
                    # existing execution-log row; publishing a timestamp
                    # calculated before that wait creates an immediately
                    # expired claim on PostgreSQL.
                    if not await self._lock_claim_candidate(task):
                        return None
                    durable_claim = await self._locked_claim_metadata(task)
                    if durable_claim is None:
                        # A concurrent administrative delete won after due
                        # selection. The locked row is gone, so no execution
                        # can be admitted from this stale snapshot.
                        return None
                    (
                        durable_execution_id,
                        durable_scheduled_for,
                        durable_attempt_count,
                    ) = durable_claim
                    if (
                        durable_execution_id is not None
                        and durable_scheduled_for == scheduled_for
                    ):
                        execution_id = durable_execution_id
                        attempt = durable_attempt_count + 1
                    else:
                        execution_id = str(uuid.uuid4())
                        attempt = 1
                    await self._lock_claim_execution_log(execution_id, task)
                    lease_assignment, lease_assignment_params = (
                        self._database_lease_expiry_sql()
                    )
                    lease_expiry_predicate = self._database_lease_expired_sql()
                    log_clock = self._database_now_sql()
                    now_iso: Optional[str] = None
                else:
                    # Retain the historic adapter path for deliberately
                    # minimal unit-test doubles which do not identify a
                    # concrete backend.
                    claim_now = datetime.now(timezone.utc)
                    now_iso = claim_now.isoformat()
                    lease_expires = (
                        claim_now + timedelta(seconds=self._lease_seconds)
                    ).isoformat()
                    lease_assignment = "?"
                    lease_assignment_params = (lease_expires,)
                    lease_expiry_predicate = (
                        "(lease_expires_at IS NULL OR lease_expires_at <= ?)"
                    )
                    log_clock = "?"
                updated = await self._db.execute(
                    f"""
                    UPDATE scheduled_tasks
                    SET enabled = 0, scheduler_claim_fenced = 1,
                        lease_owner = ?, lease_expires_at = {lease_assignment}, claim_token = ?,
                        claim_execution_id = ?, claim_scheduled_for = ?,
                        attempt_count = CASE
                            WHEN claim_scheduled_for = ?
                             AND claim_execution_id IS NOT NULL
                            THEN COALESCE(attempt_count, 0) + 1
                            ELSE 1
                        END,
                        idempotency_key = COALESCE(idempotency_key, ?)
                    WHERE id = ? AND agent_id = ?
                      AND agent_id IN ({authorization_scope})
                      AND scheduler_protocol_version = ?
                      AND scheduler_rollout_fenced = 0
                      AND next_run_at = ?
                      -- Do not claim a row whose stable base changed after due
                      -- selection. The post-CAS log must use exactly the base
                      -- we validated above; a different persisted base is
                      -- retried on the next tick and then visibly disabled if
                      -- invalid.
                      AND (idempotency_key IS NULL OR idempotency_key = ?)
                      AND (
                            (enabled = 1 AND scheduler_claim_fenced = 0)
                         OR (enabled = 0 AND scheduler_claim_fenced = 1)
                      )
                      AND {lease_expiry_predicate}
                    """,
                    (
                        self._owner_id,
                        *lease_assignment_params,
                        token,
                        execution_id,
                        scheduled_for,
                        scheduled_for,
                        base_idempotency,
                        task.id,
                        task.agent_id,
                        *authorized_agent_ids,
                        SCHEDULER_PROTOCOL_VERSION,
                        scheduled_for,
                        base_idempotency,
                        *((now_iso,) if now_iso is not None else ()),
                    ),
                )
                if not self._updated(updated):
                    return None
                claimed_row = await self._db.fetchone(
                    """
                    SELECT claim_execution_id, attempt_count, idempotency_key,
                           lease_expires_at
                    FROM scheduled_tasks
                    WHERE id = ? AND agent_id = ? AND lease_owner = ?
                      AND claim_token = ?
                    """,
                    (task.id, task.agent_id, self._owner_id, token),
                )
                if claimed_row is None:
                    # Production backends return an integer rowcount, so a
                    # missing row here proves a broken transaction and must
                    # not execute. Retain the historical permissiveness only
                    # for lightweight test doubles that cannot model a
                    # post-CAS read.
                    if isinstance(updated, (bool, int)):
                        raise RuntimeError(
                            f"scheduler claim {task.id} disappeared after its successful CAS"
                        )
                elif len(claimed_row) >= 3:
                    execution_id = claimed_row[0]
                    attempt = int(claimed_row[1] or 0)
                    base_idempotency = claimed_row[2] or base_idempotency
                    if len(claimed_row) >= 4:
                        lease_expires = claimed_row[3]
                    idempotency_key = self._occurrence_idempotency_key(
                        base_idempotency, task.id, scheduled_for
                    )
                else:
                    # Production reads always select all three fields. A few
                    # lightweight historical test doubles use a short generic
                    # row for unrelated reads; retain their local provisional
                    # values without weakening the real backend check.
                    logger.debug(
                        "Scheduler claim %s test-double post-CAS row was short; "
                        "using local claim metadata",
                        task.id,
                    )
                await self._db.execute(
                    f"""
                    INSERT INTO task_execution_log
                        (id, task_id, agent_id, status, result_text, duration_ms,
                         executed_at, outcome_signal, occurrence_at, idempotency_key,
                         attempt_count, claimed_at, completed_at)
                    VALUES (?, ?, ?, 'claimed', NULL, 0, {log_clock}, NULL, ?, ?, ?, {log_clock}, NULL)
                    ON CONFLICT(id) DO UPDATE SET
                        attempt_count = CASE
                            WHEN task_execution_log.status = 'claimed'
                            THEN excluded.attempt_count
                            ELSE task_execution_log.attempt_count END,
                        claimed_at = CASE
                            WHEN task_execution_log.status = 'claimed'
                            THEN excluded.claimed_at
                            ELSE task_execution_log.claimed_at END
                    """,
                    (
                        (
                            execution_id,
                            task.id,
                            task.agent_id,
                            now_iso,
                            scheduled_for,
                            idempotency_key,
                            attempt,
                            now_iso,
                        )
                        if now_iso is not None
                        else (
                            execution_id,
                            task.id,
                            task.agent_id,
                            scheduled_for,
                            idempotency_key,
                            attempt,
                        )
                    ),
                )
                if uses_database_clock:
                    log_row = await self._db.fetchone(
                        """
                        SELECT status FROM task_execution_log
                        WHERE id = ? AND task_id = ? AND agent_id = ?
                        """,
                        (execution_id, task.id, task.agent_id),
                    )
                    if log_row is None or not log_row:
                        raise RuntimeError(
                            f"claimed execution log {execution_id} disappeared after claim"
                        )
                    if log_row[0] != "claimed":
                        await self._terminalize_inconsistent_execution_log(
                            task,
                            token=token,
                            execution_id=execution_id,
                            status=str(log_row[0]) if log_row[0] is not None else None,
                        )
                        return None
        return replace(
            task,
            lease_owner=self._owner_id,
            lease_expires_at=lease_expires,
            claim_token=token,
            claim_execution_id=execution_id,
            claim_scheduled_for=scheduled_for,
            attempt_count=attempt,
            idempotency_key=base_idempotency,
            enabled=False,
            scheduler_claim_fenced=True,
        )

    @staticmethod
    def _occurrence_idempotency_key(base: str, schedule_id: str, scheduled_for: str) -> str:
        base_error = validate_schedule_idempotency_base(base)
        if base_error is not None:
            raise ValueError(f"idempotency_key {base_error}")
        digest = hashlib.sha256(f"{schedule_id}\x00{scheduled_for}".encode()).hexdigest()
        key = f"{base}:{digest}"
        if len(key.encode("utf-8")) > MAX_EXECUTION_IDEMPOTENCY_KEY_BYTES:
            # Keep this defensive failure even though the base invariant above
            # should make it unreachable: an SDK context must never receive an
            # oversized stable user key.
            raise ValueError("derived scheduler idempotency_key exceeds 512 UTF-8 bytes")
        return key

    async def _disable_invalid_idempotency_key(
        self,
        task: ScheduledTask,
        base_error: str,
    ) -> None:
        """Fail closed and visibly preserve an invalid persisted user key."""

        if task.next_run_at is None:
            return
        authorized_agent_ids = await self._current_authorized_agent_ids()
        if task.agent_id not in authorized_agent_ids:
            return
        authorization_scope = self._authorized_agent_placeholders(
            authorized_agent_ids
        )
        async with self._transaction():
            uses_database_clock = self._uses_database_clock()
            now_iso = datetime.now(timezone.utc).isoformat()
            terminal_clock = (
                self._database_now_sql() if uses_database_clock else "?"
            )
            expired_predicate = (
                self._database_lease_expired_sql()
                if uses_database_clock
                else "(lease_expires_at IS NULL OR lease_expires_at <= ?)"
            )
            reason = f"schedule disabled: idempotency_key {base_error}"
            updated = await self._db.execute(
                f"""
                UPDATE scheduled_tasks
                SET enabled = 0, scheduler_claim_fenced = 0,
                    lease_owner = NULL, lease_expires_at = NULL,
                    claim_token = NULL, claim_execution_id = NULL,
                    claim_scheduled_for = NULL,
                    terminal_status = 'invalid_idempotency_key', terminal_at = {terminal_clock}
                WHERE id = ? AND agent_id = ?
                  AND agent_id IN ({authorization_scope})
                  AND scheduler_protocol_version = ?
                  AND scheduler_rollout_fenced = 0
                  AND next_run_at = ?
                  AND idempotency_key = ?
                  AND (
                        (enabled = 1 AND scheduler_claim_fenced = 0)
                     OR (enabled = 0 AND scheduler_claim_fenced = 1)
                  )
                  AND {expired_predicate}
                """,
                (
                    *((now_iso,) if not uses_database_clock else ()),
                    task.id, task.agent_id,
                    *authorized_agent_ids, SCHEDULER_PROTOCOL_VERSION,
                    task.next_run_at, task.idempotency_key,
                    *((now_iso,) if not uses_database_clock else ()),
                ),
            )
            if not self._updated(updated):
                return
            claimed_execution_id = task.claim_execution_id
            terminalized_claim = False
            if claimed_execution_id:
                terminalized = await self._db.execute(
                    f"""
                    UPDATE task_execution_log
                    SET status = 'invalid_idempotency_key', result_text = ?,
                        duration_ms = 0, executed_at = {terminal_clock},
                        completed_at = {terminal_clock}
                    WHERE id = ? AND task_id = ? AND agent_id = ?
                      AND status = 'claimed'
                    """,
                    (
                        (
                            reason,
                            claimed_execution_id,
                            task.id,
                            task.agent_id,
                        )
                        if uses_database_clock
                        else (
                            reason,
                            now_iso,
                            now_iso,
                            claimed_execution_id,
                            task.id,
                            task.agent_id,
                        )
                    ),
                )
                terminalized_claim = self._updated(terminalized)

            if not terminalized_claim:
                # A key may be invalid before any execution was claimed, or a
                # hand-edited DB may have lost its claimed-log row. In either
                # case preserve a visible terminal record rather than quietly
                # discarding the bad persisted state.
                await self._db.execute(
                    f"""
                    INSERT INTO task_execution_log
                        (id, task_id, agent_id, status, result_text, duration_ms,
                         executed_at, outcome_signal, occurrence_at, idempotency_key,
                         attempt_count, claimed_at, completed_at)
                    VALUES (?, ?, ?, 'invalid_idempotency_key', ?, 0, {terminal_clock}, NULL,
                            ?, NULL, 0, {terminal_clock}, {terminal_clock})
                    """,
                    (
                        (
                            str(uuid.uuid4()),
                            task.id,
                            task.agent_id,
                            reason,
                            task.next_run_at,
                        )
                        if uses_database_clock
                        else (
                            str(uuid.uuid4()),
                            task.id,
                            task.agent_id,
                            reason,
                            now_iso,
                            task.next_run_at,
                            now_iso,
                            now_iso,
                        )
                    ),
                )
        logger.error(
            "Disabled scheduler task %s for invalid persisted idempotency key: %s",
            task.id,
            base_error,
        )

    async def _execute_claim(self, task: ScheduledTask) -> None:
        if not await self._agent_is_currently_authorized(task.agent_id):
            logger.warning(
                "Refusing to execute scheduler claim %s for unauthorized agent %s",
                task.claim_execution_id,
                task.agent_id,
            )
            return
        assert task.claim_execution_id and task.claim_token and task.next_run_at
        # The claim transaction may have waited on the execution log and its
        # commit may in turn have waited on durability. Establish a fresh lease
        # before waiting for the active rollout epoch; the exact non-locking
        # token/live admission check occurs inside that epoch immediately before
        # the target effect. A stale or fenced token is never executable.
        if not await self._renew_lease_once(task):
            logger.warning(
                "Refusing to execute scheduler claim %s: lease is no longer live",
                task.claim_execution_id,
            )
            return
        execution = SchedulerExecution(
            id=task.claim_execution_id,
            schedule_id=task.id,
            agent_id=task.agent_id,
            task_name=task.task_name,
            args=task.args,
            scheduled_for=task.next_run_at,
            idempotency_key=self._occurrence_idempotency_key(
                task.idempotency_key or self._legacy_base_idempotency_key(task.id),
                task.id,
                task.next_run_at,
            ),
            attempt=task.attempt_count,
            owner=self._owner_id,
        )
        renewal_state = _LeaseRenewalState()
        renewal = asyncio.create_task(
            self._monitor_lease_renewal(task, renewal_state),
            name=f"scheduler-lease:{execution.id}",
        )
        started = time.monotonic()
        in_preparation = True
        try:
            # A prepared executor resolves/cold-starts before the PostgreSQL
            # control-row transaction. AgentManager retains a shared DID
            # execution lease through the eventual effect, so DELETE cannot
            # revoke the target between preparation and dispatch. Renewal is
            # already active while this may take time.
            async with self._prepared_execution(execution) as dispatch:
                in_preparation = False
                if renewal_state.lost.is_set():
                    self._log_lease_loss(execution, phase="preparation")
                    return
                async with self._active_dispatch_admission(task) as admitted:
                    if not admitted:
                        logger.warning(
                            "Refusing scheduler effect for %s: rollout is no longer active",
                            execution.id,
                        )
                        return
                    if renewal_state.lost.is_set():
                        self._log_lease_loss(execution, phase="admission")
                        return
                    now = (
                        await self._database_clock()
                        if self._uses_database_clock()
                        else datetime.now(timezone.utc)
                    )
                    late = self._seconds_late(task.next_run_at, now)
                    grace = (
                        self._misfire_grace_seconds
                        if task.misfire_grace_seconds is None
                        else max(0, int(task.misfire_grace_seconds))
                    )
                    policy = (
                        task.misfire_policy
                        if task.misfire_policy in MISFIRE_POLICIES
                        else MISFIRE_SKIP
                    )
                    if policy == MISFIRE_SKIP and grace and late > grace:
                        if renewal_state.lost.is_set():
                            self._log_lease_loss(execution, phase="misfire finalization")
                            return
                        await self._finalize(
                            task,
                            execution,
                            status="skipped_misfire",
                            result_text=(
                                f"skipped: {late:.0f}s late (> {grace}s misfire grace); "
                                "policy=skip"
                            ),
                            duration_ms=0,
                            outcome_signal=None,
                            ran=False,
                        )
                        return

                    status = "success"
                    result_text: Optional[str] = None
                    outcome_signal: Optional[float] = None
                    pause_schedule = False
                    try:
                        scope = _SchedulerExecutionScope(execution)
                        token = _current_execution.set(scope)
                        try:
                            # This is the final effect boundary. Preparation is
                            # complete (and, for AgentManager, the target is
                            # already cold-loaded under its lifecycle lock), so
                            # the non-locking exact-token read and live-authority
                            # check occur as late as possible without making an
                            # ``ACCESS SHARE`` lock block feature bootstrap DDL.
                            if (
                                renewal_state.lost.is_set()
                                or not await self._agent_is_currently_authorized(
                                    task.agent_id
                                )
                                or not await self._claim_token_is_live(task)
                                or renewal_state.lost.is_set()
                            ):
                                logger.warning(
                                    "Refusing scheduler effect for %s: agent was revoked, "
                                    "claim was fenced or expired, or renewal was lost",
                                    execution.id,
                                )
                                return
                            completed, raw = await self._run_dispatch_while_lease_live(
                                dispatch,
                                execution,
                                renewal_state,
                            )
                            if not completed:
                                self._log_lease_loss(execution, phase="effect")
                                return
                        finally:
                            # Invalidate before the parent context is reset.
                            # Child tasks created by a target inherit this same
                            # scope, so they can no longer present a completed
                            # occurrence as trusted scheduler work after the
                            # runner cancels or completes its owned effect.
                            scope.revoke()
                            _current_execution.reset(token)
                        (
                            status,
                            result_text,
                            outcome_signal,
                            pause_schedule,
                        ) = self._normalise_result(raw, task)
                    except asyncio.CancelledError:
                        raise
                    except SchedulerFeatureUnavailable:
                        # A runtime disable can race preparation/admission.
                        # It is not a task failure and must not advance this
                        # occurrence. Leaving the exact claim live prevents a
                        # hot retry; once it expires, a later re-enable can
                        # recover the same durable execution identity.
                        logger.info(
                            "Deferring scheduler claim %s because %s has no enabled "
                            "SchedulerFeature",
                            execution.id,
                            task.agent_id,
                        )
                        return
                    except Exception as e:
                        status = "failed"
                        result_text = f"{type(e).__name__}: {e}"
                        logger.error(
                            "Scheduled task %s (%s) failed: %s",
                            task.id,
                            task.task_name,
                            e,
                        )
                    if renewal_state.lost.is_set():
                        self._log_lease_loss(execution, phase="finalization")
                        return
                    # Keep renewal alive until the terminal compare-and-set
                    # commits. Cancelling it before this await creates a
                    # lease-expiry window in which a recovery worker can win
                    # while this worker still writes.
                    await self._finalize(
                        task,
                        execution,
                        status=status,
                        result_text=result_text,
                        duration_ms=int((time.monotonic() - started) * 1000),
                        outcome_signal=outcome_signal,
                        ran=True,
                        pause_schedule=pause_schedule,
                    )
        except asyncio.CancelledError:
            raise
        except SchedulerFeatureUnavailable:
            # Cold preparation can discover that a globally disabled or
            # otherwise unavailable feature was not visible in the host's
            # pre-claim configuration. Preserve the claimed occurrence (and
            # its recovery log) until the lease expires rather than recording
            # a synthetic failure that consumes/advances the schedule.
            logger.info(
                "Deferring scheduler claim %s because %s has no enabled "
                "SchedulerFeature",
                execution.id,
                task.agent_id,
            )
            return
        except Exception as error:
            if not in_preparation:
                # Database/admission failures remain scheduler infrastructure
                # failures. Do not disguise one as an ordinary cold-load task
                # result merely because the prepared context is still open.
                raise
            # Preparation errors were historically executor errors: retain a
            # visible failed execution when we still own a live claim, rather
            # than escalating a cold-agent startup failure into a runner-wide
            # protocol outage. Never finalize after the renewal monitor has
            # reported loss, however.
            logger.error(
                "Scheduled task %s (%s) preparation failed: %s",
                task.id,
                task.task_name,
                error,
            )
            if renewal_state.lost.is_set():
                self._log_lease_loss(execution, phase="preparation failure")
                return
            async with self._active_dispatch_admission(task) as admitted:
                if not admitted or renewal_state.lost.is_set():
                    logger.warning(
                        "Refusing scheduler preparation failure finalization for %s",
                        execution.id,
                    )
                    return
                await self._finalize(
                    task,
                    execution,
                    status="failed",
                    result_text=f"{type(error).__name__}: {error}",
                    duration_ms=int((time.monotonic() - started) * 1000),
                    outcome_signal=None,
                    ran=False,
                )
        finally:
            await self._stop_renewal(renewal, task, execution)

    def _prepared_executor_method(self) -> Optional[Callable[[SchedulerExecution], Any]]:
        """Return a structurally supplied preparation method, if any.

        Static discovery preserves compatibility with ``AsyncMock`` callable
        executors, which manufacture arbitrary dynamic attributes and must not
        be mistaken for hosted/prepared implementations.
        """

        try:
            inspect.getattr_static(self._executor, "prepare_scheduled")
        except AttributeError:
            return None
        method = getattr(self._executor, "prepare_scheduled", None)
        return method if callable(method) else None

    @asynccontextmanager
    async def _prepared_execution(
        self, execution: SchedulerExecution
    ) -> AsyncIterator[PreparedScheduledDispatch]:
        """Yield an executor dispatch after optional structural preparation."""

        prepare = self._prepared_executor_method()
        if prepare is None:
            async def dispatch() -> Any:
                return await self._run_executor(execution)

            yield dispatch
            return

        context = prepare(execution)
        # A normal ``@asynccontextmanager`` factory returns its context
        # directly. Accept an async factory too, which keeps the structural
        # contract friendly to integrations that construct one asynchronously.
        if inspect.isawaitable(context):
            context = await context
        if not hasattr(context, "__aenter__") or not hasattr(context, "__aexit__"):
            raise TypeError(
                "prepare_scheduled must return an async context manager"
            )
        async with context as dispatch:
            if not callable(dispatch):
                raise TypeError(
                    "prepare_scheduled must yield an async dispatch callable"
                )
            yield dispatch

    async def _monitor_lease_renewal(
        self, task: ScheduledTask, state: _LeaseRenewalState
    ) -> None:
        """Run a renewal worker and make any terminal outcome fail closed."""

        try:
            await self._renew_lease(task)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            state.mark_lost(error)
            raise
        else:
            # The renewal loop has no successful terminal state. A normal
            # return means the exact claim or live authority was lost.
            state.mark_lost()

    def _log_lease_loss(self, execution: SchedulerExecution, *, phase: str) -> None:
        """Report a fail-closed lease loss without exposing backend details."""

        logger.warning(
            "Refusing scheduler execution %s after lease renewal loss during %s",
            execution.id,
            phase,
        )

    async def _run_dispatch_while_lease_live(
        self,
        dispatch: PreparedScheduledDispatch,
        execution: SchedulerExecution,
        renewal_state: _LeaseRenewalState,
    ) -> tuple[bool, Any]:
        """Await an owned effect, cancelling it if its lease is lost.

        A target may be awaiting I/O when the renewal task discovers expiry or
        revocation.  Cancellation cannot undo an already-committed external
        effect, but it prevents further scheduler-owned work and, crucially,
        prevents this worker from finalizing as the stale owner.  Both owned
        tasks are joined so cancellation never leaves a context-bearing effect
        detached in the background.
        """

        async def invoke_dispatch() -> Any:
            return await dispatch()

        effect = asyncio.create_task(
            invoke_dispatch(), name=f"scheduler-effect:{execution.id}"
        )
        lease_lost = asyncio.create_task(
            renewal_state.lost.wait(), name=f"scheduler-lease-watch:{execution.id}"
        )
        try:
            await asyncio.wait(
                {effect, lease_lost}, return_when=asyncio.FIRST_COMPLETED
            )
            if renewal_state.lost.is_set():
                if not effect.done():
                    effect.cancel()
                outcome = await await_owned_task(effect)
                if outcome.cancellation is not None:
                    raise outcome.cancellation
                if outcome.error is not None and not isinstance(
                    outcome.error, asyncio.CancelledError
                ):
                    logger.warning(
                        "Scheduler effect %s failed while cancelling after lease loss: %s",
                        execution.id,
                        outcome.error,
                    )
                return False, None

            outcome = await await_owned_task(effect)
            if outcome.cancellation is not None:
                raise outcome.cancellation
            if outcome.error is not None:
                raise outcome.error
            # Give a renewal completion that raced the effect task precedence:
            # even a completed effect must not terminalize under a lost lease.
            if renewal_state.lost.is_set():
                return False, None
            return True, outcome.result
        finally:
            if not effect.done():
                effect.cancel()
                await await_owned_task(effect)
            if not lease_lost.done():
                lease_lost.cancel()
                await await_owned_task(lease_lost)

    async def _run_executor(self, execution: SchedulerExecution) -> Any:
        # ``AsyncMock`` and similar dynamic proxies fabricate any attribute on
        # ordinary callable executors.  Discover the structural hosted method
        # statically first so a legacy callable is not misclassified merely
        # because its test double happens to manufacture ``execute_scheduled``.
        try:
            inspect.getattr_static(self._executor, "execute_scheduled")
        except AttributeError:
            hosted = None
        else:
            hosted = getattr(self._executor, "execute_scheduled", None)
        # ``HostedExecutionExecutor`` is structural: integrations only need to
        # provide ``execute_scheduled``.  Requiring the implementation detail
        # marker from ``HostedSchedulerExecutor`` made otherwise valid custom
        # executors fall through to the legacy callable path and fail at
        # dispatch time.
        if callable(hosted):
            return await hosted(execution)
        return await self._executor(execution.task_name, execution.args)  # type: ignore[misc]

    async def _stop_renewal(
        self,
        renewal: asyncio.Task[None],
        task: ScheduledTask,
        execution: SchedulerExecution,
    ) -> None:
        """Cancel and join the owned renewal task despite caller cancellation."""

        renewal.cancel()
        outcome = await await_owned_task(renewal)
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if outcome.error is not None and not isinstance(
            outcome.error, asyncio.CancelledError
        ):
            logger.error(
                "Scheduler lease renewal failed for task %s execution %s; "
                "stale completion was refused",
                task.id,
                execution.id,
                exc_info=(
                    type(outcome.error),
                    outcome.error,
                    outcome.error.__traceback__,
                ),
            )

    async def _renew_lease(self, task: ScheduledTask) -> None:
        if not await self._agent_is_currently_authorized(task.agent_id):
            return
        # A one-second lease must renew around 0.33s, not at its expiry.
        # Keep the interval strictly inside the lease even if this becomes a
        # fractional configuration in a future release.
        interval = min(self._lease_seconds / 3, self._lease_seconds - 0.001)
        interval = max(0.001, interval)
        while True:
            await asyncio.sleep(interval)
            if not await self._renew_lease_once(task):
                logger.warning("Lost scheduler lease for task %s execution %s", task.id, task.claim_execution_id)
                return

    async def _renew_lease_once(self, task: ScheduledTask) -> bool:
        """Renew a token-guarded claim using statement-time database time."""

        if (
            not task.claim_token
            or not task.claim_execution_id
            or not await self._agent_is_currently_authorized(task.agent_id)
        ):
            return False
        if self._uses_database_clock():
            # Lock the exact token in one statement, then evaluate the live
            # predicate in a second statement. On PostgreSQL an UPDATE's
            # volatile WHERE clause may have been evaluated before waiting for
            # a row lock; evaluating the predicate after this transaction owns
            # the row prevents an expired token from being resurrected.
            async with self._transaction():
                if not await self._lock_claim_token_for_renewal(task):
                    return False
                expires, expiry_params = self._database_lease_expiry_sql()
                updated = await self._db.execute(
                    f"""
                    UPDATE scheduled_tasks SET lease_expires_at = {expires}
                    WHERE id = ? AND agent_id = ?
                      AND scheduler_protocol_version = ?
                      AND scheduler_rollout_fenced = 0
                      AND enabled = 0 AND scheduler_claim_fenced = 1
                      AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
                      AND {self._database_lease_live_sql()}
                    """,
                    (
                        *expiry_params,
                        task.id,
                        task.agent_id,
                        SCHEDULER_PROTOCOL_VERSION,
                        self._owner_id,
                        task.claim_token,
                        task.claim_execution_id,
                    ),
                )
                if not self._updated(updated):
                    return False
            # A successful write is not enough if commit/connection pressure
            # consumed its interval. Recheck the same token against a fresh DB
            # clock before reporting a claim as runnable.
            return await self._claim_token_is_live(task)

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        expires = (now + timedelta(seconds=self._lease_seconds)).isoformat()
        updated = await self._db.execute(
            """
            UPDATE scheduled_tasks SET lease_expires_at = ?
            WHERE id = ? AND agent_id = ?
              AND scheduler_protocol_version = ?
              AND scheduler_rollout_fenced = 0
              AND enabled = 0 AND scheduler_claim_fenced = 1
              AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
              AND lease_expires_at > ?
            """,
            (
                expires,
                task.id,
                task.agent_id,
                SCHEDULER_PROTOCOL_VERSION,
                self._owner_id,
                task.claim_token,
                task.claim_execution_id,
                now_iso,
            ),
        )
        return self._updated(updated)

    @staticmethod
    def _normalise_result(raw: Any, task: ScheduledTask) -> tuple[str, Optional[str], Optional[float], bool]:
        if isinstance(raw, ScheduledTaskOutcome):
            return (
                raw.status,
                f"{raw.result_text} Schedule id: {task.id}.",
                None,
                raw.pause_schedule,
            )
        if isinstance(raw, tuple) and len(raw) == 2:
            text = raw[0] if isinstance(raw[0], str) else (str(raw[0]) if raw[0] is not None else None)
            try:
                signal = None if raw[1] is None else max(0.0, min(1.0, float(raw[1])))
            except (TypeError, ValueError):
                logger.warning("Task %s returned non-numeric outcome signal %r; dropping", task.id, raw[1])
                signal = None
            return "success", text, signal, False
        return "success", raw if isinstance(raw, str) else (str(raw) if raw is not None else None), None, False

    async def _finalize(
        self,
        task: ScheduledTask,
        execution: SchedulerExecution,
        *,
        status: str,
        result_text: Optional[str],
        duration_ms: int,
        outcome_signal: Optional[float],
        ran: bool,
        pause_schedule: bool = False,
    ) -> None:
        if not await self._agent_is_currently_authorized(task.agent_id):
            logger.warning(
                "Refusing to finalize scheduler execution %s for unauthorized agent %s",
                execution.id,
                task.agent_id,
            )
            return
        # A paused/deleted task must win over an in-flight execution.  The
        # compare-and-set also rejects an old worker that lost its lease to a
        # recovery worker; it must never overwrite the newer worker's outcome.
        async with self._transaction():
            uses_database_clock = self._uses_database_clock()
            if uses_database_clock:
                # This locks the token first, then reads database time for cron
                # progression. The actual terminal UPDATE uses its own fresh
                # statement-time predicate below, so neither row-lock waits nor
                # host skew can make a stale worker publish a transition.
                if not await self._lock_live_claim_for_finalization(task, execution):
                    await self._mark_cancelled_if_no_longer_runnable(execution)
                    return
                schedule_now = await self._database_clock()
            else:
                schedule_now = datetime.now(timezone.utc)

            terminal = task.schedule_kind == SCHEDULE_ONE_SHOT or pause_schedule
            enabled = 0 if terminal else 1
            terminal_status = status if terminal else None
            next_at: Optional[str] = None
            if not terminal:
                try:
                    after = schedule_now
                    if task.misfire_policy == MISFIRE_CATCH_UP and task.next_run_at:
                        after = self._parse_utc(task.next_run_at) or schedule_now
                    next_at = next_run(
                        task.cron_expression,
                        after=after,
                        timezone_name=task.timezone_name,
                    ).isoformat()
                except CronParseError as e:
                    status = "failed"
                    result_text = (
                        f"{result_text or ''} scheduler cannot compute next run: {e}"
                    ).strip()
                    enabled = 0
                    terminal_status = "invalid_cron"

            # A recurring row now represents a *new* occurrence. Recovery of
            # the same occurrence increments its count in _claim; normal cron
            # progress must not carry that retry count into the next occurrence.
            next_attempt_count = 0 if next_at is not None else execution.attempt
            # PostgreSQL infers a boolean type for a ``CASE WHEN ?``
            # condition, whereas SQLite accepts its integer boolean
            # representation. Keep this as an actual Python bool so the shared
            # placeholder contract is valid for both backends.
            record_terminal_at = bool(terminal or terminal_status is not None)
            if uses_database_clock:
                now_sql = self._database_now_sql()
                last_run_sql = (
                    f"last_run_at = {now_sql}" if ran else "last_run_at = last_run_at"
                )
                updated = await self._db.execute(
                    f"""
                    UPDATE scheduled_tasks
                    SET {last_run_sql}, next_run_at = ?, enabled = ?,
                        attempt_count = ?, terminal_status = ?,
                        terminal_at = CASE WHEN ? THEN {now_sql} ELSE NULL END,
                        lease_owner = NULL, lease_expires_at = NULL, claim_token = NULL,
                        claim_execution_id = NULL, claim_scheduled_for = NULL,
                        scheduler_claim_fenced = 0
                    WHERE id = ? AND agent_id = ?
                      AND scheduler_protocol_version = ?
                      AND scheduler_rollout_fenced = 0
                      AND enabled = 0 AND scheduler_claim_fenced = 1
                      AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
                      AND {self._database_lease_live_sql()}
                    """,
                    (
                        next_at,
                        enabled,
                        next_attempt_count,
                        terminal_status,
                        record_terminal_at,
                        task.id,
                        task.agent_id,
                        SCHEDULER_PROTOCOL_VERSION,
                        self._owner_id,
                        task.claim_token,
                        execution.id,
                    ),
                )
                if not self._updated(updated):
                    await self._mark_cancelled_if_no_longer_runnable(execution)
                    return
                updated_log = await self._db.execute(
                    f"""
                    UPDATE task_execution_log
                    SET status = ?, result_text = ?, duration_ms = ?,
                        executed_at = {now_sql}, outcome_signal = ?,
                        attempt_count = ?, completed_at = {now_sql}
                    WHERE id = ? AND task_id = ? AND agent_id = ? AND status = 'claimed'
                    """,
                    (
                        status, result_text, duration_ms, outcome_signal,
                        execution.attempt, execution.id, task.id, task.agent_id,
                    ),
                )
            else:
                # Preserve the adapter path for intentionally minimal unit DB
                # doubles. Real SQLite and PostgreSQL always use the branch
                # above and therefore never publish host-clock lease state.
                now_iso = datetime.now(timezone.utc).isoformat()
                terminal_at = now_iso if record_terminal_at else None
                last_run_sql = "last_run_at = ?" if ran else "last_run_at = last_run_at"
                params: list[Any] = []
                if ran:
                    params.append(now_iso)
                params.extend([
                    next_at, enabled, next_attempt_count, terminal_status, terminal_at,
                    task.id, task.agent_id, SCHEDULER_PROTOCOL_VERSION,
                    self._owner_id, task.claim_token, execution.id, now_iso,
                ])
                updated = await self._db.execute(
                    f"""
                    UPDATE scheduled_tasks
                    SET {last_run_sql}, next_run_at = ?, enabled = ?,
                        attempt_count = ?, terminal_status = ?, terminal_at = ?,
                        lease_owner = NULL, lease_expires_at = NULL, claim_token = NULL,
                        claim_execution_id = NULL, claim_scheduled_for = NULL,
                        scheduler_claim_fenced = 0
                    WHERE id = ? AND agent_id = ?
                      AND scheduler_protocol_version = ?
                      AND scheduler_rollout_fenced = 0
                      AND enabled = 0 AND scheduler_claim_fenced = 1
                      AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
                      AND lease_expires_at > ?
                    """,
                    tuple(params),
                )
                if not self._updated(updated):
                    await self._mark_cancelled_if_no_longer_runnable(execution)
                    return
                updated_log = await self._db.execute(
                    """
                    UPDATE task_execution_log
                    SET status = ?, result_text = ?, duration_ms = ?, executed_at = ?,
                        outcome_signal = ?, attempt_count = ?, completed_at = ?
                    WHERE id = ? AND task_id = ? AND agent_id = ? AND status = 'claimed'
                    """,
                    (
                        status, result_text, duration_ms, now_iso, outcome_signal,
                        execution.attempt, now_iso, execution.id, task.id, task.agent_id,
                    ),
                )
            if not self._updated(updated_log):
                raise RuntimeError(
                    f"claimed execution log {execution.id} disappeared before finalization"
                )

        logger.info(
            "Scheduled task %s (%s) finalized %s (%dms, attempt=%d)",
            task.id, task.task_name, status, duration_ms, execution.attempt,
        )

    async def _mark_cancelled_if_no_longer_runnable(self, execution: SchedulerExecution) -> None:
        if not await self._agent_is_currently_authorized(execution.agent_id):
            return
        row = await self._db.fetchone(
            """
            SELECT enabled, scheduler_claim_fenced
            FROM scheduled_tasks WHERE id = ? AND agent_id = ?
            """,
            (execution.schedule_id, execution.agent_id),
        )
        # A recovery owner keeps the row enabled=0 while its compatibility
        # claim fence is active. That is still runnable work, not an operator
        # pause, so an old worker must not terminalize the shared execution log.
        if row is not None and (bool(row[0]) or bool(row[1])):
            return
        completed_at = (
            self._database_now_sql()
            if self._uses_database_clock()
            else "?"
        )
        await self._db.execute(
            f"""
            UPDATE task_execution_log
            SET status = 'cancelled', result_text = 'schedule removed or paused while execution was in flight',
                completed_at = {completed_at}
            WHERE id = ? AND agent_id = ? AND status = 'claimed'
            """,
            (
                (execution.id, execution.agent_id)
                if self._uses_database_clock()
                else (
                    datetime.now(timezone.utc).isoformat(),
                    execution.id,
                    execution.agent_id,
                )
            ),
        )

    @staticmethod
    def _parse_utc(value: str) -> Optional[datetime]:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    @staticmethod
    def _seconds_late(next_run_at_iso: Optional[str], now: datetime) -> float:
        scheduled = SchedulerRunner._parse_utc(next_run_at_iso) if next_run_at_iso else None
        return max(0.0, (now - scheduled).total_seconds()) if scheduled else 0.0

    async def _acquire_scheduler_schema_lock(self) -> None:
        """Serialize global scheduler bootstrap on a shared PostgreSQL DB."""

        if self._database_backend_type() == "postgres":
            # This lock deliberately precedes *all* global-table/provenance
            # reads. PostgreSQL transactions on separate replicas otherwise
            # both see an absent marker: one creates a fresh table while the
            # other mistakes that just-created table for a legacy upgrade.
            await self._db.fetchval(
                "SELECT pg_advisory_xact_lock(?, ?)",
                # This scope intentionally does not encode the local protocol
                # version. A v2 process must serialize its compatibility
                # preflight with a future binary before either can mutate the
                # shared bootstrap/provenance state.
                (2715, _SCHEDULER_BOOTSTRAP_ADVISORY_LOCK_SCOPE),
            )

    @staticmethod
    def _is_newer_protocol_version(value: Any) -> bool:
        """Whether a persisted integer protocol version is newer than ours."""

        # The durable columns are INTEGER. Keep an exact integer conversion so
        # hand-edited text values such as ``"3"`` also fail closed, without
        # treating malformed legacy data as an invented future protocol.
        try:
            return int(value) > SCHEDULER_PROTOCOL_VERSION
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _future_protocol_version_sql(column: str) -> tuple[str, tuple[Any, ...]]:
        """Return a portable, bounded predicate for future integer versions.

        Scheduler columns are INTEGER in the managed schema, but legacy or
        hand-created relations can expose text values.  Avoid backend-specific
        integer casts (PostgreSQL rejects malformed text while SQLite coerces
        it) by comparing a canonical decimal string.  This preserves the
        fail-closed future check for values such as ``"3"`` and ``"+003"``
        while leaving malformed/non-integer values outside the future-protocol
        authority boundary, matching :meth:`_is_newer_protocol_version`.
        """

        if column not in {"scheduler_protocol_version", "protocol_version"}:
            raise ValueError("unsupported scheduler protocol version column")
        text = f"trim(CAST({column} AS TEXT))"
        unsigned = (
            f"(CASE WHEN substr({text}, 1, 1) = '+' "
            f"THEN substr({text}, 2) ELSE {text} END)"
        )
        non_digits = unsigned
        for digit in "0123456789":
            non_digits = f"replace({non_digits}, '{digit}', '')"
        normalized = f"COALESCE(NULLIF(ltrim({unsigned}, '0'), ''), '0')"
        current = str(SCHEDULER_PROTOCOL_VERSION)
        current_length = len(current)
        return (
            f"""
            {column} IS NOT NULL
            AND {unsigned} <> ''
            AND {non_digits} = ''
            AND (
                length({normalized}) > ?
                OR (length({normalized}) = ? AND {normalized} > ?)
            )
            """,
            (current_length, current_length, current),
        )

    async def _scheduler_table_exists(self, table: str) -> bool:
        """Read a scheduler table's existence without mutating the database."""

        table_exists = getattr(self._db, "table_exists", None)
        if not callable(table_exists):
            # Lightweight unit doubles predate schema introspection. They are
            # not persistent upgrade sources, so retain their no-table view.
            return False
        result = table_exists(table)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def _reject_newer_scheduler_protocol_state(self) -> None:
        """Fail before changing state written by a newer scheduler binary."""

        if await self._scheduled_tasks_protocol_column_exists():
            future, params = self._future_protocol_version_sql(
                "scheduler_protocol_version"
            )
            row = await self._db.fetchone(
                f"SELECT 1 FROM scheduled_tasks WHERE {future} LIMIT 1",
                params,
            )
            if row is not None:
                raise SchedulerProtocolVersionIncompatible()

        if await self._scheduler_table_exists("scheduler_protocol_schema"):
            row = await self._db.fetchone(
                """
                SELECT protocol_version FROM scheduler_protocol_schema
                WHERE singleton = 1
                """
            )
            if row is not None and row and self._is_newer_protocol_version(row[0]):
                raise SchedulerProtocolVersionIncompatible()

        if await self._scheduler_table_exists("scheduler_protocol_rollout"):
            future, params = self._future_protocol_version_sql("protocol_version")
            row = await self._db.fetchone(
                f"SELECT 1 FROM scheduler_protocol_rollout WHERE {future} LIMIT 1",
                params,
            )
            if row is not None:
                raise SchedulerProtocolVersionIncompatible()

    async def _scheduled_tasks_protocol_column_exists(self) -> bool:
        """Return whether an existing schedule table exposes the v2 marker.

        This read-only shape probe is required before bootstrap DDL. A legacy
        ``scheduled_tasks`` relation lacks the column and remains eligible for
        the normal controlled rollout; a future row with the column is an
        authority boundary that this binary must not normalize.
        """

        if not await self._scheduled_tasks_table_exists():
            return False
        if self._database_backend_type() == "postgres":
            row = await self._db.fetchone(
                """
                SELECT 1 FROM pg_attribute
                WHERE attrelid = to_regclass('scheduled_tasks')
                  AND attname = 'scheduler_protocol_version'
                  AND attnum > 0 AND NOT attisdropped
                LIMIT 1
                """
            )
        else:
            row = await self._db.fetchone(
                """
                SELECT 1 FROM pragma_table_info('scheduled_tasks')
                WHERE name = ?
                LIMIT 1
                """,
                ("scheduler_protocol_version",),
            )
        return bool(row)

    async def _establish_scheduler_schema_provenance(self) -> bool:
        """Return whether ``scheduled_tasks`` predated this v2 bootstrap.

        The singleton is database-global, not DID-local. It distinguishes a
        genuinely fresh v2 fleet (where later configured DIDs are safe to seed
        active) from a database whose schedule table could have been created
        by an origin/main binary. ``_ensure_tables`` invokes this only while
        the backend-wide bootstrap boundary owns the transaction and lock.
        """

        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduler_protocol_schema (
                singleton INTEGER PRIMARY KEY,
                provenance TEXT NOT NULL,
                protocol_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        existing = await self._db.fetchone(
            """
            SELECT provenance, protocol_version FROM scheduler_protocol_schema
            WHERE singleton = 1
            """
        )
        if existing is not None:
            if len(existing) > 1 and self._is_newer_protocol_version(existing[1]):
                raise SchedulerProtocolVersionIncompatible()
            provenance = existing[0] if len(existing) else None
            if provenance not in {
                SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2,
                SCHEDULER_SCHEMA_PROVENANCE_LEGACY_UNKNOWN,
            }:
                # Unknown provenance is never silently treated as fresh.
                provenance = SCHEDULER_SCHEMA_PROVENANCE_LEGACY_UNKNOWN
            self._schema_provenance = provenance
            return provenance == SCHEDULER_SCHEMA_PROVENANCE_LEGACY_UNKNOWN

        preexisting_schedule_table = await self._scheduled_tasks_table_exists()
        provenance = (
            SCHEDULER_SCHEMA_PROVENANCE_LEGACY_UNKNOWN
            if preexisting_schedule_table
            else SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2
        )
        await self._db.execute(
            """
            INSERT INTO scheduler_protocol_schema
                (singleton, provenance, protocol_version, created_at)
            VALUES (1, ?, ?, ?)
            """,
            (
                provenance,
                SCHEDULER_PROTOCOL_VERSION,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._schema_provenance = provenance
        return preexisting_schedule_table

    async def _ensure_tables(self):
        """Create/migrate schema under one durable bootstrap boundary."""

        blocked: list[tuple[str, str]] = []
        async with self._bootstrap_serialization_boundary():
            # This must remain the first database operation in the boundary:
            # do not create, alter, normalize, or rotate a newer protocol's
            # state just to discover that this binary cannot safely own it.
            await self._reject_newer_scheduler_protocol_state()
            blocked = await self._ensure_tables_mutations()

        # A quiescing fence is useful only if it commits. Raising inside the
        # outer transaction would roll it back and reopen legacy execution, so
        # construct and raise the operator-facing error after the boundary.
        if blocked:
            raise self._rollout_quiescence_error(blocked)
        self._protocol_ready = True

    async def prepare_tenant_registration(
        self,
    ) -> SchedulerTenantProtocolRegistration:
        """Durably activate one dynamic DID and create rollback ownership.

        The short-lived runner used for this operation must have one fixed DID.
        Holding the same backend-global bootstrap boundary as ordinary startup
        keeps runtime tenant activation ordered with schema/protocol changes.
        Rows later seeded by the pending agent carry ``registration_nonce``;
        rollback never infers ownership from a point-in-time DID snapshot.
        """

        if len(self._authorized_agent_ids) != 1:
            raise ValueError(
                "dynamic scheduler tenant registration requires exactly one DID"
            )
        agent_id = self._authorized_agent_ids[0]
        registration_nonce = secrets.token_urlsafe(24)
        blocked: list[tuple[str, str]] = []
        async with self._bootstrap_serialization_boundary():
            await self._reject_newer_scheduler_protocol_state()
            rollout_preexisting = False
            if await self._scheduler_table_exists("scheduler_protocol_rollout"):
                rollout_preexisting = (
                    await self._db.fetchone(
                        """
                        SELECT 1 FROM scheduler_protocol_rollout
                        WHERE agent_id = ?
                        """,
                        (agent_id,),
                    )
                    is not None
                )
            blocked = await self._ensure_tables_mutations()
            if not blocked:
                if rollout_preexisting:
                    # A second host registering the same DID can use the
                    # active control row even before it has written a
                    # schedule. Its prepare adopts that row, so the first
                    # host's later rollback cannot delete shared state.
                    await self._db.execute(
                        """
                        UPDATE scheduler_protocol_rollout
                        SET scheduler_registration_nonce = NULL
                        WHERE agent_id = ?
                        """,
                        (agent_id,),
                    )
                else:
                    # The bootstrap boundary made the absent-row observation
                    # and active-row creation atomic. Stamp only that exact
                    # fresh active row; a quiescing legacy transition never
                    # reaches a returned registration.
                    await self._db.execute(
                        """
                        UPDATE scheduler_protocol_rollout
                        SET scheduler_registration_nonce = ?
                        WHERE agent_id = ? AND protocol_version = ?
                          AND state = ? AND activation_nonce IS NULL
                        """,
                        (
                            registration_nonce,
                            agent_id,
                            SCHEDULER_PROTOCOL_VERSION,
                            SCHEDULER_ROLLOUT_STATE_ACTIVE,
                        ),
                    )

        if blocked:
            raise self._rollout_quiescence_error(blocked)
        self._protocol_ready = True
        return SchedulerTenantProtocolRegistration(
            agent_id=agent_id,
            rollout_preexisting=rollout_preexisting,
            registration_nonce=registration_nonce,
        )

    async def rollback_tenant_registration(
        self,
        registration: SchedulerTenantProtocolRegistration,
    ) -> None:
        """Remove only scheduler state created by a failed dynamic onboarding.

        A different host replica can begin scheduling this DID after prepare
        releases the bootstrap boundary.  Its rows did not exist at prepare
        time, but they are never evidence that this registration owns them.
        Delete only rows stamped by this registration's unpredictable nonce.
        """

        if (
            len(self._authorized_agent_ids) != 1
            or registration.agent_id != self._authorized_agent_ids[0]
        ):
            raise ValueError("scheduler tenant rollback scope does not match runner")
        agent_id = registration.agent_id
        async with self._bootstrap_serialization_boundary():
            await self._reject_newer_scheduler_protocol_state()
            await self._delete_registration_owned_rows(registration)
            if not registration.rollout_preexisting:
                await self._db.execute(
                    """
                    DELETE FROM scheduler_protocol_rollout
                    WHERE agent_id = ? AND protocol_version = ?
                      AND state = ? AND activation_nonce IS NULL
                      AND scheduler_registration_nonce = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM scheduled_tasks WHERE agent_id = ?
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM task_execution_log WHERE agent_id = ?
                      )
                    """,
                    (
                        agent_id,
                        SCHEDULER_PROTOCOL_VERSION,
                        SCHEDULER_ROLLOUT_STATE_ACTIVE,
                        registration.registration_nonce,
                        agent_id,
                        agent_id,
                    ),
                )

    async def _delete_registration_owned_rows(
        self,
        registration: SchedulerTenantProtocolRegistration,
    ) -> None:
        """Delete logs only for schedules deleted under this registration nonce.

        A pending registration's schedule can be adopted by another replica
        between statements.  PostgreSQL must therefore derive both deletes
        from one data-modifying CTE: a prior read of IDs can become stale
        after the adopter clears the nonce and publishes its claim log.
        SQLite lacks data-modifying CTEs, but the enclosing bootstrap
        transaction holds its exclusive per-DID rollout gate.  Its
        ``DELETE ... RETURNING`` consequently records exactly the rows this
        rollback removed before their logs are deleted in that same
        transaction.
        """

        agent_id = registration.agent_id
        backend_type = self._database_backend_type()
        if backend_type == "postgres":
            await self._db.execute(
                """
                WITH deleted_schedules AS (
                    DELETE FROM scheduled_tasks
                    WHERE agent_id = ? AND scheduler_registration_nonce = ?
                    RETURNING id
                )
                DELETE FROM task_execution_log AS execution_log
                USING deleted_schedules
                WHERE execution_log.agent_id = ?
                  AND execution_log.task_id = deleted_schedules.id
                """,
                (agent_id, registration.registration_nonce, agent_id),
            )
            return

        if backend_type == "sqlite":
            deleted_schedule_ids = tuple(
                str(row[0])
                for row in await self._db.fetchall(
                    """
                    DELETE FROM scheduled_tasks
                    WHERE agent_id = ? AND scheduler_registration_nonce = ?
                    RETURNING id
                    """,
                    (agent_id, registration.registration_nonce),
                )
            )
            if not deleted_schedule_ids:
                return
            placeholders = ", ".join("?" for _ in deleted_schedule_ids)
            await self._db.execute(
                f"""
                DELETE FROM task_execution_log
                WHERE agent_id = ? AND task_id IN ({placeholders})
                """,
                (agent_id, *deleted_schedule_ids),
            )
            return

        raise RuntimeError("scheduler registration rollback requires PostgreSQL or SQLite")

    async def _ensure_tables_mutations(self) -> list[tuple[str, str]]:
        """Apply bootstrap mutations while the caller owns the global gate."""
        preexisting_schedule_table = await self._establish_scheduler_schema_provenance()
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                task_name TEXT NOT NULL,
                cron_expression TEXT NOT NULL,
                args_json TEXT DEFAULT '{}',
                enabled INTEGER DEFAULT 1,
                last_run_at TEXT,
                next_run_at TEXT,
                created_at TEXT NOT NULL,
                schedule_kind TEXT NOT NULL DEFAULT 'cron',
                run_at TEXT,
                timezone_name TEXT NOT NULL DEFAULT 'UTC',
                misfire_policy TEXT NOT NULL DEFAULT 'skip',
                misfire_grace_seconds INTEGER,
                idempotency_key TEXT,
                lease_owner TEXT,
                lease_expires_at TEXT,
                claim_token TEXT,
                claim_execution_id TEXT,
                claim_scheduled_for TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                terminal_status TEXT,
                terminal_at TEXT,
                -- Deliberately no v2 default: an origin/main writer that
                -- omits this unknown column remains provenance-visible as
                -- NULL, never looking like a v2 writer during a mixed rollout.
                scheduler_protocol_version INTEGER,
                scheduler_rollout_fenced INTEGER NOT NULL DEFAULT 0,
                scheduler_rollout_nonce TEXT,
                scheduler_rollout_snapshot TEXT,
                -- Set only while dynamic host onboarding is pending.  It
                -- lets rollback target its own seeded schedules without
                -- claiming rows another replica created for the same DID.
                scheduler_registration_nonce TEXT,
                scheduler_claim_fenced INTEGER NOT NULL DEFAULT 0,
                scheduler_rollout_fenced_at TEXT
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS task_execution_log (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                status TEXT NOT NULL,
                result_text TEXT,
                duration_ms INTEGER NOT NULL,
                executed_at TEXT NOT NULL,
                outcome_signal REAL,
                occurrence_at TEXT,
                idempotency_key TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 1,
                claimed_at TEXT,
                completed_at TEXT
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduler_protocol_rollout (
                agent_id TEXT PRIMARY KEY,
                protocol_version INTEGER NOT NULL,
                state TEXT NOT NULL,
                activation_nonce TEXT,
                scheduler_registration_nonce TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        await self._add_column_if_missing(
            "scheduler_protocol_rollout",
            "scheduler_registration_nonce",
            "TEXT",
        )
        await self._lock_active_rollout_controls_for_bootstrap()
        scheduled_columns = {
            "schedule_kind": "TEXT NOT NULL DEFAULT 'cron'",
            "run_at": "TEXT",
            "timezone_name": "TEXT NOT NULL DEFAULT 'UTC'",
            "misfire_policy": "TEXT NOT NULL DEFAULT 'skip'",
            "misfire_grace_seconds": "INTEGER",
            "idempotency_key": "TEXT",
            "lease_owner": "TEXT",
            "lease_expires_at": "TEXT",
            "claim_token": "TEXT",
            "claim_execution_id": "TEXT",
            "claim_scheduled_for": "TEXT",
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
            "terminal_status": "TEXT",
            "terminal_at": "TEXT",
            # Nullable on an additive migration is intentional: NULL proves a
            # row could have been written by a legacy, pre-claim runner. Every
            # v2 writer explicitly supplies the marker; there is deliberately
            # no schema default that could hide an old writer.
            "scheduler_protocol_version": "INTEGER",
            "scheduler_rollout_fenced": "INTEGER NOT NULL DEFAULT 0",
            "scheduler_rollout_nonce": "TEXT",
            "scheduler_rollout_snapshot": "TEXT",
            "scheduler_registration_nonce": "TEXT",
            "scheduler_claim_fenced": "INTEGER NOT NULL DEFAULT 0",
            "scheduler_rollout_fenced_at": "TEXT",
        }
        log_columns = {
            "outcome_signal": "REAL",
            "occurrence_at": "TEXT",
            "idempotency_key": "TEXT",
            "attempt_count": "INTEGER NOT NULL DEFAULT 1",
            "claimed_at": "TEXT",
            "completed_at": "TEXT",
        }
        for table, columns in (("scheduled_tasks", scheduled_columns), ("task_execution_log", log_columns)):
            for column, definition in columns.items():
                await self._add_column_if_missing(table, column, definition)
        # Existing UTC cron schedules retain their old behavior, now made
        # explicit and suitable for occurrence-level idempotency.
        if self._authorized_agent_ids:
            authorization_scope = self._authorized_agent_placeholders(
                self._authorized_agent_ids
            )
            authorization_params = tuple(self._authorized_agent_ids)
            await self._db.execute(
                f"""
                UPDATE scheduled_tasks SET schedule_kind = 'cron'
                WHERE schedule_kind IS NULL AND agent_id IN ({authorization_scope})
                """,
                authorization_params,
            )
            await self._db.execute(
                f"""
                UPDATE scheduled_tasks SET timezone_name = 'UTC'
                WHERE (timezone_name IS NULL OR timezone_name = '')
                  AND agent_id IN ({authorization_scope})
                """,
                authorization_params,
            )
            await self._db.execute(
                f"""
                UPDATE scheduled_tasks SET misfire_policy = 'skip'
                WHERE (misfire_policy IS NULL OR misfire_policy = '')
                  AND agent_id IN ({authorization_scope})
                """,
                authorization_params,
            )
            await self._backfill_legacy_idempotency_keys()
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_agent_next ON scheduled_tasks(agent_id, enabled, next_run_at)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_due_claim ON scheduled_tasks(enabled, next_run_at, lease_expires_at)")
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_protocol_claim
            ON scheduled_tasks(agent_id, scheduler_protocol_version,
                               scheduler_rollout_fenced,
                               scheduler_claim_fenced, next_run_at)
            """
        )
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_task_execution_log_task ON task_execution_log(task_id, executed_at DESC)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_task_execution_log_idempotency ON task_execution_log(agent_id, idempotency_key)")
        return await self._ensure_protocol_rollout(
            preexisting_schedule_table=preexisting_schedule_table,
            rollout_gates_held=True,
            defer_quiescence_error=True,
        )

    async def _lock_active_rollout_controls_for_bootstrap(self) -> None:
        """Drain admitted PostgreSQL effects before taking schema DDL locks.

        The enclosing bootstrap boundary has already acquired every
        fixed-scope PostgreSQL effect gate, so no newly admitted v2 effect can
        enter while DDL is pending and any earlier one has completed. Lock the
        current active control rows as an additional compatibility drain for a
        pre-gate v2 process sharing the database. SQLite has already acquired
        every DID advisory gate in the enclosing boundary, so it needs no
        extra database write here.
        """

        if self._database_backend_type() != "postgres":
            return
        for agent_id in self._authorized_agent_ids:
            await self._db.execute(
                """
                UPDATE scheduler_protocol_rollout
                SET updated_at = updated_at, scheduler_registration_nonce = NULL
                WHERE agent_id = ? AND protocol_version = ? AND state = 'active'
                """,
                (agent_id, SCHEDULER_PROTOCOL_VERSION),
            )

    async def _scheduled_tasks_table_exists(self) -> bool:
        """Return whether this runner is upgrading an existing schedule table."""

        return await self._scheduler_table_exists("scheduled_tasks")

    async def _backfill_legacy_idempotency_keys(self) -> None:
        """Give keyless legacy rows a bounded, stable generated base key."""

        authorization_scope = self._authorized_agent_placeholders(
            self._authorized_agent_ids
        )
        rows = await self._db.fetchall(
            f"""
            SELECT id, agent_id FROM scheduled_tasks
            WHERE agent_id IN ({authorization_scope})
              AND (idempotency_key IS NULL OR idempotency_key = '')
            """,
            tuple(self._authorized_agent_ids),
        )
        for schedule_id, agent_id in rows:
            base = self._legacy_base_idempotency_key(str(schedule_id))
            await self._db.execute(
                """
                UPDATE scheduled_tasks SET idempotency_key = ?
                WHERE id = ? AND agent_id = ?
                  AND (idempotency_key IS NULL OR idempotency_key = '')
                """,
                (base, schedule_id, agent_id),
            )

    def _rollout_acknowledges(self, nonce: Optional[str]) -> bool:
        """Whether the process presents the exact one-time durable nonce."""

        if not nonce:
            return False
        return any(
            hmac.compare_digest(candidate, nonce)
            for candidate in self._rollout_acknowledgements
        )

    @staticmethod
    def _rollout_snapshot(values: tuple[Any, ...]) -> str:
        """Fingerprint the fields an origin/main scheduler can mutate.

        The rollout nonce alone distinguishes a freshly inserted legacy row,
        but an old binary can also update an *existing* row without knowing
        about our new columns.  A durable snapshot makes that write observable
        before acknowledgement: re-enabling or rescheduling a fenced row
        invalidates the old nonce instead of being silently adopted.
        """

        payload = json.dumps(
            list(values), ensure_ascii=False, separators=(",", ":"), default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _rollout_rows_need_refence(
        self, agent_id: str, nonce: str
    ) -> bool:
        """Whether a quiescing DID gained or changed a legacy-visible row."""

        rows = await self._db.fetchall(
            """
            SELECT id, agent_id, task_name, cron_expression, args_json,
                   enabled, last_run_at, next_run_at, created_at,
                   scheduler_rollout_nonce, scheduler_rollout_snapshot
            FROM scheduled_tasks WHERE agent_id = ?
            """,
            (agent_id,),
        )
        for row in rows:
            # A row not stamped by this exact fence is an insertion after the
            # baseline (or a partially migrated corrupt row), never safe to
            # activate under an acknowledgement for the earlier epoch.
            if row[9] != nonce:
                return True
            recorded = row[10]
            if not isinstance(recorded, str) or not recorded:
                return True
            if not hmac.compare_digest(self._rollout_snapshot(tuple(row[:9])), recorded):
                return True
        return False

    async def _scheduler_schema_is_fresh_v2(
        self, *, preexisting_schedule_table: bool
    ) -> bool:
        """Return durable fresh-v2 provenance, failing closed when absent."""

        if self._schema_provenance is not None:
            return self._schema_provenance == SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2
        try:
            row = await self._db.fetchone(
                """
                SELECT provenance, protocol_version FROM scheduler_protocol_schema
                WHERE singleton = 1
                """
            )
        except Exception:
            # Lightweight pre-schema doubles retain their historical fresh
            # bootstrap behavior; real startup always establishes the global
            # provenance first and treats any failure as a safety outage.
            return not preexisting_schedule_table
        if row is not None and len(row) > 1 and self._is_newer_protocol_version(row[1]):
            raise SchedulerProtocolVersionIncompatible()
        provenance = row[0] if row is not None and len(row) else None
        self._schema_provenance = (
            provenance
            if provenance in {
                SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2,
                SCHEDULER_SCHEMA_PROVENANCE_LEGACY_UNKNOWN,
            }
            else SCHEDULER_SCHEMA_PROVENANCE_LEGACY_UNKNOWN
        )
        return self._schema_provenance == SCHEDULER_SCHEMA_PROVENANCE_FRESH_V2

    async def _ensure_protocol_rollout(
        self,
        *,
        preexisting_schedule_table: bool,
        rollout_gates_held: bool = False,
        defer_quiescence_error: bool = False,
    ) -> list[tuple[str, str]]:
        """Fence legacy rows until every old scheduler process is drained.

        State is per agent DID so a host can never re-enable another fleet's
        schedules while activating its own. A NULL/unknown row is never silently
        adopted: it returns the owning DID to ``quiescing`` and makes start fail
        until the stored nonce is explicitly acknowledged.
        """

        # ``_tick`` calls this independently of bootstrap. Keep the version
        # preflight here as well so a later newer-protocol writer cannot be
        # normalized into this runner's version by a periodic reconciliation.
        await self._reject_newer_scheduler_protocol_state()
        blocked: list[tuple[str, str]] = []
        fresh_v2_schema = await self._scheduler_schema_is_fresh_v2(
            preexisting_schedule_table=preexisting_schedule_table
        )
        for agent_id in await self._current_authorized_agent_ids():
            # Bootstrap already owns every gate in stable order, so it enters
            # the mutating reconciliation directly. Steady state first takes
            # a read-only probe: an active v2 row is overwhelmingly normal and
            # must not wait behind an unrelated long-lived effect merely to
            # discover it needs no transition. When the probe finds a possible
            # mutation, take the exclusive gate and re-read inside
            # ``_ensure_protocol_rollout_agent`` before fencing/activating.
            # That closes new effect admission and drains existing effects
            # without allowing the read/probe race to cross a transition.
            if rollout_gates_held:
                blocked_nonce = await self._ensure_protocol_rollout_agent(
                    agent_id, fresh_v2_schema=fresh_v2_schema
                )
            else:
                transition_needed, blocked_nonce = (
                    await self._protocol_rollout_transition_needed(agent_id)
                )
                if transition_needed:
                    if self._database_backend_type() == "postgres":
                        async with self._postgres_rollout_transition_gate(agent_id):
                            blocked_nonce = await self._ensure_protocol_rollout_agent(
                                agent_id, fresh_v2_schema=fresh_v2_schema
                            )
                    else:
                        async with self._sqlite_rollout_gate(agent_id):
                            blocked_nonce = await self._ensure_protocol_rollout_agent(
                                agent_id, fresh_v2_schema=fresh_v2_schema
                            )
            if blocked_nonce is not None:
                blocked.append((agent_id, blocked_nonce))

        if blocked and not defer_quiescence_error:
            raise self._rollout_quiescence_error(blocked)
        return blocked

    @staticmethod
    def _rollout_quiescence_error(
        blocked: Collection[tuple[str, str]],
    ) -> SchedulerRolloutQuiescenceRequired:
        """Build the drain instruction only after its fences are durable."""

        nonces = ", ".join(nonce for _, nonce in blocked)
        agents = ", ".join(agent_id for agent_id, _ in blocked)
        return SchedulerRolloutQuiescenceRequired(
            "Scheduler protocol v2 is quiescing for agent DID(s) "
            f"{agents}. Stop and drain every legacy scheduler replica and "
            "all scheduler mutations, then restart with "
            f"{SCHEDULER_ROLLOUT_ACK_ENV} containing this one-time nonce "
            f"(comma-separated when multiple DIDs are fenced): {nonces}"
        )

    async def _protocol_rollout_transition_needed(
        self, agent_id: str
    ) -> tuple[bool, Optional[str]]:
        """Read whether one DID needs an exclusive rollout transition.

        The result is advisory: callers must re-read under the exclusive gate
        before changing rollout state.  Its purpose is solely to keep ordinary
        active-v2 polling from serializing behind every admitted effect.  A
        quiescing DID with an unacknowledged stable nonce is already fenced and
        returns that nonce without taking a writer gate.
        """

        state_row = await self._db.fetchone(
            """
            SELECT protocol_version, state, activation_nonce
            FROM scheduler_protocol_rollout WHERE agent_id = ?
            """,
            (agent_id,),
        )
        if (
            state_row is not None
            and state_row
            and self._is_newer_protocol_version(state_row[0])
        ):
            raise SchedulerProtocolVersionIncompatible()

        rollout_nonce = (
            state_row[2]
            if state_row is not None
            and len(state_row) > 2
            and state_row[1] == SCHEDULER_ROLLOUT_STATE_QUIESCING
            else None
        )
        if rollout_nonce is None:
            has_unknown_rows = await self._db.fetchone(
                """
                SELECT 1 FROM scheduled_tasks
                WHERE agent_id = ?
                  AND (scheduler_protocol_version IS NULL
                       OR scheduler_protocol_version <> ?)
                LIMIT 1
                """,
                (agent_id, SCHEDULER_PROTOCOL_VERSION),
            )
            needs_refence = has_unknown_rows is not None
        else:
            needs_refence = await self._rollout_rows_need_refence(
                agent_id, rollout_nonce
            )

        if state_row is None:
            # Even a fresh-v2 DID needs its first active control row created.
            return True, None

        protocol_version, state, nonce = state_row
        state_is_known = state in {
            SCHEDULER_ROLLOUT_STATE_ACTIVE,
            SCHEDULER_ROLLOUT_STATE_QUIESCING,
        }
        requires_new_nonce = (
            protocol_version != SCHEDULER_PROTOCOL_VERSION
            or not state_is_known
            or needs_refence
            or (state == SCHEDULER_ROLLOUT_STATE_QUIESCING and not nonce)
        )
        if state == SCHEDULER_ROLLOUT_STATE_QUIESCING:
            if requires_new_nonce or self._rollout_acknowledges(nonce):
                return True, None
            return False, nonce or "<missing nonce>"
        return requires_new_nonce, None

    async def _ensure_protocol_rollout_agent(
        self, agent_id: str, *, fresh_v2_schema: bool
    ) -> Optional[str]:
        """Reconcile one DID's rollout row and return a required ACK nonce."""

        async with self._transaction():
            # Conditional state writes make PostgreSQL's read-committed
            # transactions safe when two new replicas observe an active row
            # concurrently. A loser rereads the winner's nonce rather than
            # returning an acknowledgement another replica can overwrite.
            for _ in range(4):
                state_row = await self._db.fetchone(
                    """
                    SELECT protocol_version, state, activation_nonce
                    FROM scheduler_protocol_rollout WHERE agent_id = ?
                    """,
                    (agent_id,),
                )
                if (
                    state_row is not None
                    and state_row
                    and self._is_newer_protocol_version(state_row[0])
                ):
                    # Do not rotate a future control row down to v2. The
                    # database state is an authority boundary, not an input
                    # this older process may repair or reinterpret.
                    raise SchedulerProtocolVersionIncompatible()
                rollout_nonce = (
                    state_row[2]
                    if state_row is not None
                    and state_row[1] == SCHEDULER_ROLLOUT_STATE_QUIESCING
                    else None
                )
                if rollout_nonce is None:
                    has_unknown_rows = await self._db.fetchone(
                        """
                        SELECT 1 FROM scheduled_tasks
                        WHERE agent_id = ?
                          AND (scheduler_protocol_version IS NULL
                               OR scheduler_protocol_version <> ?)
                        LIMIT 1
                        """,
                        (agent_id, SCHEDULER_PROTOCOL_VERSION),
                    )
                else:
                    has_unknown_rows = (
                        True
                        if await self._rollout_rows_need_refence(
                            agent_id, rollout_nonce
                        )
                        else None
                    )

                if state_row is None:
                    # Table existence is database-global, while this state
                    # row is DID-local. A fresh-v2 provenance means a
                    # second/newly-added DID is not a legacy upgrade merely
                    # because another DID already created the table. Any
                    # NULL/non-v2 row still proves an old writer and wins.
                    needs_quiesce = (
                        not fresh_v2_schema or has_unknown_rows is not None
                    )
                    nonce = secrets.token_urlsafe(24)
                    inserted = await self._db.execute(
                        """
                        INSERT INTO scheduler_protocol_rollout
                            (agent_id, protocol_version, state,
                             activation_nonce, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(agent_id) DO NOTHING
                        """,
                        (
                            agent_id,
                            SCHEDULER_PROTOCOL_VERSION,
                            (
                                SCHEDULER_ROLLOUT_STATE_QUIESCING
                                if needs_quiesce
                                else SCHEDULER_ROLLOUT_STATE_ACTIVE
                            ),
                            nonce if needs_quiesce else None,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    if not self._updated(inserted):
                        continue
                    if needs_quiesce:
                        await self._fence_legacy_agent_rows(agent_id, nonce)
                        return nonce
                    return None

                protocol_version, state, nonce = state_row
                state_is_known = state in {
                    SCHEDULER_ROLLOUT_STATE_ACTIVE,
                    SCHEDULER_ROLLOUT_STATE_QUIESCING,
                }
                requires_new_nonce = (
                    protocol_version != SCHEDULER_PROTOCOL_VERSION
                    or not state_is_known
                    or has_unknown_rows is not None
                    or (
                        state == SCHEDULER_ROLLOUT_STATE_QUIESCING
                        and not nonce
                    )
                )

                if state == SCHEDULER_ROLLOUT_STATE_QUIESCING:
                    if requires_new_nonce:
                        new_nonce = secrets.token_urlsafe(24)
                        rotated = await self._rotate_rollout_nonce(
                            agent_id,
                            expected_protocol_version=protocol_version,
                            expected_state=state,
                            expected_nonce=nonce,
                            new_nonce=new_nonce,
                        )
                        if not rotated:
                            continue
                        await self._fence_legacy_agent_rows(agent_id, new_nonce)
                        return new_nonce
                    if self._rollout_acknowledges(nonce):
                        if await self._activate_protocol_rollout_agent(agent_id, nonce):
                            return None
                        # An unknown row appeared while the operator was
                        # acknowledging. Reread it and rotate the nonce; an
                        # old acknowledgement can never activate it.
                        continue
                    return nonce or "<missing nonce>"

                if requires_new_nonce:
                    new_nonce = secrets.token_urlsafe(24)
                    rotated = await self._rotate_rollout_nonce(
                        agent_id,
                        expected_protocol_version=protocol_version,
                        expected_state=state,
                        expected_nonce=nonce,
                        new_nonce=new_nonce,
                    )
                    if not rotated:
                        continue
                    await self._fence_legacy_agent_rows(agent_id, new_nonce)
                    return new_nonce

                # A current active control row and only v2 rows are safe.
                return None

        # Four losers in a row signals pathological external mutation. Fail
        # closed instead of guessing a nonce.
        return "<rollout state changed concurrently>"

    async def _rotate_rollout_nonce(
        self,
        agent_id: str,
        *,
        expected_protocol_version: Any,
        expected_state: Any,
        expected_nonce: Optional[str],
        new_nonce: str,
    ) -> bool:
        """CAS one observed control row into a fresh quiescing epoch."""

        # The caller already owns this DID's backend-neutral rollout gate.
        # For PostgreSQL that is the dedicated session gate acquired before
        # the operational transaction in ``_ensure_protocol_rollout`` (or the
        # sorted bootstrap boundary); taking a transaction advisory lock here
        # would make a one-connection query pool wait on itself.
        nonce_predicate = (
            "activation_nonce IS NULL"
            if expected_nonce is None
            else "activation_nonce = ?"
        )
        params: list[Any] = [
            SCHEDULER_PROTOCOL_VERSION,
            SCHEDULER_ROLLOUT_STATE_QUIESCING,
            new_nonce,
            datetime.now(timezone.utc).isoformat(),
            agent_id,
            expected_protocol_version,
            expected_state,
        ]
        if expected_nonce is not None:
            params.append(expected_nonce)
        updated = await self._db.execute(
            f"""
            UPDATE scheduler_protocol_rollout
            SET protocol_version = ?, state = ?, activation_nonce = ?,
                updated_at = ?
            WHERE agent_id = ?
              AND protocol_version = ?
              AND state = ?
              AND {nonce_predicate}
            """,
            tuple(params),
        )
        return self._updated(updated)

    async def _fence_legacy_agent_rows(self, agent_id: str, nonce: str) -> None:
        """Hide rows from legacy selectors and snapshot their visible state.

        This update locks every target row through the enclosing transaction.
        Recording the snapshot after ``enabled`` is forced to zero means an
        origin/main writer that later re-enables or reschedules the row is
        observed on the next acknowledgement attempt.
        """

        claim_present = (
            "(claim_execution_id IS NOT NULL AND claim_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL)"
        )
        fence_condition = (
            f"(enabled = 1 OR scheduler_rollout_fenced = 1 OR {claim_present})"
        )
        if self._uses_database_clock():
            fenced_at = self._database_now_sql()
            fenced_at_params: tuple[Any, ...] = ()
        else:
            fenced_at = "?"
            fenced_at_params = (datetime.now(timezone.utc).isoformat(),)
        await self._db.execute(
            f"""
            UPDATE scheduled_tasks
            SET enabled = 0,
                scheduler_rollout_fenced = CASE WHEN {fence_condition} THEN 1
                                                 ELSE scheduler_rollout_fenced END,
                scheduler_claim_fenced = CASE
                    WHEN {claim_present}
                    THEN 1 ELSE scheduler_claim_fenced END,
                -- A pre-fence worker retains its in-memory token. Leaving
                -- that token durable would let it finalize after ACK clears
                -- the rollout fence and re-enable a recurring row. Preserve
                -- only the stable execution/occurrence identity; revoke the
                -- ownership tuple so activation creates an immediately
                -- recoverable claim for a new v2 worker.
                lease_owner = CASE WHEN {claim_present} THEN NULL ELSE lease_owner END,
                lease_expires_at = CASE WHEN {claim_present} THEN NULL ELSE lease_expires_at END,
                claim_token = CASE WHEN {claim_present} THEN NULL ELSE claim_token END,
                scheduler_rollout_fenced_at = CASE
                    WHEN {fence_condition} THEN {fenced_at}
                    ELSE scheduler_rollout_fenced_at END,
                scheduler_rollout_nonce = ?
            WHERE agent_id = ?
            """,
            (*fenced_at_params, nonce, agent_id),
        )
        rows = await self._db.fetchall(
            """
            SELECT id, agent_id, task_name, cron_expression, args_json,
                   enabled, last_run_at, next_run_at, created_at
            FROM scheduled_tasks
            WHERE agent_id = ? AND scheduler_rollout_nonce = ?
            """,
            (agent_id, nonce),
        )
        for row in rows:
            await self._db.execute(
                """
                UPDATE scheduled_tasks SET scheduler_rollout_snapshot = ?
                WHERE id = ? AND agent_id = ? AND scheduler_rollout_nonce = ?
                """,
                (
                    self._rollout_snapshot(tuple(row)),
                    row[0],
                    agent_id,
                    nonce,
                ),
            )

    async def _activate_protocol_rollout_agent(
        self,
        agent_id: str,
        nonce: Optional[str],
    ) -> bool:
        """CAS-activate one quiesced DID after explicit legacy drain proof."""

        if not nonce:
            return False

        # Lock the exact control row before reading/converting its baseline.
        # This serializes acknowledgement with active->quiescing and with v2
        # schedule writers, while the schedule-row update below prevents a
        # legacy update from slipping between snapshot verification and the
        # activation CAS.  If this loses, do not convert anything: a newer
        # quiescing epoch owns the schedule rows.
        now_iso = datetime.now(timezone.utc).isoformat()
        locked = await self._db.execute(
            """
            UPDATE scheduler_protocol_rollout
            SET updated_at = updated_at
            WHERE agent_id = ? AND state = ? AND activation_nonce = ?
              AND protocol_version = ?
            """,
            (
                agent_id,
                SCHEDULER_ROLLOUT_STATE_QUIESCING,
                nonce,
                SCHEDULER_PROTOCOL_VERSION,
            ),
        )
        if not self._updated(locked):
            return False

        # Take row locks before verifying fingerprints. A legacy update that
        # committed before this point is visible in the snapshot comparison;
        # one attempting to commit afterwards remains blocked until this
        # transaction completes, which is why the operator acknowledgement
        # still requires all old replicas and in-flight executions drained.
        await self._db.execute(
            """
            UPDATE scheduled_tasks
            SET scheduler_rollout_snapshot = scheduler_rollout_snapshot
            WHERE agent_id = ? AND scheduler_rollout_nonce = ?
            """,
            (agent_id, nonce),
        )
        if await self._rollout_rows_need_refence(agent_id, nonce):
            return False

        # Flip the control row only after the exact baseline is locked and
        # verified. The ``NOT EXISTS`` predicate catches a post-baseline row
        # that is not part of this nonce's fenced epoch. The enclosing
        # transaction makes this CAS and baseline conversion visible together.
        updated = await self._db.execute(
            """
            UPDATE scheduler_protocol_rollout
            SET protocol_version = ?, state = ?, activation_nonce = NULL,
                updated_at = ?
            WHERE agent_id = ? AND state = ? AND activation_nonce = ?
              AND protocol_version = ?
              AND NOT EXISTS (
                  SELECT 1 FROM scheduled_tasks
                  WHERE agent_id = ?
                    AND (
                          scheduler_rollout_nonce IS NULL
                       OR scheduler_rollout_nonce <> ?
                    )
              )
            """,
            (
                SCHEDULER_PROTOCOL_VERSION,
                SCHEDULER_ROLLOUT_STATE_ACTIVE,
                now_iso,
                agent_id,
                SCHEDULER_ROLLOUT_STATE_QUIESCING,
                nonce,
                SCHEDULER_PROTOCOL_VERSION,
                agent_id,
                nonce,
            ),
        )
        if not self._updated(updated):
            return False

        # Convert every pre-v2 row only after the exact nonce was supplied.
        # Existing current-branch leases become claim-fenced recovery rows;
        # normal formerly-enabled rows are restored only if they are not due at
        # activation. An origin/main worker can select a due row just before
        # fencing, dispatch it, then re-read ``enabled=0`` and leave the old
        # ``next_run_at`` in place. Re-enabling that ambiguous occurrence would
        # replay its external effect, so it remains visibly paused and requires
        # an explicit operator resume. Limit this conversion to baseline rows
        # stamped by this nonce; a late legacy write is never silently adopted.
        if self._uses_database_clock():
            due_now = self._database_due_sql()
            terminal_now = self._database_now_sql()
            activation_params: tuple[Any, ...] = ()
        else:
            due_now = "next_run_at IS NOT NULL AND next_run_at <= ?"
            terminal_now = "?"
            activation_now = datetime.now(timezone.utc).isoformat()
            # ``due_now`` occurs in each CASE and ``terminal_now`` once.
            activation_params = (
                activation_now,
                activation_now,
                activation_now,
                activation_now,
            )
        claim_recovery = (
            "(scheduler_claim_fenced = 1 AND claim_execution_id IS NOT NULL "
            "AND claim_scheduled_for IS NOT NULL)"
        )
        ambiguous_due = (
            f"(scheduler_rollout_fenced = 1 AND NOT {claim_recovery} "
            f"AND {due_now})"
        )
        await self._db.execute(
            f"""
            UPDATE scheduled_tasks
            SET scheduler_protocol_version = ?,
                enabled = CASE
                    WHEN {claim_recovery} THEN 0
                    WHEN {ambiguous_due} THEN 0
                    WHEN scheduler_rollout_fenced = 1 THEN 1
                    ELSE enabled END,
                scheduler_claim_fenced = CASE
                    WHEN {claim_recovery} THEN 1
                    ELSE 0 END,
                terminal_status = CASE
                    WHEN {ambiguous_due} THEN '{ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE}'
                    ELSE terminal_status END,
                terminal_at = CASE
                    WHEN {ambiguous_due} THEN {terminal_now}
                    ELSE terminal_at END,
                scheduler_rollout_fenced = 0,
                scheduler_rollout_nonce = NULL,
                scheduler_rollout_fenced_at = NULL
            WHERE agent_id = ? AND scheduler_rollout_nonce = ?
            """,
            (
                SCHEDULER_PROTOCOL_VERSION,
                *activation_params,
                agent_id,
                nonce,
            ),
        )
        return True

    async def _add_column_if_missing(self, table: str, column: str, definition: str) -> None:
        """Add one scheduler column without poisoning a caller transaction.

        PostgreSQL aborts the whole surrounding transaction after a duplicate
        ``ADD COLUMN`` error, even if Python catches that error.  Use its
        native idempotent DDL instead.  SQLite has no equivalent syntax, so
        inspect its table metadata before issuing ordinary ``ALTER TABLE``.
        Both paths intentionally let introspection and DDL failures surface.
        """
        backend_type = self._database_backend_type()
        if backend_type == "postgres":
            await self._db.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {definition}"
            )
            logger.info("Applied scheduler migration %s.%s", table, column)
            return

        if backend_type == "sqlite":
            existing = await self._db.fetchone(
                f"SELECT 1 FROM pragma_table_info('{table}') WHERE name = ?",
                (column,),
            )
            if existing is not None:
                return

        # Keep lightweight legacy test doubles usable.  All production
        # backends reach one of the explicit paths above, and no backend path
        # uses exception-driven duplicate-column detection.
        await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        logger.info("Applied scheduler migration %s.%s", table, column)
