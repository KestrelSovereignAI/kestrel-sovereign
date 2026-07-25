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
import inspect
import json
import logging
import time
import uuid
from collections.abc import Collection
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Protocol, Union

from kestrel_sovereign.features.scheduler.cron import CronParseError, next_run
from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome

logger = logging.getLogger(__name__)

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


class HostedExecutionExecutor(Protocol):
    """Executor contract for a host that runs schedules for cold agents."""

    async def execute_scheduled(self, execution: SchedulerExecution) -> Any:
        """Resolve/wake the target and dispatch ``execution``."""


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

    async def execute_scheduled(self, execution: SchedulerExecution) -> Any:
        agent = await self._resolve_agent(execution.agent_id)
        features = getattr(agent, "features", {}) or {}
        for feature in features.values():
            dispatch = getattr(feature, "_dispatch_scheduled_task", None)
            if callable(dispatch):
                return await dispatch(execution.task_name, execution.args)
        raise RuntimeError(
            f"woken agent {execution.agent_id!r} has no SchedulerFeature dispatcher"
        )


class AgentManagerHostedSchedulerExecutor(HostedSchedulerExecutor):
    """Hosted adapter that loads an unloaded agent through ``AgentManager``.

    ``agent_configs`` maps durable agent DIDs to the manager's ``(name,
    LocalAgentConfig)`` pair.  The host builds that map while it still has its
    fleet configuration.  Per-agent locks ensure two due schedules cannot
    cold-start the same target concurrently.
    """

    def __init__(self, agent_manager: Any, agent_configs: Dict[str, tuple[str, Any]]):
        self._agent_manager = agent_manager
        self._agent_configs = dict(agent_configs)
        self._locks: Dict[str, asyncio.Lock] = {}
        super().__init__(self._resolve_or_wake)

    async def _resolve_or_wake(self, agent_id: str) -> Any:
        lock = self._locks.setdefault(agent_id, asyncio.Lock())
        async with lock:
            agents = self._agent_manager.list_agents()
            for agent in agents.values():
                loaded_agent_id = (
                    getattr(agent, "did", None)
                    or getattr(agent, "agent_id", None)
                )
                if loaded_agent_id == agent_id:
                    return agent
            config = self._agent_configs.get(agent_id)
            if config is None:
                raise LookupError(f"No hosted agent configuration for {agent_id!r}")
            name, local_config = config
            return await self._agent_manager.load_agent(name, local_config)


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
        executor: Union[TaskExecutor, HostedExecutionExecutor],
        poll_interval: int = POLL_INTERVAL,
        misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        owner_id: Optional[str] = None,
        authorized_agent_ids: Optional[Collection[str]] = None,
    ):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if authorized_agent_ids is not None:
            authorized = tuple(sorted(set(authorized_agent_ids)))
            if not authorized or any(
                not isinstance(value, str) or not value for value in authorized
            ):
                raise ValueError(
                    "authorized_agent_ids must contain at least one non-empty agent ID"
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
        self._executor = executor
        self._poll_interval = poll_interval
        self._misfire_grace_seconds = max(0, int(misfire_grace_seconds))
        self._max_concurrent_tasks = max(1, int(max_concurrent_tasks))
        self._lease_seconds = int(lease_seconds)
        self._owner_id = owner_id or f"scheduler:{uuid.uuid4()}"
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        await self._ensure_tables()
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
            except Exception:
                logger.exception("SchedulerRunner tick error")
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise

    async def _tick(self):
        now = datetime.now(timezone.utc)
        rows = await self._due_rows(now)
        if not rows:
            return
        semaphore = asyncio.Semaphore(self._max_concurrent_tasks)

        async def run_one(task: ScheduledTask) -> None:
            async with semaphore:
                claimed = await self._claim(task, now)
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

    async def _due_rows(self, now: datetime) -> List[tuple]:
        authorization_scope = self._authorized_agent_placeholders()
        params: tuple = (*self._authorized_agent_ids, now.isoformat(), now.isoformat())
        return await self._db.fetchall(
            f"""
            SELECT id, agent_id, task_name, cron_expression, args_json,
                   enabled, last_run_at, next_run_at, created_at,
                   schedule_kind, run_at, timezone_name, misfire_policy,
                   misfire_grace_seconds, idempotency_key, lease_owner,
                   lease_expires_at, claim_token, claim_execution_id,
                   claim_scheduled_for, attempt_count, terminal_status, terminal_at
            FROM scheduled_tasks
            WHERE enabled = 1
              AND agent_id IN ({authorization_scope})
              AND next_run_at IS NOT NULL AND next_run_at <= ?
              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
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
        async with context:
            yield

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

    def _authorized_agent_placeholders(self) -> str:
        """Return the fixed placeholder list for this runner's fleet scope."""

        return ", ".join("?" for _ in self._authorized_agent_ids)

    async def _claim(self, task: ScheduledTask, now: datetime) -> Optional[ScheduledTask]:
        if task.next_run_at is None or task.agent_id not in self._authorized_agent_ids:
            return None
        scheduled_for = task.next_run_at
        execution_id = (
            task.claim_execution_id
            if task.claim_scheduled_for == scheduled_for and task.claim_execution_id
            else str(uuid.uuid4())
        )
        token = str(uuid.uuid4())
        base_idempotency = task.idempotency_key or f"legacy:{task.id}"
        idempotency_key = self._occurrence_idempotency_key(base_idempotency, task.id, scheduled_for)
        now_iso = now.isoformat()
        lease_expires = (now + timedelta(seconds=self._lease_seconds)).isoformat()

        authorization_scope = self._authorized_agent_placeholders()
        async with self._transaction():
            updated = await self._db.execute(
                f"""
                UPDATE scheduled_tasks
                SET lease_owner = ?, lease_expires_at = ?, claim_token = ?,
                    claim_execution_id = ?, claim_scheduled_for = ?,
                    attempt_count = COALESCE(attempt_count, 0) + 1,
                    idempotency_key = COALESCE(idempotency_key, ?)
                WHERE id = ? AND agent_id = ? AND enabled = 1
                  AND agent_id IN ({authorization_scope})
                  AND next_run_at = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                """,
                (
                    self._owner_id, lease_expires, token, execution_id, scheduled_for,
                    base_idempotency, task.id, task.agent_id,
                    *self._authorized_agent_ids, scheduled_for, now_iso,
                ),
            )
            if not self._updated(updated):
                return None
            await self._db.execute(
                """
                INSERT INTO task_execution_log
                    (id, task_id, agent_id, status, result_text, duration_ms,
                     executed_at, outcome_signal, occurrence_at, idempotency_key,
                     attempt_count, claimed_at, completed_at)
                VALUES (?, ?, ?, 'claimed', NULL, 0, ?, NULL, ?, ?, ?, ?, NULL)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    execution_id, task.id, task.agent_id, now_iso, scheduled_for,
                    idempotency_key, task.attempt_count + 1, now_iso,
                ),
            )
        return replace(
            task,
            lease_owner=self._owner_id,
            lease_expires_at=lease_expires,
            claim_token=token,
            claim_execution_id=execution_id,
            claim_scheduled_for=scheduled_for,
            attempt_count=task.attempt_count + 1,
            idempotency_key=base_idempotency,
        )

    @staticmethod
    def _occurrence_idempotency_key(base: str, schedule_id: str, scheduled_for: str) -> str:
        digest = hashlib.sha256(f"{schedule_id}\x00{scheduled_for}".encode()).hexdigest()
        return f"{base}:{digest}"

    async def _execute_claim(self, task: ScheduledTask) -> None:
        if task.agent_id not in self._authorized_agent_ids:
            logger.warning(
                "Refusing to execute scheduler claim %s for unauthorized agent %s",
                task.claim_execution_id,
                task.agent_id,
            )
            return
        assert task.claim_execution_id and task.claim_token and task.next_run_at
        execution = SchedulerExecution(
            id=task.claim_execution_id,
            schedule_id=task.id,
            agent_id=task.agent_id,
            task_name=task.task_name,
            args=task.args,
            scheduled_for=task.next_run_at,
            idempotency_key=self._occurrence_idempotency_key(
                task.idempotency_key or f"legacy:{task.id}", task.id, task.next_run_at,
            ),
            attempt=task.attempt_count,
            owner=self._owner_id,
        )
        now = datetime.now(timezone.utc)
        late = self._seconds_late(task.next_run_at, now)
        grace = self._misfire_grace_seconds if task.misfire_grace_seconds is None else max(0, int(task.misfire_grace_seconds))
        policy = task.misfire_policy if task.misfire_policy in MISFIRE_POLICIES else MISFIRE_SKIP
        if policy == MISFIRE_SKIP and grace and late > grace:
            await self._finalize(
                task, execution, status="skipped_misfire",
                result_text=(f"skipped: {late:.0f}s late (> {grace}s misfire grace); "
                             "policy=skip"),
                duration_ms=0, outcome_signal=None, ran=False,
            )
            return

        renewal = asyncio.create_task(self._renew_lease(task), name=f"scheduler-lease:{execution.id}")
        started = time.monotonic()
        status = "success"
        result_text: Optional[str] = None
        outcome_signal: Optional[float] = None
        pause_schedule = False
        try:
            scope = _SchedulerExecutionScope(execution)
            token = _current_execution.set(scope)
            try:
                raw = await self._run_executor(execution)
            finally:
                # Invalidate before the parent context is reset.  Child tasks
                # created by a target inherit this same scope, so they can no
                # longer present a completed occurrence as trusted scheduler
                # work after they outlive the dispatch.
                scope.revoke()
                _current_execution.reset(token)
            status, result_text, outcome_signal, pause_schedule = self._normalise_result(raw, task)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            status = "failed"
            result_text = f"{type(e).__name__}: {e}"
            logger.error("Scheduled task %s (%s) failed: %s", task.id, task.task_name, e)
        finally:
            renewal.cancel()
            try:
                await renewal
            except asyncio.CancelledError:
                pass
            except Exception:
                # Lease renewal is advisory once dispatch has returned: the
                # completion CAS remains the authority for this occurrence.
                # Harvest this task's exception so it cannot skip finalization
                # or surface later as an unobserved task failure.
                logger.exception(
                    "Scheduler lease renewal failed for task %s execution %s; "
                    "attempting completion CAS",
                    task.id,
                    execution.id,
                )
        await self._finalize(
            task, execution, status=status, result_text=result_text,
            duration_ms=int((time.monotonic() - started) * 1000),
            outcome_signal=outcome_signal, ran=True, pause_schedule=pause_schedule,
        )

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

    async def _renew_lease(self, task: ScheduledTask) -> None:
        if task.agent_id not in self._authorized_agent_ids:
            return
        interval = max(1.0, self._lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            expires = (datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds)).isoformat()
            updated = await self._db.execute(
                """
                UPDATE scheduled_tasks SET lease_expires_at = ?
                WHERE id = ? AND agent_id = ? AND enabled = 1
                  AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
                """,
                (
                    expires,
                    task.id,
                    task.agent_id,
                    self._owner_id,
                    task.claim_token,
                    task.claim_execution_id,
                ),
            )
            if not self._updated(updated):
                logger.warning("Lost scheduler lease for task %s execution %s", task.id, task.claim_execution_id)
                return

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
        if task.agent_id not in self._authorized_agent_ids:
            logger.warning(
                "Refusing to finalize scheduler execution %s for unauthorized agent %s",
                execution.id,
                task.agent_id,
            )
            return
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        terminal = task.schedule_kind == SCHEDULE_ONE_SHOT or pause_schedule
        enabled = 0 if terminal else 1
        terminal_status = status if terminal else None
        terminal_at = now_iso if terminal else None
        next_at: Optional[str] = None
        if not terminal:
            try:
                after = now
                if task.misfire_policy == MISFIRE_CATCH_UP and task.next_run_at:
                    after = self._parse_utc(task.next_run_at) or now
                next_at = next_run(task.cron_expression, after=after, timezone_name=task.timezone_name).isoformat()
            except CronParseError as e:
                status = "failed"
                result_text = f"{result_text or ''} scheduler cannot compute next run: {e}".strip()
                enabled = 0
                terminal_status = "invalid_cron"
                terminal_at = now_iso

        # A paused/deleted task must win over an in-flight execution.  The
        # compare-and-set also rejects an old worker that lost its lease to a
        # recovery worker; it must never overwrite the newer worker's outcome.
        async with self._transaction():
            last_run_sql = "last_run_at = ?" if ran else "last_run_at = last_run_at"
            params: list[Any] = []
            if ran:
                params.append(now_iso)
            params.extend([
                next_at, enabled, terminal_status, terminal_at,
                task.id, task.agent_id, self._owner_id, task.claim_token, execution.id,
            ])
            updated = await self._db.execute(
                f"""
                UPDATE scheduled_tasks
                SET {last_run_sql}, next_run_at = ?, enabled = ?,
                    terminal_status = ?, terminal_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL, claim_token = NULL,
                    claim_execution_id = NULL, claim_scheduled_for = NULL
                WHERE id = ? AND agent_id = ? AND enabled = 1
                  AND lease_owner = ? AND claim_token = ? AND claim_execution_id = ?
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
                raise RuntimeError(f"claimed execution log {execution.id} disappeared before finalization")

        logger.info(
            "Scheduled task %s (%s) finalized %s (%dms, attempt=%d)",
            task.id, task.task_name, status, duration_ms, execution.attempt,
        )

    async def _mark_cancelled_if_no_longer_runnable(self, execution: SchedulerExecution) -> None:
        if execution.agent_id not in self._authorized_agent_ids:
            return
        row = await self._db.fetchone(
            "SELECT enabled FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
            (execution.schedule_id, execution.agent_id),
        )
        if row is not None and bool(row[0]):
            return
        await self._db.execute(
            """
            UPDATE task_execution_log
            SET status = 'cancelled', result_text = 'schedule removed or paused while execution was in flight',
                completed_at = ?
            WHERE id = ? AND agent_id = ? AND status = 'claimed'
            """,
            (datetime.now(timezone.utc).isoformat(), execution.id, execution.agent_id),
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

    async def _ensure_tables(self):
        """Create and additively migrate the durable scheduler schema."""
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
                terminal_at TEXT
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
        await self._db.execute("UPDATE scheduled_tasks SET schedule_kind = 'cron' WHERE schedule_kind IS NULL")
        await self._db.execute("UPDATE scheduled_tasks SET timezone_name = 'UTC' WHERE timezone_name IS NULL OR timezone_name = ''")
        await self._db.execute("UPDATE scheduled_tasks SET misfire_policy = 'skip' WHERE misfire_policy IS NULL OR misfire_policy = ''")
        await self._db.execute("UPDATE scheduled_tasks SET idempotency_key = 'legacy:' || id WHERE idempotency_key IS NULL OR idempotency_key = ''")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_agent_next ON scheduled_tasks(agent_id, enabled, next_run_at)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_due_claim ON scheduled_tasks(enabled, next_run_at, lease_expires_at)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_task_execution_log_task ON task_execution_log(task_id, executed_at DESC)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_task_execution_log_idempotency ON task_execution_log(agent_id, idempotency_key)")

    async def _add_column_if_missing(self, table: str, column: str, definition: str) -> None:
        try:
            await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info("Applied scheduler migration %s.%s", table, column)
        except Exception as e:
            message = str(e).lower()
            if "duplicate column" in message or "already exists" in message or f"column {column}" in message:
                return
            raise
