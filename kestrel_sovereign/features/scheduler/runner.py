"""
Background scheduler runner for executing tasks on a cron schedule.

The SchedulerRunner runs as an asyncio background task, checking every
POLL_INTERVAL seconds for tasks whose next_run_at has passed. When a
task is due it invokes the registered callback (typically a feature tool)
and records the execution result in the task_execution_log table.

Tasks are persisted in the scheduled_tasks table so they survive restarts.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from kestrel_sovereign.features.scheduler.cron import next_run, CronParseError

logger = logging.getLogger(__name__)

# How often the runner wakes up to check for due tasks (seconds)
POLL_INTERVAL = 30

# Default misfire grace (seconds). A task more than this far past its
# scheduled time is assumed to have been slept through (host suspend) or
# starved, and is skipped-and-re-anchored to its next occurrence rather
# than fired late in a post-wake burst (#1545). 600s (10 min) is well
# above normal poll jitter (POLL_INTERVAL=30s plus execution time) yet
# short enough that a brief lid-close coalesces a missed hourly/daily run
# to the next one. 0 disables the rail (legacy fire-everything behaviour).
DEFAULT_MISFIRE_GRACE_SECONDS = 600


@dataclass
class ScheduledTask:
    """In-memory representation of a scheduled task row."""
    id: str
    agent_id: str
    task_name: str
    cron_expression: str
    args_json: str  # JSON-encoded arguments
    enabled: bool
    last_run_at: Optional[str]  # ISO timestamp or None
    next_run_at: Optional[str]  # ISO timestamp or None
    created_at: str

    @property
    def args(self) -> dict:
        if not self.args_json:
            return {}
        try:
            return json.loads(self.args_json)
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class ExecutionRecord:
    """Result of a single task execution."""
    id: str
    task_id: str
    agent_id: str
    status: str  # "success", "failed", "skipped"
    result_text: Optional[str]
    duration_ms: int
    executed_at: str


# Type alias for the callback that actually runs a task
TaskExecutor = Callable[[str, dict], Coroutine[Any, Any, str]]


class SchedulerRunner:
    """
    Background asyncio loop that checks for due tasks and executes them.

    Usage:
        runner = SchedulerRunner(db, agent_id, executor_fn)
        await runner.start()   # launches background task
        ...
        await runner.stop()    # graceful shutdown
    """

    def __init__(
        self,
        db,
        agent_id: str,
        executor: TaskExecutor,
        poll_interval: int = POLL_INTERVAL,
        misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
    ):
        """
        Args:
            db: An AsyncDatabase-like object with execute/fetchall/fetchone.
            agent_id: The owning agent's identifier.
            executor: Async callable(task_name, args_dict) -> result_text.
            poll_interval: Seconds between poll cycles.
            misfire_grace_seconds: A due task more than this many seconds late
                is skipped-and-re-anchored instead of executed (host-suspend
                resilience, #1545). 0 disables the rail.
        """
        self._db = db
        self._agent_id = agent_id
        self._executor = executor
        self._poll_interval = poll_interval
        self._misfire_grace_seconds = max(0, int(misfire_grace_seconds))
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Create DB tables (if needed) and launch the background loop."""
        await self._ensure_tables()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="scheduler-runner")
        logger.info("SchedulerRunner started (poll every %ds)", self._poll_interval)

    async def stop(self):
        """Cancel the background loop and wait for it to finish."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SchedulerRunner stopped")

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self):
        """Poll for due tasks and execute them."""
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
        """Single poll cycle: find due tasks and run them."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        rows = await self._db.fetchall(
            """
            SELECT id, agent_id, task_name, cron_expression, args_json,
                   enabled, last_run_at, next_run_at, created_at
            FROM scheduled_tasks
            WHERE agent_id = ? AND enabled = 1 AND next_run_at <= ?
            ORDER BY next_run_at ASC
            """,
            (self._agent_id, now_iso),
        )
        for row in rows:
            task = ScheduledTask(
                id=row[0],
                agent_id=row[1],
                task_name=row[2],
                cron_expression=row[3],
                args_json=row[4] or "{}",
                enabled=bool(row[5]),
                last_run_at=row[6],
                next_run_at=row[7],
                created_at=row[8],
            )
            # Misfire grace (#1545): a task far past its scheduled time was
            # almost certainly slept through (host suspend) or starved.
            # Firing it late — and firing every overdue task at once in the
            # first post-wake tick — is rarely what the operator wants for
            # "run around time T" cron work. Skip-and-re-anchor to the next
            # occurrence instead. Within grace, behave exactly as before.
            late = self._seconds_late(task.next_run_at, now)
            if self._misfire_grace_seconds and late > self._misfire_grace_seconds:
                await self._skip_misfire(task, now, late)
            else:
                await self._execute_task(task)

    @staticmethod
    def _seconds_late(next_run_at_iso: Optional[str], now: datetime) -> float:
        """Seconds between a task's scheduled ``next_run_at`` and ``now``.

        Returns 0.0 if the timestamp is missing or unparseable (treat as
        on-time rather than infinitely late — an unparseable timestamp is a
        separate bug that should not silently suppress execution)."""
        if not next_run_at_iso:
            return 0.0
        try:
            scheduled = datetime.fromisoformat(next_run_at_iso)
        except (ValueError, TypeError):
            return 0.0
        if scheduled.tzinfo is None:
            scheduled = scheduled.replace(tzinfo=timezone.utc)
        return max(0.0, (now - scheduled).total_seconds())

    async def _skip_misfire(self, task: ScheduledTask, now: datetime, late: float):
        """Record a skipped (slept-through) run and re-anchor the task to its
        next occurrence WITHOUT executing it.

        Writes a ``skipped_misfire`` row to ``task_execution_log`` so the
        skip is auditable (and visible in ``!schedule history``), then
        advances ``next_run_at`` past ``now``. ``last_run_at`` is left
        untouched — the task did not actually run."""
        now_iso = now.isoformat()
        logger.warning(
            "Scheduled task %s (%s) was %.0fs late (> %ds grace); "
            "skipping the slept-through run and re-anchoring",
            task.id, task.task_name, late, self._misfire_grace_seconds,
        )

        record_id = str(uuid.uuid4())
        try:
            await self._db.execute(
                """
                INSERT INTO task_execution_log
                    (id, task_id, agent_id, status, result_text, duration_ms,
                     executed_at, outcome_signal)
                VALUES (?, ?, ?, 'skipped_misfire', ?, 0, ?, NULL)
                """,
                (
                    record_id, task.id, self._agent_id,
                    f"skipped: {late:.0f}s late (> {self._misfire_grace_seconds}s "
                    "misfire grace); assumed host suspend",
                    now_iso,
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to record misfire skip for task %s: %s", task.id, e
            )

        # Re-read in case cron/enabled changed; mirror _execute_task's care.
        cron_expr = task.cron_expression
        task_still_live = True
        try:
            fresh = await self._db.fetchone(
                "SELECT cron_expression, enabled FROM scheduled_tasks WHERE id = ?",
                (task.id,),
            )
            if fresh is None:
                task_still_live = False
            else:
                cron_expr = fresh[0]
                if not bool(fresh[1]):
                    task_still_live = False
        except Exception as e:
            logger.warning("Failed to re-read task %s for misfire: %s", task.id, e)

        if not task_still_live:
            return

        try:
            next_iso = next_run(cron_expr, after=now).isoformat()
        except CronParseError:
            logger.warning(
                "Could not compute next_run for misfired task %s", task.id
            )
            return
        try:
            await self._db.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ? AND enabled = 1",
                (next_iso, task.id),
            )
        except Exception as e:
            logger.warning(
                "Failed to re-anchor misfired task %s: %s", task.id, e
            )

    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task and record the result."""
        start = time.monotonic()
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        status = "success"
        result_text: Optional[str] = None
        outcome_signal: Optional[float] = None

        try:
            raw = await self._executor(task.task_name, task.args)
            # Executors may return either a plain string or a (text, signal) tuple.
            # The signal must be numeric in [0.0, 1.0]; we clamp and drop non-
            # numeric values rather than propagating garbage to the DB.
            if isinstance(raw, tuple) and len(raw) == 2:
                result_text = raw[0] if isinstance(raw[0], str) else (str(raw[0]) if raw[0] is not None else None)
                signal_raw = raw[1]
                if signal_raw is None:
                    outcome_signal = None
                else:
                    try:
                        outcome_signal = max(0.0, min(1.0, float(signal_raw)))
                    except (TypeError, ValueError):
                        logger.warning(
                            "Task %s (%s) returned non-numeric outcome signal %r; dropping",
                            task.id, task.task_name, signal_raw,
                        )
                        outcome_signal = None
            else:
                result_text = raw if isinstance(raw, str) else (str(raw) if raw is not None else None)
        except Exception as e:
            status = "failed"
            result_text = str(e)
            logger.error("Scheduled task %s (%s) failed: %s", task.id, task.task_name, e)

        duration_ms = int((time.monotonic() - start) * 1000)

        # Record execution — keep the generated id so executors can attach
        # outcome signals after the fact (e.g. when a user responds to a
        # dispatched message minutes later).
        record_id = str(uuid.uuid4())
        try:
            await self._db.execute(
                """
                INSERT INTO task_execution_log
                    (id, task_id, agent_id, status, result_text, duration_ms, executed_at, outcome_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record_id, task.id, self._agent_id, status, result_text, duration_ms, now_iso, outcome_signal),
            )
        except Exception as e:
            logger.warning("Failed to record execution for task %s: %s", task.id, e)

        # Re-read the task row in case cron or enabled changed mid-flight.
        # If the task was paused or deleted while running, we must not compute
        # a new next_run_at — the pause intent wins over the runner's stale
        # in-memory snapshot.
        task_still_live = True
        cron_expr = task.cron_expression
        try:
            fresh = await self._db.fetchone(
                "SELECT cron_expression, enabled FROM scheduled_tasks WHERE id = ?",
                (task.id,),
            )
            if fresh is None:
                # Deleted mid-flight — don't try to reschedule.
                task_still_live = False
            else:
                cron_expr = fresh[0]
                if not bool(fresh[1]):
                    task_still_live = False
        except Exception as e:
            logger.warning("Failed to re-read task %s: %s", task.id, e)

        next_iso: Optional[str] = None
        if task_still_live:
            try:
                nxt = next_run(cron_expr, after=now)
                next_iso = nxt.isoformat()
            except CronParseError:
                logger.warning("Could not compute next_run for task %s", task.id)

        # Always record last_run_at even if the task was paused/deleted —
        # execution actually happened. Only touch next_run_at when still live.
        try:
            if task_still_live:
                await self._db.execute(
                    """
                    UPDATE scheduled_tasks
                    SET last_run_at = ?, next_run_at = ?
                    WHERE id = ? AND enabled = 1
                    """,
                    (now_iso, next_iso, task.id),
                )
            else:
                await self._db.execute(
                    """
                    UPDATE scheduled_tasks
                    SET last_run_at = ?
                    WHERE id = ?
                    """,
                    (now_iso, task.id),
                )
        except Exception as e:
            logger.warning("Failed to update task %s after execution: %s", task.id, e)

        logger.info(
            "Scheduled task %s (%s) executed: %s (%dms, signal=%s)",
            task.id, task.task_name, status, duration_ms, outcome_signal,
        )

    # ------------------------------------------------------------------
    # Table setup
    # ------------------------------------------------------------------

    async def _ensure_tables(self):
        """Create the scheduled_tasks and task_execution_log tables if needed."""
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
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_agent_next
            ON scheduled_tasks(agent_id, enabled, next_run_at)
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
                outcome_signal REAL
            )
            """
        )
        # Additive migration for pre-existing databases that lack outcome_signal.
        # Only swallow the specific "column already exists" case; anything else
        # (locked DB, permission errors, schema corruption) is a real problem
        # and must be surfaced — silently continuing could let inserts fail
        # later with confusing "no such column" errors.
        try:
            await self._db.execute(
                "ALTER TABLE task_execution_log ADD COLUMN outcome_signal REAL"
            )
            logger.info("Applied outcome_signal column migration")
        except Exception as e:
            msg = str(e).lower()
            is_duplicate = (
                "duplicate column" in msg
                or "already exists" in msg
                or "column outcome_signal" in msg
            )
            if is_duplicate:
                logger.debug("outcome_signal column already present — migration skipped")
            else:
                logger.error("Unexpected failure applying outcome_signal migration: %s", e)
                raise
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_execution_log_task
            ON task_execution_log(task_id, executed_at DESC)
            """
        )
