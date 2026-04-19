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
    ):
        """
        Args:
            db: An AsyncDatabase-like object with execute/fetchall/fetchone.
            agent_id: The owning agent's identifier.
            executor: Async callable(task_name, args_dict) -> result_text.
            poll_interval: Seconds between poll cycles.
        """
        self._db = db
        self._agent_id = agent_id
        self._executor = executor
        self._poll_interval = poll_interval
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
        now_iso = datetime.now(timezone.utc).isoformat()
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
            await self._execute_task(task)

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
            # The signal is a float in [0.0, 1.0] reporting downstream engagement.
            if isinstance(raw, tuple) and len(raw) == 2:
                result_text, outcome_signal = raw[0], raw[1]
            else:
                result_text = raw
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

        # Re-read the task row in case the cron expression was updated
        # mid-flight by schedule_update. Recomputing from the stale in-memory
        # task would ignore the user's change.
        try:
            fresh = await self._db.fetchone(
                "SELECT cron_expression FROM scheduled_tasks WHERE id = ?",
                (task.id,),
            )
            cron_expr = fresh[0] if fresh else task.cron_expression
        except Exception:
            cron_expr = task.cron_expression

        try:
            nxt = next_run(cron_expr, after=now)
            next_iso = nxt.isoformat()
        except CronParseError:
            next_iso = None
            logger.warning("Could not compute next_run for task %s", task.id)

        try:
            await self._db.execute(
                """
                UPDATE scheduled_tasks
                SET last_run_at = ?, next_run_at = ?
                WHERE id = ?
                """,
                (now_iso, next_iso, task.id),
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
        try:
            await self._db.execute(
                "ALTER TABLE task_execution_log ADD COLUMN outcome_signal REAL"
            )
        except Exception:
            # Column already exists — expected on fresh installs and second runs.
            pass
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_execution_log_task
            ON task_execution_log(task_id, executed_at DESC)
            """
        )
