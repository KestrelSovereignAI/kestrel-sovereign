"""
Scheduler Feature -- cron-based scheduled task execution for agents.

Allows agents to create, manage, and monitor scheduled tasks that run on
cron expressions. Tasks are persisted in the database and survive restarts.
A background asyncio runner checks for due tasks every 30 seconds.

Built-in task names (registered by other features):
    memory_consolidation  -- consolidate short-term memory into episodes
    wellness_checkpoint   -- run a wellness check
    audit_anchor          -- anchor the audit trail

Arbitrary feature tool invocations can also be scheduled by name.

Tools:
    !schedule list                          -- list all scheduled tasks
    !schedule add <cron> <task> [args_json] -- add a new task
    !schedule remove <task_id>             -- remove a task
    !schedule pause <task_id>              -- pause a task
    !schedule resume <task_id>             -- resume a paused task
    !schedule history                      -- recent execution history
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.scheduler.cron import CronParseError, next_run, parse
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class SchedulerFeature(Feature):
    """
    Cron/scheduler system for running agent tasks on a schedule.

    On initialize(), creates DB tables and starts a background runner that
    polls for due tasks every 30 seconds. Each due task is executed by
    invoking the registered callback (typically a feature tool).
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage scheduled tasks - add, remove, pause, resume cron-based "
            "tasks for memory consolidation, wellness checks, audit anchoring, "
            "and custom operations"
        )

    async def initialize(self):
        """Initialize the scheduler: set up DB refs and start the background runner."""
        self._db = None
        self._agent_id = ""
        self._runner: Optional[SchedulerRunner] = None

        self._db = resolve_feature_database(self.agent)

        # Agent identity (DID is the canonical source of truth)
        self._agent_id = self.agent.did

        if self._db is None:
            logger.warning("SchedulerFeature: no database available, running in no-op mode")
            return

        # Start background runner
        self._runner = SchedulerRunner(
            db=self._db,
            agent_id=self._agent_id,
            executor=self._execute_scheduled_task,
        )
        await self._runner.start()
        logger.info("SchedulerFeature initialized")

    async def post_all_features_loaded(self, agent):
        """Register default scheduled tasks after all features are loaded.

        Idempotent — checks for existing tasks before adding.
        Only schedules reflection/training if ReflectionFeature is available.
        """
        if not self._db:
            return

        # Check what's already scheduled
        existing = await self.schedule_list()
        existing_names = {t["task_name"] for t in existing.get("tasks", [])}

        # Reflection-dependent schedules only if ReflectionFeature is loaded
        has_reflection = "ReflectionFeature" in agent.features

        defaults = [
            ("backup_snapshot", "0 */4 * * *", "{}"),
            ("morning_signal", "0 8 * * *", "{}"),
            ("signal_dispatch", "5 8 * * *", "{}"),
        ]

        # Memory consolidation only if MemoryFeature is loaded (it owns the tool)
        if "MemoryFeature" in agent.features:
            defaults.append(
                ("memory_consolidate", "0 4 * * *", "{}"),  # nightly at 4am
            )

        if has_reflection:
            defaults.extend([
                ("reflect", "0 */4 * * *", '{"scope":"all","depth":"normal"}'),
                ("training_cycle", "0 3 * * *", '{"iterations":3,"depth":"normal"}'),
            ])

        for task_name, cron, args in defaults:
            if task_name in existing_names:
                logger.debug("Schedule '%s' already exists, skipping", task_name)
                continue
            result = await self.schedule_add(
                cron_expression=cron, task_name=task_name, args_json=args,
            )
            if result.get("success"):
                logger.info("Scheduled '%s' (%s), next: %s", task_name, cron, result.get("next_run_at"))
            else:
                logger.warning("Failed to schedule '%s': %s", task_name, result.get("error"))

    async def shutdown(self):
        """Stop the background runner."""
        if self._runner:
            await self._runner.stop()

    # ------------------------------------------------------------------
    # Task executor callback
    # ------------------------------------------------------------------

    async def _execute_scheduled_task(self, task_name: str, args: dict) -> str:
        """
        Execute a scheduled task by name.

        Looks up the task_name as a feature tool across all registered features
        on the agent, then invokes it with the supplied args.

        Args:
            task_name: Name of the tool to invoke (e.g. "wellness_check")
            args: Dict of keyword arguments for the tool

        Returns:
            JSON-encoded result string
        """
        # Built-in tasks (not feature tools)
        if task_name == "backup_snapshot":
            sync = getattr(self.agent, "_sync_service", None)
            if sync:
                results = await sync.force_snapshot()
                return json.dumps(
                    {t: {"success": r.success, "bytes": r.bytes_synced} for t, r in results.items()},
                    default=str,
                )
            return json.dumps({"error": "no sync service configured"})

        # Search all features for a matching tool
        features = getattr(self.agent, "features", {})
        for feature in features.values():
            if not hasattr(feature, "get_tools"):
                continue
            for agent_tool in feature.get_tools():
                if agent_tool.name == task_name:
                    result = await agent_tool.execute(**args)
                    return json.dumps(result, default=str)

        # Also check our own tools
        for agent_tool in self.get_tools():
            if agent_tool.name == task_name:
                result = await agent_tool.execute(**args)
                return json.dumps(result, default=str)

        raise ValueError(f"Unknown task: {task_name}")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        "schedule_list",
        "List all scheduled tasks for this agent",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule list",
    )
    async def schedule_list(self) -> Dict[str, Any]:
        """
        List all scheduled tasks for the current agent.

        Returns:
            Dict with list of tasks and count
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            rows = await self._db.fetchall(
                """
                SELECT id, task_name, cron_expression, args_json,
                       enabled, last_run_at, next_run_at, created_at
                FROM scheduled_tasks
                WHERE agent_id = ?
                ORDER BY created_at ASC
                """,
                (self._agent_id,),
            )

            tasks = []
            for row in rows:
                tasks.append({
                    "id": row[0],
                    "task_name": row[1],
                    "cron_expression": row[2],
                    "args": json.loads(row[3]) if row[3] else {},
                    "enabled": bool(row[4]),
                    "last_run_at": row[5],
                    "next_run_at": row[6],
                    "created_at": row[7],
                })

            return {"tasks": tasks, "count": len(tasks)}

        except Exception as e:
            logger.error("Failed to list scheduled tasks: %s", e)
            return {"success": False, "error": str(e)}

    @tool(
        "schedule_add",
        "Add a new scheduled task with a cron expression",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule add",
    )
    async def schedule_add(
        self,
        cron_expression: str,
        task_name: str,
        args_json: str = "{}",
    ) -> Dict[str, Any]:
        """
        Add a new scheduled task.

        Args:
            cron_expression: Cron expression (5 fields) or alias like @daily, @hourly
            task_name: Name of the tool to execute (e.g. wellness_check, audit_anchor)
            args_json: JSON-encoded arguments to pass to the tool (default: {})

        Returns:
            Dict with the created task details
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        # Validate cron expression
        try:
            parse(cron_expression)
        except CronParseError as e:
            return {"success": False, "error": f"Invalid cron expression: {e}"}

        # Validate args JSON
        try:
            parsed_args = json.loads(args_json)
            if not isinstance(parsed_args, dict):
                return {"success": False, "error": "args_json must be a JSON object"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid args_json: {e}"}

        # Compute first run time
        now = datetime.now(timezone.utc)
        try:
            first_run = next_run(cron_expression, after=now)
            next_run_at = first_run.isoformat()
        except CronParseError as e:
            return {"success": False, "error": f"Cannot compute next run: {e}"}

        task_id = str(uuid.uuid4())
        now_iso = now.isoformat()

        try:
            await self._db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json,
                     enabled, last_run_at, next_run_at, created_at)
                VALUES (?, ?, ?, ?, ?, 1, NULL, ?, ?)
                """,
                (task_id, self._agent_id, task_name, cron_expression,
                 args_json, next_run_at, now_iso),
            )
        except Exception as e:
            logger.error("Failed to add scheduled task: %s", e)
            return {"success": False, "error": str(e)}

        logger.info(
            "Scheduled task added: %s (%s) cron=%s next=%s",
            task_id, task_name, cron_expression, next_run_at,
        )

        return {
            "success": True,
            "task_id": task_id,
            "task_name": task_name,
            "cron_expression": cron_expression,
            "next_run_at": next_run_at,
            "created_at": now_iso,
        }

    @tool(
        "schedule_remove",
        "Remove a scheduled task by ID",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule remove",
    )
    async def schedule_remove(self, task_id: str) -> Dict[str, Any]:
        """
        Remove a scheduled task.

        Args:
            task_id: The UUID of the task to remove

        Returns:
            Dict with removal status
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            # Check task exists and belongs to this agent
            row = await self._db.fetchone(
                "SELECT id FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )
            if not row:
                return {"success": False, "error": f"Task {task_id} not found"}

            await self._db.execute(
                "DELETE FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )

            logger.info("Scheduled task removed: %s", task_id)
            return {"success": True, "task_id": task_id, "status": "removed"}

        except Exception as e:
            logger.error("Failed to remove task %s: %s", task_id, e)
            return {"success": False, "error": str(e)}

    @tool(
        "schedule_pause",
        "Pause a scheduled task (stops it from running until resumed)",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule pause",
    )
    async def schedule_pause(self, task_id: str) -> Dict[str, Any]:
        """
        Pause a scheduled task.

        Args:
            task_id: The UUID of the task to pause

        Returns:
            Dict with pause status
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            row = await self._db.fetchone(
                "SELECT id, enabled FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )
            if not row:
                return {"success": False, "error": f"Task {task_id} not found"}

            if not row[1]:
                return {"success": True, "task_id": task_id, "status": "already_paused"}

            await self._db.execute(
                "UPDATE scheduled_tasks SET enabled = 0 WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )

            logger.info("Scheduled task paused: %s", task_id)
            return {"success": True, "task_id": task_id, "status": "paused"}

        except Exception as e:
            logger.error("Failed to pause task %s: %s", task_id, e)
            return {"success": False, "error": str(e)}

    @tool(
        "schedule_resume",
        "Resume a paused scheduled task",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule resume",
    )
    async def schedule_resume(self, task_id: str) -> Dict[str, Any]:
        """
        Resume a paused scheduled task.

        Args:
            task_id: The UUID of the task to resume

        Returns:
            Dict with resume status
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            row = await self._db.fetchone(
                "SELECT id, enabled, cron_expression FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )
            if not row:
                return {"success": False, "error": f"Task {task_id} not found"}

            if row[1]:
                return {"success": True, "task_id": task_id, "status": "already_running"}

            # Recompute next_run_at from now
            cron_expr = row[2]
            now = datetime.now(timezone.utc)
            try:
                nxt = next_run(cron_expr, after=now)
                next_run_at = nxt.isoformat()
            except CronParseError:
                next_run_at = None

            await self._db.execute(
                "UPDATE scheduled_tasks SET enabled = 1, next_run_at = ? WHERE id = ? AND agent_id = ?",
                (next_run_at, task_id, self._agent_id),
            )

            logger.info("Scheduled task resumed: %s (next_run=%s)", task_id, next_run_at)
            return {
                "success": True,
                "task_id": task_id,
                "status": "resumed",
                "next_run_at": next_run_at,
            }

        except Exception as e:
            logger.error("Failed to resume task %s: %s", task_id, e)
            return {"success": False, "error": str(e)}

    @tool(
        "schedule_update",
        "Update the cron expression of an existing scheduled task",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule update",
    )
    async def schedule_update(
        self,
        task_id: str,
        cron_expression: str,
    ) -> Dict[str, Any]:
        """
        Update the cron expression on an existing scheduled task and recompute
        its next_run_at. Used by dynamic/self-adjusting schedulers (e.g. the
        reflection loop) to change task cadence without removing and re-adding.

        Args:
            task_id: The UUID of the task to update
            cron_expression: New cron expression (5 fields or alias)

        Returns:
            Dict with success, task_id, old_cron, new_cron, next_run_at
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        # Validate new cron expression
        try:
            parse(cron_expression)
        except CronParseError as e:
            return {"success": False, "error": f"Invalid cron expression: {e}"}

        try:
            row = await self._db.fetchone(
                "SELECT cron_expression, enabled FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )
            if not row:
                return {"success": False, "error": f"Task {task_id} not found"}

            old_cron = row[0]
            enabled = bool(row[1])

            if old_cron == cron_expression:
                return {
                    "success": True,
                    "task_id": task_id,
                    "status": "unchanged",
                    "cron_expression": cron_expression,
                }

            now = datetime.now(timezone.utc)
            next_run_at: Optional[str] = None
            if enabled:
                try:
                    next_run_at = next_run(cron_expression, after=now).isoformat()
                except CronParseError as e:
                    return {"success": False, "error": f"Cannot compute next run: {e}"}

            await self._db.execute(
                """
                UPDATE scheduled_tasks
                SET cron_expression = ?, next_run_at = ?
                WHERE id = ? AND agent_id = ?
                """,
                (cron_expression, next_run_at, task_id, self._agent_id),
            )

            logger.info(
                "Scheduled task updated: %s cron: %s -> %s (next=%s)",
                task_id, old_cron, cron_expression, next_run_at,
            )
            return {
                "success": True,
                "task_id": task_id,
                "status": "updated",
                "old_cron": old_cron,
                "cron_expression": cron_expression,
                "next_run_at": next_run_at,
            }

        except Exception as e:
            logger.error("Failed to update task %s: %s", task_id, e)
            return {"success": False, "error": str(e)}

    @tool(
        "schedule_record_outcome",
        "Attach an engagement signal (0.0-1.0) to a past task execution",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule outcome",
    )
    async def schedule_record_outcome(
        self,
        execution_id: str,
        signal: float,
    ) -> Dict[str, Any]:
        """
        Record an engagement signal on a past execution. Used when the
        downstream outcome is only known later — e.g. a morning_signal
        dispatched at 8am that the user replied to at 9:15am scores 1.0
        when the reply arrives.

        Signal is clamped to [0.0, 1.0].

        Args:
            execution_id: The id returned by schedule_history
            signal: Engagement score in [0.0, 1.0]

        Returns:
            Dict with success, execution_id, signal
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            clamped = max(0.0, min(1.0, float(signal)))
        except (TypeError, ValueError):
            return {"success": False, "error": f"signal must be numeric, got {signal!r}"}

        try:
            row = await self._db.fetchone(
                "SELECT id FROM task_execution_log WHERE id = ? AND agent_id = ?",
                (execution_id, self._agent_id),
            )
            if not row:
                return {"success": False, "error": f"Execution {execution_id} not found"}

            await self._db.execute(
                "UPDATE task_execution_log SET outcome_signal = ? WHERE id = ? AND agent_id = ?",
                (clamped, execution_id, self._agent_id),
            )
            return {"success": True, "execution_id": execution_id, "signal": clamped}

        except Exception as e:
            logger.error("Failed to record outcome for %s: %s", execution_id, e)
            return {"success": False, "error": str(e)}

    @tool(
        "schedule_engagement",
        "Report aggregate engagement scores per scheduled task",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule engagement",
    )
    async def schedule_engagement(self, days: int = 7) -> Dict[str, Any]:
        """
        Aggregate recent outcome signals per task.

        Returns one row per task with execution count, signal count (non-null),
        and mean signal over the window. A task with many executions but few
        signals indicates the downstream integration isn't reporting back —
        useful for diagnosing dead loops.

        Args:
            days: Look-back window in days (1-365, default: 7)

        Returns:
            Dict with per-task aggregates
        """
        try:
            days = int(days)
        except (TypeError, ValueError):
            return {"success": False, "error": f"days must be an integer, got {days!r}"}
        if days < 1 or days > 365:
            return {"success": False, "error": "days must be in [1, 365]"}

        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            from datetime import timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            rows = await self._db.fetchall(
                """
                SELECT
                    st.id, st.task_name, st.cron_expression,
                    COUNT(el.id) AS executions,
                    COUNT(el.outcome_signal) AS signals,
                    AVG(el.outcome_signal) AS mean_signal
                FROM scheduled_tasks st
                LEFT JOIN task_execution_log el
                    ON el.task_id = st.id AND el.executed_at >= ?
                WHERE st.agent_id = ?
                GROUP BY st.id, st.task_name, st.cron_expression
                ORDER BY st.task_name
                """,
                (since, self._agent_id),
            )

            tasks = [
                {
                    "task_id": row[0],
                    "task_name": row[1],
                    "cron_expression": row[2],
                    "executions": row[3] or 0,
                    "signals": row[4] or 0,
                    "mean_signal": float(row[5]) if row[5] is not None else None,
                }
                for row in rows
            ]
            return {"window_days": days, "tasks": tasks, "count": len(tasks)}

        except Exception as e:
            logger.error("Failed to aggregate engagement: %s", e)
            return {"success": False, "error": str(e)}

    @tool(
        "schedule_history",
        "Show recent task execution history",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule history",
    )
    async def schedule_history(self, limit: int = 20) -> Dict[str, Any]:
        """
        Show recent task execution history.

        Args:
            limit: Maximum number of execution records to return (default: 20)

        Returns:
            Dict with list of execution records
        """
        if not self._db:
            return {"success": False, "error": "Database not available"}

        try:
            rows = await self._db.fetchall(
                """
                SELECT el.id, el.task_id, el.status, el.result_text,
                       el.duration_ms, el.executed_at, st.task_name, el.outcome_signal
                FROM task_execution_log el
                LEFT JOIN scheduled_tasks st ON st.id = el.task_id
                WHERE el.agent_id = ?
                ORDER BY el.executed_at DESC
                LIMIT ?
                """,
                (self._agent_id, limit),
            )

            records = []
            for row in rows:
                records.append({
                    "id": row[0],
                    "task_id": row[1],
                    "status": row[2],
                    "result_text": row[3],
                    "duration_ms": row[4],
                    "executed_at": row[5],
                    "task_name": row[6],
                    "outcome_signal": row[7],
                })

            return {"executions": records, "count": len(records)}

        except Exception as e:
            logger.error("Failed to get execution history: %s", e)
            return {"success": False, "error": str(e)}
