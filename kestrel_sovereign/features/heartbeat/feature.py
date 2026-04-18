"""
Heartbeat Feature - periodic system health checks.

Runs configurable periodic health checks on the agent's infrastructure:
- Database connectivity
- LLM service availability
- Memory system health
- Disk space
- Context window utilization

Results are stored in a heartbeat_log table and exposed via tool commands
and the /health/detailed API endpoint.

The background task uses asyncio.create_task() and shuts down gracefully
via the Feature lifecycle.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.tools.base import ToolCategory

from .checks import (
    check_context_budget,
    check_database,
    check_disk_space,
    check_llm_service,
    check_memory_system,
)

logger = logging.getLogger(__name__)

# Default interval in seconds for the background heartbeat loop.
DEFAULT_INTERVAL_SECONDS = 60

# Maximum heartbeat results to keep in memory.
MAX_IN_MEMORY_HISTORY = 100


def _derive_overall_status(checks: List[Dict[str, Any]]) -> str:
    """Derive overall status from individual check results.

    - healthy: all checks pass
    - degraded: at least one warn, no fails
    - unhealthy: at least one critical check fails (database, llm_service)

    Non-critical checks that fail produce "degraded" rather than "unhealthy".
    """
    statuses = [c.get("status", "pass") for c in checks]
    # Critical checks: database and llm_service
    critical_names = {"database", "llm_service"}
    critical_checks = [c for c in checks if c.get("name") in critical_names]
    critical_statuses = [c.get("status", "pass") for c in critical_checks]

    if "fail" in critical_statuses:
        return "unhealthy"
    if "fail" in statuses or "warn" in statuses:
        return "degraded"
    return "healthy"


class HeartbeatFeature(Feature):
    """
    Periodic agent system health check feature.

    Provides:
    - !heartbeat: run a manual heartbeat and show results
    - !heartbeat-status: show last N heartbeats and uptime
    - !heartbeat-interval: change the background check interval

    The background loop is started in initialize() and stopped in shutdown().
    """

    @property
    def tool_description(self) -> str:
        return (
            "System heartbeat - run health checks on database, LLM service, "
            "memory system, disk space, and context budget. "
            "View heartbeat history and change check interval."
        )

    async def initialize(self):
        """Initialize the heartbeat feature, create DB table, start background loop."""
        self._db = None
        self._agent_id = ""
        self._interval_seconds = DEFAULT_INTERVAL_SECONDS
        self._background_task: Optional[asyncio.Task] = None
        self._running = False
        self._in_memory_history: List[Dict[str, Any]] = []
        self._start_time = time.monotonic()

        self._db = resolve_feature_database(self.agent)

        # Agent identity (DID is the canonical source of truth)
        self._agent_id = self.agent.did

        # Create heartbeat_log table
        if self._db:
            try:
                await self._db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS heartbeat_log (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        checks_json TEXT NOT NULL,
                        overall_healthy INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                await self._db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_heartbeat_log_agent
                    ON heartbeat_log(agent_id, created_at DESC)
                    """
                )
                logger.info("HeartbeatFeature: heartbeat_log table ready")
            except Exception as e:
                logger.warning(f"HeartbeatFeature: could not create table: {e}")

        # Start background heartbeat loop
        self._start_background_loop()

    async def shutdown(self):
        """Stop the background heartbeat loop gracefully."""
        self._running = False
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        self._background_task = None
        logger.info("HeartbeatFeature: background loop stopped")

    # =========================================================================
    # Tool commands
    # =========================================================================

    @tool(
        name="heartbeat_check",
        description="Run a manual heartbeat health check and show results",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat",
    )
    async def heartbeat_check(self) -> Dict[str, Any]:
        """Run all health checks and return the results.

        Returns:
            Dict with check results, overall status, and heartbeat ID
        """
        return await self._run_heartbeat()

    @tool(
        name="heartbeat_status",
        description="Show recent heartbeat history and uptime",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat-status",
    )
    async def heartbeat_status(self, limit: int = 10) -> Dict[str, Any]:
        """Show the last N heartbeat results and uptime information.

        Args:
            limit: Maximum number of heartbeat records to return (default 10)

        Returns:
            Dict with heartbeat history, uptime, and trend info
        """
        uptime_seconds = time.monotonic() - self._start_time

        # Try database first
        history = []
        if self._db:
            try:
                exists = await self._db.table_exists("heartbeat_log")
                if exists:
                    rows = await self._db.fetchall(
                        """
                        SELECT id, status, checks_json, overall_healthy, created_at
                        FROM heartbeat_log
                        WHERE agent_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (self._agent_id, limit),
                    )
                    for row in rows:
                        try:
                            checks = json.loads(row[2]) if row[2] else []
                        except (json.JSONDecodeError, TypeError):
                            checks = []
                        history.append(
                            {
                                "id": row[0],
                                "status": row[1],
                                "checks": checks,
                                "overall_healthy": bool(row[3]),
                                "created_at": row[4],
                            }
                        )
            except Exception as e:
                logger.warning(f"HeartbeatFeature: history query failed: {e}")

        # Fallback to in-memory history if DB is empty
        if not history:
            history = list(reversed(self._in_memory_history[-limit:]))

        # Compute trend from recent heartbeats
        trend = "unknown"
        if len(history) >= 2:
            latest_healthy = history[0].get("overall_healthy", True)
            previous_healthy = history[1].get("overall_healthy", True)
            if latest_healthy and not previous_healthy:
                trend = "recovering"
            elif not latest_healthy and previous_healthy:
                trend = "declining"
            elif latest_healthy and previous_healthy:
                trend = "stable"
            else:
                trend = "unstable"

        return {
            "uptime_seconds": round(uptime_seconds, 1),
            "interval_seconds": self._interval_seconds,
            "background_running": self._running,
            "history": history,
            "history_count": len(history),
            "trend": trend,
        }

    @tool(
        name="heartbeat_interval",
        description="Change the heartbeat check interval",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat-interval",
    )
    async def heartbeat_interval(self, seconds: int = 60) -> Dict[str, Any]:
        """Change the background heartbeat check interval.

        Args:
            seconds: New interval in seconds (minimum 10, maximum 3600)

        Returns:
            Dict confirming the new interval
        """
        seconds = max(10, min(seconds, 3600))
        old_interval = self._interval_seconds
        self._interval_seconds = seconds

        # Restart the background loop with the new interval
        if self._running:
            await self.shutdown()
            self._start_background_loop()

        return {
            "old_interval_seconds": old_interval,
            "new_interval_seconds": seconds,
            "background_running": self._running,
        }

    # =========================================================================
    # Core heartbeat logic
    # =========================================================================

    async def _run_heartbeat(self) -> Dict[str, Any]:
        """Execute all health checks and persist the result."""
        heartbeat_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        checks = []

        # 1. Database connectivity
        checks.append(await check_database(self._db))

        # 2. LLM service availability
        checks.append(await check_llm_service(self.agent))

        # 3. Memory system health
        checks.append(await check_memory_system(self.agent))

        # 4. Disk space
        checks.append(await check_disk_space())

        # 5. Context budget
        checks.append(await check_context_budget(self.agent))

        # Derive overall status
        overall_status = _derive_overall_status(checks)
        overall_healthy = overall_status == "healthy"

        result = {
            "heartbeat_id": heartbeat_id,
            "agent_id": self._agent_id,
            "status": overall_status,
            "checks": checks,
            "overall_healthy": overall_healthy,
            "created_at": now,
        }

        # Persist to database
        if self._db:
            try:
                await self._db.execute(
                    """
                    INSERT INTO heartbeat_log
                    (id, agent_id, status, checks_json, overall_healthy, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        heartbeat_id,
                        self._agent_id,
                        overall_status,
                        json.dumps(checks),
                        1 if overall_healthy else 0,
                        now,
                    ),
                )
            except Exception as e:
                logger.warning(f"HeartbeatFeature: failed to persist heartbeat: {e}")

        # Also keep in memory
        self._in_memory_history.append(result)
        if len(self._in_memory_history) > MAX_IN_MEMORY_HISTORY:
            self._in_memory_history = self._in_memory_history[-MAX_IN_MEMORY_HISTORY:]

        return result

    # =========================================================================
    # Background loop
    # =========================================================================

    def _start_background_loop(self) -> None:
        """Start the background heartbeat loop via asyncio.create_task()."""
        self._running = True
        self._background_task = asyncio.create_task(self._background_loop())
        logger.info(
            f"HeartbeatFeature: background loop started "
            f"(interval={self._interval_seconds}s)"
        )

    async def _background_loop(self) -> None:
        """Run heartbeat checks at the configured interval."""
        # Wait one interval before the first tick to avoid hammering at startup
        try:
            await asyncio.sleep(self._interval_seconds)
        except asyncio.CancelledError:
            return

        while self._running:
            try:
                result = await self._run_heartbeat()
                logger.info(
                    f"HeartbeatFeature: tick status={result['status']}"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"HeartbeatFeature: tick error: {e}", exc_info=True)

            try:
                await asyncio.sleep(self._interval_seconds)
            except asyncio.CancelledError:
                break

    # =========================================================================
    # Public API for endpoint integration
    # =========================================================================

    async def get_latest_heartbeat(self) -> Dict[str, Any]:
        """Return the most recent heartbeat result.

        Used by the /health/detailed endpoint.

        Returns:
            Latest heartbeat dict, or a fresh run if none exist
        """
        if self._in_memory_history:
            return self._in_memory_history[-1]
        # No history yet -- run one now
        return await self._run_heartbeat()
