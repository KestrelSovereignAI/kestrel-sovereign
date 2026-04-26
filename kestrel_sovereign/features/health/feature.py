"""
Health Feature - periodic liveness probes.

Runs configurable periodic checks on the agent's infrastructure:
- Database connectivity
- LLM service availability (object exists + providers populated, no LLM call)
- Memory system health
- Disk space
- Context window utilization

This is a **liveness / readiness probe**, not an agent heartbeat. In
the OpenClaw tradition (see ``openclaw/docs/gateway/heartbeat.md``) a
heartbeat is a scheduled agent turn that reads ``HEARTBEAT.md`` and
surfaces work needing attention — that's
:class:`kestrel_sovereign.heartbeat.HeartbeatRunner`, a different concept.
This feature never invokes the LLM.

Results are stored in the ``health_log`` table and exposed via tool
commands (``!health``, ``!health-history``, ``!health-interval``) and
the ``/agent/health/*`` HTTP endpoints.

Legacy aliases ``!heartbeat``, ``!heartbeat-status``, ``!heartbeat-interval``
are retained for one release and emit a deprecation warning when invoked;
the old ``heartbeat_log`` table is untouched so historical queries keep
working, but this feature no longer writes to it.

The background task uses ``asyncio.create_task()`` and shuts down
gracefully via the Feature lifecycle.
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

# Default interval in seconds for the background liveness loop.
DEFAULT_INTERVAL_SECONDS = 60

# Maximum health results to keep in memory.
MAX_IN_MEMORY_HISTORY = 100


def _derive_overall_status(checks: List[Dict[str, Any]]) -> str:
    """Derive overall status from individual check results.

    - healthy: all checks pass
    - degraded: at least one warn, no fails
    - unhealthy: at least one critical check fails (database, llm_service)
    """
    statuses = [c.get("status", "pass") for c in checks]
    critical_names = {"database", "llm_service"}
    critical_checks = [c for c in checks if c.get("name") in critical_names]
    critical_statuses = [c.get("status", "pass") for c in critical_checks]

    if "fail" in critical_statuses:
        return "unhealthy"
    if "fail" in statuses or "warn" in statuses:
        return "degraded"
    return "healthy"


class HealthFeature(Feature):
    """
    Periodic agent-subsystem liveness probe.

    Provides:
    - ``!health``: run a manual liveness check and show results
    - ``!health-history``: show last N checks and uptime
    - ``!health-interval``: change the background check interval
    - ``!heartbeat`` / ``!heartbeat-status`` / ``!heartbeat-interval``:
      legacy aliases that log a deprecation warning and forward to the
      ``!health*`` handlers.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Agent liveness probe - periodically checks database, LLM "
            "service wiring, memory system, disk space, and context "
            "budget. View history and change check interval."
        )

    async def initialize(self):
        """Initialize the feature, create DB table, start background loop."""
        self._db = None
        self._agent_id = ""
        self._interval_seconds = DEFAULT_INTERVAL_SECONDS
        self._background_task: Optional[asyncio.Task] = None
        self._running = False
        self._in_memory_history: List[Dict[str, Any]] = []
        self._start_time = time.monotonic()

        self._db = resolve_feature_database(self.agent)
        self._agent_id = self.agent.did

        if self._db:
            try:
                await self._db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS health_log (
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
                    CREATE INDEX IF NOT EXISTS idx_health_log_agent
                    ON health_log(agent_id, created_at DESC)
                    """
                )
                logger.info("HealthFeature: health_log table ready")
            except Exception as e:
                logger.warning(f"HealthFeature: could not create table: {e}")

        self._start_background_loop()

    async def shutdown(self):
        """Stop the background loop gracefully."""
        self._running = False
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        self._background_task = None
        logger.info("HealthFeature: background loop stopped")

    # =========================================================================
    # Tool commands (canonical !health* form)
    # =========================================================================

    @tool(
        name="health_check",
        description="Run a manual liveness check and show results",
        category=ToolCategory.SYSTEM,
        command_prefix="!health",
    )
    async def health_check(self) -> Dict[str, Any]:
        """Run all liveness checks and return the results."""
        return await self._run_health()

    @tool(
        name="health_history",
        description="Show recent liveness-check history and uptime",
        category=ToolCategory.SYSTEM,
        command_prefix="!health-history",
    )
    async def health_history(self, limit: int = 10) -> Dict[str, Any]:
        """Show the last N health results and uptime information."""
        return await self._load_history(limit)

    @tool(
        name="health_interval",
        description="Change the liveness-check interval",
        category=ToolCategory.SYSTEM,
        command_prefix="!health-interval",
    )
    async def health_interval(self, seconds: int = 60) -> Dict[str, Any]:
        """Change the background liveness-check interval."""
        return await self._apply_interval(seconds)

    # =========================================================================
    # Legacy !heartbeat* aliases (deprecation-warn + forward)
    # =========================================================================

    @tool(
        name="heartbeat_check",
        description="[deprecated] alias for !health",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat",
    )
    async def heartbeat_check_alias(self) -> Dict[str, Any]:
        self._warn_deprecated("!heartbeat", "!health")
        return await self._run_health()

    @tool(
        name="heartbeat_status",
        description="[deprecated] alias for !health-history",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat-status",
    )
    async def heartbeat_status_alias(self, limit: int = 10) -> Dict[str, Any]:
        self._warn_deprecated("!heartbeat-status", "!health-history")
        return await self._load_history(limit)

    @tool(
        name="heartbeat_interval",
        description="[deprecated] alias for !health-interval",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat-interval",
    )
    async def heartbeat_interval_alias(self, seconds: int = 60) -> Dict[str, Any]:
        self._warn_deprecated("!heartbeat-interval", "!health-interval")
        return await self._apply_interval(seconds)

    def _warn_deprecated(self, old: str, new: str) -> None:
        logger.warning(
            "Deprecated command %s used; switch to %s. The alias will be "
            "removed in a future release.",
            old,
            new,
        )

    # =========================================================================
    # Shared command implementations
    # =========================================================================

    async def _load_history(self, limit: int) -> Dict[str, Any]:
        uptime_seconds = time.monotonic() - self._start_time

        history: List[Dict[str, Any]] = []
        if self._db:
            try:
                exists = await self._db.table_exists("health_log")
                if exists:
                    rows = await self._db.fetchall(
                        """
                        SELECT id, status, checks_json, overall_healthy, created_at
                        FROM health_log
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
                logger.warning(f"HealthFeature: history query failed: {e}")

        if not history:
            history = list(reversed(self._in_memory_history[-limit:]))

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

    async def _apply_interval(self, seconds: int) -> Dict[str, Any]:
        seconds = max(10, min(seconds, 3600))
        old_interval = self._interval_seconds
        self._interval_seconds = seconds

        if self._running:
            await self.shutdown()
            self._start_background_loop()

        return {
            "old_interval_seconds": old_interval,
            "new_interval_seconds": seconds,
            "background_running": self._running,
        }

    # =========================================================================
    # Core liveness logic
    # =========================================================================

    async def _run_health(self) -> Dict[str, Any]:
        """Execute all liveness checks and persist the result to health_log."""
        check_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        checks = []
        checks.append(await check_database(self._db))
        checks.append(await check_llm_service(self.agent))
        checks.append(await check_memory_system(self.agent))
        checks.append(await check_disk_space())
        checks.append(await check_context_budget(self.agent))

        overall_status = _derive_overall_status(checks)
        overall_healthy = overall_status == "healthy"

        result = {
            "id": check_id,
            "agent_id": self._agent_id,
            "status": overall_status,
            "checks": checks,
            "overall_healthy": overall_healthy,
            "created_at": now,
        }

        if self._db:
            try:
                await self._db.execute(
                    """
                    INSERT INTO health_log
                    (id, agent_id, status, checks_json, overall_healthy, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        check_id,
                        self._agent_id,
                        overall_status,
                        json.dumps(checks),
                        1 if overall_healthy else 0,
                        now,
                    ),
                )
            except Exception as e:
                logger.warning(f"HealthFeature: failed to persist: {e}")

        self._in_memory_history.append(result)
        if len(self._in_memory_history) > MAX_IN_MEMORY_HISTORY:
            self._in_memory_history = self._in_memory_history[-MAX_IN_MEMORY_HISTORY:]

        return result

    # =========================================================================
    # Background loop
    # =========================================================================

    def _start_background_loop(self) -> None:
        """Start the background loop via ``asyncio.create_task()``."""
        self._running = True
        self._background_task = asyncio.create_task(self._background_loop())
        logger.info(
            f"HealthFeature: background loop started "
            f"(interval={self._interval_seconds}s)"
        )

    async def _background_loop(self) -> None:
        """Run liveness checks at the configured interval."""
        try:
            await asyncio.sleep(self._interval_seconds)
        except asyncio.CancelledError:
            return

        while self._running:
            try:
                result = await self._run_health()
                logger.info(f"HealthFeature: tick status={result['status']}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"HealthFeature: tick error: {e}", exc_info=True)

            try:
                await asyncio.sleep(self._interval_seconds)
            except asyncio.CancelledError:
                break

    # =========================================================================
    # Public API (used by the /agent/health/* endpoints)
    # =========================================================================

    async def run_once(self) -> Dict[str, Any]:
        """Run a single liveness check synchronously."""
        return await self._run_health()

    async def get_latest(self) -> Dict[str, Any]:
        """Return the most recent result, running one if none exist."""
        if self._in_memory_history:
            return self._in_memory_history[-1]
        return await self._run_health()

    def get_status(self) -> Dict[str, Any]:
        """Return static status info about the feature (no LLM, no IO)."""
        uptime_seconds = time.monotonic() - self._start_time
        last = self._in_memory_history[-1] if self._in_memory_history else None
        return {
            "enabled": True,
            "running": self._running,
            "interval_seconds": self._interval_seconds,
            "uptime_seconds": round(uptime_seconds, 1),
            "history_count": len(self._in_memory_history),
            "last_result": last,
        }
