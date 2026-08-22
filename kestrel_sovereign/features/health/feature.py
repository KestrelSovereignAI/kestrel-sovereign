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
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from . import checks as health_checks
from .checks import derive_overall_status as _derive_overall_status

logger = logging.getLogger(__name__)

# Default interval in seconds for the background liveness loop.
DEFAULT_INTERVAL_SECONDS = 60

# Maximum health results to keep in memory.
MAX_IN_MEMORY_HISTORY = 100


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
        self._health_persist_tasks: set[asyncio.Task] = set()
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
        await self._stop_background_loop()

        # A timed-out health persistence wait remains live so it cannot install
        # a global SQLite cancellation fence. Teardown still owns that work:
        # cancel and reap it before the feature/database lifecycle continues.
        pending_persists = [
            task
            for task in getattr(self, "_health_persist_tasks", set())
            if not task.done()
        ]
        for task in pending_persists:
            task.cancel()
        if pending_persists:
            await asyncio.gather(*pending_persists, return_exceptions=True)
        self._health_persist_tasks = set()
        await super().shutdown()
        logger.info("HealthFeature: background loop stopped")

    async def _stop_background_loop(self) -> None:
        """Stop only the periodic loop, leaving finite persistence owned."""
        self._running = False
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        self._background_task = None

    # =========================================================================
    # Tool commands (canonical !health* form)
    # =========================================================================

    # ------------------------------------------------------------------
    # Wrappers from the dict-returning internal helpers to ToolResult.
    #
    # The helpers (``_run_health``, ``_load_history``, ``_apply_interval``)
    # keep dict returns because the public API methods (``run_once``,
    # ``get_latest``) and HTTP endpoints (``/agent/health/*``) rely on
    # that shape, and the background liveness loop reads ``result['status']``
    # directly. Wrapping at the @tool boundary keeps the migration
    # localized.
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_health_result(result: Dict[str, Any]) -> ToolResult:
        """Render a ``_run_health`` dict as a ToolResult.

        Honesty: the check itself ran successfully even when the system
        is degraded/unhealthy. We return ToolResult.ok with the status
        explicit in the confirmation so the LLM cannot claim the
        system is healthy without reading the status field. The
        structured data carries the per-check breakdown.
        """
        status = result.get("status", "unknown")
        checks = result.get("checks", []) or []
        failed = [c for c in checks if c.get("status") == "fail"]
        warned = [c for c in checks if c.get("status") == "warn"]

        confirmation = (
            f"Liveness probe ran: overall={status}, "
            f"{len(checks)} check(s)"
            + (f", {len(failed)} failed" if failed else "")
            + (f", {len(warned)} warned" if warned else "")
        )
        return ToolResult.ok(
            confirmation=confirmation,
            data=dict(result),
        )

    @tool(
        name="health_check",
        description="Run a manual liveness check and show results",
        category=ToolCategory.SYSTEM,
        command_prefix="!health",
    )
    async def health_check(self) -> ToolResult:
        """Run all liveness checks and return the results."""
        return self._wrap_health_result(await self._run_health())

    @tool(
        name="health_history",
        description="Show recent liveness-check history and uptime",
        category=ToolCategory.SYSTEM,
        command_prefix="!health-history",
    )
    async def health_history(self, limit: int = 10) -> ToolResult:
        """Show the last N health results and uptime information.

        Args:
            limit: Maximum entries to return (the request — actual count
                returned may be lower if fewer entries exist).
        """
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"limit must be an integer, got {limit!r}"
            )
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        data = await self._load_history(limit_val)
        return ToolResult.ok(
            confirmation=(
                f"Health history: {data['history_count']} entry(ies), "
                f"uptime={data['uptime_seconds']}s, trend={data['trend']}, "
                f"interval={data['interval_seconds']}s"
            ),
            data=dict(data),
        )

    @tool(
        name="health_interval",
        description="Change the liveness-check interval",
        category=ToolCategory.SYSTEM,
        command_prefix="!health-interval",
    )
    async def health_interval(self, seconds: int = 60) -> ToolResult:
        """Change the background liveness-check interval.

        Args:
            seconds: Requested interval in seconds. Clamped to [10, 3600].
        """
        try:
            seconds_val = int(seconds)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"seconds must be an integer, got {seconds!r}"
            )

        # _apply_interval silently clamps to [10, 3600]. Honesty: if
        # the request was outside the range, surface as PARTIAL with
        # the actual applied value so the LLM cannot claim "set to 5"
        # when the runtime accepted 10.
        clamped = seconds_val < 10 or seconds_val > 3600
        data = await self._apply_interval(seconds_val)
        if clamped:
            return ToolResult.partial(
                confirmation=(
                    f"Interval changed: {data['old_interval_seconds']}s "
                    f"→ {data['new_interval_seconds']}s"
                ),
                error=(
                    f"requested {seconds_val}s is outside the allowed "
                    f"range [10, 3600]; clamped to "
                    f"{data['new_interval_seconds']}s"
                ),
                data={**data, "requested_seconds": seconds_val},
            )
        return ToolResult.ok(
            confirmation=(
                f"Interval changed: {data['old_interval_seconds']}s "
                f"→ {data['new_interval_seconds']}s"
            ),
            data={**data, "requested_seconds": seconds_val},
        )

    # =========================================================================
    # Legacy !heartbeat* aliases (deprecation-warn + forward)
    # =========================================================================

    @tool(
        name="heartbeat_check",
        description="[deprecated] alias for !health — use health instead",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat",
    )
    async def heartbeat_check_alias(self) -> ToolResult:
        self._warn_deprecated("!heartbeat", "!health")
        result = self._wrap_health_result(await self._run_health())
        return self._tag_alias(result, "!heartbeat")

    @tool(
        name="heartbeat_status",
        description="[deprecated] alias for !health-history — use health_history instead",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat-status",
    )
    async def heartbeat_status_alias(self, limit: int = 10) -> ToolResult:
        self._warn_deprecated("!heartbeat-status", "!health-history")
        result = await self.health_history(limit=limit)
        return self._tag_alias(result, "!heartbeat-status")

    @tool(
        name="heartbeat_interval",
        description="[deprecated] alias for !health-interval — use health_interval instead",
        category=ToolCategory.SYSTEM,
        command_prefix="!heartbeat-interval",
    )
    async def heartbeat_interval_alias(self, seconds: int = 60) -> ToolResult:
        self._warn_deprecated("!heartbeat-interval", "!health-interval")
        result = await self.health_interval(seconds=seconds)
        return self._tag_alias(result, "!heartbeat-interval")

    @staticmethod
    def _tag_alias(result: ToolResult, alias: str) -> ToolResult:
        """Inject ``deprecated_alias`` into a forwarded ToolResult.

        Preserve the alias marker across OK/PARTIAL/ERROR so the
        LLM/audit payload can always tell that the deprecated alias
        was used — including on input-validation errors raised by
        the canonical tool. (Round 1 codex finding for #1082.)
        """
        merged_data = {**(result.data or {}), "deprecated_alias": alias}
        status = result.status.value
        if status == "ok":
            return ToolResult.ok(
                confirmation=result.confirmation,
                data=merged_data,
            )
        if status == "partial":
            return ToolResult.partial(
                confirmation=result.confirmation,
                error=result.error,
                data=merged_data,
            )
        # ERROR: preserve the error string and surface the alias marker
        # via data so traceability holds for the failure path too.
        return ToolResult.failed(
            result.error or "unknown error",
            data=merged_data,
        )

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
                table_exists = self._db.table_exists
                if callable(
                    getattr(type(self._db), "table_exists_diagnostic", None)
                ):
                    table_exists = self._db.table_exists_diagnostic
                exists = await table_exists("health_log")
                if exists:
                    fetchall = self._db.fetchall
                    if callable(
                        getattr(type(self._db), "fetchall_diagnostic", None)
                    ):
                        fetchall = self._db.fetchall_diagnostic
                    rows = await fetchall(
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
            await self._stop_background_loop()
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

        # Shared with server.py's no-feature fallback so the two lists cannot
        # drift; a check present in one and absent from the other reports
        # `healthy` for a state its sibling calls a warning.
        checks = await health_checks.run_standard_checks(self.agent, self._db)

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

        database_failed = any(
            check.get("name") == "database" and check.get("status") == "fail"
            for check in checks
        )
        pending_persists = getattr(self, "_health_persist_tasks", None)
        if pending_persists is None:
            pending_persists = set()
            self._health_persist_tasks = pending_persists
        pending_persists.difference_update(
            {task for task in pending_persists if task.done()}
        )
        if self._db and not database_failed and not pending_persists:
            # Use the running loop directly: tests and embedders may replace
            # ``asyncio.create_task`` specifically to suppress the long-lived
            # health loop, but this finite persistence task must remain real.
            persist_task = asyncio.get_running_loop().create_task(
                self._db.execute(
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
                ),
                name=f"health-log-persist:{self._agent_id}",
            )
            pending_persists.add(persist_task)

            def _persist_done(task: asyncio.Task) -> None:
                pending_persists.discard(task)
                if task.cancelled():
                    return
                error = task.exception()
                if error is not None:
                    logger.warning("HealthFeature: failed to persist: %s", error)

            persist_task.add_done_callback(_persist_done)
            # The liveness call has a bounded wait, but expiry must not cancel a
            # shared-connection write. A cancelled aiosqlite operation installs
            # a backend-wide rollback fence. ``asyncio.wait`` returns with the
            # task still owned here and avoids creating a discarded shield
            # future that could report a late exception as unhandled.
            done, _ = await asyncio.wait(
                {persist_task},
                timeout=health_checks.DATABASE_HEALTH_CHECK_TIMEOUT_S,
            )
            if not done:
                logger.warning(
                    "HealthFeature: persistence still pending after %g seconds",
                    health_checks.DATABASE_HEALTH_CHECK_TIMEOUT_S,
                )
        elif self._db and not database_failed:
            logger.warning(
                "HealthFeature: skipping persistence while a prior health-log "
                "write is still pending"
            )

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
