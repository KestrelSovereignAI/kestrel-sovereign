"""
Individual health check functions for the Heartbeat Feature.

Each check function returns a dict with:
    name: str       - human-readable check name
    status: str     - "pass", "warn", or "fail"
    message: str    - description of the result
    duration_ms: float - wall-clock time for the check

Checks are designed to be fast and non-destructive.
"""

import logging
import shutil
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def check_database(db) -> Dict[str, Any]:
    """Check database connectivity with a simple query.

    Args:
        db: AsyncDatabase instance (may be None)

    Returns:
        Check result dict
    """
    start = time.monotonic()

    if db is None:
        return {
            "name": "database",
            "status": "fail",
            "message": "No database connection available",
            "duration_ms": _elapsed(start),
        }

    try:
        # Simple read query to verify the connection is alive
        row = await db.fetchone("SELECT 1")
        if row and row[0] == 1:
            return {
                "name": "database",
                "status": "pass",
                "message": "Database connection healthy",
                "duration_ms": _elapsed(start),
            }
        return {
            "name": "database",
            "status": "warn",
            "message": "Database returned unexpected result",
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        return {
            "name": "database",
            "status": "fail",
            "message": f"Database query failed: {e}",
            "duration_ms": _elapsed(start),
        }


async def check_llm_service(agent) -> Dict[str, Any]:
    """Verify that an LLM provider is configured and accessible.

    This does NOT make an LLM call -- it only checks that the
    service object exists and has a provider configured.

    Args:
        agent: KestrelAgent instance

    Returns:
        Check result dict
    """
    start = time.monotonic()

    llm_service = getattr(agent, "llm_service", None)
    if llm_service is None:
        return {
            "name": "llm_service",
            "status": "fail",
            "message": "No LLM service configured",
            "duration_ms": _elapsed(start),
        }

    try:
        # Check that at least one provider is available
        providers = getattr(llm_service, "providers", None) or []

        # Resolve the active model via the canonical routing source so the
        # heartbeat reports what the agent will *actually* use, not just
        # what was configured or defaulted.
        active_model = None
        if hasattr(llm_service, "get_active_model_id"):
            active_model = llm_service.get_active_model_id()
            if active_model == "auto":
                active_model = None

        # Also surface the vendor (and route, if set) from mandate preference.
        active_vendor = None
        active_route = None
        if hasattr(llm_service, "get_model_preference"):
            pref = llm_service.get_model_preference()
            active_vendor = pref.get("vendor")
            active_route = pref.get("route")

        if providers:
            names = [p.get("name", p.get("vendor", "?")) for p in providers[:3]]
            msg = f"LLM providers available: {', '.join(names)}"
            if len(providers) > 3:
                msg += f" (+{len(providers) - 3} more)"
            if active_model:
                if active_vendor and active_route:
                    model_label = f"{active_vendor}:{active_route}/{active_model}"
                elif active_vendor:
                    model_label = f"{active_vendor}/{active_model}"
                else:
                    model_label = active_model
                msg += f" (active: {model_label})"
            return {
                "name": "llm_service",
                "status": "pass",
                "message": msg,
                "duration_ms": _elapsed(start),
            }

        return {
            "name": "llm_service",
            "status": "warn",
            "message": "LLM service exists but no providers initialized",
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        return {
            "name": "llm_service",
            "status": "fail",
            "message": f"LLM service check failed: {e}",
            "duration_ms": _elapsed(start),
        }


async def check_memory_system(agent) -> Dict[str, Any]:
    """Verify that the memory retriever and consolidator are accessible.

    Args:
        agent: KestrelAgent instance

    Returns:
        Check result dict
    """
    start = time.monotonic()

    try:
        storage = getattr(agent, "storage", None)
        if storage is None:
            return {
                "name": "memory_system",
                "status": "fail",
                "message": "No storage system available",
                "duration_ms": _elapsed(start),
            }

        # Check for memory retriever
        retriever = getattr(storage, "retriever", None) or getattr(
            agent, "memory_retriever", None
        )
        # Check for memory consolidator
        consolidator = getattr(storage, "consolidator", None) or getattr(
            agent, "memory_consolidator", None
        )

        components = []
        if retriever:
            components.append("retriever")
        if consolidator:
            components.append("consolidator")

        db = getattr(storage, "db", None)
        if db:
            components.append("database")

        if not components:
            return {
                "name": "memory_system",
                "status": "warn",
                "message": "Storage exists but no memory components found",
                "duration_ms": _elapsed(start),
            }

        return {
            "name": "memory_system",
            "status": "pass",
            "message": f"Memory system components: {', '.join(components)}",
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        return {
            "name": "memory_system",
            "status": "fail",
            "message": f"Memory system check failed: {e}",
            "duration_ms": _elapsed(start),
        }


async def check_disk_space(threshold_mb: int = 100) -> Dict[str, Any]:
    """Check available disk space on the root filesystem.

    Args:
        threshold_mb: Minimum free MB before warning (default 100)

    Returns:
        Check result dict
    """
    start = time.monotonic()

    try:
        usage = shutil.disk_usage("/")
        free_mb = usage.free / (1024 * 1024)
        total_mb = usage.total / (1024 * 1024)
        used_pct = (usage.used / usage.total) * 100 if usage.total > 0 else 0

        if free_mb < threshold_mb:
            return {
                "name": "disk_space",
                "status": "warn" if free_mb > threshold_mb / 2 else "fail",
                "message": (
                    f"Low disk space: {free_mb:.0f}MB free "
                    f"({used_pct:.1f}% used of {total_mb:.0f}MB)"
                ),
                "duration_ms": _elapsed(start),
            }

        return {
            "name": "disk_space",
            "status": "pass",
            "message": (
                f"Disk space OK: {free_mb:.0f}MB free "
                f"({used_pct:.1f}% used of {total_mb:.0f}MB)"
            ),
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        return {
            "name": "disk_space",
            "status": "fail",
            "message": f"Disk space check failed: {e}",
            "duration_ms": _elapsed(start),
        }


async def check_context_budget(agent) -> Dict[str, Any]:
    """Check context window utilization.

    Args:
        agent: KestrelAgent instance

    Returns:
        Check result dict
    """
    start = time.monotonic()

    try:
        ctx = getattr(agent, "context_manager", None)
        if ctx is not None:
            tokens_used = getattr(ctx, "tokens_used", 0) or 0
            tokens_max = getattr(ctx, "max_tokens", 0) or getattr(
                ctx, "tokens_max", 0
            ) or 0

            if tokens_max > 0:
                utilization = tokens_used / tokens_max
                status = "pass"
                if utilization > 0.9:
                    status = "fail"
                elif utilization > 0.75:
                    status = "warn"

                return {
                    "name": "context_budget",
                    "status": status,
                    "message": (
                        f"Context: {tokens_used}/{tokens_max} tokens "
                        f"({utilization * 100:.1f}% used)"
                    ),
                    "duration_ms": _elapsed(start),
                }

        # Fallback: try llm_service.token_budget
        llm = getattr(agent, "llm_service", None)
        if llm is not None:
            budget = getattr(llm, "token_budget", None)
            if budget is not None:
                used = getattr(budget, "used", 0) or 0
                total = getattr(budget, "total", 0) or 0
                if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
                    utilization = used / total
                    status = "pass"
                    if utilization > 0.9:
                        status = "fail"
                    elif utilization > 0.75:
                        status = "warn"

                    return {
                        "name": "context_budget",
                        "status": status,
                        "message": (
                            f"Token budget: {used}/{total} tokens "
                            f"({utilization * 100:.1f}% used)"
                        ),
                        "duration_ms": _elapsed(start),
                    }

        return {
            "name": "context_budget",
            "status": "pass",
            "message": "Context budget not tracked (no context manager)",
            "duration_ms": _elapsed(start),
        }
    except Exception as e:
        return {
            "name": "context_budget",
            "status": "fail",
            "message": f"Context budget check failed: {e}",
            "duration_ms": _elapsed(start),
        }


async def check_bootstrap_state(agent, threshold_seconds: int = 3600) -> Dict[str, Any]:
    """Check whether first-contact bootstrap is stuck in PENDING."""
    start = time.monotonic()

    bootstrap_service = getattr(agent, "bootstrap_service", None)
    if bootstrap_service is None:
        return {
            "name": "bootstrap_state",
            "status": "pass",
            "message": "Bootstrap service not configured",
            "duration_ms": _elapsed(start),
        }

    storage = getattr(agent, "storage", None)
    agent_node = None
    if storage is not None:
        try:
            agent_node = await storage.get_node(getattr(agent, "agent_id", None))
        except Exception as exc:
            logger.debug("Bootstrap health check could not load agent node: %s", exc)

    try:
        stale = await bootstrap_service.check_pending_timeout(
            agent_node=agent_node,
            storage=storage,
            threshold_seconds=threshold_seconds,
            mark_stale=False,
        )
    except Exception as exc:
        return {
            "name": "bootstrap_state",
            "status": "warn",
            "message": f"Bootstrap state check failed: {exc}",
            "duration_ms": _elapsed(start),
        }

    if stale.is_stale:
        age_minutes = (stale.age_seconds or 0) / 60.0
        return {
            "name": "bootstrap_state",
            "status": "warn",
            "message": (
                "bootstrap_state is pending for "
                f"{age_minutes:.1f} minutes; status=stale_bootstrap"
            ),
            "duration_ms": _elapsed(start),
            "details": stale.to_dict(),
        }

    return {
        "name": "bootstrap_state",
        "status": "pass",
        "message": f"Bootstrap state OK: {stale.state.value}",
        "duration_ms": _elapsed(start),
        "details": stale.to_dict(),
    }


def _elapsed(start: float) -> float:
    """Return elapsed time in milliseconds since start."""
    return round((time.monotonic() - start) * 1000, 2)
