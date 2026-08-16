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
from datetime import datetime
from typing import Any, Dict, List

from kestrel_sdk.signals import ResourceLock
from kestrel_sovereign.signals import lock_manager as resource_lock_manager

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
            reachability = getattr(llm_service, "reachability", None)
            if not isinstance(reachability, list):
                reachability = None
            status = "pass"
            if reachability:
                if any(r.get("status") == "unreachable" for r in reachability):
                    status = "warn"
            return {
                "name": "llm_service",
                "status": status,
                "message": msg,
                "details": {"reachability": reachability} if reachability else {},
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


async def check_signal_audit_log(agent) -> Dict[str, Any]:
    """Report ``signal_log`` audit rows this process failed to persist (#2660).

    A dropped audit row is a permanent loss, but until this check existed its
    only trace was an ERROR line: 3,323 accumulated over two months of
    production before anyone looked. The dispatcher counts them; this surfaces
    the count where operators actually look.

    ``warn`` rather than ``fail`` — the agent keeps serving correctly and the
    audit trail has a hole. It does not clear when later writes succeed,
    because the rows lost earlier are still lost; an operator acknowledging
    that is the intent, not noise to be reset away.
    """
    start = time.monotonic()

    dispatcher = getattr(agent, "dispatcher", None)
    if dispatcher is None:
        return {
            "name": "signal_audit_log",
            "status": "pass",
            "message": "No signal dispatcher on this agent",
            "duration_ms": _elapsed(start),
        }

    # Duck-typed on purpose: third-party agents and test doubles implement the
    # DispatcherAgent protocol, not this accounting. A bare truthiness test
    # would warn on any stand-in whose attribute access auto-creates a value
    # (a MagicMock dispatcher reports a truthy count), so require a real
    # positive integer before claiming rows were lost. Reporting a loss that
    # did not happen is the same failure as hiding one that did.
    dropped = getattr(dispatcher, "log_write_failure_count", 0)
    if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped <= 0:
        return {
            "name": "signal_audit_log",
            "status": "pass",
            "message": "No signal_log writes dropped",
            "duration_ms": _elapsed(start),
        }

    last = getattr(dispatcher, "last_log_write_failure", None)
    details: Dict[str, Any] = {"dropped": dropped}
    if last is not None:
        details["last_signal_id"] = str(getattr(last, "signal_id", ""))
        details["last_error"] = str(getattr(last, "error", ""))
        failed_at = getattr(last, "failed_at", None)
        if isinstance(failed_at, datetime):
            details["last_failed_at"] = failed_at.isoformat()

    return {
        "name": "signal_audit_log",
        "status": "warn",
        "message": (
            f"{dropped} signal_log audit row(s) dropped since start; "
            "the audit trail is incomplete"
        ),
        "duration_ms": _elapsed(start),
        "details": details,
    }


async def check_resource_locks(agent) -> Dict[str, Any]:
    """Surface a MEMORY lock that is both long-held and blocking work.

    MEMORY protects bounded maintenance passes, so prolonged contention is an
    actionable health signal. Other resources have different healthy hold
    envelopes: in particular, CONVERSATION spans an entire agentic turn and
    must not make routine multi-minute work degrade readiness. The snapshot is
    in-memory and cannot itself block behind the resource being diagnosed.
    """
    start = time.monotonic()
    dispatcher = getattr(agent, "dispatcher", None)
    lock_manager = (
        getattr(dispatcher, "lock_manager", None)
        if dispatcher is not None
        else None
    )
    snapshot = getattr(lock_manager, "active_hold_diagnostics", None)
    if not callable(snapshot):
        return {
            "name": "resource_locks",
            "status": "pass",
            "message": "Resource lock diagnostics not available",
            "duration_ms": _elapsed(start),
        }

    raw_holds = snapshot()
    if not isinstance(raw_holds, list):
        return {
            "name": "resource_locks",
            "status": "pass",
            "message": "Resource lock diagnostics not available",
            "duration_ms": _elapsed(start),
        }

    slow_holds: list[dict[str, Any]] = []
    for raw in raw_holds:
        if not isinstance(raw, dict):
            continue
        if raw.get("resource") != ResourceLock.MEMORY.value:
            continue
        held_seconds = raw.get("held_seconds")
        blocked = raw.get("blocked_acquirers")
        if (
            isinstance(held_seconds, bool)
            or not isinstance(held_seconds, (int, float))
            or isinstance(blocked, bool)
            or not isinstance(blocked, int)
            or held_seconds < resource_lock_manager.SLOW_HOLD_WARN_SECONDS
            or blocked < 0
        ):
            continue
        slow_holds.append(dict(raw))

    if not slow_holds:
        return {
            "name": "resource_locks",
            "status": "pass",
            "message": "The memory lock is not held past the slow-hold threshold",
            "duration_ms": _elapsed(start),
        }

    blocked_holds = [
        hold for hold in slow_holds if hold["blocked_acquirers"] > 0
    ]
    if not blocked_holds:
        return {
            "name": "resource_locks",
            "status": "pass",
            "message": (
                f"{len(slow_holds)} resource lock(s) are long-held, but no "
                "acquirers are blocked"
            ),
            "duration_ms": _elapsed(start),
            "details": {"held_locks": slow_holds},
        }

    longest = max(blocked_holds, key=lambda hold: hold["held_seconds"])
    held_seconds = float(longest["held_seconds"])
    held_for = (
        f"{held_seconds / 3600:.1f} hours"
        if held_seconds >= 3600
        else f"{held_seconds:.0f} seconds"
    )
    escalated = any(
        hold["held_seconds"] >= resource_lock_manager.BLOCKED_HOLD_ERROR_SECONDS
        for hold in blocked_holds
    )
    return {
        "name": "resource_locks",
        "status": "fail" if escalated else "warn",
        "message": (
            f"{longest.get('resource', 'unknown')} lock has been held for "
            f"{held_for} by {longest.get('label', 'unknown')} with "
            f"{longest['blocked_acquirers']} acquirer(s) blocked"
        ),
        "duration_ms": _elapsed(start),
        "details": {
            "held_locks": slow_holds,
            "escalation_threshold_seconds": (
                resource_lock_manager.BLOCKED_HOLD_ERROR_SECONDS
            ),
        },
    }


async def check_birth_record(agent) -> Dict[str, Any]:
    """Report birth-record capability the runtime database does not have (#2871).

    On a host whose runtime database is not the one inception wrote to, boot
    copies the birth record across. Whatever the local anchor cannot supply —
    a constitution it never held, chunk ownership a pre-#2649 database could
    not prove — stays missing, and no retry will change that.

    The agent still boots: refusing would convert a degraded agent into one
    that can never boot again, with no operator verb to fix it, and identity
    failures (no agent node, a fabricated placeholder) are refused at boot
    already. But the loss must not be silent — #2871's whole defect was an
    agent with zero constitution chunks while ``/health`` reported ok.

    ``warn`` rather than ``fail``: the agent serves correctly, and the thing it
    cannot do is constitutional retrieval. It does not clear on its own.
    """
    start = time.monotonic()

    # Duck-typed like the sibling checks: third-party agents and test doubles
    # implement the health protocol, not this accounting. Require a real
    # non-empty list of strings before claiming a loss — reporting one that did
    # not happen is the same failure as hiding one that did.
    shortfall = getattr(agent, "_birth_record_shortfall", None)
    if not isinstance(shortfall, list) or not shortfall:
        return {
            "name": "birth_record",
            "status": "pass",
            "message": "Birth record complete in the runtime database",
            "duration_ms": _elapsed(start),
        }
    reasons = [str(reason) for reason in shortfall]
    # A completed pass that could not supply this means the anchor does not
    # have it; a pass that DIED is usually transient. Telling an operator a
    # dropped connection is unrepairable sends them to rebuild an anchor when
    # a restart would have fixed it.
    retryable = bool(getattr(agent, "_birth_record_shortfall_retryable", False))

    return {
        "name": "birth_record",
        "status": "warn",
        "message": (
            "Birth record incomplete in the runtime database; "
            + (
                "the copy failed this pass and a restart may complete it: "
                if retryable
                else "the local anchor cannot supply it: "
            )
            + "; ".join(reasons)
        ),
        "duration_ms": _elapsed(start),
        "details": {"missing": reasons, "retryable": retryable},
    }


async def run_standard_checks(agent, db) -> List[Dict[str, Any]]:
    """The checks every health surface runs, in one place.

    There are two callers — ``HealthFeature._run_health`` and the fallback in
    ``server.py`` for agents without the feature, which is not mandatory and so
    is reachable by design. They were two hand-maintained lists, and they had
    already drifted: a check added to one left the other reporting ``healthy``
    for a state the first calls a warning. Sharing the list is what stops the
    next one drifting.
    """
    return [
        await check_database(db),
        await check_llm_service(agent),
        await check_memory_system(agent),
        await check_disk_space(),
        await check_context_budget(agent),
        await check_bootstrap_state(agent),
        await check_signal_audit_log(agent),
        await check_resource_locks(agent),
        await check_birth_record(agent),
    ]


def _elapsed(start: float) -> float:
    """Return elapsed time in milliseconds since start."""
    return round((time.monotonic() - start) * 1000, 2)
