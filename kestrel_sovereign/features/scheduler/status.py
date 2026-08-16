"""Canonical scheduler runtime and schedule-state reporting.

The scheduler has two independent facts to report: whether a worker is
checking the durable queue, and what that queue currently contains.  Keeping
those facts in one durable report prevents an absent worker report from being
misread as a valid report containing zero schedule rows.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Collection, Dict, Optional


RUNTIME_STATUS_TABLE = "scheduler_runtime_status"
DEFAULT_STALE_AFTER_TICKS = 3
DEFAULT_MISFIRE_GRACE_SECONDS = 600


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def classify_disablement(
    *,
    enabled: bool,
    terminal_status: Optional[str],
    rollout_fenced: bool = False,
    claim_fenced: bool = False,
    lease_owner: Optional[str] = None,
    lease_expires_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe why a row is not enabled without collapsing transient fences.

    Rows created before explicit disablement reporting can still be classified
    safely: an ordinary disabled row with no scheduler terminal reason is an
    operator pause, while scheduler safety paths always retain a terminal
    status.  Rollout and claim fences are transient protocol state, not pauses.
    """

    if rollout_fenced:
        return {
            "state": "rollout_fenced",
            "source": "scheduler_protocol",
            "reason": "scheduler_protocol_rollout",
            "recoverable": False,
            "recovery_action": "complete the scheduler protocol rollout",
        }
    lease_expiry = _parse_utc(lease_expires_at)
    lease_is_live = bool(
        claim_fenced
        and lease_owner
        and lease_expiry is not None
        and lease_expiry > datetime.now(timezone.utc)
    )
    if lease_is_live:
        return {
            "state": "executing",
            "source": "scheduler",
            "reason": "claimed_occurrence_executing",
            "recoverable": False,
            "recovery_action": None,
        }
    if claim_fenced:
        return {
            "state": "claim_recovery",
            "source": "scheduler_protocol",
            "reason": "claimed_occurrence_recovery",
            "recoverable": False,
            "recovery_action": "allow the scheduler lease recovery to finish",
        }
    if enabled:
        return {
            "state": "enabled",
            "source": None,
            "reason": None,
            "recoverable": False,
            "recovery_action": None,
        }
    if terminal_status:
        recoverable = terminal_status != "invalid_idempotency_key"
        action = (
            "run schedule_resume with acknowledge_ambiguous_effect=true after "
            "verifying the legacy occurrence's effect"
            if terminal_status == "rollout_ambiguous_legacy_occurrence"
            else (
                "repair or recreate the schedule with a valid idempotency key"
                if not recoverable
                else "run schedule_resume after correcting the reported cause"
            )
        )
        return {
            "state": "disabled",
            "source": "scheduler",
            "reason": terminal_status,
            "recoverable": recoverable,
            "recovery_action": action,
        }
    return {
        "state": "paused",
        "source": "operator",
        "reason": "operator_paused",
        "recoverable": True,
        "recovery_action": "run schedule_resume",
    }


async def ensure_runtime_status_table(db: Any) -> None:
    """Create the durable worker-report table used by every scheduler surface."""

    await db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUNTIME_STATUS_TABLE} (
            agent_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            worker_state TEXT NOT NULL,
            reported_at TEXT NOT NULL,
            last_tick_started_at TEXT,
            last_tick_completed_at TEXT,
            restart_count INTEGER NOT NULL DEFAULT 0,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            last_error_type TEXT,
            configured_enabled_count INTEGER NOT NULL DEFAULT 0,
            enabled_count INTEGER NOT NULL DEFAULT 0,
            executing_count INTEGER NOT NULL DEFAULT 0,
            disabled_count INTEGER NOT NULL DEFAULT 0,
            fenced_count INTEGER NOT NULL DEFAULT 0,
            system_disabled_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )


async def _schedule_inventory(
    db: Any,
    agent_id: str,
    *,
    default_misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
    overdue_grace_floor_seconds: int = 0,
) -> Dict[str, Any]:
    rows = await db.fetchall(
        """
        SELECT enabled, next_run_at, terminal_status,
               scheduler_rollout_fenced, scheduler_claim_fenced,
               lease_owner, lease_expires_at, misfire_grace_seconds
        FROM scheduled_tasks
        WHERE agent_id = ?
        """,
        (agent_id,),
    )
    configured_enabled_count = 0
    enabled_count = 0
    executing_count = 0
    disabled_count = 0
    fenced_count = 0
    system_disabled_count = 0
    disabled_reasons: Dict[str, int] = {}
    next_runs: list[datetime] = []
    overdue_candidates: list[datetime] = []
    overdue_deadlines: list[datetime] = []
    for row in rows:
        enabled = bool(row[0])
        disablement = classify_disablement(
            enabled=enabled,
            terminal_status=row[2],
            rollout_fenced=bool(row[3]),
            claim_fenced=bool(row[4]),
            lease_owner=row[5],
            lease_expires_at=row[6],
        )
        state = disablement["state"]
        if state in {"enabled", "executing"}:
            configured_enabled_count += 1
        if state == "enabled":
            enabled_count += 1
        elif state == "executing":
            executing_count += 1
        elif state in {"rollout_fenced", "claim_recovery"}:
            fenced_count += 1
        elif state in {"paused", "disabled"}:
            disabled_count += 1
            reason = str(disablement["reason"])
            disabled_reasons[reason] = disabled_reasons.get(reason, 0) + 1
            if disablement["source"] != "operator":
                system_disabled_count += 1

        next_at = _parse_utc(row[1])
        if disablement["state"] == "enabled" and next_at is not None:
            next_runs.append(next_at)
            overdue_candidates.append(next_at)
            task_grace = (
                default_misfire_grace_seconds
                if row[7] is None
                else max(0, int(row[7]))
            )
            overdue_deadlines.append(
                next_at
                + timedelta(
                    seconds=max(overdue_grace_floor_seconds, task_grace)
                )
            )

    return {
        "schedule_count": len(rows),
        "configured_enabled_count": configured_enabled_count,
        "enabled_count": enabled_count,
        "executing_count": executing_count,
        "disabled_count": disabled_count,
        "fenced_count": fenced_count,
        "system_disabled_count": system_disabled_count,
        "disabled_reasons": disabled_reasons,
        "next_run_at": min(next_runs).isoformat() if next_runs else None,
        "oldest_unclaimed_run_at": (
            min(overdue_candidates).isoformat() if overdue_candidates else None
        ),
        "oldest_unclaimed_overdue_at": (
            min(overdue_deadlines).isoformat() if overdue_deadlines else None
        ),
    }


def scheduler_status_parameters(feature: Any) -> Dict[str, Any]:
    """Return the canonical lifecycle context for scheduler status readers.

    A standalone runner is not expected to emit telemetry until its final
    ``on_agent_ready`` arm. Before that point the startup grace remains open,
    regardless of how long feature loading takes. Once arm is requested, the
    grace is measured from that runner lifecycle transition, not from feature
    construction. Host-managed features have no local runner, so their feature
    initialization timestamp remains the only local lifecycle anchor.
    """

    runner = getattr(feature, "_runner", None)
    poll_interval = max(
        1, int(getattr(runner, "_poll_interval", 30) or 30)
    )
    grace_seconds = poll_interval * DEFAULT_STALE_AFTER_TICKS
    expected_owner_id = getattr(runner, "_owner_id", None)
    expected_started_at = getattr(runner, "_arm_requested_at", None)

    arm_requested = getattr(runner, "_arm_requested", None)
    if runner is not None and arm_requested is False:
        startup_grace_remaining = float(grace_seconds)
    else:
        lifecycle_started = (
            getattr(runner, "_arm_requested_monotonic", None)
            if runner is not None
            else getattr(feature, "_initialized_monotonic", None)
        )
        if not isinstance(lifecycle_started, (int, float)):
            lifecycle_started = getattr(feature, "_initialized_monotonic", None)
        lifecycle_age = (
            max(0.0, time.monotonic() - lifecycle_started)
            if isinstance(lifecycle_started, (int, float))
            else float(grace_seconds)
        )
        startup_grace_remaining = max(0.0, grace_seconds - lifecycle_age)

    default_misfire_grace = getattr(
        runner, "_misfire_grace_seconds", None
    )
    if default_misfire_grace is None:
        load_misfire_grace = getattr(
            feature, "_load_misfire_grace_seconds", None
        )
        default_misfire_grace = (
            load_misfire_grace()
            if callable(load_misfire_grace)
            else DEFAULT_MISFIRE_GRACE_SECONDS
        )

    return {
        "poll_interval": poll_interval,
        "startup_grace_remaining": startup_grace_remaining,
        "expected_owner_id": expected_owner_id,
        "expected_started_at": expected_started_at,
        "default_misfire_grace_seconds": max(
            0, int(default_misfire_grace or 0)
        ),
    }


async def emit_runtime_status(
    db: Any,
    *,
    agent_ids: Collection[str],
    owner_id: str,
    worker_state: str,
    last_tick_started_at: Optional[str],
    last_tick_completed_at: Optional[str],
    restart_count: int,
    consecutive_failures: int,
    last_error_type: Optional[str],
) -> None:
    """Emit one explicit report per authorized agent, including zero counts."""

    now = datetime.now(timezone.utc).isoformat()
    for agent_id in sorted(set(agent_ids)):
        inventory = await _schedule_inventory(db, agent_id)
        await db.execute(
            f"""
            INSERT INTO {RUNTIME_STATUS_TABLE}
                (agent_id, owner_id, worker_state, reported_at,
                 last_tick_started_at, last_tick_completed_at, restart_count,
                 consecutive_failures, last_error_type,
                 configured_enabled_count, enabled_count, executing_count,
                 disabled_count, fenced_count, system_disabled_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                owner_id = excluded.owner_id,
                worker_state = excluded.worker_state,
                reported_at = excluded.reported_at,
                last_tick_started_at = excluded.last_tick_started_at,
                last_tick_completed_at = excluded.last_tick_completed_at,
                restart_count = excluded.restart_count,
                consecutive_failures = excluded.consecutive_failures,
                last_error_type = excluded.last_error_type,
                configured_enabled_count = excluded.configured_enabled_count,
                enabled_count = excluded.enabled_count,
                executing_count = excluded.executing_count,
                disabled_count = excluded.disabled_count,
                fenced_count = excluded.fenced_count,
                system_disabled_count = excluded.system_disabled_count
            """,
            (
                agent_id,
                owner_id,
                worker_state,
                now,
                last_tick_started_at,
                last_tick_completed_at,
                restart_count,
                consecutive_failures,
                last_error_type,
                inventory["configured_enabled_count"],
                inventory["enabled_count"],
                inventory["executing_count"],
                inventory["disabled_count"],
                inventory["fenced_count"],
                inventory["system_disabled_count"],
            ),
        )


async def scheduler_status(
    db: Any,
    *,
    agent_id: str,
    poll_interval: int,
    stale_after_ticks: int = DEFAULT_STALE_AFTER_TICKS,
    startup_grace_remaining: float = 0.0,
    expected_owner_id: Optional[str] = None,
    expected_started_at: Optional[str] = None,
    default_misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
) -> Dict[str, Any]:
    """Read and diagnose scheduler liveness plus current durable row state."""

    stale_after_seconds = max(1, int(poll_interval)) * max(
        1, int(stale_after_ticks)
    )
    inventory = await _schedule_inventory(
        db,
        agent_id,
        default_misfire_grace_seconds=max(
            0, int(default_misfire_grace_seconds)
        ),
        overdue_grace_floor_seconds=stale_after_seconds,
    )
    report = await db.fetchone(
        f"""
        SELECT owner_id, worker_state, reported_at, last_tick_started_at,
               last_tick_completed_at, restart_count, consecutive_failures,
               last_error_type, configured_enabled_count, enabled_count,
               executing_count, disabled_count, fenced_count,
               system_disabled_count
        FROM {RUNTIME_STATUS_TABLE}
        WHERE agent_id = ?
        """,
        (agent_id,),
    )
    now = datetime.now(timezone.utc)
    details: Dict[str, Any] = {
        **inventory,
        "telemetry_received": report is not None,
        "current_owner_telemetry_received": False,
        "stale_after_seconds": stale_after_seconds,
    }
    if report is None:
        details.update({
            "state": (
                "awaiting_telemetry"
                if startup_grace_remaining > 0
                else "no_telemetry"
            ),
            "worker_state": "unknown",
            "reported_at": None,
            "report_age_seconds": None,
            "restart_count": 0,
            "consecutive_failures": 0,
            "last_error_type": None,
            "status": "warn" if startup_grace_remaining > 0 else "fail",
        })
        return details

    report_owner_id = str(report[0] or "")
    reported_at = _parse_utc(report[2])
    report_age = (
        max(0.0, (now - reported_at).total_seconds())
        if reported_at is not None
        else None
    )
    worker_state = str(report[1] or "unknown")
    owner_matches = not expected_owner_id or report_owner_id == expected_owner_id
    expected_start = _parse_utc(expected_started_at)
    report_predates_expected_start = bool(
        expected_start is not None
        and (reported_at is None or reported_at < expected_start)
    )
    details.update({
        "state": "running",
        "owner_id": report_owner_id,
        "expected_owner_id": expected_owner_id,
        "expected_started_at": expected_started_at,
        "report_predates_expected_start": report_predates_expected_start,
        "current_owner_telemetry_received": owner_matches,
        "worker_state": worker_state,
        "reported_at": report[2],
        "report_age_seconds": (
            round(report_age, 3) if report_age is not None else None
        ),
        "last_tick_started_at": report[3],
        "last_tick_completed_at": report[4],
        "restart_count": int(report[5] or 0),
        "consecutive_failures": int(report[6] or 0),
        "last_error_type": report[7],
        "reported_configured_enabled_count": int(report[8] or 0),
        "reported_enabled_count": int(report[9] or 0),
        "reported_executing_count": int(report[10] or 0),
        "reported_disabled_count": int(report[11] or 0),
        "reported_fenced_count": int(report[12] or 0),
        "reported_system_disabled_count": int(report[13] or 0),
    })

    tick_started = _parse_utc(report[3])
    tick_completed = _parse_utc(report[4])
    details["tick_in_progress"] = bool(
        tick_started is not None
        and (tick_completed is None or tick_started > tick_completed)
    )

    if expected_owner_id and not owner_matches:
        details["state"] = (
            "awaiting_telemetry"
            if startup_grace_remaining > 0
            else "no_current_telemetry"
        )
        details["status"] = "warn" if startup_grace_remaining > 0 else "fail"
        return details

    if report_predates_expected_start:
        details["state"] = (
            "awaiting_telemetry"
            if startup_grace_remaining > 0
            else "no_current_telemetry"
        )
        details["status"] = "warn" if startup_grace_remaining > 0 else "fail"
        return details

    prior_owner_grace = bool(
        startup_grace_remaining > 0 and not expected_owner_id
    )
    if report_age is None or report_age > stale_after_seconds:
        details["state"] = (
            "awaiting_telemetry" if prior_owner_grace else "stale"
        )
        details["status"] = "warn" if prior_owner_grace else "fail"
        return details
    if worker_state == "starting":
        details["state"] = (
            "awaiting_telemetry"
            if startup_grace_remaining > 0
            else "starting"
        )
        details["status"] = "warn" if startup_grace_remaining > 0 else "fail"
        return details
    if worker_state != "running":
        details["state"] = (
            "awaiting_telemetry" if prior_owner_grace else worker_state
        )
        details["status"] = "warn" if prior_owner_grace else "fail"
        return details

    oldest = _parse_utc(inventory["oldest_unclaimed_run_at"])
    overdue_at = _parse_utc(inventory["oldest_unclaimed_overdue_at"])
    # A current tick owns every row selected in its due batch, including rows
    # queued behind the concurrency semaphore but not yet claimed. Those rows
    # are healthy in-flight work, not evidence that polling stopped.
    if (
        not details["tick_in_progress"]
        and oldest is not None
        and overdue_at is not None
        and overdue_at < now
    ):
        details["state"] = "overdue"
        details["overdue_seconds"] = round((now - oldest).total_seconds(), 3)
        details["status"] = "fail"
        return details
    if inventory["system_disabled_count"]:
        details["state"] = "system_disabled_schedules"
        details["status"] = "warn"
        return details
    if inventory["fenced_count"]:
        details["state"] = "protocol_fenced_schedules"
        details["status"] = "warn"
        return details
    if inventory["enabled_count"] == 0 and inventory["executing_count"] == 0:
        if inventory["disabled_count"]:
            details["state"] = "running_only_operator_paused_schedules"
        else:
            details["state"] = "running_zero_schedules"
    details["status"] = "pass"
    return details


__all__ = [
    "DEFAULT_MISFIRE_GRACE_SECONDS",
    "DEFAULT_STALE_AFTER_TICKS",
    "RUNTIME_STATUS_TABLE",
    "classify_disablement",
    "emit_runtime_status",
    "ensure_runtime_status_table",
    "scheduler_status",
    "scheduler_status_parameters",
]
