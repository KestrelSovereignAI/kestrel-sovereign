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

from kestrel_sovereign.features.scheduler.constants import (
    ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE,
    SCHEDULER_SAFETY_DISABLEMENT_REASONS,
)
from kestrel_sovereign.storage.database_clock import (
    database_clock as scheduler_database_clock,
    database_now_sql as scheduler_database_now_sql,
)


RUNTIME_STATUS_TABLE = "scheduler_runtime_status"
DEFAULT_STALE_AFTER_TICKS = 3
DEFAULT_MISFIRE_GRACE_SECONDS = 600
DEFAULT_LEASE_SECONDS = 120
RUNTIME_STATUS_RETENTION_SECONDS = 24 * 60 * 60

_RUNTIME_STATUS_TABLE_SQL = f"""
    CREATE TABLE IF NOT EXISTS {RUNTIME_STATUS_TABLE} (
        agent_id TEXT NOT NULL,
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
        system_disabled_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (agent_id, owner_id)
    )
"""


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
    schedule_kind: str,
    terminal_status: Optional[str],
    database_now: datetime,
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
        and lease_expiry > database_now
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
    if (
        schedule_kind == "one_shot"
        and terminal_status
        and terminal_status not in SCHEDULER_SAFETY_DISABLEMENT_REASONS
    ):
        return {
            "state": "terminal",
            "source": "scheduler_history",
            "reason": terminal_status,
            "recoverable": False,
            "recovery_action": None,
        }
    if terminal_status:
        one_shot_inconsistent = (
            schedule_kind == "one_shot"
            and terminal_status == "execution_log_inconsistent"
        )
        recoverable = (
            terminal_status != "invalid_idempotency_key"
            and not one_shot_inconsistent
        )
        if terminal_status == ROLLOUT_AMBIGUOUS_LEGACY_OCCURRENCE:
            action = (
                "run schedule_resume with acknowledge_ambiguous_effect=true "
                "after verifying the legacy occurrence's effect"
            )
        elif one_shot_inconsistent:
            action = "reconcile the execution log and recreate the one-shot schedule"
        elif recoverable:
            action = "run schedule_resume after correcting the reported cause"
        else:
            action = "repair or recreate the schedule with a valid idempotency key"
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

    backend_type = str(getattr(db, "backend_type", "") or "").lower()
    if backend_type == "sqlite":
        await _ensure_sqlite_runtime_status_table(db)
    elif backend_type == "postgres":
        await _ensure_postgres_runtime_status_table(db)
    else:
        await db.execute(_RUNTIME_STATUS_TABLE_SQL)
    await db.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{RUNTIME_STATUS_TABLE}_agent_reported "
        f"ON {RUNTIME_STATUS_TABLE}(agent_id, reported_at)"
    )


async def _ensure_sqlite_runtime_status_table(db: Any) -> None:
    legacy_table = f"{RUNTIME_STATUS_TABLE}_legacy_agent_owner"
    tables = await db.fetchall(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
        (RUNTIME_STATUS_TABLE, legacy_table),
    )
    names = {str(row[0]) for row in tables}
    if RUNTIME_STATUS_TABLE not in names and legacy_table not in names:
        await db.execute(_RUNTIME_STATUS_TABLE_SQL)
        return
    if RUNTIME_STATUS_TABLE in names:
        columns = await db.fetchall(f"PRAGMA table_info({RUNTIME_STATUS_TABLE})")
        key_columns = [
            str(row[1])
            for row in sorted(columns, key=lambda row: int(row[5]))
            if row[5]
        ]
        if key_columns == ["agent_id"]:
            if legacy_table in names:
                raise RuntimeError(
                    "scheduler runtime ownership migration found both legacy and live tables"
                )
            await db.execute(
                f"ALTER TABLE {RUNTIME_STATUS_TABLE} RENAME TO {legacy_table}"
            )
            names.remove(RUNTIME_STATUS_TABLE)
            names.add(legacy_table)
        elif key_columns != ["agent_id", "owner_id"]:
            raise RuntimeError(
                "scheduler runtime status has an unsupported primary key"
            )
    if legacy_table not in names:
        return
    await db.execute(_RUNTIME_STATUS_TABLE_SQL)
    columns = (
        "agent_id, owner_id, worker_state, reported_at, last_tick_started_at, "
        "last_tick_completed_at, restart_count, consecutive_failures, "
        "last_error_type, configured_enabled_count, enabled_count, "
        "executing_count, disabled_count, fenced_count, system_disabled_count"
    )
    await db.execute(
        f"INSERT INTO {RUNTIME_STATUS_TABLE} ({columns}) "
        f"SELECT {columns} FROM {legacy_table}"
    )
    mismatch = await db.fetchone(
        f"SELECT COUNT(*) FROM {legacy_table} legacy "
        f"LEFT JOIN {RUNTIME_STATUS_TABLE} current "
        "ON current.agent_id = legacy.agent_id "
        "AND current.owner_id = legacy.owner_id "
        "WHERE current.agent_id IS NULL"
    )
    if mismatch is None or int(mismatch[0]) != 0:
        raise RuntimeError(
            "scheduler runtime ownership migration could not verify historical rows"
        )
    await db.execute(f"DROP TABLE {legacy_table}")


def _quote_postgres_identifier(identifier: Any) -> str:
    if (
        type(identifier) is not str
        or not identifier
        or "\x00" in identifier
        or len(identifier.encode("utf-8")) > 63
        or any(ord(character) < 32 for character in identifier)
    ):
        raise RuntimeError("scheduler runtime status has an invalid primary-key name")
    return '"' + identifier.replace('"', '""') + '"'


async def _ensure_postgres_runtime_status_table(db: Any) -> None:
    await db.execute(_RUNTIME_STATUS_TABLE_SQL)

    primary_key, key_columns = await _postgres_runtime_status_primary_key(db)
    if key_columns == ["agent_id", "owner_id"]:
        return
    if primary_key is None or key_columns != ["agent_id"]:
        raise RuntimeError("scheduler runtime status has an unsupported primary key")

    # The common already-migrated path takes no table lock. A legacy-PK swap
    # is bounded so a live peer cannot hang health and telemetry indefinitely.
    await db.fetchval(
        "SELECT set_config('lock_timeout', ?, true)", ("5s",)
    )
    await db.execute(
        f"LOCK TABLE {RUNTIME_STATUS_TABLE} IN ACCESS EXCLUSIVE MODE"
    )
    primary_key, key_columns = await _postgres_runtime_status_primary_key(db)
    if key_columns != ["agent_id", "owner_id"]:
        if primary_key is None or key_columns != ["agent_id"]:
            raise RuntimeError(
                "scheduler runtime status has an unsupported primary key"
            )
        constraint = _quote_postgres_identifier(primary_key)
        await db.execute(
            f"ALTER TABLE {RUNTIME_STATUS_TABLE} DROP CONSTRAINT {constraint}"
        )
        await db.execute(
            f"ALTER TABLE {RUNTIME_STATUS_TABLE} "
            "ADD PRIMARY KEY (agent_id, owner_id)"
        )
    # A failed statement aborts the transaction and rollback clears SET LOCAL;
    # reset only on success so cleanup cannot mask the original lock failure.
    await db.fetchval(
        "SELECT set_config('lock_timeout', ?, true)", ("0",)
    )


async def _postgres_runtime_status_primary_key(
    db: Any,
) -> tuple[Optional[str], list[str]]:
    """Return the PK name and ordered columns for the runtime table."""

    primary_key = await db.fetchone(
        """SELECT con.conname FROM pg_constraint con
           WHERE con.conrelid = to_regclass(?) AND con.contype = 'p'""",
        (RUNTIME_STATUS_TABLE,),
    )
    columns = await db.fetchall(
        """SELECT attribute.attname
           FROM pg_constraint con
           JOIN unnest(con.conkey) WITH ORDINALITY AS key_column(attnum, ordinal)
             ON TRUE
           JOIN pg_attribute attribute
             ON attribute.attrelid = con.conrelid
            AND attribute.attnum = key_column.attnum
           WHERE con.conrelid = to_regclass(?) AND con.contype = 'p'
           ORDER BY key_column.ordinal""",
        (RUNTIME_STATUS_TABLE,),
    )
    key_columns = [str(row[0]) for row in columns]
    return (str(primary_key[0]) if primary_key is not None else None), key_columns


def _inventory_from_rows(
    rows: Collection[Any],
    *,
    database_now: datetime,
    default_misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
    overdue_grace_floor_seconds: int = 0,
) -> Dict[str, Any]:
    configured_enabled_count = 0
    enabled_count = 0
    executing_count = 0
    disabled_count = 0
    fenced_count = 0
    system_disabled_count = 0
    terminal_count = 0
    disabled_reasons: Dict[str, int] = {}
    next_runs: list[datetime] = []
    overdue_candidates: list[datetime] = []
    overdue_entries: list[tuple[datetime, datetime]] = []
    for row in rows:
        enabled = bool(row[0])
        disablement = classify_disablement(
            enabled=enabled,
            terminal_status=row[2],
            schedule_kind=str(row[3] or "cron"),
            database_now=database_now,
            rollout_fenced=bool(row[4]),
            claim_fenced=bool(row[5]),
            lease_owner=row[6],
            lease_expires_at=row[7],
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
        elif state == "terminal":
            terminal_count += 1
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
                if row[8] is None
                else max(0, int(row[8]))
            )
            overdue_entries.append(
                (
                    next_at
                    + timedelta(
                        seconds=max(overdue_grace_floor_seconds, task_grace)
                    ),
                    next_at,
                )
            )

    oldest_overdue = (
        min(overdue_entries, key=lambda entry: (entry[0], entry[1]))
        if overdue_entries
        else None
    )

    return {
        "schedule_count": len(rows),
        "configured_enabled_count": configured_enabled_count,
        "enabled_count": enabled_count,
        "executing_count": executing_count,
        "disabled_count": disabled_count,
        "fenced_count": fenced_count,
        "system_disabled_count": system_disabled_count,
        "terminal_count": terminal_count,
        "disabled_reasons": disabled_reasons,
        "next_run_at": min(next_runs).isoformat() if next_runs else None,
        "oldest_unclaimed_run_at": (
            min(overdue_candidates).isoformat() if overdue_candidates else None
        ),
        "oldest_unclaimed_overdue_at": (
            oldest_overdue[0].isoformat() if oldest_overdue else None
        ),
        "oldest_unclaimed_overdue_run_at": (
            oldest_overdue[1].isoformat() if oldest_overdue else None
        ),
    }


async def _schedule_inventories(
    db: Any,
    agent_ids: Collection[str],
    *,
    database_now: datetime,
    default_misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
    overdue_grace_floor_seconds: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Load one grouped inventory snapshot for a worker's whole scope."""

    normalized_ids = tuple(sorted(set(agent_ids)))
    if not normalized_ids:
        return {}
    placeholders = ", ".join("?" for _ in normalized_ids)
    rows = await db.fetchall(
        f"""
        SELECT agent_id, enabled, next_run_at, terminal_status, schedule_kind,
               scheduler_rollout_fenced, scheduler_claim_fenced,
               lease_owner, lease_expires_at, misfire_grace_seconds
        FROM scheduled_tasks
        WHERE agent_id IN ({placeholders})
        """,
        normalized_ids,
    )
    grouped: Dict[str, list[Any]] = {agent_id: [] for agent_id in normalized_ids}
    for row in rows:
        agent_id = str(row[0])
        if agent_id in grouped:
            grouped[agent_id].append(row[1:])
    return {
        agent_id: _inventory_from_rows(
            grouped[agent_id],
            database_now=database_now,
            default_misfire_grace_seconds=default_misfire_grace_seconds,
            overdue_grace_floor_seconds=overdue_grace_floor_seconds,
        )
        for agent_id in normalized_ids
    }


async def _schedule_inventory(
    db: Any,
    agent_id: str,
    *,
    database_now: datetime,
    default_misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
    overdue_grace_floor_seconds: int = 0,
) -> Dict[str, Any]:
    return (
        await _schedule_inventories(
            db,
            (agent_id,),
            database_now=database_now,
            default_misfire_grace_seconds=default_misfire_grace_seconds,
            overdue_grace_floor_seconds=overdue_grace_floor_seconds,
        )
    )[agent_id]


def scheduler_status_parameters(feature: Any) -> Dict[str, Any]:
    """Return the canonical lifecycle context for scheduler status readers.

    A standalone runner is not expected to emit telemetry until its final
    ``on_agent_ready`` arm, but that hook can fail without aborting startup.
    Before arm, grace is therefore finite from feature initialization. A
    successful arm replaces the anchor with that runner lifecycle transition.
    Host-managed features have no local runner, so their feature initialization
    timestamp remains the local lifecycle anchor.
    """

    runner = getattr(feature, "_runner", None)
    poll_interval = max(
        1, int(getattr(runner, "_poll_interval", 30) or 30)
    )
    grace_seconds = poll_interval * DEFAULT_STALE_AFTER_TICKS
    expected_owner_id = getattr(runner, "_owner_id", None)
    expected_started_at = getattr(runner, "_arm_requested_at", None)

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
    lease_seconds = getattr(runner, "_lease_seconds", None)
    if lease_seconds is None:
        load_lease_seconds = getattr(feature, "_load_lease_seconds", None)
        lease_seconds = (
            load_lease_seconds()
            if callable(load_lease_seconds)
            else DEFAULT_LEASE_SECONDS
        )

    return {
        "poll_interval": poll_interval,
        "startup_grace_remaining": startup_grace_remaining,
        "expected_owner_id": expected_owner_id,
        "expected_started_at": expected_started_at,
        "default_misfire_grace_seconds": max(
            0, int(default_misfire_grace or 0)
        ),
        "lease_seconds": max(1, int(lease_seconds or 0)),
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

    normalized_ids = tuple(sorted(set(agent_ids)))
    if not normalized_ids:
        return
    database_now = await scheduler_database_clock(db)
    reported_at_sql = scheduler_database_now_sql(db)
    inventories = await _schedule_inventories(
        db, normalized_ids, database_now=database_now
    )
    placeholders = ", ".join("?" for _ in normalized_ids)
    retention_cutoff = (
        database_now - timedelta(seconds=RUNTIME_STATUS_RETENTION_SECONDS)
    ).isoformat()
    await db.execute(
        f"DELETE FROM {RUNTIME_STATUS_TABLE} "
        f"WHERE agent_id IN ({placeholders}) AND reported_at < ?",
        (*normalized_ids, retention_cutoff),
    )
    upsert_sql = f"""
        INSERT INTO {RUNTIME_STATUS_TABLE}
            (agent_id, owner_id, worker_state, reported_at,
             last_tick_started_at, last_tick_completed_at, restart_count,
             consecutive_failures, last_error_type,
             configured_enabled_count, enabled_count, executing_count,
             disabled_count, fenced_count, system_disabled_count)
        VALUES (?, ?, ?, {reported_at_sql}, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(agent_id, owner_id) DO UPDATE SET
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
    """
    params_list = []
    for agent_id in normalized_ids:
        inventory = inventories[agent_id]
        params_list.append((
            agent_id,
            owner_id,
            worker_state,
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
        ))
    await db.execute_many(upsert_sql, params_list)


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
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> Dict[str, Any]:
    """Read and diagnose scheduler liveness plus current durable row state."""

    stale_after_seconds = max(1, int(poll_interval)) * max(
        1, int(stale_after_ticks)
    )
    database_now = await scheduler_database_clock(db)
    inventory = await _schedule_inventory(
        db,
        agent_id,
        database_now=database_now,
        default_misfire_grace_seconds=max(
            0, int(default_misfire_grace_seconds)
        ),
        overdue_grace_floor_seconds=stale_after_seconds,
    )
    report_columns = (
        "owner_id, worker_state, reported_at, last_tick_started_at, "
        "last_tick_completed_at, restart_count, consecutive_failures, "
        "last_error_type, configured_enabled_count, enabled_count, "
        "executing_count, disabled_count, fenced_count, system_disabled_count"
    )
    freshness_cutoff = (
        database_now - timedelta(seconds=stale_after_seconds)
    ).isoformat()
    owner_predicate = " AND owner_id = ?" if expected_owner_id else ""
    owner_params: tuple[Any, ...] = (
        (expected_owner_id,) if expected_owner_id else ()
    )
    reports = await db.fetchall(
        f"SELECT {report_columns} FROM {RUNTIME_STATUS_TABLE} "
        f"WHERE agent_id = ?{owner_predicate} AND reported_at >= ? "
        "ORDER BY reported_at DESC",
        (agent_id, *owner_params, freshness_cutoff),
    )
    if not reports:
        latest = await db.fetchone(
            f"SELECT {report_columns} FROM {RUNTIME_STATUS_TABLE} "
            f"WHERE agent_id = ?{owner_predicate} "
            "ORDER BY reported_at DESC LIMIT 1",
            (agent_id, *owner_params),
        )
        reports = [latest] if latest is not None else []
    if not reports and expected_owner_id:
        # One bounded prior-owner sample distinguishes a never-reported agent
        # from a restart that is still awaiting its new owner's first report.
        prior_owner = await db.fetchone(
            f"SELECT {report_columns} FROM {RUNTIME_STATUS_TABLE} "
            "WHERE agent_id = ? ORDER BY reported_at DESC LIMIT 1",
            (agent_id,),
        )
        reports = [prior_owner] if prior_owner is not None else []

    def reported_at(report: Any) -> Optional[datetime]:
        return _parse_utc(report[2])

    def report_age(report: Any) -> Optional[float]:
        stamp = reported_at(report)
        return (
            max(0.0, (database_now - stamp).total_seconds())
            if stamp is not None
            else None
        )

    def newest(rows: list[Any]) -> Any:
        return max(
            rows,
            key=lambda row: reported_at(row) or datetime.min.replace(
                tzinfo=timezone.utc
            ),
        )

    expected_start = _parse_utc(expected_started_at)
    owner_reports = [
        report for report in reports
        if not expected_owner_id or str(report[0] or "") == expected_owner_id
    ]
    current_reports = [
        report for report in owner_reports
        if expected_start is None
        or (
            reported_at(report) is not None
            and reported_at(report) >= expected_start
        )
    ]
    fresh_reports = [
        report for report in current_reports
        if report_age(report) is not None
        and report_age(report) <= stale_after_seconds
    ]
    running_reports = [
        report for report in fresh_reports
        if str(report[1] or "unknown") == "running"
    ]

    details: Dict[str, Any] = {
        **inventory,
        "telemetry_received": bool(reports),
        "current_owner_telemetry_received": bool(current_reports),
        "stale_after_seconds": stale_after_seconds,
        "worker_reports_received": len(reports),
        "fresh_worker_count": len(fresh_reports),
        "running_worker_count": len(running_reports),
        "fresh_non_running_worker_count": (
            len(fresh_reports) - len(running_reports)
        ),
        "stale_worker_count": len(current_reports) - len(fresh_reports),
    }
    if not reports:
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

    selected_report = (
        newest(running_reports)
        if running_reports
        else (
            newest(fresh_reports)
            if fresh_reports
            else (
                newest(current_reports)
                if current_reports
                else newest(owner_reports if owner_reports else list(reports))
            )
        )
    )
    report_owner_id = str(selected_report[0] or "")
    selected_report_age = report_age(selected_report)
    worker_state = str(selected_report[1] or "unknown")
    report_predates_expected_start = bool(
        expected_start is not None
        and owner_reports
        and not current_reports
    )
    details.update({
        "state": "running",
        "owner_id": report_owner_id,
        "expected_owner_id": expected_owner_id,
        "expected_started_at": expected_started_at,
        "report_predates_expected_start": report_predates_expected_start,
        "current_owner_telemetry_received": bool(current_reports),
        "worker_state": worker_state,
        "reported_at": selected_report[2],
        "report_age_seconds": (
            round(selected_report_age, 3)
            if selected_report_age is not None
            else None
        ),
        "last_tick_started_at": selected_report[3],
        "last_tick_completed_at": selected_report[4],
        "restart_count": int(selected_report[5] or 0),
        "consecutive_failures": int(selected_report[6] or 0),
        "last_error_type": selected_report[7],
        "reported_configured_enabled_count": int(selected_report[8] or 0),
        "reported_enabled_count": int(selected_report[9] or 0),
        "reported_executing_count": int(selected_report[10] or 0),
        "reported_disabled_count": int(selected_report[11] or 0),
        "reported_fenced_count": int(selected_report[12] or 0),
        "reported_system_disabled_count": int(selected_report[13] or 0),
    })

    tick_in_progress_limit_seconds = stale_after_seconds + max(
        1, int(lease_seconds)
    )

    def tick_in_progress(report: Any) -> bool:
        tick_started = _parse_utc(report[3])
        tick_completed = _parse_utc(report[4])
        tick_age = (
            (database_now - tick_started).total_seconds()
            if tick_started is not None
            else None
        )
        return bool(
            tick_started is not None
            and tick_age is not None
            and 0 <= tick_age <= tick_in_progress_limit_seconds
            and (tick_completed is None or tick_started > tick_completed)
        )

    details["tick_in_progress"] = any(
        tick_in_progress(report) for report in running_reports
    )
    details["tick_in_progress_limit_seconds"] = (
        tick_in_progress_limit_seconds
    )

    if expected_owner_id and not current_reports:
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
    if not fresh_reports:
        details["state"] = (
            "awaiting_telemetry" if prior_owner_grace else "stale"
        )
        details["status"] = "warn" if prior_owner_grace else "fail"
        return details
    if not running_reports and worker_state == "starting":
        details["state"] = (
            "awaiting_telemetry"
            if startup_grace_remaining > 0
            else "starting"
        )
        details["status"] = "warn" if startup_grace_remaining > 0 else "fail"
        return details
    if not running_reports:
        details["state"] = (
            "awaiting_telemetry" if prior_owner_grace else worker_state
        )
        details["status"] = "warn" if prior_owner_grace else "fail"
        return details

    oldest = _parse_utc(inventory["oldest_unclaimed_overdue_run_at"])
    overdue_at = _parse_utc(inventory["oldest_unclaimed_overdue_at"])
    # A current tick owns every row selected in its due batch, including rows
    # queued behind the concurrency semaphore but not yet claimed. Those rows
    # are healthy in-flight work, not evidence that polling stopped.
    if (
        not details["tick_in_progress"]
        and oldest is not None
        and overdue_at is not None
        and overdue_at < database_now
    ):
        details["state"] = "overdue"
        details["overdue_seconds"] = round(
            (database_now - oldest).total_seconds(), 3
        )
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
        elif inventory["terminal_count"]:
            details["state"] = "running_only_terminal_schedules"
        else:
            details["state"] = "running_zero_schedules"
    details["status"] = "pass"
    return details


__all__ = [
    "DEFAULT_MISFIRE_GRACE_SECONDS",
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_STALE_AFTER_TICKS",
    "RUNTIME_STATUS_TABLE",
    "RUNTIME_STATUS_RETENTION_SECONDS",
    "SCHEDULER_SAFETY_DISABLEMENT_REASONS",
    "classify_disablement",
    "emit_runtime_status",
    "ensure_runtime_status_table",
    "scheduler_status",
    "scheduler_status_parameters",
]
