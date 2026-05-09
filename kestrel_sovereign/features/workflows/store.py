"""Workflow storage (Phase 0 chunk B).

Implements the three workflow-only tables called out in §5 of
``docs/architecture/WORKFLOWS_FEATURE_DESIGN.md`` (v4.1):

- ``workflow_definitions`` — versioned, signed specs.
- ``workflow_runs`` — one row per execution.
- ``workflow_stage_links`` — join row to ``signal_log.id`` plus
  workflow-only fields (gate outcome, compensate state, actor signature).

Everything else (durable signal dispatch, redacted payloads, retention,
locks, causation) lives in the dispatcher's existing infrastructure. We
do NOT duplicate signal_log here.

SQL dialect: this file uses :class:`UnifiedStoreBase`'s dialect helpers
exclusively (``timestamp_type``, ``json_type``, ``boolean_type``,
``now_default``, ``interval_days``, ``to_timestamp_param``,
``to_bool_param``). That keeps SQLite parity automatic — no per-table
SQLite migration code; the same DDL works against both backends. This
matches the pattern proven by ``signal_log`` and the unified A2A stores.

Phase 0 deliverable: schema only — no read/write helpers beyond
``initialize`` and ``purge_expired`` (the retention sweep). Tool surface
and runner-driven inserts/updates land in Phase 1.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.features.workflows.models import WorkflowSpec
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)

# Bounded batch size for retention sweep deletes — well under both
# SQLite's SQLITE_MAX_VARIABLE_NUMBER (≥999, modern default 32766) and
# Postgres' 65535 bind-parameter cap. 500 also keeps the per-batch
# transaction small enough that a long-running sweep stays
# preemptable.
_PURGE_BATCH_SIZE = 500


class WorkflowStore(UnifiedStoreBase):
    """Backend-agnostic persistence for workflow definitions, runs, and
    stage-link rows.

    Phase 0 ships only the migration surface: ``initialize()`` creates
    the three tables and indexes idempotently. Phase 1 adds:

    - ``put_definition(spec)`` / ``get_definition(name, version)`` /
      ``revoke_definition(...)``.
    - Run lifecycle: ``insert_run(run)``, ``update_run_status(...)``,
      ``set_cancel_barrier(...)``, ``mark_finished(...)``.
    - Stage-link lifecycle: ``insert_stage_link(link)``,
      ``update_gate_outcome(...)``, ``update_compensate_state(...)``.
    - ``purge_expired_runs(now)`` — retention sweep, mirroring
      ``SignalLogStore.purge_expired``.

    Splitting the layers this way keeps Phase 0 reviewable on its own:
    the migration is correct against the design doc and the dataclass
    contract before any business logic runs over it.
    """

    DEFINITIONS_TABLE = "workflow_definitions"
    RUNS_TABLE = "workflow_runs"
    STAGE_LINKS_TABLE = "workflow_stage_links"

    def __init__(self, backend: DatabaseBackend):
        super().__init__(backend)

    def _bool_default(self, value: bool) -> str:
        """Dialect-portable DDL boolean default.

        Codex round-2 P2: Postgres' ``BOOLEAN DEFAULT 0`` is a syntax
        error; SQLite stores booleans as INTEGER and uses ``0``/``1``.
        The base ``UnifiedStoreBase`` exposes value-side helpers
        (``to_bool_param``) but not a DDL-side helper, so we keep the
        literal mapping local here.
        """
        if self.is_postgres:
            return "TRUE" if value else "FALSE"
        return "1" if value else "0"

    async def initialize(self) -> None:
        await self._create_definitions_table()
        await self._create_runs_table()
        await self._create_stage_links_table()
        logger.info(
            "WorkflowStore initialized (%s)", self._backend.backend_type
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def _create_definitions_table(self) -> None:
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()

        # Design §5: ``spec_json`` is JSONB on Postgres, TEXT on SQLite
        # (caller writes JSON-validated text via ``json.dumps`` on the
        # canonical payload). Compound primary key (name, version) so a
        # workflow can have multiple versions; ``deleted_at`` marks
        # revoked rows without losing audit trail (in-flight runs that
        # pinned to a revoked version still have a row to dereference).
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS {self.DEFINITIONS_TABLE} (
                name              TEXT NOT NULL,
                version           INTEGER NOT NULL,
                spec_json         {json_type} NOT NULL,
                spec_hash         TEXT NOT NULL,
                author_did        TEXT NOT NULL,
                author_sig        TEXT NOT NULL,
                retention_days    INTEGER,
                created_at        {ts_type} NOT NULL {ts_default},
                deleted_at        {ts_type},
                revocation_reason TEXT,
                PRIMARY KEY (name, version)
            )
        """)
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.DEFINITIONS_TABLE}_author "
            f"ON {self.DEFINITIONS_TABLE}(author_did, created_at)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.DEFINITIONS_TABLE}_active "
            f"ON {self.DEFINITIONS_TABLE}(name, deleted_at) "
            f"WHERE deleted_at IS NULL"
        )

    async def _create_runs_table(self) -> None:
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        bool_type = self.boolean_type()
        # ``params_json`` and ``current_stages_json`` are TEXT-validated
        # JSON on both backends because SQLite has no JSONB and we need
        # round-trippable parsing on read in Phase 1. ``status`` is
        # checked at the dataclass boundary, not the database boundary —
        # we do not add a CHECK constraint here because adding new
        # statuses in a future Phase would require a migration; the
        # closed vocabulary lives in code, not in the DDL.
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS {self.RUNS_TABLE} (
                run_id                    TEXT PRIMARY KEY,
                workflow_name             TEXT NOT NULL,
                workflow_ver              INTEGER NOT NULL,
                parent_run_id             TEXT,
                params_json               TEXT NOT NULL,
                status                    TEXT NOT NULL,
                current_stages_json       TEXT NOT NULL DEFAULT '[]',
                cancel_barrier_at         {ts_type},
                started_by_did            TEXT NOT NULL,
                scheduler_task_id         TEXT,
                signature_post_revocation {bool_type} NOT NULL DEFAULT {self._bool_default(False)},
                started_at                {ts_type} NOT NULL {ts_default},
                finished_at               {ts_type},
                deleted_at                {ts_type},
                FOREIGN KEY (parent_run_id) REFERENCES {self.RUNS_TABLE}(run_id)
                    ON DELETE SET NULL
            )
        """)
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.RUNS_TABLE}_workflow "
            f"ON {self.RUNS_TABLE}(workflow_name, workflow_ver, status)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.RUNS_TABLE}_status_started "
            f"ON {self.RUNS_TABLE}(status, started_at)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.RUNS_TABLE}_parent "
            f"ON {self.RUNS_TABLE}(parent_run_id) "
            f"WHERE parent_run_id IS NOT NULL"
        )
        # Run history per workflow excluding deleted rows — supports the
        # ``workflow_history`` tool's per-workflow scan.
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.RUNS_TABLE}_history "
            f"ON {self.RUNS_TABLE}(workflow_name, finished_at) "
            f"WHERE deleted_at IS NULL"
        )
        # Cron-triggered runs need a fast lookup by scheduler_task_id
        # for the resume path (design §3 ``WorkflowRun``).
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.RUNS_TABLE}_scheduler "
            f"ON {self.RUNS_TABLE}(scheduler_task_id) "
            f"WHERE scheduler_task_id IS NOT NULL"
        )

    async def _create_stage_links_table(self) -> None:
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        bool_type = self.boolean_type()
        # ``signal_id`` is intentionally NOT a hard FK to signal_log.id
        # (design §8 Open Q4 leans soft reference): signal_log retention
        # can outlive or under-live workflow runs, and a hard FK with
        # cascade would make a signal_log purge fail when a stage link
        # row references the to-be-deleted signal. We index it instead.
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS {self.STAGE_LINKS_TABLE} (
                link_id           TEXT PRIMARY KEY,
                run_id            TEXT NOT NULL,
                stage_name        TEXT NOT NULL,
                attempt_number    INTEGER NOT NULL,
                signal_id         TEXT,
                idempotency_key   TEXT NOT NULL,
                gate_outcome      TEXT,
                gate_reason       TEXT,
                compensate_state  TEXT,
                post_cancel       {bool_type} NOT NULL DEFAULT {self._bool_default(False)},
                actor_did         TEXT NOT NULL,
                actor_sig         TEXT NOT NULL,
                occurred_at       {ts_type} NOT NULL {ts_default},
                FOREIGN KEY (run_id) REFERENCES {self.RUNS_TABLE}(run_id)
                    ON DELETE CASCADE,
                UNIQUE (run_id, stage_name, attempt_number)
            )
        """)
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS "
            f"idx_{self.STAGE_LINKS_TABLE}_run_occurred "
            f"ON {self.STAGE_LINKS_TABLE}(run_id, occurred_at)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS "
            f"idx_{self.STAGE_LINKS_TABLE}_signal "
            f"ON {self.STAGE_LINKS_TABLE}(signal_id) "
            f"WHERE signal_id IS NOT NULL"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS "
            f"idx_{self.STAGE_LINKS_TABLE}_actor "
            f"ON {self.STAGE_LINKS_TABLE}(actor_did, occurred_at)"
        )
        # Open gates / failures dashboard — partial index keeps it
        # small (most rows complete and don't need the lookup).
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS "
            f"idx_{self.STAGE_LINKS_TABLE}_open_gates "
            f"ON {self.STAGE_LINKS_TABLE}(gate_outcome) "
            f"WHERE gate_outcome IN ('fail', 'pending')"
        )
        # Idempotency-key uniqueness is implied by the
        # (run_id, stage_name, attempt_number) UNIQUE constraint above
        # because the design's idempotency_key is derived from those
        # three plus an engine nonce — the row-level uniqueness already
        # blocks duplicate inserts. A separate UNIQUE on idempotency_key
        # alone would be redundant and breaks the legitimate "same
        # stage runs again with a fresh attempt_number" case.

    # ------------------------------------------------------------------
    # Phase 0 surface: minimal helpers needed for the migration tests
    # ------------------------------------------------------------------

    async def insert_definition_for_test(
        self,
        spec: WorkflowSpec,
    ) -> None:
        """Inserts a row directly from a :class:`WorkflowSpec`.

        Phase 0 surface point — Phase 1 wraps this in a tool that adds
        signature verification, retention-policy resolution, and a
        write lock. Exposed today so the migration can be exercised
        end-to-end in unit tests without running the full feature.
        """
        if not spec.spec_hash or not spec.author_sig:
            raise ValueError(
                "WorkflowStore: cannot persist an unsigned spec "
                "(missing spec_hash or author_sig)"
            )
        await self._backend.execute(
            f"""
            INSERT INTO {self.DEFINITIONS_TABLE}
                (name, version, spec_json, spec_hash, author_did,
                 author_sig, retention_days)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.name,
                spec.version,
                json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")),
                spec.spec_hash,
                spec.author_did,
                spec.author_sig,
                spec.retention_days,
            ),
        )

    async def get_definition_row(
        self,
        name: str,
        version: int,
    ) -> Optional[dict[str, Any]]:
        row = await self._backend.fetch_one(
            f"""
            SELECT name, version, spec_json, spec_hash, author_did,
                   author_sig, retention_days, created_at, deleted_at,
                   revocation_reason
            FROM {self.DEFINITIONS_TABLE}
            WHERE name = ? AND version = ?
            """,
            (name, version),
        )
        if row is None:
            return None
        return {
            "name": row[0],
            "version": row[1],
            "spec_json": row[2],
            "spec_hash": row[3],
            "author_did": row[4],
            "author_sig": row[5],
            "retention_days": row[6],
            "created_at": self.from_timestamp_field(row[7]),
            "deleted_at": self.from_timestamp_field(row[8]),
            "revocation_reason": row[9],
        }

    async def purge_expired_runs(
        self, *, now: Optional[datetime] = None
    ) -> int:
        """Delete finished workflow runs past their definition's
        retention window.

        Codex round-1 P1: this used to compare ``finished_at < now``,
        which deleted every finished run with a retention policy
        immediately on the next sweep regardless of ``retention_days``.
        Per-row interval arithmetic differs between Postgres
        (``r.finished_at + interval``) and SQLite (``datetime(...,
        '+N days')``), so we resolve the cutoff per row in Python and
        bulk-delete the IDs that have actually expired. The retention
        sweep is a cron task — not a hot path — so the extra round
        trip is fine and the dialect-portability win is the more
        important property.

        Stage links cascade via ON DELETE CASCADE on the FK, so this
        also tears down the per-stage history rows for every purged
        run. (Codex round-1 P2.)

        Returns the number of run rows purged.
        """
        cutoff_now = now if now is not None else datetime.now(timezone.utc)
        rows = await self._backend.fetch_all(
            f"""
            SELECT r.run_id, r.finished_at, d.retention_days
            FROM {self.RUNS_TABLE} r
            JOIN {self.DEFINITIONS_TABLE} d
              ON d.name = r.workflow_name
             AND d.version = r.workflow_ver
            WHERE r.finished_at IS NOT NULL
              AND d.retention_days IS NOT NULL
            """
        )

        expired: list[str] = []
        for run_id, finished_at_value, retention_days in rows:
            finished_at = self.from_timestamp_field(finished_at_value)
            if finished_at is None:
                continue
            # ``finished_at`` may be naive on SQLite (we store ISO
            # without tz); promote to UTC so comparison is correct.
            if finished_at.tzinfo is None:
                finished_at = finished_at.replace(tzinfo=timezone.utc)
            expiry = finished_at + timedelta(days=int(retention_days))
            if expiry < cutoff_now:
                expired.append(run_id)

        if not expired:
            return 0

        # Codex round-3 P2: a single unbounded ``IN (?, ?, ...)`` delete
        # would hit SQLite's SQLITE_MAX_VARIABLE_NUMBER (≥999, often
        # 32766 on modern builds) or Postgres' 65535 bind-parameter
        # cap once enough runs accumulate, leaving the entire expired
        # history unpurged. Batch in chunks well under both limits so
        # the sweep makes forward progress on any backend.
        purged_total = 0
        for batch_start in range(0, len(expired), _PURGE_BATCH_SIZE):
            batch = expired[batch_start : batch_start + _PURGE_BATCH_SIZE]
            placeholders = ", ".join("?" * len(batch))
            purged_total += await self._backend.execute(
                f"DELETE FROM {self.RUNS_TABLE} "
                f"WHERE run_id IN ({placeholders})",
                tuple(batch),
            )
        return purged_total
