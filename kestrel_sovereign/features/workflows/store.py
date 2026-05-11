"""Workflow storage for the Workflows feature.

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

Phase 0 shipped the schema and retention sweep. Phase 1 adds the
runner-facing definition/run/stage-link helpers below while keeping the
same dialect-helper discipline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.features.workflows.models import (
    GateOutcome,
    RevocationReason,
    RunStatus,
    StageLink,
    WorkflowRun,
    WorkflowSpec,
    _DID_RE,
)
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

    ``initialize()`` creates the three tables and indexes idempotently;
    the remaining methods are the runner/tool persistence surface:

    - ``put_definition(spec)`` / ``get_definition(name, version)`` /
      ``revoke_definition(...)``.
    - Run lifecycle: ``insert_run(run)``, ``get_run(...)``,
      ``update_run_status(...)``, ``set_cancel_barrier(...)``.
    - Stage-link lifecycle: ``insert_stage_link(link)``,
      ``update_stage_link_transition(...)``,
      ``update_compensate_state(...)``.
    - ``purge_expired_runs(now)`` — retention sweep.
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
                revocation_authority_did TEXT,
                revocation_authority_sig TEXT,
                PRIMARY KEY (name, version)
            )
        """)
        for ddl in (
            f"ALTER TABLE {self.DEFINITIONS_TABLE} "
            "ADD COLUMN revocation_authority_did TEXT",
            f"ALTER TABLE {self.DEFINITIONS_TABLE} "
            "ADD COLUMN revocation_authority_sig TEXT",
        ):
            try:
                await self._backend.execute(ddl)
            except Exception as exc:  # noqa: BLE001 - duplicate column is benign
                if (
                    "duplicate" not in str(exc).lower()
                    and "exists" not in str(exc).lower()
                ):
                    raise
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
                engine_nonce              TEXT NOT NULL,
                current_stages_json       TEXT NOT NULL DEFAULT '[]',
                cancel_barrier_at         {ts_type},
                started_by_did            TEXT NOT NULL,
                scheduler_task_id         TEXT,
                signature_post_revocation {bool_type} NOT NULL DEFAULT {self._bool_default(False)},
                started_at                {ts_type} NOT NULL {ts_default},
                finished_at               {ts_type},
                deleted_at                {ts_type},
                FOREIGN KEY (parent_run_id) REFERENCES {self.RUNS_TABLE}(run_id)
                    ON DELETE SET NULL,
                -- Codex chunk-D round-3 P2: composite FK to the signed
                -- definition keeps in-flight runs joinable to their
                -- pinned spec even after revocation. ``ON DELETE NO
                -- ACTION`` is the safe choice — definitions are soft-
                -- deleted (deleted_at) rather than hard-deleted, so a
                -- run row can never become orphaned by a revoke.
                FOREIGN KEY (workflow_name, workflow_ver)
                    REFERENCES {self.DEFINITIONS_TABLE}(name, version)
                    ON DELETE NO ACTION
            )
        """)
        # Phase 1 adds ``engine_nonce`` for auditable idempotency-key
        # derivation. Existing Phase 0 databases created the table
        # without this column, so initialize() carries the lightweight
        # additive migration too. The all-zero default only backfills
        # pre-run-start rows; new runner-created rows always store a
        # fresh 16-byte nonce.
        try:
            await self._backend.execute(
                f"ALTER TABLE {self.RUNS_TABLE} "
                "ADD COLUMN engine_nonce TEXT NOT NULL "
                "DEFAULT '00000000000000000000000000000000'"
            )
        except Exception as exc:  # noqa: BLE001 - duplicate column is benign
            if "duplicate" not in str(exc).lower() and "exists" not in str(exc).lower():
                raise
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
                forced            {bool_type} NOT NULL DEFAULT {self._bool_default(False)},
                actor_did         TEXT NOT NULL,
                actor_sig         TEXT NOT NULL,
                occurred_at       {ts_type} NOT NULL {ts_default},
                FOREIGN KEY (run_id) REFERENCES {self.RUNS_TABLE}(run_id)
                    ON DELETE CASCADE,
                UNIQUE (run_id, stage_name, attempt_number),
                -- Codex chunk-D round-3 P2: idempotency_key uniqueness
                -- is a defense-in-depth correctness check. The
                -- (run_id, stage_name, attempt_number) UNIQUE above
                -- *implies* idempotency_key uniqueness GIVEN the
                -- design's derivation formula (sha256(run_id||stage||
                -- sha256(input||attempt||nonce))). Adding the explicit
                -- UNIQUE catches a runner bug that produces the same
                -- key for different (run, stage, attempt) tuples —
                -- exactly the kind of corruption that would silently
                -- collapse two distinct attempts into one dispatch.
                UNIQUE (idempotency_key)
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
        # idempotency_key UNIQUE constraint is declared on the table
        # itself (see above). It catches the rare runner-bug shape
        # where two different (run_id, stage_name, attempt_number)
        # tuples accidentally derive the same idempotency_key.
        try:
            await self._backend.execute(
                f"ALTER TABLE {self.STAGE_LINKS_TABLE} "
                f"ADD COLUMN forced {bool_type} NOT NULL "
                f"DEFAULT {self._bool_default(False)}"
            )
        except Exception as exc:  # noqa: BLE001 - duplicate column is benign
            if "duplicate" not in str(exc).lower() and "exists" not in str(exc).lower():
                raise

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
                   revocation_reason, revocation_authority_did,
                   revocation_authority_sig
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
            "revocation_authority_did": row[10],
            "revocation_authority_sig": row[11],
        }

    async def put_definition(self, spec: WorkflowSpec) -> None:
        """Persist a signed workflow definition.

        Phase 1's feature/tool layer performs signature verification
        before calling this. The store still refuses unsigned drafts so
        direct callers cannot bypass the signed-artifact invariant.
        """
        await self.insert_definition_for_test(spec)

    async def get_definition(
        self, name: str, version: int
    ) -> Optional[WorkflowSpec]:
        row = await self.get_definition_row(name, version)
        if row is None:
            return None
        return WorkflowSpec.from_dict(json.loads(row["spec_json"]))

    async def get_latest_definition(self, name: str) -> Optional[WorkflowSpec]:
        row = await self._backend.fetch_one(
            f"""
            SELECT spec_json
            FROM {self.DEFINITIONS_TABLE}
            WHERE name = ? AND deleted_at IS NULL
            ORDER BY version DESC
            LIMIT 1
            """,
            (name,),
        )
        if row is None:
            return None
        return WorkflowSpec.from_dict(json.loads(row[0]))

    async def list_definitions(self) -> list[dict[str, Any]]:
        rows = await self._backend.fetch_all(
            f"""
            SELECT name, version, spec_hash, author_did, retention_days,
                   created_at, deleted_at, revocation_reason,
                   revocation_authority_did, revocation_authority_sig
            FROM {self.DEFINITIONS_TABLE}
            ORDER BY name, version DESC
            """
        )
        return [
            {
                "name": row[0],
                "version": row[1],
                "spec_hash": row[2],
                "author_did": row[3],
                "retention_days": row[4],
                "created_at": self.from_timestamp_field(row[5]),
                "deleted_at": self.from_timestamp_field(row[6]),
                "revocation_reason": row[7],
                "revocation_authority_did": row[8],
                "revocation_authority_sig": row[9],
            }
            for row in rows
        ]

    def _validate_revocation_authority(
        self,
        authority_did: str,
        authority_sig: str,
    ) -> None:
        if not isinstance(authority_did, str) or not _DID_RE.fullmatch(
            authority_did
        ):
            raise ValueError("revocation authority DID must be a valid DID")
        if not isinstance(authority_sig, str) or not authority_sig:
            raise ValueError("revocation authority signature must be a non-empty string")

    async def revoke_definition(
        self,
        name: str,
        version: int,
        *,
        reason: RevocationReason | str,
        authority_did: str,
        authority_sig: str,
        revoked_at: Optional[datetime] = None,
    ) -> bool:
        reason_value = RevocationReason(reason).value
        self._validate_revocation_authority(authority_did, authority_sig)
        when = revoked_at or datetime.now(timezone.utc)
        changed = await self._backend.execute(
            f"""
            UPDATE {self.DEFINITIONS_TABLE}
            SET deleted_at = ?,
                revocation_reason = ?,
                revocation_authority_did = ?,
                revocation_authority_sig = ?
            WHERE name = ? AND version = ?
              AND (
                deleted_at IS NULL
                OR (
                  ? = 'compromised'
                  AND (
                    revocation_reason IS NULL
                    OR revocation_reason != 'compromised'
                  )
                )
              )
            """,
            (
                self.to_timestamp_param(when),
                reason_value,
                authority_did,
                authority_sig,
                name,
                version,
                reason_value,
            ),
        )
        return changed > 0

    async def list_runs_for_definition(
        self,
        name: str,
        version: int,
        *,
        statuses: Optional[set[RunStatus] | tuple[RunStatus, ...]] = None,
    ) -> list[WorkflowRun]:
        filters = ["workflow_name = ?", "workflow_ver = ?"]
        params: list[Any] = [name, version]
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            filters.append(f"status IN ({placeholders})")
            params.extend(status.value for status in statuses)
        rows = await self._backend.fetch_all(
            f"""
            SELECT run_id
            FROM {self.RUNS_TABLE}
            WHERE {' AND '.join(filters)}
            ORDER BY started_at ASC
            """,
            tuple(params),
        )
        runs: list[WorkflowRun] = []
        for row in rows:
            run = await self.get_run(row[0])
            if run is not None:
                runs.append(run)
        return runs

    async def insert_run(self, run: WorkflowRun) -> None:
        await self._backend.execute(
            f"""
            INSERT INTO {self.RUNS_TABLE}
                (run_id, workflow_name, workflow_ver, parent_run_id,
                 params_json, status, engine_nonce, current_stages_json,
                 cancel_barrier_at, started_by_did, scheduler_task_id,
                 signature_post_revocation, started_at, finished_at, deleted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.workflow_name,
                run.workflow_ver,
                run.parent_run_id,
                json.dumps(
                    run.to_dict()["params"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                run.status.value,
                run.engine_nonce,
                json.dumps(list(run.current_stages)),
                self.to_timestamp_param(run.cancel_barrier_at),
                run.started_by_did,
                run.scheduler_task_id,
                self.to_bool_param(run.signature_post_revocation),
                self.to_timestamp_param(run.started_at or datetime.now(timezone.utc)),
                self.to_timestamp_param(run.finished_at),
                self.to_timestamp_param(run.deleted_at),
            ),
        )

    async def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        row = await self._backend.fetch_one(
            f"""
            SELECT run_id, workflow_name, workflow_ver, parent_run_id,
                   params_json, status, engine_nonce, current_stages_json,
                   cancel_barrier_at, started_by_did, scheduler_task_id,
                   signature_post_revocation, started_at, finished_at,
                   deleted_at
            FROM {self.RUNS_TABLE}
            WHERE run_id = ?
            """,
            (run_id,),
        )
        if row is None:
            return None
        return WorkflowRun(
            run_id=row[0],
            workflow_name=row[1],
            workflow_ver=row[2],
            parent_run_id=row[3],
            params=json.loads(row[4]),
            status=row[5],
            engine_nonce=row[6],
            current_stages=json.loads(row[7]),
            cancel_barrier_at=self.from_timestamp_field(row[8]),
            started_by_did=row[9],
            scheduler_task_id=row[10],
            signature_post_revocation=bool(row[11]),
            started_at=self.from_timestamp_field(row[12]),
            finished_at=self.from_timestamp_field(row[13]),
            deleted_at=self.from_timestamp_field(row[14]),
        )

    async def list_runs(
        self,
        *,
        workflow_name: Optional[str] = None,
        status: Optional[RunStatus | str] = None,
        limit: int = 50,
    ) -> list[WorkflowRun]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        filters: list[str] = []
        params: list[Any] = []
        if workflow_name is not None:
            filters.append("workflow_name = ?")
            params.append(workflow_name)
        if status is not None:
            filters.append("status = ?")
            params.append(RunStatus(status).value)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        params.append(limit)
        rows = await self._backend.fetch_all(
            f"""
            SELECT run_id
            FROM {self.RUNS_TABLE}
            {where}
            ORDER BY started_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        runs: list[WorkflowRun] = []
        for row in rows:
            run = await self.get_run(row[0])
            if run is not None:
                runs.append(run)
        return runs

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        current_stages: Optional[list[str]] = None,
        finished_at: Optional[datetime] = None,
        clear_finished_at: bool = False,
        if_not_terminal: bool = False,
    ) -> bool:
        parsed_status = RunStatus(status)
        fields = ["status = ?"]
        params: list[Any] = [parsed_status.value]
        if current_stages is not None:
            fields.append("current_stages_json = ?")
            params.append(json.dumps(current_stages))
        if clear_finished_at and finished_at is not None:
            raise ValueError("clear_finished_at and finished_at are mutually exclusive")
        if clear_finished_at:
            fields.append("finished_at = NULL")
        if finished_at is not None:
            fields.append("finished_at = ?")
            params.append(self.to_timestamp_param(finished_at))
        where_clause = "WHERE run_id = ?"
        params.append(run_id)
        if if_not_terminal:
            where_clause += " AND status NOT IN (?, ?, ?, ?)"
            params.extend(
                [
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                    RunStatus.CANCELLED_WITH_IRREVERSIBLE_RESIDUE.value,
                ]
            )
        changed = await self._backend.execute(
            f"UPDATE {self.RUNS_TABLE} SET {', '.join(fields)} {where_clause}",
            tuple(params),
        )
        return changed > 0

    async def merge_run_params(
        self,
        run_id: str,
        updates: dict[str, Any],
        *,
        if_not_terminal: bool = False,
    ) -> bool:
        if not isinstance(updates, dict):
            raise ValueError("workflow run param updates must be an object")
        run = await self.get_run(run_id)
        if run is None:
            return False
        merged = {**run.to_dict()["params"], **updates}
        where_clause = "WHERE run_id = ?"
        params: list[Any] = [json.dumps(merged, sort_keys=True), run_id]
        if if_not_terminal:
            where_clause += " AND status NOT IN (?, ?, ?, ?)"
            params.extend(
                [
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                    RunStatus.CANCELLED_WITH_IRREVERSIBLE_RESIDUE.value,
                ]
            )
        changed = await self._backend.execute(
            f"UPDATE {self.RUNS_TABLE} SET params_json = ? {where_clause}",
            tuple(params),
        )
        return changed > 0

    async def mark_run_signature_post_revocation(self, run_id: str) -> None:
        await self._backend.execute(
            f"""
            UPDATE {self.RUNS_TABLE}
            SET signature_post_revocation = ?
            WHERE run_id = ?
            """,
            (self.to_bool_param(True), run_id),
        )

    async def set_cancel_barrier(
        self, run_id: str, *, cancelled_at: Optional[datetime] = None
    ) -> bool:
        when = cancelled_at or datetime.now(timezone.utc)
        changed = await self._backend.execute(
            f"""
            UPDATE {self.RUNS_TABLE}
            SET cancel_barrier_at = ?, status = ?
            WHERE run_id = ? AND cancel_barrier_at IS NULL
            """,
            (self.to_timestamp_param(when), RunStatus.COMPENSATING.value, run_id),
        )
        return changed > 0

    async def insert_stage_link(self, link: StageLink) -> None:
        await self._backend.execute(
            f"""
            INSERT INTO {self.STAGE_LINKS_TABLE}
                (link_id, run_id, stage_name, attempt_number, signal_id,
                 idempotency_key, gate_outcome, gate_reason,
                 compensate_state, post_cancel, forced, actor_did, actor_sig,
                 occurred_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                link.link_id,
                link.run_id,
                link.stage_name,
                link.attempt_number,
                link.signal_id,
                link.idempotency_key,
                link.gate_outcome.value if link.gate_outcome else None,
                link.gate_reason,
                link.compensate_state,
                self.to_bool_param(link.post_cancel),
                self.to_bool_param(link.forced),
                link.actor_did,
                link.actor_sig,
                self.to_timestamp_param(
                    link.occurred_at or datetime.now(timezone.utc)
                ),
            ),
        )

    async def update_stage_link_transition(
        self,
        link_id: str,
        *,
        signal_id: Optional[str],
        gate_outcome: Optional[GateOutcome | str],
        gate_reason: Optional[str],
        actor_did: str,
        actor_sig: str,
        post_cancel: bool = False,
        forced: bool = False,
    ) -> None:
        parsed_outcome = (
            GateOutcome(gate_outcome).value if gate_outcome is not None else None
        )
        where_clause = "WHERE link_id = ?"
        where_params: list[Any] = [link_id]
        if not forced:
            where_clause += " AND forced = ?"
            where_params.append(self.to_bool_param(False))
        await self._backend.execute(
            f"""
            UPDATE {self.STAGE_LINKS_TABLE}
            SET signal_id = ?, gate_outcome = ?, gate_reason = ?,
                actor_did = ?, actor_sig = ?, post_cancel = ?, forced = ?
            {where_clause}
            """,
            (
                signal_id,
                parsed_outcome,
                gate_reason,
                actor_did,
                actor_sig,
                self.to_bool_param(post_cancel),
                self.to_bool_param(forced),
                *where_params,
            ),
        )

    async def update_compensate_state(
        self, link_id: str, compensate_state: str
    ) -> None:
        await self._backend.execute(
            f"""
            UPDATE {self.STAGE_LINKS_TABLE}
            SET compensate_state = ?
            WHERE link_id = ?
            """,
            (compensate_state, link_id),
        )

    async def list_stage_links(self, run_id: str) -> list[StageLink]:
        rows = await self._backend.fetch_all(
            f"""
            SELECT link_id, run_id, stage_name, attempt_number, signal_id,
                   idempotency_key, gate_outcome, gate_reason,
                   compensate_state, post_cancel, forced, actor_did, actor_sig,
                   occurred_at
            FROM {self.STAGE_LINKS_TABLE}
            WHERE run_id = ?
            ORDER BY occurred_at, attempt_number
            """,
            (run_id,),
        )
        return [
            StageLink(
                link_id=row[0],
                run_id=row[1],
                stage_name=row[2],
                attempt_number=row[3],
                signal_id=row[4],
                idempotency_key=row[5],
                gate_outcome=row[6],
                gate_reason=row[7],
                compensate_state=row[8],
                post_cancel=bool(row[9]),
                forced=bool(row[10]),
                actor_did=row[11],
                actor_sig=row[12],
                occurred_at=self.from_timestamp_field(row[13]),
            )
            for row in rows
        ]

    async def next_attempt_number(self, run_id: str, stage_name: str) -> int:
        row = await self._backend.fetch_one(
            f"""
            SELECT MAX(attempt_number)
            FROM {self.STAGE_LINKS_TABLE}
            WHERE run_id = ? AND stage_name = ?
            """,
            (run_id, stage_name),
        )
        current = row[0] if row is not None else None
        return int(current or 0) + 1

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
              AND r.status IN (?, ?, ?, ?)
              AND d.retention_days IS NOT NULL
            """,
            (
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
                RunStatus.CANCELLED_WITH_IRREVERSIBLE_RESIDUE.value,
            ),
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
