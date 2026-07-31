"""One-time data migrations for sovereign-core SQLAlchemy entities.

Sovereign-core's schema is managed by raw SQL ``CREATE TABLE IF NOT
EXISTS`` statements in ``async_database.py`` (see ``CORE_SCHEMA``),
not Alembic. Migrations live here as idempotent async functions that
``AsyncDatabase`` runs on startup AFTER the schema-create step.

Phase 2 of #1447: add a parallel ``embedding_vec`` column to
``saved_items``:

- On Postgres: ``vector(N)``, indexed with HNSW for fast cosine kNN.
  The existing ``embedding`` BYTEA column stays — legacy raw IO paths
  in :class:`SavedItemsStore` continue to write/read it unchanged, and
  the new ``save_item`` dual-write keeps the two in sync.
- On SQLite: also ``BLOB``. Same dual-write keeps both columns in
  sync there too. PurePythonBackend can read either; the ORM uses
  ``embedding_vec`` so the code path is the same as PG.

The parallel column lets us flip the vector-backend factory to
``PgVectorBackend`` on PG without rewriting the raw INSERT / SELECT
paths that bind ``embedding`` as float32 bytes. (Caught by codex
review — an in-place ``ALTER COLUMN TYPE`` would have broken
``save_item()`` and ``SavedItem.from_row()`` on PG.)
"""

from __future__ import annotations

import json
import hashlib
import logging
import struct
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from ..async_assertion_store import (
    _erasure_receipt_key,
    _legacy_erasure_assertion_key,
    _now,
)

if TYPE_CHECKING:
    from ..async_database import AsyncDatabase

logger = logging.getLogger(__name__)


_SEMANTIC_ASSERTION_SCHEMA_VERSION = "semantic_assertion_store_v5"
_SEMANTIC_VALIDATION_SCHEMA_VERSION = "semantic_validation_reports_v1"
_SEMANTIC_MAINTENANCE_SCHEMA_VERSION = "semantic_maintenance_v1"
_SEMANTIC_MAINTENANCE_CURSOR_SCHEMA_VERSION = "semantic_maintenance_v2_cursor"
_SEMANTIC_MAINTENANCE_REPAIR_CURSOR_SCHEMA_VERSION = "semantic_maintenance_v3_repair_cursor"
_SEMANTIC_MAINTENANCE_RESUME_SCHEMA_VERSION = "semantic_maintenance_v4_resume_state"
_SEMANTIC_MAINTENANCE_AUDIT_REVISION_SCHEMA_VERSION = "semantic_maintenance_v5_audit_revision"
_SEMANTIC_MAINTENANCE_REPAIR_MODE_SCHEMA_VERSION = "semantic_maintenance_v6_repair_mode"
_SEMANTIC_MAINTENANCE_ERASURE_REDACTION_SCHEMA_VERSION = (
    "semantic_maintenance_v7_erasure_redaction"
)
_SEMANTIC_MAINTENANCE_LEASE_PRECISION_SCHEMA_VERSION = (
    "semantic_maintenance_v8_lease_precision"
)
_SEMANTIC_ASSERTION_LOCK_DOMAIN = b"kestrel:semantic-assertion-schema:v1\0"


def _semantic_assertion_lock_id() -> int:
    """Stable transaction-scoped PostgreSQL lock for semantic-schema DDL."""
    return int.from_bytes(
        hashlib.sha256(_SEMANTIC_ASSERTION_LOCK_DOMAIN).digest()[:8],
        "big",
        signed=True,
    )


async def _semantic_schema_marker_exists(db: "AsyncDatabase") -> bool:
    row = await db.fetchone(
        "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
        (_SEMANTIC_ASSERTION_SCHEMA_VERSION,),
    )
    return row is not None


@asynccontextmanager
async def _semantic_validation_migration_transaction(db: "AsyncDatabase"):
    """Serialize a validation-schema marker read with SQLite's writer slot."""
    if db.backend_type == "sqlite":
        # The normal SQLite transaction begins deferred.  Once it has read a
        # schema marker, attempting to promote it to a writer races another
        # initializer and fails immediately instead of observing busy_timeout.
        # Begin as the writer so every marker read below is serialized.
        async with db.backend.transaction(immediate=True):  # type: ignore[call-arg]
            yield
        return
    async with db.transaction():
        yield


async def _ensure_erasure_receipts_are_opaque(db: "AsyncDatabase") -> None:
    """Upgrade prior JSON receipts without retaining erased identifiers.

    The first implementation stored full erasure targets in ``receipt``.  A
    pre-release database may still have that column, so move just its numeric
    generation into a dedicated column and overwrite the JSON before marking
    the migration complete.  New databases never create ``receipt`` at all.
    """
    if db.backend_type == "postgres":
        columns = await db.fetchall(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'semantic_assertion_erasure_receipts'",
            (),
        )
    else:
        columns = await db.fetchall(
            "SELECT name FROM pragma_table_info('semantic_assertion_erasure_receipts')",
            (),
        )
    column_names = {str(row[0]) for row in columns}
    if "generation" not in column_names:
        if db.backend_type == "postgres":
            await db.execute(
                "ALTER TABLE semantic_assertion_erasure_receipts "
                "ADD COLUMN IF NOT EXISTS generation INTEGER",
                (),
            )
        else:
            await db.execute(
                "ALTER TABLE semantic_assertion_erasure_receipts "
                "ADD COLUMN generation INTEGER",
                (),
            )
        column_names.add("generation")
    if "receipt" not in column_names:
        return

    rows = await db.fetchall(
        "SELECT tenant_id, operation_id, receipt FROM semantic_assertion_erasure_receipts",
        (),
    )
    for tenant_id, operation_id, receipt in rows:
        try:
            generation = json.loads(receipt).get("generation")
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            # A malformed legacy retry record must not keep an opaque payload
            # around.  Deleting it fails closed: retrying after restart reports
            # that the target is absent rather than resurrecting an identifier.
            await db.execute(
                "DELETE FROM semantic_assertion_erasure_receipts "
                "WHERE tenant_id = ? AND operation_id = ?",
                (tenant_id, operation_id),
            )
            continue
        # Generations are ledger ordinals, not numeric measurements.  Reject
        # booleans and floats instead of silently truncating a legacy value
        # such as 1.5 to generation 1 and authenticating the wrong fence.
        if type(generation) is not int or generation < 1:
            await db.execute(
                "DELETE FROM semantic_assertion_erasure_receipts "
                "WHERE tenant_id = ? AND operation_id = ?",
                (tenant_id, operation_id),
            )
            continue
        await db.execute(
            "UPDATE semantic_assertion_erasure_receipts "
            "SET operation_id = ?, generation = ?, receipt = '{}' "
            "WHERE tenant_id = ? AND operation_id = ?",
            (
                _erasure_receipt_key(str(operation_id)),
                generation,
                tenant_id,
                operation_id,
            ),
        )


async def migrate_semantic_assertion_store(db: "AsyncDatabase") -> None:
    """Create the normalized canonical-assertion authority.

    This is deliberately additive and backend-neutral.  Assertion records are
    not graph rows or opaque properties: current pointers, immutable
    revisions, provenance links, derivation supports, projection eligibility,
    idempotency receipts, and change events have separate relational rows.
    One transaction covers the complete DDL set so a failed migration neither
    advertises a partial authority nor leaves a subset of its indexes behind.
    """
    statements = (
        """CREATE TABLE IF NOT EXISTS semantic_assertion_tenants (
            tenant_id TEXT PRIMARY KEY,
            generation INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_assertions (
            tenant_id TEXT NOT NULL,
            assertion_id TEXT NOT NULL,
            owning_agent_id TEXT NOT NULL,
            current_revision_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, assertion_id)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_assertion_revisions (
            revision_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            assertion_id TEXT NOT NULL,
            owning_agent_id TEXT NOT NULL,
            status TEXT NOT NULL,
            epistemic_state TEXT NOT NULL,
            subject_value TEXT NOT NULL,
            predicate_value TEXT NOT NULL,
            object_kind TEXT NOT NULL,
            object_value TEXT NOT NULL,
            object_datatype TEXT,
            object_language TEXT,
            asserted_at TEXT NOT NULL,
            observed_start TEXT,
            observed_end TEXT,
            valid_start TEXT,
            valid_end TEXT,
            supersedes_revision_id TEXT,
            lineage_kind TEXT NOT NULL,
            eligible INTEGER NOT NULL,
            accepted_order INTEGER NOT NULL,
            assertion_mapping TEXT NOT NULL,
            CHECK (status IN ('active', 'superseded', 'retracted', 'quarantined', 'deleted')),
            CHECK (epistemic_state IN ('asserted', 'observed', 'reported', 'inferred', 'hypothesis', 'disputed', 'retracted')),
            CHECK (lineage_kind IN ('direct', 'derived')),
            CHECK (eligible IN (0, 1)),
            CHECK (accepted_order > 0),
            CHECK ((status = 'retracted') = (epistemic_state = 'retracted')),
            CHECK (status NOT IN ('retracted', 'quarantined', 'deleted') OR supersedes_revision_id IS NULL),
            UNIQUE (tenant_id, assertion_id, revision_id)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_source_occurrences (
            tenant_id TEXT NOT NULL,
            source_occurrence_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            locator TEXT NOT NULL,
            received_at TEXT NOT NULL,
            content_digest TEXT,
            actor TEXT,
            selector TEXT,
            source_mapping TEXT NOT NULL,
            PRIMARY KEY (tenant_id, source_occurrence_id)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_revision_sources (
            tenant_id TEXT NOT NULL,
            revision_id TEXT NOT NULL,
            source_occurrence_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            CHECK (ordinal >= 0),
            PRIMARY KEY (tenant_id, revision_id, source_occurrence_id),
            UNIQUE (tenant_id, revision_id, ordinal)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_derivation_inputs (
            tenant_id TEXT NOT NULL,
            derived_revision_id TEXT NOT NULL,
            input_revision_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            CHECK (ordinal >= 0),
            PRIMARY KEY (tenant_id, derived_revision_id, input_revision_id),
            UNIQUE (tenant_id, derived_revision_id, ordinal)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_inference_runs (
            tenant_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            profile_key TEXT NOT NULL,
            ontology_namespace TEXT NOT NULL,
            ontology_version TEXT NOT NULL,
            ontology_digest TEXT NOT NULL,
            source_generation INTEGER NOT NULL,
            status TEXT NOT NULL,
            incomplete_reason TEXT,
            result_mapping TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            PRIMARY KEY (tenant_id, run_id),
            CHECK (source_generation >= 0),
            CHECK (status IN ('running', 'complete', 'incomplete', 'failed'))
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_inference_state (
            tenant_id TEXT NOT NULL,
            profile_key TEXT NOT NULL,
            run_id TEXT NOT NULL,
            source_generation INTEGER NOT NULL,
            status TEXT NOT NULL,
            incomplete_reason TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, profile_key),
            CHECK (source_generation >= 0),
            CHECK (status IN ('running', 'complete', 'incomplete', 'failed'))
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_inference_derivations (
            tenant_id TEXT NOT NULL,
            derivation_id TEXT NOT NULL,
            derived_assertion_id TEXT NOT NULL,
            derived_revision_id TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            profile_key TEXT NOT NULL,
            rule_profile_version TEXT NOT NULL,
            ontology_namespace TEXT NOT NULL,
            ontology_version TEXT NOT NULL,
            ontology_digest TEXT NOT NULL,
            run_id TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            active INTEGER NOT NULL,
            PRIMARY KEY (tenant_id, derivation_id),
            CHECK (active IN (0, 1))
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_inference_derivation_inputs (
            tenant_id TEXT NOT NULL,
            derivation_id TEXT NOT NULL,
            input_revision_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (tenant_id, derivation_id, input_revision_id),
            UNIQUE (tenant_id, derivation_id, ordinal),
            CHECK (ordinal >= 0)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_projection_eligibility (
            tenant_id TEXT NOT NULL,
            revision_id TEXT NOT NULL,
            eligible INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (eligible IN (0, 1)),
            PRIMARY KEY (tenant_id, revision_id)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_projection_outbox (
            event_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            assertion_id TEXT NOT NULL,
            revision_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            generation INTEGER NOT NULL,
            eligible INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            CHECK (eligible IN (0, 1)),
            CHECK (generation > 0)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_projection_erasure_outbox (
            event_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            generation INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            CHECK (operation = 'erased'),
            CHECK (generation > 0)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_assertion_operations (
            tenant_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            receipt TEXT NOT NULL,
            assertion_ids TEXT NOT NULL,
            revision_ids TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, operation_id)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_assertion_erasure_receipts (
            tenant_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            generation INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, operation_id)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_assertion_erased_operation_tombstones (
            tenant_id TEXT NOT NULL,
            purpose TEXT NOT NULL,
            operation_key TEXT NOT NULL,
            request_key TEXT NOT NULL,
            generation INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, purpose, operation_key),
            UNIQUE (tenant_id, operation_key),
            CHECK (generation > 0)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_assertion_legacy_erasure_fences (
            tenant_id TEXT NOT NULL,
            assertion_key TEXT NOT NULL,
            generation INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, assertion_key),
            CHECK (generation > 0)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_semantic_assertion_current ON semantic_assertions(tenant_id, current_revision_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_revision_query ON semantic_assertion_revisions(tenant_id, status, subject_value, predicate_value, revision_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_revision_object ON semantic_assertion_revisions(tenant_id, predicate_value, object_kind, object_value)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_revision_valid_time ON semantic_assertion_revisions(tenant_id, valid_start, valid_end)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_revision_sources_source ON semantic_revision_sources(tenant_id, source_occurrence_id, revision_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_derivation_input ON semantic_derivation_inputs(tenant_id, input_revision_id, derived_revision_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_inference_runs_profile ON semantic_inference_runs(tenant_id, profile_key, source_generation)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_inference_derivation_revision ON semantic_inference_derivations(tenant_id, derived_revision_id, active)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_inference_derivation_input ON semantic_inference_derivation_inputs(tenant_id, input_revision_id, derivation_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_outbox_changes ON semantic_projection_outbox(tenant_id, generation, created_at, event_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_erasure_changes ON semantic_projection_erasure_outbox(tenant_id, generation, created_at, event_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_erased_operation_tombstones ON semantic_assertion_erased_operation_tombstones(tenant_id, operation_key)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_legacy_erasure_fences ON semantic_assertion_legacy_erasure_fences(tenant_id, assertion_key)",
    )
    async with db.transaction():
        # PostgreSQL's catalog DDL can race even with IF NOT EXISTS.  A
        # transaction-scoped advisory lock keeps a concurrent multi-agent boot
        # from attempting this complete DDL set simultaneously; the marker is
        # committed atomically with the schema.  SQLite promotes to its single
        # writer slot before the marker read for the corresponding behavior.
        if db.backend_type == "postgres":
            await db.execute(
                "SELECT pg_advisory_xact_lock(?)",
                (_semantic_assertion_lock_id(),),
            )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS semantic_schema_migrations ("
            "version TEXT PRIMARY KEY, completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)",
            (),
        )
        if db.backend_type == "sqlite":
            await db.execute("DELETE FROM semantic_schema_migrations WHERE 0", ())
        if await _semantic_schema_marker_exists(db):
            return
        legacy_upgrade = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations "
            "WHERE version IN (?, ?)",
            (
                "semantic_assertion_store_v3",
                "semantic_assertion_store_v4",
            ),
        )
        for statement in statements:
            await db.execute(statement, ())
        await _ensure_erasure_receipts_are_opaque(db)
        if legacy_upgrade is not None:
            # v3 erased content-bearing operation receipts before blinded
            # per-operation tombstones existed.  Preserve only an opaque,
            # per-assertion fence derived from the erasure request digest.
            # This cannot recover the lost operation IDs or content, but it
            # lets the adapter reject the exact deterministic semantic
            # identity instead of imposing a tenant-wide write freeze.
            legacy_erasure_rows = await db.fetchall(
                "SELECT tenant_id, request_digest, MAX(generation) "
                "FROM semantic_assertion_erasure_receipts "
                "GROUP BY tenant_id, request_digest",
                (),
            )
            for tenant_id, request_digest, generation in legacy_erasure_rows:
                request_digest = str(request_digest)
                if (
                    len(request_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in request_digest
                    )
                    or type(generation) is not int
                    or generation < 1
                ):
                    raise ValueError(
                        "legacy semantic erasure receipt has malformed opaque state"
                    )
                assertion_key = _legacy_erasure_assertion_key(
                    request_digest
                )
                existing = await db.fetchone(
                    "SELECT generation "
                    "FROM semantic_assertion_legacy_erasure_fences "
                    "WHERE tenant_id = ? AND assertion_key = ?",
                    (tenant_id, assertion_key),
                )
                if existing is None:
                    await db.execute(
                        "INSERT INTO semantic_assertion_legacy_erasure_fences "
                        "(tenant_id, assertion_key, generation, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            tenant_id,
                            assertion_key,
                            generation,
                            _now(),
                        ),
                    )
        await db.execute(
            "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
            (_SEMANTIC_ASSERTION_SCHEMA_VERSION,),
        )


async def migrate_semantic_validation_reports(db: "AsyncDatabase") -> None:
    """Create the tenant-bound SHACL report tables in the canonical database.

    This is deliberately a companion migration rather than a second database:
    validation reports are auditable projections of the same semantic tenant
    and have no independent assertion or eligibility write path.
    """
    statements = (
        """CREATE TABLE IF NOT EXISTS semantic_validation_reports (
            report_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            report_version INTEGER NOT NULL,
            assertion_ids TEXT NOT NULL,
            shape_set_id TEXT NOT NULL,
            shape_set_version TEXT NOT NULL,
            validation_profile_id TEXT NOT NULL,
            validation_profile_version TEXT NOT NULL,
            checkpoint_generation INTEGER,
            run_id TEXT NOT NULL,
            state TEXT NOT NULL,
            action TEXT NOT NULL,
            source TEXT NOT NULL,
            evaluated_at TEXT NOT NULL,
            report_mapping TEXT NOT NULL,
            CHECK (report_version = 1),
            CHECK (checkpoint_generation IS NULL OR checkpoint_generation >= 0),
            CHECK (state IN ('conforms', 'nonconformant', 'incomplete')),
            CHECK (action IN ('accept', 'accept-with-report', 'reject', 'quarantine')),
            CHECK (source IN ('asserted', 'imported', 'inferred', 'revalidation'))
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_validation_results (
            tenant_id TEXT NOT NULL,
            report_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            assertion_id TEXT,
            shape_id TEXT,
            constraint_code TEXT NOT NULL,
            severity TEXT NOT NULL,
            PRIMARY KEY (tenant_id, report_id, ordinal),
            CHECK (ordinal >= 0),
            CHECK (severity IN ('info', 'warning', 'violation'))
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_validation_report_assertions (
            tenant_id TEXT NOT NULL,
            report_id TEXT NOT NULL,
            assertion_id TEXT NOT NULL,
            PRIMARY KEY (tenant_id, report_id, assertion_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_semantic_validation_report_tenant_time ON semantic_validation_reports(tenant_id, evaluated_at, report_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_validation_report_assertion ON semantic_validation_report_assertions(tenant_id, assertion_id, report_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_validation_result_tenant_assertion ON semantic_validation_results(tenant_id, assertion_id, report_id)",
    )
    async with _semantic_validation_migration_transaction(db):
        if db.backend_type == "postgres":
            await db.execute("SELECT pg_advisory_xact_lock(?)", (_semantic_assertion_lock_id(),))
        await db.execute(
            "CREATE TABLE IF NOT EXISTS semantic_schema_migrations "
            "(version TEXT PRIMARY KEY)",
            (),
        )
        existing = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_VALIDATION_SCHEMA_VERSION,),
        )
        if existing is not None:
            return
        for statement in statements:
            await db.execute(statement, ())
        await db.execute(
            "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
            (_SEMANTIC_VALIDATION_SCHEMA_VERSION,),
        )


async def migrate_semantic_maintenance(db: "AsyncDatabase") -> None:
    """Create durable, tenant-scoped state for bounded semantic maintenance.

    The state is deliberately separate from the inference ledger.  Validation,
    expiry/provenance repair, contradiction review, and materialization have
    one shared sleep checkpoint, but inference profiles retain their own
    independently auditable closure checkpoint.
    """
    statements = (
        """CREATE TABLE IF NOT EXISTS semantic_maintenance_state (
            tenant_id TEXT PRIMARY KEY,
            profile_key TEXT NOT NULL,
            checkpoint_generation INTEGER NOT NULL,
            checkpoint_event_id TEXT,
            run_id TEXT NOT NULL,
            status TEXT NOT NULL,
            capability_versions TEXT NOT NULL,
            repair_cursor_revision_id TEXT,
            repair_active INTEGER NOT NULL DEFAULT 0,
            repair_mode TEXT,
            repair_scan_complete INTEGER NOT NULL DEFAULT 0,
            repair_checkpoint_generation INTEGER,
            repair_checkpoint_event_id TEXT,
            repair_reconcile_cursor_derivation_id TEXT,
            audit_assertion_id TEXT,
            audit_assertion_revision_id TEXT,
            audit_competitor_cursor_revision_id TEXT,
            updated_at TEXT NOT NULL,
            CHECK (checkpoint_generation >= 0),
            CHECK (repair_active IN (0, 1)),
            CHECK (repair_scan_complete IN (0, 1)),
            CHECK (repair_mode IS NULL OR repair_mode IN ('full_rebuild', 'profile_change', 'current_scan')),
            CHECK (status IN ('complete', 'partial', 'failed', 'no_op'))
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_maintenance_runs (
            tenant_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            profile_key TEXT NOT NULL,
            source_generation INTEGER NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            result_mapping TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            PRIMARY KEY (tenant_id, run_id),
            CHECK (source_generation >= 0),
            CHECK (status IN ('running', 'complete', 'partial', 'failed', 'no_op'))
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_maintenance_leases (
            tenant_id TEXT PRIMARY KEY,
            holder_id TEXT NOT NULL,
            fencing_token INTEGER NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (fencing_token > 0)
        )""",
        """CREATE TABLE IF NOT EXISTS semantic_maintenance_reports (
            tenant_id TEXT NOT NULL,
            report_id TEXT NOT NULL,
            report_kind TEXT NOT NULL,
            evidence_digest TEXT NOT NULL,
            status TEXT NOT NULL,
            evidence_mapping TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, report_id),
            UNIQUE (tenant_id, evidence_digest),
            CHECK (report_kind IN ('contradiction_candidate', 'supersession_candidate', 'orphan_provenance', 'expired_assertion', 'ineligible_assertion')),
            CHECK (status IN ('review_required', 'deterministic_action_applied'))
        )""",
        "CREATE INDEX IF NOT EXISTS idx_semantic_maintenance_runs_tenant_time ON semantic_maintenance_runs(tenant_id, started_at DESC, run_id)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_maintenance_reports_tenant_kind ON semantic_maintenance_reports(tenant_id, report_kind, status)",
    )
    async with _semantic_validation_migration_transaction(db):
        if db.backend_type == "postgres":
            await db.execute("SELECT pg_advisory_xact_lock(?)", (_semantic_assertion_lock_id(),))
        await db.execute(
            "CREATE TABLE IF NOT EXISTS semantic_schema_migrations "
            "(version TEXT PRIMARY KEY)",
            (),
        )
        existing = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_SCHEMA_VERSION,),
        )
        if existing is None:
            for statement in statements:
                await db.execute(statement, ())
            await db.execute(
                "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
                (_SEMANTIC_MAINTENANCE_SCHEMA_VERSION,),
            )

        cursor_migration = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_CURSOR_SCHEMA_VERSION,),
        )
        if cursor_migration is None:
            if db.backend_type == "postgres":
                cursor_column = await db.fetchone(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'semantic_maintenance_state' "
                    "AND column_name = 'checkpoint_event_id'",
                    (),
                )
            else:
                columns = await db.fetchall(
                    "PRAGMA table_info(semantic_maintenance_state)", ()
                )
                cursor_column = next(
                    (column for column in columns if column[1] == "checkpoint_event_id"),
                    None,
                )
            if cursor_column is None:
                await db.execute(
                    "ALTER TABLE semantic_maintenance_state "
                    "ADD COLUMN checkpoint_event_id TEXT",
                    (),
                )
            await db.execute(
                "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
                (_SEMANTIC_MAINTENANCE_CURSOR_SCHEMA_VERSION,),
            )

        repair_cursor_migration = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_REPAIR_CURSOR_SCHEMA_VERSION,),
        )
        if repair_cursor_migration is None:
            if db.backend_type == "postgres":
                columns = await db.fetchall(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'semantic_maintenance_state' "
                    "AND column_name IN ('repair_cursor_revision_id', 'repair_active')",
                    (),
                )
                column_names = {str(row[0]) for row in columns}
            else:
                columns = await db.fetchall(
                    "PRAGMA table_info(semantic_maintenance_state)", ()
                )
                column_names = {str(row[1]) for row in columns}
            if "repair_cursor_revision_id" not in column_names:
                await db.execute(
                    "ALTER TABLE semantic_maintenance_state "
                    "ADD COLUMN repair_cursor_revision_id TEXT",
                    (),
                )
            if "repair_active" not in column_names:
                await db.execute(
                    "ALTER TABLE semantic_maintenance_state "
                    "ADD COLUMN repair_active INTEGER NOT NULL DEFAULT 0",
                    (),
                )
            await db.execute(
                "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
                (_SEMANTIC_MAINTENANCE_REPAIR_CURSOR_SCHEMA_VERSION,),
            )

        resume_migration = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_RESUME_SCHEMA_VERSION,),
        )
        if resume_migration is None:
            if db.backend_type == "postgres":
                columns = await db.fetchall(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'semantic_maintenance_state' "
                    "AND column_name IN ("
                    "'repair_checkpoint_generation', 'repair_checkpoint_event_id', "
                    "'audit_assertion_id', 'audit_competitor_cursor_revision_id'"
                    ")",
                    (),
                )
                column_names = {str(row[0]) for row in columns}
            else:
                columns = await db.fetchall(
                    "PRAGMA table_info(semantic_maintenance_state)", ()
                )
                column_names = {str(row[1]) for row in columns}
            for column, declaration in (
                ("repair_checkpoint_generation", "INTEGER"),
                ("repair_checkpoint_event_id", "TEXT"),
                ("audit_assertion_id", "TEXT"),
                ("audit_competitor_cursor_revision_id", "TEXT"),
            ):
                if column not in column_names:
                    await db.execute(
                        "ALTER TABLE semantic_maintenance_state "
                        f"ADD COLUMN {column} {declaration}",
                        (),
                    )
            await db.execute(
                "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
                (_SEMANTIC_MAINTENANCE_RESUME_SCHEMA_VERSION,),
            )

        audit_revision_migration = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_AUDIT_REVISION_SCHEMA_VERSION,),
        )
        if audit_revision_migration is None:
            if db.backend_type == "postgres":
                column = await db.fetchone(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'semantic_maintenance_state' "
                    "AND column_name = 'audit_assertion_revision_id'",
                    (),
                )
            else:
                columns = await db.fetchall(
                    "PRAGMA table_info(semantic_maintenance_state)", ()
                )
                column = next(
                    (
                        item
                        for item in columns
                        if item[1] == "audit_assertion_revision_id"
                    ),
                    None,
                )
            if column is None:
                await db.execute(
                    "ALTER TABLE semantic_maintenance_state "
                    "ADD COLUMN audit_assertion_revision_id TEXT",
                    (),
                )
            await db.execute(
                "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
                (_SEMANTIC_MAINTENANCE_AUDIT_REVISION_SCHEMA_VERSION,),
            )

        repair_mode_migration = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_REPAIR_MODE_SCHEMA_VERSION,),
        )
        if repair_mode_migration is None:
            if db.backend_type == "postgres":
                columns = await db.fetchall(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'semantic_maintenance_state' "
                    "AND column_name IN ("
                    "'repair_mode', 'repair_scan_complete', "
                    "'repair_reconcile_cursor_derivation_id'"
                    ")",
                    (),
                )
                column_names = {str(row[0]) for row in columns}
            else:
                columns = await db.fetchall(
                    "PRAGMA table_info(semantic_maintenance_state)", ()
                )
                column_names = {str(row[1]) for row in columns}
            for column, declaration in (
                ("repair_mode", "TEXT"),
                ("repair_scan_complete", "INTEGER NOT NULL DEFAULT 0"),
                ("repair_reconcile_cursor_derivation_id", "TEXT"),
            ):
                if column not in column_names:
                    await db.execute(
                        "ALTER TABLE semantic_maintenance_state "
                        f"ADD COLUMN {column} {declaration}",
                        (),
                    )
            await db.execute(
                "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
                (_SEMANTIC_MAINTENANCE_REPAIR_MODE_SCHEMA_VERSION,),
            )

        erasure_redaction_migration = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_ERASURE_REDACTION_SCHEMA_VERSION,),
        )
        if erasure_redaction_migration is None:
            # Earlier maintenance releases retained assertion/revision IDs in
            # report JSON and resumable cursor columns.  A completed erasure
            # deliberately leaves only opaque receipts, so an upgrade cannot
            # identify every prior target without recreating the privacy leak.
            # Invalidate the rebuildable maintenance layer only for tenants
            # with a retained opaque erasure event.  The next bounded worker
            # replays from generation zero and receives that event before it
            # can claim a fresh checkpoint; tenants with no erasure keep
            # their unrelated reports and state intact.
            affected_tenants = await db.fetchall(
                "SELECT DISTINCT tenant_id FROM semantic_projection_erasure_outbox",
                (),
            )
            tenant_ids = tuple(str(row[0]) for row in affected_tenants)
            if tenant_ids:
                for tenant_id in tenant_ids:
                    # Maintenance writes renew this lease before locking the
                    # canonical tenant.  Replacing it first fences a worker
                    # that started before the upgrade from recreating a
                    # redacted report after this transaction commits.
                    await db.execute(
                        "INSERT INTO semantic_maintenance_leases "
                        "(tenant_id, holder_id, fencing_token, expires_at, updated_at) "
                        "VALUES (?, 'schema-erasure-redaction', 1, 0, ?) "
                        "ON CONFLICT(tenant_id) DO UPDATE SET "
                        "holder_id = excluded.holder_id, "
                        "fencing_token = semantic_maintenance_leases.fencing_token + 1, "
                        "expires_at = excluded.expires_at, "
                        "updated_at = excluded.updated_at",
                        (tenant_id, _now()),
                    )
                placeholders = ", ".join("?" for _ in tenant_ids)
                await db.execute(
                    "DELETE FROM semantic_maintenance_reports WHERE tenant_id IN ("
                    + placeholders
                    + ")",
                    tenant_ids,
                )
                await db.execute(
                    "UPDATE semantic_maintenance_state SET "
                    "checkpoint_generation = 0, checkpoint_event_id = NULL, "
                    "status = 'partial', repair_cursor_revision_id = NULL, "
                    "repair_active = 0, repair_mode = NULL, repair_scan_complete = 0, "
                    "repair_checkpoint_generation = NULL, repair_checkpoint_event_id = NULL, "
                    "repair_reconcile_cursor_derivation_id = NULL, audit_assertion_id = NULL, "
                    "audit_assertion_revision_id = NULL, "
                    "audit_competitor_cursor_revision_id = NULL, updated_at = ? "
                    "WHERE tenant_id IN ("
                    + placeholders
                    + ")",
                    (_now(), *tenant_ids),
                )
            await db.execute(
                "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
                (_SEMANTIC_MAINTENANCE_ERASURE_REDACTION_SCHEMA_VERSION,),
            )

        lease_precision_migration = await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_LEASE_PRECISION_SCHEMA_VERSION,),
        )
        if lease_precision_migration is None:
            if db.backend_type == "postgres":
                lease_expiry_type = await db.fetchone(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'semantic_maintenance_leases' "
                    "AND column_name = 'expires_at'",
                    (),
                )
                if lease_expiry_type is None:
                    raise RuntimeError(
                        "semantic_maintenance_leases.expires_at is missing"
                    )
                # These migration statements are executed directly rather
                # than through normalize_schema(). V1 therefore created
                # PostgreSQL REAL/float4, whose 2026 epoch-second resolution
                # is 128 seconds -- longer than the 60-second lease. Upgrade
                # legacy installs in-place before advertising the marker.
                if str(lease_expiry_type[0]) != "float8":
                    await db.execute(
                        "ALTER TABLE semantic_maintenance_leases "
                        "ALTER COLUMN expires_at TYPE DOUBLE PRECISION "
                        "USING expires_at::DOUBLE PRECISION",
                        (),
                    )
            # SQLite's REAL storage class is already an IEEE-754 binary64.
            # It accepts the explicit DOUBLE PRECISION declaration used for
            # fresh databases, while legacy REAL declarations need no rewrite.
            await db.execute(
                "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
                (_SEMANTIC_MAINTENANCE_LEASE_PRECISION_SCHEMA_VERSION,),
            )


# Default embedding dimension if no embedded rows exist yet (fresh DB).
# Matches the default Ollama ``nomic-embed-text`` model. Kept in sync
# with ``saved_item.SAVED_ITEM_EMBEDDING_DIM``.
_DEFAULT_DIM = 768


async def migrate_saved_items_add_embedding_vec(db: "AsyncDatabase") -> None:
    """Add a parallel ``embedding_vec`` column to ``saved_items`` and
    backfill from the existing ``embedding`` BYTEA / BLOB.

    Idempotent: skips cleanly when the column already exists.

    On Postgres:
        1. ``CREATE EXTENSION IF NOT EXISTS vector``.
        2. Sniff the dimension from existing rows (or default to 768).
        3. ``ALTER TABLE ... ADD COLUMN embedding_vec vector(<dim>)`` if
           the column doesn't already exist.
        4. Backfill: for each row with non-null BYTEA ``embedding``,
           unpack to floats, format as pgvector text, set
           ``embedding_vec``.
        5. HNSW index on ``embedding_vec vector_cosine_ops``.

    On SQLite:
        1. ``ALTER TABLE ... ADD COLUMN embedding_vec BLOB`` (no-op via
           ``pragma_table_info`` check if already present).
        2. Backfill: ``UPDATE`` copies bytes from ``embedding`` to
           ``embedding_vec``. (Yes, same data twice — keeps the ORM
           code paths identical across dialects.)

    Runs inside ``db.transaction()`` so any partial failure rolls back
    cleanly. The advertised idempotency depends on this — without a
    transaction, a crash partway through could leave the schema in a
    half-migrated state that later runs misdetect.
    """
    backend_type = getattr(db, "backend_type", None)

    if backend_type == "postgres":
        await _migrate_pg(db)
    elif backend_type == "sqlite":
        await _migrate_sqlite(db)
    # Other dialects: no-op. The ORM column maps to LargeBinary on
    # non-PG dialects so any reasonable backend should work, but we
    # don't try to introspect arbitrary engines.


async def _migrate_pg(db: "AsyncDatabase") -> None:
    """Postgres path — adds vector column + backfills + HNSW index."""

    # Check if embedding_vec already exists; if so, the migration ran
    # in a prior boot.
    rows = await db.fetchall(
        """SELECT 1 FROM information_schema.columns
           WHERE table_name = 'saved_items' AND column_name = 'embedding_vec'""",
        (),
    )
    if rows:
        logger.debug(
            "saved_items.embedding_vec already present — skipping Phase-2 PG migration."
        )
        return

    # Confirm the source table + column exist; otherwise let
    # ``_init_schema`` handle it on the next boot.
    src = await db.fetchall(
        """SELECT udt_name FROM information_schema.columns
           WHERE table_name = 'saved_items' AND column_name = 'embedding'""",
        (),
    )
    if not src:
        logger.debug(
            "saved_items table not yet present — skipping Phase-2 PG migration."
        )
        return

    # Sniff the dimension from existing rows. We INTENTIONALLY don't
    # guess at a default for fresh DBs: an embedding service may be
    # configured for any of nomic-embed-text (768), mxbai-embed-large
    # (1024), or OpenAI ada-002 (1536), and creating ``vector(768)``
    # against a 1536-dim writer would make every subsequent
    # ``_write_embedding_vec`` fail with a dim mismatch. Skip column
    # creation here; the next boot will re-run, sniff the actual dim
    # from saved rows, and create the column at the right size.
    # (Caught by codex review on the Phase 2 PR.)
    sample = await db.fetchall(
        """SELECT octet_length(embedding) FROM saved_items
           WHERE embedding IS NOT NULL LIMIT 1""",
        (),
    )
    if not sample:
        logger.info(
            "saved_items has no embedded rows yet — deferring embedding_vec "
            "column creation until the next boot (we don't guess a dim)."
        )
        return

    byte_len = sample[0][0]
    if byte_len % 4 != 0 or byte_len <= 0:
        logger.error(
            "saved_items.embedding has non-float32 byte length %d. "
            "Refusing to migrate.", byte_len,
        )
        return
    dim = byte_len // 4
    logger.info(
        "Sniffed embedding dimension %d (%d bytes) from existing rows.",
        dim, byte_len,
    )

    expected_bytes = dim * 4

    async with db.transaction():
        # pgvector extension must exist BEFORE ``ALTER TABLE`` references
        # ``vector(N)`` — on a fresh DB without the extension, the ALTER
        # fails with ``type "vector" does not exist``. ``IF NOT EXISTS``
        # so repeat-installs are silent. (Caught by codex review on the
        # Phase 2 PR — the original order had the ALTER first, which
        # broke the migration on fresh Postgres databases.)
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())

        # Add the parallel vector column.
        await db.execute(
            f"ALTER TABLE saved_items ADD COLUMN embedding_vec vector({dim})",
            (),
        )

        # Backfill row-by-row. For typical saved_items volumes this
        # fits in memory comfortably; swap to a server-side cursor if
        # scale ever changes.
        all_rows = await db.fetchall(
            "SELECT id, embedding FROM saved_items WHERE embedding IS NOT NULL",
            (),
        )
        converted = 0
        skipped = 0
        for row_id, embedding_bytes in all_rows:
            if embedding_bytes is None:
                continue
            if len(embedding_bytes) != expected_bytes:
                logger.warning(
                    "Skipping row %s: %d bytes != expected %d (different "
                    "embedding-model dim — will not appear in vector search "
                    "until re-embedded).",
                    row_id, len(embedding_bytes), expected_bytes,
                )
                skipped += 1
                continue
            floats = struct.unpack(f"<{dim}f", bytes(embedding_bytes))
            vec_text = "[" + ",".join(repr(float(v)) for v in floats) + "]"
            await db.execute(
                "UPDATE saved_items SET embedding_vec = $1::vector WHERE id = $2",
                (vec_text, row_id),
            )
            converted += 1

        # HNSW cosine index for ``<=>`` queries.
        await db.execute(
            """CREATE INDEX IF NOT EXISTS idx_saved_items_embedding_vec_hnsw
               ON saved_items USING hnsw (embedding_vec vector_cosine_ops)""",
            (),
        )

    logger.info(
        "saved_items Phase-2 PG migration complete: added embedding_vec "
        "vector(%d), backfilled %d rows, skipped %d mismatched, HNSW index.",
        dim, converted, skipped,
    )


async def _migrate_sqlite(db: "AsyncDatabase") -> None:
    """SQLite path — adds embedding_vec BLOB + copies bytes."""

    # SQLite's ``pragma_table_info`` lets us check column presence
    # idempotently. (``ALTER TABLE ADD COLUMN IF NOT EXISTS`` isn't a
    # thing on SQLite.)
    rows = await db.fetchall(
        "SELECT name FROM pragma_table_info('saved_items') WHERE name = 'embedding_vec'",
        (),
    )
    if rows:
        logger.debug(
            "saved_items.embedding_vec already present — skipping Phase-2 SQLite migration."
        )
        return

    table_exists = await db.fetchall(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='saved_items'",
        (),
    )
    if not table_exists:
        logger.debug(
            "saved_items table not yet present — skipping Phase-2 SQLite migration."
        )
        return

    async with db.transaction():
        await db.execute(
            "ALTER TABLE saved_items ADD COLUMN embedding_vec BLOB", ()
        )
        # Copy existing bytes so the ORM column has data on row 1.
        await db.execute(
            "UPDATE saved_items SET embedding_vec = embedding "
            "WHERE embedding IS NOT NULL",
            (),
        )

    logger.info(
        "saved_items Phase-2 SQLite migration complete: added embedding_vec "
        "BLOB, copied existing embeddings."
    )


# =============================================================================
# document_chunks (kestrel-sovereign #1447 follow-up — same pattern as
# saved_items, applied to AsyncRAGStore.)
# =============================================================================


async def migrate_document_chunks_add_embedding_vec(db: "AsyncDatabase") -> None:
    """Add a parallel ``embedding_vec`` column to ``document_chunks``
    and backfill from the existing ``embedding`` BYTEA / BLOB.

    Same shape as :func:`migrate_saved_items_add_embedding_vec` —
    different table. Idempotent, transactional, defers column creation
    on fresh DBs (no dim guess).
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        await _migrate_pg_table(db, "document_chunks", "chunk_id")
    elif backend_type == "sqlite":
        await _migrate_sqlite_table(db, "document_chunks")
    # Other dialects: no-op.


async def _migrate_pg_table(db: "AsyncDatabase", table: str, id_col: str) -> None:
    """Generic Postgres migration that adds ``embedding_vec`` to ``table``.

    Factored out so the saved_items + document_chunks paths share the
    cast-bytea-to-vector logic. The two callers differ only by table
    name + id column name.
    """
    rows = await db.fetchall(
        f"""SELECT 1 FROM information_schema.columns
            WHERE table_name = '{table}' AND column_name = 'embedding_vec'""",
        (),
    )
    if rows:
        logger.debug(
            "%s.embedding_vec already present — skipping Phase-2 PG migration.",
            table,
        )
        return

    src = await db.fetchall(
        f"""SELECT udt_name FROM information_schema.columns
            WHERE table_name = '{table}' AND column_name = 'embedding'""",
        (),
    )
    if not src:
        logger.debug(
            "%s table not yet present — skipping Phase-2 PG migration.",
            table,
        )
        return

    sample = await db.fetchall(
        f"""SELECT octet_length(embedding) FROM {table}
            WHERE embedding IS NOT NULL LIMIT 1""",
        (),
    )
    if not sample:
        logger.info(
            "%s has no embedded rows yet — deferring embedding_vec column "
            "creation until the next boot.",
            table,
        )
        return

    byte_len = sample[0][0]
    if byte_len % 4 != 0 or byte_len <= 0:
        logger.error(
            "%s.embedding has non-float32 byte length %d. Refusing to migrate.",
            table, byte_len,
        )
        return
    dim = byte_len // 4
    logger.info(
        "Sniffed embedding dimension %d (%d bytes) from existing %s rows.",
        dim, byte_len, table,
    )
    expected_bytes = dim * 4

    async with db.transaction():
        # pgvector extension MUST exist before the ALTER (codex review
        # on #1454).
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN embedding_vec vector({dim})", ()
        )

        all_rows = await db.fetchall(
            f"SELECT {id_col}, embedding FROM {table} WHERE embedding IS NOT NULL",
            (),
        )
        converted = 0
        skipped = 0
        for row_id, embedding_bytes in all_rows:
            if embedding_bytes is None:
                continue
            if len(embedding_bytes) != expected_bytes:
                logger.warning(
                    "Skipping %s row %s: %d bytes != expected %d.",
                    table, row_id, len(embedding_bytes), expected_bytes,
                )
                skipped += 1
                continue
            floats = struct.unpack(f"<{dim}f", bytes(embedding_bytes))
            vec_text = "[" + ",".join(repr(float(v)) for v in floats) + "]"
            await db.execute(
                f"UPDATE {table} SET embedding_vec = $1::vector WHERE {id_col} = $2",
                (vec_text, row_id),
            )
            converted += 1

        await db.execute(
            f"""CREATE INDEX IF NOT EXISTS idx_{table}_embedding_vec_hnsw
                ON {table} USING hnsw (embedding_vec vector_cosine_ops)""",
            (),
        )

    logger.info(
        "%s Phase-2 PG migration complete: added embedding_vec vector(%d), "
        "backfilled %d rows, skipped %d mismatched, HNSW index.",
        table, dim, converted, skipped,
    )


async def _migrate_sqlite_table(db: "AsyncDatabase", table: str) -> None:
    """Generic SQLite migration that adds ``embedding_vec`` BLOB to ``table``."""
    rows = await db.fetchall(
        f"SELECT name FROM pragma_table_info('{table}') WHERE name = 'embedding_vec'",
        (),
    )
    if rows:
        logger.debug(
            "%s.embedding_vec already present — skipping Phase-2 SQLite migration.",
            table,
        )
        return

    table_exists = await db.fetchall(
        f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}'",
        (),
    )
    if not table_exists:
        logger.debug(
            "%s table not yet present — skipping Phase-2 SQLite migration.",
            table,
        )
        return

    async with db.transaction():
        await db.execute(f"ALTER TABLE {table} ADD COLUMN embedding_vec BLOB", ())
        await db.execute(
            f"UPDATE {table} SET embedding_vec = embedding "
            f"WHERE embedding IS NOT NULL",
            (),
        )

    logger.info(
        "%s Phase-2 SQLite migration complete: added embedding_vec BLOB, "
        "copied existing embeddings.", table,
    )


# =============================================================================
# conversation_history (greenfield — no legacy embedding column to migrate
# from). Adds ``embedding_vec`` at the configured dim plus HNSW on PG.
# This prepares the storage side for MemoryRetriever cosine scoring;
# the current retriever still falls back to keyword/concept overlap
# until the embedding writer/read path is wired through.
# =============================================================================


async def migrate_conversation_history_add_embedding_vec(db: "AsyncDatabase") -> None:
    """Add an ``embedding_vec`` column to ``conversation_history``.

    Greenfield migration — there is NO pre-existing ``embedding``
    column on ``conversation_history`` to copy from, so the dim is
    picked from
    :data:`~kestrel_sovereign.storage.sqla.conversation_message.CONVERSATION_MESSAGE_EMBEDDING_DIM`
    (driven by the ``KESTREL_EMBEDDING_DIM`` env var; default 768 for
    Ollama ``nomic-embed-text``). Operators that switch models AFTER
    rows have been embedded need an explicit re-embedding script;
    this migration won't drop or resize the column. The read path is
    intentionally staged: the column exists before MemoryRetriever
    starts depending on it for cosine semantic scoring.

    Idempotent: skips cleanly if the column already exists. Wrapped
    in a transaction so a partial failure rolls back.
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        await _migrate_pg_greenfield(
            db,
            table="conversation_history",
        )
    elif backend_type == "sqlite":
        await _migrate_sqlite_greenfield(
            db,
            table="conversation_history",
        )
    # Other dialects: no-op.


async def _migrate_pg_greenfield(db: "AsyncDatabase", *, table: str) -> None:
    """Postgres greenfield migration — add ``embedding_vec vector(N)`` +
    HNSW on a table that has no existing embedding column.

    The dim is read lazily from
    ``conversation_message.CONVERSATION_MESSAGE_EMBEDDING_DIM`` rather
    than passed as an argument, so callers don't have to plumb the
    same constant through. Local import avoids a module-import cycle
    with ``sqla/__init__.py``.
    """
    # Local import: ``sqla.__init__`` imports this module at package
    # load, which would otherwise close the cycle.
    from .conversation_message import CONVERSATION_MESSAGE_EMBEDDING_DIM

    dim = CONVERSATION_MESSAGE_EMBEDDING_DIM

    rows = await db.fetchall(
        f"""SELECT 1 FROM information_schema.columns
            WHERE table_name = '{table}' AND column_name = 'embedding_vec'""",
        (),
    )
    if rows:
        logger.debug(
            "%s.embedding_vec already present — skipping greenfield PG migration.",
            table,
        )
        return

    table_exists = await db.fetchall(
        f"""SELECT 1 FROM information_schema.tables
            WHERE table_name = '{table}'""",
        (),
    )
    if not table_exists:
        logger.debug(
            "%s table not yet present — skipping greenfield PG migration.", table,
        )
        return

    async with db.transaction():
        # Extension MUST exist before ``ALTER TABLE`` references
        # ``vector(N)``. Mirrors the lesson from #1454 — original order
        # broke on fresh PG without the extension preloaded.
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN embedding_vec vector({dim})", (),
        )
        await db.execute(
            f"""CREATE INDEX IF NOT EXISTS idx_{table}_embedding_vec_hnsw
                ON {table} USING hnsw (embedding_vec vector_cosine_ops)""",
            (),
        )

    logger.info(
        "%s greenfield PG migration complete: added embedding_vec vector(%d), "
        "HNSW index. No backfill (greenfield column).", table, dim,
    )


# --- #1477 embedding_profile_id stamping ------------------------------------

async def migrate_conversation_lexical_index(db: "AsyncDatabase") -> None:
    """Create the #2339 keyed blind-token index on both storage backends."""
    async with db.transaction():
        if db.backend_type == "postgres":
            await db.execute(
                "ALTER TABLE conversation_history ADD COLUMN IF NOT EXISTS "
                "lexical_index_id TEXT"
            )
            await db.execute(
                "ALTER TABLE conversation_history ADD COLUMN IF NOT EXISTS "
                "lexical_index_version TEXT"
            )
        else:
            for column in ("lexical_index_id", "lexical_index_version"):
                exists = await db.fetchone(
                    "SELECT COUNT(*) FROM pragma_table_info('conversation_history') "
                    "WHERE name = ?",
                    (column,),
                )
                if not exists or not exists[0]:
                    await db.execute(
                        f"ALTER TABLE conversation_history ADD COLUMN {column} TEXT"
                    )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS conversation_lexical_tokens ("
            "agent_id TEXT NOT NULL, lexical_index_id TEXT NOT NULL, "
            "token_hash TEXT NOT NULL, "
            "PRIMARY KEY (agent_id, lexical_index_id, token_hash))"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_lexical_token_lookup "
            "ON conversation_lexical_tokens(agent_id, token_hash)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_lexical_coverage "
            "ON conversation_history(agent_id, lexical_index_version, "
            "lexical_index_id)"
        )
        # Orphan-token cleanup probes by agent + stable blind-index key.  The
        # coverage index above cannot serve that lookup efficiently because
        # lexical_index_version sits between those columns.  Without this
        # index a cleanup after a large backfill degenerates into one full
        # per-agent history scan for every token row.
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_lexical_message "
            "ON conversation_history(agent_id, lexical_index_id)"
        )


async def migrate_add_embedding_profile_id(
    db: "AsyncDatabase", *, table: str
) -> None:
    """Add a nullable ``embedding_profile_id`` column to ``table``.

    Idempotent across both backends and tables — pre-checks the
    column via ``information_schema`` / ``pragma_table_info``.
    Wrapped in ``db.transaction()`` so a partial failure rolls back
    cleanly. Existing rows stay NULL; profile-filtered kNN will skip
    them so a deployment that upgrades into 0.21 sees no false
    positives from un-stamped rows. Operators can backfill with the
    ``kestrel-sovereign embeddings reindex`` subcommand once per
    agent.
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        rows = await db.fetchall(
            f"""SELECT 1 FROM information_schema.columns
                WHERE table_name = '{table}'
                  AND column_name = 'embedding_profile_id'""",
            (),
        )
        if rows:
            logger.debug(
                "%s.embedding_profile_id already present — skipping #1477 PG migration.",
                table,
            )
            return
        table_exists = await db.fetchall(
            f"""SELECT 1 FROM information_schema.tables
                WHERE table_name = '{table}'""",
            (),
        )
        if not table_exists:
            logger.debug(
                "%s table not yet present — skipping #1477 PG migration.", table,
            )
            return
        async with db.transaction():
            await db.execute(
                f"ALTER TABLE {table} ADD COLUMN embedding_profile_id TEXT",
                (),
            )
            # An index on the new column accelerates the profile
            # filter in the kNN WHERE clause; without it pgvector
            # has to scan every row that matches the other filters
            # before applying the cosine sort. Cheap to add on a
            # nullable column.
            await db.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_{table}_embedding_profile_id
                    ON {table}(embedding_profile_id)
                    WHERE embedding_profile_id IS NOT NULL""",
                (),
            )
        logger.info(
            "%s #1477 PG migration complete: added embedding_profile_id TEXT + "
            "partial index.", table,
        )
    elif backend_type == "sqlite":
        rows = await db.fetchall(
            f"SELECT name FROM pragma_table_info('{table}') "
            f"WHERE name = 'embedding_profile_id'",
            (),
        )
        if rows:
            logger.debug(
                "%s.embedding_profile_id already present — skipping #1477 SQLite migration.",
                table,
            )
            return
        table_exists = await db.fetchall(
            f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}'",
            (),
        )
        if not table_exists:
            logger.debug(
                "%s table not yet present — skipping #1477 SQLite migration.",
                table,
            )
            return
        async with db.transaction():
            await db.execute(
                f"ALTER TABLE {table} ADD COLUMN embedding_profile_id TEXT", (),
            )
            await db.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_{table}_embedding_profile_id
                    ON {table}(embedding_profile_id)""",
                (),
            )
        logger.info(
            "%s #1477 SQLite migration complete: added embedding_profile_id TEXT.",
            table,
        )


async def migrate_create_embedding_profiles(db: "AsyncDatabase") -> None:
    """Create the ``embedding_profiles`` registry table (#1477).

    Tiny operator-visibility table — one row per
    ``(provider, model, dim, space_id, normalized)`` seen in the
    deployment. Storage code upserts on every successful write; the
    audit CLI reads it. The kNN filter does NOT join this table — it
    matches against the stamped id directly — so this is purely a
    human-readable mapping.

    Idempotent + transactional.
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        rows = await db.fetchall(
            """SELECT 1 FROM information_schema.tables
               WHERE table_name = 'embedding_profiles'""",
            (),
        )
        if rows:
            logger.debug(
                "embedding_profiles already present — skipping #1477 PG migration."
            )
            return
        async with db.transaction():
            await db.execute(
                """CREATE TABLE embedding_profiles (
                    id          TEXT PRIMARY KEY,
                    provider    TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    dim         INTEGER NOT NULL,
                    space_id    TEXT NOT NULL,
                    normalized  BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""",
                (),
            )
        logger.info("embedding_profiles created (PG, #1477).")
    elif backend_type == "sqlite":
        rows = await db.fetchall(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='embedding_profiles'",
            (),
        )
        if rows:
            logger.debug(
                "embedding_profiles already present — skipping #1477 SQLite migration."
            )
            return
        async with db.transaction():
            await db.execute(
                """CREATE TABLE embedding_profiles (
                    id          TEXT PRIMARY KEY,
                    provider    TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    dim         INTEGER NOT NULL,
                    space_id    TEXT NOT NULL,
                    normalized  INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                )""",
                (),
            )
        logger.info("embedding_profiles created (SQLite, #1477).")


async def migrate_embedding_profiles_add_parity(db: "AsyncDatabase") -> None:
    """Add ``parity_cosine`` to ``embedding_profiles`` (#2290).

    Records the measured worst-case pairwise cosine from the shared-space
    parity probe on the pinned space's registry row, so operators can see how
    far two servings of the same weights drifted before the alias was accepted.
    Nullable and descriptive — not part of the profile-id hash and never read
    by the kNN filter. Idempotent: skips when the column already exists.
    """
    backend_type = getattr(db, "backend_type", None)
    if backend_type == "postgres":
        rows = await db.fetchall(
            """SELECT 1 FROM information_schema.columns
               WHERE table_name = 'embedding_profiles'
                 AND column_name = 'parity_cosine'""",
            (),
        )
        if rows:
            return
        async with db.transaction():
            await db.execute(
                "ALTER TABLE embedding_profiles ADD COLUMN parity_cosine "
                "DOUBLE PRECISION",
                (),
            )
        logger.info("embedding_profiles.parity_cosine added (PG, #2290).")
    elif backend_type == "sqlite":
        cols = await db.fetchall(
            "PRAGMA table_info(embedding_profiles)", ()
        )
        names = {row[1] for row in cols} if cols else set()
        if not names or "parity_cosine" in names:
            # No table yet (create migration will make it) or already migrated.
            return
        async with db.transaction():
            await db.execute(
                "ALTER TABLE embedding_profiles ADD COLUMN parity_cosine REAL",
                (),
            )
        logger.info("embedding_profiles.parity_cosine added (SQLite, #2290).")


async def _migrate_sqlite_greenfield(db: "AsyncDatabase", *, table: str) -> None:
    """SQLite greenfield migration — add ``embedding_vec BLOB`` to a
    table that has no existing embedding column.

    No backfill, no copy. SQLite has no HNSW equivalent; the
    PurePythonBackend reads the column directly and computes cosine
    in Python. FEAT-8's ``SqliteVecBackend`` would add a virtual
    table later; that's not coupled to this migration.
    """
    rows = await db.fetchall(
        f"SELECT name FROM pragma_table_info('{table}') WHERE name = 'embedding_vec'",
        (),
    )
    if rows:
        logger.debug(
            "%s.embedding_vec already present — skipping greenfield SQLite migration.",
            table,
        )
        return

    table_exists = await db.fetchall(
        f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}'",
        (),
    )
    if not table_exists:
        logger.debug(
            "%s table not yet present — skipping greenfield SQLite migration.", table,
        )
        return

    async with db.transaction():
        await db.execute(f"ALTER TABLE {table} ADD COLUMN embedding_vec BLOB", ())

    logger.info(
        "%s greenfield SQLite migration complete: added embedding_vec BLOB. "
        "No backfill (greenfield column).", table,
    )


# =============================================================================
# compress → compact terminology rename (session-context shrinking)
# =============================================================================


async def migrate_compaction_terminology(db: "AsyncDatabase") -> None:
    """One-time data rewrite for the compress → compact terminology
    rename: session-context shrinking is "compaction" (the industry
    term), and the persisted metadata strings move with the code so
    readers never need dual-string compat.

    Rewrites ``conversation_history.metadata`` (plaintext JSON TEXT on
    both dialects — only ``content`` is encrypted at rest, see #1401):

    - ``type: "compression"`` → ``"compaction"``
    - ``type: "hierarchical_compression"`` → ``"hierarchical_compaction"``
    - key ``messages_compressed`` → ``messages_compacted``
    - key ``compressed_at`` → ``compacted_at``
    - ``salvage_reason: "manual-compress"`` → ``"manual-compact"``
    - ``excluded_reason: "Replaced by compression"`` → ``"Replaced by compaction"``

    Legacy ``[COMPRESSED CONTEXT …]`` / ``[HIERARCHICAL COMPRESSION …]``
    ``content`` markers are deliberately left alone: content may be
    encrypted at rest, and no code path parses the marker text — it is
    display prose inside the message body.

    Idempotent by construction: rewritten rows no longer match the
    LIKE filter, so re-running is a no-op. The migration runs on every
    boot (no completion sentinel exists), so the filter is kept tight:
    ``compression"`` matches all three quoted JSON values
    (``"compression"``, ``"hierarchical_compression"``, ``"Replaced by
    compression"``) and ``manual-compress`` matches the salvage reason
    — without fetching ordinary rows whose metadata merely mentions the
    word compress. The key renames (``messages_compressed``,
    ``compressed_at``) only occur on marker rows already matched by the
    type filter. Dialect-neutral: plain ``?`` placeholders, no DDL.
    """
    rows = await db.fetchall(
        "SELECT id, metadata FROM conversation_history "
        "WHERE metadata LIKE '%compression\"%' "
        "   OR metadata LIKE '%manual-compress%'",
        (),
    )
    if not rows:
        return

    rewritten = 0
    async with db.transaction():
        for row_id, raw in rows:
            if not raw:
                continue
            try:
                meta = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(meta, dict):
                continue

            new_meta = dict(meta)
            marker_type = new_meta.get("type")
            if marker_type == "compression":
                new_meta["type"] = "compaction"
            elif marker_type == "hierarchical_compression":
                new_meta["type"] = "hierarchical_compaction"
            if "messages_compressed" in new_meta:
                new_meta["messages_compacted"] = new_meta.pop("messages_compressed")
            if "compressed_at" in new_meta:
                new_meta["compacted_at"] = new_meta.pop("compressed_at")
            if new_meta.get("salvage_reason") == "manual-compress":
                new_meta["salvage_reason"] = "manual-compact"
            if new_meta.get("excluded_reason") == "Replaced by compression":
                new_meta["excluded_reason"] = "Replaced by compaction"

            if new_meta == meta:
                continue
            await db.execute(
                "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                (json.dumps(new_meta), row_id),
            )
            rewritten += 1

    if rewritten:
        logger.info(
            "compaction-terminology migration: rewrote %d "
            "conversation_history metadata row(s) (compression → "
            "compaction).", rewritten,
        )


def _extract_session_id(raw):
    """Return ``(meta_dict, session_id)`` from a metadata JSON string.

    ``(None, None)`` if the row has no metadata or it isn't a JSON object.
    """
    if not raw:
        return None, None
    try:
        meta = json.loads(raw)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(meta, dict):
        return None, None
    return meta, meta.get("session_id")


async def migrate_canonical_session_ids(db: "AsyncDatabase") -> None:
    """One-time relink of non-canonical (integer) session ids (#2012).

    The conversation-list endpoint historically keyed each session by the
    row-id of its first message, so the web UI round-tripped a bare integer
    (e.g. ``"1314"``) as the ``session_id`` on the next turn. Continued
    messages were then persisted with ``metadata.session_id = "1314"`` while
    the session's own ``new_session`` marker carried a UUID — splitting one
    conversation across two keys. On a hard refresh the message pane rendered
    empty even though every row was intact in the DB (identity mismatch, no
    data loss).

    This rewrites every integer ``metadata.session_id`` that names a
    ``new_session`` marker row to that marker's UUID, in both
    ``conversation_history`` and ``conversation_titles`` (so user-assigned
    names follow their conversation). Integer session_ids that name a genuine
    legacy time-gap anchor (a plain first message with no marker UUID) are
    left unchanged — the read path still resolves those via the row-id
    fallback.

    Idempotent by construction: a rewritten row carries a UUID session_id,
    which no longer matches the integer condition, so re-running is a no-op.
    Dialect-neutral: plain ``?`` placeholders, no DDL. Plaintext metadata
    only — ``content`` encryption at rest is untouched (#1401).
    """
    # Fetch every row carrying a session_id once, in id (≈chronological)
    # order, and reuse it for both map-building and the rewrite. Sessions are
    # agent-scoped, so collision analysis is keyed by (agent_id, UUID) — a UUID
    # reused across agents (imported/restored data) is NOT a collision.
    history_rows = await db.fetchall(
        "SELECT id, agent_id, metadata FROM conversation_history "
        "WHERE metadata LIKE '%session_id%' "
        "ORDER BY id ASC",
        (),
    )

    # Markers (new_session rows) grouped by (agent_id, UUID). The inheritance
    # bug can put the SAME UUID on several markers, each anchoring a DISTINCT
    # conversation whose turns are still keyed by that marker's integer row-id.
    markers_by_key: dict[tuple, list[int]] = {}
    marker_key_of_rowid: dict[int, tuple] = {}
    for row_id, agent_id, raw in history_rows:
        meta, sid = _extract_session_id(raw)
        if meta is None or sid is None:
            continue
        sid_str = str(sid)
        if not sid_str.isdigit() and meta.get("new_session"):
            key = (agent_id, sid_str)
            markers_by_key.setdefault(key, []).append(row_id)
            marker_key_of_rowid[row_id] = key

    # Attribute every CONTENT (non-marker) row to the conversation it belongs
    # to, then count how many distinct conversations share each (agent, UUID).
    # A UUID is safe to consolidate iff it covers a SINGLE conversation.
    #
    # Attribution:
    #  - integer-keyed content (``session_id`` == a marker's row-id) belongs to
    #    THAT marker;
    #  - UUID-keyed content belongs to the most recent ``new_session`` marker
    #    carrying that UUID and preceding it (by id). UUID-keyed content with
    #    NO preceding marker is an ``orphan`` — a prior implicit conversation
    #    that owned the UUID before any marker inherited it.
    markers_with_content: set[int] = set()
    orphan_keys: set[tuple] = set()

    # Integer-keyed content → its marker (row-ids are a global PK, so the
    # integer unambiguously names one marker regardless of agent).
    for row_id, agent_id, raw in history_rows:
        meta, sid = _extract_session_id(raw)
        if meta is None or sid is None or meta.get("new_session"):
            continue
        sid_str = str(sid)
        if sid_str.isdigit():
            mid = int(sid_str)
            if mid in marker_key_of_rowid:
                markers_with_content.add(mid)

    # UUID-keyed rows (markers + content), walked per (agent, UUID) in id order
    # so each content row is charged to its most recent preceding marker.
    rows_by_key: dict[tuple, list[tuple]] = {}
    for row_id, agent_id, raw in history_rows:
        meta, sid = _extract_session_id(raw)
        if meta is None or sid is None:
            continue
        sid_str = str(sid)
        if sid_str.isdigit():
            continue
        key = (agent_id, sid_str)
        if key not in markers_by_key:
            continue
        rows_by_key.setdefault(key, []).append((row_id, bool(meta.get("new_session"))))
    for key, items in rows_by_key.items():
        items.sort()
        current_marker = None
        for rid, is_marker in items:
            if is_marker:
                current_marker = rid
            elif current_marker is None:
                orphan_keys.add(key)
            else:
                markers_with_content.add(current_marker)

    # Consolidate a UUID only when it covers exactly one conversation.
    marker_uuid_by_rowid: dict[str, str] = {}
    skipped_inherited = 0
    for key, marker_ids in markers_by_key.items():
        uuid = key[1]
        conversations = sum(1 for m in marker_ids if m in markers_with_content)
        if key in orphan_keys:
            conversations += 1
        if conversations <= 1:
            # Map only markers that BELONG to the single conversation: the one
            # content-bearing marker (at most one, since conversations<=1), or
            # a degenerate sole marker (titled but empty). An empty marker that
            # merely INHERITED the UUID — when the content is owned by another
            # marker or a prior orphan — is a separate session; mapping it would
            # move its own title onto the owner's conversation.
            sole_marker = len(marker_ids) == 1
            for marker_row_id in marker_ids:
                owns = marker_row_id in markers_with_content
                if owns or (sole_marker and key not in orphan_keys):
                    marker_uuid_by_rowid[str(marker_row_id)] = uuid
        else:
            skipped_inherited += len(marker_ids)

    if skipped_inherited:
        logger.info(
            "canonical-session-id migration (#2012): skipped %d new_session "
            "marker(s) whose UUID collided across distinct conversations "
            "(ambiguous — not relinked to avoid merging).", skipped_inherited,
        )

    if not marker_uuid_by_rowid:
        return

    # Rewrite conversation_history rows whose session_id is an integer naming
    # one of those owned markers.
    rewritten_history = 0
    async with db.transaction():
        for row_id, _agent_id, raw in history_rows:
            meta, sid = _extract_session_id(raw)
            if meta is None or sid is None:
                continue
            sid_str = str(sid)
            if not sid_str.isdigit():
                continue
            canonical = marker_uuid_by_rowid.get(sid_str)
            if not canonical or canonical == sid_str:
                continue
            new_meta = dict(meta)
            new_meta["session_id"] = canonical
            await db.execute(
                "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                (json.dumps(new_meta), row_id),
            )
            rewritten_history += 1

    # Step 3: remap user-assigned conversation names keyed by the integer
    # session_id. PK is (agent_id, session_id); if a UUID-keyed name already
    # exists it is authoritative, so drop the stale integer row rather than
    # colliding on the upsert.
    title_rows = await db.fetchall(
        "SELECT agent_id, session_id FROM conversation_titles",
        (),
    )
    remapped_titles = 0
    async with db.transaction():
        for agent_id, sid in title_rows:
            sid_str = str(sid)
            canonical = marker_uuid_by_rowid.get(sid_str)
            if not canonical or canonical == sid_str:
                continue
            existing = await db.fetchone(
                "SELECT 1 FROM conversation_titles "
                "WHERE agent_id = ? AND session_id = ?",
                (agent_id, canonical),
            )
            if existing:
                await db.execute(
                    "DELETE FROM conversation_titles "
                    "WHERE agent_id = ? AND session_id = ?",
                    (agent_id, sid_str),
                )
            else:
                await db.execute(
                    "UPDATE conversation_titles SET session_id = ? "
                    "WHERE agent_id = ? AND session_id = ?",
                    (canonical, agent_id, sid_str),
                )
            remapped_titles += 1

    if rewritten_history or remapped_titles:
        logger.info(
            "canonical-session-id migration (#2012): relinked %d "
            "conversation_history row(s) and %d conversation name(s) "
            "from integer keys to marker UUIDs.",
            rewritten_history, remapped_titles,
        )


async def migrate_legacy_graph_fact_migration_state(db: "AsyncDatabase") -> None:
    """Install #2752's additive, tenant-scoped operator-run bookkeeping.

    This deliberately creates no data migration and does not inspect or alter
    ``graph_nodes``.  It is shared DDL for SQLite/PostgreSQL; the actual
    migration remains an explicit call on agent-bound storage.
    """
    async with db.transaction():
        await db.execute(
            "CREATE TABLE IF NOT EXISTS legacy_fact_migration_records ("
            "tenant_id TEXT NOT NULL, node_id TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, source_occurrence_id TEXT, "
            "assertion_id TEXT, revision_id TEXT, outcome TEXT NOT NULL, "
            "created_at TIMESTAMP NOT NULL, "
            "PRIMARY KEY (tenant_id, node_id))",
            (),
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS legacy_fact_migration_checkpoints ("
            "tenant_id TEXT NOT NULL, migration_name TEXT NOT NULL, "
            "last_node_id TEXT, state TEXT NOT NULL, updated_at TIMESTAMP NOT NULL, "
            "PRIMARY KEY (tenant_id, migration_name))",
            (),
        )
