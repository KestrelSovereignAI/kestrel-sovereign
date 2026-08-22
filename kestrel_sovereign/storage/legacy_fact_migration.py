"""Bounded, restart-safe import of the one verified legacy fact shape.

``graph_nodes`` remains an application graph; it is *not* a semantic assertion
store.  This module deliberately accepts only the historical ``fact`` node
shape documented in :data:`LEGACY_FACT_SHAPE`.  In particular it never infers
meaning from a label, graph edge, arbitrary JSON key, or ``properties.agent_id``.

The runner is constructed from an already agent-bound :class:`AsyncStorage`.
That matters: graph ownership comes from ``graph_node_owners`` and canonical
authority comes from the storage boot capability, rather than from a value in
the legacy JSON.  Each canonical proposal is sent through the normal governed
SHACL write path.  Legacy rows are never changed or removed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from kestrel_sovereign.knowledge import (
    Assertion,
    DirectLineage,
    EpistemicState,
    SourceOccurrence,
)
from kestrel_sovereign.knowledge.shacl_validation import ValidationSource

from .timestamps import utc_timestamp_parameter

if TYPE_CHECKING:
    from .async_storage import AsyncStorage


MIGRATION_NAME = "legacy_graph_fact_to_assertion_v1"
MIGRATION_VERSION = "legacy-graph-fact-v1"
_INVALIDATION_PAGE_SIZE = 500
_MAX_INVALIDATION_PAGES = 20
LEGACY_FACT_SHAPE = {
    "node_type": "fact",
    "required_properties": ("subject", "predicate", "value", "created_at"),
    "ownership": "exactly one graph_node_owners ledger row matching the bound tenant",
}
_SAFE_REJECTION_CODES = frozenset(
    {
        "invalid_node_id",
        "malformed_properties",
        "missing_or_invalid_fact_fields",
        "missing_or_invalid_created_at",
        "invalid_confidence",
        "invalid_unicode",
        "shared_or_ambiguous_ownership",
        "unsupported_semantic_mapping",
    }
)


class LegacyFactMigrationError(ValueError):
    """An operator asked the migration to cross an unsafe boundary."""


@dataclass(frozen=True, slots=True)
class LegacyFactCandidate:
    """A parsed legacy row with values kept in-memory only."""

    node_id: str
    subject: str
    predicate: str
    value: str
    created_at: str
    confidence: Decimal
    content_hash: str


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """Content-safe inventory for an operator before any write."""

    tenant_id: str
    target_ontology: str
    target_ontology_version: str
    scanned: int
    eligible: int
    already_recorded: int
    recorded_by_outcome: Mapping[str, int]
    rejected: Mapping[str, int]
    by_agent: Mapping[str, int]
    by_source: Mapping[str, int]
    content_hashes: tuple[str, ...]
    truncated: bool
    compatibility_flag_enabled: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "migration": MIGRATION_NAME,
            "tenant_id": self.tenant_id,
            "target_ontology": self.target_ontology,
            "target_ontology_version": self.target_ontology_version,
            "scanned": self.scanned,
            "eligible": self.eligible,
            "already_recorded": self.already_recorded,
            "recorded_by_outcome": dict(self.recorded_by_outcome),
            "rejected": dict(self.rejected),
            "by_agent": dict(self.by_agent),
            "by_source": dict(self.by_source),
            "content_hashes": list(self.content_hashes),
            "truncated": self.truncated,
            "compatibility_flag_enabled": self.compatibility_flag_enabled,
        }


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """A bounded migration or rollback result, without fact text."""

    tenant_id: str
    processed: int
    migrated: int
    idempotent: int
    rejected: Mapping[str, int]
    checkpoint: str | None
    complete: bool
    index_invalidation_requested: bool


@dataclass(frozen=True, slots=True)
class MigrationSourceReview:
    """Content-safe source-set drift audit required before checkpoint resume."""

    reviewed: int
    late_added_before_checkpoint: int
    changed: int
    missing_or_unowned: int

    @property
    def requires_reset(self) -> bool:
        return self.late_added_before_checkpoint > 0

    @property
    def requires_operator_review(self) -> bool:
        return self.changed > 0 or self.missing_or_unowned > 0


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _safe_rejection_code(error: Exception) -> str:
    """Convert legacy/parser errors to stable diagnostics without echoing data."""
    if isinstance(error, LegacyFactMigrationError):
        candidate = str(error)
        if candidate in _SAFE_REJECTION_CODES:
            return candidate
    return "invalid_legacy_fact"


def _map_verified_legacy_fact(*, subject: str, predicate: str, value: str, tenant_id: str):
    """Load the feature-owned local vocabulary only when migration runs.

    Core storage must remain importable without loading the memory-agency
    feature.  The feature owns the closed local-term mapping; this adapter
    translates its data-bearing error into a fixed migration diagnostic.
    """
    from kestrel_sovereign.features.memory_agency.semantic_facts import (
        FactMappingError,
        map_legacy_fact,
    )

    try:
        return map_legacy_fact(subject, predicate, value, tenant_id=tenant_id)
    except FactMappingError as error:
        raise LegacyFactMigrationError("unsupported_semantic_mapping") from error


def _raw_properties_hash(node_id: object, raw_properties: object) -> str:
    """Hash malformed/rejected source bytes without rendering their content."""
    if not isinstance(node_id, str) or not isinstance(raw_properties, str):
        return "sha256:" + hashlib.sha256(b"non-string-legacy-source").hexdigest()
    encoded = node_id.encode("utf-8", "surrogatepass") + b"\0" + raw_properties.encode(
        "utf-8", "surrogatepass"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _utc_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise LegacyFactMigrationError("missing_or_invalid_created_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LegacyFactMigrationError("missing_or_invalid_created_at") from error
    if parsed.tzinfo is None:
        raise LegacyFactMigrationError("missing_or_invalid_created_at")
    return parsed.astimezone(timezone.utc).isoformat()


def _legacy_candidate(node_id: object, raw_properties: object) -> LegacyFactCandidate:
    """Parse exactly the documented shape, rejecting without logging values."""
    if not isinstance(node_id, str) or not node_id:
        raise LegacyFactMigrationError("invalid_node_id")
    if not isinstance(raw_properties, str):
        raise LegacyFactMigrationError("malformed_properties")
    try:
        properties = json.loads(raw_properties)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise LegacyFactMigrationError("malformed_properties") from error
    if not isinstance(properties, dict):
        raise LegacyFactMigrationError("malformed_properties")
    values = {key: properties.get(key) for key in ("subject", "predicate", "value")}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise LegacyFactMigrationError("missing_or_invalid_fact_fields")
    try:
        node_id.encode("utf-8")
        for value in values.values():
            value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise LegacyFactMigrationError("invalid_unicode") from error
    created_at = _utc_timestamp(properties.get("created_at"))
    raw_confidence = properties.get("confidence", "1")
    if isinstance(raw_confidence, bool):
        raise LegacyFactMigrationError("invalid_confidence")
    try:
        confidence = Decimal(str(raw_confidence))
    except (InvalidOperation, ValueError) as error:
        raise LegacyFactMigrationError("invalid_confidence") from error
    if not confidence.is_finite() or not Decimal("0") <= confidence <= Decimal("1"):
        raise LegacyFactMigrationError("invalid_confidence")
    content_hash = _canonical_hash(
        {
            "node_id": node_id,
            "subject": values["subject"],
            "predicate": values["predicate"],
            "value": values["value"],
            "created_at": created_at,
            "confidence": format(confidence, "f"),
        }
    )
    return LegacyFactCandidate(
        node_id=node_id,
        subject=values["subject"],
        predicate=values["predicate"],
        value=values["value"],
        created_at=created_at,
        confidence=confidence,
        content_hash=content_hash,
    )


class LegacyGraphFactMigration:
    """Migrate a single authenticated tenant in restart-safe bounded batches.

    ``compatibility_read_enabled`` has no write effect.  It only enables the
    deterministic, content-safe equivalence measurement returned by
    :meth:`compatibility_metrics`; no application reader is diverted to legacy
    graph data and there is deliberately no dual-write path.
    """

    def __init__(
        self,
        storage: "AsyncStorage",
        *,
        compatibility_read_enabled: bool = False,
        index_invalidator: Callable[[str, Sequence[str]], Any] | None = None,
    ) -> None:
        if not isinstance(compatibility_read_enabled, bool):
            raise LegacyFactMigrationError("compatibility_read_enabled must be boolean")
        self._storage = storage
        self._compatibility_read_enabled = compatibility_read_enabled
        self._index_invalidator = index_invalidator

    async def _ready(self) -> tuple[Any, str]:
        if not self._storage._initialized:  # noqa: SLF001 - storage facade state gate
            await self._storage.initialize()
        binding = self._storage.semantic_assertion_binding()
        db = self._storage.db
        if db is None:
            raise RuntimeError("initialized storage has no database")
        if not isinstance(binding.tenant_id, str) or not binding.tenant_id:
            raise LegacyFactMigrationError("invalid_migration_tenant")
        return db, binding.tenant_id

    async def _rows_after(self, db: Any, tenant_id: str, after: str | None, limit: int):
        """Read fact nodes only through the authoritative ownership ledger."""
        if limit < 1 or limit > 500:
            raise LegacyFactMigrationError("batch limit must be in [1, 500]")
        predicate = "graph_nodes.node_id > ?" if after is not None else "1 = 1"
        parameters: tuple[object, ...] = (
            (tenant_id, after, limit) if after is not None else (tenant_id, limit)
        )
        return await db.fetchall(
            "SELECT graph_nodes.node_id, graph_nodes.properties, "
            "(SELECT COUNT(*) FROM graph_node_owners all_owners "
            " WHERE all_owners.node_id = graph_nodes.node_id) AS owner_count, "
            "(SELECT records.outcome FROM legacy_fact_migration_records records "
            " WHERE records.tenant_id = ? AND records.node_id = graph_nodes.node_id) AS recorded_outcome, "
            "(SELECT records.content_hash FROM legacy_fact_migration_records records "
            " WHERE records.tenant_id = ? AND records.node_id = graph_nodes.node_id) AS recorded_hash "
            "FROM graph_nodes JOIN graph_node_owners owner "
            " ON owner.node_id = graph_nodes.node_id AND owner.agent_id = ? "
            "WHERE graph_nodes.node_type = 'fact' AND " + predicate + " "
            "ORDER BY graph_nodes.node_id ASC LIMIT ?",
            ((tenant_id, tenant_id) + parameters),
        )

    async def plan(self, *, scan_limit: int = 500) -> MigrationPlan:
        """Return a dry-run inventory.  It has no writes and reveals no values."""
        if scan_limit < 1 or scan_limit > 5_000:
            raise LegacyFactMigrationError("scan_limit must be in [1, 5000]")
        db, tenant_id = await self._ready()
        # A plan never creates migration records/checkpoints and never writes
        # a legacy or canonical row.
        rows = await self._rows_after(db, tenant_id, None, min(scan_limit + 1, 500))
        # Larger plans use bounded pages rather than a single unbounded scan.
        while len(rows) < scan_limit + 1 and rows and len(rows) % 500 == 0:
            next_rows = await self._rows_after(db, tenant_id, rows[-1][0], min(500, scan_limit + 1 - len(rows)))
            rows.extend(next_rows)
            if not next_rows:
                break
        visible = rows[:scan_limit]
        rejected: dict[str, int] = {}
        hashes: list[str] = []
        eligible = recorded = 0
        recorded_by_outcome: dict[str, int] = {}
        for node_id, properties, owner_count, recorded_outcome, _recorded_hash in visible:
            if int(owner_count) != 1:
                rejected["shared_or_ambiguous_ownership"] = rejected.get("shared_or_ambiguous_ownership", 0) + 1
                continue
            try:
                candidate = _legacy_candidate(node_id, properties)
                _map_verified_legacy_fact(
                    subject=candidate.subject,
                    predicate=candidate.predicate,
                    value=candidate.value,
                    tenant_id=tenant_id,
                )
            except LegacyFactMigrationError as error:
                reason = _safe_rejection_code(error)
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            hashes.append(candidate.content_hash)
            if recorded_outcome is not None:
                recorded += 1
                recorded_by_outcome[str(recorded_outcome)] = (
                    recorded_by_outcome.get(str(recorded_outcome), 0) + 1
                )
            else:
                eligible += 1
        mapping = _map_verified_legacy_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="plan-probe",
            tenant_id=tenant_id,
        )
        return MigrationPlan(
            tenant_id=tenant_id,
            target_ontology=mapping.ontology.namespace,
            target_ontology_version=mapping.ontology.version,
            scanned=len(visible),
            eligible=eligible,
            already_recorded=recorded,
            recorded_by_outcome=recorded_by_outcome,
            rejected=rejected,
            by_agent={tenant_id: len(visible)},
            by_source={"legacy_graph_node": len(visible)},
            content_hashes=tuple(hashes),
            truncated=len(rows) > scan_limit,
            compatibility_flag_enabled=self._compatibility_read_enabled,
        )

    async def run(self, *, batch_size: int = 100, max_batches: int = 1) -> MigrationResult:
        """Migrate bounded pages and checkpoint only after each completed page."""
        if max_batches < 1 or max_batches > 1_000:
            raise LegacyFactMigrationError("max_batches must be in [1, 1000]")
        db, tenant_id = await self._ready()
        row = await db.fetchone(
            "SELECT last_node_id FROM legacy_fact_migration_checkpoints "
            "WHERE tenant_id = ? AND migration_name = ?",
            (tenant_id, MIGRATION_NAME),
        )
        cursor = str(row[0]) if row and row[0] is not None else None
        review = await self._review_source_set(db, tenant_id, cursor)
        if review.requires_operator_review:
            raise LegacyFactMigrationError("legacy_fact_source_review_required")
        if review.requires_reset:
            raise LegacyFactMigrationError("legacy_fact_checkpoint_reset_required")
        processed = migrated = idempotent = 0
        rejected: dict[str, int] = {}
        complete = False
        for _ in range(max_batches):
            rows = await self._rows_after(db, tenant_id, cursor, batch_size)
            if not rows:
                complete = True
                break
            for node_id, properties, owner_count, recorded_outcome, _recorded_hash in rows:
                processed += 1
                cursor = str(node_id)
                if recorded_outcome is not None:
                    idempotent += 1
                    continue
                if int(owner_count) != 1:
                    await self._record(
                        db, tenant_id, str(node_id), None,
                        "rejected:shared_or_ambiguous_ownership",
                        content_hash=_raw_properties_hash(node_id, properties),
                    )
                    rejected["shared_or_ambiguous_ownership"] = rejected.get("shared_or_ambiguous_ownership", 0) + 1
                    continue
                try:
                    candidate = _legacy_candidate(node_id, properties)
                    mapping = _map_verified_legacy_fact(
                        subject=candidate.subject,
                        predicate=candidate.predicate,
                        value=candidate.value,
                        tenant_id=tenant_id,
                    )
                except LegacyFactMigrationError as error:
                    reason = _safe_rejection_code(error)
                    await self._record(
                        db, tenant_id, str(node_id), None, f"rejected:{reason}",
                        content_hash=_raw_properties_hash(node_id, properties),
                    )
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                outcome, assertion_id, revision_id = await self._migrate_candidate(candidate, mapping)
                await self._record(
                    db,
                    tenant_id,
                    candidate.node_id,
                    candidate,
                    outcome,
                    assertion_id=assertion_id,
                    revision_id=revision_id,
                )
                if outcome in {"migrated", "source_appended"}:
                    migrated += 1
                elif outcome == "idempotent":
                    idempotent += 1
                else:
                    reason = outcome.removeprefix("rejected:")
                    rejected[reason] = rejected.get(reason, 0) + 1
            await self._checkpoint(db, tenant_id, cursor, "running")
            if len(rows) < batch_size:
                complete = True
                break
            # A page exactly equal to the batch size may still be terminal.
            # Check one bounded next page before reporting a resumable run as
            # incomplete; no migration state is advanced by this observation.
            if not await self._rows_after(db, tenant_id, cursor, 1):
                complete = True
                break
        if complete:
            await self._checkpoint(db, tenant_id, cursor, "complete")
        invalidated = await self._deliver_pending_invalidations(db, tenant_id)
        # There is no assertion vector/search index in core today.  A caller
        # with a feature-owned projection must provide its public invalidator;
        # silently guessing a private feature table would violate the boundary.
        return MigrationResult(tenant_id, processed, migrated, idempotent, rejected, cursor, complete, invalidated)

    async def review_source_set(self) -> MigrationSourceReview:
        """Inspect the entire source set before an explicit checkpoint reset.

        The audit is intentionally read-only and paged.  It prevents a
        completed keyset cursor from silently missing a later low-ID insert or
        accepting a corrected legacy source as though it were its old value.
        """
        db, tenant_id = await self._ready()
        row = await db.fetchone(
            "SELECT last_node_id FROM legacy_fact_migration_checkpoints "
            "WHERE tenant_id = ? AND migration_name = ?",
            (tenant_id, MIGRATION_NAME),
        )
        cursor = str(row[0]) if row and row[0] is not None else None
        return await self._review_source_set(db, tenant_id, cursor)

    async def reset_checkpoint_after_review(self) -> MigrationSourceReview:
        """Safely replay a source set only when it gained unrecorded low IDs.

        A changed or disappeared source needs operator adjudication (normally
        rollback plus a fresh governed assertion), never an automatic rewrite.
        Records remain in place, so reset replays only unrecorded nodes.
        """
        db, tenant_id = await self._ready()
        row = await db.fetchone(
            "SELECT last_node_id FROM legacy_fact_migration_checkpoints "
            "WHERE tenant_id = ? AND migration_name = ?",
            (tenant_id, MIGRATION_NAME),
        )
        cursor = str(row[0]) if row and row[0] is not None else None
        review = await self._review_source_set(db, tenant_id, cursor)
        if review.requires_operator_review:
            raise LegacyFactMigrationError("legacy_fact_source_review_required")
        if review.requires_reset:
            await self._checkpoint(db, tenant_id, None, "reset_after_review")
        return review

    async def rollback(self, *, batch_size: int = 100) -> MigrationResult:
        """Withdraw only assertions proven to be created by this migration.

        Rollback is explicit and reversible at the canonical lifecycle level;
        it never deletes legacy graph rows or migration audit records.
        """
        if batch_size < 1 or batch_size > 500:
            raise LegacyFactMigrationError("batch_size must be in [1, 500]")
        db, tenant_id = await self._ready()
        rows = await db.fetchall(
            "SELECT node_id, source_occurrence_id, assertion_id, revision_id FROM legacy_fact_migration_records "
            "WHERE tenant_id = ? AND outcome IN ('migrated', 'idempotent', 'source_appended') "
            "ORDER BY node_id ASC LIMIT ?",
            (tenant_id, batch_size),
        )
        processed = migrated = idempotent = 0
        rejected: dict[str, int] = {}
        groups: dict[str, list[tuple[object, object, object]]] = {}
        for node_id, source_id, assertion_id, _revision_id in rows:
            if not isinstance(assertion_id, str) or not assertion_id:
                groups.setdefault("", []).append((node_id, source_id, _revision_id))
            else:
                groups.setdefault(assertion_id, []).append(
                    (node_id, source_id, _revision_id)
                )
        for assertion_id, group in groups.items():
            if not assertion_id or any(
                not isinstance(source_id, str) or not source_id
                for _node_id, source_id, _revision_id in group
            ):
                processed += len(group)
                for node_id, _source_id, _revision_id in group:
                    await self._set_outcome(
                        db, tenant_id, node_id, "rollback_refused_invalid_record"
                    )
                rejected["rollback_refused_invalid_record"] = (
                    rejected.get("rollback_refused_invalid_record", 0) + len(group)
                )
                continue
            # Always reconstruct the complete active source group before
            # looking up the canonical assertion.  This is also the recovery
            # path when a prior process committed the canonical withdrawal but
            # crashed before finalizing migration receipts.
            all_active_records = await db.fetchall(
                "SELECT node_id, source_occurrence_id, revision_id "
                "FROM legacy_fact_migration_records "
                "WHERE tenant_id = ? AND assertion_id = ? "
                "AND outcome IN ('migrated', 'idempotent', 'source_appended') "
                "ORDER BY node_id ASC",
                (tenant_id, assertion_id),
            )
            group = [
                (node_id, source_id, revision_id)
                for node_id, source_id, revision_id in all_active_records
            ]
            processed += len(group)
            assertion = await self._storage.get_assertion(assertion_id)
            if assertion is None:
                await self._finalize_rollback_group(
                    db, tenant_id, assertion_id, group
                )
                idempotent += len(group)
                continue
            sources = await self._storage.list_assertion_sources(assertion_id)
            expected_source_ids = {
                str(source_id) for _node_id, source_id, _revision_id in group
            }
            actual_source_ids = {
                source.source_occurrence_id for source in sources
            }
            if actual_source_ids != expected_source_ids:
                for node_id, _source_id, _revision_id in group:
                    await self._set_outcome(
                        db, tenant_id, node_id, "rollback_refused_provenance"
                    )
                rejected["rollback_refused_provenance"] = (
                    rejected.get("rollback_refused_provenance", 0) + len(group)
                )
                continue
            await self._storage.delete_assertion(
                assertion_id,
                assertion.revision_id,
                operation_id=(
                    f"{MIGRATION_VERSION}:rollback:"
                    f"{hashlib.sha256('|'.join(sorted(expected_source_ids)).encode()).hexdigest()}"
                ),
            )
            await self._finalize_rollback_group(
                db, tenant_id, assertion_id, group
            )
            migrated += len(group)
        invalidated = await self._deliver_pending_invalidations(db, tenant_id)
        remaining = await db.fetchone(
            "SELECT COUNT(*) FROM legacy_fact_migration_records "
            "WHERE tenant_id = ? "
            "AND outcome IN ('migrated', 'idempotent', 'source_appended')",
            (tenant_id,),
        )
        return MigrationResult(
            tenant_id,
            processed,
            migrated,
            idempotent,
            rejected,
            None,
            bool(remaining and int(remaining[0]) == 0),
            invalidated,
        )

    async def compatibility_metrics(self) -> dict[str, object]:
        """Measure migration coverage without enabling a production dual read."""
        plan = await self.plan()
        if plan.truncated:
            return {
                "enabled": self._compatibility_read_enabled,
                "complete_inventory": False,
                "removal_safe": False,
                "reason": "inventory_truncated",
                "scanned": plan.scanned,
            }
        return {
            "enabled": self._compatibility_read_enabled,
            "complete_inventory": True,
            "removal_safe": False,
            "legacy_eligible": plan.eligible + plan.already_recorded,
            "canonical_recorded": (
                plan.recorded_by_outcome.get("migrated", 0)
                + plan.recorded_by_outcome.get("idempotent", 0)
            ),
            "unmigrated": plan.eligible,
            "removal_condition": "unmigrated == 0 and rejected counts reviewed by an operator",
        }

    async def _review_source_set(
        self,
        db: Any,
        tenant_id: str,
        cursor: str | None,
    ) -> MigrationSourceReview:
        """Compare all currently owned fact rows with durable migration receipts."""
        after: str | None = None
        reviewed = late_added = changed = 0
        seen_record_nodes: set[str] = set()
        while True:
            rows = await self._rows_after(db, tenant_id, after, 500)
            if not rows:
                break
            for node_id, properties, owner_count, outcome, recorded_hash in rows:
                node_id = str(node_id)
                after = node_id
                if outcome is not None:
                    seen_record_nodes.add(node_id)
                if int(owner_count) != 1:
                    if outcome in {"migrated", "idempotent", "source_appended"}:
                        changed += 1
                    continue
                try:
                    candidate = _legacy_candidate(node_id, properties)
                    _map_verified_legacy_fact(
                        subject=candidate.subject,
                        predicate=candidate.predicate,
                        value=candidate.value,
                        tenant_id=tenant_id,
                    )
                    current_hash = candidate.content_hash
                except LegacyFactMigrationError:
                    current_hash = _raw_properties_hash(node_id, properties)
                reviewed += 1
                if outcome is None:
                    if cursor is not None and node_id <= cursor:
                        late_added += 1
                elif str(recorded_hash) != current_hash:
                    changed += 1
            if len(rows) < 500:
                break
        records = await db.fetchall(
            "SELECT node_id FROM legacy_fact_migration_records WHERE tenant_id = ?",
            (tenant_id,),
        )
        missing_or_unowned = sum(
            1 for record in records if str(record[0]) not in seen_record_nodes
        )
        return MigrationSourceReview(
            reviewed=reviewed,
            late_added_before_checkpoint=late_added,
            changed=changed,
            missing_or_unowned=missing_or_unowned,
        )

    async def _migrate_candidate(
        self,
        candidate: LegacyFactCandidate,
        mapping: Any,
    ) -> tuple[str, str | None, str | None]:
        binding = self._storage.semantic_assertion_binding()
        digest = candidate.content_hash.removeprefix("sha256:")
        source = SourceOccurrence(
            source_occurrence_id=f"source:{MIGRATION_VERSION}:{digest}",
            source_kind="legacy_graph_node",
            locator=f"graph-node:{candidate.node_id}",
            received_at=candidate.created_at,
            content_digest=candidate.content_hash,
            actor=binding.owning_agent_id,
            selector="verified-fact-properties-v1",
        )
        assertion = Assertion(
            tenant_id=binding.tenant_id,
            owning_agent_id=binding.owning_agent_id,
            subject=mapping.subject,
            predicate=mapping.predicate,
            object=mapping.object,
            revision_id=source.source_occurrence_id,
            confidence=candidate.confidence,
            confidence_method=MIGRATION_VERSION,
            confidence_basis="verified-legacy-graph-shape",
            epistemic_state=EpistemicState.REPORTED,
            asserted_at=source.received_at,
            ontology_version=mapping.ontology,
            lineage=DirectLineage((source.source_occurrence_id,)),
            privacy_classification=binding.privacy_classification,
            release_policy_reference=binding.release_policy_reference,
            visibility=binding.visibility,
        )
        existing = await self._storage.get_assertion(assertion.assertion_id)
        if existing is not None:
            return await self._append_duplicate_source(
                existing,
                assertion=assertion,
                source=source,
                candidate=candidate,
            )
        result = await self._storage.put_validated_assertion(
            assertion,
            source_occurrences=(source,),
            source=ValidationSource.IMPORTED,
            operation_id=f"{MIGRATION_VERSION}:import:{digest}",
            run_id=MIGRATION_NAME,
        )
        if result.write is None:
            return f"rejected:{result.report.state.value}:{result.report.action.value}", None, None
        return (
            "idempotent" if result.write.idempotent else "migrated",
            result.write.assertion.assertion_id,
            result.write.assertion.revision_id,
        )

    async def _append_duplicate_source(
        self,
        existing: Assertion,
        *,
        assertion: Assertion,
        source: SourceOccurrence,
        candidate: LegacyFactCandidate,
    ) -> tuple[str, str | None, str | None]:
        """Attach a second verified legacy occurrence through public lifecycle.

        Equal assertion identities intentionally coalesce semantic claims.  A
        different legacy graph row is still independent evidence, so it must
        append one direct source revision—not attempt another initial write.
        """
        if (
            existing.confidence_method != MIGRATION_VERSION
            or existing.confidence_basis != "verified-legacy-graph-shape"
            or existing.confidence != candidate.confidence
            or not isinstance(existing.lineage, DirectLineage)
        ):
            return "rejected:canonical_claim_conflict", None, None
        sources = await self._storage.list_assertion_sources(existing.assertion_id)
        if source.source_occurrence_id in {
            item.source_occurrence_id for item in sources
        }:
            return "idempotent", existing.assertion_id, existing.revision_id
        replacement_mapping = existing.to_mapping()
        replacement_mapping.update(
            {
                "revision_id": source.source_occurrence_id,
                "asserted_at": source.received_at.to_mapping(),
                "lineage": DirectLineage(
                    (*existing.lineage.source_occurrence_ids, source.source_occurrence_id)
                ).to_mapping(),
                "supersedes_revision_id": None,
            }
        )
        replacement = Assertion.from_mapping(replacement_mapping)
        result = await self._storage.append_assertion_source(
            existing.revision_id,
            replacement,
            source_occurrences=(source,),
            operation_id=(
                f"{MIGRATION_VERSION}:append:"
                f"{candidate.content_hash.removeprefix('sha256:')}"
            ),
        )
        if result.write is None:
            return f"rejected:{result.report.state.value}:{result.report.action.value}", None, None
        return (
            "idempotent" if result.write.idempotent else "source_appended",
            result.replacement.assertion_id,
            result.replacement.revision_id,
        )

    @staticmethod
    async def _record(
        db: Any,
        tenant_id: str,
        node_id: str,
        candidate: LegacyFactCandidate | None,
        outcome: str,
        *,
        assertion_id: str | None = None,
        content_hash: str | None = None,
        revision_id: str | None = None,
    ) -> None:
        source_id = None if candidate is None else f"source:{MIGRATION_VERSION}:{candidate.content_hash.removeprefix('sha256:')}"
        revision_id = revision_id or source_id
        content_hash = content_hash or (None if candidate is None else candidate.content_hash)
        async with db.transaction():
            recorded_at = utc_timestamp_parameter(
                db.backend_type, datetime.now(timezone.utc)
            )
            await db.execute(
                "INSERT INTO legacy_fact_migration_records (tenant_id, node_id, content_hash, source_occurrence_id, assertion_id, revision_id, outcome, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(tenant_id, node_id) DO NOTHING",
                (
                    tenant_id,
                    node_id,
                    content_hash or "",
                    source_id,
                    assertion_id,
                    revision_id,
                    outcome,
                    recorded_at,
                ),
            )
            if assertion_id is not None and outcome in {"migrated", "idempotent", "source_appended"}:
                await db.execute(
                    "INSERT INTO legacy_fact_migration_invalidations "
                    "(tenant_id, migration_name, assertion_id, state, generation, created_at, delivered_at) "
                    "VALUES (?, ?, ?, 'pending', 1, ?, NULL) "
                    "ON CONFLICT(tenant_id, migration_name, assertion_id) DO NOTHING",
                    (tenant_id, MIGRATION_NAME, assertion_id, recorded_at),
                )

    @staticmethod
    async def _finalize_rollback_group(
        db: Any,
        tenant_id: str,
        assertion_id: str,
        group: list[tuple[object, object, object]],
    ) -> None:
        """Atomically queue withdrawal projection work and close all receipts."""
        async with db.transaction():
            await db.execute(
                "UPDATE legacy_fact_migration_invalidations "
                "SET state = 'pending', generation = generation + 1, delivered_at = NULL "
                "WHERE tenant_id = ? AND migration_name = ? AND assertion_id = ?",
                (tenant_id, MIGRATION_NAME, assertion_id),
            )
            for node_id, _source_id, _revision_id in group:
                await db.execute(
                    "UPDATE legacy_fact_migration_records SET outcome = 'rolled_back' "
                    "WHERE tenant_id = ? AND node_id = ?",
                    (tenant_id, node_id),
                )

    async def _deliver_pending_invalidations(self, db: Any, tenant_id: str) -> bool:
        """Drain bounded projection pages; only acknowledged pages are marked done."""
        if self._index_invalidator is None:
            return False
        delivered = False
        for _page in range(_MAX_INVALIDATION_PAGES):
            rows = await db.fetchall(
                "SELECT assertion_id, generation FROM legacy_fact_migration_invalidations "
                "WHERE tenant_id = ? AND migration_name = ? AND state = 'pending' "
                f"ORDER BY assertion_id ASC LIMIT {_INVALIDATION_PAGE_SIZE}",
                (tenant_id, MIGRATION_NAME),
            )
            if not rows:
                return delivered
            deliveries = tuple((str(row[0]), int(row[1])) for row in rows)
            assertion_ids = tuple(assertion_id for assertion_id, _generation in deliveries)
            response = self._index_invalidator(tenant_id, assertion_ids)
            if hasattr(response, "__await__"):
                await response
            async with db.transaction():
                delivered_at = utc_timestamp_parameter(
                    db.backend_type, datetime.now(timezone.utc)
                )
                for assertion_id, generation in deliveries:
                    await db.execute(
                        "UPDATE legacy_fact_migration_invalidations "
                        "SET state = 'delivered', delivered_at = ? "
                        "WHERE tenant_id = ? AND migration_name = ? "
                        "AND assertion_id = ? AND state = 'pending' "
                        "AND generation = ?",
                        (
                            delivered_at,
                            tenant_id,
                            MIGRATION_NAME,
                            assertion_id,
                            generation,
                        ),
                    )
            delivered = True
        residual = await db.fetchone(
            "SELECT 1 FROM legacy_fact_migration_invalidations "
            "WHERE tenant_id = ? AND migration_name = ? AND state = 'pending' LIMIT 1",
            (tenant_id, MIGRATION_NAME),
        )
        if residual is not None:
            raise LegacyFactMigrationError(
                "projection_invalidation_delivery_budget_exhausted"
            )
        return delivered

    @staticmethod
    async def _set_outcome(db: Any, tenant_id: str, node_id: str, outcome: str) -> None:
        await db.execute(
            "UPDATE legacy_fact_migration_records SET outcome = ? WHERE tenant_id = ? AND node_id = ?",
            (outcome, tenant_id, node_id),
        )

    @staticmethod
    async def _checkpoint(db: Any, tenant_id: str, cursor: str | None, state: str) -> None:
        if db.backend_type == "postgres":
            sql = (
                "INSERT INTO legacy_fact_migration_checkpoints (tenant_id, migration_name, last_node_id, state, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (tenant_id, migration_name) DO UPDATE SET "
                "last_node_id = EXCLUDED.last_node_id, state = EXCLUDED.state, updated_at = EXCLUDED.updated_at"
            )
        else:
            sql = (
                "INSERT INTO legacy_fact_migration_checkpoints (tenant_id, migration_name, last_node_id, state, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(tenant_id, migration_name) DO UPDATE SET "
                "last_node_id = excluded.last_node_id, state = excluded.state, updated_at = excluded.updated_at"
            )
        await db.execute(
            sql,
            (
                tenant_id,
                MIGRATION_NAME,
                cursor,
                state,
                utc_timestamp_parameter(db.backend_type, datetime.now(timezone.utc)),
            ),
        )
