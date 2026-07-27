"""Tenant-bound persistence for canonical semantic assertions.

``AsyncAssertionStore`` deliberately owns normalized assertion state rather
than serialising facts into ``graph_nodes.properties``.  The property graph is
an optional projection/input boundary; it has no write path back into these
tables.  All public mutation methods require a store bound to one tenant, and
the tenant predicate is applied before every lookup or traversal.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Iterable, Sequence
from uuid import uuid4

from kestrel_sovereign.knowledge import (
    Assertion,
    AssertionQuery,
    AssertionStatus,
    DerivedLineage,
    EpistemicState,
    SourceOccurrence,
)
from kestrel_sovereign.knowledge.shacl_validation import (
    ShaclValidationReport,
    ValidationWriteAction,
)

from .async_database import AsyncDatabase
from .db.interface import TransactionError


_ASSERTION_STORE_FACTORY_TOKEN = object()
_ASSERTION_TENANT_CAPABILITY_TOKEN = object()
_ERASURE_JOB_TTL_SECONDS = 300.0
_MAX_ERASURE_JOBS = 256


@dataclass(frozen=True, slots=True, init=False)
class _AssertionTenantCapability:
    """Opaque authority issued when storage resolves an agent tenant.

    A caller that merely obtains a :class:`DatabaseBackend` must not be able to
    turn a chosen string into canonical-assertion authority.  The storage
    owner resolves the tenant once and carries this unforgeable-in-normal-use
    capability to the assertion-store factory.
    """

    tenant_id: str
    owning_agent_id: str

    def __init__(self, token: object, tenant_id: str, owning_agent_id: str) -> None:
        if token is not _ASSERTION_TENANT_CAPABILITY_TOKEN:
            raise TypeError("assertion tenant capabilities are issued by AsyncStorage")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise AssertionStoreError("Assertion tenant binding requires a non-empty tenant_id")
        if not isinstance(owning_agent_id, str) or not owning_agent_id:
            raise AssertionStoreError("Assertion tenant binding requires a non-empty owning_agent_id")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "owning_agent_id", owning_agent_id)


def _issue_assertion_tenant_capability(
    agent_id: str,
) -> _AssertionTenantCapability:
    """Issue the private capability after a storage owner has resolved a DID."""
    return _AssertionTenantCapability(
        _ASSERTION_TENANT_CAPABILITY_TOKEN,
        agent_id,
        agent_id,
    )


@dataclass(frozen=True, slots=True, init=False)
class _AssertionStoreScope:
    """Immutable storage-issued tenant authority for one assertion store.

    This object is deliberately not a public construction surface.  The
    storage facade creates it only after it has established its own agent
    scope; raw database access alone cannot select an assertion tenant.
    """

    database: AsyncDatabase
    tenant_id: str
    owning_agent_id: str

    def __init__(
        self,
        token: object,
        database: AsyncDatabase,
        capability: _AssertionTenantCapability,
    ) -> None:
        if token is not _ASSERTION_STORE_FACTORY_TOKEN:
            raise TypeError("assertion store scopes are issued by AsyncStorage")
        if type(capability) is not _AssertionTenantCapability:
            raise TypeError("assertion store scopes require a storage-issued tenant capability")
        object.__setattr__(self, "database", database)
        object.__setattr__(self, "tenant_id", capability.tenant_id)
        object.__setattr__(self, "owning_agent_id", capability.owning_agent_id)

    @classmethod
    def _issue(
        cls,
        token: object,
        database: AsyncDatabase,
        capability: _AssertionTenantCapability,
    ) -> "_AssertionStoreScope":
        return cls(token, database, capability)


class AssertionStoreError(ValueError):
    """A canonical assertion mutation cannot be accepted."""


class TenantIsolationError(AssertionStoreError):
    """A caller attempted to cross the store's authoritative tenant scope."""


class AssertionConflictError(AssertionStoreError):
    """A lifecycle compare-and-swap or idempotency check did not match."""


@dataclass(frozen=True, slots=True)
class AssertionWriteResult:
    assertion: Assertion
    generation: int
    event_id: str
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class SupersessionResult:
    predecessor: Assertion
    replacement: Assertion
    generation: int
    event_ids: tuple[str, ...]
    invalidated_revision_ids: tuple[str, ...]
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class RetractionResult:
    retracted: tuple[Assertion, ...]
    invalidated_revision_ids: tuple[str, ...]
    generation: int
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class DeletionResult:
    deleted: Assertion
    invalidated: tuple[Assertion, ...]
    invalidated_revision_ids: tuple[str, ...]
    generation: int
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class ValidationQuarantineResult:
    """A validation-driven lifecycle transition and its derived invalidations."""

    quarantined: Assertion
    invalidated: tuple[Assertion, ...]
    invalidated_revision_ids: tuple[str, ...]
    generation: int
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class ErasureResult:
    erased_assertion_ids: tuple[str, ...]
    erased_revision_ids: tuple[str, ...]
    generation: int
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class AssertionCheckpoint:
    tenant_id: str
    generation: int
    latest_event_id: str | None


@dataclass(frozen=True, slots=True)
class AssertionChange:
    event_id: str
    assertion_id: str | None
    revision_id: str | None
    operation: str
    generation: int
    eligible: bool


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _operation_digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _erasure_receipt_key(operation_id: str) -> str:
    """Derive the durable, opaque lookup key for an erasure retry."""
    return _operation_digest({"erasure_operation_id": operation_id})


def _placeholders(values: Sequence[object]) -> str:
    if not values:
        raise AssertionStoreError("an empty SQL membership set is not permitted")
    return ", ".join("?" for _ in values)


def _assertion_with(
    assertion: Assertion,
    *,
    revision_id: str,
    status: AssertionStatus,
    supersedes_revision_id: str | None,
    epistemic_state: EpistemicState | None = None,
) -> Assertion:
    mapping = assertion.to_mapping()
    mapping["revision_id"] = revision_id
    mapping["status"] = status.value
    mapping["supersedes_revision_id"] = supersedes_revision_id
    mapping["asserted_at"] = {"schema_version": 1, "value": _now()}
    if epistemic_state is not None:
        mapping["epistemic_state"] = epistemic_state.value
    return Assertion.from_mapping(mapping)


class AsyncAssertionStore:
    """One canonical, normalized assertion authority for a single tenant.

    The constructor accepts only a storage-issued scope, never a raw database
    plus caller-selected tenant strings.  A bound store has no per-call tenant
    override; code needing another tenant must use that agent's authenticated
    :class:`AsyncStorage` facade.
    """

    __slots__ = ("__scope", "_erasure_jobs")

    def __init__(self, scope: _AssertionStoreScope, /) -> None:
        if type(scope) is not _AssertionStoreScope:
            raise TypeError(
                "AsyncAssertionStore is created only by an agent-bound AsyncStorage facade"
            )
        self.__scope = scope
        # Physical-erasure targets are intentionally process-local.  A retry
        # in this short window can receive the original deterministic result,
        # while a durable receipt after restart proves only that the erasure
        # completed and where incremental consumers must resume.
        self._erasure_jobs: dict[str, tuple[float, str, ErasureResult]] = {}

    @property
    def tenant_id(self) -> str:
        """The immutable tenant bound by the owning storage facade."""
        return self.__scope.tenant_id

    @property
    def owning_agent_id(self) -> str:
        """The immutable owner bound by the owning storage facade."""
        return self.__scope.owning_agent_id

    @property
    def _database(self) -> AsyncDatabase:
        """Internal database handle; deliberately not a public store surface."""
        return self.__scope.database

    def _require_scope(self) -> tuple[str, str]:
        return self.tenant_id, self.owning_agent_id

    def _check_assertion_scope(self, assertion: Assertion) -> None:
        tenant_id, owner = self._require_scope()
        if assertion.tenant_id != tenant_id:
            raise TenantIsolationError("Assertion tenant_id does not match the bound tenant")
        if assertion.owning_agent_id != owner:
            raise TenantIsolationError("Assertion owning_agent_id does not match the bound owner")

    async def _generation(self) -> int:
        """Read generation without creating durable tenant state."""
        tenant_id, _ = self._require_scope()
        row = await self._database.fetchone(
            "SELECT generation FROM semantic_assertion_tenants WHERE tenant_id = ?",
            (tenant_id,),
        )
        return int(row[0]) if row is not None else 0

    async def _ensure_generation(self) -> int:
        """Create the generation row only inside a mutation transaction."""
        tenant_id, _ = self._require_scope()
        row = await self._database.fetchone(
            "SELECT generation FROM semantic_assertion_tenants WHERE tenant_id = ?",
            (tenant_id,),
        )
        if row is None:
            insert = (
                "INSERT INTO semantic_assertion_tenants (tenant_id, generation, updated_at) "
                "VALUES (?, 0, ?) ON CONFLICT DO NOTHING"
                if self._database.backend_type == "postgres"
                else "INSERT OR IGNORE INTO semantic_assertion_tenants "
                "(tenant_id, generation, updated_at) VALUES (?, 0, ?)"
            )
            await self._database.execute(
                insert,
                (tenant_id, _now()),
            )
            row = await self._database.fetchone(
                "SELECT generation FROM semantic_assertion_tenants WHERE tenant_id = ?",
                (tenant_id,),
            )
            if row is None:
                raise AssertionConflictError("tenant generation row could not be initialized")
        return int(row[0])

    async def _lock_tenant(self) -> None:
        """Serialize one tenant's pointer/generation mutation on both backends."""
        tenant_id, _ = self._require_scope()
        await self._ensure_generation()
        if self._database.backend_type == "postgres":
            await self._database.fetchone(
                "SELECT generation FROM semantic_assertion_tenants WHERE tenant_id = ? FOR UPDATE",
                (tenant_id,),
            )

    async def _advance_generation(self) -> int:
        tenant_id, _ = self._require_scope()
        generation = await self._generation() + 1
        await self._database.execute(
            "UPDATE semantic_assertion_tenants SET generation = ?, updated_at = ? WHERE tenant_id = ?",
            (generation, _now(), tenant_id),
        )
        return generation

    @asynccontextmanager
    async def _mutation(self):
        """Run one canonical mutation with a tenant serialization boundary.

        Database backends deliberately wrap any exception raised inside a
        transaction in ``TransactionError``.  Assertion contract violations
        are expected, caller-actionable rejections, so preserve those domain
        errors after the backend has rolled back the complete mutation.
        """
        try:
            async with self._database.transaction():
                await self._lock_tenant()
                yield
        except TransactionError as error:
            if isinstance(error.__cause__, AssertionStoreError):
                raise error.__cause__ from error
            raise

    async def _current(self, assertion_id: str) -> Assertion | None:
        tenant_id, _ = self._require_scope()
        row = await self._database.fetchone(
            "SELECT r.assertion_mapping FROM semantic_assertions a "
            "JOIN semantic_assertion_revisions r ON r.revision_id = a.current_revision_id "
            "WHERE a.tenant_id = ? AND a.assertion_id = ?",
            (tenant_id, assertion_id),
        )
        return Assertion.from_mapping(json.loads(row[0])) if row else None

    async def _revision(self, revision_id: str) -> Assertion | None:
        tenant_id, _ = self._require_scope()
        row = await self._database.fetchone(
            "SELECT assertion_mapping FROM semantic_assertion_revisions "
            "WHERE tenant_id = ? AND revision_id = ?",
            (tenant_id, revision_id),
        )
        return Assertion.from_mapping(json.loads(row[0])) if row else None

    async def get_assertion(self, assertion_id: str, *, include_inactive: bool = False) -> Assertion | None:
        assertion = await self._current(assertion_id)
        if assertion is None:
            return None
        if not include_inactive and assertion.status is not AssertionStatus.ACTIVE:
            return None
        return assertion

    async def get_revision(self, revision_id: str) -> Assertion | None:
        return await self._revision(revision_id)

    async def list_revisions(self, assertion_id: str) -> list[Assertion]:
        tenant_id, _ = self._require_scope()
        rows = await self._database.fetchall(
            "SELECT assertion_mapping FROM semantic_assertion_revisions "
            "WHERE tenant_id = ? AND assertion_id = ? ORDER BY accepted_order ASC",
            (tenant_id, assertion_id),
        )
        return [Assertion.from_mapping(json.loads(row[0])) for row in rows]

    async def list_source_occurrences(self, assertion_id: str) -> list[SourceOccurrence]:
        tenant_id, _ = self._require_scope()
        rows = await self._database.fetchall(
            "SELECT DISTINCT s.source_mapping FROM semantic_assertion_revisions r "
            "JOIN semantic_revision_sources rs ON rs.tenant_id = r.tenant_id AND rs.revision_id = r.revision_id "
            "JOIN semantic_source_occurrences s ON s.tenant_id = rs.tenant_id "
            "  AND s.source_occurrence_id = rs.source_occurrence_id "
            "WHERE r.tenant_id = ? AND r.assertion_id = ? ORDER BY s.received_at, s.source_occurrence_id",
            (tenant_id, assertion_id),
        )
        return [SourceOccurrence.from_mapping(json.loads(row[0])) for row in rows]

    async def get_source_occurrence(self, source_occurrence_id: str) -> SourceOccurrence | None:
        """Read one immutable provenance record in the bound tenant only."""
        tenant_id, _ = self._require_scope()
        row = await self._database.fetchone(
            "SELECT source_mapping FROM semantic_source_occurrences "
            "WHERE tenant_id = ? AND source_occurrence_id = ?",
            (tenant_id, source_occurrence_id),
        )
        return SourceOccurrence.from_mapping(json.loads(row[0])) if row else None

    async def derivation_inputs(self, revision_id: str) -> list[Assertion]:
        tenant_id, _ = self._require_scope()
        rows = await self._database.fetchall(
            "SELECT r.assertion_mapping FROM semantic_derivation_inputs d "
            "JOIN semantic_assertion_revisions r ON r.tenant_id = d.tenant_id "
            "  AND r.revision_id = d.input_revision_id "
            "WHERE d.tenant_id = ? AND d.derived_revision_id = ? ORDER BY d.ordinal ASC",
            (tenant_id, revision_id),
        )
        return [Assertion.from_mapping(json.loads(row[0])) for row in rows]

    async def _source_exists(self, source_id: str) -> bool:
        tenant_id, _ = self._require_scope()
        return bool(await self._database.fetchone(
            "SELECT 1 FROM semantic_source_occurrences WHERE tenant_id = ? AND source_occurrence_id = ?",
            (tenant_id, source_id),
        ))

    async def _store_sources(self, sources: Sequence[SourceOccurrence]) -> None:
        tenant_id, _ = self._require_scope()
        for source in sources:
            if not isinstance(source, SourceOccurrence):
                raise AssertionStoreError("source_occurrences must contain SourceOccurrence values")
            existing = await self._database.fetchone(
                "SELECT source_mapping FROM semantic_source_occurrences "
                "WHERE tenant_id = ? AND source_occurrence_id = ?",
                (tenant_id, source.source_occurrence_id),
            )
            encoded = _json(source.to_mapping())
            if existing is not None:
                if existing[0] != encoded:
                    raise AssertionConflictError("source occurrence id is already bound to different provenance")
                continue
            await self._database.execute(
                "INSERT INTO semantic_source_occurrences "
                "(tenant_id, source_occurrence_id, source_kind, locator, received_at, content_digest, actor, selector, source_mapping) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tenant_id, source.source_occurrence_id, source.source_kind,
                    source.locator, source.received_at.value, source.content_digest,
                    source.actor, source.selector, encoded,
                ),
            )

    async def _validate_lineage(self, assertion: Assertion, sources: Sequence[SourceOccurrence]) -> None:
        if isinstance(assertion.lineage, DerivedLineage):
            if sources:
                raise AssertionStoreError("derived assertions must not attach direct source occurrences")
            for revision_id in assertion.lineage.input_revision_ids:
                support = await self._revision(revision_id)
                if support is None:
                    raise TenantIsolationError("derived assertion input revision is absent from the bound tenant")
                if not await self._is_current_active_eligible_revision(revision_id):
                    raise AssertionStoreError(
                        "derived assertion inputs must be current, active, and eligible"
                    )
            return
        expected = set(assertion.lineage.source_occurrence_ids)
        supplied = {source.source_occurrence_id for source in sources}
        if supplied - expected:
            raise AssertionStoreError("a supplied source occurrence is absent from direct lineage")
        for source_id in expected:
            if source_id not in supplied and not await self._source_exists(source_id):
                raise AssertionStoreError("direct lineage references an unknown tenant-local source occurrence")

    async def _is_current_active_eligible_revision(self, revision_id: str) -> bool:
        """Return whether one bound-tenant revision is a valid derived support.

        Derived facts are valid only while every support is the current active
        revision and remains projection-eligible.  Looking up the revision alone
        would admit historical, superseded, retracted, or invalidated facts and
        make a new derived assertion appear valid after its support was removed.
        """
        tenant_id, _ = self._require_scope()
        row = await self._database.fetchone(
            "SELECT r.status, r.eligible, e.eligible FROM semantic_assertions a "
            "JOIN semantic_assertion_revisions r ON r.tenant_id = a.tenant_id "
            "  AND r.revision_id = a.current_revision_id "
            "LEFT JOIN semantic_projection_eligibility e ON e.tenant_id = r.tenant_id "
            "  AND e.revision_id = r.revision_id "
            "WHERE a.tenant_id = ? AND r.revision_id = ?",
            (tenant_id, revision_id),
        )
        return bool(
            row is not None
            and row[0] == AssertionStatus.ACTIVE.value
            and bool(row[1])
            and bool(row[2])
        )

    @staticmethod
    def _flat_terms(assertion: Assertion) -> tuple[str, str, str, str, str | None, str | None]:
        object_mapping = assertion.object.identity_mapping()
        return (
            assertion.subject.value,
            assertion.predicate.value,
            str(object_mapping["kind"]),
            str(object_mapping["value"]),
            object_mapping.get("datatype") if isinstance(object_mapping.get("datatype"), str) else None,
            object_mapping.get("language") if isinstance(object_mapping.get("language"), str) else None,
        )

    async def _write_revision(self, assertion: Assertion, sources: Sequence[SourceOccurrence]) -> str:
        tenant_id, _ = self._require_scope()
        encoded = _json(assertion.to_mapping())
        prior = await self._database.fetchone(
            "SELECT assertion_mapping FROM semantic_assertion_revisions WHERE revision_id = ?",
            (assertion.revision_id,),
        )
        if prior is not None:
            if prior[0] != encoded:
                raise AssertionConflictError("revision_id is immutable and already stores a different assertion")
            return ""
        subject, predicate, object_kind, object_value, object_datatype, object_language = self._flat_terms(assertion)
        accepted_order = await self._database.fetchval(
            "SELECT COALESCE(MAX(accepted_order), 0) + 1 FROM semantic_assertion_revisions WHERE tenant_id = ?",
            (tenant_id,),
        )
        observed_start = assertion.observed_time.start.value if assertion.observed_time and assertion.observed_time.start else None
        observed_end = assertion.observed_time.end.value if assertion.observed_time and assertion.observed_time.end else None
        valid_start = assertion.valid_time.start.value if assertion.valid_time and assertion.valid_time.start else None
        valid_end = assertion.valid_time.end.value if assertion.valid_time and assertion.valid_time.end else None
        await self._database.execute(
            "INSERT INTO semantic_assertion_revisions "
            "(revision_id, tenant_id, assertion_id, owning_agent_id, status, epistemic_state, "
            "subject_value, predicate_value, object_kind, object_value, object_datatype, object_language, "
            "asserted_at, observed_start, observed_end, valid_start, valid_end, supersedes_revision_id, "
            "lineage_kind, eligible, accepted_order, assertion_mapping) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                assertion.revision_id, tenant_id, assertion.assertion_id, assertion.owning_agent_id,
                assertion.status.value, assertion.epistemic_state.value, subject, predicate,
                object_kind, object_value, object_datatype, object_language, assertion.asserted_at.value,
                observed_start, observed_end, valid_start, valid_end, assertion.supersedes_revision_id,
                assertion.lineage.kind, 1 if assertion.status is AssertionStatus.ACTIVE else 0,
                int(accepted_order), encoded,
            ),
        )
        if isinstance(assertion.lineage, DerivedLineage):
            for ordinal, input_revision_id in enumerate(assertion.lineage.input_revision_ids):
                await self._database.execute(
                    "INSERT INTO semantic_derivation_inputs "
                    "(tenant_id, derived_revision_id, input_revision_id, ordinal) VALUES (?, ?, ?, ?)",
                    (tenant_id, assertion.revision_id, input_revision_id, ordinal),
                )
        else:
            for ordinal, source_id in enumerate(assertion.lineage.source_occurrence_ids):
                await self._database.execute(
                    "INSERT INTO semantic_revision_sources "
                    "(tenant_id, revision_id, source_occurrence_id, ordinal) VALUES (?, ?, ?, ?)",
                    (tenant_id, assertion.revision_id, source_id, ordinal),
                )
        await self._database.execute(
            "INSERT INTO semantic_projection_eligibility (tenant_id, revision_id, eligible, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (tenant_id, assertion.revision_id, 1 if assertion.status is AssertionStatus.ACTIVE else 0, _now()),
        )
        return assertion.revision_id

    async def _set_current(self, assertion: Assertion) -> None:
        tenant_id, _ = self._require_scope()
        existing = await self._database.fetchone(
            "SELECT owning_agent_id FROM semantic_assertions WHERE tenant_id = ? AND assertion_id = ?",
            (tenant_id, assertion.assertion_id),
        )
        if existing is not None and existing[0] != assertion.owning_agent_id:
            raise TenantIsolationError("an assertion identity cannot be claimed by a second owner")
        if existing is None:
            await self._database.execute(
                "INSERT INTO semantic_assertions (tenant_id, assertion_id, owning_agent_id, current_revision_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tenant_id, assertion.assertion_id, assertion.owning_agent_id, assertion.revision_id, _now()),
            )
        else:
            await self._database.execute(
                "UPDATE semantic_assertions SET current_revision_id = ? WHERE tenant_id = ? AND assertion_id = ?",
                (assertion.revision_id, tenant_id, assertion.assertion_id),
            )

    async def _invalidate_eligibility(self, revision_id: str) -> None:
        """Make a historical revision unusable by every index/training reader.

        The separate eligibility row is the projection-facing tombstone.  The
        revision column mirrors it so a projection rebuilding directly from
        canonical rows cannot accidentally revive an old active revision after
        a lifecycle transition or restart.
        """
        tenant_id, _ = self._require_scope()
        await self._database.execute(
            "UPDATE semantic_assertion_revisions SET eligible = 0 WHERE tenant_id = ? AND revision_id = ?",
            (tenant_id, revision_id),
        )
        await self._database.execute(
            "UPDATE semantic_projection_eligibility SET eligible = 0, updated_at = ? "
            "WHERE tenant_id = ? AND revision_id = ?",
            (_now(), tenant_id, revision_id),
        )

    async def _event(self, assertion: Assertion, operation: str, generation: int) -> str:
        tenant_id, _ = self._require_scope()
        event_id = uuid4().hex
        eligible = assertion.status is AssertionStatus.ACTIVE
        await self._database.execute(
            "INSERT INTO semantic_projection_outbox "
            "(event_id, tenant_id, assertion_id, revision_id, operation, generation, eligible, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, tenant_id, assertion.assertion_id, assertion.revision_id,
             operation, generation, 1 if eligible else 0, _now()),
        )
        return event_id

    async def _erasure_event(self, generation: int) -> str:
        """Record an opaque resynchronization signal after physical erasure.

        Physical erasure removes every ordinary outbox row naming erased
        revisions. Consumers that already projected those rows still need a
        durable signal to discard their tenant-scoped projection. This stream
        retains only an opaque event ID and generation, so cleanup can be
        retried by rescanning the tenant without retaining erased identities.
        """
        tenant_id, _ = self._require_scope()
        event_id = uuid4().hex
        await self._database.execute(
            "INSERT INTO semantic_projection_erasure_outbox "
            "(event_id, tenant_id, operation, generation, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, tenant_id, "erased", generation, _now()),
        )
        return event_id

    async def _operation(self, operation_id: str, operation: str, request: object):
        tenant_id, _ = self._require_scope()
        if not isinstance(operation_id, str) or not operation_id:
            raise AssertionStoreError("operation_id must be a non-empty string")
        digest = _operation_digest(request)
        row = await self._database.fetchone(
            "SELECT operation, request_digest, receipt FROM semantic_assertion_operations "
            "WHERE tenant_id = ? AND operation_id = ?",
            (tenant_id, operation_id),
        )
        if row is None:
            erasure = await self._database.fetchone(
                "SELECT 1 FROM semantic_assertion_erasure_receipts "
                "WHERE tenant_id = ? AND operation_id = ?",
                (tenant_id, _erasure_receipt_key(operation_id)),
            )
            if erasure is not None:
                raise AssertionConflictError(
                    "operation_id was already used for an erasure mutation"
                )
            return digest, None
        if row[0] != operation or row[1] != digest:
            raise AssertionConflictError("operation_id was already used for a different semantic mutation")
        return digest, json.loads(row[2])

    async def _record_operation(
        self, operation_id: str, operation: str, digest: str, receipt: dict[str, object], assertion_ids: Iterable[str], revision_ids: Iterable[str],
    ) -> None:
        tenant_id, _ = self._require_scope()
        await self._database.execute(
            "INSERT INTO semantic_assertion_operations "
            "(tenant_id, operation_id, operation, request_digest, receipt, assertion_ids, revision_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (tenant_id, operation_id, operation, digest, _json(receipt), _json(sorted(set(assertion_ids))),
             _json(sorted(set(revision_ids))), _now()),
        )

    async def _delete_operations_referencing(
        self,
        assertion_ids: set[str],
        revision_ids: set[str],
    ) -> None:
        """Remove idempotency receipts that retain any physically erased ID.

        The identifiers are stored as JSON arrays for backend-neutral audit
        data. Parsing them avoids SQL ``LIKE`` semantics, which would treat
        ``%`` and ``_`` in a caller-provided ID as wildcards and could leave a
        recoverable receipt behind.
        """
        tenant_id, _ = self._require_scope()
        rows = await self._database.fetchall(
            "SELECT operation_id, assertion_ids, revision_ids "
            "FROM semantic_assertion_operations WHERE tenant_id = ?",
            (tenant_id,),
        )
        operation_ids: list[str] = []
        for operation_id, encoded_assertion_ids, encoded_revision_ids in rows:
            try:
                recorded_assertion_ids = json.loads(encoded_assertion_ids)
                recorded_revision_ids = json.loads(encoded_revision_ids)
            except (TypeError, json.JSONDecodeError) as error:
                raise AssertionStoreError(
                    "semantic operation receipt contains malformed identifier data"
                ) from error
            if not (
                isinstance(recorded_assertion_ids, list)
                and all(isinstance(item, str) for item in recorded_assertion_ids)
                and isinstance(recorded_revision_ids, list)
                and all(isinstance(item, str) for item in recorded_revision_ids)
            ):
                raise AssertionStoreError(
                    "semantic operation receipt contains malformed identifier data"
                )
            if (
                assertion_ids.intersection(recorded_assertion_ids)
                or revision_ids.intersection(recorded_revision_ids)
            ):
                operation_ids.append(operation_id)
        if operation_ids:
            await self._database.execute(
                "DELETE FROM semantic_assertion_operations WHERE tenant_id = ? "
                f"AND operation_id IN ({_placeholders(operation_ids)})",
                (tenant_id,) + tuple(operation_ids),
            )

    async def _erasure_operation(self, operation_id: str, request: object):
        """Resolve an idempotent physical-erasure receipt within this tenant.

        Erasure receipts have their own ledger because ordinary operation rows
        name assertion and revision IDs and must disappear with the erased
        closure.  The receipt is accessible only through this already-bound
        store and is never joined into assertion/inference/query paths.
        """
        tenant_id, _ = self._require_scope()
        if not isinstance(operation_id, str) or not operation_id:
            raise AssertionStoreError("operation_id must be a non-empty string")
        digest = _operation_digest(request)
        normal = await self._database.fetchone(
            "SELECT 1 FROM semantic_assertion_operations WHERE tenant_id = ? AND operation_id = ?",
            (tenant_id, operation_id),
        )
        if normal is not None:
            raise AssertionConflictError("operation_id was already used for a different semantic mutation")
        row = await self._database.fetchone(
            "SELECT request_digest, generation FROM semantic_assertion_erasure_receipts "
            "WHERE tenant_id = ? AND operation_id = ?",
            (tenant_id, _erasure_receipt_key(operation_id)),
        )
        if row is None:
            return digest, None
        if row[0] != digest:
            raise AssertionConflictError("operation_id was already used for a different erasure mutation")
        return digest, int(row[1])

    async def _record_erasure_operation(
        self,
        operation_id: str,
        digest: str,
        generation: int,
    ) -> None:
        """Persist the tenant-scoped retry receipt after physical erasure.

        The receipt deliberately contains neither assertion/revision columns
        nor a JSON payload.  It is only an opaque idempotency key and a
        generation checkpoint; sensitive target identities live briefly in
        ``_erasure_jobs`` and disappear when that protected in-memory map
        expires or the process restarts.
        """
        tenant_id, _ = self._require_scope()
        await self._database.execute(
            "INSERT INTO semantic_assertion_erasure_receipts "
            "(tenant_id, operation_id, request_digest, generation, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (tenant_id, _erasure_receipt_key(operation_id), digest, generation, _now()),
        )

    def _remember_erasure_job(
        self,
        operation_id: str,
        digest: str,
        result: ErasureResult,
    ) -> None:
        """Keep one immediate-retry result without retaining it durably."""
        now = time.monotonic()
        expired = [
            job_id for job_id, (expires_at, _, _) in self._erasure_jobs.items()
            if expires_at <= now
        ]
        for job_id in expired:
            del self._erasure_jobs[job_id]
        while len(self._erasure_jobs) >= _MAX_ERASURE_JOBS:
            oldest = next(iter(self._erasure_jobs))
            del self._erasure_jobs[oldest]
        self._erasure_jobs[operation_id] = (
            now + _ERASURE_JOB_TTL_SECONDS,
            digest,
            result,
        )

    def _remembered_erasure_job(
        self,
        operation_id: str,
        digest: str,
    ) -> ErasureResult | None:
        remembered = self._erasure_jobs.get(operation_id)
        if remembered is None:
            return None
        expires_at, remembered_digest, result = remembered
        if expires_at <= time.monotonic():
            del self._erasure_jobs[operation_id]
            return None
        if remembered_digest != digest:
            return None
        return ErasureResult(
            result.erased_assertion_ids,
            result.erased_revision_ids,
            result.generation,
            True,
        )

    async def put_assertion(
        self,
        assertion: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> AssertionWriteResult:
        """Persist one initial assertion revision atomically and idempotently."""
        operation_id, request = self._validate_initial_assertion_write(
            assertion,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
            expected_generation=expected_generation,
        )
        async with self._mutation():
            written, _ = await self._put_initial_assertion_in_mutation(
                assertion,
                source_occurrences=source_occurrences,
                operation_id=operation_id,
                request=request,
                expected_generation=expected_generation,
            )
            return written

    async def put_assertion_with_validation_report(
        self,
        assertion: Assertion,
        report: ShaclValidationReport,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> AssertionWriteResult:
        """Atomically accept one assertion and its required SHACL report.

        A governed write cannot expose an accepted canonical revision until its
        report, normalized report links/results, generation, and outbox receipt
        can commit in the same assertion-authority transaction.  Rejected
        pre-publication reports use :meth:`persist_validation_report` instead,
        because there is no canonical assertion to commit with them.
        """
        operation_id, request = self._validate_initial_assertion_write(
            assertion,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
            expected_generation=expected_generation,
        )
        self._validate_accepted_validation_report(assertion, report)
        async with self._mutation():
            written, replay = await self._put_initial_assertion_in_mutation(
                assertion,
                source_occurrences=source_occurrences,
                operation_id=operation_id,
                request=request,
                expected_generation=expected_generation,
                validation_report_id=report.report_id,
            )
            if replay is None:
                await self._persist_validation_report_in_transaction(report)
            else:
                await self._validation_report_from_receipt(replay)
            return written

    def _validate_initial_assertion_write(
        self,
        assertion: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence],
        operation_id: str | None,
        expected_generation: int | None,
    ) -> tuple[str, dict[str, object]]:
        if not isinstance(assertion, Assertion):
            raise AssertionStoreError("put_assertion requires a canonical Assertion")
        self._check_assertion_scope(assertion)
        if assertion.status is not AssertionStatus.ACTIVE or assertion.supersedes_revision_id is not None:
            raise AssertionStoreError(
                "put_assertion only accepts an initial active revision with no superseded predecessor"
            )
        if expected_generation is not None and (
            type(expected_generation) is not int or expected_generation < 0
        ):
            raise AssertionStoreError("expected_generation must be a non-negative integer")
        resolved_operation_id = operation_id or f"revision:{assertion.revision_id}"
        request = {"assertion": assertion.to_mapping(), "sources": [s.to_mapping() for s in source_occurrences]}
        return resolved_operation_id, request

    async def _put_initial_assertion_in_mutation(
        self,
        assertion: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence],
        operation_id: str,
        request: dict[str, object],
        expected_generation: int | None,
        validation_report_id: str | None = None,
    ) -> tuple[AssertionWriteResult, dict[str, object] | None]:
        """Write an already-validated initial revision inside ``_mutation``."""
        digest, replay = await self._operation(operation_id, "put", request)
        if replay is not None:
            replayed = await self._revision(str(replay["revision_id"]))
            if replayed is None:
                raise AssertionConflictError("idempotent assertion receipt no longer has a revision")
            return (
                AssertionWriteResult(replayed, int(replay["generation"]), str(replay["event_id"]), True),
                replay,
            )
        if expected_generation is not None and await self._generation() != expected_generation:
            raise AssertionConflictError("validation generation is no longer current; revalidate before writing")
        existing_revision = await self._revision(assertion.revision_id)
        if existing_revision is not None:
            if existing_revision != assertion:
                raise AssertionConflictError("revision_id is immutable and already stores a different assertion")
            checkpoint = await self.checkpoint()
            return AssertionWriteResult(assertion, checkpoint.generation, "", True), None
        current = await self._current(assertion.assertion_id)
        if current is not None:
            raise AssertionConflictError("append through supersede/retract; put_assertion only creates an initial revision")
        await self._store_sources(source_occurrences)
        await self._validate_lineage(assertion, source_occurrences)
        await self._write_revision(assertion, source_occurrences)
        await self._set_current(assertion)
        generation = await self._advance_generation()
        event_id = await self._event(assertion, "accepted", generation)
        receipt = {
            "revision_id": assertion.revision_id,
            "generation": generation,
            "event_id": event_id,
        }
        if validation_report_id is not None:
            receipt["validation_report_id"] = validation_report_id
        await self._record_operation(
            operation_id, "put", digest,
            receipt,
            [assertion.assertion_id], [assertion.revision_id],
        )
        return AssertionWriteResult(assertion, generation, event_id), None

    async def replay_governed_initial_write(
        self,
        assertion: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
    ) -> tuple[ShaclValidationReport, AssertionWriteResult] | None:
        """Return the original report and receipt for one governed write retry.

        This lookup deliberately happens before a caller snapshots or rejects
        an already-current assertion.  The operation ledger, report, and
        revision commit in one transaction, so a matching receipt is the sole
        authoritative idempotency result for the governed boundary.
        """
        resolved_operation_id, request = self._validate_initial_assertion_write(
            assertion,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
            expected_generation=None,
        )
        _, replay = await self._operation(resolved_operation_id, "put", request)
        if replay is None:
            return None
        report = await self._validation_report_from_receipt(replay)
        replayed = await self._revision(str(replay["revision_id"]))
        if replayed is None:
            raise AssertionConflictError("idempotent assertion receipt no longer has a revision")
        self._validate_accepted_validation_report(replayed, report)
        return report, AssertionWriteResult(
            replayed,
            int(replay["generation"]),
            str(replay["event_id"]),
            True,
        )

    def _validate_validation_report(self, report: ShaclValidationReport) -> None:
        if not isinstance(report, ShaclValidationReport):
            raise AssertionStoreError("validation report must be a ShaclValidationReport")
        if report.tenant_id != self.tenant_id:
            raise TenantIsolationError("validation report tenant does not match the bound assertion tenant")

    def _validate_accepted_validation_report(
        self,
        assertion: Assertion,
        report: ShaclValidationReport,
    ) -> None:
        self._validate_validation_report(report)
        if report.action not in (
            ValidationWriteAction.ACCEPT,
            ValidationWriteAction.ACCEPT_WITH_REPORT,
        ):
            raise AssertionStoreError("only accepted validation reports can accompany a canonical assertion write")
        if report.assertion_ids != (assertion.assertion_id,):
            raise AssertionStoreError(
                "accepted validation report must identify exactly the assertion committed with it"
            )

    async def _validation_report_from_receipt(
        self,
        receipt: dict[str, object],
    ) -> ShaclValidationReport:
        """Load the report atomically paired with a governed operation receipt."""
        report_id = receipt.get("validation_report_id")
        if not isinstance(report_id, str) or not report_id:
            raise AssertionConflictError(
                "operation_id was already used for an ungoverned semantic mutation"
            )
        row = await self._database.fetchone(
            "SELECT report_mapping FROM semantic_validation_reports "
            "WHERE tenant_id = ? AND report_id = ?",
            (self.tenant_id, report_id),
        )
        if row is None:
            raise AssertionConflictError(
                "governed semantic operation receipt no longer has its validation report"
            )
        try:
            report = ShaclValidationReport.from_mapping(json.loads(row[0]))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise AssertionConflictError(
                "governed semantic operation receipt has a malformed validation report"
            ) from error
        self._validate_validation_report(report)
        return report

    async def persist_validation_report(self, report: ShaclValidationReport) -> ShaclValidationReport:
        """Persist a tenant-bound report when no canonical write is pending."""
        self._validate_validation_report(report)
        async with self._database.transaction():
            await self._persist_validation_report_in_transaction(report)
        return report

    async def _persist_validation_report_in_transaction(
        self,
        report: ShaclValidationReport,
    ) -> None:
        """Write report rows in the caller's existing canonical transaction."""
        encoded = _json(report.to_mapping())
        existing = await self._database.fetchone(
            "SELECT report_mapping FROM semantic_validation_reports "
            "WHERE tenant_id = ? AND report_id = ?",
            (self.tenant_id, report.report_id),
        )
        if existing is not None:
            if existing[0] != encoded:
                raise AssertionStoreError(
                    "validation report id is immutable and already stores different data"
                )
            return
        await self._database.execute(
            "INSERT INTO semantic_validation_reports "
            "(report_id, tenant_id, report_version, assertion_ids, shape_set_id, shape_set_version, "
            "validation_profile_id, validation_profile_version, checkpoint_generation, run_id, "
            "state, action, source, evaluated_at, report_mapping) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report.report_id,
                self.tenant_id,
                report.report_version,
                _json(list(report.assertion_ids)),
                report.shape_set.identifier,
                report.shape_set.version,
                report.validation_profile.identifier,
                str(report.validation_profile.version),
                report.checkpoint_generation,
                report.run_id,
                report.state.value,
                report.action.value,
                report.source.value,
                report.evaluated_at,
                encoded,
            ),
        )
        report_assertion_ids = set(report.assertion_ids)
        for finding in report.findings:
            report_assertion_ids.update(finding.affected_assertion_ids)
        for assertion_id in sorted(report_assertion_ids):
            await self._database.execute(
                "INSERT INTO semantic_validation_report_assertions "
                "(tenant_id, report_id, assertion_id) VALUES (?, ?, ?)",
                (self.tenant_id, report.report_id, assertion_id),
            )
        for ordinal, finding in enumerate(report.findings):
            await self._database.execute(
                "INSERT INTO semantic_validation_results "
                "(tenant_id, report_id, ordinal, assertion_id, shape_id, constraint_code, severity) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.tenant_id,
                    report.report_id,
                    ordinal,
                    finding.focus_assertion_id,
                    finding.shape_id,
                    finding.code,
                    finding.severity.value,
                ),
            )

    async def supersede(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
    ) -> SupersessionResult:
        """Compare-and-swap an active revision with a superseded state and replacement."""
        operation_id, request = self._validate_supersession(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
            expected_generation=None,
        )
        async with self._mutation():
            result, _ = await self._supersede_in_mutation(
                expected_predecessor_revision_id,
                replacement,
                source_occurrences=source_occurrences,
                operation_id=operation_id,
                request=request,
                expected_generation=None,
            )
            return result

    async def supersede_with_validation_report(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        report: ShaclValidationReport,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> SupersessionResult:
        """Atomically supersede an assertion and persist its accepted SHACL report."""
        operation_id, request = self._validate_supersession(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
            expected_generation=expected_generation,
        )
        self._validate_accepted_validation_report(replacement, report)
        async with self._mutation():
            result, replay = await self._supersede_in_mutation(
                expected_predecessor_revision_id,
                replacement,
                source_occurrences=source_occurrences,
                operation_id=operation_id,
                request=request,
                expected_generation=expected_generation,
                validation_report_id=report.report_id,
            )
            if replay is None:
                await self._persist_validation_report_in_transaction(report)
            else:
                await self._validation_report_from_receipt(replay)
            return result

    def _validate_supersession(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence],
        operation_id: str | None,
        expected_generation: int | None,
    ) -> tuple[str, dict[str, object]]:
        if not isinstance(replacement, Assertion):
            raise AssertionStoreError("supersede requires a canonical Assertion replacement")
        self._check_assertion_scope(replacement)
        if replacement.status is not AssertionStatus.ACTIVE:
            raise AssertionStoreError("a supersession replacement must be active")
        if expected_generation is not None and (
            type(expected_generation) is not int or expected_generation < 0
        ):
            raise AssertionStoreError("expected_generation must be a non-negative integer")
        resolved_operation_id = operation_id or f"supersede:{expected_predecessor_revision_id}:{replacement.revision_id}"
        request = {
            "expected": expected_predecessor_revision_id,
            "replacement": replacement.to_mapping(),
            "sources": [s.to_mapping() for s in source_occurrences],
        }
        return resolved_operation_id, request

    async def replay_governed_supersession(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
    ) -> tuple[ShaclValidationReport, SupersessionResult] | None:
        """Return the original accepted report and receipt for a retry."""
        resolved_operation_id, request = self._validate_supersession(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
            expected_generation=None,
        )
        _, replay = await self._operation(resolved_operation_id, "supersede", request)
        if replay is None:
            return None
        report = await self._validation_report_from_receipt(replay)
        predecessor = await self._revision(str(replay["predecessor_revision_id"]))
        applied = await self._revision(str(replay["replacement_revision_id"]))
        if predecessor is None or applied is None:
            raise AssertionConflictError("idempotent supersession receipt no longer has its revisions")
        self._validate_accepted_validation_report(applied, report)
        return report, SupersessionResult(
            predecessor,
            applied,
            int(replay["generation"]),
            tuple(replay["event_ids"]),
            tuple(replay["invalidated_revision_ids"]),
            True,
        )

    async def _supersede_in_mutation(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence],
        operation_id: str,
        request: dict[str, object],
        expected_generation: int | None,
        validation_report_id: str | None = None,
    ) -> tuple[SupersessionResult, dict[str, object] | None]:
        """Apply a pre-validated supersession inside the tenant mutation."""
        digest, replay = await self._operation(operation_id, "supersede", request)
        if replay is not None:
            predecessor = await self._revision(str(replay["predecessor_revision_id"]))
            applied = await self._revision(str(replay["replacement_revision_id"]))
            if predecessor is None or applied is None:
                raise AssertionConflictError("idempotent supersession receipt no longer has its revisions")
            return (
                SupersessionResult(
                    predecessor, applied, int(replay["generation"]),
                    tuple(replay["event_ids"]), tuple(replay["invalidated_revision_ids"]), True,
                ),
                replay,
            )
        if expected_generation is not None and await self._generation() != expected_generation:
            raise AssertionConflictError("validation generation is no longer current; revalidate before superseding")
        predecessor = await self._revision(expected_predecessor_revision_id)
        if predecessor is None or predecessor.status is not AssertionStatus.ACTIVE:
            raise AssertionConflictError("expected predecessor is not an active tenant revision")
        current = await self._current(predecessor.assertion_id)
        if current is None or current.revision_id != expected_predecessor_revision_id:
            raise AssertionConflictError("expected predecessor is no longer the current revision")
        if replacement.supersedes_revision_id not in (None, expected_predecessor_revision_id):
            raise AssertionStoreError("replacement cannot name an unrelated superseded revision")
        await self._store_sources(source_occurrences)
        await self._validate_lineage(replacement, source_occurrences)
        predecessor_state = _assertion_with(
            predecessor, revision_id=uuid4().hex, status=AssertionStatus.SUPERSEDED,
            supersedes_revision_id=expected_predecessor_revision_id,
        )
        replacement_state = _assertion_with(
            replacement, revision_id=replacement.revision_id, status=AssertionStatus.ACTIVE,
            supersedes_revision_id=predecessor_state.revision_id,
        )
        # Every derived current revision that names the predecessor loses a
        # support. Retraction is transitive in this same transaction, so a
        # projection never observes a replacement alongside stale support.
        dependent_states: list[Assertion] = []
        invalidated_revision_ids = [expected_predecessor_revision_id]
        for dependent in await self._dependent_current_revisions([expected_predecessor_revision_id]):
            if dependent.status is not AssertionStatus.ACTIVE:
                continue
            await self._invalidate_eligibility(dependent.revision_id)
            invalidated_revision_ids.append(dependent.revision_id)
            state = _assertion_with(
                dependent, revision_id=uuid4().hex, status=AssertionStatus.RETRACTED,
                supersedes_revision_id=None, epistemic_state=EpistemicState.RETRACTED,
            )
            await self._write_revision(state, ())
            await self._set_current(state)
            dependent_states.append(state)
        await self._invalidate_eligibility(expected_predecessor_revision_id)
        await self._write_revision(predecessor_state, ())
        await self._set_current(predecessor_state)
        await self._write_revision(replacement_state, source_occurrences)
        await self._set_current(replacement_state)
        generation = await self._advance_generation()
        old_event = await self._event(predecessor_state, "superseded", generation)
        new_event = await self._event(replacement_state, "accepted", generation)
        dependent_events = [await self._event(item, "retracted", generation) for item in dependent_states]
        receipt = {
            "predecessor_revision_id": predecessor_state.revision_id,
            "replacement_revision_id": replacement_state.revision_id,
            "generation": generation,
            "event_ids": [old_event, new_event, *dependent_events],
            "invalidated_revision_ids": invalidated_revision_ids,
        }
        if validation_report_id is not None:
            receipt["validation_report_id"] = validation_report_id
        await self._record_operation(
            operation_id, "supersede", digest, receipt,
            [predecessor.assertion_id, replacement_state.assertion_id, *[item.assertion_id for item in dependent_states]],
            [predecessor_state.revision_id, replacement_state.revision_id, *[item.revision_id for item in dependent_states]],
        )
        return (
            SupersessionResult(
                predecessor_state, replacement_state, generation,
                (old_event, new_event, *dependent_events), tuple(invalidated_revision_ids),
            ),
            None,
        )

    async def _dependent_current_revisions(self, input_revision_ids: Iterable[str]) -> list[Assertion]:
        tenant_id, _ = self._require_scope()
        pending = set(input_revision_ids)
        discovered: dict[str, Assertion] = {}
        while pending:
            batch = tuple(sorted(pending))
            pending.clear()
            rows = await self._database.fetchall(
                "SELECT DISTINCT r.assertion_id, r.revision_id, r.assertion_mapping FROM semantic_derivation_inputs d "
                "JOIN semantic_assertions a ON a.tenant_id = d.tenant_id "
                "  AND a.current_revision_id = d.derived_revision_id "
                "JOIN semantic_assertion_revisions r ON r.tenant_id = a.tenant_id "
                "  AND r.revision_id = d.derived_revision_id "
                f"WHERE d.tenant_id = ? AND d.input_revision_id IN ({_placeholders(batch)}) "
                "ORDER BY r.assertion_id ASC, r.revision_id ASC",
                (tenant_id,) + batch,
            )
            for row in rows:
                assertion = Assertion.from_mapping(json.loads(row[2]))
                if assertion.revision_id not in discovered:
                    discovered[assertion.revision_id] = assertion
                    pending.add(assertion.revision_id)
        return list(discovered.values())

    async def retract(
        self,
        assertion_id: str,
        expected_revision_id: str,
        *,
        operation_id: str | None = None,
    ) -> RetractionResult:
        """Retract a current assertion and transitively invalidate derived current revisions."""
        operation_id = operation_id or f"retract:{assertion_id}:{expected_revision_id}"
        request = {"assertion_id": assertion_id, "expected_revision_id": expected_revision_id}
        async with self._mutation():
            digest, replay = await self._operation(operation_id, "retract", request)
            if replay is not None:
                revisions = [await self._revision(str(item)) for item in replay["revision_ids"]]
                if any(item is None for item in revisions):
                    raise AssertionConflictError("idempotent retraction receipt no longer has its revisions")
                return RetractionResult(tuple(item for item in revisions if item is not None), tuple(replay["invalidated_revision_ids"]), int(replay["generation"]), True)
            current = await self._current(assertion_id)
            if current is None or current.revision_id != expected_revision_id or current.status is not AssertionStatus.ACTIVE:
                raise AssertionConflictError("expected revision is not the active current tenant assertion")
            dependents = await self._dependent_current_revisions([current.revision_id])
            to_retract = [current] + [item for item in dependents if item.status is AssertionStatus.ACTIVE]
            retracted: list[Assertion] = []
            invalidated: list[str] = []
            for item in to_retract:
                await self._invalidate_eligibility(item.revision_id)
                state = _assertion_with(
                    item, revision_id=uuid4().hex, status=AssertionStatus.RETRACTED,
                    supersedes_revision_id=None, epistemic_state=EpistemicState.RETRACTED,
                )
                await self._write_revision(state, ())
                await self._set_current(state)
                retracted.append(state)
                invalidated.append(item.revision_id)
            generation = await self._advance_generation()
            for item in retracted:
                await self._event(item, "retracted", generation)
            await self._record_operation(
                operation_id, "retract", digest,
                {"revision_ids": [item.revision_id for item in retracted], "invalidated_revision_ids": invalidated, "generation": generation},
                [item.assertion_id for item in retracted], [item.revision_id for item in retracted],
            )
            return RetractionResult(tuple(retracted), tuple(invalidated), generation)

    async def quarantine_for_validation(
        self,
        assertion_id: str,
        expected_revision_id: str,
        *,
        report_id: str,
        operation_id: str | None = None,
    ) -> ValidationQuarantineResult:
        """Hide a failed current assertion through the normal change outbox.

        Validation owns neither projection tables nor derived-row lifecycle.
        It asks the canonical assertion authority for this transition, which
        invalidates the prior revision and every dependent in the same tenant
        transaction before publishing the #2744 change metadata consumers use
        to evict retrieval, inference, and training projections.
        """
        if not isinstance(report_id, str) or not report_id:
            raise AssertionStoreError("validation quarantine requires a non-empty report_id")
        operation_id = operation_id or f"validation-quarantine:{report_id}:{assertion_id}:{expected_revision_id}"
        request = {
            "assertion_id": assertion_id,
            "expected_revision_id": expected_revision_id,
            "report_id": report_id,
        }
        async with self._mutation():
            digest, replay = await self._operation(operation_id, "validation-quarantine", request)
            if replay is not None:
                quarantined = await self._revision(str(replay["quarantined_revision_id"]))
                invalidated = [
                    await self._revision(str(item))
                    for item in replay["invalidation_state_revision_ids"]
                ]
                if quarantined is None or any(item is None for item in invalidated):
                    raise AssertionConflictError("idempotent validation quarantine receipt lost its revisions")
                return ValidationQuarantineResult(
                    quarantined,
                    tuple(item for item in invalidated if item is not None),
                    tuple(replay["invalidated_revision_ids"]),
                    int(replay["generation"]),
                    True,
                )
            current = await self._current(assertion_id)
            if current is None or current.revision_id != expected_revision_id or current.status is not AssertionStatus.ACTIVE:
                raise AssertionConflictError("expected revision is not the active current tenant assertion")
            dependents = await self._dependent_current_revisions([current.revision_id])
            invalidated_revision_ids = [current.revision_id]
            await self._invalidate_eligibility(current.revision_id)
            quarantined = _assertion_with(
                current,
                revision_id=uuid4().hex,
                status=AssertionStatus.QUARANTINED,
                supersedes_revision_id=None,
            )
            await self._write_revision(quarantined, ())
            await self._set_current(quarantined)
            invalidated: list[Assertion] = []
            for dependent in dependents:
                if dependent.status is not AssertionStatus.ACTIVE:
                    continue
                await self._invalidate_eligibility(dependent.revision_id)
                invalidated_revision_ids.append(dependent.revision_id)
                state = _assertion_with(
                    dependent,
                    revision_id=uuid4().hex,
                    status=AssertionStatus.RETRACTED,
                    supersedes_revision_id=None,
                    epistemic_state=EpistemicState.RETRACTED,
                )
                await self._write_revision(state, ())
                await self._set_current(state)
                invalidated.append(state)
            generation = await self._advance_generation()
            quarantine_event = await self._event(quarantined, "validation_quarantined", generation)
            dependent_events = [await self._event(item, "retracted", generation) for item in invalidated]
            await self._record_operation(
                operation_id,
                "validation-quarantine",
                digest,
                {
                    "quarantined_revision_id": quarantined.revision_id,
                    "invalidation_state_revision_ids": [item.revision_id for item in invalidated],
                    "invalidated_revision_ids": invalidated_revision_ids,
                    "generation": generation,
                    "event_ids": [quarantine_event, *dependent_events],
                },
                [quarantined.assertion_id, *[item.assertion_id for item in invalidated]],
                [quarantined.revision_id, *[item.revision_id for item in invalidated]],
            )
            return ValidationQuarantineResult(
                quarantined,
                tuple(invalidated),
                tuple(invalidated_revision_ids),
                generation,
            )

    async def delete(
        self,
        assertion_id: str,
        expected_revision_id: str,
        *,
        operation_id: str | None = None,
    ) -> DeletionResult:
        """Apply a non-erasure lifecycle deletion and invalidate dependents.

        Unlike :meth:`erase`, this preserves the audit shell as a current
        ``deleted`` revision.  It remains ineligible for query, index, and
        training consumption; physical erasure removes that shell entirely.
        """
        operation_id = operation_id or f"delete:{assertion_id}:{expected_revision_id}"
        request = {"assertion_id": assertion_id, "expected_revision_id": expected_revision_id}
        async with self._mutation():
            digest, replay = await self._operation(operation_id, "delete", request)
            if replay is not None:
                deleted = await self._revision(str(replay["deleted_revision_id"]))
                invalidated = [
                    await self._revision(str(item))
                    for item in replay["invalidation_state_revision_ids"]
                ]
                if deleted is None or any(item is None for item in invalidated):
                    raise AssertionConflictError("idempotent deletion receipt no longer has its revisions")
                return DeletionResult(
                    deleted, tuple(item for item in invalidated if item is not None),
                    tuple(replay["invalidated_revision_ids"]), int(replay["generation"]), True,
                )
            current = await self._current(assertion_id)
            if current is None or current.revision_id != expected_revision_id or current.status is not AssertionStatus.ACTIVE:
                raise AssertionConflictError("expected revision is not the active current tenant assertion")
            dependents = await self._dependent_current_revisions([current.revision_id])
            invalidated_revision_ids = [current.revision_id]
            await self._invalidate_eligibility(current.revision_id)
            deleted = _assertion_with(
                current, revision_id=uuid4().hex, status=AssertionStatus.DELETED,
                supersedes_revision_id=None,
            )
            await self._write_revision(deleted, ())
            await self._set_current(deleted)
            invalidated: list[Assertion] = []
            for dependent in dependents:
                if dependent.status is not AssertionStatus.ACTIVE:
                    continue
                await self._invalidate_eligibility(dependent.revision_id)
                invalidated_revision_ids.append(dependent.revision_id)
                state = _assertion_with(
                    dependent, revision_id=uuid4().hex, status=AssertionStatus.RETRACTED,
                    supersedes_revision_id=None, epistemic_state=EpistemicState.RETRACTED,
                )
                await self._write_revision(state, ())
                await self._set_current(state)
                invalidated.append(state)
            generation = await self._advance_generation()
            await self._event(deleted, "deleted", generation)
            for item in invalidated:
                await self._event(item, "retracted", generation)
            await self._record_operation(
                operation_id, "delete", digest,
                {
                    "deleted_revision_id": deleted.revision_id,
                    "invalidated_revision_ids": invalidated_revision_ids,
                    "invalidation_state_revision_ids": [item.revision_id for item in invalidated],
                    "generation": generation,
                },
                [deleted.assertion_id, *[item.assertion_id for item in invalidated]],
                [deleted.revision_id, *[item.revision_id for item in invalidated]],
            )
            return DeletionResult(deleted, tuple(invalidated), tuple(invalidated_revision_ids), generation)

    async def _sanitize_surviving_references_after_erasure(
        self,
        erased_revision_ids: Sequence[str],
    ) -> None:
        """Remove predecessor links from every surviving assertion revision.

        Erasure can remove an inferred historical revision while retaining a
        later direct fact with the same deterministic assertion identity.  A
        later supersession can make that direct fact historical too, so every
        retained revision must lose any ``supersedes`` pointer into erased
        history, including in its canonical mapping.  This is an erasure-only
        redaction of otherwise immutable revisions: it leaves direct facts and
        their revision identities intact, while ensuring no durable row can
        reconstruct the removed lineage.
        """
        if not erased_revision_ids:
            return
        tenant_id, _ = self._require_scope()
        erased = tuple(sorted(set(erased_revision_ids)))
        rows = await self._database.fetchall(
            "SELECT r.revision_id, r.assertion_mapping "
            "FROM semantic_assertion_revisions r "
            "WHERE r.tenant_id = ? "
            f"AND r.supersedes_revision_id IN ({_placeholders(erased)}) "
            f"AND r.revision_id NOT IN ({_placeholders(erased)})",
            (tenant_id,) + erased + erased,
        )
        for revision_id, encoded in rows:
            try:
                assertion = Assertion.from_mapping(json.loads(encoded))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise AssertionStoreError(
                    "surviving assertion row contains an invalid canonical mapping"
                ) from error
            if (
                assertion.revision_id != revision_id
                or assertion.supersedes_revision_id not in erased
            ):
                raise AssertionStoreError(
                    "surviving assertion row is inconsistent with its predecessor reference"
                )
            sanitized_mapping = assertion.to_mapping()
            sanitized_mapping["supersedes_revision_id"] = None
            sanitized = Assertion.from_mapping(sanitized_mapping)
            await self._database.execute(
                "UPDATE semantic_assertion_revisions "
                "SET supersedes_revision_id = NULL, assertion_mapping = ? "
                "WHERE tenant_id = ? AND revision_id = ?",
                (_json(sanitized.to_mapping()), tenant_id, revision_id),
            )

    async def _delete_validation_reports_referencing(
        self,
        assertion_ids: Sequence[str],
    ) -> None:
        """Erase report artifacts that retain a physically erased identity."""
        if not assertion_ids:
            return
        tenant_id, _ = self._require_scope()
        identifiers = tuple(sorted(set(assertion_ids)))
        report_rows = await self._database.fetchall(
            "SELECT DISTINCT report_id FROM semantic_validation_report_assertions "
            "WHERE tenant_id = ? "
            f"AND assertion_id IN ({_placeholders(identifiers)})",
            (tenant_id,) + identifiers,
        )
        report_ids = tuple(row[0] for row in report_rows)
        if not report_ids:
            return
        await self._database.execute(
            "DELETE FROM semantic_validation_results WHERE tenant_id = ? "
            f"AND report_id IN ({_placeholders(report_ids)})",
            (tenant_id,) + report_ids,
        )
        await self._database.execute(
            "DELETE FROM semantic_validation_report_assertions WHERE tenant_id = ? "
            f"AND report_id IN ({_placeholders(report_ids)})",
            (tenant_id,) + report_ids,
        )
        await self._database.execute(
            "DELETE FROM semantic_validation_reports WHERE tenant_id = ? "
            f"AND report_id IN ({_placeholders(report_ids)})",
            (tenant_id,) + report_ids,
        )

    async def erase(self, assertion_id: str, *, operation_id: str | None = None) -> ErasureResult:
        """Physically erase an assertion and its transitive derived closure.

        This operation deliberately removes projection eligibility and outbox
        rows that name the erased identities.  Pins and derivation references
        are not retaining authorities and therefore cannot block the delete.
        """
        # Keep the durable retry key opaque even for callers that omit one.
        # Clients that need to retry after losing a response should supply a
        # stable operation_id; this deterministic fallback only avoids storing
        # the target assertion ID in the receipt key itself.
        operation_id = operation_id or f"erase:{_operation_digest({'assertion_id': assertion_id})}"
        request = {"assertion_id": assertion_id}
        async with self._mutation():
            digest, replay = await self._erasure_operation(operation_id, request)
            if replay is not None:
                remembered = self._remembered_erasure_job(operation_id, digest)
                if remembered is not None:
                    return remembered
                # Restart-safe idempotency is intentionally identity-free.
                # The durable receipt proves the completed checkpoint without
                # retaining erased assertion or revision identifiers.
                return ErasureResult((), (), replay, True)
            tenant_id, _ = self._require_scope()
            seed_rows = await self._database.fetchall(
                "SELECT revision_id FROM semantic_assertion_revisions WHERE tenant_id = ? AND assertion_id = ?",
                (tenant_id, assertion_id),
            )
            if not seed_rows:
                raise AssertionConflictError("assertion is absent from the bound tenant")
            revision_ids = {row[0] for row in seed_rows}
            assertion_ids = {assertion_id}
            pending = set(revision_ids)
            while pending:
                batch = tuple(sorted(pending))
                pending.clear()
                rows = await self._database.fetchall(
                    "SELECT DISTINCT r.assertion_id, r.revision_id, a.current_revision_id "
                    "FROM semantic_derivation_inputs d "
                    "JOIN semantic_assertion_revisions r ON r.tenant_id = d.tenant_id "
                    "  AND r.revision_id = d.derived_revision_id "
                    "JOIN semantic_assertions a ON a.tenant_id = r.tenant_id "
                    "  AND a.assertion_id = r.assertion_id "
                    f"WHERE d.tenant_id = ? AND d.input_revision_id IN ({_placeholders(batch)}) "
                    "ORDER BY r.assertion_id ASC, r.revision_id ASC",
                    (tenant_id,) + batch,
                )
                for derived_assertion_id, derived_revision_id, current_revision_id in rows:
                    # Derivation rows are historical evidence. Follow them
                    # after a dependent is superseded, otherwise a root
                    # erasure leaves recoverable lineage behind.
                    if derived_revision_id not in revision_ids:
                        revision_ids.add(derived_revision_id)
                        pending.add(derived_revision_id)
                    if (
                        current_revision_id != derived_revision_id
                        or derived_assertion_id in assertion_ids
                    ):
                        # A newer independent current revision owns this
                        # deterministic assertion identity. Remove only its
                        # reachable historical lineage, never that replacement.
                        continue
                    assertion_ids.add(derived_assertion_id)
                    all_revisions = await self._database.fetchall(
                        "SELECT revision_id FROM semantic_assertion_revisions WHERE tenant_id = ? AND assertion_id = ? ORDER BY revision_id ASC",
                        (tenant_id, derived_assertion_id),
                    )
                    for (revision_id,) in all_revisions:
                        if revision_id not in revision_ids:
                            revision_ids.add(revision_id)
                            pending.add(revision_id)
            revision_tuple = tuple(sorted(revision_ids))
            assertion_tuple = tuple(sorted(assertion_ids))
            await self._sanitize_surviving_references_after_erasure(
                revision_tuple,
            )
            source_rows = await self._database.fetchall(
                "SELECT DISTINCT source_occurrence_id FROM semantic_revision_sources "
                f"WHERE tenant_id = ? AND revision_id IN ({_placeholders(revision_tuple)})",
                (tenant_id,) + revision_tuple,
            )
            await self._database.execute(
                f"DELETE FROM semantic_projection_outbox WHERE tenant_id = ? AND revision_id IN ({_placeholders(revision_tuple)})",
                (tenant_id,) + revision_tuple,
            )
            await self._database.execute(
                f"DELETE FROM semantic_projection_eligibility WHERE tenant_id = ? AND revision_id IN ({_placeholders(revision_tuple)})",
                (tenant_id,) + revision_tuple,
            )
            await self._database.execute(
                f"DELETE FROM semantic_derivation_inputs WHERE tenant_id = ? AND (derived_revision_id IN ({_placeholders(revision_tuple)}) OR input_revision_id IN ({_placeholders(revision_tuple)}))",
                (tenant_id,) + revision_tuple + revision_tuple,
            )
            await self._database.execute(
                f"DELETE FROM semantic_revision_sources WHERE tenant_id = ? AND revision_id IN ({_placeholders(revision_tuple)})",
                (tenant_id,) + revision_tuple,
            )
            await self._delete_operations_referencing(assertion_ids, revision_ids)
            await self._delete_validation_reports_referencing(assertion_tuple)
            await self._database.execute(
                f"DELETE FROM semantic_assertions WHERE tenant_id = ? AND assertion_id IN ({_placeholders(assertion_tuple)})",
                (tenant_id,) + assertion_tuple,
            )
            await self._database.execute(
                f"DELETE FROM semantic_assertion_revisions WHERE tenant_id = ? AND revision_id IN ({_placeholders(revision_tuple)})",
                (tenant_id,) + revision_tuple,
            )
            for (source_id,) in source_rows:
                referenced = await self._database.fetchone(
                    "SELECT 1 FROM semantic_revision_sources WHERE tenant_id = ? AND source_occurrence_id = ?",
                    (tenant_id, source_id),
                )
                if referenced is None:
                    await self._database.execute(
                        "DELETE FROM semantic_source_occurrences WHERE tenant_id = ? AND source_occurrence_id = ?",
                        (tenant_id, source_id),
                    )
            generation = await self._advance_generation()
            await self._erasure_event(generation)
            result = ErasureResult(assertion_tuple, revision_tuple, generation)
            await self._record_erasure_operation(
                operation_id,
                digest,
                generation,
            )
            self._remember_erasure_job(operation_id, digest, result)
            return result

    async def query(self, query: AssertionQuery | None = None) -> list[Assertion]:
        """Query current tenant assertions; unqualified reads return active rows only."""
        query = query or AssertionQuery()
        if not isinstance(query, AssertionQuery):
            raise AssertionStoreError("query requires AssertionQuery")
        tenant_id, _ = self._require_scope()
        clauses = ["a.tenant_id = ?", "a.current_revision_id = r.revision_id"]
        params: list[object] = [tenant_id]
        if query.subject is not None:
            clauses.append("r.subject_value = ?")
            params.append(query.subject.value)
        if query.predicate is not None:
            clauses.append("r.predicate_value = ?")
            params.append(query.predicate.value)
        if query.object is not None:
            mapping = query.object.identity_mapping()
            clauses.extend([
                "r.object_kind = ?", "r.object_value = ?",
                "(r.object_datatype = ? OR (r.object_datatype IS NULL AND ? IS NULL))",
                "(r.object_language = ? OR (r.object_language IS NULL AND ? IS NULL))",
            ])
            datatype = mapping.get("datatype")
            language = mapping.get("language")
            params.extend([mapping["kind"], mapping["value"], datatype, datatype, language, language])
        if query.assertion_ids:
            clauses.append(f"a.assertion_id IN ({_placeholders(query.assertion_ids)})")
            params.extend(query.assertion_ids)
        statuses = query.statuses or (AssertionStatus.ACTIVE,)
        clauses.append(f"r.status IN ({_placeholders(statuses)})")
        params.extend(status.value for status in statuses)
        if query.epistemic_states:
            clauses.append(f"r.epistemic_state IN ({_placeholders(query.epistemic_states)})")
            params.extend(state.value for state in query.epistemic_states)
        if query.valid_at is not None:
            clauses.append("(r.valid_start IS NULL OR r.valid_start <= ?)")
            clauses.append("(r.valid_end IS NULL OR r.valid_end >= ?)")
            params.extend([query.valid_at.value, query.valid_at.value])
        if query.observed_at is not None:
            clauses.append("(r.observed_start IS NULL OR r.observed_start <= ?)")
            clauses.append("(r.observed_end IS NULL OR r.observed_end >= ?)")
            params.extend([query.observed_at.value, query.observed_at.value])
        if query.cursor is not None:
            clauses.append("r.revision_id > ?")
            params.append(query.cursor)
        params.append(query.limit)
        rows = await self._database.fetchall(
            "SELECT r.assertion_mapping FROM semantic_assertions a "
            "JOIN semantic_assertion_revisions r ON r.tenant_id = a.tenant_id "
            "WHERE " + " AND ".join(clauses) + " ORDER BY r.revision_id ASC LIMIT ?",
            tuple(params),
        )
        return [Assertion.from_mapping(json.loads(row[0])) for row in rows]

    async def inference_inputs(self, query: AssertionQuery | None = None) -> list[Assertion]:
        """Return only current, active, projection-eligible tenant assertions."""
        values = await self.query(query)
        return [value for value in values if value.status is AssertionStatus.ACTIVE]

    async def export_snapshot(self, query: AssertionQuery | None = None) -> tuple[AssertionCheckpoint, tuple[Assertion, ...]]:
        """Return a tenant-bound active snapshot for a caller already scoped here.

        The store deliberately has no per-call tenant argument.  Export callers
        receive only records selected through the same lifecycle-filtered query
        used by inference; a higher layer may further apply release policy and
        destination governance without widening this canonical scope.
        """
        checkpoint = await self.checkpoint()
        return checkpoint, tuple(await self.query(query))

    async def checkpoint(self) -> AssertionCheckpoint:
        tenant_id, _ = self._require_scope()
        generation = await self._generation()
        row = await self._database.fetchone(
            "SELECT event_id FROM ("
            "SELECT event_id, generation, created_at FROM semantic_projection_outbox WHERE tenant_id = ? "
            "UNION ALL "
            "SELECT event_id, generation, created_at FROM semantic_projection_erasure_outbox WHERE tenant_id = ?"
            ") AS assertion_events ORDER BY generation DESC, created_at DESC, event_id DESC LIMIT 1",
            (tenant_id, tenant_id),
        )
        return AssertionCheckpoint(tenant_id, generation, row[0] if row else None)

    async def changes_since(self, generation: int, *, limit: int = 100) -> list[AssertionChange]:
        if type(generation) is not int or generation < 0:
            raise AssertionStoreError("generation must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise AssertionStoreError("limit must be an integer in [1, 1000]")
        tenant_id, _ = self._require_scope()
        rows = await self._database.fetchall(
            "SELECT event_id, assertion_id, revision_id, operation, generation, eligible FROM ("
            "SELECT event_id, assertion_id, revision_id, operation, generation, eligible, created_at "
            "FROM semantic_projection_outbox WHERE tenant_id = ? AND generation > ? "
            "UNION ALL "
            "SELECT event_id, NULL AS assertion_id, NULL AS revision_id, operation, generation, 0 AS eligible, created_at "
            "FROM semantic_projection_erasure_outbox WHERE tenant_id = ? AND generation > ?"
            ") AS assertion_changes ORDER BY generation ASC, created_at ASC, event_id ASC LIMIT ?",
            (tenant_id, generation, tenant_id, generation, limit),
        )
        return [AssertionChange(row[0], row[1], row[2], row[3], int(row[4]), bool(row[5])) for row in rows]


def _create_agent_bound_assertion_store(
    database: AsyncDatabase,
    *,
    tenant_capability: _AssertionTenantCapability,
) -> AsyncAssertionStore:
    """Issue the sole assertion authority for an initialized agent storage.

    This is intentionally private to the storage package.  A raw
    :class:`AsyncDatabase` never carries a tenant-selection capability.
    """
    scope = _AssertionStoreScope._issue(
        _ASSERTION_STORE_FACTORY_TOKEN,
        database,
        tenant_capability,
    )
    return AsyncAssertionStore(scope)


# Architecture documents use the role name; retain it as an explicit alias so
# callers do not create a competing persistence implementation to obtain it.
SemanticAssertionStore = AsyncAssertionStore


__all__ = [
    "AssertionChange", "AssertionCheckpoint", "AssertionConflictError", "AssertionStoreError",
    "AssertionWriteResult", "AsyncAssertionStore", "DeletionResult", "ErasureResult", "RetractionResult",
    "SemanticAssertionStore", "SupersessionResult", "TenantIsolationError", "ValidationQuarantineResult",
]
