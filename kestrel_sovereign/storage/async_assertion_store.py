"""Tenant-bound persistence for canonical semantic assertions.

``AsyncAssertionStore`` deliberately owns normalized assertion state rather
than serialising facts into ``graph_nodes.properties``.  The property graph is
an optional projection/input boundary; it has no write path back into these
tables.  All public mutation methods require a store bound to one tenant, and
the tenant predicate is applied before every lookup or traversal.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

from kestrel_sovereign.knowledge import (
    Assertion,
    AssertionQuery,
    AssertionStatus,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    IRI,
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
_RAW_ASSERTION_MUTATION_CAPABILITY_TOKEN = object()
_ERASURE_JOB_TTL_SECONDS = 300.0
_MAX_ERASURE_JOBS = 256
_EXPLICIT_FACT_FORGET_NOOP_OPERATION = "explicit_fact_forget_noop"
_LEGACY_ERASED_EXPLICIT_FACT_OPERATION = "legacy_erased_explicit_fact"
_EXPLICIT_FACT_SAVE_OPERATION_PREFIX = "memory-agency-save-fact-v1:save:"
_EXPLICIT_FACT_FORGET_OPERATION_PREFIX = "memory-agency-save-fact-v1:forget:"


@dataclass(frozen=True, slots=True)
class _MaintenanceFence:
    """Lease identity carried only while semantic maintenance is committing."""

    tenant_id: str
    holder_id: str
    fencing_token: int
    lease_seconds: float


_MAINTENANCE_FENCE: ContextVar[_MaintenanceFence | None] = ContextVar(
    "semantic_maintenance_fence",
    default=None,
)


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
class _RawAssertionMutationCapability:
    """Private migration-only authority for bypassing governed ingestion."""

    tenant_id: str

    def __init__(self, token: object, tenant_id: str) -> None:
        if token is not _RAW_ASSERTION_MUTATION_CAPABILITY_TOKEN:
            raise TypeError("raw assertion mutation capabilities are issued internally")
        object.__setattr__(self, "tenant_id", tenant_id)


def _issue_raw_assertion_mutation_capability(
    tenant_capability: _AssertionTenantCapability,
) -> _RawAssertionMutationCapability:
    """Issue a narrowly scoped private capability for one verified migration."""
    if type(tenant_capability) is not _AssertionTenantCapability:
        raise TypeError("raw assertion mutation capability requires tenant authority")
    return _RawAssertionMutationCapability(
        _RAW_ASSERTION_MUTATION_CAPABILITY_TOKEN,
        tenant_capability.tenant_id,
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


class MaintenanceLeaseLostError(AssertionStoreError):
    """A fenced maintenance mutation no longer owns its tenant lease."""


class TenantIsolationError(AssertionStoreError):
    """A caller attempted to cross the store's authoritative tenant scope."""


class AssertionConflictError(AssertionStoreError):
    """A lifecycle compare-and-swap or idempotency check did not match."""


class AssertionOperationErasedError(AssertionConflictError):
    """A matching operation completed earlier but its semantic data was erased."""


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
class GovernedAssertionOperationReplay:
    """The exact accepted receipt for one governed assertion write.

    This is intentionally ledger-derived: callers use it for a delivery retry
    before inspecting the mutable current revision.  A subsequent append or
    supersession therefore cannot make an older operation look like a new,
    conflicting proposal.
    """

    operation: str
    report: ShaclValidationReport
    assertion: Assertion
    predecessor: Assertion | None
    generation: int


@dataclass(frozen=True, slots=True)
class _GovernedAssertionReplayBinding:
    """Immutable explicit-fact result identity required for narrow replay.

    The adapter can rebuild these fields from its invocation after canonical
    state has moved on.  Delivery timestamps and predecessor/current state are
    deliberately absent: neither is stable across retries.
    """

    assertion_id: str
    revision_id: str
    source_occurrence_id: str
    adapter_request_digest: str
    proposal_fingerprint: str


@dataclass(frozen=True, slots=True)
class ErasedGovernedAssertionOperationReplay:
    """Identity-free terminal replay after physical semantic erasure."""

    operation: str
    generation: int
    terminal_erased: bool = True


@dataclass(frozen=True, slots=True)
class SupersessionLifecyclePlan:
    """The ledger-aware active graph a supersession will leave behind.

    The canonical derivation-input table contains one chosen lineage per
    inferred assertion, while the inference ledger records every independently
    sufficient proof.  A supersession must use the latter before deciding
    which conclusions leave the active graph.  This plan is deliberately
    shared by governed pre-commit validation and the canonical mutation, so a
    SHACL report cannot validate a graph different from the graph committed.
    """

    predecessor: Assertion
    dependent_assertions: tuple[Assertion, ...]
    withdrawn_revision_ids: tuple[str, ...]
    deactivated_derivation_ids: tuple[str, ...]
    post_state: tuple[Assertion, ...]


@dataclass(frozen=True, slots=True)
class RestorationLifecyclePlan:
    """The terminal direct shell and validated graph for one re-teaching."""

    predecessor: Assertion
    post_state: tuple[Assertion, ...]


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
class ErasedDeletionOperationReplay:
    """Identity-free terminal replay for a physically erased deletion."""

    generation: int
    idempotent: bool = True
    terminal_erased: bool = True


@dataclass(frozen=True, slots=True)
class ExplicitFactForgetReplay:
    """Exact ledger outcome for one explicit-fact forget invocation."""

    deletion: DeletionResult | None
    idempotent: bool

    @property
    def deleted(self) -> bool:
        return self.deletion is not None


@dataclass(frozen=True, slots=True)
class ValidationQuarantineResult:
    """A validation-driven lifecycle transition and its derived invalidations."""

    quarantined: Assertion
    invalidated: tuple[Assertion, ...]
    invalidated_revision_ids: tuple[str, ...]
    generation: int
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class ValidationQuarantineBatchResult:
    """One report-backed, all-or-nothing validation repair transition."""

    quarantined: tuple[Assertion, ...]
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
class InferenceRevocationResult:
    """Tenant-local outcome of disabling one materialization engine."""

    retracted_assertions: int
    deactivated_derivations: int
    generation: int


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
    """Hash semantic operation intent, excluding transport delivery clocks.

    A retry preserves the invocation and its semantic proposal, but it is a
    distinct HTTP delivery and therefore has a new honest ``received_at``
    timestamp.  The assertion mirrors that source timestamp in
    ``asserted_at``.  Neither volatile clock may prevent the operation ledger
    from returning the original canonical receipt; the committed assertion
    and source still retain the first delivery's actual timestamps.
    """
    def without_delivery_timestamps(item: object) -> object:
        if isinstance(item, dict):
            return {
                key: without_delivery_timestamps(nested)
                for key, nested in item.items()
                if key not in {"asserted_at", "received_at"}
            }
        if isinstance(item, (list, tuple)):
            return [without_delivery_timestamps(nested) for nested in item]
        return item

    normalized = without_delivery_timestamps(value)
    return hashlib.sha256(_json(normalized).encode("utf-8")).hexdigest()


def _erasure_receipt_key(operation_id: str) -> str:
    """Derive the durable, opaque lookup key for an erasure retry."""
    return _operation_digest({"erasure_operation_id": operation_id})


def _erased_operation_key(operation_id: str) -> str:
    """Blind one ordinary semantic operation ID retained after erasure."""
    return _operation_digest(
        {
            "namespace": "semantic-assertion-erased-operation-v1",
            "operation_id": operation_id,
        }
    )


def _erased_operation_request_key(
    operation_id: str,
    purpose: str,
    request_digest: str,
) -> str:
    """Bind a blinded tombstone to its exact prior purpose and request.

    The stored value is not the ordinary request digest.  Keying this second
    hash with the opaque operation ID prevents the tombstone from becoming a
    content-confirmation oracle while still rejecting reuse with a different
    request.
    """
    return _operation_digest(
        {
            "namespace": "semantic-assertion-erased-operation-request-v1",
            "operation_id": operation_id,
            "purpose": purpose,
            "request_digest": request_digest,
        }
    )


def _validate_governed_replay_binding(
    binding: _GovernedAssertionReplayBinding,
) -> None:
    if not isinstance(binding, _GovernedAssertionReplayBinding):
        raise AssertionStoreError(
            "governed assertion replay requires an immutable result binding"
        )
    values = (
        binding.assertion_id,
        binding.revision_id,
        binding.source_occurrence_id,
        binding.adapter_request_digest,
        binding.proposal_fingerprint,
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise AssertionStoreError(
            "governed assertion replay binding fields must be non-empty strings"
        )
    if (
        len(binding.adapter_request_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in binding.adapter_request_digest
        )
    ):
        raise AssertionStoreError(
            "governed assertion replay adapter digest must be lowercase sha256"
        )
    if (
        len(binding.proposal_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in binding.proposal_fingerprint
        )
    ):
        raise AssertionStoreError(
            "governed assertion replay proposal fingerprint must be lowercase sha256"
        )


def _governed_assertion_replay_binding(
    assertion: Assertion,
    source: SourceOccurrence,
    adapter_request_digest: str,
) -> _GovernedAssertionReplayBinding:
    """Bind every immutable proposal field while ignoring delivery clocks."""
    if not isinstance(assertion, Assertion) or not isinstance(
        source,
        SourceOccurrence,
    ):
        raise AssertionStoreError(
            "governed assertion replay binding requires canonical proposal data"
        )
    assertion_mapping = assertion.to_mapping()
    for field in (
        "revision_id",
        "status",
        "supersedes_revision_id",
        "asserted_at",
        "lineage",
    ):
        assertion_mapping.pop(field, None)
    source_mapping = source.to_mapping()
    source_mapping.pop("received_at", None)
    fingerprint = _operation_digest(
        {
            "namespace": "semantic-explicit-fact-proposal-v1",
            "assertion": assertion_mapping,
            "direct_source": source_mapping,
        }
    )
    binding = _GovernedAssertionReplayBinding(
        assertion_id=assertion.assertion_id,
        revision_id=assertion.revision_id,
        source_occurrence_id=source.source_occurrence_id,
        adapter_request_digest=adapter_request_digest,
        proposal_fingerprint=fingerprint,
    )
    _validate_governed_replay_binding(binding)
    return binding


def _erased_explicit_fact_result_key(
    operation_id: str,
    purpose: str,
    binding: _GovernedAssertionReplayBinding,
) -> str:
    """Blind the immutable accepted result needed to authenticate a retry."""
    _validate_governed_replay_binding(binding)
    return _operation_digest(
        {
            "namespace": "semantic-explicit-fact-erased-result-v1",
            "operation_id": operation_id,
            "purpose": purpose,
            "assertion_id": binding.assertion_id,
            "revision_id": binding.revision_id,
            "source_occurrence_id": binding.source_occurrence_id,
            "adapter_request_digest": binding.adapter_request_digest,
            "proposal_fingerprint": binding.proposal_fingerprint,
        }
    )


def _erased_explicit_fact_forget_selector_key(
    operation_id: str,
    subject: IRI,
    predicate: IRI,
) -> str:
    """Blind an authenticated adapter deletion selector after erasure."""
    if (
        not operation_id.startswith(_EXPLICIT_FACT_FORGET_OPERATION_PREFIX)
        or not isinstance(subject, IRI)
        or not isinstance(predicate, IRI)
    ):
        raise AssertionStoreError(
            "explicit fact forget binding requires its deterministic selector"
        )
    return _operation_digest(
        {
            "namespace": "semantic-explicit-fact-erased-forget-v1",
            "operation_id": operation_id,
            "purpose": "delete",
            "subject": subject.value,
            "predicate": predicate.value,
        }
    )


def _explicit_fact_binding_from_request(
    operation_id: str,
    purpose: str,
    request: object,
) -> _GovernedAssertionReplayBinding | None:
    """Extract only immutable result fields from a governed write request."""
    if (
        not operation_id.startswith(_EXPLICIT_FACT_SAVE_OPERATION_PREFIX)
        or purpose not in {"put", "supersede", "restore"}
        or not isinstance(request, Mapping)
    ):
        return None
    assertion_mapping = request.get(
        "assertion" if purpose == "put" else "replacement"
    )
    source_mappings = request.get("sources")
    if not isinstance(assertion_mapping, Mapping) or not isinstance(
        source_mappings, list
    ):
        return None
    try:
        assertion = Assertion.from_mapping(assertion_mapping)
        sources = tuple(
            SourceOccurrence.from_mapping(item)
            for item in source_mappings
            if isinstance(item, Mapping)
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        len(sources) != len(source_mappings)
        or not isinstance(assertion.lineage, DirectLineage)
        or assertion.revision_id not in assertion.lineage.source_occurrence_ids
    ):
        return None
    source = next(
        (
            item
            for item in sources
            if item.source_occurrence_id == assertion.revision_id
        ),
        None,
    )
    digest_prefix = "sha256:"
    if (
        source is None
        or not isinstance(source.content_digest, str)
        or not source.content_digest.startswith(digest_prefix)
    ):
        return None
    digest = source.content_digest[len(digest_prefix) :]
    try:
        binding = _governed_assertion_replay_binding(
            assertion,
            source,
            digest,
        )
        _validate_governed_replay_binding(binding)
    except AssertionStoreError:
        return None
    return binding


def _legacy_erasure_assertion_key(erasure_request_digest: str) -> str:
    """Blind a legacy erasure's already-opaque assertion request digest."""
    return _operation_digest(
        {
            "namespace": "semantic-assertion-legacy-erasure-fence-v1",
            "erasure_request_digest": erasure_request_digest,
        }
    )


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


def _validation_report_targets(report: ShaclValidationReport) -> tuple[str, ...]:
    """Return each assertion a quarantine report requires us to repair."""
    targets = {
        assertion_id
        for finding in report.findings
        for assertion_id in finding.affected_assertion_ids
    }
    if not targets or any(
        not finding.affected_assertion_ids for finding in report.findings
    ):
        targets.update(report.assertion_ids)
    return tuple(sorted(targets))


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

    def _require_raw_mutation_capability(
        self,
        capability: object | None,
    ) -> None:
        if type(capability) is not _RawAssertionMutationCapability:
            raise AssertionStoreError(
                "raw assertion mutation requires a migration-only capability"
            )
        if capability.tenant_id != self.tenant_id:
            raise TenantIsolationError(
                "raw assertion mutation capability does not match the bound tenant"
            )

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
    async def maintenance_fence(
        self,
        *,
        holder_id: str,
        fencing_token: int,
        lease_seconds: float,
    ):
        """Fence canonical mutations to one active semantic-maintenance lease.

        The context carries no write authority by itself.  It makes every
        canonical mutation performed in its scope conditionally renew the
        matching lease *inside that mutation's transaction*.  A stale worker
        therefore cannot publish a validation quarantine, audit retraction,
        or inference revision after another worker has acquired the lease.
        """
        if not isinstance(holder_id, str) or not holder_id:
            raise AssertionStoreError("maintenance fence holder_id must be non-empty")
        if type(fencing_token) is not int or fencing_token < 1:
            raise AssertionStoreError("maintenance fence token must be a positive integer")
        if (
            not isinstance(lease_seconds, (int, float))
            or isinstance(lease_seconds, bool)
            or lease_seconds <= 0
        ):
            raise AssertionStoreError("maintenance fence lease_seconds must be positive")
        fence = _MaintenanceFence(
            self.tenant_id,
            holder_id,
            fencing_token,
            float(lease_seconds),
        )
        current = _MAINTENANCE_FENCE.get()
        if current is not None:
            if current != fence:
                raise AssertionStoreError("nested maintenance fence does not match active fence")
            yield
            return
        token = _MAINTENANCE_FENCE.set(fence)
        try:
            yield
        finally:
            _MAINTENANCE_FENCE.reset(token)

    async def _renew_maintenance_fence_in_mutation(self) -> None:
        """Renew and lock the active maintenance lease in this transaction."""
        fence = _MAINTENANCE_FENCE.get()
        if fence is None:
            return
        tenant_id, _ = self._require_scope()
        if fence.tenant_id != tenant_id:
            raise MaintenanceLeaseLostError("semantic_maintenance_lease_lost")
        if self._database.backend_type == "postgres":
            database_now = "EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)"
        else:
            database_now = "CAST(strftime('%s', 'now') AS REAL)"
        changed = await self._database.execute(
            "UPDATE semantic_maintenance_leases "
            f"SET expires_at = {database_now} + ?, updated_at = ? "
            f"WHERE tenant_id = ? AND holder_id = ? AND fencing_token = ? "
            f"AND expires_at > {database_now}",
            (
                fence.lease_seconds,
                _now(),
                tenant_id,
                fence.holder_id,
                fence.fencing_token,
            ),
        )
        if changed != 1:
            raise MaintenanceLeaseLostError("semantic_maintenance_lease_lost")

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
                await self._renew_maintenance_fence_in_mutation()
                await self._lock_tenant()
                yield
        except TransactionError as error:
            if isinstance(error.__cause__, AssertionStoreError):
                raise error.__cause__ from error
            raise

    @asynccontextmanager
    async def inference_publication(self):
        """Serialize one inference publication with canonical assertions.

        Materialization calculates a closure outside a transaction so it does
        not hold a tenant lock while applying bounded rules.  Publishing that
        closure is different: inferred revisions, their derivation ledger,
        and the durable completion checkpoint must observe one serialized
        tenant generation.  Reuse the canonical mutation boundary so SQLite
        and PostgreSQL get the same guarantee, and so inference writes cannot
        race a direct assertion lifecycle mutation.

        This is intentionally a narrow peer-service primitive rather than a
        general transaction API.  Callers still use the canonical assertion
        mutation methods inside the scope.
        """
        async with self._mutation():
            yield

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
            "SELECT DISTINCT s.source_mapping, s.received_at, s.source_occurrence_id "
            "FROM semantic_assertion_revisions r "
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

    async def _invalidate_eligibility(
        self,
        revision_id: str,
        *,
        deactivate_inference_derivations: bool = True,
    ) -> None:
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
        if deactivate_inference_derivations:
            await self._deactivate_inference_derivations_for_inputs((revision_id,))

    async def _event(
        self,
        assertion: Assertion,
        operation: str,
        generation: int,
        *,
        eligible: bool | None = None,
    ) -> str:
        tenant_id, _ = self._require_scope()
        event_id = uuid4().hex
        eligible = assertion.status is AssertionStatus.ACTIVE if eligible is None else eligible
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
            erased = await self._erased_operation_tombstone(operation_id)
            if erased is not None:
                prior_purpose, prior_request_key, _ = erased
                explicit_fact_binding = _explicit_fact_binding_from_request(
                    operation_id,
                    operation,
                    request,
                )
                explicit_fact_selector = (
                    request.get("explicit_fact_selector")
                    if isinstance(request, Mapping)
                    else None
                )
                if (
                    operation == "delete"
                    and operation_id.startswith(
                        _EXPLICIT_FACT_FORGET_OPERATION_PREFIX
                    )
                    and isinstance(explicit_fact_selector, Mapping)
                    and isinstance(
                        explicit_fact_selector.get("subject"),
                        str,
                    )
                    and isinstance(
                        explicit_fact_selector.get("predicate"),
                        str,
                    )
                ):
                    expected_request_key = (
                        _erased_explicit_fact_forget_selector_key(
                            operation_id,
                            IRI(explicit_fact_selector["subject"]),
                            IRI(explicit_fact_selector["predicate"]),
                        )
                    )
                elif explicit_fact_binding is not None:
                    expected_request_key = _erased_explicit_fact_result_key(
                        operation_id,
                        operation,
                        explicit_fact_binding,
                    )
                else:
                    expected_request_key = _erased_operation_request_key(
                        operation_id,
                        operation,
                        digest,
                    )
                if (
                    prior_purpose != operation
                    or prior_request_key != expected_request_key
                ):
                    raise AssertionConflictError(
                        "operation_id was already used for a different semantic mutation"
                    )
                raise AssertionOperationErasedError(
                    "semantic operation completed previously but was physically erased"
                )
            return digest, None
        if row[0] != operation or row[1] != digest:
            raise AssertionConflictError("operation_id was already used for a different semantic mutation")
        return digest, json.loads(row[2])

    async def _erased_operation_tombstone(
        self,
        operation_id: str,
    ) -> tuple[str, str, int] | None:
        """Resolve one exact blinded tombstone without offering enumeration."""
        tenant_id, _ = self._require_scope()
        if not isinstance(operation_id, str) or not operation_id:
            raise AssertionStoreError("operation_id must be a non-empty string")
        row = await self._database.fetchone(
            "SELECT purpose, request_key, generation "
            "FROM semantic_assertion_erased_operation_tombstones "
            "WHERE tenant_id = ? AND operation_key = ?",
            (tenant_id, _erased_operation_key(operation_id)),
        )
        if row is None:
            return None
        return str(row[0]), str(row[1]), int(row[2])

    async def _erased_governed_operation_replay(
        self,
        operation_id: str,
        binding: _GovernedAssertionReplayBinding,
    ) -> ErasedGovernedAssertionOperationReplay | None:
        """Resolve only exact governed/legacy tombstones after a replay race."""
        _validate_governed_replay_binding(binding)
        erased = await self._erased_operation_tombstone(operation_id)
        if erased is None:
            return None
        purpose, request_key, generation = erased
        if purpose in {"put", "supersede", "restore"}:
            expected_request_key = _erased_explicit_fact_result_key(
                operation_id,
                purpose,
                binding,
            )
        elif purpose == _LEGACY_ERASED_EXPLICIT_FACT_OPERATION:
            expected_request_key = _erased_operation_request_key(
                operation_id,
                purpose,
                binding.adapter_request_digest,
            )
        else:
            raise AssertionConflictError(
                "operation_id resolves to a non-governed erased semantic mutation"
            )
        if request_key != expected_request_key:
            # Unreleased v4 branch databases wrote ordinary canonical-request
            # keys into these tombstones.  Their migration also creates the
            # same per-assertion opaque fence used for released v3 data.  A
            # matching fence is enough only to fail closed as terminal: it is
            # not enough to rewrite/authenticate the old tombstone, because a
            # cross-producer operation-key collision is indistinguishable
            # after erasure.  Fresh stores have no such fence and conflict.
            candidate_erasure_digest = _operation_digest(
                {"assertion_id": binding.assertion_id}
            )
            legacy_fence = await self._database.fetchone(
                "SELECT 1 FROM semantic_assertion_legacy_erasure_fences "
                "WHERE tenant_id = ? AND assertion_key = ?",
                (
                    self.tenant_id,
                    _legacy_erasure_assertion_key(
                        candidate_erasure_digest
                    ),
                ),
            )
            if legacy_fence is not None:
                return ErasedGovernedAssertionOperationReplay(
                    operation=_LEGACY_ERASED_EXPLICIT_FACT_OPERATION,
                    generation=generation,
                )
            raise AssertionConflictError(
                "operation_id resolves to a different erased governed assertion result"
            )
        return ErasedGovernedAssertionOperationReplay(
            operation=purpose,
            generation=generation,
        )

    async def terminalize_legacy_erased_explicit_fact_operation(
        self,
        operation_id: str,
        binding: _GovernedAssertionReplayBinding,
    ) -> ErasedGovernedAssertionOperationReplay | None:
        """Lazily tombstone an exact v3-erased explicit-fact identity.

        v3 retained only ``H({assertion_id})`` in the opaque erasure receipt
        after deleting ordinary operation receipts.  The v5 migration blinds
        those digests again into per-tenant assertion fences.  Because
        explicit-fact assertion identities are deterministic, a retry can
        compare its in-memory proposal without recovering or storing the
        erased identifier or content.  A match is information-theoretically
        ambiguous with an intentional same-content re-teach, so this path
        fails closed; unrelated post-upgrade facts remain writable.  Restoring
        the exact erased identity would require a future explicit
        consent/epoch override contract, not this ordinary retry surface.
        """
        if not isinstance(operation_id, str) or not operation_id:
            raise AssertionStoreError(
                "legacy erased explicit-fact replay requires non-empty opaque identifiers"
            )
        _validate_governed_replay_binding(binding)
        adapter_request_digest = binding.adapter_request_digest
        assertion_id = binding.assertion_id
        candidate_erasure_digest = _operation_digest(
            {"assertion_id": assertion_id}
        )
        row = await self._database.fetchone(
            "SELECT generation "
            "FROM semantic_assertion_legacy_erasure_fences "
            "WHERE tenant_id = ? AND assertion_key = ?",
            (
                self.tenant_id,
                _legacy_erasure_assertion_key(candidate_erasure_digest),
            ),
        )
        if row is None:
            # Fresh v5 stores and unrelated post-upgrade facts stay on the
            # ordinary path without acquiring the tenant writer slot.
            return None
        fence_generation = int(row[0])
        async with self._mutation():
            if await self._recorded_operation(operation_id) is not None:
                # A concurrent normal commit won.  The caller will re-enter
                # its bounded loop and replay that complete receipt.
                return None
            erased = await self._erased_operation_tombstone(operation_id)
            if erased is not None:
                purpose, request_key, generation = erased
                if purpose in {"put", "supersede", "restore"}:
                    if request_key != _erased_explicit_fact_result_key(
                        operation_id,
                        purpose,
                        binding,
                    ):
                        raise AssertionConflictError(
                            "operation_id resolves to a different erased governed assertion result"
                        )
                    return ErasedGovernedAssertionOperationReplay(
                        operation=purpose,
                        generation=generation,
                    )
                expected_request_key = _erased_operation_request_key(
                    operation_id,
                    _LEGACY_ERASED_EXPLICIT_FACT_OPERATION,
                    adapter_request_digest,
                )
                if (
                    purpose != _LEGACY_ERASED_EXPLICIT_FACT_OPERATION
                    or request_key != expected_request_key
                ):
                    raise AssertionConflictError(
                        "operation_id was already used for a different semantic mutation"
                    )
                return ErasedGovernedAssertionOperationReplay(
                    operation=purpose,
                    generation=generation,
                )

            purpose = _LEGACY_ERASED_EXPLICIT_FACT_OPERATION
            await self._database.execute(
                "INSERT INTO semantic_assertion_erased_operation_tombstones "
                "(tenant_id, purpose, operation_key, request_key, generation, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.tenant_id,
                    purpose,
                    _erased_operation_key(operation_id),
                    _erased_operation_request_key(
                        operation_id,
                        purpose,
                        adapter_request_digest,
                    ),
                    fence_generation,
                    _now(),
                ),
            )
            return ErasedGovernedAssertionOperationReplay(
                operation=purpose,
                generation=fence_generation,
            )

    async def _recorded_operation(
        self,
        operation_id: str,
    ) -> tuple[str, dict[str, object]] | None:
        """Load one tenant-local operation receipt without reconstructing intent.

        A replay caller already possesses the stable, adapter-derived
        operation ID.  It must not derive a request from the current canonical
        state just to rediscover an older receipt: later revisions are allowed
        to change that state.  Mutations still use :meth:`_operation`, which
        verifies both the operation ID and complete request digest.
        """
        tenant_id, _ = self._require_scope()
        if not isinstance(operation_id, str) or not operation_id:
            raise AssertionStoreError("operation_id must be a non-empty string")
        row = await self._database.fetchone(
            "SELECT operation, receipt FROM semantic_assertion_operations "
            "WHERE tenant_id = ? AND operation_id = ?",
            (tenant_id, operation_id),
        )
        if row is None:
            return None
        try:
            receipt = json.loads(row[1])
        except (TypeError, json.JSONDecodeError) as error:
            raise AssertionConflictError(
                "semantic operation receipt is malformed"
            ) from error
        if not isinstance(receipt, dict):
            raise AssertionConflictError("semantic operation receipt is malformed")
        return str(row[0]), receipt

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

    async def _explicit_fact_binding_from_recorded_result(
        self,
        operation_id: str,
        purpose: str,
        receipt: Mapping[str, object],
    ) -> _GovernedAssertionReplayBinding | None:
        """Derive a fact result binding before its revisions are erased."""
        if (
            not operation_id.startswith(_EXPLICIT_FACT_SAVE_OPERATION_PREFIX)
            or purpose not in {"put", "supersede", "restore"}
        ):
            return None
        result_field = (
            "revision_id" if purpose == "put" else "replacement_revision_id"
        )
        revision_id = receipt.get(result_field)
        if not isinstance(revision_id, str) or not revision_id:
            return None
        assertion = await self._revision(revision_id)
        if (
            assertion is None
            or not isinstance(assertion.lineage, DirectLineage)
        ):
            return None
        predecessor: Assertion | None = None
        if purpose in {"supersede", "restore"}:
            predecessor_id = receipt.get("predecessor_revision_id")
            if not isinstance(predecessor_id, str) or not predecessor_id:
                return None
            predecessor = await self._revision(predecessor_id)
            if predecessor is None:
                return None
        if not self._governed_replay_result_structure_matches(
            purpose,
            assertion,
            predecessor,
            revision_id,
        ):
            return None
        row = await self._database.fetchone(
            "SELECT s.source_mapping "
            "FROM semantic_revision_sources rs "
            "JOIN semantic_source_occurrences s "
            "  ON s.tenant_id = rs.tenant_id "
            " AND s.source_occurrence_id = rs.source_occurrence_id "
            "WHERE rs.tenant_id = ? AND rs.revision_id = ? "
            "AND rs.source_occurrence_id = ?",
            (self.tenant_id, revision_id, revision_id),
        )
        if row is None:
            return None
        try:
            source = SourceOccurrence.from_mapping(json.loads(row[0]))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError):
            return None
        digest_prefix = "sha256:"
        if (
            not isinstance(source.content_digest, str)
            or not source.content_digest.startswith(digest_prefix)
            or assertion.asserted_at != source.received_at
        ):
            return None
        try:
            binding = _governed_assertion_replay_binding(
                assertion,
                source,
                source.content_digest[len(digest_prefix) :],
            )
            _validate_governed_replay_binding(binding)
        except AssertionStoreError:
            return None
        return binding

    async def _explicit_fact_forget_selector_from_recorded_result(
        self,
        operation_id: str,
        purpose: str,
        receipt: Mapping[str, object],
    ) -> tuple[IRI, IRI] | None:
        """Authenticate an adapter deletion before its result is erased."""
        if (
            purpose != "delete"
            or not operation_id.startswith(
                _EXPLICIT_FACT_FORGET_OPERATION_PREFIX
            )
        ):
            return None
        selector = receipt.get("explicit_fact_selector")
        deleted_revision_id = receipt.get("deleted_revision_id")
        if (
            not isinstance(selector, Mapping)
            or not isinstance(selector.get("subject"), str)
            or not isinstance(selector.get("predicate"), str)
            or not isinstance(deleted_revision_id, str)
            or not deleted_revision_id
        ):
            return None
        try:
            subject = IRI(selector["subject"])
            predicate = IRI(selector["predicate"])
        except (TypeError, ValueError):
            return None
        deleted = await self._revision(deleted_revision_id)
        if (
            deleted is None
            or deleted.status is not AssertionStatus.DELETED
            or deleted.subject != subject
            or deleted.predicate != predicate
            or deleted.confidence_method != "memory-agency-save-fact-v1"
            or deleted.confidence_basis != "explicit-tool-invocation"
            or not isinstance(deleted.lineage, DirectLineage)
        ):
            return None
        rows = await self._database.fetchall(
            "SELECT s.source_mapping "
            "FROM semantic_revision_sources rs "
            "JOIN semantic_source_occurrences s "
            "  ON s.tenant_id = rs.tenant_id "
            " AND s.source_occurrence_id = rs.source_occurrence_id "
            "WHERE rs.tenant_id = ? AND rs.revision_id = ?",
            (self.tenant_id, deleted_revision_id),
        )
        try:
            sources = [
                SourceOccurrence.from_mapping(json.loads(row[0]))
                for row in rows
            ]
        except (KeyError, TypeError, json.JSONDecodeError, ValueError):
            return None
        if not any(
            source.source_occurrence_id.startswith(
                "source:memory-agency-save-fact-v1:"
            )
            and source.selector == "tool-arguments"
            and (
                (
                    source.source_kind == "agent_tool_invocation"
                    and source.locator.startswith(
                        "tool:memory_agency.save_fact#"
                    )
                )
                or (
                    source.source_kind == "http_request"
                    and "#tool:memory_agency.save_fact#"
                    in source.locator
                )
            )
            for source in sources
        ):
            return None
        return subject, predicate

    async def _tombstone_and_delete_operations_referencing(
        self,
        assertion_ids: set[str],
        revision_ids: set[str],
        generation: int,
    ) -> None:
        """Blind then remove receipts that retain any physically erased ID.

        The identifiers are stored as JSON arrays for backend-neutral audit
        data. Parsing them avoids SQL ``LIKE`` semantics, which would treat
        ``%`` and ``_`` in a caller-provided ID as wildcards and could leave a
        recoverable receipt behind.  Every selected operation first receives
        an identity-free tombstone inside the erasure transaction.  A delayed
        retry therefore terminates without resurrecting data after the
        content-bearing receipt is deleted.
        """
        tenant_id, _ = self._require_scope()
        rows = await self._database.fetchall(
            "SELECT operation_id, operation, request_digest, receipt, "
            "assertion_ids, revision_ids "
            "FROM semantic_assertion_operations WHERE tenant_id = ?",
            (tenant_id,),
        )
        operations: list[
            tuple[
                str,
                str,
                str,
                _GovernedAssertionReplayBinding | None,
                tuple[IRI, IRI] | None,
            ]
        ] = []
        for (
            operation_id,
            purpose,
            request_digest,
            encoded_receipt,
            encoded_assertion_ids,
            encoded_revision_ids,
        ) in rows:
            try:
                receipt = json.loads(encoded_receipt)
                recorded_assertion_ids = json.loads(encoded_assertion_ids)
                recorded_revision_ids = json.loads(encoded_revision_ids)
            except (TypeError, json.JSONDecodeError) as error:
                raise AssertionStoreError(
                    "semantic operation receipt contains malformed identifier data"
                ) from error
            if not (
                isinstance(receipt, dict)
                and isinstance(recorded_assertion_ids, list)
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
                binding = await self._explicit_fact_binding_from_recorded_result(
                    str(operation_id),
                    str(purpose),
                    receipt,
                )
                forget_selector = (
                    await self._explicit_fact_forget_selector_from_recorded_result(
                        str(operation_id),
                        str(purpose),
                        receipt,
                    )
                )
                operations.append(
                    (
                        str(operation_id),
                        str(purpose),
                        str(request_digest),
                        binding,
                        forget_selector,
                    )
                )
        if operations:
            for (
                operation_id,
                purpose,
                request_digest,
                binding,
                forget_selector,
            ) in operations:
                operation_key = _erased_operation_key(operation_id)
                request_key = (
                    _erased_explicit_fact_forget_selector_key(
                        operation_id,
                        forget_selector[0],
                        forget_selector[1],
                    )
                    if forget_selector is not None
                    else _erased_explicit_fact_result_key(
                        operation_id,
                        purpose,
                        binding,
                    )
                    if binding is not None
                    else _erased_operation_request_key(
                        operation_id,
                        purpose,
                        request_digest,
                    )
                )
                existing = await self._database.fetchone(
                    "SELECT purpose, request_key, generation "
                    "FROM semantic_assertion_erased_operation_tombstones "
                    "WHERE tenant_id = ? AND operation_key = ?",
                    (tenant_id, operation_key),
                )
                if existing is None:
                    await self._database.execute(
                        "INSERT INTO semantic_assertion_erased_operation_tombstones "
                        "(tenant_id, purpose, operation_key, request_key, generation, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            tenant_id,
                            purpose,
                            operation_key,
                            request_key,
                            generation,
                            _now(),
                        ),
                    )
                elif (
                    str(existing[0]) != purpose
                    or str(existing[1]) != request_key
                    or int(existing[2]) != generation
                ):
                    raise AssertionConflictError(
                        "erased semantic operation tombstone conflicts with its prior request"
                    )
            operation_ids = [
                operation_id for operation_id, _, _, _, _ in operations
            ]
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
        erased = await self._erased_operation_tombstone(operation_id)
        if erased is not None:
            raise AssertionConflictError(
                "operation_id was already used for a physically erased semantic mutation"
            )
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
        _migration_capability: _RawAssertionMutationCapability | None = None,
    ) -> AssertionWriteResult:
        """Private migration-only raw initial assertion mutation."""
        self._require_raw_mutation_capability(_migration_capability)
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

    async def publish_inferred_assertion(
        self,
        assertion: Assertion,
        *,
        operation_id: str | None = None,
    ) -> AssertionWriteResult:
        """Publish a lineage-validated result from the bounded materializer.

        This is deliberately narrower than the migration-only raw writer:
        only an active inferred assertion with canonical derived lineage can
        cross it.  Public asserted/imported ingestion remains report-governed
        through :class:`AsyncStorage`; this preserves one trusted, bounded
        publication capability for the inference engine without reopening a
        generic validation bypass.
        """
        if (
            not isinstance(assertion, Assertion)
            or assertion.status is not AssertionStatus.ACTIVE
            or assertion.epistemic_state is not EpistemicState.INFERRED
            or not isinstance(assertion.lineage, DerivedLineage)
        ):
            raise AssertionStoreError(
                "inference publication requires an active inferred assertion with derived lineage"
            )
        operation_id, request = self._validate_initial_assertion_write(
            assertion,
            source_occurrences=(),
            operation_id=operation_id,
            expected_generation=None,
        )
        async with self._mutation():
            written, _ = await self._put_initial_assertion_in_mutation(
                assertion,
                source_occurrences=(),
                operation_id=operation_id,
                request=request,
                expected_generation=None,
            )
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
        # Validation reports are durable semantic state.  Reuse the canonical
        # mutation boundary so a maintenance-scoped revalidation verifies and
        # renews its fence in the very transaction that writes the report.
        async with self._mutation():
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

    async def persist_validation_report_and_quarantine(
        self,
        report: ShaclValidationReport,
        *,
        expected_revisions: Mapping[str, str],
    ) -> ValidationQuarantineBatchResult:
        """Persist one quarantine report and repair every target atomically.

        The snapshot revision map is a compare-and-swap fence.  This one
        assertion-authority transaction commits the report, every lifecycle
        transition, eligibility invalidation, generation update, and emitted
        outbox event together; a stale target or any write failure rolls all of
        them back rather than leaving an authoritative report beside active
        invalid data.
        """
        self._validate_validation_report(report)
        if report.action is not ValidationWriteAction.QUARANTINE:
            raise AssertionStoreError(
                "atomic validation repair requires a quarantine report"
            )
        targets = _validation_report_targets(report)
        if not isinstance(expected_revisions, Mapping):
            raise AssertionStoreError("validation repair requires target revision mappings")
        normalized_revisions: dict[str, str] = {}
        for assertion_id, revision_id in expected_revisions.items():
            if not isinstance(assertion_id, str) or not assertion_id:
                raise AssertionStoreError("validation repair assertion ids must be non-empty strings")
            if not isinstance(revision_id, str) or not revision_id:
                raise AssertionStoreError("validation repair revision ids must be non-empty strings")
            normalized_revisions[assertion_id] = revision_id
        if not set(targets).issubset(normalized_revisions):
            raise AssertionConflictError(
                "validation snapshot is missing a report-affected assertion revision"
            )
        operation_id = f"validation-report-quarantine:{report.report_id}"
        request = {
            "report_id": report.report_id,
            "targets": [[assertion_id, normalized_revisions[assertion_id]] for assertion_id in targets],
        }
        async with self._mutation():
            digest, replay = await self._operation(
                operation_id,
                "validation-report-quarantine",
                request,
            )
            if replay is not None:
                await self._validation_report_from_receipt(replay)
                quarantined = tuple(
                    item
                    for item in (
                        await self._revision(str(revision_id))
                        for revision_id in replay["quarantined_revision_ids"]
                    )
                    if item is not None
                )
                invalidated = tuple(
                    item
                    for item in (
                        await self._revision(str(revision_id))
                        for revision_id in replay["invalidation_state_revision_ids"]
                    )
                    if item is not None
                )
                if (
                    len(quarantined) != len(replay["quarantined_revision_ids"])
                    or len(invalidated)
                    != len(replay["invalidation_state_revision_ids"])
                ):
                    raise AssertionConflictError(
                        "idempotent validation repair receipt lost its lifecycle revisions"
                    )
                return ValidationQuarantineBatchResult(
                    quarantined,
                    invalidated,
                    tuple(str(item) for item in replay["invalidated_revision_ids"]),
                    int(replay["generation"]),
                    True,
                )

            current_by_assertion_id: dict[str, Assertion] = {}
            for assertion_id in targets:
                current = await self._current(assertion_id)
                if (
                    current is None
                    or current.revision_id != normalized_revisions[assertion_id]
                    or current.status is not AssertionStatus.ACTIVE
                ):
                    raise AssertionConflictError(
                        "validation target is no longer the expected active tenant assertion"
                    )
                current_by_assertion_id[assertion_id] = current

            # This is intentionally inside the same transaction as the CAS
            # fence and transitions.  A report never outlives a failed repair.
            await self._persist_validation_report_in_transaction(report)
            roots = tuple(current_by_assertion_id[assertion_id] for assertion_id in targets)
            root_revision_ids = {item.revision_id for item in roots}
            dependents = [
                item
                for item in await self._dependent_current_revisions(root_revision_ids)
                if item.status is AssertionStatus.ACTIVE
                and item.revision_id not in root_revision_ids
            ]
            invalidated_revision_ids = [*sorted(root_revision_ids)]
            quarantined: list[Assertion] = []
            for current in roots:
                await self._invalidate_eligibility(current.revision_id)
                state = _assertion_with(
                    current,
                    revision_id=uuid4().hex,
                    status=AssertionStatus.QUARANTINED,
                    supersedes_revision_id=None,
                )
                await self._write_revision(state, ())
                await self._set_current(state)
                quarantined.append(state)

            invalidated: list[Assertion] = []
            for dependent in dependents:
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

            # A no-target report has no canonical data to invalidate.  Retain
            # it atomically with its tenant serialization but do not fabricate
            # a generation/event for a graph that did not change.
            generation = await self._advance_generation() if roots else await self._generation()
            quarantine_events = [
                await self._event(item, "validation_quarantined", generation)
                for item in quarantined
            ]
            dependent_events = [
                await self._event(item, "retracted", generation) for item in invalidated
            ]
            receipt = {
                "validation_report_id": report.report_id,
                "quarantined_revision_ids": [item.revision_id for item in quarantined],
                "invalidation_state_revision_ids": [item.revision_id for item in invalidated],
                "invalidated_revision_ids": invalidated_revision_ids,
                "generation": generation,
                "event_ids": [*quarantine_events, *dependent_events],
            }
            await self._record_operation(
                operation_id,
                "validation-report-quarantine",
                digest,
                receipt,
                [item.assertion_id for item in (*quarantined, *invalidated)],
                [item.revision_id for item in (*quarantined, *invalidated)],
            )
            return ValidationQuarantineBatchResult(
                tuple(quarantined),
                tuple(invalidated),
                tuple(invalidated_revision_ids),
                generation,
            )

    async def supersede(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
        _migration_capability: _RawAssertionMutationCapability | None = None,
    ) -> SupersessionResult:
        """Private migration-only raw canonical supersession."""
        self._require_raw_mutation_capability(_migration_capability)
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

    async def restore_with_validation_report(
        self,
        expected_terminal_revision_id: str,
        replacement: Assertion,
        report: ShaclValidationReport,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> SupersessionResult:
        """Atomically restore a terminal direct shell with fresh evidence."""
        operation_id, request = self._validate_restoration(
            expected_terminal_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
            expected_generation=expected_generation,
        )
        self._validate_accepted_validation_report(replacement, report)
        async with self._mutation():
            result, replay = await self._restore_in_mutation(
                expected_terminal_revision_id,
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

    def _validate_restoration(
        self,
        expected_terminal_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence],
        operation_id: str | None,
        expected_generation: int | None,
    ) -> tuple[str, dict[str, object]]:
        if not isinstance(replacement, Assertion):
            raise AssertionStoreError(
                "restore requires a canonical Assertion replacement"
            )
        self._check_assertion_scope(replacement)
        if (
            replacement.status is not AssertionStatus.ACTIVE
            or replacement.supersedes_revision_id is not None
            or not isinstance(replacement.lineage, DirectLineage)
        ):
            raise AssertionStoreError(
                "restoration requires an initial-form active direct replacement"
            )
        if (
            len(source_occurrences) != 1
            or not isinstance(source_occurrences[0], SourceOccurrence)
            or replacement.lineage.source_occurrence_ids
            != (source_occurrences[0].source_occurrence_id,)
            or replacement.revision_id
            != source_occurrences[0].source_occurrence_id
        ):
            raise AssertionStoreError(
                "restoration requires exactly one fresh attached source revision"
            )
        if expected_generation is not None and (
            type(expected_generation) is not int or expected_generation < 0
        ):
            raise AssertionStoreError(
                "expected_generation must be a non-negative integer"
            )
        resolved_operation_id = operation_id or (
            f"restore:{expected_terminal_revision_id}:"
            f"{replacement.revision_id}"
        )
        source = source_occurrences[0]
        adapter_digest = resolved_operation_id.removeprefix(
            _EXPLICIT_FACT_SAVE_OPERATION_PREFIX
        )
        if (
            not resolved_operation_id.startswith(
                _EXPLICIT_FACT_SAVE_OPERATION_PREFIX
            )
            or replacement.confidence_method
            != "memory-agency-save-fact-v1"
            or replacement.confidence_basis != "explicit-tool-invocation"
            or source.source_occurrence_id
            != f"source:memory-agency-save-fact-v1:{adapter_digest}"
            or source.content_digest != f"sha256:{adapter_digest}"
        ):
            raise AssertionStoreError(
                "restoration is restricted to an exact save_fact proposal"
            )
        request = {
            "expected": expected_terminal_revision_id,
            "replacement": replacement.to_mapping(),
            "sources": [s.to_mapping() for s in source_occurrences],
        }
        return resolved_operation_id, request

    async def replay_governed_restoration(
        self,
        expected_terminal_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence] = (),
        operation_id: str | None = None,
    ) -> tuple[ShaclValidationReport, SupersessionResult] | None:
        """Return the exact accepted report and restoration receipt."""
        resolved_operation_id, request = self._validate_restoration(
            expected_terminal_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
            expected_generation=None,
        )
        _, replay = await self._operation(
            resolved_operation_id,
            "restore",
            request,
        )
        if replay is None:
            return None
        report = await self._validation_report_from_receipt(replay)
        predecessor = await self._revision(
            str(replay["predecessor_revision_id"])
        )
        applied = await self._revision(str(replay["replacement_revision_id"]))
        if predecessor is None or applied is None:
            raise AssertionConflictError(
                "idempotent restoration receipt no longer has its revisions"
            )
        self._validate_accepted_validation_report(applied, report)
        return report, SupersessionResult(
            predecessor,
            applied,
            int(replay["generation"]),
            tuple(replay["event_ids"]),
            (),
            True,
        )

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

    async def _assert_governed_replay_binding(
        self,
        operation: str,
        assertion: Assertion,
        predecessor: Assertion | None,
        binding: _GovernedAssertionReplayBinding,
    ) -> None:
        """Authenticate a live governed receipt against its adapter proposal."""
        _validate_governed_replay_binding(binding)
        if (
            assertion.assertion_id != binding.assertion_id
            or assertion.revision_id != binding.revision_id
            or not self._governed_replay_result_structure_matches(
                operation,
                assertion,
                predecessor,
                binding.source_occurrence_id,
            )
        ):
            raise AssertionConflictError(
                "operation_id resolves to a different governed assertion result"
            )
        row = await self._database.fetchone(
            "SELECT s.source_mapping "
            "FROM semantic_revision_sources rs "
            "JOIN semantic_source_occurrences s "
            "  ON s.tenant_id = rs.tenant_id "
            " AND s.source_occurrence_id = rs.source_occurrence_id "
            "WHERE rs.tenant_id = ? AND rs.revision_id = ? "
            "AND rs.source_occurrence_id = ?",
            (
                self.tenant_id,
                binding.revision_id,
                binding.source_occurrence_id,
            ),
        )
        if row is None:
            raise AssertionConflictError(
                "governed assertion receipt lacks its expected attached source"
            )
        try:
            source = SourceOccurrence.from_mapping(json.loads(row[0]))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise AssertionConflictError(
                "governed assertion receipt has malformed attached provenance"
            ) from error
        if (
            source.source_occurrence_id != binding.source_occurrence_id
            or source.content_digest
            != f"sha256:{binding.adapter_request_digest}"
            or assertion.asserted_at != source.received_at
        ):
            raise AssertionConflictError(
                "operation_id resolves to different governed assertion provenance"
            )
        actual_binding = _governed_assertion_replay_binding(
            assertion,
            source,
            binding.adapter_request_digest,
        )
        if (
            actual_binding.proposal_fingerprint
            != binding.proposal_fingerprint
        ):
            raise AssertionConflictError(
                "operation_id resolves to different governed assertion metadata"
            )

    @staticmethod
    def _governed_replay_result_structure_matches(
        operation: str,
        assertion: Assertion,
        predecessor: Assertion | None,
        expected_source_occurrence_id: str,
    ) -> bool:
        """Check the exact direct-lineage lifecycle shape selected by a receipt."""
        if (
            assertion.status is not AssertionStatus.ACTIVE
            or not isinstance(assertion.lineage, DirectLineage)
        ):
            return False
        source_ids = assertion.lineage.source_occurrence_ids
        if operation == "put":
            return (
                predecessor is None
                and assertion.supersedes_revision_id is None
                and source_ids == (expected_source_occurrence_id,)
            )
        if (
            operation == "restore"
            and predecessor is not None
            and predecessor.status is AssertionStatus.DELETED
            and predecessor.assertion_id == assertion.assertion_id
        ):
            return (
                assertion.supersedes_revision_id == predecessor.revision_id
                and source_ids == (expected_source_occurrence_id,)
            )
        if (
            operation != "supersede"
            or predecessor is None
            or predecessor.status is not AssertionStatus.SUPERSEDED
            or not isinstance(predecessor.lineage, DirectLineage)
            or assertion.supersedes_revision_id != predecessor.revision_id
        ):
            return False
        if predecessor.assertion_id == assertion.assertion_id:
            return source_ids == (
                *predecessor.lineage.source_occurrence_ids,
                expected_source_occurrence_id,
            )
        return source_ids == (expected_source_occurrence_id,)

    async def replay_governed_assertion_operation(
        self,
        operation_id: str,
        binding: _GovernedAssertionReplayBinding,
    ) -> (
        GovernedAssertionOperationReplay
        | ErasedGovernedAssertionOperationReplay
        | None
    ):
        """Return one exact governed-write receipt or erased disposition.

        The normal retry APIs accept a full request and verify its digest.  An
        explicit-fact adapter cannot safely rebuild that request after later
        canonical transitions, however: the later state may contain extra
        source occurrences or a newer replacement.  This narrow read surface
        resolves only a previously committed, report-backed governed write by
        its stable operation ID.  After physical erasure it resolves the
        blinded tombstone instead, returning no assertion, revision, source,
        report, or content.  It cannot enumerate, publish, or mutate anything.
        """
        _validate_governed_replay_binding(binding)
        recorded = await self._recorded_operation(operation_id)
        if recorded is None:
            return await self._erased_governed_operation_replay(
                operation_id,
                binding,
            )
        operation, receipt = recorded
        try:
            if operation == "put":
                revision_id = receipt.get("revision_id")
                if not isinstance(revision_id, str) or not revision_id:
                    raise AssertionConflictError("governed put receipt is malformed")
                assertion = await self._revision(revision_id)
                if assertion is None:
                    raise AssertionConflictError(
                        "idempotent assertion receipt no longer has a revision"
                    )
                report = await self._validation_report_from_receipt(receipt)
                self._validate_accepted_validation_report(assertion, report)
                await self._assert_governed_replay_binding(
                    operation,
                    assertion,
                    None,
                    binding,
                )
                generation = receipt.get("generation")
                if type(generation) is not int:
                    raise AssertionConflictError("governed put receipt is malformed")
                return GovernedAssertionOperationReplay(
                    operation=operation,
                    report=report,
                    assertion=assertion,
                    predecessor=None,
                    generation=generation,
                )
            if operation in {"supersede", "restore"}:
                predecessor_id = receipt.get("predecessor_revision_id")
                replacement_id = receipt.get("replacement_revision_id")
                if not (
                    isinstance(predecessor_id, str)
                    and predecessor_id
                    and isinstance(replacement_id, str)
                    and replacement_id
                ):
                    raise AssertionConflictError(
                        "governed supersession receipt is malformed"
                    )
                predecessor = await self._revision(predecessor_id)
                assertion = await self._revision(replacement_id)
                if predecessor is None or assertion is None:
                    raise AssertionConflictError(
                        "idempotent supersession receipt no longer has its revisions"
                    )
                report = await self._validation_report_from_receipt(receipt)
                self._validate_accepted_validation_report(assertion, report)
                await self._assert_governed_replay_binding(
                    operation,
                    assertion,
                    predecessor,
                    binding,
                )
                generation = receipt.get("generation")
                if type(generation) is not int:
                    raise AssertionConflictError(
                        "governed supersession receipt is malformed"
                    )
                return GovernedAssertionOperationReplay(
                    operation=operation,
                    report=report,
                    assertion=assertion,
                    predecessor=predecessor,
                    generation=generation,
                )
            raise AssertionConflictError(
                "operation_id resolves to a non-governed assertion write"
            )
        except AssertionConflictError:
            # Physical erasure can commit between the operation-row read and
            # its revision/report reads.  Prefer the now-durable terminal
            # tombstone over leaking that implementation race as a transient
            # conflict.  Without a matching tombstone, preserve the original
            # corruption/conflict signal.
            erased = await self._erased_governed_operation_replay(
                operation_id,
                binding,
            )
            if erased is not None:
                return erased
            raise

    async def validate_source_append(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence],
    ) -> None:
        """Prove that a replacement is only a direct-provenance append.

        A provenance encounter is itself a canonical revision, but it must not
        become a back door for changing claim terms or confidence.  The
        governed validation service calls this before it routes the append
        through the normal validated supersession lifecycle.  Replays bypass
        this preflight only after the operation ledger has supplied the exact
        original receipt.
        """
        if not isinstance(replacement, Assertion):
            raise AssertionStoreError(
                "source append requires a canonical Assertion replacement"
            )
        self._check_assertion_scope(replacement)
        if len(source_occurrences) != 1 or not isinstance(
            source_occurrences[0], SourceOccurrence
        ):
            raise AssertionStoreError(
                "source append requires exactly one SourceOccurrence"
            )
        predecessor = await self._revision(expected_predecessor_revision_id)
        if predecessor is None or predecessor.status is not AssertionStatus.ACTIVE:
            raise AssertionConflictError(
                "source append requires an active predecessor in the bound tenant"
            )
        current = await self._current(predecessor.assertion_id)
        if current is None or current.revision_id != expected_predecessor_revision_id:
            raise AssertionConflictError(
                "source append predecessor is no longer the current tenant revision"
            )
        if not isinstance(predecessor.lineage, DirectLineage) or not isinstance(
            replacement.lineage, DirectLineage
        ):
            raise AssertionStoreError(
                "source append is available only for direct assertions"
            )
        source = source_occurrences[0]
        previous_source_ids = predecessor.lineage.source_occurrence_ids
        if source.source_occurrence_id in previous_source_ids:
            raise AssertionStoreError(
                "source append requires a new source occurrence"
            )
        if replacement.lineage.source_occurrence_ids != (
            *previous_source_ids,
            source.source_occurrence_id,
        ):
            raise AssertionStoreError(
                "source append replacement must preserve prior direct provenance "
                "and append exactly the supplied source occurrence"
            )
        if replacement.asserted_at != source.received_at:
            raise AssertionStoreError(
                "source append assertion time must match the new source encounter"
            )
        predecessor_mapping = predecessor.to_mapping()
        replacement_mapping = replacement.to_mapping()
        for field in (
            "revision_id",
            "status",
            "supersedes_revision_id",
            "lineage",
            "asserted_at",
        ):
            predecessor_mapping.pop(field, None)
            replacement_mapping.pop(field, None)
        if predecessor_mapping != replacement_mapping:
            raise AssertionStoreError(
                "source append cannot alter canonical claim or epistemic metadata"
            )

    async def plan_supersession_lifecycle(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
    ) -> SupersessionLifecyclePlan:
        """Compute the exact active graph for one prospective supersession.

        This is a read-only peer primitive for the governed validation service
        and ``_supersede_in_mutation``.  It must remain the sole place that
        decides whether an inferred conclusion loses all of its proofs: the
        canonical lineage can be withdrawn while another active ledger proof
        keeps the conclusion live.
        """
        self._check_assertion_scope(replacement)
        predecessor = await self._revision(expected_predecessor_revision_id)
        if predecessor is None or predecessor.status is not AssertionStatus.ACTIVE:
            raise AssertionConflictError("expected predecessor is not an active tenant revision")
        current = await self._current(predecessor.assertion_id)
        if current is None or current.revision_id != expected_predecessor_revision_id:
            raise AssertionConflictError("expected predecessor is no longer the current revision")
        if replacement.supersedes_revision_id not in (None, expected_predecessor_revision_id):
            raise AssertionStoreError("replacement cannot name an unrelated superseded revision")

        dependents, deactivated_derivation_ids = (
            await self._plan_dependent_current_revisions(
                (expected_predecessor_revision_id,)
            )
        )
        withdrawn_revision_ids = (
            expected_predecessor_revision_id,
            *(assertion.revision_id for assertion in dependents),
        )
        withdrawn = set(withdrawn_revision_ids)
        current_assertions = await self._complete_active_assertions()
        if expected_predecessor_revision_id not in {
            assertion.revision_id for assertion in current_assertions
        }:
            raise AssertionConflictError("expected predecessor is no longer the current revision")
        post_state = tuple(
            assertion
            for assertion in current_assertions
            if assertion.revision_id not in withdrawn
        ) + (replacement,)
        return SupersessionLifecyclePlan(
            predecessor=predecessor,
            dependent_assertions=tuple(dependents),
            withdrawn_revision_ids=tuple(withdrawn_revision_ids),
            deactivated_derivation_ids=tuple(deactivated_derivation_ids),
            post_state=post_state,
        )

    async def plan_restoration_lifecycle(
        self,
        expected_terminal_revision_id: str,
        replacement: Assertion,
    ) -> RestorationLifecyclePlan:
        """Plan a fresh direct revision over an immutable terminal shell."""
        self._check_assertion_scope(replacement)
        predecessor = await self._revision(expected_terminal_revision_id)
        if (
            predecessor is None
            or predecessor.status is not AssertionStatus.DELETED
            or not isinstance(predecessor.lineage, DirectLineage)
            or not isinstance(replacement.lineage, DirectLineage)
            or predecessor.assertion_id != replacement.assertion_id
        ):
            raise AssertionConflictError(
                "expected predecessor is not a restorable deleted direct revision"
            )
        current = await self._current(predecessor.assertion_id)
        if (
            current is None
            or current.revision_id != expected_terminal_revision_id
        ):
            raise AssertionConflictError(
                "expected terminal predecessor is no longer current"
            )
        if replacement.supersedes_revision_id not in (
            None,
            expected_terminal_revision_id,
        ):
            raise AssertionStoreError(
                "restoration cannot name an unrelated superseded revision"
            )
        predecessor_mapping = predecessor.to_mapping()
        replacement_mapping = replacement.to_mapping()
        for field in (
            "revision_id",
            "status",
            "supersedes_revision_id",
            "lineage",
            "asserted_at",
            "epistemic_state",
        ):
            predecessor_mapping.pop(field, None)
            replacement_mapping.pop(field, None)
        if predecessor_mapping != replacement_mapping:
            raise AssertionStoreError(
                "restoration cannot alter canonical claim or policy metadata"
            )
        current_assertions = await self._complete_active_assertions()
        return RestorationLifecyclePlan(
            predecessor=predecessor,
            post_state=(*current_assertions, replacement),
        )

    async def _restore_in_mutation(
        self,
        expected_terminal_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences: Sequence[SourceOccurrence],
        operation_id: str,
        request: dict[str, object],
        expected_generation: int | None,
        validation_report_id: str | None = None,
    ) -> tuple[SupersessionResult, dict[str, object] | None]:
        """Apply one pre-validated terminal-shell restoration."""
        digest, replay = await self._operation(
            operation_id,
            "restore",
            request,
        )
        if replay is not None:
            predecessor = await self._revision(
                str(replay["predecessor_revision_id"])
            )
            applied = await self._revision(
                str(replay["replacement_revision_id"])
            )
            if predecessor is None or applied is None:
                raise AssertionConflictError(
                    "idempotent restoration receipt no longer has its revisions"
                )
            return (
                SupersessionResult(
                    predecessor,
                    applied,
                    int(replay["generation"]),
                    tuple(replay["event_ids"]),
                    (),
                    True,
                ),
                replay,
            )
        if (
            expected_generation is not None
            and await self._generation() != expected_generation
        ):
            raise AssertionConflictError(
                "validation generation is no longer current; "
                "revalidate before restoring"
            )
        plan = await self.plan_restoration_lifecycle(
            expected_terminal_revision_id,
            replacement,
        )
        prior_source = await self.get_source_occurrence(
            source_occurrences[0].source_occurrence_id
        )
        if prior_source is not None:
            raise AssertionConflictError(
                "restoration source occurrence was already accepted"
            )
        await self._store_sources(source_occurrences)
        await self._validate_lineage(replacement, source_occurrences)
        replacement_mapping = replacement.to_mapping()
        replacement_mapping["supersedes_revision_id"] = (
            expected_terminal_revision_id
        )
        replacement_state = Assertion.from_mapping(replacement_mapping)
        await self._write_revision(replacement_state, source_occurrences)
        await self._set_current(replacement_state)
        generation = await self._advance_generation()
        event_id = await self._event(
            replacement_state,
            "accepted",
            generation,
        )
        receipt = {
            "predecessor_revision_id": plan.predecessor.revision_id,
            "replacement_revision_id": replacement_state.revision_id,
            "generation": generation,
            "event_ids": [event_id],
            "invalidated_revision_ids": [],
        }
        if validation_report_id is not None:
            receipt["validation_report_id"] = validation_report_id
        await self._record_operation(
            operation_id,
            "restore",
            digest,
            receipt,
            [replacement_state.assertion_id],
            [plan.predecessor.revision_id, replacement_state.revision_id],
        )
        return (
            SupersessionResult(
                plan.predecessor,
                replacement_state,
                generation,
                (event_id,),
                (),
            ),
            None,
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
        await self._store_sources(source_occurrences)
        await self._validate_lineage(replacement, source_occurrences)
        plan = await self.plan_supersession_lifecycle(
            expected_predecessor_revision_id,
            replacement,
        )
        predecessor = plan.predecessor
        predecessor_state = _assertion_with(
            predecessor, revision_id=uuid4().hex, status=AssertionStatus.SUPERSEDED,
            supersedes_revision_id=expected_predecessor_revision_id,
        )
        replacement_mapping = replacement.to_mapping()
        replacement_mapping["supersedes_revision_id"] = (
            predecessor_state.revision_id
        )
        replacement_state = Assertion.from_mapping(replacement_mapping)
        # Apply the same plan used to construct the governed tentative graph.
        # In particular, only ledger proofs selected by the planner deactivate;
        # an alternate proof keeps its conclusion in the active state.
        await self._deactivate_inference_derivation_ids(
            plan.deactivated_derivation_ids
        )
        dependent_states: list[Assertion] = []
        invalidated_revision_ids = list(plan.withdrawn_revision_ids)
        for dependent in plan.dependent_assertions:
            if dependent.status is not AssertionStatus.ACTIVE:
                continue
            await self._invalidate_eligibility(
                dependent.revision_id,
                deactivate_inference_derivations=False,
            )
            state = _assertion_with(
                dependent, revision_id=uuid4().hex, status=AssertionStatus.RETRACTED,
                supersedes_revision_id=None, epistemic_state=EpistemicState.RETRACTED,
            )
            await self._write_revision(state, ())
            await self._set_current(state)
            dependent_states.append(state)
        await self._invalidate_eligibility(
            expected_predecessor_revision_id,
            deactivate_inference_derivations=False,
        )
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

    async def _deactivate_inference_derivations_for_inputs(
        self,
        input_revision_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Deactivate every active proof that names a withdrawn support.

        Canonical derivation inputs drive the lifecycle cascade, while the
        inference ledger records all alternate proofs.  Both need the same
        invalidation point: otherwise ``explain()`` can expose a proof whose
        premise has already been retracted, superseded, deleted, or rejected
        by validation.  The caller is always inside the tenant mutation
        transaction, so the conclusion and its active proof set change
        atomically.
        """
        tenant_id, _ = self._require_scope()
        revision_ids = tuple(sorted(set(input_revision_ids)))
        if not revision_ids:
            return ()
        rows = await self._database.fetchall(
            "SELECT DISTINCT d.derivation_id "
            "FROM semantic_inference_derivations d "
            "JOIN semantic_inference_derivation_inputs i "
            "  ON i.tenant_id = d.tenant_id AND i.derivation_id = d.derivation_id "
            "WHERE d.tenant_id = ? AND d.active = 1 "
            f"AND i.input_revision_id IN ({_placeholders(revision_ids)}) "
            "ORDER BY d.derivation_id ASC",
            (tenant_id,) + revision_ids,
        )
        derivation_ids = tuple(str(row[0]) for row in rows)
        await self._deactivate_inference_derivation_ids(derivation_ids)
        return derivation_ids

    async def _deactivate_inference_derivation_ids(
        self,
        derivation_ids: Iterable[str],
    ) -> None:
        """Deactivate an already-planned set of tenant-local ledger proofs."""
        resolved = tuple(sorted(set(derivation_ids)))
        if not resolved:
            return
        tenant_id, _ = self._require_scope()
        await self._database.execute(
            "UPDATE semantic_inference_derivations SET active = 0 "
            "WHERE tenant_id = ? AND active = 1 "
            f"AND derivation_id IN ({_placeholders(resolved)})",
            (tenant_id,) + resolved,
        )

    async def _plan_dependent_current_revisions(
        self,
        input_revision_ids: Iterable[str],
    ) -> tuple[list[Assertion], tuple[str, ...]]:
        """Find the grounded surviving inference closure after a withdrawal.

        Ledger proof rows are not independently sufficient merely because all
        of their immediate inputs are still current.  In particular, two
        inferred conclusions can point at each other after their only direct
        support is withdrawn.  Treating either side of that SCC as surviving
        would retain a stale closure indefinitely.

        The lifecycle graph therefore uses a least fixed point: seed it with
        current, eligible direct facts, then admit an inferred current revision
        only when one active derivation has *every* premise already grounded.
        The prospective supersession planner and every live lifecycle cascade
        share this method, keeping validation and mutation semantics identical.
        """
        tenant_id, _ = self._require_scope()
        withdrawn = set(input_revision_ids)
        rows = await self._database.fetchall(
            "SELECT r.revision_id, r.assertion_mapping "
            "FROM semantic_assertions a "
            "JOIN semantic_assertion_revisions r ON r.tenant_id = a.tenant_id "
            "  AND r.revision_id = a.current_revision_id "
            "JOIN semantic_projection_eligibility e ON e.tenant_id = r.tenant_id "
            "  AND e.revision_id = r.revision_id "
            "WHERE a.tenant_id = ? AND r.status = ? "
            "  AND r.eligible = 1 AND e.eligible = 1 "
            "ORDER BY r.revision_id ASC",
            (tenant_id, AssertionStatus.ACTIVE.value),
        )
        current = {
            str(revision_id): Assertion.from_mapping(json.loads(mapping))
            for revision_id, mapping in rows
        }
        inferred = {
            revision_id: assertion
            for revision_id, assertion in current.items()
            if isinstance(assertion.lineage, DerivedLineage)
        }

        # The canonical lineage is a compatibility fallback only for inferred
        # revisions that predate the inference ledger.  Once a revision has
        # ledger history, an inactive proof is an intentional withdrawal (for
        # example, a disabled profile), not permission to revive its canonical
        # lineage as a synthetic proof.
        ledger_history_rows = await self._database.fetchall(
            "SELECT DISTINCT derived_revision_id "
            "FROM semantic_inference_derivations WHERE tenant_id = ?",
            (tenant_id,),
        )
        revisions_with_ledger_history = {
            str(row[0]) for row in ledger_history_rows
        }
        derivation_rows = await self._database.fetchall(
            "SELECT d.derivation_id, d.derived_revision_id, i.input_revision_id "
            "FROM semantic_inference_derivations d "
            "LEFT JOIN semantic_inference_derivation_inputs i "
            "  ON i.tenant_id = d.tenant_id AND i.derivation_id = d.derivation_id "
            "WHERE d.tenant_id = ? AND d.active = 1 "
            "ORDER BY d.derivation_id ASC, i.ordinal ASC",
            (tenant_id,),
        )
        active_derivations: dict[str, tuple[str, tuple[str, ...]]] = {}
        for derivation_id, derived_revision_id, input_revision_id in derivation_rows:
            resolved_id = str(derivation_id)
            derived_id = str(derived_revision_id)
            existing = active_derivations.get(resolved_id)
            inputs = () if existing is None else existing[1]
            if input_revision_id is not None:
                inputs += (str(input_revision_id),)
            active_derivations[resolved_id] = (derived_id, inputs)

        grounded = {
            revision_id
            for revision_id, assertion in current.items()
            if revision_id not in withdrawn
            and not isinstance(assertion.lineage, DerivedLineage)
        }
        grounded_derivations: set[str] = set()
        changed = True
        while changed:
            changed = False
            for derivation_id in sorted(active_derivations):
                derived_revision_id, premises = active_derivations[derivation_id]
                if (
                    derived_revision_id in withdrawn
                    or derived_revision_id not in inferred
                    or not premises
                    or not set(premises).issubset(grounded)
                ):
                    continue
                if derivation_id not in grounded_derivations:
                    grounded_derivations.add(derivation_id)
                    changed = True
                if derived_revision_id not in grounded:
                    grounded.add(derived_revision_id)
                    changed = True
            for revision_id in sorted(inferred):
                assertion = inferred[revision_id]
                if (
                    revision_id in withdrawn
                    or revision_id in revisions_with_ledger_history
                    or not assertion.lineage.input_revision_ids
                    or not set(assertion.lineage.input_revision_ids).issubset(grounded)
                ):
                    continue
                if revision_id not in grounded:
                    grounded.add(revision_id)
                    changed = True

        # Lifecycle receipts and outbox events expose this sequence, so use the
        # immutable assertion identity rather than an incidental revision or
        # derivation traversal order.
        dependent_assertions = sorted(
            (
                assertion
                for revision_id, assertion in inferred.items()
                if revision_id not in withdrawn and revision_id not in grounded
            ),
            key=lambda assertion: assertion.assertion_id,
        )
        return (
            dependent_assertions,
            tuple(sorted(set(active_derivations).difference(grounded_derivations))),
        )

    async def _dependent_current_revisions(self, input_revision_ids: Iterable[str]) -> list[Assertion]:
        """Apply the shared grounded lifecycle plan to a live mutation."""
        dependents, deactivated_derivation_ids = (
            await self._plan_dependent_current_revisions(input_revision_ids)
        )
        await self._deactivate_inference_derivation_ids(deactivated_derivation_ids)
        return dependents

    async def revoke_semantic_inference(
        self,
        engine_version: str,
    ) -> InferenceRevocationResult:
        """Retract this tenant's current materializations for a disabled engine.

        A configured ``enabled = false`` is an operator decision, not an
        instruction to merely skip the next maintenance pass.  The canonical
        inferred revisions and every active ledger entry that supported them
        must stop participating in reads before sleep reports the profile as
        disabled.  Derived dependents outside the selected engine still use
        normal alternate-proof handling, while the disabled engine itself is
        always revoked even when it has another valid semantic proof.
        """
        if not isinstance(engine_version, str) or not engine_version:
            raise AssertionStoreError("inference engine_version must be non-empty")
        async with self._mutation():
            tenant_id, _ = self._require_scope()
            rows = await self._database.fetchall(
                "SELECT r.assertion_mapping FROM semantic_assertions a "
                "JOIN semantic_assertion_revisions r ON r.tenant_id = a.tenant_id "
                "  AND r.revision_id = a.current_revision_id "
                "WHERE a.tenant_id = ? AND r.status = ? AND r.epistemic_state = ? "
                "ORDER BY r.assertion_id ASC",
                (
                    tenant_id,
                    AssertionStatus.ACTIVE.value,
                    EpistemicState.INFERRED.value,
                ),
            )
            selected = [
                Assertion.from_mapping(json.loads(encoded))
                for (encoded,) in rows
            ]
            selected = [
                assertion for assertion in selected
                if isinstance(assertion.lineage, DerivedLineage)
                and assertion.lineage.engine_version == engine_version
            ]
            if not selected:
                return InferenceRevocationResult(0, 0, await self._generation())

            selected_revision_ids = [item.revision_id for item in selected]
            ledger_rows = await self._database.fetchall(
                "SELECT derivation_id FROM semantic_inference_derivations "
                "WHERE tenant_id = ? AND active = 1 "
                f"AND derived_revision_id IN ({_placeholders(selected_revision_ids)})",
                (tenant_id,) + tuple(selected_revision_ids),
            )
            disabled_derivation_ids = tuple(str(row[0]) for row in ledger_rows)
            if disabled_derivation_ids:
                await self._database.execute(
                    "UPDATE semantic_inference_derivations SET active = 0 "
                    "WHERE tenant_id = ? "
                    f"AND derivation_id IN ({_placeholders(disabled_derivation_ids)})",
                    (tenant_id,) + disabled_derivation_ids,
                )

            # Once the selected engine's ledgers are inactive, non-selected
            # derived assertions can still retain an independent proof but
            # cannot keep a conclusion alive solely through disabled rules.
            dependents = await self._dependent_current_revisions(
                selected_revision_ids
            )
            by_revision = {item.revision_id: item for item in selected}
            for dependent in dependents:
                by_revision.setdefault(dependent.revision_id, dependent)
            to_retract = [
                by_revision[revision_id]
                for revision_id in sorted(by_revision)
                if by_revision[revision_id].status is AssertionStatus.ACTIVE
            ]
            old_revision_ids = [item.revision_id for item in to_retract]
            all_ledger_rows = await self._database.fetchall(
                "SELECT derivation_id FROM semantic_inference_derivations "
                "WHERE tenant_id = ? AND active = 1 "
                f"AND derived_revision_id IN ({_placeholders(old_revision_ids)})",
                (tenant_id,) + tuple(old_revision_ids),
            )
            all_deactivated_ids = tuple(
                sorted(
                    set(disabled_derivation_ids).union(
                        str(row[0]) for row in all_ledger_rows
                    )
                )
            )
            if all_deactivated_ids:
                await self._database.execute(
                    "UPDATE semantic_inference_derivations SET active = 0 "
                    "WHERE tenant_id = ? "
                    f"AND derivation_id IN ({_placeholders(all_deactivated_ids)})",
                    (tenant_id,) + all_deactivated_ids,
                )

            retracted: list[Assertion] = []
            for assertion in to_retract:
                await self._invalidate_eligibility(assertion.revision_id)
                state = _assertion_with(
                    assertion,
                    revision_id=uuid4().hex,
                    status=AssertionStatus.RETRACTED,
                    supersedes_revision_id=None,
                    epistemic_state=EpistemicState.RETRACTED,
                )
                await self._write_revision(state, ())
                await self._set_current(state)
                retracted.append(state)
            generation = await self._advance_generation()
            for assertion in retracted:
                await self._event(assertion, "retracted", generation)
            return InferenceRevocationResult(
                len(retracted), len(all_deactivated_ids), generation
            )

    async def reactivate_inferred(
        self,
        assertion: Assertion,
        *,
        operation_id: str | None = None,
    ) -> AssertionWriteResult:
        """Make a previously inactive inferred identity current again.

        A canonical assertion ID describes the claim, not a particular
        derivation.  Re-materialization after a premise or profile change must
        therefore create a fresh immutable inferred revision instead of trying
        to mutate an old retracted revision or creating a second fact store.
        Direct facts are never replaced by this inference-only operation.
        """
        if not isinstance(assertion, Assertion):
            raise AssertionStoreError("reactivate_inferred requires a canonical Assertion")
        self._check_assertion_scope(assertion)
        if (
            assertion.status is not AssertionStatus.ACTIVE
            or assertion.epistemic_state is not EpistemicState.INFERRED
            or not isinstance(assertion.lineage, DerivedLineage)
        ):
            raise AssertionStoreError("reactivate_inferred requires an active inferred derived assertion")
        operation_id = operation_id or f"reactivate-inferred:{assertion.revision_id}"
        request = {"assertion": assertion.to_mapping()}
        async with self._mutation():
            digest, replay = await self._operation(operation_id, "reactivate_inferred", request)
            if replay is not None:
                restored = await self._revision(str(replay["revision_id"]))
                if restored is None:
                    raise AssertionConflictError("idempotent reactivation receipt no longer has a revision")
                return AssertionWriteResult(
                    restored, int(replay["generation"]), str(replay["event_id"]), True
                )
            current = await self._current(assertion.assertion_id)
            if current is None:
                raise AssertionConflictError("reactivation requires an existing inferred assertion identity")
            if current.status is AssertionStatus.ACTIVE:
                if current.epistemic_state is EpistemicState.INFERRED:
                    return AssertionWriteResult(current, await self._generation(), "", True)
                raise AssertionConflictError("inference cannot replace an active direct assertion")
            if (
                current.status is not AssertionStatus.RETRACTED
                or current.epistemic_state is not EpistemicState.RETRACTED
                or not isinstance(current.lineage, DerivedLineage)
            ):
                raise AssertionConflictError("inference cannot reactivate a non-derived assertion identity")
            await self._validate_lineage(assertion, ())
            await self._write_revision(assertion, ())
            await self._set_current(assertion)
            generation = await self._advance_generation()
            event_id = await self._event(assertion, "inferred", generation)
            await self._record_operation(
                operation_id,
                "reactivate_inferred",
                digest,
                {"revision_id": assertion.revision_id, "generation": generation, "event_id": event_id},
                [assertion.assertion_id],
                [assertion.revision_id],
            )
            return AssertionWriteResult(assertion, generation, event_id)

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
            await self._invalidate_eligibility(current.revision_id)
            dependents = await self._dependent_current_revisions([current.revision_id])
            to_retract = [current] + [item for item in dependents if item.status is AssertionStatus.ACTIVE]
            retracted: list[Assertion] = []
            invalidated: list[str] = []
            for item in to_retract:
                if item.revision_id != current.revision_id:
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
        _migration_capability: _RawAssertionMutationCapability | None = None,
    ) -> ValidationQuarantineResult:
        """Private migration-only single-assertion lifecycle repair.

        Normal validation must use
        :meth:`persist_validation_report_and_quarantine`, which keeps report
        persistence and every report target in one transaction.  This legacy
        primitive is capability-restricted so it cannot become a public path
        that pairs an arbitrary report id with a partial lifecycle change.
        """
        self._require_raw_mutation_capability(_migration_capability)
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
        explicit_fact_selector: tuple[IRI, IRI] | None = None,
    ) -> DeletionResult:
        """Apply a non-erasure lifecycle deletion and invalidate dependents.

        Unlike :meth:`erase`, this preserves the audit shell as a current
        ``deleted`` revision.  It remains ineligible for query, index, and
        training consumption; physical erasure removes that shell entirely.
        """
        operation_id = operation_id or f"delete:{assertion_id}:{expected_revision_id}"
        request: dict[str, object] = {
            "assertion_id": assertion_id,
            "expected_revision_id": expected_revision_id,
        }
        if explicit_fact_selector is not None:
            if (
                not isinstance(explicit_fact_selector, tuple)
                or len(explicit_fact_selector) != 2
                or not isinstance(explicit_fact_selector[0], IRI)
                or not isinstance(explicit_fact_selector[1], IRI)
                or not operation_id.startswith(
                    _EXPLICIT_FACT_FORGET_OPERATION_PREFIX
                )
            ):
                raise AssertionStoreError(
                    "explicit fact deletion requires a deterministic mapped selector"
                )
            request["explicit_fact_selector"] = {
                "subject": explicit_fact_selector[0].value,
                "predicate": explicit_fact_selector[1].value,
            }
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
            await self._invalidate_eligibility(current.revision_id)
            dependents = await self._dependent_current_revisions([current.revision_id])
            invalidated_revision_ids = [current.revision_id]
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
                    "expected_predecessor_revision_id": expected_revision_id,
                    "deleted_revision_id": deleted.revision_id,
                    "invalidated_revision_ids": invalidated_revision_ids,
                    "invalidation_state_revision_ids": [item.revision_id for item in invalidated],
                    "generation": generation,
                    **(
                        {
                            "explicit_fact_selector": request[
                                "explicit_fact_selector"
                            ]
                        }
                        if "explicit_fact_selector" in request
                        else {}
                    ),
                },
                [deleted.assertion_id, *[item.assertion_id for item in invalidated]],
                [deleted.revision_id, *[item.revision_id for item in invalidated]],
            )
            return DeletionResult(deleted, tuple(invalidated), tuple(invalidated_revision_ids), generation)

    async def replay_delete_operation(
        self,
        operation_id: str,
    ) -> DeletionResult | ErasedDeletionOperationReplay | None:
        """Return the exact deletion receipt selected by an earlier operation.

        Deletion revisions intentionally leave their historical active
        predecessors in the append-only ledger.  A retry must therefore not
        search for an ``ACTIVE`` historical row: source-appends can produce
        several of them.  The delete receipt is the authoritative binding from
        the invocation's operation ID to its predecessor and deletion state.
        """
        recorded = await self._recorded_operation(operation_id)
        if recorded is None:
            erased = await self._erased_operation_tombstone(operation_id)
            if erased is None:
                return None
            purpose, _, generation = erased
            if purpose != "delete":
                raise AssertionConflictError(
                    "operation_id resolves to a non-deletion erased semantic mutation"
                )
            return ErasedDeletionOperationReplay(generation)
        operation, receipt = recorded
        if operation != "delete":
            raise AssertionConflictError(
                "operation_id resolves to a non-deletion semantic mutation"
            )
        predecessor_id = receipt.get("expected_predecessor_revision_id")
        deleted_id = receipt.get("deleted_revision_id")
        invalidated_ids = receipt.get("invalidated_revision_ids")
        state_ids = receipt.get("invalidation_state_revision_ids")
        generation = receipt.get("generation")
        if not (
            isinstance(predecessor_id, str)
            and predecessor_id
            and isinstance(deleted_id, str)
            and deleted_id
            and isinstance(invalidated_ids, list)
            and all(isinstance(item, str) for item in invalidated_ids)
            and isinstance(state_ids, list)
            and all(isinstance(item, str) for item in state_ids)
            and type(generation) is int
        ):
            raise AssertionConflictError("deletion operation receipt is malformed")
        # The predecessor is deliberately resolved from the receipt, not the
        # current graph.  Retain the check so a malformed/future receipt cannot
        # claim a deletion unrelated to this tenant's immutable history.
        try:
            if await self._revision(predecessor_id) is None:
                raise AssertionConflictError(
                    "deletion operation receipt no longer has its predecessor revision"
                )
            deleted = await self._revision(deleted_id)
            invalidated = [await self._revision(item) for item in state_ids]
            if deleted is None or any(item is None for item in invalidated):
                raise AssertionConflictError(
                    "idempotent deletion receipt no longer has its revisions"
                )
            return DeletionResult(
                deleted,
                tuple(item for item in invalidated if item is not None),
                tuple(invalidated_ids),
                generation,
                True,
            )
        except AssertionConflictError:
            erased = await self._erased_operation_tombstone(operation_id)
            if erased is not None:
                purpose, _, erased_generation = erased
                if purpose != "delete":
                    raise AssertionConflictError(
                        "operation_id resolves to a non-deletion erased semantic mutation"
                    )
                return ErasedDeletionOperationReplay(erased_generation)
            raise

    async def replay_explicit_fact_forget(
        self,
        operation_id: str,
        subject: IRI,
        predicate: IRI,
    ) -> ExplicitFactForgetReplay | None:
        """Replay either the exact delete or exact absent-result tombstone."""
        if (
            not operation_id.startswith(
                _EXPLICIT_FACT_FORGET_OPERATION_PREFIX
            )
            or not isinstance(subject, IRI)
            or not isinstance(predicate, IRI)
        ):
            raise AssertionStoreError(
                "explicit fact forget replay requires its mapped selector"
            )
        recorded = await self._recorded_operation(operation_id)
        if recorded is None:
            erased = await self._erased_operation_tombstone(operation_id)
            if erased is None:
                return None
            purpose, request_key, _ = erased
            if purpose == "delete":
                if request_key != _erased_explicit_fact_forget_selector_key(
                    operation_id,
                    subject,
                    predicate,
                ):
                    raise AssertionConflictError(
                        "operation_id resolves to a different erased explicit fact deletion"
                    )
                # Physical erasure removed both the deleted shell and its
                # content-bearing receipt.  The exact forget invocation is
                # still terminal, but no erased identity is reconstructed.
                return ExplicitFactForgetReplay(None, True)
            raise AssertionConflictError(
                "operation_id resolves to an unrelated erased semantic mutation"
            )
        operation, receipt = recorded
        if operation == "delete":
            selector = await self._explicit_fact_forget_selector_from_recorded_result(
                operation_id,
                operation,
                receipt,
            )
            if selector != (subject, predicate):
                erased = await self._erased_operation_tombstone(operation_id)
                if (
                    erased is not None
                    and erased[0] == "delete"
                    and erased[1]
                    == _erased_explicit_fact_forget_selector_key(
                        operation_id,
                        subject,
                        predicate,
                    )
                ):
                    return ExplicitFactForgetReplay(None, True)
                raise AssertionConflictError(
                    "operation_id resolves to a different explicit fact deletion"
                )
            deletion = await self.replay_delete_operation(operation_id)
            if deletion is None:  # pragma: no cover - row was just observed
                raise AssertionConflictError(
                    "explicit fact deletion receipt disappeared during replay"
                )
            if isinstance(deletion, ErasedDeletionOperationReplay):
                return ExplicitFactForgetReplay(None, True)
            return ExplicitFactForgetReplay(deletion, True)
        if operation == _EXPLICIT_FACT_FORGET_NOOP_OPERATION:
            request = {
                "subject": subject.value,
                "predicate": predicate.value,
                "outcome": "absent",
            }
            _, exact_receipt = await self._operation(
                operation_id,
                _EXPLICIT_FACT_FORGET_NOOP_OPERATION,
                request,
            )
            if exact_receipt != {"outcome": "absent"}:
                raise AssertionConflictError(
                    "explicit fact no-op forget receipt is malformed"
                )
            return ExplicitFactForgetReplay(None, True)
        raise AssertionConflictError(
            "operation_id resolves to an unrelated semantic mutation"
        )

    async def record_explicit_fact_forget_noop(
        self,
        operation_id: str,
        subject: IRI,
        predicate: IRI,
    ) -> ExplicitFactForgetReplay:
        """Linearizably record that a fact selector had no active assertion.

        The operation ledger is part of the durable lifecycle contract even
        when no assertion is deleted.  The absence check and receipt insert
        share the tenant mutation transaction, so a concurrent teaching either
        commits before this check (and forces a caller re-decision) or after the
        no-op tombstone.  Retrying the same invocation can never affect that
        later fact.
        """
        if not isinstance(subject, IRI) or not isinstance(predicate, IRI):
            raise AssertionStoreError(
                "explicit fact no-op requires canonical subject and predicate IRIs"
            )
        request = {
            "subject": subject.value,
            "predicate": predicate.value,
            "outcome": "absent",
        }
        async with self._mutation():
            digest, replay = await self._operation(
                operation_id,
                _EXPLICIT_FACT_FORGET_NOOP_OPERATION,
                request,
            )
            if replay is not None:
                if replay != {"outcome": "absent"}:
                    raise AssertionConflictError(
                        "explicit fact no-op forget receipt is malformed"
                    )
                return ExplicitFactForgetReplay(None, True)
            current = [
                assertion
                for assertion in await self._complete_active_assertions()
                if assertion.subject == subject and assertion.predicate == predicate
            ]
            if current:
                raise AssertionConflictError(
                    "explicit fact selector gained a current assertion before no-op commit"
                )
            receipt = {"outcome": "absent"}
            await self._record_operation(
                operation_id,
                _EXPLICIT_FACT_FORGET_NOOP_OPERATION,
                digest,
                receipt,
                (),
                (),
            )
            return ExplicitFactForgetReplay(None, False)

    async def invalidate_assertion_eligibility(
        self,
        assertion_id: str,
        expected_revision_id: str,
        *,
        operation_id: str | None = None,
    ) -> RetractionResult:
        """Withdraw validation eligibility and retract unsupported conclusions.

        Validation loss is not a semantic retraction of the source statement:
        its audit revision remains current, but it may no longer support an
        inferred assertion.  This path deactivates every affected ledger proof
        and retracts only conclusions with no independent active derivation.
        """
        operation_id = operation_id or (
            f"invalidate-eligibility:{assertion_id}:{expected_revision_id}"
        )
        request = {
            "assertion_id": assertion_id,
            "expected_revision_id": expected_revision_id,
        }
        async with self._mutation():
            digest, replay = await self._operation(
                operation_id, "invalidate_eligibility", request
            )
            if replay is not None:
                retracted = [
                    await self._revision(str(revision_id))
                    for revision_id in replay["retracted_revision_ids"]
                ]
                if any(item is None for item in retracted):
                    raise AssertionConflictError(
                        "idempotent eligibility receipt no longer has its revisions"
                    )
                return RetractionResult(
                    tuple(item for item in retracted if item is not None),
                    tuple(replay["invalidated_revision_ids"]),
                    int(replay["generation"]),
                    True,
                )
            current = await self._current(assertion_id)
            if (
                current is None
                or current.revision_id != expected_revision_id
                or current.status is not AssertionStatus.ACTIVE
            ):
                raise AssertionConflictError(
                    "expected revision is not the active current tenant assertion"
                )
            await self._invalidate_eligibility(current.revision_id)
            dependents = await self._dependent_current_revisions((current.revision_id,))
            invalidated_revision_ids = [current.revision_id]
            retracted: list[Assertion] = []
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
                retracted.append(state)
            generation = await self._advance_generation()
            source_event = await self._event(
                current,
                "validation_ineligible",
                generation,
                eligible=False,
            )
            dependent_events = [
                await self._event(item, "retracted", generation)
                for item in retracted
            ]
            await self._record_operation(
                operation_id,
                "invalidate_eligibility",
                digest,
                {
                    "retracted_revision_ids": [item.revision_id for item in retracted],
                    "invalidated_revision_ids": invalidated_revision_ids,
                    "generation": generation,
                    "event_ids": [source_event, *dependent_events],
                },
                [current.assertion_id, *[item.assertion_id for item in retracted]],
                [current.revision_id, *[item.revision_id for item in retracted]],
            )
            return RetractionResult(
                tuple(retracted), tuple(invalidated_revision_ids), generation
            )

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

    async def _erasure_replacement_lineage(
        self,
        assertion: Assertion,
        erased_revision_ids: set[str],
        refreshed_revision_ids: dict[str, str],
        run_id: str,
    ) -> DerivedLineage | None:
        """Choose one valid proof for a current conclusion being re-lined.

        The canonical ``semantic_derivation_inputs`` row names one primary
        proof, while the inference ledger holds every independent proof.  An
        erasure may remove the primary proof without removing the conclusion.
        In that case the conclusion receives a fresh canonical revision whose
        lineage names a surviving complete premise set; the old revision is
        still physically erased.
        """
        if not isinstance(assertion.lineage, DerivedLineage):
            return None

        async def resolved_inputs(
            input_revision_ids: Sequence[str],
        ) -> tuple[str, ...] | None:
            resolved = tuple(
                refreshed_revision_ids.get(revision_id, revision_id)
                for revision_id in input_revision_ids
            )
            if len(set(resolved)) != len(resolved):
                return None
            for original_revision_id, resolved_revision_id in zip(
                input_revision_ids, resolved
            ):
                if (
                    original_revision_id in erased_revision_ids
                    and original_revision_id not in refreshed_revision_ids
                ):
                    return None
                if resolved_revision_id in erased_revision_ids:
                    return None
                if not await self._is_current_active_eligible_revision(
                    resolved_revision_id
                ):
                    return None
            return resolved

        primary_inputs = await resolved_inputs(
            assertion.lineage.input_revision_ids
        )
        generated_at = _now()
        if primary_inputs is not None:
            return DerivedLineage(
                rule_id=assertion.lineage.rule_id,
                engine_version=assertion.lineage.engine_version,
                profile_version=assertion.lineage.profile_version,
                input_revision_ids=primary_inputs,
                input_digest=_operation_digest({"premises": primary_inputs}),
                run_id=run_id,
                generated_at=generated_at,
                derivation_reference=assertion.lineage.derivation_reference,
            )

        tenant_id, _ = self._require_scope()
        derivations = await self._database.fetchall(
            "SELECT derivation_id, rule_id, rule_profile_version "
            "FROM semantic_inference_derivations "
            "WHERE tenant_id = ? AND derived_revision_id = ? AND active = 1 "
            "ORDER BY derivation_id ASC",
            (tenant_id, assertion.revision_id),
        )
        for derivation_id, rule_id, profile_version in derivations:
            rows = await self._database.fetchall(
                "SELECT input_revision_id FROM semantic_inference_derivation_inputs "
                "WHERE tenant_id = ? AND derivation_id = ? ORDER BY ordinal ASC",
                (tenant_id, derivation_id),
            )
            inputs = await resolved_inputs(tuple(str(row[0]) for row in rows))
            if inputs is None:
                continue
            return DerivedLineage(
                rule_id=str(rule_id),
                engine_version=assertion.lineage.engine_version,
                profile_version=str(profile_version),
                input_revision_ids=inputs,
                input_digest=_operation_digest({"premises": inputs}),
                run_id=run_id,
                generated_at=generated_at,
                derivation_reference=f"urn:kestrel:inference:{derivation_id}",
            )
        return None

    async def _freshen_inferred_for_erasure(
        self,
        assertion: Assertion,
        lineage: DerivedLineage,
    ) -> Assertion:
        """Publish a fresh current inferred revision without retaining old lineage."""
        if (
            assertion.status is not AssertionStatus.ACTIVE
            or assertion.epistemic_state is not EpistemicState.INFERRED
            or not isinstance(assertion.lineage, DerivedLineage)
        ):
            raise AssertionStoreError(
                "only an active inferred assertion can be re-lined for erasure"
            )
        mapping = assertion.to_mapping()
        mapping["revision_id"] = uuid4().hex
        mapping["status"] = AssertionStatus.ACTIVE.value
        mapping["epistemic_state"] = EpistemicState.INFERRED.value
        mapping["supersedes_revision_id"] = None
        mapping["asserted_at"] = {"schema_version": 1, "value": _now()}
        mapping["lineage"] = lineage.to_mapping()
        replacement = Assertion.from_mapping(mapping)
        await self._validate_lineage(replacement, ())
        await self._invalidate_eligibility(
            assertion.revision_id, deactivate_inference_derivations=False
        )
        await self._write_revision(replacement, ())
        await self._set_current(replacement)
        return replacement

    async def _reconcile_inference_derivations_after_erasure(
        self,
        erased_revision_ids: set[str],
        refreshed_revision_ids: dict[str, str],
    ) -> None:
        """Delete erased proofs and retarget independently surviving proofs.

        A refreshed conclusion has a different revision ID.  Every surviving
        ledger proof that names the old revision must point to the fresh one;
        every proof with an actually erased input must disappear so an erased
        revision is never recoverable through the explanation ledger.
        """
        if not erased_revision_ids:
            return
        tenant_id, _ = self._require_scope()
        affected = tuple(sorted(erased_revision_ids))
        rows = await self._database.fetchall(
            "SELECT DISTINCT d.derivation_id, d.derived_revision_id "
            "FROM semantic_inference_derivations d "
            "LEFT JOIN semantic_inference_derivation_inputs i "
            "  ON i.tenant_id = d.tenant_id AND i.derivation_id = d.derivation_id "
            "WHERE d.tenant_id = ? "
            f"AND (d.derived_revision_id IN ({_placeholders(affected)}) "
            f"OR i.input_revision_id IN ({_placeholders(affected)}))",
            (tenant_id,) + affected + affected,
        )
        for derivation_id, derived_revision_id in rows:
            input_rows = await self._database.fetchall(
                "SELECT input_revision_id FROM semantic_inference_derivation_inputs "
                "WHERE tenant_id = ? AND derivation_id = ? ORDER BY ordinal ASC",
                (tenant_id, derivation_id),
            )
            original_inputs = tuple(str(row[0]) for row in input_rows)
            original_derived = str(derived_revision_id)
            if (
                original_derived in erased_revision_ids
                and original_derived not in refreshed_revision_ids
            ) or any(
                revision_id in erased_revision_ids
                and revision_id not in refreshed_revision_ids
                for revision_id in original_inputs
            ):
                await self._database.execute(
                    "DELETE FROM semantic_inference_derivation_inputs "
                    "WHERE tenant_id = ? AND derivation_id = ?",
                    (tenant_id, derivation_id),
                )
                await self._database.execute(
                    "DELETE FROM semantic_inference_derivations "
                    "WHERE tenant_id = ? AND derivation_id = ?",
                    (tenant_id, derivation_id),
                )
                continue
            replacement_derived = refreshed_revision_ids.get(
                original_derived, original_derived
            )
            replacement_inputs = tuple(
                refreshed_revision_ids.get(revision_id, revision_id)
                for revision_id in original_inputs
            )
            if (
                replacement_derived == original_derived
                and replacement_inputs == original_inputs
            ):
                continue
            if len(set(replacement_inputs)) != len(replacement_inputs):
                raise AssertionStoreError(
                    "erasure rematerialization produced duplicate derivation inputs"
                )
            await self._database.execute(
                "UPDATE semantic_inference_derivations SET derived_revision_id = ? "
                "WHERE tenant_id = ? AND derivation_id = ?",
                (replacement_derived, tenant_id, derivation_id),
            )
            await self._database.execute(
                "DELETE FROM semantic_inference_derivation_inputs "
                "WHERE tenant_id = ? AND derivation_id = ?",
                (tenant_id, derivation_id),
            )
            for ordinal, revision_id in enumerate(replacement_inputs):
                await self._database.execute(
                    "INSERT INTO semantic_inference_derivation_inputs "
                    "(tenant_id, derivation_id, input_revision_id, ordinal) "
                    "VALUES (?, ?, ?, ?)",
                    (tenant_id, derivation_id, revision_id, ordinal),
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
            refreshed_revision_ids: dict[str, str] = {}
            erasure_run_id = f"inference:erasure-rematerialization:{uuid4().hex}"
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
                    if current_revision_id != derived_revision_id:
                        # A newer independent current revision owns this
                        # deterministic assertion identity. Remove only its
                        # reachable historical lineage, never that replacement.
                        continue
                    if derived_assertion_id in assertion_ids:
                        # This assertion has no valid alternate proof and is
                        # part of the physical-erasure closure.
                        all_revisions = await self._database.fetchall(
                            "SELECT revision_id FROM semantic_assertion_revisions "
                            "WHERE tenant_id = ? AND assertion_id = ? ORDER BY revision_id ASC",
                            (tenant_id, derived_assertion_id),
                        )
                        for (revision_id,) in all_revisions:
                            if revision_id not in revision_ids:
                                revision_ids.add(revision_id)
                                pending.add(revision_id)
                        continue
                    current = await self._current(str(derived_assertion_id))
                    if current is None or current.revision_id != derived_revision_id:
                        raise AssertionStoreError(
                            "current assertion changed during serialized erasure"
                        )
                    all_revisions = await self._database.fetchall(
                        "SELECT revision_id FROM semantic_assertion_revisions "
                        "WHERE tenant_id = ? AND assertion_id = ? ORDER BY revision_id ASC",
                        (tenant_id, derived_assertion_id),
                    )
                    replacement_lineage = await self._erasure_replacement_lineage(
                        current,
                        revision_ids,
                        refreshed_revision_ids,
                        erasure_run_id,
                    )
                    if replacement_lineage is not None:
                        # Capture every pre-existing revision before writing
                        # the replacement: the fresh revision is the retained
                        # conclusion, while historical proof rows remain
                        # reachable erasure data.
                        for (revision_id,) in all_revisions:
                            if revision_id not in revision_ids:
                                revision_ids.add(revision_id)
                                pending.add(revision_id)
                        replacement = await self._freshen_inferred_for_erasure(
                            current, replacement_lineage
                        )
                        refreshed_revision_ids[current.revision_id] = (
                            replacement.revision_id
                        )
                        continue

                    assertion_ids.add(str(derived_assertion_id))
                    for (revision_id,) in all_revisions:
                        if revision_id not in revision_ids:
                            revision_ids.add(revision_id)
                            pending.add(revision_id)
            await self._reconcile_inference_derivations_after_erasure(
                revision_ids,
                refreshed_revision_ids,
            )
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
            inference_rows = await self._database.fetchall(
                "SELECT derivation_id FROM semantic_inference_derivations "
                f"WHERE tenant_id = ? AND derived_revision_id IN ({_placeholders(revision_tuple)}) "
                "UNION "
                "SELECT derivation_id FROM semantic_inference_derivation_inputs "
                f"WHERE tenant_id = ? AND input_revision_id IN ({_placeholders(revision_tuple)})",
                (tenant_id,) + revision_tuple + (tenant_id,) + revision_tuple,
            )
            inference_derivation_ids = tuple(sorted({str(row[0]) for row in inference_rows}))
            if inference_derivation_ids:
                await self._database.execute(
                    "DELETE FROM semantic_inference_derivation_inputs WHERE tenant_id = ? "
                    f"AND derivation_id IN ({_placeholders(inference_derivation_ids)})",
                    (tenant_id,) + inference_derivation_ids,
                )
                await self._database.execute(
                    "DELETE FROM semantic_inference_derivations WHERE tenant_id = ? "
                    f"AND derivation_id IN ({_placeholders(inference_derivation_ids)})",
                    (tenant_id,) + inference_derivation_ids,
                )
            await self._database.execute(
                f"DELETE FROM semantic_derivation_inputs WHERE tenant_id = ? AND (derived_revision_id IN ({_placeholders(revision_tuple)}) OR input_revision_id IN ({_placeholders(revision_tuple)}))",
                (tenant_id,) + revision_tuple + revision_tuple,
            )
            # Reserve the committed erasure checkpoint while revisions and
            # their attached source rows are still present.  Fact operation
            # tombstones bind a blinded immutable accepted-result fingerprint,
            # so that binding must be derived before provenance is removed.
            # The surrounding mutation keeps the generation, tombstones,
            # physical deletes, and opaque erasure event atomic.
            generation = await self._advance_generation()
            await self._tombstone_and_delete_operations_referencing(
                assertion_ids,
                revision_ids,
                generation,
            )
            await self._database.execute(
                f"DELETE FROM semantic_revision_sources WHERE tenant_id = ? AND revision_id IN ({_placeholders(revision_tuple)})",
                (tenant_id,) + revision_tuple,
            )
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
        return await self._query_current(query)

    async def _query_current(
        self,
        query: AssertionQuery,
        *,
        eligible_only: bool = False,
    ) -> list[Assertion]:
        """Run one tenant-bound current-assertion query with optional eligibility.

        Materialization must filter eligibility in SQL, before a page enters
        memory.  Keeping that predicate in this shared query primitive also
        ensures the service cannot accidentally observe a different tenant's
        projection tombstones.
        """
        tenant_id, _ = self._require_scope()
        clauses = ["a.tenant_id = ?", "a.current_revision_id = r.revision_id"]
        params: list[object] = [tenant_id]
        eligibility_join = ""
        if eligible_only:
            eligibility_join = (
                " JOIN semantic_projection_eligibility e "
                "ON e.tenant_id = r.tenant_id AND e.revision_id = r.revision_id"
            )
            clauses.extend(("r.eligible = 1", "e.eligible = 1"))
        if query.subject is not None:
            clauses.append("r.subject_value = ?")
            params.append(query.subject.value)
        if query.predicate is not None:
            clauses.append("r.predicate_value = ?")
            params.append(query.predicate.value)
        if query.object is not None:
            mapping = query.object.identity_mapping()
            clauses.extend(("r.object_kind = ?", "r.object_value = ?"))
            datatype = mapping.get("datatype")
            language = mapping.get("language")
            params.extend((mapping["kind"], mapping["value"]))
            # PostgreSQL cannot infer the type of a standalone NULL bind
            # parameter.  IRI and blank-node terms have no datatype or
            # language, so express their canonical identity with SQL NULL
            # predicates rather than ``column = ? OR (? IS NULL ...)``.
            if datatype is None:
                clauses.append("r.object_datatype IS NULL")
            else:
                clauses.append("r.object_datatype = ?")
                params.append(datatype)
            if language is None:
                clauses.append("r.object_language IS NULL")
            else:
                clauses.append("r.object_language = ?")
                params.append(language)
        if query.assertion_ids:
            clauses.append(f"a.assertion_id IN ({_placeholders(query.assertion_ids)})")
            params.extend(query.assertion_ids)
        if query.exclude_assertion_ids:
            clauses.append(
                f"a.assertion_id NOT IN ({_placeholders(query.exclude_assertion_ids)})"
            )
            params.extend(query.exclude_assertion_ids)
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
            (
                "SELECT r.assertion_mapping FROM semantic_assertions a "
                "JOIN semantic_assertion_revisions r ON r.tenant_id = a.tenant_id "
                + eligibility_join
                + " WHERE "
                + " AND ".join(clauses)
                + " ORDER BY r.revision_id ASC LIMIT ?"
            ),
            tuple(params),
        )
        return [Assertion.from_mapping(json.loads(row[0])) for row in rows]

    async def inference_inputs(self, query: AssertionQuery | None = None) -> list[Assertion]:
        """Return only current, active, projection-eligible tenant assertions."""
        query = query or AssertionQuery()
        if not isinstance(query, AssertionQuery):
            raise AssertionStoreError("inference_inputs requires AssertionQuery")
        return await self._query_current(query, eligible_only=True)

    async def inference_inputs_page(
        self,
        *,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> list[Assertion]:
        """Return one bounded page of eligible inference inputs.

        The cursor is a revision ID in the store's deterministic order.  This
        intentionally accepts no tenant parameter: the store's authenticated
        scope remains the only source of tenant selection.
        """
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise AssertionStoreError("inference input cursor must be a non-empty string or null")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise AssertionStoreError("inference input limit must be an integer in [1, 1000]")
        return await self._query_current(
            AssertionQuery(limit=limit, cursor=cursor), eligible_only=True
        )

    async def active_assertion_count(self, *, cursor: str | None = None) -> int:
        """Count active current assertions after an optional revision cursor.

        Semantic repair uses this aggregate for an exact backlog count without
        materializing the rest of the tenant graph.
        """
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise AssertionStoreError(
                "active assertion cursor must be a non-empty string or null"
            )
        tenant_id, _ = self._require_scope()
        clauses = [
            "a.tenant_id = ?",
            "a.current_revision_id = r.revision_id",
            "r.status = ?",
        ]
        params: list[object] = [tenant_id, AssertionStatus.ACTIVE.value]
        if cursor is not None:
            clauses.append("r.revision_id > ?")
            params.append(cursor)
        value = await self._database.fetchval(
            "SELECT COUNT(*) FROM semantic_assertions a "
            "JOIN semantic_assertion_revisions r ON r.tenant_id = a.tenant_id "
            "WHERE " + " AND ".join(clauses),
            tuple(params),
        )
        return int(value or 0)

    async def _complete_active_assertions(self) -> tuple[Assertion, ...]:
        """Return every active current assertion in this store's bound tenant.

        ``AssertionQuery`` defaults to an API-safe page of 100 rows.  That is
        appropriate for ordinary reads, but lifecycle planning and governed
        SHACL validation must operate on the complete active graph.  Keep the
        cursor walk here so every such caller has one authoritative snapshot
        collection path rather than silently inheriting the public page limit.
        """
        assertions: list[Assertion] = []
        cursor: str | None = None
        while True:
            page = await self._query_current(
                AssertionQuery(limit=1000, cursor=cursor)
            )
            assertions.extend(page)
            if len(page) < 1000:
                return tuple(assertions)
            cursor = page[-1].revision_id

    async def export_snapshot(self, query: AssertionQuery | None = None) -> tuple[AssertionCheckpoint, tuple[Assertion, ...]]:
        """Return a tenant-bound active snapshot for a caller already scoped here.

        The store deliberately has no per-call tenant argument.  Export callers
        receive only records selected through the same lifecycle-filtered query
        used by inference; a higher layer may further apply release policy and
        destination governance without widening this canonical scope.  An
        unqualified snapshot is the complete active graph, not the first
        public-query page; an explicit query retains its caller-selected page
        semantics.
        """
        checkpoint = await self.checkpoint()
        if query is not None:
            return checkpoint, tuple(await self.query(query))
        return checkpoint, await self._complete_active_assertions()

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

    async def changes_after(
        self,
        checkpoint: AssertionCheckpoint,
        *,
        limit: int = 100,
    ) -> list[AssertionChange]:
        """Read the canonical stream strictly after an event-level cursor.

        Generations are transaction boundaries, not event sequence numbers:
        one lifecycle transition can emit several outbox events at the same
        generation.  A maintenance cursor must therefore retain the event ID
        and resume in the stream's canonical ``generation, created_at,
        event_id`` order.  A legacy generation-only checkpoint is recovered
        conservatively by replaying its whole final generation once.
        """
        if not isinstance(checkpoint, AssertionCheckpoint):
            raise AssertionStoreError("changes_after requires an AssertionCheckpoint")
        if checkpoint.tenant_id != self.tenant_id:
            raise TenantIsolationError("change checkpoint tenant does not match bound store")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise AssertionStoreError("limit must be an integer in [1, 1000]")
        if checkpoint.latest_event_id is not None and (
            not isinstance(checkpoint.latest_event_id, str)
            or not checkpoint.latest_event_id
        ):
            raise AssertionStoreError("checkpoint latest_event_id must be non-empty or null")

        tenant_id, _ = self._require_scope()
        stream = (
            "SELECT event_id, assertion_id, revision_id, operation, generation, eligible, created_at "
            "FROM semantic_projection_outbox WHERE tenant_id = ? "
            "UNION ALL "
            "SELECT event_id, NULL AS assertion_id, NULL AS revision_id, operation, generation, "
            "0 AS eligible, created_at FROM semantic_projection_erasure_outbox WHERE tenant_id = ?"
        )
        event_id = checkpoint.latest_event_id
        if event_id is None:
            # v1 maintenance state could only store a generation.  Replay the
            # final generation rather than assuming it was fully consumed;
            # report and canonical writes are idempotent, while skipping a
            # same-generation event is not recoverable.
            comparator = ">" if checkpoint.generation == 0 else ">="
            rows = await self._database.fetchall(
                "SELECT event_id, assertion_id, revision_id, operation, generation, eligible "
                f"FROM ({stream}) AS assertion_changes WHERE generation {comparator} ? "
                "ORDER BY generation ASC, created_at ASC, event_id ASC LIMIT ?",
                (tenant_id, tenant_id, checkpoint.generation, limit),
            )
        else:
            cursor = await self._database.fetchone(
                "SELECT generation, created_at FROM ("
                + stream
                + ") AS assertion_changes WHERE event_id = ? LIMIT 1",
                (tenant_id, tenant_id, event_id),
            )
            if cursor is None:
                # Physical erasure removes ordinary outbox rows.  Its opaque
                # resynchronization event is later than every removed row, so
                # replaying this generation is the safe recovery if a durable
                # cursor itself was erased.
                rows = await self._database.fetchall(
                    "SELECT event_id, assertion_id, revision_id, operation, generation, eligible "
                    f"FROM ({stream}) AS assertion_changes WHERE generation >= ? "
                    "ORDER BY generation ASC, created_at ASC, event_id ASC LIMIT ?",
                    (tenant_id, tenant_id, checkpoint.generation, limit),
                )
            else:
                cursor_generation, cursor_created_at = int(cursor[0]), str(cursor[1])
                if cursor_generation != checkpoint.generation:
                    raise AssertionStoreError(
                        "checkpoint generation does not match its event cursor"
                    )
                rows = await self._database.fetchall(
                    "SELECT event_id, assertion_id, revision_id, operation, generation, eligible "
                    f"FROM ({stream}) AS assertion_changes "
                    "WHERE generation > ? OR (generation = ? AND "
                    "(created_at > ? OR (created_at = ? AND event_id > ?))) "
                    "ORDER BY generation ASC, created_at ASC, event_id ASC LIMIT ?",
                    (
                        tenant_id,
                        tenant_id,
                        checkpoint.generation,
                        checkpoint.generation,
                        cursor_created_at,
                        cursor_created_at,
                        event_id,
                        limit,
                    ),
                )
        return [
            AssertionChange(row[0], row[1], row[2], row[3], int(row[4]), bool(row[5]))
            for row in rows
        ]


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
    "SemanticAssertionStore", "SupersessionResult", "TenantIsolationError", "ValidationQuarantineBatchResult",
    "ValidationQuarantineResult",
]
