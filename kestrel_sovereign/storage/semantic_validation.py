"""Tenant-bound SHACL report persistence and canonical revalidation service.

The report repository shares the canonical assertion database and is scoped by
an already-authenticated assertion store.  It is not a writable validation
database, and it never updates projection eligibility itself: failed
revalidation goes through the assertion store's atomic
``persist_validation_report_and_quarantine()`` transition, which emits the
canonical change metadata consumed by downstream projections.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from rdflib import Graph, Literal as RdfLiteral, RDF, URIRef

from kestrel_sovereign.knowledge import (
    Assertion,
    AssertionQuery,
    DirectLineage,
    GovernedShaclValidationService,
    ShaclValidationLimits,
    ShaclValidationReport,
    ShapeSetReference,
    ValidationSource,
    ValidationState,
    ValidationWriteAction,
)

from .async_assertion_store import (
    AssertionConflictError,
    AssertionStoreError,
    AssertionWriteResult,
    AsyncAssertionStore,
    MaintenanceLeaseLostError,
    SupersessionResult,
    TenantIsolationError,
)


_KESTREL = "https://kestrel.ai/vocab/"
_PROV = "http://www.w3.org/ns/prov#"
_REVISION_PREFIX = "urn:kestrel:semantic-revision:"
_SOURCE_PREFIX = "urn:kestrel:semantic-source:"
_DERIVATION_PREFIX = "urn:kestrel:semantic-derivation:"
_LINEAGE_MEMBER_PREFIX = "urn:kestrel:semantic-lineage-member:"


class SemanticValidationStoreError(ValueError):
    """A validation report cannot be persisted or read in the bound tenant."""


@dataclass(frozen=True, slots=True)
class GovernedAssertionWriteResult:
    """A write policy decision and, only when accepted, its canonical receipt."""

    report: ShaclValidationReport
    write: AssertionWriteResult | None

    @property
    def accepted(self) -> bool:
        return self.write is not None

    # These compatibility accessors let legacy callers that only consumed an
    # accepted receipt migrate without discarding the now-required report.  A
    # rejected decision has no receipt and therefore cannot masquerade as one.
    @property
    def assertion(self) -> Assertion:
        if self.write is None:
            raise SemanticValidationStoreError("rejected governed write has no assertion receipt")
        return self.write.assertion

    @property
    def generation(self) -> int:
        if self.write is None:
            raise SemanticValidationStoreError("rejected governed write has no generation receipt")
        return self.write.generation

    @property
    def event_id(self) -> str:
        if self.write is None:
            raise SemanticValidationStoreError("rejected governed write has no event receipt")
        return self.write.event_id

    @property
    def idempotent(self) -> bool:
        return self.write.idempotent if self.write is not None else False


@dataclass(frozen=True, slots=True)
class GovernedAssertionSupersessionResult:
    """A SHACL policy decision and an optional canonical supersession receipt."""

    report: ShaclValidationReport
    write: SupersessionResult | None

    @property
    def accepted(self) -> bool:
        return self.write is not None

    @property
    def predecessor(self) -> Assertion:
        if self.write is None:
            raise SemanticValidationStoreError("rejected governed supersession has no receipt")
        return self.write.predecessor

    @property
    def replacement(self) -> Assertion:
        if self.write is None:
            raise SemanticValidationStoreError("rejected governed supersession has no receipt")
        return self.write.replacement

    @property
    def generation(self) -> int:
        if self.write is None:
            raise SemanticValidationStoreError("rejected governed supersession has no generation receipt")
        return self.write.generation

    @property
    def event_ids(self) -> tuple[str, ...]:
        if self.write is None:
            raise SemanticValidationStoreError("rejected governed supersession has no event receipt")
        return self.write.event_ids

    @property
    def invalidated_revision_ids(self) -> tuple[str, ...]:
        if self.write is None:
            raise SemanticValidationStoreError("rejected governed supersession has no receipt")
        return self.write.invalidated_revision_ids

    @property
    def idempotent(self) -> bool:
        return self.write.idempotent if self.write is not None else False


class AsyncSemanticValidationReportStore:
    """Persist privacy-safe validation reports in an assertion store's tenant."""

    def __init__(self, assertion_store: AsyncAssertionStore, /) -> None:
        if not isinstance(assertion_store, AsyncAssertionStore):
            raise TypeError("validation reports require an agent-bound AsyncAssertionStore")
        self._assertions = assertion_store

    @property
    def tenant_id(self) -> str:
        return self._assertions.tenant_id

    @property
    def _database(self):
        return self._assertions._database

    async def persist(self, report: ShaclValidationReport) -> ShaclValidationReport:
        if not isinstance(report, ShaclValidationReport):
            raise SemanticValidationStoreError("persist requires a SHACL validation report")
        if report.tenant_id != self.tenant_id:
            raise SemanticValidationStoreError("validation report tenant does not match the bound assertion tenant")
        try:
            return await self._assertions.persist_validation_report(report)
        except MaintenanceLeaseLostError:
            raise
        except AssertionStoreError as error:
            raise SemanticValidationStoreError(str(error)) from error

    async def get(self, report_id: str) -> ShaclValidationReport | None:
        if not isinstance(report_id, str) or not report_id:
            raise SemanticValidationStoreError("report_id must be non-empty")
        row = await self._database.fetchone(
            "SELECT report_mapping FROM semantic_validation_reports "
            "WHERE tenant_id = ? AND report_id = ?",
            (self.tenant_id, report_id),
        )
        return ShaclValidationReport.from_mapping(json.loads(row[0])) if row is not None else None

    async def list(
        self,
        *,
        assertion_id: str | None = None,
        limit: int = 100,
    ) -> list[ShaclValidationReport]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise SemanticValidationStoreError("limit must be an integer in [1, 1000]")
        clauses = ["tenant_id = ?"]
        params: list[object] = [self.tenant_id]
        if assertion_id is not None:
            if not isinstance(assertion_id, str) or not assertion_id:
                raise SemanticValidationStoreError("assertion_id must be non-empty when supplied")
            # A normalized reference row permits parity-safe targeted lookup
            # even when the report has no violations (and therefore no result
            # rows) or on a backend without JSON containment operators.
            clauses.append(
                "report_id IN (SELECT report_id FROM semantic_validation_report_assertions "
                "WHERE tenant_id = ? AND assertion_id = ?)"
            )
            params.extend((self.tenant_id, assertion_id))
        params.append(limit)
        rows = await self._database.fetchall(
            "SELECT report_mapping FROM semantic_validation_reports WHERE "
            + " AND ".join(clauses)
            + " ORDER BY evaluated_at DESC, report_id DESC LIMIT ?",
            tuple(params),
        )
        return [ShaclValidationReport.from_mapping(json.loads(row[0])) for row in rows]


class GovernedSemanticValidationService:
    """Validate canonical assertions, persist reports, and repair via the outbox.

    Maintenance may opt into a bounded focused run that reads only its supplied
    current assertions. The default focused contract retains the complete graph
    for direct validation callers that need cross-assertion findings. The
    validator still fails closed for non-local shape paths; neither method has
    a direct SQL eligibility update path.
    """

    def __init__(
        self,
        assertion_store: AsyncAssertionStore,
        *,
        report_store: AsyncSemanticValidationReportStore | None = None,
        validator: GovernedShaclValidationService | None = None,
    ) -> None:
        if not isinstance(assertion_store, AsyncAssertionStore):
            raise TypeError("governed semantic validation requires an agent-bound AsyncAssertionStore")
        self._assertions = assertion_store
        self._reports = report_store or AsyncSemanticValidationReportStore(assertion_store)
        if self._reports.tenant_id != assertion_store.tenant_id:
            raise SemanticValidationStoreError("report store and assertion store must have the same tenant")
        self._validator = validator or GovernedShaclValidationService()

    @property
    def reports(self) -> AsyncSemanticValidationReportStore:
        return self._reports

    async def validate_current(
        self,
        *,
        assertion_ids: Sequence[str] | None = None,
        shape_set: ShapeSetReference = ShapeSetReference("kestrel-assertion-shapes", "1.0.0"),
        validation_capability: str = "validation-profile:shacl-core-20170720",
        profile_version: str | None = None,
        allow_experimental: bool = False,
        limits: ShaclValidationLimits = ShaclValidationLimits(),
        run_id: str | None = None,
        max_quarantine_retries: int = 2,
        bounded_focus_only: bool = False,
    ) -> ShaclValidationReport:
        """Run a full-graph or affected-focus revalidation and persist its result."""
        if type(max_quarantine_retries) is not int or not 1 <= max_quarantine_retries <= 10:
            raise SemanticValidationStoreError("max_quarantine_retries must be an integer in [1, 10]")
        if type(bounded_focus_only) is not bool:
            raise SemanticValidationStoreError("bounded_focus_only must be a boolean")
        requested = _assertion_ids(assertion_ids)
        for attempt in range(max_quarantine_retries):
            if requested and bounded_focus_only:
                checkpoint = await self._assertions.checkpoint()
                selected = await self._assertions.query(
                    AssertionQuery(assertion_ids=requested, limit=len(requested))
                )
                active_by_id = {
                    assertion.assertion_id: assertion for assertion in selected
                }
                missing = set(requested) - active_by_id.keys()
                if missing:
                    raise SemanticValidationStoreError(
                        "validation target is absent from this tenant's active canonical graph"
                )
                validation_assertion_ids = requested
                selected = [active_by_id[item] for item in requested]
            else:
                checkpoint, assertions = await self._assertions.export_snapshot()
                active_by_id = {
                    assertion.assertion_id: assertion for assertion in assertions
                }
                if requested:
                    missing = set(requested) - active_by_id.keys()
                    if missing:
                        raise SemanticValidationStoreError(
                            "validation target is absent from this tenant's active canonical graph"
                        )
                    validation_assertion_ids = requested
                    selected = [active_by_id[item] for item in requested]
                else:
                    validation_assertion_ids = tuple(sorted(active_by_id))
                    selected = list(assertions)
            data_graph, focus_map, affected_nodes = _canonical_validation_graph(
                selected if bounded_focus_only else assertions, selected
            )
            report = self._validator.validate(
                data_graph,
                tenant_id=self._assertions.tenant_id,
                assertion_ids=validation_assertion_ids,
                shape_set=shape_set,
                validation_capability=validation_capability,
                profile_version=profile_version,
                allow_experimental=allow_experimental,
                # Canonical graph validation is always a revalidation.  The
                # source cannot be caller-selected, because a reject policy
                # would otherwise leave known-invalid current data eligible.
                source=ValidationSource.REVALIDATION,
                checkpoint_generation=checkpoint.generation,
                run_id=run_id,
                focus_nodes=(affected_nodes if requested else None),
                focus_assertion_ids=focus_map,
                require_complete_focus=bounded_focus_only,
                limits=limits,
            )
            # A bounded incremental caller has deliberately not materialized
            # unbounded shape context. Non-local shapes therefore defer rather
            # than quarantining an assertion from an incomplete graph.
            if bounded_focus_only and report.state is ValidationState.INCOMPLETE:
                await self._reports.persist(report)
                return report
            if report.action is not ValidationWriteAction.QUARANTINE:
                await self._reports.persist(report)
                return report
            snapshot_revisions = {
                assertion.assertion_id: assertion.revision_id
                for assertion in (selected if bounded_focus_only else assertions)
            }
            try:
                await self._persist_and_quarantine_failed_current_assertions(
                    report,
                    snapshot_revisions,
                )
            except AssertionConflictError:
                if attempt + 1 == max_quarantine_retries:
                    raise SemanticValidationStoreError(
                        "validation conflict: canonical data changed during bounded revalidation"
                    ) from None
                # The next iteration validates a fresh snapshot.  Do not
                # apply stale findings to a replacement revision.
                continue
            return report
        raise AssertionError("bounded revalidation retry loop exhausted without returning")

    async def put_assertion(
        self,
        assertion: Assertion,
        *,
        source_occurrences=(),
        source: ValidationSource = ValidationSource.ASSERTED,
        shape_set: ShapeSetReference = ShapeSetReference("kestrel-assertion-shapes", "1.0.0"),
        validation_capability: str = "validation-profile:shacl-core-20170720",
        profile_version: str | None = None,
        allow_experimental: bool = False,
        limits: ShaclValidationLimits = ShaclValidationLimits(),
        operation_id: str | None = None,
        run_id: str | None = None,
        max_commit_retries: int = 2,
    ) -> GovernedAssertionWriteResult:
        """Validate a complete tentative graph before accepting an initial write.

        A rejected or quarantined candidate is never handed to the canonical
        store.  Import quarantine therefore remains a tenant-private report,
        rather than a malformed initial assertion disguised as canonical
        ``status=quarantined`` state.
        """
        if not isinstance(assertion, Assertion):
            raise SemanticValidationStoreError("governed assertion write requires a canonical Assertion")
        if assertion.tenant_id != self._assertions.tenant_id:
            raise TenantIsolationError("candidate assertion tenant does not match the bound tenant")
        if type(max_commit_retries) is not int or not 1 <= max_commit_retries <= 10:
            raise SemanticValidationStoreError("max_commit_retries must be an integer in [1, 10]")
        replay = await self._assertions.replay_governed_initial_write(
            assertion,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )
        if replay is not None:
            report, written = replay
            return GovernedAssertionWriteResult(report, written)
        for attempt in range(max_commit_retries):
            checkpoint, current = await self._assertions.export_snapshot()
            # The first replay lookup is the fast path.  A simultaneous
            # delivery can commit after that lookup but before this snapshot;
            # the canonical ledger must win over the lifecycle rejection in
            # that narrow interval.  The store's mutation path checks the same
            # ledger again while holding the tenant lock.
            replay = await self._assertions.replay_governed_initial_write(
                assertion,
                source_occurrences=source_occurrences,
                operation_id=operation_id,
            )
            if replay is not None:
                report, written = replay
                return GovernedAssertionWriteResult(report, written)
            if any(item.assertion_id == assertion.assertion_id for item in current):
                raise SemanticValidationStoreError("governed initial write cannot replace an existing assertion")
            post_state = (*current, assertion)
            data_graph, focus_map, _ = _canonical_validation_graph(post_state, post_state)
            report = self._validator.validate(
                data_graph,
                tenant_id=self._assertions.tenant_id,
                assertion_ids=(assertion.assertion_id,),
                shape_set=shape_set,
                validation_capability=validation_capability,
                profile_version=profile_version,
                allow_experimental=allow_experimental,
                source=source,
                checkpoint_generation=checkpoint.generation,
                run_id=run_id,
                focus_assertion_ids=focus_map,
                limits=limits,
            )
            if report.action not in (
                ValidationWriteAction.ACCEPT,
                ValidationWriteAction.ACCEPT_WITH_REPORT,
            ):
                # The candidate has not crossed the canonical boundary.  A
                # retained reject/quarantine report is useful to the owning
                # tenant, but it must not preserve an unaccepted assertion
                # identity or validation result keyed to that identity.
                report = report.without_assertion_identity()
                await self._reports.persist(report)
                return GovernedAssertionWriteResult(report, None)
            try:
                written = await self._assertions.put_assertion_with_validation_report(
                    assertion,
                    report,
                    source_occurrences=source_occurrences,
                    operation_id=operation_id,
                    expected_generation=checkpoint.generation,
                )
            except AssertionConflictError:
                if attempt + 1 == max_commit_retries:
                    raise SemanticValidationStoreError(
                        "validation conflict: canonical generation changed before the bounded commit retry"
                    ) from None
                continue
            if written.idempotent:
                replay = await self._assertions.replay_governed_initial_write(
                    assertion,
                    source_occurrences=source_occurrences,
                    operation_id=operation_id,
                )
                if replay is None:
                    raise SemanticValidationStoreError(
                        "idempotent governed write receipt is missing its validation report"
                    )
                report, written = replay
            return GovernedAssertionWriteResult(report, written)
        raise AssertionError("bounded validation retry loop exhausted without returning")

    async def supersede_assertion(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences=(),
        source: ValidationSource = ValidationSource.ASSERTED,
        shape_set: ShapeSetReference = ShapeSetReference("kestrel-assertion-shapes", "1.0.0"),
        validation_capability: str = "validation-profile:shacl-core-20170720",
        profile_version: str | None = None,
        allow_experimental: bool = False,
        limits: ShaclValidationLimits = ShaclValidationLimits(),
        operation_id: str | None = None,
        run_id: str | None = None,
        max_commit_retries: int = 2,
    ) -> GovernedAssertionSupersessionResult:
        """Validate a complete tentative replacement state before superseding.

        Supersession is an assertion write, not merely a lifecycle operation:
        the resulting active graph can affect SHACL constraints on unrelated
        assertions sharing the replacement's RDF focus node.  The generation
        check below prevents committing a report against a changed graph.
        """
        if not isinstance(replacement, Assertion):
            raise SemanticValidationStoreError("governed supersession requires a canonical Assertion replacement")
        if replacement.tenant_id != self._assertions.tenant_id:
            raise TenantIsolationError("replacement assertion tenant does not match the bound tenant")
        if type(max_commit_retries) is not int or not 1 <= max_commit_retries <= 10:
            raise SemanticValidationStoreError("max_commit_retries must be an integer in [1, 10]")
        replay = await self._assertions.replay_governed_supersession(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )
        if replay is not None:
            report, written = replay
            return GovernedAssertionSupersessionResult(report, written)
        for attempt in range(max_commit_retries):
            checkpoint, _ = await self._assertions.export_snapshot()
            # As with initial writes, a completed matching operation is
            # authoritative before lifecycle planning rejects a predecessor
            # that a concurrent delivery has already superseded.
            replay = await self._assertions.replay_governed_supersession(
                expected_predecessor_revision_id,
                replacement,
                source_occurrences=source_occurrences,
                operation_id=operation_id,
            )
            if replay is not None:
                report, written = replay
                return GovernedAssertionSupersessionResult(report, written)
            try:
                plan = await self._assertions.plan_supersession_lifecycle(
                    expected_predecessor_revision_id,
                    replacement,
                )
            except AssertionConflictError as error:
                replay = await self._assertions.replay_governed_supersession(
                    expected_predecessor_revision_id,
                    replacement,
                    source_occurrences=source_occurrences,
                    operation_id=operation_id,
                )
                if replay is not None:
                    report, written = replay
                    return GovernedAssertionSupersessionResult(report, written)
                raise SemanticValidationStoreError(
                    "governed supersession requires an active predecessor in the bound tenant"
                ) from error
            # ``plan_supersession_lifecycle`` reads the alternate-proof ledger
            # as well as canonical lineage.  Reject a plan assembled across a
            # changed generation before validating it; the next bounded retry
            # derives both the report graph and eventual mutation from one
            # current lifecycle decision.
            if (await self._assertions.checkpoint()).generation != checkpoint.generation:
                if attempt + 1 == max_commit_retries:
                    raise SemanticValidationStoreError(
                        "validation conflict: canonical generation changed before the bounded supersession retry"
                    )
                continue
            data_graph, focus_map, _ = _canonical_validation_graph(
                plan.post_state,
                plan.post_state,
            )
            report = self._validator.validate(
                data_graph,
                tenant_id=self._assertions.tenant_id,
                assertion_ids=(replacement.assertion_id,),
                shape_set=shape_set,
                validation_capability=validation_capability,
                profile_version=profile_version,
                allow_experimental=allow_experimental,
                source=source,
                checkpoint_generation=checkpoint.generation,
                run_id=run_id,
                focus_assertion_ids=focus_map,
                limits=limits,
            )
            if report.action not in (
                ValidationWriteAction.ACCEPT,
                ValidationWriteAction.ACCEPT_WITH_REPORT,
            ):
                await self._reports.persist(report.without_assertion_identity())
                return GovernedAssertionSupersessionResult(report.without_assertion_identity(), None)
            try:
                written = await self._assertions.supersede_with_validation_report(
                    expected_predecessor_revision_id,
                    replacement,
                    report,
                    source_occurrences=source_occurrences,
                    operation_id=operation_id,
                    expected_generation=checkpoint.generation,
                )
            except AssertionConflictError:
                if attempt + 1 == max_commit_retries:
                    raise SemanticValidationStoreError(
                        "validation conflict: canonical generation changed before the bounded supersession retry"
                    ) from None
                continue
            if written.idempotent:
                replay = await self._assertions.replay_governed_supersession(
                    expected_predecessor_revision_id,
                    replacement,
                    source_occurrences=source_occurrences,
                    operation_id=operation_id,
                )
                if replay is None:
                    raise SemanticValidationStoreError(
                        "idempotent governed supersession receipt is missing its validation report"
                    )
                report, written = replay
            return GovernedAssertionSupersessionResult(report, written)
        raise AssertionError("bounded validation supersession retry loop exhausted without returning")

    async def append_assertion_source(
        self,
        expected_predecessor_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences=(),
        **validation_options,
    ) -> GovernedAssertionSupersessionResult:
        """Append one direct source through a validated canonical revision.

        Source provenance is immutable evidence, not an auxiliary side table.
        A distinct explicit invocation therefore creates a new revision with
        the old occurrence set plus one new occurrence.  The normal governed
        supersession path supplies its atomic lifecycle, validation report,
        operation-ledger replay, change event, and dependent invalidation.
        """
        operation_id = validation_options.get("operation_id")
        replay = await self._assertions.replay_governed_supersession(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )
        if replay is not None:
            report, written = replay
            return GovernedAssertionSupersessionResult(report, written)
        await self._assertions.validate_source_append(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
        )
        return await self.supersede_assertion(
            expected_predecessor_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            **validation_options,
        )

    async def _restore_explicit_fact_assertion(
        self,
        expected_terminal_revision_id: str,
        replacement: Assertion,
        *,
        source_occurrences=(),
        source: ValidationSource = ValidationSource.ASSERTED,
        shape_set: ShapeSetReference = ShapeSetReference(
            "kestrel-assertion-shapes",
            "1.0.0",
        ),
        validation_capability: str = (
            "validation-profile:shacl-core-20170720"
        ),
        profile_version: str | None = None,
        allow_experimental: bool = False,
        limits: ShaclValidationLimits = ShaclValidationLimits(),
        operation_id: str | None = None,
        run_id: str | None = None,
        max_commit_retries: int = 2,
    ) -> GovernedAssertionSupersessionResult:
        """Validate fresh direct evidence over an immutable terminal shell."""
        if not isinstance(replacement, Assertion):
            raise SemanticValidationStoreError(
                "governed restoration requires a canonical Assertion replacement"
            )
        if replacement.tenant_id != self._assertions.tenant_id:
            raise TenantIsolationError(
                "restoration assertion tenant does not match the bound tenant"
            )
        if type(max_commit_retries) is not int or not (
            1 <= max_commit_retries <= 10
        ):
            raise SemanticValidationStoreError(
                "max_commit_retries must be an integer in [1, 10]"
            )
        replay = await self._assertions.replay_governed_restoration(
            expected_terminal_revision_id,
            replacement,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )
        if replay is not None:
            report, written = replay
            return GovernedAssertionSupersessionResult(report, written)
        for attempt in range(max_commit_retries):
            checkpoint, _ = await self._assertions.export_snapshot()
            replay = await self._assertions.replay_governed_restoration(
                expected_terminal_revision_id,
                replacement,
                source_occurrences=source_occurrences,
                operation_id=operation_id,
            )
            if replay is not None:
                report, written = replay
                return GovernedAssertionSupersessionResult(report, written)
            try:
                plan = await self._assertions.plan_restoration_lifecycle(
                    expected_terminal_revision_id,
                    replacement,
                )
            except AssertionConflictError as error:
                replay = await self._assertions.replay_governed_restoration(
                    expected_terminal_revision_id,
                    replacement,
                    source_occurrences=source_occurrences,
                    operation_id=operation_id,
                )
                if replay is not None:
                    report, written = replay
                    return GovernedAssertionSupersessionResult(report, written)
                raise SemanticValidationStoreError(
                    "governed restoration requires a current terminal "
                    "direct predecessor"
                ) from error
            if (
                await self._assertions.checkpoint()
            ).generation != checkpoint.generation:
                if attempt + 1 == max_commit_retries:
                    raise SemanticValidationStoreError(
                        "validation conflict: canonical generation changed "
                        "before the bounded restoration retry"
                    )
                continue
            data_graph, focus_map, _ = _canonical_validation_graph(
                plan.post_state,
                plan.post_state,
            )
            report = self._validator.validate(
                data_graph,
                tenant_id=self._assertions.tenant_id,
                assertion_ids=(replacement.assertion_id,),
                shape_set=shape_set,
                validation_capability=validation_capability,
                profile_version=profile_version,
                allow_experimental=allow_experimental,
                source=source,
                checkpoint_generation=checkpoint.generation,
                run_id=run_id,
                focus_assertion_ids=focus_map,
                limits=limits,
            )
            if report.action not in (
                ValidationWriteAction.ACCEPT,
                ValidationWriteAction.ACCEPT_WITH_REPORT,
            ):
                sanitized = report.without_assertion_identity()
                await self._reports.persist(sanitized)
                return GovernedAssertionSupersessionResult(sanitized, None)
            try:
                written = (
                    await self._assertions.restore_with_validation_report(
                        expected_terminal_revision_id,
                        replacement,
                        report,
                        source_occurrences=source_occurrences,
                        operation_id=operation_id,
                        expected_generation=checkpoint.generation,
                    )
                )
            except AssertionConflictError:
                if attempt + 1 == max_commit_retries:
                    raise SemanticValidationStoreError(
                        "validation conflict: canonical generation changed "
                        "before the bounded restoration retry"
                    ) from None
                continue
            if written.idempotent:
                replay = await self._assertions.replay_governed_restoration(
                    expected_terminal_revision_id,
                    replacement,
                    source_occurrences=source_occurrences,
                    operation_id=operation_id,
                )
                if replay is None:
                    raise SemanticValidationStoreError(
                        "idempotent governed restoration receipt is missing "
                        "its validation report"
                    )
                report, written = replay
            return GovernedAssertionSupersessionResult(report, written)
        raise AssertionError(
            "bounded validation restoration retry loop exhausted"
        )

    async def full_audit_and_repair(
        self,
        *,
        shape_set: ShapeSetReference = ShapeSetReference("kestrel-assertion-shapes", "1.0.0"),
        validation_capability: str = "validation-profile:shacl-core-20170720",
        profile_version: str | None = None,
        allow_experimental: bool = False,
        limits: ShaclValidationLimits = ShaclValidationLimits(),
        run_id: str | None = None,
    ) -> ShaclValidationReport:
        """Explicit full audit/repair entry point; incomplete work quarantines."""
        return await self.validate_current(
            assertion_ids=None,
            shape_set=shape_set,
            validation_capability=validation_capability,
            profile_version=profile_version,
            allow_experimental=allow_experimental,
            limits=limits,
            run_id=run_id,
        )

    async def _persist_and_quarantine_failed_current_assertions(
        self,
        report: ShaclValidationReport,
        snapshot_revisions: Mapping[str, str],
    ) -> None:
        await self._assertions.persist_validation_report_and_quarantine(
            report,
            # The assertion store derives report targets once, then requires a
            # snapshot revision for every one.  Passing the complete snapshot
            # lets an incomplete finding fail safe to report.assertion_ids.
            expected_revisions=snapshot_revisions,
        )


def _canonical_validation_graph(
    assertions: Sequence[Assertion],
    affected: Sequence[Assertion],
) -> tuple[Graph, Mapping[URIRef, tuple[str, ...]], set[URIRef]]:
    """Materialize a read-only RDF view without storing its private values."""
    graph = Graph()
    focus_map: dict[URIRef, set[str]] = {}
    affected_nodes: set[URIRef] = set()
    affected_ids = {assertion.assertion_id for assertion in affected}
    revision_predicate = URIRef(_KESTREL + "AssertionRevision")
    source_occurrence_predicate = URIRef(_KESTREL + "SourceOccurrence")
    has_source = URIRef(_KESTREL + "hasSourceOccurrence")
    derivation_predicate = URIRef(_KESTREL + "derivation")
    input_membership = URIRef(_KESTREL + "inputMembership")
    input_revision_id = URIRef(_KESTREL + "inputRevisionId")
    was_derived_from = URIRef(_PROV + "wasDerivedFrom")
    activity = URIRef(_PROV + "Activity")
    for assertion in assertions:
        revision = _revision_node(assertion)
        graph.add((revision, RDF.type, revision_predicate))
        subject = URIRef(assertion.subject.value)
        predicate = URIRef(assertion.predicate.value)
        object_ = _rdf_object(assertion)
        graph.add((subject, predicate, object_))
        _add_focus_assertion(focus_map, revision, assertion.assertion_id)
        _add_focus_assertion(focus_map, subject, assertion.assertion_id)
        if isinstance(object_, URIRef):
            _add_focus_assertion(focus_map, object_, assertion.assertion_id)
        if isinstance(assertion.lineage, DirectLineage):
            for source_id in assertion.lineage.source_occurrence_ids:
                source = _source_node(source_id)
                graph.add((revision, has_source, source))
                graph.add((source, RDF.type, source_occurrence_predicate))
                _add_focus_assertion(focus_map, source, assertion.assertion_id)
                if assertion.assertion_id in affected_ids:
                    affected_nodes.add(source)
        else:
            # Validation needs provenance structure, not source payloads.  The
            # stable revision links are sufficient to prove that an inferred
            # assertion has actual derivation inputs, while preserving the
            # privacy-safe no-value report contract.
            derivation = _derivation_node(assertion)
            graph.add((revision, derivation_predicate, derivation))
            graph.add((derivation, RDF.type, activity))
            _add_focus_assertion(focus_map, derivation, assertion.assertion_id)
            if assertion.assertion_id in affected_ids:
                affected_nodes.add(derivation)
            for position, input_id in enumerate(assertion.lineage.input_revision_ids):
                input_revision = _revision_node_from_id(input_id)
                member = _lineage_member_node(assertion, position)
                graph.add((revision, was_derived_from, input_revision))
                graph.add((derivation, input_membership, member))
                graph.add((member, input_revision_id, RdfLiteral(input_id)))
                _add_focus_assertion(focus_map, input_revision, assertion.assertion_id)
                _add_focus_assertion(focus_map, member, assertion.assertion_id)
                if assertion.assertion_id in affected_ids:
                    affected_nodes.update((input_revision, member))
        if assertion.assertion_id in affected_ids:
            affected_nodes.update((revision, subject))
            if isinstance(object_, URIRef):
                affected_nodes.add(object_)
    return graph, {node: tuple(sorted(assertion_ids)) for node, assertion_ids in focus_map.items()}, affected_nodes


def _revision_node(assertion: Assertion) -> URIRef:
    return _revision_node_from_id(assertion.revision_id)


def _revision_node_from_id(revision_id: str) -> URIRef:
    return URIRef(_REVISION_PREFIX + hashlib.sha256(revision_id.encode("utf-8")).hexdigest())


def _source_node(source_id: str) -> URIRef:
    return URIRef(_SOURCE_PREFIX + hashlib.sha256(source_id.encode("utf-8")).hexdigest())


def _derivation_node(assertion: Assertion) -> URIRef:
    return URIRef(
        _DERIVATION_PREFIX
        + hashlib.sha256(assertion.revision_id.encode("utf-8")).hexdigest()
    )


def _lineage_member_node(assertion: Assertion, position: int) -> URIRef:
    digest = hashlib.sha256(
        f"{assertion.revision_id}:{position}".encode("utf-8")
    ).hexdigest()
    return URIRef(_LINEAGE_MEMBER_PREFIX + digest)


def _rdf_object(assertion: Assertion) -> URIRef | RdfLiteral:
    object_ = assertion.object
    if hasattr(object_, "datatype_iri"):
        if object_.language is not None:
            # RDFLib models the RDF 1.1 language datatype implicitly from the
            # language tag and rejects receiving both parameters together.
            return RdfLiteral(object_.lexical_form, lang=object_.language)
        return RdfLiteral(
            object_.lexical_form,
            datatype=URIRef(object_.datatype_iri),
        )
    return URIRef(object_.value)


def _add_focus_assertion(
    focus_map: dict[URIRef, set[str]],
    node: URIRef,
    assertion_id: str,
) -> None:
    focus_map.setdefault(node, set()).add(assertion_id)


def _assertion_ids(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise SemanticValidationStoreError("assertion_ids must be a sequence of identifiers")
    result = tuple(values)
    if len(set(result)) != len(result) or any(not isinstance(item, str) or not item for item in result):
        raise SemanticValidationStoreError("assertion_ids must be unique non-empty strings")
    return result


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "AsyncSemanticValidationReportStore",
    "GovernedAssertionWriteResult",
    "GovernedAssertionSupersessionResult",
    "GovernedSemanticValidationService",
    "SemanticValidationStoreError",
]
