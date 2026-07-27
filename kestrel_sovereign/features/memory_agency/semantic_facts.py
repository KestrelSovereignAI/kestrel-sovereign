"""Governed adapter for the explicit ``save_fact`` teaching tool.

The legacy tool accepts compact local strings.  They are not semantic terms on
their own, so this module owns the deliberately small, versioned mapping from
that legacy surface to one canonical assertion proposal.  It never writes a
property-graph node or reaches through a storage facade to a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from typing import Any

from kestrel_sovereign.knowledge import (
    Assertion,
    AssertionQuery,
    AssertionStatus,
    DirectLineage,
    EpistemicState,
    IRI,
    Literal,
    OntologyRef,
    SourceOccurrence,
    XSD_STRING,
)
from kestrel_sovereign.knowledge.registry import get_knowledge_registry
from kestrel_sovereign.agent.invocation import (
    current_invocation_provenance,
    ensure_invocation_id,
)
from kestrel_sovereign.storage.semantic_binding import SemanticAssertionBinding


FACT_ADAPTER_VERSION = "memory-agency-save-fact-v1"
"""The closed local-term and provenance grammar used by this adapter."""

ONTOLOGY_IDENTIFIER = "kestrel-vocab"
ONTOLOGY_VERSION = "1.0.0"
_USER_SUBJECT_SUFFIX = "principal:user"
_SUPPORTED_PREDICATES = {
    "preferred_deploy_region": "preferredDeployRegion",
}
_CONFIDENCE_METHOD = FACT_ADAPTER_VERSION
_CONFIDENCE_BASIS = "explicit-tool-invocation"
_SOURCE_KIND = "agent_tool_invocation"


class FactMappingError(ValueError):
    """A legacy fact input cannot be represented by this bounded adapter."""


class FactLifecycleError(ValueError):
    """The current canonical state cannot be safely changed by this adapter."""


@dataclass(frozen=True, slots=True)
class FactMapping:
    """The complete, pinned semantic representation of a legacy tool input."""

    subject: IRI
    predicate: IRI
    object: Literal
    ontology: OntologyRef


@dataclass(frozen=True, slots=True)
class FactWriteReceipt:
    """Observable outcome returned by the explicit teaching adapter."""

    saved: bool
    assertion_id: str | None
    revision_id: str | None
    validation_disposition: str
    validation_report_id: str | None
    provenance_reference: str | None
    provenance_digest: str | None
    operation_id: str
    idempotent: bool
    superseded_assertion_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FactDeleteReceipt:
    """Observable canonical lifecycle result for a fact this adapter created."""

    deleted: bool
    assertion_id: str | None
    revision_id: str | None
    provenance_reference: str | None
    operation_id: str
    idempotent: bool
    error: str | None = None


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FactMappingError(f"{field} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise FactMappingError(f"{field} must be valid Unicode text") from error
    return value


def _ontology() -> OntologyRef:
    """Resolve the exact local ontology pin and its declared predicate term."""
    resource = get_knowledge_registry().resolve(
        ONTOLOGY_IDENTIFIER,
        ONTOLOGY_VERSION,
    )
    selected_term = "kestrel:preferredDeployRegion"
    if selected_term not in resource.selected_terms:
        raise FactMappingError(
            "the pinned ontology does not declare the save_fact predicate mapping"
        )
    return OntologyRef(
        namespace=resource.namespace,
        version=str(resource.version),
        content_digest=resource.sha256,
        compatibility_profile="semantic-kb-v1",
    )


def map_legacy_fact(
    subject: object,
    predicate: object,
    value: object,
    *,
    tenant_id: str,
) -> FactMapping:
    """Map one supported legacy fact to closed, canonical semantic terms.

    Mapping is exact on purpose.  Case folding, Unicode normalization, aliases,
    and arbitrary predicate-to-IRI construction would turn a convenience tool
    into an ad-hoc ontology authoring surface.
    """
    subject = _require_text(subject, "subject")
    predicate = _require_text(predicate, "predicate")
    value = _require_text(value, "value")
    if subject != "user":
        raise FactMappingError(
            "unsupported subject; save_fact currently supports only 'user'"
        )
    try:
        predicate_term = _SUPPORTED_PREDICATES[predicate]
    except KeyError as error:
        supported = ", ".join(sorted(_SUPPORTED_PREDICATES))
        raise FactMappingError(
            f"unsupported predicate {predicate!r}; supported predicates: {supported}"
        ) from error
    ontology = _ontology()
    return FactMapping(
        subject=IRI(f"urn:kestrel:agent:{tenant_id}:{_USER_SUBJECT_SUFFIX}"),
        predicate=IRI(f"{ontology.namespace}{predicate_term}"),
        object=Literal(value, XSD_STRING),
        ontology=ontology,
    )


def _invocation_key(value: object) -> str:
    """Hash an already-validated, per-invocation opaque identity."""
    invocation_id = ensure_invocation_id(value)
    return "request:" + hashlib.sha256(invocation_id.encode("utf-8")).hexdigest()


def _operation_material(
    *,
    action: str,
    subject: str,
    predicate: str,
    value: str | None,
    confidence_requested: float | None,
    invocation_id: object,
    lifecycle_target: str | None = None,
) -> tuple[str, str]:
    material: dict[str, object] = {
        "adapter_version": FACT_ADAPTER_VERSION,
        "action": action,
        "invocation": _invocation_key(invocation_id),
        "predicate": predicate,
        "subject": subject,
    }
    if value is not None:
        material["value"] = value
    if confidence_requested is not None:
        material["confidence_requested"] = repr(confidence_requested)
    if lifecycle_target is not None:
        material["lifecycle_target"] = lifecycle_target
    digest = _canonical_digest(material)
    return f"{FACT_ADAPTER_VERSION}:{action}:{digest}", digest


def _source_for_operation(
    operation_id: str,
    digest: str,
    owning_agent_id: str,
    *,
    received_at: str | None = None,
) -> SourceOccurrence:
    """Build provenance from the trusted turn context, never tool content."""
    provenance = current_invocation_provenance()
    if provenance is None:
        source_kind = _SOURCE_KIND
        locator = f"tool:memory_agency.save_fact#{operation_id}"
        actor = owning_agent_id
        observed_at = received_at or datetime.now(timezone.utc).isoformat()
    else:
        # A request route is a truthful locator for an explicit tool call; it
        # does not pretend the fact came from a conversation-message record.
        source_kind = provenance.source_kind
        locator = (
            f"{provenance.source_locator}#tool:memory_agency.save_fact#"
            f"{operation_id}"
        )
        actor = provenance.actor or owning_agent_id
        observed_at = received_at or provenance.received_at
    return SourceOccurrence(
        source_occurrence_id=f"source:{FACT_ADAPTER_VERSION}:{digest}",
        source_kind=source_kind,
        locator=locator,
        received_at=observed_at,
        content_digest=f"sha256:{digest}",
        actor=actor,
        selector="tool-arguments",
    )


def _adapter_owned(assertion: Assertion, mapping: FactMapping) -> bool:
    return (
        assertion.confidence_method == _CONFIDENCE_METHOD
        and assertion.confidence_basis == _CONFIDENCE_BASIS
        and assertion.ontology_version == mapping.ontology
    )


def _is_adapter_source(source: SourceOccurrence) -> bool:
    """Recognize the bounded provenance grammar owned by this adapter."""
    is_legacy_tool_locator = (
        source.source_kind == _SOURCE_KIND
        and source.locator.startswith("tool:memory_agency.save_fact#")
    )
    is_http_tool_locator = (
        source.source_kind == "http_request"
        and "#tool:memory_agency.save_fact#" in source.locator
    )
    return (
        source.source_occurrence_id.startswith(
            f"source:{FACT_ADAPTER_VERSION}:"
        )
        and (is_legacy_tool_locator or is_http_tool_locator)
        and source.selector == "tool-arguments"
    )


async def _has_adapter_provenance(storage, assertion: Assertion) -> bool:
    sources = await storage.list_assertion_sources(assertion.assertion_id)
    return any(
        _is_adapter_source(source) for source in sources
    )


async def _current_for_mapping(storage, mapping: FactMapping) -> list[Assertion]:
    assertions = await storage.query_assertions(
        AssertionQuery(subject=mapping.subject, predicate=mapping.predicate)
    )
    foreign = [
        item
        for item in assertions
        if not _adapter_owned(item, mapping)
        or not await _has_adapter_provenance(storage, item)
    ]
    if foreign:
        raise FactLifecycleError(
            "the mapped subject/predicate already has a canonical assertion "
            "outside save_fact; refusing to create a competing preference"
        )
    if len(assertions) > 1:
        raise FactLifecycleError(
            "multiple current save_fact assertions exist for the mapped "
            "subject/predicate; repair canonical state before changing it"
        )
    return assertions


async def _deleted_for_mapping(storage, mapping: FactMapping) -> list[Assertion]:
    """Find a deleted adapter assertion only for a same-request replay.

    A normal forget operation resolves an active current assertion.  The
    deletion receipt leaves that assertion current-but-deleted, so a delivery
    retry can recover the original active revision and ask the canonical store
    to replay its own operation receipt.  The caller's request identity and
    that predecessor revision select the matching deletion operation when
    several historical shells exist.
    """
    assertions = await storage.query_assertions(
        AssertionQuery(
            subject=mapping.subject,
            predicate=mapping.predicate,
            statuses=(AssertionStatus.DELETED,),
        )
    )
    return [
        item
        for item in assertions
        if _adapter_owned(item, mapping)
        and await _has_adapter_provenance(storage, item)
    ]


async def _deleted_predecessor_revision(storage, deleted: Assertion) -> str:
    """Recover the active revision named by a canonical delete receipt."""
    revisions = await storage.list_assertion_revisions(deleted.assertion_id)
    active = [
        revision.revision_id
        for revision in revisions
        if revision.status is AssertionStatus.ACTIVE
    ]
    if len(active) != 1:
        raise FactLifecycleError(
            "deleted save_fact assertion has no unique active predecessor for retry"
        )
    return active[0]


async def _matching_source(storage, assertion: Assertion, source_id: str):
    for occurrence in await storage.list_assertion_sources(assertion.assertion_id):
        if occurrence.source_occurrence_id == source_id:
            return occurrence
    return None


class GovernedFactAdapter:
    """The sole explicit-fact producer over privacy-governed canonical storage."""

    def __init__(self, storage) -> None:
        self._storage = storage

    def _binding(self) -> SemanticAssertionBinding:
        binding = self._storage.semantic_assertion_binding()
        if not isinstance(binding, SemanticAssertionBinding):
            raise FactLifecycleError(
                "save_fact requires an agent-bound PrivacyEnforcingStorage binding"
            )
        return binding

    @staticmethod
    def _assertion(
        *,
        binding: SemanticAssertionBinding,
        mapping: FactMapping,
        source: SourceOccurrence,
        confidence: float,
    ) -> Assertion:
        return Assertion(
            tenant_id=binding.tenant_id,
            owning_agent_id=binding.owning_agent_id,
            subject=mapping.subject,
            predicate=mapping.predicate,
            object=mapping.object,
            revision_id=source.source_occurrence_id,
            confidence=Decimal(str(confidence)),
            confidence_method=_CONFIDENCE_METHOD,
            confidence_basis=_CONFIDENCE_BASIS,
            epistemic_state=EpistemicState.REPORTED,
            # The source occurrence owns the invocation timestamp.  Reusing it
            # for a delivery retry keeps the governed request byte-identical;
            # the canonical store still records the actual commit boundary.
            asserted_at=source.received_at,
            ontology_version=mapping.ontology,
            lineage=DirectLineage((source.source_occurrence_id,)),
            privacy_classification=binding.privacy_classification,
            release_policy_reference=binding.release_policy_reference,
            visibility=binding.visibility,
        )

    async def save(
        self,
        *,
        subject: str,
        predicate: str,
        value: str,
        confidence: float,
        confidence_requested: float | None = None,
        invocation_id: object = None,
    ) -> FactWriteReceipt:
        invocation_id = ensure_invocation_id(invocation_id)
        binding = self._binding()
        mapping = map_legacy_fact(
            subject,
            predicate,
            value,
            tenant_id=binding.tenant_id,
        )
        operation_id, digest = _operation_material(
            action="save",
            subject=subject,
            predicate=predicate,
            value=value,
            confidence_requested=(
                confidence if confidence_requested is None else confidence_requested
            ),
            invocation_id=invocation_id,
        )
        provisional_source = _source_for_operation(
            operation_id,
            digest,
            binding.owning_agent_id,
        )
        current = (await _current_for_mapping(self._storage, mapping))
        prior = current[0] if current else None
        source = provisional_source
        if prior is not None and prior.object == mapping.object:
            prior_source = await _matching_source(
                self._storage,
                prior,
                provisional_source.source_occurrence_id,
            )
            if prior_source is None:
                return FactWriteReceipt(
                    saved=True,
                    assertion_id=prior.assertion_id,
                    revision_id=prior.revision_id,
                    validation_disposition="existing_current",
                    validation_report_id=None,
                    provenance_reference=None,
                    provenance_digest=None,
                    operation_id=operation_id,
                    idempotent=True,
                )
            source = prior_source
        assertion = self._assertion(
            binding=binding,
            mapping=mapping,
            source=source,
            confidence=confidence,
        )
        if (
            prior is not None
            and prior.object == mapping.object
            and source is not provisional_source
            and prior.supersedes_revision_id is not None
        ):
            # A replayed replacement is no longer adjacent to its original
            # active predecessor: the current replacement points at the
            # superseded-history revision, which in turn records that original
            # revision.  Recover that stored request field solely to let the
            # canonical store match its idempotency receipt before lifecycle
            # validation; do not use it to drive a new mutation.
            predecessor_state = await self._storage.get_assertion_revision(
                prior.supersedes_revision_id
            )
            expected_predecessor = (
                predecessor_state.supersedes_revision_id
                if predecessor_state is not None
                else None
            )
            if expected_predecessor is None:
                raise FactLifecycleError(
                    "save_fact supersession retry cannot recover its canonical predecessor"
                )
            result = await self._storage.supersede_assertion(
                expected_predecessor,
                assertion,
                source_occurrences=(source,),
                operation_id=operation_id,
            )
            if not result.accepted:
                return FactWriteReceipt(
                    saved=False,
                    assertion_id=None,
                    revision_id=None,
                    validation_disposition=f"{result.report.state.value}:{result.report.action.value}",
                    validation_report_id=result.report.report_id,
                    provenance_reference=source.source_occurrence_id,
                    provenance_digest=source.content_digest,
                    operation_id=operation_id,
                    idempotent=False,
                    superseded_assertion_id=prior.assertion_id,
                    error="canonical validation did not accept the replacement fact",
                )
            return FactWriteReceipt(
                saved=True,
                assertion_id=result.replacement.assertion_id,
                revision_id=result.replacement.revision_id,
                validation_disposition=f"{result.report.state.value}:{result.report.action.value}",
                validation_report_id=result.report.report_id,
                provenance_reference=source.source_occurrence_id,
                provenance_digest=source.content_digest,
                operation_id=operation_id,
                idempotent=result.idempotent,
                superseded_assertion_id=result.predecessor.assertion_id,
            )
        if prior is None or prior.object == mapping.object:
            result = await self._storage.put_assertion(
                assertion,
                source_occurrences=(source,),
                operation_id=operation_id,
            )
            if not result.accepted:
                return FactWriteReceipt(
                    saved=False,
                    assertion_id=None,
                    revision_id=None,
                    validation_disposition=f"{result.report.state.value}:{result.report.action.value}",
                    validation_report_id=result.report.report_id,
                    provenance_reference=source.source_occurrence_id,
                    provenance_digest=source.content_digest,
                    operation_id=operation_id,
                    idempotent=False,
                    error="canonical validation did not accept the fact",
                )
            return FactWriteReceipt(
                saved=True,
                assertion_id=result.assertion.assertion_id,
                revision_id=result.assertion.revision_id,
                validation_disposition=f"{result.report.state.value}:{result.report.action.value}",
                validation_report_id=result.report.report_id,
                provenance_reference=source.source_occurrence_id,
                provenance_digest=source.content_digest,
                operation_id=operation_id,
                idempotent=result.idempotent,
            )

        result = await self._storage.supersede_assertion(
            prior.revision_id,
            assertion,
            source_occurrences=(source,),
            operation_id=operation_id,
        )
        if not result.accepted:
            return FactWriteReceipt(
                saved=False,
                assertion_id=None,
                revision_id=None,
                validation_disposition=f"{result.report.state.value}:{result.report.action.value}",
                validation_report_id=result.report.report_id,
                provenance_reference=source.source_occurrence_id,
                provenance_digest=source.content_digest,
                operation_id=operation_id,
                idempotent=False,
                superseded_assertion_id=prior.assertion_id,
                error="canonical validation did not accept the replacement fact",
            )
        return FactWriteReceipt(
            saved=True,
            assertion_id=result.replacement.assertion_id,
            revision_id=result.replacement.revision_id,
            validation_disposition=f"{result.report.state.value}:{result.report.action.value}",
            validation_report_id=result.report.report_id,
            provenance_reference=source.source_occurrence_id,
            provenance_digest=source.content_digest,
            operation_id=operation_id,
            idempotent=result.idempotent,
            superseded_assertion_id=result.predecessor.assertion_id,
        )

    async def forget(
        self,
        *,
        subject: str,
        predicate: str,
        invocation_id: object = None,
    ) -> FactDeleteReceipt:
        invocation_id = ensure_invocation_id(invocation_id)
        binding = self._binding()
        mapping = map_legacy_fact(
            subject,
            predicate,
            "forgetting-target",
            tenant_id=binding.tenant_id,
        )
        current = await _current_for_mapping(self._storage, mapping)
        if not current:
            deleted = await _deleted_for_mapping(self._storage, mapping)
            # A subject/predicate may have more than one historical deletion.
            # The request-bound operation id includes the active predecessor,
            # so ask the canonical lifecycle to replay only the one whose
            # ledger receipt matches this invocation.  A non-matching deleted
            # shell cannot mutate: ``delete_assertion`` requires an ACTIVE
            # expected revision before it can create any new state.
            for target in deleted:
                sources = await self._storage.list_assertion_sources(
                    target.assertion_id
                )
                source = next(
                    (
                        item
                        for item in sources
                        if _is_adapter_source(item)
                    ),
                    None,
                )
                if source is None:
                    raise FactLifecycleError(
                        "deleted assertion lacks save_fact provenance and cannot be replayed"
                    )
                predecessor_revision = await _deleted_predecessor_revision(
                    self._storage, target
                )
                operation_id, _ = _operation_material(
                    action="forget",
                    subject=subject,
                    predicate=predicate,
                    value=None,
                    confidence_requested=None,
                    invocation_id=invocation_id,
                    lifecycle_target=predecessor_revision,
                )
                try:
                    result = await self._storage.delete_assertion(
                        target.assertion_id,
                        predecessor_revision,
                        operation_id=operation_id,
                    )
                except ValueError:
                    # The canonical lifecycle rejects a deleted shell whose
                    # operation receipt does not match this retry.  It cannot
                    # mutate because the supplied predecessor is no longer
                    # ACTIVE; continue only to another deleted shell.
                    continue
                if not result.idempotent:
                    raise FactLifecycleError(
                        "a deleted save_fact assertion unexpectedly accepted a new deletion"
                    )
                return FactDeleteReceipt(
                    deleted=True,
                    assertion_id=result.deleted.assertion_id,
                    revision_id=result.deleted.revision_id,
                    provenance_reference=source.source_occurrence_id,
                    operation_id=operation_id,
                    idempotent=result.idempotent,
                )
            operation_id, _ = _operation_material(
                action="forget",
                subject=subject,
                predicate=predicate,
                value=None,
                confidence_requested=None,
                invocation_id=invocation_id,
            )
            return FactDeleteReceipt(
                deleted=False,
                assertion_id=None,
                revision_id=None,
                provenance_reference=None,
                operation_id=operation_id,
                idempotent=True,
                error="no current save_fact assertion exists for the mapped subject/predicate",
            )
        target = current[0]
        operation_id, _ = _operation_material(
            action="forget",
            subject=subject,
            predicate=predicate,
            value=None,
            confidence_requested=None,
            invocation_id=invocation_id,
            lifecycle_target=target.revision_id,
        )
        sources = await self._storage.list_assertion_sources(target.assertion_id)
        adapter_sources = [
            source
            for source in sources
            if _is_adapter_source(source)
        ]
        if not adapter_sources:
            raise FactLifecycleError(
                "current assertion lacks save_fact provenance and cannot be deleted by this tool"
            )
        result = await self._storage.delete_assertion(
            target.assertion_id,
            target.revision_id,
            operation_id=operation_id,
        )
        return FactDeleteReceipt(
            deleted=True,
            assertion_id=result.deleted.assertion_id,
            revision_id=result.deleted.revision_id,
            provenance_reference=adapter_sources[0].source_occurrence_id,
            operation_id=operation_id,
            idempotent=result.idempotent,
        )
