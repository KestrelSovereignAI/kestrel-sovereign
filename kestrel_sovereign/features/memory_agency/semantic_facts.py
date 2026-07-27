"""Governed lifecycle for the explicit ``save_fact`` teaching tool.

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


def _assertion(
    *,
    binding: SemanticAssertionBinding,
    mapping: FactMapping,
    source: SourceOccurrence,
    confidence: float,
) -> Assertion:
    """Build one proposal after the privacy wrapper has bound its metadata."""
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
        asserted_at=source.received_at,
        ontology_version=mapping.ontology,
        lineage=DirectLineage((source.source_occurrence_id,)),
        privacy_classification=binding.privacy_classification,
        release_policy_reference=binding.release_policy_reference,
        visibility=binding.visibility,
    )


def _disposition(report) -> str:
    return f"{report.state.value}:{report.action.value}"


def _write_receipt_from_replay(
    replay,
    *,
    source: SourceOccurrence,
    operation_id: str,
) -> FactWriteReceipt:
    if getattr(replay, "terminal_erased", False):
        return FactWriteReceipt(
            saved=False,
            assertion_id=None,
            revision_id=None,
            validation_disposition="erased:terminal",
            validation_report_id=None,
            provenance_reference=None,
            provenance_digest=None,
            operation_id=operation_id,
            idempotent=True,
            error=(
                "the original semantic write was physically erased and "
                "cannot be replayed"
            ),
        )
    return FactWriteReceipt(
        saved=True,
        assertion_id=replay.assertion.assertion_id,
        revision_id=replay.assertion.revision_id,
        validation_disposition=_disposition(replay.report),
        validation_report_id=replay.report.report_id,
        provenance_reference=source.source_occurrence_id,
        provenance_digest=source.content_digest,
        operation_id=operation_id,
        idempotent=True,
        superseded_assertion_id=(
            replay.predecessor.assertion_id
            if replay.predecessor is not None
            else None
        ),
    )
