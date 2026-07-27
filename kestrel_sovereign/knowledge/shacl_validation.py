"""Governed, offline SHACL validation for canonical semantic datasets.

The production validator implements the 2017 SHACL Core subset represented by
the selected, pinned shape set.  It does not execute SPARQL, rules, JavaScript,
agent instructions, intents, or any other shape-supplied program.  Draft SHACL
1.2 selections are capability-gated experiments; unsupported constructs return
an *incomplete* report instead of being treated as conformant.

Validation reports intentionally contain stable condition codes rather than
RDF values or shape-provided messages.  This makes them useful for a tenant's
audit trail without becoming a route for private graph values into logs or a
different agent's context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
import time
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

from rdflib import BNode, Graph, Literal, RDF, URIRef
from rdflib.exceptions import ParserError
from rdflib.namespace import SH, XSD

from .registry import (
    ArtifactPin,
    ExperimentalCapabilityError,
    KnowledgeRegistryError,
    ResourceKind,
    SemanticKnowledgeRegistry,
    SemanticResource,
    SemanticVersion,
    get_knowledge_registry,
)


_RDF_FIRST = RDF.first
_RDF_NIL = RDF.nil
_RDF_REST = RDF.rest
_SH = str(SH)
_TRUE = Literal(True)
_KIND_IRI = URIRef(_SH + "IRI")
_KIND_BLANK = URIRef(_SH + "BlankNode")
_KIND_LITERAL = URIRef(_SH + "Literal")
_KIND_BLANK_OR_IRI = URIRef(_SH + "BlankNodeOrIRI")
_KIND_BLANK_OR_LITERAL = URIRef(_SH + "BlankNodeOrLiteral")
_KIND_IRI_OR_LITERAL = URIRef(_SH + "IRIOrLiteral")
_EXPERIMENTAL_TERMS = frozenset(
    URIRef(_SH + value)
    for value in (
        "ReifierShape",
        "reifierShape",
        "nodeExpression",
        "expression",
        "agentInstruction",
        "intent",
    )
)
_EXECUTABLE_TERMS = frozenset(
    URIRef(_SH + value)
    for value in (
        "sparql",
        "rule",
        "rules",
        "js",
        "jsFunctionName",
        "select",
        "ask",
        "construct",
        "SPARQLConstraint",
        "SPARQLSelectValidator",
        "SPARQLAskValidator",
        "SPARQLExecutable",
    )
)


class ShaclValidationError(ValueError):
    """A requested validation profile or shape set is not safe to run."""


class ShaclCapabilityUnavailable(ShaclValidationError):
    """The exact requested SHACL capability is unavailable in this runtime."""


class ShaclSnapshotMismatch(ShaclValidationError):
    """A caller requested a profile or shape-set version other than its pin."""


class ValidationState(str, Enum):
    """A report's completion/conformance state.

    ``INCOMPLETE`` is deliberately distinct from nonconformance: it says the
    validator did not finish its bounded work and can never be used as proof of
    conformance.
    """

    CONFORMS = "conforms"
    NONCONFORMANT = "nonconformant"
    INCOMPLETE = "incomplete"


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"


class ValidationSource(str, Enum):
    ASSERTED = "asserted"
    IMPORTED = "imported"
    INFERRED = "inferred"
    REVALIDATION = "revalidation"


class ValidationWriteAction(str, Enum):
    ACCEPT = "accept"
    ACCEPT_WITH_REPORT = "accept-with-report"
    REJECT = "reject"
    QUARANTINE = "quarantine"


@dataclass(frozen=True, slots=True)
class ShaclValidationLimits:
    """Hard caps for one validation run; values are intentionally finite."""

    max_graph_triples: int = 100_000
    max_shape_triples: int = 20_000
    max_shape_depth: int = 32
    max_results: int = 1_000
    max_path_nodes: int = 100_000
    max_wall_time_seconds: float = 5.0

    def __post_init__(self) -> None:
        for field_name in (
            "max_graph_triples",
            "max_shape_triples",
            "max_shape_depth",
            "max_results",
            "max_path_nodes",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ShaclValidationError(f"{field_name} must be a positive integer")
        if (
            not isinstance(self.max_wall_time_seconds, (int, float))
            or not math.isfinite(self.max_wall_time_seconds)
            or self.max_wall_time_seconds <= 0
        ):
            raise ShaclValidationError("max_wall_time_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ShapeSetReference:
    """An exact shape-set pin; version omission is never interpreted as latest."""

    identifier: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ShaclValidationError("shape-set identifier must be non-empty")
        SemanticVersion.parse(self.version)


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """A privacy-safe result row with no copied RDF value or message text."""

    severity: ValidationSeverity
    code: str
    shape_id: str | None = None
    focus_assertion_id: str | None = None
    path: str | None = None
    focus_assertion_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ShaclValidationError("validation finding code must be non-empty")
        for field_name in ("shape_id", "focus_assertion_id", "path"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ShaclValidationError(f"{field_name} must be a non-empty string when present")
        if (
            len(set(self.focus_assertion_ids)) != len(self.focus_assertion_ids)
            or any(not isinstance(item, str) or not item for item in self.focus_assertion_ids)
        ):
            raise ShaclValidationError("focus_assertion_ids must be unique non-empty strings")
        if (
            self.focus_assertion_id is not None
            and self.focus_assertion_ids
            and self.focus_assertion_id not in self.focus_assertion_ids
        ):
            raise ShaclValidationError(
                "focus_assertion_id must be included in focus_assertion_ids when both are present"
            )

    @property
    def affected_assertion_ids(self) -> tuple[str, ...]:
        """All canonical assertions represented by this RDF focus node.

        ``focus_assertion_id`` remains a compact compatibility field for
        consumers that can display one ID, but lifecycle repair must always
        use this complete set.  Multiple assertions can validly share one RDF
        subject, object, or other SHACL focus node.
        """
        if self.focus_assertion_ids:
            return self.focus_assertion_ids
        return (self.focus_assertion_id,) if self.focus_assertion_id is not None else ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "shape_id": self.shape_id,
            "focus_assertion_id": self.focus_assertion_id,
            "focus_assertion_ids": list(self.focus_assertion_ids),
            "path": self.path,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ValidationFinding":
        focus_assertion_id = _optional_text(value.get("focus_assertion_id"))
        focus_assertion_ids = value.get("focus_assertion_ids")
        if focus_assertion_ids is None:
            normalized_focus_assertion_ids = (
                (focus_assertion_id,) if focus_assertion_id is not None else ()
            )
        elif isinstance(focus_assertion_ids, Sequence) and not isinstance(
            focus_assertion_ids, (str, bytes)
        ):
            normalized_focus_assertion_ids = tuple(focus_assertion_ids)
        else:
            raise ShaclValidationError("focus_assertion_ids must be a sequence when present")
        return cls(
            severity=ValidationSeverity(str(value["severity"])),
            code=str(value["code"]),
            shape_id=_optional_text(value.get("shape_id")),
            focus_assertion_id=focus_assertion_id,
            focus_assertion_ids=normalized_focus_assertion_ids,
            path=_optional_text(value.get("path")),
        )


@dataclass(frozen=True, slots=True)
class ShaclValidationReport:
    """Versioned, tenant-scoped result of one fixed-profile validation run."""

    report_id: str
    tenant_id: str
    assertion_ids: tuple[str, ...]
    shape_set: ShapeSetReference
    validation_profile: ArtifactPin
    shape_set_pin: ArtifactPin
    checkpoint_generation: int | None
    run_id: str
    evaluated_at: str
    state: ValidationState
    action: ValidationWriteAction
    source: ValidationSource
    findings: tuple[ValidationFinding, ...] = ()
    report_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.report_id, str) or not self.report_id:
            raise ShaclValidationError("report_id must be non-empty")
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise ShaclValidationError("tenant_id must be non-empty")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ShaclValidationError("run_id must be non-empty")
        if self.checkpoint_generation is not None and (
            type(self.checkpoint_generation) is not int or self.checkpoint_generation < 0
        ):
            raise ShaclValidationError("checkpoint_generation must be a non-negative integer")
        if type(self.report_version) is not int or self.report_version != 1:
            raise ShaclValidationError("unsupported SHACL validation report version")
        if not isinstance(self.source, ValidationSource):
            raise ShaclValidationError("validation report source must be a ValidationSource")
        if len(set(self.assertion_ids)) != len(self.assertion_ids) or any(
            not isinstance(item, str) or not item for item in self.assertion_ids
        ):
            raise ShaclValidationError("assertion_ids must be unique non-empty strings")
        if self.state is ValidationState.CONFORMS and self.action not in (
            ValidationWriteAction.ACCEPT,
            ValidationWriteAction.ACCEPT_WITH_REPORT,
        ):
            raise ShaclValidationError("a conformant report cannot reject or quarantine")
        if self.state is ValidationState.NONCONFORMANT and self.action not in (
            ValidationWriteAction.REJECT,
            ValidationWriteAction.QUARANTINE,
        ):
            raise ShaclValidationError("a nonconformant report cannot be accepted")
        if self.state is ValidationState.INCOMPLETE and self.action in (
            ValidationWriteAction.ACCEPT,
            ValidationWriteAction.ACCEPT_WITH_REPORT,
        ):
            raise ShaclValidationError("incomplete validation must never be reported as acceptance")

    @property
    def conforms(self) -> bool:
        return self.state is ValidationState.CONFORMS

    def to_mapping(self) -> dict[str, object]:
        return {
            "report_version": self.report_version,
            "report_id": self.report_id,
            "tenant_id": self.tenant_id,
            "assertion_ids": list(self.assertion_ids),
            "shape_set": {"identifier": self.shape_set.identifier, "version": self.shape_set.version},
            "validation_profile": _pin_mapping(self.validation_profile),
            "shape_set_pin": _pin_mapping(self.shape_set_pin),
            "checkpoint_generation": self.checkpoint_generation,
            "run_id": self.run_id,
            "evaluated_at": self.evaluated_at,
            "state": self.state.value,
            "action": self.action.value,
            "source": self.source.value,
            "findings": [finding.to_mapping() for finding in self.findings],
        }

    def without_assertion_identity(self) -> "ShaclValidationReport":
        """Remove speculative identifiers before retaining a failed proposal.

        A rejected pre-publication proposal has no canonical assertion identity.
        Its report may be retained as a tenant-private quarantine artifact, but
        it must not turn an unaccepted deterministic candidate ID into durable
        canonical-looking state.
        """
        return ShaclValidationReport(
            report_id=self.report_id,
            tenant_id=self.tenant_id,
            assertion_ids=(),
            shape_set=self.shape_set,
            validation_profile=self.validation_profile,
            shape_set_pin=self.shape_set_pin,
            checkpoint_generation=self.checkpoint_generation,
            run_id=self.run_id,
            evaluated_at=self.evaluated_at,
            state=self.state,
            action=self.action,
            source=self.source,
            findings=tuple(
                ValidationFinding(
                    severity=finding.severity,
                    code=finding.code,
                    shape_id=finding.shape_id,
                    path=finding.path,
                )
                for finding in self.findings
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ShaclValidationReport":
        shape_set = value.get("shape_set")
        profile = value.get("validation_profile")
        shape_pin = value.get("shape_set_pin")
        findings = value.get("findings", ())
        if not isinstance(shape_set, Mapping) or not isinstance(profile, Mapping) or not isinstance(shape_pin, Mapping):
            raise ShaclValidationError("validation report is missing pinned profile metadata")
        if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)):
            raise ShaclValidationError("validation report findings must be a sequence")
        if any(not isinstance(item, Mapping) for item in findings):
            raise ShaclValidationError("validation report findings contain malformed entries")
        assertion_ids = value.get("assertion_ids", ())
        if not isinstance(assertion_ids, Sequence) or isinstance(assertion_ids, (str, bytes)):
            raise ShaclValidationError("validation report assertion_ids must be a sequence")
        return cls(
            report_version=int(value["report_version"]),
            report_id=str(value["report_id"]),
            tenant_id=str(value["tenant_id"]),
            assertion_ids=tuple(str(item) for item in assertion_ids),
            shape_set=ShapeSetReference(str(shape_set["identifier"]), str(shape_set["version"])),
            validation_profile=_pin_from_mapping(profile),
            shape_set_pin=_pin_from_mapping(shape_pin),
            checkpoint_generation=(
                None if value.get("checkpoint_generation") is None else int(value["checkpoint_generation"])
            ),
            run_id=str(value["run_id"]),
            evaluated_at=str(value["evaluated_at"]),
            state=ValidationState(str(value["state"])),
            action=ValidationWriteAction(str(value["action"])),
            source=ValidationSource(str(value["source"])),
            findings=tuple(ValidationFinding.from_mapping(item) for item in findings),
        )


@dataclass(frozen=True, slots=True)
class ShaclWritePolicy:
    """The accepted writer disposition matrix for source and result severity."""

    def action_for(self, source: ValidationSource, report: ShaclValidationReport) -> ValidationWriteAction:
        if report.state is ValidationState.CONFORMS:
            return (
                ValidationWriteAction.ACCEPT_WITH_REPORT
                if report.findings
                else ValidationWriteAction.ACCEPT
            )
        # A result cap, timeout, malformed shape, or unavailable extension can
        # never prove conformance. Imports and later audits retain a tenant-local
        # quarantine record; interactive and inferred candidates are rejected.
        if source in (ValidationSource.IMPORTED, ValidationSource.REVALIDATION):
            return ValidationWriteAction.QUARANTINE
        return ValidationWriteAction.REJECT


DEFAULT_SHACL_WRITE_POLICY = ShaclWritePolicy()


class GovernedShaclValidationService:
    """Offline SHACL service with exact registry pins and bounded Core checks.

    This class is intentionally validation-only.  It cannot receive a tool
    registry, an authorization context, or an instruction executor.  Callers
    persist its reports through the tenant-bound validation repository and use
    the assertion store's normal outbox transition for any invalidation.
    """

    def __init__(
        self,
        registry: SemanticKnowledgeRegistry | None = None,
        *,
        write_policy: ShaclWritePolicy = DEFAULT_SHACL_WRITE_POLICY,
    ) -> None:
        self._registry = registry or get_knowledge_registry()
        self._write_policy = write_policy

    def validate(
        self,
        data_graph: Graph,
        *,
        tenant_id: str,
        assertion_ids: Iterable[str] = (),
        shape_set: ShapeSetReference = ShapeSetReference("kestrel-assertion-shapes", "1.0.0"),
        validation_capability: str = "validation-profile:shacl-core-20170720",
        profile_version: str | None = None,
        allow_experimental: bool = False,
        source: ValidationSource = ValidationSource.ASSERTED,
        checkpoint_generation: int | None = None,
        run_id: str | None = None,
        focus_nodes: Iterable[URIRef | BNode] | None = None,
        focus_assertion_ids: Mapping[URIRef | BNode, str | Sequence[str]] | None = None,
        limits: ShaclValidationLimits = ShaclValidationLimits(),
    ) -> ShaclValidationReport:
        """Validate a fixed graph against one exact local profile and shape set.

        ``focus_nodes`` supports incremental revalidation.  The entire data
        graph still participates in every evaluation, so cardinality and other
        cross-node constraints see the same state as an explicit full audit.
        A focus filter is used only when the selected shape graph has a
        statically provable local dependency boundary.  For inverse, sequence,
        repeated, and other non-local paths (or targets), validation falls
        back to a full audit rather than risking a false conformant report.
        """
        if not isinstance(data_graph, Graph):
            raise ShaclValidationError("data_graph must be an rdflib Graph")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ShaclValidationError("tenant_id must be non-empty")
        if not isinstance(source, ValidationSource):
            raise ShaclValidationError("source must be a ValidationSource")
        if not isinstance(limits, ShaclValidationLimits):
            raise ShaclValidationError("limits must be ShaclValidationLimits")
        assertion_id_tuple = _safe_assertion_ids(assertion_ids)
        if checkpoint_generation is not None and (
            type(checkpoint_generation) is not int or checkpoint_generation < 0
        ):
            raise ShaclValidationError("checkpoint_generation must be a non-negative integer")

        profile, shapes_resource = self._select_pins(
            validation_capability,
            profile_version=profile_version,
            shape_set=shape_set,
            allow_experimental=allow_experimental,
        )
        report_id = uuid4().hex
        effective_run_id = run_id or uuid4().hex
        findings: list[ValidationFinding] = []
        started = time.monotonic()
        focus_mapping = _normalized_focus_assertion_ids(focus_assertion_ids or {})
        requested_focus = set(focus_nodes) if focus_nodes is not None else None

        if len(data_graph) > limits.max_graph_triples:
            findings.append(_system_finding("graph_size_limit_exhausted"))
            return self._report(
                report_id, tenant_id, assertion_id_tuple, shape_set, profile, shapes_resource,
                checkpoint_generation, effective_run_id, ValidationState.INCOMPLETE, source, findings,
            )

        try:
            shapes = self._load_shapes(shapes_resource, limits)
            unsupported = _unsupported_shape_construct(shapes, profile.resource, allow_experimental)
            if unsupported is not None:
                findings.append(_system_finding(unsupported))
                return self._report(
                    report_id, tenant_id, assertion_id_tuple, shape_set, profile, shapes_resource,
                    checkpoint_generation, effective_run_id, ValidationState.INCOMPLETE, source, findings,
                )
            if requested_focus is not None and not _focus_filter_is_complete(shapes):
                # The caller's changed-node set is only complete for local
                # shapes.  In particular, a target reached through an inverse
                # or sequence path can depend on a changed assertion without
                # being among that assertion's RDF terms.  Full evaluation is
                # the only sound fallback until a complete closure is proven.
                requested_focus = None
            context = _ValidationContext(
                data_graph=data_graph,
                shapes_graph=shapes,
                limits=limits,
                started_at=started,
                findings=findings,
                focus_assertion_ids=focus_mapping,
            )
            node_shapes = tuple(sorted(
                set(shapes.subjects(RDF.type, SH.NodeShape)), key=_term_key
            ))
            property_shapes = tuple(sorted(
                set(shapes.subjects(RDF.type, SH.PropertyShape)), key=_term_key
            ))
            for shape in node_shapes:
                context.check_budget()
                for focus in _targets_for_shape(shapes, data_graph, shape):
                    if requested_focus is not None and focus not in requested_focus:
                        continue
                    context.validate_node_shape(shape, focus, depth=0)
            for shape in property_shapes:
                context.check_budget()
                for focus in _targets_for_shape(shapes, data_graph, shape):
                    if requested_focus is not None and focus not in requested_focus:
                        continue
                    context.validate_property_shape(
                        shape,
                        focus,
                        _severity(shapes.value(shape, SH.severity)),
                        depth=0,
                    )
        except _ValidationIncomplete as error:
            findings.append(_system_finding(error.code))
            state = ValidationState.INCOMPLETE
        except (ParserError, SyntaxError, ValueError, TypeError, UnicodeError):
            # Never include an arbitrary data/shape value in an exception or a
            # report.  The condition code is intentionally the entire detail.
            findings.append(_system_finding("malformed_shape"))
            state = ValidationState.INCOMPLETE
        else:
            state = (
                ValidationState.NONCONFORMANT
                if any(item.severity is ValidationSeverity.VIOLATION for item in findings)
                else ValidationState.CONFORMS
            )
        return self._report(
            report_id, tenant_id, assertion_id_tuple, shape_set, profile, shapes_resource,
            checkpoint_generation, effective_run_id, state, source, findings,
        )

    def _select_pins(
        self,
        capability: str,
        *,
        profile_version: str | None,
        shape_set: ShapeSetReference,
        allow_experimental: bool,
    ) -> tuple[object, SemanticResource]:
        try:
            profile = self._registry.select_capability(
                capability, allow_experimental=allow_experimental
            )
            if profile.resource.kind is not ResourceKind.VALIDATION_PROFILE:
                raise ShaclCapabilityUnavailable("requested capability is not a validation profile")
            if profile_version is not None and str(profile.resource.version) != profile_version:
                raise ShaclSnapshotMismatch("requested SHACL profile version does not match its registry pin")
            shapes = self._registry.resolve_capability(
                shape_set.identifier,
                shape_set.version,
                allow_experimental=allow_experimental,
            )
        except ExperimentalCapabilityError as error:
            raise ShaclCapabilityUnavailable(
                "experimental SHACL capability requires explicit allow_experimental=True"
            ) from error
        except KnowledgeRegistryError as error:
            raise ShaclCapabilityUnavailable("requested SHACL capability or shape set is unavailable") from error
        if shapes.resource.kind is not ResourceKind.SHAPE_SET:
            raise ShaclCapabilityUnavailable("requested semantic resource is not a SHACL shape set")
        # A stable shape set must explicitly import the selected stable Core
        # baseline.  An experiment may import an experimental profile, but it
        # cannot silently run under the stable default.
        closure_keys = {(item.identifier, str(item.version)) for item in shapes.import_closure}
        if (profile.resource.identifier, str(profile.resource.version)) not in closure_keys:
            raise ShaclSnapshotMismatch("shape-set pin does not import the selected SHACL profile")
        return profile, shapes.resource

    def _load_shapes(self, resource: SemanticResource, limits: ShaclValidationLimits) -> Graph:
        data = self._registry.read_verified_resource(resource)
        graph = Graph()
        graph.parse(data=data.decode("utf-8"), format="turtle", publicID=resource.uri)
        if len(graph) > limits.max_shape_triples:
            raise _ValidationIncomplete("shape_graph_size_limit_exhausted")
        return graph

    def _report(
        self,
        report_id: str,
        tenant_id: str,
        assertion_ids: tuple[str, ...],
        shape_set: ShapeSetReference,
        profile: object,
        shapes_resource: SemanticResource,
        checkpoint_generation: int | None,
        run_id: str,
        state: ValidationState,
        source: ValidationSource,
        findings: list[ValidationFinding],
    ) -> ShaclValidationReport:
        # The state/action need a constructed report to route policy.  Use a
        # temporary conservative action for nonconformance/incomplete first.
        provisional = ShaclValidationReport(
            report_id=report_id,
            tenant_id=tenant_id,
            assertion_ids=assertion_ids,
            shape_set=shape_set,
            validation_profile=profile.resource.pin,  # type: ignore[attr-defined]
            shape_set_pin=shapes_resource.pin,
            checkpoint_generation=checkpoint_generation,
            run_id=run_id,
            evaluated_at=_now(),
            state=state,
            action=(
                ValidationWriteAction.ACCEPT if state is ValidationState.CONFORMS else ValidationWriteAction.REJECT
            ),
            source=source,
            findings=tuple(findings),
        )
        action = self._write_policy.action_for(source, provisional)
        return ShaclValidationReport(
            report_id=provisional.report_id,
            tenant_id=provisional.tenant_id,
            assertion_ids=provisional.assertion_ids,
            shape_set=provisional.shape_set,
            validation_profile=provisional.validation_profile,
            shape_set_pin=provisional.shape_set_pin,
            checkpoint_generation=provisional.checkpoint_generation,
            run_id=provisional.run_id,
            evaluated_at=provisional.evaluated_at,
            state=provisional.state,
            action=action,
            source=provisional.source,
            findings=provisional.findings,
        )


class _ValidationIncomplete(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code


@dataclass(slots=True)
class _ValidationContext:
    data_graph: Graph
    shapes_graph: Graph
    limits: ShaclValidationLimits
    started_at: float
    findings: list[ValidationFinding]
    focus_assertion_ids: Mapping[URIRef | BNode, Sequence[str]]
    _active: set[tuple[object, object]] = field(default_factory=set)

    def check_budget(self) -> None:
        if time.monotonic() - self.started_at > self.limits.max_wall_time_seconds:
            raise _ValidationIncomplete("wall_time_limit_exhausted")
        # Reserve one bounded slot for the terminal incomplete condition so a
        # cap is observable without exceeding the promised report size.
        if len(self.findings) >= self.limits.max_results - 1:
            raise _ValidationIncomplete("report_limit_exhausted")

    def finding(
        self,
        severity: ValidationSeverity,
        code: str,
        shape: object,
        focus: object,
        path: object | None = None,
    ) -> None:
        self.check_budget()
        assertion_ids = tuple(self.focus_assertion_ids.get(focus, ()))
        self.findings.append(
            ValidationFinding(
                severity=severity,
                code=code,
                shape_id=_safe_shape_id(shape),
                focus_assertion_id=assertion_ids[0] if assertion_ids else None,
                focus_assertion_ids=assertion_ids,
                path=str(path) if isinstance(path, URIRef) else None,
            )
        )

    def validate_node_shape(self, shape: object, focus: object, *, depth: int) -> bool:
        self.check_budget()
        if depth > self.limits.max_shape_depth:
            raise _ValidationIncomplete("shape_recursion_limit_exhausted")
        marker = (shape, focus)
        if marker in self._active:
            raise _ValidationIncomplete("recursive_shape_cycle")
        if _boolean(self.shapes_graph.value(shape, SH.deactivated)):
            return True
        before = len(self.findings)
        self._active.add(marker)
        try:
            severity = _severity(self.shapes_graph.value(shape, SH.severity))
            self._validate_common_constraints(shape, focus, (focus,), severity, depth)
            for property_shape in self.shapes_graph.objects(shape, SH.property):
                self.validate_property_shape(
                    property_shape,
                    focus,
                    severity,
                    depth + 1,
                    parent_shape=shape,
                )
        finally:
            self._active.remove(marker)
        return len(self.findings) == before

    def validate_property_shape(
        self,
        shape: object,
        focus: object,
        inherited_severity: ValidationSeverity,
        depth: int,
        *,
        parent_shape: object | None = None,
    ) -> bool:
        self.check_budget()
        if depth > self.limits.max_shape_depth:
            raise _ValidationIncomplete("shape_recursion_limit_exhausted")
        # sh:deactivated applies to property shapes just as it applies to node
        # shapes.  It must short-circuit before evaluating the path: a disabled
        # shape is permitted to contain an otherwise malformed or unavailable
        # path without affecting conformance.
        if _boolean(self.shapes_graph.value(shape, SH.deactivated)):
            return True
        path = self.shapes_graph.value(shape, SH.path)
        if path is None:
            raise ValueError("property shape has no path")
        values = tuple(
            _path_values(
                self.data_graph,
                focus,
                path,
                self.shapes_graph,
                self.limits.max_shape_depth,
                self.limits.max_path_nodes,
            )
        )
        severity = _severity(self.shapes_graph.value(shape, SH.severity), inherited_severity)
        before = len(self.findings)
        self._validate_common_constraints(
            shape,
            focus,
            values,
            severity,
            depth,
            path,
            parent_shape=parent_shape,
        )
        return len(self.findings) == before

    def _validate_common_constraints(
        self,
        shape: object,
        focus: object,
        values: tuple[object, ...],
        severity: ValidationSeverity,
        depth: int,
        path: object | None = None,
        *,
        parent_shape: object | None = None,
    ) -> None:
        graph = self.shapes_graph
        count = len(values)
        min_count = _integer(graph.value(shape, SH.minCount))
        max_count = _integer(graph.value(shape, SH.maxCount))
        if min_count is not None and count < min_count:
            self.finding(severity, "min_count", shape, focus, path)
        if max_count is not None and count > max_count:
            self.finding(severity, "max_count", shape, focus, path)
        for expected in graph.objects(shape, SH.hasValue):
            if expected not in values:
                self.finding(severity, "has_value", shape, focus, path)
        allowed = graph.value(shape, SH["in"])
        if allowed is not None:
            permitted = set(_rdf_list(graph, allowed, self.limits.max_shape_depth))
            for value in values:
                if value not in permitted:
                    self.finding(severity, "in", shape, focus, path)
        datatype = graph.value(shape, SH.datatype)
        if datatype is not None:
            for value in values:
                datatype_matches = (
                    isinstance(value, Literal)
                    and (
                        value.datatype == datatype
                        # RDFLib represents an RDF 1.1 simple literal with a
                        # null datatype even though its semantic datatype is
                        # xsd:string.
                        or (value.datatype is None and datatype == XSD.string)
                    )
                )
                if not datatype_matches:
                    self.finding(severity, "datatype", shape, focus, path)
        node_kind = graph.value(shape, SH.nodeKind)
        if node_kind is not None:
            for value in values:
                if not _matches_node_kind(value, node_kind):
                    self.finding(severity, "node_kind", shape, focus, path)
        for class_ in graph.objects(shape, SH["class"]):
            for value in values:
                if (value, RDF.type, class_) not in self.data_graph:
                    self.finding(severity, "class", shape, focus, path)
        for predicate, code in ((SH.minLength, "min_length"), (SH.maxLength, "max_length")):
            bound = _integer(graph.value(shape, predicate))
            if bound is None:
                continue
            for value in values:
                length = len(str(value)) if isinstance(value, Literal) else None
                if length is None or (predicate == SH.minLength and length < bound) or (predicate == SH.maxLength and length > bound):
                    self.finding(severity, code, shape, focus, path)
        pattern = graph.value(shape, SH.pattern)
        if pattern is not None:
            import re

            flags_text = str(graph.value(shape, SH.flags) or "")
            flags = 0
            if "i" in flags_text:
                flags |= re.IGNORECASE
            if "m" in flags_text:
                flags |= re.MULTILINE
            if "s" in flags_text:
                flags |= re.DOTALL
            if "x" in flags_text:
                flags |= re.VERBOSE
            try:
                compiled = re.compile(str(pattern), flags)
            except re.error as error:
                raise ValueError("invalid pattern") from error
            for value in values:
                if not isinstance(value, Literal) or compiled.search(str(value)) is None:
                    self.finding(severity, "pattern", shape, focus, path)
        for predicate, code, compare in (
            (SH.minInclusive, "min_inclusive", lambda value, bound: value >= bound),
            (SH.maxInclusive, "max_inclusive", lambda value, bound: value <= bound),
            (SH.minExclusive, "min_exclusive", lambda value, bound: value > bound),
            (SH.maxExclusive, "max_exclusive", lambda value, bound: value < bound),
        ):
            bound = graph.value(shape, predicate)
            if bound is None:
                continue
            for value in values:
                try:
                    valid = isinstance(value, Literal) and compare(value.toPython(), bound.toPython())
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    self.finding(severity, code, shape, focus, path)
        language_in = graph.value(shape, SH.languageIn)
        if language_in is not None:
            allowed_languages = {str(item).lower() for item in _rdf_list(graph, language_in, self.limits.max_shape_depth)}
            for value in values:
                if not isinstance(value, Literal) or not value.language or value.language.lower() not in allowed_languages:
                    self.finding(severity, "language_in", shape, focus, path)
        if _boolean(graph.value(shape, SH.uniqueLang)):
            languages = [str(value.language).lower() for value in values if isinstance(value, Literal) and value.language]
            if len(languages) != len(set(languages)):
                self.finding(severity, "unique_lang", shape, focus, path)
        for predicate, code, comparator in (
            (SH.equals, "equals", lambda left, right: left == right),
            (SH.disjoint, "disjoint", lambda left, right: left.isdisjoint(right)),
            (SH.lessThan, "less_than", lambda left, right: all(_term_less(a, b) for a in left for b in right)),
            (SH.lessThanOrEquals, "less_than_or_equals", lambda left, right: all(_term_less_or_equal(a, b) for a in left for b in right)),
        ):
            other_path = graph.value(shape, predicate)
            if other_path is None:
                continue
            others = set(
                _path_values(
                    self.data_graph,
                    focus,
                    other_path,
                    graph,
                    self.limits.max_shape_depth,
                    self.limits.max_path_nodes,
                )
            )
            try:
                valid = comparator(set(values), others)
            except (TypeError, ValueError):
                valid = False
            if not valid:
                self.finding(severity, code, shape, focus, path)
        for nested in graph.objects(shape, SH.node):
            for value in values:
                if not self._probe_node_shape(nested, value, depth=depth + 1):
                    self.finding(severity, "node", shape, focus, path)
        for nested in graph.objects(shape, SH["not"]):
            for value in values:
                if self._probe_node_shape(nested, value, depth=depth + 1):
                    self.finding(severity, "not", shape, focus, path)
        for predicate, code, evaluator in (
            (SH["and"], "and", lambda results: all(results)),
            (SH["or"], "or", lambda results: any(results)),
            (SH.xone, "xone", lambda results: sum(results) == 1),
        ):
            members = graph.value(shape, predicate)
            if members is None:
                continue
            nested_shapes = _rdf_list(graph, members, self.limits.max_shape_depth)
            for value in values:
                results = [self._probe_node_shape(nested, value, depth=depth + 1) for nested in nested_shapes]
                if not evaluator(results):
                    self.finding(severity, code, shape, focus, path)
        qualified_shape = graph.value(shape, SH.qualifiedValueShape)
        if qualified_shape is not None:
            qualified_values = {
                value
                for value in values
                if self._probe_node_shape(qualified_shape, value, depth=depth + 1)
            }
            if (
                _boolean(graph.value(shape, SH.qualifiedValueShapesDisjoint))
                and parent_shape is not None
            ):
                # SHACL Core's disjoint qualified-values parameter excludes a
                # value from this property shape's qualified count when it
                # also conforms to any *sibling* qualified value shape.  The
                # predicate is not a standalone violation; it changes the
                # count used by qualifiedMinCount/qualifiedMaxCount.
                sibling_qualified_shapes = (
                    graph.value(sibling, SH.qualifiedValueShape)
                    for sibling in graph.objects(parent_shape, SH.property)
                    if sibling != shape
                )
                for sibling_shape in sibling_qualified_shapes:
                    if sibling_shape is None:
                        continue
                    qualified_values = {
                        value
                        for value in qualified_values
                        if not self._probe_node_shape(
                            sibling_shape, value, depth=depth + 1
                        )
                    }
            qualified_count = len(qualified_values)
            minimum = _integer(graph.value(shape, SH.qualifiedMinCount))
            maximum = _integer(graph.value(shape, SH.qualifiedMaxCount))
            if minimum is not None and qualified_count < minimum:
                self.finding(severity, "qualified_min_count", shape, focus, path)
            if maximum is not None and qualified_count > maximum:
                self.finding(severity, "qualified_max_count", shape, focus, path)
        if _boolean(graph.value(shape, SH.closed)):
            allowed_paths = {
                candidate
                for property_shape in graph.objects(shape, SH.property)
                if (candidate := graph.value(property_shape, SH.path)) is not None and isinstance(candidate, URIRef)
            }
            ignored = graph.value(shape, SH.ignoredProperties)
            if ignored is not None:
                allowed_paths.update(item for item in _rdf_list(graph, ignored, self.limits.max_shape_depth) if isinstance(item, URIRef))
            for _, predicate, _ in self.data_graph.triples((focus, None, None)):
                if predicate not in allowed_paths and predicate != RDF.type:
                    self.finding(severity, "closed", shape, focus, path)

    def _probe_node_shape(self, shape: object, focus: object, *, depth: int) -> bool:
        """Evaluate a nested shape without leaking trial failures into results.

        SHACL logical and qualified constraints use child conformance as a
        predicate.  A failed ``sh:not`` child or a non-qualified value is not
        itself a parent violation, so retaining those trial findings would turn
        a conformant parent shape into a false nonconformance report.
        """
        before = len(self.findings)
        try:
            return self.validate_node_shape(shape, focus, depth=depth)
        finally:
            del self.findings[before:]


def _targets_for_shape(shapes: Graph, data: Graph, shape: object) -> tuple[object, ...]:
    targets: set[object] = set()
    for class_ in shapes.objects(shape, SH.targetClass):
        targets.update(data.subjects(RDF.type, class_))
    targets.update(shapes.objects(shape, SH.targetNode))
    for predicate in shapes.objects(shape, SH.targetSubjectsOf):
        targets.update(data.subjects(predicate=predicate))
    for predicate in shapes.objects(shape, SH.targetObjectsOf):
        targets.update(data.objects(predicate=predicate))
    return tuple(sorted(targets, key=_term_key))


_LOCAL_TARGET_DECLARATIONS = frozenset((SH.targetClass, SH.targetSubjectsOf))
_NONLOCAL_TARGET_DECLARATIONS = frozenset((SH.targetNode, SH.targetObjectsOf))
_LOCAL_METADATA_PREDICATES = frozenset((RDF.type, SH.severity, SH.deactivated))
_LOCAL_VALUE_CONSTRAINT_PREDICATES = frozenset(
    (
        SH.minCount,
        SH.maxCount,
        SH.hasValue,
        SH["in"],
        SH.datatype,
        SH.nodeKind,
        SH.minLength,
        SH.maxLength,
        SH.pattern,
        SH.flags,
        SH.minInclusive,
        SH.maxInclusive,
        SH.minExclusive,
        SH.maxExclusive,
        SH.languageIn,
        SH.uniqueLang,
        SH.equals,
        SH.disjoint,
        SH.lessThan,
        SH.lessThanOrEquals,
        SH.closed,
        SH.ignoredProperties,
    )
)
_LOGICAL_SHAPE_PREDICATES = frozenset((SH["and"], SH["or"], SH.xone, SH["not"]))
_NONLOCAL_VALUE_CONSTRAINT_PREDICATES = frozenset(
    (SH["class"], SH.node, SH.qualifiedValueShape, SH.qualifiedMinCount, SH.qualifiedMaxCount)
)


def _focus_filter_is_complete(shapes: Graph) -> bool:
    """Whether filtering to changed RDF terms is provably complete.

    The incremental caller knows only the materialized terms of the changed
    assertion.  Direct, outgoing property paths and ``targetClass`` /
    ``targetSubjectsOf`` are local to those terms.  Any other target or path
    can reach a different focus node, so this deliberately narrow proof
    rejects it and lets the caller execute a full audit.
    """
    roots = set(shapes.subjects(RDF.type, SH.NodeShape))
    roots.update(shapes.subjects(RDF.type, SH.PropertyShape))
    for predicate in (*_LOCAL_TARGET_DECLARATIONS, *_NONLOCAL_TARGET_DECLARATIONS):
        roots.update(shapes.subjects(predicate=predicate))
    return all(_shape_is_local(shapes, root, set()) for root in roots)


def _shape_is_local(shapes: Graph, shape: object, active: set[object]) -> bool:
    """Recognize the strict local SHACL-Core subset used for focus filtering."""
    if not isinstance(shape, (URIRef, BNode)) or shape in active:
        return False
    active.add(shape)
    try:
        is_property_shape = shapes.value(shape, SH.path) is not None
        for predicate, value in shapes.predicate_objects(shape):
            if predicate in _LOCAL_METADATA_PREDICATES:
                continue
            if predicate in _LOCAL_TARGET_DECLARATIONS:
                continue
            if predicate in _NONLOCAL_TARGET_DECLARATIONS:
                return False
            if predicate == SH.path:
                # An IRI path reads direct outgoing triples from the focus
                # node.  Blank-node paths include inverse, sequence,
                # alternative, or repeated traversal and require full audit.
                if not isinstance(value, URIRef):
                    return False
                continue
            if predicate == SH.property:
                if not _shape_is_local(shapes, value, active):
                    return False
                continue
            if predicate in _LOGICAL_SHAPE_PREDICATES:
                # On a node shape logical members still evaluate the same
                # focus node.  On a property shape they evaluate values, whose
                # updates may originate at another assertion focus.
                if is_property_shape:
                    return False
                try:
                    members = _rdf_list(shapes, value, 32)
                except (ValueError, _ValidationIncomplete):
                    return False
                if not all(_shape_is_local(shapes, member, active) for member in members):
                    return False
                continue
            if predicate in _LOCAL_VALUE_CONSTRAINT_PREDICATES:
                continue
            if predicate in _NONLOCAL_VALUE_CONSTRAINT_PREDICATES:
                # ``sh:class`` is local only when it inspects a node shape's
                # focus itself; property values can be changed elsewhere.
                if predicate == SH["class"] and not is_property_shape:
                    continue
                return False
            # Unknown semantics must not inherit a correctness claim from the
            # current evaluator.  A later validator extension can therefore
            # not silently make existing focused revalidation unsound.
            return False
        return True
    finally:
        active.remove(shape)


def _path_values(
    graph: Graph,
    node: object,
    path: object,
    shapes: Graph,
    depth: int,
    max_path_nodes: int,
) -> tuple[object, ...]:
    if depth < 1:
        raise _ValidationIncomplete("shape_recursion_limit_exhausted")
    if isinstance(path, URIRef):
        return tuple(graph.objects(node, path))
    if not isinstance(path, BNode):
        raise ValueError("invalid SHACL path")
    inverse = shapes.value(path, SH.inversePath)
    if inverse is not None:
        return _inverse_path_values(graph, node, inverse, shapes, depth - 1, max_path_nodes)
    alternatives = shapes.value(path, SH.alternativePath)
    if alternatives is not None:
        values: set[object] = set()
        for candidate in _rdf_list(shapes, alternatives, depth):
            values.update(_path_values(graph, node, candidate, shapes, depth - 1, max_path_nodes))
        return tuple(values)
    for predicate, minimum, maximum in (
        (SH.zeroOrMorePath, 0, None),
        (SH.oneOrMorePath, 1, None),
        (SH.zeroOrOnePath, 0, 1),
    ):
        nested = shapes.value(path, predicate)
        if nested is None:
            continue
        return _repeated_path_values(
            graph,
            node,
            nested,
            shapes,
            depth - 1,
            max_path_nodes,
            minimum=minimum,
            maximum=maximum,
        )
    if shapes.value(path, _RDF_FIRST) is not None:
        values: set[object] = {node}
        for component in _rdf_list(shapes, path, depth):
            following: set[object] = set()
            for current in values:
                following.update(_path_values(graph, current, component, shapes, depth - 1, max_path_nodes))
            if len(following) > max_path_nodes:
                raise _ValidationIncomplete("path_expansion_limit_exhausted")
            values = following
        return tuple(values)
    raise ValueError("unsupported SHACL path")


def _inverse_path_values(
    graph: Graph,
    node: object,
    path: object,
    shapes: Graph,
    depth: int,
    max_path_nodes: int,
) -> tuple[object, ...]:
    """Follow the inverse of a SHACL property path from one focus node."""
    if depth < 1:
        raise _ValidationIncomplete("shape_recursion_limit_exhausted")
    if isinstance(path, URIRef):
        return tuple(graph.subjects(path, node))
    if not isinstance(path, BNode):
        raise ValueError("invalid inverse SHACL path")
    nested_inverse = shapes.value(path, SH.inversePath)
    if nested_inverse is not None:
        return _path_values(graph, node, nested_inverse, shapes, depth - 1, max_path_nodes)
    alternatives = shapes.value(path, SH.alternativePath)
    if alternatives is not None:
        values: set[object] = set()
        for candidate in _rdf_list(shapes, alternatives, depth):
            values.update(_inverse_path_values(graph, node, candidate, shapes, depth - 1, max_path_nodes))
        return tuple(values)
    for predicate, minimum, maximum in (
        (SH.zeroOrMorePath, 0, None),
        (SH.oneOrMorePath, 1, None),
        (SH.zeroOrOnePath, 0, 1),
    ):
        nested = shapes.value(path, predicate)
        if nested is not None:
            return _repeated_path_values(
                graph,
                node,
                nested,
                shapes,
                depth - 1,
                max_path_nodes,
                minimum=minimum,
                maximum=maximum,
                inverse=True,
            )
    if shapes.value(path, _RDF_FIRST) is not None:
        values: set[object] = {node}
        for component in reversed(_rdf_list(shapes, path, depth)):
            following: set[object] = set()
            for current in values:
                following.update(
                    _inverse_path_values(graph, current, component, shapes, depth - 1, max_path_nodes)
                )
            if len(following) > max_path_nodes:
                raise _ValidationIncomplete("path_expansion_limit_exhausted")
            values = following
        return tuple(values)
    raise ValueError("unsupported inverse SHACL path")


def _repeated_path_values(
    graph: Graph,
    node: object,
    nested: object,
    shapes: Graph,
    depth: int,
    max_path_nodes: int,
    *,
    minimum: int,
    maximum: int | None,
    inverse: bool = False,
) -> tuple[object, ...]:
    visited = {node}
    frontier = {node}
    results: set[object] = {node} if minimum == 0 else set()
    steps = 0
    follow = _inverse_path_values if inverse else _path_values
    while frontier:
        if maximum is not None and steps >= maximum:
            break
        steps += 1
        following: set[object] = set()
        for current in frontier:
            following.update(follow(graph, current, nested, shapes, depth - 1, max_path_nodes))
        results.update(following)
        frontier = following - visited
        visited.update(following)
        if len(visited) > max_path_nodes:
            raise _ValidationIncomplete("path_expansion_limit_exhausted")
    if minimum == 1:
        results.discard(node)
    return tuple(results)


def _rdf_list(graph: Graph, head: object, max_depth: int) -> tuple[object, ...]:
    values: list[object] = []
    current = head
    seen: set[object] = set()
    while current != _RDF_NIL:
        if current in seen or len(seen) >= max_depth:
            raise _ValidationIncomplete("cyclic_shape_list")
        seen.add(current)
        first = graph.value(current, _RDF_FIRST)
        rest = graph.value(current, _RDF_REST)
        if first is None or rest is None:
            raise ValueError("malformed SHACL list")
        values.append(first)
        current = rest
    return tuple(values)


def _unsupported_shape_construct(shapes: Graph, profile: SemanticResource, allow_experimental: bool) -> str | None:
    terms = {predicate for _, predicate, _ in shapes}
    terms.update(obj for _, _, obj in shapes if isinstance(obj, URIRef))
    if terms.intersection(_EXECUTABLE_TERMS):
        return "executable_shape_construct_unavailable"
    if terms.intersection(_EXPERIMENTAL_TERMS):
        if profile.maturity.value != "experimental" or not allow_experimental:
            return "experimental_shape_construct_unavailable"
        # These terms are deliberately inert metadata even in an explicitly
        # selected experiment.  There is no evaluator or authority path here.
    if profile.identifier == "shacl12-sparql-20260130-experimental":
        return "shacl_sparql_capability_unavailable"
    return None


def _severity(value: object | None, default: ValidationSeverity = ValidationSeverity.VIOLATION) -> ValidationSeverity:
    if value is None:
        return default
    normalized = str(value)
    if normalized == str(SH.Info):
        return ValidationSeverity.INFO
    if normalized == str(SH.Warning):
        return ValidationSeverity.WARNING
    return ValidationSeverity.VIOLATION


def _integer(value: object | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Literal):
        raise ValueError("SHACL count is not a literal")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("SHACL count is invalid") from error
    if integer < 0:
        raise ValueError("SHACL count is negative")
    return integer


def _boolean(value: object | None) -> bool:
    return value == _TRUE or (isinstance(value, Literal) and str(value).lower() == "true")


def _matches_node_kind(value: object, expected: object) -> bool:
    if expected == _KIND_IRI:
        return isinstance(value, URIRef)
    if expected == _KIND_BLANK:
        return isinstance(value, BNode)
    if expected == _KIND_LITERAL:
        return isinstance(value, Literal)
    if expected == _KIND_BLANK_OR_IRI:
        return isinstance(value, (BNode, URIRef))
    if expected == _KIND_BLANK_OR_LITERAL:
        return isinstance(value, (BNode, Literal))
    if expected == _KIND_IRI_OR_LITERAL:
        return isinstance(value, (URIRef, Literal))
    raise ValueError("unknown SHACL node kind")


def _term_less(left: object, right: object) -> bool:
    if not isinstance(left, Literal) or not isinstance(right, Literal):
        return False
    try:
        return left.toPython() < right.toPython()
    except TypeError:
        return False


def _term_less_or_equal(left: object, right: object) -> bool:
    return left == right or _term_less(left, right)


def _safe_shape_id(value: object) -> str | None:
    # Shapes originate from a verified package resource, but a digest is still
    # preferable to serialising a blank-node label as a persistent API value.
    if isinstance(value, URIRef):
        return str(value)
    if isinstance(value, BNode):
        return "shape:sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return None


def _term_key(value: object) -> str:
    return str(value)


def _safe_assertion_ids(values: Iterable[str]) -> tuple[str, ...]:
    items = tuple(values)
    if len(set(items)) != len(items) or any(not isinstance(item, str) or not item for item in items):
        raise ShaclValidationError("assertion_ids must be unique non-empty strings")
    return items


def _normalized_focus_assertion_ids(
    values: Mapping[URIRef | BNode, str | Sequence[str]],
) -> dict[URIRef | BNode, tuple[str, ...]]:
    """Normalize compatibility singleton mappings to complete focus ownership.

    Early callers passed ``node -> assertion_id``.  Keeping that input form
    avoids a source-compatible break while making the validation core retain
    every assertion sharing a focus node.
    """
    normalized: dict[URIRef | BNode, tuple[str, ...]] = {}
    for focus, assertion_ids in values.items():
        if not isinstance(focus, (URIRef, BNode)):
            raise ShaclValidationError("focus assertion mappings require RDF URI or blank focus nodes")
        items = (assertion_ids,) if isinstance(assertion_ids, str) else tuple(assertion_ids)
        normalized[focus] = _safe_assertion_ids(items)
    return normalized


def _system_finding(code: str) -> ValidationFinding:
    return ValidationFinding(ValidationSeverity.VIOLATION, code)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _pin_mapping(pin: ArtifactPin) -> dict[str, str]:
    return {
        "identifier": pin.identifier,
        "version": str(pin.version),
        "uri": pin.uri,
        "published_date": pin.published_date,
        "sha256": pin.sha256,
        "package_resource": pin.package_resource,
        "maturity": pin.maturity.value,
    }


def _pin_from_mapping(value: Mapping[str, object]) -> ArtifactPin:
    from .registry import StandardsMaturity

    return ArtifactPin(
        identifier=str(value["identifier"]),
        version=SemanticVersion.parse(str(value["version"])),
        uri=str(value["uri"]),
        published_date=str(value["published_date"]),
        sha256=str(value["sha256"]),
        package_resource=str(value["package_resource"]),
        maturity=StandardsMaturity(str(value["maturity"])),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_SHACL_WRITE_POLICY",
    "GovernedShaclValidationService",
    "ShaclCapabilityUnavailable",
    "ShaclSnapshotMismatch",
    "ShaclValidationError",
    "ShaclValidationLimits",
    "ShaclValidationReport",
    "ShaclWritePolicy",
    "ShapeSetReference",
    "ValidationFinding",
    "ValidationSeverity",
    "ValidationSource",
    "ValidationState",
    "ValidationWriteAction",
]
