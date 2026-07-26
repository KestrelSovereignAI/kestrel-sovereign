"""Bounded RDF projections for the canonical assertion contract.

The canonical :mod:`kestrel_sovereign.knowledge.assertion` values remain the
identity and persistence boundary.  This module only translates those values
to and from a small, typed RDF data model.  It deliberately does not parse
RDF syntax, dereference a URI, execute SPARQL, or write to a store.

``rdf11-reification`` is the production projection.  It uses a statement
resource with ``rdf:subject``, ``rdf:predicate`` and ``rdf:object`` and is
serializable as deterministic N-Triples 1.1.  ``rdf12-triple-term`` is an
optional projection: it is selected only after the installed semantic registry
confirms the exact experimental capability and version advertised by the
implementation.  The experimental code is contained in this module; removing
it does not alter assertion values or storage.

The stable import edge accepts only raw RDF 1.1 N-Triples bytes.  Its maintained
parser adapter is deliberately fixed to an offline, bytes-only configuration
before it constructs the internal ``RdfImportDocument`` consumed here.  This
codec then applies structural and canonical-assertion checks.  No application
caller supplies a parser report or chooses a parser configuration.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
from itertools import islice
import math
import time
from typing import Callable, Iterable, Mapping, Protocol, TypeAlias
from urllib.parse import urlsplit

from .assertion import (
    XSD_DATETIME_STAMP,
    XSD_DECIMAL,
    XSD_INTEGER,
    XSD_STRING,
    Assertion,
    AssertionQuery,
    AssertionResult,
    DerivedLineage,
    DirectLineage,
    IRI,
    Lineage,
    Literal,
    OntologyRef,
    TemporalInterval,
    normalize_iri,
)
from .registry import (
    ArtifactPin,
    ExperimentalCapabilityError,
    ResourceNotFoundError,
    SemanticKnowledgeRegistry,
    get_knowledge_registry,
)


RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDF_TYPE = RDF_NAMESPACE + "type"
RDF_STATEMENT = RDF_NAMESPACE + "Statement"
RDF_SUBJECT = RDF_NAMESPACE + "subject"
RDF_PREDICATE = RDF_NAMESPACE + "predicate"
RDF_OBJECT = RDF_NAMESPACE + "object"
# These names are deliberately confined to this module's experimental branch.
RDF_REIFIER = RDF_NAMESPACE + "Reifier"
RDF_REIFIES = RDF_NAMESPACE + "reifies"


class RdfCodecError(ValueError):
    """Base error for a rejected RDF projection or import."""


class UnsupportedRdfCapabilityError(RdfCodecError):
    """A requested RDF draft feature was not explicitly capability-selected."""


class RdfImportBudgetError(RdfCodecError):
    """An import exceeded a declared resource, time, or nesting budget."""


class RdfImportSecurityError(RdfCodecError):
    """An adapter reported a prohibited parser side effect or unsafe URI."""


class RdfOwnershipError(RdfCodecError):
    """RDF attempted to contradict governed import ownership."""


class RdfProjectionKind(str, Enum):
    """The only projection shapes accepted by this codec."""

    RDF11_REIFICATION = "rdf11-reification"
    RDF12_TRIPLE_TERM = "rdf12-triple-term"


@dataclass(frozen=True, slots=True)
class RdfIri:
    """A normalized absolute RDF IRI."""

    value: str

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "value", normalize_iri(self.value))
        except ValueError as error:
            raise RdfCodecError(f"invalid RDF IRI: {self.value!r}") from error


@dataclass(frozen=True, slots=True)
class RdfBlankNode:
    """An RDF blank node accepted only for parser-adapter diagnostics.

    Canonical assertions cannot use blank or source-local IDs as subject,
    predicate, or object.  The importer rejects them rather than inventing an
    IRI and thereby changing identity.
    """

    identifier: str

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier or any(
            character.isspace() or ord(character) < 0x21 for character in self.identifier
        ):
            raise RdfCodecError("RDF blank-node identifier must be a non-empty token")


@dataclass(frozen=True, slots=True)
class RdfLiteral:
    """A typed RDF literal without backend-library objects.

    ``direction`` is retained only so an adapter cannot erase an RDF 1.2
    directional-language distinction.  The canonical assertion contract does
    not admit such a literal, so import fails explicitly instead of degrading
    it to ``rdf:langString``.
    """

    lexical_form: str
    datatype_iri: str = XSD_STRING
    language: str | None = None
    direction: str | None = None

    def __post_init__(self) -> None:
        try:
            literal = Literal(
                self.lexical_form,
                self.datatype_iri,
                language=self.language,
                direction=self.direction,
            )
        except ValueError as error:
            raise RdfCodecError(f"invalid RDF literal: {error}") from error
        object.__setattr__(self, "lexical_form", literal.lexical_form)
        object.__setattr__(self, "datatype_iri", literal.datatype_iri)
        object.__setattr__(self, "language", literal.language)
        object.__setattr__(self, "direction", literal.direction)


RdfAtomicTerm: TypeAlias = RdfIri | RdfBlankNode | RdfLiteral


@dataclass(frozen=True, slots=True)
class RdfTripleTerm:
    """An RDF 1.2 triple term, isolated from the stable projection."""

    subject: RdfIri | RdfBlankNode | "RdfTripleTerm"
    predicate: RdfIri
    object: RdfAtomicTerm | "RdfTripleTerm"

    def __post_init__(self) -> None:
        if not isinstance(self.subject, (RdfIri, RdfBlankNode, RdfTripleTerm)):
            raise RdfCodecError("RDF triple-term subject must be a resource or triple term")
        if not isinstance(self.predicate, RdfIri):
            raise RdfCodecError("RDF triple-term predicate must be an IRI")
        if not isinstance(self.object, (RdfIri, RdfBlankNode, RdfLiteral, RdfTripleTerm)):
            raise RdfCodecError("RDF triple-term object must be an RDF term")


RdfTerm: TypeAlias = RdfAtomicTerm | RdfTripleTerm


@dataclass(frozen=True, slots=True)
class RdfTriple:
    """One RDF triple in the typed codec data model."""

    subject: RdfIri | RdfBlankNode | RdfTripleTerm
    predicate: RdfIri
    object: RdfTerm

    def __post_init__(self) -> None:
        if not isinstance(self.subject, (RdfIri, RdfBlankNode, RdfTripleTerm)):
            raise RdfCodecError("RDF triple subject must be a resource or triple term")
        if not isinstance(self.predicate, RdfIri):
            raise RdfCodecError("RDF triple predicate must be an IRI")
        if not isinstance(self.object, (RdfIri, RdfBlankNode, RdfLiteral, RdfTripleTerm)):
            raise RdfCodecError("RDF triple object must be an RDF term")


@dataclass(frozen=True, slots=True)
class RdfDataset:
    """An immutable RDF graph used at the codec boundary.

    The graph is a tuple rather than a backend graph object.  It keeps parser
    behaviour, blank-node scopes, and query execution out of the application
    layer while still representing standard RDF triples precisely.
    """

    triples: tuple[RdfTriple, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.triples, tuple) or not all(
            isinstance(triple, RdfTriple) for triple in self.triples
        ):
            raise RdfCodecError("RDF dataset triples must be a tuple of RdfTriple values")

    def serialize_ntriples(self) -> bytes:
        """Serialize the RDF 1.1 subset deterministically as N-Triples 1.1.

        This is a serializer, not a parser.  Triple terms and directional
        literals have no RDF 1.1 N-Triples representation and fail clearly.
        """
        rows: list[str] = []
        for triple in self.triples:
            if isinstance(triple.subject, RdfTripleTerm) or isinstance(triple.object, RdfTripleTerm):
                raise UnsupportedRdfCapabilityError(
                    "RDF 1.2 triple terms cannot be serialized as RDF 1.1 N-Triples"
                )
            rows.append(
                f"{_ntriples_term(triple.subject)} {_ntriples_term(triple.predicate)} "
                f"{_ntriples_term(triple.object)} .\n"
            )
        return "".join(sorted(rows)).encode("utf-8")


def _ntriples_term(term: RdfAtomicTerm) -> str:
    if isinstance(term, RdfIri):
        return f"<{term.value}>"
    if isinstance(term, RdfBlankNode):
        return f"_:{term.identifier}"
    if term.direction is not None:
        raise UnsupportedRdfCapabilityError(
            "RDF 1.2 directional literals cannot be serialized as RDF 1.1 N-Triples"
        )
    escaped = _escape_ntriples_string(term.lexical_form)
    if term.language is not None:
        return f'"{escaped}"@{term.language}'
    return f'"{escaped}"^^<{term.datatype_iri}>'


def _escape_ntriples_string(value: str) -> str:
    """Escape every control character N-Triples does not admit literally."""
    escaped: list[str] = []
    simple = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    for character in value:
        if character in simple:
            escaped.append(simple[character])
        elif ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F:
            escaped.append(f"\\u{ord(character):04X}")
        else:
            escaped.append(character)
    return "".join(escaped)


@dataclass(frozen=True, slots=True)
class RdfImportLimits:
    """Fail-closed import budgets applied after an adapter parses a document."""

    max_bytes: int = 1_048_576
    max_statements: int = 10_000
    max_nesting: int = 8
    max_term_bytes: int = 8_192
    max_parse_seconds: float = 5.0
    # This applies only to document URIs a parser might resolve.  RDF term
    # IRIs are inert data at this boundary and must accept the canonical IRI
    # profile so a projection can round-trip without any network operation.
    allowed_uri_schemes: frozenset[str] = frozenset({"did", "http", "https", "urn"})

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_statements", "max_nesting", "max_term_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise RdfCodecError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_parse_seconds, bool)
            or not isinstance(self.max_parse_seconds, (int, float))
            or not math.isfinite(self.max_parse_seconds)
            or self.max_parse_seconds <= 0
        ):
            raise RdfCodecError("max_parse_seconds must be a positive finite number")
        if not self.allowed_uri_schemes or any(
            not isinstance(scheme, str) or not scheme.islower() for scheme in self.allowed_uri_schemes
        ):
            raise RdfCodecError("allowed_uri_schemes must contain lowercase schemes")


@dataclass(frozen=True, slots=True)
class RdfImportDocument:
    """A parser-adapter result accepted by :meth:`RdfAssertionCodec.import_assertion`.

    No raw RDF bytes are accepted here on purpose.  The adapter selects a
    maintained standards parser and must disable network dereferencing,
    ``owl:imports``, JSON-LD remote contexts, file loaders, and parser hooks.
    The codec validates the report rather than trusting a parser configuration
    hidden in a backend library.
    """

    dataset: RdfDataset
    received_bytes: int
    parse_seconds: float = 0.0
    remote_contexts: tuple[str, ...] = ()
    followed_imports: tuple[str, ...] = ()
    parser_side_effects: tuple[str, ...] = ()
    document_uri: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, RdfDataset):
            raise RdfCodecError("RDF import document requires an RdfDataset")
        if type(self.received_bytes) is not int or self.received_bytes < 0:
            raise RdfCodecError("received_bytes must be a non-negative integer")
        if (
            isinstance(self.parse_seconds, bool)
            or not isinstance(self.parse_seconds, (int, float))
            or not math.isfinite(self.parse_seconds)
            or self.parse_seconds < 0
        ):
            raise RdfCodecError("parse_seconds must be a non-negative finite number")
        for name in ("remote_contexts", "followed_imports", "parser_side_effects"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
                raise RdfCodecError(f"{name} must be a tuple of strings")
        if self.document_uri is not None and not isinstance(self.document_uri, str):
            raise RdfCodecError("document_uri must be a string or null")


@dataclass(frozen=True, slots=True)
class RdfImportOwnership:
    """Authority supplied by the governed import boundary, never RDF content."""

    tenant_id: str
    owning_agent_id: str

    def __post_init__(self) -> None:
        # Assertion supplies the definitive validation while this constructor
        # rejects accidental empty authority before any RDF processing begins.
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise RdfOwnershipError("governed import tenant_id is required")
        if not isinstance(self.owning_agent_id, str) or not self.owning_agent_id:
            raise RdfOwnershipError("governed import owning_agent_id is required")


@dataclass(frozen=True, slots=True)
class RdfCodecConfiguration:
    """Explicit opt-in selections advertised by the concrete RDF implementation.

    A selected capability and its exact version are both mandatory.  Supplying
    an unknown version is an error; it is never treated as the nearest draft.
    """

    rdf12_capability: str | None = None
    rdf12_version: str | None = None
    sparql12_capability: str | None = None
    sparql12_version: str | None = None

    def __post_init__(self) -> None:
        for capability, version, label in (
            (self.rdf12_capability, self.rdf12_version, "rdf12"),
            (self.sparql12_capability, self.sparql12_version, "sparql12"),
        ):
            if (capability is None) != (version is None):
                raise UnsupportedRdfCapabilityError(
                    f"{label} capability and exact version must be selected together"
                )
            if capability is not None and (not capability or not version):
                raise UnsupportedRdfCapabilityError(
                    f"{label} capability and exact version must be non-empty"
                )


@dataclass(frozen=True, slots=True)
class RdfCapabilityReport:
    """Registry-pinned capabilities actually active in one codec instance."""

    rdf11: ArtifactPin
    ntriples11: ArtifactPin
    assertion_ontology: ArtifactPin
    prov_o: ArtifactPin
    owl_time: ArtifactPin
    sparql11: ArtifactPin
    rdf12: ArtifactPin | None
    sparql12: ArtifactPin | None

    @property
    def experimental_enabled(self) -> bool:
        return self.rdf12 is not None or self.sparql12 is not None


@dataclass(frozen=True, slots=True)
class RdfTypedQuery:
    """Application-facing bounded read request for an RDF-backed resolver."""

    query: AssertionQuery
    ownership: RdfImportOwnership

    def __post_init__(self) -> None:
        if not isinstance(self.query, AssertionQuery):
            raise RdfCodecError("RDF typed query requires AssertionQuery")
        if not isinstance(self.ownership, RdfImportOwnership):
            raise RdfCodecError("RDF typed query requires governed ownership")


class RdfAssertionReadAdapter(Protocol):
    """Backend-neutral typed read boundary; raw SPARQL never crosses it."""

    def read_assertions(self, request: RdfTypedQuery) -> tuple[AssertionResult, ...]:
        """Return at most ``request.query.limit`` assertion results."""


class _SparqlBackend(Protocol):
    """Private adapter-edge protocol for a bounded, cancellable query execution.

    Backends must enforce ``timeout_seconds`` while producing the iterator and
    must make ``cancel_readonly`` stop a still-live execution promptly.  The
    profile pin prevents a backend configured for one standards snapshot from
    being silently used for another.
    """

    def execute_readonly(
        self,
        query_text: str,
        *,
        profile: ArtifactPin,
        timeout_seconds: float,
    ) -> Iterable[Mapping[str, object]]:
        """Execute a precompiled, tenant-scoped read-only query within a timeout."""

    def cancel_readonly(self, *, profile: ArtifactPin) -> None:
        """Cancel the active read for ``profile`` after the adapter stops early."""


DEFAULT_SPARQL_EXECUTION_TIMEOUT_SECONDS = 5.0


class _SparqlAssertionReadAdapter:
    """Shared bounded execution for snapshot-specific SPARQL read adapters."""

    def __init__(
        self,
        backend: _SparqlBackend,
        decode_row: Callable[[Mapping[str, object], RdfImportOwnership], AssertionResult],
        *,
        query_profile: ArtifactPin,
        compile_query: Callable[[RdfTypedQuery], str],
        execution_timeout_seconds: float,
    ) -> None:
        if (
            isinstance(execution_timeout_seconds, bool)
            or not isinstance(execution_timeout_seconds, (int, float))
            or not math.isfinite(execution_timeout_seconds)
            or execution_timeout_seconds <= 0
        ):
            raise RdfCodecError("SPARQL execution timeout must be a positive finite number")
        self._backend = backend
        self._decode_row = decode_row
        self._query_profile = query_profile
        self._compile_query = compile_query
        self._execution_timeout_seconds = float(execution_timeout_seconds)

    def read_assertions(self, request: RdfTypedQuery) -> tuple[AssertionResult, ...]:
        query_text = self._compile_query(request)
        rows = tuple(
            islice(
                self._backend.execute_readonly(
                    query_text,
                    profile=self._query_profile,
                    timeout_seconds=self._execution_timeout_seconds,
                ),
                request.query.limit + 1,
            )
        )
        if len(rows) > request.query.limit:
            self._backend.cancel_readonly(profile=self._query_profile)
            raise RdfImportBudgetError(
                "SPARQL backend returned more rows than the typed query limit"
            )
        return tuple(self._decode_row(row, request.ownership) for row in rows)


class Sparql11AssertionReadAdapter(_SparqlAssertionReadAdapter):
    """Optional adapter that compiles :class:`RdfTypedQuery` at the backend edge.

    This narrow class intentionally does not expose a method accepting a query
    string.  Applications use ``read_assertions``; the injected backend is the
    only component that sees the generated SPARQL text.
    """

    def __init__(
        self,
        backend: _SparqlBackend,
        decode_row: Callable[[Mapping[str, object], RdfImportOwnership], AssertionResult],
        *,
        vocabulary_namespace: str,
        time_namespace: str,
        query_profile: ArtifactPin,
        execution_timeout_seconds: float = DEFAULT_SPARQL_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        try:
            normalized_vocabulary_namespace = normalize_iri(vocabulary_namespace)
            normalized_time_namespace = normalize_iri(time_namespace)
        except ValueError as error:
            raise RdfCodecError("SPARQL adapter requires registry vocabulary namespaces") from error
        super().__init__(
            backend,
            decode_row,
            query_profile=query_profile,
            compile_query=lambda request: _compile_sparql11(
                request,
                vocabulary_namespace=normalized_vocabulary_namespace,
                time_namespace=normalized_time_namespace,
            ),
            execution_timeout_seconds=execution_timeout_seconds,
        )


class Sparql12AssertionReadAdapter(_SparqlAssertionReadAdapter):
    """Registry-pinned SPARQL 1.2 adapter for the typed assertion-read subset.

    The typed subset deliberately has no raw triple-term or reifier syntax:
    canonical assertions remain RDF 1.1-compatible.  The distinct adapter and
    exact profile pin ensure a selected SPARQL 1.2 backend is used only through
    its negotiated draft snapshot rather than silently taking the 1.1 path.
    """

    def __init__(
        self,
        backend: _SparqlBackend,
        decode_row: Callable[[Mapping[str, object], RdfImportOwnership], AssertionResult],
        *,
        vocabulary_namespace: str,
        time_namespace: str,
        query_profile: ArtifactPin,
        execution_timeout_seconds: float = DEFAULT_SPARQL_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        try:
            normalized_vocabulary_namespace = normalize_iri(vocabulary_namespace)
            normalized_time_namespace = normalize_iri(time_namespace)
        except ValueError as error:
            raise RdfCodecError("SPARQL adapter requires registry vocabulary namespaces") from error
        super().__init__(
            backend,
            decode_row,
            query_profile=query_profile,
            compile_query=lambda request: _compile_sparql12(
                request,
                vocabulary_namespace=normalized_vocabulary_namespace,
                time_namespace=normalized_time_namespace,
            ),
            execution_timeout_seconds=execution_timeout_seconds,
        )


def _compile_sparql11(
    request: RdfTypedQuery,
    *,
    vocabulary_namespace: str,
    time_namespace: str,
) -> str:
    """Compile the stable SPARQL 1.1 typed-read subset at the adapter edge."""
    return _compile_typed_assertion_read(
        request,
        vocabulary_namespace=vocabulary_namespace,
        time_namespace=time_namespace,
        profile_label="SPARQL 1.1",
        statement_filters=_sparql11_statement_filters,
    )


def _compile_sparql12(
    request: RdfTypedQuery,
    *,
    vocabulary_namespace: str,
    time_namespace: str,
) -> str:
    """Compile the selected SPARQL 1.2 typed-read subset at the adapter edge."""
    return _compile_typed_assertion_read(
        request,
        vocabulary_namespace=vocabulary_namespace,
        time_namespace=time_namespace,
        profile_label="SPARQL 1.2",
        statement_filters=_sparql12_statement_filters,
    )


def _compile_typed_assertion_read(
    request: RdfTypedQuery,
    *,
    vocabulary_namespace: str,
    time_namespace: str,
    profile_label: str,
    statement_filters: Callable[[AssertionQuery], tuple[str, ...]],
) -> str:
    """Compile every SPARQL-supported :class:`AssertionQuery` narrowing field."""
    # Values are emitted only from closed assertion values, then N-Triples
    # escaped.  This is not a public query language or general SPARQL parser.
    query = request.query
    if query.cursor is not None:
        raise RdfCodecError(
            "SPARQL assertion reads do not support AssertionQuery.cursor; "
            "the canonical cursor encoding has not been negotiated"
        )
    filters = [
        f'?revision <{vocabulary_namespace}projectedTenantId> '
        f'{_ntriples_term(RdfLiteral(request.ownership.tenant_id))} .',
        f"?revision <{vocabulary_namespace}assertionId> ?assertionId .",
        f"?revision <{vocabulary_namespace}revisionId> ?revisionId .",
    ]
    filters.extend(statement_filters(query))
    if query.assertion_ids:
        values = " ".join(_ntriples_term(RdfLiteral(value)) for value in query.assertion_ids)
        filters.append(f"VALUES ?assertionId {{ {values} }}")
    if query.statuses:
        values = " ".join(_ntriples_term(RdfLiteral(status.value)) for status in query.statuses)
        filters.append(
            f"?revision <{vocabulary_namespace}status> ?status . "
            f"VALUES ?status {{ {values} }}"
        )
    if query.epistemic_states:
        values = " ".join(
            _ntriples_term(RdfLiteral(state.value)) for state in query.epistemic_states
        )
        filters.append(
            f"?revision <{vocabulary_namespace}epistemicState> ?epistemicState . "
            f"VALUES ?epistemicState {{ {values} }}"
        )
    if query.valid_at is not None:
        filters.extend(
            _at_time_filters(
                revision="?revision",
                relation=f"{vocabulary_namespace}validTime",
                interval_variable="?validInterval",
                start_variable="?validStart",
                end_variable="?validEnd",
                instant=query.valid_at.value,
                time_namespace=time_namespace,
            )
        )
    if query.observed_at is not None:
        filters.extend(
            _at_time_filters(
                revision="?revision",
                relation=f"{vocabulary_namespace}observedTime",
                interval_variable="?observedInterval",
                start_variable="?observedStart",
                end_variable="?observedEnd",
                instant=query.observed_at.value,
                time_namespace=time_namespace,
            )
        )
    return (
        f"# Kestrel {profile_label} typed assertion read\n"
        "SELECT ?revision WHERE { "
        + " ".join(filters)
        + f" }} ORDER BY ?assertionId ?revisionId LIMIT {query.limit}"
    )


def _sparql11_statement_filters(query: AssertionQuery) -> tuple[str, ...]:
    """Match the stable RDF 1.1 statement-resource projection."""
    filters: list[str] = []
    if query.subject is not None:
        filters.append(f"?revision <{RDF_SUBJECT}> <{query.subject.value}> .")
    if query.predicate is not None:
        filters.append(f"?revision <{RDF_PREDICATE}> <{query.predicate.value}> .")
    if query.object is not None:
        filters.append(f"?revision <{RDF_OBJECT}> {_assertion_term_to_ntriples(query.object)} .")
    return tuple(filters)


def _sparql12_statement_filters(query: AssertionQuery) -> tuple[str, ...]:
    """Match the negotiated RDF 1.2 reifier/triple-term projection.

    The RDF 1.2 projection has no ``rdf:subject``, ``rdf:predicate``, or
    ``rdf:object`` triples.  Binding directly through ``rdf:reifies`` keeps
    the typed read aligned with the selected draft representation rather than
    silently querying an RDF 1.1 shape.
    """
    subject = (
        _ntriples_term(RdfIri(query.subject.value))
        if query.subject is not None
        else "?statementSubject"
    )
    predicate = (
        _ntriples_term(RdfIri(query.predicate.value))
        if query.predicate is not None
        else "?statementPredicate"
    )
    object_ = (
        _assertion_term_to_ntriples(query.object)
        if query.object is not None
        else "?statementObject"
    )
    return (
        f"?revision <{RDF_REIFIES}> <<( {subject} {predicate} {object_} )>> .",
    )


def _at_time_filters(
    *,
    revision: str,
    relation: str,
    interval_variable: str,
    start_variable: str,
    end_variable: str,
    instant: str,
    time_namespace: str,
) -> tuple[str, ...]:
    """Return an inclusive interval containment predicate for one query instant."""
    instant_term = _ntriples_term(RdfLiteral(instant, XSD_DATETIME_STAMP))
    return (
        f"{revision} <{relation}> {interval_variable} .",
        f"OPTIONAL {{ {interval_variable} <{time_namespace}hasBeginning> {start_variable} . }}",
        f"OPTIONAL {{ {interval_variable} <{time_namespace}hasEnd> {end_variable} . }}",
        f"FILTER (!BOUND({start_variable}) || {start_variable} <= {instant_term})",
        f"FILTER (!BOUND({end_variable}) || {end_variable} >= {instant_term})",
    )


def _assertion_term_to_ntriples(term: IRI | Literal) -> str:
    if isinstance(term, IRI):
        return _ntriples_term(RdfIri(term.value))
    return _ntriples_term(
        RdfLiteral(term.lexical_form, term.datatype_iri, term.language, term.direction)
    )


class RdfAssertionCodec:
    """Project canonical assertions through registry-selected RDF profiles."""

    def __init__(
        self,
        *,
        registry: SemanticKnowledgeRegistry | None = None,
        configuration: RdfCodecConfiguration | None = None,
    ) -> None:
        self._registry = registry or get_knowledge_registry()
        self._configuration = configuration or RdfCodecConfiguration()
        self._rdf11 = self._registry.select_capability("rdf-profile:rdf11")
        self._ntriples11 = self._registry.select_capability("serialization:ntriples-20140225")
        self._ontology = self._registry.select_capability("ontology:kestrel-vocab-1.1")
        self._prov = self._registry.select_capability("vocabulary:prov-o")
        self._time = self._registry.select_capability("vocabulary:owl-time")
        self._sparql11 = self._registry.select_capability("query-profile:sparql11-readonly")
        self._rdf12 = self._select_experimental(
            configuration=self._configuration.rdf12_capability,
            version=self._configuration.rdf12_version,
            label="RDF 1.2 triple-term/reifier",
            capability_prefix="rdf-profile:rdf12",
        )
        self._sparql12 = self._select_experimental(
            configuration=self._configuration.sparql12_capability,
            version=self._configuration.sparql12_version,
            label="SPARQL 1.2",
            capability_prefix="query-profile:sparql12",
        )
        self._k = self._ontology.resource.namespace
        self._prov_namespace = self._prov.resource.namespace
        self._time_namespace = self._time.resource.namespace
        self._structural_predicates = frozenset(
            self._term(name)
            for name in (
                "hasRevision",
                "observedTime",
                "validTime",
                "ontologyVersion",
                "sourceMembership",
                "hasSourceOccurrence",
                "derivation",
                "inputMembership",
            )
        )

    def _select_experimental(
        self,
        *,
        configuration: str | None,
        version: str | None,
        label: str,
        capability_prefix: str,
    ):
        if configuration is None:
            return None
        if not configuration.startswith(capability_prefix):
            raise UnsupportedRdfCapabilityError(
                f"{label} requires a registry capability beginning {capability_prefix!r}"
            )
        try:
            selected = self._registry.select_capability(configuration, allow_experimental=True)
        except (ResourceNotFoundError, ExperimentalCapabilityError) as error:
            raise UnsupportedRdfCapabilityError(
                f"{label} capability {configuration!r} is not available from the semantic registry"
            ) from error
        if str(selected.resource.version) != version:
            raise UnsupportedRdfCapabilityError(
                f"{label} requires exact registry version {version!r}; "
                f"capability {configuration!r} pins {selected.resource.version}"
            )
        return selected

    @property
    def capability_report(self) -> RdfCapabilityReport:
        return RdfCapabilityReport(
            rdf11=self._rdf11.resource.pin,
            ntriples11=self._ntriples11.resource.pin,
            assertion_ontology=self._ontology.resource.pin,
            prov_o=self._prov.resource.pin,
            owl_time=self._time.resource.pin,
            sparql11=self._sparql11.resource.pin,
            rdf12=self._rdf12.resource.pin if self._rdf12 else None,
            sparql12=self._sparql12.resource.pin if self._sparql12 else None,
        )

    def typed_query(self, query: AssertionQuery, ownership: RdfImportOwnership) -> RdfTypedQuery:
        """Bind a typed assertion read to this registry-selected vocabulary."""
        return RdfTypedQuery(query=query, ownership=ownership)

    def sparql11_read_adapter(
        self,
        backend: _SparqlBackend,
        decode_row: Callable[[Mapping[str, object], RdfImportOwnership], AssertionResult],
        *,
        execution_timeout_seconds: float = DEFAULT_SPARQL_EXECUTION_TIMEOUT_SECONDS,
    ) -> Sparql11AssertionReadAdapter:
        """Create an optional SPARQL 1.1 adapter bound to this registry vocabulary."""
        return Sparql11AssertionReadAdapter(
            backend,
            decode_row,
            vocabulary_namespace=self._k,
            time_namespace=self._time_namespace,
            query_profile=self._sparql11.resource.pin,
            execution_timeout_seconds=execution_timeout_seconds,
        )

    def sparql12_read_adapter(
        self,
        backend: _SparqlBackend,
        decode_row: Callable[[Mapping[str, object], RdfImportOwnership], AssertionResult],
        *,
        execution_timeout_seconds: float = DEFAULT_SPARQL_EXECUTION_TIMEOUT_SECONDS,
    ) -> Sparql12AssertionReadAdapter:
        """Create a SPARQL 1.2 adapter only for an exact selected draft capability."""
        if self._sparql12 is None:
            raise UnsupportedRdfCapabilityError(
                "SPARQL 1.2 reads require an explicitly selected registry-pinned "
                "SPARQL 1.2 capability; SPARQL 1.1 was not substituted"
            )
        if self._rdf12 is None:
            raise UnsupportedRdfCapabilityError(
                "SPARQL 1.2 typed reads require the matching explicitly selected "
                "RDF 1.2 triple-term/reifier projection capability"
            )
        return Sparql12AssertionReadAdapter(
            backend,
            decode_row,
            vocabulary_namespace=self._k,
            time_namespace=self._time_namespace,
            query_profile=self._sparql12.resource.pin,
            execution_timeout_seconds=execution_timeout_seconds,
        )

    def project(
        self,
        assertion: Assertion,
        *,
        projection: RdfProjectionKind = RdfProjectionKind.RDF11_REIFICATION,
    ) -> RdfDataset:
        """Create a lossless RDF projection without changing canonical identity."""
        if not isinstance(assertion, Assertion):
            raise RdfCodecError("RDF projection requires an Assertion")
        if projection is RdfProjectionKind.RDF12_TRIPLE_TERM and self._rdf12 is None:
            raise UnsupportedRdfCapabilityError(
                "RDF 1.2 triple-term/reifier projection requires an explicitly selected "
                "registry-pinned RDF 1.2 capability; RDF 1.1 reification was not substituted"
            )
        if projection not in (RdfProjectionKind.RDF11_REIFICATION, RdfProjectionKind.RDF12_TRIPLE_TERM):
            raise RdfCodecError(f"unsupported RDF projection {projection!r}")

        assertion_node = RdfIri(assertion.assertion_id)
        revision = self._node(assertion, "revision")
        triples: list[RdfTriple] = []
        add = lambda subject, predicate, object_: triples.append(
            RdfTriple(subject, RdfIri(predicate), object_)
        )

        add(assertion_node, RDF_TYPE, RdfIri(self._term("SemanticAssertion")))
        add(assertion_node, self._term("hasRevision"), revision)
        add(revision, RDF_TYPE, RdfIri(self._term("AssertionRevision")))
        add(revision, self._term("assertionId"), _string(assertion.assertion_id))
        # Projected authority is audit metadata only.  Import obtains authority
        # from RdfImportOwnership and rejects a contradiction.
        add(revision, self._term("projectedTenantId"), _string(assertion.tenant_id))
        add(revision, self._term("projectedOwnerId"), _string(assertion.owning_agent_id))
        add(revision, self._term("revisionId"), _string(assertion.revision_id))

        subject = RdfIri(assertion.subject.value)
        predicate = RdfIri(assertion.predicate.value)
        object_ = _rdf_term(assertion.object)
        if projection is RdfProjectionKind.RDF11_REIFICATION:
            add(revision, RDF_TYPE, RdfIri(RDF_STATEMENT))
            add(revision, RDF_SUBJECT, subject)
            add(revision, RDF_PREDICATE, predicate)
            add(revision, RDF_OBJECT, object_)
        else:
            add(revision, RDF_TYPE, RdfIri(RDF_REIFIER))
            add(revision, RDF_REIFIES, RdfTripleTerm(subject, predicate, object_))
            # The direct triple is a view convenience, while the revision
            # resource is where Kestrel metadata remains attached.
            add(subject, predicate.value, object_)

        self._project_metadata(add, assertion, revision)
        return RdfDataset(tuple(triples))

    def serialize_rdf11_ntriples(self, assertion: Assertion) -> bytes:
        """Return the documented, deterministic RDF 1.1-compatible export."""
        return self.project(assertion).serialize_ntriples()

    def _project_metadata(self, add, assertion: Assertion, revision: RdfIri) -> None:
        add(revision, self._term("identityVersion"), _string(assertion.identity_version))
        add(revision, self._term("confidence"), RdfLiteral(format(assertion.confidence, "f"), XSD_DECIMAL))
        add(revision, self._term("confidenceMethod"), _string(assertion.confidence_method))
        add(revision, self._term("confidenceBasis"), _string(assertion.confidence_basis))
        add(revision, self._term("epistemicState"), _string(assertion.epistemic_state.value))
        add(revision, self._term("assertedAt"), RdfLiteral(assertion.asserted_at.value, XSD_DATETIME_STAMP))
        add(revision, self._term("status"), _string(assertion.status.value))
        if assertion.supersedes_revision_id is not None:
            add(revision, self._term("supersedesRevisionId"), _string(assertion.supersedes_revision_id))
        add(revision, self._term("visibility"), _string(assertion.visibility.value))
        add(revision, self._term("privacyClassification"), _string(assertion.privacy_classification))
        add(revision, self._term("releasePolicyReference"), _string(assertion.release_policy_reference))
        add(revision, self._term("iriProfile"), _string(assertion.iri_profile))
        add(revision, self._term("literalProfile"), _string(assertion.literal_profile))

        ontology = self._node(assertion, "ontology")
        add(revision, self._term("ontologyVersion"), ontology)
        add(ontology, self._term("ontologyNamespace"), _string(assertion.ontology_version.namespace))
        add(ontology, self._term("ontologyVersionLabel"), _string(assertion.ontology_version.version))
        add(ontology, self._term("ontologyContentDigest"), _string(assertion.ontology_version.content_digest))
        add(ontology, self._term("compatibilityProfile"), _string(assertion.ontology_version.compatibility_profile))

        self._project_interval(add, assertion, revision, "observedTime", assertion.observed_time)
        self._project_interval(add, assertion, revision, "validTime", assertion.valid_time)
        if isinstance(assertion.lineage, DirectLineage):
            add(revision, self._term("lineageType"), _string("direct"))
            for position, source_id in enumerate(assertion.lineage.source_occurrence_ids):
                member = self._node(assertion, f"source-member-{position}")
                source = self._node(assertion, f"source-{position}")
                add(revision, self._term("sourceMembership"), member)
                add(member, RDF_TYPE, RdfIri(self._term("LineageMember")))
                add(member, self._term("lineagePosition"), RdfLiteral(str(position), XSD_INTEGER))
                add(member, self._term("hasSourceOccurrence"), source)
                add(source, RDF_TYPE, RdfIri(self._term("SourceOccurrence")))
                add(source, self._term("sourceOccurrenceId"), _string(source_id))
            return

        lineage = assertion.lineage
        add(revision, self._term("lineageType"), _string("derived"))
        derivation = self._node(assertion, "derivation")
        add(revision, self._term("derivation"), derivation)
        add(derivation, RDF_TYPE, RdfIri(self._prov_namespace + "Activity"))
        add(derivation, self._term("ruleId"), _string(lineage.rule_id))
        add(derivation, self._term("engineVersion"), _string(lineage.engine_version))
        add(derivation, self._term("profileVersion"), _string(lineage.profile_version))
        add(derivation, self._term("inputDigest"), _string(lineage.input_digest))
        add(derivation, self._term("runId"), _string(lineage.run_id))
        add(derivation, self._term("generatedAt"), RdfLiteral(lineage.generated_at.value, XSD_DATETIME_STAMP))
        if lineage.derivation_reference is not None:
            add(derivation, self._term("derivationReference"), _string(lineage.derivation_reference))
        for position, revision_id in enumerate(lineage.input_revision_ids):
            member = self._node(assertion, f"input-member-{position}")
            add(derivation, self._term("inputMembership"), member)
            add(member, RDF_TYPE, RdfIri(self._term("LineageMember")))
            add(member, self._term("lineagePosition"), RdfLiteral(str(position), XSD_INTEGER))
            add(member, self._term("inputRevisionId"), _string(revision_id))

    def _project_interval(
        self,
        add,
        assertion: Assertion,
        revision: RdfIri,
        name: str,
        interval: TemporalInterval | None,
    ) -> None:
        if interval is None:
            return
        node = self._node(assertion, name)
        add(revision, self._term(name), node)
        add(node, RDF_TYPE, RdfIri(self._time_namespace + "Interval"))
        if interval.start is not None:
            add(node, self._time_namespace + "hasBeginning", RdfLiteral(interval.start.value, XSD_DATETIME_STAMP))
        if interval.end is not None:
            add(node, self._time_namespace + "hasEnd", RdfLiteral(interval.end.value, XSD_DATETIME_STAMP))

    def import_assertion(
        self,
        payload: bytes,
        *,
        ownership: RdfImportOwnership,
        limits: RdfImportLimits | None = None,
    ) -> Assertion:
        """Import raw RDF 1.1 N-Triples through the fixed offline parser.

        This is the production import boundary.  The parser receives the raw
        bytes and applies byte, statement-count, and elapsed-time limits before
        it creates the internal document decoded below.  It has no configurable
        format, document URI, remote context, import, or loader hook.
        """
        limits = limits or RdfImportLimits()
        if not isinstance(payload, bytes):
            raise RdfCodecError("serialized RDF import requires raw bytes")
        # Import at the production boundary rather than module initialization:
        # this avoids a circular model import and keeps projection-only uses
        # from loading parser implementation code.
        from .rdf_parser import RdfLibNTriplesParser

        document = RdfLibNTriplesParser().parse(payload, limits=limits)
        return self._decode_import_document(document, ownership=ownership, limits=limits)

    def import_projected_dataset(
        self,
        dataset: RdfDataset,
        *,
        ownership: RdfImportOwnership,
        limits: RdfImportLimits | None = None,
    ) -> Assertion:
        """Decode a typed in-process projection without a serialized parser.

        This supports a capability-negotiated RDF 1.2 projection created by
        :meth:`project`.  It is intentionally not a serialized untrusted-data
        import path: external RDF bytes must use :meth:`import_assertion`.
        """
        if not isinstance(dataset, RdfDataset):
            raise RdfCodecError("typed RDF projection import requires an RdfDataset")
        limits = limits or RdfImportLimits()
        document = RdfImportDocument(
            dataset=dataset,
            received_bytes=_dataset_byte_weight(dataset),
        )
        return self._decode_import_document(document, ownership=ownership, limits=limits)

    def _decode_import_document(
        self,
        document: RdfImportDocument,
        *,
        ownership: RdfImportOwnership,
        limits: RdfImportLimits,
    ) -> Assertion:
        """Decode an internal parser result using governed ownership, not RDF claims."""
        started = time.monotonic()
        self._validate_import_document(document, limits)
        by_subject = _by_subject(document.dataset.triples)
        revisions = tuple(
            subject
            for subject, triples in by_subject.items()
            if _has(triples, RDF_TYPE, self._term("AssertionRevision"))
        )
        if len(revisions) != 1:
            raise RdfCodecError("RDF assertion import requires exactly one kestrel:AssertionRevision")
        revision = revisions[0]
        if not isinstance(revision, RdfIri):
            raise RdfCodecError("RDF assertion revision must use an IRI statement resource")
        assertion_id = _one_string(by_subject, revision, self._term("assertionId"))
        assertion_node = RdfIri(assertion_id)
        if not _has(by_subject.get(assertion_node, ()), RDF_TYPE, self._term("SemanticAssertion")) or not _has(
            by_subject.get(assertion_node, ()), self._term("hasRevision"), revision
        ):
            raise RdfCodecError("RDF assertion identity resource must link to its revision")

        projected_tenant = _one_string(by_subject, revision, self._term("projectedTenantId"))
        projected_owner = _one_string(by_subject, revision, self._term("projectedOwnerId"))
        if projected_tenant != ownership.tenant_id or projected_owner != ownership.owning_agent_id:
            raise RdfOwnershipError(
                "RDF projected tenant/owner contradicts the governed import boundary; "
                "RDF content cannot self-assert authoritative ownership"
            )

        subject, predicate, object_ = self._decode_statement(by_subject, revision)
        lineage = self._decode_lineage(by_subject, revision)
        result = Assertion(
            assertion_id=assertion_id,
            tenant_id=ownership.tenant_id,
            owning_agent_id=ownership.owning_agent_id,
            subject=subject,
            predicate=predicate,
            object=object_,
            revision_id=_one_string(by_subject, revision, self._term("revisionId")),
            confidence=_one_literal(by_subject, revision, self._term("confidence"), XSD_DECIMAL),
            confidence_method=_one_string(by_subject, revision, self._term("confidenceMethod")),
            confidence_basis=_one_string(by_subject, revision, self._term("confidenceBasis")),
            epistemic_state=_one_string(by_subject, revision, self._term("epistemicState")),
            asserted_at=_one_literal(by_subject, revision, self._term("assertedAt"), XSD_DATETIME_STAMP),
            observed_time=self._decode_interval(by_subject, revision, "observedTime"),
            valid_time=self._decode_interval(by_subject, revision, "validTime"),
            status=_one_string(by_subject, revision, self._term("status")),
            supersedes_revision_id=_optional_string(by_subject, revision, self._term("supersedesRevisionId")),
            visibility=_one_string(by_subject, revision, self._term("visibility")),
            privacy_classification=_one_string(by_subject, revision, self._term("privacyClassification")),
            release_policy_reference=_one_string(by_subject, revision, self._term("releasePolicyReference")),
            ontology_version=self._decode_ontology(by_subject, revision),
            identity_version=_one_string(by_subject, revision, self._term("identityVersion")),
            iri_profile=_one_string(by_subject, revision, self._term("iriProfile")),
            literal_profile=_one_string(by_subject, revision, self._term("literalProfile")),
            lineage=lineage,
        )
        if time.monotonic() - started > limits.max_parse_seconds:
            raise RdfImportBudgetError("RDF codec processing exceeded the import time budget")
        return result

    def _decode_statement(self, by_subject, revision: RdfIri) -> tuple[IRI, IRI, IRI | Literal]:
        triples = by_subject[revision]
        reifies = _values(triples, RDF_REIFIES)
        if reifies:
            if self._rdf12 is None:
                raise UnsupportedRdfCapabilityError(
                    "RDF 1.2 triple-term/reifier input requires an explicitly selected "
                    "registry-pinned RDF 1.2 capability"
                )
            if (
                not _has(triples, RDF_TYPE, RDF_REIFIER)
                or len(reifies) != 1
                or not isinstance(reifies[0], RdfTripleTerm)
            ):
                raise RdfCodecError("RDF 1.2 reifier projection must contain one rdf:reifies triple term")
            triple_term = reifies[0]
            return _canonical_statement_terms(triple_term.subject, triple_term.predicate, triple_term.object)
        if self._contains_triple_term(by_subject.values()):
            raise UnsupportedRdfCapabilityError(
                "RDF 1.2 triple-term input is not enabled; RDF 1.1 projection was not inferred"
            )
        if not _has(triples, RDF_TYPE, RDF_STATEMENT):
            raise RdfCodecError("RDF 1.1 statement-resource projection requires rdf:Statement")
        subject = _one_value(triples, RDF_SUBJECT)
        predicate = _one_value(triples, RDF_PREDICATE)
        object_ = _one_value(triples, RDF_OBJECT)
        return _canonical_statement_terms(subject, predicate, object_)

    def _decode_interval(self, by_subject, revision: RdfIri, name: str) -> TemporalInterval | None:
        values = _values(by_subject[revision], self._term(name))
        if not values:
            return None
        if len(values) != 1 or not isinstance(values[0], RdfIri):
            raise RdfCodecError(f"{name} must point to one interval resource")
        node = values[0]
        return TemporalInterval(
            start=_optional_literal(by_subject, node, self._time_namespace + "hasBeginning", XSD_DATETIME_STAMP),
            end=_optional_literal(by_subject, node, self._time_namespace + "hasEnd", XSD_DATETIME_STAMP),
        )

    def _decode_ontology(self, by_subject, revision: RdfIri) -> OntologyRef:
        node = _one_value(by_subject[revision], self._term("ontologyVersion"))
        if not isinstance(node, RdfIri):
            raise RdfCodecError("ontologyVersion must point to an IRI resource")
        return OntologyRef(
            namespace=_one_string(by_subject, node, self._term("ontologyNamespace")),
            version=_one_string(by_subject, node, self._term("ontologyVersionLabel")),
            content_digest=_one_string(by_subject, node, self._term("ontologyContentDigest")),
            compatibility_profile=_one_string(by_subject, node, self._term("compatibilityProfile")),
        )

    def _decode_lineage(self, by_subject, revision: RdfIri) -> Lineage:
        kind = _one_string(by_subject, revision, self._term("lineageType"))
        if kind == "direct":
            members = _values(by_subject[revision], self._term("sourceMembership"))
            source_ids: list[str] = []
            for member in _ordered_members(
                by_subject,
                members,
                self._term("hasSourceOccurrence"),
                self._term("lineagePosition"),
            ):
                source = _one_value(by_subject[member], self._term("hasSourceOccurrence"))
                if not isinstance(source, RdfIri):
                    raise RdfCodecError("source membership must point to an IRI source occurrence")
                source_ids.append(_one_string(by_subject, source, self._term("sourceOccurrenceId")))
            return DirectLineage(tuple(source_ids))
        if kind != "derived":
            raise RdfCodecError("lineageType must be direct or derived")
        node = _one_value(by_subject[revision], self._term("derivation"))
        if not isinstance(node, RdfIri):
            raise RdfCodecError("derived lineage must point to an IRI derivation resource")
        members = _values(by_subject[node], self._term("inputMembership"))
        input_ids = tuple(
            _one_string(by_subject, member, self._term("inputRevisionId"))
            for member in _ordered_members(
                by_subject,
                members,
                self._term("inputRevisionId"),
                self._term("lineagePosition"),
            )
        )
        return DerivedLineage(
            rule_id=_one_string(by_subject, node, self._term("ruleId")),
            engine_version=_one_string(by_subject, node, self._term("engineVersion")),
            profile_version=_one_string(by_subject, node, self._term("profileVersion")),
            input_revision_ids=input_ids,
            input_digest=_one_string(by_subject, node, self._term("inputDigest")),
            run_id=_one_string(by_subject, node, self._term("runId")),
            generated_at=_one_literal(by_subject, node, self._term("generatedAt"), XSD_DATETIME_STAMP),
            derivation_reference=_optional_string(by_subject, node, self._term("derivationReference")),
        )

    def _validate_import_document(self, document: RdfImportDocument, limits: RdfImportLimits) -> None:
        if document.received_bytes > limits.max_bytes:
            raise RdfImportBudgetError("RDF import exceeds the byte budget")
        if len(document.dataset.triples) > limits.max_statements:
            raise RdfImportBudgetError("RDF import exceeds the statement-count budget")
        if document.parse_seconds > limits.max_parse_seconds:
            raise RdfImportBudgetError("RDF parser exceeded the time budget")
        if document.remote_contexts or document.followed_imports or document.parser_side_effects:
            raise RdfImportSecurityError(
                "RDF import parser must be offline with remote contexts, imports, and side effects disabled"
            )
        if document.document_uri is not None:
            self._validate_document_uri(document.document_uri, limits)
        self._validate_structural_graph(document.dataset, limits)
        for triple in document.dataset.triples:
            self._validate_term(triple.subject, limits, depth=0)
            self._validate_term(triple.predicate, limits, depth=0)
            self._validate_term(triple.object, limits, depth=0)
        self._validate_triple_term_positions(document.dataset)
        decoded_bytes = _dataset_byte_weight(document.dataset)
        if decoded_bytes > limits.max_bytes:
            raise RdfImportBudgetError("RDF decoded graph exceeds the byte budget")

    def _validate_triple_term_positions(self, dataset: RdfDataset) -> None:
        """Reject draft syntax the assertion contract would otherwise ignore."""
        for triple in dataset.triples:
            if isinstance(triple.subject, RdfTripleTerm):
                raise RdfCodecError(
                    "RDF triple terms are accepted only as the object of rdf:reifies"
                )
            if isinstance(triple.object, RdfTripleTerm) and triple.predicate.value != RDF_REIFIES:
                raise RdfCodecError(
                    "RDF triple terms are accepted only as the object of rdf:reifies"
                )

    def _validate_structural_graph(self, dataset: RdfDataset, limits: RdfImportLimits) -> None:
        edges: dict[RdfIri, set[RdfIri]] = defaultdict(set)
        for triple in dataset.triples:
            if (
                isinstance(triple.subject, RdfIri)
                and isinstance(triple.object, RdfIri)
                and triple.predicate.value in self._structural_predicates
            ):
                edges[triple.subject].add(triple.object)
        visiting: set[RdfIri] = set()
        completed: set[RdfIri] = set()

        def visit(node: RdfIri, depth: int) -> None:
            if depth > limits.max_nesting:
                raise RdfImportBudgetError("RDF structural graph exceeds the nesting budget")
            if node in visiting:
                raise RdfImportBudgetError("RDF structural graph contains a cycle")
            if node in completed:
                return
            visiting.add(node)
            for child in edges.get(node, ()):
                visit(child, depth + 1)
            visiting.remove(node)
            completed.add(node)

        for node in tuple(edges):
            visit(node, 0)

    def _validate_term(self, term: RdfTerm, limits: RdfImportLimits, *, depth: int) -> None:
        if depth > limits.max_nesting:
            raise RdfImportBudgetError("RDF term exceeds the nesting budget")
        if isinstance(term, RdfIri):
            if len(term.value.encode("utf-8")) > limits.max_term_bytes:
                raise RdfImportBudgetError("RDF IRI exceeds the term-size budget")
            self._validate_term_iri(term.value)
            return
        if isinstance(term, RdfBlankNode):
            raise RdfCodecError(
                "RDF blank/local identifiers cannot be promoted into a canonical assertion identity"
            )
        if isinstance(term, RdfLiteral):
            if len(term.lexical_form.encode("utf-8")) > limits.max_term_bytes:
                raise RdfImportBudgetError("RDF literal exceeds the term-size budget")
            self._validate_term_iri(term.datatype_iri)
            if term.direction is not None:
                raise UnsupportedRdfCapabilityError(
                    "RDF 1.2 directional literals cannot be represented by the canonical assertion contract"
                )
            return
        if self._rdf12 is None:
            raise UnsupportedRdfCapabilityError(
                "RDF 1.2 triple terms require an explicitly selected registry-pinned capability"
            )
        self._validate_term(term.subject, limits, depth=depth + 1)
        self._validate_term(term.predicate, limits, depth=depth + 1)
        self._validate_term(term.object, limits, depth=depth + 1)

    @staticmethod
    def _validate_term_iri(value: str) -> None:
        """Validate an inert RDF term IRI without treating it as a fetch target."""
        try:
            normalize_iri(value)
        except ValueError as error:
            raise RdfImportSecurityError(f"RDF import contains an invalid IRI: {value!r}") from error

    def _validate_document_uri(self, value: str, limits: RdfImportLimits) -> None:
        """Apply the scheme allowlist only to the document URI a parser may dereference."""
        self._validate_term_iri(value)
        normalized = normalize_iri(value)
        scheme = urlsplit(normalized).scheme.lower()
        if scheme not in limits.allowed_uri_schemes:
            raise RdfImportSecurityError(f"RDF import document URI scheme {scheme!r} is disabled")

    def _contains_triple_term(self, triple_groups: Iterable[tuple[RdfTriple, ...]]) -> bool:
        def contains(term: RdfTerm) -> bool:
            if isinstance(term, RdfTripleTerm):
                return True
            return False

        return any(
            contains(triple.subject) or contains(triple.object)
            for triples in triple_groups
            for triple in triples
        )

    def _term(self, name: str) -> str:
        return self._k + name

    @staticmethod
    def _node(assertion: Assertion, label: str) -> RdfIri:
        digest = hashlib.sha256(
            f"{assertion.assertion_id}\x00{assertion.revision_id}\x00{label}".encode("utf-8")
        ).hexdigest()
        return RdfIri(f"urn:kestrel:rdf:{label}:sha256:{digest}")


def _string(value: str) -> RdfLiteral:
    return RdfLiteral(value, XSD_STRING)


def _rdf_term(term: IRI | Literal) -> RdfIri | RdfLiteral:
    if isinstance(term, IRI):
        return RdfIri(term.value)
    return RdfLiteral(term.lexical_form, term.datatype_iri, term.language, term.direction)


def _canonical_statement_terms(
    subject: RdfTerm,
    predicate: RdfTerm,
    object_: RdfTerm,
) -> tuple[IRI, IRI, IRI | Literal]:
    if not isinstance(subject, RdfIri) or not isinstance(predicate, RdfIri):
        raise RdfCodecError("canonical assertion subject and predicate must be RDF IRIs")
    if not isinstance(object_, (RdfIri, RdfLiteral)):
        raise RdfCodecError(
            "blank/local IDs and RDF 1.2 triple terms cannot become canonical assertion objects"
        )
    if isinstance(object_, RdfLiteral) and object_.direction is not None:
        raise UnsupportedRdfCapabilityError(
            "RDF 1.2 directional literal cannot become a canonical assertion object"
        )
    try:
        object_value: IRI | Literal
        if isinstance(object_, RdfIri):
            object_value = IRI(object_.value)
        else:
            object_value = Literal(
                object_.lexical_form,
                object_.datatype_iri,
                language=object_.language,
                direction=object_.direction,
            )
        return IRI(subject.value), IRI(predicate.value), object_value
    except ValueError as error:
        raise RdfCodecError(f"RDF statement cannot form a canonical assertion: {error}") from error


def _term_byte_weight(term: RdfTerm) -> int:
    """Bound decoded graph expansion even when an adapter underreports bytes."""
    if isinstance(term, RdfIri):
        return len(term.value.encode("utf-8"))
    if isinstance(term, RdfBlankNode):
        return len(term.identifier.encode("utf-8"))
    if isinstance(term, RdfLiteral):
        return (
            len(term.lexical_form.encode("utf-8"))
            + len(term.datatype_iri.encode("utf-8"))
            + (len(term.language.encode("utf-8")) if term.language is not None else 0)
        )
    return (
        _term_byte_weight(term.subject)
        + _term_byte_weight(term.predicate)
        + _term_byte_weight(term.object)
    )


def _dataset_byte_weight(dataset: RdfDataset) -> int:
    """Return a conservative in-memory size bound for a typed RDF dataset."""
    return sum(
        _term_byte_weight(triple.subject)
        + _term_byte_weight(triple.predicate)
        + _term_byte_weight(triple.object)
        for triple in dataset.triples
    )


def _by_subject(triples: tuple[RdfTriple, ...]) -> dict[RdfIri | RdfBlankNode | RdfTripleTerm, tuple[RdfTriple, ...]]:
    grouped: dict[RdfIri | RdfBlankNode | RdfTripleTerm, list[RdfTriple]] = defaultdict(list)
    for triple in triples:
        grouped[triple.subject].append(triple)
    return {subject: tuple(items) for subject, items in grouped.items()}


def _values(triples: tuple[RdfTriple, ...], predicate: str) -> tuple[RdfTerm, ...]:
    return tuple(triple.object for triple in triples if triple.predicate.value == predicate)


def _has(triples: tuple[RdfTriple, ...], predicate: str, object_value: str | RdfTerm) -> bool:
    for value in _values(triples, predicate):
        if isinstance(object_value, str):
            if isinstance(value, RdfIri) and value.value == object_value:
                return True
        elif value == object_value:
            return True
    return False


def _one_value(triples: tuple[RdfTriple, ...], predicate: str) -> RdfTerm:
    values = _values(triples, predicate)
    if len(values) != 1:
        raise RdfCodecError(f"RDF projection requires exactly one <{predicate}> value")
    return values[0]


def _one_literal(by_subject, subject, predicate: str, datatype: str) -> str:
    value = _one_value(by_subject.get(subject, ()), predicate)
    if not isinstance(value, RdfLiteral) or value.datatype_iri != datatype or value.language is not None:
        raise RdfCodecError(f"RDF projection requires <{predicate}> as {datatype}")
    return value.lexical_form


def _optional_literal(by_subject, subject, predicate: str, datatype: str) -> str | None:
    values = _values(by_subject.get(subject, ()), predicate)
    if not values:
        return None
    if len(values) != 1 or not isinstance(values[0], RdfLiteral) or values[0].datatype_iri != datatype:
        raise RdfCodecError(f"RDF projection requires at most one <{predicate}> as {datatype}")
    return values[0].lexical_form


def _one_string(by_subject, subject, predicate: str) -> str:
    return _one_literal(by_subject, subject, predicate, XSD_STRING)


def _optional_string(by_subject, subject, predicate: str) -> str | None:
    return _optional_literal(by_subject, subject, predicate, XSD_STRING)


def _ordered_members(
    by_subject,
    values: tuple[RdfTerm, ...],
    target_predicate: str,
    position_predicate: str,
) -> tuple[RdfIri, ...]:
    indexed: list[tuple[int, RdfIri]] = []
    for value in values:
        if not isinstance(value, RdfIri):
            raise RdfCodecError("lineage membership must use IRI resources")
        # Ensure the member has the intended target relation before reading its
        # position; this catches a graph that tries to reuse arbitrary nodes.
        _one_value(by_subject.get(value, ()), target_predicate)
        text = _one_literal(by_subject, value, position_predicate, XSD_INTEGER)
        try:
            position = int(text)
        except ValueError as error:
            raise RdfCodecError("lineage position must be an xsd:integer") from error
        if position < 0:
            raise RdfCodecError("lineage position must be non-negative")
        indexed.append((position, value))
    indexed.sort()
    if [position for position, _ in indexed] != list(range(len(indexed))):
        raise RdfCodecError("lineage positions must be contiguous from zero")
    return tuple(value for _, value in indexed)


__all__ = [
    "RDF_NAMESPACE",
    "RDF_OBJECT",
    "RDF_PREDICATE",
    "RDF_REIFIES",
    "RDF_REIFIER",
    "RDF_STATEMENT",
    "RDF_SUBJECT",
    "RDF_TYPE",
    "RdfAssertionCodec",
    "RdfAssertionReadAdapter",
    "RdfBlankNode",
    "RdfCapabilityReport",
    "RdfCodecConfiguration",
    "RdfCodecError",
    "RdfDataset",
    "RdfImportBudgetError",
    "RdfImportDocument",
    "RdfImportLimits",
    "RdfImportOwnership",
    "RdfImportSecurityError",
    "RdfIri",
    "RdfLiteral",
    "RdfOwnershipError",
    "RdfProjectionKind",
    "RdfTerm",
    "RdfTriple",
    "RdfTripleTerm",
    "RdfTypedQuery",
    "Sparql11AssertionReadAdapter",
    "Sparql12AssertionReadAdapter",
    "UnsupportedRdfCapabilityError",
]
