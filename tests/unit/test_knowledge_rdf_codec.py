"""Conformance coverage for the bounded RDF assertion codec."""

from __future__ import annotations

from decimal import Decimal
import hashlib
from importlib import resources
import time

import pytest

from kestrel_sovereign.knowledge import (
    RDF_LANG_STRING,
    XSD_INTEGER,
    Assertion,
    AssertionQuery,
    AssertionResult,
    AssertionStatus,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    IRI,
    Literal,
    OntologyRef,
    RdfAssertionCodec,
    RdfBlankNode,
    RdfCodecConfiguration,
    RdfCodecError,
    RdfDataset,
    RdfImportBudgetError,
    RdfImportLimits,
    RdfImportOwnership,
    RdfIri,
    RdfLiteral,
    RdfOwnershipError,
    RdfProjectionKind,
    RdfTriple,
    RdfTripleTerm,
    RdfTypedQuery,
    TemporalInterval,
    UnsupportedRdfCapabilityError,
)
from kestrel_sovereign.knowledge.rdf_codec import (
    RDF_OBJECT,
    RDF_PREDICATE,
    RDF_REIFIES,
    RDF_SUBJECT,
    RDF_TYPE,
)
from kestrel_sovereign.knowledge.registry import get_knowledge_registry


TENANT = "did:example:rdf-codec"
OWNER = "did:example:rdf-codec"


def assertion(**overrides: object) -> Assertion:
    values: dict[str, object] = {
        "tenant_id": TENANT,
        "owning_agent_id": OWNER,
        "subject": IRI("https://example.test/subject"),
        "predicate": IRI("https://example.test/predicate"),
        "object": Literal("hello", RDF_LANG_STRING, language="en"),
        "revision_id": "revision-rdf-1",
        "confidence": Decimal("0.92"),
        "confidence_method": "operator-direct-v1",
        "confidence_basis": "operator-approved",
        "epistemic_state": EpistemicState.REPORTED,
        "asserted_at": "2026-07-26T14:02:11Z",
        "observed_time": TemporalInterval(start="2026-07-26T14:00:00Z"),
        "valid_time": TemporalInterval(
            start="2026-07-26T14:00:00Z", end="2026-07-27T14:00:00Z"
        ),
        "ontology_version": OntologyRef(
            "kestrel-vocab", "1.0.0", "sha256:ontology", "semantic-kb-v1"
        ),
        "lineage": DirectLineage(("source:one", "source:two")),
        "privacy_classification": "normal",
        "release_policy_reference": "policy:private-v1",
    }
    values.update(overrides)
    return Assertion(**values)  # type: ignore[arg-type]


def ownership() -> RdfImportOwnership:
    return RdfImportOwnership(TENANT, OWNER)


def document(codec: RdfAssertionCodec, value: Assertion) -> RdfDataset:
    return codec.project(value)


def _golden_digests() -> dict[str, str]:
    text = (
        resources.files("kestrel_sovereign")
        .joinpath("data", "semantic", "fixtures", "rdf11-projection-digests.txt")
        .read_text(encoding="utf-8")
    )
    return {
        name: digest
        for line in text.splitlines()
        if line and not line.startswith("#")
        for name, digest in (line.split(),)
    }


@pytest.mark.parametrize(
    ("golden_name", "object_value", "lineage", "status", "state", "supersedes"),
    (
        (
            "iri-direct-active",
            IRI("https://example.test/object"),
            DirectLineage(("source:one", "source:two")),
            AssertionStatus.ACTIVE,
            EpistemicState.REPORTED,
            None,
        ),
        (
            "typed-direct-superseded",
            Literal("001", XSD_INTEGER),
            DirectLineage(("source:one",)),
            AssertionStatus.SUPERSEDED,
            EpistemicState.OBSERVED,
            "revision-before",
        ),
        (
            "language-derived",
            Literal("hello", RDF_LANG_STRING, language="ar"),
            DerivedLineage(
                rule_id="rdfs-rule-1",
                engine_version="engine-1",
                profile_version="profile-1",
                input_revision_ids=("input-1", "input-2"),
                input_digest="sha256:inputs",
                run_id="run-1",
                generated_at="2026-07-26T14:02:12Z",
                derivation_reference="derivation:1",
            ),
            AssertionStatus.ACTIVE,
            EpistemicState.INFERRED,
            None,
        ),
        (
            "retracted",
            Literal("withdrawn"),
            DirectLineage(("source:one",)),
            AssertionStatus.RETRACTED,
            EpistemicState.RETRACTED,
            None,
        ),
        (
            "quarantined",
            Literal("isolated"),
            DirectLineage(("source:one",)),
            AssertionStatus.QUARANTINED,
            EpistemicState.REPORTED,
            None,
        ),
        (
            "deleted",
            Literal("removed"),
            DirectLineage(("source:one",)),
            AssertionStatus.DELETED,
            EpistemicState.REPORTED,
            None,
        ),
    ),
)
def test_rdf11_projection_round_trips_terms_lineage_temporal_metadata_and_lifecycle(
    golden_name, object_value, lineage, status, state, supersedes
):
    codec = RdfAssertionCodec()
    original = assertion(
        object=object_value,
        lineage=lineage,
        status=status,
        epistemic_state=state,
        supersedes_revision_id=supersedes,
    )

    projected = codec.project(original)
    restored = codec.import_projected_dataset(document(codec, original), ownership=ownership())

    assert restored == original
    assert projected.serialize_ntriples() == codec.serialize_rdf11_ntriples(original)
    assert b"rdf-syntax-ns#Statement" in projected.serialize_ntriples()
    assert b"projectedTenantId" in projected.serialize_ntriples()
    assert hashlib.sha256(projected.serialize_ntriples()).hexdigest() == _golden_digests()[golden_name]


def test_rdf11_golden_fixture_is_a_packaged_deterministic_export():
    value = Assertion(
        tenant_id="did:example:golden",
        owning_agent_id="did:example:golden",
        subject=IRI("https://example.test/subject"),
        predicate=IRI("https://example.test/predicate"),
        object=Literal("hello", RDF_LANG_STRING, language="en"),
        revision_id="golden-revision",
        confidence=Decimal("0.75"),
        confidence_method="operator-v1",
        confidence_basis="golden",
        epistemic_state=EpistemicState.REPORTED,
        asserted_at="2026-07-26T14:02:11Z",
        ontology_version=OntologyRef(
            "kestrel-vocab", "1.0.0", "sha256:golden", "semantic-kb-v1"
        ),
        lineage=DirectLineage(("source:one", "source:two")),
        privacy_classification="normal",
        release_policy_reference="policy:golden-v1",
    )
    fixture = (
        resources.files("kestrel_sovereign")
        .joinpath("data", "semantic", "fixtures", "rdf11-direct-language.nt")
        .read_bytes()
    )

    codec = RdfAssertionCodec()
    assert codec.serialize_rdf11_ntriples(value) == fixture
    assert codec.capability_report.rdf11.identifier == "rdf11-concepts-20140225"
    assert codec.capability_report.ntriples11.identifier == "ntriples-20140225"
    assert codec.capability_report.assertion_ontology.version == get_knowledge_registry().resolve(
        "kestrel-vocab", "1.1.0"
    ).version
    assert codec.capability_report.prov_o.identifier == "prov-o-20130430"
    assert codec.capability_report.owl_time.identifier == "owl-time-20171019"


def test_serialized_rdf11_import_uses_the_fixed_offline_ntriples_parser():
    expected = Assertion(
        tenant_id="did:example:golden",
        owning_agent_id="did:example:golden",
        subject=IRI("https://example.test/subject"),
        predicate=IRI("https://example.test/predicate"),
        object=Literal("hello", RDF_LANG_STRING, language="en"),
        revision_id="golden-revision",
        confidence=Decimal("0.75"),
        confidence_method="operator-v1",
        confidence_basis="golden",
        epistemic_state=EpistemicState.REPORTED,
        asserted_at="2026-07-26T14:02:11Z",
        ontology_version=OntologyRef(
            "kestrel-vocab", "1.0.0", "sha256:golden", "semantic-kb-v1"
        ),
        lineage=DirectLineage(("source:one", "source:two")),
        privacy_classification="normal",
        release_policy_reference="policy:golden-v1",
    )
    fixture = (
        resources.files("kestrel_sovereign")
        .joinpath("data", "semantic", "fixtures", "rdf11-direct-language.nt")
        .read_bytes()
    )

    assert RdfAssertionCodec().import_assertion(
        fixture,
        ownership=RdfImportOwnership("did:example:golden", "did:example:golden"),
    ) == expected


def test_serialized_rdf_import_rejects_non_bytes_before_loading_a_parser():
    with pytest.raises(RdfCodecError, match="raw bytes"):
        RdfAssertionCodec().import_assertion("not RDF bytes", ownership=ownership())  # type: ignore[arg-type]


def test_serialized_rdf_parser_enforces_raw_budgets_and_has_no_format_switch():
    from kestrel_sovereign.knowledge import rdf_parser

    codec = RdfAssertionCodec()
    with pytest.raises(RdfImportBudgetError, match="byte budget"):
        codec.import_assertion(b"x" * 101, ownership=ownership(), limits=RdfImportLimits(max_bytes=100))
    with pytest.raises(RdfCodecError, match="N-Triples"):
        codec.import_assertion(
            b'{"@context": "https://example.test/remote-context"}', ownership=ownership()
        )
    with pytest.raises(RdfImportBudgetError, match="statement-count"):
        codec.import_assertion(
            b"<https://example.test/s1> <https://example.test/p> <https://example.test/o> .\n"
            b"<https://example.test/s2> <https://example.test/p> <https://example.test/o> .\n",
            ownership=ownership(),
            limits=RdfImportLimits(max_statements=1),
        )
    # The invalid third line would be reached by an unrestricted full-graph
    # parse.  The streaming sink rejects on the second triple first.
    with pytest.raises(RdfImportBudgetError, match="statement-count"):
        codec.import_assertion(
            b"<https://example.test/s1> <https://example.test/p> <https://example.test/o> .\n"
            b"<https://example.test/s2> <https://example.test/p> <https://example.test/o> .\n"
            b"this is not N-Triples\n",
            ownership=ownership(),
            limits=RdfImportLimits(max_statements=1),
        )

    # The adapter uses RDFLib's streaming N-Triples sink, never Graph.parse(),
    # so parsed triples are capped before an unrestricted graph is materialized.
    assert not hasattr(rdf_parser, "Graph")
    with pytest.raises(RdfCodecError, match="requires exactly one"):
        codec.import_assertion(
            b"<urn:kestrel:inert-term> <https://example.test/p> <https://example.test/o> .\n",
            ownership=ownership(),
        )

    # The parent owns the deadline, so a worker must be terminated even if it
    # cannot yield to a parser-local monotonic-time check.
    with pytest.raises(RdfImportBudgetError, match="parser exceeded"):
        codec.import_assertion(
            b"<https://example.test/s> <https://example.test/p> <https://example.test/o> .\n",
            ownership=ownership(),
            limits=RdfImportLimits(max_parse_seconds=1e-12),
        )


def test_serialized_rdf_parser_parent_terminates_a_stalled_worker():
    """The time limit is enforceable even when a parser worker never yields."""
    from kestrel_sovereign.knowledge import rdf_parser

    context = rdf_parser._parser_context()
    receive, send = context.Pipe(duplex=False)
    worker = context.Process(target=time.sleep, args=(60,), daemon=True)
    worker.start()
    try:
        with pytest.raises(RdfImportBudgetError, match="parser exceeded"):
            rdf_parser._receive_worker_result(
                receive,
                worker,
                time.monotonic(),
                RdfImportLimits(max_parse_seconds=0.01),
            )
        assert not worker.is_alive()
    finally:
        send.close()
        receive.close()
        if worker.is_alive():
            rdf_parser._stop_worker(worker)
        else:
            worker.join()


def test_rdf12_projection_requires_exact_explicit_registry_capability_without_fallback():
    stable = RdfAssertionCodec()
    value = assertion()

    with pytest.raises(UnsupportedRdfCapabilityError, match="not substituted"):
        stable.project(value, projection=RdfProjectionKind.RDF12_TRIPLE_TERM)

    registry = get_knowledge_registry()
    capability = registry.select_capability(
        "rdf-profile:rdf12-cr-20260407-experimental", allow_experimental=True
    )
    experimental = RdfAssertionCodec(
        configuration=RdfCodecConfiguration(
            rdf12_capability="rdf-profile:rdf12-cr-20260407-experimental",
            rdf12_version=str(capability.resource.version),
        )
    )
    projection = experimental.project(value, projection=RdfProjectionKind.RDF12_TRIPLE_TERM)
    restored = experimental.import_projected_dataset(projection, ownership=ownership())

    assert restored == value
    assert experimental.capability_report.rdf12 == capability.resource.pin
    assert any(
        triple.predicate.value.endswith("reifies") and isinstance(triple.object, RdfTripleTerm)
        for triple in projection.triples
    )
    unbound_triple_term = RdfTripleTerm(
        RdfIri("https://example.test/unbound-s"),
        RdfIri("https://example.test/unbound-p"),
        RdfIri("https://example.test/unbound-o"),
    )
    with pytest.raises(RdfCodecError, match="only as the object of rdf:reifies"):
        experimental.import_projected_dataset(
            RdfDataset(
                projection.triples
                + (
                    RdfTriple(
                        RdfIri("https://example.test/extra"),
                        RdfIri("https://example.test/metadata"),
                        unbound_triple_term,
                    ),
                )
            ),
            ownership=ownership(),
        )


@pytest.mark.parametrize(
    ("status", "epistemic_state"),
    (
        (AssertionStatus.ACTIVE, EpistemicState.HYPOTHESIS),
        (AssertionStatus.SUPERSEDED, EpistemicState.OBSERVED),
        (AssertionStatus.RETRACTED, EpistemicState.RETRACTED),
        (AssertionStatus.QUARANTINED, EpistemicState.DISPUTED),
        (AssertionStatus.DELETED, EpistemicState.REPORTED),
    ),
)
def test_rdf12_reifier_projection_never_materializes_a_canonical_claim_as_a_fact(
    status, epistemic_state
):
    registry = get_knowledge_registry()
    capability = registry.select_capability(
        "rdf-profile:rdf12-cr-20260407-experimental", allow_experimental=True
    )
    codec = RdfAssertionCodec(
        configuration=RdfCodecConfiguration(
            rdf12_capability="rdf-profile:rdf12-cr-20260407-experimental",
            rdf12_version=str(capability.resource.version),
        )
    )
    value = assertion(status=status, epistemic_state=epistemic_state)

    projection = codec.project(value, projection=RdfProjectionKind.RDF12_TRIPLE_TERM)
    revision = next(
        triple.object
        for triple in projection.triples
        if triple.predicate.value.endswith("hasRevision")
    )
    assert isinstance(revision, RdfIri)
    assert any(
        triple.subject == revision
        and triple.predicate.value == RDF_TYPE
        and triple.object == RdfIri("https://kestrel.ai/vocab/AssertionRevision")
        for triple in projection.triples
    )
    assert not any(
        triple.subject == revision
        and triple.predicate.value == RDF_TYPE
        and triple.object == RdfIri("http://www.w3.org/1999/02/22-rdf-syntax-ns#Reifier")
        for triple in projection.triples
    )
    assert [
        triple
        for triple in projection.triples
        if (
            triple.subject == RdfIri(value.subject.value)
            and triple.predicate == RdfIri(value.predicate.value)
            and triple.object
            == RdfLiteral(value.object.lexical_form, value.object.datatype_iri, value.object.language)
        )
    ] == []
    reifying_triples = [
        triple
        for triple in projection.triples
        if triple.subject == revision
        and triple.predicate.value == RDF_REIFIES
        and isinstance(triple.object, RdfTripleTerm)
    ]
    assert len(reifying_triples) == 1


@pytest.mark.parametrize(
    "configuration",
    (
        RdfCodecConfiguration(
            rdf12_capability="rdf-profile:rdf12-cr-20260407-experimental",
            rdf12_version="9.9.9",
        ),
        RdfCodecConfiguration(
            rdf12_capability="rdf-profile:not-a-pinned-snapshot",
            rdf12_version="0.1.0",
        ),
        RdfCodecConfiguration(
            sparql12_capability="query-profile:sparql12-20260605-experimental",
            sparql12_version="9.9.9",
        ),
    ),
)
def test_unknown_or_wrong_draft_selection_fails_without_compatibility_guess(configuration):
    with pytest.raises(UnsupportedRdfCapabilityError):
        RdfAssertionCodec(configuration=configuration)


def test_rdf12_triple_terms_and_directional_literals_are_not_silently_lossy():
    codec = RdfAssertionCodec()
    triple_term = RdfTripleTerm(
        RdfIri("https://example.test/s"),
        RdfIri("https://example.test/p"),
        RdfIri("https://example.test/o"),
    )
    triple_dataset = RdfDataset(
        (RdfTriple(RdfIri("https://example.test/r"), RdfIri("https://example.test/p"), triple_term),)
    )
    directional_dataset = RdfDataset(
        (
            RdfTriple(
                RdfIri("https://example.test/r"),
                RdfIri("https://example.test/p"),
                RdfLiteral("rtl", RDF_LANG_STRING, "ar", "rtl"),
            ),
        )
    )

    with pytest.raises(UnsupportedRdfCapabilityError, match="triple terms"):
        codec.import_projected_dataset(triple_dataset, ownership=ownership())
    with pytest.raises(UnsupportedRdfCapabilityError, match="directional"):
        codec.import_projected_dataset(directional_dataset, ownership=ownership())


def test_typed_projection_rejects_blank_local_identifiers():
    codec = RdfAssertionCodec()
    dataset = RdfDataset(
        (
            RdfTriple(
                RdfIri("https://example.test/s"),
                RdfIri("https://example.test/p"),
                RdfBlankNode("source-local"),
            ),
        )
    )

    with pytest.raises(RdfCodecError, match="blank/local"):
        codec.import_projected_dataset(dataset, ownership=ownership())


def test_import_rejects_statement_and_nesting_budgets_and_structural_cycles():
    codec = RdfAssertionCodec()
    value = assertion()
    good = document(codec, value)
    with pytest.raises(RdfImportBudgetError, match="statement-count"):
        codec.import_projected_dataset(good, ownership=ownership(), limits=RdfImportLimits(max_statements=1))
    with pytest.raises(RdfImportBudgetError, match="byte budget"):
        codec.import_projected_dataset(
            good,
            ownership=ownership(),
            limits=RdfImportLimits(max_bytes=100),
        )

    revision = next(
        triple.object
        for triple in good.triples
        if triple.predicate.value.endswith("hasRevision")
    )
    assert isinstance(revision, RdfIri)
    cyclic = RdfDataset(
        good.triples
        + (RdfTriple(revision, RdfIri("https://kestrel.ai/vocab/hasRevision"), revision),)
    )
    with pytest.raises(RdfImportBudgetError, match="cycle"):
        codec.import_projected_dataset(cyclic, ownership=ownership())

    nested = RdfTripleTerm(
        RdfIri("https://example.test/s"),
        RdfIri("https://example.test/p"),
        RdfTripleTerm(
            RdfIri("https://example.test/s2"),
            RdfIri("https://example.test/p2"),
            RdfIri("https://example.test/o2"),
        ),
    )
    registry = get_knowledge_registry()
    selected = registry.select_capability(
        "rdf-profile:rdf12-cr-20260407-experimental", allow_experimental=True
    )
    rdf12 = RdfAssertionCodec(
        configuration=RdfCodecConfiguration(
            rdf12_capability="rdf-profile:rdf12-cr-20260407-experimental",
            rdf12_version=str(selected.resource.version),
        )
    )
    nested_dataset = RdfDataset(
        (RdfTriple(RdfIri("https://example.test/r"), RdfIri("https://example.test/p"), nested),)
    )
    with pytest.raises(RdfImportBudgetError, match="nesting"):
        rdf12.import_projected_dataset(
            nested_dataset, ownership=ownership(), limits=RdfImportLimits(max_nesting=1)
        )


def test_deep_rdf12_triple_term_exceeds_nesting_budget_without_recursing():
    registry = get_knowledge_registry()
    selected = registry.select_capability(
        "rdf-profile:rdf12-cr-20260407-experimental", allow_experimental=True
    )
    rdf12 = RdfAssertionCodec(
        configuration=RdfCodecConfiguration(
            rdf12_capability="rdf-profile:rdf12-cr-20260407-experimental",
            rdf12_version=str(selected.resource.version),
        )
    )
    nested: RdfIri | RdfTripleTerm = RdfIri("https://example.test/leaf")
    for _ in range(1_200):
        nested = RdfTripleTerm(
            RdfIri("https://example.test/subject"),
            RdfIri("https://example.test/predicate"),
            nested,
        )

    with pytest.raises(RdfImportBudgetError, match="nesting"):
        rdf12.import_projected_dataset(
            RdfDataset(
                (
                    RdfTriple(
                        RdfIri("https://example.test/revision"),
                        RdfIri(RDF_REIFIES),
                        nested,
                    ),
                )
            ),
            ownership=ownership(),
            limits=RdfImportLimits(max_nesting=32),
        )


@pytest.mark.parametrize("ownership_field", ("projectedTenantId", "projectedOwnerId"))
def test_import_uses_governed_ownership_instead_of_rdf_tenant_claim(ownership_field):
    codec = RdfAssertionCodec()
    value = assertion()
    projected = codec.project(value)
    replacement = tuple(
        RdfTriple(triple.subject, triple.predicate, RdfLiteral("did:example:attacker"))
        if triple.predicate.value.endswith(ownership_field)
        else triple
        for triple in projected.triples
    )

    with pytest.raises(RdfOwnershipError, match="cannot self-assert"):
        codec.import_projected_dataset(RdfDataset(replacement), ownership=ownership())


def test_inert_canonical_iris_round_trip_without_enabling_document_dereferencing():
    codec = RdfAssertionCodec()
    original = assertion(
        subject=IRI("tel:+15551234"),
        object=IRI("mailto:alice@example.com"),
    )

    assert codec.import_projected_dataset(document(codec, original), ownership=ownership()) == original


def test_typed_query_compiles_all_supported_narrowing_fields_deterministically():
    codec = RdfAssertionCodec()
    request = codec.typed_query(
        AssertionQuery(
            subject=IRI("https://example.test/subject"),
            predicate=IRI("https://example.test/predicate"),
            object=Literal("object"),
            assertion_ids=("urn:kestrel:assertion:sha256:" + "a" * 64,),
            statuses=(AssertionStatus.ACTIVE,),
            epistemic_states=(EpistemicState.REPORTED,),
            valid_at="2026-07-26T14:00:00Z",
            observed_at="2026-07-26T15:00:00Z",
            limit=2,
        ),
        ownership(),
    )
    seen: list[str] = []
    profiles = []

    class Backend:
        def execute_readonly(self, query_text: str, *, profile, timeout_seconds: float):
            seen.append(query_text)
            profiles.append((profile, timeout_seconds))
            return ()

        def cancel_readonly(self, *, profile):
            pytest.fail("an empty result must not be cancelled")

    adapter = codec.sparql11_read_adapter(Backend(), lambda row, owner: pytest.fail("no rows"))
    assert adapter.read_assertions(request) == ()
    assert "# Kestrel SPARQL 1.1 typed assertion read" in seen[0]
    assert "VALUES ?assertionId" in seen[0]
    assert "VALUES ?status" in seen[0]
    assert "VALUES ?epistemicState" in seen[0]
    assert "projectedOwnerId" in seen[0]
    assert f'"{OWNER}"^^<http://www.w3.org/2001/XMLSchema#string>' in seen[0]
    assert "?validInterval" in seen[0]
    assert "?observedInterval" in seen[0]
    assert "ORDER BY ?assertionId ?revisionId LIMIT 2" in seen[0]
    assert profiles[0][0] == codec.capability_report.sparql11
    assert profiles[0][1] > 0
    with pytest.raises(RdfCodecError):
        RdfTypedQuery("not-an-assertion-query", ownership())


@pytest.mark.parametrize(
    ("field", "foreign_value"),
    (
        ("tenant_id", "did:example:another-tenant"),
        ("owning_agent_id", "did:example:another-source"),
    ),
)
def test_sparql_typed_read_filters_and_post_validates_governed_ownership(field, foreign_value):
    codec = RdfAssertionCodec()
    unsafe_result = AssertionResult(
        assertion(**{field: foreign_value}), matched_revision_id="revision-rdf-1"
    )

    class Backend:
        def execute_readonly(self, query_text: str, *, profile, timeout_seconds: float):
            assert "projectedTenantId" in query_text
            assert "projectedOwnerId" in query_text
            return ({"result": unsafe_result},)

        def cancel_readonly(self, *, profile):
            pytest.fail("one result must not be cancelled")

    adapter = codec.sparql11_read_adapter(Backend(), lambda row, owner: row["result"])
    with pytest.raises(RdfOwnershipError, match="governed tenant/source ownership"):
        adapter.read_assertions(codec.typed_query(AssertionQuery(limit=1), ownership()))


def test_sparql_cursor_is_explicitly_rejected_until_the_canonical_encoding_is_negotiated():
    codec = RdfAssertionCodec()

    class Backend:
        def execute_readonly(self, query_text: str, *, profile, timeout_seconds: float):
            pytest.fail("a rejected cursor must not reach a backend")

        def cancel_readonly(self, *, profile):
            pytest.fail("a rejected cursor must not be cancelled")

    adapter = codec.sparql11_read_adapter(Backend(), lambda row, owner: pytest.fail("no rows"))
    request = codec.typed_query(AssertionQuery(cursor="opaque-cursor"), ownership())
    with pytest.raises(RdfCodecError, match="cursor"):
        adapter.read_assertions(request)


def test_sparql_adapter_stops_after_limit_plus_one_and_cancels_an_overproducing_backend():
    codec = RdfAssertionCodec()
    yielded = 0
    cancelled = []

    class Backend:
        def execute_readonly(self, query_text: str, *, profile, timeout_seconds: float):
            nonlocal yielded
            while True:
                yielded += 1
                yield {"row": yielded}

        def cancel_readonly(self, *, profile):
            cancelled.append(profile)

    adapter = codec.sparql11_read_adapter(Backend(), lambda row, owner: pytest.fail("no rows"))
    request = codec.typed_query(AssertionQuery(limit=2), ownership())

    with pytest.raises(RdfImportBudgetError, match="more rows"):
        adapter.read_assertions(request)

    assert yielded == 3
    assert cancelled == [codec.capability_report.sparql11]


def test_sparql12_adapter_is_selected_only_with_its_exact_pinned_capability():
    registry = get_knowledge_registry()
    rdf12_capability = registry.select_capability(
        "rdf-profile:rdf12-cr-20260407-experimental", allow_experimental=True
    )
    sparql12_capability = registry.select_capability(
        "query-profile:sparql12-20260605-experimental", allow_experimental=True
    )
    codec = RdfAssertionCodec(
        configuration=RdfCodecConfiguration(
            rdf12_capability="rdf-profile:rdf12-cr-20260407-experimental",
            rdf12_version=str(rdf12_capability.resource.version),
            sparql12_capability="query-profile:sparql12-20260605-experimental",
            sparql12_version=str(sparql12_capability.resource.version),
        )
    )
    seen: list[str] = []
    profiles = []

    class Backend:
        def execute_readonly(self, query_text: str, *, profile, timeout_seconds: float):
            seen.append(query_text)
            profiles.append(profile)
            return ()

        def cancel_readonly(self, *, profile):
            pytest.fail("an empty result must not be cancelled")

    assert codec.capability_report.sparql12 == sparql12_capability.resource.pin
    assert codec.sparql12_read_adapter(Backend(), lambda row, owner: pytest.fail("no rows")).read_assertions(
        codec.typed_query(AssertionQuery(limit=1), ownership())
    ) == ()
    assert "# Kestrel SPARQL 1.2 typed assertion read" in seen[0]
    assert profiles == [sparql12_capability.resource.pin]

    with pytest.raises(UnsupportedRdfCapabilityError, match="SPARQL 1.1 was not substituted"):
        RdfAssertionCodec().sparql12_read_adapter(Backend(), lambda row, owner: pytest.fail("no rows"))

    sparql12_only = RdfAssertionCodec(
        configuration=RdfCodecConfiguration(
            sparql12_capability="query-profile:sparql12-20260605-experimental",
            sparql12_version=str(sparql12_capability.resource.version),
        )
    )
    with pytest.raises(UnsupportedRdfCapabilityError, match="RDF 1.2 triple-term/reifier"):
        sparql12_only.sparql12_read_adapter(Backend(), lambda row, owner: pytest.fail("no rows"))


def test_sparql12_typed_read_matches_an_actual_rdf12_reifier_projection():
    registry = get_knowledge_registry()
    rdf12 = registry.select_capability(
        "rdf-profile:rdf12-cr-20260407-experimental", allow_experimental=True
    )
    sparql12 = registry.select_capability(
        "query-profile:sparql12-20260605-experimental", allow_experimental=True
    )
    codec = RdfAssertionCodec(
        configuration=RdfCodecConfiguration(
            rdf12_capability="rdf-profile:rdf12-cr-20260407-experimental",
            rdf12_version=str(rdf12.resource.version),
            sparql12_capability="query-profile:sparql12-20260605-experimental",
            sparql12_version=str(sparql12.resource.version),
        )
    )
    value = assertion()
    projection = codec.project(value, projection=RdfProjectionKind.RDF12_TRIPLE_TERM)
    expected_statement = RdfTripleTerm(
        RdfIri(value.subject.value),
        RdfIri(value.predicate.value),
        RdfLiteral(value.object.lexical_form, value.object.datatype_iri, value.object.language),
    )

    class ProjectionBackend:
        def execute_readonly(self, query_text: str, *, profile, timeout_seconds: float):
            assert profile == sparql12.resource.pin
            assert f"<{RDF_REIFIES}> <<( " in query_text
            assert f"<<( <{value.subject.value}> <{value.predicate.value}> \"hello\"@en )>>" in query_text
            assert RDF_SUBJECT not in query_text
            assert RDF_PREDICATE not in query_text
            assert RDF_OBJECT not in query_text
            assert any(
                triple.predicate.value == RDF_REIFIES and triple.object == expected_statement
                for triple in projection.triples
            )
            return ({"result": AssertionResult(value, matched_revision_id=value.revision_id)},)

        def cancel_readonly(self, *, profile):
            pytest.fail("one matching projection result must not be cancelled")

    result = codec.sparql12_read_adapter(
        ProjectionBackend(), lambda row, owner: row["result"]
    ).read_assertions(
        codec.typed_query(
            AssertionQuery(
                subject=value.subject,
                predicate=value.predicate,
                object=value.object,
                limit=1,
            ),
            ownership(),
        )
    )

    assert result == (AssertionResult(value, matched_revision_id=value.revision_id),)
