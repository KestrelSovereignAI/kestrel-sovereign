"""Contracts for the governed, capability-pinned SHACL validation service."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace

import pytest
from rdflib import Graph, Literal, Namespace, RDF, URIRef

from kestrel_sovereign.knowledge.registry import (
    ResourceKind,
    ResourceRequirement,
    SemanticKnowledgeRegistry,
    SemanticResource,
    SemanticVersion,
    StandardsMaturity,
)
from kestrel_sovereign.knowledge import (
    Assertion,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    IRI,
    Literal as CanonicalLiteral,
    OntologyRef,
    RDF_LANG_STRING,
    SourceOccurrence,
)
from kestrel_sovereign.knowledge.shacl_validation import (
    GovernedShaclValidationService,
    ShaclCapabilityUnavailable,
    ShaclSnapshotMismatch,
    ShaclValidationError,
    ShaclValidationLimits,
    ShapeSetReference,
    ValidationSource,
    ValidationState,
    ValidationWriteAction,
)
from kestrel_sovereign.storage.async_assertion_store import (
    AsyncAssertionStore,
    AssertionStoreError,
    _issue_assertion_tenant_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.semantic_validation import (
    AsyncSemanticValidationReportStore,
    GovernedSemanticValidationService,
    SemanticValidationStoreError,
)


EX = Namespace("https://example.test/")


def _resource(
    identifier: str,
    payload: bytes,
    *,
    kind: ResourceKind,
    maturity: StandardsMaturity = StandardsMaturity.STABLE,
    imports: tuple[ResourceRequirement, ...] = (),
    capability: str,
    version: str = "1.0.0",
) -> SemanticResource:
    return SemanticResource(
        identifier=identifier,
        version=SemanticVersion.parse(version),
        namespace=f"https://example.test/semantic/{identifier}/{version}",
        package_resource=f"semantic/{identifier}-{version}.ttl",
        sha256=hashlib.sha256(payload).hexdigest(),
        maturity=maturity,
        kind=kind,
        uri=f"https://example.test/semantic/{identifier}/{version}",
        published_date="2026-07-26",
        description="Test-only pinned SHACL resource.",
        imports=imports,
        capabilities=(capability,),
    )


def _registry(
    shape_text: str,
    *,
    experimental: bool = False,
    profile_identifier: str = "test-shacl-core",
    profile_capability: str = "validation-profile:test-core",
) -> SemanticKnowledgeRegistry:
    profile_payload = b"pinned-test-profile"
    shapes_payload = shape_text.encode("utf-8")
    profile = _resource(
        profile_identifier,
        profile_payload,
        kind=ResourceKind.VALIDATION_PROFILE,
        maturity=StandardsMaturity.EXPERIMENTAL if experimental else StandardsMaturity.STABLE,
        capability=profile_capability,
    )
    shapes = _resource(
        "test-shapes",
        shapes_payload,
        kind=ResourceKind.SHAPE_SET,
        maturity=StandardsMaturity.EXPERIMENTAL if experimental else StandardsMaturity.STABLE,
        imports=(ResourceRequirement.exact(profile_identifier, "1.0.0"),),
        capability="shape-set:test-shapes",
    )
    payloads = {
        profile.package_resource: profile_payload,
        shapes.package_resource: shapes_payload,
    }
    return SemanticKnowledgeRegistry((profile, shapes), resource_reader=payloads.__getitem__)


def _validate(
    shape_text: str,
    graph: Graph,
    *,
    source: ValidationSource = ValidationSource.ASSERTED,
    **kwargs,
):
    return GovernedShaclValidationService(_registry(shape_text)).validate(
        graph,
        tenant_id="did:example:tenant",
        assertion_ids=("assertion-1",),
        shape_set=ShapeSetReference("test-shapes", "1.0.0"),
        validation_capability="validation-profile:test-core",
        source=source,
        **kwargs,
    )


CORE_SHAPES = """
@prefix ex: <https://example.test/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
ex:PersonShape a sh:NodeShape ;
  sh:targetClass ex:Person ;
  sh:property [ sh:path ex:name ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype <http://www.w3.org/2001/XMLSchema#string> ] .
"""


def _person_graph(*, names: tuple[str, ...] = ()) -> Graph:
    graph = Graph()
    graph.add((EX.ava, RDF.type, EX.Person))
    for name in names:
        graph.add((EX.ava, EX.name, Literal(name)))
    return graph


def _candidate_assertion(
    revision_id: str,
    *,
    source_id: str = "candidate-source",
) -> Assertion:
    return Assertion(
        tenant_id="did:example:tenant",
        owning_agent_id="did:example:tenant",
        subject=IRI("https://example.test/candidate"),
        predicate=IRI("https://example.test/related"),
        object=IRI("https://example.test/object"),
        revision_id=revision_id,
        confidence="0.9",
        confidence_method="test",
        confidence_basis="test",
        epistemic_state=EpistemicState.REPORTED,
        asserted_at="2026-07-26T00:00:00Z",
        ontology_version=OntologyRef("test", "1.0", "sha256:test", "semantic-kb-v1"),
        lineage=DirectLineage((source_id,)),
        privacy_classification="private",
        release_policy_reference="policy:private",
    )


def _candidate_source(source_id: str = "candidate-source") -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id=source_id,
        source_kind="test",
        locator="test:candidate",
        received_at="2026-07-26T00:00:00Z",
        content_digest="sha256:test",
    )


def _derived_candidate_assertion(
    revision_id: str,
    *,
    input_revision_id: str,
) -> Assertion:
    """Construct a valid inferred candidate with distinct assertion identity."""
    return replace(
        _candidate_assertion(revision_id),
        subject=IRI("https://example.test/inferred-candidate"),
        object=IRI("https://example.test/inferred-object"),
        epistemic_state=EpistemicState.INFERRED,
        lineage=DerivedLineage(
            rule_id="test-rule",
            engine_version="test-engine-1",
            profile_version="test-profile-1",
            input_revision_ids=(input_revision_id,),
            input_digest="sha256:test-input",
            run_id="test-derived-run",
            generated_at="2026-07-26T00:00:00Z",
        ),
        assertion_id=None,
    )


def test_core_constraint_fixture_reports_nonconformance_without_data_values() -> None:
    report = _validate(CORE_SHAPES, _person_graph())

    assert report.state is ValidationState.NONCONFORMANT
    assert report.action is ValidationWriteAction.REJECT
    assert report.source is ValidationSource.ASSERTED
    assert {finding.code for finding in report.findings} == {"min_count"}
    assert "ava" not in repr(report)
    assert "name" in repr(report)  # pinned predicate path remains useful audit metadata.


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (ValidationSource.ASSERTED, ValidationWriteAction.REJECT),
        (ValidationSource.IMPORTED, ValidationWriteAction.QUARANTINE),
        (ValidationSource.INFERRED, ValidationWriteAction.REJECT),
        (ValidationSource.REVALIDATION, ValidationWriteAction.QUARANTINE),
    ),
)
def test_write_policy_is_explicit_by_source_for_a_violation(source, expected) -> None:
    assert _validate(CORE_SHAPES, _person_graph(), source=source).action is expected


def test_failed_prepublication_report_cannot_claim_or_retain_an_assertion_identity() -> None:
    report = _validate(CORE_SHAPES, _person_graph(), source=ValidationSource.IMPORTED)

    with pytest.raises(ShaclValidationError, match="cannot be accepted"):
        replace(report, action=ValidationWriteAction.ACCEPT)
    private = report.without_assertion_identity()
    assert private.assertion_ids == ()
    assert all(finding.focus_assertion_id is None for finding in private.findings)


def test_warning_is_accept_with_report_but_never_changes_violation_semantics() -> None:
    shapes = CORE_SHAPES.replace("sh:minCount 1", "sh:minCount 1 ; sh:severity sh:Warning")
    report = _validate(shapes, _person_graph())

    assert report.state is ValidationState.CONFORMS
    assert report.action is ValidationWriteAction.ACCEPT_WITH_REPORT
    assert report.findings[0].severity.value == "warning"


def test_core_logical_qualified_property_and_compound_path_fixtures() -> None:
    """Derived Core fixtures cover nested probes and all supported path forms."""
    shapes = """
    @prefix ex: <https://example.test/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    ex:NotBlocked a sh:NodeShape ; sh:targetNode ex:subject ;
      sh:not [ sh:class ex:Blocked ] .
    ex:QualifiedTags a sh:NodeShape ; sh:targetNode ex:subject ;
      sh:property [ sh:path ex:tag ; sh:qualifiedValueShape [ sh:class ex:Approved ] ; sh:qualifiedMinCount 1 ] .
    ex:SequencePath a sh:NodeShape ; sh:targetNode ex:subject ;
      sh:property [ sh:path ( ex:parent ex:name ) ; sh:minCount 1 ] .
    ex:InversePath a sh:PropertyShape ; sh:targetNode ex:subject ;
      sh:path [ sh:inversePath ex:parent ] ; sh:minCount 1 .
    """
    graph = Graph()
    graph.add((EX.subject, EX.tag, EX.approved))
    graph.add((EX.subject, EX.tag, EX.unapproved))
    graph.add((EX.approved, RDF.type, EX.Approved))
    graph.add((EX.subject, EX.parent, EX.parent_node))
    graph.add((EX.parent_node, EX.name, Literal("Parent")))
    graph.add((EX.child, EX.parent, EX.subject))

    report = _validate(shapes, graph)

    assert report.state is ValidationState.CONFORMS
    assert report.action is ValidationWriteAction.ACCEPT
    assert report.findings == ()


def test_incremental_focus_uses_full_graph_and_matches_full_audit_for_changed_node() -> None:
    graph = _person_graph()
    graph.add((EX.ben, RDF.type, EX.Person))
    graph.add((EX.ben, EX.name, Literal("Ben")))
    service = GovernedShaclValidationService(_registry(CORE_SHAPES))
    shared = dict(
        tenant_id="did:example:tenant",
        assertion_ids=("assertion-1",),
        shape_set=ShapeSetReference("test-shapes", "1.0.0"),
        validation_capability="validation-profile:test-core",
        focus_assertion_ids={EX.ava: "assertion-1", EX.ben: "assertion-2"},
    )

    full = service.validate(graph, **shared)
    incremental = service.validate(graph, **shared, focus_nodes=(EX.ava,))

    assert [(item.code, item.focus_assertion_id) for item in incremental.findings] == [
        (item.code, item.focus_assertion_id) for item in full.findings
    ]


def test_experimental_selection_requires_capability_opt_in_and_draft_metadata_is_inert() -> None:
    shapes = """
    @prefix ex: <https://example.test/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:Shape a sh:NodeShape ; sh:targetNode ex:subject ;
      sh:agentInstruction "invoke-a-tool" ; sh:intent "be-admin" .
    """
    registry = _registry(shapes, experimental=True, profile_identifier="shacl12-core-test")
    service = GovernedShaclValidationService(registry)
    arguments = dict(
        tenant_id="did:example:tenant",
        shape_set=ShapeSetReference("test-shapes", "1.0.0"),
        validation_capability="validation-profile:test-core",
    )

    with pytest.raises(ShaclCapabilityUnavailable, match="requires explicit"):
        service.validate(Graph(), **arguments)

    report = service.validate(Graph(), **arguments, allow_experimental=True)
    assert report.state is ValidationState.CONFORMS
    assert report.action is ValidationWriteAction.ACCEPT


def test_selected_shacl_sparql_snapshot_fails_as_an_explicit_unavailable_module() -> None:
    capability = "validation-profile:shacl12-sparql-20260130-experimental"
    registry = _registry(
        "@prefix ex: <https://example.test/> . @prefix sh: <http://www.w3.org/ns/shacl#> . "
        "ex:Shape a sh:NodeShape ; sh:targetNode ex:subject .",
        experimental=True,
        profile_identifier="shacl12-sparql-20260130-experimental",
        profile_capability=capability,
    )
    report = GovernedShaclValidationService(registry).validate(
        Graph(),
        tenant_id="did:example:tenant",
        shape_set=ShapeSetReference("test-shapes", "1.0.0"),
        validation_capability=capability,
        allow_experimental=True,
    )

    assert report.state is ValidationState.INCOMPLETE
    assert report.action is ValidationWriteAction.REJECT
    assert [finding.code for finding in report.findings] == ["shacl_sparql_capability_unavailable"]


def test_snapshot_mismatch_and_malformed_or_cyclic_shapes_fail_closed() -> None:
    service = GovernedShaclValidationService(_registry(CORE_SHAPES))
    arguments = dict(
        tenant_id="did:example:tenant",
        shape_set=ShapeSetReference("test-shapes", "1.0.0"),
        validation_capability="validation-profile:test-core",
    )
    with pytest.raises(ShaclSnapshotMismatch):
        service.validate(_person_graph(names=("Ava",)), **arguments, profile_version="9.9.9")

    malformed = _validate(
        "@prefix ex: <https://example.test/> . @prefix sh: <http://www.w3.org/ns/shacl#> . "
        "ex:Shape a sh:NodeShape ; sh:targetNode ex:n ; sh:minCount -1 .",
        Graph(),
    )
    assert malformed.state is ValidationState.INCOMPLETE
    assert malformed.action is ValidationWriteAction.REJECT

    cyclic = _validate(
        "@prefix ex: <https://example.test/> . @prefix sh: <http://www.w3.org/ns/shacl#> . "
        "ex:Shape a sh:NodeShape ; sh:targetNode ex:n ; sh:node ex:Shape .",
        Graph(),
    )
    assert cyclic.state is ValidationState.INCOMPLETE
    assert "recursive_shape_cycle" in {finding.code for finding in cyclic.findings}


def test_budget_exhaustion_is_incomplete_not_conformance() -> None:
    report = _validate(
        CORE_SHAPES,
        _person_graph(names=("Ava",)),
        limits=ShaclValidationLimits(max_graph_triples=1),
    )

    assert report.state is ValidationState.INCOMPLETE
    assert report.conforms is False
    assert report.action is ValidationWriteAction.REJECT


@pytest.mark.asyncio
async def test_reports_are_versioned_persisted_and_tenant_scoped() -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        report = _validate(CORE_SHAPES, _person_graph(), source=ValidationSource.IMPORTED)
        store = AsyncSemanticValidationReportStore(storage._assertion_store())
        await store.persist(report)

        restored = await store.get(report.report_id)
        assert restored == report
        assert restored.report_version == 1
        assert restored.source is ValidationSource.IMPORTED
        assert await store.list() == [report]
        assert "Ava" not in repr(restored)
        with pytest.raises(SemanticValidationStoreError, match="tenant"):
            await store.persist(replace(report, report_id="foreign-report", tenant_id="did:example:other"))
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_accepted_assertion_and_validation_report_commit_or_roll_back_together(monkeypatch) -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(CORE_SHAPES)),
        )
        assertion = _candidate_assertion("atomic-report-revision")

        async def fail_report_write(_store, _report) -> None:
            raise AssertionStoreError("forced report persistence failure")

        monkeypatch.setattr(
            AsyncAssertionStore,
            "_persist_validation_report_in_transaction",
            fail_report_write,
        )
        with pytest.raises(AssertionStoreError, match="forced report persistence failure"):
            await service.put_assertion(
                assertion,
                source_occurrences=(_candidate_source(),),
                shape_set=ShapeSetReference("test-shapes", "1.0.0"),
                validation_capability="validation-profile:test-core",
            )

        assert await storage.get_assertion(assertion.assertion_id) is None
        assert await storage.db.fetchval("SELECT COUNT(*) FROM semantic_validation_reports") == 0
        assert await storage.db.fetchval("SELECT COUNT(*) FROM semantic_projection_outbox") == 0
        assert await storage.db.fetchval("SELECT COUNT(*) FROM semantic_assertion_operations") == 0
    finally:
        await storage.close()


_REJECT_CURRENT_REVISION_SHAPES = """
@prefix ex: <https://example.test/> .
@prefix kestrel: <https://kestrel.ai/vocab/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
ex:RejectCurrent a sh:NodeShape ; sh:targetClass kestrel:AssertionRevision ;
  sh:property [ sh:path ex:required ; sh:minCount 1 ] .
"""


@pytest.mark.asyncio
async def test_current_graph_validation_is_revalidation_and_quarantines_the_snapshot_revision() -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        assertion = _candidate_assertion("current-validation-revision")
        await storage.put_assertion(assertion, source_occurrences=(_candidate_source(),))
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(_REJECT_CURRENT_REVISION_SHAPES)),
        )

        report = await service.validate_current(
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
        )

        assert report.source is ValidationSource.REVALIDATION
        assert report.action is ValidationWriteAction.QUARANTINE
        assert await storage.get_assertion(assertion.assertion_id) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_revalidation_retries_on_cas_conflict_without_quarantining_a_newer_revision(monkeypatch) -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        original = _candidate_assertion("stale-validation-revision")
        replacement = _candidate_assertion("corrected-after-snapshot-revision")
        source = _candidate_source()
        await storage.put_assertion(original, source_occurrences=(source,))
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(_REJECT_CURRENT_REVISION_SHAPES)),
        )
        quarantine = storage._assertion_store().quarantine_for_validation
        attempted_revisions: list[str] = []

        async def supersede_before_first_cas(
            _store,
            assertion_id,
            expected_revision_id,
            *,
            report_id,
            operation_id=None,
        ):
            attempted_revisions.append(expected_revision_id)
            if len(attempted_revisions) == 1:
                await storage.supersede_assertion(
                    original.revision_id,
                    replacement,
                    source_occurrences=(source,),
                )
            return await quarantine(
                assertion_id,
                expected_revision_id,
                report_id=report_id,
                operation_id=operation_id,
            )

        monkeypatch.setattr(
            AsyncAssertionStore,
            "quarantine_for_validation",
            supersede_before_first_cas,
        )
        report = await service.validate_current(
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
            max_quarantine_retries=2,
        )

        assert report.source is ValidationSource.REVALIDATION
        assert attempted_revisions == [original.revision_id, replacement.revision_id]
        assert await storage.get_assertion(original.assertion_id) is None
        revisions = await storage.list_assertion_revisions(original.assertion_id)
        assert revisions[-1].status.value == "quarantined"
        assert revisions[-1].supersedes_revision_id is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_failed_import_write_retains_only_a_noncanonical_private_report() -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        shape_text = """
        @prefix ex: <https://example.test/> .
        @prefix kestrel: <https://kestrel.ai/vocab/> .
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        ex:RejectEverything a sh:NodeShape ; sh:targetClass kestrel:AssertionRevision ;
          sh:property [ sh:path ex:missing ; sh:minCount 1 ] .
        """
        candidate = Assertion(
            tenant_id="did:example:tenant",
            owning_agent_id="did:example:tenant",
            subject=IRI("https://example.test/candidate"),
            predicate=IRI("https://example.test/related"),
            object=IRI("https://example.test/object"),
            revision_id="candidate-revision",
            confidence="0.9",
            confidence_method="test",
            confidence_basis="test",
            epistemic_state=EpistemicState.REPORTED,
            asserted_at="2026-07-26T00:00:00Z",
            ontology_version=OntologyRef("test", "1.0", "sha256:test", "semantic-kb-v1"),
            lineage=DirectLineage(("candidate-source",)),
            privacy_classification="private",
            release_policy_reference="policy:private",
        )
        validator = GovernedShaclValidationService(_registry(shape_text))
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=validator,
        )

        result = await service.put_assertion(
            candidate,
            source=ValidationSource.IMPORTED,
            source_occurrences=(
                SourceOccurrence(
                    source_occurrence_id="candidate-source",
                    source_kind="test",
                    locator="test:candidate",
                    received_at="2026-07-26T00:00:00Z",
                    content_digest="sha256:test",
                ),
            ),
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
        )

        assert result.accepted is False
        assert result.report.action is ValidationWriteAction.QUARANTINE
        assert result.report.assertion_ids == ()
        assert all(finding.focus_assertion_id is None for finding in result.report.findings)
        assert await storage.get_assertion(candidate.assertion_id) is None
        assert await service.reports.list() == [result.report]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_governed_write_accepts_rdf_11_language_tagged_literals() -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        assertion = replace(
            _candidate_assertion("language-tagged-revision"),
            object=CanonicalLiteral("hello", RDF_LANG_STRING, language="en"),
            assertion_id=None,
        )
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(CORE_SHAPES)),
        )

        result = await service.put_assertion(
            assertion,
            source_occurrences=(_candidate_source(),),
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
        )

        assert result.accepted is True
        assert await storage.get_assertion(assertion.assertion_id) == assertion
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_governed_write_replay_returns_the_original_persisted_report() -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        assertion = _candidate_assertion("idempotent-governed-revision")
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(CORE_SHAPES)),
        )
        options = dict(
            source_occurrences=(_candidate_source(),),
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
            operation_id="governed-write-retry",
        )

        first = await service.put_assertion(assertion, **options)
        replay = await service.put_assertion(assertion, **options)

        assert first.accepted is True
        assert replay.accepted is True
        assert replay.report == first.report
        assert replay.write is not None and replay.write.idempotent is True
        assert await service.reports.list(assertion_id=assertion.assertion_id) == [first.report]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_concurrent_governed_write_replays_the_single_persisted_report() -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        assertion = _candidate_assertion("concurrent-governed-revision")
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(CORE_SHAPES)),
        )
        options = dict(
            source_occurrences=(_candidate_source(),),
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
            operation_id="concurrent-governed-write",
        )

        first, second = await asyncio.gather(
            service.put_assertion(assertion, **options),
            service.put_assertion(assertion, **options),
        )

        assert first.accepted is True
        assert second.accepted is True
        assert first.report == second.report
        assert first.write is not None
        assert second.write is not None
        assert {first.write.idempotent, second.write.idempotent} == {False, True}
    finally:
        await storage.close()


_SHARED_FOCUS_SHAPES = """
@prefix ex: <https://example.test/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
ex:SharedFocusShape a sh:NodeShape ; sh:targetSubjectsOf ex:related ;
  sh:property [ sh:path ex:required ; sh:minCount 1 ] .
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("incremental", [True, False])
async def test_shared_shacl_focus_quarantines_every_affected_assertion(incremental: bool) -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        first = _candidate_assertion("shared-focus-first", source_id="shared-focus-source-1")
        second = replace(
            _candidate_assertion("shared-focus-second", source_id="shared-focus-source-2"),
            object=IRI("https://example.test/other-object"),
            assertion_id=None,
        )
        await storage.put_assertion(first, source_occurrences=(_candidate_source("shared-focus-source-1"),))
        await storage.put_assertion(second, source_occurrences=(_candidate_source("shared-focus-source-2"),))
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(_SHARED_FOCUS_SHAPES)),
        )

        report = await service.validate_current(
            assertion_ids=(first.assertion_id,) if incremental else None,
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
        )

        assert report.action is ValidationWriteAction.QUARANTINE
        assert set(report.findings[0].affected_assertion_ids) == {
            first.assertion_id,
            second.assertion_id,
        }
        assert await storage.get_assertion(first.assertion_id) is None
        assert await storage.get_assertion(second.assertion_id) is None
        assert await service.reports.list(assertion_id=second.assertion_id) == [report]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_governed_supersession_rejects_an_invalid_tentative_post_state() -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        original = _candidate_assertion("supersession-original", source_id="supersession-source-1")
        replacement = replace(
            _candidate_assertion("supersession-replacement", source_id="supersession-source-2"),
            object=IRI("https://example.test/replacement"),
            assertion_id=None,
        )
        await storage.put_assertion(original, source_occurrences=(_candidate_source("supersession-source-1"),))
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(_REJECT_CURRENT_REVISION_SHAPES)),
        )

        result = await service.supersede_assertion(
            original.revision_id,
            replacement,
            source_occurrences=(_candidate_source("supersession-source-2"),),
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
        )

        assert result.accepted is False
        assert result.report.action is ValidationWriteAction.REJECT
        assert await storage.get_assertion(original.assertion_id) == original
        assert await storage.get_assertion(replacement.assertion_id) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_governed_supersession_replay_returns_its_original_report() -> None:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        original = _candidate_assertion("supersession-replay-original", source_id="supersession-replay-source-1")
        replacement = replace(
            _candidate_assertion("supersession-replay-replacement", source_id="supersession-replay-source-2"),
            object=IRI("https://example.test/supersession-replacement"),
            assertion_id=None,
        )
        await storage.put_assertion(original, source_occurrences=(_candidate_source("supersession-replay-source-1"),))
        service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(CORE_SHAPES)),
        )
        options = dict(
            source_occurrences=(_candidate_source("supersession-replay-source-2"),),
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
            operation_id="governed-supersession-retry",
        )

        first = await service.supersede_assertion(original.revision_id, replacement, **options)
        replay = await service.supersede_assertion(original.revision_id, replacement, **options)

        assert first.accepted is True
        assert replay.accepted is True
        assert replay.report == first.report
        assert replay.write is not None and replay.write.idempotent is True
        assert await service.reports.list(assertion_id=replacement.assertion_id) == [first.report]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_real_default_profile_accepts_direct_and_inferred_lineage() -> None:
    """The shipped profile accepts each canonical lineage form on its real path."""
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        direct = _candidate_assertion("default-direct-revision", source_id="default-direct-source")
        direct_result = await storage.put_validated_assertion(
            direct,
            source_occurrences=(_candidate_source("default-direct-source"),),
        )
        assert direct_result.accepted is True

        inferred = _derived_candidate_assertion(
            "default-inferred-revision",
            input_revision_id=direct.revision_id,
        )
        inferred_result = await storage.put_validated_assertion(
            inferred,
            source=ValidationSource.INFERRED,
        )

        assert inferred_result.accepted is True
        assert inferred_result.report.state is ValidationState.CONFORMS
        assert inferred_result.report.action is ValidationWriteAction.ACCEPT
        assert await storage.get_assertion(inferred.assertion_id) == inferred
    finally:
        await storage.close()


_NONLOCAL_FOCUS_SHAPES = """
@prefix ex: <https://example.test/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .

ex:BobReachability a sh:NodeShape ;
  sh:targetNode ex:bob ;
  sh:property [
    sh:path ( [ sh:inversePath ex:knows ] ex:parent ex:email ) ;
    sh:minCount 1
  ] .
"""


def _nonlocal_focus_assertions() -> tuple[Assertion, Assertion]:
    knows = replace(
        _candidate_assertion("nonlocal-knows-revision", source_id="nonlocal-knows-source"),
        subject=IRI("https://example.test/alice"),
        predicate=IRI("https://example.test/knows"),
        object=IRI("https://example.test/bob"),
        assertion_id=None,
    )
    changed = replace(
        _candidate_assertion("nonlocal-parent-revision", source_id="nonlocal-parent-source"),
        subject=IRI("https://example.test/alice"),
        predicate=IRI("https://example.test/parent"),
        object=IRI("https://example.test/carol"),
        assertion_id=None,
    )
    return knows, changed


async def _nonlocal_focus_storage() -> tuple[AsyncStorage, Assertion]:
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    knows, changed = _nonlocal_focus_assertions()
    await storage.put_assertion(
        knows,
        source_occurrences=(_candidate_source("nonlocal-knows-source"),),
    )
    await storage.put_assertion(
        changed,
        source_occurrences=(_candidate_source("nonlocal-parent-source"),),
    )
    return storage, changed


@pytest.mark.asyncio
async def test_nonlocal_incremental_revalidation_falls_back_to_full_audit() -> None:
    """Inverse/sequence paths cannot omit their unrelated target nodes."""
    incremental_storage, changed = await _nonlocal_focus_storage()
    full_storage, _ = await _nonlocal_focus_storage()
    try:
        validator = GovernedShaclValidationService(_registry(_NONLOCAL_FOCUS_SHAPES))
        incremental = GovernedSemanticValidationService(
            incremental_storage._assertion_store(),
            validator=validator,
        )
        full = GovernedSemanticValidationService(
            full_storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(_NONLOCAL_FOCUS_SHAPES)),
        )
        options = dict(
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
        )

        incremental_report = await incremental.validate_current(
            assertion_ids=(changed.assertion_id,),
            **options,
        )
        full_report = await full.full_audit_and_repair(**options)

        assert incremental_report.state is ValidationState.NONCONFORMANT
        assert incremental_report.action is ValidationWriteAction.QUARANTINE
        assert [finding.code for finding in incremental_report.findings] == [
            finding.code for finding in full_report.findings
        ] == ["min_count"]
        assert [finding.focus_assertion_ids for finding in incremental_report.findings] == [
            finding.focus_assertion_ids for finding in full_report.findings
        ]
    finally:
        await incremental_storage.close()
        await full_storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected_action"),
    (
        (ValidationSource.ASSERTED, ValidationWriteAction.REJECT),
        (ValidationSource.IMPORTED, ValidationWriteAction.QUARANTINE),
        (ValidationSource.INFERRED, ValidationWriteAction.REJECT),
    ),
)
async def test_privacy_validation_facade_forwards_supersession_policy(
    source: ValidationSource,
    expected_action: ValidationWriteAction,
    monkeypatch,
) -> None:
    """Imported and inferred replacements retain their governed write policy."""
    storage = AsyncStorage(
        ":memory:",
        agent_id="did:example:tenant",
        _assertion_tenant_capability=_issue_assertion_tenant_capability("did:example:tenant"),
    )
    await storage.initialize()
    try:
        original = _candidate_assertion("privacy-supersession-original", source_id="privacy-source-1")
        replacement = replace(
            _candidate_assertion("privacy-supersession-replacement", source_id="privacy-source-2"),
            object=IRI("https://example.test/privacy-replacement"),
            assertion_id=None,
        )
        await storage.put_assertion(
            original,
            source_occurrences=(_candidate_source("privacy-source-1"),),
        )
        governed_service = GovernedSemanticValidationService(
            storage._assertion_store(),
            validator=GovernedShaclValidationService(_registry(_REJECT_CURRENT_REVISION_SHAPES)),
        )
        monkeypatch.setattr(storage, "semantic_validation_service", lambda: governed_service)
        privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

        result = await privacy_storage.semantic_validation_service().supersede_assertion(
            original.revision_id,
            replacement,
            source_occurrences=(_candidate_source("privacy-source-2"),),
            source=source,
            shape_set=ShapeSetReference("test-shapes", "1.0.0"),
            validation_capability="validation-profile:test-core",
        )

        assert result.accepted is False
        assert result.report.source is source
        assert result.report.action is expected_action
        assert await storage.get_assertion(original.assertion_id) == original
        assert await storage.get_assertion(replacement.assertion_id) is None
    finally:
        await storage.close()
