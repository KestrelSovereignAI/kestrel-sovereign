"""Conformance tests for the dependency-free semantic assertion contract."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
import subprocess
import sys

import pytest

from kestrel_sovereign.knowledge import (
    IDENTITY_VERSION,
    RDF_LANG_STRING,
    XSD_BOOLEAN,
    XSD_DATE,
    XSD_DATETIME,
    XSD_DECIMAL,
    XSD_INTEGER,
    XSD_STRING,
    XSD_TIME,
    Assertion,
    AssertionQuery,
    AssertionResult,
    AssertionStatus,
    AssertionValidationError,
    BlankNode,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    IRI,
    Instant,
    Literal,
    LocalIdentifier,
    OntologyRef,
    SourceOccurrence,
    TemporalInterval,
    derive_assertion_id,
    identity_preimage,
    normalize_iri,
)


TENANT = "did:example:ava"
SUBJECT = IRI("urn:kestrel:agent:did:example:ava:principal:user")
PREDICATE = IRI("https://kestrel.ai/vocab/preferredDeployRegion")
ONTOLOGY = OntologyRef("kestrel-vocab", "1", "sha256:ontology", "semantic-kb-v1")


def direct_assertion(**overrides: object) -> Assertion:
    values: dict[str, object] = {
        "tenant_id": TENANT,
        "owning_agent_id": "did:example:ava",
        "subject": SUBJECT,
        "predicate": PREDICATE,
        "object": Literal("us-central1", XSD_STRING),
        "revision_id": "revision-1",
        "confidence": Decimal("0.92"),
        "confidence_method": "user_direct_statement-v1",
        "confidence_basis": "operator-approved",
        "epistemic_state": EpistemicState.REPORTED,
        "asserted_at": "2026-07-26T09:02:11-05:00",
        "observed_time": TemporalInterval(start="2026-07-26T09:02:11-05:00"),
        "valid_time": TemporalInterval(start="2026-07-26T14:02:11Z"),
        "ontology_version": ONTOLOGY,
        "lineage": DirectLineage(("conversation_history:884#message-body",)),
        "privacy_classification": "normal",
        "release_policy_reference": "policy:private-v1",
    }
    values.update(overrides)
    return Assertion(**values)  # type: ignore[arg-type]


def test_fixed_identity_vector_and_dict_ordering_are_exact() -> None:
    object_value = Literal("us-central1", XSD_STRING)
    expected_preimage = (
        b'{"identity_version":"kestrel-assertion-id-v1","object":{"datatype":"http://www.w3.org/2001/XMLSchema#string",'
        b'"kind":"literal","language":null,"value":"us-central1"},"predicate":"https://kestrel.ai/vocab/preferredDeployRegion",'
        b'"subject":{"kind":"iri","value":"urn:kestrel:agent:did:example:ava:principal:user"},"tenant_id":"did:example:ava"}'
    )
    expected_id = "urn:kestrel:assertion:sha256:09221b64ffa7584a9c65105d061fb0ca51e352838547d296fb45de7336180e60"

    assert identity_preimage(tenant_id=TENANT, subject=SUBJECT, predicate=PREDICATE, object=object_value) == expected_preimage
    assert derive_assertion_id(tenant_id=TENANT, subject=SUBJECT, predicate=PREDICATE, object=object_value) == expected_id

    reordered = json.loads(expected_preimage.decode())
    reordered = {"tenant_id": reordered["tenant_id"], **reordered}
    assert json.dumps(reordered, sort_keys=True, separators=(",", ":")).encode() == expected_preimage


def test_iri_object_uses_the_complete_v1_object_term_grammar() -> None:
    preimage = identity_preimage(
        tenant_id=TENANT,
        subject=SUBJECT,
        predicate=PREDICATE,
        object=IRI("https://example.test/object"),
    )
    assert b'"object":{"datatype":null,"kind":"iri","language":null,"value":"https://example.test/object"}' in preimage


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (IRI("https://example.test/resource"), IRI("https://example.test/other")),
        (Literal("1", XSD_STRING), Literal("1", XSD_INTEGER)),
        (Literal("value", RDF_LANG_STRING, language="en"), Literal("value", RDF_LANG_STRING, language="fr")),
        (Literal("false", XSD_BOOLEAN), Literal("true", XSD_BOOLEAN)),
    ],
)
def test_identity_changes_for_meaningful_term_distinctions(left: object, right: object) -> None:
    assert derive_assertion_id(tenant_id=TENANT, subject=SUBJECT, predicate=PREDICATE, object=left) != derive_assertion_id(
        tenant_id=TENANT, subject=SUBJECT, predicate=PREDICATE, object=right
    )


def test_identity_is_stable_across_a_fresh_python_process() -> None:
    code = """
from kestrel_sovereign.knowledge import IRI, Literal, XSD_STRING, derive_assertion_id
print(derive_assertion_id(
    tenant_id='did:example:ava',
    subject=IRI('urn:kestrel:agent:did:example:ava:principal:user'),
    predicate=IRI('https://kestrel.ai/vocab/preferredDeployRegion'),
    object=Literal('us-central1', XSD_STRING),
))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == direct_assertion().assertion_id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Example.COM:080/a/./b/../%7Ealice?q=%7e%2f#F%2a", "http://example.com/a/~alice?q=~%2F#F%2A"),
        ("https://EXAMPLE.com:0443/%2e%2E/a", "https://example.com/a"),
        ("did:Example:agent:443", "did:Example:agent:443"),
    ],
)
def test_pinned_iri_profile_normalizes_contract_vectors(raw: str, expected: str) -> None:
    assert normalize_iri(raw) == expected


def test_pinned_iri_profile_normalizes_case_insensitive_ipvfuture_marker() -> None:
    assert normalize_iri("https://[Vf.Example-Host]/a") == "https://[vf.example-host]/a"


def test_host_percent_equivalence_has_same_identity_preimage() -> None:
    first = IRI("HTTP://%45XAMPLE.test/a")
    second = IRI("http://example.test/a")
    assert first == second
    assert identity_preimage(tenant_id=TENANT, subject=first, predicate=PREDICATE, object=Literal("x")) == identity_preimage(
        tenant_id=TENANT, subject=second, predicate=PREDICATE, object=Literal("x")
    )


@pytest.mark.parametrize("raw", ["https://exämple.test/a", "https://example.test:65536/a", "https://example.test/%zz"])
def test_pinned_iri_profile_rejects_unsupported_forms(raw: str) -> None:
    with pytest.raises(AssertionValidationError, match="unsupported_iri_form"):
        IRI(raw)


@pytest.mark.parametrize(
    "term",
    [
        IRI("https://example.test/resource"),
        BlankNode("source-blank-1"),
        LocalIdentifier("source-local-1"),
        Literal("plain", XSD_STRING),
        Literal("hello", RDF_LANG_STRING, language="EN-us"),
        Literal("hello", RDF_LANG_STRING, language="ar", direction="rtl"),
        Literal("001", XSD_INTEGER),
        Literal("+1.2300", XSD_DECIMAL),
    ],
)
def test_resource_and_literal_variants_have_lossless_versioned_mapping_round_trips(term: object) -> None:
    mapping = term.to_mapping()  # type: ignore[union-attr]
    if isinstance(term, (IRI, BlankNode, LocalIdentifier)):
        restored = type(term).from_mapping(mapping)
    else:
        restored = Literal.from_mapping(mapping)
    assert restored == term


@pytest.mark.parametrize(
    ("lexical_form", "datatype_iri", "language"),
    [
        (" value ", XSD_STRING, None),
        ("\tvalue", XSD_STRING, None),
        ("value\u00a0", RDF_LANG_STRING, "en"),
    ],
)
def test_literal_boundary_whitespace_is_rejected_in_identity_and_mapping_paths(
    lexical_form: str, datatype_iri: str, language: str | None
) -> None:
    with pytest.raises(AssertionValidationError, match="leading or trailing whitespace"):
        Literal(lexical_form, datatype_iri, language=language)

    mapping = Literal("value", datatype_iri, language=language).to_mapping()
    mapping["lexical_form"] = lexical_form
    with pytest.raises(AssertionValidationError, match="leading or trailing whitespace"):
        Literal.from_mapping(mapping)


@pytest.mark.parametrize(
    ("status", "state", "supersedes"),
    [
        (AssertionStatus.ACTIVE, EpistemicState.REPORTED, None),
        (AssertionStatus.SUPERSEDED, EpistemicState.REPORTED, "revision-older"),
        (AssertionStatus.RETRACTED, EpistemicState.RETRACTED, None),
        (AssertionStatus.QUARANTINED, EpistemicState.REPORTED, None),
        (AssertionStatus.DELETED, EpistemicState.REPORTED, None),
    ],
)
def test_assertion_mapping_round_trips_every_lifecycle_state(
    status: AssertionStatus, state: EpistemicState, supersedes: str | None
) -> None:
    assertion = direct_assertion(status=status, epistemic_state=state, supersedes_revision_id=supersedes)
    mapping = assertion.to_mapping()
    assert Assertion.from_mapping(mapping) == assertion
    assert mapping == Assertion.from_mapping(mapping).to_mapping()


def test_timezone_normalization_and_interval_validation() -> None:
    assertion = direct_assertion(
        asserted_at="2026-07-26T09:02:11.1200-05:00",
        observed_time=TemporalInterval(start="2026-07-26T09:00:00-05:00", end="2026-07-26T14:01:00Z"),
    )
    assert assertion.asserted_at == Instant("2026-07-26T14:02:11.12Z")
    assert assertion.observed_time == TemporalInterval(start="2026-07-26T14:00:00Z", end="2026-07-26T14:01:00Z")
    assert TemporalInterval(start="2026-07-26T14:00:00Z", end="2026-07-26T14:00:00.1Z")

    with pytest.raises(AssertionValidationError, match="timezone"):
        TemporalInterval(start="2026-07-26T14:00:00")
    with pytest.raises(AssertionValidationError, match="start must not be after"):
        TemporalInterval(start="2026-07-26T14:01:00Z", end="2026-07-26T14:00:00Z")
    with pytest.raises(AssertionValidationError, match="start must not be after"):
        TemporalInterval(start="2026-07-26T14:00:00.1Z", end="2026-07-26T14:00:00Z")


@pytest.mark.parametrize(
    ("lexical_form", "datatype_iri"),
    [
        ("٢٠٢٦-07-26", XSD_DATE),
        ("14:02:١١Z", XSD_TIME),
        ("2026-07-26T14:02:11.١Z", XSD_DATETIME),
    ],
)
def test_literal_temporal_forms_reject_non_ascii_digits(lexical_form: str, datatype_iri: str) -> None:
    with pytest.raises(AssertionValidationError):
        Literal(lexical_form, datatype_iri)


def test_direct_and_derived_provenance_round_trip_including_derivation_reference() -> None:
    source = SourceOccurrence(
        source_occurrence_id="conversation_history:884#message-body",
        source_kind="conversation_message",
        locator="conversation:884#message-body",
        received_at="2026-07-26T14:02:11Z",
        content_digest="sha256:source",
        actor="did:example:ava",
        selector="message-body",
    )
    assert SourceOccurrence.from_mapping(source.to_mapping()) == source
    derived = direct_assertion(
        revision_id="derived-revision",
        epistemic_state=EpistemicState.INFERRED,
        lineage=DerivedLineage(
            rule_id="rule:access-v1",
            engine_version="engine-v1",
            profile_version="profile-v1",
            input_revision_ids=("revision-support-a", "revision-support-b"),
            input_digest="sha256:inputs",
            run_id="run-1",
            generated_at="2026-07-26T14:03:00Z",
            derivation_reference="urn:kestrel:derivation:run-1",
        ),
    )
    assert isinstance(derived.lineage, DerivedLineage)
    assert derived.lineage.derivation_reference == "urn:kestrel:derivation:run-1"
    assert Assertion.from_mapping(derived.to_mapping()) == derived


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "tenant with spaces"},
        {"tenant_id": "cafe\u0301"},
        {"confidence": "1.0001"},
        {"confidence": -1},
        {"confidence": 0.5},
        {"object": "not-a-term"},
        {"status": "not-a-status"},
        {"supersedes_revision_id": "revision-1"},
        {"status": AssertionStatus.RETRACTED, "epistemic_state": EpistemicState.REPORTED},
        {"epistemic_state": EpistemicState.INFERRED},
    ],
)
def test_fail_fast_validation_for_tenant_confidence_type_status_and_supersession(overrides: dict[str, object]) -> None:
    with pytest.raises(AssertionValidationError):
        direct_assertion(**overrides)


def test_decimal_metadata_rejects_unbounded_fixed_point_mapping_before_rendering() -> None:
    within_bound = direct_assertion(confidence="1e-1000").to_mapping()["confidence"]
    assert isinstance(within_bound, str)
    assert len(within_bound) == 1002
    assert within_bound.endswith("1")

    with pytest.raises(AssertionValidationError, match="confidence decimal exponent"):
        direct_assertion(confidence="1e-100000000")
    with pytest.raises(AssertionValidationError, match="score decimal exponent"):
        AssertionResult(assertion=direct_assertion(), score="1e-100000000")
    with pytest.raises(AssertionValidationError, match="fixed-point decimal serialization"):
        AssertionResult(assertion=direct_assertion(), score="9" * 1025)


def test_atomic_supersession_can_append_an_unlinked_old_state_revision() -> None:
    old_state_revision = direct_assertion(status=AssertionStatus.SUPERSEDED)
    replacement = direct_assertion(
        revision_id="replacement-revision",
        supersedes_revision_id=old_state_revision.revision_id,
    )

    assert old_state_revision.supersedes_revision_id is None
    assert Assertion.from_mapping(old_state_revision.to_mapping()) == old_state_revision
    assert replacement.status is AssertionStatus.ACTIVE
    assert replacement.supersedes_revision_id == old_state_revision.revision_id


def test_caller_assertion_id_must_match_the_derived_identity() -> None:
    assertion = direct_assertion()
    assert direct_assertion(assertion_id=assertion.assertion_id).assertion_id == assertion.assertion_id
    with pytest.raises(AssertionValidationError, match="does not match"):
        direct_assertion(assertion_id="urn:kestrel:assertion:sha256:" + "0" * 64)


def test_v1_decoder_accepts_the_pre_versioned_v1_shape_but_rejects_ambiguity() -> None:
    mapping = direct_assertion().to_mapping()
    legacy = deepcopy(mapping)
    legacy.pop("schema_version")
    assert Assertion.from_mapping(legacy) == direct_assertion()

    ambiguous = deepcopy(mapping)
    ambiguous["backend_object"] = "not permitted"
    with pytest.raises(AssertionValidationError, match="ambiguous unknown"):
        Assertion.from_mapping(ambiguous)
    mapping["schema_version"] = 2
    with pytest.raises(AssertionValidationError, match="schema_version"):
        Assertion.from_mapping(mapping)


def test_query_and_result_contracts_are_typed_versioned_and_backend_neutral() -> None:
    query = AssertionQuery(subject=SUBJECT, statuses=(AssertionStatus.ACTIVE,), valid_at="2026-07-26T09:00:00-05:00")
    result = AssertionResult(assertion=direct_assertion(), score="1.25", matched_revision_id="revision-1")
    assert AssertionQuery.from_mapping(query.to_mapping()) == query
    assert AssertionResult.from_mapping(result.to_mapping()) == result
    assert result.score == Decimal("1.25")
    assert IDENTITY_VERSION in direct_assertion().identity_preimage.decode()
