"""Public, backend-neutral contract tests for governed learning corpus reads."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from kestrel_sovereign.knowledge import (
    Assertion,
    CorpusCheckpoint,
    CorpusEligibilityReason,
    CorpusValidationStatus,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    GovernedAssertionCorpusService,
    GovernedCorpusBudgetExceeded,
    GovernedCorpusError,
    GovernedCorpusLimits,
    GovernedCorpusPolicy,
    GovernedCorpusUnavailable,
    IRI,
    Literal,
    OntologyRef,
    SemanticMaintenanceTrainingReadiness,
    SourceOccurrence,
    SourceReferenceKind,
    ValidationState,
    ValidationWriteAction,
    Visibility,
    XSD_STRING,
    source_reference_segment,
)


TENANT = "did:example:corpus"
ONTOLOGY = OntologyRef("https://example.test/ontology", "1.0.0", "sha256:test", "semantic-kb-v1")
CAPABILITIES = {"semantic_maintenance": "v3", "shape_set": "shapes@1"}


def _assertion(
    revision: str,
    *,
    state: EpistemicState = EpistemicState.ASSERTED,
    privacy: str = "normal",
    consent: str = "policy:training-v1",
    grounding: str = "operator-attested",
    visibility: Visibility = Visibility.PRIVATE,
) -> Assertion:
    source_id = f"source:{revision}"
    return Assertion(
        tenant_id=TENANT,
        owning_agent_id=TENANT,
        subject=IRI("urn:kestrel:agent:did:example:corpus:principal:user"),
        predicate=IRI("https://example.test/predicate"),
        object=Literal(revision, XSD_STRING),
        revision_id=revision,
        confidence=Decimal("1"),
        confidence_method="operator",
        confidence_basis=grounding,
        epistemic_state=state,
        asserted_at="2026-07-31T12:00:00Z",
        ontology_version=ONTOLOGY,
        lineage=DirectLineage((source_id,)),
        privacy_classification=privacy,
        release_policy_reference=consent,
        visibility=visibility,
    )


def _source(assertion: Assertion, *, source_kind: str = "operator-note") -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id=assertion.lineage.source_occurrence_ids[0],
        source_kind=source_kind,
        locator=f"note:{assertion.revision_id}",
        received_at="2026-07-31T12:00:00Z",
        content_digest="sha256:source",
    )


def _derived(revision: str, premise: Assertion) -> Assertion:
    return Assertion(
        tenant_id=TENANT,
        owning_agent_id=TENANT,
        subject=IRI("urn:kestrel:agent:did:example:corpus:principal:user"),
        predicate=IRI("https://example.test/derived-predicate"),
        object=Literal(revision, XSD_STRING),
        revision_id=revision,
        confidence=Decimal("1"),
        confidence_method="rule",
        confidence_basis="operator-attested",
        epistemic_state=EpistemicState.INFERRED,
        asserted_at="2026-07-31T12:00:01Z",
        ontology_version=ONTOLOGY,
        lineage=DerivedLineage(
            rule_id="test-rule",
            engine_version="test-engine",
            profile_version="rule-profile-1",
            input_revision_ids=(premise.revision_id,),
            input_digest="sha256:inputs",
            run_id="run:derived",
            generated_at="2026-07-31T12:00:01Z",
        ),
        privacy_classification="normal",
        release_policy_reference="policy:training-v1",
        visibility=Visibility.PRIVATE,
    )


def _policy(**changes) -> GovernedCorpusPolicy:
    values = {
        "policy_id": "parametric-self-corpus",
        "policy_version": "1",
        "accepted_epistemic_states": (EpistemicState.ASSERTED,),
        "accepted_visibility": (Visibility.PRIVATE,),
        "accepted_privacy_classifications": ("normal",),
        "accepted_consent_references": ("policy:training-v1",),
        "accepted_grounding_classes": ("operator-attested",),
        "accepted_source_kinds": ("operator-note",),
        "accepted_ontology_pins": (ONTOLOGY,),
        "accepted_semantic_capability_versions": tuple(CAPABILITIES.items()),
    }
    values.update(changes)
    return GovernedCorpusPolicy(**values)


class _CorpusHost:
    """Out-of-tree-consumer-shaped host capability; no storage internals."""

    def __init__(self, assertions: list[Assertion]) -> None:
        self.assertions = assertions
        self.checkpoint = CorpusCheckpoint(TENANT, 7, "event:7")
        self.ready = SemanticMaintenanceTrainingReadiness(True, None)
        self.capabilities = dict(CAPABILITIES)
        self.changes = []
        self.source_batch_calls = []
        self.sources = {
            item.revision_id: [_source(item)]
            for item in assertions
            if isinstance(item.lineage, DirectLineage)
        }
        self.derivation_inputs = {}
        self.validations = {
            item.assertion_id: CorpusValidationStatus(
                ValidationState.CONFORMS, ValidationWriteAction.ACCEPT,
                "kestrel-assertion-shapes", "1.0.0", "1.0.0",
            )
            for item in assertions
        }

    async def assertion_event_checkpoint(self):
        return self.checkpoint

    async def assertion_changes_after(self, checkpoint, *, limit=100):
        assert checkpoint.tenant_id == TENANT
        return self.changes[:limit]

    async def assertion_inference_inputs(self, query=None):
        values = self.assertions
        if query is not None and query.assertion_ids:
            values = [item for item in values if item.assertion_id in query.assertion_ids]
        return list(values[: query.limit if query is not None else None])

    async def list_assertion_revision_sources(self, revision_id):
        return list(self.sources.get(revision_id, ()))

    async def list_assertion_revision_sources_batch(self, revision_ids):
        self.source_batch_calls.append(tuple(revision_ids))
        return {
            revision_id: list(self.sources.get(revision_id, ()))
            for revision_id in revision_ids
        }

    async def get_derivation_inputs(self, revision_id):
        return list(self.derivation_inputs.get(revision_id, ()))

    async def assertion_validation_statuses(self, assertions):
        return {
            item.assertion_id: self.validations[item.assertion_id]
            for item in assertions
            if item.assertion_id in self.validations
        }

    async def semantic_maintenance_training_readiness(self, *args, **kwargs):
        expected_checkpoint = kwargs.get("expected_checkpoint")
        if expected_checkpoint is not None and expected_checkpoint != self.checkpoint:
            return SemanticMaintenanceTrainingReadiness(
                False, "semantic_maintenance_checkpoint_changed"
            )
        return self.ready

    async def semantic_maintenance_capability_versions(self, *args, **kwargs):
        return dict(self.capabilities)


@pytest.mark.asyncio
async def test_snapshot_is_deterministic_and_carries_auditable_lineage() -> None:
    first, second = _assertion("revision-a"), _assertion("revision-b")
    left_host = _CorpusHost([second, first])
    left = await GovernedAssertionCorpusService(left_host).snapshot(
        policy=_policy(), inference_profile=None
    )
    right = await GovernedAssertionCorpusService(_CorpusHost([first, second])).snapshot(
        policy=_policy(), inference_profile=None
    )

    assert left.snapshot_hash == right.snapshot_hash
    assert [item.assertion.revision_id for item in left.examples] == ["revision-a", "revision-b"]
    assert left.examples[0].source_occurrences[0].source_occurrence_id == "source:revision-a"
    assert left.examples[0].decision.reason is CorpusEligibilityReason.INCLUDED
    assert left.examples[0].split_key == right.examples[0].split_key
    assert left.observability.included == 2
    assert left_host.source_batch_calls == [("revision-b", "revision-a")]
    with pytest.raises(TypeError):
        left.capability_versions["new"] = "pin"  # type: ignore[index]


@pytest.mark.asyncio
async def test_snapshot_applies_policy_matrix_and_never_includes_unvalidated_data() -> None:
    good = _assertion("good")
    wrong_privacy = _assertion("privacy", privacy="restricted")
    ungrounded = _assertion("grounding", grounding="model-guess")
    bad_source = _assertion("source")
    host = _CorpusHost([good, wrong_privacy, ungrounded, bad_source])
    host.sources[bad_source.revision_id] = [
        _source(bad_source, source_kind="web-unverified")
    ]
    host.validations[ungrounded.assertion_id] = CorpusValidationStatus(
        ValidationState.NONCONFORMANT, ValidationWriteAction.ACCEPT_WITH_REPORT
    )

    snapshot = await GovernedAssertionCorpusService(host).snapshot(policy=_policy(), inference_profile=None)

    assert [item.assertion.revision_id for item in snapshot.examples] == ["good"]
    assert snapshot.observability.excluded == {
        "privacy_classification_disallowed": 1,
        "source_class_disallowed": 1,
        "validation_not_conformant": 1,
    }


@pytest.mark.asyncio
async def test_initial_snapshot_rejects_capability_pins_and_excludes_ontology_mismatch() -> None:
    host = _CorpusHost([_assertion("capability")])
    host.capabilities["shape_set"] = "shapes@2"
    with pytest.raises(GovernedCorpusUnavailable, match="capability_versions_mismatch"):
        await GovernedAssertionCorpusService(host).snapshot(
            policy=_policy(), inference_profile=None
        )

    other_ontology = OntologyRef(
        "https://example.test/ontology", "2.0.0", "sha256:other", "semantic-kb-v1"
    )
    assertion = replace(_assertion("ontology"), ontology_version=other_ontology)
    ontology_host = _CorpusHost([assertion])
    snapshot = await GovernedAssertionCorpusService(ontology_host).snapshot(
        policy=_policy(), inference_profile=None
    )
    assert snapshot.examples == ()
    assert snapshot.observability.excluded == {"ontology_pin_disallowed": 1}


@pytest.mark.asyncio
async def test_derived_assertion_fails_closed_on_restricted_transitive_premise() -> None:
    premise = _assertion("restricted-premise", privacy="restricted")
    derived = _derived("derived", premise)
    host = _CorpusHost([premise, derived])
    host.derivation_inputs[derived.revision_id] = [premise]
    policy = _policy(
        accepted_epistemic_states=(EpistemicState.ASSERTED, EpistemicState.INFERRED),
        allow_inferred=True,
        accepted_derivation_profiles=("rule-profile-1",),
    )

    snapshot = await GovernedAssertionCorpusService(host).snapshot(
        policy=policy, inference_profile=None
    )

    assert snapshot.examples == ()
    assert snapshot.observability.excluded == {
        "derivation_input_ineligible": 1,
        "privacy_classification_disallowed": 1,
    }


@pytest.mark.asyncio
async def test_snapshot_fails_closed_for_stale_maintenance_budgets_and_unpersisted_prior() -> None:
    host = _CorpusHost([_assertion("only")])
    service = GovernedAssertionCorpusService(host)
    snapshot = await service.snapshot(policy=_policy(), inference_profile=None)

    host.ready = SemanticMaintenanceTrainingReadiness(False, "semantic_maintenance_partial")
    with pytest.raises(GovernedCorpusUnavailable, match="partial"):
        await service.snapshot(policy=_policy(), inference_profile=None)
    with pytest.raises(GovernedCorpusUnavailable, match="requires_host_persistence"):
        await service.snapshot(
            policy=_policy(policy_version="2"), inference_profile=None,
            prior_verified_snapshot=snapshot, allow_prior_verified_snapshot=True,
        )
    with pytest.raises(GovernedCorpusUnavailable, match="requires_host_persistence"):
        await service.snapshot(
            policy=_policy(), inference_profile=None,
            prior_verified_snapshot=snapshot, allow_prior_verified_snapshot=True,
        )

    ready_host = _CorpusHost([_assertion("one"), _assertion("two"), _assertion("three")])
    with pytest.raises(GovernedCorpusBudgetExceeded, match="max_assertions"):
        await GovernedAssertionCorpusService(ready_host).snapshot(
            policy=_policy(), inference_profile=None, limits=GovernedCorpusLimits(max_assertions=2)
        )


@pytest.mark.asyncio
async def test_incremental_read_returns_additions_and_first_class_tombstones() -> None:
    initial = _assertion("initial")
    host = _CorpusHost([initial])
    service = GovernedAssertionCorpusService(host)
    snapshot = await service.snapshot(policy=_policy(), inference_profile=None)
    added = _assertion("added")
    host.assertions.append(added)
    host.sources[added.revision_id] = [_source(added)]
    host.validations[added.assertion_id] = CorpusValidationStatus(
        ValidationState.CONFORMS, ValidationWriteAction.ACCEPT
    )
    host.checkpoint = CorpusCheckpoint(TENANT, 9, "event:9")
    host.changes = [
        SimpleNamespace(
            event_id="event:8", assertion_id=added.assertion_id, revision_id=added.revision_id,
            operation="accepted", generation=8, eligible=True,
        ),
        SimpleNamespace(
            event_id="event:9", assertion_id=initial.assertion_id, revision_id=initial.revision_id,
            operation="deleted", generation=9, eligible=False,
        ),
    ]

    delta = await service.changes_since(snapshot, policy=_policy(), inference_profile=None)

    assert [item.assertion.revision_id for item in delta.additions] == ["added"]
    assert [(item.operation, item.reason) for item in delta.tombstones] == [("deleted", "deleted")]
    assert delta.checkpoint_generation == 9
    assert delta.checkpoint.latest_event_id == "event:9"


@pytest.mark.asyncio
async def test_incremental_read_never_advertises_an_undelivered_head_event() -> None:
    initial = _assertion("watermark-initial")
    host = _CorpusHost([initial])
    service = GovernedAssertionCorpusService(host)
    snapshot = await service.snapshot(policy=_policy(), inference_profile=None)
    host.checkpoint = CorpusCheckpoint(TENANT, 9, "event:9")
    host.changes = [
        SimpleNamespace(
            event_id="event:8", assertion_id=None, revision_id=None,
            operation="erasure_resync", generation=8, eligible=False,
        )
    ]

    with pytest.raises(GovernedCorpusBudgetExceeded, match="does not reach"):
        await service.changes_since(snapshot, policy=_policy(), inference_profile=None)


@pytest.mark.asyncio
async def test_no_change_delta_is_wall_time_bounded_around_host_awaits() -> None:
    class _SlowHost(_CorpusHost):
        async def semantic_maintenance_training_readiness(self, *args, **kwargs):
            await asyncio.sleep(0.05)
            return self.ready

    host = _CorpusHost([_assertion("timeout")])
    snapshot = await GovernedAssertionCorpusService(host).snapshot(
        policy=_policy(), inference_profile=None
    )
    with pytest.raises(GovernedCorpusBudgetExceeded, match="wall_time"):
        await GovernedAssertionCorpusService(_SlowHost([])).changes_since(
            snapshot,
            policy=_policy(),
            inference_profile=None,
            limits=GovernedCorpusLimits(max_wall_time_seconds=0.001),
        )


@pytest.mark.asyncio
async def test_snapshot_rejects_a_lifecycle_change_during_assembly() -> None:
    class _RacingHost(_CorpusHost):
        def __init__(self, assertions):
            super().__init__(assertions)
            self._checkpoint_reads = 0

        async def assertion_event_checkpoint(self):
            self._checkpoint_reads += 1
            if self._checkpoint_reads > 1:
                return CorpusCheckpoint(TENANT, 8, "event:8")
            return self.checkpoint

    with pytest.raises(GovernedCorpusUnavailable, match="changed_during_snapshot"):
        await GovernedAssertionCorpusService(_RacingHost([_assertion("racing")])).snapshot(
            policy=_policy(), inference_profile=None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("incremental", [False, True])
async def test_readiness_is_bound_to_checkpoint_captured_before_the_gate(
    incremental: bool,
) -> None:
    assertion = _assertion("readiness-race")
    stable_host = _CorpusHost([assertion])
    snapshot = await GovernedAssertionCorpusService(stable_host).snapshot(
        policy=_policy(), inference_profile=None
    )

    class _ReadinessRaceHost(_CorpusHost):
        async def semantic_maintenance_training_readiness(self, *args, **kwargs):
            # Reproduce a canonical write after the corpus captures its head
            # but before readiness reads the maintenance state.
            self.checkpoint = CorpusCheckpoint(TENANT, 8, "event:8")
            return await super().semantic_maintenance_training_readiness(
                *args, **kwargs
            )

    service = GovernedAssertionCorpusService(_ReadinessRaceHost([assertion]))
    with pytest.raises(GovernedCorpusUnavailable, match="checkpoint_changed"):
        if incremental:
            await service.changes_since(
                snapshot, policy=_policy(), inference_profile=None
            )
        else:
            await service.snapshot(policy=_policy(), inference_profile=None)


@pytest.mark.asyncio
async def test_event_checkpoint_backend_failure_is_a_governed_error() -> None:
    class _BrokenCheckpointHost(_CorpusHost):
        async def assertion_event_checkpoint(self):
            raise RuntimeError("missing event row")

    with pytest.raises(GovernedCorpusUnavailable, match="event_checkpoint_unavailable"):
        await GovernedAssertionCorpusService(
            _BrokenCheckpointHost([_assertion("broken-checkpoint")])
        ).snapshot(policy=_policy(), inference_profile=None)


EXAM_SESSION = "dcd6f767-829d-4a7d-b8c7-d7a722e5de13"
EXAM_ITEM = "469a149e"


def _exam_source(assertion: Assertion, *segments: str) -> SourceOccurrence:
    """A source read during a turn, naming the saved items/sessions it read."""
    return SourceOccurrence(
        source_occurrence_id=assertion.lineage.source_occurrence_ids[0],
        source_kind="operator-note",
        locator="#".join(
            ("POST:/api/agent/invoke", *segments, f"note:{assertion.revision_id}")
        ),
        received_at="2026-10-25T12:00:00Z",
        content_digest="sha256:exam",
    )


def _corpus_text(examples) -> str:
    return json.dumps([example.canonical_mapping() for example in examples])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("segments", "exclusion"),
    [
        (
            (source_reference_segment(SourceReferenceKind.SAVED_ITEM, EXAM_ITEM),),
            {"excluded_saved_item_ids": (EXAM_ITEM, "ed452a4c")},
        ),
        (
            (source_reference_segment(SourceReferenceKind.SESSION, EXAM_SESSION),),
            {"excluded_session_ids": (EXAM_SESSION,)},
        ),
        ((), {"excluded_source_occurrence_ids": ("source:exam-answer",)}),
    ],
    ids=["saved-item", "session", "source-occurrence"],
)
async def test_excluded_material_is_held_out_even_when_every_class_rule_accepts_it(
    segments: tuple[str, ...], exclusion: dict[str, tuple[str, ...]]
) -> None:
    exam = _assertion("exam-answer")
    everyday = _assertion("everyday-note")
    host = _CorpusHost([exam, everyday])
    host.sources[exam.revision_id] = [_exam_source(exam, *segments)]
    service = GovernedAssertionCorpusService(host)

    unexcluded = await service.snapshot(policy=_policy(), inference_profile=None)
    snapshot = await service.snapshot(policy=_policy(**exclusion), inference_profile=None)

    # Without the exclusion every class rule admits the exam text.
    assert {item.assertion.revision_id for item in unexcluded.examples} == {
        "exam-answer", "everyday-note",
    }
    assert [item.assertion.revision_id for item in snapshot.examples] == ["everyday-note"]
    assert snapshot.observability.excluded == {"explicitly_excluded": 1}
    assert "exam-answer" in _corpus_text(unexcluded.examples)
    assert "exam-answer" not in _corpus_text(snapshot.examples)


@pytest.mark.asyncio
async def test_exclusion_precedes_class_rules_and_holds_out_derived_assertions() -> None:
    exam = _assertion("exam-premise")
    restricted_exam = _assertion("restricted-exam", privacy="restricted")
    inference = _derived("exam-inference", exam)
    host = _CorpusHost([exam, restricted_exam, inference])
    session = source_reference_segment(SourceReferenceKind.SESSION, EXAM_SESSION)
    for item in (exam, restricted_exam):
        host.sources[item.revision_id] = [_exam_source(item, session)]
    host.derivation_inputs[inference.revision_id] = [exam]
    inferred = {
        "accepted_epistemic_states": (EpistemicState.ASSERTED, EpistemicState.INFERRED),
        "allow_inferred": True,
        "accepted_derivation_profiles": ("rule-profile-1",),
    }
    service = GovernedAssertionCorpusService(host)

    unexcluded = await service.snapshot(policy=_policy(**inferred), inference_profile=None)
    snapshot = await service.snapshot(
        policy=_policy(**inferred, excluded_session_ids=(EXAM_SESSION,)),
        inference_profile=None,
    )

    assert {item.assertion.revision_id for item in unexcluded.examples} == {
        "exam-premise", "exam-inference",
    }
    assert unexcluded.observability.excluded == {"privacy_classification_disallowed": 1}
    # The restricted exam reports its exclusion, not its privacy class, and
    # the inference drawn from the exam is held out with its premise.
    assert snapshot.examples == ()
    assert snapshot.observability.excluded == {"explicitly_excluded": 3}
    # Provenance is read in one batch per snapshot, ineligible revisions included.
    assert len(host.source_batch_calls) == 2


@pytest.mark.asyncio
async def test_incremental_additions_hold_out_excluded_material() -> None:
    policy = _policy(excluded_session_ids=(EXAM_SESSION,))
    host = _CorpusHost([_assertion("before-exam")])
    service = GovernedAssertionCorpusService(host)
    snapshot = await service.snapshot(policy=policy, inference_profile=None)
    exam = _assertion("exam-answer")
    host.assertions.append(exam)
    host.sources[exam.revision_id] = [
        _exam_source(exam, source_reference_segment(SourceReferenceKind.SESSION, EXAM_SESSION))
    ]
    host.validations[exam.assertion_id] = CorpusValidationStatus(
        ValidationState.CONFORMS, ValidationWriteAction.ACCEPT
    )
    host.checkpoint = CorpusCheckpoint(TENANT, 8, "event:8")
    host.changes = [
        SimpleNamespace(
            event_id="event:8", assertion_id=exam.assertion_id, revision_id=exam.revision_id,
            operation="accepted", generation=8, eligible=True,
        )
    ]

    delta = await service.changes_since(snapshot, policy=policy, inference_profile=None)

    assert delta.additions == ()
    assert delta.observability.excluded == {"explicitly_excluded": 1}


@pytest.mark.asyncio
async def test_exclusions_are_part_of_the_policy_value_and_corpus_provenance() -> None:
    without = _policy()
    held_out = _policy(
        excluded_saved_item_ids=("ed452a4c", EXAM_ITEM),
        excluded_session_ids=(EXAM_SESSION,),
    )

    # A policy with no exclusions keeps the digest it had before they existed,
    # so adapter manifests recorded under it still verify after an upgrade.
    assert without.digest == "cca6d3e04faea101dc46369353451c50739a4087e4a2e7bd3ab6318b56a04d23"
    assert not {"excluded_saved_item_ids", "excluded_session_ids"} & without.to_mapping().keys()
    assert held_out.to_mapping()["excluded_saved_item_ids"] == [EXAM_ITEM, "ed452a4c"]
    assert held_out.to_mapping()["excluded_session_ids"] == [EXAM_SESSION]
    assert held_out.digest != without.digest
    assert held_out.digest == _policy(
        excluded_saved_item_ids=(EXAM_ITEM, "ed452a4c"),
        excluded_session_ids=(EXAM_SESSION,),
    ).digest

    service = GovernedAssertionCorpusService(_CorpusHost([_assertion("note")]))
    plain = await service.snapshot(policy=without, inference_profile=None)
    governed = await service.snapshot(policy=held_out, inference_profile=None)
    assert governed.snapshot_hash != plain.snapshot_hash
    assert governed.observability.policy_digest == held_out.digest
    assert governed.examples[0].decision.policy_digest == held_out.digest
    with pytest.raises(GovernedCorpusError, match="same corpus policy"):
        await service.changes_since(plain, policy=held_out, inference_profile=None)
    for malformed in (EXAM_ITEM, (EXAM_ITEM, EXAM_ITEM), ("",), None):
        with pytest.raises(GovernedCorpusError):
            _policy(excluded_saved_item_ids=malformed)


def test_policy_mapping_round_trips_exclusions_and_refuses_unknown_fields() -> None:
    policy = _policy(
        accepted_epistemic_states=(EpistemicState.INFERRED, EpistemicState.ASSERTED),
        allow_inferred=True,
        accepted_derivation_profiles=("rule-profile-1",),
        excluded_session_ids=(EXAM_SESSION,),
        excluded_source_occurrence_ids=("source:exam-answer",),
    )
    mapping = policy.to_mapping()

    restored = GovernedCorpusPolicy.from_mapping(json.loads(json.dumps(mapping)))

    assert restored == policy
    assert restored.digest == policy.digest
    assert GovernedCorpusPolicy.from_mapping(_policy().to_mapping()) == _policy()
    with pytest.raises(GovernedCorpusError, match="unknown"):
        GovernedCorpusPolicy.from_mapping({**mapping, "excluded_conversation_ids": ["x"]})
    with pytest.raises(GovernedCorpusError, match="missing"):
        GovernedCorpusPolicy.from_mapping(
            {key: value for key, value in mapping.items() if key != "accepted_source_kinds"}
        )
    with pytest.raises(GovernedCorpusError, match="schema_version"):
        GovernedCorpusPolicy.from_mapping({**mapping, "schema_version": 2})
    with pytest.raises(GovernedCorpusError, match="array"):
        GovernedCorpusPolicy.from_mapping({**mapping, "excluded_session_ids": EXAM_SESSION})
    with pytest.raises(GovernedCorpusError, match="name/version"):
        GovernedCorpusPolicy.from_mapping(
            {**mapping, "accepted_semantic_capability_versions": [["shape_set", "shapes@1"]]}
        )


@pytest.mark.asyncio
async def test_saved_item_exclusion_needs_the_full_id_not_a_display_prefix() -> None:
    # Ids are compared exactly. A display prefix would also hold out any other
    # item sharing it, so the operator resolves it to the stored id instead.
    full_id = "469a149e-8f0c-4d7e-9a51-1c2b3d4e5f60"
    exam = _assertion("exam-v01")
    host = _CorpusHost([exam])
    host.sources[exam.revision_id] = [
        _exam_source(exam, source_reference_segment(SourceReferenceKind.SAVED_ITEM, full_id))
    ]
    service = GovernedAssertionCorpusService(host)

    prefix = await service.snapshot(
        policy=_policy(excluded_saved_item_ids=(EXAM_ITEM,)), inference_profile=None
    )
    exact = await service.snapshot(
        policy=_policy(excluded_saved_item_ids=(full_id,)), inference_profile=None
    )

    assert [item.assertion.revision_id for item in prefix.examples] == ["exam-v01"]
    assert exact.examples == ()


@pytest.mark.asyncio
async def test_provenance_that_does_not_cover_the_lineage_is_not_cleared() -> None:
    # The host returns one of two lineage sources. The missing one could name
    # an excluded session, so the revision cannot be shown to be clear of it.
    partial = replace(
        _assertion("partial"),
        lineage=DirectLineage(("source:partial", "source:partial-unreturned")),
    )
    host = _CorpusHost([partial])
    service = GovernedAssertionCorpusService(host)

    snapshot = await service.snapshot(
        policy=_policy(excluded_session_ids=(EXAM_SESSION,)), inference_profile=None
    )
    # An excluded source named in the lineage is held out as excluded even
    # though the host did not return that source.
    named = await service.snapshot(
        policy=_policy(excluded_source_occurrence_ids=("source:partial-unreturned",)),
        inference_profile=None,
    )

    assert snapshot.examples == ()
    assert snapshot.observability.excluded == {"source_class_disallowed": 1}
    assert named.observability.excluded == {"explicitly_excluded": 1}


def test_public_contract_is_consumable_without_storage_or_graph_imports() -> None:
    # A feature package needs only this public import path and the host method;
    # it never imports AsyncAssertionStore, graph_nodes, sqlite3, or a DB path.
    from kestrel_sovereign.knowledge import GovernedAssertionCorpusService as imported

    assert imported is GovernedAssertionCorpusService
