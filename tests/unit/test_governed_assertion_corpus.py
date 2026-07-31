"""Public, backend-neutral contract tests for governed learning corpus reads."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from kestrel_sovereign.knowledge import (
    Assertion,
    CorpusCheckpoint,
    CorpusEligibilityReason,
    CorpusValidationStatus,
    DirectLineage,
    EpistemicState,
    GovernedAssertionCorpusService,
    GovernedCorpusBudgetExceeded,
    GovernedCorpusLimits,
    GovernedCorpusPolicy,
    GovernedCorpusUnavailable,
    IRI,
    Literal,
    OntologyRef,
    SemanticMaintenanceTrainingReadiness,
    SourceOccurrence,
    ValidationState,
    ValidationWriteAction,
    Visibility,
    XSD_STRING,
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
        self.sources = {item.assertion_id: [_source(item)] for item in assertions}
        self.validations = {
            item.assertion_id: CorpusValidationStatus(
                ValidationState.CONFORMS, ValidationWriteAction.ACCEPT,
                "kestrel-assertion-shapes", "1.0.0", "1.0.0",
            )
            for item in assertions
        }

    async def assertion_checkpoint(self):
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
        return list(self.sources[_assertion(revision_id).assertion_id])

    async def assertion_validation_statuses(self, assertion_ids):
        return {item: self.validations[item] for item in assertion_ids if item in self.validations}

    async def semantic_maintenance_training_readiness(self, *args, **kwargs):
        return self.ready

    async def semantic_maintenance_capability_versions(self, *args, **kwargs):
        return dict(self.capabilities)


@pytest.mark.asyncio
async def test_snapshot_is_deterministic_and_carries_auditable_lineage() -> None:
    first, second = _assertion("revision-a"), _assertion("revision-b")
    left = await GovernedAssertionCorpusService(_CorpusHost([second, first])).snapshot(
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
    with pytest.raises(TypeError):
        left.capability_versions["new"] = "pin"  # type: ignore[index]


@pytest.mark.asyncio
async def test_snapshot_applies_policy_matrix_and_never_includes_unvalidated_data() -> None:
    good = _assertion("good")
    wrong_privacy = _assertion("privacy", privacy="restricted")
    ungrounded = _assertion("grounding", grounding="model-guess")
    bad_source = _assertion("source")
    host = _CorpusHost([good, wrong_privacy, ungrounded, bad_source])
    host.sources[bad_source.assertion_id] = [_source(bad_source, source_kind="web-unverified")]
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
    host.sources[added.assertion_id] = [_source(added)]
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
    assert [(item.operation, item.reason) for item in delta.tombstones] == [("deleted", "ineligible")]
    assert delta.checkpoint_generation == 9


@pytest.mark.asyncio
async def test_snapshot_rejects_a_lifecycle_change_during_assembly() -> None:
    class _RacingHost(_CorpusHost):
        def __init__(self, assertions):
            super().__init__(assertions)
            self._checkpoint_reads = 0

        async def assertion_checkpoint(self):
            self._checkpoint_reads += 1
            if self._checkpoint_reads > 1:
                return CorpusCheckpoint(TENANT, 8, "event:8")
            return self.checkpoint

    with pytest.raises(GovernedCorpusUnavailable, match="changed_during_snapshot"):
        await GovernedAssertionCorpusService(_RacingHost([_assertion("racing")])).snapshot(
            policy=_policy(), inference_profile=None
        )


def test_public_contract_is_consumable_without_storage_or_graph_imports() -> None:
    # A feature package needs only this public import path and the host method;
    # it never imports AsyncAssertionStore, graph_nodes, sqlite3, or a DB path.
    from kestrel_sovereign.knowledge import GovernedAssertionCorpusService as imported

    assert imported is GovernedAssertionCorpusService
