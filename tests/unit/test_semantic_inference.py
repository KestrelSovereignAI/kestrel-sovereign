"""Bounded semantic materialization contracts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.agent.sleep import (
    SleepHookContract,
    SleepHookPhase,
    SleepMixin,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.knowledge import (
    Assertion,
    AssertionQuery,
    BoundedInferenceService,
    ClosureStatus,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    IRI,
    InferenceError,
    InferenceLimits,
    InferenceProfile,
    Literal,
    OntologyRef,
    SemanticMaintenanceLimits,
    SemanticMaintenanceError,
    SemanticMaintenanceService,
    SemanticMaintenanceStatus,
    SourceOccurrence,
    ValidationState,
    ValidationWriteAction,
    XSD_STRING,
    inference_limits_from_config,
    inference_profile_from_config,
    maintenance_allows_prior_verified_snapshot,
    maintenance_limits_from_config,
)
from kestrel_sovereign.knowledge.inference import ENGINE_VERSION, validate_inference_profile
from kestrel_sovereign.storage.async_assertion_store import (
    AssertionConflictError,
    MaintenanceLeaseLostError,
)
from kestrel_sovereign.storage.semantic_validation import GovernedSemanticValidationService
from kestrel_sovereign.storage.db import TransactionError
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage


RDF_TYPE = IRI("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDFS_SUBCLASS = IRI("http://www.w3.org/2000/01/rdf-schema#subClassOf")
RDFS_SUBPROPERTY = IRI("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")
ONTOLOGY = OntologyRef(
    "http://www.w3.org/2000/01/rdf-schema#",
    "1.0.0",
    "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
    "semantic-kb-v1",
)
_GOVERNED_STORAGES: dict[int, AsyncStorage] = {}


@pytest.fixture
async def assertion_store(tmp_path):
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Semantic inference test"
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(credentials.agent_did, identity)
    storage = AsyncStorage(":memory:", agent_id=credentials.agent_did, _assertion_tenant_capability=capability)
    await storage.initialize()
    try:
        store = storage._assertion_store()
        # Materialization operates on the bound canonical store, while source
        # fixtures must cross the same governed ingestion boundary as agents.
        _GOVERNED_STORAGES[id(store)] = storage
        yield store
    finally:
        _GOVERNED_STORAGES.pop(id(storage._assertion_store()), None)
        await storage.close()


def _source(source_id: str) -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id=source_id,
        source_kind="fixture",
        locator=f"fixture:{source_id}",
        received_at="2026-07-26T12:00:00Z",
    )


def _assertion(store, revision_id: str, subject: IRI, predicate: IRI, object_: IRI | Literal) -> Assertion:
    source_id = f"source:{revision_id}"
    return Assertion(
        tenant_id=store.tenant_id,
        owning_agent_id=store.owning_agent_id,
        subject=subject,
        predicate=predicate,
        object=object_,
        revision_id=revision_id,
        confidence=Decimal("1"),
        confidence_method="fixture",
        confidence_basis="fixture",
        epistemic_state=EpistemicState.ASSERTED,
        asserted_at="2026-07-26T12:00:00Z",
        ontology_version=ONTOLOGY,
        lineage=DirectLineage((source_id,)),
        privacy_classification="normal",
        release_policy_reference="policy:test",
    )


async def _put(store, revision_id: str, subject: IRI, predicate: IRI, object_: IRI | Literal) -> Assertion:
    assertion = _assertion(store, revision_id, subject, predicate, object_)
    result = await _GOVERNED_STORAGES[id(store)].put_assertion(
        assertion,
        source_occurrences=(_source(f"source:{revision_id}"),),
    )
    assert result.accepted
    return assertion


def _profile(*, owl: bool = False) -> InferenceProfile:
    return InferenceProfile(ONTOLOGY, "1.0.0", "1.0.0" if owl else None)


async def _materialize_ungrounded_alternate_proof_cycle(
    assertion_store,
) -> tuple[Assertion, BoundedInferenceService, Assertion, Assertion]:
    """Build the RDFS cycle used to test grounded lifecycle retraction."""
    subject = IRI("https://example.test/subject")
    object_ = IRI("https://example.test/object")
    property_p = IRI("https://example.test/p")
    property_q = IRI("https://example.test/q")
    property_r = IRI("https://example.test/r")
    predecessor = await _put(
        assertion_store, "p-sub-q", property_p, RDFS_SUBPROPERTY, property_q
    )
    await _put(assertion_store, "q-sub-r", property_q, RDFS_SUBPROPERTY, property_r)
    await _put(assertion_store, "r-sub-q", property_r, RDFS_SUBPROPERTY, property_q)
    await _put(assertion_store, "statement", subject, property_p, object_)

    service = BoundedInferenceService(assertion_store, _profile())
    assert (await service.materialize_incremental()).complete
    conclusion_q = (
        await assertion_store.query(
            AssertionQuery(subject=subject, predicate=property_q, object=object_)
        )
    )[0]
    conclusion_r = (
        await assertion_store.query(
            AssertionQuery(subject=subject, predicate=property_r, object=object_)
        )
    )[0]
    assert len(await service.explain(conclusion_q.assertion_id)) >= 2
    assert len(await service.explain(conclusion_r.assertion_id)) >= 2
    return predecessor, service, conclusion_q, conclusion_r


def test_explicit_semantic_inference_config_constructs_versioned_profile() -> None:
    profile = inference_profile_from_config(
        {
            "enabled": True,
            "rdfs_version": "1.0.0",
            "owl2rl_version": "1.0.0",
            "ontology": {
                "namespace": ONTOLOGY.namespace,
                "version": ONTOLOGY.version,
                "content_digest": ONTOLOGY.content_digest,
                "compatibility_profile": ONTOLOGY.compatibility_profile,
            },
        }
    )
    assert profile == _profile(owl=True)


def test_semantic_inference_limits_are_strictly_operator_configured() -> None:
    config = {
        "enabled": True,
        "rdfs_version": "1.0.0",
        "ontology": {
            "namespace": ONTOLOGY.namespace,
            "version": ONTOLOGY.version,
            "content_digest": ONTOLOGY.content_digest,
            "compatibility_profile": ONTOLOGY.compatibility_profile,
        },
        "limits": {
            "max_source_assertions": 23,
            "max_iterations": 7,
            "max_generated_assertions": 31,
            "max_wall_time_seconds": 1.25,
            "max_memory_items": 47,
        },
    }

    assert inference_limits_from_config(config) == InferenceLimits(
        max_source_assertions=23,
        max_iterations=7,
        max_generated_assertions=31,
        max_wall_time_seconds=1.25,
        max_memory_items=47,
    )
    with pytest.raises(InferenceError, match="unsupported fields"):
        inference_limits_from_config({**config, "limits": {"unknown": 1}})
    with pytest.raises(InferenceError, match="must be an integer"):
        inference_limits_from_config(
            {**config, "limits": {"max_source_assertions": True}}
        )


def test_semantic_maintenance_limits_are_strictly_operator_configured() -> None:
    assert maintenance_limits_from_config(
        {
            "max_wall_time_seconds": 2.5,
            "max_assertions": 7,
            "max_derivations": 11,
            "max_shapes": 1,
            "max_reports": 5,
        }
    ) == SemanticMaintenanceLimits(
        max_wall_time_seconds=2.5,
        max_assertions=7,
        max_derivations=11,
        max_shapes=1,
        max_reports=5,
    )
    with pytest.raises(ValueError, match="unsupported fields"):
        maintenance_limits_from_config({"unbounded": 1})


def test_semantic_maintenance_prior_snapshot_exception_is_disabled() -> None:
    with pytest.raises(SemanticMaintenanceError, match="durable governed corpus snapshot"):
        maintenance_allows_prior_verified_snapshot(
            {"allow_prior_verified_snapshot": True}
        )
    assert not maintenance_allows_prior_verified_snapshot({})
    with pytest.raises(ValueError, match="must be a boolean"):
        maintenance_limits_from_config({"allow_prior_verified_snapshot": 1})


@pytest.mark.parametrize("wall_time_literal", ("nan", "+inf"))
def test_toml_semantic_inference_limits_reject_non_finite_wall_time(
    tmp_path, wall_time_literal: str
) -> None:
    """TOML's accepted non-finite floats must not disable the time budget."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "kestrel.toml").write_text(
        "\n".join(
            (
                "[semantic_inference]",
                "enabled = false",
                "",
                "[semantic_inference.limits]",
                f"max_wall_time_seconds = {wall_time_literal}",
            )
        )
    )

    with pytest.raises(RuntimeError, match=r"Invalid \[semantic_inference\] configuration"):
        KestrelAgent(
            did="did:test:non-finite-inference-limit",
            storage_path=str(agent_dir / "kestrel_prime.db"),
        )


def test_semantic_inference_profile_requires_an_exact_registry_ontology_pin() -> None:
    invalid = InferenceProfile(
        OntologyRef(
            ONTOLOGY.namespace,
            ONTOLOGY.version,
            "sha256:not-the-registry-digest",
            ONTOLOGY.compatibility_profile,
        ),
        "1.0.0",
    )

    with pytest.raises(InferenceError, match="ontology digest"):
        validate_inference_profile(invalid)


def test_semantic_profile_load_is_independent_of_malformed_privacy_config(tmp_path) -> None:
    """An unrelated privacy typo cannot turn off an explicitly approved profile."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "kestrel.toml").write_text(
        '\n'.join(
            (
                'privacy = "not-a-table"',
                '',
                '[semantic_inference]',
                'enabled = true',
                'rdfs_version = "1.0.0"',
                '',
                '[semantic_inference.ontology]',
                f'namespace = "{ONTOLOGY.namespace}"',
                f'version = "{ONTOLOGY.version}"',
                f'content_digest = "{ONTOLOGY.content_digest}"',
                f'compatibility_profile = "{ONTOLOGY.compatibility_profile}"',
                '',
                '[semantic_inference.limits]',
                'max_generated_assertions = 19',
            )
        )
    )

    agent = KestrelAgent(
        did="did:test:semantic-config",
        storage_path=str(agent_dir / "kestrel_prime.db"),
    )

    assert agent.semantic_inference_profile == _profile()
    assert agent.semantic_inference_limits.max_generated_assertions == 19
    assert agent._privacy_computer_access is False


def test_explicitly_disabled_semantic_profile_remains_an_operator_state(tmp_path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "kestrel.toml").write_text(
        "[semantic_inference]\nenabled = false\n"
    )

    agent = KestrelAgent(
        did="did:test:semantic-config-disabled",
        storage_path=str(agent_dir / "kestrel_prime.db"),
    )

    assert agent.semantic_inference_profile is None
    assert agent.semantic_inference_configured is True


def test_validation_only_semantic_maintenance_is_an_operator_state(tmp_path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "kestrel.toml").write_text(
        "[semantic_maintenance]\nmax_assertions = 7\n"
    )

    agent = KestrelAgent(
        did="did:test:semantic-maintenance-config",
        storage_path=str(agent_dir / "kestrel_prime.db"),
    )

    assert agent.semantic_inference_profile is None
    assert agent.semantic_maintenance_configured is True
    assert agent.semantic_maintenance_limits.max_assertions == 7
    assert agent.semantic_maintenance_allow_prior_verified_snapshot is False


def test_managed_maintenance_limits_override_stale_agent_toml(tmp_path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "kestrel.toml").write_text(
        "[semantic_maintenance]\nmax_assertions = 1\n"
    )

    agent = KestrelAgent(
        did="did:test:managed-semantic-maintenance-config",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        semantic_maintenance_limits=SemanticMaintenanceLimits(max_assertions=7),
        semantic_maintenance_configured=True,
    )

    assert agent.semantic_maintenance_configured is True
    assert agent.semantic_maintenance_limits.max_assertions == 7


@pytest.mark.asyncio
async def test_rdfs_multihop_closure_is_idempotent_and_explainable(assertion_store) -> None:
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    class_c = IRI("https://example.test/ClassC")
    subject = IRI("https://example.test/subject")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "b-sub-c", class_b, RDFS_SUBCLASS, class_c)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)

    service = BoundedInferenceService(assertion_store, _profile())
    first = await service.materialize_incremental()
    assert first.status is ClosureStatus.COMPLETE

    inferred = await assertion_store.query(AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_c))
    assert len(inferred) == 1
    explanations = await service.explain(inferred[0].assertion_id)
    assert explanations
    assert all(explanation.premise_revision_ids for explanation in explanations)

    second = await service.materialize_incremental()
    assert second.status is ClosureStatus.COMPLETE
    assert second.generated_assertions == 0


@pytest.mark.asyncio
async def test_independent_derivation_survives_primary_premise_retraction(assertion_store) -> None:
    subject = IRI("https://example.test/subject")
    object_ = IRI("https://example.test/object")
    property_p = IRI("https://example.test/p")
    property_q = IRI("https://example.test/q")
    property_r = IRI("https://example.test/r")
    direct = await _put(assertion_store, "statement", subject, property_p, object_)
    direct_path = await _put(assertion_store, "p-sub-q", property_p, RDFS_SUBPROPERTY, property_q)
    await _put(assertion_store, "p-sub-r", property_p, RDFS_SUBPROPERTY, property_r)
    await _put(assertion_store, "r-sub-q", property_r, RDFS_SUBPROPERTY, property_q)

    service = BoundedInferenceService(assertion_store, _profile())
    result = await service.materialize_incremental()
    assert result.status is ClosureStatus.COMPLETE
    conclusion = (await assertion_store.query(AssertionQuery(subject=subject, predicate=property_q, object=object_)))[0]
    assert len(await service.explain(conclusion.assertion_id)) >= 2

    await assertion_store.retract(direct_path.assertion_id, direct_path.revision_id)
    # The assertion store consults durable alternate derivations during the
    # lifecycle cascade, so the conclusion remains live before the next batch.
    retained = await assertion_store.get_assertion(conclusion.assertion_id)
    assert retained is not None and retained.status.value == "active"
    explanations = await service.explain(conclusion.assertion_id)
    assert explanations
    assert all(
        direct_path.revision_id not in explanation.premise_revision_ids
        for explanation in explanations
    )
    assert await assertion_store.get_assertion(direct.assertion_id) is not None


@pytest.mark.asyncio
async def test_supersession_retracts_ungrounded_alternate_proof_cycle(
    assertion_store,
) -> None:
    """A proof cycle cannot keep its conclusions alive after its only seed leaves.

    ``S Q O`` has a direct proof through ``P subPropertyOf Q`` and an
    alternate proof through inferred ``S R O``.  ``S R O`` in turn has an
    alternate proof through ``S Q O``.  Once the only ``P -> Q`` source is
    superseded, the surviving ledger rows form an ungrounded SCC and must be
    withdrawn before the governed tentative graph is validated or committed.
    """
    predecessor, service, conclusion_q, conclusion_r = (
        await _materialize_ungrounded_alternate_proof_cycle(assertion_store)
    )
    property_p = IRI("https://example.test/p")
    property_t = IRI("https://example.test/t")

    replacement = _assertion(
        assertion_store, "p-sub-t", property_p, RDFS_SUBPROPERTY, property_t
    )
    plan = await assertion_store.plan_supersession_lifecycle(
        predecessor.revision_id, replacement
    )
    planned_revision_ids = {assertion.revision_id for assertion in plan.post_state}
    assert conclusion_q.revision_id not in planned_revision_ids
    assert conclusion_r.revision_id not in planned_revision_ids

    result = await _GOVERNED_STORAGES[id(assertion_store)].supersede_assertion(
        predecessor.revision_id,
        replacement,
        source_occurrences=(_source("source:p-sub-t"),),
    )
    assert result.accepted
    assert await assertion_store.get_assertion(conclusion_q.assertion_id) is None
    assert await assertion_store.get_assertion(conclusion_r.assertion_id) is None
    assert await service.explain(conclusion_q.assertion_id) == ()
    assert await service.explain(conclusion_r.assertion_id) == ()


@pytest.mark.asyncio
async def test_retraction_retracts_ungrounded_alternate_proof_cycle(
    assertion_store,
) -> None:
    """The live retraction cascade uses the same grounded lifecycle plan."""
    predecessor, service, conclusion_q, conclusion_r = (
        await _materialize_ungrounded_alternate_proof_cycle(assertion_store)
    )

    await assertion_store.retract(predecessor.assertion_id, predecessor.revision_id)

    assert await assertion_store.get_assertion(conclusion_q.assertion_id) is None
    assert await assertion_store.get_assertion(conclusion_r.assertion_id) is None
    assert await service.explain(conclusion_q.assertion_id) == ()
    assert await service.explain(conclusion_r.assertion_id) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("delete", "supersede", "validation_loss"))
async def test_lifecycle_changes_deactivate_inference_ledger_before_sleep(
    assertion_store,
    operation: str,
) -> None:
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    subject = IRI("https://example.test/subject")
    hierarchy = await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)
    service = BoundedInferenceService(assertion_store, _profile())
    assert (await service.materialize_incremental()).complete
    conclusion = (
        await assertion_store.query(
            AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
        )
    )[0]

    if operation == "delete":
        await assertion_store.delete(hierarchy.assertion_id, hierarchy.revision_id)
    elif operation == "supersede":
        replacement = _assertion(
            assertion_store,
            "a-sub-b-replacement",
            class_a,
            RDFS_SUBCLASS,
            class_b,
        )
        result = await _GOVERNED_STORAGES[id(assertion_store)].supersede_assertion(
            hierarchy.revision_id,
            replacement,
            source_occurrences=(_source("source:a-sub-b-replacement"),),
        )
        assert result.accepted
    else:
        await assertion_store.invalidate_assertion_eligibility(
            hierarchy.assertion_id, hierarchy.revision_id
        )

    assert await service.explain(conclusion.assertion_id) == ()
    assert await assertion_store.query(
        AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
    ) == []


@pytest.mark.asyncio
async def test_erasure_rematerializes_independently_supported_conclusion(assertion_store) -> None:
    """Physical erasure must not delete a conclusion with another proof."""
    subject = IRI("https://example.test/subject")
    object_ = IRI("https://example.test/object")
    property_p = IRI("https://example.test/p")
    property_q = IRI("https://example.test/q")
    property_r = IRI("https://example.test/r")
    await _put(assertion_store, "statement", subject, property_p, object_)
    direct_path = await _put(
        assertion_store, "p-sub-q", property_p, RDFS_SUBPROPERTY, property_q
    )
    await _put(assertion_store, "p-sub-r", property_p, RDFS_SUBPROPERTY, property_r)
    indirect_terminal = await _put(
        assertion_store, "r-sub-q", property_r, RDFS_SUBPROPERTY, property_q
    )

    service = BoundedInferenceService(assertion_store, _profile())
    assert (await service.materialize_incremental()).complete
    conclusion = (
        await assertion_store.query(
            AssertionQuery(subject=subject, predicate=property_q, object=object_)
        )
    )[0]
    old_conclusion_revision = conclusion.revision_id
    # The proof selected for canonical lineage is deterministic but hash
    # ordered. Erase its path-specific direct premise so the other complete
    # proof must be selected for the fresh canonical revision.
    primary_revision_ids = set(conclusion.lineage.input_revision_ids)
    erased_premise = (
        direct_path
        if direct_path.revision_id in primary_revision_ids
        else indirect_terminal
    )
    assert erased_premise.revision_id in primary_revision_ids

    erasure = await assertion_store.erase(erased_premise.assertion_id)

    retained = await assertion_store.query(
        AssertionQuery(subject=subject, predicate=property_q, object=object_)
    )
    assert len(retained) == 1
    fresh = retained[0]
    assert fresh.revision_id != old_conclusion_revision
    assert fresh.revision_id not in erasure.erased_revision_ids
    assert isinstance(fresh.lineage, DerivedLineage)
    assert not set(fresh.lineage.input_revision_ids).intersection(
        erasure.erased_revision_ids
    )
    assert not set(
        item.revision_id
        for item in await assertion_store.derivation_inputs(fresh.revision_id)
    ).intersection(erasure.erased_revision_ids)
    explanations = await service.explain(fresh.assertion_id)
    assert explanations
    assert all(
        not set(explanation.premise_revision_ids).intersection(
            erasure.erased_revision_ids
        )
        for explanation in explanations
    )


@pytest.mark.asyncio
async def test_explicit_inference_revocation_retracts_materializations_and_ledgers(
    assertion_store,
) -> None:
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    subject = IRI("https://example.test/subject")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)
    service = BoundedInferenceService(assertion_store, _profile())
    assert (await service.materialize_incremental()).complete
    inferred = await assertion_store.query(
        AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
    )
    assert inferred

    result = await assertion_store.revoke_semantic_inference(ENGINE_VERSION)

    assert result.retracted_assertions >= 1
    assert result.deactivated_derivations >= 1
    assert await assertion_store.query(
        AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
    ) == []
    active_ledgers = await assertion_store._database.fetchval(  # noqa: SLF001
        "SELECT COUNT(*) FROM semantic_inference_derivations "
        "WHERE tenant_id = ? AND active = 1",
        (assertion_store.tenant_id,),
    )
    assert active_ledgers == 0


@pytest.mark.asyncio
async def test_initialized_privacy_storage_revokes_disabled_inference(tmp_path) -> None:
    """The normal initialized agent path keeps disablement governed and effective."""
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Privacy semantic test"
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(
        credentials.agent_did, identity
    )
    raw_storage = AsyncStorage(
        ":memory:",
        agent_id=credentials.agent_did,
        _assertion_tenant_capability=capability,
    )
    await raw_storage.initialize()
    try:
        store = raw_storage._assertion_store()
        _GOVERNED_STORAGES[id(store)] = raw_storage
        class_a = IRI("https://example.test/ClassA")
        class_b = IRI("https://example.test/ClassB")
        subject = IRI("https://example.test/subject")
        await _put(store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
        await _put(store, "subject-a", subject, RDF_TYPE, class_a)
        storage = PrivacyEnforcingStorage(raw_storage, "normal")
        assert (await storage.materialize_semantic_inference(_profile())).complete

        class Agent(SleepMixin):
            semantic_inference_profile = None
            semantic_inference_configured = True

            def __init__(self) -> None:
                self.storage = storage

        report = await Agent().sleep(
            skip_consolidation=True,
            skip_export=True,
            skip_reflection=True,
        )
        assert report.semantic_inference is not None
        assert report.semantic_inference["status"] == "disabled"
        assert await raw_storage.query_assertions(
            AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
        ) == []
    finally:
        _GOVERNED_STORAGES.pop(id(store), None)
        await raw_storage.close()


@pytest.mark.asyncio
async def test_inference_reactivation_cannot_replace_retracted_direct_assertion(assertion_store) -> None:
    direct = await _put(
        assertion_store,
        "direct-statement",
        IRI("https://example.test/subject"),
        IRI("https://example.test/property"),
        IRI("https://example.test/object"),
    )
    await assertion_store.retract(direct.assertion_id, direct.revision_id)
    attempted_inference = replace(
        direct,
        revision_id="inferred-revision",
        confidence=Decimal("1"),
        confidence_method="semantic-kb-materializer-v1",
        confidence_basis="fixture",
        epistemic_state=EpistemicState.INFERRED,
        lineage=DerivedLineage(
            rule_id="rdfs:fixture",
            engine_version="semantic-kb-materializer-v1",
            profile_version="rdfs-v1@1.0.0",
            input_revision_ids=("independent-premise",),
            input_digest="sha256:fixture",
            run_id="inference:fixture",
            generated_at="2026-07-26T12:00:00Z",
        ),
    )

    with pytest.raises(AssertionConflictError, match="non-derived assertion identity"):
        await assertion_store.reactivate_inferred(attempted_inference)

    current = await assertion_store.get_assertion(direct.assertion_id, include_inactive=True)
    assert current is not None
    assert current.epistemic_state is EpistemicState.RETRACTED
    assert not isinstance(current.lineage, DerivedLineage)


@pytest.mark.asyncio
async def test_concurrent_direct_write_is_not_covered_by_complete_checkpoint(assertion_store) -> None:
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    class_c = IRI("https://example.test/ClassC")
    subject = IRI("https://example.test/subject")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)

    service = BoundedInferenceService(assertion_store, _profile())
    original_persist = service._persist_facts
    direct_write: asyncio.Task | None = None

    async def persist_while_direct_write_arrives(facts, run_id, started):
        nonlocal direct_write
        if direct_write is None:
            direct_write = asyncio.create_task(
                _put(assertion_store, "b-sub-c", class_b, RDFS_SUBCLASS, class_c)
            )
            # Let the writer reach the tenant lock while the materializer is
            # publishing.  It must wait until the complete checkpoint commits.
            await asyncio.sleep(0)
        return await original_persist(facts, run_id, started)

    service._persist_facts = persist_while_direct_write_arrives  # type: ignore[method-assign]
    first = await service.materialize_incremental()
    assert first.complete
    assert direct_write is not None
    await direct_write

    second = await service.materialize_incremental()
    assert second.complete
    assert await assertion_store.query(
        AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_c)
    )


@pytest.mark.asyncio
async def test_sleep_runs_incremental_inference_for_approved_profile() -> None:
    profile = _profile()
    limits = InferenceLimits(max_source_assertions=23)

    class Storage:
        def __init__(self) -> None:
            self.profiles = []

        async def run_semantic_maintenance(
            self,
            selected_profile,
            *,
            inference_limits=None,
            maintenance_limits=None,
        ):
            self.profiles.append(selected_profile)
            self.limits = inference_limits
            return SimpleNamespace(
                status=SemanticMaintenanceStatus.COMPLETE,
                reason=None,
                source_generation=3,
                checkpoint_generation=3,
                assertions_inferred=0,
                assertions_retracted=0,
                to_mapping=lambda: {
                    "status": "complete",
                    "source_generation": 3,
                    "checkpoint_generation": 3,
                    "changes_consumed": 0,
                    "assertions_validated": 0,
                    "assertions_inferred": 0,
                    "assertions_retracted": 0,
                },
            )

    class Agent(SleepMixin):
        def __init__(self) -> None:
            self.semantic_inference_profile = profile
            self.semantic_inference_limits = limits
            self.storage = Storage()

    agent = Agent()
    report = await agent.sleep(
        skip_consolidation=True,
        skip_export=True,
        skip_reflection=True,
    )
    assert agent.storage.profiles == [profile]
    assert agent.storage.limits is limits
    assert report.semantic_inference == {
        "status": "complete",
        "incomplete_reason": None,
        "source_generation": 3,
        "checkpoint_generation": 3,
        "generated_assertions": 0,
        "retracted_assertions": 0,
    }


@pytest.mark.asyncio
async def test_sleep_runs_validation_only_maintenance_without_an_inference_profile() -> None:
    class Storage:
        def __init__(self) -> None:
            self.profiles: list[object] = []

        async def run_semantic_maintenance(self, selected_profile, **kwargs):
            self.profiles.append(selected_profile)
            assert kwargs["maintenance_limits"].max_assertions == 3
            return SimpleNamespace(
                status=SemanticMaintenanceStatus.COMPLETE,
                reason=None,
                source_generation=3,
                checkpoint_generation=3,
                assertions_inferred=0,
                assertions_retracted=0,
                to_mapping=lambda: {"status": "complete"},
            )

    class Agent(SleepMixin):
        def __init__(self) -> None:
            self.semantic_inference_profile = None
            self.semantic_inference_configured = False
            self.semantic_maintenance_configured = True
            self.semantic_maintenance_limits = SemanticMaintenanceLimits(max_assertions=3)
            self.storage = Storage()

    agent = Agent()
    report = await agent.sleep(
        skip_consolidation=True,
        skip_export=True,
        skip_reflection=True,
    )

    assert agent.storage.profiles == [None]
    assert report.semantic_maintenance == {"status": "complete"}


@pytest.mark.asyncio
async def test_semantic_maintenance_second_unchanged_run_is_a_true_noop(
    assertion_store, monkeypatch
) -> None:
    """No-change sleep must not wake either validator or reasoner again."""
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=_profile(),
        limits=SemanticMaintenanceLimits(max_assertions=3, max_derivations=3),
    )
    async def should_not_validate(*args, **kwargs):
        raise AssertionError("no-change maintenance called validation")

    async def should_not_reason(*args, **kwargs):
        raise AssertionError("no-change maintenance called inference")

    monkeypatch.setattr(
        "kestrel_sovereign.storage.semantic_validation.GovernedSemanticValidationService.validate_current",
        should_not_validate,
    )
    monkeypatch.setattr(
        BoundedInferenceService,
        "materialize_incremental",
        should_not_reason,
    )

    first = await service.run()
    assert first.status is SemanticMaintenanceStatus.NO_OP
    second = await service.run()
    assert second.status is SemanticMaintenanceStatus.NO_OP
    assert second.changes_consumed == 0
    assert second.assertions_validated == 0
    assert second.assertions_inferred == 0


@pytest.mark.asyncio
async def test_semantic_maintenance_replays_its_generated_derivation_before_checkpointing(
    assertion_store,
) -> None:
    """A derived assertion is a fresh input for the next bounded unit."""
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    subject = IRI("https://example.test/subject")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)

    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=_profile(),
        limits=SemanticMaintenanceLimits(max_assertions=3, max_derivations=3),
    )
    first = await service.run()
    assert first.status is SemanticMaintenanceStatus.PARTIAL
    assert first.reason == "repair_change_replay"
    assert first.assertions_inferred == 1

    state_row = await assertion_store._database.fetchone(  # noqa: SLF001 - durable cursor contract
        "SELECT checkpoint_event_id FROM semantic_maintenance_state WHERE tenant_id = ?",
        (assertion_store.tenant_id,),
    )
    assert state_row is not None
    assert state_row[0] != (await assertion_store.checkpoint()).latest_event_id

    second = await service.run()
    assert second.status is SemanticMaintenanceStatus.COMPLETE
    assert second.changes_consumed == 1
    assert second.assertions_validated == 1
    # A subsequent unchanged unit must leave the durable training prerequisite
    # ready; it cannot skip the derived assertion by checkpointing it early.
    third = await service.run()
    assert third.status is SemanticMaintenanceStatus.NO_OP
    assert third.changes_consumed == 0
    readiness = await service.training_readiness()
    assert readiness.ready


@pytest.mark.asyncio
async def test_training_readiness_rejects_historical_maintenance_without_snapshot(
    assertion_store,
) -> None:
    """A historical run is not a durable governed corpus snapshot."""
    await _put(
        assertion_store,
        "verified-first",
        IRI("https://example.test/subject"),
        IRI("https://example.test/predicate"),
        Literal("first", XSD_STRING),
    )
    service = SemanticMaintenanceService(assertion_store, inference_profile=None)
    assert (await service.run()).status is SemanticMaintenanceStatus.COMPLETE
    assert (await service.training_readiness()).ready

    await _put(
        assertion_store,
        "unmaintained-second",
        IRI("https://example.test/subject"),
        IRI("https://example.test/predicate"),
        Literal("second", XSD_STRING),
    )
    current = await service.training_readiness()
    assert not current.ready
    assert current.reason == "semantic_maintenance_checkpoint_behind"

    with pytest.raises(SemanticMaintenanceError, match="durable governed corpus snapshot"):
        await service.training_readiness(allow_prior_verified_snapshot=True)


@pytest.mark.asyncio
async def test_semantic_repair_pages_with_a_durable_cursor(assertion_store) -> None:
    """Repeated repair calls resume after completed pages instead of rescanning."""
    predicate = IRI("https://example.test/repair-predicate")
    for revision in ("repair-a", "repair-b", "repair-c"):
        await _put(
            assertion_store,
            revision,
            IRI(f"https://example.test/{revision}"),
            predicate,
            Literal(revision, XSD_STRING),
        )

    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=1),
    )

    first = await service.rebuild()
    assert first.status is SemanticMaintenanceStatus.PARTIAL
    assert first.backlog_assertions == 2
    second = await service.rebuild()
    assert second.status is SemanticMaintenanceStatus.PARTIAL
    assert second.backlog_assertions == 1
    third = await service.rebuild()
    assert third.status is SemanticMaintenanceStatus.COMPLETE
    assert third.backlog_assertions == 0

    state = await assertion_store._database.fetchone(  # noqa: SLF001 - cursor contract
        "SELECT repair_cursor_revision_id, repair_active "
        "FROM semantic_maintenance_state WHERE tenant_id = ?",
        (assertion_store.tenant_id,),
    )
    assert state == (None, 0)


@pytest.mark.asyncio
async def test_explicit_repair_uses_bounded_context_on_each_durable_page(
    assertion_store, monkeypatch
) -> None:
    """Explicit repair never hides an unbounded snapshot behind one page."""
    predicate = IRI("https://example.test/repair-complete-context")
    for revision in ("context-a", "context-b", "context-c"):
        await _put(
            assertion_store,
            revision,
            IRI(f"https://example.test/{revision}"),
            predicate,
            Literal(revision, XSD_STRING),
        )
    observed: list[bool] = []
    original = GovernedSemanticValidationService.validate_current

    async def capture_context(self, *args, **kwargs):
        observed.append(kwargs["bounded_focus_only"])
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(
        GovernedSemanticValidationService,
        "validate_current",
        capture_context,
    )
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=1),
    )

    assert (await service.rebuild()).status is SemanticMaintenanceStatus.PARTIAL
    assert (await service.rebuild()).status is SemanticMaintenanceStatus.PARTIAL
    assert (await service.rebuild()).status is SemanticMaintenanceStatus.COMPLETE
    assert observed == [True, True, True]


@pytest.mark.asyncio
async def test_interrupted_explicit_repair_keeps_its_durable_mode_on_sleep_resume(
    assertion_store, monkeypatch
) -> None:
    """A normal sleep continues the explicit repair instead of resetting it."""
    predicate = IRI("https://example.test/repair-resume-mode")
    for revision in ("resume-a", "resume-b", "resume-c"):
        await _put(
            assertion_store,
            revision,
            IRI(f"https://example.test/{revision}"),
            predicate,
            Literal(revision, XSD_STRING),
        )
    observed: list[bool] = []
    original = GovernedSemanticValidationService.validate_current

    async def capture_context(self, *args, **kwargs):
        observed.append(kwargs["bounded_focus_only"])
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(
        GovernedSemanticValidationService,
        "validate_current",
        capture_context,
    )
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=1),
    )

    assert (await service.rebuild()).status is SemanticMaintenanceStatus.PARTIAL
    state = await assertion_store._database.fetchone(  # noqa: SLF001 - durable repair contract
        "SELECT repair_mode, repair_active FROM semantic_maintenance_state "
        "WHERE tenant_id = ?",
        (assertion_store.tenant_id,),
    )
    assert state == ("full_rebuild", 1)

    assert (await service.run()).status is SemanticMaintenanceStatus.PARTIAL
    assert (await service.run()).status is SemanticMaintenanceStatus.COMPLETE
    assert observed == [True, True, True]


@pytest.mark.asyncio
async def test_explicit_repair_never_exports_the_tenant_for_one_page(
    assertion_store, monkeypatch
) -> None:
    """Repair uses the same bounded validation context as incremental sleep."""
    for revision in ("repair-bounded-a", "repair-bounded-b"):
        await _put(
            assertion_store,
            revision,
            IRI(f"https://example.test/{revision}"),
            IRI("https://example.test/repair-bounded-predicate"),
            Literal(revision, XSD_STRING),
        )

    async def must_not_export(*_args, **_kwargs):
        raise AssertionError("explicit repair exported the tenant graph")

    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_assertion_store.AsyncAssertionStore.export_snapshot",
        must_not_export,
    )
    result = await SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=1),
    ).rebuild()

    assert result.status is SemanticMaintenanceStatus.PARTIAL
    assert result.assertions_validated == 1


@pytest.mark.asyncio
async def test_semantic_repair_replays_a_write_behind_its_lexical_cursor(
    assertion_store,
) -> None:
    """A concurrent write cannot be skipped merely because its revision sorts earlier."""
    predicate = IRI("https://example.test/repair-race-predicate")
    for revision in ("repair-a", "repair-b", "repair-c"):
        await _put(
            assertion_store,
            revision,
            IRI(f"https://example.test/{revision}"),
            predicate,
            Literal(revision, XSD_STRING),
        )
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=1),
    )

    first = await service.rebuild()
    assert first.status is SemanticMaintenanceStatus.PARTIAL
    await _put(
        assertion_store,
        "repair-0",
        IRI("https://example.test/repair-race-late"),
        predicate,
        Literal("late", XSD_STRING),
    )

    assert (await service.rebuild()).status is SemanticMaintenanceStatus.PARTIAL
    replay = await service.rebuild()
    assert replay.status is SemanticMaintenanceStatus.PARTIAL
    assert replay.reason == "repair_change_replay"

    consumed = await service.run()
    assert consumed.status is SemanticMaintenanceStatus.COMPLETE
    assert consumed.changes_consumed == 1
    assert consumed.assertions_validated == 1


@pytest.mark.asyncio
async def test_bounded_maintenance_uses_targeted_inference_not_a_global_source_scan(
    assertion_store,
) -> None:
    """A one-assertion maintenance page advances even when the KB has two sources."""
    class_a = IRI("https://example.test/TargetedClassA")
    class_b = IRI("https://example.test/TargetedClassB")
    subject = IRI("https://example.test/targeted-subject")
    await _put(assertion_store, "targeted-a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "targeted-subject-a", subject, RDF_TYPE, class_a)
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=_profile(),
        limits=SemanticMaintenanceLimits(max_assertions=1, max_derivations=3),
    )

    first = await service.run()
    assert first.status is SemanticMaintenanceStatus.PARTIAL
    assert first.reason != "source_assertions"
    second = await service.run()
    assert second.reason != "source_assertions"
    # The generated assertion is deliberately replayed before the durable
    # maintenance cursor reaches the final no-op state.
    for _ in range(3):
        result = await service.run()
        if result.status is SemanticMaintenanceStatus.NO_OP:
            break
        assert result.reason != "source_assertions"
    else:
        pytest.fail("targeted inference did not drain its bounded maintenance work")


@pytest.mark.asyncio
async def test_targeted_inference_defers_when_indexed_context_overflows(
    assertion_store,
) -> None:
    """A filled context page must not be mistaken for a complete closure."""
    subject = IRI("https://example.test/context-overflow-subject")
    predicate = IRI("https://example.test/context-overflow-predicate")
    target = await _put(
        assertion_store,
        "context-overflow-a-target",
        subject,
        predicate,
        IRI("https://example.test/context-overflow-object-a"),
    )
    await _put(
        assertion_store,
        "context-overflow-z-peer",
        subject,
        predicate,
        IRI("https://example.test/context-overflow-object-z"),
    )

    result = await BoundedInferenceService(
        assertion_store,
        _profile(),
        limits=InferenceLimits(max_source_assertions=2),
    ).materialize_targets(
        (target.assertion_id,),
        max_context_assertions=1,
    )

    assert result.status is ClosureStatus.INCOMPLETE
    assert result.incomplete_reason == "context_assertions"


@pytest.mark.asyncio
async def test_incremental_maintenance_never_exports_the_tenant_for_one_target(
    assertion_store, monkeypatch
) -> None:
    """``max_assertions`` bounds primary validation reads, not just focus IDs."""
    for revision in ("bounded-a", "bounded-b"):
        await _put(
            assertion_store,
            revision,
            IRI(f"https://example.test/{revision}"),
            IRI("https://example.test/bounded-predicate"),
            Literal(revision, XSD_STRING),
        )

    async def must_not_export(*_args, **_kwargs):
        raise AssertionError("incremental validation exported the tenant graph")

    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_assertion_store.AsyncAssertionStore.export_snapshot",
        must_not_export,
    )
    result = await SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=1),
    ).run()

    assert result.status is SemanticMaintenanceStatus.PARTIAL
    assert result.assertions_validated == 1
    assert result.backlog_assertions == 1


@pytest.mark.asyncio
async def test_incremental_contradiction_compares_changed_assertion_to_older_peer(
    assertion_store,
) -> None:
    """Pair deduplication must not depend on assertion-ID ordering."""
    subject = IRI("https://example.test/subject")
    property_p = IRI("https://example.test/p")
    property_q = IRI("https://example.test/q")
    object_y = IRI("https://example.test/y")
    object_z = IRI("https://example.test/z")
    await _put(assertion_store, "p-sub-q", property_p, RDFS_SUBPROPERTY, property_q)
    await _put(assertion_store, "source-p", subject, property_p, object_y)
    await _put(assertion_store, "existing-q", subject, property_q, object_z)

    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=_profile(),
        limits=SemanticMaintenanceLimits(max_assertions=3, max_derivations=3),
    )
    first = await service.run()
    assert first.status is SemanticMaintenanceStatus.PARTIAL
    assert first.reason == "repair_change_replay"

    result = await service.run()
    assert result.status is SemanticMaintenanceStatus.COMPLETE
    assert result.contradictions == 1
    report = await assertion_store._database.fetchone(  # noqa: SLF001 - durable evidence
        "SELECT report_kind, status FROM semantic_maintenance_reports "
        "WHERE tenant_id = ?",
        (assertion_store.tenant_id,),
    )
    assert report == ("contradiction_candidate", "review_required")


@pytest.mark.asyncio
async def test_contradiction_context_excludes_the_changed_assertion_at_query_time(
    assertion_store,
) -> None:
    """A one-row contextual page must contain a competitor, not the target itself."""
    subject = IRI("https://example.test/context-subject")
    predicate = IRI("https://example.test/context-predicate")
    await _put(assertion_store, "context-existing", subject, predicate, Literal("old", XSD_STRING))
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=1, max_context_assertions=1),
    )
    assert (await service.run()).status is SemanticMaintenanceStatus.COMPLETE

    await _put(assertion_store, "context-changed", subject, predicate, Literal("new", XSD_STRING))
    result = await service.run()

    assert result.status is SemanticMaintenanceStatus.COMPLETE
    assert result.contradictions == 1


@pytest.mark.asyncio
async def test_contradiction_context_pages_remaining_competitors(assertion_store) -> None:
    """A contested predicate resumes after the contextual competitor cursor."""
    subject = IRI("https://example.test/context-page-subject")
    predicate = IRI("https://example.test/context-page-predicate")
    await _put(assertion_store, "context-peer-a", subject, predicate, Literal("a", XSD_STRING))
    await _put(assertion_store, "context-peer-b", subject, predicate, Literal("b", XSD_STRING))
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=2, max_context_assertions=1),
    )
    assert (await service.run()).status is SemanticMaintenanceStatus.COMPLETE

    await _put(assertion_store, "context-page-changed", subject, predicate, Literal("c", XSD_STRING))
    first = await service.run()
    assert first.status is SemanticMaintenanceStatus.PARTIAL
    assert first.reason == "contradiction_context_budget"
    second = await service.run()
    assert second.status is SemanticMaintenanceStatus.COMPLETE
    reports = await assertion_store._database.fetchval(  # noqa: SLF001 - durable backlog contract
        "SELECT COUNT(*) FROM semantic_maintenance_reports WHERE tenant_id = ? "
        "AND report_kind = 'contradiction_candidate'",
        (assertion_store.tenant_id,),
    )
    assert reports == 3


@pytest.mark.asyncio
async def test_incomplete_validation_is_not_reused_when_maintenance_budget_changes(
    assertion_store, monkeypatch
) -> None:
    """A larger budget gets a new validation run rather than stale partial work."""
    await _put(
        assertion_store,
        "budget-subject",
        IRI("https://example.test/subject"),
        IRI("https://example.test/predicate"),
        Literal("object", XSD_STRING),
    )
    calls: list[str] = []
    original = GovernedSemanticValidationService.validate_current

    async def incomplete_once(self, *args, **kwargs):
        calls.append(kwargs["run_id"])
        report = await original(self, *args, **kwargs)
        return replace(
            report,
            state=ValidationState.INCOMPLETE,
            action=ValidationWriteAction.REJECT,
        )

    monkeypatch.setattr(
        GovernedSemanticValidationService,
        "validate_current",
        incomplete_once,
    )
    tight = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_shapes=1),
    )
    first = await tight.run()
    assert first.status is SemanticMaintenanceStatus.PARTIAL
    assert first.reason == "validation_incomplete"

    expanded = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_shapes=2),
    )
    second = await expanded.run()
    assert second.status is SemanticMaintenanceStatus.PARTIAL
    assert second.reason == "validation_incomplete"
    assert len(calls) == 2
    assert calls[0] != calls[1]


@pytest.mark.asyncio
async def test_context_budget_change_invalidates_completed_maintenance_checkpoint(
    assertion_store,
) -> None:
    """Context bounds are part of the maintenance capability identity."""
    await _put(
        assertion_store,
        "context-budget-capability",
        IRI("https://example.test/context-budget-subject"),
        IRI("https://example.test/context-budget-predicate"),
        Literal("value", XSD_STRING),
    )
    initial = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_context_assertions=1),
    )
    assert (await initial.run()).status is SemanticMaintenanceStatus.COMPLETE

    expanded = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_context_assertions=2),
    )
    result = await expanded.run()

    assert result.status is SemanticMaintenanceStatus.COMPLETE
    assert result.assertions_validated == 1
    assert result.status is not SemanticMaintenanceStatus.NO_OP


@pytest.mark.asyncio
async def test_semantic_maintenance_records_review_only_contradiction_candidate(
    assertion_store,
) -> None:
    """Different current values become review evidence, never LLM arbitration."""
    subject = IRI("https://example.test/user")
    predicate = IRI("https://example.test/preferred-region")
    await _put(
        assertion_store,
        "region-a",
        subject,
        predicate,
        Literal("us-central1", XSD_STRING),
    )
    await _put(
        assertion_store,
        "region-b",
        subject,
        predicate,
        Literal("europe-west4", XSD_STRING),
    )

    result = await SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=2, max_reports=2),
    ).run()

    assert result.status is SemanticMaintenanceStatus.COMPLETE
    assert result.contradictions == 1
    assert result.supersession_candidates == 1
    report = await assertion_store._database.fetchone(  # noqa: SLF001 - durable audit assertion
        "SELECT report_kind, status FROM semantic_maintenance_reports "
        "WHERE tenant_id = ?",
        (assertion_store.tenant_id,),
    )
    assert report == ("contradiction_candidate", "review_required")


@pytest.mark.asyncio
async def test_semantic_maintenance_report_budget_limits_candidate_writes(
    assertion_store,
) -> None:
    subject = IRI("https://example.test/user")
    predicate = IRI("https://example.test/preferred-region")
    for revision, region in (
        ("region-a", "us-central1"),
        ("region-b", "europe-west4"),
        ("region-c", "asia-south1"),
    ):
        await _put(
            assertion_store,
            revision,
            subject,
            predicate,
            Literal(region, XSD_STRING),
        )

    result = await SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=3, max_reports=2),
    ).run()

    assert result.status is SemanticMaintenanceStatus.PARTIAL
    assert result.reason == "report_budget"
    # One report slot is the bounded validation report; one remains for a
    # contradiction candidate.  The next candidate stays behind the checkpoint.
    assert result.reports_created == 2
    assert result.backlog_assertions >= 1
    candidate_count = await assertion_store._database.fetchval(  # noqa: SLF001
        "SELECT COUNT(*) FROM semantic_maintenance_reports WHERE tenant_id = ?",
        (assertion_store.tenant_id,),
    )
    assert candidate_count == 1


@pytest.mark.asyncio
async def test_semantic_maintenance_lease_is_atomic_and_fences_state_writes(
    assertion_store,
) -> None:
    first_service = SemanticMaintenanceService(assertion_store, inference_profile=None)
    second_service = SemanticMaintenanceService(assertion_store, inference_profile=None)

    first, second = await asyncio.gather(
        first_service._acquire_lease("first"),  # noqa: SLF001 - lease contract
        second_service._acquire_lease("second"),  # noqa: SLF001 - lease contract
    )
    winner = first or second
    assert winner is not None
    assert (first is None) != (second is None)
    try:
        await assertion_store._database.execute(  # noqa: SLF001 - force expiry
            "UPDATE semantic_maintenance_leases SET expires_at = 0 WHERE tenant_id = ?",
            (assertion_store.tenant_id,),
        )
        successor = await second_service._acquire_lease("successor")  # noqa: SLF001
        assert successor is not None
        try:
            result = first_service._result(  # noqa: SLF001 - fixed state fixture
                run_id="fenced-state",
                status=SemanticMaintenanceStatus.COMPLETE,
                reason=None,
                source_generation=0,
                checkpoint_generation=0,
                capability_versions={},
            )
            with pytest.raises(
                SemanticMaintenanceError, match="semantic_maintenance_lease_lost"
            ):
                await first_service._record_state(  # noqa: SLF001 - fencing contract
                    result,
                    "profile",
                    winner,
                )
        finally:
            await second_service._release_lease(successor)  # noqa: SLF001
    finally:
        await first_service._release_lease(winner)  # noqa: SLF001


@pytest.mark.asyncio
async def test_semantic_maintenance_resumes_same_generation_supersession_events(
    assertion_store,
) -> None:
    """A partial batch advances by event, never by generation alone."""
    subject = IRI("https://example.test/subject")
    predicate = IRI("https://example.test/current-value")
    predecessor = await _put(
        assertion_store,
        "value-before",
        subject,
        predicate,
        Literal("before", XSD_STRING),
    )
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=1),
    )
    assert (await service.run()).status is SemanticMaintenanceStatus.COMPLETE

    replacement = _assertion(
        assertion_store,
        "value-after",
        subject,
        predicate,
        Literal("after", XSD_STRING),
    )
    supersession = await _GOVERNED_STORAGES[id(assertion_store)].supersede_assertion(
        predecessor.revision_id,
        replacement,
        source_occurrences=(_source("source:value-after"),),
    )
    assert supersession.accepted
    assert len(supersession.event_ids) == 2

    first = await service.run()
    assert first.status is SemanticMaintenanceStatus.PARTIAL
    assert first.reason == "assertion_budget"
    assert first.checkpoint_generation == supersession.generation

    cursor = await assertion_store._database.fetchone(  # noqa: SLF001 - durable cursor contract
        "SELECT checkpoint_event_id FROM semantic_maintenance_state WHERE tenant_id = ?",
        (assertion_store.tenant_id,),
    )
    assert cursor is not None
    assert cursor[0] in supersession.event_ids

    second = await service.run()
    assert second.status is SemanticMaintenanceStatus.COMPLETE
    assert second.status is not SemanticMaintenanceStatus.NO_OP
    assert second.changes_consumed == 1


@pytest.mark.asyncio
async def test_semantic_maintenance_revalidates_active_neighbours_after_delete(
    assertion_store,
    monkeypatch,
) -> None:
    """A hidden deleted revision still schedules its active shape neighbours."""
    subject = IRI("https://example.test/account")
    predicate = IRI("https://example.test/has-status")
    deleted = await _put(
        assertion_store,
        "status-old",
        subject,
        predicate,
        Literal("pending", XSD_STRING),
    )
    surviving = await _put(
        assertion_store,
        "status-current",
        subject,
        predicate,
        Literal("active", XSD_STRING),
    )
    service = SemanticMaintenanceService(
        assertion_store,
        inference_profile=None,
        limits=SemanticMaintenanceLimits(max_assertions=2),
    )
    assert (await service.run()).status is SemanticMaintenanceStatus.COMPLETE

    await assertion_store.delete(deleted.assertion_id, deleted.revision_id)
    observed_focus: list[tuple[str, ...] | None] = []
    original = GovernedSemanticValidationService.validate_current

    async def capture_focus(self, *args, **kwargs):
        observed_focus.append(kwargs.get("assertion_ids"))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(
        GovernedSemanticValidationService,
        "validate_current",
        capture_focus,
    )
    result = await service.run()

    assert result.status is SemanticMaintenanceStatus.COMPLETE
    assert observed_focus
    assert surviving.assertion_id in set(observed_focus[0] or ())
    assert await assertion_store.get_assertion(surviving.assertion_id) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("validation", "audit", "inference"))
async def test_semantic_maintenance_fence_blocks_expired_phase_writes(
    assertion_store,
    phase: str,
) -> None:
    """Expiry blocks validation, audit, and inference publication at commit time."""
    subject = IRI("https://example.test/fenced-subject")
    predicate = IRI("https://example.test/fenced-predicate")
    assertion = await _put(
        assertion_store,
        f"fenced-{phase}",
        subject,
        predicate,
        Literal("value", XSD_STRING),
    )
    maintenance = SemanticMaintenanceService(assertion_store, inference_profile=None)
    lease = await maintenance._acquire_lease(f"{phase}-holder")  # noqa: SLF001 - fence contract
    assert lease is not None
    try:
        async with assertion_store.maintenance_fence(
            holder_id=lease.holder_id,
            fencing_token=lease.fencing_token,
            lease_seconds=maintenance._LEASE_SECONDS,  # noqa: SLF001 - matching contract lifetime
        ):
            await assertion_store._database.execute(  # noqa: SLF001 - force stale worker
                "UPDATE semantic_maintenance_leases SET expires_at = 0 WHERE tenant_id = ?",
                (assertion_store.tenant_id,),
            )
            with pytest.raises(MaintenanceLeaseLostError):
                if phase == "validation":
                    await GovernedSemanticValidationService(
                        assertion_store
                    ).validate_current(
                        assertion_ids=(assertion.assertion_id,),
                        run_id=f"fenced-{phase}",
                    )
                elif phase == "audit":
                    await assertion_store.retract(
                        assertion.assertion_id,
                        assertion.revision_id,
                        operation_id=f"fenced-{phase}",
                    )
                else:
                    service = BoundedInferenceService(assertion_store, _profile())
                    await service.materialize_incremental()
        assert await assertion_store.get_assertion(assertion.assertion_id) is not None
    finally:
        await maintenance._release_lease(lease)  # noqa: SLF001 - fence cleanup


@pytest.mark.asyncio
async def test_sleep_blocks_training_hook_after_partial_core_semantic_maintenance() -> None:
    """The core phase is a real #2749 dependency, not a late side effect."""
    profile = _profile()
    calls: list[str] = []

    class Storage:
        async def run_semantic_maintenance(self, *args, **kwargs):
            return SimpleNamespace(
                to_mapping=lambda: {
                    "status": "partial",
                    "reason": "assertion_budget",
                    "source_generation": 3,
                    "checkpoint_generation": 2,
                    "changes_consumed": 1,
                    "assertions_validated": 1,
                    "assertions_inferred": 0,
                    "assertions_retracted": 0,
                },
                status=SimpleNamespace(value="partial"),
                reason="assertion_budget",
                source_generation=3,
                checkpoint_generation=2,
                assertions_inferred=0,
                assertions_retracted=0,
            )

    class TrainingHook:
        sleep_hook_contract = SleepHookContract(
            hook_id="test.training",
            phase=SleepHookPhase.TRAINING,
        )

        async def on_post_consolidation(self, agent, result):
            calls.append("training")
            return {"success": True}

    class Agent(SleepMixin):
        def __init__(self) -> None:
            self.semantic_inference_profile = profile
            self.semantic_inference_limits = InferenceLimits()
            self.semantic_maintenance_limits = SemanticMaintenanceLimits()
            self.storage = Storage()
            self.sleep_hooks = [TrainingHook()]
            self.on_sleep_complete = None

        async def _consolidate_memories(self):
            return {"episodes_created": 1}

    report = await Agent().sleep(skip_export=True)

    assert calls == []
    core = next(
        item
        for item in report.hook_results
        if item.hook_id == "kestrel_sovereign.semantic_maintenance"
    )
    training = next(item for item in report.hook_results if item.hook_id == "test.training")
    assert core.status.value == "failed"
    assert training.status.value == "blocked"
    assert report.semantic_maintenance == {
        "status": "partial",
        "reason": "assertion_budget",
        "source_generation": 3,
        "checkpoint_generation": 2,
        "changes_consumed": 1,
        "assertions_validated": 1,
        "assertions_inferred": 0,
        "assertions_retracted": 0,
    }


@pytest.mark.asyncio
async def test_sleep_marks_success_false_when_enabled_inference_fails() -> None:
    profile = _profile()

    class Storage:
        async def run_semantic_maintenance(self, selected_profile, **kwargs):
            assert selected_profile == profile
            raise RuntimeError("publication failed")

    class Agent(SleepMixin):
        def __init__(self) -> None:
            self.semantic_inference_profile = profile
            self.storage = Storage()
            self.sleep_hooks = None
            self.on_sleep_complete = None

        async def _consolidate_memories(self):
            return {"episodes_created": 1}

    report = await Agent().sleep(skip_export=True, skip_reflection=True)

    assert report.success is False
    assert report.error == "semantic_maintenance_failed"
    assert report.semantic_inference == {
        "status": "failed",
        "reason": "semantic_maintenance_failed",
    }
    assert report.semantic_maintenance == report.semantic_inference


@pytest.mark.asyncio
async def test_sleep_explicit_disabled_inference_revokes_prior_materialization() -> None:
    class Storage:
        async def revoke_semantic_inference(self):
            return SimpleNamespace(
                retracted_assertions=4,
                deactivated_derivations=7,
                generation=12,
            )

    class Agent(SleepMixin):
        def __init__(self) -> None:
            self.semantic_inference_profile = None
            self.semantic_inference_configured = True
            self.storage = Storage()

    report = await Agent().sleep(
        skip_consolidation=True,
        skip_export=True,
        skip_reflection=True,
    )

    assert report.semantic_inference == {
        "status": "disabled",
        "retracted_assertions": 4,
        "deactivated_derivations": 7,
        "generation": 12,
    }


@pytest.mark.asyncio
async def test_sleep_reports_revocation_failure_as_failed_maintenance() -> None:
    class Storage:
        async def revoke_semantic_inference(self):
            raise RuntimeError("storage failure")

    class Agent(SleepMixin):
        def __init__(self) -> None:
            self.semantic_inference_profile = None
            self.semantic_inference_configured = True
            self.storage = Storage()

    report = await Agent().sleep(
        skip_consolidation=True,
        skip_export=True,
        skip_reflection=True,
    )

    assert report.success is False
    assert report.semantic_inference == {
        "status": "failed",
        "reason": "semantic_inference_revocation_failed",
    }
    assert report.semantic_maintenance == report.semantic_inference


@pytest.mark.asyncio
async def test_budget_exhaustion_is_durable_and_does_not_claim_closure(assertion_store) -> None:
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    class_c = IRI("https://example.test/ClassC")
    subject = IRI("https://example.test/subject")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "b-sub-c", class_b, RDFS_SUBCLASS, class_c)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)

    service = BoundedInferenceService(
        assertion_store, _profile(),
        limits=InferenceLimits(max_generated_assertions=1),
    )
    result = await service.materialize_incremental()
    assert result.status is ClosureStatus.INCOMPLETE
    assert result.incomplete_reason == "generated_assertions"
    state = await service.closure_state()
    assert state is not None and not state.complete
    assert await assertion_store.query(AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_c)) == []


@pytest.mark.asyncio
async def test_publication_failure_records_terminal_failed_state(assertion_store) -> None:
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    subject = IRI("https://example.test/subject")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)

    service = BoundedInferenceService(assertion_store, _profile())

    async def fail_ledger_replacement(*_args) -> None:
        raise RuntimeError("ledger write failed")

    service._replace_active_derivations = fail_ledger_replacement  # type: ignore[method-assign]
    with pytest.raises(TransactionError):
        await service.materialize_incremental()

    state = await service.closure_state()
    assert state is not None
    assert state.status is ClosureStatus.FAILED
    run = await assertion_store._database.fetchone(  # noqa: SLF001 - durable ledger assertion
        "SELECT status, completed_at FROM semantic_inference_runs "
        "WHERE tenant_id = ? AND run_id = ?",
        (assertion_store.tenant_id, state.run_id),
    )
    assert run is not None
    assert run[0] == ClosureStatus.FAILED.value
    assert run[1] is not None
    assert await assertion_store.query(
        AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
    ) == []


@pytest.mark.asyncio
async def test_publication_time_budget_is_recorded_incomplete_after_rollback(
    assertion_store,
) -> None:
    """A budget error wrapped by the publication transaction is not a failure."""
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    subject = IRI("https://example.test/subject")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)

    service = BoundedInferenceService(
        assertion_store,
        _profile(),
        limits=InferenceLimits(max_wall_time_seconds=60),
    )
    original_replace = service._replace_active_derivations

    async def replace_after_time_limit(facts, run_id, started) -> None:
        # The ledger method checks the supplied start time after it has begun
        # publication.  The transaction wrapper turns that _BudgetExceeded
        # into TransactionError, which must still surface as INCOMPLETE.
        await original_replace(facts, run_id, started - 61)

    service._replace_active_derivations = replace_after_time_limit  # type: ignore[method-assign]
    result = await service.materialize_incremental()

    assert result.status is ClosureStatus.INCOMPLETE
    assert result.incomplete_reason == "wall_time"
    state = await service.closure_state()
    assert state is not None
    assert state.status is ClosureStatus.INCOMPLETE
    assert state.incomplete_reason == "wall_time"
    assert await assertion_store.query(
        AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
    ) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        (InferenceLimits(max_source_assertions=1), "source_assertions"),
        (InferenceLimits(max_iterations=1), "iterations"),
        (InferenceLimits(max_memory_items=1), "memory"),
        (InferenceLimits(max_wall_time_seconds=1e-12), "wall_time"),
    ],
)
async def test_every_non_generation_budget_is_visible(assertion_store, limits, reason) -> None:
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    class_c = IRI("https://example.test/ClassC")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "b-sub-c", class_b, RDFS_SUBCLASS, class_c)

    result = await BoundedInferenceService(assertion_store, _profile(), limits=limits).materialize_incremental()
    assert result.status is ClosureStatus.INCOMPLETE
    assert result.incomplete_reason == reason


@pytest.mark.asyncio
async def test_allowlisted_owl_rules_materialize_without_same_as(assertion_store) -> None:
    subject = IRI("https://example.test/subject")
    object_ = IRI("https://example.test/object")
    terminal = IRI("https://example.test/terminal")
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    property_p = IRI("https://example.test/p")
    property_q = IRI("https://example.test/q")
    property_s = IRI("https://example.test/symmetric")
    property_t = IRI("https://example.test/transitive")
    property_first = IRI("https://example.test/first")
    property_second = IRI("https://example.test/second")
    property_chain = IRI("https://example.test/chain")
    owl_equivalent_class = IRI("http://www.w3.org/2002/07/owl#equivalentClass")
    owl_equivalent_property = IRI("http://www.w3.org/2002/07/owl#equivalentProperty")
    owl_inverse_of = IRI("http://www.w3.org/2002/07/owl#inverseOf")
    owl_symmetric = IRI("http://www.w3.org/2002/07/owl#SymmetricProperty")
    owl_transitive = IRI("http://www.w3.org/2002/07/owl#TransitiveProperty")
    owl_chain = IRI("http://www.w3.org/2002/07/owl#propertyChainAxiom")
    owl_same_as = IRI("http://www.w3.org/2002/07/owl#sameAs")

    await _put(assertion_store, "eq-class", class_a, owl_equivalent_class, class_b)
    await _put(assertion_store, "typed", subject, RDF_TYPE, class_a)
    await _put(assertion_store, "eq-property", property_p, owl_equivalent_property, property_q)
    await _put(assertion_store, "property-statement", subject, property_p, object_)
    await _put(assertion_store, "inverse", property_p, owl_inverse_of, property_q)
    await _put(assertion_store, "symmetric-type", property_s, RDF_TYPE, owl_symmetric)
    await _put(assertion_store, "symmetric-statement", subject, property_s, object_)
    await _put(assertion_store, "transitive-type", property_t, RDF_TYPE, owl_transitive)
    await _put(assertion_store, "transitive-left", subject, property_t, object_)
    await _put(assertion_store, "transitive-right", object_, property_t, terminal)
    await _put(assertion_store, "chain-axiom", property_chain, owl_chain, Literal(
        '["https://example.test/first","https://example.test/second"]', XSD_STRING
    ))
    await _put(assertion_store, "chain-first", subject, property_first, object_)
    await _put(assertion_store, "chain-second", object_, property_second, terminal)
    await _put(assertion_store, "forbidden-same-as", subject, owl_same_as, terminal)

    result = await BoundedInferenceService(assertion_store, _profile(owl=True)).materialize_incremental()
    assert result.complete
    for expected_subject, expected_predicate, expected_object in (
        (subject, RDF_TYPE, class_b),
        (subject, property_q, object_),
        (object_, property_q, subject),
        (object_, property_s, subject),
        (subject, property_t, terminal),
        (subject, property_chain, terminal),
    ):
        assert await assertion_store.query(AssertionQuery(
            subject=expected_subject, predicate=expected_predicate, object=expected_object
        ))
    # owl:sameAs is explicitly excluded from the selected profile and does
    # not grant a substitute subject/object identity inference path.
    assert await assertion_store.query(AssertionQuery(subject=terminal, predicate=owl_same_as, object=subject)) == []


@pytest.mark.asyncio
async def test_status_and_lineage_survive_sqlite_restart(tmp_path) -> None:
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Semantic restart test"
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(credentials.agent_did, identity)
    database_path = str(tmp_path / "semantic.db")
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    subject = IRI("https://example.test/subject")
    profile = _profile()

    first = AsyncStorage(database_path, agent_id=credentials.agent_did, _assertion_tenant_capability=capability)
    await first.initialize()
    try:
        store = first._assertion_store()
        _GOVERNED_STORAGES[id(store)] = first
        await _put(store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
        await _put(store, "subject-a", subject, RDF_TYPE, class_a)
        result = await first.materialize_semantic_inference(profile)
        inferred = (await first.query_assertions(AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)))[0]
    finally:
        _GOVERNED_STORAGES.pop(id(store), None)
        await first.close()

    restarted = AsyncStorage(database_path, agent_id=credentials.agent_did, _assertion_tenant_capability=capability)
    await restarted.initialize()
    try:
        state = await restarted.semantic_inference_state(profile)
        explanations = await restarted.explain_semantic_inference(inferred.assertion_id, profile)
        assert state is not None and state.complete
        assert state.source_generation == result.checkpoint_generation
        assert explanations and explanations[0].premise_revision_ids
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_ontology_version_change_invalidates_prior_materialization(assertion_store) -> None:
    class_a = IRI("https://example.test/ClassA")
    class_b = IRI("https://example.test/ClassB")
    subject = IRI("https://example.test/subject")
    await _put(assertion_store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
    await _put(assertion_store, "subject-a", subject, RDF_TYPE, class_a)
    first = BoundedInferenceService(assertion_store, _profile())
    assert (await first.materialize_incremental()).complete
    assert await assertion_store.query(AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b))

    newer_ontology = OntologyRef(
        "https://kestrel.ai/vocab/",
        "1.1.0",
        "2d14444f4f42fd8beda98f8da5b052e44652a624755943a0a6e7927fef395ebb",
        "semantic-kb-v1",
    )
    newer_profile = InferenceProfile(newer_ontology, "1.0.0")
    assert (await BoundedInferenceService(assertion_store, newer_profile).materialize_incremental()).complete
    assert await assertion_store.query(AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)) == []


@pytest.mark.asyncio
async def test_profile_change_reconciles_obsolete_proofs_before_maintenance_completes(
    assertion_store,
) -> None:
    """A new profile cannot leave facts proven only by the old profile active."""
    class_a = IRI("https://example.test/ProfileClassA")
    class_b = IRI("https://example.test/ProfileClassB")
    subject = IRI("https://example.test/profile-subject")
    owl_equivalent_class = IRI("http://www.w3.org/2002/07/owl#equivalentClass")
    await _put(assertion_store, "profile-a-equivalent-b", class_a, owl_equivalent_class, class_b)
    await _put(assertion_store, "profile-subject-a", subject, RDF_TYPE, class_a)
    assert (
        await BoundedInferenceService(
            assertion_store, _profile(owl=True)
        ).materialize_incremental()
    ).complete
    assert await assertion_store.query(
        AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
    )

    maintenance = SemanticMaintenanceService(
        assertion_store,
        inference_profile=_profile(),
        limits=SemanticMaintenanceLimits(max_assertions=10, max_derivations=1),
    )

    first = await maintenance.run()
    assert first.status is SemanticMaintenanceStatus.PARTIAL, first.reason
    assert first.reason in {"derivation_budget", "repair_change_replay"}
    for _ in range(4):
        if await assertion_store.query(
            AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)
        ) == []:
            break
        follow_up = await maintenance.run()
        assert follow_up.status is SemanticMaintenanceStatus.PARTIAL
    else:
        pytest.fail("obsolete profile conclusion remained active after reconciliation pages")
    for _ in range(10):
        final = await maintenance.run()
        if final.status in (
            SemanticMaintenanceStatus.COMPLETE,
            SemanticMaintenanceStatus.NO_OP,
        ):
            break
    else:
        pytest.fail("profile-change maintenance did not drain its replay")
    proofs = await assertion_store._database.fetchval(  # noqa: SLF001 - proof retirement contract
        "SELECT COUNT(*) FROM semantic_inference_derivations "
        "WHERE tenant_id = ? AND active = 1",
        (assertion_store.tenant_id,),
    )
    assert proofs == 0
    assert (await maintenance.training_readiness()).ready
