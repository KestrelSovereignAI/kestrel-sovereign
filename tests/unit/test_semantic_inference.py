"""Bounded semantic materialization contracts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.agent.sleep import SleepMixin
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
    InferenceLimits,
    InferenceProfile,
    Literal,
    OntologyRef,
    SourceOccurrence,
    XSD_STRING,
    inference_profile_from_config,
)
from kestrel_sovereign.knowledge.inference import ENGINE_VERSION
from kestrel_sovereign.storage.async_assertion_store import AssertionConflictError
from kestrel_sovereign.storage.db import TransactionError
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage


RDF_TYPE = IRI("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDFS_SUBCLASS = IRI("http://www.w3.org/2000/01/rdf-schema#subClassOf")
RDFS_SUBPROPERTY = IRI("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")
ONTOLOGY = OntologyRef("kestrel-test", "1", "sha256:test", "semantic-kb-v1")


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
        yield storage._assertion_store()  # Exercise the same bound facade materializers receive.
    finally:
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
    await store.put_assertion(assertion, source_occurrences=(_source(f"source:{revision_id}"),))
    return assertion


def _profile(*, owl: bool = False) -> InferenceProfile:
    return InferenceProfile(ONTOLOGY, "1.0.0", "1.0.0" if owl else None)


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
            )
        )
    )

    agent = KestrelAgent(
        did="did:test:semantic-config",
        storage_path=str(agent_dir / "kestrel_prime.db"),
    )

    assert agent.semantic_inference_profile == _profile()
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
    assert await assertion_store.get_assertion(direct.assertion_id) is not None


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

    async def persist_while_direct_write_arrives(facts, run_id):
        nonlocal direct_write
        if direct_write is None:
            direct_write = asyncio.create_task(
                _put(assertion_store, "b-sub-c", class_b, RDFS_SUBCLASS, class_c)
            )
            # Let the writer reach the tenant lock while the materializer is
            # publishing.  It must wait until the complete checkpoint commits.
            await asyncio.sleep(0)
        return await original_persist(facts, run_id)

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

    class Storage:
        def __init__(self) -> None:
            self.profiles = []

        async def materialize_semantic_inference(self, selected_profile):
            self.profiles.append(selected_profile)
            return SimpleNamespace(
                status=ClosureStatus.COMPLETE,
                incomplete_reason=None,
                source_generation=3,
                checkpoint_generation=3,
                generated_assertions=0,
                retracted_assertions=0,
            )

    class Agent(SleepMixin):
        def __init__(self) -> None:
            self.semantic_inference_profile = profile
            self.storage = Storage()

    agent = Agent()
    report = await agent.sleep(
        skip_consolidation=True,
        skip_export=True,
        skip_reflection=True,
    )
    assert agent.storage.profiles == [profile]
    assert report.semantic_inference == {
        "status": "complete",
        "incomplete_reason": None,
        "source_generation": 3,
        "checkpoint_generation": 3,
        "generated_assertions": 0,
        "retracted_assertions": 0,
    }


@pytest.mark.asyncio
async def test_sleep_marks_success_false_when_enabled_inference_fails() -> None:
    profile = _profile()

    class Storage:
        async def materialize_semantic_inference(self, selected_profile):
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
    assert report.error == "semantic_inference_failed"
    assert report.semantic_inference == {
        "status": "failed",
        "reason": "semantic_inference_failed",
    }


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
        await _put(store, "a-sub-b", class_a, RDFS_SUBCLASS, class_b)
        await _put(store, "subject-a", subject, RDF_TYPE, class_a)
        result = await first.materialize_semantic_inference(profile)
        inferred = (await first.query_assertions(AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)))[0]
    finally:
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

    newer_ontology = OntologyRef("kestrel-test", "2", "sha256:new", "semantic-kb-v1")
    newer_profile = InferenceProfile(newer_ontology, "1.0.0")
    assert (await BoundedInferenceService(assertion_store, newer_profile).materialize_incremental()).complete
    assert await assertion_store.query(AssertionQuery(subject=subject, predicate=RDF_TYPE, object=class_b)) == []
