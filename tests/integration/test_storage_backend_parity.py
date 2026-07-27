"""SQLite/PostgreSQL semantic parity contracts for storage seams."""

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kestrel_sovereign.endpoints.database import (
    _get_table_columns,
    _list_table_names,
    list_database_tables,
    query_database_table,
)
from kestrel_sovereign.a2a.stores.unified import ObservabilityStore, TaskStore
from kestrel_sovereign.a2a.types import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from kestrel_sovereign.features.webhooks.feature import WebhookFeature
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.associative_linker import AssociativeLinker
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.db.interface import QueryError
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
from kestrel_sovereign.storage.schema_router import SchemaRouter
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.identity.runtime_identity import (
    AgentIdentity,
    load_agent_identity,
)
from kestrel_sovereign.inception_service import create_kestrel_identity_async


def _semantic_source(source_id: str):
    from kestrel_sovereign.knowledge import SourceOccurrence

    return SourceOccurrence(
        source_occurrence_id=source_id,
        source_kind="parity-test",
        locator=f"parity:{source_id}",
        received_at="2026-07-26T14:02:11Z",
        content_digest="sha256:parity",
        actor="test",
        selector="record",
    )


def _semantic_assertion(tenant_id: str, revision_id: str, *, value: str = "value"):
    from kestrel_sovereign.knowledge import (
        Assertion,
        DirectLineage,
        EpistemicState,
        IRI,
        Literal,
        OntologyRef,
        XSD_STRING,
    )

    return Assertion(
        tenant_id=tenant_id,
        owning_agent_id=tenant_id,
        subject=IRI(f"urn:kestrel:agent:{tenant_id}:principal:user"),
        predicate=IRI("https://kestrel.ai/vocab/parity"),
        object=Literal(value, XSD_STRING),
        revision_id=revision_id,
        confidence="1",
        confidence_method="test",
        confidence_basis="parity",
        epistemic_state=EpistemicState.REPORTED,
        asserted_at="2026-07-26T14:02:11Z",
        ontology_version=OntologyRef("parity", "1", "sha256:parity", "semantic-kb-v1"),
        lineage=DirectLineage(("parity-source",)),
        privacy_classification="normal",
        release_policy_reference="policy:private-v1",
    )


def _derived_semantic_assertion(tenant_id: str, revision_id: str, input_revision_id: str, marker: str):
    from kestrel_sovereign.knowledge import (
        Assertion,
        DerivedLineage,
        EpistemicState,
        IRI,
        Literal,
        OntologyRef,
        XSD_STRING,
    )

    return Assertion(
        tenant_id=tenant_id,
        owning_agent_id=tenant_id,
        subject=IRI(f"urn:kestrel:agent:{tenant_id}:principal:user"),
        predicate=IRI(f"https://kestrel.ai/vocab/parityDerived/{marker}"),
        object=Literal("true", XSD_STRING),
        revision_id=revision_id,
        confidence="1",
        confidence_method="rule",
        confidence_basis="parity",
        epistemic_state=EpistemicState.INFERRED,
        asserted_at="2026-07-26T14:02:12Z",
        ontology_version=OntologyRef("parity", "1", "sha256:parity", "semantic-kb-v1"),
        lineage=DerivedLineage(
            rule_id=f"parity-{marker}", engine_version="1", profile_version="1",
            input_revision_ids=(input_revision_id,), input_digest="sha256:parity-inputs",
            run_id=f"parity-{marker}", generated_at="2026-07-26T14:02:12Z",
        ),
        privacy_classification="normal",
        release_policy_reference="policy:private-v1",
    )


async def _incepted_assertion_identity(
    identity_dir,
    label: str,
) -> tuple[str, AgentIdentity]:
    """Create and load a real identity for semantic authority parity tests."""
    credentials = await create_kestrel_identity_async(
        str(identity_dir / label),
        identity_method="did:pkh",
        agent_name=f"Semantic parity {label}",
    )
    tenant_id = credentials.agent_did
    key_id = f"kestrel_{tenant_id.rsplit(':', 1)[-1]}"
    return tenant_id, load_agent_identity(key_id, identity_dir / label)


async def _assertion_storage_for_backend(
    db_backend,
    tenant_id: str,
    identity: AgentIdentity,
) -> AsyncStorage:
    """Open a boot-resolver-authorized assertion store for parity coverage."""
    capability = _resolve_authenticated_agent_assertion_capability(
        tenant_id,
        identity,
    )
    if db_backend.backend_type == "sqlite":
        storage = AsyncStorage(
            db_backend.db_path,
            agent_id=tenant_id,
            _assertion_tenant_capability=capability,
        )
        await storage.initialize()
        return storage
    dsn = getattr(db_backend, "_dsn", None)
    if not dsn:
        raise RuntimeError("PostgreSQL parity backend did not expose its test DSN")
    storage = AsyncStorage(
        backend="postgres",
        dsn=dsn,
        agent_id=tenant_id,
        _assertion_tenant_capability=capability,
    )
    await storage.initialize()
    return storage


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_constitution_runtime_state_round_trips_on_both_backends(db_backend):
    """Safe Mode and UTC audit deadlines survive the SQLite/Postgres codecs."""
    from kestrel_sovereign.constitution.runtime_state import (
        ConstitutionRuntimeState,
        ConstitutionRuntimeStateStore,
    )

    agent_id = f"did:test:{uuid4()}"
    entered_at = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    last_audit_at = entered_at - timedelta(hours=2)
    state = ConstitutionRuntimeState(
        agent_id=agent_id,
        safe_mode=True,
        safe_mode_reason="backend parity probe",
        safe_mode_entered_at=entered_at,
        safe_mode_exited_at=None,
        safe_mode_exit_authorization=None,
        last_successful_audit_at=last_audit_at,
        interaction_count=37,
        updated_at=entered_at,
        bootstrap_pending=True,
    )

    writer = ConstitutionRuntimeStateStore(db_backend)
    await writer.initialize()
    await writer.write(
        state,
        event_type="safe_mode_entered",
        event_reason=state.safe_mode_reason,
    )

    # A new store instance represents the restart-side reader while reusing the
    # fixture-managed connection/pool. On CI the parametrized Postgres case runs
    # against the pgvector service and exercises asyncpg's datetime codec.
    reader = ConstitutionRuntimeStateStore(db_backend)
    await reader.initialize()
    restored = await reader.load(agent_id)

    assert restored is not None
    assert restored.safe_mode is True
    assert restored.safe_mode_reason == state.safe_mode_reason
    assert restored.safe_mode_entered_at == entered_at
    assert restored.last_successful_audit_at == last_audit_at
    assert restored.interaction_count == 37
    assert restored.bootstrap_pending is True
    assert restored.safe_mode_entered_at.tzinfo == timezone.utc
    assert restored.last_successful_audit_at.tzinfo == timezone.utc


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_canonical_assertion_store_has_tenant_and_lifecycle_parity(
    db_backend,
    tmp_path,
):
    """The normalized assertion authority has the same observable contract on SQLite and PostgreSQL."""
    tenant, identity = await _incepted_assertion_identity(tmp_path, "primary")
    other_tenant, other_identity = await _incepted_assertion_identity(tmp_path, "foreign")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    other_storage = await _assertion_storage_for_backend(
        db_backend,
        other_tenant,
        other_identity,
    )
    try:
        store = storage
        assertion = _semantic_assertion(tenant, "parity-revision")
        written = await store.put_assertion(assertion, source_occurrences=(_semantic_source("parity-source"),))

        assert await store.get_assertion(assertion.assertion_id) == assertion
        assert (await store.assertion_changes_since(0))[0].event_id == written.event_id
        assert await other_storage.get_assertion(assertion.assertion_id) is None

        retracted = await store.retract_assertion(assertion.assertion_id, assertion.revision_id)
        assert len(retracted.retracted) == 1
        assert await store.get_assertion(assertion.assertion_id) is None
        assert await storage.db.fetchval(
            "SELECT eligible FROM semantic_projection_eligibility WHERE tenant_id = ? AND revision_id = ?",
            (tenant, assertion.revision_id),
        ) == 0
        with pytest.raises(QueryError):
            await storage.db.execute(
                "INSERT INTO semantic_projection_outbox "
                "(event_id, tenant_id, assertion_id, revision_id, operation, generation, eligible, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("invalid-parity-event", tenant, "assertion", "revision", "accepted", 0, 2, "2026-07-26T14:02:11Z"),
            )
    finally:
        await other_storage.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_save_fact_adapter_has_canonical_create_retry_supersede_delete_restart_parity(
    db_backend,
    tmp_path,
):
    """The explicit teaching tool never needs a learned_fact graph row as truth."""
    from kestrel_sovereign.features.memory_agency.semantic_facts import GovernedFactAdapter

    tenant, identity = await _incepted_assertion_identity(tmp_path, "save-fact-adapter")
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        adapter = GovernedFactAdapter(storage)
        first = await adapter.save(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="http-invoke-nonce",
        )
        replay = await adapter.save(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="http-invoke-nonce",
        )
        replacement = await adapter.save(
            subject="user",
            predicate="preferred_deploy_region",
            value="europe-west4",
            confidence=0.9,
            invocation_id="http-invoke-replacement",
        )
        replacement_replay = await adapter.save(
            subject="user",
            predicate="preferred_deploy_region",
            value="europe-west4",
            confidence=0.9,
            invocation_id="http-invoke-replacement",
        )

        assert first.saved is True
        assert replay.saved is True
        assert replay.idempotent is True
        assert replay.assertion_id == first.assertion_id
        assert replacement.saved is True
        assert replacement.superseded_assertion_id == first.assertion_id
        assert replacement.assertion_id != first.assertion_id
        assert replacement.provenance_reference is not None
        assert replacement.provenance_digest is not None
        assert replacement_replay.saved is True
        assert replacement_replay.idempotent is True
        assert replacement_replay.assertion_id == replacement.assertion_id
        assert await raw_storage.get_nodes_by_type("learned_fact") == []

        deleted = await adapter.forget(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="http-invoke-delete",
        )
        assert deleted.deleted is True
        assert await storage.get_assertion(replacement.assertion_id) is None
        deleted_replay = await adapter.forget(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="http-invoke-delete",
        )
        assert deleted_replay.deleted is True
        assert deleted_replay.idempotent is True
        assert deleted_replay.assertion_id == replacement.assertion_id

        # Historical deleted shells must not make a later, distinct explicit
        # fact ambiguous.  Its delete retry selects the original canonical
        # operation by request identity and active predecessor revision.
        after_delete = await adapter.save(
            subject="user",
            predicate="preferred_deploy_region",
            value="asia-south1",
            confidence=0.9,
            invocation_id="http-invoke-after-delete",
        )
        after_delete_removed = await adapter.forget(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="http-invoke-after-delete-remove",
        )
        after_delete_replay = await adapter.forget(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="http-invoke-after-delete-remove",
        )
        assert after_delete.saved is True
        assert after_delete_removed.deleted is True
        assert after_delete_replay.deleted is True
        assert after_delete_replay.idempotent is True
        assert after_delete_replay.assertion_id == after_delete.assertion_id
    finally:
        await raw_storage.close()

    restarted_raw = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        restarted = PrivacyEnforcingStorage(restarted_raw, PrivacyMode.NORMAL)
        assert await restarted.get_assertion(replacement.assertion_id) is None
        assert await restarted_raw.get_nodes_by_type("learned_fact") == []
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_save_fact_concurrent_retry_replays_first_delivery_provenance(
    db_backend,
    tmp_path,
):
    """One retry ID has one canonical receipt despite distinct delivery clocks."""
    from kestrel_sovereign.agent.invocation import invocation_scope, request_provenance
    from kestrel_sovereign.features.memory_agency.semantic_facts import GovernedFactAdapter

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-concurrent-retry",
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        adapter = GovernedFactAdapter(storage)

        async def deliver(received_at: str):
            provenance = request_provenance(
                actor="parity-user",
                source_kind="http_request",
                source_locator="POST:/api/agent/invoke",
                received_at=received_at,
            )
            with invocation_scope("concurrent-retry-2765", provenance=provenance):
                return await adapter.save(
                    subject="user",
                    predicate="preferred_deploy_region",
                    value="us-central1",
                    confidence=0.9,
                    invocation_id="concurrent-retry-2765",
                )

        first, second = await asyncio.gather(
            deliver("2026-07-26T14:02:11Z"),
            deliver("2026-07-26T14:02:12Z"),
        )

        assert first.saved is True
        assert second.saved is True
        assert first.assertion_id == second.assertion_id
        assert first.revision_id == second.revision_id
        assert first.provenance_reference == second.provenance_reference
        assert {first.idempotent, second.idempotent} == {False, True}

        sources = await raw_storage.list_assertion_sources(first.assertion_id)
        assert len(sources) == 1
        assert sources[0].source_occurrence_id == first.provenance_reference
        assert sources[0].received_at.value in {
            "2026-07-26T14:02:11Z",
            "2026-07-26T14:02:12Z",
        }
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_canonical_assertion_iri_object_query_has_backend_parity(
    db_backend,
    tmp_path,
):
    """IRI-object lookups do not bind untyped NULLs on PostgreSQL."""
    from kestrel_sovereign.knowledge import AssertionQuery, IRI

    tenant, identity = await _incepted_assertion_identity(tmp_path, "iri-object-query")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        object_ = IRI("https://example.test/object")
        assertion = replace(
            _semantic_assertion(tenant, "iri-object-query"),
            object=object_,
            assertion_id=None,
        )
        await storage.put_assertion(
            assertion,
            source_occurrences=(_semantic_source("parity-source"),),
        )

        assert await storage.query_assertions(
            AssertionQuery(object=object_)
        ) == [assertion]
        assert await storage.query_assertions(
            AssertionQuery(object=IRI("https://example.test/missing"))
        ) == []
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_derived_assertion_lifecycle_parity(db_backend, tmp_path):
    """All lifecycle transitions preserve derived lineage on both backends."""
    tenant, identity = await _incepted_assertion_identity(tmp_path, "lifecycle")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        store = storage

        async def write_pair(marker: str):
            root = _semantic_assertion(tenant, f"{marker}-root", value=marker)
            await store.put_assertion(root, source_occurrences=(_semantic_source("parity-source"),))
            child = _derived_semantic_assertion(tenant, f"{marker}-derived", root.revision_id, marker)
            await store.put_assertion(child)
            return root, child

        superseded_root, superseded_child = await write_pair("supersede")
        replacement = _semantic_assertion(tenant, "supersede-replacement", value="replacement")
        supersession = await store.supersede_assertion(
            superseded_root.revision_id,
            replacement,
            source_occurrences=(_semantic_source("parity-source"),),
        )
        assert superseded_child.revision_id in supersession.invalidated_revision_ids
        assert await store.get_assertion(superseded_child.assertion_id) is None
        assert (await store.list_assertion_revisions(superseded_child.assertion_id))[-1].status.value == "retracted"

        retracted_root, retracted_child = await write_pair("retract")
        retraction = await store.retract_assertion(retracted_root.assertion_id, retracted_root.revision_id)
        assert {item.assertion_id for item in retraction.retracted} == {
            retracted_root.assertion_id,
            retracted_child.assertion_id,
        }
        assert (await store.list_assertion_revisions(retracted_child.assertion_id))[-1].status.value == "retracted"

        deleted_root, deleted_child = await write_pair("delete")
        deletion = await store.delete_assertion(deleted_root.assertion_id, deleted_root.revision_id)
        assert deletion.invalidated_revision_ids == (
            deleted_root.revision_id,
            deleted_child.revision_id,
        )
        assert [item.assertion_id for item in deletion.invalidated] == [deleted_child.assertion_id]
        assert (await store.list_assertion_revisions(deleted_child.assertion_id))[-1].status.value == "retracted"
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_semantic_inference_ledger_retracts_invalid_proofs_on_both_backends(
    db_backend,
    tmp_path,
):
    """PostgreSQL and SQLite deactivate stale alternate-proof rows identically."""
    from kestrel_sovereign.knowledge import (
        Assertion,
        AssertionQuery,
        BoundedInferenceService,
        DirectLineage,
        EpistemicState,
        InferenceProfile,
        IRI,
        OntologyRef,
        SourceOccurrence,
    )

    tenant, identity = await _incepted_assertion_identity(tmp_path, "inference-ledger")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    ontology = OntologyRef(
        "http://www.w3.org/2000/01/rdf-schema#",
        "1.0.0",
        "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
        "semantic-kb-v1",
    )
    profile = InferenceProfile(ontology, "1.0.0")
    subproperty = IRI("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")

    def semantic_assertion(revision_id, subject, predicate, object_):
        source_id = f"inference-source:{revision_id}"
        return Assertion(
            tenant_id=tenant,
            owning_agent_id=tenant,
            subject=subject,
            predicate=predicate,
            object=object_,
            revision_id=revision_id,
            confidence="1",
            confidence_method="parity",
            confidence_basis="parity",
            epistemic_state=EpistemicState.ASSERTED,
            asserted_at="2026-07-26T14:02:11Z",
            ontology_version=ontology,
            lineage=DirectLineage((source_id,)),
            privacy_classification="normal",
            release_policy_reference="policy:private-v1",
        )

    async def put(revision_id, subject, predicate, object_):
        assertion = semantic_assertion(revision_id, subject, predicate, object_)
        await storage.put_assertion(
            assertion,
            source_occurrences=(
                SourceOccurrence(
                    source_occurrence_id=f"inference-source:{revision_id}",
                    source_kind="parity-test",
                    locator=f"parity:{revision_id}",
                    received_at="2026-07-26T14:02:11Z",
                ),
            ),
        )
        return assertion

    try:
        subject = IRI("https://example.test/subject")
        object_ = IRI("https://example.test/object")
        property_p = IRI("https://example.test/p")
        property_q = IRI("https://example.test/q")
        property_r = IRI("https://example.test/r")
        direct_path = await put("p-sub-q", property_p, subproperty, property_q)
        await put("p-sub-r", property_p, subproperty, property_r)
        await put("r-sub-q", property_r, subproperty, property_q)
        await put("statement", subject, property_p, object_)

        service = BoundedInferenceService(storage._assertion_store(), profile)
        assert (await service.materialize_incremental()).complete
        conclusion = (
            await storage.query_assertions(
                AssertionQuery(subject=subject, predicate=property_q, object=object_)
            )
        )[0]
        await storage.delete_assertion(direct_path.assertion_id, direct_path.revision_id)

        retained = await storage.get_assertion(conclusion.assertion_id)
        assert retained is not None
        explanations = await service.explain(conclusion.assertion_id)
        assert explanations
        assert all(
            direct_path.revision_id not in explanation.premise_revision_ids
            for explanation in explanations
        )
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_inference_derivations d "
            "JOIN semantic_inference_derivation_inputs i "
            "  ON i.tenant_id = d.tenant_id AND i.derivation_id = d.derivation_id "
            "WHERE d.tenant_id = ? AND d.active = 1 AND i.input_revision_id = ?",
            (tenant, direct_path.revision_id),
        ) == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_erasure_scrubs_historical_derived_lineage_on_both_backends(
    db_backend,
    tmp_path,
):
    """Erasure removes stale links from current and historical direct revisions."""
    from kestrel_sovereign.knowledge import Assertion, DirectLineage, EpistemicState

    tenant, identity = await _incepted_assertion_identity(tmp_path, "historical-lineage")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        root = _semantic_assertion(tenant, "historical-root", value="historical-root")
        await storage.put_assertion(
            root,
            source_occurrences=(_semantic_source("parity-source"),),
        )
        derived = _derived_semantic_assertion(
            tenant,
            "historical-derived",
            root.revision_id,
            "historical",
        )
        await storage.put_assertion(derived)

        replacement_mapping = derived.to_mapping()
        replacement_mapping["revision_id"] = "historical-direct-replacement"
        replacement_mapping["lineage"] = DirectLineage(
            ("historical-direct-source",)
        ).to_mapping()
        replacement_mapping["epistemic_state"] = EpistemicState.REPORTED.value
        replacement = Assertion.from_mapping(replacement_mapping)
        assert replacement.assertion_id == derived.assertion_id
        supersession = await storage.supersede_assertion(
            derived.revision_id,
            replacement,
            source_occurrences=(_semantic_source("historical-direct-source"),),
        )

        second_mapping = replacement.to_mapping()
        second_mapping["revision_id"] = "historical-direct-second-replacement"
        second_mapping["lineage"] = DirectLineage(
            ("historical-direct-second-source",)
        ).to_mapping()
        second_replacement = Assertion.from_mapping(second_mapping)
        second_supersession = await storage.supersede_assertion(
            supersession.replacement.revision_id,
            second_replacement,
            source_occurrences=(_semantic_source("historical-direct-second-source"),),
        )

        erased = await storage.erase_assertion(root.assertion_id)

        assert derived.assertion_id not in erased.erased_assertion_ids
        assert derived.revision_id in erased.erased_revision_ids
        assert supersession.predecessor.revision_id in erased.erased_revision_ids
        surviving = await storage.get_assertion(derived.assertion_id)
        assert surviving is not None
        assert surviving.revision_id == second_supersession.replacement.revision_id
        assert surviving.supersedes_revision_id == second_supersession.predecessor.revision_id
        historical_row = await storage.db.fetchone(
            "SELECT supersedes_revision_id, assertion_mapping "
            "FROM semantic_assertion_revisions WHERE tenant_id = ? AND revision_id = ?",
            (tenant, supersession.replacement.revision_id),
        )
        assert historical_row is not None
        assert historical_row[0] is None
        assert Assertion.from_mapping(
            json.loads(historical_row[1])
        ).supersedes_revision_id is None
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_derivation_inputs WHERE tenant_id = ?",
            (tenant,),
        ) == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_erasure_emits_an_opaque_retryable_change_on_both_backends(
    db_backend,
    tmp_path,
):
    """An incremental consumer can resynchronize after identity-free erasure."""
    tenant, identity = await _incepted_assertion_identity(tmp_path, "erasure")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        store = storage
        assertion = _semantic_assertion(tenant, "erasure-revision", value="erasure")
        written = await store.put_assertion(
            assertion,
            source_occurrences=(_semantic_source("parity-source"),),
        )

        erased = await store.erase_assertion(assertion.assertion_id, operation_id="parity-erasure")
        replay = await store.erase_assertion(assertion.assertion_id, operation_id="parity-erasure")
        assert replay.idempotent is True
        assert replay.erased_assertion_ids == erased.erased_assertion_ids
        assert replay.erased_revision_ids == erased.erased_revision_ids
        first_read = await store.assertion_changes_since(written.generation)
        retry_read = await store.assertion_changes_since(written.generation)

        assert first_read == retry_read
        assert len(first_read) == 1
        change = first_read[0]
        assert change.operation == "erased"
        assert change.assertion_id is None
        assert change.revision_id is None
        assert change.eligible is False
        assert change.generation == erased.generation
        assert (await store.assertion_checkpoint()).latest_event_id == change.event_id
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_projection_outbox WHERE tenant_id = ?",
            (tenant,),
        ) == 0
        assert await storage.db.fetchall(
            "SELECT operation, generation FROM semantic_projection_erasure_outbox WHERE tenant_id = ?",
            (tenant,),
        ) == [("erased", erased.generation)]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_shacl_reports_and_governed_write_are_backend_neutral(db_backend, tmp_path):
    """Pinned reports and their assertion links round-trip on SQLite and PostgreSQL."""
    tenant, identity = await _incepted_assertion_identity(tmp_path, "shacl-report")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        assertion = _semantic_assertion(tenant, "shacl-report-revision", value="validated")
        result = await storage.put_validated_assertion(
            assertion,
            source_occurrences=(_semantic_source("parity-source"),),
        )

        assert result.accepted is True
        assert result.report.conforms is True
        reports = await storage.semantic_validation_service().reports.list(
            assertion_id=assertion.assertion_id
        )
        assert reports == [result.report]
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_validation_report_assertions "
            "WHERE tenant_id = ? AND assertion_id = ?",
            (tenant, assertion.assertion_id),
        ) == 1
        await storage.erase_assertion(assertion.assertion_id)
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_validation_reports WHERE tenant_id = ?",
            (tenant,),
        ) == 0
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_validation_results WHERE tenant_id = ?",
            (tenant,),
        ) == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_concurrent_postgres_initializers_serialize_semantic_migration(db_backend):
    """A shared PostgreSQL fleet must not race assertion-schema DDL at boot."""
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL-specific concurrent schema regression")
    dsn = getattr(db_backend, "_dsn", None)
    if not dsn:
        raise RuntimeError("PostgreSQL parity backend did not expose its test DSN")

    storages = [
        AsyncStorage(backend="postgres", dsn=dsn, agent_id=f"did:test:{uuid4()}")
        for _ in range(4)
    ]
    try:
        await asyncio.gather(*(storage.initialize() for storage in storages))
        marker_rows = await db_backend.fetch_all(
            "SELECT version FROM semantic_schema_migrations "
            "WHERE version = ?",
            ("semantic_assertion_store_v3",),
        )
        assert marker_rows == [("semantic_assertion_store_v3",)]
    finally:
        await asyncio.gather(*(storage.close() for storage in storages))


def _project_rows(rows):
    return [(row[1], row[2]) for row in rows]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_conversation_session_queries_are_backend_neutral(db_backend):
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    agent_id = f"did:test:{uuid4()}"
    other_agent_id = f"did:test:{uuid4()}"
    start = datetime(2026, 4, 16, 12, 0, 0)

    await storage.db.execute_many(
        """
        INSERT INTO conversation_history (agent_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                agent_id,
                "system",
                "[New conversation started]",
                '{"new_session": true, "type": "session_marker"}',
                start,
            ),
            (agent_id, "user", "hello", '{"topic": "parity"}', start + timedelta(minutes=1)),
            (agent_id, "assistant", "hi there", "{}", start + timedelta(minutes=2)),
            (other_agent_id, "user", "not yours", "{}", start + timedelta(minutes=3)),
        ],
    )

    inserted = await storage.db.fetchall(
        """
        SELECT id, role, content, metadata, created_at
        FROM conversation_history
        WHERE agent_id = ?
        ORDER BY created_at ASC
        """,
        (agent_id,),
    )
    session_id = str(inserted[0][0])

    listed = await privacy_storage.query_conversations(agent_id, limit=10)
    assert _project_rows(listed) == [
        ("assistant", "hi there"),
        ("user", "hello"),
        ("system", "[New conversation started]"),
    ]

    start_row = await privacy_storage.query_conversation_start(session_id, agent_id)
    assert start_row is not None

    session_rows = await privacy_storage.query_conversation_messages(
        agent_id,
        start_row[0],
        limit=10,
    )
    assert _project_rows(session_rows) == [
        ("system", "[New conversation started]"),
        ("user", "hello"),
        ("assistant", "hi there"),
    ]

    other_rows = await privacy_storage.query_conversation_messages(
        other_agent_id,
        start_row[0],
        limit=10,
    )
    assert _project_rows(other_rows) == [("user", "not yours")]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a2a_task_store_filters_and_payloads_are_backend_neutral(db_backend):
    store = TaskStore(db_backend)
    await store.initialize()

    session_a = f"session-{uuid4()}"
    session_b = f"session-{uuid4()}"
    user_a = f"user-{uuid4()}"
    user_b = f"user-{uuid4()}"
    task_a = f"task-{uuid4()}"
    task_b = f"task-{uuid4()}"
    other_task = f"task-{uuid4()}"

    await store.save(
        Task(
            id=task_a,
            sessionId=session_a,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[Message(role="user", parts=[TextPart(text="build the thing")])],
            metadata={"task_type": "audit", "user_id": user_a, "marker": "first"},
        )
    )
    await store.save(
        Task(
            id=task_b,
            sessionId=session_a,
            status=TaskStatus(state=TaskState.COMPLETED),
            metadata={"task_type": "audit", "user_id": user_a, "marker": "second"},
        )
    )
    await store.save(
        Task(
            id=other_task,
            sessionId=session_b,
            status=TaskStatus(state=TaskState.SUBMITTED),
            metadata={"task_type": "audit", "user_id": user_b, "marker": "other"},
        )
    )

    await store.update_status(
        task_a,
        TaskStatus(
            state=TaskState.WORKING,
            message=Message(role="agent", parts=[TextPart(text="underway")]),
        ),
    )
    await store.add_artifact(
        task_a,
        Artifact(
            name="result.txt",
            parts=[TextPart(text="semantic parity")],
        ),
    )

    retrieved = await store.get(task_a)
    assert retrieved is not None
    assert retrieved.sessionId == session_a
    assert retrieved.status.state == TaskState.WORKING
    assert retrieved.status.message is not None
    assert retrieved.status.message.parts[0].text == "underway"
    assert retrieved.history is not None
    assert retrieved.history[0].parts[0].text == "build the thing"
    assert retrieved.artifacts is not None
    assert retrieved.artifacts[0].parts[0].text == "semantic parity"
    assert retrieved.metadata == {"task_type": "audit", "user_id": user_a, "marker": "first"}

    session_tasks = await store.list_tasks(session_id=session_a, user_id=user_a, limit=10)
    assert {task.id for task in session_tasks} == {task_a, task_b}
    assert {task.metadata["marker"] for task in session_tasks} == {"first", "second"}

    working_tasks = await store.list_tasks(user_id=user_a, status=TaskState.WORKING, limit=10)
    assert [task.id for task in working_tasks] == [task_a]

    assert await store.delete(task_a) is True
    assert await store.get(task_a) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_database_introspection_helpers_are_backend_neutral(db_backend):
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()

    table_names = await _list_table_names(storage.db)
    assert "conversation_history" in table_names
    assert "graph_nodes" in table_names

    columns = await _get_table_columns(storage.db, "conversation_history")
    by_name = {column["name"]: column for column in columns}

    assert by_name["id"]["pk"] is True
    assert by_name["agent_id"]["nullable"] is False
    assert by_name["role"]["nullable"] is False
    assert by_name["content"]["type"]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_webhook_registration_and_audit_history_are_backend_neutral(db_backend):
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()

    agent_id = f"did:test:{uuid4()}"
    agent = SimpleNamespace(
        did=agent_id,
        agent_id=agent_id,
        storage=SimpleNamespace(db=storage.db),
        _raw_storage=None,
        features=[],
    )
    feature = WebhookFeature(agent)
    await feature.initialize()

    # Webhook tools migrated to ToolResult (#1061 wave 26).
    from kestrel_sdk.tools.result import ToolResultStatus
    webhook_name = f"audit-{uuid4().hex}"
    registered = await feature.webhooks_register(
        name=webhook_name,
        auth_type="none",
        event_type="sync",
        rate_limit=0,
        allow_unauthenticated=True,  # #1677: acknowledge the open endpoint
    )
    assert registered.status is ToolResultStatus.OK

    listed = await feature.webhooks_list()
    assert listed.data["count"] == 1
    assert listed.data["webhooks"][0]["name"] == webhook_name

    await feature.log_webhook_event(
        webhook_name=webhook_name,
        source_ip="127.0.0.1",
        authenticated=True,
        status_code=200,
        payload_hash="abc123",
    )

    history = await feature.webhooks_history(limit=5)
    assert history.data["count"] == 1
    assert history.data["events"][0]["webhook_name"] == webhook_name
    assert history.data["events"][0]["authenticated"] is True
    assert history.data["events"][0]["payload_hash"] == "abc123"


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_db_explorer_scopes_rows_to_requesting_agent(db_backend):
    """#1651: the /api/db/tables explorer must only return the requesting
    agent's rows for agent-scoped tables, never another agent's data in a
    shared multi-agent database — and the scope must survive a free-text
    search."""
    agent_id = f"did:test:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    other_agent_id = f"did:test:{uuid4()}"
    start = datetime(2026, 4, 16, 12, 0, 0)

    await storage.db.execute_many(
        """
        INSERT INTO conversation_history (agent_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (agent_id, "user", "mine-1", "{}", start),
            (agent_id, "assistant", "mine-2", "{}", start + timedelta(minutes=1)),
            (other_agent_id, "user", "NOT-YOURS", "{}", start + timedelta(minutes=2)),
        ],
    )

    agent = SimpleNamespace(agent_id=agent_id, storage=privacy_storage)
    request = SimpleNamespace(state=SimpleNamespace(agent=agent))

    result = await query_database_table(
        request, "conversation_history", limit=50, offset=0, search=None
    )
    contents = {r["content"] for r in result["rows"]}
    assert contents == {"mine-1", "mine-2"}
    assert result["total_rows"] == 2
    assert all(r["agent_id"] == agent_id for r in result["rows"])

    # The agent scope must AND with search — the other agent's "NOT-YOURS"
    # matches the term but must stay invisible.
    searched = await query_database_table(
        request, "conversation_history", limit=50, offset=0, search="YOURS"
    )
    assert searched["rows"] == []

    # list_database_tables row counts are scoped too.
    listing = await list_database_tables(request)
    conv = next(t for t in listing["tables"] if t["name"] == "conversation_history")
    assert conv["row_count"] == 2


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_rag_chunks_are_scoped_through_file_ownership(db_backend):
    """All backend-neutral RAG reads enforce the document capability."""
    agent_a = f"did:test:{uuid4()}"
    agent_b = f"did:test:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_a)
    await storage.initialize()
    files_b = AsyncFileStore(storage.db, agent_id=agent_b)
    rag_b = AsyncRAGStore(storage.db, agent_id=agent_b)

    hash_a = await storage.files.store_file(b"alpha", "alpha.txt")
    hash_b = await files_b.store_file(b"bravo", "bravo.txt")
    await storage.rag.chunk_document(
        hash_a, "alpha-backend-private", compute_embeddings=False
    )
    await rag_b.chunk_document(
        hash_b, "bravo-backend-private", compute_embeddings=False
    )

    assert await storage.rag.get_chunks_for_file(hash_b) == []
    assert await storage.rag._search_by_like("bravo-backend-private") == []
    assert await rag_b.get_chunks_for_file(hash_b) == ["bravo-backend-private"]
    with pytest.raises(ValueError, match="outside the bound agent"):
        await storage.rag.chunk_document(
            hash_b, "unauthorized", compute_embeddings=False
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_saved_item_and_rag_bodies_are_currently_plaintext_on_both_backends(
    db_backend,
):
    """Characterize the PostgreSQL seam as well as the default SQLite path.

    Child A/C/D intentionally replace these assertions. Until then, this proves
    that the shared production writers and backend-normalized schemas expose
    both bodies directly to a database reader.
    """
    agent_id = f"did:test:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()

    saved_sentinel = f"saved-plaintext-{uuid4()}"
    saved = await SavedItemsStore(storage.db, agent_id=agent_id).save_item(
        item_type="stash",
        name="Backend plaintext characterization",
        content=saved_sentinel,
        compute_embedding=False,
        deduplicate=False,
    )
    saved_row = await storage.db.fetchone(
        "SELECT content FROM saved_items WHERE id = ? AND agent_id = ?",
        (saved.id, agent_id),
    )

    rag_sentinel = f"rag-plaintext-{uuid4()}"
    file_hash = await storage.files.store_file(
        b"memory encryption characterization",
        "memory-encryption-characterization.txt",
    )
    await storage.rag.chunk_document(
        file_hash,
        rag_sentinel,
        chunk_size=1000,
        compute_embeddings=False,
    )
    rag_row = await storage.db.fetchone(
        "SELECT content FROM document_chunks WHERE file_hash = ?",
        (file_hash,),
    )

    saved_columns = {
        column["name"]
        for column in await _get_table_columns(storage.db, "saved_items")
    }
    rag_columns = {
        column["name"]
        for column in await _get_table_columns(storage.db, "document_chunks")
    }
    assert saved_row is not None and saved_row[0] == saved_sentinel
    assert rag_row is not None and rag_row[0] == rag_sentinel
    assert "content_ciphertext" not in saved_columns
    assert "content_ciphertext" not in rag_columns


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_db_explorer_hides_agent_rows_in_ephemeral_mode(db_backend):
    """#1651: for agent-scoped tables, EPHEMERAL/ISOLATED modes must not
    surface persisted rows through the raw explorer."""
    agent_id = f"did:test:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    await storage.db.execute_many(
        """
        INSERT INTO conversation_history (agent_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(agent_id, "user", "secret", "{}", datetime(2026, 4, 16, 12, 0, 0))],
    )

    privacy_storage.set_privacy_mode(PrivacyMode.EPHEMERAL)
    agent = SimpleNamespace(agent_id=agent_id, storage=privacy_storage)
    request = SimpleNamespace(state=SimpleNamespace(agent=agent))

    result = await query_database_table(
        request, "conversation_history", limit=50, offset=0, search=None
    )
    assert result["rows"] == []
    assert result["total_rows"] == 0
    assert "privacy mode" in result.get("note", "").lower()

    # The listing must not reveal the row exists via its count either.
    listing = await list_database_tables(request)
    conv = next(t for t in listing["tables"] if t["name"] == "conversation_history")
    assert conv["row_count"] == 0


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_db_explorer_scopes_graph_nodes_by_canonical_ownership(db_backend):
    """Graph ownership includes the untagged canonical root and tagged nodes."""
    agent_id = f"did:test:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    other_agent_id = f"did:test:{uuid4()}"

    await storage.db.execute_many(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, ?, ?, ?)",
        [
            (agent_id, "agent", "my-root", "{}"),
            (f"n-{uuid4()}", "concept", "mine", json.dumps({"agent_id": agent_id})),
            (f"n-{uuid4()}", "concept", "also-mine", json.dumps({"agent_id": agent_id})),
            (other_agent_id, "agent", "their-root", "{}"),
            (f"n-{uuid4()}", "concept", "theirs", json.dumps({"agent_id": other_agent_id})),
        ],
    )
    rows = await storage.db.fetchall(
        "SELECT node_id, properties FROM graph_nodes "
        "WHERE label IN ('my-root', 'mine', 'also-mine', 'their-root', 'theirs')"
    )
    owners = []
    for node_id, properties in rows:
        parsed = json.loads(properties or "{}")
        owners.append((node_id, parsed.get("agent_id") or node_id))
    await storage.db.execute_many(
        "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
        owners,
    )

    agent = SimpleNamespace(agent_id=agent_id, storage=privacy_storage)
    request = SimpleNamespace(state=SimpleNamespace(agent=agent))

    result = await query_database_table(
        request, "graph_nodes", limit=50, offset=0, search=None
    )
    labels = {r["label"] for r in result["rows"]}
    assert labels == {"my-root", "mine", "also-mine"}
    assert result["total_rows"] == 3


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_db_explorer_keeps_associative_edges_and_excludes_cross_tenant_edge(
    db_backend,
):
    """#2649: production AssociativeLinker edges retain direct ownership.

    This exercises the production writer that historically omitted message
    ownership.  Its own ``mentions`` edges must remain visible while an edge
    from that message to another tenant's node stays unowned and invisible.
    """
    agent_id = f"did:test:{uuid4()}"
    other = f"did:test:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    await storage.graph.add_node(
        GraphNode(agent_id, "agent", "my-root", {})
    )
    linker = AssociativeLinker(storage.graph)
    linked = await linker.extract_and_link(
        "message-a", "I called mom before work", agent_id
    )
    assert linked
    stored_message = await storage.graph.get_node(
        f"message:{agent_id}:message-a"
    )
    assert stored_message.properties["agent_id"] == agent_id

    other_graph = AsyncGraphStore(storage.db, agent_id=other)
    await other_graph.add_node(GraphNode(other, "agent", "their-root", {}))
    foreign_node = f"concept:{other}:foreign"
    await other_graph.add_node(
        GraphNode(
            foreign_node,
            "concept",
            "their-private-concept",
            {"agent_id": other},
        )
    )

    message_node = f"message:{agent_id}:message-a"
    await storage.db.execute(
        "INSERT INTO graph_edges (source_id, target_id, label, properties) "
        "VALUES (?, ?, ?, ?)",
        (
            message_node,
            foreign_node,
            "cross_tenant",
            json.dumps({"secret": "foreign-edge"}),
        ),
    )

    agent = SimpleNamespace(agent_id=agent_id, storage=privacy_storage)
    request = SimpleNamespace(state=SimpleNamespace(agent=agent))

    result = await query_database_table(
        request, "graph_edges", limit=50, offset=0, search=None
    )
    labels = {r["label"] for r in result["rows"]}
    assert "mentions" in labels
    assert "cross_tenant" not in labels
    assert result["total_rows"] >= len(linked)
    assert other not in repr(result)
    assert foreign_node not in repr(result)
    assert "foreign-edge" not in repr(result)

    # Scope must AND with search: the stored secret matches but its unowned
    # cross-tenant edge remains excluded, proving scope params precede search.
    searched = await query_database_table(
        request, "graph_edges", limit=50, offset=0, search="foreign-edge"
    )
    assert searched["rows"] == []
    assert searched["total_rows"] == 0


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_action_routing_without_concepts_keeps_message_edge_atomic(db_backend):
    """A concept-free commitment still has an owned message source node."""
    agent_id = f"did:test:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    await storage.graph.add_node(GraphNode(agent_id, "agent", "root", {}))
    linker = AssociativeLinker(storage.graph)
    router = SchemaRouter(storage.graph, storage.db, agent_id)
    message_id = "concept-free-action"

    concepts = await linker.extract_and_link(
        message_id, "I need to buy milk.", agent_id
    )
    assert concepts == []
    summary = await router.route(
        message_id,
        "I need to buy milk.",
        concepts,
        role="user",
    )

    assert summary["action_items"] == 1
    message_node = f"message:{agent_id}:{message_id}"
    assert await storage.graph.get_node(message_node) is not None
    action_nodes = await storage.graph.get_nodes_by_type("action_item")
    assert len(action_nodes) == 1
    edges = await storage.graph.get_edges(message_node, direction="out")
    assert [(edge.target_id, edge.label) for edge in edges] == [
        (action_nodes[0].node_id, "records_action")
    ]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_metric_summary_is_backend_neutral(db_backend):
    """#969: get_metric_summary filters metric_name via each backend's native
    JSON accessor (json_extract on SQLite, ->> on Postgres-JSONB). This proves
    the predicate is valid on both — a LIKE on JSONB would 500 on Postgres."""
    store = ObservabilityStore(db_backend)
    await store.initialize()

    await store.log_metric(
        agent_name="did:test:emma",
        metric_name="assistant_turn_persist_failed",
        metric_value=1.0,
        metadata={"session_id": "s-1", "error_type": "TimeoutError"},
    )
    # A newer, higher-volume different metric must not crowd out the rare one.
    for i in range(5):
        await store.log_metric(
            agent_name="did:test:emma",
            metric_name="feature_tools_built_streaming",
            metric_value=float(i),
            metadata={},
        )

    rare = await store.get_metric_summary("assistant_turn_persist_failed", limit=3)
    assert rare["count"] == 1
    assert rare["by_agent"] == {"did:test:emma": 1}
    assert rare["last_seen"] is not None
    assert rare["samples"][0]["metadata"].get("error_type") == "TimeoutError"

    other = await store.get_metric_summary("feature_tools_built_streaming")
    assert other["count"] == 5

    missing = await store.get_metric_summary("never_emitted")
    assert missing["count"] == 0
    assert missing["samples"] == []
