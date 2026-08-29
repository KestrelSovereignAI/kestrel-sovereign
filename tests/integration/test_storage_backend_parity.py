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
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage, PrivacyViolationError
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
from kestrel_sovereign.storage.schema_router import SchemaRouter
from kestrel_sovereign.storage.sqla.migrations import (
    _SEMANTIC_MAINTENANCE_LEASE_PRECISION_SCHEMA_VERSION,
    migrate_semantic_maintenance,
)
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.identity.runtime_identity import (
    AgentIdentity,
    load_agent_identity,
)
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.knowledge import InferenceProfile, OntologyRef
from kestrel_sovereign.knowledge.capabilities import semantic_capabilities_from_config


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_governed_artifact_erasure_lifecycle_has_backend_parity(
    db_backend, tmp_path,
):
    """SQLite/PostgreSQL share authenticated registration, expiry, and ACK semantics."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from kestrel_sovereign.knowledge import (
        GovernedArtifactConsumerAuthentication,
        GovernedArtifactDeletionOwner,
        GovernedArtifactDeletionProof,
    )

    tenant, identity = await _incepted_assertion_identity(tmp_path, "artifact-parity")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    private_key = Ed25519PrivateKey.generate()
    unique = uuid4().hex
    assertion = None
    try:
        assertion = _semantic_assertion(
            tenant, f"artifact-parity-revision-{unique}",
            value=f"artifact-parity-{unique}",
            source_id=f"artifact-parity-source-{unique}",
        )
        await storage.put_assertion(
            assertion,
            source_occurrences=(_semantic_source(f"artifact-parity-source-{unique}"),),
        )
        checkpoint = await storage.assertion_checkpoint()
        artifact_id = str(uuid4())
        produced_checkpoint, produced = await storage.export_assertion_snapshot(
            artifact_id=artifact_id,
            consumer_id="parity-consumer",
            consumer_key_id="parity-key",
            consumer_public_key=private_key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw,
            ).hex(),
            retention_seconds=300,
        )
        assert produced_checkpoint.generation == checkpoint.generation
        assert assertion.revision_id in {item.revision_id for item in produced}
        storage._artifact_clock = lambda: datetime.now(timezone.utc) + timedelta(minutes=10)
        storage._assertion_store()._artifact_clock = storage._artifact_clock
        assert await storage.sweep_expired_governed_semantic_artifacts() == 1
        authentication = GovernedArtifactConsumerAuthentication(
            "parity-consumer", "parity-key", str(uuid4()),
            datetime.now(timezone.utc).isoformat(), "0" * 128,
        )
        authentication = replace(
            authentication,
            signature=private_key.sign(authentication.signable_bytes(tenant)).hex(),
        )
        deleted: list[str] = []

        async def delete_artifact(lease):
            deleted.append(lease.artifact_key)
            proof = GovernedArtifactDeletionProof(
                datetime.now(timezone.utc).isoformat(), "0" * 128,
            )
            return replace(
                proof, signature=private_key.sign(proof.signable_bytes(lease)).hex(),
            )

        receipt = await storage.process_governed_semantic_artifact_revocation(
            authentication,
            GovernedArtifactDeletionOwner("parity-consumer", "parity-key", delete_artifact),
        )
        assert receipt is not None
        assert deleted == [receipt.artifact_key]
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_governed_artifact_lineage WHERE tenant_id = ?",
            (tenant,),
        ) == 0
    finally:
        if assertion is not None:
            current = await storage.get_assertion(assertion.assertion_id)
            if current is not None:
                await storage.erase_assertion(
                    assertion.assertion_id,
                    operation_id=f"artifact-parity-cleanup-{unique}",
                )
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_empty_export_and_corpus_expiry_have_backend_parity(
    db_backend, tmp_path,
):
    """Zero-lineage artifacts remain legitimate, revocable, and non-resurrectable."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from kestrel_sovereign.knowledge import (
        EpistemicState,
        GovernedArtifactConsumerAuthentication,
        GovernedArtifactDeletionOwner,
        GovernedArtifactDeletionProof,
        GovernedArtifactError,
        GovernedCorpusPolicy,
        Visibility,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path, "empty-artifact-parity",
    )
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw,
    ).hex()
    consumer = {
        "consumer_id": "empty-artifact-consumer",
        "consumer_key_id": "empty-artifact-key",
        "consumer_public_key": public_key,
        "retention_seconds": 300,
    }
    export_artifact_id = str(uuid4())
    corpus_artifact_id = str(uuid4())
    try:
        maintenance = await storage.run_semantic_maintenance(None)
        assert maintenance.status.value == "no_op"
        capability_versions = await storage.semantic_maintenance_capability_versions(None)
        policy = GovernedCorpusPolicy(
            policy_id="empty-artifact-parity",
            policy_version="1",
            accepted_epistemic_states=(EpistemicState.REPORTED,),
            accepted_visibility=(Visibility.PRIVATE,),
            accepted_privacy_classifications=("normal",),
            accepted_consent_references=("policy:private-v1",),
            accepted_grounding_classes=("parity",),
            accepted_source_kinds=("parity-test",),
            accepted_ontology_pins=(
                OntologyRef("parity", "1", "sha256:parity", "semantic-kb-v1"),
            ),
            accepted_semantic_capability_versions=tuple(capability_versions.items()),
        )
        export_checkpoint, exported = await storage.export_assertion_snapshot(
            artifact_id=export_artifact_id, **consumer,
        )
        corpus = await storage.governed_assertion_corpus_snapshot(
            policy=policy,
            inference_profile=None,
            artifact_id=corpus_artifact_id,
            **consumer,
        )
        assert exported == ()
        assert corpus.examples == ()
        assert export_checkpoint.generation == corpus.checkpoint.generation == 0
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_governed_artifact_lineage WHERE tenant_id = ?",
            (tenant,),
        ) == 0

        storage._artifact_clock = lambda: datetime.now(timezone.utc) + timedelta(minutes=10)
        storage._assertion_store()._artifact_clock = storage._artifact_clock
        assert await storage.sweep_expired_governed_semantic_artifacts() == 2
        pending = await storage.governed_semantic_artifact_erasure_observation(
            expected_generation=0,
        )
        assert (pending.export_snapshots, pending.governed_corpus) == (0, 0)
        assert pending.pending_revocations == 2

        deleted: list[str] = []

        async def delete_artifact(lease):
            deleted.append(lease.artifact_key)
            proof = GovernedArtifactDeletionProof(
                datetime.now(timezone.utc).isoformat(), "0" * 128,
            )
            return replace(
                proof, signature=private_key.sign(proof.signable_bytes(lease)).hex(),
            )

        owner = GovernedArtifactDeletionOwner(
            "empty-artifact-consumer", "empty-artifact-key", delete_artifact,
        )
        for _ in range(2):
            authentication = GovernedArtifactConsumerAuthentication(
                "empty-artifact-consumer", "empty-artifact-key", str(uuid4()),
                datetime.now(timezone.utc).isoformat(), "0" * 128,
            )
            authentication = replace(
                authentication,
                signature=private_key.sign(authentication.signable_bytes(tenant)).hex(),
            )
            assert await storage.process_governed_semantic_artifact_revocation(
                authentication, owner,
            ) is not None
        assert len(deleted) == 2
        completed = await storage.governed_semantic_artifact_erasure_observation(
            expected_generation=0,
        )
        assert (completed.pending_revocations, completed.completed_revocations) == (0, 2)

        with pytest.raises(GovernedArtifactError, match="previously revoked"):
            await storage.export_assertion_snapshot(
                artifact_id=export_artifact_id, **consumer,
            )
        with pytest.raises(GovernedArtifactError, match="previously revoked"):
            await storage.governed_assertion_corpus_snapshot(
                policy=policy,
                inference_profile=None,
                artifact_id=corpus_artifact_id,
                **consumer,
            )
    finally:
        await storage.close()


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


def _semantic_assertion(
    tenant_id: str, revision_id: str, *, value: str = "value", source_id: str = "parity-source",
):
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
        lineage=DirectLineage((source_id,)),
        privacy_classification="normal",
        release_policy_reference="policy:private-v1",
    )


_RECALL_PROFILE = InferenceProfile(
    OntologyRef(
        "http://www.w3.org/2000/01/rdf-schema#", "1.0.0",
        "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
        "semantic-kb-v1",
    ),
    "1.0.0",
)


async def _reach_current_semantic_maintenance(storage: AsyncStorage) -> None:
    """Drive the real bounded maintenance cursor to a durable checkpoint."""
    for _ in range(4):
        result = await storage.run_semantic_maintenance(_RECALL_PROFILE)
        if result.status.value in {"complete", "no_op"}:
            return
    raise AssertionError(f"maintenance did not converge: {result}")


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_experimental_capability_selection_is_backend_neutral_and_non_migrating(
    db_backend,
    tmp_path,
):
    """A draft selection reaches real maintenance without rewriting canonical data."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "experimental-capabilities",
    )
    selected = semantic_capabilities_from_config(
        {
            "mode": "experimental",
            "rdf12": {
                "capability": "rdf-profile:rdf12-cr-20260407-experimental",
                "version": "0.1.0",
            },
            "sparql12": {
                "capability": "query-profile:sparql12-20260605-experimental",
                "version": "0.1.0",
            },
            "shacl12": {
                "capability": "validation-profile:shacl12-core-20260602-experimental",
                "version": "0.1.0",
            },
            "shape_set": {
                "identifier": "kestrel-assertion-shapes-shacl12-experimental",
                "version": "0.1.0",
            },
        }
    )
    storage = await _assertion_storage_for_backend(
        db_backend, tenant, identity, semantic_capabilities=selected
    )
    try:
        before = await storage.query_assertions()
        result = await storage.run_semantic_maintenance(
            None,
            semantic_capabilities=selected,
        )
        after = await storage.query_assertions()

        assert result.status.value == "no_op"
        assert before == after == []
        assert result.capability_versions["semantic_capability_mode"] == "experimental"
        assert result.capability_versions["rdf12_version"] == "0.1.0"
        assert result.capability_versions["sparql12_version"] == "0.1.0"
        assert result.capability_versions["validation_profile_version"] == "0.1.0"
    finally:
        await storage.close()


def _explicit_fact_proposal(
    storage: AsyncStorage,
    *,
    value: str,
    confidence: float,
    invocation_id: str,
):
    """Build the deterministic adapter proposal without crossing its writer."""
    from kestrel_sovereign.features.memory_agency.semantic_facts import (
        _assertion,
        _operation_material,
        _source_for_operation,
        map_legacy_fact,
    )

    binding = storage.semantic_assertion_binding()
    operation_id, digest = _operation_material(
        action="save",
        subject="user",
        predicate="preferred_deploy_region",
        value=value,
        confidence_requested=confidence,
        invocation_id=invocation_id,
    )
    source = _source_for_operation(
        operation_id,
        digest,
        binding.owning_agent_id,
    )
    proposal = _assertion(
        binding=binding,
        mapping=map_legacy_fact(
            "user",
            "preferred_deploy_region",
            value,
            tenant_id=binding.tenant_id,
        ),
        source=source,
        confidence=confidence,
    )
    return operation_id, source, proposal


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
    *,
    semantic_capabilities=None,
    llm_service=None,
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
            semantic_capabilities=semantic_capabilities,
            llm_service=llm_service,
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
        semantic_capabilities=semantic_capabilities,
        llm_service=llm_service,
    )
    await storage.initialize()
    return storage


class _RollbackLeasePrecisionProbe(Exception):
    """Sentinel used to leave the shared PostgreSQL fixture unchanged."""


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_semantic_maintenance_lease_precision_upgrade_is_backend_neutral(
    db_backend,
    tmp_path,
):
    """Legacy PostgreSQL leases upgrade to float8 without changing SQLite."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "maintenance-lease-precision",
    )
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    probe_tenant = f"lease-precision:{uuid4()}"
    # 1,774,000,000 is a multiple of 128. At this epoch, float4 rounds a
    # 60.125-second lease back to the base while float8 preserves the duration.
    epoch_base = 1_774_000_000.0
    expected_expiry = epoch_base + 60.125
    try:
        try:
            async with storage.db.transaction():
                await storage.db.execute(
                    "DELETE FROM semantic_schema_migrations WHERE version = ?",
                    (_SEMANTIC_MAINTENANCE_LEASE_PRECISION_SCHEMA_VERSION,),
                )
                if storage.db.backend_type == "postgres":
                    await storage.db.execute(
                        "ALTER TABLE semantic_maintenance_leases "
                        "ALTER COLUMN expires_at TYPE REAL "
                        "USING expires_at::REAL",
                        (),
                    )
                    assert await storage.db.fetchone(
                        "SELECT udt_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() "
                        "AND table_name = 'semantic_maintenance_leases' "
                        "AND column_name = 'expires_at'",
                        (),
                    ) == ("float4",)
                    await storage.db.execute(
                        "INSERT INTO semantic_maintenance_leases "
                        "(tenant_id, holder_id, fencing_token, expires_at, updated_at) "
                        "VALUES (?, 'legacy-float4', 1, ?, '2026-07-27T00:00:00Z')",
                        (probe_tenant, expected_expiry),
                    )
                    legacy_expiry = float(
                        await storage.db.fetchval(
                            "SELECT expires_at FROM semantic_maintenance_leases "
                            "WHERE tenant_id = ?",
                            (probe_tenant,),
                        )
                    )
                    assert abs(legacy_expiry - expected_expiry) >= 1.0

                await migrate_semantic_maintenance(storage.db)

                marker_count = await storage.db.fetchval(
                    "SELECT COUNT(*) FROM semantic_schema_migrations "
                    "WHERE version = ?",
                    (_SEMANTIC_MAINTENANCE_LEASE_PRECISION_SCHEMA_VERSION,),
                )
                assert marker_count == 1
                if storage.db.backend_type == "postgres":
                    assert await storage.db.fetchone(
                        "SELECT udt_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() "
                        "AND table_name = 'semantic_maintenance_leases' "
                        "AND column_name = 'expires_at'",
                        (),
                    ) == ("float8",)
                else:
                    lease_columns = await storage.db.fetchall(
                        "PRAGMA table_info(semantic_maintenance_leases)",
                        (),
                    )
                    expires_at = next(
                        row for row in lease_columns if row[1] == "expires_at"
                    )
                    assert str(expires_at[2]).upper() == "DOUBLE PRECISION"

                await storage.db.execute(
                    "INSERT INTO semantic_maintenance_leases "
                    "(tenant_id, holder_id, fencing_token, expires_at, updated_at) "
                    "VALUES (?, 'float8', 1, ?, '2026-07-27T00:00:00Z') "
                    "ON CONFLICT(tenant_id) DO UPDATE SET "
                    "holder_id = excluded.holder_id, "
                    "fencing_token = excluded.fencing_token, "
                    "expires_at = excluded.expires_at, "
                    "updated_at = excluded.updated_at",
                    (probe_tenant, expected_expiry),
                )
                stored_expiry = float(
                    await storage.db.fetchval(
                        "SELECT expires_at FROM semantic_maintenance_leases "
                        "/* post-float8-upgrade */ "
                        "WHERE tenant_id = ?",
                        (probe_tenant,),
                    )
                )
                assert stored_expiry == expected_expiry
                assert stored_expiry - epoch_base == 60.125

                # A second startup observes the marker and remains a no-op.
                await migrate_semantic_maintenance(storage.db)
                assert await storage.db.fetchval(
                    "SELECT COUNT(*) FROM semantic_schema_migrations "
                    "WHERE version = ?",
                    (_SEMANTIC_MAINTENANCE_LEASE_PRECISION_SCHEMA_VERSION,),
                ) == 1
                raise _RollbackLeasePrecisionProbe
        except TransactionError as exc:
            if not isinstance(exc.__cause__, _RollbackLeasePrecisionProbe):
                raise
    finally:
        await storage.close()


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
async def test_governed_semantic_recall_storage_seam_has_backend_parity(db_backend, tmp_path):
    """Discovery and final provenance hydration use the same SQL contract."""
    from kestrel_sovereign.storage.async_assertion_store import SemanticRecallUnavailableError

    tenant, identity = await _incepted_assertion_identity(tmp_path, "recall-parity")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        fact = _semantic_assertion(tenant, "recall-current", value="current", source_id="recall-current")
        expired = _semantic_assertion(tenant, "recall-expired", value="expired", source_id="recall-expired")
        ineligible = _semantic_assertion(tenant, "recall-ineligible", value="ineligible", source_id="recall-ineligible")
        await storage.put_assertion(fact, source_occurrences=(_semantic_source("recall-current"),))
        await storage.put_assertion(expired, source_occurrences=(_semantic_source("recall-expired"),))
        await storage.put_assertion(ineligible, source_occurrences=(_semantic_source("recall-ineligible"),))
        await storage.db.execute(
            "UPDATE semantic_assertion_revisions SET valid_end = ? "
            "WHERE tenant_id = ? AND revision_id = ?",
            ("2020-01-01T00:00:00Z", tenant, expired.revision_id),
        )
        await storage.db.execute(
            "UPDATE semantic_projection_eligibility SET eligible = 0 "
            "WHERE tenant_id = ? AND revision_id = ?",
            (tenant, ineligible.revision_id),
        )
        await _reach_current_semantic_maintenance(storage)

        discovered = await storage.semantic_recall_candidates(
            query="current", candidate_scan_limit=10, inference_profile=_RECALL_PROFILE,
        )
        assert [candidate.assertion.assertion_id for candidate in discovered.candidates] == [fact.assertion_id]
        hydrated = await storage.hydrate_semantic_recall_candidates(
            [fact.assertion_id], expected_checkpoint_generation=discovered.checkpoint_generation,
            inference_profile=_RECALL_PROFILE,
        )
        assert hydrated[0].source_occurrences == (_semantic_source("recall-current"),)

        overflow = _semantic_assertion(tenant, "recall-overflow", value="overflow", source_id="recall-overflow")
        await storage.put_assertion(overflow, source_occurrences=(_semantic_source("recall-overflow"),))
        await _reach_current_semantic_maintenance(storage)
        with pytest.raises(SemanticRecallUnavailableError, match="semantic_recall_candidate_window_exceeded"):
            await storage.semantic_recall_candidates(
                query="", candidate_scan_limit=1, inference_profile=_RECALL_PROFILE,
            )
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_assertion_vector_projection_cursor_and_lineage_have_backend_parity(db_backend, tmp_path):
    """Exercise the event CAS and bounded vector read on SQLite/PostgreSQL."""
    from kestrel_sovereign.storage.async_assertion_store import (
        _issue_raw_assertion_mutation_capability,
    )
    from kestrel_sovereign.storage.semantic_vector_projection import SemanticVectorProfile

    async def embed(text: str):
        return [float(len(text)), 1.0]

    profile = SemanticVectorProfile(
        "semantic-vector-parity-v1", "c" * 64,
        provider="parity-provider", model="parity-model", dimension=2,
    )

    class EmbeddingService:
        def describe(self):
            return SimpleNamespace(
                provider=profile.provider, model=profile.model,
                dim=profile.dimension, profile_id=profile.profile_id,
            )

        def semantic_vector_destination(self):
            return profile.embedding_destination

        async def aembed(self, text):
            return await embed(text)

    tenant, identity = await _incepted_assertion_identity(tmp_path, "vector-projection-parity")
    storage = await _assertion_storage_for_backend(
        db_backend, tenant, identity,
        llm_service=SimpleNamespace(get_embedding_service=lambda: EmbeddingService()),
    )
    try:
        run_id = uuid4().hex
        first_id = f"vector-first-{run_id}"
        second_id = f"vector-second-{run_id}"
        first = _semantic_assertion(tenant, first_id, value="first", source_id=first_id)
        second = _semantic_assertion(tenant, second_id, value="second", source_id=second_id)
        # Projection parity is below the governed SHACL acceptance seam; use
        # the already tenant-bound canonical owner so this test isolates SQL
        # cursor/vector behavior from concurrent validation-retry coverage.
        migration_capability = _issue_raw_assertion_mutation_capability(
            storage._assertion_tenant_capability
        )
        await storage._assertion_store().put_assertion(
            first, source_occurrences=(_semantic_source(first_id),),
            _migration_capability=migration_capability,
        )
        await storage._assertion_store().put_assertion(
            second, source_occurrences=(_semantic_source(second_id),),
            _migration_capability=migration_capability,
        )
        projection = storage.semantic_assertion_vector_projection(profile)
        checkpoint = await projection.sync()
        terminal = await storage._assertion_store().event_checkpoint()
        assert (checkpoint.generation, checkpoint.event_id) == (
            terminal.generation, terminal.latest_event_id,
        )
        assert {candidate.revision_id for candidate in await projection.recall([1.0, 1.0])} == {
            first.revision_id, second.revision_id,
        }
        stored = await storage.db.fetchall(
            "SELECT embedding_provider, embedding_model, embedding_dimension, renderer_version, "
            "revision_digest FROM semantic_assertion_vector_projection_entries WHERE tenant_id = ?",
            (tenant,),
        )
        assert len(stored) == 2
        assert all(row[:4] == ("parity-provider", "parity-model", 2, "semantic-recall-claim-v1") for row in stored)
        assert all(len(row[4]) == 64 for row in stored)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_vector_export_erasure_lifecycle_has_backend_parity(db_backend, tmp_path):
    """One physical erase fences vector recall and external artifact serving."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from kestrel_sovereign.knowledge import (
        GovernedArtifactConsumerAuthentication,
        GovernedArtifactDeletionOwner,
        GovernedArtifactDeletionProof,
        GovernedArtifactError,
    )
    from kestrel_sovereign.storage.semantic_vector_projection import (
        SemanticVectorProfile,
        SemanticVectorProjectionError,
    )

    async def embed(text: str):
        return [float(len(text)), 1.0]

    profile = SemanticVectorProfile(
        "vector-artifact-erasure-v1", "d" * 64,
        provider="parity-provider", model="parity-model", dimension=2,
    )

    class EmbeddingService:
        def describe(self):
            return SimpleNamespace(
                provider=profile.provider, model=profile.model,
                dim=profile.dimension, profile_id=profile.profile_id,
            )

        def semantic_vector_destination(self):
            return profile.embedding_destination

        async def aembed(self, text):
            return await embed(text)

    tenant, identity = await _incepted_assertion_identity(
        tmp_path, "vector-artifact-erasure",
    )
    storage = await _assertion_storage_for_backend(
        db_backend, tenant, identity,
        llm_service=SimpleNamespace(
            get_embedding_service=lambda: EmbeddingService(),
        ),
    )
    private_key = Ed25519PrivateKey.generate()
    try:
        unique = uuid4().hex
        assertion = _semantic_assertion(
            tenant, f"vector-artifact-erasure-{unique}",
            value=f"vector-artifact-erasure-{unique}",
            source_id=f"vector-artifact-erasure-source-{unique}",
        )
        await storage.put_assertion(
            assertion,
            source_occurrences=(
                _semantic_source(f"vector-artifact-erasure-source-{unique}"),
            ),
        )
        projection = storage.semantic_assertion_vector_projection(profile)
        checkpoint = await projection.sync()
        artifact_id = str(uuid4())
        _, exported = await storage.export_assertion_snapshot(
            artifact_id=artifact_id,
            consumer_id="vector-artifact-parity-consumer",
            consumer_key_id="vector-artifact-parity-key",
            consumer_public_key=private_key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw,
            ).hex(),
            retention_seconds=300,
        )
        assert [item.revision_id for item in exported] == [assertion.revision_id]
        await storage.consume_governed_semantic_artifact(
            artifact_id, expected_generation=checkpoint.generation,
        )

        erased = await storage.erase_assertion(
            assertion.assertion_id, operation_id=f"vector-artifact-erase-{unique}",
        )
        with pytest.raises(GovernedArtifactError, match="not registered or has been revoked"):
            await storage.consume_governed_semantic_artifact(
                artifact_id, expected_generation=erased.generation,
            )
        pending = await storage.governed_semantic_artifact_erasure_observation(
            expected_generation=erased.generation,
        )
        assert (pending.export_snapshots, pending.governed_corpus, pending.future_corpus) == (0, 0, 0)
        assert (pending.pending_revocations, pending.completed_revocations) == (1, 0)
        with pytest.raises(SemanticVectorProjectionError, match="checkpoint_stale"):
            await projection.recall([1.0, 1.0])

        # The opaque canonical erasure event forces a complete survivor
        # rebuild before the vector projection can become recall-ready again.
        await projection.sync()
        assert await projection.recall([1.0, 1.0]) == ()
        assert (await projection.erasure_observation()).candidate_count == 0

        authentication = GovernedArtifactConsumerAuthentication(
            "vector-artifact-parity-consumer", "vector-artifact-parity-key",
            str(uuid4()), datetime.now(timezone.utc).isoformat(), "0" * 128,
        )
        authentication = replace(
            authentication,
            signature=private_key.sign(authentication.signable_bytes(tenant)).hex(),
        )

        async def delete_artifact(lease):
            proof = GovernedArtifactDeletionProof(
                datetime.now(timezone.utc).isoformat(), "0" * 128,
            )
            return replace(
                proof, signature=private_key.sign(proof.signable_bytes(lease)).hex(),
            )

        receipt = await storage.process_governed_semantic_artifact_revocation(
            authentication,
            GovernedArtifactDeletionOwner(
                "vector-artifact-parity-consumer", "vector-artifact-parity-key",
                delete_artifact,
            ),
        )
        assert receipt is not None
        completed = await storage.governed_semantic_artifact_erasure_observation(
            expected_generation=erased.generation,
        )
        assert (completed.export_snapshots, completed.governed_corpus, completed.future_corpus) == (0, 0, 0)
        assert (completed.pending_revocations, completed.completed_revocations) == (0, 1)
        assert await storage.db.fetchall(
            "SELECT artifact_id FROM semantic_governed_artifacts WHERE tenant_id = ? "
            "UNION ALL SELECT artifact_id FROM semantic_governed_artifact_lineage WHERE tenant_id = ?",
            (tenant, tenant),
        ) == []
        # The remaining acknowledgement record is content-free retry evidence.
        assert await storage.db.fetchall(
            "SELECT artifact_digest, consumer_public_key "
            "FROM semantic_governed_artifact_revocations WHERE tenant_id = ?",
            (tenant,),
        ) == [(None, "")]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_isolated_kite_erasure_drill_uses_real_core_owners(
    db_backend, tmp_path, monkeypatch,
):
    """Storage refuses every direct call; only the loopback typed route runs it."""
    tenant, identity = await _incepted_assertion_identity(tmp_path, "kite-erasure-drill")
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
    try:
        with pytest.raises(PrivacyViolationError, match="unavailable"):
            await storage.semantic_release_erasure_drill(capability=object())
        monkeypatch.setenv("KESTREL_KITE_RELEASE_EVIDENCE", "1")
        evidence_home = tmp_path / "kite-evidence-home"
        evidence_home.mkdir(mode=0o700)
        monkeypatch.setenv("KESTREL_HOME", str(evidence_home))
        monkeypatch.setenv("KESTREL_KITE_RELEASE_EVIDENCE_ROOT", str(evidence_home))
        with pytest.raises(PrivacyViolationError, match="one-shot authority"):
            await storage.semantic_release_erasure_drill(capability=object())

        # Even an endpoint-issued capability cannot escape its route-local
        # task scope and become a lower-level release-evidence shortcut.
        from kestrel_sovereign.knowledge.kite_erasure_authority import (
            _issue_kite_erasure_drill_capability,
            _typed_kite_erasure_endpoint_issuance_scope,
        )
        from kestrel_sovereign.knowledge.kite_evidence_signing import (
            consume_kite_evidence_nonce,
        )

        with _typed_kite_erasure_endpoint_issuance_scope():
            capability = _issue_kite_erasure_drill_capability(
                consume_kite_evidence_nonce("a" * 64, issue_receipt=True),
                operation="erasure_core_snapshot",
            )
        with pytest.raises(PrivacyViolationError, match="one-shot authority"):
            await storage.semantic_release_erasure_drill(capability=capability)
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_save_fact_adapter_has_canonical_create_retry_supersede_delete_restart_parity(
    db_backend,
    tmp_path,
):
    """The explicit teaching tool never needs a learned_fact graph row as truth."""
    tenant, identity = await _incepted_assertion_identity(tmp_path, "save-fact-adapter")
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        first = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="http-invoke-nonce",
        )
        replay = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="http-invoke-nonce",
        )
        assert first.saved is True
        assert replay.saved is True
        assert replay.idempotent is True
        assert replay.assertion_id == first.assertion_id
        same_value_new_invocation = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="http-invoke-second-source",
        )
        same_value_replay = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="http-invoke-second-source",
        )
        assert same_value_new_invocation.saved is True
        assert same_value_new_invocation.idempotent is False
        assert same_value_new_invocation.assertion_id == first.assertion_id
        assert same_value_new_invocation.provenance_reference != first.provenance_reference
        assert same_value_replay.idempotent is True
        assert same_value_replay.provenance_reference == same_value_new_invocation.provenance_reference
        assert len(await storage.list_assertion_sources(first.assertion_id)) == 2
        replacement = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="europe-west4",
            confidence=0.9,
            invocation_id="http-invoke-replacement",
        )
        replacement_replay = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="europe-west4",
            confidence=0.9,
            invocation_id="http-invoke-replacement",
        )
        assert replacement.saved is True
        assert replacement.superseded_assertion_id == first.assertion_id
        assert replacement.assertion_id != first.assertion_id
        assert replacement.provenance_reference is not None
        assert replacement.provenance_digest is not None
        assert replacement_replay.saved is True
        assert replacement_replay.idempotent is True
        assert replacement_replay.assertion_id == replacement.assertion_id
        assert await raw_storage.get_nodes_by_type("learned_fact") == []

        # A recalled answer is a derivative only when the persisted turn
        # carries this exact canonical identity. Same text without the link is
        # intentionally unrelated and must survive governed forget.
        await raw_storage.add_conversation(
            "assistant",
            "kite-2748-region-7f3b",
            metadata={
                "semantic_recall_dependencies": [
                    {
                        "assertion_id": replacement.assertion_id,
                        "revision_id": replacement.revision_id,
                    }
                ]
            },
        )
        await raw_storage.add_conversation("assistant", "kite-2748-region-7f3b")
        recall_rows = await raw_storage.conversation.get_full_history_with_ids(
            include_excluded=True
        )
        linked_row = next(
            row
            for row in recall_rows
            if row["metadata"].get("semantic_recall_dependencies")
        )
        episode_id = f"episode:{tenant}:semantic-recall-derivative"
        await raw_storage.db.execute(
            "INSERT INTO memory_episodes "
            "(id, agent_id, title, summary, key_message_ids, excluded_from_context) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (
                episode_id,
                tenant,
                "semantic recall derivative",
                "must be excluded with its linked turn",
                json.dumps([str(linked_row["id"])]),
            ),
        )

        deleted = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="http-invoke-delete",
        )
        assert deleted.deleted is True
        assert await storage.get_assertion(replacement.assertion_id) is None
        hidden_row = next(
            row
            for row in await raw_storage.conversation.get_full_history_with_ids(
                include_excluded=True
            )
            if row["id"] == linked_row["id"]
        )
        assert hidden_row["metadata"]["excluded_from_context"] is True
        assert (
            await raw_storage.db.fetchone(
                "SELECT excluded_from_context FROM memory_episodes WHERE id = ?",
                (episode_id,),
            )
        ) == (1,)
        visible_same_text = await raw_storage.get_conversation_history()
        assert any(
            row["content"] == "kite-2748-region-7f3b"
            and not row.get("metadata", {}).get("semantic_recall_dependencies")
            for row in visible_same_text
        )
        deleted_replay = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="http-invoke-delete",
        )
        assert deleted_replay.deleted is True
        assert deleted_replay.idempotent is True
        assert deleted_replay.assertion_id == replacement.assertion_id

        # Historical deleted shells must not make a later, distinct explicit
        # fact ambiguous.  Its delete retry replays the original canonical
        # ledger receipt rather than scanning historical active revisions.
        after_delete = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="asia-south1",
            confidence=0.9,
            invocation_id="http-invoke-after-delete",
        )
        after_delete_removed = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="http-invoke-after-delete-remove",
        )
        after_delete_replay = await storage.forget_explicit_fact(
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
async def test_physical_erasure_scrubs_exact_semantic_recall_lineage(
    db_backend, tmp_path
):
    """Permanent erasure leaves the derivative hidden but ID-free."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path, "semantic-recall-physical-erasure"
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        fact = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="kite-2748-region-7f3b",
            confidence=0.9,
            invocation_id="semantic-recall-physical-erasure-save",
        )
        await raw_storage.add_conversation(
            "assistant",
            "kite-2748-region-7f3b",
            metadata={
                "semantic_recall_dependencies": [
                    {
                        "assertion_id": fact.assertion_id,
                        "revision_id": fact.revision_id,
                    }
                ]
            },
        )
        linked_row = next(
            row
            for row in await raw_storage.conversation.get_full_history_with_ids(
                include_excluded=True
            )
            if row["metadata"].get("semantic_recall_dependencies")
        )

        erased = await storage.erase_assertion(
            fact.assertion_id,
            operation_id="semantic-recall-physical-erasure-delete",
        )

        assert fact.assertion_id in erased.erased_assertion_ids
        assert await raw_storage.get_assertion(fact.assertion_id) is None
        hidden_row = next(
            row
            for row in await raw_storage.conversation.get_full_history_with_ids(
                include_excluded=True
            )
            if row["id"] == linked_row["id"]
        )
        assert hidden_row["metadata"]["excluded_from_context"] is True
        assert hidden_row["metadata"]["semantic_recall_dependencies"] == []
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_governed_supersession_excludes_only_predecessor_derivatives(
    db_backend, tmp_path
):
    """A new current revision never revives an old semantic-recall answer."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path, "semantic-recall-supersession"
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        first = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="superseded-region",
            confidence=0.9,
            invocation_id="semantic-recall-supersession-first",
        )
        await raw_storage.add_conversation(
            "assistant",
            "superseded-region",
            metadata={
                "semantic_recall_dependencies": [
                    {
                        "assertion_id": first.assertion_id,
                        "revision_id": first.revision_id,
                    }
                ]
            },
        )
        # The identical but unlinked text is a regression guard against
        # content matching; only the exact old revision may be excluded.
        await raw_storage.add_conversation("assistant", "superseded-region")

        replacement = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="replacement-region",
            confidence=0.9,
            invocation_id="semantic-recall-supersession-replacement",
        )
        await raw_storage.add_conversation(
            "assistant",
            "replacement-region",
            metadata={
                "semantic_recall_dependencies": [
                    {
                        "assertion_id": replacement.assertion_id,
                        "revision_id": replacement.revision_id,
                    }
                ]
            },
        )
        # A same-value teach is a source append: it keeps the canonical
        # assertion identity but advances its revision.  Revocation must stay
        # revision-scoped or it would hide this new current derivative too.
        source_append = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="replacement-region",
            confidence=0.9,
            invocation_id="semantic-recall-supersession-source-append",
        )
        assert source_append.assertion_id == replacement.assertion_id
        assert source_append.revision_id != replacement.revision_id
        await raw_storage.add_conversation(
            "assistant",
            "replacement-region-current",
            metadata={
                "semantic_recall_dependencies": [
                    {
                        "assertion_id": source_append.assertion_id,
                        "revision_id": source_append.revision_id,
                    }
                ]
            },
        )

        rows = await raw_storage.conversation.get_full_history_with_ids(
            include_excluded=True
        )
        old = next(
            row
            for row in rows
            if row["metadata"].get("semantic_recall_dependencies")
            == [{"assertion_id": first.assertion_id, "revision_id": first.revision_id}]
        )
        prior_current = next(
            row
            for row in rows
            if row["metadata"].get("semantic_recall_dependencies")
            == [
                {
                    "assertion_id": replacement.assertion_id,
                    "revision_id": replacement.revision_id,
                }
            ]
        )
        current = next(
            row
            for row in rows
            if row["metadata"].get("semantic_recall_dependencies")
            == [
                {
                    "assertion_id": source_append.assertion_id,
                    "revision_id": source_append.revision_id,
                }
            ]
        )
        assert old["metadata"]["excluded_from_context"] is True
        assert prior_current["metadata"]["excluded_from_context"] is True
        assert current["metadata"].get("excluded_from_context") is not True
        assert any(
            row["content"] == "superseded-region"
            and not row["metadata"].get("semantic_recall_dependencies")
            and row["metadata"].get("excluded_from_context") is not True
            for row in rows
        )
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("operation", ("retract_assertion", "delete_assertion"))
async def test_generic_lifecycle_withdrawal_excludes_exact_derivative(
    db_backend, tmp_path, operation
):
    """Non-adapter governed lifecycle operations share the revocation gate."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path, f"semantic-recall-{operation}"
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        fact = _semantic_assertion(
            tenant,
            f"semantic-recall-{operation}-revision",
            value=f"{operation}-value",
        )
        await raw_storage.put_assertion(
            fact,
            source_occurrences=(_semantic_source("parity-source"),),
        )
        await raw_storage.add_conversation(
            "assistant",
            f"{operation}-value",
            metadata={
                "semantic_recall_dependencies": [
                    {
                        "assertion_id": fact.assertion_id,
                        "revision_id": fact.revision_id,
                    }
                ]
            },
        )
        result = await getattr(storage, operation)(
            fact.assertion_id,
            fact.revision_id,
            operation_id=f"semantic-recall-{operation}",
        )
        assert fact.revision_id in result.invalidated_revision_ids
        linked = next(
            row
            for row in await raw_storage.conversation.get_full_history_with_ids(
                include_excluded=True
            )
            if row["metadata"].get("semantic_recall_dependencies")
        )
        assert linked["metadata"]["excluded_from_context"] is True
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_erasure_scrubs_derivative_of_historical_surviving_identity(
    db_backend, tmp_path
):
    """Erasure closure must match a revision even if its assertion survives."""
    from kestrel_sovereign.knowledge import (
        Assertion,
        DirectLineage,
        EpistemicState,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path, "semantic-recall-historical-erasure"
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        root = _semantic_assertion(tenant, "semantic-recall-erasure-root")
        await raw_storage.put_assertion(
            root, source_occurrences=(_semantic_source("parity-source"),)
        )
        derived = _derived_semantic_assertion(
            tenant,
            "semantic-recall-erasure-derived",
            root.revision_id,
            "semantic-recall-erasure",
        )
        await raw_storage.put_assertion(derived)
        await raw_storage.add_conversation(
            "assistant",
            "historical-derived-answer",
            metadata={
                "semantic_recall_dependencies": [
                    {
                        "assertion_id": derived.assertion_id,
                        "revision_id": derived.revision_id,
                    }
                ]
            },
        )

        replacement_mapping = derived.to_mapping()
        replacement_mapping["revision_id"] = "semantic-recall-erasure-direct"
        replacement_mapping["lineage"] = DirectLineage(
            ("semantic-recall-erasure-source",)
        ).to_mapping()
        replacement_mapping["epistemic_state"] = EpistemicState.REPORTED.value
        replacement = Assertion.from_mapping(replacement_mapping)
        await raw_storage.supersede_assertion(
            derived.revision_id,
            replacement,
            source_occurrences=(
                _semantic_source("semantic-recall-erasure-source"),
            ),
        )

        erased = await PrivacyEnforcingStorage(
            raw_storage, PrivacyMode.NORMAL
        ).erase_assertion(root.assertion_id)
        assert derived.assertion_id not in erased.erased_assertion_ids
        assert derived.revision_id in erased.erased_revision_ids
        linked = next(
            row
            for row in await raw_storage.conversation.get_full_history_with_ids(
                include_excluded=True
            )
            if row["content"] == "historical-derived-answer"
        )
        assert linked["metadata"]["excluded_from_context"] is True
        assert linked["metadata"]["semantic_recall_dependencies"] == []
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_late_normal_and_streaming_derivatives_fail_closed_after_withdrawal(
    db_backend, tmp_path
):
    """Persistence rechecks liveness after a slow response crosses deletion."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path, "semantic-recall-late-persistence"
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        fact = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="late-persistence-region",
            confidence=0.9,
            invocation_id="semantic-recall-late-persistence-save",
        )
        await storage.delete_assertion(
            fact.assertion_id,
            fact.revision_id,
            operation_id="semantic-recall-late-persistence-delete",
        )
        metadata = {
            "semantic_recall_dependencies": [
                {"assertion_id": fact.assertion_id, "revision_id": fact.revision_id}
            ]
        }
        # Normal invoke persists the user+assistant pair; streaming persists
        # the same pair on its terminal path.  Both reach AsyncStorage's
        # persistence fence, so role is deliberately varied here.
        await storage.add_conversation("user", "late normal prompt", metadata=metadata)
        await storage.add_conversation(
            "assistant", "late streaming answer", metadata=metadata
        )
        late_rows = [
            row
            for row in await raw_storage.conversation.get_full_history_with_ids(
                include_excluded=True
            )
            if row["content"] in {"late normal prompt", "late streaming answer"}
        ]
        assert len(late_rows) == 2
        assert all(
            row["metadata"].get("excluded_from_context") is True
            and row["metadata"].get("excluded_reason")
            == "semantic_assertion_not_current"
            for row in late_rows
        )
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_deleted_fact_same_value_reteach_restores_fresh_validated_revision(
    db_backend,
    tmp_path,
):
    """A distinct exact re-teach restores a deleted shell and replays exactly."""
    from kestrel_sovereign.knowledge import AssertionStatus

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-same-value-restoration",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        first = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="same-after-delete",
            confidence=0.9,
            invocation_id="same-after-delete-first",
        )
        deleted = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="same-after-delete-forget",
        )
        deleted_shell = await storage.get_assertion(
            first.assertion_id,
            include_inactive=True,
        )
        assert deleted.deleted is True
        assert deleted_shell.status is AssertionStatus.DELETED

        restored = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="same-after-delete",
            confidence=0.9,
            invocation_id="same-after-delete-restored",
        )
        retry = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="same-after-delete",
            confidence=0.9,
            invocation_id="same-after-delete-restored",
        )

        assert restored.saved is True
        assert restored.idempotent is False
        assert restored.assertion_id == first.assertion_id
        assert restored.revision_id != first.revision_id
        assert restored.provenance_reference != first.provenance_reference
        assert retry.saved is True
        assert retry.idempotent is True
        assert retry.revision_id == restored.revision_id
        assert retry.provenance_reference == restored.provenance_reference

        current = await storage.get_assertion(first.assertion_id)
        restored_source = await storage.get_source_occurrence(
            restored.provenance_reference
        )
        assert current.status is AssertionStatus.ACTIVE
        assert current.supersedes_revision_id == deleted_shell.revision_id
        assert current.asserted_at == restored_source.received_at
        revisions = await storage.list_assertion_revisions(
            first.assertion_id
        )
        assert [item.status for item in revisions] == [
            AssertionStatus.ACTIVE,
            AssertionStatus.DELETED,
            AssertionStatus.ACTIVE,
        ]
        assert len(await storage.list_assertion_sources(first.assertion_id)) == 2
    finally:
        await raw_storage.close()

    restarted_raw = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        restarted = PrivacyEnforcingStorage(
            restarted_raw,
            PrivacyMode.NORMAL,
        )
        restarted_retry = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="same-after-delete",
            confidence=0.9,
            invocation_id="same-after-delete-restored",
        )
        assert restarted_retry.saved is True
        assert restarted_retry.idempotent is True
        assert restarted_retry.revision_id == restored.revision_id
        assert (
            restarted_retry.provenance_reference
            == restored.provenance_reference
        )
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_concurrent_same_value_reteaches_after_delete_preserve_both_sources(
    db_backend,
    tmp_path,
):
    """One restoration and one bounded redecision retain both encounters."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-concurrent-restoration",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        first = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="concurrent-restore-region",
            confidence=0.9,
            invocation_id="concurrent-restore-first",
        )
        await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="concurrent-restore-forget",
        )

        original_restore = raw_storage._restore_explicit_fact_assertion
        both_arrived = asyncio.Event()
        release = asyncio.Event()
        arrivals = 0

        async def synchronized_restore(*args, **kwargs):
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
            await release.wait()
            return await original_restore(*args, **kwargs)

        raw_storage._restore_explicit_fact_assertion = synchronized_restore

        async def reteach(invocation_id):
            return await storage.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="concurrent-restore-region",
                confidence=0.9,
                invocation_id=invocation_id,
            )

        tasks = (
            asyncio.create_task(reteach("concurrent-restore-a")),
            asyncio.create_task(reteach("concurrent-restore-b")),
        )
        await asyncio.wait_for(both_arrived.wait(), timeout=5)
        release.set()
        results = await asyncio.gather(*tasks)

        assert all(result.saved for result in results)
        assert all(not result.idempotent for result in results)
        assert {result.assertion_id for result in results} == {
            first.assertion_id
        }
        assert len(
            {
                result.provenance_reference
                for result in results
            }
        ) == 2
        sources = await storage.list_assertion_sources(first.assertion_id)
        assert len(sources) == 3
        current = await storage.get_assertion(first.assertion_id)
        assert set(current.lineage.source_occurrence_ids) == {
            result.provenance_reference for result in results
        }
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_save_fact_live_replay_rejects_forged_matching_result_ids(
    db_backend,
    tmp_path,
):
    """Exact IDs cannot hide forged immutable assertion metadata."""
    from decimal import Decimal

    from kestrel_sovereign.storage.async_assertion_store import (
        AssertionConflictError,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-live-result-binding",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        operation_id, source, proposal = _explicit_fact_proposal(
            raw_storage,
            value="binding-live",
            confidence=0.9,
            invocation_id="binding-live-invocation",
        )
        forged = replace(
            proposal,
            confidence=Decimal("0.1"),
            confidence_method="forged-confidence",
            confidence_basis="untrusted-preseed",
        )
        await raw_storage.put_assertion(
            forged,
            source_occurrences=(source,),
            operation_id=operation_id,
        )

        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        with pytest.raises(
            AssertionConflictError,
            match="different governed assertion metadata",
        ):
            await storage.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="binding-live",
                confidence=0.9,
                invocation_id="binding-live-invocation",
            )
        current = await storage.get_assertion(proposal.assertion_id)
        assert current.confidence == Decimal("0.1")
        assert current.confidence_method == "forged-confidence"
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_save_fact_erased_replay_rejects_forged_matching_result_ids(
    db_backend,
    tmp_path,
):
    """The blinded erased result key enforces the same proposal binding."""
    from decimal import Decimal

    from kestrel_sovereign.storage.async_assertion_store import (
        AssertionConflictError,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-erased-result-binding",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        operation_id, source, proposal = _explicit_fact_proposal(
            raw_storage,
            value="binding-erased",
            confidence=0.9,
            invocation_id="binding-erased-invocation",
        )
        forged = replace(
            proposal,
            confidence=Decimal("0.1"),
            privacy_classification="forged-private",
            release_policy_reference="policy:forged-v1",
        )
        await raw_storage.put_assertion(
            forged,
            source_occurrences=(source,),
            operation_id=operation_id,
        )
        await raw_storage.erase_assertion(
            proposal.assertion_id,
            operation_id="binding-erased-physical-erasure",
        )
    finally:
        await raw_storage.close()

    restarted_raw = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        restarted = PrivacyEnforcingStorage(
            restarted_raw,
            PrivacyMode.NORMAL,
        )
        with pytest.raises(
            AssertionConflictError,
            match="different erased governed assertion result",
        ):
            await restarted.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="binding-erased",
                confidence=0.9,
                invocation_id="binding-erased-invocation",
            )
        assert await restarted.query_assertions() == []
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_save_fact_replay_rejects_reordered_append_lineage(
    db_backend,
    tmp_path,
):
    """A colliding append must be predecessor lineage plus one new source."""
    from kestrel_sovereign.knowledge import DirectLineage
    from kestrel_sovereign.storage.async_assertion_store import (
        AssertionConflictError,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-reordered-append-binding",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        first = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="lineage-region",
            confidence=0.9,
            invocation_id="lineage-first",
        )
        operation_id, new_source, _ = _explicit_fact_proposal(
            raw_storage,
            value="lineage-region",
            confidence=0.9,
            invocation_id="lineage-reordered",
        )
        current = await storage.get_assertion(first.assertion_id)
        reordered = replace(
            current,
            revision_id=new_source.source_occurrence_id,
            asserted_at=new_source.received_at,
            supersedes_revision_id=None,
            lineage=DirectLineage(
                (
                    new_source.source_occurrence_id,
                    *current.lineage.source_occurrence_ids,
                )
            ),
        )
        await raw_storage.supersede_assertion(
            current.revision_id,
            reordered,
            source_occurrences=(new_source,),
            operation_id=operation_id,
        )

        with pytest.raises(
            AssertionConflictError,
            match="different governed assertion result",
        ):
            await storage.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="lineage-region",
                confidence=0.9,
                invocation_id="lineage-reordered",
            )
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_save_fact_erased_replay_rejects_extra_direct_source(
    db_backend,
    tmp_path,
):
    """Erasure authenticates only the adapter's exact one-source put shape."""
    from kestrel_sovereign.knowledge import DirectLineage, SourceOccurrence
    from kestrel_sovereign.storage.async_assertion_store import (
        AssertionConflictError,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-erased-extra-source-binding",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        operation_id, source, proposal = _explicit_fact_proposal(
            raw_storage,
            value="extra-source-region",
            confidence=0.9,
            invocation_id="extra-source-invocation",
        )
        extra_source = SourceOccurrence(
            source_occurrence_id=f"{source.source_occurrence_id}:extra",
            source_kind=source.source_kind,
            locator=f"{source.locator}:extra",
            received_at=source.received_at,
            content_digest="sha256:extra-source",
            actor=source.actor,
            selector=source.selector,
        )
        forged = replace(
            proposal,
            lineage=DirectLineage(
                (
                    source.source_occurrence_id,
                    extra_source.source_occurrence_id,
                )
            ),
        )
        await raw_storage.put_assertion(
            forged,
            source_occurrences=(source, extra_source),
            operation_id=operation_id,
        )
        await raw_storage.erase_assertion(
            proposal.assertion_id,
            operation_id="extra-source-physical-erasure",
        )
    finally:
        await raw_storage.close()

    restarted_raw = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        restarted = PrivacyEnforcingStorage(
            restarted_raw,
            PrivacyMode.NORMAL,
        )
        with pytest.raises(
            AssertionConflictError,
            match="different erased governed assertion result",
        ):
            await restarted.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="extra-source-region",
                confidence=0.9,
                invocation_id="extra-source-invocation",
            )
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_forget_replay_rejects_unrelated_raw_delete_preseed(
    db_backend,
    tmp_path,
):
    """A future forget operation ID cannot bind an unrelated deletion."""
    from kestrel_sovereign.features.memory_agency.semantic_facts import (
        _operation_material,
    )
    from kestrel_sovereign.storage.async_assertion_store import (
        AssertionConflictError,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "forget-live-selector-binding",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        desired = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="forget-live-desired",
            confidence=0.9,
            invocation_id="forget-live-save",
        )
        unrelated = _semantic_assertion(
            tenant,
            "forget-live-unrelated-revision",
            value="forget-live-unrelated",
        )
        await raw_storage.put_assertion(
            unrelated,
            source_occurrences=(_semantic_source("parity-source"),),
            operation_id="forget-live-unrelated-put",
        )
        forget_operation_id, _ = _operation_material(
            action="forget",
            subject="user",
            predicate="preferred_deploy_region",
            value=None,
            confidence_requested=None,
            invocation_id="forget-live-collision",
        )
        await raw_storage.delete_assertion(
            unrelated.assertion_id,
            unrelated.revision_id,
            operation_id=forget_operation_id,
        )

        with pytest.raises(
            AssertionConflictError,
            match="different explicit fact deletion",
        ):
            await storage.forget_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                invocation_id="forget-live-collision",
            )
        assert await storage.get_assertion(desired.assertion_id) is not None
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_forget_erased_replay_rejects_unrelated_raw_delete_preseed(
    db_backend,
    tmp_path,
):
    """A blinded unrelated delete cannot terminate the intended selector."""
    from kestrel_sovereign.features.memory_agency.semantic_facts import (
        _operation_material,
    )
    from kestrel_sovereign.storage.async_assertion_store import (
        AssertionConflictError,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "forget-erased-selector-binding",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        desired = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="forget-erased-desired",
            confidence=0.9,
            invocation_id="forget-erased-save",
        )
        unrelated = _semantic_assertion(
            tenant,
            "forget-erased-unrelated-revision",
            value="forget-erased-unrelated",
        )
        await raw_storage.put_assertion(
            unrelated,
            source_occurrences=(_semantic_source("parity-source"),),
            operation_id="forget-erased-unrelated-put",
        )
        forget_operation_id, _ = _operation_material(
            action="forget",
            subject="user",
            predicate="preferred_deploy_region",
            value=None,
            confidence_requested=None,
            invocation_id="forget-erased-collision",
        )
        deleted = await raw_storage.delete_assertion(
            unrelated.assertion_id,
            unrelated.revision_id,
            operation_id=forget_operation_id,
        )
        await raw_storage.erase_assertion(
            deleted.deleted.assertion_id,
            operation_id="forget-erased-unrelated-erasure",
        )

        with pytest.raises(
            AssertionConflictError,
            match="different erased explicit fact deletion",
        ):
            await storage.forget_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                invocation_id="forget-erased-collision",
            )
        assert await storage.get_assertion(desired.assertion_id) is not None
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_forget_noop_replay_rejects_different_selector_preseed(
    db_backend,
    tmp_path,
):
    """An absent receipt for selector A cannot suppress selector B."""
    from kestrel_sovereign.features.memory_agency.semantic_facts import (
        _operation_material,
    )
    from kestrel_sovereign.knowledge import IRI
    from kestrel_sovereign.storage.async_assertion_store import (
        AssertionConflictError,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "forget-noop-selector-binding",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        forget_operation_id, _ = _operation_material(
            action="forget",
            subject="user",
            predicate="preferred_deploy_region",
            value=None,
            confidence_requested=None,
            invocation_id="forget-noop-collision",
        )
        await raw_storage._record_explicit_fact_forget_noop(
            forget_operation_id,
            IRI(f"urn:kestrel:agent:{tenant}:principal:other"),
            IRI("https://kestrel.ai/vocab/unrelatedSelector"),
        )
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        desired = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="forget-noop-desired",
            confidence=0.9,
            invocation_id="forget-noop-save",
        )

        with pytest.raises(
            AssertionConflictError,
            match="different semantic mutation",
        ):
            await storage.forget_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                invocation_id="forget-noop-collision",
            )
        assert await storage.get_assertion(desired.assertion_id) is not None
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_save_fact_concurrent_retry_replays_first_delivery_provenance(
    db_backend,
    tmp_path,
):
    """One retry ID has one canonical receipt despite distinct delivery clocks."""
    from kestrel_sovereign.agent.invocation import invocation_scope, request_provenance
    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-concurrent-retry",
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        async def deliver(received_at: str):
            provenance = request_provenance(
                actor="parity-user",
                source_kind="http_request",
                source_locator="POST:/api/agent/invoke",
                received_at=received_at,
            )
            with invocation_scope("concurrent-retry-2765", provenance=provenance):
                return await storage.save_explicit_fact(
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
async def test_concurrent_distinct_same_value_teachings_preserve_both_sources(
    db_backend,
    tmp_path,
):
    """A stale initial-write decision is retried as a provenance append."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-concurrent-distinct",
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        original_put = storage.put_assertion
        both_initial_writes_ready = asyncio.Event()
        arrival_lock = asyncio.Lock()
        arrivals = 0

        async def synchronized_initial_put(assertion, **kwargs):
            nonlocal arrivals
            async with arrival_lock:
                arrivals += 1
                if arrivals == 2:
                    both_initial_writes_ready.set()
            await both_initial_writes_ready.wait()
            return await original_put(assertion, **kwargs)

        # Both invocations observe no current fact before either initial write
        # reaches storage.  One must re-decide after its stale put conflicts.
        storage.put_assertion = synchronized_initial_put

        async def teach(invocation_id: str):
            return await storage.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="us-central1",
                confidence=0.9,
                invocation_id=invocation_id,
            )

        first, second = await asyncio.gather(
            teach("concurrent-distinct-a"),
            teach("concurrent-distinct-b"),
        )

        assert first.saved is True
        assert second.saved is True
        assert first.idempotent is False
        assert second.idempotent is False
        assert first.assertion_id == second.assertion_id
        assert first.provenance_reference != second.provenance_reference

        sources = await storage.list_assertion_sources(first.assertion_id)
        assert {source.source_occurrence_id for source in sources} == {
            first.provenance_reference,
            second.provenance_reference,
        }

        first_retry, second_retry = await asyncio.gather(
            teach("concurrent-distinct-a"),
            teach("concurrent-distinct-b"),
        )
        assert first_retry.idempotent is True
        assert second_retry.idempotent is True
        assert first_retry.revision_id == first.revision_id
        assert second_retry.revision_id == second.revision_id
        assert len(await storage.list_assertion_sources(first.assertion_id)) == 2
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_noop_forget_replay_never_deletes_a_later_teaching(
    db_backend,
    tmp_path,
):
    """An absent-result forget is a durable tombstone, including after restart."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-noop-forget",
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        absent = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="noop-before-teach",
        )
        taught = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="europe-west4",
            confidence=0.9,
            invocation_id="teach-after-noop",
        )
        stale_retry = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="noop-before-teach",
        )

        assert absent.deleted is False
        assert absent.idempotent is False
        assert stale_retry.deleted is False
        assert stale_retry.idempotent is True
        assert await storage.get_assertion(taught.assertion_id) is not None
    finally:
        await raw_storage.close()

    restarted_raw = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        restarted = PrivacyEnforcingStorage(restarted_raw, PrivacyMode.NORMAL)
        restarted_retry = await restarted.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="noop-before-teach",
        )
        assert restarted_retry.deleted is False
        assert restarted_retry.idempotent is True
        assert await restarted.get_assertion(taught.assertion_id) is not None
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_explicit_fact_replays_append_and_delete_from_the_operation_ledger(
    db_backend,
    tmp_path,
):
    """Later source appends cannot rewrite an older retry's receipt.

    This is the adversarial lifecycle that exposed #2765's reconstruction
    bug: A → B → C all carry the same claim but distinct provenance.  Retrying
    B must return B's original validated revision, and deleting C then
    retrying that delete must use its persisted receipt rather than counting
    historical ``ACTIVE`` revisions.
    """
    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-ledger-replay",
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        first = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="ledger-a",
        )
        second = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="ledger-b",
        )
        third = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="ledger-c",
        )

        assert first.idempotent is False
        assert second.idempotent is False
        assert third.idempotent is False
        assert len(await storage.list_assertion_sources(first.assertion_id)) == 3

        retry_second = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="ledger-b",
        )
        assert retry_second.idempotent is True
        assert retry_second.revision_id == second.revision_id
        assert retry_second.provenance_reference == second.provenance_reference
        current = await storage.get_assertion(first.assertion_id)
        assert current is not None
        assert current.revision_id == third.revision_id

        deleted = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="ledger-delete",
        )
        deleted_retry = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="ledger-delete",
        )
        assert deleted.deleted is True
        assert deleted_retry.deleted is True
        assert deleted_retry.idempotent is True
        assert deleted_retry.revision_id == deleted.revision_id
        assert await storage.get_assertion(first.assertion_id) is None
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_erased_explicit_fact_operations_are_terminal_across_restart(
    db_backend,
    tmp_path,
):
    """Erasure blinds both the original save and later source-append receipts."""
    from kestrel_sovereign.storage.async_assertion_store import (
        AssertionConflictError,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-erased-operation-tombstones",
    )
    raw_storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        original = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="secret-region",
            confidence=0.9,
            invocation_id="erased-original-save",
        )
        appended = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="secret-region",
            confidence=0.9,
            invocation_id="erased-source-append",
        )
        assert original.saved is True
        assert appended.saved is True
        assert original.revision_id != appended.revision_id

        await storage.erase_assertion(
            original.assertion_id,
            operation_id="erase-explicit-fact-history",
        )

        assert await raw_storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertion_operations "
            "WHERE tenant_id = ?",
            (tenant,),
        ) == 0
        tombstones = await raw_storage.db.fetchall(
            "SELECT purpose, operation_key, request_key, generation "
            "FROM semantic_assertion_erased_operation_tombstones "
            "WHERE tenant_id = ? ORDER BY purpose",
            (tenant,),
        )
        assert [row[0] for row in tombstones] == ["put", "supersede"]
        encoded_tombstones = repr(tombstones)
        for erased_value in (
            "secret-region",
            original.operation_id,
            appended.operation_id,
            original.assertion_id,
            original.revision_id,
            appended.revision_id,
            original.provenance_reference,
            appended.provenance_reference,
        ):
            assert erased_value not in encoded_tombstones
    finally:
        await raw_storage.close()

    restarted_raw = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        restarted = PrivacyEnforcingStorage(
            restarted_raw,
            PrivacyMode.NORMAL,
        )
        original_replay = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="secret-region",
            confidence=0.9,
            invocation_id="erased-original-save",
        )
        append_replay = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="secret-region",
            confidence=0.9,
            invocation_id="erased-source-append",
        )
        for replay in (original_replay, append_replay):
            assert replay.saved is False
            assert replay.idempotent is True
            assert replay.validation_disposition == "erased:terminal"
            assert replay.assertion_id is None
            assert replay.revision_id is None
            assert replay.validation_report_id is None
            assert replay.provenance_reference is None
            assert replay.provenance_digest is None
        assert await restarted.query_assertions() == []

        conflicting = _semantic_assertion(
            tenant,
            "different-content-revision",
            value="different-content",
        )
        with pytest.raises(
            AssertionConflictError,
            match="different semantic mutation",
        ):
            await restarted_raw.put_assertion(
                conflicting,
                source_occurrences=(_semantic_source("parity-source"),),
                operation_id=original.operation_id,
            )
        assert await restarted.query_assertions() == []
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_erased_changed_value_supersession_replays_terminally(
    db_backend,
    tmp_path,
):
    """A changed-claim supersession gets the same blinded result binding."""
    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "save-fact-erased-changed-supersession",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="changed-erasure-old",
            confidence=0.9,
            invocation_id="changed-erasure-old-save",
        )
        changed = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="changed-erasure-new",
            confidence=0.9,
            invocation_id="changed-erasure-new-save",
        )
        await storage.erase_assertion(
            changed.assertion_id,
            operation_id="changed-erasure-physical",
        )
    finally:
        await raw_storage.close()

    restarted_raw = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        restarted = PrivacyEnforcingStorage(
            restarted_raw,
            PrivacyMode.NORMAL,
        )
        replay = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="changed-erasure-new",
            confidence=0.9,
            invocation_id="changed-erasure-new-save",
        )
        assert replay.saved is False
        assert replay.idempotent is True
        assert replay.validation_disposition == "erased:terminal"
        assert replay.assertion_id is None
        assert replay.revision_id is None
        assert replay.provenance_reference is None
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_legacy_erasure_fence_blocks_only_matching_semantic_identity(
    db_backend,
    tmp_path,
):
    """The v3 migration's opaque output is backend-neutral and narrowly scoped."""
    from kestrel_sovereign.storage.async_assertion_store import (
        _legacy_erasure_assertion_key,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "legacy-erasure-fence-parity",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        saved = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-fenced-region",
            confidence=0.9,
            invocation_id="legacy-fenced-save",
        )
        await storage.erase_assertion(
            saved.assertion_id,
            operation_id="legacy-fenced-erasure",
        )
        erasure_row = await raw_storage.db.fetchone(
            "SELECT request_digest, generation "
            "FROM semantic_assertion_erasure_receipts "
            "WHERE tenant_id = ?",
            (tenant,),
        )
        assert erasure_row is not None
        await raw_storage.db.execute(
            "DELETE FROM semantic_assertion_erased_operation_tombstones "
            "WHERE tenant_id = ?",
            (tenant,),
        )
        await raw_storage.db.execute(
            "INSERT INTO semantic_assertion_legacy_erasure_fences "
            "(tenant_id, assertion_key, generation, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                tenant,
                _legacy_erasure_assertion_key(str(erasure_row[0])),
                int(erasure_row[1]),
                "2026-07-27T00:00:00Z",
            ),
        )
    finally:
        await raw_storage.close()

    restarted_raw = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    try:
        restarted = PrivacyEnforcingStorage(
            restarted_raw,
            PrivacyMode.NORMAL,
        )
        stale = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-fenced-region",
            confidence=0.9,
            invocation_id="legacy-fenced-save",
        )
        assert stale.saved is False
        assert stale.idempotent is True
        assert stale.validation_disposition == "erased:terminal"
        assert await restarted.query_assertions() == []

        unrelated = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="new-unrelated-region",
            confidence=0.9,
            invocation_id="new-unrelated-save",
        )
        assert unrelated.saved is True
        assert unrelated.idempotent is False
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_governed_replay_racing_physical_erasure_is_terminal(
    db_backend,
    tmp_path,
    monkeypatch,
):
    """A replay split across erasure observes its new blinded tombstone."""
    from kestrel_sovereign.storage.async_assertion_store import (
        AsyncAssertionStore,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "governed-replay-erasure-race",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    release_replay = asyncio.Event()
    operation_observed = asyncio.Event()
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        saved = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="race-erased-region",
            confidence=0.9,
            invocation_id="race-erased-save",
        )
        target_store = raw_storage._assertion_store()
        original_recorded = AsyncAssertionStore._recorded_operation
        blocked = False

        async def recorded_with_barrier(store, operation_id):
            nonlocal blocked
            recorded = await original_recorded(store, operation_id)
            if (
                store is target_store
                and operation_id == saved.operation_id
                and recorded is not None
                and not blocked
            ):
                blocked = True
                operation_observed.set()
                await release_replay.wait()
            return recorded

        monkeypatch.setattr(
            AsyncAssertionStore,
            "_recorded_operation",
            recorded_with_barrier,
        )
        replay_task = asyncio.create_task(
            storage.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="race-erased-region",
                confidence=0.9,
                invocation_id="race-erased-save",
            )
        )
        await asyncio.wait_for(operation_observed.wait(), timeout=5)
        await storage.erase_assertion(
            saved.assertion_id,
            operation_id="race-erasure",
        )
        release_replay.set()
        replay = await replay_task

        assert replay.saved is False
        assert replay.idempotent is True
        assert replay.validation_disposition == "erased:terminal"
        assert replay.assertion_id is None
        assert replay.revision_id is None
        assert replay.provenance_reference is None
    finally:
        release_replay.set()
        await raw_storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_delete_replay_racing_physical_erasure_is_terminal(
    db_backend,
    tmp_path,
    monkeypatch,
):
    """The analogous delete replay never leaks a transient missing revision."""
    from kestrel_sovereign.storage.async_assertion_store import (
        AsyncAssertionStore,
    )

    tenant, identity = await _incepted_assertion_identity(
        tmp_path,
        "delete-replay-erasure-race",
    )
    raw_storage = await _assertion_storage_for_backend(
        db_backend,
        tenant,
        identity,
    )
    release_replay = asyncio.Event()
    operation_observed = asyncio.Event()
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)
        saved = await storage.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="delete-race-region",
            confidence=0.9,
            invocation_id="delete-race-save",
        )
        deleted = await storage.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="delete-race-forget",
        )
        target_store = raw_storage._assertion_store()
        original_recorded = AsyncAssertionStore._recorded_operation
        blocked = False

        async def recorded_with_barrier(store, operation_id):
            nonlocal blocked
            recorded = await original_recorded(store, operation_id)
            if (
                store is target_store
                and operation_id == deleted.operation_id
                and recorded is not None
                and not blocked
            ):
                blocked = True
                operation_observed.set()
                await release_replay.wait()
            return recorded

        monkeypatch.setattr(
            AsyncAssertionStore,
            "_recorded_operation",
            recorded_with_barrier,
        )
        replay_task = asyncio.create_task(
            storage.forget_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                invocation_id="delete-race-forget",
            )
        )
        await asyncio.wait_for(operation_observed.wait(), timeout=5)
        await storage.erase_assertion(
            saved.assertion_id,
            operation_id="delete-race-erasure",
        )
        release_replay.set()
        replay = await replay_task

        assert replay.deleted is False
        assert replay.idempotent is True
        assert replay.assertion_id is None
        assert replay.revision_id is None
        assert replay.provenance_reference is None
    finally:
        release_replay.set()
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
async def test_erasure_redacts_maintenance_artifacts_on_both_backends(
    db_backend,
    tmp_path,
):
    """JSON report evidence and every resumable cursor share erasure semantics."""
    tenant, identity = await _incepted_assertion_identity(tmp_path, "maintenance-erasure")
    storage = await _assertion_storage_for_backend(db_backend, tenant, identity)
    try:
        root = _semantic_assertion(
            tenant,
            "maintenance-erasure-root",
            value="maintenance-erasure-secret",
        )
        unrelated = _semantic_assertion(
            tenant,
            "maintenance-erasure-unrelated",
            value="maintenance-unrelated",
        )
        written = await storage.put_assertion(
            root,
            source_occurrences=(_semantic_source("parity-source"),),
        )
        await storage.put_assertion(
            unrelated,
            source_occurrences=(_semantic_source("parity-source"),),
        )
        await storage.db.execute(
            "INSERT INTO semantic_maintenance_state "
            "(tenant_id, profile_key, checkpoint_generation, checkpoint_event_id, run_id, "
            "status, capability_versions, repair_cursor_revision_id, repair_active, repair_mode, "
            "repair_scan_complete, repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, audit_assertion_revision_id, "
            "audit_competitor_cursor_revision_id, updated_at) "
            "VALUES (?, 'maintenance-profile', ?, ?, 'maintenance-run', 'complete', '{}', "
            "?, 1, 'current_scan', 1, ?, NULL, ?, ?, ?, ?, '2026-07-27T00:00:00Z')",
            (
                tenant,
                written.generation,
                written.event_id,
                root.revision_id,
                written.generation,
                root.revision_id,
                root.assertion_id,
                root.revision_id,
                root.revision_id,
            ),
        )

        async def record_report(report_id: str, evidence: object) -> None:
            await storage.db.execute(
                "INSERT INTO semantic_maintenance_reports "
                "(tenant_id, report_id, report_kind, evidence_digest, status, evidence_mapping, "
                "created_at, updated_at) VALUES (?, ?, 'contradiction_candidate', ?, "
                "'review_required', ?, '2026-07-27T00:00:00Z', '2026-07-27T00:00:00Z')",
                (
                    tenant,
                    report_id,
                    f"digest:{report_id}",
                    json.dumps(evidence, sort_keys=True),
                ),
            )

        await record_report(
            "maintenance-erasure-flat",
            {
                "assertion_id": root.assertion_id,
                "revision_id": root.revision_id,
                "content": "maintenance-erasure-secret",
            },
        )
        await record_report(
            "maintenance-erasure-nested",
            {
                "nested": [
                    {"assertion_ids": [root.assertion_id]},
                    {"cursor": f"revision:{root.revision_id}"},
                ]
            },
        )
        await record_report(
            "maintenance-erasure-unrelated",
            {
                "assertion_id": unrelated.assertion_id,
                "revision_id": unrelated.revision_id,
            },
        )

        erased = await storage.erase_assertion(
            root.assertion_id,
            operation_id="maintenance-erasure-parity",
        )

        assert await storage.db.fetchall(
            "SELECT report_id FROM semantic_maintenance_reports "
            "WHERE tenant_id = ? ORDER BY report_id",
            (tenant,),
        ) == [("maintenance-erasure-unrelated",)]
        assert await storage.db.fetchone(
            "SELECT checkpoint_generation, checkpoint_event_id, status, "
            "repair_cursor_revision_id, repair_active, repair_mode, repair_scan_complete, "
            "repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, "
            "audit_assertion_revision_id, audit_competitor_cursor_revision_id "
            "FROM semantic_maintenance_state WHERE tenant_id = ?",
            (tenant,),
        ) == (
            erased.generation,
            None,
            "partial",
            None,
            0,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        for table_name in (
            "semantic_maintenance_state",
            "semantic_maintenance_runs",
            "semantic_maintenance_leases",
            "semantic_maintenance_reports",
        ):
            rows = await storage.db.fetchall(
                f"SELECT * FROM {table_name} WHERE tenant_id = ?",
                (tenant,),
            )
            serialized = repr(rows)
            assert root.assertion_id not in serialized
            assert root.revision_id not in serialized
            assert "maintenance-erasure-secret" not in serialized
        changes = await storage.assertion_changes_since(written.generation)
        erasure_changes = [change for change in changes if change.operation == "erased"]
        assert len(erasure_changes) == 1
        assert erasure_changes[0].assertion_id is None
        assert erasure_changes[0].revision_id is None
        assert erasure_changes[0].generation == erased.generation
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
            ("semantic_assertion_store_v5",),
        )
        assert marker_rows == [("semantic_assertion_store_v5",)]
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

    # The conversation list, which since #2960 pages the sessions table rather
    # than a window of raw rows. Scoped to this agent on both backends: the
    # other agent's row must not appear in it.
    page = await privacy_storage.list_session_page(agent_id, limit=10)
    assert [s["session_id"] for s in page["sessions"]] == [session_id]
    assert page["sessions"][0]["message_count"] == 2
    assert page["sessions"][0]["preview_content"] == "hello"
    assert page["next_cursor"] is None

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
        ),
        creator_agent_id="did:test:creator-a",
        recipient_agent_id="did:test:recipient-a",
    )
    await store.save(
        Task(
            id=task_b,
            sessionId=session_a,
            status=TaskStatus(state=TaskState.COMPLETED),
            metadata={"task_type": "audit", "user_id": user_a, "marker": "second"},
        ),
        creator_agent_id="did:test:creator-a",
        recipient_agent_id="did:test:recipient-a",
    )
    await store.save(
        Task(
            id=other_task,
            sessionId=session_b,
            status=TaskStatus(state=TaskState.SUBMITTED),
            metadata={"task_type": "audit", "user_id": user_b, "marker": "other"},
        ),
        creator_agent_id="did:test:creator-b",
        recipient_agent_id="did:test:recipient-b",
    )

    await store.update_status(
        task_a,
        TaskStatus(
            state=TaskState.WORKING,
            message=Message(role="agent", parts=[TextPart(text="underway")]),
        ),
        recipient_agent_id="did:test:recipient-a",
    )
    await store.add_artifact(
        task_a,
        Artifact(
            name="result.txt",
            parts=[TextPart(text="semantic parity")],
        ),
        recipient_agent_id="did:test:recipient-a",
    )

    retrieved = await store._get_unscoped(task_a)
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

    session_tasks = await store.list_tasks(
        recipient_agent_id="did:test:recipient-a",
        session_id=session_a,
        user_id=user_a,
        limit=10,
    )
    assert {task.id for task in session_tasks} == {task_a, task_b}
    assert {task.metadata["marker"] for task in session_tasks} == {"first", "second"}

    working_tasks = await store.list_tasks(
        recipient_agent_id="did:test:recipient-a",
        user_id=user_a,
        status=TaskState.WORKING,
        limit=10,
    )
    assert [task.id for task in working_tasks] == [task_a]

    assert await store.delete(task_a) is True
    assert await store._get_unscoped(task_a) is None


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
