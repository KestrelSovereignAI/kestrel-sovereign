"""Canonical assertion persistence contracts on the SQLite backend."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from kestrel_sovereign.identity.runtime_identity import (
    AgentIdentity,
    load_agent_identity,
)
from kestrel_sovereign.inception_service import create_kestrel_identity
from kestrel_sovereign.knowledge import (
    Assertion,
    AssertionQuery,
    AssertionStatus,
    CorpusCheckpoint,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    IRI,
    Literal,
    OntologyRef,
    SourceOccurrence,
    XSD_STRING,
)
from kestrel_sovereign.storage.async_assertion_store import (
    AssertionStoreError,
    AsyncAssertionStore,
    TenantIsolationError,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage, PrivacyViolationError
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.sqla.migrations import (
    _SEMANTIC_MAINTENANCE_ERASURE_REDACTION_SCHEMA_VERSION,
    migrate_semantic_assertion_store,
    migrate_semantic_validation_reports,
)
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)


TENANT = "did:example:semantic-test"
OTHER_TENANT = "did:example:other-semantic-test"
OWNER = "did:example:semantic-test"
ONTOLOGY = OntologyRef("kestrel-test", "1", "sha256:test", "semantic-kb-v1")
SUBJECT = IRI("urn:kestrel:agent:did:example:semantic-test:principal:user")
PREDICATE = IRI("https://kestrel.ai/vocab/preferredRegion")
_LOADER_VERIFIED_IDENTITY: AgentIdentity | None = None


@pytest.fixture(scope="module", autouse=True)
def _incept_loader_verified_assertion_tenant(tmp_path_factory) -> None:
    """Use the same incept → load boundary that production boot uses."""
    global TENANT, OWNER, SUBJECT, _LOADER_VERIFIED_IDENTITY

    identity_dir = tmp_path_factory.mktemp("semantic-assertion-identity")
    credentials = create_kestrel_identity(
        str(identity_dir),
        identity_method="did:pkh",
        agent_name="Semantic Assertion Store Test",
    )
    tenant_id = credentials.agent_did
    key_id = f"kestrel_{tenant_id.rsplit(':', 1)[-1]}"
    _LOADER_VERIFIED_IDENTITY = load_agent_identity(key_id, identity_dir)
    TENANT = tenant_id
    OWNER = tenant_id
    SUBJECT = IRI(f"urn:kestrel:agent:{tenant_id}:principal:user")


async def _storage() -> AsyncStorage:
    storage = AsyncStorage(
        ":memory:",
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await storage.initialize()
    return storage


def _assertion_capability(tenant_id: str):
    assert _LOADER_VERIFIED_IDENTITY is not None
    return _resolve_authenticated_agent_assertion_capability(
        tenant_id,
        _LOADER_VERIFIED_IDENTITY,
    )


def source(identifier: str) -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id=identifier,
        source_kind="conversation",
        locator=f"conversation:{identifier}",
        received_at="2026-07-26T14:02:11Z",
        content_digest="sha256:source",
        actor="operator",
        selector="body",
    )


def direct(
    revision_id: str,
    value: str = "us-central1",
    source_id: str = "source-1",
    *,
    tenant: str | None = None,
    owner: str | None = None,
) -> Assertion:
    tenant = tenant or TENANT
    owner = owner or tenant
    return Assertion(
        tenant_id=tenant,
        owning_agent_id=owner,
        subject=IRI(f"urn:kestrel:agent:{tenant}:principal:user"),
        predicate=PREDICATE,
        object=Literal(value, XSD_STRING),
        revision_id=revision_id,
        confidence=Decimal("0.92"),
        confidence_method="operator",
        confidence_basis="test",
        epistemic_state=EpistemicState.REPORTED,
        asserted_at="2026-07-26T14:02:11Z",
        ontology_version=ONTOLOGY,
        lineage=DirectLineage((source_id,)),
        privacy_classification="normal",
        release_policy_reference="policy:private-v1",
    )


def derived(
    revision_id: str,
    input_revision_id: str,
    *,
    marker: str = "eligibleForRegion",
    tenant: str | None = None,
) -> Assertion:
    tenant = tenant or TENANT
    return Assertion(
        tenant_id=tenant,
        owning_agent_id=tenant,
        subject=IRI(f"urn:kestrel:agent:{tenant}:principal:user"),
        predicate=IRI(f"https://kestrel.ai/vocab/{marker}"),
        object=Literal("true", XSD_STRING),
        revision_id=revision_id,
        confidence="1",
        confidence_method="rule",
        confidence_basis="test",
        epistemic_state=EpistemicState.INFERRED,
        asserted_at="2026-07-26T14:02:12Z",
        ontology_version=ONTOLOGY,
        lineage=DerivedLineage(
            rule_id=f"{marker}-rule", engine_version="1", profile_version="1",
            input_revision_ids=(input_revision_id,), input_digest="sha256:inputs",
            run_id="run-1", generated_at="2026-07-26T14:02:12Z",
        ),
        privacy_classification="normal",
        release_policy_reference="policy:private-v1",
    )


@pytest.mark.asyncio
async def test_assertion_crud_provenance_idempotency_and_checkpoint() -> None:
    storage = await _storage()
    db = storage.db
    try:
        store = storage
        assertion = direct("revision-1")

        written = await store.put_assertion(assertion, source_occurrences=(source("source-1"),), operation_id="put-1")
        replay = await store.put_assertion(assertion, source_occurrences=(source("source-1"),), operation_id="put-1")

        assert written.idempotent is False
        assert replay.idempotent is True
        assert await store.get_assertion(assertion.assertion_id) == assertion
        assert await store.query_assertions(AssertionQuery(subject=SUBJECT)) == [assertion]
        assert await store.list_assertion_sources(assertion.assertion_id) == [source("source-1")]
        assert await store.list_assertion_revision_sources(assertion.revision_id) == [
            source("source-1")
        ]
        assert await store.list_assertion_revision_sources_batch(
            (assertion.revision_id, "missing-revision")
        ) == {
            assertion.revision_id: [source("source-1")],
            "missing-revision": [],
        }
        validation = await store.assertion_validation_statuses([assertion.assertion_id])
        assert validation[assertion.assertion_id].state.value == "conforms"
        assert validation[assertion.assertion_id].action.value in {"accept", "accept-with-report"}
        checkpoint = await store.assertion_checkpoint()
        assert checkpoint.generation == 1
        assert checkpoint.latest_event_id == written.event_id
        assert [change.revision_id for change in await store.assertion_changes_since(0)] == [assertion.revision_id]
        assert [
            change.revision_id
            for change in await store.assertion_changes_after(
                CorpusCheckpoint(TENANT, 0, None)
            )
        ] == [assertion.revision_id]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_list_assertion_sources_selects_distinct_sort_keys_for_postgres() -> None:
    """The portable DISTINCT query projects every PostgreSQL ORDER BY key."""
    from unittest.mock import AsyncMock, patch

    storage = await _storage()
    try:
        store = storage._assertion_store()
        fetchall = AsyncMock(return_value=[])
        with patch.object(store._database, "fetchall", fetchall):
            assert await store.list_source_occurrences("postgres-distinct-order") == []

        query, params = fetchall.await_args.args
        assert params == (TENANT, "postgres-distinct-order")
        assert (
            "SELECT DISTINCT s.source_mapping, s.received_at, s.source_occurrence_id "
            "FROM semantic_assertion_revisions r "
        ) in query
        assert "ORDER BY s.received_at, s.source_occurrence_id" in query
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_governed_source_append_creates_a_validated_provenance_revision() -> None:
    """A distinct evidence occurrence is a canonical revision, not a no-op."""
    storage = await _storage()
    try:
        governed = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        first = direct("append-first", source_id="append-source-a")
        await governed.put_assertion(
            first,
            source_occurrences=(source("append-source-a"),),
            operation_id="append-first-operation",
        )
        replacement = replace(
            first,
            revision_id="append-second",
            lineage=DirectLineage(("append-source-a", "append-source-b")),
        )

        appended = await governed.append_assertion_source(
            first.revision_id,
            replacement,
            source_occurrences=(source("append-source-b"),),
            operation_id="append-second-operation",
        )
        replay = await governed.append_assertion_source(
            first.revision_id,
            replacement,
            source_occurrences=(source("append-source-b"),),
            operation_id="append-second-operation",
        )

        assert appended.accepted is True
        assert appended.idempotent is False
        assert replay.accepted is True
        assert replay.idempotent is True
        assert appended.report.report_id == replay.report.report_id
        current = await governed.get_assertion(first.assertion_id)
        assert current is not None
        assert current.revision_id == replacement.revision_id
        assert current.lineage == replacement.lineage
        assert [
            item.source_occurrence_id
            for item in await governed.list_assertion_sources(first.assertion_id)
        ] == ["append-source-a", "append-source-b"]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_raw_storage_binding_cannot_authorize_the_explicit_fact_adapter() -> None:
    """Raw storage, module helpers, and method aliasing cannot mint authority."""

    storage = await _storage()
    try:
        assert not hasattr(storage, "save_explicit_fact")
        assert not hasattr(storage, "forget_explicit_fact")

        class ForwardingShim:
            def __init__(self, wrapped) -> None:
                self._wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        forged = ForwardingShim(storage)
        assert not hasattr(forged, "save_explicit_fact")
        assert not hasattr(forged, "forget_explicit_fact")
        adapter_ledger_helpers = (
            "replay_governed_assertion_operation",
            "terminalize_legacy_erased_explicit_fact_operation",
            "replay_delete_assertion_operation",
            "replay_explicit_fact_forget_operation",
            "record_explicit_fact_forget_noop",
        )
        for helper_name in adapter_ledger_helpers:
            assert not hasattr(storage, helper_name)
            assert not hasattr(forged, helper_name)
        from kestrel_sovereign.storage.semantic_binding import SemanticAssertionBinding
        import kestrel_sovereign.storage as storage_module
        import kestrel_sovereign.storage.semantic_binding as binding_module
        import kestrel_sovereign.features.memory_agency.semantic_facts as fact_module

        raw_binding = storage.semantic_assertion_binding()
        assert isinstance(raw_binding, SemanticAssertionBinding)
        assert not hasattr(storage_module, "GovernedAssertionReplayBinding")
        assert not hasattr(storage_module, "governed_assertion_replay_binding")
        assert not hasattr(raw_binding, "_governed_marker")
        assert not hasattr(binding_module, "_issue_governed_semantic_assertion_binding")
        assert not hasattr(binding_module, "is_governed_semantic_assertion_binding")
        assert not hasattr(fact_module, "GovernedFactAdapter")
        assert not hasattr(fact_module, "_save_explicit_fact")
        assert not hasattr(fact_module, "_forget_explicit_fact")

        ephemeral = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        with pytest.raises(PrivacyViolationError):
            await ephemeral.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="us-central1",
                confidence=0.9,
                invocation_id="ephemeral-direct-call",
            )

        # Copying the public descriptor does not copy a wrapper's captured
        # executor.  The shim therefore cannot turn raw storage into the
        # explicit-fact authority that PrivacyEnforcingStorage owns.
        forged.save_explicit_fact = (
            PrivacyEnforcingStorage.save_explicit_fact.__get__(
                forged,
                ForwardingShim,
            )
        )
        with pytest.raises(AttributeError, match="__save_explicit_fact"):
            await forged.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="us-central1",
                confidence=0.9,
                invocation_id="shim-method-alias",
            )
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertion_revisions"
        ) == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_foreign_tenant_and_missing_lineage_fail_without_writes() -> None:
    storage = await _storage()
    db = storage.db
    try:
        store = storage
        with pytest.raises(TenantIsolationError):
            await store.put_assertion(
                direct("other-revision", tenant=OTHER_TENANT, owner=OTHER_TENANT),
                source_occurrences=(source("other-source"),),
            )
        with pytest.raises(AssertionStoreError, match="unknown tenant-local source"):
            await store.put_assertion(direct("missing-source", source_id="absent"))
        count = await db.fetchval("SELECT COUNT(*) FROM semantic_assertion_revisions")
        assert count == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_raw_database_cannot_issue_a_forged_assertion_tenant_scope() -> None:
    storage = await _storage()
    try:
        with pytest.raises(TypeError, match="agent-bound AsyncStorage"):
            AsyncAssertionStore(storage.db)
    finally:
        await storage.close()


def test_assertion_authority_requires_a_loader_verified_identity() -> None:
    assert _LOADER_VERIFIED_IDENTITY is not None
    assert _resolve_authenticated_agent_assertion_capability(TENANT, None) is None
    with pytest.raises(TypeError, match="loader-verified AgentIdentity"):
        _resolve_authenticated_agent_assertion_capability(
            TENANT,
            SimpleNamespace(legacy_did=TENANT, new_did=None),
        )
    with pytest.raises(TypeError, match="loader-verified AgentIdentity"):
        _resolve_authenticated_agent_assertion_capability(
            TENANT,
            AgentIdentity(
                legacy_did=TENANT,
                legacy_keypair=_LOADER_VERIFIED_IDENTITY.legacy_keypair,
                legacy_did_document=_LOADER_VERIFIED_IDENTITY.legacy_did_document,
            ),
        )
    with pytest.raises(ValueError, match="not bound"):
        _resolve_authenticated_agent_assertion_capability(
            OTHER_TENANT,
            _LOADER_VERIFIED_IDENTITY,
        )
    copied_and_mutated = copy.copy(_LOADER_VERIFIED_IDENTITY)
    object.__setattr__(copied_and_mutated, "legacy_did", OTHER_TENANT)
    with pytest.raises(TypeError, match="loader-verified AgentIdentity"):
        _resolve_authenticated_agent_assertion_capability(
            OTHER_TENANT,
            copied_and_mutated,
        )
    capability = _assertion_capability(TENANT)
    assert capability is not None and capability.tenant_id == TENANT


@pytest.mark.asyncio
async def test_public_sqlite_factory_cannot_mint_assertion_authority(tmp_path) -> None:
    """Public storage factories scope ordinary data but cannot grant semantic authority."""
    raw_storage = AsyncStorage(":memory:", agent_id=TENANT)
    await raw_storage.initialize()
    try:
        assertion = direct("factory-revision", source_id="factory-source")
        with pytest.raises(RuntimeError, match="agent-bound AsyncStorage"):
            await raw_storage.put_assertion(
                assertion,
                source_occurrences=(source("factory-source"),),
            )
    finally:
        await raw_storage.close()

    factory = await AsyncStorage.create_sqlite(
        str(tmp_path / "factory-semantic.db"),
        agent_id=TENANT,
    )
    try:
        with pytest.raises(RuntimeError, match="agent-bound AsyncStorage"):
            await factory.put_assertion(
                direct("factory-result", source_id="factory-result-source"),
                source_occurrences=(source("factory-result-source"),),
            )
    finally:
        await factory.close()


@pytest.mark.asyncio
async def test_supersession_is_atomic_and_exposes_invalidation_data() -> None:
    storage = await _storage()
    db = storage.db
    try:
        store = storage
        first = direct("first-revision", source_id="first-source")
        await store.put_assertion(first, source_occurrences=(source("first-source"),))
        replacement = direct("replacement-revision", value="europe-west4", source_id="replacement-source")

        result = await store.supersede_assertion(
            first.revision_id, replacement,
            source_occurrences=(source("replacement-source"),), operation_id="replace-region",
        )

        assert result.accepted is True
        assert result.report.conforms is True
        assert result.write is not None
        assert result.predecessor.status.value == "superseded"
        assert result.replacement.supersedes_revision_id == result.predecessor.revision_id
        assert result.invalidated_revision_ids == (first.revision_id,)
        assert await store.get_assertion(replacement.assertion_id) == result.replacement
        assert await store.get_assertion(first.assertion_id) is None
        assert await db.fetchval(
            "SELECT eligible FROM semantic_projection_eligibility WHERE tenant_id = ? AND revision_id = ?",
            (TENANT, first.revision_id),
        ) == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_retraction_and_erasure_cascade_to_derived_and_eligibility() -> None:
    storage = await _storage()
    db = storage.db
    try:
        store = storage
        root = direct("root-revision", source_id="root-source")
        await store.put_assertion(root, source_occurrences=(source("root-source"),))
        inferred = derived("derived-revision", root.revision_id)
        await store.put_assertion(inferred)
        unrelated = direct("unrelated-revision", value="unrelated", source_id="unrelated-source")
        await store.put_assertion(unrelated, source_occurrences=(source("unrelated-source"),))

        retraction = await store.retract_assertion(root.assertion_id, root.revision_id)
        assert {item.assertion_id for item in retraction.retracted} == {root.assertion_id, inferred.assertion_id}
        assert await store.get_assertion(root.assertion_id) is None
        assert await store.get_assertion(inferred.assertion_id) is None
        assert await store.get_assertion(unrelated.assertion_id) == unrelated
        old_eligible = await db.fetchval(
            "SELECT eligible FROM semantic_projection_eligibility WHERE tenant_id = ? AND revision_id = ?",
            (TENANT, root.revision_id),
        )
        assert old_eligible == 0

        erased = await store.erase_assertion(root.assertion_id)
        assert set(erased.erased_assertion_ids) == {root.assertion_id, inferred.assertion_id}
        assert all([
            await store.get_assertion_revision(revision_id) is None
            for revision_id in erased.erased_revision_ids
        ])
        assert await store.get_assertion(unrelated.assertion_id) == unrelated
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_assertion_revisions") == 1
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_derivation_inputs") == 0
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_projection_eligibility") == 1
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_projection_outbox") == 1
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_projection_erasure_outbox") == 1
        changes = await store.assertion_changes_since(retraction.generation)
        assert len(changes) == 1
        assert changes[0].operation == "erased"
        assert changes[0].assertion_id is None
        assert changes[0].revision_id is None
        assert changes[0].eligible is False
        assert changes[0].generation == erased.generation
        assert (await store.assertion_checkpoint()).latest_event_id == changes[0].event_id
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_erasure_scrubs_maintenance_artifacts_and_forces_restart_resync(
    tmp_path,
) -> None:
    """Maintenance evidence/cursors cannot retain an erased canonical closure."""
    db_path = tmp_path / "maintenance-erasure.db"
    storage = AsyncStorage(
        str(db_path),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await storage.initialize()
    db = storage.db
    root = direct(
        "maintenance-erasure-root",
        value="erased-maintenance-secret",
        source_id="maintenance-erasure-source",
    )
    unrelated = direct(
        "maintenance-erasure-unrelated",
        value="maintenance-unrelated",
        source_id="maintenance-unrelated-source",
    )
    other_tenant = f"{OTHER_TENANT}:maintenance-erasure"
    other_state = (
        "other-profile",
        17,
        "other-event",
        "other-run",
        "complete",
        '{"other":"capability"}',
        "other-repair-cursor",
        1,
        "current_scan",
        1,
        16,
        "other-repair-event",
        "other-derivation-cursor",
        "other-assertion",
        "other-revision",
        "other-competitor-cursor",
        "2026-07-27T00:00:00Z",
    )
    try:
        written = await storage.put_assertion(
            root,
            source_occurrences=(source("maintenance-erasure-source"),),
        )
        await storage.put_assertion(
            unrelated,
            source_occurrences=(source("maintenance-unrelated-source"),),
        )
        # Establish a real matching maintenance state before replacing every
        # resumable cursor with an erased value.  The public run also proves
        # the erased state has a normal worker path before this regression.
        await storage.run_semantic_maintenance(None)
        await db.execute(
            "UPDATE semantic_maintenance_state SET "
            "checkpoint_generation = ?, checkpoint_event_id = ?, status = 'complete', "
            "repair_cursor_revision_id = ?, repair_active = 1, repair_mode = 'current_scan', "
            "repair_scan_complete = 1, repair_checkpoint_generation = ?, "
            "repair_checkpoint_event_id = NULL, repair_reconcile_cursor_derivation_id = ?, "
            "audit_assertion_id = ?, audit_assertion_revision_id = ?, "
            "audit_competitor_cursor_revision_id = ? "
            "WHERE tenant_id = ?",
            (
                written.generation,
                written.event_id,
                root.revision_id,
                written.generation,
                root.revision_id,
                root.assertion_id,
                root.revision_id,
                root.revision_id,
                TENANT,
            ),
        )
        await db.execute(
            "INSERT INTO semantic_maintenance_state "
            "(tenant_id, profile_key, checkpoint_generation, checkpoint_event_id, run_id, "
            "status, capability_versions, repair_cursor_revision_id, repair_active, repair_mode, "
            "repair_scan_complete, repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, audit_assertion_revision_id, "
            "audit_competitor_cursor_revision_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (other_tenant, *other_state),
        )

        async def record_report(
            tenant_id: str,
            report_id: str,
            evidence: object,
        ) -> None:
            await db.execute(
                "INSERT INTO semantic_maintenance_reports "
                "(tenant_id, report_id, report_kind, evidence_digest, status, evidence_mapping, "
                "created_at, updated_at) VALUES (?, ?, 'contradiction_candidate', ?, "
                "'review_required', ?, '2026-07-27T00:00:00Z', '2026-07-27T00:00:00Z')",
                (
                    tenant_id,
                    report_id,
                    f"digest:{report_id}",
                    json.dumps(evidence, sort_keys=True),
                ),
            )

        await record_report(
            TENANT,
            "maintenance-erased-flat",
            {
                "assertion_id": root.assertion_id,
                "revision_id": root.revision_id,
                "content": "erased-maintenance-secret",
            },
        )
        await record_report(
            TENANT,
            "maintenance-erased-nested",
            {
                "evidence": [
                    {"assertion_ids": [unrelated.assertion_id, root.assertion_id]},
                    {"cursor": f"revision:{root.revision_id}"},
                ]
            },
        )
        await record_report(
            TENANT,
            "maintenance-unrelated",
            {
                "assertion_id": unrelated.assertion_id,
                "revision_id": unrelated.revision_id,
                "content": "maintenance-unrelated",
            },
        )
        await record_report(
            other_tenant,
            "maintenance-other-tenant",
            {
                "assertion_id": "other-assertion",
                "revision_id": "other-revision",
                "content": "other-tenant-content",
            },
        )
        await db.execute(
            "INSERT INTO semantic_maintenance_leases "
            "(tenant_id, holder_id, fencing_token, expires_at, updated_at) "
            "VALUES (?, 'stale-maintenance-worker', 4, 9999999999, "
            "'2026-07-27T00:00:00Z')",
            (TENANT,),
        )

        erased = await storage.erase_assertion(
            root.assertion_id,
            operation_id="maintenance-artifact-erasure",
        )
        state = await db.fetchone(
            "SELECT checkpoint_generation, checkpoint_event_id, status, "
            "repair_cursor_revision_id, repair_active, repair_mode, repair_scan_complete, "
            "repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, "
            "audit_assertion_revision_id, audit_competitor_cursor_revision_id "
            "FROM semantic_maintenance_state WHERE tenant_id = ?",
            (TENANT,),
        )
        assert state == (
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
        assert await db.fetchone(
            "SELECT holder_id, fencing_token, expires_at "
            "FROM semantic_maintenance_leases WHERE tenant_id = ?",
            (TENANT,),
        ) == ("physical-erasure", 5, 0.0)
        assert await db.fetchone(
            "SELECT checkpoint_generation, checkpoint_event_id, status, "
            "repair_cursor_revision_id, repair_active, repair_mode, repair_scan_complete, "
            "repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, "
            "audit_assertion_revision_id, audit_competitor_cursor_revision_id "
            "FROM semantic_maintenance_state WHERE tenant_id = ?",
            (other_tenant,),
        ) == (
            other_state[1],
            other_state[2],
            other_state[4],
            other_state[6],
            other_state[7],
            other_state[8],
            other_state[9],
            other_state[10],
            other_state[11],
            other_state[12],
            other_state[13],
            other_state[14],
            other_state[15],
        )
        report_ids = await db.fetchall(
            "SELECT report_id FROM semantic_maintenance_reports "
            "WHERE tenant_id = ? ORDER BY report_id",
            (TENANT,),
        )
        assert report_ids == [("maintenance-unrelated",)]
        assert await db.fetchone(
            "SELECT evidence_mapping FROM semantic_maintenance_reports "
            "WHERE tenant_id = ? AND report_id = ?",
            (other_tenant, "maintenance-other-tenant"),
        ) is not None
        for table_name in (
            "semantic_maintenance_state",
            "semantic_maintenance_runs",
            "semantic_maintenance_leases",
            "semantic_maintenance_reports",
        ):
            rows = await db.fetchall(f"SELECT * FROM {table_name}")
            serialized = repr(rows)
            assert root.assertion_id not in serialized
            assert root.revision_id not in serialized
            assert "erased-maintenance-secret" not in serialized
        changes = await storage.assertion_changes_since(written.generation)
        erasure_changes = [change for change in changes if change.operation == "erased"]
        assert len(erasure_changes) == 1
        assert erasure_changes[0].assertion_id is None
        assert erasure_changes[0].revision_id is None
        assert erasure_changes[0].generation == erased.generation
    finally:
        await storage.close()

    restarted = AsyncStorage(
        str(db_path),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await restarted.initialize()
    try:
        resync = await restarted.run_semantic_maintenance(None)
        assert resync.status.value != "no_op"
        assert resync.changes_consumed >= 1
        assert await restarted.get_assertion(root.assertion_id) is None
        assert await restarted.db.fetchone(
            "SELECT 1 FROM semantic_maintenance_reports "
            "WHERE tenant_id = ? AND report_id = ?",
            (TENANT, "maintenance-unrelated"),
        ) is not None
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_maintenance_migration_redacts_legacy_erasure_artifacts(tmp_path) -> None:
    """An upgrade cannot retain pre-redaction report/cursor identifiers."""
    db_path = tmp_path / "legacy-maintenance-erasure.db"
    storage = AsyncStorage(
        str(db_path),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await storage.initialize()
    legacy_assertion_id = "legacy-erased-assertion"
    legacy_revision_id = "legacy-erased-revision"
    legacy_content = "legacy-erased-content"
    unaffected_tenant = f"{OTHER_TENANT}:legacy-maintenance"
    try:
        await storage.db.execute(
            "INSERT INTO semantic_maintenance_state "
            "(tenant_id, profile_key, checkpoint_generation, checkpoint_event_id, run_id, "
            "status, capability_versions, repair_cursor_revision_id, repair_active, repair_mode, "
            "repair_scan_complete, repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, audit_assertion_revision_id, "
            "audit_competitor_cursor_revision_id, updated_at) "
            "VALUES (?, 'legacy-profile', 12, 'legacy-event', 'legacy-run', 'complete', '{}', "
            "?, 1, 'current_scan', 1, 11, 'legacy-repair-event', ?, ?, ?, ?, "
            "'2026-07-27T00:00:00Z')",
            (
                TENANT,
                legacy_revision_id,
                legacy_revision_id,
                legacy_assertion_id,
                legacy_revision_id,
                legacy_revision_id,
            ),
        )
        await storage.db.execute(
            "INSERT INTO semantic_maintenance_reports "
            "(tenant_id, report_id, report_kind, evidence_digest, status, evidence_mapping, "
            "created_at, updated_at) VALUES (?, 'legacy-report', 'contradiction_candidate', "
            "'legacy-digest', 'review_required', ?, '2026-07-27T00:00:00Z', "
            "'2026-07-27T00:00:00Z')",
            (
                TENANT,
                json.dumps(
                    {
                        "assertion_id": legacy_assertion_id,
                        "revision_id": legacy_revision_id,
                        "content": legacy_content,
                    }
                ),
            ),
        )
        await storage.db.execute(
            "INSERT INTO semantic_maintenance_leases "
            "(tenant_id, holder_id, fencing_token, expires_at, updated_at) "
            "VALUES (?, 'legacy-worker', 4, 9999999999, '2026-07-27T00:00:00Z')",
            (TENANT,),
        )
        await storage.db.execute(
            "INSERT INTO semantic_maintenance_state "
            "(tenant_id, profile_key, checkpoint_generation, checkpoint_event_id, run_id, "
            "status, capability_versions, repair_cursor_revision_id, repair_active, repair_mode, "
            "repair_scan_complete, repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, audit_assertion_revision_id, "
            "audit_competitor_cursor_revision_id, updated_at) "
            "VALUES (?, 'unaffected-profile', 3, 'unaffected-event', 'unaffected-run', "
            "'complete', '{}', NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL, "
            "'2026-07-27T00:00:00Z')",
            (unaffected_tenant,),
        )
        await storage.db.execute(
            "INSERT INTO semantic_maintenance_reports "
            "(tenant_id, report_id, report_kind, evidence_digest, status, evidence_mapping, "
            "created_at, updated_at) VALUES (?, 'unaffected-report', 'contradiction_candidate', "
            "'unaffected-digest', 'review_required', '{\"safe\":true}', "
            "'2026-07-27T00:00:00Z', '2026-07-27T00:00:00Z')",
            (unaffected_tenant,),
        )
        await storage.db.execute(
            "INSERT INTO semantic_projection_erasure_outbox "
            "(event_id, tenant_id, operation, generation, created_at) "
            "VALUES ('legacy-erasure-event', ?, 'erased', 12, '2026-07-27T00:00:00Z')",
            (TENANT,),
        )
        await storage.db.execute(
            "DELETE FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_ERASURE_REDACTION_SCHEMA_VERSION,),
        )
    finally:
        await storage.close()

    restarted = AsyncStorage(
        str(db_path),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await restarted.initialize()
    try:
        assert await restarted.db.fetchval(
            "SELECT COUNT(*) FROM semantic_maintenance_reports WHERE tenant_id = ?",
            (TENANT,),
        ) == 0
        assert await restarted.db.fetchone(
            "SELECT checkpoint_generation, checkpoint_event_id, status, "
            "repair_cursor_revision_id, repair_active, repair_mode, repair_scan_complete, "
            "repair_checkpoint_generation, repair_checkpoint_event_id, "
            "repair_reconcile_cursor_derivation_id, audit_assertion_id, "
            "audit_assertion_revision_id, audit_competitor_cursor_revision_id "
            "FROM semantic_maintenance_state WHERE tenant_id = ?",
            (TENANT,),
        ) == (0, None, "partial", None, 0, None, 0, None, None, None, None, None, None)
        assert await restarted.db.fetchone(
            "SELECT holder_id, fencing_token, expires_at "
            "FROM semantic_maintenance_leases WHERE tenant_id = ?",
            (TENANT,),
        ) == ("schema-erasure-redaction", 5, 0.0)
        assert await restarted.db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            (_SEMANTIC_MAINTENANCE_ERASURE_REDACTION_SCHEMA_VERSION,),
        ) == (1,)
        assert await restarted.db.fetchone(
            "SELECT checkpoint_generation, checkpoint_event_id, status "
            "FROM semantic_maintenance_state WHERE tenant_id = ?",
            (unaffected_tenant,),
        ) == (3, "unaffected-event", "complete")
        assert await restarted.db.fetchone(
            "SELECT evidence_mapping FROM semantic_maintenance_reports "
            "WHERE tenant_id = ? AND report_id = 'unaffected-report'",
            (unaffected_tenant,),
        ) == ('{"safe":true}',)
        for table_name in (
            "semantic_maintenance_state",
            "semantic_maintenance_runs",
            "semantic_maintenance_leases",
            "semantic_maintenance_reports",
        ):
            rows = await restarted.db.fetchall(f"SELECT * FROM {table_name}")
            serialized = repr(rows)
            assert legacy_assertion_id not in serialized
            assert legacy_revision_id not in serialized
            assert legacy_content not in serialized
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_direct_validation_quarantine_is_refused_outside_the_governed_repair_path() -> None:
    storage = await _storage()
    try:
        root = direct("validation-root")
        await storage.put_assertion(root, source_occurrences=(source("source-1"),))
        child = derived("validation-child", root.revision_id)
        await storage.put_assertion(child)
        with pytest.raises(RuntimeError, match="Direct validation quarantine is unavailable"):
            await storage.quarantine_assertion_for_validation(
                root.assertion_id,
                root.revision_id,
                report_id="validation-report-1",
            )

        assert await storage.get_assertion(root.assertion_id) == root
        assert await storage.get_assertion(child.assertion_id) == child
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_governed_assertion_write_persists_a_pinned_validation_report_before_acceptance() -> None:
    storage = await _storage()
    try:
        assertion = direct("governed-validation-write")

        result = await storage.put_validated_assertion(
            assertion,
            source_occurrences=(source("source-1"),),
        )

        assert result.accepted is True
        assert result.write is not None
        assert result.report.conforms is True
        assert await storage.get_assertion(assertion.assertion_id) == assertion
        reports = await storage.semantic_validation_service().reports.list(assertion_id=assertion.assertion_id)
        assert reports == [result.report]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_derived_supports_must_be_current_active_and_eligible() -> None:
    storage = await _storage()
    try:
        root = direct("support-root", source_id="support-source")
        await storage.put_assertion(root, source_occurrences=(source("support-source"),))

        await storage.db.execute(
            "UPDATE semantic_projection_eligibility SET eligible = 0 "
            "WHERE tenant_id = ? AND revision_id = ?",
            (TENANT, root.revision_id),
        )
        with pytest.raises(AssertionStoreError, match="current, active, and eligible"):
            await storage.put_assertion(derived("ineligible-support", root.revision_id))

        await storage.db.execute(
            "UPDATE semantic_projection_eligibility SET eligible = 1 "
            "WHERE tenant_id = ? AND revision_id = ?",
            (TENANT, root.revision_id),
        )
        await storage.retract_assertion(root.assertion_id, root.revision_id)
        with pytest.raises(AssertionStoreError, match="current, active, and eligible"):
            await storage.put_assertion(derived("retracted-support", root.revision_id))

        active = direct("superseded-support", value="old", source_id="superseded-source")
        await storage.put_assertion(active, source_occurrences=(source("superseded-source"),))
        replacement = direct(
            "superseded-support-replacement", value="new", source_id="replacement-source",
        )
        await storage.supersede_assertion(
            active.revision_id,
            replacement,
            source_occurrences=(source("replacement-source"),),
        )
        with pytest.raises(AssertionStoreError, match="current, active, and eligible"):
            await storage.put_assertion(derived("superseded-support-derived", active.revision_id))
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_lifecycle_closure_preserves_independent_current_replacements() -> None:
    storage = await _storage()
    try:
        root = direct("lineage-root", source_id="lineage-root-source")
        await storage.put_assertion(root, source_occurrences=(source("lineage-root-source"),))
        derived_child = derived("lineage-child", root.revision_id)
        await storage.put_assertion(derived_child)

        independent = direct(
            "lineage-independent", value="independent", source_id="lineage-independent-source",
        )
        replacement = await storage.supersede_assertion(
            derived_child.revision_id,
            independent,
            source_occurrences=(source("lineage-independent-source"),),
        )

        await storage.retract_assertion(root.assertion_id, root.revision_id)
        assert await storage.get_assertion(independent.assertion_id) == replacement.replacement

        erased = await storage.erase_assertion(root.assertion_id)
        assert derived_child.assertion_id in erased.erased_assertion_ids
        assert independent.assertion_id not in erased.erased_assertion_ids
        surviving = await storage.get_assertion(independent.assertion_id)
        assert surviving is not None
        assert surviving.revision_id == replacement.replacement.revision_id
        assert surviving.supersedes_revision_id is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_erasure_scrubs_historical_lineage_but_preserves_same_identity_direct_replacement() -> None:
    storage = await _storage()
    db = storage.db
    try:
        root = direct("same-identity-root", source_id="same-identity-root-source")
        await storage.put_assertion(
            root,
            source_occurrences=(source("same-identity-root-source"),),
        )
        derived_child = derived("same-identity-derived", root.revision_id)
        await storage.put_assertion(derived_child)

        replacement_mapping = derived_child.to_mapping()
        replacement_mapping["revision_id"] = "same-identity-direct"
        replacement_mapping["lineage"] = DirectLineage(
            ("same-identity-direct-source",)
        ).to_mapping()
        replacement_mapping["epistemic_state"] = EpistemicState.REPORTED.value
        direct_replacement = Assertion.from_mapping(replacement_mapping)
        assert direct_replacement.assertion_id == derived_child.assertion_id

        supersession = await storage.supersede_assertion(
            derived_child.revision_id,
            direct_replacement,
            source_occurrences=(source("same-identity-direct-source"),),
        )

        erased = await storage.erase_assertion(root.assertion_id)

        assert root.assertion_id in erased.erased_assertion_ids
        assert derived_child.assertion_id not in erased.erased_assertion_ids
        assert derived_child.revision_id in erased.erased_revision_ids
        assert supersession.predecessor.revision_id in erased.erased_revision_ids
        assert supersession.replacement.revision_id not in erased.erased_revision_ids
        surviving = await storage.get_assertion(derived_child.assertion_id)
        assert surviving is not None
        assert surviving.revision_id == supersession.replacement.revision_id
        assert surviving.supersedes_revision_id is None
        assert await storage.get_assertion_revision(derived_child.revision_id) is None
        assert await storage.get_assertion_revision(supersession.predecessor.revision_id) is None
        assert await db.fetchall(
            "SELECT revision_id FROM semantic_assertion_revisions "
            "WHERE tenant_id = ? AND assertion_id = ?",
            (TENANT, derived_child.assertion_id),
        ) == [(supersession.replacement.revision_id,)]
        assert await db.fetchval(
            "SELECT COUNT(*) FROM semantic_derivation_inputs WHERE tenant_id = ?",
            (TENANT,),
        ) == 0
        assert await db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertion_operations WHERE tenant_id = ?",
            (TENANT,),
        ) == 0
        surviving_row = await db.fetchone(
            "SELECT supersedes_revision_id, assertion_mapping "
            "FROM semantic_assertion_revisions WHERE tenant_id = ? AND revision_id = ?",
            (TENANT, supersession.replacement.revision_id),
        )
        assert surviving_row is not None
        assert surviving_row[0] is None
        assert Assertion.from_mapping(
            json.loads(surviving_row[1])
        ).supersedes_revision_id is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_erasure_sanitizes_historical_direct_replacement_references() -> None:
    """A second replacement must not leave the first pointing into erasure."""
    storage = await _storage()
    db = storage.db
    try:
        root = direct("multi-replacement-root", source_id="multi-replacement-root-source")
        await storage.put_assertion(
            root,
            source_occurrences=(source("multi-replacement-root-source"),),
        )
        derived_child = derived("multi-replacement-derived", root.revision_id)
        await storage.put_assertion(derived_child)

        first_mapping = derived_child.to_mapping()
        first_mapping["revision_id"] = "multi-replacement-direct-first"
        first_mapping["lineage"] = DirectLineage(
            ("multi-replacement-direct-first-source",)
        ).to_mapping()
        first_mapping["epistemic_state"] = EpistemicState.REPORTED.value
        first_direct = Assertion.from_mapping(first_mapping)
        first_supersession = await storage.supersede_assertion(
            derived_child.revision_id,
            first_direct,
            source_occurrences=(source("multi-replacement-direct-first-source"),),
        )

        second_mapping = first_direct.to_mapping()
        second_mapping["revision_id"] = "multi-replacement-direct-second"
        second_mapping["lineage"] = DirectLineage(
            ("multi-replacement-direct-second-source",)
        ).to_mapping()
        second_direct = Assertion.from_mapping(second_mapping)
        second_supersession = await storage.supersede_assertion(
            first_supersession.replacement.revision_id,
            second_direct,
            source_occurrences=(source("multi-replacement-direct-second-source"),),
        )

        assert first_supersession.replacement.supersedes_revision_id == first_supersession.predecessor.revision_id
        erased = await storage.erase_assertion(root.assertion_id)

        assert first_supersession.predecessor.revision_id in erased.erased_revision_ids
        assert first_supersession.replacement.revision_id not in erased.erased_revision_ids
        first_row = await db.fetchone(
            "SELECT supersedes_revision_id, assertion_mapping "
            "FROM semantic_assertion_revisions WHERE tenant_id = ? AND revision_id = ?",
            (TENANT, first_supersession.replacement.revision_id),
        )
        assert first_row is not None
        assert first_row[0] is None
        assert Assertion.from_mapping(json.loads(first_row[1])).supersedes_revision_id is None

        current = await storage.get_assertion(derived_child.assertion_id)
        assert current is not None
        assert current.revision_id == second_supersession.replacement.revision_id
        assert current.supersedes_revision_id == second_supersession.predecessor.revision_id
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_lifecycle_delete_keeps_an_ineligible_audit_shell() -> None:
    storage = await _storage()
    db = storage.db
    try:
        store = storage
        assertion = direct("delete-revision", source_id="delete-source")
        await store.put_assertion(assertion, source_occurrences=(source("delete-source"),))

        deleted = await store.delete_assertion(assertion.assertion_id, assertion.revision_id)

        assert deleted.deleted.status.value == "deleted"
        assert await store.get_assertion(assertion.assertion_id) is None
        audit = await store.get_assertion(assertion.assertion_id, include_inactive=True)
        assert audit is not None and audit.status.value == "deleted"
        assert await db.fetchval(
            "SELECT eligible FROM semantic_projection_eligibility WHERE tenant_id = ? AND revision_id = ?",
            (TENANT, assertion.revision_id),
        ) == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_delete_replay_returns_created_invalidation_revisions() -> None:
    storage = await _storage()
    try:
        root = direct("delete-replay-root", source_id="delete-replay-source")
        await storage.put_assertion(root, source_occurrences=(source("delete-replay-source"),))
        dependent = derived("delete-replay-dependent", root.revision_id)
        await storage.put_assertion(dependent)

        first = await storage.delete_assertion(
            root.assertion_id,
            root.revision_id,
            operation_id="delete-replay",
        )
        replay = await storage.delete_assertion(
            root.assertion_id,
            root.revision_id,
            operation_id="delete-replay",
        )

        assert replay.idempotent is True
        assert replay.invalidated == first.invalidated
        assert [item.status for item in replay.invalidated] == [AssertionStatus.RETRACTED]
        assert replay.invalidated[0].revision_id != dependent.revision_id
        assert replay.invalidated_revision_ids == (root.revision_id, dependent.revision_id)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_dependent_invalidation_order_is_stable() -> None:
    storage = await _storage()
    try:
        root = direct("ordered-root", source_id="ordered-source")
        await storage.put_assertion(root, source_occurrences=(source("ordered-source"),))
        later = derived("ordered-z", root.revision_id, marker="zDerived")
        earlier = derived("ordered-a", root.revision_id, marker="aDerived")
        await storage.put_assertion(later)
        await storage.put_assertion(earlier)

        deleted = await storage.delete_assertion(root.assertion_id, root.revision_id)
        expected_dependents = sorted(
            (later, earlier), key=lambda assertion: assertion.assertion_id,
        )

        assert [item.assertion_id for item in deleted.invalidated] == [
            item.assertion_id for item in expected_dependents
        ]
        assert deleted.invalidated_revision_ids == (
            root.revision_id,
            *(item.revision_id for item in expected_dependents),
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_erasure_replay_survives_restart_without_retaining_semantic_ids(tmp_path) -> None:
    db_path = tmp_path / "erasure-replay.db"
    storage = AsyncStorage(
        str(db_path),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await storage.initialize()
    try:
        root = direct("restart-root", source_id="restart-source")
        await storage.put_assertion(root, source_occurrences=(source("restart-source"),))
        dependent = derived("restart-dependent", root.revision_id)
        await storage.put_assertion(dependent)
        erased = await storage.erase_assertion(root.assertion_id, operation_id="erasure-replay")
    finally:
        await storage.close()

    restarted = AsyncStorage(
        str(db_path),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await restarted.initialize()
    try:
        replay = await restarted.erase_assertion(root.assertion_id, operation_id="erasure-replay")

        assert replay.idempotent is True
        assert replay.erased_assertion_ids == ()
        assert replay.erased_revision_ids == ()
        assert replay.generation == erased.generation
        assert await restarted.db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertion_revisions WHERE tenant_id = ?",
            (TENANT,),
        ) == 0
        assert await restarted.db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertion_erasure_receipts WHERE tenant_id = ?",
            (TENANT,),
        ) == 1
        receipt_columns = await restarted.db.fetchall(
            "SELECT name FROM pragma_table_info('semantic_assertion_erasure_receipts')",
        )
        assert {column[0] for column in receipt_columns}.isdisjoint(
            {"assertion_id", "revision_id", "receipt"}
        )
        persisted = await restarted.db.fetchall(
            "SELECT * FROM semantic_assertion_erasure_receipts WHERE tenant_id = ?",
            (TENANT,),
        )
        assert root.assertion_id not in repr(persisted)
        assert root.revision_id not in repr(persisted)
        assert dependent.assertion_id not in repr(persisted)
        assert dependent.revision_id not in repr(persisted)
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_legacy_request_key_tombstone_with_matching_fence_fails_closed() -> None:
    """An unreleased v4 tombstone remains terminal without being trusted."""
    from kestrel_sovereign.storage.async_assertion_store import (
        _erased_operation_key,
        _erased_operation_request_key,
        _legacy_erasure_assertion_key,
    )

    storage = await _storage()
    try:
        governed = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        saved = await governed.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-request-key-region",
            confidence=0.9,
            invocation_id="legacy-request-key-save",
        )
        operation_row = await storage.db.fetchone(
            "SELECT operation, request_digest "
            "FROM semantic_assertion_operations "
            "WHERE tenant_id = ? AND operation_id = ?",
            (TENANT, saved.operation_id),
        )
        await governed.erase_assertion(
            saved.assertion_id,
            operation_id="legacy-request-key-erasure",
        )
        erasure_row = await storage.db.fetchone(
            "SELECT request_digest, generation "
            "FROM semantic_assertion_erasure_receipts "
            "WHERE tenant_id = ?",
            (TENANT,),
        )
        legacy_request_key = _erased_operation_request_key(
            saved.operation_id,
            str(operation_row[0]),
            str(operation_row[1]),
        )
        await storage.db.execute(
            "UPDATE semantic_assertion_erased_operation_tombstones "
            "SET request_key = ? "
            "WHERE tenant_id = ? AND operation_key = ?",
            (
                legacy_request_key,
                TENANT,
                _erased_operation_key(saved.operation_id),
            ),
        )
        await storage.db.execute(
            "INSERT INTO semantic_assertion_legacy_erasure_fences "
            "(tenant_id, assertion_key, generation, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                TENANT,
                _legacy_erasure_assertion_key(str(erasure_row[0])),
                int(erasure_row[1]),
                "2026-07-27T00:00:00Z",
            ),
        )

        replay = await governed.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-request-key-region",
            confidence=0.9,
            invocation_id="legacy-request-key-save",
        )
        assert replay.saved is False
        assert replay.idempotent is True
        assert replay.validation_disposition == "erased:terminal"
        assert replay.assertion_id is None
        assert replay.revision_id is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_v3_post_erasure_upgrade_blocks_lost_save_and_append_replays(
    tmp_path,
) -> None:
    """Opaque v3 erasure residue fences only the erased semantic identity."""
    db_path = tmp_path / "v3-post-erasure-upgrade.db"
    raw = AsyncStorage(
        str(db_path),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await raw.initialize()
    try:
        governed = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        original = await governed.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-erased-region",
            confidence=0.9,
            invocation_id="legacy-erased-original",
        )
        appended = await governed.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-erased-region",
            confidence=0.9,
            invocation_id="legacy-erased-append",
        )
        await governed.erase_assertion(
            original.assertion_id,
            operation_id="legacy-v3-erasure",
        )

        # Reproduce a durable v3 post-erasure database: the opaque erasure
        # receipt survived, while every source operation receipt disappeared
        # before per-operation tombstones or per-identity fences existed.
        await raw.db.execute(
            "DROP TABLE semantic_assertion_erased_operation_tombstones",
        )
        await raw.db.execute(
            "DROP TABLE semantic_assertion_legacy_erasure_fences",
        )
        await raw.db.execute(
            "DELETE FROM semantic_schema_migrations WHERE version = ?",
            ("semantic_assertion_store_v5",),
        )
        await raw.db.execute(
            "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
            ("semantic_assertion_store_v3",),
        )
    finally:
        await raw.close()

    restarted_raw = AsyncStorage(
        str(db_path),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await restarted_raw.initialize()
    try:
        restarted = PrivacyEnforcingStorage(
            restarted_raw,
            PrivacyMode.NORMAL,
        )
        old_save = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-erased-region",
            confidence=0.9,
            invocation_id="legacy-erased-original",
        )
        old_append = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-erased-region",
            confidence=0.9,
            invocation_id="legacy-erased-append",
        )
        for replay in (old_save, old_append):
            assert replay.saved is False
            assert replay.idempotent is True
            assert replay.validation_disposition == "erased:terminal"
            assert replay.assertion_id is None
            assert replay.revision_id is None
            assert replay.provenance_reference is None
        assert await restarted.query_assertions() == []
        assert await restarted_raw.db.fetchval(
            "SELECT COUNT(*) "
            "FROM semantic_assertion_erased_operation_tombstones "
            "WHERE tenant_id = ?",
            (TENANT,),
        ) == 2

        fence_rows = await restarted_raw.db.fetchall(
            "SELECT assertion_key, generation "
            "FROM semantic_assertion_legacy_erasure_fences "
            "WHERE tenant_id = ?",
            (TENANT,),
        )
        assert len(fence_rows) == 1
        encoded_fences = repr(fence_rows)
        for erased_value in (
            "legacy-erased-region",
            original.assertion_id,
            original.revision_id,
            appended.revision_id,
            original.operation_id,
            appended.operation_id,
        ):
            assert erased_value not in encoded_fences

        ambiguous_same_content = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legacy-erased-region",
            confidence=0.9,
            invocation_id="intentional-but-ambiguous-reteach",
        )
        assert ambiguous_same_content.saved is False
        assert (
            ambiguous_same_content.validation_disposition
            == "erased:terminal"
        )

        unrelated = await restarted.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="legitimate-post-upgrade-region",
            confidence=0.9,
            invocation_id="legitimate-post-upgrade",
        )
        assert unrelated.saved is True
        assert unrelated.idempotent is False
        assert await restarted.get_assertion(unrelated.assertion_id) is not None
    finally:
        await restarted_raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [
    AssertionStatus.SUPERSEDED,
    AssertionStatus.RETRACTED,
    AssertionStatus.QUARANTINED,
    AssertionStatus.DELETED,
])
async def test_put_rejects_terminal_initial_revisions(status: AssertionStatus) -> None:
    storage = await _storage()
    db = storage.db
    try:
        store = storage
        candidate = direct("terminal-revision", source_id="terminal-source")
        mapping = candidate.to_mapping()
        mapping["status"] = status.value
        if status is AssertionStatus.RETRACTED:
            mapping["epistemic_state"] = EpistemicState.RETRACTED.value
        terminal = Assertion.from_mapping(mapping)

        with pytest.raises(AssertionStoreError, match="initial active revision"):
            await store.put_assertion(terminal, source_occurrences=(source("terminal-source"),))
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_assertion_revisions") == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_put_rejects_an_initial_revision_with_a_predecessor() -> None:
    storage = await _storage()
    db = storage.db
    try:
        store = storage
        mapping = direct("predecessor-revision", source_id="predecessor-source").to_mapping()
        mapping["supersedes_revision_id"] = "prior-revision"
        candidate = Assertion.from_mapping(mapping)

        with pytest.raises(AssertionStoreError, match="initial active revision"):
            await store.put_assertion(candidate, source_occurrences=(source("predecessor-source"),))
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_assertion_revisions") == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_migration_is_idempotent_and_failed_write_rolls_back() -> None:
    storage = await _storage()
    db = storage.db
    try:
        await migrate_semantic_assertion_store(db)
        await migrate_semantic_assertion_store(db)
        store = storage
        bad = derived("bad-derived", "missing-input")
        with pytest.raises(TenantIsolationError):
            await store.put_assertion(bad)
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_assertion_revisions") == 0
        assert await db.fetchval("SELECT COUNT(*) FROM semantic_assertion_operations") == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_inference_and_validation_schema_migrations_coexist_idempotently() -> None:
    """Independent schema markers preserve both durable semantic ledgers."""
    db = await AsyncDatabase.sqlite(":memory:")
    try:
        await migrate_semantic_assertion_store(db)
        await migrate_semantic_validation_reports(db)
        await migrate_semantic_assertion_store(db)
        await migrate_semantic_validation_reports(db)

        assert await db.table_exists("semantic_inference_derivations")
        assert await db.table_exists("semantic_inference_derivation_inputs")
        assert await db.table_exists("semantic_validation_reports")
        assert await db.table_exists("semantic_validation_results")
        assert await db.fetchall(
            "SELECT version FROM semantic_schema_migrations "
            "WHERE version IN (?, ?) ORDER BY version",
            ("semantic_assertion_store_v5", "semantic_validation_reports_v1"),
        ) == [
            ("semantic_assertion_store_v5",),
            ("semantic_validation_reports_v1",),
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v3_semantic_store_upgrades_with_erased_operation_tombstones() -> None:
    """Existing semantic databases receive the v5 blinded replay ledgers."""
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    db = AsyncDatabase(backend)
    try:
        await db.execute(
            "CREATE TABLE semantic_schema_migrations ("
            "version TEXT PRIMARY KEY, completed_at TEXT NOT NULL "
            "DEFAULT CURRENT_TIMESTAMP)",
        )
        await db.execute(
            "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
            ("semantic_assertion_store_v3",),
        )

        await migrate_semantic_assertion_store(db)

        assert await db.table_exists(
            "semantic_assertion_erased_operation_tombstones"
        )
        assert await db.table_exists(
            "semantic_assertion_legacy_erasure_fences"
        )
        assert await db.fetchall(
            "SELECT version FROM semantic_schema_migrations ORDER BY version",
        ) == [
            ("semantic_assertion_store_v3",),
            ("semantic_assertion_store_v5",),
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v3_migration_rejects_non_integral_legacy_generation() -> None:
    """A decimal ledger ordinal must not be truncated into an authenticated fence."""
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    db = AsyncDatabase(backend)
    try:
        await db.execute(
            "CREATE TABLE semantic_schema_migrations ("
            "version TEXT PRIMARY KEY, completed_at TEXT NOT NULL "
            "DEFAULT CURRENT_TIMESTAMP)",
        )
        await db.execute(
            "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
            ("semantic_assertion_store_v3",),
        )
        # REAL is deliberate: malformed pre-release schemas can contain a
        # numeric value that SQLite would otherwise let int() silently truncate.
        await db.execute(
            "CREATE TABLE semantic_assertion_erasure_receipts ("
            "tenant_id TEXT NOT NULL, operation_id TEXT NOT NULL, "
            "request_digest TEXT NOT NULL, generation REAL NOT NULL, "
            "created_at TEXT NOT NULL, PRIMARY KEY (tenant_id, operation_id))"
        )
        await db.execute(
            "INSERT INTO semantic_assertion_erasure_receipts "
            "(tenant_id, operation_id, request_digest, generation, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                TENANT,
                "malformed-generation",
                "a" * 64,
                1.5,
                "2026-07-26T14:02:11Z",
            ),
        )

        with pytest.raises(
            TransactionError,
            match="malformed opaque state",
        ) as error:
            await migrate_semantic_assertion_store(db)

        assert isinstance(error.value.__cause__, ValueError)
        assert await db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            ("semantic_assertion_store_v5",),
        ) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_scrubs_legacy_erasure_receipt_identifiers() -> None:
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    db = AsyncDatabase(backend)
    try:
        await db.execute(
            "CREATE TABLE semantic_assertion_erasure_receipts ("
            "tenant_id TEXT NOT NULL, operation_id TEXT NOT NULL, "
            "request_digest TEXT NOT NULL, receipt TEXT NOT NULL, "
            "created_at TEXT NOT NULL, PRIMARY KEY (tenant_id, operation_id))"
        )
        await db.execute(
            "INSERT INTO semantic_assertion_erasure_receipts "
            "(tenant_id, operation_id, request_digest, receipt, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                TENANT,
                "old-erasure",
                "digest",
                '{"assertion_ids":["erased-assertion"],"generation":7,'
                '"revision_ids":["erased-revision"]}',
                "2026-07-26T14:02:11Z",
            ),
        )

        await migrate_semantic_assertion_store(db)

        assert await db.fetchone(
            "SELECT receipt, generation FROM semantic_assertion_erasure_receipts "
            "WHERE tenant_id = ?",
            (TENANT,),
        ) == ("{}", 7)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_relational_constraints_reject_invalid_projection_state() -> None:
    db = await AsyncDatabase.sqlite(":memory:")
    try:
        with pytest.raises(QueryError):
            await db.execute(
                "INSERT INTO semantic_projection_outbox "
                "(event_id, tenant_id, assertion_id, revision_id, operation, generation, eligible, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("bad-event", TENANT, "assertion", "revision", "accepted", 0, 2, "2026-07-26T14:02:11Z"),
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_rolls_back_its_partial_schema_on_ddl_failure() -> None:
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    db = AsyncDatabase(backend)
    try:
        # A deliberately incompatible legacy name makes the later index DDL
        # fail. The migration's transaction must leave none of its preceding
        # new tables behind.
        await db.execute("CREATE TABLE semantic_assertions (legacy_value TEXT)")
        with pytest.raises(TransactionError):
            await migrate_semantic_assertion_store(db)
        assert await db.table_exists("semantic_assertions")
        assert not await db.table_exists("semantic_assertion_tenants")
        assert not await db.table_exists("semantic_assertion_revisions")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_privacy_wrapper_governs_assertions_and_graph_proxy_denies_new_surface(tmp_path) -> None:
    storage = AsyncStorage(
        str(tmp_path / "semantic.db"),
        agent_id=TENANT,
        _assertion_tenant_capability=_assertion_capability(TENANT),
    )
    await storage.initialize()
    try:
        assertion = direct("privacy-revision", source_id="privacy-source")
        ephemeral = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        isolated = PrivacyEnforcingStorage(storage, PrivacyMode.ISOLATED)
        with pytest.raises(PrivacyViolationError):
            await ephemeral.put_assertion(assertion, source_occurrences=(source("privacy-source"),))
        with pytest.raises(PrivacyViolationError):
            await isolated.put_assertion(assertion, source_occurrences=(source("privacy-source"),))
        appended = replace(
            assertion,
            revision_id="privacy-append-revision",
            lineage=DirectLineage(("privacy-source", "privacy-append-source")),
        )
        with pytest.raises(PrivacyViolationError):
            await ephemeral.append_assertion_source(
                assertion.revision_id,
                appended,
                source_occurrences=(source("privacy-append-source"),),
            )
        with pytest.raises(PrivacyViolationError):
            await isolated.append_assertion_source(
                assertion.revision_id,
                appended,
                source_occurrences=(source("privacy-append-source"),),
            )
        with pytest.raises(PrivacyViolationError, match="checkpoint"):
            await ephemeral.assertion_checkpoint()
        with pytest.raises(PrivacyViolationError, match="checkpoint"):
            await isolated.assertion_checkpoint()
        assert await storage.db.fetchval("SELECT COUNT(*) FROM semantic_assertion_revisions") == 0
        assert await storage.db.fetchval("SELECT COUNT(*) FROM semantic_assertion_tenants") == 0

        normal = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        with pytest.raises(PrivacyViolationError, match="refuses to forward"):
            normal.graph.put_assertion
        with pytest.warns(DeprecationWarning):
            guarded_db = normal.db
        with pytest.raises(PrivacyViolationError, match="not exposed"):
            guarded_db.backend
        with pytest.raises(PrivacyViolationError, match="Direct semantic assertion"):
            await guarded_db.fetchval("SELECT COUNT(*) FROM semantic_assertion_revisions")
        # A backend supplied through an untrusted construction path cannot
        # create a tenant-bound assertion authority from a string alone.
        forged = AsyncStorage.from_backend(storage.db.backend, agent_id=OTHER_TENANT)
        await forged.initialize()
        try:
            with pytest.raises(RuntimeError, match="agent-bound AsyncStorage"):
                await forged.put_assertion(
                    direct(
                        "forged-revision",
                        source_id="forged-source",
                        tenant=OTHER_TENANT,
                        owner=OTHER_TENANT,
                    ),
                    source_occurrences=(source("forged-source"),),
                )
        finally:
            # This borrowed backend is owned by ``storage`` and must remain
            # connected for its enclosing test, so the borrowed facade does
            # not own a close here.
            pass
        written = await normal.put_assertion(assertion, source_occurrences=(source("privacy-source"),))
        assert written.accepted is True
        assert written.write is not None
        assert written.write.assertion == assertion
        assert await normal.semantic_validation_service().reports.list(
            assertion_id=assertion.assertion_id
        ) == [written.report]

        replacement = direct(
            "privacy-supersession-revision",
            value="europe-west1",
            source_id="privacy-supersession-source",
        )
        superseded = await normal.supersede_assertion(
            assertion.revision_id,
            replacement,
            source_occurrences=(source("privacy-supersession-source"),),
        )
        assert superseded.accepted is True
        assert superseded.write is not None
        assert superseded.write.replacement.assertion_id == replacement.assertion_id
        assert await normal.semantic_validation_service().reports.list(
            assertion_id=replacement.assertion_id
        ) == [superseded.report]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_volatile_privacy_modes_cannot_read_populated_assertion_authority(mode) -> None:
    storage = await _storage()
    try:
        assertion = direct("volatile-read-revision", source_id="volatile-read-source")
        await storage.put_assertion(
            assertion,
            source_occurrences=(source("volatile-read-source"),),
        )
        wrapper = PrivacyEnforcingStorage(storage, mode)

        reads = (
            wrapper.get_assertion(assertion.assertion_id),
            wrapper.get_assertion_revision(assertion.revision_id),
            wrapper.query_assertions(),
            wrapper.list_assertion_revisions(assertion.assertion_id),
            wrapper.list_assertion_sources(assertion.assertion_id),
            wrapper.get_source_occurrence("volatile-read-source"),
            wrapper.get_derivation_inputs(assertion.revision_id),
            wrapper.assertion_checkpoint(),
            wrapper.assertion_changes_since(0),
            wrapper.assertion_inference_inputs(),
            wrapper.export_assertion_snapshot(),
        )
        for read in reads:
            with pytest.raises(PrivacyViolationError, match="volatile privacy modes"):
                await read
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_volatile_privacy_modes_block_lifecycle_results_with_durable_content(mode) -> None:
    storage = await _storage()
    try:
        root = direct("volatile-lifecycle-root", source_id="volatile-lifecycle-source")
        await storage.put_assertion(
            root,
            source_occurrences=(source("volatile-lifecycle-source"),),
        )
        dependent = derived("volatile-lifecycle-dependent", root.revision_id)
        await storage.put_assertion(dependent)
        wrapper = PrivacyEnforcingStorage(storage, mode)

        with pytest.raises(PrivacyViolationError, match="volatile privacy modes"):
            await wrapper.retract_assertion(root.assertion_id, root.revision_id)
        with pytest.raises(PrivacyViolationError, match="volatile privacy modes"):
            await wrapper.delete_assertion(root.assertion_id, root.revision_id)

        assert await storage.get_assertion(root.assertion_id) == root
        assert await storage.get_assertion(dependent.assertion_id) == dependent
    finally:
        await storage.close()
