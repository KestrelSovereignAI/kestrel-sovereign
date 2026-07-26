"""Canonical assertion persistence contracts on the SQLite backend."""

from __future__ import annotations

import copy
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
from kestrel_sovereign.storage.sqla.migrations import migrate_semantic_assertion_store
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
        checkpoint = await store.assertion_checkpoint()
        assert checkpoint.generation == 1
        assert checkpoint.latest_event_id == written.event_id
        assert [change.revision_id for change in await store.assertion_changes_since(0)] == [assertion.revision_id]
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
        assert await storage.get_assertion(independent.assertion_id) == replacement.replacement
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
        assert await storage.get_assertion(derived_child.assertion_id) == supersession.replacement
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
        assert written.assertion == assertion
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
