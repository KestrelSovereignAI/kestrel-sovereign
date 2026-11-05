"""Governed export/corpus lifecycle tests (#2831)."""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.identity.runtime_identity import AgentIdentity, load_agent_identity
from kestrel_sovereign.agent.invocation import invocation_scope, request_provenance
from kestrel_sovereign.agent.sleep import SleepMixin
from kestrel_sovereign.features.memory_agency.feature import MemoryAgencyFeature
from kestrel_sovereign.features.memory_agency.semantic_facts import _is_adapter_source
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.knowledge import (
    Assertion,
    AssertionQuery,
    DirectLineage,
    EpistemicState,
    GovernedArtifactError,
    GovernedArtifactConsumerAuthentication,
    GovernedArtifactDeletionOwner,
    GovernedArtifactDeletionProof,
    GovernedArtifactKind,
    GovernedArtifactLineage,
    GovernedArtifactRegistration,
    GovernedCorpusPolicy,
    IRI,
    Literal,
    OntologyRef,
    SourceOccurrence,
    SourceReferenceKind,
    Visibility,
    XSD_STRING,
    source_reference_segment,
)
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import (
    PrivacyEnforcingStorage,
    PrivacyViolationError,
)
from kestrel_sovereign.privacy import PrivacyMode
from tests.utils.live_turn import TurnBoundAgent, live_turn


async def _open_storage(db_path: str, tenant_id: str, capability) -> AsyncStorage:
    instance = AsyncStorage(
        db_path, agent_id=tenant_id, _assertion_tenant_capability=capability,
    )
    await instance.initialize()
    return instance


@pytest.fixture
async def storage(tmp_path):
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Artifact lifecycle test",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity: AgentIdentity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(credentials.agent_did, identity)
    instance = await _open_storage(":memory:", credentials.agent_did, capability)
    try:
        yield instance
    finally:
        await instance.close()


def _source() -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id="artifact-source", source_kind="conversation",
        locator="conversation:artifact-source", received_at="2026-07-30T00:00:00Z",
        content_digest="sha256:artifact-source", actor="operator", selector="body",
    )


def _assertion(tenant_id: str, revision_id: str = "artifact-revision") -> Assertion:
    return Assertion(
        tenant_id=tenant_id, owning_agent_id=tenant_id,
        subject=IRI(f"urn:kestrel:agent:{tenant_id}:principal:user"),
        predicate=IRI("https://kestrel.ai/vocab/artifactTest"),
        object=Literal("active", XSD_STRING), revision_id=revision_id,
        confidence=Decimal("0.9"), confidence_method="test", confidence_basis="test",
        epistemic_state=EpistemicState.REPORTED, asserted_at="2026-07-30T00:00:00Z",
        ontology_version=OntologyRef("test", "1", "sha256:test", "semantic-kb-v1"),
        lineage=DirectLineage(("artifact-source",)), privacy_classification="normal",
        release_policy_reference="policy:private-v1",
    )


def _public_key(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()


def _authentication(
    private_key: Ed25519PrivateKey, tenant_id: str, *, consumer_id: str = "parametric-self-test",
    nonce: str | None = None, issued_at: str | None = None,
) -> GovernedArtifactConsumerAuthentication:
    authentication = GovernedArtifactConsumerAuthentication(
        consumer_id, "parametric-self-key-v1", nonce or str(uuid4()),
        issued_at or datetime.now(timezone.utc).isoformat(), "0" * 128,
    )
    return replace(
        authentication,
        signature=private_key.sign(authentication.signable_bytes(tenant_id)).hex(),
    )


def _owner(
    private_key: Ed25519PrivateKey, deleted: list[str],
) -> GovernedArtifactDeletionOwner:
    async def delete_artifact(lease):
        deleted.append(lease.artifact_key)
        proof = GovernedArtifactDeletionProof(
            datetime.now(timezone.utc).isoformat(), "0" * 128,
        )
        return replace(
            proof, signature=private_key.sign(proof.signable_bytes(lease)).hex()
        )

    return GovernedArtifactDeletionOwner(
        "parametric-self-test", "parametric-self-key-v1", delete_artifact
    )


def _registration(
    assertion: Assertion, generation: int, *, private_key: Ed25519PrivateKey,
    artifact_id: str | None = None,
) -> GovernedArtifactRegistration:
    return GovernedArtifactRegistration(
        artifact_id=artifact_id or str(uuid4()), kind=GovernedArtifactKind.CORPUS_MANIFEST,
        tenant_id=assertion.tenant_id, consumer_id="parametric-self-test",
        consumer_key_id="parametric-self-key-v1",
        consumer_public_key=_public_key(private_key),
        checkpoint_generation=generation, policy_pin="a" * 64,
        capability_pins={"semantic-kb-v1": "b" * 64},
        lineage=(GovernedArtifactLineage(assertion.assertion_id, assertion.revision_id),),
        retention_expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        artifact_digest="c" * 64,
    )


@pytest.mark.asyncio
async def test_registered_artifact_is_generation_fenced_and_erasure_leaves_only_blinded_revocation(storage: AsyncStorage) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id)
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    registration = _registration(assertion, checkpoint.generation, private_key=private_key)
    registered = await storage._assertion_store().register_governed_artifact(registration)
    with pytest.raises(GovernedArtifactError, match="different public key"):
        await storage._assertion_store().register_governed_artifact(
            _registration(
                assertion, checkpoint.generation,
                private_key=Ed25519PrivateKey.generate(),
            )
        )
    assert registration.artifact_id not in registered.receipt_digest
    assert await storage.consume_governed_semantic_artifact(
        registration.artifact_id, expected_generation=checkpoint.generation,
    ) == registered

    erased = await storage.erase_assertion(assertion.assertion_id)
    observation = await storage.governed_semantic_artifact_erasure_observation(
        expected_generation=erased.generation,
    )
    assert observation.governed_corpus == 0
    assert observation.pending_revocations == 1
    with pytest.raises(GovernedArtifactError):
        await storage.consume_governed_semantic_artifact(
            registration.artifact_id, expected_generation=erased.generation,
        )

    rows = await storage.db.fetchall(
        "SELECT artifact_id FROM semantic_governed_artifacts WHERE tenant_id = ? "
        "UNION ALL SELECT artifact_id FROM semantic_governed_artifact_lineage WHERE tenant_id = ?",
        (storage.agent_id, storage.agent_id),
    )
    assert rows == []
    residue = await storage.db.fetchall(
        "SELECT artifact_key FROM semantic_governed_artifact_revocations WHERE tenant_id = ?",
        (storage.agent_id,),
    )
    assert residue == [(registered.artifact_key,)]

    authentication = _authentication(private_key, storage.agent_id)
    deleted: list[str] = []
    intruder_key = Ed25519PrivateKey.generate()
    expired_authentication = _authentication(
        private_key,
        storage.agent_id,
        issued_at=(datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
    )
    with pytest.raises(GovernedArtifactError, match="expired"):
        await storage.claim_governed_semantic_artifact_revocation(
            expired_authentication
        )
    with pytest.raises(GovernedArtifactError):
        await storage.claim_governed_semantic_artifact_revocation(
            _authentication(intruder_key, storage.agent_id)
        )
    lease = await storage.claim_governed_semantic_artifact_revocation(authentication)
    assert lease is not None
    owner = _owner(private_key, deleted)
    proof = await owner.delete_artifact(lease)
    forged = replace(
        proof, signature=intruder_key.sign(proof.signable_bytes(lease)).hex()
    )
    with pytest.raises(GovernedArtifactError, match="signature"):
        await storage.acknowledge_governed_semantic_artifact_revocation(
            lease, forged
        )
    acknowledged = await storage.acknowledge_governed_semantic_artifact_revocation(
        lease, proof
    )
    with pytest.raises(GovernedArtifactError):
        await storage.acknowledge_governed_semantic_artifact_revocation(
            lease, proof
        )
    assert deleted == [registered.artifact_key]
    assert acknowledged.artifact_key == registered.artifact_key
    assert (await storage.governed_semantic_artifact_erasure_observation(
        expected_generation=erased.generation,
    )).completed_revocations == 1


@pytest.mark.asyncio
async def test_registration_rejects_forged_stale_or_expired_lineage(storage: AsyncStorage) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id)
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    registration = _registration(assertion, checkpoint.generation, private_key=private_key)
    forged = GovernedArtifactRegistration(
        artifact_id=registration.artifact_id, kind=registration.kind, tenant_id=registration.tenant_id,
        consumer_id=registration.consumer_id, consumer_key_id=registration.consumer_key_id,
        consumer_public_key=registration.consumer_public_key,
        checkpoint_generation=registration.checkpoint_generation,
        policy_pin=registration.policy_pin, capability_pins=registration.capability_pins,
        lineage=(GovernedArtifactLineage(assertion.assertion_id, "forged-revision"),),
        retention_expires_at=registration.retention_expires_at, artifact_digest=registration.artifact_digest,
    )
    with pytest.raises(GovernedArtifactError, match="lineage"):
        await storage._assertion_store().register_governed_artifact(forged)

    expired = _registration(assertion, checkpoint.generation, private_key=private_key)
    expired = GovernedArtifactRegistration(
        artifact_id=expired.artifact_id, kind=expired.kind, tenant_id=expired.tenant_id,
        consumer_id=expired.consumer_id, consumer_key_id=expired.consumer_key_id,
        consumer_public_key=expired.consumer_public_key,
        checkpoint_generation=expired.checkpoint_generation,
        policy_pin=expired.policy_pin, capability_pins=expired.capability_pins,
        lineage=expired.lineage, retention_expires_at="2020-01-01T00:00:00Z",
        artifact_digest=expired.artifact_digest,
    )
    with pytest.raises(GovernedArtifactError, match="expiry"):
        await storage._assertion_store().register_governed_artifact(expired)

    await storage._assertion_store().register_governed_artifact(registration)
    await storage.retract_assertion(assertion.assertion_id, assertion.revision_id)
    with pytest.raises(GovernedArtifactError, match="resurrected"):
        await storage._assertion_store().register_governed_artifact(registration)


@pytest.mark.asyncio
async def test_pending_revocation_survives_restart_and_replays_only_to_its_consumer(tmp_path) -> None:
    private_key = Ed25519PrivateKey.generate()
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Artifact restart test",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(credentials.agent_did, identity)
    database_path = str(tmp_path / "artifacts.db")
    first = await _open_storage(database_path, credentials.agent_did, capability)
    assertion = _assertion(credentials.agent_did)
    try:
        await first.put_assertion(assertion, source_occurrences=(_source(),))
        checkpoint = await first.assertion_checkpoint()
        await first._assertion_store().register_governed_artifact(
            _registration(assertion, checkpoint.generation, private_key=private_key)
        )
        await first.erase_assertion(assertion.assertion_id)
    finally:
        await first.close()

    restarted = await _open_storage(database_path, credentials.agent_did, capability)
    try:
        with pytest.raises(GovernedArtifactError):
            await restarted.claim_governed_semantic_artifact_revocation(
                _authentication(Ed25519PrivateKey.generate(), credentials.agent_did)
            )
        authentication = _authentication(private_key, credentials.agent_did)
        await restarted.process_governed_semantic_artifact_revocation(
            authentication, _owner(private_key, []),
        )
        with pytest.raises(GovernedArtifactError, match="nonce"):
            await restarted.claim_governed_semantic_artifact_revocation(authentication)
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_concurrent_registration_and_erasure_never_leaves_a_servable_artifact(storage: AsyncStorage) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id)
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    registration = _registration(assertion, checkpoint.generation, private_key=private_key)
    registered, erased = await asyncio.gather(
        storage._assertion_store().register_governed_artifact(registration),
        storage.erase_assertion(assertion.assertion_id),
        return_exceptions=True,
    )
    assert not isinstance(erased, Exception)
    active = await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifacts WHERE tenant_id = ? AND state = 'active'",
        (storage.agent_id,),
    )
    assert active == 0
    if not isinstance(registered, Exception):
        with pytest.raises(GovernedArtifactError):
            await storage.consume_governed_semantic_artifact(
                registration.artifact_id, expected_generation=erased.generation,
            )


@pytest.mark.asyncio
async def test_expiry_sweep_revokes_without_a_consume_attempt(storage: AsyncStorage) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id)
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    registration = _registration(
        assertion, checkpoint.generation, private_key=private_key,
    )
    await storage._assertion_store().register_governed_artifact(registration)
    storage._artifact_clock = lambda: datetime.now(timezone.utc) + timedelta(hours=2)
    storage._assertion_store()._artifact_clock = storage._artifact_clock
    swept = await storage.sweep_expired_governed_semantic_artifacts()
    assert swept == 1
    assert await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifacts WHERE tenant_id = ?",
        (storage.agent_id,),
    ) == 0
    assert await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifact_revocations "
        "WHERE tenant_id = ? AND acknowledged_at IS NULL",
        (storage.agent_id,),
    ) == 1


@pytest.mark.asyncio
async def test_consume_time_expiry_commits_pending_deletion_before_rejecting(
    storage: AsyncStorage,
) -> None:
    """A serving attempt must not roll its own expiry revocation back."""
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id, "consume-expiry-revision")
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    registration = _registration(
        assertion, checkpoint.generation, private_key=private_key,
    )
    await storage._assertion_store().register_governed_artifact(registration)
    await storage.db.execute(
        "UPDATE semantic_governed_artifacts SET retention_expires_at = ? "
        "WHERE tenant_id = ? AND artifact_id = ?",
        (
            (datetime.now(timezone.utc) - timedelta(seconds=1))
            .isoformat().replace("+00:00", "Z"),
            storage.agent_id,
            registration.artifact_id,
        ),
    )

    with pytest.raises(GovernedArtifactError, match="retention expiry"):
        await storage.consume_governed_semantic_artifact(
            registration.artifact_id, expected_generation=checkpoint.generation,
        )

    assert await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifacts "
        "WHERE tenant_id = ? AND artifact_id = ?",
        (storage.agent_id, registration.artifact_id),
    ) == 0
    assert await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifact_revocations "
        "WHERE tenant_id = ? AND acknowledged_at IS NULL",
        (storage.agent_id,),
    ) == 1


@pytest.mark.asyncio
async def test_consume_time_expiry_revokes_only_the_exact_overlapping_artifact(
    storage: AsyncStorage,
) -> None:
    """Shared assertion lineage must not make one artifact's TTL contagious."""
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id, "overlapping-expiry-revision")
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    expired = _registration(
        assertion, checkpoint.generation, private_key=private_key,
    )
    live = _registration(
        assertion, checkpoint.generation, private_key=private_key,
    )
    await storage._assertion_store().register_governed_artifact(expired)
    await storage._assertion_store().register_governed_artifact(live)
    await storage.db.execute(
        "UPDATE semantic_governed_artifacts SET retention_expires_at = ? "
        "WHERE tenant_id = ? AND artifact_id = ?",
        (
            (datetime.now(timezone.utc) - timedelta(seconds=1))
            .isoformat().replace("+00:00", "Z"),
            storage.agent_id,
            expired.artifact_id,
        ),
    )

    with pytest.raises(GovernedArtifactError, match="retention expiry"):
        await storage.consume_governed_semantic_artifact(
            expired.artifact_id, expected_generation=checkpoint.generation,
        )
    assert await storage.consume_governed_semantic_artifact(
        live.artifact_id, expected_generation=checkpoint.generation,
    )
    assert await storage.db.fetchall(
        "SELECT artifact_id FROM semantic_governed_artifacts WHERE tenant_id = ?",
        (storage.agent_id,),
    ) == [(live.artifact_id,)]
    assert await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifact_revocations "
        "WHERE tenant_id = ? AND acknowledged_at IS NULL",
        (storage.agent_id,),
    ) == 1


@pytest.mark.asyncio
async def test_expiry_sweep_prunes_only_stale_consumer_authentication_nonces(
    storage: AsyncStorage,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id, "nonce-prune-revision")
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    registration = _registration(
        assertion, checkpoint.generation, private_key=private_key,
    )
    await storage._assertion_store().register_governed_artifact(registration)
    stale_nonce, fresh_nonce = str(uuid4()), str(uuid4())
    now = datetime.now(timezone.utc)
    await storage.db.execute(
        "INSERT INTO semantic_governed_artifact_auth_nonces "
        "(tenant_id, consumer_id, nonce, used_at) VALUES (?, ?, ?, ?), (?, ?, ?, ?)",
        (
            storage.agent_id, registration.consumer_id, stale_nonce,
            (now - timedelta(minutes=6)).isoformat().replace("+00:00", "Z"),
            storage.agent_id, registration.consumer_id, fresh_nonce,
            (now - timedelta(minutes=4)).isoformat().replace("+00:00", "Z"),
        ),
    )

    assert await storage.sweep_expired_governed_semantic_artifacts() == 0
    assert await storage.db.fetchall(
        "SELECT nonce FROM semantic_governed_artifact_auth_nonces "
        "WHERE tenant_id = ? ORDER BY nonce",
        (storage.agent_id,),
    ) == [(fresh_nonce,)]


@pytest.mark.asyncio
async def test_privacy_wrapper_blocks_artifact_persistence_and_serving_in_volatile_modes(storage: AsyncStorage) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id)
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    registration = _registration(
        assertion, checkpoint.generation, private_key=private_key,
    )
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
    with pytest.raises(PrivacyViolationError):
        await wrapper.export_assertion_snapshot(
            artifact_id=registration.artifact_id,
            consumer_id=registration.consumer_id,
            consumer_key_id=registration.consumer_key_id,
            consumer_public_key=registration.consumer_public_key,
            retention_seconds=60,
        )
    with pytest.raises(PrivacyViolationError):
        await wrapper.consume_governed_semantic_artifact(
            registration.artifact_id, expected_generation=checkpoint.generation
        )
    # Cleanup remains callable even in a volatile mode so a mode transition
    # cannot strand bytes written while persistence was allowed.
    assert await wrapper.sweep_expired_governed_semantic_artifacts() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("producer_name", "raw_method"),
    [
        ("export", "export_assertion_snapshot"),
        ("corpus", "governed_assertion_corpus_snapshot"),
        ("delta", "governed_assertion_corpus_changes_since"),
    ],
)
async def test_artifact_producer_lease_blocks_transition_and_releases_on_cancel(
    storage: AsyncStorage, monkeypatch, producer_name: str, raw_method: str,
) -> None:
    """No restrictive transition can bisect an admitted producer await."""
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    entered = asyncio.Event()
    never_release = asyncio.Event()

    async def blocked_producer(*args, **kwargs):
        entered.set()
        await never_release.wait()

    monkeypatch.setattr(storage, raw_method, blocked_producer)
    governance = {
        "artifact_id": str(uuid4()),
        "consumer_id": "lease-test",
        "consumer_key_id": "lease-test-key",
        "consumer_public_key": "0" * 64,
        "retention_seconds": 60,
    }
    if producer_name == "export":
        task = asyncio.create_task(wrapper.export_assertion_snapshot(**governance))
    elif producer_name == "corpus":
        task = asyncio.create_task(
            wrapper.governed_assertion_corpus_snapshot(
                policy=None, inference_profile=None, **governance,
            )
        )
    else:
        task = asyncio.create_task(
            wrapper.governed_assertion_corpus_changes_since(
                object(), policy=None, inference_profile=None, **governance,
            )
        )
    await entered.wait()
    with pytest.raises(PrivacyViolationError, match="artifact producer"):
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wrapper._active_semantic_artifact_producer_leases == 0
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.DEIDENTIFIED, PrivacyMode.ANONYMOUS])
async def test_privacy_modes_reject_all_durable_artifact_producers_without_rows(
    storage: AsyncStorage, mode: PrivacyMode
) -> None:
    wrapper = PrivacyEnforcingStorage(storage, mode)
    governance = {
        "artifact_id": str(uuid4()),
        "consumer_id": "privacy-test",
        "consumer_key_id": "privacy-key",
        "consumer_public_key": "0" * 64,
        "retention_seconds": 60,
    }
    with pytest.raises(PrivacyViolationError):
        await wrapper.export_assertion_snapshot(**governance)
    with pytest.raises(PrivacyViolationError):
        await wrapper.governed_assertion_corpus_snapshot(
            policy=None, inference_profile=None, **governance
        )
    with pytest.raises(PrivacyViolationError):
        await wrapper.governed_assertion_corpus_changes_since(
            object(), policy=None, inference_profile=None, **governance
        )
    for table in (
        "semantic_governed_artifacts",
        "semantic_governed_artifact_lineage",
        "semantic_governed_artifact_consumers",
        "semantic_governed_artifact_revocations",
        "semantic_governed_artifact_receipts",
    ):
        assert await storage.db.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE tenant_id = ?",
            (storage.agent_id,),
        ) == 0


@pytest.mark.asyncio
async def test_tombstone_delta_is_revoked_and_physically_deleted_after_erasure(
    storage: AsyncStorage,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id, "future-delta-tombstone-revision")
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    await storage.run_semantic_maintenance(None)
    capabilities = await storage.semantic_maintenance_capability_versions(None)
    policy = GovernedCorpusPolicy(
        policy_id="future-delta-erasure",
        policy_version="1",
        accepted_epistemic_states=(EpistemicState.REPORTED,),
        accepted_visibility=(Visibility.PRIVATE,),
        accepted_privacy_classifications=("normal",),
        accepted_consent_references=("policy:private-v1",),
        accepted_grounding_classes=("test",),
        accepted_source_kinds=("conversation",),
        accepted_ontology_pins=(assertion.ontology_version,),
        accepted_semantic_capability_versions=tuple(capabilities.items()),
    )
    producer = {
        "consumer_id": "parametric-self-test",
        "consumer_key_id": "parametric-self-key-v1",
        "consumer_public_key": _public_key(private_key),
        "retention_seconds": 300,
    }
    snapshot = await storage.governed_assertion_corpus_snapshot(
        policy=policy,
        inference_profile=None,
        artifact_id=str(uuid4()),
        **producer,
    )
    retraction = await storage.retract_assertion(
        assertion.assertion_id, assertion.revision_id
    )
    tombstone_revision_id = retraction.retracted[0].revision_id
    await storage.run_semantic_maintenance(None)
    delta_artifact_id = str(uuid4())
    delta = await storage.governed_assertion_corpus_changes_since(
        snapshot,
        policy=policy,
        inference_profile=None,
        artifact_id=delta_artifact_id,
        **producer,
    )
    assert any(
        item.assertion_id == assertion.assertion_id
        and item.revision_id == tombstone_revision_id
        for item in delta.tombstones
    )
    assert await storage.db.fetchall(
        "SELECT assertion_id, revision_id FROM semantic_governed_artifact_lineage "
        "WHERE tenant_id = ? AND artifact_id = ?",
        (storage.agent_id, delta_artifact_id),
    ) == [(assertion.assertion_id, tombstone_revision_id)]
    delivered = await storage.consume_governed_semantic_artifact(
        delta_artifact_id, expected_generation=delta.checkpoint.generation
    )
    await storage.erase_assertion(assertion.assertion_id)
    deleted: list[str] = []
    owner = _owner(private_key, deleted)
    while True:
        receipt = await storage.process_governed_semantic_artifact_revocation(
            _authentication(private_key, storage.agent_id), owner
        )
        if receipt is None:
            break
    assert delivered.artifact_key in deleted
    assert await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifact_revocations "
        "WHERE tenant_id = ? AND acknowledged_at IS NULL",
        (storage.agent_id,),
    ) == 0


_CORPUS_CONSUMER_PUBLIC_KEY = _public_key(Ed25519PrivateKey.generate())


async def _corpus_snapshot(storage: AsyncStorage, policy: GovernedCorpusPolicy):
    """One corpus snapshot for the same consumer key on every call."""
    return await storage.governed_assertion_corpus_snapshot(
        policy=policy,
        inference_profile=None,
        artifact_id=str(uuid4()),
        consumer_id="parametric-self-test",
        consumer_key_id="parametric-self-key-v1",
        consumer_public_key=_CORPUS_CONSUMER_PUBLIC_KEY,
        retention_seconds=300,
    )


@pytest.mark.asyncio
async def test_corpus_holds_out_a_session_named_in_stored_provenance(
    storage: AsyncStorage,
) -> None:
    exam_session = "dcd6f767-829d-4a7d-b8c7-d7a722e5de13"
    stored = []
    for name, segments in (
        ("exam", (source_reference_segment(SourceReferenceKind.SESSION, exam_session),)),
        ("everyday", ()),
    ):
        source = replace(
            _source(),
            source_occurrence_id=f"source:{name}",
            locator="#".join(("POST:/api/agent/invoke", *segments, f"note:{name}")),
        )
        assertion = replace(
            _assertion(storage.agent_id, f"{name}-revision"),
            object=Literal(f"{name}-answer", XSD_STRING),
            lineage=DirectLineage((source.source_occurrence_id,)),
            assertion_id=None,
        )
        await storage.put_assertion(assertion, source_occurrences=(source,))
        stored.append(assertion)
    await storage.run_semantic_maintenance(None)
    capabilities = await storage.semantic_maintenance_capability_versions(None)
    policy = GovernedCorpusPolicy(
        policy_id="exam-holdout",
        policy_version="1",
        accepted_epistemic_states=(EpistemicState.REPORTED,),
        accepted_visibility=(Visibility.PRIVATE,),
        accepted_privacy_classifications=("normal",),
        accepted_consent_references=("policy:private-v1",),
        accepted_grounding_classes=("test",),
        accepted_source_kinds=("conversation",),
        accepted_ontology_pins=(stored[0].ontology_version,),
        accepted_semantic_capability_versions=tuple(capabilities.items()),
    )

    unexcluded = await _corpus_snapshot(storage, policy)
    held_out = await _corpus_snapshot(
        storage, replace(policy, excluded_session_ids=(exam_session,))
    )

    assert {item.assertion.revision_id for item in unexcluded.examples} == {
        "exam-revision", "everyday-revision",
    }
    assert [item.assertion.revision_id for item in held_out.examples] == ["everyday-revision"]
    assert held_out.observability.excluded == {"explicitly_excluded": 1}
    assert all(
        "exam-answer" not in str(item.canonical_mapping()) for item in held_out.examples
    )


EXAM_SESSION = "dcd6f767-829d-4a7d-b8c7-d7a722e5de13"


def _fact_feature(storage: AsyncStorage):
    """``save_fact`` over the real privacy wrapper, for an agent with a real turn lifecycle."""
    governed = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    agent = TurnBoundAgent(storage=governed, did=storage.agent_id)
    feature = MemoryAgencyFeature(agent)
    feature.storage = governed
    return feature, agent, governed


async def _teach(feature: MemoryAgencyFeature, invocation_id: str, value: str) -> dict:
    """One ``save_fact`` call on an HTTP turn, as the invoke endpoint binds it."""
    provenance = request_provenance(
        actor="owner@example.test",
        source_kind="http_request",
        source_locator="POST:/api/agent/invoke",
    )
    with invocation_scope(invocation_id, provenance=provenance):
        result = await feature.save_fact(
            subject="user", predicate="preferred_deploy_region", value=value
        )
    assert result.data["saved"] is True, result
    return result.data


async def _taught_fact_policy(
    storage: AsyncStorage, saved: dict
) -> GovernedCorpusPolicy:
    """A corpus policy every class filter of which accepts a taught fact."""
    assertion = await storage.get_assertion(saved["assertion_id"])
    await storage.run_semantic_maintenance(None)
    capabilities = await storage.semantic_maintenance_capability_versions(None)
    return GovernedCorpusPolicy(
        policy_id="exam-holdout",
        policy_version="1",
        accepted_epistemic_states=(EpistemicState.REPORTED,),
        accepted_visibility=(Visibility.PRIVATE,),
        accepted_privacy_classifications=("normal",),
        accepted_consent_references=("policy:privacy:normal-v1",),
        accepted_grounding_classes=("explicit-tool-invocation",),
        accepted_source_kinds=("http_request",),
        accepted_ontology_pins=(assertion.ontology_version,),
        accepted_semantic_capability_versions=tuple(capabilities.items()),
    )


@pytest.mark.asyncio
async def test_save_fact_during_a_session_is_held_out_by_excluding_that_session(
    storage: AsyncStorage,
) -> None:
    """#3875: a fact taught during the exam session names it, so excluding it holds."""
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, EXAM_SESSION):
        saved = await _teach(feature, "exam-turn", "exam-answer-region")

    sources = await governed.list_assertion_revision_sources(saved["revision_id"])
    assert [source.references for source in sources] == [
        frozenset({(SourceReferenceKind.SESSION, EXAM_SESSION)})
    ]
    # The session segment does not hide the source from the adapter that owns it.
    assert _is_adapter_source(sources[0])

    policy = await _taught_fact_policy(storage, saved)
    unexcluded = await _corpus_snapshot(storage, policy)
    held_out = await _corpus_snapshot(
        storage, replace(policy, excluded_session_ids=(EXAM_SESSION,))
    )

    assert [item.assertion.revision_id for item in unexcluded.examples] == [
        saved["revision_id"]
    ]
    assert held_out.examples == ()
    assert held_out.observability.excluded == {"explicitly_excluded": 1}


PADDED_SESSION = " exam-session "


@pytest.mark.asyncio
async def test_save_fact_names_a_padded_session_exactly_as_its_rows_are_filed(
    storage: AsyncStorage,
) -> None:
    """A session key is named as filed, never trimmed.

    HTTP admission and the conversation store keep a caller's session key
    verbatim, surrounding whitespace included. A source naming a trimmed copy
    names a session no row is filed under, so excluding the session the rows
    are filed under would let the fact into the corpus.
    """
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, PADDED_SESSION):
        await governed.add_conversation(
            "user", "Remember my region.", session_id=PADDED_SESSION
        )
        saved = await _teach(feature, "padded-session-turn", "exam-answer-region")

    history = await governed.get_conversation_history(limit=10)
    assert {entry["metadata"]["session_id"] for entry in history} == {PADDED_SESSION}
    sources = await governed.list_assertion_revision_sources(saved["revision_id"])
    assert [source.references for source in sources] == [
        frozenset({(SourceReferenceKind.SESSION, PADDED_SESSION)})
    ]

    policy = await _taught_fact_policy(storage, saved)
    held_out = await _corpus_snapshot(
        storage, replace(policy, excluded_session_ids=(PADDED_SESSION,))
    )
    assert held_out.examples == ()
    assert held_out.observability.excluded == {"explicitly_excluded": 1}


@pytest.mark.asyncio
async def test_save_fact_names_a_session_only_from_its_own_live_turn(
    storage: AsyncStorage,
) -> None:
    """Off-turn work and a session-less turn name no session.

    The agent-global session belongs to whichever turn is live. A call that is
    not that turn, such as a scheduled tool, must not borrow it, and a turn
    without a session has its rows filed under sessions the conversation store
    derives row by row, so there is no one session to name.
    """
    feature, agent, governed = _fact_feature(storage)
    agent._active_session_id = EXAM_SESSION
    off_turn = await _teach(feature, "scheduled-call", "us-central1")
    with live_turn(agent, None):
        sessionless = await _teach(feature, "sessionless-turn", "europe-west4")

    for saved in (off_turn, sessionless):
        sources = await governed.list_assertion_revision_sources(saved["revision_id"])
        assert [source.references for source in sources] == [frozenset()]


@pytest.mark.asyncio
async def test_save_fact_names_the_session_its_turn_rows_are_filed_under(
    storage: AsyncStorage,
) -> None:
    """An older console echoes a session as its marker row id (#2012).

    The conversation store files that turn's rows under the marker's UUID, so
    the fact's source names the UUID, not the echoed row id.
    """
    feature, agent, governed = _fact_feature(storage)
    await governed.add_conversation(
        "system", "New conversation", metadata={"new_session": True}
    )
    marker_id = await storage.db.fetchval(
        "SELECT id FROM conversation_history WHERE agent_id = ?", (storage.agent_id,)
    )
    echoed = str(marker_id)
    with live_turn(agent, echoed):
        await governed.add_conversation("user", "Remember my region.", session_id=echoed)
        saved = await _teach(feature, "echoed-session-turn", "us-central1")

    history = await governed.get_conversation_history(limit=10)
    filed = {entry["metadata"]["session_id"] for entry in history}
    assert len(filed) == 1 and echoed not in filed
    sources = await governed.list_assertion_revision_sources(saved["revision_id"])
    assert [source.references for source in sources] == [
        frozenset({(SourceReferenceKind.SESSION, filed.pop())})
    ]


async def _semantic_row_counts(storage: AsyncStorage) -> dict[str, int]:
    """How many rows every semantic table holds."""
    tables = await storage.db.fetchall(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'semantic%'"
    )
    assert tables
    return {
        name: await storage.db.fetchval(f"SELECT COUNT(*) FROM {name}")
        for (name,) in tables
    }


@pytest.mark.asyncio
async def test_save_fact_refuses_a_session_it_cannot_name_and_writes_nothing(
    storage: AsyncStorage,
) -> None:
    """A session key a locator cannot carry is refused, not dropped.

    Dropping the session would write a fact whose source names no session, and
    excluding the session it was taught in would then miss it. The refusal
    comes before any write, so nothing is left behind.
    """
    feature, agent, governed = _fact_feature(storage)
    before = await _semantic_row_counts(storage)

    with live_turn(agent, "\texam-session\t"):
        with invocation_scope("control-character-session-turn"):
            result = await feature.save_fact(
                subject="user", predicate="preferred_deploy_region", value="exam-region"
            )

    assert result.status is ToolResultStatus.ERROR
    assert "control character" in result.error
    assert await _semantic_row_counts(storage) == before


async def _ledger_feature(storage: AsyncStorage, tmp_path):
    """The strategy ledger over the real privacy wrapper and turn lifecycle."""
    from kestrel_sovereign.features.strategic_memory.feature import (
        StrategicMemoryFeature,
    )

    governed = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    agent = TurnBoundAgent(
        storage=governed, agent_id=storage.agent_id, agent_data_dir=str(tmp_path)
    )
    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    return feature, agent, governed


async def _ledger_row_references(governed: PrivacyEnforcingStorage) -> dict:
    """Each projected ledger row's IRI, mapped to the references its source names."""
    named = {}
    for held in await governed.query_assertions(AssertionQuery(limit=50)):
        sources = await governed.list_assertion_revision_sources(held.revision_id)
        named[held.object.value] = [source.references for source in sources]
    return named


@pytest.mark.asyncio
@pytest.mark.parametrize("projecting_session", ["later-session", None])
async def test_a_ledger_row_names_the_session_that_recorded_it_not_the_one_projecting(
    storage: AsyncStorage, tmp_path, projecting_session
) -> None:
    """#3875: a row's session is the session of the turn that wrote it.

    Projection is best-effort. A row whose first projection failed is projected
    by whichever turn next reindexes, in another session or in none. Naming
    that turn's session would misattribute the row, or name none, and either
    way excluding the session the row was recorded in would not hold it out.
    """
    feature, agent, governed = await _ledger_feature(storage, tmp_path)
    store_put = governed.put_assertion

    async def unavailable(*args, **kwargs):
        raise RuntimeError("assertion store unavailable")

    governed.put_assertion = unavailable
    with live_turn(agent, EXAM_SESSION):
        recorded = await feature.strategy_add_pattern(
            pattern="Learned during the exam", source="exam", implication="hold out"
        )
    assert recorded.data["persisted"] is True
    assert await governed.query_assertions(AssertionQuery(limit=50)) == []
    exam_row = recorded.data["pattern_id"]

    governed.put_assertion = store_put
    with live_turn(agent, projecting_session, turn_id="projecting-turn"):
        later = await feature.strategy_add_pattern(
            pattern="Learned later", source="later", implication="keep"
        )
    later_row = later.data["pattern_id"]

    named = {
        iri.rsplit(":", 1)[-1]: references
        for iri, references in (await _ledger_row_references(governed)).items()
    }
    later_references = (
        frozenset({(SourceReferenceKind.SESSION, projecting_session)})
        if projecting_session is not None
        else frozenset()
    )
    assert named == {
        exam_row: [frozenset({(SourceReferenceKind.SESSION, EXAM_SESSION)})],
        later_row: [later_references],
    }

    by_row = {
        held.object.value.rsplit(":", 1)[-1]: held
        for held in await governed.query_assertions(AssertionQuery(limit=50))
    }
    exam_assertion = by_row[exam_row]
    (exam_source,) = await governed.list_assertion_revision_sources(
        exam_assertion.revision_id
    )
    await storage.run_semantic_maintenance(None)
    capabilities = await storage.semantic_maintenance_capability_versions(None)
    policy = GovernedCorpusPolicy(
        policy_id="exam-holdout",
        policy_version="1",
        accepted_epistemic_states=(exam_assertion.epistemic_state,),
        accepted_visibility=(exam_assertion.visibility,),
        accepted_privacy_classifications=(exam_assertion.privacy_classification,),
        accepted_consent_references=(exam_assertion.release_policy_reference,),
        accepted_grounding_classes=(exam_assertion.confidence_basis,),
        accepted_source_kinds=(exam_source.source_kind,),
        accepted_ontology_pins=(exam_assertion.ontology_version,),
        accepted_semantic_capability_versions=tuple(capabilities.items()),
    )

    unexcluded = await _corpus_snapshot(storage, policy)
    held_out = await _corpus_snapshot(
        storage, replace(policy, excluded_session_ids=(EXAM_SESSION,))
    )

    assert {item.assertion.revision_id for item in unexcluded.examples} == {
        exam_assertion.revision_id,
        by_row[later_row].revision_id,
    }
    assert [item.assertion.revision_id for item in held_out.examples] == [
        by_row[later_row].revision_id
    ]
    assert held_out.observability.excluded == {"explicitly_excluded": 1}


@pytest.mark.asyncio
async def test_export_producer_registers_exact_runtime_artifact_before_return(storage: AsyncStorage) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id, "producer-export-revision")
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    artifact_id = str(uuid4())

    checkpoint, exported = await storage.export_assertion_snapshot(
        artifact_id=artifact_id,
        consumer_id="export-owner",
        consumer_key_id="export-owner-key",
        consumer_public_key=_public_key(private_key),
        retention_seconds=300,
    )

    assert tuple(item.revision_id for item in exported) == (assertion.revision_id,)
    row = await storage.db.fetchone(
        "SELECT checkpoint_generation, policy_pin, capability_digest, artifact_digest "
        "FROM semantic_governed_artifacts WHERE tenant_id = ? AND artifact_id = ?",
        (storage.agent_id, artifact_id),
    )
    assert row is not None and int(row[0]) == checkpoint.generation
    assert all(isinstance(value, str) and len(value) == 64 for value in row[1:])
    await storage.consume_governed_semantic_artifact(
        artifact_id, expected_generation=checkpoint.generation
    )
    assert not hasattr(storage, "register_governed_semantic_artifact")
    with pytest.raises(TypeError):
        await storage.export_assertion_snapshot()


@pytest.mark.asyncio
async def test_sleep_drives_expiry_with_storage_owned_clock(storage: AsyncStorage) -> None:
    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id, "sleep-expiry-revision")
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    checkpoint = await storage.assertion_checkpoint()
    registration = _registration(
        assertion, checkpoint.generation, private_key=private_key
    )
    await storage._assertion_store().register_governed_artifact(registration)
    storage._artifact_clock = lambda: datetime.now(timezone.utc) + timedelta(hours=2)
    storage._assertion_store()._artifact_clock = storage._artifact_clock

    class SleepHarness(SleepMixin):
        sleep_hooks = []
        semantic_inference_configured = False
        semantic_maintenance_configured = False
        semantic_capabilities_configured = False
        semantic_inference_profile = None

        def __init__(self, bound_storage):
            self.storage = bound_storage

    await SleepHarness(storage).sleep(
        skip_consolidation=True, skip_export=True, skip_reflection=True
    )
    assert await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifact_revocations "
        "WHERE tenant_id = ? AND acknowledged_at IS NULL",
        (storage.agent_id,),
    ) == 1
    assert "now" not in inspect.signature(
        storage.sweep_expired_governed_semantic_artifacts
    ).parameters


@pytest.mark.asyncio
async def test_artifact_expiry_sweep_without_tenant_authority_is_unconfigured_noop(
    tmp_path, monkeypatch
) -> None:
    unconfigured = AsyncStorage(
        str(tmp_path / "unconfigured-artifact-sweep.db"),
        agent_id="did:example:legacy",
    )

    async def fail_if_initialized() -> None:
        raise AssertionError("unconfigured artifact sweep must not initialize storage")

    monkeypatch.setattr(unconfigured, "initialize", fail_if_initialized)

    assert await unconfigured.sweep_expired_governed_semantic_artifacts() == 0
    assert unconfigured._initialized is False


@pytest.mark.asyncio
async def test_artifact_expiry_sweep_with_tenant_authority_propagates_initialize_failure(
    storage: AsyncStorage, tmp_path, monkeypatch
) -> None:
    configured = AsyncStorage(
        str(tmp_path / "configured-artifact-sweep.db"),
        agent_id=storage.agent_id,
        _assertion_tenant_capability=storage._assertion_tenant_capability,
    )

    async def fail_initialize() -> None:
        raise RuntimeError("forced artifact storage initialization failure")

    monkeypatch.setattr(configured, "initialize", fail_initialize)

    with pytest.raises(
        RuntimeError, match="forced artifact storage initialization failure"
    ):
        await configured.sweep_expired_governed_semantic_artifacts()


@pytest.mark.asyncio
async def test_v1_migration_quarantines_and_makes_legacy_rows_claimable(
    tmp_path, monkeypatch
) -> None:
    private_key = Ed25519PrivateKey.generate()
    monkeypatch.delenv("KESTREL_LEGACY_ARTIFACT_MIGRATION_PUBLIC_KEY", raising=False)
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Legacy migration"
    )
    tenant_id = credentials.agent_did
    identity_key = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(identity_key, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(tenant_id, identity)
    db_path = tmp_path / "legacy-artifacts.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE semantic_schema_migrations (version TEXT PRIMARY KEY);
        INSERT INTO semantic_schema_migrations VALUES ('semantic_governed_artifact_lifecycle_v1');
        CREATE TABLE semantic_governed_artifacts (
          tenant_id TEXT, artifact_id TEXT, kind TEXT, consumer_id TEXT,
          checkpoint_generation INTEGER, policy_pin TEXT, capability_digest TEXT,
          artifact_digest TEXT, retention_expires_at TEXT, state TEXT,
          invalidated_generation INTEGER, created_at TEXT, updated_at TEXT,
          PRIMARY KEY (tenant_id, artifact_id));
        CREATE TABLE semantic_governed_artifact_lineage (
          tenant_id TEXT, artifact_id TEXT, assertion_id TEXT, revision_id TEXT,
          PRIMARY KEY (tenant_id, artifact_id, assertion_id, revision_id));
        CREATE TABLE semantic_governed_artifact_revocations (
          tenant_id TEXT, revocation_id TEXT, artifact_key TEXT, kind TEXT,
          consumer_id TEXT, attempt INTEGER, lease_token TEXT, lease_expires_at TEXT,
          acknowledged_at TEXT, deletion_proof_digest TEXT,
          invalidated_generation INTEGER, PRIMARY KEY (tenant_id, revocation_id));
        CREATE TABLE semantic_governed_artifact_receipts (
          receipt_id TEXT PRIMARY KEY, tenant_id TEXT, artifact_key TEXT, kind TEXT,
          state TEXT, generation INTEGER, receipt_digest TEXT, created_at TEXT);
        """
    )
    artifact_id = str(uuid4())
    connection.execute(
        "INSERT INTO semantic_governed_artifacts VALUES (?, ?, 'export_snapshot', "
        "'unauthenticated-v1', 7, ?, ?, ?, '2099-01-01T00:00:00Z', 'active', NULL, ?, ?)",
        (tenant_id, artifact_id, "a" * 64, "b" * 64, "c" * 64,
         "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    )
    connection.execute(
        "INSERT INTO semantic_governed_artifact_lineage VALUES (?, ?, 'raw-assertion', 'raw-revision')",
        (tenant_id, artifact_id),
    )
    connection.commit()
    connection.close()

    blocked = AsyncStorage(
        str(db_path), agent_id=tenant_id, _assertion_tenant_capability=capability
    )
    try:
        with pytest.raises(Exception, match="explicit.*MIGRATION_PUBLIC_KEY"):
            await blocked.initialize()
    finally:
        await blocked.close()
    monkeypatch.setenv(
        "KESTREL_LEGACY_ARTIFACT_MIGRATION_PUBLIC_KEY", _public_key(private_key)
    )
    migrated = await _open_storage(str(db_path), tenant_id, capability)
    try:
        assert await migrated.db.fetchval("SELECT COUNT(*) FROM semantic_governed_artifacts") == 0
        assert await migrated.db.fetchval("SELECT COUNT(*) FROM semantic_governed_artifact_lineage") == 0
        row = await migrated.db.fetchone(
            "SELECT consumer_id, consumer_key_id, consumer_public_key, artifact_digest "
            "FROM semantic_governed_artifact_revocations"
        )
        assert row == (
            "kestrel-artifact-migration", "legacy-quarantine-v1",
            _public_key(private_key), "c" * 64,
        )
        auth = GovernedArtifactConsumerAuthentication(
            "kestrel-artifact-migration", "legacy-quarantine-v1", str(uuid4()),
            datetime.now(timezone.utc).isoformat(), "0" * 128,
        )
        auth = replace(
            auth, signature=private_key.sign(auth.signable_bytes(tenant_id)).hex()
        )
        assert await migrated.claim_governed_semantic_artifact_revocation(auth) is not None
    finally:
        await migrated.close()
