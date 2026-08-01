"""Governed export/corpus lifecycle tests (#2831)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kestrel_sovereign.identity.runtime_identity import AgentIdentity, load_agent_identity
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.knowledge import (
    Assertion,
    DirectLineage,
    EpistemicState,
    GovernedArtifactError,
    GovernedArtifactConsumerAuthentication,
    GovernedArtifactDeletionOwner,
    GovernedArtifactDeletionProof,
    GovernedArtifactKind,
    GovernedArtifactLineage,
    GovernedArtifactRegistration,
    IRI,
    Literal,
    OntologyRef,
    SourceOccurrence,
    XSD_STRING,
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
    registered = await storage.register_governed_semantic_artifact(registration)
    with pytest.raises(GovernedArtifactError, match="different public key"):
        await storage.register_governed_semantic_artifact(
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
        await storage.register_governed_semantic_artifact(forged)

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
        await storage.register_governed_semantic_artifact(expired)

    await storage.register_governed_semantic_artifact(registration)
    await storage.retract_assertion(assertion.assertion_id, assertion.revision_id)
    with pytest.raises(GovernedArtifactError, match="resurrected"):
        await storage.register_governed_semantic_artifact(registration)


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
        await first.register_governed_semantic_artifact(
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
        storage.register_governed_semantic_artifact(registration),
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
    await storage.register_governed_semantic_artifact(registration)
    swept = await storage.sweep_expired_governed_semantic_artifacts(
        now=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    )
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
        await wrapper.register_governed_semantic_artifact(registration)
    with pytest.raises(PrivacyViolationError):
        await wrapper.consume_governed_semantic_artifact(
            registration.artifact_id, expected_generation=checkpoint.generation
        )
    # Cleanup remains callable even in a volatile mode so a mode transition
    # cannot strand bytes written while persistence was allowed.
    assert await wrapper.sweep_expired_governed_semantic_artifacts() == 0
