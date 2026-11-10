"""Governed export/corpus lifecycle tests (#2831)."""

from __future__ import annotations

import asyncio
import inspect
import json
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
from tests.utils.corpus_exclusions import ExclusionAuthority
from tests.utils.live_turn import TurnBoundAgent, live_turn


async def _open_storage(db_path: str, tenant_id: str, capability) -> AsyncStorage:
    instance = AsyncStorage(
        db_path, agent_id=tenant_id, _assertion_tenant_capability=capability,
    )
    await instance.initialize()
    return instance


@pytest.fixture
def exclusion_authority(tmp_path, monkeypatch) -> ExclusionAuthority:
    """The host's Sovereign, who signs its corpus exclusion manifest (#3893)."""
    return ExclusionAuthority(tmp_path / "sovereign", monkeypatch)


@pytest.fixture
async def storage(tmp_path, exclusion_authority):
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Artifact lifecycle test",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity: AgentIdentity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(credentials.agent_did, identity)
    exclusion_authority.admit(credentials.agent_did)
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


def _admitted_receipt():
    """The receipt of the exclusion manifest this test host admits now (#3893)."""
    from kestrel_sovereign.knowledge.corpus_exclusions import load_corpus_exclusions

    return load_corpus_exclusions().receipt


def _elsewhere_receipt():
    """A manifest no test host admits: a corpus registered under it is never served."""
    from kestrel_sovereign.knowledge import CorpusExclusionReceipt

    return CorpusExclusionReceipt("d" * 64, 1, "2026-11-01T00:00:00+00:00")


def _registration(
    assertion: Assertion, generation: int, *, private_key: Ed25519PrivateKey,
    artifact_id: str | None = None, exclusion_manifest=None,
) -> GovernedArtifactRegistration:
    """A corpus registration, built under the host's admitted manifest by default."""
    receipt = exclusion_manifest or _admitted_receipt()
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
        exclusion_manifest_sha256=receipt.manifest_sha256,
        exclusion_manifest_sequence=receipt.sequence,
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
        exclusion_manifest_sha256=registration.exclusion_manifest_sha256,
        exclusion_manifest_sequence=registration.exclusion_manifest_sequence,
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
        exclusion_manifest_sha256=expired.exclusion_manifest_sha256,
        exclusion_manifest_sequence=expired.exclusion_manifest_sequence,
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
            _registration(
                assertion, checkpoint.generation, private_key=private_key,
                exclusion_manifest=_elsewhere_receipt(),
            )
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


async def _ledger_rows(governed: PrivacyEnforcingStorage) -> dict:
    """Each current ledger assertion, keyed by the row id its object names."""
    return {
        held.object.value.rsplit(":", 1)[-1]: held
        for held in await governed.query_assertions(AssertionQuery(limit=50))
    }


async def _ledger_corpus_policy(
    storage: AsyncStorage, governed: PrivacyEnforcingStorage
) -> GovernedCorpusPolicy:
    """A policy every ledger assertion's class rules accept, and no exclusions."""
    await storage.run_semantic_maintenance(None)
    capabilities = await storage.semantic_maintenance_capability_versions(None)
    held = next(iter((await _ledger_rows(governed)).values()))
    sources = await governed.list_assertion_revision_sources(held.revision_id)
    return GovernedCorpusPolicy(
        policy_id="exam-holdout",
        policy_version="1",
        accepted_epistemic_states=(held.epistemic_state,),
        accepted_visibility=(held.visibility,),
        accepted_privacy_classifications=(held.privacy_classification,),
        accepted_consent_references=(held.release_policy_reference,),
        accepted_grounding_classes=(held.confidence_basis,),
        accepted_source_kinds=tuple(sorted({source.source_kind for source in sources})),
        accepted_ontology_pins=(held.ontology_version,),
        accepted_semantic_capability_versions=tuple(capabilities.items()),
    )


def _corpus_rows(snapshot) -> set:
    """The ledger row ids a corpus admitted."""
    return {
        item.assertion.object.value.rsplit(":", 1)[-1] for item in snapshot.examples
    }


def _edit_ledger_file(feature, edit) -> None:
    """Edit ``STRATEGY_LEDGER.yaml`` by hand, as an agent with file access can."""
    import yaml

    path = feature._ledger.path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data["patterns_learned"])
    path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")


def _rename_row(rows) -> None:
    rows[0]["id"] = "pat_renamed"


def _name_another_session(rows) -> None:
    rows[0]["session_id"] = "everyday-session"


def _edit_row_text(rows) -> None:
    rows[0]["pattern"] = "Learned during the exam, reworded"


def _leave_row(rows) -> None:
    raise AssertionError("the untouched file is not rewritten")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (_leave_row, "explicitly_excluded"),
        (_name_another_session, "explicitly_excluded"),
        (_rename_row, "session_unattested"),
        (_edit_row_text, "session_unattested"),
    ],
)
async def test_an_edit_after_a_failed_projection_never_lets_a_row_escape_its_session(
    storage: AsyncStorage, tmp_path, edit, reason
) -> None:
    """#3990: the row's session comes from what host code attested, never the file.

    A row recorded in the exam session whose first projection failed is
    projected later, after the agent edited ``STRATEGY_LEDGER.yaml``. A session
    field it wrote names nothing; a row whose id or text it changed has content
    nothing attested. Neither escapes a corpus excluding the exam session, and
    neither is lost from a corpus that excludes no session.
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

    if edit is not _leave_row:
        _edit_ledger_file(feature, edit)
    governed.put_assertion = store_put
    with live_turn(agent, "everyday-session", turn_id="later-turn"):
        later = await feature.strategy_add_pattern(
            pattern="Learned later", source="later", implication="keep"
        )
    assert later.data.get("reloaded_from_disk", False) is (edit is not _leave_row)
    rows = await _ledger_rows(governed)
    assert len(rows) == 2
    exam_row = next(row for row in rows if row != later.data["pattern_id"])

    policy = await _ledger_corpus_policy(storage, governed)
    unexcluded = await _corpus_snapshot(storage, policy)
    held_out = await _corpus_snapshot(
        storage, replace(policy, excluded_session_ids=(EXAM_SESSION,))
    )

    assert _corpus_rows(unexcluded) == {exam_row, later.data["pattern_id"]}
    assert _corpus_rows(held_out) == {later.data["pattern_id"]}
    assert held_out.observability.excluded == {reason: 1}


@pytest.mark.asyncio
async def test_identical_content_recorded_in_two_sessions_is_held_out_with_either(
    storage: AsyncStorage, tmp_path
) -> None:
    """One digest may carry several sessions; excluding any one holds it out.

    The exam row's projection fails and the agent deletes it from the file. The
    same pattern recorded later, in another session on the same day, has the
    same id and content, so its digest is attested in both sessions.
    """
    feature, agent, governed = await _ledger_feature(storage, tmp_path)
    store_put = governed.put_assertion

    async def unavailable(*args, **kwargs):
        raise RuntimeError("assertion store unavailable")

    governed.put_assertion = unavailable
    with live_turn(agent, EXAM_SESSION):
        first = await feature.strategy_add_pattern(
            pattern="Learned twice", source="s", implication="i"
        )
    _edit_ledger_file(feature, lambda rows: rows.clear())
    governed.put_assertion = store_put
    with live_turn(agent, "everyday-session", turn_id="later-turn"):
        again = await feature.strategy_add_pattern(
            pattern="Learned twice", source="s", implication="i"
        )
    assert again.data["pattern_id"] == first.data["pattern_id"]
    assert list(await _ledger_rows(governed)) == [again.data["pattern_id"]]

    policy = await _ledger_corpus_policy(storage, governed)
    assert _corpus_rows(await _corpus_snapshot(storage, policy)) == {
        again.data["pattern_id"]
    }
    for excluded in (EXAM_SESSION, "everyday-session"):
        held_out = await _corpus_snapshot(
            storage, replace(policy, excluded_session_ids=(excluded,))
        )
        assert held_out.examples == ()
        assert held_out.observability.excluded == {"explicitly_excluded": 1}


@pytest.mark.asyncio
async def test_a_row_whose_attestation_fails_is_never_recorded(
    storage: AsyncStorage, tmp_path
) -> None:
    """A failed attestation refuses the write; the row is never admitted unattested."""
    feature, agent, governed = await _ledger_feature(storage, tmp_path)
    await storage.db.execute(
        "CREATE TRIGGER refuse_attestation BEFORE INSERT ON "
        "strategy_ledger_row_attestations BEGIN "
        "SELECT RAISE(ABORT, 'attestation refused'); END"
    )
    with live_turn(agent, EXAM_SESSION):
        result = await feature.strategy_add_pattern(
            pattern="Learned during the exam", source="exam", implication="hold out"
        )

    assert result.status is ToolResultStatus.ERROR
    assert "Nothing was recorded" in result.error
    assert result.data == {"recorded": False, "persisted": False}
    assert feature._ledger.patterns == []
    assert not feature._ledger.path.exists() or "Learned during the exam" not in (
        feature._ledger.path.read_text(encoding="utf-8")
    )
    assert await governed.query_assertions(AssertionQuery(limit=50)) == []


@pytest.mark.asyncio
async def test_a_legacy_row_is_held_out_only_of_a_session_excluding_corpus(
    storage: AsyncStorage, tmp_path
) -> None:
    """A row nothing attested has an unknown session (#3990).

    That covers a row written into the file by hand or before attestations
    existed, and one projected before then, whose stored locator carries no
    marker: the corpus decides from the attestations as they are when it is
    built, never from the stored locator alone.
    """
    feature, agent, governed = await _ledger_feature(storage, tmp_path)
    with live_turn(agent, None):
        recorded = await feature.strategy_add_pattern(
            pattern="Recorded before attestations", source="s", implication="i"
        )
    row = recorded.data["pattern_id"]
    (held,) = (await _ledger_rows(governed)).values()
    (source,) = await governed.list_assertion_revision_sources(held.revision_id)
    assert source.references == frozenset()
    # As if the row had been projected before the attestation record existed.
    await storage.db.execute("DELETE FROM strategy_ledger_row_attestations")

    _edit_ledger_file(
        feature,
        lambda rows: rows.append(
            {"id": "pat_by_hand", "pattern": "Written by hand", "session_id": "x"}
        ),
    )
    with live_turn(agent, None, turn_id="later-turn"):
        later = await feature.strategy_add_pattern(
            pattern="Recorded since", source="s", implication="i"
        )
    assert set(await _ledger_rows(governed)) == {
        row, "pat_by_hand", later.data["pattern_id"]
    }

    policy = await _ledger_corpus_policy(storage, governed)
    assert _corpus_rows(await _corpus_snapshot(storage, policy)) == {
        row, "pat_by_hand", later.data["pattern_id"]
    }
    held_out = await _corpus_snapshot(
        storage, replace(policy, excluded_session_ids=(EXAM_SESSION,))
    )
    assert _corpus_rows(held_out) == {later.data["pattern_id"]}
    assert held_out.observability.excluded == {"session_unattested": 2}
    saved_item_only = await _corpus_snapshot(
        storage, replace(policy, excluded_saved_item_ids=("item-1",))
    )
    assert _corpus_rows(saved_item_only) == {
        row, "pat_by_hand", later.data["pattern_id"]
    }


@pytest.mark.asyncio
async def test_a_session_attested_after_projection_still_holds_the_row_out(
    storage: AsyncStorage, tmp_path
) -> None:
    """The stored locator was written at projection; attestations grow after it."""
    from kestrel_sovereign.features.strategic_memory import ledger_assertions

    feature, agent, governed = await _ledger_feature(storage, tmp_path)
    with live_turn(agent, "everyday-session"):
        recorded = await feature.strategy_add_pattern(
            pattern="Learned twice", source="s", implication="i"
        )
    (row,) = feature._ledger.patterns
    await governed.attest_strategy_ledger_row(
        section_key="patterns_learned",
        content_digest=ledger_assertions.attestation_digest("patterns_learned", row),
        session_id=EXAM_SESSION,
    )
    (held,) = (await _ledger_rows(governed)).values()
    (source,) = await governed.list_assertion_revision_sources(held.revision_id)
    assert source.references == frozenset(
        {(SourceReferenceKind.SESSION, "everyday-session")}
    )

    policy = await _ledger_corpus_policy(storage, governed)
    held_out = await _corpus_snapshot(
        storage, replace(policy, excluded_session_ids=(EXAM_SESSION,))
    )
    assert recorded.data["pattern_id"] == row["id"]
    assert held_out.examples == ()
    assert held_out.observability.excluded == {"explicitly_excluded": 1}


@pytest.fixture
async def file_storage(tmp_path, exclusion_authority):
    """Native storage over a SQLite file, as an agent's database is."""
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Ledger delta test",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(
        credentials.agent_did, identity
    )
    exclusion_authority.admit(credentials.agent_did)
    instance = await _open_storage(
        str(tmp_path / "native.db"), credentials.agent_did, capability
    )
    try:
        yield instance
    finally:
        await instance.close()


async def _corpus_delta(storage: AsyncStorage, base, policy: GovernedCorpusPolicy):
    """One incremental read against ``base``, registered for the same consumer."""
    return await storage.governed_assertion_corpus_changes_since(
        base,
        policy=policy,
        inference_profile=None,
        artifact_id=str(uuid4()),
        consumer_id="parametric-self-test",
        consumer_key_id="parametric-self-key-v1",
        consumer_public_key=_CORPUS_CONSUMER_PUBLIC_KEY,
        retention_seconds=300,
    )


@pytest.mark.asyncio
async def test_an_incremental_read_withdraws_a_row_since_attested_in_an_excluded_session(
    file_storage: AsyncStorage, tmp_path
) -> None:
    """#3990: a new attestation, not only an assertion change, withdraws a row.

    The row is recorded in an everyday session and admitted to a corpus that
    excludes the exam session. The agent deletes it from the file, and the
    same content is then recorded during the exam: same id, same digest, so
    the stored assertion is unchanged and the change stream is empty. A fresh
    snapshot holds the row out, and an incremental read against the earlier
    snapshot withdraws it.
    """
    storage = file_storage
    feature, agent, governed = await _ledger_feature(storage, tmp_path)
    with live_turn(agent, "everyday-session"):
        recorded = await feature.strategy_add_pattern(
            pattern="Learned twice", source="s", implication="i"
        )
    row = recorded.data["pattern_id"]
    policy = await _ledger_corpus_policy(storage, governed)
    excluding = replace(policy, excluded_session_ids=(EXAM_SESSION,))
    base = await _corpus_snapshot(storage, excluding)
    unexcluded_base = await _corpus_snapshot(storage, policy)
    assert _corpus_rows(base) == _corpus_rows(unexcluded_base) == {row}
    (example,) = base.examples
    checkpoint = await storage.assertion_event_checkpoint()

    _edit_ledger_file(feature, lambda rows: rows.clear())
    with live_turn(agent, EXAM_SESSION, turn_id="exam-turn"):
        again = await feature.strategy_add_pattern(
            pattern="Learned twice", source="s", implication="i"
        )
    assert again.data["persisted"] is True
    assert again.data["pattern_id"] == row
    assert await storage.assertion_event_checkpoint() == checkpoint
    fresh = await _corpus_snapshot(storage, excluding)
    assert fresh.examples == ()

    delta = await _corpus_delta(storage, base, excluding)

    assert delta.additions == ()
    (tombstone,) = delta.tombstones
    assert (tombstone.assertion_id, tombstone.revision_id) == (
        example.assertion.assertion_id,
        example.assertion.revision_id,
    )
    assert (tombstone.operation, tombstone.reason) == (
        "session_attestation",
        "explicitly_excluded",
    )
    assert tombstone.generation == checkpoint.generation
    assert (await _corpus_delta(storage, base, excluding)).tombstones == delta.tombstones
    # A corpus that excludes no session never consults attestations.
    assert (await _corpus_delta(storage, unexcluded_base, policy)).tombstones == ()


# --- The host's Sovereign-signed exclusion manifest (#3893) -----------------


async def _governed_snapshot(governed: PrivacyEnforcingStorage, policy):
    """A corpus read as a learning consumer reads it: through the privacy gate."""
    return await governed.governed_assertion_corpus_snapshot(
        policy=policy,
        inference_profile=None,
        artifact_id=str(uuid4()),
        consumer_id="parametric-self-test",
        consumer_key_id="parametric-self-key-v1",
        consumer_public_key=_CORPUS_CONSUMER_PUBLIC_KEY,
        retention_seconds=300,
    )


async def _corpus_artifact_count(storage: AsyncStorage) -> int:
    return await storage.db.fetchval(
        "SELECT COUNT(*) FROM semantic_governed_artifacts "
        "WHERE tenant_id = ? AND kind = 'corpus_manifest'",
        (storage.agent_id,),
    )


async def _store_alongside(storage: AsyncStorage, taught: dict, source) -> dict:
    """Another fact of the taught one's classes, from a source of its own."""
    assertion = await storage.get_assertion(taught["assertion_id"])
    other_source = replace(
        source,
        source_occurrence_id=f"source:test-everyday-v1:{'e' * 64}",
        locator="POST:/api/agent/invoke#note:everyday",
    )
    other = replace(
        assertion,
        assertion_id=None,
        revision_id="everyday-revision",
        object=Literal("everyday-region", XSD_STRING),
        lineage=DirectLineage((other_source.source_occurrence_id,)),
    )
    await storage.put_assertion(other, source_occurrences=(other_source,))
    return {"revision_id": other.revision_id}


@pytest.mark.asyncio
async def test_a_manifest_source_occurrence_is_held_out_whatever_the_consumer_policy(
    storage: AsyncStorage, exclusion_authority: ExclusionAuthority
) -> None:
    """The exam's source is excluded on the corpus path a consumer actually calls.

    The consumer's policy names no exclusions, as parametric-self's does: the
    host supplies them from its signed manifest, and the policy, its digest and
    the registered ``policy_pin`` stay the consumer's own.
    """
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    (exam_source,) = await governed.list_assertion_revision_sources(exam["revision_id"])
    everyday = await _store_alongside(storage, exam, exam_source)
    policy = await _taught_fact_policy(storage, exam)
    assert policy.excluded_source_occurrence_ids == ()

    exclusion_authority.exclude(
        storage.agent_id, source_occurrence=[exam_source.source_occurrence_id]
    )
    snapshot = await _governed_snapshot(governed, policy)

    assert [item.assertion.revision_id for item in snapshot.examples] == [
        everyday["revision_id"]
    ]
    assert snapshot.observability.excluded == {"explicitly_excluded": 1}
    assert "sealed-exam-answer" not in str(
        [item.canonical_mapping() for item in snapshot.examples]
    )
    assert snapshot.policy == policy
    assert snapshot.observability.policy_digest == policy.digest
    assert snapshot.exclusion_manifest.sequence == exclusion_authority.sequence
    assert await storage.db.fetchone(
        "SELECT policy_pin, artifact_digest FROM semantic_governed_artifacts "
        "WHERE tenant_id = ? AND kind = 'corpus_manifest'",
        (storage.agent_id,),
    ) == (policy.digest, snapshot.snapshot_hash)
    assert await storage.db.fetchone(
        "SELECT sequence, manifest_sha256 FROM corpus_exclusion_watermarks "
        "WHERE agent_id = ?",
        (storage.agent_id,),
    ) == (
        snapshot.exclusion_manifest.sequence,
        snapshot.exclusion_manifest.manifest_sha256,
    )

    # The same material under an empty section enters, under another hash.
    exclusion_authority.admit(storage.agent_id)
    admitted = await _governed_snapshot(governed, policy)
    assert {item.assertion.revision_id for item in admitted.examples} == {
        exam["revision_id"], everyday["revision_id"],
    }
    assert admitted.snapshot_hash != snapshot.snapshot_hash


@pytest.mark.asyncio
async def test_a_manifest_session_holds_out_what_was_taught_in_it(
    storage: AsyncStorage, exclusion_authority: ExclusionAuthority
) -> None:
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, EXAM_SESSION):
        exam = await _teach(feature, "exam-turn", "exam-answer-region")
    policy = await _taught_fact_policy(storage, exam)

    exclusion_authority.exclude(storage.agent_id, session=[EXAM_SESSION])
    snapshot = await _governed_snapshot(governed, policy)

    assert snapshot.examples == ()
    assert snapshot.observability.excluded == {"explicitly_excluded": 1}


def _sign_unchecked(authority: ExclusionAuthority, manifest: dict) -> dict:
    """Sign ``manifest`` as written: the builders would refuse a malformed one."""
    from kestrel_sovereign.knowledge.corpus_exclusions import (
        canonical_corpus_exclusions_bytes,
    )
    from kestrel_sovereign.security.crypto_suite import (
        ALG_ECDSA_SECP256K1_SHA256,
        get_suite,
    )

    signature = get_suite(ALG_ECDSA_SECP256K1_SHA256).sign(
        canonical_corpus_exclusions_bytes(manifest), authority.private_key
    )
    return {
        **manifest,
        "signature": {
            "alg": ALG_ECDSA_SECP256K1_SHA256, "kid": "keys-1", "sig": signature.hex(),
        },
    }


def _with_section(authority, agent_id, section) -> dict:
    manifest = authority.sign({agent_id: {}}, sequence=authority.sequence + 1)
    manifest.pop("signature")
    manifest["agents"] = {agent_id: section}
    return _sign_unchecked(authority, manifest)


def _missing_manifest(authority, agent_id):
    authority.manifest_path.unlink()


def _not_json(authority, agent_id):
    authority.manifest_path.write_text("{not json")


def _tampered(authority, agent_id):
    manifest = json.loads(authority.manifest_path.read_text())
    manifest["agents"][agent_id] = {}
    manifest["reason"] = "excludes less than the Sovereign signed"
    authority.write(manifest)


def _forged(authority, agent_id):
    from kestrel_sovereign.knowledge.corpus_exclusions import (
        build_legacy_signed_corpus_exclusions,
    )
    from kestrel_sovereign.security.crypto_suite import Secp256k1Suite

    authority.write(build_legacy_signed_corpus_exclusions(
        signer_did=authority.did_document["id"],
        sequence=authority.sequence + 1,
        agents={agent_id: {}},
        private_key=Secp256k1Suite().generate_keypair().private_key,
    ))


def _no_section(authority, agent_id):
    authority.write(authority.sign({}, sequence=authority.sequence + 1))


def _unknown_kind(authority, agent_id):
    authority.write(_with_section(authority, agent_id, {"conversation": ["x"]}))


def _shortened_saved_item(authority, agent_id):
    authority.write(_with_section(authority, agent_id, {"saved_item": ["2ca40dbd"]}))


def _shortened_source(authority, agent_id):
    authority.write(_with_section(
        authority, agent_id,
        {"source_occurrence": ["source:memory-agency-save-fact-v1:3a9b12"]},
    ))


def _shortened_session(authority, agent_id):
    authority.write(_with_section(authority, agent_id, {"session": ["dcd6f767"]}))


def _unmatchable_saved_item(authority, agent_id):
    authority.exclude(agent_id, saved_item=["2ca40dbd-0000-4000-8000-000000000000"])


def _unsigned_field(authority, agent_id):
    manifest = json.loads(authority.manifest_path.read_text())
    manifest["excluded_by_default"] = False
    authority.write(manifest)


def _duplicate_key(authority, agent_id):
    text = authority.manifest_path.read_text()
    authority.manifest_path.write_text(text[:-1] + ', "reason": "other"}')


def _no_trust_root(authority, agent_id):
    authority.trust_root_path.unlink()


def _agent_as_trust_root(authority, agent_id):
    document = {**authority.did_document, "id": agent_id}
    authority.trust_root_path.write_text(json.dumps(document))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("break_manifest", "message"),
    [
        (_missing_manifest, "no manifest at"),
        (_not_json, "not valid JSON"),
        (_tampered, "signature verification failed"),
        (_forged, "signature verification failed"),
        (_no_section, "has no section for"),
        (_unknown_kind, "unknown id kind 'conversation'"),
        (_shortened_saved_item, "malformed or shortened"),
        (_shortened_source, "malformed or shortened"),
        (_shortened_session, "malformed or shortened"),
        (_unmatchable_saved_item, "no producer records"),
        (_unsigned_field, "unsigned field"),
        (_duplicate_key, "repeats a key"),
        (_no_trust_root, "no Sovereign trust root"),
        (_agent_as_trust_root, "agent-owned DID"),
    ],
)
async def test_a_corpus_build_refuses_rather_than_run_without_its_exclusions(
    storage: AsyncStorage,
    exclusion_authority: ExclusionAuthority,
    break_manifest,
    message: str,
) -> None:
    from kestrel_sovereign.knowledge.corpus_exclusions import CorpusExclusionManifestError

    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    policy = await _taught_fact_policy(storage, exam)
    break_manifest(exclusion_authority, storage.agent_id)

    with pytest.raises(CorpusExclusionManifestError, match=message):
        await _governed_snapshot(governed, policy)
    assert await _corpus_artifact_count(storage) == 0


@pytest.mark.asyncio
async def test_an_older_signed_manifest_cannot_be_put_back(
    storage: AsyncStorage, exclusion_authority: ExclusionAuthority
) -> None:
    from kestrel_sovereign.knowledge.corpus_exclusions import CorpusExclusionManifestError

    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    (exam_source,) = await governed.list_assertion_revision_sources(exam["revision_id"])
    policy = await _taught_fact_policy(storage, exam)
    older = json.loads(exclusion_authority.manifest_path.read_text())
    exclusion_authority.exclude(
        storage.agent_id, source_occurrence=[exam_source.source_occurrence_id]
    )
    assert (await _governed_snapshot(governed, policy)).examples == ()

    # Validly signed, and it excludes less: the build refuses it.
    exclusion_authority.write(older)
    with pytest.raises(CorpusExclusionManifestError, match="lower than sequence 2"):
        await _governed_snapshot(governed, policy)
    # Re-signed at the admitted sequence with other content: refused too.
    exclusion_authority.write(
        exclusion_authority.sign({storage.agent_id: {}}, sequence=2)
    )
    with pytest.raises(CorpusExclusionManifestError, match="already admitted"):
        await _governed_snapshot(governed, policy)
    assert await storage.db.fetchval(
        "SELECT sequence FROM corpus_exclusion_watermarks WHERE agent_id = ?",
        (storage.agent_id,),
    ) == 2
    assert await _corpus_artifact_count(storage) == 1


@pytest.mark.asyncio
async def test_another_agents_unmatchable_section_does_not_refuse_this_agent(
    storage: AsyncStorage, exclusion_authority: ExclusionAuthority
) -> None:
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "an-answer")
    policy = await _taught_fact_policy(storage, exam)
    exclusion_authority.exclude(
        "did:web:agents.example:meridian",
        saved_item=["469a149e-0000-4000-8000-000000000000"],
    )

    snapshot = await _governed_snapshot(governed, policy)

    assert [item.assertion.revision_id for item in snapshot.examples] == [
        exam["revision_id"]
    ]


@pytest.mark.asyncio
async def test_an_incremental_read_refuses_once_the_manifest_has_changed(
    storage: AsyncStorage, exclusion_authority: ExclusionAuthority
) -> None:
    """A newer manifest may hold out what the base corpus already holds."""
    from kestrel_sovereign.knowledge import GovernedCorpusUnavailable

    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    (exam_source,) = await governed.list_assertion_revision_sources(exam["revision_id"])
    policy = await _taught_fact_policy(storage, exam)
    base = await _governed_snapshot(governed, policy)
    assert (await _corpus_delta(storage, base, policy)).exclusion_manifest == (
        base.exclusion_manifest
    )

    exclusion_authority.exclude(
        storage.agent_id, source_occurrence=[exam_source.source_occurrence_id]
    )
    with pytest.raises(GovernedCorpusUnavailable, match="corpus_exclusion_manifest_changed"):
        await _corpus_delta(storage, base, policy)


async def _registered_corpus(governed: PrivacyEnforcingStorage, policy):
    """A corpus as its consumer receives it, with the artifact id it consumes by."""
    artifact_id = str(uuid4())
    snapshot = await governed.governed_assertion_corpus_snapshot(
        policy=policy,
        inference_profile=None,
        artifact_id=artifact_id,
        consumer_id="parametric-self-test",
        consumer_key_id="parametric-self-key-v1",
        consumer_public_key=_CORPUS_CONSUMER_PUBLIC_KEY,
        retention_seconds=300,
    )
    return snapshot, artifact_id


async def _pending_corpus_revocations(storage: AsyncStorage) -> list[str]:
    return [
        str(row[0]) for row in await storage.db.fetchall(
            "SELECT kind FROM semantic_governed_artifact_revocations "
            "WHERE tenant_id = ? AND acknowledged_at IS NULL ORDER BY kind",
            (storage.agent_id,),
        )
    ]


async def _active_artifact_ids(storage: AsyncStorage) -> set[str]:
    return {
        str(row[0]) for row in await storage.db.fetchall(
            "SELECT artifact_id FROM semantic_governed_artifacts "
            "WHERE tenant_id = ? AND state = 'active'",
            (storage.agent_id,),
        )
    }


_STALE_CORPUS = "built under corpus exclusion manifest sequence {} .*not the current manifest, sequence {}"


@pytest.mark.asyncio
async def test_a_corpus_built_before_the_exam_was_excluded_is_refused_until_rebuilt(
    file_storage: AsyncStorage, exclusion_authority: ExclusionAuthority
) -> None:
    """Consumption is where an added exclusion takes effect (#3893).

    A corpus holding the exam, built under sequence 1, stays on its
    consumer's disk. Once sequence 2 excludes the exam, consuming that corpus
    refuses; it is left registered rather than revoked, and a rebuild under
    sequence 2 is served without the exam.
    """
    storage = file_storage
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    (exam_source,) = await governed.list_assertion_revision_sources(exam["revision_id"])
    policy = await _taught_fact_policy(storage, exam)
    with_exam, with_exam_id = await _registered_corpus(governed, policy)
    assert [item.assertion.revision_id for item in with_exam.examples] == [
        exam["revision_id"]
    ]
    assert with_exam.exclusion_manifest.sequence == 1
    generation = with_exam.checkpoint.generation
    assert await governed.consume_governed_semantic_artifact(
        with_exam_id, expected_generation=generation
    )

    exclusion_authority.exclude(
        storage.agent_id, source_occurrence=[exam_source.source_occurrence_id]
    )
    with pytest.raises(GovernedArtifactError, match=_STALE_CORPUS.format(1, 2)):
        await governed.consume_governed_semantic_artifact(
            with_exam_id, expected_generation=generation
        )
    assert with_exam_id in await _active_artifact_ids(storage)
    assert await _pending_corpus_revocations(storage) == []

    rebuilt, rebuilt_id = await _registered_corpus(governed, policy)
    assert rebuilt.examples == ()
    assert rebuilt.observability.excluded == {"explicitly_excluded": 1}
    assert rebuilt.exclusion_manifest.sequence == 2
    assert await governed.consume_governed_semantic_artifact(
        rebuilt_id, expected_generation=rebuilt.checkpoint.generation
    )
    # Rebuilding serves the new corpus; it does not reopen the old one.
    with pytest.raises(GovernedArtifactError, match=_STALE_CORPUS.format(1, 2)):
        await governed.consume_governed_semantic_artifact(
            with_exam_id, expected_generation=generation
        )


@pytest.mark.asyncio
async def test_a_newer_manifest_leaves_older_corpora_registered_but_unusable(
    storage: AsyncStorage, exclusion_authority: ExclusionAuthority
) -> None:
    """A build under a newer manifest revokes nothing; consumption refuses.

    A corpus and the delta built on it are refused once another manifest is
    current; an export, which no manifest governs, is still served.
    """
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    policy = await _taught_fact_policy(storage, exam)
    base, base_id = await _registered_corpus(governed, policy)
    delta_id = str(uuid4())
    delta = await storage.governed_assertion_corpus_changes_since(
        base, policy=policy, inference_profile=None, artifact_id=delta_id,
        consumer_id="parametric-self-test", consumer_key_id="parametric-self-key-v1",
        consumer_public_key=_CORPUS_CONSUMER_PUBLIC_KEY, retention_seconds=300,
    )
    export_id = str(uuid4())
    exported_at, _ = await storage.export_assertion_snapshot(
        artifact_id=export_id, consumer_id="parametric-self-test",
        consumer_key_id="parametric-self-key-v1",
        consumer_public_key=_CORPUS_CONSUMER_PUBLIC_KEY, retention_seconds=300,
    )

    exclusion_authority.exclude(storage.agent_id, session=[EXAM_SESSION])
    rebuilt, rebuilt_id = await _registered_corpus(governed, policy)

    assert await _active_artifact_ids(storage) == {base_id, delta_id, export_id, rebuilt_id}
    assert await _pending_corpus_revocations(storage) == []
    for stale_id, stale_generation in (
        (base_id, base.checkpoint.generation),
        (delta_id, delta.checkpoint.generation),
    ):
        with pytest.raises(GovernedArtifactError, match=_STALE_CORPUS.format(1, 2)):
            await governed.consume_governed_semantic_artifact(
                stale_id, expected_generation=stale_generation
            )
    assert await storage.consume_governed_semantic_artifact(
        export_id, expected_generation=exported_at.generation
    )
    assert await governed.consume_governed_semantic_artifact(
        rebuilt_id, expected_generation=rebuilt.checkpoint.generation
    )
    assert await storage.db.fetchone(
        "SELECT exclusion_manifest_sha256, exclusion_manifest_sequence "
        "FROM semantic_governed_artifacts WHERE tenant_id = ? AND artifact_id = ?",
        (storage.agent_id, rebuilt_id),
    ) == (rebuilt.exclusion_manifest.manifest_sha256, exclusion_authority.sequence)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("break_manifest", "message"),
    [
        (_missing_manifest, "no manifest at"),
        (_not_json, "not valid JSON"),
        (_tampered, "signature verification failed"),
        (_forged, "signature verification failed"),
        (_no_section, "has no section for"),
        (_no_trust_root, "no Sovereign trust root"),
        (_agent_as_trust_root, "agent-owned DID"),
    ],
)
async def test_no_corpus_is_served_while_no_manifest_can_be_verified(
    file_storage: AsyncStorage,
    exclusion_authority: ExclusionAuthority,
    break_manifest,
    message: str,
) -> None:
    """A missing or unverifiable manifest refuses consumption of every corpus.

    An agent can remove or replace the manifest. Without a verified one there
    is no current manifest to judge a corpus against, so none is served,
    including one built under the manifest just removed; nothing is revoked,
    and an export, which no manifest governs, is still served.
    """
    from kestrel_sovereign.knowledge.corpus_exclusions import CorpusExclusionManifestError

    storage = file_storage
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    policy = await _taught_fact_policy(storage, exam)
    corpus, corpus_id = await _registered_corpus(governed, policy)
    assert await governed.consume_governed_semantic_artifact(
        corpus_id, expected_generation=corpus.checkpoint.generation
    )
    export_id = str(uuid4())
    exported_at, _ = await storage.export_assertion_snapshot(
        artifact_id=export_id, consumer_id="parametric-self-test",
        consumer_key_id="parametric-self-key-v1",
        consumer_public_key=_CORPUS_CONSUMER_PUBLIC_KEY, retention_seconds=300,
    )

    break_manifest(exclusion_authority, storage.agent_id)
    with pytest.raises(CorpusExclusionManifestError, match=message):
        await governed.consume_governed_semantic_artifact(
            corpus_id, expected_generation=corpus.checkpoint.generation
        )
    assert await storage.consume_governed_semantic_artifact(
        export_id, expected_generation=exported_at.generation
    )
    assert corpus_id in await _active_artifact_ids(storage)
    assert await _pending_corpus_revocations(storage) == []


@pytest.mark.asyncio
async def test_a_rolled_back_manifest_cannot_reopen_a_stale_corpus(
    file_storage: AsyncStorage, exclusion_authority: ExclusionAuthority
) -> None:
    """Consumption admits the manifest it reads, so putting back the old one fails.

    Without that, an agent that restored the validly signed sequence 1 would
    make a corpus built under it, exam and all, current again. Sequence 2 is
    admitted here by a consumption alone, with no build under it.
    """
    from kestrel_sovereign.knowledge.corpus_exclusions import CorpusExclusionManifestError

    storage = file_storage
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    (exam_source,) = await governed.list_assertion_revision_sources(exam["revision_id"])
    policy = await _taught_fact_policy(storage, exam)
    older = json.loads(exclusion_authority.manifest_path.read_text())
    with_exam, with_exam_id = await _registered_corpus(governed, policy)
    exclusion_authority.exclude(
        storage.agent_id, source_occurrence=[exam_source.source_occurrence_id]
    )
    with pytest.raises(GovernedArtifactError, match=_STALE_CORPUS.format(1, 2)):
        await governed.consume_governed_semantic_artifact(
            with_exam_id, expected_generation=with_exam.checkpoint.generation
        )

    exclusion_authority.write(older)
    with pytest.raises(CorpusExclusionManifestError, match="lower than sequence 2"):
        await governed.consume_governed_semantic_artifact(
            with_exam_id, expected_generation=with_exam.checkpoint.generation
        )


@pytest.mark.asyncio
async def test_a_corpus_registered_after_its_manifest_was_replaced_is_not_served(
    file_storage: AsyncStorage, exclusion_authority: ExclusionAuthority, monkeypatch
) -> None:
    """A build that read sequence 1 and registers after sequence 2 is refused.

    Its registration is harmless: consumption rejects it, and it withdraws
    nothing, so the corpus built meanwhile under sequence 2 stays served.
    """
    storage = file_storage
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    (exam_source,) = await governed.list_assertion_revision_sources(exam["revision_id"])
    policy = await _taught_fact_policy(storage, exam)
    register = storage._register_produced_semantic_artifact
    paused, release = asyncio.Event(), asyncio.Event()

    async def register_late(**kwargs):
        if kwargs["exclusion_manifest"].sequence == 1:
            paused.set()
            await release.wait()
        return await register(**kwargs)

    monkeypatch.setattr(storage, "_register_produced_semantic_artifact", register_late)
    late_build = asyncio.create_task(_registered_corpus(governed, policy))
    await asyncio.wait_for(paused.wait(), 10)

    exclusion_authority.exclude(
        storage.agent_id, source_occurrence=[exam_source.source_occurrence_id]
    )
    current, current_id = await _registered_corpus(governed, policy)
    assert current.examples == ()
    release.set()
    late, late_id = await asyncio.wait_for(late_build, 10)

    assert [item.assertion.revision_id for item in late.examples] == [exam["revision_id"]]
    assert late.exclusion_manifest.sequence == 1
    assert await _active_artifact_ids(storage) == {late_id, current_id}
    with pytest.raises(GovernedArtifactError, match=_STALE_CORPUS.format(1, 2)):
        await governed.consume_governed_semantic_artifact(
            late_id, expected_generation=late.checkpoint.generation
        )
    assert await governed.consume_governed_semantic_artifact(
        current_id, expected_generation=current.checkpoint.generation
    )
    assert await _pending_corpus_revocations(storage) == []


@pytest.mark.asyncio
async def test_a_consumption_overtaken_by_a_newer_admitted_manifest_is_refused(
    file_storage: AsyncStorage, exclusion_authority: ExclusionAuthority, monkeypatch
) -> None:
    """Consumption judges a corpus against the manifest the agent admitted last.

    A consumption reads sequence 1, then a build installs and admits sequence
    2, which excludes the exam, before the consumption reaches the store.
    The store reads the watermark in the consumption's own transaction, so the
    sequence-1 corpus, exam and all, is not served; the rebuild is.
    """
    storage = file_storage
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    (exam_source,) = await governed.list_assertion_revision_sources(exam["revision_id"])
    policy = await _taught_fact_policy(storage, exam)
    with_exam, with_exam_id = await _registered_corpus(governed, policy)
    read_manifest = storage._host_corpus_exclusions
    overtaken = []

    async def read_then_overtaken(*, deadline):
        exclusions = await read_manifest(deadline=deadline)
        if not overtaken:
            overtaken.append(exclusions.receipt.sequence)
            exclusion_authority.exclude(
                storage.agent_id, source_occurrence=[exam_source.source_occurrence_id]
            )
            overtaken.append(await _registered_corpus(governed, policy))
        return exclusions

    monkeypatch.setattr(storage, "_host_corpus_exclusions", read_then_overtaken)
    with pytest.raises(
        GovernedArtifactError,
        match="admitted corpus exclusion manifest is sequence 2 .*not the sequence 1",
    ):
        await governed.consume_governed_semantic_artifact(
            with_exam_id, expected_generation=with_exam.checkpoint.generation
        )
    assert overtaken[0] == 1
    rebuilt, rebuilt_id = overtaken[1]
    assert rebuilt.examples == ()
    assert rebuilt.exclusion_manifest.sequence == 2
    assert await governed.consume_governed_semantic_artifact(
        rebuilt_id, expected_generation=rebuilt.checkpoint.generation
    )
    assert await _pending_corpus_revocations(storage) == []


@pytest.mark.asyncio
async def test_a_corpus_registered_before_manifests_existed_is_refused(
    storage: AsyncStorage,
) -> None:
    """A corpus row with no manifest was built with no exclusions at all."""
    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "sealed-exam-answer")
    policy = await _taught_fact_policy(storage, exam)
    corpus, corpus_id = await _registered_corpus(governed, policy)
    await storage.db.execute(
        "UPDATE semantic_governed_artifacts SET exclusion_manifest_sha256 = NULL, "
        "exclusion_manifest_sequence = NULL WHERE tenant_id = ? AND artifact_id = ?",
        (storage.agent_id, corpus_id),
    )

    with pytest.raises(GovernedArtifactError, match="names no corpus exclusion manifest"):
        await storage.consume_governed_semantic_artifact(
            corpus_id, expected_generation=corpus.checkpoint.generation
        )
    assert corpus_id in await _active_artifact_ids(storage)
    assert await _pending_corpus_revocations(storage) == []


@pytest.mark.asyncio
async def test_the_store_serves_a_corpus_only_under_the_manifest_it_was_built_under(
    storage: AsyncStorage,
) -> None:
    """The store's own check: equal receipts or a named refusal, never a write.

    The receipt it is handed must also be the one the agent's watermark
    records, read in the consumption's own transaction (#3893).
    """
    from kestrel_sovereign.knowledge import CorpusExclusionReceipt
    from kestrel_sovereign.storage.corpus_exclusion_watermarks import admit_manifest

    private_key = Ed25519PrivateKey.generate()
    assertion = _assertion(storage.agent_id, "store-binding-revision")
    await storage.put_assertion(assertion, source_occurrences=(_source(),))
    generation = (await storage.assertion_checkpoint()).generation
    admitted = _admitted_receipt()
    store = storage._assertion_store()
    served = _registration(assertion, generation, private_key=private_key)
    stale = _registration(assertion, generation, private_key=private_key)
    await store.register_governed_artifact(served)
    await store.register_governed_artifact(stale)

    with pytest.raises(GovernedArtifactError, match="current, verified exclusion manifest"):
        await store.consume_governed_artifact(
            served.artifact_id, expected_generation=generation
        )
    # A receipt nothing has admitted is not established as current.
    with pytest.raises(GovernedArtifactError, match="no corpus exclusion manifest has been admitted"):
        await store.consume_governed_artifact(
            served.artifact_id, expected_generation=generation, exclusion_manifest=admitted
        )
    await admit_manifest(storage.db, agent_id=storage.agent_id, receipt=admitted)
    assert await store.consume_governed_artifact(
        served.artifact_id, expected_generation=generation, exclusion_manifest=admitted
    )
    # Same sequence, other content: not the manifest the agent admitted.
    resigned = CorpusExclusionReceipt("e" * 64, admitted.sequence, admitted.issued_at)
    with pytest.raises(GovernedArtifactError, match="not the sequence"):
        await store.consume_governed_artifact(
            served.artifact_id, expected_generation=generation, exclusion_manifest=resigned
        )
    newer = CorpusExclusionReceipt("e" * 64, admitted.sequence + 1, admitted.issued_at)
    await admit_manifest(storage.db, agent_id=storage.agent_id, receipt=newer)
    with pytest.raises(GovernedArtifactError, match="not the current manifest"):
        await store.consume_governed_artifact(
            stale.artifact_id, expected_generation=generation, exclusion_manifest=newer
        )
    # Once a newer manifest is admitted, the receipt a consumption read before
    # it no longer serves the corpus built under that receipt.
    with pytest.raises(
        GovernedArtifactError,
        match=f"admitted corpus exclusion manifest is sequence {newer.sequence} .*"
        f"not the sequence {admitted.sequence}",
    ):
        await store.consume_governed_artifact(
            served.artifact_id, expected_generation=generation, exclusion_manifest=admitted
        )
    assert await _active_artifact_ids(storage) == {served.artifact_id, stale.artifact_id}
    assert await _pending_corpus_revocations(storage) == []


def test_a_corpus_registration_must_name_its_manifest_and_an_export_must_not() -> None:
    assertion = _assertion("did:example:registration-binding")
    corpus = _registration(
        assertion, 1, private_key=Ed25519PrivateKey.generate(),
        exclusion_manifest=_elsewhere_receipt(),
    )
    with pytest.raises(GovernedArtifactError, match="must name the exclusion manifest"):
        replace(corpus, exclusion_manifest_sha256=None)
    with pytest.raises(GovernedArtifactError, match="exclusion_manifest_sha256"):
        replace(corpus, exclusion_manifest_sha256="d" * 63)
    with pytest.raises(GovernedArtifactError, match="positive integer"):
        replace(corpus, exclusion_manifest_sequence=0)
    with pytest.raises(GovernedArtifactError, match="export_snapshot artifact is not built"):
        replace(corpus, kind=GovernedArtifactKind.EXPORT_SNAPSHOT)
    assert replace(
        corpus,
        kind=GovernedArtifactKind.EXPORT_SNAPSHOT,
        exclusion_manifest_sha256=None,
        exclusion_manifest_sequence=None,
    ).kind is GovernedArtifactKind.EXPORT_SNAPSHOT


@pytest.mark.asyncio
async def test_the_manifest_is_read_off_the_event_loop_within_the_build_budget(
    storage: AsyncStorage, monkeypatch
) -> None:
    """A manifest read that does not return cannot stall the host (#3893).

    The read runs in a worker thread while the loop keeps serving, and it
    spends the build's own ``max_wall_time_seconds``.
    """
    import threading
    import time

    from kestrel_sovereign.knowledge import GovernedCorpusBudgetExceeded, GovernedCorpusLimits
    from kestrel_sovereign.knowledge import corpus_exclusions
    from kestrel_sovereign.knowledge.corpus_exclusions import CorpusExclusionManifestError

    feature, agent, governed = _fact_feature(storage)
    with live_turn(agent, None):
        exam = await _teach(feature, "exam-turn", "an-answer")
    policy = await _taught_fact_policy(storage, exam)
    released = threading.Event()

    def stalled_read(**_kwargs):
        if not released.wait(5):
            raise CorpusExclusionManifestError("test stall was never released")
        raise CorpusExclusionManifestError("released")

    monkeypatch.setattr(corpus_exclusions, "load_corpus_exclusions", stalled_read)
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    beating = asyncio.create_task(heartbeat())
    started = time.monotonic()
    try:
        with pytest.raises(GovernedCorpusBudgetExceeded, match="exclusion manifest"):
            await storage.governed_assertion_corpus_snapshot(
                policy=policy, inference_profile=None,
                limits=GovernedCorpusLimits(max_wall_time_seconds=0.3),
                artifact_id=str(uuid4()), consumer_id="parametric-self-test",
                consumer_key_id="parametric-self-key-v1",
                consumer_public_key=_CORPUS_CONSUMER_PUBLIC_KEY, retention_seconds=300,
            )
        elapsed = time.monotonic() - started
    finally:
        released.set()
        beating.cancel()
    assert elapsed < 2.0
    assert ticks >= 5
    assert await _corpus_artifact_count(storage) == 0


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
