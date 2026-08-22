"""Direct storage contracts for governed semantic-recall discovery/hydration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.knowledge import (
    Assertion,
    DirectLineage,
    EpistemicState,
    IRI,
    InferenceProfile,
    Literal,
    OntologyRef,
    SourceOccurrence,
    TemporalInterval,
    XSD_STRING,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage.async_assertion_store import SemanticRecallUnavailableError
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage, PrivacyViolationError


ONTOLOGY = OntologyRef(
    "http://www.w3.org/2000/01/rdf-schema#", "1.0.0",
    "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5", "semantic-kb-v1",
)
PROFILE = InferenceProfile(ONTOLOGY, "1.0.0")


async def _storage(tmp_path, label="recall"):
    identity_dir = tmp_path / label
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name=f"Recall {label}",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    storage = AsyncStorage(
        ":memory:", agent_id=credentials.agent_did,
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(
            credentials.agent_did, identity,
        ),
    )
    await storage.initialize()
    return storage


def _source(revision: str) -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id=f"source:{revision}", source_kind="recall-test",
        locator=f"private://body/{revision}", received_at="2026-07-27T00:00:00Z",
        content_digest=f"sha256:{revision}", actor="operator", selector="private-body",
    )


def _assertion(storage, revision: str, *, valid_time=None) -> Assertion:
    source = _source(revision)
    return Assertion(
        tenant_id=storage.agent_id, owning_agent_id=storage.agent_id,
        subject=IRI(f"urn:kestrel:agent:{storage.agent_id}:principal:user"),
        predicate=IRI("https://example.test/recall"), object=Literal(revision, XSD_STRING),
        revision_id=revision, confidence="1", confidence_method="test", confidence_basis="test",
        epistemic_state=EpistemicState.REPORTED, asserted_at="2026-07-27T00:00:00Z",
        ontology_version=ONTOLOGY, lineage=DirectLineage((source.source_occurrence_id,)),
        privacy_classification="normal", release_policy_reference="policy:test", valid_time=valid_time,
    )


async def _put(storage, revision: str, **kwargs) -> Assertion:
    item = _assertion(storage, revision, **kwargs)
    result = await storage.put_assertion(item, source_occurrences=(_source(revision),))
    assert result.accepted
    return item


async def _ready(storage):
    # Maintenance deliberately reports PARTIAL after source/projection changes
    # until its bounded cursor has replayed the generation.  Drive its public
    # lifecycle to a current checkpoint instead of manufacturing readiness.
    for _ in range(4):
        result = await storage.run_semantic_maintenance(PROFILE)
        if result.status.value in {"complete", "no_op"}:
            return
    pytest.fail(f"maintenance did not reach a current checkpoint: {result}")


@pytest.mark.asyncio
async def test_semantic_recall_is_tenant_bound_and_privacy_denial_never_calls_inner(tmp_path):
    storage = await _storage(tmp_path, "owner")
    other = await _storage(tmp_path, "other")
    try:
        fact = await _put(storage, "owner-fact")
        await _ready(storage)
        await _ready(other)
        discovered = await storage.semantic_recall_candidates(
            query="fact", candidate_scan_limit=10, inference_profile=PROFILE,
        )
        assert [item.assertion.assertion_id for item in discovered.candidates] == [fact.assertion_id]
        # Exercise the canonical AsyncAssertionStore implementation too; the
        # AsyncStorage method is deliberately only its initialized facade.
        direct_store = await storage._assertion_store().recall_candidates(
            query="fact", candidate_scan_limit=10, inference_profile=PROFILE,
        )
        assert direct_store.checkpoint_generation == discovered.checkpoint_generation
        assert (await storage._assertion_store().hydrate_recall_candidates(
            [fact.assertion_id], expected_checkpoint_generation=direct_store.checkpoint_generation,
            inference_profile=PROFILE,
        ))[0].assertion == fact
        assert not (await other.semantic_recall_candidates(
            query="fact", candidate_scan_limit=10, inference_profile=PROFILE,
        )).candidates

        inner = AsyncMock()
        denied = PrivacyEnforcingStorage(inner, PrivacyMode.EPHEMERAL)
        with pytest.raises(PrivacyViolationError):
            await denied.semantic_recall_candidates(
                query="fact", candidate_scan_limit=1, inference_profile=PROFILE,
            )
        inner.semantic_recall_candidates.assert_not_awaited()
    finally:
        await other.close()
        await storage.close()


@pytest.mark.asyncio
async def test_semantic_recall_discovers_only_current_valid_eligible_rows_and_exact_window(tmp_path):
    storage = await _storage(tmp_path)
    try:
        now = datetime.now(timezone.utc)
        current = await _put(storage, "current")
        await _put(storage, "expired", valid_time=TemporalInterval(end=now - timedelta(seconds=1)))
        await _put(storage, "future", valid_time=TemporalInterval(start=now + timedelta(days=1)))
        old = await _put(storage, "old")
        replacement = _assertion(storage, "replacement")
        await storage.supersede_assertion(old.revision_id, replacement, source_occurrences=(_source("replacement"),))
        deleted = await _put(storage, "deleted")
        await storage.delete_assertion(deleted.assertion_id, deleted.revision_id)
        retracted = await _put(storage, "retracted")
        await storage.retract_assertion(retracted.assertion_id, retracted.revision_id)
        quarantined = await _put(storage, "quarantined")
        await storage.db.execute(
            "UPDATE semantic_assertion_revisions SET status = 'quarantined', eligible = 0 "
            "WHERE tenant_id = ? AND revision_id = ?",
            (storage.agent_id, quarantined.revision_id),
        )
        ineligible = await _put(storage, "ineligible")
        await storage.db.execute(
            "UPDATE semantic_projection_eligibility SET eligible = 0 WHERE tenant_id = ? AND revision_id = ?",
            (storage.agent_id, ineligible.revision_id),
        )
        await _ready(storage)
        result = await storage.semantic_recall_candidates(
            query="", candidate_scan_limit=10, inference_profile=PROFILE,
        )
        ids = {item.assertion.revision_id for item in result.candidates}
        assert current.revision_id in ids and replacement.revision_id in ids
        assert not ids.intersection({
            "expired", "future", "old", "deleted", "retracted", "quarantined", "ineligible",
        })

        with pytest.raises(SemanticRecallUnavailableError, match="semantic_recall_candidate_window_exceeded"):
            await storage.semantic_recall_candidates(
                query="", candidate_scan_limit=1, inference_profile=PROFILE,
            )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_semantic_recall_readiness_and_hydration_are_fenced_and_batch_provenance(tmp_path, monkeypatch):
    storage = await _storage(tmp_path)
    try:
        first, second = await _put(storage, "first"), await _put(storage, "second")
        await _ready(storage)
        discovered = await storage.semantic_recall_candidates(
            query="", candidate_scan_limit=10, inference_profile=PROFILE,
        )
        selected = [first.assertion_id, second.assertion_id]
        calls = 0
        original = storage._assertion_store()._database.fetchall

        async def counted(sql, params=()):
            nonlocal calls
            if "semantic_revision_sources" in sql:
                calls += 1
            return await original(sql, params)

        monkeypatch.setattr(storage._assertion_store()._database, "fetchall", counted)
        hydrated = await storage.hydrate_semantic_recall_candidates(
            selected, expected_checkpoint_generation=discovered.checkpoint_generation,
            inference_profile=PROFILE,
        )
        assert calls == 1
        assert {item.assertion.assertion_id for item in hydrated} == set(selected)
        assert {source.source_occurrence_id for item in hydrated for source in item.source_occurrences} == {
            "source:first", "source:second",
        }

        # A row can lose projection authority after discovery without a source
        # generation bump; final hydration must re-authorize, not publish it.
        await storage.db.execute(
            "UPDATE semantic_projection_eligibility SET eligible = 0 WHERE tenant_id = ? AND revision_id = ?",
            (storage.agent_id, second.revision_id),
        )
        reauthorized = await storage.hydrate_semantic_recall_candidates(
            selected, expected_checkpoint_generation=discovered.checkpoint_generation,
            inference_profile=PROFILE,
        )
        assert [item.assertion.assertion_id for item in reauthorized] == [first.assertion_id]

        await storage.delete_assertion(first.assertion_id, first.revision_id)
        with pytest.raises(SemanticRecallUnavailableError, match="semantic_maintenance_checkpoint_behind"):
            await storage.hydrate_semantic_recall_candidates(
                selected, expected_checkpoint_generation=discovered.checkpoint_generation,
                inference_profile=PROFILE,
            )

        await storage.db.execute(
            "UPDATE semantic_maintenance_state SET status = 'partial' WHERE tenant_id = ?",
            (storage.agent_id,),
        )
        with pytest.raises(SemanticRecallUnavailableError, match="semantic_maintenance_partial"):
            await storage.semantic_recall_candidates(
                query="", candidate_scan_limit=10, inference_profile=PROFILE,
            )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_semantic_recall_hydration_rechecks_maintenance_after_provenance_io(tmp_path, monkeypatch):
    """A same-generation COMPLETE→PARTIAL transition cannot publish sources."""
    storage = await _storage(tmp_path)
    try:
        fact = await _put(storage, "post-io-fence")
        await _ready(storage)
        discovered = await storage.semantic_recall_candidates(
            query="", candidate_scan_limit=10, inference_profile=PROFILE,
        )
        database = storage._assertion_store()._database
        original = database.fetchall
        changed = False

        async def downgrade_after_sources(sql, params=()):
            nonlocal changed
            rows = await original(sql, params)
            if "semantic_revision_sources" in sql and not changed:
                changed = True
                await database.execute(
                    "UPDATE semantic_maintenance_state SET status = 'partial' WHERE tenant_id = ?",
                    (storage.agent_id,),
                )
            return rows

        monkeypatch.setattr(database, "fetchall", downgrade_after_sources)
        with pytest.raises(SemanticRecallUnavailableError, match="semantic_maintenance_partial"):
            await storage.hydrate_semantic_recall_candidates(
                [fact.assertion_id], expected_checkpoint_generation=discovered.checkpoint_generation,
                inference_profile=PROFILE,
            )
        assert changed
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("partial", "failed"))
async def test_semantic_recall_requires_current_complete_matching_maintenance(tmp_path, status):
    storage = await _storage(tmp_path, status)
    try:
        await _put(storage, f"{status}-fact")
        await _ready(storage)
        await storage.db.execute(
            "UPDATE semantic_maintenance_state SET status = ? WHERE tenant_id = ?",
            (status, storage.agent_id),
        )
        with pytest.raises(SemanticRecallUnavailableError, match=f"semantic_maintenance_{status}"):
            await storage.semantic_recall_candidates(
                query="", candidate_scan_limit=10, inference_profile=PROFILE,
            )

        await _ready(storage)
        await _put(storage, f"stale-{status}")
        with pytest.raises(SemanticRecallUnavailableError, match="semantic_maintenance_checkpoint_behind"):
            await storage.semantic_recall_candidates(
                query="", candidate_scan_limit=10, inference_profile=PROFILE,
            )

        await _ready(storage)
        await storage.db.execute(
            "UPDATE semantic_maintenance_state SET profile_key = 'wrong-profile' WHERE tenant_id = ?",
            (storage.agent_id,),
        )
        with pytest.raises(SemanticRecallUnavailableError, match="semantic_maintenance"):
            await storage.semantic_recall_candidates(
                query="", candidate_scan_limit=10, inference_profile=PROFILE,
            )
    finally:
        await storage.close()
