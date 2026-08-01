"""Durable assertion-vector projection contracts (#2830)."""

from __future__ import annotations

import hashlib

import pytest

from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.knowledge import (
    Assertion, DirectLineage, EpistemicState, IRI, Literal, OntologyRef,
    SourceOccurrence, XSD_STRING,
)
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.semantic_vector_projection import (
    SemanticVectorProfile, SemanticVectorProjectionError,
)


ONTOLOGY = OntologyRef(
    "http://www.w3.org/2000/01/rdf-schema#", "1.0.0",
    "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5", "semantic-kb-v1",
)
PROFILE = SemanticVectorProfile("semantic-assertion-test-v1", "a" * 64)


async def _storage(tmp_path, label: str) -> AsyncStorage:
    identity_dir = tmp_path / f"identity-{label}"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name=f"Vector {label}",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    storage = AsyncStorage(
        str(tmp_path / f"{label}.db"), agent_id=credentials.agent_did,
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(
            credentials.agent_did, identity,
        ),
    )
    await storage.initialize()
    return storage


def _source(name: str) -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id=f"source:{name}", source_kind="test",
        locator=f"private://test/{name}", received_at="2026-08-01T00:00:00Z",
        content_digest=f"sha256:{hashlib.sha256(name.encode()).hexdigest()}", actor="test", selector="body",
    )


def _assertion(storage: AsyncStorage, name: str) -> Assertion:
    source = _source(name)
    return Assertion(
        tenant_id=storage.agent_id, owning_agent_id=storage.agent_id,
        subject=IRI(f"urn:kestrel:agent:{storage.agent_id}:principal:{name}"),
        predicate=IRI("https://example.test/vector"), object=Literal(name, XSD_STRING),
        revision_id=f"revision:{name}", confidence="1", confidence_method="test", confidence_basis="test",
        epistemic_state=EpistemicState.REPORTED, asserted_at="2026-08-01T00:00:00Z",
        ontology_version=ONTOLOGY, lineage=DirectLineage((source.source_occurrence_id,)),
        privacy_classification="normal", release_policy_reference="policy:test",
    )


async def _embed(text: str):
    # Deterministic local stand-in for a real embedding capability; the
    # projection receives vectors only through this injected provider boundary.
    return [float(len(text)), float(sum(text.encode()) % 31 + 1)]


@pytest.mark.asyncio
async def test_projection_is_assertion_revision_linked_and_canonically_fenced(tmp_path):
    storage = await _storage(tmp_path, "lineage")
    try:
        fact = _assertion(storage, "alpha")
        await storage.put_assertion(fact, source_occurrences=(_source("alpha"),))
        projection = storage.semantic_assertion_vector_projection(PROFILE, _embed)
        checkpoint = await projection.sync()
        assert checkpoint.generation == (await storage.assertion_checkpoint()).generation
        candidates = await projection.recall([1.0, 1.0])
        assert [(hit.assertion_id, hit.revision_id) for hit in candidates] == [
            (fact.assertion_id, fact.revision_id)
        ]

        await storage.retract_assertion(fact.assertion_id, fact.revision_id)
        with pytest.raises(SemanticVectorProjectionError, match="checkpoint_stale"):
            await projection.recall([1.0, 1.0])
        await projection.sync()
        assert await projection.recall([1.0, 1.0]) == ()
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_physical_erasure_rebuilds_unrelated_survivor_after_restart(tmp_path):
    storage = await _storage(tmp_path, "erase")
    try:
        erased = _assertion(storage, "erased")
        survivor = _assertion(storage, "survivor")
        await storage.put_assertion(erased, source_occurrences=(_source("erased"),))
        await storage.put_assertion(survivor, source_occurrences=(_source("survivor"),))
        projection = storage.semantic_assertion_vector_projection(PROFILE, _embed)
        await projection.sync()
        before = await projection.erasure_observation()
        assert before.candidate_count == 2

        await storage.erase_assertion(erased.assertion_id, operation_id="erase-vector-root")
        # The old vector cursor is fenced before it can serve post-delete data.
        with pytest.raises(SemanticVectorProjectionError, match="checkpoint_stale"):
            await projection.erasure_observation()
        # Simulate a crash before delivery.  The erased ordinary outbox rows
        # are gone, so restart must conservatively replay the old cursor and
        # consume the opaque erasure event without reviving the erased vector.
        await storage.close()
        await storage.initialize()
        restarted = storage.semantic_assertion_vector_projection(PROFILE, _embed)
        await restarted.sync()
        assert (await restarted.erasure_observation()).candidate_count == 1
        assert [hit.assertion_id for hit in await restarted.recall([1.0, 1.0])] == [survivor.assertion_id]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_projection_never_crosses_tenant_boundary(tmp_path):
    owner = await _storage(tmp_path, "owner")
    other = await _storage(tmp_path, "other")
    try:
        fact = _assertion(owner, "owner-only")
        await owner.put_assertion(fact, source_occurrences=(_source("owner-only"),))
        own_projection = owner.semantic_assertion_vector_projection(PROFILE, _embed)
        other_projection = other.semantic_assertion_vector_projection(PROFILE, _embed)
        await own_projection.sync()
        assert (await own_projection.recall([1.0, 1.0]))
        empty = await other_projection.sync()
        assert empty.generation == 0
        assert await other_projection.recall([1.0, 1.0]) == ()
    finally:
        await owner.close()
        await other.close()
