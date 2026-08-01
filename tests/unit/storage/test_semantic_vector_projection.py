"""Durable assertion-vector projection contracts (#2830)."""

from __future__ import annotations

import hashlib
import asyncio
from collections.abc import Sequence
from types import SimpleNamespace

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
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import (
    PrivacyEnforcingStorage,
    PrivacyViolationError,
)
from kestrel_sovereign.storage.semantic_vector_projection import (
    SemanticVectorProfile, SemanticVectorProjectionError,
    SemanticVectorEmbeddingProvider,
    _resolve_host_semantic_vector_embedding_provider,
)


ONTOLOGY = OntologyRef(
    "http://www.w3.org/2000/01/rdf-schema#", "1.0.0",
    "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5", "semantic-kb-v1",
)
PROFILE = SemanticVectorProfile(
    "semantic-assertion-test-v1", "a" * 64,
    provider="test-provider", model="test-model", dimension=2,
)


class _EmbeddingService:
    def __init__(self, profile, embedder):
        self.configure(profile, embedder)

    def configure(self, profile, embedder):
        self.profile = profile
        self.embedder = embedder

    def describe(self):
        return SimpleNamespace(
            provider=self.profile.provider, model=self.profile.model,
            dim=self.profile.dimension, profile_id=self.profile.profile_id,
        )

    def semantic_vector_destination(self):
        return self.profile.embedding_destination

    async def aembed(self, text):
        return await self.embedder(text)


class _HostLLM:
    def __init__(self, service):
        self.service = service

    def get_embedding_service(self):
        return self.service


async def _storage(tmp_path, label: str) -> AsyncStorage:
    identity_dir = tmp_path / f"identity-{label}"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name=f"Vector {label}",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    service = _EmbeddingService(PROFILE, _embed)
    storage = AsyncStorage(
        str(tmp_path / f"{label}.db"), agent_id=credentials.agent_did,
        llm_service=_HostLLM(service),
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(
            credentials.agent_did, identity,
        ),
    )
    storage._test_vector_service = service
    await storage.initialize()
    return storage


def _source(name: str) -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id=f"source:{name}", source_kind="test",
        locator=f"private://test/{name}", received_at="2026-08-01T00:00:00Z",
        content_digest=f"sha256:{hashlib.sha256(name.encode()).hexdigest()}", actor="test", selector="body",
    )


def _assertion(
    storage: AsyncStorage, name: str, *, visibility="private", privacy="normal",
) -> Assertion:
    source = _source(name)
    return Assertion(
        tenant_id=storage.agent_id, owning_agent_id=storage.agent_id,
        subject=IRI(f"urn:kestrel:agent:{storage.agent_id}:principal:{name}"),
        predicate=IRI("https://example.test/vector"), object=Literal(name, XSD_STRING),
        revision_id=f"revision:{name}", confidence="1", confidence_method="test", confidence_basis="test",
        epistemic_state=EpistemicState.REPORTED, asserted_at="2026-08-01T00:00:00Z",
        ontology_version=ONTOLOGY, lineage=DirectLineage((source.source_occurrence_id,)),
        privacy_classification=privacy, release_policy_reference="policy:test",
        visibility=visibility,
    )


async def _embed(text: str):
    # Deterministic local stand-in for a real embedding capability; the
    # projection receives vectors only through this injected provider boundary.
    return [float(len(text)), float(sum(text.encode()) % 31 + 1)]


def _projection(storage, profile: SemanticVectorProfile, embedder=_embed):
    storage._test_vector_service.configure(profile, embedder)
    return storage.semantic_assertion_vector_projection(profile)


@pytest.mark.asyncio
async def test_projection_is_assertion_revision_linked_and_canonically_fenced(tmp_path):
    storage = await _storage(tmp_path, "lineage")
    try:
        fact = _assertion(storage, "alpha")
        await storage.put_assertion(fact, source_occurrences=(_source("alpha"),))
        projection = _projection(storage, PROFILE)
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
        projection = _projection(storage, PROFILE)
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
        restarted = _projection(storage, PROFILE)
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
        own_projection = _projection(owner, PROFILE)
        other_projection = _projection(other, PROFILE)
        await own_projection.sync()
        assert (await own_projection.recall([1.0, 1.0]))
        empty = await other_projection.sync()
        assert empty.generation == 0
        assert await other_projection.recall([1.0, 1.0]) == ()
    finally:
        await owner.close()
        await other.close()


@pytest.mark.asyncio
async def test_same_generation_partial_page_never_becomes_recall_ready(tmp_path):
    storage = await _storage(tmp_path, "partial-page")
    try:
        original = _assertion(storage, "before")
        await storage.put_assertion(original, source_occurrences=(_source("before"),))
        projection = _projection(storage, PROFILE)
        await projection.sync()

        replacement = _assertion(storage, "after")
        result = await storage.supersede_assertion(
            original.revision_id,
            replacement,
            source_occurrences=(_source("after"),),
        )
        partial = await projection.sync(limit=1)
        terminal = await storage._assertion_store().event_checkpoint()
        assert partial.generation == result.generation == terminal.generation
        assert partial.event_id != terminal.latest_event_id
        with pytest.raises(SemanticVectorProjectionError, match="checkpoint_stale"):
            await projection.recall([1.0, 1.0])
        with pytest.raises(SemanticVectorProjectionError, match="checkpoint_stale"):
            await projection.erasure_observation()

        await projection.sync(limit=1)
        assert [hit.assertion_id for hit in await projection.recall([1.0, 1.0])] == [
            replacement.assertion_id
        ]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_profile_dimension_drift_and_timeout_fail_without_advancing(tmp_path):
    storage = await _storage(tmp_path, "bounds")
    try:
        fact = _assertion(storage, "bounded")
        await storage.put_assertion(fact, source_occurrences=(_source("bounded"),))

        async def wrong_dimension(_text):
            return [1.0]

        projection = _projection(storage, PROFILE, wrong_dimension)
        with pytest.raises(SemanticVectorProjectionError, match="dimension"):
            await projection.sync()
        assert (await projection.checkpoint()).generation == 0

        async def slow(_text):
            import asyncio

            await asyncio.sleep(0.05)
            return [1.0, 1.0]

        timeout_profile = SemanticVectorProfile(
            "semantic-assertion-timeout-v1", "b" * 64,
            provider="test-provider", model="test-model", dimension=2,
            embed_timeout_seconds=0.001,
        )
        timed = _projection(storage, timeout_profile, slow)
        with pytest.raises(SemanticVectorProjectionError, match="timed out"):
            await timed.sync()
        assert (await timed.checkpoint()).generation == 0

        valid_profile = SemanticVectorProfile(
            "semantic-assertion-drift-v1", "d" * 64,
            provider="test-provider", model="test-model", dimension=2,
        )
        valid = _projection(storage, valid_profile)
        await valid.sync()
        await storage.db.execute(
            "UPDATE semantic_assertion_vector_projection_entries SET embedding_model = ? "
            "WHERE tenant_id = ? AND profile_id = ?",
            ("other-model", storage.agent_id, valid_profile.profile_id),
        )
        with pytest.raises(SemanticVectorProjectionError, match="profile_drift"):
            await valid.recall([1.0, 1.0])
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_remote_destination_rejects_private_content_before_embedder(tmp_path):
    storage = await _storage(tmp_path, "remote-private")
    seen: list[str] = []

    async def remote_embed(text: str):
        seen.append(text)
        return [1.0, 1.0]

    try:
        sensitive = _assertion(storage, "secret-medical-detail")
        await storage.put_assertion(
            sensitive, source_occurrences=(_source("secret-medical-detail"),),
        )
        remote = SemanticVectorProfile(
            "semantic-assertion-remote-v1", "e" * 64,
            provider="remote-provider", model="remote-model", dimension=2,
            embedding_destination="remote",
        )
        projection = _projection(storage, remote, remote_embed)
        with pytest.raises(SemanticVectorProjectionError, match="remote destination"):
            await projection.sync()
        assert seen == []
        assert (await projection.checkpoint()).generation == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_remote_requires_public_visibility_and_public_privacy(tmp_path):
    storage = await _storage(tmp_path, "remote-effective-public")
    seen: list[str] = []

    async def remote_embed(text: str):
        seen.append(text)
        return [1.0, 1.0]

    try:
        mixed = _assertion(storage, "public-label-private-policy", visibility="public")
        await storage.put_assertion(mixed, source_occurrences=(_source("public-label-private-policy"),))
        remote = SemanticVectorProfile(
            "semantic-assertion-remote-public-v1", "1" * 64,
            provider="remote-provider", model="remote-model", dimension=2,
            embedding_destination="remote", visibility_ceiling="public",
            privacy_ceiling="normal",
        )
        projection = _projection(storage, remote, remote_embed)
        with pytest.raises(SemanticVectorProjectionError, match="effective-public"):
            await projection.sync()
        assert seen == []
        assert (await projection.checkpoint()).generation == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_raw_callable_cannot_claim_a_fake_local_destination(tmp_path):
    storage = await _storage(tmp_path, "fake-local")
    called = False

    async def disguised_remote(_text):
        nonlocal called
        called = True
        return [1.0, 1.0]

    try:
        with pytest.raises(TypeError):
            storage.semantic_assertion_vector_projection(PROFILE, disguised_remote)
        with pytest.raises(TypeError, match="issued by the host"):
            SemanticVectorEmbeddingProvider(
                object(), provider=PROFILE.provider, model=PROFILE.model,
                profile_id=PROFILE.profile_id, dimension=PROFILE.dimension,
                destination="local", service=storage._test_vector_service,
            )
        with pytest.raises(TypeError, match="host authority"):
            _resolve_host_semantic_vector_embedding_provider(
                PROFILE, _HostLLM(storage._test_vector_service), host_authority=object(),
            )
        assert called is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_oversize_vector_is_rejected_before_iteration(tmp_path):
    storage = await _storage(tmp_path, "oversize")

    class OversizeSequence(Sequence):
        def __len__(self):
            return 8_193

        def __getitem__(self, _index):
            raise AssertionError("oversize vector must not be materialized")

    async def oversized(_text):
        return OversizeSequence()

    try:
        fact = _assertion(storage, "oversize")
        await storage.put_assertion(fact, source_occurrences=(_source("oversize"),))
        projection = _projection(storage, PROFILE, oversized)
        with pytest.raises(SemanticVectorProjectionError, match="dimension"):
            await projection.sync()
        assert (await projection.checkpoint()).generation == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_erasure_rebuild_budget_leaves_projection_unready(tmp_path):
    storage = await _storage(tmp_path, "rebuild-budget")
    bounded = SemanticVectorProfile(
        "semantic-assertion-bounded-rebuild-v1", "f" * 64,
        provider="test-provider", model="test-model", dimension=2,
        rebuild_page_size=1, max_rebuild_rows=1,
    )
    try:
        assertions = [_assertion(storage, name) for name in ("erase", "keep-a", "keep-b")]
        for assertion, name in zip(assertions, ("erase", "keep-a", "keep-b")):
            await storage.put_assertion(assertion, source_occurrences=(_source(name),))
        projection = _projection(storage, bounded)
        await projection.sync()
        await storage.erase_assertion(assertions[0].assertion_id, operation_id="bounded-erase")
        with pytest.raises(SemanticVectorProjectionError, match="row budget"):
            await projection.sync()
        with pytest.raises(SemanticVectorProjectionError, match="checkpoint_stale"):
            await projection.recall([1.0, 1.0])
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_v1_projection_schema_is_disposed_and_rebuilt(tmp_path):
    from kestrel_sovereign.storage.sqla.migrations import migrate_semantic_vector_projection

    storage = await _storage(tmp_path, "legacy-v1")
    try:
        await storage.db.execute("DROP TABLE semantic_assertion_vector_projection_entries", ())
        await storage.db.execute("DROP TABLE semantic_assertion_vector_projection_state", ())
        await storage.db.execute(
            "DELETE FROM semantic_schema_migrations WHERE version = ?",
            ("semantic_assertion_vector_projection_v2",),
        )
        await storage.db.execute(
            "INSERT INTO semantic_schema_migrations (version) VALUES (?)",
            ("semantic_assertion_vector_projection_v1",),
        )
        await storage.db.execute(
            "CREATE TABLE semantic_assertion_vector_projection_entries ("
            "tenant_id TEXT, profile_id TEXT, capability_digest TEXT, assertion_id TEXT)",
            (),
        )
        await storage.db.execute(
            "CREATE TABLE semantic_assertion_vector_projection_state (tenant_id TEXT)",
            (),
        )
        await storage.db.execute(
            "INSERT INTO semantic_assertion_vector_projection_entries VALUES (?, ?, ?, ?)",
            (storage.agent_id, "legacy", "a" * 64, "sensitive-legacy-id"),
        )

        await migrate_semantic_vector_projection(storage.db)
        columns = {
            row[1] for row in await storage.db.fetchall(
                "PRAGMA table_info(semantic_assertion_vector_projection_entries)", ()
            )
        }
        assert columns == {
            "tenant_id", "profile_id", "capability_digest", "embedding_provider",
            "embedding_model", "embedding_dimension", "renderer_version",
            "embedding_destination", "visibility_ceiling", "privacy_ceiling",
            "visibility", "privacy_classification", "assertion_id", "revision_id",
            "revision_digest", "source_generation", "vector_json", "created_at",
        }
        state_columns = {
            row[1] for row in await storage.db.fetchall(
                "PRAGMA table_info(semantic_assertion_vector_projection_state)", ()
            )
        }
        assert state_columns == {
            "tenant_id", "profile_id", "capability_digest", "checkpoint_generation",
            "checkpoint_event_id", "updated_at",
        }
        assert await storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertion_vector_projection_entries"
        ) == 0
        assert not await storage.db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            ("semantic_assertion_vector_projection_v1",),
        )
        assert await storage.db.fetchone(
            "SELECT 1 FROM semantic_schema_migrations WHERE version = ?",
            ("semantic_assertion_vector_projection_v2",),
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_retained_projection_handle_rechecks_privacy_after_mode_transition(tmp_path):
    storage = await _storage(tmp_path, "privacy-transition")
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    try:
        storage._test_vector_service.configure(PROFILE, _embed)
        handle = wrapper.semantic_assertion_vector_projection(PROFILE)
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        with pytest.raises(PrivacyViolationError):
            await handle.sync()
        with pytest.raises(PrivacyViolationError):
            await handle.recall([1.0, 1.0])
        with pytest.raises(PrivacyViolationError):
            await handle.recall_hydrated([1.0, 1.0])
        with pytest.raises(PrivacyViolationError):
            await handle.erasure_observation()
        wrapper.set_privacy_mode(PrivacyMode.ISOLATED)
        with pytest.raises(PrivacyViolationError):
            await handle.sync()
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_mode_transition_refuses_during_blocked_vector_provider_and_recall(tmp_path):
    storage = await _storage(tmp_path, "privacy-lease-concurrency")
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    embed_started = asyncio.Event()
    release_embed = asyncio.Event()

    async def blocked_embed(_text):
        embed_started.set()
        await release_embed.wait()
        return [1.0, 1.0]

    try:
        fact = _assertion(storage, "lease-sensitive")
        await storage.put_assertion(fact, source_occurrences=(_source("lease-sensitive"),))
        storage._test_vector_service.configure(PROFILE, blocked_embed)
        handle = wrapper.semantic_assertion_vector_projection(PROFILE)
        sync_task = asyncio.create_task(handle.sync())
        await asyncio.wait_for(embed_started.wait(), timeout=2)
        with pytest.raises(PrivacyViolationError, match="vector operation"):
            wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        assert wrapper.privacy_mode is PrivacyMode.NORMAL
        release_embed.set()
        await sync_task

        recall_started = asyncio.Event()
        release_recall = asyncio.Event()
        original_recall = handle._projection.recall

        async def blocked_recall(*args, **kwargs):
            recall_started.set()
            await release_recall.wait()
            return await original_recall(*args, **kwargs)

        handle._projection.recall = blocked_recall
        recall_task = asyncio.create_task(handle.recall([1.0, 1.0]))
        await asyncio.wait_for(recall_started.wait(), timeout=2)
        with pytest.raises(PrivacyViolationError, match="vector operation"):
            wrapper.set_privacy_mode(PrivacyMode.ISOLATED)
        assert wrapper.privacy_mode is PrivacyMode.NORMAL
        release_recall.set()
        assert await recall_task
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        assert wrapper.privacy_mode is PrivacyMode.EPHEMERAL
    finally:
        release_embed.set()
        await storage.close()
