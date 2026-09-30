"""The constitution reanchor embeds RAG chunks the way the live agent does (#3418).

``kestrel constitution reanchor`` re-indexes the new constitution through
``chunk_document(compute_embeddings=True)``. It used to resolve the embedding
service with a bare process-local ``LLMService()``, which never applies the
agent's persisted ``embedding_route`` from ``agent_metadata``. On a host whose
first chat route cannot embed (the fleet lists ``anthropic:plan`` first), that
resolves nothing, and all 47 ``KESTREL_CONSTITUTION.md`` chunks per agent were
stored with no vector and no profile id, invisible to vector search (#3415).

These run real inception and a real forced reanchor against a SQLite agent
database. Only ``LLMService`` is replaced: every ``LLMService()`` the code
builds returns :func:`_process_local_service`, a fresh service with the
fleet's shape and no persisted config applied, exactly what a CLI process
constructs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from kestrel_sovereign import cli_embeddings
from kestrel_sovereign.constitution.amendment_artifact import (
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.llm.embedding_service import (
    ProviderEmbeddingService,
    derive_embedding_profile,
)
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
from kestrel_sovereign.setup.constitution_reanchor import reanchor_constitution
from kestrel_sovereign.storage.async_database import AsyncDatabase

CONSTITUTION_V1 = b"""# Kestrel Constitution (Test V1)

## Book I: Universal Values

Honesty. Sovereignty. Transparency.

This is version 1, the anchor at inception.
""" * 6

CONSTITUTION_V2 = b"""# Kestrel Constitution (Test V2 - AMENDED)

## Book I: Universal Values

Honesty. Sovereignty. Transparency. Calibrated uncertainty.

This is version 2; the agent reanchors to it.
""" * 6

PINNED_MODEL = "qwen3-embedding:8b"
DIM = 16
PINNED_PROFILE = derive_embedding_profile(
    provider="ollama", model=PINNED_MODEL, dim=DIM,
).profile_id

_SUITE = Secp256k1Suite()
_ROOT_KEYPAIR = _SUITE.generate_keypair()
_ROOT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000003418"


class _EmbeddingAdapter:
    """An Ollama-shaped adapter that embeds every text at ``DIM``."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def list_embedding_models(self, client):
        return []

    async def aembed_batch(self, client, texts, model=None, **kwargs):
        self.embedded.extend(texts)
        return [[float(len(t) % 5) + i for i in range(DIM)] for t in texts]


class _ProcessLocalService(LLMService):
    """A real ``LLMService`` whose ``close`` records the call.

    The real ``close`` drains route state this double never builds.
    """

    closed = False

    async def close(self) -> None:
        self.closed = True


def _route(name: str, adapter: object, *, capabilities: dict) -> dict:
    vendor, route = name.split(":")
    return {
        "name": name,
        "vendor": vendor,
        "route": route,
        "adapter": adapter,
        "client": object(),
        "model": "auto",
        "is_local": vendor == "ollama",
        "is_cloud": vendor != "ollama",
        "capabilities": capabilities,
    }


def _process_local_service(providers: list[dict]) -> _ProcessLocalService:
    """What ``LLMService()`` builds from static config, before any agent state."""
    service = _ProcessLocalService.__new__(_ProcessLocalService)
    service.providers = providers
    service.disabled = False
    service._disabled_routes = {}
    service._mandate_preference = {}
    service._mandate_fallbacks = []
    service._route_embedding_model_overrides = {}
    service._route_embedding_caps_backup = {}
    service._route_embedding_model_persistence_callback = None
    service._embedding_route_persistence_callback = None
    service._embedding_discovery_cache = []  # discovery finds nothing new
    service._embedding_space_change_warnings = {}
    service._corpus_embedding_profile_provider = None
    service._force_local_only_provider = None
    service._embedding_space_pins = None
    service._verified_space_pins = {}
    service._embedding_route = None
    service._preference_persistence_tasks = set()
    return service


class _Host:
    """Installs ``_process_local_service`` as every ``LLMService()`` built."""

    def __init__(self, monkeypatch, providers_factory) -> None:
        self.adapter = _EmbeddingAdapter()
        self.built: list[_ProcessLocalService] = []
        self._providers_factory = providers_factory

        def _build(*args, **kwargs):
            service = _process_local_service(self._providers_factory(self.adapter))
            self.built.append(service)
            return service

        monkeypatch.setattr("kestrel_sovereign.llm.service.LLMService", _build)


def _fleet_routes(adapter: _EmbeddingAdapter) -> list[dict]:
    """The fleet host: a first chat route that cannot embed, and no sibling.

    ``ollama:local`` advertises embedding support only through the agent's
    persisted model pin, as it does on the fleet.
    """
    return [
        _route("anthropic:plan", object(), capabilities={}),
        _route("ollama:local", adapter, capabilities={}),
    ]


@pytest.fixture(autouse=True)
def _no_ambient_trust_root(monkeypatch):
    monkeypatch.delenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", raising=False)


@pytest.fixture
def constitution_path(tmp_path, monkeypatch) -> Path:
    """The governing source; reanchor refuses any other (#2463)."""
    import kestrel_sovereign.config as ks_config

    path = tmp_path / "KESTREL_CONSTITUTION.md"
    path.write_bytes(CONSTITUTION_V1)
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(path))
    return path


async def _incept(tmp_path: Path, constitution_path: Path) -> tuple[Path, str]:
    agent_dir = tmp_path / "agent_data" / "Emma"
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(constitution_path),
        agent_name="Emma",
    )
    return agent_dir, creds.agent_did


def _persist(db_path: Path, agent_did: str, **values) -> None:
    """Write ``agent_metadata`` the way the settings API/UI persists it."""
    with closing(sqlite3.connect(db_path)) as conn:
        for key, value in values.items():
            conn.execute(
                "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value) "
                "VALUES (?, ?, ?)",
                (agent_did, key, json.dumps(value)),
            )
        conn.commit()


def _persist_fleet_embedding_config(db_path: Path, agent_did: str) -> None:
    """The fleet agents' persisted state: route plus its model pin."""
    _persist(
        db_path,
        agent_did,
        embedding_model_overrides={
            "ollama:local": {"model": PINNED_MODEL, "dim": DIM},
        },
        embedding_route="ollama:local",
    )


def _chunks(db_path: Path, file_hash: str) -> list[tuple[bytes | None, str | None]]:
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(
            "SELECT embedding_vec, embedding_profile_id FROM document_chunks "
            "WHERE file_hash = ? ORDER BY chunk_id",
            (file_hash,),
        ).fetchall()


async def _force_reanchor(tmp_path: Path, agent_dir: Path, constitution_path: Path):
    constitution_path.write_bytes(CONSTITUTION_V2)
    new_hash = hashlib.sha256(CONSTITUTION_V2).hexdigest()
    root_path = tmp_path / "root.did.json"
    root_path.write_text(
        json.dumps(
            did_document_from_legacy_public_key(_ROOT_DID, _ROOT_KEYPAIR.public_key)
        ),
        encoding="utf-8",
    )
    artifact_path = tmp_path / "reanchor.signed.json"
    artifact_path.write_text(
        json.dumps(
            build_legacy_signed_reanchor_artifact(
                signer_did=_ROOT_DID,
                constitution_sha256=new_hash,
                private_key=_ROOT_KEYPAIR.private_key,
                reason="#3418",
            )
        ),
        encoding="utf-8",
    )
    result = await reanchor_constitution(
        agent_name="Emma",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=True,
        authorization="test #3418",
        amendment_artifact_path=artifact_path,
        sovereign_trust_root_path=root_path,
    )
    assert result.reanchored, result.error
    assert result.new_hash == new_hash
    return result


@pytest.mark.asyncio
async def test_reanchor_embeds_through_the_persisted_embedding_route(
    tmp_path, constitution_path, monkeypatch
):
    """The fleet shape: persisted ``ollama:local`` differs from chat ``anthropic:plan``."""
    host = _Host(monkeypatch, _fleet_routes)
    agent_dir, agent_did = await _incept(tmp_path, constitution_path)
    db_path = agent_dir / "kestrel_prime.db"

    # The bug's precondition, measured rather than assumed: inception indexes
    # through a bare service too, and on this host it stores v1 with nothing.
    v1_hash = hashlib.sha256(CONSTITUTION_V1).hexdigest()
    v1_chunks = _chunks(db_path, v1_hash)
    assert v1_chunks
    assert all(vec is None and pid is None for vec, pid in v1_chunks)

    _persist_fleet_embedding_config(db_path, agent_did)
    result = await _force_reanchor(tmp_path, agent_dir, constitution_path)

    stored = _chunks(db_path, result.new_hash)
    assert stored
    assert all(vec is not None for vec, _ in stored), stored
    assert {pid for _, pid in stored} == {PINNED_PROFILE}
    assert _chunks(db_path, v1_hash) == []
    assert result.rag_index is not None
    assert result.rag_index.agent_did == agent_did
    assert result.rag_index.chunks == len(stored)
    assert result.rag_index.unembedded == 0
    assert result.rag_index.misprofiled == 0
    assert result.rag_index.needs_reindex == 0
    assert result.rag_index.reason is None
    # The reanchor closed the service it built for the re-index.
    assert host.built[-1].closed is True


@pytest.mark.asyncio
async def test_reanchor_with_no_embedding_service_reports_the_unembedded_count(
    tmp_path, constitution_path, monkeypatch, caplog
):
    """Nothing resolves: the chunks are stored, counted, and said to lack vectors."""
    _Host(monkeypatch, _fleet_routes)
    agent_dir, agent_did = await _incept(tmp_path, constitution_path)
    db_path = agent_dir / "kestrel_prime.db"

    with caplog.at_level(logging.WARNING):
        result = await _force_reanchor(tmp_path, agent_dir, constitution_path)

    stored = _chunks(db_path, result.new_hash)
    assert stored
    assert all(vec is None and pid is None for vec, pid in stored)
    rag = result.rag_index
    assert rag is not None
    assert rag.chunks == rag.unembedded == rag.needs_reindex == len(stored)
    assert rag.misprofiled == 0
    assert "no embedding-capable provider resolves" in rag.reason
    assert any(
        f"stored {len(stored)} of {len(stored)} constitution chunks" in r.message
        and agent_did in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), [r.message for r in caplog.records]


@pytest.mark.asyncio
async def test_reanchor_does_not_embed_when_the_persisted_route_cannot_apply(
    tmp_path, constitution_path, monkeypatch
):
    """A route the bare service CAN embed with is not the operator's choice.

    ``embeddings reindex`` refuses here rather than stamp rows with a profile
    other than the persisted route's; the reanchor stores the chunks without
    vectors for the same reason, and says why.
    """
    def _chat_route_embeds(adapter):
        return [
            _route(
                "ollama:local",
                adapter,
                capabilities={
                    "supports_embeddings": True,
                    "embedding_model": "nomic-embed-text",
                    "embedding_dim": DIM,
                },
            ),
        ]

    host = _Host(monkeypatch, _chat_route_embeds)
    agent_dir, agent_did = await _incept(tmp_path, constitution_path)
    db_path = agent_dir / "kestrel_prime.db"
    _persist(db_path, agent_did, embedding_route="openrouter:api")
    host.adapter.embedded.clear()

    result = await _force_reanchor(tmp_path, agent_dir, constitution_path)

    stored = _chunks(db_path, result.new_hash)
    assert stored
    assert all(vec is None for vec, _ in stored)
    assert host.adapter.embedded == []
    assert result.rag_index.unembedded == len(stored)
    assert "'openrouter:api' is no longer valid" in result.rag_index.reason


@pytest.mark.asyncio
async def test_reindex_and_reanchor_resolve_the_same_profile(
    tmp_path, monkeypatch, capsys
):
    """``embeddings reindex`` goes through the helper the reanchor uses."""
    _Host(monkeypatch, _fleet_routes)
    db = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    try:
        await db.execute_commit(
            "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
            ("did:x", "embedding_model_overrides",
             json.dumps({"ollama:local": {"model": PINNED_MODEL, "dim": DIM}})),
        )
        await db.execute_commit(
            "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
            ("did:x", "embedding_route", json.dumps("ollama:local")),
        )
        resolution = await cli_embeddings.resolve_agent_embedding(db, "did:x")
        assert resolution.error is None
        assert resolution.profile_id == PINNED_PROFILE

        monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: DIM)
        rc = await cli_embeddings._reindex(
            db, table=None, agent_id="did:x", batch=10, rate_limit=0.0,
            dry_run=True, apply=False,
        )
    finally:
        await db.close()
    assert rc == 0
    assert f"target_profile: {PINNED_PROFILE}" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_service_that_fails_to_close_does_not_fail_a_committed_reanchor(
    tmp_path, constitution_path, monkeypatch
):
    """The service closes after the transaction commits; its error is not the outcome."""
    async def _close_raises(self) -> None:
        raise RuntimeError("adapter would not close")

    monkeypatch.setattr(_ProcessLocalService, "close", _close_raises)
    _Host(monkeypatch, _fleet_routes)
    agent_dir, agent_did = await _incept(tmp_path, constitution_path)
    db_path = agent_dir / "kestrel_prime.db"
    _persist_fleet_embedding_config(db_path, agent_did)

    result = await _force_reanchor(tmp_path, agent_dir, constitution_path)

    assert result.error is None
    assert {pid for _, pid in _chunks(db_path, result.new_hash)} == {PINNED_PROFILE}


def _raise_outage(self, *args, **kwargs):
    raise RuntimeError("provider outage")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner", "method"),
    [
        (_ProcessLocalService, "get_embedding_service"),
        (ProviderEmbeddingService, "current_profile_id"),
    ],
    ids=["get_embedding_service", "current_profile_id"],
)
async def test_a_provider_that_raises_while_resolving_does_not_abort_the_reanchor(
    tmp_path, constitution_path, monkeypatch, caplog, owner, method
):
    """An embedding outage costs the vectors, never the governance write.

    Before resolution was shared, ``chunk_document`` caught a failing
    provider and still stored keyword-searchable chunks. A raise escaping the
    resolution aborted a forced reanchor before its transaction instead.
    """
    host = _Host(monkeypatch, _fleet_routes)
    agent_dir, agent_did = await _incept(tmp_path, constitution_path)
    db_path = agent_dir / "kestrel_prime.db"
    _persist_fleet_embedding_config(db_path, agent_did)
    monkeypatch.setattr(owner, method, _raise_outage)

    with caplog.at_level(logging.WARNING):
        result = await _force_reanchor(tmp_path, agent_dir, constitution_path)

    assert result.error is None
    stored = _chunks(db_path, result.new_hash)
    assert stored
    assert all(vec is None and pid is None for vec, pid in stored)
    rag = result.rag_index
    assert rag.unembedded == rag.needs_reindex == len(stored)
    assert "RuntimeError: provider outage" in rag.reason
    assert any(
        f"stored {len(stored)} of {len(stored)} constitution chunks" in r.message
        and agent_did in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), [r.message for r in caplog.records]
    assert host.built[-1].closed is True


def _raise_on_restamp(self):
    raise RuntimeError("profile lookup failed")


def _foreign_restamp(self):
    return "ollama:nomic-embed-text@16"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("restamp", "stamped"),
    [(_raise_on_restamp, None), (_foreign_restamp, "ollama:nomic-embed-text@16")],
    ids=["null-profile", "foreign-profile"],
)
async def test_a_vector_without_the_agents_profile_counts_as_needing_reindex(
    tmp_path, constitution_path, monkeypatch, caplog, restamp, stamped
):
    """Vector search filters by profile, so a vector is not proof of findability.

    ``chunk_document`` resolves the profile again after embedding and stamps
    NULL when that fails; a service that describes itself differently the
    second time stamps a profile the agent does not search.
    """
    _Host(monkeypatch, _fleet_routes)
    agent_dir, agent_did = await _incept(tmp_path, constitution_path)
    db_path = agent_dir / "kestrel_prime.db"
    _persist_fleet_embedding_config(db_path, agent_did)

    resolve = cli_embeddings.resolve_agent_embedding

    async def _resolve_then_restamp(*args, **kwargs):
        resolution = await resolve(*args, **kwargs)
        assert resolution.profile_id == PINNED_PROFILE
        monkeypatch.setattr(ProviderEmbeddingService, "current_profile_id", restamp)
        return resolution

    monkeypatch.setattr(
        cli_embeddings, "resolve_agent_embedding", _resolve_then_restamp,
    )

    with caplog.at_level(logging.WARNING):
        result = await _force_reanchor(tmp_path, agent_dir, constitution_path)

    stored = _chunks(db_path, result.new_hash)
    assert stored
    assert all(vec is not None for vec, _ in stored)
    assert {pid for _, pid in stored} == {stamped}
    rag = result.rag_index
    assert rag.unembedded == 0
    assert rag.misprofiled == rag.needs_reindex == len(stored)
    assert f"not stamped with the agent's profile {PINNED_PROFILE}" in rag.reason
    assert any(
        f"stored {len(stored)} of {len(stored)} constitution chunks" in r.message
        and agent_did in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), [r.message for r in caplog.records]


@pytest.mark.asyncio
async def test_reindex_refuses_rather_than_crashes_when_the_provider_raises(
    tmp_path, monkeypatch, capsys
):
    """The shared resolution turns a raise into the refusal ``reindex`` prints."""
    _Host(monkeypatch, _fleet_routes)
    monkeypatch.setattr(_ProcessLocalService, "get_embedding_service", _raise_outage)
    db = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    try:
        rc = await cli_embeddings._reindex(
            db, table=None, agent_id="did:x", batch=10, rate_limit=0.0,
            dry_run=True, apply=False,
        )
    finally:
        await db.close()
    assert rc == 2
    assert (
        "ERROR: resolving the embedding service failed "
        "(RuntimeError: provider outage)"
    ) in capsys.readouterr().err
