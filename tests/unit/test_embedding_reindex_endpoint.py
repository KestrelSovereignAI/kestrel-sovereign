"""Unit tests for the UI reindex endpoint ``POST /api/embedding/reindex`` (#2336).

Drives the endpoint coroutines directly (single event loop) against a real
seeded mixed-profile SQLite DB so the background job, its DB writes, and the
progress-polling GET all share one loop — avoiding the cross-loop aiosqlite
hazard a sync ``TestClient`` would introduce.

Covered:

- dry-run reports per-table stale counts and touches nothing;
- execute re-embeds every stale row (scoped to the request's agent) and the
  progress job reaches ``done`` with matching counts;
- refusal (409) when ``embedding_route = "none"``;
- refusal (409) on a resolved-dim ≠ column-width mismatch;
- an unknown table name is a 400.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from kestrel_sovereign.endpoints import models as model_endpoints
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.embedding_reindex import EmbeddingReindexer
from tests.unit.test_embedding_reindex import (
    DIM,
    TARGET,
    FakeEmbeddingService,
    _seed,
)


class DeadEmbeddingService:
    """A service that *resolves* fine but returns empty vectors for every row.

    Models the #2360 live failure: a cached/mis-resolved embedding service
    describes itself correctly (so ``target_profile`` resolves and the dim
    guard passes) but ``aembed_batch`` yields ``None`` per row with no
    exception — the silent scanned-N / reembedded-0 outcome.
    """

    def __init__(self, *, dim: int = DIM):
        self.embedding_dim = dim
        self._dim = dim
        self.calls = 0

    async def aembed_batch(self, texts):
        self.calls += 1
        return [None for _ in texts]

    def current_profile_id(self) -> str:
        return TARGET

    def describe(self):
        return None


class _DimHonoringAdapter:
    """Fake adapter that only returns column-width vectors when ``dimensions``
    is forwarded — otherwise it returns its NATIVE (wider) dimension.

    Models a Matryoshka embedding model (e.g. qwen3-embedding-8b, native 4096)
    reached through a route pinned to a narrower dim. If the service forgets to
    forward ``dimensions`` (the #2371 bug), the adapter hands back native-dim
    vectors the reindexer's column guard then rejects as dim-mismatch — the
    exact "scanned 24, skipped_dim_mismatch 24" failure. When the fix forwards
    ``dimensions``, vectors come back at the pinned width and rows flip.
    """

    NATIVE_DIM = DIM * 2

    def __init__(self):
        self.seen_dimensions: list = []

    def _one(self, text: str, dimensions):
        width = int(dimensions) if dimensions is not None else self.NATIVE_DIM
        base = float(len(text) % 7) + 1.0
        return [base + i for i in range(width)]

    async def aembed(self, client, text, *, model=None, dimensions=None, **kwargs):
        self.seen_dimensions.append(dimensions)
        return self._one(text, dimensions)

    async def aembed_batch(self, client, texts, *, model=None, dimensions=None, **kwargs):
        self.seen_dimensions.append(dimensions)
        return [self._one(t, dimensions) for t in texts]


def _provider_service(dim: int = DIM):
    """A real ``ProviderEmbeddingService`` over a dimension-honoring adapter."""
    from kestrel_sovereign.llm.embedding_service import ProviderEmbeddingService

    adapter = _DimHonoringAdapter()
    provider = {
        "adapter": adapter,
        "client": object(),
        "vendor": "ollama",
        "name": "ollama:local",
        "capabilities": {
            "embedding_model": "qwen3-embedding:8b",
            "embedding_dim": dim,
        },
    }
    return ProviderEmbeddingService(provider), adapter


class _FakeLLM:
    """Minimal LLM service exposing the embedding-resolution surface."""

    def __init__(self, service, route: str = "ollama:local"):
        self._service = service
        self._route = route

    def get_embedding_route(self):
        return self._route

    def get_embedding_service(self):
        return self._service

    def set_embedding_route(self, route, *, persist: bool = True):
        self._route = route


class _FakeRequest:
    """Just enough Request for ``get_agent`` + ``request.json()``."""

    def __init__(self, agent, body):
        self.state = SimpleNamespace(agent=agent)
        self._body = body

    async def json(self):
        return self._body


def _agent(db, service, route: str = "ollama:local", agent_id: str = "a1"):
    return SimpleNamespace(
        llm_service=_FakeLLM(service, route=route),
        storage=SimpleNamespace(db=db),
        agent_id=agent_id,
    )


@pytest.fixture
async def seeded_db():
    with tempfile.TemporaryDirectory() as tmp:
        db = await AsyncDatabase.sqlite(str(Path(tmp) / "reindex_endpoint.db"))
        await _seed(db)
        yield db
        await db.close()


@pytest.fixture(autouse=True)
def _fixed_column_dim(monkeypatch):
    # Keep the resolved dim == column width so the dimension guard passes.
    from kestrel_sovereign import cli_embeddings

    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: DIM)
    # A fresh job registry per test.
    model_endpoints._REINDEX_JOBS.clear()


@pytest.mark.asyncio
async def test_reindex_dry_run_reports_counts_and_touches_nothing(seeded_db):
    svc = FakeEmbeddingService()
    request = _FakeRequest(_agent(seeded_db, svc), {"dry_run": True})

    result = await model_endpoints.reindex_embeddings(request)

    assert result["dry_run"] is True
    assert result["target_profile"] == TARGET
    # agent a1 scope: conv 2 stale + saved_items 2 + document_chunks 2 = 6.
    assert result["total_stale"] == 6
    assert result["stale_rows"]["conversation_history"] == 2
    assert result["stale_rows"]["saved_items"] == 2
    assert result["stale_rows"]["document_chunks"] == 2
    # Dry-run embeds nothing.
    assert svc.calls == 0
    remaining = EmbeddingReindexer(seeded_db, svc, TARGET, column_dim=DIM)
    assert sum((await remaining.count_all_stale(agent_id="a1")).values()) == 6


@pytest.mark.asyncio
async def test_reindex_execute_round_trip_reembeds_all(seeded_db):
    svc = FakeEmbeddingService()
    request = _FakeRequest(_agent(seeded_db, svc), {"dry_run": False})

    started = await model_endpoints.reindex_embeddings(request)
    assert started["dry_run"] is False
    assert started["total_stale"] == 6
    job_id = started["job_id"]
    assert started["status"] == "running"

    # Poll the progress endpoint until the background job finishes.
    job = None
    for _ in range(200):
        await asyncio.sleep(0.01)
        job = await model_endpoints.get_reindex_job(job_id, request)
        if job["status"] in ("done", "error"):
            break
    assert job is not None and job["status"] == "done", job
    assert job["total_reembedded"] == 6

    # Every stale row for a1 is now on the target profile.
    remaining = EmbeddingReindexer(seeded_db, svc, TARGET, column_dim=DIM)
    assert sum((await remaining.count_all_stale(agent_id="a1")).values()) == 0


@pytest.mark.asyncio
async def test_reindex_dead_service_is_error_not_false_success(seeded_db):
    # #2360: a service that scans every stale row but re-embeds none (empty
    # vectors, no exception) must NOT report success. The old endpoint returned
    # status "done" / error null with total_reembedded 0 — a false success that
    # hid 6 stranded rows. The job must now finish "error", surface the failed
    # counters, and leave the rows un-flipped.
    svc = DeadEmbeddingService()
    request = _FakeRequest(_agent(seeded_db, svc), {"dry_run": False})

    started = await model_endpoints.reindex_embeddings(request)
    assert started["total_stale"] == 6
    job_id = started["job_id"]

    job = None
    for _ in range(200):
        await asyncio.sleep(0.01)
        job = await model_endpoints.get_reindex_job(job_id, request)
        if job["status"] in ("done", "error", "partial"):
            break
    assert job is not None, job
    # The load-bearing assertion: scanned-but-zero-reembedded is an error.
    assert job["status"] == "error", job
    assert job["total_reembedded"] == 0
    assert job["error"], job
    # The previously-hidden counters are now visible.
    assert job["total_failed"] == 6, job
    assert sum(s.get("failed", 0) for s in job["stats"].values()) == 6

    # And the rows really were NOT re-embedded — still all stale.
    remaining = EmbeddingReindexer(seeded_db, svc, TARGET, column_dim=DIM)
    assert sum((await remaining.count_all_stale(agent_id="a1")).values()) == 6


@pytest.mark.asyncio
async def test_reindex_execute_empty_corpus_reports_done_inline(seeded_db):
    # First re-embed everything, then a second execute has nothing to do and
    # must report done inline (no job handle).
    svc = FakeEmbeddingService()
    reindexer = EmbeddingReindexer(seeded_db, svc, TARGET, column_dim=DIM)
    from kestrel_sovereign.storage.embedding_reindex import REINDEX_TABLES

    for table in REINDEX_TABLES:
        await reindexer.reindex_table(table, agent_id="a1")

    request = _FakeRequest(_agent(seeded_db, svc), {"dry_run": False})
    result = await model_endpoints.reindex_embeddings(request)
    assert result["dry_run"] is False
    assert result["status"] == "done"
    assert result["total_stale"] == 0
    assert "job_id" not in result


@pytest.mark.asyncio
async def test_reindex_pinned_dim_forwarded_rows_flip_profile(seeded_db):
    # #2371: the resolved service carries a pinned embedding_dim (768 in prod).
    # It MUST forward that as the Matryoshka ``dimensions`` param on every embed,
    # or the adapter returns native-dim vectors the column guard rejects — every
    # row skipped_dim_mismatch even though the pin was accepted. With the fix the
    # dim reaches the adapter, vectors match the column, and rows flip profile.
    service, adapter = _provider_service(dim=DIM)
    target = service.current_profile_id()
    assert target  # a real profile id resolves from the pinned model@dim
    request = _FakeRequest(_agent(seeded_db, service), {"dry_run": False})

    started = await model_endpoints.reindex_embeddings(request)
    # 6 seeded stale rows for a1 + the one row seeded under the placeholder
    # "targetprofile" id, which is stale relative to this real model@dim target.
    assert started["total_stale"] == 7
    job_id = started["job_id"]

    job = None
    for _ in range(200):
        await asyncio.sleep(0.01)
        job = await model_endpoints.get_reindex_job(job_id, request)
        if job["status"] in ("done", "error"):
            break
    assert job is not None and job["status"] == "done", job
    assert job["total_reembedded"] == 7, job
    # Nothing was rejected as a dim mismatch — the pin was honoured.
    assert sum(s.get("skipped_dim_mismatch", 0) for s in job["stats"].values()) == 0
    # The adapter actually received the pinned dimension on every call.
    assert adapter.seen_dimensions, "adapter was never called"
    assert all(d == DIM for d in adapter.seen_dimensions), adapter.seen_dimensions

    # Every stale row for a1 is now on the target profile.
    remaining = EmbeddingReindexer(seeded_db, service, target, column_dim=DIM)
    assert sum((await remaining.count_all_stale(agent_id="a1")).values()) == 0


@pytest.mark.asyncio
async def test_reindex_without_dim_forwarding_would_dim_mismatch(seeded_db):
    # Guardrail proving the fix is load-bearing: the same dimension-honoring
    # adapter, when NOT handed a pinned dim, returns native-width vectors that
    # the column guard rejects — reproducing the pre-#2371 all-skipped outcome.
    service, adapter = _provider_service(dim=DIM)
    service.embedding_dim = None  # simulate the pin never reaching the service
    target = "targetprofile"
    reindexer = EmbeddingReindexer(seeded_db, service, target, column_dim=DIM)
    stats = await reindexer.reindex_table("conversation_history", agent_id="a1")
    assert stats.reembedded == 0
    assert stats.skipped_dim_mismatch == stats.scanned - stats.skipped_empty
    assert all(d is None for d in adapter.seen_dimensions)


@pytest.mark.asyncio
async def test_reindex_refuses_when_route_is_none(seeded_db):
    svc = FakeEmbeddingService()
    request = _FakeRequest(_agent(seeded_db, svc, route="none"), {"dry_run": True})
    with pytest.raises(HTTPException) as exc:
        await model_endpoints.reindex_embeddings(request)
    assert exc.value.status_code == 409
    assert '"none"' in exc.value.detail


@pytest.mark.asyncio
async def test_reindex_refuses_on_dim_mismatch(seeded_db, monkeypatch):
    from kestrel_sovereign import cli_embeddings

    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: 768)
    svc = FakeEmbeddingService(dim=1536)  # resolves at 1536, column is 768
    request = _FakeRequest(_agent(seeded_db, svc), {"dry_run": True})
    with pytest.raises(HTTPException) as exc:
        await model_endpoints.reindex_embeddings(request)
    assert exc.value.status_code == 409
    assert "does not match the vector-column width" in exc.value.detail


@pytest.mark.asyncio
async def test_reindex_refuses_when_persisted_route_cannot_be_applied(seeded_db):
    # #2360 review finding P2: when a persisted embedding_route exists but can no
    # longer be applied (stale/removed route), the endpoint must REFUSE (409) —
    # not silently fall through and reindex production rows into whatever route
    # was already active (the wrong embedding profile). Mirrors the CLI, which
    # exits non-zero here.
    import json

    await seeded_db.execute_commit(
        "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
        ("a1", "embedding_route", json.dumps("gone:route")),
    )

    class _RaisingLLM(_FakeLLM):
        def set_embedding_route(self, route, *, persist: bool = True):
            raise ValueError(f"route {route!r} no longer resolves")

    svc = FakeEmbeddingService()
    agent = SimpleNamespace(
        llm_service=_RaisingLLM(svc, route="ollama:local"),
        storage=SimpleNamespace(db=seeded_db),
        agent_id="a1",
    )
    request = _FakeRequest(agent, {"dry_run": True})
    with pytest.raises(HTTPException) as exc:
        await model_endpoints.reindex_embeddings(request)
    assert exc.value.status_code == 409
    assert "no longer valid" in exc.value.detail
    assert "gone:route" in exc.value.detail


@pytest.mark.asyncio
async def test_reindex_rejects_unknown_table(seeded_db):
    svc = FakeEmbeddingService()
    request = _FakeRequest(
        _agent(seeded_db, svc), {"dry_run": True, "tables": ["not_a_table"]}
    )
    with pytest.raises(HTTPException) as exc:
        await model_endpoints.reindex_embeddings(request)
    assert exc.value.status_code == 400
    assert "unknown table" in exc.value.detail


@pytest.mark.asyncio
async def test_reindex_job_not_readable_by_other_agent(seeded_db):
    # A job started under agent a1 must not be pollable by agent a2 (#2336):
    # the opaque job_id alone must not grant cross-agent access. a2 gets 404,
    # not the job — and not even confirmation the job exists.
    svc = FakeEmbeddingService()
    a1_request = _FakeRequest(_agent(seeded_db, svc, agent_id="a1"), {"dry_run": False})
    started = await model_endpoints.reindex_embeddings(a1_request)
    job_id = started["job_id"]

    a2_request = _FakeRequest(_agent(seeded_db, svc, agent_id="a2"), {})
    with pytest.raises(HTTPException) as exc:
        await model_endpoints.get_reindex_job(job_id, a2_request)
    assert exc.value.status_code == 404

    # The owning agent can still read it (drain the job to completion).
    job = None
    for _ in range(200):
        await asyncio.sleep(0.01)
        job = await model_endpoints.get_reindex_job(job_id, a1_request)
        if job["status"] in ("done", "error"):
            break
    assert job is not None and job["status"] == "done", job
    # The owner-scoping key is never leaked in the public view.
    assert "owner_agent_id" not in job


@pytest.mark.asyncio
async def test_reindex_scopes_tables_to_request(seeded_db):
    # Restricting to document_chunks only re-embeds the 2 global chunks.
    svc = FakeEmbeddingService()
    request = _FakeRequest(
        _agent(seeded_db, svc),
        {"dry_run": True, "tables": ["document_chunks"]},
    )
    result = await model_endpoints.reindex_embeddings(request)
    assert result["total_stale"] == 2
    assert set(result["stale_rows"]) == {"document_chunks"}
