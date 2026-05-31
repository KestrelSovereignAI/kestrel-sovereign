"""Tests for the SQLA-backed search path in :mod:`saved_items_store`
and the supporting :mod:`kestrel_sovereign.storage.sqla` module.

Covers:

- ``make_session_factory`` shape: SQLite-on-disk works, ``:memory:``
  refuses, PG-without-DSN refuses, PG DSN rewrites scheme.
- ``SAVED_ITEM_SPEC`` is wired to the right columns + dimension.
- ``SavedItemsStore.search`` routes through the vector backend when a
  session factory is available, and degrades to the legacy in-Python
  path when ``make_session_factory`` fails.
- End-to-end SQLite kNN: real ``saved_items`` rows with embeddings,
  semantic search returns the right ordering.

The end-to-end test exercises the full stack (SQLA session →
``PurePythonBackend.knn`` → in-Python cosine → row materialization via
``AsyncDatabase``) — the closest thing to an integration test we can
run in unit-test time on SQLite.
"""

from __future__ import annotations

import os
import struct
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
from kestrel_sovereign.storage.sqla import (
    SavedItem,
    SovereignSqlaSessionFactory,
    build_saved_item_spec,
    make_session_factory,
)


# ----------------------------------------------------------------- spec wiring


def test_build_saved_item_spec_matches_entity_columns():
    """``build_saved_item_spec`` must reference real ``SavedItem``
    columns — if someone renames a column on the entity without
    updating the builder, every search call would 500 at query-build
    time. This is the cheapest possible check.
    """
    spec = build_saved_item_spec(dimension=768)
    assert spec.entity is SavedItem
    assert spec.id_column is SavedItem.id
    assert spec.embedding_column is SavedItem.embedding
    assert spec.required_filter_keys == ("agent_id",)
    assert set(spec.filter_columns) == {"agent_id", "item_type"}
    assert spec.tenant_id_filter_key is None
    assert spec.dimension == 768


def test_build_saved_item_spec_accepts_any_positive_dimension():
    """The spec dimension is dynamic so any embedding model works.
    Default Ollama nomic-embed-text → 768, mxbai-embed-large → 1024,
    OpenAI ada-002 → 1536. All valid.
    (Caught by codex review: a fixed 1536 silently broke every
    default-model search.)
    """
    for d in (384, 768, 1024, 1536, 3072):
        spec = build_saved_item_spec(dimension=d)
        assert spec.dimension == d


def test_build_saved_item_spec_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="dimension"):
        build_saved_item_spec(dimension=0)
    with pytest.raises(ValueError, match="dimension"):
        build_saved_item_spec(dimension=-1)


# ----------------------------------------------------------------- make_session_factory


def _fake_db(backend_type: str, **backend_attrs):
    """Build a duck-typed ``AsyncDatabase`` enough for
    ``make_session_factory`` to introspect."""
    return SimpleNamespace(
        backend_type=backend_type,
        backend=SimpleNamespace(**backend_attrs),
    )


def test_make_session_factory_rejects_memory_sqlite():
    """``:memory:`` SQLite DBs are per-connection — a fresh SQLAlchemy
    engine wouldn't see the AsyncDatabase's data. Refuse rather than
    silently miss rows."""
    db = _fake_db("sqlite", db_path=":memory:")
    with pytest.raises(ValueError, match=":memory:"):
        make_session_factory(db)


def test_make_session_factory_returns_factory_for_sqlite_file():
    """Real on-disk SQLite path → factory with the right shape."""
    with tempfile.NamedTemporaryFile(suffix=".db") as fp:
        db = _fake_db("sqlite", db_path=fp.name)
        sf = make_session_factory(db)
    assert isinstance(sf, SovereignSqlaSessionFactory)
    assert sf.engine is not None
    assert sf.engine.dialect.name == "sqlite"


def test_make_session_factory_rejects_pg_from_pool_no_dsn():
    """``AsyncDatabase.from_pool`` strands ``_dsn`` as None — we can't
    recover the URL needed to build a SQLAlchemy engine."""
    db = _fake_db("postgres", _dsn=None)
    with pytest.raises(NotImplementedError, match="from_pool"):
        make_session_factory(db)


def test_make_session_factory_rewrites_postgres_scheme():
    """``postgres://`` and ``postgresql://`` DSNs get rewritten to
    ``postgresql+asyncpg://`` so SQLAlchemy picks the async dialect.
    We don't actually connect — just check the URL the engine carries.
    """
    db = _fake_db("postgres", _dsn="postgresql://user:pw@localhost/db")
    sf = make_session_factory(db)
    assert "postgresql+asyncpg://" in str(sf.engine.url)


def test_make_session_factory_unknown_backend_type():
    db = _fake_db("mysql")
    with pytest.raises(ValueError, match="backend_type"):
        make_session_factory(db)


def test_make_session_factory_caches_per_db():
    """Regression: per-request ``SavedItemsStore`` instances should NOT
    each get a fresh SQLAlchemy engine. ``make_session_factory`` caches
    the factory on the ``AsyncDatabase`` so a thousand calls return
    the same engine (and the same connection pool). Without this,
    Postgres slot limits would be exhausted under normal API load.
    (Caught by codex review on the saved_items SQLA PR.)
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as fp:
        db = _fake_db("sqlite", db_path=fp.name)
        sf1 = make_session_factory(db)
        sf2 = make_session_factory(db)
    assert sf1 is sf2, "factory must be cached per AsyncDatabase, not rebuilt"


@pytest.mark.asyncio
async def test_async_database_close_disposes_cached_factory():
    """``AsyncDatabase.close()`` must dispose the cached SQLA factory so
    the engine + pool are released on app shutdown.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "close_test.db")
        db = await AsyncDatabase.sqlite(db_path)
        sf = make_session_factory(db)
        assert getattr(db, "_sovereign_sqla_factory", None) is sf
        await db.close()
        # After close, the cached attribute is cleared (the engine is
        # disposed inside close()).
        assert getattr(db, "_sovereign_sqla_factory", None) is None


# ----------------------------------------------------------------- search-path fallbacks


@pytest.mark.asyncio
async def test_search_with_no_embedding_service_falls_back_to_text():
    """No embedding service → straight to text-LIKE. Vector backend
    is never invoked."""
    mock_db = MagicMock()
    mock_db.fetchall = AsyncMock(return_value=[])
    store = SavedItemsStore(mock_db, agent_id="agent-1")
    store._get_embedding_service = lambda: None

    result = await store.search("hello")
    assert result == []
    # Reached the text-search SELECT, not the embedding-using SELECT.
    text_calls = [c for c in mock_db.fetchall.call_args_list if "LIKE" in c.args[0]]
    assert text_calls, "expected text-search fallback to issue a LIKE query"


@pytest.mark.asyncio
async def test_search_caches_session_factory_unavailable_signal():
    """When ``make_session_factory`` fails we cache the failure so
    repeated searches don't keep retrying it.
    """
    mock_db = MagicMock()
    mock_db.backend_type = "unknown-backend"  # make_session_factory raises
    mock_db.fetchall = AsyncMock(return_value=[])

    store = SavedItemsStore(mock_db, agent_id="agent-1")

    # First call — triggers the failure once
    sf1 = store._get_vector_session_factory()
    assert sf1 is None
    assert store._sqla_factory_unavailable is True

    # Second call — must short-circuit without re-attempting the
    # construction (which would log again).
    sf2 = store._get_vector_session_factory()
    assert sf2 is None


@pytest.mark.asyncio
async def test_search_via_vector_backend_returns_none_on_failure():
    """An unexpected error inside the vector backend path must NOT
    propagate — the caller falls back to the legacy in-Python search."""
    mock_db = MagicMock()
    store = SavedItemsStore(mock_db, agent_id="agent-1")

    # Patch the backend constructor to blow up. The wrapper should
    # swallow the exception and return None, letting search() fall
    # back to ``_legacy_in_python_search``.
    import kestrel_sovereign.storage.vector as sov_vector
    real_backend = sov_vector.PurePythonBackend

    class _Boom(real_backend):
        def __init__(self, *a, **k):
            raise RuntimeError("simulated backend failure")

    sov_vector.PurePythonBackend = _Boom
    try:
        result = await store._search_via_vector_backend(
            session_factory=MagicMock(),
            query="original query",
            query_embedding=[1.0] + [0.0] * 1535,
            item_type=None,
            limit=5,
        )
    finally:
        sov_vector.PurePythonBackend = real_backend
    assert result is None


@pytest.mark.asyncio
async def test_search_preserves_query_when_falling_back_to_text():
    """Regression: when the vector path runs but returns no top-k
    (e.g. agent has no embedded items yet), the text-search fallback
    must run with the ORIGINAL query string, not an empty string.
    A blank query would make ``_text_search`` do ``LIKE '%%'`` and
    return the newest items regardless of the user's terms.
    (Caught by codex review on the saved_items SQLA PR.)
    """
    mock_db = MagicMock()
    mock_db.backend_type = "unknown-backend"  # force legacy fallback path
    mock_db.fetchall = AsyncMock(return_value=[])

    store = SavedItemsStore(mock_db, agent_id="agent-1")
    service = SimpleNamespace(
        aembed=AsyncMock(return_value=[1.0] + [0.0] * 1535)
    )
    store._get_embedding_service = lambda: service

    await store.search("special-term", item_type=None, limit=5)

    # ``_legacy_in_python_search`` ran (no embedded rows) → fell back
    # to ``_text_search``. The text-search SQL must carry the original
    # query in its LIKE bind parameter.
    text_calls = [
        c for c in mock_db.fetchall.call_args_list
        if "LIKE" in c.args[0]
    ]
    assert text_calls, "expected text-search fallback"
    bind_args = text_calls[0].args[1]
    assert any("%special-term%" in str(x).lower() for x in bind_args), (
        f"text-search bind args should carry the query, got {bind_args!r}"
    )


# ----------------------------------------------------------------- end-to-end SQLite


@pytest.mark.asyncio
async def test_search_end_to_end_against_real_sqlite():
    """Insert items with embeddings via the legacy path, then search
    via the new sovereign vector backend through a real
    ``make_session_factory`` against on-disk SQLite. Returns the most
    similar item first.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "saved_items_test.db")
        db = await AsyncDatabase.sqlite(db_path)

        try:
            store = SavedItemsStore(db, agent_id="agent-e2e")

            # Two items with engineered embeddings. ada-002 is 1536-dim;
            # we build canonical basis-like vectors so cosine is
            # unambiguous.
            target = [1.0] + [0.0] * 1535
            distractor = [0.0, 1.0] + [0.0] * 1534

            # Phase 2: rows are written to BOTH the legacy ``embedding``
            # column (BYTEA / BLOB used by raw IO) AND ``embedding_vec``
            # (the ORM-mapped column the vector backend reads). In
            # production ``save_item()``'s dual-write keeps these in
            # sync; the test does it manually here to keep the search
            # path isolated from the embedding-service path.
            for row_id, name, vec in [
                ("id-target", "Target", target),
                ("id-distractor", "Distractor", distractor),
            ]:
                packed = struct.pack("<1536f", *vec)
                await db.execute(
                    """INSERT INTO saved_items
                       (id, agent_id, item_type, name, summary, content,
                        content_hash, ipfs_cid, embedding, source_type,
                        source_ref, schema_id, tags, metadata,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row_id, "agent-e2e", "stash", name, None,
                        "content x", row_id, None, packed,
                        None, None, None, "[]", "{}",
                        "2026-01-01T00:00:00", "2026-01-01T00:00:00",
                    ),
                )
                await db.execute(
                    "UPDATE saved_items SET embedding_vec = ? WHERE id = ?",
                    (packed, row_id),
                )
            await db.commit()

            # Stub the embedding service so search() picks the kNN
            # path. Returning ``target`` as the query embedding makes
            # id-target the highest-similarity result.
            class _StubEmbed:
                async def aembed(self, _q):
                    return target

            store._get_embedding_service = lambda: _StubEmbed()

            results = await store.search("anything")
            assert len(results) == 2
            assert results[0]["item"]["id"] == "id-target"
            assert results[0]["score"] > results[1]["score"]
        finally:
            await db.close()
