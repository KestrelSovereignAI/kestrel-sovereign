"""Tests for the optional embedding-write path on
:meth:`AsyncConversationStore.add_conversation`.

Embeddings are sourced from the active LLM provider's embedding
capability via :meth:`AsyncConversationStore._lazy_embedding_service`
(see #1471 + this PR). Absent service / aembed failure / empty
content all fall through to the legacy INSERT shape so the row
still persists and the retriever keeps working via keyword overlap.
"""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore,
    _format_pgvector_text,
    _serialize_embedding,
)


# ----------------------------------------------------------------- helpers


def _make_store(
    backend_type: str = "sqlite",
    embedding_service: object = None,
    agent_id: str = "agent-1",
) -> tuple[AsyncConversationStore, MagicMock]:
    """Build an ``AsyncConversationStore`` with a stub ``AsyncDatabase``.

    ``embedding_service`` here is the per-instance override applied via
    monkey-patching ``_lazy_embedding_service`` (the method the store
    consults inside ``_maybe_embed``). Passing ``None`` produces a
    store whose embedding lookup returns ``None`` — equivalent to a
    provider without embeddings.
    """
    db = MagicMock()
    db.backend_type = backend_type
    db.execute_commit = AsyncMock(return_value=1)
    db.fetchone = AsyncMock(return_value=None)  # no prior history → new session
    db.fetchall = AsyncMock(return_value=[])
    store = AsyncConversationStore(db=db, agent_id=agent_id)
    # Per-instance override of the provider-embedding lookup. Doing this
    # post-construction instead of via ``__init__`` keeps the production
    # constructor surface narrow (just ``llm_service``) and matches the
    # pattern saved_items + RAG already use.
    store._lazy_embedding_service = lambda: embedding_service  # type: ignore[assignment]
    return store, db


def _insert_call(db_mock: MagicMock) -> tuple[str, tuple]:
    """Return the (sql, params) of the actual ``INSERT INTO
    conversation_history (...) VALUES ...`` call. ``add_conversation``
    also does a ``SELECT metadata, created_at FROM conversation_history``
    on the implicit-session derivation path which we filter out.
    """
    for call in db_mock.execute_commit.call_args_list:
        sql = call.args[0]
        if "INSERT INTO conversation_history" in sql:
            return sql, call.args[1]
    raise AssertionError(
        f"no INSERT call in {db_mock.execute_commit.call_args_list!r}"
    )


# ----------------------------------------------------------------- serialize helpers


def test_serialize_embedding_round_trips_float32():
    """The PurePythonBackend's ``_unpack`` is the inverse. If
    ``_serialize_embedding`` ever drifted (endianness, dtype) the
    backend would silently read garbage."""
    vec = [0.1, -0.2, 0.3, 1e6]
    packed = _serialize_embedding(vec)
    assert len(packed) == len(vec) * 4
    unpacked = list(struct.unpack(f"<{len(vec)}f", packed))
    assert unpacked == pytest.approx(vec, rel=1e-5)


def test_format_pgvector_text_round_trips():
    """pgvector's text shape is ``[v1,v2,…]``. Asyncpg ships this as a
    string + ``::vector`` cast handles the rest."""
    out = _format_pgvector_text([0.1, -0.2, 0.3])
    assert out.startswith("[") and out.endswith("]")
    # Parseable as a comma-separated float list.
    parsed = [float(x) for x in out[1:-1].split(",")]
    assert parsed == pytest.approx([0.1, -0.2, 0.3], rel=1e-5)


# ----------------------------------------------------------------- legacy path (no service)


@pytest.mark.asyncio
async def test_add_conversation_without_service_uses_legacy_insert():
    """No embedding service on the active provider (e.g. Anthropic,
    which has no embedding API) — emit the pre-existing column list
    (no ``embedding_vec``) so a deployment without an embedding-capable
    provider keeps working unchanged."""
    store, db = _make_store(embedding_service=None)
    await store.add_conversation(role="assistant", content="hello world")

    sql, params = _insert_call(db)
    assert "embedding_vec" not in sql, (
        f"legacy path must omit embedding_vec column, got: {sql!r}"
    )
    # 5 bound positional params (agent_id, role, content, rendered_content, metadata).
    assert len(params) == 5


# ----------------------------------------------------------------- embed write path


@pytest.fixture
def small_embedding_dim(monkeypatch):
    """Override the conversation embedding dim to 4 so tests can use
    short, readable embeddings rather than synthesizing 768-dim
    vectors. The dim-mismatch validation in :meth:`_maybe_embed`
    reads the constant at call time, so patching the module attribute
    is enough."""
    monkeypatch.setattr(
        "kestrel_sovereign.storage.sqla.conversation_message."
        "CONVERSATION_MESSAGE_EMBEDDING_DIM",
        4,
    )


@pytest.mark.asyncio
async def test_add_conversation_with_service_writes_embedding_vec_sqlite(
    small_embedding_dim,
):
    """SQLite: the embedding goes in as packed float32 bytes against
    a single ``?`` placeholder (BLOB column)."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=embedding)

    store, db = _make_store(backend_type="sqlite", embedding_service=svc)
    await store.add_conversation(role="assistant", content="hello")

    svc.aembed.assert_awaited_once_with("hello")
    sql, params = _insert_call(db)
    assert "embedding_vec" in sql
    # No ::vector cast on SQLite.
    assert "::vector" not in sql
    # Bind value at the trailing slot is packed bytes.
    assert isinstance(params[-1], (bytes, bytearray))
    assert params[-1] == _serialize_embedding(embedding)


@pytest.mark.asyncio
async def test_add_conversation_with_service_writes_embedding_vec_postgres(
    small_embedding_dim,
):
    """Postgres: the embedding goes in as a pgvector text literal +
    ``?::vector`` cast (asyncpg + pgvector)."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=embedding)

    store, db = _make_store(backend_type="postgres", embedding_service=svc)
    await store.add_conversation(role="assistant", content="hello")

    sql, params = _insert_call(db)
    assert "embedding_vec" in sql
    assert "::vector" in sql
    # Bind value at the trailing slot is the bracketed text literal.
    bound = params[-1]
    assert isinstance(bound, str)
    assert bound.startswith("[") and bound.endswith("]")


# ----------------------------------------------------------------- failure / absence


@pytest.mark.asyncio
async def test_add_conversation_falls_back_when_aembed_returns_none():
    """Provider service down → ``aembed`` returns ``None``. The row
    MUST still be inserted via the legacy path so chat persistence
    isn't blocked by an embedding-service outage."""
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=None)

    store, db = _make_store(embedding_service=svc)
    await store.add_conversation(role="assistant", content="hello")

    sql, _ = _insert_call(db)
    assert "embedding_vec" not in sql


@pytest.mark.asyncio
async def test_add_conversation_falls_back_when_aembed_raises():
    """Network blip / unexpected exception → still persist the row.

    Embedding generation is a best-effort enrichment, not a write
    barrier."""
    svc = MagicMock()
    svc.aembed = AsyncMock(side_effect=RuntimeError("provider timeout"))

    store, db = _make_store(embedding_service=svc)
    await store.add_conversation(role="assistant", content="hello")

    sql, _ = _insert_call(db)
    assert "embedding_vec" not in sql


@pytest.mark.asyncio
async def test_add_conversation_skips_embedding_when_content_empty():
    """Empty content (rare — guardrails normally catch it earlier)
    would produce a zero-norm embedding that vector backends explicitly
    skip anyway. Don't call the embedding service at all."""
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=[0.0] * 4)

    store, db = _make_store(embedding_service=svc)
    await store.add_conversation(role="assistant", content="")

    svc.aembed.assert_not_awaited()
    sql, _ = _insert_call(db)
    assert "embedding_vec" not in sql


@pytest.mark.asyncio
async def test_add_conversation_skips_embedding_on_dim_mismatch(caplog):
    """Provider returns a different dim than the column was created
    with (e.g. migrated against Ollama-768, switched to OpenAI-1536
    after restart). The row still lands but ``embedding_vec`` is
    skipped — better to fall back to keyword overlap than persist
    bytes the retriever can't decode. (Codex P2 on PR-B.)
    """
    from kestrel_sovereign.storage.sqla.conversation_message import (
        CONVERSATION_MESSAGE_EMBEDDING_DIM,
    )
    mismatched = [0.1] * (CONVERSATION_MESSAGE_EMBEDDING_DIM + 16)
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=mismatched)

    store, db = _make_store(embedding_service=svc)
    import logging
    with caplog.at_level(logging.ERROR):
        await store.add_conversation(role="assistant", content="hi")

    sql, _ = _insert_call(db)
    assert "embedding_vec" not in sql
    assert any("dim mismatch" in r.message.lower() for r in caplog.records), (
        "expected a clear error log on dim mismatch"
    )


@pytest.mark.asyncio
async def test_add_conversation_dim_mismatch_logs_once(caplog):
    """A misconfigured agent will hit the mismatch on every turn.
    Log once per store instance so logs aren't flooded."""
    from kestrel_sovereign.storage.sqla.conversation_message import (
        CONVERSATION_MESSAGE_EMBEDDING_DIM,
    )
    mismatched = [0.1] * (CONVERSATION_MESSAGE_EMBEDDING_DIM + 16)
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=mismatched)

    store, db = _make_store(embedding_service=svc)
    import logging
    with caplog.at_level(logging.ERROR):
        await store.add_conversation(role="assistant", content="hi 1")
        await store.add_conversation(role="assistant", content="hi 2")
        await store.add_conversation(role="assistant", content="hi 3")

    dim_records = [r for r in caplog.records if "dim mismatch" in r.message.lower()]
    assert len(dim_records) == 1, (
        f"expected exactly one dim-mismatch log per store, got {len(dim_records)}"
    )


@pytest.mark.asyncio
async def test_add_conversation_falls_back_when_migration_not_run(
    small_embedding_dim,
):
    """A live deployment where Phase-2 migration hasn't completed yet
    (column missing) raises on the embedding INSERT. Catch it, log,
    and retry with the legacy column list so the row still lands.

    Simulated by making the FIRST ``execute_commit`` raise — the
    fallback re-executes the legacy SQL."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=embedding)

    store, db = _make_store(embedding_service=svc)

    call_count = {"n": 0}

    async def execute_commit(sql, params):
        call_count["n"] += 1
        if "embedding_vec" in sql:
            raise RuntimeError('column "embedding_vec" does not exist')
        return 1

    db.execute_commit = AsyncMock(side_effect=execute_commit)
    await store.add_conversation(role="assistant", content="hello")

    # First call tried the embedding-aware SQL; second call retried
    # with the legacy column list and succeeded.
    insert_calls = [
        c for c in db.execute_commit.call_args_list
        if "INSERT INTO conversation_history" in c.args[0]
    ]
    assert len(insert_calls) == 2
    assert "embedding_vec" in insert_calls[0].args[0]
    assert "embedding_vec" not in insert_calls[1].args[0]


# ----------------------------------------------------------------- provider lookup


@pytest.mark.asyncio
async def test_lazy_embedding_service_routes_through_active_provider(monkeypatch):
    """``_lazy_embedding_service`` must call
    ``get_provider_embedding_service`` with the store's ``llm_service``
    so the active chat provider's embedding capability is used. This
    is the contract that keeps saved_items + RAG +
    conversation_history all on the same provider stack (#1471)."""
    received: dict[str, object] = {}

    def fake_get_provider(llm_service):
        received["llm_service"] = llm_service
        return MagicMock()

    monkeypatch.setattr(
        "kestrel_sovereign.llm.embedding_service.get_provider_embedding_service",
        fake_get_provider,
    )
    sentinel = MagicMock(name="LLMService")
    db = MagicMock()
    db.backend_type = "sqlite"
    store = AsyncConversationStore(db=db, agent_id="a", llm_service=sentinel)
    out = store._lazy_embedding_service()
    assert received["llm_service"] is sentinel
    assert out is not None


def test_lazy_embedding_service_honours_opt_out(monkeypatch):
    """``KESTREL_DISABLE_CONVERSATION_EMBEDDINGS=true`` short-circuits
    the lookup entirely — useful for deployments where embeddings are
    expensive or the provider isn't reachable from chat workers."""
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")

    def should_not_be_called(*a, **kw):
        pytest.fail("get_provider_embedding_service must not be called")

    monkeypatch.setattr(
        "kestrel_sovereign.llm.embedding_service.get_provider_embedding_service",
        should_not_be_called,
    )
    db = MagicMock()
    db.backend_type = "sqlite"
    store = AsyncConversationStore(db=db, agent_id="a")
    assert store._lazy_embedding_service() is None


def test_lazy_embedding_service_swallows_provider_errors(monkeypatch):
    """A misconfigured provider must NOT crash chat writes. The store
    treats a raised exception from ``get_provider_embedding_service``
    as "no service" → legacy column-set path."""
    def boom(*a, **kw):
        raise RuntimeError("provider not configured")

    monkeypatch.setattr(
        "kestrel_sovereign.llm.embedding_service.get_provider_embedding_service",
        boom,
    )
    db = MagicMock()
    db.backend_type = "sqlite"
    store = AsyncConversationStore(db=db, agent_id="a")
    assert store._lazy_embedding_service() is None
