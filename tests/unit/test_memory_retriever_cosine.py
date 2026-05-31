"""Tests for the vector-cosine semantic-score path on
:class:`MemoryRetriever`.

PR-C swaps the keyword-overlap ``_score_semantic`` for true cosine
similarity when both the query and the row have embeddings. Keyword
fallback is preserved for rows without an embedding (legacy data,
pre-migration deployments, rows written while Ollama was down) so a
single ``retrieve()`` call can mix both paths cleanly.

Covers:

- ``_cosine_unit`` returns ``None`` for unusable inputs, otherwise a
  value in ``[0, 1]``. Identical vectors → 1.0, opposite → 0.0,
  orthogonal → 0.5.
- ``_embed_query`` returns ``None`` when the conversation store has
  no service / on empty query / on aembed failure.
- ``_score_semantic`` takes the cosine path when both embeddings are
  present, falls back to keyword overlap otherwise.
- ``retrieve`` plumbs the query embedding + row embeddings through
  end-to-end: it calls ``aembed`` once, batches the row-embedding
  fetch, and assigns higher scores to semantically similar rows that
  have no token overlap.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.memory_retriever import (
    MemoryRetriever,
    _cosine_unit,
)


# ----------------------------------------------------------------- _cosine_unit


def test_cosine_unit_identical_vectors_is_one():
    """Identical normalised vectors → cosine 1.0 → unit 1.0."""
    v = [1.0, 0.0, 0.0]
    assert _cosine_unit(v, v) == pytest.approx(1.0)


def test_cosine_unit_opposite_vectors_is_zero():
    """Opposite direction → cosine -1 → unit 0.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_unit(a, b) == pytest.approx(0.0)


def test_cosine_unit_orthogonal_vectors_is_half():
    """Perpendicular → cosine 0 → unit 0.5. (Neutral signal,
    same as the keyword-path's "no meaningful query words"
    return.)"""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_unit(a, b) == pytest.approx(0.5)


def test_cosine_unit_returns_none_on_length_mismatch():
    """Different-dim inputs → None so caller can fall back to keyword
    overlap instead of mistaking "no signal" for "neutral score." This
    is the only place a None return from cosine matters — silently
    coercing to 0.5 here would let a mis-shaped embedding pull a real
    match below a less-relevant keyword match."""
    assert _cosine_unit([1.0, 0.0], [1.0, 0.0, 0.0]) is None


def test_cosine_unit_returns_none_on_zero_norm():
    """A zero-vector has undefined direction; returning a number here
    would silently let pgvector / numpy NaN propagate into the final
    score."""
    assert _cosine_unit([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) is None
    assert _cosine_unit([1.0], []) is None


# ----------------------------------------------------------------- _embed_query


@pytest.mark.asyncio
async def test_embed_query_returns_none_without_service():
    """Conversation store with no embedding service → query stays
    in keyword-fallback mode for the whole call."""
    conv = MagicMock()
    conv.embedding_service = None
    retriever = MemoryRetriever(conv)
    assert await retriever._embed_query("hello") is None


@pytest.mark.asyncio
async def test_embed_query_returns_none_for_empty_query():
    """Empty query → don't even call aembed (zero-norm result)."""
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=[0.0] * 4)
    conv = MagicMock()
    conv.embedding_service = svc
    retriever = MemoryRetriever(conv)
    assert await retriever._embed_query("") is None
    assert await retriever._embed_query("   ") is None
    svc.aembed.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_query_returns_none_on_aembed_failure():
    """Ollama outage / model missing → fall back to keyword
    overlap for this call."""
    svc = MagicMock()
    svc.aembed = AsyncMock(side_effect=RuntimeError("ollama timeout"))
    conv = MagicMock()
    conv.embedding_service = svc
    retriever = MemoryRetriever(conv)
    assert await retriever._embed_query("hello") is None


@pytest.mark.asyncio
async def test_embed_query_returns_floats_on_success():
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    conv = MagicMock()
    conv.embedding_service = svc
    retriever = MemoryRetriever(conv)
    out = await retriever._embed_query("hello")
    assert out == [0.1, 0.2, 0.3]


# ----------------------------------------------------------------- _score_semantic dispatch


def test_score_semantic_uses_cosine_when_embeddings_present():
    """Embeddings on both sides → cosine path. Use vectors that have
    NO token overlap with the query string so the keyword path would
    score zero — proves we exercised the cosine branch."""
    conv = MagicMock()
    retriever = MemoryRetriever(conv)
    score = retriever._score_semantic(
        content="completely unrelated text",
        query="the original utterance",
        expanded_concepts=[],
        query_embedding=[1.0, 0.0, 0.0],
        content_embedding=[1.0, 0.0, 0.0],  # identical → cosine 1.0
    )
    # 1.0 cosine + 0 concept = 0.7 (70% weight on cosine).
    assert score == pytest.approx(0.7)


def test_score_semantic_falls_back_to_keyword_without_embeddings():
    """No embeddings → keyword overlap. Verifies the original
    behaviour is unchanged on rows without embeddings."""
    conv = MagicMock()
    retriever = MemoryRetriever(conv)
    score = retriever._score_semantic(
        content="hello there world friend",
        query="hello world",
        expanded_concepts=[],
        query_embedding=None,
        content_embedding=None,
    )
    # All 2 query words match → keyword_score=1.0, no concepts → 0.7.
    assert score == pytest.approx(0.7)


def test_score_semantic_falls_back_when_content_embedding_missing():
    """Row was written before Phase-2 migration / while Ollama was
    down — embedding is None. Keyword path still works."""
    conv = MagicMock()
    retriever = MemoryRetriever(conv)
    score = retriever._score_semantic(
        content="hello world",
        query="hello",
        expanded_concepts=[],
        query_embedding=[1.0, 0.0],
        content_embedding=None,
    )
    # 1/1 query word matches → 1.0 keyword, no concepts → 0.7.
    assert score == pytest.approx(0.7)


def test_score_semantic_falls_back_on_dim_mismatch():
    """A dim mismatch shouldn't crash the score — _cosine_unit returns
    None and we fall back to keyword overlap for THIS row. Other rows
    in the same retrieve() call (with correctly-dim embeddings) still
    take the cosine path."""
    conv = MagicMock()
    retriever = MemoryRetriever(conv)
    score = retriever._score_semantic(
        content="hello world",
        query="hello",
        expanded_concepts=[],
        query_embedding=[1.0, 0.0],
        content_embedding=[1.0, 0.0, 0.0],  # wrong dim
    )
    # Falls back: 1/1 query word matches → 0.7.
    assert score == pytest.approx(0.7)


# ----------------------------------------------------------------- retrieve end-to-end


@pytest.mark.asyncio
async def test_retrieve_calls_aembed_once_and_batches_row_load():
    """One Ollama round-trip per retrieve(), regardless of slice
    size. The row-embedding load is batched in a single
    get_message_embeddings call."""
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=[1.0, 0.0])
    conv = MagicMock()
    conv.embedding_service = svc
    conv.get_conversation_history = AsyncMock(return_value=[
        {"id": 1, "role": "assistant", "content": "alpha"},
        {"id": 2, "role": "assistant", "content": "beta"},
    ])
    conv.get_message_embeddings = AsyncMock(return_value={
        1: [1.0, 0.0],
        2: [0.0, 1.0],
    })

    retriever = MemoryRetriever(conv)
    await retriever.retrieve("question", agent_id="a")

    svc.aembed.assert_awaited_once_with("question")
    conv.get_message_embeddings.assert_awaited_once()
    # Single batched call with both ids.
    (called_ids,) = conv.get_message_embeddings.call_args.args
    assert set(called_ids) == {1, 2}


@pytest.mark.asyncio
async def test_retrieve_ranks_cosine_match_above_keyword_only_row():
    """Two rows: row A has an embedding identical to the query
    embedding but ZERO token overlap with the query. Row B has full
    token overlap but no embedding. Cosine path on A must beat
    keyword path on B so we know cosine is actually doing work."""
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=[1.0, 0.0, 0.0])
    conv = MagicMock()
    conv.embedding_service = svc
    conv.get_conversation_history = AsyncMock(return_value=[
        {"id": 1, "role": "assistant", "content": "totally unrelated text",
         "metadata": {}, "created_at": None},
        {"id": 2, "role": "assistant", "content": "fooquery barquery",
         "metadata": {}, "created_at": None},
    ])
    conv.get_message_embeddings = AsyncMock(return_value={
        1: [1.0, 0.0, 0.0],   # identical to query → cosine 1.0
    })
    conv.atomic_increment_metadata_counter = AsyncMock()
    retriever = MemoryRetriever(conv)
    retriever.linker = None

    results = await retriever.retrieve(
        "fooquery barquery", agent_id="a", limit=2, min_score=0.0,
    )
    assert results[0]["id"] == 1


@pytest.mark.asyncio
async def test_retrieve_skips_row_embedding_load_when_query_unembeddable():
    """Conversation store with no service → ``_embed_query`` returns
    None → no row-embedding fetch. Saves a SQL round-trip in the
    keyword-only deployment shape."""
    conv = MagicMock()
    conv.embedding_service = None
    conv.get_conversation_history = AsyncMock(return_value=[
        {"id": 1, "role": "assistant", "content": "hello",
         "metadata": {}, "created_at": None},
    ])
    conv.get_message_embeddings = AsyncMock(return_value={})
    conv.atomic_increment_metadata_counter = AsyncMock()
    retriever = MemoryRetriever(conv)
    retriever.linker = None

    await retriever.retrieve("hi", agent_id="a", min_score=0.0)
    conv.get_message_embeddings.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_message_embeddings_chunks_to_avoid_sqlite_var_limit():
    """SQLite's default ``SQLITE_MAX_VARIABLE_NUMBER`` is 999. The
    retriever passes up to 1000 message ids here — single-query would
    raise ``too many SQL variables`` and silently disable vector
    recall for long conversations. Verify the lookup chunks the IN
    clause into batches well under the limit. (Codex P2 on PR-C.)
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    db = MagicMock()
    db.backend_type = "sqlite"
    db.fetchall = AsyncMock(return_value=[])
    store = AsyncConversationStore(db=db, agent_id="a")
    ids = list(range(1000))
    await store.get_message_embeddings(ids)

    assert db.fetchall.await_count >= 2, (
        "expected the IN clause to be chunked into multiple queries"
    )
    for call in db.fetchall.call_args_list:
        params = call.args[1]
        assert len(params) <= 999, (
            f"chunk too large for SQLite var limit: {len(params)} binds"
        )


@pytest.mark.asyncio
async def test_get_message_embeddings_parses_pgvector_text_shape():
    """asyncpg without a registered pgvector codec returns
    ``embedding_vec`` as a string like ``'[0.1,0.2,…]'``. The
    original loop iterated the string character-by-character and
    raised on the leading ``[``, silently dropping every PG row's
    embedding. Verify the text-shape parser handles this. (Codex P2
    on PR-C caught this.)
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    db = MagicMock()
    db.backend_type = "postgres"
    db.fetchall = AsyncMock(
        return_value=[(1, "[0.1,0.2,0.3]"), (2, "[]"), (3, None)]
    )
    store = AsyncConversationStore(db=db, agent_id="a", llm_service=None)
    out = await store.get_message_embeddings([1, 2, 3])
    assert out == {1: pytest.approx([0.1, 0.2, 0.3])}


@pytest.mark.asyncio
async def test_get_message_embeddings_skips_unexpected_pgvector_string_shape():
    """Anything that doesn't start with ``[`` and end with ``]`` is
    garbage — skip rather than parse with an exception that breaks
    the rest of the batch."""
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    db = MagicMock()
    db.backend_type = "postgres"
    db.fetchall = AsyncMock(
        return_value=[
            (1, "garbage without brackets"),
            (2, "[1.0,2.0]"),  # Good row mixed with bad one.
        ]
    )
    store = AsyncConversationStore(db=db, agent_id="a", llm_service=None)
    out = await store.get_message_embeddings([1, 2])
    assert 1 not in out
    assert out[2] == pytest.approx([1.0, 2.0])


@pytest.mark.asyncio
async def test_retrieve_falls_back_when_row_load_fails():
    """``get_message_embeddings`` raises → keyword overlap for the
    whole call. Row would otherwise be dropped silently."""
    svc = MagicMock()
    svc.aembed = AsyncMock(return_value=[1.0, 0.0])
    conv = MagicMock()
    conv.embedding_service = svc
    # Content distinct from query so the near-duplicate dedup
    # (``content.strip().lower() == query_normalized``) doesn't
    # silently skip the row.
    conv.get_conversation_history = AsyncMock(return_value=[
        {"id": 1, "role": "assistant", "content": "hello world there",
         "metadata": {}, "created_at": None},
    ])
    conv.get_message_embeddings = AsyncMock(
        side_effect=RuntimeError("column missing"),
    )
    conv.atomic_increment_metadata_counter = AsyncMock()
    retriever = MemoryRetriever(conv)
    retriever.linker = None

    results = await retriever.retrieve(
        "hello world", agent_id="a", min_score=0.0,
    )
    # Falls back to keyword overlap → row scores 0.7 on semantic
    # × weights ≥ min_score.
    assert len(results) == 1
    assert results[0]["id"] == 1
