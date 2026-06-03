"""End-to-end smoke for saved_items + async_rag_store on real data (#1491).

Mirrors the conversation_history smoke that surfaced #1481 + #1486.
Pre-#1491 only unit tests covered the saved_items and RAG paths;
this file flexes the actual write + read flow against a real
embedding-capable provider (Ollama nomic-embed-text) so the same
class of "infrastructure correct, consumer broken" latent bugs
either surface or get ruled out.

Coverage:

- ``saved_items``
    1. Save four items with diverse content using
       ``SavedItemsStore.save_item``.
    2. Verify each row gets a non-NULL ``embedding_vec`` and a
       non-NULL ``embedding_profile_id`` stamp (#1477).
    3. Search via ``SavedItemsStore.search`` for semantic recall
       (not just substring match) and assert the right item ranks
       top-1.
    4. Inject a stale-profile row directly via SQL and verify it
       does NOT surface in the next profile-filtered search.

- ``document_chunks`` (RAG)
    1. ``AsyncRAGStore.chunk_document`` over a small multi-topic
       document.
    2. Verify chunks get ``embedding_vec`` + ``embedding_profile_id``.
    3. ``AsyncRAGStore.search_chunks`` with a semantic query and
       assert the right chunk ranks top.
    4. Same stale-profile isolation check.

- ``empty content`` edge case
    Both stores must tolerate ``aembed("")`` (ticket explicitly
    calls this out as a likely-bug class).

Skips when Ollama isn't reachable. Closes out the vector-lift
hardening epic.
"""

from __future__ import annotations

import os
import struct
import tempfile

import pytest
import pytest_asyncio

# Skip the whole file if ollama isn't installed; even with the
# package present, `skip_if_no_ollama` further guards on the
# daemon being reachable.
ollama = pytest.importorskip("ollama")

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore


pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def skip_if_no_ollama():
    """Skip if ollama daemon isn't reachable (same pattern as
    test_agent_tools_e2e.py).

    Also ensures the embedding model we depend on is present —
    otherwise pull would block for minutes during the test run
    (which CI hates) and the test would falsely appear to time out
    rather than skip cleanly.
    """
    try:
        client = ollama.Client()
        models = client.list()
    except Exception as exc:
        pytest.skip(f"Ollama not available: {exc}")
    names = []
    for entry in getattr(models, "models", []) or []:
        # Newer ollama client returns objects, older returns dicts.
        name = getattr(entry, "model", None) or (
            entry.get("model") if isinstance(entry, dict) else None
        )
        if name:
            names.append(name)
    if not any(
        n.startswith("nomic-embed-text") for n in names
    ):
        pytest.skip(
            "ollama nomic-embed-text not present (run "
            "`ollama pull nomic-embed-text` to enable this smoke)"
        )


@pytest_asyncio.fixture
async def db():
    """Temp SQLite DB with the full sovereign schema + #1477 migrations."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_ = await AsyncDatabase.sqlite(path)
    yield db_
    await db_.close()
    os.unlink(path)


class _OllamaEmbeddingService:
    """Minimal ``ProviderEmbeddingService``-shaped facade so the smoke
    doesn't have to spin up an entire :class:`LLMService` with
    discovery, mandate persistence, etc.

    Mimics the small public surface the storage layer reaches for:
    ``aembed(text)`` returning ``list[float] | None``,
    ``aembed_batch(texts)``, and the #1477
    ``describe()`` / ``current_profile_id()`` pair.

    Anything richer would re-test the LLMService factory rather
    than the smoke flow this file is responsible for.
    """

    model = "nomic-embed-text"
    embedding_dim = 768

    def __init__(self):
        self._client = ollama.AsyncClient()

    async def aembed(self, text):
        if not text:
            return None
        try:
            r = await self._client.embed(model=self.model, input=text)
        except Exception:
            return None
        embeddings = r.get("embeddings") or []
        if not embeddings:
            return None
        return list(embeddings[0])

    async def aembed_batch(self, texts):
        if not texts:
            return []
        try:
            r = await self._client.embed(model=self.model, input=list(texts))
        except Exception:
            return [None] * len(texts)
        return [list(e) for e in (r.get("embeddings") or [None] * len(texts))]

    def describe(self):
        from kestrel_sovereign.llm.embedding_service import (
            derive_embedding_profile,
        )
        return derive_embedding_profile(
            provider="ollama",
            model=self.model,
            dim=self.embedding_dim,
        )

    def current_profile_id(self):
        profile = self.describe()
        return profile.profile_id if profile else None


# -------------------------------------------------------------------- saved_items


async def test_saved_items_smoke_writes_stamp_and_recall(
    db, skip_if_no_ollama,
):
    """save_item → embedding_vec populated + embedding_profile_id stamped,
    then search returns the semantically-relevant item top-1."""
    embedding_service = _OllamaEmbeddingService()
    store = SavedItemsStore(db, agent_id="agent-smoke")
    # Wire the embedding service via the same hook saved_items uses.
    store._get_embedding_service = lambda: embedding_service

    items = [
        ("sailing-wanderer",
         "My favorite hobby is sailing. I own an old wooden sloop named Wanderer."),
        ("coffee-roasting",
         "I roast coffee at home — Ethiopian Yirgacheffe is my favorite single-origin."),
        ("rust-tokio",
         "The Tokio async runtime in Rust uses a work-stealing scheduler."),
        ("python-typing",
         "Python type hints from typing module help static analysis tools catch bugs."),
    ]
    for name, content in items:
        await store.save_item(item_type="stash", name=name, content=content)

    # Verify writes landed with embedding_vec + embedding_profile_id.
    rows = await db.fetchall(
        """SELECT name, embedding_vec IS NOT NULL,
                  embedding_profile_id
           FROM saved_items WHERE agent_id = 'agent-smoke'""",
        (),
    )
    assert len(rows) == 4, "all four items must persist"
    profile_ids = set()
    for name, has_vec, profile_id in rows:
        assert has_vec, f"saved_item {name!r} missing embedding_vec"
        assert profile_id, f"saved_item {name!r} missing embedding_profile_id"
        profile_ids.add(profile_id)
    assert len(profile_ids) == 1, (
        "all four items came from one embedding service — they must share "
        f"one profile id (got {profile_ids})"
    )

    # Recall: semantic query about boats / hobbies must surface the
    # sailing item, not the coffee/rust/python rows.
    expected_profile = embedding_service.current_profile_id()
    results = await store.search(
        query="What do you remember about my hobbies on the lake?",
        limit=4,
    )
    assert results, "search returned nothing — semantic recall is broken"
    top = results[0]
    top_name = top["item"]["name"]
    assert top_name == "sailing-wanderer", (
        f"semantic recall failure: expected 'sailing-wanderer' top-1, "
        f"got {top_name!r} with full ordering "
        f"{[r['item']['name'] for r in results]}"
    )


async def test_saved_items_smoke_profile_filter_isolates_stale_rows(
    db, skip_if_no_ollama,
):
    """A row stamped with a foreign profile id must NOT surface in any
    saved_items search path.

    Exercises BOTH the kNN path (cosine) AND the LIKE fallback path
    (``_text_search``) so a future regression on either path
    surfaces here. Codex P2 on the first version of this file
    flagged that identical content didn't exercise the keyword
    fallback — the foreign-content row in this version contains a
    unique keyword that LIKE would latch onto.
    """
    embedding_service = _OllamaEmbeddingService()
    store = SavedItemsStore(db, agent_id="agent-isolation")
    store._get_embedding_service = lambda: embedding_service

    # Real save under the active profile.
    await store.save_item(
        item_type="stash",
        name="sailing-real",
        content="Sailing on Lake Michigan is wonderful.",
    )

    # Foreign-profile row with a UNIQUE keyword the query references.
    await store.save_item(
        item_type="stash",
        name="pomegranate-foreign",
        content=(
            "Pomegranate seeds are tart. Pomegranates ripen in autumn."
        ),
    )
    await db.execute(
        "UPDATE saved_items SET embedding_profile_id = 'FFFFFFFFFFFF' "
        "WHERE name = 'pomegranate-foreign'",
        (),
    )
    await db.commit()

    # Two queries — one semantic, one keyword — to confirm BOTH paths
    # filter foreign-profile rows.
    semantic = await store.search(
        query="What do I love about Lake Michigan?",
        limit=10,
    )
    surfaced_names = {r["item"]["name"] for r in semantic}
    assert "sailing-real" in surfaced_names, "real row must surface"
    assert "pomegranate-foreign" not in surfaced_names, (
        "foreign row leaked through the kNN path"
    )

    keyword = await store.search(
        query="pomegranate",  # LIKE would surface the foreign row
        limit=10,
    )
    surfaced_names = {r["item"]["name"] for r in keyword}
    assert "pomegranate-foreign" not in surfaced_names, (
        "foreign row leaked through the LIKE fallback path — keyword "
        "isolation is broken"
    )


async def test_saved_items_empty_content_does_not_crash(
    db, skip_if_no_ollama,
):
    """``aembed("")`` / whitespace must not crash the save path.

    Codex P3 on the first version of this test caught that passing a
    non-empty ``summary`` made ``save_item`` embed the summary
    instead of the empty content, hiding any regression in the
    empty-content path. This version passes no summary so
    ``save_item`` actually calls ``aembed("   ")``.
    """
    # Spy on the embedding service to confirm what it actually saw —
    # so a future regression where the store stops calling aembed
    # at all wouldn't silently look like a pass.
    aembed_calls: list[str] = []
    embedding_service = _OllamaEmbeddingService()

    class _SpyService:
        def __init__(self, inner):
            self._inner = inner
            self.model = inner.model
            self.embedding_dim = inner.embedding_dim

        async def aembed(self, text):
            aembed_calls.append(text)
            return await self._inner.aembed(text)

        async def aembed_batch(self, texts):
            for t in texts:
                aembed_calls.append(t)
            return await self._inner.aembed_batch(texts)

        def describe(self):
            return self._inner.describe()

        def current_profile_id(self):
            return self._inner.current_profile_id()

    spy = _SpyService(embedding_service)

    store = SavedItemsStore(db, agent_id="agent-empty")
    store._get_embedding_service = lambda: spy

    item = await store.save_item(
        item_type="stash",
        name="empty-content",
        content="   ",  # whitespace-only, no summary
    )
    assert item is not None
    # Confirm the spy actually saw the empty-ish input — proves the
    # save path reached aembed, not that the store short-circuited
    # before getting there.
    assert aembed_calls, (
        "save_item never called aembed on whitespace content — "
        "store short-circuited before reaching the empty-content path"
    )
    assert aembed_calls[-1].strip() == "", (
        f"aembed should have seen whitespace content; got "
        f"{aembed_calls[-1]!r}"
    )
    row = await db.fetchall(
        "SELECT embedding_vec, embedding_profile_id FROM saved_items "
        "WHERE name = 'empty-content'",
        (),
    )
    assert row, "row should have persisted even with empty content"
    # Embedding may be NULL (aembed returned None for whitespace) or
    # populated (Ollama returned a vector for whitespace input) — both
    # are acceptable; the test is that save_item didn't crash and the
    # row landed cleanly.


# ------------------------------------------------------------------------ RAG


async def test_saved_items_null_profile_stays_searchable_by_keyword(
    db, skip_if_no_ollama,
):
    """Codex P2 round 3 regression: a row with NULL embedding_profile_id
    (text-only save, ``aembed`` returned None, embedding service down,
    pre-#1477) MUST still be findable via the LIKE fallback.

    The earlier strict ``embedding_profile_id = ?`` filter hid these
    legitimate rows. To actually exercise ``_text_search`` (the only
    path with that filter), this test seeds ONLY a NULL-profile row
    so the kNN backend returns empty and saved_items.search falls
    through to text_search.
    """
    embedding_service = _OllamaEmbeddingService()
    store = SavedItemsStore(db, agent_id="agent-null-profile")
    store._get_embedding_service = lambda: embedding_service

    # Only one row, with NULL profile id (simulates aembed returning
    # None or embedding service being unavailable at write time).
    await db.execute(
        """INSERT INTO saved_items (id, agent_id, item_type, name, content,
            embedding_profile_id, embedding_vec, created_at)
           VALUES (?, ?, ?, ?, ?, NULL, NULL, CURRENT_TIMESTAMP)""",
        ("text-only", "agent-null-profile", "stash", "text-only-note",
         "An unembedded note that mentions pomegranates.",),
    )
    await db.commit()

    # kNN returns [] (no embedded rows) → search() falls through to
    # _text_search. The new filter MUST keep the NULL row visible.
    results = await store.search(query="pomegranates", limit=10)
    surfaced = {r["item"]["name"] for r in results}
    assert "text-only-note" in surfaced, (
        "NULL-profile row dropped from keyword search — regression of "
        f"codex P2 round 3 fix. Got: {surfaced}"
    )


async def test_rag_null_profile_chunk_stays_searchable_by_keyword(
    db, skip_if_no_ollama,
):
    """Same regression on the RAG side: chunks with NULL profile id
    (``compute_embeddings=False`` or aembed failure) must still
    surface via the LIKE fallback.

    Seeds only NULL-profile chunks so both the embedding path
    (filters out NULL) and BM25 (degenerate on tiny single-class
    corpora — terms in every doc score zero from IDF) return
    empty, forcing ``_search_by_like`` to run. The new
    NULL-tolerant filter there is what we're verifying.
    """
    embedding_service = _OllamaEmbeddingService()
    store = AsyncRAGStore(db)
    store._get_embedding_service = lambda: embedding_service

    # Only NULL-profile chunks — no embedded rows exist for the
    # kNN path to score.
    await store.chunk_document(
        file_hash="text-only-doc",
        content="The text-only document mentions pomegranates and "
                "their tart seeds.",
        chunk_size=500,
        compute_embeddings=False,
    )

    null_rows = await db.fetchall(
        "SELECT chunk_id FROM document_chunks "
        "WHERE embedding_profile_id IS NULL",
        (),
    )
    assert null_rows, "chunk should have NULL profile id"

    # Force BM25 index rebuild.
    store._bm25_built = False

    results = await store.search_chunks(
        query="pomegranates", limit=10,
    )
    surfaced = {r["file_hash"] for r in results}
    assert "text-only-doc" in surfaced, (
        "NULL-profile chunk dropped from RAG search — regression "
        f"of codex P2 round 3 fix. Got: {surfaced}"
    )


async def test_rag_smoke_chunk_stamp_and_recall(db, skip_if_no_ollama):
    """chunk_document → chunks stamped + ranked by cosine in search_chunks."""
    embedding_service = _OllamaEmbeddingService()
    store = AsyncRAGStore(db)
    # Same hook the production read/write paths use.
    store._get_embedding_service = lambda: embedding_service

    document = (
        "Section A: Sailing on Lake Michigan is calm in early summer. "
        "The wind is steady from the southwest. Wooden sloops are "
        "particularly enjoyable in these conditions.\n\n"
        "Section B: Coffee roasting at home requires careful temperature "
        "control. Ethiopian Yirgacheffe peaks around first crack at "
        "395 degrees Fahrenheit.\n\n"
        "Section C: Tokio's runtime in Rust schedules tasks using a "
        "work-stealing scheduler across multiple worker threads."
    )

    n_chunks = await store.chunk_document(
        file_hash="smoke-doc-1", content=document, chunk_size=200,
    )
    assert n_chunks > 0

    rows = await db.fetchall(
        """SELECT chunk_id, embedding_vec IS NOT NULL,
                  embedding_profile_id
           FROM document_chunks WHERE file_hash = 'smoke-doc-1'""",
        (),
    )
    assert len(rows) == n_chunks
    for chunk_id, has_vec, profile_id in rows:
        assert has_vec, f"chunk {chunk_id} missing embedding_vec"
        assert profile_id, f"chunk {chunk_id} missing embedding_profile_id"

    # Semantic query → the sailing chunk should outrank the coffee /
    # tokio chunks. We don't pin top-1 exactly because the chunker
    # may split section A across two chunks; just assert "sailing"
    # appears in the top-2 and outranks the other sections.
    results = await store.search_chunks(
        query="What does the document say about boats and the lake?",
        limit=4,
    )
    assert results, "RAG search returned nothing"
    top_text = results[0]["content"].lower()
    assert ("sail" in top_text or "wooden" in top_text or "lake" in top_text), (
        f"semantic recall failure on RAG: top hit was "
        f"{results[0]['content']!r}"
    )


async def test_rag_smoke_profile_filter_isolates_stale_chunks(
    db, skip_if_no_ollama,
):
    """A chunk stamped with a foreign profile id must NOT surface in a
    profile-filtered RAG search.

    Uses a UNIQUE keyword in the foreign chunk so that BM25 would
    score it highly — exercises the hybrid embedding-AND-BM25 merge
    path, not just the embedding-only branch. Codex P2 on the first
    version caught that identical content made BM25 score zero,
    leaving the BM25 isolation path untested.
    """
    embedding_service = _OllamaEmbeddingService()
    store = AsyncRAGStore(db)
    store._get_embedding_service = lambda: embedding_service

    # Real chunk under the active profile, normal content.
    await store.chunk_document(
        file_hash="real-doc",
        content="Sailing on Lake Michigan in early summer is calm.",
        chunk_size=500,
    )
    # Foreign-profile chunk with a UNIQUE keyword the query will
    # latch onto via BM25.
    await store.chunk_document(
        file_hash="foreign-doc",
        content=(
            "Pomegranate seeds are tart and ruby-colored. "
            "Pomegranates ripen in autumn."
        ),
        chunk_size=500,
    )
    await db.execute(
        "UPDATE document_chunks SET embedding_profile_id = 'FFFFFFFFFFFF' "
        "WHERE file_hash = 'foreign-doc'",
        (),
    )
    await db.commit()

    # Query mentions "pomegranate" — BM25 would score the foreign
    # chunk top-1 on keyword alone. The profile filter at every
    # layer (embedding, BM25, LIKE) must keep it out.
    results = await store.search_chunks(
        query="What does the document say about pomegranates?",
        limit=10,
    )
    surfaced_hashes = {r["file_hash"] for r in results}
    assert "foreign-doc" not in surfaced_hashes, (
        "foreign-profile chunk leaked through the hybrid RAG search "
        f"(BM25 + embedding + RRF merge); surfaced docs: "
        f"{surfaced_hashes}. Semantic space isolation is broken on "
        "the keyword path."
    )
