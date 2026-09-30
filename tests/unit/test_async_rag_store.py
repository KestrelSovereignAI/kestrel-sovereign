"""AsyncRAGStore writes: chunk content at rest (#2677), chunks without a vector (#3415)."""

import logging

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore


@pytest.mark.asyncio
async def test_current_writer_persists_document_chunk_content_as_plaintext(tmp_path):
    """Pin the pre-encryption behavior that Child D must intentionally replace."""
    db_path = tmp_path / "rag.db"
    db = await AsyncDatabase.sqlite(str(db_path))
    agent_id = "did:test:plaintext-characterization"
    file_hash = "characterization-file"
    sentinel = "rag-chunk-plaintext-sentinel-2677"
    try:
        await db.execute(
            "INSERT INTO files (content_hash, original_name) VALUES (?, ?)",
            (file_hash, "characterization.txt"),
        )
        await db.execute(
            "INSERT INTO file_owners "
            "(content_hash, agent_id, original_name) VALUES (?, ?, ?)",
            (file_hash, agent_id, "characterization.txt"),
        )
        store = AsyncRAGStore(db, agent_id=agent_id)

        inserted = await store.chunk_document(
            file_hash=file_hash,
            content=sentinel,
            chunk_size=100,
            compute_embeddings=False,
        )
        raw_row = await db.fetchone(
            "SELECT content FROM document_chunks WHERE file_hash = ?",
            (file_hash,),
        )
        columns = await db.fetchall("PRAGMA table_info(document_chunks)")

        assert inserted == 1
        assert raw_row == (sentinel,)
        assert "content_ciphertext" not in {column[1] for column in columns}
    finally:
        await db.close()

    # Prove the current SQLite artifact itself contains the body after the
    # writer has closed, rather than only checking a hydrated return value.
    assert sentinel.encode("utf-8") in db_path.read_bytes()


class _Service:
    """An embedding service whose batch call returns or raises *outcome*."""

    def __init__(self, outcome):
        self.outcome = outcome

    async def aembed_batch(self, texts):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


_VECTOR = [0.5, 0.25]


@pytest.mark.parametrize(
    ("outcome", "stored_without", "reason"),
    [
        (None, 2, "no embedding service resolved"),
        (RuntimeError("provider down"), 2, "the embedding call failed: provider down"),
        ([None, None], 2, "the embedding model returned no vector"),
        ([_VECTOR, None], 1, "the embedding model returned no vector"),
        # zip() used to drop the second chunk without storing it at all.
        ([_VECTOR], 2, "the embedding model returned 1 vectors for 2 chunks"),
    ],
    ids=["no-service", "raises", "all-empty", "partial", "short-batch"],
)
async def test_chunk_document_warns_about_chunks_stored_without_a_vector(
    tmp_path, caplog, outcome, stored_without, reason
):
    # #3415: a constitution reanchor stored 47 chunks per agent with no
    # vector, and nothing above INFO said so.
    db = await AsyncDatabase.sqlite(str(tmp_path / "rag.db"))
    try:
        store = AsyncRAGStore(db)
        service = None if outcome is None else _Service(outcome)
        store._get_embedding_service = lambda: service

        with caplog.at_level(logging.WARNING, logger=AsyncRAGStore.__module__):
            # chunk_size 10 overlaps by 2, so 12 characters make two chunks.
            stored = await store.chunk_document("doc", "abcdefghijkl", chunk_size=10)

        rows = await db.fetchall(
            "SELECT content, embedding_vec FROM document_chunks ORDER BY chunk_id", ()
        )
    finally:
        await db.close()

    assert stored == 2
    assert [content for content, _ in rows] == ["abcdefghij", "ijkl"]
    assert sum(vec is None for _, vec in rows) == stored_without
    assert (
        f"Stored {stored_without} of 2 chunks of doc without an embedding "
        f"({reason})."
    ) in caplog.text
    assert "`kestrel embeddings reindex --yes`" in caplog.text


@pytest.mark.parametrize("compute_embeddings", [True, False])
async def test_chunk_document_is_quiet_when_nothing_was_left_unembedded(
    tmp_path, caplog, compute_embeddings
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "rag.db"))
    try:
        store = AsyncRAGStore(db)
        store._get_embedding_service = lambda: _Service([_VECTOR, _VECTOR])

        with caplog.at_level(logging.WARNING, logger=AsyncRAGStore.__module__):
            await store.chunk_document(
                "doc", "abcdefghijkl", chunk_size=10,
                compute_embeddings=compute_embeddings,
            )
    finally:
        await db.close()

    assert "without an embedding" not in caplog.text
