"""Characterization seams for RAG chunk content encryption issue #2677."""

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore


@pytest.mark.asyncio
async def test_current_writer_persists_document_chunk_content_as_plaintext(tmp_path):
    """Pin the pre-encryption behavior that Child D must intentionally replace."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "rag.db"))
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
