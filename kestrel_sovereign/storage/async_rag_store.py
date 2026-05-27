"""
Async RAG Store for Kestrel Storage.

Provides async document chunking and retrieval for RAG operations.
Uses hybrid search: Ollama embeddings (primary) + BM25 keyword search (fallback/boost).

WHEN TO USE THIS vs MemoryRetriever
-----------------------------------
Use AsyncRAGStore (this module) when:
  - Searching INDEXED DOCUMENTS (uploaded files, ingested knowledge bases)
  - You need vector similarity search over chunks of static content
  - Content is referential/factual (not conversational)
  - Examples: "find sections of the user guide about X", "search uploaded PDFs"

Use MemoryRetriever (storage/memory_retriever.py) when:
  - Searching CONVERSATION HISTORY and message-level memories
  - You want emotional weighting, importance, and decay applied
  - Content is experiential/episodic (what was said, felt, remembered)
  - Examples: "recall what we discussed about X", "find emotionally important moments"

The two systems intentionally do NOT share an interface — they answer
different questions about different data. RAG = "what does the document say?"
Memory = "what did we experience together?"

See docs/architecture/MEMORY_SYSTEM.md for the full decision matrix.
"""
import logging
import struct
from typing import List, Dict, Any, Optional, Tuple

from .async_database import AsyncDatabase
from .bm25_index import AsyncBM25Index, BM25_AVAILABLE

logger = logging.getLogger(__name__)

# Lazy load embedding service
_embedding_service = None


def _get_embedding_service():
    """Lazy load the Ollama embedding service."""
    global _embedding_service
    if _embedding_service is None:
        try:
            from kestrel_sovereign.llm.embedding_service import get_embedding_service
            _embedding_service = get_embedding_service()
            logger.info("Ollama embedding service initialized")
        except Exception as e:
            logger.warning(f"Embedding service not available: {e}")
            _embedding_service = False  # Mark as unavailable
    return _embedding_service if _embedding_service else None


def _serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding to bytes for SQLite storage."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def _deserialize_embedding(data: bytes) -> List[float]:
    """Deserialize embedding from bytes."""
    count = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f'{count}f', data))


class AsyncRAGStore:
    """
    Async RAG (Retrieval-Augmented Generation) storage.

    Uses hybrid search combining:
    - Ollama embeddings for semantic similarity (primary)
    - BM25 for keyword matching (fallback/boost)

    Results are merged using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self._bm25_index: Optional[AsyncBM25Index] = None
        self._bm25_built = False
    
    async def chunk_document(
        self,
        file_hash: str,
        content: str,
        chunk_size: int = 500,
        compute_embeddings: bool = True
    ) -> int:
        """
        Chunk a document and store the chunks with optional embeddings.

        Args:
            file_hash: Unique identifier for the document
            content: Document content to chunk
            chunk_size: Target chunk size in characters
            compute_embeddings: Whether to compute and store embeddings

        Returns:
            Number of chunks created
        """
        # Simple chunking by character count with overlap
        chunks = []
        overlap = chunk_size // 5  # 20% overlap
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            if chunk.strip():  # Skip empty chunks
                chunks.append(chunk)
            start = end - overlap

        if not chunks:
            return 0

        # Get embeddings if requested and service available
        embeddings = [None] * len(chunks)
        if compute_embeddings:
            embedding_service = _get_embedding_service()
            if embedding_service:
                try:
                    embeddings = await embedding_service.aembed_batch(chunks)
                    logger.debug(f"Computed embeddings for {len(chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Failed to compute embeddings: {e}")

        # Store chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            embedding_blob = None
            if embedding:
                embedding_blob = _serialize_embedding(embedding)

            await self.db.execute(
                "INSERT INTO document_chunks (file_hash, content, embedding) VALUES (?, ?, ?)",
                (file_hash, chunk, embedding_blob)
            )

        await self.db.commit()
        self._bm25_built = False  # Invalidate BM25 index

        return len(chunks)
    
    async def search_chunks(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using hybrid search.

        Uses embedding similarity (primary) + BM25 keyword matching (boost/fallback).
        Results are merged using Reciprocal Rank Fusion (RRF).

        Args:
            query: Search query
            limit: Maximum results to return
            min_score: Cosine-similarity floor (#1404) applied to the
                embedding-search source. Candidates with similarity
                below this value are dropped before RRF so weak
                semantic matches don't survive the merge. BM25
                (keyword) candidates are NOT thresholded — they have
                no comparable similarity scalar; keyword-only matches
                still surface for queries the embedding model misses.
                Default ``0.0`` preserves legacy behavior (no floor).

        Returns:
            List of matching chunks with scores
        """
        if not query.strip():
            return []

        # Get candidates from both methods. Embedding source is gated
        # by min_score; BM25 stays unfiltered (no per-candidate
        # similarity scalar).
        embedding_results = await self._search_by_embedding(
            query, limit=limit * 2, min_score=min_score,
        )
        bm25_results = await self._search_by_bm25(query, limit=limit * 2)

        # If both methods fail, fall back to LIKE search
        if not embedding_results and not bm25_results:
            return await self._search_by_like(query, limit)

        # Merge results using RRF
        merged = self._merge_rrf(embedding_results, bm25_results, limit)

        return merged

    async def _search_by_embedding(
        self, query: str, limit: int = 10, min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search using embedding similarity.

        ``min_score`` (#1404) filters candidates by cosine similarity
        before sort/limit so weak semantic matches don't survive into
        the RRF merge upstream.
        """
        embedding_service = _get_embedding_service()
        if not embedding_service:
            return []

        try:
            from kestrel_sovereign.llm.embedding_service import cosine_similarity

            # Get query embedding
            query_embedding = await embedding_service.aembed(query)
            if not query_embedding:
                return []

            # Get all chunks with embeddings
            rows = await self.db.fetchall(
                "SELECT chunk_id, file_hash, content, embedding FROM document_chunks "
                "WHERE embedding IS NOT NULL"
            )

            if not rows:
                return []

            # Score by cosine similarity, dropping candidates below
            # the relevance floor (#1404). Floor applies pre-sort so
            # the limit cap is filled with above-floor candidates only.
            scored = []
            for row in rows:
                chunk_id, file_hash, content, embedding_blob = row
                if embedding_blob:
                    chunk_embedding = _deserialize_embedding(embedding_blob)
                    score = cosine_similarity(query_embedding, chunk_embedding)
                    if score < min_score:
                        continue
                    scored.append({
                        'chunk_id': chunk_id,
                        'file_hash': file_hash,
                        'content': content,
                        'score': score,
                        'source': 'embedding'
                    })

            # Sort by score descending
            scored.sort(key=lambda x: x['score'], reverse=True)
            return scored[:limit]

        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []

    async def _search_by_bm25(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25 keyword matching."""
        if not BM25_AVAILABLE:
            return []

        try:
            # Build index if needed
            if not self._bm25_built:
                await self._build_bm25_index()

            if not self._bm25_index:
                return []

            # Search
            results = await self._bm25_index.asearch(query, limit)

            return [
                {
                    'chunk_id': int(r.doc_id),
                    'file_hash': r.metadata.get('file_hash', ''),
                    'content': r.content,
                    'score': r.score,
                    'source': 'bm25'
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    async def _build_bm25_index(self):
        """Build BM25 index from all chunks."""
        if not BM25_AVAILABLE:
            return

        rows = await self.db.fetchall(
            "SELECT chunk_id, file_hash, content FROM document_chunks"
        )

        if not rows:
            return

        self._bm25_index = AsyncBM25Index()
        for chunk_id, file_hash, content in rows:
            self._bm25_index.add_document(
                doc_id=str(chunk_id),
                content=content,
                metadata={'file_hash': file_hash}
            )

        await self._bm25_index.abuild()
        self._bm25_built = True
        logger.debug(f"Built BM25 index with {len(rows)} documents")

    async def _search_by_like(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback LIKE-based search."""
        words = query.lower().split()
        if not words:
            return []

        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in words])
        params = tuple(f"%{word}%" for word in words)

        rows = await self.db.fetchall(
            f"SELECT chunk_id, file_hash, content FROM document_chunks "
            f"WHERE {conditions} LIMIT ?",
            params + (limit,)
        )

        return [
            {
                'chunk_id': row[0],
                'file_hash': row[1],
                'content': row[2],
                'score': 1.0,  # No real score for LIKE
                'source': 'like'
            }
            for row in rows
        ]

    def _merge_rrf(
        self,
        embedding_results: List[Dict],
        bm25_results: List[Dict],
        limit: int,
        k: int = 60  # RRF constant
    ) -> List[Dict[str, Any]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each result list containing the doc.
        """
        scores: Dict[int, float] = {}
        docs: Dict[int, Dict] = {}

        # Score embedding results
        for rank, doc in enumerate(embedding_results):
            chunk_id = doc['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            docs[chunk_id] = doc

        # Score BM25 results
        for rank, doc in enumerate(bm25_results):
            chunk_id = doc['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            if chunk_id not in docs:
                docs[chunk_id] = doc

        # Sort by RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build result list
        results = []
        for chunk_id, rrf_score in ranked[:limit]:
            doc = docs[chunk_id]
            results.append({
                'file_hash': doc['file_hash'],
                'content': doc['content'],
                'score': rrf_score,
                'chunk_id': chunk_id,
            })

        return results
    
    async def get_chunks_for_file(self, file_hash: str) -> List[str]:
        """Get all chunks for a specific file."""
        rows = await self.db.fetchall(
            "SELECT content FROM document_chunks WHERE file_hash = ? ORDER BY chunk_id",
            (file_hash,)
        )
        return [row[0] for row in rows]
    
    async def delete_chunks_for_file(self, file_hash: str) -> None:
        """Delete all chunks for a file."""
        await self.db.execute_commit(
            "DELETE FROM document_chunks WHERE file_hash = ?",
            (file_hash,)
        )
    
    async def search_case_law(self, query: str, failures: List[Dict],
                              top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches for relevant past audit failures ("case law") using semantic search.

        Args:
            query: The search query
            failures: List of audit failure records from conversation store
            top_k: Number of top results to return

        Returns:
            List of matching case law entries with scores
        """
        embedding_service = _get_embedding_service()
        if embedding_service is None:
            logger.warning("Embedding service not available for case law search")
            return []

        if not failures:
            return []

        try:
            from kestrel_sovereign.llm.embedding_service import semantic_search, cosine_similarity

            # Build corpus from failures
            failure_texts = [
                f"Prompt: {f['content']} Result: {f.get('metadata', {}).get('audit_reasoning', '')}"
                for f in failures
            ]

            # Use semantic search from embedding service
            search_results = await semantic_search(
                query, failure_texts, embedding_service, top_k=top_k
            )

            # Map back to original failure records
            results = []
            for hit in search_results:
                results.append({
                    "score": hit['score'],
                    "document": failures[hit['index']]
                })

            return results

        except Exception as e:
            logger.error(f"Case law search failed: {e}")
            return []
