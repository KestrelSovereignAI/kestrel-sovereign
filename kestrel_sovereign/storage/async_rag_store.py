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

        # Store chunks with embeddings. The legacy ``embedding`` BYTEA
        # / BLOB column is written here as before; the parallel
        # ``embedding_vec`` column added by the Phase-2 migration is
        # populated by ``_write_embedding_vec`` so the vector search
        # backend (PgVectorBackend on PG, PurePythonBackend on SQLite)
        # can pick it up. (kestrel-sovereign #1454 followup: same
        # parallel-column design used for saved_items.)
        new_chunk_ids: List[Tuple[int, List[float]]] = []
        for chunk, embedding in zip(chunks, embeddings):
            embedding_blob = None
            if embedding:
                embedding_blob = _serialize_embedding(embedding)

            cursor = await self.db.execute(
                "INSERT INTO document_chunks (file_hash, content, embedding) VALUES (?, ?, ?)",
                (file_hash, chunk, embedding_blob)
            )
            if embedding:
                # Capture chunk_id so we can populate embedding_vec
                # outside the INSERT (the column may not exist yet on
                # a DB whose migration hasn't run; _write_embedding_vec
                # handles that gracefully).
                chunk_id = getattr(cursor, "lastrowid", None)
                if chunk_id is None:
                    # Fallback for backends that don't expose lastrowid
                    # on the execute() return — look up by file_hash +
                    # content. Slow but only runs on backends that lack
                    # cursor.lastrowid support.
                    row = await self.db.fetchone(
                        "SELECT chunk_id FROM document_chunks WHERE file_hash = ? "
                        "AND content = ? ORDER BY chunk_id DESC LIMIT 1",
                        (file_hash, chunk),
                    )
                    if row:
                        chunk_id = row[0]
                if chunk_id is not None:
                    new_chunk_ids.append((chunk_id, embedding))

        # Dual-write embedding_vec for every chunk we just inserted.
        # Outside the INSERT loop because the column might not exist
        # yet on a DB whose Phase-2 migration hasn't run — the helper
        # catches that and logs info.
        for chunk_id, embedding in new_chunk_ids:
            await self._write_embedding_vec(chunk_id, embedding)

        await self.db.commit()
        self._bm25_built = False  # Invalidate BM25 index

        return len(chunks)

    async def _write_embedding_vec(
        self, chunk_id: int, embedding: List[float]
    ) -> None:
        """Dual-write the embedding to the parallel ``embedding_vec`` column.

        - On Postgres, formats the list as pgvector's text shape
          (``[v1,v2,…]``) and binds with a ``::vector`` cast.
        - On SQLite, packs to float32 little-endian bytes — same shape
          stored in the legacy ``embedding`` BLOB column.

        Errors are non-fatal: the most likely cause is that the
        Phase-2 migration hasn't created the column yet on this DB,
        in which case the legacy ``embedding`` column is already
        written and search degrades to the in-Python fallback.
        """
        backend_type = getattr(self.db, "backend_type", None)
        try:
            if backend_type == "postgres":
                vec_text = "[" + ",".join(repr(float(v)) for v in embedding) + "]"
                await self.db.execute(
                    "UPDATE document_chunks SET embedding_vec = ?::vector "
                    "WHERE chunk_id = ?",
                    (vec_text, chunk_id),
                )
            else:
                await self.db.execute(
                    "UPDATE document_chunks SET embedding_vec = ? "
                    "WHERE chunk_id = ?",
                    (_serialize_embedding(embedding), chunk_id),
                )
        except Exception as e:
            logger.info(
                "Could not write document_chunks.embedding_vec for chunk %s: "
                "%s. Vector search will use the in-Python fallback until the "
                "next boot's migration runs.",
                chunk_id, e,
            )
    
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

        Routes through :mod:`kestrel_sovereign.storage.vector` so PG
        deployments hit pgvector's native ``<=>`` operator + HNSW
        index, while SQLite still uses the in-Python numpy cosine
        path. ``min_score`` (#1404) filters weak semantic matches
        before the RRF merge upstream.

        Falls back to the legacy in-Python loop if the SQLAlchemy
        session factory can't be built (e.g. ``AsyncDatabase`` from a
        bare pool with no DSN, or the Phase-2 migration hasn't run yet).
        """
        embedding_service = _get_embedding_service()
        if not embedding_service:
            return []

        try:
            query_embedding = await embedding_service.aembed(query)
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return []

        sf = self._get_vector_session_factory()
        if sf is not None:
            scored = await self._search_via_vector_backend(
                sf, query_embedding, limit, min_score,
            )
            if scored is not None:
                return scored

        # Fallback: legacy in-Python loop. Same logic as pre-#1447 —
        # used when the SQLA session factory can't be built or the
        # vector backend errors. The legacy path reads the BYTEA
        # ``embedding`` column, which is still populated by
        # ``chunk_document`` for every embedded chunk.
        return await self._legacy_in_python_search(
            query_embedding, limit, min_score,
        )

    def _get_vector_session_factory(self):
        """Lazy-build and cache a SQLAlchemy session factory.

        Falls back to ``None`` (legacy search path) if
        ``make_session_factory`` can't construct one for this
        ``AsyncDatabase`` (in-memory SQLite, ``from_pool`` PG without
        DSN, etc.).
        """
        if getattr(self, "_sqla_factory", None) is not None:
            return self._sqla_factory
        if getattr(self, "_sqla_factory_unavailable", False):
            return None
        try:
            from kestrel_sovereign.storage.sqla import make_session_factory
            self._sqla_factory = make_session_factory(self.db)
            return self._sqla_factory
        except Exception as e:
            logger.info(
                "SQLAlchemy session factory unavailable for document_chunks "
                "vector search (%s); falling back to in-Python search.",
                e,
            )
            self._sqla_factory_unavailable = True
            return None

    async def _search_via_vector_backend(
        self,
        session_factory,
        query_embedding: List[float],
        limit: int,
        min_score: float,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run kNN through the sovereign vector backend.

        Returns ``None`` on any unexpected failure so the caller falls
        back to the legacy in-Python path. ``min_score`` is applied
        after the backend returns top-k, since neither
        ``PgVectorBackend`` nor ``PurePythonBackend`` exposes a
        server-side similarity threshold.
        """
        try:
            from kestrel_sovereign.storage.sqla import build_document_chunk_spec
            from kestrel_sovereign.storage.vector import get_vector_backend

            packed = _serialize_embedding(query_embedding)
            spec = build_document_chunk_spec(dimension=len(query_embedding))

            backend = get_vector_backend(session_factory, spec)
            top_k = await backend.knn(packed, k=limit, filter=None)
        except Exception as e:
            logger.warning(
                "Vector-backend RAG search failed (%s); falling back to "
                "in-Python legacy path.", e,
            )
            return None

        if not top_k:
            return []

        # Materialize: fetch full chunk rows by chunk_id. Preserves
        # similarity ordering. Each id from the backend is a string
        # (the backend stringifies for cross-dialect uniformity); we
        # need an INTEGER chunk_id for the SQL bind, so cast back.
        scored: List[Dict[str, Any]] = []
        for chunk_id_str, score in top_k:
            if score < min_score:
                continue
            try:
                chunk_id = int(chunk_id_str)
            except (TypeError, ValueError):
                continue
            row = await self.db.fetchone(
                "SELECT chunk_id, file_hash, content FROM document_chunks "
                "WHERE chunk_id = ?",
                (chunk_id,),
            )
            if not row:
                # Row deleted between knn() and the lookup — skip.
                continue
            scored.append({
                "chunk_id": row[0],
                "file_hash": row[1],
                "content": row[2],
                "score": float(score),
                "source": "embedding",
            })
        return scored

    async def _legacy_in_python_search(
        self,
        query_embedding: List[float],
        limit: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        """Fallback used when the SQLA session factory isn't available.

        Reads the legacy ``embedding`` BYTEA / BLOB column and runs
        cosine in Python. Matches the pre-#1447 RAG search behavior
        exactly, including the #1404 ``min_score`` floor.
        """
        try:
            from kestrel_sovereign.llm.embedding_service import cosine_similarity

            rows = await self.db.fetchall(
                "SELECT chunk_id, file_hash, content, embedding FROM document_chunks "
                "WHERE embedding IS NOT NULL"
            )
            if not rows:
                return []

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
