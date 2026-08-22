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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Sequence, Tuple

from .async_database import AsyncDatabase
from .bm25_index import AsyncBM25Index, BM25_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedChunk:
    """One stored chunk with the embedding already computed for it.

    Carries the profile id alongside the vector because kNN filters by the
    active embedding profile — a vector separated from the model that produced
    it is not searchable, only storable.
    """

    content: str
    embedding: List[float] = field(default_factory=list)
    profile_id: Optional[str] = None


def _get_embedding_service(llm_service: Optional[Any] = None):
    """Resolve the active chat provider's embedding service."""
    try:
        from kestrel_sovereign.llm.embedding_service import get_provider_embedding_service

        return get_provider_embedding_service(llm_service)
    except Exception as e:
        logger.warning(f"Embedding service not available: {e}")
        return None


def _serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding to little-endian float32 bytes for SQLite storage.

    Explicit little-endian (``<``) so the bytes round-trip on big-endian
    hosts and across a cross-architecture migration (mirrors
    async_conversation_store; #1653).
    """
    return struct.pack(f'<{len(embedding)}f', *embedding)


def _deserialize_embedding(data: bytes) -> List[float]:
    """Deserialize embedding from little-endian float32 bytes.

    A length that isn't a multiple of 4 can't be a valid float32 vector —
    skip it rather than ``// 4``-truncate to noise (#1653).
    """
    if len(data) % 4 != 0:
        logger.warning(
            "Skipping embedding: %d bytes not a multiple of 4.", len(data),
        )
        return []
    count = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f'<{count}f', data))


class AsyncRAGStore:
    """
    Async RAG (Retrieval-Augmented Generation) storage.

    Uses hybrid search combining:
    - Ollama embeddings for semantic similarity (primary)
    - BM25 for keyword matching (fallback/boost)

    Results are merged using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        db: AsyncDatabase,
        llm_service: Optional[Any] = None,
        agent_id: str = "",
    ):
        self.db = db
        self._bm25_index: Optional[AsyncBM25Index] = None
        self._bm25_built = False
        self._llm_service = llm_service
        self.agent_id = agent_id

    def bind_agent(self, agent_id: str) -> None:
        """Bind document reads and writes to one tenant capability."""
        if not agent_id:
            raise ValueError("RAG ownership binding requires a non-empty agent_id")
        if self.agent_id and self.agent_id != agent_id:
            raise ValueError("RAG store is already bound to a different agent")
        self.agent_id = agent_id
        self._bm25_index = None
        self._bm25_built = False

    def _owner_scope(
        self,
        chunk_table: str = "document_chunks",
        owner_alias: str = "rag_chunk_owner",
    ) -> Tuple[str, Tuple[Any, ...]]:
        """Return chunk and file capability predicates for tenant reads."""
        # Some narrowly-scoped callers construct a store with ``__new__`` to
        # exercise retrieval without initializing an ownership capability.
        # Preserve that legacy unbound behavior; ordinary construction always
        # defines ``agent_id`` in ``__init__``.
        agent_id = getattr(self, "agent_id", "")
        if not agent_id:
            return "1 = 1", ()
        file_owner_alias = f"{owner_alias}_file"
        return (
            "EXISTS (SELECT 1 FROM document_chunk_owners AS "
            f"{owner_alias} WHERE {owner_alias}.chunk_id = "
            f"{chunk_table}.chunk_id AND {owner_alias}.agent_id = ?) "
            "AND EXISTS (SELECT 1 FROM file_owners AS "
            f"{file_owner_alias} WHERE {file_owner_alias}.content_hash = "
            f"{chunk_table}.file_hash AND {file_owner_alias}.agent_id = ?)",
            (agent_id, agent_id),
        )

    def _get_embedding_service(self):
        llm_service = getattr(self, "_llm_service", None)
        if llm_service is not None:
            return _get_embedding_service(llm_service)
        return _get_embedding_service()
    
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
        if self.agent_id:
            owned = await self.db.fetchone(
                "SELECT 1 FROM file_owners "
                "WHERE content_hash = ? AND agent_id = ?",
                (file_hash, self.agent_id),
            )
            if not owned:
                raise ValueError("Cannot index a file outside the bound agent")

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
            embedding_service = self._get_embedding_service()
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

            chunk_id: int
            async with self.db.transaction():
                if self.db.backend_type == "postgres":
                    row = await self.db.fetchone(
                        "INSERT INTO document_chunks "
                        "(file_hash, content, embedding) VALUES (?, ?, ?) "
                        "RETURNING chunk_id",
                        (file_hash, chunk, embedding_blob),
                    )
                else:
                    await self.db.execute(
                        "INSERT INTO document_chunks "
                        "(file_hash, content, embedding) VALUES (?, ?, ?)",
                        (file_hash, chunk, embedding_blob),
                    )
                    row = await self.db.fetchone("SELECT last_insert_rowid()")
                if not row:
                    raise RuntimeError("Could not resolve inserted RAG chunk id")
                chunk_id = int(row[0])
                if self.agent_id:
                    # Chunk plaintext is caller-supplied, so ownership must be
                    # recorded atomically with the row. File co-owners do not
                    # implicitly share derived annotations or chunk content.
                    await self.db.execute(
                        "INSERT INTO document_chunk_owners "
                        "(chunk_id, agent_id) VALUES (?, ?)",
                        (chunk_id, self.agent_id),
                    )
            if embedding:
                new_chunk_ids.append((chunk_id, embedding))

        # #1477 — derive the active embedding profile id once (same
        # for every chunk we just batched) so kNN can filter
        # cross-model rows out of semantic recall.
        profile_id: Optional[str] = None
        if compute_embeddings and new_chunk_ids:
            embedding_service = self._get_embedding_service()
            if embedding_service is not None and hasattr(
                embedding_service, "current_profile_id"
            ):
                try:
                    profile_id = embedding_service.current_profile_id()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "current_profile_id() failed for RAG chunk batch: %s",
                        exc,
                    )
            if profile_id is not None and embedding_service is not None:
                from .sqla.embedding_profile import upsert_embedding_profile
                try:
                    await upsert_embedding_profile(
                        self.db, embedding_service, profile_id,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "RAG profile registry upsert failed (non-fatal): %s",
                        exc,
                    )

        # Dual-write embedding_vec + embedding_profile_id for every
        # chunk we just inserted. Outside the INSERT loop because the
        # column might not exist yet on a DB whose Phase-2 migration
        # hasn't run — the helper catches that and logs info.
        for chunk_id, embedding in new_chunk_ids:
            await self._write_embedding_vec(chunk_id, embedding, profile_id)

        await self.db.commit()
        self._bm25_built = False  # Invalidate BM25 index

        return len(chunks)

    async def read_indexed_chunks(self, file_hash: str) -> List[IndexedChunk]:
        """Return this tenant's chunks for ``file_hash`` with their embeddings.

        The counterpart to :meth:`store_precomputed_chunks`. Together they move
        an already-indexed document between databases without re-running the
        embedding model — the chunk text, its vector and the profile that
        produced the vector are carried verbatim, so the copy retrieves
        identically to the original.

        ``embedding_profile_id`` predates #1477 on some databases; a database
        without the column still yields chunks, with ``profile_id`` None.
        """
        owner_scope, owner_params = self._owner_scope()
        # Ask whether the column exists rather than letting a SELECT fail to
        # find out. A failed statement aborts the whole transaction on
        # PostgreSQL, so probing by exception would destroy a caller's open
        # unit of work — the defect fixed in ``_write_embedding_vec`` — and a
        # blanket ``except`` would also swallow a transient error and silently
        # drop every chunk's profile id, dropping them all out of
        # profile-filtered kNN.
        has_profile_id = await self.db._column_exists(
            "document_chunks", "embedding_profile_id",
        )
        columns = (
            "content, embedding, embedding_profile_id"
            if has_profile_id
            else "content, embedding"
        )
        rows = await self.db.fetchall(
            f"SELECT {columns} FROM document_chunks "
            f"WHERE file_hash = ? AND {owner_scope} ORDER BY chunk_id",
            (file_hash,) + owner_params,
        )

        chunks: List[IndexedChunk] = []
        for row in rows:
            blob = row[1]
            embedding = _deserialize_embedding(bytes(blob)) if blob else []
            profile_id = row[2] if len(row) > 2 else None
            chunks.append(
                IndexedChunk(
                    content=row[0],
                    embedding=embedding,
                    profile_id=profile_id,
                )
            )
        return chunks

    async def store_precomputed_chunks(
        self, file_hash: str, chunks: Sequence[IndexedChunk],
    ) -> int:
        """Store chunks whose embeddings were computed elsewhere.

        Unlike :meth:`chunk_document` this neither re-chunks the document nor
        calls the embedding model: the caller already holds both, and
        re-deriving them would silently produce a *different* index (chunk
        boundaries depend on a ``chunk_size`` the rows do not record, and a
        re-embed under a changed model yields vectors in another coordinate
        space).

        Idempotent: this tenant's existing chunks for ``file_hash`` are
        replaced, so re-running after a partial or interrupted copy converges
        on exactly ``len(chunks)`` chunks rather than accumulating duplicates.
        """
        if self.agent_id:
            owned = await self.db.fetchone(
                "SELECT 1 FROM file_owners "
                "WHERE content_hash = ? AND agent_id = ?",
                (file_hash, self.agent_id),
            )
            if not owned:
                raise ValueError("Cannot index a file outside the bound agent")

        await self.delete_chunks_for_file(file_hash)

        written: List[Tuple[int, IndexedChunk]] = []
        for chunk in chunks:
            embedding_blob = (
                _serialize_embedding(chunk.embedding) if chunk.embedding else None
            )
            async with self.db.transaction():
                if self.db.backend_type == "postgres":
                    row = await self.db.fetchone(
                        "INSERT INTO document_chunks "
                        "(file_hash, content, embedding) VALUES (?, ?, ?) "
                        "RETURNING chunk_id",
                        (file_hash, chunk.content, embedding_blob),
                    )
                else:
                    await self.db.execute(
                        "INSERT INTO document_chunks "
                        "(file_hash, content, embedding) VALUES (?, ?, ?)",
                        (file_hash, chunk.content, embedding_blob),
                    )
                    row = await self.db.fetchone("SELECT last_insert_rowid()")
                if not row:
                    raise RuntimeError("Could not resolve inserted RAG chunk id")
                chunk_id = int(row[0])
                if self.agent_id:
                    await self.db.execute(
                        "INSERT INTO document_chunk_owners "
                        "(chunk_id, agent_id) VALUES (?, ?)",
                        (chunk_id, self.agent_id),
                    )
            written.append((chunk_id, chunk))

        # Same dual-write as chunk_document, and outside the insert loop for
        # the same reason: the parallel column may not exist yet.
        for chunk_id, chunk in written:
            if chunk.embedding:
                await self._write_embedding_vec(
                    chunk_id, chunk.embedding, chunk.profile_id,
                )

        await self.db.commit()
        self._bm25_built = False  # Invalidate BM25 index
        return len(written)

    async def _write_embedding_vec(
        self,
        chunk_id: int,
        embedding: List[float],
        profile_id: Optional[str] = None,
    ) -> None:
        """Dual-write the embedding (+ #1477 profile id) to the parallel
        ``embedding_vec`` column.

        - On Postgres, formats the list as pgvector's text shape
          (``[v1,v2,…]``) and binds with a ``::vector`` cast.
        - On SQLite, packs to float32 little-endian bytes — same shape
          stored in the legacy ``embedding`` BLOB column.

        ``profile_id`` may be NULL on pre-#1477 deployments; kNN
        filters by the active profile so NULL rows correctly stay
        out of mixed-coordinate-space recall.

        Errors are non-fatal: the most likely cause is that a
        migration hasn't created the column yet on this DB, in which
        case the legacy ``embedding`` column is already written and
        search degrades to the in-Python fallback.

        Each attempt runs in its own ``transaction()`` so "non-fatal"
        stays true when a CALLER holds a transaction open. PostgreSQL
        aborts the whole transaction on any failed statement, so an
        unwrapped failure here would poison the caller's transaction:
        every later statement raises, both fallbacks below are
        swallowed at debug level, and the caller commits nothing while
        being told it succeeded. Reproduced on a live PostgreSQL where
        ``embedding_vec`` was deferred (``migrations.py`` skips the
        column until the table has an embedded row, which is exactly a
        fresh runtime database): 47 chunks reported written, 0
        committed. A nested ``transaction()`` is a savepoint on
        PostgreSQL (#1726) and a no-op yield on SQLite, so this
        contains the damage without changing either backend's
        semantics.
        """
        backend_type = getattr(self.db, "backend_type", None)
        try:
            async with self.db.transaction():
                if backend_type == "postgres":
                    vec_text = "[" + ",".join(repr(float(v)) for v in embedding) + "]"
                    await self.db.execute(
                        "UPDATE document_chunks SET embedding_vec = ?::vector, "
                        "embedding_profile_id = ? WHERE chunk_id = ?",
                        (vec_text, profile_id, chunk_id),
                    )
                else:
                    await self.db.execute(
                        "UPDATE document_chunks SET embedding_vec = ?, "
                        "embedding_profile_id = ? WHERE chunk_id = ?",
                        (_serialize_embedding(embedding), profile_id, chunk_id),
                    )
            return
        except Exception as e:
            # Partial-migration shape (only one of the two new columns
            # present). Try each column independently so the row gets
            # the maximum amount of metadata its DB supports. (Codex
            # P2 round 4 on #1477.)
            logger.info(
                "Could not write document_chunks.embedding_vec + "
                "embedding_profile_id for chunk %s in one UPDATE (%s); "
                "trying each column independently.", chunk_id, e,
            )

        # Best-effort vec-only write. Own savepoint, same reason as above.
        try:
            async with self.db.transaction():
                if backend_type == "postgres":
                    vec_text = "[" + ",".join(
                        repr(float(v)) for v in embedding
                    ) + "]"
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
        except Exception as e2:
            logger.debug(
                "document_chunks.embedding_vec write failed for chunk "
                "%s: %s (column likely missing — Phase-2 migration "
                "pending).", chunk_id, e2,
            )
        # Best-effort profile-id-only write so the legacy in-Python
        # fallback (which filters by profile id) still sees this
        # chunk when only the Phase-2 column is missing.
        if profile_id is not None:
            try:
                async with self.db.transaction():
                    await self.db.execute(
                        "UPDATE document_chunks SET embedding_profile_id = ? "
                        "WHERE chunk_id = ?",
                        (profile_id, chunk_id),
                    )
            except Exception as e3:
                logger.debug(
                    "document_chunks.embedding_profile_id write failed "
                    "for chunk %s: %s (column likely missing — #1477 "
                    "migration pending).", chunk_id, e3,
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
        embedding_service = self._get_embedding_service()
        if not embedding_service:
            return []

        try:
            from kestrel_sovereign.llm.embedding_service import (
                aembed_retrieval_query,
            )

            query_embedding = await aembed_retrieval_query(
                embedding_service, query
            )
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return []

        # The generic document-chunk vector spec has no ownership join. Bound
        # stores therefore use the scoped legacy path until the vector layer
        # can express a file_owners semi-join; filtering only after top-k could
        # both starve results and expose cross-tenant candidates.
        # ``__new__``-constructed legacy stores are unbound, just as a store
        # constructed with the default empty ``agent_id`` is. They may use the
        # generic vector backend; a bound store must stay on the ownership-
        # scoped path.
        agent_id = getattr(self, "agent_id", "")
        sf = None if agent_id else self._get_vector_session_factory()
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

            # #1477 — filter kNN by the active embedding profile id
            # so chunks from a different model living in a different
            # coordinate space stay out of cosine. ``None`` means
            # "no filter" — preserves legacy behaviour pre-migration.
            filter_kwargs: Optional[Dict[str, Any]] = None
            current_profile_id: Optional[str] = None
            embedding_service_for_profile = self._get_embedding_service()
            if embedding_service_for_profile is not None and hasattr(
                embedding_service_for_profile, "current_profile_id"
            ):
                try:
                    current_profile_id = (
                        embedding_service_for_profile.current_profile_id()
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "current_profile_id() failed at RAG read time: %s",
                        exc,
                    )
            if current_profile_id is not None:
                filter_kwargs = {"embedding_profile_id": current_profile_id}

            backend = get_vector_backend(session_factory, spec)
            top_k = await backend.knn(packed, k=limit, filter=filter_kwargs)
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
            owner_scope, owner_params = self._owner_scope()
            row = await self.db.fetchone(
                "SELECT chunk_id, file_hash, content FROM document_chunks "
                f"WHERE chunk_id = ? AND {owner_scope}",
                (chunk_id,) + owner_params,
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

        #1477: also applies the profile-id filter so cross-model chunks
        can't sneak into cosine on the legacy path. Tries with the
        filter first; if the column doesn't exist (pre-migration DB),
        retries unfiltered. Catches the codex P2 about the legacy
        fallback bypassing the new filter.
        """
        try:
            from kestrel_sovereign.llm.embedding_service import cosine_similarity

            # Derive current profile id once for the read.
            current_profile_id: Optional[str] = None
            embedding_service_for_profile = self._get_embedding_service()
            if embedding_service_for_profile is not None and hasattr(
                embedding_service_for_profile, "current_profile_id"
            ):
                try:
                    current_profile_id = (
                        embedding_service_for_profile.current_profile_id()
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "current_profile_id() failed in legacy RAG "
                        "search: %s", exc,
                    )

            owner_scope, owner_params = self._owner_scope()
            if current_profile_id is not None:
                try:
                    rows = await self.db.fetchall(
                        "SELECT chunk_id, file_hash, content, embedding "
                        "FROM document_chunks "
                        f"WHERE {owner_scope} AND embedding IS NOT NULL "
                        "AND embedding_profile_id = ?",
                        owner_params + (current_profile_id,),
                    )
                except Exception as exc:
                    # Profile column doesn't exist yet (#1477 migration
                    # hasn't run on this DB). Retry without the filter.
                    logger.debug(
                        "Legacy RAG search failed with profile filter "
                        "(%s); retrying unfiltered.", exc,
                    )
                    rows = await self.db.fetchall(
                        "SELECT chunk_id, file_hash, content, embedding "
                        "FROM document_chunks "
                        f"WHERE {owner_scope} AND embedding IS NOT NULL",
                        owner_params,
                    )
            else:
                rows = await self.db.fetchall(
                    "SELECT chunk_id, file_hash, content, embedding "
                    "FROM document_chunks "
                    f"WHERE {owner_scope} AND embedding IS NOT NULL",
                    owner_params,
                )
            if not rows:
                return []

            scored: List[Dict[str, Any]] = []
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
        """Search using BM25 keyword matching.

        #1477 hardening (codex P2 on #1491): post-filters results by
        the active ``embedding_profile_id`` so a foreign-profile
        chunk can't sneak into the hybrid merge via BM25. The
        cosine isolation principle doesn't strictly require this
        (BM25 doesn't compare vectors) but the operator expectation
        after a profile switch is "old chunks are dormant in every
        search path." Over-fetches to ``limit * 3`` so we still
        return ``limit`` candidates after the filter trims.
        """
        if not BM25_AVAILABLE:
            return []

        try:
            # Build index if needed
            if not self._bm25_built:
                await self._build_bm25_index()

            if not self._bm25_index:
                return []

            # #1477 — resolve the current profile id (best-effort).
            current_profile_id: Optional[str] = None
            embedding_service_for_profile = self._get_embedding_service()
            if embedding_service_for_profile is not None and hasattr(
                embedding_service_for_profile, "current_profile_id"
            ):
                try:
                    current_profile_id = (
                        embedding_service_for_profile.current_profile_id()
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "current_profile_id() failed in BM25: %s", exc,
                    )

            # When the profile filter is active, pass the index's
            # full corpus size as ``raw_limit`` so foreign-profile
            # rows dominating the top ranks can't starve the
            # filter (codex P2 rounds 4 + 6). BM25 ``get_scores``
            # already computes a score per document; slicing the
            # already-sorted list deeper is the same O(N) work.
            # ``BM25Index.search`` caps at ``len(documents)``
            # internally, so passing a huge ``raw_limit`` just
            # surfaces every positive-score hit.
            raw_limit = limit
            if current_profile_id is not None:
                doc_count = len(
                    getattr(self._bm25_index, "documents", None) or []
                )
                raw_limit = max(doc_count, limit)
            results = await self._bm25_index.asearch(query, raw_limit)

            if results and (
                current_profile_id is not None
                or getattr(self, "agent_id", "")
            ):
                # Lookup profile ids for the candidate chunk_ids in
                # bounded batches so SQLite's default ~999-variable
                # parameter limit doesn't crash the IN-list and
                # silently disable the filter (codex P2 round 5).
                # NULL-tolerant: rows that never got an embedding
                # stamp (compute_embeddings=False, aembed failed,
                # pre-#1477) stay in the results; only foreign-
                # profile rows drop (codex P2 round 3).
                _BATCH = 500
                ids = [int(r.doc_id) for r in results]

                # First detect pre-migration shape: if the column
                # doesn't exist, skip only the profile comparison. Ownership
                # candidates are still revalidated before they can return.
                column_present = True
                try:
                    await self.db.fetchall(
                        "SELECT embedding_profile_id FROM document_chunks "
                        "LIMIT 1", (),
                    )
                except Exception as exc:
                    column_present = False
                    logger.debug(
                        "BM25 profile filter unavailable (column missing): "
                        "%s; applying ownership filter only.", exc,
                    )

                profile_by_id: Dict[int, Optional[str]] = {}
                # Track which ids we successfully looked up so a transient
                # failure cannot admit an unverified ownership candidate.
                looked_up: set[int] = set()
                owner_scope, owner_params = self._owner_scope()
                for start in range(0, len(ids), _BATCH):
                    chunk = ids[start:start + _BATCH]
                    placeholders = ",".join("?" for _ in chunk)
                    try:
                        if column_present:
                            profile_rows = await self.db.fetchall(
                                f"SELECT chunk_id, embedding_profile_id "
                                f"FROM document_chunks "
                                f"WHERE chunk_id IN ({placeholders}) "
                                f"AND {owner_scope}",
                                tuple(chunk) + owner_params,
                            )
                            for row in profile_rows:
                                profile_by_id[row[0]] = row[1]
                        else:
                            profile_rows = await self.db.fetchall(
                                f"SELECT chunk_id FROM document_chunks "
                                f"WHERE chunk_id IN ({placeholders}) "
                                f"AND {owner_scope}",
                                tuple(chunk) + owner_params,
                            )
                        looked_up.update(row[0] for row in profile_rows)
                    except Exception as exc:
                        logger.warning(
                            "BM25 ownership/profile lookup batch %d-%d failed "
                            "(%s); dropping those candidates.",
                            start, start + len(chunk), exc,
                        )

                results = [
                    r for r in results
                    if int(r.doc_id) in looked_up
                    and (
                        current_profile_id is None
                        or profile_by_id.get(int(r.doc_id))
                        in (current_profile_id, None)
                    )
                ][:limit]

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

        owner_scope, owner_params = self._owner_scope()
        rows = await self.db.fetchall(
            "SELECT chunk_id, file_hash, content FROM document_chunks "
            f"WHERE {owner_scope}",
            owner_params,
        )

        if not rows:
            self._bm25_index = None
            self._bm25_built = True
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
        """Fallback LIKE-based search.

        #1477 hardening (codex P2 on #1491): applies the active
        ``embedding_profile_id`` filter so a foreign-profile chunk
        can't surface here either. Falls back to the legacy
        unfiltered query if the column doesn't exist (pre-migration
        DB).
        """
        words = query.lower().split()
        if not words:
            return []

        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in words])
        like_params = tuple(f"%{word}%" for word in words)

        # Resolve current profile id (best-effort).
        current_profile_id: Optional[str] = None
        embedding_service_for_profile = self._get_embedding_service()
        if embedding_service_for_profile is not None and hasattr(
            embedding_service_for_profile, "current_profile_id"
        ):
            try:
                current_profile_id = (
                    embedding_service_for_profile.current_profile_id()
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "current_profile_id() failed in LIKE search: %s", exc,
                )

        profile_clause = ""
        profile_params: Tuple[Any, ...] = ()
        if current_profile_id is not None:
            # NULL-tolerant: same reasoning as the BM25 branch —
            # non-embedded chunks (compute_embeddings=False, aembed
            # failure, pre-#1477) stay searchable; only foreign-
            # profile chunks are excluded.
            profile_clause = (
                " AND (embedding_profile_id = ? "
                "OR embedding_profile_id IS NULL)"
            )
            profile_params = (current_profile_id,)

        owner_scope, owner_params = self._owner_scope()
        try:
            rows = await self.db.fetchall(
                f"SELECT chunk_id, file_hash, content FROM document_chunks "
                f"WHERE {owner_scope} AND ({conditions}){profile_clause} LIMIT ?",
                owner_params + like_params + profile_params + (limit,),
            )
        except Exception as exc:
            # Pre-migration DB → retry unfiltered.
            logger.debug(
                "RAG LIKE search with profile filter failed (%s); "
                "retrying unfiltered.", exc,
            )
            rows = await self.db.fetchall(
                f"SELECT chunk_id, file_hash, content FROM document_chunks "
                f"WHERE {owner_scope} AND ({conditions}) LIMIT ?",
                owner_params + like_params + (limit,),
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
        owner_scope, owner_params = self._owner_scope()
        rows = await self.db.fetchall(
            "SELECT content FROM document_chunks "
            f"WHERE file_hash = ? AND {owner_scope} ORDER BY chunk_id",
            (file_hash,) + owner_params,
        )
        return [row[0] for row in rows]
    
    async def delete_chunks_for_file(self, file_hash: str) -> None:
        """Delete only chunks owned by this tenant capability."""
        if self.agent_id:
            async with self.db.transaction():
                owned_rows = await self.db.fetchall(
                    "SELECT chunks.chunk_id FROM document_chunks chunks "
                    "JOIN document_chunk_owners owners "
                    "  ON owners.chunk_id = chunks.chunk_id "
                    " AND owners.agent_id = ? "
                    "WHERE chunks.file_hash = ?",
                    (self.agent_id, file_hash),
                )
                owned_ids = [int(row[0]) for row in owned_rows]
                for start in range(0, len(owned_ids), 500):
                    batch = owned_ids[start:start + 500]
                    placeholders = ",".join("?" for _ in batch)
                    await self.db.execute(
                        "DELETE FROM document_chunk_owners "
                        f"WHERE agent_id = ? AND chunk_id IN ({placeholders})",
                        (self.agent_id, *batch),
                    )
                    await self.db.execute(
                        "DELETE FROM document_chunks "
                        f"WHERE chunk_id IN ({placeholders}) AND NOT EXISTS ("
                        "  SELECT 1 FROM document_chunk_owners owners "
                        "  WHERE owners.chunk_id = document_chunks.chunk_id"
                        ")",
                        tuple(batch),
                    )
            return
        async with self.db.transaction():
            await self.db.execute(
                "DELETE FROM document_chunk_owners WHERE chunk_id IN ("
                "  SELECT chunk_id FROM document_chunks WHERE file_hash = ?"
                ")",
                (file_hash,),
            )
            await self.db.execute(
                "DELETE FROM document_chunks WHERE file_hash = ?",
                (file_hash,),
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
        embedding_service = self._get_embedding_service()
        if embedding_service is None:
            logger.warning("Embedding service not available for case law search")
            return []

        if not failures:
            return []

        try:
            from kestrel_sovereign.llm.embedding_service import semantic_search

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
