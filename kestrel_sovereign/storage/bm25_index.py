"""
BM25 search index for RAG fallback/boost.

Uses BM25Okapi from rank_bm25 for probabilistic document ranking.
This provides keyword-based search that complements embedding search.

"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import rank_bm25, but don't fail if unavailable
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available, BM25 search disabled")


@dataclass
class BM25Result:
    """Result from BM25 search."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class BM25Index:
    """
    BM25 search index for document chunks.

    Tokenization:
    - Lowercase normalization
    - Split on non-alphanumeric characters
    - Filter tokens with length <= 1
    """

    def __init__(self):
        """Initialize empty index."""
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._built = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        if not text:
            return []

        # Lowercase and split on non-alphanumeric
        text_lower = text.lower()
        tokens = re.split(r'[^a-z0-9]+', text_lower)

        # Filter short tokens
        return [t for t in tokens if len(t) > 1]

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a document to the index.

        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Optional metadata dict
        """
        tokens = self.tokenize(content)

        self.documents.append({
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "token_count": len(tokens),
        })
        self.tokenized_docs.append(tokens)
        self._built = False

    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        Add multiple documents to the index.

        Args:
            docs: List of dicts with 'id', 'content', and optional 'metadata'
        """
        for doc in docs:
            self.add_document(
                doc_id=doc.get("id", str(len(self.documents))),
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
            )

    def build(self):
        """Build the BM25 index from added documents."""
        if not BM25_AVAILABLE:
            logger.warning("Cannot build index: rank_bm25 not available")
            return

        if not self.tokenized_docs:
            logger.warning("No documents to index")
            return

        self.bm25 = BM25Okapi(self.tokenized_docs)
        self._built = True
        logger.info(f"Built BM25 index with {len(self.documents)} documents")

    def search(self, query: str, limit: int = 10) -> List[BM25Result]:
        """
        Search the index.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of BM25Result sorted by score descending
        """
        if not BM25_AVAILABLE:
            logger.warning("BM25 search unavailable: rank_bm25 not installed")
            return []

        if not self._built:
            self.build()

        if not self.bm25 or not self.documents:
            return []

        # Tokenize query
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort by score descending
        scored_docs = list(zip(self.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        results = []
        for doc, score in scored_docs[:limit]:
            if score > 0:  # Only include matches
                results.append(BM25Result(
                    doc_id=doc["doc_id"],
                    content=doc["content"],
                    score=score,
                    metadata=doc["metadata"],
                ))

        return results

    def get_top_n(self, query: str, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get top N document indices with scores.

        Args:
            query: Search query
            n: Number of results

        Returns:
            List of (document_index, score) tuples
        """
        if not BM25_AVAILABLE or not self._built or not self.bm25:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get indices sorted by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [(idx, score) for idx, score in indexed_scores[:n] if score > 0]

    def clear(self):
        """Clear the index."""
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
        self._built = False

    @property
    def size(self) -> int:
        """Number of documents in index."""
        return len(self.documents)

    @property
    def is_available(self) -> bool:
        """Check if BM25 is available."""
        return BM25_AVAILABLE


class AsyncBM25Index(BM25Index):
    """
    Async wrapper for BM25Index.

    Since BM25 operations are CPU-bound, this just wraps
    the sync methods. In a production setting, you might
    run these in a thread pool.
    """

    async def asearch(self, query: str, limit: int = 10) -> List[BM25Result]:
        """Async search (wraps sync search)."""
        return self.search(query, limit)

    async def abuild(self):
        """Async build (wraps sync build)."""
        self.build()


def create_bm25_index() -> BM25Index:
    """Factory function to create BM25 index."""
    return BM25Index()


def create_async_bm25_index() -> AsyncBM25Index:
    """Factory function to create async BM25 index."""
    return AsyncBM25Index()
