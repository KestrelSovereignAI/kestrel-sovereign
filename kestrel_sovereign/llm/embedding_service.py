"""
Embedding Service for Kestrel.

Provides text embeddings using Ollama's embedding models.
This replaces the need for local sentence-transformers installation.
"""
import logging
from typing import List, Optional
import numpy as np

from kestrel_sovereign.kestrel_config.defaults import get_ollama_url

logger = logging.getLogger(__name__)

# Optional ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False


class EmbeddingService:
    """
    Embedding service using Ollama's embedding models.

    Default model: nomic-embed-text (768 dimensions)
    Alternative: mxbai-embed-large (1024 dimensions)
    """

    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None
    ):
        """
        Initialize the embedding service.

        Args:
            model: Ollama embedding model to use
            base_url: Ollama server URL (defaults to get_ollama_url())
        """
        self.model = model
        self.base_url = base_url or get_ollama_url()
        self._client = None
        self._async_client = None

    @property
    def client(self):
        """Lazy load sync client."""
        if self._client is None and OLLAMA_AVAILABLE:
            self._client = ollama.Client(host=self.base_url)
        return self._client

    @property
    def async_client(self):
        """Lazy load async client."""
        if self._async_client is None and OLLAMA_AVAILABLE:
            self._async_client = ollama.AsyncClient(host=self.base_url)
        return self._async_client

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text (sync).

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding, or None on failure
        """
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available for embeddings")
            return None

        try:
            response = self.client.embed(model=self.model, input=text)
            # Ollama returns embeddings in response['embeddings']
            embeddings = response.get('embeddings', [])
            if embeddings:
                return embeddings[0]
            return None
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts (sync).

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (or None for failed texts)
        """
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available for embeddings")
            return [None] * len(texts)

        try:
            response = self.client.embed(model=self.model, input=texts)
            return response.get('embeddings', [None] * len(texts))
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [None] * len(texts)

    async def aembed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text (async).

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding, or None on failure
        """
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available for embeddings")
            return None

        try:
            response = await self.async_client.embed(model=self.model, input=text)
            embeddings = response.get('embeddings', [])
            if embeddings:
                return embeddings[0]
            return None
        except Exception as e:
            logger.error(f"Async embedding failed: {e}")
            return None

    async def aembed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts (async).

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (or None for failed texts)
        """
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available for embeddings")
            return [None] * len(texts)

        try:
            response = await self.async_client.embed(model=self.model, input=texts)
            return response.get('embeddings', [None] * len(texts))
        except Exception as e:
            logger.error(f"Async batch embedding failed: {e}")
            return [None] * len(texts)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0-1, higher is more similar)
    """
    a_np = np.array(a)
    b_np = np.array(b)

    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


async def semantic_search(
    query: str,
    documents: List[str],
    embedding_service: EmbeddingService,
    top_k: int = 5
) -> List[dict]:
    """
    Perform semantic search over documents.

    Args:
        query: Search query
        documents: List of documents to search
        embedding_service: Embedding service to use
        top_k: Number of results to return

    Returns:
        List of dicts with 'index', 'document', 'score' keys
    """
    if not documents:
        return []

    # Get query embedding
    query_embedding = await embedding_service.aembed(query)
    if query_embedding is None:
        logger.warning("Failed to embed query, falling back to empty results")
        return []

    # Get document embeddings
    doc_embeddings = await embedding_service.aembed_batch(documents)

    # Calculate similarities
    results = []
    for i, (doc, emb) in enumerate(zip(documents, doc_embeddings)):
        if emb is not None:
            score = cosine_similarity(query_embedding, emb)
            results.append({
                'index': i,
                'document': doc,
                'score': score
            })

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:top_k]


# Singleton instance for convenience
_default_service: Optional[EmbeddingService] = None


def get_embedding_service(
    model: str = EmbeddingService.DEFAULT_MODEL,
    base_url: Optional[str] = None
) -> EmbeddingService:
    """
    Get or create the default embedding service.

    Args:
        model: Model to use (only used on first call)
        base_url: Ollama URL (defaults to get_ollama_url(), only used on first call)

    Returns:
        EmbeddingService instance
    """
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService(model=model, base_url=base_url)
    else:
        # Warn if trying to initialize with different params
        normalized_base_url = base_url or get_ollama_url()
        if (_default_service.model != model or
            _default_service.base_url != normalized_base_url):
            logger.warning(
                f"Attempted to re-initialize embedding service with different params. "
                f"Existing: model={_default_service.model}, base_url={_default_service.base_url}. "
                f"Requested: model={model}, base_url={normalized_base_url}. "
                f"Ignoring new params and returning existing instance."
            )
    return _default_service


def reset_embedding_service() -> None:
    """
    Reset the default embedding service singleton.

    This is primarily for testing purposes to allow re-initialization
    with different parameters.
    """
    global _default_service
    _default_service = None
