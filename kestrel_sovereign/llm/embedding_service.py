"""
Embedding Service for Kestrel.

Provides text embeddings using Ollama's embedding models.
This replaces the need for local sentence-transformers installation.
"""
import logging
from typing import Any, List, Optional
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
        # Tracks whether we've already logged the "model not installed"
        # hint for this service instance. Embedding gets called dozens
        # of times during agent startup (RAG indexing, memory seeding);
        # a missing model would otherwise flood the log with identical
        # red errors. #657.
        self._model_missing_warned = False

    def _handle_embed_error(self, exc: Exception, context: str) -> None:
        """Log an embedding failure at the right severity.

        "Model not found" is a setup issue (``ollama pull <name>``
        fixes it) — log it as a one-time WARNING with the actionable
        command, not as a per-call ERROR. Anything else stays ERROR
        because it suggests something's actually broken.
        """
        status = getattr(exc, "status_code", None)
        msg = str(exc).lower()
        is_missing_model = status == 404 or "not found, try pulling it first" in msg
        if is_missing_model:
            if not self._model_missing_warned:
                logger.warning(
                    "Embedding model %r is not installed in Ollama — "
                    "semantic memory / RAG search will be disabled. "
                    "Run: ollama pull %s",
                    self.model, self.model,
                )
                self._model_missing_warned = True
            return
        logger.error(f"{context}: {exc}")

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
            self._handle_embed_error(e, "Embedding failed")
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
            self._handle_embed_error(e, "Batch embedding failed")
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
            self._handle_embed_error(e, "Async embedding failed")
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
            self._handle_embed_error(e, "Async batch embedding failed")
            return [None] * len(texts)


class ProviderEmbeddingService:
    """Embedding service backed by an initialized LLM provider route.

    Kestrel's storage code consumes one common shape regardless of provider:
    ``aembed(text) -> Optional[list[float]]`` and
    ``aembed_batch(texts) -> list[Optional[list[float]]]``. The provider
    capability metadata records the embedding model and dimension that produced
    those vectors; vectors from different providers/models are not semantically
    interchangeable and should be re-embedded before mixing in one index.
    """

    def __init__(self, provider: dict[str, Any]):
        self.provider = provider
        self.adapter = provider["adapter"]
        self.client = provider["client"]
        capabilities = provider.get("capabilities") or {}
        self.model = capabilities.get("embedding_model")
        self.embedding_dim = capabilities.get("embedding_dim")

    async def aembed(self, text: str) -> Optional[List[float]]:
        return await self.adapter.aembed(
            self.client,
            text,
            model=self.model,
        )

    async def aembed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        return await self.adapter.aembed_batch(
            self.client,
            texts,
            model=self.model,
        )


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0-1, higher is more similar)
    """
    if len(a) != len(b):
        return 0.0

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


def get_provider_embedding_service(llm_service: Optional[Any] = None) -> Optional[ProviderEmbeddingService]:
    """Get the active chat provider's embedding service, if it has one.

    When ``llm_service`` is omitted, a process-local ``LLMService`` is created
    lazily. Callers that already own an agent-scoped ``LLMService`` should pass
    it so model preference and route disabling are respected exactly.
    """
    if llm_service is not None:
        return llm_service.get_embedding_service()

    try:
        from kestrel_sovereign.llm.service import LLMService

        return LLMService().get_embedding_service()
    except Exception as exc:
        logger.warning("Provider embedding service not available: %s", exc)
        return None


def reset_embedding_service() -> None:
    """
    Reset the default embedding service singleton.

    This is primarily for testing purposes to allow re-initialization
    with different parameters.
    """
    global _default_service
    _default_service = None
