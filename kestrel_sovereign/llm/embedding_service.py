"""
Embedding Service for Kestrel.

Provides text embeddings using Ollama's embedding models.
This replaces the need for local sentence-transformers installation.
"""
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

from kestrel_sovereign.kestrel_config.defaults import get_ollama_url

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingProfile:
    """Identity of an embedding configuration (#1477).

    Two vectors produced by the same ``EmbeddingProfile`` live in the
    same semantic coordinate space and can be compared by cosine.
    Vectors with different profile ids — even at the same dimension —
    cannot. The 12-char hex digest is stable across restarts so old
    rows match new ones whenever the operator's config didn't change.

    ``space_id`` defaults to ``"<provider>:<model>"`` but operators can
    override to force-merge profiles they have evidence live in the
    same space (e.g. two providers wrapping the same upstream model).
    ``normalized`` records whether the provider returns L2-normalized
    vectors — semantically equivalent unit vectors from a model that
    returns both normalized and raw outputs would otherwise have
    different cosine semantics under the same model name.
    """

    provider: str
    model: str
    dim: int
    space_id: str
    normalized: bool

    @property
    def profile_id(self) -> str:
        """Stable 12-char hex digest derived from (space_id, dim, normalized).

        ``space_id`` defaults to ``"<provider>:<model>"`` so distinct
        provider/model pairs naturally land on distinct ids. When an
        operator sets ``embedding_space_id`` capability to force-merge
        two services they have evidence live in the same coordinate
        space (e.g. an OpenRouter-wrapped model vs the upstream
        original), both services produce the SAME profile id and
        their rows are visible to each other in cosine kNN. Including
        the raw provider/model in the hash would defeat the override.
        (Codex P2 on #1477.)
        """
        payload = (
            f"{self.space_id}|{int(self.dim)}"
            f"|{str(bool(self.normalized)).lower()}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def derive_embedding_profile(
    *,
    provider: str,
    model: str,
    dim: int,
    normalized: bool = False,
    space_id: Optional[str] = None,
) -> EmbeddingProfile:
    """Build an ``EmbeddingProfile`` from raw fields.

    ``space_id`` defaults to ``"<provider>:<model>"`` when None — the
    "two different vendors of the same upstream model" override is
    opt-in and rare.
    """
    if not provider or not model or not dim or int(dim) <= 0:
        raise ValueError(
            "derive_embedding_profile requires non-empty provider, model, "
            f"and positive dim (got provider={provider!r}, model={model!r}, "
            f"dim={dim!r})"
        )
    return EmbeddingProfile(
        provider=str(provider),
        model=str(model),
        dim=int(dim),
        space_id=str(space_id) if space_id else f"{provider}:{model}",
        normalized=bool(normalized),
    )

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

    def describe(self) -> Optional[EmbeddingProfile]:
        """Return the ``EmbeddingProfile`` for this Ollama-only service.

        Hard-codes ``provider="ollama"`` and the well-known dims for
        the two default models (#1477). Returns ``None`` for any
        model not in the known dim table — the legacy service has no
        provider capability metadata to probe, so storage falls back
        to leaving ``embedding_profile_id`` NULL (= invisible to
        profile-filtered kNN) rather than guessing a dim and risking
        mixed-profile garbage. Operators that want stamping for
        unusual Ollama models should use the
        :class:`ProviderEmbeddingService` path via ``LLMService``
        instead.
        """
        # Known dims for the two recommended Ollama embedding models.
        # Other models fall through to None — operators get a clear
        # signal (NULL profile id) and stable behavior rather than a
        # silently-wrong guess.
        known_dims = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
        }
        dim = known_dims.get(self.model)
        if not dim:
            return None
        try:
            return derive_embedding_profile(
                provider="ollama",
                model=self.model,
                dim=dim,
                normalized=False,
            )
        except ValueError:
            return None

    def current_profile_id(self) -> Optional[str]:
        """Convenience: ``describe().profile_id`` or ``None``."""
        profile = self.describe()
        return profile.profile_id if profile else None


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
        # #1477 normalization flag — capability-declared; defaults to
        # False because most providers (OpenAI, Vertex, Ollama
        # nomic-embed-text) return raw vectors and let the caller
        # cosine-normalize. Operators flip this for providers known
        # to return unit vectors.
        self._normalized = bool(capabilities.get("embedding_normalized", False))
        # Optional space-id override to merge two providers that wrap
        # the same upstream model (rare). Default profile uses
        # ``"<provider>:<model>"``.
        self._space_id = capabilities.get("embedding_space_id")

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

    def describe(self) -> Optional[EmbeddingProfile]:
        """Return the ``EmbeddingProfile`` for this service, or ``None``.

        ``None`` indicates the service is missing the metadata needed
        to build a stable profile id (no embedding_model or no
        embedding_dim in capabilities). Storage code treats this as
        "don't stamp" and the row's ``embedding_profile_id`` stays
        NULL — making it invisible to profile-filtered kNN reads,
        consistent with pre-0.21 rows. Better to lose recall than to
        mix vectors with no provenance.
        """
        if not self.model or not self.embedding_dim:
            return None
        # ``provider`` is the route's vendor — the human-readable label
        # operators see in config (``"openai"``, ``"anthropic"``,
        # ``"ollama"``). Falls back to the full ``"<vendor>:<route>"``
        # name when vendor isn't set (shouldn't happen for routes built
        # via ``ProviderRegistry`` but defensive against third-party
        # adapters).
        provider_label = (
            self.provider.get("vendor")
            or self.provider.get("name")
            or "unknown"
        )
        try:
            return derive_embedding_profile(
                provider=provider_label,
                model=self.model,
                dim=self.embedding_dim,
                normalized=self._normalized,
                space_id=self._space_id,
            )
        except ValueError as exc:
            logger.warning(
                "Could not derive embedding profile for %s: %s",
                provider_label, exc,
            )
            return None

    def current_profile_id(self) -> Optional[str]:
        """Convenience: ``describe().profile_id`` or ``None``."""
        profile = self.describe()
        return profile.profile_id if profile else None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score in ``[-1, 1]`` (higher = more similar);
        ``0.0`` when either vector is zero, empty, or has a different
        length from the other. The length guard catches mismatched-dim
        bugs that ``numpy.dot`` would otherwise execute happily,
        returning a meaningless number.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # #1477 — defense-in-depth alongside the embedding_profile_id
        # filter. The kNN backends should never feed a mismatched-dim
        # row through to here (they enforce dim at the column level)
        # and the profile filter cuts mixed-model rows; but if a
        # caller bypasses those layers, we still fail safe instead of
        # returning a mathematically-defined-but-meaningless dot
        # product.
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
