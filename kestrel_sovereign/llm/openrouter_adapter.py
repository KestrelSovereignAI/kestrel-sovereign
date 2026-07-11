"""
OpenRouter LLM Adapter

Adapter for OpenRouter - a meta-provider aggregating multiple LLM providers.
Extends OpenAIAdapter since OpenRouter uses OpenAI-compatible API.

OpenRouter provides access to models from:
- Anthropic (Claude)
- OpenAI (GPT-4, etc.)
- Google (Gemini)
- Meta (Llama)
- Mistral
- And many more...

Key features:
- Dynamic model discovery from OpenRouter's /models endpoint
- Rich model metadata (pricing, context length, capabilities)
- OpenAI-compatible chat completions API
"""
import os
import logging
from dataclasses import replace
import httpx
from typing import List, Optional, Dict, Any, Type, AsyncIterator, TYPE_CHECKING, Union
import openai

if TYPE_CHECKING:
    from .embedding_discovery import EmbeddingModelInfo

from pydantic import BaseModel

from .adapter import LLMResponse
from kestrel_sdk.llm import ProviderCapabilities
from .openai_adapter import OpenAIAdapter
from .model_metadata import ModelInfo, ModelCategory
from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT
from kestrel_sovereign.kestrel_config.defaults import get_openrouter_api_base

logger = logging.getLogger(__name__)


class OpenRouterAdapter(OpenAIAdapter):
    """
    Adapter for OpenRouter - a meta-provider aggregating multiple LLM providers.

    OpenRouter provides a unified OpenAI-compatible API for accessing models from
    many providers. This adapter extends OpenAIAdapter with OpenRouter-specific
    model discovery.

    Environment variables:
        OPENROUTER_API_KEY: Required API key for OpenRouter
        OPENROUTER_SITE_URL: Optional site URL for leaderboard attribution
        OPENROUTER_APP_NAME: Optional app name for leaderboard attribution
    """

    def __init__(
        self,
        *,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        supports_embeddings: Optional[bool] = None,
    ):
        # Start from the base with embeddings OFF; OpenRouter must NOT inherit
        # OpenAI's ``text-embedding-3-small`` default. As a meta-provider it
        # only embeds when a route explicitly configures an embedding model
        # (#2288) — advertising what the ROUTE serves, not what the vendor's
        # catalog theoretically offers.
        super().__init__(name="openrouter", supports_embeddings=False)
        self._embedding_model = str(embedding_model) if embedding_model else None
        self._embedding_dim = int(embedding_dim) if embedding_dim else None
        if supports_embeddings is None:
            supports_embeddings = bool(self._embedding_model)
        # A route can't claim embeddings without a model to serve them.
        self._supports_embeddings = bool(supports_embeddings) and bool(
            self._embedding_model
        )

        self.base_url = get_openrouter_api_base()

        # Get API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY")

        # Optional attribution headers
        self.site_url = os.environ.get("OPENROUTER_SITE_URL", "https://kestrel.ai")
        self.app_name = os.environ.get("OPENROUTER_APP_NAME", "Kestrel")

    def provider_capabilities(self) -> ProviderCapabilities:
        capabilities = super().provider_capabilities()
        return replace(
            capabilities,
            # Truthful, ROUTE-scoped embedding advertisement (#2288): only when
            # an embedding model is actually configured for this route.
            supports_embeddings=self._supports_embeddings,
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
            model_dependent=("tools", "vision", "structured_output"),
            notes=(
                "OpenRouter forwards requests to many upstream providers; per-model support is authoritative.",
                "The adapter can send OpenAI-compatible tools, images, and response_format payloads.",
            ),
        )

    def embedding_space_id(self) -> Optional[str]:
        """Model-keyed embedding space id for the meta-provider route (#2288).

        OpenRouter is a meta-provider, so the coordinate space a vector lives
        in is determined by the *upstream* model, not by "openrouter". Two
        different upstream models reached through the same route are different
        spaces; the SAME upstream model reached through OpenRouter or directly
        is the same space. Key on the upstream model id (stripped of the
        ``vendor/`` routing prefix) plus the served dimension — e.g.
        ``qwen/qwen3-embedding-0.6b`` at 768 dims → ``qwen3-embedding-0.6b@768``.
        Matryoshka truncation makes the same model at a different dimension a
        different space, so the dim is part of the key. Returns ``None`` when no
        embedding model / dimension is configured (nothing to stamp).
        """
        if not self._embedding_model or not self._embedding_dim:
            return None
        upstream = self._embedding_model.split("/", 1)[-1]
        return f"{upstream}@{int(self._embedding_dim)}"

    async def aembed(
        self,
        client: openai.AsyncOpenAI,
        text: str,
        *,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[List[float]]:
        """Embed one text via OpenRouter's OpenAI-compatible ``/v1/embeddings``.

        No hardcoded default model: a meta-provider embedding call requires an
        explicit model (route config ``embedding_model`` or the ``model`` arg).
        Forwards the configured ``dimensions`` for Matryoshka-capable models.
        """
        model = model or self._embedding_model
        if not model:
            raise ValueError(
                "OpenRouter embeddings require an explicit embedding model "
                "(set route-level 'embedding_model', e.g. "
                "'qwen/qwen3-embedding-0.6b'); no default is assumed for a "
                "meta-provider."
            )
        if dimensions is None:
            dimensions = self._embedding_dim
        return await super().aembed(
            client, text, model=model, dimensions=dimensions, **kwargs
        )

    async def aembed_batch(
        self,
        client: openai.AsyncOpenAI,
        texts: List[str],
        *,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Optional[List[float]]]:
        model = model or self._embedding_model
        if not model:
            raise ValueError(
                "OpenRouter embeddings require an explicit embedding model "
                "(set route-level 'embedding_model', e.g. "
                "'qwen/qwen3-embedding-0.6b'); no default is assumed for a "
                "meta-provider."
            )
        if dimensions is None:
            dimensions = self._embedding_dim
        return await super().aembed_batch(
            client, texts, model=model, dimensions=dimensions, **kwargs
        )

    def _get_client(self) -> openai.AsyncOpenAI:
        """Create an OpenAI client configured for OpenRouter."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        return openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            },
            # max_retries=0: llm/retry.py is the single retry owner.
            max_retries=0,
        )

    @staticmethod
    def _with_usage_accounting(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Inject OpenRouter's ``usage: {include: true}`` into ``extra_body``.

        Surfaces the exact per-generation ``usage.cost`` (USD) so the metering
        callback can bill actual cost (+ margin) instead of recomputing from a
        models-pricing table that drifts from the real charge. Merges rather
        than clobbers a caller-supplied ``extra_body``.
        See kestrel #1806 / frinz #359.
        """
        kwargs = dict(kwargs)
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        usage_opt = dict(extra_body.get("usage") or {})
        usage_opt.setdefault("include", True)
        extra_body["usage"] = usage_opt
        kwargs["extra_body"] = extra_body
        return kwargs

    async def get_response(
        self,
        client: Optional[openai.AsyncOpenAI],
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Get a response from an OpenRouter model.

        If no client is provided, creates one configured for OpenRouter.
        """
        # Use OpenRouter client if none provided
        if client is None:
            client = self._get_client()

        return await super().get_response(
            client=client,
            model=model,
            messages=messages,
            format=format,
            tools=tools,
            response_format=response_format,
            **self._with_usage_accounting(kwargs)
        )

    async def get_streaming_response(
        self,
        client: Optional[openai.AsyncOpenAI],
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Get a streaming response from an OpenRouter model."""
        if client is None:
            client = self._get_client()

        async for chunk in super().get_streaming_response(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            **kwargs
        ):
            yield chunk

    async def get_streaming_response_with_tools(
        self,
        client: Optional[openai.AsyncOpenAI],
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Get a streaming response with tool call detection."""
        if client is None:
            client = self._get_client()

        async for item in super().get_streaming_response_with_tools(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            **self._with_usage_accounting(kwargs)
        ):
            yield item

    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        """Fetch models from OpenRouter API with rich metadata.

        ``client`` is accepted for contract symmetry with
        :meth:`get_response` (SDK 0.5.0). OpenRouter's catalog is
        served from a fixed URL with bearer-token auth that this
        adapter manages via its own ``httpx.AsyncClient``, so the
        parameter is ignored here.

        OpenRouter's /models endpoint returns detailed information including:
        - Model ID (e.g., "anthropic/claude-3-opus")
        - Display name
        - Description
        - Context length
        - Pricing (per token)
        - Supported features (vision, tools, etc.)

        Returns:
            List of ModelInfo objects for all available OpenRouter models
        """
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set, returning empty model list")
            return []

        try:
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=HTTP_TIMEOUT_DEFAULT
                )
                response.raise_for_status()
                data = response.json()

            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if not model_id:
                    continue

                # Extract underlying provider from model ID (e.g., "anthropic/claude-3-opus")
                underlying_provider = model_id.split("/")[0] if "/" in model_id else None

                # Parse capabilities
                architecture = m.get("architecture", {})
                supported_params = m.get("supported_parameters", [])
                input_modalities = architecture.get("input_modalities", [])

                supports_vision = "image" in input_modalities
                supports_tools = "tools" in supported_params or "tool_choice" in supported_params

                # Parse pricing (OpenRouter uses per-token pricing)
                pricing = m.get("pricing", {})
                # OpenRouter prices are strings like "0.00001" per token
                try:
                    input_price = float(pricing.get("prompt", "0")) * 1_000_000  # Convert to per-million
                    output_price = float(pricing.get("completion", "0")) * 1_000_000
                except (ValueError, TypeError):
                    input_price = 0.0
                    output_price = 0.0

                # Determine category
                category = ModelCategory.CHAT
                if "embedding" in model_id.lower():
                    category = ModelCategory.EMBEDDING
                elif "image" in model_id.lower() or "vision" in model_id.lower():
                    category = ModelCategory.IMAGE

                models.append(ModelInfo(
                    id=model_id,
                    provider="openrouter",
                    display_name=m.get("name", model_id),
                    description=m.get("description"),
                    category=category,
                    is_featured=False,  # Will be enriched from catalog
                    is_hidden=False,
                    context_limit=m.get("context_length", 4096),
                    supports_vision=supports_vision,
                    supports_tools=supports_tools,
                    supports_streaming=True,  # OpenRouter streams every chat route
                    # Carry the upstream substrate (e.g. ``anthropic`` from
                    # ``anthropic/claude-3-opus``) so UI can facet the
                    # meta-provider catalog without re-parsing ids (#2262).
                    underlying_provider=underlying_provider,
                ))

            logger.info(f"OpenRouter: discovered {len(models)} models")
            return models

        except httpx.HTTPError as e:
            logger.warning(f"OpenRouter HTTP error during model discovery: {e}")
            return []
        except Exception as e:
            logger.warning(f"OpenRouter model discovery failed: {e}")
            return []

    async def list_embedding_models(self, client: Any = None) -> List["EmbeddingModelInfo"]:
        """Discover OpenRouter embedding models from the DEDICATED endpoint (#2338).

        OpenRouter serves embedding models from ``GET /api/v1/embeddings/models``,
        NOT the generic ``/models`` list (which omits them entirely, and
        ``?category=embedding`` returns empty). Verified live 2026-07-10: 26
        models with metadata (gemini-embedding-2, pplx-embed-v1,
        qwen3-embedding-8b, gte, e5, ...).

        ``client`` is accepted for contract symmetry; OpenRouter's catalog is
        fetched with this adapter's own bearer-token ``httpx.AsyncClient``.
        """
        from .embedding_discovery import EmbeddingModelInfo

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY not set, returning empty embedding model list"
            )
            return []

        url = f"{self.base_url}/embeddings/models"
        try:
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=HTTP_TIMEOUT_DEFAULT,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as e:
            logger.warning(f"OpenRouter embedding discovery HTTP error: {e}")
            return []
        except Exception as e:
            logger.warning(f"OpenRouter embedding discovery failed: {e}")
            return []

        results: List[EmbeddingModelInfo] = []
        for m in payload.get("data", []):
            model_id = m.get("id") or m.get("canonical_slug") or ""
            if not model_id:
                continue
            native_dim, dim_options = self._parse_openrouter_embedding_dims(m)
            results.append(EmbeddingModelInfo(
                id=model_id,
                provider="openrouter",
                display_name=m.get("name") or model_id,
                native_dim=native_dim,
                dim_options=dim_options,
                context_limit=m.get("context_length"),
                description=m.get("description"),
            ))

        logger.info(f"OpenRouter: discovered {len(results)} embedding models")
        return results

    @staticmethod
    def _parse_openrouter_embedding_dims(model_data: Dict[str, Any]):
        """Extract (native_dim, dim_options) from an OpenRouter embedding entry.

        The dedicated endpoint reports dimensions in a few shapes across models:
        a scalar ``output_dimensions`` / ``dimensions`` for fixed-size models,
        and a ``[min, max]`` (or explicit list) Matryoshka range for MRL models.
        Missing dims are fine — the set-time probe (#2326) proves the served
        size before use.
        """
        arch = model_data.get("architecture") or {}
        raw = (
            model_data.get("output_dimensions")
            or model_data.get("dimensions")
            or arch.get("output_dimensions")
            or arch.get("dimensions")
        )
        native_dim: Optional[int] = None
        dim_options: List[int] = []
        if isinstance(raw, (int, float)):
            native_dim = int(raw)
        elif isinstance(raw, dict):
            hi = raw.get("max") or raw.get("default")
            if hi is not None:
                native_dim = int(hi)
            dim_options = [int(v) for v in raw.values() if isinstance(v, (int, float))]
        elif isinstance(raw, (list, tuple)) and raw:
            nums = [int(v) for v in raw if isinstance(v, (int, float))]
            if nums:
                native_dim = max(nums)
                dim_options = sorted(set(nums))
        return native_dim, dim_options

    def _extract_tags(self, model_data: Dict[str, Any]) -> List[str]:
        """Extract tags from OpenRouter model data."""
        tags = []

        architecture = model_data.get("architecture", {})

        # Add modality tags
        for modality in architecture.get("input_modalities", []):
            if modality != "text":  # text is assumed
                tags.append(modality)

        # Add capability tags
        if "tools" in model_data.get("supported_parameters", []):
            tags.append("function-calling")

        # Add size/speed indicators from model ID
        model_id = model_data.get("id", "").lower()
        if "mini" in model_id or "small" in model_id:
            tags.append("fast")
        if "large" in model_id or "opus" in model_id:
            tags.append("powerful")

        return tags

    # ---- Provider metadata (SDK 0.6.0) -------------------------------------
    #
    # OpenRouterAdapter inherits from OpenAIAdapter, so substrate_type
    # / display_name / key_env_var must be re-overridden to avoid
    # reporting "gpt" / "OpenAI" / "OPENAI_API_KEY" for what is in
    # fact a meta-aggregator pointing at many vendors.

    def cost_per_1m_tokens(self) -> Optional[Dict[str, float]]:
        # Per-model pricing comes through ModelInfo from OpenRouter's
        # /models endpoint; the adapter-level rate is meaningless for
        # an aggregator.
        return None

    def substrate_type(self) -> Optional[str]:
        # OpenRouter routes to many substrates (claude, gpt, gemini,
        # llama). Substrate-aware paths read the per-model id (e.g.
        # ``anthropic/claude-3.5-sonnet``) rather than the route's
        # adapter substrate.
        return None

    def display_name(self) -> Optional[str]:
        return "OpenRouter"

    def key_env_var(self) -> Optional[str]:
        return "OPENROUTER_API_KEY"
