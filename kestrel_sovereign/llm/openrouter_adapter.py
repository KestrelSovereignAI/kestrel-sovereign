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
import httpx
from typing import List, Optional, Dict, Any, Type, AsyncIterator, Union
import openai

from pydantic import BaseModel

from .adapter import LLMResponse
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

    def __init__(self):
        self.name = "openrouter"
        self.base_url = get_openrouter_api_base()

        # Get API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY")

        # Optional attribution headers
        self.site_url = os.environ.get("OPENROUTER_SITE_URL", "https://kestrel.ai")
        self.app_name = os.environ.get("OPENROUTER_APP_NAME", "Kestrel")

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
            **kwargs
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
            **kwargs
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
                ))

            logger.info(f"OpenRouter: discovered {len(models)} models")
            return models

        except httpx.HTTPError as e:
            logger.warning(f"OpenRouter HTTP error during model discovery: {e}")
            return []
        except Exception as e:
            logger.warning(f"OpenRouter model discovery failed: {e}")
            return []

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
