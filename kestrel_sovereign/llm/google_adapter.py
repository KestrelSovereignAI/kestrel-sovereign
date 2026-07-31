"""
Google Gemini Adapter

Adapter for Google's Gemini API with support for:
- Tool/function calling
- Vision (image inputs)
- Streaming responses
- API-based model discovery
"""
import logging
import os
from typing import Any, Dict, List, Optional, Union, AsyncIterator

from .adapter import LLMAdapter, LLMResponse, ToolCall
from kestrel_sdk.llm import (
    ProviderCapabilities,
    StructuredOutputMode,
    ToolStreamingMode,
    VisionInputMode,
)
from .model_metadata import ModelInfo, ModelCategory
from .image_utils import process_images

logger = logging.getLogger(__name__)


class GoogleAdapter(LLMAdapter):
    """
    Adapter for Google Gemini API.

    Note: Gemini uses a different message format than OpenAI.
    Uses 'contents' with 'role' and 'parts'.
    """

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_vision=True,
            supports_structured_output=False,
            supports_embeddings=True,
            structured_output_mode=StructuredOutputMode.NONE,
            tool_streaming_mode=ToolStreamingMode.NONSTREAM_FALLBACK,
            vision_input_mode=VisionInputMode.GEMINI_INLINE_DATA,
            embedding_model="text-embedding-004",
            embedding_dim=768,
            notes=(
                "Direct Gemini adapter does not yet wire response_format into generation_config.",
                "Streaming tool calls use the framework's non-streaming fallback path.",
            ),
        )

    @staticmethod
    def _embedding_values(item: Any) -> Optional[List[float]]:
        if isinstance(item, dict):
            values = item.get("values")
        else:
            values = getattr(item, "values", None)
        return list(values) if values is not None else None

    @classmethod
    def _embeddings_from_response(cls, response: Any, count: int) -> List[Optional[List[float]]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        out: List[Optional[List[float]]] = [None] * count
        for idx, item in enumerate(embeddings or []):
            if idx >= count:
                break
            out[idx] = cls._embedding_values(item)
        return out

    async def aembed(
        self,
        client: Any,
        text: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[List[float]]:
        # Use the maintained google-genai async client surface (mirrors
        # VertexAIAdapter). The registry now hands GoogleAdapter a
        # ``google.genai.Client``, not the deprecated module client.
        response = await client.aio.models.embed_content(
            model=model or "text-embedding-004",
            contents=text,
        )
        return self._embeddings_from_response(response, 1)[0]

    async def aembed_batch(
        self,
        client: Any,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Optional[List[float]]]:
        if not texts:
            return []
        response = await client.aio.models.embed_content(
            model=model or "text-embedding-004",
            contents=texts,
        )
        return self._embeddings_from_response(response, len(texts))

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create messages in Google Gemini format.

        Gemini uses 'contents' with role and parts.
        System prompts are included as initial user/model exchange.
        """
        messages = []

        # Add system instruction as user message with model acknowledgment
        if system_prompt:
            messages.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
            messages.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })

        # Add actual user prompt with optional images
        if user_prompt or images:
            parts = []

            if user_prompt:
                parts.append({"text": user_prompt})

            # Handle images using centralized image_utils
            if images:
                for processed in process_images(images):
                    parts.append({
                        "inline_data": {
                            "mime_type": processed.mime_type,
                            "data": processed.data
                        }
                    })

            messages.append({"role": "user", "parts": parts})

        return messages

    def _convert_tools_to_gemini_format(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Gemini format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        Gemini format (function declarations):
        {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
        """
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_declarations.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}})
                })
        return function_declarations

    async def get_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Get response from Google Gemini API.

        Args:
            client: google-genai ``Client`` instance (``client.aio.models``)
            model: Model name (e.g., 'gemini-2.0-flash-exp')
            messages: List of message dicts in Gemini format
            format: Response format (ignored for Gemini)
            tools: Optional tools in OpenAI format (will be converted)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and/or tool calls
        """
        try:
            config: Dict[str, Any] = {
                "max_output_tokens": kwargs.get("max_tokens", 8192),
            }

            if "temperature" in kwargs:
                config["temperature"] = kwargs["temperature"]

            # Prepare tool config
            if tools:
                config["tools"] = [{
                    "function_declarations": self._convert_tools_to_gemini_format(tools)
                }]

            # Generate content via the maintained google-genai async client,
            # honoring the routed model (mirrors VertexAIAdapter).
            response = await client.aio.models.generate_content(
                model=model,
                contents=messages,
                config=config,
            )

            # Parse response
            content = None
            parsed_tool_calls = None

            # Guard the candidate/content/parts chain: on a safety-blocked prompt
            # or a MAX_TOKENS-empty candidate the google-genai response has
            # candidate.content is None (or content.parts is None), so iterating
            # it unguarded raises AttributeError/TypeError instead of returning a
            # documented LLMResponse. Mirror VertexAIAdapter's guard.
            candidate = response.candidates[0] if response.candidates else None
            parts = None
            if candidate is not None and getattr(candidate, "content", None) is not None:
                parts = getattr(candidate.content, "parts", None)
            for part in parts or []:
                if getattr(part, "text", None):
                    content = part.text
                # google-genai Part ALWAYS carries a `function_call` attribute
                # (defaults to None), so `hasattr` is always True — check the
                # value, or a plain text/thought part (function_call=None) would
                # enter this branch and crash on fc.name / dict(fc.args).
                elif getattr(part, "function_call", None) is not None:
                    if parsed_tool_calls is None:
                        parsed_tool_calls = []
                    fc = part.function_call
                    # Gemini function_call has name and args
                    args = dict(fc.args) if getattr(fc, "args", None) else {}
                    parsed_tool_calls.append(ToolCall(
                        id=f"gemini_call_{len(parsed_tool_calls)}",
                        name=fc.name,
                        arguments=args
                    ))

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response
            )

        except Exception as e:
            logger.error(f"Google Gemini API error: {e}", exc_info=True)
            raise

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Get streaming response from Google Gemini.

        Yields:
            Text chunks as they arrive
        """
        try:
            config: Dict[str, Any] = {
                "max_output_tokens": kwargs.get("max_tokens", 8192),
            }

            if "temperature" in kwargs:
                config["temperature"] = kwargs["temperature"]

            if tools:
                config["tools"] = [{
                    "function_declarations": self._convert_tools_to_gemini_format(tools)
                }]

            # Stream via the maintained google-genai async client, honoring the
            # routed model (mirrors VertexAIAdapter).
            stream = await client.aio.models.generate_content_stream(
                model=model,
                contents=messages,
                config=config,
            )

            async for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Google Gemini streaming error: {e}", exc_info=True)
            raise

    async def continue_with_tool_results(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Continue conversation after executing tool calls.

        Gemini expects function responses in a specific format.
        """
        extended_messages = messages.copy()

        # Add function response parts
        function_responses = []
        for result in tool_results:
            function_responses.append({
                "function_response": {
                    "name": result.get("name", "unknown"),
                    "response": {"result": result["content"]}
                }
            })

        extended_messages.append({
            "role": "function",
            "parts": function_responses
        })

        return await self.get_response(
            client=client,
            model=model,
            messages=extended_messages,
            tools=tools,
            **kwargs
        )

    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        """List available models from Google Gemini API.

        ``client`` is accepted for contract symmetry with
        :meth:`get_response` (SDK 0.5.0). The Google generative-ai SDK
        is module-scoped (``genai.configure`` + ``genai.list_models``)
        rather than client-scoped, so the parameter is ignored here.

        Uses the google-generativeai SDK's genai.list_models().

        Returns:
            List of ModelInfo objects for each available model
        """
        try:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY/GEMINI_API_KEY not set, returning empty model list")
                return []

            try:
                import google.generativeai as genai
            except ImportError:
                logger.warning("google-generativeai not installed, returning empty model list")
                return []

            genai.configure(api_key=api_key)
            models = []

            for model in genai.list_models():
                model_name = model.name if hasattr(model, 'name') else str(model)
                # Remove "models/" prefix if present
                model_id = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
                display_name = getattr(model, 'display_name', model_id)
                description = getattr(model, 'description', None)

                # Detect model category
                category = ModelCategory.CHAT
                lower_id = model_id.lower()
                if "embed" in lower_id:
                    category = ModelCategory.EMBEDDING
                elif "image" in lower_id or "imagen" in lower_id:
                    category = ModelCategory.IMAGE

                # Detect vision support
                supports_vision = "gemini" in lower_id or "vision" in lower_id

                models.append(ModelInfo(
                    id=model_id,
                    provider="google",
                    display_name=display_name,
                    category=category,
                    description=description,
                    context_limit=getattr(model, 'input_token_limit', None),
                    supports_vision=supports_vision,
                    supports_tools="gemini" in lower_id,  # Only Gemini models support tools
                    supports_streaming=True,
                ))

            logger.info(f"Google returned {len(models)} models")
            return models

        except Exception as e:
            logger.error(f"Failed to list Google models: {e}", exc_info=True)
            return []

    # ---- Provider metadata (SDK 0.6.0) -------------------------------------

    def cost_per_1m_tokens(self) -> Optional[Dict[str, float]]:
        # Gemini Pro / 2.5 Flash midpoint pricing as the conservative
        # default. Per-model rates land on ModelInfo when discovery
        # surfaces them.
        return {"input": 1.25, "output": 5.00}

    def substrate_type(self) -> Optional[str]:
        return "gemini"

    def display_name(self) -> Optional[str]:
        return "Google Gemini"

    def key_env_var(self) -> Optional[str]:
        # The framework prefers GOOGLE_API_KEY but accepts GEMINI_API_KEY
        # as a fallback (see list_models). Surfacing the canonical name
        # here for the keys-UI; the alternate is read at call time.
        return "GOOGLE_API_KEY"
