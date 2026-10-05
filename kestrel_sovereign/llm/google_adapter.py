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
from enum import Enum
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
from .output_ceiling import (
    OutputCeilingUnknownError,
    OutputCeilings,
    attach_stop_reason,
    join_output_ceiling_notice,
    output_ceiling_notice_chunk,
    output_limit_cut_notice,
    reported_token_limit,
)

logger = logging.getLogger(__name__)

#: Gemini's ``finish_reason`` for a response cut at ``max_output_tokens``.
_MAX_TOKENS_FINISH_REASON = "MAX_TOKENS"


def _gemini_model_id(model: str) -> str:
    """``model`` without the ``models/`` resource prefix discovery strips."""
    return model[len("models/"):] if model.startswith("models/") else model


def _finish_reason(candidate: Any) -> Optional[str]:
    """A Gemini candidate's finish reason by name (``"MAX_TOKENS"``), if any.

    google-genai reports it as a ``FinishReason`` member; the name is the
    provider's own wire value, and what telemetry records.
    """
    reason = getattr(candidate, "finish_reason", None)
    if isinstance(reason, Enum):
        reason = reason.name
    return reason if isinstance(reason, str) and reason else None


def _normalized_google_genai_usage(
    usage: Any,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Return SDK-semantic Gemini usage with cached prompt tokens split out.

    google-genai reports ``prompt_token_count`` and ``total_token_count``
    inclusive of ``cached_content_token_count``. ``LLMResponse`` defines its
    input and total fields as excluding separately reported cache buckets, so
    normalize that provider shape once for both Google API and Vertex routes.
    """

    def field(name: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    def token_count(name: str) -> Optional[int]:
        value = field(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    input_tokens = token_count("prompt_token_count")
    output_tokens = token_count("candidates_token_count")
    total_tokens = token_count("total_token_count")
    cache_read_input_tokens = token_count("cached_content_token_count")

    if cache_read_input_tokens is not None:
        if input_tokens is not None:
            input_tokens = max(0, input_tokens - cache_read_input_tokens)
        if total_tokens is not None:
            total_tokens = max(0, total_tokens - cache_read_input_tokens)
    if total_tokens is None and (
        input_tokens is not None or output_tokens is not None
    ):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return (
        input_tokens,
        output_tokens,
        total_tokens,
        cache_read_input_tokens,
    )


class GoogleAdapter(LLMAdapter):
    """
    Adapter for Google Gemini API.

    Note: Gemini uses a different message format than OpenAI.
    Uses 'contents' with 'role' and 'parts'.
    """

    def __init__(self) -> None:
        super().__init__()
        # Output ceilings Gemini reported (``outputTokenLimit``), keyed by
        # model id (#3355). Filled by model discovery and, for a model
        # discovery has not described, by one ``models.get`` on first use.
        self._output_ceilings = OutputCeilings()

    # ---- Output ceiling (#3355) --------------------------------------------

    async def _model_output_ceiling(self, client: Any, model: str) -> int:
        """The largest ``max_output_tokens`` Gemini accepts for ``model``.

        The model's own ``outputTokenLimit``, never a framework number: a
        literal 8,192 silently cut every Gemini turn (#3355). Raises
        :class:`OutputCeilingUnknownError` when Gemini does not report one —
        a guess is never sent.
        """
        model_id = _gemini_model_id(model)
        known = self._output_ceilings.get(model_id)
        if known is not None:
            return known
        record = await client.aio.models.get(model=model)
        limit = reported_token_limit(getattr(record, "output_token_limit", None))
        if limit is None:
            raise OutputCeilingUnknownError(model_id, provider="Google Gemini")
        self._output_ceilings.remember(model_id, limit)
        return limit

    async def _generation_config(
        self,
        client: Any,
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        call_kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional[int]]:
        """The ``generate_content`` config, and the model ceiling if it was used.

        A ``max_tokens`` the caller chose is sent as given and no ceiling is
        returned: a cut at that budget is the caller's to read from the
        finish reason. Otherwise the model's own ceiling is sent and returned,
        so a response that exhausted the model's whole output can be told from
        one that stopped at a smaller, deliberate budget.
        """
        requested = call_kwargs.get("max_tokens")
        model_ceiling = None
        if requested is None:
            model_ceiling = await self._model_output_ceiling(client, model)
        config: Dict[str, Any] = {
            "max_output_tokens": requested if requested is not None else model_ceiling,
        }
        if "temperature" in call_kwargs:
            config["temperature"] = call_kwargs["temperature"]
        if tools:
            config["tools"] = [{
                "function_declarations": self._convert_tools_to_gemini_format(tools)
            }]
        return config, model_ceiling

    @staticmethod
    def _output_ceiling_notice(
        model: str,
        finish_reason: Optional[str],
        model_ceiling: Optional[int],
    ) -> Optional[str]:
        """The notice for a response cut at the model's own ceiling, else None."""
        return output_limit_cut_notice(
            provider="Google Gemini",
            model=model,
            stop_reason=finish_reason,
            cut_reason=_MAX_TOKENS_FINISH_REASON,
            caller_budget=model_ceiling is None,
            model_ceiling=model_ceiling,
        )

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
            config, model_ceiling = await self._generation_config(
                client, model, tools, kwargs,
            )

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
            parts = list(parts or [])

            # #3355: Gemini's own verdict on whether the response is finished.
            # A MAX_TOKENS stop cuts generation inside its LAST part; when
            # that part is a function call it was cut mid-generation, and it
            # must not be handed on as a call to execute.
            finish_reason = _finish_reason(candidate)
            cut_part = (
                parts[-1]
                if finish_reason == _MAX_TOKENS_FINISH_REASON and parts
                and getattr(parts[-1], "function_call", None) is not None
                else None
            )
            if cut_part is not None:
                logger.warning(
                    "Dropping function call %s from a Gemini response cut by "
                    "%s: it is incomplete",
                    getattr(cut_part.function_call, "name", None), finish_reason,
                )

            for part in parts:
                if part is cut_part:
                    continue
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

            input_tokens = output_tokens = total_tokens = None
            cache_read_input_tokens = None
            usage_metadata = getattr(response, "usage_metadata", None)
            if usage_metadata is not None:
                (
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_read_input_tokens,
                ) = _normalized_google_genai_usage(usage_metadata)

            notice = self._output_ceiling_notice(model, finish_reason, model_ceiling)
            if notice is not None:
                content = join_output_ceiling_notice(content, notice)

            return attach_stop_reason(LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ), finish_reason)

        except Exception as e:
            logger.error(f"Google Gemini API error: {e}", exc_info=True)
            raise

    async def _stream_with_usage(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Stream text plus one terminal response carrying final usage.

        Yields:
            Text chunks as they arrive, then a usage-bearing response.
        """
        try:
            config, model_ceiling = await self._generation_config(
                client, model, tools, kwargs,
            )

            # Stream via the maintained google-genai async client, honoring the
            # routed model (mirrors VertexAIAdapter).
            stream = await client.aio.models.generate_content_stream(
                model=model,
                contents=messages,
                config=config,
            )

            text_content = ""
            usage_metadata = None
            finish_reason = None
            async for chunk in stream:
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage_metadata = chunk.usage_metadata
                # #3355: the finish reason rides on the last chunk's candidate.
                candidates = getattr(chunk, "candidates", None)
                if candidates:
                    finish_reason = _finish_reason(candidates[0]) or finish_reason
                text = getattr(chunk, "text", None)
                if text:
                    text_content += text
                    yield text

            # #3355: the streamed text is already on screen, so a cut at the
            # model's ceiling is marked by a final chunk — and carried on the
            # terminal response's content, which mirrors what was shown.
            notice = self._output_ceiling_notice(model, finish_reason, model_ceiling)
            if notice is not None:
                notice_chunk = output_ceiling_notice_chunk(
                    notice, follows_text=bool(text_content),
                )
                text_content += notice_chunk
                yield notice_chunk

            input_tokens = output_tokens = total_tokens = None
            cache_read_input_tokens = None
            if usage_metadata is not None:
                (
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_read_input_tokens,
                ) = _normalized_google_genai_usage(usage_metadata)
            yield attach_stop_reason(LLMResponse(
                content=text_content or None,
                tool_calls=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ), finish_reason)

        except Exception as e:
            logger.error(f"Google Gemini streaming error: {e}", exc_info=True)
            raise

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Preserve the public text-only stream while retaining one source."""
        async for item in self._stream_with_usage(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        ):
            if isinstance(item, str):
                yield item

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Expose Gemini's usage-bearing stream to the service finalizer.

        Google declares non-streaming tool fallback. Probe once when tools are
        present so function calls remain structured; text-only calls use the
        provider stream and always finish with one usage-bearing response.
        """
        if tools:
            response = await self.get_response(
                client=client,
                model=model,
                messages=messages,
                tools=tools,
                **kwargs,
            )
            if response.has_tool_calls:
                yield response
                return
            if response.content:
                yield response.content
            yield response
            return

        async for item in self._stream_with_usage(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        ):
            yield item

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
                    context_limit=reported_token_limit(
                        getattr(model, 'input_token_limit', None)
                    ),
                    # #3355: the ceiling requests send for this model.
                    output_limit=reported_token_limit(
                        getattr(model, 'output_token_limit', None)
                    ),
                    supports_vision=supports_vision,
                    supports_tools="gemini" in lower_id,  # Only Gemini models support tools
                    supports_streaming=True,
                ))

            self._output_ceilings.learn(models)
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
