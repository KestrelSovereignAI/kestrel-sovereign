"""
Ollama LLM Adapter

Adapter for local Ollama instance with support for:
- Tool/function calling (Ollama 0.4+)
- Streaming responses
- Vision models (llama-vision, llava)
- Structured output via JSON schema (Pydantic models)
- API-based model discovery
"""
import json
import logging
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING, AsyncIterator

import httpx
from pydantic import BaseModel

from .adapter import (
    LLMAdapter,
    LLMResponse,
    ThinkingContentSplitter,
    ThinkingDelta,
    ToolCall,
    split_thinking_from_content,
)
from kestrel_sdk.llm import (
    ProviderCapabilities,
    StructuredOutputMode,
    ToolStreamingMode,
    VisionInputMode,
)
from .model_metadata import ModelInfo, ModelCategory
from .image_utils import get_base64_only
from .retry import with_retry

# Optional ollama import (not available in remote-only containers)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

if TYPE_CHECKING:
    import ollama

logger = logging.getLogger(__name__)


def _extract_message_fields(payload: Any) -> tuple[Optional[str], Optional[str]]:
    """Return ``(thinking, content)`` from an Ollama chat payload.

    Handles both dict shapes (raw HTTP / aiohttp paths) and the Pydantic-
    style ``ChatResponse`` objects returned by the ollama-python SDK.
    ``thinking`` is the native ``message.thinking`` field populated by
    Ollama for thinking-capable models (Gemma 4, gpt-oss, etc.); empty
    or missing fields are returned as ``None`` rather than ``""`` so
    callers can branch with a simple truthiness check.
    """
    if payload is None:
        return None, None

    if isinstance(payload, dict):
        message = payload.get("message") or {}
        if not isinstance(message, dict):
            return None, None
        thinking = message.get("thinking")
        content = message.get("content")
    else:
        message = getattr(payload, "message", None)
        if message is None:
            return None, None
        thinking = getattr(message, "thinking", None)
        content = getattr(message, "content", None)

    return (thinking or None), (content or None)


class OllamaAdapter(LLMAdapter):
    """
    Adapter for interacting with a local Ollama instance.

    Supports:
    - Chat completions with tool calling (Ollama 0.4+)
    - Streaming responses
    - Vision models with base64 images
    """

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_vision=True,
            supports_structured_output=True,
            supports_embeddings=True,
            structured_output_mode=StructuredOutputMode.SCHEMA_FORMAT,
            tool_streaming_mode=ToolStreamingMode.NONSTREAM_FALLBACK,
            vision_input_mode=VisionInputMode.OLLAMA_IMAGES,
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            model_dependent=("tools", "vision", "structured_output"),
            notes=(
                "Tool and vision support are model-dependent in Ollama.",
                "Structured output passes a JSON schema via Ollama's format option.",
            ),
        )

    async def probe_reachable(
        self,
        client: Any,
        *,
        base_url: Optional[str] = None,
        timeout: float = 1.5,
    ) -> Optional[bool]:
        """Probe Ollama's lightweight model-tags endpoint."""
        host = base_url
        if not host:
            http_client = getattr(client, "_client", None)
            host = str(getattr(http_client, "base_url", "") or "") or None
        if not host:
            return None

        url = f"{str(host).rstrip('/')}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                response = await http.get(url)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def aembed(
        self,
        client: "ollama.AsyncClient",
        text: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[List[float]]:
        try:
            response = await client.embed(model=model or "nomic-embed-text", input=text)
        except Exception as exc:
            logger.warning("Ollama embedding failed: %s", exc)
            return None
        embeddings = response.get("embeddings", []) if isinstance(response, dict) else getattr(response, "embeddings", [])
        return list(embeddings[0]) if embeddings else None

    async def aembed_batch(
        self,
        client: "ollama.AsyncClient",
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Optional[List[float]]]:
        if not texts:
            return []
        try:
            response = await client.embed(model=model or "nomic-embed-text", input=texts)
        except Exception as exc:
            logger.warning("Ollama batch embedding failed: %s", exc)
            return [None] * len(texts)
        embeddings = response.get("embeddings", []) if isinstance(response, dict) else getattr(response, "embeddings", [])
        out = [list(item) if item is not None else None for item in embeddings]
        return (out + [None] * len(texts))[:len(texts)]

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Creates messages for Ollama.

        Ollama expects content as plain string, not OpenAI's structured format.
        Images are passed as base64 in a separate 'images' key.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt or images:
            msg = {"role": "user", "content": user_prompt or ""}

            # Handle images for vision models using centralized image_utils
            # Pass provider for auto-resize to Ollama's 1120x1120 limit
            if images:
                images_data = get_base64_only(images, provider="ollama")
                if images_data:
                    msg["images"] = images_data

            messages.append(msg)

        return messages

    async def get_response(
        self,
        client: "ollama.AsyncClient",
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Gets a response from the Ollama model.

        Args:
            client: Ollama async client
            model: Model name to use
            messages: Chat messages
            format: Response format (e.g., "json") - DEPRECATED, use response_format
            tools: Optional list of tools in OpenAI function calling format
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            LLMResponse with content and/or tool calls
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not available in this environment")

        try:
            extra_kwargs = {}

            # Handle structured output via Pydantic model
            # Ollama accepts JSON schema directly via format parameter
            if response_format is not None and issubclass(response_format, BaseModel):
                extra_kwargs["format"] = response_format.model_json_schema()
            elif format == "json":
                extra_kwargs["format"] = "json"

            # Ollama tool calling (Ollama 0.4+)
            if tools:
                extra_kwargs["tools"] = tools

            # Pass through options (temperature, max_tokens, etc.)
            options = {}
            if "temperature" in kwargs:
                options["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                options["num_predict"] = kwargs["max_tokens"]
            if options:
                extra_kwargs["options"] = options

            response = await with_retry(
                client.chat,
                model=model,
                messages=messages,
                **extra_kwargs
            )

            # Extract content - handle both dict and Pydantic model responses
            if isinstance(response, dict):
                content = response.get('message', {}).get('content')
            elif hasattr(response, 'message'):
                content = response.message.content if hasattr(response.message, 'content') else None
            else:
                content = None

            should_split_thinking = content and "<think" in content.lower()
            if (
                response_format is None
                and not kwargs.get("_preserve_thinking_content")
                and should_split_thinking
            ):
                _, content = split_thinking_from_content(content)

            # Native `message.thinking` (Ollama 0.x+ thinking-capable models —
            # Gemma 4, gpt-oss, etc.) is preserved on the raw response so
            # streaming-with-tools fallback can re-surface it as a
            # ThinkingDelta. Non-streaming callers that only want the visible
            # content already get it cleanly via `LLMResponse.content`.

            # If using structured output, validate against the Pydantic model
            if response_format is not None and content:
                try:
                    # Content should be JSON - validate it
                    validated = response_format.model_validate_json(content)
                    # Return as JSON string for consistency with other adapters
                    content = validated.model_dump_json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Ollama structured output validation failed: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during Ollama structured output validation: {e}", exc_info=True)
                    # Return raw content if validation fails

            # Parse tool calls if present (Ollama returns them in message.tool_calls)
            parsed_tool_calls = None
            if isinstance(response, dict):
                raw_tool_calls = response.get('message', {}).get('tool_calls')
            elif hasattr(response, 'message') and hasattr(response.message, 'tool_calls'):
                raw_tool_calls = response.message.tool_calls
            else:
                raw_tool_calls = None

            if raw_tool_calls:
                parsed_tool_calls = []
                for i, tc in enumerate(raw_tool_calls):
                    # Ollama tool call format - handle both dict and object
                    if isinstance(tc, dict):
                        func = tc.get('function', {})
                        args = func.get('arguments', {})
                    elif hasattr(tc, 'function'):
                        func = tc.function
                        args = func.arguments if hasattr(func, 'arguments') else {}
                    else:
                        continue

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            # SDK 0.7.0 malformed-JSON sentinel.
                            args = {"_raw": args}

                    parsed_tool_calls.append(ToolCall(
                        id=f"ollama_call_{i}",  # Ollama doesn't provide IDs
                        name=func.get('name', 'unknown') if isinstance(func, dict) else getattr(func, 'name', 'unknown'),
                        arguments=args
                    ))

            # Extract token usage from response
            # Ollama uses prompt_eval_count and eval_count
            input_tokens = None
            output_tokens = None
            total_tokens = None
            if isinstance(response, dict):
                input_tokens = response.get('prompt_eval_count')
                output_tokens = response.get('eval_count')
            else:
                input_tokens = getattr(response, 'prompt_eval_count', None)
                output_tokens = getattr(response, 'eval_count', None)
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama adapter failed: {e}", exc_info=True)
            raise

    async def _stream_with_usage(
        self,
        client: "ollama.AsyncClient",
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, ThinkingDelta, LLMResponse]]:
        """Stream text/thinking chunks then emit one terminal LLMResponse.

        Shared by :meth:`get_streaming_response` (which filters the terminal
        response out to preserve its text/thinking contract) and the no-tools
        branch of :meth:`get_streaming_response_with_tools` (which forwards it
        so the service layer can meter streamed turns — #1684). Ollama reports
        ``prompt_eval_count`` / ``eval_count`` on the final (``done``) chunk.
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not available in this environment")

        try:
            logger.info(f"Starting Ollama stream for model: {model}")

            extra_kwargs = {}

            # Handle structured output via Pydantic model
            if response_format is not None and issubclass(response_format, BaseModel):
                extra_kwargs["format"] = response_format.model_json_schema()

            # Pass through options
            options = {}
            if "temperature" in kwargs:
                options["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                options["num_predict"] = kwargs["max_tokens"]
            if options:
                extra_kwargs["options"] = options

            stream = await with_retry(
                client.chat,
                model=model,
                messages=messages,
                stream=True,
                **extra_kwargs
            )

            chunk_count = 0
            response_accum = ""
            text_content = ""
            input_tokens = None
            output_tokens = None
            splitter = ThinkingContentSplitter(provider="ollama")
            async for chunk in stream:
                # Usage arrives on the final (done) chunk; keep the latest.
                if isinstance(chunk, dict):
                    it = chunk.get('prompt_eval_count')
                    ot = chunk.get('eval_count')
                else:
                    it = getattr(chunk, 'prompt_eval_count', None)
                    ot = getattr(chunk, 'eval_count', None)
                if it is not None:
                    input_tokens = it
                if ot is not None:
                    output_tokens = ot

                thinking, content = _extract_message_fields(chunk)

                if thinking and response_format is None:
                    yield ThinkingDelta(thinking, provider="ollama")

                if content:
                    chunk_count += 1
                    text_content += content
                    if response_format is not None:
                        # Accumulate for final validation
                        response_accum += content
                        yield content
                    else:
                        for event in splitter.feed(content):
                            yield event

            if response_format is None:
                for event in splitter.flush():
                    yield event

            logger.info(f"Ollama stream completed. Total chunks: {chunk_count}")

            # For structured output, validate the accumulated response at the end
            # Note: This is logged but not yielded since we already yielded chunks
            if response_format is not None and response_accum:
                try:
                    response_format.model_validate_json(response_accum)
                    logger.info("Ollama streaming structured output validated successfully")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Ollama streaming structured output validation failed: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during Ollama streaming structured output validation: {e}", exc_info=True)

            total_tokens = None
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            yield LLMResponse(
                content=text_content if text_content else None,
                tool_calls=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        except ollama.ResponseError as e:
            logger.error(f"Ollama streaming API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}", exc_info=True)
            raise

    async def get_streaming_response(
        self,
        client: "ollama.AsyncClient",
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, ThinkingDelta]]:
        """
        Gets a streaming response from Ollama (text/thinking contract).

        Note: Tool calling during streaming is not well-supported by Ollama.
        For tool calls, use the non-streaming get_response method.

        Delegates to :meth:`_stream_with_usage` and drops the terminal
        usage-bearing :class:`LLMResponse` so existing callers keep their
        ``AsyncIterator[Union[str, ThinkingDelta]]`` contract.

        Yields:
            Text chunks as they arrive. For structured output, yields JSON chunks.
        """
        async for item in self._stream_with_usage(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            **kwargs
        ):
            if not isinstance(item, LLMResponse):
                yield item

    async def get_streaming_response_with_tools(
        self,
        client: "ollama.AsyncClient",
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, ThinkingDelta, LLMResponse]]:
        """
        Stream response with tool call detection.

        FALLBACK STRATEGY: Ollama doesn't support tool calls during streaming well.
        This method uses a fallback approach:
        1. If tools are provided, make a non-streaming call first to detect tool calls
        2. If tool calls are detected, yield the LLMResponse immediately
        3. Otherwise, stream the text response

        This is slightly less efficient than true streaming with tool detection,
        but provides a consistent interface across all providers.

        Args:
            client: Ollama async client
            model: Model name
            messages: Chat messages
            tools: Optional tools in OpenAI format
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters

        Yields:
            str: Text content chunks as they arrive
            LLMResponse: Terminal response at end-of-stream carrying token usage (and tool_calls when present)
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not available in this environment")

        try:
            logger.info(f"Starting Ollama stream with tools for model: {model}")

            # If tools are provided, check for tool calls first via non-streaming
            if tools:
                response = await self.get_response(
                    client=client,
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    _preserve_thinking_content=True,
                    **kwargs
                )

                # If tool calls detected, yield the response immediately
                if response.has_tool_calls:
                    logger.info(f"Ollama detected {len(response.tool_calls)} tool calls")
                    yield response
                    return

                # No tool calls - yield native thinking (if any) + text content.
                # Structured-output mode suppresses thinking to keep the
                # caller's JSON stream clean (matches the regular streaming
                # path's `response_format is None` guard).
                if response_format is None:
                    native_thinking, _ = _extract_message_fields(response.raw)
                    if native_thinking:
                        yield ThinkingDelta(native_thinking, provider="ollama")
                if response.content:
                    should_split_thinking = "<think" in response.content.lower()
                    if not should_split_thinking:
                        yield response.content
                    else:
                        thinking, clean = split_thinking_from_content(response.content)
                        if thinking:
                            yield ThinkingDelta(thinking, provider="ollama")
                        response.content = clean
                        if response.content:
                            yield response.content
                # Terminal LLMResponse carrying usage so the service layer meters
                # this turn (#1684); content was already streamed above. `response`
                # holds the token counts from the non-streaming probe.
                yield response
                return

            # No tools - stream text and forward the terminal usage response so
            # the service layer meters text-only streamed turns (#1684).
            async for item in self._stream_with_usage(
                client=client,
                model=model,
                messages=messages,
                response_format=response_format,
                **kwargs
            ):
                yield item

        except ollama.ResponseError as e:
            logger.error(f"Ollama streaming with tools API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama streaming with tools failed: {e}", exc_info=True)
            raise

    async def continue_with_tool_results(
        self,
        client: "ollama.AsyncClient",
        model: str,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Continue a conversation after executing tool calls.

        Args:
            client: Ollama async client
            model: Model name
            messages: Original messages
            tool_results: List of tool results
            tools: Optional tools for multi-turn

        Returns:
            LLMResponse with the model's follow-up
        """
        # Ollama expects tool results in a specific format
        extended_messages = messages.copy()

        # Add assistant's tool call message (should already be in messages)
        # Add tool results
        for result in tool_results:
            extended_messages.append({
                "role": "tool",
                "content": result["content"]
            })

        return await self.get_response(
            client=client,
            model=model,
            messages=extended_messages,
            tools=tools
        )

    async def _check_tool_support(self, client: "ollama.AsyncClient", model_name: str) -> bool:
        """Check if a model supports tool calling via /api/show metadata.

        Ollama exposes tool capability directly in recent versions and older
        templates include ``.Tools`` when the chat template knows how to render
        function schemas. Trust those provider signals; parameter-count gates
        falsely mark small but tool-capable models such as qwen2.5:0.5b as
        unsupported.

        Returns:
            True if the model advertises tool calling support.
        """
        try:
            info = await client.show(model_name)
            # Extract template — check for .Tools presence
            template = ""
            capabilities = []
            if isinstance(info, dict):
                template = info.get("template", "")
                capabilities = info.get("capabilities") or []
            elif hasattr(info, "template"):
                template = info.template or ""
                capabilities = getattr(info, "capabilities", None) or []

            has_tools_template = ".Tools" in template
            supports = "tools" in capabilities or has_tools_template
            logger.debug(
                f"Tool support for {model_name}: template={has_tools_template}, "
                f"capabilities={capabilities}, supports_tools={supports}"
            )
            return supports
        except Exception as e:
            logger.warning(f"Could not check tool support for {model_name}: {e}")
            return False

    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        """List available models from the local Ollama instance.

        ``client`` is accepted for contract symmetry with
        :meth:`get_response` (SDK 0.5.0). Ollama's discovery uses its
        own ``ollama.AsyncClient()`` against a fixed local endpoint, so
        the parameter is ignored here.

        Uses ollama.list() to discover locally available models,
        then ollama.show() per model to detect tool calling capability.

        Returns:
            List of ModelInfo objects for each available model
        """
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama library not available, returning empty model list")
            return []

        try:
            # Use async client to list models
            client = ollama.AsyncClient()
            response = await client.list()

            # Handle both dict response (older ollama) and Pydantic model (newer ollama)
            if hasattr(response, 'models'):
                raw_models = response.models  # Pydantic model
            elif isinstance(response, dict):
                raw_models = response.get("models", [])
            else:
                raw_models = []

            models = []
            for model_data in raw_models:
                # Handle both dict and Pydantic model objects
                # Newer ollama library uses .model attribute, older used .name or dict
                if hasattr(model_data, 'model'):
                    # Newer Pydantic model (ollama 0.4+)
                    model_name = model_data.model or ""
                    size_bytes = model_data.size if hasattr(model_data, 'size') else 0
                elif hasattr(model_data, 'name'):
                    # Older Pydantic model
                    model_name = model_data.name or ""
                    size_bytes = model_data.size if hasattr(model_data, 'size') else 0
                elif isinstance(model_data, dict):
                    # Dict response
                    model_name = model_data.get("name", "") or model_data.get("model", "")
                    size_bytes = model_data.get("size", 0)
                else:
                    continue

                # Skip models with empty names
                if not model_name:
                    continue

                # Parse size in GB
                size_gb = round(size_bytes / (1024**3), 2) if size_bytes else None

                # Extract display name (remove tag if simple)
                display_name = model_name
                if ":" in model_name:
                    base, tag = model_name.rsplit(":", 1)
                    if tag == "latest":
                        display_name = base
                    else:
                        display_name = f"{base} ({tag})"

                # Detect embedding models
                category = ModelCategory.CHAT
                lower_name = model_name.lower()
                if "embed" in lower_name or "nomic" in lower_name or "minilm" in lower_name:
                    category = ModelCategory.EMBEDDING

                # Detect vision support
                supports_vision = any(v in lower_name for v in ["vision", "llava", "llama3.2"])

                # Detect tool support from API metadata (template + param count)
                supports_tools = await self._check_tool_support(client, model_name)

                models.append(ModelInfo(
                    id=model_name,
                    provider="ollama",
                    display_name=display_name,
                    category=category,
                    supports_vision=supports_vision,
                    supports_tools=supports_tools,
                    supports_streaming=True,
                    size_gb=size_gb,
                ))

            logger.info(f"Ollama returned {len(models)} local models")
            return models

        except ollama.ResponseError as e:
            logger.error(f"Ollama API error while listing models: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}", exc_info=True)
            return []

    # ---- Provider metadata (SDK 0.6.0) -------------------------------------

    def cost_per_1m_tokens(self) -> Optional[Dict[str, float]]:
        # Local inference — no per-token cost. Returning a real
        # ``{"input": 0.0, "output": 0.0}`` (rather than ``None``) lets
        # cost-aware routing prefer Ollama as the cheap option without
        # falling through to the conservative fallback.
        return {"input": 0.0, "output": 0.0}

    def substrate_type(self) -> Optional[str]:
        # Ollama serves many model families (Llama, Mistral, Phi,
        # DeepSeek, Qwen, ...). No single substrate captures it; the
        # substrate-aware paths read the per-model id when they need
        # specifics.
        return None

    def display_name(self) -> Optional[str]:
        return "Ollama"

    def key_env_var(self) -> Optional[str]:
        # Local-only, no API key. None signals "no key-env-var pattern"
        # so the keys UI doesn't prompt for one.
        return None
