"""
OpenAI LLM Adapter

Adapter for OpenAI's chat completion API with full support for:
- Tool/function calling
- Streaming responses
- Vision (image inputs)
- API-based model discovery
"""
import json
import os
import openai
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, AsyncIterator, Type, Union

import httpx
from pydantic import BaseModel

from kestrel_sdk.llm import ToolCallStarted

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
try:  # SDK v5 optional surface.
    from kestrel_sdk.llm import (
        BatchHandle,
        BatchMode,
        BatchRequest,
        BatchResult,
        BatchStatus,
        CodeExecOptions,
        FileRef,
        FilesMode,
        PromptCacheMode,
        RawResponse,
        ReasoningControlMode,
        RequestOptions,
        ServerToolMode,
        TokenCount,
        TokenCountMode,
        WebSearchOptions,
    )
except ImportError:  # pragma: no cover - compatibility with SDK v4 checkouts.
    BatchHandle = BatchMode = BatchRequest = BatchResult = BatchStatus = None
    CodeExecOptions = FileRef = FilesMode = PromptCacheMode = None
    RawResponse = ReasoningControlMode = RequestOptions = ServerToolMode = None
    TokenCount = TokenCountMode = WebSearchOptions = None
from .gpt5_overlay import prepend_gpt5_overlay
from .model_metadata import ModelInfo, ModelCategory
from .retry import with_retry

logger = logging.getLogger(__name__)

_split_thinking_from_content = split_thinking_from_content
_ThinkingContentSplitter = ThinkingContentSplitter


def _capability_kwargs(**kwargs: Any) -> Dict[str, Any]:
    fields = getattr(ProviderCapabilities, "__dataclass_fields__", {})
    if not fields:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in fields}


def _enum_value(enum_type: Any, name: str, fallback: str) -> Any:
    if enum_type is None:
        return fallback
    return getattr(enum_type, name)


def _object_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return {k: v for k, v in asdict(obj).items() if v is not None}
    data: Dict[str, Any] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except Exception:
            continue
        if callable(value) or value is None:
            continue
        data[key] = value
    return data


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for interacting with the OpenAI API.

    Supports:
    - Chat completions with tool calling
    - Streaming responses
    - JSON mode
    """

    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    DEFAULT_EMBEDDING_DIM = 1536

    def __init__(
        self,
        name: str = "openai",
        *,
        supports_embeddings: Optional[bool] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        native_openai: bool = False,
    ):
        self.name = name
        if supports_embeddings is None:
            supports_embeddings = name == "openai"
        if supports_embeddings and embedding_model is None:
            embedding_model = self.DEFAULT_EMBEDDING_MODEL
        if supports_embeddings and embedding_dim is None:
            embedding_dim = self.DEFAULT_EMBEDDING_DIM
        self._supports_embeddings = supports_embeddings
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        # Canonical OpenAI (real openai vendor on the official base_url) is the
        # only deployment that exposes /batches, /files, /responses. This is set
        # explicitly by the registry — NOT inferred from embedding support,
        # since an OpenAI-compatible route can enable embeddings too. Defaults
        # False so any other construction path is treated as compatible-only.
        self._native_openai = bool(native_openai)

    def provider_capabilities(self) -> ProviderCapabilities:
        # OpenAI-native endpoints — /batches, /files, /responses and the
        # prompt_cache_key param — are exposed only by canonical OpenAI, not by
        # the OpenAI-*compatible* routes this adapter also serves via a custom
        # base_url (Kimi, DeepSeek, OpenRouter, …). Gate those on the same
        # native-OpenAI signal used for embeddings so the framework never routes
        # batch/file/raw operations to a compatible endpoint that would 404.
        # Token counting (local tiktoken estimate) and reasoning effort (a
        # request param) are endpoint-agnostic and stay ungated.
        native = self._native_openai
        return ProviderCapabilities(**_capability_kwargs(
            supports_tools=True,
            supports_streaming=True,
            supports_vision=True,
            supports_structured_output=True,
            supports_embeddings=self._supports_embeddings,
            supports_token_counting=True,
            supports_reasoning_control=True,
            supports_batch=native,
            supports_files=native,
            supports_prompt_cache=native,
            supports_raw_passthrough=native,
            structured_output_mode=StructuredOutputMode.JSON_SCHEMA,
            tool_streaming_mode=ToolStreamingMode.NATIVE_DELTA,
            vision_input_mode=VisionInputMode.OPENAI_IMAGE_URL,
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
            reasoning_control_mode=_enum_value(
                ReasoningControlMode, "EFFORT", "effort"
            ),
            prompt_cache_mode=_enum_value(
                PromptCacheMode, "AUTOMATIC" if native else "NONE",
                "automatic" if native else "none",
            ),
            batch_mode=_enum_value(
                BatchMode, "FILE_BASED" if native else "NONE",
                "file_based" if native else "none",
            ),
            files_mode=_enum_value(
                FilesMode, "UPLOAD" if native else "NONE",
                "upload" if native else "none",
            ),
            token_count_mode=_enum_value(TokenCountMode, "ESTIMATE", "estimate"),
            reasoning_effort_levels=("minimal", "low", "medium", "high"),
            raw_operations=(
                (
                    "chat.completions.create",
                    "responses.create",
                    "responses.retrieve",
                    "files.create",
                    "files.list",
                    "files.retrieve",
                    "files.delete",
                    "files.content",
                    "batches.create",
                    "batches.retrieve",
                    "batches.cancel",
                )
                if native
                else ()
            ),
            model_dependent=("vision",),
            notes=(
                "Structured output uses response_format=json_schema for Pydantic models.",
                "OpenAI prompt caching is implicit for stable prefixes.",
                "Token counting is estimated locally with tiktoken.",
            ),
        ))

    def contract_features(self) -> frozenset[str]:
        features = {"token_counting", "reasoning_control"}
        if self._native_openai:
            features |= {"batch", "files", "prompt_cache", "raw_passthrough"}
        return frozenset(features)

    async def probe_reachable(
        self,
        client: Any,
        *,
        base_url: Optional[str] = None,
        timeout: float = 1.5,
    ) -> Optional[bool]:
        """Probe an OpenAI-compatible local route's models endpoint."""
        root = base_url
        if not root:
            root = str(getattr(client, "base_url", "") or "") or None
        if not root:
            return None

        root = str(root).rstrip("/")
        url = f"{root}/models" if root.endswith("/v1") else f"{root}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                response = await http.get(url)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def aembed(
        self,
        client: openai.AsyncOpenAI,
        text: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[List[float]]:
        response = await with_retry(
            client.embeddings.create,
            model=model or self.DEFAULT_EMBEDDING_MODEL,
            input=text,
        )
        data = getattr(response, "data", None) or []
        if not data:
            return None
        embedding = getattr(data[0], "embedding", None)
        return list(embedding) if embedding is not None else None

    async def aembed_batch(
        self,
        client: openai.AsyncOpenAI,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Optional[List[float]]]:
        if not texts:
            return []
        response = await with_retry(
            client.embeddings.create,
            model=model or self.DEFAULT_EMBEDDING_MODEL,
            input=texts,
        )
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        for item in getattr(response, "data", None) or []:
            index = getattr(item, "index", None)
            embedding = getattr(item, "embedding", None)
            if isinstance(index, int) and 0 <= index < len(embeddings) and embedding is not None:
                embeddings[index] = list(embedding)
        return embeddings

    def contribute_system_prompt(
        self, model_id: str, base: Optional[str]
    ) -> Optional[str]:
        """Inject the GPT-5 behavior contract for gpt-5 family models.

        See #807 / #806. Same overlay as ``CodexAdapter`` because GPT-5 needs
        the discipline contracts regardless of which OAuth/API path the model
        is reached through.
        """
        return prepend_gpt5_overlay(base, model_id)

    async def get_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Gets a response from the OpenAI model.

        Args:
            client: OpenAI async client
            model: Model name to use
            messages: Chat messages
            format: Response format (e.g., "json") - DEPRECATED, use response_format
            tools: Optional list of tools in OpenAI function calling format
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            LLMResponse with content and/or tool calls
        """
        try:
            # Apply provider/model-specific system-prompt contributions (e.g. the
            # GPT-5 behavior contract). #807 / #806.
            messages = self._apply_system_prompt_contribution(messages, model)
            # Normalize messages - OpenAI requires tool_calls arguments to be JSON strings
            normalized_messages = self._normalize_messages(messages)

            extra_kwargs = {}

            # Handle structured output via Pydantic model
            if response_format is not None and issubclass(response_format, BaseModel):
                # OpenAI structured output format with JSON schema
                # OpenAI requires additionalProperties: false for strict mode
                schema = response_format.model_json_schema()
                schema["additionalProperties"] = False
                extra_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": schema,
                        "strict": True
                    }
                }
            elif format == "json":
                extra_kwargs["response_format"] = {"type": "json_object"}

            if tools:
                extra_kwargs["tools"] = tools
                # Let the model decide whether to use tools or respond directly
                extra_kwargs["tool_choice"] = "auto"

            # Pass through additional kwargs (max_tokens, temperature, etc.)
            # Note: newer models (gpt-5.x, o1, o3, etc.) use max_completion_tokens instead of max_tokens
            if "max_tokens" in kwargs:
                extra_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                if key in kwargs:
                    extra_kwargs[key] = kwargs[key]

            # Provider-specific body extensions (e.g. llama.cpp's `cache_prompt`,
            # which tells llama-server to be aggressive about retaining the
            # slot's KV cache for prefix matching).  The service layer decides
            # when to set this and passes it through via `extra_body` kwarg.
            # See issue #704.
            if "extra_body" in kwargs and kwargs["extra_body"]:
                extra_kwargs["extra_body"] = kwargs["extra_body"]

            request_options = kwargs.get("request_options")
            if request_options is not None:
                extra_kwargs = self.apply_request_options(
                    extra_kwargs,
                    request_options,
                    model=model,
                )

            response = await with_retry(
                client.chat.completions.create,
                model=model,
                messages=normalized_messages,
                **extra_kwargs
            )

            message = response.choices[0].message

            # Parse tool calls if present
            # Use getattr to safely access tool_calls (may not exist on mock objects)
            parsed_tool_calls = None
            message_tool_calls = getattr(message, 'tool_calls', None)
            if message_tool_calls:
                parsed_tool_calls = []
                for tc in message_tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        # SDK 0.7.0 malformed-JSON sentinel; same
                        # convention as the streaming path.
                        args = {"_raw": tc.function.arguments}

                    parsed_tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args
                    ))

            # Extract token usage from response
            input_tokens = None
            output_tokens = None
            total_tokens = None
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'prompt_tokens', None)
                output_tokens = getattr(response.usage, 'completion_tokens', None)
                total_tokens = getattr(response.usage, 'total_tokens', None)

            _, content = _split_thinking_from_content(
                message.content,
                getattr(message, 'reasoning_content', None),
            )

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI adapter failed: {e}", exc_info=True)
            raise

    async def get_streaming_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, ThinkingDelta]]:
        """
        Gets a streaming response from the OpenAI model.

        Note: Streaming with tools is complex as tool calls arrive in chunks.
        This implementation yields text content only. For tool calls during
        streaming, use the non-streaming get_response method.

        Args:
            client: OpenAI async client
            model: Model name to use
            messages: Chat messages
            tools: Optional tools
            response_format: Optional Pydantic model (streaming will yield JSON chunks)
            **kwargs: Additional parameters

        Yields:
            Text chunks as they arrive
        """
        try:
            # Apply provider/model-specific system-prompt contributions (e.g. the
            # GPT-5 behavior contract). #807 / #806.
            messages = self._apply_system_prompt_contribution(messages, model)
            # Normalize messages for OpenAI compatibility
            normalized_messages = self._normalize_messages(messages)

            extra_kwargs = {}
            if tools:
                extra_kwargs["tools"] = tools
                extra_kwargs["tool_choice"] = "auto"

            # Handle structured output in streaming mode
            if response_format is not None and issubclass(response_format, BaseModel):
                # OpenAI requires additionalProperties: false for strict mode
                schema = response_format.model_json_schema()
                schema["additionalProperties"] = False
                extra_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": schema,
                        "strict": True
                    }
                }

            # Pass through additional kwargs
            if "max_tokens" in kwargs:
                extra_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                if key in kwargs:
                    extra_kwargs[key] = kwargs[key]

            # Provider-specific body extensions (issue #704).  See get_response().
            if "extra_body" in kwargs and kwargs["extra_body"]:
                extra_kwargs["extra_body"] = kwargs["extra_body"]

            request_options = kwargs.get("request_options")
            if request_options is not None:
                extra_kwargs = self.apply_request_options(
                    extra_kwargs,
                    request_options,
                    model=model,
                )

            logger.info(f"Starting OpenAI stream for model: {model}")
            stream = await with_retry(
                client.chat.completions.create,
                model=model,
                messages=normalized_messages,
                stream=True,
                **extra_kwargs
            )

            splitter = _ThinkingContentSplitter(provider=self.name)
            chunk_count = 0
            async for chunk in stream:
                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning_content, str) and reasoning_content:
                    yield ThinkingDelta(reasoning_content, provider=self.name)
                content_delta = getattr(delta, "content", None)
                if isinstance(content_delta, str) and content_delta:
                    for item in splitter.feed(content_delta):
                        if isinstance(item, str):
                            chunk_count += 1
                        yield item

            for item in splitter.flush():
                if isinstance(item, str):
                    chunk_count += 1
                yield item

            logger.info(f"Stream completed. Total chunks: {chunk_count}")

        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded during streaming: {e}")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error during streaming: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error during streaming: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}", exc_info=True)
            raise

    async def get_streaming_response_with_tools(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, ThinkingDelta, LLMResponse]]:
        """
        Stream response with tool call detection.

        This method streams text content as it arrives AND detects tool calls
        from the streaming chunks. Tool calls arrive as deltas with indices
        that need to be accumulated and assembled.

        Args:
            client: OpenAI async client
            model: Model name to use
            messages: Chat messages
            tools: Optional tools in OpenAI function calling format
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Yields:
            str: Text content chunks as they arrive
            LLMResponse: Terminal response at end-of-stream carrying token usage (and tool_calls when present)

        Example:
            async for item in adapter.get_streaming_response_with_tools(...):
                if isinstance(item, str):
                    print(item, end='', flush=True)  # Stream text to user
                elif isinstance(item, LLMResponse):
                    if item.has_tool_calls:
                        for tc in item.tool_calls:
                            result = execute_tool(tc)
                            # Continue conversation with tool results
        """
        try:
            # Apply provider/model-specific system-prompt contributions (e.g. the
            # GPT-5 behavior contract). #807 / #806.
            messages = self._apply_system_prompt_contribution(messages, model)
            # Normalize messages for OpenAI compatibility
            normalized_messages = self._normalize_messages(messages)

            extra_kwargs = {}
            if tools:
                extra_kwargs["tools"] = tools
                extra_kwargs["tool_choice"] = "auto"

            # Handle structured output in streaming mode
            if response_format is not None and issubclass(response_format, BaseModel):
                schema = response_format.model_json_schema()
                schema["additionalProperties"] = False
                extra_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": schema,
                        "strict": True
                    }
                }

            # Pass through additional kwargs
            if "max_tokens" in kwargs:
                extra_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                if key in kwargs:
                    extra_kwargs[key] = kwargs[key]

            # Provider-specific body extensions (issue #704).  See get_response().
            if "extra_body" in kwargs and kwargs["extra_body"]:
                extra_kwargs["extra_body"] = kwargs["extra_body"]

            request_options = kwargs.get("request_options")
            if request_options is not None:
                extra_kwargs = self.apply_request_options(
                    extra_kwargs,
                    request_options,
                    model=model,
                )

            # Request streaming with usage stats
            extra_kwargs["stream_options"] = {"include_usage": True}

            logger.info(f"Starting OpenAI stream with tools for model: {model}")
            stream = await with_retry(
                client.chat.completions.create,
                model=model,
                messages=normalized_messages,
                stream=True,
                **extra_kwargs
            )

            # Accumulator for tool calls - keyed by index
            # Each tool call arrives in chunks with the same index
            tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
            splitter = _ThinkingContentSplitter(provider=self.name)
            text_content = ""
            reasoning_content = ""
            chunk_count = 0
            input_tokens = None
            output_tokens = None
            total_tokens = None
            call_cost = None

            async for chunk in stream:
                # Extract usage from final chunk (OpenAI sends it with stream_options)
                if hasattr(chunk, 'usage') and chunk.usage:
                    input_tokens = getattr(chunk.usage, 'prompt_tokens', None)
                    output_tokens = getattr(chunk.usage, 'completion_tokens', None)
                    total_tokens = getattr(chunk.usage, 'total_tokens', None)
                    # OpenRouter reports exact per-call cost on the usage-only
                    # final chunk when ``usage: {include: true}`` was sent
                    # (kestrel #1806). Stash it for the terminal response.
                    _cost = getattr(chunk.usage, 'cost', None)
                    if _cost is None:
                        _extra = getattr(chunk.usage, 'model_extra', None)
                        if isinstance(_extra, dict):
                            _cost = _extra.get('cost')
                    if _cost is not None:
                        try:
                            call_cost = float(_cost)
                        except (TypeError, ValueError):
                            call_cost = None

                # Skip chunks with no choices (like the final usage-only chunk)
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle text content - yield immediately for real-time streaming
                delta_reasoning_content = getattr(delta, "reasoning_content", None)
                if isinstance(delta_reasoning_content, str) and delta_reasoning_content:
                    reasoning_content += delta_reasoning_content
                    yield ThinkingDelta(delta_reasoning_content, provider=self.name)

                content_delta = getattr(delta, "content", None)
                if isinstance(content_delta, str) and content_delta:
                    for item in splitter.feed(content_delta):
                        if isinstance(item, str):
                            chunk_count += 1
                            text_content += item
                        yield item

                # Handle tool call deltas - accumulate by index
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index

                        # Initialize accumulator for this tool call if
                        # needed AND emit the SDK 0.7.0 ToolCallStarted
                        # marker — exactly once per distinct index, on
                        # the first delta for that index. OpenAI's
                        # first delta typically carries id and name (in
                        # the same fragment that introduces the index),
                        # but the contract permits None for either when
                        # the provider stream hasn't surfaced them yet.
                        # We capture whatever's on this delta and emit
                        # ``None`` for any field that arrives empty —
                        # the final LLMResponse is the source of truth
                        # for the assembled call.
                        is_new_call = idx not in tool_calls_accumulator
                        if is_new_call:
                            tool_calls_accumulator[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": ""
                            }

                        # Accumulate id (usually comes in first chunk)
                        if tc_delta.id:
                            tool_calls_accumulator[idx]["id"] += tc_delta.id

                        # Accumulate function name (usually comes in first chunk)
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_accumulator[idx]["name"] += tc_delta.function.name
                            # Accumulate arguments (comes in multiple chunks)
                            if tc_delta.function.arguments:
                                tool_calls_accumulator[idx]["arguments"] += tc_delta.function.arguments

                        if is_new_call:
                            # Emit ToolCallStarted only after we've
                            # absorbed the first delta's id/name so the
                            # marker carries them when present rather
                            # than always being ``(None, None)``.
                            current = tool_calls_accumulator[idx]
                            yield ToolCallStarted(
                                index=idx,
                                id=current["id"] or None,
                                name=current["name"] or None,
                            )

            logger.info(f"Stream completed. Text chunks: {chunk_count}, Tool calls: {len(tool_calls_accumulator)}")

            for item in splitter.flush():
                if isinstance(item, str):
                    chunk_count += 1
                    text_content += item
                yield item

            # Assemble any tool calls collected during the stream.
            parsed_tool_calls = None
            if tool_calls_accumulator:
                parsed_tool_calls = []
                for idx in sorted(tool_calls_accumulator.keys()):
                    tc_data = tool_calls_accumulator[idx]
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        # SDK 0.7.0 malformed-JSON fallback — see
                        # contract docstring on
                        # LLMAdapter.get_streaming_response_with_tools.
                        args = {"_raw": tc_data["arguments"]}

                    parsed_tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args
                    ))
                parsed_tool_calls = parsed_tool_calls or None

            # Always emit a terminal LLMResponse carrying token usage, even for
            # text-only turns. Previously this was nested under `if
            # tool_calls_accumulator`, so text-only streams — the common case —
            # dropped their usage entirely, a silent billing undercount. The
            # service layer meters streamed turns from this terminal response;
            # consumers read it only for tool_calls / usage (visible content was
            # already streamed as chunks). Mirrors the anthropic adapter fix
            # (#1686/#1684). OpenRouter inherits this via super() delegation.
            _raw = {}
            if reasoning_content:
                _raw["reasoning_content"] = reasoning_content
            if call_cost is not None:
                _raw["cost"] = call_cost
            yield LLMResponse(
                content=text_content if text_content else None,
                tool_calls=parsed_tool_calls,
                raw=_raw or None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded during streaming with tools: {e}")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error during streaming with tools: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error during streaming with tools: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI streaming with tools failed: {e}", exc_info=True)
            raise

    async def continue_with_tool_results(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Continue a conversation after executing tool calls.

        Args:
            client: OpenAI async client
            model: Model name
            messages: Original messages (including assistant's tool call message)
            tool_results: List of tool results in format:
                [{"tool_call_id": str, "content": str}, ...]
            tools: Optional tools to include for multi-turn

        Returns:
            LLMResponse with the model's follow-up
        """
        # Append tool results as tool messages
        extended_messages = messages.copy()
        for result in tool_results:
            extended_messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["content"]
            })

        return await self.get_response(
            client=client,
            model=model,
            messages=extended_messages,
            tools=tools
        )

    async def count_tokens(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        total = self._estimate_message_tokens(model, messages)
        return TokenCount(input_tokens=total, total_tokens=total) if TokenCount else {
            "input_tokens": total,
            "total_tokens": total,
        }

    async def batch_submit(
        self,
        client: Any,
        requests: List[Any],
        **kwargs: Any,
    ) -> Any:
        lines = []
        for request in requests:
            body = self._batch_request_body(request)
            lines.append(
                json.dumps(
                    {
                        "custom_id": getattr(request, "custom_id", ""),
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    },
                    separators=(",", ":"),
                )
            )
        filename = kwargs.get("filename", "kestrel-openai-batch.jsonl")
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        uploaded = await self.file_upload(
            client,
            (filename, payload),
            purpose="batch",
        )
        batch_kwargs = {
            "input_file_id": uploaded.id,
            "endpoint": "/v1/chat/completions",
            "completion_window": kwargs.get("completion_window", "24h"),
        }
        if kwargs.get("metadata") is not None:
            batch_kwargs["metadata"] = kwargs["metadata"]
        batch = await with_retry(client.batches.create, **batch_kwargs)
        return self._batch_handle(batch)

    async def batch_poll(self, client: Any, handle: Any, **kwargs: Any) -> Any:
        batch = await with_retry(client.batches.retrieve, getattr(handle, "id", handle))
        return self._batch_handle(batch)

    async def batch_results(self, client: Any, handle: Any, **kwargs: Any) -> List[Any]:
        raw_handle = getattr(handle, "raw", None) or handle
        output_file_id = (
            getattr(raw_handle, "output_file_id", None)
            or getattr(handle, "output_file_id", None)
            or kwargs.get("output_file_id")
        )
        if not output_file_id:
            refreshed = await self.batch_poll(client, handle)
            raw_handle = getattr(refreshed, "raw", None) or refreshed
            output_file_id = getattr(raw_handle, "output_file_id", None)
        if not output_file_id:
            return []
        content_response = await with_retry(client.files.content, output_file_id)
        content = await self._read_file_content(content_response)
        results = []
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id", "")
            error = item.get("error")
            response_body = ((item.get("response") or {}).get("body") or None)
            response = (
                self._llm_response_from_chat_completion(response_body)
                if response_body
                else None
            )
            error_text = json.dumps(error) if error is not None else None
            if BatchResult:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        response=response,
                        error=error_text,
                        raw=item,
                    )
                )
            else:
                results.append(
                    {
                        "custom_id": custom_id,
                        "response": response,
                        "error": error_text,
                        "raw": item,
                    }
                )
        return results

    async def batch_cancel(self, client: Any, handle: Any, **kwargs: Any) -> Any:
        batch = await with_retry(client.batches.cancel, getattr(handle, "id", handle))
        return self._batch_handle(batch)

    async def file_upload(
        self,
        client: Any,
        file: Any,
        *,
        purpose: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        uploaded = await with_retry(
            client.files.create,
            file=file,
            purpose=purpose or kwargs.get("purpose") or "assistants",
        )
        return self._file_ref(uploaded)

    async def file_list(self, client: Any, **kwargs: Any) -> List[Any]:
        response = await with_retry(client.files.list, **kwargs)
        return [self._file_ref(item) for item in getattr(response, "data", [])]

    async def file_get(self, client: Any, file_id: str, **kwargs: Any) -> Any:
        response = await with_retry(client.files.retrieve, file_id)
        return self._file_ref(response)

    async def file_delete(self, client: Any, file_id: str, **kwargs: Any) -> bool:
        response = await with_retry(client.files.delete, file_id)
        return bool(getattr(response, "deleted", False) or getattr(response, "id", None))

    def file_reference(self, file_ref: Any) -> Dict[str, Any]:
        file_id = getattr(file_ref, "id", None)
        if not file_id:
            raise ValueError("file_ref.id is required")
        return {"file_id": file_id}

    def apply_request_options(
        self,
        request_kwargs: Dict[str, Any],
        options: Any,
        *,
        model: str,
    ) -> Dict[str, Any]:
        out = request_kwargs
        # OpenAI chat-completions accepts ``reasoning_effort`` directly; there
        # is no top-level ``reasoning`` argument (that is a Responses-API field).
        if getattr(options, "reasoning_effort", None):
            out["reasoning_effort"] = options.reasoning_effort

        if getattr(options, "cache_markers", None):
            # chat.completions supports prompt_cache_key (a stable string that
            # pins the cache prefix); it has no `cache_markers` field, so derive
            # the key from the markers and send only that.
            extra_body = dict(out.get("extra_body") or {})
            extra_body["prompt_cache_key"] = self._cache_key_for_options(
                model, options.cache_markers
            )
            out["extra_body"] = extra_body

        # NOTE: web_search / code_execution are intentionally NOT translated
        # here. They are Responses/Assistants-API server tools — the chat
        # completions endpoint rejects ``web_search_preview`` / ``code_interpreter``
        # tool entries. This adapter therefore does not advertise those
        # capabilities (see provider_capabilities). They can be added once this
        # adapter routes through the Responses API.

        raw = getattr(options, "raw", None)
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key == "extra_body" and isinstance(value, dict):
                    out.setdefault("extra_body", {}).update(value)
                else:
                    out[key] = value
        return out

    async def raw_request(
        self,
        client: Any,
        operation: str,
        payload: Optional[Any] = None,
        *,
        http_method: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        target: Any = client
        for part in operation.split("."):
            target = getattr(target, part)
        call_kwargs = dict(payload or {}) if isinstance(payload, dict) else {}
        call_kwargs.update(kwargs)
        if payload is not None and not isinstance(payload, dict):
            data = await _maybe_await(target(payload, **kwargs))
        else:
            data = await _maybe_await(target(**call_kwargs))
        if RawResponse:
            return RawResponse(operation=operation, data=data, raw=data)
        return {"operation": operation, "data": data, "raw": data}

    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        """List available models from OpenAI API (or any OpenAI-compatible
        endpoint the route was initialized against).

        Uses the framework-initialized ``client`` so the call goes to the
        same ``base_url`` and authenticates with the same key the route's
        ``get_response`` calls do. Routes pointed at custom endpoints
        (Azure, Kimi, DeepSeek, OpenRouter-via-OpenAI-compat) get their
        own catalog rather than silently falling through to api.openai.com.

        Args:
            client: The route's ``openai.AsyncOpenAI`` client. The
                framework always passes this; the env-var fallback below
                is only for legacy callers (existing tests, scripts that
                build a bare ``OpenAIAdapter()`` and call ``list_models()``
                directly) that pass ``None`` or omit the argument.
                Liskov-widened to optional from the SDK 0.5.0 abstract
                signature so those callers keep working.

        Returns:
            List of ModelInfo objects.
        """
        try:
            if client is None:
                # Legacy fallback. Modern callers always pass the route
                # client; warn and let env vars rebuild a canonical-OpenAI
                # client so we don't lose discovery for callers that
                # haven't migrated yet.
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.warning(
                        "OpenAIAdapter.list_models called with client=None "
                        "and no OPENAI_API_KEY; returning empty model list"
                    )
                    return []
                logger.warning(
                    "OpenAIAdapter.list_models called with client=None — "
                    "rebuilding from OPENAI_API_KEY (canonical OpenAI only)"
                )
                client = openai.AsyncOpenAI(api_key=api_key)

            response = await client.models.list()

            models = []
            for model in response.data:
                # Generate a display name from the model ID
                display_name = model.id.replace("-", " ").title()

                models.append(ModelInfo(
                    id=model.id,
                    provider="openai",
                    display_name=display_name,
                    category=ModelCategory.CHAT,  # Will be enriched by catalog service
                    supports_tools=True,  # OpenAI models support tools
                    supports_streaming=True,  # OpenAI streams every chat model
                    created_at=str(model.created) if hasattr(model, 'created') else None,
                ))

            logger.info(f"OpenAI returned {len(models)} models")
            return models

        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded while listing models: {e}")
            return []
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error while listing models: {e}")
            return []
        except openai.APIError as e:
            logger.error(f"OpenAI API error while listing models: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}", exc_info=True)
            return []

    @staticmethod
    def _cache_key_for_options(model: str, markers: List[Any]) -> str:
        payload = json.dumps(
            [getattr(marker, "label", None) or getattr(marker, "index", None) for marker in markers],
            sort_keys=True,
            default=str,
        )
        return f"kestrel:{model}:{payload}"

    @staticmethod
    def _estimate_message_tokens(model: str, messages: List[Dict[str, Any]]) -> int:
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("o200k_base")
            total = 0
            for message in messages:
                total += 3
                for key, value in message.items():
                    total += len(encoding.encode(OpenAIAdapter._token_text(value)))
                    if key == "name":
                        total += 1
            return total + 3
        except Exception:
            text = "\n".join(OpenAIAdapter._token_text(message) for message in messages)
            return max(1, len(text) // 4)

    @staticmethod
    def _token_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(OpenAIAdapter._token_text(item) for item in value)
        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return value["text"]
            return json.dumps(value, sort_keys=True, default=str)
        return str(value)

    def _batch_request_body(self, request: Any) -> Dict[str, Any]:
        model = getattr(request, "model", None) or "gpt-5-mini"
        # Mirror get_response: apply the model's system-prompt contribution
        # (e.g. the GPT-5 behavior overlay) before normalizing, so batched
        # calls don't silently lose it.
        messages = self._apply_system_prompt_contribution(
            getattr(request, "messages", []) or [], model
        )
        body = {
            "model": model,
            "messages": self._normalize_messages(messages),
        }
        if getattr(request, "tools", None):
            body["tools"] = getattr(request, "tools")
            body["tool_choice"] = "auto"
        if getattr(request, "format", None) == "json":
            body["response_format"] = {"type": "json_object"}
        body.update(getattr(request, "kwargs", None) or {})
        request_options = getattr(request, "request_options", None)
        if request_options is not None:
            body = self.apply_request_options(
                body,
                request_options,
                model=body["model"],
            )
        return body

    @staticmethod
    def _batch_handle(batch: Any) -> Any:
        status_map = {
            "validating": "running",
            "in_progress": "running",
            "finalizing": "running",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
            "cancelling": "cancelled",
            "expired": "expired",
        }
        status_value = status_map.get(str(getattr(batch, "status", "") or ""), "unknown")
        status = (
            getattr(BatchStatus, status_value.upper(), status_value)
            if BatchStatus is not None
            else status_value
        )
        if BatchHandle:
            return BatchHandle(
                id=getattr(batch, "id", ""),
                status=status,
                created_at=getattr(batch, "created_at", None),
                expires_at=getattr(batch, "expires_at", None),
                raw=batch,
            )
        return {"id": getattr(batch, "id", ""), "status": status, "raw": batch}

    @staticmethod
    def _file_ref(file_obj: Any) -> Any:
        if FileRef:
            return FileRef(
                id=getattr(file_obj, "id", ""),
                filename=getattr(file_obj, "filename", None),
                purpose=getattr(file_obj, "purpose", None),
                size_bytes=getattr(file_obj, "bytes", None),
                created_at=getattr(file_obj, "created_at", None),
                raw=file_obj,
            )
        return {
            "id": getattr(file_obj, "id", ""),
            "filename": getattr(file_obj, "filename", None),
            "purpose": getattr(file_obj, "purpose", None),
            "raw": file_obj,
        }

    @staticmethod
    async def _read_file_content(content_response: Any) -> bytes:
        if isinstance(content_response, bytes):
            return content_response
        if isinstance(content_response, str):
            return content_response.encode("utf-8")
        if hasattr(content_response, "read"):
            return await _maybe_await(content_response.read())
        if hasattr(content_response, "content"):
            content = content_response.content
            return content if isinstance(content, bytes) else str(content).encode("utf-8")
        return str(content_response).encode("utf-8")

    def _llm_response_from_chat_completion(self, response_body: Any) -> LLMResponse:
        body = response_body if isinstance(response_body, dict) else _object_to_dict(response_body)
        choices = body.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    parsed = {"_raw": args}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=parsed,
                    )
                )
        usage = body.get("usage") or {}
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            raw=response_body,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize messages for OpenAI API compatibility.
        
        OpenAI requires tool_calls arguments to be JSON strings, not dicts.
        This method converts any dict arguments to JSON strings.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Normalized messages with string arguments in tool_calls
        """
        normalized = []
        for msg in messages:
            if "tool_calls" in msg and msg["tool_calls"]:
                # Deep copy the message to avoid modifying original
                new_msg = {**msg}
                new_tool_calls = []
                for tc in msg["tool_calls"]:
                    new_tc = {**tc}
                    if "function" in tc and "arguments" in tc["function"]:
                        args = tc["function"]["arguments"]
                        if isinstance(args, dict):
                            new_tc["function"] = {
                                **tc["function"],
                                "arguments": json.dumps(args)
                            }
                    new_tool_calls.append(new_tc)
                new_msg["tool_calls"] = new_tool_calls
                normalized.append(new_msg)
            else:
                normalized.append(msg)
        return normalized

    # ---- Provider metadata (SDK 0.6.0) -------------------------------------

    def substrate_type(self) -> Optional[str]:
        return "gpt"

    def display_name(self) -> Optional[str]:
        return "OpenAI"

    def key_env_var(self) -> Optional[str]:
        return "OPENAI_API_KEY"

    # cost_per_1m_tokens: omitted — OpenAI pricing varies dramatically by
    # model (gpt-5-mini vs o3 vs gpt-3.5-turbo), so per-model pricing
    # belongs on ModelInfo rather than the adapter. Council costing falls
    # back to the framework's conservative default when this returns None.

    # deliberation_style: omitted — OpenAI spans the speed/cost spectrum
    # (gpt-5-mini fast, o3 careful), so a single-adapter hint isn't
    # meaningful. Council leaves this routing decision to model-level
    # heuristics.
