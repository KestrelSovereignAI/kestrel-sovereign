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
import re
import openai
import logging
from typing import Any, Dict, List, Optional, AsyncIterator, Type, Union

from pydantic import BaseModel

from .adapter import LLMAdapter, LLMResponse, ToolCall
from .gpt5_overlay import prepend_gpt5_overlay
from .model_metadata import ModelInfo, ModelCategory
from .retry import with_retry

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for interacting with the OpenAI API.

    Supports:
    - Chat completions with tool calling
    - Streaming responses
    - JSON mode
    """

    def __init__(self):
        self.name = "openai"

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
                        args = {"raw": tc.function.arguments}

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

            # Strip chain-of-thought reasoning from response
            content = message.content

            # 1. Use reasoning_content if server separated it (--reasoning-format deepseek)
            reasoning_content = getattr(message, 'reasoning_content', None)
            if reasoning_content and content:
                # Server already stripped thinking — content is clean
                pass
            elif content:
                # 2. Strip <think>...</think> tags (DeepSeek-R1 format)
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)

                # 3. Strip inline chain-of-thought from models that dump reasoning
                #    as plain text (Kimi K2.5). Reasoning paragraphs start with
                #    meta-cognitive phrases and don't contain markdown formatting.
                paragraphs = content.split('\n\n')
                if len(paragraphs) >= 2:
                    reasoning_markers = [
                        'The user', 'I should', 'I need to', 'Let me',
                        'I\'ll ', 'I will ', 'This is a', 'Since ',
                        'Looking at', 'Based on', 'I can see',
                        'Overall,', 'Now I', 'I have',
                    ]
                    cleaned = []
                    for p in paragraphs:
                        stripped = p.strip()
                        # Skip paragraphs that look like reasoning (start with
                        # marker AND don't contain markdown formatting like
                        # headers, tables, lists, or bold text)
                        is_reasoning = (
                            any(stripped.startswith(m) for m in reasoning_markers)
                            and not any(c in stripped for c in ['#', '|', '- ', '* ', '**'])
                        )
                        if not is_reasoning:
                            cleaned.append(p)
                    if cleaned:
                        content = '\n\n'.join(cleaned).strip()

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
    ) -> AsyncIterator[str]:
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

            logger.info(f"Starting OpenAI stream for model: {model}")
            stream = await with_retry(
                client.chat.completions.create,
                model=model,
                messages=normalized_messages,
                stream=True,
                **extra_kwargs
            )

            chunk_count = 0
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunk_count += 1
                    yield delta.content

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
    ) -> AsyncIterator[Union[str, LLMResponse]]:
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
            LLMResponse: Final response with tool_calls (only at end if tools were called)

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
            text_content = ""
            chunk_count = 0
            input_tokens = None
            output_tokens = None
            total_tokens = None

            async for chunk in stream:
                # Extract usage from final chunk (OpenAI sends it with stream_options)
                if hasattr(chunk, 'usage') and chunk.usage:
                    input_tokens = getattr(chunk.usage, 'prompt_tokens', None)
                    output_tokens = getattr(chunk.usage, 'completion_tokens', None)
                    total_tokens = getattr(chunk.usage, 'total_tokens', None)

                # Skip chunks with no choices (like the final usage-only chunk)
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle text content - yield immediately for real-time streaming
                if delta.content:
                    chunk_count += 1
                    text_content += delta.content
                    yield delta.content

                # Handle tool call deltas - accumulate by index
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index

                        # Initialize accumulator for this tool call if needed
                        if idx not in tool_calls_accumulator:
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

            logger.info(f"Stream completed. Text chunks: {chunk_count}, Tool calls: {len(tool_calls_accumulator)}")

            # If we have tool calls, yield a final LLMResponse with assembled tool calls
            if tool_calls_accumulator:
                parsed_tool_calls = []
                for idx in sorted(tool_calls_accumulator.keys()):
                    tc_data = tool_calls_accumulator[idx]
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {"raw": tc_data["arguments"]}

                    parsed_tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args
                    ))

                yield LLMResponse(
                    content=text_content if text_content else None,
                    tool_calls=parsed_tool_calls,
                    raw=None,  # No raw response for streaming
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

    async def list_models(self) -> List[ModelInfo]:
        """
        List available models from OpenAI API.

        Calls client.models.list() to get all available models.

        Returns:
            List of ModelInfo objects
        """
        try:
            # Create a client for listing models
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set, returning empty model list")
                return []

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