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

from pydantic import BaseModel

from .adapter import LLMAdapter, LLMResponse, ToolCall
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


class OllamaAdapter(LLMAdapter):
    """
    Adapter for interacting with a local Ollama instance.

    Supports:
    - Chat completions with tool calling (Ollama 0.4+)
    - Streaming responses
    - Vision models with base64 images
    """

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

            # If using structured output, validate against the Pydantic model
            if response_format is not None and content:
                try:
                    # Content should be JSON - validate it
                    validated = response_format.model_validate_json(content)
                    # Return as JSON string for consistency with other adapters
                    content = validated.model_dump_json()
                except Exception as e:
                    logger.warning(f"Ollama structured output validation failed: {e}")
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
                            args = {"raw": args}

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

        except Exception as e:
            logger.error(f"Ollama adapter failed: {e}")
            raise

    async def get_streaming_response(
        self,
        client: "ollama.AsyncClient",
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Gets a streaming response from Ollama.

        Note: Tool calling during streaming is not well-supported by Ollama.
        For tool calls, use the non-streaming get_response method.

        Args:
            client: Ollama async client
            model: Model name
            messages: Chat messages
            tools: Optional tools (not well-supported in streaming)
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters

        Yields:
            Text chunks as they arrive. For structured output, yields JSON chunks.
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
            async for chunk in stream:
                content = None
                if isinstance(chunk, dict):
                    content = chunk.get('message', {}).get('content')
                elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    content = chunk.message.content

                if content:
                    chunk_count += 1
                    if response_format is not None:
                        # Accumulate for final validation
                        response_accum += content
                    yield content

            logger.info(f"Ollama stream completed. Total chunks: {chunk_count}")

            # For structured output, validate the accumulated response at the end
            # Note: This is logged but not yielded since we already yielded chunks
            if response_format is not None and response_accum:
                try:
                    response_format.model_validate_json(response_accum)
                    logger.info("Ollama streaming structured output validated successfully")
                except Exception as e:
                    logger.warning(f"Ollama streaming structured output validation failed: {e}")

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    async def get_streaming_response_with_tools(
        self,
        client: "ollama.AsyncClient",
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
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
            LLMResponse: Final response with tool_calls (only if tools were called)
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
                    **kwargs
                )

                # If tool calls detected, yield the response immediately
                if response.has_tool_calls:
                    logger.info(f"Ollama detected {len(response.tool_calls)} tool calls")
                    yield response
                    return

                # No tool calls - yield the text content and we're done
                if response.content:
                    yield response.content
                return

            # No tools - use regular streaming
            async for chunk in self.get_streaming_response(
                client=client,
                model=model,
                messages=messages,
                response_format=response_format,
                **kwargs
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Ollama streaming with tools failed: {e}")
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

    async def list_models(self) -> List[ModelInfo]:
        """
        List available models from local Ollama instance.

        Uses ollama.list() to discover locally available models.

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

                models.append(ModelInfo(
                    id=model_name,
                    provider="ollama",
                    display_name=display_name,
                    category=category,
                    supports_vision=supports_vision,
                    supports_tools=True,  # Ollama 0.4+ supports tools
                    supports_streaming=True,
                    size_gb=size_gb,
                ))

            logger.info(f"Ollama returned {len(models)} local models")
            return models

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
