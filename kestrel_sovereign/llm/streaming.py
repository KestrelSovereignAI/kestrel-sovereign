"""Streaming response logic for LLM Service.

Extracted from service.py to reduce file size. These methods handle:
- Basic streaming responses with provider fallback
- Streaming with pre-built message arrays
- Streaming with tool call detection and assembly
- Unified streaming with remote GPU fallback
"""
import logging
from typing import List, Dict, Any, Optional, Union, Type, AsyncIterator

from pydantic import BaseModel

from .adapter import LLMResponse
from .error_handling import LLMError

logger = logging.getLogger(__name__)


class LLMStreamingError(LLMError):
    """Raised when streaming operation fails."""


class StreamingMixin:
    """Mixin class providing streaming methods for LLMService.

    Expects the following attributes on the host class:
    - providers: List[Dict[str, Any]]
    - _backend: BackendType
    - _remote_client: Optional[AsyncOpenAI]
    - _remote_adapter: OpenAIAdapter
    - _remote_config: Optional[RemoteGPUConfig]
    - _last_remote_error: Optional[str]
    - _mandate_preference: Dict[str, Optional[str]]
    - _ensure_remote_active() -> None
    - _deactivate_remote_backend(reason: Optional[str]) -> None
    """

    async def get_streaming_response(
        self,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: str = None,
        response_format: Optional[Type[BaseModel]] = None
    ):
        """Get a streaming response from the LLM.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers (Ollama)
            model_override: Override the model selection (format: "provider/model")
            response_format: Optional Pydantic model for structured output.
                Note: Not all providers support streaming with structured output.
                OpenAI supports it natively, others may fall back to non-streaming.

        Yields:
            Text chunks as they arrive
        """
        providers_to_use = self.providers

        if model_override and "/" in model_override:
            provider_name, model_name = model_override.split("/", 1)
            override_provider = None
            other_providers = []
            for p in self.providers:
                if p["name"] == provider_name:
                    override_provider = dict(p)
                    override_provider["model"] = model_name
                else:
                    other_providers.append(p)
            if override_provider:
                providers_to_use = [override_provider] + other_providers
            else:
                raise RuntimeError(
                    f"Provider '{provider_name}' not available. "
                    f"Available: {[p['name'] for p in self.providers]}"
                )
            logger.info(f"Model override: {provider_name}/{model_name}")

        if force_local_only:
            providers_to_use = [p for p in providers_to_use if p["name"] in ["ollama"]]
            if not providers_to_use:
                raise RuntimeError("No local providers available.")

        last_error = None
        for provider in providers_to_use:
            try:
                provider_name = provider["name"]
                model_to_use = provider["model"]

                logger.info(f"Attempting streaming from {provider_name} with {model_to_use}")
                messages = provider["adapter"].create_messages(user_prompt=user_prompt, system_prompt=system_prompt)

                adapter = provider["adapter"]

                # For structured output, only some providers support streaming
                # OpenAI and Vertex support streaming with response_format
                # Anthropic does NOT support streaming with structured output (uses tool_use pattern)
                supports_streaming_structured = provider_name in ["openai", "vertex_ai"]

                # Use streaming if supported (or no structured output requested)
                if hasattr(adapter, "get_streaming_response"):
                    if response_format is None or supports_streaming_structured:
                        try:
                            async for chunk in adapter.get_streaming_response(
                                client=provider["client"],
                                model=model_to_use,
                                messages=messages,
                                response_format=response_format
                            ):
                                yield chunk
                            logger.info(f"Streaming completed from {provider_name}")
                            return
                        except NotImplementedError:
                            # Adapter doesn't support streaming, fall through to non-streaming
                            pass

                # Fallback: use non-streaming response (required for Anthropic with structured output)
                response = await adapter.get_response(
                    client=provider["client"],
                    model=model_to_use,
                    messages=messages,
                    response_format=response_format
                )
                # Yield content as string (LLMResponse.content) to match streaming behavior
                yield response.content or ""
                logger.info(f"Non-streaming fallback from {provider_name}")
                return

            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                # Surface failure to user via stream — but NOT for structured output
                # where injected text would corrupt the JSON that the caller parses
                if response_format is None:
                    yield f"\n[Provider {provider['name']} unavailable, trying next...]\n"
                continue

        provider_type = "local" if force_local_only else "all"
        logger.error(f"All {provider_type} providers failed for streaming. Last error: {last_error}")
        raise RuntimeError(f"All {provider_type} providers failed for streaming.")

    async def generate_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        """Stream text using the active backend with automatic fallback.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers
            model_override: Override model selection
            response_format: Optional Pydantic model for structured output.
                Note: Streaming with structured output is provider-dependent.
                OpenAI supports it natively, others may fall back to non-streaming.

        Yields:
            Text chunks as they arrive (JSON chunks if response_format provided)
        """
        from .service import BackendType

        # Try remote GPU first if active
        if self._backend == BackendType.REMOTE_GPU and self._remote_client and not force_local_only:
            try:
                self._ensure_remote_active()
                messages = self._remote_adapter.create_messages(user_prompt=user_prompt, system_prompt=system_prompt)
                model = model_override or self._remote_config.model
                async for chunk in self._remote_adapter.get_streaming_response(
                    client=self._remote_client,
                    model=model,
                    messages=messages,
                    response_format=response_format,
                ):
                    yield chunk
                return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard streaming
        async for chunk in self.get_streaming_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_local_only=force_local_only,
            model_override=model_override,
            response_format=response_format,
        ):
            yield chunk

    async def stream_with_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        force_local_only: bool = False,
        model_override: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response using a pre-built messages array.

        Use this for streaming the final response after tool execution,
        where you need to pass the full conversation history including
        tool results.

        Args:
            messages: Pre-built message list including tool results
            force_local_only: Only use local providers
            model_override: Override model selection

        Yields:
            Text chunks as they arrive from the LLM
        """
        from .service import BackendType

        # Try remote GPU first if active
        if self._backend == BackendType.REMOTE_GPU and self._remote_client and not force_local_only:
            try:
                self._ensure_remote_active()
                model = model_override or self._remote_config.model
                if hasattr(self._remote_adapter, "get_streaming_response"):
                    async for chunk in self._remote_adapter.get_streaming_response(
                        client=self._remote_client,
                        model=model,
                        messages=messages,
                    ):
                        yield chunk
                    return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard providers
        providers = self.providers
        if force_local_only:
            providers = [p for p in providers if p["name"] in ["ollama"]]

        last_error = None
        for provider in providers:
            try:
                adapter = provider["adapter"]
                model = model_override or provider["model"]

                if hasattr(adapter, "get_streaming_response"):
                    async for chunk in adapter.get_streaming_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                    ):
                        yield chunk
                    return
                else:
                    # Fallback to non-streaming if adapter doesn't support it
                    response = await adapter.get_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                    )
                    yield response.content if hasattr(response, 'content') else str(response)
                    return
            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                yield f"\n[Provider {provider['name']} unavailable, trying next...]\n"
                continue

        logger.error(f"All providers failed for stream_with_messages: {last_error}")
        raise LLMStreamingError("All providers failed for stream_with_messages.")

    async def stream_with_tool_detection(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """
        Stream response with tool call detection.

        This is the unified method for streaming with tool detection across all providers.
        It yields text chunks as they arrive, and if tool calls are detected, yields
        an LLMResponse with the assembled tool calls at the end.

        This eliminates the "double LLM call" pattern where you first call non-streaming
        to detect tools, then call streaming for text.

        Args:
            messages: Pre-built message list
            tools: Optional tools for function calling
            force_local_only: Only use local providers (Ollama)
            model_override: Override model selection (format: "provider/model" or just "model")
            system_prompt: Optional system prompt (only used for Anthropic adapter)

        Yields:
            str: Text content chunks as they arrive
            LLMResponse: Final response with tool_calls (only at end if tools were called)

        Example:
            tool_response = None
            async for item in service.stream_with_tool_detection(messages=msgs, tools=tools):
                if isinstance(item, str):
                    print(item, end='', flush=True)  # Stream to user
                elif isinstance(item, LLMResponse):
                    tool_response = item

            if tool_response and tool_response.has_tool_calls:
                # Execute tools and continue
                for tc in tool_response.tool_calls:
                    result = await execute_tool(tc)
        """
        from .service import BackendType

        # Try remote GPU first if active
        if self._backend == BackendType.REMOTE_GPU and self._remote_client and not force_local_only:
            try:
                self._ensure_remote_active()
                model = model_override or self._remote_config.model
                if hasattr(self._remote_adapter, "get_streaming_response_with_tools"):
                    async for item in self._remote_adapter.get_streaming_response_with_tools(
                        client=self._remote_client,
                        model=model,
                        messages=messages,
                        tools=tools,
                    ):
                        yield item
                    return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming with tools failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Determine providers to use
        providers = self.providers
        if force_local_only:
            providers = [p for p in providers if p["name"] in ["ollama"]]
            if not providers:
                raise LLMStreamingError("No local providers available")

        # Handle model override with provider prefix (e.g., "openai/gpt-5-mini")
        target_provider = None
        target_model = None
        if model_override:
            if "/" in model_override:
                provider_name, target_model = model_override.split("/", 1)
                # Find the specified provider
                for p in providers:
                    if p["name"] == provider_name:
                        target_provider = p
                        break
                if target_provider:
                    providers = [target_provider] + [p for p in providers if p != target_provider]
            else:
                target_model = model_override
        else:
            # Check mandate preference (set by !model-set command or UI selection)
            pref_model = self._mandate_preference.get("model")
            pref_provider = self._mandate_preference.get("provider")
            if pref_model:
                target_model = pref_model
                if pref_provider:
                    # When provider is explicitly set, ONLY use that provider
                    # Don't fall back to others - they won't have the same model
                    for p in providers:
                        if p["name"] == pref_provider:
                            target_provider = p
                            break
                    if target_provider:
                        # Use ONLY the specified provider - no fallbacks with wrong model
                        providers = [target_provider]
                        logger.info(f"Model mandate set: using only {pref_provider} with {pref_model}")
                    else:
                        logger.warning(f"Mandated provider '{pref_provider}' not found in available providers")

        last_error = None
        for provider in providers:
            try:
                adapter = provider["adapter"]
                model = target_model or provider["model"]
                provider_name = provider["name"]

                logger.info(f"Attempting streaming with tools from {provider_name} with {model}")

                # Check if adapter supports streaming with tool detection
                if hasattr(adapter, "get_streaming_response_with_tools"):
                    # Build kwargs for provider-specific parameters
                    kwargs = {}
                    if provider_name == "anthropic" and system_prompt:
                        kwargs["system_prompt"] = system_prompt

                    async for item in adapter.get_streaming_response_with_tools(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                        tools=tools,
                        **kwargs
                    ):
                        yield item
                    logger.info(f"Streaming with tools completed from {provider_name}")
                    return
                else:
                    # Fallback: use non-streaming for tool detection, then stream text
                    logger.warning(f"{provider_name} doesn't support streaming with tools, using fallback")
                    if tools:
                        response = await adapter.get_response(
                            client=provider["client"],
                            model=model,
                            messages=messages,
                            tools=tools,
                        )
                        if response.has_tool_calls:
                            yield response
                            return
                        # No tool calls, yield content
                        if response.content:
                            yield response.content
                        return
                    else:
                        # No tools, just stream
                        async for chunk in adapter.get_streaming_response(
                            client=provider["client"],
                            model=model,
                            messages=messages,
                        ):
                            yield chunk
                        return

            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                yield f"\n[Provider {provider['name']} unavailable, trying next...]\n"
                continue

        logger.error(f"All providers failed for stream_with_tool_detection: {last_error}")
        raise LLMStreamingError("All providers failed for stream_with_tool_detection.")
