"""Streaming response logic for LLM Service.

Extracted from service.py to reduce file size. These methods handle:
- Basic streaming responses with provider fallback
- Streaming with pre-built message arrays
- Streaming with tool call detection and assembly
- Unified streaming with remote GPU fallback

No-silent-fallback rule
-----------------------
When ``resolve_provider_routing`` narrows the candidate list by an explicit
mandate or ``model_override``, the streaming loop must NOT silently fall
through to a different provider on failure. The user selected a specific
backend; answering from a different one without saying so is lying. We
enforce this by failing loudly (``LLMStreamingError``) whenever the
provider list has exactly one entry — that covers every mandate-restricted
or override-restricted case. Multi-provider default chains still retry
through the list, but the fallback happens in server logs, not by
injecting a ``[Provider X unavailable, trying next...]`` note into the
chat stream where it corrupts the agent's response.
"""
import logging
from typing import List, Dict, Any, Optional, Union, Type, AsyncIterator

from pydantic import BaseModel

from .adapter import LLMResponse
from .error_handling import LLMError

logger = logging.getLogger(__name__)


class LLMStreamingError(LLMError):
    """Raised when streaming operation fails.

    Carries the failing provider's composite name and the underlying error
    so callers (and ultimately the user) get a specific actionable message
    instead of ``"all providers failed"``.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        underlying: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.underlying = underlying


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

    def _check_model_tool_support(
        self,
        providers: list,
        tools: Optional[list],
        model_override: Optional[str] = None,
    ) -> Optional[list]:
        """Check if the target model supports tools; strip them if not.

        Cloud routes always support tools (every cloud vendor's chat API does).
        Local routes may run small models that can't tool-call — we fall
        through to the discovered ``ModelInfo.supports_tools`` flag.
        """
        if not tools:
            return tools

        if not providers:
            return tools
        target_route = providers[0]
        is_cloud = (
            target_route.get("is_cloud")
            if isinstance(target_route, dict)
            else getattr(target_route, "is_cloud", True)
        )
        if is_cloud:
            return tools  # Cloud routes always support tools.

        # Resolve which model we'll actually use
        target_model = model_override
        if target_model and "/" in target_model:
            _, target_model = target_model.split("/", 1)
        if not target_model and providers:
            p = providers[0]
            target_model = p["model"] if isinstance(p, dict) else getattr(p, "model", None)

        if not target_model:
            return tools  # Can't determine model, pass tools through

        # Check discovered model info (exact match only — no substring matching)
        from .model_cache import get_shared_model_cache
        cache = get_shared_model_cache().get_any()
        if not cache:
            return tools  # No discovery data yet, pass tools through

        for model_info in cache:
            if model_info.id == target_model:
                if not model_info.supports_tools:
                    logger.info(
                        f"Model {target_model} does not support tools "
                        f"({model_info.size_gb or '?'}GB) — sending without tools"
                    )
                    return None
                return tools

        return tools  # Model not in cache, pass tools through

    def _get_local_provider_names(self) -> set:
        """Route keys (``"vendor:route"``) for all local routes.

        Retained as a convenience for call sites that pre-date the
        ``is_local`` flag on provider dicts; prefer reading ``p["is_local"]``
        directly in new code.
        """
        try:
            if hasattr(self, 'provider_registry') and self.provider_registry:
                locals_ = self.provider_registry.get_local_providers()
                if locals_:
                    return {p.name for p in locals_}
        except (TypeError, AttributeError):
            pass
        return set()

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
        providers_to_use, target_model = self.resolve_provider_routing(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        mandate_restricted = len(providers_to_use) == 1

        last_error = None
        last_provider_name = None
        for provider in providers_to_use:
            try:
                provider_name = provider["name"]
                last_provider_name = provider_name
                model_to_use = target_model or provider["model"]

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
                if mandate_restricted:
                    # No silent fallthrough when the user has explicitly narrowed
                    # routing. Fail loudly — the caller / agent / user must see
                    # the specific error, not a response from a different model.
                    raise LLMStreamingError(
                        f"Selected route {provider_name} failed: {e}",
                        provider=provider_name,
                        underlying=e,
                    )
                # Default multi-provider chain: log the fallback server-side;
                # don't corrupt the stream with a note about it.
                logger.warning(
                    "Falling through from %s to next provider in chain: %s",
                    provider_name, e,
                )
                continue

        provider_type = "local" if force_local_only else "all"
        logger.error(f"All {provider_type} providers failed for streaming. Last error: {last_error}")
        raise LLMStreamingError(
            f"All {provider_type} providers failed: {last_error}",
            provider=last_provider_name,
            underlying=last_error,
        )

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
        from .remote_backend import BackendType

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
        from .remote_backend import BackendType

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

        # Fall back to standard providers — use centralized routing
        providers, target_model = self.resolve_provider_routing(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        mandate_restricted = len(providers) == 1

        last_error = None
        last_provider_name = None
        for provider in providers:
            try:
                last_provider_name = provider["name"]
                adapter = provider["adapter"]
                model = target_model or provider["model"]

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
                if mandate_restricted:
                    raise LLMStreamingError(
                        f"Selected route {provider['name']} failed: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                logger.warning(
                    "Falling through from %s: %s", provider["name"], e,
                )
                continue

        logger.error(f"All providers failed for stream_with_messages: {last_error}")
        raise LLMStreamingError(
            f"All providers failed: {last_error}",
            provider=last_provider_name,
            underlying=last_error,
        )

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
        from .remote_backend import BackendType

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

        # Use centralized provider routing
        providers, target_model = self.resolve_provider_routing(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        mandate_restricted = len(providers) == 1

        # Strip tools if the target model can't handle them
        tools = self._check_model_tool_support(providers, tools, model_override)

        last_error = None
        last_provider_name = None
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
                last_provider_name = provider["name"]
                if mandate_restricted:
                    raise LLMStreamingError(
                        f"Selected route {provider['name']} failed: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                logger.warning(
                    "Falling through from %s: %s", provider["name"], e,
                )
                continue

        logger.error(f"All providers failed for stream_with_tool_detection: {last_error}")
        raise LLMStreamingError(
            f"All providers failed: {last_error}",
            provider=last_provider_name,
            underlying=last_error,
        )
