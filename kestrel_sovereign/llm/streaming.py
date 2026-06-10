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
import time
from typing import Awaitable, Callable, List, Dict, Any, Optional, Union, Type, AsyncIterator

from pydantic import BaseModel

from kestrel_sdk.llm import ToolCallStarted

from .adapter import LLMResponse, ThinkingDelta, messages_for
from .codex_app_server import CodexAppServerTransportError
from .error_handling import LLMError
from .provider_registry import provider_cache_body

logger = logging.getLogger(__name__)


def _is_harness_owned_transport_error(exc: BaseException) -> bool:
    """True when ``exc`` is a *transport* failure from a harness that
    owns its own transport (timeouts, retries, websocket lifecycle)
    and therefore must NOT be treated as evidence the route is broken.

    Sovereign's only such harness today is the codex app-server bridge
    (``openai:plan``): a transient codex/ChatGPT-Plus stall raises
    ``CodexAppServerTransportError`` (idle timeout, RPC timeout,
    app-server connection closed) and is **not** a signal that openai
    is down. Rotating to a different provider on this error gives the
    user a wrong-model response without warning — the upstream
    antipattern openclaw fixed in commit ``3a64dc7623`` ("keep turn
    timeouts inside Codex").

    Narrowed to ``CodexAppServerTransportError`` specifically (not the
    supertype ``CodexAppServerError``) so caller-config / protocol /
    codex-reported-turn-failure errors retain their normal fallback
    semantics — those *should* let the chain try the next provider.
    """
    return isinstance(exc, CodexAppServerTransportError)


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

    async def _record_streamed_usage(
        self,
        response: Any,
        model: str,
        provider_name: str,
        *,
        duration_ms: int,
    ) -> None:
        """Meter a streamed turn from its terminal :class:`LLMResponse`.

        The streaming path never reached ``_track_model_usage`` /
        ``_log_llm_call`` (the non-streaming chokepoint), so every streamed
        turn silently bypassed usage tracking and the billing meter. This
        mirrors the non-streaming recording (service.py) from the terminal
        response. Best-effort: a recording failure must never break the
        stream the user is consuming.
        """
        if not isinstance(response, LLMResponse):
            return
        try:
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
            await self._track_model_usage(model, provider_name, tokens=total_tokens)
            await self._log_llm_call(
                provider=provider_name,
                model=model,
                duration_ms=duration_ms,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=getattr(
                    response, "cache_creation_input_tokens", None
                ),
                cache_read_input_tokens=getattr(
                    response, "cache_read_input_tokens", None
                ),
                tools_used=bool(getattr(response, "tool_calls", None)),
                metadata={"streamed": True},
            )
        except Exception as exc:  # noqa: BLE001 - metering must not break stream
            logger.warning("Failed to record streamed usage: %s", exc)

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
        self._check_policy()
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
                model_to_use = self._resolve_concrete_model(target_model, provider)

                logger.info(f"Attempting streaming from {provider_name} with {model_to_use}")
                messages = messages_for(provider["adapter"], user_prompt=user_prompt, system_prompt=system_prompt)

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
                                response_format=response_format,
                                extra_body=provider_cache_body(provider),
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
                    response_format=response_format,
                    extra_body=provider_cache_body(provider),
                )
                # Yield content as string (LLMResponse.content) to match streaming behavior
                yield response.content or ""
                logger.info(f"Non-streaming fallback from {provider_name}")
                return

            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                if _is_harness_owned_transport_error(e):
                    # Harness-owned transport error (codex app-server idle
                    # stall, app-server connection closed, etc.). Don't
                    # rotate to a different provider — that would answer
                    # the user from a wrong model on a transient codex
                    # stall. Also skip ``_maybe_disable_route``: even an
                    # auth-shaped codex message ("session expired",
                    # "unauthorized") is the harness's responsibility, not
                    # evidence the kestrel route is broken — disabling it
                    # for the rest of the process would skip codex on
                    # every future turn even after the operator
                    # re-authenticates. See #1429 and openclaw commit
                    # 3a64dc7623.
                    raise LLMStreamingError(
                        f"Harness-owned route {provider_name} failed: {e}",
                        provider=provider_name,
                        underlying=e,
                    )
                self._maybe_disable_route(provider, e)
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
        self._check_policy()
        from .remote_backend import BackendType

        # Try remote GPU first when active AND routing isn't pinned — #734.
        if (
            self._backend == BackendType.REMOTE_GPU
            and self._remote_client
            and not force_local_only
            and self._remote_first_allowed(model_override)
        ):
            try:
                self._ensure_remote_active()
                messages = messages_for(self._remote_adapter, user_prompt=user_prompt, system_prompt=system_prompt)
                model = self._scrub_auto(model_override) or self._remote_config.model
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
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response using a pre-built messages array.

        Use this for streaming the final response after tool execution,
        where you need to pass the full conversation history including
        tool results.

        Args:
            messages: Pre-built message list including tool results
            force_local_only: Only use local providers
            model_override: Override model selection
            session_id: See ``generate_with_messages``. #808.

        Yields:
            Text chunks as they arrive from the LLM
        """
        self._check_policy()
        from .remote_backend import BackendType

        # Try remote GPU first when active AND routing isn't pinned — #734.
        if (
            self._backend == BackendType.REMOTE_GPU
            and self._remote_client
            and not force_local_only
            and self._remote_first_allowed(model_override)
        ):
            try:
                self._ensure_remote_active()
                model = self._scrub_auto(model_override) or self._remote_config.model
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
                model = self._resolve_concrete_model(target_model, provider)

                if hasattr(adapter, "get_streaming_response"):
                    async for chunk in adapter.get_streaming_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                        extra_body=provider_cache_body(provider),
                        session_id=session_id,
                    ):
                        yield chunk
                    return
                else:
                    # Fallback to non-streaming if adapter doesn't support it
                    response = await adapter.get_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                        extra_body=provider_cache_body(provider),
                        session_id=session_id,
                    )
                    yield response.content if hasattr(response, 'content') else str(response)
                    return
            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                if _is_harness_owned_transport_error(e):
                    # See #1429: skip _maybe_disable_route too — harness
                    # owns auth, kestrel doesn't disable the route on its
                    # behalf.
                    raise LLMStreamingError(
                        f"Harness-owned route {provider['name']} failed: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                self._maybe_disable_route(provider, e)
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
        session_id: Optional[str] = None,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
    ) -> AsyncIterator[Union[str, ThinkingDelta, ToolCallStarted, LLMResponse]]:
        """
        Stream response with tool call detection.

        Unified streaming-with-tools across all providers. Yields the
        SDK 0.7+ tagged union from
        :meth:`LLMAdapter.get_streaming_response_with_tools`:

        * ``str`` — text content chunks as they arrive.
        * :class:`ToolCallStarted` — emitted the moment a tool call
          first appears in the provider stream (one event per
          distinct ``index``, in stream order). The constitutional
          honesty layer (#1042 layer 2 / #1045) gates pre-tool prose
          on this signal — consumers that pipe text directly to a
          chat UI should clear or revise the in-flight bubble when a
          marker arrives. ``stream_with_tool_detection`` does not
          itself process or filter markers; it forwards them
          unchanged from the underlying adapter.
        * :class:`ThinkingDelta` — provider-separated model reasoning
          that should be displayed as expandable UI affordance, not as
          assistant answer text or persisted conversation content.
        * :class:`LLMResponse` — exactly once at end-of-stream when
          tool calls were detected. Source of truth for the assembled
          tool calls (id, name, arguments) and token usage.

        This eliminates the "double LLM call" pattern where you first
        called non-streaming to detect tools then streaming for text.

        Args:
            messages: Pre-built message list
            tools: Optional tools for function calling
            force_local_only: Only use local providers (Ollama)
            model_override: Override model selection (format:
                ``"provider/model"`` or just ``"model"``)
            system_prompt: Optional system prompt (only used for
                Anthropic adapter)

        Yields:
            ``Union[str, ToolCallStarted, LLMResponse]`` per the
            stream contract above.

        Example:
            tool_response = None
            async for item in service.stream_with_tool_detection(messages=msgs, tools=tools):
                if isinstance(item, str):
                    print(item, end='', flush=True)  # stream text to user
                elif isinstance(item, ToolCallStarted):
                    # Stop optimistic text rendering; a tool call is
                    # about to fire. Frontend may clear the in-flight
                    # message bubble here.
                    on_tool_starting(item)
                elif isinstance(item, LLMResponse):
                    tool_response = item

            if tool_response and tool_response.has_tool_calls:
                # Execute tools and continue
                for tc in tool_response.tool_calls:
                    result = await execute_tool(tc)
        """
        self._check_policy()
        from .remote_backend import BackendType

        # Try remote GPU first when active AND routing isn't pinned — #734.
        if (
            self._backend == BackendType.REMOTE_GPU
            and self._remote_client
            and not force_local_only
            and self._remote_first_allowed(model_override)
        ):
            try:
                self._ensure_remote_active()
                model = self._scrub_auto(model_override) or self._remote_config.model
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
                model = self._resolve_concrete_model(target_model, provider)
                provider_name = provider["name"]

                logger.info(f"Attempting streaming with tools from {provider_name} with {model}")

                # Check if adapter supports streaming with tool detection
                if hasattr(adapter, "get_streaming_response_with_tools"):
                    # Build kwargs for provider-specific parameters
                    kwargs = {}
                    if provider_name == "anthropic" and system_prompt:
                        kwargs["system_prompt"] = system_prompt
                    cache_body = provider_cache_body(provider)
                    if cache_body:
                        kwargs["extra_body"] = cache_body
                    if session_id:
                        kwargs["session_id"] = session_id
                    if tool_executor is not None:
                        kwargs["tool_executor"] = tool_executor

                    # Meter the streamed turn from its terminal LLMResponse.
                    # The `finally` records even if the consumer stops iterating
                    # after the terminal response arrives. (A true mid-stream
                    # abort, before the terminal response, still loses usage —
                    # that needs adapter-level incremental token tracking,
                    # tracked separately.)
                    stream_start = time.monotonic()
                    final_response = None
                    try:
                        async for item in adapter.get_streaming_response_with_tools(
                            client=provider["client"],
                            model=model,
                            messages=messages,
                            tools=tools,
                            **kwargs
                        ):
                            if isinstance(item, LLMResponse):
                                final_response = item
                            yield item
                    finally:
                        if final_response is not None:
                            await self._record_streamed_usage(
                                final_response, model, provider_name,
                                duration_ms=int((time.monotonic() - stream_start) * 1000),
                            )
                    logger.info(f"Streaming with tools completed from {provider_name}")
                    return
                else:
                    # Fallback: use non-streaming for tool detection, then stream text
                    logger.warning(f"{provider_name} doesn't support streaming with tools, using fallback")
                    if tools:
                        fb_start = time.monotonic()
                        response = await adapter.get_response(
                            client=provider["client"],
                            model=model,
                            messages=messages,
                            tools=tools,
                            extra_body=provider_cache_body(provider),
                            session_id=session_id,
                        )
                        # adapter.get_response does not meter (only the service's
                        # non-streaming path does), so record it here too.
                        await self._record_streamed_usage(
                            response, model, provider_name,
                            duration_ms=int((time.monotonic() - fb_start) * 1000),
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
                            extra_body=provider_cache_body(provider),
                            session_id=session_id,
                        ):
                            yield chunk
                        return

            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                last_provider_name = provider["name"]
                if _is_harness_owned_transport_error(e):
                    # See #1429: skip _maybe_disable_route too — harness
                    # owns auth, kestrel doesn't disable the route on its
                    # behalf.
                    raise LLMStreamingError(
                        f"Harness-owned route {provider['name']} failed: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                self._maybe_disable_route(provider, e)
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
