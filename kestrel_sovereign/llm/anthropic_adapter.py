"""
Anthropic Claude Adapter

Adapter for Anthropic's Claude API with support for:
- Tool/function calling
- Vision (image inputs)
- Streaming responses
- API-based model discovery
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Type

import httpx
from pydantic import BaseModel

from .adapter import LLMAdapter, LLMResponse, ToolCall
from .model_metadata import ModelInfo, ModelCategory
from .image_utils import process_images
from .retry import with_retry
from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompt caching helpers (issue #705)
# --------------------------------------------------------------------------
# Anthropic's prompt cache is opt-in via `cache_control: {"type": "ephemeral"}`
# markers attached to content blocks.  Up to 4 breakpoints per request.
# Cache hits are charged at 0.1× input tokens; misses at 1× ; writes at
# 1.25× .  Default TTL is 5 min (1 hour opt-in).
#
# Anthropic silently no-ops when the prefix is below the per-model minimum
# (1024 tokens for Opus/Sonnet, 2048 for Haiku), so the marker placement
# code doesn't need its own threshold check — Anthropic handles it.

_DEFAULT_CACHE_TTL = "1h"  # default; bumped from "5m" to widen the warm window
# Activates the longer TTLs (5m, 1h) — without it, Anthropic ignores the `ttl`
# field and falls back to its 5-minute default cache.
_EXTENDED_CACHE_TTL_BETA = "extended-cache-ttl-2025-04-11"

CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral", "ttl": _DEFAULT_CACHE_TTL}


def _ensure_anthropic_beta_header(api_params: Dict[str, Any], beta: str) -> None:
    """Merge ``beta`` into ``api_params['extra_headers']['anthropic-beta']``.

    Anthropic accepts the header as comma-separated for multiple beta opt-ins;
    callers may have already added one (e.g. `prompt-caching-2024-07-31`), so
    we append rather than overwrite when present.
    """
    headers = dict(api_params.get("extra_headers") or {})
    existing = headers.get("anthropic-beta", "")
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if beta not in parts:
        parts.append(beta)
    headers["anthropic-beta"] = ",".join(parts)
    api_params["extra_headers"] = headers


def _attach_cache_control(block: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `block` with ``cache_control: {"type":"ephemeral"}``
    attached.  Input blocks are not mutated so callers passing shared
    structures (e.g. tool schemas reused across requests) are safe.
    """
    new_block = dict(block)
    new_block["cache_control"] = CACHE_CONTROL_EPHEMERAL
    return new_block


def _system_as_cacheable_array(system_text: str) -> List[Dict[str, Any]]:
    """Convert the plain-string system prompt to the content-block array
    form that Anthropic requires for `cache_control` attachment, with the
    marker on the single (and therefore final) block.  Covers marker #1:
    end-of-system.
    """
    return [
        {
            "type": "text",
            "text": system_text,
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
    ]


def _tools_with_final_cache_marker(
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a new list of tools with `cache_control` attached to the
    final tool — covers marker #2: end-of-tools.  Caching tool schemas is
    valuable because they rarely change across turns in a conversation.
    """
    if not tools:
        return tools
    marked = list(tools[:-1])
    marked.append(_attach_cache_control(tools[-1]))
    return marked


def _mark_message_content(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *msg* with ``cache_control`` attached to its content.
    Handles plain-string and content-block-array forms. Returns the input
    unchanged when content is empty/None so the caller never emits a
    malformed request.
    """
    content = msg.get("content")
    if isinstance(content, str):
        marked = dict(msg)
        marked["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": CACHE_CONTROL_EPHEMERAL,
            }
        ]
        return marked
    if isinstance(content, list) and content:
        marked = dict(msg)
        blocks = list(content[:-1])
        blocks.append(_attach_cache_control(content[-1]))
        marked["content"] = blocks
        return marked
    return msg


def _messages_with_penultimate_cache_marker(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach ``cache_control`` to the history messages before the current
    user turn — covers end-of-history caching.

    **Why two markers, not one.** Anthropic's prompt cache is indexed by
    marker position, not by arbitrary longest-prefix match. A marker at
    ``messages[-2]`` alone creates a cache entry at THIS turn's [-2]
    position, but NEXT turn's cache lookup only queries entries that
    correspond to NEXT turn's markers. Next turn's [-2] is a new position
    (the just-generated assistant response), so the previous turn's
    cached entry is never queried.

    To hit the previous turn's cache entry, mark ``messages[-4]`` — which
    at turn N+1 is the same position that was ``messages[-2]`` at turn N,
    i.e. the end-of-history boundary from the prior request. That lookup
    hits the cached entry and produces a compound cache read.

    Markers placed (when sufficient history exists):
        • ``messages[-2]`` — end of latest completed exchange
        • ``messages[-4]`` — end of second-to-last exchange (matches the
          prior turn's [-2] position so cache entries compound)

    Verified empirically against ``api.anthropic.com`` in a 3-turn
    benchmark: dropping the [-4] marker collapses T3 cache_read to the
    system-only size (Anthropic stops compounding). See
    ``tests/integration/test_anthropic_cache_real.py``.

    The current user turn (``messages[-1]``) is never marked because it
    changes every request by definition. If ``messages`` has < 2 entries
    the final entry IS the current user turn — returned unchanged.

    Compound caching only works when the bytes at ``messages[-4]`` of
    turn N+1 are byte-identical to ``messages[-2]`` of turn N. That is
    the atomic-storage contract in ``agent/context_builder.py`` —
    user-turn sent-form is persisted verbatim with
    ``metadata.sent_form=True`` and replayed byte-exactly on load.
    """
    if len(messages) < 2:
        return messages

    mark_at = {len(messages) - 2}
    if len(messages) >= 4:
        mark_at.add(len(messages) - 4)

    return [
        _mark_message_content(msg) if i in mark_at else msg
        for i, msg in enumerate(messages)
    ]


class AnthropicAdapter(LLMAdapter):
    """
    Adapter for Anthropic Claude API.

    Note: Anthropic uses a different message format than OpenAI.
    The system prompt is passed separately, not in messages.
    """

    @staticmethod
    def _apply_cache_control(
        api_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach ``cache_control: ephemeral`` markers at the canonical
        positions in the Anthropic request, respecting the 4-breakpoint
        limit.  Takes and returns a new ``api_params`` dict so callers
        keep their input intact for logging/retry.

        Markers placed (all 4 slots consumed):
            1. End of ``system`` (converts string → content-block array).
            2. End of ``tools`` (final tool gets the marker).
            3. ``messages[-2]`` — end of latest completed exchange.
            4. ``messages[-4]`` — end of second-to-last exchange. Needed
               so turn N+1's lookup hits the cache entry turn N wrote
               at its own [-2] position (which is N+1's [-4] position).

        Why both history markers. Anthropic's cache is indexed by marker
        position, not longest-prefix match. A single marker at [-2]
        alone creates entries at THIS turn's [-2] but never queries the
        PREVIOUS turn's [-2] entry — so cache_read stays flat at the
        system-prefix size across turns. The [-4] marker is what asks
        Anthropic to look up the entry the prior turn wrote.
        See ``tests/integration/test_anthropic_cache_real.py``'s 3-turn
        compound-caching test for the empirical check.

        Compound caching additionally requires ``messages[-4]`` of turn
        N+1 to be byte-identical to ``messages[-2]`` of turn N. The
        atomic-storage contract (``metadata.sent_form=True`` on user
        rows, honored by ``context_builder.format_conversation_history``)
        is what keeps the bytes stable so the marker actually hits.

        Anthropic silently no-ops markers whose prefix is below the
        per-model minimum (1024 tokens for Opus/Sonnet, 2048 for Haiku),
        so no explicit threshold check is needed here.
        """
        updated = dict(api_params)

        system = updated.get("system")
        if isinstance(system, str) and system:
            updated["system"] = _system_as_cacheable_array(system)

        tools = updated.get("tools")
        if isinstance(tools, list) and tools:
            updated["tools"] = _tools_with_final_cache_marker(tools)

        messages = updated.get("messages")
        if isinstance(messages, list) and len(messages) >= 2:
            updated["messages"] = _messages_with_penultimate_cache_marker(
                messages
            )

        return updated

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create messages in Anthropic format.

        Note: System prompts are included with role "system" for consistency
        with OpenAI format. The get_response method will extract them and
        pass them via Anthropic's top-level system parameter.
        """
        messages = []
        
        # Include system prompt in messages array for consistency with OpenAI format.
        # get_response will extract it and pass via Anthropic's system parameter.
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt or images:
            content = []

            if user_prompt:
                content.append({"type": "text", "text": user_prompt})

            # Handle images using centralized image_utils
            # Pass provider for auto-resize to Anthropic's 1568x1568 limit
            if images:
                for processed in process_images(images, provider="anthropic"):
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": processed.mime_type,
                            "data": processed.data
                        }
                    })

            messages.append({"role": "user", "content": content})

        return messages

    def _convert_tools_to_anthropic_format(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Anthropic format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        Anthropic format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {...}
        }
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        return anthropic_tools

    async def get_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Get response from Anthropic Claude API.

        Args:
            client: Anthropic AsyncClient instance
            model: Model name (e.g., 'claude-sonnet-4-6')
            messages: List of message dicts
            format: Response format (ignored for Anthropic)
            tools: Optional tools in OpenAI format (will be converted)
            response_format: Optional Pydantic model for structured output
            system_prompt: System prompt (passed separately)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and/or tool calls
        """
        try:
            # Convert OpenAI message format to Anthropic format:
            # - Extract system messages to top-level system parameter
            # - Convert tool role messages to user messages with tool_result blocks
            system_messages = []
            filtered_messages = []
            pending_tool_results = []
            
            for msg in messages:
                role = msg.get("role")
                
                if role == "system":
                    # Extract system messages for top-level parameter
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            block.get("text", "") for block in content 
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    system_messages.append(content)
                    
                elif role == "tool":
                    # Accumulate tool results to combine into a user message
                    # OpenAI format: {"role": "tool", "tool_call_id": "...", "content": "..."}
                    # Anthropic format: user message with tool_result content blocks
                    pending_tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", "")
                    })
                    
                else:
                    # Flush any pending tool results before adding this message
                    if pending_tool_results:
                        filtered_messages.append({
                            "role": "user",
                            "content": pending_tool_results
                        })
                        pending_tool_results = []
                    
                    # Convert the message to Anthropic format
                    converted_msg = {"role": role}
                    content = msg.get("content")
                    tool_calls = msg.get("tool_calls")
                    
                    if role == "assistant" and tool_calls:
                        # Convert OpenAI tool_calls to Anthropic tool_use content blocks
                        content_blocks = []
                        if content:
                            # Add text content first if present
                            if isinstance(content, str):
                                content_blocks.append({"type": "text", "text": content})
                            elif isinstance(content, list):
                                content_blocks.extend(content)
                        # Add tool_use blocks for each tool call
                        for tc in tool_calls:
                            tool_use_block = {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("function", {}).get("name", tc.get("name", "")),
                                "input": tc.get("function", {}).get("arguments", tc.get("arguments", {}))
                            }
                            # Parse arguments if it's a string
                            if isinstance(tool_use_block["input"], str):
                                try:
                                    tool_use_block["input"] = json.loads(tool_use_block["input"])
                                except (json.JSONDecodeError, TypeError):
                                    tool_use_block["input"] = {}
                            content_blocks.append(tool_use_block)
                        converted_msg["content"] = content_blocks
                    elif content is not None:
                        # Normalize content format
                        if isinstance(content, str):
                            converted_msg["content"] = content
                        elif isinstance(content, list):
                            converted_msg["content"] = content
                        else:
                            converted_msg["content"] = str(content)
                    else:
                        # Empty content - use empty string
                        converted_msg["content"] = ""
                    
                    filtered_messages.append(converted_msg)
            
            # Flush any remaining tool results at the end
            if pending_tool_results:
                filtered_messages.append({
                    "role": "user",
                    "content": pending_tool_results
                })
            
            # Combine extracted system messages with explicit system_prompt
            combined_system = "\n\n".join(filter(None, system_messages))
            if system_prompt:
                combined_system = f"{combined_system}\n\n{system_prompt}" if combined_system else system_prompt
            
            api_params = {
                "model": model,
                "messages": filtered_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
            }

            if combined_system:
                api_params["system"] = combined_system

            if "temperature" in kwargs:
                api_params["temperature"] = kwargs["temperature"]

            # Handle structured output via tool_use pattern
            # Anthropic recommends using tools to get structured output
            structured_output_tool_name = None
            if response_format is not None and issubclass(response_format, BaseModel):
                structured_output_tool_name = f"output_{response_format.__name__}"
                schema = response_format.model_json_schema()
                structured_tool = {
                    "name": structured_output_tool_name,
                    "description": f"Output structured response as {response_format.__name__}",
                    "input_schema": schema
                }
                # Add the structured output tool
                api_params["tools"] = [structured_tool]
                # Force the model to use this tool
                api_params["tool_choice"] = {"type": "tool", "name": structured_output_tool_name}
            elif tools:
                # Convert and add tools
                api_params["tools"] = self._convert_tools_to_anthropic_format(tools)

            # Apply prompt-cache markers at up to three positions:
            # end-of-system, end-of-tools, and end-of-history (the message
            # just before the current user turn).  See issue #705 and the
            # helper docstrings in this module.  Anthropic silently no-ops
            # below the per-model minimum cache size, so we don't gate.
            api_params = self._apply_cache_control(api_params)

            # CACHE_CONTROL_EPHEMERAL carries `ttl: "1h"` (issue #797), which
            # Anthropic only honors when this beta header is present. Without
            # it the cache silently falls back to the 5-minute default.
            _ensure_anthropic_beta_header(api_params, _EXTENDED_CACHE_TTL_BETA)

            response = await with_retry(client.messages.create, **api_params)

            # Parse response
            content = None
            parsed_tool_calls = None

            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    # If this is our structured output tool, extract as JSON content
                    if structured_output_tool_name and block.name == structured_output_tool_name:
                        # Return the structured output as JSON string content
                        content = json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                    else:
                        if parsed_tool_calls is None:
                            parsed_tool_calls = []
                        parsed_tool_calls.append(ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {}
                        ))

            # Extract token usage from response
            input_tokens = None
            output_tokens = None
            total_tokens = None
            cache_creation_input_tokens = None
            cache_read_input_tokens = None
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'input_tokens', None)
                output_tokens = getattr(response.usage, 'output_tokens', None)
                # Anthropic reports cache usage separately; either may be 0
                # (no write/read on this call) or absent on older API versions.
                cache_creation_input_tokens = getattr(
                    response.usage, 'cache_creation_input_tokens', None
                )
                cache_read_input_tokens = getattr(
                    response.usage, 'cache_read_input_tokens', None
                )
                if input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            raise

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Get streaming response from Anthropic Claude.

        Args:
            client: Anthropic AsyncClient
            model: Model name
            messages: Chat messages
            tools: Optional tools
            response_format: Optional Pydantic model (not fully supported in streaming)
            system_prompt: System prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks as they arrive

        Note:
            Structured output with response_format is not well-supported in streaming mode
            for Anthropic. Use non-streaming get_response for structured output.
        """
        try:
            # Convert OpenAI message format to Anthropic format
            system_messages = []
            filtered_messages = []
            pending_tool_results = []
            
            for msg in messages:
                role = msg.get("role")
                if role == "system":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            block.get("text", "") for block in content 
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    system_messages.append(content)
                elif role == "tool":
                    pending_tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", "")
                    })
                else:
                    if pending_tool_results:
                        filtered_messages.append({"role": "user", "content": pending_tool_results})
                        pending_tool_results = []
                    
                    # Convert assistant messages with tool_calls to Anthropic format
                    converted_msg = {"role": role}
                    content = msg.get("content")
                    tool_calls = msg.get("tool_calls")
                    
                    if role == "assistant" and tool_calls:
                        content_blocks = []
                        if content:
                            if isinstance(content, str):
                                content_blocks.append({"type": "text", "text": content})
                            elif isinstance(content, list):
                                content_blocks.extend(content)
                        for tc in tool_calls:
                            tool_input = tc.get("function", {}).get("arguments", tc.get("arguments", {}))
                            if isinstance(tool_input, str):
                                try:
                                    tool_input = json.loads(tool_input)
                                except (json.JSONDecodeError, TypeError):
                                    tool_input = {}
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("function", {}).get("name", tc.get("name", "")),
                                "input": tool_input
                            })
                        converted_msg["content"] = content_blocks
                    elif content is not None:
                        converted_msg["content"] = content if isinstance(content, (str, list)) else str(content)
                    else:
                        converted_msg["content"] = ""
                    
                    filtered_messages.append(converted_msg)
            
            if pending_tool_results:
                filtered_messages.append({"role": "user", "content": pending_tool_results})
            
            combined_system = "\n\n".join(filter(None, system_messages))
            if system_prompt:
                combined_system = f"{combined_system}\n\n{system_prompt}" if combined_system else system_prompt
            
            api_params = {
                "model": model,
                "messages": filtered_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
            }

            if combined_system:
                api_params["system"] = combined_system

            if tools:
                api_params["tools"] = self._convert_tools_to_anthropic_format(tools)

            # Note: response_format not implemented for streaming as it requires
            # tool_use which doesn't work well with streaming text output

            # Attach cache_control markers — see issue #705 and _apply_cache_control.
            api_params = self._apply_cache_control(api_params)
            _ensure_anthropic_beta_header(api_params, _EXTENDED_CACHE_TTL_BETA)

            async with client.messages.stream(**api_params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}", exc_info=True)
            raise

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """
        Stream response with tool call detection.

        This method streams text content as it arrives AND detects tool calls
        from the streaming events. Anthropic uses content_block events with
        different types for text vs tool_use.

        Args:
            client: Anthropic AsyncClient
            model: Model name (e.g., 'claude-sonnet-4-6')
            messages: Chat messages
            tools: Optional tools in OpenAI format (will be converted)
            response_format: Optional Pydantic model (handled via tool pattern)
            system_prompt: System prompt
            **kwargs: Additional parameters

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
        """
        try:
            # Convert OpenAI message format to Anthropic format
            system_messages = []
            filtered_messages = []
            pending_tool_results = []
            
            for msg in messages:
                role = msg.get("role")
                if role == "system":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            block.get("text", "") for block in content 
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    system_messages.append(content)
                elif role == "tool":
                    pending_tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", "")
                    })
                else:
                    if pending_tool_results:
                        filtered_messages.append({"role": "user", "content": pending_tool_results})
                        pending_tool_results = []
                    
                    # Convert assistant messages with tool_calls to Anthropic format
                    converted_msg = {"role": role}
                    content = msg.get("content")
                    tool_calls = msg.get("tool_calls")
                    
                    if role == "assistant" and tool_calls:
                        content_blocks = []
                        if content:
                            if isinstance(content, str):
                                content_blocks.append({"type": "text", "text": content})
                            elif isinstance(content, list):
                                content_blocks.extend(content)
                        for tc in tool_calls:
                            tool_input = tc.get("function", {}).get("arguments", tc.get("arguments", {}))
                            if isinstance(tool_input, str):
                                try:
                                    tool_input = json.loads(tool_input)
                                except (json.JSONDecodeError, TypeError):
                                    tool_input = {}
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("function", {}).get("name", tc.get("name", "")),
                                "input": tool_input
                            })
                        converted_msg["content"] = content_blocks
                    elif content is not None:
                        converted_msg["content"] = content if isinstance(content, (str, list)) else str(content)
                    else:
                        converted_msg["content"] = ""
                    
                    filtered_messages.append(converted_msg)
            
            if pending_tool_results:
                filtered_messages.append({"role": "user", "content": pending_tool_results})
            
            combined_system = "\n\n".join(filter(None, system_messages))
            if system_prompt:
                combined_system = f"{combined_system}\n\n{system_prompt}" if combined_system else system_prompt
            
            api_params = {
                "model": model,
                "messages": filtered_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
            }

            if combined_system:
                api_params["system"] = combined_system

            if "temperature" in kwargs:
                api_params["temperature"] = kwargs["temperature"]

            # Handle structured output via tool_use pattern
            structured_output_tool_name = None
            if response_format is not None and issubclass(response_format, BaseModel):
                structured_output_tool_name = f"output_{response_format.__name__}"
                schema = response_format.model_json_schema()
                structured_tool = {
                    "name": structured_output_tool_name,
                    "description": f"Output structured response as {response_format.__name__}",
                    "input_schema": schema
                }
                api_params["tools"] = [structured_tool]
                api_params["tool_choice"] = {"type": "tool", "name": structured_output_tool_name}
            elif tools:
                api_params["tools"] = self._convert_tools_to_anthropic_format(tools)

            # Attach cache_control markers — see issue #705 and _apply_cache_control.
            api_params = self._apply_cache_control(api_params)
            _ensure_anthropic_beta_header(api_params, _EXTENDED_CACHE_TTL_BETA)

            logger.info(f"Starting Anthropic stream with tools for model: {model}")

            # Accumulators for tool calls
            # Anthropic sends content_block_start with tool info, then input_json_delta for args
            tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
            current_tool_block_index: Optional[int] = None
            text_content = ""
            chunk_count = 0
            input_tokens = None
            output_tokens = None

            async with client.messages.stream(**api_params) as stream:
                async for event in stream:
                    # Handle different event types
                    event_type = getattr(event, 'type', None)

                    # Message start - contains usage info
                    if event_type == 'message_start':
                        if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                            input_tokens = getattr(event.message.usage, 'input_tokens', None)

                    # Message delta - contains output token count at end
                    elif event_type == 'message_delta':
                        if hasattr(event, 'usage'):
                            output_tokens = getattr(event.usage, 'output_tokens', None)

                    # Content block start - marks beginning of text or tool_use block
                    elif event_type == 'content_block_start':
                        if hasattr(event, 'content_block'):
                            block = event.content_block
                            block_index = getattr(event, 'index', 0)

                            if block.type == 'tool_use':
                                # Start accumulating a new tool call
                                tool_calls_accumulator[block_index] = {
                                    "id": block.id,
                                    "name": block.name,
                                    "arguments": ""
                                }
                                current_tool_block_index = block_index

                    # Content block delta - actual content chunks
                    elif event_type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            delta = event.delta
                            delta_type = getattr(delta, 'type', None)

                            if delta_type == 'text_delta':
                                # Text content - yield immediately
                                text = getattr(delta, 'text', '')
                                if text:
                                    chunk_count += 1
                                    text_content += text
                                    yield text

                            elif delta_type == 'input_json_delta':
                                # Tool input JSON chunk - accumulate
                                partial_json = getattr(delta, 'partial_json', '')
                                if partial_json and current_tool_block_index is not None:
                                    if current_tool_block_index in tool_calls_accumulator:
                                        tool_calls_accumulator[current_tool_block_index]["arguments"] += partial_json

                    # Content block stop - marks end of a block
                    elif event_type == 'content_block_stop':
                        # Reset current tool block when it ends
                        current_tool_block_index = None

            logger.info(f"Stream completed. Text chunks: {chunk_count}, Tool calls: {len(tool_calls_accumulator)}")

            # If we have tool calls, yield a final LLMResponse with assembled tool calls
            if tool_calls_accumulator:
                parsed_tool_calls = []
                for idx in sorted(tool_calls_accumulator.keys()):
                    tc_data = tool_calls_accumulator[idx]

                    # If this is our structured output tool, handle it specially
                    if structured_output_tool_name and tc_data["name"] == structured_output_tool_name:
                        # Return the structured output as text content instead of tool call
                        text_content = tc_data["arguments"]
                        continue

                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {"raw": tc_data["arguments"]}

                    parsed_tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args
                    ))

                # Calculate total tokens
                total_tokens = None
                if input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens

                # Only yield LLMResponse if we have actual tool calls (not structured output)
                if parsed_tool_calls:
                    yield LLMResponse(
                        content=text_content if text_content else None,
                        tool_calls=parsed_tool_calls,
                        raw=None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                    )
                elif text_content and structured_output_tool_name:
                    # Structured output - yield the JSON as final text
                    yield text_content

        except Exception as e:
            logger.error(f"Anthropic streaming with tools failed: {e}", exc_info=True)
            raise

    async def continue_with_tool_results(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Continue conversation after executing tool calls.

        Anthropic expects tool results in a specific format with tool_use_id.
        """
        extended_messages = messages.copy()

        # Anthropic expects a user message with tool_result blocks
        tool_result_content = []
        for result in tool_results:
            tool_result_content.append({
                "type": "tool_result",
                "tool_use_id": result["tool_call_id"],
                "content": result["content"]
            })

        extended_messages.append({
            "role": "user",
            "content": tool_result_content
        })

        return await self.get_response(
            client=client,
            model=model,
            messages=extended_messages,
            tools=tools,
            system_prompt=system_prompt,
            **kwargs
        )

    async def list_models(self) -> List[ModelInfo]:
        """
        List available models from Anthropic API.

        Uses the /v1/models endpoint to discover available Claude models.
        See: https://docs.anthropic.com/en/api/models

        Returns:
            List of ModelInfo objects for each available model
        """
        try:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set, returning empty model list")
                return []

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    timeout=HTTP_TIMEOUT_DEFAULT
                )
                response.raise_for_status()
                data = response.json()

            models = []
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                display_name = model_data.get("display_name", model_id)

                # Anthropic's /v1/models returns the input window as
                # ``max_input_tokens``. Without reading it, every model
                # gets ``context_limit=None`` and ``register_discovered_limits``
                # skips it — lookups then fall to DEFAULT_CONTEXT_LIMIT=32768,
                # which surfaced as bogus "Context 100% full" warnings on
                # Opus 4.7's 1M-token window.
                context_limit = model_data.get("max_input_tokens")

                # All Anthropic models are chat models
                models.append(ModelInfo(
                    id=model_id,
                    provider="anthropic",
                    display_name=display_name,
                    category=ModelCategory.CHAT,
                    created_at=model_data.get("created_at"),
                    context_limit=context_limit,
                    supports_vision=True,  # Claude 3+ supports vision
                    supports_tools=True,   # Claude supports tools
                    supports_streaming=True,
                ))

            logger.info(f"Anthropic returned {len(models)} models")
            return models

        except httpx.HTTPStatusError as e:
            logger.error(f"Anthropic API HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Anthropic API connection error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to list Anthropic models: {e}", exc_info=True)
            return []
