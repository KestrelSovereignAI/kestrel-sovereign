"""
Codex Provider Adapter (OpenAI ChatGPT Backend)

Adapter for using OpenAI models via ChatGPT Plus/Pro subscription OAuth,
hitting the same private backend API that the Codex CLI and OpenClaw use:
``https://chatgpt.com/backend-api/codex/responses``

This is the OpenAI equivalent of ClaudeMaxAdapter — subscription-included
usage, not API key billing.

The protocol is the standard OpenAI Responses API but served from the
ChatGPT backend with OAuth Bearer auth + chatgpt-account-id header.

Requirements:
- Active ChatGPT Plus/Pro subscription
- `codex login` completed (stores token in ~/.codex/auth.json)
  OR CODEX_AUTH_TOKEN env var set
"""
import base64
import hashlib
import json
import logging
import platform
from typing import Any, Dict, List, Optional, AsyncIterator, Tuple, Type, Union

import httpx
from pydantic import BaseModel

from .adapter import LLMAdapter, LLMResponse, ToolCall
from .continuation_store import (
    ContinuationCursor,
    ContinuationStore,
    InMemoryContinuationStore,
)
from .gpt5_overlay import prepend_gpt5_overlay
from .model_metadata import ModelInfo
logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _extract_account_id(token: str) -> str:
    """Extract chatgpt_account_id from the JWT access token claims."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT token")
        # Decode JWT payload (base64url)
        payload_b64 = parts[1]
        # Add padding
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        account_id = payload.get(JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not account_id:
            raise ValueError("No chatgpt_account_id in token claims")
        return account_id
    except Exception as e:
        raise ValueError(f"Failed to extract account ID from token: {e}") from e


def _build_headers(token: str, account_id: str) -> dict:
    """Build request headers matching the OpenClaw/Codex protocol."""
    ua = f"kestrel ({platform.system()} {platform.release()}; {platform.machine()})"
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "User-Agent": ua,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }


def _extract_instructions_and_input(messages):
    """Split messages into instructions (system prompt) and input messages."""
    instructions = None
    input_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            instructions = content
        else:
            input_messages.append(msg)
    return instructions, input_messages


def _convert_tools_to_responses_format(tools):
    """Convert OpenAI function calling format to Responses API tool format."""
    if not tools:
        return None
    responses_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            responses_tools.append({
                "type": "function",
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })
    return responses_tools or None


def _content_to_text(content: Any) -> str:
    """Coerce Chat-Completions content (str | list of parts | None) to plain text.

    Image and other non-text parts are dropped. Vision support on the Responses
    API uses different part-type names (``input_text`` / ``input_image``); a
    follow-up will add proper conversion when a vision-capable Codex test case
    exists. See #828 non-goals.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return str(content)


def _convert_messages_to_responses_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Chat-Completions-style messages into Responses API input items.

    The Responses API accepts a flat list of typed items rather than role-tagged
    messages. The orchestrator (``OrchestratorEngineMixin._handle_orchestrator_response``)
    builds messages in Chat Completions format, which is the right normalized
    shape for a provider-agnostic agent loop. Adapter-side translation keeps
    the orchestrator clean.

    Rewriting rules:
    - ``role=user`` and ``role=assistant`` text turns pass through with content
      coerced to a plain string.
    - ``role=assistant`` with ``tool_calls`` is replaced by one
      ``{"type": "function_call", "call_id", "name", "arguments"}`` item per
      tool call. If the assistant message also carried text content, a
      sibling ``{"role": "assistant", "content": ...}`` is emitted first.
    - ``role=tool`` becomes ``{"type": "function_call_output", "call_id", "output"}``.
    - Unknown roles pass through unchanged so we don't silently swallow
      future shapes.

    Fixes #828: the Responses API rejects ``input[*].tool_calls`` and
    ``role=tool`` with ``Unknown parameter`` errors.
    """
    items: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            items.append({"role": "user", "content": _content_to_text(msg.get("content"))})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            text = _content_to_text(msg.get("content"))
            if text:
                items.append({"role": "assistant", "content": text})
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args)
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": args,
                })
            if not tool_calls and not text:
                # Empty assistant message — drop. The Responses API rejects
                # bare assistant items with no content and no function_call.
                continue
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": _content_to_text(msg.get("content")),
            })
        else:
            items.append(msg)
    return items


def _build_request_body(
    model: str,
    input_messages: list,
    instructions: Optional[str] = None,
    tools: Optional[list] = None,
    stream: bool = True,
    previous_response_id: Optional[str] = None,
    **kwargs,
) -> dict:
    """Build Responses API request body matching the ChatGPT backend protocol."""
    body: Dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": stream,
        "input": input_messages,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
    }
    if instructions:
        body["instructions"] = instructions
    if tools:
        body["tools"] = tools
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    if "max_tokens" in kwargs:
        body["max_output_tokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs:
        body["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        body["top_p"] = kwargs["top_p"]
    return body


def _compute_request_signature(
    instructions: Optional[str],
    tools: Optional[list],
) -> str:
    """Stable hash of (instructions, tools) for continuation drift detection.

    If the next turn's signature does not match the cursor's recorded signature,
    the server's prior reasoning was conditioned on a different prompt and
    sending ``previous_response_id`` would either error or produce confused
    output. The adapter drops continuation in that case and resubmits full
    context. See #808.
    """
    canonical = json.dumps(
        {"instructions": instructions or "", "tools": tools or []},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _plan_continuation(
    *,
    cursor: Optional[ContinuationCursor],
    messages_count: int,
    signature: str,
) -> Tuple[Optional[str], Optional[int]]:
    """Decide whether to send a delta turn given the current cursor.

    Returns ``(previous_response_id, slice_start)``:
    - ``previous_response_id`` for the request body, or ``None`` for full input.
    - ``slice_start`` index into the input message list, or ``None`` for full input.

    Branches:
    - No cursor → fresh turn, full input.
    - Signature mismatch → drift (tools or instructions changed mid-conversation),
      drop continuation, full input.
    - Signature match + new messages → delta turn from ``last_message_count``.
    - Signature match + no new messages → defensive full input (avoid empty input).
    """
    if cursor is None:
        return None, None
    if cursor.last_request_signature != signature:
        return None, None
    if messages_count <= cursor.last_message_count:
        return None, None
    return cursor.last_response_id, cursor.last_message_count


async def _parse_sse_events(response: httpx.Response):
    """Parse SSE events from an httpx streaming response."""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


class CodexAdapter(LLMAdapter):
    """
    Adapter for OpenAI Codex subscription via ChatGPT backend.

    Uses httpx to hit chatgpt.com/backend-api/codex/responses with
    OAuth Bearer token + chatgpt-account-id header. The response format
    is the standard OpenAI Responses API SSE stream.
    """

    def __init__(self, continuation_store: Optional[ContinuationStore] = None):
        self.name = "openai_plan"
        # One adapter instance per route (see provider_registry); per-instance
        # continuation state therefore aligns with per-conversation lifetime.
        # Multi-worker uvicorn deployments swap a shared backend in here. #808.
        self._continuation_store: ContinuationStore = (
            continuation_store or InMemoryContinuationStore()
        )

    def contribute_system_prompt(
        self, model_id: str, base: Optional[str]
    ) -> Optional[str]:
        """Inject the GPT-5 behavior contract for gpt-5 family models.

        See #807 / #806. The contract gives GPT-5 the act/ask, tool-discipline,
        and completion semantics that prose-style guidance does not enforce on
        Responses-API models. No-op for non-gpt-5 ids.
        """
        return prepend_gpt5_overlay(base, model_id)

    def _plan_request_continuation(
        self,
        session_id: Optional[str],
        instructions: Optional[str],
        responses_tools: Optional[List[Dict[str, Any]]],
        input_messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        """Decide ``previous_response_id`` and the input slice for this turn.

        Returns ``(previous_response_id, input_to_send, signature)``. When no
        ``session_id`` is given, behavior reduces to ``(None, input_messages,
        signature)`` — i.e. a fresh, stateless call (the existing behavior).
        """
        signature = _compute_request_signature(instructions, responses_tools)
        if not session_id:
            return None, input_messages, signature
        cursor = self._continuation_store.get(self.name, session_id)
        prev_id, slice_start = _plan_continuation(
            cursor=cursor,
            messages_count=len(input_messages),
            signature=signature,
        )
        input_to_send = (
            input_messages if slice_start is None else input_messages[slice_start:]
        )
        return prev_id, input_to_send, signature

    def _record_continuation(
        self,
        session_id: Optional[str],
        response_id: Optional[str],
        full_messages_count: int,
        signature: str,
    ) -> None:
        """Persist the cursor for the next turn after a successful response."""
        if not session_id or not response_id:
            return
        self._continuation_store.put(
            self.name,
            session_id,
            ContinuationCursor(
                last_response_id=response_id,
                last_message_count=full_messages_count,
                last_request_signature=signature,
            ),
        )

    async def get_response(
        self,
        client: Any,  # OAuth token string stored as client
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> LLMResponse:
        """Get a response from the ChatGPT backend Responses API.

        The ChatGPT backend requires stream=true, so we consume the
        SSE stream internally and assemble the final response.
        """
        token = client  # Provider registry stores token as "client"
        if not isinstance(token, str):
            raise RuntimeError(
                "Codex adapter requires an OAuth token. "
                "Run `codex login` or set CODEX_AUTH_TOKEN."
            )

        # Conversation continuation (#808): when a session_id is provided
        # and a cursor exists for it, send only the new input items plus
        # ``previous_response_id`` so the server can preserve encrypted
        # reasoning across turns. Pop from kwargs so it does not leak into the
        # request body builder.
        session_id = kwargs.pop("session_id", None)

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        responses_tools = _convert_tools_to_responses_format(tools)

        prev_response_id, input_to_send, signature = self._plan_request_continuation(
            session_id, instructions, responses_tools, input_messages,
        )
        # Translate Chat-Completions tool_calls / role=tool messages into
        # Responses-API ``function_call`` / ``function_call_output`` items.
        # Done *after* the continuation slice so the watermark
        # (``last_message_count``) continues to track the original
        # Chat-Completions count we slice on. #828.
        input_to_send = _convert_messages_to_responses_format(input_to_send)

        body = _build_request_body(
            model=model,
            input_messages=input_to_send,
            instructions=instructions,
            tools=responses_tools,
            stream=True,  # ChatGPT backend requires streaming
            previous_response_id=prev_response_id,
            **kwargs,
        )

        # Consume SSE stream and assemble final response
        content_parts: List[str] = []
        parsed_tool_calls = None
        func_calls: Dict[int, Dict[str, str]] = {}
        final_usage: Dict[str, Any] = {}
        last_response_id: Optional[str] = None

        async with httpx.AsyncClient(timeout=120) as http:
            async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    error_text = resp.text[:500]
                    logger.error(f"Codex API error {resp.status_code}: {error_text}")
                    raise RuntimeError(
                        f"Codex API returned {resp.status_code}: {error_text}"
                    )

                async for event in _parse_sse_events(resp):
                    event_type = event.get("type", "")

                    if event_type == "response.output_text.delta":
                        content_parts.append(event.get("delta", ""))

                    elif event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            idx = event.get("output_index", 0)
                            func_calls[idx] = {
                                "id": item.get("id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                            }

                    elif event_type == "response.function_call_arguments.delta":
                        idx = event.get("output_index", 0)
                        if idx not in func_calls:
                            func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        func_calls[idx]["arguments"] += event.get("delta", "")

                    elif event_type == "response.function_call_arguments.done":
                        idx = event.get("output_index", 0)
                        if idx not in func_calls:
                            func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        func_calls[idx]["arguments"] = event.get("arguments", "")

                    elif event_type == "response.completed":
                        resp_data = event.get("response", {})
                        final_usage = resp_data.get("usage", {})
                        last_response_id = resp_data.get("id") or last_response_id

        # Persist cursor on success so the next turn for this conversation
        # picks up from this response. ``len(input_messages)`` is the *full*
        # message count, not the slice — the server owns history; we just
        # track the watermark from which to send deltas next time.
        self._record_continuation(
            session_id, last_response_id, len(input_messages), signature,
        )

        content = "".join(content_parts) if content_parts else None

        if func_calls:
            parsed_tool_calls = []
            for idx in sorted(func_calls.keys()):
                fc = func_calls[idx]
                try:
                    args = json.loads(fc["arguments"]) if fc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"raw": fc["arguments"]}
                parsed_tool_calls.append(ToolCall(
                    id=fc["id"],
                    name=fc["name"],
                    arguments=args,
                ))

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            raw=None,
            input_tokens=final_usage.get("input_tokens"),
            output_tokens=final_usage.get("output_tokens"),
            total_tokens=final_usage.get("total_tokens"),
        )

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Get a streaming response from the ChatGPT backend."""
        token = client
        if not isinstance(token, str):
            raise RuntimeError("Codex adapter requires an OAuth token.")

        # Conversation continuation (#808). See ``get_response`` for rationale.
        session_id = kwargs.pop("session_id", None)

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        # Tools aren't passed on the text-only stream path, but signature still
        # includes them as ``[]`` so a downstream switch to the tool path with
        # the same session_id correctly invalidates continuation.
        prev_response_id, input_to_send, signature = self._plan_request_continuation(
            session_id, instructions, None, input_messages,
        )
        input_to_send = _convert_messages_to_responses_format(input_to_send)  # #828

        body = _build_request_body(
            model=model,
            input_messages=input_to_send,
            instructions=instructions,
            stream=True,
            previous_response_id=prev_response_id,
            **kwargs,
        )

        logger.info(f"Starting Codex stream for model: {model}")
        chunk_count = 0
        last_response_id: Optional[str] = None

        async with httpx.AsyncClient(timeout=120) as http:
            async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    error_text = resp.text[:500]
                    logger.error(f"Codex stream error {resp.status_code}: {error_text}")
                    raise RuntimeError(
                        f"Codex API returned {resp.status_code}: {error_text}"
                    )

                async for event in _parse_sse_events(resp):
                    event_type = event.get("type", "")
                    if event_type == "response.output_text.delta":
                        chunk_count += 1
                        yield event.get("delta", "")
                    elif event_type == "response.completed":
                        resp_data = event.get("response", {})
                        last_response_id = resp_data.get("id") or last_response_id

        self._record_continuation(
            session_id, last_response_id, len(input_messages), signature,
        )
        logger.info(f"Codex stream completed. Total chunks: {chunk_count}")

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Stream response with tool call detection."""
        token = client
        if not isinstance(token, str):
            raise RuntimeError("Codex adapter requires an OAuth token.")

        # Conversation continuation (#808). See ``get_response`` for rationale.
        session_id = kwargs.pop("session_id", None)

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        responses_tools = _convert_tools_to_responses_format(tools)

        prev_response_id, input_to_send, signature = self._plan_request_continuation(
            session_id, instructions, responses_tools, input_messages,
        )
        input_to_send = _convert_messages_to_responses_format(input_to_send)  # #828

        body = _build_request_body(
            model=model,
            input_messages=input_to_send,
            instructions=instructions,
            tools=responses_tools,
            stream=True,
            previous_response_id=prev_response_id,
            **kwargs,
        )

        logger.info(f"Starting Codex stream with tools for model: {model}")
        text_content = ""
        chunk_count = 0
        func_calls: Dict[int, Dict[str, str]] = {}
        final_usage: Dict[str, Any] = {}
        last_response_id: Optional[str] = None

        async with httpx.AsyncClient(timeout=120) as http:
            async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    error_text = resp.text[:500]
                    raise RuntimeError(
                        f"Codex API returned {resp.status_code}: {error_text}"
                    )

                async for event in _parse_sse_events(resp):
                    event_type = event.get("type", "")

                    if event_type == "response.output_text.delta":
                        chunk_count += 1
                        delta = event.get("delta", "")
                        text_content += delta
                        yield delta

                    elif event_type == "response.function_call_arguments.delta":
                        idx = event.get("output_index", 0)
                        if idx not in func_calls:
                            func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        func_calls[idx]["arguments"] += event.get("delta", "")

                    elif event_type == "response.function_call_arguments.done":
                        idx = event.get("output_index", 0)
                        if idx not in func_calls:
                            func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        func_calls[idx]["arguments"] = event.get("arguments", "")

                    elif event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            idx = event.get("output_index", 0)
                            func_calls[idx] = {
                                "id": item.get("id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                            }

                    elif event_type == "response.completed":
                        resp_data = event.get("response", {})
                        usage = resp_data.get("usage", {})
                        final_usage = usage
                        last_response_id = resp_data.get("id") or last_response_id

        self._record_continuation(
            session_id, last_response_id, len(input_messages), signature,
        )

        # Yield final tool call response if any
        if func_calls:
            parsed_tool_calls = []
            for idx in sorted(func_calls.keys()):
                fc = func_calls[idx]
                try:
                    args = json.loads(fc["arguments"]) if fc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"raw": fc["arguments"]}
                parsed_tool_calls.append(ToolCall(
                    id=fc["id"],
                    name=fc["name"],
                    arguments=args,
                ))

            yield LLMResponse(
                content=text_content if text_content else None,
                tool_calls=parsed_tool_calls,
                raw=None,
                input_tokens=final_usage.get("input_tokens"),
                output_tokens=final_usage.get("output_tokens"),
                total_tokens=final_usage.get("total_tokens"),
            )

        logger.info(
            f"Codex stream completed. Text chunks: {chunk_count}, "
            f"Tool calls: {len(func_calls)}"
        )

    async def list_models(self) -> List[ModelInfo]:
        """OpenAI plan uses canonical OpenAI discovery; this wrapper has no catalog."""
        raise NotImplementedError(
            "OpenAI plan model discovery is provided by the canonical openai provider."
        )
