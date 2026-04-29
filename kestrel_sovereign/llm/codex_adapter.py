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


def _messages_have_tool_history(messages: List[Dict[str, Any]]) -> bool:
    """True if any ``role=assistant`` message carries ``tool_calls``.

    Marker for ""this input represents an agent loop already in progress""
    versus ""this input is a fresh user message starting a new loop"". Used
    to decide whether replaying a session's cached reasoning items is safe
    or whether the cache is from a prior loop and should be cleared (#875).
    """
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return True
    return False


def _content_to_text(content: Any) -> str:
    """Coerce Chat-Completions content (str | list of parts | None) to plain text.

    Non-text parts (e.g. ``image_url``) are dropped with a warning so the gap
    is visible in logs rather than silently producing text-only output.
    Multimodal Codex is tracked separately as a follow-up; this function is
    deliberately the single chokepoint where the drop happens, so when the
    follow-up lands the change is local.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        dropped: List[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, dict):
                dropped.append(p.get("type") or "<no-type>")
            else:
                dropped.append(type(p).__name__)
        if dropped:
            logger.warning(
                "CodexAdapter: dropping non-text content parts %s — "
                "multimodal Codex not yet supported (#847).",
                dropped,
            )
        return "".join(parts)
    raise TypeError(
        f"CodexAdapter: unsupported content type {type(content).__name__}; "
        "expected str, list of typed parts, or None."
    )


def _convert_messages_to_responses_format(
    messages: List[Dict[str, Any]],
    cached_turn_outputs: Optional[List[List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
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
    - Unknown roles raise ``ValueError`` at the adapter boundary rather than
      forwarding to the wire. A server 400 there is opaque ("Unknown
      parameter") and discovered post-network; the adapter is the right place
      to fail fast with a clear message naming the offending role.

    Reasoning replay (#842 + #875): when ``cached_turn_outputs`` is supplied
    the converter looks up cached ``function_call`` items by ``call_id`` and
    replays them — preserving the original ``encrypted_content`` of any
    reasoning items that accompanied them. Lookup is **id-based**, not
    positional: only cached function_calls whose ``call_id`` appears in the
    current input's orchestrator ``tool_calls`` are emitted. Stale cache
    entries (call_ids no longer in the message list — typical when the
    cache accumulated across separate agent loops sharing one ``session_id``)
    are silently ignored, so they never inject orphan function_call items
    that would trigger ""No tool call found for function call output""
    errors on the wire. See #875 for the position-vs-id failure mode.

    Reasoning items are emitted once per turn group, before the first
    matching function_call from that turn. A turn's reasoning is only
    replayed when at least one of that turn's function_calls also matches —
    i.e. encrypted reasoning rides along with the tool exchange it was
    bound to.

    Fixes #828 (unknown-parameter errors), #842 (reasoning continuity),
    and #875 (cache-scope mismatch).
    """
    cached_turn_outputs = cached_turn_outputs or []

    # Build call_id → (turn_idx, fc_item). The cache may hold many turns from
    # the session's history; we'll only emit the ones whose call_id matches
    # an orchestrator tool_call in the current input.
    fc_by_call_id: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    turn_reasoning: Dict[int, List[Dict[str, Any]]] = {}
    for turn_idx, turn in enumerate(cached_turn_outputs):
        turn_reasoning[turn_idx] = [
            it for it in turn if it.get("type") == "reasoning"
        ]
        for it in turn:
            if it.get("type") == "function_call":
                cid = it.get("call_id")
                if cid:
                    fc_by_call_id[cid] = (turn_idx, it)

    items: List[Dict[str, Any]] = []
    emitted_turn_reasoning: set = set()  # turn indices whose reasoning is already emitted
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            items.append({"role": "user", "content": _content_to_text(msg.get("content"))})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            text = _content_to_text(msg.get("content"))
            empty = not tool_calls and not text
            if empty:
                # Empty assistant message — drop. The Responses API rejects
                # bare assistant items with no content and no function_call.
                continue
            if text:
                items.append({"role": "assistant", "content": text})
            for tc in tool_calls:
                cid = tc.get("id") or ""
                cache_hit = fc_by_call_id.get(cid)
                if cache_hit is not None:
                    turn_idx, cached_fc = cache_hit
                    # Emit this turn's reasoning items once, just before the
                    # first matching function_call from that turn.
                    if turn_idx not in emitted_turn_reasoning:
                        for r in turn_reasoning.get(turn_idx, []):
                            items.append(r)
                        emitted_turn_reasoning.add(turn_idx)
                    items.append(cached_fc)
                else:
                    # No cache match — synthesize the function_call from the
                    # orchestrator's tool_call. This is the post-#828 path
                    # for fresh tool exchanges (no prior reasoning to replay).
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", "")
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    items.append({
                        "type": "function_call",
                        "call_id": cid,
                        "name": fn.get("name", ""),
                        "arguments": args,
                    })
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": _content_to_text(msg.get("content")),
            })
        else:
            raise ValueError(
                f"CodexAdapter received unsupported message role {role!r}; "
                "the orchestrator should produce only system/user/assistant/tool. "
                "Failing at the adapter boundary so the upstream caller is "
                "obvious — a server 400 from forwarding the message would be "
                "opaque and post-network."
            )
    return items


def _build_request_body(
    model: str,
    input_messages: list,
    instructions: Optional[str] = None,
    tools: Optional[list] = None,
    stream: bool = True,
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
    """Stable hash of (instructions, tools) recorded with the cursor.

    Originally designed for #808's drift detection on ``previous_response_id``
    continuation. The ChatGPT-backend Responses API rejects that parameter
    (#841), so the signature is no longer used to *gate* continuation — it's
    kept as a per-session record for diagnostics and for any future backend
    that supports reasoning-item resubmission.
    """
    canonical = json.dumps(
        {"instructions": instructions or "", "tools": tools or []},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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
        # Explicit ``is not None`` check, NOT ``or``: ``InMemoryContinuationStore``
        # defines ``__len__``, so an empty caller-supplied store evaluates falsy
        # under ``or`` and gets silently discarded. The bug shipped in PR #811
        # and was caught only by the live test added in #841 — unit tests
        # passed because they read ``adapter._continuation_store`` (the
        # internal one), not the external store passed in.
        self._continuation_store: ContinuationStore = (
            continuation_store
            if continuation_store is not None
            else InMemoryContinuationStore()
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

    def _load_replay_outputs(
        self,
        session_id: Optional[str],
        signature: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Load cached output items per turn for replay (#842 + #875).

        Returns a list of per-turn output-item lists. Empty list when:
        - No ``session_id``.
        - No cursor exists for the session.
        - Signature drift (cached reasoning was conditioned on different
          ``(instructions, tools)``).
        - The input ``messages`` show no tool history — i.e. the orchestrator
          is starting a fresh agent loop. In that case the cache from the
          prior loop is stale and we clear it (#875). Without this guard the
          cache grows unboundedly and stale function_call items get injected
          into unrelated agent loops.
        """
        if not session_id:
            return []
        cursor = self._continuation_store.get(self.name, session_id)
        if cursor is None:
            return []
        if cursor.last_request_signature != signature:
            # Drift: do not replay reasoning bound to a different prompt.
            return []
        # Fresh-loop guard: if the current input has no assistant.tool_calls
        # messages, the prior cache cannot apply (no function_calls to match
        # against). Clear it so it doesn't accumulate across user messages.
        if messages is not None and not _messages_have_tool_history(messages):
            self._continuation_store.clear(self.name, session_id)
            return []
        try:
            return [json.loads(s) for s in cursor.turn_outputs]
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse cached turn_outputs for session %s; dropping",
                session_id,
            )
            return []

    def _record_continuation(
        self,
        session_id: Optional[str],
        response_id: Optional[str],
        full_messages_count: int,
        signature: str,
        new_turn_outputs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist the cursor for the next turn after a successful response.

        ``new_turn_outputs`` is the list of output items emitted by the model
        on THIS turn (typically reasoning + function_call items). Appended to
        any prior cached turns so subsequent calls can replay them as input.
        See #842.
        """
        if not session_id or not response_id:
            return
        prior_outputs: Tuple[str, ...] = ()
        existing = self._continuation_store.get(self.name, session_id)
        if existing is not None and existing.last_request_signature == signature:
            prior_outputs = existing.turn_outputs
        new_outputs = prior_outputs
        if new_turn_outputs:
            new_outputs = (*prior_outputs, json.dumps(new_turn_outputs))
        self._continuation_store.put(
            self.name,
            session_id,
            ContinuationCursor(
                last_response_id=response_id,
                last_message_count=full_messages_count,
                last_request_signature=signature,
                turn_outputs=new_outputs,
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

        session_id = kwargs.pop("session_id", None)

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        responses_tools = _convert_tools_to_responses_format(tools)

        # The ChatGPT backend rejects ``previous_response_id`` (#841). Reasoning
        # continuity is preserved instead by capturing the model's per-turn
        # output items (reasoning + function_call) and replaying them as input
        # items on subsequent turns — see #842.
        signature = _compute_request_signature(instructions, responses_tools)
        cached_outputs = self._load_replay_outputs(session_id, signature, messages=input_messages)
        input_to_send = _convert_messages_to_responses_format(
            input_messages, cached_turn_outputs=cached_outputs,
        )

        body = _build_request_body(
            model=model,
            input_messages=input_to_send,
            instructions=instructions,
            tools=responses_tools,
            stream=True,  # ChatGPT backend requires streaming
            **kwargs,
        )

        # Consume SSE stream and assemble final response
        content_parts: List[str] = []
        parsed_tool_calls = None
        func_calls: Dict[int, Dict[str, str]] = {}
        final_usage: Dict[str, Any] = {}
        last_response_id: Optional[str] = None
        # Capture the model's output items for this turn so the next call can
        # replay encrypted reasoning + function_calls as input items (#842).
        new_turn_outputs: List[Dict[str, Any]] = []

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
                            # Capture ``call_id`` (the tool-call id used by
                            # function_call_output to match against), NOT the
                            # output-item ``id``. Live capture (#857) shows
                            # both fields are present on this event. The
                            # Responses API matches function_call ↔
                            # function_call_output by ``call_id``; using
                            # ``id`` produces 400 ""No tool output found""
                            # on the replay path because the cached
                            # function_call carries the real ``call_id`` and
                            # the orchestrator's tool_call_id (set from this
                            # ToolCall.id) carried the wrong field.
                            func_calls[idx] = {
                                "id": item.get("call_id") or item.get("id", ""),
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

                    elif event_type == "response.output_item.done":
                        item = event.get("item", {})
                        if item.get("type") in ("reasoning", "function_call", "message"):
                            new_turn_outputs.append(item)

                    elif event_type == "response.completed":
                        resp_data = event.get("response", {})
                        final_usage = resp_data.get("usage", {})
                        last_response_id = resp_data.get("id") or last_response_id

        # Persist cursor + this turn's output items so the next call replays
        # them as input. ``len(input_messages)`` is the Chat-Completions count
        # after system extraction.
        self._record_continuation(
            session_id, last_response_id, len(input_messages), signature,
            new_turn_outputs=new_turn_outputs,
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

        session_id = kwargs.pop("session_id", None)

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        # See ``get_response`` for the full rationale on #841 and #842.
        signature = _compute_request_signature(instructions, None)
        cached_outputs = self._load_replay_outputs(session_id, signature, messages=input_messages)
        input_to_send = _convert_messages_to_responses_format(
            input_messages, cached_turn_outputs=cached_outputs,
        )

        body = _build_request_body(
            model=model,
            input_messages=input_to_send,
            instructions=instructions,
            stream=True,
            **kwargs,
        )

        logger.info(f"Starting Codex stream for model: {model}")
        chunk_count = 0
        last_response_id: Optional[str] = None
        new_turn_outputs: List[Dict[str, Any]] = []

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
                    elif event_type == "response.output_item.done":
                        item = event.get("item", {})
                        if item.get("type") in ("reasoning", "function_call", "message"):
                            new_turn_outputs.append(item)
                    elif event_type == "response.completed":
                        resp_data = event.get("response", {})
                        last_response_id = resp_data.get("id") or last_response_id

        self._record_continuation(
            session_id, last_response_id, len(input_messages), signature,
            new_turn_outputs=new_turn_outputs,
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

        session_id = kwargs.pop("session_id", None)

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        responses_tools = _convert_tools_to_responses_format(tools)

        # See ``get_response`` for the full rationale on #841 and #842.
        signature = _compute_request_signature(instructions, responses_tools)
        cached_outputs = self._load_replay_outputs(session_id, signature, messages=input_messages)
        input_to_send = _convert_messages_to_responses_format(
            input_messages, cached_turn_outputs=cached_outputs,
        )

        body = _build_request_body(
            model=model,
            input_messages=input_to_send,
            instructions=instructions,
            tools=responses_tools,
            stream=True,
            **kwargs,
        )

        logger.info(f"Starting Codex stream with tools for model: {model}")
        text_content = ""
        chunk_count = 0
        func_calls: Dict[int, Dict[str, str]] = {}
        final_usage: Dict[str, Any] = {}
        last_response_id: Optional[str] = None
        new_turn_outputs: List[Dict[str, Any]] = []

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
                            # Capture ``call_id`` (the tool-call id used by
                            # function_call_output to match against), NOT the
                            # output-item ``id``. Live capture (#857) shows
                            # both fields are present on this event. The
                            # Responses API matches function_call ↔
                            # function_call_output by ``call_id``; using
                            # ``id`` produces 400 ""No tool output found""
                            # on the replay path because the cached
                            # function_call carries the real ``call_id`` and
                            # the orchestrator's tool_call_id (set from this
                            # ToolCall.id) carried the wrong field.
                            func_calls[idx] = {
                                "id": item.get("call_id") or item.get("id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                            }

                    elif event_type == "response.output_item.done":
                        item = event.get("item", {})
                        if item.get("type") in ("reasoning", "function_call", "message"):
                            new_turn_outputs.append(item)

                    elif event_type == "response.completed":
                        resp_data = event.get("response", {})
                        usage = resp_data.get("usage", {})
                        final_usage = usage
                        last_response_id = resp_data.get("id") or last_response_id

        self._record_continuation(
            session_id, last_response_id, len(input_messages), signature,
            new_turn_outputs=new_turn_outputs,
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
