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
import asyncio
import base64
import hashlib
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator, Tuple, Type, Union

import httpx
from pydantic import BaseModel

from kestrel_sdk.llm import ToolCallStarted

from .adapter import LLMAdapter, LLMResponse, ThinkingContentSplitter, ThinkingDelta, ToolCall
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

# OAuth refresh endpoint + the codex CLI's published OAuth client_id (the
# ``aud`` claim of the id_token in ``~/.codex/auth.json``). Used by
# ``_refresh_codex_oauth_token`` when our access_token has expired and the
# auth-file copy is also stale. See #887 for why the adapter handles this
# itself rather than letting a transient ``token_expired`` 401 disable the
# route for the rest of the session.
OPENAI_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH_FILE = Path.home() / ".codex" / "auth.json"


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


def _read_codex_auth_file() -> Optional[Dict[str, Any]]:
    """Return the parsed contents of ``~/.codex/auth.json`` or None.

    Used both to pick up tokens refreshed externally (e.g. by the codex CLI
    or its companion runtime) and to source the ``refresh_token`` when we
    refresh ourselves. See #887.
    """
    try:
        return json.loads(CODEX_AUTH_FILE.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read {CODEX_AUTH_FILE}: {e}")
        return None


def _access_token_from_auth_data(data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract the access_token from the auth.json shape produced by ``codex login``.

    Tolerant of the two shapes seen in the wild:

      {"tokens": {"access_token": ...}, "auth_mode": ""oauth"", ...}
      {"access_token": ..., ...}
    """
    if not data:
        return None
    tokens = data.get("tokens") or {}
    return tokens.get("access_token") or data.get("access_token")


def _write_codex_auth_file(data: Dict[str, Any]) -> None:
    """Write the auth.json back atomically. Preserves any unknown top-level
    fields the codex CLI may have set so its own state isn't clobbered.
    """
    CODEX_AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CODEX_AUTH_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, CODEX_AUTH_FILE)


async def _refresh_codex_oauth_token(refresh_token: str) -> Dict[str, Any]:
    """Exchange ``refresh_token`` for fresh tokens via OpenAI's OAuth endpoint.

    Returns the raw token response (typically ``{access_token, id_token,
    refresh_token, expires_in, ...}``). Raises ``RuntimeError`` on any non-200
    so callers can fall through to ""truly permanent auth failure"" handling.
    """
    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CODEX_CLI_CLIENT_ID,
    }
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            OPENAI_OAUTH_TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"OAuth refresh failed with {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()


def _persist_refreshed_tokens(token_response: Dict[str, Any]) -> None:
    """Merge the refresh response back into auth.json under the ``tokens`` key.

    Best-effort: failure to write doesn't prevent using the new token in
    memory — the next process restart will fall back to the (still-stale)
    file copy and refresh again, which is fine.
    """
    data = _read_codex_auth_file() or {}
    tokens = dict(data.get("tokens") or {})
    if "access_token" in token_response:
        tokens["access_token"] = token_response["access_token"]
    if "id_token" in token_response:
        tokens["id_token"] = token_response["id_token"]
    # Refresh tokens may rotate — keep the new one if present, else preserve.
    if token_response.get("refresh_token"):
        tokens["refresh_token"] = token_response["refresh_token"]
    data["tokens"] = tokens
    data["last_refresh"] = datetime.now(timezone.utc).isoformat()
    try:
        _write_codex_auth_file(data)
    except OSError as e:
        logger.warning(f"Failed to persist refreshed Codex tokens to disk: {e}")


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

    Reasoning replay (#842 + #875). Cached turns are split into two classes:

    - **Tool turns** (cached output contains at least one ``function_call``):
      replayed by **id-match** against the orchestrator's
      ``assistant.tool_calls``. Only cached ``function_call`` items whose
      ``call_id`` appears in the current input are emitted. Reasoning items
      from a tool turn ride along, emitted once when the first matching
      function_call from that turn is replayed. Stale cached function_calls
      (call_ids absent from the current input — typical when the cache
      accumulated across separate agent loops on one ``session_id``) are
      silently skipped, never producing orphan items on the wire (#875).

    - **Text turns** (cached output has reasoning and/or message items but
      no function_call): replayed **positionally** against text-only
      ``role=assistant`` messages. This preserves #842's encrypted-reasoning
      continuity for plain text follow-ups (e.g. ""multiply 17 × 23"" → ""now
      multiply by 2""). Position-match is anchored on the chronological
      order of text-only turns in both cache and input; the orchestrator's
      storage preserves that order.

    Fixes #828 (unknown-parameter errors), #842 (reasoning continuity for
    both text and tool turns), and #875 (cross-loop cache-scope mismatch).
    """
    cached_turn_outputs = cached_turn_outputs or []

    # Classify each cached turn. ``fc_by_call_id`` powers tool-turn id-match;
    # ``text_only_turns`` powers text-turn position-match.
    fc_by_call_id: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    turn_reasoning: Dict[int, List[Dict[str, Any]]] = {}
    text_only_turns: List[List[Dict[str, Any]]] = []
    for turn_idx, turn in enumerate(cached_turn_outputs):
        turn_reasoning[turn_idx] = [
            it for it in turn if it.get("type") == "reasoning"
        ]
        has_fc = False
        for it in turn:
            if it.get("type") == "function_call":
                has_fc = True
                cid = it.get("call_id")
                if cid:
                    fc_by_call_id[cid] = (turn_idx, it)
        if not has_fc:
            # Reasoning + message items, no function_calls. Position-anchored
            # against text-only assistant messages in the input.
            text_only_turns.append([
                it for it in turn if it.get("type") in ("reasoning", "message")
            ])

    items: List[Dict[str, Any]] = []
    emitted_turn_reasoning: set = set()  # tool-turn indices already replayed
    text_turn_idx = 0
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
            if tool_calls:
                # Tool turn: id-match cached function_calls.
                if text:
                    items.append({"role": "assistant", "content": text})
                for tc in tool_calls:
                    cid = tc.get("id") or ""
                    cache_hit = fc_by_call_id.get(cid)
                    if cache_hit is not None:
                        turn_idx, cached_fc = cache_hit
                        if turn_idx not in emitted_turn_reasoning:
                            for r in turn_reasoning.get(turn_idx, []):
                                items.append(r)
                            emitted_turn_reasoning.add(turn_idx)
                        items.append(cached_fc)
                    else:
                        # Cache miss — synthesize from the orchestrator's tc.
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
            else:
                # Text turn: position-match against cached text-only turns,
                # so reasoning + message items from a prior text response
                # ride along on the next text follow-up. If the cache has
                # no entry at this position (fresh conversation, or more
                # text turns in input than cache), fall back to the literal
                # message text.
                if text_turn_idx < len(text_only_turns):
                    items.extend(text_only_turns[text_turn_idx])
                else:
                    items.append({"role": "assistant", "content": text})
                text_turn_idx += 1
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
        # Cached access token: when we refresh on 401, subsequent calls use
        # the new token without re-reading auth.json or hitting OAuth again.
        # The provider registry supplied the original token via the ``client``
        # parameter; this overrides it once we know it's stale. See #887.
        self._refreshed_token: Optional[str] = None
        # Serialize concurrent refresh attempts. Multiple in-flight calls all
        # 401-ing at once would otherwise hammer the OAuth endpoint with the
        # same refresh request.
        self._refresh_lock = asyncio.Lock()

    def _current_token(self, fallback: str) -> str:
        """Return the latest known access token (refresh override or fallback)."""
        return self._refreshed_token or fallback

    async def _refresh_token_if_possible(self, current_token: str) -> Optional[str]:
        """Try to recover a working access token after a 401. Returns the new
        token or None if recovery isn't possible (no refresh_token, or the
        OAuth endpoint itself rejected the refresh — i.e. truly permanent).

        Two-step recovery, ordered cheap → expensive:

        1. Re-read ``~/.codex/auth.json``. If the file's access_token is
           newer than the one we got the 401 with, the codex CLI (or a peer
           process) refreshed it externally — adopt and retry.
        2. Otherwise, perform the OAuth ``refresh_token`` grant ourselves,
           write the new tokens back to disk, and return the new
           access_token.

        Caller serializes via ``self._refresh_lock`` so concurrent 401s
        produce one refresh, not N.
        """
        async with self._refresh_lock:
            # Another in-flight call may have already refreshed while we
            # were waiting on the lock. Pick that up instead of redoing.
            if self._refreshed_token and self._refreshed_token != current_token:
                return self._refreshed_token

            file_data = _read_codex_auth_file()
            file_token = _access_token_from_auth_data(file_data)
            if file_token and file_token != current_token:
                logger.info(
                    "Codex token refreshed externally; adopting from auth.json"
                )
                self._refreshed_token = file_token
                return file_token

            rt = (file_data or {}).get("tokens", {}).get("refresh_token")
            if not rt:
                logger.warning(
                    "Codex 401 with no refresh_token in auth.json — "
                    "cannot recover; treating as permanent."
                )
                return None
            try:
                token_response = await _refresh_codex_oauth_token(rt)
            except Exception as e:
                logger.warning(f"Codex OAuth refresh failed: {e}")
                return None
            new_token = token_response.get("access_token")
            if not new_token:
                logger.warning("OAuth refresh response had no access_token field")
                return None
            _persist_refreshed_tokens(token_response)
            self._refreshed_token = new_token
            logger.info("Codex access token refreshed via OAuth")
            return new_token

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
    ) -> List[List[Dict[str, Any]]]:
        """Load cached output items per turn for replay (#842 + #875).

        Returns a list of per-turn output-item lists. Empty when no session,
        no cursor, or signature drift (cached reasoning was conditioned on a
        different ``(instructions, tools)``).

        Stale entries from prior agent loops are NOT cleared here. The
        converter's id-based matching for ``function_call`` items handles
        cross-loop safety: cached fc's whose ``call_id`` isn't in the
        current input's ``tool_calls`` are silently skipped, so they can't
        produce orphan function_call items on the wire (#875). Reasoning
        items from prior text-only turns continue to replay positionally
        against assistant text messages, preserving #842's reasoning
        continuity for text turns.
        """
        if not session_id:
            return []
        cursor = self._continuation_store.get(self.name, session_id)
        if cursor is None:
            return []
        if cursor.last_request_signature != signature:
            # Drift: do not replay reasoning bound to a different prompt.
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
        # Adopt any token a previous call refreshed in-memory.
        token = self._current_token(token)

        session_id = kwargs.pop("session_id", None)

        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        responses_tools = _convert_tools_to_responses_format(tools)

        # The ChatGPT backend rejects ``previous_response_id`` (#841). Reasoning
        # continuity is preserved instead by capturing the model's per-turn
        # output items (reasoning + function_call) and replaying them as input
        # items on subsequent turns — see #842.
        signature = _compute_request_signature(instructions, responses_tools)
        cached_outputs = self._load_replay_outputs(session_id, signature)
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

        # Two attempts: the first uses our current token, the second uses a
        # refreshed token if the first 401'd. See #887 — ``token_expired``
        # is transient and recovers via ``_refresh_token_if_possible``;
        # only after refresh fails do we treat the 401 as permanent.
        async with httpx.AsyncClient(timeout=120) as http:
            for attempt in range(2):
                account_id = _extract_account_id(token)
                headers = _build_headers(token, account_id)
                async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                    if resp.status_code == 401 and attempt == 0:
                        await resp.aread()
                        new_token = await self._refresh_token_if_possible(token)
                        if new_token and new_token != token:
                            token = new_token
                            continue  # retry once with refreshed token
                        # Refresh didn't help — propagate as before.
                        error_text = resp.text[:500]
                        logger.error(f"Codex API error 401: {error_text}")
                        raise RuntimeError(f"Codex API returned 401: {error_text}")
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
                    break  # successful stream consumed; exit retry loop

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
                    # SDK 0.7.0 malformed-JSON sentinel.
                    args = {"_raw": fc["arguments"]}
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
    ) -> AsyncIterator[Union[str, ThinkingDelta]]:
        """Get a streaming response from the ChatGPT backend."""
        token = client
        if not isinstance(token, str):
            raise RuntimeError("Codex adapter requires an OAuth token.")
        token = self._current_token(token)

        session_id = kwargs.pop("session_id", None)

        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        # See ``get_response`` for the full rationale on #841 and #842.
        signature = _compute_request_signature(instructions, None)
        cached_outputs = self._load_replay_outputs(session_id, signature)
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
        splitter = ThinkingContentSplitter(provider="codex")

        async with httpx.AsyncClient(timeout=120) as http:
            for attempt in range(2):
                account_id = _extract_account_id(token)
                headers = _build_headers(token, account_id)
                async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                    if resp.status_code == 401 and attempt == 0:
                        await resp.aread()
                        new_token = await self._refresh_token_if_possible(token)
                        if new_token and new_token != token:
                            token = new_token
                            continue
                        error_text = resp.text[:500]
                        logger.error(f"Codex stream error 401: {error_text}")
                        raise RuntimeError(f"Codex API returned 401: {error_text}")
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
                            for item in splitter.feed(event.get("delta", "")):
                                yield item
                        elif event_type in (
                            "response.reasoning_text.delta",
                            "response.reasoning_summary_text.delta",
                        ):
                            delta = event.get("delta", "")
                            if delta:
                                yield ThinkingDelta(delta, provider="codex")
                        elif event_type == "response.output_item.done":
                            item = event.get("item", {})
                            if item.get("type") in ("reasoning", "function_call", "message"):
                                new_turn_outputs.append(item)
                        elif event_type == "response.completed":
                            resp_data = event.get("response", {})
                            last_response_id = resp_data.get("id") or last_response_id
                    break  # successful stream consumed

        for item in splitter.flush():
            yield item

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
    ) -> AsyncIterator[Union[str, ThinkingDelta, LLMResponse]]:
        """Stream response with tool call detection."""
        token = client
        if not isinstance(token, str):
            raise RuntimeError("Codex adapter requires an OAuth token.")
        token = self._current_token(token)

        session_id = kwargs.pop("session_id", None)

        instructions, input_messages = _extract_instructions_and_input(messages)
        instructions = self.contribute_system_prompt(model, instructions)
        responses_tools = _convert_tools_to_responses_format(tools)

        # See ``get_response`` for the full rationale on #841 and #842.
        signature = _compute_request_signature(instructions, responses_tools)
        cached_outputs = self._load_replay_outputs(session_id, signature)
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
        splitter = ThinkingContentSplitter(provider="codex")

        async with httpx.AsyncClient(timeout=120) as http:
            for attempt in range(2):
                account_id = _extract_account_id(token)
                headers = _build_headers(token, account_id)
                async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                    if resp.status_code == 401 and attempt == 0:
                        await resp.aread()
                        new_token = await self._refresh_token_if_possible(token)
                        if new_token and new_token != token:
                            token = new_token
                            continue
                        error_text = resp.text[:500]
                        raise RuntimeError(f"Codex API returned 401: {error_text}")
                    if resp.status_code != 200:
                        await resp.aread()
                        error_text = resp.text[:500]
                        raise RuntimeError(
                            f"Codex API returned {resp.status_code}: {error_text}"
                        )

                    # Track which tool-call indices we've already
                    # announced via ToolCallStarted so we emit exactly
                    # once per index even when the SDK delivers
                    # function-call arguments BEFORE the output-item
                    # added event (which is the path that populates
                    # id/name). The first event for an index — whichever
                    # of the three branches below it lands in — wins.
                    started_indices: set = set()

                    async for event in _parse_sse_events(resp):
                        event_type = event.get("type", "")

                        if event_type == "response.output_text.delta":
                            chunk_count += 1
                            delta = event.get("delta", "")
                            for item in splitter.feed(delta):
                                if isinstance(item, str):
                                    text_content += item
                                yield item

                        elif event_type in (
                            "response.reasoning_text.delta",
                            "response.reasoning_summary_text.delta",
                        ):
                            delta = event.get("delta", "")
                            if delta:
                                yield ThinkingDelta(delta, provider="codex")

                        elif event_type == "response.function_call_arguments.delta":
                            idx = event.get("output_index", 0)
                            is_new = idx not in func_calls
                            if is_new:
                                func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                            func_calls[idx]["arguments"] += event.get("delta", "")
                            if is_new and idx not in started_indices:
                                # Arguments delta arrived before the
                                # output-item.added event — id/name not
                                # yet known. Same MAY-BE-NONE case the
                                # contract documents for OpenAI's first
                                # delta path.
                                started_indices.add(idx)
                                yield ToolCallStarted(
                                    index=idx, id=None, name=None,
                                )

                        elif event_type == "response.function_call_arguments.done":
                            idx = event.get("output_index", 0)
                            is_new = idx not in func_calls
                            if is_new:
                                func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                            func_calls[idx]["arguments"] = event.get("arguments", "")
                            if is_new and idx not in started_indices:
                                started_indices.add(idx)
                                yield ToolCallStarted(
                                    index=idx, id=None, name=None,
                                )

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
                                call_id = item.get("call_id") or item.get("id", "")
                                call_name = item.get("name", "")
                                # MERGE id/name into the existing entry
                                # rather than replacing it. The two
                                # arguments-event branches above may have
                                # already created func_calls[idx] and
                                # accumulated argument deltas — replacing
                                # with ``arguments: ""`` would silently
                                # drop them, and the orchestrator would
                                # later execute the tool with ``{}``.
                                existing = func_calls.get(idx)
                                if existing is None:
                                    func_calls[idx] = {
                                        "id": call_id,
                                        "name": call_name,
                                        "arguments": "",
                                    }
                                else:
                                    existing["id"] = call_id
                                    existing["name"] = call_name
                                    # Preserve existing["arguments"].
                                if idx not in started_indices:
                                    # Output-item-added is the typical
                                    # first event for an index — id and
                                    # name are populated, so the marker
                                    # carries them.
                                    started_indices.add(idx)
                                    yield ToolCallStarted(
                                        index=idx,
                                        id=call_id or None,
                                        name=call_name or None,
                                    )

                        elif event_type == "response.output_item.done":
                            item = event.get("item", {})
                            if item.get("type") in ("reasoning", "function_call", "message"):
                                new_turn_outputs.append(item)

                        elif event_type == "response.completed":
                            resp_data = event.get("response", {})
                            usage = resp_data.get("usage", {})
                            final_usage = usage
                            last_response_id = resp_data.get("id") or last_response_id
                    break  # successful stream consumed

        for item in splitter.flush():
            if isinstance(item, str):
                text_content += item
            yield item

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
                    # SDK 0.7.0 malformed-JSON sentinel.
                    args = {"_raw": fc["arguments"]}
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

    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        """OpenAI plan uses canonical OpenAI discovery; this wrapper has no catalog.

        ``client`` is accepted for contract symmetry with the SDK 0.5.0
        :meth:`LLMAdapter.list_models` signature; not used because this
        wrapper deliberately raises rather than producing a catalog.
        """
        raise NotImplementedError(
            "OpenAI plan model discovery is provided by the canonical openai provider."
        )

    # ---- Provider metadata (SDK 0.6.0) -------------------------------------

    def cost_per_1m_tokens(self) -> Optional[Dict[str, float]]:
        # Plan-included usage — no per-token billing visible from the
        # adapter side. Returning ``{"input": 0.0, "output": 0.0}``
        # rather than ``None`` so cost-aware routing reflects "free at
        # the margin" rather than falling through to a conservative
        # paid-API default.
        return {"input": 0.0, "output": 0.0}

    def substrate_type(self) -> Optional[str]:
        return "gpt"

    def display_name(self) -> Optional[str]:
        return "OpenAI Codex (plan)"

    def key_env_var(self) -> Optional[str]:
        # OAuth-based plan auth — no env-var API key.
        return None
