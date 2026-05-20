"""Codex Provider Adapter — OpenAI ChatGPT subscription via the official
``@openai/codex`` app-server.

OpenAI officially sanctions third-party harnesses on a ChatGPT
subscription (Altman, 2026-05-02: *"you can sign in to openclaw with
your chatgpt account now and use your subscription there"*). The
sanctioned mechanism — the one OpenClaw uses — is to drive the official
``codex app-server`` binary over local stdio JSON-RPC. The binary owns
OAuth (``~/.codex/auth.json``), token refresh, account identification
and the chatgpt.com transport. We identify as ourselves in the
handshake; no impersonation.

This replaces the previous hand-rolled HTTP client that POSTed directly
to ``chatgpt.com/backend-api/codex/responses``. That reimplementation
got cut off by OpenAI's edge with opaque ``503 upstream connect`` errors
once the backend required traffic through the versioned app-server
protocol. All of the old OAuth/JWT/SSE/refresh machinery is gone — the
binary does it now.

Statefulness: the app-server is thread-stateful (it retains conversation
history server-side per thread). Kestrel's adapter contract is stateless
(full history every call). We bridge by keying a Codex ``threadId`` to
kestrel's stable ``session_id``.

Tool calls: the app-server runs a *server-driven* tool loop —
``dynamicTools`` → server emits ``item/tool/call`` (server→client RPC)
mid-turn, expects an inline reply, then resumes. We bridge that to
kestrel's existing security-gated tool dispatcher via a per-turn
``tool_executor`` callback: every call still fires kestrel's
``PRE_TOOL_USE``/``POST_TOOL_USE`` hooks, ``SecurityHook``, the approval
queue, and denied-tool stripping — exactly as today. The adapter does
no execution itself; it just relays the call into kestrel's hooked
executor and the result back to the app-server.
"""
from __future__ import annotations

import json
import logging
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union,
)

from pydantic import BaseModel

from kestrel_sdk.llm import ToolCallStarted

from .adapter import LLMAdapter, LLMResponse, ThinkingDelta, ToolCall
from .codex_app_server import CodexAppServerClient, CodexAppServerError
from .continuation_store import ContinuationStore, InMemoryContinuationStore
from .gpt5_overlay import prepend_gpt5_overlay
from .model_metadata import ModelInfo

logger = logging.getLogger(__name__)

# Signature of the kestrel-side tool executor the orchestrator wires in.
# Returns a dict with at minimum ``success: bool``; ``result`` carries
# the tool's payload (already passed through POST_TOOL_USE hooks).
ToolExecutor = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _extract_instructions_and_input(messages):
    """Split messages into instructions (system prompt) and the rest."""
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


def _convert_tools_to_codex_dynamic_tools(tools):
    """Convert OpenAI function-tool defs to the app-server's
    ``CodexDynamicToolSpec`` shape: ``{name, description, inputSchema}``.

    The field name diverges from the Responses-API ``parameters`` and the
    wrapper ``{"type":"function", "function":{…}}`` is dropped — the
    app-server tool spec is its own protocol (see ``protocol.ts:67-71``
    in ``kestrel-claw/extensions/codex``).
    """
    if not tools:
        return None
    out = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            out.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "inputSchema": func.get("parameters", {"type": "object"}),
            })
    return out or None


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "input_text")
        )
    return "" if content is None else str(content)


def _build_turn_input(input_messages: List[Dict[str, Any]], *, fresh_thread: bool) -> str:
    """Text to send for this turn.

    Existing thread: only the latest user message (server holds history).
    Fresh thread with prior history: seed a compact transcript so context
    isn't lost, then the latest user message.
    """
    last_user_idx = None
    for i in range(len(input_messages) - 1, -1, -1):
        if input_messages[i].get("role") == "user":
            last_user_idx = i
            break
    latest = (
        _msg_text(input_messages[last_user_idx]["content"])
        if last_user_idx is not None else ""
    )
    if not fresh_thread or last_user_idx in (None, 0):
        return latest
    lines = []
    for m in input_messages[:last_user_idx]:
        role = m.get("role", "user")
        txt = _msg_text(m.get("content"))
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = ", ".join(
                tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]
            )
            txt = (txt + f" [called tools: {calls}]").strip()
        if txt:
            lines.append(f"{role}: {txt}")
    if not lines:
        return latest
    return (
        "Conversation so far (for context):\n"
        + "\n".join(lines)
        + f"\n\nCurrent message:\n{latest}"
    )


def _usage_from(tu: dict) -> Dict[str, Optional[int]]:
    total = (tu or {}).get("total", {}) or {}
    inp = total.get("inputTokens")
    out = total.get("outputTokens")
    tot = total.get("totalTokens")
    if tot is None and (inp is not None or out is not None):
        tot = (inp or 0) + (out or 0)
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": tot,
        "cache_read_input_tokens": total.get("cachedInputTokens"),
    }


def _result_to_codex_response(result: Any) -> Dict[str, Any]:
    """Marshal a kestrel tool result into the codex
    ``CodexDynamicToolCallResponse`` shape.

    Kestrel tool dispatchers conventionally return a dict with
    ``success: bool`` and either ``result`` (payload) or ``error`` (msg).
    Anything else is best-effort serialized.
    """
    if isinstance(result, dict):
        success = bool(result.get("success", True))
        if success and "result" in result:
            text = result["result"]
        elif "error" in result:
            text = result["error"]
        else:
            text = result
    else:
        success = True
        text = result
    if not isinstance(text, str):
        try:
            text = json.dumps(text, default=str)
        except Exception:
            text = str(text)
    return {
        "contentItems": [{"type": "inputText", "text": text}],
        "success": success,
    }


class CodexAdapter(LLMAdapter):
    """OpenAI ChatGPT-subscription adapter backed by the codex app-server."""

    def __init__(self, continuation_store: Optional[ContinuationStore] = None):
        self.name = "openai_plan"
        # Accepted for API stability; the app-server owns continuity via
        # server-side threads, so it is not used for replay anymore.
        self._continuation_store: ContinuationStore = (
            continuation_store if continuation_store is not None
            else InMemoryContinuationStore()
        )
        self._client: Optional[CodexAppServerClient] = None
        # session_id → (thread_id, fingerprint). The fingerprint guards
        # against silently reusing a thread whose initial config (model,
        # system prompt, tool set) no longer matches what the caller is
        # asking for — those settings only take effect at thread/start,
        # so a mismatch must force a fresh thread (loses server-side
        # history for that session, same posture as OpenClaw's
        # ``dynamicToolsFingerprint`` reset).
        self._session_threads: Dict[str, Tuple[str, str]] = {}

    # ----------------------------------------------------------- app-server glue
    def _app_server(self) -> CodexAppServerClient:
        if self._client is None:
            self._client = CodexAppServerClient()
        return self._client

    @staticmethod
    def _model_param(model: str) -> Optional[str]:
        # The route's configured model is often "auto"; the app-server
        # rejects that. Omit so it uses the subscription default.
        if not model or model in ("auto", "default"):
            return None
        return model

    @staticmethod
    def _thread_fingerprint(
        model_param: Optional[str], instructions: Optional[str],
        dynamic_tools: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Stable hash of every thread-scoped setting the app-server
        only consumes at thread/start. Used to invalidate a cached
        thread when the caller asks for different model/instructions/
        tools — those changes are otherwise silently ignored."""
        import hashlib

        payload = json.dumps(
            {
                "m": model_param or "",
                "i": instructions or "",
                "t": dynamic_tools or [],
            },
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    async def _ensure_thread(
        self, app: CodexAppServerClient, session_id: Optional[str],
        model: str, instructions: Optional[str],
        dynamic_tools: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, bool]:
        """Return ``(thread_id, is_fresh)`` for this session.

        ``dynamic_tools`` (codex spec shape) are registered at
        ``thread/start`` per the kestrel-claw reference — turn-level
        registration is a documented protocol option but not what
        OpenClaw uses, and empirically the model only sees tools when
        they're declared at the thread level.

        On a session whose cached thread was built with different
        model/instructions/tools, start a fresh thread (the app-server
        won't apply those changes to an existing thread). This loses
        server-side history for the session at that boundary —
        unavoidable given the protocol; mirrors OpenClaw's fingerprint
        reset behaviour.
        """
        m = self._model_param(model)
        fingerprint = self._thread_fingerprint(m, instructions, dynamic_tools)
        if session_id and session_id in self._session_threads:
            cached_id, cached_fp = self._session_threads[session_id]
            if cached_fp == fingerprint:
                return cached_id, False
            logger.info(
                "codex session %s thread config changed (%s → %s); "
                "starting fresh thread", session_id, cached_fp, fingerprint,
            )
            # Cached thread no longer matches; drop it.
            self._session_threads.pop(session_id, None)
        params: Dict[str, Any] = {"sandbox": "read-only"}
        if m:
            params["model"] = m
        if instructions:
            params["developerInstructions"] = instructions
        if dynamic_tools:
            params["dynamicTools"] = dynamic_tools
        result = await app.request("thread/start", params, timeout=60)
        thread_id = (result or {}).get("thread", {}).get("id")
        if not thread_id:
            raise CodexAppServerError(
                f"thread/start returned no thread id: {result!r}"
            )
        if session_id:
            self._session_threads[session_id] = (thread_id, fingerprint)
        return thread_id, True

    def _make_tool_call_handler(
        self, executor: ToolExecutor, thread_id: str
    ) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
        """Wrap ``executor`` into an app-server ``item/tool/call``
        handler scoped to this turn's thread.

        Server-request handlers are registered globally per method; if a
        callback arrives for a *different* thread (e.g. a concurrent
        turn's), reply with an explicit failure rather than running the
        wrong executor against the wrong session.
        """

        async def handler(params: Dict[str, Any]) -> Dict[str, Any]:
            if params.get("threadId") != thread_id:
                return {
                    "contentItems": [{
                        "type": "inputText",
                        "text": "tool call belonged to a different turn",
                    }],
                    "success": False,
                }
            name = params.get("tool") or params.get("name") or ""
            args = params.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            try:
                result = await executor(name, args)
            except Exception as e:
                logger.warning("tool_executor(%s) raised: %s", name, e)
                return {
                    "contentItems": [{
                        "type": "inputText",
                        "text": f"tool {name!r} failed: {e}",
                    }],
                    "success": False,
                }
            return _result_to_codex_response(result)

        return handler

    async def _run_turn(
        self, model: str, messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]], session_id: Optional[str],
        tool_executor: Optional[ToolExecutor],
    ) -> AsyncIterator[dict]:
        """Drive one turn; yield normalized events:

        ``{"text": str}`` | ``{"thinking": str}`` |
        ``{"tool_call": ToolCall}`` | ``{"final": (text, [ToolCall], usage)}``
        """
        app = self._app_server()
        await app.ensure_started()
        instructions, input_messages = _extract_instructions_and_input(messages)
        # Nothing in the LLM pipeline calls contribute_system_prompt for
        # us — the adapter owns it (regresses the GPT-5 <persona_latch>
        # overlay otherwise).
        instructions = self.contribute_system_prompt(model, instructions)
        # Sanctioned subscription model + tools requires a hook-enforcing
        # executor (the app-server runs an inline tool loop). Without
        # one, the model would request tools and we'd have no safe way
        # to run them — fail loud rather than silently decline + corrupt
        # the loop.
        dyn = _convert_tools_to_codex_dynamic_tools(tools)
        if dyn and tool_executor is None:
            raise CodexAppServerError(
                "openai:plan (codex app-server) requires a tool_executor "
                "callback when tools are provided. The orchestrator must "
                "thread its hook-enforcing executor through "
                "generate_with_messages / stream_with_tool_detection."
            )
        thread_id, fresh = await self._ensure_thread(
            app, session_id, model, instructions, dyn,
        )
        turn_input = _build_turn_input(input_messages, fresh_thread=fresh)

        unregister = None
        if tool_executor is not None:
            # Thread-scoped registration: concurrent turns on different
            # threads each get their own ``item/tool/call`` handler and
            # the dispatcher routes by ``params.threadId``. Without
            # scoping, a second turn's registration would silently
            # overwrite an in-flight turn's handler.
            unregister = app.register_server_request_handler(
                "item/tool/call",
                self._make_tool_call_handler(tool_executor, thread_id),
                thread_id=thread_id,
            )

        sink = app.open_turn_sink(thread_id)
        try:
            # dynamicTools are registered at thread/start (above); turn/start
            # only carries the user input.
            await app.request("turn/start", {
                "threadId": thread_id,
                "input": [{"type": "text", "text": turn_input}],
            }, timeout=60)

            text_parts: List[str] = []
            final_text: Optional[str] = None
            tool_calls: List[ToolCall] = []
            usage: Dict[str, Optional[int]] = {}
            seen_tool_ids: set = set()

            async for ev in app.iter_turn_events(sink):
                method = ev.get("method")
                p = ev.get("params") or {}
                if method == "item/agentMessage/delta":
                    delta = p.get("delta") or ""
                    if delta:
                        text_parts.append(delta)
                        yield {"text": delta}
                elif method and method.endswith("/reasoning/delta"):
                    d = p.get("delta") or ""
                    if d:
                        yield {"thinking": d}
                elif method == "item/completed":
                    item = p.get("item") or {}
                    itype = item.get("type")
                    if itype == "agentMessage":
                        final_text = item.get("text") or final_text
                    elif itype in ("functionCall", "toolCall", "function_call",
                                   "mcpToolCall", "customToolCall"):
                        cid = item.get("id") or item.get("callId") or ""
                        if cid in seen_tool_ids:
                            continue
                        seen_tool_ids.add(cid)
                        raw_args = item.get("arguments")
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except ValueError:
                                raw_args = {"_raw": raw_args}
                        tc = ToolCall(
                            id=cid,
                            name=item.get("name") or item.get("tool") or "",
                            arguments=raw_args if isinstance(raw_args, dict) else {},
                        )
                        tool_calls.append(tc)
                        yield {"tool_call": tc}
                elif method == "thread/tokenUsage/updated":
                    usage = _usage_from(p.get("tokenUsage") or {})
                elif method == "turn/failed":
                    err = p.get("error") or p.get("turn", {}).get("error") or {}
                    raise CodexAppServerError(
                        f"codex turn failed: "
                        f"{err.get('message') or err or 'unknown'}"
                    )
                # turn/completed terminates iter_turn_events.

            content = final_text if final_text is not None else "".join(text_parts)
            yield {"final": (content or None, tool_calls or None, usage)}
        finally:
            app.close_turn_sink(thread_id)
            if unregister is not None:
                unregister()

    # --------------------------------------------------------------- public API
    async def get_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:
        session_id = kwargs.get("session_id")
        tool_executor = kwargs.get("tool_executor")
        content: Optional[str] = None
        tool_calls: Optional[List[ToolCall]] = None
        usage: Dict[str, Optional[int]] = {}
        async for ev in self._run_turn(
            model, messages, tools, session_id, tool_executor
        ):
            if "final" in ev:
                content, tool_calls, usage = ev["final"]
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
        )

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> AsyncIterator[Union[str, ThinkingDelta]]:
        session_id = kwargs.get("session_id")
        # Tools intentionally not passed: text-only streaming surface.
        async for ev in self._run_turn(model, messages, None, session_id, None):
            if "text" in ev:
                yield ev["text"]
            elif "thinking" in ev:
                yield ThinkingDelta(ev["thinking"], provider="codex")

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> AsyncIterator[Union[str, ThinkingDelta, ToolCallStarted, LLMResponse]]:
        session_id = kwargs.get("session_id")
        tool_executor = kwargs.get("tool_executor")
        idx = 0
        async for ev in self._run_turn(
            model, messages, tools, session_id, tool_executor
        ):
            if "text" in ev:
                yield ev["text"]
            elif "thinking" in ev:
                yield ThinkingDelta(ev["thinking"], provider="codex")
            elif "tool_call" in ev:
                tc = ev["tool_call"]
                yield ToolCallStarted(idx, tc.id or None, tc.name or None)
                idx += 1
            elif "final" in ev:
                content, tcs, usage = ev["final"]
                yield LLMResponse(
                    content=content,
                    tool_calls=tcs,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                )

    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        """Model discovery is the canonical openai provider's job.

        Kept as ``NotImplementedError`` (unchanged contract): the plan
        route deliberately defers its catalog to ``openai:api``.
        """
        raise NotImplementedError(
            "OpenAI plan model discovery is provided by the canonical openai provider."
        )

    def contribute_system_prompt(
        self, model_id: str, base: Optional[str]
    ) -> Optional[str]:
        return prepend_gpt5_overlay(base, model_id)

    def cost_per_1m_tokens(self) -> Optional[Dict[str, float]]:
        return {"input": 0.0, "output": 0.0}

    def substrate_type(self) -> Optional[str]:
        return "gpt"

    def display_name(self) -> Optional[str]:
        return "OpenAI Codex (plan)"

    def key_env_var(self) -> Optional[str]:
        # Auth is delegated to the codex binary's ~/.codex/auth.json.
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
