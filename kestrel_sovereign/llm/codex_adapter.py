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

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union,
)

from pydantic import BaseModel

from kestrel_sdk.llm import ToolCallStarted

from .adapter import LLMAdapter, LLMResponse, ThinkingDelta, ToolCall
from kestrel_sdk.llm import (
    ProviderCapabilities,
    StructuredOutputMode,
    ToolStreamingMode,
    VisionInputMode,
)
from .codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    CodexAppServerTransportError,
)
from .continuation_store import ContinuationStore, InMemoryContinuationStore
from .gpt5_overlay import prepend_gpt5_overlay
from .model_metadata import ModelInfo

logger = logging.getLogger(__name__)

# Signature of the kestrel-side tool executor the orchestrator wires in.
# Returns a dict with at minimum ``success: bool``; ``result`` carries
# the tool's payload (already passed through POST_TOOL_USE hooks).
# Async ``(name, args) -> (effective_args, result)``. The kestrel-side
# wrapper returns the post-hook args alongside the result so the
# adapter can record the *actually-executed* arguments into audit /
# UI breadcrumbs — pre-hook args may carry values a PRE_TOOL_USE
# redactor intentionally stripped.
ToolExecutor = Callable[
    [str, Dict[str, Any]],
    Awaitable[Tuple[Dict[str, Any], Any]],
]


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


# Codex dynamicTool namespace for kestrel tools. The codex app-server
# has its own native tool surface (``spawn_agent``, ``shell``,
# ``apply_patch``, ``web_search``, ``read_file``, …) registered in a
# process-global handler registry inside ``codex_core``. Advertising a
# dynamicTool with the same name but no namespace collides with codex's
# own — the second registration panics with
# ``handler for tool <name> already registered`` and the app-server
# exits. Kestrel hit this with the SpawnFeature (``spawn_agent``,
# ``core = true`` in feature_registry.toml) as soon as a session
# triggered a second ``thread/start`` (model switch / config change
# / fingerprint mismatch).
#
# OpenClaw solves the same class of collision by namespacing every
# dynamicToolSpec under ``"openclaw"`` (kestrel-claw
# extensions/codex/src/app-server/dynamic-tools.ts:64). We follow the
# same pattern with ``"kestrel"``. The codex protocol's
# ``CodexDynamicToolCallParams`` carries ``namespace`` back to us on
# every ``item/tool/call``; the tool ``name`` field stays as-is, so
# kestrel-side dispatch via ``execute_named_tool`` continues to work
# without a stripping step.
_KESTREL_TOOL_NAMESPACE = "kestrel"


def _convert_tools_to_codex_dynamic_tools(tools):
    """Convert OpenAI function-tool defs to the app-server's
    ``CodexDynamicToolSpec`` shape: ``{name, description, inputSchema,
    namespace}``.

    The field name diverges from the Responses-API ``parameters`` and the
    wrapper ``{"type":"function", "function":{…}}`` is dropped — the
    app-server tool spec is its own protocol (see ``protocol.ts:67-71``
    in ``kestrel-claw/extensions/codex``).

    ``namespace`` is set to ``"kestrel"`` on every entry so our tools
    register in their own slot inside ``codex_core``'s tool handler
    registry, avoiding collisions with codex-native built-ins (most
    visibly ``spawn_agent``, which kestrel also exposes via the
    SpawnFeature).
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
                "namespace": _KESTREL_TOOL_NAMESPACE,
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


# Item types the app-server emits for a "the model invoked a tool" lifecycle.
# These all surface tool activity to the user — whether they ran inside the
# app-server (commandExecution, fileChange, webSearch) or were dispatched
# back to kestrel via item/tool/call (dynamicToolCall, mcpToolCall,
# functionCall, customToolCall). The chat-UI activity parser keys on the
# 🔧 / ✓ / ❌ marker lines we yield as TEXT for each of these — same wire
# protocol the orchestrator-dispatched path uses (see
# ``orchestrator_engine._stream_with_tools`` line "Calling {tc.name}...").
_TOOL_ITEM_TYPES = frozenset({
    "dynamicToolCall", "mcpToolCall", "functionCall", "function_call",
    "toolCall", "customToolCall",
    "commandExecution", "fileChange", "webSearch",
})

# These item types resolve to an inline kestrel tool-call (via the
# ``item/tool/call`` server→client RPC). They MUST be tracked in
# ``executed_tool_calls`` so the orchestrator can fold them into the
# persisted chat-history breadcrumbs. Other tool item types
# (commandExecution, fileChange, webSearch) run INSIDE the app-server's
# native sandbox — kestrel's hook stack never sees them, so they appear
# only as on-screen markers, never as executed_tool_calls entries.
_KESTREL_DISPATCHED_TOOL_ITEM_TYPES = frozenset({
    "dynamicToolCall", "mcpToolCall", "functionCall", "function_call",
    "toolCall", "customToolCall",
})


def _item_display_label(item: Dict[str, Any]) -> str:
    """Human label for an item/started or item/completed marker.

    Tools dispatched back to kestrel use the tool name; codex-native
    items (commandExecution, fileChange) use a stable short label so the
    chat-UI parser groups successive output lines under one card.
    """
    itype = item.get("type") or ""
    if itype == "commandExecution":
        return "shell"
    if itype == "fileChange":
        return "apply_patch"
    if itype == "webSearch":
        return "web_search"
    return item.get("name") or item.get("tool") or itype or "tool"


def _format_command_summary(command: Any) -> Optional[str]:
    """One-line preview of a shell command for the start marker.

    Strings stay as-is (truncated); list shapes (``["bash","-lc","..."]``)
    are joined with spaces so the user sees the actual command, not a
    JSON literal. Returns ``None`` when there is nothing worth showing.
    """
    if isinstance(command, str):
        text = command
    elif isinstance(command, list):
        text = " ".join(str(c) for c in command if c is not None)
    else:
        return None
    text = text.strip()
    if not text:
        return None
    # Single-line, capped — fits one marker line in chat without
    # collapsing a multi-line heredoc into noise.
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > 200:
        text = text[:197] + "..."
    return text


def _result_to_codex_response(
    result: Any,
    *,
    tool_name: str = "",
    feature_name: str = "",
    recent_decisions: Optional[list] = None,
) -> Dict[str, Any]:
    """Marshal a kestrel tool result into the codex
    ``CodexDynamicToolCallResponse`` shape.

    Kestrel tool dispatchers conventionally return a dict with
    ``success: bool`` and either ``result`` (payload) or ``error`` (msg).
    Anything else is best-effort serialized.

    On a FAILURE result, the raw ``error`` text is rewritten through
    :func:`classify_and_render_failure` so the LLM never sees Codex's
    misleading ``"rejected by user"`` substring as a literal claim of
    user denial when no audit row backs it (#1563 root cause). The
    typed outcome + recovery hint replaces the raw string in the
    LLM-visible ``inputText``; the raw error is preserved inside the
    rendered block so the LLM still has the original wording for
    debugging context.
    """
    # ToolResult (#1061): kestrel's standard envelope. After PR-E
    # every in-tree @tool returns one — the legacy ``{success: bool}``
    # dict path is the EXCEPTION now, not the rule. Catch ToolResult
    # before the dict / non-dict branches so a ``ToolResult.failed(...)``
    # carrying a misleading raw error string still routes through the
    # classifier (codex P1 round 1 — a direct repro showed the rewrite
    # would otherwise stringify it raw with ``success=True``).
    if _is_toolresult(result):
        status = str(getattr(result, "status", "")).lower()
        err = getattr(result, "error", None)
        # ``ToolResultStatus.ERROR`` / ``PARTIAL`` are non-success;
        # ``OK`` is success. The status enum string-coerces with a
        # ``ToolResultStatus.OK`` repr on Python 3.13, but
        # ``str(enum) == "ToolResultStatus.OK"`` — match either.
        success = "error" not in status and "partial" not in status
        if success:
            data = getattr(result, "data", None)
            confirmation = getattr(result, "confirmation", None)
            text = (
                confirmation
                if isinstance(confirmation, str) and confirmation
                else data
            )
        else:
            text = err if isinstance(err, str) and err else str(result)
    elif isinstance(result, dict):
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
    if not success:
        text = classify_and_render_failure(
            text,
            tool_name=tool_name,
            feature_name=feature_name,
            recent_decisions=recent_decisions,
        )
    return {
        "contentItems": [{"type": "inputText", "text": text}],
        "success": success,
    }


def _is_toolresult(obj: Any) -> bool:
    """Duck-typed ToolResult check — avoids a hard import of
    ``kestrel_sdk.tools.result`` at the LLM-adapter layer so a future
    SDK move can't break the codex marshaller. The contract this
    relies on is the public ``status`` + ``error`` + ``data`` +
    ``confirmation`` field set documented at the ToolResult class.
    """
    return (
        hasattr(obj, "status")
        and hasattr(obj, "error")
        and hasattr(obj, "data")
        and hasattr(obj, "confirmation")
    )


# Recovery hints per outcome — what the LLM should TRY NEXT for each
# typed failure. Pairs with the escalation classifier so a downstream
# LLM gets actionable next-step guidance, not just a typed label.
_RECOVERY_HINTS: Dict[str, str] = {
    "user_denied": (
        "respect the user's denial. Stop attempting this "
        "command. Ask the user whether they want a different approach."
    ),
    "policy_blocked": (
        "try the operator-approval queue (file a request "
        "via the security feature), or use a tool that runs without "
        "this policy gate. Do NOT claim the user denied — this was "
        "a policy/plumbing refusal, not a user decision."
    ),
    "sandbox_blocked": (
        "this is a sandbox refusal, NOT a user denial. "
        "Try a less-privileged path, request elevation via the "
        "restart_coordinator update path if appropriate, or surface "
        "the block honestly to the user. The Codex CLI's literal "
        "\"rejected by user\" wording is its sandbox diagnostic, not "
        "evidence the user decided anything."
    ),
    "tooling_error": (
        "check that the tool/binary is installed and "
        "reachable, retry once if the error looks transient (RPC "
        "timeout, network blip), or surface the tooling problem to "
        "the user. Do NOT attribute this to a user denial."
    ),
    "unconfirmed": (
        "the outcome is unconfirmed. Tell the user the "
        "command result could not be confirmed and ask how they "
        "want to proceed. Do NOT guess that the user denied it."
    ),
}


def classify_and_render_failure(
    raw_error: str,
    *,
    tool_name: str = "",
    feature_name: str = "",
    recent_decisions: Optional[list] = None,
) -> str:
    """Convert a raw tool error string into the LLM-visible failure
    block: typed outcome + classifier reasoning + recovery hint +
    preserved raw error.

    The block format is deliberately structured so the downstream
    LLM can parse it (or just read the ``Outcome:`` and ``Recovery:``
    lines) without having to interpret the misleading raw wording
    Codex hands us. The raw error stays inside the block so any
    other diagnostic value (file paths, exit codes, etc.) is not
    lost.
    """
    from kestrel_sovereign.llm.escalation_classifier import (
        classify_escalation_failure,
    )
    decision = classify_escalation_failure(
        raw_error,
        recent_decisions=recent_decisions or [],
        tool_name=tool_name,
        feature_name=feature_name,
    )
    outcome = decision.outcome.value
    hint = _RECOVERY_HINTS.get(outcome, _RECOVERY_HINTS["unconfirmed"])
    raw_block = (raw_error or "").strip()
    if not raw_block:
        raw_block = "(no raw error message)"
    return (
        f"Tool failed.\n"
        f"Outcome: {outcome}\n"
        f"Reason: {decision.reason}\n"
        f"Evidence: {decision.evidence_source}\n"
        f"Recovery: {hint}\n"
        f"Raw error (for diagnostic context — do NOT echo verbatim "
        f"as a user denial claim):\n"
        f"  {raw_block}"
    )


_CODEX_RETRY_BASE_SECONDS = 5.0
_CODEX_RETRY_JITTER_SECONDS = 2.0


def _codex_retry_wait_seconds() -> float:
    """How long ``_run_turn_with_retry`` waits between attempts.

    Module-level seam: tests monkey-patch this to ``lambda: 0.0`` to
    avoid burning real wall-clock seconds in the retry path. Production
    returns a 5-7s jittered float — minute-scale stalls don't benefit
    from sub-second backoff and a small jitter prevents synchronized
    second-attempt spikes when multiple agents hit the same upstream
    blip.
    """
    return _CODEX_RETRY_BASE_SECONDS + random.uniform(0, _CODEX_RETRY_JITTER_SECONDS)


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
        # Per-session lock for ``_ensure_thread`` lookup-or-create — two
        # concurrent first calls on the same session_id would otherwise
        # both call thread/start and race to overwrite the cache,
        # leaving one thread orphaned with whichever turn was mid-flight.
        self._session_locks: Dict[str, "asyncio.Lock"] = {}
        # Per-thread serialization: the codex app-server allows only one
        # active turn per thread. Two concurrent ``_run_turn`` calls on
        # the same thread (same session_id) would race on the turn-sink
        # registration; the lock makes them queue cleanly instead.
        self._thread_locks: Dict[str, "asyncio.Lock"] = {}
        # #1563 pre-response rewrite: optional agent reference whose
        # ``SecurityFeature.permission_store`` we can cross-reference
        # when classifying a failure. When unset (test stubs, headless
        # host) the classifier falls through to raw-error pattern
        # matching, which is sufficient for the primary bug — a Codex
        # ``"rejected by user"`` sandbox string still classifies as
        # SANDBOX_BLOCKED on raw-error patterns alone. The audit
        # cross-check is defense-in-depth: even with no audit, the
        # post-response audit hook (#1568) catches any LLM that
        # produces user-denial wording from the rewritten typed
        # block.
        self._agent_for_audit: Any = None
        # Codex approval bridge (#1575) state. Registered ONCE per
        # (adapter, agent, app_client) — not per turn — so concurrent
        # turns can't race on registration/unregistration of the
        # shared (method, None) key. Re-bind to a new agent OR to a
        # fresh CodexAppServerClient (post-aclose) will unregister +
        # re-register (codex review #1575 round 3 P2).
        self._codex_bridge_for_agent: Any = None
        self._codex_bridge_for_app: Any = None
        self._codex_bridge_unregisters: List[Callable[[], None]] = []

    def attach_agent_for_audit(self, agent: Any) -> None:
        """Bind the running agent so failure-result rewrite can pull
        recent ``SecurityFeature.permission_store.get_audit_log`` rows
        (#1563). Optional — see ``_recent_security_decisions``.
        Idempotent; safe to call before every turn.

        Re-binding to a different agent drops any previously-registered
        codex approval bridge so the next turn rewires for the new
        agent (#1575).
        """
        if self._agent_for_audit is not agent:
            self._teardown_codex_approval_bridge()
        self._agent_for_audit = agent

    def _teardown_codex_approval_bridge(self) -> None:
        """Drop bridge handlers registered against a prior (agent, app)."""
        for _unreg in self._codex_bridge_unregisters:
            try:
                _unreg()
            except Exception:  # noqa: BLE001
                pass
        self._codex_bridge_unregisters = []
        self._codex_bridge_for_agent = None
        self._codex_bridge_for_app = None

    def _ensure_codex_approval_bridge(
        self, app: "CodexAppServerClient",
    ) -> None:
        """Register the codex sandbox-approval bridge handlers exactly
        once per (adapter, agent, app_client). Idempotent: subsequent
        calls for the same triple are no-ops.

        Lives outside the per-turn ``unregisters`` list so two
        concurrent turns can't race on the same (method, None) key —
        a later turn's unregister would otherwise tear down the bridge
        while an earlier turn was still in flight (codex review #1575
        round 2 P2). Re-keys on agent OR app-client change so a
        post-``aclose`` reconnect doesn't reuse a dead-handler binding
        (round 3 P2).
        """
        agent = self._agent_for_audit
        if agent is None:
            return
        if (
            self._codex_bridge_for_agent is agent
            and self._codex_bridge_for_app is app
        ):
            return
        # Agent or app-client changed → clean reset.
        self._teardown_codex_approval_bridge()
        for _kind in ("commandExecution", "fileChange"):
            _method = f"item/{_kind}/requestApproval"
            self._codex_bridge_unregisters.append(
                app.register_server_request_handler(
                    _method,
                    self._make_codex_approval_handler(agent, _kind),
                    thread_id=None,
                )
            )
        self._codex_bridge_for_agent = agent
        self._codex_bridge_for_app = app

    async def _recent_security_decisions(self, limit: int = 50) -> list:
        """Pull recent audit rows for the pre-response classifier.

        Best-effort: missing agent / SecurityFeature / permission_store
        / read failure all degrade to an empty list so the classifier
        falls through to raw-error pattern matching. A failure-result
        rewrite must never raise out of this path.
        """
        agent = self._agent_for_audit
        if agent is None:
            return []
        try:
            features = getattr(agent, "features", {}) or {}
            if isinstance(features, dict):
                security = features.get("SecurityFeature")
            else:
                security = next(
                    (f for f in features
                     if type(f).__name__ == "SecurityFeature"),
                    None,
                )
            if security is None:
                return []
            store = getattr(security, "permission_store", None)
            if store is None or not hasattr(store, "get_audit_log"):
                return []
            return await store.get_audit_log(limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "CodexAdapter: recent_security_decisions failed: %s", exc,
            )
            return []

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_vision=False,
            supports_structured_output=False,
            structured_output_mode=StructuredOutputMode.NONE,
            tool_streaming_mode=ToolStreamingMode.INLINE_EXECUTOR,
            vision_input_mode=VisionInputMode.NONE,
            notes=(
                "Codex app-server tools are executed through dynamic tool events.",
                "Generic response_format and image input are not part of this adapter contract.",
            ),
        )

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

        # Per-session lock; bare path for session-less calls (each
        # ephemeral thread is independent — no race possible).
        if session_id:
            lock = self._session_locks.setdefault(session_id, asyncio.Lock())
            await lock.acquire()
        else:
            lock = None
        try:
            if session_id and session_id in self._session_threads:
                cached_id, cached_fp = self._session_threads[session_id]
                if cached_fp == fingerprint:
                    return cached_id, False
                logger.info(
                    "codex session %s thread config changed (%s → %s); "
                    "starting fresh thread",
                    session_id, cached_fp, fingerprint,
                )
                # Cached thread no longer matches; drop it.
                self._session_threads.pop(session_id, None)
            # cwd: codex's native shell tool runs `cd` and resolves relative
            # paths against the thread cwd, not the process cwd. Without
            # this, ``pwd`` inside a shell tool returns whatever directory
            # the kestrel process happened to start in (often the install
            # prefix). Anchor the thread to the user's current working
            # directory so file lookups behave like an editor session.
            # ``KESTREL_CODEX_CWD`` overrides for daemon/agent runtimes
            # where ``Path.cwd()`` may not match the intended workspace.
            cwd = os.environ.get("KESTREL_CODEX_CWD") or str(Path.cwd())
            # Parity with kestrel-claw's thread-lifecycle.ts:599-619:
            # ``experimentalRawEvents`` flips codex's notification stream from
            # the legacy aggregated shape (which our adapter never wired up)
            # to the granular ``item/agentMessage/delta`` /
            # ``item/reasoning/textDelta`` / ``item/tool/call`` events that
            # ``_run_turn`` reads. Without it, codex enters ``session_loop``
            # and the response surfaces via a different channel, so kestrel
            # sees zero events and trips the 300s idle timeout.
            # ``persistExtendedHistory`` mirrors what claw asks for so
            # multi-turn sessions retain the full prior context server-side.
            params: Dict[str, Any] = {
                "sandbox": "read-only",
                "cwd": cwd,
                "experimentalRawEvents": True,
                "persistExtendedHistory": True,
                # Matches kestrel-claw's CODEX_CODE_MODE_DISABLED_THREAD_CONFIG
                # — kestrel uses dynamicTools for its own dispatch, not
                # codex's native code-mode bridge. Without this field,
                # codex 0.131+ enters session_loop on turn/start and never
                # makes the upstream Responses API call (hangs indefinitely
                # at thread-lifecycle.ts:599).
                "config": {
                    "features.code_mode": False,
                    "features.code_mode_only": False,
                },
            }
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
            # #1503 follow-up: codex's ``thread/start`` response carries
            # the server-reported per-thread / per-plan context window.
            # Codex's binary surfaces fields named ``model_context_window``,
            # ``max_context_window``, ``auto_compact_token_limit``, and
            # ``effective_context_window_percent`` on its ThreadSettings
            # snapshot. The smallest is the actual per-turn ceiling for
            # this session under THIS plan — feed it to the catalog as
            # the runtime-discovered route cap so context-status + the
            # ContextManager elastic budget track ground truth instead
            # of an empirical static guess that goes stale every time
            # OpenAI tunes the subscription tier.
            self._record_discovered_route_cap_from_thread_start(m, result)
            if session_id:
                self._session_threads[session_id] = (thread_id, fingerprint)
            return thread_id, True
        finally:
            if lock is not None:
                lock.release()

    # Field names codex's ``thread/start`` response carries on its
    # ThreadSettings snapshot (verified from the codex binary strings
    # table). camelCase first per codex-rs convention, snake_case
    # fallbacks in case the wire format drifts.
    _DISCOVERED_CAP_FIELDS = (
        "autoCompactTokenLimit",
        "auto_compact_token_limit",
        "modelContextWindow",
        "model_context_window",
        "maxContextWindow",
        "max_context_window",
    )

    # Catalog route key this adapter discovers caps for. CodexAdapter
    # is openai:plan-only by design; if a future adapter ever serves a
    # different capped route (e.g. a hypothetical openai:plan-pro
    # bridge) it would carry its own route key here. Hardcoded
    # because ``self.name`` is the SDK identifier ("openai_plan") and
    # downstream catalog / context-status look up by the route form
    # ``"openai:plan"`` — codex round 1 P2 on this PR.
    _DISCOVERED_CAP_ROUTE_KEY = "openai:plan"

    def _record_discovered_route_cap_from_thread_start(
        self,
        full_model_id: Optional[str],
        thread_start_result: Optional[Dict[str, Any]],
    ) -> None:
        """Capture the per-turn ceiling codex reports on ``thread/start``.

        Codex's response carries (server-named) fields on the thread
        settings snapshot that describe the actual upstream ceiling
        for this session under this plan. ``autoCompactTokenLimit``
        is the closest to a per-turn cap (it's where codex itself
        starts compacting); ``modelContextWindow`` /
        ``maxContextWindow`` describe the model's headroom. Take the
        smallest positive integer of whatever fields are present and
        fold it into the catalog as a runtime-discovered route cap
        keyed by ``openai:plan`` (the route, NOT the model id —
        downstream lookups are route-qualified) so context-status +
        the ContextManager elastic budget read it.

        Best-effort: missing fields / unparseable values silently
        degrade to "no discovery this turn" (operator's env override
        and the static config layers still apply via the normal
        precedence chain). ``full_model_id`` is taken only for the
        diagnostic log; the discovery applies to the entire route.
        """
        if not isinstance(thread_start_result, dict):
            return
        # Probe the top level of the response AND its nested
        # ``thread`` / ``threadSettings`` / ``thread_settings`` /
        # ``settings`` blocks, because the field placement isn't stable
        # across codex versions (the v0.135 strings table shows both
        # shapes). Also descend ONE level into the thread block so a
        # ``{"thread": {"settings": {...}}}`` shape is found — codex
        # round 2 P2 on this PR.
        probe_sources: List[Dict[str, Any]] = [thread_start_result]
        _SETTINGS_KEYS = ("thread", "threadSettings", "thread_settings", "settings")
        for nest_key in _SETTINGS_KEYS:
            nested = thread_start_result.get(nest_key)
            if isinstance(nested, dict):
                probe_sources.append(nested)
                # Codex sometimes places the snapshot one level deeper
                # under ``thread.settings`` / ``thread.threadSettings``.
                for inner_key in _SETTINGS_KEYS:
                    inner = nested.get(inner_key)
                    if isinstance(inner, dict):
                        probe_sources.append(inner)
        candidates: List[int] = []
        for src in probe_sources:
            for field in self._DISCOVERED_CAP_FIELDS:
                value = src.get(field)
                if isinstance(value, bool):
                    # ``True`` is technically an int subclass; skip it
                    # so a stray bool flag doesn't get treated as a
                    # 1-token cap.
                    continue
                if isinstance(value, int) and value > 0:
                    candidates.append(value)
        if not candidates:
            return
        discovered_cap = min(candidates)
        try:
            from .model_catalog import get_catalog_service
            catalog = get_catalog_service()
        except Exception as e:
            logger.debug(
                "codex discovered route cap %d for %s but catalog "
                "service unavailable (%s); not recording",
                discovered_cap, full_model_id, e,
            )
            return
        try:
            catalog.set_discovered_route_context_cap(
                self._DISCOVERED_CAP_ROUTE_KEY, discovered_cap,
            )
            logger.info(
                "codex thread/start: discovered route cap for %s = %d "
                "(model=%s, fields={%s})",
                self._DISCOVERED_CAP_ROUTE_KEY, discovered_cap,
                full_model_id or "auto",
                ", ".join(
                    f"{field}={src.get(field)}"
                    for src in probe_sources for field in self._DISCOVERED_CAP_FIELDS
                    if isinstance(src.get(field), int)
                    and not isinstance(src.get(field), bool)
                ),
            )
        except Exception as e:
            logger.debug(
                "failed to record discovered route cap %d on %s: %s",
                discovered_cap, self._DISCOVERED_CAP_ROUTE_KEY, e,
            )

    # Notification methods (codex sends these on the wire after a turn
    # advances; ``method`` is the JSON-RPC method name). Only the first
    # is the canonical per-turn ceiling, but accept either of the
    # historical snake_case spellings in case codex versions differ.
    _DISCOVERED_CAP_EVENT_METHODS = frozenset({
        "thread/tokenUsage/updated",
        "thread/token_usage/updated",
    })

    def _record_discovered_route_cap_from_event(
        self,
        params: Optional[Dict[str, Any]],
    ) -> None:
        """Capture the per-turn ceiling codex reports on the wire.

        Codex's ``thread/tokenUsage/updated`` notification carries
        ``tokenUsage.modelContextWindow`` — the per-session, per-plan
        ceiling the upstream actually enforces for the current
        thread. Verified on a live ChatGPT-Pro session 2026-06-03
        (#1518). The event fires AFTER each turn's final answer is
        emitted but before ``turn/completed``, so the captured value
        applies starting at the NEXT turn (the catalog's discovered
        layer then wins precedence per #1517).

        Best-effort: missing fields / unparseable values silently
        degrade to "no discovery this turn" (operator's env override
        and the static config layers still apply via the normal
        precedence chain). Catalog write failures are logged at debug
        and dropped — the recorder cannot crash the turn pipeline.
        """
        if not isinstance(params, dict):
            return
        token_usage = params.get("tokenUsage") or params.get("token_usage")
        if not isinstance(token_usage, dict):
            return
        for field in ("modelContextWindow", "model_context_window"):
            value = token_usage.get(field)
            if isinstance(value, bool):
                # ``True`` is an int subclass; reject so a stray flag
                # doesn't get treated as a 1-token cap.
                continue
            if isinstance(value, int) and value > 0:
                self._fold_discovered_cap(value, source=f"tokenUsage.{field}")
                return

    def _fold_discovered_cap(self, cap: int, *, source: str) -> None:
        """Push a discovered cap into the catalog. Shared sink for the
        thread/start path and the event path so they stay symmetric in
        catalog state, logging, and error handling."""
        try:
            from .model_catalog import get_catalog_service
            catalog = get_catalog_service()
        except Exception as e:
            logger.debug(
                "discovered route cap %d from %s but catalog service "
                "unavailable (%s); not recording",
                cap, source, e,
            )
            return
        try:
            # Compare against the DISCOVERED layer specifically, not the
            # merged precedence value. ``get_route_context_cap`` returns
            # the env override when one's set, which would never equal
            # the codex-reported value and so would defeat the anti-flood
            # guard (codex round 2 P3 on #1518).
            previous = getattr(catalog, "_discovered_route_context_caps", {}).get(
                self._DISCOVERED_CAP_ROUTE_KEY
            )
            catalog.set_discovered_route_context_cap(
                self._DISCOVERED_CAP_ROUTE_KEY, cap,
            )
            if previous != cap:
                logger.info(
                    "codex %s: route cap for %s now %d (was %s)",
                    source, self._DISCOVERED_CAP_ROUTE_KEY, cap, previous,
                )
        except Exception as e:
            logger.debug(
                "failed to record discovered route cap %d on %s from %s: %s",
                cap, self._DISCOVERED_CAP_ROUTE_KEY, source, e,
            )

    def _make_tool_call_handler(
        self, executor: ToolExecutor, thread_id: str,
        allowed_tools: frozenset,
        executed_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
        """Wrap ``executor`` into an app-server ``item/tool/call``
        handler scoped to this turn's thread.

        Defenses:

        * ``threadId`` mismatch → reject (server-request handlers are
          keyed by method+thread; a stray callback for another turn
          must not execute against the wrong session).
        * ``tool`` name not in ``allowed_tools`` → reject. The advertised
          dynamic-tool set for this turn is the only thing the executor
          should run; a hallucinated/malformed tool name could otherwise
          resolve through the orchestrator's full registry, bypassing
          per-turn restrictions (denied-tool stripping, capability
          gating).

        ``executed_log`` (optional): each successful execution appends
        ``{id, name, arguments, result}`` so callers can surface the
        same chat-history breadcrumbs the orchestrator-dispatched path
        produces. ``id`` is the app-server's own ``callId`` so audit
        trails align with codex-side ids.
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
            if name not in allowed_tools:
                logger.warning(
                    "codex requested tool %r not in advertised set %s; "
                    "refusing dispatch", name, sorted(allowed_tools),
                )
                return {
                    "contentItems": [{
                        "type": "inputText",
                        "text": (
                            f"tool {name!r} not advertised for this turn"
                        ),
                    }],
                    "success": False,
                }
            args = params.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            call_id = params.get("callId") or params.get("id") or ""
            # The executor returns (effective_args, result) — effective
            # args are what actually ran after PRE_TOOL_USE hooks, so
            # the breadcrumb reflects the redacted/rewritten payload,
            # not the model's pre-hook arguments.
            effective_args: Dict[str, Any] = (
                args if isinstance(args, dict) else {}
            )
            try:
                ret = await executor(name, args)
            except Exception as e:
                logger.warning("tool_executor(%s) raised: %s", name, e)
                err_result = {"success": False, "error": f"{e}"}
                # Failed inline tool calls must remain observable —
                # mirrors the orchestrator-dispatched path, which
                # records error results in tool_results/tool_events.
                if executed_log is not None:
                    executed_log.append({
                        "id": call_id, "name": name,
                        "arguments": effective_args,
                        "result": err_result,
                    })
                # Same classifier-rewrite the success-shaped failure
                # path uses, so a raised-exception tool error never
                # bypasses the honesty contract (#1563).
                rendered = classify_and_render_failure(
                    f"tool {name!r} raised: {e}",
                    tool_name=name,
                    recent_decisions=await self._recent_security_decisions(),
                )
                return {
                    "contentItems": [{
                        "type": "inputText",
                        "text": rendered,
                    }],
                    "success": False,
                }
            # Real orchestrator-side executor returns (effective_args,
            # result). Permissive: simpler tools/tests may return just
            # the result — in which case effective_args == args.
            if (
                isinstance(ret, tuple) and len(ret) == 2
                and isinstance(ret[0], dict)
            ):
                effective_args, result = ret
            else:
                result = ret
            if executed_log is not None:
                executed_log.append({
                    "id": call_id, "name": name,
                    "arguments": effective_args,
                    "result": result,
                })
            # Pass tool/feature scope + audit slice so the classifier
            # has the right evidence to classify a failure against
            # (#1563). For success results these are unused. The
            # failure-detection here must mirror _result_to_codex_response
            # so a ToolResult.failed / PARTIAL handler path still gets
            # the audit (codex P1 round 2 — the previous dict-only check
            # missed the ToolResult path entirely).
            is_failure = False
            if _is_toolresult(result):
                status = str(getattr(result, "status", "")).lower()
                is_failure = "error" in status or "partial" in status
            elif isinstance(result, dict):
                is_failure = not bool(result.get("success", True))
            return _result_to_codex_response(
                result,
                tool_name=name,
                recent_decisions=(
                    await self._recent_security_decisions()
                    if is_failure
                    else None
                ),
            )

        return handler

    def _make_codex_approval_handler(
        self, agent: Any, kind: str,
    ) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
        """Bridge codex's sandbox-approval RPCs through Kestrel's
        :class:`ApprovalQueue` (#1575).

        ``kind`` is ``"commandExecution"`` or ``"fileChange"``. The
        returned handler is registered globally on the app-server
        (codex's approval params don't carry ``threadId`` the way
        ``item/tool/call`` does); concurrent same-adapter turns share
        it, which is correct because they share the attached agent.

        Two gates fire in order:

        1. **Kestrel policy gate.** ``BinaryPolicy`` for commandExecution
           (binary not allow-listed → hard DENY); ``PathPolicy`` for
           fileChange (path on deny-list → hard DENY). Without this,
           auto-mode would auto-approve through the queue and codex
           could run any binary or write any path. Codex review #1575
           round 1 P1.
        2. **ApprovalQueue.** Keyed as ``feature_name="codex_native"``,
           ``tool_name=kind``. Auto-mode covers it; operator DENY in
           the permission store hard-stops it; every decision is
           audited.

        Reply shape mirrors codex's
        ``CommandExecutionApprovalDecision`` /
        ``FileChangeApprovalDecision`` enums (verified against the
        codex binary's serde labels): ``accept`` /
        ``acceptForSession`` / ``decline`` / ``abort``. ANY approval
        maps to ``accept`` (single-RPC scope is safe; the queue's
        permission store handles "always" scope on the Kestrel side).

        Failure modes are all conservative: missing SecurityFeature,
        missing ApprovalQueue, exception → ``decline``.
        """

        async def handler(params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Gate 1: Kestrel policy (BinaryPolicy / PathPolicy)
                # must hard-deny before the queue can auto-approve.
                policy_decline = self._codex_policy_precheck(
                    agent, kind, params or {},
                )
                if policy_decline is not None:
                    logger.info(
                        "codex_approval_bridge(%s): policy DENY (%s) — "
                        "declining", kind, policy_decline,
                    )
                    return {"decision": "decline"}

                # Gate 2: ApprovalQueue routing.
                features = getattr(agent, "features", {}) or {}
                if isinstance(features, dict):
                    security = features.get("SecurityFeature")
                else:
                    security = next(
                        (f for f in features
                         if type(f).__name__ == "SecurityFeature"),
                        None,
                    )
                queue = (
                    getattr(security, "approval_queue", None)
                    if security is not None
                    else None
                )
                if queue is None:
                    return {"decision": "decline"}
                # Bound the wait so a turn can't hang forever if no
                # operator is at the wheel. 120s is below codex's
                # 300s turn watchdog
                # (reference_codex_app_server_idle_timeout) — long
                # enough for a human to decide, short enough that a
                # runaway prompt times out cleanly.
                approved, _scope = await queue.request_approval(
                    feature_name="codex_native",
                    tool_name=kind,
                    tool_args=params or {},
                    timeout=120.0,
                )
                return {"decision": "accept" if approved else "decline"}
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "codex_approval_bridge(%s) failed: %s — declining",
                    kind, exc,
                )
                return {"decision": "decline"}

        return handler

    @staticmethod
    def _codex_policy_precheck(
        agent: Any, kind: str, params: Dict[str, Any],
    ) -> Optional[str]:
        """Run BinaryPolicy/PathPolicy against an inbound codex
        approval request. Returns ``None`` to pass through; a short
        reason string on DENY.

        Best-effort: any failure to evaluate (missing feature, missing
        policy attribute, parse error) → ``None`` pass-through, so the
        ApprovalQueue gate still gets to decide. The policy gate is
        the *anti-auto-mode-blanket* shield; if the gate itself is
        unhealthy, the queue remains the floor.
        """
        try:
            from kestrel_sovereign.features.computer_use.policy import (
                Decision,
            )
        except Exception:  # noqa: BLE001 - import path failure → pass-through
            return None
        try:
            cu = None
            features = getattr(agent, "features", {}) or {}
            if isinstance(features, dict):
                cu = features.get("ComputerUseFeature")
            else:
                cu = next(
                    (f for f in features
                     if type(f).__name__ == "ComputerUseFeature"),
                    None,
                )
            if cu is None:
                # Fail closed (codex review #1575 round 2 P1): an agent
                # without ComputerUseFeature has no policy gate to
                # evaluate, so blanket auto-mode approval would let
                # codex run any command / write any path. Decline
                # rather than pass through to the queue.
                return "no_computer_use_feature"
            if kind == "commandExecution":
                policy = getattr(cu, "_binary_policy", None)
                if policy is None:
                    return "no_binary_policy"
                # Codex's params shape evolved: legacy carried a flat
                # ``command`` string; newer
                # ``CommandExecutionRequestApprovalParams`` carries
                # ``commandActions`` (parsed argv variants). Accept
                # both. Fail closed if NEITHER is present (codex
                # review #1575 round 3 P1): with no command to
                # evaluate, the policy gate is effectively absent and
                # auto-mode would blanket-approve.
                argv_lists: List[Any] = []
                actions = params.get("commandActions") or []
                # Strict per-entry validation (codex review #1575
                # round 5 P1): any malformed entry in the array fails
                # closed. Silent-skipping a bad entry while a sibling
                # passes would let auto-mode approve the whole request.
                for act in actions or []:
                    if not isinstance(act, dict):
                        return f"malformed_command_action:{type(act).__name__}"
                    argv = act.get("argv") or act.get("command")
                    if not argv:
                        return "command_action_missing_argv"
                    argv_lists.append(argv)
                cmd = params.get("command")
                if cmd:
                    argv_lists.append(cmd)
                if not argv_lists:
                    return "no_command_in_params"
                for argv in argv_lists:
                    # Per-argv fail-closed (codex review #1575 round
                    # 4 P1): an unparseable shell string would
                    # otherwise raise out of BinaryPolicy.evaluate and
                    # the outer except would swallow → pass through to
                    # the queue auto-approve. Decline instead.
                    try:
                        result = policy.evaluate(argv)
                    except Exception as eval_exc:  # noqa: BLE001
                        logger.warning(
                            "codex_policy_precheck commandExecution: "
                            "BinaryPolicy.evaluate(%r) raised %s — "
                            "declining (fail closed)",
                            argv, eval_exc,
                        )
                        return f"binary_eval_failed:{argv!r}"
                    if getattr(result, "decision", None) is Decision.DENY:
                        return f"binary:{argv}"
            elif kind == "fileChange":
                policy = getattr(cu, "_path_policy", None)
                if policy is None:
                    return "no_path_policy"
                # Newer shape: ``fileChanges`` list with ``path`` per entry.
                # Legacy/alt: ``changes`` or ``files``.
                changes = (
                    params.get("fileChanges")
                    or params.get("changes")
                    or params.get("files")
                    or []
                )
                # Resolve cwd-relative paths before evaluating (codex
                # review #1575 round 2 P1): a deny-list of absolute
                # paths would miss ``foo/bar.txt`` even when the
                # resolved absolute is under a deny prefix. Prefer
                # params.cwd → KESTREL_CODEX_CWD → process cwd.
                cwd_str = (
                    params.get("cwd")
                    or os.environ.get("KESTREL_CODEX_CWD")
                    or str(Path.cwd())
                )
                cwd = Path(cwd_str)
                # Fail closed if no changes were sent (codex review
                # #1575 round 3 P1): an empty/malformed payload would
                # otherwise pass through to the queue and auto-mode
                # would blanket-approve.
                if not changes:
                    return "no_changes_in_params"
                # Strict per-entry validation (codex review #1575
                # round 5 P1): a non-dict entry or a dict missing
                # ``path`` / ``absolutePath`` must decline. Silently
                # skipping while a sibling passes would let auto-mode
                # approve the whole request.
                for change in changes or []:
                    if not isinstance(change, dict):
                        return f"malformed_change:{type(change).__name__}"
                    path_str = change.get("path") or change.get("absolutePath")
                    if not path_str:
                        return "change_missing_path"
                evaluated_any = False
                for change in changes or []:
                    path_str = change.get("path") or change.get("absolutePath")
                    evaluated_any = True
                    p = Path(path_str).expanduser()
                    if not p.is_absolute():
                        p = (cwd / p)
                    # Lexical absolute — avoids FileNotFoundError on
                    # resolve(strict=True) for files codex is about to
                    # CREATE. Symlink-escape protection lives in the
                    # PathPolicy's allow/deny lists themselves.
                    try:
                        p = Path(os.path.normpath(str(p)))
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        result = policy.evaluate(p, write=True)
                    except Exception as eval_exc:  # noqa: BLE001
                        # Same per-path fail-closed as commandExecution
                        # (codex review #1575 round 4 P1).
                        logger.warning(
                            "codex_policy_precheck fileChange: "
                            "PathPolicy.evaluate(%s) raised %s — "
                            "declining (fail closed)",
                            p, eval_exc,
                        )
                        return f"path_eval_failed:{p}"
                    if getattr(result, "decision", None) is Decision.DENY:
                        return f"path:{p}"
                if not evaluated_any:
                    return "no_evaluable_path_in_params"
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "codex_policy_precheck(%s) skipped: %s", kind, exc,
            )
        return None

    async def _iter_with_overflow_hint(
        self,
        app: Any,
        sink: "asyncio.Queue[dict]",
        est_payload_tokens: int,
    ) -> AsyncIterator[dict]:
        """Stream turn events; rewrite the idle-timeout error with a
        route-cap hint so the operator can act on it.

        The app-server raises ``CodexAppServerError("codex turn idle
        for Ns with no completion")`` when ChatGPT-subscription
        (openai:plan) accepts a turn whose payload exceeds the
        per-turn cap — codex's upstream websocket opens, no events
        come back, our 300s idle watchdog fires. The base message
        doesn't tell the operator *why*, which makes #1395 the kind
        of bug you only diagnose by re-reading the transport doc.
        Here we attach the route, the est payload size, and the env
        knob so the operator can raise the cap or shorten the turn.
        """
        try:
            async for ev in app.iter_turn_events(sink):
                yield ev
        except CodexAppServerError as e:
            msg = str(e)
            if "idle for" in msg and "no completion" in msg:
                from .model_catalog import get_catalog_service
                try:
                    # ``get_route_context_cap`` — caps moved to
                    # ``[route_context_caps]`` in PR #1396 round-3;
                    # ``get_context_limit`` would always report
                    # ``unset`` for the operator (codex round-4 P3).
                    cap = get_catalog_service().get_route_context_cap(
                        "openai:plan"
                    )
                except Exception:
                    cap = None
                # Branch the hint by payload vs cap (#1410). When the
                # payload is *under* the cap, "compact or raise the
                # cap" is wrong advice — the failure is an upstream
                # codex/ChatGPT-Plus stall, not a payload-too-big
                # rejection. iter_turn_events has already appended
                # stderr + codex-rs log tails to ``msg``; the operator
                # sees the actual root cause without being misdirected
                # toward compaction that can't help.
                exceeds_cap = (
                    isinstance(cap, int) and est_payload_tokens > cap
                )
                if exceeds_cap:
                    hint = (
                        f"openai:plan (codex app-server) — est turn "
                        f"payload ~{est_payload_tokens} tokens EXCEEDS "
                        f"the openai:plan per-turn cap ({cap}). "
                        "ChatGPT-Plus has a per-turn payload limit "
                        "below the model's context window — when the "
                        "payload exceeds it, codex opens the upstream "
                        "websocket but never returns response.completed. "
                        "Either compact this session's history "
                        "(kestrel context compact) or raise the cap "
                        "via KESTREL_OPENAI_PLAN_CONTEXT_CAP."
                    )
                else:
                    cap_str = f"{cap}" if isinstance(cap, int) else "unset"
                    hint = (
                        f"openai:plan (codex app-server) — est turn "
                        f"payload ~{est_payload_tokens} tokens is "
                        f"within the per-turn cap ({cap_str}); this "
                        "is a transient upstream codex / ChatGPT-Plus "
                        "stall, not a payload-cap problem. Retry once; "
                        "if persistent, switch to a non-plan route "
                        "(openai:api) or check server logs for the "
                        "codex stderr / codex-rs log tail (kept out "
                        "of this message to avoid cross-session leaks)."
                    )
                # The original ``e`` is a ``CodexAppServerTransportError``
                # (idle timeout from ``iter_turn_events``). Preserve the
                # transport classification so streaming.py's harness-owned
                # check still recognizes it after the hint rewrite. See
                # #1429.
                rewritten = CodexAppServerTransportError(f"{msg} — {hint}")
                # Surface the cap-vs-payload determination as an
                # attribute so ``_run_turn_with_retry`` (#1411) can gate
                # its one-shot retry on "under cap only" without
                # re-parsing the hint string. ONLY set the attribute
                # when we actually have a known integer cap to compare
                # against — leaving it unset for unknown caps means the
                # retry wrapper's ``getattr(e, "exceeds_cap", True)``
                # default kicks in and skips retry. Otherwise a missing
                # ``[route_context_caps]`` entry would let us silently
                # retry an over-cap stall and burn another 300s. Codex
                # review round 4 caught this.
                if isinstance(cap, int):
                    rewritten.exceeds_cap = exceeds_cap
                raise rewritten from e
            raise

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
            # The app-server runs tools inline mid-turn via item/tool/call;
            # without a kestrel-side executor wired through the orchestrator
            # we'd have no safe way to honor that callback. The only caller
            # that supplies tools today is the orchestrator (which always
            # passes ``tool_executor`` per ``_make_inline_tool_executor``).
            # If you're hitting this from anywhere else, route that
            # tool-using call through the orchestrator path, or pick
            # ``openai:api`` (the standard stateless interface composes
            # with kestrel's tool model directly).
            raise CodexAppServerError(
                "openai:plan (codex app-server) requires a tool_executor "
                "callback when tools are provided — orchestrator-driven "
                "dispatch only. For non-orchestrator tool-using callers, "
                "use openai:api or thread an executor through "
                "generate_with_messages / stream_with_tool_detection."
            )
        thread_id, fresh = await self._ensure_thread(
            app, session_id, model, instructions, dyn,
        )
        turn_input = _build_turn_input(input_messages, fresh_thread=fresh)

        # Serialize per-thread: the app-server runs one active turn per
        # thread; concurrent ``_run_turn`` calls that share a thread
        # (same session_id) must not race on the turn-sink / handler
        # registrations. ``setdefault`` keeps lock identity stable.
        lock = self._thread_locks.setdefault(thread_id, asyncio.Lock())
        await lock.acquire()

        # Defense against concurrent same-session idle-timeout invalidation:
        # while we were queued on this thread lock, another same-session
        # turn may have idle-timed-out on the same thread and popped the
        # session→thread cache in its ``except CodexAppServerTransportError``
        # arm below. If our cached ``thread_id`` no longer matches what
        # ``_session_threads[session_id]`` points at, the thread is hung
        # on the remote side — running ``turn/start`` against it now
        # would yield stale events from the failed turn or be rejected
        # outright. Re-resolve via ``_ensure_thread`` to get a fresh
        # thread. Repeats while a poisoned thread keeps being handed to
        # us by a tight cascade of failures (extremely unlikely, but the
        # while-loop costs nothing in the common no-race case).
        # Guard on truthy ``session_id`` — empty-string session ids are
        # treated as sessionless by ``_ensure_thread`` (it gates on
        # ``if session_id``), so the cache is never written and this
        # loop would spin forever rebuilding fresh threads. Codex review
        # round 4 caught the empty-string case.
        # See #1411 codex review rounds 3 and 4.
        while session_id:
            cached = self._session_threads.get(session_id)
            if cached is not None and cached[0] == thread_id:
                break
            lock.release()
            thread_id, fresh = await self._ensure_thread(
                app, session_id, model, instructions, dyn,
            )
            turn_input = _build_turn_input(input_messages, fresh_thread=fresh)
            lock = self._thread_locks.setdefault(thread_id, asyncio.Lock())
            await lock.acquire()

        # Per-turn record of every inline-executed tool call. Surfaced
        # on the final LLMResponse so the orchestrator can render
        # chat-history breadcrumbs generically (any adapter that runs
        # tools inline uses this same channel).
        executed_log: List[Dict[str, Any]] = []
        unregisters: List[Callable[[], None]] = []
        if tool_executor is not None and dyn:
            # Only register an item/tool/call handler when tools were
            # actually advertised this turn. A handler on a text-only
            # turn would be a footgun — an unsolicited or hallucinated
            # tool request from the app-server could otherwise execute
            # against the orchestrator's full tool registry, bypassing
            # the per-turn restriction. Combined with the allow-set
            # check inside the handler, this is defense in depth.
            allowed_tools = frozenset(t["name"] for t in dyn if t.get("name"))
            # Thread-scoped registration: concurrent turns on different
            # threads each get their own ``item/tool/call`` handler and
            # the dispatcher routes by ``params.threadId``. Without
            # scoping, a second turn's registration would silently
            # overwrite an in-flight turn's handler.
            unregisters.append(app.register_server_request_handler(
                "item/tool/call",
                self._make_tool_call_handler(
                    tool_executor, thread_id, allowed_tools, executed_log,
                ),
                thread_id=thread_id,
            ))

        # #1575: bridge codex's own sandbox-approval RPCs through
        # Kestrel's ApprovalQueue when an agent is attached. Without
        # this, the app-server's native commandExecution/fileChange
        # approval requests are answered with the hardcoded decline
        # default in ``_DEFAULT_APPROVAL_REPLIES`` regardless of
        # auto-mode — silently denying gh / git / fs writes the
        # Sovereign actually authorized.
        #
        # Once-per-adapter registration: codex's approval params don't
        # reliably carry ``threadId`` (round 1 P1), so the handler is
        # global, AND registering it per-turn would let a later
        # turn's unregister tear it down while an earlier turn was
        # still in flight (round 2 P2). The lifetime is the
        # (adapter, agent) pair.
        self._ensure_codex_approval_bridge(app)

        sink = app.open_turn_sink(thread_id)
        try:
            # dynamicTools are registered at thread/start; turn/start carries
            # the user input plus per-turn overrides that mirror what
            # kestrel-claw's working client sends (run-attempt.ts:2313).
            # Without these, codex 0.131.0+ in app-server mode hangs after
            # ``session_loop: enter`` — the model/cwd/effort defaults from
            # thread/start aren't picked up by the turn pipeline.
            turn_params: Dict[str, Any] = {
                "threadId": thread_id,
                "input": [{"type": "text", "text": turn_input}],
                "cwd": os.environ.get("KESTREL_CODEX_CWD") or str(Path.cwd()),
            }
            # Reuse _model_param's sentinel filter — "auto"/"default" are
            # kestrel-side route placeholders, not real model ids; the
            # app-server rejects them. Codex review (#1388 P1) caught
            # that the previous unconditional pass would break
            # ``openai:plan`` agents whose route was configured with
            # ``model = "auto"``.
            m_for_turn = self._model_param(model)
            if m_for_turn:
                turn_params["model"] = m_for_turn
            await app.request("turn/start", turn_params, timeout=60)

            text_parts: List[str] = []
            final_text: Optional[str] = None
            tool_calls: List[ToolCall] = []
            usage: Dict[str, Optional[int]] = {}
            seen_tool_ids: set = set()
            # Tool items the user has seen a "🔧 Calling X..." line for.
            # We dedupe BOTH on start and on complete so retries or
            # split-event quirks can't produce a stray marker line that
            # the chat-UI parser would group into a phantom card.
            started_item_ids: set = set()
            completed_item_ids: set = set()

            turn_input_len_chars = len(turn_input)
            instructions_len_chars = len(instructions or "")
            # Estimated payload-cap hint for the idle-timeout branch
            # below. We don't want to import the TokenCounter here
            # (circular with agent/), so a ~4-chars-per-token estimate
            # is enough to tell the operator whether they're near the
            # cap. Real budget enforcement lives in ContextManager
            # via the catalog override #1395 wired in.
            est_payload_tokens = (
                turn_input_len_chars + instructions_len_chars
            ) // 4

            async for ev in self._iter_with_overflow_hint(
                app, sink, est_payload_tokens
            ):
                method = ev.get("method")
                p = ev.get("params") or {}
                # #1518: codex's per-session per-plan ceiling is reported on
                # the notification stream (NOT on the thread/start response —
                # confirmed by live observation 2026-06-03 against pro plan,
                # see #1518 body for the dump). ``thread/tokenUsage/updated``
                # carries ``tokenUsage.modelContextWindow`` which is the real
                # binding constraint codex enforces for THIS thread. Capture
                # it and let the catalog's discovered layer win over the
                # static config. Best-effort; no-op when the field is absent.
                if method in self._DISCOVERED_CAP_EVENT_METHODS:
                    self._record_discovered_route_cap_from_event(p)
                if method == "item/agentMessage/delta":
                    delta = p.get("delta") or ""
                    if delta:
                        text_parts.append(delta)
                        yield {"text": delta}
                elif method in (
                    "item/reasoning/textDelta",
                    "item/reasoning/summaryTextDelta",
                ):
                    d = p.get("delta") or ""
                    if d:
                        yield {"thinking": d}
                elif method == "item/started":
                    # Tool activity marker for the chat-UI parser. Fires
                    # BEFORE the kestrel-dispatched ``item/tool/call`` RPC
                    # AND before codex-native tool execution
                    # (commandExecution / fileChange / webSearch), so the
                    # user sees "🔧 Calling X..." right as codex commits
                    # to using the tool — same UX timing as the
                    # orchestrator-dispatched path (which yields the same
                    # marker shape immediately after a tool_calls
                    # detection).
                    item = p.get("item") or {}
                    itype = item.get("type") or ""
                    iid = item.get("id") or item.get("callId") or ""
                    # Only dedupe when the protocol gives us an id;
                    # falling back to ``""`` would collapse every
                    # id-less item into one dedupe slot and silently
                    # drop subsequent start markers (codex P2).
                    already_started = bool(iid) and iid in started_item_ids
                    if itype in _TOOL_ITEM_TYPES and not already_started:
                        if iid:
                            started_item_ids.add(iid)
                        label = _item_display_label(item)
                        # Shell items get a one-line command preview so
                        # the user sees what's running, not just "shell"
                        # — codex's native shell is the dominant tool in
                        # this surface, and a bare "shell" marker is the
                        # bulk of what made the chat feel opaque in
                        # Nellie's session.
                        detail = (
                            _format_command_summary(item.get("command"))
                            if itype == "commandExecution" else None
                        )
                        line = (
                            f"\U0001f527 Calling {label}: {detail}...\n"
                            if detail else f"\U0001f527 Calling {label}...\n"
                        )
                        # NOT appended to ``text_parts`` — those rebuild
                        # ``LLMResponse.content`` as a clean string when
                        # the app-server doesn't deliver an agentMessage
                        # item/completed. Wire markers leaking into
                        # ``content`` would surface as literal "🔧"
                        # characters in audit logs, narration checks,
                        # and any non-chat reader of the response.
                        yield {"text": line}
                elif method == "item/completed":
                    item = p.get("item") or {}
                    itype = item.get("type")
                    iid = item.get("id") or item.get("callId") or ""
                    if itype == "agentMessage":
                        final_text = item.get("text") or final_text
                    elif itype in _TOOL_ITEM_TYPES:
                        # Completion marker. Emit even if we missed the
                        # paired item/started (some app-server builds
                        # collapse start+complete for very fast items)
                        # — without a completion line the chat-UI parser
                        # leaves the card in a "running" state forever.
                        # Same dedupe carve-out as item/started: only
                        # gate on iid when one was supplied; missing-id
                        # items always emit so they're not collapsed
                        # into a single phantom (codex P2).
                        already_completed = bool(iid) and iid in completed_item_ids
                        if already_completed:
                            # Defensive: never two ✓ lines for one item.
                            pass
                        else:
                            if iid:
                                completed_item_ids.add(iid)
                            label = _item_display_label(item)
                            status = (
                                item.get("status")
                                or item.get("state")
                                or ""
                            )
                            failed = (
                                isinstance(status, str)
                                and status.lower() in ("failed", "error", "blocked")
                            ) or bool(item.get("error"))
                            line = (
                                f"❌ {label} failed\n"
                                if failed
                                else f"✓ {label} complete\n"
                            )
                            # See start-marker note: NOT appended to
                            # ``text_parts``; markers stay on the wire,
                            # not in LLMResponse.content.
                            yield {"text": line}
                        # The kestrel-dispatched subset additionally
                        # surfaces ToolCallStarted for the honesty-layer
                        # revising sentinel + populates the executed_log
                        # the orchestrator folds into chat history. The
                        # codex-native subset (shell/patch/web) stays
                        # marker-only — those execute inside the
                        # app-server and have no kestrel-side audit row.
                        if itype in _KESTREL_DISPATCHED_TOOL_ITEM_TYPES:
                            # See dedupe carve-out above. The
                            # ``ToolCallStarted`` honesty-layer signal
                            # MUST fire per tool — collapsing two
                            # id-less calls into one would silently
                            # break narration auditing.
                            if iid and iid in seen_tool_ids:
                                continue
                            if iid:
                                seen_tool_ids.add(iid)
                            raw_args = item.get("arguments")
                            if isinstance(raw_args, str):
                                try:
                                    raw_args = json.loads(raw_args)
                                except ValueError:
                                    raw_args = {"_raw": raw_args}
                            tc = ToolCall(
                                id=iid,
                                name=item.get("name") or item.get("tool") or "",
                                arguments=(
                                    raw_args if isinstance(raw_args, dict) else {}
                                ),
                            )
                            # Critical: the app-server has ALREADY executed
                            # this tool inline via our item/tool/call handler.
                            # Yield ToolCallStarted for UI/honesty-layer
                            # signaling, but DO NOT add to ``tool_calls`` —
                            # surfacing it would make the orchestrator
                            # re-dispatch through ``_execute_tool_batch`` and
                            # duplicate every tool's side effects.
                            yield {"tool_call": tc}
                elif method == "thread/tokenUsage/updated":
                    usage = _usage_from(p.get("tokenUsage") or {})
                elif method == "turn/failed":
                    err = p.get("error") or p.get("turn", {}).get("error") or {}
                    raise CodexAppServerError(
                        f"codex turn failed: "
                        f"{err.get('message') or err or 'unknown'}"
                    )
                elif method == "error":
                    # Standalone error event — codex emits this (instead of,
                    # or in addition to, ``turn/failed``) when the upstream
                    # Responses API rejects the request. Pre-#1438 this was
                    # silently ignored: the read loop kept going, the turn
                    # ended with empty content, and the caller got HTTP 200
                    # with ``response=""`` — the honesty floor was breached
                    # because codex told us the request failed and we
                    # relayed it as a clean response. Smoking gun: a
                    # ChatGPT-Plus account asking for ``gpt-5.5-pro`` got
                    # "model is not supported when using Codex with a
                    # ChatGPT account" silently swallowed instead of
                    # surfaced. ``willRetry=True`` is codex's signal that
                    # it's about to retry internally — leave those alone;
                    # only escalate when retry is off.
                    will_retry = bool(p.get("willRetry", False))
                    if not will_retry:
                        err = p.get("error") or {}
                        msg = (
                            err.get("message") if isinstance(err, dict)
                            else str(err)
                        )
                        raise CodexAppServerError(
                            f"codex turn failed: {msg or 'unknown'}"
                        )
                elif method == "turn/completed":
                    # Terminal event. If the turn entered failed state
                    # (upstream rejection that didn't already raise via
                    # the ``error`` branch), surface it as an exception
                    # rather than letting the loop fall through to the
                    # final yield with empty content. Same honesty
                    # rationale as the ``error`` branch.
                    turn_info = p.get("turn") or {}
                    if turn_info.get("status") == "failed":
                        err = turn_info.get("error") or {}
                        msg = (
                            err.get("message") if isinstance(err, dict)
                            else str(err)
                        )
                        raise CodexAppServerError(
                            f"codex turn completed in failed state: "
                            f"{msg or 'unknown'}"
                        )
                    # Otherwise terminates iter_turn_events normally.

            content = final_text if final_text is not None else "".join(text_parts)
            yield {
                "final": (content or None, tool_calls or None, usage),
                # The orchestrator reads this via getattr(response,
                # "executed_tool_calls", None) and produces standard
                # chat-history breadcrumbs from it.
                "executed": list(executed_log),
            }
        except CodexAppServerTransportError as e:
            # Invalidate the session→thread cache BEFORE the ``finally``
            # below releases the per-thread lock, so a same-session call
            # queued on ``_session_locks[session_id]`` sees the popped
            # cache and starts a fresh thread via ``_ensure_thread``.
            # If we relied on the retry wrapper to pop after this method
            # returns, the lock release would already have unblocked the
            # queued caller — it could grab the still-hung thread before
            # our pop ran. See #1411 codex review round 2.
            #
            # Only pop on the assistant-stage idle marker. A
            # ``turn/start`` RPC timeout or a connection drop don't
            # leave codex-rs with an in-flight turn on this thread;
            # popping there would just lose conversation context for
            # no safety gain.
            msg = str(e)
            # Match ``_ensure_thread``'s truthy session_id semantics —
            # empty-string session_ids never write to ``_session_threads``,
            # so popping them is a no-op but also signals intent.
            if (
                session_id
                and "idle for" in msg
                and "no completion" in msg
            ):
                self._session_threads.pop(session_id, None)
            raise
        finally:
            app.close_turn_sink(thread_id)
            for _unreg in unregisters:
                try:
                    _unreg()
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    pass
            lock.release()

    async def _run_turn_with_retry(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        session_id: Optional[str],
        tool_executor: Optional[ToolExecutor],
    ) -> AsyncIterator[dict]:
        """One-shot retry around :meth:`_run_turn` for the narrow case
        of a transient codex/ChatGPT-Plus idle stall.

        Mirrors openclaw's "harness owns transport" pattern (commit
        ``3a64dc7623`` and the per-turn watchdog work that followed):
        a codex idle-timeout under cap is the kind of transient
        upstream stall that one retry can recover, but escalating it
        to a different provider would give the user a wrong-model
        answer (the case #1431 just locked down at the streaming
        boundary).

        Retry fires only when **all** gates pass:

        1. The exception is :class:`CodexAppServerTransportError`.
        2. The message contains ``"idle for"`` *and* ``"no completion"``
           — i.e. the assistant-stage idle watchdog tripped, not a
           prompt-stage RPC timeout or a connection drop.
        3. The exception's ``exceeds_cap`` attribute is ``False``.
           Over-cap stalls are caused by the ChatGPT-Plus per-turn
           payload limit; retry without compaction can't recover.
        4. Zero events have been yielded to the caller on this attempt.
           If any text / thinking / tool-call delta has already
           reached the caller, a retry would duplicate observable
           output — worse than failing.

        Tool-bearing turns (``tools is not None``) bypass the retry
        wrapper entirely; the safe-replay analysis for inline tool
        execution is deferred to a future ticket per #1411 scope. The
        wrapper then behaves exactly like ``_run_turn``.

        On retry exhaustion, the second-attempt exception is augmented
        with ``" (retried once after first idle-timeout; second attempt
        also failed)"`` so operators can see the retry happened. The
        transport classification and ``exceeds_cap`` attribute are
        preserved so the streaming.py harness-owned check still kicks
        in (#1429 contract).
        """
        if tools:
            # See docstring: tool-bearing turns are outside the safe
            # retry envelope today. Delegate straight through.
            # Truthy gate (``if tools:``) — an empty list means the
            # caller declared "no tools," same effect as ``None``, and
            # the underlying turn won't be tool-bearing. Skipping
            # retry for ``tools=[]`` would deny the one-shot recovery
            # to callers that normalize "no tools" to an empty list.
            # Codex review round 4 caught this.
            async for ev in self._run_turn(
                model, messages, tools, session_id, tool_executor,
            ):
                yield ev
            return

        MAX_ATTEMPTS = 2
        for attempt in range(1, MAX_ATTEMPTS + 1):
            events_yielded = 0
            try:
                async for ev in self._run_turn(
                    model, messages, None, session_id, None,
                ):
                    events_yielded += 1
                    yield ev
                return
            except CodexAppServerTransportError as e:
                if attempt >= MAX_ATTEMPTS:
                    raise self._annotate_retry_exhaustion(e) from e
                if events_yielded > 0:
                    raise
                msg = str(e)
                if "idle for" not in msg or "no completion" not in msg:
                    raise
                # ``exceeds_cap`` is False only when the hint-rewrite
                # in ``_iter_with_overflow_hint`` actually ran and
                # determined the payload is within the cap. Default
                # to True (skip retry) when the attribute is absent —
                # safer than retrying a stall we don't fully classify.
                if getattr(e, "exceeds_cap", True):
                    raise
                # Belt-and-suspenders cache invalidation. ``_run_turn``
                # itself pops the cache in its ``except`` arm BEFORE
                # releasing the per-thread lock — that's the race-free
                # invalidation point. This pop is redundant in the
                # normal path; kept so a future refactor that drops
                # ``_run_turn``'s pop still has the wrapper's pop as a
                # last line of defense. Truthy guard matches
                # ``_ensure_thread``'s ``if session_id`` gate so an
                # empty-string id (treated as sessionless upstream)
                # doesn't trigger a spurious pop attempt.
                if session_id:
                    self._session_threads.pop(session_id, None)
                wait_s = _codex_retry_wait_seconds()
                logger.warning(
                    "codex idle-timeout under cap with zero events; "
                    "retrying once on fresh thread after %.1fs "
                    "(session=%s, model=%s)",
                    wait_s, session_id, model,
                )
                await asyncio.sleep(wait_s)

    @staticmethod
    def _annotate_retry_exhaustion(
        exc: CodexAppServerTransportError,
    ) -> CodexAppServerTransportError:
        """Wrap a second-attempt transport error with retry context.

        Preserves the transport classification (so the streaming
        harness-owned check still recognizes it) and the
        ``exceeds_cap`` attribute (so any future caller-side logic
        can read it).
        """
        suffix = " (retried once after first idle-timeout; second attempt also failed)"
        annotated = CodexAppServerTransportError(f"{exc}{suffix}")
        if hasattr(exc, "exceeds_cap"):
            annotated.exceeds_cap = exc.exceeds_cap
        return annotated

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
        executed: List[Dict[str, Any]] = []
        async for ev in self._run_turn_with_retry(
            model, messages, tools, session_id, tool_executor,
        ):
            if "final" in ev:
                content, tool_calls, usage = ev["final"]
                executed = ev.get("executed") or []
        resp = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
        )
        # Attach inline-executed tool record as a runtime attribute. The
        # SDK ``LLMResponse`` dataclass is not frozen, so this is a
        # legal (if currently undocumented) extension. A follow-up SDK
        # release will formalize ``executed_tool_calls`` as a typed
        # field; the orchestrator already reads via ``getattr`` so the
        # typed upgrade is a no-op behaviorally.
        if executed:
            resp.executed_tool_calls = executed
        return resp

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
        # Uses ``_run_turn_with_retry`` so a transient codex idle stall
        # gets one chance to recover before the error surfaces. See #1411.
        async for ev in self._run_turn_with_retry(
            model, messages, None, session_id, None,
        ):
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
                resp = LLMResponse(
                    content=content,
                    tool_calls=tcs,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                )
                executed = ev.get("executed") or []
                if executed:
                    resp.executed_tool_calls = executed
                yield resp

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
