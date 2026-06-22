"""Streaming response handling for Kestrel Agent."""
import asyncio
import inspect
import json
import logging
import time
from typing import Dict, Any, Optional

from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sdk.llm import ToolCallStarted
from kestrel_sovereign.llm.adapter import LLMResponse, ThinkingDelta
from kestrel_sovereign.security.input_guardrails import (
    wrap_user_input,
    check_prompt_injection,
)
from kestrel_sovereign.telemetry import start_span, end_span


# Wave 5E in-band revising sentinel — kestrel-sovereign #1086.
#
# The chat client uses this sentinel as the AUTHORITATIVE boundary
# between pre-tool prose and post-tool synthesis on the
# /api/agent/stream text/plain channel. It's strictly ordered with
# the chunks themselves, so unlike the parallel SSE `revising` event
# it can't race the post-tool chunks.
#
# Wire format: ``\x1eKESTREL:REVISE:<json>\x1e``
#   * ``\x1e`` (Record Separator, ASCII 30) bookends — chosen because
#     LLMs don't emit it, it's UTF-8-safe, and HTTP middleware
#     doesn't strip it.
#   * Fixed prefix ``KESTREL:REVISE:`` — namespace + intent.
#   * JSON payload: ``{"index": int, "tool_call_id": str|None,
#     "tool_name": str|None}``. Same fields as the SSE event minus
#     request_id/session_id (the in-band sentinel is implicitly
#     scoped to the stream that carries it).
#
# The Wave 5C SSE `revising` event is still emitted for non-chat
# consumers (audit dashboards, accessibility tools); chat clients
# treat the two signals as idempotent — whichever arrives first sets
# ``pane.pendingRevise``; the other is a no-op.
REVISE_SENTINEL_PREFIX = "\x1eKESTREL:REVISE:"
REVISE_SENTINEL_SUFFIX = "\x1e"

# #1662 eager-vision read-path bounds (independent of the upload endpoint's
# per-file/count caps, since a content-addressed read can name any stored
# hash). Keep peak memory for one turn's pasted images bounded.
_MAX_EAGER_IMAGES = 6
_MAX_EAGER_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MB (a touch over the 10 MB upload cap)


def _looks_like_image(data: bytes) -> bool:
    """Magic-number check that ``data`` is a real PNG/JPEG/GIF/WEBP.

    The client's declared ``kind``/``mime`` are not trusted: a tampered ref
    could mark a stored PDF/text hash as an inline image, and the image
    pipeline defaults unknown bytes to JPEG, so a non-image would be shipped to
    a vision provider as a bogus image. Sniff the actual bytes instead — only
    these four signatures are accepted for eager vision.
    """
    return (
        data.startswith(b"\x89PNG")
        or data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )
THINKING_SENTINEL_PREFIX = "\x1eKESTREL:THINK:"
THINKING_SENTINEL_SUFFIX = "\x1e"
# #1659: tool activity is the last signal class still emitted as
# regex-scraped emoji text. It now rides the same in-band sentinel
# channel as REVISE/THINK. Payload:
#   {"phase":"start"|"done"|"error","name":str,"index":int|None,
#    "ms":int|None,"detail":str|None}
# Like REVISE/THINK it is NEVER persisted in the assistant text — the
# structured ``tool_events`` metadata (now position-stamped) is the
# single source of truth for rendering tool cards, live and on reload.
TOOL_SENTINEL_PREFIX = "\x1eKESTREL:TOOL:"
TOOL_SENTINEL_SUFFIX = "\x1e"

# All three sentinel classes share the \x1e suffix and are mutually
# exclusive in the stream; this tuple drives every strip/extract loop.
_ALL_SENTINEL_PREFIXES = (
    REVISE_SENTINEL_PREFIX,
    THINKING_SENTINEL_PREFIX,
    TOOL_SENTINEL_PREFIX,
)
_SENTINEL_SUFFIX = REVISE_SENTINEL_SUFFIX  # "\x1e" — identical for all


def _build_tool_sentinel(
    phase: str,
    name: str,
    index: Optional[int] = None,
    ms: Optional[int] = None,
    detail: Optional[str] = None,
) -> str:
    """Construct an in-band tool-activity sentinel.

    ``phase`` is "start" | "done" | "error". ``detail`` carries the
    shell-command preview (start) or error text (error); ``ms`` the
    dispatch duration (done).
    """
    payload = json.dumps({
        "phase": phase,
        "name": name,
        "index": index,
        "ms": ms,
        "detail": detail,
    }, separators=(",", ":"))
    return f"{TOOL_SENTINEL_PREFIX}{payload}{TOOL_SENTINEL_SUFFIX}"


def is_only_sentinels(text: str) -> bool:
    """True when ``text`` is one or more complete sentinels and nothing
    else (no visible prose). Used to keep a tool/think sentinel chunk
    from materializing a pending revise paragraph boundary — sentinels
    are wire bytes, not the visible text the weld waits for.
    """
    clean, _ = _parse_stream_sentinels(text)
    return text != "" and clean == ""


def _parse_stream_sentinels(text: str, base_offset: int = 0):
    """Strip all in-band sentinels from ``text`` and extract structured
    tool parts, in a single pass.

    Returns ``(clean_text, tool_parts)`` where ``tool_parts`` is a list
    of the TOOL sentinel payloads, each augmented with ``pos`` — the
    character offset into ``clean_text`` (plus ``base_offset``) at which
    the tool sentinel sat. ``pos`` is what lets the renderer place a tool
    card back at the right point between prose segments, live and on
    reload, without the marker living in the prose.

    REVISE sentinels weld a ``\\n\\n`` paragraph boundary at their point
    (mirroring the chat client's ``stripStreamSentinels`` — #1547); THINK
    and TOOL sentinels are stripped without welding (they are not prose
    boundaries). Operates on a complete string, so the cross-packet edge
    cases the client handles don't arise here.
    """
    if not any(p in text for p in _ALL_SENTINEL_PREFIXES):
        return text, []
    result = ""
    tool_parts: list = []
    i = 0
    n = len(text)
    weld_pending = False  # a removed revise sentinel awaits its next visible char
    while i < n:
        nxt_idx = -1
        nxt_prefix = ""
        for p in _ALL_SENTINEL_PREFIXES:
            idx = text.find(p, i)
            if idx >= 0 and (nxt_idx < 0 or idx < nxt_idx):
                nxt_idx, nxt_prefix = idx, p
        seg_end = nxt_idx if nxt_idx >= 0 else n
        seg = text[i:seg_end]
        if seg:
            if weld_pending and seg.strip():
                if result and not result[-1].isspace() and not seg[0].isspace():
                    result += "\n\n"
                weld_pending = False
            result += seg
        if nxt_idx < 0:
            break
        close = text.find(_SENTINEL_SUFFIX, nxt_idx + len(nxt_prefix))
        if close < 0:
            # Unterminated sentinel at the tail — drop the remainder,
            # matching the client's partial-buffer behavior.
            break
        if nxt_prefix == TOOL_SENTINEL_PREFIX:
            payload = text[nxt_idx + len(nxt_prefix):close]
            try:
                evt = json.loads(payload)
                if isinstance(evt, dict):
                    evt["pos"] = base_offset + len(result)
                    tool_parts.append(evt)
            except (ValueError, TypeError):
                pass
        elif nxt_prefix == REVISE_SENTINEL_PREFIX:
            weld_pending = True
        i = close + len(_SENTINEL_SUFFIX)
    return result, tool_parts


def _tool_parts_to_events(parts: list) -> list:
    """Convert parsed TOOL sentinel parts (phase/name/ms/detail/pos) into the
    persisted ``tool_events`` metadata shape (type/tool/ms/error/pos). Used on
    paths where the sentinels are the ONLY structured signal (e.g. codex-native
    tools that don't populate executed_tool_calls) so reload can reconstruct
    the cards instead of losing them after the sentinels are stripped.
    """
    events: list = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        phase = p.get("phase")
        name = p.get("name", "")
        pos = p.get("pos")
        if phase == "start":
            events.append({"type": "start", "tool": name, "pos": pos})
        elif phase == "done":
            ev = {"type": "complete", "tool": name, "pos": pos}
            if p.get("ms") is not None:
                ev["ms"] = p["ms"]
            events.append(ev)
        elif phase == "error":
            events.append({
                "type": "error", "tool": name,
                "error": p.get("detail") or "", "pos": pos,
            })
    return events


_TOOL_EVENT_PHASE = {"start": "start", "complete": "done", "error": "error"}


def _stamp_tool_event_positions(tool_events: list, parts: list) -> list:
    """Assign each ``tool_events`` entry a ``pos`` (clean-text offset) by
    matching it to the TOOL sentinel part that produced it.

    The by-ref ``tool_events`` (built server-side) and the in-band sentinel
    ``parts`` (which carry the positions) can arrive in different orders — the
    orchestrator emits all start sentinels up front, then the batch appends
    start/complete interleaved — so we match each event to the next unconsumed
    part with the same (phase, name) rather than relying on positional order.
    A startless terminal event (e.g. a follow-up-timeout ``error`` for ``llm``)
    therefore keeps its OWN position instead of inheriting the prior tool's.
    Events with no matching part fall back to the most recent stamped pos.
    Mutates + returns the list.
    """
    consumed = [False] * len(parts)
    last_pos = 0
    for ev in tool_events:
        if not isinstance(ev, dict):
            continue
        want_phase = _TOOL_EVENT_PHASE.get(ev.get("type"))
        name = ev.get("tool", "")
        matched = None
        for i, p in enumerate(parts):
            if consumed[i]:
                continue
            if p.get("phase") == want_phase and p.get("name", "") == name:
                matched = i
                break
        if matched is not None:
            consumed[matched] = True
            pos = parts[matched].get("pos")
            if isinstance(pos, int):
                last_pos = pos
        ev["pos"] = last_pos
    return tool_events


def _build_revise_sentinel(marker: ToolCallStarted) -> str:
    """Construct the in-band revise sentinel for a ToolCallStarted marker."""
    payload = json.dumps({
        "index": marker.index,
        "tool_call_id": marker.id,
        "tool_name": marker.name,
    }, separators=(",", ":"))
    return f"{REVISE_SENTINEL_PREFIX}{payload}{REVISE_SENTINEL_SUFFIX}"


def _build_thinking_sentinel(delta: ThinkingDelta) -> str:
    """Construct an in-band thinking sentinel for chat thought bubbles."""
    payload = json.dumps({
        "content": delta.content,
        "provider": delta.provider,
    }, separators=(",", ":"))
    return f"{THINKING_SENTINEL_PREFIX}{payload}{THINKING_SENTINEL_SUFFIX}"


def strip_revise_sentinels(chunk: str) -> str:
    """Strip all complete in-band stream-control sentinels from a chunk.

    Server-side helper for non-chat consumers of
    ``process_input_streaming`` — voice/TTS, bridge stream, anything
    that re-publishes chunks downstream. The chat /stream endpoint
    passes chunks through verbatim because the chat *client* strips
    them, but other consumers would otherwise display or speak
    ``\\x1eKESTREL:REVISE:...`` literally. Codex P1 of #1089 caught
    the TTS-reading-aloud regression.

    Handles complete sentinels only — split sentinels (across chunks)
    leak the partial in this function. Server-side consumers that
    care about the split-chunk case should buffer at their own layer.
    In practice the server emits the sentinel as a single Python
    yield, so single-chunk delivery is the overwhelming common case.
    """
    prefixes = _ALL_SENTINEL_PREFIXES
    if not any(prefix in chunk for prefix in prefixes):
        return chunk
    out = []
    i = 0
    while i < len(chunk):
        matches = [
            (idx, prefix)
            for prefix in prefixes
            for idx in [chunk.find(prefix, i)]
            if idx >= 0
        ]
        if not matches:
            out.append(chunk[i:])
            break
        prefix_idx, prefix = min(matches, key=lambda pair: pair[0])
        out.append(chunk[i:prefix_idx])
        close_idx = chunk.find(
            REVISE_SENTINEL_SUFFIX,
            prefix_idx + len(prefix),
        )
        if close_idx < 0:
            # Split sentinel — no close in this chunk. Drop everything
            # from the prefix on; the sentinel-close half lands in
            # the next chunk and looks like leading wire metadata
            # there. Single-chunk-yield from the server makes this
            # path effectively unreachable.
            break
        i = close_idx + len(REVISE_SENTINEL_SUFFIX)
    return "".join(out)


def _strip_and_weld_revise_sentinels(text: str) -> str:
    """Strip in-band stream sentinels and weld a paragraph boundary at
    each revise point — the server-side mirror of the chat client's
    ``stripStreamSentinels`` (kestrel-sovereign #1547).

    Why this exists: the post-tool half of a streamed turn can carry
    embedded revise sentinels (the orchestrator emits one before each
    follow-up tool call). Persisting the raw join would (a) leak
    ``\\x1eKESTREL:REVISE:...`` wire bytes into the assistant row and
    (b) glue the prose on either side of a removed sentinel into
    "...part one.🔧 part two". The browser strips the same markers and
    welds a break where they sat between two non-whitespace chars; this
    function reproduces that exactly so the persisted turn equals what
    the user actually saw.

    Operates on a complete string (not a packet stream), so the
    cross-packet edge cases the client handles don't arise here. The
    pre-tool half needs no welding of its own: its boundary against the
    post-tool half was already recorded as a ``\\n\\n`` in
    ``full_response`` at the first ``ToolCallStarted`` marker, mirroring
    the client's weld at the first revise sentinel.

    Thin wrapper over :func:`_parse_stream_sentinels` (#1659) — callers
    that only want the cleaned text keep this name; the tool-part
    extraction is ignored here.
    """
    clean, _tool_parts = _parse_stream_sentinels(text)
    return clean


class StreamingMixin:
    """Mixin class providing streaming response methods."""

    @staticmethod
    def _lazy_attachment_hint(attachments) -> str:
        """A compact note listing this turn's LAZY (non-inline) attachments so
        the agent knows their ids for `read_attachment`. Empty when there are
        none. Eager (inline) images are folded as vision and not listed.
        """
        if not attachments:
            return ""
        lazy = [
            a for a in attachments
            if isinstance(a, dict) and a.get("hash") and not a.get("inline")
        ]
        if not lazy:
            return ""
        lines = [
            f"- {a.get('name') or 'attachment'} (id: {a['hash']})"
            for a in lazy
        ]
        return (
            "\n\n[Attachments available to read with the read_attachment tool:\n"
            + "\n".join(lines)
            + "\nCall read_attachment with an id above to read that file.]"
        )

    async def _resolve_eager_images(self, attachments) -> list:
        """Resolve this turn's *inline* image attachments (#1662 eager vision)
        to raw bytes for the model.

        ``inline`` marks an image pasted or dropped straight into the composer
        — the user's intent is "see this now", so it rides this turn as vision
        input. Non-inline attachments are lazy references the agent reads on
        demand via a tool (PR C), and documents are always lazy; both are
        skipped here. The ref is already persisted on the user turn, so an
        attachment whose bytes are missing/unreadable is skipped without losing
        the history record.

        Resolution is bounded independently of the upload endpoint. The
        content-addressed read can name *any* stored hash (up to the 50 MB
        store cap), not just files that came through the 10 MB upload gate, so
        a turn could otherwise pull many large blobs into memory at once. We
        cap both the number of eager images and each image's size, and log
        loudly when something is dropped.
        """
        if not attachments:
            return []
        storage = getattr(self, "storage", None)
        if storage is None or not hasattr(storage, "retrieve_file"):
            return []
        images: list = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            if att.get("kind") != "image" or not att.get("inline"):
                continue
            content_hash = att.get("hash")
            if not isinstance(content_hash, str):
                continue
            if len(images) >= _MAX_EAGER_IMAGES:
                logging.warning(
                    "Eager vision: more than %d inline images this turn; "
                    "extras ignored.", _MAX_EAGER_IMAGES,
                )
                break
            try:
                data = await storage.retrieve_file(content_hash)
            except Exception as exc:
                logging.warning(
                    "Eager vision: could not load attachment %s: %s",
                    content_hash[:12], exc,
                )
                continue
            if not data:
                continue
            if len(data) > _MAX_EAGER_IMAGE_BYTES:
                logging.warning(
                    "Eager vision: attachment %s is %d bytes (> %d cap); "
                    "skipped.", content_hash[:12], len(data),
                    _MAX_EAGER_IMAGE_BYTES,
                )
                continue
            if not _looks_like_image(data):
                # Client lied about kind/mime, or the file isn't really an
                # image — don't ship non-image bytes to a vision provider.
                logging.warning(
                    "Eager vision: attachment %s is not a recognized image "
                    "(PNG/JPEG/GIF/WEBP); skipped.", content_hash[:12],
                )
                continue
            images.append(data)
        return images

    async def process_input_streaming(
        self,
        user_input: str,
        model_override: str = None,
        audit_before_streaming: bool = False,
        session_id: str = None,
        caller=None,
        request_id: Optional[str] = None,
        attachments: Optional[list] = None,
    ):
        """
        Streaming version of process_input. Yields text chunks as generated.

        Uses single-call streaming with tool detection pattern:
        1. Build context and feature tools (same as process_input)
        2. Single streaming LLM call that yields text AND detects tools
        3. If tool calls detected at end, execute them and continue streaming
        4. Text chunks streamed to user in real-time

        Args:
            user_input: The user's message
            model_override: Optional model to use
            audit_before_streaming: Deprecated, ignored. Kept for API compat.
            session_id: Optional session ID to load conversation context from a specific session
            caller: Optional CallerContext with auth identity and role.
            request_id: The endpoint-assigned id for this stream. Used as
                the routing key on emitted ``revising`` events so the
                frontend can match the event to the correct in-flight
                pane when multiple streams are active concurrently.
                Falls back to ``self._current_request_id`` (the legacy
                agent-global) when omitted, but callers should prefer
                explicit values — the global is overwritten by the next
                ``register_active_request`` call and races between
                overlapping streams can otherwise misroute events.
        """
        # CONSTITUTION AUDIT CHECK: Trigger periodic integrity audits
        await self._maybe_audit()

        # Commands are not streamable - delegate to non-streaming handler
        if user_input.startswith("!"):
            result = await self.process_input(user_input, model_override, session_id=session_id, caller=caller)
            yield result
            return

        # Start OTEL span for streaming request lifecycle
        _otel_span = start_span("agent.process_input_streaming", {
            "agent.did": self.did,
            "agent.session_id": session_id or "",
            "agent.input_length": len(user_input),
            "agent.streaming": True,
        })

        try:
            transition_lock = self._get_privacy_transition_lock()
            async with transition_lock:
                async with self._turn_lifecycle():
                    async for chunk in self._process_input_streaming_traced_locked(
                        user_input, model_override, session_id, _otel_span,
                        request_id=request_id, attachments=attachments,
                    ):
                        yield chunk
        except Exception as exc:
            end_span(_otel_span, error=exc)
            raise
        else:
            end_span(_otel_span)
            return

    async def _process_input_streaming_traced_locked(
        self, user_input, model_override, session_id, _otel_span,
        request_id: Optional[str] = None, attachments: Optional[list] = None,
    ):
        """Inner streaming logic wrapped in an OTEL span.

        Caller MUST hold the turn lifecycle (CONVERSATION lock). The
        wrapper `process_input_streaming` enters the lifecycle once;
        delegating here from a held-lifecycle context avoids the
        non-reentrant-lock self-deadlock that the dispatcher (Phase 1)
        would otherwise hit when COGNITION signals route through this
        path."""
        # #1662: record THIS turn's session so tools that must scope to the
        # active conversation (read_attachment) have an authoritative value —
        # the tool-call `session_id` arg is model-controlled and usually omitted,
        # and the inline executor binds session only into hook context, not tool
        # args. The turn-lifecycle lock serializes turns per agent, so this is
        # safe per-turn.
        self._active_session_id = session_id
        # Prompt injection detection (log-only, does not block)
        check_prompt_injection(user_input)

        # Fire USER_PROMPT_SUBMIT hook
        if self.hooks_manager:
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.USER_PROMPT_SUBMIT.value,
                user_message=user_input,
            )
            hook_output = await self.hooks_manager.execute_hooks(
                HookEvent.USER_PROMPT_SUBMIT, hook_input
            )
            if hook_output.permission_decision == PermissionDecision.DENY:
                yield f"[Input rejected: {hook_output.permission_reason}]"
                return
            # The manager applies updated_input to hook_input.tool_input;
            # check if hooks modified the user_message via that path.
            if hook_input.tool_input and "user_message" in hook_input.tool_input:
                user_input = hook_input.tool_input["user_message"]

        # #1844 Stage 2: Kestrel-owned compaction for openai:plan. Compact +
        # reset the codex thread BEFORE fetching history / building context so
        # the freshly-seeded thread carries the compacted view rather than
        # letting codex auto-compact opaquely server-side. Gated on the codex
        # adapter's own recorded occupancy (not route prediction). Best-effort.
        #
        # The hook lives on KestrelAgent; StreamingMixin is reusable, so guard
        # it as OPTIONAL — a standalone-mixin host (or a partial test host) that
        # lacks the real coroutine helper is a clean no-op rather than an
        # ``await MagicMock()`` TypeError (codex review r9).
        _compact_hook = getattr(self, "_maybe_compact_codex_thread", None)
        if inspect.iscoroutinefunction(_compact_hook):
            await _compact_hook(session_id)

        # Build full context using context_manager (same as process_input)
        force_local_only = not self.privacy_agent.privacy_config.allows_cloud_llm()
        privacy_mode = self.privacy_agent.privacy_mode.name

        # Get session-filtered conversation history
        history = await self.privacy_agent.get_conversation_history(limit=50, session_id=session_id)

        # Get constitution the same way as process_input
        constitution = await self._get_governing_constitution()

        # Pass the session-filtered history so context_manager uses it
        # Fetch reflection guidance for prompt injection
        reflection_guidance = None
        reflection_feature = self.features.get("ReflectionFeature") if hasattr(self, 'features') else None
        if reflection_feature:
            try:
                reflection_guidance = await reflection_feature.get_active_guidance()
            except Exception:
                pass  # Non-critical — log already handled in feature

        context_result = await self.context_manager.build_context(
            query=user_input,
            constitution=constitution,
            include_briefing=True,
            include_memories=True,
            include_rag=True,
            privacy_mode=privacy_mode,
            conversation_history=history,
            reflection_guidance=reflection_guidance,
        )

        # B / #1309 + C / #1311 degraded-mode fail-closed (streaming
        # path). When ``build_context`` returns ``degraded_mode=True``
        # (mandatory governance floor doesn't fit, or durable-salvage
        # write failed under C's feature flag), the LLM call MUST NOT
        # proceed. Yield a refusal chunk so the stream consumer sees
        # the failure and stops, instead of silently proceeding to
        # ``stream_with_tool_detection`` against an empty context.
        # Explicit ``is True`` so MagicMock-returning test fixtures
        # don't trip the gate inadvertently.
        if getattr(context_result, "degraded_mode", False) is True:
            warn_text = " | ".join(context_result.warnings or ["context build degraded"])
            logging.error(
                "DEGRADED MODE on streaming build_context — refusing to "
                "issue the model call. Warnings: %s",
                warn_text,
            )
            yield (
                "I cannot continue this turn safely: the context window "
                f"is in a degraded state. Details: {warn_text}"
            )
            return

        # Build user prompt. `context` carries the per-turn retrieved content
        # (memories + RAG) — kept OUT of the system message so the system prefix
        # is stable across turns and prompt caches can hit (see issue #703).
        wrapped_user = wrap_user_input(user_input)
        prompt = self.user_prompt_template.format(
            context=context_result.dynamic_user_context,
            query=wrapped_user
        )

        # Canonical/transport split (#1402): persist raw wrapped user turn
        # as ``content`` and the rendered prompt (memories + RAG baked in)
        # as ``rendered_content``. History-load replays ``rendered_content``
        # verbatim so Anthropic's cache_control marker at messages[-2]
        # still compounds across turns, while every other consumer
        # (search, audit, UI, memory ingestion) sees clean user speech.
        # #1662: persist attachment refs on the user turn so the composer's
        # images/docs survive reload (and a later turn can resolve them). The
        # bytes live in the encrypted file store; only the refs ride here.
        _user_meta = {"sent_form": True}
        if attachments:
            _user_meta["attachments"] = attachments
        await self.privacy_agent.add_conversation(
            "user", wrapped_user,
            metadata=_user_meta,
            session_id=session_id,
            rendered_content=prompt,
        )
        # Apply post-build assembly via the agent helper so this path
        # uses the same route-aware budget gate as the non-streaming
        # path (#1399). Without sharing the helper, route-capped
        # subscriptions hit the same codex-stdout-close 500 from the
        # streaming endpoint that the non-streaming endpoint just got
        # protected against.
        system_prompt = self._assemble_post_build_system_prompt(
            context_result.system_prompt, context_result,
            user_prompt=prompt,
        )

        if self.extension:
            try:
                prefix = self.extension.get_system_prompt_prefix()
                if prefix:
                    system_prompt = f"{prefix}\n{system_prompt}"
            except Exception:
                pass

        # Determine model
        effective_model = model_override
        if not effective_model:
            effective_model = await self.check_solvency()

        # Build feature tools for the orchestrator (A2A pattern)
        # Includes feature dispatch tools + any direct tools from explored features
        feature_tools = self._build_all_tools()

        # Log tool availability
        tool_names = [t['function']['name'] for t in feature_tools] if feature_tools else []
        await self.observability_store.log_metric(
            agent_name=self.did,
            metric_name="feature_tools_built_streaming",
            metric_value=len(feature_tools),
            metadata={
                "tool_names": tool_names,
                "model": effective_model,
            }
        )

        # Build full messages array for multi-turn conversation
        # Format: [system, ...history, user]
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_result.messages)  # Add conversation history
        # #1662: tell the agent which LAZY attachments it can read this turn, so
        # it knows the ids to pass to `read_attachment`. Live-only — appended to
        # the LLM-bound user content, NOT the persisted prompt (keeps sent-form
        # cache stability and the raw-user extraction clean). Inline (eager)
        # image attachments are folded as vision downstream and aren't listed.
        lazy_hint = self._lazy_attachment_hint(attachments)
        messages.append({"role": "user", "content": prompt + lazy_hint})

        logging.debug(f"[CONTEXT-STREAM] Sending {len(messages)} messages to LLM")

        # Single streaming call with tool detection
        llm_start = time.time()
        llm_event_id = await self.observability_store.log_tool_call(
            agent_name=self.did,
            tool_name="llm_stream_with_tool_detection",
            metadata={
                "model": effective_model,
                "tools_count": len(feature_tools) if feature_tools else 0,
                "force_local_only": force_local_only,
                "history_messages": len(context_result.messages),
            }
        )

        # Accumulate text and watch for tool response at end
        full_response = []
        tool_response = None
        # Snapshot of streamed text at the moment the FIRST
        # ToolCallStarted marker arrives — this is the boundary between
        # "what the agent said it was about to do" and "what the agent
        # actually saw the tool return" for the narration check
        # (#1042 layer 3). ``None`` when no marker fired this turn;
        # empty string when the marker fired before any text arrived.
        pre_tool_prose_snapshot: Optional[str] = None
        # #1547: mirrors the chat client's `pendingReviseBoundary`. A
        # ToolCallStarted marker arms a paragraph boundary that is only
        # MATERIALIZED (as `\n\n`) when the next visible text actually
        # arrives — either a later text chunk in this loop (inline
        # execution) or the post-tool synthesis at the join below. Lazy,
        # not eager, so a turn that ends right after the marker (e.g. the
        # user cancels before tools run) never persists a dangling `\n\n`
        # the client never rendered.
        pending_visible_boundary = False

        # #1662 eager vision: resolve this turn's inline (pasted/dropped) image
        # attachments to bytes so the LLM service can fold them into the user
        # message in the resolved provider's native format. Lazy refs and
        # documents are skipped here — the agent reads those on demand.
        eager_images = (
            await self._resolve_eager_images(attachments) if attachments else None
        )
        cancel_token = (
            (lambda: self.is_request_cancelled(request_id))
            if request_id else None
        )

        async for item in self.llm_service.stream_with_tool_detection(
            messages=messages,
            tools=feature_tools if feature_tools else None,
            force_local_only=force_local_only,
            model_override=effective_model,
            system_prompt=system_prompt,
            session_id=session_id,
            tool_executor=self._make_inline_tool_executor(session_id),
            images=eager_images or None,
            cancel_token=cancel_token,
        ):
            # #1256: Honor stop-button cancellation INSIDE the agent
            # loop, not just at the HTTP response layer. Before this
            # check existed, /api/agent/stop only set a flag that the
            # endpoint generator inspected — the upstream LLM stream
            # kept pulling tokens (and any tool-using turn kept
            # dispatching tools with side effects) right through to
            # completion. Breaking out here closes the underlying SDK
            # stream and leaves ``tool_response is None`` so the
            # has_tool_calls branch is skipped; partial text already in
            # ``full_response`` flows through to the no-tools persist
            # path below and the cancellation marker lands in metadata.
            if request_id and self.is_request_cancelled(request_id):
                break
            if isinstance(item, str):
                # #1547: materialize a pending revise boundary lazily —
                # only when real post-marker text lands, and only when it
                # would otherwise weld two non-whitespace chars. Mirrors
                # the client weld so the persisted turn equals the render.
                # #1659: a tool sentinel is wire bytes, not the visible
                # text the weld waits for — never let it materialize (or
                # clear) the boundary, and don't weld a \n\n in front of
                # it (it'd survive the persist-time sentinel strip).
                if (
                    pending_visible_boundary
                    and item.strip()
                    and not is_only_sentinels(item)
                ):
                    acc = "".join(full_response)
                    if acc and not acc[-1].isspace() and not item[:1].isspace():
                        full_response.append("\n\n")
                    pending_visible_boundary = False
                # Text chunk - yield immediately for real-time streaming
                full_response.append(item)
                yield item
            elif isinstance(item, ThinkingDelta):
                yield _build_thinking_sentinel(item)
            elif isinstance(item, ToolCallStarted):
                # Honesty-layer signal (#1042 layer 2 / #1045): the LLM
                # has just begun emitting a tool call. Any pre-tool prose
                # the user is currently watching is about to be obsolete
                # — the agent will substitute the tool's actual result.
                # Fire a "revising" SSE event so subscribers can clear
                # the in-flight bubble; carries the request_id for pane
                # routing and the marker's `index` for ordering.
                await self._emit_revising_event(
                    item, session_id=session_id, request_id=request_id,
                )
                # Wave 5E: in-band sentinel on the chat stream itself.
                # Strictly ordered with the chunks, so the chat client
                # can never race the post-tool synthesis. The Wave 5C
                # SSE event is the reliability backup; chat clients
                # treat both as idempotent. NOT appended to
                # ``full_response`` — the persisted assistant turn
                # must not contain wire-protocol bytes.
                yield _build_revise_sentinel(item)
                # Snapshot pre-tool prose at the FIRST marker only —
                # subsequent markers arrive between tool calls of the
                # same LLM turn and don't introduce new pre-tool text
                # the user hadn't already seen. The narration check
                # downstream wants the user-visible prose that
                # PRECEDED any tool call, not the inter-tool prose.
                if pre_tool_prose_snapshot is None:
                    pre_tool_prose_snapshot = "".join(full_response)
                # #1547: arm — do NOT eagerly write — a paragraph boundary
                # at this revise point. It materializes when the next
                # visible text arrives (lazy weld above, or the pre/post
                # seam at the join below), so a turn that stops right here
                # never persists a trailing `\n\n` the client never drew.
                pending_visible_boundary = True
            elif isinstance(item, LLMResponse):
                # Tool calls detected at end of stream
                tool_response = item

        # Log LLM response
        llm_duration = int((time.time() - llm_start) * 1000)
        has_tool_calls = tool_response is not None and tool_response.has_tool_calls
        # Inline-executing adapters (codex app-server today; potentially
        # MCP/voice realtime tomorrow) execute tools INSIDE the LLM call
        # and report what they did via an ``executed_tool_calls`` runtime
        # attribute on LLMResponse. Surface that to the same persisted
        # ``tool_events`` metadata channel the chat UI subscribes to.
        inline_executed = (
            getattr(tool_response, "executed_tool_calls", None)
            if tool_response is not None else None
        )

        logging.info(f"[AGENTIC-STREAM] LLM stream complete: has_tool_calls={has_tool_calls}, text_chunks={len(full_response)}")

        await self.observability_store.log_tool_response(
            event_id=llm_event_id,
            success=True,
            duration_ms=llm_duration,
        )

        # #1256: If the user hit Stop after the LLM finished but before
        # we dispatched tools, skip the tool batch entirely. The tools
        # are about to mutate state (send messages, write files, hit
        # external APIs) — running them after the user said "stop" is
        # the surprising-side-effect bug. Persist whatever pre-tool
        # prose the LLM already yielded so the user can see what the
        # agent had been about to do.
        if has_tool_calls and request_id and self.is_request_cancelled(request_id):
            # #1659: strip any codex inline tool sentinels from the
            # partial pre-tool text before persisting.
            cancelled_text, _ = _parse_stream_sentinels("".join(full_response))
            await self._persist_assistant_turn_safely(
                cancelled_text, metadata=None, session_id=session_id,
                request_id=request_id,
            )
            return

        # Handle tool calls if present (A2A pattern)
        if has_tool_calls:
            logging.info(f"[AGENTIC-STREAM] Tool calls: {[tc.name for tc in tool_response.tool_calls]}")
            for tc in tool_response.tool_calls:
                await self.observability_store.log_tool_call(
                    agent_name=self.did,
                    tool_name=f"llm_tool_call:{tc.name}",
                    metadata={"arguments": tc.arguments}
                )

            # Stream tokens as they arrive from LLM after tool execution.
            # ``tool_results`` accumulates each call's serialized return
            # envelope as ``{tool_call_id, name, result}``. The orchestrator
            # writes into it from ``_dispatch_tool_call`` so the post-
            # response hook can run the narration check (#1042 layer 3)
            # over the same envelopes the LLM saw.
            tool_response_chunks = []
            tool_events = []
            tool_results: list = []
            async for chunk in self._handle_orchestrator_response_streaming(
                response=tool_response,
                feature_tools=feature_tools,
                system_prompt=system_prompt,
                force_local_only=force_local_only,
                effective_model=effective_model,
                user_message=user_input,
                tool_events=tool_events,
                tool_results=tool_results,
                session_id=session_id,
                request_id=request_id,
                images=eager_images or None,
            ):
                if isinstance(chunk, ThinkingDelta):
                    yield _build_thinking_sentinel(chunk)
                    continue
                tool_response_chunks.append(chunk)
                yield chunk
                # #1256: belt-and-suspenders cancel check at the outer
                # boundary. The inner orchestrator generator also checks
                # ``is_request_cancelled`` at iteration boundaries; this
                # outer break stops accumulating tokens if a cancel
                # arrives between the inner check and the next yield.
                if request_id and self.is_request_cancelled(request_id):
                    break
            # Persist only the post-tool synthesis in ``content``. Any
            # pre-tool prose was already retracted from the SSE client by
            # the ToolCallStarted/revising protocol, so storing it in
            # content makes the lie reappear on history reload. Keep that
            # prose in metadata instead; ContextBuilder replays it to the
            # next-turn LLM context so #877 self-recall is preserved.
            # #1547/#1659: strip + weld in-band sentinels from BOTH halves
            # and recover each tool sentinel's clean-text position. The pre
            # half can carry codex inline tool sentinels (emitted on the LLM
            # stream itself); the post half carries the orchestrator's
            # per-batch start sentinels plus any follow-up revise sentinels.
            pre_tool_text, pre_parts = _parse_stream_sentinels(
                "".join(full_response)
            )
            post_tool_text, post_parts = _parse_stream_sentinels(
                "".join(tool_response_chunks)
            )
            # Materialize the pre/post seam boundary armed by the first
            # ToolCallStarted marker — but ONLY when the marker fired
            # (`pending_visible_boundary`) and the seam would weld two
            # non-whitespace chars. This mirrors the client, which welds
            # at the in-band revise sentinel; when no marker fired the
            # client glues, so we glue too (plain concat).
            if (
                pending_visible_boundary
                and pre_tool_text
                and post_tool_text
                and not pre_tool_text[-1].isspace()
                and not post_tool_text[:1].isspace()
            ):
                seam = "\n\n"
            else:
                seam = ""
            context_replay_text = pre_tool_text + seam + post_tool_text
            pending_visible_boundary = False
            # Position-stamp tool_events so the renderer can place each card
            # back between the right prose segments (live + on reload). Pre
            # positions already index the final text (pre half is at the
            # front); post positions shift past the clean pre half + seam.
            post_base = len(pre_tool_text) + len(seam)
            combined_parts = (
                list(pre_parts)
                + [{**p, "pos": post_base + p["pos"]} for p in post_parts]
            )
            _stamp_tool_event_positions(tool_events, combined_parts)
            # Narration check (#1042 layer 3): the hook receives the
            # pre-tool prose snapshot taken at the first ToolCallStarted
            # marker boundary, plus the tool calls + result envelopes.
            # If the marker never fired (model emitted tool_use without
            # streaming markers), fall back to the pre-tool half of
            # full_response — that's still the text the user saw before
            # the tool ran.
            pre_tool_for_audit = (
                _parse_stream_sentinels(pre_tool_prose_snapshot)[0]
                if pre_tool_prose_snapshot is not None
                else pre_tool_text
            )
            tool_calls_payload = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in tool_response.tool_calls
            ]
            tool_final_text = await self._fire_post_response_hook(
                post_tool_text, session_id,
                pre_tool_prose=pre_tool_for_audit,
                tool_calls=tool_calls_payload,
                tool_results=tool_results,
            )
            # tool_results carry the structured call/result envelopes;
            # persisting them alongside tool_events means a future
            # conversation-history reader can reconstruct *which tools
            # produced which output*, not just *that some tool ran*.
            # Without this the actual result content was lost after the
            # turn — same gap the other-agent diagnosis flagged for the
            # codex inline_executed branch, but it applied identically
            # here (every adapter that goes through the multi-iteration
            # tool loop). Close it in both branches uniformly.
            meta: Optional[Dict[str, Any]] = None
            if tool_events or tool_results or pre_tool_text:
                meta = {}
                if pre_tool_text:
                    meta['pre_tool_reasoning'] = {
                        'content': pre_tool_text,
                        'seam': seam,
                        'context_replay': context_replay_text,
                    }
                if tool_events:
                    meta['tool_events'] = tool_events
                if tool_results:
                    meta['tool_results'] = tool_results
            await self._persist_assistant_turn_safely(
                tool_final_text, metadata=meta, session_id=session_id,
                request_id=request_id,
            )
            final_assistant_text = tool_final_text
            stop_tool_results: Optional[list] = tool_results
            # Derive stop_tool_calls from the accumulated tool_results
            # envelopes — multi-iteration tool flows (model calls A,
            # sees result, calls B) require this so tool_calls and
            # tool_results stay aligned across all iterations.
            # Cancellation edge: when the request was stopped after the
            # initial LLM emitted tool_calls but before any result was
            # appended, `tool_results` is empty but the LLM *did* emit
            # tools. Fall back to the initial ``tool_calls_payload`` in
            # that case so a STOP subscriber can distinguish
            # "cancelled mid-tool" from "no tools at all" (codex review).
            if tool_results:
                stop_tool_calls: Optional[list] = [
                    {
                        "id": env.get("tool_call_id"),
                        "name": env.get("name"),
                        "arguments": env.get("arguments"),
                    }
                    for env in tool_results
                ]
            elif tool_calls_payload:
                stop_tool_calls = list(tool_calls_payload)
            else:
                stop_tool_calls = None
        elif inline_executed:
            # Inline-executed branch: the adapter ran tools mid-call
            # (codex app-server's item/tool/call RPC). No
            # ``_dispatch_tool_call`` ran, so synthesize the same
            # ``tool_events`` / ``tool_results`` envelopes the
            # orchestrator-dispatched path would have produced and
            # persist via the metadata path the chat UI already reads.
            #
            # Also persist into ``a2a_tool_dispatches`` via
            # ``_log_tool_dispatch`` so structured observability sees
            # the dominant streaming-path execution today (gpt-5.5 plan
            # / codex app-server). Without this, the inline branch
            # silently bypassed the structured log and runtime
            # tool-call queries returned stale data from before the
            # inline-execution migration.
            from kestrel_sovereign.agent.orchestrator_engine import (
                OrchestratorEngineMixin,
                _extract_inline_error_fields,
            )
            from kestrel_sovereign.features.base import _serialize_tool_result
            from kestrel_sovereign.security.narration_check import (
                summarize_tool_result_for_audit,
            )
            synth_tool_events = []
            tool_results: list = []
            features_by_tool_name = (
                self._visible_features_by_tool_name()
                if hasattr(self, "_visible_features_by_tool_name") else {}
            )
            for e in inline_executed:
                synth_tool_events.append({"type": "start", "tool": e["name"]})
                synth_tool_events.append({"type": "complete", "tool": e["name"], "ms": 0})
                serialized = _serialize_tool_result(e["result"])
                tool_results.append({
                    "tool_call_id": e["id"],
                    "name": e["name"],
                    "arguments": e["arguments"],
                    "result": summarize_tool_result_for_audit(serialized),
                })
                # Structured dispatch log (codex app-server inline path).
                # Latency=0 by design — accurate per-call latency needs
                # adapter-level instrumentation, see codex_adapter's
                # ``_make_inline_tool_executor``.
                args = e["arguments"]
                adapter = "inline:" + OrchestratorEngineMixin._tool_call_adapter(
                    self, e["name"], features_by_tool_name,
                )
                err_msg, err_class = _extract_inline_error_fields(e["result"])
                await OrchestratorEngineMixin._log_tool_dispatch(
                    self,
                    tool_name=e["name"],
                    adapter=adapter,
                    args=args if isinstance(args, dict) else {},
                    result=e["result"],
                    session_id=session_id,
                    dispatch_start=time.time(),
                    error_class=err_class,
                    error_message=err_msg,
                )
            # #1659: the codex tool sentinels rode the LLM stream itself, so
            # they're embedded in full_response. Strip them out of the
            # persisted text and recover their positions to stamp onto the
            # synthesized events (inline_executed order == start-sentinel
            # order on the wire).
            context_replay_text, inline_parts = _parse_stream_sentinels(
                "".join(full_response)
            )
            _stamp_tool_event_positions(synth_tool_events, inline_parts)
            tool_calls_payload = [
                {"id": e["id"], "name": e["name"], "arguments": e["arguments"]}
                for e in inline_executed
            ]
            pre_tool_text = _parse_stream_sentinels(
                pre_tool_prose_snapshot or ""
            )[0]
            post_tool_text = context_replay_text
            seam = ""
            if pre_tool_text and context_replay_text.startswith(pre_tool_text):
                post_tool_text = context_replay_text[len(pre_tool_text):]
                if post_tool_text.startswith("\n\n"):
                    seam = "\n\n"
                    post_tool_text = post_tool_text[2:]
            final_text = await self._fire_post_response_hook(
                post_tool_text, session_id,
                pre_tool_prose=pre_tool_text,
                tool_calls=tool_calls_payload,
                tool_results=tool_results,
            )
            # See parallel comment in the has_tool_calls branch above:
            # tool_results in the persisted metadata preserves the
            # actual call envelopes (id, name, arguments, summarized
            # result) so a future loader can see what produced what,
            # not just *that* something ran.
            await self._persist_assistant_turn_safely(
                final_text,
                metadata={
                    **({
                        "pre_tool_reasoning": {
                            "content": pre_tool_text,
                            "seam": seam,
                            "context_replay": context_replay_text,
                        }
                    } if pre_tool_text else {}),
                    "tool_events": synth_tool_events,
                    "tool_results": tool_results,
                },
                session_id=session_id, request_id=request_id,
            )
            final_assistant_text = final_text
            stop_tool_results = tool_results
            stop_tool_calls = tool_calls_payload
        else:
            # No tool calls - text was already streamed above. Pre-tool
            # prose / tool_calls / tool_results all stay None: a hook
            # writing the narration check should treat ``tool_results
            # is None`` as "this turn didn't call any tools, no
            # narration to verify". (#1659: strip in-band sentinels from the
            # persisted text.) Codex-native tools (commandExecution, fileChange,
            # webSearch) that don't populate executed_tool_calls land here with
            # their TOOL sentinels embedded — build tool_events from them so the
            # cards survive reload rather than vanishing once stripped.
            final_text, tool_parts = _parse_stream_sentinels("".join(full_response))
            final_text = await self._fire_post_response_hook(
                final_text, session_id,
            )
            _no_tool_events = _tool_parts_to_events(tool_parts)
            await self._persist_assistant_turn_safely(
                final_text,
                metadata=({"tool_events": _no_tool_events} if _no_tool_events else None),
                session_id=session_id,
                request_id=request_id,
            )
            final_assistant_text = final_text
            stop_tool_results = None
            stop_tool_calls = None

        # Post-response memory pipeline — parity with the non-streaming path
        # (kestrel_agent._post_response_pipeline). Routed through the SINGLE
        # privacy gate so EPHEMERAL/ISOLATED (and deidentified) turns never
        # derive durable state (temporal patterns, concept-graph nodes,
        # emotional metadata, embeddings) from raw user input. The guard is
        # also the gate's call site for the streaming path: when it returns
        # True the pipeline is never entered, so no background memory work is
        # scheduled. Mirrors the non-streaming fire at kestrel_agent.py.
        post_pipeline = getattr(self, "_post_response_pipeline", None)
        if callable(post_pipeline) and not self._privacy_blocks_background_memory():
            await post_pipeline(user_input, final_assistant_text, session_id)

        # Fire STOP hook (streaming response cycle complete).
        #
        # STOP HookInput carries the turn's user message, the final visible
        # assistant text, and (when applicable) tool_calls + tool_results
        # so per-turn subscribers — e.g. the kestrel-feature-reflection
        # `on_stop` handler for #1238 — don't have to round-trip through
        # storage to reconstruct the turn that just completed. Without
        # this, every subscriber would issue its own query against
        # `conversation_history` / `a2a_tool_dispatches` at hook-fire
        # time, multiplying read load and risking races with the
        # persistence write above. SDK HookInput already declares
        # `user_message`, `response_text`, `tool_calls`, `tool_results`
        # fields (used by POST_RESPONSE) — we're just populating them
        # for STOP too.
        hooks_manager = getattr(self, "hooks_manager", None)
        if hooks_manager:
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.STOP.value,
                user_message=user_input,
                response_text=final_assistant_text,
                tool_calls=stop_tool_calls,
                tool_results=stop_tool_results,
            )
            await hooks_manager.execute_hooks_parallel(
                HookEvent.STOP, hook_input
            )

    async def _emit_revising_event(
        self,
        marker: ToolCallStarted,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Emit a ``revising`` event when a ToolCallStarted marker arrives.

        Routed through ``self.emit_event`` so the existing
        ``/api/agent/notifications/sse`` channel forwards it to the
        browser as ``event: revising / data: {...}``. Frontend (Wave 5C)
        listens on that channel and clears the in-flight optimistic
        message bubble for the matching ``request_id``.

        ``request_id`` is the routing key. Callers should always pass
        the endpoint-assigned id for the stream that produced the
        marker — the legacy ``self._current_request_id`` fallback is
        agent-global and is overwritten by the next concurrent stream's
        ``register_active_request`` call, so two overlapping streams
        could otherwise misroute events to the wrong pane.

        Failures here must never break the chat stream — the user is
        already receiving text through the parallel channel. ``emit_event``
        itself swallows per-listener errors; this wrapper additionally
        guards the case where the host class doesn't expose ``emit_event``
        at all (e.g. tests mocking a bare agent).
        """
        emit_event = getattr(self, "emit_event", None)
        if not callable(emit_event):
            return
        effective_request_id = request_id
        if effective_request_id is None:
            effective_request_id = getattr(self, "_current_request_id", None)
        try:
            await emit_event("revising", {
                "type": "revising",
                "request_id": effective_request_id,
                "session_id": session_id,
                "index": marker.index,
                "tool_call_id": marker.id,
                "tool_name": marker.name,
            })
        except Exception:
            logging.warning(
                "Failed to emit revising event for index=%s",
                getattr(marker, "index", None), exc_info=True,
            )

    async def _persist_assistant_turn_safely(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Persist the assistant turn under cancellation-safe handling.

        FastAPI's ``StreamingResponse`` cancels the wrapping async
        generator on client disconnect — browser nav, agent switch,
        network blip. Without protection, a disconnect after the last
        chunk is yielded but BEFORE this insert completes would lose
        the assistant turn entirely: the user already saw the response,
        but the next turn's history loader can't find it. Surfaced by
        Meridian's "I don't see my own quantum response" report.

        ``asyncio.shield`` keeps the persist task alive even when the
        outer generator is cancelled. The exception handler logs and
        emits a metric so production can detect this happening at all.
        We never re-raise — the request is already over from the
        client's perspective; raising here would only mask the original
        cancellation.

        When ``request_id`` is supplied and the request has been
        cancelled by the user (stop button), stamp ``cancelled: True``
        into the persisted metadata. The next turn's history loader and
        any UI surface that re-renders the turn can then distinguish a
        user-aborted partial from a normally-completed turn. The check
        runs here rather than in the caller so the marker lands even on
        the disconnect-before-persist path that the shield protects.
        """
        if request_id and self.is_request_cancelled(request_id):
            metadata = {**(metadata or {}), "cancelled": True}
        try:
            await asyncio.shield(
                self.privacy_agent.add_conversation(
                    "assistant", text, metadata=metadata, session_id=session_id
                )
            )
        except asyncio.CancelledError:
            # Outer task cancelled mid-persist. shield() means the
            # add_conversation coroutine keeps running to completion in
            # the background — we just don't get to await it. Re-raise
            # so the cancellation propagates correctly.
            raise
        except Exception as exc:
            logging.error(
                "Failed to persist assistant turn (session_id=%s): %s",
                session_id, exc, exc_info=True,
            )
            try:
                await self.observability_store.log_metric(
                    agent_name=self.did,
                    metric_name="assistant_turn_persist_failed",
                    metric_value=1.0,
                    metadata={
                        "session_id": session_id or "",
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc)[:500],
                    },
                )
            except Exception:
                # Telemetry failures must never propagate from a
                # post-response persist path. If observability is also
                # broken, the logged ERROR above is the last line of
                # defense.
                pass

    async def _fire_post_response_hook(
        self,
        response_text: str,
        session_id: str = None,
        *,
        pre_tool_prose: Optional[str] = None,
        tool_calls: Optional[list] = None,
        tool_results: Optional[list] = None,
    ) -> str:
        """Fire POST_RESPONSE hooks on completed response text.

        In streaming mode the text has already been sent to the client, so
        DENY prevents storage (and logs the issue) while MODIFY can annotate
        what gets stored.

        Args:
            response_text: Final assembled assistant text (post-tool
                synthesis when tools fired; otherwise the only text).
            session_id: Active session id.
            pre_tool_prose: Text the agent streamed before the first
                ``ToolCallStarted`` marker arrived. ResponseAuditHook
                narration check (#1042 layer 3) compares this against
                the actual ``tool_results`` to catch confident-lie
                patterns ("Saved!" before the tool returned). ``None``
                or empty string means no pre-tool prose was streamed.
            tool_calls: Tool calls issued in this turn, as JSON-shaped
                dicts ``{id, name, arguments}``.
            tool_results: Tool result envelopes observed in this turn,
                as ``{tool_call_id, name, result}`` dicts. ``result``
                is whatever ``_serialize_tool_result`` produced —
                usually a ``ToolResult.to_dict()`` envelope (#1042
                layer 4) or a legacy ``{success, ...}`` dict.

        Returns:
            Possibly modified response_text.
        """
        hooks_manager = getattr(self, "hooks_manager", None)
        if not hooks_manager or not hooks_manager.get_enabled_hooks(HookEvent.POST_RESPONSE):
            return response_text

        hook_input = HookInput(
            session_id=session_id or "",
            hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text=response_text,
            pre_tool_prose=pre_tool_prose,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        hook_output = await hooks_manager.execute_hooks(HookEvent.POST_RESPONSE, hook_input)
        if hook_output.permission_decision == PermissionDecision.DENY:
            logging.warning(f"POST_RESPONSE hook denied (streaming): {hook_output.permission_reason}")
            return f"[Response blocked by audit: {hook_output.permission_reason}]"
        elif hook_output.updated_input and "response_text" in hook_output.updated_input:
            return hook_output.updated_input["response_text"]
        return response_text
