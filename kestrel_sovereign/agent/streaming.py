"""Streaming response handling for Kestrel Agent."""
import asyncio
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
THINKING_SENTINEL_PREFIX = "\x1eKESTREL:THINK:"
THINKING_SENTINEL_SUFFIX = "\x1e"


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
    prefixes = (REVISE_SENTINEL_PREFIX, THINKING_SENTINEL_PREFIX)
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


class StreamingMixin:
    """Mixin class providing streaming response methods."""

    async def process_input_streaming(
        self,
        user_input: str,
        model_override: str = None,
        audit_before_streaming: bool = False,
        session_id: str = None,
        caller=None,
        request_id: Optional[str] = None,
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
                        request_id=request_id,
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
        request_id: Optional[str] = None,
    ):
        """Inner streaming logic wrapped in an OTEL span.

        Caller MUST hold the turn lifecycle (CONVERSATION lock). The
        wrapper `process_input_streaming` enters the lifecycle once;
        delegating here from a held-lifecycle context avoids the
        non-reentrant-lock self-deadlock that the dispatcher (Phase 1)
        would otherwise hit when COGNITION signals route through this
        path."""
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
        prompt = self.user_prompt_template.format(
            context=context_result.dynamic_user_context,
            query=wrap_user_input(user_input)
        )

        # Store the user turn AFTER context build (so memory retrieval sees
        # the pre-current-turn state) and AFTER hook mutation (so stored bytes
        # match sent bytes). Persisting the rendered sent-form — not raw
        # user_input — makes history-load at turn N+1 byte-match what was
        # sent at turn N, which is the prerequisite for Anthropic's
        # cache_control marker at messages[-2] to compound across turns.
        await self.privacy_agent.add_conversation(
            "user", prompt, metadata={"sent_form": True}, session_id=session_id
        )
        # Apply post-build assembly via the agent helper so this path
        # uses the same route-aware budget gate as the non-streaming
        # path (#1399). Without sharing the helper, route-capped
        # subscriptions hit the same codex-stdout-close 500 from the
        # streaming endpoint that the non-streaming endpoint just got
        # protected against.
        system_prompt = self._assemble_post_build_system_prompt(
            context_result.system_prompt, context_result,
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
        messages.append({"role": "user", "content": prompt})

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

        async for item in self.llm_service.stream_with_tool_detection(
            messages=messages,
            tools=feature_tools if feature_tools else None,
            force_local_only=force_local_only,
            model_override=effective_model,
            system_prompt=system_prompt,
            session_id=session_id,
            tool_executor=self._make_inline_tool_executor(session_id),
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
            cancelled_text = "".join(full_response)
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
            # Persist the FULL visible assistant text — pre-tool reasoning
            # the LLM emitted before deciding to call tools (full_response,
            # already yielded to the client at the first stream loop) plus
            # the post-tool synthesizing answer (tool_response_chunks).
            # The user saw both streams concatenated; persisting only the
            # post-tool half meant the next turn's history loader was blind
            # to any pre-tool explanation, and the agent couldn't see the
            # reasoning it had just shown the user. Surfaced by Meridian's
            # "I don't see my own quantum response" transcript.
            pre_tool_text = "".join(full_response)
            post_tool_text = "".join(tool_response_chunks)
            tool_final_text = pre_tool_text + post_tool_text
            # Narration check (#1042 layer 3): the hook receives the
            # pre-tool prose snapshot taken at the first ToolCallStarted
            # marker boundary, plus the tool calls + result envelopes.
            # If the marker never fired (model emitted tool_use without
            # streaming markers), fall back to the pre-tool half of
            # full_response — that's still the text the user saw before
            # the tool ran.
            pre_tool_for_audit = (
                pre_tool_prose_snapshot
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
                tool_final_text, session_id,
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
            if tool_events or tool_results:
                meta = {}
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
            from kestrel_sovereign.features.base import _serialize_tool_result
            from kestrel_sovereign.security.narration_check import (
                summarize_tool_result_for_audit,
            )
            synth_tool_events = []
            tool_results: list = []
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
            final_text = "".join(full_response)
            tool_calls_payload = [
                {"id": e["id"], "name": e["name"], "arguments": e["arguments"]}
                for e in inline_executed
            ]
            final_text = await self._fire_post_response_hook(
                final_text, session_id,
                pre_tool_prose=pre_tool_prose_snapshot or "",
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
            # narration to verify".
            final_text = "".join(full_response)
            final_text = await self._fire_post_response_hook(
                final_text, session_id,
            )
            await self._persist_assistant_turn_safely(
                final_text, metadata=None, session_id=session_id,
                request_id=request_id,
            )
            final_assistant_text = final_text
            stop_tool_results = None
            stop_tool_calls = None

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
