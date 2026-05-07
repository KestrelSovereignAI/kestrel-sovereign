"""Streaming response handling for Kestrel Agent."""
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sdk.llm import ToolCallStarted
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.security.input_guardrails import (
    wrap_user_input,
    check_prompt_injection,
    append_security_addendum,
)
from kestrel_sovereign.telemetry import start_span, end_span


class StreamingMixin:
    """Mixin class providing streaming response methods."""

    async def process_input_streaming(
        self,
        user_input: str,
        model_override: str = None,
        audit_before_streaming: bool = False,
        session_id: str = None,
        caller=None,
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
                        user_input, model_override, session_id, _otel_span
                    ):
                        yield chunk
        except Exception as exc:
            end_span(_otel_span, error=exc)
            raise
        else:
            end_span(_otel_span)
            return

    async def _process_input_streaming_traced_locked(
        self, user_input, model_override, session_id, _otel_span
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
        system_prompt = context_result.system_prompt

        # Append the security + honesty addenda. Single source of truth
        # for assembly order; see append_security_addendum's docstring.
        system_prompt = append_security_addendum(system_prompt)

        # Add cached features section
        if self._cached_features_prompt:
            system_prompt = f"{system_prompt}{self._cached_features_prompt}"

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

        async for item in self.llm_service.stream_with_tool_detection(
            messages=messages,
            tools=feature_tools if feature_tools else None,
            force_local_only=force_local_only,
            model_override=effective_model,
            system_prompt=system_prompt,
            session_id=session_id,
        ):
            if isinstance(item, str):
                # Text chunk - yield immediately for real-time streaming
                full_response.append(item)
                yield item
            elif isinstance(item, ToolCallStarted):
                # Honesty-layer signal (#1042 layer 2 / #1045): the LLM
                # has just begun emitting a tool call. Any pre-tool prose
                # the user is currently watching is about to be obsolete
                # — the agent will substitute the tool's actual result.
                # Fire a "revising" SSE event so subscribers can clear
                # the in-flight bubble; carries the request_id for pane
                # routing and the marker's `index` for ordering.
                await self._emit_revising_event(item, session_id=session_id)
            elif isinstance(item, LLMResponse):
                # Tool calls detected at end of stream
                tool_response = item

        # Log LLM response
        llm_duration = int((time.time() - llm_start) * 1000)
        has_tool_calls = tool_response is not None and tool_response.has_tool_calls

        logging.info(f"[AGENTIC-STREAM] LLM stream complete: has_tool_calls={has_tool_calls}, text_chunks={len(full_response)}")

        await self.observability_store.log_tool_response(
            event_id=llm_event_id,
            success=True,
            duration_ms=llm_duration,
        )

        # Handle tool calls if present (A2A pattern)
        if has_tool_calls:
            logging.info(f"[AGENTIC-STREAM] Tool calls: {[tc.name for tc in tool_response.tool_calls]}")
            for tc in tool_response.tool_calls:
                await self.observability_store.log_tool_call(
                    agent_name=self.did,
                    tool_name=f"llm_tool_call:{tc.name}",
                    metadata={"arguments": tc.arguments}
                )

            # Stream tokens as they arrive from LLM after tool execution
            tool_response_chunks = []
            tool_events = []
            async for chunk in self._handle_orchestrator_response_streaming(
                response=tool_response,
                feature_tools=feature_tools,
                system_prompt=system_prompt,
                force_local_only=force_local_only,
                effective_model=effective_model,
                user_message=user_input,
                tool_events=tool_events,
                session_id=session_id,
            ):
                tool_response_chunks.append(chunk)
                yield chunk
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
            tool_final_text = await self._fire_post_response_hook(tool_final_text, session_id)
            meta = {'tool_events': tool_events} if tool_events else None
            await self._persist_assistant_turn_safely(
                tool_final_text, metadata=meta, session_id=session_id
            )
        else:
            # No tool calls - text was already streamed above
            final_text = "".join(full_response)
            final_text = await self._fire_post_response_hook(final_text, session_id)
            await self._persist_assistant_turn_safely(
                final_text, metadata=None, session_id=session_id
            )

        # Fire STOP hook (streaming response cycle complete)
        hooks_manager = getattr(self, "hooks_manager", None)
        if hooks_manager:
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.STOP.value,
            )
            await hooks_manager.execute_hooks_parallel(
                HookEvent.STOP, hook_input
            )

    async def _emit_revising_event(
        self,
        marker: ToolCallStarted,
        session_id: Optional[str] = None,
    ) -> None:
        """Emit a ``revising`` event when a ToolCallStarted marker arrives.

        Routed through ``self.emit_event`` so the existing
        ``/api/agent/notifications/sse`` channel forwards it to the
        browser as ``event: revising / data: {...}``. Frontend (Wave 5C)
        listens on that channel and clears the in-flight optimistic
        message bubble for the matching ``request_id``.

        Failures here must never break the chat stream — the user is
        already receiving text through the parallel channel. ``emit_event``
        itself swallows per-listener errors; this wrapper additionally
        guards the case where the host class doesn't expose ``emit_event``
        at all (e.g. tests mocking a bare agent).
        """
        emit_event = getattr(self, "emit_event", None)
        if not callable(emit_event):
            return
        try:
            await emit_event("revising", {
                "type": "revising",
                "request_id": getattr(self, "_current_request_id", None),
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
        """
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

    async def _fire_post_response_hook(self, response_text: str, session_id: str = None) -> str:
        """Fire POST_RESPONSE hooks on completed response text.

        In streaming mode the text has already been sent to the client, so
        DENY prevents storage (and logs the issue) while MODIFY can annotate
        what gets stored.

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
        )
        hook_output = await hooks_manager.execute_hooks(HookEvent.POST_RESPONSE, hook_input)
        if hook_output.permission_decision == PermissionDecision.DENY:
            logger.warning(f"POST_RESPONSE hook denied (streaming): {hook_output.permission_reason}")
            return f"[Response blocked by audit: {hook_output.permission_reason}]"
        elif hook_output.updated_input and "response_text" in hook_output.updated_input:
            return hook_output.updated_input["response_text"]
        return response_text
