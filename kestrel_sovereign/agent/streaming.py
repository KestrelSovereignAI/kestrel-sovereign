"""Streaming response handling for Kestrel Agent."""
import logging
import os
import time
from decimal import Decimal
from typing import Dict, Any

from kestrel_sovereign.llm.adapter import LLMResponse


class StreamingMixin:
    """Mixin class providing streaming response methods."""

    async def process_input_streaming(
        self,
        user_input: str,
        model_override: str = None,
        audit_before_streaming: bool = False,
        session_id: str = None
    ):
        """
        Streaming version of process_input. Yields text chunks as generated.

        Uses single-call streaming with tool detection pattern:
        1. Build context and feature tools (same as process_input)
        2. Single streaming LLM call that yields text AND detects tools
        3. If tool calls detected at end, execute them and continue streaming
        4. Text chunks streamed to user in real-time

        This eliminates the "double LLM call" pattern where a non-streaming
        call was made first just to detect tools.

        Args:
            user_input: The user's message
            model_override: Optional model to use
            audit_before_streaming: Whether to audit before streaming
            session_id: Optional session ID to load conversation context from a specific session
        """
        # CONSTITUTION AUDIT CHECK: Trigger periodic integrity audits
        await self._maybe_audit()

        # Commands are not streamable - delegate to non-streaming handler
        if user_input.startswith("!"):
            result = await self.process_input(user_input, model_override, session_id=session_id)
            yield result
            return

        # Store user message (linked to session for resumed conversations)
        await self.privacy_agent.add_conversation("user", user_input, session_id=session_id)

        # Build full context using context_manager (same as process_input)
        force_local_only = not self.privacy_agent.privacy_config.allows_cloud_llm()
        privacy_mode = self.privacy_agent.privacy_mode.name

        # Get session-filtered conversation history
        history = await self.privacy_agent.get_conversation_history(limit=50, session_id=session_id)

        # Get constitution the same way as process_input
        constitution = await self._get_governing_constitution()

        # Pass the session-filtered history so context_manager uses it
        context_result = await self.context_manager.build_context(
            query=user_input,
            constitution=constitution,
            include_briefing=True,
            include_memories=True,
            include_rag=True,
            privacy_mode=privacy_mode,
            conversation_history=history,
        )

        # Build user prompt with context (same as process_input)
        prompt = self.user_prompt_template.format(
            context="[Context included in system prompt]",
            query=user_input
        )
        system_prompt = context_result.system_prompt

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
        ):
            if isinstance(item, str):
                # Text chunk - yield immediately for real-time streaming
                full_response.append(item)
                if not audit_before_streaming:
                    yield item
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

            # Execute tools and stream the final response
            if audit_before_streaming:
                # For pre-audit, we need the full response first
                response_text = await self._handle_orchestrator_response(
                    response=tool_response,
                    feature_tools=feature_tools,
                    system_prompt=system_prompt,
                    force_local_only=force_local_only,
                    effective_model=effective_model,
                    user_message=user_input
                )
                audited_response = await self._perform_integrity_audit(response_text)
                await self.privacy_agent.add_conversation("assistant", audited_response, session_id=session_id)
                chunk_size = 50
                for i in range(0, len(audited_response), chunk_size):
                    yield audited_response[i:i + chunk_size]
            else:
                # Stream tokens as they arrive from LLM after tool execution
                tool_response_chunks = []
                async for chunk in self._handle_orchestrator_response_streaming(
                    response=tool_response,
                    feature_tools=feature_tools,
                    system_prompt=system_prompt,
                    force_local_only=force_local_only,
                    effective_model=effective_model,
                    user_message=user_input
                ):
                    tool_response_chunks.append(chunk)
                    yield chunk
                # Store the full response after streaming completes
                await self.privacy_agent.add_conversation("assistant", "".join(tool_response_chunks), session_id=session_id)
        else:
            # No tool calls - text was already streamed (unless audit_before_streaming)
            final_text = "".join(full_response)

            if audit_before_streaming:
                # Audit the accumulated response, then yield in chunks
                async for chunk in self._stream_text_with_pre_audit(final_text, session_id=session_id):
                    yield chunk
            else:
                # Text was already yielded above, just handle post-audit
                if self.audit_enabled:
                    try:
                        audit_result = await self.get_audit_response(final_text)
                        risk_level = audit_result.get("risk_level", 3)
                        if risk_level >= 3:
                            reason = audit_result.get("reasoning", "No reasoning provided.")
                            warning = (
                                f"\n\n⚠️ **INTEGRITY WARNING**: Response failed post-hoc audit. "
                                f"Risk level: {risk_level}. Reason: {reason}"
                            )
                            yield warning
                            final_text += warning
                    except Exception as e:
                        logging.warning(f"Post-hoc audit failed: {e}")

                await self.privacy_agent.add_conversation("assistant", final_text, session_id=session_id)

    async def _stream_text_with_pre_audit(self, response_text: str, session_id: str = None):
        """Audit text response, then stream it."""
        audited_response = await self._perform_integrity_audit(response_text)
        await self.privacy_agent.add_conversation("assistant", audited_response, session_id=session_id)

        chunk_size = 50
        for i in range(0, len(audited_response), chunk_size):
            yield audited_response[i:i + chunk_size]

    async def _stream_text_with_post_audit(self, response_text: str, session_id: str = None):
        """Stream text response, audit after completion."""
        full_response = response_text

        chunk_size = 50
        for i in range(0, len(full_response), chunk_size):
            yield full_response[i:i + chunk_size]

        if self.audit_enabled:
            try:
                audit_result = await self.get_audit_response(full_response)
                risk_level = audit_result.get("risk_level", 3)

                if risk_level >= 3:
                    reason = audit_result.get("reasoning", "No reasoning provided.")
                    warning = (
                        f"\n\n⚠️ **INTEGRITY WARNING**: Response failed post-hoc audit. "
                        f"Risk level: {risk_level}. Reason: {reason}"
                    )
                    yield warning
                    logging.warning(f"Streamed response failed audit. Risk: {risk_level}")
                    full_response += warning
                else:
                    logging.info(f"Post-hoc audit passed. Risk level: {risk_level}")
            except Exception as e:
                logging.warning(f"Post-hoc audit failed: {e}")

        await self.privacy_agent.add_conversation("assistant", full_response, session_id=session_id)

    async def _perform_integrity_audit(self, response: str, retry_count: int = 0) -> str:
        """Uses a high-capability model to audit the response for constitutional alignment."""
        if retry_count >= 2:
            logging.error("Integrity audit failed after multiple retries.")
            return "Error: Integrity audit failed."

        audit_fee = Decimal("0.1")
        if not self.wallet.can_afford_audit(audit_fee):
            logging.warning("Cannot afford integrity audit.")
            return response

        if not self.audit_enabled:
            return response

        logging.info("Performing integrity audit on response...")
        try:
            audit_result = await self.get_audit_response(response)
        except Exception as e:
            audit_mode = os.environ.get("KESTREL_AUDIT_MODE", "strict").lower()
            if audit_mode == "skip":
                logging.warning(f"Audit skipped due to error: {e}")
                return response
            elif audit_mode == "warn":
                logging.warning(f"Audit failed: {e}. Response unverified.")
                return f"[UNVERIFIED] {response}"
            else:
                logging.error(f"Audit failed: {e}. Blocking in strict mode.")
                return "Error: Could not verify response compliance."

        risk_level = audit_result.get("risk_level", 3)
        reason = audit_result.get("reasoning", "No reasoning provided.")

        if risk_level == 3 and "provider failed" in reason.lower():
            audit_mode = os.environ.get("KESTREL_AUDIT_MODE", "strict").lower()
            if audit_mode == "skip":
                return response
            elif audit_mode == "warn":
                return f"[UNVERIFIED - {reason}] {response}"
            else:
                return "Error: Could not verify response compliance."

        if risk_level < 3:
            logging.info(f"Integrity audit passed with risk level {risk_level}.")
            await self.wallet.deduct_audit_fee(audit_fee, "Integrity Audit")
            return response
        else:
            logging.warning(f"Integrity audit failed! Risk level {risk_level}. Reason: {reason}")
            return f"SYSTEM_CORRECTION: Response was unconstitutional. Reason: {reason}."
