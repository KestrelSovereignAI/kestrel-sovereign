"""
Orchestrator engine mixin for tool execution and response handling.

Extracted from kestrel_agent.py — handles the core orchestrator loop:
- Tool execution with hook enforcement (PRE/POST_TOOL_USE)
- Non-streaming orchestrator response handling
- Streaming orchestrator response handling
- Message pruning for context pressure management
- Diminishing returns detection (stop loops producing negligible output)
- Context analysis: duplicate detection and token attribution by source
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from kestrel_sovereign.hooks import HookEvent, HookInput, PermissionDecision
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.security.input_guardrails import validate_tool_arguments
from kestrel_sovereign.telemetry import optional_span

# These constants are also defined in kestrel_agent.py — import from there at
# runtime so that env-var overrides are respected.  The mixin accesses them via
# module-level names; kestrel_agent.py re-exports them.
MAX_TOOL_ITERATIONS = None  # set by _init_constants()
MAX_TOOL_RESULT_CHARS = None
CONTEXT_RESERVE_FRACTION = None
KESTREL_DIMINISHING_THRESHOLD = None
KESTREL_MAX_LOW_DELTA = None
KESTREL_BUDGET_STOP_PCT = None
MAX_TOOL_CONCURRENCY = int(os.environ.get("KESTREL_MAX_TOOL_CONCURRENCY", "10"))


def _init_constants():
    """Lazily import constants from kestrel_agent to avoid circular imports."""
    global MAX_TOOL_ITERATIONS, MAX_TOOL_RESULT_CHARS, CONTEXT_RESERVE_FRACTION
    global KESTREL_DIMINISHING_THRESHOLD, KESTREL_MAX_LOW_DELTA, KESTREL_BUDGET_STOP_PCT
    if MAX_TOOL_ITERATIONS is None:
        from kestrel_sovereign import kestrel_agent as _ka
        MAX_TOOL_ITERATIONS = _ka.MAX_TOOL_ITERATIONS
        MAX_TOOL_RESULT_CHARS = _ka.MAX_TOOL_RESULT_CHARS
        CONTEXT_RESERVE_FRACTION = _ka.CONTEXT_RESERVE_FRACTION
        KESTREL_DIMINISHING_THRESHOLD = _ka.KESTREL_DIMINISHING_THRESHOLD
        KESTREL_MAX_LOW_DELTA = _ka.KESTREL_MAX_LOW_DELTA
        KESTREL_BUDGET_STOP_PCT = _ka.KESTREL_BUDGET_STOP_PCT


@dataclass
class IterationTracker:
    """Track consecutive low-output reasoning-only iterations for diminishing returns detection.

    Only counts iterations where the LLM produced no tool calls (reasoning-only).
    Iterations with tool calls are intentionally small text output and are not
    counted toward the diminishing returns threshold.
    """

    consecutive_low_delta: int = 0
    threshold: int = 500       # min output tokens per reasoning-only iteration
    max_low_delta: int = 5     # consecutive low-delta iterations before stop
    budget_stop_pct: int = 90  # stop if iteration > this % of max_iterations

    def record(self, response: LLMResponse) -> None:
        """Record an iteration's output and update the consecutive counter.

        Only reasoning-only iterations (no tool calls) are evaluated.
        Iterations where the LLM requested tool calls reset the counter.
        """
        if response.has_tool_calls:
            # Tool-call iterations are intentionally small — reset counter
            self.consecutive_low_delta = 0
            return

        output_tokens = response.output_tokens
        if output_tokens is None:
            # Fallback: estimate tokens from content length
            content_len = len(response.content or "")
            output_tokens = content_len // 4

        if output_tokens < self.threshold:
            self.consecutive_low_delta += 1
        else:
            self.consecutive_low_delta = 0

    def should_stop(self, iteration: int, max_iterations: int) -> bool:
        """Return True if the loop should stop due to diminishing returns."""
        # Check consecutive low-delta reasoning-only iterations
        if self.consecutive_low_delta >= self.max_low_delta:
            return True
        # Check budget exhaustion
        budget_limit = max_iterations * self.budget_stop_pct / 100
        if iteration >= budget_limit:
            return True
        return False


PREVIEW_HEAD_CHARS = 2000
PREVIEW_TAIL_CHARS = 500


def _build_persisted_preview(result_json: str, tool_name: str, original_len: int) -> str:
    """
    Build a preview for a large tool result.

    Includes the head and tail of the result so the agent sees both
    the beginning context and any trailing errors/summaries.

    Args:
        result_json: The full serialized tool result
        tool_name: Name of the tool that produced the result
        original_len: Original length in characters

    Returns:
        Preview string with head, tail, and size metadata
    """
    head = result_json[:PREVIEW_HEAD_CHARS]
    tail = result_json[-PREVIEW_TAIL_CHARS:] if original_len > PREVIEW_HEAD_CHARS + PREVIEW_TAIL_CHARS else ""

    parts = [
        f"[Large tool output — {original_len:,} chars from {tool_name}]",
        "",
        f"Preview (first {PREVIEW_HEAD_CHARS} chars):",
        head,
    ]
    if tail:
        parts.extend([
            "",
            f"... [{original_len - PREVIEW_HEAD_CHARS - PREVIEW_TAIL_CHARS:,} chars omitted] ...",
            "",
            f"Tail (last {PREVIEW_TAIL_CHARS} chars):",
            tail,
        ])

    return "\n".join(parts)


class ContextStats:
    """
    Incremental accumulator for context analysis.

    Records tool calls as they happen during dispatch, enabling O(1) status
    queries for duplicate detection and token attribution by source.

    Designed to be attached to the agent instance and reset on session change
    or context compression.
    """

    def __init__(self):
        self._last_session_id: Optional[str] = None
        self.reset()

    def check_session(self, session_id: Optional[str]):
        """Reset stats if session has changed."""
        if session_id and session_id != self._last_session_id:
            if self._last_session_id is not None:
                logging.debug(f"[CONTEXT_STATS] Session changed, resetting stats")
            self._last_session_id = session_id
            self.reset()

    def reset(self):
        """Clear all accumulated stats. Call on session change or compression."""
        # List of recorded tool calls in dispatch order
        self._calls: List[Dict[str, Any]] = []
        # Set of normalized signatures for duplicate detection
        self._seen_signatures: set = set()
        # Counts by tool name
        self._tool_counts: Dict[str, int] = {}
        # Total result chars by tool name (proxy for token cost)
        self._tool_result_chars: Dict[str, int] = {}
        # Duplicate call entries
        self._duplicates: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize_input(tool_name: str, args: dict) -> str:
        """
        Build a normalized signature for duplicate detection.

        Normalizes file paths with os.path.abspath and strips whitespace
        on string values to catch near-duplicate calls.
        """
        normalized_args = {}
        for key, value in sorted(args.items()):
            if isinstance(value, str):
                stripped = value.strip()
                # Normalize anything that looks like a file path
                if "/" in stripped or "\\" in stripped:
                    try:
                        stripped = os.path.abspath(stripped)
                    except (ValueError, OSError):
                        pass
                normalized_args[key] = stripped
            else:
                normalized_args[key] = value
        return f"{tool_name}:{json.dumps(normalized_args, sort_keys=True)}"

    def record(self, tool_name: str, args: dict, result_chars: int):
        """
        Record a tool call. Called from _dispatch_tool_call at dispatch time.

        Args:
            tool_name: Name of the tool that was called.
            args: Arguments passed to the tool.
            result_chars: Character count of the serialized result.
        """
        signature = self._normalize_input(tool_name, args)
        is_duplicate = signature in self._seen_signatures
        self._seen_signatures.add(signature)

        entry = {
            "tool_name": tool_name,
            "input_summary": self._summarize_input(args),
            "result_chars": result_chars,
            "is_duplicate": is_duplicate,
            "call_index": len(self._calls),
        }
        self._calls.append(entry)

        # Update counters
        self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1
        self._tool_result_chars[tool_name] = (
            self._tool_result_chars.get(tool_name, 0) + result_chars
        )

        if is_duplicate:
            self._duplicates.append(entry)

    @staticmethod
    def _summarize_input(args: dict, max_len: int = 120) -> str:
        """Create a short human-readable summary of tool input args."""
        if not args:
            return "(no args)"
        parts = []
        for key, value in args.items():
            val_str = str(value)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            parts.append(f"{key}={val_str}")
        summary = ", ".join(parts)
        if len(summary) > max_len:
            summary = summary[: max_len - 3] + "..."
        return summary

    def get_analysis(self) -> Dict[str, Any]:
        """
        Return the accumulated analysis dict. O(1) — reads pre-computed data.

        Designed to be nested into the existing get_status() response.
        """
        total_calls = len(self._calls)
        duplicate_count = len(self._duplicates)
        total_result_chars = sum(self._tool_result_chars.values())

        # Build per-tool attribution
        attribution = {}
        for tool_name in self._tool_counts:
            chars = self._tool_result_chars.get(tool_name, 0)
            attribution[tool_name] = {
                "calls": self._tool_counts[tool_name],
                "result_chars": chars,
                "percent_of_total": (
                    round(chars / total_result_chars * 100, 1)
                    if total_result_chars > 0
                    else 0.0
                ),
            }

        # Sort attribution by result_chars descending
        sorted_attribution = dict(
            sorted(attribution.items(), key=lambda x: x[1]["result_chars"], reverse=True)
        )

        return {
            "total_tool_calls": total_calls,
            "unique_tool_calls": total_calls - duplicate_count,
            "duplicate_tool_calls": duplicate_count,
            "duplicate_percent": (
                round(duplicate_count / total_calls * 100, 1)
                if total_calls > 0
                else 0.0
            ),
            "total_result_chars": total_result_chars,
            "attribution_by_tool": sorted_attribution,
            "duplicates": [
                {
                    "tool_name": d["tool_name"],
                    "input_summary": d["input_summary"],
                    "call_index": d["call_index"],
                }
                for d in self._duplicates
            ],
        }


class OrchestratorEngineMixin:
    """Mixin providing orchestrator loop methods for KestrelAgent."""

    async def _execute_tool_with_hooks(
        self,
        tool_name: str,
        feature_name: str,
        args: dict,
        session_id: str,
        execute_fn,
    ) -> dict:
        """
        Execute a tool with PRE_TOOL_USE and POST_TOOL_USE hook enforcement.

        This is the single entry point for all tool execution in the orchestrator
        loop, ensuring security hooks (permissions, audit logging) are always
        invoked regardless of whether the tool is a feature subagent dispatch
        or a direct tool call.

        Args:
            tool_name: Name of the tool being called
            feature_name: Name of the owning feature (for permission lookup)
            args: Arguments to pass to the tool
            session_id: Session ID for hook context
            execute_fn: Async callable that performs the actual tool execution.
                        Called with no arguments; should return the tool result.

        Returns:
            Tool result dict, or an error dict if permission was denied.
        """
        # --- PRE_TOOL_USE hooks ---
        hook_input = HookInput(
            session_id=session_id,
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name=tool_name,
            tool_input=args,
            feature_name=feature_name,
        )

        hook_output = await self.hooks_manager.execute_hooks(
            HookEvent.PRE_TOOL_USE,
            hook_input,
        )

        if hook_output.permission_decision == PermissionDecision.DENY:
            reason = hook_output.permission_reason or "Blocked by security policy"
            logging.info(f"[HOOKS] Tool denied: {feature_name}.{tool_name} - {reason}")
            return {"success": False, "error": f"Permission denied: {reason}"}

        # If hooks modified the input, update args (callers that need it can
        # inspect the returned result; the execute_fn closure already captured
        # the original args, so we pass updated_input through the result).
        if hook_output.updated_input:
            args = hook_output.updated_input

        # --- Execute the tool ---
        exec_start = time.time()
        with optional_span("agent.tool_execution", {
            "tool.name": tool_name,
            "tool.feature": feature_name,
        }) as tool_span:
            result = await execute_fn()
            exec_duration_ms = int((time.time() - exec_start) * 1000)
            if tool_span:
                tool_span.set_attribute("tool.duration_ms", exec_duration_ms)

        # --- POST_TOOL_USE hooks (parallel, non-blocking) ---
        post_hook_input = HookInput(
            session_id=session_id,
            hook_event_name=HookEvent.POST_TOOL_USE.value,
            tool_name=tool_name,
            tool_input=args,
            feature_name=feature_name,
            tool_response=result if isinstance(result, dict) else {"result": str(result)},
            execution_time_ms=exec_duration_ms,
        )
        await self.hooks_manager.execute_hooks_parallel(
            HookEvent.POST_TOOL_USE,
            post_hook_input,
        )

        return result

    # ------------------------------------------------------------------
    # Shared helpers used by both streaming and non-streaming loops
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_calls_msg(tool_calls) -> list:
        """Convert LLMResponse tool_calls to OpenAI-format dicts."""
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    # ollama library expects arguments as dict, not string
                    "arguments": tc.arguments if isinstance(tc.arguments, dict)
                    else json.loads(tc.arguments) if tc.arguments else {}
                }
            }
            for tc in tool_calls
        ]

    async def _dispatch_tool_call(
        self,
        tool_call,
        features_by_tool_name: dict,
        known_tools: set,
        messages: list,
        iteration: int,
        user_message: Optional[str],
        *,
        tool_events: Optional[list] = None,
        streaming: bool = False,
    ):
        """
        Dispatch a single tool call — used by both streaming and non-streaming loops.

        Yields status strings when streaming=True.
        Returns the tool result dict.
        """
        _init_constants()
        tool_name = tool_call.name
        args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        log_prefix = "[ORCHESTRATOR-STREAM]" if streaming else "[ORCHESTRATOR]"

        # Validate tool arguments before execution
        is_valid, validation_error = validate_tool_arguments(
            tool_name, args, known_tools=known_tools
        )
        if not is_valid:
            logging.warning(f"{log_prefix} Tool validation failed: {validation_error}")
            result = {"success": False, "error": f"Tool validation failed: {validation_error}"}
            from kestrel_sovereign.features.base import _serialize_tool_result
            result_json = json.dumps(_serialize_tool_result(result))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_json
            })
            return

        # Stream tool start indicator
        if streaming and tool_events is not None:
            tool_events.append({'type': 'start', 'tool': tool_name})

        dispatch_start = time.time()
        dispatch_event_id = await self.observability_store.log_tool_call(
            agent_name=self.did,
            tool_name=f"feature_dispatch:{tool_name}",
            metadata={"arguments": args, "iteration": iteration}
        )

        feature = features_by_tool_name.get(tool_name)
        result = None

        if feature:
            result = await self._dispatch_feature_tool(
                tool_call, feature, args, dispatch_start, dispatch_event_id,
                user_message, tool_events=tool_events, streaming=streaming,
            )
        elif tool_name in self._direct_tools:
            result = await self._dispatch_direct_tool(
                tool_call, tool_name, args, dispatch_start, dispatch_event_id,
                tool_events=tool_events, streaming=streaming,
            )
        else:
            result = {"success": False, "error": f"Unknown feature tool: {tool_name}"}
            dispatch_duration = int((time.time() - dispatch_start) * 1000)
            await self.observability_store.log_tool_response(
                event_id=dispatch_event_id,
                success=False,
                duration_ms=dispatch_duration,
                error_message=f"Unknown feature tool: {tool_name}",
            )
            if not streaming:
                await self.observability_store.log_error(
                    agent_name=self.did,
                    error_type="unknown_feature_tool",
                    error_message=f"Unknown feature tool: {tool_name}",
                    metadata={"tool_name": tool_name, "available": list(features_by_tool_name.keys())}
                )
            if streaming and tool_events is not None:
                tool_events.append({'type': 'error', 'tool': tool_name, 'error': f'Unknown feature tool: {tool_name}'})

        # Add tool result to messages (with persistence for large results)
        from kestrel_sovereign.features.base import _serialize_tool_result
        result_json = json.dumps(_serialize_tool_result(result))

        if len(result_json) > MAX_TOOL_RESULT_CHARS:
            original_len = len(result_json)
            result_json = _build_persisted_preview(result_json, tool_name, original_len)
            logging.info(f"{log_prefix} Large tool result ({original_len} chars) replaced with preview")
        else:
            logging.info(f"{log_prefix} Tool result ({len(result_json)} chars): {result_json[:200]}...")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_json
        })

        # Record into context stats accumulator (if available on the agent)
        context_stats = getattr(self, "context_stats", None)
        if context_stats is not None:
            context_stats.record(tool_name, args, len(result_json))

        return result

    async def _dispatch_feature_tool(
        self, tool_call, feature, args, dispatch_start, dispatch_event_id,
        user_message, *, tool_events=None, streaming=False,
    ):
        """Dispatch to a feature subagent with hook enforcement."""
        tool_name = tool_call.name
        hook_feature_name = type(feature).__name__

        # --- PRE_SUBAGENT_CALL hooks ---
        subagent_hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
            tool_name=tool_name,
            tool_input=args,
            feature_name=hook_feature_name,
        )
        subagent_hook_output = await self.hooks_manager.execute_hooks(
            HookEvent.PRE_SUBAGENT_CALL, subagent_hook_input
        )
        if subagent_hook_output.permission_decision == PermissionDecision.DENY:
            reason = subagent_hook_output.permission_reason or "Subagent call blocked by policy"
            logging.warning(f"[HOOKS] Subagent denied: {hook_feature_name}.{tool_name} - {reason}")
            result = {"success": False, "error": f"Permission denied: {reason}"}

            dispatch_duration = int((time.time() - dispatch_start) * 1000)
            await self.observability_store.log_tool_response(
                event_id=dispatch_event_id,
                success=False,
                duration_ms=dispatch_duration,
                error_message=reason,
            )
            if streaming and tool_events is not None:
                tool_events.append({'type': 'error', 'tool': tool_name, 'error': reason[:200]})
            return result

        # Subagent hook allowed — collect denied tools before dispatch
        denied_tools = await self._get_denied_tools(hook_feature_name)
        if denied_tools:
            logging.info(f"[SECURITY] Stripping denied tools from {hook_feature_name}: {denied_tools}")

        async def _exec_feature(f=feature, a=args, dt=denied_tools):
            task = a.get("task", "")
            context = a.get("context")
            if not context and user_message:
                context = f"User's original request: {user_message}"
            log_tag = "[STREAM] " if streaming else ""
            logging.info(f"{log_tag}Dispatching to feature subagent: {f.tool_name}")
            with optional_span("agent.feature_dispatch", {"feature.name": f.tool_name}):
                r = await f.execute_as_subagent(task=task, context=context, denied_tools=dt)
            self._register_explored_feature_tools(f)
            return r

        try:
            result = await self._execute_tool_with_hooks(
                tool_name=tool_name,
                feature_name=hook_feature_name,
                args=args,
                session_id="orchestrator",
                execute_fn=_exec_feature,
            )

            dispatch_duration = int((time.time() - dispatch_start) * 1000)
            await self.observability_store.log_tool_response(
                event_id=dispatch_event_id,
                success=True,
                duration_ms=dispatch_duration,
            )

            # Fire POST_SUBAGENT_CALL hook (non-blocking, parallel)
            post_hook_input = HookInput(
                session_id="orchestrator",
                hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
                tool_name=tool_name,
                tool_input=args,
                feature_name=hook_feature_name,
                tool_response=result if isinstance(result, dict) else {"result": str(result)},
                execution_time_ms=dispatch_duration,
            )
            await self.hooks_manager.execute_hooks_parallel(
                HookEvent.POST_SUBAGENT_CALL, post_hook_input
            )

            if streaming and tool_events is not None:
                tool_events.append({'type': 'complete', 'tool': tool_name, 'ms': dispatch_duration})
            return result

        except (ConnectionError, TimeoutError, ValueError, KeyError, TypeError, AttributeError) as e:
            return await self._handle_feature_error(
                e, tool_name, hook_feature_name, args, dispatch_start,
                dispatch_event_id, tool_events=tool_events, streaming=streaming,
            )
        except Exception as e:
            return await self._handle_feature_error(
                e, tool_name, hook_feature_name, args, dispatch_start,
                dispatch_event_id, tool_events=tool_events, streaming=streaming,
                log_traceback=True,
            )

    async def _get_denied_tools(self, feature_name: str) -> set:
        """Get tools denied by security policy for a feature.

        Queries the permission store to find which of this feature's tools
        are set to DENY. These tools should be stripped from the subagent's
        tool palette before dispatch — the LLM cannot call what it cannot see.
        """
        try:
            security_feature = self.features.get("SecurityFeature")
            if not security_feature:
                return set()

            store = getattr(security_feature, "permission_store", None)
            if not store or not callable(getattr(store, "get_permission", None)):
                return set()

            from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
            if not isinstance(store, PermissionStore):
                return set()

            # Get the actual feature instance to enumerate its tools
            feature = None
            for f in self.features.values():
                if type(f).__name__ == feature_name:
                    feature = f
                    break

            if not feature or not callable(getattr(feature, "get_tools", None)):
                return set()

            denied = set()
            for tool_obj in feature.get_tools():
                level = await store.get_permission(feature_name, tool_obj.name)
                if level == PermissionLevel.DENY:
                    denied.add(tool_obj.name)

            return denied
        except Exception as e:
            logging.debug(f"[SECURITY] Could not check denied tools: {e}")
            return set()

    async def _handle_feature_error(
        self, error, tool_name, hook_feature_name, args, dispatch_start,
        dispatch_event_id, *, tool_events=None, streaming=False, log_traceback=False,
    ):
        """Handle feature execution error with logging and hooks."""
        if log_traceback:
            logging.error(f"Feature {tool_name} execution failed: {error}", exc_info=True)
        else:
            logging.error(f"Feature {tool_name} execution failed: {error}")
        result = {"success": False, "error": str(error)}

        dispatch_duration = int((time.time() - dispatch_start) * 1000)
        await self.observability_store.log_tool_response(
            event_id=dispatch_event_id,
            success=False,
            duration_ms=dispatch_duration,
            error_message=str(error),
        )

        # Fire POST_SUBAGENT_CALL hook on failure
        post_hook_input = HookInput(
            session_id="orchestrator",
            hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
            tool_name=tool_name,
            tool_input=args,
            feature_name=hook_feature_name,
            tool_response={"success": False, "error": str(error)},
            execution_time_ms=dispatch_duration,
        )
        await self.hooks_manager.execute_hooks_parallel(
            HookEvent.POST_SUBAGENT_CALL, post_hook_input
        )

        if streaming and tool_events is not None:
            tool_events.append({'type': 'error', 'tool': tool_name, 'error': str(error)[:200]})
        return result

    async def _dispatch_direct_tool(
        self, tool_call, tool_name, args, dispatch_start, dispatch_event_id,
        *, tool_events=None, streaming=False,
    ):
        """Dispatch a direct tool call (no subagent LLM hop)."""
        tool = self._direct_tools[tool_name]
        hook_feature_name = self._tool_to_feature.get(tool_name, tool_name)

        async def _exec_direct(t=tool, a=args):
            return await t.execute(**a)

        try:
            result = await self._execute_tool_with_hooks(
                tool_name=tool_name,
                feature_name=hook_feature_name,
                args=args,
                session_id="orchestrator",
                execute_fn=_exec_direct,
            )

            dispatch_duration = int((time.time() - dispatch_start) * 1000)
            await self.observability_store.log_tool_response(
                event_id=dispatch_event_id,
                success=True,
                duration_ms=dispatch_duration,
            )
            if not streaming:
                logging.info(f"[DIRECT-TOOL] {tool_name} ({dispatch_duration}ms)")
            if streaming and tool_events is not None:
                tool_events.append({'type': 'complete', 'tool': tool_name, 'ms': dispatch_duration})
            return result
        except Exception as e:
            logging.error(f"[DIRECT-TOOL] {tool_name} failed: {e}")
            result = {"success": False, "error": str(e)}

            dispatch_duration = int((time.time() - dispatch_start) * 1000)
            await self.observability_store.log_tool_response(
                event_id=dispatch_event_id,
                success=False,
                duration_ms=dispatch_duration,
                error_message=str(e),
            )
            if streaming and tool_events is not None:
                tool_events.append({'type': 'error', 'tool': tool_name, 'error': str(e)[:200]})
            return result

    # ------------------------------------------------------------------
    # Tool concurrency batching
    # ------------------------------------------------------------------

    def _is_direct_tool_concurrency_safe(self, tool_name: str) -> bool:
        """Check if a direct tool (promoted via LRU) is concurrency-safe."""
        tool = self._direct_tools.get(tool_name)
        if tool and hasattr(tool, 'schema') and hasattr(tool.schema, 'is_concurrency_safe'):
            return tool.schema.is_concurrency_safe
        return False

    def _partition_tool_calls(self, tool_calls, features_by_tool_name: dict) -> list:
        """
        Partition tool calls into batches for concurrent/serial execution.

        Consecutive concurrency-safe DIRECT tools are grouped into parallel
        batches. Feature subagent dispatches and unsafe tools run serially.

        Returns list of (is_parallel, [tool_calls]) tuples.
        """
        batches = []
        current_parallel = []

        for tc in tool_calls:
            name = tc.name
            # Feature dispatches are NEVER parallel (they use shared LLM)
            is_feature = name in features_by_tool_name
            is_direct_safe = (not is_feature and
                              name in self._direct_tools and
                              self._is_direct_tool_concurrency_safe(name))

            if is_direct_safe:
                current_parallel.append(tc)
            else:
                # Flush any pending parallel batch
                if current_parallel:
                    batches.append((True, current_parallel))
                    current_parallel = []
                batches.append((False, [tc]))

        if current_parallel:
            batches.append((True, current_parallel))

        return batches

    async def _execute_tool_batch(
        self,
        tool_calls,
        features_by_tool_name: dict,
        known_tools: set,
        messages: list,
        iteration: int,
        user_message: Optional[str],
        *,
        tool_events: Optional[list] = None,
        streaming: bool = False,
    ):
        """
        Execute a batch of tool calls, using parallelism where safe.

        Partitions into parallel/serial batches. Parallel batches use
        asyncio.gather with a semaphore. All tool results are appended
        to messages in the ORIGINAL REQUEST ORDER.

        Key design: parallel tools still go through _dispatch_tool_call(),
        which handles hooks, observability, context_stats, and result
        formatting. Nothing is bypassed.
        """
        batches = self._partition_tool_calls(tool_calls, features_by_tool_name)

        if len(batches) == 1 and not batches[0][0]:
            # Single serial batch (most common case) — no overhead
            for tc in batches[0][1]:
                await self._dispatch_tool_call(
                    tc, features_by_tool_name, known_tools, messages,
                    iteration, user_message,
                    tool_events=tool_events, streaming=streaming,
                )
            return

        semaphore = asyncio.Semaphore(MAX_TOOL_CONCURRENCY)

        for is_parallel, batch_tcs in batches:
            if not is_parallel or len(batch_tcs) == 1:
                # Serial execution
                for tc in batch_tcs:
                    await self._dispatch_tool_call(
                        tc, features_by_tool_name, known_tools, messages,
                        iteration, user_message,
                        tool_events=tool_events, streaming=streaming,
                    )
            else:
                # Parallel execution of concurrency-safe direct tools
                logging.info(
                    f"[CONCURRENCY] Executing {len(batch_tcs)} tools in parallel "
                    f"(max {MAX_TOOL_CONCURRENCY})"
                )

                # Each tool dispatches into its own temporary message list
                # to avoid ordering issues on the shared messages list
                per_tool_messages = {tc.id: [] for tc in batch_tcs}

                async def _run_one(tc, msg_list):
                    async with semaphore:
                        await self._dispatch_tool_call(
                            tc, features_by_tool_name, known_tools, msg_list,
                            iteration, user_message,
                            tool_events=tool_events, streaming=streaming,
                        )

                await asyncio.gather(
                    *[_run_one(tc, per_tool_messages[tc.id]) for tc in batch_tcs]
                )

                # Append results in original request order
                for tc in batch_tcs:
                    messages.extend(per_tool_messages[tc.id])

    # ------------------------------------------------------------------
    # Non-streaming orchestrator response handler
    # ------------------------------------------------------------------

    async def _handle_orchestrator_response(
        self,
        response: Union[str, LLMResponse],
        feature_tools: List[Dict[str, Any]],
        system_prompt: str,
        force_local_only: bool,
        effective_model: str,
        max_iterations: int = None,
        user_message: str = None
    ) -> str:
        """
        Handle the orchestrator's response, executing any tool calls.

        If the LLM returns tool_calls, we dispatch them to the appropriate
        features (as subagents), then continue the conversation with results.
        """
        _init_constants()
        if max_iterations is None:
            max_iterations = MAX_TOOL_ITERATIONS

        if isinstance(response, str):
            return response

        if not response.has_tool_calls:
            return response.content or ""

        # Build message history for multi-turn tool calling
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if user_message:
            messages.append({"role": "user", "content": user_message})
            logging.debug(f"[ORCHESTRATOR] Added user message to context: {user_message[:100]}...")
        else:
            logging.warning("[ORCHESTRATOR] No user_message provided - LLM won't have context for tool results!")

        # Add initial assistant response with tool calls
        assistant_msg = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            assistant_msg["tool_calls"] = self._build_tool_calls_msg(response.tool_calls)
        messages.append(assistant_msg)

        features_by_tool_name = {f.tool_name: f for f in self.features.values()}
        known_tools = set(features_by_tool_name.keys()) | set(self._direct_tools.keys())
        # Also accept direct calls to each feature's @tool-decorated sub-tools
        # (e.g. ``get_current_model`` under the ModelAgent feature). Without
        # this, the top-level LLM sees a sub-tool in the exposed tool list,
        # emits a call by that name, the guardrail silently rejects it, and
        # the agent hallucinates a "result" from training data.
        for _feat in self.features.values():
            if hasattr(_feat, "get_tools"):
                try:
                    for _t in _feat.get_tools():
                        known_tools.add(_t.name)
                except Exception:
                    pass

        tracker = IterationTracker(
            threshold=KESTREL_DIMINISHING_THRESHOLD,
            max_low_delta=KESTREL_MAX_LOW_DELTA,
            budget_stop_pct=KESTREL_BUDGET_STOP_PCT,
        )

        for iteration in range(max_iterations):
            if iteration >= max_iterations * 0.8:
                logging.warning(f"[ORCHESTRATOR] Approaching max iterations: {iteration + 1}/{max_iterations}")

            await self._execute_tool_batch(
                response.tool_calls, features_by_tool_name, known_tools,
                messages, iteration, user_message,
            )

            # Continue conversation with tool results
            all_tools = self._build_all_tools()
            messages = self._prune_orchestrator_messages(messages, all_tools)

            logging.info(f"[ORCHESTRATOR] Calling LLM with {len(messages)} messages, {len(all_tools)} tools")
            response = await self.llm_service.generate_with_messages(
                messages=messages,
                tools=all_tools or None,
                force_local_only=force_local_only,
                model_override=effective_model
            )

            if isinstance(response, str):
                logging.info(f"[ORCHESTRATOR] Final response (string): {response[:300]}...")
                return response

            if not response.has_tool_calls:
                final_content = response.content or ""
                logging.info(f"[ORCHESTRATOR] Final response (no more tool calls): {final_content[:300]}...")
                return final_content

            # Track diminishing returns on this response before continuing
            tracker.record(response)
            if tracker.should_stop(iteration, max_iterations):
                logging.warning(
                    f"[ORCHESTRATOR] Diminishing returns detected at iteration {iteration + 1}/{max_iterations} "
                    f"(consecutive_low_delta={tracker.consecutive_low_delta}). Stopping."
                )
                return response.content or "Stopped: diminishing returns detected"

            assistant_msg = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_msg["tool_calls"] = self._build_tool_calls_msg(response.tool_calls)
            messages.append(assistant_msg)

        logging.warning("Max tool call iterations reached")
        return response.content or "Error: Maximum tool call iterations exceeded"

    # ------------------------------------------------------------------
    # Message pruning
    # ------------------------------------------------------------------

    def _prune_orchestrator_messages(
        self, messages: list, tools: list, context_limit: int = None
    ) -> list:
        """Prune orchestrator messages to stay within context limits.

        Strategy: keep system + user messages (first 2), keep the most recent
        assistant+tool exchange, and progressively drop older tool results
        (replacing with summaries) until we fit.

        Uses char-based estimation: ~4 chars per token.
        """
        _init_constants()
        if context_limit is None:
            context_limit = getattr(self.llm_service, '_context_limit', None) or 131072

        chars_per_token = 3.5
        max_chars = int(context_limit * chars_per_token * (1 - CONTEXT_RESERVE_FRACTION))

        tool_chars = sum(len(json.dumps(t)) for t in tools) if tools else 0
        max_message_chars = max_chars - tool_chars

        def _total_chars(msgs):
            total = 0
            for m in msgs:
                total += len(m.get("content", "") or "")
                for tc in m.get("tool_calls", []):
                    total += len(json.dumps(tc.get("function", {}).get("arguments", {})))
            return total

        current = _total_chars(messages)

        if current <= max_message_chars:
            return messages

        logging.warning(
            f"[ORCHESTRATOR] Context pressure: ~{int(current / chars_per_token)}tok "
            f"messages + ~{int(tool_chars / chars_per_token)}tok tools "
            f"vs {context_limit}tok limit. Pruning old tool results."
        )

        protected = messages[:2]  # system + user
        middle = messages[2:]

        for i, msg in enumerate(middle):
            if _total_chars(protected + middle) <= max_message_chars:
                break
            if msg.get("role") == "tool" and len(msg.get("content", "")) > 200:
                original_len = len(msg["content"])
                middle[i] = {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": f"[Result truncated: was {original_len} chars. Tool completed successfully.]"
                }

        result = protected + middle
        new_total = _total_chars(result)
        logging.info(
            f"[ORCHESTRATOR] After pruning: ~{int(new_total / chars_per_token)}tok "
            f"(removed ~{int((current - new_total) / chars_per_token)}tok)"
        )
        return result

    # ------------------------------------------------------------------
    # Streaming orchestrator response handler
    # ------------------------------------------------------------------

    async def _handle_orchestrator_response_streaming(
        self,
        response,
        feature_tools: list,
        system_prompt: str,
        force_local_only: bool,
        effective_model: str,
        max_iterations: int = None,
        user_message: str = None,
        tool_events: list = None,
    ):
        """
        Streaming version of _handle_orchestrator_response.

        Executes tool calls synchronously, then streams the final LLM response.
        Tool execution cannot be streamed (we need complete results), but the
        final text response streams token-by-token.

        Yields:
            Text chunks as they arrive from the LLM
        """
        _init_constants()
        if max_iterations is None:
            max_iterations = MAX_TOOL_ITERATIONS

        if isinstance(response, str):
            yield response
            return

        if not response.has_tool_calls:
            yield response.content or ""
            return

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if user_message:
            messages.append({"role": "user", "content": user_message})
            logging.debug(f"[ORCHESTRATOR-STREAM] Added user message to context: {user_message[:100]}...")
        else:
            logging.warning("[ORCHESTRATOR-STREAM] No user_message provided!")

        assistant_msg = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            assistant_msg["tool_calls"] = self._build_tool_calls_msg(response.tool_calls)
        messages.append(assistant_msg)

        features_by_tool_name = {f.tool_name: f for f in self.features.values()}
        known_tools = set(features_by_tool_name.keys()) | set(self._direct_tools.keys())
        # Also accept direct calls to each feature's @tool-decorated sub-tools
        # (e.g. ``get_current_model`` under the ModelAgent feature). Without
        # this, the top-level LLM sees a sub-tool in the exposed tool list,
        # emits a call by that name, the guardrail silently rejects it, and
        # the agent hallucinates a "result" from training data.
        for _feat in self.features.values():
            if hasattr(_feat, "get_tools"):
                try:
                    for _t in _feat.get_tools():
                        known_tools.add(_t.name)
                except Exception:
                    pass

        tracker = IterationTracker(
            threshold=KESTREL_DIMINISHING_THRESHOLD,
            max_low_delta=KESTREL_MAX_LOW_DELTA,
            budget_stop_pct=KESTREL_BUDGET_STOP_PCT,
        )

        for iteration in range(max_iterations):
            # Stream tool names for user visibility
            for tc in response.tool_calls:
                yield f"\U0001f527 Calling {tc.name}...\n"

            await self._execute_tool_batch(
                response.tool_calls, features_by_tool_name, known_tools,
                messages, iteration, user_message,
                tool_events=tool_events, streaming=True,
            )

            all_tools = self._build_all_tools()
            messages = self._prune_orchestrator_messages(messages, all_tools)

            logging.info(f"[ORCHESTRATOR-STREAM] Checking for more tool calls with {len(messages)} messages, {len(all_tools)} tools")
            response = await self.llm_service.generate_with_messages(
                messages=messages,
                tools=all_tools or None,
                force_local_only=force_local_only,
                model_override=effective_model
            )

            if isinstance(response, str):
                yield response
                return

            if not response.has_tool_calls:
                logging.info("[ORCHESTRATOR-STREAM] Streaming final response")
                yield "\n---\n"
                async for chunk in self.llm_service.stream_with_messages(
                    messages=messages,
                    force_local_only=force_local_only,
                    model_override=effective_model
                ):
                    yield chunk
                return

            # Track diminishing returns on this response before continuing
            tracker.record(response)
            if tracker.should_stop(iteration, max_iterations):
                logging.warning(
                    f"[ORCHESTRATOR-STREAM] Diminishing returns detected at iteration {iteration + 1}/{max_iterations} "
                    f"(consecutive_low_delta={tracker.consecutive_low_delta}). Stopping."
                )
                yield response.content or "Stopped: diminishing returns detected"
                return

            assistant_msg = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_msg["tool_calls"] = self._build_tool_calls_msg(response.tool_calls)
            messages.append(assistant_msg)

        logging.warning("Max tool call iterations reached")
        yield "Error: Maximum tool call iterations exceeded"
