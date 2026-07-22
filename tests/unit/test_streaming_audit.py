"""
Unit tests for streaming response functionality.

Tests streaming behavior after per-response audit removal.
Only the local constitution hash check (_maybe_audit) remains.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestStreamingBasics:
    """Tests for basic streaming behavior."""

    @pytest.mark.asyncio
    async def test_command_input_uses_regular_processing(self):
        """Test that commands fall back to non-streaming processing."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        mock_agent = MagicMock()
        mock_agent._maybe_audit = AsyncMock()
        mock_agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
        mock_agent.process_input = AsyncMock(return_value="Command executed successfully")
        mock_agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(mock_agent)

        chunks = []
        async for chunk in mock_agent.process_input_streaming("!help"):
            chunks.append(chunk)

        mock_agent._maybe_audit.assert_called_once()
        mock_agent.process_input.assert_called_once_with(
            "!help", None, session_id=None, caller=None, invocation_context=None,
        )
        assert "Command executed" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_preinitialization_streaming_defers_before_genesis_gate(self):
        """Streaming matches the retryable pre-init behavior of process_input."""
        from kestrel_sovereign.agent.streaming import StreamingMixin

        mock_agent = MagicMock()
        mock_agent.storage = None
        mock_agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
        mock_agent.process_input_streaming = (
            StreamingMixin.process_input_streaming.__get__(mock_agent)
        )

        with pytest.raises(RuntimeError, match="agent not fully initialized"):
            async for _chunk in mock_agent.process_input_streaming("hello"):
                pass

        mock_agent._genesis_audit_cognition_block.assert_not_called()


class TestRealStreaming:
    """Tests for real LLM streaming (not fake chunking)."""

    @pytest.mark.asyncio
    async def test_stream_with_messages_yields_chunks(self):
        """Test that stream_with_messages yields chunks from the LLM."""
        from kestrel_sovereign.llm.service import LLMService

        mock_service = MagicMock(spec=LLMService)

        async def mock_stream(**kwargs):
            for word in ["Hello", " ", "World", "!"]:
                yield word

        mock_service.stream_with_messages = mock_stream

        chunks = []
        async for chunk in mock_service.stream_with_messages(
            messages=[{"role": "user", "content": "test"}],
            force_local_only=False,
            model_override=None
        ):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert "".join(chunks) == "Hello World!"

    @pytest.mark.asyncio
    async def test_orchestrator_streaming_yields_real_chunks(self):
        """Test that _handle_orchestrator_response_streaming yields real chunks."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm.adapter import LLMResponse

        mock_agent = MagicMock()
        mock_agent.features = {}
        mock_agent.did = "test-did"

        mock_agent.observability_store = MagicMock()
        mock_agent.observability_store.log_tool_call = AsyncMock(return_value="event-1")
        mock_agent.observability_store.log_tool_response = AsyncMock()

        response = LLMResponse(content="Simple response", tool_calls=[])

        mock_agent._handle_orchestrator_response_streaming = (
            KestrelAgent._handle_orchestrator_response_streaming.__get__(mock_agent)
        )

        chunks = []
        async for chunk in mock_agent._handle_orchestrator_response_streaming(
            response=response,
            feature_tools=[],
            system_prompt="test",
            force_local_only=False,
            effective_model="test-model",
            user_message="test message"
        ):
            chunks.append(chunk)

        assert "".join(chunks) == "Simple response"

    @pytest.mark.asyncio
    async def test_orchestrator_streaming_with_tool_calls(self):
        """Test that tool calls are executed then response is streamed."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
        from kestrel_sovereign.hooks import HooksManager

        mock_agent = MagicMock()
        mock_agent.did = "test-did"

        mock_feature = MagicMock()
        mock_feature.tool_name = "test_tool"
        mock_feature.name = "test_feature"
        mock_feature.execute_as_subagent = AsyncMock(return_value={"success": True, "data": "result"})
        mock_feature.to_orchestrator_tool.return_value = {
            "type": "function",
            "function": {"name": "test_tool", "description": "test", "parameters": {}}
        }
        mock_agent.features = {"test_feature": mock_feature}

        mock_agent.hooks_manager = HooksManager()

        mock_agent.observability_store = MagicMock()
        mock_agent.observability_store.log_tool_call = AsyncMock(return_value="event-1")
        mock_agent.observability_store.log_tool_response = AsyncMock()

        mock_agent._direct_tools = {}
        mock_agent._tool_to_feature = {}

        mock_agent.llm_service = MagicMock()

        first_response = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={"task": "do something"})]
        )
        second_response = LLMResponse(content="Final streamed response", tool_calls=[])

        async def mock_stream(**kwargs):
            # stream_with_tool_detection: yield text chunks as they
            # arrive, then a final LLMResponse carrying detected
            # tool_calls (empty here = no further iteration).
            for word in ["Final", " ", "streamed", " ", "response"]:
                yield word
            yield second_response

        mock_agent.llm_service.stream_with_tool_detection = mock_stream

        # Bind all orchestrator engine and tool registry mixin methods
        for method_name in (
            '_handle_orchestrator_response_streaming',
            '_execute_tool_with_hooks',
            '_execute_tool_batch',
            '_partition_tool_calls',
            '_dispatch_tool_call',
            '_dispatch_feature_tool',
            '_dispatch_direct_tool',
            '_get_denied_tools',
            '_handle_feature_error',
            '_prune_orchestrator_messages',
            '_build_all_tools',
            '_build_feature_tools',
            '_visible_features_by_tool_name',
            '_visible_known_tool_names',
            '_hidden_context_features',
            '_hidden_context_tools',
            '_feature_hidden_from_context',
            '_direct_tool_hidden_from_context',
        ):
            setattr(mock_agent, method_name,
                    getattr(KestrelAgent, method_name).__get__(mock_agent))
        mock_agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg
        mock_agent._explored_features = {}
        mock_agent._direct_tool_defs = []
        mock_agent._register_explored_feature_tools = MagicMock()

        chunks = []
        async for chunk in mock_agent._handle_orchestrator_response_streaming(
            response=first_response,
            feature_tools=[],
            system_prompt="test",
            force_local_only=False,
            effective_model="test-model",
            user_message="test message"
        ):
            chunks.append(chunk)

        mock_feature.execute_as_subagent.assert_called_once()

        full_output = "".join(chunks)
        assert "Final streamed response" in full_output
        assert "test_tool" in full_output


# =========================================================================
# #2674 — strict (fail-closed) response audit must withhold streaming output
# until the POST_RESPONSE verdict exists.
# =========================================================================

from contextlib import asynccontextmanager


@asynccontextmanager
async def _passthrough():
    """Stand-in for the turn-lifecycle / privacy-transition locks."""
    yield


def _make_streaming_audit_agent(
    add_convo_calls,
    *,
    mode: str,
    audit_response: dict = None,
    audit_side_effect=None,
    hook_timeout: float = None,
    register_hook: bool = True,
):
    """Build a mock agent wired with a REAL HooksManager + ResponseAuditHook.

    The streaming methods under test are bound from ``StreamingMixin`` so the
    live buffering path runs unchanged; only the environment (LLM stream, audit
    verdict, persistence sink) is mocked. Returns ``(agent, hook)``.

    ``register_hook=False`` leaves the manager EMPTY (no POST_RESPONSE hook at
    turn start) and returns ``(agent, None)`` — used to drive the real
    ``ResponseAuditFeature.enable_audit`` lifecycle mid-turn (#2674 self-review).
    """
    from kestrel_sovereign.agent.streaming import StreamingMixin
    from kestrel_sovereign.hooks import HooksManager
    from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: add_convo_calls.append({
            "role": role, "content": content, **kw,
        })
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "test-did"
    agent.extension = None
    agent._cached_features_prompt = ""
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent.emit_event = AsyncMock()
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent.operator_signal_producer = None
    agent._post_response_pipeline = None
    agent._privacy_blocks_background_memory = MagicMock(return_value=True)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered prompt"

    context_result = MagicMock()
    context_result.system_prompt = "system"
    context_result.dynamic_user_context = "ctx"
    context_result.messages = []
    context_result.degraded_mode = False
    context_result.warnings = []
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=context_result)

    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()

    agent.llm_service = MagicMock()
    if audit_side_effect is not None:
        agent.llm_service.get_audit_response = AsyncMock(side_effect=audit_side_effect)
    else:
        agent.llm_service.get_audit_response = AsyncMock(
            return_value=audit_response or {"risk_level": 1, "reasoning": "clean"}
        )

    manager = HooksManager()
    hook = None
    if register_hook:
        hook = ResponseAuditHook(agent=agent, mode=mode, risk_threshold=3)
        if hook_timeout is not None:
            hook.timeout = hook_timeout
        manager.register(hook)
    agent.hooks_manager = manager

    # Bind the real streaming methods (the code under test).
    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    agent._emit_revising_event = StreamingMixin._emit_revising_event.__get__(agent)
    agent._fire_post_response_hook = (
        StreamingMixin._fire_post_response_hook.__get__(agent)
    )
    return agent, hook


# A response long enough to pass the hook's 20-char audit gate.
_LONG_TEXT_CHUNKS = ["Hello ", "world, ", "this is ", "a long ", "enough ", "answer."]
_LONG_TEXT = "".join(_LONG_TEXT_CHUNKS)


class TestStrictStreamingAuditBuffering:
    """#2674: an enforcing POST_RESPONSE hook must gate every visible byte."""

    async def _drive_no_tool(self, agent, exhausted_flag, session_id="s-audit"):
        """Drive a no-tool streaming turn, capturing each chunk together with
        whether the upstream LLM stream had already been fully drained when the
        chunk was emitted. Buffered output can only appear post-drain."""
        captured = []
        async for chunk in agent.process_input_streaming("hi there", session_id=session_id):
            captured.append((chunk, exhausted_flag["done"]))
        return captured

    @pytest.mark.asyncio
    async def test_strict_deny_exposes_no_original_text(self):
        """A strict DENY must expose none of the original assistant text; only
        the fail-closed block message reaches the client, and only AFTER the
        upstream stream was fully drained (no pre-verdict byte escaped)."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "harmful content"},
        )

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = await self._drive_no_tool(agent, exhausted)
        joined = "".join(c for c, _ in captured)

        # No original word leaked to the live client.
        for word in ("Hello", "world", "answer"):
            assert word not in joined, f"leaked {word!r} before the verdict"
        # Only the fail-closed block message was released.
        assert "[Response blocked by audit:" in joined
        # Every released chunk came after the upstream stream fully drained.
        assert captured, "generator produced nothing"
        assert all(done for _, done in captured), (
            "a chunk escaped before the audit verdict"
        )
        # Persistence stores the block message, never the original text.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        assert "Hello" not in persisted[0]["content"]

    @pytest.mark.asyncio
    async def test_strict_allow_emits_reviewed_text_once(self):
        """A strict ALLOW releases the reviewed text exactly once, after the
        verdict — byte-identical to what was streamed, just later."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = await self._drive_no_tool(agent, exhausted)
        joined = "".join(c for c, _ in captured)

        assert joined == _LONG_TEXT
        assert joined.count(_LONG_TEXT) == 1
        # Reviewed text released only after the stream drained (buffered).
        assert all(done for _, done in captured)
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert persisted[0]["content"] == _LONG_TEXT

    @pytest.mark.asyncio
    async def test_strict_provider_failure_emits_only_block_message(self):
        """Audit provider failure (get_audit_response raises) in strict mode
        fails closed: only the block message is emitted, no original bytes."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_side_effect=RuntimeError("provider down"),
        )

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = await self._drive_no_tool(agent, exhausted)
        joined = "".join(c for c, _ in captured)

        assert "Hello" not in joined and "answer" not in joined
        assert "[Response blocked by audit:" in joined
        assert all(done for _, done in captured)

    @pytest.mark.asyncio
    async def test_strict_audit_timeout_emits_only_block_message(self):
        """A hung audit provider that trips the hook's manager-level timeout
        resolves to DENY (fail-closed) — only the block message escapes, and
        no original byte reaches the client before the verdict."""
        import asyncio

        add_convo_calls = []

        async def _hang(_text):
            await asyncio.sleep(5.0)
            return {"risk_level": 1, "reasoning": "late"}

        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_side_effect=_hang,
            hook_timeout=0.05,  # manager cancels the hung audit
        )

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = await self._drive_no_tool(agent, exhausted)
        joined = "".join(c for c, _ in captured)

        assert "Hello" not in joined and "answer" not in joined
        # Manager fails the enforcing hook closed → block message only.
        assert "blocked by" in joined.lower()
        assert all(done for _, done in captured)

    @pytest.mark.asyncio
    async def test_warn_mode_streams_incrementally(self):
        """Advisory/warn mode is NOT enforcing, so it must NOT inherit the
        strict-mode latency — text streams incrementally as before."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="warn",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = await self._drive_no_tool(agent, exhausted)
        joined = "".join(c for c, _ in captured)

        # At least one visible chunk emitted BEFORE the stream fully drained.
        visible = [(c, done) for c, done in captured if c.strip()]
        assert visible, "no visible chunks"
        assert visible[0][1] is False, "warn mode should stream incrementally"
        assert joined == _LONG_TEXT

    @pytest.mark.asyncio
    async def test_strict_deny_with_tools_blocks_and_drops_parts(self):
        """The tool path: a strict DENY must leak neither the post-tool prose
        nor any typed part / revise sentinel to the live client, and must not
        persist the scrubbed part."""
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
        from kestrel_sovereign.agent.parts import build_part_sentinel

        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )

        async def mock_stream(**kwargs):
            yield "Updating the list. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        agent.llm_service.stream_with_tool_detection = mock_stream

        part_sentinel = build_part_sentinel(
            {"type": "todo", "data": {"secret": "leak-me"}, "id": "t1"}
        )

        async def mock_orchestrator(**kwargs):
            yield "Done — "
            yield part_sentinel
            yield "added a secret item."

        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("update todos", session_id="s-tool"):
            yielded.append(chunk)
        joined = "".join(yielded)

        # No post-tool prose, part sentinel, or revise sentinel leaked.
        assert "Done" not in joined and "secret item" not in joined
        assert "\x1eKESTREL:PART:" not in joined
        assert "leak-me" not in joined
        assert "\x1eKESTREL:REVISE:" not in joined
        assert "[Response blocked by audit:" in joined

        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        metadata = persisted[0].get("metadata") or {}
        assert "parts" not in metadata

    @pytest.mark.asyncio
    async def test_strict_allow_with_tools_releases_only_reviewed_text(self):
        """The tool path: a strict ALLOW releases ONLY the reviewed post-tool
        text — the exact bytes the audit examined. The withheld pre-tool prose,
        revise sentinel, thinking and typed-part sentinels are NOT replayed live
        (the audit never reviewed them, and pre-tool prose is retracted from the
        persisted turn). The part still persists to metadata and renders on
        history reload."""
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
        from kestrel_sovereign.agent.parts import build_part_sentinel

        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )

        async def mock_stream(**kwargs):
            yield "Updating the list. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        agent.llm_service.stream_with_tool_detection = mock_stream

        part_sentinel = build_part_sentinel(
            {"type": "todo", "data": {"title": "ship #2674"}, "id": "t1"}
        )

        async def mock_orchestrator(**kwargs):
            yield "Done — "
            yield part_sentinel
            yield "added one item."

        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("update todos", session_id="s-tool2"):
            yielded.append(chunk)
        joined = "".join(yielded)

        # ONLY the reviewed post-tool synthesis reached the client.
        assert joined == "Done — added one item."
        # Unreviewed / retracted material never surfaced live.
        assert "Updating the list. " not in joined
        assert "\x1eKESTREL:REVISE:" not in joined
        assert "\x1eKESTREL:PART:" not in joined

        # The part still persists (renders on reload); content is post-tool text.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert persisted[0]["content"] == "Done — added one item."
        metadata = persisted[0].get("metadata") or {}
        assert "parts" in metadata
        assert metadata["parts"][0]["type"] == "todo"

    @pytest.mark.asyncio
    async def test_strict_deny_with_tools_drops_pre_tool_reasoning_metadata(self):
        """#2674 finding 2: a strict DENY on a tool turn must NOT persist the
        rejected pre-tool prose in ``pre_tool_reasoning`` / ``context_replay``.
        ContextBuilder reinjects that metadata into the next LLM turn's context
        (context_builder.py:527), so retaining it would smuggle denied content
        forward. Neither history reload (persisted state) nor the next-turn
        context — both derived solely from this metadata — may contain it."""
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

        secret_pre_tool = "SECRET-PRETOOL-REASONING-should-not-persist"

        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )

        async def mock_stream(**kwargs):
            yield secret_pre_tool + ". "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        agent.llm_service.stream_with_tool_detection = mock_stream

        async def mock_orchestrator(**kwargs):
            yield "Post-tool synthesis that is long enough to audit."

        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("go", session_id="s-deny-meta"):
            yielded.append(chunk)
        joined = "".join(yielded)

        # Client only saw the block message; the pre-tool prose never leaked.
        assert secret_pre_tool not in joined
        assert "[Response blocked by audit:" in joined

        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        metadata = persisted[0].get("metadata") or {}
        # No content-bearing original-output metadata survives a DENY.
        assert "pre_tool_reasoning" not in metadata
        # Belt-and-suspenders: the rejected prose appears NOWHERE in the
        # persisted turn (content or any metadata value) — so neither a history
        # reload nor ContextBuilder's metadata-driven reinjection can surface it.
        import json as _json
        assert secret_pre_tool not in _json.dumps(persisted[0], default=str)

    @pytest.mark.asyncio
    async def test_strict_cancel_before_dispatch_persists_no_unaudited_text(self):
        """#2674 finding 1: when a strict turn is cancelled after the LLM reports
        tool calls but BEFORE dispatch, the withheld partial pre-tool prose was
        never audited. It must not be persisted as assistant content (it would
        resurface on history reload); the turn is recorded empty + cancelled."""
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

        partial_prose = "PARTIAL-UNAUDITED-PRETOOL-PROSE"

        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            yield partial_prose + ". "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )
            # Stop arrives only after the LLM stream fully drains — so the
            # in-loop cancel check never trips, and we reach the post-loop
            # cancel-before-dispatch branch with tool calls detected.
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream
        agent.is_request_cancelled = MagicMock(side_effect=lambda rid: exhausted["done"])

        # A dispatch would have to go through this; assert it never runs.
        agent._handle_orchestrator_response_streaming = MagicMock(
            side_effect=AssertionError("tools dispatched after cancellation")
        )

        yielded = []
        async for chunk in agent.process_input_streaming(
            "go", session_id="s-cancel", request_id="req-cancel",
        ):
            yielded.append(chunk)
        joined = "".join(yielded)

        # Nothing reached the client, and the unaudited prose was not persisted.
        assert partial_prose not in joined
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert persisted[0]["content"] == ""
        assert partial_prose not in (persisted[0]["content"] or "")
        assert (persisted[0].get("metadata") or {}).get("cancelled") is True

    @pytest.mark.asyncio
    async def test_fire_post_response_hook_returns_explicit_verdict(self):
        """#2674 finding 3: the release/metadata rules must key off an EXPLICIT
        verdict, not ``text == original``. ``_fire_post_response_hook`` returns a
        str carrying ``denied`` / ``modified`` so a DENY whose block message
        equals the original prose is still unambiguously a DENY."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        denied = await agent._fire_post_response_hook(
            "A long enough answer to reach the audit gate.", "s1",
        )
        assert getattr(denied, "denied", False) is True
        assert getattr(denied, "modified", False) is True
        assert str(denied).startswith("[Response blocked by audit:")

        # Flip the same hook to ALLOW and confirm the verdict flips too.
        _hook.mode = "warn"  # warn never denies clean text
        allowed = await agent._fire_post_response_hook(
            "A long enough answer to reach the audit gate.", "s1",
        )
        assert getattr(allowed, "denied", False) is False

    @pytest.mark.asyncio
    async def test_strict_deny_drops_parts_even_on_equality_collision(self):
        """#2674 finding 3 (reproduced): a strict DENY whose block message
        happens to EQUAL the original post-tool prose must still drop the typed
        part and release only the block message. The old ``final == original``
        inference misread this as ALLOW and replayed the raw buffer (part +
        secret). The explicit verdict makes it a DENY regardless."""
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
        from kestrel_sovereign.agent.parts import build_part_sentinel

        # The block message is deterministic: strict deny reason is
        # ``Audit risk level {risk}: {reasoning}`` (hook._apply_audit_decision),
        # wrapped as ``[Response blocked by audit: {reason}]``.
        collision = "[Response blocked by audit: Audit risk level 3: collide]"

        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "collide"},
        )

        async def mock_stream(**kwargs):
            yield "pre. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        agent.llm_service.stream_with_tool_detection = mock_stream

        part_sentinel = build_part_sentinel(
            {"type": "todo", "data": {"secret": "leak-me"}, "id": "t1"}
        )

        async def mock_orchestrator(**kwargs):
            # Post-tool prose engineered to equal the DENY block message.
            yield collision
            yield part_sentinel

        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("go", session_id="s-collide"):
            yielded.append(chunk)
        joined = "".join(yielded)

        # The secret part never reached the client and never persisted, even
        # though the block message text collides with the original prose.
        assert "\x1eKESTREL:PART:" not in joined
        assert "leak-me" not in joined
        assert joined == collision
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        metadata = persisted[0].get("metadata") or {}
        assert "parts" not in metadata
        import json as _json
        assert "leak-me" not in _json.dumps(persisted[0], default=str)

    @pytest.mark.asyncio
    async def test_warn_to_strict_switch_midturn_takes_effect_next_turn(self):
        """#2674 (self-review P1): a tool that flips an already-registered audit
        hook from warn to strict INSIDE a tool turn must not enforce
        retroactively. ``buffer_audit`` was decided (False) before any byte
        streamed, so pre-/post-tool prose already reached the client — a
        completion-time DENY would create a raw-bytes-streamed /
        block-message-persisted split (the fail-open leak). The mode switch is
        pinned to its turn-start value for this turn's audit and takes effect on
        the NEXT turn instead."""
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

        add_convo_calls = []
        agent, hook = _make_streaming_audit_agent(
            add_convo_calls, mode="warn",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )

        async def mock_stream(**kwargs):
            yield "Pre-tool prose. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        agent.llm_service.stream_with_tool_detection = mock_stream

        async def mock_orchestrator(**kwargs):
            # A tool flips the audit hook warn→strict mid-turn, exactly as
            # ResponseAuditFeature.enable_audit("strict") mutates hook.mode.
            hook.mode = "strict"
            yield "Post-tool synthesis that is plenty long to audit."

        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("go", session_id="s-flip"):
            yielded.append(chunk)
        joined = "".join(yielded)

        # buffer_audit was False at turn start, so the turn streamed live and the
        # mid-turn strict switch did NOT retroactively block: no block message
        # replaced the answer, and the post-tool prose reached the client.
        assert "[Response blocked by audit:" not in joined
        assert "Post-tool synthesis" in joined

        # Persistence must NOT be a block message — that split (raw streamed /
        # block persisted) is exactly the fail-open leak this fix closes. This
        # turn stays at warn (annotate at most).
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" not in persisted[0]["content"]
        assert "Post-tool synthesis" in persisted[0]["content"]

        # The switch is durable: the hook is strict now, so the NEXT turn buffers.
        assert hook.mode == "strict"

    @pytest.mark.asyncio
    async def test_strict_deny_discards_hook_emitted_parts_no_tool(self):
        """#2674 (self-review P2): a POST_RESPONSE hook that runs BEFORE the
        denying audit hook can stash the response text into an emitted part. On a
        strict DENY the drained hook-emitted parts must be discarded, not
        persisted — otherwise the denied text re-renders on reload beside the
        block message. Clearing the parsed-from-stream parts is not enough: the
        leak rides the per-turn collector. (No-tool branch.)"""
        from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
        from kestrel_sovereign.llm.adapter import LLMResponse

        class _PartEmittingHook(Hook):
            def __init__(self):
                super().__init__(
                    name="part_emitter",
                    events=[HookEvent.POST_RESPONSE],
                    priority=10,  # runs BEFORE the audit hook (priority 50)
                    timeout=5.0,
                )

            async def execute(self, hook_input):
                from kestrel_sovereign.agent.parts import emit_part
                emit_part("leaked", {"text": hook_input.response_text})
                return HookOutput.allow()

        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        agent.hooks_manager.register(_PartEmittingHook())

        async def mock_stream(**kwargs):
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])

        agent.llm_service.stream_with_tool_detection = mock_stream

        yielded = []
        async for chunk in agent.process_input_streaming("hi there", session_id="s-hookpart"):
            yielded.append(chunk)
        joined = "".join(yielded)

        # Only the block message reached the client — no part sentinel, no text.
        assert "[Response blocked by audit:" in joined
        assert "\x1eKESTREL:PART:" not in joined
        for word in ("Hello", "world", "answer"):
            assert word not in joined

        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        metadata = persisted[0].get("metadata") or {}
        # The hook-emitted part must NOT persist — it carried the denied text.
        assert "parts" not in metadata
        import json as _json
        blob = _json.dumps(persisted[0], default=str)
        for word in ("Hello", "world", "answer"):
            assert word not in blob

    @pytest.mark.asyncio
    async def test_strict_deny_discards_hook_emitted_parts_with_tools(self):
        """#2674 (self-review P2): same leak on the tool branch — a
        part-emitting POST_RESPONSE hook running before the denying audit hook
        must not persist its part when the audit DENIES."""
        from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

        class _PartEmittingHook(Hook):
            def __init__(self):
                super().__init__(
                    name="part_emitter",
                    events=[HookEvent.POST_RESPONSE],
                    priority=10,
                    timeout=5.0,
                )

            async def execute(self, hook_input):
                from kestrel_sovereign.agent.parts import emit_part
                emit_part("leaked", {"text": hook_input.response_text})
                return HookOutput.allow()

        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        agent.hooks_manager.register(_PartEmittingHook())

        async def mock_stream(**kwargs):
            yield "pre. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        agent.llm_service.stream_with_tool_detection = mock_stream

        async def mock_orchestrator(**kwargs):
            yield "SECRET-POST-TOOL-that-is-long-enough-to-audit-please."

        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("go", session_id="s-hookpart-tool"):
            yielded.append(chunk)
        joined = "".join(yielded)

        assert "[Response blocked by audit:" in joined
        assert "\x1eKESTREL:PART:" not in joined
        assert "SECRET-POST-TOOL" not in joined

        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        metadata = persisted[0].get("metadata") or {}
        assert "parts" not in metadata
        import json as _json
        assert "SECRET-POST-TOOL" not in _json.dumps(persisted[0], default=str)

    @pytest.mark.asyncio
    async def test_skip_to_enable_strict_midturn_no_retroactive_deny(self):
        """#2674 (self-review P1): a tool that calls ``audit_enable("strict")``
        while audit started in ``skip`` REGISTERS a brand-new strict hook
        mid-turn. ``buffer_audit`` was False (no POST_RESPONSE hook at turn
        start), so the turn already streamed raw. Enforcement must run only the
        turn-start snapshot — an EMPTY set — not the live registry, so the
        newly-registered hook cannot retroactively DENY and split raw-streamed
        bytes against a block-message persist (the fail-open leak). The new hook
        takes effect on the NEXT turn."""
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sdk.hooks.base import HookEvent
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
        from kestrel_sovereign.features.response_audit.feature import (
            ResponseAuditFeature,
        )

        add_convo_calls = []
        # Empty manager at turn start. The audit provider would DENY (risk 3) if
        # it ever ran on this turn — with the fix it is never consulted.
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        feature = ResponseAuditFeature(agent)
        feature._mode = "skip"
        feature._hook = None

        async def mock_stream(**kwargs):
            yield "Pre-tool prose. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        agent.llm_service.stream_with_tool_detection = mock_stream

        async def mock_orchestrator(**kwargs):
            # A tool enables strict audit mid-turn via the REAL feature
            # lifecycle — this registers a new strict hook in the live registry.
            result = await feature.enable_audit("strict")
            assert result.data.get("status") == "enabled"
            yield "Post-tool synthesis that is plenty long to audit."

        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming(
            "go", session_id="s-skip-enable",
        ):
            yielded.append(chunk)
        joined = "".join(yielded)

        # buffer_audit was False at turn start → the turn streamed live and the
        # mid-turn-registered strict hook did NOT retroactively block it.
        assert "[Response blocked by audit:" not in joined
        assert "Post-tool synthesis" in joined

        # No raw-streamed / block-persisted split: persistence is the real text.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" not in persisted[0]["content"]
        assert "Post-tool synthesis" in persisted[0]["content"]

        # The audit never ran this turn — the newly-registered hook is excluded
        # from the turn-start snapshot, so the provider was never consulted.
        agent.llm_service.get_audit_response.assert_not_called()

        # The enable is durable: the hook is registered + strict, so the NEXT
        # turn will buffer and enforce.
        enabled = agent.hooks_manager.get_enabled_hooks(HookEvent.POST_RESPONSE)
        assert any(getattr(h, "mode", None) == "strict" for h in enabled)

    @pytest.mark.asyncio
    async def test_strict_to_disable_midturn_still_audits_that_turn(self):
        """#2674 (self-review P1): a tool that calls ``audit_disable`` mid-turn
        drops the hook from the live enabled list. But the turn was BUFFERED
        because it was strict at turn start, so enforcement must still run the
        turn-start hook via the snapshot — otherwise the withheld turn releases
        with NO audit at all. The disable takes effect on the NEXT turn."""
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sdk.hooks.base import HookEvent
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
        from kestrel_sovereign.features.response_audit.feature import (
            ResponseAuditFeature,
        )

        add_convo_calls = []
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        feature = ResponseAuditFeature(agent)
        feature._mode = "skip"
        feature._hook = None
        # Turn STARTS strict (hook enabled) via the real feature lifecycle.
        await feature.enable_audit("strict")

        async def mock_stream(**kwargs):
            yield "pre. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        agent.llm_service.stream_with_tool_detection = mock_stream

        async def mock_orchestrator(**kwargs):
            # A tool disables audit mid-turn — but this turn started strict and
            # is already buffered, so the turn-start hook must still audit it.
            result = await feature.disable_audit()
            assert result.data.get("status") == "disabled"
            yield "SECRET-POST-TOOL-that-is-long-enough-to-audit-please."

        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming(
            "go", session_id="s-strict-disable",
        ):
            yielded.append(chunk)
        joined = "".join(yielded)

        # The turn-start strict hook still audited (risk 3 → DENY): only the
        # block message escaped; the withheld post-tool prose never reached the
        # client and was never persisted.
        assert "SECRET-POST-TOOL" not in joined
        assert "[Response blocked by audit:" in joined
        # Proof the audit actually ran (not released unaudited).
        agent.llm_service.get_audit_response.assert_awaited()

        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        assert "SECRET-POST-TOOL" not in persisted[0]["content"]

        # The disable is durable: no enabled hook remains, so the NEXT turn
        # streams incrementally.
        assert agent.hooks_manager.get_enabled_hooks(HookEvent.POST_RESPONSE) == []

    @pytest.mark.asyncio
    async def test_strict_approval_ask_withholds_text_fail_closed(self):
        """#2674 (self-review P1): a POST_RESPONSE approval hook that BUFFERS the
        turn (``awaits_user_input`` → ``_hook_is_enforcing`` → buffer) and returns
        ASK must fail CLOSED. ASK means "not approved yet", so none of the
        original assistant text may reach the live client — only the fail-closed
        block message is released. The prior code handled DENY only and let ASK
        fall through to the raw ALLOW passthrough, leaking the unapproved
        response."""
        from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
        from kestrel_sovereign.llm.adapter import LLMResponse

        class _ApprovalHook(Hook):
            def __init__(self):
                super().__init__(
                    name="approval_gate",
                    events=[HookEvent.POST_RESPONSE],
                    priority=50,
                    awaits_user_input=True,  # enforcing → the turn buffers
                )

            async def execute(self, hook_input):
                # Queue for human approval — the response is NOT yet approved.
                return HookOutput.ask(
                    approval_id="appr-1", reason="needs human approval",
                )

        add_convo_calls = []
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
        )
        agent.hooks_manager.register(_ApprovalHook())

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = await self._drive_no_tool(agent, exhausted, session_id="s-ask")
        joined = "".join(c for c, _ in captured)

        # No original word leaked; ASK failed closed to the block message.
        for word in ("Hello", "world", "answer"):
            assert word not in joined, f"leaked {word!r} on an ASK (fail-open)"
        assert "[Response blocked by audit:" in joined
        # Every released chunk came after the upstream stream fully drained.
        assert captured, "generator produced nothing"
        assert all(done for _, done in captured), (
            "a chunk escaped before the approval verdict"
        )

        # Persistence stores the block message, never the unapproved text.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        assert "Hello" not in persisted[0]["content"]
