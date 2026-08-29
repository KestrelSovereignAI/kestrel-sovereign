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
        mock_agent.process_input.assert_called_once()
        args, kwargs = mock_agent.process_input.call_args
        assert args == ("!help", None)
        assert kwargs["session_id"] is None
        assert kwargs["caller"] is None
        assert kwargs["invocation_context"] is None
        assert isinstance(kwargs["invocation_id"], str)
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
        # #2674 test-quality: the REAL dispatch path awaits log_tool_dispatch;
        # give it an awaitable so ``await`` doesn't raise (and get swallowed to
        # stderr) — the genuine dispatch-logging path runs clean.
        mock_agent.observability_store.log_tool_dispatch = AsyncMock(return_value="d-1")
        mock_agent.observability_store.log_structured_tool_call = AsyncMock(
            return_value="d-1"
        )

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
            '_execute_tool_batch_at_stop_boundary',
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
            '_known_tool_names',
            '_registered_tool_names',
            '_registered_features_by_tool_name',
            '_feature_supports_subagent_dispatch',
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
    # #2674 test-quality: the REAL orchestrator dispatch path (bound in the
    # two-iteration tests via ``_bind_real_orchestrator``) awaits
    # ``observability_store.log_tool_dispatch``. Leaving it a bare MagicMock made
    # ``await`` raise ``'MagicMock' object can't be awaited``, which the
    # best-effort dispatch wrapper swallowed to stderr — a real error masked, and
    # the dispatch-logging path never faithfully exercised. Provide an awaitable
    # so the genuine path runs clean. ``log_structured_tool_call`` is the legacy
    # fallback name the same helper probes; make it awaitable too.
    agent.observability_store.log_tool_dispatch = AsyncMock(return_value="dispatch-1")
    agent.observability_store.log_structured_tool_call = AsyncMock(
        return_value="dispatch-1"
    )

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
    async def test_strict_allow_with_tools_releases_and_persists_only_reviewed_text(self):
        """#2674 finding 2: a strict ALLOW releases AND persists ONLY the reviewed
        post-tool text — the exact bytes the audit examined. Under an enforcing
        (buffered) audit the client saw nothing live, so persisting the typed part
        would (a) surface structured tool data the audit never reviewed and (b)
        make the reloaded turn richer than what was released. A tool can smuggle
        secret/untrusted data into a PART's ``data`` while the benign final prose
        passes audit — so the part must NOT survive to metadata / history reload.
        The withheld pre-tool prose, revise sentinel, thinking and part sentinels
        are likewise never replayed live.

        (Pre-repair this test asserted ``"parts" in metadata`` — codifying exactly
        the retention leak this finding closes. Warn / no-audit turns keep their
        parts; the drop is specific to the fail-closed buffered path.)"""
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
            {"type": "todo", "data": {"secret": "smuggled-past-audit"}, "id": "t1"}
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
        assert "smuggled-past-audit" not in joined

        # Persistence == the reviewed release: no unreviewed side channels.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert persisted[0]["content"] == "Done — added one item."
        metadata = persisted[0].get("metadata") or {}
        assert "parts" not in metadata
        assert "pre_tool_reasoning" not in metadata
        assert "tool_events" not in metadata
        assert "tool_results" not in metadata
        # The smuggled part data appears NOWHERE in the persisted turn.
        import json as _json
        assert "smuggled-past-audit" not in _json.dumps(persisted[0], default=str)

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
    async def test_strict_deny_drops_secret_bearing_tool_event_metadata(self):
        """#2674 P1 (denied tool metadata): a codex-native tool turn embeds a
        TOOL *error* sentinel whose ``detail`` carries a secret (an exception /
        stack string the frontend renders on the tool card). That sentinel is
        the REAL production path for ``tool_events`` on the no-tool branch —
        ``_parse_stream_sentinels`` extracts it and ``_tool_parts_to_events``
        lifts ``detail`` into ``tool_events[i]['error']``. Under a strict
        (buffered) DENY the persist path drops the WHOLE metadata envelope, so
        neither the denied client stream NOR the persisted assistant row may
        contain the secret, and the row's metadata must carry no ``tool_events``,
        ``tool_results``, ``parts`` or ``pre_tool_reasoning``. One representative
        branch is exercised here: production applies the same ``buffer_audit``
        invariant to the has_tool_calls / inline / no-tool branches alike, and
        existing tests cover branch parity."""
        from kestrel_sovereign.llm.adapter import LLMResponse
        from kestrel_sovereign.agent.streaming import _build_tool_sentinel

        secret = "SECRET-TOOL-ERROR-detail-should-not-persist-abc123"

        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )

        error_sentinel = _build_tool_sentinel(
            "error", "shell", index=0, detail=secret,
        )

        async def mock_stream(**kwargs):
            # A codex-native tool ran INSIDE the LLM stream: it emitted a TOOL
            # error sentinel (secret in ``detail``) with the answer prose around
            # it. No ``tool_calls`` on the terminal LLMResponse → the no-tool
            # branch, where tool_events are built from the embedded sentinels.
            yield "Ran the shell command for you, here is a long enough answer. "
            yield error_sentinel
            yield LLMResponse(content="", tool_calls=[])

        agent.llm_service.stream_with_tool_detection = mock_stream

        yielded = []
        async for chunk in agent.process_input_streaming(
            "run it", session_id="s-tool-evt",
        ):
            yielded.append(chunk)
        joined = "".join(yielded)

        # Client saw ONLY the block message — no prose, no TOOL sentinel, no
        # secret error detail.
        assert "[Response blocked by audit:" in joined
        assert secret not in joined
        assert "\x1eKESTREL:TOOL:" not in joined
        assert "Ran the shell command" not in joined

        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        metadata = persisted[0].get("metadata") or {}
        # The buffered DENY drops the whole unreviewed metadata envelope.
        assert "tool_events" not in metadata
        assert "tool_results" not in metadata
        assert "parts" not in metadata
        assert "pre_tool_reasoning" not in metadata
        # The secret error detail appears NOWHERE in the persisted turn.
        import json as _json
        assert secret not in _json.dumps(persisted[0], default=str)

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
    async def test_strict_block_message_does_not_reflect_audit_reasoning(self):
        """#2674 finding 2: the audit hook's reasoning is UNTRUSTED — the audit
        LLM may quote raw response content (secrets and all) into it. The block
        message is the ONLY text a buffered turn releases AND persists, so it
        must be a CONSTANT sanitized string that never reflects the reasoning.
        The full reason still reaches the operator log (a different boundary)."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={
                "risk_level": 3, "reasoning": "response leaked SECRET-abc123",
            },
        )
        denied = await agent._fire_post_response_hook(
            "A long enough answer to reach the audit gate.", "s1",
        )
        assert getattr(denied, "denied", False) is True
        # The untrusted reasoning (and any secret in it) is NOT in the release.
        assert "SECRET-abc123" not in str(denied)
        assert str(denied) == (
            "[Response blocked by audit: response withheld by audit policy]"
        )

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

        # The block message is deterministic AND constant (#2674 finding 2): the
        # untrusted hook reason is NOT reflected into the released/persisted text,
        # so a strict DENY always releases exactly this sanitized string.
        collision = "[Response blocked by audit: response withheld by audit policy]"

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


class TestCommandWrapperEnforcingVerdict:
    """#2674 (follow-up): the streaming command wrapper must decide whether to
    release a turn's typed parts from the enforcing verdict stamped on the
    audited turn's own result — NOT from a fresh POST_RESPONSE registry snapshot
    taken in the wrapper. The two can diverge: ``process_input`` captures its
    audit snapshot at turn start and ``_fire_post_response_hook`` runs exactly
    that set, but a hook registered/enabled/disabled/mode-flipped between the
    wrapper's snapshot and the turn-start snapshot would let a strict-audited
    turn's unaudited parts escape while the wrapper believed the turn was
    non-enforcing.
    """

    def _make_command_agent(self, process_input):
        from unittest.mock import MagicMock
        from kestrel_sovereign.agent.streaming import StreamingMixin

        agent = MagicMock()
        agent._maybe_audit = AsyncMock()
        agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
        # Model the DIVERGENCE: the live registry is now EMPTY, so a fresh
        # in-wrapper snapshot would report "non-enforcing" and (wrongly) release
        # the parts. The fix reads the verdict off ``result`` instead, so this
        # empty live registry must NOT influence the release decision.
        agent.hooks_manager = MagicMock()
        agent.hooks_manager.get_enabled_hooks.return_value = []
        agent.process_input = process_input
        agent.process_input_streaming = (
            StreamingMixin.process_input_streaming.__get__(agent)
        )
        return agent

    async def _run(self, agent):
        chunks = []
        async for chunk in agent.process_input_streaming("!continue"):
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_enforcing_audited_result_drops_parts_despite_empty_live_registry(self):
        """An audited ``_PostResponseText(enforcing=True)`` whose live registry is
        now empty must still WITHHOLD the turn's typed parts. The old wrapper
        re-snapshotted the (now empty) registry, decided "non-enforcing", and
        would have leaked the part past the fail-closed gate."""
        from kestrel_sovereign.agent.streaming import _PostResponseText
        from kestrel_sovereign.agent.parts import emit_part, PART_SENTINEL_PREFIX

        async def process_input(user_input, model_override, **kwargs):
            # A tool emitted a typed part into the active turn collector; the
            # audit never reviewed it.
            emit_part("todo", {"secret": "leak-me"}, "t1")
            return _PostResponseText(
                "Reviewed and approved prose.", enforcing=True,
            )

        agent = self._make_command_agent(process_input)
        chunks = await self._run(agent)
        joined = "".join(chunks)

        assert "Reviewed and approved prose." in joined
        assert PART_SENTINEL_PREFIX not in joined
        assert "leak-me" not in joined

    @pytest.mark.asyncio
    async def test_nonenforcing_audited_result_releases_parts(self):
        """A warn-mode audited turn (``enforcing=False``) is not fail-closed, so
        its typed parts still flow — the ALLOW passthrough is preserved."""
        from kestrel_sovereign.agent.streaming import _PostResponseText
        from kestrel_sovereign.agent.parts import emit_part, PART_SENTINEL_PREFIX

        async def process_input(user_input, model_override, **kwargs):
            emit_part("todo", {"note": "visible"}, "t1")
            return _PostResponseText(
                "Reviewed prose.", enforcing=False,
            )

        agent = self._make_command_agent(process_input)
        chunks = await self._run(agent)
        joined = "".join(chunks)

        assert "Reviewed prose." in joined
        assert PART_SENTINEL_PREFIX in joined
        assert "visible" in joined

    @pytest.mark.asyncio
    async def test_pure_local_command_result_releases_parts(self):
        """A pure-local command returns a plain ``str`` (never an audited LLM
        turn): ``enforcing`` / ``denied`` fall back to ``False`` via ``getattr``,
        so its parts — e.g. a ``!todo add`` card — always flow, even when a
        strict audit is live."""
        from kestrel_sovereign.agent.parts import emit_part, PART_SENTINEL_PREFIX

        async def process_input(user_input, model_override, **kwargs):
            emit_part("todo", {"note": "card"}, "t1")
            return "Added todo."  # plain str — pure-local command result

        agent = self._make_command_agent(process_input)
        # A strict audit is live in the registry — proves the pure-local result
        # is judged by its OWN (non-enforcing) verdict, not the live registry.
        agent.hooks_manager.get_enabled_hooks.return_value = [MagicMock(fail_closed=True)]
        chunks = await self._run(agent)
        joined = "".join(chunks)

        assert "Added todo." in joined
        assert PART_SENTINEL_PREFIX in joined
        assert "card" in joined

    @pytest.mark.asyncio
    async def test_denied_audited_result_drops_parts(self):
        """A DENY verdict drops parts regardless of the enforcing flag — the
        client receives only the block message."""
        from kestrel_sovereign.agent.streaming import _PostResponseText
        from kestrel_sovereign.agent.parts import emit_part, PART_SENTINEL_PREFIX

        async def process_input(user_input, model_override, **kwargs):
            emit_part("todo", {"secret": "leak-me"}, "t1")
            return _PostResponseText(
                "[Response blocked by audit: unsafe]",
                denied=True, modified=True, enforcing=True,
            )

        agent = self._make_command_agent(process_input)
        chunks = await self._run(agent)
        joined = "".join(chunks)

        assert "[Response blocked by audit: unsafe]" in joined
        assert PART_SENTINEL_PREFIX not in joined
        assert "leak-me" not in joined


# =========================================================================
# #2674 P0 — command-produced LLM output through the REAL non-streaming
# ``process_input`` path.
#
# ``process_input_streaming("!continue")`` delegates to the real
# ``KestrelAgent.process_input`` (``!continue`` is rewritten to a continuation
# prompt and falls through to a normal LLM turn). That non-streaming turn is
# the code actually under test here: it snapshots the POST_RESPONSE hook set at
# turn start (``_process_input_traced_locked``) and fires the SHARED
# ``_fire_post_response_hook`` with that snapshot. These regressions bind the
# real methods (NOT a preconstructed ``_PostResponseText``, which would only
# exercise the streaming wrapper) and assert the two fail-closed invariants the
# repair prompt required: an enforcing ASK fails closed, and a mid-turn hook
# mutation (unregister / mode-flip) does not change what enforces on THIS turn.
# =========================================================================

from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput


class _CommandApprovalHook(Hook):
    """An enforcing (``awaits_user_input``) POST_RESPONSE approval hook that
    always returns ASK — "not approved yet". ``_hook_is_enforcing`` is True, so
    the turn is fail-closed and ASK must withhold the raw LLM output."""

    def __init__(self):
        super().__init__(
            name="command_approval_gate",
            events=[HookEvent.POST_RESPONSE],
            priority=50,
            awaits_user_input=True,
        )

    async def execute(self, hook_input):
        return HookOutput.ask(
            approval_id="appr-cmd", reason="needs human approval",
        )


def _wire_real_command_llm_path(agent, *, llm_content: str):
    """Bind the REAL non-streaming ``process_input`` chain onto an agent built by
    :func:`_make_streaming_audit_agent`, so ``process_input_streaming("!continue")``
    drives the genuine LLM turn (not a stubbed ``_PostResponseText``).

    The upstream LLM ``generate_with_messages`` and the orchestrator both return
    the raw (unaudited) ``llm_content``; only the environment is mocked. The
    turn-start POST_RESPONSE snapshot and ``_fire_post_response_hook`` run for
    real, so the fail-closed verdict is produced by the code under test.
    """
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.adapter import LLMResponse

    # Collaborators the non-streaming path touches beyond the streaming fixture.
    agent.storage = MagicMock()
    agent.bootstrap_service = None  # skip the bootstrap branch
    agent._session_briefed = True
    agent._privacy_mode = MagicMock()
    agent.context_stats = MagicMock()
    agent._maybe_refresh_user_byok_resolver = AsyncMock()
    agent._maybe_compact_codex_thread = AsyncMock()
    agent._assemble_post_build_system_prompt = MagicMock(return_value="system")
    agent._conversation_response_identity = MagicMock(return_value={})
    # Non-streaming path awaits the pipeline unconditionally (unlike streaming,
    # which guards on ``callable``), so a real awaitable is required here.
    agent._post_response_pipeline = AsyncMock()

    # The non-streaming LLM turn returns raw content with NO tool calls, so the
    # turn resolves through ``_handle_orchestrator_response`` → the shared audit.
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content=llm_content, tool_calls=[])
    )
    agent._handle_orchestrator_response = AsyncMock(return_value=llm_content)

    # Bind the REAL non-streaming methods — the code under test.
    agent.process_input = KestrelAgent.process_input.__get__(agent)
    agent._process_input_traced_locked = (
        KestrelAgent._process_input_traced_locked.__get__(agent)
    )
    agent._persist_assistant_conversation = (
        KestrelAgent._persist_assistant_conversation.__get__(agent)
    )
    return agent


# Raw LLM output that must NEVER reach the client / persistence on a fail-closed
# turn. Long enough to clear the audit hook's 20-char gate; no "!" so the
# non-streaming path's tool-calling-ignored log_error branch stays untaken.
_RAW_LLM_OUTPUT = "Here is the raw unaudited assistant answer that must be withheld."


class TestCommandRealProcessInputFailClosed:
    """#2674 P0: ``!continue`` (and any command that falls through to an LLM
    turn) is audited by the REAL non-streaming ``process_input``. An enforcing
    POST_RESPONSE hook must fail closed there, and the turn-start snapshot must
    pin enforcement to the turn even when a tool mutates the registry mid-turn."""

    async def _run_continue(self, agent):
        chunks = []
        async for chunk in agent.process_input_streaming("!continue"):
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_continue_enforcing_ask_fails_closed_real_path(self):
        """An enforcing approval hook returning ASK on the real command LLM turn
        must fail CLOSED: the raw LLM output is neither yielded to the client nor
        persisted — only the fail-closed block message is."""
        add_convo_calls = []
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
        )
        agent.hooks_manager.register(_CommandApprovalHook())
        _wire_real_command_llm_path(agent, llm_content=_RAW_LLM_OUTPUT)

        chunks = await self._run_continue(agent)
        joined = "".join(chunks)

        # Client saw only the block message; the raw LLM output never leaked.
        assert "[Response blocked by audit:" in joined
        assert _RAW_LLM_OUTPUT not in joined

        # Persistence stored the block message, never the raw output.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        assert _RAW_LLM_OUTPUT not in persisted[0]["content"]

    @pytest.mark.asyncio
    async def test_continue_snapshot_pins_enforcement_when_hook_unregistered_midturn(self):
        """Turn-start snapshot semantics (command path): a tool that UNREGISTERS
        the enforcing approval hook mid-turn (during the LLM/tool turn) must not
        disarm THIS turn — ``_fire_post_response_hook`` runs the turn-start
        snapshot, so the pinned hook still ASKs and the turn fails closed. The
        removal takes effect on the NEXT turn (the live registry is now empty)."""
        add_convo_calls = []
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
        )
        approval = _CommandApprovalHook()
        agent.hooks_manager.register(approval)
        _wire_real_command_llm_path(agent, llm_content=_RAW_LLM_OUTPUT)

        # The orchestrator (tool turn) unregisters the enforcing hook mid-turn,
        # AFTER the turn-start snapshot was captured, then returns the raw text.
        async def _unregister_midturn(**kwargs):
            agent.hooks_manager.unregister(approval)
            return _RAW_LLM_OUTPUT

        agent._handle_orchestrator_response = _unregister_midturn

        chunks = await self._run_continue(agent)
        joined = "".join(chunks)

        # THIS turn still failed closed via the pinned snapshot.
        assert "[Response blocked by audit:" in joined
        assert _RAW_LLM_OUTPUT not in joined
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        assert _RAW_LLM_OUTPUT not in persisted[0]["content"]

        # The removal is durable: the live registry is empty for the NEXT turn.
        assert agent.hooks_manager.get_enabled_hooks(HookEvent.POST_RESPONSE) == []

    @pytest.mark.asyncio
    async def test_continue_snapshot_pins_mode_when_flipped_midturn(self):
        """Turn-start snapshot semantics (command path): a strict audit hook
        flipped to ``warn`` mid-turn must still DENY THIS turn — the fire pins
        each snapshotted hook to its turn-start mode (strict). The switch is
        durable: the hook reads ``warn`` after the turn, so the NEXT turn is
        advisory."""
        from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook

        add_convo_calls = []
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        # A real strict ResponseAuditHook (fail_closed) registered at turn start.
        audit_hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)
        agent.hooks_manager.register(audit_hook)
        _wire_real_command_llm_path(agent, llm_content=_RAW_LLM_OUTPUT)

        # A tool flips the audit hook strict→warn mid-turn (as
        # ResponseAuditFeature.enable_audit("warn") mutates hook.mode).
        async def _flip_mode_midturn(**kwargs):
            audit_hook.mode = "warn"
            return _RAW_LLM_OUTPUT

        agent._handle_orchestrator_response = _flip_mode_midturn

        chunks = await self._run_continue(agent)
        joined = "".join(chunks)

        # THIS turn still DENYs — the fire pinned the hook to its turn-start
        # strict mode, so risk 3 blocks despite the mid-turn flip to warn.
        assert "[Response blocked by audit:" in joined
        assert _RAW_LLM_OUTPUT not in joined
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]

        # The switch is durable: the hook is warn now, so the NEXT turn is
        # advisory (the pin restores the live, mid-turn-mutated mode on exit).
        assert audit_hook.mode == "warn"


_BENIGN_NONSTREAM_ANSWER = "A benign reviewed answer, long enough to audit."


def _nonstream_secret_orchestrator(secret):
    """A non-streaming ``_handle_orchestrator_response`` that appends a
    SECRET-bearing envelope to the turn's ``tool_results`` collector (the STOP
    payload is derived from it) and returns a benign, audit-passing answer."""

    async def _orch(*, tool_results=None, **kwargs):
        if tool_results is not None:
            tool_results.append({
                "tool_call_id": "tc-sec",
                "name": "db_query",
                "arguments": {"q": "lookup"},
                "result": {"status": "ok", "data": secret},
            })
        return _BENIGN_NONSTREAM_ANSWER

    return _orch


class TestNonStreamingStopHookSanitized:
    """#2674 finding 4: the NON-streaming path (``process_input`` / ``!continue``)
    must null the STOP hook's raw ``tool_calls`` / ``tool_results`` under an
    enforcing audit, exactly as the streaming path does — otherwise an arbitrary
    ``on_stop`` subscriber receives the unreviewed tool envelopes the streaming
    path already drops."""

    @staticmethod
    def _dump(stop):
        import json as _json
        return _json.dumps({
            "response_text": stop.response_text,
            "tool_calls": stop.tool_calls,
            "tool_results": stop.tool_results,
        }, default=str)

    @pytest.mark.asyncio
    async def test_enforcing_allow_nulls_stop_tool_payload(self):
        secret = "NONSTREAM-SECRET-tool-result-xyz789"
        stop_inputs: list = []
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.hooks_manager.register(_RecordingStopHook(stop_inputs))
        _wire_real_command_llm_path(agent, llm_content=_BENIGN_NONSTREAM_ANSWER)
        agent._handle_orchestrator_response = _nonstream_secret_orchestrator(secret)

        async for _ in agent.process_input_streaming("!continue"):
            pass

        assert len(stop_inputs) == 1
        stop = stop_inputs[0]
        # Reviewed text only; raw tool envelopes nulled; secret never present.
        assert stop.tool_calls is None
        assert stop.tool_results is None
        assert secret not in self._dump(stop)

    @pytest.mark.asyncio
    async def test_advisory_retains_stop_tool_payload(self):
        """Warn (non-enforcing) mode keeps the STOP tool payload intact — the
        null-out is specific to the enforcing buffered path, not a blanket
        change to the non-streaming STOP contract."""
        marker = "NONSTREAM-warn-tool-visible-marker"
        stop_inputs: list = []
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="warn",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.hooks_manager.register(_RecordingStopHook(stop_inputs))
        _wire_real_command_llm_path(agent, llm_content=_BENIGN_NONSTREAM_ANSWER)
        agent._handle_orchestrator_response = _nonstream_secret_orchestrator(marker)

        async for _ in agent.process_input_streaming("!continue"):
            pass

        assert len(stop_inputs) == 1
        stop = stop_inputs[0]
        assert stop.tool_results is not None and len(stop.tool_results) == 1
        assert stop.tool_results[0]["result"]["data"] == marker
        assert stop.tool_calls is not None and len(stop.tool_calls) == 1
        assert stop.tool_calls[0]["name"] == "db_query"


class TestNonStreamingContentPreviewRedaction:
    """#2674 finding 5: the raw assistant ``content_preview`` the non-streaming
    path writes to DURABLE observability when the model ignores function calling
    must be REDACTED under an enforcing audit (the whole turn is withheld pending
    the verdict), while advisory / no-audit turns keep the operator diagnostic."""

    @pytest.mark.asyncio
    async def test_enforcing_redacts_content_preview(self):
        raw = "Use the tool! secret-token-abc123 leaked in prose here."
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.observability_store.log_error = AsyncMock()
        _wire_real_command_llm_path(agent, llm_content=raw)

        async for _ in agent.process_input_streaming("!continue"):
            pass

        agent.observability_store.log_error.assert_called_once()
        meta = agent.observability_store.log_error.call_args.kwargs["metadata"]
        assert "secret-token-abc123" not in meta["content_preview"]
        assert "redacted" in meta["content_preview"]

    @pytest.mark.asyncio
    async def test_advisory_keeps_content_preview(self):
        raw = "Use the tool! diagnostic-visible-marker in prose here."
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="warn",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.observability_store.log_error = AsyncMock()
        _wire_real_command_llm_path(agent, llm_content=raw)

        async for _ in agent.process_input_streaming("!continue"):
            pass

        agent.observability_store.log_error.assert_called_once()
        meta = agent.observability_store.log_error.call_args.kwargs["metadata"]
        assert "diagnostic-visible-marker" in meta["content_preview"]


# =========================================================================
# #2674 (fresh Terra findings) — regressions for the four repairs.
# =========================================================================


def _revising_event_calls(agent):
    """The ``revising`` SSE events ``_emit_revising_event`` fired via
    ``agent.emit_event`` this turn — the REAL out-of-band notifications call
    path (``/api/agent/notifications/sse``), which the helper binds for real."""
    return [
        call for call in agent.emit_event.call_args_list
        if call.args and call.args[0] == "revising"
    ]


def _tool_turn_stream():
    """A one-tool LLM stream: pre-tool prose, a ToolCallStarted marker (the
    trigger for the revising signals), then the terminal tool-call response."""
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    async def mock_stream(**kwargs):
        yield "Pre-tool prose the strict client never sees. "
        yield ToolCallStarted(index=0, id="tc-rev", name="todo_add")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc-rev", name="todo_add", arguments={})],
        )

    return mock_stream


async def _clean_post_tool_orchestrator(**kwargs):
    yield "Post-tool synthesis that is long enough to audit cleanly."


class TestStrictRevisingEventSuppression:
    """#2674 finding 1: under an enforcing (buffered) audit the ToolCallStarted
    path must NOT fire the out-of-band ``revising`` SSE event — the strict client
    never received the pre-tool prose (nothing to retract), and the event body
    carries response-derived tool id/name that would escape the fail-closed gate
    on the PARALLEL notifications channel before the verdict exists. The in-band
    revise sentinel is suppressed too. Warn mode still emits both, as before."""

    @pytest.mark.asyncio
    async def test_strict_allow_emits_no_revising_event(self):
        """Strict ALLOW: the turn completes and releases the reviewed text, but
        no revising event fired at any point (before OR after the ALLOW), and no
        in-band revise sentinel leaked."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.llm_service.stream_with_tool_detection = _tool_turn_stream()
        agent._handle_orchestrator_response_streaming = _clean_post_tool_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("go", session_id="s-rev-allow"):
            yielded.append(chunk)
        joined = "".join(yielded)

        # The turn completed normally (reviewed text released) ...
        assert "Post-tool synthesis" in joined
        # ... yet NO revising event fired and no in-band revise sentinel leaked.
        assert _revising_event_calls(agent) == []
        assert "\x1eKESTREL:REVISE:" not in joined

    @pytest.mark.asyncio
    async def test_strict_deny_emits_no_revising_event(self):
        """Strict DENY: only the block message reaches the client, and — as on
        ALLOW — no revising event fired and no revise sentinel leaked."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        agent.llm_service.stream_with_tool_detection = _tool_turn_stream()
        agent._handle_orchestrator_response_streaming = _clean_post_tool_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("go", session_id="s-rev-deny"):
            yielded.append(chunk)
        joined = "".join(yielded)

        assert "[Response blocked by audit:" in joined
        assert _revising_event_calls(agent) == []
        assert "\x1eKESTREL:REVISE:" not in joined

    @pytest.mark.asyncio
    async def test_warn_still_emits_revising_event(self):
        """Warn mode is not fail-closed (buffer_audit False): the revising event
        fires with the marker's tool id/name and the in-band sentinel rides the
        live chat stream — the established behavior, preserved."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="warn",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.llm_service.stream_with_tool_detection = _tool_turn_stream()
        agent._handle_orchestrator_response_streaming = _clean_post_tool_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming("go", session_id="s-rev-warn"):
            yielded.append(chunk)
        joined = "".join(yielded)

        events = _revising_event_calls(agent)
        assert len(events) == 1
        payload = events[0].args[1]
        assert payload["tool_call_id"] == "tc-rev"
        assert payload["tool_name"] == "todo_add"
        assert "\x1eKESTREL:REVISE:" in joined


class _RecordingStopHook(Hook):
    """A STOP-event subscriber that records the HookInput it receives — the
    stand-in for an arbitrary ``on_stop`` plugin that could persist or emit it
    (e.g. the reflection handler). Registered alongside the POST_RESPONSE audit
    hook; the STOP fire reaches it through the real ``execute_hooks_parallel``."""

    def __init__(self, sink):
        super().__init__(
            name="recording_stop_hook", events=[HookEvent.STOP], priority=50,
        )
        self._sink = sink

    async def execute(self, hook_input):
        self._sink.append(hook_input)
        return HookOutput.allow()


def _secret_tool_turn_stream():
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    async def mock_stream(**kwargs):
        yield "Working on it. "
        yield ToolCallStarted(index=0, id="tc-sec", name="db_query")
        yield LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc-sec", name="db_query", arguments={"q": "lookup"})
            ],
        )

    return mock_stream


def _secret_tool_orchestrator(secret):
    """Post-tool orchestrator that appends a SECRET-bearing envelope to the
    turn's ``tool_results`` collector (the has_tool_calls branch derives the
    STOP payload from this list) and streams a benign, audit-passing answer."""

    async def mock_orchestrator(*, tool_results=None, **kwargs):
        if tool_results is not None:
            tool_results.append({
                "tool_call_id": "tc-sec",
                "name": "db_query",
                "arguments": {"q": "lookup"},
                "result": {"status": "ok", "data": secret},
            })
        yield "Here is the benign reviewed answer, long enough to audit."

    return mock_orchestrator


class TestStrictStopHookSanitized:
    """#2674 finding 2: under an enforcing (buffered) audit the STOP HookInput
    must expose ONLY the reviewed final text — ``tool_calls`` / ``tool_results``
    (raw, unaudited tool envelopes) are nulled on every verdict so an arbitrary
    ``on_stop`` subscriber cannot persist or emit unreviewed data. Warn /
    no-audit turns keep the full STOP payload (also test_stop_hook_payload.py)."""

    @staticmethod
    def _stop_input_dump(stop):
        import json as _json
        return _json.dumps({
            "response_text": stop.response_text,
            "user_message": stop.user_message,
            "tool_calls": stop.tool_calls,
            "tool_results": stop.tool_results,
        }, default=str)

    @pytest.mark.asyncio
    async def test_strict_allow_stop_hook_omits_secret_tool_data(self):
        secret = "SECRET-TOOL-RESULT-must-not-reach-stop-hook-xyz789"
        stop_inputs: list = []
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.hooks_manager.register(_RecordingStopHook(stop_inputs))
        agent.llm_service.stream_with_tool_detection = _secret_tool_turn_stream()
        agent._handle_orchestrator_response_streaming = _secret_tool_orchestrator(secret)

        yielded = []
        async for chunk in agent.process_input_streaming("q", session_id="s-stop-allow"):
            yielded.append(chunk)
        joined = "".join(yielded)

        # The client saw only the reviewed answer; the secret never streamed.
        assert "benign reviewed answer" in joined
        assert secret not in joined

        # The STOP hook received the reviewed text and NO tool side channel.
        assert len(stop_inputs) == 1
        stop = stop_inputs[0]
        assert "benign reviewed answer" in (stop.response_text or "")
        assert stop.tool_calls is None
        assert stop.tool_results is None
        assert secret not in self._stop_input_dump(stop)

        # And the persisted assistant row carries no secret either.
        import json as _json
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert secret not in _json.dumps(persisted[0], default=str)

    @pytest.mark.asyncio
    async def test_strict_deny_stop_hook_omits_secret_tool_data(self):
        secret = "SECRET-TOOL-RESULT-denied-turn-must-not-leak-abc456"
        stop_inputs: list = []
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        agent.hooks_manager.register(_RecordingStopHook(stop_inputs))
        agent.llm_service.stream_with_tool_detection = _secret_tool_turn_stream()
        agent._handle_orchestrator_response_streaming = _secret_tool_orchestrator(secret)

        yielded = []
        async for chunk in agent.process_input_streaming("q", session_id="s-stop-deny"):
            yielded.append(chunk)
        joined = "".join(yielded)

        assert "[Response blocked by audit:" in joined
        assert secret not in joined

        assert len(stop_inputs) == 1
        stop = stop_inputs[0]
        # STOP carries the block message and no tool side channel.
        assert "[Response blocked by audit:" in (stop.response_text or "")
        assert stop.tool_calls is None
        assert stop.tool_results is None
        assert secret not in self._stop_input_dump(stop)

        import json as _json
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert secret not in _json.dumps(persisted[0], default=str)

    @pytest.mark.asyncio
    async def test_warn_stop_hook_retains_tool_data(self):
        """Warn mode (buffer_audit False) keeps the STOP tool_calls/tool_results
        intact — proving the null-out is specific to the enforcing buffered path,
        not a blanket change to the STOP contract."""
        marker = "warn-tool-result-visible-to-stop"
        stop_inputs: list = []
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="warn",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.hooks_manager.register(_RecordingStopHook(stop_inputs))
        agent.llm_service.stream_with_tool_detection = _secret_tool_turn_stream()
        agent._handle_orchestrator_response_streaming = _secret_tool_orchestrator(marker)

        async for _ in agent.process_input_streaming("q", session_id="s-stop-warn"):
            pass

        assert len(stop_inputs) == 1
        stop = stop_inputs[0]
        assert stop.tool_results is not None and len(stop.tool_results) == 1
        assert stop.tool_results[0]["result"]["data"] == marker
        assert stop.tool_calls is not None and len(stop.tool_calls) == 1
        assert stop.tool_calls[0]["name"] == "db_query"


class TestSnapshotFailClosed:
    """#2674 finding 3: a POST_RESPONSE hook-registry read failure at turn start
    must FAIL CLOSED — raise, not swallow into an empty snapshot. An empty
    snapshot would set ``buffer_audit=False`` and release/persist raw output with
    NO audit while masquerading as "no audit configured". Both entry paths
    capture the snapshot before any provider/output work, so raising invokes no
    LLM and yields/persists no raw response data."""

    def test_snapshot_raises_on_registry_read_failure(self):
        """The unit contract: a raising ``get_enabled_hooks`` surfaces as
        ``PostResponseHookSnapshotError`` — never an empty list."""
        from kestrel_sovereign.agent.streaming import (
            _snapshot_post_response_hooks, PostResponseHookSnapshotError,
        )
        mgr = MagicMock()
        mgr.get_enabled_hooks = MagicMock(side_effect=RuntimeError("registry down"))
        with pytest.raises(PostResponseHookSnapshotError):
            _snapshot_post_response_hooks(mgr)

    def test_snapshot_empty_without_hooks_manager(self):
        """A missing hooks_manager is a LEGITIMATE no-audit state (empty
        snapshot), NOT an infrastructure failure — it must not raise."""
        from kestrel_sovereign.agent.streaming import _snapshot_post_response_hooks
        assert _snapshot_post_response_hooks(None) == []

    @pytest.mark.asyncio
    async def test_streaming_snapshot_failure_fails_closed_no_llm_no_persist(self):
        """Streaming entry path: a turn-start registry read failure raises
        before the LLM stream is opened and persists no assistant row."""
        from kestrel_sovereign.agent.streaming import PostResponseHookSnapshotError

        add_convo_calls = []
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
        )
        # The turn-start POST_RESPONSE registry read blows up (infra failure).
        agent.hooks_manager.get_enabled_hooks = MagicMock(
            side_effect=RuntimeError("hook registry unavailable")
        )
        llm_called = {"n": 0}

        async def _never_stream(**kwargs):
            llm_called["n"] += 1
            yield "raw unaudited output that must never stream"

        agent.llm_service.stream_with_tool_detection = _never_stream

        with pytest.raises(PostResponseHookSnapshotError):
            async for _ in agent.process_input_streaming("hi there", session_id="s-snap"):
                pass

        # Failed closed BEFORE any provider work or assistant persistence.
        assert llm_called["n"] == 0
        assert [c for c in add_convo_calls if c["role"] == "assistant"] == []
        # #2674 finding 2: the turn-start snapshot is now captured at TRUE turn
        # start — before USER_PROMPT_SUBMIT and before the user turn is persisted
        # (that persistence happens later, during context assembly). So an
        # infrastructure failure reading the hook registry aborts the ENTIRE turn
        # before anything at all is written — a strictly cleaner fail-closed than
        # the prior ordering (which had already stored the user row). Nothing —
        # not even the user's own message — is persisted on this abort.
        assert add_convo_calls == []

    @pytest.mark.asyncio
    async def test_command_continue_snapshot_failure_fails_closed_no_llm_no_persist(self):
        """Non-streaming command-produced LLM path: ``!continue`` delegates to
        the REAL ``process_input`` → ``_process_input_traced_locked``, whose
        turn-start snapshot read failure must raise before any LLM turn and
        persist no assistant row (the same fail-closed invariant as streaming)."""
        from kestrel_sovereign.agent.streaming import PostResponseHookSnapshotError

        add_convo_calls = []
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
        )
        _wire_real_command_llm_path(agent, llm_content=_RAW_LLM_OUTPUT)
        agent.hooks_manager.get_enabled_hooks = MagicMock(
            side_effect=RuntimeError("hook registry unavailable")
        )

        with pytest.raises(PostResponseHookSnapshotError):
            async for _ in agent.process_input_streaming("!continue"):
                pass

        # No LLM turn ran, and no raw assistant output was persisted.
        agent.llm_service.generate_with_messages.assert_not_called()
        agent._handle_orchestrator_response.assert_not_called()
        assert [c for c in add_convo_calls if c["role"] == "assistant"] == []
        assert not any(
            _RAW_LLM_OUTPUT in (c.get("content") or "") for c in add_convo_calls
        )


# =========================================================================
# #2674 finding 1 — the REAL orchestrator continuation must honor the
# enforcing buffer. A follow-up ToolCallStarted in a multi-tool continuation
# fires an out-of-band ``revising`` SSE event (parallel notifications channel)
# and yields the in-band revise sentinel; both must be suppressed under a
# buffered (strict) audit, since the outer loop cannot retract the SSE event.
# These bind the REAL ``_handle_orchestrator_response_streaming`` (prior tests
# replaced it) and drive a genuine two-iteration tool flow.
# =========================================================================


def _bind_real_orchestrator(agent):
    """Bind the REAL orchestrator-engine + tool-registry methods so a streaming
    turn runs its genuine multi-iteration tool loop (code under test)."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    for m in (
        "_handle_orchestrator_response_streaming",
        "_execute_tool_with_hooks", "_execute_tool_batch", "_partition_tool_calls",
        "_dispatch_tool_call", "_dispatch_feature_tool", "_dispatch_direct_tool",
        "_get_denied_tools", "_handle_feature_error", "_prune_orchestrator_messages",
        "_build_feature_tools", "_visible_features_by_tool_name",
        "_visible_known_tool_names", "_known_tool_names",
        "_registered_tool_names", "_registered_features_by_tool_name",
        "_feature_supports_subagent_dispatch", "_hidden_context_features",
        "_hidden_context_tools", "_feature_hidden_from_context",
        "_direct_tool_hidden_from_context", "_build_assistant_tool_history_msg",
        "_append_executed_tool_breadcrumbs", "_make_inline_tool_executor",
        "_repair_premature_turn_yield", "_log_tool_dispatch", "_tool_call_adapter",
    ):
        bound = getattr(KestrelAgent, m, None)
        if bound is not None:
            setattr(agent, m, bound.__get__(agent))
    agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg
    agent._explored_features = {}
    agent._direct_tool_defs = []
    agent._direct_tools = {}
    agent._tool_to_feature = {}
    agent._register_explored_feature_tools = MagicMock()
    return agent


def _two_iteration_tool_feature():
    """A feature whose ``test_tool`` succeeds — dispatched twice across a
    two-iteration continuation."""
    feature = MagicMock()
    feature.tool_name = "test_tool"
    feature.name = "test_feature"
    feature.execute_as_subagent = AsyncMock(
        return_value={"success": True, "data": "tool ok"}
    )
    feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {"name": "test_tool", "description": "t", "parameters": {}},
    }
    return feature


def _two_iteration_streams():
    """Build a ``stream_with_tool_detection`` mock driving a two-iteration tool
    continuation: main call → tool_calls; follow-up #1 → a ToolCallStarted
    (the orchestrator revise trigger) + another tool round; follow-up #2 →
    final synthesis text with no more tools."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
    from kestrel_sdk.llm import ToolCallStarted

    state = {"n": 0}

    async def mock_stream(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            # Main call: go straight to a tool call (no pre-tool prose, so the
            # ONLY revise trigger is the orchestrator follow-up below).
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={"a": 1})],
            )
        elif state["n"] == 2:
            # Follow-up for iteration 0 — the multi-tool continuation branch:
            # a ToolCallStarted (fires _emit_revising_event at engine:~2700)
            # AND another detected tool call forcing iteration 1.
            yield ToolCallStarted(index=0, id="tc2", name="test_tool")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="test_tool", arguments={"b": 2})],
            )
        else:
            # Follow-up for iteration 1 — final synthesis, no more tools.
            for w in ("This ", "is ", "the ", "final ", "reviewed ", "answer."):
                yield w
            yield LLMResponse(content="", tool_calls=[])

    return mock_stream, state


class TestFinding1OrchestratorReviseSuppression:
    """#2674 finding 1: real two-iteration continuation honors the buffer."""

    def _agent(self, add_convo, *, mode):
        agent, hook = _make_streaming_audit_agent(
            add_convo, mode=mode,
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.features = {"test_feature": _two_iteration_tool_feature()}
        _bind_real_orchestrator(agent)
        stream, _state = _two_iteration_streams()
        agent.llm_service.stream_with_tool_detection = stream
        return agent, hook

    async def _drive(self, agent):
        chunks = []
        async for c in agent.process_input_streaming("hi there", session_id="s1"):
            chunks.append(c)
        return chunks

    @pytest.mark.asyncio
    async def test_strict_allow_two_iterations_suppresses_revise(self):
        add_convo = []
        agent, _hook = self._agent(add_convo, mode="strict")
        chunks = await self._drive(agent)
        joined = "".join(str(c) for c in chunks)

        # No in-band revise sentinel reached the client on the buffered turn.
        assert "KESTREL:REVISE" not in joined
        # No out-of-band `revising` SSE event fired (the outer loop could never
        # retract it) — this is the core of finding 1.
        revising = [
            call for call in agent.emit_event.call_args_list
            if call.args and call.args[0] == "revising"
        ]
        assert revising == [], f"revising SSE leaked under buffer: {revising}"
        # ALLOW released the reviewed synthesis exactly once, after the verdict.
        assert "final reviewed answer." in joined
        persisted = [c for c in add_convo if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "final reviewed answer." in persisted[0]["content"]

    @pytest.mark.asyncio
    async def test_strict_deny_two_iterations_blocks_and_suppresses_revise(self):
        add_convo = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "harmful"},
        )
        agent.features = {"test_feature": _two_iteration_tool_feature()}
        _bind_real_orchestrator(agent)
        stream, _state = _two_iteration_streams()
        agent.llm_service.stream_with_tool_detection = stream

        chunks = await self._drive(agent)
        joined = "".join(str(c) for c in chunks)

        # DENY exposes only the block message; the synthesis text never leaks.
        assert "[Response blocked by audit:" in joined
        assert "final reviewed answer" not in joined
        assert "KESTREL:REVISE" not in joined
        revising = [
            call for call in agent.emit_event.call_args_list
            if call.args and call.args[0] == "revising"
        ]
        assert revising == [], f"revising SSE leaked under buffered DENY: {revising}"
        persisted = [c for c in add_convo if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        assert "final reviewed answer" not in persisted[0]["content"]

    @pytest.mark.asyncio
    async def test_warn_two_iterations_still_emits_revise(self):
        """Advisory (warn) turns are NOT buffered — the follow-up revise SSE +
        in-band sentinel must still fire (no behavior change off the gate)."""
        add_convo = []
        agent, _hook = self._agent(add_convo, mode="warn")
        chunks = await self._drive(agent)
        joined = "".join(str(c) for c in chunks)

        # The in-band revise sentinel streamed live.
        assert "KESTREL:REVISE" in joined
        # The out-of-band revising SSE event fired for the follow-up tool.
        revising = [
            call for call in agent.emit_event.call_args_list
            if call.args and call.args[0] == "revising"
        ]
        assert revising, "warn turn must still emit the revising SSE event"
        assert "final reviewed answer." in joined


# =========================================================================
# #2674 (Terra P1) — a strict/buffered continuation that TIMES OUT must not
# collapse into a silent empty 200. In advisory mode a follow-up-LLM timeout
# renders as an ❌ tool card (a tool sentinel); under an enforcing (buffered)
# audit that sentinel is stripped by the buffering gate, leaving empty audited
# text, empty persisted content, and an empty response. The strict path must
# instead deliver + persist a deterministic non-empty safe block, discarding any
# partial/protected buffered continuation, while advisory behavior is preserved.
# These bind the REAL orchestrator continuation loop and force a genuine
# ``asyncio.timeout`` on the follow-up synthesis call.
# =========================================================================

import asyncio
from unittest.mock import patch


def _tool_then_timeout_streams(*, partial_marker: str = ""):
    """``stream_with_tool_detection`` mock: main call returns a tool call; the
    orchestrator continuation optionally yields ``partial_marker`` prose and then
    HANGS so the per-call ``asyncio.timeout`` fires. ``partial_marker`` lets a
    test prove protected/partial buffered text is discarded, never released."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    state = {"n": 0}

    async def mock_stream(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            # Main call → a single tool call (drives the has_tool_calls branch).
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={"a": 1})],
            )
            return
        # Continuation (post-tool synthesis): optionally stream partial prose,
        # then hang until the orchestrator's asyncio.timeout cancels us.
        if partial_marker:
            yield partial_marker
        await asyncio.sleep(30)
        yield LLMResponse(content="", tool_calls=[])  # pragma: no cover

    return mock_stream, state


class TestStrictContinuationTimeoutSafeBlock:
    """#2674 (Terra P1): strict buffered continuation timeout → safe block."""

    def _agent(self, add_convo, *, mode, partial_marker=""):
        agent, hook = _make_streaming_audit_agent(
            add_convo, mode=mode,
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        agent.features = {"test_feature": _two_iteration_tool_feature()}
        _bind_real_orchestrator(agent)
        stream, _state = _tool_then_timeout_streams(partial_marker=partial_marker)
        agent.llm_service.stream_with_tool_detection = stream
        return agent, hook

    async def _drive(self, agent):
        chunks = []
        async for c in agent.process_input_streaming("hi there", session_id="s1"):
            chunks.append(c)
        return chunks

    @pytest.mark.asyncio
    async def test_strict_timeout_delivers_nonempty_safe_block(self):
        from kestrel_sovereign.agent.streaming import (
            STRICT_AUDIT_CONTINUATION_TIMEOUT_BLOCK as BLOCK,
        )

        add_convo = []
        agent, _hook = self._agent(add_convo, mode="strict")
        # Force the per-call watchdog to fire almost immediately.
        with patch(
            "kestrel_sovereign.agent.orchestrator_engine."
            "ORCHESTRATOR_TURN_TIMEOUT_SECS", 0.05,
        ):
            chunks = await self._drive(agent)
        joined = "".join(str(c) for c in chunks)

        # The strict client gets a NON-EMPTY, deterministic safe block — never a
        # silent empty 200.
        assert joined.strip(), "strict timeout produced an empty response"
        assert BLOCK in joined
        # No wire-protocol tool sentinel escaped (advisory-only rendering), and no
        # raw timeout/error detail leaked.
        assert "KESTREL:TOOL" not in joined
        assert "timeout after" not in joined
        # Persisted assistant content EQUALS the delivered block (so a history
        # reload replays exactly what was released — the #2674 invariant).
        persisted = [c for c in add_convo if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert persisted[0]["content"] == BLOCK
        assert joined.count(BLOCK) == 1

    @pytest.mark.asyncio
    async def test_strict_timeout_discards_partial_protected_text(self):
        """Partial synthesis prose buffered before the timeout is PROTECTED,
        unfinished, unreviewed content — it must be discarded, never released or
        persisted, and replaced wholesale by the safe block."""
        from kestrel_sovereign.agent.streaming import (
            STRICT_AUDIT_CONTINUATION_TIMEOUT_BLOCK as BLOCK,
        )

        marker = "PROTECTED_PARTIAL_MARKER_must_never_surface"
        add_convo = []
        agent, _hook = self._agent(add_convo, mode="strict", partial_marker=marker)
        with patch(
            "kestrel_sovereign.agent.orchestrator_engine."
            "ORCHESTRATOR_TURN_TIMEOUT_SECS", 0.05,
        ):
            chunks = await self._drive(agent)
        joined = "".join(str(c) for c in chunks)

        assert marker not in joined, "partial protected text leaked to client"
        assert joined.strip() and BLOCK in joined
        persisted = [c for c in add_convo if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert marker not in persisted[0]["content"]
        assert persisted[0]["content"] == BLOCK

    @pytest.mark.asyncio
    async def test_advisory_timeout_preserves_error_sentinel(self):
        """Warn (advisory) turns are NOT buffered: a continuation timeout must
        still render the ❌ error tool card (tool sentinel) exactly as before —
        the safe-block substitution is strict-only."""
        from kestrel_sovereign.agent.streaming import (
            STRICT_AUDIT_CONTINUATION_TIMEOUT_BLOCK as BLOCK,
        )

        add_convo = []
        agent, _hook = self._agent(add_convo, mode="warn")
        with patch(
            "kestrel_sovereign.agent.orchestrator_engine."
            "ORCHESTRATOR_TURN_TIMEOUT_SECS", 0.05,
        ):
            chunks = await self._drive(agent)
        joined = "".join(str(c) for c in chunks)

        # Advisory path keeps the in-band error tool sentinel with its detail...
        assert "KESTREL:TOOL" in joined
        assert "timeout after" in joined
        # ...and does NOT substitute the strict safe block.
        assert BLOCK not in joined


# =========================================================================
# #2674 finding 2 — principled enforcing POST_RESPONSE pipeline. When an
# enforcing gate hook is present, ONLY the gate sees the raw response; advisory
# observers (any priority) run AFTER on the REVIEWED text and cannot mutate the
# release. When no enforcing hook exists, the normal priority/MODIFY chain is
# preserved.
# =========================================================================

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput


class _RecorderHook(Hook):
    """Advisory observer that records the response_text it is handed (models an
    external recorder that persists/emits)."""

    def __init__(self, priority: int):
        super().__init__(
            name=f"recorder-{priority}", events=[HookEvent.POST_RESPONSE],
            priority=priority, timeout=5.0,
        )
        self.seen = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.seen.append(input.response_text)
        return HookOutput.allow()


class _ModifierHook(Hook):
    """Advisory observer that tries to rewrite response_text."""

    def __init__(self, priority: int, new_text: str):
        super().__init__(
            name=f"modifier-{priority}", events=[HookEvent.POST_RESPONSE],
            priority=priority, timeout=5.0,
        )
        self.new_text = new_text
        self.seen = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.seen.append(input.response_text)
        return HookOutput.modify(
            updated_input={"response_text": self.new_text}, reason="rewrite",
        )


class _EnforcingGateHook(Hook):
    """A fail-closed gate hook whose verdict is scripted (ALLOW/MODIFY/DENY/ASK)."""

    def __init__(self, priority: int, verdict: str, *, text: str = None):
        super().__init__(
            name=f"gate-{priority}", events=[HookEvent.POST_RESPONSE],
            priority=priority, timeout=5.0,
        )
        self.fail_closed = True
        self.verdict = verdict
        self.text = text
        self.seen = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.seen.append(input.response_text)
        if self.verdict == "allow":
            return HookOutput.allow()
        if self.verdict == "modify":
            return HookOutput.modify(
                updated_input={"response_text": self.text}, reason="gate rewrite",
            )
        if self.verdict == "deny":
            return HookOutput.deny(self.text or "gate deny")
        if self.verdict == "ask":
            out = HookOutput.deny(self.text or "needs approval")
            from kestrel_sdk.hooks.base import PermissionDecision
            out.permission_decision = PermissionDecision.ASK
            return out
        return HookOutput.allow()


def _snap(manager):
    from kestrel_sovereign.agent.streaming import _snapshot_post_response_hooks
    return _snapshot_post_response_hooks(manager)


class TestFinding2EnforcingPipeline:
    """#2674 finding 2 — direct ``_fire_post_response_hook`` pipeline tests."""

    RAW = "The raw unaudited assistant answer with sensitive detail."

    def _agent_with_gate(self, add_convo, *, mode="strict", audit_risk=1):
        agent, hook = _make_streaming_audit_agent(
            add_convo, mode=mode,
            audit_response={"risk_level": audit_risk, "reasoning": "x"},
        )
        return agent, hook

    @pytest.mark.asyncio
    async def test_low_priority_recorder_never_sees_raw_on_deny(self):
        """A lower-priority advisory recorder must receive the SANITIZED block,
        never the raw response, even though its priority is BEFORE the gate."""
        agent, _hook = self._agent_with_gate([], mode="strict", audit_risk=3)
        recorder = _RecorderHook(priority=10)  # BEFORE the strict gate (50)
        agent.hooks_manager.register(recorder)

        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=_snap(agent.hooks_manager),
        )
        assert result.denied is True
        assert "[Response blocked by audit:" in str(result)
        # The recorder ran on the sanitized block — the raw text never reached it.
        assert recorder.seen, "recorder did not run"
        assert all(self.RAW not in seen for seen in recorder.seen)
        assert all("[Response blocked by audit:" in seen for seen in recorder.seen)

    @pytest.mark.asyncio
    async def test_high_priority_modifier_cannot_rewrite_release(self):
        """A higher-priority advisory modifier (AFTER the gate) must not rewrite
        the gate-approved release; the reviewed text is released unchanged."""
        agent, _hook = self._agent_with_gate([], mode="strict", audit_risk=1)
        modifier = _ModifierHook(priority=90, new_text="UNAUDITED REWRITE")
        agent.hooks_manager.register(modifier)

        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=_snap(agent.hooks_manager),
        )
        assert result.denied is False
        assert str(result) == self.RAW
        assert "UNAUDITED REWRITE" not in str(result)
        # The modifier observed the reviewed text (== RAW on ALLOW) but its
        # rewrite was dropped.
        assert modifier.seen == [self.RAW]

    @pytest.mark.asyncio
    async def test_gate_modify_releases_reviewed_rewrite_to_observers(self):
        """An enforcing gate MODIFY becomes the reviewed release; observers see
        the rewritten (reviewed) text, never the raw."""
        add_convo = []
        agent, hook = _make_streaming_audit_agent(add_convo, mode="warn")
        # Replace the warn hook with a scripted enforcing MODIFY gate.
        agent.hooks_manager.unregister(hook)
        gate = _EnforcingGateHook(priority=50, verdict="modify", text="REVIEWED REWRITE")
        recorder = _RecorderHook(priority=10)
        agent.hooks_manager.register(gate)
        agent.hooks_manager.register(recorder)

        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=_snap(agent.hooks_manager),
        )
        assert result.modified is True
        assert str(result) == "REVIEWED REWRITE"
        assert recorder.seen == ["REVIEWED REWRITE"]
        assert all(self.RAW not in s for s in recorder.seen)

    @pytest.mark.asyncio
    async def test_gate_ask_blocks_and_observer_sees_sanitized(self):
        add_convo = []
        agent, hook = _make_streaming_audit_agent(add_convo, mode="warn")
        agent.hooks_manager.unregister(hook)
        gate = _EnforcingGateHook(priority=50, verdict="ask", text="secret reason")
        recorder = _RecorderHook(priority=10)
        agent.hooks_manager.register(gate)
        agent.hooks_manager.register(recorder)

        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=_snap(agent.hooks_manager),
        )
        assert result.denied is True
        assert "withheld pending approval" in str(result)
        assert "secret reason" not in str(result)
        assert recorder.seen and all(self.RAW not in s for s in recorder.seen)

    @pytest.mark.asyncio
    async def test_observer_can_reblock_approved_release(self):
        """A stricter observer DENY re-blocks an otherwise-approved release
        (fail-safe), but only sees the reviewed text."""
        agent, _hook = self._agent_with_gate([], mode="strict", audit_risk=1)

        class _DenyObserver(Hook):
            def __init__(self):
                super().__init__(name="deny-obs", events=[HookEvent.POST_RESPONSE],
                                 priority=90, timeout=5.0)
                self.seen = []
            async def execute(self, input):
                self.seen.append(input.response_text)
                return HookOutput.deny("observer policy")

        obs = _DenyObserver()
        agent.hooks_manager.register(obs)
        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=_snap(agent.hooks_manager),
        )
        assert result.denied is True
        assert "[Response blocked by audit:" in str(result)
        # The observer saw the reviewed text (== RAW on gate ALLOW), never before.
        assert obs.seen == [self.RAW]

    @pytest.mark.asyncio
    async def test_no_enforcing_hook_preserves_modify_chain(self):
        """With NO enforcing hook, the normal priority-ordered MODIFY chain is
        preserved: a lower-priority modifier's rewrite feeds the higher one and
        the final rewrite is returned."""
        add_convo = []
        agent, hook = _make_streaming_audit_agent(add_convo, mode="warn")
        agent.hooks_manager.unregister(hook)
        first = _ModifierHook(priority=10, new_text="FIRST REWRITE")
        second = _RecorderHook(priority=20)
        agent.hooks_manager.register(first)
        agent.hooks_manager.register(second)

        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=_snap(agent.hooks_manager),
        )
        assert str(result) == "FIRST REWRITE"
        # Normal chaining: the second hook saw the FIRST hook's rewrite.
        assert second.seen == ["FIRST REWRITE"]


class TestFinding2EndToEndStreaming:
    """#2674 finding 2 — an external recorder in the live streaming path only
    ever sees the reviewed/sanitized release, not the raw buffered response."""

    @pytest.mark.asyncio
    async def test_streaming_strict_deny_recorder_sees_sanitized(self):
        add_convo = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "harmful"},
        )
        recorder = _RecorderHook(priority=10)
        agent.hooks_manager.register(recorder)

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])

        agent.llm_service.stream_with_tool_detection = mock_stream

        chunks = []
        async for c in agent.process_input_streaming("hi there", session_id="s1"):
            chunks.append(c)
        joined = "".join(str(c) for c in chunks)

        assert "[Response blocked by audit:" in joined
        assert "Hello" not in joined
        # The external recorder observed only the sanitized block message.
        assert recorder.seen, "recorder never ran"
        assert all(_LONG_TEXT not in s for s in recorder.seen)
        assert all("Hello" not in s for s in recorder.seen)


def _install_real_resolver(agent):
    """Give the mock llm_service a REAL invocation-context resolver so
    ``resolve_turn_invocation_context`` yields a genuine ``LLMInvocationContext``
    (the redaction wiring guards on ``isinstance`` to tolerate test doubles)."""
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
    from dataclasses import replace

    def _resolve(ctx=None, session_id=None):
        base = ctx if isinstance(ctx, LLMInvocationContext) else LLMInvocationContext()
        if session_id and not base.session_id:
            base = replace(base, session_id=session_id)
        return base

    agent.llm_service._resolve_invocation_context = _resolve
    return agent


class TestFinding3MainCallRedactionWiring:
    """#2674 finding 3 — the enforcing turn threads ``redact_content`` onto the
    FROZEN per-turn context handed to the main provider call (streaming and
    non-streaming), so its durable telemetry redacts. The redaction itself is
    unit-tested in test_llm_content_redaction_audit.py; here we prove the wiring
    end-to-end through the real turn paths."""

    @pytest.mark.asyncio
    async def test_streaming_strict_marks_context_redacting(self):
        add_convo = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        _install_real_resolver(agent)
        seen = {}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            seen["ctx"] = kwargs.get("invocation_context")
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])

        agent.llm_service.stream_with_tool_detection = mock_stream
        async for _ in agent.process_input_streaming("hi there", session_id="s1"):
            pass
        assert seen["ctx"] is not None
        assert seen["ctx"].redact_content is True

    @pytest.mark.asyncio
    async def test_streaming_warn_context_not_redacting(self):
        add_convo = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo, mode="warn",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        _install_real_resolver(agent)
        seen = {}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            seen["ctx"] = kwargs.get("invocation_context")
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])

        agent.llm_service.stream_with_tool_detection = mock_stream
        async for _ in agent.process_input_streaming("hi there", session_id="s1"):
            pass
        assert seen["ctx"] is not None
        assert seen["ctx"].redact_content is False

    @pytest.mark.asyncio
    async def test_nonstream_strict_marks_context_redacting(self):
        add_convo = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        _install_real_resolver(agent)
        _wire_real_command_llm_path(agent, llm_content=_RAW_LLM_OUTPUT)
        # Drive the real non-streaming turn via the !continue fall-through.
        async for _ in agent.process_input_streaming("!continue", session_id="s1"):
            pass
        _args, kwargs = agent.llm_service.generate_with_messages.call_args
        ctx = kwargs.get("invocation_context")
        assert ctx is not None and ctx.redact_content is True


# #2674 finding 6: the endpoint-level regression that arbitrary exception text
# is never reflected to the client lives in ``tests/unit/test_agent_cancellation.py``
# (``TestStreamEndpointErrorContract``), which drives the REAL FastAPI streaming
# endpoint through a TestClient. The prior ``TestFinding0EndpointExceptionNoLeak``
# here only proved that ONE hand-picked provider error happened not to embed the
# buffered text — it never exercised the endpoint's ``str(e)`` renderer, which was
# the actual leak. It was removed in favor of the TestClient contract.


# =========================================================================
# #2674 finding 1 — the post-gate OBSERVER phase must receive the reviewed
# release ONLY. Under an enforcing gate there is no reviewed equivalent of
# the raw side channels, so pre_tool_prose / tool_calls / tool_results are
# nulled in the observer's HookInput. An observer is an arbitrary advisory
# consumer (recorder / webhook) that may persist or forward its input; handing
# it those raw fields would leak past the fail-closed gate. This drives the
# SHARED _fire_post_response_hook (streaming AND non-streaming) directly across
# ALLOW / MODIFY / ASK / DENY; advisory-only turns never enter this phase.
# =========================================================================


def _gate_and_observer(gate_output):
    """A fail-closed gate hook returning ``gate_output`` + a non-enforcing
    observer that records the HookInput it was handed. Returns
    ``(gate, observer, snapshot)`` where snapshot is the turn-start
    ``[(hook, mode, enforcing), ...]`` list the shared fire path pins."""
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
    from kestrel_sovereign.agent.streaming import _NO_MODE

    class _Gate(Hook):
        def __init__(self):
            super().__init__(name="gate", events=[HookEvent.POST_RESPONSE], priority=10)

        @property
        def fail_closed(self):
            return True

        async def execute(self, input):
            return gate_output

    class _Observer(Hook):
        def __init__(self):
            super().__init__(name="observer", events=[HookEvent.POST_RESPONSE], priority=100)
            self.seen = None

        @property
        def fail_closed(self):
            return False

        async def execute(self, input):
            self.seen = input
            return HookOutput.allow()

    gate, observer = _Gate(), _Observer()
    # #2674 finding 1: the snapshot tuple is (hook, mode, enforcing) — enforcement
    # captured at turn start. Gate is fail-closed (enforcing), observer is not.
    snapshot = [(gate, _NO_MODE, True), (observer, _NO_MODE, False)]
    return gate, observer, snapshot


class TestFinding1ObserverReceivesReviewedOnly:
    """The observer never sees raw pre_tool_prose / tool_calls / tool_results."""

    async def _run(self, gate_output):
        from kestrel_sdk.hooks.base import HookEvent
        add_convo = []
        agent, _none = _make_streaming_audit_agent(
            add_convo, mode="strict", register_hook=False,
        )
        gate, observer, snapshot = _gate_and_observer(gate_output)
        for h in (gate, observer):
            agent.hooks_manager.register(h)
        verdict = await agent._fire_post_response_hook(
            "RAW ASSISTANT RESPONSE",
            "sess",
            pre_tool_prose="RAW PRE-TOOL PROSE that mentions a SECRET",
            tool_calls=[{"id": "t1", "name": "send", "arguments": {"x": "SECRET"}}],
            tool_results=[{"tool_call_id": "t1", "name": "send", "result": "SECRET RESULT"}],
            hook_snapshot=snapshot,
        )
        return verdict, observer

    async def _assert_observer_clean(self, gate_output, *, expected_text_contains=None,
                                     expected_text_absent=None):
        verdict, observer = await self._run(gate_output)
        assert observer.seen is not None, "observer must have run"
        # The three raw side channels are nulled — no unreviewed leak.
        assert observer.seen.pre_tool_prose is None
        assert observer.seen.tool_calls is None
        assert observer.seen.tool_results is None
        # The observer sees exactly the reviewed release, nothing else.
        assert observer.seen.response_text == str(verdict)
        import json as _json
        blob = "|".join([
            observer.seen.response_text or "",
            _json.dumps(observer.seen.tool_calls),
            _json.dumps(observer.seen.tool_results),
            observer.seen.pre_tool_prose or "",
        ])
        assert "SECRET" not in blob
        if expected_text_contains:
            assert expected_text_contains in observer.seen.response_text
        if expected_text_absent:
            assert expected_text_absent not in observer.seen.response_text
        return verdict, observer

    @pytest.mark.asyncio
    async def test_allow(self):
        from kestrel_sdk.hooks.base import HookOutput
        await self._assert_observer_clean(HookOutput.allow())

    @pytest.mark.asyncio
    async def test_modify(self):
        from kestrel_sdk.hooks.base import HookOutput
        await self._assert_observer_clean(
            HookOutput.modify(updated_input={"response_text": "REVIEWED REWRITE"}),
            expected_text_contains="REVIEWED REWRITE",
        )

    @pytest.mark.asyncio
    async def test_deny(self):
        from kestrel_sdk.hooks.base import HookOutput
        # DENY releases only the sanitized block message; observer sees THAT,
        # never the raw response or the raw side channels.
        await self._assert_observer_clean(
            HookOutput.deny("model leaked SECRET"),
            expected_text_absent="RAW ASSISTANT RESPONSE",
        )

    @pytest.mark.asyncio
    async def test_ask(self):
        from kestrel_sdk.hooks.base import HookOutput
        await self._assert_observer_clean(
            HookOutput.ask(approval_id="a1", reason="needs approval"),
            expected_text_absent="RAW ASSISTANT RESPONSE",
        )


# =========================================================================
# #2674 finding 2 — the POST_RESPONSE hook snapshot is captured at TRUE turn
# start, BEFORE USER_PROMPT_SUBMIT. A USER_PROMPT_SUBMIT hook that unregisters
# / disables the strict audit must NOT turn the fail-closed gate off for the
# very turn it was pinned at the start of; registration / removal / mode flips
# are NEXT-turn transitions. Streaming AND non-streaming entry paths.
# =========================================================================


def _user_prompt_hook_that(fn):
    """A USER_PROMPT_SUBMIT hook whose ``execute`` runs ``fn(manager)`` — used
    to mutate the POST_RESPONSE registry mid-lifecycle at the exact point the
    finding targets (between real turn start and the snapshot's OLD location)."""
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput

    class _UP(Hook):
        def __init__(self, manager):
            super().__init__(name="up_mutator", events=[HookEvent.USER_PROMPT_SUBMIT], priority=1)
            self._manager = manager

        async def execute(self, input):
            fn(self._manager)
            return HookOutput.allow()

    return _UP


class TestFinding2SnapshotBeforeUserPrompt:
    """A USER_PROMPT hook cannot disable the strict audit for the current turn."""

    @pytest.mark.asyncio
    async def test_streaming_user_prompt_unregister_still_enforces(self):
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "harmful content"},
        )
        # A USER_PROMPT_SUBMIT hook unregisters the strict audit as its side
        # effect. Under the pre-#2674-finding-2 ordering (snapshot AFTER
        # USER_PROMPT_SUBMIT) this would empty the snapshot → buffer_audit False
        # → the raw text streams unaudited. The turn-start snapshot closes it.
        up_cls = _user_prompt_hook_that(
            lambda mgr: mgr.unregister_by_name("response_audit")
        )
        agent.hooks_manager.register(up_cls(agent.hooks_manager))

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = []
        async for chunk in agent.process_input_streaming("hi there", session_id="s"):
            captured.append((chunk, exhausted["done"]))
        joined = "".join(c for c, _ in captured)

        # The audit STILL enforced: only the block message, and only post-drain.
        for word in ("Hello", "world", "answer"):
            assert word not in joined, f"leaked {word!r} — audit was bypassed"
        assert "[Response blocked by audit:" in joined
        assert captured and all(done for _, done in captured)
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "Hello" not in persisted[0]["content"]

    @pytest.mark.asyncio
    async def test_streaming_user_prompt_disable_still_enforces(self):
        """Disabling (not unregistering) the audit from USER_PROMPT is also a
        next-turn transition."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "harmful content"},
        )
        up_cls = _user_prompt_hook_that(
            lambda mgr: mgr.set_hook_enabled("response_audit", False)
        )
        agent.hooks_manager.register(up_cls(agent.hooks_manager))

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        joined = ""
        async for chunk in agent.process_input_streaming("hi there", session_id="s"):
            joined += chunk

        assert "[Response blocked by audit:" in joined
        assert "Hello" not in joined

    @pytest.mark.asyncio
    async def test_nonstream_user_prompt_unregister_still_enforces(self):
        """Non-streaming path: the same turn-start-snapshot guarantee. Drive the
        REAL ``process_input`` → ``_process_input_traced_locked`` so its
        USER_PROMPT_SUBMIT-then-snapshot ordering is exercised end to end."""
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "harmful content"},
        )
        _wire_real_command_llm_path(agent, llm_content=_RAW_LLM_OUTPUT)
        up_cls = _user_prompt_hook_that(
            lambda mgr: mgr.unregister_by_name("response_audit")
        )
        agent.hooks_manager.register(up_cls(agent.hooks_manager))

        result = await agent.process_input("please answer", session_id="s")

        # The audit still fired and DENIED — the raw model output never surfaced.
        assert _RAW_LLM_OUTPUT not in str(result)
        assert "[Response blocked by audit:" in str(result)
        assert not any(
            _RAW_LLM_OUTPUT in (c.get("content") or "") for c in add_convo_calls
        )


# =========================================================================
# #2674 finding 3 (DEFENDED) — PRE_TOOL governance is a SEPARATE trust boundary
# from the POST_RESPONSE assistant-output audit. Approval/permission hooks MUST
# see the model-generated tool args BEFORE the tool executes, so a human (or a
# policy) can approve/deny the call. A POST_RESPONSE audit reviews the final
# NARRATION and runs long after tools have executed; it cannot — and must not —
# retroactively withhold the tool args the governance gate needs. This contract
# test pins that boundary: even with a strict POST_RESPONSE audit registered,
# the PRE_TOOL_USE hook receives the full, un-redacted tool arguments.
# =========================================================================


class TestFinding3PreToolBoundaryDefended:
    @pytest.mark.asyncio
    async def test_pre_tool_hook_sees_full_args_under_strict_audit(self):
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput

        add_convo = []
        # A strict POST_RESPONSE audit is active for the turn...
        agent, _hook = _make_streaming_audit_agent(
            add_convo, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )

        seen = {}

        class _PreToolCapture(Hook):
            def __init__(self):
                super().__init__(
                    name="pre_tool_capture",
                    events=[HookEvent.PRE_TOOL_USE], priority=1,
                )

            async def execute(self, input):
                # Governance input: the proposed tool call, args and all.
                seen["tool_name"] = input.tool_name
                seen["tool_input"] = input.tool_input
                return HookOutput.allow()

        agent.hooks_manager.register(_PreToolCapture())
        agent._execute_tool_with_hooks = (
            KestrelAgent._execute_tool_with_hooks.__get__(agent)
        )

        async def _execute_fn(post_hook_args):
            return {"success": True, "data": "ok"}

        result = await agent._execute_tool_with_hooks(
            tool_name="send_email",
            feature_name="EmailFeature",
            args={"to": "user@example.com", "body": "governance must see this"},
            session_id="s",
            execute_fn=_execute_fn,
        )

        assert result["success"] is True
        # The PRE_TOOL governance hook saw the FULL model-generated args — the
        # POST_RESPONSE audit neither gates nor redacts this separate boundary.
        assert seen["tool_name"] == "send_email"
        assert seen["tool_input"] == {
            "to": "user@example.com", "body": "governance must see this",
        }


class _ModeFlippingUserPromptHook:
    """A USER_PROMPT_SUBMIT hook that flips an already-registered POST_RESPONSE
    audit hook's ``mode`` mid-turn (#2674 finding 1).

    Models a USER_PROMPT hook (or a tool that runs during prompt submission)
    that calls ``ResponseAuditFeature.enable_audit("warn")`` AFTER the turn-start
    snapshot is captured but BEFORE ``buffer_audit`` is computed. Not a
    ``Hook`` subclass on purpose — the harness's real ``HooksManager`` only needs
    ``events`` / ``enabled`` / ``priority`` / ``matches`` / ``execute`` to run it.
    """

    def __init__(self, audit_hook, new_mode="warn"):
        from kestrel_sdk.hooks.base import HookEvent
        self.name = "mode_flipper"
        self.events = [HookEvent.USER_PROMPT_SUBMIT]
        self.enabled = True
        self.priority = 10
        self.timeout = 5.0
        self._audit_hook = audit_hook
        self._new_mode = new_mode

    def matches(self, tool_name):
        return True

    async def execute(self, hook_input):
        from kestrel_sdk.hooks.base import HookOutput
        # The strict→warn flip lands here, after _snapshot_post_response_hooks
        # captured (mode=strict, enforcing=True) and before buffer_audit reads it.
        self._audit_hook.mode = self._new_mode
        return HookOutput.allow()


class TestFinding1TurnStartModeRace:
    """#2674 finding 1: the withhold (buffer_audit) decision must derive from the
    enforcement state captured at TRUE turn start, not a live hook re-read after
    USER_PROMPT_SUBMIT. A USER_PROMPT hook that flips strict→warn between the
    snapshot and the buffer decision must NOT open the fail-open split (raw bytes
    streamed live, strict block persisted)."""

    @pytest.mark.asyncio
    async def test_user_prompt_strict_to_warn_flip_still_buffers_this_turn(self):
        add_convo_calls = []
        agent, hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 3, "reasoning": "unsafe"},
        )
        # A USER_PROMPT hook flips the audit strict→warn mid-turn-start.
        agent.hooks_manager.register(_ModeFlippingUserPromptHook(hook))

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = []
        async for chunk in agent.process_input_streaming("hi there", session_id="s-race"):
            captured.append((chunk, exhausted["done"]))
        joined = "".join(c for c, _ in captured)

        # The flip happened, but buffer_audit was pinned to the turn-start strict
        # enforcement: NO raw assistant byte streamed before the verdict.
        for word in ("Hello", "world", "answer"):
            assert word not in joined, f"leaked {word!r} — buffer decision raced the flip"
        # Every released chunk landed only after the upstream stream fully drained
        # (i.e. it was withheld, not streamed incrementally).
        assert captured, "no output at all"
        assert all(drained for _c, drained in captured), (
            "a chunk was released before the stream drained — turn was not buffered"
        )
        # The pinned STRICT audit governed this turn: only the block message is
        # released AND persisted — never a raw-streamed / block-persisted split.
        assert "[Response blocked by audit:" in joined
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        for word in ("Hello", "world", "answer"):
            assert word not in persisted[0]["content"]
        # The flip is durable and takes effect NEXT turn: the hook is warn now.
        assert hook.mode == "warn"


# =========================================================================
# #2674 P0-1 — the ENFORCEMENT captured in the turn-start snapshot is
# AUTHORITATIVE for the completion audit: the returned verdict's ``enforcing``
# flag, the gate/observer partition, AND the manager's crash/timeout fail-closed
# resolution. A generic (mode-less) POST_RESPONSE hook whose ``fail_closed``
# flips mid-turn cannot desync the completion audit from the buffering decision
# (both derive from the same snapshot). Fire-level + end-to-end streaming.
# =========================================================================


class _GenericFlipGate(Hook):
    """A GENERIC (mode-less) POST_RESPONSE gate whose ``fail_closed`` can flip
    after the turn-start snapshot. Unlike ``ResponseAuditHook`` its enforcement
    is a plain mutable attribute — not a mode-derived property — so
    ``_pinned_hook_modes`` (which pins ``mode``) cannot protect it. Scriptable to
    ALLOW / raise / hang so the captured-enforcement contract can be exercised in
    every failure mode (#2674 P0-1)."""

    def __init__(self, *, fail_closed=True, behavior="allow", priority=50,
                 timeout=0.05):
        super().__init__(
            name="generic_flip_gate", events=[HookEvent.POST_RESPONSE],
            priority=priority, timeout=timeout,
        )
        self.fail_closed = fail_closed
        self.behavior = behavior
        self.executed = 0

    async def execute(self, input):
        self.executed += 1
        if self.behavior == "flip_then_raise":
            # The live re-read a naive crash handler would take now says
            # "advisory" — but the turn was buffered off the captured True.
            self.fail_closed = False
            raise RuntimeError("generic gate backend exploded")
        if self.behavior == "raise":
            raise RuntimeError("generic gate backend exploded")
        if self.behavior == "hang":
            import asyncio as _asyncio
            await _asyncio.sleep(5.0)
        return HookOutput.allow()


class TestP0CapturedEnforcementAuthoritative:
    """#2674 P0-1: captured turn-start enforcement drives the verdict, the
    partition, and the fail-closed crash/timeout handling — never a live re-read
    that a mid-turn ``fail_closed`` flip could have moved."""

    RAW = "The raw unaudited assistant answer with a SECRET detail."

    def _agent(self):
        agent, _none = _make_streaming_audit_agent(
            [], mode="strict", register_hook=False,
        )
        return agent

    @pytest.mark.asyncio
    async def test_fire_captured_gate_crash_fails_closed_despite_flip(self):
        """Captured enforcing=True, ``fail_closed`` flipped False BEFORE the fire,
        then the gate CRASHES. Both the partition (captured → gate) and the crash
        handler (captured → DENY) use the snapshot value, so the fire fails closed
        and exposes none of the raw text."""
        agent = self._agent()
        gate = _GenericFlipGate(fail_closed=True, behavior="raise")
        agent.hooks_manager.register(gate)
        snap = _snap(agent.hooks_manager)  # captures enforcing=True
        gate.fail_closed = False           # flip AFTER capture, BEFORE the fire

        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=snap,
        )
        assert result.denied is True
        assert result.enforcing is True
        assert self.RAW not in str(result)
        assert "SECRET" not in str(result)
        assert "[Response blocked by audit:" in str(result)
        assert gate.executed == 1

    @pytest.mark.asyncio
    async def test_fire_captured_gate_timeout_fails_closed_despite_flip(self):
        """Same, but the gate HANGS past its timeout: the captured override still
        fails it closed."""
        agent = self._agent()
        gate = _GenericFlipGate(fail_closed=True, behavior="hang", timeout=0.05)
        agent.hooks_manager.register(gate)
        snap = _snap(agent.hooks_manager)
        gate.fail_closed = False

        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=snap,
        )
        assert result.denied is True
        assert result.enforcing is True
        assert self.RAW not in str(result)

    @pytest.mark.asyncio
    async def test_fire_captured_nonenforcing_stays_advisory_despite_flip(self):
        """Inverse transition: captured enforcing=False, ``fail_closed`` flipped
        True before the fire, then the (now-observer) hook crashes. The captured
        value keeps it advisory → the crash is skipped, the verdict is a
        non-enforcing ALLOW passthrough, and ``enforcing`` is False (so the
        streaming wrapper would NOT have withheld this turn's parts)."""
        agent = self._agent()
        gate = _GenericFlipGate(fail_closed=False, behavior="raise")
        agent.hooks_manager.register(gate)
        snap = _snap(agent.hooks_manager)  # captures enforcing=False
        gate.fail_closed = True            # flip to enforcing AFTER capture

        result = await agent._fire_post_response_hook(
            self.RAW, "s1", hook_snapshot=snap,
        )
        assert result.denied is False
        assert result.enforcing is False
        assert str(result) == self.RAW

    @pytest.mark.asyncio
    async def test_streaming_end_to_end_flip_then_crash_fails_closed(self):
        """End-to-end buffering proof: a generic gate enabled at turn start
        buffers the turn (captured enforcing=True). Its ``execute`` flips
        ``fail_closed`` False and CRASHES at completion — the exact P0-1 repro.
        The turn must fail CLOSED: only the block message is released AND
        persisted, no raw byte, and only after the stream drained."""
        add_convo_calls = []
        agent, _none = _make_streaming_audit_agent(
            add_convo_calls, mode="strict", register_hook=False,
        )
        gate = _GenericFlipGate(fail_closed=True, behavior="flip_then_raise")
        agent.hooks_manager.register(gate)

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            for w in _LONG_TEXT_CHUNKS:
                yield w
            yield LLMResponse(content="", tool_calls=[])
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream

        captured = []
        async for chunk in agent.process_input_streaming(
            "hi there", session_id="s-p0-flip",
        ):
            captured.append((chunk, exhausted["done"]))
        joined = "".join(c for c, _ in captured)

        # No raw byte streamed live; the turn was buffered off the captured flag.
        for word in ("Hello", "world", "answer"):
            assert word not in joined, f"leaked {word!r} — fell open on the flip"
        assert "[Response blocked by audit:" in joined
        assert captured and all(done for _, done in captured)
        # Persistence stores only the block message.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert "[Response blocked by audit:" in persisted[0]["content"]
        assert "Hello" not in persisted[0]["content"]
        # The gate actually flipped its live flag mid-execute (the drift a naive
        # live re-read would have followed into a fail-open ALLOW).
        assert gate.fail_closed is False


class TestP0CancellationDuringAudit:
    """#2674 P0-2: cancellation that lands WHILE the strict audit provider call
    is in flight (then the audit returns ALLOW) must NOT persist the withheld,
    now-cancelled prose. Every strict completion branch rechecks cancellation
    AFTER the audit await and persists exactly ONE empty cancelled row, releasing
    nothing and firing neither the STOP hook nor the memory pipeline."""

    @pytest.mark.asyncio
    async def test_no_tool_cancel_during_audit_persists_empty_cancelled_row(self):
        marker = "SECRET_NO_TOOL_ALLOWED_PROSE_cancelled_mid_audit"
        add_convo_calls = []
        cancel_state = {"cancelled": False}

        async def _cancel_during_audit(response_text, **kwargs):
            # The user hits stop WHILE the audit provider call is in flight; the
            # audit then resolves ALLOW (risk 1). Nothing else flipped the flag.
            cancel_state["cancelled"] = True
            return {"risk_level": 1, "reasoning": "clean"}

        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_side_effect=_cancel_during_audit,
        )
        # Make the memory pipeline + STOP fire OBSERVABLE so "did not run" is a
        # real assertion (not merely "was None / privacy-blocked").
        agent._post_response_pipeline = AsyncMock()
        agent._privacy_blocks_background_memory = MagicMock(return_value=False)
        stop_sink = []
        agent.hooks_manager.register(_RecordingStopHook(stop_sink))
        agent.is_request_cancelled = MagicMock(
            side_effect=lambda rid: cancel_state["cancelled"]
        )

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            yield marker + " a full answer long enough to reach the audit gate."
            yield LLMResponse(content="", tool_calls=[])

        agent.llm_service.stream_with_tool_detection = mock_stream

        yielded = []
        async for chunk in agent.process_input_streaming(
            "go", session_id="s-nt-mid", request_id="req-nt-mid",
        ):
            yielded.append(chunk)
        joined = "".join(yielded)

        # No client marker: nothing streamed live (buffered) and the release was
        # discarded on the post-audit cancel recheck.
        assert joined == ""
        assert marker not in joined
        # Exactly one persisted row: EMPTY content, cancelled marker, NO side
        # metadata (the ALLOWed-but-withheld prose is fully discarded).
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert persisted[0]["content"] == ""
        meta = persisted[0].get("metadata") or {}
        assert meta.get("cancelled") is True
        for leaky in ("parts", "tool_events", "tool_results", "pre_tool_reasoning"):
            assert leaky not in meta
        import json as _json
        assert marker not in _json.dumps(persisted[0], default=str)
        # Neither the STOP hook nor the memory pipeline ran over the discarded turn.
        assert stop_sink == []
        agent._post_response_pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_cancel_during_audit_persists_empty_cancelled_row(self):
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

        prose_marker = "SECRET_POST_TOOL_ALLOWED_SYNTHESIS_cancelled_mid_audit"
        tool_secret = "TOOL_RESULT_SECRET_should_never_persist"
        add_convo_calls = []
        cancel_state = {"cancelled": False}

        async def _cancel_during_audit(response_text, **kwargs):
            cancel_state["cancelled"] = True
            return {"risk_level": 1, "reasoning": "clean"}

        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_side_effect=_cancel_during_audit,
        )
        agent._post_response_pipeline = AsyncMock()
        agent._privacy_blocks_background_memory = MagicMock(return_value=False)
        stop_sink = []
        agent.hooks_manager.register(_RecordingStopHook(stop_sink))
        agent.is_request_cancelled = MagicMock(
            side_effect=lambda rid: cancel_state["cancelled"]
        )

        async def mock_stream(**kwargs):
            yield "Working. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        async def mock_orchestrator(*, tool_results=None, **kwargs):
            if tool_results is not None:
                tool_results.append({
                    "tool_call_id": "tc1", "name": "todo_add",
                    "arguments": {}, "result": {"secret": tool_secret},
                })
            yield prose_marker + " reviewed-looking synthesis, long enough to audit."

        agent.llm_service.stream_with_tool_detection = mock_stream
        agent._handle_orchestrator_response_streaming = mock_orchestrator

        yielded = []
        async for chunk in agent.process_input_streaming(
            "go", session_id="s-tool-mid", request_id="req-tool-mid",
        ):
            yielded.append(chunk)
        joined = "".join(yielded)

        # Nothing reached the client on any channel.
        assert prose_marker not in joined
        assert tool_secret not in joined
        assert "[Response blocked by audit:" not in joined
        # One empty cancelled row; the withheld synthesis AND the raw tool_results
        # side channel are discarded.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert persisted[0]["content"] == ""
        meta = persisted[0].get("metadata") or {}
        assert meta.get("cancelled") is True
        for leaky in ("parts", "tool_events", "tool_results", "pre_tool_reasoning"):
            assert leaky not in meta
        import json as _json
        blob = _json.dumps(persisted[0], default=str)
        assert prose_marker not in blob and tool_secret not in blob
        # No STOP / memory pipeline over the discarded content.
        assert stop_sink == []
        agent._post_response_pipeline.assert_not_called()


class TestFinding2StrictCancellationDiscardsWithheldProse:
    """#2674 finding 2: a strict (buffered) turn cancelled BEFORE its reviewed
    release must discard ALL withheld prose / parts / metadata / tool side
    channels and persist an EMPTY cancelled row — matching the already-safe
    strict cancel-before-dispatch path — for BOTH the no-tool and the post-tool
    continuation cancellation cases."""

    @pytest.mark.asyncio
    async def test_strict_no_tool_cancellation_discards_withheld_prose(self):
        marker = "SECRET_NO_TOOL_WITHHELD_PROSE_must_never_persist"
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        # A STOP subscriber must never see the withheld marker either.
        stop_sink = []
        agent.hooks_manager.register(_RecordingStopHook(stop_sink))

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            from kestrel_sovereign.llm.adapter import LLMResponse
            yield marker + " partial answer that is plenty long to audit "
            yield LLMResponse(content="", tool_calls=[])
            # Stop arrives only after the stream fully drains — so the in-loop
            # cancel check never trips and we reach the no-tool completion path
            # with the withheld partial buffered.
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream
        agent.is_request_cancelled = MagicMock(side_effect=lambda rid: exhausted["done"])

        yielded = []
        async for chunk in agent.process_input_streaming(
            "go", session_id="s-nt-cancel", request_id="req-nt",
        ):
            yielded.append(chunk)
        joined = "".join(yielded)

        # Nothing reached the client (no live output, no post-verdict release).
        assert marker not in joined
        assert "[Response blocked by audit:" not in joined
        # The withheld prose was NOT persisted; an empty cancelled row is stored.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert persisted[0]["content"] == ""
        assert marker not in (persisted[0]["content"] or "")
        meta = persisted[0].get("metadata") or {}
        assert meta.get("cancelled") is True
        # No reload/context side channels survive (pre_tool_reasoning replays into
        # the next turn's context; parts/tool_results render on reload).
        for leaky in ("pre_tool_reasoning", "parts", "tool_events", "tool_results"):
            assert leaky not in meta
        # The STOP hook never fired over the discarded content.
        assert stop_sink == []

    @pytest.mark.asyncio
    async def test_strict_post_tool_continuation_cancellation_discards_withheld_prose(self):
        from kestrel_sdk.llm import ToolCallStarted
        from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

        marker = "SECRET_POST_TOOL_CONTINUATION_PROSE_must_never_persist"
        add_convo_calls = []
        agent, _hook = _make_streaming_audit_agent(
            add_convo_calls, mode="strict",
            audit_response={"risk_level": 1, "reasoning": "clean"},
        )
        stop_sink = []
        agent.hooks_manager.register(_RecordingStopHook(stop_sink))

        exhausted = {"done": False}

        async def mock_stream(**kwargs):
            yield "pre-tool prose. "
            yield ToolCallStarted(index=0, id="tc1", name="todo_add")
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="todo_add", arguments={})],
            )

        async def mock_orchestrator(*, tool_results=None, **kwargs):
            if tool_results is not None:
                tool_results.append({
                    "tool_call_id": "tc1", "name": "todo_add",
                    "arguments": {}, "result": {"status": "ok"},
                })
            yield marker + " continuation synthesis, plenty long to audit."
            # Stop lands after the withheld continuation was buffered but before
            # the reviewed release — the post-tool cancellation case.
            exhausted["done"] = True

        agent.llm_service.stream_with_tool_detection = mock_stream
        agent._handle_orchestrator_response_streaming = mock_orchestrator
        agent.is_request_cancelled = MagicMock(side_effect=lambda rid: exhausted["done"])

        yielded = []
        async for chunk in agent.process_input_streaming(
            "go", session_id="s-pt-cancel", request_id="req-pt",
        ):
            yielded.append(chunk)
        joined = "".join(yielded)

        # No withheld continuation prose reached the client on any channel.
        assert marker not in joined
        assert "[Response blocked by audit:" not in joined
        # Empty cancelled row — the unfinished synthesis is not persisted.
        persisted = [c for c in add_convo_calls if c["role"] == "assistant"]
        assert len(persisted) == 1
        assert persisted[0]["content"] == ""
        assert marker not in (persisted[0]["content"] or "")
        meta = persisted[0].get("metadata") or {}
        assert meta.get("cancelled") is True
        for leaky in ("pre_tool_reasoning", "parts", "tool_events", "tool_results"):
            assert leaky not in meta
        # STOP never fired over the discarded, unreviewed content.
        assert stop_sink == []
