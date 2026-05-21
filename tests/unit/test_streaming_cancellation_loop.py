"""
Issue #1256: the agent loop must honor stop-button cancellation, not
just the HTTP response layer.

Before the fix, ``/api/agent/stop`` only flagged the request in
``_cancelled_requests``; the endpoint generator checked that flag and
yielded a "Request stopped" marker, but the underlying
``stream_with_tool_detection`` async generator kept pulling tokens
through to completion and the orchestrator kept dispatching tools.
This meant tools with side effects (sending messages, hitting external
APIs, writing files) ran AFTER the user clicked stop — invisible to
the user because the SSE was already closed client-side.

These tests pin the new contract:
  1. ``_persist_assistant_turn_safely`` stamps ``metadata.cancelled =
     True`` when the request was cancelled, merging with caller-
     supplied metadata.
  2. The inner LLM streaming loop breaks out when
     ``is_request_cancelled`` flips to True mid-stream.
  3. Cancellation arriving after the LLM call finished but before tool
     dispatch began skips the tool batch entirely (no side effects)
     and persists pre-tool prose with the cancellation marker.
  4. The orchestrator's iteration loop checks cancellation between
     tool batches and aborts cleanly.
"""
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.streaming import StreamingMixin
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Direct unit tests on _persist_assistant_turn_safely's new metadata path.
# ---------------------------------------------------------------------------


def _make_persist_agent(*, cancelled_request_id=None):
    """Mock agent exposing only what ``_persist_assistant_turn_safely``
    needs. ``cancelled_request_id`` configures ``is_request_cancelled``
    to return True for exactly that id."""
    captured = []

    async def capture(role, content, metadata=None, session_id=None):
        captured.append({
            "role": role, "content": content,
            "metadata": metadata, "session_id": session_id,
        })

    agent = MagicMock()
    agent.did = "did:test"
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.add_conversation = capture
    agent.observability_store = MagicMock()
    agent.observability_store.log_metric = AsyncMock()

    def is_cancelled(rid=None):
        return rid is not None and rid == cancelled_request_id
    agent.is_request_cancelled = is_cancelled

    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    agent._captured = captured
    return agent


@pytest.mark.asyncio
async def test_persist_marks_cancelled_when_request_was_stopped():
    """When ``is_request_cancelled(request_id)`` is True at persist
    time, the row is stored with ``metadata.cancelled = True`` so the
    next turn's history loader can distinguish a user-aborted partial
    from a normally-completed turn."""
    agent = _make_persist_agent(cancelled_request_id="req-stopped")

    await agent._persist_assistant_turn_safely(
        "partial answer",
        metadata=None,
        session_id="s1",
        request_id="req-stopped",
    )

    assert len(agent._captured) == 1
    row = agent._captured[0]
    assert row["role"] == "assistant"
    assert row["content"] == "partial answer"
    assert row["metadata"] == {"cancelled": True}


@pytest.mark.asyncio
async def test_persist_merges_cancelled_with_caller_metadata():
    """The cancellation marker must not clobber tool_events or any
    other caller-supplied metadata — the helper merges rather than
    replaces."""
    agent = _make_persist_agent(cancelled_request_id="req-stopped")

    await agent._persist_assistant_turn_safely(
        "text",
        metadata={"tool_events": [{"type": "start", "tool": "github"}]},
        session_id="s1",
        request_id="req-stopped",
    )

    md = agent._captured[0]["metadata"]
    assert md["cancelled"] is True
    assert md["tool_events"] == [{"type": "start", "tool": "github"}]


@pytest.mark.asyncio
async def test_persist_no_marker_when_not_cancelled():
    """Control case: a normally-completed turn must not carry a stale
    ``cancelled`` marker just because we passed ``request_id``."""
    agent = _make_persist_agent(cancelled_request_id="OTHER-REQ")

    await agent._persist_assistant_turn_safely(
        "complete answer",
        metadata=None,
        session_id="s1",
        request_id="req-active",
    )

    assert agent._captured[0]["metadata"] is None


@pytest.mark.asyncio
async def test_persist_no_marker_when_request_id_not_provided():
    """Callers that don't pass ``request_id`` (non-streaming flows,
    legacy callers, tests) must see today's behavior — no implicit
    cancellation lookup against ``_current_request_id``."""
    agent = _make_persist_agent(cancelled_request_id="anything")
    # Even if a legacy global is set, with request_id=None the helper
    # MUST NOT consult it.
    agent._current_request_id = "anything"

    await agent._persist_assistant_turn_safely(
        "answer", metadata={"k": "v"}, session_id=None,
        # request_id intentionally omitted
    )

    # metadata passes through unmodified.
    assert agent._captured[0]["metadata"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Integration tests on the full streaming flow.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _passthrough():
    yield


def _build_mock_agent(*, cancel_on_call: int = None):
    """Mock agent for ``process_input_streaming``. ``cancel_on_call``
    makes ``is_request_cancelled`` return True starting from the Nth
    call (1-indexed). None means never cancel."""
    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "did:test"
    agent.extension = None
    agent._cached_features_prompt = ""

    state = {"calls": 0}

    def is_cancelled(rid=None):
        state["calls"] += 1
        if cancel_on_call is None:
            return False
        return state["calls"] >= cancel_on_call
    agent.is_request_cancelled = is_cancelled
    agent._is_cancelled_state = state

    agent._maybe_audit = AsyncMock()
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = None
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"
    agent._current_request_id = None
    agent.emit_event = AsyncMock()

    ctx = MagicMock()
    ctx.system_prompt = "system"
    ctx.dynamic_user_context = ""
    ctx.messages = []
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=ctx)

    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()

    agent.process_input_streaming = (
        StreamingMixin.process_input_streaming.__get__(agent)
    )
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    agent._emit_revising_event = (
        StreamingMixin._emit_revising_event.__get__(agent)
    )
    agent._fire_post_response_hook = (
        StreamingMixin._fire_post_response_hook.__get__(agent)
    )

    # Tool path stubbed; tests that exercise tools override it.
    async def post_tool(*args, **kwargs):
        yield "should not run"
    agent._handle_orchestrator_response_streaming = post_tool
    return agent


@pytest.mark.asyncio
async def test_stream_loop_breaks_when_cancel_arrives_mid_stream():
    """The inner ``async for item in stream_with_tool_detection`` loop
    must break the moment ``is_request_cancelled`` returns True.
    Chunks that follow the cancellation point must NOT reach the
    client, and the partial assistant turn must be persisted with the
    cancellation marker in metadata."""
    # Cancellation flips True on the 3rd is_request_cancelled call.
    # Loop checks at top of each iteration — so iter 1, iter 2 produce
    # output; iter 3 detects cancel and breaks before processing.
    agent = _build_mock_agent(cancel_on_call=3)

    async def stream(**kw):
        yield "first "
        yield "second "
        yield "should-not-reach-client"
        # Tool detection would normally yield LLMResponse here; we
        # don't because the cancel breaks first.
    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = stream

    chunks = []
    async for chunk in agent.process_input_streaming(
        "hello", request_id="req-1"
    ):
        chunks.append(chunk)

    text = "".join(chunks)
    assert "first " in text
    assert "second " in text
    assert "should-not-reach-client" not in text

    # Partial turn persisted, with cancellation marker.
    agent.privacy_agent.add_conversation.assert_awaited()
    # Find the assistant-role call (the user turn is also persisted).
    assistant_calls = [
        c for c in agent.privacy_agent.add_conversation.await_args_list
        if c.args and c.args[0] == "assistant"
    ]
    assert len(assistant_calls) == 1
    call = assistant_calls[0]
    persisted_text = call.args[1]
    assert "first " in persisted_text
    assert "second " in persisted_text
    assert "should-not-reach-client" not in persisted_text
    assert call.kwargs.get("metadata") == {"cancelled": True}


@pytest.mark.asyncio
async def test_cancel_between_llm_and_tool_dispatch_skips_tools():
    """If the user hits stop AFTER the LLM finished but BEFORE the
    orchestrator dispatched the first tool, tools with side effects
    must NOT run. The pre-tool prose persists with the cancellation
    marker. This is the original surprising-side-effect bug class —
    when the SSE closes client-side but the server keeps obeying the
    LLM's tool_use intent, message/email/file tools fire ghost
    actions."""
    # cancel_on_call: 1st call is in the stream loop (returns False —
    # we want the stream to complete). 2nd call is the new branch we
    # added before tool dispatch. Set cancel to fire on the 2nd call.
    agent = _build_mock_agent(cancel_on_call=2)

    async def stream(**kw):
        yield "pre-tool prose "
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="send_message", arguments={})],
        )
    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = stream

    # If the orchestrator were called we'd see "TOOL RAN" in output;
    # the test asserts it isn't.
    orchestrator_called = {"value": False}

    async def post_tool(*args, **kwargs):
        orchestrator_called["value"] = True
        yield "TOOL RAN — this is the regression"
    agent._handle_orchestrator_response_streaming = post_tool

    chunks = []
    async for chunk in agent.process_input_streaming(
        "do a thing", request_id="req-2"
    ):
        chunks.append(chunk)

    assert orchestrator_called["value"] is False, (
        "tool dispatch must be skipped when cancellation arrives "
        "between LLM end and orchestrator start"
    )
    assert "TOOL RAN" not in "".join(chunks)

    assistant_calls = [
        c for c in agent.privacy_agent.add_conversation.await_args_list
        if c.args and c.args[0] == "assistant"
    ]
    assert len(assistant_calls) == 1
    call = assistant_calls[0]
    assert "pre-tool prose" in call.args[1]
    assert call.kwargs.get("metadata") == {"cancelled": True}


@pytest.mark.asyncio
async def test_no_cancel_path_unaffected_for_normal_completion():
    """Control: a turn that never gets cancelled persists with no
    ``cancelled`` marker in metadata. Guards against the marker leaking
    onto normally-completed turns."""
    agent = _build_mock_agent(cancel_on_call=None)

    async def stream(**kw):
        yield "complete "
        yield "answer"
    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = stream

    chunks = []
    async for chunk in agent.process_input_streaming(
        "hi", request_id="req-3"
    ):
        chunks.append(chunk)

    assert "".join(chunks).strip() == "complete answer"

    assistant_calls = [
        c for c in agent.privacy_agent.add_conversation.await_args_list
        if c.args and c.args[0] == "assistant"
    ]
    assert len(assistant_calls) == 1
    md = assistant_calls[0].kwargs.get("metadata")
    # No cancelled key when not cancelled.
    if md is not None:
        assert "cancelled" not in md


# ---------------------------------------------------------------------------
# Orchestrator iteration loop: cancellation between tool batches.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_loop_returns_early_when_cancelled_between_iterations():
    """``_handle_orchestrator_response_streaming`` runs a
    ``for iteration in range(max_iterations)`` loop where each iteration
    runs a tool batch then a follow-up LLM call. The cancel check at
    the top of each iteration (and after the batch) means an in-flight
    batch finishes cleanly, but the next iteration is skipped — no
    runaway tool loops after the user hits stop."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    state = {"calls": 0}

    def is_cancelled(rid=None):
        # Return False during the first iteration's batch; True
        # afterward so the next-iteration cancel check fires.
        state["calls"] += 1
        return state["calls"] > 1
    state["is_cancelled"] = is_cancelled

    mock_feature = MagicMock()
    mock_feature.tool_name = "test_tool"
    mock_feature.name = "test_feature"
    mock_feature.execute_as_subagent = AsyncMock(
        return_value={"success": True, "data": "result"}
    )
    mock_feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {"name": "test_tool", "description": "", "parameters": {}},
    }

    mock_agent = MagicMock()
    mock_agent.did = "did:test"
    mock_agent.features = {"test_feature": mock_feature}
    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="e1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()
    mock_agent._direct_tools = {}
    mock_agent._tool_to_feature = {}
    from kestrel_sovereign.hooks import HooksManager
    mock_agent.hooks_manager = HooksManager()
    mock_agent.is_request_cancelled = is_cancelled
    mock_agent._explored_features = {}
    mock_agent._direct_tool_defs = []
    mock_agent._register_explored_feature_tools = MagicMock()

    # llm_service: stream_with_tool_detection yields ANOTHER tool call
    # (forcing iteration 2), so we can verify cancellation stops the
    # loop before iteration 2 runs.
    second_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc2", name="test_tool", arguments={})],
    )

    async def streamed_followup(**_kw):
        yield second_response

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = streamed_followup

    for method_name in (
        "_handle_orchestrator_response_streaming",
        "_execute_tool_with_hooks", "_execute_tool_batch",
        "_partition_tool_calls", "_dispatch_tool_call",
        "_dispatch_feature_tool", "_dispatch_direct_tool",
        "_get_denied_tools", "_handle_feature_error",
        "_prune_orchestrator_messages", "_build_all_tools",
        "_build_feature_tools", "_visible_features_by_tool_name",
        "_visible_known_tool_names", "_hidden_context_features",
        "_hidden_context_tools", "_feature_hidden_from_context",
        "_direct_tool_hidden_from_context",
    ):
        setattr(
            mock_agent, method_name,
            getattr(KestrelAgent, method_name).__get__(mock_agent),
        )
    mock_agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg

    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = []
    async for chunk in mock_agent._handle_orchestrator_response_streaming(
        response=first_response,
        feature_tools=[],
        system_prompt="sys",
        force_local_only=False,
        effective_model="m",
        user_message="hi",
        request_id="req-cancel-between",
    ):
        chunks.append(chunk)

    # First iteration's tool ran once. Second iteration's tool MUST
    # NOT have run — the cancel check after the batch stopped it.
    assert mock_feature.execute_as_subagent.await_count == 1, (
        "the second iteration's tool batch must be skipped when the "
        "user cancels between iterations"
    )


@pytest.mark.asyncio
async def test_orchestrator_loop_runs_normally_without_request_id():
    """Without ``request_id``, the cancel predicate inside the
    orchestrator loop is a no-op. Existing callers that don't thread
    request_id (non-streaming paths, tests) must see the pre-#1256
    behavior."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    mock_feature = MagicMock()
    mock_feature.tool_name = "test_tool"
    mock_feature.name = "test_feature"
    mock_feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}
    )
    mock_feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {"name": "test_tool", "description": "", "parameters": {}},
    }

    mock_agent = MagicMock()
    mock_agent.did = "did:test"
    mock_agent.features = {"test_feature": mock_feature}
    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="e1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()
    mock_agent._direct_tools = {}
    mock_agent._tool_to_feature = {}
    from kestrel_sovereign.hooks import HooksManager
    mock_agent.hooks_manager = HooksManager()
    # If this is consulted, the test catches it via cancel side-effects.
    mock_agent.is_request_cancelled = MagicMock(return_value=True)
    mock_agent._explored_features = {}
    mock_agent._direct_tool_defs = []
    mock_agent._register_explored_feature_tools = MagicMock()

    mock_agent.llm_service = MagicMock()
    final_response = LLMResponse(content="final", tool_calls=[])

    async def streamed_followup(**_kw):
        # stream_with_tool_detection: yield text chunks, then a final
        # LLMResponse with detected tool_calls (empty here = no more
        # tool calls, terminating turn).
        yield "final"
        yield final_response

    mock_agent.llm_service.stream_with_tool_detection = streamed_followup

    for method_name in (
        "_handle_orchestrator_response_streaming",
        "_execute_tool_with_hooks", "_execute_tool_batch",
        "_partition_tool_calls", "_dispatch_tool_call",
        "_dispatch_feature_tool", "_dispatch_direct_tool",
        "_get_denied_tools", "_handle_feature_error",
        "_prune_orchestrator_messages", "_build_all_tools",
        "_build_feature_tools", "_visible_features_by_tool_name",
        "_visible_known_tool_names", "_hidden_context_features",
        "_hidden_context_tools", "_feature_hidden_from_context",
        "_direct_tool_hidden_from_context",
    ):
        setattr(
            mock_agent, method_name,
            getattr(KestrelAgent, method_name).__get__(mock_agent),
        )
    mock_agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg

    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = []
    async for chunk in mock_agent._handle_orchestrator_response_streaming(
        response=first_response,
        feature_tools=[],
        system_prompt="sys",
        force_local_only=False,
        effective_model="m",
        user_message="hi",
        # request_id omitted — cancel must not fire even with
        # is_request_cancelled stubbed to True.
    ):
        chunks.append(chunk)

    # Tool ran and synthesis text reached us — cancel was correctly
    # ignored because no request_id was threaded.
    assert mock_feature.execute_as_subagent.await_count == 1
    assert "final" in "".join(chunks)


@pytest.mark.asyncio
async def test_orchestrator_loop_streams_followup_text_in_real_time():
    """The fix that prompted this: Emma's chat showed
    "🔧 Calling mesh_inbox..." then silence. Root cause was the
    multi-iteration loop's follow-up LLM call used the non-streaming
    ``generate_with_messages`` (buffered entire response before yielding
    a single byte) followed by a second streaming call. A slow / stuck /
    silent upstream produced dead air for the whole buffered round-trip.

    After the fix, ``stream_with_tool_detection`` yields text chunks as
    they arrive AND a final LLMResponse with detected tool_calls. This
    test asserts the text chunks flow OUT of the orchestrator loop in
    the same order they came IN — proving no buffering happens at the
    iteration boundary."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    mock_feature = MagicMock()
    mock_feature.tool_name = "test_tool"
    mock_feature.name = "test_feature"
    mock_feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}
    )
    mock_feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {"name": "test_tool", "description": "", "parameters": {}},
    }

    mock_agent = MagicMock()
    mock_agent.did = "did:test"
    mock_agent.features = {"test_feature": mock_feature}
    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="e1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()
    mock_agent._direct_tools = {}
    mock_agent._tool_to_feature = {}
    from kestrel_sovereign.hooks import HooksManager
    mock_agent.hooks_manager = HooksManager()
    mock_agent.is_request_cancelled = MagicMock(return_value=False)
    mock_agent._explored_features = {}
    mock_agent._direct_tool_defs = []
    mock_agent._register_explored_feature_tools = MagicMock()

    final_response = LLMResponse(
        content="Looks like the mailbox is empty", tool_calls=[],
    )

    async def streamed_followup(**_kw):
        # The follow-up is the LLM's interpretation of the tool result.
        # Each yield is a chunk; the test asserts they reach the chat
        # in stream order rather than being buffered.
        for word in ["Looks", " ", "like", " ", "the", " ", "mailbox", " ",
                     "is", " ", "empty"]:
            yield word
        yield final_response

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = streamed_followup

    for method_name in (
        "_handle_orchestrator_response_streaming",
        "_execute_tool_with_hooks", "_execute_tool_batch",
        "_partition_tool_calls", "_dispatch_tool_call",
        "_dispatch_feature_tool", "_dispatch_direct_tool",
        "_get_denied_tools", "_handle_feature_error",
        "_prune_orchestrator_messages", "_build_all_tools",
        "_build_feature_tools", "_visible_features_by_tool_name",
        "_visible_known_tool_names", "_hidden_context_features",
        "_hidden_context_tools", "_feature_hidden_from_context",
        "_direct_tool_hidden_from_context",
    ):
        setattr(
            mock_agent, method_name,
            getattr(KestrelAgent, method_name).__get__(mock_agent),
        )
    mock_agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg

    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = []
    async for chunk in mock_agent._handle_orchestrator_response_streaming(
        response=first_response, feature_tools=[],
        system_prompt="sys", force_local_only=False,
        effective_model="m", user_message="check inbox",
    ):
        chunks.append(chunk)

    # The tool ran exactly once (single iteration).
    assert mock_feature.execute_as_subagent.await_count == 1
    # The follow-up's words reach us as individual chunks in order —
    # no buffering at the iteration boundary.
    text = "".join(c for c in chunks if isinstance(c, str))
    assert "Looks like the mailbox is empty" in text
    # And we saw the individual word chunks (not just one buffered
    # blob), confirming the stream wasn't collapsed.
    assert "Looks" in chunks
    assert "empty" in chunks


@pytest.mark.asyncio
async def test_orchestrator_loop_timeout_surfaces_failed_marker(monkeypatch):
    """Per-iteration ``asyncio.timeout`` wraps the follow-up
    ``stream_with_tool_detection`` so a stuck upstream surfaces as a
    visible ❌ failure marker instead of silent dead air. The chat-UI
    parser keys on ``❌ <name> failed`` to render the tool card's
    error state — without this the user saw nothing after the first
    🔧 marker (Emma's bug)."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    import kestrel_sovereign.agent.orchestrator_engine as oe

    # Drop the timeout to a tiny value so the test runs quickly while
    # still exercising the real ``asyncio.timeout`` path.
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", 0.2)

    mock_feature = MagicMock()
    mock_feature.tool_name = "test_tool"
    mock_feature.name = "test_feature"
    mock_feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}
    )
    mock_feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {"name": "test_tool", "description": "", "parameters": {}},
    }

    mock_agent = MagicMock()
    mock_agent.did = "did:test"
    mock_agent.features = {"test_feature": mock_feature}
    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="e1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()
    mock_agent._direct_tools = {}
    mock_agent._tool_to_feature = {}
    from kestrel_sovereign.hooks import HooksManager
    mock_agent.hooks_manager = HooksManager()
    mock_agent.is_request_cancelled = MagicMock(return_value=False)
    mock_agent._explored_features = {}
    mock_agent._direct_tool_defs = []
    mock_agent._register_explored_feature_tools = MagicMock()

    async def stuck_followup(**_kw):
        # Simulates a hung anthropic-style call: yields nothing,
        # never completes. The orchestrator's timeout must trip.
        await asyncio.sleep(10)
        yield "this never arrives"

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = stuck_followup

    for method_name in (
        "_handle_orchestrator_response_streaming",
        "_execute_tool_with_hooks", "_execute_tool_batch",
        "_partition_tool_calls", "_dispatch_tool_call",
        "_dispatch_feature_tool", "_dispatch_direct_tool",
        "_get_denied_tools", "_handle_feature_error",
        "_prune_orchestrator_messages", "_build_all_tools",
        "_build_feature_tools", "_visible_features_by_tool_name",
        "_visible_known_tool_names", "_hidden_context_features",
        "_hidden_context_tools", "_feature_hidden_from_context",
        "_direct_tool_hidden_from_context",
    ):
        setattr(
            mock_agent, method_name,
            getattr(KestrelAgent, method_name).__get__(mock_agent),
        )
    mock_agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg

    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = []
    async for chunk in mock_agent._handle_orchestrator_response_streaming(
        response=first_response, feature_tools=[],
        system_prompt="sys", force_local_only=False,
        effective_model="m", user_message="hi",
    ):
        chunks.append(chunk)

    text = "".join(c for c in chunks if isinstance(c, str))
    # User sees the failure marker — chat UI groups this under the
    # tool card as the error state.
    assert "❌" in text and "timeout" in text, (
        f"expected '❌ llm call failed: timeout' in stream, got: {text!r}"
    )
    # Tool still ran exactly once — only the FOLLOW-UP timed out.
    assert mock_feature.execute_as_subagent.await_count == 1


@pytest.mark.asyncio
async def test_orchestrator_loop_followup_separator_emitted_once_per_turn():
    """The "\\n---\\n" separator is the chat UI's wire-protocol marker
    between tool activity and final synthesis. Pre-rewrite it fired
    exactly once per turn. Post-rewrite the orchestrator must preserve
    that semantic — multiple iterations of the loop yield only ONE
    separator total, not one per iteration."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    mock_feature = MagicMock()
    mock_feature.tool_name = "test_tool"
    mock_feature.name = "test_feature"
    mock_feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}
    )
    mock_feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {"name": "test_tool", "description": "", "parameters": {}},
    }

    mock_agent = MagicMock()
    mock_agent.did = "did:test"
    mock_agent.features = {"test_feature": mock_feature}
    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="e1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()
    mock_agent._direct_tools = {}
    mock_agent._tool_to_feature = {}
    from kestrel_sovereign.hooks import HooksManager
    mock_agent.hooks_manager = HooksManager()
    mock_agent.is_request_cancelled = MagicMock(return_value=False)
    mock_agent._explored_features = {}
    mock_agent._direct_tool_defs = []
    mock_agent._register_explored_feature_tools = MagicMock()

    call_count = {"n": 0}

    async def streamed_followup(**_kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First iteration's follow-up: more tool_calls (forces
            # another iteration).
            yield LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="test_tool", arguments={})],
            )
        else:
            # Second iteration's follow-up: text-only (terminates).
            yield "done"
            yield LLMResponse(content="done", tool_calls=[])

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = streamed_followup

    for method_name in (
        "_handle_orchestrator_response_streaming",
        "_execute_tool_with_hooks", "_execute_tool_batch",
        "_partition_tool_calls", "_dispatch_tool_call",
        "_dispatch_feature_tool", "_dispatch_direct_tool",
        "_get_denied_tools", "_handle_feature_error",
        "_prune_orchestrator_messages", "_build_all_tools",
        "_build_feature_tools", "_visible_features_by_tool_name",
        "_visible_known_tool_names", "_hidden_context_features",
        "_hidden_context_tools", "_feature_hidden_from_context",
        "_direct_tool_hidden_from_context",
    ):
        setattr(
            mock_agent, method_name,
            getattr(KestrelAgent, method_name).__get__(mock_agent),
        )
    mock_agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg

    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = []
    async for chunk in mock_agent._handle_orchestrator_response_streaming(
        response=first_response, feature_tools=[],
        system_prompt="sys", force_local_only=False,
        effective_model="m", user_message="hi",
    ):
        chunks.append(chunk)

    # Two iterations ran (tool ran twice).
    assert mock_feature.execute_as_subagent.await_count == 2
    # But the separator fired exactly once.
    text = "".join(c for c in chunks if isinstance(c, str))
    assert text.count("\n---\n") == 1, (
        f"expected exactly one separator per turn, got {text.count('---')}: {text!r}"
    )
