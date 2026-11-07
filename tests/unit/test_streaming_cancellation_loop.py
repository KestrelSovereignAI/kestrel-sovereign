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
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.streaming import (
    STRICT_AUDIT_CANCELLED_TOOL_BATCH_CHECKPOINT,
    StreamingMixin,
    _STRICT_AUDIT_TOOL_BATCH_CHECKPOINT_METADATA,
)
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.features.base import Feature
from tests.utils.native_streaming import streaming_agent, collect_native_stream
from tests.utils.native_tool_effects import drive_native


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


async def _streaming_agent(native_tool_effect_agent, *, cancel_on_call=None):
    agent = await streaming_agent(native_tool_effect_agent)
    state = {"calls": 0}

    def is_cancelled(request_id=None):
        state["calls"] += 1
        return cancel_on_call is not None and state["calls"] >= cancel_on_call

    agent.is_request_cancelled = is_cancelled
    agent._is_cancelled_state = state

    async def post_tool(*args, **kwargs):
        yield "should not run"

    agent._handle_orchestrator_response_streaming = post_tool
    return agent


class _CancellationFeature(Feature):
    @property
    def tool_name(self):
        return "test_tool"

    @property
    def tool_description(self):
        return "Owned cancellation-loop component"

    async def initialize(self):
        pass


async def _orchestrator_agent(native_tool_effect_agent):
    agent = await streaming_agent(native_tool_effect_agent)
    feature = _CancellationFeature(agent)
    feature.name = "test_feature"
    agent.features[feature.name] = feature
    return agent, feature


async def _native_persist_agent(native_tool_effect_agent):
    captured = []
    agent = await streaming_agent(native_tool_effect_agent, persisted=captured)
    agent._captured = captured
    return agent


@pytest.mark.asyncio
async def test_stream_loop_breaks_when_cancel_arrives_mid_stream(native_tool_effect_agent):
    """The inner ``async for item in stream_with_tool_detection`` loop
    must break the moment ``is_request_cancelled`` returns True.
    Chunks that follow the cancellation point must NOT reach the
    client. The stopped generation is not an answer (#3552): the turn
    persists an empty row with the cancellation marker in metadata,
    not the partial the client saw live."""
    # Cancellation flips True on the 3rd is_request_cancelled call.
    # Loop checks at top of each iteration — so iter 1, iter 2 produce
    # output; iter 3 detects cancel and breaks before processing.
    agent = await _streaming_agent(native_tool_effect_agent, cancel_on_call=3)

    async def stream(**kw):
        yield "first "
        yield "second "
        yield "should-not-reach-client"
        # Tool detection would normally yield LLMResponse here; we
        # don't because the cancel breaks first.
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

    # An empty turn persisted, with cancellation marker.
    agent.privacy_agent.add_conversation.assert_awaited()
    # Find the assistant-role call (the user turn is also persisted).
    assistant_calls = [
        c for c in agent.privacy_agent.add_conversation.await_args_list
        if c.args and c.args[0] == "assistant"
    ]
    assert len(assistant_calls) == 1
    call = assistant_calls[0]
    assert call.args[1] == ""
    assert call.kwargs.get("metadata") == {"cancelled": True}


@pytest.mark.asyncio
async def test_cancel_between_llm_and_tool_dispatch_skips_tools(native_tool_effect_agent):
    """If the user hits stop AFTER the LLM finished but BEFORE the
    orchestrator dispatched the first tool, tools with side effects
    must NOT run. The pre-tool prose persists with the cancellation
    marker. This is the original surprising-side-effect bug class —
    when the SSE closes client-side but the server keeps obeying the
    LLM's tool_use intent, message/email/file tools fire ghost
    actions. The stopped turn records no answer (#3552), so its
    pre-tool prose is not persisted either."""
    # cancel_on_call: 1st call is in the stream loop (returns False —
    # we want the stream to complete). 2nd call is the new branch we
    # added before tool dispatch. Set cancel to fire on the 2nd call.
    agent = await _streaming_agent(native_tool_effect_agent, cancel_on_call=2)

    async def stream(**kw):
        yield "pre-tool prose "
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="send_message", arguments={})],
        )
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
    assert call.args[1] == ""
    assert call.kwargs.get("metadata") == {"cancelled": True}


@pytest.mark.asyncio
async def test_cancel_during_tool_batch_persists_completed_result_before_unwind(native_tool_effect_agent):
    """A completed external effect must cross the history checkpoint on Stop.

    The stopped turn records no answer (#3552): the row is the fixed
    checkpoint of the completed batch, as under an enforcing audit."""

    from kestrel_sovereign.agent.orchestrator_engine import (
        OrchestratorEngineMixin,
    )

    agent = await _streaming_agent(native_tool_effect_agent, cancel_on_call=None)
    cancelled = False
    batch_started = asyncio.Event()
    release_batch = asyncio.Event()
    batch_completed = asyncio.Event()

    agent.is_request_cancelled = lambda _request_id=None: cancelled
    persist_assistant_turn = agent._persist_assistant_turn_safely
    persistence_requirements = []

    async def capture_persistence_requirement(*args, **kwargs):
        persistence_requirements.append(kwargs.get("require_success", False))
        return await persist_assistant_turn(*args, **kwargs)

    agent._persist_assistant_turn_safely = capture_persistence_requirement
    agent._persist_completed_tool_stop_checkpoint = (
        OrchestratorEngineMixin._persist_completed_tool_stop_checkpoint.__get__(agent)
    )
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._known_tool_names = MagicMock(return_value=set())
    agent._handle_orchestrator_response_streaming = (
        OrchestratorEngineMixin._handle_orchestrator_response_streaming.__get__(
            agent
        )
    )
    agent._execute_tool_batch_at_stop_boundary = (
        OrchestratorEngineMixin._execute_tool_batch_at_stop_boundary.__get__(
            agent
        )
    )

    async def execute_batch(*_args, **kwargs):
        batch_started.set()
        await release_batch.wait()
        kwargs["tool_events"].extend(
            [
                {"type": "start", "tool": "send_message"},
                {"type": "complete", "tool": "send_message", "ms": 1},
            ]
        )
        kwargs["tool_results"].append(
            {
                "tool_call_id": "tc1",
                "name": "send_message",
                "arguments": {"text": "sent once"},
                "result": {"success": True},
            }
        )
        batch_completed.set()

    agent._execute_tool_batch = execute_batch

    async def stream(**_kwargs):
        yield LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="send_message",
                    arguments={"text": "sent once"},
                )
            ],
        )

    agent.llm_service.stream_with_tool_detection = stream

    async def consume():
        return [
            chunk
            async for chunk in agent.process_input_streaming(
                "send it", request_id="req-stop-in-batch"
            )
        ]

    turn = asyncio.create_task(consume())
    await batch_started.wait()
    cancelled = True
    turn.cancel()
    await asyncio.sleep(0)
    assert batch_completed.is_set() is False

    release_batch.set()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert batch_completed.is_set() is True
    assistant_calls = [
        call
        for call in agent.privacy_agent.add_conversation.await_args_list
        if call.args and call.args[0] == "assistant"
    ]
    assert len(assistant_calls) == 1
    assert persistence_requirements == [True]
    assert assistant_calls[0].kwargs["metadata"]["cancelled"] is True
    # Cancellation now closes every forwarding layer before unwinding. When
    # that close interrupts a batch's result marker, the real turn owner uses
    # its fixed anti-repeat checkpoint, not an unpublished provider envelope.
    assert assistant_calls[0].args[1] == STRICT_AUDIT_CANCELLED_TOOL_BATCH_CHECKPOINT
    assert assistant_calls[0].kwargs["metadata"] == {
        **_STRICT_AUDIT_TOOL_BATCH_CHECKPOINT_METADATA,
        "cancelled": True,
    }


@pytest.mark.asyncio
async def test_nonstreaming_cancelled_batch_checkpoints_before_unwind(native_tool_effect_agent):
    """The non-streaming orchestrator cannot strand a completed effect."""

    from kestrel_sovereign.agent.orchestrator_engine import (
        OrchestratorEngineMixin,
    )

    agent = await _native_persist_agent(native_tool_effect_agent)
    batch_started = asyncio.Event()
    release_batch = asyncio.Event()
    batch_completed = asyncio.Event()
    required = []
    persist = agent._persist_assistant_turn_safely

    async def capture_required(*args, **kwargs):
        required.append(kwargs.get("require_success", False))
        return await persist(*args, **kwargs)

    agent._persist_assistant_turn_safely = capture_required
    agent._persist_completed_tool_stop_checkpoint = (
        OrchestratorEngineMixin._persist_completed_tool_stop_checkpoint.__get__(
            agent
        )
    )
    agent._execute_tool_batch_at_stop_boundary = (
        OrchestratorEngineMixin._execute_tool_batch_at_stop_boundary.__get__(
            agent
        )
    )
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._known_tool_names = MagicMock(return_value=set())

    async def execute_batch(*_args, **kwargs):
        batch_started.set()
        await release_batch.wait()
        kwargs["tool_results"].append(
            {
                "tool_call_id": "tc-nonstream",
                "name": "send_message",
                "arguments": {"text": "sent once"},
                "result": {"success": True},
            }
        )
        batch_completed.set()

    agent._execute_tool_batch = execute_batch
    response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="tc-nonstream",
                name="send_message",
                arguments={"text": "sent once"},
            )
        ],
    )

    turn = asyncio.create_task(
        drive_native(agent, lambda: OrchestratorEngineMixin._handle_orchestrator_response(
            agent,
            response=response,
            feature_tools=[],
            system_prompt="system",
            force_local_only=False,
            effective_model="test",
            user_message="send it",
            session_id="s-nonstream",
            tool_results=[],
        ))
    )
    await batch_started.wait()
    turn.cancel()
    release_batch.set()

    with pytest.raises(asyncio.CancelledError):
        await turn

    assert batch_completed.is_set()
    assert required == [True]
    assert len(agent._captured) == 1
    assert agent._captured[0]["metadata"] == {
        "tool_batch_checkpoint": {
            "status": "completed",
            "details_withheld": True,
        }
    }


@pytest.mark.asyncio
async def test_nonstreaming_cancel_after_batch_before_synthesis_checkpoints(native_tool_effect_agent):
    """A later provider await remains behind the completed-batch fence."""

    from kestrel_sovereign.agent.orchestrator_engine import (
        OrchestratorEngineMixin,
    )

    agent = await _native_persist_agent(native_tool_effect_agent)
    synthesis_started = asyncio.Event()
    required = []
    persist = agent._persist_assistant_turn_safely

    async def capture_required(*args, **kwargs):
        required.append(kwargs.get("require_success", False))
        return await persist(*args, **kwargs)

    async def execute_batch(*_args, **kwargs):
        kwargs["tool_results"].append(
            {
                "tool_call_id": "tc-late-cancel",
                "name": "send_message",
                "arguments": {"text": "sent once"},
                "result": {"success": True},
            }
        )

    async def wait_for_synthesis(**_kwargs):
        synthesis_started.set()
        await asyncio.Event().wait()

    agent._persist_assistant_turn_safely = capture_required
    agent._persist_completed_tool_stop_checkpoint = (
        OrchestratorEngineMixin._persist_completed_tool_stop_checkpoint.__get__(
            agent
        )
    )
    agent._execute_tool_batch_at_stop_boundary = (
        OrchestratorEngineMixin._execute_tool_batch_at_stop_boundary.__get__(
            agent
        )
    )
    agent._execute_tool_batch = execute_batch
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._known_tool_names = MagicMock(return_value=set())
    agent._build_all_tools = MagicMock(return_value=[])
    agent._prune_orchestrator_messages = MagicMock(
        side_effect=lambda messages, *_args, **_kwargs: messages
    )
    agent._make_inline_tool_executor = MagicMock(return_value=None)
    agent.llm_service.generate_with_messages = AsyncMock(
        side_effect=wait_for_synthesis
    )
    response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="tc-late-cancel",
                name="send_message",
                arguments={"text": "sent once"},
            )
        ],
    )

    turn = asyncio.create_task(
        drive_native(agent, lambda: OrchestratorEngineMixin._handle_orchestrator_response(
            agent,
            response=response,
            feature_tools=[],
            system_prompt="system",
            force_local_only=False,
            effective_model="test",
            user_message="send it",
            session_id="s-late-cancel",
            tool_results=[],
        ))
    )
    await synthesis_started.wait()
    turn.cancel()

    with pytest.raises(asyncio.CancelledError):
        await turn

    assert required == [True]
    assert len(agent._captured) == 1


@pytest.mark.asyncio
async def test_inline_effect_requires_successful_history_persistence(native_tool_effect_agent):
    """Codex inline effects cannot complete behind a best-effort history write."""

    from kestrel_sovereign.llm.adapter import LLMResponse

    agent = await _streaming_agent(native_tool_effect_agent, cancel_on_call=None)

    async def fail_assistant_persistence(role, _content, **_kwargs):
        if role == "assistant":
            raise RuntimeError("history unavailable")

    agent.privacy_agent.add_conversation = AsyncMock(
        side_effect=fail_assistant_persistence
    )
    response = LLMResponse(content="", tool_calls=[])
    response.executed_tool_calls = [
        {
            "id": "inline-effect",
            "name": "send_message",
            "arguments": {"text": "sent once"},
            "result": {"success": True},
        }
    ]

    async def stream(**_kwargs):
        yield "sent"
        yield response

    agent.llm_service.stream_with_tool_detection = stream

    with pytest.raises(RuntimeError, match="history unavailable"):
        async for _chunk in agent.process_input_streaming(
            "send it",
            session_id="inline-session",
            request_id="inline-required-persist",
        ):
            pass


@pytest.mark.asyncio
async def test_no_cancel_path_unaffected_for_normal_completion(native_tool_effect_agent):
    """Control: a turn that never gets cancelled persists with no
    ``cancelled`` marker in metadata. Guards against the marker leaking
    onto normally-completed turns."""
    agent = await _streaming_agent(native_tool_effect_agent, cancel_on_call=None)

    async def stream(**kw):
        yield "complete "
        yield "answer"
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
async def test_orchestrator_loop_returns_early_when_cancelled_between_iterations(native_tool_effect_agent):
    """A real Stop after the first acknowledged batch forbids another batch."""
    from kestrel_sovereign.agent.invocation import current_invocation_id, InvocationCancelledError

    agent, feature = await _orchestrator_agent(native_tool_effect_agent)
    feature.execute_as_subagent = AsyncMock(return_value={"success": True, "data": "result"})
    execute_batch = agent._execute_tool_batch

    async def stop_after_batch(*args, **kwargs):
        result = await execute_batch(*args, **kwargs)
        assert agent.cancel_current_request(current_invocation_id())
        return result

    agent._execute_tool_batch = stop_after_batch

    async def followup(**kwargs):
        yield LLMResponse(content="", tool_calls=[ToolCall(id="tc2", name="test_tool", arguments={})])

    agent.llm_service.stream_with_tool_detection = followup
    first = LLMResponse(content="", tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})])
    with pytest.raises(InvocationCancelledError):
        await collect_native_stream(agent, lambda: agent._handle_orchestrator_response_streaming(
            response=first, feature_tools=[], system_prompt="sys", force_local_only=False,
            effective_model="m", user_message="hi", request_id=current_invocation_id(),
        ))
    assert feature.execute_as_subagent.await_count == 1
    assert await agent._raw_storage.db.fetchone("SELECT state FROM tool_effect_claims") == ("RETURNED_OK",)


@pytest.mark.asyncio
async def test_orchestrator_loop_runs_with_its_assigned_native_request(native_tool_effect_agent):
    """An input without a supplied ID still owns one real native invocation."""

    agent, feature = await _orchestrator_agent(native_tool_effect_agent)
    feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}
    )

    final_response = LLMResponse(content="final", tool_calls=[])

    ids = []
    async def streamed_followup(**_kw):
        from kestrel_sovereign.agent.invocation import current_invocation_id
        ids.append(current_invocation_id())
        # stream_with_tool_detection: yield text chunks, then a final
        # LLMResponse with detected tool_calls (empty here = no more
        # tool calls, terminating turn).
        yield "final"
        yield final_response

    agent.llm_service.stream_with_tool_detection = streamed_followup


    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = await collect_native_stream(agent, lambda: agent._handle_orchestrator_response_streaming(
        response=first_response,
        feature_tools=[],
        system_prompt="sys",
        force_local_only=False,
        effective_model="m",
        user_message="hi",
    ))

    assert ids and ids[0]
    assert feature.execute_as_subagent.await_count == 1
    assert "final" in "".join(chunks)


@pytest.mark.asyncio
async def test_orchestrator_loop_streams_followup_text_in_real_time(native_tool_effect_agent):
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

    agent, feature = await _orchestrator_agent(native_tool_effect_agent)
    feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}
    )

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

    agent.llm_service.stream_with_tool_detection = streamed_followup


    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = await collect_native_stream(agent, lambda: agent._handle_orchestrator_response_streaming(
        response=first_response, feature_tools=[],
        system_prompt="sys", force_local_only=False,
        effective_model="m", user_message="check inbox",
    ))

    # The tool ran exactly once (single iteration).
    assert feature.execute_as_subagent.await_count == 1
    # The follow-up's words reach us as individual chunks in order —
    # no buffering at the iteration boundary.
    text = "".join(c for c in chunks if isinstance(c, str))
    assert "Looks like the mailbox is empty" in text
    # And we saw the individual word chunks (not just one buffered
    # blob), confirming the stream wasn't collapsed.
    assert "Looks" in chunks
    assert "empty" in chunks


async def _timeout_followup(native_tool_effect_agent, monkeypatch):
    """Retain streamed observations even when the native turn raises its timeout."""
    import kestrel_sovereign.agent.orchestrator_engine as oe

    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", 0.2)
    agent, feature = await _orchestrator_agent(native_tool_effect_agent)
    feature.execute_as_subagent = AsyncMock(return_value={"success": True})

    async def stuck_followup(**kwargs):
        await asyncio.sleep(10)
        yield "this never arrives"

    agent.llm_service.stream_with_tool_detection = stuck_followup
    first = LLMResponse(content="", tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})])
    chunks = []
    with pytest.raises(oe.LLMCallInactivityTimeout):
        await collect_native_stream(agent, lambda: agent._handle_orchestrator_response_streaming(
            response=first, feature_tools=[], system_prompt="sys", force_local_only=False,
            effective_model="m", user_message="hi",
        ), chunks=chunks)
    return chunks, feature


@pytest.mark.asyncio
async def test_orchestrator_loop_timeout_surfaces_failed_marker(native_tool_effect_agent, monkeypatch):
    from kestrel_sovereign.agent.streaming import _parse_stream_sentinels

    chunks, feature = await _timeout_followup(native_tool_effect_agent, monkeypatch)
    text = "".join(c for c in chunks if isinstance(c, str))
    _clean, parts, _ = _parse_stream_sentinels(text)
    assert any(p["phase"] == "error" and "timeout" in (p.get("detail") or "") for p in parts), (
        f"expected an error tool sentinel with timeout, got: {text!r}"
    )
    assert feature.execute_as_subagent.await_count == 1


@pytest.mark.asyncio
async def test_orchestrator_loop_timeout_marker_precedes_separator(native_tool_effect_agent, monkeypatch):
    from kestrel_sovereign.agent.streaming import _parse_stream_sentinels

    chunks, _feature = await _timeout_followup(native_tool_effect_agent, monkeypatch)
    text = "".join(c for c in chunks if isinstance(c, str))
    assert "\n---\n" not in text, f"separator must NOT precede timeout marker. Got: {text!r}"
    _clean, parts, _ = _parse_stream_sentinels(text)
    assert any(p["phase"] == "error" and "timeout" in (p.get("detail") or "") for p in parts), (
        f"expected an error tool sentinel with timeout, got: {text!r}"
    )


@pytest.mark.asyncio
async def test_orchestrator_loop_follow_up_tool_call_emits_revise_sentinel(native_tool_effect_agent):
    """Codex P1 on PR #1346: a follow-up iteration's stream can yield
    text BEFORE another ToolCallStarted. That mid-stream tool start
    must fire the same honesty-layer revise sentinel the first-level
    streaming path does, so the chat client can clear any speculative
    pre-tool prose for THIS tool call. Without this, the model's
    "I'll check Y" text stays on screen unverified before Y runs."""
    from kestrel_sdk.llm import ToolCallStarted as TCS

    agent, feature = await _orchestrator_agent(native_tool_effect_agent)
    feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}
    )
    # Stub the SSE emit — we only care that the in-band sentinel
    # makes it into the stream.
    agent._emit_revising_event = AsyncMock()

    async def streamed_followup(**_kw):
        if feature.execute_as_subagent.await_count >= 2:
            yield LLMResponse(content="done", tool_calls=[])
            return
        # Pre-tool prose, then a ToolCallStarted, then the terminal
        # LLMResponse carrying that tool call.
        yield "Let me also check Y "
        yield TCS(index=0, id="tc2", name="test_tool")
        yield LLMResponse(
            content="Let me also check Y ",
            tool_calls=[ToolCall(id="tc2", name="test_tool", arguments={})],
        )

    agent.llm_service.stream_with_tool_detection = streamed_followup


    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = await collect_native_stream(agent, lambda: agent._handle_orchestrator_response_streaming(
        response=first_response, feature_tools=[], system_prompt="sys", force_local_only=False,
        effective_model="m", user_message="hi", request_id="req-rev-followup",
    ))

    text = "".join(c for c in chunks if isinstance(c, str))
    # In-band revise sentinel for the follow-up tool call must appear
    # in the stream. The chat client uses this to clear the
    # "Let me also check Y" bubble before the tool runs.
    assert "\x1eKESTREL:REVISE:" in text, (
        f"follow-up ToolCallStarted must emit the in-band revise "
        f"sentinel for the chat client to clear speculative pre-tool "
        f"prose. Got stream: {text!r}"
    )
    # The corresponding SSE emit should also have been invoked at
    # least once (parity with first-level honesty-layer signaling).
    assert agent._emit_revising_event.await_count >= 1


@pytest.mark.asyncio
async def test_orchestrator_loop_followup_separator_emitted_once_per_turn(native_tool_effect_agent):
    """The "\\n---\\n" separator is the chat UI's wire-protocol marker
    between tool activity and final synthesis. Pre-rewrite it fired
    exactly once per turn. Post-rewrite the orchestrator must preserve
    that semantic — multiple iterations of the loop yield only ONE
    separator total, not one per iteration."""

    agent, feature = await _orchestrator_agent(native_tool_effect_agent)
    feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}
    )

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

    agent.llm_service.stream_with_tool_detection = streamed_followup


    first_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
    )

    chunks = await collect_native_stream(agent, lambda: agent._handle_orchestrator_response_streaming(
        response=first_response, feature_tools=[],
        system_prompt="sys", force_local_only=False,
        effective_model="m", user_message="hi",
    ))

    # Two iterations ran (tool ran twice).
    assert feature.execute_as_subagent.await_count == 2
    # But the separator fired exactly once.
    text = "".join(c for c in chunks if isinstance(c, str))
    assert text.count("\n---\n") == 1, (
        f"expected exactly one separator per turn, got {text.count('---')}: {text!r}"
    )
