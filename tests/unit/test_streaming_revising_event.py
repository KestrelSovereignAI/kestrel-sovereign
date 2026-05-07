"""
Wave 5B: ToolCallStarted markers must drive a ``revising`` SSE event.

The honesty layer (#1042 layer 2 / #1045) gates the in-flight chat bubble
on the moment a tool call first appears in the provider stream. Until
Wave 5B, ``stream_with_tool_detection`` already forwarded the marker,
but ``agent/streaming.py``'s consumer loop silently dropped it (no
``elif`` branch). This test pinning makes that drop a regression: every
``ToolCallStarted`` from the LLM stream must trigger
``agent.emit_event("revising", ...)`` so the existing
``/api/agent/notifications/sse`` channel surfaces it to subscribers.

What we DON'T test here:
* Frontend behavior on receipt — that's Wave 5C.
* Audit-hook gating on the marker — that's Wave 5D.
* The SSE wire-format itself — already covered by the notifications-sse
  endpoint tests; we only verify ``emit_event`` is invoked correctly.
"""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.llm import ToolCallStarted


@asynccontextmanager
async def _passthrough():
    yield


def _build_mock_agent():
    """Reusable mock-agent factory for the streaming-mixin call paths."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
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
    agent._maybe_audit = AsyncMock()
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = None
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"
    agent._current_request_id = "req-abc"

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

    # Bind the real mixin methods to the mock agent.
    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    agent._emit_revising_event = StreamingMixin._emit_revising_event.__get__(agent)
    return agent


@pytest.mark.asyncio
async def test_tool_call_started_marker_emits_revising_event():
    """Every ToolCallStarted in the stream → exactly one ``revising`` emit_event call."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()
    agent.emit_event = AsyncMock()

    async def stream():
        yield "I'll check the github epic. "
        yield "Pulling it now."
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="github", arguments={})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async def post_tool(**kw):
        for piece in ["Found ", "the epic."]:
            yield piece

    agent._handle_orchestrator_response_streaming = post_tool

    chunks = []
    async for c in agent.process_input_streaming("status?", session_id="sess-7"):
        chunks.append(c)

    # The ToolCallStarted itself MUST NOT leak into the user-visible
    # text stream (the bubble is text/plain, markers are control flow).
    assert all(isinstance(c, str) for c in chunks)
    visible = "".join(chunks)
    assert "I'll check the github epic." in visible
    assert "Found the epic." in visible

    # Exactly one revising event for the single marker, with the
    # marker's fields and the active request_id.
    revising_calls = [
        call for call in agent.emit_event.call_args_list
        if call.args and call.args[0] == "revising"
    ]
    assert len(revising_calls) == 1, agent.emit_event.call_args_list
    payload = revising_calls[0].args[1]
    assert payload["type"] == "revising"
    assert payload["request_id"] == "req-abc"
    assert payload["session_id"] == "sess-7"
    assert payload["index"] == 0
    assert payload["tool_call_id"] == "tc1"
    assert payload["tool_name"] == "github"


@pytest.mark.asyncio
async def test_multiple_markers_emit_one_event_each_in_order():
    """Two distinct ToolCallStarted markers → two revising events, in arrival order."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()
    agent.emit_event = AsyncMock()

    async def stream():
        yield "thinking..."
        yield ToolCallStarted(index=0, id="tc-a", name="search")
        yield ToolCallStarted(index=1, id="tc-b", name="fetch")
        yield LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc-a", name="search", arguments={}),
                ToolCall(id="tc-b", name="fetch", arguments={}),
            ],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async def post_tool(**kw):
        yield "done."

    agent._handle_orchestrator_response_streaming = post_tool

    async for _ in agent.process_input_streaming("go", session_id="s"):
        pass

    revising_calls = [
        call for call in agent.emit_event.call_args_list
        if call.args and call.args[0] == "revising"
    ]
    assert len(revising_calls) == 2
    assert revising_calls[0].args[1]["index"] == 0
    assert revising_calls[0].args[1]["tool_name"] == "search"
    assert revising_calls[1].args[1]["index"] == 1
    assert revising_calls[1].args[1]["tool_name"] == "fetch"


@pytest.mark.asyncio
async def test_no_marker_no_revising_event():
    """No ToolCallStarted in the stream → no revising event at all."""
    agent = _build_mock_agent()
    agent.emit_event = AsyncMock()

    async def stream():
        yield "plain answer with no tools."

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async for _ in agent.process_input_streaming("hello", session_id="s"):
        pass

    revising_calls = [
        call for call in agent.emit_event.call_args_list
        if call.args and call.args[0] == "revising"
    ]
    assert revising_calls == []


@pytest.mark.asyncio
async def test_emit_event_failure_does_not_break_stream():
    """If a listener raises, the chat stream still yields its full text.

    ``emit_event`` already swallows per-listener errors inside the agent
    event-bus. We additionally guard the wrapper so a malformed listener
    or a missing attribute can never tear down the user-visible stream.
    """
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()
    agent.emit_event = AsyncMock(side_effect=RuntimeError("listener boom"))

    async def stream():
        yield "pre-tool. "
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="github", arguments={})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async def post_tool(**kw):
        yield "post-tool synthesis."

    agent._handle_orchestrator_response_streaming = post_tool

    chunks = []
    async for c in agent.process_input_streaming("go", session_id="s"):
        chunks.append(c)

    visible = "".join(chunks)
    assert "pre-tool." in visible
    assert "post-tool synthesis." in visible


@pytest.mark.asyncio
async def test_explicit_request_id_overrides_agent_global():
    """When the endpoint passes ``request_id`` to ``process_input_streaming``,
    the emitted revising event carries that id — NOT the legacy
    ``self._current_request_id`` global.

    This prevents the race where stream B's ``register_active_request``
    overwrites ``_current_request_id`` between stream A's start and
    stream A's first ToolCallStarted marker. Without the explicit
    plumb-through, A's revising event would carry B's request id and
    the frontend would clear the wrong pane's bubble.
    """
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()
    agent.emit_event = AsyncMock()

    # Simulate the race: agent-global was set by stream A, then stream B
    # came in and overwrote it before stream A reached its marker.
    agent._current_request_id = "req-from-stream-B"

    async def stream():
        yield "stream A pre-tool. "
        yield ToolCallStarted(index=0, id="tcA", name="search")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tcA", name="search", arguments={})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async def post_tool(**kw):
        yield "done."

    agent._handle_orchestrator_response_streaming = post_tool

    # Call with the explicit request_id for stream A.
    async for _ in agent.process_input_streaming(
        "go", session_id="s", request_id="req-from-stream-A",
    ):
        pass

    revising_calls = [
        call for call in agent.emit_event.call_args_list
        if call.args and call.args[0] == "revising"
    ]
    assert len(revising_calls) == 1
    payload = revising_calls[0].args[1]
    # Must carry stream A's id even though the agent-global says B.
    assert payload["request_id"] == "req-from-stream-A", (
        "explicit request_id must override the racy _current_request_id global"
    )


@pytest.mark.asyncio
async def test_request_id_falls_back_to_global_when_not_passed():
    """Backwards-compat: pre-Wave-5B callers that don't pass request_id
    still get the legacy ``_current_request_id`` value. Single-stream
    deployments work unchanged.
    """
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()
    agent.emit_event = AsyncMock()
    agent._current_request_id = "legacy-rid"

    async def stream():
        yield ToolCallStarted(index=0, id="tc1", name="t")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="t", arguments={})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async def post_tool(**kw):
        yield "x"

    agent._handle_orchestrator_response_streaming = post_tool

    # No request_id passed — legacy call shape.
    async for _ in agent.process_input_streaming("go", session_id="s"):
        pass

    revising_calls = [
        call for call in agent.emit_event.call_args_list
        if call.args and call.args[0] == "revising"
    ]
    assert revising_calls[0].args[1]["request_id"] == "legacy-rid"


@pytest.mark.asyncio
async def test_agent_without_emit_event_is_safe():
    """If the host class doesn't expose ``emit_event`` at all, the
    wrapper short-circuits silently and the stream completes normally.

    Covers the case of bare-bones test agents and any future
    decomposition where an agent variant ships without the event bus.
    """
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()
    # Force attribute lookup to fail by deleting any pre-existing mock attr.
    if hasattr(agent, "emit_event"):
        del agent.emit_event
    # MagicMock auto-vivifies attributes; spec=[] disables that for safety.
    agent.mock_add_spec([])  # no auto-attributes
    # Re-bind the methods we actually need (mock_add_spec wiped them).
    from kestrel_sovereign.agent.streaming import StreamingMixin
    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._emit_revising_event = StreamingMixin._emit_revising_event.__get__(agent)

    # The mock_add_spec call wiped privacy/context wiring too. Cheaper
    # to just call the helper directly with a synthesized marker.
    marker = ToolCallStarted(index=0, id="tc1", name="github")
    # Should not raise even though emit_event is absent.
    await agent._emit_revising_event(marker, session_id="s")
