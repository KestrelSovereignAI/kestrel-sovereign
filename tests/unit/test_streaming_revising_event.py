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
from tests.utils.native_streaming import streaming_agent
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.llm import ToolCallStarted







@pytest.mark.asyncio
async def test_tool_call_started_marker_emits_revising_event(native_tool_effect_agent):
    """Every ToolCallStarted in the stream → exactly one ``revising`` emit_event call."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = await streaming_agent(native_tool_effect_agent)
    agent.emit_event = AsyncMock()

    async def stream():
        yield "I'll check the github epic. "
        yield "Pulling it now."
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="github", arguments={})],
        )

    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async def post_tool(**kw):
        for piece in ["Found ", "the epic."]:
            yield piece

    agent._handle_orchestrator_response_streaming = post_tool

    chunks = []
    async for c in agent.process_input_streaming("status?", session_id="sess-7", request_id="req-abc"):
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
async def test_multiple_markers_emit_one_event_each_in_order(native_tool_effect_agent):
    """Two distinct ToolCallStarted markers → two revising events, in arrival order."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = await streaming_agent(native_tool_effect_agent)
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
async def test_no_marker_no_revising_event(native_tool_effect_agent):
    """No ToolCallStarted in the stream → no revising event at all."""
    agent = await streaming_agent(native_tool_effect_agent)
    agent.emit_event = AsyncMock()

    async def stream():
        yield "plain answer with no tools."

    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async for _ in agent.process_input_streaming("hello", session_id="s"):
        pass

    revising_calls = [
        call for call in agent.emit_event.call_args_list
        if call.args and call.args[0] == "revising"
    ]
    assert revising_calls == []


@pytest.mark.asyncio
async def test_emit_event_failure_does_not_break_stream(native_tool_effect_agent):
    """If a listener raises, the chat stream still yields its full text.

    ``emit_event`` already swallows per-listener errors inside the agent
    event-bus. We additionally guard the wrapper so a malformed listener
    or a missing attribute can never tear down the user-visible stream.
    """
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = await streaming_agent(native_tool_effect_agent)
    agent.emit_event = AsyncMock(side_effect=RuntimeError("listener boom"))

    async def stream():
        yield "pre-tool. "
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="github", arguments={})],
        )

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
async def test_explicit_request_id_overrides_agent_global(native_tool_effect_agent):
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

    agent = await streaming_agent(native_tool_effect_agent)
    agent.emit_event = AsyncMock()

    # Simulate the race: agent-global was set by stream A, then stream B
    # came in and overwrote it before stream A reached its marker.
    agent._current_request_id = "req-from-stream-B"

    async def stream():
        agent._current_request_id = "req-from-stream-B"
        yield "stream A pre-tool. "
        yield ToolCallStarted(index=0, id="tcA", name="search")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tcA", name="search", arguments={})],
        )

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
async def test_generated_request_id_carries_the_original_native_invocation(native_tool_effect_agent):
    """An omitted request ID gets one real native invocation for this stream."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = await streaming_agent(native_tool_effect_agent)
    agent.emit_event = AsyncMock()
    agent._current_request_id = "legacy-rid"

    invocation_ids = []
    async def stream():
        from kestrel_sovereign.agent.invocation import current_invocation_id
        invocation_ids.append(current_invocation_id())
        yield ToolCallStarted(index=0, id="tc1", name="t")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="t", arguments={})],
        )

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
    assert invocation_ids[0] and invocation_ids[0] != "legacy-rid"
    assert revising_calls[0].args[1]["request_id"] == invocation_ids[0]


@pytest.mark.asyncio
async def test_agent_without_emit_event_is_safe():
    """The event-emission helper accepts a host without an optional bus."""
    from types import SimpleNamespace
    from kestrel_sovereign.agent.streaming import StreamingMixin

    agent = SimpleNamespace()
    emit = StreamingMixin._emit_revising_event.__get__(agent)
    await emit(ToolCallStarted(index=0, id="tc1", name="github"), session_id="s")
