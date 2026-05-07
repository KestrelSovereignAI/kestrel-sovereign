"""
Wave 5D: streaming path populates SDK 0.9 narration fields on POST_RESPONSE.

Pinning tests — verify that ``_fire_post_response_hook`` receives
``pre_tool_prose``, ``tool_calls``, and ``tool_results`` matching what
the user saw and what the LLM observed. The actual narration verdict
is tested in ``test_narration_check.py``; this test guards the
plumbing between the marker boundary, orchestrator, and hook.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.llm import ToolCallStarted


@asynccontextmanager
async def _passthrough():
    yield


class _CapturingHooks:
    """Shared fake hooks_manager that records the POST_RESPONSE
    HookInput while accepting STOP / USER_PROMPT_SUBMIT firings as
    no-ops. Each test instantiates its own and reads ``.captured``.
    """

    def __init__(self):
        self.captured = {}

    def get_enabled_hooks(self, event):
        from kestrel_sdk.hooks.base import HookEvent
        if event == HookEvent.POST_RESPONSE:
            return [object()]
        return []

    async def execute_hooks(self, _event, hook_input):
        from kestrel_sdk.hooks.base import HookOutput
        self.captured["input"] = hook_input
        return HookOutput.allow("ok")

    async def execute_hooks_parallel(self, _event, _hook_input):
        return None


def _build_mock_agent():
    from kestrel_sovereign.agent.streaming import StreamingMixin

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
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._maybe_audit = AsyncMock()
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = None
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"

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

    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    agent._fire_post_response_hook = (
        StreamingMixin._fire_post_response_hook.__get__(agent)
    )
    agent._emit_revising_event = (
        StreamingMixin._emit_revising_event.__get__(agent)
    )
    return agent


@pytest.mark.asyncio
async def test_post_response_hook_receives_narration_fields_when_tools_fire():
    """Streaming path with marker + tool execution: the POST_RESPONSE
    hook input must carry pre_tool_prose (snapshot at marker
    boundary), tool_calls (LLM-issued shape), and tool_results
    (envelopes from the orchestrator)."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()

    hooks = _CapturingHooks()
    agent.hooks_manager = hooks

    async def stream():
        # Pre-tool prose the user sees before the marker.
        yield "Saving that now"
        yield "..."
        # The marker — boundary for pre_tool_prose snapshot.
        yield ToolCallStarted(index=0, id="tc-1", name="save_fact")
        # LLM finishes its first turn with the tool call.
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc-1", name="save_fact", arguments={"fact": "color=teal"})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()
    agent.emit_event = AsyncMock()

    # Orchestrator stub: yields post-tool synthesis chunks AND writes
    # into the tool_results list the streaming layer passes in.
    async def post_tool(*, tool_results=None, **kw):
        if tool_results is not None:
            tool_results.append({
                "tool_call_id": "tc-1",
                "name": "save_fact",
                "result": {"status": "error", "error": "no store"},
            })
        for piece in ["Looking at the result, ", "the save did not persist."]:
            yield piece

    agent._handle_orchestrator_response_streaming = post_tool

    chunks = []
    async for c in agent.process_input_streaming("save my color", session_id="s1"):
        chunks.append(c)

    visible = "".join(chunks)
    assert "Saving that now" in visible
    assert "did not persist" in visible

    hook_input = hooks.captured["input"]
    # Pre-tool prose: snapshot taken at the FIRST marker, so contains
    # everything streamed before that marker arrived.
    assert hook_input.pre_tool_prose == "Saving that now..."
    # Tool calls: shape produced by the streaming layer from
    # tool_response.tool_calls.
    assert hook_input.tool_calls == [{
        "id": "tc-1",
        "name": "save_fact",
        "arguments": {"fact": "color=teal"},
    }]
    # Tool results: envelope appended by the orchestrator stub.
    assert hook_input.tool_results == [{
        "tool_call_id": "tc-1",
        "name": "save_fact",
        "result": {"status": "error", "error": "no store"},
    }]
    # The post-tool text reaches the hook in response_text.
    assert "did not persist" in hook_input.response_text


@pytest.mark.asyncio
async def test_post_response_hook_no_tools_omits_narration_fields():
    """When no tool calls fire, the new fields stay None — the
    narration check should treat them as 'nothing to verify'."""
    agent = _build_mock_agent()
    hooks = _CapturingHooks()
    agent.hooks_manager = hooks

    async def stream():
        yield "Hello! No tools needed."

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async for _ in agent.process_input_streaming("hi", session_id="s2"):
        pass

    hook_input = hooks.captured["input"]
    assert hook_input.response_text == "Hello! No tools needed."
    assert hook_input.pre_tool_prose is None
    assert hook_input.tool_calls is None
    assert hook_input.tool_results is None


@pytest.mark.asyncio
async def test_marker_never_fires_falls_back_to_pre_tool_buffer():
    """Defensive: some adapter/path may yield tool_calls without
    emitting a ToolCallStarted marker (e.g. legacy plugin not yet on
    SDK 0.7). The pre_tool_prose passed to the hook must still be
    populated — falls back to the buffered text-before-LLMResponse."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_mock_agent()
    agent.emit_event = AsyncMock()
    hooks = _CapturingHooks()
    agent.hooks_manager = hooks

    async def stream():
        yield "I'll attempt to save. "
        # NO ToolCallStarted marker — directly the LLMResponse.
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc-1", name="save", arguments={})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async def post_tool(*, tool_results=None, **kw):
        if tool_results is not None:
            tool_results.append({"tool_call_id": "tc-1", "name": "save", "result": {"status": "ok"}})
        yield "Saved."

    agent._handle_orchestrator_response_streaming = post_tool

    async for _ in agent.process_input_streaming("save", session_id="s3"):
        pass

    hook_input = hooks.captured["input"]
    # No marker → fallback to the full pre-tool buffer.
    assert hook_input.pre_tool_prose == "I'll attempt to save. "
