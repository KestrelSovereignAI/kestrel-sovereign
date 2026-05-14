"""STOP HookInput enrichment for #1238.

When a turn ends, the STOP hook used to fire with only ``session_id`` and
``hook_event_name``. Subscribers — e.g. the kestrel-feature-reflection
``on_stop`` handler that #1238 needs — therefore had to round-trip
through storage to reconstruct what just happened.

The fix populates the existing SDK ``HookInput`` fields with the turn's
context at the firing site (mirroring what POST_RESPONSE already
carries):

- ``user_message`` — the raw user input
- ``response_text`` — the final visible assistant text
- ``tool_calls``   — the tool-call dicts the LLM emitted (None if no tools)
- ``tool_results`` — the result envelopes the tools returned (None if no tools)

These tests pin that contract end-to-end through both the streaming and
non-streaming paths.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.hooks.base import HookEvent, HookInput


@asynccontextmanager
async def _passthrough():
    """Stand-in for the privacy-transition-lock and turn-lifecycle context
    managers. The hook-payload tests don't depend on their semantics."""
    yield


def _record_stop_hook_input(captured: list):
    """Return an execute_hooks_parallel callable that records STOP inputs."""

    async def _capture(event, hook_input):
        if event == HookEvent.STOP:
            captured.append(hook_input)

    return _capture


def _build_streaming_mock_agent(*, stop_captured: list, hooks_present: bool = True):
    """Build a MagicMock agent wired up to run process_input_streaming with
    the real StreamingMixin methods bound. Captures any STOP HookInput
    fired via execute_hooks_parallel into ``stop_captured``."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "did:test:streaming"
    agent.extension = None
    agent._cached_features_prompt = ""
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._maybe_audit = AsyncMock()
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"

    ctx = MagicMock()
    ctx.system_prompt = "system"
    ctx.dynamic_user_context = ""
    ctx.messages = []
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=ctx)

    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()

    if hooks_present:
        hooks_manager = MagicMock()
        # USER_PROMPT_SUBMIT and POST_RESPONSE both go through ``execute_hooks``;
        # return a benign output so the streaming pipeline doesn't choke. We
        # only care about STOP, which goes through ``execute_hooks_parallel``.
        from kestrel_sdk.hooks.base import HookOutput
        benign_output = HookOutput()
        hooks_manager.execute_hooks = AsyncMock(return_value=benign_output)
        hooks_manager.get_enabled_hooks = MagicMock(return_value=[])
        hooks_manager.execute_hooks_parallel = AsyncMock(
            side_effect=_record_stop_hook_input(stop_captured)
        )
        agent.hooks_manager = hooks_manager
    else:
        agent.hooks_manager = None

    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    return agent


@pytest.mark.asyncio
async def test_streaming_stop_hook_carries_user_message_and_response_text_no_tools():
    """Plain text turn (no tool calls) — STOP HookInput must carry user
    message + response text, with tool_calls/tool_results None."""
    captured: list[HookInput] = []
    agent = _build_streaming_mock_agent(stop_captured=captured)

    async def _stream_no_tools(**kwargs):
        for piece in ["Hello ", "world."]:
            yield piece

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = _stream_no_tools

    user_msg = "What's the weather like?"
    yielded = []
    async for chunk in agent.process_input_streaming(user_msg, session_id="s-1"):
        yielded.append(chunk)

    assert len(captured) == 1, f"expected one STOP fire; got {len(captured)}"
    stop = captured[0]
    assert stop.hook_event_name == HookEvent.STOP.value
    assert stop.session_id == "s-1"
    assert stop.user_message == user_msg
    assert stop.response_text == "Hello world."
    assert stop.tool_calls is None
    assert stop.tool_results is None


@pytest.mark.asyncio
async def test_streaming_stop_hook_carries_tool_calls_and_results_when_tools_fired():
    """Tool-using turn — STOP HookInput must carry the tool_calls the LLM
    emitted and the tool_results envelopes the dispatcher gathered, so a
    per-turn subscriber sees the complete shape of the turn."""
    from kestrel_sovereign.agent.streaming import StreamingMixin
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    captured: list[HookInput] = []
    agent = _build_streaming_mock_agent(stop_captured=captured)

    async def _stream_with_tool(**kwargs):
        yield "Looking that up. "
        yield LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc-A", name="github_view", arguments={"issue": 1238})
            ],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = _stream_with_tool

    # The orchestrator-streaming generator's signature accepts
    # ``tool_results`` as an out-parameter — our mock populates it the
    # same way the real orchestrator would. Envelope shape includes
    # ``arguments`` so STOP can derive tool_calls from the same list.
    async def _orchestrator_stream(*, tool_results=None, **kwargs):
        if tool_results is not None:
            tool_results.append(
                {
                    "tool_call_id": "tc-A",
                    "name": "github_view",
                    "arguments": {"issue": 1238},
                    "result": {"status": "ok", "data": {"title": "the issue"}},
                }
            )
        for piece in ["Found ", "it."]:
            yield piece

    agent._handle_orchestrator_response_streaming = _orchestrator_stream

    yielded = []
    async for chunk in agent.process_input_streaming(
        "what's on #1238?", session_id="s-tool"
    ):
        yielded.append(chunk)

    assert len(captured) == 1
    stop = captured[0]
    assert stop.user_message == "what's on #1238?"
    # Visible response = pre-tool prose + post-tool synthesis (per the
    # Meridian self-recall fix). STOP carries the same text.
    assert "Looking that up." in (stop.response_text or "")
    assert "Found it." in (stop.response_text or "")
    # Tool calls derived from accumulated tool_results envelopes —
    # name/arguments/id line up by index.
    assert stop.tool_calls is not None and len(stop.tool_calls) == 1
    assert stop.tool_calls[0]["name"] == "github_view"
    assert stop.tool_calls[0]["arguments"] == {"issue": 1238}
    assert stop.tool_calls[0]["id"] == "tc-A"
    # Tool results carry the envelope the dispatcher recorded.
    assert stop.tool_results is not None and len(stop.tool_results) == 1
    assert stop.tool_results[0]["name"] == "github_view"
    assert stop.tool_results[0]["result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_streaming_stop_hook_captures_chained_tool_iterations():
    """Multi-iteration tool flow (model calls A, sees result, calls B)
    must produce a STOP payload where tool_calls and tool_results line up
    by index across all iterations — not just the first. Without this,
    subscribers see tool_results rows with no matching call metadata and
    fall back to storage queries (codex review caught this on v1)."""
    from kestrel_sovereign.agent.streaming import StreamingMixin
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    captured: list[HookInput] = []
    agent = _build_streaming_mock_agent(stop_captured=captured)

    async def _stream_with_initial_tool(**kwargs):
        yield "Step one. "
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc-A", name="lookup_a", arguments={"q": "first"})],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = _stream_with_initial_tool

    # Orchestrator stream simulates the second iteration: after lookup_a
    # returns, the model decides to call lookup_b. Both envelopes get
    # appended to tool_results, even though only lookup_a appears in the
    # initial LLMResponse.
    async def _orchestrator_stream(*, tool_results=None, **kwargs):
        if tool_results is not None:
            tool_results.append({
                "tool_call_id": "tc-A",
                "name": "lookup_a",
                "arguments": {"q": "first"},
                "result": {"status": "ok"},
            })
            tool_results.append({
                "tool_call_id": "tc-B",
                "name": "lookup_b",
                "arguments": {"q": "second"},
                "result": {"status": "ok"},
            })
        yield "All done."

    agent._handle_orchestrator_response_streaming = _orchestrator_stream

    async for _ in agent.process_input_streaming("chain please", session_id="s-chain"):
        pass

    assert len(captured) == 1
    stop = captured[0]
    assert stop.tool_calls is not None and len(stop.tool_calls) == 2
    assert stop.tool_results is not None and len(stop.tool_results) == 2
    # Index alignment: tool_calls[i] matches tool_results[i].
    for i in range(2):
        assert stop.tool_calls[i]["id"] == stop.tool_results[i]["tool_call_id"]
        assert stop.tool_calls[i]["name"] == stop.tool_results[i]["name"]
    assert {tc["name"] for tc in stop.tool_calls} == {"lookup_a", "lookup_b"}


@pytest.mark.asyncio
async def test_streaming_stop_hook_skipped_when_no_hooks_manager():
    """If the agent has no hooks_manager, the STOP block is a no-op —
    no AttributeError. Pure regression guard."""
    captured: list[HookInput] = []
    agent = _build_streaming_mock_agent(stop_captured=captured, hooks_present=False)

    async def _stream(**kwargs):
        yield "done"

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = _stream

    yielded = []
    async for chunk in agent.process_input_streaming("hi", session_id="s"):
        yielded.append(chunk)

    # No fires recorded; no exception raised.
    assert captured == []


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_response_accepts_tool_results_out_param():
    """``_handle_orchestrator_response`` mirrors the streaming sibling's
    ``tool_results`` plumbing — the caller passes a list, the dispatcher
    appends each envelope. Verifies the new param threads to
    ``_execute_tool_batch``."""
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = MagicMock()
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="done", tool_calls=None)
    )
    agent._build_tool_calls_msg = MagicMock(
        return_value=[
            {
                "id": "tc-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
    )

    # Capture: does _execute_tool_batch receive the tool_results list we passed?
    captured_batches: list[dict] = []

    async def _capture_batch(tool_calls, features, known, messages, iteration, user_msg, **kwargs):
        captured_batches.append({
            "tool_results_is": kwargs.get("tool_results"),
            "session_id": kwargs.get("session_id"),
        })

    agent._execute_tool_batch = _capture_batch
    agent._build_all_tools = MagicMock(return_value=[])
    agent._prune_orchestrator_messages = MagicMock(side_effect=lambda msgs, _t: msgs)
    # Reflection / repair / finalize phases all no-op for this test.
    agent._signals_unfinished_tool_work = MagicMock(return_value=False)

    handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)

    caller_tool_results: list = []
    initial = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc-1", name="lookup", arguments={})],
    )
    result = await handler(
        response=initial,
        feature_tools=[],
        system_prompt="sys",
        force_local_only=False,
        effective_model="m",
        user_message="hi",
        session_id="s-non",
        tool_results=caller_tool_results,
    )

    assert result == "done"
    # The batch was called exactly once, and the caller's tool_results list
    # was forwarded by identity (so any appends the dispatcher makes show
    # up in the caller's list).
    assert len(captured_batches) == 1
    assert captured_batches[0]["tool_results_is"] is caller_tool_results
    assert captured_batches[0]["session_id"] == "s-non"


def test_handle_orchestrator_response_signature_has_tool_results_kwarg():
    """Signature pin so a future refactor doesn't silently drop the
    out-parameter that STOP HookInput enrichment depends on."""
    import inspect
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin

    sig = inspect.signature(OrchestratorEngineMixin._handle_orchestrator_response)
    assert "tool_results" in sig.parameters
    assert sig.parameters["tool_results"].default is None
