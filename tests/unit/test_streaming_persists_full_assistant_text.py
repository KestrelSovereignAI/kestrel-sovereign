"""
Regression: streaming tool turns keep honest history and self-recall.

When the LLM emits explanatory text BEFORE deciding to call tools, the
streaming client briefly sees those chunks, then retracts them on the
ToolCallStarted/revising marker. Persisting that pre-tool prose in the
assistant ``content`` field makes the retracted text reappear after a
conversation-history reload.

The contract is now split: ``content`` stores only the post-tool synthesis
that remains user-visible, while ``metadata.pre_tool_reasoning`` keeps the
retracted prose available to the next-turn LLM context. That preserves the
#877 Meridian self-recall fix without leaking the retracted prose through
history GET payloads.
"""
import asyncio
from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock, MagicMock


@asynccontextmanager
async def _passthrough():
    """Stand-in for the privacy-transition-lock and turn-lifecycle
    async context managers. Real implementations live elsewhere; the
    persistence behavior under test doesn't depend on them."""
    yield


@pytest.mark.asyncio
async def test_streaming_persists_post_tool_content_and_pre_tool_metadata():
    """The assistant content is post-tool only; metadata preserves recall."""
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.agent.context_builder import ContextBuilder
    from kestrel_sovereign.agent.streaming import StreamingMixin
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    # Capture what add_conversation gets called with — the sole assertion
    # target. We don't care about anything else the persistence layer
    # does; only that the persisted text matches the user-visible stream.
    add_convo_calls = []
    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: add_convo_calls.append({
            "role": role, "content": content, **kw,
        })
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.privacy_agent = privacy_agent
    mock_agent.features = {}
    mock_agent.did = "test-did"
    mock_agent.extension = None
    mock_agent._cached_features_prompt = ""
    mock_agent.is_request_cancelled = MagicMock(return_value=False)
    mock_agent.emit_event = AsyncMock()
    mock_agent._maybe_audit = AsyncMock()
    mock_agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    mock_agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    mock_agent.hooks_manager = None  # skip USER_PROMPT_SUBMIT hook
    mock_agent._get_governing_constitution = AsyncMock(return_value="")
    mock_agent.check_solvency = AsyncMock(return_value="test-model")
    mock_agent._build_all_tools = MagicMock(return_value=[])
    mock_agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    mock_agent.user_prompt_template = MagicMock()
    mock_agent.user_prompt_template.format.return_value = "rendered prompt"

    context_result = MagicMock()
    context_result.system_prompt = "system"
    context_result.dynamic_user_context = "ctx"
    context_result.messages = []
    mock_agent.context_manager = MagicMock()
    mock_agent.context_manager.build_context = AsyncMock(return_value=context_result)

    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()

    mock_agent.hooks_manager = None  # skip STOP hook

    # Stream-with-tool-detection: yield two pre-tool string chunks,
    # then an LLMResponse carrying tool calls. This mirrors the real
    # flow where the LLM explains what it's about to do before invoking
    # a tool.
    async def mock_stream_with_tool_detection(**kwargs):
        yield "I'll check the github epic. "
        yield "Pulling it now."
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="github", arguments={})],
        )

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = mock_stream_with_tool_detection

    # The post-tool orchestrator stream — synthesizing answer after the
    # tool returns. The test checks that BOTH halves end up in the DB.
    async def mock_orchestrator_streaming(**kwargs):
        for piece in ["Found ", "the epic. ", "Wave 2 is in flight."]:
            yield piece

    mock_agent._handle_orchestrator_response_streaming = mock_orchestrator_streaming

    # Bind the actual mixin so the code under test runs unchanged.
    mock_agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(mock_agent)
    mock_agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(mock_agent)
    )
    mock_agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(mock_agent)
    )
    mock_agent._emit_revising_event = (
        StreamingMixin._emit_revising_event.__get__(mock_agent)
    )

    # Drain the stream like an HTTP client would.
    yielded = []
    async for chunk in mock_agent.process_input_streaming(
        "what's the epic status?", session_id="sess-1"
    ):
        yielded.append(chunk)

    # The live stream still contains the brief pre-tool text, then an
    # in-band revise marker, then post-tool synthesis. The client uses the
    # marker to clear the in-flight bubble before rendering the final answer.
    assert yielded[0:2] == ["I'll check the github epic. ", "Pulling it now."]
    assert any("\x1eKESTREL:REVISE:" in chunk for chunk in yielded)
    assert "Wave 2 is in flight." in "".join(yielded)

    # Find the assistant-row insert (there's also a user-row insert).
    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1, (
        f"expected exactly one assistant-row persist; got {len(assistant_inserts)}"
    )

    persisted = assistant_inserts[0]["content"]
    metadata = assistant_inserts[0].get("metadata") or {}

    assert persisted == "Found the epic. Wave 2 is in flight."
    assert "I'll check the github epic." not in persisted
    assert "pre_tool_reasoning" in metadata
    assert metadata["pre_tool_reasoning"]["content"] == (
        "I'll check the github epic. Pulling it now."
    )

    cb = ContextBuilder.__new__(ContextBuilder)
    cb._llm_service = None
    cb._model_fallback = "test-stub"

    class _Counter:
        def count(self, s):
            return max(1, len(s) // 4)

        def truncate_to_tokens(self, s, n):
            return s[: n * 4]

    cb._counter = _Counter()
    cb._counter_model = "test-stub"
    formatted = cb.format_conversation_history(
        [{"role": "assistant", "content": persisted, "metadata": metadata}],
        max_tokens=10_000,
    )
    assert formatted == [{
        "role": "assistant",
        "content": (
            "I'll check the github epic. Pulling it now.\n\n"
            "Found the epic. Wave 2 is in flight."
        ),
    }]


@pytest.mark.asyncio
async def test_streaming_no_tool_calls_persists_full_response_unchanged():
    """Regression guard: the no-tool-calls path was already correct
    (it persisted ``"".join(full_response)``). The fix to the tool path
    must not perturb this path."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    add_convo_calls = []
    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: add_convo_calls.append({
            "role": role, "content": content, **kw,
        })
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.privacy_agent = privacy_agent
    mock_agent.features = {}
    mock_agent.did = "test-did"
    mock_agent.extension = None
    mock_agent._cached_features_prompt = ""
    mock_agent.is_request_cancelled = MagicMock(return_value=False)
    mock_agent._maybe_audit = AsyncMock()
    mock_agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    mock_agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    mock_agent.hooks_manager = None  # skip USER_PROMPT_SUBMIT hook
    mock_agent._get_governing_constitution = AsyncMock(return_value="")
    mock_agent.check_solvency = AsyncMock(return_value="test-model")
    mock_agent._build_all_tools = MagicMock(return_value=[])
    mock_agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    mock_agent.user_prompt_template = MagicMock()
    mock_agent.user_prompt_template.format.return_value = "rendered"

    ctx = MagicMock()
    ctx.system_prompt = "system"; ctx.dynamic_user_context = ""; ctx.messages = []
    mock_agent.context_manager = MagicMock()
    mock_agent.context_manager.build_context = AsyncMock(return_value=ctx)

    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()
    mock_agent.hooks_manager = None

    async def mock_stream_no_tools(**kwargs):
        for piece in ["Hello", " ", "world"]:
            yield piece
        # No LLMResponse with tool_calls — clean text stream.

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = mock_stream_no_tools
    mock_agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(mock_agent)
    mock_agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(mock_agent)
    )
    mock_agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(mock_agent)
    )

    yielded = []
    async for chunk in mock_agent.process_input_streaming("hi", session_id="sess-2"):
        yielded.append(chunk)

    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1
    assert assistant_inserts[0]["content"] == "Hello world"
