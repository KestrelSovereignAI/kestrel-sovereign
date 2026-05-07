"""
Regression: streaming agent must persist the FULL visible assistant text.

When the LLM emits explanatory text BEFORE deciding to call tools, those
chunks are streamed to the client (the user sees them) and accumulated
into `full_response`. The synthesizing answer AFTER tool execution is
accumulated separately in `tool_response_chunks`. The user sees both
streams concatenated in the chat pane.

The persisted assistant message used to be ONLY the post-tool half
(tool_response_chunks). On the next user turn, the conversation-history
loader showed the agent only the post-tool synthesis — the pre-tool
reasoning the user had just seen was missing. Surfaced by Meridian's
"I don't see my own quantum response" transcript.

The fix: persist ``pre_tool_text + post_tool_text`` so the next turn's
context contains exactly what the user saw.
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
async def test_streaming_persists_pre_tool_plus_post_tool_text():
    """The assistant turn persisted to the DB must include BOTH the
    pre-tool reasoning chunks and the post-tool synthesis chunks."""
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

    # Drain the stream like an HTTP client would.
    yielded = []
    async for chunk in mock_agent.process_input_streaming(
        "what's the epic status?", session_id="sess-1"
    ):
        yielded.append(chunk)

    # User-visible stream = pre-tool chunks + post-tool chunks concatenated.
    visible_text = "".join(yielded)
    assert "I'll check the github epic." in visible_text
    assert "Wave 2 is in flight." in visible_text

    # Find the assistant-row insert (there's also a user-row insert).
    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1, (
        f"expected exactly one assistant-row persist; got {len(assistant_inserts)}"
    )

    persisted = assistant_inserts[0]["content"]

    # The bug: persisted used to equal ONLY the post-tool synthesizing
    # chunks. The fix is to persist pre-tool + post-tool. Assert both.
    assert "I'll check the github epic." in persisted, (
        "pre-tool reasoning must be persisted — the user saw it and "
        "the next turn's context loader needs it"
    )
    assert "Pulling it now." in persisted, "all pre-tool chunks, not just the first"
    assert "Wave 2 is in flight." in persisted, "post-tool synthesis must remain persisted"

    # Persisted text should match the user-visible stream byte-for-byte
    # (post-response-hook is identity in this test).
    assert persisted == visible_text, (
        "persisted assistant turn must match exactly what was streamed to the user"
    )


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
