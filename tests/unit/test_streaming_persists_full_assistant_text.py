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
from tests.utils.native_streaming import streaming_agent

import pytest
from unittest.mock import AsyncMock, MagicMock




@pytest.mark.asyncio
async def test_streaming_persists_post_tool_content_and_pre_tool_metadata(native_tool_effect_agent):
    """The assistant content is post-tool only; metadata preserves recall."""
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.agent.context_builder import ContextBuilder
    from kestrel_sovereign.agent.streaming import StreamingMixin
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    # Capture what add_conversation gets called with — the sole assertion
    # target. We don't care about anything else the persistence layer
    # does; only that the persisted text matches the user-visible stream.
    add_convo_calls = []
    mock_agent = await streaming_agent(native_tool_effect_agent, persisted=add_convo_calls)
    mock_agent.context_manager.build_context.return_value.semantic_recall_dependencies = (
        ("assertion-opaque-id", "revision-opaque-id"),
    )

    async def mock_stream_with_tool_detection(**kwargs):
        yield "I'll check the github epic. "
        yield "Pulling it now."
        yield ToolCallStarted(index=0, id="tc1", name="github")
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="github", arguments={})],
        )

    mock_agent.llm_service.stream_with_tool_detection = mock_stream_with_tool_detection

    # The post-tool orchestrator stream — synthesizing answer after the
    # tool returns. The test checks that BOTH halves end up in the DB.
    async def mock_orchestrator_streaming(**kwargs):
        for piece in ["Found ", "the epic. ", "Wave 2 is in flight."]:
            yield piece

    mock_agent._handle_orchestrator_response_streaming = mock_orchestrator_streaming

    # Bind the actual mixin so the code under test runs unchanged.

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
    assert metadata["semantic_recall_dependencies"] == [
        {"assertion_id": "assertion-opaque-id", "revision_id": "revision-opaque-id"}
    ]
    user_inserts = [c for c in add_convo_calls if c["role"] == "user"]
    assert user_inserts[0]["metadata"]["semantic_recall_dependencies"] == [
        {"assertion_id": "assertion-opaque-id", "revision_id": "revision-opaque-id"}
    ]

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
async def test_streaming_no_tool_calls_persists_full_response_unchanged(native_tool_effect_agent):
    """Regression guard: the no-tool-calls path was already correct
    (it persisted ``"".join(full_response)``). The fix to the tool path
    must not perturb this path."""
    from kestrel_sovereign.agent.streaming import StreamingMixin

    add_convo_calls = []
    mock_agent = await streaming_agent(native_tool_effect_agent, persisted=add_convo_calls)

    async def mock_stream_no_tools(**kwargs):
        for piece in ["Hello", " ", "world"]:
            yield piece
        # No LLMResponse with tool_calls — clean text stream.

    mock_agent.llm_service.stream_with_tool_detection = mock_stream_no_tools

    yielded = []
    async for chunk in mock_agent.process_input_streaming("hi", session_id="sess-2"):
        yielded.append(chunk)

    assistant_inserts = [c for c in add_convo_calls if c["role"] == "assistant"]
    assert len(assistant_inserts) == 1
    assert assistant_inserts[0]["content"] == "Hello world"
