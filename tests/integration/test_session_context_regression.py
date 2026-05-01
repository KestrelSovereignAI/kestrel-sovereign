"""
Regression tests for session context loading in LLM calls.

These tests verify that conversation history is actually passed to the LLM,
not just fetched but ignored.

CRITICAL: This was a major bug where context_result.messages was computed
but never passed to llm_service.generate(). The fix was to use
generate_with_messages() with the full conversation history.

See commit: fix(llm): Pass conversation history to LLM using generate_with_messages
"""
import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_history():
    """Sample conversation history."""
    return [
        {"role": "user", "content": "Hello, remember the code ALPHA-123"},
        {"role": "assistant", "content": "I'll remember the code ALPHA-123."},
        {"role": "user", "content": "What topics should we discuss?"},
        {"role": "assistant", "content": "We could discuss many things."},
    ]


@pytest.mark.asyncio
async def test_context_manager_uses_passed_history(mock_history):
    """
    REGRESSION TEST: Verify context_manager preserves passed history.

    The bug was that context_manager.build_context() would fetch its own
    history instead of using the session-filtered history passed to it.
    """
    from kestrel_sovereign.agent.context_manager import ContextManager

    # Create mock storage that returns WRONG global history
    mock_storage = MagicMock()
    mock_storage.get_conversation_history = AsyncMock(return_value=[
        {"role": "user", "content": "WRONG - this is global history"},
        {"role": "assistant", "content": "WRONG - should not see this"},
    ])

    # Create mock counter
    mock_counter = MagicMock()
    mock_counter.count = MagicMock(return_value=10)
    mock_counter.count_messages = MagicMock(return_value=20)
    mock_counter.truncate_to_tokens = MagicMock(side_effect=lambda x, y: x)

    # Session-specific history that should be used
    session_history = [
        {"role": "user", "content": "CORRECT - session-specific message"},
        {"role": "assistant", "content": "CORRECT - session-specific response"},
    ]

    # Create context manager
    context_manager = ContextManager(
        storage=mock_storage,
        model="gpt-5",
        consolidator=None
    )
    context_manager.counter = mock_counter

    # Build context with pre-fetched session history
    result = await context_manager.build_context(
        query="Test query",
        constitution="Test constitution",
        include_briefing=False,
        include_memories=False,
        include_rag=False,
        privacy_mode="NORMAL",
        conversation_history=session_history  # Pass session-filtered history
    )

    # CRITICAL ASSERTION: Result should contain session history, not global
    assert len(result.messages) == 2, f"Expected 2 messages from session, got {len(result.messages)}"

    # Verify the correct history was used
    assert "CORRECT" in result.messages[0]["content"], "Should use passed session history, not fetch global"
    assert "WRONG" not in str(result.messages), "Should NOT use global history when session history is passed"


@pytest.mark.asyncio
async def test_context_manager_falls_back_to_storage_without_history():
    """
    Test that when no history is passed, context_manager fetches from kestrel_sovereign.storage.
    """
    from kestrel_sovereign.agent.context_manager import ContextManager

    # Create mock storage with properly mocked conversation attribute
    storage_history = [
        {"role": "user", "content": "Message from storage"},
        {"role": "assistant", "content": "Response from storage"},
    ]
    mock_storage = MagicMock()

    # Mock the conversation store methods (context_manager uses get_conversation_history)
    mock_conversation = AsyncMock()
    mock_conversation.get_full_history = AsyncMock(return_value=storage_history)
    mock_conversation.get_conversation_history = AsyncMock(return_value=storage_history)
    mock_storage.conversation = mock_conversation

    # Also mock get_conversation_history for direct calls
    mock_storage.get_conversation_history = AsyncMock(return_value=storage_history)

    # Create mock counter
    mock_counter = MagicMock()
    mock_counter.count = MagicMock(return_value=10)
    mock_counter.count_messages = MagicMock(return_value=20)
    mock_counter.truncate_to_tokens = MagicMock(side_effect=lambda x, y: x)

    # Create context manager
    context_manager = ContextManager(
        storage=mock_storage,
        model="gpt-5",
        consolidator=None
    )
    context_manager.counter = mock_counter

    # Build context WITHOUT passing history
    result = await context_manager.build_context(
        query="Test query",
        constitution="Test constitution",
        include_briefing=False,
        include_memories=False,
        include_rag=False,
        privacy_mode="NORMAL",
        # conversation_history NOT passed
    )

    # Should fall back to storage
    assert len(result.messages) == 2
    assert "Message from storage" in result.messages[0]["content"]


@pytest.mark.asyncio
async def test_formatted_history_has_correct_structure(mock_history):
    """
    Verify that formatted history has OpenAI-compatible structure.
    """
    from kestrel_sovereign.agent.context_builder import ContextBuilder

    mock_storage = MagicMock()
    mock_counter = MagicMock()
    mock_counter.count = MagicMock(return_value=10)

    builder = ContextBuilder(storage=mock_storage)
    builder.counter = mock_counter

    formatted = builder.format_conversation_history(mock_history)

    # Each message should have role and content
    for msg in formatted:
        assert "role" in msg, "Each message must have a role"
        assert "content" in msg, "Each message must have content"
        assert msg["role"] in ("user", "assistant", "system"), f"Invalid role: {msg['role']}"


@pytest.mark.asyncio
async def test_kestrel_agent_uses_generate_with_messages():
    """
    REGRESSION TEST: Verify kestrel_agent.py calls generate_with_messages, not generate.

    This is verified by reading the source code.
    """
    import inspect
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    # process_input itself enters the turn lifecycle (Phase 2 of #889 —
    # bootstrap + command paths must be inside the lifecycle), then calls
    # _process_input_traced_locked where the actual LLM call lives.
    source = inspect.getsource(KestrelAgent.process_input)
    locked_source = inspect.getsource(KestrelAgent._process_input_traced_locked)
    combined = source + locked_source

    # CRITICAL: Must use generate_with_messages, not generate
    assert "generate_with_messages" in combined, \
        "process_input (or its delegate) MUST call generate_with_messages to pass conversation history"

    # Verify we build the messages array
    assert "messages.extend(context_result.messages)" in combined or "context_result.messages" in combined, \
        "process_input (or its delegate) MUST include context_result.messages in the messages array"


@pytest.mark.asyncio
async def test_streaming_uses_messages_with_history():
    """
    REGRESSION TEST: Verify streaming.py passes conversation history in messages.

    Streaming uses stream_with_tool_detection(messages=messages) instead of
    generate_with_messages, but the key requirement is that conversation history
    is included in the messages array passed to the LLM.
    """
    import inspect
    from kestrel_sovereign.agent.streaming import StreamingMixin

    # process_input_streaming → _process_input_streaming_traced_locked
    # (the wrapper enters the turn lifecycle inline; the body lives in _locked).
    source = inspect.getsource(StreamingMixin.process_input_streaming)
    locked_source = inspect.getsource(StreamingMixin._process_input_streaming_traced_locked)
    combined = source + locked_source

    # CRITICAL: Must pass messages to the streaming call
    # Streaming uses stream_with_tool_detection(messages=messages, ...)
    assert "messages=messages" in combined, \
        "process_input_streaming (or its delegate) MUST pass messages array to LLM service"

    # Verify we build the messages array with conversation history
    assert "messages.extend(context_result.messages)" in combined or "context_result.messages" in combined, \
        "process_input_streaming (or its delegate) MUST include context_result.messages in the messages array"
