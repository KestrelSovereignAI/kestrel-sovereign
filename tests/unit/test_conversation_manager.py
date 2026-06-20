"""
Unit tests for ConversationManager.

Tests conversation state management, history retrieval, compaction,
and message operations with mocked AsyncStorage.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import List, Dict, Any

from kestrel_sovereign.agent.conversation_manager import ConversationManager


# =============================================================================
# Mock Classes
# =============================================================================


class MockTokenCounter:
    """Mock token counter for testing."""

    def count(self, text: str) -> int:
        """Simple token count based on word count."""
        return len(text.split())


class MockTokenBudget:
    """Mock token budget for testing."""

    def __init__(self, total_budget: int = 1000):
        self.total_budget = total_budget
        self.history = int(total_budget * 0.4)  # Match AdaptiveTokenBudget default


class MockConversationStore:
    """Mock conversation store for testing."""

    def __init__(self, agent_id: str = "test-agent"):
        self.agent_id = agent_id
        self.messages = []
        self.db = AsyncMock()

    async def get_conversation_history(self, limit: int = 100, session_id: str = None) -> List[Dict]:
        """Return recent messages, filtering excluded ones."""
        result = [m for m in self.messages if not m.get("metadata", {}).get("excluded_from_context")]
        return result[-limit:]

    async def get_full_history(self) -> List[Dict]:
        """Return all messages (unfiltered, for compaction)."""
        return self.messages.copy()

    async def get_full_history_with_ids(self) -> List[Dict]:
        """Return all messages with IDs."""
        return self.messages.copy()

    async def add_conversation(self, role: str, content: str, metadata: Dict = None,
                               session_id: str = None, rendered_content: str = None):
        """Add a message to conversation history."""
        msg_id = len(self.messages) + 1
        msg = {
            "id": msg_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        if rendered_content is not None:
            msg["rendered_content"] = rendered_content
        self.messages.append(msg)
        return msg_id

    async def get_messages_by_ids(self, message_ids: List[int]) -> List[Dict]:
        """Get messages by their IDs."""
        return [m for m in self.messages if m.get("id") in message_ids]

    async def update_messages_metadata(self, message_ids: List[int], metadata: Dict) -> int:
        """Update metadata for multiple messages."""
        count = 0
        for msg in self.messages:
            if msg.get("id") in message_ids:
                msg["metadata"].update(metadata)
                count += 1
        return count

    async def get_excluded_messages(self, limit: int = 1000) -> List[Dict]:
        """Get messages that are excluded from context."""
        return [m for m in self.messages if m.get("metadata", {}).get("excluded_from_context")]

    async def search_messages_by_content(self, query: str, limit: int = 50) -> List[Dict]:
        """Search messages by content (simple substring match)."""
        results = []
        for msg in self.messages:
            if query.lower() in msg.get("content", "").lower():
                results.append(msg)
                if len(results) >= limit:
                    break
        return results


class MockAsyncStorage:
    """Mock AsyncStorage for testing."""

    def __init__(self, agent_id: str = "test-agent"):
        self.conversation = MockConversationStore(agent_id)
        self.agent_id = agent_id


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, response: str = "This is a test summary."):
        self.response = response
        self.generate_calls = []

    async def generate(self, *, system_prompt: str = None, user_prompt: str = None,
                       model_override: str = None, **kwargs):
        """Mock generate — mirrors the REAL keyword-only LLMService.generate
        signature (system_prompt/user_prompt), so it would reject the old
        ``prompt=`` call the way production does. The legacy ``prompt`` key is
        kept (mapped to user_prompt) for existing assertions."""
        self.generate_calls.append({
            "prompt": user_prompt,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "model_override": model_override,
        })
        return self.response


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_storage():
    """Create mock AsyncStorage."""
    return MockAsyncStorage()


@pytest.fixture
def conversation_manager(mock_storage):
    """Create ConversationManager with mock storage."""
    return ConversationManager(storage=mock_storage, agent_id="test-agent")


@pytest.fixture
def mock_counter():
    """Create mock token counter."""
    return MockTokenCounter()


@pytest.fixture
def mock_llm():
    """Create mock LLM service."""
    return MockLLMService()


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        {
            "id": 1,
            "role": "user",
            "content": "Hello, how are you?",
            "metadata": {},
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "I'm doing well, thank you for asking!",
            "metadata": {},
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        },
        {
            "id": 3,
            "role": "user",
            "content": "What's the weather like?",
            "metadata": {},
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        },
        {
            "id": 4,
            "role": "assistant",
            "content": "I don't have access to real-time weather data.",
            "metadata": {},
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        },
        {
            "id": 5,
            "role": "user",
            "content": "Tell me a joke",
            "metadata": {},
            "created_at": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        },
    ]


# =============================================================================
# Priority 1 — Core Operations
# =============================================================================


@pytest.mark.asyncio
async def test_get_conversation_history_success(conversation_manager, mock_storage, sample_messages):
    """Test retrieving conversation history from storage."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute
    history = await conversation_manager.get_conversation_history()

    # Assert
    assert len(history) == 5
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_conversation_history_empty(conversation_manager):
    """Test retrieving empty conversation history."""
    # Execute
    history = await conversation_manager.get_conversation_history()

    # Assert
    assert history == []


@pytest.mark.asyncio
async def test_get_conversation_history_fallback_method(mock_storage):
    """Test fallback method for storage without conversation attribute."""
    # Setup - remove conversation attribute and add fallback method
    delattr(mock_storage, 'conversation')
    mock_storage.get_conversation_history = AsyncMock(return_value=[
        {"role": "user", "content": "test"}
    ])

    manager = ConversationManager(storage=mock_storage)

    # Execute
    history = await manager.get_conversation_history()

    # Assert
    assert len(history) == 1
    assert history[0]["role"] == "user"


@pytest.mark.asyncio
async def test_get_conversation_history_no_method(mock_storage):
    """Test behavior when no conversation history method is available."""
    # Setup - remove conversation attribute
    delattr(mock_storage, 'conversation')

    manager = ConversationManager(storage=mock_storage)

    # Execute
    history = await manager.get_conversation_history()

    # Assert
    assert history == []


@pytest.mark.asyncio
async def test_compact_session_success(conversation_manager, mock_storage, mock_llm, mock_counter, sample_messages):
    """Test successful session compaction."""
    # Setup - need more messages for compaction to trigger (preserve_recent + 5 threshold)
    # Add more messages to sample_messages
    extended_messages = sample_messages.copy()
    for i in range(6, 16):  # Add 10 more messages
        extended_messages.append({
            "id": i,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}",
            "metadata": {},
            "created_at": (datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat()
        })

    mock_storage.conversation.messages = extended_messages

    # Mock the database fetchone for getting compaction marker ID
    mock_storage.conversation.db.fetchone = AsyncMock(return_value=(16,))

    # Execute
    result = await conversation_manager.compact_session(
        llm_service=mock_llm,
        counter=mock_counter,
        preserve_recent=2,
        force=False
    )

    # Assert
    assert result["success"] is True
    assert result["messages_compacted"] == 13  # First 13 messages compacted (15 total - 2 preserved)
    assert result["messages_preserved"] == 2  # Last 2 preserved
    assert "tokens_before" in result
    assert "tokens_after" in result
    assert "tokens_saved" in result
    assert "summary_preview" in result

    # Verify LLM was called
    assert len(mock_llm.generate_calls) == 1

    # Verify original messages were marked as excluded
    excluded = await mock_storage.conversation.get_excluded_messages()
    assert len(excluded) == 13  # The 13 compacted messages


@pytest.mark.asyncio
async def test_compact_session_not_enough_messages(conversation_manager, mock_llm, mock_counter):
    """Test compaction fails when not enough messages."""
    # Setup - only 2 messages
    conversation_manager.storage.conversation.messages = [
        {"id": 1, "role": "user", "content": "Hi", "metadata": {}},
        {"id": 2, "role": "assistant", "content": "Hello", "metadata": {}},
    ]

    # Execute
    result = await conversation_manager.compact_session(
        llm_service=mock_llm,
        counter=mock_counter,
        preserve_recent=10
    )

    # Assert
    assert result["success"] is False
    assert "Not enough messages" in result["reason"]


@pytest.mark.asyncio
async def test_compact_session_force(conversation_manager, mock_storage, mock_llm, mock_counter):
    """Test forced compaction even with few messages."""
    # Setup - just enough messages when forced
    mock_storage.conversation.messages = [
        {"id": 1, "role": "user", "content": "Message one", "metadata": {}},
        {"id": 2, "role": "assistant", "content": "Response one", "metadata": {}},
        {"id": 3, "role": "user", "content": "Message two", "metadata": {}},
        {"id": 4, "role": "assistant", "content": "Response two", "metadata": {}},
    ]
    mock_storage.conversation.db.fetchone = AsyncMock(return_value=(5,))

    # Execute with force=True
    result = await conversation_manager.compact_session(
        llm_service=mock_llm,
        counter=mock_counter,
        preserve_recent=1,
        force=True
    )

    # Assert
    assert result["success"] is True
    assert result["messages_compacted"] == 3


@pytest.mark.asyncio
async def test_compact_session_llm_error(conversation_manager, mock_storage, mock_counter, sample_messages):
    """Test compaction handles LLM errors gracefully."""
    # Setup - need enough messages to pass the threshold
    extended_messages = sample_messages.copy()
    for i in range(6, 16):
        extended_messages.append({
            "id": i,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}",
            "metadata": {},
            "created_at": (datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat()
        })
    mock_storage.conversation.messages = extended_messages

    # Create mock LLM that raises an error
    mock_llm = MockLLMService()
    mock_llm.generate = AsyncMock(side_effect=Exception("LLM API error"))

    # Execute
    result = await conversation_manager.compact_session(
        llm_service=mock_llm,
        counter=mock_counter,
        preserve_recent=2
    )

    # Assert
    assert result["success"] is False
    assert "Compaction failed" in result["reason"]


@pytest.mark.asyncio
async def test_check_compaction_needed_over_threshold(conversation_manager, mock_storage, mock_counter, sample_messages):
    """Test compaction check when utilization is over threshold."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Mock create_budget to return a small budget
    with patch("kestrel_sovereign.agent.token_budget.create_budget") as mock_create_budget:
        mock_create_budget.return_value = MockTokenBudget(total_budget=10)  # Very small budget

        # Execute
        result = await conversation_manager.check_compaction_needed(
            counter=mock_counter,
            model="test-model",
            utilization_threshold=50.0
        )

    # Assert
    assert result["compaction_recommended"] is True
    assert result["utilization_percent"] > 50.0
    assert result["message_count"] == 5


@pytest.mark.asyncio
async def test_check_compaction_needed_under_threshold(conversation_manager, mock_storage, mock_counter, sample_messages):
    """Test compaction check when utilization is under threshold."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Mock create_budget to return a large budget
    with patch("kestrel_sovereign.agent.token_budget.create_budget") as mock_create_budget:
        mock_create_budget.return_value = MockTokenBudget(total_budget=10000)  # Large budget

        # Execute
        result = await conversation_manager.check_compaction_needed(
            counter=mock_counter,
            model="test-model",
            utilization_threshold=70.0
        )

    # Assert
    assert result["compaction_recommended"] is False
    assert result["utilization_percent"] < 70.0


# =============================================================================
# Priority 2 — Message Management
# =============================================================================


@pytest.mark.asyncio
async def test_get_messages_for_selection_by_ids(conversation_manager, mock_storage, sample_messages):
    """Test message selection by direct IDs."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute
    messages = await conversation_manager.get_messages_for_selection(
        mode="messages",
        criteria="1,3,5"
    )

    # Assert
    assert len(messages) == 3
    assert messages[0]["id"] == 1
    assert messages[1]["id"] == 3
    assert messages[2]["id"] == 5


@pytest.mark.asyncio
async def test_get_messages_for_selection_last_n(conversation_manager, mock_storage, sample_messages):
    """Test message selection for last N messages."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute
    messages = await conversation_manager.get_messages_for_selection(
        mode="last_n",
        criteria="3"
    )

    # Assert
    assert len(messages) == 3
    assert messages[0]["id"] == 3
    assert messages[1]["id"] == 4
    assert messages[2]["id"] == 5


@pytest.mark.asyncio
async def test_get_messages_for_selection_by_topic(conversation_manager, mock_storage, sample_messages):
    """Test message selection by topic/content search."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute
    messages = await conversation_manager.get_messages_for_selection(
        mode="topic",
        criteria="weather"
    )

    # Assert
    assert len(messages) >= 1
    assert any("weather" in m["content"].lower() for m in messages)


@pytest.mark.asyncio
async def test_get_messages_for_selection_time_range_last_hours(conversation_manager, mock_storage, sample_messages):
    """Test message selection by time range (last N hours)."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute - get messages from last hour
    messages = await conversation_manager.get_messages_for_selection(
        mode="time_range",
        criteria="last_1_hours"
    )

    # Assert - should get messages 3, 4, 5 (within last hour)
    assert len(messages) >= 1


@pytest.mark.asyncio
async def test_get_messages_for_selection_invalid_ids(conversation_manager, mock_storage):
    """Test message selection with invalid ID format."""
    # Execute
    messages = await conversation_manager.get_messages_for_selection(
        mode="messages",
        criteria="invalid,ids"
    )

    # Assert
    assert messages == []


@pytest.mark.asyncio
async def test_mark_messages_protect(conversation_manager, mock_storage, sample_messages):
    """Test marking messages as protected."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute
    result = await conversation_manager.mark_messages(
        message_ids=[1, 2],
        action="protect",
        reason="Important conversation"
    )

    # Assert
    assert result["success"] is True
    assert result["marked_count"] == 2

    # Verify metadata was updated
    messages = await mock_storage.conversation.get_messages_by_ids([1, 2])
    for msg in messages:
        assert msg["metadata"]["context_priority"] == "protected"


@pytest.mark.asyncio
async def test_mark_messages_droppable(conversation_manager, mock_storage, sample_messages):
    """Test marking messages as droppable."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute
    result = await conversation_manager.mark_messages(
        message_ids=[3],
        action="droppable",
        reason="Low priority"
    )

    # Assert
    assert result["success"] is True
    assert result["marked_count"] == 1

    # Verify metadata
    messages = await mock_storage.conversation.get_messages_by_ids([3])
    assert messages[0]["metadata"]["context_priority"] == "droppable"


@pytest.mark.asyncio
async def test_mark_messages_clear(conversation_manager, mock_storage, sample_messages):
    """Test clearing message marks."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()
    sample_messages[0]["metadata"]["context_priority"] = "protected"

    # Execute
    result = await conversation_manager.mark_messages(
        message_ids=[1],
        action="clear"
    )

    # Assert
    assert result["success"] is True
    assert result["marked_count"] == 1


@pytest.mark.asyncio
async def test_mark_messages_protected_cannot_be_droppable(conversation_manager, mock_storage, sample_messages):
    """Test that protected messages cannot be marked as droppable."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()
    mock_storage.conversation.messages[0]["metadata"]["decay_protected"] = True

    # Execute
    result = await conversation_manager.mark_messages(
        message_ids=[1],
        action="droppable"
    )

    # Assert
    assert result["success"] is False
    assert result["protected_count"] == 1


@pytest.mark.asyncio
async def test_exclude_messages_success(conversation_manager, mock_storage, sample_messages):
    """Test excluding messages from context."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute
    result = await conversation_manager.exclude_messages(
        message_ids=[2, 3],
        reason="Outdated information"
    )

    # Assert
    assert result["success"] is True
    assert result["excluded_count"] == 2

    # Verify metadata
    messages = await mock_storage.conversation.get_messages_by_ids([2, 3])
    for msg in messages:
        assert msg["metadata"]["excluded_from_context"] is True
        assert msg["metadata"]["excluded_reason"] == "Outdated information"


@pytest.mark.asyncio
async def test_exclude_messages_protected(conversation_manager, mock_storage, sample_messages):
    """Test that protected messages cannot be excluded."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()
    mock_storage.conversation.messages[0]["metadata"]["context_priority"] = "protected"

    # Execute
    result = await conversation_manager.exclude_messages(
        message_ids=[1],
        reason="Test"
    )

    # Assert
    assert result["success"] is False
    assert result["protected_count"] == 1


@pytest.mark.asyncio
async def test_restore_messages_specific_ids(conversation_manager, mock_storage, sample_messages):
    """Test restoring specific excluded messages."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()
    mock_storage.conversation.messages[1]["metadata"]["excluded_from_context"] = True
    mock_storage.conversation.messages[2]["metadata"]["excluded_from_context"] = True

    # Execute
    result = await conversation_manager.restore_messages(message_ids=[2, 3])

    # Assert
    assert result["success"] is True
    assert result["restored_count"] == 2

    # Verify metadata cleared
    messages = await mock_storage.conversation.get_messages_by_ids([2, 3])
    for msg in messages:
        assert msg["metadata"].get("excluded_from_context") is False


@pytest.mark.asyncio
async def test_restore_messages_all(conversation_manager, mock_storage, sample_messages):
    """Test restoring all excluded messages."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()
    mock_storage.conversation.messages[1]["metadata"]["excluded_from_context"] = True
    mock_storage.conversation.messages[2]["metadata"]["excluded_from_context"] = True

    # Execute - no message_ids means restore all
    result = await conversation_manager.restore_messages(message_ids=None)

    # Assert
    assert result["success"] is True
    assert result["restored_count"] == 2


@pytest.mark.asyncio
async def test_restore_messages_none_excluded(conversation_manager, mock_storage, sample_messages):
    """Test restoring when no messages are excluded."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Execute
    result = await conversation_manager.restore_messages(message_ids=None)

    # Assert
    assert result["success"] is True
    assert result["restored_count"] == 0


# =============================================================================
# Priority 3 — Summarization
# =============================================================================


@pytest.mark.asyncio
async def test_summarize_messages_success(conversation_manager, mock_storage, mock_llm, mock_counter, sample_messages):
    """Test successful message summarization."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()
    mock_storage.conversation.db.fetchone = AsyncMock(return_value=(6,))

    # Execute
    result = await conversation_manager.summarize_messages(
        llm_service=mock_llm,
        counter=mock_counter,
        message_ids=[1, 2, 3],
        preserve_key_facts=True
    )

    # Assert
    assert result["success"] is True
    assert result["messages_summarized"] == 3
    assert "tokens_before" in result
    assert "tokens_after" in result
    assert "tokens_saved" in result
    assert "summary_preview" in result

    # Verify LLM was called
    assert len(mock_llm.generate_calls) == 1
    assert "Preserve:" in mock_llm.generate_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_summarize_messages_without_key_facts(conversation_manager, mock_storage, mock_llm, mock_counter, sample_messages):
    """Test summarization without preserving key facts."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()
    mock_storage.conversation.db.fetchone = AsyncMock(return_value=(6,))

    # Execute
    result = await conversation_manager.summarize_messages(
        llm_service=mock_llm,
        counter=mock_counter,
        message_ids=[1, 2],
        preserve_key_facts=False
    )

    # Assert
    assert result["success"] is True
    # Verify key facts instruction not in prompt
    assert "Preserve:" not in mock_llm.generate_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_summarize_messages_too_few(conversation_manager, mock_storage, mock_counter):
    """Test summarization fails with too few messages."""
    # Setup
    mock_storage.conversation.messages = [
        {"id": 1, "role": "user", "content": "Hi", "metadata": {}}
    ]

    # Execute
    result = await conversation_manager.summarize_messages(
        llm_service=MockLLMService(),
        counter=mock_counter,
        message_ids=[1]
    )

    # Assert
    assert result["success"] is False
    assert "at least 2 messages" in result["error"]


@pytest.mark.asyncio
async def test_summarize_messages_excludes_protected(conversation_manager, mock_storage, mock_llm, mock_counter, sample_messages):
    """Test that protected messages are excluded from summarization."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()
    mock_storage.conversation.messages[0]["metadata"]["context_priority"] = "protected"
    mock_storage.conversation.messages[1]["metadata"]["decay_protected"] = True
    mock_storage.conversation.db.fetchone = AsyncMock(return_value=(6,))

    # Execute
    result = await conversation_manager.summarize_messages(
        llm_service=mock_llm,
        counter=mock_counter,
        message_ids=[1, 2, 3, 4]
    )

    # Assert
    assert result["success"] is True
    assert result["messages_summarized"] == 2  # Only 3 and 4, not 1 and 2
    assert result["protected_count"] == 2


@pytest.mark.asyncio
async def test_summarize_messages_llm_error(conversation_manager, mock_storage, mock_counter, sample_messages):
    """Test summarization handles LLM errors gracefully."""
    # Setup
    mock_storage.conversation.messages = sample_messages.copy()

    # Create mock LLM that raises an error
    mock_llm = MockLLMService()
    mock_llm.generate = AsyncMock(side_effect=Exception("LLM API error"))

    # Execute
    result = await conversation_manager.summarize_messages(
        llm_service=mock_llm,
        counter=mock_counter,
        message_ids=[1, 2, 3]
    )

    # Assert
    assert result["success"] is False
    assert "LLM API error" in result["error"]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.asyncio
async def test_get_messages_for_selection_no_conversation_store(mock_storage):
    """Test message selection when conversation store is not available."""
    # Setup - remove conversation store
    delattr(mock_storage, 'conversation')
    manager = ConversationManager(storage=mock_storage)

    # Execute
    messages = await manager.get_messages_for_selection(
        mode="last_n",
        criteria="5"
    )

    # Assert
    assert messages == []


@pytest.mark.asyncio
async def test_mark_messages_no_conversation_store(mock_storage):
    """Test marking messages when conversation store is not available."""
    # Setup
    delattr(mock_storage, 'conversation')
    manager = ConversationManager(storage=mock_storage)

    # Execute
    result = await manager.mark_messages(
        message_ids=[1],
        action="protect"
    )

    # Assert
    assert result["success"] is False
    assert "not available" in result["error"]


@pytest.mark.asyncio
async def test_exclude_messages_no_conversation_store(mock_storage):
    """Test excluding messages when conversation store is not available."""
    # Setup
    delattr(mock_storage, 'conversation')
    manager = ConversationManager(storage=mock_storage)

    # Execute
    result = await manager.exclude_messages(
        message_ids=[1],
        reason="test"
    )

    # Assert
    assert result["success"] is False
    assert "not available" in result["error"]


@pytest.mark.asyncio
async def test_restore_messages_no_conversation_store(mock_storage):
    """Test restoring messages when conversation store is not available."""
    # Setup
    delattr(mock_storage, 'conversation')
    manager = ConversationManager(storage=mock_storage)

    # Execute
    result = await manager.restore_messages()

    # Assert
    assert result["success"] is False
    assert "not available" in result["error"]


@pytest.mark.asyncio
async def test_summarize_messages_no_conversation_store(mock_storage, mock_counter):
    """Test summarizing messages when conversation store is not available."""
    # Setup
    delattr(mock_storage, 'conversation')
    manager = ConversationManager(storage=mock_storage)

    # Execute
    result = await manager.summarize_messages(
        llm_service=MockLLMService(),
        counter=mock_counter,
        message_ids=[1, 2]
    )

    # Assert
    assert result["success"] is False
    assert "not available" in result["error"]


@pytest.mark.asyncio
async def test_message_timestamp_parsing(conversation_manager):
    """Test internal timestamp parsing methods."""
    # Test _message_before
    msg_with_timestamp = {
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    cutoff = datetime.now(timezone.utc) + timedelta(hours=1)
    assert conversation_manager._message_before(msg_with_timestamp, cutoff) is True

    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    assert conversation_manager._message_before(msg_with_timestamp, cutoff) is False

    # Test _message_after
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    assert conversation_manager._message_after(msg_with_timestamp, cutoff) is True

    # Test message without timestamp
    msg_no_timestamp = {}
    assert conversation_manager._message_before(msg_no_timestamp, cutoff) is False
    assert conversation_manager._message_after(msg_no_timestamp, cutoff) is True

    # Test _message_in_range
    start = datetime.now(timezone.utc) - timedelta(hours=2)
    end = datetime.now(timezone.utc) + timedelta(hours=2)
    assert conversation_manager._message_in_range(msg_with_timestamp, start, end) is True


@pytest.mark.asyncio
async def test_get_conversation_store_hierarchical_fallback(mock_storage):
    """Test _get_conversation_store with hierarchical storage structure."""
    # Test with nested _storage attribute
    mock_storage._storage = Mock()
    mock_storage._storage.conversation = MockConversationStore()
    delattr(mock_storage, 'conversation')

    manager = ConversationManager(storage=mock_storage)
    conv_store = manager._get_conversation_store()

    assert conv_store is not None
    assert conv_store == mock_storage._storage.conversation
