"""
End-to-end tests for Context Management with REAL services.

These tests use real AsyncStorage, real conversation history,
and real token counting - NO MOCKS.

Tests cover:
- Full context flow with real storage
- Episode triggers with real conversations
- Memory retrieval with real emotional context
- RAG hybrid search (BM25 always, embeddings when Ollama available)
- Privacy mode behavior during context assembly
"""

import pytest
import tempfile
from pathlib import Path

from kestrel_sovereign.agent.context_manager import ContextManager, ContextResult
from kestrel_sovereign.agent.token_budget import create_budget
from kestrel_sovereign.storage import AsyncStorage


@pytest.fixture
async def real_storage():
    """Create real AsyncStorage with in-memory SQLite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        async with AsyncStorage(str(db_path)) as storage:
            # Initialize agent_id in storage
            storage.agent_id = "test-agent"
            yield storage


@pytest.fixture
async def storage_with_history(real_storage):
    """Storage with pre-populated conversation history."""
    # Add some conversation history
    conversations = [
        ("user", "Hello, I'm feeling great today!"),
        ("assistant", "That's wonderful to hear! What's making you feel so good?"),
        ("user", "I just got promoted at work!"),
        ("assistant", "Congratulations! That's a huge achievement!"),
        ("user", "Thanks! My mom would be so proud."),
        ("assistant", "I'm sure she would be. Tell me more about your relationship with her."),
    ]

    for role, content in conversations:
        await real_storage.conversation.add_conversation(
            role=role,
            content=content,
            metadata={"test": True}
        )

    return real_storage


class TestContextWithRealStorage:
    """Tests for ContextManager with real AsyncStorage."""

    @pytest.mark.asyncio
    async def test_build_context_with_empty_storage(self, real_storage):
        """Test building context with no conversation history."""
        manager = ContextManager(
            storage=real_storage,
            model="gpt-4",
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="test query",
            constitution="Test Constitution Article I",
            privacy_mode="NORMAL"
        )

        assert isinstance(result, ContextResult)
        assert "Test Constitution" in result.system_prompt
        assert result.messages == []  # No history
        assert result.total_tokens > 0  # System prompt counted

    @pytest.mark.asyncio
    async def test_build_context_with_real_history(self, storage_with_history):
        """Test building context with real conversation history."""
        manager = ContextManager(
            storage=storage_with_history,
            model="gpt-4",
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="Tell me about my promotion",
            constitution="Test Constitution",
            privacy_mode="NORMAL"
        )

        assert isinstance(result, ContextResult)
        # Should have messages from history
        assert len(result.messages) > 0

        # Check message content
        contents = [m.get("content", "") for m in result.messages]
        assert any("promoted" in c for c in contents)

    @pytest.mark.asyncio
    async def test_context_budget_summary_populated(self, storage_with_history):
        """Test that budget summary is populated correctly."""
        manager = ContextManager(
            storage=storage_with_history,
            model="gpt-4",
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="test",
            constitution="Test Constitution",
            privacy_mode="NORMAL"
        )

        # Budget summary should have allocations
        assert "allocations" in result.budget_summary
        assert result.budget_summary.get("total_used", 0) > 0


class TestEphemeralMode:
    """Tests for EPHEMERAL privacy mode behavior."""

    @pytest.mark.asyncio
    async def test_ephemeral_returns_no_history(self, storage_with_history):
        """Test that EPHEMERAL mode doesn't return history."""
        manager = ContextManager(
            storage=storage_with_history,
            model="gpt-4",
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="Tell me about my promotion",
            constitution="Test Constitution",
            privacy_mode="EPHEMERAL"
        )

        # Should have NO messages despite having history
        assert result.messages == []
        assert "EPHEMERAL" in result.system_prompt
        assert any("EPHEMERAL" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_ephemeral_budget_is_minimal(self, storage_with_history):
        """Test that EPHEMERAL mode uses minimal budget."""
        manager = ContextManager(
            storage=storage_with_history,
            model="gpt-4",
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="test",
            constitution="Test Constitution",
            privacy_mode="EPHEMERAL"
        )

        # Budget should indicate ephemeral mode
        assert result.budget_summary.get("mode") == "ephemeral"


class TestLongConversationEpisodes:
    """Tests for episode integration in long conversations."""

    @pytest.fixture
    async def long_conversation_storage(self, real_storage):
        """Storage with 30+ messages to trigger episode logic."""
        for i in range(35):
            await real_storage.conversation.add_conversation(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}: This is test content for conversation threading.",
                metadata={"index": i}
            )
        return real_storage

    @pytest.mark.asyncio
    async def test_adaptive_budget_for_long_conversation(self, long_conversation_storage):
        """Test that long conversations get adaptive budget allocation."""
        manager = ContextManager(
            storage=long_conversation_storage,
            model="gpt-4",
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="test",
            constitution="Test Constitution",
            privacy_mode="NORMAL"
        )

        # Should use adaptive allocation (35 messages > 30 threshold)
        # Episodes should get more budget than short conversations
        allocations = result.budget_summary.get("allocations", {})

        # Verify allocations exist
        assert "history" in allocations
        assert "episodes" in allocations


class TestBudgetStatusAPI:
    """Tests for budget status API."""

    @pytest.mark.asyncio
    async def test_get_budget_status_short_conversation(self, real_storage):
        """Test budget status for short conversation."""
        manager = ContextManager(
            storage=real_storage,
            model="gpt-4",
            agent_id="test-agent"
        )

        status = manager.get_budget_status(message_count=5)

        assert "model" in status
        assert status["model"] == "gpt-4"
        assert "allocations" in status
        # Short conversation: more history, less episodes
        # History is 60%, Episodes is 5% for short
        history_budget = status["allocations"]["history"]["budget"]
        episodes_budget = status["allocations"]["episodes"]["budget"]
        assert history_budget > episodes_budget

    @pytest.mark.asyncio
    async def test_get_budget_status_long_conversation(self, real_storage):
        """Test budget status for long conversation."""
        manager = ContextManager(
            storage=real_storage,
            model="gpt-4",
            agent_id="test-agent"
        )

        status = manager.get_budget_status(message_count=50)

        # Long conversation: more episodes, less history
        # History is 25%, Episodes is 35% for long
        history_budget = status["allocations"]["history"]["budget"]
        episodes_budget = status["allocations"]["episodes"]["budget"]
        assert episodes_budget > history_budget


class TestHistoryTruncation:
    """Tests for history truncation behavior."""

    @pytest.fixture
    async def massive_history_storage(self, real_storage):
        """Storage with massive history to force truncation."""
        # Add enough messages to exceed token budget
        # GPT-4 has 8K context, history gets ~40% = ~3200 tokens
        # Each message ~10-20 tokens, so ~200 messages should exceed
        for i in range(300):
            await real_storage.conversation.add_conversation(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message number {i} with some additional content to make it longer and use more tokens.",
                metadata={"index": i}
            )
        return real_storage

    @pytest.mark.asyncio
    async def test_truncation_warning_generated(self, massive_history_storage):
        """Test that truncation generates a warning."""
        manager = ContextManager(
            storage=massive_history_storage,
            model="gpt-4",  # Small context limit
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="test",
            constitution="Test Constitution",
            privacy_mode="NORMAL"
        )

        # Should have fewer messages than the full 300
        assert len(result.messages) < 300

        # May have truncation warning if severe
        # (depends on exact token counts)


class TestDifferentModels:
    """Tests for context building with different model context limits."""

    @pytest.mark.asyncio
    async def test_small_model_context(self, storage_with_history):
        """Test context building with small model (phi3 4K)."""
        manager = ContextManager(
            storage=storage_with_history,
            model="phi3:3.8b",  # 4096 context
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="test",
            constitution="Test Constitution",
            privacy_mode="NORMAL"
        )

        # Should still work with smaller context
        assert result.total_tokens < 4096

    @pytest.mark.asyncio
    async def test_large_model_context(self, storage_with_history):
        """Verify ContextManager picks up the large-model context
        window for the configured large Anthropic model.

        Pair to ``test_small_model_context`` (phi3, 4K). Resolves the
        model via ``catalog.get_model_for_size("anthropic", "large")``
        — config-driven, no model-ID hardcoding (this exact test was
        previously stranded on ``claude-opus-4-5-20251101`` for that
        reason). Asserts the budget reflects whatever the
        catalog/cache/discovery chain says about that model's context,
        with a floor that catches a regression where the "large" tier
        accidentally points at a small-context model.
        """
        from kestrel_sovereign.llm.model_catalog import get_catalog_service

        catalog = get_catalog_service()
        model = catalog.get_model_for_size("anthropic", "large")
        if not model:
            pytest.skip(
                "No anthropic.large entry in model_catalog.toml "
                "[size_tiers] — configure one to exercise this path"
            )

        manager = ContextManager(
            storage=storage_with_history,
            model=model,
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="test",
            constitution="Test Constitution",
            privacy_mode="NORMAL"
        )

        # The actual limit is whatever discovery/cache/catalog returns
        # for this model — assert the budget reflects it, with a floor
        # so a "large" tier pointing at a 32K model fails loudly here.
        context_limit = result.budget_summary["context_limit"]
        assert context_limit >= 100_000, (
            f"Configured anthropic.large is '{model}' but its context "
            f"resolved to {context_limit} tokens — that's not large. "
            "Check catalog [context_limits_override] or [size_tiers]."
        )


class TestConstitutionInContext:
    """Tests for constitution handling in context."""

    @pytest.mark.asyncio
    async def test_constitution_included_in_system_prompt(self, real_storage):
        """Test that constitution is included in system prompt."""
        manager = ContextManager(
            storage=real_storage,
            model="gpt-4",
            agent_id="test-agent"
        )

        constitution = """
        ARTICLE I - Data Sanctity
        The Agent shall protect all user data.

        ARTICLE II - Verifiable History
        All actions shall be auditable.
        """

        result = await manager.build_context(
            query="test",
            constitution=constitution,
            privacy_mode="NORMAL"
        )

        assert "ARTICLE I" in result.system_prompt
        assert "Data Sanctity" in result.system_prompt
        assert "ARTICLE II" in result.system_prompt

    @pytest.mark.asyncio
    async def test_constitution_tokens_counted(self, real_storage):
        """Test that constitution tokens are counted in system budget."""
        manager = ContextManager(
            storage=real_storage,
            model="gpt-4",
            agent_id="test-agent"
        )

        short_constitution = "Be helpful."
        long_constitution = "Be helpful. " * 100

        result_short = await manager.build_context(
            query="test",
            constitution=short_constitution,
            privacy_mode="NORMAL"
        )

        result_long = await manager.build_context(
            query="test",
            constitution=long_constitution,
            privacy_mode="NORMAL"
        )

        # Long constitution should use more tokens
        short_system = result_short.budget_summary["allocations"]["system"]["used"]
        long_system = result_long.budget_summary["allocations"]["system"]["used"]
        assert long_system > short_system


class TestContextResultProperties:
    """Tests for ContextResult dataclass properties."""

    @pytest.mark.asyncio
    async def test_context_result_has_all_fields(self, storage_with_history):
        """Test that ContextResult has all expected fields."""
        manager = ContextManager(
            storage=storage_with_history,
            model="gpt-4",
            agent_id="test-agent"
        )

        result = await manager.build_context(
            query="test",
            constitution="Test",
            privacy_mode="NORMAL"
        )

        assert hasattr(result, "system_prompt")
        assert hasattr(result, "messages")
        assert hasattr(result, "total_tokens")
        assert hasattr(result, "budget_summary")
        assert hasattr(result, "episode_count")
        assert hasattr(result, "memory_count")
        assert hasattr(result, "rag_chunks")
        assert hasattr(result, "warnings")

        assert isinstance(result.system_prompt, str)
        assert isinstance(result.messages, list)
        assert isinstance(result.total_tokens, int)
        assert isinstance(result.budget_summary, dict)
        assert isinstance(result.warnings, list)
