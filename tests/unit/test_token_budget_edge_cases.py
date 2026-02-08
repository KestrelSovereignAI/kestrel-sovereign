"""
Tests for token budget edge cases.

Tests cover:
- Budget overflow handling and warnings
- Token reallocation from unused sources
- Cross-model budget differences (GPT-4 vs Claude vs Gemini)
- Model switching mid-conversation
- Concurrent budget allocation
"""

import pytest
import asyncio
from kestrel_sovereign.agent.token_budget import (
    TokenBudget, AdaptiveTokenBudget, TokenAllocation,
    create_budget, DEFAULT_ALLOCATION, RESPONSE_RESERVE
)
from kestrel_sovereign.agent.token_counter import get_token_counter


class TestBudgetOverflow:
    """Tests for budget overflow scenarios."""

    def test_use_returns_false_when_exceeds_budget(self):
        """Test that use() returns False when exceeding budget."""
        budget = TokenBudget("gpt-4")
        history_budget = budget.allocations["history"].budget

        # First use within budget
        assert budget.use("history", history_budget - 100) is True

        # Second use exceeds budget
        assert budget.use("history", 200) is False

    def test_overflow_is_tracked_accurately(self):
        """Test that overflow amount is tracked correctly."""
        budget = TokenBudget("gpt-4")
        initial_budget = budget.allocations["history"].budget

        # Exceed by 500 tokens
        budget.use("history", initial_budget + 500)

        assert budget.allocations["history"].used == initial_budget + 500
        assert budget.allocations["history"].remaining == 0

    def test_can_fit_respects_remaining(self):
        """Test can_fit() correctly checks remaining capacity."""
        budget = TokenBudget("gpt-4")
        budget.use("history", budget.allocations["history"].budget - 100)

        assert budget.can_fit("history", 100) is True
        assert budget.can_fit("history", 101) is False
        assert budget.can_fit("history", 0) is True

    def test_multiple_sources_overflow_independently(self):
        """Test that overflow in one source doesn't affect others."""
        budget = TokenBudget("gpt-4")

        # Overflow history
        budget.use("history", budget.allocations["history"].budget + 1000)

        # Other sources should still have capacity
        assert budget.allocations["system"].remaining > 0
        assert budget.allocations["episodes"].remaining > 0
        assert budget.allocations["rag"].remaining > 0


class TestReallocateUnused:
    """Tests for token reallocation from unused sources."""

    def test_reallocate_unused_boosts_history_and_episodes(self):
        """Test that unused tokens get redistributed to history and episodes."""
        budget = TokenBudget("gpt-4")

        # Use up system, leave memories and rag unused
        budget.use("system", budget.allocations["system"].budget)

        # Track initial allocations
        initial_history = budget.allocations["history"].budget
        initial_episodes = budget.allocations["episodes"].budget

        # Get unused from memories and rag
        unused_memories = budget.allocations["memories"].remaining
        unused_rag = budget.allocations["rag"].remaining
        total_unused = unused_memories + unused_rag

        # Reallocate
        budget.reallocate_unused()

        # If there was enough unused, should see boost
        if total_unused > 100:
            # History gets 70%, episodes gets 30%
            expected_history_boost = int(total_unused * 0.7)
            expected_episodes_boost = int(total_unused * 0.3)

            assert budget.allocations["history"].budget >= initial_history + expected_history_boost - 1
            assert budget.allocations["episodes"].budget >= initial_episodes + expected_episodes_boost - 1

    def test_reallocate_unused_ignores_small_amounts(self):
        """Test that reallocation skips if unused < 100 tokens."""
        budget = TokenBudget("gpt-4")

        # Use almost everything
        budget.use("memories", budget.allocations["memories"].budget - 50)
        budget.use("rag", budget.allocations["rag"].budget - 50)

        initial_history = budget.allocations["history"].budget

        budget.reallocate_unused()

        # With only 100 tokens unused (< threshold), history should be unchanged
        assert budget.allocations["history"].budget == initial_history

    def test_reallocate_unused_only_takes_from_memories_and_rag(self):
        """Test that reallocation only pulls from memories and rag, not system."""
        budget = TokenBudget("gpt-4")

        # Leave system unused (should not be taken)
        system_remaining = budget.allocations["system"].remaining

        budget.reallocate_unused()

        # System should still have its full remaining (not taken from)
        assert budget.allocations["system"].remaining == system_remaining


class TestCrossModelBudgets:
    """Tests for budget differences across model context limits."""

    def test_gpt4_vs_claude_budget_allocation(self):
        """Test that Claude's larger context yields larger budgets."""
        gpt4_budget = create_budget("gpt-4", message_count=50)
        claude_budget = create_budget("claude-opus-4-5-20251101", message_count=50)

        # Claude has 1M context vs GPT-4's 8K
        # All allocations should be proportionally larger
        assert claude_budget.allocations["history"].budget > gpt4_budget.allocations["history"].budget
        assert claude_budget.allocations["episodes"].budget > gpt4_budget.allocations["episodes"].budget
        assert claude_budget.allocations["rag"].budget > gpt4_budget.allocations["rag"].budget

    def test_gemini_has_largest_budget(self):
        """Test that Gemini 3 Pro's 2M context yields largest budgets."""
        gpt4_budget = create_budget("gpt-4", message_count=50)
        gemini_budget = create_budget("gemini-3-pro", message_count=50)

        # Gemini has 2M context (from TOML), gpt-4 gets DEFAULT_CONTEXT_LIMIT
        assert gemini_budget.total_budget > gpt4_budget.total_budget

    def test_small_context_model_fits_allocation(self):
        """Test that small context models (phi3:3.8b 4K) still work."""
        # phi3 has 4096 context
        budget = create_budget("phi3:3.8b", message_count=5)

        # Total available = 4096 - 1024 (reserve) = 3072
        assert budget.total_budget == 4096 - RESPONSE_RESERVE

        # All allocations should be positive
        for alloc in budget.allocations.values():
            assert alloc.budget > 0

    def test_unknown_model_uses_default_context(self):
        """Test that unknown models get default 8192 context."""
        from kestrel_sovereign.agent.token_counter import DEFAULT_CONTEXT_LIMIT

        budget = create_budget("totally-made-up-model", message_count=10)

        # Should use default limit
        assert budget.context_limit == DEFAULT_CONTEXT_LIMIT


class TestModelSwitching:
    """Tests for scenarios where model changes mid-conversation."""

    def test_budget_recalculated_on_model_change(self):
        """Test that creating new budget with different model recalculates."""
        gpt4_budget = create_budget("gpt-4", message_count=50)
        gpt4_history = gpt4_budget.allocations["history"].budget

        # "Switch" to Claude by creating new budget
        claude_budget = create_budget("claude-opus-4-5-20251101", message_count=50)
        claude_history = claude_budget.allocations["history"].budget

        # Should be different
        assert claude_history != gpt4_history

    def test_used_tokens_dont_transfer(self):
        """Test that used tokens from old budget don't transfer to new."""
        old_budget = create_budget("gpt-4", message_count=50)
        old_budget.use("history", 1000)

        new_budget = create_budget("gpt-5", message_count=50)

        # New budget should be fresh
        assert new_budget.total_used == 0
        assert new_budget.allocations["history"].used == 0

    def test_counter_model_change(self):
        """Test that TokenCounter handles model switching."""
        counter1 = get_token_counter("gpt-4")
        counter2 = get_token_counter("claude-opus-4-5-20251101")

        # Different models, different limits
        assert counter1.get_context_limit() != counter2.get_context_limit()

        # Same text, potentially different counts (both use cl100k_base for estimation)
        text = "Hello, world!" * 100
        count1 = counter1.count(text)
        count2 = counter2.count(text)
        # Counts might be similar since both fall back to cl100k_base
        assert count1 > 0
        assert count2 > 0


class TestConcurrentBudgetAllocation:
    """Tests for concurrent access to token budgets."""

    @pytest.mark.asyncio
    async def test_concurrent_use_calls(self):
        """Test that concurrent use() calls are handled safely."""
        budget = TokenBudget("gpt-4")

        async def use_tokens(amount: int):
            return budget.use("history", amount)

        # Run 10 concurrent use() calls
        results = await asyncio.gather(*[use_tokens(100) for _ in range(10)])

        # Total used should be 1000
        assert budget.allocations["history"].used == 1000

        # Some may have succeeded, some may have exceeded
        total_history = budget.allocations["history"].budget
        if total_history >= 1000:
            assert all(r is True for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_can_fit_checks(self):
        """Test that concurrent can_fit() calls return consistent results."""
        budget = TokenBudget("gpt-4")
        remaining = budget.get_remaining("history")

        async def check_fit(amount: int):
            return budget.can_fit("history", amount)

        # All should be consistent
        results = await asyncio.gather(*[check_fit(remaining - 10) for _ in range(10)])
        assert all(r is True for r in results)

        results = await asyncio.gather(*[check_fit(remaining + 10) for _ in range(10)])
        assert all(r is False for r in results)


class TestBudgetSummary:
    """Tests for budget summary and reporting."""

    def test_summary_includes_all_allocations(self):
        """Test that get_summary() includes all allocation types."""
        budget = TokenBudget("gpt-4")
        budget.use("system", 100)
        budget.use("history", 200, items=5)

        summary = budget.get_summary()

        assert summary["model"] == "gpt-4"
        assert summary["context_limit"] == 32768
        assert summary["total_used"] == 300

        # Check allocations
        assert "system" in summary["allocations"]
        assert "history" in summary["allocations"]
        assert summary["allocations"]["history"]["used"] == 200
        assert summary["allocations"]["history"]["items"] == 5

    def test_utilization_percentage_format(self):
        """Test that utilization is formatted as percentage string."""
        budget = TokenBudget("gpt-4")
        budget.use("history", budget.allocations["history"].budget // 2)

        summary = budget.get_summary()

        # Should be around 50%
        util_str = summary["allocations"]["history"]["utilization"]
        assert "%" in util_str
        assert "50" in util_str or "49" in util_str or "51" in util_str


class TestAdaptiveBudgetEdgeCases:
    """Tests for AdaptiveTokenBudget edge cases."""

    def test_zero_messages_uses_short_allocation(self):
        """Test that 0 messages uses short conversation allocation."""
        budget = AdaptiveTokenBudget("gpt-4", message_count=0)

        # Short conversation: 60% history, 5% episodes
        total = budget.total_budget
        assert budget.allocations["history"].budget == int(total * 0.60)
        assert budget.allocations["episodes"].budget == int(total * 0.05)

    def test_boundary_at_10_messages(self):
        """Test allocation changes at exactly 10 messages."""
        budget_9 = AdaptiveTokenBudget("gpt-4", message_count=9)
        budget_10 = AdaptiveTokenBudget("gpt-4", message_count=10)

        # At 9: short conversation (60% history)
        # At 10: medium conversation (40% history)
        assert budget_9.allocations["history"].budget > budget_10.allocations["history"].budget

    def test_boundary_at_30_messages(self):
        """Test allocation changes at exactly 30 messages."""
        budget_29 = AdaptiveTokenBudget("gpt-4", message_count=29)
        budget_30 = AdaptiveTokenBudget("gpt-4", message_count=30)

        # At 29: medium conversation (40% history, 20% episodes)
        # At 30: long conversation (25% history, 35% episodes)
        assert budget_29.allocations["history"].budget > budget_30.allocations["history"].budget
        assert budget_29.allocations["episodes"].budget < budget_30.allocations["episodes"].budget

    def test_very_long_conversation(self):
        """Test allocation with very long conversation (10K messages)."""
        budget = AdaptiveTokenBudget("gpt-4", message_count=10000)

        # Should use long conversation allocation (same as 30+)
        total = budget.total_budget
        assert budget.allocations["history"].budget == int(total * 0.25)
        assert budget.allocations["episodes"].budget == int(total * 0.35)


class TestTokenAllocationProperties:
    """Tests for TokenAllocation dataclass properties."""

    def test_remaining_with_partial_use(self):
        """Test remaining calculation with partial use."""
        alloc = TokenAllocation(name="test", budget=1000, used=750)
        assert alloc.remaining == 250

    def test_remaining_with_full_use(self):
        """Test remaining is 0 when fully used."""
        alloc = TokenAllocation(name="test", budget=1000, used=1000)
        assert alloc.remaining == 0

    def test_remaining_with_overflow(self):
        """Test remaining is 0 when overflowed (never negative)."""
        alloc = TokenAllocation(name="test", budget=1000, used=1500)
        assert alloc.remaining == 0

    def test_utilization_with_zero_use(self):
        """Test utilization is 0.0 when nothing used."""
        alloc = TokenAllocation(name="test", budget=1000, used=0)
        assert alloc.utilization == 0.0

    def test_utilization_with_full_use(self):
        """Test utilization is 1.0 when fully used."""
        alloc = TokenAllocation(name="test", budget=1000, used=1000)
        assert alloc.utilization == 1.0

    def test_utilization_with_overflow(self):
        """Test utilization > 1.0 when overflowed."""
        alloc = TokenAllocation(name="test", budget=1000, used=1500)
        assert alloc.utilization == 1.5
