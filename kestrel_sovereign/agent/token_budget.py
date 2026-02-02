"""
Token budget allocation for context management.

Manages token allocation across different context sources:
- System prompt (constitution, instructions)
- Conversation history
- Episode summaries (for long conversations)
- Memory retrieval (emotionally-weighted past)
- RAG documents

Based on approved allocation:
- System: 15%
- History: 40%
- Episodes: 20%
- Memories: 10%
- RAG: 15%
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .token_counter import TokenCounter, get_token_counter

logger = logging.getLogger(__name__)


# Default allocation percentages
DEFAULT_ALLOCATION = {
    "system": 0.15,     # Constitution, instructions, briefing
    "history": 0.40,    # Recent conversation messages
    "episodes": 0.20,   # Consolidated narrative summaries
    "memories": 0.10,   # Emotionally-weighted past memories
    "rag": 0.15,        # Document chunks from RAG
}

# Reserve tokens for response generation
RESPONSE_RESERVE = 1024


@dataclass
class TokenAllocation:
    """Token allocation for a specific context source."""
    name: str
    budget: int
    used: int = 0
    items: int = 0

    @property
    def remaining(self) -> int:
        """Tokens remaining in this allocation."""
        return max(0, self.budget - self.used)

    @property
    def utilization(self) -> float:
        """Percentage of budget used."""
        if self.budget == 0:
            return 0.0
        return self.used / self.budget


@dataclass
class TokenBudget:
    """
    Manages token budget allocation across context sources.

    Ensures total context never exceeds model's limit while
    distributing tokens according to configured percentages.
    """

    model: str
    counter: TokenCounter = field(init=False)
    context_limit: int = field(init=False)
    response_reserve: int = RESPONSE_RESERVE
    allocations: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize counter and compute allocations."""
        self.counter = get_token_counter(self.model)
        self.context_limit = self.counter.get_context_limit()
        self._compute_allocations()

    def _compute_allocations(self):
        """Compute token allocations based on percentages."""
        available = self.context_limit - self.response_reserve

        for name, percentage in DEFAULT_ALLOCATION.items():
            budget = int(available * percentage)
            self.allocations[name] = TokenAllocation(name=name, budget=budget)

        logger.debug(
            f"Token budget for {self.model}: {available} available "
            f"({self.context_limit} limit - {self.response_reserve} reserve)"
        )

    @property
    def system(self) -> int:
        """Budget for system prompt."""
        return self.allocations["system"].budget

    @property
    def history(self) -> int:
        """Budget for conversation history."""
        return self.allocations["history"].budget

    @property
    def episodes(self) -> int:
        """Budget for episode summaries."""
        return self.allocations["episodes"].budget

    @property
    def memories(self) -> int:
        """Budget for memory retrieval."""
        return self.allocations["memories"].budget

    @property
    def rag(self) -> int:
        """Budget for RAG documents."""
        return self.allocations["rag"].budget

    @property
    def total_budget(self) -> int:
        """Total available budget (excluding reserve)."""
        return self.context_limit - self.response_reserve

    @property
    def total_used(self) -> int:
        """Total tokens used across all allocations."""
        return sum(a.used for a in self.allocations.values())

    @property
    def total_remaining(self) -> int:
        """Total tokens remaining."""
        return self.total_budget - self.total_used

    def use(self, source: str, tokens: int, items: int = 1) -> bool:
        """
        Record token usage for a source.

        Args:
            source: Source name (system, history, episodes, memories, rag)
            tokens: Number of tokens used
            items: Number of items included

        Returns:
            True if usage was within budget, False if exceeded
        """
        if source not in self.allocations:
            logger.warning(f"Unknown source: {source}")
            return False

        allocation = self.allocations[source]
        allocation.used += tokens
        allocation.items += items

        if allocation.used > allocation.budget:
            logger.warning(
                f"{source} exceeded budget: {allocation.used}/{allocation.budget}"
            )
            return False

        return True

    def can_fit(self, source: str, tokens: int) -> bool:
        """
        Check if tokens can fit in a source's budget.

        Args:
            source: Source name
            tokens: Number of tokens to check

        Returns:
            True if tokens fit within remaining budget
        """
        if source not in self.allocations:
            return False
        return tokens <= self.allocations[source].remaining

    def get_remaining(self, source: str) -> int:
        """
        Get remaining tokens for a source.

        Args:
            source: Source name

        Returns:
            Remaining token budget
        """
        if source not in self.allocations:
            return 0
        return self.allocations[source].remaining

    def reallocate_unused(self):
        """
        Reallocate unused tokens from one source to others.

        Called after initial allocation to redistribute surplus.
        Prioritizes history and episodes.
        """
        # Collect unused tokens
        unused = sum(
            a.remaining for a in self.allocations.values()
            if a.name in ("memories", "rag")  # Take from less critical
        )

        if unused > 100:  # Only reallocate if meaningful
            # Give 70% to history, 30% to episodes
            history_boost = int(unused * 0.7)
            episodes_boost = int(unused * 0.3)

            self.allocations["history"].budget += history_boost
            self.allocations["episodes"].budget += episodes_boost

            logger.debug(
                f"Reallocated {unused} tokens: "
                f"+{history_boost} history, +{episodes_boost} episodes"
            )

    def get_summary(self) -> dict:
        """
        Get summary of all allocations.

        Returns:
            Dict with allocation details
        """
        return {
            "model": self.model,
            "context_limit": self.context_limit,
            "response_reserve": self.response_reserve,
            "total_budget": self.total_budget,
            "total_used": self.total_used,
            "allocations": {
                name: {
                    "budget": a.budget,
                    "used": a.used,
                    "remaining": a.remaining,
                    "items": a.items,
                    "utilization": f"{a.utilization:.1%}",
                }
                for name, a in self.allocations.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"TokenBudget(model={self.model}, "
            f"used={self.total_used}/{self.total_budget})"
        )


class AdaptiveTokenBudget(TokenBudget):
    """
    Adaptive token budget that adjusts based on conversation length.

    - Short conversations (<10 msgs): Simple allocation (more history)
    - Medium conversations (10-30 msgs): Standard allocation
    - Long conversations (>30 msgs): More episodes, less raw history
    """

    def __init__(self, model: str, message_count: int = 0):
        """
        Initialize adaptive budget.

        Args:
            model: Model name
            message_count: Number of messages in conversation
        """
        self.message_count = message_count
        super().__init__(model)

    def _compute_allocations(self):
        """Compute allocations based on conversation length."""
        available = self.context_limit - self.response_reserve

        if self.message_count < 10:
            # Short conversation: more history, minimal episodes
            allocation = {
                "system": 0.15,
                "history": 0.60,  # More history
                "episodes": 0.05,  # Minimal episodes
                "memories": 0.05,
                "rag": 0.15,
            }
        elif self.message_count < 30:
            # Medium: standard allocation
            allocation = DEFAULT_ALLOCATION
        else:
            # Long conversation: rely on episodes, trim history
            allocation = {
                "system": 0.15,
                "history": 0.25,  # Less raw history
                "episodes": 0.35,  # More episodes
                "memories": 0.10,
                "rag": 0.15,
            }

        for name, percentage in allocation.items():
            budget = int(available * percentage)
            self.allocations[name] = TokenAllocation(name=name, budget=budget)

        logger.debug(
            f"Adaptive budget for {self.message_count} messages: "
            f"history={allocation['history']:.0%}, "
            f"episodes={allocation['episodes']:.0%}"
        )


def create_budget(model: str, message_count: int = 0, adaptive: bool = True) -> TokenBudget:
    """
    Factory function to create appropriate token budget.

    Args:
        model: Model name
        message_count: Current message count in conversation
        adaptive: Whether to use adaptive allocation

    Returns:
        TokenBudget instance
    """
    if adaptive:
        return AdaptiveTokenBudget(model, message_count)
    return TokenBudget(model)
