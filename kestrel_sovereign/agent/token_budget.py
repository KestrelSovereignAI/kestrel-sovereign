"""
Token budget allocation for context management.

Manages token allocation across different context sources:
- System prompt (constitution, instructions)
- Conversation history
- Episode summaries (for long conversations)
- Memory retrieval (emotionally-weighted past)
- RAG documents

Two shapes are available:

- ``TokenBudget`` / ``AdaptiveTokenBudget`` — legacy static-percentage
  allocation (15 / 40 / 20 / 10 / 15). Kept for back-compat with all
  existing callers.
- ``ElasticTokenBudget`` — the #1309 contract Emma signed off on in
  PR #1306. Mandatory governance content is a **non-borrowable
  measured floor**; idle section budget flows to any over-demanded
  eligible section by turn-intent priority (default: history-first,
  then episodes, memories, rag); if the mandatory floor cannot fit
  for this model, the budget raises ``DegradedModeError`` — never
  silently drops governance content to make the call appear "normal"
  (Emma's 2026-05-20 hardening invariant).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

# Default turn-intent priority for elastic slack distribution. History
# is the default beneficiary because that is where the silent-prune
# correctness hole hurts most (Emma's 2026-05-20 review, addressed by
# C/#1311); RAG-heavy or memory-recall turns can legitimately deserve
# slack too, so callers can override.
DEFAULT_ELASTIC_PRIORITY: List[str] = ["history", "episodes", "memories", "rag"]


class DegradedModeError(Exception):
    """Raised when the mandatory governance floor cannot fit the
    model's context window.

    Emma's 2026-05-20 hardening invariant (PR #1306): mandatory
    governance content is **non-borrowable**. If the measured floor
    exceeds the available budget for an agent/model, that is a
    hard-failure / degraded-mode condition — never silently absorbed
    by crowding out history, RAG, memories, etc. Production callers
    must surface this loudly (UI, telemetry, structured warning)
    instead of pretending the call is "normal."
    """

    def __init__(
        self,
        mandatory_system_tokens: int,
        total_budget: int,
        model: str,
        *,
        detail: Optional[str] = None,
    ):
        self.mandatory_system_tokens = mandatory_system_tokens
        self.total_budget = total_budget
        self.model = model
        msg = (
            f"degraded-mode: mandatory governance floor "
            f"({mandatory_system_tokens} tokens) does not fit context "
            f"budget ({total_budget} tokens) for model {model!r}"
        )
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(msg)


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


class ElasticTokenBudget(AdaptiveTokenBudget):
    """Elastic budget per #1309 (epic #1307).

    Two contracts that distinguish this from the legacy adaptive shape:

    1. **Mandatory governance floor is non-borrowable.** The caller
       passes ``mandatory_system_tokens`` (measured per-agent /
       per-model — typically constitution + identity + state-of-mind +
       any AGENTS.md operator policy the agent must obey). That floor
       carves a hard quota out of the available budget before any
       section is allocated. ``can_fit("system", n)`` always allows up
       to the mandatory floor regardless of consumption elsewhere; the
       slack/elastic pool can never pull below the floor. If the floor
       exceeds the total budget, the constructor raises
       ``DegradedModeError`` — fail closed, never silently absorbed.
    2. **Idle section budget flows to over-demanded eligible
       sections.** As production assembly progresses through sections,
       ``mark_section_finalized(name)`` releases any unused budget into
       the elastic pool. Subsequent sections whose own remaining budget
       is short can borrow from the pool transparently via the normal
       ``can_fit`` / ``use`` API. The order in which sections borrow is
       the constructor's ``demand_priority`` (default
       ``DEFAULT_ELASTIC_PRIORITY`` — history first, then episodes,
       memories, rag). Operators can override per turn-intent (e.g. a
       RAG-heavy turn can pass ``["rag", "history", ...]``).
    """

    # Names that may not be trimmed to fund the mandatory floor and
    # may not appear in the elastic priority — the floor lives in
    # ``system``, so trimming ``system`` to fund itself is a bug.
    _NON_TRIMMABLE_SOURCES = frozenset({"system"})

    def __init__(
        self,
        model: str,
        message_count: int = 0,
        *,
        mandatory_system_tokens: int = 0,
        demand_priority: Optional[List[str]] = None,
    ):
        if mandatory_system_tokens < 0:
            raise ValueError("mandatory_system_tokens must be non-negative")
        self._mandatory_system_tokens = mandatory_system_tokens
        self._elastic_pool: int = 0
        self._finalized: set = set()
        # Validate priority: dedupe, strip non-trimmable / unknown
        # sources, then extend with any missing borrowable sources so
        # floor funding can always fall back to the full set (codex
        # round 1 #3). A caller passing ``["rag"]`` should not cause a
        # spurious DegradedModeError when the floor could fit by
        # trimming history/episodes/memories.
        raw = list(demand_priority) if demand_priority else list(DEFAULT_ELASTIC_PRIORITY)
        seen: set = set()
        cleaned: List[str] = []
        for name in raw:
            if (
                name in self._NON_TRIMMABLE_SOURCES
                or name in seen
                or name not in DEFAULT_ALLOCATION
            ):
                continue
            seen.add(name)
            cleaned.append(name)
        # Extend with any borrowable sections the caller omitted so
        # trim-for-floor has the full toolkit. Order is stable —
        # caller-specified sections retain priority.
        for name in DEFAULT_ELASTIC_PRIORITY:
            if name not in seen and name in DEFAULT_ALLOCATION:
                cleaned.append(name)
                seen.add(name)
        self._priority: List[str] = cleaned
        super().__init__(model, message_count)

    @property
    def mandatory_system_tokens(self) -> int:
        return self._mandatory_system_tokens

    @property
    def elastic_pool(self) -> int:
        return self._elastic_pool

    def _compute_allocations(self):
        """Adaptive allocation + non-borrowable mandatory system floor.

        After the adaptive percentages produce per-section budgets, the
        system slice is raised to ``max(system, mandatory_system_tokens)``.
        The bump (if any) is funded by trimming proportionally from
        eligible sections in the elastic priority list, in reverse
        order (so the lowest-priority section gives up budget first
        before history is asked to). If even after trimming every
        elastic source the mandatory floor still doesn't fit, raise
        ``DegradedModeError``.
        """
        super()._compute_allocations()
        available = self.total_budget
        if self._mandatory_system_tokens > available:
            raise DegradedModeError(
                self._mandatory_system_tokens,
                available,
                self.model,
                detail="mandatory floor exceeds the entire post-reserve budget",
            )
        system_alloc = self.allocations["system"]
        if self._mandatory_system_tokens > system_alloc.budget:
            shortfall = self._mandatory_system_tokens - system_alloc.budget
            # Trim from lowest priority first (last entries of
            # ``self._priority``). The default order is
            # ["history", "episodes", "memories", "rag"]; trimming
            # reverse-order means rag → memories → episodes → history.
            for source in reversed(self._priority):
                if shortfall <= 0:
                    break
                if source not in self.allocations:
                    continue
                give = min(shortfall, self.allocations[source].budget)
                self.allocations[source].budget -= give
                shortfall -= give
            if shortfall > 0:
                raise DegradedModeError(
                    self._mandatory_system_tokens,
                    available,
                    self.model,
                    detail=(
                        f"after trimming elastic sections, {shortfall} tokens "
                        "still short of the mandatory floor"
                    ),
                )
            system_alloc.budget = self._mandatory_system_tokens

    def can_fit(self, source: str, tokens: int) -> bool:
        """Allow borrowing from the elastic pool when the source's own
        remaining is short.

        The system slice never borrows below the mandatory floor: any
        ``use("system", …)`` past ``mandatory_system_tokens`` will be
        treated as borrowable but the floor itself stays protected.
        """
        if source not in self.allocations:
            return False
        local = self.allocations[source].remaining
        if tokens <= local:
            return True
        deficit = tokens - local
        # The pool is the sum of (1) explicitly-released slack from
        # finalized sections and (2) currently-unused budget from
        # sections we haven't entered yet (those not in finalized AND
        # not in `source` and below their priority). The simpler model:
        # we only let sections borrow from the explicitly-released pool.
        # That's predictable and matches the "mark_section_finalized"
        # API. Sections that haven't been entered yet keep their full
        # quota for themselves.
        return deficit <= self._elastic_pool

    def use(self, source: str, tokens: int, items: int = 1) -> bool:
        """Record usage; auto-borrow from the elastic pool when needed.

        Returns True when the tokens were accepted (either from the
        section's own quota or from the elastic pool). Returns False
        only when neither the section quota nor the elastic pool can
        cover the request — that's the signal to caller code that the
        section must be skipped (matches the legacy ``TokenBudget.use``
        return semantics).
        """
        if source not in self.allocations:
            logger.warning(f"Unknown source: {source}")
            return False
        allocation = self.allocations[source]
        local = allocation.remaining
        if tokens <= local:
            allocation.used += tokens
            allocation.items += items
            return True
        # Need to borrow from the pool
        deficit = tokens - local
        if deficit > self._elastic_pool:
            logger.warning(
                f"{source} cannot fit {tokens}: local remaining {local}, "
                f"elastic pool {self._elastic_pool}"
            )
            return False
        # Take the deficit from the pool — bump the section's budget so
        # used / remaining math stays consistent for callers.
        self._elastic_pool -= deficit
        allocation.budget += deficit
        allocation.used += tokens
        allocation.items += items
        logger.debug(
            f"{source} borrowed {deficit} tokens from elastic pool "
            f"(pool now {self._elastic_pool})"
        )
        return True

    def mark_section_finalized(self, source: str) -> int:
        """Release ``source``'s unused budget into the elastic pool.

        Production assembly calls this when it moves past a section so
        later sections can borrow the slack. The system section
        specifically respects the mandatory floor: slack released is
        ``budget - max(used, mandatory_system_tokens)`` — the floor's
        bytes are never returned to the pool, even when system used
        less than the floor (which can happen on agents with very
        small constitutions; the floor is still a guarantee, not a
        consumption record).

        ``allocation.budget`` is **not** mutated by this call — the
        original adaptive allocation stays visible in the summary so
        operators can see what was planned vs what was spent
        (``used``) vs what was redistributed (``elastic_pool_remaining``).

        Returns the number of tokens released into the pool.
        """
        if source not in self.allocations:
            return 0
        if source in self._finalized:
            return 0
        self._finalized.add(source)
        allocation = self.allocations[source]
        if source == "system":
            protected = max(allocation.used, self._mandatory_system_tokens)
        else:
            protected = allocation.used
        slack = max(0, allocation.budget - protected)
        if slack > 0:
            self._elastic_pool += slack
            logger.debug(
                f"{source} released {slack} tokens into elastic pool "
                f"(pool now {self._elastic_pool})"
            )
        return slack

    def effective_budget(self, source: str) -> int:
        """Spend-ceiling for ``source`` including the elastic pool.

        Callers that size a downstream operation against the budget
        (for example, ``format_conversation_history(max_tokens=…)``)
        should pass this instead of ``budget.<source>`` so the section
        can actually grow into the slack released by earlier
        finalized sections.

        Without this method the legacy
        ``format_conversation_history(max_tokens=budget.history)`` call
        would never *ask* for more messages than the static history
        slice, so released slack would sit unused — defeating the
        whole #1309 contract. Codex round 1 #2 caught this.
        """
        if source not in self.allocations:
            return 0
        return self.allocations[source].remaining + self._elastic_pool

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary["mandatory_system_tokens"] = self._mandatory_system_tokens
        summary["elastic_pool_remaining"] = self._elastic_pool
        summary["elastic_priority"] = list(self._priority)
        summary["finalized_sections"] = sorted(self._finalized)
        return summary


def create_budget(
    model: str,
    message_count: int = 0,
    adaptive: bool = True,
    *,
    elastic: bool = False,
    mandatory_system_tokens: int = 0,
    demand_priority: Optional[List[str]] = None,
) -> TokenBudget:
    """
    Factory function to create appropriate token budget.

    Args:
        model: Model name
        message_count: Current message count in conversation
        adaptive: Whether to use adaptive allocation (legacy contract)
        elastic: When True, return an ``ElasticTokenBudget`` honoring
            the non-borrowable mandatory floor and slack redistribution
            described in #1309. Implies ``adaptive=True`` (mandatory
            content needs the conversation-length-aware splits).
        mandatory_system_tokens: Measured non-borrowable floor for
            mandatory governance content. Ignored when ``elastic`` is
            False.
        demand_priority: Override the elastic slack-distribution order.
            Ignored when ``elastic`` is False.

    Returns:
        TokenBudget instance
    """
    if elastic:
        return ElasticTokenBudget(
            model,
            message_count,
            mandatory_system_tokens=mandatory_system_tokens,
            demand_priority=demand_priority,
        )
    if adaptive:
        return AdaptiveTokenBudget(model, message_count)
    return TokenBudget(model)
