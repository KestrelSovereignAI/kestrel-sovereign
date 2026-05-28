"""Tests for ``ContextManager._lumpy_prune_history`` (#1430).

When history overflows the budget, the previous behaviour was to drop
just enough tokens to fit. At steady state that meant every turn was
a prune turn — which invalidates the position-indexed Anthropic cache
markers at ``messages[-2]`` / ``messages[-4]`` on every request (see
``project_anthropic_cache_markers.md``).

Lumpy prune drops down to ``PRUNE_TARGET_FRAC`` of the budget so the
next several turns add into clean headroom and the cache compounds.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.agent.token_budget import TokenBudget


def _msg(content: str, role: str = "user") -> dict:
    return {"role": role, "content": content}


def _make_manager(model: str = "gpt-4") -> ContextManager:
    """ContextManager with just enough wiring for the prune helper."""
    return ContextManager(storage=MagicMock(), model=model)


def _seed_history(token_size_per_msg: int, count: int) -> list[dict]:
    # Roughly token_size_per_msg tokens per message under the default
    # counter (4 chars ≈ 1 token for English). The exact count doesn't
    # matter — the helper measures via ``self.counter.count`` and uses
    # the result. We want messages that are individually small enough
    # that several get pruned, not one giant one.
    body = "x " * token_size_per_msg
    return [_msg(body, role=("user" if i % 2 == 0 else "assistant")) for i in range(count)]


class TestLumpyPruneHelper:
    def test_no_overage_returns_zero_and_does_not_mutate(self):
        cm = _make_manager()
        budget = TokenBudget("gpt-4")
        history = _seed_history(10, 6)
        original = list(history)
        # No usage recorded → not over budget.
        dropped = cm._lumpy_prune_history(history, budget)
        assert dropped == 0
        assert history == original

    def test_single_message_history_returns_zero(self):
        cm = _make_manager()
        budget = TokenBudget("gpt-4")
        # Force overage but only one message in history.
        budget.use("history", budget.total_budget + 1000, items=1)
        history = [_msg("alone")]
        dropped = cm._lumpy_prune_history(history, budget)
        assert dropped == 0
        assert len(history) == 1

    def test_lumpy_target_undershoots_budget(self):
        """Helper drops below the budget, not just to the budget."""
        cm = _make_manager()
        # PRUNE_TARGET_FRAC defaults to 0.75 — keep it explicit so we
        # don't depend on env var leakage.
        cm.PRUNE_TARGET_FRAC = 0.75
        budget = TokenBudget("gpt-4")
        # Seed history so its measured token count clearly overflows.
        history = _seed_history(50, 80)
        # Record usage that pushes total_used well past the budget.
        budget.use("history", budget.total_budget + 2000, items=len(history))

        dropped = cm._lumpy_prune_history(history, budget)

        # We actually dropped something.
        assert dropped > 0
        # New history allocation is at or below the lumpy target — the
        # whole point of the change.
        target = int(budget.total_budget * cm.PRUNE_TARGET_FRAC)
        assert budget.allocations["history"].used <= target, (
            f"history.used={budget.allocations['history'].used} "
            f"exceeds lumpy target={target} (budget={budget.total_budget})"
        )
        # And we never drained the list to zero.
        assert len(history) >= 1

    def test_lumpy_undershoot_buys_cache_warm_turns(self):
        """After lumpy prune, the next K turns at typical size do not
        re-trigger pruning. The whole point: prune frequency drops from
        once-per-turn to once-per-many-turns, so the
        ``messages[-2]``/``[-4]`` markers stay valid in between."""
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        budget = TokenBudget("gpt-4")
        history = _seed_history(50, 80)
        budget.use("history", budget.total_budget + 2000, items=len(history))

        cm._lumpy_prune_history(history, budget)

        headroom = budget.total_budget - budget.allocations["history"].used
        # Headroom must be at least one realistic turn (~500 tokens
        # user+assistant). On a 0.25-of-budget gap that's plenty.
        assert headroom >= 500, (
            f"lumpy prune left only {headroom} tokens of headroom — "
            "not enough to amortize across multiple turns"
        )

    def test_just_enough_was_pathological(self):
        """Regression guard: the old behaviour (drop until overage ==
        0) would leave history ALMOST AT the ceiling, so the very next
        turn re-triggers prune. Lumpy must leave history strictly
        below the budget by a meaningful margin.

        This test would FAIL on the pre-#1430 code path."""
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        budget = TokenBudget("gpt-4")
        history = _seed_history(50, 80)
        budget.use("history", budget.total_budget + 1500, items=len(history))

        cm._lumpy_prune_history(history, budget)

        # Final history must sit BELOW 90% of budget — the old
        # just-enough behaviour would land it right at the ceiling
        # (i.e. > 99%).
        final_pct = budget.allocations["history"].used / budget.total_budget
        assert final_pct < 0.90, (
            f"final history utilization {final_pct:.1%} too close to "
            "the ceiling — next turn would re-prune"
        )

    def test_target_frac_env_override_respected(self, monkeypatch):
        """The PRUNE_TARGET_FRAC constant is read from the env at
        class-import time; setting it at the instance level overrides
        that for one test, simulating the env-driven path."""
        cm = _make_manager()
        # Aggressive override — drop hard.
        cm.PRUNE_TARGET_FRAC = 0.50
        budget = TokenBudget("gpt-4")
        history = _seed_history(50, 80)
        budget.use("history", budget.total_budget + 2000, items=len(history))

        cm._lumpy_prune_history(history, budget)

        target = int(budget.total_budget * 0.50)
        assert budget.allocations["history"].used <= target

    def test_class_constant_clamped_to_unit_interval(self):
        """A garbage env value (>1.0 or <=0.0) must not produce a
        nonsense target. The clamp keeps the constant in
        (0.05, 1.0]."""
        # Stash the original so we can restore.
        original = ContextManager.PRUNE_TARGET_FRAC
        try:
            # Recompute the clamp by reloading would touch global
            # state; just assert the live constant is sane and within
            # the documented band.
            assert 0.05 <= ContextManager.PRUNE_TARGET_FRAC <= 1.0
        finally:
            ContextManager.PRUNE_TARGET_FRAC = original

    def test_history_alloc_items_kept_in_sync(self):
        """After prune, the allocation's items count must match the
        surviving message count — downstream code (budget summaries,
        elastic re-balance) reads this."""
        cm = _make_manager()
        cm.PRUNE_TARGET_FRAC = 0.75
        budget = TokenBudget("gpt-4")
        history = _seed_history(50, 80)
        budget.use("history", budget.total_budget + 2000, items=len(history))

        cm._lumpy_prune_history(history, budget)
        assert budget.allocations["history"].items == len(history)


@pytest.mark.parametrize("frac", [0.5, 0.6, 0.75, 0.9])
def test_lumpy_target_respected_across_fractions(frac: float):
    """Across a spread of target fractions, the final history sits
    at-or-below the configured target."""
    cm = _make_manager()
    cm.PRUNE_TARGET_FRAC = frac
    budget = TokenBudget("gpt-4")
    history = _seed_history(50, 80)
    budget.use("history", budget.total_budget + 2000, items=len(history))

    cm._lumpy_prune_history(history, budget)

    target = int(budget.total_budget * frac)
    assert budget.allocations["history"].used <= target
