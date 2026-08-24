"""Canonical contracts for the legacy static and adaptive token budgets.

Elastic-budget behavior has its own focused module. Model catalog lookups are
covered by the token-counter and model-catalog suites.
"""

import pytest

from kestrel_sovereign.agent.token_budget import (
    DEFAULT_ALLOCATION,
    RESPONSE_RESERVE,
    AdaptiveTokenBudget,
    TokenAllocation,
    TokenBudget,
)


@pytest.fixture(autouse=True)
def _pinned_context_limit(kestrel_toml_catalog):
    """State the context limit these allocation tests budget against.

    Per the module docstring, catalog resolution belongs to the
    token-counter and model-catalog suites. Pinning ``gpt-4`` here keeps
    the allocation math deterministic instead of reading whichever
    ``kestrel.toml`` the machine happens to carry (#3087).
    """
    kestrel_toml_catalog("context_limits_override", {"gpt-4": 8192})


def test_token_allocation_remaining_and_utilization_matrix():
    cases = (
        (1000, 0, 1000, 0.0),
        (1000, 500, 500, 0.5),
        (1000, 1000, 0, 1.0),
        (1000, 1500, 0, 1.5),
        (0, 0, 0, 0.0),
    )

    for budget, used, remaining, utilization in cases:
        allocation = TokenAllocation("test", budget=budget, used=used)
        assert allocation.remaining == remaining
        assert allocation.utilization == utilization


def test_static_budget_allocates_available_context_by_default_percentages():
    budget = TokenBudget("gpt-4")

    assert budget.model == "gpt-4"
    assert budget.context_limit == 8192
    assert budget.response_reserve == RESPONSE_RESERVE
    assert budget.total_budget == budget.context_limit - RESPONSE_RESERVE
    assert {
        name: allocation.budget for name, allocation in budget.allocations.items()
    } == {
        name: int(budget.total_budget * percentage)
        for name, percentage in DEFAULT_ALLOCATION.items()
    }


def test_use_tracks_usage_items_and_exact_fit_boundary():
    budget = TokenBudget("gpt-4")
    history_capacity = budget.history

    assert budget.can_fit("history", history_capacity)
    assert budget.use("history", history_capacity, items=3)
    assert budget.get_remaining("history") == 0
    assert budget.total_used == history_capacity
    assert not budget.can_fit("history", 1)
    assert not budget.use("history", 1)
    assert budget.allocations["history"].used == history_capacity + 1
    assert budget.allocations["history"].items == 4


def test_sources_account_independently():
    budget = TokenBudget("gpt-4")
    history_capacity = budget.history
    system_capacity = budget.system

    assert not budget.use("history", history_capacity + 1)
    assert budget.use("system", 100, items=2)
    assert budget.get_remaining("system") == system_capacity - 100
    assert budget.get_remaining("episodes") == budget.episodes
    assert budget.total_used == history_capacity + 101


def test_unknown_source_fails_without_changing_usage():
    budget = TokenBudget("gpt-4")

    assert not budget.can_fit("unknown", 0)
    assert not budget.use("unknown", 100)
    assert budget.get_remaining("unknown") == 0
    assert budget.total_used == 0


def test_zero_tokens_and_items_are_accepted_no_ops():
    budget = TokenBudget("gpt-4")

    assert budget.can_fit("history", 0)
    assert budget.use("history", 0, items=0)
    assert budget.get_remaining("history") == budget.history
    assert budget.allocations["history"].used == 0
    assert budget.allocations["history"].items == 0
    assert budget.total_used == 0


def test_negative_tokens_or_items_raise_and_leave_state_unchanged():
    budget = TokenBudget("gpt-4")
    assert budget.use("history", 100, items=2)
    before_used = budget.allocations["history"].used
    before_items = budget.allocations["history"].items

    for tokens, items in ((-1, 1), (100, -1), (-100, -1)):
        with pytest.raises(ValueError):
            budget.use("history", tokens, items=items)

    with pytest.raises(ValueError):
        budget.can_fit("history", -1)

    assert budget.allocations["history"].used == before_used
    assert budget.allocations["history"].items == before_items
    assert budget.total_used == before_used


def test_summary_reports_budget_and_allocation_state():
    budget = TokenBudget("gpt-4")
    assert budget.use("history", 100, items=5)

    summary = budget.get_summary()

    assert summary == {
        "model": "gpt-4",
        "context_limit": 8192,
        "response_reserve": RESPONSE_RESERVE,
        "total_budget": budget.total_budget,
        "total_used": 100,
        "allocations": {
            name: {
                "budget": allocation.budget,
                "used": allocation.used,
                "remaining": allocation.remaining,
                "items": allocation.items,
                "utilization": f"{allocation.utilization:.1%}",
            }
            for name, allocation in budget.allocations.items()
        },
    }


@pytest.mark.parametrize(
    ("message_count", "history_share", "episode_share"),
    (
        (0, 0.60, 0.05),
        (9, 0.60, 0.05),
        (10, 0.40, 0.20),
        (29, 0.40, 0.20),
        (30, 0.25, 0.35),
        (10_000, 0.25, 0.35),
    ),
)
def test_adaptive_budget_thresholds(
    message_count: int,
    history_share: float,
    episode_share: float,
):
    budget = AdaptiveTokenBudget("gpt-4", message_count=message_count)

    assert budget.allocations["history"].budget == int(
        budget.total_budget * history_share
    )
    assert budget.allocations["episodes"].budget == int(
        budget.total_budget * episode_share
    )
