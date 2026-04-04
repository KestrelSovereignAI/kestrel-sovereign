"""Contracts for shared council-session cost estimation."""

from scripts import (
    convene_agent_participation,
    convene_council,
    convene_council_rebuttal,
    convene_progress_review,
    convene_sqlite_decision,
)

from kestrel_feature_intelligence.council.costing import calculate_estimated_cost


def test_calculate_estimated_cost_uses_provider_level_defaults():
    cost = calculate_estimated_cost("vertex_ai", 1_000_000, 1_000_000)
    assert cost == 6.25


def test_calculate_estimated_cost_falls_back_to_openai_for_unknown_provider():
    cost = calculate_estimated_cost("unknown", 1_000_000, 1_000_000)
    assert cost == 20.0


def test_council_scripts_use_shared_costing_wrapper():
    session = type(
        "Session",
        (),
        {
            "tokens_by_member": lambda self: {
                "Claude": {"provider": "anthropic", "model": "auto", "input": 1_000_000, "output": 1_000_000}
            },
            "total_tokens": lambda self: {"input": 1_000_000, "output": 1_000_000, "total": 2_000_000},
            "token_usage": [],
        },
    )()

    expected = 90.0
    assert convene_council.print_token_usage(session) == expected
    assert convene_progress_review.print_token_usage(session) == expected
    assert convene_agent_participation.print_token_usage(session) == expected
    assert convene_council_rebuttal.print_token_usage(session) == expected
    assert convene_sqlite_decision.print_token_usage(session) == expected
