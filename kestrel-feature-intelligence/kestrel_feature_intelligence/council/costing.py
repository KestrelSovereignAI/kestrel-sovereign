"""Shared token-usage cost estimation for council sessions."""

from __future__ import annotations

from typing import Any, Dict


# Approximate per-provider pricing per 1M tokens.
# This stays intentionally provider-level so scripts do not drift every time
# an individual model version changes.
PROVIDER_PRICING: Dict[str, Dict[str, float]] = {
    "anthropic": {"input": 15.00, "output": 75.00},
    "openai": {"input": 5.00, "output": 15.00},
    "google": {"input": 1.25, "output": 5.00},
    "vertex_ai": {"input": 1.25, "output": 5.00},
    "ollama": {"input": 0.00, "output": 0.00},
    "xai": {"input": 5.00, "output": 15.00},
    "groq": {"input": 0.27, "output": 0.27},
}


def calculate_estimated_cost(
    provider: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for a token usage record."""
    prices = PROVIDER_PRICING.get(provider, PROVIDER_PRICING["openai"])
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def print_token_usage_summary(session: Any) -> float:
    """Print a token usage table for a council session and return total cost."""
    print()
    print("=" * 70)
    print("TOKEN USAGE & COST SUMMARY")
    print("=" * 70)
    print()

    by_member = session.tokens_by_member()
    total_cost = 0.0

    print(f"{'Member':<12} {'Provider':<12} {'Input':>10} {'Output':>10} {'Est. Cost':>12}")
    print("-" * 60)

    for member_name, data in by_member.items():
        provider = data.get("provider", "unknown")
        input_tokens = data["input"]
        output_tokens = data["output"]
        cost = calculate_estimated_cost(provider, input_tokens, output_tokens)
        total_cost += cost

        print(f"{member_name:<12} {provider:<12} {input_tokens:>10,} {output_tokens:>10,} ${cost:>10.4f}")

    print("-" * 60)
    totals = session.total_tokens()
    print(f"{'TOTAL':<12} {'':<12} {totals['input']:>10,} {totals['output']:>10,} ${total_cost:>10.4f}")
    print()

    if getattr(session, "token_usage", None):
        print("Per-round breakdown:")
        rounds_seen = set()
        for usage in session.token_usage:
            if usage.round_number in rounds_seen:
                continue
            rounds_seen.add(usage.round_number)
            round_tokens = sum(
                u.total_tokens for u in session.token_usage
                if u.round_number == usage.round_number
            )
            print(f"  Round {usage.round_number}: {round_tokens:,} tokens")
        print()

    return total_cost
