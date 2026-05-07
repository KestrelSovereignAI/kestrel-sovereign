"""Shared token-usage cost estimation for council sessions.

Pricing resolution strategy (SDK 0.6.0+):

1. **Adapter-first.** When an LLM service registry is available
   (online cost reporting), look up the route for ``provider`` and
   call ``adapter.cost_per_1m_tokens()``. Plugin authors are
   first-class participants — a third-party Kimi or DeepSeek plugin
   that overrides ``cost_per_1m_tokens`` gets accurate pricing here
   without any edit to this module.

2. **Static fallback table.** Used by offline analysis tools that
   import this module without a live registry (e.g. log-replay cost
   reports). The table is canonical pricing for known providers as
   of the version that ships it; it's *documentation*, not an
   architectural couple.

3. **Conservative default.** When neither path knows about the
   provider, default to a paid-API midpoint so cost-aware routing
   doesn't accidentally prefer an unknown-cost provider.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Step 2: offline fallback table. Known per-provider pricing per 1M
# tokens. Provider-level rather than per-model so log-replay scripts
# don't drift every time an individual model version changes.
# Runtime cost reporting prefers the adapter (Step 1) over this.
PROVIDER_PRICING_FALLBACK: Dict[str, Dict[str, float]] = {
    "anthropic": {"input": 15.00, "output": 75.00},
    "openai": {"input": 5.00, "output": 15.00},
    "google": {"input": 1.25, "output": 5.00},
    "vertex_ai": {"input": 1.25, "output": 5.00},
    "ollama": {"input": 0.00, "output": 0.00},
    "xai": {"input": 5.00, "output": 15.00},
    "groq": {"input": 0.27, "output": 0.27},
}

# Backwards compat for any external script still importing the old name.
PROVIDER_PRICING = PROVIDER_PRICING_FALLBACK

# Step 3: conservative paid-API default for unknown providers.
CONSERVATIVE_FALLBACK_PRICING: Dict[str, float] = {"input": 5.00, "output": 15.00}


def _adapter_pricing_for(provider: str) -> Optional[Dict[str, float]]:
    """Look up pricing from the adapter registered for ``provider``.

    Returns ``None`` when no live LLM service is reachable (offline
    cost analysis), no route matches the provider name, or the
    adapter doesn't expose pricing. Caller falls back to the static
    table.
    """
    try:
        # Lazy import — costing.py is also used by offline scripts that
        # don't want the framework's full LLM stack imported.
        from kestrel_sovereign.llm.service import get_llm_service
    except Exception:
        return None

    try:
        svc = get_llm_service()
    except Exception:
        return None
    if svc is None:
        return None

    routes = getattr(svc, "providers", None) or []
    for route in routes:
        # Routes are dicts under the new vendor/route schema (see
        # provider_registry.ProviderInfo + LLMService._convert_providers_format).
        # Match either the bare vendor name (``anthropic``) or the
        # composite name prefix (``anthropic:plan``).
        vendor = route.get("vendor")
        name = route.get("name", "")
        if vendor != provider and not name.startswith(f"{provider}:"):
            continue
        adapter = route.get("adapter")
        if adapter is None:
            continue
        try:
            cost = adapter.cost_per_1m_tokens()
        except Exception as e:
            logger.debug("adapter.cost_per_1m_tokens() raised for %r: %s", provider, e)
            continue
        if cost is not None:
            return cost
    return None


def calculate_estimated_cost(
    provider: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for a token usage record.

    Resolution: adapter-first → static fallback → conservative
    paid-API default. See module docstring for the rationale.
    """
    prices = (
        _adapter_pricing_for(provider)
        or PROVIDER_PRICING_FALLBACK.get(provider)
        or CONSERVATIVE_FALLBACK_PRICING
    )
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
