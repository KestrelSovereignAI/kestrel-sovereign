"""
Claude Max Subscription Adapter

Adapter for using Claude via a Max subscription ($100/$200/month)
using OAuth token authentication instead of API key billing.

Uses the standard Anthropic Python SDK with `auth_token` parameter,
which sends a Bearer token instead of an x-api-key header. This gives
full API access (multi-turn, native tool use, streaming, vision) using
subscription-included usage.

Requirements:
- pip install anthropic
- Active Claude Max subscription
- ANTHROPIC_AUTH_TOKEN env var set (from `claude login` / `claude setup-token`)

How it works:
1. Anthropic SDK is initialized with auth_token= instead of api_key=
2. All API calls use Bearer token auth automatically
3. Everything else (messages, tools, streaming) is identical to AnthropicAdapter
"""
import logging

from .anthropic_adapter import AnthropicAdapter

logger = logging.getLogger(__name__)


class ClaudeMaxAdapter(AnthropicAdapter):
    """
    Adapter for Claude Max subscription using OAuth token auth.

    Subclasses AnthropicAdapter — the only difference is authentication
    (auth_token vs api_key) which is handled at client creation time in
    the provider registry. All message handling, tool use, streaming,
    and structured output are inherited from AnthropicAdapter.
    """

    async def list_models(self, client=None):
        """Claude plan uses Anthropic discovery; this execution wrapper has no catalog.

        ``client`` is accepted for contract symmetry with the SDK 0.5.0
        :meth:`LLMAdapter.list_models` signature; not used because this
        wrapper deliberately raises rather than producing a catalog.
        """
        raise NotImplementedError(
            "Claude plan model discovery is provided by the canonical anthropic provider."
        )

    # ---- Provider metadata (SDK 0.6.0) -------------------------------------
    #
    # substrate_type / deliberation_style inherit from AnthropicAdapter
    # ("claude" / "sequential") — those are correct for the plan route.
    # Override the cost/display/key fields where the plan diverges from
    # the canonical anthropic provider.

    def cost_per_1m_tokens(self):
        # Plan-included usage — no per-token billing. Returning
        # ``{"input": 0.0, "output": 0.0}`` rather than ``None`` so
        # cost-aware routing reflects "free at the margin" rather than
        # falling through to AnthropicAdapter's API pricing.
        return {"input": 0.0, "output": 0.0}

    def display_name(self):
        return "Claude (Max plan)"

    def key_env_var(self):
        return "ANTHROPIC_AUTH_TOKEN"
