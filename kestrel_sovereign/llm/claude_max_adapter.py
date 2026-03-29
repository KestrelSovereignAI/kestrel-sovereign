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
from typing import List

from .anthropic_adapter import AnthropicAdapter
from .model_metadata import ModelInfo, ModelCategory

logger = logging.getLogger(__name__)


class ClaudeMaxAdapter(AnthropicAdapter):
    """
    Adapter for Claude Max subscription using OAuth token auth.

    Subclasses AnthropicAdapter — the only difference is authentication
    (auth_token vs api_key) which is handled at client creation time in
    the provider registry. All message handling, tool use, streaming,
    and structured output are inherited from AnthropicAdapter.
    """

    async def list_models(self) -> List[ModelInfo]:
        """
        Return available models for Claude Max subscription.

        Hardcoded because model discovery requires an API key header,
        and Max subscriptions use OAuth tokens instead.
        """
        return [
            ModelInfo(
                id="claude-sonnet-4-6",
                display_name="Claude Sonnet 4.6 (Max)",
                provider="claude_max",
                category=ModelCategory.CHAT,
                context_limit=200000,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                is_featured=True,
            ),
            ModelInfo(
                id="claude-opus-4-6",
                display_name="Claude Opus 4.6 (Max)",
                provider="claude_max",
                category=ModelCategory.CHAT,
                context_limit=200000,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                is_featured=True,
            ),
        ]
