"""Contracts for shared config-driven model selection helpers."""

from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.model_selection import resolve_provider_default


def test_resolve_provider_default_prefers_explicit_model():
    resolved = resolve_provider_default(
        "openai",
        llm_config={"openai": {"model": "gpt-5.1"}},
        catalog_config={},
    )

    assert resolved == "gpt-5.1"


def test_resolve_provider_default_uses_selection_hints_against_cached_models():
    resolved = resolve_provider_default(
        "anthropic",
        llm_config={"anthropic": {"model": "auto", "selection_hints": ["opus", "sonnet"]}},
        catalog_config={},
        cached_models=[
            ModelInfo(
                id="claude-sonnet-4-6",
                provider="anthropic",
                display_name="Claude Sonnet 4.6",
                category=ModelCategory.CHAT,
                supports_tools=True,
            ),
            ModelInfo(
                id="claude-opus-4-5-20251101",
                provider="anthropic",
                display_name="Claude Opus 4.5",
                category=ModelCategory.CHAT,
                supports_tools=True,
            ),
        ],
    )

    assert resolved == "claude-opus-4-5-20251101"


def test_resolve_provider_default_prefers_newest_matching_model_from_discovery():
    resolved = resolve_provider_default(
        "anthropic",
        llm_config={"anthropic": {"model": "auto", "selection_hints": ["sonnet"]}},
        catalog_config={},
        cached_models=[
            ModelInfo(
                id="claude-sonnet-4-20250514",
                provider="anthropic",
                display_name="Claude Sonnet 4",
                category=ModelCategory.CHAT,
                supports_tools=True,
                created_at="2025-05-14T00:00:00Z",
            ),
            ModelInfo(
                id="claude-sonnet-4-5-20250929",
                provider="anthropic",
                display_name="Claude Sonnet 4.5",
                category=ModelCategory.CHAT,
                supports_tools=True,
                created_at="2025-09-29T00:00:00Z",
            ),
            ModelInfo(
                id="claude-sonnet-4-6",
                provider="anthropic",
                display_name="Claude Sonnet 4.6",
                category=ModelCategory.CHAT,
                supports_tools=True,
                created_at="2026-04-13T00:00:00Z",
            ),
        ],
    )

    assert resolved == "claude-sonnet-4-6"


def test_resolve_provider_default_uses_vendor_catalog_for_subscription_route():
    """A vendor's routes share the discovery catalog.

    Under the vendor/route architecture, ``openai:plan`` (ChatGPT subscription)
    is a route on the ``openai`` vendor — not a separate provider. Model
    selection must read the vendor's catalog, not look for a separate
    pseudo-provider.
    """
    resolved = resolve_provider_default(
        "openai:plan",
        llm_config={
            "vendors": {
                "openai": {
                    "routes": {
                        "api": {"model": "auto", "adapter": "OpenAIAdapter"},
                        "plan": {"model": "auto", "adapter": "CodexAdapter",
                                 "selection_hints": ["gpt-5.4"]},
                    }
                }
            }
        },
        catalog_config={},
        cached_models=[
            ModelInfo(
                id="gpt-5.4",
                provider="openai",
                display_name="GPT-5.4",
                category=ModelCategory.CHAT,
                supports_tools=True,
            ),
            ModelInfo(
                id="gpt-5.4-mini",
                provider="openai",
                display_name="GPT-5.4 Mini",
                category=ModelCategory.CHAT,
                supports_tools=True,
            ),
        ],
    )

    assert resolved == "gpt-5.4"
