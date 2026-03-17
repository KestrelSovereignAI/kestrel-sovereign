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
