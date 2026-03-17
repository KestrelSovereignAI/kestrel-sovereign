"""Contracts for shared config-driven model selection helpers."""

from kestrel_sovereign.llm.model_selection import resolve_provider_default


def test_resolve_provider_default_prefers_explicit_model():
    resolved = resolve_provider_default(
        "openai",
        llm_config={"openai": {"model": "gpt-5.1"}},
        catalog_config={"featured": {"openai": ["gpt-5-mini"]}},
    )

    assert resolved == "gpt-5.1"


def test_resolve_provider_default_uses_selection_hints_against_featured_catalog():
    resolved = resolve_provider_default(
        "anthropic",
        llm_config={"anthropic": {"model": "auto", "selection_hints": ["opus", "sonnet"]}},
        catalog_config={
            "featured": {
                "anthropic": ["claude-sonnet-4-6", "claude-opus-4-5-20251101"],
            }
        },
    )

    assert resolved == "claude-opus-4-5-20251101"
