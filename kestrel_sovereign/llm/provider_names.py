"""Canonical LLM execution-provider names."""

from __future__ import annotations

DISCOVERY_PROVIDER_ALIASES = {
    "claude_plan": "anthropic",
    "openai_plan": "openai",
}


def normalize_provider_name(provider_name: str | None) -> str | None:
    """Return the provider name unchanged."""
    return provider_name


def resolve_discovery_provider(provider_name: str | None) -> str | None:
    """Return the canonical provider whose discovery/catalog data should be used."""
    normalized = normalize_provider_name(provider_name)
    if normalized is None:
        return None
    return DISCOVERY_PROVIDER_ALIASES.get(normalized, normalized)


def provider_name_candidates(provider_name: str | None) -> set[str]:
    """Return accepted spellings for a provider name."""
    if provider_name is None:
        return set()
    return {provider_name}
