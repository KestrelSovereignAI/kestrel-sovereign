"""Canonical LLM execution-provider names and compatibility aliases."""

from __future__ import annotations

CANONICAL_PROVIDER_ALIASES = {
    "claude_max": "claude_plan",
    "codex": "openai_plan",
}

LEGACY_PROVIDER_ALIASES = {canonical: legacy for legacy, canonical in CANONICAL_PROVIDER_ALIASES.items()}

DISCOVERY_PROVIDER_ALIASES = {
    "claude_plan": "anthropic",
    "claude_max": "anthropic",
    "openai_plan": "openai",
    "codex": "openai",
}


def normalize_provider_name(provider_name: str | None) -> str | None:
    """Normalize legacy provider aliases to the canonical execution provider."""
    if provider_name is None:
        return None
    return CANONICAL_PROVIDER_ALIASES.get(provider_name, provider_name)


def resolve_discovery_provider(provider_name: str | None) -> str | None:
    """Return the canonical provider whose discovery/catalog data should be used."""
    normalized = normalize_provider_name(provider_name)
    if normalized is None:
        return None
    return DISCOVERY_PROVIDER_ALIASES.get(normalized, normalized)


def provider_name_candidates(provider_name: str | None) -> set[str]:
    """Return canonical and legacy spellings accepted for a provider name."""
    if provider_name is None:
        return set()
    normalized = normalize_provider_name(provider_name)
    candidates = {normalized}
    legacy = LEGACY_PROVIDER_ALIASES.get(normalized)
    if legacy:
        candidates.add(legacy)
    return {candidate for candidate in candidates if candidate}
