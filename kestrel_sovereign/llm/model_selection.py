"""Shared helpers for config-driven model selection before live discovery."""

from __future__ import annotations

from typing import Any, Dict, Optional

from kestrel_sovereign.config import load_config


def resolve_provider_default(
    provider_name: str,
    llm_config: Optional[Dict[str, Any]] = None,
    catalog_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve a concrete model from config intent and local catalog cache."""
    llm_config = llm_config or load_config("llm_config.toml")
    catalog_config = catalog_config or load_config("model_catalog.toml")

    provider_config = llm_config.get(provider_name, {}) or {}
    configured_model = provider_config.get("model")
    if configured_model and configured_model != "auto":
        return configured_model

    featured = (catalog_config.get("featured", {}) or {}).get(provider_name, []) or []
    selection_hints = provider_config.get("selection_hints", []) or []

    for hint in selection_hints:
        hint_lower = str(hint).lower()
        for model in featured:
            if hint_lower in model.lower():
                return model

    if featured:
        return featured[0]

    raise ValueError(
        f"Could not resolve a default model for provider '{provider_name}'. "
        "Set a concrete model override or update llm_config.toml/model_catalog.toml."
    )
