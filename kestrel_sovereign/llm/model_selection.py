"""Shared helpers for config-driven model selection before live discovery."""

from __future__ import annotations

from typing import Any, Dict, Optional

from kestrel_sovereign.config import load_config
from kestrel_sovereign.llm.model_catalog import get_catalog_service
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo


def _rank_cached_candidates(models: list[ModelInfo]) -> list[ModelInfo]:
    """Rank cached candidate models for pre-discovery selection."""
    def sort_key(model: ModelInfo):
        model_lower = model.id.lower()
        display_lower = (model.display_name or "").lower()
        previewish = any(
            token in model_lower or token in display_lower
            for token in ("preview", "beta", "experimental", "exp")
        )
        return (
            not model.supports_tools,
            previewish,
            model.id.lower(),
        )

    return sorted(models, key=sort_key)


def resolve_provider_default(
    provider_name: str,
    llm_config: Optional[Dict[str, Any]] = None,
    catalog_config: Optional[Dict[str, Any]] = None,
    cached_models: Optional[list[ModelInfo]] = None,
) -> str:
    """Resolve a concrete model from config intent and local discovery cache."""
    llm_config = llm_config or load_config("llm_config.toml")
    catalog_config = catalog_config or load_config("model_catalog.toml")

    provider_config = llm_config.get(provider_name, {}) or {}
    configured_model = provider_config.get("model")
    if configured_model and configured_model != "auto":
        return configured_model

    if cached_models is None:
        cached_models = get_catalog_service().load_cache() or []

    provider_models = [
        model for model in cached_models
        if model.provider == provider_name
        and model.category == ModelCategory.CHAT
        and not model.is_hidden
    ]

    selection_hints = provider_config.get("selection_hints", []) or []
    for hint in selection_hints:
        hint_lower = str(hint).lower()
        matches = [
            model for model in provider_models
            if hint_lower in model.id.lower() or hint_lower in (model.display_name or "").lower()
        ]
        ranked_matches = _rank_cached_candidates(matches)
        if ranked_matches:
            return ranked_matches[0].id

    ranked_models = _rank_cached_candidates(provider_models)
    if ranked_models:
        return ranked_models[0].id

    # Legacy fallback: allow old-style [featured] lists if still present.
    featured = (catalog_config.get("featured", {}) or {}).get(provider_name, []) or []
    if featured:
        for hint in selection_hints:
            hint_lower = str(hint).lower()
            for model in featured:
                if hint_lower in model.lower():
                    return model
        return featured[0]

    raise ValueError(
        f"Could not resolve a default model for provider '{provider_name}'. "
        "Set a concrete model override or refresh model discovery cache."
    )
