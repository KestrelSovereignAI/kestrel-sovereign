"""Shared helpers for config-driven model selection before live discovery."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from kestrel_sovereign.config import load_config, load_section
from kestrel_sovereign.llm.model_catalog import get_catalog_service
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo


def _numeric_rank(text: str, max_parts: int = 4) -> tuple[int, ...]:
    """Return descending-sort numeric rank while ignoring date suffixes.

    Strips contiguous (``20251215``), dashed (``2025-12-15``), underscored,
    and space-separated (``2025 12 15``) ISO dates. The space form shows up
    when display names normalize an id like ``gpt-audio-mini-2025-12-15``
    into ``"Gpt Audio Mini 2025 12 15"`` — without stripping it, the ``2025``
    leaks in as a huge number and outranks the actual model version.
    """
    without_dates = re.sub(r"20\d{2}[-_\s]?\d{2}[-_\s]?\d{2}", "", text)
    numbers = [int(part) for part in re.findall(r"\d+", without_dates)]
    padded = (numbers + [0] * max_parts)[:max_parts]
    return tuple(-number for number in padded)


def _created_rank(created_at: str | None, max_parts: int = 6) -> tuple[int, ...]:
    """Return descending-sort rank for provider creation timestamps."""
    numbers = [int(part) for part in re.findall(r"\d+", created_at or "")]
    padded = (numbers + [0] * max_parts)[:max_parts]
    return tuple(-number for number in padded)


def _rank_cached_candidates(models: list[ModelInfo]) -> list[ModelInfo]:
    """Rank cached candidate models for pre-discovery selection."""
    def sort_key(model: ModelInfo):
        model_lower = model.id.lower()
        display_lower = (model.display_name or "").lower()
        previewish = any(
            token in model_lower or token in display_lower
            for token in ("preview", "beta", "experimental", "exp")
        )
        # Rank on the canonical id, not the formatted display_name. display_name
        # rewrites separators (e.g. ``"Gpt 5.4 Mini 2026 03 17"``) which made
        # the legacy date-stripping regex miss embedded dates.
        return (
            not model.supports_tools,
            previewish,
            _numeric_rank(model.id),
            _created_rank(model.created_at),
            not model.is_featured,
            model.id.lower(),
        )

    return sorted(models, key=sort_key)


def resolve_provider_default(
    provider_name: str,
    llm_config: Optional[Dict[str, Any]] = None,
    catalog_config: Optional[Dict[str, Any]] = None,
    cached_models: Optional[list[ModelInfo]] = None,
) -> str:
    """Resolve a concrete model for a vendor (or ``"vendor:route"``) from config
    intent and the local discovery cache.

    Callers pass a vendor name (``"openai"``) or a composite route key
    (``"anthropic:plan"``). The vendor's discovery catalog is the source of
    truth — all routes for the same vendor share the same models.
    """
    llm_config = llm_config if llm_config is not None else load_section("llm")
    catalog_config = catalog_config or load_config("model_catalog.toml")

    if ":" in provider_name:
        vendor, route = provider_name.split(":", 1)
    else:
        vendor, route = provider_name, None

    # Locate the route config in the new vendor/route shape, or fall back to
    # legacy flat shape (test fixtures may still use it).
    route_cfg: Dict[str, Any] = {}
    vendors = llm_config.get("vendors") or {}
    vendor_cfg = vendors.get(vendor) or {}
    routes = vendor_cfg.get("routes") or {}
    if route and routes.get(route):
        route_cfg = routes[route]
    elif routes:
        # Prefer 'api' when present, otherwise first-declared.
        route_cfg = routes.get("api") or next(iter(routes.values()))
    else:
        # Legacy flat-shape fallback: [<vendor>] section with model/selection_hints.
        route_cfg = llm_config.get(vendor) or {}

    configured_model = route_cfg.get("model")
    if configured_model and configured_model != "auto":
        return configured_model

    if cached_models is None:
        cached_models = get_catalog_service().load_cache() or []

    vendor_models = [
        model for model in cached_models
        if model.provider == vendor
        and model.category == ModelCategory.CHAT
        and not model.is_hidden
    ]

    selection_hints = route_cfg.get("selection_hints", []) or []
    for hint in selection_hints:
        hint_lower = str(hint).lower()
        matches = [
            model for model in vendor_models
            if hint_lower in model.id.lower() or hint_lower in (model.display_name or "").lower()
        ]
        ranked_matches = _rank_cached_candidates(matches)
        if ranked_matches:
            return ranked_matches[0].id

    ranked_models = _rank_cached_candidates(vendor_models)
    if ranked_models:
        return ranked_models[0].id

    raise ValueError(
        f"Could not resolve a default model for vendor '{vendor}'"
        + (f" (route '{route}')" if route else "")
        + ". Refresh model discovery or set a concrete model override."
    )
