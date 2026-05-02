"""
Model Catalog Service

Manages manual overrides (hidden, categories, context limits) from model_catalog.toml
and a discovery cache (model_discovery_cache.json) for fast startup.

The TOML file contains ONLY data that APIs don't provide reliably.
Everything else comes from API discovery.
"""
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .model_metadata import ModelInfo, ModelCategory


# ID fragments that signal a dated snapshot (vendor-specific pinned version):
#   * `...-20250929` — ISO-date suffix (Anthropic-style)
#   * `...-2026-03-17` — dashed-date suffix (OpenAI-style dated releases)
#   * `...-0613`, `...-1106` — MMDD snapshots (older OpenAI)
#   * `...-preview-MMDD` — dated preview tags
#
# Pattern is purely structural; no specific model IDs are encoded.
_DATE_SUFFIX_RE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|\d{4})$")


def _strip_date_suffix(model_id: str) -> str:
    """Return the lineage root for an ID by stripping trailing date snapshots."""
    return _DATE_SUFFIX_RE.sub("", model_id)


def _has_date_suffix(model_id: str) -> bool:
    return bool(_DATE_SUFFIX_RE.search(model_id))


def _mark_canonical_aliases(models: List[ModelInfo]) -> None:
    """Mark each model's ``is_canonical_alias`` using lineage analysis.

    Within a vendor, an ID is a *canonical alias* when it has no date suffix
    and another ID exists whose date-stripped form equals it. That's the
    vendor's moving pointer to the current default in that lineage.

    Pure string analysis — no per-model IDs are embedded anywhere.
    """
    by_vendor: dict[str, list[ModelInfo]] = defaultdict(list)
    for m in models:
        by_vendor[m.provider].append(m)

    for vendor_models in by_vendor.values():
        # Lineage root -> set of models sharing that root.
        lineages: dict[str, list[ModelInfo]] = defaultdict(list)
        for m in vendor_models:
            lineages[_strip_date_suffix(m.id)].append(m)

        for root, members in lineages.items():
            if len(members) < 2:
                continue
            # Canonical = no date suffix in the ID.
            undated = [m for m in members if not _has_date_suffix(m.id)]
            if undated:
                for m in undated:
                    m.is_canonical_alias = True
                    # Canonical alias auto-features in the chat dropdown.
                    m.is_featured = True

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CATALOG_PATH = Path("model_catalog.toml")
DEFAULT_CACHE_PATH = Path(__file__).parent.parent / "model_discovery_cache.json"


class ModelCatalogService:
    """
    Service for managing model catalog configuration.

    Loads manual overrides from model_catalog.toml:
    - hidden: models to never show
    - categories: embedding/image/audio classification
    - context_limits_override: context window sizes for providers that don't report them
    - display_name_overrides: optional display name fixes

    Featured status is NOT managed here — it's computed dynamically:
    - Models configured in llm_config.toml are featured
    - Models recently used (frecency > 0) are featured
    """

    def __init__(self, config_path: Optional[Path] = None, cache_path: Optional[Path] = None):
        """
        Initialize the catalog service.

        Args:
            config_path: Path to model_catalog.toml (default: project root)
            cache_path: Path to model_discovery_cache.json (default: alongside catalog)
        """
        self.config_path = config_path or DEFAULT_CATALOG_PATH
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self._config: Dict = {}
        self._hidden: Dict[str, Set[str]] = {}
        self._categories: Dict[str, Dict[str, List[str]]] = {}
        self._context_limits: Dict[str, int] = {}
        self._display_names: Dict[str, str] = {}
        self._tool_support: Dict[str, bool] = {}
        # vendor -> {"small": model_id, "medium": ..., "large": ...}
        self._size_tiers: Dict[str, Dict[str, str]] = {}

        # Legacy support: if old-style [featured] section exists, still load it
        self._featured: Dict[str, Set[str]] = {}

        self._loaded = False

    def load(self) -> None:
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            logger.warning(f"Model catalog not found at {self.config_path}, using defaults")
            self._loaded = True
            return

        try:
            with open(self.config_path, "rb") as f:
                self._config = tomllib.load(f)

            # Parse hidden models
            hidden = self._config.get("hidden", {})
            for provider, models in hidden.items():
                self._hidden[provider] = set(models)

            # Parse categories
            categories = self._config.get("categories", {})
            for category, providers in categories.items():
                self._categories[category] = providers

            # Parse context limits — support both old and new key names
            self._context_limits = (
                self._config.get("context_limits_override", {})
                or self._config.get("context_limits", {})
            )

            # Parse display name overrides — support both old and new key names
            self._display_names = (
                self._config.get("display_name_overrides", {})
                or self._config.get("display_names", {})
            )

            # Parse tool support overrides
            self._tool_support = self._config.get("tool_support", {})

            # Parse size tiers (vendor -> small/medium/large -> model_id).
            # See [size_tiers] in model_catalog.toml — discovery seam for
            # "give me the canonical small / medium / large model for this
            # vendor" without hardcoding model IDs in Python.
            self._size_tiers = self._config.get("size_tiers", {})

            # Legacy: load [featured] if it exists (for backward compat)
            featured = self._config.get("featured", {})
            for provider, models in featured.items():
                self._featured[provider] = set(models)

            self._loaded = True
            logger.info(f"Loaded model catalog from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load model catalog: {e}")
            self._loaded = True  # Mark as loaded to avoid retry loops

    def _ensure_loaded(self) -> None:
        """Ensure configuration is loaded."""
        if not self._loaded:
            self.load()

    def is_featured(self, provider: str, model_id: str) -> bool:
        """Check if a model is in the legacy featured list.

        Note: Featured status is now computed dynamically (configured + MRU).
        This method only checks the legacy [featured] TOML section.
        """
        self._ensure_loaded()
        featured_set = self._featured.get(provider, set())
        return model_id in featured_set

    def is_hidden(self, provider: str, model_id: str) -> bool:
        """Check if a model should be hidden."""
        self._ensure_loaded()
        hidden_set = self._hidden.get(provider, set())
        return model_id in hidden_set

    def get_display_name(self, model_id: str, default: Optional[str] = None) -> str:
        """Get display name override for a model."""
        self._ensure_loaded()
        return self._display_names.get(model_id, default or model_id)

    def _get_explicit_category(self, provider: str, model_id: str) -> Optional[ModelCategory]:
        """
        Get category only if model is explicitly listed in catalog.

        Returns None if model is not in any category list.
        This allows adapters to preserve their own detection logic.
        """
        self._ensure_loaded()

        for category_name in ["embedding", "image", "audio", "completion"]:
            category_config = self._categories.get(category_name, {})
            provider_models = category_config.get(provider, [])
            if model_id in provider_models:
                return ModelCategory(category_name)

        return None

    def get_category(self, provider: str, model_id: str) -> ModelCategory:
        """
        Determine the category of a model.

        Checks embedding, image, and audio categories in config.
        Defaults to CHAT if not found in any explicit category list.
        """
        self._ensure_loaded()

        explicit = self._get_explicit_category(provider, model_id)
        if explicit is not None:
            return explicit

        return ModelCategory.CHAT

    def get_context_limit(self, model_id: str) -> Optional[int]:
        """
        Get the context limit for a model.

        Args:
            model_id: The model ID to look up

        Returns:
            Context limit in tokens, or None if not configured
        """
        self._ensure_loaded()

        # Try exact match first
        if model_id in self._context_limits:
            return self._context_limits[model_id]

        # Try base model name (before :)
        base_model = model_id.split(":")[0]
        if base_model in self._context_limits:
            return self._context_limits[base_model]

        # Try partial match (e.g., "gpt-4" matches "gpt-4-turbo")
        model_lower = model_id.lower()
        for known_model, limit in self._context_limits.items():
            if known_model.lower() in model_lower:
                return limit

        return None

    def get_model_for_size(self, vendor: str, size: str) -> Optional[str]:
        """Look up the canonical model ID for a vendor at a size tier.

        Size tiers are vendor-relative — "large" means "the biggest
        general-purpose chat model this vendor publishes," not a fixed
        token threshold. They're config (model_catalog.toml
        [size_tiers]) so a new generation requires editing one section,
        not chasing version numbers across the codebase.

        Args:
            vendor: e.g. "anthropic", "openai", "ollama"
            size:   "small", "medium", or "large"

        Returns:
            The configured model ID, or None when the vendor or size
            tier isn't declared. Callers decide whether None is a skip
            (tests) or a fallback (production selection).
        """
        self._ensure_loaded()
        vendor_tiers = self._size_tiers.get(vendor, {})
        return vendor_tiers.get(size)

    def get_tool_support(self, model_id: str) -> Optional[bool]:
        """Get tool support override for a model.

        Returns:
            True/False if explicitly configured, None if not in catalog.
        """
        self._ensure_loaded()

        if model_id in self._tool_support:
            return self._tool_support[model_id]

        # Try base model name (before :)
        base_model = model_id.split(":")[0]
        if base_model in self._tool_support:
            return self._tool_support[base_model]

        return None

    # Deprecation keywords that, when present in a vendor's description, mark
    # the model is_deprecated. Config-driven terms would overwrite this; the
    # default covers the wording every major vendor uses.
    _DEPRECATION_KEYWORDS = (
        "deprecated",
        "legacy",
        "will be retired",
        "retired on",
        "end of life",
        "eol",
        "sunset",
    )

    def enrich_model(self, model: ModelInfo) -> ModelInfo:
        """
        Enrich a ModelInfo with catalog + heuristic signals.

        Applies (in order): hidden override, category override (including
        ``completion``), display name, context limit, tool support, deprecation
        from description.

        Featured status is OR-additive (legacy ``[featured]`` can promote but
        never demote).
        """
        self._ensure_loaded()

        # Featured: OR logic — never unfeature a model that was already featured
        if self.is_featured(model.provider, model.id):
            model.is_featured = True
        # If not in legacy featured list, preserve existing is_featured value

        # Hidden: emergency override from [hidden].
        model.is_hidden = self.is_hidden(model.provider, model.id)

        # Category: only override when the catalog explicitly knows about this
        # model. Preserves adapter-detected categories (e.g. Ollama embedding).
        catalog_category = self._get_explicit_category(model.provider, model.id)
        if catalog_category is not None:
            model.category = catalog_category

        # Display name override
        override = self._display_names.get(model.id)
        if override:
            model.display_name = override

        # Context limit: catalog is a fallback for providers that don't
        # report a context window (per the [context_limits_override] section
        # header). When discovery DID return a value, trust it — provider
        # APIs are the source of truth, the catalog is the offline safety
        # net. Without this guard, stale catalog entries silently demote a
        # real 1M-window model to whatever the TOML says when the vendor
        # quietly expanded the window.
        if model.context_limit is None:
            context_limit = self.get_context_limit(model.id)
            if context_limit is not None:
                model.context_limit = context_limit

        # Tool support override
        tool_support = self.get_tool_support(model.id)
        if tool_support is not None:
            model.supports_tools = tool_support

        # Deprecation from vendor-reported description.
        if model.description:
            desc_lower = model.description.lower()
            if any(kw in desc_lower for kw in self._DEPRECATION_KEYWORDS):
                model.is_deprecated = True

        return model

    def enrich_models(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Enrich a list of models. Runs per-model enrichment, then adds
        cross-model signals (canonical-alias detection) that need the full list.
        """
        enriched = [self.enrich_model(m) for m in models]
        _mark_canonical_aliases(enriched)
        return enriched

    def get_featured_models(self, provider: str) -> Set[str]:
        """Get the set of featured model IDs for a provider.

        Note: This returns from the legacy [featured] section only.
        True featured status is now computed dynamically.
        """
        self._ensure_loaded()
        return self._featured.get(provider, set())

    def get_all_providers(self) -> List[str]:
        """Get list of all configured providers."""
        self._ensure_loaded()
        providers = set(self._featured.keys())
        providers.update(self._hidden.keys())
        return sorted(providers)

    # --- Discovery Cache ---

    def write_cache(self, models: List[ModelInfo]) -> None:
        """Write discovered models to cache file for fast startup.

        Args:
            models: List of enriched ModelInfo objects from discovery
        """
        try:
            from datetime import datetime, timezone
            cache_data = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model_count": len(models),
                "models": [m.to_dict() for m in models],
            }
            with open(self.cache_path, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)
            logger.info(f"Wrote discovery cache: {len(models)} models to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write discovery cache: {e}")

    def load_cache(self) -> Optional[List[ModelInfo]]:
        """Load models from discovery cache file.

        Returns:
            List of ModelInfo objects if cache exists, None otherwise
        """
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "r") as f:
                cache_data = json.load(f)

            models = [ModelInfo.from_dict(m) for m in cache_data.get("models", [])]
            generated_at = cache_data.get("generated_at", "unknown")
            logger.info(f"Loaded discovery cache: {len(models)} models (generated: {generated_at})")
            return models
        except Exception as e:
            logger.warning(f"Failed to load discovery cache: {e}")
            return None


# Global singleton instance
_catalog_service: Optional[ModelCatalogService] = None


def get_catalog_service() -> ModelCatalogService:
    """Get or create the global catalog service instance."""
    global _catalog_service
    if _catalog_service is None:
        _catalog_service = ModelCatalogService()
    return _catalog_service
