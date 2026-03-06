"""
Model Catalog Service

Manages featured models, display name overrides, and category mappings.
Loads configuration from model_catalog.toml.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .model_metadata import ModelInfo, ModelCategory

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CATALOG_PATH = Path(__file__).parent.parent / "model_catalog.toml"


class ModelCatalogService:
    """
    Service for managing model catalog configuration.

    Loads featured models, display names, categories, and hidden lists
    from model_catalog.toml. Enriches ModelInfo objects with this data.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the catalog service.

        Args:
            config_path: Path to model_catalog.toml (default: project root)
        """
        self.config_path = config_path or DEFAULT_CATALOG_PATH
        self._config: Dict = {}
        self._featured: Dict[str, Set[str]] = {}
        self._display_names: Dict[str, str] = {}
        self._categories: Dict[str, Dict[str, List[str]]] = {}
        self._hidden: Dict[str, Set[str]] = {}
        self._context_limits: Dict[str, int] = {}
        self._tool_support: Dict[str, bool] = {}
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

            # Parse featured models
            featured = self._config.get("featured", {})
            for provider, models in featured.items():
                self._featured[provider] = set(models)

            # Parse display name overrides
            self._display_names = self._config.get("display_names", {})

            # Parse categories
            categories = self._config.get("categories", {})
            for category, providers in categories.items():
                self._categories[category] = providers

            # Parse hidden models
            hidden = self._config.get("hidden", {})
            for provider, models in hidden.items():
                self._hidden[provider] = set(models)

            # Parse context limits
            self._context_limits = self._config.get("context_limits", {})

            # Parse tool support overrides
            self._tool_support = self._config.get("tool_support", {})

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
        """Check if a model is in the featured list."""
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

        for category_name in ["embedding", "image", "audio"]:
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

    def enrich_model(self, model: ModelInfo) -> ModelInfo:
        """
        Enrich a ModelInfo with catalog data.

        Updates is_featured, is_hidden, display_name, and context_limit.
        Only overrides category if explicitly configured in catalog.
        """
        self._ensure_loaded()

        model.is_featured = self.is_featured(model.provider, model.id)
        model.is_hidden = self.is_hidden(model.provider, model.id)

        # Only override category if catalog explicitly knows about this model
        # This preserves adapter-detected categories (e.g., Ollama embedding detection)
        catalog_category = self._get_explicit_category(model.provider, model.id)
        if catalog_category is not None:
            model.category = catalog_category

        # Apply display name override if exists
        override = self._display_names.get(model.id)
        if override:
            model.display_name = override

        # Set context limit from catalog
        context_limit = self.get_context_limit(model.id)
        if context_limit is not None:
            model.context_limit = context_limit

        # Override tool support if explicitly configured (catalog wins over adapter)
        tool_support = self.get_tool_support(model.id)
        if tool_support is not None:
            model.supports_tools = tool_support

        return model

    def enrich_models(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Enrich a list of models with catalog data."""
        return [self.enrich_model(m) for m in models]

    def get_featured_models(self, provider: str) -> Set[str]:
        """Get the set of featured model IDs for a provider."""
        self._ensure_loaded()
        return self._featured.get(provider, set())

    def get_all_providers(self) -> List[str]:
        """Get list of all configured providers."""
        self._ensure_loaded()
        providers = set(self._featured.keys())
        providers.update(self._hidden.keys())
        return sorted(providers)


# Global singleton instance
_catalog_service: Optional[ModelCatalogService] = None


def get_catalog_service() -> ModelCatalogService:
    """Get or create the global catalog service instance."""
    global _catalog_service
    if _catalog_service is None:
        _catalog_service = ModelCatalogService()
    return _catalog_service
