"""
Model Discovery Service

Discovers available models from all LLM providers using their APIs.
Enriches results with manual overrides from model_catalog.toml.
Provides in-memory caching and disk-based cache for fast startup.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Set

from .model_metadata import ModelInfo, ModelCategory
from .model_catalog import get_catalog_service, ModelCatalogService

logger = logging.getLogger(__name__)


class ModelDiscoveryMixin:
    """
    Mixin class providing model discovery methods for LLMService.

    Uses adapter.list_models() for each provider to get available models,
    then enriches them with catalog data (featured, hidden, display names).
    """

    # Cache configuration
    _cache_ttl: int = 300  # 5 minutes
    _model_cache: Optional[List[ModelInfo]] = None
    _cache_timestamp: Optional[float] = None

    async def discover_all_models(
        self,
        use_cache: bool = True,
        featured_only: bool = False,
        category: Optional[ModelCategory] = None,
        providers: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """
        Discover all available models from all configured providers.

        Args:
            use_cache: Whether to use cached results if available
            featured_only: Only return featured models
            category: Filter by category (CHAT, EMBEDDING, etc.)
            providers: Filter by provider names

        Returns:
            List of ModelInfo objects, enriched with catalog data
        """
        # Check cache
        if use_cache and self._model_cache is not None and self._cache_timestamp is not None:
            age = time.time() - self._cache_timestamp
            if age < self._cache_ttl:
                logger.debug(f"Using cached models (age: {age:.0f}s)")
                return self._filter_models(
                    self._model_cache,
                    featured_only=featured_only,
                    category=category,
                    providers=providers
                )

        logger.info("Discovering models from all providers...")
        all_models: List[ModelInfo] = []
        catalog = get_catalog_service()

        # Collect models from all adapters in parallel
        discovery_tasks = []

        for adapter_name, adapter in self._get_adapters().items():
            discovery_tasks.append(
                self._discover_from_adapter(adapter_name, adapter)
            )

        # Wait for all discoveries
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Model discovery failed: {result}")
            elif isinstance(result, list):
                all_models.extend(result)

        # Add configured provider models (from llm_config.toml) that weren't discovered
        # This ensures models like xai/grok show up even without API discovery
        discovered_ids = set(m.id for m in all_models)
        api_discovered_ids = set(discovered_ids)  # Snapshot before synthetic additions
        if hasattr(self, 'providers') and isinstance(self.providers, list):
            for provider in self.providers:
                provider_name = provider.get('name')
                model_id = provider.get('model')
                if model_id and model_id != "auto" and model_id not in discovered_ids:
                    # Cloud providers always support tools; only Ollama needs detection
                    is_cloud = provider_name not in ("ollama",)
                    all_models.append(ModelInfo(
                        id=model_id,
                        provider=provider_name,
                        display_name=model_id,
                        category=ModelCategory.CHAT,
                        is_featured=True,  # Configured models are featured
                        is_hidden=False,
                        supports_tools=is_cloud,
                    ))
                    discovered_ids.add(model_id)
                    logger.debug(f"Added configured model: {provider_name}/{model_id}")

        # Enrich models with catalog data
        all_models = catalog.enrich_models(all_models)

        # Register discovered context limits into TokenCounter
        from kestrel_sovereign.agent.token_counter import register_discovered_limits
        register_discovered_limits(all_models)

        # Auto-resolve providers with model="auto" to the first discovered chat model
        if hasattr(self, 'providers') and isinstance(self.providers, list):
            for provider in self.providers:
                if provider.get("model") == "auto":
                    provider_name = provider.get("name")
                    provider_models = [
                        m for m in all_models
                        if m.provider == provider_name and m.category == ModelCategory.CHAT
                    ]
                    if provider_models:
                        provider["model"] = provider_models[0].id
                        logger.info(f"Auto-resolved model for {provider_name}: {provider['model']}")
                    else:
                        logger.warning(f"No chat models discovered for {provider_name} — 'auto' unresolved")

        # Freshness check: warn about configured models not found via API discovery
        # Uses api_discovered_ids (snapshot before synthetic additions) so the
        # warning fires even when the model was synthetically added above
        if hasattr(self, 'providers') and isinstance(self.providers, list):
            for provider in self.providers:
                model = provider.get("model")
                if model and model != "auto" and model not in api_discovered_ids:
                    logger.warning(
                        f"Configured model '{model}' for {provider.get('name')} "
                        f"not found in discovery — may be deprecated"
                    )

        # Cache results in memory and on disk
        self._model_cache = all_models
        self._cache_timestamp = time.time()
        catalog.write_cache(all_models)

        logger.info(f"Discovered {len(all_models)} models total")

        return self._filter_models(
            all_models,
            featured_only=featured_only,
            category=category,
            providers=providers
        )

    def _load_from_disk_cache(self) -> bool:
        """Load models from disk cache if no in-memory cache exists.

        Called on first access to provide immediate model availability
        before API discovery completes.

        Returns:
            True if cache was loaded, False otherwise
        """
        if self._model_cache is not None:
            return False  # Already have in-memory data

        catalog = get_catalog_service()
        cached = catalog.load_cache()
        if cached:
            self._model_cache = cached
            self._cache_timestamp = 0.0  # Expired — will refresh on next discover_all_models()
            logger.info(f"Pre-populated {len(cached)} models from disk cache")

            # Register context limits from cache
            from kestrel_sovereign.agent.token_counter import register_discovered_limits
            register_discovered_limits(cached)
            return True
        return False

    async def _discover_from_adapter(
        self,
        adapter_name: str,
        adapter: Any
    ) -> List[ModelInfo]:
        """
        Discover models from a single adapter.

        Args:
            adapter_name: Name of the adapter (for logging)
            adapter: The adapter instance

        Returns:
            List of ModelInfo objects from this adapter
        """
        # Skip adapter-based discovery for OpenAI-compatible providers that reuse
        # OpenAIAdapter — they'd incorrectly return OpenAI's model list.
        # Local providers (llama_cpp etc.) are discovered via direct /v1/models query below.
        SKIP_DISCOVERY = {'runpod', 'xai', 'groq', 'together', 'mistral', 'perplexity', 'fireworks', 'azure_openai'}

        # Also skip local providers that reuse OpenAIAdapter — discover them directly
        if hasattr(self, 'config') and isinstance(self.config, dict):
            provider_config = self.config.get(adapter_name, {})
            if isinstance(provider_config, dict) and provider_config.get("local"):
                return await self._discover_local_openai_compatible(adapter_name, provider_config)

        if adapter_name in SKIP_DISCOVERY:
            logger.debug(f"{adapter_name}: skipping model discovery (OpenAI-compatible provider)")
            return []

        try:
            if hasattr(adapter, 'list_models'):
                models = await adapter.list_models()
                logger.debug(f"{adapter_name}: discovered {len(models)} models")
                return models
            else:
                logger.debug(f"{adapter_name}: list_models not implemented")
                return []
        except NotImplementedError:
            logger.debug(f"{adapter_name}: list_models not implemented")
            return []
        except Exception as e:
            logger.warning(f"{adapter_name}: model discovery failed: {e}")
            return []

    async def _discover_local_openai_compatible(
        self, provider_name: str, provider_config: dict
    ) -> List[ModelInfo]:
        """Discover models from a local OpenAI-compatible server (llama.cpp, etc.)."""
        base_url = provider_config.get("base_url", "")
        if not base_url:
            return []

        import httpx
        models_url = f"{base_url.rstrip('/')}/models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(models_url)
                resp.raise_for_status()
                data = resp.json()

            results = []
            # Handle both OpenAI format {"data": [...]} and Ollama format {"models": [...]}
            model_list = data.get("data") or data.get("models") or []
            for m in model_list:
                model_id = m.get("id") or m.get("model") or m.get("name", "")
                if not model_id:
                    continue
                results.append(ModelInfo(
                    id=model_id,
                    provider=provider_name,
                    display_name=model_id.split("/")[-1].replace(".gguf", ""),
                    category=ModelCategory.CHAT,
                    supports_tools=True,
                    is_featured=True,
                ))
            logger.info(f"{provider_name}: discovered {len(results)} models from {models_url}")
            return results
        except Exception as e:
            logger.warning(f"{provider_name}: local model discovery failed ({models_url}): {e}")
            return []

    def _get_adapters(self) -> Dict[str, Any]:
        """
        Get all configured adapters.

        Looks at providers list structure (LLMService style) and
        individual adapter attributes.
        """
        adapters = {}

        # Check providers list (LLMService stores adapters in provider dicts)
        if hasattr(self, 'providers') and isinstance(self.providers, list):
            for provider in self.providers:
                if isinstance(provider, dict):
                    name = provider.get('name')
                    adapter = provider.get('adapter')
                    if name and adapter and name not in adapters:
                        adapters[name] = adapter

        # Check for common adapter attributes (fallback) - only if not already in adapters
        if 'openai' not in adapters and hasattr(self, 'openai_adapter') and self.openai_adapter:
            adapters['openai'] = self.openai_adapter
        if 'anthropic' not in adapters and hasattr(self, 'anthropic_adapter') and self.anthropic_adapter:
            adapters['anthropic'] = self.anthropic_adapter
        if 'ollama' not in adapters and hasattr(self, 'ollama_adapter') and self.ollama_adapter:
            adapters['ollama'] = self.ollama_adapter
        if 'vertex_ai' not in adapters and hasattr(self, 'vertex_adapter') and self.vertex_adapter:
            adapters['vertex_ai'] = self.vertex_adapter
        if 'google' not in adapters and hasattr(self, 'google_adapter') and self.google_adapter:
            adapters['google'] = self.google_adapter

        # Check for adapters dict - don't override existing entries
        if hasattr(self, 'adapters') and isinstance(self.adapters, dict):
            for name, adapter in self.adapters.items():
                if name not in adapters:
                    adapters[name] = adapter

        return adapters

    def _filter_models(
        self,
        models: List[ModelInfo],
        featured_only: bool = False,
        category: Optional[ModelCategory] = None,
        providers: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """
        Filter models based on criteria.

        Args:
            models: List of models to filter
            featured_only: Only return featured models
            category: Filter by category
            providers: Filter by provider names

        Returns:
            Filtered list of models
        """
        result = models

        # Filter out hidden models (always)
        result = [m for m in result if not m.is_hidden]

        # Apply filters
        if featured_only:
            result = [m for m in result if m.is_featured]

        if category:
            result = [m for m in result if m.category == category]

        if providers:
            provider_set = set(providers)
            result = [m for m in result if m.provider in provider_set]

        return result

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Synchronous method for backwards compatibility.

        Returns configured models from providers list.
        For full discovery, use discover_all_models() instead.
        """
        models = []

        if hasattr(self, 'providers'):
            for provider in self.providers:
                provider_name = provider.get("name", "unknown")
                model_id = provider.get("model", "unknown")

                models.append({
                    "id": model_id,
                    "provider": provider_name,
                    "name": model_id,
                    "description": f"{provider_name.capitalize()} model"
                })

        return models

    def get_models_by_provider(
        self,
        models: Optional[List[ModelInfo]] = None
    ) -> Dict[str, List[ModelInfo]]:
        """
        Group models by provider.

        Args:
            models: Models to group (uses cache if None)

        Returns:
            Dict mapping provider name to list of models
        """
        if models is None:
            models = self._model_cache or []

        result: Dict[str, List[ModelInfo]] = {}
        for model in models:
            if model.provider not in result:
                result[model.provider] = []
            result[model.provider].append(model)

        return result

    def get_featured_models(
        self,
        models: Optional[List[ModelInfo]] = None
    ) -> List[ModelInfo]:
        """
        Get only featured models.

        Args:
            models: Models to filter (uses cache if None)

        Returns:
            List of featured models
        """
        if models is None:
            models = self._model_cache or []

        return [m for m in models if m.is_featured and not m.is_hidden]

    def set_default_model(self, model_id: str):
        """Set the default model for this LLM service."""
        if hasattr(self, 'default_model'):
            self.default_model = model_id
            logger.info(f"Default model set to: {model_id}")

    def clear_model_cache(self):
        """Clear the model cache to force rediscovery."""
        self._model_cache = None
        self._cache_timestamp = None
        logger.info("Model cache cleared")
