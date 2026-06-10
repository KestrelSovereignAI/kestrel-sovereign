"""
Model Discovery Service

Discovers available models from all LLM providers using their APIs.
Enriches results with manual overrides from model_catalog.toml.
Provides in-memory caching and disk-based cache for fast startup.
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

from .model_metadata import ModelInfo, ModelCategory
from .model_catalog import get_catalog_service, ModelCatalogService
from .model_cache import get_shared_model_cache

logger = logging.getLogger(__name__)


class ModelDiscoveryMixin:
    """
    Mixin class providing model discovery methods for LLMService.

    Uses adapter.list_models() for each provider to get available models,
    then enriches them with catalog data (featured, hidden, display names).
    """

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
        # Check shared process-wide cache
        shared_cache = get_shared_model_cache()
        if use_cache:
            cached = shared_cache.get()
            if cached is not None:
                logger.debug("Using shared model cache")
                return self._filter_models(
                    cached,
                    featured_only=featured_only,
                    category=category,
                    providers=providers
                )

        logger.info("Discovering models per vendor...")
        all_models: List[ModelInfo] = []
        catalog = get_catalog_service()

        # Discovery runs PER VENDOR, not per route. A vendor may have multiple
        # routes (anthropic:api + anthropic:plan); they share the catalog, so
        # we pick the first route per vendor whose adapter can list models.
        discovery_tasks = []
        for vendor, route in self._select_discovery_routes():
            discovery_tasks.append(
                self._discover_for_vendor_route(vendor, route)
            )

        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Vendor discovery failed: {result}")
            elif isinstance(result, list):
                all_models.extend(result)

        # Add configured provider models (from kestrel.toml [llm]) that weren't discovered
        # This ensures models like xai/grok show up even without API discovery
        discovered_ids = set(m.id for m in all_models)
        api_discovered_ids = set(discovered_ids)  # Snapshot before synthetic additions
        if hasattr(self, 'providers') and isinstance(self.providers, list):
            for provider in self.providers:
                provider_name = provider.get('name')
                model_id = provider.get('model')
                if model_id and model_id != "auto" and model_id not in discovered_ids:
                    # Tool support follows the route's cloud/local flag,
                    # which is set during provider initialization based on
                    # config (vendors.<name>.is_cloud) rather than guessed
                    # from the provider name. SDK-only LLM plugins (Kimi,
                    # DeepSeek, etc.) flow through this path correctly
                    # without needing to be enumerated in a hardcoded
                    # exclusion list.
                    is_cloud = provider.get("is_cloud", True)
                    all_models.append(ModelInfo(
                        id=model_id,
                        provider=provider_name,
                        display_name=model_id,
                        category=ModelCategory.CHAT,
                        is_featured=True,  # Configured models are featured
                        is_hidden=False,
                        supports_tools=is_cloud,
                        # Configured chat models — assume streaming. The SDK
                        # 0.5.0 ModelInfo default is False (conservative);
                        # this synthetic-model path predates that and
                        # encodes the legacy "every chat model streams"
                        # assumption explicitly so it doesn't drift.
                        supports_streaming=True,
                    ))
                    discovered_ids.add(model_id)
                    logger.debug(f"Added configured model: {provider_name}/{model_id}")

        # Enrich models with catalog data
        all_models = catalog.enrich_models(all_models)

        # Register discovered context limits into TokenCounter
        from kestrel_sovereign.agent.token_counter import register_discovered_limits
        register_discovered_limits(all_models)

        # Auto-resolve providers with model="auto" to the first discovered chat model
        self._resolve_auto_providers(all_models)

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

        # Cache results in shared memory cache and on disk
        shared_cache.set(all_models)
        catalog.write_cache(all_models)

        logger.info(f"Discovered {len(all_models)} models total")

        return self._filter_models(
            all_models,
            featured_only=featured_only,
            category=category,
            providers=providers
        )

    def _resolve_auto_providers(self, models: list) -> None:
        """Resolve routes whose configured model is ``"auto"`` using discovered models.

        Each route inherits the model catalog of its vendor. Selection is driven
        by the route's ``selection_hints`` (config), then by rank heuristics.
        """
        if not hasattr(self, 'providers') or not isinstance(self.providers, list):
            return
        for provider in self.providers:
            if provider.get("model") != "auto":
                continue
            vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
            route_key = provider.get("name")
            candidates = [
                m for m in models
                if m.provider == vendor
                and m.category == ModelCategory.CHAT
                and not m.is_hidden
            ]
            hints = list(provider.get("selection_hints") or [])
            selected = self._select_auto_model_for_route(candidates, hints)
            if selected:
                provider["model"] = selected.id
                logger.info(f"Auto-resolved model for {route_key}: {provider['model']}")
            else:
                logger.warning(f"No chat models discovered for {route_key} (vendor={vendor}) — 'auto' unresolved")

    def _select_auto_model_for_route(
        self,
        candidates: list[ModelInfo],
        selection_hints: list[str],
    ) -> Optional[ModelInfo]:
        """Pick one model for a route: hint-matched first, then ranked.

        ``selection_hints`` are substring patterns (e.g. ``"mini"``, ``"sonnet"``)
        that live in config; no specific model IDs are hardcoded here.
        """
        if not candidates:
            return None
        for hint in selection_hints:
            hint_lower = str(hint).lower()
            matches = [
                m for m in candidates
                if hint_lower in m.id.lower() or hint_lower in (m.display_name or "").lower()
            ]
            ranked_matches = self._rank_auto_candidates(matches)
            if ranked_matches:
                return ranked_matches[0]

        featured = [m for m in candidates if m.is_featured]
        ranked_featured = self._rank_auto_candidates(featured)
        if ranked_featured:
            return ranked_featured[0]

        ranked_all = self._rank_auto_candidates(candidates)
        return ranked_all[0] if ranked_all else None

    def _rank_auto_candidates(self, models: list[ModelInfo]) -> list[ModelInfo]:
        """Rank candidate models for auto-selection.

        Delegates to the canonical ranker in ``model_selection`` so startup
        auto-resolution and per-agent default resolution agree on order.
        Previously this used ``datetime.fromisoformat(created_at)`` — but
        ``created_at`` is stored as a Unix-timestamp integer, the parse
        fails, and the function fell through to alphabetical (``gpt-4.1-mini``
        beating ``gpt-5.4-mini`` by ASCII order). Sharing the ranker keeps
        any future fixes in one place.
        """
        from .model_selection import _rank_cached_candidates
        return _rank_cached_candidates(models)

    def _load_from_disk_cache(self) -> bool:
        """Load models from disk cache into the shared process-wide cache.

        Called during LLMService init to provide immediate model availability
        before API discovery completes. Only the first LLMService instance
        to call this actually reads from disk; subsequent instances find
        the shared cache already populated.

        Returns:
            True if cache was loaded, False otherwise
        """
        shared_cache = get_shared_model_cache()
        if shared_cache.has_data():
            # Another LLMService instance already populated the shared cache.
            # Still resolve auto providers for THIS instance's provider list.
            cached = shared_cache.get_any()
            if cached:
                self._resolve_auto_providers(cached)
            return False

        catalog = get_catalog_service()
        cached = catalog.load_cache()
        if cached:
            shared_cache.set_stale(cached)
            logger.info(f"Pre-populated {len(cached)} models from disk cache")

            # Register context limits from cache
            from kestrel_sovereign.agent.token_counter import register_discovered_limits
            register_discovered_limits(cached)

            # Resolve "auto" providers from cached models
            self._resolve_auto_providers(cached)
            return True
        return False

    def _select_discovery_routes(self) -> list[tuple[str, dict]]:
        """Pick one route per vendor to drive discovery.

        Discovery is per-vendor; routes share the catalog. We prefer routes
        whose adapter actually implements ``list_models`` (subscription
        adapters like ClaudeMaxAdapter/CodexAdapter raise NotImplementedError).
        """
        if not hasattr(self, 'providers') or not isinstance(self.providers, list):
            return []

        from .claude_max_adapter import ClaudeMaxAdapter
        from .codex_adapter import CodexAdapter

        routes_by_vendor: dict[str, list[dict]] = {}
        for provider in self.providers:
            vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
            routes_by_vendor.setdefault(vendor, []).append(provider)

        chosen: list[tuple[str, dict]] = []
        for vendor, routes in routes_by_vendor.items():
            # Prefer a route whose adapter is not a subscription wrapper.
            non_sub = [r for r in routes if not isinstance(r.get("adapter"), (ClaudeMaxAdapter, CodexAdapter))]
            chosen.append((vendor, (non_sub or routes)[0]))
        return chosen

    async def _discover_for_vendor_route(self, vendor: str, route: dict) -> List[ModelInfo]:
        """Run discovery for one vendor using the chosen route.

        Dispatches based on adapter class + route config:
          * ``OpenAIAdapter`` with a ``base_url`` (not canonical OpenAI) → query
            that server's ``/models`` with the route's API key, tagged by vendor.
          * ``OpenAIAdapter`` with ``is_local=True`` → query local ``/v1/models``.
          * Any other adapter → ``adapter.list_models()``; tolerate
            ``NotImplementedError`` for subscription-only wrappers.
        """
        from .openai_adapter import OpenAIAdapter

        adapter = route.get("adapter")
        client = route.get("client")
        base_url = route.get("base_url")
        is_local = route.get("is_local")
        route_cfg = dict(route)

        # OpenAI-compatible clients (xai, runpod, groq, llama.cpp, ...) used to
        # be a special case here because the pre-SDK-0.5.0
        # OpenAIAdapter.list_models() rebuilt a fresh client from
        # OPENAI_API_KEY, which returned api.openai.com's catalog for every
        # such route. As of 0.5.0 the contract passes the route's own
        # client into list_models, so canonical-OpenAI and OpenAI-compatible
        # routes can take the same path. The is_local / base_url branches
        # remain because they query the /v1/models endpoint with extra
        # context (server_context_limit, etc.) that the adapter cannot
        # know about.
        if isinstance(adapter, OpenAIAdapter):
            if vendor == "openai" and not base_url:
                return await self._safe_list_models(vendor, adapter, client)
            if base_url and is_local:
                return await self._discover_local_openai_compatible(vendor, route_cfg)
            if base_url:
                return await self._discover_openai_compatible_remote(vendor, route_cfg)

        return await self._safe_list_models(vendor, adapter, client)

    async def _safe_list_models(self, vendor: str, adapter, client) -> List[ModelInfo]:
        """Call adapter.list_models(client) with full error tolerance.

        ``client`` is the route's framework-initialized provider-native
        client; the SDK 0.5.0 contract requires it be passed to
        discovery so authenticated /models endpoints reach the right
        endpoint for routes with custom ``base_url``.
        """
        try:
            if hasattr(adapter, 'list_models'):
                models = await adapter.list_models(client)
                logger.debug("%s: discovered %d models", vendor, len(models))
                return models
        except NotImplementedError:
            logger.debug("%s: adapter.list_models not implemented", vendor)
        except Exception as e:
            logger.warning("%s: model discovery failed: %s", vendor, e)
        return []

    async def _discover_local_openai_compatible(
        self, provider_name: str, provider_config: dict
    ) -> List[ModelInfo]:
        """Discover models from a local OpenAI-compatible server (llama.cpp, etc.)."""
        base_url = provider_config.get("base_url", "")
        if not base_url:
            return []

        # Config-level context_limit override (most reliable for local servers)
        config_context_limit = provider_config.get("context_limit")

        import httpx
        models_url = f"{base_url.rstrip('/')}/models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(models_url)
                resp.raise_for_status()
                data = resp.json()

            # Detect context window from server props (llama.cpp exposes /props)
            server_context_limit = config_context_limit
            if not server_context_limit:
                server_context_limit = await self._query_local_server_context(
                    base_url, httpx
                )

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
                    supports_streaming=True,  # OpenAI-compat servers stream
                    is_featured=True,
                    context_limit=server_context_limit,
                ))
            logger.info(
                f"{provider_name}: discovered {len(results)} models from {models_url}"
                + (f" (context: {server_context_limit})" if server_context_limit else "")
            )
            return results
        except Exception as e:
            logger.warning(f"{provider_name}: local model discovery failed ({models_url}): {e}")
            return []

    async def _discover_openai_compatible_remote(
        self, provider_name: str, provider_config: dict
    ) -> List[ModelInfo]:
        """Discover models from a remote OpenAI-compatible provider (xai, groq, etc.).

        Queries the provider's own /models endpoint using their API key,
        rather than reusing the OpenAI adapter (which would return OpenAI's models).
        """
        import os
        base_url = provider_config.get("base_url", "")
        if not base_url:
            logger.debug(f"{provider_name}: no base_url configured, skipping discovery")
            return []

        # Resolve API key from config or convention
        api_key_env = provider_config.get("api_key_env", f"{provider_name.upper()}_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            logger.debug(f"{provider_name}: {api_key_env} not set, skipping discovery")
            return []

        import httpx
        models_url = f"{base_url.rstrip('/')}/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    models_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            model_list = data.get("data") or data.get("models") or []
            for m in model_list:
                model_id = m.get("id") or m.get("model") or m.get("name", "")
                if not model_id:
                    continue
                results.append(ModelInfo(
                    id=model_id,
                    provider=provider_name,
                    display_name=m.get("name") or model_id,
                    category=ModelCategory.CHAT,
                    supports_tools=True,
                    supports_streaming=True,  # OpenAI-compat servers stream
                    is_featured=False,
                    context_limit=m.get("context_length") or m.get("context_window"),
                    created_at=str(m.get("created")) if m.get("created") else None,
                ))
            logger.info(f"{provider_name}: discovered {len(results)} models from {models_url}")
            return results
        except Exception as e:
            logger.warning(f"{provider_name}: remote model discovery failed ({models_url}): {e}")
            return []

    @staticmethod
    async def _query_local_server_context(base_url: str, httpx_module) -> Optional[int]:
        """Query a local server for its context window size.

        Tries llama.cpp /props endpoint, then /slots, to detect n_ctx.
        """
        stripped = base_url.rstrip("/")
        # Strip /v1 suffix if present — llama.cpp serves /props at root
        if stripped.endswith("/v1"):
            stripped = stripped[:-3]

        for endpoint in ["/props", "/slots"]:
            try:
                async with httpx_module.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{stripped}{endpoint}")
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    # llama.cpp /props: n_ctx in default_generation_settings or top-level
                    if isinstance(data, dict):
                        ctx = (
                            data.get("n_ctx")
                            or data.get("default_generation_settings", {}).get("n_ctx")
                        )
                        if ctx:
                            logger.info(f"Detected context window {ctx} from {endpoint}")
                            return int(ctx)
                    # /slots returns a list; each slot has n_ctx
                    if isinstance(data, list) and data:
                        ctx = data[0].get("n_ctx")
                        if ctx:
                            logger.info(f"Detected context window {ctx} from {endpoint}")
                            return int(ctx)
            except Exception:
                continue
        return None

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
            models: Models to group (uses shared cache if None)

        Returns:
            Dict mapping provider name to list of models
        """
        if models is None:
            models = get_shared_model_cache().get_any() or []

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
            models: Models to filter (uses shared cache if None)

        Returns:
            List of featured models
        """
        if models is None:
            models = get_shared_model_cache().get_any() or []

        return [m for m in models if m.is_featured and not m.is_hidden]

    def clear_model_cache(self):
        """Clear the shared model cache to force rediscovery."""
        get_shared_model_cache().clear()
        logger.info("Model cache cleared")
