"""
Model Discovery Service

Discovers available models from all LLM providers using their APIs.
Enriches results with manual overrides from model_catalog.toml.
Provides in-memory caching and disk-based cache for fast startup.
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

from .model_metadata import ModelInfo, ModelCategory
from .model_catalog import get_catalog_service, ModelCatalogService
from .model_cache import get_shared_model_cache

if TYPE_CHECKING:
    from .embedding_discovery import EmbeddingModelInfo

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
                # Resolve THIS instance's ``model="auto"`` routes against the
                # cached models before returning. A route registered after the
                # cache was populated (e.g. an OpenRouter bootstrap route minted
                # by ``finalize_providers()``) is still ``"auto"`` on a cache
                # hit, and the early return would otherwise skip the
                # ``_resolve_auto_providers`` pass that only runs on a cache
                # miss — leaving it unresolved even when the cached snapshot
                # already contains that vendor's models (#2247).
                self._resolve_auto_providers(cached)
                # Same reason applies to embedding capabilities (#2338): the
                # early return would otherwise skip the reconcile pass that only
                # runs on a cache miss, leaving a dynamically-discovered
                # embedding route's ``supports_embeddings`` unset on this
                # instance even though its embedding catalog is discoverable.
                await self.reconcile_embedding_capabilities(use_cache=True)
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
        # Concrete models pinned in config are operator intent — they must stay
        # featured regardless of recency (a private/just-released model has no
        # useful created_at and would otherwise fall outside the recency top-N).
        # We pass these through enrichment as an explicit featured signal rather
        # than relying on the incoming is_featured flag, which the authoritative
        # featured recomputation treats as untrusted cache state. (#2015 codex r2)
        #
        # Key by the bare VENDOR, not provider['name']: 'name' is the composite
        # route key ("openai:api") on multi-route providers, while discovered
        # ModelInfo.provider is the vendor ("openai") — which is what
        # _apply_recency_visibility looks up. (#2015 codex r3)
        def _vendor_of(provider) -> Optional[str]:
            vendor = provider.get('vendor')
            if vendor:
                return vendor
            name = provider.get('name')
            return name.split(':', 1)[0] if name else None

        configured_featured: Dict[str, set] = {}
        if hasattr(self, 'providers') and isinstance(self.providers, list):
            for provider in self.providers:
                vendor = _vendor_of(provider)
                model_id = provider.get('model')
                if model_id and model_id != "auto" and vendor:
                    configured_featured.setdefault(vendor, set()).add(model_id)
        if hasattr(self, 'providers') and isinstance(self.providers, list):
            for provider in self.providers:
                provider_name = _vendor_of(provider)
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
        all_models = catalog.enrich_models(all_models, pinned_featured=configured_featured)

        # Register discovered context limits into TokenCounter
        from kestrel_sovereign.agent.token_counter import register_discovered_limits
        register_discovered_limits(all_models)

        # Build per-route catalogs for routes whose adapter exposes its OWN
        # serveable set (e.g. openai:plan via codex's models_cache.json),
        # distinct from the vendor's shared discovery. Must precede
        # auto-resolution so those routes resolve against their own catalog.
        await self._build_route_catalogs()

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

        # Fold embedding discovery into route capabilities in the SAME pass as
        # chat ``auto`` resolution (#2338). This is the non-local warm-up path
        # every chat/UI discovery funnels through, so a dynamically-discovered
        # embedding route (e.g. OpenRouter with no TOML pin) has
        # ``supports_embeddings`` set on its provider dict before the sync
        # runtime path (``resolve_embedding_provider``, storage writes) reads
        # it — not just before the UI list. Local-only turns never reach here
        # (they use ``_resolve_local_auto_routes``), so cloud embedding
        # endpoints are not contacted under a privacy-gated turn.
        await self.reconcile_embedding_capabilities(use_cache=True)

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

    async def discover_models_for_route(
        self,
        vendor: str,
        route: Optional[str] = None,
        *,
        use_cache: bool = True,
        featured_only: bool = False,
        category: Optional[ModelCategory] = None,
    ) -> List[ModelInfo]:
        """Discover the models a SPECIFIC ``(vendor, route)`` can actually serve.

        A vendor's routes can expose different model sets: ``anthropic:plan``
        (Claude subscription via OAuth) serves the Claude-CLI set while
        ``anthropic:api`` serves the metered API catalog; ``openai:plan``
        (CodexAdapter → codex app-server) serves the codex set while
        ``openai:api`` serves the full platform catalog. The vendor-keyed
        discovery + ``llm.catalog`` metadata are a DEFAULT; a route whose
        adapter exposes its OWN serveable set (tracked in
        ``self._route_catalogs``, e.g. codex/``openai:plan``) OVERRIDES it, so
        an api-only model never leaks into a plan route's list (#2262).

        Routes without a route-specific catalog (today: everything except
        CodexAdapter — ``ClaudeMaxAdapter``'s ``list_models`` still raises
        ``NotImplementedError``) inherit the vendor's discovered set, which is
        the best available answer until that adapter discovers its own models.

        The route catalog carries only MEMBERSHIP (which model ids the route
        serves); it is still enriched through the vendor-keyed ``llm.catalog``
        so display names / categories / hidden / context overrides apply — the
        catalog decorates, it does not inject membership.
        """
        # Run full discovery so BOTH the vendor catalog and the per-route
        # catalogs (self._route_catalogs) are populated for this instance.
        vendor_models = await self.discover_all_models(
            use_cache=use_cache,
            providers=[vendor],
        )

        route_key = f"{vendor}:{route}" if route else None
        route_catalogs = getattr(self, "_route_catalogs", None) or {}

        if route_key is not None and route_key in route_catalogs:
            # Route-scoped (e.g. codex/openai:plan): the route's own adapter
            # discovery is authoritative. Enrich it with the vendor catalog
            # for metadata, but NEVER fall back to the vendor's membership —
            # an empty route catalog means "this route advertises no explicit
            # set", not "show the api-only catalog".
            catalog = get_catalog_service()
            scoped = route_catalogs[route_key] or []
            models = catalog.enrich_models(list(scoped))
        else:
            models = vendor_models

        return self._filter_models(
            models,
            featured_only=featured_only,
            category=category,
            providers=None,
        )

    async def discover_embedding_models(
        self,
        *,
        vendor: Optional[str] = None,
        route: Optional[str] = None,
        use_cache: bool = True,
    ) -> List["EmbeddingModelInfo"]:
        """Discover embedding models across configured routes (#2338).

        The embedding facet of :meth:`discover_all_models`. Each vendor's
        adapter exposes ``list_embedding_models`` (OpenRouter's dedicated
        ``/embeddings/models`` endpoint, Ollama's ``/api/show`` capability
        check, OpenAI's id-prefix filter); this aggregates them with the same
        per-instance TTL cache semantics as chat discovery.

        Config keys (``embedding_model`` / ``embedding_dim``) are folded in as an
        OVERRIDE/pin (``is_pinned=True``) — surfaced even if the live catalog
        omits them, and never a *prerequisite* for a route to be discovered.

        Args:
            vendor: restrict to this vendor (matches the discovered
                ``provider`` field).
            route: restrict to a single route. Embedding capability is
                route-specific (``openai:api`` embeds, ``openai:plan`` does
                not), so discovery tags each model with its originating
                ``"<vendor>:<route>"`` name; passing ``route`` filters to
                exactly that route (#2338).
        """
        cache = getattr(self, "_embedding_discovery_cache", None)
        if use_cache and cache is not None:
            models = list(cache)
        else:
            models = await self._discover_embedding_models_uncached()
            self._embedding_discovery_cache = list(models)

        if vendor:
            models = [m for m in models if m.provider == vendor]
        if route:
            models = [m for m in models if m.route == route]
        return models

    async def _discover_embedding_models_uncached(self) -> List["EmbeddingModelInfo"]:
        from .embedding_discovery import EmbeddingModelInfo

        # Discover per ACTUAL route, not one-route-per-vendor: embedding
        # capability is route-specific in production, so collapsing by vendor
        # (as chat catalog discovery does) would let one route's embeddings be
        # attributed to a sibling route that can't embed — e.g. openai:api's
        # models leaking onto openai:plan/codex (#2338). Subscription adapters
        # (codex/claude-max) simply lack ``list_embedding_models`` and return
        # []; the ``hasattr`` guard in ``_discover_embedding_for_route`` makes
        # querying every route cheap and correct.
        providers = getattr(self, "providers", None)
        if not isinstance(providers, list):
            providers = []

        tasks = []
        for provider in providers:
            vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
            tasks.append(self._discover_embedding_for_route(vendor, provider))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        discovered: List[EmbeddingModelInfo] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Embedding discovery failed: {result}")
            elif isinstance(result, list):
                discovered.extend(result)

        # Fold config pins in as overrides. A pin is declared on a specific
        # ROUTE, so match/synthesize with route identity (#2338) — otherwise a
        # pin on ``openai:api`` would attach to whichever same-vendor route
        # discovery happened to return first. A pinned model that discovery also
        # returned is marked pinned (operator-featured); a pin discovery missed
        # is added synthetically so the settings UI still offers it.
        seen = {(m.route, m.id) for m in discovered}
        if hasattr(self, "providers") and isinstance(self.providers, list):
            for provider in self.providers:
                vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
                route_name = provider.get("name") or vendor
                caps = provider.get("capabilities") or {}
                pinned_model = provider.get("embedding_model") or caps.get("embedding_model")
                pinned_dim = provider.get("embedding_dim") or caps.get("embedding_dim")
                if not pinned_model:
                    continue
                match = next(
                    (
                        m for m in discovered
                        if m.route == route_name and m.id == pinned_model
                    ),
                    None,
                )
                if match is not None:
                    match.is_pinned = True
                    if pinned_dim and match.native_dim is None:
                        match.native_dim = int(pinned_dim)
                elif (route_name, pinned_model) not in seen:
                    discovered.append(EmbeddingModelInfo(
                        id=pinned_model,
                        provider=vendor,
                        route=route_name,
                        native_dim=int(pinned_dim) if pinned_dim else None,
                        is_pinned=True,
                    ))
                    seen.add((route_name, pinned_model))

        return discovered

    async def _discover_embedding_for_route(
        self, vendor: str, route: dict
    ) -> List["EmbeddingModelInfo"]:
        """Call one route's adapter embedding facet with error tolerance."""
        adapter = route.get("adapter")
        client = route.get("client")
        route_name = route.get("name") or vendor
        if adapter is None or not hasattr(adapter, "list_embedding_models"):
            return []
        try:
            models = await adapter.list_embedding_models(client)
            # Retag to the vendor key so aggregation/filtering is consistent
            # even if an adapter reports its own name, AND stamp the originating
            # route so capability advertisement stays route-specific (#2338).
            for m in models:
                m.provider = vendor
                m.route = route_name
            logger.debug("%s: discovered %d embedding models", route_name, len(models))
            return models or []
        except NotImplementedError:
            return []
        except Exception as e:
            logger.warning("%s: embedding discovery failed: %s", vendor, e)
            return []

    async def route_advertises_embeddings(self, provider: dict) -> bool:
        """True when discovery finds ≥1 embedding model for ``provider``'s route (#2338).

        Capability is advertised because embeddings were DISCOVERED, not because
        an operator pinned a model. A config pin still counts (it's folded into
        discovery as an override), so a pinned route stays truthful too.

        Matching is ROUTE-specific: only models discovered on THIS exact
        ``"<vendor>:<route>"`` count, so a vendor's embedding-capable route
        (``openai:api``) never flips capability on for a sibling route that
        can't embed (``openai:plan``/codex). Falls back to a vendor match only
        when the provider carries no ``name`` to key on.
        """
        route_name = provider.get("name")
        if route_name:
            models = await self.discover_embedding_models(route=route_name)
            return len(models) > 0
        vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
        models = await self.discover_embedding_models(vendor=vendor)
        return len(models) > 0

    async def shared_embedding_space_candidates(self) -> List["EmbeddingModelInfo"]:
        """Compute shared local+cloud embedding models by intersection (#2290/#2337).

        The "Universal" shared-space option must be COMPUTED, not hardcoded to
        qwen3: intersect the normalized ids of discovered LOCAL embedding models
        with discovered CLOUD embedding models. A model present on both sides
        (same weights local and in the cloud) can share one coordinate space.
        """
        from .embedding_discovery import normalize_embedding_model_id

        if not hasattr(self, "providers") or not isinstance(self.providers, list):
            return []

        local_vendors = {
            (p.get("vendor") or p.get("name", "").split(":", 1)[0])
            for p in self.providers
            if p.get("is_local")
        }

        all_models = await self.discover_embedding_models()
        local_norm: Dict[str, "EmbeddingModelInfo"] = {}
        cloud_norm: set = set()
        for m in all_models:
            norm = normalize_embedding_model_id(m.id)
            if m.provider in local_vendors:
                local_norm[norm] = m
            else:
                cloud_norm.add(norm)

        return [m for norm, m in local_norm.items() if norm in cloud_norm]

    async def universal_embedding_space_options(self) -> List[dict]:
        """Featured "Universal" options: shared models WITH member routes (#2337).

        :meth:`shared_embedding_space_candidates` proves a model is usable both
        locally and in the cloud, but returns only the LOCAL entry — not enough
        for the embeddings UI's featured "Universal — <model> (local + cloud,
        one search space)" option, which on selection must pin the model on
        EVERY member route. This enriches each shared candidate with its member
        routes, each carrying that route's OWN model id (the same weights are
        ``qwen3-embedding-0.6b`` on Ollama but ``qwen/qwen3-embedding-0.6b`` on
        OpenRouter — guided setup must pin each route's real slug, not one id
        forced across both). Computed by intersection, never hardcoded to qwen3.

        Each option::

            {"model": "<display id>", "display_name": str,
             "dim": int|None, "dim_options": [int, ...],
             "members": [{"route": "<vendor>:<route>", "model": "<route slug>",
                          "provider": str, "is_local": bool,
                          "native_dim": int|None, "dim_options": [int, ...]}, ...]}
        """
        from .embedding_discovery import normalize_embedding_model_id

        shared = await self.shared_embedding_space_candidates()
        if not shared:
            return []

        all_models = await self.discover_embedding_models()
        by_norm: Dict[str, List["EmbeddingModelInfo"]] = {}
        for m in all_models:
            by_norm.setdefault(normalize_embedding_model_id(m.id), []).append(m)

        # Map route name -> is_local so members can be labelled without a second
        # discovery pass.
        local_routes = {
            p.get("name")
            for p in (self.providers if isinstance(self.providers, list) else [])
            if p.get("is_local")
        }

        options: List[dict] = []
        for cand in shared:
            norm = normalize_embedding_model_id(cand.id)
            members = []
            seen_routes: set = set()
            for m in by_norm.get(norm, []):
                if not m.route or m.route in seen_routes:
                    continue
                seen_routes.add(m.route)
                members.append(
                    {
                        "route": m.route,
                        "model": m.id,
                        "provider": m.provider,
                        "is_local": m.route in local_routes,
                        "native_dim": m.native_dim,
                        "dim_options": list(m.dim_options),
                    }
                )
            # Only a model reachable on at least two DISTINCT routes is universal.
            if len(members) < 2:
                continue
            options.append(
                {
                    "model": cand.id,
                    "display_name": cand.display_name,
                    "dim": cand.native_dim,
                    "dim_options": list(cand.dim_options),
                    "members": members,
                }
            )
        return options

    async def resolve_default_embedding_model(
        self, provider: dict
    ) -> Optional["EmbeddingModelInfo"]:
        """Pick a route's default embedding model, mirroring chat ``auto`` (#2338).

        Resolution order matches chat auto-selection
        (:meth:`_select_auto_model_for_route`): an explicit config pin wins, then
        the route's ``selection_hints`` (substring patterns from config, no
        hardcoded ids), then the first discovered model. Returns ``None`` when
        discovery found nothing for the route — the caller then has no embedding
        capability to advertise, which is truthful.
        """
        route_name = provider.get("name")
        vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
        if route_name:
            candidates = await self.discover_embedding_models(route=route_name)
        else:
            candidates = await self.discover_embedding_models(vendor=vendor)
        if not candidates:
            return None

        # A config pin is operator intent — honour it before hint matching.
        pinned = next((m for m in candidates if m.is_pinned), None)
        if pinned is not None:
            return pinned

        for hint in provider.get("selection_hints") or []:
            hint_lower = str(hint).lower()
            match = next(
                (
                    m for m in candidates
                    if hint_lower in m.id.lower()
                    or hint_lower in (m.display_name or "").lower()
                ),
                None,
            )
            if match is not None:
                return match

        return candidates[0]

    async def reconcile_embedding_capabilities(self, *, use_cache: bool = True) -> None:
        """Fold live embedding discovery into each route's static capabilities (#2338).

        The embedding counterpart to :meth:`_resolve_auto_providers`. The
        runtime embedding path (``resolve_embedding_provider`` /
        ``_validate_embedding_route``) and the storage embedding resolver read
        the SYNC ``provider["capabilities"]["supports_embeddings"]`` flag — they
        can't await discovery on every write. Chat solves this by resolving
        ``model="auto"`` into ``provider["model"]`` once, post-discovery; this
        does the same for embeddings: every route whose discovery returned ≥1
        embedding model gets ``supports_embeddings=True`` (plus a default
        ``embedding_model``/``embedding_dim`` when none was pinned) written into
        its capabilities, so a dynamically-discovered route (e.g. OpenRouter
        with no TOML pin) becomes selectable via ``/api/embedding/settings`` and
        usable by storage — not just visible in ``/api/models``.

        Route-specific: capability is only turned ON for the exact route that
        discovered embeddings (never a same-vendor sibling), and it is only ever
        turned ON — a config pin / prior TOML capability is never downgraded
        here (the #2326 set-time probe owns dead-model detection).
        """
        providers = getattr(self, "providers", None)
        if not isinstance(providers, list):
            return
        try:
            discovered = await self.discover_embedding_models(use_cache=use_cache)
        except Exception as e:  # pragma: no cover - never break init on discovery
            logger.debug("embedding capability reconcile skipped: %s", e)
            return

        by_route: Dict[str, list] = {}
        for m in discovered:
            if m.route:
                by_route.setdefault(m.route, []).append(m)

        for provider in providers:
            route_name = provider.get("name")
            # Only advertise where THIS route actually discovered embeddings —
            # do not fall back to a vendor match, which is the exact
            # false-advertisement failure this per-route path prevents.
            if not route_name or route_name not in by_route:
                continue
            caps = provider.get("capabilities")
            if not isinstance(caps, dict):
                caps = {}
                provider["capabilities"] = caps
            caps["supports_embeddings"] = True
            default = await self.resolve_default_embedding_model(provider)
            if default is not None:
                if not caps.get("embedding_model"):
                    caps["embedding_model"] = default.id
                    logger.info(
                        "Auto-resolved embedding model for %s: %s",
                        route_name,
                        default.id,
                    )
                if not caps.get("embedding_dim") and default.native_dim:
                    caps["embedding_dim"] = default.native_dim

    def clear_embedding_discovery_cache(self) -> None:
        """Drop the per-instance embedding-discovery cache to force rediscovery."""
        self._embedding_discovery_cache = None

    async def _resolve_local_auto_routes(self) -> None:
        """Resolve ``model="auto"`` LOCAL routes WITHOUT contacting cloud.

        The force-local-only generation path (ISOLATED/EPHEMERAL privacy
        session) needs a cold-cache local route's ``"auto"`` to resolve to a
        concrete model, but must not enumerate or contact cloud vendors
        (privacy) nor write the shared/disk cache with a local-only model set
        (which would poison it for a later cloud request). So discovery here is
        scoped to local routes only, and ``_resolve_auto_providers`` mutates the
        provider list in place against just those models — cloud routes (which
        the force-local filter drops from the turn anyway) keep ``"auto"``.
        Unlike :meth:`discover_all_models`, this never calls
        ``shared_cache.set``/``write_cache``.
        """
        if not hasattr(self, "providers") or not isinstance(self.providers, list):
            return
        # Iterate the ACTUAL local provider routes — NOT _select_discovery_routes,
        # which collapses to one route per vendor and may pick the cloud route,
        # dropping a local route that shares that vendor (leaving it "auto").
        local_providers = [p for p in self.providers if p.get("is_local")]
        if not local_providers:
            return
        models: list = []
        for route in local_providers:
            vendor = route.get("vendor") or route.get("name", "").split(":", 1)[0]
            try:
                models.extend(await self._discover_for_vendor_route(vendor, route))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Local-only discovery failed for %s: %s", route.get("name"), exc
                )
        if models:
            # Mutate ONLY local routes — never a cloud route that happens to
            # share a vendor with a local route (else a later non-local request
            # would send this local-only model id to the cloud route).
            self._resolve_auto_providers(models, only_providers=local_providers)

    def _resolve_auto_providers(self, models: list, only_providers: Optional[list] = None) -> None:
        """Resolve routes whose configured model is ``"auto"`` using discovered models.

        Most routes inherit the model catalog of their *vendor*; selection is
        driven by the route's ``selection_hints`` (config), then rank heuristics.

        A route whose adapter exposes its OWN serveable catalog (e.g.
        ``openai:plan`` via codex's ``models_cache.json``) resolves against
        THAT catalog instead — its serveable set differs from the vendor's
        full discovery (``openai:api``'s full OpenAI catalog), so resolving
        against the shared catalog picks models codex rejects.

        ``only_providers``: restrict mutation to this exact subset of route
        dicts instead of every provider. The local-only resolver
        (:meth:`_resolve_local_auto_routes`) passes its local routes here so a
        cloud route that *shares a vendor* with a ``local = true`` route is NOT
        resolved to a locally-discovered model id — which would later send a
        local-only model to the cloud route.
        """
        if not hasattr(self, 'providers') or not isinstance(self.providers, list):
            return
        targets = only_providers if only_providers is not None else self.providers
        # On the sync cache-hit paths (``_load_from_disk_cache``) route catalogs
        # haven't been built by the async discovery phase yet — build them now
        # so route-scoped routes still resolve correctly. No-op if already built.
        self._ensure_route_catalogs_sync()
        route_catalogs = getattr(self, "_route_catalogs", None) or {}
        for provider in targets:
            if provider.get("model") != "auto":
                continue
            vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
            route_key = provider.get("name")
            if route_key in route_catalogs:
                # Route-scoped (e.g. codex/openai:plan): resolve ONLY against the
                # route's own serveable catalog — never the vendor catalog. An
                # empty catalog yields no candidate, leaving ``auto`` unresolved
                # so the adapter sends no model and codex uses its own default.
                candidates = [
                    m for m in route_catalogs[route_key]
                    if m.category == ModelCategory.CHAT and not m.is_hidden
                ]
            else:
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

    def _route_specific_catalog_adapters(self):
        """Yield ``(route_key, adapter)`` for routes that expose their OWN catalog.

        A route is "route-specific" when its adapter overrides ``list_models``
        away from the base subscription contract (which raises
        ``NotImplementedError``). Today only ``CodexAdapter`` (openai:plan)
        qualifies; ``ClaudeMaxAdapter`` (anthropic:plan) still raises, so it
        is correctly skipped and keeps inheriting the vendor catalog.
        """
        if not hasattr(self, 'providers') or not isinstance(self.providers, list):
            return
        try:
            from .codex_adapter import CodexAdapter
        except Exception:  # pragma: no cover - import-safety
            return
        for provider in self.providers:
            adapter = provider.get("adapter")
            route_key = provider.get("name")
            if route_key and isinstance(adapter, CodexAdapter):
                yield route_key, adapter

    async def _build_route_catalogs(self) -> None:
        """Populate ``self._route_catalogs`` from route-specific adapters.

        Keyed by route name (e.g. ``"openai:plan"``). A route-specific route is
        ALWAYS entered (even with an empty catalog) so it never falls back to
        the vendor's shared discovery — that fallback would *resolve* the route
        to an API-only model codex rejects (e.g. ``gpt-5.5-pro``). An empty
        catalog instead leaves ``auto`` unresolved, so the adapter sends no
        model and codex uses its own serveable subscription default (e.g. on a
        fresh install before ``models_cache.json`` exists).
        """
        self._route_catalogs = await self._collect_route_catalogs()

    async def _collect_route_catalogs(self) -> dict:
        """Build and RETURN the route-scoped catalog map (no self mutation).

        Split out from :meth:`_build_route_catalogs` so the sync helper can
        drive it on a worker thread and assign the result on the owning thread
        (avoids cross-thread mutation of ``self._route_catalogs``).
        """
        catalogs: dict[str, list] = {}
        for route_key, adapter in self._route_specific_catalog_adapters():
            try:
                models = await adapter.list_models()
            except NotImplementedError:
                continue
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("route %s: catalog build failed: %s", route_key, e)
                models = []
            # Register even when empty: membership marks the route as
            # route-scoped so it never inherits the vendor catalog.
            catalogs[route_key] = models or []
            logger.debug("route %s: %d route-specific models", route_key, len(models or []))
        return catalogs

    def _ensure_route_catalogs_sync(self) -> None:
        """Synchronously ensure ``self._route_catalogs`` is populated.

        Used by the sync disk-cache resolution path (``_load_from_disk_cache``
        runs in ``LLMService.__init__``, outside any event loop) AND by the
        sync ``set_model_preference`` validator, which may run *inside* a
        running loop (the ``set_model`` tool is async). Route-specific adapters
        (codex) build their catalog from a local JSON cache with no awaits, so
        the async builder is loop-independent and safe to drive to completion
        on a worker thread when a loop is already running. No-op if catalogs
        were already built by the async discovery phase.
        """
        import asyncio

        if getattr(self, "_route_catalogs", None) is not None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to drive the async builder synchronously.
            try:
                asyncio.run(self._build_route_catalogs())
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("sync route-catalog build skipped: %s", e)
                self._route_catalogs = {}
        else:
            # A loop is already running. We can't ``asyncio.run`` here, but the
            # route-specific builder only reads a local JSON cache (no awaits
            # that depend on THIS loop), so drive it to completion on a worker
            # thread with its own loop. This yields the REAL catalog rather
            # than an empty placeholder, so the validator can distinguish a
            # model that codex can't serve from "catalog not built yet" — the
            # placeholder otherwise let an api-only model land on the plan
            # route (gpt-5.5-pro). On any failure, fall back to registering
            # every route-specific route as EMPTY so it stays route-scoped
            # (never inherits the vendor catalog); the async discovery phase
            # later overwrites with the real catalog.
            import concurrent.futures

            def _build_in_thread() -> dict:
                return asyncio.run(self._collect_route_catalogs())

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    self._route_catalogs = pool.submit(_build_in_thread).result()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("threaded route-catalog build skipped: %s", e)
                self._route_catalogs = {
                    route_key: []
                    for route_key, _ in self._route_specific_catalog_adapters()
                }

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
