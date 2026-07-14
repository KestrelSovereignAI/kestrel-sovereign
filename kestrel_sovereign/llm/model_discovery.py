"""
Model Discovery Service

Discovers available models from all LLM providers using their APIs.
Enriches results with manual overrides from model_catalog.toml.
Provides in-memory caching and disk-based cache for fast startup.
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING

from .model_metadata import ModelInfo, ModelCategory
from .model_catalog import get_catalog_service, ModelCatalogService
from .model_cache import get_shared_model_cache

if TYPE_CHECKING:
    from .embedding_discovery import EmbeddingModelInfo

logger = logging.getLogger(__name__)


def _model_offers_dim(model: "EmbeddingModelInfo", dim: int) -> bool:
    """True when a discovered model can serve vectors of width ``dim`` (#2366).

    Matches the model's native dimension or any of its offered truncation
    options (Matryoshka range), so a qwen3-embedding model that truncates to
    768 satisfies a 768-wide deployment column.
    """
    try:
        target = int(dim)
    except (TypeError, ValueError):
        return False
    if model.native_dim == target:
        return True
    return target in (model.dim_options or [])


def _resolve_deployment_embedding_dim() -> Optional[int]:
    """Return the deployment's effective embedding dim (``KESTREL_EMBEDDING_DIM``).

    Best-effort — returns ``None`` when the value can't be resolved so the
    caller simply skips the dim-preference branch (#2366).
    """
    try:
        from kestrel_sovereign.storage.sqla.conversation_message import (
            resolve_embedding_dim,
        )

        dim = resolve_embedding_dim()
        return int(dim) if dim else None
    except Exception:  # pragma: no cover - defensive
        return None


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
        providers: Optional[List[str]] = None,
        stale_while_revalidate: bool = False,
    ) -> List[ModelInfo]:
        """
        Discover all available models from all configured providers.

        Args:
            use_cache: Whether to use cached results if available
            featured_only: Only return featured models
            category: Filter by category (CHAT, EMBEDDING, etc.)
            providers: Filter by provider names
            stale_while_revalidate: When the cache is expired but populated,
                return its last catalog immediately and coalesce one background
                provider refresh. Intended for latency-sensitive catalog UI;
                live routing remains server-authoritative.

        Returns:
            List of ModelInfo objects, enriched with catalog data
        """
        # Check shared process-wide cache
        shared_cache = get_shared_model_cache()
        if use_cache:
            async def _return_cached(cached_models: List[ModelInfo]) -> List[ModelInfo]:
                logger.debug("Using shared model cache")
                # Resolve THIS instance's ``model="auto"`` routes against the
                # cached models before returning. A route registered after the
                # cache was populated (e.g. an OpenRouter bootstrap route minted
                # by ``finalize_providers()``) is still ``"auto"`` on a cache
                # hit, and the early return would otherwise skip the
                # ``_resolve_auto_providers`` pass that only runs on a cache
                # miss — leaving it unresolved even when the cached snapshot
                # already contains that vendor's models (#2247).
                self._resolve_auto_providers(cached_models)
                # Rebuild the route-keyed chat-id snapshot from the cached
                # catalog BEFORE reconciling (#2433). Embedding discovery
                # id-filters this snapshot instead of issuing its own
                # ``/v1/models`` request; on a fresh discovery it is populated
                # inline, but a shared-cache HIT skips that path entirely, so
                # without rebuilding it here a production multi-agent host would
                # reconcile against a stale/absent snapshot and reintroduce the
                # per-route embedding ``/models`` probe (the RunPod 404).
                self._snapshot_chat_models_by_route(cached_models)
                # Same reason applies to embedding capabilities (#2338): the
                # early return would otherwise skip the reconcile pass that only
                # runs on a cache miss, leaving a dynamically-discovered
                # embedding route's ``supports_embeddings`` unset on this
                # instance even though its embedding catalog is discoverable.
                await self.reconcile_embedding_capabilities(use_cache=True)
                return self._filter_models(
                    cached_models,
                    featured_only=featured_only,
                    category=category,
                    providers=providers,
                )

            cached = shared_cache.get()
            if cached is not None:
                return await _return_cached(cached)

            if (
                not stale_while_revalidate
                and await shared_cache.wait_for_refresh()
            ):
                cached = shared_cache.get()
                if cached is not None:
                    return await _return_cached(cached)

            stale = shared_cache.get_any() if stale_while_revalidate else None
            if stale is not None:
                logger.debug("Serving stale model catalog while refreshing")
                self._resolve_auto_providers(stale)
                shared_cache.refresh_in_background(
                    lambda: self.discover_all_models(use_cache=False)
                )
                return self._filter_models(
                    stale,
                    featured_only=featured_only,
                    category=category,
                    providers=providers,
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

        # Snapshot the discovered model ids so embedding discovery can id-filter
        # them WITHOUT a second network request (#2433). OpenAI's ``/v1/models``
        # listing already includes ``text-embedding-*`` ids, so a generic
        # OpenAI-compatible route derives its embedding catalog from what chat
        # discovery fetched — a chat-only route (RunPod vLLM whose model-list
        # 404s) is never probed for embeddings a second time.
        self._snapshot_chat_models_by_route(all_models)

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

    def _snapshot_chat_models_by_route(self, models: List[ModelInfo]) -> None:
        """Snapshot discovered chat ids keyed by the ROUTE that fetched them (#2433).

        Embedding discovery reuses a route's OWN chat listing to id-filter
        embedding models without a second ``/v1/models`` request. The snapshot
        is keyed by ROUTE (``provider["name"]``), NOT vendor: chat discovery
        runs one route per vendor (:meth:`_select_discovery_routes`), so the
        listing reflects exactly THAT route's serveable set. Keying by vendor
        would hand the same list to every sibling route under the vendor —
        letting one route's discovered embedding ids retag onto a chat-only
        sibling that can't embed. So we map each discovery route's name to its
        vendor's discovered ids, and record an explicit ``[]`` for a discovery
        route whose listing came back empty (RunPod's 404) — a KNOWN-empty
        listing that still stops the re-probe. A sibling route that did NOT
        drive discovery gets no entry, so :meth:`_discover_embedding_models_uncached`
        passes ``None`` and its adapter fetches its OWN listing rather than
        reusing another route's.
        """
        by_vendor: Dict[str, List[str]] = {}
        for m in models:
            by_vendor.setdefault(m.provider, []).append(m.id)
        by_route: Dict[str, List[str]] = {}
        for vendor, route in self._select_discovery_routes():
            route_name = route.get("name")
            if route_name:
                by_route[route_name] = by_vendor.get(vendor, [])
        self._chat_models_by_route = by_route

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
            models = await self._discover_embedding_models_coalesced()

        if vendor:
            models = [m for m in models if m.provider == vendor]
        if route:
            models = [m for m in models if m.route == route]
        return models

    async def _discover_embedding_models_coalesced(
        self,
    ) -> List["EmbeddingModelInfo"]:
        """Single-flight embedding discovery (#2433).

        The "twice" in the original repro is a concurrent refresh and invoke
        both probing the same routes. Coalesce concurrent discovery behind one
        in-flight future so overlapping callers share a single provider sweep
        (and a single INFO line per unsupported route) instead of each issuing
        its own.
        """
        inflight = getattr(self, "_embedding_discovery_inflight", None)
        if inflight is not None and not inflight.done():
            return await inflight

        task = asyncio.ensure_future(self._discover_embedding_models_uncached())
        self._embedding_discovery_inflight = task
        try:
            models = await task
        finally:
            if getattr(self, "_embedding_discovery_inflight", None) is task:
                self._embedding_discovery_inflight = None
        self._embedding_discovery_cache = list(models)
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

        # Chat model ids already discovered this run, keyed by ROUTE (#2433).
        # Generic OpenAI-compatible routes id-filter their OWN route's listing
        # instead of issuing a second network request, so a chat-only route is
        # never probed twice. Keying by route (not vendor) stops one route's
        # discovered embedding ids from retagging onto a sibling route under the
        # same vendor. A discovery route whose chat listing was empty carries an
        # explicit ``[]`` entry (known-empty → no re-probe); a route with no
        # entry (a sibling that didn't drive discovery, or a direct call before
        # any chat discovery) gets ``None`` so its adapter fetches its own
        # listing rather than reusing another route's.
        chat_by_route = getattr(self, "_chat_models_by_route", None) or {}

        tasks = []
        for provider in providers:
            vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
            route_name = provider.get("name")
            tasks.append(
                self._discover_embedding_for_route(
                    vendor, provider, chat_by_route.get(route_name)
                )
            )

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
                # A GENUINE operator pin is either config (``provider["embedding_model"]``
                # / a route-level TOML capability) or a runtime override — both land
                # in ``capabilities`` WITHOUT the auto-resolved marker. A default
                # that ``resolve_route_embedding_model`` / ``reconcile_embedding_capabilities``
                # wrote back into ``capabilities`` is NOT a pin (#2372): treating it
                # as one after a cache invalidation would freeze the route on a stale
                # model/dim and block the corpus/deployment fallback the resolver
                # order promises.
                auto_resolved = bool(caps.get("embedding_model_auto_resolved"))
                caps_model = None if auto_resolved else caps.get("embedding_model")
                caps_dim = None if auto_resolved else caps.get("embedding_dim")
                pinned_model = provider.get("embedding_model") or caps_model
                pinned_dim = provider.get("embedding_dim") or caps_dim
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

    @staticmethod
    def _route_claims_embedding_support(route: dict) -> bool:
        """True when a route EXPLICITLY claims embedding support (#2433).

        A claim is operator intent: a configured ``embedding_model`` (provider
        level or a genuine capability pin) or a pinned ``supports_embeddings``.
        A value the resolver auto-wrote back (``embedding_model_auto_resolved``)
        is discovery-derived, NOT a claim — so it does not escalate a probe
        failure to a loud ERROR. Discovery advertisement is circular with a
        claim (#2338), so we never treat "advertises embeddings" as a claim.
        """
        if route.get("embedding_model"):
            return True
        caps = route.get("capabilities") or {}
        if caps.get("embedding_model_auto_resolved"):
            return False
        return bool(caps.get("embedding_model") or caps.get("supports_embeddings"))

    async def _discover_embedding_for_route(
        self, vendor: str, route: dict, chat_model_ids: Optional[List[str]] = None
    ) -> List["EmbeddingModelInfo"]:
        """Call one route's adapter embedding facet with error tolerance (#2433).

        Generic OpenAI-compatible adapters id-filter the already-fetched chat
        listing (``chat_model_ids``) with no extra request; adapters with a
        dedicated embedding source make their own call. An empty/absent catalog
        is a normal unsupported-capability state (single INFO line, negative
        cached with the aggregate discovery cache) — an ERROR with traceback is
        reserved for a route that explicitly claims embedding support and then
        fails unexpectedly.
        """
        adapter = route.get("adapter")
        client = route.get("client")
        route_name = route.get("name") or vendor
        if adapter is None or not hasattr(adapter, "list_embedding_models"):
            return []
        try:
            if getattr(adapter, "derives_embeddings_from_chat_listing", False):
                models = await adapter.list_embedding_models(
                    client, chat_models=chat_model_ids
                )
            else:
                models = await adapter.list_embedding_models(client)
            # Retag to the vendor key so aggregation/filtering is consistent
            # even if an adapter reports its own name, AND stamp the originating
            # route so capability advertisement stays route-specific (#2338).
            for m in models:
                m.provider = vendor
                m.route = route_name
            if models:
                logger.debug(
                    "%s: discovered %d embedding models", route_name, len(models)
                )
            else:
                logger.info("%s: no embedding models discovered", route_name)
            return models or []
        except NotImplementedError:
            return []
        except Exception as e:
            if self._route_claims_embedding_support(route):
                logger.error(
                    "%s: embedding discovery failed for a route that claims "
                    "embedding support: %s",
                    route_name,
                    e,
                    exc_info=True,
                )
            else:
                logger.info(
                    "%s: embedding discovery unavailable (route advertises no "
                    "embeddings): %s",
                    route_name,
                    e,
                )
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
        self,
        provider: dict,
        *,
        corpus_profile: Optional[Dict[str, Any]] = None,
        deployment_dim: Optional[int] = None,
    ) -> Tuple[Optional["EmbeddingModelInfo"], Optional[int]]:
        """Pick a route's default embedding ``(model, dim)``, mirroring chat ``auto`` (#2338).

        Resolution order (#2366 — continuity beats catalog order):

        1. An explicit config pin (operator intent) always wins.
        2. A discovered model matching the DB's DOMINANT existing embedding
           profile — same model identity (:func:`normalize_embedding_model_id`),
           or the same shared space via a verified #2290 pin. A non-empty corpus
           should keep landing new memories in the space old memories already
           live in, rather than silently splitting recall across two spaces.
        3. A discovered model whose dimension matches ``deployment_dim``
           (``kestrel_embedding_dim``) — no column migration needed.
        4. Only then the route's ``selection_hints`` (substring patterns from
           config, no hardcoded ids), then the first discovered model.

        Every non-``None`` model is returned WITH the dim the branch resolved it
        on — never a bare model (#2376). A corpus-matched auto-resolution means
        "keep this space", and a space is ``<model>@<dim>``, so the dominant
        profile's dim IS the answer even when discovery never exposed the model's
        Matryoshka range (``dim_options`` empty). The dim-compatibility branch
        returns the matched ``deployment_dim``; the hint/catalog fallback returns
        the model's native/advertised dim. A resolved embedding-capable state
        with ``embedding_dim: None`` is invalid by construction — it embeds at
        the provider's native width and the column guard then refuses the write.

        ``corpus_profile`` is the dominant existing profile as a dict
        (``{"provider", "model", "dim", "space_id", ...}``) or ``None`` when the
        corpus is empty / unreadable. ``deployment_dim`` is the deployment's
        effective embedding dimension. Returns ``(None, None)`` when discovery
        found nothing for the route — the caller then has no embedding capability
        to advertise, which is truthful.
        """
        route_name = provider.get("name")
        vendor = provider.get("vendor") or provider.get("name", "").split(":", 1)[0]
        if route_name:
            candidates = await self.discover_embedding_models(route=route_name)
        else:
            candidates = await self.discover_embedding_models(vendor=vendor)
        if not candidates:
            return None, None

        # A config pin is operator intent — honour it before everything else.
        # Its own dim lives in ``capabilities`` and the route caller keeps it;
        # native_dim is a truthful default the caller only falls back to.
        pinned = next((m for m in candidates if m.is_pinned), None)
        if pinned is not None:
            return pinned, pinned.native_dim

        # #2366 (1) — prefer continuity with the existing corpus space. The
        # corpus's dominant dim IS the resolved dim: a corpus match keeps the
        # existing space (``<model>@<dim>``), so 768 old rows pin 768 new ones —
        # even when discovery couldn't advertise 768 as a dim option (#2376).
        if corpus_profile:
            corpus_match = self._match_corpus_profile(candidates, corpus_profile)
            if corpus_match is not None:
                corpus_dim = corpus_profile.get("dim")
                dim = int(corpus_dim) if corpus_dim else corpus_match.native_dim
                return corpus_match, dim

        # #2366 (2) — otherwise prefer a model that fits the deployment's vector
        # column dimension, so recall stays available without a re-embed. The
        # dim it matched on is the resolved dim (#2376).
        if deployment_dim:
            dim_match = next(
                (m for m in candidates if _model_offers_dim(m, deployment_dim)),
                None,
            )
            if dim_match is not None:
                return dim_match, int(deployment_dim)

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
                return match, match.native_dim

        return candidates[0], candidates[0].native_dim

    def _match_corpus_profile(
        self, candidates: List["EmbeddingModelInfo"], corpus_profile: Dict[str, Any]
    ) -> Optional["EmbeddingModelInfo"]:
        """Return the discovered model that preserves the corpus space (#2366).

        Matches on the dominant profile's model identity first (same weights,
        cross-route via :func:`normalize_embedding_model_id`), then on a verified
        #2290 shared-space pin whose ``space_id`` equals the corpus space — a
        member route of that pin lands new rows in the same coordinate space.
        """
        from .embedding_discovery import normalize_embedding_model_id

        dom_model = normalize_embedding_model_id(corpus_profile.get("model") or "")
        if dom_model:
            match = next(
                (
                    m for m in candidates
                    if normalize_embedding_model_id(m.id) == dom_model
                ),
                None,
            )
            if match is not None:
                return match

        dom_space = corpus_profile.get("space_id")
        pins = getattr(self, "_embedding_space_pins", None)
        verified = getattr(self, "_verified_space_pins", None)
        if dom_space and pins:
            for m in candidates:
                for pin in pins:
                    if getattr(pin, "space_id", None) != dom_space:
                        continue
                    if not pin.covers(m.route, m.route.split(":", 1)[0]):
                        continue
                    parity = verified.get(pin.name) if verified else None
                    if parity is not None and getattr(parity, "passed", False):
                        return m
        return None

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

        # #2372 — the corpus/deployment continuity signals are route-independent,
        # so resolve them ONCE for the whole sweep rather than once per route.
        # The per-provider fan-out (one corpus DB lookup per route) regressed
        # lifespan startup past the 5s test-fixture timeout.
        corpus_profile = await self._get_corpus_embedding_profile()
        deployment_dim = _resolve_deployment_embedding_dim()

        for provider in providers:
            route_name = provider.get("name")
            # Only advertise where THIS route actually discovered embeddings —
            # do not fall back to a vendor match, which is the exact
            # false-advertisement failure this per-route path prevents.
            if not route_name or route_name not in by_route:
                continue
            # #2372 — funnel every discovering route through the SINGLE resolver
            # so capability writes match the settings GET / route-model echo /
            # reindex target exactly. It honours #2366's order (pin → corpus →
            # deployment-dim → hints), computes the dim with corpus/deployment
            # continuity (not just ``native_dim``), persists
            # ``supports_embeddings``/``embedding_model``/``embedding_dim``, marks
            # auto-resolved writes so they are never later mistaken for an
            # operator pin, and records any space-change warning. A config/static
            # pin is honoured verbatim (never downgraded).
            try:
                await self.resolve_route_embedding_model(
                    provider,
                    corpus_profile=corpus_profile,
                    deployment_dim=deployment_dim,
                )
            except Exception as exc:  # pragma: no cover - never break init
                logger.debug(
                    "embedding capability resolve skipped for %s: %s",
                    route_name,
                    exc,
                )

    def _note_embedding_space_change(
        self,
        route_name: str,
        chosen: "EmbeddingModelInfo",
        corpus_profile: Optional[Dict[str, Any]],
    ) -> None:
        """Record + loudly log an auto-default that changes the corpus space (#2366).

        When a non-empty corpus exists and the freshly auto-resolved model does
        NOT match its dominant profile, new memories will land in a different
        embedding space than the existing ones. That is a real (if contained)
        recall split, so we log a warning and stash a structured record that the
        settings GET surfaces (the UI's mismatch banner renders it).

        The warning is re-evaluated on EVERY resolve, so a route that later
        resolves back onto the corpus space (a re-embed changed the dominant
        profile, or a pin was set) has its stale warning CLEARED rather than
        surfaced forever — the round-4 #2372 "stale readout" was exactly a
        warning that outlived the condition that produced it.
        """
        from .embedding_discovery import normalize_embedding_model_id

        dominant_model = (corpus_profile or {}).get("model")
        no_space_change = (
            not corpus_profile
            or not dominant_model
            or normalize_embedding_model_id(chosen.id)
            == normalize_embedding_model_id(dominant_model)
        )
        if no_space_change:
            # Coherent now — drop any warning a prior (mismatched) resolve left
            # so the settings GET stops surfacing a banner that no longer holds.
            self._clear_embedding_space_change_warning(route_name)
            return
        warning = {
            "route": route_name,
            "chosen_model": chosen.id,
            "corpus_model": dominant_model,
            "corpus_dim": corpus_profile.get("dim"),
            "corpus_space_id": corpus_profile.get("space_id"),
            "corpus_row_count": corpus_profile.get("row_count"),
        }
        logger.warning(
            "Auto-resolved embedding model for %s (%s) changes the embedding "
            "space away from the corpus's dominant profile (%s @%s, %s rows). "
            "New memories will not be comparable to existing ones until a "
            "re-embed — set an explicit embedding model or re-embed to keep one "
            "space.",
            route_name,
            chosen.id,
            dominant_model,
            corpus_profile.get("dim"),
            corpus_profile.get("row_count"),
        )
        warnings = getattr(self, "_embedding_space_change_warnings", None)
        if not isinstance(warnings, dict):
            warnings = {}
            self._embedding_space_change_warnings = warnings
        warnings[route_name] = warning

    def _clear_embedding_space_change_warning(self, route_name: Optional[str]) -> None:
        """Drop a stale space-change warning for *route_name* if one is recorded."""
        warnings = getattr(self, "_embedding_space_change_warnings", None)
        if isinstance(warnings, dict) and route_name in warnings:
            warnings.pop(route_name, None)

    async def _get_corpus_embedding_profile(self) -> Optional[Dict[str, Any]]:
        """Return the DB's dominant existing embedding profile, or ``None`` (#2366).

        Delegates to a provider callback wired by the agent
        (:meth:`LLMService.set_corpus_embedding_profile_provider`) so the LLM
        service stays storage-agnostic. Best-effort: any failure (no callback,
        empty corpus, unreadable table) yields ``None`` and the caller falls
        back to hint/catalog order.
        """
        callback = getattr(self, "_corpus_embedding_profile_provider", None)
        if callback is None:
            return None
        try:
            import inspect

            result = callback()
            if inspect.isawaitable(result):
                result = await result
            return result or None
        except Exception as exc:  # pragma: no cover - never break reconcile
            logger.debug("corpus embedding profile lookup failed: %s", exc)
            return None

    async def resolve_route_embedding_model(
        self,
        provider: dict,
        *,
        corpus_profile: Optional[Dict[str, Any]] = None,
        deployment_dim: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[int]]:
        """SINGLE source of truth for a route's ``(embedding_model, dim)`` (#2372).

        Route → model → dim, honoring #2366's documented order (explicit pin →
        corpus-dominant match by normalized id → deployment-dim match →
        ``selection_hints`` → first discovered) and PERSISTING the resolution
        into ``provider["capabilities"]`` so every reader agrees. The settings
        GET, the route-model echo, and the reindex target resolver all funnel
        through this instead of each re-deriving a model a different way — the
        #2372 incoherence was exactly that divergence (a cleared pin surfaced as
        ``None`` on the GET, as another route's slug on the echo, and as a third
        stale profile on reindex).

        Precedence:

        - An explicit operator pin (runtime override or config) surfaces as the
          ``is_pinned`` discovered candidate and is honoured verbatim, keeping
          the pin's own dimension.
        - A non-pinned route is (re)resolved on every call, so a stale
          capability left behind by a prior route/pin state is corrected rather
          than served.

        Returns ``(None, None)`` when discovery finds NO embedding model for the
        route (a truthful "off"), OR when the resolved model has no concrete dim
        from any branch (#2376) — an embedding-capable state without a dim is
        invalid by construction, so we fail closed rather than persist
        ``embedding_model`` without ``embedding_dim``. Otherwise writes
        ``supports_embeddings`` / ``embedding_model`` / ``embedding_dim`` into the
        route's capabilities together, as a side effect, so the sync readers
        (``get_embedding_settings``, ``ProviderEmbeddingService.describe``)
        observe the same answer and ``embedding_model`` NEVER stands without a
        ``embedding_dim``.
        """
        if not isinstance(provider, dict):
            return None, None
        caps = provider.get("capabilities")
        if not isinstance(caps, dict):
            caps = {}
            provider["capabilities"] = caps

        # The corpus/deployment continuity signals are route-independent — a
        # bulk caller (``reconcile_embedding_capabilities``) computes them ONCE
        # and passes them in so startup doesn't issue one corpus DB lookup per
        # route (#2372: that per-provider fan-out pushed lifespan startup over
        # the 5s test timeout). Single-route callers omit them and we fetch.
        if corpus_profile is None:
            corpus_profile = await self._get_corpus_embedding_profile()
        if deployment_dim is None:
            deployment_dim = _resolve_deployment_embedding_dim()
        chosen, resolved_dim = await self.resolve_default_embedding_model(
            provider,
            corpus_profile=corpus_profile,
            deployment_dim=deployment_dim,
        )
        if chosen is None:
            # Nothing discovered for this route — never fabricate a model. Report
            # whatever a static config pin already established (may be None).
            return caps.get("embedding_model"), caps.get("embedding_dim")

        dim = self._resolve_route_embedding_dim(caps, chosen, resolved_dim)
        if dim is None:
            # #2376 — an embedding-capable state with no concrete dim is invalid
            # BY CONSTRUCTION: the adapter would embed at the provider's native
            # width, the column guard would then refuse the write, and the read
            # path would build a kNN spec at the wrong width. Discovery supplied
            # no dim for this model (``native_dim`` unknown, ``dim_options``
            # empty — the ollama/OpenRouter case in the issue), and neither the
            # corpus nor the deployment pinned one. We CANNOT truthfully
            # advertise embeddings for this route, so fail closed: leave the
            # capabilities untouched (never persist ``embedding_model`` without
            # ``embedding_dim``) and report whatever prior state stood.
            logger.debug(
                "embedding resolve for %s skipped: model %s has no known dim "
                "(corpus/deployment/native all unset) — not advertising "
                "embeddings without a dim (#2376)",
                provider.get("name"),
                chosen.id,
            )
            return caps.get("embedding_model"), caps.get("embedding_dim")
        caps["supports_embeddings"] = True
        caps["embedding_model"] = chosen.id
        caps["embedding_dim"] = int(dim)
        if chosen.is_pinned:
            # Operator intent — not an auto default. Drop any stale auto marker so
            # the pin is honoured verbatim on subsequent discovery, and clear any
            # space-change warning a prior auto-resolve left (a deliberate pin is
            # not an accidental space split) so the settings GET stops surfacing
            # a banner that no longer holds (#2372 round-4 stale readout).
            caps.pop("embedding_model_auto_resolved", None)
            self._clear_embedding_space_change_warning(provider.get("name"))
        else:
            # Mark the write as auto-resolved so a later discovery (e.g. after the
            # reindex path clears the cache) does NOT mistake it for an operator
            # pin (#2372) — it must stay re-resolvable through the corpus/deployment
            # fallback order.
            caps["embedding_model_auto_resolved"] = True
            # A non-pin default that moves off the corpus space is a real recall
            # split — record it loudly (the settings GET surfaces the banner).
            self._note_embedding_space_change(
                provider.get("name"), chosen, corpus_profile
            )
        return caps.get("embedding_model"), caps.get("embedding_dim")

    @staticmethod
    def _resolve_route_embedding_dim(
        caps: Dict[str, Any],
        chosen: "EmbeddingModelInfo",
        resolved_dim: Optional[int],
    ) -> Optional[int]:
        """Pick the embedding dim to persist for a resolved model (#2372/#2376).

        An explicit pin's own dim is authoritative — it was written into
        ``capabilities`` at pin time, so keep it. Otherwise the resolver already
        chose the dim the model was matched on (the corpus's dominant dim for a
        corpus match, the deployment dim for a dim-compat match, or the model's
        native dim for the hint/catalog fallback), so honour it — falling back to
        the model's native dim only if the branch left it unset. Attaching that
        dim is what stops an auto-resolved corpus-matched model from embedding at
        the provider's native width and tripping the column guard (#2376).
        """
        if chosen.is_pinned and caps.get("embedding_dim") is not None:
            return caps.get("embedding_dim")
        if resolved_dim is not None:
            return int(resolved_dim)
        return chosen.native_dim

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

        ``OpenRouterAdapter`` is excluded from the generic OpenAI-compatible
        ``base_url`` branches even though it subclasses ``OpenAIAdapter`` and
        declares a ``base_url``. Its own ``list_models`` is authoritative and
        authenticates with the adapter's ``self.api_key`` — which
        ``finalize_providers()`` sets to the minted bootstrap child key when a
        route registered management-key-only (no static ``OPENROUTER_API_KEY``).
        The generic ``_discover_openai_compatible_remote`` path instead
        re-resolves the key from ``os.environ["OPENROUTER_API_KEY"]``, which is
        unset in that mode, so it silently returned ``[]`` and chat ``auto``
        never resolved — while the embedding path (``list_embedding_models``,
        also on ``self.api_key``) discovered fine (#2436).
        """
        from .openai_adapter import OpenAIAdapter
        from .openrouter_adapter import OpenRouterAdapter

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
        if isinstance(adapter, OpenAIAdapter) and not isinstance(
            adapter, OpenRouterAdapter
        ):
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
