"""
Model Discovery Service

Discovers available models from all LLM providers using their APIs.
Enriches results with manual overrides from model_catalog.toml.
Provides in-memory caching and disk-based cache for fast startup.
"""
import asyncio
import contextlib
import dataclasses
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Set, Tuple

from kestrel_sdk.llm import ModelInfo as SDKModelInfo

from kestrel_sovereign.execution_custody import (
    await_execution_work,
    await_execution_work_group,
    execution_work_operation,
    is_execution_control_error,
    execution_cleanup_debts,
    owned_native_http_client,
    require_execution_work,
)
from kestrel_sovereign.turn_scope import capture_turn_scope

from .discovery_deadline import DiscoveryStepTimeout, discovery_route_deadline_seconds
from .model_cache import (
    DiscoveryStatus,
    ModelCatalogSnapshot,
    SharedModelCache,
    VendorDiscoveryOutcome,
    get_shared_model_cache,
)
from .model_catalog import get_catalog_service
from .model_metadata import ModelCategory, ModelInfo

if TYPE_CHECKING:
    from .embedding_discovery import EmbeddingModelInfo

logger = logging.getLogger(__name__)

@dataclasses.dataclass(eq=False)
class _DiscoveryPass:
    """The outcomes one discovery pass has observed so far (#3197).

    A pass's vendors run as gathered child tasks; each copies this context, so
    every one writes into the same pass-owned map. A task created during the
    pass can also outlive it, still holding that context: once the pass ends
    it is closed, and such a task's later discovery is outside any pass rather
    than recording into a map nothing will publish (#3646).
    """

    outcomes: Dict[str, VendorDiscoveryOutcome]
    #: False once the pass's scope has ended.
    open: bool = True


_PASS: ContextVar[Optional[_DiscoveryPass]] = ContextVar(
    "kestrel_model_discovery_pass", default=None
)


def _pass_outcomes() -> Optional[Dict[str, VendorDiscoveryOutcome]]:
    """The outcome map of the discovery pass running in this context, if any."""
    scope = _PASS.get()
    return scope.outcomes if scope is not None and scope.open else None


@contextlib.contextmanager
def collect_discovery_outcomes(
    outcomes: Optional[Dict[str, VendorDiscoveryOutcome]] = None,
) -> Iterator[Dict[str, VendorDiscoveryOutcome]]:
    """Scope one discovery pass: yield the map its outcomes are recorded into.

    The outcomes reach the shared cache only when the pass publishes them with
    the catalog they produced, so no reader sees an outcome before its catalog.
    A pass that raises before publishing publishes neither.

    Pass ``outcomes``, the map an earlier part of the same pass yielded, to
    record into it again: a pass's capability reconciliation runs after its
    catalog requests, and its overruns are the same pass's outcomes (#3646).
    """
    scope = _DiscoveryPass({} if outcomes is None else outcomes)
    token = _PASS.set(scope)
    try:
        yield scope.outcomes
    finally:
        scope.open = False
        _PASS.reset(token)


@dataclasses.dataclass(eq=False)
class _DiscoveryCall:
    """What one discovery call has learned about its routes so far (#3646).

    Gathered child tasks copy the context and so share it. So does any task
    created during the call, which may outlive it: once the call ends the
    scope is closed, and such a task's later discovery starts a call of its
    own rather than inheriting stale silence and a stale sweep.
    """

    #: False once the call has ended.
    open: bool = True
    #: Routes that overran a discovery step earlier in this call. The call's
    #: later steps skip them, so a route that never answers costs the call one
    #: deadline, not one per step.
    silent: Set[str] = dataclasses.field(default_factory=set)
    #: An incomplete embedding sweep's models, by ``id()`` of the service that
    #: ran or joined it. An incomplete sweep is not cached, so without this
    #: every later embedding lookup in the call (one per route a reconcile
    #: resolves) would sweep again and ask the silent route again.
    embedding_sweeps: Dict[int, List["EmbeddingModelInfo"]] = dataclasses.field(
        default_factory=dict
    )


_DISCOVERY_CALL: ContextVar[Optional[_DiscoveryCall]] = ContextVar(
    "kestrel_model_discovery_call", default=None
)


@contextlib.contextmanager
def discovery_call() -> Iterator[None]:
    """Scope one discovery call (#3646).

    Within it a route that overran one discovery step is not asked again, and
    an incomplete embedding sweep is reused rather than repeated. A call
    nested in another (a reconcile inside :meth:`discover_all_models`) is part
    of it. The next call asks every route again.
    """
    if _current_discovery_call() is not None:
        yield
        return
    call = _DiscoveryCall()
    token = _DISCOVERY_CALL.set(call)
    try:
        yield
    finally:
        call.open = False
        _DISCOVERY_CALL.reset(token)


def _current_discovery_call() -> Optional[_DiscoveryCall]:
    """The discovery call running in this context, if it has not ended."""
    call = _DISCOVERY_CALL.get()
    return call if call is not None and call.open else None


def _silent_routes() -> Optional[Set[str]]:
    call = _current_discovery_call()
    return None if call is None else call.silent


#: Every field the framework ``ModelInfo`` constructor accepts (#3270).
_FRAMEWORK_MODEL_FIELDS = tuple(
    field.name for field in dataclasses.fields(ModelInfo) if field.init
)


def _as_framework_models(vendor: str, models: Any) -> List[ModelInfo]:
    """Return an adapter's ``list_models`` result as framework ``ModelInfo`` (#3270).

    The catalog is a list of the framework ``ModelInfo``, which extends the SDK
    dataclass with ``underlying_provider`` and ``output_limit``. External LLM
    plugins depend only on the SDK and build its base ``ModelInfo``, so their
    records used to enter the catalog without those fields: a fresh discovery
    returned two types, while the same catalog reloaded from the disk cache
    returned one. Each SDK record is lifted into the framework type with every
    field it carries; the fields it lacks take their defaults.

    An item that is not a ``ModelInfo`` at all breaks the adapter contract. It
    is dropped with a warning naming the vendor, never passed through. A result
    that is not a list means the call itself produced no catalog, so it raises
    for the caller to record as that vendor's discovery failure.
    """
    if not isinstance(models, (list, tuple)):
        raise TypeError(
            f"list_models returned {type(models).__name__}, not a list of ModelInfo"
        )
    lifted: List[ModelInfo] = []
    rejected: List[str] = []
    for model in models:
        if isinstance(model, ModelInfo):
            lifted.append(model)
        elif isinstance(model, SDKModelInfo):
            lifted.append(ModelInfo(**{
                name: getattr(model, name)
                for name in _FRAMEWORK_MODEL_FIELDS
                if hasattr(model, name)
            }))
        else:
            rejected.append(type(model).__name__)
    if rejected:
        logger.warning(
            "%s: model discovery dropped %d item(s) that are not ModelInfo: %s",
            vendor, len(rejected), ", ".join(sorted(set(rejected))),
        )
    return lifted


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
    except Exception as execution_error:  # pragma: no cover - defensive
        if is_execution_control_error(execution_error) or execution_cleanup_debts(execution_error):
            raise
        return None


# The capabilities a route's embedding profile is built from: whether the
# route embeds at all, and with which model at which width. These are what
# resolution decides (#2372); the auto-resolved marker it also writes changes
# no profile.
_EMBEDDING_PROFILE_CAPABILITIES = (
    "supports_embeddings",
    "embedding_model",
    "embedding_dim",
)


def _embedding_profile_capabilities(provider: dict) -> Tuple[Any, ...]:
    """The values of a route's ``_EMBEDDING_PROFILE_CAPABILITIES`` (#3606)."""
    caps = provider.get("capabilities")
    if not isinstance(caps, dict):
        caps = {}
    return tuple(caps.get(key) for key in _EMBEDDING_PROFILE_CAPABILITIES)


class ModelDiscoveryMixin:
    """
    Mixin class providing model discovery methods for LLMService.

    Uses adapter.list_models() for each provider to get available models,
    then enriches them with catalog data (featured, hidden, display names).
    """

    def _note_discovery_outcome(
        self, vendor: str, error: Optional[BaseException]
    ) -> None:
        """Record that ``vendor``'s catalog request completed or raised.

        Deliberately records ONLY what a request observed. Earlier revisions
        also tried to infer failure from the RETURNED VALUE — an empty list, or
        a catalog that looked reduced — and that cannot work: every adapter
        handles its own errors differently. ``AnthropicAdapter`` swallows a 401
        and returns ``[]``; ``VertexAIAdapter`` swallows and returns a
        non-empty STALE disk catalog; the OpenAI-compatible helpers raise. Five
        review rounds each found one more convention, because the return value
        does not carry the fact being inferred.

        So this records the one thing it actually observes. A vendor that
        swallows its own error is invisible here, and that is stated rather
        than papered over: fixing THAT needs an explicit outcome contract from
        the adapters, which is a larger change than this one.

        The outcome belongs to the discovery pass running in this context
        (:func:`collect_discovery_outcomes`), which publishes it with the
        catalog it produced (#3197). A cache hit's reconciliation collects its
        own, and its failures amend the snapshot the hit served (#3646).
        Outside a pass, as in local-only route resolution, no catalog is
        published, so this records nothing; a discovery step that overruns
        its deadline there amends the current catalog instead
        (:meth:`_record_step_overrun`).

        Nothing gates routing on this. It drives the ``model_discovery`` health
        check, so a dead credential is visible instead of living only in a log
        line — which is the operator-facing gap #3190 was filed for.
        """
        outcomes = _pass_outcomes()
        if outcomes is None:
            logger.debug("%s: discovery outcome outside a discovery pass", vendor)
            return
        if error is None:
            outcomes[vendor] = VendorDiscoveryOutcome.ok()
            return
        previous = outcomes.get(vendor)
        if previous is not None and previous.status is DiscoveryStatus.FAILED:
            # A pass can see a vendor fail more than once: its catalog request,
            # then a reconciliation step that overran its deadline (#3646). The
            # first failure stays, since the later ones usually follow from it.
            logger.debug("%s: further discovery failure this pass: %s", vendor, error)
            return
        outcomes[vendor] = VendorDiscoveryOutcome.failed(
            f"{type(error).__name__}: {error}"[:300]
        )

    def _discovery_route_deadline_seconds(self) -> float:
        """Seconds each route gets for each discovery step (#3646).

        ``[llm] discovery_route_deadline_seconds``; ``LLMService`` validates
        it at construction, so a malformed value never reaches discovery.
        """
        return discovery_route_deadline_seconds(getattr(self, "config", None))

    @contextlib.asynccontextmanager
    async def _discovery_deadline(
        self, vendor: str, route: str, step: str
    ) -> AsyncIterator[None]:
        """Run one route's discovery ``step`` under the per-route deadline (#3646).

        A route that accepts the connection and never answers used to stall
        the whole discovery pass, and with it every caller. An overrun now
        cancels the step, is logged at WARNING naming the vendor, route and
        step, and is recorded as the vendor's discovery failure
        (:meth:`_record_step_overrun`). It then raises
        :class:`DiscoveryStepTimeout`, so the caller leaves that route out of
        what it returns rather than waiting on it.

        A ``TimeoutError`` the step raises itself (the Ollama adapter bounds
        its listing calls by the same value) is the same failure.

        The deadline can only cancel a step that awaits. A listing that blocks
        the event loop (a synchronous SDK call) cannot be cut off, and stalls
        every other route's step and deadline with it, so adapter listings
        must use asynchronous clients (the Google and Vertex listings did not
        before #3646).

        Within one discovery call (:func:`discovery_call`), a route that
        overran one step is not asked again: its later steps raise a skipped
        :class:`DiscoveryStepTimeout` at once, without contacting it.
        """
        seconds = self._discovery_route_deadline_seconds()
        silent = _silent_routes()
        if silent is not None and route in silent:
            skipped = DiscoveryStepTimeout(
                vendor=vendor, route=route, step=step, seconds=seconds, skipped=True
            )
            logger.debug("%s: %s", vendor, skipped)
            raise skipped
        # Outside a discovery pass an overrun amends the catalog that was
        # current when the step began, so capture it now.
        current = (
            get_shared_model_cache().snapshot()
            if _pass_outcomes() is None
            else None
        )
        try:
            async with asyncio.timeout(seconds) as scope:
                yield
        except TimeoutError as exc:
            if is_execution_control_error(exc) or execution_cleanup_debts(exc):
                raise
            timeout = DiscoveryStepTimeout(
                vendor=vendor,
                route=route,
                step=step,
                seconds=seconds,
                cause=None if scope.expired() else exc,
            )
            logger.warning(
                "%s: %s; %s", vendor, timeout,
                self._record_step_overrun(timeout, current),
            )
            raise timeout from exc

    def _record_step_overrun(
        self,
        overrun: DiscoveryStepTimeout,
        current: Optional[ModelCatalogSnapshot],
    ) -> str:
        """Record ``overrun`` as its vendor's discovery failure (#3646).

        In a discovery pass (a refresh, or a cache hit's reconciliation) it is
        the pass's outcome for the vendor. Outside one (local-only route
        resolution, or an embedding or decision reconcile called on its own)
        nothing will publish it, so it amends ``current``, the catalog that was
        current when the step began, as a cache hit's failures do. Either way
        the ``model_discovery`` health check reports it. The route also joins
        the silent routes of the discovery call running in this context.

        Returns how it was recorded, for the overrun's log line.
        """
        silent = _silent_routes()
        if silent is not None:
            silent.add(overrun.route)
        if _pass_outcomes() is not None:
            self._note_discovery_outcome(overrun.vendor, overrun)
            return "recorded as that vendor's discovery failure"
        if current is None:
            return "no model catalog has been published, so not recorded"
        with collect_discovery_outcomes() as outcomes:
            self._note_discovery_outcome(overrun.vendor, overrun)
        if not self._amend_catalog_failures(get_shared_model_cache(), current, outcomes):
            return "the model catalog was replaced while it ran, so not recorded"
        return "recorded as that vendor's discovery failure"

    def _absorb_shared_overruns(self, overruns: List[DiscoveryStepTimeout]) -> None:
        """Take on the overruns of a sweep another caller started (#3646).

        A caller that joins a coalesced embedding sweep receives what its
        routes returned but ran none of its steps. The sweep's owner recorded
        each overrun (into its own pass, or against the current catalog). This
        caller's discovery pass, if any, records them too, so the catalog it
        publishes does not report a silent vendor healthy, and its discovery
        call does not ask those routes again.
        """
        silent = _silent_routes()
        for overrun in overruns:
            if silent is not None:
                silent.add(overrun.route)
            self._note_discovery_outcome(overrun.vendor, overrun)

    async def _discover_vendor_catalog(
        self, vendor: str, route: dict
    ) -> List[ModelInfo]:
        """:meth:`_discover_for_vendor_route` under the route's deadline (#3646).

        A route whose catalog request overruns contributes no models, and the
        vendor's discovery is recorded as failed.
        """
        try:
            async with self._discovery_deadline(
                vendor, route.get("name") or vendor, "chat catalog"
            ):
                return await self._discover_for_vendor_route(vendor, route)
        except DiscoveryStepTimeout as error:
            if is_execution_control_error(error) or execution_cleanup_debts(error):
                raise
            return []

    @execution_work_operation
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

        Each route's discovery steps run under the per-route deadline
        (``[llm] discovery_route_deadline_seconds``), and a route that overruns
        one is not asked again in this call (#3646). A route that never answers
        therefore delays this call by one deadline and is recorded as its
        vendor's discovery failure, rather than stalling it.
        """
        with discovery_call():
            return await self._discover_all_models(
                use_cache=use_cache,
                featured_only=featured_only,
                category=category,
                providers=providers,
                stale_while_revalidate=stale_while_revalidate,
            )

    async def _discover_all_models(
        self,
        *,
        use_cache: bool,
        featured_only: bool,
        category: Optional[ModelCategory],
        providers: Optional[List[str]],
        stale_while_revalidate: bool,
    ) -> List[ModelInfo]:
        """:meth:`discover_all_models` within its discovery-call scope."""
        # Check shared process-wide cache
        shared_cache = get_shared_model_cache()
        # use_cache selects catalog reuse, not permission to bypass an original
        # shared refresh's unresolved control or enter replacement discovery.
        shared_cache.snapshot()
        if use_cache:
            async def _return_cached(snapshot: ModelCatalogSnapshot) -> List[ModelInfo]:
                logger.debug("Using shared model cache")
                cached_models = list(snapshot.models)
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
                #
                # Reconciliation is still discovery: a route that overruns its
                # deadline here is that vendor's failure, recorded against the
                # catalog this hit served (#3646).
                with collect_discovery_outcomes() as reconcile_outcomes:
                    await self.reconcile_embedding_capabilities(use_cache=True)
                    # Decision models (#3424): reuse cached decision state;
                    # only never-discovered routes are contacted on a cache hit.
                    await self.reconcile_decision_capabilities(use_cache=True)
                self._amend_catalog_failures(
                    shared_cache, snapshot, reconcile_outcomes
                )
                # Reconciliation yields to the original shared refresh. A
                # healthy earlier snapshot cannot outlive its newly known
                # control, even when there are no availability amendments.
                shared_cache.snapshot()
                return self._filter_models(
                    cached_models,
                    featured_only=featured_only,
                    category=category,
                    providers=providers,
                )

            cached = shared_cache.fresh_snapshot()
            if cached is not None:
                return await _return_cached(cached)

            if (
                not stale_while_revalidate
                and await await_execution_work(self, lambda: shared_cache.wait_for_refresh())
            ):
                cached = shared_cache.fresh_snapshot()
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
        discovery_vendors: List[str] = []
        discovery_tasks = []
        # Opened before any vendor's discovery is scheduled, so every one runs
        # in this pass's context however its task is created.
        with collect_discovery_outcomes() as discovery_outcomes:
            for vendor, route in self._select_discovery_routes():
                discovery_vendors.append(vendor)
                discovery_tasks.append(
                    lambda vendor=vendor, route=route: self._discover_vendor_catalog(vendor, route)
                )

            results = await await_execution_work_group(self, discovery_tasks, return_exceptions=True)

            # Zip vendors back onto results (#3190): ``gather`` returns a bare
            # list, so the previous loop could neither name the failing vendor
            # in its warning nor record the outcome against it.
            for vendor, result in zip(discovery_vendors, results):
                if isinstance(result, BaseException):
                    logger.warning("%s: vendor discovery failed: %s", vendor, result)
                    self._note_discovery_outcome(vendor, result)
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
        #
        # Reconciliation belongs to this pass: a route whose embedding or
        # decision listing overruns its deadline is recorded as that vendor's
        # discovery failure, published with this catalog (#3646).
        with collect_discovery_outcomes(discovery_outcomes):
            await self.reconcile_embedding_capabilities(use_cache=True)
            # Decision models (#3424) on the same non-local warm-up path; a
            # cache miss refreshes every decision route and re-runs pin
            # canaries.
            await self.reconcile_decision_capabilities(use_cache=False)

        # Every vendor the pass covered, with the outcome its discovery recorded.
        # A vendor that recorded nothing issued no catalog request.
        pass_outcomes = {
            vendor: discovery_outcomes.get(
                vendor, VendorDiscoveryOutcome.not_attempted()
            )
            for vendor in discovery_vendors
        }

        # Cache results in shared memory cache and on disk. The catalog and the
        # outcomes that produced it are published as one snapshot (#3197).
        shared_cache.set(all_models, outcomes=pass_outcomes)
        catalog.write_cache(all_models)

        logger.info(f"Discovered {len(all_models)} models total")

        return self._filter_models(
            all_models,
            featured_only=featured_only,
            category=category,
            providers=providers
        )

    @staticmethod
    def _amend_catalog_failures(
        shared_cache: SharedModelCache,
        snapshot: ModelCatalogSnapshot,
        outcomes: Dict[str, VendorDiscoveryOutcome],
    ) -> bool:
        """Publish failures that discovery saw against ``snapshot``'s catalog (#3646).

        Discovery that publishes no catalog (a cache hit's reconciliation, or
        a discovery step outside any pass) amends the catalog it read instead:
        the ``model_discovery`` health check then reports a route that overran
        its deadline whichever service saw it, not only on a service that
        refreshed the catalog. Amendments from concurrent services merge
        (:meth:`SharedModelCache.record_failures`).

        Returns ``False`` when a later catalog replaced ``snapshot``'s, which
        leaves the failures unrecorded.
        """
        failures = {
            vendor: outcome
            for vendor, outcome in outcomes.items()
            if outcome.status is DiscoveryStatus.FAILED
        }
        if not failures or shared_cache.record_failures(
            snapshot.catalog_version, failures
        ):
            return True
        logger.info(
            "Discovery failures for %s were seen against model catalog %d, "
            "which a later catalog replaced; not recorded",
            sorted(failures), snapshot.catalog_version,
        )
        return False

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

    @execution_work_operation
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

    @execution_work_operation
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
        call = _current_discovery_call()
        partial = None if call is None else call.embedding_sweeps.get(id(self))
        if use_cache and cache is not None:
            models = list(cache)
        elif use_cache and partial is not None:
            # This call's own sweep, which a route overran (#3646).
            models = list(partial)
        else:
            models = await self._discover_embedding_models_coalesced()

        if vendor:
            models = [m for m in models if m.provider == vendor]
        if route:
            models = [m for m in models if m.route == route]
        return models

    @execution_work_operation
    async def _discover_embedding_models_coalesced(
        self,
    ) -> List["EmbeddingModelInfo"]:
        """Single-flight embedding discovery (#2433).

        The "twice" in the original repro is a concurrent refresh and invoke
        both probing the same routes. Coalesce concurrent discovery behind one
        in-flight future so overlapping callers share a single provider sweep
        (and a single INFO line per unsupported route) instead of each issuing
        its own.

        Every route's step runs under the per-route deadline, so the shared
        sweep always finishes and releases every caller waiting on it (#3646).
        A sweep in which a route overran is incomplete: its callers get what
        the other routes returned, but it is not cached, so a later call asks
        the silent route again instead of reading its absence as its catalog.
        Within one discovery call (:func:`discovery_call`) the incomplete
        sweep is reused instead. A caller that joined the sweep also takes on
        its overruns (:meth:`_absorb_shared_overruns`): the sweep ran in its
        starter's context, so they are recorded only there.
        """
        inflight = getattr(self, "_embedding_discovery_inflight", None)
        from kestrel_sovereign._async_ownership import (
            await_owned_task,
            create_owned_task,
            raise_owned_outcome,
        )
        if inflight is not None and not inflight.done():
            models, overruns = raise_owned_outcome(
                await await_owned_task(inflight), operation="shared embedding discovery"
            )
            self._absorb_shared_overruns(overruns)
        else:
            task = create_owned_task(self._discover_embedding_models_uncached())
            self._embedding_discovery_inflight = task
            try:
                models, overruns = raise_owned_outcome(
                    await await_owned_task(task), operation="shared embedding discovery"
                )
            finally:
                if getattr(self, "_embedding_discovery_inflight", None) is task:
                    self._embedding_discovery_inflight = None
            require_execution_work(self)
            if not overruns:
                self._embedding_discovery_cache = list(models)
        require_execution_work(self)
        call = _current_discovery_call()
        if overruns and call is not None:
            call.embedding_sweeps[id(self)] = list(models)
        return models

    @execution_work_operation
    async def _discover_embedding_models_uncached(
        self,
    ) -> Tuple[List["EmbeddingModelInfo"], List[DiscoveryStepTimeout]]:
        """One embedding-discovery sweep over every route.

        Returns the discovered models and every route step that overran its
        deadline or was skipped for an earlier overrun (#3646). The sweep is
        complete only when there are none.
        """
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
                lambda vendor=vendor, provider=provider, route_name=route_name: self._discover_embedding_for_route(
                    vendor, provider, chat_by_route.get(route_name)
                )
            )

        results = await await_execution_work_group(self, tasks, return_exceptions=True)
        discovered: List[EmbeddingModelInfo] = []
        overruns: List[DiscoveryStepTimeout] = []
        for result in results:
            if isinstance(result, DiscoveryStepTimeout):
                # Already logged and recorded by the step's deadline.
                overruns.append(result)
            elif isinstance(result, Exception):
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

        return discovered, overruns

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

    @execution_work_operation
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

        The adapter call runs under the per-route deadline (#3646). An overrun
        raises :class:`DiscoveryStepTimeout` rather than returning ``[]``, so
        the sweep knows the route's catalog is unknown, not empty.
        """
        adapter = route.get("adapter")
        client = route.get("client")
        route_name = route.get("name") or vendor
        if adapter is None or not hasattr(adapter, "list_embedding_models"):
            return []
        try:
            async with self._discovery_deadline(
                vendor, route_name, "embedding discovery"
            ):
                if getattr(adapter, "derives_embeddings_from_chat_listing", False):
                    models = await await_execution_work(self, lambda: adapter.list_embedding_models(
                        client, chat_models=chat_model_ids
                    ))
                else:
                    models = await await_execution_work(self, lambda: adapter.list_embedding_models(client))
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
        except NotImplementedError as execution_error:
            if is_execution_control_error(execution_error) or execution_cleanup_debts(execution_error):
                raise
            return []
        except DiscoveryStepTimeout:
            raise
        except Exception as e:
            if is_execution_control_error(e) or execution_cleanup_debts(e):
                raise
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

    @execution_work_operation
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

    async def reconcile_decision_capabilities(self, use_cache: bool = True) -> None:
        """Hook for the decisions modality (#3424), called beside embedding reconciliation.

        :class:`~kestrel_sovereign.llm.decision_service.DecisionServiceMixin`
        implements it and precedes this mixin in ``LLMService``'s MRO; a bare
        discovery mixin (catalog-only tooling and tests) has no decision state
        to reconcile.
        """

    def _embedding_profile_changed(self) -> None:
        """Hook: a resolution changed the profile a route embeds with (#3606).

        Called once :meth:`resolve_route_embedding_model` (or a
        :meth:`reconcile_embedding_capabilities` sweep) has written new
        embedding capabilities. ``LLMService`` overrides it to run the
        listener set with ``set_embedding_profile_change_callback``; a bare
        discovery mixin has no listener.
        """

    @execution_work_operation
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

        A sweep that changes any route's embedding profile reports it once,
        after every route is resolved, through :meth:`_embedding_profile_changed`
        (#3606).

        The sweep and every route's resolution are one discovery call
        (:func:`discovery_call`): a route that overran the sweep is not asked
        again while its siblings resolve (#3646).
        """
        with discovery_call():
            await self._reconcile_embedding_capabilities(use_cache=use_cache)

    async def _reconcile_embedding_capabilities(self, *, use_cache: bool) -> None:
        """:meth:`reconcile_embedding_capabilities` within its discovery call."""
        providers = getattr(self, "providers", None)
        if not isinstance(providers, list):
            return
        try:
            discovered = await self.discover_embedding_models(use_cache=use_cache)
        except Exception as e:  # pragma: no cover - never break init on discovery
            if is_execution_control_error(e) or execution_cleanup_debts(e):
                raise
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

        profile_changed = False
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
            before = _embedding_profile_capabilities(provider)
            try:
                await self._resolve_route_embedding_capabilities(
                    provider,
                    corpus_profile=corpus_profile,
                    deployment_dim=deployment_dim,
                )
            except Exception as exc:  # pragma: no cover - never break init
                if is_execution_control_error(exc) or execution_cleanup_debts(exc):
                    raise
                logger.debug(
                    "embedding capability resolve skipped for %s: %s",
                    route_name,
                    exc,
                )
            if _embedding_profile_capabilities(provider) != before:
                profile_changed = True
        if profile_changed:
            self._embedding_profile_changed()

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
            if is_execution_control_error(exc) or execution_cleanup_debts(exc):
                raise
            logger.debug("corpus embedding profile lookup failed: %s", exc)
            return None

    @execution_work_operation
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

        Storage embeds with whatever those capabilities say, so a resolution
        that changes them changes the agent's embedding profile without any
        setting being persisted: clearing a pin, for one, leaves the route to
        be resolved here afterwards. Such a change is reported through
        :meth:`_embedding_profile_changed` once it is written (#3606).
        """
        if not isinstance(provider, dict):
            return None, None
        before = _embedding_profile_capabilities(provider)
        resolved = await self._resolve_route_embedding_capabilities(
            provider,
            corpus_profile=corpus_profile,
            deployment_dim=deployment_dim,
        )
        if _embedding_profile_capabilities(provider) != before:
            self._embedding_profile_changed()
        return resolved

    async def _resolve_route_embedding_capabilities(
        self,
        provider: dict,
        *,
        corpus_profile: Optional[Dict[str, Any]],
        deployment_dim: Optional[int],
    ) -> Tuple[Optional[str], Optional[int]]:
        """:meth:`resolve_route_embedding_model`, without reporting a change.

        A :meth:`reconcile_embedding_capabilities` sweep resolves every route
        through this and reports its changes once, after the last route.
        """
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

    @execution_work_operation
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
                # Bounded like the full pass: a silent local daemon leaves its
                # route ``auto`` instead of stalling the turn (#3646).
                models.extend(await self._discover_vendor_catalog(vendor, route))
            except Exception as exc:  # pragma: no cover - defensive
                if is_execution_control_error(exc) or execution_cleanup_debts(exc):
                    raise
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
        except Exception as execution_error:  # pragma: no cover - import-safety
            if is_execution_control_error(execution_error) or execution_cleanup_debts(execution_error):
                raise
            return
        for provider in self.providers:
            adapter = provider.get("adapter")
            route_key = provider.get("name")
            if route_key and isinstance(adapter, CodexAdapter):
                yield route_key, adapter

    @execution_work_operation
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

    @execution_work_operation
    async def _collect_route_catalogs(self) -> dict:
        """Build and RETURN the route-scoped catalog map (no self mutation).

        Split out from :meth:`_build_route_catalogs` so the sync helper can
        drive it on a worker thread and assign the result on the owning thread
        (avoids cross-thread mutation of ``self._route_catalogs``).
        """
        catalogs: dict[str, list] = {}
        for route_key, adapter in self._route_specific_catalog_adapters():
            try:
                models = await await_execution_work(self, lambda: adapter.list_models())
            except NotImplementedError as execution_error:
                if is_execution_control_error(execution_error) or execution_cleanup_debts(execution_error):
                    raise
                continue
            except Exception as e:  # pragma: no cover - defensive
                if is_execution_control_error(e) or execution_cleanup_debts(e):
                    raise
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

        require_execution_work(self)
        if getattr(self, "_route_catalogs", None) is not None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to drive the async builder synchronously.
            try:
                asyncio.run(self._build_route_catalogs())
            except Exception as e:  # pragma: no cover - defensive
                if is_execution_control_error(e) or execution_cleanup_debts(e):
                    raise
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

            original_scope = capture_turn_scope(self)

            def _build_in_thread() -> dict:
                with original_scope.bind():
                    return asyncio.run(self._collect_route_catalogs())

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    catalogs = pool.submit(_build_in_thread).result()
                    require_execution_work(self)
                    self._route_catalogs = catalogs
            except Exception as e:  # pragma: no cover - defensive
                if is_execution_control_error(e) or execution_cleanup_debts(e):
                    raise
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
        whose adapter actually implements ``list_models``. Codex subscription
        routes cannot discover models, while Claude plan routes can use their
        OAuth-authenticated Anthropic client.
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
            # Prefer canonical API routes when present. OAuth-only Claude plan
            # installs still fall back to their plan route and discover there.
            non_sub = [r for r in routes if not isinstance(r.get("adapter"), (ClaudeMaxAdapter, CodexAdapter))]
            chosen.append((vendor, (non_sub or routes)[0]))
        return chosen

    @execution_work_operation
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
                return await self._record_discovery(
                    vendor, self._discover_local_openai_compatible(vendor, route_cfg)
                )
            if base_url:
                return await self._record_discovery(
                    vendor, self._discover_openai_compatible_remote(vendor, route_cfg)
                )

        return await self._safe_list_models(vendor, adapter, client)

    async def _record_discovery(self, vendor: str, coro) -> List[ModelInfo]:
        """Await an OpenAI-compatible discovery helper, recording its outcome.

        These two paths bypass ``_safe_list_models`` entirely, so before #3190
        they recorded nothing: an outage on ollama, llama_cpp, xAI, Groq or
        RunPod stayed absent from health, and their collapsed catalogs could
        still veto a pinned model. RunPod's ``/models`` was in fact 404ing on
        the production host while this fix claimed to surface exactly that.

        One recorder wrapping both call sites rather than an edit inside each
        helper: the helpers raise on a real discovery error and return ``[]``
        only for absent configuration, so failure and "not attempted" stay
        distinguishable at a single place.
        """
        try:
            models = await coro
        except TimeoutError:
            # The route's deadline records a timeout (#3646).
            raise
        except Exception as e:  # noqa: BLE001 - discovery must never break init
            if is_execution_control_error(e) or execution_cleanup_debts(e):
                raise
            logger.warning("%s: model discovery failed: %s", vendor, e)
            self._note_discovery_outcome(vendor, e)
            return []
        if models is None:
            # NOT ATTEMPTED. Record nothing: the pass publishes this vendor as
            # not attempted, which keeps a standing failure standing (#3190
            # r10 P2 — unsetting a key would otherwise turn health green having
            # fetched nothing) and does not invent one.
            logger.debug("%s: discovery not attempted", vendor)
            return []

        # A request completed. Clear any prior failure even if the catalog came
        # back EMPTY — a fresh local server with no models loaded is a
        # successful fetch, and leaving it degraded reports a stale auth error
        # until the provider happens to expose a model (#3190 r11 P2).
        #
        # Rounds 10 and 11 are the same finding from opposite sides, and both
        # were right: list length cannot distinguish "no request" from "a
        # request that found nothing". That is the conflation this entire
        # change is about, reached one final time in my own code. The fix is
        # the thing the ticket concluded was needed — an explicit outcome —
        # scoped to the two helpers where both ends are under my control:
        # ``None`` means not attempted, a list means fetched.
        self._note_discovery_outcome(vendor, None)
        return models

    @execution_work_operation
    async def _safe_list_models(self, vendor: str, adapter, client) -> List[ModelInfo]:
        """Call adapter.list_models(client) with full error tolerance.

        ``client`` is the route's framework-initialized provider-native
        client; the SDK 0.5.0 contract requires it be passed to
        discovery so authenticated /models endpoints reach the right
        endpoint for routes with custom ``base_url``.

        This is where any adapter's records, including an external plugin's,
        enter the catalog, so they are normalized here
        (:func:`_as_framework_models`).
        """
        try:
            if hasattr(adapter, 'list_models'):
                models = _as_framework_models(
                    vendor, await await_execution_work(self, lambda: adapter.list_models(client))
                )
                logger.debug("%s: discovered %d models", vendor, len(models))
                self._note_discovery_outcome(vendor, None)
                return models
        except NotImplementedError as execution_error:
            if is_execution_control_error(execution_error) or execution_cleanup_debts(execution_error):
                raise
            # Not a failure: the adapter simply publishes no catalog, so no
            # request was issued and the pass records the vendor as not
            # attempted. Recording a failure would report every such vendor's
            # discovery as broken.
            logger.debug("%s: adapter.list_models not implemented", vendor)
        except TimeoutError:
            # An adapter's own listing bound (Ollama's, #3646): the route's
            # deadline records it, the same as when the deadline fires first.
            raise
        except Exception as e:
            if is_execution_control_error(e) or execution_cleanup_debts(e):
                raise
            logger.warning("%s: model discovery failed: %s", vendor, e)
            self._note_discovery_outcome(vendor, e)
        return []

    @execution_work_operation
    async def _discover_local_openai_compatible(
        self, provider_name: str, provider_config: dict
    ) -> List[ModelInfo]:
        """Discover models from a local OpenAI-compatible server (llama.cpp, etc.)."""
        base_url = provider_config.get("base_url", "")
        if not base_url:
            return None  # NOT ATTEMPTED — see _record_discovery

        # Config-level context_limit override (most reliable for local servers)
        config_context_limit = provider_config.get("context_limit")

        import httpx
        models_url = f"{base_url.rstrip('/')}/models"
        try:
            async with owned_native_http_client(httpx.AsyncClient(timeout=5.0)) as client:
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
            if is_execution_control_error(e) or execution_cleanup_debts(e):
                raise
            # Raise rather than return [] (#3190): an empty list is also what a
            # server with no models returns, and the caller cannot tell those
            # apart. _record_discovery logs and converts this to [] after
            # recording the vendor as failed.
            raise RuntimeError(
                f"local model discovery failed ({models_url}): {e}"
            ) from e

    @execution_work_operation
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
            return None  # NOT ATTEMPTED — see _record_discovery

        # Resolve API key from config or convention
        api_key_env = provider_config.get("api_key_env", f"{provider_name.upper()}_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            logger.debug(f"{provider_name}: {api_key_env} not set, skipping discovery")
            return None  # NOT ATTEMPTED — see _record_discovery

        import httpx
        models_url = f"{base_url.rstrip('/')}/models"
        try:
            async with owned_native_http_client(httpx.AsyncClient(timeout=10.0)) as client:
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
            if is_execution_control_error(e) or execution_cleanup_debts(e):
                raise
            # See _discover_local_openai_compatible: failure must be
            # distinguishable from "serves nothing" (#3190).
            raise RuntimeError(
                f"remote model discovery failed ({models_url}): {e}"
            ) from e

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
                async with owned_native_http_client(httpx_module.AsyncClient(timeout=3.0)) as client:
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
            except Exception as execution_error:
                if is_execution_control_error(execution_error) or execution_cleanup_debts(execution_error):
                    raise
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
