"""Streaming response logic for LLM Service.

Extracted from service.py to reduce file size. These methods handle:
- Basic streaming responses with provider fallback
- Streaming with pre-built message arrays
- Streaming with tool call detection and assembly
- Unified streaming with remote GPU fallback

No-silent-fallback rule
-----------------------
When ``resolve_provider_routing`` narrows the candidate list by an explicit
mandate or ``model_override``, the streaming loop must NOT silently fall
through to a different provider on failure. The user selected a specific
backend; answering from a different one without saying so is lying. We
enforce this by failing loudly (``LLMStreamingError``) whenever the
provider list has exactly one entry — that covers every mandate-restricted
or override-restricted case. Multi-provider default chains still retry
through the list, but the fallback happens in server logs, not by
injecting a ``[Provider X unavailable, trying next...]`` note into the
chat stream where it corrupts the agent's response.
"""
import logging
import time
from typing import (
    Awaitable,
    Callable,
    Iterator,
    List,
    Dict,
    Any,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    Type,
    AsyncIterator,
)

from pydantic import BaseModel

from kestrel_sdk.llm import ProviderCapabilities, StructuredOutputMode, ToolCallStarted

from .adapter import LLMResponse, ThinkingDelta, messages_for
from .cancellation import CancelToken
from .codex_app_server import CodexAppServerTransportError
from .error_handling import LLMError
from .provider_registry import provider_cache_body

logger = logging.getLogger(__name__)

# v5 typed negotiation (#1983). Routing gates read the adapter's typed
# ``ProviderCapabilities`` plus its ``contract_features()`` opt-in set instead
# of matching on bare vendor names — composite route names (``"openai:api"``)
# never matched the old ``provider_name in [...]`` literals, so those gates
# were dead. The contract-feature opt-in keeps every in-tree adapter's
# behavior unchanged until it explicitly advertises the feature.
_FEATURE_STREAMING_STRUCTURED_OUTPUT = "streaming_structured_output"
_FEATURE_TOOL_STREAM_SYSTEM_PROMPT = "tool_stream_system_prompt"

# Structured-output modes that can be produced while streaming. ``TOOL_FORCED``
# (Anthropic) cannot — it assembles the object from a buffered tool call — and
# ``NONE``/``UNKNOWN`` carry no streamable guarantee.
_STREAMABLE_STRUCTURED_MODES = frozenset(
    {
        StructuredOutputMode.JSON_OBJECT,
        StructuredOutputMode.JSON_SCHEMA,
        StructuredOutputMode.SCHEMA_FORMAT,
        StructuredOutputMode.PROVIDER_NATIVE,
    }
)


def _route_capabilities(provider: Any) -> Tuple[ProviderCapabilities, frozenset]:
    """Return a route's typed capabilities and the adapter's opt-in feature set.

    Prefers the **route-scoped** capabilities carried on the provider dict — an
    entry-point factory may set these precisely on ``ProviderInfo`` and they can
    differ from the adapter's own ``provider_capabilities()`` (e.g. an
    OpenAI-compatible adapter reused across vendors). Falls back to the adapter
    when the route didn't carry them. ``contract_features()`` always comes from
    the adapter. Returns empty defaults when nothing can answer, so callers can
    use plain attribute access.
    """
    adapter = provider.get("adapter") if isinstance(provider, dict) else provider
    raw = provider.get("capabilities") if isinstance(provider, dict) else None
    caps: Optional[ProviderCapabilities] = None
    if isinstance(raw, ProviderCapabilities):
        caps = raw
    elif isinstance(raw, dict) and raw:
        try:
            caps = ProviderCapabilities.from_mapping(raw)
        except Exception:
            caps = None
    if caps is None and adapter is not None:
        try:
            caps = adapter.provider_capabilities()
        except Exception:
            caps = None
    if not isinstance(caps, ProviderCapabilities):
        caps = ProviderCapabilities()
    features: frozenset = frozenset()
    if adapter is not None:
        try:
            features = frozenset(adapter.contract_features() or ())
        except Exception:
            features = frozenset()
    return caps, features


def _route_supports_streaming_structured(provider: Any) -> bool:
    """Whether this route can stream while honoring a ``response_format``.

    Typed replacement for the dead ``provider_name in ["openai", "vertex_ai"]``
    literal. Reads route-scoped capabilities (honoring factory-set values on
    entry-point routes) and evaluates False for every in-tree adapter until one
    opts in via ``contract_features()`` — i.e. no behavior change yet (#1983).
    """
    caps, features = _route_capabilities(provider)
    if _FEATURE_STREAMING_STRUCTURED_OUTPUT not in features:
        return False
    if not caps.supports_streaming:
        return False
    return caps.structured_output_mode in _STREAMABLE_STRUCTURED_MODES


def _route_wants_tool_stream_system_prompt(provider: Any) -> bool:
    """Whether to forward ``system_prompt`` separately into tool streaming.

    Typed replacement for the dead ``provider_name == "anthropic"`` literal.
    Anthropic-family routes carry the system prompt as a top-level field rather
    than an inline message; the typed signal is ``supports_inline_system``.
    Reads route-scoped capabilities and is gated behind ``contract_features()``
    so no in-tree adapter changes behavior until it opts in (#1983).
    """
    caps, features = _route_capabilities(provider)
    if _FEATURE_TOOL_STREAM_SYSTEM_PROMPT not in features:
        return False
    return caps.supports_inline_system


class RoutingMeta(NamedTuple):
    """No-silent-fallback authorization metadata for a routing decision.

    Computed once by :meth:`StreamingMixin._compute_route_authorization` and
    carried on every :class:`RoutingResolution` so the fallback loops never
    re-derive it (the drift bug class that produced five rounds of edge cases).

    * ``explicit_selection`` — True iff a SINGLE concrete route was deliberately
      pinned (a route-qualified ``vendor:route`` override/mandate, or a
      vendor-only selection that resolved to exactly one route). When True the
      dispatch loops fail loudly on that route's error rather than fall through.
      It is NEVER ``len(providers) == 1``: a lone incidental credentialed route
      is a blind fallback target, not a deliberate selection.

    * ``authorized_vendors`` — the vendors the operator/user/mandate actually
      authorized for THIS call, computed with the SAME branch conditions
      :meth:`LLMService.resolve_provider_routing` uses (override-precedence,
      active-mandate match, fallback-only-when-unmatched, force_local_only).
      Routes whose vendor is outside this set are unconfigured incidental
      routes the blind-fallback guard must skip.
    """

    explicit_selection: bool
    authorized_vendors: Set[str]


class RoutingResolution:
    """Result of :meth:`LLMService.resolve_provider_routing`.

    The single source of truth for a routing decision. Unpacks as the historic
    ``(providers, target_model)`` 2-tuple so every existing call site keeps
    working unchanged, while also exposing :attr:`meta` — the no-silent-fallback
    authorization computed by the SAME resolution pass. The streaming/dispatch
    loops read ``resolution.meta`` instead of re-deriving authorization in a
    parallel ``resolve_routing_meta`` that could drift.
    """

    __slots__ = ("providers", "target_model", "meta")

    def __init__(
        self,
        providers: List[Dict[str, Any]],
        target_model: Optional[str],
        meta: RoutingMeta,
    ) -> None:
        self.providers = providers
        self.target_model = target_model
        self.meta = meta

    def __iter__(self) -> Iterator[Any]:
        # Back-compat: ``providers, target_model = resolve_provider_routing(...)``
        yield self.providers
        yield self.target_model

    def __getitem__(self, index: int) -> Any:
        return (self.providers, self.target_model)[index]

    def __len__(self) -> int:
        return 2


def _is_harness_owned_transport_error(exc: BaseException) -> bool:
    """True when ``exc`` is a *transport* failure from a harness that
    owns its own transport (timeouts, retries, websocket lifecycle)
    and therefore must NOT be treated as evidence the route is broken.

    Sovereign's only such harness today is the codex app-server bridge
    (``openai:plan``): a transient codex/ChatGPT-Plus stall raises
    ``CodexAppServerTransportError`` (idle timeout, RPC timeout,
    app-server connection closed) and is **not** a signal that openai
    is down. Rotating to a different provider on this error gives the
    user a wrong-model response without warning — the upstream
    antipattern openclaw fixed in commit ``3a64dc7623`` ("keep turn
    timeouts inside Codex").

    Narrowed to ``CodexAppServerTransportError`` specifically (not the
    supertype ``CodexAppServerError``) so caller-config / protocol /
    codex-reported-turn-failure errors retain their normal fallback
    semantics — those *should* let the chain try the next provider.
    """
    return isinstance(exc, CodexAppServerTransportError)


class LLMStreamingError(LLMError):
    """Raised when streaming operation fails.

    Carries the failing provider's composite name and the underlying error
    so callers (and ultimately the user) get a specific actionable message
    instead of ``"all providers failed"``.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        underlying: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.underlying = underlying


class StreamingMixin:
    """Mixin class providing streaming methods for LLMService.

    Expects the following attributes on the host class:
    - providers: List[Dict[str, Any]]
    - _backend: BackendType
    - _remote_client: Optional[AsyncOpenAI]
    - _remote_adapter: OpenAIAdapter
    - _remote_config: Optional[RemoteGPUConfig]
    - _last_remote_error: Optional[str]
    - _mandate_preference: Dict[str, Optional[str]]
    - _ensure_remote_active() -> None
    - _deactivate_remote_backend(reason: Optional[str]) -> None
    - resolve_provider_routing(...) -> RoutingResolution
    - _available_providers() -> List[Dict[str, Any]]
    - discover_all_models(use_cache: bool) -> Awaitable[...]
    """

    async def _ensure_models_discovered(self, *, force_local_only: bool = False) -> None:
        """Trigger model discovery once if any route is still seeded ``"auto"``.

        Routes start with ``model = "auto"`` in ``kestrel.toml`` and only
        resolve to a concrete id when the disk cache is populated
        (``_load_from_disk_cache`` at ``__init__``) or the model picker UI
        calls ``discover_all_models`` via ``/api/models``. On a fresh
        deployment neither has happened when the first chat arrives, so the
        cache that backs ``resolve_provider_default`` is empty: every route
        fails auto-resolution in ``_resolve_concrete_model`` and the fallback
        walk surfaces the *last* route's ``ModelNotAvailableForRoute`` (e.g.
        ``ollama:local``) as a hard error, even when a key-backed vendor is
        configured (#2069).

        Trigger discovery here, once, on demand. ``use_cache=True`` makes
        repeat calls a cache hit, and ``discover_all_models`` re-resolves the
        provider list's ``"auto"`` entries (``_resolve_auto_providers``), so
        the guard short-circuits on subsequent turns. Discovery failure is
        non-fatal — the route walk still runs and fails loudly per route.

        ``force_local_only``: privacy gate. ``discover_all_models`` enumerates
        *every* configured vendor (via ``_select_discovery_routes`` over
        ``self.providers``) and writes the merged result to the shared/disk
        cache — contacting cloud vendors and poisoning the cache for a
        local-only turn (ISOLATED/EPHEMERAL privacy session). So for a
        local-only turn we never call it; instead, if a LOCAL route is still
        ``"auto"``, we warm via :meth:`_resolve_local_auto_routes`, which scopes
        discovery to local routes only (no cloud contact, no cache write) so a
        cold-cache local route still resolves to a concrete model — covering
        both all-local and mixed cloud/local configs.
        """
        providers = self._available_providers() or []
        if force_local_only:
            if any(p.get("model") == "auto" and p.get("is_local") for p in providers):
                try:
                    await self._resolve_local_auto_routes()
                except Exception as exc:
                    logger.warning(
                        "Local-only model discovery failed (continuing with "
                        "provider['model'] as-is): %s", exc,
                    )
            return
        if any(p.get("model") == "auto" for p in providers):
            try:
                await self.discover_all_models(use_cache=True)
            except Exception as exc:
                logger.warning(
                    "Lazy model discovery failed (continuing with "
                    "provider['model'] as-is): %s", exc,
                )

    async def _resolve_routing_with_discovery(
        self,
        *,
        model_override: Optional[str] = None,
        force_local_only: bool = False,
    ) -> "RoutingResolution":
        """:meth:`resolve_provider_routing` preceded by a lazy discovery
        warm-up (#2069).

        The async generation entry points (``get_response`` and the streaming
        paths) call this instead of ``resolve_provider_routing`` directly so a
        cold-cache fresh boot resolves ``"auto"`` to a real model rather than
        hard-failing the route walk. Sync callers of ``resolve_provider_routing``
        (e.g. embedding resolution) keep their own no-warm-up behavior.
        """
        await self._ensure_models_discovered(force_local_only=force_local_only)
        return self.resolve_provider_routing(
            model_override=model_override,
            force_local_only=force_local_only,
        )

    def _configured_route_vendors(self) -> set:
        """Vendors the operator EXPLICITLY chose via ``route_priority``.

        ``ProviderRegistry.initialize_providers`` brings up every *credentialed*
        route, but lists the operator's ``route_priority`` entries FIRST and
        appends every other credentialed route afterward (merely because its
        auth env happens to be set). So ``self.providers`` for an operator who
        configured ``route_priority = ["openai:plan", "openai:api"]`` is
        ``[openai:plan, openai:api, <anthropic:* if ANTHROPIC_API_KEY set>, ...]``.

        The vendors named in ``route_priority`` are the operator's *chosen*
        vendors; any vendor that appears ONLY because its key is in the env is
        an auto-appended fallback the operator never asked for. Falling back to
        the latter silently is the blind-fallback bug.
        """
        config = getattr(self, "config", None) or {}
        try:
            priority = config.get("route_priority") or []
        except AttributeError:
            priority = []
        vendors = set()
        for key in priority:
            if isinstance(key, str) and key:
                vendors.add(key.split(":", 1)[0])
        return vendors

    def _skip_unconfigured_route(
        self, provider: dict, configured_vendors: Optional[set] = None
    ) -> bool:
        """True when the loop must NOT even *attempt* ``provider``.

        When the operator declared ``route_priority`` (or an explicit mandate
        with fallbacks), only the vendors in ``configured_vendors`` are
        legitimate fallback targets. ``ProviderRegistry`` appends every other
        *credentialed* route after the chosen ones (merely because its auth env
        is set), so the chain can contain unconfigured-vendor routes
        interleaved among — even *before* — the operator's own later routes.
        Attempting such a route would produce a blind cross-vendor answer the
        operator never asked for (``feedback_no_blind_fallbacks``).

        We therefore skip these candidates *before dispatch*: the decision is
        made on the route actually about to be tried, not on whether some
        later route in the tail happens to be configured. This closes the gap
        where an unconfigured vendor positioned ahead of a configured route
        would still be hit first.

        ``configured_vendors`` is the resolved authorized-vendor set from
        :meth:`_compute_route_authorization` — the vendors authorized for THIS
        call (``route_priority`` ∪ the explicitly selected route's vendor ∪ any
        operator-declared ``_mandate_fallbacks`` vendor that an active unmatched
        mandate actually engages). Callers pass it so a mandate fallback to a
        vendor outside ``route_priority`` is still attempted. When omitted we
        fall back to the static ``route_priority`` set.

        When the authorized set is empty we can't distinguish a chosen vendor
        from an incidental one, so we never skip — preserving the historical
        default-chain behavior.
        """
        if configured_vendors is None:
            configured_vendors = self._configured_route_vendors()
        if not configured_vendors:
            return False  # No explicit operator choice — keep legacy chain.
        return provider.get("vendor") not in configured_vendors

    def _configured_routes_exhausted(
        self,
        providers: list,
        failed_index: int,
        configured_vendors: Optional[set] = None,
    ) -> bool:
        """True when a *configured* route at ``failed_index`` has failed and no
        remaining route belongs to an operator-configured vendor.

        Falling through here would land only on unconfigured-vendor routes
        (which :meth:`_skip_unconfigured_route` skips), so the chain has no
        legitimate next candidate. The caller must raise the loud diagnostic
        naming the failed route + underlying error instead of degrading into a
        generic "all providers failed" — the operator's chosen routes really
        are spent.

        ``configured_vendors`` is the resolved authorized-vendor set from
        :meth:`_compute_route_authorization` (route_priority ∪ explicitly-selected
        vendor ∪ engaged mandate-fallback vendors); callers pass it so a mandate
        fallback to a vendor outside ``route_priority`` counts as a legitimate
        remaining candidate. When omitted we fall back to the static
        ``route_priority`` set.

        Returns False (don't raise yet) when:
          - no vendor is authorized (legacy default chain), or
          - the failed route belongs to an unconfigured vendor (we're already
            past the operator's chosen routes — which only happens for routes
            the loop chose to attempt; in practice such routes are skipped), or
          - at least one remaining route is from a configured vendor (a
            legitimate same-/configured-cross-vendor fallback exists).
        """
        if not providers:
            return False
        if configured_vendors is None:
            configured_vendors = self._configured_route_vendors()
        if not configured_vendors:
            return False  # No explicit operator choice — keep legacy chain.
        failed_vendor = providers[failed_index].get("vendor")
        if failed_vendor not in configured_vendors:
            return False  # Already past the operator's chosen vendors.
        remaining = providers[failed_index + 1:]
        if not remaining:
            return False  # Nothing left — normal "all providers failed".
        # A legitimate next candidate exists only if some remaining route is
        # from a configured vendor; unconfigured routes will be skipped.
        if any(p.get("vendor") in configured_vendors for p in remaining):
            return False
        # Every remaining route is an unconfigured vendor (the loop will skip
        # them all) — the operator's chosen routes are spent. Raise loudly.
        return True

    @staticmethod
    def _match_selector(providers: list, selector: str) -> list:
        """Vendor-or-composite-route match (mirrors
        ``LLMService._filter_providers_by_selector``). Kept local so the
        no-silent-fallback signals are computable on any ``StreamingMixin``
        host without depending on the full service surface.
        """
        if not selector:
            return []
        if ":" in selector:
            return [p for p in providers if p.get("name") == selector]
        return [p for p in providers if p.get("vendor") == selector]

    def _compute_route_authorization(
        self,
        *,
        model_override: Optional[str] = None,
        force_local_only: bool = False,
    ) -> RoutingMeta:
        """Single implementation of "what's authorized + is it explicit".

        This is the ONE place the no-silent-fallback signals are derived. Both
        :meth:`LLMService.resolve_provider_routing` (chat/streaming paths, which
        wraps the result in a :class:`RoutingResolution`) and
        :meth:`LLMService.generate_with_messages` (which resolves routing inline)
        call it, so there is exactly one copy of the branching logic. The prior
        ``resolve_routing_meta`` re-derived these signals in PARALLEL to
        ``resolve_provider_routing`` and the two drifted — that drift is the root
        cause of the five rounds of edge bugs.

        The branch conditions MIRROR ``resolve_provider_routing`` exactly:

        1. **Override precedence.** A concrete ``model_override`` (``vendor/...``
           or ``vendor:route/...``) wins. ``resolve_provider_routing`` takes the
           override branch and does NOT consult the persisted mandate, so when an
           override is present we do NOT fold the mandate vendor — nor the mandate
           fallbacks — into the authorized set (P2a). Authorizing a stale
           mandate's vendor alongside a bare override let
           ``_skip_unconfigured_route`` drop the very route serving the override.

        2. **Active-mandate match.** With no override, a persisted
           ``{vendor, model, route?}`` mandate authorizes its vendor and — when it
           narrows to exactly one route (route-qualified, or vendor-only matching
           a single route) — marks the selection explicit.

        3. **Fallback-only-when-unmatched.** ``_mandate_fallbacks`` vendors are
           authorized ONLY when an active mandate exists AND its preferred route
           did NOT match an available route — the exact condition under which
           ``resolve_provider_routing`` builds the fallback chain. Folding stale
           fallbacks in unconditionally (P2b) let a leftover fallback list wrongly
           authorize/drop default providers on requests that have no active
           mandate.

        4. **force_local_only.** A deliberate privacy constraint (#1492): every
           local route's vendor is authorized even when absent from
           ``route_priority``.

        Reads host attributes (``_available_providers``, ``_mandate_preference``,
        ``_mandate_fallbacks``) defensively so process-local hosts that omit them
        degrade to the ``route_priority``-only set.
        """
        # Mirror resolve_provider_routing's "auto" normalization so a bare
        # "auto" override is treated as no-override here too.
        if model_override == "auto":
            model_override = None

        authorized_vendors: Set[str] = set(self._configured_route_vendors())
        explicit_selection = False

        available_fn = getattr(self, "_available_providers", None)
        if callable(available_fn):
            try:
                available = available_fn()
            except Exception:
                available = list(getattr(self, "providers", None) or [])
        else:
            available = list(getattr(self, "providers", None) or [])

        # --- force_local_only (resolve order step 3, privacy pin) ---
        # The local routes it pins are authorized even when their vendor (e.g.
        # ``ollama``) is absent from ``route_priority`` — they are the
        # operator/privacy-layer's chosen targets, not blind incidental
        # fallbacks. Fold every local vendor in so the unconfigured-route skip
        # never strips the only privacy-safe route.
        if force_local_only:
            for p in available:
                if p.get("is_local") and p.get("vendor"):
                    authorized_vendors.add(p["vendor"])

        # A concrete vendor(:route) prefix on the override is what triggers
        # override-precedence in resolve_provider_routing.
        override_has_prefix = bool(
            model_override and ("/" in model_override or ":" in model_override)
        )

        # --- 1. model_override branch (override precedence) ---
        if override_has_prefix:
            # Vendor or vendor:route prefix. The left side before the first
            # "/" is the route selector; a bare ":" with no "/" is itself a
            # vendor:route selector with no model.
            if "/" in model_override:
                left = model_override.split("/", 1)[0]
            else:
                left = model_override
            override_vendor = left.split(":", 1)[0]
            if override_vendor:
                authorized_vendors.add(override_vendor)
            # Explicit only when the selector pins ONE concrete route. A
            # route-qualified selector (``vendor:route``, contains ":") names a
            # single route by composite key, so it is always explicit when it
            # matches. A vendor-only selector (``vendor``, no ":") matches every
            # route for that vendor — it is explicit only if it narrows to
            # EXACTLY ONE route. A vendor-wide selector that matches 2+ routes
            # (e.g. "openai" → openai:plan AND openai:api) must NOT be explicit,
            # so same-vendor fallback among the matched routes still works.
            matched = self._match_selector(available, left)
            if matched and (":" in left or len(matched) == 1):
                explicit_selection = True
            # P2a: override precedence — resolve_provider_routing ignores the
            # persisted mandate (and therefore its fallbacks) entirely when a
            # concrete override is present. Return now so neither the mandate
            # vendor nor stale mandate fallbacks contaminate the authorized set.
            return RoutingMeta(explicit_selection, authorized_vendors)

        # --- 2. mandate branch (no override) ---
        pref = getattr(self, "_mandate_preference", None) or {}
        pref_vendor = pref.get("vendor")
        pref_route = pref.get("route")
        mandate_active = bool(pref.get("model") and pref_vendor)
        if mandate_active:
            authorized_vendors.add(pref_vendor)
            selector = f"{pref_vendor}:{pref_route}" if pref_route else pref_vendor
            mandate_matched = self._match_selector(available, selector)
            # Explicit only when the mandate pins ONE concrete route: a
            # route-qualified selector (``pref_route`` set → "vendor:route")
            # names a single route, while a vendor-only mandate ("vendor")
            # matches every route for that vendor and is explicit only if it
            # narrows to exactly one. A vendor-wide mandate matching 2+ routes
            # must NOT be explicit so same-vendor fallback still works.
            if mandate_matched and (pref_route or len(mandate_matched) == 1):
                explicit_selection = True

            # --- 3. mandate fallbacks (only when the mandate is active AND
            # its preferred route did NOT match) — the exact condition under
            # which resolve_provider_routing builds the fallback chain. P2b:
            # NEVER fold these in for a default-routing request with no active
            # mandate, or a stale fallback list would wrongly authorize/drop
            # default providers.
            if not mandate_matched:
                for fb in (getattr(self, "_mandate_fallbacks", None) or []):
                    fb_vendor = fb.get("vendor") or fb.get("provider")
                    if fb_vendor:
                        authorized_vendors.add(fb_vendor)

        return RoutingMeta(explicit_selection, authorized_vendors)

    def _check_model_tool_support(
        self,
        providers: list,
        tools: Optional[list],
        model_override: Optional[str] = None,
    ) -> Optional[list]:
        """Check if the target model supports tools; strip them if not.

        Cloud routes always support tools (every cloud vendor's chat API does).
        Local routes may run small models that can't tool-call — we fall
        through to the discovered ``ModelInfo.supports_tools`` flag.
        """
        if not tools:
            return tools

        if not providers:
            return tools
        target_route = providers[0]
        is_cloud = (
            target_route.get("is_cloud")
            if isinstance(target_route, dict)
            else getattr(target_route, "is_cloud", True)
        )
        if is_cloud:
            return tools  # Cloud routes always support tools.

        # Resolve which model we'll actually use
        target_model = model_override
        if target_model and "/" in target_model:
            _, target_model = target_model.split("/", 1)
        if not target_model and providers:
            p = providers[0]
            target_model = p["model"] if isinstance(p, dict) else getattr(p, "model", None)

        if not target_model:
            return tools  # Can't determine model, pass tools through

        # Check discovered model info (exact match only — no substring matching)
        from .model_cache import get_shared_model_cache
        cache = get_shared_model_cache().get_any()
        if not cache:
            return tools  # No discovery data yet, pass tools through

        for model_info in cache:
            if model_info.id == target_model:
                if not model_info.supports_tools:
                    logger.info(
                        f"Model {target_model} does not support tools "
                        f"({model_info.size_gb or '?'}GB) — sending without tools"
                    )
                    return None
                return tools

        return tools  # Model not in cache, pass tools through

    def _get_local_provider_names(self) -> set:
        """Route keys (``"vendor:route"``) for all local routes.

        Retained as a convenience for call sites that pre-date the
        ``is_local`` flag on provider dicts; prefer reading ``p["is_local"]``
        directly in new code.
        """
        try:
            if hasattr(self, 'provider_registry') and self.provider_registry:
                locals_ = self.provider_registry.get_local_providers()
                if locals_:
                    return {p.name for p in locals_}
        except (TypeError, AttributeError):
            pass
        return set()

    def _streamed_call_cost(self, response: Any) -> Optional[float]:
        """Per-call cost for a streamed turn (#1806), if the composed service
        exposes the extractor. Defensive so a StreamingMixin-only consumer
        (e.g. a unit fake) doesn't break — returns None when unavailable."""
        extractor = getattr(self, "_extract_provider_cost", None)
        if extractor is None:
            return None
        try:
            return extractor(response)
        except Exception:  # noqa: BLE001 - cost is best-effort
            return None

    async def _record_streamed_usage(
        self,
        response: Any,
        model: str,
        provider_name: str,
        *,
        duration_ms: int,
        partial: bool = False,
    ) -> None:
        """Meter a streamed turn from its terminal :class:`LLMResponse`.

        The streaming path never reached ``_track_model_usage`` /
        ``_log_llm_call`` (the non-streaming chokepoint), so every streamed
        turn silently bypassed usage tracking and the billing meter. This
        mirrors the non-streaming recording (service.py) from the terminal
        response. Best-effort: a recording failure must never break the
        stream the user is consuming.

        ``partial=True`` flags a mid-stream-abort flush (no terminal response
        arrived). Tokens were still consumed/billed by the provider, so the
        usage is recorded; the metadata flag lets telemetry tell it apart.
        """
        if not isinstance(response, LLMResponse):
            return
        try:
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
            await self._track_model_usage(model, provider_name, tokens=total_tokens)
            metadata = {"streamed": True}
            if partial:
                metadata["partial_abort"] = True
            await self._log_llm_call(
                provider=provider_name,
                model=model,
                duration_ms=duration_ms,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=getattr(
                    response, "cache_creation_input_tokens", None
                ),
                cache_read_input_tokens=getattr(
                    response, "cache_read_input_tokens", None
                ),
                tools_used=bool(getattr(response, "tool_calls", None)),
                metadata=metadata,
                cost=self._streamed_call_cost(response),
            )
        except Exception as exc:  # noqa: BLE001 - metering must not break stream
            logger.warning("Failed to record streamed usage: %s", exc)

    async def get_streaming_response(
        self,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: str = None,
        response_format: Optional[Type[BaseModel]] = None,
        cancel_token: Optional[CancelToken] = None,
    ):
        """Get a streaming response from the LLM.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers (Ollama)
            model_override: Override the model selection (format: "provider/model")
            response_format: Optional Pydantic model for structured output.
                Note: Not all providers support streaming with structured output.
                OpenAI supports it natively, others may fall back to non-streaming.

        Yields:
            Text chunks as they arrive
        """
        self._check_policy()
        resolution = await self._resolve_routing_with_discovery(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        providers_to_use, target_model = resolution
        explicit_selection, configured_vendors = resolution.meta

        last_error = None
        last_provider_name = None
        for provider_index, provider in enumerate(providers_to_use):
            if not explicit_selection and self._skip_unconfigured_route(
                provider, configured_vendors
            ):
                logger.warning(
                    "Skipping unconfigured-vendor route %s (vendor %s not "
                    "authorized); refusing blind cross-vendor fallback.",
                    provider.get("name"), provider.get("vendor"),
                )
                continue
            try:
                provider_name = provider["name"]
                last_provider_name = provider_name
                model_to_use = self._resolve_concrete_model(target_model, provider)

                logger.info(f"Attempting streaming from {provider_name} with {model_to_use}")
                messages = messages_for(provider["adapter"], user_prompt=user_prompt, system_prompt=system_prompt)

                adapter = provider["adapter"]

                # For structured output, only routes whose typed capabilities
                # advertise a streamable structured-output mode (and opt in via
                # contract_features) can stream a response_format. Anthropic's
                # TOOL_FORCED mode buffers a tool call, so it stays on the
                # non-streaming fallback below.
                supports_streaming_structured = _route_supports_streaming_structured(provider)

                # Use streaming if supported (or no structured output requested)
                if hasattr(adapter, "get_streaming_response"):
                    if response_format is None or supports_streaming_structured:
                        try:
                            async for chunk in adapter.get_streaming_response(
                                client=provider["client"],
                                model=model_to_use,
                                messages=messages,
                                response_format=response_format,
                                extra_body=provider_cache_body(provider),
                                cancel_token=cancel_token,
                            ):
                                yield chunk
                            logger.info(f"Streaming completed from {provider_name}")
                            return
                        except NotImplementedError:
                            # Adapter doesn't support streaming, fall through to non-streaming
                            pass

                # Fallback: use non-streaming response (required for Anthropic with structured output)
                response = await adapter.get_response(
                    client=provider["client"],
                    model=model_to_use,
                    messages=messages,
                    response_format=response_format,
                    extra_body=provider_cache_body(provider),
                    cancel_token=cancel_token,
                )
                # Yield content as string (LLMResponse.content) to match streaming behavior
                yield response.content or ""
                logger.info(f"Non-streaming fallback from {provider_name}")
                return

            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                if _is_harness_owned_transport_error(e):
                    # Harness-owned transport error (codex app-server idle
                    # stall, app-server connection closed, etc.). Don't
                    # rotate to a different provider — that would answer
                    # the user from a wrong model on a transient codex
                    # stall. Also skip ``_maybe_disable_route``: even an
                    # auth-shaped codex message ("session expired",
                    # "unauthorized") is the harness's responsibility, not
                    # evidence the kestrel route is broken — disabling it
                    # for the rest of the process would skip codex on
                    # every future turn even after the operator
                    # re-authenticates. See #1429 and openclaw commit
                    # 3a64dc7623.
                    raise LLMStreamingError(
                        f"Harness-owned route {provider_name} failed: {e}",
                        provider=provider_name,
                        underlying=e,
                    )
                self._maybe_disable_route(provider, e)
                if explicit_selection:
                    # No silent fallthrough when the user has explicitly narrowed
                    # routing. Fail loudly — the caller / agent / user must see
                    # the specific error, not a response from a different model.
                    raise LLMStreamingError(
                        f"Selected route {provider_name} failed: {e}",
                        provider=provider_name,
                        underlying=e,
                    )
                if self._configured_routes_exhausted(
                    providers_to_use, provider_index, configured_vendors
                ):
                    logger.error(
                        "Configured routes exhausted: preferred vendor %s route "
                        "%s failed and every remaining route is an unconfigured "
                        "vendor. Error: %s",
                        providers_to_use[0].get("vendor"), provider_name, e,
                    )
                    raise LLMStreamingError(
                        f"Preferred route {provider_name} failed and the only "
                        f"remaining routes are unconfigured vendors; refusing to "
                        f"silently swap vendors: {e}",
                        provider=provider_name,
                        underlying=e,
                    )
                # Default multi-provider chain: log the fallback server-side;
                # don't corrupt the stream with a note about it.
                logger.warning(
                    "Falling through from %s to next provider in chain: %s",
                    provider_name, e,
                )
                continue

        provider_type = "local" if force_local_only else "all"
        logger.error(f"All {provider_type} providers failed for streaming. Last error: {last_error}")
        raise LLMStreamingError(
            f"All {provider_type} providers failed: {last_error}",
            provider=last_provider_name,
            underlying=last_error,
        )

    async def generate_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        cancel_token: Optional[CancelToken] = None,
    ):
        """Stream text using the active backend with automatic fallback.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers
            model_override: Override model selection
            response_format: Optional Pydantic model for structured output.
                Note: Streaming with structured output is provider-dependent.
                OpenAI supports it natively, others may fall back to non-streaming.

        Yields:
            Text chunks as they arrive (JSON chunks if response_format provided)
        """
        self._check_policy()
        from .remote_backend import BackendType

        # Try remote GPU first when active AND routing isn't pinned — #734.
        if (
            self._backend == BackendType.REMOTE_GPU
            and self._remote_client
            and not force_local_only
            and self._remote_first_allowed(model_override)
        ):
            try:
                self._ensure_remote_active()
                messages = messages_for(self._remote_adapter, user_prompt=user_prompt, system_prompt=system_prompt)
                model = self._scrub_auto(model_override) or self._remote_config.model
                async for chunk in self._remote_adapter.get_streaming_response(
                    client=self._remote_client,
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    cancel_token=cancel_token,
                ):
                    yield chunk
                return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard streaming
        async for chunk in self.get_streaming_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_local_only=force_local_only,
            model_override=model_override,
            response_format=response_format,
            cancel_token=cancel_token,
        ):
            yield chunk

    async def stream_with_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        session_id: Optional[str] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> AsyncIterator[str]:
        """Stream response using a pre-built messages array.

        Use this for streaming the final response after tool execution,
        where you need to pass the full conversation history including
        tool results.

        Args:
            messages: Pre-built message list including tool results
            force_local_only: Only use local providers
            model_override: Override model selection
            session_id: See ``generate_with_messages``. #808.

        Yields:
            Text chunks as they arrive from the LLM
        """
        self._check_policy()
        from .remote_backend import BackendType

        # Try remote GPU first when active AND routing isn't pinned — #734.
        if (
            self._backend == BackendType.REMOTE_GPU
            and self._remote_client
            and not force_local_only
            and self._remote_first_allowed(model_override)
        ):
            try:
                self._ensure_remote_active()
                model = self._scrub_auto(model_override) or self._remote_config.model
                if hasattr(self._remote_adapter, "get_streaming_response"):
                    async for chunk in self._remote_adapter.get_streaming_response(
                        client=self._remote_client,
                        model=model,
                        messages=messages,
                        cancel_token=cancel_token,
                    ):
                        yield chunk
                    return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard providers — use centralized routing
        resolution = await self._resolve_routing_with_discovery(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        providers, target_model = resolution
        explicit_selection, configured_vendors = resolution.meta

        last_error = None
        last_provider_name = None
        for provider_index, provider in enumerate(providers):
            if not explicit_selection and self._skip_unconfigured_route(
                provider, configured_vendors
            ):
                logger.warning(
                    "Skipping unconfigured-vendor route %s (vendor %s not "
                    "authorized) in stream_with_messages; refusing blind "
                    "cross-vendor fallback.",
                    provider.get("name"), provider.get("vendor"),
                )
                continue
            try:
                last_provider_name = provider["name"]
                adapter = provider["adapter"]
                model = self._resolve_concrete_model(target_model, provider)

                if hasattr(adapter, "get_streaming_response"):
                    async for chunk in adapter.get_streaming_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                        extra_body=provider_cache_body(provider),
                        session_id=session_id,
                        cancel_token=cancel_token,
                    ):
                        yield chunk
                    return
                else:
                    # Fallback to non-streaming if adapter doesn't support it
                    response = await adapter.get_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                        extra_body=provider_cache_body(provider),
                        session_id=session_id,
                        cancel_token=cancel_token,
                    )
                    yield response.content if hasattr(response, 'content') else str(response)
                    return
            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                if _is_harness_owned_transport_error(e):
                    # See #1429: skip _maybe_disable_route too — harness
                    # owns auth, kestrel doesn't disable the route on its
                    # behalf.
                    raise LLMStreamingError(
                        f"Harness-owned route {provider['name']} failed: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                self._maybe_disable_route(provider, e)
                if explicit_selection:
                    raise LLMStreamingError(
                        f"Selected route {provider['name']} failed: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                if self._configured_routes_exhausted(
                    providers, provider_index, configured_vendors
                ):
                    logger.error(
                        "Configured routes exhausted in stream_with_messages: "
                        "preferred vendor %s route %s failed and every remaining "
                        "route is an unconfigured vendor. Error: %s",
                        providers[0].get("vendor"), provider["name"], e,
                    )
                    raise LLMStreamingError(
                        f"Preferred route {provider['name']} failed and the "
                        f"only remaining routes are unconfigured vendors; "
                        f"refusing to silently swap vendors: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                logger.warning(
                    "Falling through from %s: %s", provider["name"], e,
                )
                continue

        logger.error(f"All providers failed for stream_with_messages: {last_error}")
        raise LLMStreamingError(
            f"All providers failed: {last_error}",
            provider=last_provider_name,
            underlying=last_error,
        )

    @staticmethod
    def _adapter_supports_vision(adapter: Any) -> bool:
        """True when the adapter *family* can accept image input."""
        try:
            caps = adapter.provider_capabilities()
        except Exception:
            return False
        return bool(getattr(caps, "supports_vision", False))

    @staticmethod
    def _discovered_model_supports_vision(provider_name: str, model: str):
        """Per-model vision support from discovery, or ``None`` if unknown.

        The adapter-family flag is too coarse: an Ollama/OpenAI/OpenRouter
        route reports ``supports_vision=True`` at the adapter level even when
        the *configured* model is text-only (vision is in ``model_dependent``).
        Discovery already computes per-model ``supports_vision``; consult it so
        an image isn't shipped to a model that will reject it.
        """
        try:
            from .model_cache import get_shared_model_cache
            models = get_shared_model_cache().get_any() or []
        except Exception:
            return None
        # Route names are ``vendor:route`` (e.g. ``openai:api``) but
        # ModelInfo.provider is the bare vendor (``openai``); compare on vendor.
        vendor = (provider_name or "").split(":", 1)[0]
        for info in models:
            if getattr(info, "id", None) != model:
                continue
            prov = getattr(info, "provider", None)
            # Guard against id collisions across providers.
            if prov in (None, provider_name, vendor):
                return bool(getattr(info, "supports_vision", False))
        return None

    def _turn_can_see_images(self, adapter: Any, provider_name: str, model: str) -> bool:
        """Whether this concrete provider+model can accept image input.

        Concrete-model metadata wins when discovery knows the model; otherwise
        fall back to the adapter-family capability (the conservative default
        for models discovery hasn't catalogued).
        """
        model_vision = self._discovered_model_supports_vision(provider_name, model)
        if model_vision is not None:
            return model_vision
        return self._adapter_supports_vision(adapter)

    def _apply_eager_vision(
        self,
        adapter: Any,
        messages: List[Dict[str, Any]],
        images: Optional[List[Union[str, bytes]]],
        provider_name: str,
        model: str,
    ) -> List[Dict[str, Any]]:
        """Fold this turn's eager image attachments (#1662) into the last user
        message in the *resolved* provider's native vision format.

        Vision shape is provider-specific and routing is dynamic, so the fold
        happens here — after a provider is chosen — not at the agent layer. If
        the resolved model can't see images, leave the messages untouched and
        warn LOUDLY rather than silently dropping the user's image (no blind
        fallbacks): the turn still runs as text, and the log says why.
        """
        if not images:
            return messages
        if not self._turn_can_see_images(adapter, provider_name, model):
            logger.warning(
                "Eager vision: %s/%s is not vision-capable; %d image "
                "attachment(s) were NOT sent to the model this turn. Switch to "
                "a vision-capable model to let the agent see pasted images.",
                provider_name, model, len(images),
            )
            return messages
        if not hasattr(adapter, "attach_images_to_last_user_message"):
            # Vision-capable, but a third-party adapter that doesn't implement
            # the fold helper — say THAT, don't mislabel the model as blind.
            logger.error(
                "Eager vision: %s/%s reports vision support but cannot fold "
                "images (no attach_images_to_last_user_message); %d "
                "attachment(s) dropped this turn.",
                provider_name, model, len(images),
            )
            return messages
        return adapter.attach_images_to_last_user_message(messages, images)

    async def stream_with_tool_detection(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        keep_trailing_system: bool = False,
        cancel_token: Optional[CancelToken] = None,
    ) -> AsyncIterator[Union[str, ThinkingDelta, ToolCallStarted, LLMResponse]]:
        """
        Stream response with tool call detection.

        Unified streaming-with-tools across all providers. Yields the
        SDK 0.7+ tagged union from
        :meth:`LLMAdapter.get_streaming_response_with_tools`:

        * ``str`` — text content chunks as they arrive.
        * :class:`ToolCallStarted` — emitted the moment a tool call
          first appears in the provider stream (one event per
          distinct ``index``, in stream order). The constitutional
          honesty layer (#1042 layer 2 / #1045) gates pre-tool prose
          on this signal — consumers that pipe text directly to a
          chat UI should clear or revise the in-flight bubble when a
          marker arrives. ``stream_with_tool_detection`` does not
          itself process or filter markers; it forwards them
          unchanged from the underlying adapter.
        * :class:`ThinkingDelta` — provider-separated model reasoning
          that should be displayed as expandable UI affordance, not as
          assistant answer text or persisted conversation content.
        * :class:`LLMResponse` — exactly once at end-of-stream when
          tool calls were detected. Source of truth for the assembled
          tool calls (id, name, arguments) and token usage.

        This eliminates the "double LLM call" pattern where you first
        called non-streaming to detect tools then streaming for text.

        Args:
            messages: Pre-built message list
            tools: Optional tools for function calling
            force_local_only: Only use local providers (Ollama)
            model_override: Override model selection (format:
                ``"provider/model"`` or just ``"model"``)
            system_prompt: Optional system prompt (only used for
                Anthropic adapter)
            images: Optional eager image attachments for *this* turn
                (#1662). Folded into the last user message in the resolved
                provider's native vision format once a provider is chosen;
                ignored with a loud warning if that provider can't see images.

        Yields:
            ``Union[str, ToolCallStarted, LLMResponse]`` per the
            stream contract above.

        Example:
            tool_response = None
            async for item in service.stream_with_tool_detection(messages=msgs, tools=tools):
                if isinstance(item, str):
                    print(item, end='', flush=True)  # stream text to user
                elif isinstance(item, ToolCallStarted):
                    # Stop optimistic text rendering; a tool call is
                    # about to fire. Frontend may clear the in-flight
                    # message bubble here.
                    on_tool_starting(item)
                elif isinstance(item, LLMResponse):
                    tool_response = item

            if tool_response and tool_response.has_tool_calls:
                # Execute tools and continue
                for tc in tool_response.tool_calls:
                    result = await execute_tool(tc)
        """
        self._check_policy()
        from .remote_backend import BackendType

        # Try remote GPU first when active AND routing isn't pinned — #734.
        if (
            self._backend == BackendType.REMOTE_GPU
            and self._remote_client
            and not force_local_only
            and self._remote_first_allowed(model_override)
            # Never shortcut to the remote GPU for an image-bearing turn (#1662).
            # `_remote_adapter` is a fixed OpenAIAdapter whose static capability
            # flag can't tell us whether the *configured* remote model (often a
            # text-only local GGUF) can actually see images — so probing it
            # would lie. Fall through to normal routing, which resolves vision
            # per concrete model and picks a vision-capable provider.
            and not images
        ):
            try:
                self._ensure_remote_active()
                model = self._scrub_auto(model_override) or self._remote_config.model
                self._stamp_response_identity(
                    None, model=model, provider="remote_gpu",
                )
                if hasattr(self._remote_adapter, "get_streaming_response_with_tools"):
                    async for item in self._remote_adapter.get_streaming_response_with_tools(
                        client=self._remote_client,
                        model=model,
                        messages=messages,
                        tools=tools,
                        cancel_token=cancel_token,
                    ):
                        if isinstance(item, LLMResponse):
                            self._stamp_response_identity(
                                item, model=model, provider="remote_gpu",
                            )
                        yield item
                    return
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU streaming with tools failed: {exc}, falling back")
                self._deactivate_remote_backend(reason=str(exc))

        # Use centralized provider routing
        resolution = await self._resolve_routing_with_discovery(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        providers, target_model = resolution
        explicit_selection, configured_vendors = resolution.meta

        # Strip tools if the target model can't handle them
        tools = self._check_model_tool_support(providers, tools, model_override)

        last_error = None
        last_provider_name = None
        for provider_index, provider in enumerate(providers):
            if not explicit_selection and self._skip_unconfigured_route(
                provider, configured_vendors
            ):
                logger.warning(
                    "Skipping unconfigured-vendor route %s (vendor %s not "
                    "authorized) in stream_with_tool_detection; refusing "
                    "blind cross-vendor fallback.",
                    provider.get("name"), provider.get("vendor"),
                )
                continue
            try:
                adapter = provider["adapter"]
                model = self._resolve_concrete_model(target_model, provider)
                provider_name = provider["name"]
                self._stamp_response_identity(
                    None, model=model, provider=provider_name,
                )

                logger.info(f"Attempting streaming with tools from {provider_name} with {model}")

                # Fold this turn's eager images into the last user message in
                # the resolved provider's native format BEFORE either dispatch
                # branch, so the non-streaming fallback (e.g. Gemini, which has
                # no get_streaming_response_with_tools) sees them too (#1662).
                adapter_messages = self._apply_eager_vision(
                    adapter, messages, images, provider_name, model
                )

                # Check if adapter supports streaming with tool detection
                if hasattr(adapter, "get_streaming_response_with_tools"):
                    # Build kwargs for provider-specific parameters
                    kwargs = {}
                    if system_prompt and _route_wants_tool_stream_system_prompt(provider):
                        kwargs["system_prompt"] = system_prompt
                    cache_body = provider_cache_body(provider)
                    if cache_body:
                        kwargs["extra_body"] = cache_body
                    if session_id:
                        kwargs["session_id"] = session_id
                    if tool_executor is not None:
                        kwargs["tool_executor"] = tool_executor
                    if cancel_token is not None:
                        kwargs["cancel_token"] = cancel_token
                    if keep_trailing_system:
                        kwargs["keep_trailing_system"] = True

                    # Meter the streamed turn from its terminal LLMResponse.
                    # The `finally` records even if the consumer stops iterating
                    # after the terminal response arrives. For adapters that
                    # report usage incrementally (Anthropic), pass a usage_sink
                    # so a true mid-stream abort — before the terminal response
                    # — can still flush the partial usage the provider billed
                    # (#1684). Adapters that only surface usage at stream end
                    # leave the sink empty, so the abort path records nothing
                    # (there is nothing to record).
                    usage_sink: Dict[str, Any] = {}
                    if getattr(adapter, "supports_partial_usage_flush", False):
                        kwargs["usage_sink"] = usage_sink
                    stream_start = time.monotonic()
                    final_response = None
                    try:
                        async for item in adapter.get_streaming_response_with_tools(
                            client=provider["client"],
                            model=model,
                            messages=adapter_messages,
                            tools=tools,
                            **kwargs
                        ):
                            if isinstance(item, LLMResponse):
                                self._stamp_response_identity(
                                    item, model=model, provider=provider_name,
                                )
                                final_response = item
                            yield item
                    finally:
                        duration_ms = int((time.monotonic() - stream_start) * 1000)
                        if final_response is not None:
                            # Normal end-of-stream: terminal response carries the
                            # authoritative usage (and supersedes the sink).
                            await self._record_streamed_usage(
                                final_response, model, provider_name,
                                duration_ms=duration_ms,
                            )
                        elif usage_sink:
                            # Aborted before the terminal response — flush what
                            # the adapter captured incrementally.
                            await self._record_streamed_usage(
                                LLMResponse(
                                    content=None,
                                    tool_calls=None,
                                    input_tokens=usage_sink.get("input_tokens"),
                                    output_tokens=usage_sink.get("output_tokens"),
                                ),
                                model, provider_name,
                                duration_ms=duration_ms,
                                partial=True,
                            )
                    logger.info(f"Streaming with tools completed from {provider_name}")
                    return
                else:
                    # Fallback: use non-streaming for tool detection, then stream text
                    logger.warning(f"{provider_name} doesn't support streaming with tools, using fallback")
                    if tools:
                        fb_start = time.monotonic()
                        response = await adapter.get_response(
                            client=provider["client"],
                            model=model,
                            messages=adapter_messages,
                            tools=tools,
                            extra_body=provider_cache_body(provider),
                            session_id=session_id,
                            keep_trailing_system=keep_trailing_system,
                            cancel_token=cancel_token,
                        )
                        self._stamp_response_identity(
                            response, model=model, provider=provider_name,
                        )
                        # adapter.get_response does not meter (only the service's
                        # non-streaming path does), so record it here too.
                        await self._record_streamed_usage(
                            response, model, provider_name,
                            duration_ms=int((time.monotonic() - fb_start) * 1000),
                        )
                        if response.has_tool_calls:
                            yield response
                            return
                        # No tool calls, yield content
                        if response.content:
                            yield response.content
                        return
                    else:
                        # No tools, just stream
                        async for chunk in adapter.get_streaming_response(
                            client=provider["client"],
                            model=model,
                            messages=adapter_messages,
                            extra_body=provider_cache_body(provider),
                            session_id=session_id,
                            keep_trailing_system=keep_trailing_system,
                            cancel_token=cancel_token,
                        ):
                            yield chunk
                        return

            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                last_error = e
                last_provider_name = provider["name"]
                if _is_harness_owned_transport_error(e):
                    # See #1429: skip _maybe_disable_route too — harness
                    # owns auth, kestrel doesn't disable the route on its
                    # behalf.
                    raise LLMStreamingError(
                        f"Harness-owned route {provider['name']} failed: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                self._maybe_disable_route(provider, e)
                if explicit_selection:
                    raise LLMStreamingError(
                        f"Selected route {provider['name']} failed: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                if self._configured_routes_exhausted(
                    providers, provider_index, configured_vendors
                ):
                    logger.error(
                        "Configured routes exhausted in "
                        "stream_with_tool_detection: preferred vendor %s route "
                        "%s failed and every remaining route is an unconfigured "
                        "vendor. Error: %s",
                        providers[0].get("vendor"), provider["name"], e,
                    )
                    raise LLMStreamingError(
                        f"Preferred route {provider['name']} failed and the "
                        f"only remaining routes are unconfigured vendors; "
                        f"refusing to silently swap vendors: {e}",
                        provider=provider["name"],
                        underlying=e,
                    )
                logger.warning(
                    "Falling through from %s: %s", provider["name"], e,
                )
                continue

        logger.error(f"All providers failed for stream_with_tool_detection: {last_error}")
        raise LLMStreamingError(
            f"All providers failed: {last_error}",
            provider=last_provider_name,
            underlying=last_error,
        )
