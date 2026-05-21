"""
LLM Service - Unified LLM provider management with remote GPU support.

This is the single entry point for all LLM operations. It handles:
- Multiple provider initialization and fallback
- Remote GPU backend switching (RunPod, etc.)
- Model mandate routing
- Usage tracking
"""
import logging
import re
import json
import os
import time
import asyncio
import inspect
from kestrel_sovereign.kestrel_config.constants import STORAGE_CACHE_TTL_SECONDS
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, List, Dict, Any, Optional, Union, Type, TYPE_CHECKING

import openai
import httpx

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase
from dotenv import load_dotenv
from pydantic import BaseModel

from .provider_registry import (
    ProviderRegistry,
    ProviderInfo,
    ProviderInitializationError,
    provider_cache_body,
)
from .error_handling import (
    handle_llm_errors,
    handle_observability_errors,
    handle_storage_errors,
    LLMError,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMAllProvidersFailedError
)
from .openai_adapter import OpenAIAdapter
from .adapter import LLMResponse, messages_for
from .model_discovery import ModelDiscoveryMixin
from .mandate import ModelMandateMixin
from .usage_tracking import UsageTrackingMixin
from .streaming import StreamingMixin
from .constitutional_awareness import ConstitutionalAwarenessMixin
from .remote_backend import RemoteBackendMixin, BackendType, RemoteGPUConfig
from kestrel_sovereign.kestrel_config.constants import (
    HTTP_TIMEOUT_MEDIUM,
    CLIENT_CLOSE_TIMEOUT,
)
from kestrel_sovereign.config import load_config, load_section
from kestrel_sovereign.telemetry import optional_span

logger = logging.getLogger(__name__)


async def _wait_for_close_result(result: Any) -> None:
    """Await asynchronous close results while accepting synchronous close APIs."""
    if inspect.isawaitable(result):
        await asyncio.wait_for(asyncio.shield(result), timeout=CLIENT_CLOSE_TIMEOUT)


def resolve_active_model_selection(llm_service) -> Dict[str, Optional[str]]:
    """Resolve canonical current-selection metadata for any LLM-service-like object.

    Returns a dict with keys:
        vendor:     e.g. ``"openai"`` — None only if no routes are configured.
        route:      e.g. ``"api"`` — None when the mandate specifies only a vendor.
        model_name: the model ID.
        model:      display form ``"<vendor>/<model_name>"`` or ``"<vendor>:<route>/<model_name>"``.
    """
    pref = llm_service.get_model_preference() or {}
    model_name = pref.get("model")
    vendor = pref.get("vendor")
    route = pref.get("route")

    providers = getattr(llm_service, "providers", None)
    if not model_name and providers:
        first = providers[0]
        vendor = vendor or first.get("vendor") or first.get("name")
        route = route or first.get("route")
        model_name = first.get("model")

    if not model_name:
        model_name = "auto"

    if vendor and route:
        full = f"{vendor}:{route}/{model_name}"
    elif vendor:
        full = f"{vendor}/{model_name}"
    else:
        full = model_name
    return {
        "model": full,
        "vendor": vendor,
        "route": route,
        "model_name": model_name,
    }


class LLMServiceError(LLMError):
    """Raised when LLM service cannot fulfill a request."""


class PolicyDeniedError(LLMServiceError):
    """Raised when a generation call is attempted on an LLMService whose
    PayerPolicy slot is `PayerKind.NONE`.

    The agent-init layer sets `LLMService.disabled = True` when the
    agent's policy says "no LLM at all" for this agent. Every public
    generation entry point on `LLMService` (and its mixins) calls
    `_check_policy()` at the top, which raises this error when
    `disabled` is True. Callers (chat endpoints, reflection loops,
    audit pipelines) treat this the same way they treat
    `KeyNotConfiguredError` today: the agent has no LLM available;
    surface a structured error rather than silently falling through to
    a shared host key.
    """


class LLMServiceAlreadyAttachedError(LLMServiceError):
    """Raised when a second agent tries to claim an LLMService that is
    already attached to a different agent.

    The PayerPolicy work in #1156's follow-up requires that each
    `KestrelAgent` instance holds its own `LLMService` instance —
    `LLMService.use_agent_key()` mutates `self.providers` in place, so a
    shared instance would silently leak the last-loaded agent's
    OpenRouter client to every other agent. Production code already
    constructs one service per agent (see
    `kestrel_sovereign/multi_agent/agent_manager.py:90-91`), but the
    invariant is now enforced at construction time so it cannot
    silently regress.
    """


class ModelNotAvailableForRoute(LLMError):
    """Raised by _try_single_provider when the target model isn't in the
    route's vendor catalog.

    Signals the outer fallback loop to skip this provider and try the next,
    instead of firing a request that will either 404/400 (cloud provider
    rejects the model) or silently serve the wrong weights (llama.cpp
    ignores the model name).
    """

    def __init__(self, vendor: Optional[str], route: Optional[str], model: str):
        self.vendor = vendor
        self.route = route
        self.model = model
        route_key = f"{vendor}:{route}" if vendor and route else (vendor or "unknown")
        super().__init__(
            f"Model '{model}' is not available for route '{route_key}'. "
            "Skipping — callers should target a vendor that serves this model."
        )


def _warn_no_llm_config_found() -> None:
    """Loud warning when ``[llm]`` is empty/missing on LLMService startup.

    Pre-#940 the LLMService would auto-copy ``llm_config.toml.example`` to
    ``llm_config.toml`` on first run. After #940 we read ``[llm]`` from
    ``kestrel.toml`` directly with no auto-copy fallback, so a fresh
    checkout (or one where the user never ran ``kestrel migrate-llm-config``)
    silently boots with zero providers — the UI shows empty Provider and
    "loading" Model with no log signal pointing at the real cause.

    This makes the failure loud and tells the user exactly which command
    fixes it. Logging at WARNING (not ERROR) because the registry will
    raise a clearer LLMServiceError later if anyone actually tries to use
    a route — this just guarantees the diagnostic shows up at startup.
    """
    from pathlib import Path

    legacy = Path("llm_config.toml")
    legacy_bak = Path("llm_config.toml.bak")
    example = Path("kestrel.toml.example")

    hint_lines = [
        "LLM config: no [llm] section found in kestrel.toml.",
        "Routes will not initialize and the model selector will appear empty.",
    ]
    if legacy.exists():
        hint_lines.append(
            "  Legacy llm_config.toml detected — run `kestrel migrate-llm-config` "
            "to fold it into kestrel.toml [llm]."
        )
    elif legacy_bak.exists():
        hint_lines.append(
            "  Legacy llm_config.toml.bak detected — your previous migration "
            "may have produced an empty [llm]; check kestrel.toml or restore "
            "from .bak and re-run `kestrel migrate-llm-config --force`."
        )
    elif example.exists():
        hint_lines.append(
            "  Run `kestrel setup llm` for interactive setup, or "
            "`cp kestrel.toml.example kestrel.toml` for the shipped defaults."
        )
    else:
        hint_lines.append(
            "  Run `kestrel setup llm` to create one."
        )

    logger.warning("\n  ".join(hint_lines))


class LLMService(ModelDiscoveryMixin, ModelMandateMixin, UsageTrackingMixin, StreamingMixin, ConstitutionalAwarenessMixin, RemoteBackendMixin):
    """Unified LLM service with provider fallback and remote GPU support."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize LLM service.

        Reads LLM configuration from the ``[llm]`` section of ``kestrel.toml``
        in the current project. The legacy standalone ``llm_config.toml``
        is no longer supported — run ``kestrel migrate-llm-config`` to fold
        an old file in.

        Args:
            database_url: Optional PostgreSQL connection URL for usage tracking.
                         If provided, uses PostgreSQL. Otherwise checks env vars,
                         then falls back to SQLite.
        """
        load_dotenv()

        # default_model is derived from first provider in provider_priority
        # (see provider_registry and endpoints/models.py)
        self.default_model = None  # Deprecated: use providers[0] instead
        self.config = load_section("llm")
        if not self.config:
            _warn_no_llm_config_found()
        self.mandate_config = load_config("model_mandate.toml")

        # Initialize provider registry
        self.provider_registry = ProviderRegistry(self.config)
        try:
            self.providers = self._convert_providers_format(self.provider_registry.initialize_providers())
        except ProviderInitializationError as e:
            logger.error(f"Failed to initialize providers: {e}")
            self.providers = []

        # Model discovery uses process-wide SharedModelCache (see model_cache.py).
        # Pre-populate from disk if this is the first LLMService instance.
        self._load_from_disk_cache()  # Immediate availability before API discovery

        # Storage info cache
        self._storage_cache = None
        self._storage_cache_timestamp = None
        self._storage_cache_ttl = STORAGE_CACHE_TTL_SECONDS

        # Per-agent claim. None until `attach_to_agent(agent_did)` is called;
        # set thereafter. Subsequent `attach_to_agent` calls with the SAME
        # DID are idempotent; calls with a DIFFERENT DID raise
        # LLMServiceAlreadyAttachedError. See `attach_to_agent` for the
        # invariant rationale.
        self._owner_agent_did: Optional[str] = None

        # PayerPolicy NONE flag. Set to True by the agent-init layer when
        # the agent's policy slot for LLM is `PayerKind.NONE`. Phase 3b
        # adds the `_check_policy()` guard on every generation entry point
        # that raises PolicyDeniedError when this is True. Phase 3a only
        # plumbs the flag; generation calls still go through (consistent
        # with HOST_ENV behavior) until 3b lands the guard.
        self.disabled: bool = False

        # Database for model usage tracking (uses abstract data layer)
        self._init_usage_tracking(database_url)

        # Constitutional profile service
        self._init_constitutional_profiles()

        # Runtime mandate state
        # Mandate preference schema: {"vendor": str|None, "model": str|None, "route": str|None}.
        # vendor + model are the primary selectors; route is optional and narrows
        # to an exact (vendor, route) pair. Stale rows using the old {"model", "provider"}
        # shape are dropped by model_preference._load_model_preference().
        self._mandate_preference = {"vendor": None, "model": None, "route": None}
        self._mandate_fallbacks = []

        # Remote GPU backend state (merged from BrainRouter)
        self._backend = BackendType.CLOUD
        self._default_backend = BackendType.CLOUD
        self._remote_config: Optional[RemoteGPUConfig] = None
        self._remote_client: Optional[openai.AsyncOpenAI] = None
        self._remote_adapter = OpenAIAdapter()
        self._last_remote_error: Optional[str] = None

        # Observability store for logging LLM calls (A2A-compatible)
        # Set via set_observability_store() after initialization
        self._observability_store = None
        self._observability_context: Dict[str, Any] = {}

        # Metering callback for usage billing (Vending Machine)
        # Set via set_metering_callback() after initialization
        self._metering_callback = None

        # Persistence callback for model preference (writes to database)
        # Set via set_preference_persistence_callback() after initialization
        self._preference_persistence_callback = None
        self._preference_persistence_tasks: set[asyncio.Task[None]] = set()

        # Routes that have failed with a permanent auth error (401/403 or
        # the equivalent "User not found" / "invalid api key" message). Once
        # a route lands here, subsequent fallback iterations skip it for the
        # lifetime of this service — retrying a dead API key every user turn
        # wastes a round-trip and logs a red error on every request (#655).
        # Cleared by restarting the service (picks up rotated keys). Keys
        # are provider route names (matches provider["name"]).
        self._disabled_routes: dict[str, str] = {}

    def set_preference_persistence_callback(self, callback) -> None:
        """Set the persistence callback for model preference.

        The callback will be called with (model: str|None, provider: str|None)
        whenever the model preference changes, so the caller can persist it.

        Args:
            callback: Async function(model, provider) to call on preference change
        """
        self._preference_persistence_callback = callback
        logger.info("Model preference persistence enabled")

    @staticmethod
    def _is_permanent_auth_error(exc: Exception) -> bool:
        """Return True for 401/403-class failures that will never recover
        without rotating the key or restarting the service.

        Matches the auth-specific subset of ``retry.NON_RETRYABLE_PATTERNS``
        plus explicit 401/403 status codes. Intentionally narrower than
        ``is_retryable_error`` negated — we don't want to disable a route
        because it rejected a single malformed request (400) or a missing
        model (404); those aren't problems with the route itself.
        """
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status in (401, 403):
            return True
        msg = str(exc).lower()
        auth_patterns = (
            "user not found",
            "invalid api key",
            "invalid_api_key",
            "unauthorized",
            "permission_denied",
            "permission denied",
            "authentication",
        )
        return any(p in msg for p in auth_patterns)

    def _maybe_disable_route(self, provider: Dict[str, Any], exc: Exception) -> None:
        """Record a route as permanently disabled if *exc* looks like a
        401/403. No-op for transient or caller-side errors. Logs once per
        route so repeated failures don't flood the log.
        """
        name = provider.get("name") if isinstance(provider, dict) else None
        if not name or name in self._disabled_routes:
            return
        if not self._is_permanent_auth_error(exc):
            return
        reason = str(exc)[:200]
        self._disabled_routes[name] = reason
        logger.warning(
            "Route %s disabled for the rest of this session after permanent "
            "auth failure: %s. Rotate the key and restart to re-enable.",
            name, reason,
        )

    def _available_providers(
        self, providers: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter out routes known to fail with a permanent auth error.

        Callers that iterate the fallback chain should use this instead of
        ``self.providers`` so each user turn skips dead routes immediately.
        If every route has been disabled (pathological), return an empty
        list — the caller raises a clear error rather than pretending.
        """
        src = providers if providers is not None else self.providers
        if not self._disabled_routes:
            return list(src)
        return [p for p in src if p.get("name") not in self._disabled_routes]

    def set_model_preference(
        self,
        model: str,
        vendor: Optional[str] = None,
        route: Optional[str] = None,
    ) -> None:
        """Set the mandated model selection for this session.

        When ``vendor`` is omitted, we resolve it from the discovery catalog
        before persisting. A bare model id with no vendor would otherwise
        broadcast across every provider in priority order on the next request
        (anthropic:plan → openai:plan → openrouter:api → ...), eventually
        landing on whichever backend happens to serve *something* with a
        matching id — which is how a "switch to gpt-5-mini" call ended up
        routing to OpenRouter's Gemini. The mandate must name a vendor.

        Args:
            model: The model ID to use. ``"auto"`` is ignored (means default routing).
            vendor: Optional vendor name (``"openai"``, ``"anthropic"``, ...).
                When given, only routes for this vendor are used. When
                omitted, auto-resolved from discovery.
            route: Optional route name (``"api"``, ``"plan"``, ``"local"``).
                When given with a vendor, narrows to the exact ``<vendor>:<route>``.

        Raises:
            ValueError: if vendor is omitted and the discovery catalog has
                zero matches (unknown model) or multiple matches (ambiguous).
        """
        if model == "auto":
            logger.debug("Ignoring model preference 'auto' — using default routing")
            return

        if vendor is None:
            resolved = self._resolve_vendor_for_model(model)
            if isinstance(resolved, list):
                raise ValueError(
                    f"Model '{model}' is ambiguous — served by {resolved}. "
                    f"Specify vendor explicitly via set_model_preference("
                    f"'{model}', vendor='{resolved[0]}'), or use the "
                    f"'<vendor>/<model>' form."
                )
            if resolved is None:
                # Catalog has no match. This is the broadcast-bug entry point
                # (LLM-tool calling set_model with a hallucinated name, UI
                # sending a stale model id, etc.). Refuse rather than persist
                # a vendor-less mandate and let the next request broadcast.
                raise ValueError(
                    f"Cannot set model '{model}' without a vendor: the "
                    f"discovery catalog has no match. Specify vendor "
                    f"explicitly, or run model discovery first."
                )
            vendor = resolved
            logger.info(
                "Auto-resolved vendor '%s' for model '%s' from discovery catalog",
                vendor, model,
            )

        self._mandate_preference = {"vendor": vendor, "model": model, "route": route}
        if route:
            logger.info("Model preference set: %s:%s/%s", vendor, route, model)
        else:
            logger.info("Model preference set: %s/%s", vendor, model)

        if self._preference_persistence_callback:
            self._schedule_preference_persistence(model, vendor, route)

    def _resolve_vendor_for_model(self, model: str) -> Optional[Any]:
        """Resolve which vendor serves a given model id from discovery.

        Returns:
            - ``str`` — the single vendor that serves the model.
            - ``list[str]`` — multiple vendors serve this id (ambiguous);
              caller must specify.
            - ``None`` — catalog has no match (unknown model, or discovery
              hasn't populated yet).
        """
        from .model_cache import get_shared_model_cache
        cache = get_shared_model_cache().get_any()
        if not cache:
            return None
        vendors = sorted({m.provider for m in cache if m.id == model and m.provider})
        if not vendors:
            return None
        if len(vendors) == 1:
            return vendors[0]
        return vendors

    def clear_model_preference(self) -> None:
        """Clear any mandated model preference, returning to default behavior."""
        self._mandate_preference = {"vendor": None, "model": None, "route": None}
        logger.info("Model preference cleared, using default route order")

        if self._preference_persistence_callback:
            self._schedule_preference_persistence(None, None, None)

    def _schedule_preference_persistence(
        self,
        model: Optional[str],
        vendor: Optional[str],
        route: Optional[str],
    ) -> Optional[asyncio.Task[None]]:
        """Own preference persistence callbacks so close() can await them.

        Callback signature is ``async (model, vendor, route) -> None``.
        """
        if not self._preference_persistence_callback:
            return None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — skip persistence (happens in tests/sync contexts)
            return None

        task = loop.create_task(
            self._preference_persistence_callback(model, vendor, route),
            name="llm-preference-persistence",
        )
        self._preference_persistence_tasks.add(task)
        task.add_done_callback(self._handle_preference_persistence_done)
        return task

    def _handle_preference_persistence_done(self, task: asyncio.Task[None]) -> None:
        self._preference_persistence_tasks.discard(task)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.warning(
                "Model preference persistence failed: %s",
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def drain_preference_persistence(self, *, cancel: bool = False) -> None:
        """Wait for scheduled model-preference persistence callbacks."""
        tasks = set(self._preference_persistence_tasks)
        if not tasks:
            return

        if cancel:
            for task in tasks:
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self._preference_persistence_tasks.difference_update(tasks)

    def get_active_model_id(self) -> str:
        """Get the resolved model ID currently in use.

        Single source of truth for model identity. Used by TokenCounter,
        ContextBuilder, TokenBudget, etc.

        Resolution order:
        1. Mandate preference (user selected via UI or !model-set)
        2. First provider's resolved model
        3. "auto" as last resort

        Returns:
            Model ID string (e.g., "Kimi-K2.5-...", "gpt-5-mini")
        """
        pref = self._mandate_preference
        if pref.get("model") and pref["model"] != "auto":
            return pref["model"]
        if self.providers:
            model = self.providers[0].get("model", "auto")
            if model != "auto":
                return model
        return "auto"

    def get_active_model_selection(self) -> Dict[str, Optional[str]]:
        """Return canonical current-model metadata for UI, commands, and runtime.

        The provider is only included when it is an explicit part of the
        selected route, or when no mandate exists and the default provider order
        determines the active route. A model-only mandate remains model-only.
        """
        return resolve_active_model_selection(self)

    def get_model_preference(self) -> Dict[str, Optional[str]]:
        """Get the current model preference.

        Returns:
            Dict with 'model' and 'provider' keys, values may be None.
        """
        return self._mandate_preference.copy()

    def _remote_first_allowed(self, model_override: Optional[str]) -> bool:
        """Return True iff the remote-GPU fast path may run for this call.

        Before the vendor/route refactor the remote GPU backend was tried
        first whenever it was active, ignoring persisted mandates and caller
        ``model_override`` hints.  That turned routing dishonest: the UI and
        ``get_active_model_id`` could report one route/model while the
        actual answer came from the remote pod.  See issue #734.

        The centralized decision now lives in
        :meth:`resolve_provider_routing`.  We only take the remote-GPU
        shortcut when nothing has narrowed routing to a specific vendor —
        i.e. no vendor/route in the persisted mandate and no
        vendor-prefixed ``model_override``.  A bare model string (no ``:``
        or ``/``) is allowed through since the remote route has its own
        configured model and the user hasn't pinned a backend.
        """
        if "/" in (model_override or "") or ":" in (model_override or ""):
            return False
        pref = self._mandate_preference
        if pref.get("vendor") or pref.get("route"):
            return False
        return True

    def resolve_provider_routing(
        self,
        *,
        model_override: Optional[str] = None,
        force_local_only: bool = False,
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Resolve which routes and model to use for the next LLM call.

        Single source of truth for routing. All call paths funnel through here.

        Resolution order:
            1. ``model_override`` — caller-supplied ``vendor/model`` or
               ``vendor:route/model`` or bare model string. If a vendor (or
               vendor:route) prefix is given, only matching routes are used.
            2. **Mandate preference** — persisted ``{vendor, model, route?}``.
               ``vendor`` filters routes; ``route``, if set, narrows to that
               exact route. Target model comes from the mandate.
            3. **Default route order** — all initialized routes, ordered per
               ``route_priority`` in ``kestrel.toml`` ``[llm]``.

        ``force_local_only=True`` additionally filters to local routes. If the
        resolved ``target_model`` isn't the configured default for any local
        route, it's cleared so each local route uses its own model.

        Routes that have already failed in this session with a permanent auth
        error (``self._disabled_routes``) are filtered out here so the
        fallback chain skips them immediately on the next user turn (#655).
        """
        providers_to_use = self._available_providers()
        target_model: Optional[str] = None

        # --- 1. Explicit model_override ---
        if model_override:
            if "/" in model_override:
                left, model_name = model_override.split("/", 1)
                target_model = model_name
                matching = self._filter_providers_by_selector(providers_to_use, left)
                if matching:
                    providers_to_use = matching
                else:
                    # If the requested route exists but was disabled earlier
                    # in this session (#655), say so specifically — "not
                    # available" misleads the caller into thinking it was
                    # never configured.
                    raw_matching = self._filter_providers_by_selector(
                        self.providers, left,
                    )
                    disabled_match = [
                        p["name"] for p in raw_matching
                        if p.get("name") in self._disabled_routes
                    ]
                    if disabled_match:
                        reasons = "; ".join(
                            f"{n}: {self._disabled_routes[n]}"
                            for n in disabled_match
                        )
                        raise LLMServiceError(
                            f"Route '{left}' was disabled earlier this "
                            f"session after a permanent auth failure "
                            f"({reasons}). Rotate the key and restart "
                            f"the service to re-enable."
                        )
                    raise LLMProviderUnavailableError(
                        left,
                        [p["name"] for p in self.providers],
                    )
            else:
                target_model = model_override

        # --- 2. Mandate preference (persisted agent preference) ---
        elif self._mandate_preference.get("model"):
            pref_model = self._mandate_preference["model"]
            pref_vendor = self._mandate_preference.get("vendor")
            pref_route = self._mandate_preference.get("route")
            target_model = pref_model

            if pref_vendor:
                selector = f"{pref_vendor}:{pref_route}" if pref_route else pref_vendor
                matching = self._filter_providers_by_selector(providers_to_use, selector)
                if matching:
                    providers_to_use = matching
                    logger.info(
                        "Provider routing: using mandated %s with model '%s'",
                        selector,
                        pref_model,
                    )
                elif self._mandate_fallbacks:
                    logger.warning(
                        "Mandated %s unavailable; using %d fallback(s)",
                        selector,
                        len(self._mandate_fallbacks),
                    )
                    fallback_providers = []
                    for fb in self._mandate_fallbacks:
                        fb_vendor = fb.get("vendor") or fb.get("provider")
                        if fb_vendor:
                            match_list = self._filter_providers_by_selector(
                                self.providers,
                                fb_vendor,
                            )
                            if match_list:
                                fallback_providers.append(match_list[0])
                    if fallback_providers:
                        providers_to_use = fallback_providers
                        target_model = self._mandate_fallbacks[0].get("model") or target_model
                else:
                    raise LLMProviderUnavailableError(
                        selector,
                        [p["name"] for p in self.providers],
                    )
            # else: model-only mandate — each route tries the model.

        # --- 3. force_local_only filter ---
        if force_local_only:
            providers_to_use = [p for p in providers_to_use if p.get("is_local")]
            if not providers_to_use:
                raise RuntimeError("No local providers available.")

            if target_model and not any(
                target_model == p["model"] for p in providers_to_use
            ):
                logger.info("LOCAL_ONLY: ignoring non-local model '%s', using each local route's configured model", target_model)
                target_model = None

        if not providers_to_use:
            # We never want to hand back an empty provider list. Streaming
            # paths iterate this as their fallback chain — zero providers
            # means the loop runs zero times and the only error available
            # at the end is ``last_error=None``, which surfaces as the
            # misleading message "All providers failed: None" with no clue
            # *why* nothing was tried. Decide between the two real reasons
            # here and raise something legible.
            if self._disabled_routes and not self._available_providers():
                reasons = "; ".join(
                    f"{n}: {r}" for n, r in self._disabled_routes.items()
                )
                raise LLMServiceError(
                    "No usable LLM routes — every initialized route was "
                    f"disabled this session after permanent auth failures "
                    f"({reasons}). Rotate keys and restart the service."
                )
            raise LLMServiceError(
                "No LLM routes are configured. Check kestrel.toml [llm] and "
                "vendor auth envs (e.g. ANTHROPIC_API_KEY, or "
                "ANTHROPIC_AUTH_TOKEN for the Claude OAuth/plan route, "
                "OPENAI_API_KEY)."
            )

        return providers_to_use, target_model

    def _model_available_for_route(self, provider: Dict[str, Any], model_id: str) -> bool:
        """Return True iff the model is discoverable in this route's vendor catalog.

        Uses the shared discovery cache. If discovery hasn't populated yet
        (cold-start before any list-models call), we permit the call rather
        than blocking every request — this only gates against *known* mismatches,
        not unknown state.

        The route's own configured model counts as available (so a route with
        a configured default still works before discovery confirms it).
        """
        if not model_id:
            return True
        # Route's own configured default is always considered available.
        if provider.get("model") == model_id:
            return True
        vendor = provider.get("vendor")
        if not vendor:
            return True  # Unknown-shape provider — can't validate, let it through.

        from .model_cache import get_shared_model_cache
        cache = get_shared_model_cache().get_any()
        if not cache:
            return True  # No discovery yet — don't block.

        for m in cache:
            if m.provider == vendor and m.id == model_id:
                return True
        return False

    @staticmethod
    def _filter_providers_by_selector(providers: List[Dict[str, Any]], selector: str) -> List[Dict[str, Any]]:
        """Filter route dicts by selector.

        Selector forms:
            "anthropic"          → all anthropic routes (vendor-only match).
            "anthropic:plan"     → exactly the anthropic:plan route.

        Matching is exact on vendor or composite route name.
        """
        if not selector:
            return []
        if ":" in selector:
            # Composite route key, exact match.
            return [p for p in providers if p.get("name") == selector]
        return [p for p in providers if p.get("vendor") == selector]

    def _convert_providers_format(self, provider_infos: List[ProviderInfo]) -> List[Dict[str, Any]]:
        """Flatten ProviderInfo list into the dict shape consumed by service.py.

        Each entry in ``self.providers`` is a **route** (vendor/route pair),
        not a traditional single-name provider. ``name`` is the composite
        ``"<vendor>:<route>"`` key; ``vendor`` carries the grouping dimension
        that discovery and UI buckets use.
        """
        out = []
        for provider in provider_infos:
            hints = getattr(provider, "selection_hints", None)
            try:
                hints = list(hints) if hints is not None else []
            except TypeError:
                hints = []
            out.append({
                "name": provider.name,
                "vendor": getattr(provider, "vendor", None),
                "route": getattr(provider, "route", None),
                "client": provider.client,
                "adapter": provider.adapter,
                "model": provider.model,
                "is_cloud": getattr(provider, "is_cloud", True),
                "is_local": getattr(provider, "is_local", False),
                "base_url": getattr(provider, "base_url", None),
                "selection_hints": hints,
            })
        return out

    def _initialize_providers(self) -> List[Dict[str, Any]]:
        """Initialize provider clients and adapters based on config file.

        This method is deprecated. Provider initialization is now handled by ProviderRegistry.
        This method is kept for backward compatibility and now delegates to the registry.
        """
        try:
            provider_infos = self.provider_registry.initialize_providers()
            return self._convert_providers_format(provider_infos)
        except ProviderInitializationError as e:
            logger.error(f"Failed to initialize providers: {e}")
            return []

    def _check_policy(self) -> None:
        """Guard called at the top of every public generation entry point.

        When `PayerPolicy.llm.kind == PayerKind.NONE`, the agent-init
        layer sets `self.disabled = True`. This method raises
        `PolicyDeniedError` in that case, preventing every documented
        generation method (`generate`, `get_response`, the streaming
        variants, `get_audit_response`, etc.) from silently falling
        through to a shared host key.

        Centralized here so adding a new generation entry point in the
        future means one line at the top of the new method instead of
        re-implementing the gate. The Phase 3b reflection test asserts
        every async-coroutine and async-generator method whose name
        matches a generation pattern calls this guard.
        """
        if getattr(self, "disabled", False):
            raise PolicyDeniedError(
                "LLMService is disabled by PayerPolicy "
                "(llm.kind = NONE). The agent has no LLM available; "
                "callers should treat this the same as 'no key configured'."
            )

    def attach_to_agent(self, agent_did: str) -> None:
        """Claim this LLMService instance for a specific agent.

        Required invariant for the PayerPolicy work: each KestrelAgent
        gets its own LLMService instance because `use_agent_key()`
        mutates `self.providers` in place. Sharing a service across
        agents would let the last-loaded agent silently steal every
        other agent's OpenRouter client.

        Production code already constructs one service per agent. This
        method enforces that contract at construction time so the
        invariant cannot silently regress (e.g. by a future test or a
        well-meaning refactor that reuses an instance).

        Args:
            agent_did: The agent's DID. Must be non-empty.

        Raises:
            ValueError: If agent_did is empty.
            LLMServiceAlreadyAttachedError: If this service is already
                attached to a different agent.

        Idempotent for repeated attach with the same DID, so a code path
        that re-enters agent init (re-anchor flows, test reset, etc.)
        does not double-fail.
        """
        if not agent_did:
            raise ValueError("agent_did is required for attach_to_agent")
        if self._owner_agent_did is None:
            self._owner_agent_did = agent_did
            return
        if self._owner_agent_did == agent_did:
            return
        raise LLMServiceAlreadyAttachedError(
            f"LLMService is already attached to agent {self._owner_agent_did[:30]}...; "
            f"cannot re-attach to {agent_did[:30]}.... Construct a fresh LLMService "
            "per agent (each agent's OpenRouter client mutation must be isolated)."
        )

    async def use_agent_key(
        self,
        agent_did: str,
        db: "AsyncDatabase",
        provider: str = "openrouter",
    ) -> bool:
        """
        Switch to using an agent's provisioned API key for a given vendor.

        This replaces the shared key with the agent's own key for billing isolation.
        The agent's key was created at inception and stored encrypted.

        Under the vendor/route/model schema, ``provider`` names a **vendor**
        (e.g. ``"openrouter"``); every route belonging to that vendor has its
        client swapped. Base URL is read from the route config or from the
        existing client so we don't need a legacy flat ``config[provider]``.

        Args:
            agent_did: The agent's DID
            db: Database connection for ServiceKeyStorage
            provider: Vendor name (default: ``"openrouter"``)

        Returns:
            True if key was activated, False if agent has no key or no
            matching routes are initialized.
        """
        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage, KeyNotConfiguredError

        try:
            key_storage = ServiceKeyStorage(db, agent_did)
            agent_key = await key_storage.get_key(provider_id=provider)
        except KeyNotConfiguredError:
            logger.debug(f"Agent {agent_did[:20]}... has no {provider} key, using shared key")
            return False

        # Find every route for this vendor and rebuild its client with the
        # agent key. Multiple routes per vendor is the whole point of the
        # refactor — don't stop after the first match.
        matched_any = False
        for p in self.providers:
            if p.get("vendor") != provider:
                continue
            # Pull base_url from the route (set by ProviderRegistry) or fall
            # back to the current client's base_url attribute for adapters
            # that don't carry it explicitly.
            base_url = p.get("base_url")
            if not base_url:
                existing = p.get("client")
                base_url = getattr(existing, "base_url", None)
                if base_url is not None:
                    base_url = str(base_url)
            if not base_url:
                logger.warning(
                    "No base_url for vendor %s route %s — skipping key swap",
                    provider, p.get("route"),
                )
                continue

            new_client = openai.AsyncOpenAI(
                api_key=agent_key, base_url=base_url, max_retries=0,
            )
            p["client"] = new_client
            # Keep the registry's ProviderInfo in sync so later code paths
            # that walk self.provider_registry.providers see the new client.
            for info in getattr(self.provider_registry, "providers", []):
                if getattr(info, "vendor", None) == provider \
                        and getattr(info, "route", None) == p.get("route"):
                    info.client = new_client
                    break
            matched_any = True

        if matched_any:
            logger.info(
                "Activated agent key for vendor %s (DID: %s...)",
                provider, agent_did[:20],
            )
            return True

        logger.warning("Vendor %s has no initialized routes — agent key not activated", provider)
        return False

    def _get_model_for_prompt(self, user_prompt: str) -> Optional[str]:
        """Determine best model based on user prompt and mandate rules."""
        mandates = self.mandate_config.get("mandates", {})
        prompt_lower = user_prompt.lower()

        for keyword, selector in mandates.items():
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, prompt_lower):
                if self._is_banned_selector(selector):
                    logger.warning(f"Mandate selector '{selector}' is banned. Ignoring.")
                    continue
                resolved = self._resolve_model_selector(selector)
                logger.info(f"Model mandate triggered by '{keyword}'. Using: {resolved['selector'] or selector}")
                return resolved["selector"] or selector

        default_selector = self._get_default_mandate_selector()
        if self._is_banned_selector(default_selector):
            logger.warning(f"Default mandate selector '{default_selector}' is banned. Ignoring.")
            return None
        resolved_default = self._resolve_model_selector(default_selector)
        return resolved_default["selector"] or default_selector

    # ==================== Observability Methods ====================

    def set_observability_store(self, store) -> None:
        """Set the observability store for logging LLM calls.

        Args:
            store: ObservabilityStore instance from kestrel_sovereign.a2a.stores.unified
        """
        self._observability_store = store
        logger.info("LLM observability enabled")

    def set_metering_callback(self, callback) -> None:
        """Set the metering callback for usage billing (Vending Machine).

        The callback will be called with:
            await callback(
                companion_id=str,
                user_id=str,
                provider=str,
                model=str,
                prompt_tokens=int,
                completion_tokens=int,
            )

        Args:
            callback: Async function to call after each LLM call
        """
        self._metering_callback = callback
        logger.info("LLM metering enabled")

    def set_observability_context(
        self,
        session_id: Optional[str] = None,
        companion_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """Set context for observability logging (called per-request).

        Args:
            session_id: A2A session ID
            companion_id: Companion UUID
            user_id: User UUID
        """
        self._observability_context = {
            "session_id": session_id,
            "companion_id": companion_id,
            "user_id": user_id,
        }

    @handle_observability_errors
    async def _log_llm_call(
        self,
        provider: str,
        model: str,
        duration_ms: int,
        success: bool,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        response: Optional[str] = None,
        error_message: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cache_creation_input_tokens: Optional[int] = None,
        cache_read_input_tokens: Optional[int] = None,
        tools_used: Optional[bool] = None,
        structured_output: Optional[bool] = None,
    ) -> None:
        """Log an LLM call to the observability store (if configured).

        This is called automatically by get_response() and generate().
        Also triggers metering callback for billing (Vending Machine).

        Also emits a single structured ``llm.usage:`` INFO line so callers
        that downcast the response to a plain string don't lose token /
        cache telemetry. Picked up by Cloud Run / Cloud Logging via the
        multi_agent stdout tee (issue #812). See issue #819.
        """
        # Log to observability store
        if self._observability_store:
            await self._observability_store.log_llm_call(
                provider=provider,
                model=model,
                duration_ms=duration_ms,
                success=success,
                agent_did=self._owner_agent_did,
                session_id=self._observability_context.get("session_id"),
                companion_id=self._observability_context.get("companion_id"),
                user_id=self._observability_context.get("user_id"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                error_message=error_message,
                tool_calls=tool_calls,
                metadata=metadata,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Prometheus metrics (no-op when prometheus-client not installed)
        from kestrel_sdk.metrics import (
            PROMETHEUS_AVAILABLE as _prom,
            LLM_CALLS,
            LLM_DURATION,
            LLM_TOKENS,
        )
        if _prom:
            LLM_CALLS.labels(
                provider=provider, model=model, success=str(success)
            ).inc()
            LLM_DURATION.labels(provider=provider, model=model).observe(
                duration_ms / 1000
            )
            if input_tokens is not None:
                LLM_TOKENS.labels(model=model, direction="input").inc(input_tokens)
            if output_tokens is not None:
                LLM_TOKENS.labels(model=model, direction="output").inc(output_tokens)

        # Trigger metering callback for billing (Phase 1: tracking only)
        if self._metering_callback and success:
            companion_id = self._observability_context.get("companion_id")
            user_id = self._observability_context.get("user_id")

            if companion_id and user_id:
                await self._metering_callback(
                    companion_id=companion_id,
                    user_id=user_id,
                    provider=provider,
                    model=model,
                    prompt_tokens=input_tokens or 0,
                    completion_tokens=output_tokens or 0,
                )

        # Wrap in try/except so a serialization edge case can never break
        # the call path. See issue #819.
        try:
            usage_log = {
                "provider": provider,
                "model": model,
                "duration_ms": duration_ms,
                "success": success,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
                "tools": tools_used,
                "structured_output": structured_output,
            }
            logger.info("llm.usage: %s", json.dumps(usage_log, default=str))
        except Exception as log_err:
            logger.warning("llm.usage log failed: %s", log_err)

    def get_cheap_model(self) -> Optional[str]:
        """
        Return a cheap/fast model id for recursive sub-queries (RLM-inspired).

        DEPRECATED return shape: a bare model id. Callers that injected this
        as ``model_override`` into the full fallback chain produced the
        "broadcast a bogus ID across every provider" bug. Prefer
        :meth:`get_cheap_model_selector` which returns a
        ``"<vendor>/<model>"`` selector that constrains routing to the one
        vendor that actually serves the model.

        Kept for backward compat; returns only the bare model portion.
        """
        selector = self.get_cheap_model_selector()
        if not selector:
            return None
        if "/" in selector:
            return selector.split("/", 1)[1]
        return selector

    def get_cheap_model_selector(self) -> Optional[str]:
        """Return a ``"<vendor>/<model>"`` cheap-model selector for sub-queries.

        Policy (config-driven, no hardcoded IDs):
          1. ``[defaults] cheap_model`` — explicit selector, honored as-is.
          2. ``[defaults] cheap_model_hints`` — pattern list; the first route
             whose configured model matches a hint wins. We return
             ``"<vendor>/<model>"`` so callers don't broadcast the raw ID
             across every provider — the selector constrains routing.
          3. Returns ``None`` when nothing matches; caller uses its default route.

        This is what the broadcast bug was really about: previous callers
        dropped the vendor context and injected one model id into every
        provider in the fallback chain, producing 4 garbage attempts on
        purpose and lying to downstream observability about which model ran.
        """
        defaults = self.mandate_config.get("defaults", {})

        # 1. Explicit selector.
        cheap_selector = defaults.get("cheap_model")
        if cheap_selector and cheap_selector != "auto":
            resolved = self._resolve_model_selector(cheap_selector)
            provider_key = resolved.get("provider")
            model = resolved.get("model")
            if provider_key and model:
                return f"{provider_key}/{model}"
            return resolved.get("selector") or cheap_selector

        # 2. Pattern-based resolution over routes.
        cheap_patterns = defaults.get("cheap_model_hints") or []
        if not cheap_patterns:
            return None

        cheap_providers = self.provider_registry.get_providers_with_pattern(cheap_patterns)
        if not cheap_providers:
            return None

        # First hit wins. Return vendor-scoped selector so routing can't
        # broadcast to providers that don't serve this model.
        match = cheap_providers[0]
        vendor = getattr(match, "vendor", None) or match.name.split(":", 1)[0]
        model = match.model
        if vendor and model and model != "auto":
            return f"{vendor}/{model}"
        return model if model and model != "auto" else None

    @handle_llm_errors()
    async def _try_single_provider(
        self,
        provider: Dict[str, Any],
        target_model: Optional[str],
        system_prompt: str,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Type[BaseModel]],
        force_local_only: bool,
        start_time: float
    ) -> Union[str, LLMResponse]:
        """Try to get a response from a single provider.

        Refuses to call a provider with a ``target_model`` not in that provider's
        vendor catalog. This is the guard that stops:
          * llama.cpp silent-override (llama-server ignores the model ID and
            serves whatever is loaded — so callers are lied to about which
            model produced the response),
          * the cheap-model cascade (one bogus model ID broadcast onto every
            provider in the fallback chain),
          * callers mistakenly targeting a vendor that doesn't serve the model.
        Skipping raises ``ModelNotAvailableForRoute`` so the outer fallback
        loop can move on to the next provider.
        """
        messages = messages_for(provider["adapter"], user_prompt=user_prompt, system_prompt=system_prompt)

        model_to_use = provider["model"]
        if target_model:
            if not self._model_available_for_route(provider, target_model):
                raise ModelNotAvailableForRoute(
                    vendor=provider.get("vendor"),
                    route=provider.get("route"),
                    model=target_model,
                )
            model_to_use = target_model

        response = await provider["adapter"].get_response(
            client=provider["client"],
            model=model_to_use,
            messages=messages,
            tools=tools,
            response_format=response_format,
            extra_body=provider_cache_body(provider),
        )

        # Calculate duration and log to observability
        duration_ms = int((time.time() - start_time) * 1000)
        response_text = response.content if isinstance(response, LLMResponse) else str(response)
        tool_calls_data = None
        if isinstance(response, LLMResponse):
            if response.tool_calls:
                tool_calls_data = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ]
            else:
                # Inline-executing adapters (codex app-server) populate
                # ``executed_tool_calls`` instead of ``tool_calls`` —
                # the LLM call still used tools, so the audit log
                # should reflect them.
                executed = getattr(response, "executed_tool_calls", None)
                if executed:
                    tool_calls_data = [
                        {"name": e["name"], "arguments": e["arguments"]}
                        for e in executed
                    ]

        # Extract token counts from response for billing
        input_tokens = None
        output_tokens = None
        cache_creation_input_tokens = None
        cache_read_input_tokens = None
        if isinstance(response, LLMResponse):
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            cache_creation_input_tokens = response.cache_creation_input_tokens
            cache_read_input_tokens = response.cache_read_input_tokens

        # Track model usage with token count
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        await self._track_model_usage(model_to_use, provider["name"], tokens=total_tokens)

        await self._log_llm_call(
            provider=provider["name"],
            model=model_to_use,
            duration_ms=duration_ms,
            success=True,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response_text,
            tool_calls=tool_calls_data,
            metadata={"force_local_only": force_local_only},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            tools_used=tools is not None,
            structured_output=response_format is not None,
        )

        # Return full LLMResponse if tools or structured output requested
        if tools is not None or response_format is not None:
            return response
        else:
            if isinstance(response, LLMResponse):
                return response.content or ""
            return response

    async def get_audit_response(self, text_to_audit: str) -> Dict[str, Any]:
        """Get a structured audit response from the normal provider chain."""
        self._check_policy()
        if not self.providers:
            return {"risk_level": 1, "reasoning": "Audit skipped - no providers available."}

        target_selector = self._get_default_mandate_selector()
        if not target_selector:
            pref_model = self._mandate_preference.get("model")
            pref_vendor = self._mandate_preference.get("vendor")
            pref_route = self._mandate_preference.get("route")
            if pref_model:
                if pref_vendor and pref_route:
                    target_selector = f"{pref_vendor}:{pref_route}/{pref_model}"
                elif pref_vendor:
                    target_selector = f"{pref_vendor}/{pref_model}"
                else:
                    target_selector = pref_model

        available_providers = self._available_providers()
        target_model = None
        if target_selector:
            resolved = self._resolve_model_selector(target_selector, providers=available_providers)
            target_provider = resolved.get("provider")
            target_model = resolved.get("model")
            if target_provider:
                # target_provider may be a vendor or a composite "vendor:route" key.
                available_providers = self._filter_providers_by_selector(
                    available_providers, target_provider
                ) or available_providers

        system_prompt = """
You are an AI Integrity Auditor for a Kestrel agent's responses.

Your job is to check if the RESPONSE ITSELF is problematic:
- Is it misleading, deceptive, or factually incorrect?
- Does it attempt to manipulate the user?
- Does it contain harmful instructions or content?
- Is it incoherent or corrupted text?

Do NOT evaluate the system/architecture being described. An agent explaining its own design is normal.

Respond with a JSON object containing:
1. "risk_level": Integer 1-3 (1=low/normal response, 2=medium/some concern, 3=high/block response)
2. "reasoning": One-sentence explanation

Most normal, helpful responses should be risk_level 1.
No other text or formatting.
"""

        try:
            errors = {}
            for provider in available_providers:
                logger.info(f"Auditing with provider: {provider['name']}")
                effective_model = target_model or provider["model"]
                if target_model and not self._model_available_for_route(provider, target_model):
                    logger.debug(
                        "Audit: skipping %s (target model %s not in vendor catalog)",
                        provider["name"], target_model,
                    )
                    continue
                messages = messages_for(
                    provider["adapter"],
                    user_prompt=text_to_audit,
                    system_prompt=system_prompt,
                )

                try:
                    response = await provider["adapter"].get_response(
                        client=provider["client"],
                        model=effective_model,
                        messages=messages,
                        format="json",
                    )
                    response_json = json.loads(response.content)
                    if "risk_level" not in response_json or "reasoning" not in response_json:
                        raise ValueError("Missing required keys in audit response.")
                    return response_json
                except (LLMProviderError, openai.APIError, openai.APIConnectionError, httpx.HTTPError, ConnectionError, TimeoutError) as exc:
                    errors[provider["name"]] = str(exc)
                    logger.warning(f"Audit provider {provider['name']} failed: {exc}")
                    continue

            if errors:
                joined = "; ".join(f"{name}: {error}" for name, error in errors.items())
                return {"risk_level": 3, "reasoning": f"Audit provider failed: {joined}"}
            return {"risk_level": 1, "reasoning": "Audit skipped - no providers available."}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse audit JSON: {e}")
            return {"risk_level": 3, "reasoning": "Audit model returned malformed JSON."}
        except LLMProviderError as e:
            logger.error(f"Audit provider failed: {e}")
            return {"risk_level": 3, "reasoning": f"Audit provider failed: {e}"}
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"Data validation error in audit: {e}", exc_info=True)
            return {"risk_level": 3, "reasoning": f"Audit failed: {e}"}
        except (openai.APIError, openai.APIConnectionError, httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Network/API error in audit: {e}", exc_info=True)
            return {"risk_level": 3, "reasoning": f"Audit failed: {e}"}
        except Exception as e:
            logger.error(f"Unexpected audit error: {e}", exc_info=True)
            return {"risk_level": 3, "reasoning": f"Audit failed: {e}"}

    async def get_response(
        self,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None
    ) -> Union[str, LLMResponse]:
        """Get a response from providers in priority order.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers (Ollama)
            model_override: Override the model selection
            tools: Optional tools for function calling
            response_format: Optional Pydantic model for structured output

        Returns:
            String content or LLMResponse (if tools provided or structured output)
        """
        self._check_policy()
        start_time = time.time()

        if not self.providers:
            raise RuntimeError("No LLM providers initialized.")

        # Use mandate-aware model from prompt if no explicit override
        effective_override = model_override if model_override else self._get_model_for_prompt(user_prompt)

        available_providers, target_model = self.resolve_provider_routing(
            model_override=effective_override,
            force_local_only=force_local_only,
        )

        # Strip tools if the target model can't handle them
        tools = self._check_model_tool_support(available_providers, tools, model_override)

        errors = {}
        for provider in available_providers:
            try:
                provider_name = provider['name']
                logger.info(f"Attempting provider: {provider_name}")

                with optional_span("agent.llm_call", {
                    "llm.method": "get_response",
                    "llm.provider": provider_name,
                    "llm.model": target_model or provider.get("model", ""),
                }) as llm_span:
                    result = await self._try_single_provider(
                        provider=provider,
                        target_model=target_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        tools=tools,
                        response_format=response_format,
                        force_local_only=force_local_only,
                        start_time=start_time
                    )
                    if llm_span and isinstance(result, LLMResponse):
                        if result.input_tokens is not None:
                            llm_span.set_attribute("llm.usage.prompt_tokens", result.input_tokens)
                        if result.output_tokens is not None:
                            llm_span.set_attribute("llm.usage.completion_tokens", result.output_tokens)
                        if result.total_tokens is not None:
                            llm_span.set_attribute("llm.usage.total_tokens", result.total_tokens)

                logger.info(f"Success from {provider_name}")
                return result

            except ModelNotAvailableForRoute as e:
                # Route can't serve the target model. Skip silently — no HTTP
                # call was made — and try the next provider.
                logger.debug(
                    "Skipping %s: %s not in vendor catalog",
                    provider["name"], e.model,
                )
                errors[provider["name"]] = e
                continue

            except LLMProviderError as e:
                logger.warning(f"Provider {provider['name']} failed: {e}")
                errors[provider['name']] = e

                # Log failed attempt
                duration_ms = int((time.time() - start_time) * 1000)
                await self._log_llm_call(
                    provider=provider["name"],
                    model=provider["model"],
                    duration_ms=duration_ms,
                    success=False,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    error_message=str(e),
                )

        provider_type = "local" if force_local_only else "all"
        raise LLMAllProvidersFailedError(errors)

    async def get_response_with_model(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        auto_pull: bool = True
    ) -> str:
        """Get a response using a specific model."""
        self._check_policy()
        provider_for_model = None
        for provider in self.providers:
            if provider["model"] == model_id or model_id in provider["model"]:
                provider_for_model = provider
                break

        if not provider_for_model:
            if ":" in model_id and auto_pull:
                logger.info(f"Model '{model_id}' not found. Attempting auto-pull...")
                try:
                    await self.pull_model(model_id, auto_confirm=True)
                    for provider in self.providers:
                        if provider.get("vendor") == "ollama":
                            provider_for_model = provider
                            break
                except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                    logger.error(f"Auto-pull failed: {e}", exc_info=True)
                    raise ValueError(f"Model '{model_id}' not found and auto-pull failed: {e}")
                except (openai.APIError, httpx.HTTPError) as e:
                    logger.error(f"Auto-pull network error: {e}", exc_info=True)
                    raise ValueError(f"Model '{model_id}' not found and auto-pull failed: {e}")
                except Exception as e:
                    logger.error(f"Auto-pull failed: {e}", exc_info=True)
                    raise ValueError(f"Model '{model_id}' not found and auto-pull failed: {e}")

            if not provider_for_model:
                available = [p["model"] for p in self.providers]
                raise ValueError(f"Model '{model_id}' not found. Available: {', '.join(available)}")

        try:
            logger.info(f"Getting response from model: {model_id}")
            messages = messages_for(provider_for_model["adapter"], user_prompt=user_prompt, system_prompt=system_prompt)

            response = await provider_for_model["adapter"].get_response(
                client=provider_for_model["client"],
                model=model_id,
                messages=messages,
                extra_body=provider_cache_body(provider_for_model),
            )

            # Track model usage with token count
            total_tokens = 0
            if isinstance(response, LLMResponse):
                total_tokens = (response.input_tokens or 0) + (response.output_tokens or 0)
            await self._track_model_usage(model_id, provider_for_model["name"], tokens=total_tokens)
            logger.info(f"Success from {model_id}")
            return response

        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.AuthenticationError) as e:
            logger.error(f"Model {model_id} API error: {e}", exc_info=True)
            raise RuntimeError(f"Model {model_id} failed: {e}")
        except (httpx.HTTPError, ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"Model {model_id} network error: {e}", exc_info=True)
            raise RuntimeError(f"Model {model_id} failed: {e}")
        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Model {model_id} data error: {e}", exc_info=True)
            raise RuntimeError(f"Model {model_id} failed: {e}")
        except Exception as e:
            logger.error(f"Model {model_id} failed: {e}", exc_info=True)
            raise RuntimeError(f"Model {model_id} failed: {e}")

    # get_streaming_response is provided by StreamingMixin

    async def close(self):
        """Close all async HTTP clients properly."""
        await self.drain_preference_persistence()

        for provider in self.providers:
            # Adapter-owned resources (e.g. CodexAdapter's app-server
            # subprocess) — adapters that own external state should
            # expose ``aclose``. The provider's ``client`` slot doesn't
            # always carry that state (codex stores just the binary
            # path), so consult the adapter directly.
            adapter = provider.get("adapter")
            if adapter is not None and hasattr(adapter, "aclose"):
                try:
                    await _wait_for_close_result(adapter.aclose())
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    logger.warning(
                        "Error closing %s adapter: %s",
                        provider.get("name"), e, exc_info=True,
                    )

            client = provider.get("client")
            if client is None:
                continue

            try:
                if hasattr(client, "close") and callable(client.close):
                    # Wrap in shield and timeout to handle cancellation gracefully
                    try:
                        await _wait_for_close_result(client.close())
                    except asyncio.TimeoutError:
                        logger.debug(f"Timeout closing {provider.get('name')} client")
                    except asyncio.CancelledError:
                        logger.debug(f"Cancelled while closing {provider.get('name')} client")
                elif hasattr(client, "_client") and hasattr(client._client, "aclose"):
                    try:
                        await _wait_for_close_result(client._client.aclose())
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
            except (ConnectionError, OSError) as e:
                logger.debug(f"Connection error closing {provider.get('name')} client: {e}")
            except (RuntimeError, AttributeError) as e:
                logger.warning(f"Error closing {provider.get('name')} client: {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"Unexpected error closing {provider.get('name')} client: {e}", exc_info=True)

        # Close remote GPU client if active
        if self._remote_client:
            try:
                await asyncio.wait_for(self._remote_client.close(), timeout=CLIENT_CLOSE_TIMEOUT)
            except (asyncio.TimeoutError, asyncio.CancelledError, ConnectionError, OSError):
                pass
            except (RuntimeError, AttributeError) as e:
                logger.debug(f"Error closing remote client: {e}", exc_info=True)
            except Exception as e:
                logger.debug(f"Error closing remote client: {e}", exc_info=True)
            finally:
                self._remote_client = None

        # Close the async usage tracking database
        try:
            await self.close_usage_db()
        except asyncio.CancelledError:
            logger.debug("Cancelled while closing usage DB")

    # Remote GPU Backend Methods are provided by RemoteBackendMixin

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, LLMResponse]:
        """Generate text using the active backend with automatic fallback.

        This is the primary generation method that handles:
        - Remote GPU backends (if active)
        - Cloud/local provider fallback
        - Tool calling support
        - Structured output via Pydantic models

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User message
            force_local_only: Only use local providers
            model_override: Override model selection
            tools: Optional tools for function calling
            response_format: Optional Pydantic model for structured output

        Returns:
            String content or LLMResponse (if tools/structured output)
        """
        self._check_policy()
        with optional_span("agent.llm_call", {
            "llm.method": "generate",
            "llm.model": model_override or "",
            "llm.force_local_only": force_local_only,
            "llm.has_tools": tools is not None,
        }) as llm_span:
            # Try remote GPU first when active AND routing isn't pinned.
            # A persisted mandate or vendor-prefixed model_override must
            # beat the remote shortcut — see #734 and _remote_first_allowed.
            if (
                self._backend == BackendType.REMOTE_GPU
                and self._remote_client
                and not force_local_only
                and self._remote_first_allowed(model_override)
            ):
                try:
                    self._ensure_remote_active()
                    messages = messages_for(self._remote_adapter, user_prompt=user_prompt, system_prompt=system_prompt)
                    model = model_override or self._remote_config.model
                    response = await self._remote_adapter.get_response(
                        client=self._remote_client,
                        model=model,
                        messages=messages,
                        tools=tools,
                        response_format=response_format,
                    )
                    if llm_span:
                        llm_span.set_attribute("llm.provider", "remote_gpu")
                        llm_span.set_attribute("llm.model_used", model)
                    if tools is not None or response_format is not None:
                        return response
                    if isinstance(response, LLMResponse):
                        return response.content or ""
                    return response
                except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.AuthenticationError) as exc:
                    self._last_remote_error = str(exc)
                    logger.warning(f"Remote GPU API error: {exc}, falling back to providers", exc_info=True)
                    self._deactivate_remote_backend(reason=str(exc))
                except (httpx.HTTPError, ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
                    self._last_remote_error = str(exc)
                    logger.warning(f"Remote GPU network error: {exc}, falling back to providers", exc_info=True)
                    self._deactivate_remote_backend(reason=str(exc))
                except LLMServiceError as exc:
                    self._last_remote_error = str(exc)
                    logger.warning(f"Remote GPU service error: {exc}, falling back to providers", exc_info=True)
                    self._deactivate_remote_backend(reason=str(exc))
                except Exception as exc:
                    self._last_remote_error = str(exc)
                    logger.warning(f"Remote GPU failed: {exc}, falling back to providers", exc_info=True)
                    self._deactivate_remote_backend(reason=str(exc))

            # Fall back to standard provider chain
            return await self.get_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                force_local_only=force_local_only,
                model_override=model_override,
                tools=tools,
                response_format=response_format,
            )

    async def generate_with_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        session_id: Optional[str] = None,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
    ) -> Union[str, LLMResponse]:
        """Generate using existing message list (for multi-turn tool calling).

        Args:
            messages: Pre-built message list
            tools: Optional tools for function calling
            response_format: Optional Pydantic model for structured output
            force_local_only: Only use local providers
            model_override: Override model selection
            session_id: Stable id of the multi-turn conversation. When set,
                stateful providers (e.g. CodexAdapter) anchor on the prior
                response via ``previous_response_id`` and send only delta input,
                preserving encrypted reasoning across turns. Stateless adapters
                ignore it. See #808.

        Returns:
            String content or LLMResponse
        """
        self._check_policy()
        # Try remote GPU first when active AND routing isn't pinned — #734.
        if (
            self._backend == BackendType.REMOTE_GPU
            and self._remote_client
            and not force_local_only
            and self._remote_first_allowed(model_override)
        ):
            try:
                self._ensure_remote_active()
                model = model_override or self._remote_config.model
                response = await self._remote_adapter.get_response(
                    client=self._remote_client,
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                )
                if tools is not None or response_format is not None:
                    return response
                if isinstance(response, LLMResponse):
                    return response.content or ""
                return response
            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.AuthenticationError) as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU API error: {exc}, falling back", exc_info=True)
                self._deactivate_remote_backend(reason=str(exc))
            except (httpx.HTTPError, ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU network error: {exc}, falling back", exc_info=True)
                self._deactivate_remote_backend(reason=str(exc))
            except LLMServiceError as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU service error: {exc}, falling back", exc_info=True)
                self._deactivate_remote_backend(reason=str(exc))
            except Exception as exc:
                self._last_remote_error = str(exc)
                logger.warning(f"Remote GPU failed: {exc}, falling back", exc_info=True)
                self._deactivate_remote_backend(reason=str(exc))

        # Fall back to standard providers (skip any disabled by auth failure).
        providers = self._available_providers()

        # Lazy auto-resolution. Routes are seeded with ``model = "auto"`` in
        # ``kestrel.toml`` and only get resolved to a real model id when the
        # disk cache is populated (``_load_from_disk_cache`` runs at __init__)
        # or when the model-picker UI calls ``discover_all_models`` explicitly.
        # On a fresh ``--quickstart`` setup neither has happened yet, so the
        # very first chat call would otherwise hit the adapter with the
        # literal string ``"auto"`` — Ollama returns 404, which the SDK turns
        # into either a fast error (recent ollama) or an indefinite hang
        # (older client/server combos). Trigger discovery here, once, on
        # demand. ``use_cache=True`` makes subsequent calls a cache hit.
        if any(p.get("model") == "auto" for p in providers):
            try:
                await self.discover_all_models(use_cache=True)
            except Exception as exc:
                logger.warning(
                    "Lazy model discovery failed in generate_with_messages "
                    "(continuing with provider['model'] as-is): %s", exc,
                )
            providers = self._available_providers()

        if force_local_only:
            providers = [p for p in providers if p.get("is_local")]
            # Clear any cloud model override — use the local provider's own model
            if model_override and providers and not any(
                model_override == p["model"] for p in providers
            ):
                model_override = None

        # Strip tools if the target model can't handle them
        tools = self._check_model_tool_support(providers, tools, model_override)

        target_selector = model_override
        if not target_selector:
            pref_model = self._mandate_preference.get("model")
            pref_vendor = self._mandate_preference.get("vendor")
            pref_route = self._mandate_preference.get("route")
            if pref_model:
                if pref_vendor and pref_route:
                    target_selector = f"{pref_vendor}:{pref_route}/{pref_model}"
                elif pref_vendor:
                    target_selector = f"{pref_vendor}/{pref_model}"
                else:
                    target_selector = pref_model

        target_model = None
        if target_selector:
            resolved = self._resolve_model_selector(target_selector, providers=providers)
            target_provider = resolved.get("provider")
            target_model = resolved.get("model")
            if target_provider:
                matched = self._filter_providers_by_selector(providers, target_provider)
                if matched:
                    providers = matched
                    logger.info(f"Model mandate: using {target_provider} with {target_model}")
                elif model_override:
                    # Same disabled-route helpful error as in
                    # resolve_provider_routing — if the requested target
                    # matches a route that was disabled this session,
                    # tell the caller to rotate the key and restart.
                    raw_matching = self._filter_providers_by_selector(
                        self.providers, target_provider,
                    )
                    disabled_match = [
                        p["name"] for p in raw_matching
                        if p.get("name") in self._disabled_routes
                    ]
                    if disabled_match:
                        reasons = "; ".join(
                            f"{n}: {self._disabled_routes[n]}"
                            for n in disabled_match
                        )
                        raise LLMServiceError(
                            f"Route '{target_provider}' was disabled "
                            f"earlier this session after a permanent "
                            f"auth failure ({reasons}). Rotate the key "
                            f"and restart the service to re-enable."
                        )
                    raise LLMServiceError(
                        f"Route/vendor '{target_provider}' not available. "
                        f"Available: {[p['name'] for p in providers]}"
                    )

        # Same no-silent-fallback rule as the streaming paths: if routing
        # was narrowed to one provider (by mandate, route, or override),
        # failure raises with the *specific* provider+error. No cascade to
        # an unrelated backend. Never hand the caller a response from a
        # model they didn't ask for.
        mandate_restricted = len(providers) == 1
        last_error = None
        last_provider_name = None
        for provider in providers:
            last_provider_name = provider["name"]
            try:
                model = target_model or provider["model"]
                logger.info(f"Attempting provider: {provider['name']} with model: {model}")
                response = await provider["adapter"].get_response(
                    client=provider["client"],
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    extra_body=provider_cache_body(provider),
                    session_id=session_id,
                    tool_executor=tool_executor,
                )
                if tools is not None or response_format is not None:
                    return response
                if isinstance(response, LLMResponse):
                    return response.content or ""
                return response
            except openai.BadRequestError as e:
                # 400 = request problem (context too big, bad format, etc.)
                # Don't fall back — the request itself is broken, not the provider.
                logger.error(f"Provider {provider['name']} rejected request (400): {e}")
                raise LLMServiceError(f"Request rejected by {provider['name']}: {e}") from e
            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                self._maybe_disable_route(provider, e)
                last_error = e
                if mandate_restricted:
                    raise LLMServiceError(
                        f"Selected route {provider['name']} failed: {e}"
                    ) from e
                logger.warning(
                    "Falling through from %s in generate_with_messages: %s",
                    provider["name"], e,
                )
                continue

        raise LLMServiceError(
            f"All providers failed for generate_with_messages "
            f"(last: {last_provider_name}): {last_error}"
        )

    # generate_stream, stream_with_messages, and stream_with_tool_detection
    # are provided by StreamingMixin

    # _ensure_remote_active is provided by RemoteBackendMixin
