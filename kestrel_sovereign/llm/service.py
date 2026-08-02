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
import time
import asyncio
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from kestrel_sovereign.kestrel_config.constants import STORAGE_CACHE_TTL_SECONDS
from typing import Awaitable, Callable, List, Dict, Any, Optional, Union, Type, TYPE_CHECKING

import openai
import httpx
from kestrel_sdk.llm import ProviderCapabilities

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from .embedding_space import EmbeddingSpacePin, ParityResult
from dotenv import load_dotenv
from pydantic import BaseModel

from .provider_registry import (
    ProviderRegistry,
    ProviderInfo,
    ProviderInitializationError,
    provider_cache_body,
)
from .cancellation import CancelToken
from .error_handling import (
    handle_llm_errors,
    LLMError,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMAllProvidersFailedError
)
from .openai_adapter import OpenAIAdapter
from .adapter import LLMResponse, messages_for, response_usage_available
from .model_discovery import ModelDiscoveryMixin
from .mandate import ModelMandateMixin
from .usage_tracking import UsageTrackingMixin
from .streaming import StreamingMixin, RoutingResolution
from .invocation_context import (
    LLMInvocationContext,
    LLMInvocationContextState,
    resolve_invocation_context,
)
from .constitutional_awareness import ConstitutionalAwarenessMixin
from .remote_backend import BackendType, RemoteBackendMixin
from kestrel_sovereign.kestrel_config.constants import (
    CLIENT_CLOSE_TIMEOUT,
)
from kestrel_sovereign.config import load_config, load_section
from kestrel_sovereign import telemetry

logger = logging.getLogger(__name__)

# Vendors whose server ignores the requested model ID and serves whatever
# weights are loaded. Explicit-route callers must still catalog-validate on
# these, otherwise the response is silently metered as the requested model
# even though a different one produced it (codex round-2 P2 on #2352).
_MODEL_IGNORING_VENDORS = frozenset({"llama_cpp", "ollama"})


class EmbeddingSpaceConflictError(Exception):
    """A route pin would fragment a VERIFIED shared embedding space (#2440).

    Raised at set time when an operator tries to pin a route to a model/dim that
    differs from the verified shared space the route is a member of. Kept
    distinct from ``ValueError`` so the endpoint can map it to a 409 (conflict)
    rather than the 400 a malformed request gets.
    """


class AuditResult(BaseModel):
    """Structured result of a response-integrity audit.

    Requested via ``response_format`` so structured output is honored across
    adapters (Anthropic via its tool pattern, OpenAI natively) rather than the
    OpenAI-style ``format="json"`` string the Anthropic adapter ignores (#2032).
    """

    risk_level: int
    reasoning: str


async def _wait_for_close_result(result: Any) -> None:
    """Await asynchronous close results while accepting synchronous close APIs."""
    if inspect.isawaitable(result):
        await asyncio.wait_for(asyncio.shield(result), timeout=CLIENT_CLOSE_TIMEOUT)


def _redacted_content_marker(text: Any) -> str:
    """Length-tagged placeholder for prompt/response text withheld from
    telemetry under an enforcing response audit (#2674 finding 3).

    Keeps a content-FREE size hint (useful for spotting truncation / empty
    turns) without exposing a single character of the withheld prose. Used for
    both durable ``llm_calls`` columns and the OpenTelemetry LLM span values so
    the two sinks redact identically.
    """
    try:
        n = len(text) if text is not None else 0
    except TypeError:
        n = 0
    return f"[redacted: {n} chars withheld pending response audit]"


def _redact_tool_calls_content(
    tool_calls: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Blank model-generated tool-call ARGUMENTS for durable telemetry (#2674 finding 4).

    ``tool_calls`` is response-derived model output whose ``arguments`` can echo
    the withheld assistant prose. Under an enforcing response audit the durable
    ``llm_calls`` row must not carry that raw content before (or, on a DENY, ever
    after) the verdict. Keep each call's ``name`` — a safe classification of which
    tools the model invoked — and replace ``arguments`` with a content-free
    marker; drop no entries so the count/shape stays intact. ``None`` / empty
    passes through unchanged.
    """
    if not tool_calls:
        return tool_calls
    redacted: List[Dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            safe = {k: v for k, v in call.items() if k not in ("arguments", "input")}
            if "arguments" in call:
                safe["arguments"] = _redacted_content_marker(call.get("arguments"))
            if "input" in call:
                safe["input"] = _redacted_content_marker(call.get("input"))
            redacted.append(safe)
        else:
            redacted.append({"arguments": _redacted_content_marker(call)})
    return redacted


def _safe_error_label(error: Any) -> str:
    """Content-free label for an exception: its class name (#2674 finding 4).

    Provider/adapter exception *messages* routinely embed the prompt or the
    response body, so under an enforcing audit only the exception *class* — a
    safe category the operator can still triage on — may reach durable telemetry
    or an exported span. A plain string (no ``__class__`` beyond ``str``) yields
    ``"str"``; ``None`` yields ``""``.
    """
    if error is None:
        return ""
    if isinstance(error, BaseException):
        return type(error).__name__
    return type(error).__name__


def _redact_error_content(error_message: Optional[str]) -> Optional[str]:
    """Reduce a stored ``error_message`` string to a content-free class marker.

    The failure paths pass ``str(error)`` as ``error_message`` (already a string
    by the time it reaches ``_log_llm_call``), so the original class is gone; tag
    it as redacted with its length so truncation/empty is still visible without
    exposing a character of the possibly prompt/response-bearing text. ``None``
    passes through so a successful call still records no error.
    """
    if error_message is None:
        return None
    return f"[redacted error: {len(error_message)} chars withheld pending response audit]"


def _client_timeout_seconds(timeout: Any) -> Optional[float]:
    """Normalize an OpenAI/httpx client timeout to a number of seconds (#1966).

    Accepts a plain float (what we set for local routes), an ``httpx.Timeout``
    (the SDK default, which carries per-phase ``read``/``connect``/… values), or
    None. For a long generation the ``read`` phase is what matters, so we take
    the largest finite phase value. Returns None when nothing numeric is found.
    """
    if timeout is None:
        return None
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [
        getattr(timeout, attr, None)
        for attr in ("read", "timeout", "pool", "write", "connect")
    ]
    nums = [float(c) for c in candidates if isinstance(c, (int, float))]
    return max(nums) if nums else None


def resolve_active_model_selection(llm_service) -> Dict[str, Optional[str]]:
    """Resolve canonical current-selection metadata for any LLM-service-like object.

    Returns a dict with keys:
        vendor:     e.g. ``"openai"`` — None only if no routes are configured.
        route:      e.g. ``"api"`` — None when the mandate specifies only a vendor.
        model_name: the model ID.
        model:      display form ``"<vendor>/<model_name>"`` or ``"<vendor>:<route>/<model_name>"``.
        is_auto:    True when no model is pinned in the mandate, so ``model_name``
                    is an auto-resolved default that can silently change as the
                    provider lineup shifts (#2419). Callers surface this so an
                    unchosen model change reads as auto-drift, not a set setting.
    """
    pref = llm_service.get_model_preference() or {}
    model_name = pref.get("model")
    vendor = pref.get("vendor")
    route = pref.get("route")
    # Auto = the operator never pinned a model; routing picks the default for
    # whichever route runs. A pinned mandate carries a truthy ``model``.
    is_auto = not model_name

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
        "is_auto": is_auto,
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

    # Class-level default so tests that construct via ``__new__`` (bypassing
    # ``__init__``) still see ``None`` rather than ``AttributeError`` when
    # ``_current_force_local_only`` reads it. Production paths get an
    # instance-level value via ``__init__`` (set to ``None``) and update it
    # through :meth:`set_force_local_only_provider` (#1492).
    _force_local_only_provider: Optional[Callable[[], bool]] = None

    def __init__(
        self,
        database_url: Optional[str] = None,
        agent_data_dir: Optional[Any] = None,
    ):
        """Initialize LLM service.

        Reads LLM configuration from the ``[llm]`` section of ``kestrel.toml``
        in the current project. The legacy standalone ``llm_config.toml``
        is no longer supported — run ``kestrel migrate-llm-config`` to fold
        an old file in.

        Args:
            database_url: Optional PostgreSQL connection URL for usage tracking.
                         If provided, uses PostgreSQL. Otherwise checks env vars,
                         then falls back to SQLite.
            agent_data_dir: The owning agent's data root. Multi-agent callers
                         MUST pass this: one process hosts several agents, so
                         the process environment cannot name each agent's data
                         root and SQLite usage rows would otherwise all land in
                         one agent's database (#2769).
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

        # Human display name of the owning agent (``agent.agent_name``, set at
        # registration). Populated via ``set_agent_display_name`` so LLM-call
        # spans can carry ``kestrel.agent_name`` (issue #2573). Best-effort:
        # None until the registrar wires it.
        self._agent_display_name: Optional[str] = None

        # PayerPolicy NONE flag. Set to True by the agent-init layer when
        # the agent's policy slot for LLM is `PayerKind.NONE`. Phase 3b
        # adds the `_check_policy()` guard on every generation entry point
        # that raises PolicyDeniedError when this is True. Phase 3a only
        # plumbs the flag; generation calls still go through (consistent
        # with HOST_ENV behavior) until 3b lands the guard.
        self.disabled: bool = False

        # Database for model usage tracking (uses abstract data layer)
        self._init_usage_tracking(database_url, agent_data_dir=agent_data_dir)

        # Constitutional profile service
        self._init_constitutional_profiles()

        # Runtime mandate state
        # Mandate preference schema: {"vendor": str|None, "model": str|None, "route": str|None}.
        # vendor + model are the primary selectors; route is optional and narrows
        # to an exact (vendor, route) pair. Stale rows using the old {"model", "provider"}
        # shape are dropped by model_preference._load_model_preference().
        self._mandate_preference = {"vendor": None, "model": None, "route": None}
        self._mandate_fallbacks = []

        # Top-level embedding-route knob (#2263). ``[llm] embedding_route =
        # "<vendor>:<route>"`` selects the embedding channel independently of
        # the chat route — one setting instead of repeating ``embedding_sibling``
        # under every vendor. ``None`` means "auto / follow chat route" (the
        # pre-#2263 default). The config value seeds the runtime state; a
        # persisted runtime override (agent_metadata) is applied on startup and
        # wins over config, mirroring how model preference persists.
        configured_embedding_route = None
        if isinstance(self.config, dict):
            raw_embedding_route = self.config.get("embedding_route")
            if raw_embedding_route:
                configured_embedding_route = str(raw_embedding_route).strip() or None
        # Same normalization as set_embedding_route: the documented
        # ``embedding_route = "auto"`` means "follow chat" — storing the
        # literal would make resolve treat it as an explicit (nonexistent)
        # provider and silently keyword-fallback (codex P2 on #2270).
        # ``"none"`` is the deliberate off-switch (#2287) — canonicalize its
        # casing so resolve's step-0 short-circuit recognizes it.
        if configured_embedding_route:
            if configured_embedding_route.lower() == "auto":
                configured_embedding_route = None
            elif configured_embedding_route.lower() == "none":
                configured_embedding_route = "none"
        self._embedding_route: Optional[str] = configured_embedding_route

        # Shared local/cloud embedding-space pins (#2290). ``[llm.embedding_spaces]``
        # declares open-weight models served on BOTH a local and a cloud route so
        # their rows share ONE model-identity space (``<model>@<dim>``) instead of
        # fracturing by serving route. A pin is only APPLIED after its parity
        # probe passes (see ``verify_embedding_space_parity``); until then members
        # keep their own route-scoped space ids. Parse defensively so a malformed
        # declaration degrades to "no shared space" rather than crashing init.
        self._embedding_space_pins: List["EmbeddingSpacePin"] = []
        try:
            from .embedding_space import parse_embedding_space_pins

            self._embedding_space_pins = parse_embedding_space_pins(self.config)
        except Exception as exc:
            logger.error(
                "Invalid [llm.embedding_spaces] config; shared embedding spaces "
                "are DISABLED until fixed: %s",
                exc,
            )
        # Parity-probe results keyed by pin name; only a pin present here with
        # ``passed=True`` has its shared space_id applied to member routes.
        self._verified_space_pins: Dict[str, "ParityResult"] = {}

        # Provider-neutral private inference route. Infrastructure state lives
        # in an external lease provider; only the validated, host-only route is
        # retained here. The condition/refcount make route removal drain
        # in-flight calls before provider capacity can be released.
        self._backend = BackendType.CLOUD
        self._default_backend = BackendType.CLOUD
        self._remote_lease = None
        self._remote_client: Optional[openai.AsyncOpenAI] = None
        self._remote_adapter = OpenAIAdapter()
        self._last_remote_error: Optional[str] = None
        self._remote_route_condition = asyncio.Condition()
        self._remote_inflight = 0
        self._remote_accepting = False
        self._remote_capabilities: frozenset[str] = frozenset()

        # Observability store for logging LLM calls (A2A-compatible)
        # Set via set_observability_store() after initialization
        self._observability_store = None
        # Legacy ``set_observability_context`` state is task-local *and*
        # service-local.  A process can host multiple agent services in one
        # task; sharing one module ContextVar would cross their billing IDs.
        self._invocation_context_state = LLMInvocationContextState()
        self._last_response_identity: ContextVar[
            Optional[Dict[str, Optional[str]]]
        ] = ContextVar(
            f"kestrel_last_response_identity_{id(self):x}", default=None
        )

        # Metering callback for usage billing (Vending Machine)
        # Set via set_metering_callback() after initialization
        self._metering_callback = None
        # Whether the registered callback accepts the optional per-call
        # ``cost`` kwarg (#1806). Resolved in set_metering_callback().
        self._metering_callback_accepts_cost = False

        # Privacy gate for the embedding routing path (#1492). The chat
        # path threads ``force_local_only`` explicitly at call time
        # (see ``KestrelAgent`` line ~2165), but embeddings are called
        # from the storage layer which has no direct view of the
        # agent's current privacy mode. Bind a callable here that
        # returns the live ``force_local_only`` state; the agent sets
        # it once at init pointing at
        # ``not privacy_agent.privacy_config.allows_cloud_llm()`` so
        # any future privacy-mode change is picked up automatically.
        # When unset (process-local ``LLMService()`` not attached to an
        # agent) the gate defaults to OFF — matching pre-#1492
        # behavior for tooling that legitimately runs without a
        # privacy context (CLI scripts, tests).
        self._force_local_only_provider: Optional[Callable[[], bool]] = None

        # Persistence callback for model preference (writes to database)
        # Set via set_preference_persistence_callback() after initialization
        self._preference_persistence_callback = None
        self._preference_persistence_tasks: set[asyncio.Task[None]] = set()

        # Persistence callback for the top-level embedding_route knob (#2263).
        # Set via set_embedding_route_persistence_callback(); mirrors the model
        # preference persistence mechanism so a runtime change survives restart.
        self._embedding_route_persistence_callback = None

        # Per-route embedding_model overrides chosen at runtime (#2337). Keyed by
        # exact route name ("<vendor>:<route>") -> {"model": str, "dim": int|None}.
        # These mirror the config-file ``embedding_model``/``embedding_dim`` keys
        # under a route but are set from the UI (no TOML editing), and persist the
        # same way as the embedding_route knob (agent_metadata) — NOT a new store.
        self._route_embedding_model_overrides: Dict[str, Dict[str, Any]] = {}
        # Snapshot of each route's capability keys BEFORE its first runtime
        # override, so clearing restores the pre-override (config/discovery) state
        # exactly instead of leaving a stale pin behind.
        self._route_embedding_caps_backup: Dict[str, Dict[str, Any]] = {}
        # Persistence callback for the per-route embedding_model overrides (#2337).
        self._route_embedding_model_persistence_callback = None

        # Corpus embedding-profile provider (#2366). An ``async () -> dict|None``
        # callback wired by the agent that returns the DB's DOMINANT existing
        # embedding profile ({"provider", "model", "dim", "space_id",
        # "row_count"}). Auto-resolution prefers a model matching it so a fresh
        # default doesn't silently move the agent into a new embedding space.
        self._corpus_embedding_profile_provider = None
        # Per-route records of an auto-default that changed the corpus embedding
        # space (#2366). Keyed by route name; surfaced in get_embedding_settings.
        self._embedding_space_change_warnings: Dict[str, Dict[str, Any]] = {}

        # Routes that have failed with a permanent auth error (401/403 or
        # the equivalent "User not found" / "invalid api key" message). Once
        # a route lands here, subsequent fallback iterations skip it for the
        # lifetime of this service — retrying a dead API key every user turn
        # wastes a round-trip and logs a red error on every request (#655).
        # Cleared by restarting the service (picks up rotated keys). Keys
        # are provider route names (matches provider["name"]).
        self._disabled_routes: dict[str, str] = {}

        # Schema modules are imported while FastAPI builds its routes, before a
        # real agent-scoped LLM service exists. Freeze the conversation vector
        # width now, from this already-initialized service, instead of making
        # the schema import construct and discard a second LLMService. This
        # preserves provider-derived dimensions for existing non-768 deployments
        # while removing provider setup from the server import critical path.
        try:
            from kestrel_sovereign.storage.sqla.conversation_message import (
                configure_embedding_dim_from_service,
            )

            configure_embedding_dim_from_service(self)
        except Exception:
            logger.warning(
                "Could not configure conversation embedding schema width from "
                "the initialized provider; using the deployment default.",
                exc_info=True,
            )

    def _stamp_response_identity(
        self,
        response: Any,
        *,
        model: Optional[str],
        provider: Optional[str],
    ) -> None:
        """Attach the resolved route/model to an LLM response object.

        Conversation persistence needs the route that actually answered
        after fallback, not the user's requested selector. The service is
        the one layer that knows that value for every adapter, so it
        records it here and exposes the same fields to callers that
        receive a full ``LLMResponse``.
        """
        identity = {"model": model, "provider": provider}
        self._response_identity_state().set(identity)
        if isinstance(response, LLMResponse):
            try:
                setattr(response, "model", model)
                setattr(response, "provider", provider)
            except Exception:
                logger.debug(
                    "Could not stamp LLMResponse identity provider=%s model=%s",
                    provider, model,
                    exc_info=True,
                )

    def get_last_response_identity(self) -> Dict[str, Optional[str]]:
        """Return the most recent resolved LLM route/model identity."""
        return dict(self._response_identity_state().get() or {})

    def _response_identity_state(
        self,
    ) -> ContextVar[Optional[Dict[str, Optional[str]]]]:
        """Return per-service identity state, including lightweight test hosts."""

        state = getattr(self, "_last_response_identity", None)
        if state is None:
            state = ContextVar(
                f"kestrel_last_response_identity_{id(self):x}", default=None
            )
            self._last_response_identity = state
        return state

    def set_agent_display_name(self, name: Optional[str]) -> None:
        """Record the owning agent's human display name for LLM-span attribution.

        The registrar calls this once the agent's ``agent.agent_name`` is
        resolved so LLM-call spans can carry ``kestrel.agent_name`` (issue
        #2573). Best-effort: unset until wired, and never load-bearing.
        """
        self._agent_display_name = name or None

    @staticmethod
    def _llm_span_output_text(result: Any) -> Optional[str]:
        """Best-effort ``output.value`` text for an LLM span from a call result.

        A bare string is the response; an ``LLMResponse`` contributes its
        ``content`` (``None`` for a pure tool call, which annotation then skips).
        """
        if result is None:
            return None
        if isinstance(result, LLMResponse):
            return result.content
        return result if isinstance(result, str) else str(result)

    def _annotate_and_return(self, span, result: Any, *, redact: bool = False) -> Any:
        """Stamp the served model / response onto ``span`` and pass ``result`` through.

        The served route/model (post-fallback) is only known once the call
        returns; it is read from the just-stamped response identity. Guarded so
        annotation can never alter the value returned to the caller.

        ``redact`` (#2674 finding 3): under an enforcing response audit the
        assistant prose is withheld pending the verdict, so ``output.value``
        must carry a content-free marker instead of the real response — while
        the content-free token attributes (from ``response``) are preserved.
        """
        try:
            if span is not None and span.is_recording():
                served_model = None
                try:
                    served_model = (self.get_last_response_identity() or {}).get("model")
                except Exception:  # noqa: BLE001 - identity read is best-effort
                    served_model = None
                output_text = (
                    _redacted_content_marker(self._llm_span_output_text(result))
                    if redact
                    else self._llm_span_output_text(result)
                )
                telemetry.annotate_llm_response_span(
                    span,
                    output_text=output_text,
                    model_name=served_model,
                    response=result if isinstance(result, LLMResponse) else None,
                )
        except Exception:  # noqa: BLE001 - instrumentation must never break the call
            logger.debug("LLM span annotation failed", exc_info=True)
        return result

    @contextmanager
    def _llm_request_span(
        self,
        method: str,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model_override: Optional[str] = None,
        session_id: Optional[str] = None,
        redact: bool = False,
    ):
        """Open the single OpenInference LLM span for one public entry call.

        ``redact`` (#2674 finding 3): under an enforcing response audit the
        prompt content is withheld pending the verdict, so ``input.value``
        carries a content-free marker instead of the serialized prompt/messages.

        Centralizes the cheap tracing guard + prompt serialization + span open
        shared by every public completion entry (``get_response``,
        ``generate``, ``generate_with_messages``, ``get_response_with_model``).
        Issue #2573, Q1: exactly ONE span per logical request — the
        per-provider fallback loop underneath emits no span of its own, and
        failed attempts land as events on THIS span
        (:func:`telemetry.record_llm_attempt_failure`). Serialization is
        skipped entirely when no exporter is configured, so an unset OTLP
        endpoint costs only the guard.
        """
        tracing_enabled = telemetry.llm_tracing_enabled()
        span_attributes = None
        if not tracing_enabled:
            span_input = None
        else:
            if redact:
                # #2674 finding 3: contentless input marker — never serialize the
                # withheld prompt into the exported span.
                span_input = _redacted_content_marker(
                    user_prompt if user_prompt is not None else (messages or "")
                )
            else:
                span_input = telemetry.serialize_llm_input(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    messages=messages,
                )
            if session_id and session_id.strip():
                span_attributes = {telemetry.OI_SESSION_ID: session_id}
        with telemetry.llm_span(
            f"llm.{method}",
            input_value=span_input,
            model_name=model_override,
            # getattr: tests construct LLMService via __new__ (skipping __init__,
            # see test_discovery_hang_regression), so the attribute may not exist.
            agent_name=getattr(self, "_agent_display_name", None),
            attributes=span_attributes,
        ) as span:
            yield span

    def set_preference_persistence_callback(self, callback) -> None:
        """Set the persistence callback for model preference.

        The callback will be called with (model: str|None, provider: str|None)
        whenever the model preference changes, so the caller can persist it.

        Args:
            callback: Async function(model, provider) to call on preference change
        """
        self._preference_persistence_callback = callback
        logger.info("Model preference persistence enabled")

    def set_embedding_route_persistence_callback(self, callback) -> None:
        """Set the persistence callback for the embedding_route knob (#2263).

        The callback is ``async (route: str|None) -> None`` and is invoked
        whenever ``set_embedding_route`` / ``clear_embedding_route`` changes the
        value, so the caller can persist it the same way model preference is
        persisted (agent_metadata row).
        """
        self._embedding_route_persistence_callback = callback
        logger.info("Embedding route persistence enabled")

    def get_embedding_route(self) -> Optional[str]:
        """Return the configured top-level embedding_route (#2263).

        ``None`` means "auto — embedding follows the active chat route" (the
        pre-#2263 default). ``"none"`` means embeddings are deliberately off
        (#2287). Otherwise a ``"<vendor>"`` or ``"<vendor>:<route>"`` selector
        chosen deliberately by the operator.
        """
        return getattr(self, "_embedding_route", None)

    def set_embedding_route(self, route: Optional[str], *, persist: bool = True) -> None:
        """Set the top-level embedding_route knob (#2263).

        ``route`` is a ``"<vendor>"`` or ``"<vendor>:<route>"`` selector. An
        empty string or ``"auto"`` clears the knob (falls back to
        follow-chat/auto). The special value ``"none"`` (#2287) is a
        first-class off-switch: embeddings are disabled deliberately, storage
        writes skip embedding calls, and semantic search uses keyword fallback
        with no per-write warnings. The route is validated against the
        configured providers — an unknown route, or one that advertises no
        embedding support, is refused so the setting can never silently degrade
        to keyword search on the next storage write. ``"none"`` bypasses this
        validation because it names no provider.

        Args:
            route: The embedding route selector, ``"none"`` to disable
                embeddings deliberately, or ``None``/``""``/``"auto"`` to clear
                (auto/follow-chat).
            persist: When True (default) and a persistence callback is bound,
                schedule the value to be persisted. Set False when applying a
                value that was just loaded from storage.

        Raises:
            ValueError: if the route names no configured provider, or the
                matched route(s) advertise no embedding support.
        """
        if route is not None:
            route = route.strip()
            if not route or route.lower() == "auto":
                route = None
            elif route.lower() == "none":
                # Canonicalize the deliberate off-switch (#2287) to the bare
                # sentinel regardless of input casing/whitespace.
                route = "none"

        # ``"none"`` is a first-class off-switch (#2287): the operator has
        # turned embeddings off on purpose. It names no provider, so skip
        # validation — validating it would reject it as an unknown route.
        if route is not None and route != "none":
            self._validate_embedding_route(route)

        self._embedding_route = route
        if route == "none":
            logger.info(
                'Embeddings disabled by operator (embedding_route = "none"): '
                "storage writes will skip embedding calls and semantic search "
                "uses keyword fallback."
            )
        elif route:
            logger.info("Embedding route set: %s", route)
        else:
            logger.info(
                "Embedding route cleared — embeddings follow the active chat "
                "route (auto)."
            )

        if persist and self._embedding_route_persistence_callback:
            self._schedule_embedding_route_persistence(route)

    async def aset_embedding_route(
        self, route: Optional[str], *, persist: bool = True, force: bool = False
    ) -> None:
        """Set the embedding_route, adding a live upstream probe (#2326).

        ``set_embedding_route`` only runs *static* validation: the route is
        known and advertises embedding support. That is not enough for cloud
        meta-providers (OpenRouter et al.) which list models whose serving
        provider pool can be empty — the route passes static validation yet
        every real embed 404s and rows silently persist with NULL vectors.

        For an explicit **cloud** route, this async variant embeds one canary
        string through the resolved provider after static validation passes and
        refuses the set (``ValueError``) on any upstream failure, so the knob can
        never be accepted in a state that degrades to keyword search on the next
        storage write. Local routes and the ``"none"``/auto sentinels are not
        probed. Use this on explicit sets (the settings endpoint); the boot-time
        persisted-value load stays on the sync, probe-free
        :meth:`set_embedding_route`.

        Raises:
            ValueError: if static validation fails, or the live canary embed
                against a cloud route fails / returns no vector.
        """
        # Normalize identically to set_embedding_route so the probe resolves the
        # canonical selector (and "auto"/"" clear, "none" off-switch, are seen).
        normalized = route
        if normalized is not None:
            normalized = normalized.strip()
            if not normalized or normalized.lower() == "auto":
                normalized = None
            elif normalized.lower() == "none":
                normalized = "none"

        if normalized is not None and normalized != "none":
            # Static validation first (unknown route / no embedding support).
            self._validate_embedding_route(normalized)
            # Dim-compatibility gate (#2417): a route whose resolved dim differs
            # from the column dim would break every future write — refuse it at
            # set time (before the live probe, so the actionable dim error wins
            # over an unrelated upstream hiccup), unless the operator forces it.
            self._check_embedding_dim_writable(
                self._resolved_route_embedding_dim(normalized),
                route=normalized,
                force=force,
            )
            # Then the live probe for cloud routes (#2326).
            await self._probe_embedding_route_live(normalized)

        # Commit through the sync setter (re-runs the cheap static validation,
        # owns logging + persistence).
        self.set_embedding_route(route, persist=persist)

    async def _probe_embedding_route_live(self, route: str) -> None:
        """Embed one canary through a cloud embedding route to prove it's live (#2326).

        Static validation only proves the route is known and advertises
        embedding support; a meta-provider can list a model whose provider pool
        is empty (dead upstream), so every real embed 404s. Probe the resolved
        provider with a single canary — reusing the #2290 parity canaries — and
        raise ``ValueError`` on failure so the set is refused with the upstream
        error surfaced to the operator.

        Local routes are skipped: the empty-pool failure mode is a live-catalog
        property of cloud meta-providers, and a missing local model is a
        separate, already-handled setup issue. When no embedding-capable
        provider resolves (pre-init / bare harness), there is nothing live to
        probe, so this is a no-op.
        """
        candidates = self._lookup_embedding_route_candidates(route)
        target = next(
            (c for c in candidates if self._provider_supports_embeddings(c)),
            None,
        )
        if target is None:
            # Static validation (against ``self.providers``) already passed, but
            # the live probe resolves through ``_available_providers()``, which
            # filters out routes in ``_disabled_routes``. If the route statically
            # matches an embedding-capable provider yet none is *available*, the
            # route is disabled in this service — committing it would degrade
            # every subsequent write to keyword fallback with no probe. Refuse
            # the set rather than silently accepting a dead route.
            providers = getattr(self, "providers", None) or []
            if providers:
                static_matching = self._filter_providers_by_selector(
                    providers, route
                )
                if any(
                    self._provider_supports_embeddings(p) for p in static_matching
                ):
                    disabled = self._disabled_routes.get(route)
                    reason = f" ({disabled})" if disabled else ""
                    raise ValueError(
                        f"Cannot set embedding_route '{route}': the route is "
                        f"configured but not currently available"
                        f"{reason}, so it cannot be probed and would fall back "
                        f"to keyword search on the next storage write."
                    )
            # Otherwise no providers are initialized (pre-init / bare harness):
            # nothing live to probe, so this is a legitimate no-op.
            return
        if target.get("is_local") or not target.get("is_cloud", True):
            return

        from .embedding_service import ProviderEmbeddingService
        from .embedding_space import DEFAULT_PARITY_CANARIES

        service = ProviderEmbeddingService(target)
        canary = DEFAULT_PARITY_CANARIES[0]
        try:
            vector = await service.aembed(canary)
        except Exception as exc:
            hint = self._embedding_probe_hint(route, exc, subject="route")
            raise ValueError(
                f"Cannot set embedding_route '{route}': live embedding probe "
                f"failed {hint}"
            ) from exc
        if not vector:
            raise ValueError(
                f"Cannot set embedding_route '{route}': the upstream provider "
                f"returned no embedding for a canary probe. The route may list "
                f"a model that is not currently served."
            )

    def clear_embedding_route(self) -> None:
        """Clear the embedding_route knob, returning to auto/follow-chat (#2263)."""
        self.set_embedding_route(None)

    def _validate_embedding_route(self, route: str) -> None:
        """Validate an embedding_route selector against configured providers.

        Permissive about UNKNOWN state (mirrors ``_validate_explicit_mandate``):
        when no providers are configured yet (pre-init / bare harness) the
        route is trusted. Once providers exist, the selector must match at
        least one, and at least one matched route must advertise embedding
        support.
        """
        providers = getattr(self, "providers", None) or []
        if not providers:
            return

        matching = self._filter_providers_by_selector(providers, route)
        if not matching:
            known = sorted({p.get("name") for p in providers if p.get("name")})
            raise ValueError(
                f"Cannot set embedding_route: no configured route matches "
                f"'{route}'. Known routes: {known or '(none)'}."
            )
        if not any(self._provider_supports_embeddings(p) for p in matching):
            raise ValueError(
                f"Cannot set embedding_route '{route}': that route does not "
                f"advertise embedding support."
            )

    def _resolved_route_embedding_dim(self, route: str) -> Optional[int]:
        """The embedding dim the given ``route`` resolves to, or ``None`` (#2417).

        Reads the declared ``embedding_dim`` from the matching embedding-capable
        provider's capabilities — the same value the UI surfaces as the resolved
        ``embedding_dim``. ``None`` when no provider resolves or none declares a
        dim (pre-init / bare harness), which the dim gate treats as "unknown".
        """
        providers = getattr(self, "providers", None) or []
        if not providers:
            return None
        matching = self._filter_providers_by_selector(providers, route)
        for provider in matching:
            if not self._provider_supports_embeddings(provider):
                continue
            dim = (provider.get("capabilities") or {}).get("embedding_dim")
            if dim:
                return int(dim)
        return None

    @staticmethod
    def _deployment_embedding_dim() -> Optional[int]:
        """The width the memory vector columns are ACTUALLY sized to (#2417).

        Returns the frozen ``CONVERSATION_MESSAGE_EMBEDDING_DIM`` — the value
        the ORM's ``embedding_vec`` columns were built with at import/migration
        time and the exact width the write-path dim guard enforces against
        (``async_conversation_store`` rejects any embedding whose length ≠ this
        constant). It deliberately does **not** re-run ``resolve_embedding_dim()``:
        without an explicit ``KESTREL_EMBEDDING_DIM`` that resolver consults the
        *active provider* and can drift to a different dim (e.g. 1536) than the
        frozen 768-dim column — which would let the set-time gate wave through a
        route whose writes the column still silently rejects. The gate must
        compare against the same value the write path uses.

        Returns ``None`` only when the constant can't be imported (bare
        harness), which callers treat as "can't check".
        """
        try:
            from kestrel_sovereign.storage.sqla.conversation_message import (
                CONVERSATION_MESSAGE_EMBEDDING_DIM,
            )

            dim = CONVERSATION_MESSAGE_EMBEDDING_DIM
            return int(dim) if dim else None
        except Exception:  # pragma: no cover - defensive; never crash a setter
            return None

    @staticmethod
    def _dim_write_state(
        resolved_dim: Optional[int], column_dim: Optional[int]
    ) -> tuple[bool, Optional[str]]:
        """Whether the resolved embedding dim can write to the column (#2417).

        Returns ``(write_blocked, status)`` — ``write_blocked`` is ``True`` only
        when both dims are known and differ (a write in this state hits the
        storage dim guard and persists without a vector, silently pausing
        semantic memory). ``status`` is the operator-facing popover string when
        blocked, else ``None``. Shared by :meth:`get_embedding_settings` and
        :meth:`aget_embedding_settings_for_route` so the route-scoped echo can
        recompute after overlaying its own model/dim instead of carrying the
        global route's stale flag.
        """
        write_blocked = bool(
            resolved_dim is not None
            and column_dim is not None
            and int(resolved_dim) != int(column_dim)
        )
        if not write_blocked:
            return False, None
        status = (
            f"selected provider cannot write — memory vectors paused "
            f"(resolves {int(resolved_dim)}-dim, columns are "
            f"{int(column_dim)}-dim)"
        )
        return True, status

    def _check_embedding_dim_writable(
        self,
        candidate_dim: Optional[int],
        *,
        route: str,
        model: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Refuse a route/model whose resolved dim can't write to the column (#2417).

        The dim incompatibility is fully KNOWN at selection time (the resolved
        model dim vs ``kestrel_embedding_dim``). If they differ, every future
        memory write hits the storage dim guard (``provider returned N, column
        is M``) and persists WITHOUT a vector — the agent's semantic memory
        silently stops accruing. So the same set-time gate that #2326 uses for
        dead upstreams also refuses a dim mismatch here, with an actionable
        error naming both dims and the ways out.

        ``candidate_dim`` is skipped when ``None`` (unknown — nothing to check).
        ``force`` lets an operator mid-migration override the refusal (they are
        intentionally re-sizing the column + reindexing).

        Raises:
            ValueError: when the resolved dim differs from the column dim and
                ``force`` is not set.
        """
        if candidate_dim is None:
            return
        column_dim = self._deployment_embedding_dim()
        if not column_dim or int(candidate_dim) == int(column_dim):
            return
        target = (
            f"model '{model}' on route '{route}'" if model else f"route '{route}'"
        )
        if force:
            logger.warning(
                "Forced embedding %s despite dim mismatch (resolves %d-dim vs "
                "%d-dim column) — writes are broken until the column is "
                "re-migrated and reindexed.",
                target,
                int(candidate_dim),
                int(column_dim),
            )
            return
        raise ValueError(
            f"Cannot set embedding {target}: it resolves to {int(candidate_dim)}-dim "
            f"vectors but this deployment's memory columns are {int(column_dim)}-dim. "
            f"Every memory write would be refused by the dim guard and persist "
            f"without a vector, silently pausing semantic memory. Ways forward: "
            f"(a) pick a dimension the model supports that matches the column "
            f"(MRL 'dimensions'={int(column_dim)}); (b) migrate the deployment "
            f"(set KESTREL_EMBEDDING_DIM={int(candidate_dim)}, re-migrate the vector "
            f"columns, and reindex); or (c) choose a {int(column_dim)}-dim-compatible "
            f"provider. Pass force=true to override mid-migration."
        )

    def _schedule_embedding_route_persistence(
        self, route: Optional[str]
    ) -> Optional["asyncio.Task[None]"]:
        """Own embedding_route persistence callbacks so close() can await them.

        Callback signature is ``async (route) -> None``. Tasks are tracked in
        the shared ``_preference_persistence_tasks`` set so
        ``drain_preference_persistence`` covers them too.
        """
        if not self._embedding_route_persistence_callback:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — skip persistence (tests / sync contexts).
            return None
        task = loop.create_task(
            self._embedding_route_persistence_callback(route),
            name="llm-embedding-route-persistence",
        )
        self._preference_persistence_tasks.add(task)
        task.add_done_callback(self._handle_preference_persistence_done)
        return task

    # --- Per-route embedding_model overrides (#2337) -------------------------

    def set_route_embedding_model_persistence_callback(self, callback) -> None:
        """Set the persistence callback for per-route embedding_model pins (#2337).

        The callback is ``async (overrides: dict) -> None`` and is invoked
        whenever a per-route embedding_model override is set or cleared, so the
        caller can persist the whole override map the same way the embedding_route
        knob and model preference persist (agent_metadata row) — not a new store.
        """
        self._route_embedding_model_persistence_callback = callback
        logger.info("Per-route embedding_model persistence enabled")

    def get_route_embedding_model_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the runtime per-route embedding_model overrides (#2337)."""
        overrides = getattr(self, "_route_embedding_model_overrides", None) or {}
        return {route: dict(spec) for route, spec in overrides.items()}

    def set_corpus_embedding_profile_provider(self, callback) -> None:
        """Wire the DB's dominant embedding-profile provider (#2366).

        ``callback`` is ``async () -> dict|None`` returning the corpus's
        dominant embedding profile (``{"provider", "model", "dim", "space_id",
        "row_count"}``) or ``None`` for an empty/unreadable corpus. Auto
        embedding-model resolution consults it so a fresh default prefers
        continuity with the existing corpus over catalog order.
        """
        self._corpus_embedding_profile_provider = callback
        logger.info("Corpus embedding-profile provider enabled")

    def _find_route_provider(self, route: str) -> Optional[Dict[str, Any]]:
        """Return the configured provider whose ``name`` matches ``route`` exactly."""
        providers = getattr(self, "providers", None) or []
        return next((p for p in providers if p.get("name") == route), None)

    def set_route_embedding_model(
        self,
        route: str,
        model: Optional[str],
        dim: Optional[int] = None,
        *,
        persist: bool = True,
    ) -> None:
        """Pin (or clear) a route's embedding model at runtime (#2337).

        Config pins ``embedding_model``/``embedding_dim`` under a route in
        kestrel.toml; this is the runtime equivalent set from the embeddings UI
        so the operator never hand-edits TOML. Writing the pin into the route's
        ``capabilities`` re-advertises embedding support for that exact route
        (mirrors :meth:`reconcile_embedding_capabilities`), making it selectable
        and usable by storage immediately. Passing ``model`` as ``None``/``""``
        clears the override and restores the route's pre-override
        (config/discovery) capability state.

        Route-specific by design: a pin on ``openai:api`` never flips capability
        on for a sibling ``openai:plan``. ``route`` must name an exact configured
        route (``"<vendor>:<route>"``).

        Args:
            route: Exact route name to pin the model on.
            model: The embedding model id, or ``None``/``""`` to clear.
            dim: Optional embedding dimension forwarded as the Matryoshka
                ``dimensions`` param and used to key the embedding space.
            persist: When True (default) and a persistence callback is bound,
                schedule the override map to be persisted.

        Raises:
            ValueError: if ``route`` names no configured route.
        """
        if not route or not isinstance(route, str):
            raise ValueError("A route name is required to pin an embedding model.")
        route = route.strip()

        provider = self._find_route_provider(route)
        # Permissive about UNKNOWN state (mirrors _validate_embedding_route):
        # pre-init / bare harness has no providers to validate against.
        providers = getattr(self, "providers", None) or []
        if providers and provider is None:
            known = sorted({p.get("name") for p in providers if p.get("name")})
            raise ValueError(
                f"Cannot set embedding_model: no configured route matches "
                f"'{route}'. Known routes: {known or '(none)'}."
            )

        clearing = model is None or (isinstance(model, str) and not model.strip())

        # Shared-space coherence gate (#2440): refuse a pin that would fragment a
        # VERIFIED shared space BEFORE mutating capabilities/overrides. The async
        # setter runs this too (as a pre-check that also skips the live probe),
        # but production boot/settings/reindex hydration re-applies persisted
        # pins through THIS synchronous path — so a pre-existing conflicting
        # stored pin must be rejected here as well, not silently re-applied.
        if not clearing:
            self._check_route_pin_space_coherent(route, model.strip(), dim)

        if provider is not None:
            caps = provider.get("capabilities")
            if not isinstance(caps, dict):
                caps = {}
                provider["capabilities"] = caps
            # Snapshot the pre-override capability keys once, so a later clear
            # restores exactly what config/discovery had established.
            if route not in self._route_embedding_caps_backup:
                self._route_embedding_caps_backup[route] = {
                    key: caps[key]
                    for key in ("embedding_model", "embedding_dim", "supports_embeddings")
                    if key in caps
                }

            if clearing:
                backup = self._route_embedding_caps_backup.pop(route, {})
                for key in ("embedding_model", "embedding_dim", "supports_embeddings"):
                    if key in backup:
                        caps[key] = backup[key]
                    else:
                        caps.pop(key, None)
                # Drop any auto-resolved marker left by a prior default; the next
                # resolve re-adds it if the restored state is still auto (#2372).
                caps.pop("embedding_model_auto_resolved", None)
            else:
                caps["embedding_model"] = model.strip()
                if dim is not None:
                    caps["embedding_dim"] = int(dim)
                caps["supports_embeddings"] = True
                # A runtime pin is operator intent, not an auto default — clear
                # the marker so discovery folds it in as ``is_pinned`` (#2372).
                caps.pop("embedding_model_auto_resolved", None)

        if clearing:
            self._route_embedding_model_overrides.pop(route, None)
            logger.info("Cleared runtime embedding_model override for %s", route)
        else:
            spec: Dict[str, Any] = {"model": model.strip()}
            if dim is not None:
                spec["dim"] = int(dim)
            self._route_embedding_model_overrides[route] = spec
            logger.info(
                "Pinned embedding_model for %s: %s%s",
                route,
                spec["model"],
                f" @ {spec['dim']}" if spec.get("dim") is not None else "",
            )

        # Discovery results are cached per instance; invalidate so the newly
        # pinned/cleared model is reflected on the next discover call.
        self._embedding_discovery_cache = None

        if persist and self._route_embedding_model_persistence_callback:
            self._schedule_route_embedding_model_persistence()

    async def aset_route_embedding_model(
        self,
        route: str,
        model: Optional[str],
        dim: Optional[int] = None,
        *,
        persist: bool = True,
        force: bool = False,
    ) -> None:
        """Set a per-route embedding_model, adding a live probe on save (#2337/#2326).

        Mirrors :meth:`aset_embedding_route`: a cloud route's candidate model is
        embedded once (a canary) BEFORE the pin is committed, so a dead/misspelled
        upstream slug is refused with a ``ValueError`` at configuration time rather
        than silently 404'ing to keyword fallback on the next storage write. Local
        routes and the clear path skip the probe.

        Raises:
            ValueError: if ``route`` is unknown, or the live canary embed against
                a cloud route with the candidate model fails / returns no vector.
        """
        clearing = model is None or (isinstance(model, str) and not model.strip())
        if not clearing:
            # Validate the route exists before probing (raises on unknown route).
            if not route or not isinstance(route, str):
                raise ValueError("A route name is required to pin an embedding model.")
            provider = self._find_route_provider(route.strip())
            providers = getattr(self, "providers", None) or []
            if providers and provider is None:
                known = sorted({p.get("name") for p in providers if p.get("name")})
                raise ValueError(
                    f"Cannot set embedding_model: no configured route matches "
                    f"'{route.strip()}'. Known routes: {known or '(none)'}."
                )
            # Shared-space coherence gate (#2440): a route that is a member of a
            # VERIFIED shared space (#2290/#2376) cannot be pinned to a
            # model/dim that differs from the space's — that would fragment the
            # verified space. The old behaviour accepted+stored such a pin but
            # silently let the space win, echoing a model that never took
            # effect. Refuse at set time (matches the #2417 refuse-at-set-time
            # philosophy) so the operator clears the space or chooses its model
            # instead of receiving a phantom pin.
            self._check_route_pin_space_coherent(route.strip(), model.strip(), dim)
            # Dim-compatibility gate (#2417): the pinned model resolves to a
            # known dim (the explicit ``dim`` the UI passes from the catalog's
            # native_dim, else the route's declared capability dim). If it
            # differs from the column dim every write is broken — refuse unless
            # forced, before the liveness probe.
            candidate_dim = dim if dim is not None else self._resolved_route_embedding_dim(
                route.strip()
            )
            self._check_embedding_dim_writable(
                candidate_dim,
                route=route.strip(),
                model=model.strip(),
                force=force,
            )
            await self._probe_route_embedding_model_live(route.strip(), model.strip(), dim)

        self.set_route_embedding_model(route, model, dim, persist=persist)

    async def _probe_route_embedding_model_live(
        self, route: str, model: str, dim: Optional[int]
    ) -> None:
        """Embed one canary through ``route`` using ``model`` to prove it's live (#2337).

        Reuses the #2326 canary machinery, but against a candidate model that is
        not yet committed to the route's capabilities: builds a throwaway provider
        copy whose capabilities carry the candidate model/dim and embeds a single
        canary. Cloud routes only — a local route's missing model is a separate
        (already-handled) setup problem, and a bare harness with no live provider
        has nothing to probe.
        """
        provider = self._find_route_provider(route)
        if provider is None:
            # No providers initialized (pre-init / bare harness) — nothing live.
            return
        if provider.get("is_local") or not provider.get("is_cloud", True):
            return

        # Shallow-copy the provider and overlay the candidate model/dim onto a
        # copied capabilities dict so the real route config is untouched if the
        # probe fails.
        probe_provider = dict(provider)
        caps = dict(provider.get("capabilities") or {})
        caps["embedding_model"] = model
        if dim is not None:
            caps["embedding_dim"] = int(dim)
        probe_provider["capabilities"] = caps

        from .embedding_service import ProviderEmbeddingService
        from .embedding_space import DEFAULT_PARITY_CANARIES

        service = ProviderEmbeddingService(probe_provider)
        canary = DEFAULT_PARITY_CANARIES[0]
        try:
            vector = await service.aembed(canary)
        except Exception as exc:
            raise ValueError(
                self._embedding_probe_failure_message(route, model, exc)
            ) from exc
        if not vector:
            raise ValueError(
                f"Cannot pin embedding_model '{model}' on '{route}': the upstream "
                f"provider returned no embedding for a canary probe. The model may "
                f"not be currently served."
            )

    @staticmethod
    def _classify_embedding_probe_failure(exc: Exception) -> str:
        """Bucket a live embedding-probe exception (#2418).

        The old probe blamed every failure on "the model may not be served" —
        the 404 hint — even when the real cause was a dead/revoked credential
        (a 401 ``User not found.`` from an agent-scoped sub-key) or a transient
        network/timeout blip. Returns one of ``"auth"`` / ``"not_served"`` /
        ``"transient"`` / ``"unknown"`` so the caller can point the operator at
        the right remedy (re-mint the key vs. pick another model vs. retry).

        Order matters: auth is checked FIRST because a 401 ``User not found.``
        literally contains "not found" and would otherwise misclassify as a
        missing model.
        """
        # Auth (401/403 or an auth-worded message) — reuse the single source of
        # truth used to disable dead routes.
        if LLMService._is_permanent_auth_error(exc):
            return "auth"

        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        msg = str(exc).lower()

        # Model-not-served (404 / "no provider serving" / "does not exist").
        not_served_patterns = (
            "not served",
            "no provider",
            "no endpoint",
            "does not exist",
            "model_not_found",
            "no such model",
            "unknown model",
            "not found",
            "404",
        )
        if status == 404 or any(p in msg for p in not_served_patterns):
            return "not_served"

        # Transient — timeouts / connectivity / upstream overload.
        transient_patterns = (
            "timeout",
            "timed out",
            "connection",
            "unreachable",
            "temporarily",
            "try again",
            "overloaded",
            "rate limit",
            "too many requests",
        )
        if status in (408, 429, 500, 502, 503, 504) or any(
            p in msg for p in transient_patterns
        ):
            return "transient"

        return "unknown"

    @staticmethod
    def _embedding_probe_hint(route: str, exc: Exception, *, subject: str) -> str:
        """Classify a probe failure into an operator-facing hint (#2418).

        ``subject`` names what is missing/served ("model" or "route") so the
        not-served hint reads naturally for either probe. The auth hint routes
        the operator at the agent's own credential — the exact miss the issue
        described (a 401 ``User not found.`` from a dead agent-scoped sub-key,
        while the host-level key still works).
        """
        vendor = route.split(":", 1)[0] if route else "the"
        category = LLMService._classify_embedding_probe_failure(exc)
        if category == "auth":
            return (
                f"— this agent's {vendor} credential is invalid or revoked "
                f"({exc}). Check or re-mint the agent's {vendor} key; the "
                f"host-level key working does not mean this agent's key does."
            )
        if category == "transient":
            return (
                f"— the {vendor} provider timed out or was unreachable ({exc}). "
                f"This is likely transient; retry in a moment."
            )
        if category == "not_served":
            return (
                f"against the upstream provider ({exc}). "
                f"The {subject} may not be currently served."
            )
        return (
            f"against the upstream provider ({exc}). The {subject} may not be "
            f"currently served, or the credential may be invalid."
        )

    @staticmethod
    def _embedding_probe_failure_message(
        route: str, model: str, exc: Exception
    ) -> str:
        """Build the per-route pin-rejection message, classifying the cause (#2418).

        Always contains "live embedding probe failed" so existing callers/tests
        keying on that survive; the trailing hint is what changes per category.
        """
        hint = LLMService._embedding_probe_hint(route, exc, subject="model")
        return (
            f"Cannot pin embedding_model '{model}' on '{route}': live "
            f"embedding probe failed {hint}"
        )

    def _schedule_route_embedding_model_persistence(
        self,
    ) -> Optional["asyncio.Task[None]"]:
        """Own per-route embedding_model persistence so close() can await it (#2337)."""
        if not self._route_embedding_model_persistence_callback:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        task = loop.create_task(
            self._route_embedding_model_persistence_callback(
                self.get_route_embedding_model_overrides()
            ),
            name="llm-route-embedding-model-persistence",
        )
        self._preference_persistence_tasks.add(task)
        task.add_done_callback(self._handle_preference_persistence_done)
        return task

    def get_embedding_settings(self) -> Dict[str, Any]:
        """Return the resolved embedding-channel state for the active session (#2263).

        Shape (enough for a UI to render an "Auto — follow chat" default and a
        dimension-mismatch warning). ``embedding_route`` distinguishes the
        three states (#2287): ``None`` == auto/follow-chat, a
        ``"<vendor>:<route>"`` selector == explicit, and ``"none"`` ==
        deliberately off (keyword search only). In the off state
        ``resolved_route``/``embedding_model``/``embedding_dim`` are ``None``
        while ``kestrel_embedding_dim`` is still reported:

            embedding_route:        the configured knob: ``None`` == auto,
                                    ``"none"`` == off, else explicit selector.
            resolved_route:         the ``"<vendor>:<route>"`` actually resolved
                                    for the active session, or ``None`` when
                                    embedding is unavailable (keyword fallback).
            embedding_model:        the resolved provider's embedding model id.
            embedding_dim:          the resolved provider's embedding dimension.
            kestrel_embedding_dim:  the deployment's effective embedding dim
                                    (driven by ``KESTREL_EMBEDDING_DIM``), which
                                    the vector columns are sized to.
        """
        configured = getattr(self, "_embedding_route", None)
        provider = self.resolve_embedding_provider()
        resolved_route = None
        embedding_model = None
        embedding_dim = None
        if provider is not None:
            resolved_route = provider.get("name")
            capabilities = provider.get("capabilities") or {}
            embedding_model = capabilities.get("embedding_model")
            embedding_dim = capabilities.get("embedding_dim")

        # #2417 — the "column dim" is the FROZEN width the write path enforces
        # against (see :meth:`_deployment_embedding_dim`), not a re-resolved
        # provider default. Reading it any other way lets the status echo a
        # deployment dim that disagrees with what actually blocks writes.
        deployment_dim = self._deployment_embedding_dim()

        # #2290 — surface the shared local/cloud embedding space so the UI can
        # render it as ONE entry ("qwen3-embedding-0.6b — local + cloud")
        # instead of two routes. Only the pin covering the resolved route is
        # reported, with its verification/drift state.
        shared_space = self._shared_space_payload(provider)

        # #2366 — surface an auto-default that moved the resolved route into a
        # NEW embedding space (away from the corpus's dominant profile). The UI
        # mismatch banner renders this so the operator can re-embed or pin.
        space_change_warning = None
        warnings = getattr(self, "_embedding_space_change_warnings", None)
        if isinstance(warnings, dict) and resolved_route:
            space_change_warning = warnings.get(resolved_route)

        # #2417 — an agent already in the broken state (a resolved route whose
        # dim ≠ the column dim) has its memory writes silently paused: every
        # write hits the storage dim guard and persists without a vector. Surface
        # that as a first-class status the popover can render, not only server
        # logs. Distinct from the softer #2264 "re-embed" mismatch warning: this
        # says writes are *paused right now*.
        write_blocked, dim_write_status = self._dim_write_state(
            embedding_dim, deployment_dim
        )

        return {
            "embedding_route": configured,
            "resolved_route": resolved_route,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "kestrel_embedding_dim": deployment_dim,
            "dim_write_blocked": write_blocked,
            "dim_write_status": dim_write_status,
            "shared_space": shared_space,
            # #2337 — the runtime per-route embedding_model pins the UI's model
            # picker reflects (keyed by "<vendor>:<route>"). Empty when none set.
            "route_embedding_models": self.get_route_embedding_model_overrides(),
            # #2366 — non-null when the auto default changed the corpus space.
            "space_change_warning": space_change_warning,
        }

    async def aget_embedding_settings(self) -> Dict[str, Any]:
        """Async settings read that RESOLVES the active route's model/dim first (#2372).

        ``get_embedding_settings`` (sync) reads whatever is in the resolved
        route's capabilities; if a cleared pin left them empty it surfaced
        ``embedding_model: null`` — embeddings silently off — even while a
        corpus-matching model was discoverable. This resolves the active route
        through the single :meth:`resolve_route_embedding_model` (the #2366 order
        with normalized matching) BEFORE the sync read, so the GET falls through
        corpus-match → catalog fallback instead of reporting ``None``.

        Round-4 (#2372): a cleared per-route pin drops the route's
        ``supports_embeddings`` flag, and ``resolve_embedding_provider`` GATES
        the explicit-route branch on that sync flag — so it returned ``None``
        (silent-off) before we ever got a provider to resolve, even though the
        route could still discover a capable model. The capability advertise and
        the resolution are circular (can't advertise without resolving, won't
        resolve the provider ``resolve_embedding_provider`` gates away). Break the
        cycle by re-advertising capability across every discovering route FIRST
        (the single resolver, funnelled through ``reconcile_embedding_capabilities``)
        so the read is self-sufficient regardless of whether the caller
        pre-reconciled. Best-effort — a discovery hiccup must never fail the read.
        """
        if hasattr(self, "reconcile_embedding_capabilities"):
            try:
                await self.reconcile_embedding_capabilities(use_cache=True)
            except Exception as exc:  # pragma: no cover - never fail the read
                logger.debug("embedding capability reconcile skipped in aget: %s", exc)
        provider = self.resolve_embedding_provider()
        if provider is not None:
            try:
                await self.resolve_route_embedding_model(provider)
            except Exception as exc:  # pragma: no cover - never fail the read
                logger.debug("active-route embedding resolve skipped: %s", exc)
        return self.get_embedding_settings()

    async def aget_embedding_settings_for_route(
        self, route: str
    ) -> Dict[str, Any]:
        """Settings echo for a SPECIFIC route (#2372) — never crosses routes.

        The route-model POST echoes the settings the operator just configured.
        Reading the globally-resolved embedding provider crossed routes — a pin
        on ``openrouter:api`` echoed whatever the active ``embedding_route`` /
        chat route resolved (e.g. an Ollama slug). This resolves the NAMED route
        through the single :meth:`resolve_route_embedding_model` and overlays its
        own ``resolved_route`` / ``embedding_model`` / ``embedding_dim`` so the
        echo reflects the pinned route's own slug.
        """
        settings = self.get_embedding_settings()
        provider = self._find_route_provider(route)
        if provider is not None:
            try:
                model, dim = await self.resolve_route_embedding_model(provider)
            except Exception as exc:  # pragma: no cover - never fail the echo
                logger.debug("route embedding resolve skipped for %s: %s", route, exc)
            else:
                route_name = provider.get("name")
                settings["resolved_route"] = route_name
                settings["embedding_model"] = model
                settings["embedding_dim"] = dim
                # #2417 — the base echo's ``dim_write_blocked``/``dim_write_status``
                # were computed against the GLOBAL route's resolved dim. Now that
                # this route's own ``embedding_dim`` is overlaid, recompute against
                # it — else a forced mismatched pin could echo ``embedding_dim=1536``
                # yet keep the global route's ``dim_write_blocked=false``.
                (
                    settings["dim_write_blocked"],
                    settings["dim_write_status"],
                ) = self._dim_write_state(dim, settings.get("kestrel_embedding_dim"))
                # #2372 — the base echo started from ``get_embedding_settings()``,
                # which resolves the GLOBAL active provider, so its
                # ``shared_space``/``space_change_warning`` describe that route,
                # not the one being echoed. Overlay the route-scoped values so no
                # field crosses routes.
                settings["shared_space"] = self._shared_space_payload(provider)
                warnings = getattr(self, "_embedding_space_change_warnings", None)
                settings["space_change_warning"] = (
                    warnings.get(route_name)
                    if isinstance(warnings, dict)
                    else None
                )
        return settings

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
                zero matches (unknown model) or multiple matches (ambiguous);
                or if an explicit ``{vendor, route, model}`` triple names a
                route that is not configured, or a model not serveable on it.
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
        else:
            # Explicit-vendor path. The vendor-less branch above validates the
            # model against discovery before persisting; an explicit triple
            # must be held to the same bar, else a hallucinated/stale
            # ``{vendor, route, model}`` lands a broken mandate that only
            # surfaces on the NEXT request (the #1927 route-fidelity skew).
            self._validate_explicit_mandate(model, vendor, route)

        self._mandate_preference = {"vendor": vendor, "model": model, "route": route}
        if route:
            logger.info("Model preference set: %s:%s/%s", vendor, route, model)
        else:
            logger.info("Model preference set: %s/%s", vendor, model)

        if self._preference_persistence_callback:
            self._schedule_preference_persistence(model, vendor, route)

    def _validate_explicit_mandate(
        self,
        model: str,
        vendor: str,
        route: Optional[str],
    ) -> None:
        """Validate an explicit ``{vendor, route, model}`` triple before persisting.

        The vendor-less path in :meth:`set_model_preference` resolves+validates
        the model against discovery. This is the symmetric guard for the
        explicit-vendor path: an unknown vendor/route or a model that no
        configured route can serve must be refused at *set* time, not silently
        persisted to break the next request (the #1927 route-fidelity skew, hit
        by the #1925 dogfooding sweep — tracked in #1946).

        Validation is deliberately permissive about UNKNOWN state, only gating
        against *known* mismatches (mirrors ``_model_available_for_route``):

        - If no routes are configured yet (``self.providers`` empty/unset), we
          can't validate the route — trust the caller.
        - Routes ARE checked against ``self.providers`` (the same exact-match
          semantics as ``_filter_providers_by_selector``: ``vendor:route``
          matches a route ``name``; a bare ``vendor`` matches any route for
          that vendor). A miss is a hard error — the route doesn't exist.
        - The model is checked only once discovery has populated a catalog. A
          route's own configured default always counts. Route-scoped catalogs
          (e.g. ``openai:plan`` via codex) are honored: when a matched route
          carries its own catalog, the model must be in THAT catalog, not the
          vendor's broader one. An empty/absent catalog is treated as
          "unknown" → permitted (cold-start safety).

        Raises:
            ValueError: on an unconfigured route or a model that no matched
                route can serve once discovery is populated.
        """
        if not model:
            return

        selector = f"{vendor}:{route}" if route else vendor
        providers = getattr(self, "providers", None) or []

        # --- Route existence -------------------------------------------------
        # Only enforce when routes are actually configured; an empty provider
        # list means we're pre-init / in a bare harness and can't validate.
        if providers:
            matching = self._filter_providers_by_selector(providers, selector)
            if not matching:
                known = sorted({
                    p.get("name") for p in providers if p.get("name")
                })
                raise ValueError(
                    f"Cannot set model: no configured route matches "
                    f"'{selector}'. Known routes: {known or '(none)'}. "
                    f"Use list_models to discover valid vendor/route/model "
                    f"values."
                )
        else:
            matching = []

        # --- Model serveability ---------------------------------------------
        # Gate only against known mismatches. If discovery hasn't populated a
        # catalog yet, permit (cold-start) — same contract as
        # ``_model_available_for_route``.
        #
        # Ordering matters: a POPULATED route-scoped catalog (e.g. codex's
        # models_cache for ``openai:plan``) that proves the model unservable
        # must reject BEFORE the vendor-catalog fallback — otherwise an empty
        # shared vendor cache would permit e.g. ``gpt-5.5-pro`` on the plan
        # route (the #1933 skew). So we resolve the route-scoped verdict first.
        self._ensure_route_catalogs_sync()
        route_catalogs = getattr(self, "_route_catalogs", None) or {}

        # First pass: route's own default, and route-scoped catalogs.
        scoped_missed = 0  # matched routes that are scoped, populated, and missed
        for provider in matching:
            if provider.get("model") == model:
                return  # the route's own configured default is always serveable
            route_key = provider.get("name")
            if route_key in route_catalogs:
                scoped = route_catalogs[route_key]
                if not scoped:
                    return  # empty/unbuilt route catalog → unknown, permit
                if any(getattr(m, "id", None) == model for m in scoped):
                    return  # served by this route's own scoped catalog
                scoped_missed += 1

        # If EVERY matched route is route-scoped with a populated catalog and
        # none served the model, the route itself has proven the mandate
        # invalid — reject regardless of the (possibly empty) vendor cache.
        if matching and scoped_missed == len(matching):
            raise ValueError(
                f"Cannot set model '{model}' on '{selector}': it is not served "
                f"by that route's catalog. Use list_models to discover valid "
                f"vendor/route/model values."
            )

        # Fall back to the shared vendor catalog for non-route-scoped routes.
        from .model_cache import get_shared_model_cache
        catalog = get_shared_model_cache().get_any()
        if not catalog:
            return  # no vendor discovery yet → permit (cold-start)
        vendor_models = sorted({
            m.id for m in catalog if m.provider == vendor and m.id
        })
        if not vendor_models:
            # Discovery has no catalog for THIS vendor yet → unknown for this
            # vendor, permit (resolve-time still defends). Only a *populated*
            # vendor catalog can prove a model invalid.
            return
        if model in vendor_models:
            return
        raise ValueError(
            f"Cannot set model '{model}' on '{selector}': it is not served "
            f"by that vendor/route. Available for {vendor}: {vendor_models}. "
            f"Use list_models to discover valid vendor/route/model values."
        )

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

    def effective_request_timeout(
        self, providers: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[float]:
        """Largest LOCAL provider request timeout (seconds), or None.

        Coordinates the timeout layers (#1966): the orchestrator's per-call
        watchdog uses ``max(its default, this)`` so it never fires before a
        slow local model's LLM client would. Only ``is_local`` providers count
        — a local route's configured ``timeout`` (#1957) becomes the single knob.

        ``providers`` should be the **candidate routes for the specific call**
        (from :meth:`resolve_provider_routing`). The watchdog is lifted ONLY when
        EVERY candidate is local: in a mixed set the call may try a cloud route
        first (cloud-primary, local-fallback), and we must not delay
        hang-detection of that cloud call to an unrelated local route's timeout.
        Returns None (no lift, keep the default) for empty or mixed sets.
        Falls back to all providers when omitted.
        """
        src = providers if providers is not None else (self.providers or [])
        if not src or any(not p.get("is_local") for p in src):
            return None
        best: Optional[float] = None
        for provider in src:
            secs = _client_timeout_seconds(getattr(provider.get("client"), "timeout", None))
            if secs is not None and (best is None or secs > best):
                best = secs
        return best

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
    ) -> RoutingResolution:
        """Resolve which routes and model to use for the next LLM call.

        Single source of truth for routing. All call paths funnel through here.

        Returns a :class:`RoutingResolution` that unpacks as the historic
        ``(providers, target_model)`` 2-tuple AND carries ``.meta`` — the
        no-silent-fallback authorization (``explicit_selection`` +
        ``authorized_vendors``) computed in this SAME pass by
        :meth:`StreamingMixin._compute_route_authorization`. The dispatch loops
        consume ``.meta`` instead of re-deriving it, so the two can no longer
        drift (the root cause of the prior edge-bug rounds).

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

        # Normalize the sentinel "auto" to no-override. "auto" expresses
        # routing intent ("pick the default for whichever route runs"), not
        # a model identity, and must never reach a provider client — every
        # vendor 404s when asked to call a model literally named "auto"
        # (#1408). Leaving target_model as None means each route falls
        # back to its own provider["model"] in _try_single_provider.
        if model_override == "auto":
            model_override = None

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
                                chosen = match_list[0]
                                fb_model = fb.get("model")
                                if fb_model:
                                    # Pin THIS fallback's model onto its own
                                    # provider (copy — never mutate the shared
                                    # provider dict). Previously a single global
                                    # target_model from fallbacks[0] was applied
                                    # to every fallback, so any fallback whose
                                    # model differed was rejected by
                                    # _model_available_for_route and became
                                    # unreachable (#1685).
                                    chosen = {**chosen, "model": fb_model}
                                fallback_providers.append(chosen)
                    if fallback_providers:
                        providers_to_use = fallback_providers
                        # Each fallback provider now carries its own model, so
                        # clear the global target_model: the downstream loop
                        # resolves the concrete model per provider via
                        # _resolve_concrete_model(None, provider).
                        target_model = None
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

        meta = self._compute_route_authorization(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        return RoutingResolution(providers_to_use, target_model, meta)

    @staticmethod
    def _provider_supports_embeddings(provider: Dict[str, Any]) -> bool:
        capabilities = provider.get("capabilities") or {}
        return bool(capabilities.get("supports_embeddings"))

    def set_force_local_only_provider(
        self, provider: Optional[Callable[[], bool]]
    ) -> None:
        """Bind a callable that returns the live ``force_local_only`` state.

        The chat path passes ``force_local_only`` explicitly because
        it always has the agent's ``privacy_agent`` in scope. The
        embedding path is invoked from the storage layer (e.g.
        ``AsyncConversationStore.add_conversation``) which does not,
        so without this hook ISOLATED / EPHEMERAL would silently ship
        plaintext to whatever cloud embedding provider sits at the top
        of priority — violating the documented "local LLM only"
        contract (#1492).

        Pass ``None`` to clear; useful in tests.
        """
        self._force_local_only_provider = provider

    def _current_force_local_only(self) -> bool:
        """Read the live ``force_local_only`` state for the embedding path.

        Returns False if no provider is bound (process-local
        ``LLMService()`` not attached to an agent). Fails closed: if
        the provider callable raises, we assume local-only so a
        broken privacy hook doesn't accidentally leak content.
        """
        if self._force_local_only_provider is None:
            return False
        try:
            return bool(self._force_local_only_provider())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "force_local_only provider raised %s; defaulting to "
                "local-only to fail safely.",
                exc,
            )
            return True

    def _lookup_sibling_provider(
        self, sibling_name: str
    ) -> Optional[Dict[str, Any]]:
        """Look up a sibling provider by ``"<vendor>"`` or ``"<vendor>:<route>"``.

        Returns the dict from the *available* providers list (i.e.
        routes that were initialized successfully AND haven't been
        added to ``_disabled_routes`` after a permanent auth failure
        during this session, #655), or ``None`` if no such provider
        matches. ``"<vendor>"`` resolves to the first matching
        available route — explicit ``"<vendor>:<route>"`` pins to
        that exact route.

        Using ``_available_providers()`` instead of raw
        ``self.providers`` mirrors what the chat path does and means
        a sibling route that's already known to have bad credentials
        gets skipped here too — storage falls back to keyword search
        on the embedding side rather than retrying known-bad creds
        on every write.
        """
        # ``_available_providers`` is supplied by the chat-routing
        # path; tests that build LLMService via ``__new__`` may
        # legitimately leave it unimplemented. Fall back to
        # ``self.providers`` in that case so test fixtures don't
        # have to stub the helper.
        if hasattr(self, "_available_providers") and callable(
            getattr(self, "_available_providers")
        ):
            try:
                candidates = self._available_providers()
            except Exception:
                candidates = getattr(self, "providers", None) or []
        else:
            candidates = getattr(self, "providers", None) or []
        if not candidates:
            return None
        if ":" in sibling_name:
            return next(
                (p for p in candidates if p.get("name") == sibling_name),
                None,
            )
        # Vendor-only lookup — first matching route wins.
        return next(
            (p for p in candidates if p.get("vendor") == sibling_name),
            None,
        )

    def _lookup_embedding_route_candidates(
        self, selector: str
    ) -> List[Dict[str, Any]]:
        """Return ALL available providers an embedding_route selector matches.

        Same availability source and selector semantics as
        ``_lookup_sibling_provider`` (``"<vendor>"`` or ``"<vendor>:<route>"``),
        but returns every match instead of the first — the explicit
        embedding_route branch picks the first EMBEDDING-CAPABLE match, so a
        bare-vendor selector isn't defeated by a non-embedding route sorting
        first (codex P2 on #2270).
        """
        if hasattr(self, "_available_providers") and callable(
            getattr(self, "_available_providers")
        ):
            try:
                candidates = self._available_providers()
            except Exception:
                candidates = getattr(self, "providers", None) or []
        else:
            candidates = getattr(self, "providers", None) or []
        if not candidates:
            return []
        if ":" in selector:
            return [p for p in candidates if p.get("name") == selector]
        return [p for p in candidates if p.get("vendor") == selector]

    def resolve_embedding_provider(self) -> Optional[Dict[str, Any]]:
        """Return a provider that can embed text for the active chat session.

        Resolution order (#2263 — top-level embedding_route + #1494 sibling):

        1. Explicit top-level ``embedding_route`` (#2263) is set → resolve it.
           The user's deliberate choice wins even when the active chat route
           embeds natively. This branch is terminal: if the configured route is
           unavailable, refused by the privacy gate, or advertises no embedding
           support, we log and fall back to keyword search (``None``) rather
           than second-guessing with the chat route.
        2. Active chat route supports embeddings → return it. (Today's
           "embedding follows chat provider" behavior — preserved as
           the default when no explicit ``embedding_route`` is set.)
        3. Active route has ``embedding_sibling`` configured AND the
           sibling exists in ``self.providers`` AND it supports
           embeddings AND it passes the ``force_local_only`` filter →
           return the sibling.
        4. Otherwise → ``None``. Storage callers fall back to
           keyword / LIKE search; never an unrelated global Ollama
           singleton, never a cloud route under local-only mode.

        Privacy gate (#1492): the bound ``force_local_only`` provider
        is applied to EVERY branch — the explicit ``embedding_route``,
        the active chat route, AND the sibling lookup. A cloud channel
        for an ISOLATED/EPHEMERAL session is rejected — privacy wins,
        even at the cost of losing embedding for the operator who
        configured a non-local channel.

        Sibling resolution is one hop only. The chosen sibling is
        used directly; ``embedding_sibling`` on the sibling itself is
        intentionally ignored to prevent cycles and to keep "what
        provider embedded this row?" predictable.
        """
        if getattr(self, "disabled", False):
            return None

        # --- 0. Deliberate off-switch: embedding_route == "none" (#2287) -----
        # A first-class "embeddings off" set by the operator. Short-circuit to
        # None (keyword fallback) with NO per-write warning — the single INFO
        # was logged once at set time. This is distinct from the accidental
        # no-provider state below, which KEEPS its warning because that IS
        # degradation, not a choice.
        if getattr(self, "_embedding_route", None) == "none":
            return None

        force_local_only = self._current_force_local_only()

        # --- 1. Explicit top-level embedding_route knob (#2263) --------------
        # The user's deliberate choice wins even when the chat route embeds
        # natively. Terminal branch — never falls through to chat/sibling.
        explicit_route = getattr(self, "_embedding_route", None)
        if explicit_route:
            # Consider EVERY available route the selector matches, not just the
            # vendor's first route (codex P2 on #2270): a bare-vendor selector
            # like "openai" must find openai:api's embedding support even when
            # a non-embedding openai-compatible route sorts first.
            matches = self._lookup_embedding_route_candidates(explicit_route)
            if not matches:
                logger.warning(
                    "Configured embedding_route %s matches no initialized "
                    "provider; semantic storage search will use keyword "
                    "fallback.",
                    explicit_route,
                )
                return None
            if force_local_only:
                matches = [p for p in matches if p.get("is_local")]
                if not matches:
                    logger.info(
                        "Configured embedding_route %s is non-local under "
                        "force_local_only=True; semantic storage search will use "
                        "keyword fallback (privacy mode overrides embedding_route).",
                        explicit_route,
                    )
                    return None
            target = next(
                (p for p in matches if self._provider_supports_embeddings(p)),
                None,
            )
            if target is None:
                logger.warning(
                    "Configured embedding_route %s does not advertise "
                    "supports_embeddings; semantic storage search will use "
                    "keyword fallback.",
                    explicit_route,
                )
                return None
            logger.info(
                "Using configured embedding_route %s for embeddings.",
                explicit_route,
            )
            return target

        # ``resolve_provider_routing`` raises RuntimeError when
        # force_local_only=True and no local provider exists. For
        # embedding the right answer is "fall back to keyword
        # search," not "crash the storage write" — so catch it here.
        try:
            providers_to_use, target_model = self.resolve_provider_routing(
                force_local_only=force_local_only,
            )
        except RuntimeError as exc:
            logger.info(
                "Embedding provider unavailable under force_local_only=%s: %s; "
                "semantic storage search will use keyword fallback.",
                force_local_only,
                exc,
            )
            return None

        if not providers_to_use:
            return None
        provider = next(
            (
                candidate
                for candidate in providers_to_use
                if not target_model
                or target_model == "auto"
                or self._model_available_for_route(candidate, target_model)
            ),
            providers_to_use[0],
        )
        if self._provider_supports_embeddings(provider):
            return provider

        # #1494 — primary route can't embed. Consult sibling if
        # configured.
        sibling_name = provider.get("embedding_sibling")
        if sibling_name:
            sibling = self._lookup_sibling_provider(sibling_name)
            if sibling is None:
                logger.warning(
                    "Active LLM route %s declared embedding_sibling=%r but no "
                    "such initialized provider was found; semantic storage "
                    "search will use keyword fallback.",
                    provider.get("name"),
                    sibling_name,
                )
                return None
            if force_local_only and not sibling.get("is_local"):
                logger.info(
                    "Active LLM route %s declared embedding_sibling=%r but the "
                    "sibling is non-local under force_local_only=True; "
                    "semantic storage search will use keyword fallback "
                    "(privacy mode overrides sibling).",
                    provider.get("name"),
                    sibling_name,
                )
                return None
            if not self._provider_supports_embeddings(sibling):
                logger.warning(
                    "Active LLM route %s declared embedding_sibling=%r but the "
                    "sibling does not advertise supports_embeddings; semantic "
                    "storage search will use keyword fallback.",
                    provider.get("name"),
                    sibling_name,
                )
                return None
            logger.info(
                "Active LLM route %s has no embedding capability; using "
                "configured sibling %s for embeddings.",
                provider.get("name"),
                sibling.get("name"),
            )
            return sibling

        logger.info(
            "Active LLM route %s does not support embeddings; semantic "
            "storage search will use keyword fallback.",
            provider.get("name"),
        )
        return None

    def _pin_for_provider(
        self, provider: Optional[Dict[str, Any]]
    ) -> Optional["EmbeddingSpacePin"]:
        """Return the shared-space pin whose members include this provider.

        Matches on the provider's ``"<vendor>:<route>"`` name (or bare vendor).
        Returns ``None`` when no pin covers the route — the common case.
        """
        pins = getattr(self, "_embedding_space_pins", None)
        if not provider or not pins:
            return None
        name = provider.get("name")
        vendor = provider.get("vendor")
        for pin in pins:
            if pin.covers(name, vendor):
                return pin
        return None

    def _check_route_pin_space_coherent(
        self, route: str, model: str, dim: Optional[int]
    ) -> None:
        """Refuse a route pin that would fragment a VERIFIED shared space (#2440).

        A route that is a member of a shared-space pin whose parity probe has
        PASSED (#2290/#2376) is locked to the space's model/dim. Pinning a
        different model — or the same model at a different dim — would fragment
        that verified space, so it is refused with an
        :class:`EmbeddingSpaceConflictError` (mapped to a 409 by the endpoint).
        Unverified pins do NOT constrain a member: only a verified space wins,
        which is exactly the precedence the resolver already applies.
        """
        provider = self._find_route_provider(route)
        pin = self._pin_for_provider(provider)
        if pin is None:
            return
        parity = self._verified_space_pins.get(pin.name)
        if not (parity and parity.passed):
            return
        # Compare on the vendor-neutral identity (#2440): the shared-space stack
        # treats a route-native alias and the pin's slug as the SAME model
        # (``qwen3-embedding:8b`` ⇄ ``qwen/qwen3-embedding-8b``). Universal setup
        # pins each member's own route slug, so a raw string compare would
        # falsely 409 a no-op re-pin / rollback against the verified space.
        from .embedding_discovery import normalize_embedding_model_id

        model_conflict = normalize_embedding_model_id(
            model
        ) != normalize_embedding_model_id(pin.model or "")
        dim_conflict = (
            dim is not None and pin.dim is not None and int(dim) != int(pin.dim)
        )
        if not (model_conflict or dim_conflict):
            return
        raise EmbeddingSpaceConflictError(
            f"route {route!r} is a member of verified shared space "
            f"{pin.name!r} ({pin.model}@{pin.dim}) — clear the space or choose "
            f"its model; pinning {model.strip()} would fragment the space."
        )

    def _shared_space_payload(
        self, provider: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Build the #2290 shared-space echo for ``provider``'s route, or ``None``.

        Route-scoped: only the pin covering THIS provider is described, so a
        per-route echo (#2372) never reports the globally-active route's space.
        """
        pin = self._pin_for_provider(provider)
        if pin is None:
            return None
        parity = self._verified_space_pins.get(pin.name)
        return {
            "name": pin.name,
            "space_id": pin.space_id,
            "model": pin.model,
            "dim": pin.dim,
            "members": list(pin.members),
            "verified": bool(parity and parity.passed),
            "parity": parity.to_dict() if parity else None,
        }

    def get_embedding_service(self):
        """Return a provider-backed embedding service for the active route.

        When the resolved route is a member of a shared-space pin (#2290) whose
        parity probe has PASSED, the pin's model-identity ``space_id`` is force-
        applied so this route's rows land in the same coordinate space as its
        sibling members (local ⇆ cloud). An unverified pin is NOT applied — the
        route keeps its own route-scoped space id, so "detected" never becomes
        "aliased" without the mandatory parity gate.
        """
        provider = self.resolve_embedding_provider()
        return self._build_embedding_service(provider)

    def get_embedding_service_for_route(self, route: str):
        """Return a provider-backed embedding service for a SPECIFIC route (#2372).

        The route-model echo must count stale rows against the route the operator
        just configured, not the globally-resolved active route — a response that
        says ``resolved_route: openrouter:api`` must never report
        ``ollama:local``'s stale-row counts. Applies the same shared-space pin
        logic as :meth:`get_embedding_service`. Returns ``None`` when the route is
        unknown / not configured.
        """
        provider = self._find_route_provider(route) if route else None
        return self._build_embedding_service(provider)

    def _build_embedding_service(self, provider: Optional[Dict[str, Any]]):
        """Construct a :class:`ProviderEmbeddingService` for a resolved provider.

        Shared by the active-route and per-route (#2372) accessors so the pin /
        parity-gate handling stays identical. A verified #2290 shared-space pin
        force-applies its model-identity ``space_id``; an unverified pin is not
        applied. Returns ``None`` when ``provider`` is ``None``.
        """
        if provider is None:
            return None
        from .embedding_service import ProviderEmbeddingService

        pin = self._pin_for_provider(provider)
        verified = (
            pin is not None
            and self._verified_space_pins.get(pin.name) is not None
            and self._verified_space_pins[pin.name].passed
        )
        if pin is not None and verified:
            return ProviderEmbeddingService(
                provider,
                space_id_override=pin.space_id,
                normalized_override=pin.normalized,
            )
        return ProviderEmbeddingService(provider)

    async def verify_embedding_space_parity(
        self,
        pin_name: Optional[str] = None,
        *,
        record_to: Any = None,
    ) -> Dict[str, "ParityResult"]:
        """Run the mandatory parity probe for shared-space pins (#2290).

        For each pin (or just ``pin_name``), embed K canary texts through every
        pair of member routes and require pairwise cosine ``>=
        parity_threshold``. On pass, the pin is cached in
        ``self._verified_space_pins`` and its model-identity ``space_id`` starts
        being applied by ``get_embedding_service``; on fail, the pin is left
        unverified so members keep their route-scoped ids (the alias is
        refused). Also enforces the dims pin: a member whose configured
        ``embedding_dim`` differs from the pin's ``dim`` fails the probe — both
        sides must pin the SAME dims value.

        ``record_to`` (an ``AsyncDatabase``) is best-effort: when provided, the
        measured drift is written onto the pinned space's ``embedding_profiles``
        row so operators can see it. Failure to record never fails the probe.

        Returns a ``{pin_name: ParityResult}`` map for the pins probed.
        """
        from .embedding_space import ParityResult, probe_parity
        from .embedding_service import ProviderEmbeddingService

        results: Dict[str, "ParityResult"] = {}
        pins = self._embedding_space_pins
        if pin_name is not None:
            pins = [p for p in pins if p.name == pin_name]
        for pin in pins:
            member_services = []
            dim_mismatch = None
            for selector in pin.members:
                candidates = self._lookup_embedding_route_candidates(selector)
                target = next(
                    (c for c in candidates if self._provider_supports_embeddings(c)),
                    None,
                )
                if target is None:
                    continue
                caps = target.get("capabilities") or {}
                member_dim = caps.get("embedding_dim")
                if member_dim is not None and int(member_dim) != int(pin.dim):
                    dim_mismatch = (
                        f"member {selector} serves dim {member_dim} but pin "
                        f"{pin.name!r} declares dim {pin.dim} — both sides must "
                        "pin the SAME dims value"
                    )
                    break
                member_services.append(ProviderEmbeddingService(target))

            if dim_mismatch is not None:
                result = ParityResult(
                    passed=False, threshold=pin.parity_threshold,
                    min_cosine=0.0, mean_cosine=0.0, n=0, error=dim_mismatch,
                )
            elif len(member_services) < 2:
                result = ParityResult(
                    passed=False, threshold=pin.parity_threshold,
                    min_cosine=0.0, mean_cosine=0.0, n=0,
                    error=(
                        f"fewer than two embedding-capable member routes "
                        f"available for pin {pin.name!r}"
                    ),
                )
            else:
                # Probe every member against the first — one hub is enough to
                # transitively certify a shared space; the worst pair wins.
                hub = member_services[0]
                pair_results = []
                for other in member_services[1:]:
                    pair_results.append(
                        await probe_parity(
                            hub, other, threshold=pin.parity_threshold
                        )
                    )
                worst = min(pair_results, key=lambda r: r.min_cosine)
                result = worst

            results[pin.name] = result
            if result.passed:
                self._verified_space_pins[pin.name] = result
                logger.info(
                    "Shared embedding space %r verified: %d members share "
                    "space_id %s (min cosine %.4f >= %.2f).",
                    pin.name, len(member_services), pin.space_id,
                    result.min_cosine, pin.parity_threshold,
                )
                if record_to is not None:
                    await self._record_space_parity(record_to, pin, result)
            else:
                self._verified_space_pins.pop(pin.name, None)
                logger.warning(
                    "Shared embedding space %r REFUSED (parity below "
                    "threshold): %s. Member routes keep their own space ids.",
                    pin.name, result.error or f"min cosine {result.min_cosine}",
                )
        return results

    async def _record_space_parity(
        self, db: Any, pin: "EmbeddingSpacePin", result: "ParityResult"
    ) -> None:
        """Best-effort: durably persist measured drift for the pinned space.

        Upserts the canonical shared-space registry row so the parity survives
        a restart (see :meth:`hydrate_verified_space_pins`), even though the
        verify probe usually runs before any shared-space rows exist.
        """
        try:
            from kestrel_sovereign.storage.sqla.embedding_profile import (
                record_space_parity,
            )

            await record_space_parity(
                db,
                space_id=pin.space_id,
                model=pin.model,
                dim=pin.dim,
                normalized=pin.normalized,
                parity_cosine=result.min_cosine,
            )
        except Exception as exc:  # pragma: no cover - defensive, never fatal
            logger.debug("Recording embedding-space parity failed: %s", exc)

    async def hydrate_verified_space_pins(self, db: Any) -> None:
        """Re-apply previously-verified shared spaces from persisted parity (#2290).

        ``_verified_space_pins`` is process-local, so after a restart the pins
        parse again but no shared ``space_id`` would be applied until an operator
        re-POSTs the parity probe — silently stranding reindexed shared-space
        rows outside kNN. This hydrates that state from the durable
        ``embedding_profiles.parity_cosine`` written by :meth:`_record_space_parity`.

        A pin is re-verified only when its persisted parity still clears the
        pin's *current* ``parity_threshold`` — so raising the threshold (or
        changing the model/dim, which changes ``space_id`` and therefore the
        looked-up row) correctly invalidates a stale alias. Best-effort: any DB
        error leaves the pin unverified rather than crashing startup.
        """
        pins = getattr(self, "_embedding_space_pins", None)
        if db is None or not pins:
            return
        from .embedding_space import ParityResult

        for pin in pins:
            try:
                row = await db.fetchone(
                    "SELECT parity_cosine FROM embedding_profiles "
                    "WHERE space_id = ? AND parity_cosine IS NOT NULL "
                    "ORDER BY parity_cosine DESC LIMIT 1",
                    (pin.space_id,),
                )
            except Exception as exc:
                logger.debug(
                    "Hydrating parity for space %s failed: %s", pin.space_id, exc
                )
                continue
            if not row or row[0] is None:
                continue
            parity_cosine = float(row[0])
            if parity_cosine >= pin.parity_threshold:
                self._verified_space_pins[pin.name] = ParityResult(
                    passed=True,
                    threshold=pin.parity_threshold,
                    min_cosine=round(parity_cosine, 6),
                    mean_cosine=round(parity_cosine, 6),
                    n=0,
                )
                logger.info(
                    "Shared embedding space %r rehydrated from persisted parity "
                    "(cosine %.4f >= %.2f); shared space_id %s active.",
                    pin.name, parity_cosine, pin.parity_threshold, pin.space_id,
                )
            else:
                self._verified_space_pins.pop(pin.name, None)

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
            raw_capabilities = getattr(provider, "capabilities", None)
            if raw_capabilities is None and hasattr(provider.adapter, "provider_capabilities"):
                raw_capabilities = provider.adapter.provider_capabilities()
            if isinstance(raw_capabilities, dict):
                raw_capabilities = ProviderCapabilities.from_mapping(raw_capabilities)
            if not isinstance(raw_capabilities, ProviderCapabilities):
                raw_capabilities = ProviderCapabilities()
            capabilities = raw_capabilities.to_dict()
            answerability_gate = getattr(
                provider, "_kestrel_embedding_answerability_gate", None
            )
            if answerability_gate is not None:
                capabilities["embedding_answerability_gate"] = answerability_gate
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
                # #1954 per-route reasoning effort (llama.cpp local models),
                # stashed as a private attr in ``ProviderRegistry._build_route``.
                "reasoning_effort": getattr(provider, "_kestrel_reasoning_effort", None),
                "selection_hints": hints,
                "capabilities": capabilities,
                # #1494 sibling string carried over the SDK boundary
                # via a private attr in ``ProviderRegistry._build_route``.
                # Default ``None`` for entry-point providers and any
                # ``ProviderInfo`` constructed by third-party plugins
                # that don't set the attr.
                "embedding_sibling": getattr(
                    provider, "_kestrel_embedding_sibling", None
                ),
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

    async def finalize_providers(self, host_db: "Optional[AsyncDatabase]" = None) -> None:
        """Async completion pass for routes sync init couldn't bring up.

        Sync ``__init__`` builds the registry via ``initialize_providers()``,
        but some routes need an async step to register (e.g. an OpenRouter
        route configured with only ``OPENROUTER_MANAGEMENT_API_KEY``, which is
        completed by minting a bootstrap child key). Called once from the
        agent's async ``initialize()``. Safe to call multiple times.

        ``host_db`` (when the caller has a host-level store) lets the registry
        persist and reuse the bootstrap child key across restarts instead of
        minting a new one every cold start.
        """
        registry = getattr(self, "provider_registry", None)
        if registry is None or not hasattr(registry, "finalize_providers"):
            return
        before = {p.get("name") for p in (self.providers or [])}
        try:
            provider_infos = await registry.finalize_providers(host_db=host_db)
        except Exception as e:  # noqa: BLE001 - never block startup on this
            logger.warning("finalize_providers failed: %s", e)
            return
        self.providers = self._convert_providers_format(provider_infos)
        after = {p.get("name") for p in self.providers}
        # A late-registered route (e.g. an OpenRouter bootstrap route minted
        # here from a management key) is absent from any model-discovery
        # snapshot taken before this point. If discovery already populated the
        # shared cache, subsequent ``discover_all_models(use_cache=True)`` calls
        # hit that pre-finalize snapshot — the new vendor never gets a
        # ``/models`` query, ``by_vendor`` omits it, and its ``model="auto"``
        # route stays unresolved (#2247). Drop the stale cache so the next
        # discovery re-runs over the now-complete provider list.
        if after - before:
            try:
                from .model_cache import get_shared_model_cache

                get_shared_model_cache().clear()
            except Exception as e:  # noqa: BLE001 - cache clear is best-effort
                logger.debug("Could not clear model cache after finalize: %s", e)

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

        The callback MAY additionally accept an optional ``cost`` keyword
        (the provider-reported per-call cost in USD, e.g. OpenRouter
        ``usage.cost``; ``None`` when the provider does not report one).
        Callbacks that don't declare ``cost`` (or ``**kwargs``) are still
        called with the original signature — see #1806.

        Args:
            callback: Async function to call after each LLM call
        """
        self._metering_callback = callback
        # Detect whether the callback opts into the optional per-call cost
        # (#1806) so we stay backward-compatible with callbacks written
        # against the original (provider, model, tokens) signature.
        accepts_cost = False
        try:
            params = inspect.signature(callback).parameters
            accepts_cost = "cost" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (ValueError, TypeError):
            accepts_cost = False
        self._metering_callback_accepts_cost = accepts_cost
        logger.info("LLM metering enabled (per-call cost: %s)", accepts_cost)

    def set_observability_context(
        self,
        session_id: Optional[str] = None,
        companion_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Set task-local context for legacy observability callers.

        New generation call sites should pass an immutable
        :class:`LLMInvocationContext` directly.  This compatibility API now
        uses a ``ContextVar`` so two concurrent requests cannot overwrite one
        another's identity.

        Args:
            session_id: A2A session ID
            companion_id: Companion UUID
            user_id: User UUID
        """
        self._context_state().set(
            self._make_invocation_context(
                session_id=session_id, companion_id=companion_id, user_id=user_id
            )
        )

    def reset_observability_context(self) -> None:
        """Clear legacy ambient identity for the current task."""

        self._context_state().reset()

    @contextmanager
    def observability_context(
        self,
        session_id: Optional[str] = None,
        companion_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Scope legacy observability identity and restore it on exit."""

        context = self._make_invocation_context(
            session_id=session_id, companion_id=companion_id, user_id=user_id
        )
        with self._context_state().scope(context):
            yield context

    def _context_state(self) -> LLMInvocationContextState:
        """Return per-service context state, lazily for ``__new__`` test hosts."""

        state = getattr(self, "_invocation_context_state", None)
        if state is None:
            state = LLMInvocationContextState()
            self._invocation_context_state = state
        return state

    @staticmethod
    def _make_invocation_context(
        *,
        session_id: Optional[str] = None,
        companion_id: Optional[str] = None,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> LLMInvocationContext:
        return LLMInvocationContext(
            session_id=session_id,
            companion_id=companion_id,
            user_id=user_id,
            correlation_id=correlation_id,
        )

    def _resolve_invocation_context(
        self,
        context: Optional[LLMInvocationContext] = None,
        *,
        session_id: Optional[str] = None,
    ) -> LLMInvocationContext:
        """Capture one request identity before a provider call begins."""

        state = self._context_state()
        resolved = resolve_invocation_context(
            context,
            ambient=state.get(),
            session_id=session_id,
        )
        if resolved.companion_id and resolved.user_id:
            return resolved

        # Cross-task compatibility (#2569): the LLM call may be dispatched from
        # a task that never inherited the setter's ContextVar (a worker spawned
        # before the request set its identity, a sibling task tree, or a thread
        # offload), so the task-local ambient came back empty.  ``session_id``
        # threads through explicitly as a generation argument, so recover this
        # session's companion/user without a process-wide last-writer snapshot
        # that would misattribute a concurrent tenant's billing identity.
        session_context = state.get_for_session(resolved.session_id)
        if session_context is None:
            return resolved
        return LLMInvocationContext(
            session_id=resolved.session_id,
            companion_id=resolved.companion_id or session_context.companion_id,
            user_id=resolved.user_id or session_context.user_id,
            correlation_id=(
                resolved.correlation_id or session_context.correlation_id
            ),
            # #2674 finding 3: preserve the content-redaction flag across the
            # cross-task session merge — dropping it here would re-expose the
            # withheld prompt/response in telemetry on the #2569 recovery path.
            redact_content=bool(
                resolved.redact_content or session_context.redact_content
            ),
        )

    @staticmethod
    def _extract_provider_cost(response: Any) -> Optional[float]:
        """Best-effort per-call cost reported by the provider (#1806).

        OpenRouter returns an exact ``usage.cost`` (USD) per generation when
        the request opts in (``usage: {include: true}`` — the OpenRouter
        adapter sets this). The value rides on the provider-native ``raw``
        object rather than a typed SDK field, so we read it defensively and
        return ``None`` for providers that don't report a cost.
        """
        raw = getattr(response, "raw", None)
        if raw is None:
            return None
        # Streaming adapters stash the cost directly on the raw dict
        # (the provider usage object isn't retained past the stream).
        if isinstance(raw, dict) and raw.get("cost") is not None:
            try:
                return float(raw["cost"])
            except (TypeError, ValueError):
                return None
        usage = getattr(raw, "usage", None)
        if usage is None and isinstance(raw, dict):
            usage = raw.get("usage")
        if usage is None:
            return None
        cost = getattr(usage, "cost", None)
        if cost is None:
            # openai-python v2 stashes unknown response fields (OpenRouter's
            # ``cost`` extension) on the pydantic model's ``model_extra``.
            extra = getattr(usage, "model_extra", None)
            if isinstance(extra, dict):
                cost = extra.get("cost")
            elif isinstance(usage, dict):
                cost = usage.get("cost")
        try:
            return float(cost) if cost is not None else None
        except (TypeError, ValueError):
            return None

    async def _finalize_invocation(
        self,
        response: Any,
        provider_name: str,
        model: str,
        *,
        success: bool,
        path: str,
        invocation_context: LLMInvocationContext,
        duration_ms: int = 0,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        force_local_only: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools_used: Optional[bool] = None,
        error_message: Optional[str] = None,
        publish_identity: bool = True,
        usage_available: Optional[bool] = None,
        track_model_usage: Optional[bool] = None,
    ) -> None:
        """Record one provider attempt while isolating every telemetry sink.

        Ordinary telemetry failures are independent: a usage-DB outage cannot
        suppress the billing callback, and an observability-store outage cannot
        suppress Prometheus or the structured usage line.  Cancellation is
        deliberately never swallowed.
        """

        if publish_identity and success:
            self._stamp_response_identity(
                response,
                model=model,
                provider=provider_name,
            )

        input_tokens = output_tokens = total_tokens = None
        cache_creation_input_tokens = cache_read_input_tokens = None
        if isinstance(response, LLMResponse):
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            total_tokens = response.total_tokens
            cache_creation_input_tokens = response.cache_creation_input_tokens
            cache_read_input_tokens = response.cache_read_input_tokens
        if usage_available is None:
            usage_available = response_usage_available(response)
        if track_model_usage is None:
            track_model_usage = usage_available

        try:
            response_text = (
                response.content
                if isinstance(response, LLMResponse)
                else (str(response) if response is not None else None)
            )
        except Exception as exc:  # noqa: BLE001 - response text is optional telemetry
            response_text = None
            logger.warning("Could not render %s response for telemetry: %s", path, exc)

        tool_calls_data = None
        if isinstance(response, LLMResponse):
            try:
                if response.tool_calls:
                    tool_calls_data = [
                        {"name": tool_call.name, "arguments": tool_call.arguments}
                        for tool_call in response.tool_calls
                    ]
                else:
                    executed = getattr(response, "executed_tool_calls", None)
                    if executed:
                        tool_calls_data = [
                            {"name": call["name"], "arguments": call["arguments"]}
                            for call in executed
                        ]
            except Exception as exc:  # noqa: BLE001 - tool details are optional
                logger.warning("Could not normalize %s tool telemetry: %s", path, exc)

        cost = self._extract_provider_cost(response)
        record_metadata = dict(metadata or {})
        record_metadata.setdefault("path", path)
        if force_local_only is not None:
            record_metadata.setdefault("force_local_only", force_local_only)
        if invocation_context.correlation_id:
            record_metadata.setdefault(
                "correlation_id", invocation_context.correlation_id
            )
        if cost is not None:
            record_metadata.setdefault("provider_reported_cost_usd", cost)
        if not usage_available:
            record_metadata.setdefault("usage_available", False)

        usage_tracker_ready = (
            hasattr(self, "_db_initialized")
            or "_track_model_usage" in getattr(self, "__dict__", {})
        )
        tracked_tokens = total_tokens
        if tracked_tokens is None and (
            input_tokens is not None or output_tokens is not None
        ):
            tracked_tokens = (input_tokens or 0) + (output_tokens or 0)
        if track_model_usage and usage_tracker_ready and tracked_tokens is not None:
            try:
                await self._track_model_usage(
                    model, provider_name, tokens=tracked_tokens
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - independent best-effort sink
                logger.warning("Usage DB failed for %s LLM invocation: %s", path, exc)

        await self._log_llm_call(
            provider=provider_name,
            model=model,
            duration_ms=duration_ms,
            success=success,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response_text,
            error_message=error_message,
            tool_calls=tool_calls_data,
            metadata=record_metadata,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            tools_used=(tools is not None if tools_used is None else tools_used),
            structured_output=response_format is not None,
            cost=cost,
            usage_available=usage_available,
            invocation_context=invocation_context,
        )

    async def _finalize_successful_invocation(
        self,
        response: Any,
        provider_name: str,
        model: str,
        *,
        path: str,
        invocation_context: LLMInvocationContext,
        duration_ms: int = 0,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        force_local_only: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools_used: Optional[bool] = None,
        publish_identity: bool = True,
        usage_available: Optional[bool] = None,
    ) -> None:
        """Finalize one successful provider attempt exactly once."""

        await self._finalize_invocation(
            response,
            provider_name,
            model,
            success=True,
            path=path,
            invocation_context=invocation_context,
            duration_ms=duration_ms,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            response_format=response_format,
            force_local_only=force_local_only,
            metadata=metadata,
            tools_used=tools_used,
            publish_identity=publish_identity,
            usage_available=usage_available,
            track_model_usage=None,
        )

    async def _finalize_failed_invocation(
        self,
        provider_name: str,
        model: str,
        *,
        path: str,
        invocation_context: LLMInvocationContext,
        duration_ms: int,
        error: Exception,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        force_local_only: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools_used: Optional[bool] = None,
        response: Any = None,
        usage_available: bool = False,
        track_model_usage: bool = False,
    ) -> None:
        """Record one failed provider attempt without publishing its identity."""

        await self._finalize_invocation(
            response,
            provider_name,
            model,
            success=False,
            path=path,
            invocation_context=invocation_context,
            duration_ms=duration_ms,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            response_format=response_format,
            force_local_only=force_local_only,
            metadata=metadata,
            tools_used=tools_used,
            error_message=str(error),
            publish_identity=False,
            usage_available=usage_available,
            track_model_usage=track_model_usage,
        )

    async def _run_provider_attempt(
        self,
        attempt: Awaitable[Any],
        provider_name: str,
        model: str,
        *,
        path: str,
        invocation_context: LLMInvocationContext,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        force_local_only: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools_used: Optional[bool] = None,
        publish_identity: bool = True,
        error_message_override: Optional[str] = None,
    ) -> Any:
        """Await and finalize one provider call, successful or failed."""

        started = time.monotonic()
        try:
            response = await attempt
        except asyncio.CancelledError:
            # No usage evidence exists on a non-streaming cancelled call.  Keep
            # the historical cancellation contract and do not fabricate a row.
            raise
        except Exception as exc:
            await self._finalize_failed_invocation(
                provider_name,
                model,
                path=path,
                invocation_context=invocation_context,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=(
                    LLMServiceError(error_message_override)
                    if error_message_override is not None
                    else exc
                ),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                response_format=response_format,
                force_local_only=force_local_only,
                metadata=metadata,
                tools_used=tools_used,
            )
            raise

        await self._finalize_successful_invocation(
            response,
            provider_name,
            model,
            path=path,
            invocation_context=invocation_context,
            duration_ms=int((time.monotonic() - started) * 1000),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            response_format=response_format,
            force_local_only=force_local_only,
            metadata=metadata,
            tools_used=tools_used,
            publish_identity=publish_identity,
        )
        return response

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
        cost: Optional[float] = None,
        usage_available: bool = True,
        invocation_context: Optional[LLMInvocationContext] = None,
    ) -> None:
        """Log an LLM call to the observability store (if configured).

        This is called automatically by get_response() and generate().
        Also triggers metering callback for billing (Vending Machine).

        Also emits a single structured ``llm.usage:`` INFO line so callers
        that downcast the response to a plain string don't lose token /
        cache telemetry. Picked up by Cloud Run / Cloud Logging via the
        multi_agent stdout tee (issue #812). See issue #819.
        """
        # A supplied context is already the frozen invocation snapshot.  Never
        # merge it with ambient state again after the provider await.
        context = (
            invocation_context
            if invocation_context is not None
            else self._resolve_invocation_context()
        )
        # #2674 finding 3: under an enforcing (strict) response audit the turn's
        # assistant prose is WITHHELD from the user pending the verdict, so its
        # raw prompt/response must not land in DURABLE ``llm_calls`` telemetry
        # before (or, on a DENY, ever after) that verdict. Blank the two
        # content columns to a length-tagged marker while preserving every
        # content-free field — usage, timing, provider, model, cost, error. The
        # audit provider call carries the same flag because its ``user_prompt``
        # IS the withheld prose. Advisory / no-audit turns leave this False and
        # keep the full preview unchanged.
        record_metadata = dict(metadata or {})
        if getattr(context, "redact_content", False):
            if user_prompt is not None:
                user_prompt = _redacted_content_marker(user_prompt)
            if response is not None:
                response = _redacted_content_marker(response)
            # #2674 finding 4: ``tool_calls`` and ``error_message`` are the two
            # remaining content-bearing columns on the same durable ``llm_calls``
            # row. ``tool_calls`` is RESPONSE-DERIVED model output — the model's
            # own tool arguments, which can echo the withheld prose verbatim — so
            # under an enforcing audit they must not persist raw before (or, on a
            # DENY, ever after) the verdict. ``error_message`` is ``str(error)``,
            # and provider/adapter exceptions routinely embed the prompt or the
            # response body (the endpoint-leak reproductions prove they are
            # untrusted). Redact BOTH: keep each tool's NAME (a safe classification
            # of which tools ran) but blank its arguments, and reduce the error to
            # its exception class/category. Every content-free field — usage,
            # timing, provider, model, cost — is preserved untouched. This is
            # provider RESPONSE telemetry (the ``llm_calls`` row); the separate
            # operator tool-DISPATCH audit (``log_tool_call``) records that a tool
            # actually executed and is a distinct trust boundary left intact.
            tool_calls = _redact_tool_calls_content(tool_calls)
            error_message = _redact_error_content(error_message)
            record_metadata.setdefault("content_redacted", True)
        if context.correlation_id:
            record_metadata.setdefault("correlation_id", context.correlation_id)
        if cost is not None:
            record_metadata.setdefault("provider_reported_cost_usd", cost)
        if not usage_available:
            record_metadata.setdefault("usage_available", False)

        # Log to observability store
        observability_store = getattr(self, "_observability_store", None)
        if observability_store:
            try:
                await observability_store.log_llm_call(
                    provider=provider,
                    model=model,
                    duration_ms=duration_ms,
                    success=success,
                    agent_did=getattr(self, "_owner_agent_did", None),
                    session_id=context.session_id,
                    companion_id=context.companion_id,
                    user_id=context.user_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=response,
                    error_message=error_message,
                    tool_calls=tool_calls,
                    metadata=record_metadata,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - independent best-effort sink
                logger.warning("Observability store failed for LLM call: %s", exc)

        # Prometheus metrics (no-op when prometheus-client not installed)
        try:
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
                    LLM_TOKENS.labels(model=model, direction="input").inc(
                        input_tokens
                    )
                if output_tokens is not None:
                    LLM_TOKENS.labels(model=model, direction="output").inc(
                        output_tokens
                    )
        except Exception as exc:  # noqa: BLE001 - independent best-effort sink
            logger.warning("Prometheus metrics failed for LLM call: %s", exc)

        # Trigger metering callback for billing (Phase 1: tracking only)
        metering_callback = getattr(self, "_metering_callback", None)
        billable_breakdown_available = (
            input_tokens is not None or output_tokens is not None
        )
        if (
            metering_callback
            and success
            and usage_available
            and billable_breakdown_available
        ):
            companion_id = context.companion_id
            user_id = context.user_id

            if companion_id and user_id:
                meter_kwargs = dict(
                    companion_id=companion_id,
                    user_id=user_id,
                    provider=provider,
                    model=model,
                    prompt_tokens=input_tokens or 0,
                    completion_tokens=output_tokens or 0,
                )
                # Only pass the provider-reported per-call cost (#1806) to
                # callbacks that opted in; keeps the original signature
                # working for callbacks that don't declare ``cost``.
                if getattr(self, "_metering_callback_accepts_cost", False):
                    meter_kwargs["cost"] = cost
                try:
                    await metering_callback(**meter_kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - independent billing sink
                    logger.warning("LLM metering callback failed: %s", exc)

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
                "cost": cost,
                "usage_available": usage_available,
                "session_id": context.session_id,
                "correlation_id": context.correlation_id,
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

    @staticmethod
    def _scrub_auto(model_override: Optional[str]) -> Optional[str]:
        """Normalize the ``"auto"`` sentinel to ``None``. Used by entry
        points that bypass :meth:`resolve_provider_routing` — including the
        inference-lease route. See #1408 for why ``"auto"`` must never reach
        a provider client."""
        return None if model_override == "auto" else model_override

    def _resolve_concrete_model(
        self,
        target_model: Optional[str],
        provider: Dict[str, Any],
    ) -> str:
        """Resolve the concrete model id to send to a provider client.

        Single source of truth for the "auto" sentinel scrub (#1408). Both
        the non-streaming ``_try_single_provider`` and the streaming paths
        in ``streaming.py`` funnel here so neither can leak ``"auto"`` to
        the wire.

        Resolution order:
          1. ``target_model`` if it is a concrete model id (not None and
             not the ``"auto"`` sentinel).
          2. ``provider["model"]`` if it is concrete.
          3. ``resolve_provider_default(provider["name"])`` — lazy resolve
             from kestrel.toml + cached discovery (covers the fresh
             quickstart case where the route default is still ``"auto"``).
        """
        if target_model and target_model != "auto":
            return target_model
        route_model = provider.get("model")
        if route_model and route_model != "auto":
            return route_model
        # Route-scoped routes (e.g. codex/openai:plan) own their serveable
        # catalog and must NEVER fall back to the vendor discovery cache —
        # that cache can hold API-only models the route can't serve (e.g.
        # gpt-5.5-pro). When such a route is still "auto" (its own catalog
        # didn't resolve a concrete model, e.g. before codex's models_cache
        # exists), pass "auto" through so the adapter sends no model and the
        # substrate uses its own serveable default, rather than a vendor model.
        # Ensure route catalogs exist first — on a fresh cache-less startup
        # discovery may not have populated them yet, and this fallback can run
        # before discovery. _ensure_route_catalogs_sync registers route-specific
        # routes (even empty) without consulting the vendor cache.
        if hasattr(self, "_ensure_route_catalogs_sync"):
            try:
                self._ensure_route_catalogs_sync()
            except Exception:  # pragma: no cover - never block resolution
                pass
        route_catalogs = getattr(self, "_route_catalogs", None) or {}
        if provider.get("name") in route_catalogs:
            return "auto"
        from .model_selection import resolve_provider_default
        try:
            return resolve_provider_default(provider["name"])
        except ValueError as exc:
            # Route is misconfigured (model="auto" + empty discovery) AND
            # the caller didn't supply an override. Refuse to send "auto"
            # downstream — that's the #1408 bug we're fixing, no soft
            # passthrough. Raise ModelNotAvailableForRoute so the outer
            # fallback loop skips this route and tries the next; if every
            # route is in this state the loop aggregates them into
            # LLMAllProvidersFailedError which names each route's reason.
            logger.warning(
                "Skipping route %r: configured as model='auto' and "
                "discovery cache is empty. Run model discovery or set "
                "a concrete `model` in kestrel.toml to fix. Underlying: %s",
                provider["name"], exc,
            )
            raise ModelNotAvailableForRoute(
                vendor=provider.get("vendor"),
                route=provider.get("route"),
                model="auto",
            ) from exc

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
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        cancel_token: Optional[CancelToken] = None,
        explicit_selection: bool = False,
        invocation_context: Optional[LLMInvocationContext] = None,
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

        The catalog gate only guards **blind fallback**. When the caller has
        explicitly pinned this route (``explicit_selection=True`` — a
        ``vendor:route/model`` override or a route-qualified mandate, as feature
        subagents produce), the streaming chat path (``streaming.py`` never calls
        ``_model_available_for_route``) already trusts the selection and calls the
        route directly. This non-streaming path must match that contract, or a
        valid model the route genuinely serves — but which discovery hasn't
        cached (e.g. a brand-new OpenRouter slug, or a paginated/capped catalog) —
        gets a false-negative ``ModelNotAvailableForRoute`` and the whole call
        fails with "All providers failed". That asymmetry is what broke feature
        subagent dispatch (#2352): the same ``openrouter:api/openai/...`` route
        that streamed fine on the chat path was rejected here. There is no
        cross-vendor cascade risk when the route is explicitly pinned, so honour
        it exactly as streaming does.

        EXCEPTION: local vendors whose server ignores the requested model ID and
        serves whatever weights are loaded (e.g. ``llama_cpp``, ``ollama:local``
        with a preloaded model) — trusting an explicit selection there would
        silently mislabel the response (codex round-2 P2). Keep catalog
        validation on those so a mismatched pin fails loud instead of being
        metered as the wrong model.
        """
        messages = messages_for(provider["adapter"], user_prompt=user_prompt, system_prompt=system_prompt)

        skip_catalog = explicit_selection and provider.get("vendor") not in _MODEL_IGNORING_VENDORS
        if target_model and target_model != "auto" and not skip_catalog:
            if not self._model_available_for_route(provider, target_model):
                raise ModelNotAvailableForRoute(
                    vendor=provider.get("vendor"),
                    route=provider.get("route"),
                    model=target_model,
                )
        model_to_use = self._resolve_concrete_model(target_model, provider)

        frozen_context = (
            invocation_context
            if invocation_context is not None
            else self._resolve_invocation_context()
        )
        response = await self._run_provider_attempt(
            provider["adapter"].get_response(
                client=provider["client"],
                model=model_to_use,
                messages=messages,
                tools=tools,
                response_format=response_format,
                extra_body=provider_cache_body(provider),
                tool_executor=tool_executor,
                cancel_token=cancel_token,
            ),
            provider["name"],
            model_to_use,
            path="get_response",
            invocation_context=frozen_context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            response_format=response_format,
            force_local_only=force_local_only,
        )

        # Return full LLMResponse if tools or structured output requested
        if tools is not None or response_format is not None:
            return response
        else:
            if isinstance(response, LLMResponse):
                return response.content or ""
            return response

    async def get_audit_response(
        self,
        text_to_audit: str,
        invocation_context: Optional[LLMInvocationContext] = None,
        *,
        redact_content: bool = False,
    ) -> Dict[str, Any]:
        """Get a structured audit response from the normal provider chain.

        ``redact_content`` (#2674 finding 3): the audit's provider call carries
        the assistant prose as its ``user_prompt`` — under an *enforcing*
        (strict) audit that prose is being WITHHELD from the user pending this
        very verdict, so its own provider telemetry must not durably record it.
        The ResponseAuditHook passes ``redact_content=self.fail_closed`` so the
        strict path redacts and the advisory path is unchanged. Threaded on the
        frozen per-call context — never global state — so a concurrent call is
        unaffected.
        """
        self._check_policy()
        invocation_context = self._resolve_invocation_context(invocation_context)
        if redact_content and not invocation_context.redact_content:
            invocation_context = replace(invocation_context, redact_content=True)
        if not self.providers:
            return {"risk_level": 1, "reasoning": "Audit skipped - no providers available.", "audited": False}

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
                if target_model and not self._model_available_for_route(provider, target_model):
                    # Record the skip so that if EVERY route rejects the
                    # mandated model, the loop fails closed (risk=3) instead of
                    # falling through to the benign "no providers" risk=1.
                    errors[provider["name"]] = (
                        f"target model {target_model} not available for route"
                    )
                    logger.debug(
                        "Audit: skipping %s (target model %s not in vendor catalog)",
                        provider["name"], target_model,
                    )
                    continue
                # Resolve the "auto" sentinel to a concrete model (#1408): the
                # main generation/streaming paths funnel through this so they
                # never leak "auto" to the wire; the audit path must too, or an
                # agent on the default model="auto" config fails every audit
                # call (e.g. ollama 404 "model 'auto' not found").
                try:
                    effective_model = self._resolve_concrete_model(target_model, provider)
                except ModelNotAvailableForRoute as exc:
                    errors[provider["name"]] = str(exc)
                    logger.debug(
                        "Audit: skipping %s (cannot resolve concrete model: %s)",
                        provider["name"], exc,
                    )
                    continue

                # The audit relies on a Pydantic response_format to get parseable
                # JSON back. A route that does not honor structured output (e.g.
                # the direct Gemini adapter, which ignores response_format and
                # replies in prose) would reproduce the original #2032 failure on
                # another provider class. Skip such routes so we fall through to a
                # structured-capable one instead of forcing risk_level=3.
                try:
                    supports_structured = provider["adapter"].provider_capabilities().supports_structured_output
                except Exception as exc:  # capability introspection must never hard-fail the audit
                    supports_structured = False
                    logger.debug(
                        "Audit: could not read capabilities for %s (%s); treating as no structured output",
                        provider["name"], exc,
                    )
                if not supports_structured:
                    errors[provider["name"]] = "route does not support structured output (response_format)"
                    logger.debug(
                        "Audit: skipping %s (no structured-output support)",
                        provider["name"],
                    )
                    continue

                messages = messages_for(
                    provider["adapter"],
                    user_prompt=text_to_audit,
                    system_prompt=system_prompt,
                )

                try:
                    # Request structured output via a Pydantic response_format so
                    # the audit JSON is honored across adapters (Anthropic via its
                    # tool pattern, OpenAI natively). The OpenAI-style format="json"
                    # string is silently ignored by the Anthropic adapter, which
                    # made every audit return malformed JSON → risk_level=3 (#2032).
                    response = await self._run_provider_attempt(
                        provider["adapter"].get_response(
                            client=provider["client"],
                            model=effective_model,
                            messages=messages,
                            response_format=AuditResult,
                        ),
                        provider["name"],
                        effective_model,
                        path="get_audit_response",
                        invocation_context=invocation_context,
                        system_prompt=system_prompt,
                        user_prompt=text_to_audit,
                        response_format=AuditResult,
                        # An internal audit must not replace the visible
                        # assistant response identity used by persistence.
                        publish_identity=False,
                    )
                    content = response.content if isinstance(response, LLMResponse) else response
                    response_json = json.loads(content)
                    if "risk_level" not in response_json or "reasoning" not in response_json:
                        raise ValueError("Missing required keys in audit response.")
                    return response_json
                except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                    # A malformed/unparseable audit payload from this route must
                    # not short-circuit the whole audit (#2032): record it and try
                    # the next eligible provider rather than forcing risk_level=3.
                    errors[provider["name"]] = f"malformed audit response: {exc}"
                    logger.warning(f"Audit provider {provider['name']} returned unparseable JSON: {exc}")
                    continue
                except (LLMProviderError, openai.APIError, openai.APIConnectionError, httpx.HTTPError, ConnectionError, TimeoutError) as exc:
                    errors[provider["name"]] = str(exc)
                    logger.warning(f"Audit provider {provider['name']} failed: {exc}")
                    continue

            if errors:
                joined = "; ".join(f"{name}: {error}" for name, error in errors.items())
                return {
                    "risk_level": 3,
                    "reasoning": f"Audit provider failed: {joined}",
                    "audited": False,
                }
            return {"risk_level": 1, "reasoning": "Audit skipped - no providers available.", "audited": False}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse audit JSON: {e}")
            return {
                "risk_level": 3,
                "reasoning": "Audit model returned malformed JSON.",
                "audited": False,
            }
        except LLMProviderError as e:
            logger.error(f"Audit provider failed: {e}")
            return {
                "risk_level": 3,
                "reasoning": f"Audit provider failed: {e}",
                "audited": False,
            }
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"Data validation error in audit: {e}", exc_info=True)
            return {
                "risk_level": 3,
                "reasoning": f"Audit failed: {e}",
                "audited": False,
            }
        except (openai.APIError, openai.APIConnectionError, httpx.HTTPError, ConnectionError, TimeoutError) as e:
            logger.error(f"Network/API error in audit: {e}", exc_info=True)
            return {
                "risk_level": 3,
                "reasoning": f"Audit failed: {e}",
                "audited": False,
            }
        except Exception as e:
            logger.error(f"Unexpected audit error: {e}", exc_info=True)
            return {
                "risk_level": 3,
                "reasoning": f"Audit failed: {e}",
                "audited": False,
            }

    async def get_response(
        self,
        system_prompt: str,
        user_prompt: str,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        cancel_token: Optional[CancelToken] = None,
        invocation_context: Optional[LLMInvocationContext] = None,
    ) -> Union[str, LLMResponse]:
        """Get a response after freezing request identity at API entry."""

        self._check_policy()
        frozen_context = self._resolve_invocation_context(invocation_context)
        _redact = frozen_context.redact_content
        with self._llm_request_span(
            "get_response",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_override=model_override,
            session_id=frozen_context.session_id,
            redact=_redact,
        ) as span:
            result = await self._get_response_frozen(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                force_local_only=force_local_only,
                model_override=model_override,
                tools=tools,
                response_format=response_format,
                tool_executor=tool_executor,
                cancel_token=cancel_token,
                invocation_context=frozen_context,
            )
            return self._annotate_and_return(span, result, redact=_redact)

    async def _get_response_frozen(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        cancel_token: Optional[CancelToken] = None,
        invocation_context: LLMInvocationContext,
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
        if not self.providers:
            raise RuntimeError("No LLM providers initialized.")

        # Use mandate-aware model from prompt if no explicit override
        effective_override = model_override if model_override else self._get_model_for_prompt(user_prompt)

        resolution = await self._resolve_routing_with_discovery(
            model_override=effective_override,
            force_local_only=force_local_only,
        )
        available_providers, target_model = resolution
        explicit_selection, configured_vendors = resolution.meta

        # Strip tools if the target model can't handle them
        tools = self._check_model_tool_support(available_providers, tools, model_override)

        errors = {}
        for provider_index, provider in enumerate(available_providers):
            if not explicit_selection and self._skip_paid_fallback(
                provider, available_providers, provider_index
            ):
                logger.warning(
                    "Refusing silent plan->paid downgrade to %s in get_response "
                    "(a plan/free route was preferred; set "
                    "llm.allow_paid_fallback=true to permit). Not billing the "
                    "metered API on a plan failure.",
                    provider.get("name"),
                )
                errors[provider["name"]] = LLMServiceError(
                    f"Route {provider['name']} skipped: refusing silent "
                    f"plan->paid downgrade (llm.allow_paid_fallback=false)"
                )
                continue
            if not explicit_selection and self._skip_unconfigured_route(
                provider, configured_vendors
            ):
                logger.warning(
                    "Skipping unconfigured-vendor route %s (vendor %s not "
                    "authorized) in get_response; refusing blind "
                    "cross-vendor fallback.",
                    provider.get("name"), provider.get("vendor"),
                )
                errors[provider["name"]] = LLMServiceError(
                    f"Route {provider['name']} skipped: vendor "
                    f"{provider.get('vendor')} not in authorized vendor set"
                )
                continue
            try:
                provider_name = provider['name']
                logger.info(f"Attempting provider: {provider_name}")

                # The single OpenInference LLM span is opened by the public
                # entry method (issue #2573, Q1: one span per logical request).
                # This per-provider loop is a fallback retry chain, not N
                # logical calls, so it emits no span of its own.
                result = await self._try_single_provider(
                    provider=provider,
                    target_model=target_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    tools=tools,
                    response_format=response_format,
                    force_local_only=force_local_only,
                    tool_executor=tool_executor,
                    cancel_token=cancel_token,
                    explicit_selection=explicit_selection,
                    invocation_context=invocation_context,
                )
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
                # Record the failed attempt as an event on the one
                # logical-request span (opened by the public entry method).
                # #2674 finding 4: thread the per-invocation redaction flag so an
                # enforcing-audit turn records only the exception class, never the
                # raw provider error text (which can embed the withheld prompt /
                # response body). Bound to THIS context, never global state.
                telemetry.record_llm_attempt_failure(
                    provider["name"], e,
                    redact_content=getattr(
                        invocation_context, "redact_content", False
                    ),
                )

                if self._configured_routes_exhausted(
                    available_providers, provider_index, configured_vendors
                ):
                    # Operator's preferred-vendor routes are exhausted; every
                    # remaining candidate is an unconfigured vendor (which the
                    # top-of-loop guard skips). Surface the failure loudly
                    # instead of silently answering from an unconfigured vendor
                    # (feedback_no_blind_fallbacks).
                    logger.error(
                        "Configured routes exhausted in get_response: preferred "
                        "vendor %s route %s failed and every remaining route is "
                        "an unconfigured vendor. Error: %s",
                        available_providers[0].get("vendor"), provider["name"], e,
                    )
                    raise LLMServiceError(
                        f"Preferred route {provider['name']} failed and the "
                        f"only remaining routes are unconfigured vendors; "
                        f"refusing to silently swap vendors. "
                        f"Underlying error: {e}"
                    ) from e

        raise LLMAllProvidersFailedError(errors)

    async def get_response_with_model(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        auto_pull: bool = True,
        invocation_context: Optional[LLMInvocationContext] = None,
    ) -> str:
        """Get a response using a specific model."""
        self._check_policy()
        invocation_context = self._resolve_invocation_context(invocation_context)
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

            _redact = invocation_context.redact_content
            with self._llm_request_span(
                "get_response_with_model",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_override=model_id,
                session_id=invocation_context.session_id,
                redact=_redact,
            ) as span:
                response = await self._run_provider_attempt(
                    provider_for_model["adapter"].get_response(
                        client=provider_for_model["client"],
                        model=model_id,
                        messages=messages,
                        extra_body=provider_cache_body(provider_for_model),
                    ),
                    provider_for_model["name"],
                    model_id,
                    path="get_response_with_model",
                    invocation_context=invocation_context,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                logger.info(f"Success from {model_id}")
                return self._annotate_and_return(span, response, redact=_redact)

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

        # Stop accepting remote calls and drain them before discarding route
        # credentials. Provider capacity is deliberately NOT released here:
        # the durable provider reconciler owns expiry/release after host death.
        if self._remote_lease is not None:
            try:
                await self.deactivate_inference_lease(
                    self._remote_lease.lease_id,
                    require_active=False,
                )
            except LLMServiceError as exc:
                logger.warning(
                    "Private inference route cleanup did not complete during "
                    "LLMService shutdown (%s)",
                    type(exc).__name__,
                )

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
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        cancel_token: Optional[CancelToken] = None,
        invocation_context: Optional[LLMInvocationContext] = None,
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
        invocation_context = self._resolve_invocation_context(invocation_context)
        _redact = invocation_context.redact_content
        with self._llm_request_span(
            "generate",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_override=model_override,
            session_id=invocation_context.session_id,
            redact=_redact,
        ) as span:
            async with self._remote_route_attempt(
                force_local_only=force_local_only,
                model_override=model_override,
                required_capabilities=(
                    "chat",
                    *(("tools",) if tools else ()),
                    *(("structured_output",) if response_format else ()),
                ),
            ) as remote_route:
                if remote_route is None:
                    remote_response = None
                else:
                    try:
                        messages = messages_for(
                            remote_route.adapter,
                            user_prompt=user_prompt,
                            system_prompt=system_prompt,
                        )
                        model = self._scrub_auto(model_override) or remote_route.model
                        remote_response = await self._run_provider_attempt(
                            remote_route.adapter.get_response(
                                client=remote_route.client,
                                model=model,
                                messages=messages,
                                tools=tools,
                                response_format=response_format,
                                cancel_token=cancel_token,
                            ),
                            "remote_gpu",
                            model,
                            path="generate.remote_gpu",
                            invocation_context=invocation_context,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            tools=tools,
                            response_format=response_format,
                            force_local_only=force_local_only,
                            error_message_override=(
                                self._managed_remote_failure_message
                            ),
                        )
                    except Exception as exc:
                        # Adapter implementations can raise provider-specific
                        # exception types. The boundary intentionally catches
                        # them all, exposes only a safe category, and never
                        # falls through to a different provider.
                        self._raise_managed_remote_failure(exc)

            if remote_route is not None:
                if tools is not None or response_format is not None:
                    return self._annotate_and_return(
                        span,
                        remote_response,
                        redact=_redact,
                    )
                if isinstance(remote_response, LLMResponse):
                    return self._annotate_and_return(
                        span,
                        remote_response.content or "",
                        redact=_redact,
                    )
                return self._annotate_and_return(
                    span,
                    remote_response,
                    redact=_redact,
                )

            # Fall back to standard provider chain
            result = await self._get_response_frozen(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                force_local_only=force_local_only,
                model_override=model_override,
                tools=tools,
                response_format=response_format,
                tool_executor=tool_executor,
                cancel_token=cancel_token,
                invocation_context=invocation_context,
            )
            return self._annotate_and_return(span, result, redact=_redact)

    async def generate_with_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        session_id: Optional[str] = None,
        keep_trailing_system: bool = False,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        cancel_token: Optional[CancelToken] = None,
        invocation_context: Optional[LLMInvocationContext] = None,
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
        invocation_context = self._resolve_invocation_context(
            invocation_context,
            session_id=session_id,
        )
        with self._llm_request_span(
            "generate_with_messages",
            messages=messages,
            model_override=model_override,
            session_id=invocation_context.session_id,
            redact=invocation_context.redact_content,
        ) as span:
            return await self._generate_with_messages_inner(
                span,
                messages=messages,
                tools=tools,
                response_format=response_format,
                force_local_only=force_local_only,
                model_override=model_override,
                session_id=session_id,
                keep_trailing_system=keep_trailing_system,
                tool_executor=tool_executor,
                cancel_token=cancel_token,
                invocation_context=invocation_context,
            )

    async def _generate_with_messages_inner(
        self,
        span,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        force_local_only: bool = False,
        model_override: Optional[str] = None,
        session_id: Optional[str] = None,
        keep_trailing_system: bool = False,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        cancel_token: Optional[CancelToken] = None,
        invocation_context: LLMInvocationContext,
    ) -> Union[str, LLMResponse]:
        """Body of :meth:`generate_with_messages`, run inside the request's LLM span.

        Split out so the public entry opens exactly ONE OpenInference LLM span
        (issue #2573, Q1) and this inner routine executes entirely inside it:
        the provider loop's failed attempts attach to that span as events
        (:func:`telemetry.record_llm_attempt_failure`), and each successful
        return is annotated with the served model + response text.
        """
        # #2674 finding 3: content-redaction rides the frozen context; blank the
        # span's output.value under an enforcing audit (input.value was already
        # blanked when the span opened).
        _redact = invocation_context.redact_content
        async with self._remote_route_attempt(
            force_local_only=force_local_only,
            model_override=model_override,
            required_capabilities=(
                "chat",
                *(("tools",) if tools else ()),
                *(("structured_output",) if response_format else ()),
            ),
        ) as remote_route:
            if remote_route is None:
                remote_response = None
            else:
                try:
                    model = self._scrub_auto(model_override) or remote_route.model
                    remote_response = await self._run_provider_attempt(
                        remote_route.adapter.get_response(
                            client=remote_route.client,
                            model=model,
                            messages=messages,
                            tools=tools,
                            response_format=response_format,
                            cancel_token=cancel_token,
                        ),
                        "remote_gpu",
                        model,
                        path="generate_with_messages.remote_gpu",
                        invocation_context=invocation_context,
                        tools=tools,
                        response_format=response_format,
                        force_local_only=force_local_only,
                        error_message_override=(
                            self._managed_remote_failure_message
                        ),
                    )
                except Exception as exc:
                    self._raise_managed_remote_failure(exc)

        if remote_route is not None:
            if tools is not None or response_format is not None:
                return self._annotate_and_return(
                    span,
                    remote_response,
                    redact=_redact,
                )
            if isinstance(remote_response, LLMResponse):
                return self._annotate_and_return(
                    span,
                    remote_response.content or "",
                    redact=_redact,
                )
            return self._annotate_and_return(
                span,
                remote_response,
                redact=_redact,
            )

        # Fall back to standard providers (skip any disabled by auth failure).
        providers = self._available_providers()

        # Lazy auto-resolution warm-up (#2069): resolve any ``model = "auto"``
        # route before the walk so a fresh-boot cold cache doesn't surface a
        # hard ``ModelNotAvailableForRoute``. Shared with the streaming and
        # ``get_response`` paths via ``_ensure_models_discovered``. Pass
        # ``force_local_only`` so a local-only turn skips the cloud-contacting
        # warm-up (privacy). Re-fetch the local provider list afterward — this
        # path resolves routing inline (it does not call
        # ``resolve_provider_routing``), so it needs the freshly resolved
        # provider models.
        await self._ensure_models_discovered(force_local_only=force_local_only)
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

        # Narrow ``providers``/``target_model`` to the pinned selector. This
        # path resolves routing inline (it does NOT call
        # resolve_provider_routing), but the no-silent-fallback signals
        # (``explicit_selection`` + ``authorized_vendors``) are computed once
        # below by the SAME shared helper resolve_provider_routing uses, so
        # there is exactly one implementation of "what's authorized + is it
        # explicit" across both paths.
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

        # Same no-silent-fallback rule as the streaming paths: if the caller
        # explicitly pinned a single route (by mandate, route, or override),
        # failure raises with the *specific* provider+error. No cascade to an
        # unrelated backend. Both ``explicit_selection`` and
        # ``configured_vendors`` come from the SAME shared helper
        # resolve_provider_routing uses, so there's one implementation of the
        # branching logic. ``explicit_selection`` is the real pinned-route
        # signal — NOT ``len(providers) == 1``, which would also fire for an
        # incidental singleton and wrongly bypass the unconfigured-route skip.
        # The helper mirrors resolve_provider_routing's branch conditions:
        # override precedence (a bare override does NOT authorize a stale
        # mandate's vendor, P2a) and mandate fallbacks engaged ONLY when an
        # active mandate is unmatched (no stale-fallback authorization, P2b).
        explicit_selection, configured_vendors = self._compute_route_authorization(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        last_error = None
        last_provider_name = None
        for provider_index, provider in enumerate(providers):
            if not explicit_selection and self._skip_paid_fallback(
                provider, providers, provider_index
            ):
                logger.warning(
                    "Refusing silent plan->paid downgrade to %s in "
                    "generate_with_messages (a plan/free route was preferred; "
                    "set llm.allow_paid_fallback=true to permit). Not billing "
                    "the metered API on a plan failure.",
                    provider.get("name"),
                )
                continue
            if not explicit_selection and self._skip_unconfigured_route(
                provider, configured_vendors
            ):
                logger.warning(
                    "Skipping unconfigured-vendor route %s (vendor %s not "
                    "authorized) in generate_with_messages; refusing blind "
                    "cross-vendor fallback.",
                    provider.get("name"), provider.get("vendor"),
                )
                continue
            last_provider_name = provider["name"]
            try:
                model = target_model or provider["model"]
                logger.info(f"Attempting provider: {provider['name']} with model: {model}")
                response = await self._run_provider_attempt(
                    provider["adapter"].get_response(
                        client=provider["client"],
                        model=model,
                        messages=messages,
                        tools=tools,
                        response_format=response_format,
                        extra_body=provider_cache_body(provider),
                        session_id=session_id,
                        keep_trailing_system=keep_trailing_system,
                        tool_executor=tool_executor,
                        cancel_token=cancel_token,
                    ),
                    provider["name"],
                    model,
                    path="generate_with_messages",
                    invocation_context=invocation_context,
                    tools=tools,
                    response_format=response_format,
                    force_local_only=force_local_only,
                )
                if tools is not None or response_format is not None:
                    return self._annotate_and_return(span, response, redact=_redact)
                if isinstance(response, LLMResponse):
                    return self._annotate_and_return(span, response.content or "", redact=_redact)
                return self._annotate_and_return(span, response, redact=_redact)
            except openai.BadRequestError as e:
                # 400 = request problem (context too big, bad format, etc.)
                # Don't fall back — the request itself is broken, not the provider.
                logger.error(f"Provider {provider['name']} rejected request (400): {e}")
                raise LLMServiceError(f"Request rejected by {provider['name']}: {e}") from e
            except Exception as e:
                logger.error(f"Provider {provider['name']} failed: {e}")
                self._maybe_disable_route(provider, e)
                last_error = e
                if explicit_selection:
                    raise LLMServiceError(
                        f"Selected route {provider['name']} failed: {e}"
                    ) from e
                if self._configured_routes_exhausted(
                    providers, provider_index, configured_vendors
                ):
                    # The operator's preferred-vendor routes are exhausted and
                    # every remaining candidate is an unconfigured vendor (which
                    # the top-of-loop guard skips). Don't silently answer from
                    # an unconfigured vendor — surface the failure loudly
                    # (feedback_no_blind_fallbacks).
                    logger.error(
                        "Configured routes exhausted in generate_with_messages: "
                        "preferred vendor %s route %s failed and every remaining "
                        "route is an unconfigured vendor. Error: %s",
                        providers[0].get("vendor"), provider["name"], e,
                    )
                    raise LLMServiceError(
                        f"Preferred route {provider['name']} failed and the "
                        f"only remaining routes are unconfigured vendors; "
                        f"refusing to silently swap vendors. "
                        f"Underlying error: {e}"
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

    # Private inference lease routing is provided by RemoteBackendMixin.
