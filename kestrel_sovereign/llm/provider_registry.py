"""
Provider Registry for LLM Service.

Vendor/route/model data model:
    - A **vendor** is who makes the weights (openai, anthropic, ollama, ...).
    - A **route** is how to reach that vendor: base_url + auth + adapter class.
      One vendor may have multiple routes (api, plan, local, ...).
    - A **model** lives inside a vendor; all routes for that vendor share the
      model catalog.

Config shape (kestrel.toml ``[llm]`` section):

    [llm]
    route_priority = ["anthropic:plan", "openai:api", ...]

    [vendors.anthropic]
    is_cloud = true

    [vendors.anthropic.routes.api]
    adapter        = "AnthropicAdapter"
    api_key_env    = "ANTHROPIC_API_KEY"
    model          = "auto"

    [vendors.anthropic.routes.plan]
    adapter        = "ClaudeMaxAdapter"
    auth_token_env = "ANTHROPIC_AUTH_TOKEN"
    model          = "auto"

Each successfully initialized route becomes one ``ProviderInfo`` entry in
``self.providers`` with a composite name ``"<vendor>:<route>"``. Downstream
code iterates ``self.providers`` as routes; discovery groups by
``provider.vendor`` to produce per-vendor catalogs.

Entry-point providers (external packages) register under their advertised
``provider_name``; they become a single-route vendor with route ``api``.
"""
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json as _json

import openai

from kestrel_sovereign.kestrel_config.defaults import get_ollama_url, get_openrouter_api_base

try:
    import ollama
except ImportError:
    ollama = None

from kestrel_sdk.llm import LLMAdapter as _SDKLLMAdapter, ProviderInfo
from kestrel_sdk.llm import ProviderCapabilities

from .adapter import LLMAdapter
from .ollama_adapter import OllamaAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .google_adapter import GoogleAdapter
from .vertex_adapter import VertexAIAdapter
from .openrouter_adapter import OpenRouterAdapter
from .claude_max_adapter import ClaudeMaxAdapter
from .codex_adapter import CodexAdapter

logger = logging.getLogger(__name__)

LLM_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.llm_providers"


# Adapter class name (string in config) → class object. No hardcoded per-vendor
# logic here; the config names its own adapter.
_ADAPTER_REGISTRY: Dict[str, type] = {
    "OpenAIAdapter": OpenAIAdapter,
    "AnthropicAdapter": AnthropicAdapter,
    "ClaudeMaxAdapter": ClaudeMaxAdapter,
    "CodexAdapter": CodexAdapter,
    "OpenRouterAdapter": OpenRouterAdapter,
    "OllamaAdapter": OllamaAdapter,
    "GoogleAdapter": GoogleAdapter,
    "VertexAIAdapter": VertexAIAdapter,
}


def _normalize_capabilities(raw: Any) -> ProviderCapabilities:
    if isinstance(raw, ProviderCapabilities):
        return raw
    if isinstance(raw, dict):
        return ProviderCapabilities.from_mapping(raw)
    return ProviderCapabilities()


# Registration-time capability validator (#1983). Each entry pairs a v5
# ``ProviderCapabilities`` flag with the optional ``LLMAdapter`` method that
# implements it and the ``contract_features()`` key an adapter uses to opt in.
# If a route advertises the flag but still inherits the SDK's default method
# *and* hasn't declared the feature, the capability silently won't work — we
# warn (non-fatal) so the drift surfaces at startup instead of at call time.
_V5_CAPABILITY_PROBES: tuple[tuple[str, str, str], ...] = (
    ("supports_token_counting", "count_tokens", "token_counting"),
    ("supports_batch", "batch_submit", "batch"),
    ("supports_files", "file_upload", "files"),
    ("supports_raw_passthrough", "raw_request", "raw_passthrough"),
)


def _adapter_contract_features(adapter: Any) -> frozenset:
    """Best-effort read of an adapter's ``contract_features()`` opt-in set.

    Returns an empty set when the adapter predates v5 or the probe raises, so
    callers can treat "didn't declare it" uniformly.
    """
    fn = getattr(adapter, "contract_features", None)
    if fn is None:
        return frozenset()
    try:
        return frozenset(fn() or ())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "contract_features() probe failed for %s: %s",
            type(adapter).__name__,
            exc,
        )
        return frozenset()


def _warn_unimplemented_capabilities(info: ProviderInfo) -> None:
    """Warn when a route advertises a v5 capability it can't actually serve.

    Non-fatal: a misconfigured plugin should still load for its working
    features. Compares each advertised flag against the SDK base method so an
    adapter that overrides the method (or declares the feature) passes clean.
    """
    adapter = getattr(info, "adapter", None)
    caps = getattr(info, "capabilities", None)
    if adapter is None or not isinstance(caps, ProviderCapabilities):
        return
    declared = _adapter_contract_features(adapter)
    cls = type(adapter)
    for flag, probe, feature in _V5_CAPABILITY_PROBES:
        if not getattr(caps, flag, False) or feature in declared:
            continue
        impl = getattr(cls, probe, None)
        base = getattr(_SDKLLMAdapter, probe, None)
        if impl is not None and impl is base:
            logger.warning(
                "Route '%s' advertises %s but %s still uses LLMAdapter's "
                "default %s() and does not declare '%s' in contract_features(); "
                "that capability will not work until the method is implemented.",
                info.name,
                flag,
                cls.__name__,
                probe,
                feature,
            )


class ProviderInitializationError(Exception):
    """Raised when provider initialization fails."""
    pass


class ProviderRegistry:
    """Initialize route-scoped providers from a vendor/route config."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider registry.

        Args:
            config: Top-level config dict. Expected to contain ``route_priority``
                and a ``vendors`` map. Test configs with the legacy flat shape
                (``provider_priority`` + flat sections) are no longer supported.
        """
        self.config = config
        self.providers: List[ProviderInfo] = []
        self._initialized = False
        # OpenRouter routes that declared only a management key (no static
        # OPENROUTER_API_KEY). Sync init can't mint (async), so it records the
        # route here and the async ``finalize_providers()`` hook brings it up
        # with a bootstrap child key. Each entry: (vendor, route, vendor_cfg,
        # route_cfg).
        self._deferred_openrouter_routes: List[
            Tuple[str, str, Dict[str, Any], Dict[str, Any]]
        ] = []
        # Injected by ``finalize_providers()`` just before rebuilding a
        # deferred OpenRouter route, so the sync OpenRouter branch uses the
        # freshly-minted bootstrap key as the route default.
        self._bootstrap_openrouter_key: Optional[str] = None

    def initialize_providers(self) -> List[ProviderInfo]:
        """Initialize all routes declared under ``vendors``.

        Returns:
            List of successfully initialized routes.

        Raises:
            ProviderInitializationError: if no routes could be brought up.
        """
        if self._initialized:
            return self.providers

        vendors_cfg = self.config.get("vendors") or {}
        route_priority = list(self.config.get("route_priority") or [])

        # Walk every declared (vendor, route) pair.
        all_keys: List[str] = []
        for vendor_name, vendor_cfg in vendors_cfg.items():
            if not isinstance(vendor_cfg, dict):
                continue
            routes_cfg = vendor_cfg.get("routes") or {}
            for route_name in routes_cfg:
                all_keys.append(f"{vendor_name}:{route_name}")

        # Deterministic order: priority list first, then any remainder.
        ordered_keys: List[str] = []
        seen = set()
        for key in route_priority:
            if key in all_keys and key not in seen:
                ordered_keys.append(key)
                seen.add(key)
        for key in all_keys:
            if key not in seen:
                ordered_keys.append(key)
                seen.add(key)

        initialized: List[ProviderInfo] = []
        for key in ordered_keys:
            vendor_name, route_name = key.split(":", 1)
            vendor_cfg = vendors_cfg.get(vendor_name) or {}
            route_cfg = (vendor_cfg.get("routes") or {}).get(route_name) or {}
            try:
                info = self._build_route(vendor_name, route_name, vendor_cfg, route_cfg)
                if info is not None:
                    initialized.append(info)
                    logger.info("Initialized route: %s", info.name)
            except Exception as e:
                logger.warning("Failed to initialize route '%s': %s", key, e)

        # Discover external adapters registered via entry_points. They declare
        # a ``provider_name``; we treat it as a single-route vendor.
        ep_routes = self._discover_entrypoint_providers()
        existing = {p.name for p in initialized}
        for ep in ep_routes:
            if ep.name in existing:
                logger.debug("Skipping entry_point route '%s': already registered", ep.name)
                continue
            initialized.append(ep)
            existing.add(ep.name)
            logger.info("Registered entry_point route: %s", ep.name)

        if not initialized and not self._deferred_openrouter_routes:
            raise ProviderInitializationError(
                "No routes could be initialized. Check vendor auth envs "
                "(e.g. ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN for the "
                "Claude OAuth/plan route, OPENAI_API_KEY) and "
                "kestrel.toml [llm]."
            )

        # v5 capability contract check (#1983): surface advertised-but-missing
        # capabilities once, after every route (built-in + entry_point) is in.
        for info in initialized:
            _warn_unimplemented_capabilities(info)

        self.providers = initialized
        self._initialized = True
        return self.providers

    async def finalize_providers(self) -> List[ProviderInfo]:
        """Async completion pass for routes sync init could not bring up.

        ``initialize_providers()`` is synchronous, but some routes need an
        async step to register. Today that is OpenRouter routes which declared
        only ``OPENROUTER_MANAGEMENT_API_KEY`` (no static ``OPENROUTER_API_KEY``):
        we mint a process-wide, ephemeral bootstrap child key from the
        management key and register the route using it as the default. Per-user
        keys still override this in the call path.

        Idempotent and safe to call multiple times: routes already registered
        are skipped, and the bootstrap key is minted at most once per process.
        """
        if not self._deferred_openrouter_routes:
            return self.providers

        pending = list(self._deferred_openrouter_routes)
        self._deferred_openrouter_routes = []

        newly: List[ProviderInfo] = []
        existing = {p.name for p in self.providers}
        for vendor, route, vendor_cfg, route_cfg in pending:
            name = f"{vendor}:{route}"
            if name in existing:
                continue
            management_key = _openrouter_management_key(route_cfg)
            if not management_key:
                # Config changed underneath us; nothing to mint from.
                continue
            try:
                self._bootstrap_openrouter_key = await _mint_bootstrap_openrouter_key(
                    management_key
                )
                info = self._build_route(vendor, route, vendor_cfg, route_cfg)
                if info is not None:
                    self.providers.append(info)
                    existing.add(info.name)
                    newly.append(info)
                    logger.info(
                        "Initialized OpenRouter route via bootstrap key: %s",
                        info.name,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to finalize OpenRouter route '%s': %s", name, e
                )
            finally:
                self._bootstrap_openrouter_key = None

        for info in newly:
            _warn_unimplemented_capabilities(info)
        return self.providers

    # ------------------------------------------------------------------ routes

    def _build_route(
        self,
        vendor: str,
        route: str,
        vendor_cfg: Dict[str, Any],
        route_cfg: Dict[str, Any],
    ) -> Optional[ProviderInfo]:
        """Instantiate the adapter + client for one (vendor, route) pair."""
        adapter_name = route_cfg.get("adapter")
        if not adapter_name:
            raise ValueError(f"Route {vendor}:{route} missing 'adapter' field")
        adapter_cls = _ADAPTER_REGISTRY.get(adapter_name)
        if adapter_cls is None:
            raise ValueError(f"Unknown adapter class '{adapter_name}' for {vendor}:{route}")

        is_local = bool(route_cfg.get("local", False)) or not vendor_cfg.get("is_cloud", True)
        is_cloud = not is_local

        client, adapter = self._build_client_and_adapter(
            vendor=vendor,
            route=route,
            adapter_cls=adapter_cls,
            vendor_cfg=vendor_cfg,
            route_cfg=route_cfg,
        )
        if client is None:
            return None

        model = route_cfg.get("model") or "auto"
        hints = list(route_cfg.get("selection_hints") or [])
        base_url = route_cfg.get("base_url")
        if not base_url and adapter_cls is OllamaAdapter:
            base_url = (
                os.environ.get("OLLAMA_HOST")
                or route_cfg.get("host")
                or get_ollama_url()
            )

        # Optional embedding sibling (#1494). When the active chat
        # provider can't embed (Anthropic has no embedding API), the
        # sibling fills the gap. Route-level config wins over
        # vendor-level so an operator can override on a per-route
        # basis if needed. The string is the provider's lookup name:
        # ``"<vendor>"`` (first matching route for that vendor) or
        # ``"<vendor>:<route>"`` (exact). Resolution + privacy
        # filtering happens in :class:`LLMService`; here we only
        # carry the raw string into ``ProviderInfo`` so it survives
        # the SDK boundary.
        embedding_sibling = route_cfg.get("embedding_sibling")
        if embedding_sibling is None:
            embedding_sibling = vendor_cfg.get("embedding_sibling")
        if embedding_sibling is not None and not isinstance(embedding_sibling, str):
            raise ValueError(
                f"Route {vendor}:{route}: embedding_sibling must be a string "
                f"(got {type(embedding_sibling).__name__})"
            )
        normalized_sibling = (
            embedding_sibling.strip() if embedding_sibling else None
        ) or None

        info = ProviderInfo(
            name=f"{vendor}:{route}",
            vendor=vendor,
            route=route,
            client=client,
            adapter=adapter,
            model=model,
            is_cloud=is_cloud,
            is_local=is_local,
            base_url=base_url,
            selection_hints=hints,
            capabilities=adapter.provider_capabilities(),
        )
        # ProviderInfo is an SDK dataclass and we don't want to change
        # its public schema for a sovereign-only config knob. Stash
        # the string as a private attr; ``LLMService._convert_providers_format``
        # surfaces it as ``embedding_sibling`` on the dict shape that
        # routing code consumes.
        info._kestrel_embedding_sibling = normalized_sibling  # type: ignore[attr-defined]
        # Sovereign-only knob (#1954): per-route reasoning effort for local
        # reasoning models (e.g. GLM-5.2 via llama_cpp, which otherwise runs at
        # its chat-template default of Max). Stashed like the embedding sibling
        # above; surfaced as ``reasoning_effort`` on the dict shape by
        # ``LLMService`` and gated to llama.cpp in ``provider_cache_body``.
        info._kestrel_reasoning_effort = route_cfg.get("reasoning_effort")  # type: ignore[attr-defined]
        return info

    def _build_client_and_adapter(
        self,
        vendor: str,
        route: str,
        adapter_cls: type,
        vendor_cfg: Dict[str, Any],
        route_cfg: Dict[str, Any],
    ):
        """Instantiate (client, adapter) for the given adapter class.

        Adapter-class-specific logic is here, NOT per-vendor. Adding a new
        vendor that reuses OpenAIAdapter requires no code changes — just config.
        """
        # --- Anthropic SDK (api key or OAuth) ---
        if adapter_cls in (AnthropicAdapter, ClaudeMaxAdapter):
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package not installed.")
            api_key = self._resolve_secret(route_cfg, "api_key_env", "api_key")
            auth_token = self._resolve_secret(route_cfg, "auth_token_env", "auth_token")
            # The codex:plan route delegates its OAuth lifecycle to the codex
            # binary; anthropic:plan owns its SDK client, so it owns the
            # equivalent. A static auth_token, an explicit credentials file, or
            # (plan route only) the auto-discovered Claude Code CLI store all
            # initialize the OAuth route. Delegation/discovery is gated to the
            # plan adapter so the metered API-key route never reaches for the
            # subscription store.
            from .anthropic_oauth import ClaudeOAuthTokenManager

            oauth_manager = ClaudeOAuthTokenManager.from_sources(
                static_token=auth_token,
                credentials_path=route_cfg.get("oauth_credentials_file"),
                delegate=(adapter_cls is ClaudeMaxAdapter),
            )
            if oauth_manager is not None:
                client = anthropic.AsyncAnthropic(
                    auth_token=oauth_manager.initial_access_token
                )
                # The Anthropic SDK back-fills ``api_key`` from
                # ``ANTHROPIC_API_KEY`` in the environment whenever the
                # constructor arg is None (which it is here — we only
                # passed ``auth_token``). ``auth_headers`` then emits
                # BOTH ``X-Api-Key`` and ``Authorization: Bearer``, so a
                # ``plan``/OAuth route silently authenticates and bills
                # against the metered API key and dies with a spurious
                # "api key" error the moment that key is disabled. Null
                # the leaked key so this route sends Bearer ONLY.
                client.api_key = None
                logger.info("%s:%s using OAuth token", vendor, route)
            elif api_key:
                client = anthropic.AsyncAnthropic(api_key=api_key)
            else:
                raise ValueError(
                    f"{vendor}:{route} requires api_key_env, auth_token_env, or "
                    "oauth_credentials_file (ANTHROPIC_API_KEY for API-key routes; "
                    "ANTHROPIC_AUTH_TOKEN or a credentials file for OAuth)"
                )
            adapter = adapter_cls()
            if oauth_manager is not None:
                adapter._oauth_token_manager = oauth_manager
            return client, adapter

        # --- Codex / ChatGPT subscription via the official codex app-server ---
        # Auth is delegated entirely to the codex binary (~/.codex/auth.json,
        # written by `codex login`); the adapter spawns/manages the
        # app-server itself. We fail fast here if the binary can't be
        # located, and pass its path as the (otherwise-unused) client
        # slot so the route registers (a None client would skip the
        # route).
        if adapter_cls is CodexAdapter:
            from .codex_app_server import (
                CodexAppServerError,
                resolve_codex_binary,
            )
            try:
                binary = resolve_codex_binary()
            except CodexAppServerError as e:
                raise ValueError(
                    f"{vendor}:{route} codex app-server unavailable: {e}"
                ) from e
            return binary, adapter_cls()

        # --- OpenRouter (OpenAI-compatible client, custom adapter) ---
        if adapter_cls is OpenRouterAdapter:
            api_key = self._resolve_secret(route_cfg, "api_key_env", "api_key")
            if not api_key:
                # A bootstrap child key minted by ``finalize_providers()`` from
                # the management key is injected here on the async pass. When a
                # static OPENROUTER_API_KEY IS set this stays None and behavior
                # is byte-identical to before.
                api_key = self._bootstrap_openrouter_key
            if not api_key:
                # No static key and no bootstrap key yet. If a management key
                # is available, defer this route to the async
                # ``finalize_providers()`` hook (sync init can't await the
                # mint). With neither key set, fail closed exactly as before.
                if _openrouter_management_key(route_cfg):
                    deferral = (vendor, route, vendor_cfg, route_cfg)
                    if deferral not in self._deferred_openrouter_routes:
                        self._deferred_openrouter_routes.append(deferral)
                    logger.info(
                        "Deferring OpenRouter route %s:%s to async bootstrap "
                        "mint (OPENROUTER_API_KEY not set; management key present)",
                        vendor,
                        route,
                    )
                    return None, None
                raise ValueError(f"{vendor}:{route} requires OPENROUTER_API_KEY")
            base_url = route_cfg.get("base_url") or get_openrouter_api_base()
            # max_retries=0: the OpenAI SDK has its own retry layer that
            # duplicates (and contradicts) our llm/retry.py policy. One retry
            # owner only.
            client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
            # Route-level embedding config (#2288). OpenRouter's unified
            # OpenAI-compatible /v1/embeddings unlocks its whole embedding
            # catalog through the key we already configure, but the adapter
            # advertises embeddings ONLY when a route names a model — no
            # meta-provider default. The upstream model id (e.g.
            # ``qwen/qwen3-embedding-0.6b``) keys the embedding space, so two
            # different upstream models through this one route are distinct
            # spaces (see OpenRouterAdapter.embedding_space_id).
            embedding_model = route_cfg.get("embedding_model")
            embedding_dim = route_cfg.get("embedding_dim")
            if embedding_dim is not None:
                embedding_dim = int(embedding_dim)
            supports_embeddings = route_cfg.get("supports_embeddings")
            if isinstance(supports_embeddings, str):
                supports_embeddings = supports_embeddings.lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            adapter = adapter_cls(
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                supports_embeddings=supports_embeddings,
            )
            # The adapter constructs its own ``self.api_key`` from
            # OPENROUTER_API_KEY at __init__. Model discovery
            # (``list_models``) and its fallback ``_get_client`` use that
            # value directly and ignore the framework-provided client. When
            # the route registers off a minted *bootstrap* key (management
            # key only, no static OPENROUTER_API_KEY), the adapter's own
            # key is None and discovery would return an empty catalog —
            # breaking ``model="auto"`` resolution even though generation on
            # the injected client works. Point the adapter at the resolved
            # key so both surfaces authenticate with the same credential.
            adapter.api_key = api_key
            return client, adapter

        # --- Ollama (local or remote via OLLAMA_HOST) ---
        if adapter_cls is OllamaAdapter:
            if ollama is None:
                raise ImportError("ollama package not installed.")
            host = os.environ.get("OLLAMA_HOST") or route_cfg.get("host") or get_ollama_url()
            client = ollama.AsyncClient(host=host)
            return client, adapter_cls()

        # --- Google Gemini (maintained google-genai SDK) ---
        if adapter_cls is GoogleAdapter:
            try:
                from google import genai as _genai
            except ImportError:
                raise ImportError("google-genai package not installed.")
            api_key = self._resolve_secret(route_cfg, "api_key_env", "api_key") \
                or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(f"{vendor}:{route} requires GOOGLE_API_KEY")
            # The direct Gemini route uses the same google-genai client surface
            # as Vertex (client.aio.models.generate_content), just with an API
            # key instead of a service account. See GoogleAdapter.get_response.
            client = _genai.Client(api_key=api_key)
            return client, adapter_cls()

        # --- Vertex AI (new google-genai SDK, api-key or service-account) ---
        if adapter_cls is VertexAIAdapter:
            try:
                from google import genai as _genai
            except ImportError:
                raise ImportError("google-genai package not installed.")
            api_key = self._resolve_secret(route_cfg, "api_key_env", "api_key") \
                or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                client = _genai.Client(api_key=api_key)
                return client, adapter_cls()
            project_id = (
                route_cfg.get("project_id")
                or os.environ.get("GCP_PROJECT_ID")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            location = (
                route_cfg.get("location")
                or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            )
            if not project_id:
                raise ValueError(
                    f"{vendor}:{route} requires GOOGLE_API_KEY or GCP_PROJECT_ID"
                )
            client = _genai.Client(vertexai=True, project=project_id, location=location)
            adapter = adapter_cls(project_id=project_id, location=location)
            return client, adapter

        # --- Generic OpenAI-compatible (OpenAI itself, xAI, Groq, RunPod, llama.cpp, ...) ---
        if adapter_cls is OpenAIAdapter:
            # Default base_url: OpenAI if not specified. For OpenAI-proper we
            # use the SDK default (no base_url override).
            base_url = route_cfg.get("base_url")
            api_key = self._resolve_secret(route_cfg, "api_key_env", "api_key")
            is_local = bool(route_cfg.get("local", False)) or not vendor_cfg.get("is_cloud", True)
            if not api_key:
                api_key = os.environ.get(f"{vendor.upper()}_API_KEY")
            if not api_key and is_local:
                api_key = "local"
            if not api_key:
                raise ValueError(
                    f"{vendor}:{route} requires an API key "
                    f"(set {route_cfg.get('api_key_env') or vendor.upper() + '_API_KEY'})"
                )
            # max_retries=0: single retry owner (llm/retry.py). See note in
            # the OpenRouter branch above.
            kwargs = {"api_key": api_key, "max_retries": 0}
            if base_url:
                kwargs["base_url"] = base_url
            if is_local:
                # Local models can legitimately generate for many minutes (large
                # reasoning models at low tok/s). The OpenAI SDK default 600s
                # total timeout cancels these mid-flight; use a generous,
                # route-configurable timeout so long local generations complete
                # instead of being cancelled. See #1954.
                kwargs["timeout"] = float(route_cfg.get("timeout", 1800.0))
            client = openai.AsyncOpenAI(**kwargs)
            embedding_model = route_cfg.get("embedding_model")
            embedding_dim = route_cfg.get("embedding_dim")
            if embedding_dim is not None:
                embedding_dim = int(embedding_dim)
            # Computed unconditionally — both the embedding default below and
            # the native_openai signal passed to the adapter depend on it.
            official_openai_base = (
                not base_url
                or str(base_url).rstrip("/") == "https://api.openai.com/v1"
            )
            supports_embeddings = route_cfg.get("supports_embeddings")
            if supports_embeddings is None:
                supports_embeddings = (
                    vendor == "openai"
                    and official_openai_base
                ) or bool(embedding_model)
            elif isinstance(supports_embeddings, str):
                supports_embeddings = supports_embeddings.lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            return client, adapter_cls(
                name=vendor,
                supports_embeddings=bool(supports_embeddings),
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                # Only the real OpenAI vendor on the official base exposes the
                # /batches, /files, /responses surface; OpenAI-compatible routes
                # (custom base_url) do not. Pass through as a typed signal so the
                # adapter advertises those capabilities only when truly native.
                native_openai=(vendor == "openai" and official_openai_base),
            )

        # --- Fallback: try plain instantiation; let adapter fail at call time ---
        logger.warning("No client builder for adapter %s — using adapter-only", adapter_cls.__name__)
        return None, adapter_cls()

    @staticmethod
    def _resolve_secret(route_cfg: Dict[str, Any], env_key: str, inline_key: str) -> Optional[str]:
        """Read a secret from env (named by ``env_key`` in route_cfg) or inline."""
        env_name = route_cfg.get(env_key)
        if env_name:
            val = os.environ.get(env_name)
            if val:
                return val
        inline = route_cfg.get(inline_key)
        return inline or None

    # ------------------------------------------------------ entry-point providers

    def _discover_entrypoint_providers(self) -> List[ProviderInfo]:
        """Discover LLM providers registered via entry_points.

        Each external adapter is treated as a single-route vendor (route=``api``).
        The adapter's advertised ``provider_name`` becomes the vendor name; if the
        adapter exposes a ``create_provider(config)`` factory we accept whatever
        ProviderInfo it returns (normalizing ``vendor``/``route``/``name`` fields).
        """
        from kestrel_sovereign.entrypoints import discover_entry_point_classes

        providers: List[ProviderInfo] = []
        # Validate against the SDK base, not the framework-enriched
        # subclass, so third-party plugins that subclass
        # ``kestrel_sdk.llm.LLMAdapter`` directly (the documented and
        # intended entry point — they should not need to depend on
        # kestrel-sovereign) are accepted. The framework's own
        # ``LLMAdapter`` inherits from the SDK base, so in-tree
        # subclasses pass this check as well.
        classes = discover_entry_point_classes(LLM_PROVIDER_ENTRY_POINT_GROUP, _SDKLLMAdapter)

        for ep_name, cls in classes.items():
            try:
                provider_config = (self.config.get("vendors") or {}).get(ep_name) or {}
                route_cfg = ((provider_config.get("routes") or {}).get("api")) or {}

                if hasattr(cls, "create_provider") and callable(getattr(cls, "create_provider")):
                    info = cls.create_provider(route_cfg)
                    if info is None:
                        continue
                    # Normalize legacy ProviderInfo returned by external adapters.
                    vendor = getattr(info, "vendor", None) or info.name
                    route = getattr(info, "route", None) or "api"
                    info.vendor = vendor
                    info.route = route
                    info.name = f"{vendor}:{route}"
                    # External factories may set a precise ProviderInfo.capabilities
                    # themselves. Only backfill when the SDK default was left in place.
                    if getattr(info, "capabilities", None) == ProviderCapabilities():
                        info.capabilities = _normalize_capabilities(
                            info.adapter.provider_capabilities()
                        )
                    providers.append(info)
                    continue

                # Fallback: OpenAI-compatible client with base_url from config.
                base_url = route_cfg.get("base_url")
                api_key = (
                    self._resolve_secret(route_cfg, "api_key_env", "api_key")
                    or os.environ.get(f"{ep_name.upper()}_API_KEY", "external")
                )
                model = route_cfg.get("model", "auto")

                client_kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 0}
                if base_url:
                    client_kwargs["base_url"] = base_url
                client = openai.AsyncOpenAI(**client_kwargs)

                adapter = cls()
                providers.append(ProviderInfo(
                    name=f"{ep_name}:api",
                    vendor=ep_name,
                    route="api",
                    client=client,
                    adapter=adapter,
                    model=model,
                    is_cloud=True,
                    is_local=False,
                    base_url=base_url,
                    capabilities=_normalize_capabilities(
                        adapter.provider_capabilities()
                    ),
                ))
            except Exception as e:
                logger.warning("Failed to initialize entry_point LLM provider '%s': %s", ep_name, e)

        return providers

    # --------------------------------------------------------------- lookups

    def get_provider_by_name(self, name: str) -> Optional[ProviderInfo]:
        """Exact-match lookup by composite name (``"vendor:route"``) or vendor."""
        for provider in self.providers:
            if provider.name == name:
                return provider
        # Vendor-only match: return the first route for that vendor.
        for provider in self.providers:
            if provider.vendor == name:
                return provider
        return None

    def get_providers_for_vendor(self, vendor: str) -> List[ProviderInfo]:
        """All routes for a given vendor, in priority order."""
        return [p for p in self.providers if p.vendor == vendor]

    def get_provider_for_model(self, model: str) -> Optional[ProviderInfo]:
        """First route whose configured model matches."""
        for provider in self.providers:
            if model and provider.model and model in provider.model:
                return provider
        return None

    def get_local_providers(self) -> List[ProviderInfo]:
        """All routes marked local (Ollama, llama.cpp, ...)."""
        return [p for p in self.providers if p.is_local]

    def get_providers_with_pattern(self, patterns: List[str]) -> List[ProviderInfo]:
        """Routes whose model string contains any of the given patterns."""
        if not patterns:
            return []
        out: List[ProviderInfo] = []
        for provider in self.providers:
            model_lower = (provider.model or "").lower()
            if any(p.lower() in model_lower for p in patterns):
                out.append(provider)
        return out


# --------------------------------------------------------------------------
# Module-level helpers shared across service.py and streaming.py
# --------------------------------------------------------------------------

# Vendors that understand llama.cpp's `cache_prompt` body extension.  Setting
# it for other OpenAI-compatible proxies (OpenAI, OpenRouter, RunPod, xAI,
# Groq, Cerebras, …) can error on strict implementations, so we gate tightly.
# See issue #704.
_CACHE_PROMPT_VENDORS = frozenset({"llama_cpp"})


# --------------------------------------------------------------------------
# OpenRouter bootstrap child key (registry route default)
# --------------------------------------------------------------------------

# Default generous monthly limit for the ephemeral bootstrap child key. It is
# only the *route default* — per-user keys override it in the call path — so a
# roomy cap keeps the default route usable without babysitting.
_BOOTSTRAP_OPENROUTER_LIMIT_USD = 50.0

# Process-wide cache: mint the bootstrap key at most once per process *per
# management key*, even across multiple ``ProviderRegistry`` / ``LLMService``
# instances (multi-agent host). Keyed by management key so routes configured
# with DIFFERENT OpenRouter accounts never reuse each other's minted child key
# (preserves per-account billing isolation). Guarded by a lazily-created
# asyncio.Lock so concurrent finalize calls don't mint duplicates.
_BOOTSTRAP_OPENROUTER_KEYS: Dict[str, str] = {}
_BOOTSTRAP_OPENROUTER_LOCK: Optional[asyncio.Lock] = None


def _openrouter_management_key(route_cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve an OpenRouter management key for a route.

    Reads the env var named by ``management_api_key_env`` (default
    ``OPENROUTER_MANAGEMENT_API_KEY``) or an inline ``management_api_key``.
    Returns None when neither is set.
    """
    env_name = route_cfg.get("management_api_key_env") or "OPENROUTER_MANAGEMENT_API_KEY"
    return os.environ.get(env_name) or route_cfg.get("management_api_key")


async def _mint_bootstrap_openrouter_key(
    management_key: str,
    limit_usd: float = _BOOTSTRAP_OPENROUTER_LIMIT_USD,
) -> str:
    """Mint (once per process) an ephemeral OpenRouter child key.

    Used as the route default when a route declares only a management key. The
    key is NOT persisted to any DB — it is process-lived and only backs the
    default route; per-user keys override it in the call path.
    """
    global _BOOTSTRAP_OPENROUTER_LOCK
    cached = _BOOTSTRAP_OPENROUTER_KEYS.get(management_key)
    if cached is not None:
        return cached
    if _BOOTSTRAP_OPENROUTER_LOCK is None:
        _BOOTSTRAP_OPENROUTER_LOCK = asyncio.Lock()
    async with _BOOTSTRAP_OPENROUTER_LOCK:
        cached = _BOOTSTRAP_OPENROUTER_KEYS.get(management_key)
        if cached is not None:
            return cached
        # Late import: keep the provisioning surface (httpx) out of module
        # import so the registry loads on deployments that never touch it.
        from kestrel_sovereign.features.llm_keys.openrouter_provisioning import (
            OpenRouterProvisioningService,
        )

        service = OpenRouterProvisioningService(management_key=management_key)
        try:
            key_info = await service.create_agent_key(
                agent_name="bootstrap-registry-default",
                limit_usd=limit_usd,
                limit_reset="monthly",
            )
        finally:
            await service.close()
        _BOOTSTRAP_OPENROUTER_KEYS[management_key] = key_info.key
        logger.info(
            "Minted process-wide bootstrap OpenRouter key "
            "(limit $%.2f/mo) as OpenRouter route default",
            limit_usd,
        )
        return key_info.key


def _reset_bootstrap_openrouter_key_cache() -> None:
    """Test hook: clear the process-wide bootstrap-key cache."""
    global _BOOTSTRAP_OPENROUTER_LOCK
    _BOOTSTRAP_OPENROUTER_KEYS.clear()
    _BOOTSTRAP_OPENROUTER_LOCK = None


def provider_cache_body(provider: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return llama.cpp-specific ``extra_body`` kwargs for a request.

    Two llama-server body extensions, both gated to the ``llama_cpp`` vendor so
    strict OpenAI-compatible proxies never see unknown fields:

    - ``cache_prompt`` (#704): aggressive prefix-KV retention for the slot.
    - ``reasoning_effort`` (#1954): when the route configures one, control the
      reasoning budget of local reasoning models (e.g. GLM-5.2, which otherwise
      defaults to ``Max`` from its chat template with no way to override). Only
      sent when set, so the model's own default is preserved when unconfigured.

    Anthropic uses `cache_control` markers instead (#705) and Gemini a separate
    CachedContent API, so they get None. Returns None when there is nothing to
    send — the adapter's extra_body passthrough then does nothing.
    """
    if provider.get("vendor") not in _CACHE_PROMPT_VENDORS:
        return None
    body: Dict[str, Any] = {"cache_prompt": True}
    effort = provider.get("reasoning_effort")
    if effort:
        body["reasoning_effort"] = effort
    return body
