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
import logging
import os
from typing import List, Dict, Any, Optional
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

        if not initialized:
            raise ProviderInitializationError(
                "No routes could be initialized. Check vendor auth envs "
                "(e.g. ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN for the "
                "Claude OAuth/plan route, OPENAI_API_KEY) and "
                "kestrel.toml [llm]."
            )

        self.providers = initialized
        self._initialized = True
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

        return ProviderInfo(
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
        )

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
            if auth_token:
                client = anthropic.AsyncAnthropic(auth_token=auth_token)
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
                    f"{vendor}:{route} requires api_key_env or auth_token_env "
                    "(ANTHROPIC_API_KEY for API-key routes, ANTHROPIC_AUTH_TOKEN for OAuth)"
                )
            return client, adapter_cls()

        # --- Codex / ChatGPT subscription backend (raw OAuth token string) ---
        if adapter_cls is CodexAdapter:
            token = self._resolve_secret(route_cfg, "auth_token_env", "auth_token")
            if not token:
                token, _ = self._read_codex_auth_file()
            if not token:
                raise ValueError(
                    f"{vendor}:{route} OAuth token not found. "
                    "Run `codex login` or set CODEX_AUTH_TOKEN."
                )
            # Adapter uses httpx directly; client slot holds the token string.
            return token, adapter_cls()

        # --- OpenRouter (OpenAI-compatible client, custom adapter) ---
        if adapter_cls is OpenRouterAdapter:
            api_key = self._resolve_secret(route_cfg, "api_key_env", "api_key")
            if not api_key:
                raise ValueError(f"{vendor}:{route} requires OPENROUTER_API_KEY")
            base_url = route_cfg.get("base_url") or get_openrouter_api_base()
            # max_retries=0: the OpenAI SDK has its own retry layer that
            # duplicates (and contradicts) our llm/retry.py policy. One retry
            # owner only.
            client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
            return client, adapter_cls()

        # --- Ollama (local or remote via OLLAMA_HOST) ---
        if adapter_cls is OllamaAdapter:
            if ollama is None:
                raise ImportError("ollama package not installed.")
            host = os.environ.get("OLLAMA_HOST") or route_cfg.get("host") or get_ollama_url()
            client = ollama.AsyncClient(host=host)
            return client, adapter_cls()

        # --- Google Gemini (legacy generativeai SDK) ---
        if adapter_cls is GoogleAdapter:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("google-generativeai package not installed.")
            api_key = self._resolve_secret(route_cfg, "api_key_env", "api_key") \
                or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(f"{vendor}:{route} requires GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            # Client is lazily constructed per call; adapter carries the model name.
            return genai, adapter_cls()

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
            client = _genai.Client(vendorai=True, project=project_id, location=location)
            adapter = adapter_cls(project_id=project_id, location=location)
            return client, adapter

        # --- Generic OpenAI-compatible (OpenAI itself, xAI, Groq, RunPod, llama.cpp, ...) ---
        if adapter_cls is OpenAIAdapter:
            # Default base_url: OpenAI if not specified. For OpenAI-proper we
            # use the SDK default (no base_url override).
            base_url = route_cfg.get("base_url")
            api_key = self._resolve_secret(route_cfg, "api_key_env", "api_key")
            is_local = bool(route_cfg.get("local", False))
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
            client = openai.AsyncOpenAI(**kwargs)
            return client, adapter_cls(name=vendor)

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

    @staticmethod
    def _read_codex_auth_file() -> tuple:
        """Read OAuth token from ~/.codex/auth.json (written by `codex login`).

        Returns (token, auth_mode) tuple or (None, None) if not found/readable.
        """
        auth_path = Path.home() / ".codex" / "auth.json"
        if not auth_path.exists():
            return None, None
        try:
            data = _json.loads(auth_path.read_text())
            auth_mode = data.get("auth_mode", "")
            tokens = data.get("tokens", {})
            token = tokens.get("access_token") or data.get("access_token")
            if token:
                return token, auth_mode or "oauth"
            return None, None
        except Exception as e:
            logger.warning(f"Failed to read codex auth file: {e}")
            return None, None

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

                providers.append(ProviderInfo(
                    name=f"{ep_name}:api",
                    vendor=ep_name,
                    route="api",
                    client=client,
                    adapter=cls(),
                    model=model,
                    is_cloud=True,
                    is_local=False,
                    base_url=base_url,
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


def provider_cache_body(provider: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return extra_body kwargs for providers that accept explicit prompt-
    cache signaling on the request body.  Today only llama.cpp (llama-server)
    qualifies; Anthropic uses `cache_control` markers instead (see #705) and
    Gemini uses a separate CachedContent API.

    Returns None when the provider has no such extension — the adapter's
    extra_body passthrough then does nothing.
    """
    if provider.get("vendor") in _CACHE_PROMPT_VENDORS:
        return {"cache_prompt": True}
    return None
