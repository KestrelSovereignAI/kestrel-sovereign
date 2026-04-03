"""
Provider Registry for LLM Service.

This module extracts the provider initialization and routing logic from LLMService
to reduce complexity and improve maintainability.

External packages can register LLM providers via entry_points::

    [project.entry-points."kestrel_sovereign.llm_providers"]
    my_provider = "my_package:MyLLMAdapter"

Entry point classes must be subclasses of ``LLMAdapter``.  They are
instantiated and wrapped in a ``ProviderInfo`` during initialization.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import openai

from kestrel_sovereign.kestrel_config.defaults import get_ollama_url, get_openrouter_api_base

try:
    import ollama
except ImportError:
    ollama = None

from .adapter import LLMAdapter
from .ollama_adapter import OllamaAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .google_adapter import GoogleAdapter
from .vertex_adapter import VertexAIAdapter
from .openrouter_adapter import OpenRouterAdapter
from .claude_max_adapter import ClaudeMaxAdapter

logger = logging.getLogger(__name__)

LLM_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.llm_providers"


@dataclass
class ProviderInfo:
    """Information about an initialized provider."""
    name: str
    client: Any
    adapter: Any
    model: str


class ProviderInitializationError(Exception):
    """Raised when provider initialization fails."""
    pass


class ProviderRegistry:
    """Manages LLM provider initialization and routing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider registry.

        Args:
            config: Configuration dictionary containing provider settings
        """
        self.config = config
        self.providers: List[ProviderInfo] = []
        self._initialized = False

    def initialize_providers(self) -> List[ProviderInfo]:
        """Initialize provider clients and adapters based on config file.

        Returns:
            List of successfully initialized providers

        Raises:
            ProviderInitializationError: If no providers could be initialized
        """
        if self._initialized:
            return self.providers

        initialized_providers = []
        priority_list = self.config.get("provider_priority", [])

        for provider_name in priority_list:
            try:
                provider_info = self._initialize_single_provider(provider_name)
                if provider_info:
                    initialized_providers.append(provider_info)
                    logger.info(f"Initialized provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider '{provider_name}': {e}")

        # Phase 2: Discover external providers via entry_points
        ep_providers = self._discover_entrypoint_providers()
        initialized_names = {p.name for p in initialized_providers}
        for ep_provider in ep_providers:
            if ep_provider.name in initialized_names:
                logger.debug(
                    "Skipping entry_point LLM provider '%s': built-in already registered",
                    ep_provider.name,
                )
                continue
            initialized_providers.append(ep_provider)
            initialized_names.add(ep_provider.name)
            logger.info("Registered entry_point LLM provider: %s", ep_provider.name)

        if not initialized_providers:
            raise ProviderInitializationError("No providers could be initialized")

        self.providers = initialized_providers
        self._initialized = True
        return self.providers

    def _discover_entrypoint_providers(self) -> List[ProviderInfo]:
        """Discover LLM providers registered via entry_points.

        Entry points must be subclasses of ``LLMAdapter``.  Each adapter class
        must expose a ``provider_name`` class attribute and a
        ``create_provider(config)`` classmethod that returns a ``ProviderInfo``.

        If the classmethod is absent, the adapter is instantiated directly
        and wrapped with an OpenAI-compatible client using ``base_url`` and
        ``api_key`` from the provider config section.

        Returns:
            List of successfully initialized ProviderInfo objects.
        """
        from kestrel_sovereign.entrypoints import discover_entry_point_classes

        providers: List[ProviderInfo] = []
        classes = discover_entry_point_classes(LLM_PROVIDER_ENTRY_POINT_GROUP, LLMAdapter)

        for ep_name, cls in classes.items():
            try:
                provider_config = self.config.get(ep_name, {})

                # Prefer a factory classmethod if the adapter provides one
                if hasattr(cls, "create_provider") and callable(getattr(cls, "create_provider")):
                    info = cls.create_provider(provider_config)
                    if info is not None:
                        providers.append(info)
                    continue

                # Fallback: treat as OpenAI-compatible with base_url
                base_url = provider_config.get("base_url")
                api_key = provider_config.get("api_key") or os.environ.get(
                    f"{ep_name.upper()}_API_KEY", "external"
                )
                model = provider_config.get("model", "auto")

                if base_url:
                    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
                else:
                    client = openai.AsyncOpenAI(api_key=api_key)

                adapter = cls()
                name = getattr(cls, "provider_name", ep_name)

                providers.append(ProviderInfo(
                    name=name,
                    client=client,
                    adapter=adapter,
                    model=model,
                ))
            except Exception as e:
                logger.warning("Failed to initialize entry_point LLM provider '%s': %s", ep_name, e)

        return providers

    def _initialize_single_provider(self, provider_name: str) -> Optional[ProviderInfo]:
        """Initialize a single provider.

        Args:
            provider_name: Name of the provider to initialize

        Returns:
            ProviderInfo if successful, None if provider config not found

        Raises:
            Exception: If provider initialization fails
        """
        provider_config = self.config.get(provider_name)
        if not provider_config:
            logger.warning(f"Config for provider '{provider_name}' not found. Skipping.")
            return None

        if provider_name == "openai":
            return self._initialize_openai(provider_config)
        elif provider_name == "ollama":
            return self._initialize_ollama(provider_config)
        elif provider_name == "anthropic":
            return self._initialize_anthropic(provider_config)
        elif provider_name == "claude_max":
            return self._initialize_claude_max(provider_config)
        elif provider_name in ["google", "gemini"]:
            return self._initialize_google(provider_config)
        elif provider_name == "vertex_ai":
            return self._initialize_vertex_ai(provider_config)
        elif provider_name == "openrouter":
            return self._initialize_openrouter(provider_config)
        elif (provider_config.get("type") == "openai_compatible" or
              provider_name in ["azure_openai", "xai", "groq", "together", "mistral", "perplexity", "fireworks"]):
            return self._initialize_openai_compatible(provider_name, provider_config)
        else:
            logger.warning(f"Unknown provider '{provider_name}'. Skipping.")
            return None

    def _initialize_openai(self, provider_config: Dict[str, Any]) -> ProviderInfo:
        """Initialize OpenAI provider."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found.")

        model = os.environ.get("OPENAI_MODEL") or provider_config.get("model")
        if not model:
            logger.warning("No model configured for openai — set model= in llm_config.toml or OPENAI_MODEL env var")
            model = "auto"
        provider_config["model"] = model

        client = openai.AsyncOpenAI(api_key=api_key)
        adapter = OpenAIAdapter()

        return ProviderInfo(
            name="openai",
            client=client,
            adapter=adapter,
            model=model
        )

    def _initialize_ollama(self, provider_config: Dict[str, Any]) -> ProviderInfo:
        """Initialize Ollama provider."""
        if ollama is None:
            raise ImportError("ollama package not installed.")

        # Use get_ollama_url() for canonical URL resolution
        # Support legacy OLLAMA_HOST env var for backwards compatibility
        host = os.environ.get("OLLAMA_HOST") or provider_config.get("host") or get_ollama_url()
        model = os.environ.get("OLLAMA_MODEL") or provider_config.get("model")
        if not model:
            logger.warning("No model configured for ollama — set model= in llm_config.toml or OLLAMA_MODEL env var")
            model = "auto"
        provider_config["host"] = host
        provider_config["model"] = model

        client = ollama.AsyncClient(host=host)
        adapter = OllamaAdapter()

        return ProviderInfo(
            name="ollama",
            client=client,
            adapter=adapter,
            model=model
        )

    def _initialize_anthropic(self, provider_config: Dict[str, Any]) -> ProviderInfo:
        """Initialize Anthropic provider.
        
        Supports two authentication methods:
        1. API key: Set ANTHROPIC_API_KEY env var
        2. OAuth token (Claude Max): Set ANTHROPIC_AUTH_TOKEN env var
           (get token via `claude setup-token` CLI command)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed.")

        api_key = os.environ.get("ANTHROPIC_API_KEY") or provider_config.get("api_key")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or provider_config.get("auth_token")
        
        if not api_key and not auth_token:
            raise ValueError(
                "Anthropic authentication not found. Set either:\n"
                "  - ANTHROPIC_API_KEY for API key auth, or\n"
                "  - ANTHROPIC_AUTH_TOKEN for OAuth/Max subscription (from `claude setup-token`)"
            )

        # Prefer OAuth token if both are set (Max subscription is "free" usage)
        if auth_token:
            logger.info("Using Anthropic OAuth token (Claude Max subscription)")
            client = anthropic.AsyncAnthropic(auth_token=auth_token)
        else:
            client = anthropic.AsyncAnthropic(api_key=api_key)
        
        model = os.environ.get("ANTHROPIC_MODEL") or provider_config.get("model")
        if not model:
            logger.warning("No model configured for anthropic — set model= in llm_config.toml or ANTHROPIC_MODEL env var")
            model = "auto"
        provider_config["model"] = model

        adapter = AnthropicAdapter()

        return ProviderInfo(
            name="anthropic",
            client=client,
            adapter=adapter,
            model=model
        )

    def _initialize_claude_max(self, provider_config: Dict[str, Any]) -> ProviderInfo:
        """Initialize Claude Max provider using OAuth token auth.

        Uses the standard Anthropic SDK with auth_token instead of api_key.
        The token comes from ANTHROPIC_AUTH_TOKEN env var (via `claude login`).
        """
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or provider_config.get("auth_token")
        if not auth_token:
            raise ValueError(
                "ANTHROPIC_AUTH_TOKEN not set. "
                "Run `claude login` and export the token, or set auth_token in llm_config.toml."
            )

        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        model = provider_config.get("model")
        if not model:
            logger.warning("No model configured for claude_max — set model= in llm_config.toml")
            model = "auto"
        provider_config["model"] = model

        client = anthropic.AsyncAnthropic(auth_token=auth_token)
        adapter = ClaudeMaxAdapter()

        return ProviderInfo(
            name="claude_max",
            client=client,
            adapter=adapter,
            model=model
        )

    def _initialize_google(self, provider_config: Dict[str, Any]) -> ProviderInfo:
        """Initialize Google/Gemini provider."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed.")

        api_key = os.environ.get("GOOGLE_API_KEY") or provider_config.get("api_key")
        if not api_key:
            raise ValueError("Google API key not found.")

        model = os.environ.get("GOOGLE_MODEL") or provider_config.get("model")
        if not model:
            logger.warning("No model configured for google — set model= in llm_config.toml or GOOGLE_MODEL env var")
            model = "auto"
        provider_config["model"] = model

        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model)
        adapter = GoogleAdapter()

        return ProviderInfo(
            name="google",
            client=client,
            adapter=adapter,
            model=model
        )

    def _initialize_vertex_ai(self, provider_config: Dict[str, Any]) -> ProviderInfo:
        """Initialize Vertex AI provider."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package not installed.")

        model = os.environ.get("VERTEX_AI_MODEL") or provider_config.get("model")
        if not model:
            logger.warning("No model configured for vertex_ai — set model= in llm_config.toml or VERTEX_AI_MODEL env var")
            model = "auto"
        provider_config["model"] = model

        # Prefer API key (AI Studio) over service account (Vertex AI)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)
            adapter = VertexAIAdapter()
        else:
            project_id = (provider_config.get("project_id") or
                         os.environ.get("GCP_PROJECT_ID") or
                         os.environ.get("GOOGLE_CLOUD_PROJECT"))
            location = provider_config.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

            if not project_id:
                raise ValueError("Vertex AI requires GOOGLE_API_KEY or GCP_PROJECT_ID.")

            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            adapter = VertexAIAdapter(project_id=project_id, location=location)

        return ProviderInfo(
            name="vertex_ai",
            client=client,
            adapter=adapter,
            model=model
        )

    def _initialize_openrouter(self, provider_config: Dict[str, Any]) -> ProviderInfo:
        """Initialize OpenRouter provider."""
        api_key = os.environ.get("OPENROUTER_API_KEY") or provider_config.get("api_key")
        if not api_key:
            raise ValueError("OpenRouter API key not found (set OPENROUTER_API_KEY).")

        base_url = provider_config.get("base_url") or get_openrouter_api_base()
        model = os.environ.get("OPENROUTER_MODEL") or provider_config.get("model")
        if not model:
            logger.warning("No model configured for openrouter — set model= in llm_config.toml or OPENROUTER_MODEL env var")
            model = "auto"
        provider_config["model"] = model

        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        adapter = OpenRouterAdapter()

        return ProviderInfo(
            name="openrouter",
            client=client,
            adapter=adapter,
            model=model
        )

    def _initialize_openai_compatible(self, provider_name: str, provider_config: Dict[str, Any]) -> ProviderInfo:
        """Initialize OpenAI-compatible provider."""
        base_url = provider_config.get("base_url")
        if not base_url:
            raise ValueError(f"base_url must be set for '{provider_name}'.")

        is_local = provider_config.get("local", False)
        api_key_env = provider_config.get("api_key_env")
        api_key = None
        if api_key_env:
            api_key = os.environ.get(api_key_env)
        if not api_key:
            env_fallback = os.environ.get(f"{provider_name.upper()}_API_KEY")
            api_key = env_fallback or provider_config.get("api_key")
        if not api_key:
            if is_local:
                api_key = "local"  # Local servers don't need auth
            else:
                raise ValueError(f"API key not provided for '{provider_name}'.")

        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        adapter = OpenAIAdapter()

        model = provider_config.get("model", "auto")

        return ProviderInfo(
            name=provider_name,
            client=client,
            adapter=adapter,
            model=model
        )

    def get_provider_by_name(self, name: str) -> Optional[ProviderInfo]:
        """Get a provider by name.

        Args:
            name: Provider name to search for

        Returns:
            ProviderInfo if found, None otherwise
        """
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None

    def get_provider_for_model(self, model: str) -> Optional[ProviderInfo]:
        """Get the first provider that supports the specified model.

        Args:
            model: Model name to search for

        Returns:
            ProviderInfo if found, None otherwise
        """
        for provider in self.providers:
            if model in provider.model or provider.name in model:
                return provider
        return None

    def get_local_providers(self) -> List[ProviderInfo]:
        """Get all local providers (Ollama, llama.cpp, etc).

        Returns:
            List of local providers
        """
        local_names = {"ollama"}
        # Check config for providers marked as local
        for name, cfg in self.config.items():
            if isinstance(cfg, dict) and cfg.get("local", False):
                local_names.add(name)
        return [p for p in self.providers if p.name in local_names]

    def get_providers_with_pattern(self, patterns: List[str]) -> List[ProviderInfo]:
        """Get providers whose models match any of the given patterns.

        Args:
            patterns: List of patterns to match against model names

        Returns:
            List of matching providers
        """
        matching_providers = []
        for provider in self.providers:
            model_lower = provider.model.lower()
            for pattern in patterns:
                if pattern in model_lower:
                    matching_providers.append(provider)
                    break
        return matching_providers

    def update_provider_client(self, provider_name: str, new_client: Any) -> bool:
        """Update a provider's client (e.g., when switching to agent keys).

        Args:
            provider_name: Name of the provider to update
            new_client: New client instance

        Returns:
            True if provider was found and updated, False otherwise
        """
        provider = self.get_provider_by_name(provider_name)
        if provider:
            provider.client = new_client
            logger.info(f"Updated client for provider: {provider_name}")
            return True
        return False
