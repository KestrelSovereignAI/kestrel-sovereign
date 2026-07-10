import json
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from kestrel_sdk.llm import (
    ProviderCapabilities,
    StructuredOutputMode,
    ToolStreamingMode,
    VisionInputMode,
)

from kestrel_sovereign.llm.adapter import LLMAdapter
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.mock_adapter import MockAdapter
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from kestrel_sovereign.llm.provider_registry import ProviderInfo, ProviderRegistry
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
import kestrel_sovereign.llm.service as llm_service_module
from kestrel_sovereign.llm.embedding_service import get_provider_embedding_service


class BareAdapter(LLMAdapter):
    async def get_response(self, client, model, messages, **kwargs):
        return ""


class DictCapabilitiesAdapter:
    def provider_capabilities(self):
        return {
            "supports_tools": True,
            "supports_streaming": True,
            "supports_vision": True,
            "supports_structured_output": True,
            "supports_embeddings": True,
            "supports_inline_system": True,
            "structured_output_mode": "json_schema",
            "tool_streaming_mode": "native_delta",
            "vision_input_mode": "openai_image_url",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "model_dependent": ["vision"],
            "notes": ["plugin-style dict"],
        }


def test_provider_capabilities_to_dict_uses_wire_values():
    capabilities = ProviderCapabilities(
        supports_tools=True,
        structured_output_mode=StructuredOutputMode.JSON_SCHEMA,
        tool_streaming_mode=ToolStreamingMode.NATIVE_DELTA,
        vision_input_mode=VisionInputMode.OPENAI_IMAGE_URL,
        model_dependent=("vision",),
        notes=("example",),
    )

    data = capabilities.to_dict()

    # Enum fields serialize to their wire string values; scalars/tuples pass
    # through. Asserted field-by-field (not as an exact dict) so the additive
    # v5+ ProviderCapabilities surface doesn't make this brittle — the SDK's
    # own contract test owns the exhaustive field-set guard.
    expected_wire = {
        "supports_tools": True,
        "supports_streaming": False,
        "supports_vision": False,
        "supports_structured_output": False,
        "supports_embeddings": False,
        "supports_inline_system": False,
        "structured_output_mode": "json_schema",
        "tool_streaming_mode": "native_delta",
        "vision_input_mode": "openai_image_url",
        "embedding_model": None,
        "embedding_dim": None,
        "model_dependent": ["vision"],
        "notes": ["example"],
    }
    for key, value in expected_wire.items():
        assert data[key] == value, key

    # v5 additive fields are present at their conservative defaults.
    assert data["supports_batch"] is False
    assert data["batch_mode"] == "none"

    # to_dict() round-trips losslessly through from_mapping().
    assert ProviderCapabilities.from_mapping(data) == capabilities


def test_adapter_capabilities_normalizes_plugin_dicts():
    capabilities = ProviderCapabilities.from_mapping(
        DictCapabilitiesAdapter().provider_capabilities()
    )

    assert capabilities.supports_tools is True
    assert capabilities.supports_structured_output is True
    assert capabilities.supports_embeddings is True
    assert capabilities.supports_inline_system is True
    assert capabilities.embedding_model == "text-embedding-3-small"
    assert capabilities.embedding_dim == 1536
    assert capabilities.structured_output_mode == StructuredOutputMode.JSON_SCHEMA
    assert capabilities.model_dependent == ("vision",)


def test_base_adapter_capabilities_are_conservative():
    capabilities = BareAdapter().provider_capabilities()

    assert capabilities == ProviderCapabilities()


def test_llm_service_route_dicts_include_capabilities():
    service = LLMService.__new__(LLMService)
    adapter = OpenAIAdapter()
    provider = ProviderInfo(
        name="openai:api",
        vendor="openai",
        route="api",
        client=object(),
        adapter=adapter,
        model="gpt-5",
        is_cloud=True,
        is_local=False,
        selection_hints=("fast",),
        capabilities=adapter.provider_capabilities(),
    )

    [route] = service._convert_providers_format([provider])

    assert route["selection_hints"] == ["fast"]
    assert route["capabilities"]["supports_tools"] is True
    assert route["capabilities"]["supports_vision"] is True
    assert route["capabilities"]["supports_structured_output"] is True
    assert route["capabilities"]["supports_embeddings"] is True
    assert route["capabilities"]["embedding_model"] == "text-embedding-3-small"
    assert route["capabilities"]["embedding_dim"] == 1536
    assert route["capabilities"]["structured_output_mode"] == "json_schema"


def test_llm_service_embedding_provider_follows_active_route():
    service = LLMService.__new__(LLMService)
    openai_route = {
        "name": "openai:api",
        "adapter": OpenAIAdapter(),
        "client": object(),
        "capabilities": OpenAIAdapter().provider_capabilities().to_dict(),
    }
    service.resolve_provider_routing = lambda **_: ([openai_route], None)

    provider = service.resolve_embedding_provider()

    assert provider is openai_route
    embedding_service = service.get_embedding_service()
    assert embedding_service.provider is openai_route
    assert embedding_service.model == "text-embedding-3-small"
    assert embedding_service.embedding_dim == 1536


def test_llm_service_embedding_provider_degrades_when_active_route_cannot_embed():
    service = LLMService.__new__(LLMService)
    anthropic_route = {
        "name": "anthropic:api",
        "adapter": AnthropicAdapter(),
        "client": object(),
        "capabilities": AnthropicAdapter().provider_capabilities().to_dict(),
    }
    service.resolve_provider_routing = lambda **_: ([anthropic_route], None)

    assert service.resolve_embedding_provider() is None
    assert service.get_embedding_service() is None


def test_llm_service_embedding_provider_honors_force_local_only_callback():
    """#1492 — when the privacy gate says local-only, the embedding path
    must filter to local routes even if a cloud route is at higher
    priority. Reproducer: OpenAI route configured first, Ollama
    second; ISOLATED/EPHEMERAL must reach Ollama for embeddings, not
    OpenAI."""
    service = LLMService.__new__(LLMService)
    openai_route = {
        "name": "openai:api",
        "is_local": False,
        "is_cloud": True,
        "adapter": OpenAIAdapter(),
        "client": object(),
        "capabilities": OpenAIAdapter().provider_capabilities().to_dict(),
    }
    ollama_route = {
        "name": "ollama:local",
        "is_local": True,
        "is_cloud": False,
        "adapter": OpenAIAdapter(),  # adapter doesn't matter for gate test
        "client": object(),
        "capabilities": OpenAIAdapter().provider_capabilities().to_dict(),
    }

    def routing(**kwargs):
        if kwargs.get("force_local_only"):
            return [ollama_route], None
        return [openai_route, ollama_route], None

    service.resolve_provider_routing = routing

    # No gate bound → cloud route wins (pre-#1492 behavior, used by
    # CLI/test entry points without an agent attached).
    assert service.resolve_embedding_provider() is openai_route

    # Bind the gate to "local-only" (ISOLATED/EPHEMERAL).
    service.set_force_local_only_provider(lambda: True)
    assert service.resolve_embedding_provider() is ollama_route

    # Flip back to NORMAL — cloud route reachable again.
    service.set_force_local_only_provider(lambda: False)
    assert service.resolve_embedding_provider() is openai_route


def test_llm_service_embedding_provider_fails_safely_when_provider_raises():
    """#1492 — a misbehaving privacy callback must default to local-only.
    Better to lose embedding than to leak plaintext."""
    service = LLMService.__new__(LLMService)
    ollama_route = {
        "name": "ollama:local",
        "is_local": True,
        "adapter": OpenAIAdapter(),
        "client": object(),
        "capabilities": OpenAIAdapter().provider_capabilities().to_dict(),
    }

    def routing(**kwargs):
        if kwargs.get("force_local_only"):
            return [ollama_route], None
        raise AssertionError(
            "force_local_only must be True after privacy callback raises"
        )

    service.resolve_provider_routing = routing

    def boom() -> bool:
        raise RuntimeError("privacy state read failed")

    service.set_force_local_only_provider(boom)
    # Falls closed → ollama route used.
    assert service.resolve_embedding_provider() is ollama_route


def test_llm_service_embedding_returns_none_when_no_local_route():
    """#1492 — under force_local_only, if no local route exists, embedding
    must return None (keyword fallback) rather than propagate the
    underlying RuntimeError from resolve_provider_routing."""
    service = LLMService.__new__(LLMService)

    def routing(**kwargs):
        if kwargs.get("force_local_only"):
            raise RuntimeError("No local providers available.")
        return [], None

    service.resolve_provider_routing = routing
    service.set_force_local_only_provider(lambda: True)

    assert service.resolve_embedding_provider() is None
    assert service.get_embedding_service() is None


# --- #1494 embedding-sibling-route tests ------------------------------------

def _route(
    name: str,
    vendor: str,
    adapter,
    *,
    is_local: bool = False,
    embedding_sibling: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the dict shape produced by ``_convert_providers_format`` so
    each sibling-route test can wire a small provider table without
    hand-rolling every key. ``adapter`` chooses whether embeddings are
    supported (OpenAI does, Anthropic doesn't)."""
    return {
        "name": name,
        "vendor": vendor,
        "route": name.split(":", 1)[1] if ":" in name else "api",
        "adapter": adapter,
        "client": object(),
        "model": "auto",
        "is_local": is_local,
        "is_cloud": not is_local,
        "capabilities": adapter.provider_capabilities().to_dict(),
        "embedding_sibling": embedding_sibling,
    }


def test_sibling_used_when_primary_lacks_embeddings():
    """#1494 — Anthropic chat + OpenAI sibling + cloud-allowed →
    embedding routes to OpenAI (not the chat provider)."""
    service = LLMService.__new__(LLMService)
    anthropic = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling="openai:api",
    )
    openai = _route("openai:api", "openai", OpenAIAdapter())
    service.providers = [anthropic, openai]
    service.resolve_provider_routing = lambda **_: ([anthropic], None)

    assert service.resolve_embedding_provider() is openai


def test_sibling_skipped_when_primary_has_own_embeddings():
    """OpenAI chat + sibling configured anyway → own embeddings win.
    The sibling is only a fallback for providers that can't embed."""
    service = LLMService.__new__(LLMService)
    openai = _route(
        "openai:api", "openai", OpenAIAdapter(),
        embedding_sibling="ollama:local",
    )
    ollama = _route("ollama:local", "ollama", OpenAIAdapter(), is_local=True)
    service.providers = [openai, ollama]
    service.resolve_provider_routing = lambda **_: ([openai], None)

    # Primary supports embeddings → it wins. Sibling unused.
    assert service.resolve_embedding_provider() is openai


def test_sibling_rejected_when_non_local_under_force_local_only():
    """ISOLATED/EPHEMERAL must reject a cloud sibling. Privacy wins —
    operator who configured ``embedding_sibling = "openai"`` for
    Anthropic gets zero embeddings in local-only modes."""
    service = LLMService.__new__(LLMService)
    anthropic = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling="openai:api",
    )
    openai = _route("openai:api", "openai", OpenAIAdapter(), is_local=False)
    service.providers = [anthropic, openai]
    # Under force_local_only, resolve_provider_routing would normally
    # filter to local routes — but if no local route exists it raises
    # ``RuntimeError("No local providers available.")`` which the
    # embedding path catches and returns None. To exercise the sibling
    # path specifically (primary route IS the Anthropic one because
    # the routing layer returned it under non-local-only resolution),
    # we simulate the chain: routing returns Anthropic, the sibling
    # path then filters by force_local_only and rejects the cloud
    # sibling.
    service.resolve_provider_routing = lambda **kwargs: (
        [anthropic],
        None,
    )
    service.set_force_local_only_provider(lambda: True)

    assert service.resolve_embedding_provider() is None


def test_local_sibling_accepted_under_force_local_only():
    """Operator who configured a local sibling (Ollama) keeps semantic
    memory in ISOLATED/EPHEMERAL — the privacy invariant is preserved."""
    service = LLMService.__new__(LLMService)
    anthropic = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling="ollama:local",
    )
    ollama = _route("ollama:local", "ollama", OpenAIAdapter(), is_local=True)
    service.providers = [anthropic, ollama]
    service.resolve_provider_routing = lambda **kwargs: ([anthropic], None)
    service.set_force_local_only_provider(lambda: True)

    assert service.resolve_embedding_provider() is ollama


def test_sibling_returns_none_when_primary_has_no_sibling_configured():
    """Anthropic chat + no sibling → embedding path returns None and
    storage falls back to keyword search. Pre-#1494 behavior preserved
    when no sibling is set."""
    service = LLMService.__new__(LLMService)
    anthropic = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling=None,
    )
    service.providers = [anthropic]
    service.resolve_provider_routing = lambda **_: ([anthropic], None)

    assert service.resolve_embedding_provider() is None


def test_sibling_returns_none_when_sibling_not_initialized():
    """Operator configured ``embedding_sibling = "openai:api"`` but
    didn't set ``OPENAI_API_KEY`` — OpenAI never initialized, so the
    sibling lookup misses and we fall back to keyword."""
    service = LLMService.__new__(LLMService)
    anthropic = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling="openai:api",
    )
    # OpenAI NOT in providers — simulating "key missing, route skipped".
    service.providers = [anthropic]
    service.resolve_provider_routing = lambda **_: ([anthropic], None)

    assert service.resolve_embedding_provider() is None


def test_sibling_returns_none_when_sibling_cannot_embed():
    """Sibling pointed at another non-embedding provider (e.g.
    Anthropic→Anthropic). Should NOT recurse — sibling resolution is
    one hop only."""
    service = LLMService.__new__(LLMService)
    anthropic_a = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling="claude-max:plan",
    )
    anthropic_b = _route(
        "claude-max:plan", "claude-max",
        AnthropicAdapter(),  # adapter has no embedding capability
        embedding_sibling=None,
    )
    service.providers = [anthropic_a, anthropic_b]
    service.resolve_provider_routing = lambda **_: ([anthropic_a], None)

    assert service.resolve_embedding_provider() is None


def test_sibling_lookup_accepts_vendor_only_form():
    """``embedding_sibling = "openai"`` (no route) resolves to the first
    matching initialized route for that vendor."""
    service = LLMService.__new__(LLMService)
    anthropic = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling="openai",
    )
    openai_api = _route("openai:api", "openai", OpenAIAdapter())
    openai_compat = _route("openai:compat", "openai", OpenAIAdapter())
    service.providers = [anthropic, openai_api, openai_compat]
    service.resolve_provider_routing = lambda **_: ([anthropic], None)

    assert service.resolve_embedding_provider() is openai_api


def test_sibling_lookup_skips_disabled_routes():
    """#1494 — codex P2 regression: a sibling route disabled after a
    permanent auth failure (``_disabled_routes``) must NOT be returned
    by the sibling lookup. Otherwise every storage write keeps
    retrying known-bad credentials until the process restarts."""
    service = LLMService.__new__(LLMService)
    anthropic = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling="openai:api",
    )
    openai = _route("openai:api", "openai", OpenAIAdapter())
    service.providers = [anthropic, openai]
    service._disabled_routes = {"openai:api": "auth_failed"}
    # Mimic the real ``_available_providers`` filter.
    service._available_providers = lambda: [
        p for p in service.providers if p["name"] not in service._disabled_routes
    ]
    service.resolve_provider_routing = lambda **_: ([anthropic], None)

    # Sibling is disabled → fall through to keyword (None).
    assert service.resolve_embedding_provider() is None


def test_sibling_does_not_recurse_via_siblings_own_sibling():
    """Sibling resolution is one hop only — even if the sibling itself
    declares a sibling that supports embeddings, we don't chain. This
    keeps "what provider embedded this row?" predictable for
    embedding_profile_id stamping (#1477)."""
    service = LLMService.__new__(LLMService)
    anthropic = _route(
        "anthropic:api", "anthropic", AnthropicAdapter(),
        embedding_sibling="claude-max:plan",
    )
    # Sibling can't embed but declares its own sibling that can. We
    # must NOT follow the chain — return None.
    claude_max = _route(
        "claude-max:plan", "claude-max", AnthropicAdapter(),
        embedding_sibling="openai:api",
    )
    openai = _route("openai:api", "openai", OpenAIAdapter())
    service.providers = [anthropic, claude_max, openai]
    service.resolve_provider_routing = lambda **_: ([anthropic], None)

    assert service.resolve_embedding_provider() is None


def test_provider_registry_parses_route_level_sibling():
    """Route-level ``embedding_sibling`` reaches the dict shape via the
    private-attr bridge between ProviderInfo and the routing dict."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    registry = ProviderRegistry({})
    vendor_cfg = {"is_cloud": True}
    route_cfg = {
        "adapter": "AnthropicAdapter",
        "api_key_env": "X_UNSET_KEY",
        "model": "claude-3-opus-20240229",
        "embedding_sibling": "openai:api",
    }
    # Patch the secret resolver so initialization can proceed
    # without a real API key.
    registry._resolve_secret = lambda rc, env_key, plain_key: "fake-key"
    info = registry._build_route("anthropic", "api", vendor_cfg, route_cfg)
    assert info is not None
    assert getattr(info, "_kestrel_embedding_sibling", None) == "openai:api"

    # _convert_providers_format propagates the attr to the dict.
    service = LLMService.__new__(LLMService)
    [route_dict] = service._convert_providers_format([info])
    assert route_dict["embedding_sibling"] == "openai:api"


def test_provider_registry_route_level_sibling_overrides_vendor_level():
    """Route-level config wins over vendor-level. Operator can DRY the
    vendor-level setting yet override on a specific route."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    registry = ProviderRegistry({})
    vendor_cfg = {
        "is_cloud": True,
        "embedding_sibling": "openai:api",
    }
    route_cfg = {
        "adapter": "AnthropicAdapter",
        "api_key_env": "X_UNSET_KEY",
        "embedding_sibling": "ollama:local",
    }
    registry._resolve_secret = lambda rc, env_key, plain_key: "fake-key"
    info = registry._build_route("anthropic", "api", vendor_cfg, route_cfg)
    assert info is not None
    assert getattr(info, "_kestrel_embedding_sibling", None) == "ollama:local"


def test_provider_registry_vendor_level_sibling_propagates_when_route_omits():
    """If route omits the key, vendor-level supplies it."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    registry = ProviderRegistry({})
    vendor_cfg = {
        "is_cloud": True,
        "embedding_sibling": "openai:api",
    }
    route_cfg = {
        "adapter": "AnthropicAdapter",
        "api_key_env": "X_UNSET_KEY",
        # no embedding_sibling at route level
    }
    registry._resolve_secret = lambda rc, env_key, plain_key: "fake-key"
    info = registry._build_route("anthropic", "api", vendor_cfg, route_cfg)
    assert info is not None
    assert getattr(info, "_kestrel_embedding_sibling", None) == "openai:api"


def test_provider_registry_rejects_non_string_sibling():
    """Type errors on the config side fail loudly at registry init —
    better to crash with a clear message than to silently ignore."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry
    import pytest

    registry = ProviderRegistry({})
    vendor_cfg = {"is_cloud": True}
    route_cfg = {
        "adapter": "AnthropicAdapter",
        "api_key_env": "X_UNSET_KEY",
        "embedding_sibling": ["openai:api"],  # type error: list not string
    }
    registry._resolve_secret = lambda rc, env_key, plain_key: "fake-key"
    with pytest.raises(ValueError, match="embedding_sibling must be a string"):
        registry._build_route("anthropic", "api", vendor_cfg, route_cfg)


def test_provider_registry_blank_sibling_normalizes_to_none():
    """``embedding_sibling = ""`` or whitespace normalizes to None —
    saves a class of confusing 'set but empty' bugs."""
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    registry = ProviderRegistry({})
    vendor_cfg = {"is_cloud": True}
    route_cfg = {
        "adapter": "AnthropicAdapter",
        "api_key_env": "X_UNSET_KEY",
        "embedding_sibling": "   ",
    }
    registry._resolve_secret = lambda rc, env_key, plain_key: "fake-key"
    info = registry._build_route("anthropic", "api", vendor_cfg, route_cfg)
    assert info is not None
    assert getattr(info, "_kestrel_embedding_sibling", None) is None


# --- #2263 top-level embedding_route knob -----------------------------------

def _embedding_service(providers, *, embedding_route=None):
    """Build a bare LLMService wired for embedding-route resolution tests."""
    service = LLMService.__new__(LLMService)
    service.providers = list(providers)
    service._disabled_routes = {}
    service._embedding_route = embedding_route
    # Explicit branch is terminal — routing must NOT be consulted when the knob
    # resolves. Point it at an assertion so any accidental fall-through fails.
    service.resolve_provider_routing = lambda **_: (
        (_ for _ in ()).throw(
            AssertionError("explicit embedding_route branch must be terminal")
        )
    )
    return service


def test_explicit_embedding_route_wins_over_native_chat_route():
    """#2263 — an explicit embedding_route wins even when the active chat route
    embeds natively (the deliberate-choice-wins rule)."""
    openai = _route("openai:api", "openai", OpenAIAdapter())
    ollama = _route("ollama:local", "ollama", OpenAIAdapter(), is_local=True)
    service = _embedding_service([openai, ollama], embedding_route="ollama:local")

    assert service.resolve_embedding_provider() is ollama


def test_explicit_embedding_route_used_when_chat_route_cannot_embed():
    """#2263 — explicit route resolves for a chat route that can't embed."""
    anthropic = _route("anthropic:api", "anthropic", AnthropicAdapter())
    openai = _route("openai:api", "openai", OpenAIAdapter())
    service = _embedding_service([anthropic, openai], embedding_route="openai:api")

    assert service.resolve_embedding_provider() is openai


def test_explicit_embedding_route_refused_under_force_local_only():
    """#2263 + #1492 — the privacy gate applies to the explicit knob too: a
    cloud embedding_route is refused for local-only sessions (keyword
    fallback), never a crash."""
    openai = _route("openai:api", "openai", OpenAIAdapter(), is_local=False)
    service = _embedding_service([openai], embedding_route="openai:api")
    service.set_force_local_only_provider(lambda: True)

    assert service.resolve_embedding_provider() is None


def test_explicit_local_embedding_route_accepted_under_force_local_only():
    """#2263 — a local explicit route survives local-only sessions."""
    ollama = _route("ollama:local", "ollama", OpenAIAdapter(), is_local=True)
    service = _embedding_service([ollama], embedding_route="ollama:local")
    service.set_force_local_only_provider(lambda: True)

    assert service.resolve_embedding_provider() is ollama


def test_explicit_embedding_route_missing_provider_falls_back():
    """#2263 — an embedding_route pointing at an uninitialized provider falls
    back to keyword search (None), terminal — does not consult the chat route."""
    anthropic = _route("anthropic:api", "anthropic", AnthropicAdapter())
    service = _embedding_service([anthropic], embedding_route="openai:api")

    assert service.resolve_embedding_provider() is None


def test_explicit_embedding_route_non_embedding_route_falls_back():
    """#2263 — an embedding_route naming a route with no embedding support
    falls back to keyword search."""
    anthropic = _route("anthropic:api", "anthropic", AnthropicAdapter())
    service = _embedding_service([anthropic], embedding_route="anthropic:api")

    assert service.resolve_embedding_provider() is None


def test_explicit_embedding_route_vendor_only_form():
    """#2263 — a bare ``"<vendor>"`` embedding_route resolves to the first
    matching route for that vendor."""
    anthropic = _route("anthropic:api", "anthropic", AnthropicAdapter())
    openai = _route("openai:api", "openai", OpenAIAdapter())
    service = _embedding_service([anthropic, openai], embedding_route="openai")

    assert service.resolve_embedding_provider() is openai


def test_set_embedding_route_round_trip_and_clear():
    """#2263 — set/get round-trip, and clear returns to auto (None)."""
    service = LLMService.__new__(LLMService)
    service.providers = [_route("openai:api", "openai", OpenAIAdapter())]
    service._embedding_route = None
    service._embedding_route_persistence_callback = None

    service.set_embedding_route("openai:api")
    assert service.get_embedding_route() == "openai:api"

    # "auto" and "" normalize to None (cleared).
    service.set_embedding_route("auto")
    assert service.get_embedding_route() is None
    service.set_embedding_route("openai:api")
    service.clear_embedding_route()
    assert service.get_embedding_route() is None


def test_set_embedding_route_validation_rejects_unknown_and_non_embedding():
    """#2263 — set-time validation rejects an unknown route and a route with no
    embedding support."""
    import pytest

    service = LLMService.__new__(LLMService)
    service.providers = [
        _route("anthropic:api", "anthropic", AnthropicAdapter()),
        _route("openai:api", "openai", OpenAIAdapter()),
    ]
    service._embedding_route = None
    service._embedding_route_persistence_callback = None

    with pytest.raises(ValueError, match="no configured route matches"):
        service.set_embedding_route("gemini:api")
    with pytest.raises(ValueError, match="does not advertise embedding support"):
        service.set_embedding_route("anthropic:api")
    # The good route still sets.
    service.set_embedding_route("openai:api")
    assert service.get_embedding_route() == "openai:api"


# --- #2326 live upstream probe on explicit set -------------------------------


class _ProbeAdapter:
    """Adapter whose embed either raises or returns a fixed vector, so the
    #2326 live-probe path can be exercised without a real provider."""

    def __init__(self, *, raise_exc: Optional[Exception] = None, vector=None):
        self._raise_exc = raise_exc
        self._vector = vector if vector is not None else [0.1, 0.2, 0.3]

    def provider_capabilities(self):
        return OpenAIAdapter().provider_capabilities()

    async def aembed(self, client, text, model=None):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._vector

    async def aembed_batch(self, client, texts, model=None):
        return [self._vector for _ in texts]


def _probe_service(providers):
    service = LLMService.__new__(LLMService)
    service.providers = list(providers)
    service._disabled_routes = {}
    service._embedding_route = None
    service._embedding_route_persistence_callback = None
    return service


async def test_aset_embedding_route_refuses_dead_upstream_cloud_route():
    """#2326 — a cloud route that passes static validation but whose upstream
    model is dead (aembed 404s) is refused with the upstream error surfaced."""
    import pytest

    dead = _route(
        "openrouter:api", "openrouter",
        _ProbeAdapter(raise_exc=Exception(
            "Error code: 404 - No endpoints found for qwen/qwen3-embedding-0.6b."
        )),
    )
    service = _probe_service([dead])

    with pytest.raises(ValueError, match="live embedding probe failed"):
        await service.aset_embedding_route("openrouter:api")
    # The knob must NOT have been set — refusing means the prior state stands.
    assert service.get_embedding_route() is None


async def test_aset_embedding_route_refuses_when_probe_returns_no_vector():
    """#2326 — an upstream that returns no embedding (None/empty) is refused."""
    import pytest

    empty = _route("openrouter:api", "openrouter", _ProbeAdapter(vector=[]))
    service = _probe_service([empty])

    with pytest.raises(ValueError, match="returned no embedding"):
        await service.aset_embedding_route("openrouter:api")
    assert service.get_embedding_route() is None


async def test_aset_embedding_route_accepts_live_cloud_route():
    """#2326 — a cloud route whose canary probe succeeds is accepted and set."""
    live = _route("openrouter:api", "openrouter", _ProbeAdapter(vector=[0.4, 0.5]))
    service = _probe_service([live])

    await service.aset_embedding_route("openrouter:api")
    assert service.get_embedding_route() == "openrouter:api"


async def test_aset_embedding_route_skips_probe_for_local_route():
    """#2326 — local routes are not live-probed (the empty-pool failure mode is
    a cloud-meta-provider property); a would-be-failing local adapter still
    sets."""
    local = _route(
        "ollama:local", "ollama",
        _ProbeAdapter(raise_exc=Exception("would fail if probed")),
        is_local=True,
    )
    service = _probe_service([local])

    await service.aset_embedding_route("ollama:local")
    assert service.get_embedding_route() == "ollama:local"


async def test_aset_embedding_route_refuses_disabled_route():
    """#2326 — a route that statically matches an embedding-capable provider but
    is currently disabled (``_disabled_routes``) resolves to no *available*
    probe target. Committing it would degrade every write to keyword fallback
    with no probe, so the set is refused rather than silently accepted."""
    import pytest

    disabled = _route("openrouter:api", "openrouter", _ProbeAdapter(vector=[0.1]))
    service = _probe_service([disabled])
    service._disabled_routes = {"openrouter:api": "auth_failed"}

    with pytest.raises(ValueError, match="not currently available"):
        await service.aset_embedding_route("openrouter:api")
    assert service.get_embedding_route() is None


async def test_aset_embedding_route_still_enforces_static_validation():
    """#2326 — the async setter runs static validation first: an unknown route
    is refused before any probe."""
    import pytest

    openai = _route("openai:api", "openai", _ProbeAdapter(vector=[0.1]))
    service = _probe_service([openai])

    with pytest.raises(ValueError, match="no configured route matches"):
        await service.aset_embedding_route("gemini:api")


async def test_aset_embedding_route_none_and_auto_skip_probe():
    """#2326 — the ``none`` off-switch and ``auto`` clear are never probed."""
    live = _route("openrouter:api", "openrouter", _ProbeAdapter(vector=[0.1]))
    service = _probe_service([live])

    await service.aset_embedding_route("none")
    assert service.get_embedding_route() == "none"
    await service.aset_embedding_route("auto")
    assert service.get_embedding_route() is None


async def test_load_persisted_embedding_route_dead_route_logs_loud_no_crash(caplog):
    """#2326 — boot-time load of a persisted (possibly dead-upstream) route does
    NOT probe: it applies via the sync setter and logs loudly, never crashing."""
    import logging as _logging

    from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin

    mixin = ModelPreferenceMixin.__new__(ModelPreferenceMixin)
    mixin.agent_id = "a1"

    dead = _route("openrouter:api", "openrouter", _ProbeAdapter(
        raise_exc=Exception("404 - No endpoints found")))
    llm = _probe_service([dead])
    mixin.llm_service = llm

    class _DB:
        async def fetchall(self, *_a, **_k):
            return [(json.dumps("openrouter:api"),)]

    mixin._raw_storage = SimpleNamespace(db=_DB())

    with caplog.at_level(_logging.WARNING):
        await mixin._load_embedding_route()

    # No crash, route applied without a live probe, and a loud warning emitted.
    assert llm.get_embedding_route() == "openrouter:api"
    assert any(
        "without a live upstream probe" in r.getMessage() for r in caplog.records
    )


# --- #2287 first-class "none" (embeddings deliberately off) ------------------


def test_embedding_route_none_short_circuits_resolution():
    """#2287 — ``embedding_route == "none"`` is a deliberate off-switch:
    ``resolve_embedding_provider`` short-circuits to None at step 0, even when
    the active chat route embeds natively. Routing must NOT be consulted."""
    openai = _route("openai:api", "openai", OpenAIAdapter())
    service = _embedding_service([openai], embedding_route="none")

    # The chat route (openai) embeds natively, yet "none" wins — and the
    # terminal short-circuit means resolve_provider_routing (rigged to raise)
    # is never reached.
    assert service.resolve_embedding_provider() is None
    assert service.get_embedding_service() is None


def test_set_embedding_route_none_skips_validation_and_persists():
    """#2287 — setting ``"none"`` bypasses provider validation (it names no
    route) and round-trips through get_embedding_route. Casing/whitespace is
    canonicalized to the bare sentinel."""
    service = LLMService.__new__(LLMService)
    # No embedding-capable route configured — an explicit selector would be
    # refused, but "none" must still be accepted.
    service.providers = [_route("anthropic:api", "anthropic", AnthropicAdapter())]
    service._embedding_route = None
    service._embedding_route_persistence_callback = None

    service.set_embedding_route("none")
    assert service.get_embedding_route() == "none"
    # Canonicalization: mixed case / padding still lands on "none".
    service._embedding_route = None
    service.set_embedding_route("  NONE ")
    assert service.get_embedding_route() == "none"

    # Auto still clears back to None (distinct from off).
    service.set_embedding_route("auto")
    assert service.get_embedding_route() is None


def test_get_embedding_settings_reports_resolved_state_and_dims():
    """#2263 — GET-settings surfaces configured route, resolved route, model,
    embedding_dim, and the deployment KESTREL_EMBEDDING_DIM."""
    openai = _route("openai:api", "openai", OpenAIAdapter())
    service = _embedding_service([openai], embedding_route="openai:api")

    settings = service.get_embedding_settings()
    assert settings["embedding_route"] == "openai:api"
    assert settings["resolved_route"] == "openai:api"
    assert settings["embedding_model"] == "text-embedding-3-small"
    assert settings["embedding_dim"] == 1536
    assert "kestrel_embedding_dim" in settings
    assert isinstance(settings["kestrel_embedding_dim"], int)


def test_get_embedding_settings_auto_default():
    """#2263 — with no knob set and a native chat route, settings report
    embedding_route=None (auto) and the resolved native route."""
    service = LLMService.__new__(LLMService)
    openai = _route("openai:api", "openai", OpenAIAdapter())
    service.providers = [openai]
    service._disabled_routes = {}
    service._embedding_route = None
    service.resolve_provider_routing = lambda **_: ([openai], None)

    settings = service.get_embedding_settings()
    assert settings["embedding_route"] is None
    assert settings["resolved_route"] == "openai:api"
    assert settings["embedding_dim"] == 1536


def test_llm_service_embedding_provider_honors_disabled_policy():
    service = LLMService.__new__(LLMService)
    service.disabled = True
    service.resolve_provider_routing = Mock(
        side_effect=AssertionError("disabled service must not resolve routes")
    )

    assert service.resolve_embedding_provider() is None
    assert service.get_embedding_service() is None


def test_default_provider_embedding_service_resolves_each_call(monkeypatch):
    class FakeLLMService:
        calls = 0

        def get_embedding_service(self):
            type(self).calls += 1
            return SimpleNamespace(model=f"embed-{type(self).calls}")

    monkeypatch.setattr(llm_service_module, "LLMService", FakeLLMService)

    first = get_provider_embedding_service()
    second = get_provider_embedding_service()

    assert first.model == "embed-1"
    assert second.model == "embed-2"


async def test_provider_embedding_service_uses_common_batch_contract():
    from kestrel_sovereign.llm.embedding_service import ProviderEmbeddingService

    adapter = SimpleNamespace(
        aembed=AsyncMock(return_value=[1.0, 2.0]),
        aembed_batch=AsyncMock(return_value=[[1.0, 2.0], None]),
    )
    client = object()
    service = ProviderEmbeddingService(
        {
            "adapter": adapter,
            "client": client,
            "capabilities": {
                "embedding_model": "embed-model",
                "embedding_dim": 2,
            },
        }
    )

    assert await service.aembed("hello") == [1.0, 2.0]
    assert await service.aembed_batch(["a", "b"]) == [[1.0, 2.0], None]
    adapter.aembed.assert_awaited_once_with(client, "hello", model="embed-model")
    adapter.aembed_batch.assert_awaited_once_with(
        client, ["a", "b"], model="embed-model"
    )


async def test_openai_adapter_embeddings_use_route_client():
    adapter = OpenAIAdapter()
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    data=[
                        SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                        SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                    ]
                )
            )
        )
    )

    assert await adapter.aembed(client, "one") == [1.0, 0.0]
    assert await adapter.aembed_batch(client, ["one", "two"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


async def test_google_adapter_embeddings_normalize_plain_float_vector():
    adapter = GoogleAdapter()
    models = SimpleNamespace(
        embed_content=AsyncMock(
            return_value=SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])]
            )
        )
    )
    client = SimpleNamespace(aio=SimpleNamespace(models=models))

    assert await adapter.aembed(client, "hello", model="text-embedding-004") == [
        0.1,
        0.2,
    ]
    models.embed_content.assert_awaited_with(
        model="text-embedding-004",
        contents="hello",
    )


async def test_vertex_adapter_embeddings_normalize_batch_response():
    adapter = VertexAIAdapter(project_id="test-project")
    models = SimpleNamespace(
        embed_content=AsyncMock(
            return_value=SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[0.1, 0.2]),
                    SimpleNamespace(values=[0.3, 0.4]),
                ]
            )
        )
    )
    client = SimpleNamespace(aio=SimpleNamespace(models=models))

    assert await adapter.aembed_batch(client, ["a", "b"]) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    models.embed_content.assert_awaited_once_with(
        model="text-embedding-004",
        contents=["a", "b"],
    )


async def test_ollama_adapter_embeddings_use_nomic_default():
    adapter = OllamaAdapter()
    client = SimpleNamespace(
        embed=AsyncMock(return_value={"embeddings": [[0.1, 0.2]]})
    )

    assert await adapter.aembed(client, "hello") == [0.1, 0.2]
    client.embed.assert_awaited_once_with(model="nomic-embed-text", input="hello")


async def test_ollama_adapter_batch_embeddings_preserve_input_count():
    adapter = OllamaAdapter()
    client = SimpleNamespace(
        embed=AsyncMock(return_value={"embeddings": [[0.1, 0.2]]})
    )

    assert await adapter.aembed_batch(client, ["a", "b"]) == [[0.1, 0.2], None]
    client.embed.assert_awaited_once_with(model="nomic-embed-text", input=["a", "b"])


def test_provider_registry_sets_sdk_capabilities_on_built_routes(monkeypatch):
    registry = ProviderRegistry(
        {
            "vendors": {
                "openai": {
                    "routes": {
                        "api": {
                            "adapter": "OpenAIAdapter",
                            "model": "gpt-5",
                        }
                    }
                }
            }
        }
    )
    adapter = OpenAIAdapter()

    monkeypatch.setattr(
        registry,
        "_build_client_and_adapter",
        lambda **_: (object(), adapter),
    )

    info = registry._build_route(
        "openai",
        "api",
        {"is_cloud": True},
        {"adapter": "OpenAIAdapter", "model": "gpt-5"},
    )

    assert info.capabilities == adapter.provider_capabilities()
    assert info.capabilities.supports_tools is True


def test_openai_compatible_routes_do_not_inherit_openai_embeddings_by_default():
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry.config = {}

    _client, adapter = registry._build_client_and_adapter(
        vendor="xai",
        route="api",
        adapter_cls=OpenAIAdapter,
        vendor_cfg={},
        route_cfg={
            "base_url": "https://api.x.ai/v1",
            "api_key": "test-key",
        },
    )

    capabilities = adapter.provider_capabilities()
    assert capabilities.supports_embeddings is False
    assert capabilities.embedding_model is None
    assert capabilities.embedding_dim is None


def test_official_openai_base_url_keeps_default_embeddings():
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry.config = {}

    _client, adapter = registry._build_client_and_adapter(
        vendor="openai",
        route="api",
        adapter_cls=OpenAIAdapter,
        vendor_cfg={},
        route_cfg={
            "base_url": "https://api.openai.com/v1",
            "api_key": "test-key",
        },
    )

    capabilities = adapter.provider_capabilities()
    assert capabilities.supports_embeddings is True
    assert capabilities.embedding_model == "text-embedding-3-small"
    assert capabilities.embedding_dim == 1536


def test_openai_compatible_routes_can_opt_into_embeddings():
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry.config = {}

    _client, adapter = registry._build_client_and_adapter(
        vendor="local_openai",
        route="api",
        adapter_cls=OpenAIAdapter,
        vendor_cfg={},
        route_cfg={
            "base_url": "http://localhost:8000/v1",
            "api_key": "local",
            "embedding_model": "local-embed",
            "embedding_dim": "384",
        },
    )

    capabilities = adapter.provider_capabilities()
    assert capabilities.supports_embeddings is True
    assert capabilities.embedding_model == "local-embed"
    assert capabilities.embedding_dim == 384


def test_in_tree_adapter_capability_matrix():
    expected = {
        OpenAIAdapter(): (
            True,
            True,
            True,
            True,
            "text-embedding-3-small",
            1536,
            StructuredOutputMode.JSON_SCHEMA,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.OPENAI_IMAGE_URL,
        ),
        OpenRouterAdapter(): (
            True,
            True,
            True,
            False,
            None,
            None,
            StructuredOutputMode.JSON_SCHEMA,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.OPENAI_IMAGE_URL,
        ),
        AnthropicAdapter(): (
            True,
            True,
            True,
            False,
            None,
            None,
            StructuredOutputMode.TOOL_FORCED,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.ANTHROPIC_CONTENT_BLOCK,
        ),
        GoogleAdapter(): (
            True,
            True,
            False,
            True,
            "text-embedding-004",
            768,
            StructuredOutputMode.NONE,
            ToolStreamingMode.NONSTREAM_FALLBACK,
            VisionInputMode.GEMINI_INLINE_DATA,
        ),
        VertexAIAdapter(project_id="test-project"): (
            True,
            True,
            True,
            True,
            "text-embedding-004",
            768,
            StructuredOutputMode.PROVIDER_NATIVE,
            ToolStreamingMode.NONSTREAM_FALLBACK,
            VisionInputMode.GEMINI_INLINE_DATA,
        ),
        OllamaAdapter(): (
            True,
            True,
            True,
            True,
            "nomic-embed-text",
            768,
            StructuredOutputMode.SCHEMA_FORMAT,
            ToolStreamingMode.NONSTREAM_FALLBACK,
            VisionInputMode.OLLAMA_IMAGES,
        ),
        CodexAdapter(): (
            True,
            True,
            False,
            False,
            None,
            None,
            StructuredOutputMode.NONE,
            ToolStreamingMode.INLINE_EXECUTOR,
            VisionInputMode.OPENAI_IMAGE_URL,
        ),
        MockAdapter(): (
            False,
            False,
            False,
            False,
            None,
            None,
            StructuredOutputMode.NONE,
            ToolStreamingMode.NONE,
            VisionInputMode.NONE,
        ),
    }

    for adapter, (
        supports_tools,
        supports_vision,
        supports_structured_output,
        supports_embeddings,
        embedding_model,
        embedding_dim,
        structured_output_mode,
        tool_streaming_mode,
        vision_input_mode,
    ) in expected.items():
        capabilities = adapter.provider_capabilities()
        assert capabilities.supports_tools is supports_tools
        assert capabilities.supports_streaming is True
        assert capabilities.supports_vision is supports_vision
        assert capabilities.supports_structured_output is supports_structured_output
        assert capabilities.supports_embeddings is supports_embeddings
        if isinstance(adapter, AnthropicAdapter):
            assert capabilities.supports_inline_system is True
            assert "supports_inline_system" in capabilities.model_dependent
        assert capabilities.embedding_model == embedding_model
        assert capabilities.embedding_dim == embedding_dim
        assert capabilities.structured_output_mode == structured_output_mode
        assert capabilities.tool_streaming_mode == tool_streaming_mode
        assert capabilities.vision_input_mode == vision_input_mode


def test_config_seeded_auto_embedding_route_normalizes_to_none():
    """codex P2 on #2270 — the documented ``embedding_route = "auto"`` config
    value must seed as None (follow-chat), not as a literal explicit route that
    resolve treats as a nonexistent provider and keyword-falls-back."""
    with patch(
        "kestrel_sovereign.llm.service.load_section",
        return_value={"embedding_route": "auto"},
    ), patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_registry_class:
        mock_registry = MagicMock()
        mock_registry.initialize_providers.return_value = []
        mock_registry.providers = []
        mock_registry_class.return_value = mock_registry
        service = LLMService()
    assert service.get_embedding_route() is None
    # And explicit set is case-insensitive about the sentinel too.
    service.set_embedding_route("AUTO", persist=False)
    assert service.get_embedding_route() is None


def test_bare_vendor_embedding_route_prefers_embedding_capable_route():
    """codex P2 on #2270 — ``embedding_route = "openai"`` must resolve to the
    vendor's embedding-capable route even when a non-embedding route for the
    same vendor sorts first in the provider table."""
    compat = _route("openai:compat", "openai", AnthropicAdapter())  # no embeddings
    api = _route("openai:api", "openai", OpenAIAdapter())           # embeddings
    service = _embedding_service([compat, api], embedding_route="openai")

    assert service.resolve_embedding_provider() is api
