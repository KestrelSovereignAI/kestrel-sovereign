from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock

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

    assert capabilities.to_dict() == {
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
    client = SimpleNamespace(
        embed_content_async=AsyncMock(
            return_value={"embedding": {"values": [0.1, 0.2]}}
        )
    )

    assert await adapter.aembed(client, "hello", model="text-embedding-004") == [
        0.1,
        0.2,
    ]
    client.embed_content_async.assert_awaited_with(
        model="models/text-embedding-004",
        content="hello",
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
