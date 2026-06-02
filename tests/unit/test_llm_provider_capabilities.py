from types import SimpleNamespace
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
    service.resolve_provider_routing = lambda: ([openai_route], None)

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
    service.resolve_provider_routing = lambda: ([anthropic_route], None)

    assert service.resolve_embedding_provider() is None
    assert service.get_embedding_service() is None


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
            False,
            False,
            False,
            None,
            None,
            StructuredOutputMode.NONE,
            ToolStreamingMode.INLINE_EXECUTOR,
            VisionInputMode.NONE,
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
