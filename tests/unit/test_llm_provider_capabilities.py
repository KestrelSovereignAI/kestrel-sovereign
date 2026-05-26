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
            "structured_output_mode": "json_schema",
            "tool_streaming_mode": "native_delta",
            "vision_input_mode": "openai_image_url",
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
        "structured_output_mode": "json_schema",
        "tool_streaming_mode": "native_delta",
        "vision_input_mode": "openai_image_url",
        "model_dependent": ["vision"],
        "notes": ["example"],
    }


def test_adapter_capabilities_normalizes_plugin_dicts():
    capabilities = ProviderCapabilities.from_mapping(
        DictCapabilitiesAdapter().provider_capabilities()
    )

    assert capabilities.supports_tools is True
    assert capabilities.supports_structured_output is True
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
    assert route["capabilities"]["structured_output_mode"] == "json_schema"


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


def test_in_tree_adapter_capability_matrix():
    expected = {
        OpenAIAdapter(): (
            True,
            True,
            True,
            StructuredOutputMode.JSON_SCHEMA,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.OPENAI_IMAGE_URL,
        ),
        OpenRouterAdapter(): (
            True,
            True,
            True,
            StructuredOutputMode.JSON_SCHEMA,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.OPENAI_IMAGE_URL,
        ),
        AnthropicAdapter(): (
            True,
            True,
            True,
            StructuredOutputMode.TOOL_FORCED,
            ToolStreamingMode.NATIVE_DELTA,
            VisionInputMode.ANTHROPIC_CONTENT_BLOCK,
        ),
        GoogleAdapter(): (
            True,
            True,
            False,
            StructuredOutputMode.NONE,
            ToolStreamingMode.NONSTREAM_FALLBACK,
            VisionInputMode.GEMINI_INLINE_DATA,
        ),
        VertexAIAdapter(project_id="test-project"): (
            True,
            True,
            True,
            StructuredOutputMode.PROVIDER_NATIVE,
            ToolStreamingMode.NONSTREAM_FALLBACK,
            VisionInputMode.GEMINI_INLINE_DATA,
        ),
        OllamaAdapter(): (
            True,
            True,
            True,
            StructuredOutputMode.SCHEMA_FORMAT,
            ToolStreamingMode.NONSTREAM_FALLBACK,
            VisionInputMode.OLLAMA_IMAGES,
        ),
        CodexAdapter(): (
            True,
            False,
            False,
            StructuredOutputMode.NONE,
            ToolStreamingMode.INLINE_EXECUTOR,
            VisionInputMode.NONE,
        ),
        MockAdapter(): (
            False,
            False,
            False,
            StructuredOutputMode.NONE,
            ToolStreamingMode.NONE,
            VisionInputMode.NONE,
        ),
    }

    for adapter, (
        supports_tools,
        supports_vision,
        supports_structured_output,
        structured_output_mode,
        tool_streaming_mode,
        vision_input_mode,
    ) in expected.items():
        capabilities = adapter.provider_capabilities()
        assert capabilities.supports_tools is supports_tools
        assert capabilities.supports_streaming is True
        assert capabilities.supports_vision is supports_vision
        assert capabilities.supports_structured_output is supports_structured_output
        assert capabilities.structured_output_mode == structured_output_mode
        assert capabilities.tool_streaming_mode == tool_streaming_mode
        assert capabilities.vision_input_mode == vision_input_mode
