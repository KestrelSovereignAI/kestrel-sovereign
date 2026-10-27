"""
Unit tests for adapter list_models() methods.
These are unit tests that don't require external services.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

from kestrel_sovereign.llm.model_metadata import ModelInfo, ModelCategory


class TestOpenAIAdapterListModels:
    """Test OpenAIAdapter.list_models()"""

    @pytest.mark.asyncio
    async def test_list_models_returns_model_info_list(self):
        """Test that list_models returns List[ModelInfo]"""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter()

        # Create mock openai client that returns models
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-5.1"
        mock_model1.created = 1700000000
        mock_model1.owned_by = "openai"

        mock_model2 = MagicMock()
        mock_model2.id = "gpt-5-mini"
        mock_model2.created = 1700000001
        mock_model2.owned_by = "openai"

        mock_model3 = MagicMock()
        mock_model3.id = "text-embedding-3-large"
        mock_model3.created = 1700000002
        mock_model3.owned_by = "openai"

        mock_response = MagicMock()
        mock_response.data = [mock_model1, mock_model2, mock_model3]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.models.list.return_value = mock_response
            mock_openai.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                models = await adapter.list_models()

        assert isinstance(models, list)
        assert all(isinstance(m, ModelInfo) for m in models)
        assert len(models) == 3

    @pytest.mark.asyncio
    async def test_list_models_sets_provider(self):
        """Test that all models have provider='openai'"""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter()

        mock_model = MagicMock()
        mock_model.id = "gpt-5.1"
        mock_model.created = 1700000000
        mock_model.owned_by = "openai"

        mock_response = MagicMock()
        mock_response.data = [mock_model]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.models.list.return_value = mock_response
            mock_openai.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                models = await adapter.list_models()

        assert all(m.provider == "openai" for m in models)

    @pytest.mark.asyncio
    async def test_list_models_returns_chat_category_by_default(self):
        """Test that all models default to CHAT category (catalog enriches later)"""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter()

        mock_model1 = MagicMock()
        mock_model1.id = "text-embedding-3-large"
        mock_model1.created = 1700000000
        mock_model1.owned_by = "openai"

        mock_model2 = MagicMock()
        mock_model2.id = "gpt-5.1"
        mock_model2.created = 1700000001
        mock_model2.owned_by = "openai"

        mock_response = MagicMock()
        mock_response.data = [mock_model1, mock_model2]

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.models.list.return_value = mock_response
            mock_openai.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                models = await adapter.list_models()

        # All models default to CHAT, catalog service enriches with actual category
        assert all(m.category == ModelCategory.CHAT for m in models)

    @pytest.mark.asyncio
    async def test_list_models_no_api_key_returns_empty(self):
        """Test that missing API key returns empty list"""
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter()

        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY if present
            os.environ.pop("OPENAI_API_KEY", None)
            models = await adapter.list_models()

        assert models == []


class TestAnthropicAdapterListModels:
    """Test AnthropicAdapter.list_models()"""

    @pytest.mark.asyncio
    async def test_list_models_returns_model_info_list(self):
        """Test that list_models returns List[ModelInfo]"""
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "claude-sonnet-4-5-20250929", "display_name": "Claude Sonnet 4.5", "type": "model"},
                {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5", "type": "model"},
            ]
        }

        with patch("kestrel_sovereign.llm.anthropic_adapter.httpx2.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_instance

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                models = await adapter.list_models()

        assert isinstance(models, list)
        assert all(isinstance(m, ModelInfo) for m in models)

    @pytest.mark.asyncio
    async def test_list_models_sets_provider(self):
        """Test that all models have provider='anthropic'"""
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "claude-sonnet-4-5-20250929", "display_name": "Claude", "type": "model"},
            ]
        }

        with patch("kestrel_sovereign.llm.anthropic_adapter.httpx2.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_instance

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                models = await adapter.list_models()

        assert all(m.provider == "anthropic" for m in models)

    @pytest.mark.asyncio
    async def test_list_models_no_api_key_returns_empty(self):
        """Test that missing API key returns empty list"""
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter()

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            models = await adapter.list_models()

        assert models == []


class TestOllamaAdapterListModels:
    """Test OllamaAdapter.list_models()"""

    @pytest.mark.asyncio
    async def test_list_models_returns_model_info_list(self):
        """Test that list_models returns List[ModelInfo]"""
        from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter()

        # Mock the ollama client
        mock_models = {
            "models": [
                {"name": "llama3.2:3b", "size": 3_000_000_000},
                {"name": "phi4:latest", "size": 2_000_000_000},
            ]
        }

        with patch("ollama.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list.return_value = mock_models
            mock_client_class.return_value = mock_client

            models = await adapter.list_models()

        assert isinstance(models, list)
        assert all(isinstance(m, ModelInfo) for m in models)

    @pytest.mark.asyncio
    async def test_list_models_calculates_size(self):
        """Test that size_gb is calculated correctly"""
        from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter()

        mock_models = {
            "models": [
                {"name": "llama3.2:3b", "size": 3_221_225_472},  # 3GB
            ]
        }

        with patch("ollama.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list.return_value = mock_models
            mock_client_class.return_value = mock_client

            models = await adapter.list_models()

        assert len(models) == 1
        assert models[0].size_gb == pytest.approx(3.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_list_models_detects_embeddings(self):
        """Test that embedding models are detected"""
        from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter()

        mock_models = {
            "models": [
                {"name": "nomic-embed-text:latest", "size": 500_000_000},
                {"name": "mxbai-embed-large:latest", "size": 700_000_000},
            ]
        }

        with patch("ollama.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list.return_value = mock_models
            mock_client_class.return_value = mock_client

            models = await adapter.list_models()

        assert all(m.category == ModelCategory.EMBEDDING for m in models)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,capabilities,expected", [
        ("gemma4:31b", ["completion", "vision"], True),
        ("llava:latest", None, False),
        ("llama3.2:latest", ["completion"], False),
    ])
    async def test_list_models_detects_vision(self, name, capabilities, expected):
        """Use concrete daemon capabilities; model names confer no vision."""
        from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter()

        mock_models = {
            "models": [
                {"name": name, "size": 4_000_000_000},
            ]
        }

        with patch("ollama.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list.return_value = mock_models
            mock_client.show.return_value = {"capabilities": capabilities}
            mock_client_class.return_value = mock_client

            models = await adapter.list_models()

        assert len(models) == 1
        assert models[0].supports_vision is expected
        mock_client.show.assert_awaited_once_with(name)


class TestGoogleAdapterListModels:
    """Test GoogleAdapter.list_models()"""

    @pytest.mark.asyncio
    async def test_list_models_no_api_key_returns_empty(self):
        """Test that missing API key returns empty list"""
        from kestrel_sovereign.llm.google_adapter import GoogleAdapter
        adapter = GoogleAdapter()

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            os.environ.pop("GEMINI_API_KEY", None)
            models = await adapter.list_models()

        assert models == []


class TestVertexAIAdapterListModels:
    """Test VertexAIAdapter.list_models() when discovery fails (#3610).

    The fallback holds no hardcoded list (#215): it is the ``vertex_ai`` part
    of the discovery cache the last successful discovery wrote. The default
    cache sits beside the package, so what it holds depends on what last ran
    in the checkout; these tests give the catalog a cache in ``tmp_path``.
    """

    VERTEX_MODELS = [
        ModelInfo(
            id="models/gemini-2.5-flash",
            provider="vertex_ai",
            display_name="Gemini 2.5 Flash",
            category=ModelCategory.CHAT,
            is_featured=True,
            context_limit=1048576,
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
        ),
        ModelInfo(
            id="models/gemini-2.5-pro",
            provider="vertex_ai",
            display_name="Gemini 2.5 Pro",
            category=ModelCategory.CHAT,
            description="Stable release of Gemini 2.5 Pro",
            context_limit=1048576,
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
        ),
        ModelInfo(
            id="models/text-embedding-005",
            provider="vertex_ai",
            display_name="Text Embedding 005",
            category=ModelCategory.EMBEDDING,
            context_limit=2048,
            supports_tools=True,
            supports_streaming=True,
        ),
    ]
    OPENROUTER_MODELS = [
        ModelInfo(id="openai/gpt-5-mini", provider="openrouter", display_name="GPT-5 Mini"),
        ModelInfo(id="google/gemini-2.5-flash", provider="openrouter", display_name="Gemini 2.5 Flash"),
    ]

    @pytest.fixture
    def catalog(self, tmp_path, monkeypatch):
        from kestrel_sovereign.llm import model_catalog
        from kestrel_sovereign.llm.model_catalog import ModelCatalogService

        service = ModelCatalogService(cache_path=tmp_path / "model_discovery_cache.json")
        monkeypatch.setattr(model_catalog, "_catalog_service", service)
        return service

    @pytest.fixture
    def adapter(self):
        from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

        adapter = VertexAIAdapter(project_id="fallback-test-project")
        adapter._client = MagicMock()
        # Discovery lists on the async surface (#3646).
        adapter._client.aio.models.list = AsyncMock(
            side_effect=RuntimeError("Vertex discovery unavailable")
        )
        return adapter

    @pytest.mark.asyncio
    async def test_failed_discovery_returns_the_cached_vertex_models(self, catalog, adapter):
        """Discovery down: the Vertex models of the last discovery, in cache order."""
        catalog.write_cache([
            self.OPENROUTER_MODELS[0],
            *self.VERTEX_MODELS[:2],
            self.OPENROUTER_MODELS[1],
            self.VERTEX_MODELS[2],
        ])

        models = await adapter.list_models()

        adapter._client.aio.models.list.assert_awaited_once_with()
        assert len(models) >= 3
        assert all(isinstance(m, ModelInfo) for m in models)
        assert models == self.VERTEX_MODELS

    @pytest.mark.asyncio
    async def test_fallback_without_a_cache_is_empty(self, catalog, adapter):
        """No discovery has run: there is nothing to fall back to."""
        assert not catalog.cache_path.exists()

        assert await adapter.list_models() == []

    @pytest.mark.asyncio
    async def test_fallback_never_takes_another_providers_models(self, catalog, adapter):
        """A cache with no Vertex entries, as a unit test's discovery leaves, gives nothing."""
        catalog.write_cache(self.OPENROUTER_MODELS)

        assert await adapter.list_models() == []


class TestModelInfoSerialization:
    """Regression: ModelInfo must be JSON-serializable for tool results.

    The subagent tool loop calls json.dumps(result) on tool return values.
    list_models returns List[ModelInfo] which failed with
    'ModelInfo is not JSON serializable' before _serialize_tool_result was added.
    """

    def test_serialize_model_info_list(self):
        """Test that a list of ModelInfo can be serialized via _serialize_tool_result."""
        import json
        from kestrel_sovereign.features.base import _serialize_tool_result

        models = [
            ModelInfo(id="gpt-5-mini", provider="openai", display_name="GPT-5 Mini"),
            ModelInfo(
                id="claude-sonnet-4-5-20250929", provider="anthropic",
                display_name="Claude Sonnet 4.5", category=ModelCategory.EMBEDDING,
                is_featured=True, supports_vision=True, context_limit=200000,
            ),
        ]

        serialized = _serialize_tool_result(models)
        # Must not raise
        result_json = json.dumps(serialized)
        parsed = json.loads(result_json)

        assert len(parsed) == 2
        assert parsed[0]["id"] == "gpt-5-mini"
        assert parsed[0]["category"] == "chat"
        assert parsed[1]["is_featured"] is True
        assert parsed[1]["category"] == "embedding"
        assert parsed[1]["context_limit"] == 200000

    def test_serialize_nested_dict_with_model_info(self):
        """Test serialization of dict containing ModelInfo values."""
        import json
        from kestrel_sovereign.features.base import _serialize_tool_result

        result = {
            "success": True,
            "model": ModelInfo(id="gpt-5", provider="openai", display_name="GPT-5"),
        }

        serialized = _serialize_tool_result(result)
        result_json = json.dumps(serialized)
        parsed = json.loads(result_json)

        assert parsed["success"] is True
        assert parsed["model"]["id"] == "gpt-5"

    def test_serialize_plain_dict_passthrough(self):
        """Test that plain dicts pass through unchanged."""
        from kestrel_sovereign.features.base import _serialize_tool_result

        result = {"success": True, "message": "done", "count": 42}
        assert _serialize_tool_result(result) == result
