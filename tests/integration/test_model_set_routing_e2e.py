"""
Integration tests for model-set command routing.

Tests that model selection correctly identifies providers, especially for
meta-providers like OpenRouter where models use format "vendor/model"
but should route through OpenRouter's API.

This test file would have caught the bug where:
  !model-set google/gemini-3-pro-preview
incorrectly set provider="google" instead of provider="openrouter"
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from kestrel_sovereign.llm.model_metadata import ModelInfo
from kestrel_sovereign.llm.model_cache import get_shared_model_cache

# Standard test models used across fixtures
_DEFAULT_TEST_MODELS = [
    # OpenRouter models - note vendor/model format
    ModelInfo(id="google/gemini-3-pro-preview", provider="openrouter", display_name="Gemini 3 Pro"),
    ModelInfo(id="anthropic/claude-3.5-sonnet", provider="openrouter", display_name="Claude 3.5 Sonnet"),
    ModelInfo(id="meta-llama/llama-3.1-70b-instruct", provider="openrouter", display_name="Llama 3.1 70B"),
    ModelInfo(id="deepseek/deepseek-chat", provider="openrouter", display_name="DeepSeek Chat"),
    # Direct provider models
    ModelInfo(id="gpt-5", provider="openai", display_name="GPT-5"),
    ModelInfo(id="claude-opus-4-5-20251101", provider="anthropic", display_name="Claude Opus 4.5"),
    ModelInfo(id="llama3.2:3b", provider="ollama", display_name="Llama 3.2 3B"),
]


@pytest.fixture(autouse=True)
def _populate_shared_cache():
    """Populate shared model cache for tests, clean up after."""
    cache = get_shared_model_cache()
    cache.set(list(_DEFAULT_TEST_MODELS))
    yield
    cache.clear()


class TestModelSetRouting:
    """Test model-set command correctly routes to providers."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLMService with model cache."""
        service = MagicMock()
        service.set_model_preference = MagicMock()
        service.get_current_mandate = MagicMock(return_value={"preference": {}})
        service.providers = [{"name": "openai"}, {"name": "anthropic"}, {"name": "ollama"}, {"name": "openrouter"}]
        service.default_model = "gpt-5"
        return service

    @pytest.fixture
    def model_agent(self, mock_llm_service):
        """Create ModelAgent with mocked dependencies."""
        from kestrel_sovereign.features.model.feature import ModelAgent

        # Create mock storage that returns empty history
        mock_storage = MagicMock()
        mock_storage.get_conversation_history = AsyncMock(return_value=[])

        mock_parent_agent = MagicMock()
        mock_parent_agent.storage = mock_storage

        agent = ModelAgent(agent=mock_parent_agent)
        agent.llm_service = mock_llm_service
        return agent

    @pytest.mark.asyncio
    async def test_openrouter_model_routes_to_openrouter(self, model_agent, mock_llm_service):
        """OpenRouter models should route through openrouter provider, not underlying vendor."""
        # This is the bug that was fixed - google/gemini should NOT go to Google
        result = await model_agent.set_model("google/gemini-3-pro-preview")

        assert result["success"] is True
        mock_llm_service.set_model_preference.assert_called_once()
        call_args = mock_llm_service.set_model_preference.call_args

        # CRITICAL: Provider should be "openrouter", NOT "google"
        model_name, provider = call_args[0]
        assert provider == "openrouter", f"Expected provider='openrouter', got '{provider}'"
        assert model_name == "google/gemini-3-pro-preview", f"Model ID should be kept intact"

    @pytest.mark.asyncio
    async def test_openrouter_anthropic_model_routes_correctly(self, model_agent, mock_llm_service):
        """anthropic/claude-3.5-sonnet from OpenRouter should NOT go to Anthropic directly."""
        result = await model_agent.set_model("anthropic/claude-3.5-sonnet")

        assert result["success"] is True
        call_args = mock_llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        # Should route to openrouter, not anthropic
        assert provider == "openrouter"
        assert model_name == "anthropic/claude-3.5-sonnet"

    @pytest.mark.asyncio
    async def test_openrouter_meta_llama_routes_correctly(self, model_agent, mock_llm_service):
        """meta-llama models (OpenRouter-only vendor) should route to openrouter."""
        result = await model_agent.set_model("meta-llama/llama-3.1-70b-instruct")

        assert result["success"] is True
        call_args = mock_llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        assert provider == "openrouter"
        assert model_name == "meta-llama/llama-3.1-70b-instruct"

    @pytest.mark.asyncio
    async def test_deepseek_routes_to_openrouter(self, model_agent, mock_llm_service):
        """deepseek models should route to openrouter (OpenRouter-only vendor)."""
        result = await model_agent.set_model("deepseek/deepseek-chat")

        assert result["success"] is True
        call_args = mock_llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        assert provider == "openrouter"

    @pytest.mark.asyncio
    async def test_direct_openai_model_routes_to_openai(self, model_agent, mock_llm_service):
        """openai/gpt-5 should route to openai directly."""
        result = await model_agent.set_model("openai/gpt-5")

        assert result["success"] is True
        call_args = mock_llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        # This IS a direct provider format
        assert provider == "openai"
        assert model_name == "gpt-5"

    @pytest.mark.asyncio
    async def test_direct_anthropic_model_routes_to_anthropic(self, model_agent, mock_llm_service):
        """anthropic/claude-opus-4.5 should route to anthropic directly (not in OpenRouter cache)."""
        # Override shared cache to simulate model not discovered via OpenRouter
        cache = get_shared_model_cache()
        cache.set([
            ModelInfo(id="claude-opus-4-5-20251101", provider="anthropic", display_name="Claude Opus 4.5"),
        ])

        result = await model_agent.set_model("anthropic/claude-opus-4-5-20251101")

        assert result["success"] is True
        call_args = mock_llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        # Direct anthropic model
        assert provider == "anthropic"
        assert model_name == "claude-opus-4-5-20251101"

    @pytest.mark.asyncio
    async def test_ollama_model_routes_to_ollama(self, model_agent, mock_llm_service):
        """ollama/llama3.2:3b should route to ollama directly."""
        result = await model_agent.set_model("ollama/llama3.2:3b")

        assert result["success"] is True
        call_args = mock_llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        assert provider == "ollama"
        assert model_name == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_model_without_slash_routes_correctly(self, model_agent, mock_llm_service):
        """Model without provider prefix should work."""
        result = await model_agent.set_model("gpt-5")

        assert result["success"] is True
        call_args = mock_llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        # No provider specified
        assert provider is None
        assert model_name == "gpt-5"


class TestOpenRouterVendorDetection:
    """Test the _is_openrouter_model helper function."""

    @pytest.fixture
    def model_agent(self):
        """Create ModelAgent with mocked LLMService."""
        from kestrel_sovereign.features.model.feature import ModelAgent

        # Populate shared cache with test data for this class
        cache = get_shared_model_cache()
        cache.set([
            ModelInfo(id="google/gemini-3-pro-preview", provider="openrouter", display_name="Gemini"),
            ModelInfo(id="gpt-5", provider="openai", display_name="GPT-5"),
        ])

        agent = ModelAgent(agent=MagicMock())
        agent.llm_service = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_detects_model_in_cache_as_openrouter(self, model_agent):
        """Model in cache with provider=openrouter should be detected."""
        result = await model_agent._is_openrouter_model("google/gemini-3-pro-preview")
        assert result is True

    @pytest.mark.asyncio
    async def test_detects_model_not_openrouter(self, model_agent):
        """Model in cache with provider=openai should NOT be detected as openrouter."""
        result = await model_agent._is_openrouter_model("gpt-5")
        assert result is False

    @pytest.mark.asyncio
    async def test_detects_openrouter_only_vendor_without_cache(self, model_agent):
        """Known OpenRouter-only vendors should be detected even without cache."""
        get_shared_model_cache().set([])  # Empty cache

        # These vendors only exist on OpenRouter
        assert await model_agent._is_openrouter_model("deepseek/deepseek-chat") is True
        assert await model_agent._is_openrouter_model("meta-llama/llama-3.1-70b") is True
        assert await model_agent._is_openrouter_model("mistralai/mistral-large") is True
        assert await model_agent._is_openrouter_model("qwen/qwen-2.5-72b") is True
        assert await model_agent._is_openrouter_model("nous/hermes-3-llama-3.1-405b") is True

    @pytest.mark.asyncio
    async def test_unknown_vendor_not_detected(self, model_agent):
        """Unknown vendor should NOT be detected as OpenRouter."""
        get_shared_model_cache().set([])  # Empty cache

        # These could be direct provider models
        assert await model_agent._is_openrouter_model("openai/gpt-5") is False
        assert await model_agent._is_openrouter_model("anthropic/claude-5") is False
        assert await model_agent._is_openrouter_model("ollama/llama3:8b") is False

    @pytest.mark.asyncio
    async def test_handles_no_llm_service(self, model_agent):
        """Should handle missing llm_service gracefully."""
        model_agent.llm_service = None

        result = await model_agent._is_openrouter_model("google/gemini-3-pro")
        assert result is False

    @pytest.mark.asyncio
    async def test_handles_no_cache(self, model_agent):
        """Should handle missing cache gracefully."""
        get_shared_model_cache().clear()

        # Should fall back to vendor detection
        result = await model_agent._is_openrouter_model("deepseek/deepseek-chat")
        assert result is True  # deepseek is OpenRouter-only


class TestModelSetResponseFormat:
    """Test model-set response format for UI sync."""

    @pytest.fixture
    def model_agent(self):
        """Create ModelAgent with mocked dependencies."""
        from kestrel_sovereign.features.model.feature import ModelAgent

        # Create mock storage that returns empty history
        mock_storage = MagicMock()
        mock_storage.get_conversation_history = AsyncMock(return_value=[])

        mock_parent_agent = MagicMock()
        mock_parent_agent.storage = mock_storage

        # Populate shared cache for this test class
        cache = get_shared_model_cache()
        cache.set([
            ModelInfo(id="google/gemini-3-pro-preview", provider="openrouter", display_name="Gemini"),
        ])

        agent = ModelAgent(agent=mock_parent_agent)
        agent.llm_service = MagicMock()
        agent.llm_service.set_model_preference = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_response_includes_model_changed_marker(self, model_agent):
        """Response should include MODEL_CHANGED marker for UI sync."""
        result = await model_agent.set_model("google/gemini-3-pro-preview")

        assert result["success"] is True
        assert "MODEL_CHANGED:" in result["message"]

    @pytest.mark.asyncio
    async def test_response_model_changed_has_correct_provider(self, model_agent):
        """MODEL_CHANGED marker should have correct provider for OpenRouter models."""
        import json

        result = await model_agent.set_model("google/gemini-3-pro-preview")

        # Extract only the marker payload, not any surrounding message text.
        message = result["message"]
        marker = "MODEL_CHANGED:"
        json_start = message.index(marker) + len(marker)
        json_str = message[json_start:].strip()
        sync_data = json.loads(json_str)

        assert sync_data["provider"] == "openrouter"
        assert sync_data["model"] == "openrouter/google/gemini-3-pro-preview"


class TestExplicitProviderPassing:
    """Test two-arg format: !model-set <provider> <model>

    This is the new format used by the UI when both provider and model
    dropdowns are selected explicitly.
    """

    @pytest.fixture
    def model_agent(self):
        """Create ModelAgent with mocked dependencies."""
        from kestrel_sovereign.features.model.feature import ModelAgent

        # Create mock storage that returns empty history
        mock_storage = MagicMock()
        mock_storage.get_conversation_history = AsyncMock(return_value=[])

        mock_parent_agent = MagicMock()
        mock_parent_agent.storage = mock_storage

        # Populate shared cache for this test class
        cache = get_shared_model_cache()
        cache.set([
            ModelInfo(id="google/gemini-3-pro-preview", provider="openrouter", display_name="Gemini"),
            ModelInfo(id="gpt-5", provider="openai", display_name="GPT-5"),
        ])

        agent = ModelAgent(agent=mock_parent_agent)
        agent.llm_service = MagicMock()
        agent.llm_service.set_model_preference = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_explicit_openrouter_provider_two_args(self, model_agent):
        """Two-arg format: !model-set openrouter google/gemini-3-pro"""
        result = await model_agent.set_model("openrouter", "google/gemini-3-pro-preview")

        assert result["success"] is True
        call_args = model_agent.llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        # Provider should be exactly what was passed, no guessing
        assert provider == "openrouter"
        assert model_name == "google/gemini-3-pro-preview"

    @pytest.mark.asyncio
    async def test_explicit_openai_provider_two_args(self, model_agent):
        """Two-arg format: !model-set openai gpt-5"""
        result = await model_agent.set_model("openai", "gpt-5")

        assert result["success"] is True
        call_args = model_agent.llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        assert provider == "openai"
        assert model_name == "gpt-5"

    @pytest.mark.asyncio
    async def test_explicit_ollama_provider_two_args(self, model_agent):
        """Two-arg format: !model-set ollama llama3.2:3b"""
        result = await model_agent.set_model("ollama", "llama3.2:3b")

        assert result["success"] is True
        call_args = model_agent.llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        assert provider == "ollama"
        assert model_name == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_explicit_anthropic_provider_two_args(self, model_agent):
        """Two-arg format: !model-set anthropic claude-3.5-sonnet"""
        result = await model_agent.set_model("anthropic", "claude-3.5-sonnet")

        assert result["success"] is True
        call_args = model_agent.llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        assert provider == "anthropic"
        assert model_name == "claude-3.5-sonnet"

    @pytest.mark.asyncio
    async def test_single_arg_simple_model(self, model_agent):
        """Single-arg format (direct command): !model-set gpt-5"""
        result = await model_agent.set_model("gpt-5")

        assert result["success"] is True
        call_args = model_agent.llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        # No provider specified - should be None
        assert provider is None
        assert model_name == "gpt-5"

    @pytest.mark.asyncio
    async def test_single_arg_with_slash(self, model_agent):
        """Single-arg format with provider/model: !model-set openai/gpt-5"""
        result = await model_agent.set_model("openai/gpt-5")

        assert result["success"] is True
        call_args = model_agent.llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        # Should parse the slash as provider/model
        assert provider == "openai"
        assert model_name == "gpt-5"

    @pytest.mark.asyncio
    async def test_explicit_provider_overrides_slash_parsing(self, model_agent):
        """Explicit provider should be used even if model contains slash."""
        # This is the key difference: UI sends explicit provider, model kept intact
        result = await model_agent.set_model("openrouter", "anthropic/claude-3.5-sonnet")

        assert result["success"] is True
        call_args = model_agent.llm_service.set_model_preference.call_args
        model_name, provider = call_args[0]

        # Provider from explicit arg, model kept as full ID
        assert provider == "openrouter"
        assert model_name == "anthropic/claude-3.5-sonnet"
