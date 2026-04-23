"""
Integration tests for model-set command routing under the vendor/route/model schema.

Tests that model selection correctly identifies vendors, especially for
meta-vendors like OpenRouter where models use format "vendor/model" but
should route through OpenRouter's API.

This test file would have caught the bug where:
  !model-set google/gemini-3-pro-preview
incorrectly set vendor="google" instead of vendor="openrouter".
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
    # Direct vendor models
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


def _make_ctx_builder_stub():
    """Stub ContextBuilder so set_model's pre-switch safety check passes.

    set_model calls ``estimate_effective_history_tokens`` before flipping
    the mandate to verify the target model's budget can fit the current
    history. Without a stub, the MagicMock return value fails the
    numeric comparison and set_model returns success=False.
    """
    ctx = MagicMock()
    ctx.estimate_effective_history_tokens = MagicMock(return_value={
        "effective_tokens": 0,
        "raw_tokens": 0,
        "history_budget": 100000,
        "context_limit": 200000,
        "messages_kept": 0,
    })
    return ctx


def _make_parent_agent():
    """Parent agent with an empty-history storage and a ctx_builder stub."""
    mock_storage = MagicMock()
    mock_storage.get_conversation_history = AsyncMock(return_value=[])
    parent = MagicMock()
    parent.storage = mock_storage
    parent.context_builder = _make_ctx_builder_stub()
    return parent


class TestModelSetRouting:
    """Test model-set command correctly routes to vendors."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLMService with vendor/route-shaped providers."""
        service = MagicMock()
        service.set_model_preference = MagicMock()
        service.get_current_mandate = MagicMock(return_value={"preference": {}})
        # Composite names under the new schema.
        service.providers = [
            {"name": "openai:api", "vendor": "openai", "route": "api"},
            {"name": "anthropic:api", "vendor": "anthropic", "route": "api"},
            {"name": "ollama:local", "vendor": "ollama", "route": "local"},
            {"name": "openrouter:api", "vendor": "openrouter", "route": "api"},
        ]
        service.default_model = "gpt-5"
        return service

    @pytest.fixture
    def model_agent(self, mock_llm_service):
        """Create ModelAgent with mocked dependencies."""
        from kestrel_sovereign.features.model.feature import ModelAgent

        agent = ModelAgent(agent=_make_parent_agent())
        agent.llm_service = mock_llm_service
        return agent

    @pytest.mark.asyncio
    async def test_openrouter_model_routes_to_openrouter(self, model_agent, mock_llm_service):
        """OpenRouter models should route through openrouter vendor, not underlying vendor."""
        result = await model_agent.set_model("google/gemini-3-pro-preview")

        assert result["success"] is True
        mock_llm_service.set_model_preference.assert_called_once()
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        # CRITICAL: vendor should be "openrouter", NOT "google"
        assert vendor == "openrouter", f"Expected vendor='openrouter', got '{vendor}'"
        assert model_name == "google/gemini-3-pro-preview", "Model ID should be kept intact"
        assert route is None

    @pytest.mark.asyncio
    async def test_openrouter_anthropic_model_routes_correctly(self, model_agent, mock_llm_service):
        """anthropic/claude-3.5-sonnet from OpenRouter should NOT go to Anthropic directly."""
        result = await model_agent.set_model("anthropic/claude-3.5-sonnet")

        assert result["success"] is True
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        # Should route to openrouter, not anthropic
        assert vendor == "openrouter"
        assert model_name == "anthropic/claude-3.5-sonnet"

    @pytest.mark.asyncio
    async def test_openrouter_meta_llama_routes_correctly(self, model_agent, mock_llm_service):
        """meta-llama models (OpenRouter-only vendor) should route to openrouter."""
        result = await model_agent.set_model("meta-llama/llama-3.1-70b-instruct")

        assert result["success"] is True
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        assert vendor == "openrouter"
        assert model_name == "meta-llama/llama-3.1-70b-instruct"

    @pytest.mark.asyncio
    async def test_deepseek_routes_to_openrouter(self, model_agent, mock_llm_service):
        """deepseek models should route to openrouter (OpenRouter-only vendor)."""
        result = await model_agent.set_model("deepseek/deepseek-chat")

        assert result["success"] is True
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        assert vendor == "openrouter"

    @pytest.mark.asyncio
    async def test_direct_openai_model_routes_to_openai(self, model_agent, mock_llm_service):
        """openai/gpt-5 should route to openai directly."""
        result = await model_agent.set_model("openai/gpt-5")

        assert result["success"] is True
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        # This IS a direct vendor format
        assert vendor == "openai"
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
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        # Direct anthropic model
        assert vendor == "anthropic"
        assert model_name == "claude-opus-4-5-20251101"

    @pytest.mark.asyncio
    async def test_ollama_model_routes_to_ollama(self, model_agent, mock_llm_service):
        """ollama/llama3.2:3b should route to ollama directly."""
        result = await model_agent.set_model("ollama/llama3.2:3b")

        assert result["success"] is True
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        assert vendor == "ollama"
        assert model_name == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_vendor_route_composite_routes_to_exact_route(self, model_agent, mock_llm_service):
        """``anthropic:plan/claude-sonnet-4-6`` should route to anthropic with route=plan."""
        result = await model_agent.set_model("anthropic:plan/claude-sonnet-4-6")

        assert result["success"] is True
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        assert vendor == "anthropic"
        assert route == "plan"
        assert model_name == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_model_without_slash_routes_correctly(self, model_agent, mock_llm_service):
        """Model without vendor prefix passes vendor=None — resolution happens in LLMService."""
        result = await model_agent.set_model("gpt-5")

        assert result["success"] is True
        model_name, vendor, route = mock_llm_service.set_model_preference.call_args[0]

        # No vendor specified at the feature level; LLMService will auto-resolve
        # from the catalog (or refuse if the id is unknown/ambiguous).
        assert vendor is None
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

        # These could be direct vendor models
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

        # Populate shared cache for this test class
        cache = get_shared_model_cache()
        cache.set([
            ModelInfo(id="google/gemini-3-pro-preview", provider="openrouter", display_name="Gemini"),
        ])

        agent = ModelAgent(agent=_make_parent_agent())
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
    async def test_response_model_changed_has_correct_vendor(self, model_agent):
        """MODEL_CHANGED marker should carry {vendor, route, model_name, model} for OpenRouter models."""
        import json

        result = await model_agent.set_model("google/gemini-3-pro-preview")

        # Extract only the marker payload, not any surrounding message text.
        message = result["message"]
        marker = "MODEL_CHANGED:"
        json_start = message.index(marker) + len(marker)
        json_str = message[json_start:].strip()
        sync_data = json.loads(json_str)

        assert sync_data["vendor"] == "openrouter"
        assert sync_data["route"] is None
        assert sync_data["model_name"] == "google/gemini-3-pro-preview"
        assert sync_data["model"] == "openrouter/google/gemini-3-pro-preview"


class TestExplicitVendorPassing:
    """Test two-arg format: !model-set <vendor[:route]> <model>

    This is the new format used by the UI when both vendor and model
    dropdowns are selected explicitly.
    """

    @pytest.fixture
    def model_agent(self):
        """Create ModelAgent with mocked dependencies."""
        from kestrel_sovereign.features.model.feature import ModelAgent

        # Populate shared cache for this test class
        cache = get_shared_model_cache()
        cache.set([
            ModelInfo(id="google/gemini-3-pro-preview", provider="openrouter", display_name="Gemini"),
            ModelInfo(id="gpt-5", provider="openai", display_name="GPT-5"),
        ])

        agent = ModelAgent(agent=_make_parent_agent())
        agent.llm_service = MagicMock()
        agent.llm_service.set_model_preference = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_explicit_openrouter_vendor_two_args(self, model_agent):
        """Two-arg format: !model-set openrouter google/gemini-3-pro"""
        result = await model_agent.set_model("openrouter", "google/gemini-3-pro-preview")

        assert result["success"] is True
        model_name, vendor, route = model_agent.llm_service.set_model_preference.call_args[0]

        # Vendor should be exactly what was passed, no guessing
        assert vendor == "openrouter"
        assert route is None
        assert model_name == "google/gemini-3-pro-preview"

    @pytest.mark.asyncio
    async def test_explicit_openai_vendor_two_args(self, model_agent):
        """Two-arg format: !model-set openai gpt-5"""
        result = await model_agent.set_model("openai", "gpt-5")

        assert result["success"] is True
        model_name, vendor, route = model_agent.llm_service.set_model_preference.call_args[0]

        assert vendor == "openai"
        assert model_name == "gpt-5"

    @pytest.mark.asyncio
    async def test_explicit_ollama_vendor_two_args(self, model_agent):
        """Two-arg format: !model-set ollama llama3.2:3b"""
        result = await model_agent.set_model("ollama", "llama3.2:3b")

        assert result["success"] is True
        model_name, vendor, route = model_agent.llm_service.set_model_preference.call_args[0]

        assert vendor == "ollama"
        assert model_name == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_explicit_vendor_route_composite_two_args(self, model_agent):
        """Two-arg format with route: !model-set anthropic:plan claude-sonnet-4-6"""
        result = await model_agent.set_model("anthropic:plan", "claude-sonnet-4-6")

        assert result["success"] is True
        model_name, vendor, route = model_agent.llm_service.set_model_preference.call_args[0]

        assert vendor == "anthropic"
        assert route == "plan"
        assert model_name == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_single_arg_simple_model(self, model_agent):
        """Single-arg format (direct command): !model-set gpt-5"""
        result = await model_agent.set_model("gpt-5")

        assert result["success"] is True
        model_name, vendor, route = model_agent.llm_service.set_model_preference.call_args[0]

        # No vendor specified at the feature level; LLMService resolves.
        assert vendor is None
        assert model_name == "gpt-5"

    @pytest.mark.asyncio
    async def test_single_arg_with_slash(self, model_agent):
        """Single-arg format with vendor/model: !model-set openai/gpt-5"""
        result = await model_agent.set_model("openai/gpt-5")

        assert result["success"] is True
        model_name, vendor, route = model_agent.llm_service.set_model_preference.call_args[0]

        # Should parse the slash as vendor/model
        assert vendor == "openai"
        assert model_name == "gpt-5"

    @pytest.mark.asyncio
    async def test_explicit_vendor_overrides_slash_parsing(self, model_agent):
        """Explicit vendor should be used even if model contains slash."""
        # UI sends explicit vendor, model kept intact
        result = await model_agent.set_model("openrouter", "anthropic/claude-3.5-sonnet")

        assert result["success"] is True
        model_name, vendor, route = model_agent.llm_service.set_model_preference.call_args[0]

        # Vendor from explicit arg, model kept as full ID
        assert vendor == "openrouter"
        assert model_name == "anthropic/claude-3.5-sonnet"
