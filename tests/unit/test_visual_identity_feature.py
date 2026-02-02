"""
Unit tests for VisualIdentityFeature basic functionality.

Complements the LoRA integration tests with:
1. Tool decorator verification
2. Scene prompts validation
3. Error handling edge cases
4. generate_avatar tool
5. train_lora explicit trigger

For LoRA training and RunPod tests, see: tests/integration/test_visual_identity_lora.py
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.agent_id = "test-agent"
    agent.storage = MagicMock()
    agent.storage.files = MagicMock()
    agent.storage.files.store_avatar = AsyncMock(return_value="hash-abc123")
    return agent


@pytest_asyncio.fixture
async def feature_standalone():
    """Create a VisualIdentityFeature in standalone mode (no agent)."""
    from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

    feature = VisualIdentityFeature(agent=None)
    await feature.initialize()
    return feature


@pytest_asyncio.fixture
async def feature_with_agent(mock_agent):
    """Create a VisualIdentityFeature with a mock agent."""
    from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

    feature = VisualIdentityFeature(agent=mock_agent)
    await feature.initialize()
    return feature


# =============================================================================
# Initialization Tests
# =============================================================================

class TestVisualIdentityFeatureInit:
    """Tests for VisualIdentityFeature initialization."""

    @pytest.mark.asyncio
    async def test_standalone_mode_initialization(self):
        """Test initialization without an agent (standalone mode)."""
        from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

        feature = VisualIdentityFeature(agent=None)
        await feature.initialize()

        assert feature.agent is None
        assert feature.name == "VisualIdentityFeature"

    @pytest.mark.asyncio
    async def test_with_agent_initialization(self, mock_agent):
        """Test initialization with an agent."""
        from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

        feature = VisualIdentityFeature(agent=mock_agent)
        await feature.initialize()

        assert feature.agent is mock_agent

    def test_tool_description(self):
        """Test the feature has a description."""
        from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

        feature = VisualIdentityFeature(agent=None)
        desc = feature.tool_description

        assert "visual" in desc.lower()
        assert "avatar" in desc.lower() or "selfie" in desc.lower()


# =============================================================================
# Scene Prompts Tests
# =============================================================================

class TestScenePrompts:
    """Tests for scene prompt configuration."""

    def test_all_core_scenes_defined(self):
        """Verify all core scene types have prompts."""
        from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

        expected_scenes = [
            "portrait", "casual", "glamour", "flirty", "cozy",
            "adventure", "mysterious", "romantic", "playful",
            "dreamy", "confident"
        ]

        for scene in expected_scenes:
            assert scene in VisualIdentityFeature.SCENE_PROMPTS, f"Missing scene: {scene}"

    def test_scene_prompts_non_empty(self):
        """Verify all scene prompts have content."""
        from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

        for scene, prompt in VisualIdentityFeature.SCENE_PROMPTS.items():
            assert len(prompt) > 10, f"Scene '{scene}' prompt too short"

    def test_scene_prompts_are_descriptive(self):
        """Verify scene prompts contain descriptive keywords."""
        from kestrel_sovereign.features.visual_identity.feature import VisualIdentityFeature

        # Each scene should have relevant keywords
        expected_keywords = {
            "romantic": ["romantic", "lighting", "warm"],
            "casual": ["casual", "selfie", "relaxed"],
            "portrait": ["professional", "studio", "headshot"],
            "mysterious": ["dramatic", "enigmatic"],
        }

        for scene, keywords in expected_keywords.items():
            prompt = VisualIdentityFeature.SCENE_PROMPTS[scene].lower()
            has_keyword = any(kw in prompt for kw in keywords)
            assert has_keyword, f"Scene '{scene}' missing expected keywords"


# =============================================================================
# Tool Decorator Tests
# =============================================================================

class TestToolDecorators:
    """Tests for tool decorator presence and configuration."""

    @pytest.mark.asyncio
    async def test_generate_selfie_has_tool_decorator(self, feature_standalone):
        """Verify generate_selfie has the @tool decorator."""
        assert hasattr(feature_standalone.generate_selfie, "_tool_schema")
        schema = feature_standalone.generate_selfie._tool_schema
        assert schema["name"] == "generate_selfie"
        assert "selfie" in schema["description"].lower()

    @pytest.mark.asyncio
    async def test_generate_avatar_has_tool_decorator(self, feature_standalone):
        """Verify generate_avatar has the @tool decorator."""
        assert hasattr(feature_standalone.generate_avatar, "_tool_schema")
        schema = feature_standalone.generate_avatar._tool_schema
        assert schema["name"] == "generate_avatar"
        assert "avatar" in schema["description"].lower()

    @pytest.mark.asyncio
    async def test_train_lora_has_tool_decorator(self, feature_standalone):
        """Verify train_lora has the @tool decorator."""
        assert hasattr(feature_standalone.train_lora, "_tool_schema")
        schema = feature_standalone.train_lora._tool_schema
        assert schema["name"] == "train_lora"
        assert "lora" in schema["description"].lower()

    @pytest.mark.asyncio
    async def test_command_prefixes_defined(self, feature_standalone):
        """Verify all tools have command prefixes."""
        tools = [
            feature_standalone.generate_selfie,
            feature_standalone.generate_avatar,
            feature_standalone.train_lora,
        ]

        for tool_method in tools:
            schema = tool_method._tool_schema
            # Command prefix should be in metadata or schema
            assert "command_prefix" in schema or "!" in str(schema)


# =============================================================================
# Generate Selfie Error Handling
# =============================================================================

class TestGenerateSelfieErrors:
    """Tests for generate_selfie error handling."""

    @pytest.mark.asyncio
    async def test_selfie_when_disabled(self, feature_standalone):
        """Test selfie generation when service is disabled."""
        feature_standalone.enabled = False
        feature_standalone.service = None

        result = await feature_standalone.generate_selfie(scene="casual")

        assert result["success"] is False
        assert "not available" in result["error"].lower() or "replicate" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_selfie_without_lora_fails(self, feature_standalone):
        """Test that selfie generation requires LoRA (no fallback to censored Replicate)."""
        # Force enable the feature for this test
        feature_standalone.enabled = True

        # Mock the service without LoRA/RunPod
        feature_standalone.service = MagicMock()
        feature_standalone.service.generate_character_portrait = MagicMock(
            return_value=["https://example.com/image.jpg"]
        )
        feature_standalone.service.has_runpod = MagicMock(return_value=False)

        result = await feature_standalone.generate_selfie(scene="casual")

        # Should fail - LoRA is now required for uncensored selfie generation
        assert result["success"] is False
        assert "lora" in result["error"].lower() or result.get("needs_training") is True

    @pytest.mark.asyncio
    async def test_selfie_service_returns_empty(self, feature_standalone):
        """Test handling when service returns no images - but LoRA is required first."""
        # Force enable the feature for this test
        feature_standalone.enabled = True

        feature_standalone.service = MagicMock()
        feature_standalone.service.generate_character_portrait = MagicMock(return_value=[])
        feature_standalone.service.has_runpod = MagicMock(return_value=False)

        result = await feature_standalone.generate_selfie(scene="casual")

        # With LoRA-required architecture, this will fail with LoRA error, not "no results"
        assert result["success"] is False
        # Error will be about LoRA requirement since has_runpod=False
        assert "lora" in result["error"].lower() or result.get("needs_training") is True


# =============================================================================
# Generate Avatar Tests
# =============================================================================

class TestGenerateAvatar:
    """Tests for generate_avatar tool."""

    @pytest.mark.asyncio
    async def test_avatar_when_disabled(self, feature_standalone):
        """Test avatar generation when service is disabled."""
        feature_standalone.enabled = False
        feature_standalone.service = None

        result = await feature_standalone.generate_avatar(
            description="A friendly looking person"
        )

        assert result["success"] is False
        assert "not available" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_avatar_caps_num_outputs(self, feature_standalone):
        """Test that num_outputs is capped at 4."""
        if not feature_standalone.enabled:
            pytest.skip("Feature disabled - REPLICATE_API_TOKEN not set")

        feature_standalone.service = MagicMock()
        feature_standalone.service.generate_character_portrait = MagicMock(
            return_value=["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
        )

        await feature_standalone.generate_avatar(
            description="Test",
            num_outputs=10  # Should be capped
        )

        # Verify it was called with capped value
        call_args = feature_standalone.service.generate_character_portrait.call_args
        assert call_args.kwargs.get("num_outputs", call_args[1].get("num_outputs", 4)) <= 4

    @pytest.mark.asyncio
    async def test_avatar_returns_multiple_urls(self, feature_standalone):
        """Test avatar returns multiple image URLs."""
        if not feature_standalone.enabled:
            pytest.skip("Feature disabled - REPLICATE_API_TOKEN not set")

        expected_urls = [
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg"
        ]

        feature_standalone.service = MagicMock()
        feature_standalone.service.generate_character_portrait = MagicMock(
            return_value=expected_urls
        )

        result = await feature_standalone.generate_avatar(
            description="Friendly person",
            num_outputs=2
        )

        assert result["success"] is True
        assert result["image_urls"] == expected_urls

    @pytest.mark.asyncio
    async def test_avatar_service_returns_empty(self, feature_standalone):
        """Test handling when avatar service returns no images."""
        if not feature_standalone.enabled:
            pytest.skip("Feature disabled - REPLICATE_API_TOKEN not set")

        feature_standalone.service = MagicMock()
        feature_standalone.service.generate_character_portrait = MagicMock(return_value=[])

        result = await feature_standalone.generate_avatar(description="Test person")

        assert result["success"] is False
        assert "no results" in result["error"].lower()


# =============================================================================
# Train LoRA Tests
# =============================================================================

class TestTrainLoRA:
    """Tests for train_lora tool."""

    @pytest.mark.asyncio
    async def test_train_lora_no_runpod_key(self, feature_standalone):
        """Test train_lora when no training provider is available."""
        # Ensure LoRA services are not initialized
        feature_standalone._lora_initialized = False
        feature_standalone._training_provider = None

        with patch.dict("os.environ", {}, clear=True):
            result = await feature_standalone.train_lora(companion_id="test-123")

        assert result["success"] is False
        assert result["error"]  # Has some error message

    @pytest.mark.asyncio
    async def test_train_lora_already_trained(self, feature_standalone):
        """Test train_lora when companion already has trained model."""
        # Mock the lookup to return existing path
        feature_standalone._ensure_lora_services = MagicMock(return_value=True)
        feature_standalone._lookup_lora_path = AsyncMock(
            return_value="ipfs://QmExistingLoraModel"
        )

        result = await feature_standalone.train_lora(companion_id="test-123")

        assert result["success"] is True
        assert result["status"] == "already_trained"
        assert "lora_path" in result


# =============================================================================
# LoRA Service Initialization Tests
# =============================================================================

class TestLoRAServiceInit:
    """Tests for lazy LoRA service initialization via TrainingProviderFactory."""

    @pytest.mark.asyncio
    async def test_lora_services_lazy_init(self, feature_standalone):
        """Test that LoRA services are lazily initialized."""
        # Should not be initialized on startup
        assert feature_standalone._lora_initialized is False
        assert feature_standalone._training_provider is None

    @pytest.mark.asyncio
    async def test_ensure_lora_services_without_key(self, feature_standalone):
        """Test _ensure_lora_services behavior without API keys.

        Note: Returns True if any provider (including local_mps) tries to init,
        even if that provider isn't actually available. The factory attempts
        initialization and sets _lora_initialized regardless of success.
        """
        with patch.dict("os.environ", {}, clear=True):
            result = feature_standalone._ensure_lora_services()

        # The method returns True after attempting initialization
        # (even if no providers are actually available)
        assert feature_standalone._lora_initialized is True  # Marked as checked

    @pytest.mark.asyncio
    async def test_set_db_pool(self, feature_standalone):
        """Test set_db_pool sets pool directly on feature."""
        mock_pool = MagicMock()

        feature_standalone.set_db_pool(mock_pool)

        assert feature_standalone.db_pool == mock_pool

    @pytest.mark.asyncio
    async def test_set_runpod_manager(self, feature_standalone):
        """Test set_runpod_manager passes to service."""
        mock_manager = MagicMock()
        mock_service = MagicMock()
        feature_standalone.service = mock_service

        feature_standalone.set_runpod_manager(mock_manager)

        # Verify manager was passed to service
        assert mock_service.runpod_manager == mock_manager
