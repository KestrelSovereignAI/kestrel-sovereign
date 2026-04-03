"""Tests for VisualIdentityFeature (extracted package)."""

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
    from kestrel_feature_visual.feature import VisualIdentityFeature

    feature = VisualIdentityFeature(agent=None)
    await feature.initialize()
    return feature


@pytest_asyncio.fixture
async def feature_with_agent(mock_agent):
    """Create a VisualIdentityFeature with a mock agent."""
    from kestrel_feature_visual.feature import VisualIdentityFeature

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
        from kestrel_feature_visual.feature import VisualIdentityFeature

        feature = VisualIdentityFeature(agent=None)
        await feature.initialize()

        assert feature.agent is None
        assert feature.name == "VisualIdentityFeature"

    @pytest.mark.asyncio
    async def test_with_agent_initialization(self, mock_agent):
        """Test initialization with an agent."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        feature = VisualIdentityFeature(agent=mock_agent)
        await feature.initialize()

        assert feature.agent is mock_agent

    def test_tool_description(self):
        """Test the feature has a description."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

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
        from kestrel_feature_visual.feature import VisualIdentityFeature

        expected_scenes = [
            "portrait", "casual", "glamour", "flirty", "cozy",
            "adventure", "mysterious", "romantic", "playful",
            "dreamy", "confident"
        ]

        for scene in expected_scenes:
            assert scene in VisualIdentityFeature.SCENE_PROMPTS, f"Missing scene: {scene}"

    def test_scene_prompts_non_empty(self):
        """Verify all scene prompts have content."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        for scene, prompt in VisualIdentityFeature.SCENE_PROMPTS.items():
            assert len(prompt) > 10, f"Scene '{scene}' prompt too short"


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

    @pytest.mark.asyncio
    async def test_generate_avatar_has_tool_decorator(self, feature_standalone):
        """Verify generate_avatar has the @tool decorator."""
        assert hasattr(feature_standalone.generate_avatar, "_tool_schema")
        schema = feature_standalone.generate_avatar._tool_schema
        assert schema["name"] == "generate_avatar"

    @pytest.mark.asyncio
    async def test_train_lora_has_tool_decorator(self, feature_standalone):
        """Verify train_lora has the @tool decorator."""
        assert hasattr(feature_standalone.train_lora, "_tool_schema")
        schema = feature_standalone.train_lora._tool_schema
        assert schema["name"] == "train_lora"


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


# =============================================================================
# Generate Selfie Tests
# =============================================================================

class TestGenerateSelfie:
    """Tests for generate_selfie error handling."""

    @pytest.mark.asyncio
    async def test_selfie_when_disabled(self, feature_standalone):
        """Test selfie generation when service is disabled."""
        feature_standalone.enabled = False
        feature_standalone.service = None

        result = await feature_standalone.generate_selfie(scene="casual")

        assert result["success"] is False
        assert "not available" in result["error"].lower() or "replicate" in result["error"].lower()
