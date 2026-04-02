"""
Unit Tests for Feature Lifecycle Hooks.

Tests the new lifecycle methods on the Feature base class:
- on_enable / on_disable / on_remove
- get_hooks() auto-registration / auto-unregistration
- get_router()
- post_all_features_loaded()
- config_schema / get_config / set_config
"""

import asyncio
import pytest
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.hooks import Hook, HookEvent, HookInput, HookOutput, HooksManager
from kestrel_sovereign.tools.base import ToolCategory


# === Test Hook Implementation ===

class SampleHook(Hook):
    """A simple hook for testing lifecycle auto-registration."""

    def __init__(self, name: str = "test_hook"):
        super().__init__(name=name, events=[HookEvent.PRE_TOOL_USE], priority=50)
        self.call_count = 0

    async def execute(self, input: HookInput) -> HookOutput:
        self.call_count += 1
        return HookOutput.allow("test")


class AnotherSampleHook(Hook):
    """A second hook for testing multiple hook registration."""

    def __init__(self):
        super().__init__(name="another_test_hook", events=[HookEvent.POST_TOOL_USE], priority=100)

    async def execute(self, input: HookInput) -> HookOutput:
        return HookOutput.allow()


# === Test Feature Implementations ===

class SimpleFeature(Feature):
    """Minimal feature for testing base class defaults."""

    @property
    def tool_description(self) -> str:
        return "A simple test feature"

    async def initialize(self):
        pass


class HookProvidingFeature(Feature):
    """Feature that provides hooks via get_hooks()."""

    def __init__(self, agent):
        super().__init__(agent)
        self.hook1 = SampleHook("hook_from_feature")
        self.hook2 = AnotherSampleHook()
        self.on_enable_called = False
        self.on_disable_called = False
        self.on_remove_called = False
        self.post_loaded_agent = None

    @property
    def tool_description(self) -> str:
        return "Feature with hooks"

    async def initialize(self):
        pass

    def get_hooks(self) -> List[Hook]:
        return [self.hook1, self.hook2]

    async def on_enable(self):
        self.on_enable_called = True

    async def on_disable(self):
        self.on_disable_called = True

    async def on_remove(self):
        self.on_remove_called = True

    async def post_all_features_loaded(self, agent):
        self.post_loaded_agent = agent


class RouterFeature(Feature):
    """Feature that returns a router."""

    @property
    def tool_description(self) -> str:
        return "Feature with router"

    async def initialize(self):
        pass

    def get_router(self):
        # Return a mock router object
        return MagicMock(name="test_router")


class ConfigFeature(Feature):
    """Feature with config schema."""

    def __init__(self, agent):
        super().__init__(agent)
        self._config = {"threshold": 5, "enabled": True}

    @property
    def tool_description(self) -> str:
        return "Feature with config"

    async def initialize(self):
        pass

    @property
    def config_schema(self) -> Optional[Dict]:
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "integer", "minimum": 0},
                "enabled": {"type": "boolean"},
            },
        }

    async def get_config(self) -> Dict:
        return self._config.copy()

    async def set_config(self, config: Dict) -> None:
        self._config.update(config)


# === Fixtures ===

@pytest.fixture
def mock_agent():
    """Create a mock agent with hooks_manager."""
    agent = MagicMock()
    agent.hooks_manager = HooksManager()
    agent.features = {}
    return agent


# === Tests: Base Class Defaults ===

class TestFeatureBaseDefaults:
    """Test that all new lifecycle methods have sensible defaults."""

    @pytest.mark.asyncio
    async def test_on_enable_default_is_noop(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        await feature.on_enable()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_disable_default_is_noop(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        await feature.on_disable()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_remove_default_is_noop(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        await feature.on_remove()  # Should not raise

    def test_get_hooks_default_returns_empty(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        assert feature.get_hooks() == []

    def test_get_router_default_returns_none(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        assert feature.get_router() is None

    @pytest.mark.asyncio
    async def test_post_all_features_loaded_default_is_noop(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        await feature.post_all_features_loaded(mock_agent)  # Should not raise

    def test_config_schema_default_returns_none(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        assert feature.config_schema is None

    @pytest.mark.asyncio
    async def test_get_config_default_returns_empty(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        assert await feature.get_config() == {}

    @pytest.mark.asyncio
    async def test_set_config_default_is_noop(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        await feature.set_config({"key": "value"})  # Should not raise


# === Tests: Hook Auto-Registration ===

class TestHookAutoRegistration:
    """Test that get_hooks() hooks are auto-registered/unregistered."""

    def test_get_hooks_returns_hook_instances(self, mock_agent):
        feature = HookProvidingFeature(mock_agent)
        hooks = feature.get_hooks()
        assert len(hooks) == 2
        assert hooks[0].name == "hook_from_feature"
        assert hooks[1].name == "another_test_hook"

    @pytest.mark.asyncio
    async def test_hooks_registered_with_manager(self, mock_agent):
        """Simulate what _register_feature does: register get_hooks() with HooksManager."""
        feature = HookProvidingFeature(mock_agent)
        manager = mock_agent.hooks_manager

        # Simulate _register_feature behavior
        await feature.initialize()
        for hook in feature.get_hooks():
            manager.register(hook)
        await feature.on_enable()

        # Verify hooks are registered
        pre_hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        post_hooks = manager.get_hooks(HookEvent.POST_TOOL_USE)
        assert any(h.name == "hook_from_feature" for h in pre_hooks)
        assert any(h.name == "another_test_hook" for h in post_hooks)

    @pytest.mark.asyncio
    async def test_hooks_unregistered_on_disable(self, mock_agent):
        """Simulate what _disable_feature does: unregister get_hooks()."""
        feature = HookProvidingFeature(mock_agent)
        manager = mock_agent.hooks_manager

        # Register hooks
        for hook in feature.get_hooks():
            manager.register(hook)

        # Verify registered
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 1

        # Unregister hooks (simulate _disable_feature)
        await feature.on_disable()
        for hook in feature.get_hooks():
            manager.unregister(hook)

        # Verify unregistered - no stale hooks
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0
        assert len(manager.get_hooks(HookEvent.POST_TOOL_USE)) == 0

    @pytest.mark.asyncio
    async def test_no_stale_hooks_after_disable(self, mock_agent):
        """Core guarantee: disabling a feature leaves no stale hooks."""
        feature = HookProvidingFeature(mock_agent)
        manager = mock_agent.hooks_manager

        # Register
        for hook in feature.get_hooks():
            manager.register(hook)

        # Execute hook to prove it's active
        hook_input = HookInput(
            session_id="test",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="some_tool",
        )
        result = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert feature.hook1.call_count == 1

        # Disable and unregister
        for hook in feature.get_hooks():
            manager.unregister(hook)

        # Execute again — hook should not fire
        result = await manager.execute_hooks(HookEvent.PRE_TOOL_USE, hook_input)
        assert feature.hook1.call_count == 1  # Still 1, not incremented


# === Tests: Lifecycle Callbacks ===

class TestLifecycleCallbacks:
    """Test that lifecycle methods are called correctly."""

    @pytest.mark.asyncio
    async def test_on_enable_called(self, mock_agent):
        feature = HookProvidingFeature(mock_agent)
        assert not feature.on_enable_called
        await feature.on_enable()
        assert feature.on_enable_called

    @pytest.mark.asyncio
    async def test_on_disable_called(self, mock_agent):
        feature = HookProvidingFeature(mock_agent)
        assert not feature.on_disable_called
        await feature.on_disable()
        assert feature.on_disable_called

    @pytest.mark.asyncio
    async def test_on_remove_called(self, mock_agent):
        feature = HookProvidingFeature(mock_agent)
        assert not feature.on_remove_called
        await feature.on_remove()
        assert feature.on_remove_called

    @pytest.mark.asyncio
    async def test_post_all_features_loaded(self, mock_agent):
        feature = HookProvidingFeature(mock_agent)
        assert feature.post_loaded_agent is None
        await feature.post_all_features_loaded(mock_agent)
        assert feature.post_loaded_agent is mock_agent


# === Tests: get_router ===

class TestGetRouter:
    """Test the get_router lifecycle method."""

    def test_router_feature_returns_router(self, mock_agent):
        feature = RouterFeature(mock_agent)
        router = feature.get_router()
        assert router is not None

    def test_simple_feature_returns_none(self, mock_agent):
        feature = SimpleFeature(mock_agent)
        assert feature.get_router() is None


# === Tests: Config Schema ===

class TestConfigSchema:
    """Test config_schema / get_config / set_config."""

    def test_config_schema_returns_schema(self, mock_agent):
        feature = ConfigFeature(mock_agent)
        schema = feature.config_schema
        assert schema is not None
        assert schema["type"] == "object"
        assert "threshold" in schema["properties"]

    @pytest.mark.asyncio
    async def test_get_config_returns_values(self, mock_agent):
        feature = ConfigFeature(mock_agent)
        config = await feature.get_config()
        assert config["threshold"] == 5
        assert config["enabled"] is True

    @pytest.mark.asyncio
    async def test_set_config_updates_values(self, mock_agent):
        feature = ConfigFeature(mock_agent)
        await feature.set_config({"threshold": 10})
        config = await feature.get_config()
        assert config["threshold"] == 10
        assert config["enabled"] is True  # Unchanged


# === Tests: Migrated Features ===

class TestSecurityFeatureGetHooks:
    """Test that SecurityFeature returns its hook via get_hooks()."""

    def test_security_feature_get_hooks(self, mock_agent):
        """SecurityFeature.get_hooks() returns the SecurityHook when initialized."""
        from kestrel_sovereign.features.security.feature import SecurityFeature

        feature = SecurityFeature(mock_agent)
        # Before initialize, security_hook is None
        assert feature.get_hooks() == []

    @pytest.mark.asyncio
    async def test_security_feature_get_hooks_after_init(self, mock_agent):
        """After initialize, SecurityFeature.get_hooks() returns the SecurityHook."""
        from kestrel_sovereign.features.security.feature import SecurityFeature

        mock_agent.storage_path = ":memory:"
        feature = SecurityFeature(mock_agent)

        # Patch out the database initialization
        with patch.object(feature, '_async_init', new_callable=AsyncMock):
            await feature.initialize()

        hooks = feature.get_hooks()
        assert len(hooks) == 1
        assert hooks[0].name == "security_guard"


class TestObservabilityFeatureGetHooks:
    """Test that ObservabilityFeature returns its hook via get_hooks()."""

    @pytest.mark.asyncio
    async def test_observability_get_hooks_after_init(self, mock_agent):
        from kestrel_sovereign.features.observability.feature import ObservabilityFeature

        feature = ObservabilityFeature(mock_agent)
        # Before init
        assert feature.get_hooks() == []

        # After init
        await feature.initialize()
        hooks = feature.get_hooks()
        assert len(hooks) == 1
        assert hooks[0].name == "observability"
