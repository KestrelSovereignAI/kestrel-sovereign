"""
Tests for the Feature Discovery module.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from kestrel_sovereign.features import (
    discover_features,
    discover_feature_modules,
    discover_entrypoint_feature_classes,
    get_disabled_features,
    get_feature_by_name,
    find_feature_class,
    DISABLED_FEATURES_ENV,
    FEATURE_ENTRY_POINT_GROUP,
)
from kestrel_sovereign.features.base import Feature


class TestGetDisabledFeatures:
    """Tests for get_disabled_features function."""

    def test_no_env_var_returns_empty_set(self):
        """Test that missing env var returns empty set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop(DISABLED_FEATURES_ENV, None)
            result = get_disabled_features()
            assert result == set()

    def test_empty_env_var_returns_empty_set(self):
        """Test that empty env var returns empty set."""
        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: ""}):
            result = get_disabled_features()
            assert result == set()

    def test_single_feature_disabled(self):
        """Test disabling a single feature."""
        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: "RunPodFeature"}):
            result = get_disabled_features()
            assert result == {"RunPodFeature"}

    def test_multiple_features_disabled(self):
        """Test disabling multiple features."""
        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: "RunPodFeature,CreativeFeature,MCPAgent"}):
            result = get_disabled_features()
            assert result == {"RunPodFeature", "CreativeFeature", "MCPAgent"}

    def test_handles_whitespace(self):
        """Test that whitespace is trimmed."""
        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: " RunPodFeature , CreativeFeature "}):
            result = get_disabled_features()
            assert result == {"RunPodFeature", "CreativeFeature"}


class TestDiscoverFeatureModules:
    """Tests for discover_feature_modules function."""

    def test_discovers_modules(self):
        """Test that feature modules are discovered."""
        modules = discover_feature_modules()
        
        # Should find at least some known features
        module_names = [m.split(".")[-1] for m in modules]
        assert len(modules) > 0
        
        # Check for some expected modules (can be in different formats)
        found_mcp = any("mcp" in m.lower() for m in modules)
        found_model = any("model" in m.lower() for m in modules)
        found_sovereignty = any("sovereignty" in m.lower() for m in modules)
        
        assert found_mcp or found_model or found_sovereignty, \
            f"Expected to find known features. Found: {modules}"


class TestDiscoverFeatures:
    """Tests for discover_features function."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()
        return agent

    def test_discovers_features(self, mock_agent):
        """Test that features are discovered and instantiated."""
        features = discover_features(mock_agent)
        
        # Should discover at least one feature
        assert len(features) > 0
        
        # All returned items should be Feature instances
        for feature in features:
            assert isinstance(feature, Feature)

    def test_disabled_features_not_loaded(self, mock_agent):
        """Test that disabled features are not loaded."""
        # Get all features first
        all_features = discover_features(mock_agent)
        all_names = {f.__class__.__name__ for f in all_features}
        
        if not all_names:
            pytest.skip("No features discovered to test")
        
        # Pick one to disable
        feature_to_disable = list(all_names)[0]
        
        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: feature_to_disable}):
            filtered_features = discover_features(mock_agent)
            filtered_names = {f.__class__.__name__ for f in filtered_features}
            
            assert feature_to_disable not in filtered_names
            assert len(filtered_features) == len(all_features) - 1

    def test_features_have_agent_reference(self, mock_agent):
        """Test that features receive the agent reference."""
        features = discover_features(mock_agent)
        
        for feature in features:
            assert feature.agent is mock_agent


class TestGetFeatureByName:
    """Tests for get_feature_by_name function."""

    def test_finds_feature_by_class_name(self):
        """Test finding a feature by its class name."""
        mock_feature = Mock(spec=Feature)
        mock_feature.name = "TestFeature"
        mock_feature.__class__.__name__ = "TestFeature"
        
        features = [mock_feature]
        result = get_feature_by_name(features, "TestFeature")
        
        assert result is mock_feature

    def test_returns_none_for_unknown_feature(self):
        """Test that None is returned for unknown feature names."""
        mock_feature = Mock(spec=Feature)
        mock_feature.name = "TestFeature"
        mock_feature.__class__.__name__ = "TestFeature"
        
        features = [mock_feature]
        result = get_feature_by_name(features, "NonExistentFeature")
        
        assert result is None

    def test_empty_list_returns_none(self):
        """Test that empty feature list returns None."""
        result = get_feature_by_name([], "AnyFeature")
        assert result is None


class TestFindFeatureClass:
    """Tests for find_feature_class function."""

    def test_finds_feature_subclass(self):
        """Test finding a Feature subclass in a module."""
        # Create a mock module with a Feature subclass
        class TestFeatureClass(Feature):
            def initialize(self):
                pass
        
        module = Mock()
        module.__name__ = "test_module"
        TestFeatureClass.__module__ = "test_module"
        
        # Mock getmembers to return our test class
        with patch('inspect.getmembers', return_value=[("TestFeatureClass", TestFeatureClass)]):
            result = find_feature_class(module)
            assert result is TestFeatureClass

    def test_returns_none_for_empty_module(self):
        """Test that empty module returns None."""
        module = Mock()
        module.__name__ = "empty_module"
        
        with patch('inspect.getmembers', return_value=[]):
            result = find_feature_class(module)
            assert result is None


class TestFeatureProfiles:
    """Tests for per-agent feature profile filtering."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()
        return agent

    def test_allowed_features_filters_to_allowlist(self, mock_agent):
        """Test that allowed_features restricts which features load."""
        from kestrel_sovereign.rookery.config import MANDATORY_FEATURES

        # Load all features
        all_features = discover_features(mock_agent)
        all_names = {f.__class__.__name__ for f in all_features}

        # Pick a small subset to allow
        allowed = {"BootstrapFeature", "MemoryFeature", "HeartbeatFeature"}
        filtered = discover_features(mock_agent, allowed_features=allowed)
        filtered_names = {f.__class__.__name__ for f in filtered}

        # Should contain only allowed + mandatory features
        expected = allowed | (MANDATORY_FEATURES & all_names)
        assert filtered_names == expected
        assert len(filtered) < len(all_features)

    def test_mandatory_features_always_load(self, mock_agent):
        """Test that mandatory features load even when not in allowlist."""
        from kestrel_sovereign.rookery.config import MANDATORY_FEATURES

        # Allow only non-mandatory features
        allowed = {"BootstrapFeature"}
        filtered = discover_features(mock_agent, allowed_features=allowed)
        filtered_names = {f.__class__.__name__ for f in filtered}

        # All discoverable mandatory features should be present
        all_features = discover_features(mock_agent)
        all_names = {f.__class__.__name__ for f in all_features}
        expected_mandatory = MANDATORY_FEATURES & all_names

        for mandatory in expected_mandatory:
            assert mandatory in filtered_names, f"Mandatory feature {mandatory} missing"

    def test_none_allowed_features_loads_all(self, mock_agent):
        """Test that None allowed_features loads everything (backward compat)."""
        all_features = discover_features(mock_agent)
        none_features = discover_features(mock_agent, allowed_features=None)

        all_names = {f.__class__.__name__ for f in all_features}
        none_names = {f.__class__.__name__ for f in none_features}

        assert all_names == none_names

    def test_empty_allowed_features_loads_only_mandatory(self, mock_agent):
        """Test that empty set loads only mandatory features."""
        from kestrel_sovereign.rookery.config import MANDATORY_FEATURES

        filtered = discover_features(mock_agent, allowed_features=set())
        filtered_names = {f.__class__.__name__ for f in filtered}

        # Should only contain mandatory features
        all_features = discover_features(mock_agent)
        all_names = {f.__class__.__name__ for f in all_features}
        expected = MANDATORY_FEATURES & all_names

        assert filtered_names == expected


class TestEntryPointDiscovery:
    """Tests for entry_point-based feature discovery from installed packages."""

    def _make_entry_point(self, name: str, cls: type):
        """Create a mock entry point that loads to the given class."""
        ep = MagicMock()
        ep.name = name
        ep.value = f"{cls.__module__}:{cls.__name__}"
        ep.load.return_value = cls
        return ep

    def test_discovers_entrypoint_features(self):
        """Test that entry_point features are discovered."""
        class ExternalFeature(Feature):
            @property
            def tool_description(self):
                return "External"
            async def initialize(self):
                pass

        ep = self._make_entry_point("ExternalFeature", ExternalFeature)
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        with patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=mock_eps):
            classes = discover_entrypoint_feature_classes()

        assert "ExternalFeature" in classes
        assert classes["ExternalFeature"] is ExternalFeature

    def test_skips_non_feature_entrypoints(self):
        """Test that entry_points not pointing to Feature subclasses are skipped."""
        class NotAFeature:
            pass

        ep = self._make_entry_point("NotAFeature", NotAFeature)
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        with patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=mock_eps):
            classes = discover_entrypoint_feature_classes()

        assert len(classes) == 0

    def test_handles_load_failure(self):
        """Test that failing entry_points are skipped gracefully."""
        ep = MagicMock()
        ep.name = "BrokenFeature"
        ep.load.side_effect = ImportError("missing dependency")
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        with patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=mock_eps):
            classes = discover_entrypoint_feature_classes()

        assert len(classes) == 0

    def test_local_features_win_on_duplicate(self):
        """Test that local features take priority over entry_point features on name collision."""
        class DuplicateFeature(Feature):
            @property
            def tool_description(self):
                return "Entry-point version"
            async def initialize(self):
                pass

        ep = self._make_entry_point("HeartbeatFeature", DuplicateFeature)
        # Rename the class to match a real local feature
        DuplicateFeature.__name__ = "HeartbeatFeature"
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        with patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=mock_eps):
            features = discover_features(agent)

        # HeartbeatFeature should be the LOCAL version, not the entry_point one
        heartbeats = [f for f in features if f.__class__.__name__ == "HeartbeatFeature"]
        assert len(heartbeats) == 1
        # The local HeartbeatFeature won't be our DuplicateFeature class
        assert heartbeats[0].__class__ is not DuplicateFeature

    def test_entrypoint_features_loaded_when_no_local_duplicate(self):
        """Test that entry_point features are loaded when there's no local duplicate."""
        class UniqueExternalFeature(Feature):
            @property
            def tool_description(self):
                return "Unique external"
            async def initialize(self):
                pass

        ep = self._make_entry_point("UniqueExternalFeature", UniqueExternalFeature)
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        with patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=mock_eps):
            features = discover_features(agent)

        names = {f.__class__.__name__ for f in features}
        assert "UniqueExternalFeature" in names

    def test_entrypoint_features_respect_disabled(self):
        """Test that KESTREL_DISABLED_FEATURES applies to entry_point features."""
        class DisableMe(Feature):
            @property
            def tool_description(self):
                return "Should be disabled"
            async def initialize(self):
                pass

        ep = self._make_entry_point("DisableMe", DisableMe)
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        with patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=mock_eps), \
             patch.dict(os.environ, {DISABLED_FEATURES_ENV: "DisableMe"}):
            features = discover_features(agent)

        names = {f.__class__.__name__ for f in features}
        assert "DisableMe" not in names

    def test_entrypoint_features_respect_allowed_features(self):
        """Test that allowed_features filtering applies to entry_point features."""
        from kestrel_sovereign.rookery.config import MANDATORY_FEATURES

        class AllowedExternal(Feature):
            @property
            def tool_description(self):
                return "Allowed"
            async def initialize(self):
                pass

        class DeniedExternal(Feature):
            @property
            def tool_description(self):
                return "Denied"
            async def initialize(self):
                pass

        eps = [
            self._make_entry_point("AllowedExternal", AllowedExternal),
            self._make_entry_point("DeniedExternal", DeniedExternal),
        ]
        mock_eps = MagicMock()
        mock_eps.select.return_value = eps

        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        allowed = {"AllowedExternal"} | MANDATORY_FEATURES
        with patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=mock_eps):
            features = discover_features(agent, allowed_features=allowed)

        names = {f.__class__.__name__ for f in features}
        assert "AllowedExternal" in names
        assert "DeniedExternal" not in names

    def test_empty_entrypoints_no_error(self):
        """Test that no entry_points is handled gracefully."""
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=mock_eps):
            classes = discover_entrypoint_feature_classes()

        assert classes == {}

    def test_entrypoint_group_constant(self):
        """Test that the entry_point group constant is correct."""
        assert FEATURE_ENTRY_POINT_GROUP == "kestrel_sovereign.features"
