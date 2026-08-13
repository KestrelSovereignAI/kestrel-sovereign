"""
Tests for the Feature Discovery module.
"""

import importlib
import os
from types import SimpleNamespace

import pytest
from unittest.mock import Mock, patch, MagicMock
from kestrel_sovereign.features import (
    FeatureDiscoveryAmbiguityError,
    MandatoryFeatureReadinessError,
    discover_features,
    discover_feature_class_by_name,
    resolve_feature_canonical_name,
    discover_feature_modules,
    discover_entrypoint_feature_classes,
    discover_feature_selections,
    get_disabled_features,
    get_feature_by_name,
    find_feature_class,
    DISABLED_FEATURES_ENV,
    FEATURE_ENTRY_POINT_GROUP,
)
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sdk.features.base import Feature as _SDKFeature


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
        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: "VoiceFeature"}):
            result = get_disabled_features()
            assert result == {"VoiceFeature"}

    def test_multiple_features_disabled(self):
        """Test disabling multiple features."""
        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: "VoiceFeature,CreativeFeature,MCPAgent"}):
            result = get_disabled_features()
            assert result == {"VoiceFeature", "CreativeFeature", "MCPAgent"}

    def test_handles_whitespace(self):
        """Test that whitespace is trimmed."""
        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: " VoiceFeature , CreativeFeature "}):
            result = get_disabled_features()
            assert result == {"VoiceFeature", "CreativeFeature"}


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

    def test_resolves_feature_class_by_shorthand(self):
        """Runtime lifecycle tools can resolve installed features by shorthand."""
        feature_class = discover_feature_class_by_name("bootstrap")

        assert feature_class is not None
        assert feature_class.__name__ == "BootstrapFeature"


class TestResolveFeatureCanonicalName:
    """resolve_feature_canonical_name must cover isolated-venv features that
    discover_feature_class_by_name cannot import (#1946 codex follow-up)."""

    def test_resolves_in_process_feature_to_class_name(self):
        assert resolve_feature_canonical_name("bootstrap") == "BootstrapFeature"
        assert resolve_feature_canonical_name("BootstrapFeature") == "BootstrapFeature"

    def test_unknown_feature_returns_none(self):
        assert resolve_feature_canonical_name("definitely-not-a-feature") is None

    def test_resolves_isolated_feature_by_class_name_and_shorthand(self):
        """An installed isolated-venv feature (never imported in-process) must
        resolve — by exact class name AND shorthand — to its class name, so
        spawn validation does not reject a feature the child loader can load."""
        runtime = MagicMock()
        runtime.runtime = "isolated-venv"
        runtime.class_name = "WeatherFeature"

        fake_runtimes = {"WeatherFeature": runtime}
        with patch(
            "kestrel_sovereign.feature_registry.discover_installed_feature_runtimes",
            return_value=fake_runtimes,
        ):
            # Not resolvable as an importable class...
            assert discover_feature_class_by_name("WeatherFeature") is None
            # ...but resolvable as a loadable isolated feature, by class name...
            assert resolve_feature_canonical_name("WeatherFeature") == "WeatherFeature"
            # ...and by shorthand (loader keys on the exact class name).
            assert resolve_feature_canonical_name("weather") == "WeatherFeature"


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
        
        # All returned items should be Feature instances (sovereign or SDK base)
        for feature in features:
            assert isinstance(feature, (Feature, _SDKFeature)), (
                f"{feature.__class__.__name__} from {feature.__class__.__module__} "
                f"is not a Feature instance"
            )

    def test_disabled_features_not_loaded(self, mock_agent):
        """Test that disabled features are not loaded."""
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        # Get all features first
        all_features = discover_features(mock_agent)
        all_names = {f.__class__.__name__ for f in all_features}

        optional_names = sorted(all_names - MANDATORY_FEATURES)
        if not optional_names:
            pytest.skip("No optional features discovered to test")

        # Mandatory disable attempts now fail closed; this test covers the
        # still-supported optional disable path.
        feature_to_disable = optional_names[0]
        
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

    def test_isolated_entrypoint_builds_proxy_without_importing_feature(self, mock_agent, tmp_path):
        """isolated-venv entry points are proxied from metadata, not imported."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[tool.kestrel.feature]
runtime = "isolated-venv"
service = "isolated_service"
""".strip()
        )
        ep = _IsolatedEntryPoint(
            name="isolated",
            value="heavy_pkg.feature:HeavyFeature",
            dist=_IsolatedDistribution("heavy-pkg", pyproject),
        )

        with patch("kestrel_sovereign.features.discover_feature_modules", return_value=[]), \
             patch("kestrel_sovereign.features.importlib.metadata.entry_points", return_value=_IsolatedEntryPoints([ep])), \
             patch("kestrel_sovereign.feature_registry.importlib.metadata.entry_points", return_value=_IsolatedEntryPoints([ep])):
            features = discover_features(mock_agent)

        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        names = {feature.name for feature in features}
        assert names == set(MANDATORY_FEATURES) | {"HeavyFeature"}
        proxy = next(feature for feature in features if feature.name == "HeavyFeature")
        assert proxy.runtime.runtime == "isolated-venv"
        assert ep.loaded is False


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
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        # Load all features
        all_features = discover_features(mock_agent)
        all_names = {f.__class__.__name__ for f in all_features}

        # Pick a small subset to allow
        allowed = {"BootstrapFeature", "MemoryFeature", "HealthFeature"}
        filtered = discover_features(mock_agent, allowed_features=allowed)
        filtered_names = {f.__class__.__name__ for f in filtered}

        # Should contain only allowed + mandatory features
        expected = allowed | (MANDATORY_FEATURES & all_names)
        assert filtered_names == expected
        assert len(filtered) < len(all_features)

    def test_mandatory_features_always_load(self, mock_agent):
        """Test that mandatory features load even when not in allowlist."""
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        # Allow only non-mandatory features
        allowed = {"BootstrapFeature"}
        filtered = discover_features(mock_agent, allowed_features=allowed)
        filtered_name_list = [f.__class__.__name__ for f in filtered]
        filtered_names = {f.__class__.__name__ for f in filtered}

        # Every declared mandatory feature is a hard discovery postcondition.
        for mandatory in MANDATORY_FEATURES:
            assert mandatory in filtered_names, f"Mandatory feature {mandatory} missing"
            assert filtered_name_list.count(mandatory) == 1

    def test_none_allowed_features_loads_all(self, mock_agent):
        """Test that None allowed_features loads everything (backward compat)."""
        all_features = discover_features(mock_agent)
        none_features = discover_features(mock_agent, allowed_features=None)

        all_names = {f.__class__.__name__ for f in all_features}
        none_names = {f.__class__.__name__ for f in none_features}

        assert all_names == none_names

    def test_empty_allowed_features_loads_only_mandatory(self, mock_agent):
        """Test that empty set loads only mandatory features."""
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        filtered = discover_features(mock_agent, allowed_features=set())
        filtered_names = {f.__class__.__name__ for f in filtered}

        # Should only contain mandatory features
        assert filtered_names == MANDATORY_FEATURES


class TestMandatoryFeatureReadiness:
    """The sovereignty foundation fails closed at discovery."""

    @pytest.fixture
    def mock_agent(self):
        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()
        return agent

    @pytest.mark.parametrize(
        "feature_name",
        [
            "ConstitutionFeature",
            "IdentityFeature",
            "PeersFeature",
            "SecurityFeature",
            "WaitFeature",
        ],
    )
    def test_mandatory_feature_cannot_be_host_disabled(
        self, mock_agent, feature_name
    ):
        with patch.dict(
            os.environ,
            {DISABLED_FEATURES_ENV: feature_name},
            clear=False,
        ), pytest.raises(MandatoryFeatureReadinessError) as exc_info:
            discover_features(mock_agent)

        error = exc_info.value
        assert error.feature_name == feature_name
        assert error.stage == "configuration"
        assert feature_name in str(error)

    @pytest.mark.parametrize(
        ("feature_name", "module_path"),
        [
            ("IdentityFeature", "kestrel_sovereign.features.identity.feature"),
            ("SecurityFeature", "kestrel_sovereign.features.security.feature"),
            ("PeersFeature", "kestrel_sovereign.features.peers.feature"),
            ("ConstitutionFeature", "kestrel_sovereign.features.constitution"),
            ("WaitFeature", "kestrel_sovereign.features.wait.feature"),
        ],
    )
    def test_mandatory_import_failure_is_typed_and_sanitized(
        self, mock_agent, feature_name, module_path
    ):
        real_import = importlib.import_module

        def import_with_failure(name, package=None):
            if name == module_path:
                raise ImportError("secret-token=must-not-reach-health")
            return real_import(name, package)

        with patch(
            "kestrel_sovereign.features.importlib.import_module",
            side_effect=import_with_failure,
        ), pytest.raises(MandatoryFeatureReadinessError) as exc_info:
            discover_features(mock_agent)

        error = exc_info.value
        assert error.feature_name == feature_name
        assert error.stage == "import"
        assert "secret-token" not in str(error)
        assert "secret-token" in str(error.__cause__)

    @pytest.mark.parametrize(
        "feature_name",
        [
            "ConstitutionFeature",
            "IdentityFeature",
            "PeersFeature",
            "SecurityFeature",
            "WaitFeature",
        ],
    )
    def test_mandatory_constructor_failure_is_typed_and_sanitized(
        self, mock_agent, feature_name
    ):
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURE_MODULES

        module = importlib.import_module(MANDATORY_FEATURE_MODULES[feature_name])
        feature_class = getattr(module, feature_name)
        with patch.object(
            feature_class,
            "__init__",
            side_effect=RuntimeError("api-key=must-not-reach-health"),
        ), pytest.raises(MandatoryFeatureReadinessError) as exc_info:
            discover_features(mock_agent)

        error = exc_info.value
        assert error.feature_name == feature_name
        assert error.stage == "construction"
        assert "api-key" not in str(error)
        assert "api-key" in str(error.__cause__)


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

    def test_duplicate_external_class_names_fail_with_owners(self):
        """Enumeration order must never decide which distribution wins."""
        from kestrel_sovereign.features import DuplicateFeatureEntryPointError

        first = MagicMock()
        first.name = "ReflectionFeature"
        first.value = "kestrel_feature_reflection:ReflectionFeature"
        first.dist.name = "kestrel-feature-reflection"
        second = MagicMock()
        second.name = "ReflectionFeature"
        second.value = "kestrel_feature_intelligence:ReflectionFeature"
        second.dist.name = "kestrel-feature-intelligence"
        mock_eps = MagicMock()
        mock_eps.select.return_value = [second, first]

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ), pytest.raises(DuplicateFeatureEntryPointError) as exc:
            discover_entrypoint_feature_classes()

        message = str(exc.value)
        assert "ReflectionFeature" in message
        assert "kestrel-feature-intelligence" in message
        assert "kestrel-feature-reflection" in message
        assert "uninstall" in message
        first.load.assert_not_called()
        second.load.assert_not_called()

    def test_duplicate_metadata_for_same_owner_is_not_a_class_conflict(self):
        """Layered/editable sys.path duplication must not impersonate two owners."""
        class ExternalFeature(Feature):
            @property
            def tool_description(self):
                return "External"

            async def initialize(self):
                pass

        first = self._make_entry_point("ExternalFeature", ExternalFeature)
        second = self._make_entry_point("ExternalFeature", ExternalFeature)
        first.dist.name = second.dist.name = "one-distribution"
        mock_eps = MagicMock()
        mock_eps.select.return_value = [first, second]

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            classes = discover_entrypoint_feature_classes()

        assert classes == {"ExternalFeature": ExternalFeature}

    def test_unrecognized_local_external_ambiguity_fails_actionably(self):
        """A bundled/external collision must never resolve by enumeration order."""
        class DuplicateFeature(Feature):
            @property
            def tool_description(self):
                return "Entry-point version"
            async def initialize(self):
                pass

        ep = self._make_entry_point("HealthFeature", DuplicateFeature)
        # Rename the class to match a real local feature
        DuplicateFeature.__name__ = "HealthFeature"
        ep.dist.name = "surprise-health-package"
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ), pytest.raises(FeatureDiscoveryAmbiguityError) as exc:
            discover_features(agent)

        message = str(exc.value)
        assert "HealthFeature" in message
        assert "surprise-health-package" in message
        assert "no extracted-over-bundled migration" in message
        assert "feature_registry.toml" in message

    def test_external_talon_loads_as_an_ordinary_entry_point(self):
        """After cutover, Talon has no bundled predecessor to replace."""

        class ExtractedTalonCoordinatorFeature(Feature):
            @property
            def tool_description(self):
                return "Extracted Talon"

            async def initialize(self):
                pass

        ExtractedTalonCoordinatorFeature.__name__ = "TalonCoordinatorFeature"
        ExtractedTalonCoordinatorFeature.__module__ = (
            "kestrel_feature_talon.coordinator"
        )
        ep = self._make_entry_point(
            "TalonCoordinatorFeature",
            ExtractedTalonCoordinatorFeature,
        )
        ep.dist.name = "kestrel-feature-talon"
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            selections = discover_feature_selections()

        selected = selections["TalonCoordinatorFeature"]
        assert selected.feature_class is ExtractedTalonCoordinatorFeature
        assert selected.source == "entry-point"
        assert selected.distribution == "kestrel-feature-talon"
        assert selected.implementation_module.startswith("kestrel_feature_talon")

    def test_no_external_talon_means_no_talon_feature(self):
        """A core-only installation boots without a hidden fallback."""
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            selections = discover_feature_selections()

        assert "TalonCoordinatorFeature" not in selections

    def test_no_external_talon_alias_exists_in_core_only_install(self):
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            assert resolve_feature_canonical_name("talon") is None
            assert resolve_feature_canonical_name("coordinator") is None

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
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

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

    def test_isolated_external_talon_loads_proxy_without_bundled_predecessor(self):
        runtime = InstalledFeatureRuntime(
            class_name="TalonCoordinatorFeature",
            entry_point=(
                "kestrel_feature_talon.coordinator:TalonCoordinatorFeature"
            ),
            distribution="kestrel-feature-talon",
            runtime="isolated-venv",
            service="kestrel-feature-talon-service",
        )
        ep = _IsolatedEntryPoint(
            runtime.class_name,
            runtime.entry_point,
            SimpleNamespace(name=runtime.distribution),
        )
        mock_eps = _IsolatedEntryPoints([ep])

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ), patch(
            "kestrel_sovereign.feature_registry.discover_installed_feature_runtimes",
            return_value={runtime.class_name: runtime},
        ):
            selections = discover_feature_selections()
            features = discover_features(Mock())

        assert runtime.class_name not in selections
        selected = next(feature for feature in features if feature.name == runtime.class_name)
        assert selected.__class__.__name__ == "ProxyFeature"
        assert ep.loaded is False

    def test_discovery_gives_two_isolated_runtimes_distinct_stable_owners(self):
        runtimes = {
            class_name: InstalledFeatureRuntime(
                class_name=class_name,
                entry_point=f"isolated_package.feature:{class_name}",
                distribution="shared-isolated-package",
                runtime="isolated-venv",
                service=f"{class_name.lower()}-service",
            )
            for class_name in ("FirstIsolatedFeature", "SecondIsolatedFeature")
        }
        entry_points = _IsolatedEntryPoints(
            [
                _IsolatedEntryPoint(
                    runtime.class_name,
                    runtime.entry_point,
                    SimpleNamespace(name=runtime.distribution),
                )
                for runtime in runtimes.values()
            ]
        )
        agent = Mock(did="did:test:isolated-discovery")

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=entry_points,
        ), patch(
            "kestrel_sovereign.feature_registry.discover_installed_feature_runtimes",
            return_value=runtimes,
        ):
            features = discover_features(
                agent,
                allowed_features=set(runtimes),
            )

        proxies = [feature for feature in features if feature.name in runtimes]
        assert {feature.name for feature in proxies} == set(runtimes)
        assert len({feature.contribution_owner for feature in proxies}) == 2
        assert all(entry_point.loaded is False for entry_point in entry_points)

    def test_broken_external_talon_does_not_fall_back_to_core(self):
        runtime = InstalledFeatureRuntime(
            class_name="TalonCoordinatorFeature",
            entry_point=(
                "kestrel_feature_talon.coordinator:TalonCoordinatorFeature"
            ),
            distribution="kestrel-feature-talon",
            runtime="in-process",
        )
        ep = MagicMock()
        ep.name = runtime.class_name
        ep.value = runtime.entry_point
        ep.dist.name = runtime.distribution
        ep.load.side_effect = ImportError("broken extracted install")
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ), patch(
            "kestrel_sovereign.feature_registry.discover_installed_feature_runtimes",
            return_value={runtime.class_name: runtime},
        ):
            selections = discover_feature_selections()

        assert "TalonCoordinatorFeature" not in selections

    def test_external_lookup_aliases_use_implementation_module(self):
        class ForecastFeature(Feature):
            @property
            def tool_description(self):
                return "Forecast"

            async def initialize(self):
                pass

        ForecastFeature.__module__ = "vendor_weather.feature"
        ep = self._make_entry_point("forecast-entry-alias", ForecastFeature)
        ep.dist.name = "vendor-weather"
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            assert discover_feature_class_by_name("vendor_weather") is ForecastFeature
            assert discover_feature_class_by_name("forecast-entry-alias") is None


class TestEntrypointClassName:
    """`_entrypoint_class_name` derives the CLASS name (matching the allowlist
    namespace) from an entry point's value, even under an alias (#1788)."""

    def test_alias_entry_point_resolves_to_class_name(self):
        from kestrel_sovereign.features import _entrypoint_class_name
        # alias name differs from the class the value points at
        assert _entrypoint_class_name(
            "kestrel_feature_github.feature:GitHubFeature", "github"
        ) == "GitHubFeature"

    def test_class_named_entry_point_unchanged(self):
        from kestrel_sovereign.features import _entrypoint_class_name
        assert _entrypoint_class_name(
            "kestrel_feature_voice:VoiceFeature", "VoiceFeature"
        ) == "VoiceFeature"

    def test_nested_attribute_uses_innermost_name(self):
        from kestrel_sovereign.features import _entrypoint_class_name
        assert _entrypoint_class_name("mod:Outer.Inner", "x") == "Inner"

    def test_no_attribute_falls_back_to_entry_point_name(self):
        from kestrel_sovereign.features import _entrypoint_class_name
        assert _entrypoint_class_name("", "FallbackFeature") == "FallbackFeature"

    def test_dist_map_uses_value_derived_class_name(self):
        """discover_entrypoint_feature_dists keys by the class from ep.value."""
        from kestrel_sovereign.features import discover_entrypoint_feature_dists

        ep = MagicMock()
        ep.name = "github"  # alias, NOT the class name
        ep.value = "kestrel_feature_github.feature:GitHubFeature"
        ep.dist.name = "kestrel-feature-github"
        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep]

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            dist_map = discover_entrypoint_feature_dists()

        assert dist_map == {"GitHubFeature": "kestrel-feature-github"}


class _IsolatedEntryPoint:
    def __init__(self, name, value, dist):
        self.name = name
        self.value = value
        self.dist = dist
        self.loaded = False

    def load(self):
        self.loaded = True
        raise AssertionError("isolated feature entry point should not be imported")


class _IsolatedEntryPoints(list):
    def select(self, group):
        return self


class _IsolatedPackageFile:
    name = "pyproject.toml"

    def __str__(self):
        return "pyproject.toml"


class _IsolatedDistribution:
    def __init__(self, name, pyproject):
        self.name = name
        self.files = [_IsolatedPackageFile()]
        self._pyproject = pyproject

    def locate_file(self, package_file):
        return self._pyproject

    def read_text(self, name):
        return None
