"""
Clean install verification tests (Issue #548).

Verifies that the SDK, core sovereign, and feature package import paths
work correctly — the same imports that a clean install would exercise.

For full isolated-venv verification, run: scripts/verify_clean_install.sh

Test matrix:
  1. SDK-only imports
  2. Core sovereign imports + server health
  3. Feature package imports (wallet)
  4. Feature package with SDK-only base class check
  5. Full stack: sovereign + wallet + intelligence + entry_point discovery
"""

import importlib
import importlib.metadata
import sys

import pytest

# Check which optional feature packages are installed
_has_intelligence = importlib.util.find_spec("kestrel_feature_intelligence") is not None
_has_wallet = importlib.util.find_spec("kestrel_feature_wallet") is not None


# =============================================================================
# Test 1: SDK only
# =============================================================================


class TestSDKOnly:
    """Verify kestrel-sovereign-sdk imports work standalone."""

    def test_sdk_feature_base_import(self):
        """from kestrel_sdk.features.base import Feature"""
        from kestrel_sdk.features.base import Feature
        assert Feature is not None
        assert hasattr(Feature, "initialize")
        assert hasattr(Feature, "get_tools")

    def test_sdk_tool_decorator_import(self):
        """from kestrel_sdk.features.base import tool"""
        from kestrel_sdk.features.base import tool
        assert callable(tool)

    def test_sdk_tools_base_import(self):
        """SDK tool types are importable."""
        from kestrel_sdk.tools.base import ToolCategory, ToolSchema, AgentTool
        assert ToolCategory is not None
        assert ToolSchema is not None
        assert AgentTool is not None

    def test_sdk_hooks_base_import(self):
        """SDK hook types are importable."""
        from kestrel_sdk.hooks.base import Hook, HookEvent
        assert Hook is not None
        assert HookEvent is not None

    def test_sdk_a2a_types_import(self):
        """SDK A2A types are importable."""
        from kestrel_sdk.a2a.types import Task, TaskState, TaskStatus
        assert Task is not None
        assert TaskState is not None

    def test_sdk_config_constants_import(self):
        """SDK config constants are importable."""
        from kestrel_sdk.config.constants import HTTP_TIMEOUT_DEFAULT
        assert isinstance(HTTP_TIMEOUT_DEFAULT, (int, float))

    def test_sdk_security_import(self):
        """SDK security types are importable."""
        from kestrel_sdk.security.exceptions import SecurityError
        assert issubclass(SecurityError, Exception)


# =============================================================================
# Test 2: Core sovereign (no feature packages)
# =============================================================================


class TestCoreSovereign:
    """Verify core kestrel-sovereign imports work."""

    def test_sovereign_feature_base_import(self):
        """from kestrel_sovereign.features.base import Feature"""
        from kestrel_sovereign.features.base import Feature
        assert Feature is not None
        assert hasattr(Feature, "initialize")

    def test_sovereign_feature_base_reexports_sdk(self):
        """kestrel_sovereign.features.base re-exports SDK types."""
        from kestrel_sovereign.features.base import Feature as SovFeature
        from kestrel_sdk.features.base import Feature as SDKFeature
        # Sovereign Feature should be a subclass of or same as SDK Feature
        assert issubclass(SovFeature, SDKFeature) or SovFeature is SDKFeature

    def test_sovereign_feature_discovery_import(self):
        """Feature discovery module is importable."""
        from kestrel_sovereign.features import (
            discover_features,
            discover_entrypoint_feature_classes,
            FEATURE_ENTRY_POINT_GROUP,
        )
        assert callable(discover_features)
        assert callable(discover_entrypoint_feature_classes)
        assert FEATURE_ENTRY_POINT_GROUP == "kestrel_sovereign.features"

    def test_sovereign_agent_import(self):
        """Core agent class is importable."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        assert KestrelAgent is not None

    def test_sovereign_llm_service_import(self):
        """LLM service is importable."""
        from kestrel_sovereign.llm.service import LLMService
        assert LLMService is not None

    def test_sovereign_privacy_import(self):
        """Privacy modes are importable."""
        from kestrel_sovereign.privacy import PrivacyMode
        assert hasattr(PrivacyMode, "EPHEMERAL")
        assert hasattr(PrivacyMode, "NORMAL")

    def test_sovereign_server_importable(self):
        """server module can be imported (FastAPI app)."""
        import server
        assert hasattr(server, "app")

    def test_sovereign_feature_registry_import(self):
        """Feature registry is importable and loads catalog."""
        from kestrel_sovereign.feature_registry import (
            load_registry,
            FeatureStatus,
        )
        registry = load_registry()
        assert len(registry) > 0, "Registry should have entries"
        assert hasattr(FeatureStatus, "AVAILABLE")
        assert hasattr(FeatureStatus, "ENABLED")


# =============================================================================
# Test 3: Feature package import (wallet)
# =============================================================================


class TestFeaturePackageWallet:
    """Verify kestrel-feature-wallet imports work."""

    def test_wallet_feature_import(self):
        """from kestrel_feature_wallet import WalletFeature"""
        from kestrel_feature_wallet import WalletFeature
        assert WalletFeature is not None

    def test_wallet_feature_is_feature_subclass(self):
        """WalletFeature is a subclass of the SDK Feature base."""
        from kestrel_sdk.features.base import Feature
        from kestrel_feature_wallet import WalletFeature
        assert issubclass(WalletFeature, Feature)

    def test_wallet_currency_import(self):
        """Currency enum is importable from wallet package."""
        from kestrel_feature_wallet import Currency
        assert Currency is not None

    def test_wallet_entry_point_registered(self):
        """WalletFeature is registered as an entry_point."""
        eps = importlib.metadata.entry_points()
        feature_eps = eps.select(group="kestrel_sovereign.features")
        names = {ep.name for ep in feature_eps}
        assert "WalletFeature" in names, (
            f"WalletFeature not in entry_points: {names}"
        )


# =============================================================================
# Test 4: Feature package with SDK base class (dev mode pattern)
# =============================================================================


class TestFeatureDevMode:
    """Verify that feature packages can be developed against the SDK alone."""

    def test_sdk_feature_is_valid_base(self):
        """SDK Feature ABC has the required interface for feature packages."""
        from kestrel_sdk.features.base import Feature
        import inspect

        # Feature must be abstract
        assert inspect.isabstract(Feature)

        # Must have these abstract methods
        abstract_methods = Feature.__abstractmethods__
        assert "initialize" in abstract_methods
        assert "tool_description" in abstract_methods

        # Must have these concrete methods
        assert hasattr(Feature, "get_tools")
        assert hasattr(Feature, "get_hooks")
        assert hasattr(Feature, "get_router")
        assert hasattr(Feature, "shutdown")

    def test_feature_subclass_with_sdk_only(self):
        """A minimal feature can be defined using only SDK imports."""
        from kestrel_sdk.features.base import Feature, tool
        from kestrel_sdk.tools.base import ToolCategory
        import inspect

        class MinimalTestFeature(Feature):
            @property
            def tool_description(self):
                return "A minimal test feature"

            async def initialize(self):
                pass

            @tool("test_tool", "A test tool", category=ToolCategory.UTILITY)
            async def test_tool(self):
                return "test result"

        # Verify the class is a valid Feature subclass
        assert issubclass(MinimalTestFeature, Feature)
        assert not inspect.isabstract(MinimalTestFeature)

        # Verify the tool decorator attached metadata
        assert hasattr(MinimalTestFeature.test_tool, "_tool_schema")

    def test_wallet_inherits_sdk_feature(self):
        """WalletFeature MRO includes the SDK Feature base class."""
        from kestrel_sdk.features.base import Feature as SDKFeature
        from kestrel_feature_wallet import WalletFeature
        assert SDKFeature in WalletFeature.__mro__


# =============================================================================
# Test 5: Full stack — sovereign + wallet + intelligence + discovery
# =============================================================================


class TestFullStack:
    """Verify full stack with multiple feature packages.

    Tests that require kestrel-feature-intelligence are skipped if it's not
    installed. The shell script (scripts/verify_clean_install.sh) tests the
    full matrix including intelligence in an isolated venv.
    """

    @pytest.mark.skipif(not _has_wallet, reason="kestrel-feature-wallet not installed")
    def test_sovereign_and_wallet_importable(self):
        """Core sovereign and wallet package import without errors."""
        from kestrel_sovereign.features.base import Feature
        from kestrel_feature_wallet import WalletFeature
        assert Feature is not None
        assert WalletFeature is not None

    @pytest.mark.skipif(not _has_intelligence, reason="kestrel-feature-intelligence not installed")
    def test_intelligence_importable(self):
        """Intelligence package imports without errors."""
        from kestrel_feature_intelligence import ReflectionFeature, CouncilFeature
        assert ReflectionFeature is not None
        assert CouncilFeature is not None

    @pytest.mark.skipif(
        not (_has_wallet and _has_intelligence),
        reason="requires both wallet and intelligence packages",
    )
    def test_all_features_are_feature_subclasses(self):
        """All feature classes inherit from the SDK Feature base."""
        from kestrel_sdk.features.base import Feature
        from kestrel_feature_wallet import WalletFeature
        from kestrel_feature_intelligence import ReflectionFeature, CouncilFeature

        for cls in (WalletFeature, ReflectionFeature, CouncilFeature):
            assert issubclass(cls, Feature), (
                f"{cls.__name__} is not a Feature subclass"
            )

    def test_entry_points_discover_installed_features(self):
        """Entry points discover whatever feature packages are installed."""
        eps = importlib.metadata.entry_points()
        feature_eps = eps.select(group="kestrel_sovereign.features")
        names = {ep.name for ep in feature_eps}

        # At minimum, the packages listed in pyproject.toml entry_points should appear
        if _has_wallet:
            assert "WalletFeature" in names, f"WalletFeature not in entry_points: {names}"
        if _has_intelligence:
            assert "ReflectionFeature" in names, f"ReflectionFeature not in entry_points: {names}"
            assert "CouncilFeature" in names, f"CouncilFeature not in entry_points: {names}"

    def test_entry_points_loadable(self):
        """All registered feature entry_points can be loaded (not just named)."""
        eps = importlib.metadata.entry_points()
        feature_eps = eps.select(group="kestrel_sovereign.features")

        loaded = {}
        errors = []
        for ep in feature_eps:
            try:
                cls = ep.load()
                loaded[ep.name] = cls
            except Exception as e:
                errors.append(f"{ep.name}: {e}")

        assert not errors, f"Failed to load entry_points: {errors}"
        assert len(loaded) > 0, "Expected at least one feature entry_point"

    @pytest.mark.skipif(not _has_wallet, reason="kestrel-feature-wallet not installed")
    def test_feature_discovery_includes_entry_point_features(self):
        """discover_features() picks up entry_point-installed features."""
        from unittest.mock import Mock
        from kestrel_sovereign.features import discover_features

        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        features = discover_features(agent)
        names = {f.__class__.__name__ for f in features}

        assert "WalletFeature" in names, (
            f"WalletFeature not discovered. Found: {names}"
        )

    def test_no_duplicate_features(self):
        """Each feature class appears exactly once in discovery."""
        from unittest.mock import Mock
        from kestrel_sovereign.features import discover_features

        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        features = discover_features(agent)
        names = [f.__class__.__name__ for f in features]
        duplicates = [n for n in set(names) if names.count(n) > 1]
        assert not duplicates, f"Duplicate features discovered: {duplicates}"


# =============================================================================
# Verification script existence check
# =============================================================================


class TestVerificationScript:
    """Verify the shell verification script exists and is executable."""

    def test_script_exists(self):
        """scripts/verify_clean_install.sh exists."""
        import os
        script = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "verify_clean_install.sh"
        )
        assert os.path.isfile(script), f"Script not found: {script}"

    def test_script_executable(self):
        """scripts/verify_clean_install.sh is executable."""
        import os
        import stat
        script = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "verify_clean_install.sh"
        )
        st = os.stat(script)
        assert st.st_mode & stat.S_IXUSR, "Script is not executable"
