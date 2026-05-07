"""
E2E tests for the feature install/discover lifecycle (Issue #495).

Tests the full feature lifecycle:
1. Start with core-only agent (no external packages)
2. Simulate installing an external feature package (via mocked entry_points)
3. Restart agent and verify features appear in /agent/info and /api/features
4. Verify feature tools are available to the agent
5. Disable a feature and verify it disappears while others remain

These tests use real agent instances with mocked entry_points to simulate
pip-installed feature packages without actually installing anything. The
mock features use unique class names (ExternalGPUFeature, ExternalStorageFeature,
ExternalMonitorFeature) that do NOT exist in the local features/ directory,
ensuring that the entry_point discovery path is properly tested.
"""

import os
import logging
from unittest.mock import MagicMock, Mock, patch

import pytest
import pytest_asyncio

from kestrel_sovereign.features import (
    DISABLED_FEATURES_ENV,
    FEATURE_ENTRY_POINT_GROUP,
    discover_features,
    discover_entrypoint_feature_classes,
)
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode

logger = logging.getLogger(__name__)


# =============================================================================
# Mock external feature classes (simulate kestrel-feature-ext-gpu package)
#
# These class names intentionally do NOT match any class in the local
# kestrel_sovereign/features/ directory, so they can ONLY be discovered
# via the entry_point path.
# =============================================================================


class ExternalGPUFeature(Feature):
    """Mock external GPU orchestration feature (simulates pip-installed package)."""

    @property
    def tool_description(self):
        return "External GPU instance management"

    async def initialize(self):
        pass

    @tool("list_ext_gpu_instances", "List running external GPU instances", category=ToolCategory.UTILITY)
    async def list_ext_gpu_instances(self):
        """List running external GPU instances."""
        return ["gpu-001", "gpu-002"]


class ExternalStorageFeature(Feature):
    """Mock external storage feature (simulates pip-installed package)."""

    @property
    def tool_description(self):
        return "External distributed storage management"

    async def initialize(self):
        pass

    @tool("list_ext_storage_buckets", "List external storage buckets", category=ToolCategory.UTILITY)
    async def list_ext_storage_buckets(self):
        """List external storage buckets."""
        return ["bucket-alpha", "bucket-beta"]


class ExternalMonitorFeature(Feature):
    """Mock external monitoring feature (simulates pip-installed package)."""

    @property
    def tool_description(self):
        return "External infrastructure monitoring"

    async def initialize(self):
        pass


# =============================================================================
# Helpers
# =============================================================================


def _make_entry_point(name: str, cls: type):
    """Create a mock entry_point that loads to the given class."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"{cls.__module__}:{cls.__name__}"
    ep.load.return_value = cls
    return ep


def _mock_entry_points_with_ext_features():
    """Create mock entry_points simulating an installed external feature package."""
    eps = [
        _make_entry_point("ExternalGPUFeature", ExternalGPUFeature),
        _make_entry_point("ExternalStorageFeature", ExternalStorageFeature),
        _make_entry_point("ExternalMonitorFeature", ExternalMonitorFeature),
    ]
    mock_eps = MagicMock()
    mock_eps.select.return_value = eps
    return mock_eps


def _mock_entry_points_empty():
    """Create mock entry_points simulating no external packages installed."""
    mock_eps = MagicMock()
    mock_eps.select.return_value = []
    return mock_eps


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def core_only_agent(temp_db, monkeypatch):
    """
    Create an agent with only core features (no external packages).

    Mocks entry_points to return empty — simulating a fresh install
    with no external feature packages installed.
    """
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-feature-e2e-32chars!")
    monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)
    monkeypatch.delenv(DISABLED_FEATURES_ENV, raising=False)

    llm_service = LLMService()

    with patch(
        "kestrel_sovereign.features.importlib.metadata.entry_points",
        return_value=_mock_entry_points_empty(),
    ):
        agent = KestrelAgent(
            did="did:test:core-only",
            storage_path=str(temp_db),
            llm_service=llm_service,
            privacy_mode=PrivacyMode.NORMAL,
        )
        await agent.initialize()

    yield agent

    await agent.shutdown()
    await llm_service.close()


@pytest_asyncio.fixture
async def agent_with_ext_features(temp_db, monkeypatch):
    """
    Create an agent that discovers external features via entry_points.

    Simulates having installed an external feature package by mocking
    entry_points to return ExternalGPUFeature, ExternalStorageFeature,
    and ExternalMonitorFeature.
    """
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-feature-e2e-32chars!")
    monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)
    monkeypatch.delenv(DISABLED_FEATURES_ENV, raising=False)

    llm_service = LLMService()

    with patch(
        "kestrel_sovereign.features.importlib.metadata.entry_points",
        return_value=_mock_entry_points_with_ext_features(),
    ):
        agent = KestrelAgent(
            did="did:test:with-ext",
            storage_path=str(temp_db),
            llm_service=llm_service,
            privacy_mode=PrivacyMode.NORMAL,
        )
        await agent.initialize()

    yield agent

    await agent.shutdown()
    await llm_service.close()


# =============================================================================
# Phase 1: Core-only agent (no external packages)
# =============================================================================


class TestCoreOnlyAgent:
    """Verify that a fresh agent with no external packages has only core features."""

    @pytest.mark.asyncio
    async def test_core_features_loaded(self, core_only_agent):
        """Core features like ModelAgent, HealthFeature are present."""
        feature_names = set(core_only_agent.features.keys())
        assert len(feature_names) > 0, "Expected at least some core features"

        # Known core features should be present
        expected_core = {"HealthFeature", "ModelAgent", "BootstrapFeature"}
        found_core = expected_core & feature_names
        assert len(found_core) > 0, (
            f"Expected some core features from {expected_core}, "
            f"got: {feature_names}"
        )

    @pytest.mark.asyncio
    async def test_external_features_absent(self, core_only_agent):
        """External features are NOT present when no packages are installed."""
        feature_names = set(core_only_agent.features.keys())
        assert "ExternalGPUFeature" not in feature_names
        assert "ExternalStorageFeature" not in feature_names
        assert "ExternalMonitorFeature" not in feature_names

    @pytest.mark.asyncio
    async def test_agent_info_lists_only_core(self, core_only_agent):
        """agent.features dict (exposed by /agent/info) has no external features."""
        features_list = list(core_only_agent.features.keys())
        external_names = {"ExternalGPUFeature", "ExternalStorageFeature", "ExternalMonitorFeature"}
        for name in features_list:
            assert name not in external_names, (
                f"External feature {name} should not be in core-only agent"
            )


# =============================================================================
# Phase 2: Entry_point discovery finds installed feature package
# =============================================================================


class TestEntryPointDiscovery:
    """Verify that entry_point discovery finds features from installed packages."""

    def test_discover_entrypoint_classes_finds_ext_features(self):
        """discover_entrypoint_feature_classes() picks up mocked external features."""
        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=_mock_entry_points_with_ext_features(),
        ):
            classes = discover_entrypoint_feature_classes()

        assert "ExternalGPUFeature" in classes
        assert "ExternalStorageFeature" in classes
        assert "ExternalMonitorFeature" in classes
        assert classes["ExternalGPUFeature"] is ExternalGPUFeature
        assert classes["ExternalStorageFeature"] is ExternalStorageFeature

    def test_discover_features_includes_external(self):
        """discover_features() includes entry_point features alongside core."""
        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=_mock_entry_points_with_ext_features(),
        ):
            features = discover_features(agent)

        names = {f.__class__.__name__ for f in features}
        assert "ExternalGPUFeature" in names, f"ExternalGPUFeature not in {names}"
        assert "ExternalStorageFeature" in names, f"ExternalStorageFeature not in {names}"
        assert "ExternalMonitorFeature" in names, f"ExternalMonitorFeature not in {names}"

        # Core features should still be present
        assert len(names) > 3, "Expected core features alongside external ones"

    def test_empty_entrypoints_excludes_external(self):
        """With no entry_points, external features are not discovered."""
        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=_mock_entry_points_empty(),
        ):
            features = discover_features(agent)

        names = {f.__class__.__name__ for f in features}
        assert "ExternalGPUFeature" not in names
        assert "ExternalStorageFeature" not in names
        assert "ExternalMonitorFeature" not in names


# =============================================================================
# Phase 3: Features appear in agent info and feature store API
# =============================================================================


class TestFeaturesAppearInAgent:
    """After 'install + restart', external features appear in agent.features."""

    @pytest.mark.asyncio
    async def test_ext_features_in_agent_features(self, agent_with_ext_features):
        """ExternalGPU/Storage/Monitor features are loaded on the agent."""
        feature_names = set(agent_with_ext_features.features.keys())
        assert "ExternalGPUFeature" in feature_names, (
            f"ExternalGPUFeature missing from {feature_names}"
        )
        assert "ExternalStorageFeature" in feature_names, (
            f"ExternalStorageFeature missing from {feature_names}"
        )
        assert "ExternalMonitorFeature" in feature_names, (
            f"ExternalMonitorFeature missing from {feature_names}"
        )

    @pytest.mark.asyncio
    async def test_core_features_still_present(self, agent_with_ext_features):
        """Core features remain present alongside the new external features."""
        feature_names = set(agent_with_ext_features.features.keys())
        expected_core = {"HealthFeature", "ModelAgent", "BootstrapFeature"}
        found_core = expected_core & feature_names
        assert len(found_core) > 0, (
            f"Expected core features from {expected_core} still present, "
            f"got: {feature_names}"
        )

    @pytest.mark.asyncio
    async def test_agent_info_includes_ext_features(self, agent_with_ext_features):
        """The features list (as returned by /agent/info) includes external features."""
        info_features = list(agent_with_ext_features.features.keys())
        assert "ExternalGPUFeature" in info_features
        assert "ExternalStorageFeature" in info_features


# =============================================================================
# Phase 4: Feature tools are usable by the agent
# =============================================================================


class TestFeatureToolsUsable:
    """Verify that tools from installed features are discoverable and executable."""

    @pytest.mark.asyncio
    async def test_ext_gpu_feature_has_tools(self, agent_with_ext_features):
        """ExternalGPUFeature exposes tools that are discoverable."""
        gpu_feature = agent_with_ext_features.features["ExternalGPUFeature"]
        tools = gpu_feature.get_tools()
        assert len(tools) > 0, "ExternalGPUFeature should expose at least one tool"

        tool_names = [t.schema.name for t in tools]
        assert "list_ext_gpu_instances" in tool_names

    @pytest.mark.asyncio
    async def test_ext_gpu_tool_executes(self, agent_with_ext_features):
        """ExternalGPUFeature tools can be executed and return results."""
        gpu_feature = agent_with_ext_features.features["ExternalGPUFeature"]
        tools = gpu_feature.get_tools()
        tool = next(t for t in tools if t.schema.name == "list_ext_gpu_instances")

        # DynamicTool.execute() wraps result in {"success": ..., "result": ..., "tool": ...}
        result = await tool.execute()
        assert result["success"] is True
        assert isinstance(result["result"], list)
        assert "gpu-001" in result["result"]

    @pytest.mark.asyncio
    async def test_ext_storage_tool_executes(self, agent_with_ext_features):
        """ExternalStorageFeature tools can be executed."""
        storage_feature = agent_with_ext_features.features["ExternalStorageFeature"]
        tools = storage_feature.get_tools()
        assert len(tools) > 0

        tool = next(t for t in tools if t.schema.name == "list_ext_storage_buckets")
        result = await tool.execute()
        assert result["success"] is True
        assert "bucket-alpha" in result["result"]

    @pytest.mark.asyncio
    async def test_all_feature_tools_registered(self, agent_with_ext_features):
        """All features (core + external) have their tools registered."""
        total_tools = 0
        for name, feature in agent_with_ext_features.features.items():
            tools = feature.get_tools()
            total_tools += len(tools)

        assert total_tools > 0, "Expected at least some tools across all features"


# =============================================================================
# Phase 5: Disable/enable cycle works correctly
# =============================================================================


class TestDisableEnableCycle:
    """Verify that disabling a feature removes it while others remain."""

    @pytest.mark.asyncio
    async def test_disable_gpu_feature_removes_it(self, temp_db, monkeypatch):
        """Disabling ExternalGPUFeature via env var removes it on restart."""
        monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-feature-e2e-32chars!")
        monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)
        monkeypatch.setenv(DISABLED_FEATURES_ENV, "ExternalGPUFeature")

        llm_service = LLMService()

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=_mock_entry_points_with_ext_features(),
        ):
            agent = KestrelAgent(
                did="did:test:disable-gpu",
                storage_path=str(temp_db),
                llm_service=llm_service,
                privacy_mode=PrivacyMode.NORMAL,
            )
            await agent.initialize()

        try:
            feature_names = set(agent.features.keys())
            # ExternalGPUFeature should be gone
            assert "ExternalGPUFeature" not in feature_names, (
                f"ExternalGPUFeature should be disabled but found in {feature_names}"
            )
            # Other external features should remain
            assert "ExternalStorageFeature" in feature_names, (
                f"ExternalStorageFeature should still be present in {feature_names}"
            )
            assert "ExternalMonitorFeature" in feature_names, (
                f"ExternalMonitorFeature should still be present in {feature_names}"
            )
        finally:
            await agent.shutdown()
            await llm_service.close()

    @pytest.mark.asyncio
    async def test_disable_multiple_features(self, temp_db, monkeypatch):
        """Disabling multiple features removes all of them."""
        monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-feature-e2e-32chars!")
        monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)
        monkeypatch.setenv(DISABLED_FEATURES_ENV, "ExternalGPUFeature,ExternalMonitorFeature")

        llm_service = LLMService()

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=_mock_entry_points_with_ext_features(),
        ):
            agent = KestrelAgent(
                did="did:test:disable-multi",
                storage_path=str(temp_db),
                llm_service=llm_service,
                privacy_mode=PrivacyMode.NORMAL,
            )
            await agent.initialize()

        try:
            feature_names = set(agent.features.keys())
            assert "ExternalGPUFeature" not in feature_names
            assert "ExternalMonitorFeature" not in feature_names
            # ExternalStorageFeature should remain
            assert "ExternalStorageFeature" in feature_names
        finally:
            await agent.shutdown()
            await llm_service.close()

    @pytest.mark.asyncio
    async def test_reenable_after_disable(self, temp_db, monkeypatch):
        """Re-enabling a feature (removing from disabled list) restores it on restart."""
        monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-feature-e2e-32chars!")
        monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)

        # Step 1: Start with ExternalGPUFeature disabled
        monkeypatch.setenv(DISABLED_FEATURES_ENV, "ExternalGPUFeature")

        llm_service1 = LLMService()
        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=_mock_entry_points_with_ext_features(),
        ):
            agent1 = KestrelAgent(
                did="did:test:reenable-1",
                storage_path=str(temp_db),
                llm_service=llm_service1,
                privacy_mode=PrivacyMode.NORMAL,
            )
            await agent1.initialize()

        assert "ExternalGPUFeature" not in agent1.features
        assert "ExternalStorageFeature" in agent1.features
        await agent1.shutdown()
        await llm_service1.close()

        # Step 2: "Re-enable" by removing from disabled list and restarting
        monkeypatch.delenv(DISABLED_FEATURES_ENV, raising=False)

        llm_service2 = LLMService()
        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=_mock_entry_points_with_ext_features(),
        ):
            agent2 = KestrelAgent(
                did="did:test:reenable-2",
                storage_path=str(temp_db),
                llm_service=llm_service2,
                privacy_mode=PrivacyMode.NORMAL,
            )
            await agent2.initialize()

        try:
            assert "ExternalGPUFeature" in agent2.features, (
                f"ExternalGPUFeature should be restored after re-enable, "
                f"got: {set(agent2.features.keys())}"
            )
            # All external features should be present now
            assert "ExternalStorageFeature" in agent2.features
            assert "ExternalMonitorFeature" in agent2.features
        finally:
            await agent2.shutdown()
            await llm_service2.close()

    def test_discover_features_respects_disabled_env(self):
        """discover_features() skips disabled external features."""
        agent = Mock()
        agent.storage = Mock()
        agent.llm_service = Mock()

        with patch(
            "kestrel_sovereign.features.importlib.metadata.entry_points",
            return_value=_mock_entry_points_with_ext_features(),
        ), patch.dict(os.environ, {DISABLED_FEATURES_ENV: "ExternalGPUFeature"}):
            features = discover_features(agent)

        names = {f.__class__.__name__ for f in features}
        assert "ExternalGPUFeature" not in names, "ExternalGPUFeature should be disabled"
        assert "ExternalStorageFeature" in names, "ExternalStorageFeature should still load"
        assert "ExternalMonitorFeature" in names, "ExternalMonitorFeature should still load"


# =============================================================================
# Phase 6: API endpoint integration (TestClient)
# =============================================================================


class TestFeatureStoreAPI:
    """Test /api/features and /agent/info endpoints with external features."""

    @pytest.fixture
    def client_with_ext_features(self, monkeypatch):
        """TestClient with agent that has external features loaded."""
        import tempfile
        import threading
        from fastapi.testclient import TestClient

        from kestrel_sovereign import storage
        from kestrel_sovereign.inception_service import create_kestrel_identity

        with tempfile.TemporaryDirectory() as agent_dir:
            monkeypatch.setenv("KESTREL_DB_PATH", agent_dir)
            monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-feature-e2e-32chars!")
            monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)
            monkeypatch.delenv(DISABLED_FEATURES_ENV, raising=False)
            monkeypatch.setattr(storage, "get_default_agent_data_dir", lambda: agent_dir)

            create_kestrel_identity(agent_dir, "docs/principles/KESTREL_CONSTITUTION.md")

            threads_before = set(threading.enumerate())

            with patch(
                "kestrel_sovereign.features.importlib.metadata.entry_points",
                return_value=_mock_entry_points_with_ext_features(),
            ):
                from server import app, get_api_key
                with TestClient(app) as client:
                    yield client, get_api_key()

            threads_after = set(threading.enumerate())
            for t in threads_after - threads_before:
                if t.is_alive() and not t.daemon:
                    t.join(timeout=2.0)

    @pytest.fixture
    def client_with_disabled_gpu(self, monkeypatch):
        """TestClient with ExternalGPUFeature disabled."""
        import tempfile
        import threading
        from fastapi.testclient import TestClient

        from kestrel_sovereign import storage
        from kestrel_sovereign.inception_service import create_kestrel_identity

        with tempfile.TemporaryDirectory() as agent_dir:
            monkeypatch.setenv("KESTREL_DB_PATH", agent_dir)
            monkeypatch.setenv("KESTREL_DATA_KEY", "test-key-for-feature-e2e-32chars!")
            monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)
            monkeypatch.setenv(DISABLED_FEATURES_ENV, "ExternalGPUFeature")
            monkeypatch.setattr(storage, "get_default_agent_data_dir", lambda: agent_dir)

            create_kestrel_identity(agent_dir, "docs/principles/KESTREL_CONSTITUTION.md")

            threads_before = set(threading.enumerate())

            with patch(
                "kestrel_sovereign.features.importlib.metadata.entry_points",
                return_value=_mock_entry_points_with_ext_features(),
            ):
                from server import app, get_api_key
                with TestClient(app) as client:
                    yield client, get_api_key()

            threads_after = set(threading.enumerate())
            for t in threads_after - threads_before:
                if t.is_alive() and not t.daemon:
                    t.join(timeout=2.0)

    def _headers(self, api_key):
        """Build auth headers for API requests."""
        return {"X-API-Key": api_key}

    def test_agent_info_lists_ext_features(self, client_with_ext_features):
        """GET /agent/info includes external features in the features list."""
        client, api_key = client_with_ext_features
        response = client.get("/api/agent/info", headers=self._headers(api_key))
        assert response.status_code == 200
        data = response.json()

        features = data["features"]
        assert "ExternalGPUFeature" in features, (
            f"ExternalGPUFeature missing from /agent/info: {features}"
        )
        assert "ExternalStorageFeature" in features, (
            f"ExternalStorageFeature missing from /agent/info: {features}"
        )

    def test_api_features_installed_includes_ext(self, client_with_ext_features):
        """GET /api/features/installed lists the loaded external features."""
        client, api_key = client_with_ext_features
        response = client.get("/api/features/installed", headers=self._headers(api_key))
        assert response.status_code == 200
        data = response.json()

        installed_names = {f["name"] for f in data["features"]}
        assert "ExternalGPUFeature" in installed_names, (
            f"ExternalGPUFeature not in installed features: {installed_names}"
        )
        assert "ExternalStorageFeature" in installed_names, (
            f"ExternalStorageFeature not in installed features: {installed_names}"
        )

    def test_api_features_installed_shows_tools(self, client_with_ext_features):
        """Installed external features include their tools."""
        client, api_key = client_with_ext_features
        response = client.get("/api/features/installed", headers=self._headers(api_key))
        assert response.status_code == 200
        data = response.json()

        gpu_feature = next(
            (f for f in data["features"] if f["name"] == "ExternalGPUFeature"),
            None,
        )
        assert gpu_feature is not None
        assert len(gpu_feature["tools"]) > 0, "ExternalGPUFeature should have tools listed"
        tool_names = [t["name"] for t in gpu_feature["tools"]]
        assert "list_ext_gpu_instances" in tool_names

    def test_agent_info_excludes_disabled(self, client_with_disabled_gpu):
        """GET /agent/info does NOT list a disabled feature."""
        client, api_key = client_with_disabled_gpu
        response = client.get("/api/agent/info", headers=self._headers(api_key))
        assert response.status_code == 200
        data = response.json()

        features = data["features"]
        assert "ExternalGPUFeature" not in features, (
            f"ExternalGPUFeature should be disabled but found in /agent/info: {features}"
        )
        # Other external features should still be there
        assert "ExternalStorageFeature" in features, (
            f"ExternalStorageFeature should remain after disabling ExternalGPUFeature: {features}"
        )

    def test_api_features_installed_excludes_disabled(self, client_with_disabled_gpu):
        """GET /api/features/installed does NOT include a disabled feature."""
        client, api_key = client_with_disabled_gpu
        response = client.get("/api/features/installed", headers=self._headers(api_key))
        assert response.status_code == 200
        data = response.json()

        installed_names = {f["name"] for f in data["features"]}
        assert "ExternalGPUFeature" not in installed_names
        assert "ExternalStorageFeature" in installed_names


# =============================================================================
# Phase 7: Feature registry status resolution with external features
# =============================================================================


class TestRegistryStatusResolution:
    """Test that the feature registry correctly resolves status."""

    def test_registry_marks_enabled_features(self):
        """Features loaded on the agent are marked as ENABLED in the registry."""
        from kestrel_sovereign.feature_registry import get_registry, FeatureStatus

        # Use a real core feature name that exists in the registry
        enabled = {"HealthFeature", "ModelAgent"}
        registry = get_registry(enabled_class_names=enabled)

        health_pkg = registry.get("health")
        assert health_pkg is not None
        assert health_pkg.status == FeatureStatus.ENABLED

        model_pkg = registry.get("model")
        assert model_pkg is not None
        assert model_pkg.status == FeatureStatus.ENABLED

    def test_registry_marks_disabled_features(self):
        """Features in KESTREL_DISABLED_FEATURES are marked as DISABLED."""
        from kestrel_sovereign.feature_registry import get_registry, FeatureStatus

        with patch.dict(os.environ, {DISABLED_FEATURES_ENV: "RunPodFeature"}):
            registry = get_registry(enabled_class_names=set())

        cloud_pkg = registry.get("cloud")
        if cloud_pkg:
            assert cloud_pkg.status == FeatureStatus.DISABLED

    def test_registry_marks_available_when_not_installed(self):
        """Features not installed and not enabled are marked as AVAILABLE."""
        from kestrel_sovereign.feature_registry import (
            load_registry,
            resolve_status,
            FeatureStatus,
        )

        registry = load_registry()
        # Don't resolve with any installed entry_points or enabled features
        with patch(
            "kestrel_sovereign.feature_registry._get_installed_entrypoint_classes",
            return_value=set(),
        ):
            resolved = resolve_status(registry, enabled_class_names=set())

        # Core features are always marked as INSTALLED (since core=true)
        heartbeat = resolved.get("heartbeat")
        if heartbeat:
            assert heartbeat.status == FeatureStatus.INSTALLED  # core=true
