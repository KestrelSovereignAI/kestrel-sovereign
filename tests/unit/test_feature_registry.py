"""
Tests for the Feature Registry catalog and loader.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign.feature_registry import (
    EXTENSION_ENTRY_POINT_GROUPS,
    FeaturePackageInfo,
    FeatureStatus,
    InstalledFeatureRuntime,
    PackageBoundary,
    RegistryValidationError,
    SkillInfo,
    discover_installed_feature_runtimes,
    get_all_skills,
    get_installed_feature_runtime,
    get_package_for_feature,
    get_registry,
    get_skills_for_package,
    load_registry,
    resolve_status,
    validate_registry,
    REGISTRY_PATH,
)
from kestrel_sovereign.features import discover_local_feature_class_names


class TestLoadRegistry:
    """Tests for loading the static catalog."""

    def test_loads_bundled_catalog(self):
        """The bundled feature_registry.toml loads successfully."""
        registry = load_registry()
        assert len(registry) > 0

    def test_all_18_packages_present(self):
        """The bundled catalog includes the expected capability inventory."""
        registry = load_registry()
        # Historical issues specified 18 entries; the modern catalog includes
        # bundled features, optional feature packages, and provider packages.
        assert len(registry) >= 18, f"Expected >=18 packages, got {len(registry)}: {sorted(registry.keys())}"

    def test_each_entry_has_required_fields(self):
        """Every entry has common catalog fields and boundary-specific names."""
        registry = load_registry()
        for name, info in registry.items():
            assert info.package, f"{name} missing package"
            assert info.git, f"{name} missing git"
            assert info.description, f"{name} missing description"
            if info.boundary in {
                PackageBoundary.BUNDLED,
                PackageBoundary.FEATURE_PACKAGE,
            }:
                assert info.features, f"{name} has no Feature lifecycle classes"
            else:
                assert info.features == [], f"{name} conflates non-Feature classes"

    def test_each_entry_has_tags_and_icon(self):
        """Every entry has tags and icon for UI rendering."""
        registry = load_registry()
        for name, info in registry.items():
            assert len(info.tags) > 0, f"{name} has no tags"
            assert info.icon, f"{name} missing icon"

    def test_known_packages_present(self):
        """Spot-check known packages from the issue."""
        registry = load_registry()
        assert "cloud" in registry
        assert "voice" in registry
        assert "mcp" in registry
        assert "github" in registry
        assert "identity" in registry
        assert "memory" in registry

    def test_cloud_package_contents(self):
        """Verify cloud is the core operator surface, not provider packages."""
        registry = load_registry()
        cloud = registry["cloud"]
        assert cloud.package == "kestrel-sovereign"
        assert "RunPodFeature" not in cloud.features
        assert "VastAIFeature" not in cloud.features
        assert "GCPComputeFeature" not in cloud.features
        assert "DeployFeature" in cloud.features
        assert "ComputeFeature" in cloud.features
        assert "gpu" in cloud.tags
        assert cloud.icon == "cloud"
        assert cloud.core is True

    def test_voice_package_contents(self):
        """Verify the voice package matches the issue specification."""
        registry = load_registry()
        voice = registry["voice"]
        assert voice.package == "kestrel-feature-voice"
        assert "VoiceFeature" in voice.features
        assert voice.icon == "microphone"

    def test_xai_voice_provider_package_contents(self):
        """The published xAI provider distribution is installable by registry."""
        voice_xai = load_registry()["voice_xai"]
        assert voice_xai.package == "kestrel-voice-xai"
        assert voice_xai.boundary is PackageBoundary.PROVIDER_PACKAGE
        assert voice_xai.features == []
        assert set(voice_xai.provider_classes) == {
            "XAITTSProvider",
            "XAISTTProvider",
            "XAIRealtimeConversationProvider",
        }

    def test_missing_file_returns_empty(self):
        """Loading from a nonexistent path returns empty dict."""
        registry = load_registry(Path("/nonexistent/path.toml"))
        assert registry == {}

    def test_all_known_feature_classes_mapped(self):
        """Every feature class in the registry is from the known inventory."""
        registry = load_registry()
        all_classes = set()
        for info in registry.values():
            all_classes.update(info.features)
        # Spot check — these are from KESTREL_FEATURES.md
        expected = {
            "IdentityFeature", "SecurityFeature", "PeersFeature",
            "ConstitutionFeature", "MemoryFeature", "ModelAgent",
            "VoiceFeature", "MCPAgent", "DeployFeature",
            "SpawnFeature", "ObservabilityFeature",
        }
        assert expected.issubset(all_classes), f"Missing: {expected - all_classes}"

    def test_talon_models_external_feature_and_standalone_companion(self):
        registry = load_registry()

        coordinator = registry["talon"]
        assert coordinator.boundary is PackageBoundary.FEATURE_PACKAGE
        assert coordinator.package == "kestrel-feature-talon"
        assert coordinator.features == ["TalonCoordinatorFeature"]
        assert coordinator.companion == "talon_cli"
        assert coordinator.core is False

        companion = registry["talon_cli"]
        assert companion.boundary is PackageBoundary.STANDALONE_TOOL
        assert companion.package == "kestrel-talon"
        assert companion.command == "kestrel-talon"
        assert companion.features == []

    def test_privacy_is_bundled_but_not_a_feature_lifecycle_class(self):
        privacy = load_registry()["privacy"]

        assert privacy.boundary is PackageBoundary.BUNDLED_COMPONENT
        assert privacy.package == "kestrel-sovereign"
        assert privacy.features == []
        assert privacy.bundled_components == ["PrivacyAgent"]

    def test_bundled_registry_exactly_matches_in_tree_discovery(self):
        registry = load_registry()
        validate_registry(
            registry,
            bundled_feature_classes=discover_local_feature_class_names(),
        )

    @pytest.mark.parametrize(
        ("info", "message"),
        [
            (
                FeaturePackageInfo(
                    name="bad-bundle",
                    package="external-package",
                    git="https://example.com",
                    features=["BadFeature"],
                    description="bad",
                    core=True,
                    boundary=PackageBoundary.BUNDLED,
                ),
                "bundled rows must be owned",
            ),
            (
                FeaturePackageInfo(
                    name="bad-provider",
                    package="provider-package",
                    git="https://example.com",
                    features=["NotAFeature"],
                    description="bad",
                    boundary=PackageBoundary.PROVIDER_PACKAGE,
                    provider_classes=["Provider"],
                    entry_point_groups=["kestrel_feature_voice_providers"],
                ),
                "provider implementations belong",
            ),
            (
                FeaturePackageInfo(
                    name="bad-tool",
                    package="standalone",
                    git="https://example.com",
                    features=[],
                    description="bad",
                    boundary=PackageBoundary.STANDALONE_TOOL,
                ),
                "must declare their command",
            ),
        ],
    )
    def test_validation_rejects_ambiguous_boundary_combinations(
        self, info, message
    ):
        with pytest.raises(RegistryValidationError, match=message):
            validate_registry({info.name: info})

    def test_registry_toml_requires_explicit_boundary(self, tmp_path):
        path = tmp_path / "registry.toml"
        path.write_text(
            """
[ambiguous]
package = "some-package"
git = "https://example.com"
features = ["SomeFeature"]
description = "Ambiguous"
""".strip()
        )

        with pytest.raises(RegistryValidationError, match="missing required"):
            load_registry(path)


class TestResolveStatus:
    """Tests for runtime status resolution."""

    def test_default_status_is_available(self):
        """Packages default to AVAILABLE."""
        registry = {
            "test": FeaturePackageInfo(
                name="test",
                package="test-pkg",
                git="https://example.com",
                features=["TestFeature"],
                description="Test",
            ),
        }
        resolved = resolve_status(registry)
        assert registry["test"].status == FeatureStatus.AVAILABLE

    def test_core_packages_are_installed(self):
        """Core packages (shipped with kestrel-sovereign) resolve to INSTALLED."""
        registry = {
            "test": FeaturePackageInfo(
                name="test",
                package="kestrel-sovereign",
                git="https://example.com",
                features=["TestFeature"],
                description="Test",
                core=True,
            ),
        }
        resolved = resolve_status(registry)
        assert registry["test"].status == FeatureStatus.INSTALLED

    def test_voice_provider_entry_point_marks_package_installed(self):
        """Provider entry points count as installed, not only Feature classes."""
        registry = {
            "voice_xai": FeaturePackageInfo(
                name="voice_xai",
                package="kestrel-voice-xai",
                git="https://example.com/voice-xai.git",
                features=[],
                description="xAI realtime voice",
                boundary=PackageBoundary.PROVIDER_PACKAGE,
                provider_classes=["XAIRealtimeConversationProvider"],
                entry_point_groups=[
                    "kestrel_sovereign.conversation_providers"
                ],
            ),
        }
        ep = _FakeEntryPoint(
            name="XAIRealtime",
            value="xai.realtime:XAIRealtimeConversationProvider",
            dist=_FakeDistribution("kestrel-voice-xai", None),
        )

        with patch(
            "kestrel_sovereign.feature_registry.importlib.metadata.entry_points",
            return_value=_FakeEntryPoints([ep]),
        ):
            resolve_status(registry)

        assert registry["voice_xai"].status == FeatureStatus.INSTALLED

    def test_standalone_distribution_marks_tool_installed(self):
        registry = {
            "tool": FeaturePackageInfo(
                name="tool",
                package="standalone-tool",
                git="https://example.com/tool.git",
                features=[],
                description="Standalone",
                boundary=PackageBoundary.STANDALONE_TOOL,
                command="standalone-tool",
            ),
        }

        with patch(
            "kestrel_sovereign.feature_registry._is_distribution_installed",
            return_value=True,
        ):
            resolve_status(registry)

        assert registry["tool"].status == FeatureStatus.INSTALLED

    def test_provider_class_in_feature_group_does_not_mark_provider_installed(
        self,
    ):
        registry = {
            "provider": FeaturePackageInfo(
                name="provider",
                package="provider-package",
                git="https://example.com/provider.git",
                features=[],
                description="Provider",
                boundary=PackageBoundary.PROVIDER_PACKAGE,
                provider_classes=["SharedName"],
                entry_point_groups=[
                    "kestrel_sovereign.conversation_providers"
                ],
            ),
        }
        ep = _FakeEntryPoint(
            name="SharedName",
            value="feature:SharedName",
            dist=_FakeDistribution("wrong-feature-package", None),
        )

        with patch(
            "kestrel_sovereign.feature_registry.iter_extension_entry_points",
            return_value=[
                ("kestrel_sovereign.features", ep),
            ],
        ):
            resolve_status(registry)

        assert registry["provider"].status == FeatureStatus.AVAILABLE

    def test_extension_groups_cover_current_and_legacy_voice_packages(self):
        assert "kestrel_feature_voice_providers" in EXTENSION_ENTRY_POINT_GROUPS
        assert "kestrel_sovereign.voice_providers" in EXTENSION_ENTRY_POINT_GROUPS
        assert "kestrel_sovereign.conversation_providers" in EXTENSION_ENTRY_POINT_GROUPS

    def test_enabled_features_resolve(self):
        """Features in the enabled set resolve to ENABLED."""
        registry = {
            "test": FeaturePackageInfo(
                name="test",
                package="kestrel-sovereign",
                git="https://example.com",
                features=["TestFeature"],
                description="Test",
                core=True,
            ),
        }
        resolved = resolve_status(registry, enabled_class_names={"TestFeature"})
        assert registry["test"].status == FeatureStatus.ENABLED

    def test_disabled_features_resolve(self):
        """Features disabled via env var resolve to DISABLED."""
        registry = {
            "test": FeaturePackageInfo(
                name="test",
                package="kestrel-sovereign",
                git="https://example.com",
                features=["TestFeature"],
                description="Test",
                core=True,
            ),
        }
        with patch.dict(os.environ, {"KESTREL_DISABLED_FEATURES": "TestFeature"}):
            resolved = resolve_status(registry)
            assert registry["test"].status == FeatureStatus.DISABLED

    def test_disabled_takes_priority_over_enabled(self):
        """Disabled status wins over enabled."""
        registry = {
            "test": FeaturePackageInfo(
                name="test",
                package="kestrel-sovereign",
                git="https://example.com",
                features=["TestFeature"],
                description="Test",
                core=True,
            ),
        }
        with patch.dict(os.environ, {"KESTREL_DISABLED_FEATURES": "TestFeature"}):
            resolved = resolve_status(registry, enabled_class_names={"TestFeature"})
            assert registry["test"].status == FeatureStatus.DISABLED


class TestInstalledRuntimeMetadata:
    """Tests for installed feature runtime metadata."""

    def test_runtime_defaults_to_in_process_without_metadata(self):
        ep = _FakeEntryPoint(
            name="TestFeature",
            value="test_pkg.feature:TestFeature",
            dist=_FakeDistribution("test-pkg", None),
        )

        with patch(
            "kestrel_sovereign.feature_registry.importlib.metadata.entry_points",
            return_value=_FakeEntryPoints([ep]),
        ):
            runtimes = discover_installed_feature_runtimes()

        assert runtimes["TestFeature"] == InstalledFeatureRuntime(
            class_name="TestFeature",
            entry_point="test_pkg.feature:TestFeature",
            distribution="test-pkg",
        )

    def test_reads_isolated_runtime_from_installed_pyproject(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[tool.kestrel.feature]
runtime = "isolated-venv"
service = "kestrel-feature-test-service"
venv = "/var/kestrel/test/.venv"
description = "Test isolated surface"
""".strip()
        )
        ep = _FakeEntryPoint(
            name="test",
            value="test_pkg.feature:TestFeature",
            dist=_FakeDistribution("test-pkg", pyproject),
        )

        with patch(
            "kestrel_sovereign.feature_registry.importlib.metadata.entry_points",
            return_value=_FakeEntryPoints([ep]),
        ):
            runtime = get_installed_feature_runtime("TestFeature")

        assert runtime is not None
        assert runtime.runtime == "isolated-venv"
        assert runtime.service == "kestrel-feature-test-service"
        assert runtime.venv == "/var/kestrel/test/.venv"
        assert runtime.description == "Test isolated surface"


class TestGetRegistry:
    """Tests for the combined load + resolve shortcut."""

    def test_returns_resolved_registry(self):
        """get_registry returns a populated, status-resolved dict."""
        registry = get_registry()
        assert len(registry) > 0
        # All core packages should be at least INSTALLED
        for name, info in registry.items():
            if info.core:
                assert info.status in (
                    FeatureStatus.INSTALLED,
                    FeatureStatus.ENABLED,
                    FeatureStatus.DISABLED,
                ), f"{name} is core but status={info.status}"

    def test_with_enabled_features(self):
        """Passing enabled class names marks matching packages."""
        registry = get_registry(enabled_class_names={"VoiceFeature"})
        assert registry["voice"].status == FeatureStatus.ENABLED


class TestGetPackageForFeature:
    """Tests for feature class → package lookup."""

    def test_finds_known_feature(self):
        """DeployFeature maps to the cloud package."""
        info = get_package_for_feature("DeployFeature")
        assert info is not None
        assert info.name == "cloud"

    def test_cloud_providers_are_not_feature_classes(self):
        """Provider packages register elsewhere, not as feature classes."""
        assert get_package_for_feature("RunPodFeature") is None
        assert get_package_for_feature("VastAIFeature") is None
        assert get_package_for_feature("GCPComputeFeature") is None

    def test_unknown_feature_returns_none(self):
        """Unknown class name returns None."""
        info = get_package_for_feature("NonexistentFeature")
        assert info is None


class TestSkills:
    """Tests for skill discovery from the registry."""

    def test_get_all_skills_returns_list(self):
        """get_all_skills returns a non-empty flat list."""
        skills = get_all_skills()
        assert len(skills) > 0
        assert all(isinstance(s, SkillInfo) for s in skills)

    def test_skill_has_required_fields(self):
        """Every skill has name, description, and category."""
        skills = get_all_skills()
        for skill in skills:
            assert skill.name, f"Skill missing name: {skill}"
            assert skill.description, f"Skill {skill.name} missing description"
            assert skill.category, f"Skill {skill.name} missing category"

    def test_get_skills_for_package(self):
        """Skills for the model package include list_models and set_model."""
        skills = get_skills_for_package("model")
        skill_names = {s.name for s in skills}
        assert "list_models" in skill_names
        assert "set_model" in skill_names

    def test_get_skills_for_unknown_package(self):
        """Unknown package returns empty list."""
        skills = get_skills_for_package("nonexistent")
        assert skills == []

    def test_skills_loaded_from_catalog(self):
        """Skills from the cloud package include compute-related entries."""
        skills = get_skills_for_package("cloud")
        skill_names = {s.name for s in skills}
        assert "deploy_agent" in skill_names
        assert "write_script" in skill_names
        assert "run_script" in skill_names


class _FakeEntryPoint:
    def __init__(self, name, value, dist=None):
        self.name = name
        self.value = value
        self.dist = dist


class _FakeEntryPoints(list):
    def select(self, group):
        return self


class _FakePackageFile:
    name = "pyproject.toml"

    def __str__(self):
        return "pyproject.toml"


class _FakeDistribution:
    def __init__(self, name, pyproject):
        self.name = name
        self._pyproject = pyproject
        self.files = [_FakePackageFile()] if pyproject else []

    def locate_file(self, package_file):
        return self._pyproject

    def read_text(self, name):
        return None
