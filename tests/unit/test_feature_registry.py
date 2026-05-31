"""
Tests for the Feature Registry catalog and loader.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign.feature_registry import (
    FeaturePackageInfo,
    FeatureStatus,
    SkillInfo,
    get_all_skills,
    get_package_for_feature,
    get_registry,
    get_skills_for_package,
    load_registry,
    resolve_status,
    REGISTRY_PATH,
)


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
        """Every entry has package, git, features, description."""
        registry = load_registry()
        for name, info in registry.items():
            assert info.package, f"{name} missing package"
            assert info.git, f"{name} missing git"
            assert len(info.features) > 0, f"{name} has no features"
            assert info.description, f"{name} missing description"

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
