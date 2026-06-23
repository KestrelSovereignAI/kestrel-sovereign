"""Contract tests for the discoverable feature inventory.

Tests run in two configurations:
1. Core-only — features from kestrel_sovereign/features/ (local discovery)
2. Core + entry_point — core features plus simulated installed packages
"""

import importlib
from unittest.mock import Mock, MagicMock, patch

from kestrel_sovereign.features import (
    DISABLED_FEATURES_ENV,
    FEATURE_ENTRY_POINT_GROUP,
    discover_feature_modules,
    discover_features,
    discover_entrypoint_feature_classes,
    find_feature_class,
    get_feature_by_name,
)
from kestrel_sovereign.features.base import Feature
from kestrel_sdk.features.base import Feature as _SDKFeature


def _mock_agent():
    agent = Mock()
    agent.storage = Mock()
    agent.llm_service = Mock()
    agent.features = {}
    return agent


def _discoverable_feature_classes():
    classes = []
    missing = []
    for module_path in discover_feature_modules():
        module = importlib.import_module(module_path)
        feature_class = find_feature_class(module)
        if feature_class is None:
            missing.append(module_path)
            continue
        classes.append((module_path, feature_class))
    return classes, missing


# ---------------------------------------------------------------------------
# Core-only contracts
# ---------------------------------------------------------------------------


def test_every_discoverable_feature_module_exports_a_feature_class():
    classes, missing = _discoverable_feature_classes()
    assert classes, "Expected at least one discoverable feature class"
    assert missing == [], f"Discoverable modules without Feature subclass: {missing}"


def test_discover_features_matches_unique_class_inventory():
    """discover_features returns exactly the classes from local modules + entry points."""
    classes, missing = _discoverable_feature_classes()
    assert not missing

    expected_names = {feature_class.__name__ for _, feature_class in classes}
    # Entry-point features (extracted packages) are also discovered at runtime
    for cls in discover_entrypoint_feature_classes().values():
        expected_names.add(cls.__name__)
    features = discover_features(_mock_agent())

    # Isolated-venv features have no importable in-process class — discover_features
    # represents each installed isolated-venv runtime by a programmatically built
    # ProxyFeature (skipped by find_feature_class / entry-point class discovery).
    # Model that third path explicitly so an installed isolated feature (e.g.
    # WhatsAppFeature on a host that has it) doesn't break this contract.
    from kestrel_sovereign.feature_registry import discover_installed_feature_runtimes

    isolated_runtimes = [
        runtime
        for runtime in discover_installed_feature_runtimes().values()
        if runtime.runtime == "isolated-venv"
    ]
    proxies = [f for f in features if f.__class__.__name__ == "ProxyFeature"]
    non_proxy = [f for f in features if f.__class__.__name__ != "ProxyFeature"]
    non_proxy_names = {feature.__class__.__name__ for feature in non_proxy}

    # In-process features match the source inventory exactly (one each, no dups)...
    assert non_proxy_names == expected_names
    assert len(non_proxy) == len(expected_names)
    # ...and there's exactly one ProxyFeature per installed isolated-venv runtime.
    assert len(proxies) == len(isolated_runtimes)
    assert all(isinstance(feature, (Feature, _SDKFeature)) for feature in features)


def test_disabled_feature_env_filters_exact_class_names():
    classes, missing = _discoverable_feature_classes()
    assert not missing

    class_names = sorted({feature_class.__name__ for _, feature_class in classes})
    disabled = ",".join(class_names[:2])

    with patch.dict("os.environ", {DISABLED_FEATURES_ENV: disabled}):
        features = discover_features(_mock_agent())

    remaining_names = {feature.__class__.__name__ for feature in features}
    assert class_names[0] not in remaining_names
    assert class_names[1] not in remaining_names


def test_get_feature_by_name_resolves_discovered_features_by_name_and_class():
    features = discover_features(_mock_agent())
    assert features

    sample = features[0]
    by_class = get_feature_by_name(features, sample.__class__.__name__)
    by_name = get_feature_by_name(features, sample.name)

    assert by_class is sample
    assert by_name is sample


# ---------------------------------------------------------------------------
# Entry-point contract helpers
# ---------------------------------------------------------------------------


def _make_entry_point(name: str, cls: type):
    """Create a mock entry point that loads to the given class."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"{cls.__module__}:{cls.__name__}"
    ep.load.return_value = cls
    return ep


def _patch_empty_entrypoints():
    """Context manager that makes entry_point discovery return nothing."""
    mock_eps = MagicMock()
    mock_eps.select.return_value = []
    return patch(
        "kestrel_sovereign.features.importlib.metadata.entry_points",
        return_value=mock_eps,
    )


class _SyntheticExternalFeature(Feature):
    """A synthetic Feature subclass used to simulate an installed package feature."""

    @property
    def tool_description(self):
        return "Synthetic external feature for testing"

    async def initialize(self):
        pass


class _AnotherExternalFeature(Feature):
    """Second synthetic Feature for multi-package tests."""

    @property
    def tool_description(self):
        return "Another external feature"

    async def initialize(self):
        pass


# ---------------------------------------------------------------------------
# Core + entry_point contracts
# ---------------------------------------------------------------------------


def test_entrypoint_features_included_in_combined_inventory():
    """Combined: entry_point features appear alongside core features."""
    ep = _make_entry_point("_SyntheticExternalFeature", _SyntheticExternalFeature)
    mock_eps = MagicMock()
    mock_eps.select.return_value = [ep]

    with patch(
        "kestrel_sovereign.features.importlib.metadata.entry_points",
        return_value=mock_eps,
    ):
        features = discover_features(_mock_agent())

    names = {f.__class__.__name__ for f in features}

    # Core features should still be present
    core_classes, _ = _discoverable_feature_classes()
    for _, cls in core_classes:
        assert cls.__name__ in names, f"Core feature {cls.__name__} missing from combined inventory"

    # Entry-point feature should also be present
    assert "_SyntheticExternalFeature" in names


def test_combined_inventory_has_no_duplicates():
    """Combined: no duplicate class names even when entry_point overlaps with core."""
    # Use a core feature name to simulate a collision
    core_classes, _ = _discoverable_feature_classes()
    assert core_classes, "Need at least one core feature for this test"

    colliding_name = core_classes[0][1].__name__

    class CollidingFeature(Feature):
        @property
        def tool_description(self):
            return "Colliding"

        async def initialize(self):
            pass

    CollidingFeature.__name__ = colliding_name

    ep = _make_entry_point(colliding_name, CollidingFeature)
    mock_eps = MagicMock()
    mock_eps.select.return_value = [ep]

    with patch(
        "kestrel_sovereign.features.importlib.metadata.entry_points",
        return_value=mock_eps,
    ):
        features = discover_features(_mock_agent())

    # Count occurrences of the colliding name
    matches = [f for f in features if f.__class__.__name__ == colliding_name]
    assert len(matches) == 1, f"Expected exactly 1 {colliding_name}, got {len(matches)}"

    # The winner should be the core version, not the entry_point version
    assert matches[0].__class__ is not CollidingFeature


def test_combined_inventory_respects_disabled_env():
    """Combined: disabled env filters both core and entry_point features."""
    ep = _make_entry_point("_SyntheticExternalFeature", _SyntheticExternalFeature)
    mock_eps = MagicMock()
    mock_eps.select.return_value = [ep]

    with patch(
        "kestrel_sovereign.features.importlib.metadata.entry_points",
        return_value=mock_eps,
    ), patch.dict("os.environ", {DISABLED_FEATURES_ENV: "_SyntheticExternalFeature"}):
        features = discover_features(_mock_agent())

    names = {f.__class__.__name__ for f in features}
    assert "_SyntheticExternalFeature" not in names


def test_core_only_inventory_stable_without_entrypoints():
    """Core-only: with no entry_points, inventory matches local discovery exactly."""
    with _patch_empty_entrypoints():
        features = discover_features(_mock_agent())

    core_classes, _ = _discoverable_feature_classes()
    expected = {cls.__name__ for _, cls in core_classes}
    actual = {f.__class__.__name__ for f in features}

    assert actual == expected


def test_multiple_entrypoint_packages_all_discovered():
    """Combined: multiple entry_point features all appear in the inventory."""
    eps = [
        _make_entry_point("_SyntheticExternalFeature", _SyntheticExternalFeature),
        _make_entry_point("_AnotherExternalFeature", _AnotherExternalFeature),
    ]
    mock_eps = MagicMock()
    mock_eps.select.return_value = eps

    with patch(
        "kestrel_sovereign.features.importlib.metadata.entry_points",
        return_value=mock_eps,
    ):
        features = discover_features(_mock_agent())

    names = {f.__class__.__name__ for f in features}
    assert "_SyntheticExternalFeature" in names
    assert "_AnotherExternalFeature" in names


def test_entrypoint_group_constant_matches_package_namespace():
    """The entry_point group constant must match the features package path."""
    assert FEATURE_ENTRY_POINT_GROUP == "kestrel_sovereign.features"


def test_all_features_are_feature_instances_in_combined_mode():
    """Combined: every feature in the combined inventory is a Feature instance."""
    ep = _make_entry_point("_SyntheticExternalFeature", _SyntheticExternalFeature)
    mock_eps = MagicMock()
    mock_eps.select.return_value = [ep]

    with patch(
        "kestrel_sovereign.features.importlib.metadata.entry_points",
        return_value=mock_eps,
    ):
        features = discover_features(_mock_agent())

    assert all(isinstance(f, (Feature, _SDKFeature)) for f in features)
