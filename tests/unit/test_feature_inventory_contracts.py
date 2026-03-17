"""Contract tests for the discoverable feature inventory."""

import importlib
from unittest.mock import Mock, patch

from kestrel_sovereign.features import (
    DISABLED_FEATURES_ENV,
    discover_feature_modules,
    discover_features,
    find_feature_class,
    get_feature_by_name,
)
from kestrel_sovereign.features.base import Feature


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


def test_every_discoverable_feature_module_exports_a_feature_class():
    classes, missing = _discoverable_feature_classes()
    assert classes, "Expected at least one discoverable feature class"
    assert missing == [], f"Discoverable modules without Feature subclass: {missing}"


def test_discover_features_matches_unique_class_inventory():
    classes, missing = _discoverable_feature_classes()
    assert not missing

    expected_names = {feature_class.__name__ for _, feature_class in classes}
    features = discover_features(_mock_agent())
    discovered_names = {feature.__class__.__name__ for feature in features}

    assert discovered_names == expected_names
    assert len(features) == len(expected_names)
    assert all(isinstance(feature, Feature) for feature in features)


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
