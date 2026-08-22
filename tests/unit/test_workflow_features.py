"""Metadata-only workflow-to-Feature ownership resolution."""

from types import SimpleNamespace

import pytest

from kestrel_sovereign.workflow_features import (
    FEATURE_ENTRY_POINT_GROUP,
    WORKFLOW_FEATURE_ENTRY_POINT_GROUP,
    WorkflowFeatureResolutionError,
    resolve_workflow_feature,
    resolve_workflow_feature_provider,
)


WORKFLOW_NAME = "fleet_coding_pipeline"
FEATURE_NAME = "ExampleCodingFeature"
FEATURE_VALUE = "example_coding.feature:ExampleCodingFeature"
DIST_NAME = "kestrel-feature-example-coding"


def _entry_point(group, name, value, distribution=DIST_NAME):
    return SimpleNamespace(
        group=group,
        name=name,
        value=value,
        dist=SimpleNamespace(name=distribution) if distribution else None,
        load=lambda: pytest.fail("metadata-only resolver must never load entry points"),
    )


def _valid_entry_points():
    return [
        _entry_point(
            WORKFLOW_FEATURE_ENTRY_POINT_GROUP,
            WORKFLOW_NAME,
            FEATURE_VALUE,
        ),
        _entry_point(FEATURE_ENTRY_POINT_GROUP, FEATURE_NAME, FEATURE_VALUE),
    ]


def test_resolves_same_distribution_feature_without_importing_it():
    provider = resolve_workflow_feature_provider(
        WORKFLOW_NAME,
        entry_points=_valid_entry_points(),
    )
    assert provider.workflow_name == WORKFLOW_NAME
    assert provider.feature_class_name == FEATURE_NAME
    assert provider.entry_point == FEATURE_VALUE
    assert provider.distribution == DIST_NAME
    assert resolve_workflow_feature(
        WORKFLOW_NAME,
        entry_points=_valid_entry_points(),
    ) == FEATURE_NAME


def test_missing_workflow_provider_fails_actionably():
    with pytest.raises(WorkflowFeatureResolutionError, match="no installed provider"):
        resolve_workflow_feature(WORKFLOW_NAME, entry_points=[])


def test_duplicate_workflow_provider_claims_are_rejected():
    entry_points = _valid_entry_points()
    entry_points.append(
        _entry_point(
            WORKFLOW_FEATURE_ENTRY_POINT_GROUP,
            WORKFLOW_NAME,
            "other.feature:OtherFeature",
            "other-distribution",
        )
    )
    with pytest.raises(WorkflowFeatureResolutionError, match="duplicate provider claims"):
        resolve_workflow_feature(WORKFLOW_NAME, entry_points=entry_points)


@pytest.mark.parametrize(
    ("feature_entry_point", "message"),
    [
        (None, "does not publish the matching"),
        (
            _entry_point(
                FEATURE_ENTRY_POINT_GROUP,
                FEATURE_NAME,
                FEATURE_VALUE,
                "different-distribution",
            ),
            "does not publish the matching",
        ),
        (
            _entry_point(
                FEATURE_ENTRY_POINT_GROUP,
                "WrongFeatureAlias",
                FEATURE_VALUE,
            ),
            "entry point is named",
        ),
    ],
)
def test_invalid_feature_ownership_claim_fails_closed(feature_entry_point, message):
    entry_points = [_valid_entry_points()[0]]
    if feature_entry_point is not None:
        entry_points.append(feature_entry_point)
    with pytest.raises(WorkflowFeatureResolutionError, match=message):
        resolve_workflow_feature(WORKFLOW_NAME, entry_points=entry_points)


def test_provider_requires_distribution_metadata_and_module_class_value():
    missing_distribution = [
        _entry_point(
            WORKFLOW_FEATURE_ENTRY_POINT_GROUP,
            WORKFLOW_NAME,
            FEATURE_VALUE,
            distribution="",
        )
    ]
    with pytest.raises(WorkflowFeatureResolutionError, match="no owning distribution"):
        resolve_workflow_feature(WORKFLOW_NAME, entry_points=missing_distribution)

    invalid_value = [
        _entry_point(
            WORKFLOW_FEATURE_ENTRY_POINT_GROUP,
            WORKFLOW_NAME,
            "not-a-class-entry-point",
        )
    ]
    with pytest.raises(WorkflowFeatureResolutionError, match="invalid Feature"):
        resolve_workflow_feature(WORKFLOW_NAME, entry_points=invalid_value)
