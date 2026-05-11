"""FeatureFeature: workflow contracts for agent-authored features."""

from kestrel_sovereign.features.feature_features.feature import (
    FeatureFeaturesFeature,
)
from kestrel_sovereign.features.feature_features.signals import (
    build_feature_feature_registrations,
)
from kestrel_sovereign.features.feature_features.workflows import (
    FEATURE_FEATURES_COMPENSATION_SOURCES,
    FEATURE_FEATURES_REVIEWER_SOURCES,
    FEATURE_FEATURES_STAGE_SOURCES,
    FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME,
    FEATURE_PROPOSE_TOOL_WORKFLOW_NAME,
    feature_feature_workflow_payloads,
    feature_propose_package_spec_payload,
    feature_propose_tool_spec_payload,
)

__all__ = [
    "FEATURE_FEATURES_COMPENSATION_SOURCES",
    "FEATURE_FEATURES_REVIEWER_SOURCES",
    "FEATURE_FEATURES_STAGE_SOURCES",
    "FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME",
    "FEATURE_PROPOSE_TOOL_WORKFLOW_NAME",
    "FeatureFeaturesFeature",
    "build_feature_feature_registrations",
    "feature_feature_workflow_payloads",
    "feature_propose_package_spec_payload",
    "feature_propose_tool_spec_payload",
]
