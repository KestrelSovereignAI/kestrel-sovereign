"""
Wellness Feature — backward-compatible re-export shim.

The canonical implementation now lives in kestrel_feature_observability.wellness.feature.
This module re-exports WellnessFeature so existing imports continue to work.
"""

from kestrel_feature_observability.wellness.feature import (  # noqa: F401
    WellnessFeature,
    WELLNESS_TELEMETRY_ONLY,
)

__all__ = ["WellnessFeature"]
