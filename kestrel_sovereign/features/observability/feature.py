"""
Kestrel Observability Feature — backward-compatible re-export shim.

The canonical implementation now lives in kestrel_feature_observability.observability.feature.
This module re-exports ObservabilityFeature so existing imports continue to work.
"""

from kestrel_feature_observability.observability.feature import ObservabilityFeature  # noqa: F401

__all__ = ["ObservabilityFeature"]
