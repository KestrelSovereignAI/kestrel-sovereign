"""Prometheus metrics endpoint — backward-compatible re-export shim.

The canonical implementation now lives in kestrel_feature_observability.endpoints.metrics.
"""

from kestrel_feature_observability.endpoints.metrics import router  # noqa: F401
