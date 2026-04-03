"""
Kestrel Observability Hook — backward-compatible re-export shim.

The canonical implementation now lives in kestrel_feature_observability.observability.hook.
This module re-exports ObservabilityHook and constants so existing imports continue to work.
"""

from kestrel_feature_observability.observability.hook import (  # noqa: F401
    ObservabilityHook,
    EVENT_TYPE_MAP,
)
from kestrel_feature_observability.metrics import (  # noqa: F401
    PROMETHEUS_AVAILABLE,
    HOOK_EVENTS,
    TOOL_CALLS,
    TOOL_DURATION,
)
