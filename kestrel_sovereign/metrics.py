"""
Kestrel Prometheus Metrics — backward-compatible re-export shim.

The canonical implementation now lives in kestrel_feature_observability.metrics.
This module re-exports all public symbols so existing imports continue to work.
"""

from kestrel_feature_observability.metrics import (  # noqa: F401
    PROMETHEUS_AVAILABLE,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_DURATION,
    LLM_CALLS,
    LLM_DURATION,
    LLM_TOKENS,
    TOOL_CALLS,
    TOOL_DURATION,
    HOOK_EVENTS,
    HOOK_DENIALS,
    CONTEXT_PRESSURE,
    ACTIVE_SESSIONS,
    generate_metrics,
    get_content_type,
)
