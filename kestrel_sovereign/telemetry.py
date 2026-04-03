"""
OpenTelemetry tracing — backward-compatible re-export shim.

The canonical implementation now lives in kestrel_feature_observability.telemetry.
This module re-exports all public symbols so existing imports continue to work.
"""

from kestrel_feature_observability.telemetry import (  # noqa: F401
    _OTEL_AVAILABLE,
    _tracer,
    is_tracing_enabled,
    setup_tracing,
    get_tracer,
    optional_span,
    start_span,
    end_span,
)
