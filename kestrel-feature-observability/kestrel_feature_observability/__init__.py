"""
Kestrel Feature: Observability — ops monitoring, wellness metrics, and telemetry.

This package extracts observability, wellness, and metrics from kestrel-sovereign
into a standalone feature package. It provides:

- Prometheus metrics definitions and exposition
- OpenTelemetry tracing integration
- ObservabilityFeature (lifecycle event logging via hook system)
- WellnessFeature (5-dimension operational health monitoring)
- HTTP endpoints for /metrics, /api/observability/*

Install with optional dependencies:
    pip install kestrel-feature-observability[all]        # everything
    pip install kestrel-feature-observability[prometheus]  # Prometheus only
    pip install kestrel-feature-observability[opentelemetry]  # OTEL only
"""

from kestrel_feature_observability.metrics import (
    PROMETHEUS_AVAILABLE,
    REGISTRY,
    generate_metrics,
    get_content_type,
)
from kestrel_feature_observability.telemetry import (
    is_tracing_enabled,
    setup_tracing,
    get_tracer,
    optional_span,
    start_span,
    end_span,
)

__all__ = [
    "PROMETHEUS_AVAILABLE",
    "REGISTRY",
    "generate_metrics",
    "get_content_type",
    "is_tracing_enabled",
    "setup_tracing",
    "get_tracer",
    "optional_span",
    "start_span",
    "end_span",
]
