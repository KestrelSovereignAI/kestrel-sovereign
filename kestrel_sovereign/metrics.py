"""
Kestrel Prometheus Metrics — optional metric definitions for enterprise monitoring.

All metrics are defined here as a single source of truth. If prometheus-client
is not installed, the module exposes no-op stubs so callers never need to
guard imports.

Metrics follow Prometheus naming conventions:
  - kestrel_<subsystem>_<metric>_<unit>
  - Labels have bounded cardinality (no user data).
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect prometheus-client availability
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
    logger.debug("prometheus-client available — Prometheus metrics enabled")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus-client not installed — Prometheus metrics disabled")

# ---------------------------------------------------------------------------
# Registry (use a dedicated registry to avoid default process/platform metrics
# that may not be relevant and to avoid conflicts in tests)
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    REGISTRY = CollectorRegistry()
else:
    REGISTRY = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "kestrel_requests_total",
        "Total HTTP requests by method, path, and status",
        ["method", "path", "status"],
        registry=REGISTRY,
    )
    REQUEST_DURATION = Histogram(
        "kestrel_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "path"],
        registry=REGISTRY,
    )
else:
    REQUEST_COUNT = None  # type: ignore[assignment]
    REQUEST_DURATION = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# LLM metrics
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    LLM_CALLS = Counter(
        "kestrel_llm_calls_total",
        "Total LLM calls by provider, model, and success",
        ["provider", "model", "success"],
        registry=REGISTRY,
    )
    LLM_DURATION = Histogram(
        "kestrel_llm_duration_seconds",
        "LLM call latency in seconds",
        ["provider", "model"],
        registry=REGISTRY,
    )
    LLM_TOKENS = Counter(
        "kestrel_llm_tokens_total",
        "Total LLM tokens by model and direction",
        ["model", "direction"],
        registry=REGISTRY,
    )
else:
    LLM_CALLS = None  # type: ignore[assignment]
    LLM_DURATION = None  # type: ignore[assignment]
    LLM_TOKENS = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Tool metrics
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    TOOL_CALLS = Counter(
        "kestrel_tool_calls_total",
        "Total tool calls by tool name and success",
        ["tool_name", "success"],
        registry=REGISTRY,
    )
    TOOL_DURATION = Histogram(
        "kestrel_tool_duration_seconds",
        "Tool execution latency in seconds",
        ["tool_name"],
        registry=REGISTRY,
    )
else:
    TOOL_CALLS = None  # type: ignore[assignment]
    TOOL_DURATION = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Hook metrics
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    HOOK_EVENTS = Counter(
        "kestrel_hook_events_total",
        "Total hook events by event type",
        ["event_type"],
        registry=REGISTRY,
    )
    HOOK_DENIALS = Counter(
        "kestrel_hook_denials_total",
        "Hook denials by hook name",
        ["hook_name"],
        registry=REGISTRY,
    )
else:
    HOOK_EVENTS = None  # type: ignore[assignment]
    HOOK_DENIALS = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------
if PROMETHEUS_AVAILABLE:
    CONTEXT_PRESSURE = Gauge(
        "kestrel_context_pressure",
        "Context window utilization 0-1",
        registry=REGISTRY,
    )
    ACTIVE_SESSIONS = Gauge(
        "kestrel_active_sessions",
        "Current active sessions",
        registry=REGISTRY,
    )
else:
    CONTEXT_PRESSURE = None  # type: ignore[assignment]
    ACTIVE_SESSIONS = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper: generate Prometheus text output
# ---------------------------------------------------------------------------
def generate_metrics() -> bytes:
    """Return Prometheus text exposition format bytes, or empty if unavailable."""
    if not PROMETHEUS_AVAILABLE or REGISTRY is None:
        return b""
    return generate_latest(REGISTRY)


def get_content_type() -> str:
    """Return the correct Content-Type for Prometheus exposition."""
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain; charset=utf-8"
