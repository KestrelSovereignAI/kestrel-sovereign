"""
OpenTelemetry tracing integration for Kestrel Sovereign.

Provides optional distributed tracing across the agent lifecycle.
All OTEL functionality gracefully degrades to no-ops when the
opentelemetry packages are not installed.

Configuration via standard OTEL environment variables:
    OTEL_EXPORTER_OTLP_ENDPOINT  - OTLP collector endpoint
    OTEL_SERVICE_NAME            - Service name (default: kestrel-sovereign)
    OTEL_TRACES_SAMPLER          - Sampling strategy
    KESTREL_TRACING_ENABLED      - Master switch (default: auto-detect)
"""
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Sentinel: are OTEL packages available?
_OTEL_AVAILABLE = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode, Tracer
    from opentelemetry.trace.propagation import set_span_in_context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation import TraceContextTextMapPropagator

    _OTEL_AVAILABLE = True
except ImportError:
    pass


def is_tracing_enabled() -> bool:
    """Check whether tracing is enabled.

    Tracing is enabled when:
    1. OTEL packages are installed, AND
    2. KESTREL_TRACING_ENABLED is not explicitly set to a falsy value

    When KESTREL_TRACING_ENABLED is unset, auto-detect based on package
    availability and whether an OTLP endpoint is configured.
    """
    if not _OTEL_AVAILABLE:
        return False

    env_val = os.environ.get("KESTREL_TRACING_ENABLED", "").lower()
    if env_val in ("0", "false", "no", "off"):
        return False
    if env_val in ("1", "true", "yes", "on"):
        return True

    # Auto-detect: enable if OTLP endpoint is set
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def setup_tracing(app=None) -> bool:
    """Initialize OpenTelemetry tracing.

    Call once during server startup. If OTEL packages are not installed
    or tracing is disabled, this is a silent no-op.

    Args:
        app: Optional FastAPI app instance for auto-instrumentation.

    Returns:
        True if tracing was successfully initialized, False otherwise.
    """
    global _tracer

    if not is_tracing_enabled():
        logger.debug("OpenTelemetry tracing disabled or packages not installed")
        return False

    try:
        service_name = os.environ.get("OTEL_SERVICE_NAME", "kestrel-sovereign")
        resource = Resource.create({"service.name": service_name})

        provider = TracerProvider(resource=resource)

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        # W3C Trace Context propagation
        propagator = CompositePropagator([TraceContextTextMapPropagator()])
        set_global_textmap(propagator)

        _tracer = trace.get_tracer("kestrel-sovereign")

        # Auto-instrument FastAPI if app is provided
        if app is not None:
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
                FastAPIInstrumentor.instrument_app(app)
                logger.info("FastAPI auto-instrumentation enabled")
            except ImportError:
                logger.debug("opentelemetry-instrumentation-fastapi not installed, skipping")

        logger.info(
            f"OpenTelemetry tracing initialized (service={service_name}, "
            f"endpoint={endpoint or 'none'})"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry tracing: {e}")
        _tracer = None
        return False


def get_tracer() -> Optional[Any]:
    """Return the configured OTEL tracer, or None if tracing is disabled."""
    return _tracer


@contextmanager
def optional_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager that creates an OTEL span if tracing is enabled.

    If tracing is disabled, yields None and does nothing.

    Usage:
        with optional_span("agent.process_input", {"session_id": sid}) as span:
            # ... do work ...
            if span:
                span.set_attribute("result.length", len(result))
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start a new span and return it (or None if tracing is disabled).

    The caller is responsible for ending the span. Prefer optional_span()
    context manager when possible.
    """
    tracer = get_tracer()
    if tracer is None:
        return None

    span = tracer.start_span(name)
    if attributes:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    return span


def end_span(span, error: Optional[Exception] = None):
    """End a span, optionally recording an error."""
    if span is None:
        return

    if error is not None:
        span.set_status(StatusCode.ERROR, str(error))
        span.record_exception(error)
    span.end()
