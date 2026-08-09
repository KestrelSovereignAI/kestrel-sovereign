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
import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Sentinel: are OTEL packages available?
_OTEL_AVAILABLE = False
_tracer = None
# The SDK-owned OpenInference tracer used for LLM-call-site spans (issue #2573).
# Configured lazily by get_kestrel_tracer() / setup_tracing() through
# kestrel_sdk.tracing.configure(), so the whole process shares ONE
# TracerProvider rather than standing up a second competing exporter pipeline.
_kestrel_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional otel install
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
    global _tracer, _kestrel_tracer

    if not is_tracing_enabled():
        logger.debug("OpenTelemetry tracing disabled or packages not installed")
        return False

    try:
        service_name = os.environ.get("OTEL_SERVICE_NAME", "kestrel-sovereign")

        # Route provider + exporter setup through the SDK's tracing helper so the
        # whole process shares ONE TracerProvider: the LLM-call-site OpenInference
        # spans (issue #2573) and these generic lifecycle spans export through the
        # same pipeline instead of a second, competing provider. set_global=True
        # installs it as the global provider so ``optional_span``/``start_span``
        # (which read the global provider) share it too.
        _kestrel_tracer = _configure_kestrel_tracer(
            service_name=service_name, set_global=True
        )

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

        endpoint = _resolved_otlp_endpoint()
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


# ---------------------------------------------------------------------------
# OpenInference LLM-call-site tracing (issue #2573)
#
# Every agent model call becomes an OpenInference LLM span carrying the real
# prompt/response. Spans are emitted through ``kestrel_sdk.tracing`` — the ONE
# implementation of the OpenInference span conventions, shared by core, talon,
# and the observability feature. Core imports it directly (the SDK ships it
# behind the ``[tracing]`` extra), which keeps this on the right side of the
# core-never-imports-features boundary.
# ---------------------------------------------------------------------------

# Env var + default for prompt/response truncation (talon#71 convention).
KESTREL_OTEL_MAX_IO_CHARS_ENV = "KESTREL_OTEL_MAX_IO_CHARS"
_DEFAULT_MAX_IO_CHARS = 20000

# Standard OTLP endpoint env vars — presence of either enables export.
_OTLP_TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
_OTLP_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"

# OpenInference attribute keys used to annotate the span after the call (the
# served model and response text are only known post-call). Prefer the semconv
# constants; fall back to the literal convention strings.
try:
    from openinference.semconv.trace import (
        OpenInferenceSpanKindValues as _OISpanKindValues,
        SpanAttributes as _OISpanAttributes,
    )

    OI_INPUT_VALUE = _OISpanAttributes.INPUT_VALUE
    OI_OUTPUT_VALUE = _OISpanAttributes.OUTPUT_VALUE
    OI_LLM_MODEL_NAME = _OISpanAttributes.LLM_MODEL_NAME
    OI_SESSION_ID = _OISpanAttributes.SESSION_ID
    OI_SPAN_KIND = _OISpanAttributes.OPENINFERENCE_SPAN_KIND
    OI_SPAN_KIND_CHAIN = _OISpanKindValues.CHAIN.value
    OI_SPAN_KIND_TOOL = _OISpanKindValues.TOOL.value
except Exception:  # pragma: no cover - openinference optional
    OI_INPUT_VALUE = "input.value"
    OI_OUTPUT_VALUE = "output.value"
    OI_LLM_MODEL_NAME = "llm.model_name"
    OI_SESSION_ID = "session.id"
    OI_SPAN_KIND = "openinference.span.kind"
    OI_SPAN_KIND_CHAIN = "CHAIN"
    OI_SPAN_KIND_TOOL = "TOOL"

# Attribute key naming the owning agent on a span. Same key #2573/#2602 stamp on
# LLM spans, so telemetry-layer lifecycle/dispatch spans group under the right
# agent lane in Phoenix instead of "(none)" (issue #2699).
KESTREL_AGENT_NAME = "kestrel.agent_name"

# Attribute key naming the conversation session a span belongs to. The fleet
# Timeline's ``sessionKeyOf`` reads only ``kestrel.session_id`` / ``session.id``
# / ``kestrel.run_id`` — the agent runtime's long-standing ``agent.session_id``
# is in none of them, so its turns grouped by trace id instead (one band per
# turn, a staircase lane). Stamp this ALONGSIDE ``agent.session_id`` (issue
# #2916); never stamp it empty, since an absent attribute and an empty one read
# the same to the consumer but only one of them is noise.
KESTREL_SESSION_ID = "kestrel.session_id"


def _resolved_otlp_endpoint() -> Optional[str]:
    """Return the configured OTLP endpoint (traces-specific wins), or None."""
    return (
        os.environ.get(_OTLP_TRACES_ENDPOINT_ENV)
        or os.environ.get(_OTLP_ENDPOINT_ENV)
        or None
    )


class _NullSpan:
    """Inert span used when the SDK tracer is unavailable — accepts every call."""

    def set_attribute(self, *_a: Any, **_k: Any) -> None:
        pass

    def add_event(self, *_a: Any, **_k: Any) -> None:
        pass

    def is_recording(self) -> bool:
        return False


class _NullKestrelTracer:
    """Fallback tracer mirroring the SDK's no-op contract (never errors)."""

    enabled = False

    @contextmanager
    def llm_span(self, *_a: Any, **_k: Any):
        yield _NullSpan()


def _configure_kestrel_tracer(
    *, service_name: Optional[str] = None, set_global: bool = False
):
    """Build a ``KestrelTracer`` via the SDK, or a null tracer if unavailable.

    Never raises — instrumentation must not be able to break startup or a call.
    """
    try:
        import kestrel_sdk.tracing as kestrel_tracing
    except Exception as e:  # pragma: no cover - SDK installed without [tracing]
        logger.debug("kestrel_sdk.tracing unavailable (%s); LLM tracing disabled", e)
        return _NullKestrelTracer()
    if service_name is None:
        service_name = os.environ.get("OTEL_SERVICE_NAME", "kestrel-sovereign")
    try:
        return kestrel_tracing.configure(
            service_name=service_name, set_global=set_global
        )
    except Exception as e:  # pragma: no cover - configure() is itself guarded
        logger.debug("KestrelTracer configure failed (%s); LLM tracing disabled", e)
        return _NullKestrelTracer()


def get_kestrel_tracer():
    """Return the process ``KestrelTracer`` for LLM spans, configuring lazily.

    A no-op when no OTLP endpoint is configured (the SDK returns an inert
    tracer), so callers guard on ``.enabled`` and pay only a cheap check.
    """
    global _kestrel_tracer
    if _kestrel_tracer is None:
        _kestrel_tracer = _configure_kestrel_tracer()
    return _kestrel_tracer


def llm_tracing_enabled() -> bool:
    """Cheap guard: is an exporter-backed LLM tracer configured?

    Call sites use this to skip prompt serialization entirely when tracing is
    off, so an unset OTLP endpoint costs only this check (issue #2573: "no
    overhead beyond a cheap guard").
    """
    try:
        return bool(get_kestrel_tracer().enabled)
    except Exception:  # pragma: no cover - defensive; guard must never raise
        return False


def set_kestrel_tracer(tracer) -> None:
    """Install a ``KestrelTracer`` explicitly.

    Tests inject an in-memory-exporter-backed tracer this way; production wires
    the exporter-backed tracer through :func:`setup_tracing`.
    """
    global _kestrel_tracer
    _kestrel_tracer = tracer


def reset_kestrel_tracer() -> None:
    """Clear the cached tracer so the next call reconfigures from the env."""
    global _kestrel_tracer
    _kestrel_tracer = None


def _reset_tracer_for_tests() -> None:
    """Reset ALL cached tracer state so the next call re-evaluates the env.

    Clears both the generic lifecycle tracer (:func:`setup_tracing`) and the
    SDK-owned LLM tracer (:func:`get_kestrel_tracer`). The autouse test fixture
    calls this — with ``OTEL_EXPORTER_OTLP_*`` / ``KESTREL_OTEL_PROJECT`` stripped
    from the environment — so no test inherits a tracer built from a machine
    default endpoint and ships real spans to a live Phoenix store (issue #2704,
    the sovereign twin of talon#82).
    """
    global _tracer, _kestrel_tracer
    _tracer = None
    _kestrel_tracer = None


def otel_max_io_chars() -> int:
    """Max serialized chars kept in an LLM span's input/output value.

    Reads ``KESTREL_OTEL_MAX_IO_CHARS`` (default 20000); a non-positive or
    unparseable value falls back to the default.
    """
    raw = os.environ.get(KESTREL_OTEL_MAX_IO_CHARS_ENV)
    if not raw:
        return _DEFAULT_MAX_IO_CHARS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_IO_CHARS
    return value if value > 0 else _DEFAULT_MAX_IO_CHARS


def truncate_for_span(text) -> Optional[str]:
    """Truncate ``text`` to the IO cap, appending ``…[truncated N chars]``.

    Same convention as talon#71. ``None`` passes through; non-strings coerce.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:  # pragma: no cover - defensive
            return None
    cap = otel_max_io_chars()
    if len(text) <= cap:
        return text
    dropped = len(text) - cap
    return f"{text[:cap]}…[truncated {dropped} chars]"


def serialize_llm_input(
    *, system_prompt=None, user_prompt=None, messages=None
) -> str:
    """Serialize the prompt/messages sent to the model for an LLM span.

    Produces a JSON messages array when possible (the shape Phoenix renders),
    falling back to concatenated text on any serialization edge case.
    """
    try:
        if messages is not None:
            payload: Any = messages
        else:
            payload = []
            if system_prompt is not None:
                payload.append({"role": "system", "content": system_prompt})
            if user_prompt is not None:
                payload.append({"role": "user", "content": user_prompt})
        return json.dumps(payload, default=str, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive; never break the call
        parts = [p for p in (system_prompt, user_prompt) if p]
        if parts:
            return "\n\n".join(str(p) for p in parts)
        return str(messages) if messages is not None else ""


@contextmanager
def llm_span(
    name, *, input_value=None, model_name=None, agent_name=None, attributes=None
):
    """Open an OpenInference LLM span via the process ``KestrelTracer``.

    Yields the underlying span (real or inert). Nesting is automatic: if an OTel
    span is already current, this nests under it; otherwise it is a root span.

    ``input.value`` is truncated at ``KESTREL_OTEL_MAX_IO_CHARS`` before it lands
    on the span, mirroring the output truncation in
    :func:`annotate_llm_response_span` so both prompt and response honor the same
    cap (issue #2573; talon#71 convention).
    """
    tracer = get_kestrel_tracer()
    with tracer.llm_span(
        name,
        input_value=truncate_for_span(input_value),
        model_name=model_name,
        agent_name=agent_name,
        attributes=attributes,
    ) as span:
        yield span


def annotate_llm_response_span(
    span, *, output_text=None, model_name=None, response=None
) -> None:
    """Stamp ``output.value`` / ``llm.model_name`` and additive token attrs.

    The served model and response text are only known once the call returns, so
    they are set here. Token attributes keep the legacy ``llm.usage.*`` keys
    additively for continuity with the retired ``agent.llm_call`` span.
    """
    if span is None:
        return
    try:
        # Inert/non-recording spans (endpoint unset) short-circuit before any
        # truncation work so annotation is free when tracing is off.
        if not span.is_recording():
            return
    except Exception:  # pragma: no cover - defensive
        return
    try:
        if model_name:
            span.set_attribute(OI_LLM_MODEL_NAME, model_name)
        if output_text is not None:
            span.set_attribute(OI_OUTPUT_VALUE, truncate_for_span(output_text))
        if response is not None:
            for attr, key in (
                ("input_tokens", "llm.usage.prompt_tokens"),
                ("output_tokens", "llm.usage.completion_tokens"),
                ("total_tokens", "llm.usage.total_tokens"),
            ):
                value = getattr(response, attr, None)
                if value is not None:
                    span.set_attribute(key, value)
    except Exception:  # pragma: no cover - annotation must never break the call
        logger.debug("Failed to annotate LLM span", exc_info=True)


def record_llm_attempt_failure(provider_name, error, *, redact_content: bool = False) -> None:
    """Record a failed provider attempt as an event on the current LLM span.

    Per-provider fallback attempts are retries of ONE logical LLM request, not
    separate calls (issue #2573, Q1), so they are span *events* on the single
    request span (the entry method's ``llm_span``, which is the current OTel
    span here) rather than their own spans. Cheap no-op when tracing is off or
    no span is recording; never raises.

    ``redact_content`` (#2674 finding 4): under an enforcing response audit the
    turn's prompt/response is withheld pending the verdict, and a provider
    exception's ``str()`` routinely embeds that prompt or the response body. The
    caller threads the per-invocation context's redaction flag here (never global
    state, so concurrent turns are isolated); when set, the event carries only the
    exception CLASS name — a safe category the operator can still triage on —
    instead of the raw error text. The provider name is always safe classification
    and is preserved in both modes.
    """
    if not _OTEL_AVAILABLE:
        return
    try:
        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        if redact_content:
            error_attr = type(error).__name__ if error is not None else ""
        else:
            error_attr = truncate_for_span(str(error)) or ""
        span.add_event(
            "llm.attempt_failed",
            attributes={
                "llm.provider": str(provider_name) if provider_name else "",
                "error": error_attr,
            },
        )
    except Exception:  # pragma: no cover - events must never break the call
        logger.debug("Failed to record LLM attempt event", exc_info=True)
