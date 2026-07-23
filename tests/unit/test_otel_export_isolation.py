"""Regression guard for #2704: the test suite must never ship real OTLP spans.

The autouse ``_isolate_otel_export_env`` fixture in ``tests/conftest.py`` strips
``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` /
``KESTREL_OTEL_PROJECT`` from every test's environment and resets the cached
tracers, so a machine-default endpoint (autowired via the talon launcher's
``~/.config/kestrel-talon/config.env`` and inherited by the worktree ``pytest``)
can no longer make the LLM-span chokepoint build a real ``OTLPSpanExporter`` and
ship placeholder ``llm.get_response``/``llm.generate*`` spans to the live Phoenix
``default`` project. These tests pin that behaviour (the sovereign twin of
talon#82).

The module and test names carry "otel"/"span"/"telemetry" so the acceptance gate
``pytest tests/unit -q -k "telemetry or otel or span"`` selects them.
"""

import pytest

from kestrel_sdk.tracing import KestrelTracer

from kestrel_sovereign import telemetry


class _NetworkExportSentinel:
    """Stand-in for the real OTLP network exporter.

    Constructing one, or calling ``export`` on it, means a code path tried to
    ship spans to the live endpoint — the #2704 regression. Every such attempt
    is recorded so the guard can assert none happened.
    """

    attempts: list = []

    def __init__(self, *args, **kwargs):
        type(self).attempts.append(("construct", args, kwargs))

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        type(self).attempts.append(("export", list(spans)))
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


@pytest.fixture
def network_export_sentinel(monkeypatch):
    """Patch the real OTLP exporter with a sentinel; expose its attempt log.

    The sovereign LLM-span pipeline builds its exporter inside
    ``kestrel_sdk.tracing.configure`` (via ``telemetry._configure_kestrel_tracer``),
    so patching the name where that module looked it up catches any attempt to
    construct a network exporter regardless of how the span cycle is driven.
    """
    _NetworkExportSentinel.attempts = []
    monkeypatch.setattr(
        "kestrel_sdk.tracing.OTLPSpanExporter",
        _NetworkExportSentinel,
    )
    return _NetworkExportSentinel.attempts


def _emit_llm_span_cycle():
    """Drive a prompt → span → response cycle through the public span API."""
    prompt = telemetry.serialize_llm_input(
        system_prompt="You are helpful.", user_prompt="What is 6*7?"
    )
    with telemetry.llm_span(
        "llm.get_response", input_value=prompt, agent_name="Nellie"
    ) as span:
        telemetry.annotate_llm_response_span(
            span, output_text="The answer is 42.", model_name="openai:api/gpt-5.5"
        )
        return span


def test_autouse_fixture_disables_llm_tracing_by_default():
    # The autouse conftest fixture stripped OTEL_* from the env, so LLM tracing
    # is off by default even if the machine (or a parent talon process) had an
    # endpoint configured — nothing here re-enables it.
    assert telemetry.llm_tracing_enabled() is False
    assert telemetry._resolved_otlp_endpoint() is None


def test_llm_span_cycle_never_exports_over_network(network_export_sentinel):
    # With the autouse fixture active, a full LLM-span cycle through the public
    # API builds no OTLP exporter at all — the tracer is a no-op — so nothing is
    # shipped to a live endpoint and the span never records.
    assert telemetry.llm_tracing_enabled() is False

    span = _emit_llm_span_cycle()

    assert span.is_recording() is False
    assert network_export_sentinel == []


def test_endpoint_set_run_uses_in_memory_exporter_not_network(
    monkeypatch, network_export_sentinel
):
    # Regression guard: even with the endpoint env set, a span cycle through the
    # public API must reach an opt-in in-memory exporter, never the network one.
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = KestrelTracer(tracer=provider.get_tracer("test-otel-isolation"))

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:6006")
    # Opt in explicitly with the in-memory tracer seam (the pattern the suite
    # uses everywhere), so enablement never routes through the network exporter.
    telemetry.set_kestrel_tracer(tracer)

    assert telemetry.llm_tracing_enabled() is True
    _emit_llm_span_cycle()

    names = {s.name for s in exporter.get_finished_spans()}
    # The cycle really ran (the span landed in the fake exporter) ...
    assert "llm.get_response" in names
    # ... but the real OTLP network exporter was never constructed or called.
    assert network_export_sentinel == []
