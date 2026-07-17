"""OpenInference LLM-span instrumentation at the routing chokepoint (issue #2573).

Every agent model call is expected to emit a single OpenInference ``LLM`` span
through ``kestrel_sdk.tracing`` carrying the real serialized prompt in
``input.value``, the response in ``output.value``, and the served model in
``llm.model_name`` — truncated at ``KESTREL_OTEL_MAX_IO_CHARS`` — while an unset
OTLP endpoint changes nothing beyond a cheap guard.

The module and test names deliberately carry "llm" plus "span"/"tracing"/"otel"
so the acceptance gate ``pytest -k "llm and (span or tracing or otel)"`` selects
them. Spans are captured with an in-memory OTel exporter (no live LLM, no
network); the provider call is stubbed.
"""
import json

import pytest
from unittest.mock import AsyncMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from kestrel_sdk.tracing import KestrelTracer

from kestrel_sovereign import telemetry
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.service import LLMService


# OpenInference attribute keys (literal convention strings, exporter-visible).
OI_KIND = "openinference.span.kind"
OI_INPUT = "input.value"
OI_OUTPUT = "output.value"
OI_MODEL = "llm.model_name"
KESTREL_AGENT_NAME = "kestrel.agent_name"


@pytest.fixture
def otel_llm_exporter():
    """Install an in-memory-exporter-backed ``KestrelTracer`` as the LLM tracer.

    Simulates "an OTLP endpoint is set" without any network: the SDK tracer
    records real OTel spans into an in-memory exporter we can inspect. The
    process-global tracer is reset afterwards so tracing state never leaks
    between tests.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = KestrelTracer(tracer=provider.get_tracer("test-llm-tracing"))
    telemetry.set_kestrel_tracer(tracer)
    try:
        yield exporter
    finally:
        telemetry.reset_kestrel_tracer()


def _only_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected exactly one LLM span, got {len(spans)}"
    return spans[0]


class TestLlmSpanCaptureViaOtel:
    """The LLM span carries the real prompt/response/model via OpenInference."""

    def test_llm_tracing_enabled_guard_true_with_exporter(self, otel_llm_exporter):
        """The cheap guard reports tracing on when an exporter tracer is installed."""
        assert telemetry.llm_tracing_enabled() is True

    def test_llm_span_records_input_output_and_model_otel(self, otel_llm_exporter):
        """One LLM span carries serialized input, response output, and model name."""
        prompt = telemetry.serialize_llm_input(
            system_prompt="You are helpful.", user_prompt="What is 6*7?"
        )
        with telemetry.llm_span(
            "llm.get_response",
            input_value=prompt,
            agent_name="Nellie",
        ) as span:
            telemetry.annotate_llm_response_span(
                span,
                output_text="The answer is 42.",
                model_name="openai:api/gpt-5.5",
            )

        span = _only_span(otel_llm_exporter)
        attrs = dict(span.attributes)
        assert span.name == "llm.get_response"
        assert attrs[OI_KIND] == "LLM"
        # Real serialized prompt round-trips as an OpenInference messages array.
        messages = json.loads(attrs[OI_INPUT])
        assert {"role": "user", "content": "What is 6*7?"} in messages
        assert attrs[OI_OUTPUT] == "The answer is 42."
        assert attrs[OI_MODEL] == "openai:api/gpt-5.5"
        assert attrs[KESTREL_AGENT_NAME] == "Nellie"

    def test_llm_span_truncates_input_beyond_cap_otel(
        self, otel_llm_exporter, monkeypatch
    ):
        """``input.value`` is truncated at KESTREL_OTEL_MAX_IO_CHARS on the span."""
        monkeypatch.setenv(telemetry.KESTREL_OTEL_MAX_IO_CHARS_ENV, "50")
        oversized = "u" * 500
        serialized = telemetry.serialize_llm_input(user_prompt=oversized)
        assert len(serialized) > 50

        with telemetry.llm_span("llm.generate", input_value=serialized):
            pass

        attrs = dict(_only_span(otel_llm_exporter).attributes)
        recorded = attrs[OI_INPUT]
        assert len(recorded) < len(serialized)
        assert recorded.startswith(serialized[:50])
        assert recorded.endswith(f"…[truncated {len(serialized) - 50} chars]")

    def test_llm_span_truncates_output_beyond_cap_otel(
        self, otel_llm_exporter, monkeypatch
    ):
        """``output.value`` is truncated at the same cap when the response is large."""
        monkeypatch.setenv(telemetry.KESTREL_OTEL_MAX_IO_CHARS_ENV, "20")
        big_output = "z" * 300
        with telemetry.llm_span("llm.generate", input_value="hi") as span:
            telemetry.annotate_llm_response_span(span, output_text=big_output)

        recorded = dict(_only_span(otel_llm_exporter).attributes)[OI_OUTPUT]
        assert recorded == "z" * 20 + "…[truncated 280 chars]"

    def test_llm_span_records_failed_attempt_event_otel(self, otel_llm_exporter):
        """A failed provider attempt attaches as an event on the one request span."""
        with telemetry.llm_span("llm.get_response", input_value="hi"):
            telemetry.record_llm_attempt_failure(
                "openai:api", RuntimeError("rate limited")
            )

        span = _only_span(otel_llm_exporter)
        events = list(span.events)
        assert len(events) == 1
        assert events[0].name == "llm.attempt_failed"
        assert events[0].attributes["llm.provider"] == "openai:api"
        assert "rate limited" in events[0].attributes["error"]


class TestLlmTracingDisabledNoOp:
    """Unset endpoint → zero behavior change beyond a cheap guard (no spans)."""

    def test_llm_tracing_disabled_guard_false_without_endpoint(self, monkeypatch):
        """With no OTLP endpoint the LLM-tracing guard is False (no provider built)."""
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        telemetry.reset_kestrel_tracer()
        try:
            assert telemetry.llm_tracing_enabled() is False
        finally:
            telemetry.reset_kestrel_tracer()

    def test_llm_span_is_inert_when_tracing_disabled(self, monkeypatch):
        """The LLM span context manager yields an inert, non-recording span."""
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        telemetry.reset_kestrel_tracer()
        try:
            with telemetry.llm_span("llm.get_response", input_value="hi") as span:
                assert span.is_recording() is False
                # Annotation on an inert span must be a harmless no-op.
                telemetry.annotate_llm_response_span(span, output_text="x", model_name="m")
        finally:
            telemetry.reset_kestrel_tracer()


class TestLlmServiceChokepointSpan:
    """The single LLM span is emitted at the service entry-method chokepoint."""

    @pytest.mark.asyncio
    async def test_get_response_emits_single_llm_span_otel(self, otel_llm_exporter):
        """``LLMService.get_response`` emits one LLM span with real I/O + model.

        The provider call is stubbed (no live LLM); the served model is read from
        the response identity the routing layer stamps post-fallback.
        """
        service = LLMService()
        try:
            service.set_agent_display_name("Nellie")
            service._get_response_frozen = AsyncMock(
                return_value=LLMResponse(
                    content="the answer is 42",
                    input_tokens=3,
                    output_tokens=5,
                    total_tokens=8,
                )
            )
            service.get_last_response_identity = lambda: {"model": "openai:api/gpt-5.5"}

            result = await service.get_response(
                system_prompt="You are helpful.",
                user_prompt="What is 6*7?",
            )
            assert result.content == "the answer is 42"
        finally:
            await service.close()

        span = _only_span(otel_llm_exporter)
        attrs = dict(span.attributes)
        assert span.name == "llm.get_response"
        assert attrs[OI_KIND] == "LLM"
        messages = json.loads(attrs[OI_INPUT])
        assert {"role": "user", "content": "What is 6*7?"} in messages
        assert attrs[OI_OUTPUT] == "the answer is 42"
        assert attrs[OI_MODEL] == "openai:api/gpt-5.5"
        assert attrs[KESTREL_AGENT_NAME] == "Nellie"
        # Legacy token attributes retained additively for continuity.
        assert attrs["llm.usage.prompt_tokens"] == 3
        assert attrs["llm.usage.completion_tokens"] == 5
        assert attrs["llm.usage.total_tokens"] == 8

    @pytest.mark.asyncio
    async def test_get_response_skips_serialization_when_tracing_disabled_otel(
        self, monkeypatch
    ):
        """Tracing off (no OTLP endpoint) → inert span, prompt never serialized.

        The no-op ``KestrelTracer`` (exactly what the SDK returns when no endpoint
        is configured) makes ``llm_tracing_enabled()`` False, so the chokepoint
        pays only the cheap guard: it never serializes the prompt and the span is
        inert — "zero behavior change beyond a cheap guard" (issue #2573).
        """
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        # The inert tracer the SDK hands back when no endpoint is set.
        telemetry.set_kestrel_tracer(KestrelTracer(tracer=None))
        serialize_calls = 0
        real_serialize = telemetry.serialize_llm_input

        def _counting_serialize(**kwargs):
            nonlocal serialize_calls
            serialize_calls += 1
            return real_serialize(**kwargs)

        monkeypatch.setattr(telemetry, "serialize_llm_input", _counting_serialize)

        assert telemetry.llm_tracing_enabled() is False

        service = LLMService()
        try:
            service._get_response_frozen = AsyncMock(
                return_value=LLMResponse(content="ok")
            )
            service.get_last_response_identity = lambda: {"model": "ollama:local/x"}
            result = await service.get_response(system_prompt="s", user_prompt="u")
            assert result.content == "ok"
        finally:
            await service.close()
            telemetry.reset_kestrel_tracer()

        # No exporter, no serialization work — the guard short-circuited it.
        assert serialize_calls == 0
