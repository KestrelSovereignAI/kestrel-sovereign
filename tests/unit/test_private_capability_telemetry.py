"""Private routes stay private at early tracing and inner error boundaries."""
import logging

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["artifacts", "visual-computer"])
async def test_inner_error_handler_does_not_log_cached_private_url_or_exception(caplog, capability):
    from kestrel_sovereign.api_errors import api_unhandled_exception_handler
    path = f"/api/agents/private-agent/api/{capability}/private-id/file"
    request = Request({"type": "http", "method": "GET", "path": path,
                       "query_string": b"path=private-source", "headers": []})
    assert "private-id" in str(request.url)  # Simulate URL cached by auth.
    with caplog.at_level(logging.ERROR, logger="kestrel_sovereign.api_errors"):
        response = await api_unhandled_exception_handler(request, RuntimeError("private-source private-pixels"))
    assert response.status_code == 500
    assert f"Unhandled API error for GET /api/{capability}" in caplog.text
    assert all(value not in caplog.text for value in ("private-agent", "private-id", "private-source", "private-pixels"))
    assert caplog.records[-1].error_type == "RuntimeError"


@pytest.mark.asyncio
@pytest.mark.parametrize("escaped_to_outer_handler", [False, True])
async def test_legacy_artifact_error_handler_redacts_cached_url_and_exception(caplog, escaped_to_outer_handler):
    from kestrel_sovereign.api_errors import api_unhandled_exception_handler
    path = "/api/agents/private-agent/api/workflows/runs/private?run\n/artifacts/private-artifact"
    request = Request({"type": "http", "method": "GET", "path": path,
                       "query_string": b"signature=private-source", "headers": []})
    assert "private-artifact" in str(request.url)
    if escaped_to_outer_handler:
        # On an escaped failure, outer user middleware has already replaced
        # scope.path while Request.url may still hold its original cached URL.
        request.scope["path"] = "/api/workflows/artifact-metadata"
    with caplog.at_level(logging.ERROR, logger="kestrel_sovereign.api_errors"):
        response = await api_unhandled_exception_handler(request, RuntimeError("private-exception"))
    assert response.status_code == 500
    assert "Unhandled API error for GET /api/workflows/artifact-metadata" in caplog.text
    assert "private-" not in caplog.text
    assert caplog.records[-1].error_type == "RuntimeError"


@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", "OTEL_PYTHON_EXCLUDED_URLS"])
async def test_actual_otel_excludes_private_targets_before_span_creation(monkeypatch, env):
    pytest.importorskip("opentelemetry.instrumentation.fastapi")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from kestrel_sovereign import telemetry

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.delenv("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", raising=False)
    monkeypatch.delenv("OTEL_PYTHON_EXCLUDED_URLS", raising=False)
    monkeypatch.setenv(env, "/operator-excluded")
    app = FastAPI()
    @app.get("/{path:path}")
    async def endpoint(path):
        return {"ok": True}
    # Invoke production helper while supplying a local exporter; never publish.
    original = FastAPIInstrumentor.instrument_app
    def instrument(app, **kwargs):
        original(app, tracer_provider=provider, **kwargs)
    monkeypatch.setattr(FastAPIInstrumentor, "instrument_app", instrument)
    telemetry._instrument_fastapi(app)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://synthetic.test") as client:
            for path in ("/api/artifacts/private-id/head", "/api/agents/owner/api/visual-computer/sessions/private-id",
                         "/api/artifacts", "/api/visual-computer", "/operator-excluded",
                         "/api/workflows/runs/private-id/artifacts/private-artifact",
                         "/%61pi/workflows/runs/private-id/%61rtifacts/private-artifact",
                         "/api%2fworkflows%2fruns%2fprivate-id%2fartifacts%2fprivate-artifact"):
                response = await client.get(path, params={"api_key": "private-key", "path": "private-source"})
                assert response.status_code == 200
            # Starlette's route regex rejects the decoded newline; exclusion
            # must also happen before a span can expose this unrouted target.
            response = await client.get("/api/agents/owner/api/workflows/runs/private%3Frun%23id%0A/artifacts/private-artifact")
            assert response.status_code == 404
            assert not exporter.get_finished_spans()
            await client.get("/public")
        spans = exporter.get_finished_spans()
        assert spans and any(span.kind.name == "SERVER" for span in spans)
        assert "private-" not in repr([(s.name, dict(s.attributes)) for s in spans])
    finally:
        FastAPIInstrumentor.uninstrument_app(app)
        provider.shutdown()


def test_redacted_llm_span_preserves_original_failure_without_exporting_chain():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from kestrel_sdk.tracing import KestrelTracer
    from kestrel_sovereign import telemetry
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = telemetry.get_kestrel_tracer()
    telemetry.set_kestrel_tracer(KestrelTracer(tracer=provider.get_tracer("private-test")))
    original = RuntimeError("private-model-response")
    original.__cause__ = ValueError("private-screenshot")
    from kestrel_sovereign.llm.service import LLMService
    service = LLMService.__new__(LLMService)
    try:
        with pytest.raises(RuntimeError) as caught:
            with service._llm_request_span("private", redact=True):
                raise original
        assert caught.value is original
        [span] = exporter.get_finished_spans()
        assert span.status.status_code.name == "ERROR"
        assert span.status.description == "RuntimeError"
        assert "private-" not in repr([dict(event.attributes) for event in span.events])
    finally:
        telemetry.set_kestrel_tracer(previous)
        provider.shutdown()
