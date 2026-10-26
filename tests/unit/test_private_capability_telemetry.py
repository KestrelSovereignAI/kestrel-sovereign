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
                         "/api/artifacts", "/api/visual-computer", "/operator-excluded"):
                response = await client.get(path, params={"api_key": "private-key", "path": "private-source"})
                assert response.status_code == 200
            assert not exporter.get_finished_spans()
            await client.get("/public")
        spans = exporter.get_finished_spans()
        assert spans and any(span.kind.name == "SERVER" for span in spans)
        assert "private-" not in repr([(s.name, dict(s.attributes)) for s in spans])
    finally:
        FastAPIInstrumentor.uninstrument_app(app)
        provider.shutdown()
