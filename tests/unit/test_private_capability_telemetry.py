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


def _private_telemetry_calls():
    return [
        {"id": "flat-id", "type": "function", "name": "artifact_publish_text",
         "arguments": {"text": "PRIVATE-ARGUMENT-SENTINEL"},
         "unknown_payload": {"PRIVATE-ARGUMENT-SENTINEL": "PRIVATE-ARGUMENT-SENTINEL"}},
        {"id": "nested-id", "type": "function", "function": {
            "name": "feature_dispatch:visual_plan",
            "arguments": '{"prompt":"PRIVATE-ARGUMENT-SENTINEL"}'}},
        {"id": "anthropic-id", "type": "tool_use", "name": "owner__visual_computer_action",
         "input": {"text": "PRIVATE-ARGUMENT-SENTINEL"}},
        {"id": "ordinary-id", "type": "function", "function": {
            "name": "github_search", "arguments": {"query": "ordinary-visible-query"}}},
        {"name": None, "arguments": {"query": "ordinary-malformed-query"}},
        {"function": {"name": 17}, "arguments": "ordinary-unknown-shape"},
        None,
        "ordinary-unknown-call",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "public", "anonymous"])
async def test_native_llm_rows_elide_private_calls_without_mutating_inputs_or_metrics(tmp_path, mode):
    from copy import deepcopy
    import json
    from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
    from kestrel_sovereign.privacy import get_privacy_preset
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "private-llm.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    store.set_privacy_config_provider(lambda: get_privacy_preset(mode))
    calls = _private_telemetry_calls()
    original = deepcopy(calls)
    try:
        await store.log_llm_call(
            agent_did="did:test:owner", provider="owned-provider", model="owned-model",
            duration_ms=7, input_tokens=3, output_tokens=5, tool_calls=calls,
        )
        row = await backend.fetch_one(
            "SELECT tool_calls,provider,model,input_tokens,output_tokens,duration_ms,success FROM a2a_llm_calls"
        )
        assert "PRIVATE-ARGUMENT-SENTINEL" not in row[0]
        saved = json.loads(row[0])
        assert len(saved) == len(calls)
        for actual, expected in zip(saved[:3], calls[:3]):
            assert actual["id"] == expected["id"] and actual["type"] == expected["type"]
            assert actual.get("name") == expected.get("name")
        assert saved[1]["function"]["name"] == calls[1]["function"]["name"]
        assert saved[3:] == calls[3:]  # Ordinary and unknown shapes retain policy.
        assert tuple(row[1:]) == ("owned-provider", "owned-model", 3, 5, 7, 1)
        assert calls == original
    finally:
        await store.close()


@pytest.mark.parametrize("output_shape", ["sdk", "flat", "list", "envelope"])
def test_actual_llm_spans_elide_private_arguments_and_linked_results(output_shape):
    from copy import deepcopy
    import json
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from kestrel_sdk.tracing import KestrelTracer
    from kestrel_sovereign import telemetry
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
    from kestrel_sovereign.llm.service import LLMService

    calls = _private_telemetry_calls()
    messages = [
        {"role": "assistant", "tool_calls": [calls[0], calls[1], *calls[3:]]},
        {"role": "tool", "tool_call_id": "flat-id", "content": "PRIVATE-RESULT-SENTINEL",
         "unknown_payload": "PRIVATE-RESULT-SENTINEL"},
        {"role": "tool", "tool_call_id": "nested-id", "content": [{"type": "text", "text": "PRIVATE-RESULT-SENTINEL"}]},
        {"role": "assistant", "content": [calls[2], {"type": "text", "text": "ordinary-visible-text"},
            {"type": "text", "name": "visual_example", "text": "ordinary-structured-text", "extra": {"tool_calls": [{"name": "ordinary-embedded-data"}]}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "anthropic-id", "content": "PRIVATE-RESULT-SENTINEL", "is_error": True},
            {"type": "tool_result", "tool_use_id": "ordinary-id", "content": "ordinary-visible-result"},
        ]},
        {"role": "function", "name": "visual_plan", "content": "PRIVATE-RESULT-SENTINEL"},
        {"role": "tool", "tool_call_id": "unclassified-id", "content": "ordinary-unclassified-result"},
    ]
    original_messages = deepcopy(messages)
    if output_shape == "sdk":
        result = LLMResponse(content="ordinary-response", input_tokens=3, output_tokens=5, total_tokens=8,
            tool_calls=[ToolCall(id="returned-call", name="artifact_publish_text", arguments={"text": "PRIVATE-ARGUMENT-SENTINEL"})])
    elif output_shape == "flat":
        result = calls[0]
    elif output_shape == "list":
        result = calls
    else:
        result = {"tool_calls": calls, "usage": {"input_tokens": 3}}
    original_result = deepcopy(result)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = telemetry.get_kestrel_tracer()
    telemetry.set_kestrel_tracer(KestrelTracer(tracer=provider.get_tracer("private-arguments")))
    service = LLMService.__new__(LLMService)
    service.get_last_response_identity = lambda: {"model": "owned-model"}
    try:
        with service._llm_request_span("private-tools", messages=messages) as span:
            assert service._annotate_and_return(span, result) is result
        [span] = exporter.get_finished_spans()
        attributes = dict(span.attributes)
        assert "PRIVATE-ARGUMENT-SENTINEL" not in repr(attributes)
        assert "PRIVATE-RESULT-SENTINEL" not in repr(attributes)
        saved = json.loads(attributes["input.value"])
        assert len(saved) == len(messages)
        assert saved[0]["tool_calls"][0]["id"] == "flat-id"
        assert saved[0]["tool_calls"][1]["function"]["name"] == calls[1]["function"]["name"]
        assert saved[0]["tool_calls"][2:] == calls[3:]
        assert saved[1]["tool_call_id"] == "flat-id" and saved[2]["tool_call_id"] == "nested-id"
        assert saved[3]["content"][0]["id"] == "anthropic-id"
        assert saved[3]["content"][1]["text"] == "ordinary-visible-text"
        assert saved[3]["content"][2] == messages[3]["content"][2]
        assert saved[4]["content"][0]["is_error"] is True
        assert saved[4]["content"][1]["content"] == "ordinary-visible-result"
        assert saved[6]["content"] == "ordinary-unclassified-result"
        assert attributes["llm.model_name"] == "owned-model"
        if output_shape == "sdk":
            assert attributes["output.value"] == "ordinary-response"
            assert attributes["llm.usage.prompt_tokens"] == 3
            assert attributes["llm.usage.completion_tokens"] == 5
            assert attributes["llm.usage.total_tokens"] == 8
        assert messages == original_messages and result == original_result
    finally:
        telemetry.set_kestrel_tracer(previous)
        provider.shutdown()


@pytest.mark.asyncio
async def test_native_llm_row_elides_malformed_private_payload_envelope(tmp_path):
    from copy import deepcopy
    from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    call = {"name": "visual_plan", "id": {"text": "PRIVATE-MALFORMED-SENTINEL"},
            "type": ["PRIVATE-MALFORMED-SENTINEL"], "function": "PRIVATE-MALFORMED-SENTINEL",
            "arguments": None, "unexpected": "PRIVATE-MALFORMED-SENTINEL"}
    original = deepcopy(call)
    backend = SQLiteBackend(str(tmp_path / "private-malformed.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        await store.log_llm_call(provider="owned-provider", model="owned-model", duration_ms=1,
                                 tool_calls=[call])
        [payload] = await backend.fetch_one("SELECT tool_calls FROM a2a_llm_calls")
        assert "PRIVATE-MALFORMED-SENTINEL" not in payload
        assert '"name": "visual_plan"' in payload
        assert call == original
    finally:
        await store.close()


def test_actual_llm_span_bounds_cyclic_diagnostic_history_without_mutation():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from kestrel_sdk.tracing import KestrelTracer
    from kestrel_sovereign import telemetry
    from kestrel_sovereign.llm.service import LLMService

    private = {"name": "visual_plan", "arguments": {"prompt": "PRIVATE-CYCLIC-SENTINEL"}}
    messages = [{"role": "assistant", "tool_calls": [private]}]
    messages.append(messages)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = telemetry.get_kestrel_tracer()
    telemetry.set_kestrel_tracer(KestrelTracer(tracer=provider.get_tracer("private-cycle")))
    service = LLMService.__new__(LLMService)
    try:
        with service._llm_request_span("cyclic", messages=messages):
            pass
        [span] = exporter.get_finished_spans()
        assert "PRIVATE-CYCLIC-SENTINEL" not in span.attributes["input.value"]
        assert "<max-depth>" in span.attributes["input.value"]
        assert messages[1] is messages and messages[0]["tool_calls"][0] is private
        assert private["arguments"]["prompt"] == "PRIVATE-CYCLIC-SENTINEL"
    finally:
        telemetry.set_kestrel_tracer(previous)
        provider.shutdown()
