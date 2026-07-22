"""#2674 finding 3 — content-redaction of provider telemetry under an enforcing
response audit.

When a strict (fail-closed) POST_RESPONSE audit withholds the assistant prose
pending its verdict, the raw provider prompt/response must NOT land in durable
``llm_calls`` telemetry (``user_prompt`` / ``response_preview``) nor the
OpenTelemetry LLM span (``input.value`` / ``output.value``) — and the audit's
OWN provider call (whose ``user_prompt`` IS the withheld prose) must redact too.
Content-free fields (usage/timing/provider/model/error) are preserved. Advisory
turns are unchanged.

Named with "llm" + "tracing"/"observability" so the acceptance selectors pick it.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from kestrel_sdk.tracing import KestrelTracer

from kestrel_sovereign import telemetry
from kestrel_sovereign.llm import service as service_mod
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext


OI_INPUT = "input.value"
OI_OUTPUT = "output.value"
OI_MODEL = "llm.model_name"


@pytest.fixture
def otel_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry.set_kestrel_tracer(
        KestrelTracer(tracer=provider.get_tracer("test-redaction"))
    )
    try:
        yield exporter
    finally:
        telemetry.reset_kestrel_tracer()


def _only_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected one span, got {len(spans)}"
    return spans[0]


class TestSpanRedaction:
    @pytest.mark.asyncio
    async def test_redacting_context_blanks_span_input_and_output(self, otel_exporter):
        service = LLMService()
        try:
            service._get_response_frozen = AsyncMock(
                return_value=LLMResponse(
                    content="the SECRET answer is 42",
                    input_tokens=3, output_tokens=5, total_tokens=8,
                )
            )
            service.get_last_response_identity = lambda: {"model": "openai:api/gpt-5.5"}
            result = await service.get_response(
                system_prompt="You are helpful.",
                user_prompt="What is the SECRET?",
                invocation_context=LLMInvocationContext(
                    session_id="s", redact_content=True,
                ),
            )
            assert result.content == "the SECRET answer is 42"
        finally:
            await service.close()

        attrs = dict(_only_span(otel_exporter).attributes)
        # Neither the prompt nor the response content is on the span.
        assert "SECRET" not in attrs[OI_INPUT]
        assert "SECRET" not in attrs[OI_OUTPUT]
        assert "redacted" in attrs[OI_INPUT]
        assert "redacted" in attrs[OI_OUTPUT]
        # Content-free fields preserved.
        assert attrs[OI_MODEL] == "openai:api/gpt-5.5"
        assert attrs["llm.usage.prompt_tokens"] == 3
        assert attrs["llm.usage.completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_non_redacting_context_keeps_span_content(self, otel_exporter):
        service = LLMService()
        try:
            service._get_response_frozen = AsyncMock(
                return_value=LLMResponse(content="the answer is 42")
            )
            service.get_last_response_identity = lambda: {"model": "m"}
            await service.get_response(
                system_prompt="s", user_prompt="What is 6*7?",
                invocation_context=LLMInvocationContext(session_id="s"),
            )
        finally:
            await service.close()
        attrs = dict(_only_span(otel_exporter).attributes)
        assert "6*7" in attrs[OI_INPUT]
        assert attrs[OI_OUTPUT] == "the answer is 42"


class TestDbTelemetryRedaction:
    async def _capture_log(self, *, redact):
        service = LLMService.__new__(LLMService)  # skip heavy __init__
        captured = {}
        store = MagicMock()

        async def _log(**kwargs):
            captured.update(kwargs)

        store.log_llm_call = AsyncMock(side_effect=_log)
        service._observability_store = store
        service._owner_agent_did = "did"
        await service._log_llm_call(
            provider="openai:api",
            model="gpt-5.5",
            duration_ms=123,
            success=True,
            system_prompt="sys",
            user_prompt="THE SECRET PROMPT",
            response="THE SECRET RESPONSE",
            input_tokens=11,
            output_tokens=22,
            invocation_context=LLMInvocationContext(
                session_id="s", redact_content=redact,
            ),
        )
        return captured

    @pytest.mark.asyncio
    async def test_deny_redacts_db_prompt_and_response_keeps_usage(self):
        cap = await self._capture_log(redact=True)
        assert "SECRET" not in (cap["user_prompt"] or "")
        assert "SECRET" not in (cap["response"] or "")
        assert "redacted" in cap["user_prompt"]
        assert "redacted" in cap["response"]
        assert cap["metadata"].get("content_redacted") is True
        # Content-free fields preserved.
        assert cap["provider"] == "openai:api"
        assert cap["model"] == "gpt-5.5"
        assert cap["duration_ms"] == 123
        assert cap["input_tokens"] == 11
        assert cap["output_tokens"] == 22

    @pytest.mark.asyncio
    async def test_advisory_keeps_db_content(self):
        cap = await self._capture_log(redact=False)
        assert cap["user_prompt"] == "THE SECRET PROMPT"
        assert cap["response"] == "THE SECRET RESPONSE"
        assert cap["metadata"].get("content_redacted") is None


class TestDbToolCallAndErrorRedaction:
    """#2674 finding 4: the same durable ``llm_calls`` row carries two more
    content-bearing columns — ``tool_calls`` (response-derived model arguments)
    and ``error_message`` (``str(error)``, which providers embed prompt/response
    into). Both must redact under the enforcing-audit flag while every
    content-free field survives; advisory turns keep them raw."""

    async def _capture_log(self, *, redact, tool_calls, error_message, success):
        service = LLMService.__new__(LLMService)
        captured = {}

        async def _log(**kwargs):
            captured.update(kwargs)

        store = MagicMock()
        store.log_llm_call = AsyncMock(side_effect=_log)
        service._observability_store = store
        service._owner_agent_did = "did"
        await service._log_llm_call(
            provider="openai:api",
            model="gpt-5.5",
            duration_ms=99,
            success=success,
            user_prompt="prompt",
            response="response",
            tool_calls=tool_calls,
            error_message=error_message,
            input_tokens=7,
            output_tokens=8,
            invocation_context=LLMInvocationContext(
                session_id="s", redact_content=redact,
            ),
        )
        return captured

    @pytest.mark.asyncio
    async def test_strict_blanks_tool_arguments_keeps_name(self):
        cap = await self._capture_log(
            redact=True, success=True, error_message=None,
            tool_calls=[
                {"name": "send_email", "arguments": {"to": "SECRET@x.com",
                                                      "body": "SECRET BODY"}},
                {"name": "read_file", "arguments": {"path": "/SECRET/leak"}},
            ],
        )
        redacted = cap["tool_calls"]
        # Structure/count preserved; each tool NAME (safe classification) kept.
        assert [c["name"] for c in redacted] == ["send_email", "read_file"]
        # No model-generated argument content survives.
        serialized = json.dumps(redacted)
        assert "SECRET" not in serialized
        assert "leak" not in serialized
        for call in redacted:
            assert "redacted" in json.dumps(call["arguments"])

    @pytest.mark.asyncio
    async def test_strict_reduces_error_message_to_marker(self):
        cap = await self._capture_log(
            redact=True, success=False, tool_calls=None,
            error_message="ProviderError: request body was {'prompt': 'SECRET'}",
        )
        assert "SECRET" not in (cap["error_message"] or "")
        assert "redacted error" in cap["error_message"]
        # Content-free fields preserved on the failure row.
        assert cap["provider"] == "openai:api"
        assert cap["model"] == "gpt-5.5"
        assert cap["success"] is False

    @pytest.mark.asyncio
    async def test_advisory_keeps_tool_args_and_error_text(self):
        cap = await self._capture_log(
            redact=False, success=False,
            tool_calls=[{"name": "send_email",
                         "arguments": {"to": "a@x.com"}}],
            error_message="ProviderError: boom",
        )
        assert cap["tool_calls"][0]["arguments"] == {"to": "a@x.com"}
        assert cap["error_message"] == "ProviderError: boom"


class TestOtelAttemptFailureRedaction:
    """#2674 finding 4: the per-provider fallback span event records
    ``str(error)``, which can embed the withheld prompt/response. Under the
    per-invocation redaction flag it must record only the exception class."""

    def _record(self, *, redact):
        from opentelemetry import trace as otel_trace
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test-attempt-failure")
        with tracer.start_as_current_span("llm.get_response"):
            # record_llm_attempt_failure reads trace.get_current_span(); the
            # span started above is the current span within this block.
            assert otel_trace.get_current_span().is_recording()
            telemetry.record_llm_attempt_failure(
                "openai:api",
                RuntimeError("body leaked: SECRET prompt and SECRET response"),
                redact_content=redact,
            )
        return exporter.get_finished_spans()[0]

    def test_redacted_event_records_only_exception_class(self):
        span = self._record(redact=True)
        events = {e.name: dict(e.attributes) for e in span.events}
        attrs = events["llm.attempt_failed"]
        assert attrs["error"] == "RuntimeError"
        assert "SECRET" not in attrs["error"]
        # Provider name is safe classification and is preserved either way.
        assert attrs["llm.provider"] == "openai:api"

    def test_advisory_event_keeps_error_text(self):
        span = self._record(redact=False)
        events = {e.name: dict(e.attributes) for e in span.events}
        assert "SECRET" in events["llm.attempt_failed"]["error"]


class TestAuditProviderCallRedaction:
    """The audit's OWN provider call carries the withheld prose as its
    user_prompt; under strict it must redact its telemetry."""

    @pytest.mark.asyncio
    async def test_get_audit_response_threads_redaction_into_context(self, monkeypatch):
        service = LLMService.__new__(LLMService)
        service._check_policy = lambda: None
        service._resolve_invocation_context = (
            lambda ctx=None, **k: ctx or LLMInvocationContext()
        )
        adapter = MagicMock()
        caps = MagicMock()
        caps.supports_structured_output = True
        adapter.provider_capabilities.return_value = caps
        prov = {"name": "p", "adapter": adapter, "client": object()}
        service.providers = [prov]
        service._get_default_mandate_selector = MagicMock(return_value=None)
        service._mandate_preference = {}
        service._available_providers = MagicMock(return_value=[prov])
        service._resolve_concrete_model = MagicMock(return_value="m")
        monkeypatch.setattr(service_mod, "messages_for", lambda *a, **k: [])

        seen = {}

        async def _attempt(coro, name, model, **kwargs):
            seen.update(kwargs)
            return LLMResponse(content='{"risk_level": 1, "reasoning": "ok"}')

        service._run_provider_attempt = _attempt

        out = await service.get_audit_response(
            "WITHHELD ASSISTANT PROSE", redact_content=True,
        )
        assert out["risk_level"] == 1
        # The provider attempt received a redacting context AND the raw prose as
        # its user_prompt (redaction is applied downstream in _log_llm_call).
        assert seen["invocation_context"].redact_content is True
        assert seen["user_prompt"] == "WITHHELD ASSISTANT PROSE"

    @pytest.mark.asyncio
    async def test_get_audit_response_advisory_not_redacted(self, monkeypatch):
        service = LLMService.__new__(LLMService)
        service._check_policy = lambda: None
        service._resolve_invocation_context = (
            lambda ctx=None, **k: ctx or LLMInvocationContext()
        )
        adapter = MagicMock()
        caps = MagicMock()
        caps.supports_structured_output = True
        adapter.provider_capabilities.return_value = caps
        prov = {"name": "p", "adapter": adapter, "client": object()}
        service.providers = [prov]
        service._get_default_mandate_selector = MagicMock(return_value=None)
        service._mandate_preference = {}
        service._available_providers = MagicMock(return_value=[prov])
        service._resolve_concrete_model = MagicMock(return_value="m")
        monkeypatch.setattr(service_mod, "messages_for", lambda *a, **k: [])
        seen = {}

        async def _attempt(coro, name, model, **kwargs):
            seen.update(kwargs)
            return LLMResponse(content='{"risk_level": 1, "reasoning": "ok"}')

        service._run_provider_attempt = _attempt
        await service.get_audit_response("ALREADY SHOWN PROSE")
        assert seen["invocation_context"].redact_content is False


class TestAuditHookPassesRedactFlag:
    """ResponseAuditHook.execute forwards redact_content = (mode == strict)."""

    def _hook(self, mode):
        from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook
        agent = MagicMock()
        agent.features = {}
        agent.llm_service.get_audit_response = AsyncMock(
            return_value={"risk_level": 1, "reasoning": "ok", "audited": True}
        )
        return ResponseAuditHook(agent=agent, mode=mode, risk_threshold=3), agent

    @pytest.mark.asyncio
    async def test_strict_passes_redact_true(self):
        from kestrel_sdk.hooks.base import HookEvent, HookInput
        hook, agent = self._hook("strict")
        await hook.execute(HookInput(
            session_id="s", hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text="a sufficiently long assistant response to audit",
        ))
        _args, kwargs = agent.llm_service.get_audit_response.call_args
        assert kwargs.get("redact_content") is True

    @pytest.mark.asyncio
    async def test_warn_passes_redact_false(self):
        from kestrel_sdk.hooks.base import HookEvent, HookInput
        hook, agent = self._hook("warn")
        await hook.execute(HookInput(
            session_id="s", hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text="a sufficiently long assistant response to audit",
        ))
        _args, kwargs = agent.llm_service.get_audit_response.call_args
        assert kwargs.get("redact_content") is False
