"""#3576: a generation the gate rejects is recorded as a failed attempt.

A response that stops at an output cap, for a reason the generation gate does
not know, or with no stop reason at all is never an answer (#3552). It used to
be finalized as a successful attempt before the gate judged it, so it showed
up as a success in ``get_llm_stats()`` and in Prometheus while its turn failed.

Each case drives the real ``LLMService`` route walk with one scripted route
whose attempt ends one way and reports its token usage, and reads what the
attempt left in a real observability store, the ``model_usage`` sink, the
Prometheus series and the metering callback. A rejected attempt is
``success=False`` and keeps the usage its provider reported; its provider
billed that usage, so it is still metered, as it was before. A natural end is
the control: the same harness records it as a success.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.a2a.stores import ObservabilityStore
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.generation_gate import (
    NoCompletionEvidenceError,
    judge_generation,
)
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.llm.output_ceiling import (
    IncompleteGenerationError,
    attach_stop_reason,
)
from kestrel_sovereign.llm.streaming import StreamingMixin
from kestrel_sovereign.storage.db import SQLiteBackend

ROUTE = "ollama:local"
MODEL = "kestrel-gate-telemetry:1b"
TEXT = "PARTIAL ANSWER"
INPUT_TOKENS = 321
OUTPUT_TOKENS = 64
TOOLS = [{"name": "noop", "description": "does nothing",
          "parameters": {"type": "object"}}]
#: A billable caller: the metering callback bills only an attributed call.
CONTEXT = LLMInvocationContext(companion_id="companion-1", user_id="user-1")

#: How an attempt ended: the stop reason its adapter attached (``None`` for
#: none), and whether the gate accepts it.
ENDINGS = {
    "cap": ("length", False),
    "unknown": ("a-reason-nobody-has-invented-yet", False),
    "missing": (None, False),
    "natural": ("stop", True),
}

#: Every place a route attempt of a turn's answer is finalized: the
#: non-streaming route walk, a streamed route, and a streamed walk's route
#: that has no usage stream (its non-streaming fallback).
ENTRIES = {
    "generate_with_messages": "generate_with_messages",
    "stream_with_tool_detection": "stream_with_tool_detection",
    "stream_fallback": "stream_with_tool_detection.fallback",
}


def _response(stop_reason: Optional[str]) -> LLMResponse:
    return attach_stop_reason(
        LLMResponse(
            content=TEXT,
            input_tokens=INPUT_TOKENS,
            output_tokens=OUTPUT_TOKENS,
            total_tokens=INPUT_TOKENS + OUTPUT_TOKENS,
        ),
        stop_reason,
    )


class _Route:
    """A route without a usage stream: one non-streaming attempt."""

    def __init__(self, stop_reason: Optional[str]) -> None:
        self.stop_reason = stop_reason
        self.calls = 0

    async def get_response(self, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return _response(self.stop_reason)


class _StreamedRoute(_Route):
    """A route whose usage stream ends with its terminal response."""

    async def get_streaming_response_with_tools(self, **kwargs: Any):
        self.calls += 1
        yield TEXT
        yield _response(self.stop_reason)


class _FlushedUsageRoute(_Route):
    """A usage stream that ends with no terminal response, after flushing
    the usage it had consumed into the service's sink."""

    supports_partial_usage_flush = True

    async def get_streaming_response_with_tools(self, *, usage_sink, **kwargs: Any):
        self.calls += 1
        usage_sink["input_tokens"] = INPUT_TOKENS
        usage_sink["output_tokens"] = OUTPUT_TOKENS
        yield TEXT


class _Metric:
    """Records what the Prometheus series were given."""

    def __init__(self, sink: List[tuple]) -> None:
        self._sink = sink

    def labels(self, **labels: str):
        sink = self._sink

        class _Child:
            def inc(self, value: float = 1) -> None:
                sink.append((labels, value))

            def observe(self, value: float) -> None:
                sink.append((labels, value))

        return _Child()


@pytest.fixture
def prometheus(monkeypatch) -> Dict[str, List[tuple]]:
    series: Dict[str, List[tuple]] = {"calls": [], "tokens": []}
    monkeypatch.setattr("kestrel_sdk.metrics.PROMETHEUS_AVAILABLE", True)
    monkeypatch.setattr("kestrel_sdk.metrics.LLM_CALLS", _Metric(series["calls"]))
    monkeypatch.setattr("kestrel_sdk.metrics.LLM_DURATION", _Metric([]))
    monkeypatch.setattr("kestrel_sdk.metrics.LLM_TOKENS", _Metric(series["tokens"]))
    return series


@pytest.fixture
async def store(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "observability.db"))
    await backend.connect()
    observability = ObservabilityStore(backend)
    await observability.initialize()
    try:
        yield observability
    finally:
        await observability.close()


@pytest.fixture
def service(monkeypatch, store):
    from kestrel_sovereign.llm.service import LLMService

    llm = LLMService()
    monkeypatch.setattr(llm, "_ensure_models_discovered", AsyncMock())
    monkeypatch.setattr(llm, "_mandate_preference", {})
    monkeypatch.setattr(llm, "config", {"route_priority": [ROUTE]})
    monkeypatch.setattr(llm, "_record_model_usage", AsyncMock())
    llm._observability_store = store
    llm.metered = []

    async def meter(
        *, companion_id, user_id, provider, model, prompt_tokens,
        completion_tokens,
    ) -> None:
        llm.metered.append({
            "companion_id": companion_id, "user_id": user_id,
            "provider": provider, "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    llm.set_metering_callback(meter)

    def install(adapter: _Route) -> None:
        monkeypatch.setattr(llm, "providers", [{
            "name": ROUTE, "vendor": "ollama", "route": "local",
            "client": AsyncMock(), "adapter": adapter, "model": MODEL,
            "is_cloud": False, "is_local": True, "base_url": None,
            "selection_hints": [],
        }])

    llm.install_route = install
    return llm


async def _call(llm, entry: str) -> None:
    """``entry``'s route walk, to its answer or its failure."""
    messages = [{"role": "user", "content": "where do I work?"}]
    if entry == "generate_with_messages":
        await llm.generate_with_messages(
            messages=messages, tools=TOOLS, invocation_context=CONTEXT,
        )
        return
    async for _ in llm.stream_with_tool_detection(
        messages=messages, tools=TOOLS, invocation_context=CONTEXT,
    ):
        pass


def _route_for(entry: str, stop_reason: Optional[str]) -> _Route:
    if entry == "stream_with_tool_detection":
        return _StreamedRoute(stop_reason)
    return _Route(stop_reason)


async def _only_call(store):
    calls = await store.query_llm_calls()
    assert len(calls) == 1, [(call.provider, call.success) for call in calls]
    return calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", sorted(ENDINGS))
@pytest.mark.parametrize("entry", sorted(ENTRIES))
async def test_an_attempt_is_recorded_with_the_gates_verdict_and_its_usage(
    service, store, prometheus, entry, ending,
):
    stop_reason, accepted = ENDINGS[ending]
    route = _route_for(entry, stop_reason)
    service.install_route(route)

    if accepted:
        await _call(service, entry)
    else:
        with pytest.raises(Exception):
            await _call(service, entry)
    assert route.calls == 1

    call = await _only_call(store)
    assert call.provider == ROUTE and call.model == MODEL
    assert call.metadata["path"] == ENTRIES[entry]
    assert call.success is accepted
    verdict = judge_generation(_response(stop_reason), provider=ROUTE, model=MODEL)
    assert call.error_message == (None if accepted else str(verdict))
    assert call.metadata.get("stop_reason") == stop_reason

    # The provider's reported usage is kept, whatever the verdict.
    assert (call.input_tokens, call.output_tokens) == (INPUT_TOKENS, OUTPUT_TOKENS)
    service._record_model_usage.assert_awaited_once()
    args, kwargs = service._record_model_usage.await_args
    assert args == (MODEL, ROUTE)
    assert kwargs["tokens"] == INPUT_TOKENS + OUTPUT_TOKENS

    stats = await store.get_llm_stats()
    assert stats["total_calls"] == 1
    assert stats["success_rate"] == (100 if accepted else 0)
    assert stats["total_input_tokens"] == INPUT_TOKENS
    assert stats["total_output_tokens"] == OUTPUT_TOKENS

    assert prometheus["calls"] == [
        ({"provider": ROUTE, "model": MODEL, "success": str(accepted)}, 1)
    ]
    assert prometheus["tokens"] == [
        ({"model": MODEL, "direction": "input"}, INPUT_TOKENS),
        ({"model": MODEL, "direction": "output"}, OUTPUT_TOKENS),
    ]
    _assert_billed_once(service)


def _assert_billed_once(llm) -> None:
    """The provider billed the attempt's usage, whatever the gate decided."""
    assert llm.metered == [{
        "companion_id": CONTEXT.companion_id,
        "user_id": CONTEXT.user_id,
        "provider": ROUTE,
        "model": MODEL,
        "prompt_tokens": INPUT_TOKENS,
        "completion_tokens": OUTPUT_TOKENS,
    }]


@pytest.mark.asyncio
async def test_a_stream_without_a_terminal_response_keeps_its_flushed_usage(
    service, store, prometheus,
):
    """No terminal response is no stop reason: a failed attempt, which still
    records the usage its adapter flushed before the stream ended."""
    route = _FlushedUsageRoute(None)
    service.install_route(route)

    with pytest.raises(Exception):
        await _call(service, "stream_with_tool_detection")

    call = await _only_call(store)
    assert call.success is False
    assert call.error_message == str(
        NoCompletionEvidenceError(provider=ROUTE, model=MODEL)
    )
    assert call.metadata["partial_abort"] is True
    assert (call.input_tokens, call.output_tokens) == (INPUT_TOKENS, OUTPUT_TOKENS)
    service._record_model_usage.assert_awaited_once()
    assert (
        service._record_model_usage.await_args.kwargs["tokens"]
        == INPUT_TOKENS + OUTPUT_TOKENS
    )
    assert prometheus["calls"] == [
        ({"provider": ROUTE, "model": MODEL, "success": "False"}, 1)
    ]
    _assert_billed_once(service)


class _StreamingOnlyHost(StreamingMixin):
    """A ``StreamingMixin`` host with no service attempt runner."""

    def __init__(self) -> None:
        self._track_model_usage = AsyncMock()
        self._log_llm_call = AsyncMock()

    def _stamp_response_identity(self, response, *, model, provider):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", sorted(ENDINGS))
async def test_a_fallback_attempt_without_the_service_runner_records_the_verdict(
    ending,
):
    stop_reason, accepted = ENDINGS[ending]
    host = _StreamingOnlyHost()
    route = _Route(stop_reason)

    async def attempt():
        return await host._run_stream_fallback_attempt(
            route.get_response(),
            provider_name=ROUTE,
            model=MODEL,
            path=ENTRIES["stream_fallback"],
            invocation_context=None,
            require_completion=True,
        )

    if accepted:
        assert (await attempt()).content == TEXT
    else:
        with pytest.raises(IncompleteGenerationError):
            await attempt()

    host._log_llm_call.assert_awaited_once()
    logged = host._log_llm_call.await_args.kwargs
    assert logged["success"] is accepted
    # Billed either way. The override is passed only where it differs from
    # ``success``, so a host logger without the parameter keeps recording.
    assert logged.get("meter_usage", logged["success"]) is True
    assert ("meter_usage" in logged) is not accepted
    verdict = judge_generation(_response(stop_reason), provider=ROUTE, model=MODEL)
    assert logged["error_message"] == (None if accepted else str(verdict))
    assert (logged["input_tokens"], logged["output_tokens"]) == (
        INPUT_TOKENS, OUTPUT_TOKENS,
    )
    host._track_model_usage.assert_awaited_once()
    assert (
        host._track_model_usage.await_args.kwargs["tokens"]
        == INPUT_TOKENS + OUTPUT_TOKENS
    )
