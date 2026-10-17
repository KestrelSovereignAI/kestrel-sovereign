"""Offline native SDK and retained dispatcher receipt regressions for review41."""

import asyncio
import json

import httpx
import openai
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    execution_commit_outcome,
    execution_terminal_error,
    is_execution_control_error,
)
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_execution_custody_review40_native import EndingBody
from tests.unit.test_signals_dispatcher import _action_reg, _signal
from tests.unit.test_signals_dispatcher import db_path as db_path
from tests.unit.test_signals_dispatcher import (
    dispatcher_components as dispatcher_components,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openrouter", "ollama"])
@pytest.mark.parametrize("operation", ["chat", "embed"])
@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.parametrize("kind", [None, "unknown", "cancel_unknown", "group"])
async def test_nonstream_sdk_implicit_body_close_is_physically_joined(provider, operation, status, kind, preexisting_observer=False):
    import httpx2

    native_http = httpx if provider == "ollama" else httpx2
    original = receipt(kind) if kind is not None else None
    class ReceiptBody(EndingBody):
        raised = False
        async def aclose(self):
            await super().aclose()
            if original is not None and not self.raised:
                self.raised = True
                raise original
    if operation == "embed":
        payload = {"embeddings": [[0.1, 0.2]]} if provider == "ollama" else {
            "object": "list", "model": "offline", "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            ], "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    else:
        payload = {"model": "offline", "message": {"role": "assistant", "content": "ok"}, "done": True} if provider == "ollama" else {
            "id": "offline", "object": "chat.completion", "created": 0, "model": "offline",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }
    if status != 200:
        payload = {"error": "offline server error"}
    body = ReceiptBody(json.dumps(payload).encode())
    requests = []
    def respond(request):
        requests.append(request)
        return native_http.Response(status, headers={"content-type": "application/json"}, stream=body)
    http = native_http.AsyncClient(base_url="http://127.0.0.1:11434", transport=native_http.MockTransport(respond))
    if preexisting_observer:
        async def observe(response):
            await response.aread()
        http.event_hooks["response"].append(observe)
    original_http = None
    if provider == "ollama":
        import ollama
        client = ollama.AsyncClient()
        original_http, client._client = client._client, http
        adapter = OllamaAdapter()
    else:
        client = openai.AsyncOpenAI(api_key="offline-transport-only", http_client=http, max_retries=0)
        adapter = OpenAIAdapter() if provider == "openai" else OpenRouterAdapter()
    method = adapter.get_response if operation == "chat" else adapter.aembed
    arguments = {"model": "offline", "messages": [{"role": "user", "content": "hello"}]} if operation == "chat" else {"text": "hello", "model": "offline"}
    task = asyncio.create_task(method(client=client, **arguments))
    try:
        await asyncio.wait_for(body.close_entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done(), "SDK cancellation escaped with native close still pending"
        body.release_close.set()
        outcome = await await_owned_task(task)
        terminal = execution_terminal_error(outcome.error, outcome.cancellation)
        if original is not None:
            assert terminal is not None and is_execution_control_error(terminal)
            assert execution_commit_outcome(terminal) == "unknown"
            # OpenAI may wrap a native error in APIConnectionError. Its
            # explicit cause must preserve the original receipt; no provider
            # fallback/retry may convert that carrier to success.
            carriers, seen = [terminal], set()
            while carriers:
                current = carriers.pop()
                if id(current) in seen:
                    continue
                seen.add(id(current))
                if current.__cause__ is not None:
                    carriers.append(current.__cause__)
                if isinstance(current, BaseExceptionGroup):
                    carriers.extend(current.exceptions)
            assert id(original) in seen, "SDK cleanup lost the original irreversible receipt"
        else:
            assert isinstance(terminal, asyncio.CancelledError)
        assert body.closed
        assert len(requests) == 1, "uncertain native cleanup must not retry provider effects"
    finally:
        body.release_close.set()
        await await_owned_task(task)
        await body.aclose()
        await http.aclose()
        if original_http is not None:
            await original_http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openrouter", "ollama"])
async def test_response_close_is_owned_before_preexisting_body_observer(provider):
    await test_nonstream_sdk_implicit_body_close_is_physically_joined(
        provider, "chat", 200, None, preexisting_observer=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("reader", ["dispatch", "drain_all", "drain_signal"])
async def test_completed_outcome_receipt_readers_retain_bounded_tracebacks(
    dispatcher_components, monkeypatch, kind, reader,
):
    from kestrel_sdk.signals import SignalMode, SignalResult, Status
    c = dispatcher_components
    registration = _action_reg("bounded_receipt")
    c.registry.register(registration)
    original, effects = receipt(kind), []
    write = c.dispatcher._write_outcome_log
    async def write_then_receipt(*args, **kwargs):
        await write(*args, **kwargs)
        effects.append(True)
        raise original
    monkeypatch.setattr(c.dispatcher, "_write_outcome_log", write_then_receipt)
    signal = _signal("bounded_receipt")
    c.dispatcher._schedule_outcome_log(signal, registration, SignalResult(
        signal_id=signal.id, mode=SignalMode.ACTION, status=Status.OK, duration_ms=0,
    ))
    for task in tuple(c.dispatcher._outcome_log_tasks):
        await await_owned_task(task)
    await asyncio.sleep(0)
    assert c.dispatcher._outcome_log_control_error is original
    assert effects == [True]
    lengths = []
    try:
        for _ in range(25):
            with pytest.raises(BaseException) as caught:
                if reader == "dispatch":
                    await c.dispatcher.dispatch_signal(signal)
                elif reader == "drain_all":
                    await c.dispatcher._drain_outcome_log_tasks()
                else:
                    await c.dispatcher._drain_outcome_logs_for_signal(signal.id)
            assert caught.value is original
            traceback, length = original.__traceback__, 0
            while traceback is not None:
                length += 1
                traceback = traceback.tb_next
            lengths.append(length)
        assert len(set(lengths)) == 1, f"Retained receipt grew with each heartbeat/retry: {lengths}"
        assert effects == [True], "latched uncertainty must not redispatch effects"
    finally:
        monkeypatch.setattr(c.dispatcher, "_write_outcome_log", write)
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert error is original
            await c.dispatcher._stop_runtime_owner_heartbeat()
