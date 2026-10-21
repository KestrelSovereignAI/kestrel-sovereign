"""Actual SDK response teardown at the independent review40 boundary."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, execution_cleanup_debts, execution_commit_outcome,
)
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_execution_custody_review39_native import Body, openai_chunk
from tests.unit.test_signals_dispatcher import (
    _action_reg,
    _cognition_reg,
    _signal,
)
from tests.unit.test_signals_dispatcher import (
    db_path as db_path,
)
from tests.unit.test_signals_dispatcher import (
    dispatcher_components as dispatcher_components,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_native_persist_receipt_reaches_both_dispatch_lanes(
    dispatcher_components, monkeypatch, kind,
):
    c = dispatcher_components
    c.registry.register(_action_reg("persist_receipt"))
    original, committed = receipt(kind), []
    persist = c.dispatcher._durable_store.persist_signal
    async def persist_then_receipt(*args, **kwargs):
        value = await persist(*args, **kwargs)
        committed.append(value)
        raise original
    monkeypatch.setattr(c.dispatcher._durable_store, "persist_signal", persist_then_receipt)
    try:
        handle = await c.dispatcher.enqueue_signal(
            _signal("persist_receipt"), source_event_id="native-persist",
        )
        for wait in (handle.wait, handle.wait_for_durable_admission):
            with pytest.raises(BaseException) as caught:
                await wait()
            assert caught.value is original
        assert committed and committed[0].created
    finally:
        await c.dispatcher.shutdown_durable_delivery()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("boundary", ["activation", "repair", "legacy"])
async def test_native_post_commit_paths_retain_strongest_original_control(
    dispatcher_components, monkeypatch, tmp_path, kind, boundary,
):
    from kestrel_sovereign.privacy import get_privacy_preset
    c = dispatcher_components
    original, effects = receipt(kind), []
    source_graph = tuple(_explicit_control_chain(original))
    source_debts = tuple(execution_cleanup_debts(original))
    opening_failure = OSError("ordinary activation failed")
    if boundary == "legacy":
        from kestrel_sovereign.signals.sources.channels import (
            DURABLE_COGNITION_CONSUMER_ID,
        )
        from tests.unit.test_durable_signal_delivery import (
            _channel_dispatcher,
            _channel_signal,
            _close,
        )
        backend, agent, dispatcher = await _channel_dispatcher(tmp_path / "legacy.db", "did:legacy")
        signal = _channel_signal(agent.did, "native-legacy")
        # A real retained pre-cognition row, then exact provider redelivery.
        await dispatcher._durable_store.persist_signal(signal, agent_id=agent.did, source_event_id="native-legacy", retention_days=7)
        upgrade = dispatcher._durable_store.upgrade_legacy_delivery_for_redelivery
        async def upgrade_then_receipt(*args, **kwargs):
            effects.append(await upgrade(*args, **kwargs))
            raise original
        monkeypatch.setattr(dispatcher._durable_store, "upgrade_legacy_delivery_for_redelivery", upgrade_then_receipt)
        try:
            handle = await dispatcher.enqueue_durable_cognition(
                signal, source_event_id="native-legacy", consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            )
            for wait in (handle.wait, handle.wait_for_durable_admission):
                with pytest.raises(BaseException) as caught:
                    await wait()
                assert caught.value is original
            assert effects
        finally:
            await dispatcher.shutdown_durable_delivery()
            await _close(backend, agent)
        return
    c.agent.privacy_config = get_privacy_preset("ephemeral")
    c.registry.register(_action_reg("post_commit"))
    activate, repair = c.dispatcher._activate_transient_reservations, c.dispatcher._repair_post_commit_reservations
    async def activate_then_receipt(*args, **kwargs):
        await activate(*args, **kwargs)
        effects.append(args[0].created)
        raise original if boundary == "activation" else opening_failure
    async def repair_then_receipt(*args, **kwargs):
        await repair(*args, **kwargs)
        raise original
    monkeypatch.setattr(c.dispatcher, "_activate_transient_reservations", activate_then_receipt)
    if boundary == "repair":
        monkeypatch.setattr(c.dispatcher, "_repair_post_commit_reservations", repair_then_receipt)
    try:
        handle = await c.dispatcher.enqueue_signal(_signal("post_commit"), source_event_id="native-post-commit")
        for wait in (handle.wait, handle.wait_for_durable_admission):
            with pytest.raises(BaseException) as caught:
                await wait()
            chain = tuple(_explicit_control_chain(caught.value))
            assert original in chain
            if boundary == "repair":
                assert opening_failure in chain
            assert execution_commit_outcome(caught.value) == "unknown"
            assert tuple(_explicit_control_chain(original)) == source_graph
            assert tuple(execution_cleanup_debts(original)) == source_debts
        assert effects == [True]
    finally:
        await c.dispatcher.shutdown_durable_delivery()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("boundary", ["dispatch", "shutdown"])
async def test_native_outcome_writer_control_is_not_discarded(
    dispatcher_components, monkeypatch, kind, boundary,
):
    from kestrel_sdk.signals import SignalMode, SignalResult, Status
    c = dispatcher_components
    registration = _action_reg("outcome_receipt")
    c.registry.register(registration)
    original, committed = receipt(kind), []
    write = c.dispatcher._write_outcome_log
    async def write_then_receipt(*args, **kwargs):
        await write(*args, **kwargs)
        committed.append(True)
        raise original
    monkeypatch.setattr(c.dispatcher, "_write_outcome_log", write_then_receipt)
    signal = _signal("outcome_receipt")
    try:
        if boundary == "dispatch":
            with pytest.raises(BaseException) as caught:
                await c.dispatcher.dispatch_signal(signal)
        else:
            c.dispatcher._schedule_outcome_log(signal, registration, SignalResult(
                signal_id=signal.id, mode=SignalMode.ACTION, status=Status.OK, duration_ms=0,
            ))
            # Exercise completion-before-shutdown, not just pending gathers.
            tasks = tuple(c.dispatcher._outcome_log_tasks)
            for task in tasks:
                await await_owned_task(task)
            await asyncio.sleep(0)
            with pytest.raises(BaseException) as caught:
                await c.dispatcher.shutdown_durable_delivery()
        assert caught.value is original
        assert committed
    finally:
        monkeypatch.setattr(c.dispatcher, "_write_outcome_log", write)
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert error is original
            # Retained uncertain audit outcome refuses owner release. End the
            # disposable fixture's heartbeat only after every writer joined.
            await c.dispatcher._stop_runtime_owner_heartbeat()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("cancel_parent", [False, True])
async def test_monitor_withdrawal_physically_joins_native_worker_receipt(
    dispatcher_components, tmp_path, monkeypatch, kind, cancel_parent,
):
    from kestrel_sdk.signals import SignalMode
    from kestrel_sovereign import _async_ownership as ownership
    from tests.utils.original_failure import OriginalFailure
    c = dispatcher_components
    template = tmp_path / "monitor.md"
    template.write_text("payload: {payload}")
    c.registry.register(_cognition_reg(template, name="native_monitor"))
    await c.backend.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    entered, cancellation_seen, release = (asyncio.Event() for _ in range(3))
    original = receipt(kind)
    source = OriginalFailure.capture(original)
    withdrawals, task = [], None
    actual_join = ownership.await_owned_task
    async def observe(owner, *args, **kwargs):
        result = await actual_join(owner, *args, **kwargs)
        if asyncio.current_task() is task and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    async def execute(_prompt):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        await c.backend.execute("INSERT INTO effects VALUES(1)")
        raise original
    async def monitor(_signal):
        await entered.wait()
        return "durable source withdrew"
    c.agent.process_input = execute
    c.agent.monitor_cognition_signal_execution = monitor
    task = asyncio.create_task(c.dispatcher.dispatch_signal(
        _signal("native_monitor", mode=SignalMode.COGNITION),
    ))
    try:
        await asyncio.wait_for(cancellation_seen.wait(), 2)
        if cancel_parent:
            task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done(), "withdrawal released a live native worker"
        release.set()
        outcome = await await_owned_task(task)
        source.assert_retained(outcome.error)
        if cancel_parent:
            assert withdrawals and all(value in tuple(_explicit_control_chain(outcome.error)) for value in withdrawals)
        else:
            assert outcome.error is original
        rows = await c.backend.fetch_all("SELECT COUNT(*) AS n FROM effects")
        assert rows[0][0] == 1
    finally:
        release.set()
        await await_owned_task(task)
        await c.dispatcher.shutdown_durable_delivery()


class EndingBody(Body):
    async def __aiter__(self):
        yield self.chunk


@pytest.mark.asyncio
async def test_keyless_terminal_retry_uses_original_channel_sender_not_opaque_label(tmp_path, monkeypatch):
    from tests.unit.test_execution_custody_review39_native import (
        test_host_care_ack_fans_out_native_delivery_without_cognition_or_upgrade,
    )
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    await test_host_care_ack_fans_out_native_delivery_without_cognition_or_upgrade(tmp_path, "normal")


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [False, True])
async def test_ollama_implicit_sdk_http_close_has_one_retained_owner(tools):
    import ollama

    from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
    body = EndingBody(json.dumps({
        "model": "offline", "message": {"role": "assistant", "content": "hello"}, "done": False,
    }).encode() + b"\n")
    http = httpx.AsyncClient(base_url="http://offline.invalid", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "application/x-ndjson"}, stream=body)
    ))
    client, adapter = ollama.AsyncClient(host="http://offline.invalid"), OllamaAdapter()
    original_http = client._client
    client._client = http
    method = adapter.get_streaming_response_with_tools if tools else adapter.get_streaming_response
    iterator = method(client=client, model="offline", messages=[{"role": "user", "content": "hello"}])
    async def consume():
        async for _ in iterator:
            pass
    task = asyncio.create_task(consume())
    try:
        await _interrupt_native_close(task, body)
    finally:
        body.release_close.set()
        await await_owned_task(task)
        await iterator.aclose()
        await body.aclose()
        await http.aclose()
        await original_http.aclose()


@pytest.mark.asyncio
async def test_shared_ollama_http_client_retains_each_original_request_scope():
    import ollama

    from kestrel_sovereign.execution_custody import (
        bind_execution_custody,
        current_execution_custody,
    )
    from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
    from tests.unit.test_execution_custody_review39_native import Fence

    observations, bodies = [], []
    class RecordingBody(EndingBody):
        async def aclose(self):
            observations.append(current_execution_custody())
            await super().aclose()
    def response(_request):
        body = RecordingBody(json.dumps({
            "model": "offline", "message": {"role": "assistant", "content": "hello"}, "done": False,
        }).encode() + b"\n")
        body.release_close.set()
        bodies.append(body)
        return httpx.Response(200, stream=body)
    http = httpx.AsyncClient(base_url="http://offline.invalid", transport=httpx.MockTransport(response))
    client, adapter = ollama.AsyncClient(host="http://offline.invalid"), OllamaAdapter()
    original_http = client._client
    client._client = http
    scopes = []
    try:
        for _ in range(2):
            with bind_execution_custody(Fence()) as scope:
                scopes.append(scope)
                iterator = adapter.get_streaming_response(client=client, model="offline", messages=[])
                try:
                    async for _ in iterator:
                        pass
                finally:
                    await iterator.aclose()
        assert observations == [(scopes[0],), (scopes[1],)]
        assert all(body.closed for body in bodies)
        assert len(http.event_hooks["response"]) == 1
    finally:
        await http.aclose()
        await original_http.aclose()


async def _interrupt_native_close(task, body):
    await asyncio.wait_for(body.close_entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done(), "SDK cancellation escaped with native close still pending"
    body.release_close.set()
    outcome = await await_owned_task(task)
    assert isinstance(outcome.error or outcome.cancellation, asyncio.CancelledError)
    assert body.closed, "HTTPX is_closed was mistaken for physical response retirement"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("tools", [False, True])
async def test_implicit_sdk_http_close_has_one_retained_owner(provider, tools):
    import httpx2 as httpx
    body = EndingBody(openai_chunk())
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    ))
    client = openai.AsyncOpenAI(api_key="offline-transport-only", http_client=http, max_retries=0)
    adapter = OpenAIAdapter() if provider == "openai" else OpenRouterAdapter()
    method = adapter.get_streaming_response_with_tools if tools else adapter.get_streaming_response
    iterator = method(client=client, model="offline", messages=[{"role": "user", "content": "hello"}])
    async def consume():
        async for _ in iterator:
            pass
    task = asyncio.create_task(consume())
    try:
        await _interrupt_native_close(task, body)
    finally:
        body.release_close.set()
        await await_owned_task(task)
        await iterator.aclose()
        await body.aclose()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [False, True])
async def test_actual_codex_turn_releases_a_cancelled_successful_lock_acquisition(replacement):
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    adapter = CodexAdapter()
    parent = None
    class CancelAfterAcquiring(asyncio.Lock):
        async def acquire(self):
            acquired = await super().acquire()
            parent.cancel()
            return acquired
    abandoned = CancelAfterAcquiring()
    first = asyncio.Lock() if replacement else abandoned
    adapter._thread_locks = {"old": first, "new": abandoned}
    adapter._session_threads = {"session": ("new" if replacement else "old", None)}
    async def started():
        pass
    adapter._app_server = lambda: SimpleNamespace(ensure_started=started)
    async def settings(*args):
        return None
    adapter._resolve_codex_turn_settings = settings
    threads = iter([("old", False), ("new", False)])
    async def ensure(*args, **kwargs):
        return next(threads)
    adapter._ensure_thread = ensure
    async def consume():
        async for _ in adapter._run_turn(
            "offline", [{"role": "user", "content": "hello"}], None, "session", None,
        ):
            pytest.fail("cancelled lock acquisition reached app-server dispatch")
    parent = asyncio.create_task(consume())
    outcome = await await_owned_task(parent)
    assert isinstance(outcome.error or outcome.cancellation, asyncio.CancelledError)
    assert not abandoned.locked(), "cancelled acquired lock stranded the session"
    assert not first.locked()
    await asyncio.wait_for(abandoned.acquire(), 1)
    abandoned.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("final_message", [False, True])
async def test_anthropic_implicit_sdk_close_is_physically_joined(final_message):
    import httpx2 as httpx
    import anthropic

    from kestrel_sovereign.llm.anthropic_adapter import (
        _anthropic_final_message,
        _anthropic_stream_with_retry,
    )
    events = [
        {"type": "message_start", "message": {
            "id": "native", "type": "message", "role": "assistant", "content": [],
            "model": "offline", "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }},
        {"type": "message_stop"},
    ]
    chunk = b"".join(
        ("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode()
        for event in events
    )
    body = EndingBody(chunk)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    ))
    client = anthropic.AsyncAnthropic(api_key="offline-transport-only", http_client=http, max_retries=0)
    params = {"model": "offline", "max_tokens": 1, "messages": [{"role": "user", "content": "hello"}]}
    async def consume():
        if final_message:
            return await _anthropic_final_message(client, params)
        async with _anthropic_stream_with_retry(client, params) as stream:
            async for _ in stream:
                pass
    task = asyncio.create_task(consume())
    try:
        await _interrupt_native_close(task, body)
    finally:
        body.release_close.set()
        await await_owned_task(task)
        await body.aclose()
        await client.close()
