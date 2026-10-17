"""Actual SDK HTTP/SQLite ownership counterproofs for independent review39."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import httpx
import httpx2
import openai
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, owned_task_join_ancestors
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, _explicit_control_chain, bind_execution_custody,
    execution_cleanup_debts, is_execution_control_error,
)
from kestrel_sovereign.llm.cancellation import anext_or_cancelled, await_or_cancelled
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_host_context_native_cleanup import _actual_operation


class Body(httpx.AsyncByteStream, httpx2.AsyncByteStream):
    """Real SDK response resource whose native close must be physically joined."""
    def __init__(self, chunk=b""):
        self.chunk = chunk
        self.close_entered = asyncio.Event()
        self.release_close = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        if self.chunk:
            yield self.chunk
        await asyncio.Event().wait()

    async def aclose(self):
        self.close_entered.set()
        await self.release_close.wait()
        self.closed = True


class Fence:
    backend_type = "postgres"
    denial = None

    def require_work(self):
        if self.denial is not None:
            raise self.denial


def openai_chunk():
    return b"data: " + json.dumps({
        "id": "offline", "object": "chat.completion.chunk", "created": 1,
        "model": "offline", "choices": [{"index": 0,
        "delta": {"content": "hello"}, "finish_reason": None}],
    }).encode() + b"\n\n"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("acquisition_delay", [0, 1.1], ids=["immediate", "delayed"])
async def test_native_sdk_handle_survives_post_open_authority_denial(provider, tools, acquisition_delay, monkeypatch):
    import httpx2 as httpx
    body, fence = Body(), Fence()
    native_closes = []
    actual_create = asyncio.create_task

    def observe_close(operation, *args, **kwargs):
        native = _actual_operation(operation)
        task = actual_create(operation, *args, **kwargs)
        if getattr(native, "cr_code", None) is httpx.Response.aclose.__code__:
            native_closes.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", observe_close)
    original = ExecutionAuthorityError("original post-open custody denial")
    async def transport(_request):
        # A real SDK's cold initialization or transport latency is unrelated
        # to whether its acquired response remains owned through retirement.
        await asyncio.sleep(acquisition_delay)
        fence.denial = original
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    client = openai.AsyncOpenAI(api_key="offline-transport-only", http_client=http, max_retries=0)
    adapter = OpenAIAdapter() if provider == "openai" else OpenRouterAdapter()
    method = adapter.get_streaming_response_with_tools if tools else adapter.get_streaming_response
    iterator = method(client=client, model="offline", messages=[{"role": "user", "content": "hello"}])
    async def acquire():
        with bind_execution_custody(fence):
            await anext(iterator)
    task = asyncio.create_task(acquire())
    close_observer = asyncio.create_task(body.close_entered.wait())
    try:
        # Compare lifecycle events, not SDK startup speed. A premature
        # acquisition failure must fail immediately; an absent close remains
        # bounded by the complete test's timeout, including its cleanup.
        done, _ = await asyncio.wait(
            (task, close_observer), return_when=asyncio.FIRST_COMPLETED,
        )
        assert close_observer in done, "post-open denial escaped without entering native close"
        assert not task.done(), "post-open denial escaped before native close joined"
        # OpenRouter forwards through its own accepted iterator Task; OpenAI
        # joins directly. In either case the actual native closer must have a
        # real implementation waiter before the fixture releases its body.
        assert len(native_closes) == 1 and owned_task_join_ancestors(native_closes[0]), (
            "pending native HTTP close has no physical implementation joiner"
        )
        body.release_close.set()
        outcome = await await_owned_task(task)
        assert original in tuple(_explicit_control_chain(outcome.error))
        assert is_execution_control_error(outcome.error)
        assert tuple(_explicit_control_chain(original)) == (original,)
        assert not execution_cleanup_debts(original)
        assert body.closed
    finally:
        body.release_close.set()
        if not close_observer.done():
            close_observer.cancel()
        await await_owned_task(close_observer)
        await await_owned_task(task)
        for native_close in native_closes:
            await await_owned_task(native_close)
        await iterator.aclose()
        await body.aclose()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openrouter", "ollama"])
@pytest.mark.parametrize("tools", [False, True])
async def test_native_http_close_is_joined_through_every_adapter_forwarder(provider, tools):
    native_http = httpx if provider == "ollama" else httpx2
    chunk = openai_chunk() if provider != "ollama" else json.dumps({
        "model": "offline", "message": {"role": "assistant", "content": "hello"}, "done": False,
    }).encode() + b"\n"
    body = Body(chunk)
    async def transport(_request):
        return native_http.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    http = native_http.AsyncClient(base_url="http://offline.invalid", transport=native_http.MockTransport(transport))
    if provider == "ollama":
        import ollama
        from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
        client = ollama.AsyncClient(host="http://offline.invalid")
        await client._client.aclose()
        client._client = http
        adapter = OllamaAdapter()
    else:
        client = openai.AsyncOpenAI(api_key="offline-transport-only", http_client=http, max_retries=0)
        adapter = OpenAIAdapter() if provider == "openai" else OpenRouterAdapter()
    method = adapter.get_streaming_response_with_tools if tools else adapter.get_streaming_response
    iterator = method(client=client, model="offline", messages=[{"role": "user", "content": "hello"}])
    close = None
    try:
        assert await asyncio.wait_for(anext(iterator), timeout=2) == "hello"
        close = asyncio.create_task(iterator.aclose())
        await asyncio.wait_for(body.close_entered.wait(), timeout=1)
        assert not close.done(), "outer adapter reported close while inner HTTP resource remained live"
        body.release_close.set()
        outcome = await await_owned_task(close)
        assert outcome.error is None
        assert body.closed
    finally:
        body.release_close.set()
        if close is not None:
            await await_owned_task(close)
        await iterator.aclose()
        await body.aclose()
        await http.aclose()


@pytest.mark.asyncio
async def test_native_anthropic_acquisition_closes_before_denial_is_reported():
    import httpx2 as httpx
    import anthropic
    from kestrel_sovereign.llm.anthropic_adapter import _anthropic_stream_with_retry
    body, fence = Body(), Fence()
    original = ExecutionAuthorityError("original Anthropic open denial")
    async def transport(_request):
        fence.denial = original
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    client = anthropic.AsyncAnthropic(api_key="offline-transport-only", http_client=http, max_retries=0)
    async def acquire():
        with bind_execution_custody(fence):
            async with _anthropic_stream_with_retry(client, {"model": "offline", "max_tokens": 1,
                "messages": [{"role": "user", "content": "hello"}]}):
                pytest.fail("denied native acquisition was disclosed")
    task = asyncio.create_task(acquire())
    try:
        await asyncio.wait_for(body.close_entered.wait(), timeout=1)
        assert not task.done()
        body.release_close.set()
        outcome = await await_owned_task(task)
        assert original in tuple(_explicit_control_chain(outcome.error))
        assert is_execution_control_error(outcome.error)
        assert tuple(_explicit_control_chain(original)) == (original,)
        assert not execution_cleanup_debts(original)
        assert body.closed
    finally:
        body.release_close.set()
        await await_owned_task(task)
        await body.aclose()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("iterator", [False, True])
@pytest.mark.parametrize("completed", [False, True])
async def test_cancellation_harvests_original_native_child_receipt(tmp_path, kind, iterator, completed):
    agent, db = await native_owner(tmp_path)
    original = receipt(kind)
    entered = asyncio.Event()
    parent = None
    async def operation():
        entered.set()
        if not completed:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
        await db.execute("INSERT INTO effects VALUES(1)")
        if completed:
            parent.cancel()
        raise original
    async def source():
        await operation()
        yield "never"
    async def caller():
        if iterator:
            return await anext_or_cancelled(source(), None)
        return await await_or_cancelled(operation(), None)
    parent = asyncio.create_task(caller())
    try:
        await entered.wait()
        if not completed:
            parent.cancel()
        outcome = await await_owned_task(parent)
        assert outcome.error is original, "caller cancellation replaced the child native receipt"
        with sqlite3.connect(tmp_path / "native.db") as observer:
            assert observer.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 1
    finally:
        await await_owned_task(parent)
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy", ["normal", "ephemeral", "isolated", "anonymous"])
async def test_host_care_ack_fans_out_native_delivery_without_cognition_or_upgrade(tmp_path, privacy):
    from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _close
    from kestrel_sovereign.features.channels.feature import ChannelFeature
    from kestrel_sovereign.features.isolated_runtime import ProxyFeature
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from kestrel_sovereign.signals.sources.channels import (
        DURABLE_TERMINAL_CONSUMER_ID, DURABLE_TERMINAL_MARKER,
        DURABLE_TERMINAL_MARKER_VALUE, DURABLE_COGNITION_CONSUMER_ID,
        DURABLE_COGNITION_MARKER, DURABLE_COGNITION_MARKER_VALUE,
    )
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / "care.db", "did:care")
    from kestrel_sovereign.privacy import get_privacy_preset
    agent.privacy_config = get_privacy_preset(privacy)
    agent.dispatcher = dispatcher
    channel = ChannelFeature(agent)
    from kestrel_sovereign.features.channels.registry import ChannelRegistry
    channel.registry = ChannelRegistry()
    channel._agent_id = channel._authoritative_agent_id = agent.did
    channel._db = None
    cognition_calls = []
    async def cognition(prompt):
        cognition_calls.append(prompt)
        return "unexpected cognition"
    agent.process_input = cognition
    channel._durable_cognition_registration_failed = False
    channel._durable_terminal_ready = True
    channel._durable_cognition_ready = True
    from kestrel_sdk.channels import ChannelConfig
    channel.registry.register(SimpleNamespace(channel_type="telegram",
        config=ChannelConfig(channel_type="telegram", enabled=True, allowed_senders=["555"])))
    proxy = object.__new__(ProxyFeature)
    proxy.agent = agent
    proxy.name = "TelegramFeature"
    proxy._is_telegram_runtime = lambda: True
    proxy._hosted_telegram_startup_attested = True
    proxy._require_event_ingress_work = lambda: None
    proxy._authoritative_inbound_channel_type = lambda: "telegram"
    proxy._channel_feature = lambda: channel
    event = {"channel_type": "telegram", "direction": "inbound", "sender": "555",
        "recipient": "42", "content": "  ok  ", "id": "telegram:v2:bot:42:update:45",
        "metadata": {"dedupe_key": "telegram:v2:bot:42:update:45",
                     "chat_id": "123", "reply_to_message_id": "777",
                     "contact": "alice@example.com"}}
    try:
        for consumer, selector in (
            (DURABLE_TERMINAL_CONSUMER_ID, f"payload.{DURABLE_TERMINAL_MARKER}={DURABLE_TERMINAL_MARKER_VALUE}"),
            (DURABLE_COGNITION_CONSUMER_ID, f"payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}"),
            ("workflows.await:care", "payload.metadata.reply_to_message_id=777"),
        ):
            await dispatcher.register_durable_consumer(DurableConsumerRegistration(
                consumer_id=consumer, source="channel.message", agent_id=agent.did,
                correlation_selector=selector, max_attempts=0,
            ))
        result = await proxy.admit_hosted_telegram_ingress(event, acknowledgement_only=True)
        assert result.durably_admitted
        deliveries = await dispatcher.list_durable_deliveries()
        assert {d.consumer_id for d in deliveries} == {
            DURABLE_TERMINAL_CONSUMER_ID, "workflows.await:care"}
        assert cognition_calls == []
        # A normal retry may not upgrade a retained care-only source identity.
        retry = await proxy.admit_hosted_telegram_ingress(event)
        assert retry.durably_admitted
        assert cognition_calls == []
        assert not await dispatcher.list_durable_deliveries(consumer_id=DURABLE_COGNITION_CONSUMER_ID)
        for tampered in (
            {**event, "content": "ordinary cognition"},
            {**event, "sender": "999"},
            {**event, "metadata": {**event["metadata"], "reply_to_message_id": "778"}},
            {**event, "metadata": {**event["metadata"], "contact": "bob@example.com"}},
        ):
            denied = await proxy.admit_hosted_telegram_ingress(tampered)
            assert not denied.durably_admitted
            assert cognition_calls == []
        from kestrel_sovereign.features.isolated_runtime import ProtocolError
        with pytest.raises(ProtocolError):
            proxy._telegram_terminal_disposition(
                {"_telegram_terminal_disposition": {"kind": "care_acknowledgement"}},
                cursor_owned_protocol=True,
            )
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("door", ["ready", "delivered", "dispatched"])
async def test_restart_wake_consumers_preserve_native_receipts(tmp_path, monkeypatch, kind, door):
    from kestrel_sovereign.features.restart_coordinator import feature as module
    agent, db = await native_owner(tmp_path)
    original = receipt(kind)
    feature = module.RestartCoordinatorFeature(agent)
    feature._db = db
    async def native_write(*args, **kwargs):
        await db.execute("INSERT INTO effects VALUES(1)")
        raise original
    if door == "ready":
        monkeypatch.setattr(feature, "_reap_post_restart_rows", native_write)
        operation = feature.on_agent_ready()
    else:
        monkeypatch.setattr(module, "mark_wake_" + door, native_write)
        operation = getattr(feature, "_mark_wake_" + door)(SimpleNamespace(id="native"))
    try:
        outcome = await await_owned_task(asyncio.create_task(operation))
        assert outcome.error is original
        with sqlite3.connect(tmp_path / "native.db") as observer:
            assert observer.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 1
    finally:
        await retire(agent)
