"""Offline production consumers for review 35; no provider network/billing."""
import asyncio

import httpx2 as httpx
import openai
import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_runtime, execution_terminal_error,
)
from kestrel_sovereign.features.spawn.feature import SpawnFeature
from tests.unit.test_execution_custody_review30 import service_owner
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_inference_lease_coordinator import _Provider, _coordinator, _request
from tests.unit.test_private_inference_routing import _lease


@pytest.mark.asyncio
async def test_actual_remote_client_remains_owned_until_request_and_close_join(monkeypatch):
    from kestrel_sovereign.llm import service as module

    service = service_owner()
    entered, release = asyncio.Event(), asyncio.Event()

    async def handle(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"id": "local", "object": "chat.completion",
            "created": 1, "model": "local", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = openai.AsyncOpenAI(api_key="offline-test-only", http_client=http,
                                base_url="https://offline.invalid/v1", max_retries=0)
    lease = _lease()
    service._remote_lease, service._remote_client = lease, client
    service._remote_inflight = 0
    service._remote_accepting = True
    service._remote_capabilities = frozenset({"chat"})
    service._remote_route_epoch = 0
    service._remote_adapter = object()
    from kestrel_sdk.llm.types import BackendType
    service._backend = BackendType.REMOTE_GPU
    service._default_backend = BackendType.CLOUD
    service._remote_first_allowed = lambda _override: True

    async def touch(_lease_id):
        return lease

    service._remote_touch_lease = touch
    monkeypatch.setattr(module, "CLIENT_CLOSE_TIMEOUT", 0.01)

    async def native_request():
        async with service._remote_route_attempt(force_local_only=False, model_override=None) as route:
            await route.client.chat.completions.create(
                model="local", messages=[{"role": "user", "content": "offline"}],
            )

    request = asyncio.create_task(native_request())
    closer = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        try:
            await service.close()
        except RuntimeError:
            pass  # Bounded close must expose debt, not pretend it completed.
        closer = asyncio.create_task(service.wait_for_close_completion())
        done, _ = await asyncio.wait({closer}, timeout=0.15)
        assert not done, "physical close join forgot the active remote client"
        assert not http.is_closed
        release.set()
        await request
        await closer
        assert http.is_closed and service._remote_client is None
    finally:
        release.set()
        await await_owned_task(request)
        if closer is not None:
            await await_owned_task(closer)
        await service.wait_for_close_completion()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_kind", ["runtime", "ambient"])
async def test_lease_renewal_rechecks_original_authority_after_coordinator_lock(scope_kind):
    from kestrel_sovereign.execution_custody import bind_execution_custody_snapshot

    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    coordinator, _ = _coordinator({"runpod": provider})
    pending = await coordinator.acquire(_request())
    lease = await coordinator.status(pending.lease_id)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original"))
    if scope_kind == "runtime":
        coordinator._llm_service._execution_custody = scope

    async def renew():
        if scope_kind == "ambient":
            with bind_execution_custody_snapshot((scope,)):
                return await coordinator.touch(lease.lease_id)
        with bind_execution_runtime(coordinator._llm_service):
            return await coordinator.touch(lease.lease_id)

    await coordinator._lock.acquire()
    waiter = asyncio.create_task(renew())
    try:
        await asyncio.sleep(0)
        scope.revoke("original renewal authority lost")
        coordinator._lock.release()
        outcome = await await_owned_task(waiter)
        assert provider.touch_calls == 0, "provider capacity renewed after original authority loss"
        assert isinstance(outcome.error, ExecutionAuthorityError)
    finally:
        if coordinator._lock.locked():
            coordinator._lock.release()
        await await_owned_task(waiter)


def traceback_size(error):
    size, tb = 0, error.__traceback__
    while tb is not None:
        size += 1
        tb = tb.tb_next
    return size


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_actual_remote_native_close_is_not_recreated_after_uncertain_receipt(monkeypatch, kind):
    service = service_owner()
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    client = openai.AsyncOpenAI(api_key="offline-test-only", http_client=http,
                                base_url="https://offline.invalid/v1", max_retries=0)
    original_close = client.close
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)
    calls = 0

    async def native_close():
        nonlocal calls
        calls += 1
        await original_close()
        raise failure

    monkeypatch.setattr(client, "close", native_close)
    try:
        for _ in range(2):
            outcome = await await_owned_task(asyncio.create_task(
                service._close_remote_client(client, lease_id="local-only"),
            ))
            source.assert_retained(outcome.error, resource=client)
        assert calls == 1, "uncertain native client close was recreated"
        assert http.is_closed
        outcome = await await_owned_task(asyncio.create_task(service.close()))
        source.assert_retained(outcome.error, resource=client)
    finally:
        await original_close()


@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
def test_repeated_canonical_retained_receipt_has_bounded_traceback(kind):
    failure = receipt(kind)
    for _ in range(100):
        try:
            raise execution_terminal_error(failure)
        except BaseException as caught:
            assert caught is failure
    assert traceback_size(failure) <= 4, "retained receipt accumulates every consumer frame"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_repeated_delegated_result_read_preserves_bounded_original_receipt(kind):
    feature = SpawnFeature.__new__(SpawnFeature)
    feature._child_tasks = {}
    failure = receipt(kind)
    feature._child_results = {"child": {"success": False, "control": failure}}
    for _ in range(100):
        try:
            await feature.get_child_result("child")
        except BaseException as caught:
            assert caught is failure
    assert traceback_size(failure) <= 6, "delegated receipt retains unbounded caller frames"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_actual_failed_agent_initialization_receipt_survives_native_retirement(
    monkeypatch, tmp_path, kind,
):
    from tests.unit.test_execution_custody_review22 import stub_single_startup
    from tests.unit.test_execution_custody_review29 import native_owner

    server, app = stub_single_startup(monkeypatch, tmp_path, backend="sqlite")
    agent, db = await native_owner(tmp_path)
    failure = receipt(kind)
    release, entered = asyncio.Event(), asyncio.Event()

    async def failed_initialize():
        await db.execute("INSERT INTO effects VALUES (1)")
        raise failure

    async def delayed_shutdown():
        entered.set()
        await release.wait()
        await agent.storage.close()

    monkeypatch.setattr(agent, "initialize", failed_initialize)
    monkeypatch.setattr(agent, "shutdown", delayed_shutdown)
    monkeypatch.setattr(server, "LLMService", lambda: agent.llm_service)
    monkeypatch.setattr(server, "KestrelAgent", lambda **_kwargs: agent)
    closer = None
    try:
        try:
            async with server._lifespan_startup(app):
                pass
        except BaseException as caught:
            assert caught is failure
        assert app.state.startup_cleanup_agent is agent
        closer = asyncio.create_task(server._shutdown_server_agents(app))
        await asyncio.wait_for(entered.wait(), 5)
        assert db.backend._connection is not None
        release.set()
        outcome = await await_owned_task(closer)
        assert db.backend._connection is None
        assert outcome.error is failure, "failed birth retained Agent but discarded native initialization receipt"
        repeated = asyncio.create_task(server._shutdown_server_agents(app))
        assert (await await_owned_task(repeated)).error is failure
    finally:
        release.set()
        if closer is not None:
            await await_owned_task(closer)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_actual_batch_collects_original_cancelled_initializer_receipt(monkeypatch, tmp_path, kind):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
    from tests.unit.test_execution_custody_review29 import native_owner

    agent, db = await native_owner(tmp_path)
    manager = AgentManager(base_data_dir=tmp_path)
    failure = receipt(kind)

    async def initialize(*_args, **_kwargs):
        await db.execute("INSERT INTO effects VALUES (1)")
        await agent.storage.close()
        raise failure

    monkeypatch.setattr(manager, "_initialize_agent", initialize)
    config = MultiAgentConfig(agents={
        "Alpha": LocalAgentConfig(data_dir="alpha", port=8801),
    })
    try:
        outcome = await await_owned_task(asyncio.create_task(manager.load_from_config(config)))
        assert db.backend._connection is None and not manager.list_agents()
        assert outcome.error is failure, "batch gather replaced cancelled initializer's original receipt"
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_actual_batch_cancel_preserves_unpublished_native_cleanup_receipt(
    monkeypatch, tmp_path, kind,
):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager, _BATCH_STAGE_CALLBACK
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
    from tests.unit.test_execution_custody_review29 import native_owner

    path = tmp_path / "native"
    path.mkdir()
    agent, db = await native_owner(path)
    manager = AgentManager(base_data_dir=tmp_path)
    second_entered, release, closing = (asyncio.Event() for _ in range(3))
    failure = receipt(kind)
    original = OriginalFailure.capture(failure)

    async def initialize(name, *_args, **_kwargs):
        await _BATCH_STAGE_CALLBACK.get()(name, agent if name == "Alpha" else None)
        if name == "Alpha":
            return agent
        second_entered.set()
        await asyncio.Event().wait()

    async def shutdown_receipt():
        raise failure

    async def native_tail():
        closing.set()
        await release.wait()
        await db.execute("INSERT INTO effects VALUES (1)")
        await agent.storage.close()

    continuation = asyncio.create_task(native_tail())
    agent._durable_shutdown_continuation = continuation
    monkeypatch.setattr(agent, "shutdown", shutdown_receipt)
    monkeypatch.setattr(manager, "_initialize_agent", initialize)
    config = MultiAgentConfig(agents={
        "Alpha": LocalAgentConfig(data_dir="alpha", port=8801),
        "Beta": LocalAgentConfig(data_dir="beta", port=8802),
    })
    batch = asyncio.create_task(manager.load_from_config(config))
    try:
        await asyncio.wait_for(second_entered.wait(), 5)
        await asyncio.sleep(0)
        batch.cancel()
        await asyncio.wait_for(closing.wait(), 5)
        done, _ = await asyncio.wait({batch}, timeout=0.1)
        assert not done and db.backend._connection is not None
        release.set()
        outcome = await await_owned_task(batch)
        assert db.backend._connection is None and not manager.list_agents()
        original.assert_retained(outcome.error)
    finally:
        release.set()
        await await_owned_task(batch)
        await await_owned_task(continuation)
        await db.close()
