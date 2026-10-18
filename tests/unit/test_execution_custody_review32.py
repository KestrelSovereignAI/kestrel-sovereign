"""Review32 production-consumer regressions; finite, provider-free owners."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.utils.original_failure import OriginalFailure
from tests.utils.receipt_delivery import ReceiptDelivery

from kestrel_sovereign._async_ownership import await_owned_task, owned_task_error
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionCustody,
    bind_execution_runtime, require_execution_work, _explicit_control_chain, execution_commit_outcome,
)
from tests.unit.test_execution_custody import Authority
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review30 import (
    assert_recorded_preference_rows,
    service_owner,
)


def receipt(kind):
    error = ExecutionCommitOutcomeError("unknown")
    if kind == "cancel_unknown":
        wrapper = asyncio.CancelledError("original writer cancellation")
        wrapper.__cause__ = error
        return wrapper
    if kind == "group":
        return BaseExceptionGroup("original native receipt", [asyncio.CancelledError(), error])
    if kind == "authority":
        return ExecutionAuthorityError("original child authority denied")
    if kind == "ordinary":
        return OSError("ordinary writer failure")
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["model", "embedding", "overrides"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group", "ordinary"])
async def test_actual_agent_retains_storage_until_native_preference_writer_settles(
    tmp_path, monkeypatch, route, kind,
):
    from kestrel_sovereign.llm import service as service_module

    agent, db = await native_owner(tmp_path)
    service = service_owner()
    agent.llm_service = service
    agent.privacy_agent = SimpleNamespace(privacy_config=SimpleNamespace(allows_cloud_llm=lambda: False))
    await db.execute("CREATE TABLE IF NOT EXISTS agent_metadata(agent_id TEXT, key TEXT, value TEXT, updated_at TEXT, PRIMARY KEY(agent_id,key))")
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.01)
    committed, release, storage_closed = (asyncio.Event() for _ in range(3))
    original_execute, original_close = db.execute, agent.storage.close
    error = receipt(kind)
    writes, closes = [], []
    writer = shutdown = waiter = None
    withdrawals = []
    actual_wait = service_module.await_owned_task

    async def observed_wait(task, *args, **kwargs):
        outcome = await actual_wait(task, *args, **kwargs)
        if task is writer and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    monkeypatch.setattr(service_module, "await_owned_task", observed_wait)

    async def native_then_late_receipt(*args, **kwargs):
        result = await original_execute(*args, **kwargs)
        writes.append(result)
        committed.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        raise error

    async def close_storage():
        assert writer.done(), "agent retired the writer's borrowed storage early"
        closes.append(True)
        await original_close()
        storage_closed.set()

    monkeypatch.setattr(db, "execute", native_then_late_receipt)
    monkeypatch.setattr(agent.storage, "close", close_storage)
    if route == "model":
        service.set_preference_persistence_callback(agent._persist_model_preference)
        writer = service._schedule_preference_persistence("model", "local", "unit")
    elif route == "embedding":
        service.set_embedding_route_persistence_callback(agent._persist_embedding_route)
        writer = service._schedule_embedding_route_persistence(None)
    else:
        service.set_route_embedding_model_persistence_callback(agent._persist_route_embedding_models)
        writer = service._schedule_route_embedding_model_persistence()
    try:
        await asyncio.wait_for(committed.wait(), 5)
        assert_recorded_preference_rows(
            await db.fetchall("SELECT key, value FROM agent_metadata"), route,
        )
        shutdown = asyncio.create_task(agent.shutdown())
        await asyncio.wait_for(asyncio.shield(shutdown), 2)
        assert not storage_closed.is_set()
        assert not writer.done()
        assert agent._durable_shutdown_continuation is not None
        finite_close = service._llm_close_tasks["preference-writers"]
        waiter = asyncio.create_task(agent.wait_for_shutdown_completion())
        await asyncio.sleep(0.01)
        assert not waiter.done()
        release.set()
        if kind == "ordinary":
            await asyncio.wait_for(waiter, 5)
        else:
            with pytest.raises(BaseException) as caught:
                await asyncio.wait_for(waiter, 5)
            chain = tuple(_explicit_control_chain(caught.value))
            retained = tuple(_explicit_control_chain(agent._shutdown_control))
            assert error in chain and error in retained
            assert execution_commit_outcome(caught.value) == "unknown"
            assert execution_commit_outcome(agent._shutdown_control) == "unknown"
            finite_chain = tuple(_explicit_control_chain(owned_task_error(finite_close)))
            assert withdrawals and all(item in finite_chain for item in withdrawals)
            assert all(item not in chain and item not in retained for item in withdrawals)
            assert service._preference_persistence_control is error
            assert all(item not in tuple(_explicit_control_chain(error)) for item in withdrawals)
            for _ in range(3):
                with pytest.raises(BaseException) as later:
                    await service.drain_preference_persistence()
                assert later.value is error
        assert storage_closed.is_set()
        assert len(writes) == 1 and closes == [True]
        assert not service.has_pending_close_owners()
    finally:
        release.set()
        for task in (writer, shutdown, waiter, agent._durable_shutdown_continuation):
            if task is not None:
                await await_owned_task(task)
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_failed_heartbeat_rearms_before_and_after_revoked_route_retention(tmp_path, kind):
    from tests.unit.test_durable_signal_delivery import _dispatcher, _close
    from tests.unit.test_execution_custody_review10 import runtime_owner

    backend, old_agent, dispatcher = await _dispatcher(tmp_path / "heartbeat.db", "did:test:rearm")
    owner = runtime_owner()
    owner.did = old_agent.did
    owner._runtime_publication_ready.set()
    dispatcher._agent = owner
    owner._execution_custody.revoke("revoked before route retention")
    release = asyncio.Event()
    route = asyncio.create_task(release.wait())
    dispatcher._release_retained_durable_cognition_task = AsyncMock()
    error = receipt(kind)

    async def fail():
        raise error

    failed = asyncio.create_task(fail())
    await await_owned_task(failed)
    try:
        assert not dispatcher._runtime_owner_cleanup_required()
        dispatcher._finish_runtime_owner_heartbeat(failed)
        assert dispatcher._runtime_owner_heartbeat_timer is not None
        context = dispatcher._resident_timer_context()
        with pytest.raises(ExecutionAuthorityError):
            context.run(require_execution_work, owner)
        dispatcher._runtime_owner_heartbeat_timer.cancel()
        dispatcher._runtime_owner_heartbeat_timer = None
        dispatcher._retain_durable_cognition_task(route, delivery=SimpleNamespace(delivery_id="original"))
        assert dispatcher._runtime_owner_heartbeat_timer is not None
        assert dispatcher._runtime_owner_cleanup_required()
        with pytest.raises(ExecutionAuthorityError):
            require_execution_work(owner)
    finally:
        release.set()
        await await_owned_task(route)
        await asyncio.sleep(0)
        await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        await dispatcher._stop_runtime_owner_heartbeat()
        await _close(backend, old_agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group", "ordinary"])
async def test_late_native_cognition_carrier_is_harvested_before_owner_removal(tmp_path, kind):
    from tests.unit.test_durable_signal_delivery import _dispatcher, _close

    backend, agent, dispatcher = await _dispatcher(tmp_path / "late-cognition.db", "did:test:late-group")
    await backend.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    release = asyncio.Event()
    error = receipt(kind)
    delivery = ReceiptDelivery(delivery_id="original", consumer_id="consumer", lease_token="original-token")
    dispatcher._terminalize_failed_cognition = AsyncMock()
    dispatcher.release_durable_delivery_after_task = AsyncMock()

    async def late_effect():
        await backend.execute("INSERT INTO effects VALUES (1)")
        await release.wait()
        raise error

    task = asyncio.create_task(late_effect())
    dispatcher._retain_durable_cognition_task(task, delivery=delivery)
    try:
        release.set()
        await await_owned_task(task)
        await asyncio.sleep(0)
        await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        assert await backend.fetch_val("SELECT COUNT(*) FROM effects") == 1
        assert not dispatcher._retained_durable_cognition_tasks
        if kind == "ordinary":
            dispatcher.release_durable_delivery_after_task.assert_awaited_once()
            dispatcher._terminalize_failed_cognition.assert_not_awaited()
        else:
            dispatcher._terminalize_failed_cognition.assert_awaited_once_with(delivery, error)
            dispatcher.release_durable_delivery_after_task.assert_not_awaited()
    finally:
        release.set()
        await await_owned_task(task)
        await dispatcher._stop_runtime_owner_heartbeat()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["authority", "unknown", "cancel_unknown", "group", "ordinary"])
async def test_actual_spawn_tool_preserves_distinct_child_denial_with_live_parent(tmp_path, kind):
    from tests.unit.test_spawn_feature import _make_spawn_feature

    parent, _db = await native_owner(tmp_path)
    parent._execution_custody = ExecutionCustody(Authority())
    child = SimpleNamespace(_execution_custody=ExecutionCustody(Authority()))
    child._execution_custody.revoke("original child generation retired")
    error = receipt(kind)

    async def spawn(**kwargs):
        parent._execution_custody.require_work()
        with pytest.raises(ExecutionAuthorityError):
            require_execution_work(child)
        raise error

    manager = MagicMock()
    manager.spawn_agent = AsyncMock(side_effect=spawn)
    feature = _make_spawn_feature(parent_agent=parent, manager=manager)
    try:
        with bind_execution_runtime(parent):
            if kind == "ordinary":
                result = await feature.spawn_agent(name="child", purpose="provider-free authority proof")
                assert result.error == str(error)
            else:
                with pytest.raises(BaseException) as caught:
                    await feature.spawn_agent(name="child", purpose="provider-free authority proof")
                assert caught.value is error
            parent._execution_custody.require_work()
        manager.spawn_agent.assert_awaited_once()
    finally:
        parent._execution_custody.revoke("test retirement")
        await retire(parent)


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("delayed_close", [False, True])
@pytest.mark.parametrize("unknown_close", [False, True])
async def test_actual_continuation_closes_unstarted_provider_only_after_dispatcher(
    tmp_path, monkeypatch, retained, delayed_close, unknown_close,
):
    import httpx
    from kestrel_sovereign.agent.custody import ReleaseOutcome, ResourceCustody
    from kestrel_sovereign.llm import service as service_module

    agent, db = await native_owner(tmp_path)
    service = service_owner()
    client = httpx.AsyncClient()
    close_error = ExecutionCommitOutcomeError("unknown")
    source = OriginalFailure.capture(close_error)
    client_entered, release_client = asyncio.Event(), asyncio.Event()

    async def close_client():
        client_entered.set()
        if delayed_close:
            while not release_client.is_set():
                try:
                    await release_client.wait()
                except asyncio.CancelledError:
                    pass
        await client.aclose()
        if unknown_close:
            raise close_error

    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.01)
    service.providers = [{"adapter": SimpleNamespace(aclose=close_client)}]
    agent.llm_service = service
    entered, release = asyncio.Event(), asyncio.Event()
    custody = ResourceCustody(truthful_owners={"signal_dispatcher", "llm_service", "storage"})
    agent._custody = custody
    for resource in ("signal_dispatcher", "llm_service", "storage"):
        custody.acquire(resource)

    async def wait():
        entered.set()
        await release.wait()
        return ReleaseOutcome.RELEASED

    dispatcher = SimpleNamespace(
        shutdown_durable_delivery=AsyncMock(return_value=ReleaseOutcome.RETAINED if retained else False),
        wait_for_durable_shutdown_release=wait,
    )
    task = asyncio.create_task(agent._complete_durable_shutdown_continuation(
        dispatcher, agent.storage, None,
    ))
    try:
        await entered.wait()
        assert not client.is_closed
        assert "llm_service" in custody.held and "storage" in custody.held
        release.set()
        if retained:
            outcome = await await_owned_task(task)
            assert isinstance(outcome.error, RuntimeError)
            assert not client.is_closed
            assert set(custody.held) == {"signal_dispatcher", "llm_service", "storage"}
            dispatcher.shutdown_durable_delivery.return_value = ReleaseOutcome.RELEASED
            task = asyncio.create_task(agent._complete_durable_shutdown_continuation(
                dispatcher, agent.storage, None,
            ))
        await asyncio.wait_for(client_entered.wait(), 5)
        if delayed_close:
            # Outlast BOTH bounded close attempts: otherwise removing the
            # native join can still appear pending on its second short close.
            await asyncio.wait({task}, timeout=0.5)
            assert not task.done()
            assert not client.is_closed
            assert "llm_service" in custody.held and "storage" in custody.held
        release_client.set()
        outcome = await await_owned_task(task)
        if unknown_close:
            source.assert_retained(outcome.error, resource=service.providers[0]["adapter"])
        else:
            assert outcome.error is None
        assert client.is_closed
        if unknown_close:
            # The test can observe HTTPX closure, but the opaque adapter's
            # failed native close did not provide positive retirement proof.
            # Its control receipt must not release borrowed storage custody.
            assert "llm_service" in custody.held and "storage" in custody.held
            assert service.has_pending_close_owners()
        else:
            assert "llm_service" not in custody.held and "storage" not in custody.held
    finally:
        release.set()
        release_client.set()
        await await_owned_task(task)
        await client.aclose()
        await retire(agent)
        # Test-owned storage may deliberately remain in production custody
        # after an uncertain adapter close. Physically retire our own fixture.
        await db.close()
