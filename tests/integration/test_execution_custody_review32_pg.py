"""Borrowed PostgreSQL writers fence actual agent storage retirement."""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, owned_task_error
from kestrel_sovereign.execution_custody import _explicit_control_chain, bind_execution_custody, execution_commit_outcome
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review31_pg import hold_native_ack
from tests.integration.test_execution_custody_review30_pg import assert_native_preference_rows
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review30 import service_owner

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("route", ["model", "embedding", "overrides"])
@pytest.mark.parametrize("cancel_receipt", [False, True])
async def test_native_commit_writer_must_settle_before_agent_storage_close(
    native_pg, monkeypatch, tmp_path, route, cancel_receipt,
):
    from kestrel_sovereign.llm import service as service_module

    backend, observer, schema = native_pg
    await backend.execute("CREATE TABLE IF NOT EXISTS agent_metadata(agent_id TEXT, key TEXT, value TEXT, updated_at TIMESTAMPTZ, PRIMARY KEY(agent_id,key))")
    agent, old_db = await native_owner(tmp_path)
    await old_db.close()
    db, service = AsyncDatabase(backend), service_owner()
    storage = AsyncStorage(backend=backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage
    agent.llm_service = service
    agent.privacy_agent = SimpleNamespace(privacy_config=SimpleNamespace(allows_cloud_llm=lambda: False))
    committed, cancelled, release, closed = (asyncio.Event() for _ in range(4))
    original_close = storage.close
    writer = shutdown = waiter = None
    closes = []
    withdrawals = []
    actual_wait = service_module.await_owned_task

    async def observed_wait(task, *args, **kwargs):
        outcome = await actual_wait(task, *args, **kwargs)
        if task is writer and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    monkeypatch.setattr(service_module, "await_owned_task", observed_wait)
    hold_native_ack(monkeypatch, committed, cancelled, release, cancel_receipt=cancel_receipt)
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.02)

    async def close_storage():
        assert writer.done(), "native committed writer lost its borrowed storage"
        closes.append(True)
        await original_close()
        closed.set()

    monkeypatch.setattr(storage, "close", close_storage)
    try:
        with bind_execution_custody(GenerationFence()):
            if route == "model":
                service.set_preference_persistence_callback(agent._persist_model_preference)
                writer = service._schedule_preference_persistence("model", "local", "native")
            elif route == "embedding":
                service.set_embedding_route_persistence_callback(agent._persist_embedding_route)
                writer = service._schedule_embedding_route_persistence(None)
            else:
                service.set_route_embedding_model_persistence_callback(agent._persist_route_embedding_models)
                writer = service._schedule_route_embedding_model_persistence()
            # Admission stays live through the actual native COMMIT. Retirement
            # starts only after it; never create a replacement for a late ACK.
            await asyncio.wait_for(committed.wait(), 5)
        await assert_native_preference_rows(
            observer, schema, route,
            mandate={"model": "model", "vendor": "local", "route": "native"},
        )
        shutdown = asyncio.create_task(agent.shutdown())
        await asyncio.wait_for(asyncio.shield(shutdown), 2)
        assert not writer.done() and not closed.is_set()
        assert service.has_pending_close_owners()
        assert service._llm_close_tasks["preference-writers"]._kestrel_llm_close_cancel_sent
        finite_close = service._llm_close_tasks["preference-writers"]
        assert agent._durable_shutdown_continuation is not None
        waiter = asyncio.create_task(agent.wait_for_shutdown_completion())
        await asyncio.sleep(0.01)
        assert not waiter.done()
        release.set()
        await await_owned_task(writer)
        original = owned_task_error(writer)
        assert execution_commit_outcome(original) == "unknown"
        with pytest.raises(BaseException) as caught:
            await asyncio.wait_for(waiter, 5)
        chain = tuple(_explicit_control_chain(caught.value))
        retained = tuple(_explicit_control_chain(agent._shutdown_control))
        assert original in chain and original in retained
        assert execution_commit_outcome(caught.value) == "unknown"
        assert execution_commit_outcome(agent._shutdown_control) == "unknown"
        finite_chain = tuple(_explicit_control_chain(owned_task_error(finite_close)))
        assert withdrawals and all(error in finite_chain for error in withdrawals)
        assert all(error not in chain for error in withdrawals)
        assert all(error not in retained for error in withdrawals)
        # The finite shutdown receipt retains its own cancellations; the
        # shared strongest-writer latch must not become caller history.
        assert service._preference_persistence_control is original
        assert all(error not in tuple(_explicit_control_chain(original)) for error in withdrawals)
        for _ in range(3):
            with pytest.raises(BaseException) as later:
                await service.drain_preference_persistence()
            assert later.value is original
        assert closed.is_set() and closes == [True]
        assert not service.has_pending_close_owners()
        await assert_native_preference_rows(
            observer, schema, route,
            mandate={"model": "model", "vendor": "local", "route": "native"},
        )
    finally:
        release.set()
        for task in (writer, shutdown, waiter, agent._durable_shutdown_continuation):
            if task is not None:
                await await_owned_task(task)
        await retire(agent)
