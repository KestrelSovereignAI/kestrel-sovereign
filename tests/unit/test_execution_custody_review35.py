"""Actual native startup consumers preserve control after independent joins."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from tests.unit.test_execution_custody_review29 import native_owner
from tests.unit.test_execution_custody_review32 import receipt
from tests.utils.original_failure import OriginalFailure
from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.execution_custody import _explicit_control_chain


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["unpublished_batch", "owned_tasks"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_actual_native_startup_collector_preserves_receipt_after_caller_cancel(
    tmp_path, monkeypatch, surface, kind,
):
    first_path, peer_path = tmp_path / "first", tmp_path / "peer"
    first_path.mkdir()
    peer_path.mkdir()
    agent, db = await native_owner(first_path)
    peer, peer_db = await native_owner(peer_path)
    release, entered, closed, peer_closed = (asyncio.Event() for _ in range(4))
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)
    manager = AgentManager(base_data_dir=tmp_path)
    withdrawals, waiter = [], None
    actual_join = ownership.await_owned_task

    async def observe(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is waiter and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(ownership, "await_owned_task", observe)

    async def native_tail():
        entered.set()
        await release.wait()
        await db.execute("INSERT INTO effects VALUES (1)")
        await agent.storage.close()
        closed.set()

    async def shutdown_receipt():
        raise failure

    async def peer_shutdown():
        await peer_db.execute("INSERT INTO effects VALUES (2)")
        await peer.storage.close()
        peer_closed.set()

    continuation = asyncio.create_task(native_tail())
    agent._durable_shutdown_continuation = continuation
    monkeypatch.setattr(agent, "shutdown", shutdown_receipt)
    monkeypatch.setattr(peer, "shutdown", peer_shutdown)
    tasks = []
    if surface == "unpublished_batch":
        operation = manager._discard_unpublished_initialized_agents(
            [("first", agent), ("peer", peer)], already_withdrawn=True,
        )
    else:
        tasks = [asyncio.create_task(manager._shutdown_unregistered_agent(name, owner))
                 for name, owner in [("first", agent), ("peer", peer)]]
        operation = manager._settle_owned_lifecycle_tasks(tasks, description="native startup")
    waiter = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.sleep(0)
        waiter.cancel()
        done, _ = await asyncio.wait({waiter}, timeout=0.1)
        assert not done and not closed.is_set()
        release.set()
        outcome = await await_owned_task(waiter)
        assert closed.is_set() and peer_closed.is_set()
        assert db.backend._connection is None and peer_db.backend._connection is None
        if surface == "unpublished_batch":
            # The finite canonical cleanup now also retains its actual caller
            # withdrawal and failed original owner. Neither may replace or
            # mutate the shared native receipt.
            source.assert_retained(outcome.error)
            from kestrel_sovereign.execution_custody import execution_cleanup_debts
            assert any(resource is agent and task is not None and task.done()
                       for resource, task in execution_cleanup_debts(outcome.error))
        else:
            source.assert_retained(outcome.error)
            assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error))
                                       for error in withdrawals)
            assert all(task.done() for task in tasks) and continuation.done()
    finally:
        release.set()
        for task in (waiter, continuation, *tasks):
            await await_owned_task(task)
        await db.close()
        await peer_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
async def test_actual_stop_startup_joins_database_before_original_native_receipt(
    tmp_path, monkeypatch, kind,
):
    from kestrel_sovereign import server
    from kestrel_sovereign.host_features import storage as host_storage
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.stop import StopReceiptStore

    agent, db = await native_owner(tmp_path)
    failure = receipt(kind)
    source = OriginalFailure.capture(failure)
    withdrawals, waiter = [], None
    actual_join = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is waiter and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    entered, release, closed = (asyncio.Event() for _ in range(3))
    original_close = db.close

    async def failed_schema(_store):
        await db.execute("INSERT INTO effects VALUES (1)")
        raise failure

    async def delayed_close():
        entered.set()
        await release.wait()
        await original_close()
        closed.set()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setattr(host_storage, "prepare_host_database", lambda: tmp_path / "native.db")
    monkeypatch.setattr(host_storage, "validate_sqlite_family_private", lambda _path: None)
    monkeypatch.setattr(AsyncDatabase, "sqlite", AsyncMock(return_value=db))
    monkeypatch.setattr(StopReceiptStore, "ensure_schema", failed_schema)
    monkeypatch.setattr(db, "close", delayed_close)
    waiter = asyncio.create_task(server._initialize_stop_receipts(FastAPI()))
    try:
        # A group may currently bypass this cleanup branch entirely.
        observer = asyncio.create_task(entered.wait())
        done, _ = await asyncio.wait({observer, waiter}, timeout=0.1,
                                     return_when=asyncio.FIRST_COMPLETED)
        if waiter in done:
            outcome = await await_owned_task(waiter)
            assert closed.is_set(), "native group bypassed physical Stop database cleanup"
            source.assert_retained(outcome.error)
        else:
            assert observer in done
            waiter.cancel()
            done, _ = await asyncio.wait({waiter}, timeout=0.1)
            assert not done and not closed.is_set()
            release.set()
            outcome = await await_owned_task(waiter)
            assert closed.is_set() and db.backend._connection is None
            source.assert_retained(outcome.error)
            assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error)) for error in withdrawals)
    finally:
        release.set()
        await await_owned_task(waiter)
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        await original_close()
