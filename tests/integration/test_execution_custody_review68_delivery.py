"""Delivery disable joins its actual status COMMIT acknowledgement."""

import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ProcessRuntimeExecutionFence, execution_commit_outcome,
)
from kestrel_sovereign.features.delivery.queue import DeliveryQueue
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("cancel_stop", [False, True])
async def test_delivery_disable_does_not_cancel_native_status_ack(native_pg, monkeypatch, cancel_stop):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    queue = DeliveryQueue(db, "original", allow_noop_delivery=True,
        poll_interval=60, delivery_task_factory=create_owned_task)
    await queue._ensure_tables()
    first = await queue.enqueue("offline", "recipient-one", {"message": "first"})
    second = await queue.enqueue("offline", "recipient-two", {"message": "second"})
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("original-delivery-runtime"))
    backend._execution_custody = runtime
    entered, release = asyncio.Event(), asyncio.Event()
    actual_exit = asyncpg.transaction.Transaction.__aexit__
    commits = []

    async def held_ack(transaction, error_type, error, traceback):
        result = await actual_exit(transaction, error_type, error, traceback)
        name = asyncio.current_task().get_name()
        if error_type is None and not transaction._nested and (
            name == f"delivery-attempt:{first}"
            or name.startswith(f"delivery-attempt:{first}:postgres-original-")
        ):
            status = await observer.fetchval(f'SELECT status FROM "{schema}".delivery_queue WHERE id=$1', first)
            if status == "delivered":
                commits.append(transaction)
                entered.set()
                await release.wait()
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_ack)
    stopping = None
    try:
        await queue.start()
        await asyncio.wait_for(entered.wait(), 3)
        stopping = create_owned_task(queue.stop())
        if cancel_stop:
            await asyncio.sleep(0)
            stopping.cancel()
        done, _ = await asyncio.wait({stopping}, timeout=.05)
        assert not done, "Delivery disable interrupted an already committed native acknowledgement"
        release.set()
        outcome = await await_owned_task(stopping)
        assert execution_commit_outcome(outcome.error) is None
        if outcome.error is not None:
            assert isinstance(outcome.error, asyncio.CancelledError)
        assert len(commits) == 1 and queue._task.done()
        runtime.require_work()
        assert await observer.fetchval(f'SELECT status FROM "{schema}".delivery_queue WHERE id=$1', second) == "pending"
        await db.execute("INSERT INTO effects VALUES (68, 'unrelated agent work remains admitted')")
    finally:
        release.set()
        if stopping is not None:
            await await_owned_task(stopping)
        await await_owned_task(create_owned_task(queue.stop()))
        monkeypatch.undo()
        backend._execution_custody = None
