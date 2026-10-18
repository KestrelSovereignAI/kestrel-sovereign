"""A native delivery child cannot synchronously join its waiting worker.

The provider callback is a declared local SQL component, not a remote message
acceptance. Real queue rows, worker/child joins and native effects execute.
"""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    owned_task_join_ancestors,
)
from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.features.delivery import queue as queue_module
from kestrel_sovereign.features.delivery.models import DeliveryResult
from kestrel_sovereign.features.delivery.queue import DeliveryQueue
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("provider_handles_pending", [False, True])
async def test_delivery_descendant_stop_retains_actual_worker_until_ack(
    native_pg, monkeypatch, nested, provider_handles_pending
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)
    stops, starts, ancestors, provider_owners, descendants = [], [], [], [], []
    queue = None
    actual_join = queue_module.await_owned_task

    async def join(task, *args, **kwargs):
        if (
            queue is not None
            and task is queue._task
            and (task is asyncio.current_task() or task in owned_task_join_ancestors())
        ):
            # Stop before performing an actual unbreakable join cycle. This is
            # the real framework ownership graph, not a fabricated Task result.
            raise AssertionError(
                "delivery descendant attempted its original ancestor join"
            )
        return await actual_join(task, *args, **kwargs)

    monkeypatch.setattr(queue_module, "await_owned_task", join)

    async def stop():
        descendants.append(asyncio.current_task())
        ancestors.append(owned_task_join_ancestors())
        try:
            await queue.stop()
        except BaseException as error:
            stops.append(error)
            if not provider_handles_pending or not is_execution_control_error(error):
                raise
            try:
                await queue.start()
            except BaseException as startup_error:
                starts.append(startup_error)
            else:
                # A deliberately broken restart guard must fail finitely,
                # rather than leave two polling owners waiting in the test.
                queue._running = False
                queue._stop_requested.set()
                raise AssertionError("pending restart admitted a replacement worker")
            return DeliveryResult(success=True)
        pytest.fail("descendant stop claimed complete retirement of its waiting worker")

    async def deliver(*args):
        provider_owners.append(asyncio.current_task())
        await backend.execute("INSERT INTO effects VALUES (1, 'actual local provider')")
        if not nested:
            return await stop()
        descendant = create_owned_task(stop())
        outcome = await await_owned_task(descendant)
        if outcome.error is not None:
            raise outcome.error
        return outcome.result

    queue = DeliveryQueue(
        db,
        "original-descendant",
        deliver=deliver,
        poll_interval=60,
        delivery_task_factory=create_owned_task,
    )
    await queue._ensure_tables()
    entry = await queue.enqueue("local-component", "recipient", {"value": "original"})
    worker = None
    try:
        await queue.start()
        worker = queue._task
        outcome = await asyncio.wait_for(await_owned_task(worker), 5)
        assert stops and is_execution_control_error(stops[0]), (
            "delivery descendant still attempts an ancestor join instead of retaining pending retirement"
        )
        assert all(worker in view for view in ancestors)
        assert all(task.done() for task in provider_owners + descendants)
        assert any(
            resource is queue and task is worker
            for resource, task in execution_cleanup_debts(stops[0])
        )
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
        status = await observer.fetchval(
            f'SELECT status FROM "{schema}".delivery_queue WHERE id=$1', entry
        )
        if provider_handles_pending:
            assert outcome.error is None and status == "delivered"
            assert starts and is_execution_control_error(starts[0])
            assert queue._task is worker, "pending restart replaced the original worker"
            assert any(
                resource is queue and task is worker
                for resource, task in execution_cleanup_debts(starts[0])
            )
            # Successful original native ACK resolves only pending retirement,
            # not an independent control failure. Healthy restart stays valid.
            await queue.stop()
            await queue.start()
            replacement = queue._task
            assert replacement is not worker
            await queue.stop()
            assert replacement.done()
        else:
            assert outcome.error is not None and is_execution_control_error(
                outcome.error
            )
            assert status == "in_flight"
            with pytest.raises(BaseException) as refused:
                await queue.start()
            assert is_execution_control_error(refused.value)
            assert queue._task is worker
            assert (
                await observer.fetchval(
                    f'SELECT status FROM "{schema}".delivery_queue WHERE id=$1', entry
                )
                == "in_flight"
            ), "failed original delivery was automatically reclaimed"
            with pytest.raises(BaseException) as stopped:
                await queue.stop()
            assert is_execution_control_error(stopped.value)
    finally:
        queue._running = False
        queue._stop_requested.set()
        if queue._task is not None:
            await await_owned_task(queue._task)
        if worker is not None:
            await await_owned_task(worker)


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [False, True])
async def test_delivery_worker_self_stop_retains_its_actual_task(
    native_pg, monkeypatch, handled
):
    backend, observer, schema = native_pg
    pending = []
    actual_join = queue_module.await_owned_task

    async def join(task, *args, **kwargs):
        assert task is not asyncio.current_task(), (
            "delivery worker attempted to join itself"
        )
        return await actual_join(task, *args, **kwargs)

    monkeypatch.setattr(queue_module, "await_owned_task", join)

    class QueueInput(DeliveryQueue):
        async def process_pending(self):
            # Declared local polling body, not a remote delivery result/ACK.
            await backend.execute(
                "INSERT INTO effects VALUES (1, 'actual worker body')"
            )
            try:
                await self.stop()
            except BaseException as error:
                pending.append(error)
                if not handled or not is_execution_control_error(error):
                    raise
            return 1

    queue = QueueInput(AsyncDatabase(backend), "self-stop", poll_interval=60)
    try:
        await queue.start()
        worker = queue._task
        outcome = await asyncio.wait_for(await_owned_task(worker), 5)
        assert pending and is_execution_control_error(pending[0])
        assert any(
            resource is queue and task is worker
            for resource, task in execution_cleanup_debts(pending[0])
        )
        assert worker.done() and queue._task is worker
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
        if handled:
            assert outcome.error is None
            await queue.stop()
        else:
            assert is_execution_control_error(outcome.error)
            with pytest.raises(BaseException) as refused:
                await queue.start()
            assert is_execution_control_error(refused.value)
    finally:
        queue._running = False
        queue._stop_requested.set()
        if queue._task is not None:
            await await_owned_task(queue._task)
