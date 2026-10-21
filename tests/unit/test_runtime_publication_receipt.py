"""Physical native task receipts survive postaccept publication refusal."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task
from tests.integration.test_durable_ingress_runtime_handoff_pg import retire_handoff_test
from tests.unit.test_durable_callback_runtime_context import resident_dispatcher
from tests.utils.runtime_task_factory import refusing_runtime_completion_factory


@pytest.mark.asyncio
async def test_actual_runtime_tracker_retains_task_when_native_completed_callback_refuses():
    agent, _ = resident_dispatcher()
    agent._runtime_publication_ready.clear()
    receipt = OwnedTaskAdmission()
    opening = OSError("native completion callback refused")
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    entered = False

    async def operation():
        nonlocal entered
        entered = True

    submitted = operation()
    try:
        loop.set_task_factory(refusing_runtime_completion_factory(opening))
        with pytest.raises(OSError) as caught:
            agent._track_runtime_task(submitted, name="durable_cognition_owner:refusal", admission=receipt)
        assert caught.value is opening
        assert receipt.task is not None
        assert receipt.task in agent._background_tasks
        assert not entered
    finally:
        loop.set_task_factory(original_factory)
        if receipt.task is not None:
            receipt.task.cancel()
            await await_owned_task(receipt.task)
            agent._background_tasks.discard(receipt.task)
        submitted.close()
    assert receipt.task.done()
    assert not entered


@pytest.mark.asyncio
async def test_dispatcher_shutdown_failure_still_physically_joins_original_agent_tasks():
    entered, closed = asyncio.Event(), asyncio.Event()
    original = OSError("dispatcher shutdown retained original failure")

    async def child():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def fail_shutdown():
        raise original

    task = asyncio.create_task(child())
    await entered.wait()
    try:
        with pytest.raises(OSError) as caught:
            await retire_handoff_test(
                SimpleNamespace(_background_tasks={task}),
                SimpleNamespace(shutdown_durable_delivery=fail_shutdown),
            )
        assert caught.value is original
        assert task.done() and closed.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
