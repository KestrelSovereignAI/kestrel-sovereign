"""Original owned Tasks survive observation and partial fan-out admission."""

import asyncio
import inspect

import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    gather_owned,
    wait_owned_tasks_until,
)
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from tests.unit.test_cleanup_task_admission import _operation_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deadline", [True, None, "later", float("nan"), float("inf"), -float("inf")]
)
async def test_owned_task_observation_rejects_nonfinite_deadline(deadline):
    with pytest.raises(ValueError, match="deadline must be finite"):
        await wait_owned_tasks_until((), deadline=deadline)


@pytest.mark.asyncio
async def test_expired_observation_keeps_the_exact_uncancelled_task():
    release = asyncio.Event()
    task = create_owned_task(release.wait())
    try:
        outcome = await wait_owned_tasks_until(
            (task,),
            deadline=asyncio.get_running_loop().time(),
        )
        assert outcome.result == (set(), {task})
        assert outcome.error is None and outcome.cancellation is None
        assert not task.done() and task.cancelling() == 0
    finally:
        release.set()
        await await_owned_task(task)


@pytest.mark.asyncio
async def test_repeated_observer_withdrawal_preserves_original_owner_and_deadline():
    release = asyncio.Event()
    original = create_owned_task(release.wait())
    deadline = asyncio.get_running_loop().time() + 0.1
    operation = wait_owned_tasks_until((original,), deadline=deadline)
    observer = create_owned_task(operation)
    try:
        async with asyncio.timeout(5):
            while operation.cr_frame.f_locals.get("remaining") is None:
                await asyncio.sleep(0)
        observer.cancel("first observation withdrawal")
        await asyncio.sleep(0)
        observer.cancel("second observation withdrawal")
        outcome = await await_owned_task(observer)
        assert outcome.error is None
        result = outcome.result
        assert result.result == (set(), {original})
        assert result.cancellation is not None
        assert not original.done() and original.cancelling() == 0
        assert operation.cr_frame is None
    finally:
        release.set()
        await await_owned_task(observer)
        await await_owned_task(original)


@pytest.mark.asyncio
async def test_independent_fanout_attempts_later_owner_and_joins_accepted_children():
    loop, factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    accepted, refused = [], []
    refusal = RuntimeError("middle original retirement admission refused")
    source = OriginalFailure.capture(refusal)
    resource = object()
    collector = None

    async def physical_exit(index):
        entered[index].set()
        await release.wait()
        return index

    async def refused_exit():
        pytest.fail("refused native exit entered")

    def operations():
        yield physical_exit(0)
        yield refused_exit()
        yield physical_exit(1)

    def admit(loop, coroutine, **kwargs):
        if _operation_name(coroutine).endswith(".refused_exit"):
            refused.append(coroutine)
            raise refusal
        task = (
            factory(loop, coroutine, **kwargs)
            if factory
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )
        if _operation_name(coroutine).endswith(".physical_exit"):
            accepted.append(task)
        return task

    try:
        loop.set_task_factory(admit)
        collector = create_owned_task(
            gather_owned(
                operations(),
                cancel_siblings_on_error=False,
                retirement_resource=resource,
                retain_originals=True,
            )
        )
        for event in entered:
            await asyncio.wait_for(event.wait(), 5)
        assert len(accepted) == 2 and not collector.done()
        assert all(task.cancelling() == 0 for task in accepted)
        release.set()
        outcome = await await_owned_task(collector)
        source.assert_retained(outcome.error)
        assert (resource, None) in execution_cleanup_debts(outcome.error)
        assert all(task.done() for task in accepted)
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused
        )
    finally:
        loop.set_task_factory(factory)
        release.set()
        if collector is not None:
            await await_owned_task(collector)
        for task in accepted:
            await await_owned_task(task)
        for coroutine in refused:
            coroutine.close()
