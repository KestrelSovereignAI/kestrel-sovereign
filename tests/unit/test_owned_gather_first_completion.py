"""Pure Task-owner policy checks; native ingress proof is an integration test."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    gather_owned,
)
from tests.unit.test_tool_effect_execution import owner as owner


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [asyncio.FIRST_EXCEPTION, "unsupported"])
async def test_invalid_race_policy_does_not_consume_operations(policy):
    consumed = []

    def operations():
        consumed.append(True)
        yield asyncio.sleep(0)

    with pytest.raises(ValueError, match="ALL_COMPLETED or FIRST_COMPLETED"):
        await gather_owned(operations(), return_when=policy)
    assert consumed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [asyncio.ALL_COMPLETED, asyncio.FIRST_COMPLETED])
async def test_empty_owned_gather_has_no_waiter(policy):
    assert await gather_owned(iter(()), return_when=policy) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('acquire_span', [False, True])
async def test_native_gather_refusal_joins_accepted_task_without_constructing_bodies(owner, acquire_span):
    import inspect
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from tests.unit.test_host_context_native_cleanup import _actual_operation

    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    original = RuntimeError('Original native parallel allocation refused')
    allocated, refused, constructed = [], [], []

    async def body():
        raise AssertionError('A refused native batch constructed an effect body')

    def lazy_body():
        constructed.append(True)
        return body()

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, 'cr_code', None)
        native = code is not None and code.co_qualname.endswith(
            'ReentrantTransitionLock.start_owned_native_work.<locals>.run')
        if native and allocated:
            refused.extend((coroutine, operation))
            raise original
        task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if native:
            allocated.append(task)
        return task

    lock = owner._get_privacy_transition_lock()
    try:
        async with lock:
            loop.set_task_factory(factory)
            try:
                with pytest.raises(BaseException) as observed:
                    await gather_owned([lazy_body, lazy_body], retain_originals=True,
                        work_factory=lock.capture_native_work_factory(acquire_span=acquire_span))
            finally:
                loop.set_task_factory(previous)
        assert any(error is original for error in execution_evidence_carriers(observed.value))
        assert len(allocated) == 1 and allocated[0].done()
        assert not constructed and not lock._native_work
        assert all(inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED for coroutine in refused)
    finally:
        loop.set_task_factory(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [asyncio.ALL_COMPLETED, asyncio.FIRST_COMPLETED])
async def test_owned_gather_default_waits_but_race_retires_peer(policy):
    entered, release, winner = asyncio.Event(), asyncio.Event(), asyncio.Event()
    originals, cancellations = [], []

    async def first():
        await entered.wait()
        winner.set()
        return "winner"

    async def second():
        originals.append(asyncio.current_task())
        entered.set()
        try:
            await release.wait()
            return "second"
        except asyncio.CancelledError as error:
            cancellations.append(error)
            raise

    owner = create_owned_task(
        gather_owned((operation() for operation in (first, second)), return_when=policy)
    )
    try:
        await asyncio.wait_for(winner.wait(), 5)
        if policy == asyncio.ALL_COMPLETED:
            assert not owner.done() and not cancellations
            release.set()
        outcome = await asyncio.wait_for(await_owned_task(owner), 5)
        assert outcome.error is None
        assert originals[0].done()
        if policy == asyncio.FIRST_COMPLETED:
            assert cancellations and outcome.result == ["winner", None]
        else:
            assert not cancellations and outcome.result == ["winner", "second"]
    finally:
        release.set()
        await await_owned_task(owner)
