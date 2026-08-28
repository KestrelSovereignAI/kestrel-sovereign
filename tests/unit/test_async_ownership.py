"""Contract tests for cancellation-safe ownership of internal tasks."""

from __future__ import annotations

import asyncio
import threading

import pytest

from kestrel_sovereign._async_ownership import (
    OwnedAsyncIterator,
    await_owned_task,
    raise_owned_outcome,
    run_blocking_operation,
)


@pytest.mark.asyncio
async def test_owned_task_result_is_retrieved():
    task = asyncio.create_task(asyncio.sleep(0, result="complete"))

    outcome = await await_owned_task(task)

    assert raise_owned_outcome(outcome, operation="test operation") == "complete"
    assert outcome.cancellation is None
    assert outcome.error is None


@pytest.mark.asyncio
async def test_owned_task_exception_retains_original_type():
    async def fail() -> None:
        raise LookupError("owned failure")

    outcome = await await_owned_task(asyncio.create_task(fail()))

    with pytest.raises(LookupError, match="owned failure"):
        raise_owned_outcome(outcome, operation="test operation")


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_owned_task_completion():
    owned_started = asyncio.Event()
    release_owned = asyncio.Event()

    async def owned_operation() -> str:
        owned_started.set()
        await release_owned.wait()
        return "settled"

    async def owner() -> None:
        task = asyncio.create_task(owned_operation())
        outcome = await await_owned_task(task)
        raise_owned_outcome(outcome, operation="test operation")

    owner_task = asyncio.create_task(owner())
    await owned_started.wait()
    owner_task.cancel()
    await asyncio.sleep(0)
    owner_task.cancel()

    assert not owner_task.done()
    release_owned.set()
    with pytest.raises(asyncio.CancelledError):
        await owner_task


@pytest.mark.asyncio
async def test_cancellation_remains_primary_when_owned_task_fails():
    owned_started = asyncio.Event()
    release_owned = asyncio.Event()

    async def fail_owned_operation() -> None:
        owned_started.set()
        await release_owned.wait()
        raise RuntimeError("cleanup exploded")

    async def owner() -> None:
        task = asyncio.create_task(fail_owned_operation())
        outcome = await await_owned_task(task)
        raise_owned_outcome(outcome, operation="test cleanup")

    owner_task = asyncio.create_task(owner())
    await owned_started.wait()
    unhandled: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    owner_task.cancel("first cancellation")
    await asyncio.sleep(0)
    owner_task.cancel("second cancellation")
    await asyncio.sleep(0)

    assert not owner_task.done()
    try:
        release_owned.set()

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await owner_task
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(prior_handler)

    assert owner_task.cancelled()
    assert isinstance(cancelled.value.__cause__, RuntimeError)
    assert "cleanup exploded" in " ".join(cancelled.value.__notes__)
    assert unhandled == []


@pytest.mark.asyncio
async def test_preexisting_cancellation_is_propagated_after_owned_result():
    cancellation = asyncio.CancelledError("already cancelled")
    task = asyncio.create_task(asyncio.sleep(0, result=42))

    outcome = await await_owned_task(task, cancellation)

    with pytest.raises(asyncio.CancelledError, match="already cancelled"):
        raise_owned_outcome(outcome, operation="test operation")


@pytest.mark.asyncio
async def test_closing_owned_iterator_interrupts_blocked_source():
    """Close must reach a producer blocked before its first item."""

    entered = asyncio.Event()
    release = asyncio.Event()
    finalized = asyncio.Event()

    async def source():
        try:
            entered.set()
            await release.wait()
            yield "late"
        finally:
            finalized.set()

    owned = OwnedAsyncIterator(source, operation="blocked source")
    consumer = asyncio.create_task(anext(owned))
    await entered.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    close = asyncio.create_task(owned.aclose())
    await asyncio.sleep(0.05)
    closed_without_source_release = close.done()
    release.set()
    await close

    assert closed_without_source_release is True
    assert finalized.is_set()


@pytest.mark.asyncio
async def test_blocking_operation_finishes_before_cancellation_propagates():
    started = threading.Event()
    release = threading.Event()

    def blocking_operation() -> str:
        started.set()
        release.wait()
        return "complete"

    owner = asyncio.create_task(run_blocking_operation(blocking_operation))
    for _attempt in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        release.set()
        raise AssertionError("blocking operation did not start")

    owner.cancel()
    await asyncio.sleep(0)
    assert not owner.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await owner
