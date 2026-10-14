"""Tests for the loop-clock helpers in ``tests/utils/virtual_clock.py``."""

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from tests.utils.virtual_clock import (
    AdvanceableClock,
    advanceable_loop_clock,
    await_within_hang_guard,
    pinned_loop_clock,
    wait_for_thread_event,
)


def test_advanceable_clock_keeps_real_time_and_jumps_forward() -> None:
    clock = AdvanceableClock(time.monotonic)
    before = clock.time()
    time.sleep(0.001)
    assert clock.time() > before

    clock.advance(3600)
    assert clock.time() >= before + 3600

    with pytest.raises(ValueError, match="cannot move backwards"):
        clock.advance(-1)


@pytest.mark.asyncio
async def test_advance_brings_an_armed_deadline_due() -> None:
    loop = asyncio.get_running_loop()
    fired = asyncio.Event()

    with advanceable_loop_clock() as clock:
        loop.call_later(3600, fired.set)
        await asyncio.sleep(0)
        assert not fired.is_set()

        clock.advance(3600)
        await await_within_hang_guard(
            fired.wait(), step="the deadline the jump passed firing"
        )


@pytest.mark.asyncio
async def test_advanceable_clock_runs_loop_sleeps_in_real_time() -> None:
    with advanceable_loop_clock():
        await await_within_hang_guard(
            asyncio.sleep(0.001), step="a short sleep finishing"
        )


@pytest.mark.asyncio
async def test_leaving_advanceable_clock_takes_the_jumps_back_out() -> None:
    loop = asyncio.get_running_loop()

    with advanceable_loop_clock() as clock:
        clock.advance(3600)
        jumped = loop.time()

    assert "time" not in vars(loop)
    assert loop.time() < jumped


@pytest.mark.asyncio
async def test_hang_guard_returns_the_result_and_raises_the_failure() -> None:
    async def answer() -> int:
        return 42

    async def fail() -> None:
        raise RuntimeError("step failed")

    assert await await_within_hang_guard(answer(), step="answering") == 42
    with pytest.raises(RuntimeError, match="step failed"):
        await await_within_hang_guard(fail(), step="failing")


@pytest.mark.asyncio
async def test_hang_guard_fires_while_the_loop_clock_is_pinned() -> None:
    """The guard is measured on a thread: a pinned loop clock cannot stop it."""
    with pinned_loop_clock():
        with pytest.raises(
            AssertionError,
            match="the event being set did not happen within the 0.05s",
        ):
            await await_within_hang_guard(
                asyncio.Event().wait(),
                step="the event being set",
                seconds=0.05,
            )


@pytest.mark.asyncio
async def test_hang_guard_survives_a_jump_of_the_loop_clock() -> None:
    """A jump that expires every loop deadline does not fire the guard."""
    release = asyncio.Event()
    loop = asyncio.get_running_loop()

    with advanceable_loop_clock() as clock:

        async def release_after_jump() -> None:
            clock.advance(3600)
            await asyncio.sleep(0)
            release.set()

        jumper = loop.create_task(release_after_jump())
        await await_within_hang_guard(release.wait(), step="the release")
        await jumper


@pytest.mark.asyncio
async def test_hang_guard_leaves_an_awaited_task_and_cancels_its_own() -> None:
    task = asyncio.create_task(asyncio.Event().wait())
    owned_started = asyncio.Event()
    owned_cancelled = asyncio.Event()

    async def owned() -> None:
        owned_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            owned_cancelled.set()
            raise

    try:
        with pytest.raises(AssertionError, match="the task"):
            await await_within_hang_guard(task, step="the task", seconds=0.01)
        assert not task.done()

        with pytest.raises(AssertionError, match="the coroutine"):
            await await_within_hang_guard(
                owned(), step="the coroutine", seconds=0.01
            )
        assert owned_started.is_set()
        await await_within_hang_guard(
            owned_cancelled.wait(), step="the owned coroutine being cancelled"
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_hang_guard_does_not_call_instrumented_asyncio_wait() -> None:
    """Tests that count ``asyncio.wait`` calls must not see the guard's."""

    async def answer() -> int:
        return 7

    def refuse(*args, **kwargs):
        raise AssertionError("the hang guard called asyncio.wait")

    with patch.object(asyncio, "wait", refuse):
        assert await await_within_hang_guard(answer(), step="answering") == 7


@pytest.mark.asyncio
async def test_wait_for_thread_event_returns_once_another_thread_sets_it() -> None:
    event = threading.Event()
    setter = threading.Thread(target=event.set)
    setter.start()
    try:
        await wait_for_thread_event(event, step="the thread setting the event")
    finally:
        setter.join()


@pytest.mark.asyncio
async def test_wait_for_thread_event_names_the_step_it_waited_for() -> None:
    with pytest.raises(
        AssertionError, match="the worker starting did not happen within"
    ):
        await wait_for_thread_event(
            threading.Event(), step="the worker starting", seconds=0.01
        )
