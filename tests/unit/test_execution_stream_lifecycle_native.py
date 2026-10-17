"""Opt-in native producer registration and exact foreign-frame custody."""

import asyncio
import sqlite3

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, run_blocking_operation
from kestrel_sovereign.execution_custody import ExecutionStreamLifecycle, execution_stream
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock, bind_transition_lock_reentry


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_lazy_native_stream_is_registered_before_any_body_step(tmp_path, eager):
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager task factory unavailable")
    path = tmp_path / "native.db"
    lock, registered = ReentrantTransitionLock(), set()
    class Host:
        def _get_privacy_transition_lock(self):
            return lock
    host = Host()
    def commit():
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE effects(id INTEGER)")
            connection.execute("INSERT INTO effects VALUES (1)")
    async def source():
        assert asyncio.current_task() in registered
        await run_blocking_operation(commit)
        yield b"native"
    def started(iterator, task):
        assert iterator._consumer_close.owner is task
        registered.add(task)
    lifecycle = ExecutionStreamLifecycle(lock.capture_native_work_factory(), started, lambda: None)
    stream = execution_stream(host, source(), lifecycle=lifecycle)
    assert not registered and not path.exists()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        assert await anext(stream) == b"native"
        await stream.aclose()
        assert all(task.done() for task in registered)
        assert not lock.locked()
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM effects").fetchone() == (1,)
    finally:
        await stream.aclose()
        loop.set_task_factory(previous)


@pytest.mark.asyncio
async def test_exact_native_continuation_does_not_authorize_copied_context_sibling(tmp_path):
    lock = ReentrantTransitionLock()
    sibling_entered = asyncio.Event()
    sibling_tasks = []
    path = tmp_path / "native.db"
    def commit():
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE effects(id INTEGER)")
            connection.execute("INSERT INTO effects VALUES (1)")
    async def sibling():
        async with lock:
            sibling_entered.set()
    async def native_body():
        async with lock:
            async with lock:
                await run_blocking_operation(commit)
                sibling_tasks.append(create_owned_task(sibling()))
                await asyncio.sleep(0)
                assert not sibling_entered.is_set(), "copied context borrowed its creator's native frame"
    async def foreign(token):
        with bind_transition_lock_reentry(token):
            async with lock:
                child = lock.start_owned_native_work(native_body)
                outcome = await await_owned_task(child)
                assert outcome.error is None
                assert not sibling_entered.is_set()
    async with lock:
        frame = create_owned_task(foreign(lock.current_reentry_token()))
        outcome = await await_owned_task(frame)
        assert outcome.error is None
        for task in sibling_tasks:
            assert (await await_owned_task(task)).error is None
    assert sibling_entered.is_set() and not lock.locked()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM effects").fetchone() == (1,)
