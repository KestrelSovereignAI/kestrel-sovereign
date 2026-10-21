"""Physical threads and real transition locks under bounded observation."""
import asyncio
import sqlite3
import threading
from contextlib import closing

import pytest

from kestrel_sovereign._async_ownership import (
    PendingOwnedTask, await_owned_task, create_owned_task, run_blocking_operation,
    wait_owned_or_pending,
)
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, ExecutionCommitOutcomeError
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock, bind_transition_lock_reentry


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", ["unknown", "committed"])
async def test_bounded_observer_preserves_late_native_commit_on_multiple_joins(tmp_path, receipt):
    path = tmp_path / "native.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE effects(id INTEGER)")
    entered, release = threading.Event(), threading.Event()
    original = RuntimeError("native transaction acknowledgement")
    original.__cause__ = ExecutionCommitOutcomeError(receipt)
    def commit():
        entered.set()
        if not release.wait(3):
            raise TimeoutError("fixture thread not released")
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("INSERT INTO effects VALUES (1)")
        raise original
    native = create_owned_task(run_blocking_operation(commit))
    observer = create_owned_task(wait_owned_or_pending(native, cancellation_grace=.03))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        observer.cancel()
        await asyncio.sleep(.01)
        observer.cancel()
        done, _ = await asyncio.wait({observer}, timeout=.2)
        assert done, "repeated cancellation reset bounded observer deadline"
        result = (await await_owned_task(observer)).result
        assert isinstance(result, PendingOwnedTask) and result.task is native
        assert not native.done() and native.cancelling() == 1
        with pytest.raises(ExecutionAuthorityError) as pending:
            result.raise_abandonment()
        assert pending.value.native_owner is native
        assert not isinstance(pending.value, ExecutionCommitOutcomeError)
        release.set()
        first = await await_owned_task(native)
        second = await await_owned_task(native)
        assert first.error is second.error is original
        with closing(sqlite3.connect(path)) as connection, connection:
            assert connection.execute("SELECT COUNT(*) FROM effects").fetchone() == (1,)
    finally:
        release.set()
        await await_owned_task(observer)
        await await_owned_task(native)


@pytest.mark.asyncio
@pytest.mark.parametrize("grace", [0, -1, float("inf"), float("nan")])
async def test_observation_requires_a_real_positive_bound(grace):
    native = create_owned_task(asyncio.sleep(0))
    try:
        with pytest.raises(ValueError, match="finite and positive"):
            await wait_owned_or_pending(native, cancellation_grace=grace)
    finally:
        await await_owned_task(native)


@pytest.mark.asyncio
async def test_successful_foreign_native_dependency_releases_original_mutex():
    lock = ReentrantTransitionLock()
    async def native():
        return "native-success"
    async def foreign(token):
        with bind_transition_lock_reentry(token):
            async with lock:
                task = lock.start_owned_native_work(native)
                assert (await await_owned_task(task)).result == "native-success"
    async with lock:
        worker = create_owned_task(foreign(lock.current_reentry_token()))
        result = await await_owned_task(worker)
        assert result.error is None
    assert not lock.locked() and lock._reentry_idle.is_set()
    assert not lock._foreign_native_dependencies


@pytest.mark.asyncio
async def test_task_creation_failure_dispatches_no_native_body_or_dependency():
    lock = ReentrantTransitionLock()
    called = False
    async def native():
        nonlocal called
        called = True
    async def foreign(token):
        with bind_transition_lock_reentry(token):
            async with lock:
                loop = asyncio.get_running_loop()
                previous = loop.get_task_factory()
                def refuse(loop, coro, **kwargs):
                    raise RuntimeError("native task factory refused dispatch")
                loop.set_task_factory(refuse)
                try:
                    with pytest.raises(RuntimeError, match="refused dispatch"):
                        lock.start_owned_native_work(native)
                finally:
                    loop.set_task_factory(previous)
    async with lock:
        worker = create_owned_task(foreign(lock.current_reentry_token()))
        assert (await await_owned_task(worker)).error is None
    assert not called and not lock._native_work
    assert not lock._foreign_native_dependencies and not lock.locked()


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(asyncio, "eager_task_factory"), reason="native eager factory requires Python3.12+")
async def test_eager_task_factory_sees_dependency_before_native_dispatch():
    lock = ReentrantTransitionLock()
    async def native():
        current = asyncio.current_task()
        assert current in lock._native_work
        assert any(current in tasks for tasks in lock._foreign_native_dependencies.values())
        return "retained-before-dispatch"
    async def foreign(token):
        with bind_transition_lock_reentry(token):
            async with lock:
                loop = asyncio.get_running_loop()
                previous = loop.get_task_factory()
                loop.set_task_factory(asyncio.eager_task_factory)
                try:
                    task = lock.start_owned_native_work(native)
                    assert (await await_owned_task(task)).result == "retained-before-dispatch"
                finally:
                    loop.set_task_factory(previous)
    async with lock:
        worker = create_owned_task(foreign(lock.current_reentry_token()))
        assert (await await_owned_task(worker)).error is None
    assert not lock.locked()
