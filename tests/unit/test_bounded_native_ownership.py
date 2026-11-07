"""Physical threads and real transition locks under bounded observation."""
import asyncio
import sqlite3
import threading
from contextlib import AsyncExitStack, closing

import pytest

from kestrel_sovereign._async_ownership import (
    PendingOwnedTask, await_owned_task, create_owned_task, run_blocking_operation, owned_task_error,
    wait_owned_or_pending,
)
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, ExecutionCommitOutcomeError
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock, bind_transition_lock_reentry


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2])
async def test_bounded_wait_parks_its_original_reentrant_writer(depth):
    lock = ReentrantTransitionLock()
    observations = []
    entered = []

    async def native():
        entered.append(asyncio.current_task())
        assert lock._writer is asyncio.current_task()
        return "original native result"

    async def writer():
        async with AsyncExitStack() as stack:
            for _ in range(depth):
                await stack.enter_async_context(lock)
            child = lock.start_owned_native_work(native)
            result = await wait_owned_or_pending(child, cancellation_grace=.03)
            observations.append((child, result, lock._writer is asyncio.current_task(), lock._depth))

    owner = create_owned_task(writer())
    try:
        _, pending = await asyncio.wait({owner}, timeout=.3)
    finally:
        # A regression must retire its original blocked child before failing;
        # cancellation starts the existing bounded observer's handoff.
        if not owner.done():
            owner.cancel()
        settled = await await_owned_task(owner)
    assert not pending, "bounded observation did not park the original privacy writer for its accepted child"
    assert settled.error is None and settled.cancellation is None
    child, result, restored, restored_depth = observations[0]
    assert entered == [child] and child.done()
    assert result.error is None and result.result == "original native result"
    assert restored and restored_depth == depth
    assert not lock.locked() and not lock._native_work and not lock._admissions


@pytest.mark.asyncio
async def test_bounded_writer_restores_only_after_original_cancelled_child_settles():
    lock = ReentrantTransitionLock()
    entered, release, sibling_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    children, observations = [], []
    original = RuntimeError("original native acknowledgement")
    original.__cause__ = ExecutionCommitOutcomeError("committed")

    async def native():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        raise original

    async def writer():
        async with lock:
            child = lock.start_owned_native_work(native)
            children.append(child)
            result = await wait_owned_or_pending(child, cancellation_grace=.03)
            observations.append((result, lock._writer is asyncio.current_task()))

    async def sibling():
        async with lock:
            sibling_entered.set()

    owner = create_owned_task(writer())
    ready = asyncio.create_task(entered.wait())
    peer = None
    try:
        completed, _ = await asyncio.wait({ready}, timeout=.3)
        admitted = bool(completed)
        if admitted:
            peer = create_owned_task(sibling())
            owner.cancel()
            await asyncio.sleep(.05)
            held = not owner.done() and not sibling_entered.is_set()
    finally:
        release.set()
        if not entered.is_set() and not owner.done():
            owner.cancel()
        if not ready.done():
            ready.cancel()
        await await_owned_task(ready)
        settled = await await_owned_task(owner)
        if peer is not None:
            await await_owned_task(peer)
    assert admitted, "bounded observation did not park the original privacy writer for its accepted child"
    assert held, "bounded observation resumed its writer before the original child settled"
    result, restored = observations[0]
    assert restored and children[0].done() and sibling_entered.is_set()
    assert not isinstance(result, PendingOwnedTask), "bounded observation returned pending after original physical settlement"
    assert result.error is owned_task_error(children[0]) is original
    assert isinstance(result.cancellation, asyncio.CancelledError)
    assert settled.error is None and settled.cancellation is None
    assert not lock.locked() and not lock._native_work and not lock._admissions


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
    assert not lock.locked() and lock._writer is None
    assert not lock._admissions


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
                    with pytest.raises(RuntimeError) as caught:
                        lock.start_owned_native_work(native)
                    from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
                    assert any(str(error) == "native task factory refused dispatch"
                               for error in _explicit_control_chain(caught.value))
                    assert execution_cleanup_debts(caught.value), (
                        "opaque factory refusal cannot settle its native admission"
                    )
                finally:
                    loop.set_task_factory(previous)
    async with lock:
        worker = create_owned_task(foreign(lock.current_reentry_token()))
        assert (await await_owned_task(worker)).error is None
    assert not called and not lock._native_work
    assert not lock._admissions and not lock.locked()


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(asyncio, "eager_task_factory"), reason="native eager factory requires Python3.12+")
async def test_eager_task_factory_sees_dependency_before_native_dispatch():
    lock = ReentrantTransitionLock()
    async def native():
        current = asyncio.current_task()
        assert current in lock._native_work
        assert lock._admissions[current].task is current
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
