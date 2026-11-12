"""Physical threads and real transition locks under bounded observation."""
import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, closing

import pytest

from kestrel_sovereign._async_ownership import (
    PendingOwnedTask, await_owned_task, create_owned_task, run_blocking_operation, owned_task_error,
    wait_owned_or_pending,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError,
    execution_evidence_carriers, execution_observer_cancellations, execution_native_source_error,
)
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock, bind_transition_lock_reentry


@pytest.mark.asyncio
@pytest.mark.parametrize('first_cancellation', [False, True])
@pytest.mark.parametrize('receipt', ['returned', 'committed-error'])
@pytest.mark.parametrize('bounded', [False, True])
async def test_general_join_retains_native_outcome_through_writer_restoration(first_cancellation, receipt, bounded):
    lock = ReentrantTransitionLock()
    entered, release_child, release_writer = (asyncio.Event() for _ in range(3))
    children, observations = [], []
    original = RuntimeError('original general-join acknowledgement')
    original.__cause__ = ExecutionCommitOutcomeError('committed')

    async def native_writer():
        entered.set()
        await release_writer.wait()

    async def native():
        children.append(lock.start_owned_native_work(native_writer))
        try:
            await release_child.wait()
        except asyncio.CancelledError:
            await release_child.wait()
        if receipt == 'committed-error':
            raise original
        return 'original general-join result'

    async def writer():
        async with lock:
            async with lock:
                child = lock.start_owned_native_work(native, acquire_span=False)
                children.append(child)
                result = (await wait_owned_or_pending(child, cancellation_grace=.03)
                    if bounded else await await_owned_task(child))
                observations.append((result, lock._writer is asyncio.current_task(), lock._depth))

    async def restoring():
        while not lock._parked or not lock._parked[0].restoring:
            await asyncio.sleep(0)

    owner = create_owned_task(writer())
    try:
        await asyncio.wait_for(entered.wait(), .3)
        if first_cancellation:
            owner.cancel('first general-join withdrawal')
            await asyncio.sleep(0)
        release_child.set()
        await asyncio.wait_for(restoring(), .3)
        owner.cancel('general-join writer restoration withdrawal')
        await asyncio.sleep(0)
    finally:
        release_child.set()
        release_writer.set()
        settled = await await_owned_task(owner)
        for child in children:
            await await_owned_task(child)
    assert observations, 'general join discarded its native outcome during writer restoration'
    result, restored, depth = observations[0]
    assert restored and depth == 2
    assert any(error.args == ('general-join writer restoration withdrawal',)
        for error in execution_evidence_carriers(result.error))
    if receipt == 'committed-error':
        assert any(error is original for error in execution_evidence_carriers(result.error)), (
            'general join lost its original native receipt during restoration')
        assert execution_native_source_error(result.error) is original
        assert len(execution_observer_cancellations(result.error)) == 1
    else:
        assert result.result == 'original general-join result', (
            'native join discarded successful result during writer restoration')
    if first_cancellation:
        assert result.cancellation is not None and result.cancellation.args == ('first general-join withdrawal',), (
            'general join lost its first observer withdrawal')
    else:
        assert result.cancellation is None
    assert settled.error is None and settled.cancellation is None
    assert not lock.locked() and not lock._native_work and not lock._admissions


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
@pytest.mark.parametrize("repeat", [False, True])
async def test_bounded_writer_restores_only_after_original_cancelled_child_settles(repeat):
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
            owner.cancel("first observer cancellation")
            await asyncio.sleep(.05)
            held = not owner.done() and not sibling_entered.is_set()
            if repeat:
                owner.cancel("cancellation during original writer restoration")
                await asyncio.sleep(0)
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
    assert children[0].done() and owned_task_error(children[0]) is original
    assert observations, "restoration cancellation bypassed the original native outcome"
    result, restored = observations[0]
    assert restored and children[0].done() and sibling_entered.is_set()
    assert not isinstance(result, PendingOwnedTask), "bounded observation returned pending after original physical settlement"
    assert any(error is original for error in execution_evidence_carriers(result.error))
    assert isinstance(result.cancellation, asyncio.CancelledError)
    assert result.cancellation.args == ("first observer cancellation",)
    if repeat:
        assert any(error.args == ("cancellation during original writer restoration",)
            for error in execution_observer_cancellations(result.error))
    else:
        assert result.error is original
    assert settled.error is None and settled.cancellation is None
    assert not lock.locked() and not lock._native_work and not lock._admissions


@pytest.mark.asyncio
async def test_still_pending_native_owner_retains_restoration_withdrawal():
    lock = ReentrantTransitionLock()
    entered, release_writer, release_child, observed = (asyncio.Event() for _ in range(4))
    children, results = [], []

    async def native_writer():
        entered.set()
        await release_writer.wait()

    async def child():
        writer = lock.start_owned_native_work(native_writer)
        children.append(writer)
        await await_owned_task(writer)
        await release_child.wait()
        return "physically settled later"

    async def observer():
        async with lock:
            native = lock.start_owned_native_work(child, acquire_span=False)
            children.append(native)
            result = await wait_owned_or_pending(native, cancellation_grace=.03)
            results.append(result)
            observed.set()

    owner = create_owned_task(observer())
    try:
        await asyncio.wait_for(entered.wait(), .3)
        owner.cancel("first pending observer cancellation")
        await asyncio.sleep(.05)
        owner.cancel("pending writer restoration cancellation")
        await asyncio.sleep(0)
        release_writer.set()
        await asyncio.wait_for(observed.wait(), .3)
        pending, = results
        assert isinstance(pending, PendingOwnedTask) and not pending.task.done()
        assert pending.task.cancelling() == 1
        assert pending.cancellation.args == ("first pending observer cancellation",)
        with pytest.raises(ExecutionAuthorityError) as caught:
            pending.raise_abandonment()
        assert caught.value.native_owner is pending.task
        assert any(error.args == ("pending writer restoration cancellation",)
            for error in execution_observer_cancellations(caught.value)), (
                "pending native handoff lost its original restoration withdrawal")
        native_refusal = execution_native_source_error(caught.value)
        assert isinstance(native_refusal, ExecutionAuthorityError)
        assert native_refusal.native_owner is pending.task
        assert not any(isinstance(error, asyncio.CancelledError)
            for error in execution_evidence_carriers(native_refusal)), (
                "pending native handoff promoted observer withdrawal into its shared source")
    finally:
        release_writer.set()
        release_child.set()
        await await_owned_task(owner)
        for task in children:
            await await_owned_task(task)
    assert not lock.locked() and not lock._native_work and not lock._admissions


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed_middle", [False, True])
async def test_original_close_wait_delegates_only_through_accepted_native_tasks(borrowed_middle):
    lock = ReentrantTransitionLock()
    started, entered, release, sibling_entered = (asyncio.Event() for _ in range(4))
    children, restoration = {}, []

    async def source():
        entered.set()
        await release.wait()
        return "original source"

    async def sibling():
        sibling_entered.set()

    async def close_source():
        return await await_owned_task(children["source"])

    async def close_wrapper():
        if borrowed_middle:
            # Copied coordinator context is routing, never native admission.
            middle = asyncio.create_task(close_source())
            children["borrowed"] = middle
            return await await_owned_task(middle)
        return await close_source()

    async def writer():
        async with lock:
            async with lock:
                children["source"] = lock.start_owned_native_work(source)
                children["wrapper"] = create_owned_task(close_wrapper())
                children["sibling"] = lock.start_owned_native_work(sibling)
                started.set()
                result = await await_owned_task(children["wrapper"])
                restoration.append((result, lock._writer is asyncio.current_task(), lock._depth,
                    sibling_entered.is_set()))

    owner = create_owned_task(writer())
    ready = asyncio.create_task(entered.wait())
    try:
        await started.wait()
        completed, _ = await asyncio.wait({ready}, timeout=.05 if borrowed_middle else .3)
        admitted = bool(completed)
        excluded = not sibling_entered.is_set()
    finally:
        release.set()
        if borrowed_middle or not entered.is_set():
            children["source"].cancel()
        if not ready.done():
            ready.cancel()
        await await_owned_task(ready)
        settled = await await_owned_task(owner)
        for child in children.values():
            await await_owned_task(child)
    assert admitted is not borrowed_middle, "native close writer handoff must follow actual accepted waits, never borrowed observers"
    assert excluded, "native close wait lent its writer to an unrelated accepted sibling"
    result, restored, depth, sibling_before_restore = restoration[0]
    assert restored and depth == 2 and not sibling_before_restore
    assert settled.error is None and sibling_entered.is_set()
    if not borrowed_middle:
        assert result.error is None and result.result.error is None
        assert result.result.result == "original source"
    assert not lock.locked() and not lock._native_work and not lock._admissions


@pytest.mark.asyncio(loop_scope='function')
@pytest.mark.parametrize("receipt", ["unknown", "committed"])
@pytest.mark.parametrize('executor_workers', [None, 1])
async def test_bounded_observer_preserves_late_native_commit_on_multiple_joins(tmp_path, receipt, executor_workers):
    loop = asyncio.get_running_loop()
    if executor_workers is not None:
        loop.set_default_executor(ThreadPoolExecutor(max_workers=executor_workers))
    path = tmp_path / "native.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE effects(id INTEGER)")
    entered, release = asyncio.Event(), threading.Event()
    original = RuntimeError("native transaction acknowledgement")
    original.__cause__ = ExecutionCommitOutcomeError(receipt)
    def commit():
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(3):
            raise TimeoutError("fixture thread not released")
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("INSERT INTO effects VALUES (1)")
        raise original
    native = create_owned_task(run_blocking_operation(commit))
    observer = create_owned_task(wait_owned_or_pending(native, cancellation_grace=.03))
    try:
        assert await asyncio.wait_for(entered.wait(), 1)
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
